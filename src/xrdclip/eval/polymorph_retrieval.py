"""Within-family (polymorph) retrieval evaluation — the pivot's headline experiment.

The question XCCP and XtalNet cannot ask: given the (possibly degraded) pattern of ONE polymorph,
can the model pick that polymorph out of a library made only of its same-molecule siblings? Formula
is identical and space group is often identical, so composition/symmetry give NO signal — only the
packing -> pattern mapping does. Chance is 1/family_size, and a composition-only method is pinned to
chance by construction. That contrast is the paper.

Two entry points, mirroring eval/retrieve.py:
  within_family_retrieval   clean (or single-`params`) within-family top-k, model only
  within_family_sweep       degradation sweep of within-family top-1: model vs both classical baselines
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from xrdclip.data import augment as A
from xrdclip.eval import baselines as B
from xrdclip.eval.retrieve import embed_library, embed_queries


def _rank_within(scores: np.ndarray) -> np.ndarray:
    """Ranks of the diagonal (each query's own structure) within a square family score matrix."""
    order = np.argsort(-scores, axis=1)
    n = scores.shape[0]
    return np.array([int(np.where(order[i] == i)[0][0]) for i in range(n)])


def _summary(ranks: np.ndarray, sizes: np.ndarray, ks=(1, 2, 5)) -> dict:
    out = {f"top{k}": float((ranks < k).mean()) for k in ks}
    out["mrr"] = float((1.0 / (ranks + 1)).mean())
    out["chance_top1"] = float((1.0 / sizes).mean())     # random guess within each family
    out["n_query"] = int(len(ranks))
    return out


@torch.no_grad()
def within_family_retrieval(model, store, groups, params, device, rng, ks=(1, 2, 5)) -> dict:
    """Model-only within-family top-k under a single degradation `params` (default clean)."""
    ranks, sizes = [], []
    for rcs in groups.values():
        lib = embed_library(model, store, rcs, device)
        q = embed_queries(model, store, rcs, params, device, rng)
        r = _rank_within(q @ lib.T)
        ranks.extend(r.tolist())
        sizes.extend([len(rcs)] * len(rcs))
    out = _summary(np.array(ranks), np.array(sizes), ks)
    out["n_families"] = len(groups)
    return out


def _baseline_within(groups, store, params, rng, kind: str) -> "tuple[np.ndarray, np.ndarray]":
    """Within-family ranks for a classical baseline. kind in {'correlation','peak_list'}."""
    ranks, sizes = [], []
    for rcs in groups.values():
        recs = [store[rc] for rc in rcs]
        if kind == "correlation":
            refs = B.build_reference_patterns(recs, A.CLEAN)
            q = np.stack([A.standardize(A.render_sticks(r, params, rng=rng)) for r in recs])
            scores = B.correlation_search_match(q, refs)
        else:
            ref_pk = [B.extract_peaks(A.render_sticks(r, A.CLEAN)) for r in recs]
            q_pk = [B.extract_peaks(A.render_sticks(r, params, rng=rng)) for r in recs]
            scores = B.peak_list_match(q_pk, ref_pk)
        ranks.extend(_rank_within(scores).tolist())
        sizes.extend([len(rcs)] * len(rcs))
    return np.array(ranks), np.array(sizes)


def _params_brief(p: A.Params) -> dict:
    """The dials that are turned on at an operating point, for the results record."""
    return {"zero_offset": p.zero_offset, "strain": p.strain, "domain_nm": p.domain_nm,
            "march": p.march, "counts": p.counts, "bkg_scale": p.bkg_scale, "hump_scale": p.hump_scale}


@torch.no_grad()
def _family_field_ranks(model, lib, recs, ref_pat, ref_pk, params_list, device, rng):
    """Rank each family member's own query, under its OWN per-crystal params, for all three methods.

    Every method scores the SAME rendered query pattern, so the comparison is over identical physics:
      * model        z-scored pattern -> encode_pattern -> cross-modal dot with clean structure lib
      * correlation  z-scored pattern -> Pearson vs clean rendered reference patterns
      * peak-list    max-normed pattern -> F1 peak match vs clean reference peak lists
    Returns (model_ranks, corr_ranks, peak_ranks), each length len(recs).
    """
    raw = [A.render_sticks(recs[i], params_list[i], rng=rng) for i in range(len(recs))]
    q_std = np.stack([A.standardize(y) for y in raw])
    q_pk = [B.extract_peaks(y) for y in raw]

    x = torch.from_numpy(q_std).unsqueeze(1).to(device)
    q_emb = model.encode_pattern(x).cpu().numpy()

    m_ranks = _rank_within(q_emb @ lib.T)
    c_ranks = _rank_within(B.correlation_search_match(q_std, ref_pat))
    k_ranks = _rank_within(B.peak_list_match(q_pk, ref_pk))
    return m_ranks, c_ranks, k_ranks


@torch.no_grad()
def within_family_field_average(model, store, groups, points, device, seed=0) -> dict:
    """Field-average within-family top-1: model vs correlation vs peak-list vs chance.

    within_family_sweep turns one dial at a time; that answers "which physical effect does the
    embedding survive?" but not "does it win in the field, where every effect is present at once?".
    The naive answer — one Params with several dials on together — is a TRAP: a fixed same-sign
    sample-displacement and thermal strain partially CANCEL in the mid-angle range (displacement
    pushes 2theta up, cell expansion pushes it down), which artificially rescues full-profile
    correlation; flip one sign and correlation collapses again. A single deterministic operating
    point therefore reports whatever the experimenter's sign choice happens to be worth.

    So each crystal here gets its OWN sample-displacement and thermal strain, drawn independently
    (sign and magnitude) from a realistic band, on top of a fixed base of broadening / texture /
    noise / background. Over the test set both cancellation and addition occur at their natural
    rates, so the number is the honest expectation over real samples. The model, trained to be
    shift-invariant, is flat regardless; correlation pays the average cost of a shift it cannot
    count on cancelling. Each `point` is {"base": Params, "zo_band": deg, "strain_band": frac}.
    """
    rng = np.random.default_rng(seed)
    stems = list(groups.keys())
    lib_by_stem = {s: embed_library(model, store, groups[s], device) for s in stems}
    recs_by_stem = {s: [store[rc] for rc in groups[s]] for s in stems}
    refpat_by_stem = {s: np.stack([A.standardize(A.render_sticks(r, A.CLEAN)) for r in recs_by_stem[s]])
                      for s in stems}
    refpk_by_stem = {s: [B.extract_peaks(A.render_sticks(r, A.CLEAN)) for r in recs_by_stem[s]]
                     for s in stems}
    sizes_all = np.array([len(groups[s]) for s in stems for _ in groups[s]])

    out: dict = {"chance_top1": float((1.0 / sizes_all).mean()), "points": {}}
    for name, spec in points.items():
        base, zb, sb = spec["base"], spec["zo_band"], spec["strain_band"]
        m_all, c_all, k_all = [], [], []
        for s in stems:
            recs = recs_by_stem[s]
            pl = [replace(base, zero_offset=float(rng.uniform(-zb, zb)),
                          strain=float(rng.uniform(-sb, sb))) for _ in recs]
            mr, cr, kr = _family_field_ranks(model, lib_by_stem[s], recs,
                                             refpat_by_stem[s], refpk_by_stem[s], pl, device, rng)
            m_all.extend(mr.tolist()); c_all.extend(cr.tolist()); k_all.extend(kr.tolist())
        m_all, c_all, k_all = np.array(m_all), np.array(c_all), np.array(k_all)
        out["points"][name] = {
            "base": _params_brief(base), "zo_band": zb, "strain_band": sb,
            "model": {"top1": float((m_all < 1).mean()), "top2": float((m_all < 2).mean()),
                      "mrr": float((1.0 / (m_all + 1)).mean())},
            "correlation": {"top1": float((c_all < 1).mean())},
            "peak_list": {"top1": float((k_all < 1).mean())},
        }
    return out


@torch.no_grad()
def within_family_sign_decomposition(model, store, groups, base, zo, strain, device, seed=0) -> dict:
    """The mechanism behind the field average: hold |displacement| and |strain| fixed and vary only
    their RELATIVE SIGN. `add` = the two shifts reinforce; `cancel` = they oppose (mid-angle
    cancellation). The model is flat across both (genuine invariance); correlation swings wide,
    exposing its robustness as a coincidental-cancellation artifact rather than a real property."""
    combos = {
        "shifts_cancel": replace(base, zero_offset=+abs(zo), strain=+abs(strain)),   # up-shift + expansion(down) oppose
        "shifts_add":    replace(base, zero_offset=+abs(zo), strain=-abs(strain)),    # up-shift + contraction(up) reinforce
    }
    rng = np.random.default_rng(seed)
    stems = list(groups.keys())
    lib_by_stem = {s: embed_library(model, store, groups[s], device) for s in stems}
    sizes_all = np.array([len(groups[s]) for s in stems for _ in groups[s]])

    out: dict = {"chance_top1": float((1.0 / sizes_all).mean()),
                 "zo": abs(zo), "strain": abs(strain), "base": _params_brief(base), "combos": {}}
    for name, p in combos.items():
        m_ranks = []
        for s in stems:
            q = embed_queries(model, store, groups[s], p, device, rng)
            m_ranks.extend(_rank_within(q @ lib_by_stem[s].T).tolist())
        m_ranks = np.array(m_ranks)
        c_ranks, _ = _baseline_within(groups, store, p, rng, "correlation")
        k_ranks, _ = _baseline_within(groups, store, p, rng, "peak_list")
        out["combos"][name] = {
            "model": {"top1": float((m_ranks < 1).mean())},
            "correlation": {"top1": float((c_ranks < 1).mean())},
            "peak_list": {"top1": float((k_ranks < 1).mean())},
        }
    return out


@torch.no_grad()
def within_family_sweep(model, store, groups, device, seed=0) -> dict:
    """Within-family top-1 vs degradation severity: model vs correlation vs peak-list.

    Structure-side (library) embeddings are clean and query-independent, so they are computed ONCE
    and reused across every axis/level; only the query patterns are re-degraded. Same axes as the
    full-library robustness sweep (eval/retrieve.robustness_sweep) so the two figures share x-axes.
    """
    rng = np.random.default_rng(seed)
    stems = list(groups.keys())

    # one clean library embedding per family, reused across all levels
    lib_by_stem = {s: embed_library(model, store, groups[s], device) for s in stems}
    sizes_all = np.array([len(groups[s]) for s in stems for _ in groups[s]])

    axes = {
        "zero_offset": [0.0, 0.05, 0.10, 0.15, 0.20, 0.30],
        "strain": [0.0, 0.002, 0.005, 0.008, 0.012],
        "domain_nm": [1e4, 100, 50, 20, 10],
        "march": [1.0, 0.9, 0.8, 0.7, 0.5],
        "counts": [0, 1e5, 1e4, 5e3, 2e3],
    }

    out: dict = {"chance_top1": float((1.0 / sizes_all).mean()), "axes": {}}
    for axis, levels in axes.items():
        out["axes"][axis] = {}
        for lv in levels:
            p = A.sweep_params(**{axis: lv})
            m_ranks = []
            for s in stems:
                q = embed_queries(model, store, groups[s], p, device, rng)
                m_ranks.extend(_rank_within(q @ lib_by_stem[s].T).tolist())
            m_ranks = np.array(m_ranks)
            c_ranks, _ = _baseline_within(groups, store, p, rng, "correlation")
            k_ranks, _ = _baseline_within(groups, store, p, rng, "peak_list")
            out["axes"][axis][str(lv)] = {
                "model": {"top1": float((m_ranks < 1).mean()), "mrr": float((1.0 / (m_ranks + 1)).mean())},
                "correlation": {"top1": float((c_ranks < 1).mean())},
                "peak_list": {"top1": float((k_ranks < 1).mean())},
            }
    return out
