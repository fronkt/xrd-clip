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
