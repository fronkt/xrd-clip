"""Phase 0 (v2): within-family polymorph retrieval, model vs the STRONG classical baselines.

The URTC paper beat fixed-grid Pearson and a naive peak list under shift/strain. Neither can absorb
a shift by construction, so that win says little. This script reruns the within-family evaluation
with the baselines a powder-diffraction reviewer would demand (eval/strong_baselines.py):

  pearson       URTC baseline: fixed-grid Pearson                  (reference)
  peak_list     URTC baseline: F1 peak-list match                  (reference)
  pearson_zs    B1, l = 0: Pearson maximised over a 2theta zero scan
  wcc_0.5/1/2   B1: de Gelder WCC (triangle half-width l deg) + zero scan
  lnq           B2: zero scan + ln(q) lag scan (absorbs isotropic strain exactly)
  vc            B3: variable-cell Gaussian similarity (6 cell params + zero offset)
  model         the URTC checkpoint

Every query is rendered ONCE per condition and scored by every method (identical physics). Ranks are
saved per method so any later analysis (bootstrap CIs, hybrids) needs no re-rendering.

Conditions: the URTC one-dial sweeps; a NEW anisotropic thermal-expansion axis (per-axis linear
expansion coefficients drawn per crystal, mean ~55 ppm/K with ~8 % negative axes, after Bond 2021,
10.1107/S2052520621003309); the URTC field-average points; the same points with anisotropic thermal
strain replacing isotropic strain; and the URTC sign decomposition.

  PYTHONPATH=src python scripts/12_strong_baselines.py --max_families 40      # smoke
  PYTHONPATH=src python scripts/12_strong_baselines.py                        # full
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace

import numpy as np
import torch

from xrdclip.data import augment as A
from xrdclip.eval import baselines as B
from xrdclip.eval import strong_baselines as SB

FIELD = {
    "mild":     (dict(domain_nm=150.0, march=0.90, counts=1e5, bkg_scale=0.05, hump_scale=0.03), 0.05, 0.003),
    "moderate": (dict(domain_nm=70.0,  march=0.80, counts=2e4, bkg_scale=0.10, hump_scale=0.05), 0.10, 0.006),
    "harsh":    (dict(domain_nm=30.0,  march=0.70, counts=5e3, bkg_scale=0.15, hump_scale=0.08), 0.20, 0.010),
}
THERMAL_FOR_FIELD = {"mild": 100.0, "moderate": 150.0, "harsh": 200.0}   # dT in K


def thermal_axis_strain(cell, dT: float, rng) -> tuple:
    """Per-axis linear strain for a dT warm-up. alpha ~ N(55, 40) ppm/K per axis (mean volumetric
    ~165 ppm/K, Bond 2021 reports 161-173); axes of equal length share alpha so that tetragonal/
    hexagonal symmetry is preserved."""
    alpha = rng.normal(55e-6, 40e-6, 3)
    a, b, c = cell[:3]
    if abs(a - b) < 1e-3:
        alpha[1] = alpha[0]
    if abs(b - c) < 1e-3:
        alpha[2] = alpha[1]
    return tuple(float(x) for x in alpha * dT)


def conditions():
    """name -> fn(rec, rng) -> Params. One Params per query, so per-crystal draws are possible."""
    out = {"clean": lambda rec, rng: A.CLEAN}
    one_dial = {
        "zero_offset": [0.0, 0.05, 0.10, 0.15, 0.20, 0.30],
        "strain": [0.002, 0.005, 0.008, 0.012],
        "domain_nm": [100, 50, 20, 10],
        "march": [0.9, 0.8, 0.7, 0.5],
        "counts": [1e5, 1e4, 5e3, 2e3],
    }
    for axis, levels in one_dial.items():
        for lv in levels:
            p = A.sweep_params(**{axis: lv})
            out[f"{axis}={lv}"] = (lambda p: lambda rec, rng: p)(p)
    for dT in (50.0, 100.0, 150.0, 200.0):
        out[f"thermal_dT={dT:g}"] = (lambda dT: lambda rec, rng: A.sweep_params(
            axis_strain=thermal_axis_strain(rec["cell"], dT, rng)))(dT)
    for name, (base_kw, zb, sb) in FIELD.items():
        base = A.sweep_params(**base_kw)
        out[f"field_{name}"] = (lambda base, zb, sb: lambda rec, rng: replace(
            base, zero_offset=float(rng.uniform(-zb, zb)), strain=float(rng.uniform(-sb, sb))))(base, zb, sb)
        dT = THERMAL_FOR_FIELD[name]
        out[f"field_thermal_{name}"] = (lambda base, zb, dT: lambda rec, rng: replace(
            base, zero_offset=float(rng.uniform(-zb, zb)),
            axis_strain=thermal_axis_strain(rec["cell"], dT, rng)))(base, zb, dT)
    base_kw, zb, sb = FIELD["moderate"]
    base = A.sweep_params(**base_kw)
    out["sign_cancel"] = (lambda p: lambda rec, rng: p)(replace(base, zero_offset=zb, strain=sb))
    out["sign_add"] = (lambda p: lambda rec, rng: p)(replace(base, zero_offset=zb, strain=-sb))
    return out


def rank_within(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    return np.array([int(np.where(order[i] == i)[0][0]) for i in range(scores.shape[0])])


def bootstrap_top1(ranks: np.ndarray, fam_idx: np.ndarray, n_boot: int = 1000, seed: int = 0):
    """95 % CI of top-1 by resampling whole families (queries within a family are not independent)."""
    rng = np.random.default_rng(seed)
    fams = np.unique(fam_idx)
    hit = (ranks < 1).astype(float)
    per_fam_hits = np.array([hit[fam_idx == f].sum() for f in fams])
    per_fam_n = np.array([(fam_idx == f).sum() for f in fams])
    stats = []
    for _ in range(n_boot):
        s = rng.integers(0, len(fams), len(fams))
        stats.append(per_fam_hits[s].sum() / per_fam_n[s].sum())
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sticks", default="cache/sticks")
    ap.add_argument("--test_refs", default="refcodes/scaffold_test.txt")
    ap.add_argument("--families", default="refcodes/polymorph_families.json")
    ap.add_argument("--ckpt", default="runs/xrdclip_poly/best.pt")
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--out", default="results/strong_baselines.json")
    ap.add_argument("--ranks_out", default="results/strong_baselines_ranks.npz")
    ap.add_argument("--max_families", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated condition-name prefixes to run")
    ap.add_argument("--no_vc", action="store_true", help="skip B3 (the slow one)")
    args = ap.parse_args()

    from xrdclip.data.dataset import StickStore
    from xrdclip.data.polymorph import load_families, family_groups
    from xrdclip.models.clip import XRDClip
    from xrdclip.eval.retrieve import embed_library

    torch.set_num_threads(os.cpu_count() or 4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.test_refs) as fh:
        test_refs = [ln.strip() for ln in fh if ln.strip()]
    store = StickStore(args.sticks, refcodes=set(test_refs))
    present = [rc for rc in test_refs if rc in store.records]
    groups = family_groups(present, load_families(args.families))
    if args.max_families:
        groups = dict(list(groups.items())[: args.max_families])
    stems = list(groups)
    fam_idx = np.array([k for k, s in enumerate(stems) for _ in groups[s]])
    sizes = np.array([len(groups[s]) for s in stems for _ in groups[s]])
    print(f"[strong] {len(stems)} families / {len(fam_idx)} queries; chance {np.mean(1 / sizes):.3f}")

    model = XRDClip(embed_dim=args.embed_dim).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    model.eval()

    # Clean references: rendered once, shared by every condition.
    recs = {s: [store[rc] for rc in groups[s]] for s in stems}
    ref_raw = {s: np.stack([A.render_sticks(r, A.CLEAN) for r in recs[s]]) for s in stems}
    ref_std = {s: np.stack([A.standardize(y) for y in ref_raw[s]]) for s in stems}
    ref_pk = {s: [B.extract_peaks(y) for y in ref_raw[s]] for s in stems}
    with torch.no_grad():
        lib = {s: embed_library(model, store, groups[s], device) for s in stems}
    flat_recs = [r for s in stems for r in recs[s]]
    offsets = np.cumsum([0] + [len(groups[s]) for s in stems])

    conds = conditions()
    if args.only:
        pref = tuple(args.only.split(","))
        conds = {k: v for k, v in conds.items() if k.startswith(pref)}

    methods = ["model", "pearson", "peak_list", "pearson_zs", "wcc_0.5", "wcc_1", "wcc_2", "lnq"]
    if not args.no_vc:
        methods.append("vc")
    results = {"n_families": len(stems), "n_queries": int(len(fam_idx)),
               "chance_top1": float(np.mean(1 / sizes)), "conditions": {}}
    if os.path.exists(args.out) and args.only:
        with open(args.out) as fh:
            prev = json.load(fh)
        if prev.get("n_queries") == results["n_queries"]:
            results["conditions"] = prev["conditions"]
    all_ranks = dict(np.load(args.ranks_out)) if (args.only and os.path.exists(args.ranks_out)) else {}

    for cname, pfn in conds.items():
        t0 = time.time()
        rng = np.random.default_rng(abs(hash(cname)) % (2**32))
        q_raw = {s: np.stack([A.render_sticks(r, pfn(r, rng), rng=rng) for r in recs[s]]) for s in stems}
        ranks = {m: [] for m in methods}
        secs = {m: 0.0 for m in methods}

        def timed(m, fn):
            t = time.perf_counter(); v = fn(); secs[m] += time.perf_counter() - t; return v

        for s in stems:
            q, qs = q_raw[s], np.stack([A.standardize(y) for y in q_raw[s]])
            with torch.no_grad():
                emb = timed("model", lambda: model.encode_pattern(
                    torch.from_numpy(qs).unsqueeze(1).to(device)).cpu().numpy())
            ranks["model"].append(rank_within(emb @ lib[s].T))
            ranks["pearson"].append(rank_within(timed("pearson", lambda: B.correlation_search_match(qs, ref_std[s]))))
            ranks["peak_list"].append(rank_within(timed("peak_list", lambda: B.peak_list_match(
                [B.extract_peaks(y) for y in q], ref_pk[s]))))
            ranks["pearson_zs"].append(rank_within(timed("pearson_zs", lambda: SB.wcc_scores(q, ref_raw[s], l_deg=0.0))))
            for l in (0.5, 1, 2):
                ranks[f"wcc_{l}"].append(rank_within(timed(f"wcc_{l}", lambda: SB.wcc_scores(q, ref_raw[s], l_deg=l))))
            ranks["lnq"].append(rank_within(timed("lnq", lambda: SB.lnq_wcc_scores(q, ref_raw[s]))))

        if "vc" in methods:   # batched across all families: one big optimisation per condition
            t = time.perf_counter()
            qpk = [SB.query_peaks(y) for s in stems for y in q_raw[s]]
            pairs = [(offsets[k] + i, offsets[k] + j) for k, s in enumerate(stems)
                     for i in range(len(groups[s])) for j in range(len(groups[s]))]
            # Strain prior sized to thermal expansion; without it a sibling's cell is bent onto the
            # query (see vc_gpwdf_scores docstring). Tuned on 40 families only: lam 0.003 was best
            # on thermal strain (0.72 vs 0.52 unregularised). Needs critic2 validation.
            sc = SB.vc_gpwdf_scores(qpk, flat_recs, pairs, sigmas=(0.4, 0.2, 0.1),
                                    prior=(0.003, 0.01, 0.5))
            secs["vc"] = time.perf_counter() - t
            p = 0
            for k, s in enumerate(stems):
                n = len(groups[s])
                ranks["vc"].append(rank_within(sc[p: p + n * n].reshape(n, n)))
                p += n * n

        n_pairs = int(sum(len(groups[s]) ** 2 for s in stems))
        res = {}
        for m in methods:
            r = np.concatenate(ranks[m])
            all_ranks[f"{cname}|{m}"] = r
            lo, hi = bootstrap_top1(r, fam_idx)
            res[m] = {"top1": float((r < 1).mean()), "ci95": [lo, hi],
                      "mrr": float((1 / (r + 1)).mean()), "us_per_pair": 1e6 * secs[m] / n_pairs}
        results["conditions"][cname] = res
        line = "  ".join(f"{m}={res[m]['top1']:.3f}" for m in methods)
        print(f"[strong] {cname:24s} {line}  ({time.time() - t0:.0f}s)", flush=True)

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        np.savez_compressed(args.ranks_out, fam_idx=fam_idx, **all_ranks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
