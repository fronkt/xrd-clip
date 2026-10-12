"""Case study for the paper (plan Section 12b, finding 1): why every zero-scan method misses ZZZUWK.

  PYTHONPATH=src python scripts/24_case_zzzuwk.py      -> results/case_zzzuwk.json (+ cache/real_bench/case_zzzuwk.npz)

ZZZUWK has 20 of the 188 benchmark patterns. Fixed-grid Pearson ranks its own entry first for 12 of them; cos + zero
scan, WCC and ln q for none. The hypothesis: the +/-0.5 deg zero scan lets decoys that share ZZZUWK's two strong
low-angle reflections line up with them, while the fixed grid cannot move them. Test: for every ZZZUWK pattern, score
its same-form entries and every decoy that wins it (any method, uncorrected or CSLS) with fixed-grid Pearson and with
cos / WCC l = 1 deg at EACH zero offset, and record the offset at which each entry scores best.
The .npz (gitignored: simulated CSD patterns) holds the curves for the paper figure.
"""
from __future__ import annotations

import json
import sys
from importlib import import_module

import numpy as np
import torch

sys.path.insert(0, "scripts")
bench = import_module("19_real_bench")
hubm = import_module("21_hubness")

from xrdclip.data import augment as A           # noqa: E402
from xrdclip.eval.wcc_torch import TorchWCC, full_precision   # noqa: E402

CASE = "ZZZUWK"


def per_offset(spec: dict, q_raw: np.ndarray, r_raw: np.ndarray) -> "tuple[np.ndarray, float]":
    """Scores at every zero offset: (n_offsets, n_q, n_r), and the offset step in degrees."""
    tw = TorchWCC(**spec)
    (q, aq), = tw.prep_rows(q_raw)
    rw, ar = tw.prep_refs(r_raw)
    n = q.shape[1]
    den = torch.sqrt(torch.clamp(aq[:, None] * ar[None, :], min=1e-12))
    s = torch.stack([(q @ rw[:, d: d + n].T) / den for d in range(2 * tw.s_pts + 1)])
    return s.numpy(), tw.s_pts


def main() -> int:
    full_precision()
    hub = json.load(open("results/real_bench_hub.json"))
    shards = bench.shard_table(None)
    lib_rc = hubm.library_refcodes(shards)
    gidx = {rc: i for i, rc in enumerate(lib_rc)}
    qinfo, _, Q = bench.build_queries(f"{bench.DATA}/opxrd/opxrd_truth.json",
                                      f"{bench.DATA}/opxrd/opxrd_cnrs_organic.json", gidx)
    rows = [i for i, q in enumerate(qinfo) if q["stem"] == CASE]
    truth = sorted({rc for i in rows for rc in qinfo[i]["sets"]["std"][0]})
    decoys = {}                                     # refcode -> methods it wins ZZZUWK patterns under
    for key, h in hub["hubness"].items():
        for corr in ("none", "csls_aug_k10"):
            for w, n, _, top in h[f"query_top1_wrong|{corr}"]["most_common"]:
                if any(s == CASE for s, _ in top):
                    decoys.setdefault(w, []).append(f"{key}|{corr} x{dict(top)[CASE]}")
    ents = truth + sorted(decoys)
    raw = bench.render(hubm.records_at(shards, [gidx[rc] for rc in ents], bench.RENDER_FIELDS))
    qr, qs = Q["raw_bkg"][rows], Q["std_bkg"][rows]

    pear = qs @ np.stack([A.standardize(y) for y in raw]).T / raw.shape[1]          # (20, E)
    res = {"stem": CASE, "n_patterns": len(rows), "truth": truth, "decoys": decoys, "methods": {}}
    curves = {}
    nt = len(truth)
    for name, spec in {"cos_zs": dict(l_deg=0.0), "wcc_1": dict(l_deg=1.0)}.items():
        s, S = per_offset(spec, qr, raw)
        best, arg = s.max(0), s.argmax(0)                                         # (20, E)
        off = (arg - S) * A.STEP
        tbest = best[:, :nt].max(1)                                               # best same-form entry per pattern
        t0 = s[S][:, :nt].max(1)                                                  # ... at zero offset only
        per = {}
        for j, rc in enumerate(ents):
            per[rc] = {"role": "same_form" if j < nt else "decoy",
                       "pearson_mean": float(pear[:, j].mean()),
                       "score_scan_mean": float(best[:, j].mean()), "score_offset0_mean": float(s[S][:, j].mean()),
                       "best_offset_deg_median": float(np.median(off[:, j])),
                       "best_offset_deg_range": [float(off[:, j].min()), float(off[:, j].max())]}
            if j >= nt:
                per[rc]["beats_truth_with_scan"] = int((best[:, j] > tbest).sum())
                per[rc]["beats_truth_at_offset0"] = int((s[S][:, j] > t0).sum())
                per[rc]["beats_truth_pearson"] = int((pear[:, j] > pear[:, :nt].max(1)).sum())
        res["methods"][name] = per
        curves[f"{name}_offsets"] = off
    # context: how well do ZZZUWK's patterns match their own entries, relative to the other benchmark patterns?
    ctx = {}
    for key in ("pearson_bkg", "cos_zs_bkg", "wcc_1_bkg"):
        S = np.load(f"cache/real_bench/full/S_{key}.npy", mmap_mode="r")
        tmax = np.array([np.asarray(S[i])[[gidx[rc] for rc in q["sets"]["std"][0]]].max()
                         for i, q in enumerate(qinfo)])
        other = np.array([i for i in range(len(qinfo)) if i not in rows])
        ctx[key] = {"own_score_median_case": float(np.median(tmax[rows])),
                    "own_score_median_rest": float(np.median(tmax[other])),
                    "case_rank_percentile": float(np.mean([(tmax[other] < v).mean() for v in tmax[rows]]))}
    n_refl = lambda rec: int(((rec["two_theta"] >= A.GRID[0]) & (rec["two_theta"] <= A.GRID[-1])).sum())
    own = sorted({rc for q in qinfo for rc in q["sets"]["std"][0]})
    own_refl = np.array([n_refl(r) for r in hubm.records_at(shards, [gidx[rc] for rc in own], ("two_theta",))])
    case_n = n_refl(hubm.records_at(shards, [gidx[truth[0]]], ("two_theta",))[0])
    dec = sorted(decoys)
    ctx["reflections_5_50"] = {"case_entry": case_n, "benchmark_entries_median": float(np.median(own_refl)),
                               "benchmark_entries_frac_fewer": float((own_refl < case_n).mean()),
                               "decoys": {rc: n_refl(r) for rc, r in zip(dec, hubm.records_at(
                                   shards, [gidx[rc] for rc in dec], ("two_theta",)))}}
    # texture check: relative heights of the low-index reflections (P4/nmm, a ~ 6.25, c ~ 7.39 A), measured vs the
    # pattern's own Rietveld entry. Plate-like crystallites lying on c would raise (00l) and lower (hk0).
    tr = {t["id"]: t for t in json.load(open(f"{bench.DATA}/opxrd/opxrd_truth.json"))}
    refl = {"001": 12.0, "101": 18.6, "110": 20.1, "111": 23.5, "002": 24.1, "200": 28.6}
    own_of = [next(x["refcode"] for x in tr[qinfo[i]["id"]]["same_form"]
                   if x["is_anchor"] and x["is_powder_study"] and x["rmsd"] < 0.05) for i in rows]
    own_raw = bench.render(hubm.records_at(shards, [gidx[rc] for rc in own_of], bench.RENDER_FIELDS))
    hgt = lambda y, t: float(y[np.abs(A.GRID - t) <= 0.3].max() / y.max())
    ctx["texture"] = {hkl: {"measured_mean": float(np.mean([hgt(qr[j], t) for j in range(len(rows))])),
                            "own_entry_mean": float(np.mean([hgt(own_raw[j], t) for j in range(len(rows))]))}
                      for hkl, t in refl.items()}
    res["context"] = ctx
    print(json.dumps(ctx, indent=1))
    with open("results/case_zzzuwk.json", "w") as fh:
        json.dump(res, fh, indent=1)
    np.savez_compressed("cache/real_bench/case_zzzuwk.npz", entries=np.array(ents), raw=raw, q_raw=qr,
                        q_ids=np.array([qinfo[i]["id"] for i in rows]),
                        pearson=pear, grid=A.GRID, **curves)
    for name, per in res["methods"].items():
        print(f"== {name}")
        for rc, v in per.items():
            extra = (f"  beats truth: scan {v['beats_truth_with_scan']}/20, offset0 {v['beats_truth_at_offset0']}/20, "
                     f"pearson {v['beats_truth_pearson']}/20") if v["role"] == "decoy" else ""
            print(f"  {rc:9s} {v['role']:9s} pear {v['pearson_mean']:.3f}  scan {v['score_scan_mean']:.3f} "
                  f"off0 {v['score_offset0_mean']:.3f}  offset {v['best_offset_deg_median']:+.2f} "
                  f"{v['best_offset_deg_range']}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
