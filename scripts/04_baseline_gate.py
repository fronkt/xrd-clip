"""WEEK-0 GATE (SPEC §6): do the classical baselines behave as the physics predicts?

We need two things to be true before the project is worth continuing:

  1. On CLEAN simulated patterns, search-match is near-perfect. (If it isn't, our eval harness is
     broken -- clean retrieval against an identical rendering is supposed to be trivial.)

  2. Under realistic degradation, search-match DEGRADES. That degradation is the headroom
     XRD-CLIP exists to recover. If the baselines shrug off peak shift and texture, there is no
     gap to close, and we should reroute (SPEC §8) rather than train a model to beat nothing.

Prediction on record, before running:
  * correlation collapses hard under shift (strain / zero-offset), because it compares bin-by-bin
    and a shifted peak lands in the wrong bin.
  * peak-list matching is far more shift-tolerant (its tolerance window absorbs offsets) but
    should suffer under preferred orientation and noise, which corrupt which peaks are strongest.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from xrdclip.data import augment as A
from xrdclip.data.dataset import StickStore
from xrdclip.eval import baselines as B


def evaluate(store: StickStore, refcodes, params: A.Params, label: str,
             refs_z: np.ndarray, ref_peaks, rng) -> dict:
    """Render degraded queries under `params`; score with both baselines."""
    n = len(refcodes)
    q_z = np.zeros((n, A.N_POINTS), dtype=np.float32)
    q_peaks = []
    for i, rc in enumerate(refcodes):
        y = A.render_sticks(store[rc], params, rng=rng)
        q_z[i] = A.standardize(y)
        q_peaks.append(B.extract_peaks(y))

    truth = np.arange(n)
    corr = B.retrieval_metrics(B.correlation_search_match(q_z, refs_z), truth)
    peak = B.retrieval_metrics(B.peak_list_match(q_peaks, ref_peaks), truth)

    print(f"  {label:<26} corr top1={corr['top1']:.3f} top5={corr['top5']:.3f}  |  "
          f"peaks top1={peak['top1']:.3f} top5={peak['top5']:.3f}")
    return {"condition": label, "correlation": corr, "peak_list": peak}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sticks", default="cache/pilot_sticks")
    ap.add_argument("--n", type=int, default=1000, help="library size")
    ap.add_argument("--out", default="results/baseline_gate.json")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    store = StickStore(args.sticks)
    raw = store.refcodes[: args.n]

    # Collapse PXRD-indistinguishable entries (CSD redeterminations). Without this, clean
    # self-retrieval cannot reach 1.0 and every downstream number carries a spurious ceiling.
    from xrdclip.data.dedup import dedup_report
    rep = dedup_report(store, raw)
    refcodes = rep["representatives"]
    print(f"[gate] {rep['n_entries']} entries -> {rep['n_classes']} PXRD-distinct structures "
          f"({rep['n_removed']} degenerate duplicates removed, "
          f"largest class = {rep['largest_class']})")
    print(f"[gate] library size = {len(refcodes)} structures\n")

    # Reference library: rendered CLEAN, once.
    refs_z = B.build_reference_patterns([store[rc] for rc in refcodes], A.CLEAN)
    ref_peaks = [B.extract_peaks(A.render_sticks(store[rc], A.CLEAN)) for rc in refcodes]

    results = []

    print("[gate] sanity: clean query == clean reference (must be ~1.000)")
    results.append(evaluate(store, refcodes, A.CLEAN, "clean", refs_z, ref_peaks, rng))

    print("\n[gate] zero-offset shift (deg 2theta)")
    for off in (0.05, 0.10, 0.20, 0.30):
        results.append(evaluate(store, refcodes, A.sweep_params(zero_offset=off),
                                f"zero_offset={off}", refs_z, ref_peaks, rng))

    print("\n[gate] isotropic lattice strain (theta-dependent shift)")
    for eps in (0.002, 0.005, 0.010):
        results.append(evaluate(store, refcodes, A.sweep_params(strain=eps),
                                f"strain={eps}", refs_z, ref_peaks, rng))

    print("\n[gate] crystallite-size broadening")
    for d in (200.0, 50.0, 20.0, 10.0):
        results.append(evaluate(store, refcodes, A.sweep_params(domain_nm=d),
                                f"domain={d}nm", refs_z, ref_peaks, rng))

    print("\n[gate] preferred orientation (March-Dollase r)")
    for r in (0.9, 0.7, 0.5):
        results.append(evaluate(store, refcodes, A.sweep_params(march=r, po_axis=(0, 0, 1)),
                                f"march={r}", refs_z, ref_peaks, rng))

    print("\n[gate] counting noise")
    for c in (1e5, 1e4, 2e3):
        results.append(evaluate(store, refcodes, A.sweep_params(counts=c),
                                f"counts={c:.0e}", refs_z, ref_peaks, rng))

    print("\n[gate] FULL random augmentation (what the model actually trains against)")
    for trial in range(3):
        p = A.sample_params(rng)
        results.append(evaluate(store, refcodes, p, f"random_aug_{trial}", refs_z, ref_peaks, rng))

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"n_ref": len(refcodes), "results": results}, fh, indent=2)
    print(f"\n[gate] wrote {args.out}")

    # --- the verdict ---
    clean = results[0]["correlation"]["top1"]
    shifted = [r for r in results if r["condition"].startswith("zero_offset=0.2")][0]["correlation"]["top1"]
    print("\n" + "=" * 72)
    print(f"CLEAN correlation top-1      : {clean:.3f}   (expect ~1.00; else the harness is broken)")
    print(f"SHIFTED (0.2 deg) top-1      : {shifted:.3f}   (expect << clean; this is the headroom)")
    if clean > 0.95 and shifted < 0.7:
        print("VERDICT: GATE PASSED — baselines are sane and degradation opens real headroom.")
    elif clean <= 0.95:
        print("VERDICT: FAIL — clean retrieval should be trivial. Harness bug. Do not proceed.")
    else:
        print("VERDICT: NO HEADROOM — baselines resist degradation. Reconsider the claim (SPEC §8).")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
