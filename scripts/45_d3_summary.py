"""Resubmission D3 (tasks/todo.md Section D, E4): the retrained checkpoints next to the submitted models.

Reads only result files (no scoring):
  results/rescore_main.json, results/d3/rescore_main_<m>.json     real benchmark, 28 eval --tag main (corrected labels)
  results/d3/model_gap_<m>.json                                    effect ladder + pattern-to-pattern (30 eval)
  results/sbv2_w*.json, results/d3/sbv3_<run>_<group>.json         synthetic polymorph sweep (12)
and writes results/d3/summary.md. Models: poly (submitted), realism (R4c retrain in the submission), d3_fix
(K-alpha2 + March-Dollase fixes), d3_sym_s0 / d3_sym_s1 (+ space-group-aware encoder, two seeds).

  PYTHONPATH=src python scripts/45_d3_summary.py
"""
from __future__ import annotations

import glob
import json
import os


MODELS = ("poly", "realism", "d3_fix", "d3_sym_s0", "d3_sym_s1")
RUNGS = ("own_clean", "own_ka2", "own_dw", "own_all", "own_all_dw", "own_all_dw_int30", "measured")
SWEEP_SHOW = ("clean", "zero_offset=0.1", "strain=0.008", "domain_nm=20", "counts=2000.0", "march=0.5",
              "march_bfdh=0.5", "march_bfdh=0.8", "march_bfdh=1.25", "march_bfdh=2", "thermal_dT=100",
              "field_moderate", "field_thermal_moderate", "field_harsh")
CLASSICAL = ("pearson_zs", "wcc_1", "lnq")


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def ci(c):
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


def rescore(m):
    return load("results/rescore_main.json" if m == "poly" else f"results/d3/rescore_main_{m}.json")


def sweep(m):
    """condition -> method -> top1 (merged over the group files)."""
    files = sorted(glob.glob("results/sbv2_w*.json")) if m == "poly" else sorted(glob.glob(f"results/d3/sbv3_{m}_*.json"))
    out = {}
    for f in files:
        if f.endswith("_ranks.npz"):
            continue
        out.update(json.load(open(f))["conditions"])
    return out


def main() -> int:
    L = ["# D3 retrains vs the submitted models", "",
         "Corrected labels throughout (refcodes/truth_corrections.json). Compound-weighted (macro) top-1 with 95 % CI;",
         "std = all identified patterns, sc = single-crystal-reference set. d3 val = degraded validation at the best",
         "epoch (the submitted models selected on clean validation, so no val column for them).", ""]

    L += ["## Real benchmark (28 eval --tag main, model rows)", "",
          "| model | d3 val | std top-1 | std median rank | sc top-1 | paired std diff vs poly | model@1000 -> ln q std |",
          "|---|---|---|---|---|---|---|"]
    for m in MODELS:
        r = rescore(m)
        hist = f"runs/{m}/history.json"
        val = f"{max(h['val_top1'] for h in json.load(open(hist))):.3f}" if m.startswith("d3") and os.path.exists(hist) else "-"
        if r is None:
            L.append(f"| {m} | {val} | - | - | - | - | - |")
            continue
        s, c = r["macro"]["std|model_bkg"]["top1"], r["macro"]["sc|model_bkg"]["top1"]
        med = r["per_pattern"]["std|model_bkg"]["median_rank"]
        pb = r["paired_vs_base"].get("std|model_bkg", {}).get("top1") if m != "poly" else None
        hy = r["macro"].get("std|model@1000->lnq_bkg", {}).get("top1")
        L.append(f"| {m} | {val} | {s['value']:.3f} {ci(s['ci'])} | {med:,.0f} | {c['value']:.3f} {ci(c['ci'])} | "
                 + (f"{pb['diff']:+.3f} {ci(pb['ci'])}" if pb else "-") + " | "
                 + (f"{hy['value']:.3f} {ci(hy['ci'])}" if hy else "-") + " |")
    r0 = rescore("poly")
    if r0:
        L += ["", "Classical reference (same file): " + ", ".join(
            f"{k.split('|')[1]} std {r0['macro'][k]['top1']['value']:.3f}" for k in r0["macro"]
            if k.startswith("std|") and "model" not in k)]

    L += ["", "## Effect ladder (30 eval; model, std; macro top-1 / median rank)", "",
          "| model | " + " | ".join(RUNGS) + " | p2p measured |", "|---|" + "---|" * (len(RUNGS) + 1)]
    for m in MODELS:
        g = load(f"results/d3/model_gap_{m}.json")
        if g is None:
            L.append(f"| {m} | " + " | ".join("-" for _ in RUNGS) + " | - |")
            continue
        g = g["results"]
        cell = lambda k: (f"{g[k]['macro_top1']['value']:.3f} / {g[k]['per_pattern']['median_rank']:,.0f}" if k in g else "-")
        L.append(f"| {m} | " + " | ".join(cell(f"{r}|model|std") for r in RUNGS) + f" | {cell('measured|p2p|std')} |")

    L += ["", "## Synthetic polymorph sweep (12; within-family top-1, model)", "",
          "| condition | " + " | ".join(MODELS) + " | " + " | ".join(CLASSICAL) + " |",
          "|---|" + "---|" * (len(MODELS) + len(CLASSICAL))]
    S = {m: sweep(m) for m in MODELS}
    ref = next((S[m] for m in MODELS[2:] if S[m]), S["poly"])
    for cname in SWEEP_SHOW:
        row = [f"{S[m][cname]['model']['top1']:.3f}" if cname in S[m] else "-" for m in MODELS]
        cl = [f"{ref[cname][k]['top1']:.3f}" if cname in ref else "-" for k in CLASSICAL]
        L.append(f"| {cname} | " + " | ".join(row) + " | " + " | ".join(cl) + " |")
    # determinism: the classical columns depend only on the crc32-seeded draws, so every run must agree
    dev = []
    for m in MODELS:
        for cname, res in S[m].items():
            for k in CLASSICAL:
                for m2 in MODELS:
                    if m2 != m and cname in S[m2]:
                        dev.append(abs(res[k]["top1"] - S[m2][cname][k]["top1"]))
    L += ["", f"Classical columns across runs (same crc32 draws): max |diff| = {max(dev) if dev else float('nan'):.2e} "
          f"over {len(dev)} comparisons; conditions per model: " + ", ".join(f"{m} {len(S[m])}" for m in MODELS)]

    os.makedirs("results/d3", exist_ok=True)
    open("results/d3/summary.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
