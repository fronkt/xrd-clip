"""Markdown tables from scripts/19_real_bench.py output.

  python scripts/20_real_bench_report.py [results/real_bench.json] [results/real_bench_table.md]
"""
from __future__ import annotations

import json
import sys

NAMES = {
    "pearson": "Pearson, fixed grid (URTC baseline)",
    "cos_zs": "cosine + 2θ zero scan (B1, l = 0)",
    "wcc_1": "de Gelder WCC l = 1° + zero scan",
    "wcc_2": "de Gelder WCC l = 2° + zero scan",
    "lnq": "ln q lag + zero scan (B2)",
    "model": "xrd-clip embedding (learned)",
}


def name(key: str) -> str:
    if "->" in key:
        left, right = key.split("->")
        return f"model top-{left.split('@')[1]} → {name(right)}"
    m, v = key.rsplit("_", 1)
    return f"{NAMES[m]} [{v}]"


def cell(s: dict) -> str:
    if not s or not s.get("n"):
        return "–"
    lo, hi = s["top1_ci"]
    return f"{s['top1']:.2f} [{lo:.2f}, {hi:.2f}]"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    src = sys.argv[1] if len(sys.argv) > 1 else "results/real_bench.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "results/real_bench_table.md"
    r = json.load(open(src))
    S, meta = r["summary"], r["meta"]
    n_own = sum(m["own_powder"] for m in meta)
    n_sc = sum(m["n_sc_in_lib"] > 0 for m in meta)
    singles = [k for k in r["keys"] if "->" not in k]
    hybrids = [k for k in r["keys"] if "->" in k and k.endswith("bkg")]
    L = [f"# Real organic PXRD benchmark: {r['n_queries']} opXRD patterns vs {r['n_library']:,} CSD entries",
         "",
         "Ground truth: every CSD entry of the same solid form as the pattern's own opXRD structure "
         "(CCDC packing similarity, 15/15 molecules; scripts/18_...). Ranks are exact over the whole "
         "library; entries that could not be compared are ignored. top-1 with a 95 % compound-bootstrap "
         "CI in brackets.",
         "",
         f"- **std**: any same-form entry counts. {n_own} of {r['n_queries']} patterns are the data their "
         "own CSD structure was refined from, so this is mostly self-retrieval.",
         f"- **single-crystal refs**: same-form powder-study entries are ignored; the answer must be an "
         f"independent single-crystal determination (n = {n_sc}). This is the realistic case.",
         f"- **within-family**: rank among the compound's own polymorphs only.",
         f"- Excluded: {len(r['excluded'])} patterns ({'; '.join(sorted(set(r['excluded'].values())))}).",
         "",
         "| Method | std top-1 | std top-10 | std median rank | single-crystal top-1 | single-crystal top-10 "
         "| single-crystal median rank | within-family top-1 (n) |",
         "|---|---|---|---|---|---|---|---|"]
    for k in singles:
        a, b = S.get(f"std|all|{k}", {}), S.get(f"sc|all|{k}", {})
        wf = r["within_family"].get(k, {})
        wft = f"{wf['top1']:.2f} ({wf['n']})" if wf.get("n") else "–"
        L.append(f"| {name(k)} | {cell(a)} | {a.get('top10', 0):.2f} | {a.get('median_rank', 0):.0f} | {cell(b)} "
                 f"| {b.get('top10', 0):.2f} | {b.get('median_rank', 0):.0f} | {wft} |")
    L += ["", "## Hybrid: model shortlist re-ranked by a classical score (bkg variant)", "",
          "| Pipeline | std top-1 | single-crystal top-1 |", "|---|---|---|"]
    for k in hybrids:
        L.append(f"| {name(k)} | {cell(S.get(f'std|all|{k}'))} | {cell(S.get(f'sc|all|{k}'))} |")
    L += ["", "## Library-size curve (std top-1, bkg variant; random decoy subsets, 5 repeats)", ""]
    sizes = sorted({sz for c in r["library_curve"].values() for sz in c if sz != "full"}, key=int)
    L += ["| Method | " + " | ".join(f"{int(s):,} decoys" for s in sizes) + f" | full ({r['n_library']:,}) |",
          "|---|" + "---|" * (len(sizes) + 1)]
    for k, c in r["library_curve"].items():
        L.append(f"| {name(k)} | " + " | ".join(f"{c[s][0]:.2f}" for s in sizes) + f" | {c['full'][0]:.2f} |")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
