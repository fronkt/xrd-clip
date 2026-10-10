"""Merge the Phase 0 worker outputs and write the Gate G0 table + figure.

  python scripts/13_g0_report.py results/sb_w*.json

Writes results/strong_baselines.json (merged), results/strong_baselines.png (sweep panels) and
results/g0_table.md (top-1 with 95 % family-bootstrap CIs, best strong baseline vs model, per-pair
cost). The G0 question: is there any condition where the model beats the best strong baseline?
"""
from __future__ import annotations

import glob
import json
import sys

STRONG = ["pearson_zs", "wcc_0.5", "wcc_1", "wcc_2", "lnq", "vc"]
SWEEPS = {"zero_offset": "2θ zero offset (°)", "strain": "Isotropic strain", "thermal_dT": "Thermal ΔT (K), anisotropic",
          "domain_nm": "Crystallite size (nm)", "march": "March–Dollase r", "counts": "Peak counts (Poisson)"}
COLORS = {"model": "C0", "pearson": "C7", "peak_list": "C8", "pearson_zs": "C3", "wcc_1": "C1", "lnq": "C2", "vc": "C4"}


def main() -> int:
    paths = sys.argv[1:] or sorted(glob.glob("results/sb_w*.json"))
    merged = None
    for p in paths:
        with open(p) as fh:
            d = json.load(fh)
        if merged is None:
            merged = {k: v for k, v in d.items() if k != "conditions"}
            merged["conditions"] = {}
        merged["conditions"].update(d["conditions"])
    with open("results/strong_baselines.json", "w") as fh:
        json.dump(merged, fh, indent=2)

    chance = merged["chance_top1"]
    rows = ["| Condition | Model | Best strong baseline | Δ (model − best) | URTC Pearson | Model µs/pair | Best µs/pair |",
            "|---|---|---|---|---|---|---|"]
    wins = []
    for name, res in merged["conditions"].items():
        m = res["model"]
        strong = {k: v for k, v in res.items() if k in STRONG}
        bk = max(strong, key=lambda k: strong[k]["top1"])
        b = strong[bk]
        delta = m["top1"] - b["top1"]
        sig = "**" if (m["ci95"][0] > b["ci95"][1] or m["ci95"][1] < b["ci95"][0]) else ""
        if m["ci95"][0] > b["ci95"][1]:
            wins.append(name)
        rows.append(f"| {name} | {m['top1']:.3f} [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}] | "
                    f"{bk} {b['top1']:.3f} [{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}] | {sig}{delta:+.3f}{sig} | "
                    f"{res['pearson']['top1']:.3f} | {m['us_per_pair']:.0f} | {b['us_per_pair']:.0f} |")
    head = [f"# Gate G0: model vs strong classical baselines (within-family top-1, chance {chance:.3f})", "",
            f"{merged['n_families']} families / {merged['n_queries']} queries. 95 % CIs from a family bootstrap. "
            "Bold Δ = CIs do not overlap. µs/pair is CPU/GPU wall time per (query, candidate) comparison; "
            "the model's figure excludes the one-off library embedding.", "",
            f"**Conditions where the model's CI is entirely above the best strong baseline's:** "
            f"{', '.join(wins) if wins else 'none'}.", ""]
    with open("results/g0_table.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(head + rows) + "\n")
    print("\n".join(head + rows))
    _plot(merged)
    return 0


def _plot(merged: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = merged["conditions"]
    fig, axs = plt.subplots(1, len(SWEEPS), figsize=(3.6 * len(SWEEPS), 3.4), squeeze=False)
    for ax, (axis, label) in zip(axs[0], SWEEPS.items()):
        pts = []
        for name, res in conds.items():
            if name.startswith(axis + "="):
                pts.append((float(name.split("=")[1]), res))
        if "clean" in conds and axis not in ("domain_nm", "march", "counts"):
            pts.append((0.0, conds["clean"]))
        pts.sort(key=lambda t: t[0])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        for meth, col in COLORS.items():
            if meth not in pts[0][1]:
                continue
            ys = [p[1][meth]["top1"] for p in pts]
            ax.plot(xs, ys, marker="o", ms=3, lw=1.8 if meth == "model" else 1.2, color=col,
                    ls="-" if meth in ("model",) + tuple(STRONG) else ":", label=meth)
        ax.axhline(merged["chance_top1"], color="gray", ls="--", lw=0.8)
        if axis in ("domain_nm", "counts"):
            ax.set_xscale("log")
        ax.set_title(label, fontsize=9)
        ax.set_ylim(0.3, 1.02)
        ax.grid(alpha=0.3)
    axs[0][0].set_ylabel("Within-family top-1")
    axs[0][0].legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig("results/strong_baselines.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    raise SystemExit(main())
