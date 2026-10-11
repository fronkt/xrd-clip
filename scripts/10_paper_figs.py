"""Publication figures for the URTC paper, regenerated from committed results JSONs.

Reads results/polymorph.json (one-dial sweeps) + results/combined.json (field average +
sign decomposition) and writes vector PDFs sized for the IEEE two-column template:
  paper/figs/fig_sweeps.pdf   full-width, 3 panels: zero-offset / strain / crystallite size
  paper/figs/fig_combined.pdf full-width, 2 panels: field average / sign decomposition

Usage:  python scripts/10_paper_figs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

# One style for every panel: the model is the solid blue line; baselines dashed/dotted.
STYLE = {
    "model": dict(color="#1f77b4", ls="-", marker="o", lw=2.0, ms=4, label="XRD-CLIP (ours)"),
    "correlation": dict(color="#d62728", ls="--", marker="s", lw=1.6, ms=4, label="full-profile correlation"),
    "peak_list": dict(color="#2ca02c", ls=":", marker="^", lw=1.6, ms=4, label="peak-list match"),
}
CHANCE_KW = dict(color="0.45", ls=(0, (4, 2)), lw=1.2, label="composition-only chance")

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.dpi": 200, "savefig.bbox": "tight",
})


def _axis_series(axis: dict, methods=("model", "correlation", "peak_list")):
    xs = sorted(axis.keys(), key=float)
    out = {m: [axis[x][m]["top1"] for x in xs] for m in methods}
    return [float(x) for x in xs], out


def fig_sweeps():
    d = json.loads((ROOT / "results" / "polymorph.json").read_text())
    chance = d["sweep"]["chance_top1"]
    axes_data = d["sweep"]["axes"]

    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.35), sharey=True)

    # (a) zero offset
    xs, ys = _axis_series(axes_data["zero_offset"])
    for m, kw in STYLE.items():
        axs[0].plot(xs, ys[m], **kw)
    axs[0].set_xlabel(r"$2\theta$ zero offset (deg)")
    axs[0].set_title("(a) sample displacement / zero error")

    # (b) strain
    xs, ys = _axis_series(axes_data["strain"])
    for m, kw in STYLE.items():
        axs[1].plot([100 * x for x in xs], ys[m], **kw)
    axs[1].set_xlabel("lattice strain (%)")
    axs[1].set_title("(b) thermal / compositional strain")

    # (c) crystallite size -- plot small->large left->right on a log axis
    xs, ys = _axis_series(axes_data["domain_nm"])
    axs[2].set_xscale("log")
    for m, kw in STYLE.items():
        axs[2].plot(xs, ys[m], **kw)
    axs[2].set_xlabel("crystallite size (nm)")
    axs[2].invert_xaxis()  # degradation increases to the right
    axs[2].set_title("(c) Scherrer broadening (limitation)")

    for ax in axs:
        ax.axhline(chance, **CHANCE_KW)
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.25, lw=0.5)
    axs[0].set_ylabel("within-family top-1")

    fig.tight_layout(pad=0.4, rect=(0, 0.14, 1, 1))
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False, handlelength=2.2,
               columnspacing=1.4)
    fig.savefig(OUT / "fig_sweeps.pdf")
    print("wrote", OUT / "fig_sweeps.pdf")


def fig_combined():
    d = json.loads((ROOT / "results" / "combined.json").read_text())
    chance = d["field"]["chance_top1"]
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.4))

    # (a) field average bars
    levels = ["mild", "moderate", "harsh"]
    methods = ["model", "correlation", "peak_list"]
    w = 0.26
    for j, m in enumerate(methods):
        vals = [d["field"]["points"][lv][m]["top1"] for lv in levels]
        axs[0].bar([i + (j - 1) * w for i in range(3)], vals, width=w,
                   color=STYLE[m]["color"], alpha=0.9 if m == "model" else 0.6,
                   label=STYLE[m]["label"])
    axs[0].set_xticks(range(3), [f"{lv}" for lv in levels])
    axs[0].set_ylabel("within-family top-1")
    axs[0].set_title("(a) all effects at once, random shift signs")

    # (b) sign decomposition bars
    combos = ["shifts_cancel", "shifts_add"]
    for j, m in enumerate(methods):
        vals = [d["sign"]["combos"][c][m]["top1"] for c in combos]
        axs[1].bar([i + (j - 1) * w for i in range(2)], vals, width=w,
                   color=STYLE[m]["color"], alpha=0.9 if m == "model" else 0.6)
    axs[1].set_xticks(range(2), ["shifts cancel", "shifts add"])
    axs[1].set_title(r"(b) same $|$shift$|$, only relative sign flipped")

    for ax in axs:
        ax.axhline(chance, **CHANCE_KW)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25, lw=0.5, axis="y")

    fig.tight_layout(pad=0.4, rect=(0, 0.14, 1, 1))
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False)
    fig.savefig(OUT / "fig_combined.pdf")
    print("wrote", OUT / "fig_combined.pdf")


if __name__ == "__main__":
    fig_sweeps()
    fig_combined()
