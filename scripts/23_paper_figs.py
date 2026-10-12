"""Figures for the Digital Discovery manuscript (tasks/todo.md Step 2, B2).

  python scripts/23_paper_figs.py         -> paper/dd/figs/fig{2..5}_*.{pdf,png}

Inputs: results/paper_numbers.json (22), results/strong_baselines.json (13), cache/real_bench/hub_ranks.npz (21 eval)
and cache/real_bench/case_zzzuwk.npz (24; simulated CSD patterns, so the npz stays gitignored; the figure shows
derived patterns only). Fig. 1 (benchmark construction) is a schematic made separately.
Sizes follow RSC: double column 17.1 cm, single column 8.3 cm.
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

OUT = "paper/dd/figs"
DOUBLE, SINGLE = 17.1 / 2.54, 8.3 / 2.54
PANEL = "({})"                      # panel label format; one constant so the convention can change in one place
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7,
                     "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6, "xtick.labelsize": 6,
                     "ytick.labelsize": 6, "axes.linewidth": 0.6, "lines.linewidth": 1.0, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})

# Okabe-Ito colours, one per method across every figure
STYLE = {
    "model":   dict(color="#D55E00", label="Learned embedding (xrd-clip)"),
    "pearson": dict(color="#777777", label="Pearson, fixed grid"),
    "zs":      dict(color="#0072B2", label="l = 0 + zero scan"),
    "wcc_1":   dict(color="#E69F00", label="WCC, l = 1° + zero scan"),
    "wcc_2":   dict(color="#56B4E9", label="WCC, l = 2° + zero scan"),
    "lnq":     dict(color="#009E73", label="ln q lag + zero scan"),
}
REAL = {"model_bkg": "model", "pearson_bkg": "pearson", "cos_zs_bkg": "zs", "wcc_1_bkg": "wcc_1",
        "wcc_2_bkg": "wcc_2", "lnq_bkg": "lnq"}
SWEEP = {"pearson": "pearson", "pearson_zs": "zs", "wcc_1": "wcc_1", "wcc_2": "wcc_2", "lnq": "lnq", "model": "model"}


def label(ax, i, x=-0.16, y=1.04):
    ax.text(x, y, PANEL.format("abcdefgh"[i]), transform=ax.transAxes, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/{name}.pdf/.png")


# ---- Fig. 2: simulated sweep -----------------------------------------------------------------------------
def fig_sweep(sb: dict) -> None:
    axes = {"zero_offset": ("2θ zero offset (°)", False), "strain": ("Isotropic lattice strain", False),
            "thermal_dT": ("Anisotropic thermal strain, ΔT (K)", False), "domain_nm": ("Crystallite size (nm)", True),
            "march": ("March–Dollase r", False), "counts": ("Integrated counts", True)}
    conds = sb["conditions"]
    fig, axs = plt.subplots(2, 3, figsize=(DOUBLE, 0.58 * DOUBLE), sharey=True)
    for i, (ax, (axis, (xl, logx))) in enumerate(zip(axs.flat, axes.items())):
        pts = [(float(n.split("=")[1]), r) for n, r in conds.items() if n.startswith(axis + "=")]
        if axis in ("strain", "thermal_dT"):
            pts.append((0.0, conds["clean"]))
        pts.sort(key=lambda t: t[0])
        xs = np.array([p[0] for p in pts])
        for meth, key in SWEEP.items():
            st = STYLE[key]
            ys = np.array([p[1][meth]["top1"] for p in pts])
            lo = np.array([p[1][meth]["ci95"][0] for p in pts])
            hi = np.array([p[1][meth]["ci95"][1] for p in pts])
            ls = ":" if meth == "pearson" else "-"
            lw = 1.5 if meth == "model" else 1.0
            lab = "Pearson + zero scan" if meth == "pearson_zs" else st["label"]
            ax.plot(xs, ys, ls=ls, lw=lw, marker="o", ms=2.2, color=st["color"], label=lab)
            ax.fill_between(xs, lo, hi, color=st["color"], alpha=0.15, lw=0)
        ax.axhline(sb["chance_top1"], color="k", ls="--", lw=0.6)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xl)
        ax.set_ylim(0.25, 1.03)
        ax.grid(alpha=0.25, lw=0.4)
        label(ax, i, x=-0.12 if i % 3 == 0 else -0.06)
    for ax in axs[:, 0]:
        ax.set_ylabel("Within-family top-1")
    axs[0, 0].text(0.30, sb["chance_top1"] + 0.015, "chance", ha="right", va="bottom", fontsize=6)
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig2_simulated_sweep")


# ---- Fig. 3: real benchmark ------------------------------------------------------------------------------
def fig_real(pn: dict) -> None:
    bm = pn["bench"]
    order = ["lnq_bkg", "cos_zs_bkg", "pearson_bkg", "wcc_1_bkg", "wcc_2_bkg", "model_bkg"]
    names = {"lnq_bkg": "ln q lag + zero scan", "cos_zs_bkg": "Cosine + zero scan", "pearson_bkg": "Pearson, fixed grid",
             "wcc_1_bkg": "WCC, l = 1°", "wcc_2_bkg": "WCC, l = 2°", "model_bkg": "Learned embedding"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 0.36 * DOUBLE), gridspec_kw=dict(width_ratios=[1.25, 1]))
    y = np.arange(len(order))[::-1]
    for td, dy, mk, tl in (("std", 0.17, "o", "Any same-form entry"), ("sc", -0.17, "s", "Independent single-crystal entry")):
        for corr, fill in (("none", True), ("csls_aug_k10", False)):
            for yi, k in zip(y, order):
                d = bm[f"{corr}|{td}|{k}"]
                c = STYLE[REAL[k]]["color"]
                if corr == "none":
                    a.errorbar(d["macro_top1"], yi + dy, xerr=[[d["macro_top1"] - d["macro_top1_ci"][0]],
                                                               [d["macro_top1_ci"][1] - d["macro_top1"]]],
                               fmt=mk, ms=3.5, color=c, mfc=c if td == "std" else "white", elinewidth=0.8, capsize=0)
                else:
                    a.plot(d["macro_top1"], yi + dy, marker="|", ms=6, mew=1.0, color="k")
    a.set_yticks(y, [names[k] for k in order])
    a.set_xlim(-0.02, 1.0)
    a.set_xlabel("Top-1, compound-weighted (95 % CI)")
    a.grid(axis="x", alpha=0.25, lw=0.4)
    from matplotlib.lines import Line2D
    a.legend(handles=[Line2D([], [], marker="o", ls="", color="0.3", ms=3.5, label="Any same-form entry (123 compounds)"),
                      Line2D([], [], marker="s", ls="", color="0.3", mfc="white", ms=3.5,
                             label="Single-crystal entry only (30 compounds)"),
                      Line2D([], [], marker="|", ls="", color="k", ms=6, label="with CSLS")],
             loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=3, frameon=False)
    label(a, 0, x=-0.38)
    lc = pn["library_curve_macro"]
    sizes = ["2000", "20000", "100000", "full"]
    xs = [2000, 20000, 100000, pn["dataset"]["n_library"]]
    for k in ["pearson_bkg", "cos_zs_bkg", "wcc_1_bkg", "wcc_2_bkg", "lnq_bkg", "model_bkg"]:
        c = STYLE[REAL[k]]["color"]
        m = [lc[f"none|{k}"][s][0] for s in sizes]
        sd = [lc[f"none|{k}"][s][1] for s in sizes]
        b.errorbar(xs, m, yerr=sd, color=c, marker="o", ms=2.5, lw=1.2 if k == "model_bkg" else 1.0,
                   ls=":" if k == "pearson_bkg" else "-", capsize=0, elinewidth=0.6, label=names[k])
    b.set_xscale("log")
    b.set_ylim(-0.02, 1.0)
    b.set_xlabel("Library size (entries)")
    b.set_ylabel("Top-1, compound-weighted")
    b.grid(alpha=0.25, lw=0.4)
    b.legend(loc="center right", frameon=False, bbox_to_anchor=(1.0, 0.42))
    label(b, 1, x=-0.2)
    fig.tight_layout()
    save(fig, "fig3_real_benchmark")


# ---- Fig. 4: the ZZZUWK case ---------------------------------------------------------------------------
def fig_case(pn: dict, case_json: dict, z) -> None:
    ents = [str(e) for e in z["entries"]]
    ids = [str(i) for i in z["q_ids"]]
    qi = ids.index("pattern_522.json")          # 303 K pattern; its own Rietveld entry is ZZZUWK01
    grid = z["grid"]
    q = z["q_raw"][qi]
    own = z["raw"][ents.index("ZZZUWK01")]
    dec = z["raw"][ents.index("GUSRIV")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 0.34 * DOUBLE), gridspec_kw=dict(width_ratios=[1.6, 1]))
    sc = lambda v: (v - v.min()) / (v.max() - v.min())
    for off, v, c, t in ((2.2, q, "k", "measured, 303 K (opXRD, background removed)"),
                         (1.1, own, "#0072B2", "ZZZUWK01, its own Rietveld entry (simulated)"),
                         (0.0, dec, "#D55E00", "GUSRIV, unrelated decoy (simulated)")):
        a.plot(grid, sc(v) + off, color=c, lw=0.6)
        a.text(50, off + 0.75, t, ha="right", va="bottom", fontsize=6, color=c)
    a.set_xlim(5, 50)
    a.set_yticks([])
    a.spines["left"].set_visible(False)
    a.set_xlabel("2θ (°, Cu Kα1)")
    label(a, 0, x=-0.03)
    t1 = case_json["top1_of_20"]
    keys = ["pearson_bkg", "cos_zs_bkg", "wcc_1_bkg", "wcc_2_bkg", "lnq_bkg"]
    nm = ["Pearson,\nfixed", "Cosine +\nzero scan", "WCC\nl = 1°", "WCC\nl = 2°", "ln q +\nzero scan"]
    xs = np.arange(len(keys))
    b.bar(xs - 0.19, [t1[f"none|{k}"] for k in keys], 0.38, color=[STYLE[REAL[k]]["color"] for k in keys],
          label="uncorrected")
    b.bar(xs + 0.19, [t1[f"csls_aug_k10|{k}"] for k in keys], 0.38, color="white",
          edgecolor=[STYLE[REAL[k]]["color"] for k in keys], hatch="////", lw=0.8, label="with CSLS")
    for x, k in zip(xs, keys):
        for dx, c in ((-0.19, "none"), (0.19, "csls_aug_k10")):
            if t1[f"{c}|{k}"] == 0:
                b.text(x + dx, 0.3, "0", ha="center", va="bottom", fontsize=6)
    b.set_xticks(xs, nm)
    b.set_ylabel(f"ZZZUWK patterns ranked first (of {case_json['n_patterns']})")
    b.set_ylim(0, 20.5)
    b.set_yticks([0, 5, 10, 15, 20])
    b.legend(frameon=False, loc="upper right")
    label(b, 1, x=-0.2)
    fig.tight_layout()
    save(fig, "fig4_case_zzzuwk")


# ---- Fig. 5: rank distributions ------------------------------------------------------------------------
def fig_ranks(R) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 0.33 * DOUBLE), sharey=True)
    for i, (ax, td, t) in enumerate(zip(axs, ("std", "sc"), ("Any same-form entry counts", "Single-crystal entry only"))):
        for k, key in REAL.items():
            r = R[f"none|{td}|{k}"]
            r = np.sort(r[r >= 0] + 1)
            ax.step(r, np.arange(1, len(r) + 1) / len(r), where="post", color=STYLE[key]["color"],
                    lw=1.3 if key == "model" else 1.0, ls=":" if key == "pearson" else "-",
                    label=STYLE[key]["label"].replace("l = 0 + zero scan", "Cosine + zero scan"))
        ax.set_xscale("log")
        ax.set_xlim(0.8, 3.1e5)
        ax.set_xlabel("Rank of the first correct entry")
        ax.set_title(f"{t} (n = {len(r)} patterns)")
        ax.grid(alpha=0.25, lw=0.4)
        label(ax, i, x=-0.12 if i == 0 else -0.05)
    axs[0].set_ylabel("Fraction of patterns")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "fig5_rank_distributions")


def main() -> int:
    pn = json.load(open("results/paper_numbers.json"))
    fig_sweep(json.load(open("results/strong_baselines.json")))
    fig_real(pn)
    fig_case(pn, pn["case"], np.load("cache/real_bench/case_zzzuwk.npz"))
    fig_ranks(np.load("cache/real_bench/hub_ranks.npz"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
