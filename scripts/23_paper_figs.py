"""Figures for the Digital Discovery manuscript (tasks/todo.md Step 2, B2).

  python scripts/23_paper_figs.py         -> paper/dd/figs/fig{1..6}_*.{pdf,png}

Inputs: results/paper_numbers.json (22), results/strong_baselines_v2.json (13, crc32-seeded sweep with ln q v2),
cache/real_bench_lnq2/hub_ranks.npz (21 eval with ln q v2), results/critic2_rerank.json (29), results/model_gap.json
+ model_gap_realism.json (30) and cache/real_bench/case_zzzuwk.npz (24; simulated CSD patterns, so the npz stays
gitignored; the figure shows derived patterns only).
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
                     "axes.spines.top": False, "axes.spines.right": False,
                     "mathtext.fontset": "stix"})   # serif math so that "ln" cannot read as "In"

# Okabe-Ito colours, one per method across every figure
STYLE = {
    "model":   dict(color="#D55E00", label="Learned embedding (xrd-clip)"),
    "pearson": dict(color="#777777", label="Pearson, fixed grid"),
    "zs":      dict(color="#0072B2", label="l = 0 + zero scan"),
    "wcc_1":   dict(color="#E69F00", label="WCC, l = 1° + zero scan"),
    "wcc_05":  dict(color="#000000", label="WCC, l = 0.5° + zero scan"),
    "wcc_2":   dict(color="#56B4E9", label="WCC, l = 2° + zero scan"),
    "lnq":     dict(color="#009E73", label=r"$\ln q$ lag + zero scan"),
    "critic2": dict(color="#CC79A7", label=r"VC-GPWDF re-rank of the $\ln q$ top 50"),
}
REAL = {"model_bkg": "model", "pearson_bkg": "pearson", "cos_zs_bkg": "zs", "wcc_1_bkg": "wcc_1",
        "wcc_2_bkg": "wcc_2", "lnq_bkg": "lnq"}
SWEEP = {"pearson": "pearson", "pearson_zs": "zs", "wcc_0.5": "wcc_05", "wcc_1": "wcc_1", "wcc_2": "wcc_2", "lnq": "lnq",
         "model": "model"}


def label(ax, i, x=-0.16, y=1.04):
    ax.text(x, y, PANEL.format("abcdefgh"[i]), transform=ax.transAxes, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/{name}.pdf/.png")


# ---- Fig. 1: benchmark construction (schematic) ------------------------------------------------------------
def fig_schematic(pn: dict) -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    d = pn["dataset"]
    fig, ax = plt.subplots(figsize=(DOUBLE, 0.40 * DOUBLE))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#F2F2F2", ec="0.35"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6, linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.7, color="0.25"))

    def note(x, y, text, ha="center"):
        ax.text(x, y, text, ha=ha, va="bottom", fontsize=5.6, color="0.25", linespacing=1.25)

    y, h = 24, 11
    nl = "\n".join
    box(0.5, y, 20, h, nl(["Rietveld structure of", "the measured pattern", "(opXRD CNRS, CC BY 4.0)"]))
    box(27, y, 18, h, nl(["CSD candidates", "(same non-H", "element set)"]))
    box(52, y, 17, h, "Anchor entry")
    box(76, y, 23, h, nl(["Same-form set", "(15/15 vs the anchor,", "at any T or p)"]))
    for x0, x1, t in ((21.1, 26.4, ["reduced-cell search", "(1.5 %, 2°)"]),
                      (45.6, 51.4, ["packing similarity", "(15 molecules)"]),
                      (69.6, 75.4, ["same test over the", "refcode family"])):
        arrow(x0, y + h / 2, x1, y + h / 2)
        note((x0 + x1) / 2, y + h + 1.2, nl(t))
    y2 = 2
    box(0.5, y2, 20, h, nl(["Measured pattern", "(Cu Kα1, 5–50° 2θ,", "background removed)"]), fc="#FFFFFF")
    box(27, y2, 25, h, nl(["Score against", f"{d['n_library']:,} simulated", "CSD patterns"]), fc="#FBEFE6", ec="#D55E00")
    box(59, y2, 40, h, nl(["Rank of the first correct entry",
                           f"any same-form entry: {d['n_queries']} patterns, {d['n_compounds']} compounds",
                           f"single-crystal entry only: {d['n_sc_patterns']} patterns, {d['n_sc_compounds']} compounds"]),
        fc="#E3EEF8", ec="#0072B2")
    arrow(21.1, y2 + h / 2, 26.4, y2 + h / 2)
    arrow(52.6, y2 + h / 2, 58.4, y2 + h / 2)
    arrow(87.5, y - 0.6, 87.5, y2 + h + 0.6)
    note(88.5, 16.8, "correct entries", ha="left")
    save(fig, "fig1_benchmark")


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
            ls = ":" if meth == "pearson" else "-." if meth == "wcc_0.5" else "-"
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
    fig.legend(h, lab, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig2_simulated_sweep")


# ---- Fig. 3: real benchmark ------------------------------------------------------------------------------
def fig_real(pn: dict, c2: dict) -> None:
    bm = pn["bench"]
    order = ["lnq_bkg", "cos_zs_bkg", "pearson_bkg", "wcc_1_bkg", "wcc_2_bkg", "model_bkg"]
    names = {"lnq_bkg": r"$\ln q$ lag + zero scan", "cos_zs_bkg": "Cosine + zero scan", "pearson_bkg": "Pearson, fixed grid",
             "wcc_1_bkg": "WCC, l = 1°", "wcc_2_bkg": "WCC, l = 2°", "model_bkg": "Learned embedding"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 0.40 * DOUBLE), gridspec_kw=dict(width_ratios=[1.25, 1]))
    y = np.arange(len(order))[::-1]
    # the variable-cell re-rank exists for the single-crystal set only: its own row above the others
    yc, d = len(order), c2["critic2_vcgpwdf"]["top1"]
    col = STYLE["critic2"]["color"]
    a.errorbar(d["value"], yc - 0.17, xerr=[[d["value"] - d["ci"][0]], [d["ci"][1] - d["value"]]], fmt="s", ms=3.5,
               color=col, mfc="white", elinewidth=0.8, capsize=0)
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
    a.set_yticks(list(y) + [yc], [names[k] for k in order] + ["VC-GPWDF re-rank\n" + r"of the $\ln q$ top 50"])
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
    for off, v, c, t in ((2.7, q, "k", "measured (opXRD, background removed)"),
                         (1.35, own, "#0072B2", "ZZZUWK01, its own powder-study entry (simulated)"),
                         (0.0, dec, "#D55E00", "GUSRIV, unrelated decoy (simulated)")):
        a.plot(grid, sc(v) + off, color=c, lw=0.6)
        a.text(50, off + 0.98, t, ha="right", va="bottom", fontsize=6, color=c)
    for hkl, tt in (("001", 12.0), ("101", 18.6), ("110", 20.1), ("111", 23.5), ("200", 28.6)):
        for off, v in ((2.7, q), (1.35, own)):
            win = np.abs(grid - tt) <= 0.3
            a.text(tt, off + sc(v)[win].max() + 0.03, hkl, ha="center", va="bottom", fontsize=4.8, color="0.3")
    a.set_xlim(5, 50)
    a.set_yticks([])
    a.spines["left"].set_visible(False)
    a.set_xlabel("2θ (°, Cu Kα1)")
    label(a, 0, x=-0.03)
    t1 = case_json["top1_of_20"]
    keys = ["pearson_bkg", "cos_zs_bkg", "wcc_1_bkg", "wcc_2_bkg", "lnq_bkg"]
    nm = ["Pearson,\nfixed", "Cosine +\nzero scan", "WCC\nl = 1°", "WCC\nl = 2°", "$\\ln q$ +\nzero scan"]
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


# ---- Fig. 6: effect ladder (own entries rendered with the measurement's effects) ----------------------------
LADDER = [("own_clean", "clean"), ("own_mask", "+ coverage"), ("own_width", "+ peak width"), ("own_ka2", "+ Kα2"),
          ("own_bkg", "+ background"), ("own_noise", "+ noise"), ("own_zero", "+ zero offset"), ("own_dw", "+ B"), ("own_all", "all six"),
          ("own_all_dw", "all six + B"), ("own_all_int30", "all six + jitter 0.3"),
          ("own_all_int50", "all six + jitter 0.5"), ("own_all_dw_int30", "all six + B + jitter 0.3"),
          ("measured", "measured")]


def fig_ladder(orig: dict, real: dict) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 0.40 * DOUBLE))
    xs = np.arange(len(LADDER))
    series = (("pearson", orig, "Pearson, fixed grid", STYLE["pearson"]["color"], ":", "o", True),
              ("model", orig, "Learned embedding (original)", STYLE["model"]["color"], "-", "o", True),
              ("model", real, "Learned embedding (realism retrain)", STYLE["model"]["color"], "--", "s", False))
    for meth, res, lab, c, ls, mk, filled in series:
        top1 = [res[f"{k}|{meth}|std"]["macro_top1"]["value"] for k, _ in LADDER]
        med = [res[f"{k}|{meth}|std"]["per_pattern"]["median_rank"] for k, _ in LADDER]
        kw = dict(color=c, ls=ls, marker=mk, ms=3, lw=1.0, mfc=c if filled else "white", label=lab)
        a.plot(xs, top1, **kw)
        if meth == "model":
            b.plot(xs, med, **kw)
    for ax in (a, b):
        ax.axvline(len(LADDER) - 1.5, color="0.5", lw=0.6, ls="--")
        ax.set_xticks(xs, [t for _, t in LADDER], rotation=55, ha="right", rotation_mode="anchor")
        ax.grid(alpha=0.25, lw=0.4)
    a.set_ylim(-0.02, 1.03)
    a.set_ylabel("Top-1, compound-weighted")
    b.set_yscale("log")
    b.set_ylabel("Median rank of the first correct entry")
    a.legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.55))
    label(a, 0, x=-0.14)
    label(b, 1, x=-0.16)
    fig.tight_layout()
    save(fig, "fig6_effect_ladder")


def main() -> int:
    pn = json.load(open("results/paper_numbers.json"))
    fig_schematic(pn)
    fig_sweep(json.load(open("results/strong_baselines_v2.json")))
    fig_real(pn, json.load(open("results/critic2_rerank.json")))
    fig_case(pn, pn["case"], np.load("cache/real_bench/case_zzzuwk.npz"))
    fig_ranks(np.load("cache/real_bench_lnq2/hub_ranks.npz"))
    fig_ladder(json.load(open("results/model_gap.json"))["results"], json.load(open("results/model_gap_realism.json"))["results"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
