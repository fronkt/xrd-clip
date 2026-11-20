"""Figures for the RSOS manuscript (tasks/todo.md D-rest F3).

  python scripts/50_rsos_figs.py        -> paper/rsos/figs/fig{1,3,4,5,6}_*.{pdf,png}

The RSOS version of scripts/23 (which stays the record of the RSC Advances figures). Styles, sizes and helpers come
from 23; the data from results/rsos_numbers.json (48), the model's sweep (results/d3/sbv3_<model>_*.json) and
cache/real_bench/case_zzzuwk.npz (24). Figure 2, the model schematic, is paper/rsos/figs/gen_fig_model.py.
"""
from __future__ import annotations

import json
import sys
from importlib import import_module

import numpy as np

sys.path.insert(0, "scripts")
f23 = import_module("23_paper_figs")
nums = import_module("48_rsos_numbers")
plt, STYLE, REAL, SWEEP, DOUBLE, label = f23.plt, f23.STYLE, f23.REAL, f23.SWEEP, f23.DOUBLE, f23.label
f23.OUT = "paper/rsos/figs"
STYLE["model"]["label"] = "Learned model (seed 0)"
save = f23.save


def fig_schematic(rn: dict) -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    d, sp = rn["dataset"], rn["extra"]["sc_ambient_split"]
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
    box(0.5, y, 20, h, nl(["Refined structure of", "the measured pattern", "(opXRD CNRS, CC BY 4.0)"]))
    box(27, y, 18, h, nl(["CSD candidates", "(same non-H", "element set)"]))
    box(52, y, 17, h, "Anchor entries")
    box(76, y, 23, h, nl(["Same-form set", "(15/15 vs an anchor,", "at any T or p)"]))
    for x0, x1, t in ((21.1, 26.4, ["reduced-cell search", "(1.5 %, 2°)"]),
                      (45.6, 51.4, ["packing similarity", "(15 molecules)"]),
                      (69.6, 75.4, ["same test over the", "refcode family"])):
        arrow(x0, y + h / 2, x1, y + h / 2)
        note((x0 + x1) / 2, y + h + 1.2, nl(t))
    y2 = 1
    box(0.5, y2 + 0.5, 20, h, nl(["Measured pattern", "(Cu Kα1, 5–50° 2θ,", "background removed)"]), fc="#FFFFFF")
    box(27, y2 + 0.5, 25, h, nl(["Score against", f"{d['n_library']:,} simulated", "CSD patterns"]), fc="#FBEFE6", ec="#D55E00")
    box(59, y2, 40, h + 1, nl(["Rank of the first correct entry",
                               f"ambient single-crystal: {sp['n_ambient_patterns']} patterns, {sp['n_ambient_compounds']} compounds",
                               f"pressure series: {sp['n_pressure_series_patterns']} patterns, {sp['n_pressure_series_compounds']} compounds",
                               f"any same-form entry: {d['n_queries']} patterns, {d['n_compounds']} compounds"]),
        fc="#E3EEF8", ec="#0072B2")
    arrow(21.1, y2 + 0.5 + h / 2, 26.4, y2 + 0.5 + h / 2)
    arrow(52.6, y2 + 0.5 + h / 2, 58.4, y2 + 0.5 + h / 2)
    arrow(87.5, y - 0.6, 87.5, y2 + h + 1.6)
    note(88.5, 17.6, "correct entries", ha="left")
    save(fig, "fig1_benchmark")


def fig_sweep(sb: dict) -> None:
    axes = {"zero_offset": ("2θ zero offset (°)", False), "strain": ("Isotropic lattice strain", False),
            "thermal_dT": ("Anisotropic thermal strain, ΔT (K)", False), "domain_nm": ("Crystallite size (nm)", True),
            "march_bfdh": ("March–Dollase r, BFDH axis", True), "counts": ("Integrated counts", True)}
    conds = sb["conditions"]
    fig, axs = plt.subplots(2, 3, figsize=(DOUBLE, 0.58 * DOUBLE), sharey=True)
    for i, (ax, (axis, (xl, logx))) in enumerate(zip(axs.flat, axes.items())):
        pts = [(float(n.split("=")[1]), r) for n, r in conds.items() if n.startswith(axis + "=")]
        if axis in ("strain", "thermal_dT"):
            pts.append((0.0, conds["clean"]))
        if axis == "march_bfdh":
            pts.append((1.0, conds["clean"]))
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
        if axis == "march_bfdh":
            ax.set_xticks([0.5, 0.7, 1.0, 1.5, 2.0], ["0.5", "0.7", "1", "1.5", "2"])
            ax.minorticks_off()
            ax.text(0.03, 0.04, "plates", transform=ax.transAxes, fontsize=5.6, color="0.3")
            ax.text(0.97, 0.04, "r > 1", transform=ax.transAxes, fontsize=5.6, color="0.3", ha="right")
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
    save(fig, "fig3_simulated_sweep")


def fig_real(rn: dict) -> None:
    bm, sp = rn["bench"], rn["extra"]["sc_ambient_split"]
    order = ["lnq_bkg", "cos_zs_bkg", "pearson_bkg", "wcc_1_bkg", "wcc_2_bkg", "model_bkg"]
    amb = {"lnq_bkg": "lnq", "cos_zs_bkg": "cos_zs", "pearson_bkg": "pearson", "wcc_1_bkg": "wcc_1", "wcc_2_bkg": "wcc_2",
           "model_bkg": "model"}
    names = {"lnq_bkg": r"$\ln q$ lag + zero scan", "cos_zs_bkg": "Cosine + zero scan", "pearson_bkg": "Pearson, fixed grid",
             "wcc_1_bkg": "WCC, l = 1°", "wcc_2_bkg": "WCC, l = 2°", "model_bkg": "Learned model"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 0.40 * DOUBLE), gridspec_kw=dict(width_ratios=[1.25, 1]))
    y = np.arange(len(order))[::-1]
    yc, d = len(order), sp["critic2"]["macro_top1"]
    a.errorbar(d["value"], yc - 0.17, xerr=[[d["value"] - d["ci"][0]], [d["ci"][1] - d["value"]]], fmt="s", ms=3.5,
               color=STYLE["critic2"]["color"], mfc="white", elinewidth=0.8, capsize=0)
    for yi, k in zip(y, order):
        c = STYLE[REAL[k]]["color"]
        d = sp[amb[k]]["macro_top1"]
        a.errorbar(d["value"], yi - 0.17, xerr=[[d["value"] - d["ci"][0]], [d["ci"][1] - d["value"]]], fmt="s", ms=3.5,
                   color=c, mfc="white", elinewidth=0.8, capsize=0)
        d = bm[f"none|std|{k}"]
        a.errorbar(d["macro_top1"], yi + 0.17, xerr=[[d["macro_top1"] - d["macro_top1_ci"][0]],
                                                     [d["macro_top1_ci"][1] - d["macro_top1"]]],
                   fmt="o", ms=3.5, color=c, elinewidth=0.8, capsize=0)
        a.plot(bm[f"csls_aug_k10|std|{k}"]["macro_top1"], yi + 0.17, marker="|", ms=6, mew=1.0, color="k")
    a.set_yticks(list(y) + [yc], [names[k] for k in order] + ["VC-GPWDF re-rank\n" + r"of the $\ln q$ top 50"])
    a.set_xlim(-0.02, 1.0)
    a.set_xlabel("Top-1, compound-weighted (95 % CI)")
    a.grid(axis="x", alpha=0.25, lw=0.4)
    from matplotlib.lines import Line2D
    a.legend(handles=[Line2D([], [], marker="s", ls="", color="0.3", mfc="white", ms=3.5,
                             label=f"Ambient single-crystal ({sp['n_ambient_compounds']} compounds)"),
                      Line2D([], [], marker="o", ls="", color="0.3", ms=3.5,
                             label=f"Any same-form entry ({bm['none|std|lnq_bkg']['n_compounds']} compounds)"),
                      Line2D([], [], marker="|", ls="", color="k", ms=6, label="with CSLS")],
             loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=3, frameon=False)
    label(a, 0, x=-0.38)
    lc = rn["library_curve_macro"]
    sizes = ["2000", "20000", "100000", "full"]
    xs = [2000, 20000, 100000, rn["dataset"]["n_library"]]
    for k in order:
        c = STYLE[REAL[k]]["color"]
        b.errorbar(xs, [lc[f"none|{k}"][s][0] for s in sizes], yerr=[lc[f"none|{k}"][s][1] for s in sizes], color=c,
                   marker="o", ms=2.5, lw=1.2 if k == "model_bkg" else 1.0, ls=":" if k == "pearson_bkg" else "-", capsize=0,
                   elinewidth=0.6, label=names[k])
    b.set_xscale("log")
    b.set_ylim(-0.02, 1.0)
    b.set_xlabel("Library size (entries)")
    b.set_ylabel("Top-1, compound-weighted")
    b.grid(alpha=0.25, lw=0.4)
    b.legend(loc="center right", frameon=False, bbox_to_anchor=(1.0, 0.42))
    label(b, 1, x=-0.2)
    fig.tight_layout()
    save(fig, "fig4_real_benchmark")


LADDER = f23.LADDER


def ref_number(key: str, md: str = "paper/rsos/manuscript.md") -> int:
    """The Vancouver number citeproc gives `key`: its place in the order of first citation in the manuscript."""
    import re
    order: list[str] = []
    for grp in re.findall(r"\[(?:[^\]]*?)@[^\]]+\]", open(md, encoding="utf-8").read()):
        for k in re.findall(r"@([A-Za-z0-9_:-]+)", grp):
            if k not in order and not k.startswith(("fig:", "tbl:", "sec:", "eq:")):
                order.append(k)
    return order.index(key) + 1


def fig_ladder(lad: dict) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 0.40 * DOUBLE))
    xs = np.arange(len(LADDER))
    mc = STYLE["model"]["color"]
    series = (("pearson", lad[nums.MODEL]["results"], "Pearson, fixed grid", STYLE["pearson"]["color"], ":", "o", True),
              ("model", lad[nums.MODEL]["results"], "Learned model, seed 0", mc, "-", "o", True),
              ("model", lad[nums.SEED2]["results"], "Learned model, seed 1", mc, "--", "s", False),
              ("model", lad[nums.ORIGINAL]["results"], f"Earlier model (ref. {ref_number('cai2026urtc')})", "0.45", "-.", "^", False))
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
    rn = json.load(open("results/rsos_numbers.json"))
    fig_schematic(rn)
    fig_sweep(nums.sweep(nums.MODEL))
    fig_real(rn)
    f23.fig_case(rn, rn["case"], np.load("cache/real_bench/case_zzzuwk.npz"))
    import os
    os.replace("paper/rsos/figs/fig4_case_zzzuwk.pdf", "paper/rsos/figs/fig5_case_zzzuwk.pdf")
    os.replace("paper/rsos/figs/fig4_case_zzzuwk.png", "paper/rsos/figs/fig5_case_zzzuwk.png")
    fig_ladder(rn["ladder"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
