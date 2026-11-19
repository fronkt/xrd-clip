"""Schematic of the paper's model (RSOS manuscript, Section 3.4): the symmetry-aware contrastive dual encoder.

  cd paper/rsos/figs && python gen_fig_model.py && python verify.py fig_model --expect "symmetry operator"

Every number on the figure is read from the code it describes (src/xrdclip/models/encoders.py, dataset.py,
train.py defaults and runs/d3_sym_s0/best.pt args) and asserted below, so the drawing cannot drift from the model.
Colours: the model colour is the paper's (scripts/23_paper_figs.py STYLE["model"], Okabe-Ito vermilion).
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import numpy as np                                     # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle   # noqa: E402

import palette as P                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))
from xrdclip.models.encoders import DiffractionEncoder, StructureEncoder   # noqa: E402

# ---- facts drawn on the figure, each checked against the code ------------------------------------------------
N_POINTS, TT_LO, TT_HI, STEP = 2251, 5.0, 50.0, 0.02    # xrdclip.data.augment grid
EMBED, WIDTH, DEPTH, HEADS = 256, 256, 4, 8              # StructureEncoder defaults used by the D3 runs
N_LIB, BATCH = 309_080, 512
_se = StructureEncoder(symops=True)
assert _se.transformer.num_layers == DEPTH and _se.op_proj[0].in_features == 15 and _se.lattice_proj[0].in_features == 8
assert _se.transformer.layers[0].self_attn.num_heads == HEADS and _se.transformer.layers[0].linear1.in_features == WIDTH
_de = DiffractionEncoder()
assert _de.stem[0].kernel_size == (15,) and len(_de.blocks) == 5 and _de.blocks[0].conv1.kernel_size == (7,)
assert _de.blocks[-1].conv2.out_channels == 256 and _de.proj[-1].out_features == EMBED
assert round((TT_HI - TT_LO) / STEP) + 1 == N_POINTS

MODEL = "#D55E00"            # scripts/23_paper_figs.py STYLE["model"]
BOX = "#F4F4F4"
EDGE = P.PANEL_EDGE
FS, FS_S = 7.0, 6.2

P.apply()
fig = plt.figure(figsize=(17.1 / 2.54, 7.6 / 2.54))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 44)
ax.axis("off")


def box(x, y, w, h, title, body="", fc=BOX, ec=EDGE, title_color=P.INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec=ec, lw=0.7))
    ax.text(x + w / 2, y + h - 1.0, title, ha="center", va="top", fontsize=FS, fontweight="bold", color=title_color)
    if body:
        ax.text(x + w / 2, y + h - 3.6, body, ha="center", va="top", fontsize=FS_S, color=P.INK, linespacing=1.25)


def arrow(x0, y0, x1, y1, color=P.LEADER):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=color,
                                 shrinkA=0, shrinkB=0))


# ---- left: the pattern branch ----------------------------------------------------------------------------------
LX, LW = 2.0, 29.0
box(LX, 30.5, LW, 12.0, "Powder pattern")
tt = np.linspace(0, 1, 400)                     # a drawn trace, labelled as a schematic: not data
peaks = [(0.12, 0.55), (0.24, 1.0), (0.31, 0.35), (0.46, 0.7), (0.52, 0.3), (0.68, 0.45), (0.83, 0.25)]
yy = 0.05 + sum(h * np.exp(-0.5 * ((tt - c) / 0.008) ** 2) for c, h in peaks)
ax.plot(LX + 2.0 + 25.0 * tt, 34.2 + 4.2 * yy / yy.max(), color=P.INK, lw=0.6)
ax.text(LX + LW / 2, 33.6, f"Cu K$\\alpha_1$ 2$\\theta$ {TT_LO:g}–{TT_HI:g}°, {N_POINTS:,} points, standardised",
        ha="center", va="top", fontsize=FS_S)
arrow(LX + LW / 2, 30.2, LX + LW / 2, 26.8)
box(LX, 17.0, LW, 9.5, "Pattern encoder",
    "1-D residual CNN: stem (kernel 15),\n5 residual blocks (64–256 channels),\naverage pool, MLP")
arrow(LX + LW / 2, 16.7, LX + LW / 2, 13.6)
box(LX, 8.0, LW, 5.3, "", fc="white")
ax.text(LX + LW / 2, 10.65, r"$z_\mathrm{p}$: 256 numbers, unit length", ha="center", va="center", fontsize=FS)

# ---- right: the structure branch ----------------------------------------------------------------------------------
RX, RW = 54.5, 44.0
box(RX, 27.0, RW, 15.5, "Structure tokens (one crystal)")
rows = [("lattice", 1, "#BDBDBD", "lattice: log a, b, c; cos α, β, γ; log V; atoms/100"),
        ("atoms", 5, "#8C8C8C", "one per asymmetric-unit atom: element, sin and cos 2πx"),
        ("ops", 4, MODEL, "one per symmetry operator x′ = Rx + t: R, sin and cos 2πt")]
for i, (_, n, c, lab) in enumerate(rows):
    y = 37.0 - 3.6 * i
    for j in range(n):
        ax.add_patch(Rectangle((RX + 1.6 + 1.55 * j, y), 1.25, 1.9, fc=c, ec="white", lw=0.4))
    ax.text(RX + 9.9, y + 0.95, lab, ha="left", va="center", fontsize=FS_S,
            color=MODEL if c == MODEL else P.INK)
ax.text(RX + RW - 1.0, 27.7, "operator tokens: added in this work", ha="right", va="bottom", fontsize=FS_S,
        color=MODEL, style="italic")
arrow(RX + RW / 2, 26.7, RX + RW / 2, 23.3)
box(RX, 17.0, RW, 6.0, "Transformer encoder", f"{DEPTH} layers, width {WIDTH}, {HEADS} heads, every token attends to every token")
arrow(RX + RW / 2, 16.7, RX + RW / 2, 13.6)
box(RX, 8.0, RW, 5.3, "", fc="white")
ax.text(RX + RW / 2, 11.9, "lattice-token output and mean of atom outputs, concatenated, MLP", ha="center", va="center", fontsize=FS_S)
ax.text(RX + RW / 2, 9.55, r"$z_\mathrm{s}$: 256 numbers, unit length", ha="center", va="center", fontsize=FS)

# ---- middle: similarity, training and retrieval ---------------------------------------------------------------
MX, MW = 33.8, 17.6
box(MX, 8.0, MW, 5.3, "", fc="white", ec=MODEL)
ax.text(MX + MW / 2, 11.9, "cosine similarity", ha="center", va="center", fontsize=FS_S)
ax.text(MX + MW / 2, 9.55, r"$s = z_\mathrm{p}^\mathrm{T} z_\mathrm{s}$", ha="center", va="center", fontsize=FS + 0.5)
arrow(LX + LW + 0.3, 10.65, MX - 0.3, 10.65)
arrow(RX - 0.3, 10.65, MX + MW + 0.3, 10.65)
ax.text(MX + MW / 2, 25.5, "Training", ha="center", va="top", fontsize=FS, fontweight="bold")
ax.text(MX + MW / 2, 23.4, f"symmetric InfoNCE over\nbatches of {BATCH} simulated\npatterns and their structures,\n"
        "sibling polymorphs\nin the same batch", ha="center", va="top", fontsize=FS_S, linespacing=1.25)
ax.text(MX + MW / 2, 5.9, "Retrieval", ha="center", va="top", fontsize=FS, fontweight="bold")
ax.text(MX + MW / 2, 3.9, f"rank all {N_LIB:,} library entries\nby $s$ for each measured pattern",
        ha="center", va="top", fontsize=FS_S, linespacing=1.25)

for p in P.save_all(fig, "fig_model", outdir=HERE):
    print("wrote", p)
