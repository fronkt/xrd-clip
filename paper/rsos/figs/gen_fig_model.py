"""Figure 2 of the RSOS manuscript: the symmetry-aware contrastive dual encoder, drawn as illustrations (2026-11-21).

  cd paper/rsos/figs && python gen_fig_model.py && python verify.py fig_model --expect operators

Same style as figure 1 (gen_fig1_benchmark.py: shapes.py on the iso3d renderer, one camera and light). Top row: the
pattern branch (powder pattern -> 1-D residual CNN -> z_p). Bottom row: the structure branch (crystal with a symmetry
copy -> lattice, atom and operator tokens -> Transformer -> z_s). Right: both embeddings on the shared unit sphere.
Every number on the figure is read from the code it describes and asserted below, as in the box version (9d7dc19).
Vermilion = the operator tokens, new in this work (the model colour of the paper's plots).
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import numpy as np                                     # noqa: E402
from matplotlib.patches import FancyArrowPatch         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))
import palette as pal                                  # noqa: E402
from gen_fig1_benchmark import PEAKS, scene_structure, trace   # noqa: E402
from iso3d import Scene, icosphere                     # noqa: E402
from shapes import CAM, LIGHT, P, RH, TH, UP, box, dark, paint, screen, shadow   # noqa: E402
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
N_BLOCKS = len(_de.blocks)

pal.apply()
NL = chr(10)
W, H = 11.2, 5.0


def axes_in(fig, x, y, w, h, ylo=-0.35):
    """Axes at (x, y, w, h) inches whose data units are inches, origin at the bottom centre (+ylo)."""
    ax = fig.add_axes([x / W, y / H, w / W, h / H])
    ax.set_xlim(-w / 2, w / 2), ax.set_ylim(ylo, ylo + h)
    ax.set_aspect("equal"), ax.axis("off")
    return ax


# ---------------------------------------------------------------- icons
def scene_cnn():
    """The pattern encoder: one slab per stage, shorter (pooled) and taller (more channels) along the way."""
    sc = Scene(CAM, light=LIGHT)
    n = N_BLOCKS + 1                                     # stem + residual blocks
    xs = (np.arange(n) - (n - 1) / 2) * 0.5
    for k, x in enumerate(xs):
        height, thick, deep = 1.1 * 0.8 ** k, 0.06 + 0.035 * k, 0.22 + 0.07 * k   # positions pooled, channels grow
        c = P(x, 0.0, height / 2)
        shadow(sc, c, 0.2, strength=0.5)
        col = pal.TIER[0] if k == 0 else pal.TIER[1] if k < 4 else pal.TIER[2]
        box(sc, c, (thick, deep, height), col, layer=10 + k, ambient=0.8, diffuse=0.28)
    return sc, xs


def scene_vector(values, colour):
    """An embedding: a column of cubes, shade = value."""
    sc = Scene(CAM, light=LIGHT)
    s = 0.15
    shadow(sc, P(0, 0), 0.13, strength=0.6)
    base = pal.hex2rgb(colour)
    for k, v in enumerate(values):
        col = 1 - (0.15 + 0.85 * v) * (1 - base)          # pale to full colour
        box(sc, P(0, 0, s / 2 + k * s * 1.02), (s, s, s), col, layer=10 + k, ambient=0.85, diffuse=0.22)
    return sc


TOKENS = [("lattice", 1, "#c9ced4"), ("atoms", 4, "#7d8894"), ("operators", 3, pal.MODEL)]


def scene_tokens():
    sc = Scene(CAM, light=LIGHT)
    s, gap, ggap = 0.14, 0.03, 0.12
    total = sum(n for _, n, _ in TOKENS) * (s + gap) + 2 * ggap
    x, groups = -total / 2, []
    for name, n, col in TOKENS:
        x0 = x
        for _ in range(n):
            c = P(x + s / 2, 0.0, s / 2)
            shadow(sc, c, 0.1, strength=0.5)
            box(sc, c, (s, s, s), col, layer=10, ambient=0.8, diffuse=0.28)
            x += s + gap
        groups.append((name, (x0 + x - gap) / 2, col))
        x += ggap
    return sc, groups


def scene_transformer():
    sc = Scene(CAM, light=LIGHT)
    shadow(sc, P(0, 0), 0.6, strength=0.6)
    for k in range(DEPTH):
        box(sc, P(0, 0, 0.05 + 0.24 * k), (1.3, 0.62, 0.09), pal.TIER[1] if k % 2 else pal.TIER[0], layer=10 + k,
            ambient=0.82, diffuse=0.26)
    return sc


R_SPHERE = 1.0


def sphere_point(up, right):
    v = CAM.forward + right * RH + up * CAM.up
    return R_SPHERE * v / np.linalg.norm(v)


def scene_sphere():
    sc = Scene(CAM, light=LIGHT)
    shadow(sc, P(0, 0, 0) - R_SPHERE * UP, 0.8, strength=0.4)
    paint(sc, icosphere((0, 0, 0), R_SPHERE, 4), "#e3ebf4", layer=10, outline=True, ambient=0.86, diffuse=0.2,
          spec=0.15, gloss=10)
    zp, zs = sphere_point(0.55, -0.32), sphere_point(0.36, -0.12)
    rng = np.random.default_rng(7)
    others = [sphere_point(u, r) for u, r in zip(rng.uniform(-1.3, 1.2, 14), rng.uniform(-1.4, 1.4, 14))
              if np.hypot(u - 0.45, r + 0.22) > 0.45]
    return sc, zp, zs, others


# ---------------------------------------------------------------- the figure
def main():
    fig = plt.figure(figsize=(W, H))
    TOP, BOT, IH = 2.95, 0.55, 1.55                       # icon rows: bottom of the icon box (inches), height
    LBL = 0.12                                            # title sits this far under the icon box

    def label(cx, y, title, sub, colour=pal.INK):
        fig.text(cx / W, (y - LBL) / H, title, ha="center", va="top", fontsize=14, fontweight="bold", color=colour)
        fig.text(cx / W, (y - LBL - 0.3) / H, sub, ha="center", va="top", fontsize=11.5, color=pal.INK2,
                 linespacing=1.15)

    def arrow(x0, y0, x1, y1, colour=pal.BLUE_DEEP, rad=0.0):
        fig.patches.append(FancyArrowPatch((x0 / W, y0 / H), (x1 / W, y1 / H), transform=fig.transFigure,
                                           arrowstyle="-|>", mutation_scale=15, lw=1.6, color=colour,
                                           connectionstyle=f"arc3,rad={rad}"))

    # top row: pattern -> CNN -> z_p
    ax = axes_in(fig, 0.0, TOP, 2.0, IH)
    trace(ax, -0.9, 0.9, 0.1, 0.95, PEAKS, pal.INK, lw=1.5)
    ax.annotate("", xy=(0.95, 0.0), xytext=(-0.92, 0.0), arrowprops=dict(arrowstyle="-|>", color=pal.INK2, lw=0.9,
                                                                         mutation_scale=9))
    ax.text(0.95, -0.06, "2θ", ha="right", va="top", fontsize=11, color=pal.INK2)
    label(1.0, TOP, "Powder pattern", f"{N_POINTS:,} points, {TT_LO:.0f}–{TT_HI:.0f}° 2θ")
    arrow(2.05, TOP + 0.55, 2.45, TOP + 0.55)

    sc, _ = scene_cnn()
    ax = axes_in(fig, 2.5, TOP, 3.9, IH)
    sc.draw(ax)
    label(4.45, TOP, "Pattern encoder", f"1-D residual CNN: stem, {N_BLOCKS} residual blocks, pooling")
    arrow(6.45, TOP + 0.55, 6.85, TOP + 0.55)

    rng = np.random.default_rng(11)
    vp = rng.random(7)
    vs = np.clip(vp + rng.normal(0, 0.12, 7), 0, 1)       # the matching crystal's embedding: nearly the same
    scene_vector(vp, pal.BLUE_DEEP).draw(axes_in(fig, 6.85, TOP, 0.75, IH, ylo=-0.2))
    label(7.22, TOP, "$z_p$", f"{EMBED} numbers," + NL + "unit length")

    # bottom row: crystal -> tokens -> transformer -> z_s
    sc, m1, m2 = scene_structure(copy_colour=pal.MODEL)
    ax = axes_in(fig, 0.0, BOT, 2.0, IH, ylo=-0.25)
    sc.draw(ax)
    a, b = screen(m1), screen(m2)
    ax.annotate("", xy=b + np.array([-0.16, 0.06]), xytext=a + np.array([-0.12, 0.14]), zorder=pal.Z_TOP,
                arrowprops=dict(arrowstyle="-|>", color=pal.MODEL, lw=1.4, mutation_scale=11,
                                connectionstyle="arc3,rad=-0.5"))
    ax.text(-0.15, 1.1, "x′ = Rx + t", ha="center", va="bottom", fontsize=11.5, color=pal.MODEL,
            style="italic", zorder=pal.Z_TOP)
    label(1.0, BOT, "Crystal structure", "asymmetric unit, cell" + NL + "and symmetry operators")
    arrow(2.05, BOT + 0.55, 2.45, BOT + 0.55)

    sc, groups = scene_tokens()
    ax = axes_in(fig, 2.5, BOT, 1.9, IH)
    sc.draw(ax)
    for name, x, col in groups:
        ax.text(screen(P(x, 0.0, 0.0))[0], 0.42, name, ha="center", va="bottom", fontsize=10.5,
                color=dark(col, 0.85) if col == pal.MODEL else pal.INK2, rotation=0,
                fontweight="bold" if col == pal.MODEL else "normal", zorder=pal.Z_TOP)
    label(3.45, BOT, "Tokens", "one per cell, atom" + NL + "and operator")
    arrow(4.3, BOT + 0.55, 4.7, BOT + 0.55)

    scene_transformer().draw(axes_in(fig, 4.7, BOT, 1.75, IH))
    label(5.57, BOT, "Transformer", f"{DEPTH} layers, width {WIDTH}," + NL + f"{HEADS} heads")
    arrow(6.45, BOT + 0.55, 6.85, BOT + 0.55)

    scene_vector(vs, pal.BLUE_DEEP).draw(axes_in(fig, 6.85, BOT, 0.75, IH, ylo=-0.2))
    label(7.22, BOT, "$z_s$", f"{EMBED} numbers," + NL + "unit length")

    # right: the shared unit sphere
    SX, SY, SW, SH = 8.05, 1.35, 3.1, 2.75
    sc, zp, zs, others = scene_sphere()
    ax = axes_in(fig, SX, SY, SW, SH, ylo=-1.3)
    ax.set_xlim(-1.35, SW - 1.35)
    sc.draw(ax)
    for q in others:
        x, y = screen(q)
        ax.plot(x, y, "o", ms=6.5, mfc="white", mec=pal.SLATE, mew=1.2, zorder=pal.Z_TOP)
    (px, py), (qx, qy) = screen(zp), screen(zs)
    ax.plot(px, py, "o", ms=10, mfc=pal.BLUE_DEEP, mec="white", mew=1.0, zorder=pal.Z_TOP + 2)
    ax.plot(qx, qy, "s", ms=9.5, mfc=pal.BLUE, mec="white", mew=1.0, zorder=pal.Z_TOP + 2)
    ax.text(px - 0.1, py + 0.06, "$z_p$", ha="right", va="bottom", fontsize=13, color=pal.BLUE_DEEP, zorder=pal.Z_TOP)
    ax.text(qx + 0.1, qy - 0.04, "$z_s$", ha="left", va="top", fontsize=13, color=pal.BLUE, zorder=pal.Z_TOP)
    ax.text(1.08, 0.62, "s = $z_p$ · $z_s$", ha="left", va="center", fontsize=14, color=pal.INK,
            zorder=pal.Z_TOP)
    ax.text(1.08, 0.2, "cosine similarity", ha="left", va="center", fontsize=11, color=pal.INK2, zorder=pal.Z_TOP)
    ax.plot(1.14, -0.35, "o", ms=6.5, mfc="white", mec=pal.SLATE, mew=1.2, zorder=pal.Z_TOP)
    ax.text(1.26, -0.35, "other crystals", ha="left", va="center", fontsize=11, color=pal.INK2, zorder=pal.Z_TOP)
    arrow(7.65, TOP + 0.55, SX + 0.55, SY + 1.95, rad=-0.25)
    arrow(7.65, BOT + 0.55, SX + 0.55, SY + 0.35, rad=0.25)
    cx = SX + 1.35
    fig.text(cx / W, (SY - 0.05) / H, "Shared embedding", ha="center", va="top", fontsize=14, fontweight="bold",
             color=pal.BLUE_DEEP)
    fig.text(cx / W, (SY - 0.35) / H,
             f"training: same crystal pulled together," + NL + f"others pushed apart (batches of {BATCH})" + NL +
             f"retrieval: rank all {N_LIB:,} entries by s",
             ha="center", va="top", fontsize=11.5, color=pal.INK2, linespacing=1.2)

    for p in pal.save_all(fig, "fig_model", outdir=HERE, dpi=500):
        print("wrote", p)


if __name__ == "__main__":
    main()
