"""Figure 1 of the RSOS manuscript: how the benchmark is built, drawn as an illustrated strip (2026-11-21).

  cd paper/rsos/figs && python gen_fig1_benchmark.py && python verify.py fig1_benchmark --expect Identify

Five stages, left to right: Measure -> Structure -> Identify -> Search -> Rank. The first four are shaded 3-D
vector icons (shapes.py on the iso3d renderer, copied from the Indiana STS poster's schematic so both read as one
style); the last is a plain plot of the answer-set sizes. Every count is read from results/rsos_numbers.json (script
48), so the figure cannot drift from the text. Replaces the box-and-arrow version that scripts/50_rsos_figs.py drew.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
import numpy as np                                       # noqa: E402
from matplotlib.patches import FancyArrowPatch           # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import palette as pal                                    # noqa: E402
from iso3d import Scene                                  # noqa: E402
from shapes import (CAM, LIGHT, P, RH, TH, UP, ball, cylinder, dark, edge, lying, paint, ring, screen,  # noqa: E402
                    shadow, shadow_strip)

pal.apply()
RN = json.load(open(os.path.join(HERE, "..", "..", "..", "results", "rsos_numbers.json"), encoding="utf-8"))
D, SP = RN["dataset"], RN["extra"]["sc_ambient_split"]
NL = chr(10)


def depth(p):
    return float(np.asarray(p, float) @ CAM.forward)


# ---------------------------------------------------------------- 1. measure: tube, beam, powder, diffracted rays
def scene_measure():
    sc = Scene(CAM, light=LIGHT)
    c = P(0.05, 0.1)
    shadow(sc, c, 0.55)
    cylinder(sc, c, 0.52, 0.52, 0.14, pal.MOUNT, layer=10, spec=0.15, gloss=8)
    face = c + 0.141 * UP
    sc.add_polygon(ring(face, 0.4), pal.METAL_LIGHT, layer=11, ec=dark(pal.METAL_LIGHT, 0.7), lw=0.6)
    rng = np.random.default_rng(3)
    for _ in range(70):                                  # powder grains
        r, t = 0.37 * np.sqrt(rng.random()), 2 * np.pi * rng.random()
        sc.add_polygon(ring(face + 0.001 * UP + r * np.cos(t) * RH + r * np.sin(t) * TH, 0.012 + 0.012 * rng.random(), 12),
                       pal.SLATE, layer=12)
    start = P(-0.86, 0.0, 0.95)
    tube = lying(start, RH, 0.1, 0.32)
    paint(sc, tube, pal.MOUNT, layer=13, spec=0.2, gloss=8)
    edge(sc, tube[-1], dark(pal.MOUNT), 13.5)
    det = P(0.52, 0.0, 0.95)
    det_tube = lying(det, RH, 0.1, 0.3)
    paint(sc, det_tube, pal.SLATE, layer=13, spec=0.2, gloss=8)
    edge(sc, det_tube[-1], dark(pal.SLATE), 13.5)
    return sc, start + 0.32 * RH, face + 0.002 * UP, det


# ---------------------------------------------------------------- 2. structure: a unit cell with two molecules
CELL_O = P(-0.52, 0.28, 0.06)
CA, CB, CC = 1.02 * RH, -0.62 * TH + 0.12 * RH, 0.92 * UP + 0.1 * RH


def molecule(centre, u, v, flip=1.0):
    """A small aromatic molecule: a six-ring with an OH and a C=O substituent (positions, elements, bonds)."""
    r = 0.11
    ring_pts = [centre + r * (np.cos(a) * u + np.sin(a) * v) for a in np.radians(np.arange(6) * 60 + 30)]
    o1 = centre + flip * 0.21 * (np.cos(np.radians(30)) * u + np.sin(np.radians(30)) * v)
    c7 = centre - flip * 0.21 * (np.cos(np.radians(30)) * u + np.sin(np.radians(30)) * v)
    o2 = c7 - flip * 0.11 * (np.cos(np.radians(-30)) * u + np.sin(np.radians(-30)) * v)
    atoms = [(p, "C") for p in ring_pts] + [(o1, "O"), (c7, "C"), (o2, "O")]
    bonds = [(i, (i + 1) % 6) for i in range(6)] + [(0, 6) if flip > 0 else (3, 6), (3, 7) if flip > 0 else (0, 7), (7, 8)]
    return atoms, bonds


def scene_structure(copy_colour=None):
    """copy_colour: draw the inversion-related molecule in this colour (figure 2 marks it as a symmetry copy)."""
    sc = Scene(CAM, light=LIGHT)
    o = CELL_O
    corners = {(i, j, k): o + i * CA + j * CB + k * CC for i in (0, 1) for j in (0, 1) for k in (0, 1)}
    faces = [[(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)], [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
             [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)], [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)],
             [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)], [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]]
    centre = o + 0.5 * (CA + CB + CC)
    shadow(sc, o + 0.5 * (CA + CB), 0.62, strength=0.8)
    for f in faces:
        poly = np.array([corners[k] for k in f])
        n = np.cross(poly[1] - poly[0], poly[2] - poly[0])
        if n @ (poly.mean(0) - centre) < 0:
            n = -n
        front = float(n @ CAM.forward) > 0
        sc.add_polygon(poly, pal.GLASS_FILL, layer=40 if front else 10, alpha=0.28 if front else 0.9)
    hidden = min(corners, key=lambda k: depth(corners[k]))
    for a in corners:
        for b in corners:
            if a < b and sum(x != y for x, y in zip(a, b)) == 1:
                seg = np.array([corners[a], corners[b]])
                if hidden in (a, b):
                    sc.add_line(seg, pal.GLASS, layer=11, lw=0.8, ls=(0, (3, 2)))
                else:
                    sc.add_line(seg, dark(pal.GLASS, 0.8), layer=41, lw=1.3)
    # two molecules related by the cell's inversion centre
    u1 = 0.85 * RH + 0.35 * UP
    u1 /= np.linalg.norm(u1)
    v1 = np.cross(u1, CAM.forward * 0.6 + UP * 0.4)
    v1 /= np.linalg.norm(v1)
    m1 = o + 0.36 * CA + 0.45 * CB + 0.33 * CC
    m2 = 2 * centre - m1
    atoms, bonds = [], []
    for m, flip in ((m1, 1.0), (m2, -1.0)):
        a, b = molecule(m, u1, v1, flip)
        off = len(atoms)
        atoms += a
        bonds += [(i + off, j + off) for i, j in b]
    order = np.argsort([depth(p) for p, _ in atoms])
    layer_of = {int(i): 12 + 2 * r for r, i in enumerate(order)}
    for i, j in bonds:
        bc = dark(copy_colour, 0.8) if copy_colour and i >= len(atoms) // 2 else dark(pal.CARBON, 0.9)
        sc.add_line(np.array([atoms[i][0], atoms[j][0]]), bc, layer=min(layer_of[i], layer_of[j]) - 0.6,
                    lw=2.2)
    for i, (p, el) in enumerate(atoms):
        col = pal.OXYGEN if el == "O" else pal.CARBON
        if copy_colour and i >= len(atoms) // 2:
            col = copy_colour
        ball(sc, p, 0.05 if el == "O" else 0.045, col, layer=layer_of[i])
    return (sc, m1, m2) if copy_colour else sc


# ---------------------------------------------------------------- 3. identify: a funnel, then the same-form set
TIER_H = 0.34
TIERS = [(0.70, 0.36, 0.56, "CSD"), (0.35, 0.22, 0.36, "same cell"), (0.0, 0.10, 0.22, "same packing")]
TIER_COL = [pal.TIER[0], pal.TIER[2], pal.TIER[3]]


def scene_identify():
    sc = Scene(CAM, light=LIGHT)
    c = P(-0.26, -0.1)
    shadow(sc, c, 0.3)
    marks = []
    for k in range(len(TIERS) - 1, -1, -1):
        z0, r0, r1, name = TIERS[k]
        cylinder(sc, c + z0 * UP, r0, r1, TIER_H, TIER_COL[k], layer=10 + (len(TIERS) - k), ambient=0.86, diffuse=0.18)
        marks.append((screen(c + r1 * RH + (z0 + TIER_H * 0.55) * UP), name))
    xs = (np.arange(4) - 1.5) * 0.19
    for i, x in enumerate(xs):                           # the same-form entries found through the refcode family
        centre = P(-0.26 + x, 0.42 - 0.6 * x ** 2, 0.075)
        shadow(sc, centre, 0.07, strength=0.9)
        ball(sc, centre, 0.075, pal.BLUE, layer=20 + i)
    return sc, marks


# ---------------------------------------------------------------- 4. search: the simulated library
DRUM_R, DRUM_H, DRUM_GAP = 0.5, 0.2, 0.05


def scene_search():
    sc = Scene(CAM, light=LIGHT)
    c = P(0, 0.12)
    shadow(sc, c, DRUM_R * 1.1)
    mids = []
    for k in range(3):
        z0 = k * (DRUM_H + DRUM_GAP)
        cylinder(sc, c + z0 * UP, DRUM_R, DRUM_R, DRUM_H, "#c9d2dc", layer=10 + k, ambient=0.8, diffuse=0.26)
        mids.append(screen(c + DRUM_R * TH + (z0 + DRUM_H / 2) * UP))
    top = screen(c + (3 * DRUM_H + 2 * DRUM_GAP) * UP)
    return sc, mids, top


def trace(ax, x0, x1, y0, h, peaks, colour, lw=1.2):
    x = np.linspace(0, 1, 400)
    y = sum(a * np.exp(-0.5 * ((x - p) / 0.012) ** 2) for p, a in peaks)
    ax.plot(x0 + (x1 - x0) * x, y0 + h * y, color=colour, lw=lw, solid_joinstyle="round", zorder=pal.Z_TOP)


PEAKS = [(0.12, 0.55), (0.22, 1.0), (0.27, 0.35), (0.41, 0.7), (0.47, 0.3), (0.63, 0.45), (0.8, 0.22)]


# ---------------------------------------------------------------- 5. rank: the three answer sets
def draw_rank(ax):
    rows = [("ambient", SP["n_ambient_patterns"], SP["n_ambient_compounds"]),
            ("pressure series", SP["n_pressure_series_patterns"], SP["n_pressure_series_compounds"]),
            ("any same form", D["n_queries"], D["n_compounds"])]
    assert rows[2] == ("any same form", 188, 121), rows
    x0, x1 = 0.0, 1.0
    for i, (name, n_p, n_c) in enumerate(rows):
        f = n_p / D["n_queries"]
        ax.plot([x0, x1], [i, i], color=pal.TRACK, lw=2.2, solid_capstyle="butt", zorder=2)
        ax.plot([x0, x0 + f * (x1 - x0)], [i, i], color=pal.BLUE, lw=2.2, solid_capstyle="butt", zorder=3)
        ax.plot(x0 + f * (x1 - x0), i, "o", ms=10, mfc=pal.BLUE, mec=dark(pal.BLUE, 0.7), mew=1.0, zorder=4)
        ax.text(x0, i - 0.36, name, ha="left", va="bottom", fontsize=12, color=pal.INK)
        ax.text(x1 + 0.08, i, f"{n_p} / {n_c}", ha="left", va="center", fontsize=12.5, color=pal.INK,
                fontweight="bold")
    ax.text(x1 + 0.08, -0.62, "patterns /" + NL + "compounds", ha="left", va="bottom", fontsize=10.5, color=pal.INK2,
            linespacing=1.05)
    ax.set_xlim(-0.08, 1.62), ax.set_ylim(3.1, -1.6)
    ax.axis("off")


# ---------------------------------------------------------------- the strip
W, H = 11.2, 3.75
ICON_W, RESULT_W, GAP = 1.86, 2.32, 0.36
Y_ICON, ICON_H = 1.28, 2.42
YLIM = (-0.52, -0.52 + ICON_H)
STAGES = [("Measure", f"{D['n_queries']} opXRD patterns" + NL + f"of {D['n_compounds']} compounds"),
          ("Structure", "refined from each" + NL + "measured pattern"),
          ("Identify", "same cell, then same" + NL + "molecular packing"),
          ("Search", f"against {D['n_library']:,}" + NL + "simulated patterns"),
          ("Rank", "first correct entry," + NL + "three answer sets")]


def main():
    assert abs(4 * ICON_W + RESULT_W + 4 * GAP - W) < 1e-6
    fig = plt.figure(figsize=(W, H))
    lefts = [k * (ICON_W + GAP) for k in range(5)]
    widths = [ICON_W] * 4 + [RESULT_W]

    def icon_axes(k):
        ax = fig.add_axes([lefts[k] / W, Y_ICON / H, widths[k] / W, ICON_H / H])
        ax.set_xlim(-widths[k] / 2, widths[k] / 2), ax.set_ylim(*YLIM)
        ax.set_aspect("equal"), ax.axis("off")
        return ax

    # 1 measure
    sc, src, spot, det = scene_measure()
    ax = icon_axes(0)
    sc.draw(ax)
    s0, s1, s2 = screen(src), screen(spot), screen(det)
    ax.plot([s0[0], s1[0]], [s0[1], s1[1]], color=pal.BEAM, lw=2.6, solid_capstyle="round", zorder=pal.Z_TOP)
    for dy in (-0.07, 0.0, 0.07):                                          # diffracted rays into the detector
        tip = s2 + np.array([-0.01, dy])
        ax.annotate("", xy=tip, xytext=s1, zorder=pal.Z_TOP,
                    arrowprops=dict(arrowstyle="-|>", color=pal.INK2, lw=1.0, mutation_scale=9, shrinkA=0, shrinkB=2))
    ax.text(s0[0] - 0.1, s0[1] + 0.14, "X-rays", ha="left", va="bottom", fontsize=11, color=dark(pal.BEAM, 0.75),
            zorder=pal.Z_TOP)
    trace(ax, 0.05, 0.9, 1.5, 0.3, PEAKS, pal.INK)                     # the measured pattern

    # 2 structure
    scene_structure().draw(icon_axes(1))

    # 3 identify
    sc, marks = scene_identify()
    ax = icon_axes(2)
    sc.draw(ax)
    for xy, name in marks:
        ax.text(xy[0] + 0.06, xy[1], name, ha="left", va="center", fontsize=11, color=pal.INK, zorder=pal.Z_TOP)
    ax.text(-0.26, -0.36, "same form", ha="center", va="center", fontsize=11, color=pal.BLUE, fontweight="bold",
            zorder=pal.Z_TOP)

    # 4 search
    sc, mids, top = scene_search()
    ax = icon_axes(3)
    sc.draw(ax)
    ax.text(mids[1][0], mids[1][1], f"{D['n_library']:,}", ha="center", va="center", fontsize=12.5,
            fontweight="bold", color=pal.INK, zorder=pal.Z_TOP)
    trace(ax, -0.45, 0.45, 1.5, 0.3, PEAKS, pal.INK)
    ax.annotate("", xy=(top[0], top[1] + 0.22), xytext=(top[0], 1.45), zorder=pal.Z_TOP,
                arrowprops=dict(arrowstyle="-|>", color=pal.BLUE_DEEP, lw=1.6, mutation_scale=13))

    # 5 rank
    draw_rank(fig.add_axes([lefts[4] / W, Y_ICON / H, widths[4] / W, ICON_H / H]))

    for k, (head, sub) in enumerate(STAGES):
        cx = (lefts[k] + widths[k] / 2) / W
        fig.text(cx, 1.12 / H, head, ha="center", va="top", fontsize=17, fontweight="bold",
                 color=pal.BLUE_DEEP if k == 4 else pal.INK)
        fig.text(cx, 0.76 / H, sub, ha="center", va="top", fontsize=12.5, color=pal.INK2, linespacing=1.15)
    for k in range(4):
        x0, x1 = (lefts[k] + widths[k] + 0.04) / W, (lefts[k + 1] - 0.04) / W
        y = (Y_ICON + 0.36 * ICON_H) / H
        fig.patches.append(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure, arrowstyle="-|>",
                                           mutation_scale=16, lw=1.8, color=pal.BLUE_DEEP))
    for p in pal.save_all(fig, "fig1_benchmark", outdir=HERE, dpi=500):
        print("wrote", p)


if __name__ == "__main__":
    main()
