"""Compose slide-ready ROY polymorph figures from the PyMOL outline renders + CCDC-simulated PXRD."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).parent
OUT = HERE / "out"
plt.rcParams.update({"font.family": "Arial", "font.size": 11})
INK, GREY = "#1F2937", "#6B7280"
FORM = {  # crystal colour of each form (CSD colour field) -> swatch
    "Y": ("yellow", "#E3B505"), "OP": ("orange plates", "#F28C28"),
    "ON": ("orange needles", "#D9661F"), "R": ("red", "#C0392B")}
SG = {"Y": "P2$_1$/n", "OP": "P2$_1$/n", "ON": "P2$_1$/c", "R": "P$\\overline{1}$"}
THETA = {"Y": 104.7, "OP": 46.1, "ON": 52.6, "R": 21.7}
REF = {"Y": "QAXMEH01", "OP": "QAXMEH03", "ON": "QAXMEH", "R": "QAXMEH02"}
pxrd = json.load(open(HERE / "roy_pxrd.json"))


def show(ax, name):
    im = Image.open(OUT / name).convert("RGBA")
    bg = Image.new("RGBA", im.size, "white"); bg.alpha_composite(im)
    ax.imshow(np.asarray(bg.convert("RGB"))); ax.set_axis_off()


def head(ax, f, short=False):
    """Bold form label with a colour swatch placed just left of the measured text."""
    name, col = FORM[f]
    t = ax.text(0.5, 1.02, f if short else f"{f}  ({name})", transform=ax.transAxes, ha="center", va="bottom",
                fontsize=14, fontweight="bold", color=INK)
    ax.figure.canvas.draw()
    bb = t.get_window_extent().transformed(ax.transAxes.inverted())
    ax.text(bb.x0 - 0.02, 1.02, "■", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=16, color=col)


# ---- Figure A: Y vs OP, same formula AND same space group --------------------------------
fig = plt.figure(figsize=(7.2, 6.4), dpi=300)
gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.05, 0.62], hspace=0.34, wspace=0.08,
                      left=0.03, right=0.97, top=0.93, bottom=0.08)
for c, f in enumerate(["Y", "OP"]):
    a1 = fig.add_subplot(gs[0, c]); show(a1, f"roy_mol_{f}.png"); head(a1, f)
    a1.text(0.5, -0.02, f"ring twist {THETA[f]:.0f}°", transform=a1.transAxes, ha="center", va="top",
            fontsize=11, color=GREY)
    a2 = fig.add_subplot(gs[1, c]); show(a2, f"roy_cell_{f}_c.png")
    a2.text(0.5, -0.03, f"{SG[f]}, Z = 4  ·  unit cell viewed along c  ·  CSD {REF[f]}",
            transform=a2.transAxes, ha="center", va="top", fontsize=9, color=GREY)
ax = fig.add_subplot(gs[2, :])
for f, off in (("Y", 1.1), ("OP", 0.0)):
    tt, I = np.array(pxrd[f]["tt"]), np.array(pxrd[f]["I"])
    ax.plot(tt, I / I.max() + off, color=FORM[f][1], lw=1.0)
    ax.text(49.5, off + 0.55, f, color=FORM[f][1], ha="right", fontweight="bold")
ax.set_xlim(5, 50); ax.set_yticks([])
ax.set_xlabel("2θ (°, Cu Kα$_1$)  —  simulated PXRD", color=INK)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
fig.savefig(OUT / "roy_Y_vs_OP.png", dpi=300, facecolor="white")

# ---- Figure B: four conformers, one molecule, four crystals -------------------------------
fig, axs = plt.subplots(1, 4, figsize=(9.6, 3.7), dpi=300)
for ax, f in zip(axs, ["Y", "OP", "ON", "R"]):
    show(ax, f"roy_mol_{f}.png"); head(ax, f, short=True)
    ax.text(0.5, -0.04, f"{FORM[f][0]}\n{SG[f]}  ·  twist {THETA[f]:.0f}°", transform=ax.transAxes, ha="center",
            va="top", fontsize=11, color=GREY)
fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.16, wspace=0.05)
fig.savefig(OUT / "roy_four_forms.png", dpi=300, facecolor="white")
print("ok")
