"""Headline experiment: trained XRD-CLIP vs both classical baselines under one-at-a-time degradation.

Loads a trained checkpoint, runs the full robustness sweep on the test split (deduped so every
retrieval target is PXRD-distinct), writes the raw numbers to JSON, and renders the money figure:
top-1 retrieval accuracy vs degradation severity, model against correlation and peak-list.

The prediction on record (SPEC §4.3): correlation collapses under shift/strain, peak-list collapses
under broadening, and XRD-CLIP -- trained to be invariant to all of them -- degrades gracefully on
every axis. If XRD-CLIP does NOT win under shift, the augmentation is not reaching the encoder.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sticks", default="cache/pilot_sticks")
    ap.add_argument("--test_refs", default="refcodes_pilot/scaffold_test.txt")
    ap.add_argument("--ckpt", default="runs/pilot_clip/best.pt")
    ap.add_argument("--out", default="results/robustness.json")
    ap.add_argument("--fig", default="results/robustness.png")
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--max_lib", type=int, default=0, help="cap library size (0 = all test)")
    args = ap.parse_args()

    from xrdclip.data.dataset import StickStore
    from xrdclip.data.dedup import dedup_library
    from xrdclip.models.clip import XRDClip
    from xrdclip.eval.retrieve import robustness_sweep

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.test_refs) as fh:
        test_refs = [ln.strip() for ln in fh if ln.strip()]
    store = StickStore(args.sticks, refcodes=set(test_refs))

    # Dedup the eval library: retrieval targets must be PXRD-distinct or the ceiling sits below 1.0.
    reps, _ = dedup_library(store, [rc for rc in test_refs if rc in store.records])
    if args.max_lib:
        reps = reps[: args.max_lib]
    print(f"[robustness] test library: {len(reps)} PXRD-distinct structures on {device}")

    model = XRDClip(embed_dim=args.embed_dim).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[robustness] loaded {args.ckpt} (epoch {ck.get('epoch', '?')})")

    sweep = robustness_sweep(model, store, reps, device)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"n_lib": len(reps), "sweep": sweep}, fh, indent=2)
    print(f"[robustness] wrote {args.out}")

    _plot(sweep, args.fig, len(reps))
    _summary(sweep)
    return 0


AXIS_LABEL = {
    "zero_offset": ("Zero-offset shift (deg 2theta)", False),
    "strain": ("Isotropic strain (fractional)", False),
    "domain_nm": ("Crystallite size (nm)", True),      # log x, reversed severity
    "march": ("March-Dollase ratio r", True),          # 1 -> 0.5 = more texture
    "counts": ("Counts (Poisson SNR)", True),
}


def _plot(sweep: dict, path: str, n_lib: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[robustness] matplotlib unavailable ({e}); skipping figure")
        return

    axes = list(sweep.keys())
    fig, axs = plt.subplots(1, len(axes), figsize=(4 * len(axes), 3.6), squeeze=False)
    for j, axis in enumerate(axes):
        ax = axs[0][j]
        levels = list(sweep[axis].keys())
        xs = [float(l) for l in levels]
        for method, color, mk in (("model", "C0", "o"), ("correlation", "C3", "s"),
                                  ("peak_list", "C2", "^")):
            ys = [sweep[axis][l][method]["top1"] for l in levels]
            ax.plot(xs, ys, marker=mk, color=color, label=method.replace("_", "-"), lw=2, ms=5)
        label, logx = AXIS_LABEL.get(axis, (axis, False))
        if logx:
            ax.set_xscale("log")
        ax.set_title(axis)
        ax.set_xlabel(label)
        ax.set_ylim(-0.03, 1.03)
        if j == 0:
            ax.set_ylabel("Top-1 retrieval accuracy")
        ax.grid(alpha=0.3)
    axs[0][0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"XRD-CLIP vs classical search-match under degradation (library = {n_lib})", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[robustness] wrote {path}")


def _summary(sweep: dict) -> None:
    """Print the one comparison that decides the week-1 gate: shift-robustness, model vs correlation."""
    print("\n" + "=" * 64)
    print("WEEK-1 GATE — does the model beat correlation under peak shift?")
    zo = sweep.get("zero_offset", {})
    for lv in ("0.1", "0.2"):
        if lv in zo:
            m = zo[lv]["model"]["top1"]
            c = zo[lv]["correlation"]["top1"]
            verdict = "MODEL WINS" if m > c + 0.05 else "no clear win"
            print(f"  zero_offset={lv}deg : model={m:.3f}  correlation={c:.3f}  -> {verdict}")
    print("=" * 64)


if __name__ == "__main__":
    raise SystemExit(main())
