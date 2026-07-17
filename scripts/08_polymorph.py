"""Pivot headline experiment: within-family (polymorph) retrieval, model vs baselines vs chance.

Given the pattern of one polymorph, retrieve THAT polymorph from a library of its same-molecule
siblings. Composition and space group are (near-)identical within a family, so a composition/symmetry
method — the basis of XCCP and XtalNet — is pinned to chance (1/family_size) by construction. Only a
pattern<->packing model can win. This script reports:

  1. Clean within-family top-1/top-2, model vs chance (the headline number).
  2. A degradation sweep of within-family top-1, model vs correlation vs peak-list (does polymorph
     discrimination survive realistic shift/broadening/texture/noise — the crux of the paper).

Defaults are the pilot. For the full run: --sticks cache/sticks --test_refs refcodes/scaffold_test.txt
--families refcodes/polymorph_families.json --ckpt runs/xrdclip/best.pt --embed_dim 256.
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
    ap.add_argument("--families", default="refcodes/polymorph_families.json")
    ap.add_argument("--ckpt", default="runs/pilot_clip/best.pt")
    ap.add_argument("--out", default="results/polymorph.json")
    ap.add_argument("--fig", default="results/polymorph.png")
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--max_families", type=int, default=0, help="cap #families (0 = all); for debugging")
    args = ap.parse_args()

    from xrdclip.data.dataset import StickStore
    from xrdclip.data.polymorph import load_families, family_groups
    from xrdclip.models.clip import XRDClip
    from xrdclip.eval.polymorph_retrieval import within_family_retrieval, within_family_sweep

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.test_refs) as fh:
        test_refs = [ln.strip() for ln in fh if ln.strip()]
    store = StickStore(args.sticks, refcodes=set(test_refs))
    present = [rc for rc in test_refs if rc in store.records]

    families = load_families(args.families)
    groups = family_groups(present, families)
    if args.max_families and len(groups) > args.max_families:
        groups = dict(list(groups.items())[: args.max_families])
    n_targets = sum(len(v) for v in groups.values())
    if not groups:
        print("[polymorph] no polymorph families present in this test split — "
              "expected on the tiny pilot; run on the full scaffold_test.")
        return 1
    print(f"[polymorph] {len(groups):,} test polymorph families / {n_targets:,} targets on {device}")

    model = XRDClip(embed_dim=args.embed_dim).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[polymorph] loaded {args.ckpt} (epoch {ck.get('epoch', '?')})")

    from xrdclip.data import augment as A
    rng = np.random.default_rng(0)
    clean = within_family_retrieval(model, store, groups, A.CLEAN, device, rng)
    sweep = within_family_sweep(model, store, groups, device)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"n_families": len(groups), "n_targets": n_targets,
                   "clean": clean, "sweep": sweep}, fh, indent=2)
    print(f"[polymorph] wrote {args.out}")

    _plot(sweep, args.fig, len(groups))
    _summary(clean, sweep)
    return 0


AXIS_LABEL = {
    "zero_offset": "Zero-offset shift (deg 2theta)",
    "strain": "Isotropic strain (fractional)",
    "domain_nm": "Crystallite size (nm)",
    "march": "March-Dollase ratio r",
    "counts": "Counts (Poisson SNR)",
}
LOGX = {"domain_nm", "march", "counts"}


def _plot(sweep: dict, path: str, n_fam: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[polymorph] matplotlib unavailable ({e}); skipping figure")
        return

    chance = sweep["chance_top1"]
    axes = list(sweep["axes"].keys())
    fig, axs = plt.subplots(1, len(axes), figsize=(4 * len(axes), 3.6), squeeze=False)
    for j, axis in enumerate(axes):
        ax = axs[0][j]
        levels = list(sweep["axes"][axis].keys())
        xs = [float(l) for l in levels]
        for method, color, mk in (("model", "C0", "o"), ("correlation", "C3", "s"),
                                  ("peak_list", "C2", "^")):
            ys = [sweep["axes"][axis][l][method]["top1"] for l in levels]
            ax.plot(xs, ys, marker=mk, color=color, label=method.replace("_", "-"), lw=2, ms=5)
        ax.axhline(chance, color="gray", ls="--", lw=1, label="chance (composition-only)")
        if axis in LOGX:
            ax.set_xscale("log")
        ax.set_title(axis)
        ax.set_xlabel(AXIS_LABEL.get(axis, axis))
        ax.set_ylim(-0.03, 1.03)
        if j == 0:
            ax.set_ylabel("Within-family top-1")
        ax.grid(alpha=0.3)
    axs[0][0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Polymorph discrimination under degradation (families = {n_fam})", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[polymorph] wrote {path}")


def _summary(clean: dict, sweep: dict) -> None:
    print("\n" + "=" * 64)
    print("PIVOT HEADLINE — can the model tell polymorphs apart?")
    print(f"  clean within-family top-1: {clean['top1']:.3f}  (chance {clean['chance_top1']:.3f}, "
          f"n={clean['n_query']} over {clean['n_families']} families)")
    zo = sweep["axes"].get("zero_offset", {})
    if "0.1" in zo:
        m = zo["0.1"]["model"]["top1"]
        c = zo["0.1"]["correlation"]["top1"]
        print(f"  under 0.1deg shift: model={m:.3f}  correlation={c:.3f}  chance={sweep['chance_top1']:.3f}")
    print("=" * 64)


if __name__ == "__main__":
    raise SystemExit(main())
