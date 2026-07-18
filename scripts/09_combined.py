"""Combined realistic-degradation experiment — polymorph retrieval in the field.

The one-dial-at-a-time sweep (08_polymorph.py) shows the model ties or wins on the two peak-POSITION
axes (displacement, strain) and loses on the position-preserving ones (broadening, texture, noise),
because a full-profile correlation stays near-perfect whenever peaks don't move. The fair reviewer
objection: "so on 3 of 5 axes the classical baseline still wins — why care?" The answer is that a
real measurement never turns exactly one dial, and — crucially — the naive way to show this (stack
every dial at a fixed value) is itself a trap:

    A fixed same-sign sample-displacement and thermal strain PARTIALLY CANCEL. Displacement pushes
    2theta up; thermal cell-expansion pushes it down. In the mid-angle range they offset, which
    artificially rescues correlation. Flip one sign and correlation collapses again. So any single
    deterministic operating point reports whatever sign the experimenter happened to choose.

This script therefore reports two honest, complementary things:

  PANEL A — field average (the headline). Each crystal gets its OWN sample-displacement and thermal
    strain, drawn independently in sign and magnitude from a realistic band, atop fixed broadening /
    texture / noise / background. Over the test set, cancellation and addition occur at their natural
    rates, so the number is the true expectation over real samples — free of the sign artifact.

  PANEL B — sign decomposition (the mechanism). Hold |displacement| and |strain| fixed, vary only
    their relative sign. The model is flat (genuine invariance); correlation swings wide, revealing
    its apparent robustness as a coincidental cancellation the model never needed.

Full run:
  PYTHONPATH=src python scripts/09_combined.py \
      --sticks cache/sticks --test_refs refcodes/scaffold_test.txt \
      --families refcodes/polymorph_families.json --ckpt runs/xrdclip_poly/best.pt --embed_dim 256
"""
from __future__ import annotations

import argparse
import json
import os


def field_points():
    """Three field-average operating points, mild -> harsh. The peak-shift BANDS are set from effects
    that are always present in a sim-vs-real comparison of molecular crystals and are independently
    justified, not tuned to beat the baseline:
      strain band  thermal expansion. CSD structures are routinely collected at 100-150 K; a room-
                   temperature measurement expands a molecular cell by ~0.5-1% linear (alpha ~ 1e-4/K
                   over ~150-200 K). Sampled +/- because thermal history (cold vs room-temp deposit)
                   and mount error are uncorrelated.
      zo band      sample-height displacement, the dominant real-world 2theta error; +/-0.05-0.20 deg
                   is ordinary for a hand-packed powder mount, either sign.
    Broadening / texture / noise / background are held fixed per point (good -> poor crystallinity)."""
    from xrdclip.data.augment import sweep_params
    base = lambda **kw: sweep_params(**kw)  # otherwise-clean base with fixed non-shift dials
    return {
        "mild":     {"base": base(domain_nm=150.0, march=0.90, counts=1e5, bkg_scale=0.05, hump_scale=0.03),
                     "zo_band": 0.05, "strain_band": 0.003},
        "moderate": {"base": base(domain_nm=70.0,  march=0.80, counts=2e4, bkg_scale=0.10, hump_scale=0.05),
                     "zo_band": 0.10, "strain_band": 0.006},
        "harsh":    {"base": base(domain_nm=30.0,  march=0.70, counts=5e3, bkg_scale=0.15, hump_scale=0.08),
                     "zo_band": 0.20, "strain_band": 0.010},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sticks", default="cache/pilot_sticks")
    ap.add_argument("--test_refs", default="refcodes_pilot/scaffold_test.txt")
    ap.add_argument("--families", default="refcodes/polymorph_families.json")
    ap.add_argument("--ckpt", default="runs/pilot_clip/best.pt")
    ap.add_argument("--out", default="results/combined.json")
    ap.add_argument("--fig", default="results/combined.png")
    ap.add_argument("--embed_dim", type=int, default=128)
    args = ap.parse_args()

    import torch
    from xrdclip.data.dataset import StickStore
    from xrdclip.data.polymorph import load_families, family_groups
    from xrdclip.models.clip import XRDClip
    from xrdclip.eval.polymorph_retrieval import (
        within_family_field_average, within_family_sign_decomposition)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.test_refs) as fh:
        test_refs = [ln.strip() for ln in fh if ln.strip()]
    store = StickStore(args.sticks, refcodes=set(test_refs))
    present = [rc for rc in test_refs if rc in store.records]

    families = load_families(args.families)
    groups = family_groups(present, families)
    if not groups:
        print("[combined] no polymorph families in this split — run on the full scaffold_test.")
        return 1
    n_targets = sum(len(v) for v in groups.values())
    print(f"[combined] {len(groups):,} families / {n_targets:,} targets on {device}")

    model = XRDClip(embed_dim=args.embed_dim).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[combined] loaded {args.ckpt} (epoch {ck.get('epoch', '?')})")

    points = field_points()
    field = within_family_field_average(model, store, groups, points, device)
    # sign decomposition at the 'moderate' magnitudes (band upper edges), the representative point.
    sign = within_family_sign_decomposition(
        model, store, groups, points["moderate"]["base"],
        zo=points["moderate"]["zo_band"], strain=points["moderate"]["strain_band"], device=device)

    res = {"n_families": len(groups), "n_targets": n_targets, "field": field, "sign": sign}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[combined] wrote {args.out}")

    _plot(res, args.fig)
    _summary(res)
    return 0


def _plot(res: dict, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"[combined] matplotlib unavailable ({e}); skipping figure")
        return

    field, sign = res["field"], res["sign"]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [3, 2]})
    methods = [("model", "C0"), ("correlation", "C3"), ("peak_list", "C2")]

    # Panel A: field average across operating points
    names = list(field["points"].keys())
    x = np.arange(len(names)); w = 0.25
    for i, (m, c) in enumerate(methods):
        ys = [field["points"][n][m]["top1"] for n in names]
        bars = axA.bar(x + (i - 1) * w, ys, w, color=c, label=m.replace("_", "-"))
        axA.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    axA.axhline(field["chance_top1"], color="gray", ls="--", lw=1.2,
                label=f"chance = {field['chance_top1']:.2f}")
    axA.set_xticks(x); axA.set_xticklabels([n.capitalize() for n in names])
    axA.set_ylabel("Within-family top-1"); axA.set_ylim(0, 1.03)
    axA.set_xlabel("Field-average operating point (per-crystal random shift signs)")
    axA.set_title(f"(a) Polymorph retrieval in the field ({res['n_families']} families)")
    axA.legend(loc="upper right", fontsize=8); axA.grid(axis="y", alpha=0.3)

    # Panel B: sign decomposition (mechanism)
    combos = ["shifts_cancel", "shifts_add"]
    xb = np.arange(len(combos))
    for i, (m, c) in enumerate(methods):
        ys = [sign["combos"][cb][m]["top1"] for cb in combos]
        bars = axB.bar(xb + (i - 1) * w, ys, w, color=c, label=m.replace("_", "-"))
        axB.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    axB.axhline(sign["chance_top1"], color="gray", ls="--", lw=1.2)
    axB.set_xticks(xb); axB.set_xticklabels(["shifts\ncancel", "shifts\nadd"])
    axB.set_ylim(0, 1.03)
    axB.set_xlabel(f"Displacement {sign['zo']:.2f}° vs strain {sign['strain']*100:.1f}% relative sign")
    axB.set_title("(b) Correlation's robustness is\na sign-cancellation artifact")
    axB.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[combined] wrote {path}")


def _summary(res: dict) -> None:
    field, sign = res["field"], res["sign"]
    print("\n" + "=" * 74)
    print("FIELD AVERAGE — polymorph discrimination, every effect at once, honest sign average")
    print(f"  chance (composition-only) = {field['chance_top1']:.3f}")
    for name, d in field["points"].items():
        print(f"  {name:>9}: model={d['model']['top1']:.3f}  "
              f"correlation={d['correlation']['top1']:.3f}  peak-list={d['peak_list']['top1']:.3f}")
    print("-" * 74)
    print("SIGN DECOMPOSITION — same |shift| magnitudes, only relative sign changes")
    for cb, d in sign["combos"].items():
        print(f"  {cb:>13}: model={d['model']['top1']:.3f}  "
              f"correlation={d['correlation']['top1']:.3f}  peak-list={d['peak_list']['top1']:.3f}")
    print("  (model flat across sign; correlation swings -> its robustness is coincidental cancellation)")
    print("=" * 74)


if __name__ == "__main__":
    raise SystemExit(main())
