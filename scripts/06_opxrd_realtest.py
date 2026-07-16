"""Sim-to-real test scaffold using opXRD real experimental patterns (SPEC §5, docs/DATA_SOURCES.md).

This is the credibility experiment: a model trained only on SIMULATED CSD patterns, asked to work on
REAL measured patterns. It is week-2 work and it has a genuine open piece (structure-linking, below),
so this script is a runnable scaffold, not a finished experiment. It:

  1. loads opXRD via its API (if installed),
  2. reports the organic/inorganic composition -- to validate the caveat that opXRD skews inorganic
     and that only its organic subset is a fair test for a CSD-trained (molecular) model,
  3. resamples real patterns onto our 5-50 deg / 0.02 deg grid so the encoder can consume them,
  4. embeds them with a trained pattern encoder.

THE OPEN PIECE (do not fake it): retrieval needs the real pattern's TRUE structure to be present in
the reference library. opXRD's labeled patterns carry structures that may not be CSD entries. Two
honest routes, to decide in week 2:
  (a) restrict to opXRD patterns whose compound is also a CSD refcode, and retrieve against our
      CSD library; or
  (b) add the opXRD-provided structures to the reference library and retrieve against the union.
Route (b) is cleaner (no compound-matching heuristics) and is the recommended default.

Run `pip install opxrd` first; the dataset pulls from Zenodo (record 15298026) on first use.
"""
from __future__ import annotations

import argparse

import numpy as np

from xrdclip.data import augment as A


def resample_to_grid(two_theta: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """Interpolate a real pattern onto our fixed grid, then max-normalise + standardise.

    Real patterns come on arbitrary 2theta ranges/steps and often a different wavelength. We clip to
    our window and linearly interpolate; wavelength conversion (if the source is not Cu-Kalpha1) must
    happen BEFORE this call -- flag it in week 2, it is a real source of sim-to-real error.
    """
    g = A.GRID
    y = np.interp(g, two_theta, intensity, left=0.0, right=0.0).astype(np.float32)
    m = y.max()
    if m > 0:
        y = y / m
    return A.standardize(y)


def load_opxrd(cache_dir: str):
    """Return the opXRD dataset object, or None with guidance if the library is missing."""
    try:
        import opxrd
    except ImportError:
        print("[opxrd] `opxrd` not installed. Install with:  pip install opxrd")
        print("[opxrd] docs: github.com/aimat-lab/opxrd  |  data: Zenodo 15298026")
        return None
    try:
        # API surface per the opxrd README; adjust to the installed version if it differs.
        db = opxrd.load(cache_dir=cache_dir)  # downloads on first call
        return db
    except Exception as e:
        print(f"[opxrd] load failed: {e}")
        print("[opxrd] check the installed opxrd API (opxrd.OpXRD / opxrd.load) and Zenodo access")
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/opxrd")
    ap.add_argument("--ckpt", default="runs/pilot_clip/best.pt")
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--report_only", action="store_true",
                    help="just characterise the dataset (composition, ranges); no model needed")
    args = ap.parse_args()

    db = load_opxrd(args.cache)
    if db is None:
        print("\n[opxrd] scaffold ready; rerun after `pip install opxrd`. See docstring for the")
        print("        structure-linking decision (route (b): union library) before scoring.")
        return 0

    # --- step 2: characterise, to validate the organic-subset caveat -------------------------
    patterns = getattr(db, "patterns", db)
    n = len(patterns)
    print(f"[opxrd] {n} experimental patterns loaded")
    labeled = [p for p in patterns if getattr(p, "structure", None) is not None]
    print(f"[opxrd] {len(labeled)} labeled (have an associated structure)")

    def is_organic(p) -> bool:
        s = getattr(p, "structure", None)
        if s is None:
            return False
        els = {getattr(a, "symbol", str(a)) for a in getattr(s, "species", [])}
        return "C" in els and els <= set("H C N O F S Cl Br I P".split())

    organic = [p for p in labeled if is_organic(p)]
    print(f"[opxrd] {len(organic)} labeled ORGANIC (the fair test subset for a CSD-trained model)")
    if labeled:
        print(f"[opxrd] organic fraction of labeled = {len(organic)/len(labeled):.1%} "
              f"(expect this to be small -- the caveat in DATA_SOURCES.md)")

    if args.report_only:
        return 0

    # --- step 3-4: resample + embed (retrieval scoring is the week-2 open piece) --------------
    import torch
    from xrdclip.models.clip import XRDClip
    model = XRDClip(embed_dim=args.embed_dim)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.eval()

    with torch.no_grad():
        embedded = 0
        for p in organic[:50]:
            tt = np.asarray(getattr(p, "two_theta", getattr(p, "x", [])), dtype=np.float32)
            ii = np.asarray(getattr(p, "intensity", getattr(p, "y", [])), dtype=np.float32)
            if tt.size < 10:
                continue
            y = resample_to_grid(tt, ii)
            _ = model.encode_pattern(torch.from_numpy(y)[None, None])
            embedded += 1
    print(f"[opxrd] embedded {embedded} real organic patterns; wire up route-(b) union retrieval next")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
