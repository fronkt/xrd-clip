"""Wavelength of the 3 opXRD patterns that record none (referee 1, RSC Advances; resubmission D2).

The opXRD record holds each pattern's own refined structure (lattice + full atom list). For a trial wavelength the
measured axis is converted to Cu K-alpha1 through d (as in the benchmark) and correlated (Pearson, background-removed,
5-50 deg Cu grid) with the pattern simulated from that structure; a zero-offset scan of +-0.1 deg absorbs goniometer
error. The fitted wavelength maximises the correlation over a coarse-to-fine scan of 0.3-2.5 A.

Validation first: the same fit on patterns whose wavelength IS recorded (lab Cu, Co, Mo and synchrotron) must recover
it. Then the three unknowns.

    PYTHONPATH=src python scripts/41_wavelength_fit.py      -> results/wavelength_fit.json
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from xrdclip.data import augment as A                                  # noqa: E402
from xrdclip.eval.real_patterns import preprocess, to_cu_grid           # noqa: E402

DATA = "C:/Users/frank/xrd-clip-data/opxrd"
UNKNOWN = ("pattern_144.json", "pattern_758.json", "pattern_917.json")


def simulate(o: dict) -> np.ndarray:
    from pymatgen.core import Lattice, Structure
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    a, b, c, al, be, ga = eval(o["lattice"])                            # "(a, b, c, alpha, beta, gamma)"
    sp = [x["symbol"].rstrip("0+-") for x in o["basis"]]
    fr = [[x["x"], x["y"], x["z"]] for x in o["basis"]]
    occ = [x.get("occupancy", 1) for x in o["basis"]]
    s = Structure(Lattice.from_parameters(a, b, c, al, be, ga), [{e: q} for e, q in zip(sp, occ)], fr)
    p = XRDCalculator(wavelength=A.WAVELENGTH).get_pattern(s, two_theta_range=(A.TWO_THETA_MIN, A.TWO_THETA_MAX))
    hkl = np.zeros((len(p.x), 3), np.int16)
    rec = {"two_theta": np.asarray(p.x, np.float32), "intensity": np.asarray(p.y, np.float32), "hkl": hkl,
           "d_hkl": np.asarray(p.d_hkls, np.float32), "recip": np.eye(3, dtype=np.float32)}
    return A.render_sticks(rec, A.CLEAN)


def score(o: dict, ref: np.ndarray, lam: float, zero: float = 0.0) -> float:
    y, m = to_cu_grid(np.asarray(o["two_theta"]) + zero, np.asarray(o["intensity"]), lam)
    if m.sum() < 200:
        return -1.0
    q = preprocess(y, m, bkg=True)
    r = ref * m
    q, r = q[m] - q[m].mean(), r[m] - r[m].mean()
    den = np.sqrt((q * q).sum() * (r * r).sum())
    return float((q * r).sum() / den) if den > 0 else -1.0


def fit(o: dict, ref: np.ndarray) -> dict:
    zeros = np.arange(-0.1, 0.1001, 0.02)
    grid = np.exp(np.arange(np.log(0.3), np.log(2.5), 2e-3))              # 0.2 % steps
    best = max(((score(o, ref, l), l) for l in grid))
    for width, step in ((4e-3, 2e-4), (4e-4, 2e-5)):                       # refine, with the zero scan
        lams = best[1] * np.exp(np.arange(-width, width + step / 2, step))
        best = max(((score(o, ref, l, z), l, z) for l in lams for z in zeros), key=lambda t: t[0])
    s_sorted = sorted((score(o, ref, l), l) for l in grid)[::-1]
    runner = next((s for s, l in s_sorted if abs(np.log(l / best[1])) > 0.02), None)
    return {"lambda": round(float(best[1]), 5), "zero": round(float(best[2]), 3), "pearson": round(best[0], 3),
            "runner_up_pearson_other_lambda": None if runner is None else round(runner, 3)}


def main() -> int:
    pats = {o["id"]: o for o in json.load(open(f"{DATA}/opxrd_cnrs_organic.json"))["organic"]}
    truth = {t["id"]: t for t in json.load(open(f"{DATA}/opxrd_truth.json"))}
    rng = np.random.default_rng(0)
    known = [i for i, t in truth.items() if t.get("true_stems") and t.get("wavelength") and i in pats]
    by_kind = {}
    for i in known:
        lam = float(truth[i]["wavelength"])
        kind = "Cu" if abs(lam - 1.5406) < 0.01 else ("Co" if abs(lam - 1.789) < 0.01 else
                                                      ("Mo" if abs(lam - 0.7107) < 0.01 else "other"))
        by_kind.setdefault(kind, []).append(i)
    val_ids = [x for k, v in sorted(by_kind.items()) for x in rng.permutation(v)[: (8 if k in ("Cu", "other") else 3)]]
    out = {"validation": [], "unknown": []}
    for i in val_ids:
        o = pats[i]
        f = fit(o, simulate(o))
        f.update(id=i, recorded=float(truth[i]["wavelength"]),
                 rel_error=round(f["lambda"] / float(truth[i]["wavelength"]) - 1, 5))
        out["validation"].append(f)
        print("val", f, flush=True)
    for i in UNKNOWN:
        o = pats[i]
        f = fit(o, simulate(o))
        f.update(id=i, score_at_cu=round(score(o, simulate(o), A.WAVELENGTH), 3))
        out["unknown"].append(f)
        print("unknown", f, flush=True)
    errs = np.abs([v["rel_error"] for v in out["validation"]])
    out["validation_summary"] = {"n": int(len(errs)), "max_abs_rel_error": float(errs.max()),
                                 "median_abs_rel_error": float(np.median(errs))}
    json.dump(out, open("results/wavelength_fit.json", "w"), indent=1)
    print(json.dumps(out["validation_summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
