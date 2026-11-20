"""Independent check of the library simulation (board review 11-19, item A07): our pymatgen stick patterns, rendered as
the library renders them, against the CCDC powder simulation (Mercury's engine) of the same CSD entries.

  PYTHONPATH=src python scripts/51_sim_crosscheck.py pick                      -> cache/sim_check/refcodes.json
  PYTHONPATH=src python scripts/51_sim_crosscheck.py ours                      -> cache/sim_check/ours.npz
  <CCDC python> scripts/51_sim_crosscheck.py ccdc                              -> cache/sim_check/ccdc.npz  (licensed)
  PYTHONPATH=src python scripts/51_sim_crosscheck.py compare                   -> results/sim_crosscheck.json

Entries: the 20 ZZZUWK self-entries, every disordered self-entry of the benchmark, and 100 library entries drawn at
random (seed 0). Both patterns are put on the 5-50 deg Cu K-alpha1 grid (0.02 deg) and smoothed with the same Gaussian
(sigma 0.15 deg), which removes the difference between the two peak profiles; the statistic is the Pearson correlation.
Simulated patterns are CSD-derived and stay in the gitignored cache; only the correlations are written to results/.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

OUT = "cache/sim_check"
LAMBDA = 1.54056


def cmd_pick() -> None:
    sys.path.insert(0, "scripts")
    from importlib import import_module
    bench, hubm = import_module("19_real_bench"), import_module("21_hubness")
    shards = bench.shard_table(None)
    lib_rc = hubm.library_refcodes(shards)
    gidx = {rc: i for i, rc in enumerate(lib_rc)}
    qinfo, _, _ = bench.build_queries(f"{bench.DATA}/opxrd/opxrd_truth.json", f"{bench.DATA}/opxrd/opxrd_cnrs_organic.json", gidx)
    truth = {t["id"]: t for t in json.load(open(f"{bench.DATA}/opxrd/opxrd_truth.json"))}
    zzz, dis = set(), set()
    for q in qinfo:
        for y in truth[q["id"]]["same_form"]:
            own = y["is_anchor"] and y["is_powder_study"] and y["rmsd"] < 0.05 and y["refcode"] in gidx
            if own and q["stem"] == "ZZZUWK":
                zzz.add(y["refcode"])
            if own and y["has_disorder"]:
                dis.add(y["refcode"])
    rng = np.random.default_rng(0)
    rand = sorted(lib_rc[i] for i in rng.choice(len(lib_rc), 100, replace=False))
    os.makedirs(OUT, exist_ok=True)
    groups = {"zzzuwk": sorted(zzz), "disordered_self": sorted(dis - zzz), "random": rand}
    resim = "cache/real_extra/occ/resim.json"           # scripts/53: every added entry re-simulated with occupancies
    if os.path.exists(resim):
        groups["resimulated"] = sorted(k for k, v in json.load(open(resim)).items() if v["status"] == "patched")
    json.dump(groups, open(f"{OUT}/refcodes.json", "w"), indent=1)
    print(f"[pick] ZZZUWK {len(zzz)}, other disordered self-entries {len(dis - zzz)}, random {len(rand)}")


def cmd_ours() -> None:
    sys.path.insert(0, "scripts")
    from importlib import import_module
    bench, hubm = import_module("19_real_bench"), import_module("21_hubness")
    shards = bench.shard_table(None)
    lib_rc = hubm.library_refcodes(shards)
    gidx = {rc: i for i, rc in enumerate(lib_rc)}
    pick = json.load(open(f"{OUT}/refcodes.json"))
    rcs = sorted({rc for v in pick.values() for rc in v})
    recs = hubm.records_at(shards, [gidx[rc] for rc in rcs], ("two_theta", "intensity", "hkl", "d_hkl", "recip"))
    np.savez_compressed(f"{OUT}/ours.npz", refcode=np.array(rcs), y=bench.render(recs))
    print(f"[ours] {len(rcs)} patterns")


def cmd_ccdc() -> None:
    try:
        from ccdc.io import EntryReader
        from ccdc.descriptors import CrystalDescriptors
        P = CrystalDescriptors.PowderPattern
        s = P.Settings()
        s.two_theta_minimum, s.two_theta_maximum, s.two_theta_step = 5.0, 50.0, 0.02
        s.full_width_at_half_maximum = 0.1
        s.wavelength = LAMBDA
        s.second_wavelength = None
        s.include_hydrogens = True
        pick = json.load(open(f"{OUT}/refcodes.json"))
        rcs = sorted({rc for v in pick.values() for rc in v})
        reader = EntryReader("CSD")
        grid = np.round(np.arange(5.0, 50.0 + 0.01, 0.02), 4)
        ys, ok = [], []
        for rc in rcs:
            try:
                p = P.from_crystal(reader.entry(rc).crystal, s)
                x, y = np.asarray(p.two_theta, float), np.asarray(p.intensity, float)
                ys.append(np.interp(grid, x, y))
                ok.append(rc)
            except Exception as ex:
                print(f"[ccdc] {rc} failed: {ex}", flush=True)
        np.savez_compressed(f"{OUT}/ccdc.npz", refcode=np.array(ok), y=np.stack(ys).astype(np.float32))
        print(f"[ccdc] {len(ok)} of {len(rcs)} simulated", flush=True)
    finally:
        os._exit(0)          # the CCDC interpreter otherwise hangs at exit


def smooth(y: np.ndarray, sigma_deg: float = 0.15, step: float = 0.02) -> np.ndarray:
    k = np.arange(-int(4 * sigma_deg / step), int(4 * sigma_deg / step) + 1) * step
    w = np.exp(-0.5 * (k / sigma_deg) ** 2)
    return np.convolve(y, w / w.sum(), mode="same")


def cmd_compare() -> None:
    a, b = np.load(f"{OUT}/ours.npz"), np.load(f"{OUT}/ccdc.npz")
    ya = {str(r): y for r, y in zip(a["refcode"], a["y"])}
    yb = {str(r): y for r, y in zip(b["refcode"], b["y"])}
    pick = json.load(open(f"{OUT}/refcodes.json"))
    out = {"sigma_deg": 0.15, "groups": {}, "per_entry": {}}
    for g, rcs in pick.items():
        r = []
        for rc in rcs:
            if rc in ya and rc in yb:
                c = float(np.corrcoef(smooth(ya[rc]), smooth(yb[rc]))[0, 1])
                out["per_entry"][rc] = round(c, 4)
                r.append(c)
        r = np.array(r)
        out["groups"][g] = {"n": int(len(r)), "n_requested": len(rcs), "median": round(float(np.median(r)), 4),
                            "min": round(float(r.min()), 4), "frac_ge_0.95": round(float((r >= 0.95).mean()), 4),
                            "frac_ge_0.90": round(float((r >= 0.90).mean()), 4)}
    json.dump(out, open("results/sim_crosscheck.json", "w"), indent=1)
    for g, v in out["groups"].items():
        print(g, v)
    worst = sorted(out["per_entry"].items(), key=lambda kv: kv[1])[:10]
    print("lowest:", worst)


if __name__ == "__main__":
    {"pick": cmd_pick, "ours": cmd_ours, "ccdc": cmd_ccdc, "compare": cmd_compare}[sys.argv[1]]()
