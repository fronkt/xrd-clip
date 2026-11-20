"""Label sensitivity of the headline (resubmission D2; RSC Advances board and referee 1).

From the corrected-label ranks of the main rescore (cache/real_bench/rescore_main/ranks.npz, scripts/28 eval --tag main):
  strict_only        patterns anchored in the lenient packing pass (bond types, H counts and smallest components
                     ignored) removed: can a solvate/salt mix-up drive the result?
  known_wavelength   the 3 patterns whose wavelength opXRD does not record removed (Cu assumed; scripts/41 fits Cu)
  ambient_sc         single-crystal set without the variable-pressure series (results/sc_conditions.json)
and the list of lenient-anchor patterns with their anchors, for the ESI.

    PYTHONPATH=src python scripts/43_label_sensitivity.py     -> results/label_sensitivity.json
    PYTHONPATH=src python scripts/43_label_sensitivity.py --ranks cache/real_bench/rescore_main/ranks_d3_sym_s0.npz \n        --out results/d3/label_sensitivity_d3_sym_s0.json           (a retrained model's ranks, from 28 eval --ranks_out)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from importlib import import_module

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
bench = import_module("19_real_bench")
hubm = import_module("21_hubness")
pn = import_module("22_paper_numbers")

KEYS = ("pearson_bkg", "cos_zs_bkg", "wcc_1_bkg", "wcc_2_bkg", "lnq_bkg", "model_bkg")
UNKNOWN_WL = {"pattern_144.json", "pattern_758.json", "pattern_917.json"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", default="cache/real_bench/rescore_main/ranks.npz")
    ap.add_argument("--out", default="results/label_sensitivity.json")
    args = ap.parse_args()
    shards = bench.shard_table(None)
    gidx = {rc: i for i, rc in enumerate(hubm.library_refcodes(shards))}
    qinfo, _, _ = bench.build_queries(f"{bench.DATA}/opxrd/opxrd_truth.json",
                                      f"{bench.DATA}/opxrd/opxrd_cnrs_organic.json", gidx)
    truth = {t["id"]: t for t in json.load(open(f"{bench.DATA}/opxrd/opxrd_truth.json"))}
    sc_cond = json.load(open("results/sc_conditions.json"))
    z = np.load(args.ranks)
    ids = np.array([q["id"] for q in qinfo])
    stems = np.array([q["compound"] for q in qinfo])
    lenient = np.array([truth[i].get("anchor_tier") == "lenient" for i in ids])
    pressure = np.array([bool(sc_cond.get(i.replace(".json", ""), {}).get("pressure_series")) for i in ids])
    subsets = {"all": np.ones(len(ids), bool), "strict_only": ~lenient,
               "known_wavelength": ~np.isin(ids, list(UNKNOWN_WL)), "ambient": ~pressure}
    out = {"n_lenient_patterns": int(lenient.sum()), "n_lenient_compounds": int(len(set(stems[lenient]))),
           "n_pressure_patterns": int(pressure.sum()), "subsets": {}}
    for name, sel in subsets.items():
        st = stems[sel]
        cboot = np.random.default_rng(0).integers(0, len(np.unique(st)), (2000, len(np.unique(st))))
        res = {}
        for td in ("std", "sc"):
            for k in KEYS:
                r = z[f"{td}|{k}"][sel]
                m = pn.macro(r, 1, st, cboot)
                res[f"{td}|{k}"] = {"top1": round(m["value"], 4), "ci": [round(c, 4) for c in m["ci"]],
                                    "n_compounds": m["n_compounds"], "n_patterns": int((r >= 0).sum())}
        out["subsets"][name] = res
        print(name, {k: (v["top1"], v["n_compounds"]) for k, v in res.items()}, flush=True)
    out["lenient_patterns"] = [{"id": i, "stem": s, "anchors": sorted(truth[i]["anchors"])}
                               for i, s, l in zip(ids, stems, lenient) if l]
    out["ranks"] = args.ranks
    json.dump(out, open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
