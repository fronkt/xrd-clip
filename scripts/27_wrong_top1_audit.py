"""Audit of the ground truth (referee 10-12): is a "wrong" top-1 entry really a different solid form?

Same-form sets were built within the anchor's CSD refcode family. A same-form entry deposited under another
refcode, outside the reduced-cell window, would be scored as an error. This script collects every wrong top-1
entry of the classical scores (background-subtracted queries, both answer sets) and compares it with the
pattern's anchor by CCDC packing similarity, with the strict settings of scripts/18_.

  PYTHONPATH=src python scripts/27_wrong_top1_audit.py pairs      -> cache/real_bench/wrong_top1_pairs.json
  <ccdc python> scripts/27_wrong_top1_audit.py compare            -> cache/real_bench/wrong_top1_packing.json
  PYTHONPATH=src python scripts/27_wrong_top1_audit.py report     -> results/wrong_top1_audit.json
"""
from __future__ import annotations

import json
import sys

PAIRS = "cache/real_bench/wrong_top1_pairs.json"
PACK = "cache/real_bench/wrong_top1_packing.json"
KEYS = ["pearson_bkg", "cos_zs_bkg", "wcc_1_bkg", "wcc_2_bkg", "lnq_bkg"]
DATA = "C:/Users/frank/xrd-clip-data"


def pairs() -> int:
    from importlib import import_module
    import numpy as np
    sys.path.insert(0, "scripts")
    bench = import_module("19_real_bench")
    hubm = import_module("21_hubness")
    shards = bench.shard_table(None)
    lib_rc = hubm.library_refcodes(shards)
    gidx = {rc: i for i, rc in enumerate(lib_rc)}
    qinfo, _, _ = bench.build_queries(f"{DATA}/opxrd/opxrd_truth.json", f"{DATA}/opxrd/opxrd_cnrs_organic.json", gidx)
    truth = {t["id"]: t for t in json.load(open(f"{DATA}/opxrd/opxrd_truth.json"))}
    out = []
    for key in KEYS:
        S = np.load(f"cache/real_bench/full/S_{key}.npy", mmap_mode="r")
        for qi, q in enumerate(qinfo):
            row = np.asarray(S[qi], dtype=np.float32)
            for td in ("std", "sc"):
                tr, ig = q["sets"][td]
                if not tr:
                    continue
                excl = np.array(sorted({gidx[rc] for rc in tr + ig}), dtype=np.int64)
                best_true = row[[gidx[rc] for rc in tr]].max()
                masked = row.copy()
                masked[excl] = -np.inf
                w = int(np.argmax(masked))
                if masked[w] > best_true:
                    out.append({"id": q["id"], "stem": q["stem"], "key": key, "td": td, "wrong": lib_rc[w],
                                "anchors": sorted(truth[q["id"]]["anchors"])})
    json.dump(out, open(PAIRS, "w"), indent=0)
    print(f"{len(out)} wrong top-1 cases, {len({(a, p['wrong']) for p in out for a in p['anchors']})} anchor pairs")
    return 0


def compare() -> int:
    from ccdc.crystal import PackingSimilarity
    from ccdc.io import EntryReader
    reader = EntryReader("CSD")
    ps = PackingSimilarity()
    ps.settings.packing_shell_size = 15
    ps.settings.ignore_hydrogen_positions = True
    ps.settings.allow_molecular_differences = False
    ps.settings.timeout_ms = 30000
    todo = sorted({(a, p["wrong"]) for p in json.load(open(PAIRS)) for a in p["anchors"]})
    res = {}
    for i, (a, w) in enumerate(todo):
        try:
            r = ps.compare(reader.crystal(a), reader.crystal(w))
            res[f"{a}|{w}"] = None if r is None else [int(r.nmatched_molecules), float(r.rmsd)]
        except Exception as ex:                       # noqa: BLE001 - record and move on
            res[f"{a}|{w}"] = f"error: {type(ex).__name__}"
        if i % 25 == 0:
            print(f"{i}/{len(todo)}", flush=True)
    json.dump(res, open(PACK, "w"), indent=0)
    print(f"compared {len(todo)} pairs")
    return 0


def report() -> int:
    from collections import Counter
    P, C = json.load(open(PAIRS)), json.load(open(PACK))
    same = []
    for p in P:
        hits = [C.get(f"{a}|{p['wrong']}") for a in p["anchors"]]
        p["max_matched"] = max((h[0] for h in hits if isinstance(h, list)), default=None)
        if p["max_matched"] == 15:
            same.append(p)
    res = {"n_cases": len(P), "n_pairs": len(C),
           "matched_distribution": dict(Counter(str(p["max_matched"]) for p in P)),
           "same_form_by_key_td": dict(Counter(f"{p['key']}|{p['td']}" for p in same)),
           "same_form_cases": [{k: p[k] for k in ("id", "stem", "key", "td", "wrong")} for p in same]}
    json.dump(res, open("results/wrong_top1_audit.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "same_form_cases"}, indent=1))
    print("same-form wrong top-1 (distinct):", sorted({(p["stem"], p["wrong"]) for p in same}))
    return 0


if __name__ == "__main__":
    sys.exit({"pairs": pairs, "compare": compare, "report": report}[sys.argv[1]]())
