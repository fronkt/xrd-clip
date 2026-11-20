"""Re-simulate the stick patterns of the disordered added library entries with their site occupancies (board review
11-19, A07; CIFs from scripts/52).

  PYTHONPATH=src python scripts/53_resim_disorder.py        -> cache/sticks_extra/shard_extra.npz (patched in place;
                                                               the original is kept as shard_extra.npz.pre_occ)

Only the stick fields (two_theta, intensity, hkl, d_hkl, recip) of entries with partial occupancies change. The atom
lists that the learned model encodes (atom_z, atom_frac) are left as they were, because the structure encoder has no
occupancy input, so the model's library embeddings and ranks do not change. Writes cache/real_extra/occ/resim.json with
the entries patched and the number of sticks before and after.
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np

SHARD = "cache/sticks_extra/shard_extra.npz"
OCC = "cache/real_extra/occ"
FIELDS = ("two_theta", "intensity", "hkl", "d_hkl", "recip")


def main() -> int:
    from xrdclip.data.simulate import simulate_one
    index = json.load(open(f"{OCC}/index.json"))
    todo = sorted(k for k, v in index.items() if v.get("has_disorder") and v.get("n_partial", 0) > 0)
    if not os.path.exists(SHARD + ".pre_occ"):
        shutil.copy2(SHARD, SHARD + ".pre_occ")
    with np.load(SHARD + ".pre_occ", allow_pickle=True) as z:
        cols = {f: z[f].copy() for f in z.files}
    rc = [str(x) for x in cols["refcode"]]
    report = {}
    for r in todo:
        if r not in rc:
            report[r] = {"status": "not in the library (no coordinates)"}
            continue
        i = rc.index(r)
        sim = simulate_one({"refcode": r, "cif": open(f"{OCC}/{r}.cif").read(), "smiles": cols["smiles"][i],
                            "cell": cols["cell"][i]})
        if sim is None:
            report[r] = {"status": "simulation failed; kept"}
            continue
        report[r] = {"status": "patched", "n_sticks_before": int(len(cols["two_theta"][i])),
                     "n_sticks_after": int(len(sim["two_theta"]))}
        for f in FIELDS:
            cols[f][i] = np.asarray(sim[f], dtype=np.asarray(cols[f][i]).dtype)
    np.savez_compressed(SHARD, **cols)
    json.dump(report, open(f"{OCC}/resim.json", "w"), indent=1)
    n = sum(v["status"] == "patched" for v in report.values())
    print(f"[53] {len(todo)} entries with partial occupancies; {n} patched in {SHARD}; "
          f"{sum('failed' in v['status'] for v in report.values())} failed; "
          f"{sum('not in the library' in v['status'] for v in report.values())} not in the library")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
