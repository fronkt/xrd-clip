"""Symmetry operators for every cached stick pattern (resubmission D1: the structure encoder never saw the space group).

For each stick shard (cache/sticks/*.npz, cache/sticks_extra/*.npz) write a twin file under <stick dir>_symops/ with the
same refcode order and one (n_ops, 3, 4) operator array per entry, parsed from that entry's own CIF
(src/xrdclip/data/symops.py). Main interpreter; reads only the local CIF cache (no CSD API).

Check mode (--check N): for N random entries, expand the stored asymmetric unit by the operators and compare with the
full cell pymatgen parses from the CIF.

    PYTHONPATH=src python scripts/40_symops.py
    PYTHONPATH=src python scripts/40_symops.py --check 500
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from xrdclip.data.symops import expand, parse_symops  # noqa: E402

PAIRS = (("cache/csd", "cache/sticks"), ("cache/real_extra", "cache/sticks_extra"))


def load_cifs() -> dict:
    cifs = {}
    for cif_dir, _ in PAIRS:
        for f in sorted(glob.glob(os.path.join(cif_dir, "*.jsonl.gz"))):
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    cifs[r["refcode"]] = r["cif"]
    return cifs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=int, default=0)
    a = ap.parse_args()
    cifs = load_cifs()
    print(f"{len(cifs)} CIFs", flush=True)
    rng = np.random.default_rng(0)
    pairs = PAIRS[1:] if a.check and os.environ.get("SYMOPS_CHECK_EXTRA_ONLY") else PAIRS
    for _, stick_dir in pairs:
        out_dir = stick_dir + "_symops"          # dataset.symops_twin
        os.makedirs(out_dir, exist_ok=True)
        for path in sorted(glob.glob(os.path.join(stick_dir, "*.npz"))):
            with np.load(path, allow_pickle=True) as z:
                refs = z["refcode"]
                frac = z["atom_frac"] if a.check else None
            ops = [parse_symops(cifs[str(r)]) for r in refs]
            missing = sum(str(r) not in cifs for r in refs)
            if missing:
                raise SystemExit(f"{path}: {missing} refcodes have no CIF")
            if a.check:
                from pymatgen.core import Structure
                idx = rng.choice(len(refs), size=min(a.check, len(refs)), replace=False)
                bad = []
                for i in idx:
                    cif = cifs[str(refs[i])]
                    try:
                        n_full = len(Structure.from_str(cif, fmt="cif"))
                    except Exception:   # Z' < 1 entries: the same lenient parse simulate.py falls back to
                        from pymatgen.io.cif import CifParser
                        n_full = len(CifParser.from_str(cif, occupancy_tolerance=8.01).parse_structures(
                            primitive=False)[0])
                    if len(expand(frac[i], ops[i])) != n_full:
                        bad.append(str(refs[i]))
                print(f"{path}: check {len(idx)}, mismatches {len(bad)} {bad[:10]}", flush=True)
                continue
            out = os.path.join(out_dir, os.path.basename(path))
            np.savez_compressed(out, refcode=refs, ops=np.array(ops, dtype=object), allow_pickle=True)
            n = np.array([len(o) for o in ops])
            print(f"{out}: {len(refs)} entries, n_ops min {n.min()} median {int(np.median(n))} max {n.max()}",
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
