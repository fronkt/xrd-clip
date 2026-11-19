"""Symmetry operators for every cached stick pattern (resubmission D1: the structure encoder never saw the space group).

For each stick shard (cache/sticks/*.npz, cache/sticks_extra/*.npz) write a twin file under <stick dir>_symops/ with the
same refcode order and one (n_ops, 3, 4) operator array per entry, parsed from that entry's own CIF
(src/xrdclip/data/symops.py). Main interpreter; reads only the local CIF cache (no CSD API).

Check mode (--check N): for N random entries, expand the stored asymmetric unit by the operators and compare with the
full cell pymatgen parses from the CIF.

    PYTHONPATH=src python scripts/40_symops.py
    PYTHONPATH=src python scripts/40_symops.py --check 500

The open COD replica (R5) keeps no stage-1 CIF shards locally, so --cod_cif_root reads each entry's CIF from COD's
rsync layout (scripts/31 cod_path, the files 31 `shards` read) and writes the twins of --cod_sticks:

    PYTHONPATH=src python scripts/40_symops.py --cod_cif_root <dir with 1/00/00/1000006.cif ...> [--check 500]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
from importlib import import_module

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from xrdclip.data.symops import expand, parse_symops  # noqa: E402

PAIRS = (("cache/csd", "cache/sticks"), ("cache/real_extra", "cache/sticks_extra"))


class CodCifs:
    """refcode (COD id) -> CIF text, read on demand from COD's rsync layout, decoded as 31 `shards` decoded it."""

    def __init__(self, root: str):
        self.root, self.path = root, import_module("31_cod_library").cod_path

    def _p(self, rc: str) -> str:
        return os.path.join(self.root, self.path(rc))

    def __contains__(self, rc: str) -> bool:
        return os.path.exists(self._p(rc))

    def __getitem__(self, rc: str) -> str:
        with open(self._p(rc), encoding="utf-8", errors="replace") as fh:
            return fh.read()


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
    ap.add_argument("--cod_cif_root", default="", help="COD replica: CIFs from this rsync-layout directory")
    ap.add_argument("--cod_sticks", default="cod/sticks", help="COD replica: the stick shards to write twins for")
    a = ap.parse_args()
    if a.cod_cif_root:
        cifs, pairs = CodCifs(a.cod_cif_root), ((None, a.cod_sticks),)
    else:
        cifs = load_cifs()
        print(f"{len(cifs)} CIFs", flush=True)
        pairs = PAIRS[1:] if a.check and os.environ.get("SYMOPS_CHECK_EXTRA_ONLY") else PAIRS
    rng = np.random.default_rng(0)
    for _, stick_dir in pairs:
        out_dir = stick_dir + "_symops"          # dataset.symops_twin
        os.makedirs(out_dir, exist_ok=True)
        for path in sorted(glob.glob(os.path.join(stick_dir, "*.npz"))):
            with np.load(path, allow_pickle=True) as z:
                refs = z["refcode"]
                frac = z["atom_frac"] if a.check else None
            missing = [str(r) for r in refs if str(r) not in cifs]
            if missing:
                raise SystemExit(f"{path}: {len(missing)} refcodes have no CIF, e.g. {missing[:10]}")
            ops = [parse_symops(cifs[str(r)]) for r in refs]
            if a.check:
                from pymatgen.core import Structure
                idx = rng.choice(len(refs), size=min(a.check, len(refs)), replace=False)
                bad = []
                bad_pos = []
                for i in idx:
                    cif = cifs[str(refs[i])]
                    try:
                        full = Structure.from_str(cif, fmt="cif")
                    except Exception:   # Z' < 1 entries: the same lenient parse simulate.py falls back to
                        from pymatgen.io.cif import CifParser
                        full = CifParser.from_str(cif, occupancy_tolerance=8.01).parse_structures(primitive=False)[0]
                    ex = expand(frac[i], ops[i])
                    if len(ex) != len(full):
                        bad.append(str(refs[i]))
                        continue
                    # every regenerated site must be one of pymatgen's sites (same frame, periodic distance)
                    d = np.abs(ex[:, None, :] - (np.asarray(full.frac_coords) % 1.0)[None])
                    if (np.minimum(d, 1 - d).max(axis=2).min(axis=1) > 2e-3).any():
                        bad_pos.append(str(refs[i]))
                print(f"{path}: check {len(idx)}, count mismatches {len(bad)} {bad[:10]}, "
                      f"position mismatches {len(bad_pos)} {bad_pos[:10]}", flush=True)
                continue
            out = os.path.join(out_dir, os.path.basename(path))
            np.savez_compressed(out, refcode=refs, ops=np.array(ops, dtype=object), allow_pickle=True)
            n = np.array([len(o) for o in ops])
            no_loop = sum(not any(t in cifs[str(r)].lower() for t in ("_symmetry_equiv_pos_as_xyz",
                                                                       "_space_group_symop_operation_xyz")) for r in refs)
            print(f"{out}: {len(refs)} entries, n_ops min {n.min()} median {int(np.median(n))} max {n.max()}; "
                  f"{no_loop} without an operator loop (operators from the space-group symbol)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
