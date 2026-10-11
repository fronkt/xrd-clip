"""Pack the stick cache down to what a rented box needs to RENDER patterns (scripts/19_... --box_ctx).

Keeps refcode + two_theta, intensity, hkl, d_hkl, recip, in the original row order, so global
library indices match the local run. Drops atom coordinates (atom_z, atom_frac), SMILES, cell and
everything else: no structure leaves the machine, only simulated peak lists.

  python scripts/box/pack_render_sticks.py <out_dir>
  -> <out_dir>/cache/sticks/shard_*.npz, <out_dir>/cache/sticks_extra/shard_extra.npz, MANIFEST.txt
"""
from __future__ import annotations

import glob
import hashlib
import os
import sys

import numpy as np

FIELDS = ("refcode", "two_theta", "intensity", "hkl", "d_hkl", "recip")


def main() -> int:
    out = sys.argv[1]
    lines = []
    for d in ("cache/sticks", "cache/sticks_extra"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
        for p in sorted(glob.glob(os.path.join(d, "*.npz"))):
            with np.load(p, allow_pickle=True) as z:
                arrays = {f: z[f] for f in FIELDS}
            dst = os.path.join(out, d, os.path.basename(p))
            np.savez_compressed(dst, allow_pickle=True, **arrays)
            md5 = hashlib.md5(open(dst, "rb").read()).hexdigest()
            lines.append(f"{md5}  {d}/{os.path.basename(p)}  n={len(arrays['refcode'])}")
            print(lines[-1], f"{os.path.getsize(dst) / 1e6:.0f} MB", flush=True)
    with open(os.path.join(out, "MANIFEST.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
