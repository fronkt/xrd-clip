"""STAGE 2 — stick-pattern simulation.  Runs under the MAIN interpreter (pymatgen). No CSD licence.

Converts each extracted CIF into a *stick pattern*: the list of Bragg reflections with their
2theta positions, integrated intensities, Miller indices, and d-spacings — plus the reciprocal
lattice matrix.

WHY STICKS AND NOT FINISHED DIFFRACTOGRAMS
------------------------------------------
The expensive part of PXRD simulation is the structure factor calculation, and it depends only
on the structure. Everything that makes a pattern look *measured* — peak profile, broadening,
preferred orientation, background, noise, zero-offset — is a cheap transformation applied to the
sticks. So we compute the sticks once, cache them, and apply fresh degradation on the fly every
epoch in the dataloader (see augment.py).

That split is what makes strong augmentation affordable, and strong augmentation is the entire
source of the model's robustness. It is the most important architectural decision in the pipeline.

We keep `hkl` and the reciprocal lattice because two physical effects need them:
  * March-Dollase preferred orientation needs the angle between each reflection's plane normal
    and the texture axis.
  * Anisotropic/isotropic strain shifts peaks via d-spacing, which we recompute analytically
    rather than re-running the (expensive) structure-factor calculation.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import time
import warnings
from multiprocessing import Pool

import numpy as np

# pymatgen is noisy about partial occupancies / CIF quirks; we handle failures explicitly.
warnings.filterwarnings("ignore")

WAVELENGTH = 1.54056     # Cu K-alpha1, angstrom
TWO_THETA_MIN = 5.0
TWO_THETA_MAX = 50.0
MIN_INTENSITY = 1e-4     # drop reflections below this fraction of I_max (they cannot matter)


def simulate_one(rec: dict) -> "dict | None":
    """CIF -> stick pattern. Returns None on any parse/simulation failure."""
    from pymatgen.core import Structure
    from pymatgen.analysis.diffraction.xrd import XRDCalculator

    try:
        struct = Structure.from_str(rec["cif"], fmt="cif")
    except Exception:
        return None

    try:
        calc = XRDCalculator(wavelength=WAVELENGTH)
        pat = calc.get_pattern(struct, two_theta_range=(TWO_THETA_MIN, TWO_THETA_MAX))
    except Exception:
        return None

    two_theta = np.asarray(pat.x, dtype=np.float32)
    inten = np.asarray(pat.y, dtype=np.float32)
    if two_theta.size == 0:
        return None

    # pymatgen groups symmetry-equivalent reflections; take the first hkl of each group as the
    # representative plane normal (equivalents share |d| and, for our purposes, orientation stats).
    #
    # Hexagonal and trigonal cells come back in 4-index Miller-Bravais notation (h k i l), where
    # i = -(h+k) is redundant. Everything downstream (March-Dollase in particular) does hkl @ B
    # against a 3x3 reciprocal matrix, so a stray 4th index is a shape crash at best and a silently
    # wrong plane normal at worst. Normalise to 3 indices here, at the only place that sees raw
    # pymatgen output.
    raw_hkls = [g[0]["hkl"] for g in pat.hkls]
    hkls = np.array(
        [(h[0], h[1], h[3]) if len(h) == 4 else tuple(h) for h in raw_hkls],
        dtype=np.int16,
    )
    d_hkl = np.asarray(pat.d_hkls, dtype=np.float32)

    # Normalise intensities to max 100 (pymatgen already does, but be explicit) and prune.
    if inten.max() <= 0:
        return None
    inten = inten / inten.max() * 100.0
    keep = inten >= MIN_INTENSITY * 100.0
    two_theta, inten, hkls, d_hkl = two_theta[keep], inten[keep], hkls[keep], d_hkl[keep]
    if two_theta.size == 0:
        return None

    # Reciprocal lattice (rows = b1,b2,b3) so a plane normal is g = hkl @ B.
    B = np.asarray(struct.lattice.reciprocal_lattice_crystallographic.matrix, dtype=np.float32)

    # Atom features for the structure encoder. The diffraction pattern is a function of BOTH the
    # lattice (peak positions) and the atomic arrangement (intensities), so the structure encoder
    # needs the atoms, not just the cell.
    #
    # We feed the ASYMMETRIC UNIT, not the full unit cell. The full cell is all Z symmetry copies of
    # the asymmetric unit -- redundant given that the space group + lattice (both provided) generate
    # the copies -- and feeding it makes the structure encoder's O(atoms^2) attention explode
    # (a Z=8, 50-atom-molecule crystal is 400 atoms; outliers exceed 1000, blowing past 15 GB and
    # thrashing swap). The asymmetric unit is the compact, non-redundant, correct representation:
    # for a molecular crystal it is just the Z' molecules, typically tens of atoms.
    def _asym_atoms(structure):
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        try:
            sym = SpacegroupAnalyzer(structure, symprec=0.01).get_symmetrized_structure()
            reps = [grp[0] for grp in sym.equivalent_indices]  # one site per orbit
            zz = np.array([structure[i].specie.Z if structure[i].is_ordered
                           else structure[i].species.elements[0].Z for i in reps], dtype=np.int16)
            ff = np.asarray([structure[i].frac_coords for i in reps], dtype=np.float32) % 1.0
            return zz, ff
        except Exception:
            return None

    asym = _asym_atoms(struct)
    if asym is not None and len(asym[0]) > 0:
        z, frac = asym
    else:
        # Fallback: full cell, but hard-capped so a symmetry-analysis failure can never OOM training.
        z = np.array([site.specie.Z if site.is_ordered else site.species.elements[0].Z
                      for site in struct], dtype=np.int16)[:256]
        frac = (np.asarray(struct.frac_coords, dtype=np.float32) % 1.0)[:256]

    return {
        "refcode": rec["refcode"],
        "smiles": rec["smiles"],
        "two_theta": two_theta,
        "intensity": inten,
        "hkl": hkls,
        "d_hkl": d_hkl,
        "recip": B,
        "cell": np.asarray(rec["cell"], dtype=np.float32),
        "spacegroup": rec.get("spacegroup", ""),
        "n_atoms": rec.get("n_atoms", 0),
        "atom_z": z,
        "atom_frac": frac,
    }


def _process_shard(args: "tuple[str, str]") -> "tuple[str, int, int]":
    in_path, out_dir = args
    name = os.path.basename(in_path).replace(".jsonl.gz", "")
    out_path = os.path.join(out_dir, f"{name}.npz")

    records, n_in, n_fail = [], 0, 0
    with gzip.open(in_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            n_in += 1
            try:
                rec = json.loads(line)
            except Exception:
                n_fail += 1
                continue
            sim = simulate_one(rec)
            if sim is None:
                n_fail += 1
                continue
            records.append(sim)

    if not records:
        return name, n_in, n_fail

    # Ragged arrays (each structure has a different reflection count) -> object arrays.
    np.savez_compressed(
        out_path,
        refcode=np.array([r["refcode"] for r in records]),
        smiles=np.array([r["smiles"] for r in records]),
        spacegroup=np.array([r["spacegroup"] for r in records]),
        n_atoms=np.array([r["n_atoms"] for r in records], dtype=np.int32),
        cell=np.stack([r["cell"] for r in records]),
        recip=np.stack([r["recip"] for r in records]),
        two_theta=np.array([r["two_theta"] for r in records], dtype=object),
        intensity=np.array([r["intensity"] for r in records], dtype=object),
        hkl=np.array([r["hkl"] for r in records], dtype=object),
        d_hkl=np.array([r["d_hkl"] for r in records], dtype=object),
        atom_z=np.array([r["atom_z"] for r in records], dtype=object),
        atom_frac=np.array([r["atom_frac"] for r in records], dtype=object),
        allow_pickle=True,
    )
    return name, n_in, n_fail


def main() -> int:
    ap = argparse.ArgumentParser(description="Simulate stick patterns from extracted CSD shards.")
    ap.add_argument("--in", dest="inp", default="cache/csd", help="dir of *.jsonl.gz from stage 1")
    ap.add_argument("--out", default="cache/sticks")
    ap.add_argument("--workers", type=int, default=8, help="keep modest; Vast boxes report hundreds of cores")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(args.inp, "*.jsonl.gz")))
    if not shards:
        print(f"[simulate] no shards in {args.inp} — run stage 1 first")
        return 1

    print(f"[simulate] {len(shards)} shards, {args.workers} workers -> {args.out}", flush=True)
    t0 = time.time()
    tot_in = tot_fail = 0
    with Pool(args.workers) as pool:
        for name, n_in, n_fail in pool.imap_unordered(_process_shard, [(s, args.out) for s in shards]):
            tot_in += n_in
            tot_fail += n_fail
            print(f"  [{name}] {n_in - n_fail:,}/{n_in:,} simulated ({n_fail:,} failed)", flush=True)

    dt = time.time() - t0
    ok = tot_in - tot_fail
    print(f"\n[simulate] {ok:,}/{tot_in:,} stick patterns ({100*ok/max(tot_in,1):.1f}%) in {dt/60:.1f} min")
    if tot_in and tot_fail / tot_in > 0.15:
        print(f"[simulate] WARNING: {100*tot_fail/tot_in:.1f}% failure rate is high — inspect before trusting this dataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
