"""STAGE 1 — CSD extraction.  *** RUNS UNDER THE CCDC MINICONDA, NOT THE MAIN PYTHON ***

    <CCDC>/ccdc-software/csd-python-api/miniconda/python.exe -m xrdclip.data.csd_extract ...

This is the only stage that requires a CSD licence.  It scans the CSD, applies the
inclusion filters, and writes gzipped JSONL shards containing, per entry:

    refcode, cif, smiles, and the metadata needed for filtering/analysis.

Everything downstream (simulation, augmentation, splits, training) is licence-free and runs
under the main interpreter.  That separation is not cosmetic: the GPU box has no CSD licence,
so the training pipeline must never import `ccdc`.

LICENCE DISCIPLINE
------------------
The shards this writes contain CSD-derived structures and are therefore *not redistributable*.
They land in a gitignored cache.  What we publish is `refcodes/*.txt` — the identifier list —
from which any licence holder can regenerate everything.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from multiprocessing import Process, Queue

# --- inclusion filters (mirror SPEC.md §3.1) ---------------------------------------------
ALLOWED_ELEMENTS = {"H", "C", "N", "O", "F", "S", "Cl", "Br", "I", "P"}
MAX_R_FACTOR = 7.5        # percent
Z_PRIME_RANGE = (1.0, 2.0)
MAX_ATOMS = 300           # keeps the crystal graph and the flow-model state tractable


def passes(entry) -> "tuple[bool, str]":
    """Apply the inclusion filters.  Returns (kept, reason_if_rejected)."""
    try:
        if not entry.is_organic:
            return False, "not_organic"
        if not entry.has_3d_structure:
            return False, "no_3d"
        if entry.has_disorder:
            return False, "disorder"
        rf = entry.r_factor
        if rf is None or rf > MAX_R_FACTOR:
            return False, "r_factor"

        crystal = entry.crystal
        zp = crystal.z_prime
        if zp is None or not (Z_PRIME_RANGE[0] <= zp <= Z_PRIME_RANGE[1]):
            return False, "z_prime"

        mol = crystal.molecule
        if mol is None or not mol.atoms:
            return False, "no_molecule"
        # Polymeric entries have no well-defined discrete molecule -> unusable as rigid bodies.
        if getattr(entry, "is_polymeric", False):
            return False, "polymeric"

        atoms = mol.atoms
        if len(atoms) > MAX_ATOMS:
            return False, "too_many_atoms"
        if not {a.atomic_symbol for a in atoms} <= ALLOWED_ELEMENTS:
            return False, "element"

        # Every atom must carry 3D coordinates or the structure is useless to us.
        if any(a.coordinates is None for a in atoms):
            return False, "missing_coords"

        return True, ""
    except Exception as exc:  # a handful of CSD entries raise on attribute access
        return False, f"error:{type(exc).__name__}"


def extract_record(entry) -> "dict | None":
    """Pull everything downstream stages need.  None if the entry cannot be serialised."""
    try:
        crystal = entry.crystal
        mol = crystal.molecule
        smiles = mol.smiles
        if not smiles:
            return None  # no SMILES -> cannot compute a Murcko scaffold -> cannot split safely
        cif = crystal.to_string("cif")
        if not cif:
            return None
        cell = crystal.cell_lengths
        angles = crystal.cell_angles
        return {
            "refcode": entry.identifier,
            "cif": cif,
            "smiles": smiles,
            "spacegroup": crystal.spacegroup_symbol,
            "z_prime": crystal.z_prime,
            "z_value": crystal.z_value,
            "r_factor": entry.r_factor,
            "n_atoms": len(mol.atoms),
            "formula": crystal.formula,
            "cell": [cell.a, cell.b, cell.c, angles.alpha, angles.beta, angles.gamma],
            "is_solvate": bool(getattr(entry, "is_solvate", False)),
            "temperature": getattr(entry, "temperature", None),
        }
    except Exception:
        return None


def worker(wid: int, lo: int, hi: int, outdir: str, q: Queue) -> None:
    """Scan CSD entries [lo, hi) and write one shard."""
    from ccdc.io import EntryReader

    reader = EntryReader("CSD")
    path = os.path.join(outdir, f"shard_{wid:03d}.jsonl.gz")
    kept = 0
    rejects: dict[str, int] = {}
    t0 = time.time()

    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for i in range(lo, hi):
            try:
                entry = reader[i]
            except Exception:
                rejects["read_error"] = rejects.get("read_error", 0) + 1
                continue

            ok, why = passes(entry)
            if not ok:
                rejects[why] = rejects.get(why, 0) + 1
                continue

            rec = extract_record(entry)
            if rec is None:
                rejects["serialise"] = rejects.get("serialise", 0) + 1
                continue

            fh.write(json.dumps(rec) + "\n")
            kept += 1

            if kept % 2000 == 0:
                q.put(("progress", wid, i - lo, hi - lo, kept))

    q.put(("done", wid, hi - lo, hi - lo, kept, rejects, time.time() - t0))


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract a filtered molecular-crystal subset from the CSD.")
    ap.add_argument("--out", default="cache/csd", help="output dir for shards (gitignored)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="scan only the first N CSD entries (0 = all; use for a pilot)")
    args = ap.parse_args()

    from ccdc.io import EntryReader

    os.makedirs(args.out, exist_ok=True)
    total = args.limit if args.limit else len(EntryReader("CSD"))
    print(f"[extract] scanning {total:,} CSD entries with {args.workers} workers -> {args.out}", flush=True)

    bounds = [(i * total // args.workers, (i + 1) * total // args.workers) for i in range(args.workers)]
    q: Queue = Queue()
    procs = [Process(target=worker, args=(w, lo, hi, args.out, q)) for w, (lo, hi) in enumerate(bounds)]
    for p in procs:
        p.start()

    done, kept_total = 0, 0
    all_rejects: dict[str, int] = {}
    t0 = time.time()
    while done < args.workers:
        msg = q.get()
        if msg[0] == "progress":
            _, wid, cur, tot, kept = msg
            print(f"  [w{wid}] {cur:,}/{tot:,} scanned, {kept:,} kept", flush=True)
        else:
            _, wid, _, _, kept, rejects, dt = msg
            done += 1
            kept_total += kept
            for k, v in rejects.items():
                all_rejects[k] = all_rejects.get(k, 0) + v
            print(f"  [w{wid}] DONE {kept:,} kept in {dt/60:.1f} min", flush=True)

    for p in procs:
        p.join()

    dt = time.time() - t0
    print(f"\n[extract] kept {kept_total:,} / {total:,} ({100*kept_total/max(total,1):.1f}%) in {dt/60:.1f} min")
    print("[extract] rejection breakdown:")
    for k, v in sorted(all_rejects.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<20} {v:>9,}")

    # A machine-readable manifest so downstream stages (and the paper) can cite exact provenance.
    manifest = {
        "n_scanned": total,
        "n_kept": kept_total,
        "rejects": all_rejects,
        "filters": {
            "allowed_elements": sorted(ALLOWED_ELEMENTS),
            "max_r_factor": MAX_R_FACTOR,
            "z_prime_range": list(Z_PRIME_RANGE),
            "max_atoms": MAX_ATOMS,
            "organic": True, "no_disorder": True, "has_3d": True, "non_polymeric": True,
        },
        "elapsed_min": dt / 60,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[extract] wrote manifest -> {args.out}/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
