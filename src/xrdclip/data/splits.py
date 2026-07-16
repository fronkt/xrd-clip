"""STAGE 3 — scaffold splits.  Main interpreter (rdkit). No CSD licence.

THE LEAKAGE PROBLEM
-------------------
The CSD is dense with near-duplicates: redeterminations of the same compound at different
temperatures, polymorphs, close analogues, salts of the same cation. A RANDOM split scatters
these across train/test, so a test query's near-twin sits in the training set and both the
retrieval model (xrd-clip) and the generative model (pxrd-flow) can win by memorisation. The
numbers look wonderful and mean nothing.

We split by Bemis-Murcko scaffold: every entry sharing a scaffold lands in exactly one split.
That is not a perfect duplicate guard (different scaffolds can still be similar), but it removes
the dominant failure mode and it is the standard the cheminformatics community expects.

BOTH numbers get reported. The random-vs-scaffold gap quantifies how much apparent performance
in this area is memorisation — that gap is itself a result, and it is nearly free to produce.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def murcko_scaffold(smiles: str) -> str:
    """Bemis-Murcko scaffold of the largest component.

    Falls back to the largest-component canonical SMILES when a molecule has no ring system
    (an acyclic molecule has an empty Murcko scaffold, and bucketing every acyclic entry into one
    giant '' group would put a huge, chemically unrelated blob in a single split).
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""

    # Largest component: salts/solvates otherwise scaffold on the counter-ion.
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not frags:
        return ""
    mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())

    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        smi = Chem.MolToSmiles(scaf)
    except Exception:
        return ""

    if smi:
        return smi
    try:
        return Chem.MolToSmiles(mol)  # acyclic -> use the molecule itself as its own group
    except Exception:
        return ""


def load_index(sticks_dir: str) -> "tuple[list[str], list[str]]":
    """Read refcodes + SMILES from the simulated stick shards (the true dataset membership)."""
    refcodes: list[str] = []
    smiles: list[str] = []
    for path in sorted(glob.glob(os.path.join(sticks_dir, "*.npz"))):
        with np.load(path, allow_pickle=True) as z:
            refcodes.extend(z["refcode"].tolist())
            smiles.extend(z["smiles"].tolist())
    return refcodes, smiles


def group_by_scaffold(refcodes, smiles, workers: int = 8) -> "dict[str, list[str]]":
    from multiprocessing import Pool

    with Pool(workers) as pool:
        scaffolds = pool.map(murcko_scaffold, smiles, chunksize=500)

    groups: dict[str, list[str]] = defaultdict(list)
    for rc, scaf in zip(refcodes, scaffolds):
        groups[scaf if scaf else f"__nogroup__{rc}"].append(rc)
    return groups


def split_groups(groups: "dict[str, list[str]]", fracs=(0.8, 0.1, 0.1), seed: int = 0):
    """Assign whole scaffold groups to splits, targeting the requested entry-count fractions.

    Greedy largest-first bin packing: big scaffold families (which can hold thousands of entries)
    would badly skew the split if assigned at random.
    """
    rng = np.random.default_rng(seed)
    keys = list(groups.keys())
    rng.shuffle(keys)
    keys.sort(key=lambda k: -len(groups[k]))

    total = sum(len(v) for v in groups.values())
    targets = [f * total for f in fracs]
    counts = [0, 0, 0]
    out: list[list[str]] = [[], [], []]

    for k in keys:
        # send this group to whichever split is furthest below its target, proportionally
        deficits = [(targets[i] - counts[i]) / max(targets[i], 1) for i in range(3)]
        i = int(np.argmax(deficits))
        out[i].extend(groups[k])
        counts[i] += len(groups[k])

    return out, counts


def main() -> int:
    ap = argparse.ArgumentParser(description="Build scaffold (and random) splits.")
    ap.add_argument("--sticks", default="cache/sticks")
    ap.add_argument("--out", default="refcodes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    refcodes, smiles = load_index(args.sticks)
    if not refcodes:
        print(f"[splits] no stick shards in {args.sticks} — run stage 2 first")
        return 1
    print(f"[splits] {len(refcodes):,} structures")

    groups = group_by_scaffold(refcodes, smiles, args.workers)
    real = {k: v for k, v in groups.items() if not k.startswith("__nogroup__")}
    print(f"[splits] {len(groups):,} scaffold groups ({len(real):,} named)")
    if real:
        biggest = max(real.items(), key=lambda kv: len(kv[1]))
        print(f"[splits] largest scaffold family: {len(biggest[1]):,} entries")

    # --- scaffold split (the one we train on) ---
    (tr, va, te), counts = split_groups(groups, seed=args.seed)
    print(f"[splits] scaffold  train/val/test = {counts[0]:,} / {counts[1]:,} / {counts[2]:,}")

    # --- random split (reported alongside, to expose the memorisation gap) ---
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(refcodes))
    n_tr, n_va = int(0.8 * len(perm)), int(0.9 * len(perm))
    rand = [
        [refcodes[i] for i in perm[:n_tr]],
        [refcodes[i] for i in perm[n_tr:n_va]],
        [refcodes[i] for i in perm[n_va:]],
    ]

    for name, sets in (("scaffold", (tr, va, te)), ("random", rand)):
        for split, rcs in zip(("train", "val", "test"), sets):
            path = os.path.join(args.out, f"{name}_{split}.txt")
            with open(path, "w") as fh:
                fh.write("\n".join(sorted(rcs)) + "\n")
        print(f"[splits] wrote {name}_{{train,val,test}}.txt")

    # --- verify no scaffold crosses a split boundary (the whole point) ---
    rc2scaf = {}
    for scaf, rcs in groups.items():
        for rc in rcs:
            rc2scaf[rc] = scaf
    s_tr, s_va, s_te = ({rc2scaf[r] for r in s} for s in (tr, va, te))
    leaks = (s_tr & s_va) | (s_tr & s_te) | (s_va & s_te)
    assert not leaks, f"SCAFFOLD LEAK across splits: {list(leaks)[:5]}"
    print("[splits] verified: no scaffold appears in more than one split")

    with open(os.path.join(args.out, "splits_manifest.json"), "w") as fh:
        json.dump({
            "n_structures": len(refcodes),
            "n_scaffold_groups": len(groups),
            "scaffold_counts": {"train": counts[0], "val": counts[1], "test": counts[2]},
            "random_counts": {"train": len(rand[0]), "val": len(rand[1]), "test": len(rand[2])},
            "seed": args.seed,
        }, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
