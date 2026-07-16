"""Torch dataset: serves (augmented diffractogram, crystal graph) pairs for contrastive training.

The augmentation is applied ON THE FLY, per __getitem__, with a fresh random degradation every
time — so across epochs the model sees the same structure under thousands of different
instruments, crystallite sizes, textures, and noise levels, and the ONLY thing invariant across
all of them is the underlying structure. That is precisely the signal the contrastive loss is
built to extract.

Caching rendered patterns would destroy this. Do not "optimise" it by precomputing.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from xrdclip.data import augment as A

# Hard bound on atoms fed to the structure encoder, independent of the asymmetric-unit reduction in
# simulate.py. The Transformer is O(atoms^2); this guarantees a single pathological structure (huge
# Z', failed symmetry analysis) can never OOM a training run. The asymmetric unit is almost always
# well under this.
MAX_ENC_ATOMS = 256


class StickStore:
    """Lazily-loaded stick patterns keyed by refcode, backed by the .npz shards."""

    FIELDS = ("two_theta", "intensity", "hkl", "d_hkl", "atom_z", "atom_frac")

    def __init__(self, sticks_dir: str, refcodes: "list[str] | None" = None):
        self.records: dict[str, dict] = {}
        wanted = set(refcodes) if refcodes is not None else None

        for path in sorted(glob.glob(os.path.join(sticks_dir, "*.npz"))):
            with np.load(path, allow_pickle=True) as z:
                rcs = z["refcode"]
                keep = [i for i, rc in enumerate(rcs) if wanted is None or rc in wanted]
                if not keep:
                    continue
                cell, recip = z["cell"], z["recip"]
                cols = {f: z[f] for f in self.FIELDS}
                sg, na = z["spacegroup"], z["n_atoms"]
                for i in keep:
                    self.records[str(rcs[i])] = {
                        "two_theta": cols["two_theta"][i],
                        "intensity": cols["intensity"][i],
                        "hkl": cols["hkl"][i],
                        "d_hkl": cols["d_hkl"][i],
                        "atom_z": cols["atom_z"][i],
                        "atom_frac": cols["atom_frac"][i],
                        "recip": recip[i],
                        "cell": cell[i],
                        "spacegroup": str(sg[i]),
                        "n_atoms": int(na[i]),
                    }

        self.refcodes = sorted(self.records)
        if not self.refcodes:
            raise RuntimeError(f"StickStore loaded 0 records from {sticks_dir}")

    def __len__(self) -> int:
        return len(self.refcodes)

    def __getitem__(self, rc: str) -> dict:
        return self.records[rc]


def structure_features(rec: dict) -> np.ndarray:
    """Lattice descriptor fed to the structure encoder alongside the graph.

    The diffraction pattern's PEAK POSITIONS are a pure function of the lattice, so the structure
    encoder cannot possibly place peaks without seeing the cell. This is the single most important
    input to get right; a graph-only encoder is information-starved by construction (SPEC §2.2).

    We feed the reduced cell in a scale-aware form: log lengths (so a 5 A and a 50 A axis are
    comparably scaled), cos of the angles (continuous, avoids the 0/180 wraparound), and log volume.
    """
    a, b, c, al, be, ga = rec["cell"].astype(np.float64)
    ca, cb, cg = np.cos(np.radians([al, be, ga]))
    vol = a * b * c * np.sqrt(max(1e-8, 1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg))
    return np.array(
        [np.log(a), np.log(b), np.log(c), ca, cb, cg, np.log(max(vol, 1e-6)), rec["n_atoms"] / 100.0],
        dtype=np.float32,
    )


class ContrastiveXRDDataset(Dataset):
    """Yields (pattern, structure_feats, index) triples. Index is the contrastive label."""

    def __init__(self, store: StickStore, refcodes: "list[str]", augment: bool = True,
                 ranges: A.Ranges = A.Ranges(), seed: int = 0):
        self.store = store
        self.refcodes = [rc for rc in refcodes if rc in store.records]
        self.augment = augment
        self.ranges = ranges
        self.seed = seed
        self._feat_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.refcodes)

    def __getitem__(self, i: int):
        rc = self.refcodes[i]
        rec = self.store[rc]

        # Per-worker, per-epoch-varying RNG: torch reseeds workers each epoch, so this gives a
        # genuinely fresh degradation every time rather than a fixed per-index one.
        rng = np.random.default_rng(
            (self.seed, i, torch.initial_seed() % (2**32), np.random.randint(2**31))
        )

        p = A.sample_params(rng, self.ranges) if self.augment else A.CLEAN
        y = A.render_sticks(rec, p, rng=rng)
        y = A.standardize(y)

        if rc not in self._feat_cache:
            self._feat_cache[rc] = structure_features(rec)

        az = np.asarray(rec["atom_z"], dtype=np.int64)[:MAX_ENC_ATOMS]
        af = np.asarray(rec["atom_frac"], dtype=np.float32)[:MAX_ENC_ATOMS]
        return {
            "pattern": torch.from_numpy(y).unsqueeze(0),            # (1, N_POINTS)
            "lattice": torch.from_numpy(self._feat_cache[rc]),     # (8,)
            "atom_z": torch.from_numpy(az),
            "atom_frac": torch.from_numpy(af),
            "index": i,
        }


def collate(batch: "list[dict]") -> dict:
    """Pad the ragged atom sets to the batch max and build the attention mask."""
    B = len(batch)
    A = max(int(b["atom_z"].shape[0]) for b in batch)

    atom_z = torch.zeros(B, A, dtype=torch.long)             # 0 = padding (embedding padding_idx)
    atom_frac = torch.zeros(B, A, 3, dtype=torch.float32)
    mask = torch.zeros(B, A, dtype=torch.bool)
    for j, b in enumerate(batch):
        n = int(b["atom_z"].shape[0])
        atom_z[j, :n] = b["atom_z"]
        atom_frac[j, :n] = b["atom_frac"]
        mask[j, :n] = True

    return {
        "pattern": torch.stack([b["pattern"] for b in batch]),
        "lattice": torch.stack([b["lattice"] for b in batch]),
        "atom_z": atom_z,
        "atom_frac": atom_frac,
        "mask": mask,
        "index": torch.tensor([b["index"] for b in batch], dtype=torch.long),
    }
