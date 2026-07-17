"""Polymorph-family utilities for the polymorph-resolving pivot.

WHY THIS EXISTS
---------------
The competitive lit review (docs/competitive-landscape-2026.md) found xrd-clip's original framing
(contrastive PXRD-structure retrieval) pre-empted by XCCP and XtalNet. Both lean on COMPOSITION +
SPACE-GROUP fingerprints. That signal is exactly what vanishes for POLYMORPHS of one molecule: same
formula, often same space group, and the only thing that differs is the packing -> the pattern. So
the pivot is to make polymorph discrimination the headline task, where composition/symmetry methods
are structurally blind and a pure pattern<->packing model must earn its keep.

Two consumers:
  - training: `PolymorphBatchSampler` co-locates sibling polymorphs in the same batch so InfoNCE sees
    them as HARD NEGATIVES (a same-molecule sibling is the hardest possible distractor).
  - evaluation: `family_map` drives the within-family retrieval eval (eval/polymorph_retrieval.py).

The family map itself is produced offline by dedup.py's correlation split (redeterminations collapsed,
distinguishable polymorphs kept apart) and stored as refcodes/polymorph_families.json. See that file's
`description`. Refcodes only, never CIFs.
"""
from __future__ import annotations

import json

import numpy as np
from torch.utils.data import Sampler


def load_families(path: str) -> "dict[str, list[str]]":
    """Load {CSD stem -> [one representative refcode per distinguishable polymorph]}."""
    with open(path) as fh:
        return json.load(fh)["families"]


def family_map(refcodes: "list[str]", families: "dict[str, list[str]]") -> "dict[str, int]":
    """Map refcode -> family id, keeping ONLY families with >=2 representatives present here.

    A family with a single present member has no sibling to contrast against in this split, so it is
    not a polymorph family *for this split* and its member is treated as a singleton. Returns a dense
    integer id per surviving family.
    """
    present = set(refcodes)
    fam_id: dict[str, int] = {}
    out: dict[str, int] = {}
    for stem, reps in families.items():
        here = [rc for rc in reps if rc in present]
        if len(here) < 2:
            continue
        fid = fam_id.setdefault(stem, len(fam_id))
        for rc in here:
            out[rc] = fid
    return out


def family_groups(refcodes: "list[str]", families: "dict[str, list[str]]") -> "dict[str, list[str]]":
    """{stem -> present representative refcodes}, keeping only families with >=2 present.

    The eval-side counterpart to `family_map`: within-family retrieval ranks each member against its
    siblings, so it needs the refcode lists, not integer ids.
    """
    present = set(refcodes)
    out: dict[str, list[str]] = {}
    for stem, reps in families.items():
        here = sorted(rc for rc in reps if rc in present)
        if len(here) >= 2:
            out[stem] = here
    return out


class PolymorphBatchSampler(Sampler):
    """Batches that deliberately place sibling polymorphs together as hard negatives.

    Standard shuffling almost never lands two polymorphs of one molecule in the same batch, so the
    contrastive loss is rarely asked to tell them apart -- precisely the distinction the pivot cares
    about. This sampler makes each polymorph family an ATOMIC unit (its members never split across a
    batch boundary), distributes the family units across a fixed number of batches, and fills the rest
    of each batch with singleton structures.

    Design constraints that matter:
      * Exactly `len(self)` batches of exactly `batch_size` are produced every epoch, so
        OneCycleLR's `steps_per_epoch` stays correct (over-stepping it raises).
      * Families are atomic -> a sibling is always in-batch with its anchor -> guaranteed hard negatives.
      * Fresh shuffle each epoch via `set_epoch`.
    """

    def __init__(self, refcodes: "list[str]", families: "dict[str, list[str]]",
                 batch_size: int, seed: int = 0, max_per_family: int = 8):
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.max_per_family = int(max_per_family)

        fmap = family_map(refcodes, families)
        fam_members: dict[int, list[int]] = {}
        singles: list[int] = []
        for i, rc in enumerate(refcodes):
            fid = fmap.get(rc)
            if fid is None:
                singles.append(i)
            else:
                fam_members.setdefault(fid, []).append(i)

        # Cap pathologically large families so one stem cannot swamp a batch. Members BEYOND the cap
        # become singletons rather than being dropped -- silently losing structures would corrupt both
        # coverage and the batch count (and therefore the LR schedule).
        self._family_units = []
        for m in fam_members.values():
            self._family_units.append(m[: self.max_per_family])
            singles.extend(m[self.max_per_family:])
        self._singles = singles
        n_total = sum(len(u) for u in self._family_units) + len(singles)
        self._num_batches = n_total // self.batch_size
        self.n_families = len(self._family_units)
        self.n_family_structs = sum(len(u) for u in self._family_units)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self):
        rng = np.random.default_rng((self.seed, self.epoch))
        B, nb = self.batch_size, self._num_batches
        if nb == 0:
            return

        units = [list(u) for u in self._family_units]
        rng.shuffle(units)
        singles = list(self._singles)
        rng.shuffle(singles)

        # Round-robin family units into bins, spilling to the next bin with room. With ~10^4 small
        # families over ~10^2-10^3 bins, loads stay far below B, so spillage is rare.
        bins: list[list[int]] = [[] for _ in range(nb)]
        b = 0
        for u in units:
            tries = 0
            while len(bins[b]) + len(u) > B and tries < nb:
                b = (b + 1) % nb
                tries += 1
            if len(bins[b]) + len(u) <= B:
                bins[b].extend(u)
            else:                      # no bin has room (should not happen with the cap) -> drop to singles
                singles.extend(u)
            b = (b + 1) % nb

        # Fill every bin to exactly B with singletons; drop the singleton remainder (drop_last parity).
        sp = 0
        for bin_ in bins:
            need = B - len(bin_)
            bin_.extend(singles[sp:sp + need])
            sp += need
            rng.shuffle(bin_)           # so structure/pattern order within a batch is not family-blocked
            yield bin_
