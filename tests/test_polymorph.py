"""Tests for the polymorph pivot: family grouping, the hard-negative batch sampler, the family-aware
split merge, and the within-family ranking primitive.

The load-bearing guarantees these lock down:
  * a polymorph's siblings are ALWAYS in the same training batch (else there is no hard negative);
  * no polymorph family straddles a split after the merge (else the within-family eval is contaminated);
  * the within-family ranking is computed correctly (else the headline number is wrong).
"""
from __future__ import annotations

import numpy as np
import pytest

from xrdclip.data.polymorph import (
    family_map, family_groups, PolymorphBatchSampler,
)
from xrdclip.data.splits import merge_polymorph_groups
from xrdclip.eval.polymorph_retrieval import _rank_within


FAMILIES = {
    "AAA": ["AAA", "AAA01", "AAA02"],   # 3 polymorphs
    "BBB": ["BBB", "BBB01"],            # 2 polymorphs
    "CCC": ["CCC", "CCC01"],            # present only partially below
}


def test_family_map_excludes_singletons():
    refs = ["AAA", "AAA01", "AAA02", "BBB", "BBB01", "CCC", "ZZZ"]  # CCC01 absent -> CCC singleton
    fmap = family_map(refs, FAMILIES)
    assert set(fmap) == {"AAA", "AAA01", "AAA02", "BBB", "BBB01"}
    assert fmap["AAA"] == fmap["AAA01"] == fmap["AAA02"]
    assert fmap["BBB"] == fmap["BBB01"]
    assert fmap["AAA"] != fmap["BBB"]
    assert "CCC" not in fmap and "ZZZ" not in fmap


def test_family_groups_shape():
    refs = ["AAA", "AAA01", "AAA02", "BBB", "BBB01", "CCC"]
    groups = family_groups(refs, FAMILIES)
    assert set(groups) == {"AAA", "BBB"}
    assert groups["AAA"] == ["AAA", "AAA01", "AAA02"]
    assert groups["BBB"] == ["BBB", "BBB01"]


def _families_from_refs(refs):
    """Build a family dict of stem -> its refcodes for refs whose stem repeats."""
    import collections
    fam = collections.defaultdict(list)
    for rc in refs:
        fam[rc.rstrip("0123456789")].append(rc)
    return {s: v for s, v in fam.items() if len(v) >= 2}


def _base(f):
    return "FAM" + chr(65 + f // 26) + chr(65 + f % 26)   # distinct letter stem per family


def test_sampler_colocates_siblings_and_exact_size():
    # 30 polymorph structures (10 families of 3) + 170 singletons, all with realistic distinct stems.
    # (Suffixes are 2-digit so rstrip yields the letter stem; singletons end in a letter -> unique stem.)
    refs = [_base(f) + suf for f in range(10) for suf in ("", "01", "02")]
    refs += [f"SNG{i:04d}X" for i in range(170)]           # trailing letter -> each is its own stem
    fams = _families_from_refs(refs)
    assert len(fams) == 10                                 # exactly the 10 polymorph families
    B = 32
    sampler = PolymorphBatchSampler(refs, fams, batch_size=B, seed=1)

    idx_to_stem = {i: refs[i].rstrip("0123456789") for i in range(len(refs))}
    batches = list(iter(sampler))

    assert len(batches) == len(sampler) == len(refs) // B
    for b in batches:
        assert len(b) == B                                  # exact batch size
        assert len(set(b)) == B                             # no duplicate indices within a batch

    # every family's members land together in exactly one batch (atomic units)
    fam_batch = {}
    for bi, b in enumerate(batches):
        stems_here = [idx_to_stem[i] for i in b if idx_to_stem[i].startswith("FAM")]
        for s in set(stems_here):
            members = [i for i in b if idx_to_stem[i] == s]
            assert len(members) == 3, f"family {s} split across batches"
            assert s not in fam_batch, f"family {s} appears in two batches"
            fam_batch[s] = bi
    assert len(fam_batch) == 10                             # all 10 families placed


def test_sampler_reshuffles_each_epoch():
    refs = [_base(f) + suf for f in range(8) for suf in ("", "01")] + [f"S{i:04d}X" for i in range(100)]
    fams = _families_from_refs(refs)
    s = PolymorphBatchSampler(refs, fams, batch_size=16, seed=0)
    s.set_epoch(0); b0 = list(iter(s))
    s.set_epoch(1); b1 = list(iter(s))
    assert b0 != b1                                         # different composition across epochs
    assert len(b0) == len(b1)                              # but same number of steps (scheduler safety)


def test_merge_polymorph_groups_no_family_crosses():
    # scaffold grouping scattered a polymorph pair (X in one group, Z in another, e.g. acyclic nogroup)
    groups = {"scafA": ["X", "X01"], "__nogroup__Z": ["Z"], "scafB": ["Q"]}
    families = {"X": ["X", "Z"]}                            # X and Z are polymorphs of one molecule
    merged = merge_polymorph_groups(groups, families)
    # the group holding X must also hold Z
    holding = [g for g, rcs in merged.items() if "X" in rcs][0]
    assert "Z" in merged[holding]
    # unrelated group untouched
    assert any(rcs == ["Q"] for rcs in merged.values())


def test_rank_within():
    # diagonal is the max -> every query ranks its own structure first
    n = 5
    scores = np.eye(n) * 2.0 + np.random.default_rng(0).random((n, n)) * 0.1
    np.fill_diagonal(scores, 5.0)
    assert (_rank_within(scores) == 0).all()
    # diagonal is the min -> worst rank
    bad = np.ones((n, n))
    np.fill_diagonal(bad, -1.0)
    assert (_rank_within(bad) == n - 1).all()
