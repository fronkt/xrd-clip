"""PXRD-degeneracy deduplication.

THE PROBLEM (found empirically on the pilot, not anticipated on paper)
---------------------------------------------------------------------
Clean self-retrieval on the pilot scored 0.977, not 1.000 -- and every single "failure" turned out
to be a CSD redetermination of the same compound (ABAMEX01 retrieved ABAMEX, etc.). The model was
right and the metric was wrong.

This matters more than it first looks. Two determinations of the same structure produce essentially
identical powder patterns, so NO PXRD-based method can separate them, ours included. Scoring them as
errors doesn't just add noise -- it puts a hard ceiling below 1.0 on every number in the paper, and
that ceiling moves around with how many redeterminations happen to land in the library. Left alone,
it would have quietly corrupted the headline result and any reviewer who checked would have found it.

THE FIX
-------
Entries whose CLEAN patterns are indistinguishable form a "PXRD-degeneracy class". We collapse each
class to one representative, so every retrieval target is unique and the task is well posed.

Why not just group by CSD base refcode (the 6-letter stem)? Because the stem identifies the
*compound*, not the *structure* -- and POLYMORPHS of one compound share a stem while having
genuinely different packings and genuinely different patterns. Merging them would be wrong in the
opposite direction: it would credit a method for confusing two structures it ought to distinguish.

So the stem is used only as a cheap candidate filter, and the actual merge decision is made on
pattern correlation. Same compound + indistinguishable pattern = duplicate. Same compound +
distinguishable pattern = polymorphs, kept apart. That is the physically correct line.
"""
from __future__ import annotations

import numpy as np

from xrdclip.data import augment as A

# Two clean patterns above this correlation are treated as the same phase for retrieval purposes.
# 0.99 is deliberately strict: real polymorphs of the same compound sit well below it (their peaks
# move), while redeterminations sit at ~1.0.
DUP_THRESHOLD = 0.99


def base_refcode(rc: str) -> str:
    """CSD stem: 'ABAMEX01' -> 'ABAMEX'. Identifies the COMPOUND, not the structure."""
    return rc.rstrip("0123456789")


class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def find_duplicate_classes(store, refcodes: "list[str]",
                           threshold: float = DUP_THRESHOLD) -> "dict[str, int]":
    """Map refcode -> degeneracy-class id.

    Only entries sharing a CSD stem are ever compared, which keeps this O(sum of group^2) instead
    of O(n^2) -- for 300k structures the dense form would be 45 billion pairs and is not an option.
    Cross-compound pattern collisions are possible in principle but vanishingly rare, and a
    false *split* is harmless here (it only makes the task marginally harder), whereas a false
    *merge* would hide a real error. We take the safe side.
    """
    stems: dict[str, list[int]] = {}
    for i, rc in enumerate(refcodes):
        stems.setdefault(base_refcode(rc), []).append(i)

    uf = _UnionFind(len(refcodes))
    cache: dict[int, np.ndarray] = {}

    for idxs in stems.values():
        if len(idxs) < 2:
            continue
        for i in idxs:
            if i not in cache:
                cache[i] = A.standardize(A.render_sticks(store[refcodes[i]], A.CLEAN))
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                ia, ib = idxs[a], idxs[b]
                corr = float(np.dot(cache[ia], cache[ib]) / A.N_POINTS)
                if corr >= threshold:
                    uf.union(ia, ib)

    roots = {}
    out: dict[str, int] = {}
    for i, rc in enumerate(refcodes):
        r = uf.find(i)
        if r not in roots:
            roots[r] = len(roots)
        out[rc] = roots[r]
    return out


def dedup_library(store, refcodes: "list[str]",
                  threshold: float = DUP_THRESHOLD) -> "tuple[list[str], dict[str, int]]":
    """Collapse each degeneracy class to one representative (the lowest refcode).

    Returns (representative_refcodes, refcode -> class_id).
    """
    classes = find_duplicate_classes(store, refcodes, threshold)
    by_class: dict[int, list[str]] = {}
    for rc, cid in classes.items():
        by_class.setdefault(cid, []).append(rc)
    reps = sorted(min(v) for v in by_class.values())
    return reps, classes


def dedup_report(store, refcodes: "list[str]", threshold: float = DUP_THRESHOLD) -> dict:
    reps, classes = dedup_library(store, refcodes, threshold)
    n_class = len(set(classes.values()))
    sizes: dict[int, int] = {}
    for cid in classes.values():
        sizes[cid] = sizes.get(cid, 0) + 1
    multi = sum(1 for v in sizes.values() if v > 1)
    return {
        "n_entries": len(refcodes),
        "n_classes": n_class,
        "n_removed": len(refcodes) - n_class,
        "n_multi_entry_classes": multi,
        "largest_class": max(sizes.values()) if sizes else 0,
        "threshold": threshold,
        "representatives": reps,
    }
