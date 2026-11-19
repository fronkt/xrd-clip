"""Ground-truth checks for the resubmission (tasks/todo.md D2; RSC Advances board + referee 1, point 2).

Run with the CCDC miniconda interpreter (licensed), after scripts/18:

  <ccdc python> scripts/47_anchor_checks.py            -> results/anchor_checks.json

1. every_anchor. Step 4 of scripts/18 compares each stem-family member with the two lowest-RMSD anchors only (a
   speed shortcut; the text said "every anchor"). Here every member that is NOT in a same-form set is compared, strict
   tier, with every remaining anchor. A 15/15 match would be a correct answer the benchmark scores as a distractor
   (other_form) or ignores (unknown).
2. lenient. For each pattern anchored in the lenient tier: the molecular components (formula) of the opXRD structure
   and of each anchor, and which single relaxation of the strict comparison (H counts, smallest components, bond
   types, bond counts) already gives 15/15. "smallest components" is the case that could pair a solvate with an
   anhydrate or a salt with its free form.

Refcodes and comparison metadata only: no CSD structure data is written.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from importlib import import_module

from ccdc.crystal import Crystal, PackingSimilarity
from ccdc.io import EntryReader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
t18 = import_module("18_opxrd_truth_sets")
DATA, SHELL = t18.DATA, t18.SHELL
FLAGS = ("ignore_hydrogen_counts", "ignore_smallest_components", "ignore_bond_types", "ignore_bond_counts")


def make(flags=()) -> PackingSimilarity:
    """scripts/18's strict settings, plus the given relaxations."""
    ps = PackingSimilarity()
    ps.settings.packing_shell_size = SHELL
    ps.settings.ignore_hydrogen_positions = True
    ps.settings.allow_molecular_differences = False
    ps.settings.timeout_ms = 30000
    for f in flags:
        setattr(ps.settings, f, True)
    return ps


def compare(ps, a, b) -> "tuple[int, float] | None":
    try:
        r = ps.compare(a, b)
        return (r.nmatched_molecules, float(r.rmsd)) if r is not None else None
    except Exception:
        return None


def components(cry) -> "list[str]":
    try:
        return sorted(c.formula for c in cry.molecule.components)
    except Exception:
        return []


def main() -> int:
    truth = json.load(open(f"{DATA}/opxrd/opxrd_truth.json"))
    pats = {o["id"]: o for o in json.load(open(f"{DATA}/opxrd/opxrd_cnrs_organic.json"))["organic"]}
    reader = EntryReader("CSD")
    strict = make()
    crys: dict = {}

    def crystal(rc):
        if rc not in crys:
            crys[rc] = reader.entry(rc).crystal
        return crys[rc]

    t0 = time.time()
    every, n_cmp = [], 0
    for t in truth:
        anchors = sorted(t["anchors"].items(), key=lambda kv: kv[1]["rmsd"])
        if len(anchors) <= 2:
            continue
        rest = [rc for rc, _ in anchors[2:]]
        members = [x["refcode"] for x in t["other_form"]] + [x["refcode"] for x in t["unknown"]]
        new = []
        for rc in members:
            for a in rest:
                n_cmp += 1
                v = compare(strict, crystal(a), crystal(rc))
                if v is not None and v[0] == SHELL:
                    new.append({"refcode": rc, "anchor": a, "rmsd": round(v[1], 4),
                                "was": "other_form" if any(x["refcode"] == rc for x in t["other_form"]) else "unknown"})
                    break
        every.append({"id": t["id"], "n_anchors": len(anchors), "n_members_checked": len(members), "new_same_form": new})
        print(f"[every] {t['id']}: {len(anchors)} anchors, {len(members)} members, new {len(new)} "
              f"({time.time() - t0:.0f}s)", flush=True)

    lenient = []
    for t in truth:
        if t.get("anchor_tier") != "lenient":
            continue
        q = Crystal.from_string(t18.p1_cif(pats[t["id"]]), "cif")
        rows = []
        for rc in sorted(t["anchors"]):
            single = {f: compare(make((f,)), q, crystal(rc)) for f in FLAGS}
            rows.append({"refcode": rc, "csd_components": components(crystal(rc)),
                         "csd_formula": reader.entry(rc).formula,
                         "single_flag_15": [f for f, v in single.items() if v is not None and v[0] == SHELL],
                         "single_flag": {f: (list(v) if v else None) for f, v in single.items()}})
        lenient.append({"id": t["id"], "opxrd_components": components(q), "h_omitted": t["h_omitted"], "anchors": rows})
        print(f"[lenient] {t['id']}: opXRD {components(q)}; "
              + "; ".join(f"{r['refcode']} {r['csd_components']} -> {r['single_flag_15'] or 'needs >1 flag'}"
                          for r in rows) + f" ({time.time() - t0:.0f}s)", flush=True)

    out = {"every_anchor": {"patterns_checked": len(every), "comparisons": n_cmp,
                            "patterns_with_new_same_form": sum(bool(e["new_same_form"]) for e in every),
                            "rows": every},
           "lenient": {"n_patterns": len(lenient), "rows": lenient}}
    with open("results/anchor_checks.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote results/anchor_checks.json: every-anchor {out['every_anchor']['patterns_with_new_same_form']} of "
          f"{len(every)} patterns gain entries; {len(lenient)} lenient patterns ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
