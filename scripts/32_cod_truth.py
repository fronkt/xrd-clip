"""R5 ground truth on the open stand-in: which COD entries are the same solid form as each opXRD pattern.

Run with the CCDC miniconda interpreter (PackingSimilarity only reads the CIFs; no CSD entry is used):

  <ccdc python> scripts/32_cod_truth.py

Same rules as scripts/18 (CSD), adapted to COD, which has no refcode families:
  1. Candidates: scripts/31 `cands` (same element set ignoring H, primitive reduced cell within 5 % / volume within
     6 %; looser than 18's 1.5 % because there is no family enumeration to catch temperature series).
  2. Anchors: PackingSimilarity (15-molecule shell, H positions ignored) of the opXRD P1 structure against every
     candidate, strict tier; if none reaches 15/15, the lenient tier (bond types/counts, H counts and the smallest
     components ignored).
  3. Same-form set: the two closest anchors against every other candidate, COD vs COD, strict tier (18's family
     step). 15/15 = same form; < 15 = other form; no comparison possible = unknown (ignored when ranking).
  4. Powder or single crystal: COD's `method` field, else any `_pd_` tag in the CIF. The pattern's own entry is a
     powder anchor reproducing the opXRD structure (rmsd < 0.05 A).

Writes C:/Users/frank/xrd-clip-data/cod/cod_truth.json (COD ids + comparison metadata; COD is CC0).
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys
import time
from importlib import import_module

from ccdc.crystal import Crystal, PackingSimilarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
t18 = import_module("18_opxrd_truth_sets")          # p1_cif, element (its main() is not run)

DATA = "C:/Users/frank/xrd-clip-data"
COD = f"{DATA}/cod"
SHELL = 15


def meta_rows(ids: "set[str]") -> dict:
    rows = {}
    for p in sorted(glob.glob(f"{COD}/meta/v_*.csv")):
        with open(p, encoding="utf-8", errors="replace") as fh:
            for r in csv.DictReader(ln for ln in fh if not ln.startswith("#")):
                if r["file"] in ids:
                    rows[r["file"]] = r
    return rows


def is_powder(cid: str, row: dict, cif_text: str) -> bool:
    m = (row.get("method") or "").lower()
    if m:
        return m.startswith("powder")
    return bool(re.search(r"^\s*_pd_", cif_text, re.M))


def main() -> int:
    pats = {o["id"]: o for o in json.load(open(f"{DATA}/opxrd/opxrd_cnrs_organic.json"))["organic"]}
    cands = json.load(open(f"{COD}/candidates.json"))
    ids = {c for v in cands.values() for c in v}
    rows = meta_rows(ids)
    strict, lenient = PackingSimilarity(), PackingSimilarity()
    for ps in (strict, lenient):
        ps.settings.packing_shell_size = SHELL
        ps.settings.ignore_hydrogen_positions = True
        ps.settings.allow_molecular_differences = False
        ps.settings.timeout_ms = 30000
    for flag in ("ignore_bond_types", "ignore_bond_counts", "ignore_hydrogen_counts", "ignore_smallest_components"):
        setattr(lenient.settings, flag, True)

    def compare(ps, a, b):
        try:
            r = ps.compare(a, b)
            return (r.nmatched_molecules, float(r.rmsd)) if r is not None else None
        except Exception:
            return None

    cache: dict = {}

    def crystal(cid: str):
        if cid not in cache:
            p = f"{COD}/cif/{cid}.cif"
            txt = open(p, encoding="utf-8", errors="replace").read() if os.path.exists(p) else ""
            try:
                cache[cid] = (Crystal.from_string(txt, "cif") if txt else None, txt)
            except Exception:
                cache[cid] = (None, txt)
        return cache[cid]

    out, t0 = [], time.time()
    for n, (pid, o) in enumerate(sorted(pats.items(), key=lambda kv: int(re.findall(r"\d+", kv[0])[0]))):
        hits = cands.get(pid, [])
        cry = Crystal.from_string(t18.p1_cif(o), "cif")
        anchors, tier = {}, None
        for name, ps in (("strict", strict), ("lenient", lenient)):
            direct = {c: compare(ps, cry, crystal(c)[0]) for c in hits if crystal(c)[0] is not None}
            anchors = {c: v for c, v in direct.items() if v is not None and v[0] == SHELL}
            if anchors:
                tier = name
                break
        reps = [c for c, _ in sorted(anchors.items(), key=lambda kv: kv[1][1])[:2]]
        comps = {}
        for c in hits:
            if c in anchors:
                comps[c] = anchors[c]
                continue
            other = crystal(c)[0]
            best = None
            for a in reps:
                v = compare(strict, crystal(a)[0], other) if other is not None else None
                if v is not None and (best is None or (v[0], -v[1]) > (best[0], -best[1])):
                    best = v
                if best is not None and best[0] == SHELL:
                    break
            comps[c] = best

        def info(c):
            r = rows.get(c, {})
            return {"cod_id": c, "is_powder_study": is_powder(c, r, crystal(c)[1]),
                    "temperature": r.get("diffrtemp") or r.get("celltemp") or None,
                    "has_disorder": "has disorder" in (r.get("flags") or ""), "formula": r.get("formula")}

        same = {c: v for c, v in comps.items() if v is not None and v[0] == SHELL}
        oth = {c: v for c, v in comps.items() if v is not None and v[0] < SHELL}
        unknown = sorted(c for c, v in comps.items() if v is None)
        rec = {"id": pid, "h_omitted": o["h_omitted"], "wavelength": o["wavelength"], "n_candidates": len(hits),
               "anchor_tier": tier,
               "anchors": {c: {"n_matched": v[0], "rmsd": round(v[1], 4)} for c, v in anchors.items()},
               "same_form": [{**info(c), "n_matched": v[0], "rmsd": round(v[1], 4), "is_anchor": c in anchors}
                             for c, v in sorted(same.items(), key=lambda kv: kv[1][1])],
               "other_form": [{**info(c), "n_matched": v[0], "rmsd": round(v[1], 4)} for c, v in sorted(oth.items())],
               "unknown": [info(c) for c in unknown]}
        out.append(rec)
        print(f"[{n + 1}/{len(pats)}] {pid}: candidates {len(hits)} anchors {len(anchors)} ({tier}) same {len(same)} "
              f"other {len(oth)} unknown {len(unknown)} ({time.time() - t0:.0f}s)", flush=True)
        if n % 10 == 0:
            with open(f"{COD}/cod_truth.partial.json", "w") as fh:
                json.dump(out, fh, indent=1)
    with open(f"{COD}/cod_truth.json", "w") as fh:
        json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
