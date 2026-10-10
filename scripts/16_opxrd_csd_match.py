"""Phase 2: match opXRD organic phases to CSD refcodes (run with the CCDC miniconda interpreter).

For each primitive Niggli cell from scripts/15_..., run a CSD reduced-cell search (1.5 % lengths,
2 deg angles, the CCDC defaults) and keep hits whose non-H element set equals the opXRD phase's
non-H element set (powder refinements often omit H). Among the survivors, prefer entries that are in
the xrd-clip corpus, then the smallest cell mismatch. Records every candidate so the choice can be
audited, plus the CSD temperature (the thermal gap to the room-temperature pattern matters).

  <ccdc python> scripts/16_opxrd_csd_match.py <reduced_cells.json> <corpus_refcodes.txt> <out.json>
"""
from __future__ import annotations

import json
import sys

from ccdc.io import EntryReader
from ccdc.search import ReducedCellSearch


def cell_mismatch(a, b):
    """Max relative length difference + max angle difference / 100 (a crude, monotone score)."""
    dl = max(abs(x - y) / y for x, y in zip(a[:3], b[:3]))
    da = max(abs(x - y) for x, y in zip(a[3:], b[3:]))
    return dl + da / 100.0


def main() -> int:
    cells_path, corpus_path, out = sys.argv[1:4]
    with open(cells_path) as fh:
        rows = json.load(fh)
    with open(corpus_path) as fh:
        corpus = set(ln.strip() for ln in fh if ln.strip())
    reader = EntryReader("CSD")
    results = []
    for r in rows:
        want = set(r["elements"]) - {"H"}
        c = r["cell_reduced"]
        search = ReducedCellSearch(ReducedCellSearch.Query(c[:3], c[3:], "P"))
        cands = []
        for hit in search.search():
            try:
                e = reader.entry(hit.identifier)
                cry = e.crystal
                els = {a.atomic_symbol for a in e.molecule.atoms} - {"H"}
                rc = cry.reduced_cell
                hit_cell = [*rc.cell_lengths, *rc.cell_angles]
            except Exception:
                continue
            if els != want:
                continue
            cands.append({"refcode": hit.identifier, "in_corpus": hit.identifier in corpus,
                          "mismatch": cell_mismatch(hit_cell, c), "temperature": e.temperature,
                          "is_powder_study": bool(e.is_powder_study), "formula": e.formula})
        cands.sort(key=lambda x: (not x["in_corpus"], x["mismatch"]))
        results.append({"id": r["id"], "elements": r["elements"], "h_omitted": r["h_omitted"],
                        "wavelength": r["wavelength"], "n_candidates": len(cands),
                        "best": cands[0] if cands else None, "candidates": cands[:10]})
        print(r["id"], len(cands), cands[0]["refcode"] if cands else "-", flush=True)
    with open(out, "w") as fh:
        json.dump(results, fh, indent=1)
    n = len(results)
    matched = [x for x in results if x["best"]]
    inc = [x for x in matched if x["best"]["in_corpus"]]
    print(f"\n{n} patterns; {len(matched)} matched to a CSD entry; {len(inc)} to a corpus entry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
