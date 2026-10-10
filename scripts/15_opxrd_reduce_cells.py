"""Phase 2: reduce each opXRD organic phase to its primitive Niggli cell (main interpreter, pymatgen).

opXRD gives the cell and the atom basis but no space group or centring, so the cell as given may be a
centred conventional cell. Building the structure and taking its primitive Niggli-reduced lattice
gives a centring-free cell that the CSD reduced-cell search (scripts/16_..., CCDC interpreter) can
look up with lattice_centring = P.

  python -I scripts/15_opxrd_reduce_cells.py <opxrd_cnrs_organic.json> <out.json>
"""
from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter

from pymatgen.core import Lattice, Structure


def main() -> int:
    src, out = sys.argv[1], sys.argv[2]
    with open(src) as fh:
        organic = json.load(fh)["organic"]
    rows = []
    for o in organic:
        abc = ast.literal_eval(o["lattice"])
        lat = Lattice.from_parameters(*abc)
        species = [re.match(r"[A-Z][a-z]?", b["symbol"]).group(0) for b in o["basis"]]
        coords = [[b["x"], b["y"], b["z"]] for b in o["basis"]]
        row = {"id": o["id"], "cell_given": abc, "elements": o["elements"], "h_omitted": o["h_omitted"],
               "n_basis": len(species), "formula_cell": dict(Counter(species)), "wavelength": o["wavelength"]}
        try:
            s = Structure(lat, species, coords, coords_are_cartesian=False)
            prim = s.get_primitive_structure(tolerance=0.1)
            red = prim.lattice.get_niggli_reduced_lattice()
            row["cell_reduced"] = [*red.abc, *red.angles]
            row["n_prim"] = len(prim)
            row["vol_reduced"] = red.volume
        except Exception as e:   # some bases have overlapping/partial sites; fall back to the given cell
            red = lat.get_niggli_reduced_lattice()
            row["cell_reduced"] = [*red.abc, *red.angles]
            row["vol_reduced"] = red.volume
            row["prim_error"] = str(e)[:120]
        rows.append(row)
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"{len(rows)} cells reduced; {sum('prim_error' in r for r in rows)} fell back to the given cell")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
