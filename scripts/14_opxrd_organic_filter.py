"""Phase 2 Tier A: pull the molecular-organic patterns out of opXRD's CNRS (COD) subset.

opXRD (Zenodo 10.5281/zenodo.15298026, CC BY 4.0) does not report how many of its 1,052 CNRS patterns
are organic, so we count them. A pattern counts as molecular-organic when its labelled phase contains
C and H and nothing outside {C,H,N,O,S,P,F,Cl,Br,I,B,Si}. Single-phase only (a mixture has no single
CSD target).

  python -I scripts/14_opxrd_organic_filter.py /root/opxrd/CNRS results/opxrd_cnrs_organic.json

Writes the summary of ALL CNRS entries (element sets, phase count, wavelength) and the full data of
the organic ones (2theta, intensity, lattice, atom basis) for CSD matching (scripts/15_...).
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import Counter

ORGANIC_OK = {"C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si"}


def element(symbol: str) -> str:
    m = re.match(r"[A-Z][a-z]?", symbol.strip())
    return m.group(0) if m else symbol


def main() -> int:
    src, out = sys.argv[1], sys.argv[2]
    summary, organic = [], []
    for path in sorted(glob.glob(os.path.join(src, "pattern_*.json")),
                       key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0])):
        with open(path) as fh:
            d = json.load(fh)
        label = json.loads(d["label"]) if d.get("label") else {}
        phases = [json.loads(p) for p in label.get("phases", [])]
        xinfo = json.loads(label["xray_info"]) if label.get("xray_info") else {}
        rec = {"id": os.path.basename(path), "n_phases": len(phases),
               "wavelength": xinfo.get("primary_wavelength"), "temp_K": label.get("temp_K"),
               "n_points": len(d["two_theta_values"]),
               "tt_range": [d["two_theta_values"][0], d["two_theta_values"][-1]] if d["two_theta_values"] else None}
        if len(phases) == 1 and phases[0].get("basis"):   # some entries carry a cell but no atoms
            basis = [json.loads(b) for b in json.loads(phases[0]["basis"])]
            comp = Counter(element(b["symbol"]) for b in basis)
            rec["elements"] = sorted(comp)
            rec["n_basis"] = len(basis)
            # Powder refinements often omit H atoms, so C-with-no-H non-metal phases are kept too
            # (flagged h_omitted) -- they are almost always organics with unrefined hydrogens.
            rec["h_omitted"] = "H" not in comp
            rec["organic"] = bool(basis) and "C" in comp and set(comp) <= ORGANIC_OK
            if rec["organic"]:
                organic.append({**rec, "lattice": phases[0]["lattice"], "spacegroup": phases[0]["spacegroup"],
                                "basis": [{k: b[k] for k in ("x", "y", "z", "occupancy", "symbol")} for b in basis],
                                "two_theta": d["two_theta_values"], "intensity": d["intensities"]})
        summary.append(rec)
    n_single = sum(r["n_phases"] == 1 for r in summary)
    no_h = sum(1 for r in summary if r.get("elements") and "C" in r["elements"] and "H" not in r["elements"]
               and set(r["elements"]) <= ORGANIC_OK)
    stats = {"n_total": len(summary), "n_single_phase": n_single, "n_organic": len(organic),
             "n_C_but_no_H_nonmetal": no_h,
             "wavelengths_organic": Counter(str(o["wavelength"]) for o in organic)}
    with open(out, "w") as fh:
        json.dump({"stats": stats, "summary": summary, "organic": organic}, fh)
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
