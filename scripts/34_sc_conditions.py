"""Measurement conditions of the single-crystal (sc) benchmark patterns (10-14 internal referee check).

Run with the CCDC interpreter (licensed):

  <ccdc python> scripts/34_sc_conditions.py      -> results/sc_conditions.json

For each pattern whose answer set has a single-crystal entry: its own powder-study entry (the powder-study anchor
closest to the deposited structure, as in scripts/22 and 29), that entry's CSD pressure, temperature and
publication DOI, and whether it belongs to a variable-pressure study. A pattern is tagged "pressure_series" if its
own entry carries a CSD pressure, or another entry of its refcode family from the same publication does (the CSD
leaves the field empty on some entries of a pressure series, e.g. the lowest-pressure point), or it is flagged by
hand in MANUAL with the evidence. The rule was fixed
after the referee pointed at compression series, so every split by it is a secondary analysis. Output holds
refcodes, conditions and DOIs only, no structures.
"""
from __future__ import annotations

import json
import re
import sys

from ccdc import io

DATA = "C:/Users/frank/xrd-clip-data"
reader = io.EntryReader("CSD")
# Entries of a pressure study that the CSD does not mark (no pressure on any entry of the publication), flagged by
# hand with the evidence: HOJQII01/02 (J. Am. Chem. Soc. 2014, doi 10.1021/ja502753t; COD 1516256/1516258, "Heat,
# Pressure and Light-Induced Interconversion of Bisdithiazolyl Radicals and Dimers") have a = 4.12 and 3.87 A at
# 293 K against 4.55 A for HOJQII at 100 K, a compression no heating produces.
MANUAL = {"HOJQII01": "compressed cell at 293 K (a 4.12 vs 4.55 A at 100 K), pressure study",
          "HOJQII02": "compressed cell at 293 K (a 3.87 vs 4.55 A at 100 K), pressure study"}


def own_entry(t, in_lib=None):
    ps = [x["refcode"] for x in t["same_form"] if x["is_powder_study"] and x["is_anchor"]]
    return min(ps, key=lambda rc: (t["anchors"][rc]["rmsd"], rc)) if ps else None


def info(rc):
    e = reader.entry(rc)
    return {"pressure": e.pressure, "temperature": e.temperature,
            "doi": (e.publication.doi or "").lower() if e.publication else ""}


def family(stem: str) -> list:
    """Refcodes of a CSD family: the stem and stem01..stem99 that exist (the API has no family lookup)."""
    out = []
    for rc in [stem] + [f"{stem}{i:02d}" for i in range(1, 100)]:
        try:
            reader.entry(rc)
            out.append(rc)
        except RuntimeError:
            continue
    return out


def main() -> int:
    truth = {t["id"].replace(".json", ""): t for t in json.load(open(f"{DATA}/opxrd/opxrd_truth.json"))}
    plan = json.load(open("cache/critic2/plan.json"))          # the sc patterns (scripts/29 prep)
    out = {}
    fam_cache = {}
    for pid in sorted(plan):
        t = truth[pid]
        own = own_entry(t)
        row = {"stem": plan[pid]["stem"], "own": own}
        if own:
            row.update(info(own))
            stem = re.match(r"[A-Z]+", own).group(0)
            if stem not in fam_cache:
                fam_cache[stem] = {rc: info(rc) for rc in family(stem)}
            same_pub = [rc for rc, v in fam_cache[stem].items() if row["doi"] and v["doi"] == row["doi"]]
            row["same_publication_with_pressure"] = sorted(rc for rc in same_pub if fam_cache[stem][rc]["pressure"])
            row["manual_flag"] = MANUAL.get(own)
            row["pressure_series"] = bool(row["pressure"] or row["same_publication_with_pressure"] or row["manual_flag"])
        else:
            row["pressure_series"] = False
        out[pid] = row
        print(pid, row["stem"], own, row.get("pressure"), row.get("temperature"), row["pressure_series"], flush=True)
    with open("results/sc_conditions.json", "w") as fh:
        json.dump(out, fh, indent=1)
    n = sum(v["pressure_series"] for v in out.values())
    print(f"wrote results/sc_conditions.json: {n} of {len(out)} sc patterns from pressure series "
          f"({len({v['stem'] for v in out.values() if v['pressure_series']})} compounds)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
