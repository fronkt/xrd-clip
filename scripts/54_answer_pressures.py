"""CSD pressure and temperature of every single-crystal answer entry (claim audit 2026-11-20, finding 3).

Run with the CCDC interpreter (licensed):

  <ccdc python> scripts/54_answer_pressures.py      -> results/answer_pressures.json

The ambient single-crystal set is defined by the query's study (scripts/34: not part of a variable-pressure study).
This script checks the other side: whether the single-crystal same-form entries that count as correct answers carry
a CSD pressure. For each of the 60 single-crystal patterns it lists its corrected answer
entries (cache/critic2/plan.json) with their CSD pressure and temperature. Output holds refcodes and conditions only, no structures.
"""
from __future__ import annotations

import json

from ccdc import io

reader = io.EntryReader("CSD")


def main() -> int:
    # the corrected single-crystal answer sets (label corrections and compound merges applied), as scripts/29 prep
    # wrote them for all 60 sc patterns; the raw truth file would miss pattern_100, whose answers come from a merge
    plan = json.load(open("cache/critic2/plan.json"))
    out, cache = {}, {}
    for pid, pl in plan.items():
        sc = sorted(pl["truth"])
        rows = []
        for rc in sc:
            if rc not in cache:
                try:
                    e = reader.entry(rc)
                    cache[rc] = {"pressure": e.pressure or None, "temperature": e.temperature or None}
                except RuntimeError:
                    cache[rc] = {"pressure": None, "temperature": None, "missing": True}
            rows.append({"refcode": rc, **cache[rc]})
        out[pid] = {"answers": rows, "n": len(rows), "n_with_pressure": sum(bool(r["pressure"]) for r in rows)}
    json.dump(out, open("results/answer_pressures.json", "w"), indent=1)
    n_any = sum(v["n_with_pressure"] > 0 for v in out.values())
    n_all = sum(v["n_with_pressure"] == v["n"] for v in out.values())
    print(f"wrote results/answer_pressures.json: {len(out)} patterns with single-crystal answers; "
          f"{n_any} have at least one answer with a CSD pressure, {n_all} have only such answers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
