"""External set (tasks/todo.md C10): Kaduk et al. Powder Diffraction raw profiles as queries.

The measured profiles come from the papers' public Cambridge Core supplementary CIFs (cache/kaduk_supp/) and
the ICDD-deposited donepezil CIF (cache/icdd_donepezil/). Both are licence-restricted: the parsed patterns go
to the gitignored data dir only. Formats match scripts/18 + 19 so scripts/28 scores them unchanged.

  python scripts/37_kaduk_external.py extract           (main interpreter; gemmi)
  <ccdc python> scripts/37_kaduk_external.py truth      (CCDC interpreter; licensed)
  PYTHONPATH=src python scripts/28_rescore.py score --tag main --work cache/real_bench_kaduk \
      --truth <DATA>/kaduk/kaduk_truth.json --patterns <DATA>/kaduk/kaduk_patterns.json --device cuda
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

DATA = "C:/Users/frank/xrd-clip-data"
OUT = f"{DATA}/kaduk"
REFS = "refcodes/kaduk_pdj_refs.json"
SHELL = 15
DONEPEZIL = ("cache/icdd_donepezil/DONEPEZIL-Rietveld-Final.cif", "EDUPUS")
X_TAGS = ("_pd_meas_2theta_scan", "_pd_proc_2theta_corrected", "_pd_meas_2theta")
Y_TAGS = ("_pd_meas_intensity_total", "_pd_meas_counts_total")


def sources() -> "list[tuple[str, str]]":
    """(cif path, refcode) for every candidate profile CIF."""
    refs = json.load(open(REFS))["refs"]
    out = [(f"cache/kaduk_supp/{f}", r["refcode"]) for r in refs for f in r["public_supp_cifs"]]
    return out + [DONEPEZIL]


def num(s: str) -> float:
    """CIF number with or without an esd: '1234(5)' -> 1234.0."""
    return float(re.sub(r"\(.*\)$", "", s))


def profile(block) -> "dict | None":
    """Observed profile of one CIF block, or None if it has none."""
    import numpy as np
    ycol = next((t for t in Y_TAGS if len(block.find_values(t))), None)
    if ycol is None:
        return None
    y = np.array([num(v) for v in block.find_values(ycol)])
    xcol = next((t for t in X_TAGS if len(block.find_values(t)) == len(y)), None)
    if xcol is not None:
        x = np.array([num(v) for v in block.find_values(xcol)])
    else:
        lo, hi, inc = (num(block.find_value(f"_pd_meas_2theta_range_{k}")) for k in ("min", "max", "inc"))
        x = lo + inc * np.arange(len(y))
        assert abs(x[-1] - hi) < 0.5 * inc, (block.name, lo, hi, inc, len(y))
        xcol = "range min/max/inc"
    wl = block.find_value("_diffrn_radiation_wavelength") or block.find_value("_pd_proc_wavelength")
    if not wl:
        wl = next(iter(block.find_values("_diffrn_radiation_wavelength")), None)
    return {"block": block.name, "x_source": xcol, "y_source": ycol, "wavelength": num(wl),
            "two_theta": x.tolist(), "intensity": y.tolist()}


def cmd_extract() -> int:
    import gemmi
    names = {r["refcode"]: r["short"] for r in json.load(open(REFS))["refs"]}
    organic, log = [], []
    for path, rc in sources():
        doc = gemmi.cif.read_string(open(path, "rb").read().decode("latin-1"))
        profs = [p for p in (profile(b) for b in doc) if p is not None]
        if not profs:
            continue
        assert len(profs) == 1, (path, [p["block"] for p in profs])
        p = profs[0]
        pid = f"kaduk_{rc}_{os.path.basename(path).split('.')[0]}"
        tt = p["two_theta"]
        organic.append({"id": pid, "refcode": rc, "short": names[rc], "source": os.path.basename(path), **p,
                        "tt_range": [round(tt[0], 4), round(tt[-1], 4)]})
        log.append(f"  {pid:48s} {names[rc]:40s} wl {p['wavelength']:.6f}  {len(tt):6d} pts "
                   f"{tt[0]:.3f}-{tt[-1]:.3f} deg  x: {p['x_source']}  y: {p['y_source']}")
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/kaduk_patterns.json", "w") as fh:
        json.dump({"organic": organic}, fh)
    import numpy as np                                   # the CCDC interpreter reads library membership from text
    with np.load("cache/sticks_extra/shard_extra.npz", allow_pickle=True) as z:
        open(f"{OUT}/library_extra.txt", "w").write("\n".join(str(x) for x in z["refcode"]) + "\n")
    print("\n".join(log))
    print(f"[extract] {len(organic)} patterns, {len({o['refcode'] for o in organic})} compounds -> {OUT}/kaduk_patterns.json")
    return 0


def cmd_truth() -> int:
    from ccdc.crystal import PackingSimilarity
    from ccdc.io import EntryReader
    from ccdc.search import ReducedCellSearch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from importlib import import_module
    s18 = import_module("18_opxrd_truth_sets")          # family(), stem_of(); same rules as the main set

    pats = json.load(open(f"{OUT}/kaduk_patterns.json"))["organic"]
    in_lib = set(ln.strip() for f in (f"{DATA}/stick_refcodes.txt", f"{OUT}/library_extra.txt")
                 for ln in open(f) if ln.strip())
    assert len(in_lib) == 309_080, len(in_lib)
    reader = EntryReader("CSD")
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

    def meta(rc: str) -> dict:
        e = reader.entry(rc)
        return {"temperature": e.temperature, "is_powder_study": bool(e.is_powder_study),
                "has_disorder": bool(e.has_disorder), "in_cache": rc in in_lib, "formula": e.formula}

    by_rc: dict = {}
    t0 = time.time()
    for rc in sorted({o["refcode"] for o in pats}):
        anchor = reader.entry(rc)
        acry = anchor.crystal
        want = {a.atomic_symbol for a in anchor.molecule.atoms} - {"H"}
        hits = []
        q = ReducedCellSearch.Query(acry.cell_lengths, acry.cell_angles, acry.lattice_centring)
        for hit in ReducedCellSearch(q).search():
            try:
                els = {a.atomic_symbol for a in reader.entry(hit.identifier).molecule.atoms} - {"H"}
            except Exception:
                continue
            if els == want:
                hits.append(hit.identifier)
        members = sorted(set(s18.family(reader, s18.stem_of(rc))) | set(hits) | {rc})
        tier = "strict"
        if compare(strict, acry, acry) is None or compare(strict, acry, acry)[0] < SHELL:
            tier = "lenient"
        ps = strict if tier == "strict" else lenient
        comps = {m: ((SHELL, 0.0) if m == rc else compare(ps, acry, reader.entry(m).crystal)) for m in members}
        # as in scripts/18: other-form and unknown labels only within a same-form stem; a cell hit of another
        # stem that is not 15/15 stays an ordinary distractor
        same = {m: v for m, v in comps.items() if v is not None and v[0] == SHELL}
        stems = {s18.stem_of(m) for m in same}
        fam = {m: v for m, v in comps.items() if s18.stem_of(m) in stems}
        other = {m: v for m, v in fam.items() if v is not None and v[0] < SHELL}
        unknown = sorted(m for m, v in fam.items() if v is None)
        by_rc[rc] = {"tier": tier, "n_cell_hits": len(hits), "cell_hit_stems": sorted({s18.stem_of(h) for h in hits}),
                     "same_form": [{"refcode": m, "n_matched": v[0], "rmsd": round(v[1], 4), "is_anchor": m == rc,
                                    **meta(m)} for m, v in sorted(same.items(), key=lambda kv: kv[1][1])],
                     "other_form": [{"refcode": m, "n_matched": v[0], "rmsd": round(v[1], 4), **meta(m)}
                                    for m, v in sorted(other.items())],
                     "unknown": [{"refcode": m, **meta(m)} for m in unknown]}
        d = by_rc[rc]
        sc = [x["refcode"] for x in d["same_form"] if not x["is_powder_study"]]
        print(f"{rc}: {tier}; cell hits {len(hits)}; members {len(members)}; same {len(d['same_form'])} "
              f"(sc {len(sc)}, in library {sum(x in in_lib for x in sc)}: {sc}); other {len(d['other_form'])}; "
              f"unknown {len(unknown)}; anchor powder study {d['same_form'][0]['is_powder_study']} "
              f"({time.time() - t0:.0f}s)", flush=True)

    out = []
    for o in pats:
        d = by_rc[o["refcode"]]
        out.append({"id": o["id"], "h_omitted": False, "wavelength": o["wavelength"], "anchor_tier": d["tier"],
                    "true_stems": [s18.stem_of(o["refcode"])], "n_cell_hits": d["n_cell_hits"],
                    "cell_hit_stems": d["cell_hit_stems"],
                    "anchors": {o["refcode"]: {"n_matched": SHELL, "rmsd": 0.0}},
                    "same_form": d["same_form"], "other_form": d["other_form"], "unknown": d["unknown"]})
    with open(f"{OUT}/kaduk_truth.json", "w") as fh:
        json.dump(out, fh, indent=1)
    dropped = sorted({x["refcode"] for r in out for k in ("same_form", "other_form") for x in r[k] if not x["in_cache"]})
    print(f"[truth] {len(out)} patterns -> {OUT}/kaduk_truth.json; same/other-form entries outside the library "
          f"(dropped, not simulated): {len(dropped)} {dropped}")
    return 0


def cmd_summary() -> int:
    """results/kaduk_summary.json: per-pattern metadata and ranks (refcodes and numbers only, no profile data)."""
    import numpy as np
    sys.path.insert(0, "src")
    from xrdclip.eval.real_patterns import to_cu_grid
    pats = json.load(open(f"{OUT}/kaduk_patterns.json"))["organic"]
    truth = {t["id"]: t for t in json.load(open(f"{OUT}/kaduk_truth.json"))}
    in_lib = set(ln.strip() for f in (f"{DATA}/stick_refcodes.txt", f"{OUT}/library_extra.txt")
                 for ln in open(f) if ln.strip())
    z = np.load("cache/real_bench_kaduk/rescore_main/ranks.npz")
    doi = {r["refcode"]: r["doi"] for r in json.load(open(REFS))["refs"]}
    rows = []
    for i, o in enumerate(pats):
        t = truth[o["id"]]
        _, m = to_cu_grid(o["two_theta"], o["intensity"], o["wavelength"])
        same = t["same_form"]
        rows.append({"id": o["id"], "refcode": o["refcode"], "compound": o["short"], "source": o["source"],
                     "doi": doi[o["refcode"]],
                     "wavelength": o["wavelength"], "kind": "lab Cu" if abs(o["wavelength"] - 1.54059) < 0.003 else "synchrotron",
                     "tt_range": o["tt_range"], "cu_grid_coverage": round(float(m.mean()), 3),
                     "n_same_form": len(same), "n_same_form_in_library": sum(x["refcode"] in in_lib for x in same),
                     "sc_in_library": [x["refcode"] for x in same if not x["is_powder_study"] and x["refcode"] in in_lib],
                     "sc_outside_library": [x["refcode"] for x in same if not x["is_powder_study"] and x["refcode"] not in in_lib],
                     "ranks": {k: int(z[k][i]) for k in z.files if "@" not in k}})
    out = {"description": "Kaduk external set (tasks/todo.md C10): 19 measured profiles from the papers' supplementary "
                          "CIFs, scored by scripts/28 config main against the 309,080-entry library. Rank = wrong "
                          "entries above the best correct one (0 = first place, -1 = no correct entry in the library).",
           "n_patterns": len(rows), "n_compounds": len({r["refcode"] for r in rows}),
           "n_sc_compounds": len({r["refcode"] for r in rows if r["sc_in_library"]}),
           "sc_entries_outside_library": sorted({x for r in rows for x in r["sc_outside_library"]}),
           "patterns": rows}
    with open("results/kaduk_summary.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[summary] {out['n_patterns']} patterns, {out['n_compounds']} compounds, sc compounds {out['n_sc_compounds']}; "
          f"sc entries outside the library {out['sc_entries_outside_library']}")
    return 0


if __name__ == "__main__":
    sys.exit({"extract": cmd_extract, "truth": cmd_truth, "summary": cmd_summary}[sys.argv[1]]())
