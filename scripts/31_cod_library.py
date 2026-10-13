"""R5: an open stand-in for the benchmark library from the Crystallography Open Database (tasks/todo.md Step 3).

Digital Discovery asks for a public "representative dataset that enables similar conclusions" when the main data
are restricted (the CSD). opXRD's CNRS patterns were extracted from COD's own powder entries (Hollarek et al.,
arXiv:2503.05577), so COD (CC0) can host the whole benchmark: the same queries against simulated COD entries.

  meta    COD metadata of every carbon-containing entry (REST CSV, volume bins)   -> <data>/cod/meta/*.csv
  select  corpus filter mirroring the CSD corpus (stage 1) -> <data>/cod/library_ids.txt (+ filter counts)
  cands / fetch_cifs   candidate COD entries per opXRD pattern and their CIFs (for scripts/32's truth)
  paths / shards       (box) rsync list, then CIFs -> stage-1 records for the unchanged stick simulator
  truth                scripts/32's same-form sets -> 19's truth schema (<data>/cod/opxrd_truth_cod.json)
Scoring: XRD_STICK_DIRS=cod/sticks python scripts/28_rescore.py score|eval --tag main --work cache/real_bench_cod
  --truth <data>/cod/opxrd_truth_cod.json --base ""   (model column: library embeddings in the same --work)
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
import time
import urllib.request

DATA = "C:/Users/frank/xrd-clip-data/cod"
REST = "https://www.crystallography.net/cod/result?format=csv&el1=C&vmin={lo}&vmax={hi}"
# the CSD corpus filter (xrdclip.data.csd_extract.passes): elements, R <= 7.5 %, ordered, Z' 1-2
ALLOWED_ELEMENTS = {"H", "C", "N", "O", "F", "S", "Cl", "Br", "I", "P"}
MAX_R = 0.075
UA = "Mozilla/5.0 (compatible; xrd-clip-benchmark)"


def volume_bins():
    edges = [0, 100, 150, 200, 250, 300] + list(range(400, 4001, 100)) + list(range(4250, 8001, 250)) + list(range(9000, 20001, 1000))
    return list(zip(edges[:-1], edges[1:])) + [(20000, 10_000_000)]


def fetch(url: str, path: str, tries: int = 4) -> int:
    for t in range(tries):
        try:
            # the server stalls Python's default user agent (found 10-12); any explicit one works
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=600) as r:
                body = r.read()
            with open(path + ".tmp", "wb") as fh:
                fh.write(body)
            os.replace(path + ".tmp", path)
            return body.count(b'\n"')
        except Exception as exc:                       # connection resets happen; back off and retry
            print(f"[meta] retry {t + 1} {url}: {exc}", flush=True)
            time.sleep(10 * (t + 1))
    raise RuntimeError(f"failed: {url}")


def cmd_meta(args) -> int:
    out = os.path.join(DATA, "meta")
    os.makedirs(out, exist_ok=True)
    total = 0
    for lo, hi in volume_bins():
        path = os.path.join(out, f"v_{lo:08d}_{hi:08d}.csv")
        if os.path.exists(path):
            continue
        try:
            n = fetch(REST.format(lo=lo, hi=hi), path)
        except RuntimeError as exc:                    # recorded and skipped; rerun `meta` to retry
            print(f"[meta] FAILED {lo}-{hi}: {exc}", flush=True)
            continue
        total += n
        print(f"[meta] {lo}-{hi} A^3: {n} rows (total {total})", flush=True)
        time.sleep(2)
    return 0


def read_meta():
    """Every row of the metadata CSVs (bins share edges, so a file can repeat: dedup on the COD id)."""
    rows = {}
    for p in sorted(glob.glob(os.path.join(DATA, "meta", "v_*.csv"))):
        with open(p, encoding="utf-8", errors="replace") as fh:
            lines = [ln for ln in fh if not ln.startswith("#")]
        for r in csv.DictReader(lines):
            rows[r["file"]] = r
    return rows


def elements(formula: str) -> "set[str]":
    return {m.group(1) for m in re.finditer(r"([A-Z][a-z]?)\s*[0-9.]*", formula or "")}


def r_factor(r: dict) -> "float | None":
    for k in ("Robs", "RI", "Rall"):
        try:
            v = float(r.get(k) or "nan")
        except ValueError:
            continue
        if v == v:
            return v
    return None


def corpus_reason(r: dict) -> str:
    """'' if the entry passes the corpus filter, else the first failing rule (same order as the CSD filter)."""
    el = elements(r["formula"])
    if "C" not in el:
        return "not_organic"
    if "has coordinates" not in (r.get("flags") or ""):
        return "no_3d"
    if "has disorder" in (r.get("flags") or ""):
        return "disorder"
    rf = r_factor(r)
    if rf is None or rf > MAX_R:
        return "r_factor"
    try:
        zp = float(r.get("Zprime") or "nan")
    except ValueError:
        zp = float("nan")
    if not (1.0 <= zp <= 2.0):
        return "z_prime"
    if not el <= ALLOWED_ELEMENTS:
        return "element"
    if r.get("status") in ("errors", "retracted") or r.get("onhold"):
        return "status"
    return ""


def cmd_select(args) -> int:
    from collections import Counter
    rows = read_meta()
    why = Counter()
    keep = []
    for cid, r in rows.items():
        reason = corpus_reason(r)
        why[reason or "kept"] += 1
        if not reason:
            keep.append(cid)
    with open(os.path.join(DATA, "library_ids.txt"), "w") as fh:
        fh.write("\n".join(sorted(keep)) + "\n")
    summary = {"n_meta_rows": len(rows), "filter": dict(why), "n_library": len(keep),
               "rule": "C-containing COD entries (REST, duplicates/errors/retracted/theoretical excluded by COD); "
                       "elements within H C N O F S Cl Br I P; has coordinates; no disorder flag; R <= 0.075 "
                       "(Robs, else RI, else Rall); Z' 1-2; not on hold"}
    with open(os.path.join(DATA, "library_select.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------
# cands: COD entries that could be each query's solid form (cell + elements), decided by scripts/32
# ------------------------------------------------------------------------------------------------
# conventional -> primitive basis (rows = primitive vectors in conventional fractional coordinates)
CENTRING = {
    "P": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "A": [[1, 0, 0], [0, .5, -.5], [0, .5, .5]],
    "B": [[.5, 0, -.5], [0, 1, 0], [.5, 0, .5]],
    "C": [[.5, .5, 0], [-.5, .5, 0], [0, 0, 1]],
    "I": [[-.5, .5, .5], [.5, -.5, .5], [.5, .5, -.5]],
    "F": [[0, .5, .5], [.5, 0, .5], [.5, .5, 0]],
    "R": [[2 / 3, 1 / 3, 1 / 3], [-1 / 3, 1 / 3, 1 / 3], [-1 / 3, -2 / 3, 1 / 3]],   # hexagonal axes, obverse
}


def reduced_cell(r: dict) -> "tuple[list, float] | None":
    """Primitive Niggli cell (a, b, c, alpha, beta, gamma) and volume of a COD metadata row."""
    import numpy as np
    from pymatgen.core import Lattice
    try:
        lat = Lattice.from_parameters(*(float(r[k]) for k in ("a", "b", "c", "alpha", "beta", "gamma")))
        hall = (r.get("sgHall") or "").lstrip("-").strip()
        cen = hall[:1].upper() if hall[:1].upper() in CENTRING else "P"
        if cen == "R" and abs(lat.gamma - 120) > 1:     # rhombohedral axes: already primitive
            cen = "P"
        prim = Lattice(np.array(CENTRING[cen]) @ lat.matrix)
        red = prim.get_niggli_reduced_lattice()
        return [*red.abc, *red.angles], red.volume
    except Exception:
        return None


def cmd_cands(args) -> int:
    """Candidates per query: same element set ignoring H, primitive reduced volume within 6 %, sorted reduced
    lengths within 5 %. Deliberately loose (temperature series move cells by up to ~3 %): packing similarity
    (scripts/32, CCDC) decides, as in scripts/18. Uses every C-containing COD row, not only the corpus."""
    import numpy as np
    rows = read_meta()
    q = json.load(open(os.path.join(os.path.dirname(DATA), "opxrd", "opxrd_reduced_cells.json")))
    idx = []
    for cid, r in rows.items():
        rc = reduced_cell(r)
        if rc is None:
            continue
        idx.append((cid, frozenset(elements(r["formula"]) - {"H"}), rc[1], sorted(rc[0][:3])))
    print(f"[cands] {len(idx):,} COD rows with a reduced cell", flush=True)
    vol = np.array([x[2] for x in idx])
    out, need = {}, set()
    for o in q:
        want = frozenset(set(o["elements"]) - {"H"})
        v, L = o["vol_reduced"], sorted(o["cell_reduced"][:3])
        hit = []
        for j in np.nonzero(np.abs(vol / v - 1) <= 0.06)[0]:
            cid, els, _, l2 = idx[j]
            if els == want and all(abs(x / y - 1) <= 0.05 for x, y in zip(l2, L)):
                hit.append(cid)
        out[o["id"]] = hit
        need |= set(hit)
    with open(os.path.join(DATA, "candidates.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    n = [len(v) for v in out.values()]
    print(f"[cands] {len(out)} patterns; candidates per pattern median {np.median(n):.0f}, max {max(n)}, "
          f"none for {sum(x == 0 for x in n)}; {len(need)} distinct COD ids")
    return 0


def cmd_fetch_cifs(args) -> int:
    """CIFs of every candidate (for the CCDC packing comparison), one HTTP request each, politely paced."""
    cands = json.load(open(os.path.join(DATA, "candidates.json")))
    out = os.path.join(DATA, "cif")
    os.makedirs(out, exist_ok=True)
    ids = sorted({c for v in cands.values() for c in v})
    todo = [c for c in ids if not os.path.exists(os.path.join(out, f"{c}.cif"))]
    print(f"[fetch] {len(ids)} candidates, {len(todo)} to download", flush=True)
    for i, c in enumerate(todo):
        for t in range(4):
            try:
                req = urllib.request.Request(f"https://www.crystallography.net/cod/{c}.cif", headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=120) as r:
                    body = r.read()
                with open(os.path.join(out, f"{c}.cif"), "wb") as fh:
                    fh.write(body)
                break
            except Exception as exc:
                print(f"[fetch] retry {c}: {exc}", flush=True)
                time.sleep(5 * (t + 1))
        if i % 100 == 0:
            print(f"[fetch] {i + 1}/{len(todo)}", flush=True)
        time.sleep(0.2)
    return 0


# ------------------------------------------------------------------------------------------------
# box side: rsync file list, then CIF -> stage-1-style records for the unchanged stick simulator
# ------------------------------------------------------------------------------------------------
def cod_path(cid: str) -> str:
    """COD's rsync layout: cif/1/54/42/1544226.cif."""
    return f"{cid[0]}/{cid[1:3]}/{cid[3:5]}/{cid}.cif"


def cmd_paths(args) -> int:
    """library = corpus filter + every query candidate (the 'extras', as the CSD library added target families)."""
    rows = read_meta()
    lib = set(open(os.path.join(DATA, "library_ids.txt")).read().split())
    cands = json.load(open(os.path.join(DATA, "candidates.json")))
    extra = {c for v in cands.values() for c in v} - lib
    ids = sorted(lib | extra)
    with open(os.path.join(DATA, "library_all_ids.txt"), "w") as fh:
        fh.write("\n".join(ids) + "\n")
    with open(os.path.join(DATA, "rsync_paths.txt"), "w") as fh:
        fh.write("\n".join(cod_path(c) for c in ids) + "\n")
    meta = {c: {k: rows[c].get(k) for k in ("a", "b", "c", "alpha", "beta", "gamma", "sg", "formula")} for c in ids}
    with open(os.path.join(DATA, "library_meta.json"), "w") as fh:
        json.dump(meta, fh)
    print(f"[paths] corpus {len(lib):,} + candidate extras {len(extra):,} = {len(ids):,} COD entries")
    return 0


def cmd_shards(args) -> int:
    """(box) downloaded CIFs -> jsonl.gz shards in the stage-1 record format that xrdclip.data.simulate reads.
    n_atoms = atom-site rows of the CIF (the asymmetric unit, as CCDC's crystal.molecule counts for the CSD)."""
    import gzip as gz
    meta = json.load(open(args.meta))
    ids = sorted(meta)
    os.makedirs(args.out, exist_ok=True)
    per = (len(ids) + args.n_shards - 1) // args.n_shards
    missing = 0
    for s in range(args.n_shards):
        with gz.open(os.path.join(args.out, f"shard_{s:03d}.jsonl.gz"), "wt", encoding="utf-8") as fh:
            for cid in ids[s * per:(s + 1) * per]:
                p = os.path.join(args.cif_root, cod_path(cid))
                if not os.path.exists(p):
                    missing += 1
                    continue
                cif = open(p, encoding="utf-8", errors="replace").read()
                m = meta[cid]
                try:
                    import gemmi
                    n_atoms = len(gemmi.cif.read_string(cif).sole_block().find_loop("_atom_site_label"))
                except Exception:
                    n_atoms = 0
                fh.write(json.dumps({"refcode": cid, "cif": cif, "smiles": "", "spacegroup": m.get("sg") or "",
                                     "n_atoms": n_atoms, "formula": m.get("formula") or "",
                                     "cell": [float(m[k]) for k in ("a", "b", "c", "alpha", "beta", "gamma")]}) + "\n")
    print(f"[shards] {len(ids) - missing:,} records in {args.n_shards} shards ({missing} CIFs missing)")
    return 0


# ------------------------------------------------------------------------------------------------
# local: scripts/32's COD truth -> 19's truth schema (refcode = COD id), so 19/28 score the replica unchanged
# ------------------------------------------------------------------------------------------------
def reduced_formula(counts: dict) -> str:
    """Non-H element counts reduced by their gcd, e.g. {'C': 40, 'N': 8, 'O': 8} -> 'C5N1O1'."""
    from functools import reduce
    from math import gcd
    n = {k: round(v) for k, v in counts.items() if k not in ("H", "D") and round(v) > 0}
    g = reduce(gcd, n.values()) if n else 1
    return "".join(f"{k}{n[k] // g}" for k in sorted(n))


def cod_formula(s: str) -> str:
    """COD sum formula ('- C20 H10 N4 O4 -') -> reduced non-H formula."""
    counts: dict = {}
    for el, k in re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", s or ""):
        counts[el] = counts.get(el, 0) + (float(k) if k else 1.0)
    return reduced_formula(counts)


def cmd_truth(args) -> int:
    """scripts/32 output -> 19's schema. Two adaptations, both because COD has no refcode families:
    * scripts/18 builds each query's sets from the CSD family of its compound, so its 'unknown' entries are all
      the same compound (134/134 share the stem). COD candidates come from an element + cell search and include
      other compounds, which PackingSimilarity (molecular differences not allowed) cannot compare. A candidate whose
      reduced non-H COD sum formula matches none of the same-form entries' is another compound: it stays an
      ordinary library entry (counted when it ranks above the truth). Same-formula entries with no comparison stay
      unknown, as in 18. (The opXRD basis is no formula source: its P1 expansion over-counts some sites, e.g.
      pattern_98 C9ClN vs COD C3ClN.)
    * Compound labels (the compound bootstrap and macro weights) are the main benchmark's CSD stems, so the replica
      weights compounds exactly as the main table does (formula labels merge isomers: 114 vs 124 groups). They are
      plain strings in the released file; patterns identified only in COD get 'cod:<smallest same-form id>'."""
    cod = json.load(open(os.path.join(DATA, "cod_truth.json")))
    csd = {t["id"]: t for t in json.load(open(args.csd_truth))}
    out, n_cod_only, moved, kept_unk = [], 0, 0, 0
    rc = lambda xs: [{**{k: v for k, v in x.items() if k != "cod_id"}, "refcode": x["cod_id"]} for x in xs]
    for t in cod:
        same = t["same_form"]
        stems = csd.get(t["id"], {}).get("true_stems") or []
        if same and not stems:
            stems, n_cod_only = [f"cod:{min(x['cod_id'] for x in same)}"], n_cod_only + 1
        fr = sorted({cod_formula(x["formula"]) for x in same})
        unk = [x for x in t["unknown"] if cod_formula(x["formula"]) in fr]
        moved += len(t["unknown"]) - len(unk)
        kept_unk += len(unk)
        out.append({"id": t["id"], "h_omitted": t["h_omitted"], "wavelength": t["wavelength"], "formulas": fr,
                    "n_candidates": t["n_candidates"], "anchor_tier": t["anchor_tier"], "anchors": t["anchors"],
                    "true_stems": stems if same else [], "same_form": rc(same), "other_form": rc(t["other_form"]),
                    "unknown": rc(unk), "n_other_compound": len(t["unknown"]) - len(unk)})
    path = os.path.join(DATA, "opxrd_truth_cod.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    both = sum(1 for t in out if t["true_stems"] and csd.get(t["id"], {}).get("true_stems"))
    print(f"[truth] {sum(1 for t in out if t['true_stems'])} of {len(out)} patterns identified in COD "
          f"({both} also in the CSD, {n_cod_only} COD only); unknown: {moved} other-compound candidates returned "
          f"to the library, {kept_unk} same-formula kept; same-form sets with > 1 formula: "
          f"{[t['id'] for t in out if len(t['formulas']) > 1]} -> {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("meta", "select", "cands", "fetch_cifs", "paths", "shards", "truth"))
    ap.add_argument("--csd_truth", default="C:/Users/frank/xrd-clip-data/opxrd/opxrd_truth.json")
    ap.add_argument("--meta", default="cod/library_meta.json")
    ap.add_argument("--cif_root", default="cod/cif")
    ap.add_argument("--out", default="cod/shards")
    ap.add_argument("--n_shards", type=int, default=6)
    args = ap.parse_args()
    return {"meta": cmd_meta, "select": cmd_select, "cands": cmd_cands, "fetch_cifs": cmd_fetch_cifs,
            "paths": cmd_paths, "shards": cmd_shards, "truth": cmd_truth}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
