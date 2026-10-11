"""Real benchmark ground truth: which CSD entries are the SAME solid form as each opXRD pattern.

Run with the CCDC miniconda interpreter (licensed):

  <ccdc python> scripts/18_opxrd_truth_sets.py

scripts/16_... matched each pattern to a CSD entry by reduced cell + element set and picked one
"best" hit, preferring corpus entries. That label is not good enough for a benchmark: cell-match hit
lists often contain unrelated compounds with the same element set, and a polymorph's
redeterminations (other temperatures, pressures, labs) are just as correct an answer as the entry
that was picked.

opXRD ships the labelled phase as a full P1 atom basis, so identity can be checked directly:
  1. Re-run the reduced-cell search (CCDC defaults, element-set filter) with NO cap on the hit list.
  2. Enumerate the CSD refcode family (STEM, STEM01, ...) of every hit.
  3. Anchors: CCDC PackingSimilarity (15-molecule shell, H positions ignored) of the opXRD structure
     against every cell-search hit. 15/15 = same form. Strict tier first (identical molecules); if
     no hit passes, a lenient tier ignores bond types, bond counts, H counts and the smallest
     components (solvent, counter-ions). Bond perception on a P1 structure rebuilt from bare
     coordinates differs from the curated CSD molecule, so H-omitted refinements, salts and
     hydrates only match in this tier. ("Allow molecular differences" is NOT used: it ran away to
     3.7 GB on one H-omitted pattern.) Anchors record which tier found them.
  4. Same-form set: PackingSimilarity of each anchor (a CSD entry, so H is consistent) against every
     member of its stem family, strict tier. 15/15 = same form at any temperature or pressure. A
     comparison that returns nothing means the molecular content differs (solvate, salt variant,
     another compound) or the entry cannot be compared. Within the anchors' stems:
       same_form  15/15 against an anchor     -> correct answers
       other_form compared, < 15/15           -> polymorph distractors (within-family set)
       unknown    no comparison possible      -> ignored when ranking, never scored as an error
  5. CIFs of every member of a true stem that is not in the stick cache are written to a gitignored
     jsonl.gz shard for the stage-2 simulator (xrdclip.data.simulate --in ... --out ...).

Writes C:/Users/frank/xrd-clip-data/opxrd/opxrd_truth.json (refcodes + comparison metadata only)
and cache/real_extra/shard_extra.jsonl.gz (CSD-derived CIF text: licence-restricted, gitignored).
"""
from __future__ import annotations

import gzip
import json
import os
import re
import sys
import time

from ccdc.crystal import Crystal, PackingSimilarity
from ccdc.io import EntryReader
from ccdc.search import ReducedCellSearch

DATA = "C:/Users/frank/xrd-clip-data"
SHELL = 15


def stem_of(rc: str) -> str:
    return rc.rstrip("0123456789")


def element(symbol: str) -> str:
    return re.match(r"[A-Z][a-z]?", symbol.strip()).group(0)


def p1_cif(o: dict) -> str:
    """opXRD phase (lattice + full-cell basis) -> P1 CIF text."""
    a, b, c, al, be, ga = (float(x) for x in o["lattice"].strip("()").split(","))
    lines = ["data_opxrd", f"_cell_length_a {a:.5f}", f"_cell_length_b {b:.5f}", f"_cell_length_c {c:.5f}",
             f"_cell_angle_alpha {al:.4f}", f"_cell_angle_beta {be:.4f}", f"_cell_angle_gamma {ga:.4f}",
             "_symmetry_space_group_name_H-M 'P 1'", "loop_", "_symmetry_equiv_pos_as_xyz", "'x,y,z'",
             "loop_", "_atom_site_label", "_atom_site_type_symbol", "_atom_site_fract_x",
             "_atom_site_fract_y", "_atom_site_fract_z", "_atom_site_occupancy"]
    for i, at in enumerate(o["basis"]):
        el = element(at["symbol"])
        lines.append(f"{el}{i + 1} {el} {at['x']:.6f} {at['y']:.6f} {at['z']:.6f} {float(at['occupancy']):.4f}")
    return "\n".join(lines) + "\n"


def family(reader: EntryReader, stem: str, max_suffix: int = 99, max_miss: int = 25) -> "list[str]":
    """STEM, STEM01, STEM02, ... present in this CSD release (suffix gaps up to max_miss tolerated;
    COUMAR has gaps longer than 5, which silently dropped COUMAR15 in the first run)."""
    out, miss = [], 0
    for i in range(max_suffix + 1):
        rc = stem if i == 0 else f"{stem}{i:02d}"
        try:
            reader.entry(rc)
            out.append(rc)
            miss = 0
        except Exception:
            if i > 0:
                miss += 1
                if miss >= max_miss:
                    break
    return out


def extra_record(e) -> "dict | None":
    """Stage-1 record for the simulator; like csd_extract.extract_record but SMILES may be empty."""
    try:
        cry = e.crystal
        cif = cry.to_string("cif")
        if not cif:
            return None
        try:
            smiles = cry.molecule.smiles or ""
        except Exception:
            smiles = ""
        cl, ca = cry.cell_lengths, cry.cell_angles
        return {"refcode": e.identifier, "cif": cif, "smiles": smiles, "spacegroup": cry.spacegroup_symbol,
                "z_prime": cry.z_prime, "z_value": cry.z_value, "r_factor": e.r_factor,
                "n_atoms": len(cry.molecule.atoms), "formula": cry.formula,
                "cell": [cl.a, cl.b, cl.c, ca.alpha, ca.beta, ca.gamma],
                "is_solvate": bool(getattr(e, "is_solvate", False)), "temperature": e.temperature}
    except Exception:
        return None


def main() -> int:
    pats = {o["id"]: o for o in json.load(open(f"{DATA}/opxrd/opxrd_cnrs_organic.json"))["organic"]}
    cells = {r["id"]: r for r in json.load(open(f"{DATA}/opxrd/opxrd_reduced_cells.json"))}
    in_cache = set(ln.strip() for ln in open(f"{DATA}/stick_refcodes.txt") if ln.strip())
    reader = EntryReader("CSD")
    strict, lenient = PackingSimilarity(), PackingSimilarity()
    for ps in (strict, lenient):
        ps.settings.packing_shell_size = SHELL
        ps.settings.ignore_hydrogen_positions = True
        ps.settings.allow_molecular_differences = False   # True blew up to 3.7 GB on an H-omitted pattern
        ps.settings.timeout_ms = 30000
    for flag in ("ignore_bond_types", "ignore_bond_counts", "ignore_hydrogen_counts", "ignore_smallest_components"):
        setattr(lenient.settings, flag, True)

    def compare(ps, a, b) -> "tuple[int, float] | None":
        try:
            r = ps.compare(a, b)
            return (r.nmatched_molecules, float(r.rmsd)) if r is not None else None
        except Exception:
            return None

    entry_meta: dict[str, dict] = {}

    def meta(rc: str) -> dict:
        if rc not in entry_meta:
            e = reader.entry(rc)
            entry_meta[rc] = {"temperature": e.temperature, "is_powder_study": bool(e.is_powder_study),
                              "has_disorder": bool(e.has_disorder), "in_cache": rc in in_cache,
                              "formula": e.formula}
        return entry_meta[rc]

    out, families_seen, t0 = [], {}, time.time()
    for n, (pid, o) in enumerate(sorted(pats.items(), key=lambda kv: int(re.findall(r"\d+", kv[0])[0]))):
        cry = Crystal.from_string(p1_cif(o), "cif")
        want = set(o["elements"]) - {"H"}
        c = cells[pid]["cell_reduced"]
        hits = []
        for hit in ReducedCellSearch(ReducedCellSearch.Query(c[:3], c[3:], "P")).search():
            try:
                e = reader.entry(hit.identifier)
                els = {a.atomic_symbol for a in e.molecule.atoms} - {"H"}
            except Exception:
                continue
            if els == want:
                hits.append(hit.identifier)
        stems = sorted({stem_of(rc) for rc in hits})
        # 3. anchors: the opXRD structure itself against every cell-search hit
        anchors, tier = {}, None
        for name, ps in (("strict", strict), ("lenient", lenient)):
            direct = {rc: compare(ps, cry, reader.entry(rc).crystal) for rc in hits}
            anchors = {rc: v for rc, v in direct.items() if v is not None and v[0] == SHELL}
            if anchors:
                tier = name
                break
        # 4. same-form set: each anchor against its whole stem family (CSD vs CSD, strict)
        # Anchors are all the same form, so two references suffice (naphthalene has 29 anchors x 48
        # family members; comparing against every anchor took minutes per pattern).
        true_stems = sorted({stem_of(rc) for rc in anchors})
        reps = [rc for rc, _ in sorted(anchors.items(), key=lambda kv: kv[1][1])[:2]]
        rep_cry = {a: reader.entry(a).crystal for a in reps}
        comps: dict = {}
        for st in true_stems:
            if st not in families_seen:
                families_seen[st] = family(reader, st)
            members = sorted(set(families_seen[st]) | {rc for rc in hits if stem_of(rc) == st})
            for rc in members:
                if rc in anchors:
                    comps[rc] = anchors[rc]          # rmsd here is against the opXRD structure itself
                    continue
                best, other_cry = None, reader.entry(rc).crystal
                for a in reps:
                    v = compare(strict, rep_cry[a], other_cry)
                    if v is not None and (best is None or (v[0], -v[1]) > (best[0], -best[1])):
                        best = v
                    if best is not None and best[0] == SHELL:
                        break
                comps[rc] = best
        same = {rc: v for rc, v in comps.items() if v is not None and v[0] == SHELL}
        other = {rc: v for rc, v in comps.items() if v is not None and v[0] < SHELL}
        unknown = sorted(rc for rc, v in comps.items() if v is None)
        rec = {"id": pid, "h_omitted": o["h_omitted"], "wavelength": o["wavelength"], "n_cell_hits": len(hits),
               "cell_hit_stems": stems, "true_stems": true_stems, "anchor_tier": tier,
               "anchors": {rc: {"n_matched": v[0], "rmsd": round(v[1], 4)} for rc, v in anchors.items()},
               "same_form": [{"refcode": rc, "n_matched": v[0], "rmsd": round(v[1], 4),
                              "is_anchor": rc in anchors, **meta(rc)}
                             for rc, v in sorted(same.items(), key=lambda kv: kv[1][1])],
               "other_form": [{"refcode": rc, "n_matched": v[0], "rmsd": round(v[1], 4), **meta(rc)}
                              for rc, v in sorted(other.items())],
               "unknown": [{"refcode": rc, **meta(rc)} for rc in unknown]}
        out.append(rec)
        print(f"[{n + 1}/{len(pats)}] {pid}: hits {len(hits)} stems {len(stems)} true {true_stems} ({tier}) "
              f"same {len(same)} other {len(other)} unknown {len(unknown)} ({time.time() - t0:.0f}s)", flush=True)

    with open(f"{DATA}/opxrd/opxrd_truth.json", "w") as fh:
        json.dump(out, fh, indent=1)

    # CIFs for every member of a true stem that the stick cache lacks (targets + distractors).
    need = sorted({x["refcode"] for r in out for k in ("same_form", "other_form", "unknown") for x in r[k]
                   if not x["in_cache"]})
    os.makedirs("cache/real_extra", exist_ok=True)
    n_ok = 0
    with gzip.open("cache/real_extra/shard_extra.jsonl.gz", "wt", encoding="utf-8") as fh:
        for rc in need:
            rec = extra_record(reader.entry(rc))
            if rec is not None:
                fh.write(json.dumps(rec) + "\n")
                n_ok += 1
    ident = sum(bool(r["true_stems"]) for r in out)
    print(f"\n{len(out)} patterns; {ident} identified (>=1 15/15 entry); "
          f"{len(need)} family members outside the stick cache, {n_ok} CIFs written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
