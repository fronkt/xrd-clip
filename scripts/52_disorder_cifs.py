"""CIFs with site occupancies for the disordered entries the benchmark added to the library (board review 11-19, A07).

  <CCDC python> scripts/52_disorder_cifs.py      -> cache/real_extra/occ/<refcode>.cif + cache/real_extra/occ/index.json

The library's added entries (cache/real_extra/shard_extra.jsonl.gz, written by the CSD Python API's CIF writer) carry no
_atom_site_occupancy column, so pymatgen simulated every disorder site at full occupancy (ZZZUWK: four orientations of
the propylammonium cation at once). The corpus itself excludes disordered entries and is unaffected. This script reads
each added entry from the CSD and, for the disordered ones, writes a CIF with the cell, the symmetry operators and the
asymmetric-unit atoms with their CSD occupancies; scripts/53 re-simulates their stick patterns from these files.
Licensed CSD content: the CIFs stay in the gitignored cache.
"""
import gzip
import json
import os
import traceback

OUT = "cache/real_extra/occ"


def write_cif(entry, path):
    c = entry.crystal
    a, b, cc = c.cell_lengths
    al, be, ga = c.cell_angles
    lines = [f"data_{entry.identifier}",
             f"_cell_length_a {a:.5f}", f"_cell_length_b {b:.5f}", f"_cell_length_c {cc:.5f}",
             f"_cell_angle_alpha {al:.4f}", f"_cell_angle_beta {be:.4f}", f"_cell_angle_gamma {ga:.4f}",
             "loop_", "_symmetry_equiv_pos_as_xyz"]
    lines += [f"'{op}'" for op in c.symmetry_operators]
    lines += ["loop_", "_atom_site_label", "_atom_site_type_symbol", "_atom_site_fract_x", "_atom_site_fract_y",
              "_atom_site_fract_z", "_atom_site_occupancy"]
    n, occ_sum, partial = 0, 0.0, 0
    for i, at in enumerate(c.disordered_asymmetric_unit_molecule.atoms
                           if hasattr(c, "disordered_asymmetric_unit_molecule") else c.asymmetric_unit_molecule.atoms):
        fc = at.fractional_coordinates
        if fc is None:
            continue
        occ = at.occupancy if at.occupancy is not None else 1.0
        lines.append(f"{at.atomic_symbol}{i + 1} {at.atomic_symbol} {fc[0]:.6f} {fc[1]:.6f} {fc[2]:.6f} {occ:.4f}")
        n += 1
        occ_sum += occ
        partial += occ < 0.999
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return n, occ_sum, partial


def main():
    from ccdc.io import EntryReader
    os.makedirs(OUT, exist_ok=True)
    reader = EntryReader("CSD")
    rcs = [json.loads(ln)["refcode"] for ln in gzip.open("cache/real_extra/shard_extra.jsonl.gz", "rt", encoding="utf-8")]
    index = {}
    for rc in rcs:
        try:
            e = reader.entry(rc)
            row = {"has_disorder": bool(e.has_disorder)}
            if e.has_disorder:
                n, s, p = write_cif(e, f"{OUT}/{rc}.cif")
                row.update({"n_sites": n, "occupancy_sum": round(s, 3), "n_partial": p})
            index[rc] = row
        except Exception as ex:
            index[rc] = {"error": str(ex)}
    json.dump(index, open(f"{OUT}/index.json", "w"), indent=1)
    dis = [k for k, v in index.items() if v.get("has_disorder")]
    print(f"[52] {len(rcs)} added entries, {len(dis)} disordered, "
          f"{sum(1 for k in dis if index[k].get('n_partial', 0) > 0)} with partial occupancies; "
          f"{sum('error' in v for v in index.values())} errors", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    finally:
        os._exit(0)          # the CCDC interpreter otherwise hangs at exit
