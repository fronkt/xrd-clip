"""CCDC interpreter: dump packed unit cell + one molecule per ROY polymorph (coords only, never committed)."""
import json
from ccdc.io import EntryReader
from ccdc.descriptors import MolecularDescriptors as MD
r = EntryReader('CSD')
out = {}
for ref, form in [('QAXMEH01', 'Y'), ('QAXMEH03', 'OP'), ('QAXMEH', 'ON'), ('QAXMEH02', 'R')]:
    e = r.entry(ref); c = e.crystal
    pk = c.packing(box_dimensions=((0, 0, 0), (1, 1, 1)), inclusion='CentroidIncluded')
    pk.assign_bond_types(); pk.add_hydrogens(mode='missing')
    def dump(m):
        idx = {a: i for i, a in enumerate(m.atoms)}
        return {"atoms": [{"el": a.atomic_symbol, "p": list(a.coordinates), "comp": 0} for a in m.atoms],
                "bonds": [[idx[b.atoms[0]], idx[b.atoms[1]]] for b in m.bonds]}
    P = dump(pk)
    # tag atoms by component so molecules can be tinted
    k = 0
    for ci, comp in enumerate(pk.components):
        for a in comp.atoms: pass
    comp_of = {}
    for ci, comp in enumerate(pk.components):
        for a in comp.atoms: comp_of[tuple(round(x, 3) for x in a.coordinates)] = ci
    for a in P["atoms"]: a["comp"] = comp_of.get(tuple(round(x, 3) for x in a["p"]), -1)
    mol = c.asymmetric_unit_molecule.components[0]
    cp = c.cell_lengths, c.cell_angles
    o = c.orthogonal_to_fractional  # not needed; compute cell vectors from fractional
    vec = [list(row) for row in c.fractional_to_orthogonal.rotation]
    # torsion S-C-N-C (thiophene C2-N-phenyl C1) found by pattern on asym molecule
    out[form] = {"ref": ref, "sg": c.spacegroup_symbol, "cell": [*c.cell_lengths, *c.cell_angles],
                 "vec": vec, "packed": P, "mol": dump(mol), "ncomp": len(pk.components), "color": e.color}
    print(form, ref, c.spacegroup_symbol, len(pk.atoms), len(pk.components), vec[0])
json.dump(out, open('roy.json', 'w'))
