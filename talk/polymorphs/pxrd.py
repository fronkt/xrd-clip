import json
from ccdc.io import EntryReader
from ccdc.descriptors import CrystalDescriptors
PowderPattern = CrystalDescriptors.PowderPattern
r = EntryReader('CSD'); out = {}
for ref, f in [('QAXMEH01','Y'),('QAXMEH03','OP'),('QAXMEH','ON'),('QAXMEH02','R')]:
    pp = PowderPattern.from_crystal(r.crystal(ref))
    tt = list(pp.two_theta); it = list(pp.intensity)
    out[f] = {"tt": tt, "I": it}
    print(f, len(tt), min(tt), max(tt), max(it))
json.dump(out, open('roy_pxrd.json','w'))
