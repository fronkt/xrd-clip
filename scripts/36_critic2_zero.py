"""X2 (Frank 10-14): critic2 re-rank with an answer-blind zero-offset correction (sensitivity).

critic2's COMPAREVC applies no zero-offset correction, while every other score scans one. Protocol fixed in
tasks/todo.md C8 (X2) before any number: per single-crystal pattern, the native zero offset delta maximising the
cosine between the background-subtracted measured pattern, shifted by -delta and put on the Cu grid, and the
clean rendering of its ln q top-1 entry (the method's own guess, never the answer labels), over native shifts
+-0.5 deg x lambda/lambda_Cu in steps of 0.02 deg x lambda/lambda_Cu. The fitted peak positions are shifted by
-delta (equivalent to refitting a shifted pattern) -> fit_zs.peaks; scripts/29 then reruns the shortlists that
hold the answer (a miss stays a miss).

  PYTHONPATH=src python scripts/36_critic2_zero.py offsets   -> results/critic2_zero_offsets.json
  python scripts/36_critic2_zero.py write                    -> cache/critic2/<pid>/fit_zs.peaks
Then: 29 run --peaks fit_zs.peaks --suffix _zs --pids <answer-holding patterns>; 29 eval --suffix _zs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from importlib import import_module

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
bench = import_module("19_real_bench")
hubm = import_module("21_hubness")

ROOT = "cache/critic2"
OUT = "results/critic2_zero_offsets.json"


def cmd_offsets(args) -> int:
    from xrdclip.data import augment as A
    from xrdclip.eval.real_patterns import preprocess, to_cu_grid
    plan = json.load(open(os.path.join(ROOT, "plan.json")))
    pats = {o["id"].replace(".json", ""): o for o in json.load(open(args.patterns))["organic"]}
    shards = bench.shard_table(None)
    gidx = {rc: i for i, rc in enumerate(hubm.library_refcodes(shards))}
    top1 = {pid: p["shortlist"][0] for pid, p in plan.items()}
    rcs = sorted(set(top1.values()))
    ref = dict(zip(rcs, bench.render(hubm.records_at(shards, [gidx[rc] for rc in rcs], bench.RENDER_FIELDS))))
    out = {}
    for pid, p in sorted(plan.items()):
        wl = float(p["wavelength"])
        tt, y = np.asarray(pats[pid]["two_theta"], float), np.asarray(pats[pid]["intensity"], float)
        r = np.asarray(ref[top1[pid]], float)
        r = r / (np.linalg.norm(r) or 1)
        scale = wl / A.WAVELENGTH
        shifts = np.arange(-25, 26) * 0.02 * scale            # +-0.5 deg x lambda/lambda_Cu, 0.02 deg x lambda/lambda_Cu
        best, best_d = -np.inf, 0.0
        for dlt in shifts:
            yc, m = to_cu_grid(tt - dlt, y, wl)
            q = preprocess(yc, m, bkg=True).astype(float)
            c = float(q @ r / (np.linalg.norm(q) or 1))
            if c > best:
                best, best_d = c, float(dlt)
        out[pid] = {"top1_entry": top1[pid], "wavelength": wl, "delta_native_deg": round(best_d, 5),
                    "delta_cu_equiv_deg": round(best_d / scale, 4), "cosine": round(best, 4),
                    "at_edge": bool(abs(best_d) >= 0.5 * scale - 1e-9)}
        print(pid, out[pid], flush=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {OUT}")
    return 0


def cmd_write(args) -> int:
    off = json.load(open(OUT))
    for pid, v in off.items():
        src = os.path.join(ROOT, pid, "fit.peaks")
        lines = []
        for ln in open(src):
            if ln[:1].isdigit():
                f = ln.split()
                f[0] = f"{float(f[0]) - v['delta_native_deg']:.10f}"
                ln = " ".join(f) + "\n"
            lines.append(ln)
        with open(os.path.join(ROOT, pid, "fit_zs.peaks"), "w") as fh:
            fh.writelines(lines)
    print(f"[write] fit_zs.peaks for {len(off)} patterns")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("offsets", "write"))
    ap.add_argument("--patterns", default=f"{bench.DATA}/opxrd/opxrd_cnrs_organic.json")
    args = ap.parse_args()
    return {"offsets": cmd_offsets, "write": cmd_write}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
