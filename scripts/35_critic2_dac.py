"""X3 (Frank 10-14): critic2 peak refits of the pressure-series single-crystal patterns whose input is corrupted.

The automated pipeline of scripts/29 fails on the diamond-anvil patterns (positive control: own entry first for
1 of 22). On 10 of them (EPIVEF 2, LILRIJ 1, MEHNIY 7; all pressure series, no ambient pattern) the cut window
starts with 29-44 zero-intensity points, the 20-knot David-Sivia background oscillates over them and most fitted
"peaks" are background artefacts (pattern_213: own-entry DIFF 0.227). Protocol (tasks/todo.md C8, X3, as amended
before any grid run): drop the zero-intensity points and refit with critic2's defaults; no tuning. A first plan
tuned knots and peak cutoff on the own-entry DIFF, which a 2-setting test showed rewards sparse peak lists.

  PYTHONPATH=src python scripts/35_critic2_dac.py refit --procs 3   -> cache/critic2/<pid>/fit_x3.peaks (+ x3/ dir)
Then scripts/29 run/control/inject/eval --peaks fit_x3.peaks --suffix _x3 --pids <these> (box session with X2).
Everything stays under cache/ (CSD-derived CIFs: gitignored).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import import_module

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
c2 = import_module("29_critic2_rerank")

ROOT = "cache/critic2"
PEAKS_X3 = "fit_x3.peaks"


def zero_padded() -> list:
    """sc patterns whose cut input has zero-intensity points (leading, trailing or inside)."""
    plan = json.load(open(os.path.join(ROOT, "plan.json")))
    return sorted(pid for pid in plan if (np.loadtxt(os.path.join(ROOT, pid, "expt.xy"))[:, 1] == 0).any())


def refit(pid: str, timeout: int) -> dict:
    d = os.path.join(ROOT, pid, "x3")
    os.makedirs(d, exist_ok=True)
    xy = np.loadtxt(os.path.join(ROOT, pid, "expt.xy"))
    xy = xy[xy[:, 1] != 0]
    np.savetxt(os.path.join(d, "expt.xy"), xy, fmt="%.6f")
    with open(os.path.join(d, "fit.cri"), "w") as fh:
        fh.write("xrpd background expt.xy expt-bg.xy\nxrpd fit expt-bg.xy\n")
    txt = c2.critic2(os.path.join(d, "fit.cri"), d, timeout=timeout)
    m = re.search(r"RMS of the fit \(final\)\s*=\s*([0-9.eE+-]+)", txt)
    shutil.copy(os.path.join(d, c2.PEAKS), os.path.join(ROOT, pid, PEAKS_X3))
    n = sum(1 for ln in open(os.path.join(d, c2.PEAKS)) if ln[:1].isdigit())
    return {"pid": pid, "n_points": int(len(xy)), "n_peaks": n, "fit_rms": float(m.group(1)) if m else None}


def cmd_refit(args) -> int:
    pids = zero_padded()
    print(f"[refit] {len(pids)} zero-padded patterns: {pids}", flush=True)
    rows = []
    with ThreadPoolExecutor(args.procs) as ex:
        for f in as_completed([ex.submit(refit, pid, args.timeout) for pid in pids]):
            rows.append(f.result())
            print("[refit]", rows[-1], flush=True)
    with open(os.path.join(ROOT, "x3_refits.json"), "w") as fh:
        json.dump(sorted(rows, key=lambda r: r["pid"]), fh, indent=1)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("refit",))
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=14400)
    args = ap.parse_args()
    return {"refit": cmd_refit}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
