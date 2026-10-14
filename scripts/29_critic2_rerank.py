"""R3: re-rank the ln q shortlist of the single-crystal queries with critic2's VC-GPWDF (tasks/todo.md Step 3).

VC-GPWDF (Otero-de-la-Roza, J. Appl. Cryst. 57, 1401, 2024; critic2 COMPAREVC) compares a candidate crystal
structure with a measured powder pattern's fitted peak list while it varies the candidate's cell (+-10 % lengths,
+-5 deg angles in QUICK mode). It is the published variable-cell method, so it tests whether the single-crystal
case stays hard for the strongest classical method. The pipeline is the paper's, run as published:
  XRPD BACKGROUND (20 knots) -> XRPD FIT (defaults) -> COMPAREVC cand.cif pattern.peaks GLOBAL QUICK ALPHA 1
  LAMBDA <measurement wavelength>; no zero-offset correction.
One added step: each pattern is cut to the benchmark's d-spacing window (Cu K-alpha1 2theta 5-50 deg), the range
every other score sees.

Shortlist: the top-K entries of ln q v2 (scripts/28 tag r1) for every query with a single-crystal answer, ignored
entries (same-form powder studies) left out. A query whose answer is not in its shortlist is a miss.

  PYTHONPATH=src python scripts/29_critic2_rerank.py prep        # xy cut, shortlists, CIFs (local: needs the CSD)
  python scripts/29_critic2_rerank.py fit --procs 60             # critic2 peak fits (box or local)
  python scripts/29_critic2_rerank.py run --procs 120            # comparisons (CRITIC2=<binary> on the box)
  PYTHONPATH=src python scripts/29_critic2_rerank.py eval        # -> results/critic2_rerank.json
Everything under cache/critic2/ (CSD-derived CIFs: gitignored, never released; a rented box gets them only by a
manual-mode upload, scripts/box/critic2_k10_box.sh).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import import_module

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
bench = import_module("19_real_bench")
hubm = import_module("21_hubness")
pn = import_module("22_paper_numbers")
rs = import_module("28_rescore")

CRITIC2 = os.environ.get("CRITIC2", os.path.expanduser("~/opt/critic2/critic2-1.4.155-win64/bin/critic2.exe"))
ROOT = "cache/critic2"
CIF_SOURCES = [f"cache/csd/shard_{i:03d}.jsonl.gz" for i in range(6)] + ["cache/real_extra/shard_extra.jsonl.gz"]
D_MIN = 1.54056 / (2 * np.sin(np.radians(25.0)))     # Cu K-alpha1 2theta 50 deg
D_MAX = 1.54056 / (2 * np.sin(np.radians(2.5)))      # Cu K-alpha1 2theta 5 deg
DIFF_RE = re.compile(r"DIFF\s*=\s*([-+0-9.eE]+)")
# critic2 1.4.155 names the fitted peak list after the .cri input (fit.cri -> fit.peaks), on Linux and Windows
# alike (checked 10-13: box B's run found 0 comparisons because the code looked for expt-bg.peaks)
PEAKS = "fit.peaks"


def cpu_seconds(s: str | None) -> float | None:
    """critic2's 'Elapsed CPU time' string ('1h 11m 40s', '16s') in seconds."""
    if s is None:
        return None
    return sum(float(v) * {"h": 3600, "m": 60, "s": 1}[u] for v, u in re.findall(r"([0-9.]+)\s*([hms])", s))


def critic2(cri_path: str, cwd: str, timeout: int = 3600) -> str:
    """Run critic2 on an input file (single-threaded); returns the output text."""
    out = cri_path[:-4] + ".cro"
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    proc = subprocess.Popen([CRITIC2, os.path.basename(cri_path), os.path.basename(out)], cwd=cwd, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    bench.unthrottle(proc.pid)        # local Windows runs: keep the windowless child off the E-cores only
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise
    p = os.path.join(cwd, os.path.basename(out))
    return open(p, encoding="utf-8", errors="replace").read() if os.path.exists(p) else ""


def sc_queries(args):
    shards = bench.shard_table(None)
    lib_rc = hubm.library_refcodes(shards)
    gidx = {rc: i for i, rc in enumerate(lib_rc)}
    qinfo, _, _ = bench.build_queries(args.truth, args.patterns, gidx)
    keep = [(qi, q) for qi, q in enumerate(qinfo) if q["sets"]["sc"][0]]
    return lib_rc, gidx, qinfo, keep


def cmd_prep(args) -> int:
    lib_rc, gidx, qinfo, keep = sc_queries(args)
    pats = {o["id"]: o for o in json.load(open(args.patterns))["organic"]}
    S = np.load(os.path.join("cache/real_bench/rescore_r1", "S_lnq_bkg.npy"), mmap_mode="r")
    os.makedirs(os.path.join(ROOT, "cif"), exist_ok=True)
    plan, need = {}, set()
    for qi, q in keep:
        pid = q["id"].replace(".json", "")
        d = os.path.join(ROOT, pid)
        os.makedirs(d, exist_ok=True)
        wl = float(q["wavelength"] or 1.54056)
        tt = np.asarray(pats[q["id"]]["two_theta"], float)
        y = np.asarray(pats[q["id"]]["intensity"], float)
        lo = np.degrees(2 * np.arcsin(min(wl / (2 * D_MAX), 1.0)))
        hi = np.degrees(2 * np.arcsin(min(wl / (2 * D_MIN), 1.0)))
        k = (tt >= lo) & (tt <= hi)
        np.savetxt(os.path.join(d, "expt.xy"), np.c_[tt[k], y[k]], fmt="%.6f")
        tr, ig = q["sets"]["sc"]
        row = np.asarray(S[qi], dtype=np.float32).copy()
        row[[gidx[rc] for rc in ig]] = -np.inf                     # ignored entries take no shortlist slot
        short = np.argsort(-row, kind="stable")[:args.k]
        plan[pid] = {"qi": qi, "stem": q["stem"], "wavelength": wl, "n_points": int(k.sum()),
                     "truth": tr, "shortlist": [lib_rc[g] for g in short],
                     "lnq_scores": [float(row[g]) for g in short]}
        need |= set(plan[pid]["shortlist"])
    have = {f[:-4] for f in os.listdir(os.path.join(ROOT, "cif"))}
    todo = need - have
    for src in CIF_SOURCES:
        if not todo:
            break
        with gzip.open(src, "rt", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["refcode"] in todo:
                    with open(os.path.join(ROOT, "cif", r["refcode"] + ".cif"), "w", encoding="utf-8") as out:
                        out.write(r["cif"])
                    todo.discard(r["refcode"])
    with open(os.path.join(ROOT, "plan.json"), "w") as fh:
        json.dump(plan, fh, indent=1)
    print(f"[prep] {len(plan)} queries, {len(need)} distinct candidates, CIFs missing: {sorted(todo)[:10]} ({len(todo)})")
    return 0


def cmd_fit(args) -> int:
    """critic2 background + peak fit, one per pattern (-> fit.peaks)."""
    plan = json.load(open(os.path.join(ROOT, "plan.json")))

    def fit(pid):
        d = os.path.join(ROOT, pid)
        if os.path.exists(os.path.join(d, PEAKS)):
            return pid, "cached"
        with open(os.path.join(d, "fit.cri"), "w") as fh:
            fh.write("xrpd background expt.xy expt-bg.xy\nxrpd fit expt-bg.xy\n")
        try:
            txt = critic2(os.path.join(d, "fit.cri"), d, timeout=args.fit_timeout)
        except subprocess.TimeoutExpired:      # one slow fit must not take the pool down (box B, 10-13)
            return pid, f"TIMEOUT after {args.fit_timeout} s"
        m = re.search(r"RMS of the fit \(final\)\s*=\s*([0-9.eE+-]+)", txt)
        return pid, m.group(1) if m else "FAILED"
    with ThreadPoolExecutor(args.procs) as ex:
        for f in as_completed([ex.submit(fit, pid) for pid in plan]):
            print("[fit]", *f.result(), flush=True)
    return 0


def cmd_run(args) -> int:
    plan = json.load(open(os.path.join(ROOT, "plan.json")))
    res_path = os.path.join(ROOT, "diffs.jsonl")
    done = set()
    if os.path.exists(res_path):
        for line in open(res_path):
            r = json.loads(line)
            done.add((r["pid"], r["cand"]))
    # rank-major order (every pattern's 1st candidate, then every 2nd, ...): a run stopped early still leaves a
    # complete re-rank of a shorter shortlist, which `eval --k` scores (10-13: local throughput unknown)
    only = set(args.pids.split(",")) if args.pids else set(plan)
    jobs = [(pid, p["shortlist"][r]) for r in range(args.k) for pid, p in plan.items()
            if pid in only and r < len(p["shortlist"]) and (pid, p["shortlist"][r]) not in done
            and os.path.exists(os.path.join(ROOT, pid, PEAKS))]
    print(f"[run] {len(jobs)} comparisons to run ({len(done)} done) on {args.procs} processes", flush=True)

    def one(job):
        pid, cand = job
        d = os.path.join(ROOT, pid)
        tag = f"cmp_{cand}"
        cif = os.path.relpath(os.path.join(ROOT, "cif", cand + ".cif"), d).replace("\\", "/")
        with open(os.path.join(d, tag + ".cri"), "w") as fh:
            fh.write(f"comparevc {cif} {PEAKS} global quick alpha 1 lambda {plan[pid]['wavelength']:.6f}\n")
        t0 = time.time()
        try:
            txt = critic2(os.path.join(d, tag + ".cri"), d, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            txt = ""
        m = DIFF_RE.findall(txt)
        cpu = re.search(r"Elapsed CPU time:\s*(.*)", txt)
        for ext in (".cri", ".cro"):
            try:
                os.remove(os.path.join(d, tag + ext))
            except OSError:
                pass
        return {"pid": pid, "cand": cand, "diff": float(m[-1]) if m else None, "wall_s": round(time.time() - t0, 1),
                "cpu": cpu.group(1).strip() if cpu else None, "status": "ok" if m else ("timeout/empty" if not txt else "no DIFF")}

    t0 = time.time()
    with ThreadPoolExecutor(args.procs) as ex, open(res_path, "a") as out:
        futs = [ex.submit(one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            r = f.result()
            out.write(json.dumps(r) + "\n")
            out.flush()
            if i % 20 == 0:
                el = time.time() - t0
                print(f"[run] {i + 1}/{len(jobs)} {el / 60:.1f} min, eta {el / (i + 1) * (len(jobs) - i - 1) / 60:.0f} min; "
                      f"last {r}", flush=True)
    return 0


def cmd_eval(args) -> int:
    lib_rc, gidx, qinfo, keep = sc_queries(args)
    plan = json.load(open(os.path.join(ROOT, "plan.json")))
    diffs = {}
    for line in open(os.path.join(ROOT, "diffs.jsonl")):
        r = json.loads(line)
        diffs[(r["pid"], r["cand"])] = r
    nq = len(qinfo)
    rank_c2 = np.full(nq, -1, np.int64)
    rank_lnq_short = np.full(nq, -1, np.int64)
    status, n_fail, used = {}, 0, []
    for pid, p in plan.items():
        short = p["shortlist"][:args.k]
        rows = [diffs.get((pid, c)) for c in short]
        if any(r is None for r in rows):
            status[pid] = "incomplete"
            continue
        used += rows
        # a failed comparison counts as maximally dissimilar (DIFF = 1)
        dv = np.array([r["diff"] if r["diff"] is not None else 1.0 for r in rows])
        n_fail += sum(r["diff"] is None for r in rows)
        is_t = np.isin(short, p["truth"])
        qi = p["qi"]
        if not is_t.any():
            rank_c2[qi] = rank_lnq_short[qi] = args.k        # answer not in the shortlist: a miss
            continue
        rank_c2[qi] = int((dv[~is_t] < dv[is_t].min()).sum())
        sl = np.array(p["lnq_scores"][:args.k])
        rank_lnq_short[qi] = int((sl[~is_t] > sl[is_t].max()).sum())
    stems = np.array([q["stem"] for q in qinfo])
    r1 = np.load("cache/real_bench/rescore_r1/ranks.npz")["sc|lnq_bkg"]
    sel = rank_c2 >= 0
    cpu = np.array([cpu_seconds(r["cpu"]) for r in used if r["cpu"] is not None])
    res = {"meta": {"k": args.k, "n_queries": int(sel.sum()), "n_planned": len(plan), "n_comparisons": len(used),
                    "n_failed_comparisons": n_fail,
                    "failed": [[r["pid"], r["cand"], r["status"]] for r in used if r["diff"] is None],
                    "incomplete": [k for k, v in status.items() if v == "incomplete"],
                    "recall_at_k": float((rank_c2[sel] < args.k).mean()) if sel.any() else None,
                    "timing_wall_s_median": float(np.median([r["wall_s"] for r in used])) if used else None,
                    # completed comparisons only (a timed-out one reports no CPU time)
                    "cpu_s_median": float(np.median(cpu)) if len(cpu) else None,
                    "cpu_s_max": float(cpu.max()) if len(cpu) else None,
                    "cpu_h_total": float(cpu.sum() / 3600) if len(cpu) else None}}
    cboot = np.random.default_rng(0).integers(0, len(np.unique(stems)), (2000, len(np.unique(stems))))
    for name, r in (("critic2_vcgpwdf", rank_c2), ("lnq_v2_full_library", np.where(sel, r1, -1))):
        res[name] = {f"top{kk}": pn.macro(r, kk, stems, cboot) for kk in (1, 10)}
        res[name]["per_pattern_top1"] = float((r[r >= 0] == 0).mean()) if (r >= 0).any() else None
    res["paired_critic2_minus_lnq"] = {f"top{kk}": pn.paired_macro(rank_c2, np.where(sel, r1, -1), stems, kk) for kk in (1, 10)}
    # rank of the first correct entry (0 = top-1; K = answer not in the shortlist), for 22's subgroup splits + release
    res["rank_by_pattern"] = {pid: {"critic2": int(rank_c2[p["qi"]]), "lnq_shortlist": int(rank_lnq_short[p["qi"]])}
                              for pid, p in plan.items() if rank_c2[p["qi"]] >= 0}
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1)[:2000])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prep", "fit", "run", "eval"))
    ap.add_argument("--truth", default=f"{bench.DATA}/opxrd/opxrd_truth.json")
    ap.add_argument("--patterns", default=f"{bench.DATA}/opxrd/opxrd_cnrs_organic.json")
    # K = 50, not the pre-declared 100: ln q v2's single-crystal shortlist holds the answer for the same patterns at
    # 50 and 100 (36 of 59, recall 0.61), so ranks 51-100 add only competitors; disclosed in the paper. Run on a
    # rented 256-core box with a 4-h cap per comparison (10-13). A two-point timing extrapolation (cost ~V^2.15 in the
    # candidate's cell volume) first priced K = 50 at ~3,100 CPU-h and cut K to 10; the K = 10 run measured 76 CPU-h
    # for 589 comparisons (~6x below that estimate; refit ~V^1.8), so K went back to 50 on the same box
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=14400, help="per comparison (s); a timeout counts as DIFF = 1")
    ap.add_argument("--pids", default="", help="run: only these comma-separated patterns (a fit that landed late)")
    ap.add_argument("--fit_timeout", type=int, default=3600, help="per peak fit (s)")
    ap.add_argument("--out", default="results/critic2_rerank.json")
    args = ap.parse_args()
    return {"prep": cmd_prep, "fit": cmd_fit, "run": cmd_run, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
