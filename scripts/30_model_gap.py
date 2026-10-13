"""R4a/R4b: where the learned model loses the measured patterns, and pattern-to-pattern retrieval (tasks/todo.md Step 3).

scripts/26 showed the model ranks a clean rendering of each pattern's own entry well (top-1 0.21, median rank 5)
and the measured pattern badly (0.01, ~6,000). The ladder renders the own entry with ONE measured effect at a
time, then all of them together, and scores every variant with the same answer sets:
  mask    zero outside the measured 2theta range (21 % of the patterns do not cover 5-50 deg)
  width   peak width measured on the pattern (median FWHM of its strongest peaks; Caglioti U = V = 0)
  ka2     K-alpha2 doublet for lab Cu patterns (I2/I1 = 0.5)
  bkg     the measured pattern's own background added, then the queries' background removal (every other
          variant is only max-normalised, so "clean" is exactly scripts/26's own-entry rendering)
  noise   Gaussian noise at the measured pattern's noise level (MAD of first differences), before bkg removal
  zero    the zero offset that best aligns the measured pattern with its own entry (cosine scan, +-0.5 deg)
  all     every effect above
The effects are estimated from each measured pattern, two of them (zero, width) with the own entry's help: this
is a diagnosis of the model, not a method. Fixed-grid Pearson on the same variants is the classical contrast.

R4b (p2p): the clean library patterns go through the model's PATTERN encoder too, so queries are matched
pattern-to-pattern in the learned space (no structure encoder).

  PYTHONPATH=src python scripts/30_model_gap.py render     # clean library -> cache/real_bench/lib_clean_f16.npy
  PYTHONPATH=src python scripts/30_model_gap.py pemb       # pattern embeddings of the library
  PYTHONPATH=src python scripts/30_model_gap.py eval       # -> results/model_gap.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from importlib import import_module

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
bench = import_module("19_real_bench")
hubm = import_module("21_hubness")
pn = import_module("22_paper_numbers")
rs = import_module("28_rescore")

WORK = "cache/real_bench"
LIB = os.path.join(WORK, "lib_clean_f16.npy")
PEMB = os.path.join(WORK, "lib_pattern_emb.npy")
EFFECTS = ("mask", "width", "ka2", "bkg", "noise", "zero")
# Follow-up rungs (added after the first ladder, 10-12): on top of "all", an overall Debye-Waller factor
# (B = 4.45 A^2, results/r2_bvalue.json) or log-normal per-reflection intensity jitter. They test whether the
# model's remaining gap to the measured patterns is intensity mismatch, which the classical scores tolerate.
EXTRA = {"all_dw": dict(B=4.45), "all_int30": dict(jitter=0.3), "all_int50": dict(jitter=0.5),
         "all_dw_int30": dict(B=4.45, jitter=0.3)}


def _render_job(job):
    cid, path, lo, hi, g0 = job
    return g0, bench.render(bench.load_records(path, lo, hi, bench.RENDER_FIELDS)).astype(np.float16)


def cmd_render(args) -> int:
    from multiprocessing import Pool
    from xrdclip.data import augment as A
    shards = bench.shard_table(None)
    N = sum(n for _, n, _ in shards)
    out = np.lib.format.open_memmap(LIB + ".tmp.npy", mode="w+", dtype=np.float16, shape=(N, A.N_POINTS))
    jobs = bench.chunk_jobs(shards, 4096)
    t0 = time.time()
    with Pool(args.workers, initializer=bench._init_worker, initargs=({},)) as pool:
        for i, (g0, y) in enumerate(pool.imap_unordered(_render_job, jobs)):
            out[g0:g0 + len(y)] = y
            if i % 10 == 0:
                print(f"[render] {i + 1}/{len(jobs)} {(time.time() - t0) / 60:.1f} min", flush=True)
    out.flush()
    del out
    os.replace(LIB + ".tmp.npy", LIB)
    print(f"[render] {N:,} patterns -> {LIB} ({(time.time() - t0) / 60:.1f} min)")
    return 0


def load_model(args):
    import torch
    from xrdclip.models.clip import XRDClip
    torch.set_num_threads(args.threads)
    model = XRDClip(embed_dim=256)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.eval()
    return model


def encode(model, x: np.ndarray, batch: int = 256):
    """Pattern embeddings of max-normalised patterns (standardised here, as the encoder expects)."""
    import torch
    from xrdclip.data import augment as A
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            z = np.stack([A.standardize(y.astype(np.float32)) for y in x[i:i + batch]])
            out.append(model.encode_pattern(torch.from_numpy(z).unsqueeze(1)))
    return torch.cat(out)


def cmd_pemb(args) -> int:
    lib = np.load(LIB, mmap_mode="r")
    model = load_model(args)
    out = np.lib.format.open_memmap(PEMB + ".tmp.npy", mode="w+", dtype=np.float32, shape=(lib.shape[0], 256))
    t0 = time.time()
    step = 8192
    for i in range(0, lib.shape[0], step):
        out[i:i + step] = encode(model, np.asarray(lib[i:i + step])).numpy()
        print(f"[pemb] {min(i + step, lib.shape[0]):,}/{lib.shape[0]:,} {(time.time() - t0) / 60:.1f} min", flush=True)
    out.flush()
    del out
    os.replace(PEMB + ".tmp.npy", PEMB)
    return 0


# ------------------------------------------------------------------------------------------------
# measured-effect estimates
# ------------------------------------------------------------------------------------------------
def fwhm_of(y: np.ndarray, n_peaks: int = 10) -> "float | None":
    """Median full width at half maximum (deg) of the strongest peaks of a max-normalised pattern."""
    from xrdclip.data import augment as A
    from xrdclip.eval import strong_baselines as SB
    tt, h = SB.query_peaks(y, n_peaks=n_peaks, min_height=0.05)
    w = []
    for t, hh in zip(tt, h):
        i = int(round((t - A.GRID[0]) / A.STEP))
        half = hh / 2
        lo = i
        while lo > 0 and y[lo] > half:
            lo -= 1
        hi = i
        while hi < len(y) - 1 and y[hi] > half:
            hi += 1
        if lo == 0 or hi == len(y) - 1:
            continue
        # linear interpolation of both half-maximum crossings
        xl = A.GRID[lo] + (half - y[lo]) / max(y[lo + 1] - y[lo], 1e-9) * A.STEP
        xr = A.GRID[hi - 1] + (y[hi - 1] - half) / max(y[hi - 1] - y[hi], 1e-9) * A.STEP
        w.append(xr - xl)
    return float(np.median(w)) if w else None


def background_of(y: np.ndarray, m: np.ndarray) -> "tuple[np.ndarray, float]":
    """The queries' rolling-minimum background of a measured pattern, scaled so its peaks have max 1."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    from xrdclip.data import augment as A
    w = int(round(2.0 / A.STEP))
    lo = np.where(m, y, np.inf)
    base = uniform_filter1d(minimum_filter1d(np.where(np.isfinite(lo), lo, y[m].max()), w), w)
    peak = np.where(m, np.clip(y - base, 0, None), 0)
    s = peak.max() if peak.max() > 0 else 1.0
    return np.where(m, base, 0) / s, s


def noise_of(y_bkg: np.ndarray, m: np.ndarray) -> float:
    d = np.diff(y_bkg[m])
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))


def zero_of(meas: np.ndarray, own: np.ndarray, max_deg: float = 0.5) -> float:
    """Shift (deg) that maximises the cosine of the measured pattern with the shifted own-entry rendering."""
    from xrdclip.data import augment as A
    S = int(round(max_deg / A.STEP))
    best, arg = -1.0, 0
    for d in range(-S, S + 1):
        r = np.roll(own, d)
        if d > 0:
            r[:d] = 0
        elif d < 0:
            r[d:] = 0
        c = float(meas @ r / max(np.linalg.norm(meas) * np.linalg.norm(r), 1e-12))
        if c > best:
            best, arg = c, d
    return arg * A.STEP


def variant(rec: dict, eff: "set[str]", est: dict, rng, B: float = 0.0, jitter: float = 0.0) -> np.ndarray:
    """Own-entry rendering with the chosen measured effects, preprocessed like the bkg queries."""
    from xrdclip.data import augment as A
    from xrdclip.eval.real_patterns import preprocess
    if B or jitter:
        inten = np.asarray(rec["intensity"], np.float64)
        if B:
            inten = inten * np.exp(-B / (2 * np.asarray(rec["d_hkl"], np.float64) ** 2))
        if jitter:
            inten = inten * rng.lognormal(0.0, jitter, inten.shape)
        rec = {**rec, "intensity": inten}
    p = A.CLEAN
    if "width" in eff and est["fwhm"]:
        p = A.sweep_params(u=0.0, v=0.0, w=est["fwhm"] ** 2)
    if "zero" in eff:
        p = A.sweep_params(**{**p.__dict__, "zero_offset": est["zero"]})
    rho = rs.TUBES["cu"][1] / rs.TUBES["cu"][0] if ("ka2" in eff and est["tube"] == "cu") else None
    if rho is None:
        y = A.render_sticks(rec, p)
    else:
        tt = np.asarray(rec["two_theta"], np.float64)
        s2 = np.sin(np.radians(tt / 2)) * rho
        ok = s2 < 1
        r2 = {"two_theta": np.r_[tt, np.degrees(2 * np.arcsin(s2[ok]))],
              "intensity": np.r_[rec["intensity"], rs.KA2_RATIO * np.asarray(rec["intensity"])[ok]],
              "hkl": np.r_[rec["hkl"], np.asarray(rec["hkl"])[ok]], "d_hkl": np.r_[rec["d_hkl"], np.asarray(rec["d_hkl"])[ok]],
              "recip": rec["recip"]}
        y = A.render_sticks(r2, p)
    m = est["mask"] if "mask" in eff else np.ones(A.N_POINTS, bool)
    if "bkg" in eff:
        y = y + est["bkg"]
    if "noise" in eff:
        y = y + rng.normal(0.0, est["noise"], y.shape)
    y = np.where(m, y, 0).astype(np.float32)
    return preprocess(y, m, bkg="bkg" in eff)          # "clean" = the plain rendering of scripts/26


def cmd_eval(args) -> int:
    import torch
    from xrdclip.data import augment as A
    from xrdclip.eval.real_patterns import to_cu_grid
    shards = bench.shard_table(None)
    lib_rc = hubm.library_refcodes(shards)
    N = len(lib_rc)
    gidx = {rc: i for i, rc in enumerate(lib_rc)}
    qinfo, _, Q = bench.build_queries(args.truth, args.patterns, gidx)
    nq = len(qinfo)
    truth = {t["id"]: t for t in json.load(open(args.truth))}
    pats = {o["id"]: o for o in json.load(open(args.patterns))["organic"]}
    anc = [next(a for a in sorted(truth[q["id"]]["anchors"]) if a in gidx) for q in qinfo]   # as in scripts/26
    recs = hubm.records_at(shards, [gidx[a] for a in anc], bench.RENDER_FIELDS)
    own_clean = bench.render(recs)
    ests = []
    for qi, q in enumerate(qinfo):
        wl = float(q["wavelength"] or A.WAVELENGTH)
        y, m = to_cu_grid(pats[q["id"]]["two_theta"], pats[q["id"]]["intensity"], wl)
        bkg, _ = background_of(y, m)
        meas = Q["raw_bkg"][qi]
        ests.append({"mask": m, "tube": rs.tube(q["wavelength"]), "fwhm": fwhm_of(meas), "bkg": bkg,
                     "noise": noise_of(meas, m), "zero": zero_of(meas, own_clean[qi])})
    variants = {"clean": (set(), {}), **{e: ({e}, {}) for e in EFFECTS}, "all": (set(EFFECTS), {}),
                **{k: (set(EFFECTS), kw) for k, kw in EXTRA.items()}}
    if args.variants:
        variants = {k: v for k, v in variants.items() if k in args.variants.split(",")}
    X = {} if args.variants else {"measured": Q["raw_bkg"]}
    for name, (eff, kw) in variants.items():
        rng = np.random.default_rng(1)
        X[f"own_{name}"] = np.stack([variant(recs[qi], eff, ests[qi], rng, **kw) for qi in range(nq)])

    model = load_model(args)
    lib_emb = torch.from_numpy(np.concatenate([np.load(os.path.join(WORK, f"emb_{os.path.basename(os.path.dirname(p))}_"
                                                                          f"{os.path.basename(p)[:-4]}_{n}.npy")) for p, n, _ in shards]).astype(np.float32))
    pemb = torch.from_numpy(np.load(PEMB, mmap_mode="r")[:]) if os.path.exists(PEMB) else None
    lib = np.load(LIB, mmap_mode="r") if os.path.exists(LIB) else None
    sets = {td: [(np.array([gidx[rc] for rc in q["sets"][td][0]], dtype=np.int64),
                  np.array([gidx[rc] for rc in q["sets"][td][1]], dtype=np.int64)) for q in qinfo] for td in ("std", "sc")}
    stems = np.array([q["stem"] for q in qinfo])
    cboot = np.random.default_rng(0).integers(0, len(np.unique(stems)), (2000, len(np.unique(stems))))
    boot = bench.compound_bootstrap(stems)

    def pearson_all(x):
        """Fixed-grid Pearson of the rows of x against the whole clean library, in blocks."""
        from xrdclip.eval import wcc_torch as WT
        q = torch.from_numpy(WT.standardize_rows(x))
        out = np.empty((len(x), N), np.float32)
        for i in range(0, N, 32768):
            r = torch.from_numpy(WT.standardize_rows(np.asarray(lib[i:i + 32768], dtype=np.float32)))
            out[:, i:i + len(r)] = WT.pearson(q, r).numpy()
        return out

    res = {"estimates": {"fwhm_deg_median": float(np.median([e["fwhm"] for e in ests if e["fwhm"]])),
                         "fwhm_deg_iqr": [float(np.percentile([e["fwhm"] for e in ests if e["fwhm"]], p)) for p in (25, 75)],
                         "noise_median": float(np.median([e["noise"] for e in ests])),
                         "zero_deg_abs_median": float(np.median([abs(e["zero"]) for e in ests])),
                         "n_partial_coverage": int(sum((~e["mask"]).any() for e in ests)),
                         "n_cu_lab": int(sum(e["tube"] == "cu" for e in ests))},
           "results": json.load(open(args.out))["results"] if (args.variants and os.path.exists(args.out)) else {}}
    for name, x in X.items():
        e = encode(model, x)
        scorers = {"model": (e @ lib_emb.T).numpy()}
        if pemb is not None:
            scorers["p2p"] = (e @ pemb.T).numpy()
        if lib is not None:
            scorers["pearson"] = pearson_all(x)
        for sname, S in scorers.items():
            r = rs.ranks_from_scores(S, sets, nq)
            for td in ("std", "sc"):
                k = f"{name}|{sname}|{td}"
                res["results"][k] = {"per_pattern": bench.summarise(r[td], np.ones(nq, bool), boot),
                                     "macro_top1": pn.macro(r[td], 1, stems, cboot),
                                     "macro_top10": pn.macro(r[td], 10, stems, cboot)}
                pp = res["results"][k]["per_pattern"]
                print(f"{k:28s} top-1 {pp['top1']:.3f} top-10 {pp['top10']:.3f} median {pp['median_rank']:.0f}  "
                      f"macro top-1 {res['results'][k]['macro_top1']['value']:.3f}", flush=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"[eval] -> {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("render", "pemb", "eval"))
    ap.add_argument("--truth", default=f"{bench.DATA}/opxrd/opxrd_truth.json")
    ap.add_argument("--patterns", default=f"{bench.DATA}/opxrd/opxrd_cnrs_organic.json")
    ap.add_argument("--ckpt", default="runs/xrdclip_poly/best.pt")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="results/model_gap.json")
    ap.add_argument("--variants", default="", help="comma-separated variant names to (re)run; merged into --out")
    args = ap.parse_args()
    return {"render": cmd_render, "pemb": cmd_pemb, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
