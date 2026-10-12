"""Full-library real benchmark: every identified opXRD organic pattern against the whole stick cache.

scripts/17_... scored 95 patterns against a 2,108-entry library with one "best" CSD entry per pattern
as the only correct answer. This version is the benchmark run:
  * Queries: every pattern that scripts/18_... identified by packing similarity (not only the ones
    whose entry happens to be in the training corpus).
  * Correct answers: every CSD entry of the same solid form (15/15 packing match), at any
    temperature. Entries that could not be compared are ignored, never scored as errors.
  * Library: all 308,862 cached stick patterns plus the target families from cache/sticks_extra.
  * Two truth definitions:
      std  any same-form entry counts;
      sc   single-crystal references only: same-form powder-study entries are ignored, so a pattern
           whose own Rietveld structure is in the CSD has to find an independent determination.

Ranks are exact. A pre-pass scores every truth / distractor / ignore entry ("special" entries) and
fixes each query's threshold (its best-scoring correct answer). Library chunks are then streamed
through worker processes, which store one bit per (query, entry): "scores above the threshold".
Ranks for any library subset (the library-size curve) are popcounts over those bits.

Hybrid = model top-K shortlist re-ranked by a classical score; workers also record classical scores
at each query's shortlist entries.

  PYTHONPATH=src python scripts/19_real_bench.py --workers 4
  PYTHONPATH=src python scripts/19_real_bench.py --limit 3000 --work cache/real_bench_smoke --verify
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import json
import os
import time

import numpy as np

DATA = "C:/Users/frank/xrd-clip-data"
STICK_DIRS = ("cache/sticks", "cache/sticks_extra")
RENDER_FIELDS = ("two_theta", "intensity", "hkl", "d_hkl", "recip")
EMBED_FIELDS = ("atom_z", "atom_frac", "cell", "n_atoms")
SPECS = {"cos_zs": dict(l_deg=0.0), "wcc_1": dict(l_deg=1.0), "wcc_2": dict(l_deg=2.0), "lnq": dict(lnq=True)}
# lnq costs ~10x the others (a 16-step zero-offset scan per query), so it runs on the bkg variant only.
VARIANTS = {"pearson": ("raw", "bkg"), "cos_zs": ("raw", "bkg"), "wcc_1": ("raw", "bkg"),
            "wcc_2": ("raw", "bkg"), "lnq": ("bkg",)}
CLASSICAL_KEYS = [f"{m}_{v}" for m, vs in VARIANTS.items() for v in vs]
MODEL_KEYS = ["model_raw", "model_bkg"]
K_HYBRID = (100, 1000)


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------
def unthrottle() -> None:
    """Windows parks windowless background processes on the E-cores; opt this process out."""
    if os.name != "nt":
        return

    class _State(ctypes.Structure):
        _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
    state = _State(1, 1, 0)       # PROCESS_POWER_THROTTLING_EXECUTION_SPEED off
    k32.SetProcessInformation(k32.GetCurrentProcess(), 4, ctypes.byref(state), ctypes.sizeof(state))


def save_npz(path: str, **arrays) -> None:
    """Atomic write; os.replace retries because a reader holding the file blocks it on Windows."""
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    for _ in range(40):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.25)
    os.replace(tmp, path)


def shard_table(limit: "int | None") -> "list[tuple[str, int, int]]":
    """[(path, n_entries, global_offset)] over the stick dirs, optionally truncated to `limit` per shard."""
    rows, off = [], 0
    for d in STICK_DIRS:
        for p in sorted(glob.glob(os.path.join(d, "*.npz"))):
            with np.load(p, allow_pickle=True) as z:
                n = len(z["refcode"])
            n = min(n, limit) if (limit and d == STICK_DIRS[0]) else n
            rows.append((p, n, off))
            off += n
    return rows


def load_records(path: str, lo: int, hi: int, fields) -> "list[dict]":
    with np.load(path, allow_pickle=True) as z:
        cols = {f: z[f][lo:hi] for f in fields}
    return [{f: cols[f][i] for f in fields} for i in range(hi - lo)]


def render(recs: "list[dict]") -> np.ndarray:
    from xrdclip.data import augment as A
    return np.stack([A.render_sticks(r, A.CLEAN) for r in recs]).astype(np.float32)


def build_queries(truth_path: str, patterns_path: str, gidx: dict) -> "tuple[list, dict, dict]":
    """Identified opXRD patterns -> (qinfo, excluded, Q). Truth sets keep only refcodes in `gidx`."""
    from xrdclip.data import augment as A
    from xrdclip.eval.real_patterns import query_patterns, to_cu_grid
    all_truth = json.load(open(truth_path))
    pats = {o["id"]: o for o in json.load(open(patterns_path))["organic"]}
    excluded = {t["id"]: "not identified by packing similarity" for t in all_truth if not t["true_stems"]}
    for t in all_truth:
        if t["true_stems"]:
            o = pats[t["id"]]
            _, m = to_cu_grid(o["two_theta"], o["intensity"], float(t["wavelength"] or A.WAVELENGTH))
            if m.sum() * A.STEP < 10.0:       # pattern_706: 2theta axis runs 0.01-0.83 (radians?)
                excluded[t["id"]] = f"covers {m.sum() * A.STEP:.1f} deg of the 5-50 deg grid (axis {o['tt_range']})"
    truth = [t for t in all_truth if t["id"] not in excluded]
    qinfo = []
    for t in truth:
        same = [x for x in t["same_form"] if x["refcode"] in gidx]
        unk = [x["refcode"] for x in t["unknown"] if x["refcode"] in gidx]
        sets = {
            "std": ([x["refcode"] for x in same], unk),
            "sc": ([x["refcode"] for x in same if not x["is_powder_study"]],
                   unk + [x["refcode"] for x in same if x["is_powder_study"]]),
        }
        # the pattern's own Rietveld structure is in the CSD: an anchor that is a powder study and
        # reproduces the opXRD structure (rmsd against it < 0.05 A)
        own_powder = any(x["is_anchor"] and x["is_powder_study"] and x["rmsd"] < 0.05 for x in t["same_form"])
        qinfo.append({"id": t["id"], "stem": t["true_stems"][0], "sets": sets, "own_powder": own_powder,
                      "other": [x["refcode"] for x in t["other_form"] if x["refcode"] in gidx],
                      "wavelength": t["wavelength"], "n_same_total": len(t["same_form"])})
    Q = {}
    for v in ("raw", "bkg"):
        Q[f"raw_{v}"] = np.stack([query_patterns(pats[q["id"]], q["wavelength"], v) for q in qinfo])
        Q[f"std_{v}"] = np.stack([A.standardize(y) for y in Q[f"raw_{v}"]])
    return qinfo, excluded, Q


def compound_bootstrap(stems, n: int = 2000, seed: int = 0) -> "list[np.ndarray]":
    """Query-index resamples drawn by compound (stem), so repeat patterns of one compound move together."""
    rng = np.random.default_rng(seed)
    stems = np.asarray(stems)
    uniq = np.unique(stems)
    members = {s: np.nonzero(stems == s)[0] for s in uniq}
    return [np.concatenate([members[s] for s in rng.choice(uniq, len(uniq))]) for _ in range(n)]


def summarise(r: np.ndarray, sel: np.ndarray, boot: "list[np.ndarray]") -> dict:
    """Rank summary (r = number of wrong entries above the best correct one; -1 = no truth in library)."""
    ok = sel & (r >= 0)
    if not ok.any():
        return {"n": 0}
    rv = r[ok]
    d = {"n": int(ok.sum()), "median_rank": float(np.median(rv) + 1), "mrr": float((1 / (rv + 1)).mean())}
    for kk in (1, 10, 100):
        d[f"top{kk}"] = float((rv < kk).mean())
        bs = []
        for b in boot:
            bb = b[ok[b]]
            if len(bb):
                bs.append(float((r[bb] < kk).mean()))
        d[f"top{kk}_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return d


def score_block(ref_raw: np.ndarray, Q: dict) -> "dict[str, np.ndarray]":
    """All classical scores of one reference block against every query. {key: (n_query, n_ref)}."""
    from xrdclip.data import augment as A
    from xrdclip.eval import strong_baselines as SB
    out = {}
    r_std = np.stack([A.standardize(y) for y in ref_raw])
    for v in VARIANTS["pearson"]:
        out[f"pearson_{v}"] = Q[f"std_{v}"] @ r_std.T / r_std.shape[1]
    for m, spec in SPECS.items():
        lib = SB.WCCLibrary(ref_raw, **spec)
        for v in VARIANTS[m]:
            out[f"{m}_{v}"] = lib.scores(Q[f"raw_{v}"])
        del lib
    return out


# ------------------------------------------------------------------------------------------------
# worker
# ------------------------------------------------------------------------------------------------
_W: dict = {}


def _init_worker(ctx: dict) -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    unthrottle()
    _W.update(ctx)


def _embed_shard(job) -> "tuple[str, float]":
    """Structure embeddings for one shard, saved to its own .npy. Runs in a throwaway process: the
    first full run grew ~1.8 GB per embedded shard inside one long-lived process (6 GB after two)."""
    path, n, ckpt, embed_dim, out_npy = job
    if os.path.exists(out_npy):
        return out_npy, 0.0
    import torch
    from xrdclip.eval.retrieve import embed_library
    from xrdclip.models.clip import XRDClip
    unthrottle()
    torch.set_num_threads(4)
    t0 = time.time()
    model = XRDClip(embed_dim=embed_dim)
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    model.eval()
    store = {str(i): r for i, r in enumerate(load_records(path, 0, n, EMBED_FIELDS))}
    with torch.no_grad():
        emb = embed_library(model, store, [str(i) for i in range(n)], "cpu", batch=128)
    np.save(out_npy + ".tmp.npy", emb.astype(np.float32))
    os.replace(out_npy + ".tmp.npy", out_npy)
    return out_npy, time.time() - t0


def _run_chunk(job) -> "tuple[int, float]":
    cid, path, lo, hi, g0 = job
    out_path = os.path.join(_W["work"], "chunks", f"chunk_{cid:04d}.npz")
    if os.path.exists(out_path):
        return cid, 0.0
    t0 = time.time()
    S = score_block(render(load_records(path, lo, hi, RENDER_FIELDS)), _W["Q"])
    g = np.arange(g0, g0 + hi - lo)
    special = np.isin(g, _W["special"])
    arrays = {}
    for key, s in S.items():
        s = s.astype(np.float32)
        # hybrid: classical scores at shortlist entries that live in this chunk (before any masking)
        topk = _W["topk"][key.split("_")[-1]]                 # (n_query, K) global indices
        qi, ki = np.nonzero((topk >= g0) & (topk < g0 + hi - lo))
        arrays[f"{key}__hyb_q"], arrays[f"{key}__hyb_k"] = qi.astype(np.int16), ki.astype(np.int16)
        arrays[f"{key}__hyb_s"] = s[qi, topk[qi, ki] - g0]
        # diagnostics: the 10 best entries per query, unmasked
        top = np.argsort(-s, axis=1)[:, :10]
        arrays[f"{key}__top_i"] = (top + g0).astype(np.int32)
        arrays[f"{key}__top_s"] = np.take_along_axis(s, top, axis=1)
        s[:, special] = -np.inf                               # special entries are counted in the pre-pass
        for td in ("std", "sc"):
            arrays[f"{key}__bits_{td}"] = np.packbits(s > _W["thr"][td][key][:, None], axis=1)
    save_npz(out_path, n=np.array(hi - lo), g0=np.array(g0), **arrays)
    return cid, time.time() - t0


def chunk_jobs(shards, chunk: int) -> list:
    jobs, cid = [], 0
    for p, n, off in shards:
        for lo in range(0, n, chunk):
            jobs.append((cid, p, lo, min(n, lo + chunk), off + lo))
            cid += 1
    return jobs


def run_chunks(jobs: list, ctx: dict, workers: int) -> None:
    from multiprocessing import Pool
    todo = [j for j in jobs if not os.path.exists(os.path.join(ctx["work"], "chunks", f"chunk_{j[0]:04d}.npz"))]
    print(f"[bench] {len(jobs)} chunks, {len(todo)} to run on {workers} workers", flush=True)
    if not todo:
        return
    t0 = time.time()
    # maxtasksperchild=1: a fresh process per chunk, so nothing can accumulate across chunks
    with Pool(workers, initializer=_init_worker, initargs=(ctx,), maxtasksperchild=1) as pool:
        for done, (c, dt) in enumerate(pool.imap_unordered(_run_chunk, todo), 1):
            el = time.time() - t0
            print(f"[bench] chunk {c} {dt:.0f}s  ({done}/{len(todo)}, elapsed {el / 60:.1f} min, "
                  f"eta {el / done * (len(todo) - done) / 60:.0f} min)", flush=True)


def save_ctx(path: str, ctx: dict, n_library: int) -> None:
    arrays = {"special": ctx["special"], "n_library": np.array(n_library)}
    arrays.update({f"Q__{k}": v for k, v in ctx["Q"].items()})
    arrays.update({f"topk__{k}": v for k, v in ctx["topk"].items()})
    arrays.update({f"thr__{td}__{k}": v for td, d in ctx["thr"].items() for k, v in d.items()})
    save_npz(path, **arrays)


def box_main(args) -> int:
    """Rented-box stage: only the classical chunks. Library order must match the local one exactly."""
    with np.load(args.box_ctx) as z:
        ctx = {"work": args.work, "special": z["special"],
               "Q": {k[3:]: z[k] for k in z.files if k.startswith("Q__")},
               "topk": {k[6:]: z[k] for k in z.files if k.startswith("topk__")},
               "thr": {td: {k.split("__")[2]: z[k] for k in z.files if k.startswith(f"thr__{td}__")}
                       for td in ("std", "sc")}}
        n_expected = int(z["n_library"])
    shards = shard_table(args.limit)
    n = sum(s[1] for s in shards)
    assert n == n_expected, f"library mismatch: box has {n:,} entries, context expects {n_expected:,}"
    os.makedirs(os.path.join(args.work, "chunks"), exist_ok=True)
    run_chunks(chunk_jobs(shards, args.chunk), ctx, args.workers)
    print("[bench] BOX_DONE", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default=f"{DATA}/opxrd/opxrd_truth.json")
    ap.add_argument("--patterns", default=f"{DATA}/opxrd/opxrd_cnrs_organic.json")
    ap.add_argument("--ckpt", default="runs/xrdclip_poly/best.pt")
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--work", default="cache/real_bench")
    ap.add_argument("--chunk", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N entries of each cache shard")
    ap.add_argument("--verify", action="store_true", help="brute-force rank check (small libraries only)")
    ap.add_argument("--out", default="results/real_bench.json")
    ap.add_argument("--prep_only", action="store_true",
                    help="stop before the chunk pool and write <work>/box_ctx.npz for --box_ctx")
    ap.add_argument("--box_ctx", default=None,
                    help="rented-box mode: run only the classical chunks from a --prep_only context "
                         "(needs just the stick fields used for rendering; no checkpoint, no atoms)")
    args = ap.parse_args()

    if args.box_ctx:
        return box_main(args)

    import torch
    from xrdclip.models.clip import XRDClip

    unthrottle()
    torch.set_num_threads(4)
    os.makedirs(os.path.join(args.work, "chunks"), exist_ok=True)

    # ---- library index --------------------------------------------------------------------------
    shards = shard_table(args.limit)
    lib_rc = []
    for p, n, _ in shards:
        with np.load(p, allow_pickle=True) as z:
            lib_rc += [str(x) for x in z["refcode"][:n]]
    N = len(lib_rc)
    gidx = {rc: i for i, rc in enumerate(lib_rc)}
    print(f"[bench] library {N:,} entries from {len(shards)} shards", flush=True)

    # ---- queries + truth sets -------------------------------------------------------------------
    qinfo, excluded, Q = build_queries(args.truth, args.patterns, gidx)
    nq = len(qinfo)
    print(f"[bench] {nq} identified patterns; std truth in library for "
          f"{sum(bool(q['sets']['std'][0]) for q in qinfo)}, sc truth for "
          f"{sum(bool(q['sets']['sc'][0]) for q in qinfo)}", flush=True)

    # ---- model: library embeddings + query embeddings -------------------------------------------
    model = XRDClip(embed_dim=args.embed_dim)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.eval()
    from multiprocessing import Pool
    emb_jobs = [(p, n, args.ckpt, args.embed_dim,
                 os.path.join(args.work, f"emb_{os.path.basename(os.path.dirname(p))}_{os.path.basename(p)[:-4]}"
                                         f"_{n}.npy")) for p, n, _ in shards]
    t0 = time.time()
    with Pool(1, maxtasksperchild=1) as pool:          # one shard per fresh process (see _embed_shard)
        for out_npy, dt in pool.imap(_embed_shard, emb_jobs):
            print(f"[bench] embedded {out_npy} {dt:.0f}s (elapsed {time.time() - t0:.0f}s)", flush=True)
    lib_emb = np.concatenate([np.load(j[-1]) for j in emb_jobs]).astype(np.float32)
    assert lib_emb.shape[0] == N
    q_emb = {}
    with torch.no_grad():
        for v in ("raw", "bkg"):
            q_emb[v] = model.encode_pattern(torch.from_numpy(Q[f"std_{v}"]).unsqueeze(1)).numpy()
    model_S = {f"model_{v}": (q_emb[v] @ lib_emb.T).astype(np.float32) for v in ("raw", "bkg")}
    topk = {v: np.argsort(-model_S[f"model_{v}"], axis=1)[:, :max(K_HYBRID)].astype(np.int64)
            for v in ("raw", "bkg")}

    # ---- pre-pass over special entries ----------------------------------------------------------
    special_rc = sorted({rc for q in qinfo for td in ("std", "sc") for part in q["sets"][td] for rc in part}
                        | {rc for q in qinfo for rc in q["other"]})
    special = np.array(sorted(gidx[rc] for rc in special_rc), dtype=np.int64)
    sp_col = {int(g): j for j, g in enumerate(special)}
    by_shard = {}
    for g in special:
        for si, (p, n, off) in enumerate(shards):
            if off <= g < off + n:
                by_shard.setdefault(si, []).append(int(g))
    sp_recs = [None] * len(special)
    for si, gs in by_shard.items():
        p, n, off = shards[si]
        with np.load(p, allow_pickle=True) as z:
            cols = {f: z[f] for f in RENDER_FIELDS}
        for g in gs:
            sp_recs[sp_col[g]] = {f: cols[f][g - off] for f in RENDER_FIELDS}
    S_sp = score_block(render(sp_recs), Q)
    for k in MODEL_KEYS:
        S_sp[k] = model_S[k][:, special]
    print(f"[bench] pre-pass: {len(special)} special entries scored", flush=True)
    save_npz(os.path.join(args.work, "prepass.npz"), special_rc=np.array([lib_rc[g] for g in special]),
             query_ids=np.array([q["id"] for q in qinfo]), **{k: s for k, s in S_sp.items()})

    thr = {td: {} for td in ("std", "sc")}
    pre = {td: {} for td in ("std", "sc")}
    wf = {}
    for k, s in S_sp.items():
        for td in ("std", "sc"):
            th, cnt = np.full(nq, np.inf, np.float32), np.zeros(nq, np.int64)
            for qi, q in enumerate(qinfo):
                tr, ig = q["sets"][td]
                if not tr:
                    continue
                cols_t = [sp_col[gidx[rc]] for rc in tr]
                th[qi] = s[qi, cols_t].max()
                excl = set(cols_t) | {sp_col[gidx[rc]] for rc in ig}
                keep = np.array([j for j in range(len(special)) if j not in excl], dtype=np.int64)
                cnt[qi] = int((s[qi, keep] > th[qi]).sum()) if len(keep) else 0
            thr[td][k], pre[td][k] = th, cnt
        w = []
        for qi, q in enumerate(qinfo):
            tr = q["sets"]["std"][0]
            if tr and q["other"]:
                best_t = s[qi, [sp_col[gidx[rc]] for rc in tr]].max()
                w.append(int(best_t >= s[qi, [sp_col[gidx[rc]] for rc in q["other"]]].max()))
            else:
                w.append(-1)
        wf[k] = np.array(w)

    # ---- model bits (full score matrix is in memory) --------------------------------------------
    bits = {td: {} for td in ("std", "sc")}
    for k in MODEL_KEYS:
        s = model_S[k] if not args.verify else model_S[k].copy()
        s[:, special] = -np.inf
        for td in ("std", "sc"):
            bits[td][k] = s > thr[td][k][:, None]
        del s
    del lib_emb
    if not args.verify:
        del model_S        # ~0.5 GB; the low-memory reaper has killed a run like this before

    # ---- classical: stream the library through workers ------------------------------------------
    ctx = {"work": args.work, "Q": Q, "special": special, "topk": topk,
           "thr": {td: {k: thr[td][k] for k in CLASSICAL_KEYS} for td in thr}}
    if args.prep_only:
        save_ctx(os.path.join(args.work, "box_ctx.npz"), ctx, N)
        print(f"[bench] wrote {args.work}/box_ctx.npz (library {N:,}); run --box_ctx on the box, "
              f"copy chunks/ back, then rerun without flags to merge", flush=True)
        return 0
    jobs = chunk_jobs(shards, args.chunk)
    run_chunks(jobs, ctx, args.workers)

    # ---- merge ----------------------------------------------------------------------------------
    for td in ("std", "sc"):
        for k in CLASSICAL_KEYS:
            bits[td][k] = np.zeros((nq, N), bool)
    hyb = {k: np.full((nq, max(K_HYBRID)), np.nan, np.float32) for k in CLASSICAL_KEYS}
    top10 = {k: [] for k in CLASSICAL_KEYS}
    for c, p, lo, hi, g0 in jobs:
        with np.load(os.path.join(args.work, "chunks", f"chunk_{c:04d}.npz")) as z:
            n = int(z["n"])
            for k in CLASSICAL_KEYS:
                for td in ("std", "sc"):
                    bits[td][k][:, g0:g0 + n] = np.unpackbits(z[f"{k}__bits_{td}"], axis=1, count=n).astype(bool)
                hyb[k][z[f"{k}__hyb_q"].astype(int), z[f"{k}__hyb_k"].astype(int)] = z[f"{k}__hyb_s"]
                top10[k].append((z[f"{k}__top_i"], z[f"{k}__top_s"]))

    ranks = {td: {} for td in ("std", "sc")}
    for td in ("std", "sc"):
        for k in CLASSICAL_KEYS + MODEL_KEYS:
            r = pre[td][k] + bits[td][k].sum(axis=1)
            ranks[td][k] = np.where(np.isfinite(thr[td][k]), r, -1)       # -1 = no truth in library

    # hybrid: rerank the model shortlist with a classical score (truth must survive the shortlist)
    for td in ("std", "sc"):
        for k in CLASSICAL_KEYS:
            v = k.split("_")[-1]
            for K in K_HYBRID:
                rr = np.full(nq, -1, np.int64)
                for qi, q in enumerate(qinfo):
                    tr, ig = q["sets"][td]
                    if not tr:
                        continue
                    short = topk[v][qi, :K]
                    sc = hyb[k][qi, :K]
                    is_t = np.isin(short, [gidx[rc] for rc in tr])
                    is_i = np.isin(short, [gidx[rc] for rc in ig])
                    if not is_t.any():
                        rr[qi] = N                               # dropped by the shortlist
                        continue
                    t_best = sc[is_t].max()
                    rr[qi] = int((sc[~is_t & ~is_i] > t_best).sum())
                ranks[td][f"model@{K}->{k}"] = rr

    # ---- verify (small libraries): brute-force ranks from full score matrices --------------------
    if args.verify:
        assert N <= 50000, "--verify needs a small library (use --limit)"
        recs = []
        for p, n, _ in shards:
            recs += load_records(p, 0, n, RENDER_FIELDS)
        S_full = score_block(render(recs), Q)
        S_full.update(model_S)
        bad = 0
        for k, s in S_full.items():
            for td in ("std", "sc"):
                for qi, q in enumerate(qinfo):
                    tr, ig = q["sets"][td]
                    if not tr:
                        continue
                    ti = [gidx[rc] for rc in tr]
                    mask = np.ones(N, bool)
                    mask[ti] = False
                    mask[[gidx[rc] for rc in ig]] = False
                    bf = int((s[qi, mask] > s[qi, ti].max()).sum())
                    if bf != ranks[td][k][qi]:
                        bad += 1
                        if bad <= 10:
                            print(f"[verify] MISMATCH {k} {td} {q['id']}: stream {ranks[td][k][qi]} brute {bf}")
        print(f"[verify] {'OK' if not bad else f'{bad} mismatches'} "
              f"({len(S_full)} keys x 2 truth defs x {nq} queries)", flush=True)

    # ---- summaries + bootstrap CIs ----------------------------------------------------------------
    boot = compound_bootstrap([q["stem"] for q in qinfo])
    split_of = {"all": np.ones(nq, bool),
                "powder_study": np.array([q["own_powder"] for q in qinfo]),
                "independent": np.array([not q["own_powder"] for q in qinfo])}

    res = {"n_library": N, "n_queries": nq, "excluded": excluded, "keys": list(ranks["std"]), "summary": {},
           "within_family": {},
           "library_curve": {}, "meta": [{"id": q["id"], "stem": q["stem"], "own_powder": q["own_powder"],
                                          "n_same_in_lib": len(q["sets"]["std"][0]),
                                          "n_sc_in_lib": len(q["sets"]["sc"][0]), "n_other": len(q["other"])}
                                         for q in qinfo]}
    for td in ("std", "sc"):
        for k, r in ranks[td].items():
            for sp, sel in split_of.items():
                if td == "sc" and sp != "all":
                    continue
                res["summary"][f"{td}|{sp}|{k}"] = summarise(r, sel, boot)
    for k, w in wf.items():
        ok = w >= 0
        res["within_family"][k] = {"n": int(ok.sum()), "top1": float(w[ok].mean()) if ok.any() else None}

    # library-size curve: random decoy subsets (special entries always present), bkg variant, std truth
    nonspecial = np.setdiff1d(np.arange(N), special)
    for k in [x for x in CLASSICAL_KEYS + MODEL_KEYS if x.endswith("bkg")]:
        curve = {}
        for size in (2000, 20000, 100000):
            if size >= len(nonspecial):
                continue
            t1 = []
            for rep in range(5):
                sub = np.random.default_rng(rep).choice(nonspecial, size, replace=False)
                r = pre["std"][k] + bits["std"][k][:, sub].sum(axis=1)
                ok = np.isfinite(thr["std"][k])
                t1.append(float((r[ok] < 1).mean()))
            curve[str(size)] = [float(np.mean(t1)), float(np.std(t1))]
        r = ranks["std"][k]
        curve["full"] = [float((r[r >= 0] < 1).mean()), 0.0]
        res["library_curve"][k] = curve

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    np.savez_compressed(os.path.join(args.work, "ranks.npz"),
                        **{f"{td}|{k}": r for td in ranks for k, r in ranks[td].items()})
    for td, sp in (("std", "all"), ("std", "independent"), ("sc", "all")):
        print(f"\n[bench] {td} / {sp}")
        for k in ranks[td]:
            s = res["summary"].get(f"{td}|{sp}|{k}", {})
            if s.get("n"):
                print(f"  {k:24s} n={s['n']:3d} top1 {s['top1']:.2f} [{s['top1_ci'][0]:.2f},{s['top1_ci'][1]:.2f}]"
                      f"  top10 {s['top10']:.2f}  median {s['median_rank']:.0f}")
    print(f"[bench] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
