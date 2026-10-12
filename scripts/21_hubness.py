"""Hubness correction for the full-library real benchmark (plan Section 12b).

Section 12a: at full library size a few sparse library patterns (one dominant peak near 22-23 deg) win
many queries; GUSRIV is the top-1 entry for 20 of the 52 cos + zero-scan misses. Hub reduction penalises
library entries that score high against everything. That needs every score again, and scripts/19_... kept
only rank bits, so this is a second pass:

  prep   (local) queries + 2 x n_probe probe patterns drawn from the library, never the test patterns:
                 "sim" = clean simulated; "aug" = training-range degradations followed by the same
                 background removal as the real queries. Probes come only from compound families
                 (6-letter stems) that no query belongs to.                      -> <work>/hub_ctx.npz
  score  (GPU)   every classical score of queries and probes against the whole library, as lag-sliced
                 GEMMs (xrdclip.eval.wcc_torch, equal to the CPU scorer to < 1e-6). Keeps the full query
                 rows and, per library entry, its probe-score statistics.         -> <work>/full/
  eval   (local) ranks for uncorrected and corrected scores, bootstrap CIs, hub diagnostics. The model is
                 corrected here too (probe patterns through the pattern encoder).
  check  (local) torch scorer vs the CPU pre-pass scores saved by 19.

Corrections. These were fixed before any corrected number was seen; the primary is CSLS with aug probes, k = 10:
  csls_<set>_k<k>  s' = s - h_k(r) / 2, h_k(r) = mean of entry r's k best probe scores (Conneau et al.,
                   ICLR 2018). The query-side CSLS term is constant per query and cannot change a rank.
  znorm_<set>      (s - mean_r) / std_r over the probes.
  mp_<set>         mutual proximity, Gaussian (Schnitzer et al., JMLR 2012): log Phi(z_q) + log Phi(z_r),
                   z_q from the query's scores over the library, z_r from the entry's probe scores.
Same-stem probe/entry pairs are left out of every statistic: a probe's own refcode family is not a hub signal.

  PYTHONPATH=src python scripts/21_hubness.py check
  PYTHONPATH=src python scripts/21_hubness.py prep
  PYTHONPATH=src python scripts/21_hubness.py score --device cuda --workers 12        (rented box)
  PYTHONPATH=src python scripts/21_hubness.py eval
  smoke: prep / score / eval with --limit 200 --work cache/real_bench_smoke --n_probe 300 --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, deque
from importlib import import_module
from itertools import islice

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
bench = import_module("19_real_bench")

KS = (1, 5, 10, 20, 50, 100)        # CSLS neighbourhood sizes kept per library entry
NN = 100                            # nearest library entries kept per probe (k-occurrence)
METHODS = ("pearson",) + tuple(bench.SPECS)
PROBE_SETS = ("sim", "aug")
PRIMARY = "csls_aug_k10"
CORRECTIONS = (["none", PRIMARY, "csls_sim_k10"]
               + [f"csls_{ps}_k{k}" for ps in PROBE_SETS for k in KS if k != 10]
               + [f"znorm_{ps}" for ps in PROBE_SETS] + [f"mp_{ps}" for ps in PROBE_SETS])
DATA = bench.DATA


def library_refcodes(shards) -> "list[str]":
    rc = []
    for p, n, _ in shards:
        with np.load(p, allow_pickle=True) as z:
            rc += [str(x) for x in z["refcode"][:n]]
    return rc


def stems_of(refcodes) -> np.ndarray:
    """Integer id of each refcode's 6-letter CSD stem (same ids wherever the library order is the same)."""
    _, inv = np.unique(np.array([str(r)[:6] for r in refcodes]), return_inverse=True)
    return inv.astype(np.int64)


def records_at(shards, gs, fields) -> "list[dict]":
    """Records at global library indices `gs`, in that order."""
    out = [None] * len(gs)
    want = {}
    for j, g in enumerate(gs):
        for si, (p, n, off) in enumerate(shards):
            if off <= g < off + n:
                want.setdefault(si, []).append((j, int(g) - off))
    for si, items in want.items():
        with np.load(shards[si][0], allow_pickle=True) as z:
            cols = {f: z[f] for f in fields}
        for j, i in items:
            out[j] = {f: cols[f][i] for f in fields}
    return out


def special_refcodes(qinfo) -> "set[str]":
    return ({rc for q in qinfo for td in ("std", "sc") for part in q["sets"][td] for rc in part}
            | {rc for q in qinfo for rc in q["other"]})


def probe_stats(sp, mask, ks):
    """Per library entry (column): mean, std and mean-of-top-k over the probes (rows); masked pairs skipped."""
    import torch
    keep = (~mask).to(sp.dtype)
    cnt = keep.sum(0).clamp(min=1)
    mean = (sp * keep).sum(0) / cnt
    std = torch.sqrt((((sp - mean) * keep) ** 2).sum(0) / cnt)
    top = torch.topk(sp.masked_fill(mask, float("-inf")), max(ks), dim=0).values
    return mean, std, torch.stack([top[:k].mean(0) for k in ks])


def merge_nn(nn_s, nn_i, sp, mask, g0: int):
    """Running per-probe top-NN library entries (rows = probes)."""
    import torch
    v, i = torch.topk(sp.masked_fill(mask, float("-inf")), min(NN, sp.shape[1]), dim=1)
    s2, i2 = torch.cat([nn_s, v], 1), torch.cat([nn_i, i + g0], 1)
    v3, j = torch.topk(s2, NN, dim=1)
    return v3, torch.gather(i2, 1, j)


def new_stats(N: int, P: int, nk: int, dev: str) -> dict:
    import torch
    return {"mean": np.zeros(N, np.float32), "std": np.zeros(N, np.float32), "topk": np.zeros((nk, N), np.float32),
            "nn_s": torch.full((P, NN), float("-inf"), device=dev), "nn_i": torch.zeros((P, NN), dtype=torch.int64, device=dev)}


def update_stats(st: dict, sp, mask, g0: int, ks) -> None:
    C = sp.shape[1]
    mean, std, topk = probe_stats(sp, mask, ks)
    st["mean"][g0:g0 + C] = mean.cpu().numpy()
    st["std"][g0:g0 + C] = std.cpu().numpy()
    st["topk"][:, g0:g0 + C] = topk.cpu().numpy()
    st["nn_s"], st["nn_i"] = merge_nn(st["nn_s"], st["nn_i"], sp, mask, g0)


# ------------------------------------------------------------------------------------------------
# check: torch scorer vs the CPU pre-pass of 19
# ------------------------------------------------------------------------------------------------
def cmd_check(args) -> int:
    import torch
    from xrdclip.eval import wcc_torch as WT
    WT.full_precision()
    ctx = np.load(os.path.join(args.work, "box_ctx.npz"))
    pre = np.load(os.path.join(args.work, "prepass.npz"))
    R = bench.render(records_at(bench.shard_table(args.limit), ctx["special"], bench.RENDER_FIELDS))
    Q = {k[3:]: ctx[k] for k in ctx.files if k.startswith("Q__")}
    worst = 0.0
    r_std = torch.from_numpy(WT.standardize_rows(R))
    got = {f"pearson_{v}": WT.pearson(torch.from_numpy(Q[f"std_{v}"]), r_std).numpy() for v in ("raw", "bkg")}
    for m, spec in bench.SPECS.items():
        sc = WT.TorchWCC(**spec)
        refs = sc.prep_refs(R)
        for v in bench.VARIANTS[m]:
            got[f"{m}_{v}"] = sc.scores(sc.prep_rows(Q[f"raw_{v}"]), refs).numpy()
    for k, s in got.items():
        d = float(np.abs(s - pre[k]).max())
        worst = max(worst, d)
        print(f"[check] {k:12s} max |torch - cpu| {d:.2e}  argmax changes {int((s.argmax(1) != pre[k].argmax(1)).sum())}")
    print(f"[check] {'OK' if worst < 1e-5 else 'FAIL'} worst {worst:.2e} over {len(got)} keys x {R.shape[0]} entries")
    return 0 if worst < 1e-5 else 1


# ------------------------------------------------------------------------------------------------
# prep (local): queries + probe patterns
# ------------------------------------------------------------------------------------------------
def cmd_prep(args) -> int:
    from xrdclip.data import augment as A
    from xrdclip.eval.real_patterns import preprocess
    shards = bench.shard_table(args.limit)
    lib_rc = library_refcodes(shards)
    gidx = {rc: i for i, rc in enumerate(lib_rc)}
    qinfo, _, Q = bench.build_queries(args.truth, args.patterns, gidx)
    bad = {rc[:6] for rc in special_refcodes(qinfo)} | {q["stem"][:6] for q in qinfo}
    eligible = np.array([i for i, rc in enumerate(lib_rc) if rc[:6] not in bad])
    P = args.n_probe
    pick = np.random.default_rng(args.seed).choice(eligible, 2 * P, replace=False)
    recs = records_at(shards, pick, bench.RENDER_FIELDS)
    sim = bench.render(recs[:P])
    rng = np.random.default_rng(args.seed + 1)
    full = np.ones(A.N_POINTS, bool)
    aug = np.stack([preprocess(A.render_sticks(r, A.sample_params(rng), rng), full, bkg=True) for r in recs[P:]])
    out = os.path.join(args.work, "hub_ctx.npz")
    bench.save_npz(out, probe_g=pick.astype(np.int64), n_probe=np.array(P), n_library=np.array(len(lib_rc)),
                   P__raw=np.concatenate([sim, aug]).astype(np.float32), **{f"Q__{k}": v for k, v in Q.items()})
    print(f"[prep] {len(qinfo)} queries, {2 * P} probes from {len(eligible):,} eligible entries "
          f"(excluded {len(bad)} query/family stems) -> {out}", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------
# score (GPU box): full query rows + per-entry probe statistics
# ------------------------------------------------------------------------------------------------
def _render_job(job):
    from xrdclip.eval import strong_baselines as SB
    from xrdclip.eval import wcc_torch as WT
    cid, path, lo, hi, g0 = job
    raw = bench.render(bench.load_records(path, lo, hi, bench.RENDER_FIELDS))
    return cid, g0, raw, WT.standardize_rows(raw), np.stack([SB._to_lnq(y) for y in raw]).astype(np.float32)


def cmd_score(args) -> int:
    import torch
    from multiprocessing import Pool
    from xrdclip.eval import wcc_torch as WT
    WT.full_precision()
    bench.unthrottle()
    dev = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "cpu":
        torch.set_num_threads(args.threads)
    # fork the render workers before this process touches CUDA
    pool = Pool(args.workers, initializer=bench._init_worker, initargs=({},))
    hub = np.load(os.path.join(args.work, "hub_ctx.npz"))
    shards = bench.shard_table(args.limit)
    lib_rc = library_refcodes(shards)
    N = len(lib_rc)
    assert N == int(hub["n_library"]), f"library mismatch: {N:,} here vs {int(hub['n_library']):,} in hub_ctx"
    stem = torch.from_numpy(stems_of(lib_rc)).to(dev)
    P = int(hub["n_probe"])
    pstem = stem[torch.from_numpy(hub["probe_g"]).to(dev)]
    Q = {k[3:]: hub[k] for k in hub.files if k.startswith("Q__")}
    nq = Q["raw_bkg"].shape[0]
    probes = hub["P__raw"]
    ks = [k for k in KS if k <= P]

    # row blocks: every method scores [query variants..., probes] in one matrix
    rows = {"pearson": torch.from_numpy(np.concatenate([Q["std_raw"], Q["std_bkg"], WT.standardize_rows(probes)])).to(dev)}
    layout = {"pearson": {"pearson_raw": slice(0, nq), "pearson_bkg": slice(nq, 2 * nq)}}
    scorers = {}
    for m, spec in bench.SPECS.items():
        scorers[m] = WT.TorchWCC(**spec, device=dev)
        vs = bench.VARIANTS[m]
        rows[m] = scorers[m].prep_rows(np.concatenate([Q[f"raw_{v}"] for v in vs] + [probes]))
        layout[m] = {f"{m}_{v}": slice(i * nq, (i + 1) * nq) for i, v in enumerate(vs)}
    for m in METHODS:
        q_end = max(s.stop for s in layout[m].values())
        layout[m]["probes"] = slice(q_end, q_end + 2 * P)

    out = os.path.join(args.work, "full")
    os.makedirs(out, exist_ok=True)
    S = {k: np.lib.format.open_memmap(os.path.join(out, f"S_{k}.npy"), mode="w+", dtype=np.float32, shape=(nq, N))
         for k in bench.CLASSICAL_KEYS}
    stats = {(m, ps): new_stats(N, P, len(ks), dev) for m in METHODS for ps in PROBE_SETS}
    jobs = bench.chunk_jobs(shards, args.chunk)
    print(f"[score] device {dev}, library {N:,}, {nq} queries, 2 x {P} probes, {len(jobs)} chunks, "
          f"{args.workers} render workers", flush=True)

    t0 = time.time()
    t_gpu = 0.0
    with pool:
        it = iter(jobs)
        pending = deque(pool.apply_async(_render_job, (j,)) for j in islice(it, args.workers + 2))
        done = 0
        while pending:
            cid, g0, raw, rstd, rlnq = pending.popleft().get()
            nxt = next(it, None)
            if nxt is not None:                              # bounded queue: rendered chunks are ~110 MB each
                pending.append(pool.apply_async(_render_job, (nxt,)))
            tg = time.time()
            C = raw.shape[0]
            cstem = stem[g0:g0 + C]
            blocks = {"pearson": WT.pearson(rows["pearson"], torch.from_numpy(rstd).to(dev))}
            for m, sc in scorers.items():
                blocks[m] = sc.scores(rows[m], sc.prep_refs(rlnq if sc.lnq else raw, already_lnq=sc.lnq))
            for m, s in blocks.items():
                for key, sl in layout[m].items():
                    if key != "probes":
                        S[key][:, g0:g0 + C] = s[sl].cpu().numpy()
                sp = s[layout[m]["probes"]]
                for i, ps in enumerate(PROBE_SETS):
                    mask = pstem[i * P:(i + 1) * P, None] == cstem[None, :]
                    update_stats(stats[(m, ps)], sp[i * P:(i + 1) * P], mask, g0, ks)
            del blocks
            if dev == "cuda":
                torch.cuda.synchronize()
            t_gpu += time.time() - tg
            done += 1
            el = time.time() - t0
            print(f"[score] chunk {cid} ({done}/{len(jobs)}) scoring {time.time() - tg:.1f}s, elapsed {el / 60:.1f} min, "
                  f"eta {el / done * (len(jobs) - done) / 60:.1f} min", flush=True)

    for s in S.values():
        s.flush()
    arrays = {"ks": np.array(ks), "probe_g": hub["probe_g"], "n_probe": np.array(P)}
    for (m, ps), st in stats.items():
        for f in ("mean", "std", "topk"):
            arrays[f"{m}__{ps}__{f}"] = st[f]
        arrays[f"{m}__{ps}__nn_s"] = st["nn_s"].cpu().numpy()
        arrays[f"{m}__{ps}__nn_i"] = st["nn_i"].cpu().numpy().astype(np.int32)
    bench.save_npz(os.path.join(out, "hub_stats.npz"), **arrays)
    with open(os.path.join(out, "DONE"), "w") as fh:
        fh.write(f"{N} entries, {time.time() - t0:.0f}s wall, {t_gpu:.0f}s scoring, device {dev}\n")
    print(f"[score] SCORE_DONE {time.time() - t0:.0f}s wall ({t_gpu:.0f}s scoring)", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------
# eval (local)
# ------------------------------------------------------------------------------------------------
def model_side(args, shards, Q, probes, stem, ks):
    """Model scores of the queries + probe statistics of every library entry, on the CPU."""
    import torch
    from xrdclip.data import augment as A
    from xrdclip.models.clip import XRDClip
    torch.set_num_threads(args.threads)
    model = XRDClip(embed_dim=args.embed_dim)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.eval()
    emb_dir = args.emb_dir or args.work
    lib_emb = np.concatenate([np.load(os.path.join(emb_dir, f"emb_{os.path.basename(os.path.dirname(p))}_"
                                                             f"{os.path.basename(p)[:-4]}_{n}.npy")) for p, n, _ in shards])
    lib_emb = torch.from_numpy(lib_emb.astype(np.float32))

    def enc(x):
        with torch.no_grad():
            return torch.cat([model.encode_pattern(torch.from_numpy(x[i:i + 256]).unsqueeze(1)) for i in range(0, len(x), 256)])

    S = {f"model_{v}": (enc(Q[f"std_{v}"]) @ lib_emb.T).numpy() for v in ("raw", "bkg")}
    p_emb = enc(np.stack([A.standardize(y) for y in probes]))
    P = len(probes) // 2
    N = lib_emb.shape[0]
    pstem = stem[torch.from_numpy(args._probe_g)]
    stats = {}
    with torch.no_grad():
        for i, ps in enumerate(PROBE_SETS):
            st = new_stats(N, P, len(ks), "cpu")
            for g0 in range(0, N, 8000):
                C = min(8000, N - g0)
                sp = p_emb[i * P:(i + 1) * P] @ lib_emb[g0:g0 + C].T
                update_stats(st, sp, pstem[i * P:(i + 1) * P, None] == stem[None, g0:g0 + C], g0, ks)
            stats[ps] = {"mean": st["mean"], "std": st["std"], "topk": st["topk"],
                         "nn_s": st["nn_s"].numpy(), "nn_i": st["nn_i"].numpy()}
    return S, stats


def corrected_row(name: str, s: np.ndarray, st: dict, ks) -> np.ndarray:
    """One query's library scores under a correction. `st` = {probe set: stats}."""
    from scipy.special import log_ndtr
    if name == "none":
        return s
    kind, ps = name.split("_")[:2]
    t = st[ps]
    if kind == "csls":
        k = int(name.split("_k")[-1])
        return s - 0.5 * t["topk"][list(ks).index(k)]
    if kind == "znorm":
        return (s - t["mean"]) / np.maximum(t["std"], 1e-6)
    if kind == "mp":
        s64 = s.astype(np.float64)
        zq = (s64 - s64.mean()) / max(float(s64.std()), 1e-9)
        zr = (s64 - t["mean"]) / np.maximum(t["std"], 1e-6)
        return log_ndtr(zq) + log_ndtr(zr)
    raise ValueError(name)


def cmd_eval(args) -> int:
    from scipy.stats import skew
    import torch
    t_start = time.time()
    shards = bench.shard_table(args.limit)
    lib_rc = library_refcodes(shards)
    N = len(lib_rc)
    gidx = {rc: i for i, rc in enumerate(lib_rc)}
    qinfo, excluded, Q = bench.build_queries(args.truth, args.patterns, gidx)
    nq = len(qinfo)
    hub = np.load(os.path.join(args.work, "hub_ctx.npz"))
    for k, v in Q.items():
        assert np.array_equal(v, hub[f"Q__{k}"]), f"query {k} differs from hub_ctx"
    full = os.path.join(args.work, "full")
    hs = np.load(os.path.join(full, "hub_stats.npz"))
    ks = [int(k) for k in hs["ks"]]
    args._probe_g = hub["probe_g"]
    stem = torch.from_numpy(stems_of(lib_rc))
    t = time.time()
    S_model, st_model = model_side(args, shards, Q, hub["P__raw"], stem, ks)
    print(f"[eval] model scores + probe stats {time.time() - t:.0f}s", flush=True)

    special_rc = sorted(special_refcodes(qinfo))
    special = np.array(sorted(gidx[rc] for rc in special_rc), dtype=np.int64)
    sets = {td: [(np.array([gidx[rc] for rc in q["sets"][td][0]], dtype=np.int64),
                  np.array([gidx[rc] for rc in q["sets"][td][1]], dtype=np.int64)) for q in qinfo] for td in ("std", "sc")}
    other = [np.array([gidx[rc] for rc in q["other"]], dtype=np.int64) for q in qinfo]
    nonspecial = np.setdiff1d(np.arange(N), special)
    subsets = {size: [np.random.default_rng(rep).choice(nonspecial, size, replace=False) for rep in range(5)]
               for size in (2000, 20000, 100000) if size < len(nonspecial)}

    # reference ranks (19) and pre-pass scores, for the uncorrected check
    ref_ranks = np.load(args.ref_ranks) if os.path.exists(args.ref_ranks) else None
    pre_path = os.path.join(args.work, "prepass.npz")
    pre = np.load(pre_path) if os.path.exists(pre_path) else None

    keys = bench.CLASSICAL_KEYS + bench.MODEL_KEYS
    ranks = {}
    wf = {}
    top1 = {}
    curve = {}
    rank_check = {}
    for key in keys:
        method = key.rsplit("_", 1)[0]
        if key in S_model:
            Sk, st = S_model[key], st_model
        else:
            Sk = np.load(os.path.join(full, f"S_{key}.npy"), mmap_mode="r")
            st = {ps: {f: hs[f"{method}__{ps}__{f}"] for f in ("mean", "std", "topk")} for ps in PROBE_SETS}
        if pre is not None and key in pre.files:
            d = float(np.abs(np.asarray(Sk[:, special]) - pre[key]).max())
            rank_check.setdefault(key, {})["max_abs_diff_vs_prepass"] = d
        for corr in CORRECTIONS:
            r = {td: np.full(nq, -1, np.int64) for td in ("std", "sc")}
            w = np.full(nq, -1, np.int64)
            t1 = np.full(nq, -1, np.int64)
            want_curve = corr in ("none", PRIMARY) and key.endswith("bkg")
            cu = {size: np.zeros((5, nq), np.int64) for size in subsets} if want_curve else None
            for qi in range(nq):
                row = corrected_row(corr, np.asarray(Sk[qi], dtype=np.float32), st, ks)
                t1[qi] = int(np.argmax(row))
                for td in ("std", "sc"):
                    tr, ig = sets[td][qi]
                    if not len(tr):
                        continue
                    thr = row[tr].max()
                    above = row > thr
                    excl = np.union1d(tr, ig)
                    r[td][qi] = int(above.sum() - above[excl].sum())
                    if td == "std":
                        if len(other[qi]):
                            w[qi] = int(thr >= row[other[qi]].max())
                        if cu is not None:
                            sp_above = int(above[special].sum() - above[excl].sum())
                            for size, subs in subsets.items():
                                for rep, sub in enumerate(subs):
                                    cu[size][rep, qi] = sp_above + int(above[sub].sum())
            for td in ("std", "sc"):
                ranks[(corr, td, key)] = r[td]
            wf[(corr, key)] = w
            top1[(corr, key)] = t1
            if cu is not None:
                ok = r["std"] >= 0
                c = {str(size): [float(np.mean([(cu[size][rep][ok] < 1).mean() for rep in range(5)])),
                                 float(np.std([(cu[size][rep][ok] < 1).mean() for rep in range(5)]))] for size in subsets}
                c["full"] = [float((r["std"][ok] < 1).mean()), 0.0]
                curve[f"{corr}|{key}"] = c
        if ref_ranks is not None:
            for td in ("std", "sc"):
                rk = f"{td}|{key}"
                if rk in ref_ranks.files:
                    a, b = ranks[("none", td, key)], ref_ranks[rk]
                    rank_check.setdefault(key, {})[f"{td}_rank_mismatches"] = int((a != b).sum())
                    rank_check[key][f"{td}_top1_changes"] = int(((a == 0) != (b == 0)).sum())
        print(f"[eval] {key}: {len(CORRECTIONS)} corrections ranked ({time.time() - t_start:.0f}s); "
              f"check {rank_check.get(key)}", flush=True)

    # ---- summaries ---------------------------------------------------------------------------------
    boot = bench.compound_bootstrap([q["stem"] for q in qinfo])
    split_of = {"all": np.ones(nq, bool),
                "powder_study": np.array([q["own_powder"] for q in qinfo]),
                "independent": np.array([not q["own_powder"] for q in qinfo])}
    res = {"meta": {"n_library": N, "n_queries": nq, "n_probe": int(hub["n_probe"]), "ks": ks, "primary": PRIMARY,
                    "corrections": CORRECTIONS, "excluded": excluded,
                    "score_log": open(os.path.join(full, "DONE")).read().strip()},
           "rank_check": rank_check, "summary": {}, "within_family": {}, "library_curve": curve, "hubness": {}}
    for (corr, td, key), r in ranks.items():
        for sp, sel in split_of.items():
            if td == "sc" and sp != "all":
                continue
            res["summary"][f"{corr}|{td}|{sp}|{key}"] = bench.summarise(r, sel, boot)
    for (corr, key), w in wf.items():
        ok = w >= 0
        res["within_family"][f"{corr}|{key}"] = {"n": int(ok.sum()), "top1": float(w[ok].mean()) if ok.any() else None}

    # ---- compound-weighted view: a few compounds contribute many patterns (ZZZUWK 20 of 188), so the
    # per-pattern numbers mostly measure those compounds. Macro = mean over compounds of per-compound top-1.
    stems_q = np.array([q["stem"] for q in qinfo])
    res["patterns_per_compound"] = Counter(stems_q.tolist()).most_common(10)
    res["macro"], res["paired"] = {}, {}
    cboot = np.random.default_rng(0).integers(0, len(np.unique(stems_q)), (2000, len(np.unique(stems_q))))
    for (corr, td, key), r in ranks.items():
        res["macro"][f"{corr}|{td}|{key}"] = macro_top1(r, stems_q, cboot)
        if corr != "none":
            res["paired"][f"{corr}|{td}|{key}"] = paired_vs_none(ranks[("none", td, key)], r, stems_q, boot, cboot)

    # ---- hub diagnostics ---------------------------------------------------------------------------
    lens = {}

    def n_refl(g):
        if g not in lens:
            rec = records_at(shards, [g], ("two_theta", "intensity"))[0]
            inten = np.asarray(rec["intensity"], float)
            lens[g] = (int(len(rec["two_theta"])), float(inten.max() / max(inten.sum(), 1e-12)))
        return lens[g]

    for key in [k for k in keys if k.endswith("bkg")]:
        method = key.rsplit("_", 1)[0]
        d = {}
        for ps in PROBE_SETS:
            nn_i = st_model[ps]["nn_i"] if method == "model" else hs[f"{method}__{ps}__nn_i"]
            n10 = np.bincount(nn_i[:, :10].ravel().astype(np.int64), minlength=N)
            hubs = np.argsort(-n10)[:5]
            d[f"probe_{ps}"] = {"N10_skew": float(skew(n10)), "N10_max": int(n10.max()),
                                "frac_entries_never_in_a_top10": float((n10 == 0).mean()),
                                "top_hubs": [[lib_rc[g], int(n10[g]), *n_refl(int(g))] for g in hubs]}
        for corr in ("none", PRIMARY, "csls_sim_k10"):
            r, t1 = ranks[(corr, "std", key)], top1[(corr, key)]
            miss = (r > 0)
            wins = {}
            for qi in np.nonzero(miss)[0]:
                wins.setdefault(lib_rc[int(t1[qi])], []).append(qinfo[qi]["stem"])
            top = sorted(wins.items(), key=lambda kv: -len(kv[1]))[:5]
            # [winning entry, wins, distinct query compounds, top query compounds]
            d[f"query_top1_wrong|{corr}"] = {"n_miss": int(miss.sum()), "most_common": [
                [w, len(q), len(set(q)), Counter(q).most_common(3)] for w, q in top]}
        topk10 = {ps: (st_model[ps]["topk"] if method == "model" else hs[f"{method}__{ps}__topk"])[ks.index(10)]
                  for ps in PROBE_SETS}
        cnt = Counter(lib_rc[g] for g in top1[("none", key)][ranks[("none", "std", key)] > 0])
        d["h10_percentile_of_query_hubs"] = {rc: {ps: float((topk10[ps] < topk10[ps][gidx[rc]]).mean())
                                                  for ps in PROBE_SETS} for rc, _ in cnt.most_common(3)}
        res["hubness"][key] = d

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    np.savez_compressed(os.path.join(args.work, "hub_ranks.npz"),
                        **{f"{corr}|{td}|{key}": r for (corr, td, key), r in ranks.items()})
    write_table(res, args.table)
    print(f"[eval] wrote {args.out} + {args.table} ({time.time() - t_start:.0f}s)", flush=True)
    return 0


def _hits_by_compound(r: np.ndarray, stems: np.ndarray, uniq: np.ndarray) -> np.ndarray:
    """Per-compound top-1 rate (NaN for compounds with no truth in the library)."""
    ok = r >= 0
    return np.array([((r == 0) & ok)[stems == s].sum() / (ok & (stems == s)).sum() if (ok & (stems == s)).any()
                     else np.nan for s in uniq])


def macro_top1(r: np.ndarray, stems: np.ndarray, cboot: np.ndarray) -> dict:
    p = _hits_by_compound(r, stems, np.unique(stems))
    bs = np.nanmean(p[cboot], axis=1)
    return {"n_compounds": int(np.isfinite(p).sum()), "top1": float(np.nanmean(p)),
            "top1_ci": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]}


def paired_vs_none(base: np.ndarray, r: np.ndarray, stems: np.ndarray, boot, cboot: np.ndarray) -> dict:
    """Top-1 change vs the uncorrected score: per pattern and per compound, paired bootstrap CIs (by compound)."""
    ok = base >= 0
    a1, c1 = (base == 0) & ok, (r == 0) & ok
    dp = [c1[b[ok[b]]].mean() - a1[b[ok[b]]].mean() for b in boot if ok[b].any()]
    uniq = np.unique(stems)
    pa, pc = _hits_by_compound(base, stems, uniq), _hits_by_compound(r, stems, uniq)
    dm = np.nanmean(pc[cboot], axis=1) - np.nanmean(pa[cboot], axis=1)
    return {"pattern_diff": float(c1[ok].mean() - a1[ok].mean()),
            "pattern_diff_ci": [float(np.percentile(dp, 2.5)), float(np.percentile(dp, 97.5))],
            "macro_diff": float(np.nanmean(pc) - np.nanmean(pa)),
            "macro_diff_ci": [float(np.nanpercentile(dm, 2.5)), float(np.nanpercentile(dm, 97.5))],
            "fixed": Counter(stems[c1 & ~a1].tolist()).most_common(), "broken": Counter(stems[a1 & ~c1].tolist()).most_common()}


def write_table(res: dict, path: str) -> None:
    S, W = res["summary"], res["within_family"]
    names = {"pearson_bkg": "Pearson, fixed grid", "cos_zs_bkg": "cosine + zero scan", "wcc_1_bkg": "WCC, l = 1 deg",
             "wcc_2_bkg": "WCC, l = 2 deg", "lnq_bkg": "ln q + zero scan", "model_bkg": "xrd-clip model"}

    def cell(k):
        s = S.get(k, {})
        return f"{s['top1']:.2f} [{s['top1_ci'][0]:.2f}, {s['top1_ci'][1]:.2f}]" if s.get("n") else "n/a"

    def t1(k):
        s = S.get(k, {})
        return f"{s['top1']:.2f}" if s.get("n") else "n/a"

    def wf(k):
        v = W.get(k, {}).get("top1")
        return f"{v:.2f}" if v is not None else "n/a"

    m = res["meta"]
    lines = [f"# Hubness correction: full-library real benchmark ({m['n_queries']} patterns vs {m['n_library']:,} entries)",
             "", f"Background-subtracted queries. Top-1 with 95 % compound-bootstrap CIs. Primary correction "
             f"(pre-declared): `{m['primary']}` = CSLS, k = 10, {m['n_probe']} augmented library probes.", "",
             "| Method | std, none | std, CSLS | single-crystal, none | single-crystal, CSLS | within-family, none | within-family, CSLS |",
             "|---|---|---|---|---|---|---|"]
    for k, nm in names.items():
        p = m["primary"]
        lines.append(f"| {nm} | {cell(f'none|std|all|{k}')} | {cell(f'{p}|std|all|{k}')} | {cell(f'none|sc|all|{k}')} | "
                     f"{cell(f'{p}|sc|all|{k}')} | {wf(f'none|{k}')} | {wf(f'{p}|{k}')} |")
    ppc = ", ".join(f"{c} {n}" for c, n in res["patterns_per_compound"][:6])
    lines += ["", "## Compound-weighted view (the per-pattern numbers above mostly measure a few compounds)", "",
              f"Patterns per compound, largest: {ppc}. Macro = mean over compounds of per-compound top-1. "
              "Paired differences vs uncorrected, 95 % compound-bootstrap CIs.", "",
              "| Method | truth | macro, none | macro, CSLS | macro diff | per-pattern diff | fixed (by compound) | broken |",
              "|---|---|---|---|---|---|---|---|"]
    p = m["primary"]
    for k, nm in names.items():
        for td in ("std", "sc"):
            a0, a1, pd = res["macro"][f"none|{td}|{k}"], res["macro"][f"{p}|{td}|{k}"], res["paired"][f"{p}|{td}|{k}"]
            fx = ", ".join(f"{c} {n}" for c, n in pd["fixed"][:3]) + (" ..." if len(pd["fixed"]) > 3 else "")
            br = ", ".join(f"{c} {n}" for c, n in pd["broken"][:3]) + (" ..." if len(pd["broken"]) > 3 else "")
            lines.append(f"| {nm} | {td} ({a0['n_compounds']} cpds) | {a0['top1']:.2f} [{a0['top1_ci'][0]:.2f}, {a0['top1_ci'][1]:.2f}] | "
                         f"{a1['top1']:.2f} [{a1['top1_ci'][0]:.2f}, {a1['top1_ci'][1]:.2f}] | "
                         f"{pd['macro_diff']:+.3f} [{pd['macro_diff_ci'][0]:+.3f}, {pd['macro_diff_ci'][1]:+.3f}] | "
                         f"{pd['pattern_diff']:+.3f} [{pd['pattern_diff_ci'][0]:+.3f}, {pd['pattern_diff_ci'][1]:+.3f}] | "
                         f"{sum(n for _, n in pd['fixed'])}: {fx} | {sum(n for _, n in pd['broken'])}: {br} |")
    lines += ["", "## Sensitivity (top-1, std truth, all patterns; not used to pick the primary)", "",
              "| Correction | " + " | ".join(names.values()) + " |", "|---|" + "---|" * len(names)]
    for c in m["corrections"]:
        lines.append(f"| {c} | " + " | ".join(t1(f'{c}|std|all|{k}') for k in names) + " |")
    lines += ["", "## Single-crystal split (top-1)", "", "| Correction | " + " | ".join(names.values()) + " |",
              "|---|" + "---|" * len(names)]
    for c in m["corrections"]:
        lines.append(f"| {c} | " + " | ".join(t1(f'{c}|sc|all|{k}') for k in names) + " |")
    lines += ["", "## Library-size curve (top-1, std truth; mean over 5 random decoy subsets)", "",
              "| Method | Correction | 2k | 20k | 100k | full |", "|---|---|---|---|---|---|"]
    for k, nm in names.items():
        for c in ("none", m["primary"]):
            cv = res["library_curve"].get(f"{c}|{k}", {})
            lines.append(f"| {nm} | {c} | " + " | ".join(f"{cv[s][0]:.2f}" if s in cv else "n/a"
                                                          for s in ("2000", "20000", "100000", "full")) + " |")
    lines += ["", "## Hubness (k-occurrence over the probes, N10 = number of probes with the entry in their top 10)", "",
              "| Method | probes | N10 skew | N10 max | top hubs (refcode, N10, reflections, strongest-peak share) |",
              "|---|---|---|---|---|"]
    for k, nm in names.items():
        for ps in PROBE_SETS:
            h = res["hubness"][k][f"probe_{ps}"]
            hubs = "; ".join(f"{a} {b} ({c}, {d:.2f})" for a, b, c, d in h["top_hubs"][:3])
            lines.append(f"| {nm} | {ps} | {h['N10_skew']:.1f} | {h['N10_max']} | {hubs} |")
    lines += ["", "Most frequent wrong top-1 entries (std truth), with the query compounds whose patterns they win:", ""]
    for k, nm in names.items():
        h = res["hubness"][k]
        parts = [f"{c}: " + ", ".join(f"{w} x{n} ({nc} cpd: " + ", ".join(f"{s} {j}" for s, j in top) + ")"
                                      for w, n, nc, top in h[f'query_top1_wrong|{c}']['most_common'][:3])
                 + f" (of {h[f'query_top1_wrong|{c}']['n_miss']} misses)" for c in ("none", m["primary"])]
        lines.append(f"- {nm}: " + "; ".join(parts))
    lines += ["", f"Uncorrected ranks vs scripts/19_...: {json.dumps(res['rank_check'])}", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("check", "prep", "score", "eval"))
    ap.add_argument("--work", default="cache/real_bench")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--truth", default=f"{DATA}/opxrd/opxrd_truth.json")
    ap.add_argument("--patterns", default=f"{DATA}/opxrd/opxrd_cnrs_organic.json")
    ap.add_argument("--n_probe", type=int, default=4000, help="probes per set (sim, aug)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--ckpt", default="runs/xrdclip_poly/best.pt")
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--emb_dir", default=None, help="library embeddings from 19 (default: --work)")
    ap.add_argument("--ref_ranks", default=None, help="19's ranks.npz (default: <work>/ranks.npz)")
    ap.add_argument("--out", default="results/real_bench_hub.json")
    ap.add_argument("--table", default="results/real_bench_hub_table.md")
    args = ap.parse_args()
    args.ref_ranks = args.ref_ranks or os.path.join(args.work, "ranks.npz")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    return {"check": cmd_check, "prep": cmd_prep, "score": cmd_score, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
