"""Revision rescoring of the real queries against the full library (tasks/todo.md Step 3: R1, R2).

R1  ln q v2: the zero-offset scan contains 0 (17 offsets, -0.32..+0.32 deg) and patterns are resampled onto the
    ln q axis by bin averaging (strong_baselines.lnq_offsets / _to_lnq).
R2  measured-aware library and comparison. One pre-declared package; each switch is also turned off alone:
      B     overall Debye-Waller factor on library intensities, I exp(-2 B s^2) with s = 1/(2d), B from `bvalue`
      lp    non-Cu queries are converted to Cu K-alpha1 Lorentz-polarisation, I x LP_Cu(theta_Cu) / LP_orig(theta_orig)
            (lab tubes P = 1 + cos^2 2theta; synchrotron P constant)
      ka2   lab-tube queries score max(K-alpha1 library, K-alpha1 + K-alpha2 library), I2/I1 = 0.5, because
            monochromatisation is not recorded; synchrotron queries use the K-alpha1 library only
      mask  reference norms are taken over the query's measured 2theta range (the unshifted range under the zero scan)
Background-subtracted queries only (the paper's primary variant). Configs are fixed in CONFIGS.

  PYTHONPATH=src python scripts/28_rescore.py bvalue
  PYTHONPATH=src python scripts/28_rescore.py score --tag r1
  PYTHONPATH=src python scripts/28_rescore.py eval --tag r1
  smoke: score/eval --tag v1 --limit 200 --work cache/real_bench_smoke   (must equal 19's smoke ranks.npz)
"""
from __future__ import annotations

import argparse
import gzip
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
hubm = import_module("21_hubness")
pn = import_module("22_paper_numbers")

KEYS = ("pearson_bkg", "cos_zs_bkg", "wcc_1_bkg", "wcc_2_bkg", "lnq_bkg")
# K-alpha1 / K-alpha2 wavelengths in angstrom (Hoelzer et al., Phys. Rev. A 56, 4554, 1997)
TUBES = {"cu": (1.5405929, 1.5444274), "co": (1.7889965, 1.7928496), "mo": (0.7093171, 0.7136066)}
KA2_RATIO = 0.5
CONFIGS = {
    "v1":        dict(lnq=1, B=False, lp=False, ka2=False, mask=False, keys=KEYS),
    "r1":        dict(lnq=2, B=False, lp=False, ka2=False, mask=False, keys=("lnq_bkg",)),
    "main":      dict(lnq=2, B=False, lp=False, ka2=False, mask=False, keys=KEYS),   # revised primary (R5 replica)
    "r2":        dict(lnq=2, B=True,  lp=True,  ka2=True,  mask=True,  keys=KEYS),
    "r2_noB":    dict(lnq=2, B=False, lp=True,  ka2=True,  mask=True,  keys=KEYS),
    "r2_nolp":   dict(lnq=2, B=True,  lp=False, ka2=True,  mask=True,  keys=KEYS),
    "r2_noka2":  dict(lnq=2, B=True,  lp=True,  ka2=False, mask=True,  keys=KEYS),
    "r2_nomask": dict(lnq=2, B=True,  lp=True,  ka2=True,  mask=False, keys=KEYS),
    # 10-14 referee check (Frank's X1): measured patterns put on the Cu grid by bin averaging, not interpolation
    "binavg":    dict(lnq=2, B=False, lp=False, ka2=False, mask=False, grid="binavg", keys=KEYS),
}
BVALUE_JSON = "results/r2_bvalue.json"


def tube(wl) -> str:
    """'cu', 'co', 'mo' for lab K-alpha lines (the 1.5418 weighted Cu mean included; unlabelled = Cu), else 'sync'."""
    w = float(wl) if wl not in (None, "", "None") else TUBES["cu"][0]
    for name, (k1, _) in TUBES.items():
        if abs(w - k1) < 0.003:
            return name
    return "sync"


def lp_factor(tt_cu: np.ndarray, wl: float, kind: str) -> np.ndarray:
    """LP_Cu(theta_Cu) / LP_orig(theta_orig) for a reflection seen at tt_cu on the Cu K-alpha1 scale."""
    from xrdclip.data import augment as A
    th = np.radians(np.asarray(tt_cu, float) / 2)
    th0 = np.arcsin(np.clip(np.sin(th) * wl / A.WAVELENGTH, 1e-6, 1 - 1e-9))
    lp_cu = (1 + np.cos(2 * th) ** 2) / (np.sin(th) ** 2 * np.cos(th))
    pol0 = (1 + np.cos(2 * th0) ** 2) if kind in ("cu", "co", "mo") else 2.0
    return lp_cu / (pol0 / (np.sin(th0) ** 2 * np.cos(th0)))


def b_value(cfg: dict) -> float:
    if not cfg["B"]:
        return 0.0
    return float(json.load(open(BVALUE_JSON))["B"])


# ------------------------------------------------------------------------------------------------
# bvalue (R2a): overall B from the library's own CIFs, query families excluded
# ------------------------------------------------------------------------------------------------
def cmd_bvalue(args) -> int:
    import gemmi
    from xrdclip.eval.real_patterns import temp_k
    lib_rc = hubm.library_refcodes(bench.shard_table(None))
    gidx = {rc: i for i, rc in enumerate(lib_rc)}
    qinfo, _, _ = bench.build_queries(args.truth, args.patterns, gidx)
    bad = {rc[:6] for rc in hubm.special_refcodes(qinfo)} | {q["stem"][:6] for q in qinfo}
    rng = np.random.default_rng(args.seed)
    per_entry, seen, skipped = [], 0, Counter()
    shards = sorted(p for p in os.listdir("cache/csd") if p.endswith(".jsonl.gz"))
    # reservoir-free: visit shards in a random order, entries in a random order within each, until enough
    for sh in rng.permutation(shards):
        with gzip.open(os.path.join("cache/csd", sh), "rt") as fh:
            rows = fh.readlines()
        for i in rng.permutation(len(rows)):
            r = json.loads(rows[i])
            seen += 1
            if r["refcode"][:6] in bad:
                skipped["query family"] += 1
                continue
            t = temp_k(r["temperature"])                     # CSD text, e.g. "at 293 K"
            if t is None:
                skipped["no temperature in K"] += 1
                continue
            if not 283.0 <= t <= 303.0:
                skipped["temperature outside 283-303 K"] += 1
                continue
            try:
                blk = gemmi.cif.read_string(r["cif"]).sole_block()
                types = list(blk.find_loop("_atom_site_type_symbol"))
                u = list(blk.find_loop("_atom_site_U_iso_or_equiv"))
            except Exception:
                skipped["cif parse"] += 1
                continue
            vals = [gemmi.cif.as_number(x) for x, s in zip(u, types) if s.upper() not in ("H", "D")]
            vals = [v for v in vals if np.isfinite(v) and 0 < v < 1]
            if len(u) != len(types) or not vals:
                skipped["no U_eq for non-H atoms"] += 1
                continue
            per_entry.append(float(np.mean(vals)))
            if len(per_entry) >= args.n:
                break
        if len(per_entry) >= args.n:
            break
    u = np.array(per_entry)
    med = float(np.median(u))
    out = {"B": round(8 * np.pi ** 2 * med, 2), "median_mean_Ueq": med, "Ueq_iqr": [float(np.percentile(u, 25)),
           float(np.percentile(u, 75))], "n_entries": int(len(u)), "n_seen": seen, "skipped": dict(skipped),
           "rule": "B = 8 pi^2 x median over entries of the mean non-H U_iso_or_equiv; entries measured at "
                   "283-303 K, query families excluded, random order (seed %d)" % args.seed}
    os.makedirs("results", exist_ok=True)
    with open(BVALUE_JSON, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[bvalue] {out}")
    return 0


# ------------------------------------------------------------------------------------------------
# queries
# ------------------------------------------------------------------------------------------------
def build_queries(cfg: dict, gidx: dict, args):
    """(qinfo, excluded, raw bkg queries under this config, tube kind per query, 2theta coverage masks)."""
    from xrdclip.data import augment as A
    from xrdclip.eval.real_patterns import preprocess, to_cu_grid
    qinfo, excluded, Q = bench.build_queries(args.truth, args.patterns, gidx)
    pats = {o["id"]: o for o in json.load(open(args.patterns))["organic"]}
    raw, kinds, masks = [], [], []
    for q in qinfo:
        o = pats[q["id"]]
        wl = float(q["wavelength"] or A.WAVELENGTH)
        kind = tube(q["wavelength"])
        y, m = to_cu_grid(o["two_theta"], o["intensity"], wl, method=cfg.get("grid", "interp"))
        if cfg["lp"] and kind != "cu":
            y = np.where(m, y * lp_factor(A.GRID, wl, kind), 0).astype(np.float32)
        raw.append(preprocess(y, m, bkg=True))
        kinds.append(kind)
        masks.append(m)
    raw = np.stack(raw).astype(np.float32)
    if not cfg["lp"] and cfg.get("grid", "interp") == "interp":
        assert np.array_equal(raw, Q["raw_bkg"]), "query build drifted from 19"
    return qinfo, excluded, raw, np.array(kinds), np.stack(masks)


def lnq_mask(m: np.ndarray) -> np.ndarray:
    from xrdclip.data import augment as A
    from xrdclip.eval import strong_baselines as SB
    lo, hi = A.GRID[m][0], A.GRID[m][-1]
    return (SB._TT_OF_LNQ >= lo) & (SB._TT_OF_LNQ <= hi)


# ------------------------------------------------------------------------------------------------
# library rendering (worker processes)
# ------------------------------------------------------------------------------------------------
def render(recs: "list[dict]", B: float, rho: "float | None") -> np.ndarray:
    """Clean library patterns, optionally with an overall B and a K-alpha2 copy at sin(theta) x rho."""
    from xrdclip.data import augment as A
    if not B and rho is None:
        return bench.render(recs)
    ys = []
    for r in recs:
        tt = np.asarray(r["two_theta"], np.float64)
        inten = np.asarray(r["intensity"], np.float64)
        d = np.asarray(r["d_hkl"], np.float64)
        hkl = np.asarray(r["hkl"])
        if B:
            inten = inten * np.exp(-B / (2 * d ** 2))
        if rho is not None:
            s2 = np.sin(np.radians(tt / 2)) * rho
            ok = s2 < 1
            tt = np.r_[tt, np.degrees(2 * np.arcsin(s2[ok]))]
            inten = np.r_[inten, KA2_RATIO * inten[ok]]
            d = np.r_[d, d[ok]]
            hkl = np.r_[hkl, hkl[ok]]
        ys.append(A.render_sticks({"two_theta": tt, "intensity": inten, "hkl": hkl, "d_hkl": d,
                                   "recip": r["recip"]}, A.CLEAN))
    return np.stack(ys).astype(np.float32)


def _render_job(job):
    from xrdclip.eval import strong_baselines as SB
    cid, path, lo, hi, g0, B, rhos, lnq_v, need_lnq = job
    recs = bench.load_records(path, lo, hi, bench.RENDER_FIELDS)
    libs = {"k1": render(recs, B, None)}
    for name, rho in rhos.items():
        libs[name] = render(recs, B, rho)
    lnq = ({k: np.stack([SB._to_lnq(y, 0.0, lnq_v) for y in v]).astype(np.float32) for k, v in libs.items()}
           if need_lnq else {})
    return cid, g0, libs, lnq


# ------------------------------------------------------------------------------------------------
# scoring (main process, torch)
# ------------------------------------------------------------------------------------------------
def pearson_rows(raw: np.ndarray, masks: "np.ndarray | None"):
    """Query side of fixed-grid Pearson: rows standardized over the full grid, or over each query's mask."""
    import torch
    from xrdclip.eval import wcc_torch as WT
    if masks is None:
        return torch.from_numpy(WT.standardize_rows(raw)), None
    q = np.zeros_like(raw)
    for i, (y, m) in enumerate(zip(raw, masks)):
        v = y[m]
        sd = v.std()
        q[i, m] = (v - v.mean()) / sd if sd > 1e-8 else v - v.mean()
    return torch.from_numpy(q), torch.from_numpy(masks.astype(np.float32))


def pearson_scores(qt, M, r: np.ndarray):
    import torch
    from xrdclip.eval import wcc_torch as WT
    if M is None:
        return WT.pearson(qt, torch.from_numpy(WT.standardize_rows(r)).to(qt.device))
    rt = torch.from_numpy(r).to(qt.device)
    num = qt @ rt.T
    s1, s2 = M @ rt.T, M @ (rt * rt).T
    var = torch.clamp(s2 - s1 * s1 / M.sum(1, keepdim=True), min=1e-12)
    return num / torch.sqrt((qt * qt).sum(1, keepdim=True) * var)


def wcc_scores(sc, rows, r: np.ndarray, M):
    """TorchWCC scores; with a mask matrix M (n_rows x n) the reference norm is taken over each query's range."""
    import torch
    from xrdclip.eval import wcc_torch as WT
    if M is None:
        return sc.scores(rows, sc.prep_refs(r, already_lnq=sc.lnq))
    t = torch.from_numpy((r - r.min(axis=1, keepdims=True)).astype(np.float32)).to(sc.device)
    S_, n = sc.s_pts, t.shape[1]
    rw = WT._smooth(t, sc.w, S_)
    ar = torch.clamp(M @ (t * rw[:, S_:S_ + n]).T, min=1e-12)                 # (rows, refs)
    best = None
    for q, aq in rows:
        num = None
        for d in range(2 * S_ + 1):
            c = q @ rw[:, d:d + n].T
            num = c if num is None else torch.maximum(num, c, out=num)
        s = num / torch.sqrt(torch.clamp(aq[:, None] * ar, min=1e-12))
        best = s if best is None else torch.maximum(best, s, out=best)
    return best


def sub_rows(rows, idx):
    return [(q[idx], a[idx]) for q, a in rows]


def cmd_score(args) -> int:
    import torch
    from multiprocessing import Pool
    from xrdclip.eval import wcc_torch as WT
    cfg = CONFIGS[args.tag]
    WT.full_precision()
    bench.unthrottle()
    dev = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(args.threads)
    pool = Pool(args.workers, initializer=bench._init_worker, initargs=({},))
    shards = bench.shard_table(args.limit)
    lib_rc = hubm.library_refcodes(shards)
    N = len(lib_rc)
    gidx = {rc: i for i, rc in enumerate(lib_rc)}
    qinfo, _, raw, kinds, masks = build_queries(cfg, gidx, args)
    nq = len(qinfo)
    B = b_value(cfg)
    groups = {k: np.nonzero(kinds == k)[0] for k in TUBES if cfg["ka2"] and (kinds == k).any()}
    rhos = {k: TUBES[k][1] / TUBES[k][0] for k in groups}
    use_mask = cfg["mask"]
    print(f"[score] tag {args.tag} {cfg}; B = {B}; K-alpha2 groups {({k: len(v) for k, v in groups.items()})}; "
          f"partial-coverage queries {int((~masks.all(1)).sum())}; library {N:,}", flush=True)

    # query side, per method
    keys = list(cfg["keys"])
    scorers, rows, Mx = {}, {}, {}
    for key in keys:
        m = key.rsplit("_", 1)[0]
        if m == "pearson":
            qt, M = pearson_rows(raw, masks if use_mask else None)
            rows[key], Mx[key] = qt.to(dev), (None if M is None else M.to(dev))
            continue
        spec = dict(bench.SPECS[m])
        if m == "lnq":
            spec["version"] = cfg["lnq"]
        scorers[key] = WT.TorchWCC(**spec, device=dev)
        rows[key] = scorers[key].prep_rows(raw)
        if use_mask:
            mm = np.stack([lnq_mask(x) for x in masks]) if m == "lnq" else masks
            Mx[key] = torch.from_numpy(mm.astype(np.float32)).to(dev)
        else:
            Mx[key] = None
    need_lnq = "lnq_bkg" in keys

    out = os.path.join(args.work, f"rescore_{args.tag}")
    os.makedirs(out, exist_ok=True)
    S = {}
    for k in keys:
        p = os.path.join(out, f"S_{k}.npy")
        S[k] = np.lib.format.open_memmap(p, mode="r+" if os.path.exists(p) else "w+", dtype=np.float32, shape=(nq, N))
    done_path = os.path.join(out, "done_chunks.txt")
    done = set(int(x) for x in open(done_path).read().split()) if os.path.exists(done_path) else set()
    jobs = [j + (B, rhos, cfg["lnq"], need_lnq) for j in bench.chunk_jobs(shards, args.chunk) if j[0] not in done]
    print(f"[score] {len(jobs)} chunks to run ({len(done)} done), {args.workers} render workers, "
          f"{args.threads} torch threads", flush=True)

    t0 = time.time()
    with pool:
        it = iter(jobs)
        pending = deque(pool.apply_async(_render_job, (j,)) for j in islice(it, args.workers + 1))
        n_done = 0
        while pending:
            cid, g0, libs, lnq = pending.popleft().get()
            nxt = next(it, None)
            if nxt is not None:
                pending.append(pool.apply_async(_render_job, (nxt,)))
            tc = time.time()
            C = libs["k1"].shape[0]
            for key in keys:
                m = key.rsplit("_", 1)[0]
                lib = lnq if m == "lnq" else libs
                if m == "pearson":
                    s = pearson_scores(rows[key], Mx[key], lib["k1"])
                    for g, idx in groups.items():
                        ti = torch.from_numpy(idx).to(dev)
                        s2 = pearson_scores(rows[key][ti], None if Mx[key] is None else Mx[key][ti], lib[g])
                        s[ti] = torch.maximum(s[ti], s2)
                else:
                    sc = scorers[key]
                    s = wcc_scores(sc, rows[key], lib["k1"], Mx[key])
                    for g, idx in groups.items():
                        ti = torch.from_numpy(idx).to(dev)
                        s2 = wcc_scores(sc, sub_rows(rows[key], ti), lib[g], None if Mx[key] is None else Mx[key][ti])
                        s[ti] = torch.maximum(s[ti], s2)
                S[key][:, g0:g0 + C] = s.cpu().numpy()
            for a in S.values():
                a.flush()
            with open(done_path, "a") as fh:
                fh.write(f"{cid}\n")
            n_done += 1
            el = time.time() - t0
            print(f"[score] chunk {cid} ({n_done}/{len(jobs)}) {time.time() - tc:.1f}s, elapsed {el / 60:.1f} min, "
                  f"eta {el / n_done * (len(jobs) - n_done) / 60:.1f} min", flush=True)
    with open(os.path.join(out, "DONE"), "w") as fh:
        fh.write(f"{args.tag} {N} entries {nq} queries B={B} {time.time() - t0:.0f}s\n")
    print(f"[score] SCORE_DONE {args.tag} {time.time() - t0:.0f}s", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------
# eval
# ------------------------------------------------------------------------------------------------
def ranks_from_scores(Sk, sets, nq: int) -> dict:
    """0-based rank = wrong entries strictly above the best correct entry (truth + ignore sets excluded)."""
    r = {td: np.full(nq, -1, np.int64) for td in ("std", "sc")}
    for qi in range(nq):
        row = np.asarray(Sk[qi], dtype=np.float32)
        for td in ("std", "sc"):
            tr, ig = sets[td][qi]
            if not len(tr):
                continue
            above = row > row[tr].max()
            r[td][qi] = int(above.sum() - above[np.union1d(tr, ig)].sum())
    return r


def model_scores(args, shards, raw: np.ndarray) -> np.ndarray:
    import torch
    from xrdclip.data import augment as A
    from xrdclip.models.clip import XRDClip
    torch.set_num_threads(args.threads)
    model = XRDClip(embed_dim=args.embed_dim)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.eval()
    emb = np.concatenate([np.load(os.path.join(args.work, f"emb_{os.path.basename(os.path.dirname(p))}_"
                                                          f"{os.path.basename(p)[:-4]}_{n}.npy")) for p, n, _ in shards])
    with torch.no_grad():
        q = model.encode_pattern(torch.from_numpy(np.stack([A.standardize(y) for y in raw])).unsqueeze(1))
    return (q @ torch.from_numpy(emb.astype(np.float32)).T).numpy()


def hybrid_rank(s_model: np.ndarray, s_cls: np.ndarray, tr: np.ndarray, ig: np.ndarray, K: int, N: int) -> int:
    """Model top-K shortlist re-ranked by a classical score, as in 19 (truth dropped by the shortlist -> N)."""
    short = np.argsort(-s_model, kind="stable")[:K]
    is_t, is_i = np.isin(short, tr), np.isin(short, ig)
    if not is_t.any():
        return N
    sc = s_cls[short]
    return int((sc[~is_t & ~is_i] > sc[is_t].max()).sum())


def base_ranks(args, base: str) -> dict:
    p = os.path.join(args.work, f"rescore_{base}", "ranks.npz")
    if os.path.exists(p):
        z = np.load(p)
        return {k: z[k] for k in z.files}
    z = np.load(os.path.join(args.work, "hub_ranks.npz"))       # v1 = the uncorrected hub-pass ranks
    return {k[len("none|"):]: z[k] for k in z.files if k.startswith("none|")}


def cmd_eval(args) -> int:
    cfg = CONFIGS[args.tag]
    shards = bench.shard_table(args.limit)
    lib_rc = hubm.library_refcodes(shards)
    N = len(lib_rc)
    gidx = {rc: i for i, rc in enumerate(lib_rc)}
    qinfo, excluded, raw, kinds, masks = build_queries(cfg, gidx, args)
    nq = len(qinfo)
    out = os.path.join(args.work, f"rescore_{args.tag}")
    sets = {td: [(np.array([gidx[rc] for rc in q["sets"][td][0]], dtype=np.int64),
                  np.array([gidx[rc] for rc in q["sets"][td][1]], dtype=np.int64)) for q in qinfo] for td in ("std", "sc")}
    stems = np.array([q["stem"] for q in qinfo])
    ranks = {}
    S_cls = {}
    for key in cfg["keys"]:
        S_cls[key] = np.load(os.path.join(out, f"S_{key}.npy"), mmap_mode="r")
        for td, r in ranks_from_scores(S_cls[key], sets, nq).items():
            ranks[f"{td}|{key}"] = r
    keys_h = [] if args.no_model else cfg["keys"]
    if not args.no_model:
        s_model = model_scores(args, shards, raw)
        for td, r in ranks_from_scores(s_model, sets, nq).items():
            ranks[f"{td}|model_bkg"] = r
    for key in keys_h:
        for K in bench.K_HYBRID:
            for td in ("std", "sc"):
                ranks[f"{td}|model@{K}->{key}"] = np.array(
                    [hybrid_rank(s_model[qi], np.asarray(S_cls[key][qi]), *sets[td][qi], K, N) if len(sets[td][qi][0])
                     else -1 for qi in range(nq)], dtype=np.int64)
    np.savez_compressed(os.path.join(out, "ranks.npz"), **ranks)

    base = base_ranks(args, args.base) if args.base else {}
    boot = bench.compound_bootstrap(stems)
    cboot = np.random.default_rng(0).integers(0, len(np.unique(stems)), (2000, len(np.unique(stems))))
    groups = {"all": np.ones(nq, bool), "cu_lab": kinds == "cu", "non_cu": kinds != "cu",
              "partial_coverage": ~masks.all(1), "full_coverage": masks.all(1)}
    res = {"meta": {"tag": args.tag, "config": {k: v for k, v in cfg.items() if k != "keys"}, "B": b_value(cfg),
                    "n_library": N, "n_queries": nq, "base": args.base, "excluded": excluded,
                    "tube_counts": dict(Counter(kinds.tolist())), "n_partial_coverage": int((~masks.all(1)).sum()),
                    "score_log": open(os.path.join(out, "DONE")).read().strip() if os.path.exists(os.path.join(out, "DONE")) else None},
           "per_pattern": {}, "macro": {}, "paired_vs_base": {}, "groups": {}}
    for k, r in ranks.items():
        td = k.split("|")[0]
        res["per_pattern"][k] = bench.summarise(r, np.ones(nq, bool), boot)
        res["macro"][k] = {f"top{kk}": pn.macro(r, kk, stems, cboot) for kk in (1, 10)}
        if k in base:
            res["paired_vs_base"][k] = {f"top{kk}": pn.paired_macro(r, base[k], stems, kk) for kk in (1, 10)}
            ok = (r >= 0) & (base[k] >= 0)
            res["paired_vs_base"][k]["top1_changes"] = {
                "fixed": Counter(stems[ok & (r == 0) & (base[k] != 0)].tolist()).most_common(),
                "broken": Counter(stems[ok & (r != 0) & (base[k] == 0)].tolist()).most_common()}
        if "@" not in k:
            res["groups"][k] = {g: {"n": int((sel & (r >= 0)).sum()), "top1": float(((r == 0) & sel & (r >= 0)).sum()
                                    / max(1, (sel & (r >= 0)).sum())),
                                    "top1_base": (float(((base[k] == 0) & sel & (base[k] >= 0)).sum()
                                                        / max(1, (sel & (base[k] >= 0)).sum())) if k in base else None)}
                                for g, sel in groups.items()}
    path = args.out or f"results/rescore_{args.tag}.json"
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"[eval] {args.tag}: wrote {path}")
    for k in sorted(ranks):
        if "@" in k:
            continue
        mc = res["macro"][k]["top1"]
        pb = res["paired_vs_base"].get(k, {}).get("top1")
        print(f"  {k:22s} macro top-1 {mc['value']:.3f} [{mc['ci'][0]:.3f}, {mc['ci'][1]:.3f}]"
              + (f"  vs {args.base}: {pb['diff']:+.3f} [{pb['ci'][0]:+.3f}, {pb['ci'][1]:+.3f}]" if pb else ""))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("bvalue", "score", "eval"))
    ap.add_argument("--tag", choices=tuple(CONFIGS), default="r1")
    ap.add_argument("--base", default="v1", help="tag whose ranks the paired differences are taken against ('' = none)")
    ap.add_argument("--work", default="cache/real_bench")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--truth", default=f"{bench.DATA}/opxrd/opxrd_truth.json")
    ap.add_argument("--patterns", default=f"{bench.DATA}/opxrd/opxrd_cnrs_organic.json")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--ckpt", default="runs/xrdclip_poly/best.pt")
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--n", type=int, default=5000, help="bvalue: entries to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu", help="cpu | cuda | auto (score)")
    ap.add_argument("--no_model", action="store_true", help="eval: classical keys only (rented box: no checkpoint)")
    args = ap.parse_args()
    return {"bvalue": cmd_bvalue, "score": cmd_score, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
