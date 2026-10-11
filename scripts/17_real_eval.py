"""Phase 2 decisive test: model vs strong classical baselines on REAL measured organic patterns.

Queries: opXRD CNRS organic patterns whose CSD match (scripts/16_...) is in the xrd-clip corpus.
Library: every query target + every corpus polymorph sibling of a target + N random corpus decoys
(fixed seed), identical for all methods. Each method ranks the whole library per query.

Real-pattern preprocessing: convert 2theta to the Cu K-alpha1 scale through d-spacing (synchrotron
data included), resample onto the model grid (5-50 deg), zero outside the measured range, then two
variants: `raw` (max-normalised) and `bkg` (rolling-minimum background removed, then normalised).
K-alpha2 is left in lab data (classical and learned methods see the same input).

Reports global top-k / median rank, within-family top-1 for targets with corpus siblings, and the
hybrid "model top-K shortlist -> WCC rerank". Per-query ranks are saved for later analysis.

  PYTHONPATH=src python scripts/17_real_eval.py
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from xrdclip.data import augment as A
from xrdclip.eval import strong_baselines as SB
from xrdclip.eval.real_patterns import preprocess, temp_k, to_cu_grid

DATA = "C:/Users/frank/xrd-clip-data"


def ranks_of(scores: np.ndarray, truth: np.ndarray) -> np.ndarray:
    t = scores[np.arange(len(truth)), truth][:, None]
    return (scores > t).sum(axis=1)          # 0 = top-1 (ties resolved in the target's favour)


def summary(r: np.ndarray) -> dict:
    return {**{f"top{k}": float((r < k).mean()) for k in (1, 5, 10, 100)},
            "median_rank": float(np.median(r) + 1), "mrr": float((1 / (r + 1)).mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default=f"{DATA}/opxrd/opxrd_csd_match.json")
    ap.add_argument("--patterns", default=f"{DATA}/opxrd/opxrd_cnrs_organic.json")
    ap.add_argument("--families", default="refcodes/polymorph_families.json")
    ap.add_argument("--sticks", default="cache/sticks")
    ap.add_argument("--ckpt", default="runs/xrdclip_poly/best.pt")
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--n_decoys", type=int, default=20000)
    ap.add_argument("--out", default="results/real_eval.json")
    args = ap.parse_args()

    from xrdclip.data.dataset import StickStore
    from xrdclip.models.clip import XRDClip
    from xrdclip.eval.retrieve import embed_library

    match = {m["id"]: m for m in json.load(open(args.match))}
    pats = {o["id"]: o for o in json.load(open(args.patterns))["organic"]}
    fams = json.load(open(args.families))["families"]
    fam_of = {rc: stem for stem, rcs in fams.items() for rc in rcs}
    corpus = [ln.strip() for ln in open(f"{DATA}/corpus_refcodes.txt") if ln.strip()]
    splits = {}
    for sp in ("scaffold_train", "scaffold_val", "scaffold_test"):
        p = f"refcodes/{sp}.txt"
        if os.path.exists(p):
            for ln in open(p):
                splits[ln.strip()] = sp.split("_")[1]

    qs = [(pid, m["best"]) for pid, m in match.items() if m["best"] and m["best"]["in_corpus"]]
    targets = sorted({b["refcode"] for _, b in qs})
    siblings = sorted({rc for t in targets if t in fam_of for rc in fams[fam_of[t]]})
    rng = np.random.default_rng(0)
    pool = sorted(set(corpus) - set(targets) - set(siblings))
    decoys = list(rng.choice(pool, size=min(args.n_decoys, len(pool)), replace=False))
    lib_rcs = sorted(set(targets) | set(siblings)) + sorted(decoys)
    idx = {rc: i for i, rc in enumerate(lib_rcs)}
    print(f"[real] {len(qs)} queries / {len(targets)} distinct targets; library {len(lib_rcs)} "
          f"({len(siblings)} family members, {len(decoys)} decoys)", flush=True)

    store = StickStore(args.sticks, refcodes=set(lib_rcs))
    ref_raw = np.stack([A.render_sticks(store[rc], A.CLEAN) for rc in lib_rcs])
    ref_std = np.stack([A.standardize(y) for y in ref_raw])

    device = "cpu"
    model = XRDClip(embed_dim=args.embed_dim)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    model.eval()
    with torch.no_grad():
        lib_emb = embed_library(model, store, lib_rcs, device)
    print("[real] library rendered + embedded", flush=True)

    truth = np.array([idx[b["refcode"]] for _, b in qs])
    meta = [{"id": pid, "refcode": b["refcode"], "split": splits.get(b["refcode"], "?"),
             "csd_T": temp_k(b["temperature"]), "wavelength": match[pid]["wavelength"],
             "family": fam_of.get(b["refcode"]), "powder_study": b["is_powder_study"]} for pid, b in qs]

    # Built one at a time inside the loop: each caches ~330 MB of reference FFTs for 20k entries.
    lib_specs = {"cos_zs": dict(l_deg=0.0), "wcc_1": dict(l_deg=1.0), "wcc_2": dict(l_deg=2.0),
                 "lnq": dict(lnq=True)}
    res = {"n_queries": len(qs), "n_library": len(lib_rcs), "variants": {}, "meta": meta}
    for variant in ("raw", "bkg"):
        q_raw = []
        for pid, _ in qs:
            wl = float(match[pid]["wavelength"] or A.WAVELENGTH)
            y, mask = to_cu_grid(pats[pid]["two_theta"], pats[pid]["intensity"], wl)
            q_raw.append(preprocess(y, mask, bkg=(variant == "bkg")))
        q_raw = np.stack(q_raw)
        q_std = np.stack([A.standardize(y) for y in q_raw])
        with torch.no_grad():
            q_emb = model.encode_pattern(torch.from_numpy(q_std).unsqueeze(1)).numpy()
        S = {"model": q_emb @ lib_emb.T, "pearson": q_std @ ref_std.T / q_std.shape[1]}
        for name, spec in lib_specs.items():
            lib = SB.WCCLibrary(ref_raw, **spec)
            S[name] = lib.scores(q_raw)
            del lib
            print(f"[real] {variant} {name} done", flush=True)
        out = {}
        R = {m: ranks_of(s, truth) for m, s in S.items()}
        for k in (100, 1000):     # hybrid: model shortlist, then the best classical score reranks it
            for cl in ("wcc_2", "lnq"):
                sc = np.full_like(S[cl], -np.inf)
                top = np.argsort(-S["model"], axis=1)[:, :k]
                np.put_along_axis(sc, top, np.take_along_axis(S[cl], top, axis=1), axis=1)
                R[f"model@{k}->{cl}"] = ranks_of(sc, truth)
        for m, r in R.items():
            out[m] = summary(r)
            # within-family: rank among the target's corpus polymorph siblings only
            if m in S:
                wf = []
                for qi, (_, b) in enumerate(qs):
                    stem = fam_of.get(b["refcode"])
                    if stem is None or len(fams[stem]) < 2:
                        continue
                    cols = [idx[rc] for rc in fams[stem] if rc in idx]
                    wf.append(int(S[m][qi, cols].argmax() == cols.index(truth[qi])))
                out[m]["within_family_top1"] = float(np.mean(wf)) if wf else None
                out[m]["n_within_family"] = len(wf)
            out[m]["ranks"] = R[m].tolist()
        res["variants"][variant] = out
        line = "  ".join(f"{m}: t1 {v['top1']:.2f} t10 {v['top10']:.2f} med {v['median_rank']:.0f}"
                         for m, v in out.items())
        print(f"[real] {variant}: {line}", flush=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"[real] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
