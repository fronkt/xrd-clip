"""Two-phase mixture retrieval (SPEC §4.4) — the most practically interesting experiment.

Real samples are frequently multiphase, and classical full-pattern correlation is known to fail on
mixtures because no single reference matches a superposition. We train single-phase and test on
50:50 (and other ratios) two-phase queries -- a zero-shot generalisation claim.

Question: for a mixture of phases A and B, does top-2 retrieval recover BOTH constituents? We report
  both@2   : fraction of mixtures where BOTH A and B are in the top-2
  either@2 : fraction where at least one is in the top-2
for the model and for correlation search-match, across mixing fractions.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch


def mixture_retrieval(model, store, reps, device, weight, n_pairs, rng, ks=2):
    from xrdclip.data import augment as A
    from xrdclip.eval import baselines as B
    from xrdclip.eval.retrieve import embed_library

    lib = embed_library(model, store, reps, device)                 # (R, d)
    refs_z = B.build_reference_patterns([store[rc] for rc in reps], A.CLEAN)  # for correlation

    idx = np.arange(len(reps))
    pairs = [rng.choice(idx, size=2, replace=False) for _ in range(n_pairs)]

    model_both = model_either = corr_both = corr_either = 0
    for a, b in pairs:
        y = A.mixture(store[reps[a]], store[reps[b]], weight, A.CLEAN, rng=rng)
        # model
        with torch.no_grad():
            zq = model.encode_pattern(torch.from_numpy(A.standardize(y))[None, None].to(device))
        top = (zq.cpu().numpy() @ lib.T)[0].argsort()[::-1][:ks]
        model_both += int(a in top and b in top)
        model_either += int(a in top or b in top)
        # correlation
        cz = A.standardize(y)
        ctop = (cz @ refs_z.T).argsort()[::-1][:ks]
        corr_both += int(a in ctop and b in ctop)
        corr_either += int(a in ctop or b in ctop)

    n = len(pairs)
    return {
        "weight": weight, "n_pairs": n,
        "model": {"both@2": model_both / n, "either@2": model_either / n},
        "correlation": {"both@2": corr_both / n, "either@2": corr_either / n},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sticks", default="cache/pilot_sticks")
    ap.add_argument("--test_refs", default="refcodes_pilot/scaffold_test.txt")
    ap.add_argument("--ckpt", default="runs/pilot_clip/best.pt")
    ap.add_argument("--out", default="results/mixtures.json")
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--n_pairs", type=int, default=300)
    args = ap.parse_args()

    from xrdclip.data.dataset import StickStore
    from xrdclip.data.dedup import dedup_library
    from xrdclip.models.clip import XRDClip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.test_refs) as fh:
        test_refs = [ln.strip() for ln in fh if ln.strip()]
    store = StickStore(args.sticks, refcodes=set(test_refs))
    reps, _ = dedup_library(store, [rc for rc in test_refs if rc in store.records])
    print(f"[mixtures] library {len(reps)} on {device}")

    model = XRDClip(embed_dim=args.embed_dim).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    model.eval()

    rng = np.random.default_rng(0)
    results = [mixture_retrieval(model, store, reps, device, w, args.n_pairs, rng)
               for w in (0.5, 0.6, 0.7)]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"n_lib": len(reps), "results": results}, fh, indent=2)

    print("\n  weight | model both@2 either@2 | corr both@2 either@2")
    for r in results:
        print(f"  {r['weight']:.2f}   |   {r['model']['both@2']:.3f}   {r['model']['either@2']:.3f}  "
              f"|   {r['correlation']['both@2']:.3f}   {r['correlation']['either@2']:.3f}")
    print(f"\n[mixtures] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
