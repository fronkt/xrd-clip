"""Real-experimental-pattern retrieval eval (URTC de-risking; SPEC gap: sim-only evaluation).

Each entry in cache/real_patterns/*.json is a REAL measured powder pattern (from IUCr
supplementary information for the structure's original determination paper), already resampled
onto our grid (5-50deg 2theta, 0.02deg step, Cu-Kalpha1) but NOT yet normalised. This script:

  1. builds the reference-structure library (embed_library, same as every other eval script),
  2. preprocesses each real pattern the same way render_sticks output is preprocessed
     (max-normalise then z-score) so the encoder sees inputs on the same scale it was trained on,
  3. embeds each real pattern directly with model.encode_pattern (skip embed_queries -- that
     function RENDERS from simulated sticks; a real pattern already IS the rendered signal),
  4. ranks the whole library and reports where the query's true structure lands.

Real patterns whose refcode is not itself in the training corpus (e.g. PHBARB05, whose structure
is Platteau et al.'s Form I/II determination, not in the CSD-derived corpus store, though sibling
polymorphs PHBARB/PHBARB09/12/13 are) are reported with retrieval metrics only if the refcode is
present as a library entry; otherwise nearest-neighbour compound identity is reported instead --
do not fabricate a rank against a structure the model was never given a chance to retrieve.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sticks", default="cache/sticks")
    ap.add_argument("--library_refs", default="refcodes/scaffold_test.txt")
    ap.add_argument("--real_dir", default="cache/real_patterns")
    ap.add_argument("--ckpt", default="runs/xrdclip_poly/best.pt")
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--out", default="results/real_pattern_eval.json")
    ap.add_argument("--max_library", type=int, default=0,
                     help="cap library size (0 = full scaffold_test, ~31k); for fast iteration")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from xrdclip.data.dataset import StickStore, structure_features
    from xrdclip.data import augment as A
    from xrdclip.models.clip import XRDClip
    from xrdclip.eval.retrieve import embed_library
    from xrdclip.eval import baselines as B

    device = "cuda" if torch.cuda.is_available() else "cpu"

    real_files = sorted(glob.glob(os.path.join(args.real_dir, "*.json")))
    real_patterns = []
    for fp in real_files:
        with open(fp) as fh:
            real_patterns.append(json.load(fh))
    print(f"[real_eval] loaded {len(real_patterns)} real patterns: "
          f"{[r['refcode'] for r in real_patterns]}")

    with open(args.library_refs) as fh:
        library_refs = [ln.strip() for ln in fh if ln.strip()]
    real_refcodes = [r["refcode"].split("_Form")[0] for r in real_patterns]

    if args.max_library and len(library_refs) > args.max_library:
        rng = np.random.default_rng(args.seed)
        subset = set(rng.choice(library_refs, size=args.max_library, replace=False))
        subset |= set(real_refcodes)
        library_refs = [rc for rc in library_refs if rc in subset]
        print(f"[real_eval] subsampled library to {len(library_refs):,} (--max_library {args.max_library})",
              flush=True)

    want = set(library_refs) | set(real_refcodes)
    print(f"[real_eval] loading StickStore for {len(want):,} refcodes from {args.sticks} ...", flush=True)
    store = StickStore(args.sticks, refcodes=want)
    print(f"[real_eval] StickStore loaded: {len(store.records):,} records resolved", flush=True)

    library = [rc for rc in library_refs if rc in store.records]
    for rc in real_refcodes:
        if rc in store.records and rc not in library:
            library.append(rc)
    print(f"[real_eval] library = {len(library):,} structures", flush=True)

    model = XRDClip(embed_dim=args.embed_dim).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[real_eval] loaded {args.ckpt} (epoch {ck.get('epoch', '?')})", flush=True)

    print(f"[real_eval] embedding library ({len(library):,} structures, this is the slow step on CPU) ...",
          flush=True)
    lib_emb = embed_library(model, store, library, device)
    lib_index = {rc: i for i, rc in enumerate(library)}
    print(f"[real_eval] embedded library: {lib_emb.shape}", flush=True)

    results = []
    with torch.no_grad():
        for rec in real_patterns:
            refcode = rec["refcode"]
            base_refcode = refcode.split("_Form")[0]
            y = np.asarray(rec["pattern"], dtype=np.float32)
            m = np.abs(y).max()
            if m > 0:
                y = y / m
            y = A.standardize(y)
            x = torch.from_numpy(y)[None, None].to(device)
            q_emb = model.encode_pattern(x).cpu().numpy()          # (1, embed_dim)

            scores = (q_emb @ lib_emb.T)[0]                        # (n_ref,)
            order = np.argsort(-scores)
            top5 = [(library[i], float(scores[i])) for i in order[:5]]

            entry = {
                "refcode": refcode,
                "base_refcode": base_refcode,
                "source_doi": rec.get("source_doi"),
                "n_raw_points": rec.get("n_raw"),
                "n_library": len(library),
                "top5_nearest": top5,
            }
            if base_refcode in lib_index:
                truth_idx = lib_index[base_refcode]
                rank = int(np.where(order == truth_idx)[0][0])
                entry["own_structure_in_library"] = True
                entry["rank_of_true_structure"] = rank
                entry["percentile"] = 1.0 - rank / len(library)
                entry["top1"] = rank == 0
                entry["top10"] = rank < 10
                entry["top100"] = rank < 100
            else:
                entry["own_structure_in_library"] = False
                entry["note"] = (f"{base_refcode} itself is not a corpus structure "
                                  f"(sibling polymorphs may be); reporting nearest neighbours only")
            results.append(entry)
            print(f"[real_eval] {refcode}: "
                  + (f"rank={entry.get('rank_of_true_structure')}/{len(library)} "
                     f"(top1={entry.get('top1')})" if entry["own_structure_in_library"]
                     else f"nearest={top5[0][0]}"))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"n_real_patterns": len(results), "n_library": len(library),
                   "results": results}, fh, indent=2)
    print(f"[real_eval] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
