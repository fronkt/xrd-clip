"""Timing of a model search on a CPU (claim audit 2026-11-20, finding 7: the 5 ms of scripts/22 had no record).

  PYTHONPATH=src python scripts/55_model_timing.py      -> results/model_timing.json

Times, for the paper model (d3_sym_s0) on this machine's CPU with torch: (a) encoding one measured pattern with
the pattern encoder, (b) the product of its embedding with the 309,080 precomputed library embeddings and the
top-10 selection. Medians over 200 repeats after 20 warm-up runs, one query at a time.
"""
from __future__ import annotations

import glob
import json
import os
import platform
import sys
import time
from importlib import import_module

import numpy as np
import torch

sys.path.insert(0, "scripts")
bench = import_module("19_real_bench")
EMB = "cache/real_bench_d3_sym_s0"
CKPT = "runs/d3_sym_s0/best.pt"


def main() -> int:
    torch.set_num_threads(os.cpu_count() or 1)
    shards = bench.shard_table(None)
    files = [os.path.join(EMB, f"emb_{os.path.basename(os.path.dirname(p))}_{os.path.basename(p)[:-4]}_{n}.npy")
             for p, n, _ in shards]
    lib = torch.from_numpy(np.concatenate([np.load(f) for f in files]).astype(np.float32))
    from xrdclip.models.clip import load_checkpoint
    model = load_checkpoint(CKPT)
    model.eval()
    _, _, Q = bench.build_queries(f"{bench.DATA}/opxrd/opxrd_truth.json", f"{bench.DATA}/opxrd/opxrd_cnrs_organic.json",
                                  {rc: i for i, rc in enumerate(import_module("21_hubness").library_refcodes(shards))})
    x = torch.from_numpy(np.asarray(Q["std_bkg"][:1], dtype=np.float32)).unsqueeze(1)   # standardised, as in 26
    t_enc, t_search = [], []
    with torch.no_grad():
        for i in range(220):
            t0 = time.perf_counter()
            e = model.encode_pattern(x)
            t1 = time.perf_counter()
            top = torch.topk(e @ lib.T, 10)
            t2 = time.perf_counter()
            if i >= 20:
                t_enc.append(t1 - t0); t_search.append(t2 - t1)
    out = {"n_library": int(len(lib)), "dim": int(lib.shape[1]), "repeats": len(t_search),
           "encode_ms_median": round(1e3 * float(np.median(t_enc)), 2),
           "search_ms_median": round(1e3 * float(np.median(t_search)), 2),
           "cpu": platform.processor(), "threads": torch.get_num_threads(), "torch": torch.__version__}
    json.dump(out, open("results/model_timing.json", "w"), indent=1)
    print(out)
    del top
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
