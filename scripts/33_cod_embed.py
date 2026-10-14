"""R5: structure embeddings of the open COD stand-in library with the ORIGINAL checkpoint (tasks/todo.md Step 3).

Same per-shard embedder as the CSD library (19's _embed_shard), with the library swapped by XRD_STICK_DIRS:

  XRD_STICK_DIRS=cod/sticks PYTHONPATH=src python scripts/33_cod_embed.py [--procs 2]
  -> cache/real_bench_cod/emb_sticks_shard_00N_<n>.npy, which scripts/28 (--work cache/real_bench_cod) reads
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from importlib import import_module

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
bench = import_module("19_real_bench")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/xrdclip_poly/best.pt")
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--out", default="cache/real_bench_cod")
    ap.add_argument("--procs", type=int, default=2)
    args = ap.parse_args()
    assert bench.STICK_DIRS == ("cod/sticks",), f"set XRD_STICK_DIRS=cod/sticks (got {bench.STICK_DIRS})"
    os.makedirs(args.out, exist_ok=True)
    jobs = [(p, n, args.ckpt, args.embed_dim,
             os.path.join(args.out, f"emb_{os.path.basename(os.path.dirname(p))}_{os.path.basename(p)[:-4]}_{n}.npy"))
            for p, n, _ in bench.shard_table(None)]
    print(f"[cod_embed] {len(jobs)} shards, {sum(j[1] for j in jobs):,} entries", flush=True)
    # one shard per fresh process, as in 19 (a reused worker grows ~1.8 GB per shard; 10-13 the reaper hit it)
    with ProcessPoolExecutor(args.procs, max_tasks_per_child=1) as ex:
        for out, dt in ex.map(bench._embed_shard, jobs):
            print("[cod_embed]", out, round(dt), "s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
