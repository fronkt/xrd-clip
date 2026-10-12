"""Time one 2800-entry block per method on a busy box (ETA for running chunks; allocator A/B).

  PYTHONPATH=src python scripts/box/probe_chunk_speed.py
"""
import sys
import time

import numpy as np

sys.path.insert(0, "scripts")
from importlib import import_module

bench = import_module("19_real_bench")
from xrdclip.eval import strong_baselines as SB

with np.load("cache/real_bench/box_ctx.npz") as z:
    q = {v: z[f"Q__raw_{v}"][:3] for v in ("raw", "bkg")}
t = time.time()
ref = bench.render(bench.load_records("cache/sticks/shard_000.npz", 0, 2800, bench.RENDER_FIELDS))
print(f"render 2800: {time.time() - t:.1f}s", flush=True)
total = 0.0
for m, spec in bench.SPECS.items():
    lib = SB.WCCLibrary(ref, **spec)
    for v in bench.VARIANTS[m]:
        t = time.time()
        lib.scores(q[v])
        per_q = (time.time() - t) / 3
        total += per_q * 196
        print(f"{m}_{v}: {per_q:.2f} s/query -> {per_q * 196 / 60:.1f} min per chunk", flush=True)
print(f"TOTAL per 2800-entry chunk: {total / 60:.1f} min", flush=True)
