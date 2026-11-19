"""Resubmission D3 (tasks/todo.md Section D, E1): library embeddings of a retrained checkpoint, on a GPU.

Writes the two embedding sets the real-pattern evaluations read from an --emb_dir:
  emb_<stickdir>_<shard>_<n>.npy   structure embeddings of every library entry (19's naming; read by 28 eval, 30 eval)
  lib_pattern_emb.npy              pattern embeddings of the clean library renderings (30 pemb's output; 30 eval p2p)
The computations are 19's _embed_shard and 30's cmd_pemb, moved to --device with larger batches. The GPU numbers
match the CPU path (checked on the first --parity entries of each set before anything is written) only with TF32 off
AND the CUDA fused-attention inference fast path of nn.TransformerEncoderLayer off: with the fast path the structure
embeddings differ from the CPU ones by up to 6.7e-4 (cosine 1.3e-3), without it by 1.8e-6; the SDPA backend and the
batch padding change nothing (A100, torch 2.8, 11-18). The CPU parity reference keeps the default path, as 19 did.
A symops checkpoint refuses to run unless every shard has its symmetry-operator twin: without one, load_records
hands the encoder identity operators and nothing complains.

  PYTHONPATH=src python scripts/44_d3_embed.py --ckpt runs/d3_sym_s0/best.pt --emb_dir cache/real_bench_d3_sym_s0
  smoke (CPU, reproduces cache/real_bench_realism):
  PYTHONPATH=src python scripts/44_d3_embed.py --ckpt runs/xrdclip_realism/best.pt --emb_dir <tmp> --device cpu \
      --smoke 512 --compare cache/real_bench_realism
  open COD replica (structure embeddings only; its twins come from 40 --cod_cif_root):
  XRD_STICK_DIRS=cod/sticks PYTHONPATH=src python scripts/44_d3_embed.py --ckpt runs/d3_sym_s0/best.pt \
      --emb_dir cache/real_bench_cod_d3_sym_s0 --structures_only --device cpu
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from importlib import import_module

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
bench = import_module("19_real_bench")
gap = import_module("30_model_gap")


def emb_name(path: str, n: int) -> str:
    return f"emb_{os.path.basename(os.path.dirname(path))}_{os.path.basename(path)[:-4]}_{n}.npy"


def save(path: str, arr: np.ndarray) -> None:
    np.save(path + ".tmp.npy", arr.astype(np.float32))
    os.replace(path + ".tmp.npy", path)


def encode_patterns(model, x: np.ndarray, device: str, batch: int) -> np.ndarray:
    """30's encode(): standardise each max-normalised pattern, then the pattern encoder."""
    from xrdclip.data import augment as A
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            z = np.stack([A.standardize(y.astype(np.float32)) for y in x[i:i + batch]])
            out.append(model.encode_pattern(torch.from_numpy(z).unsqueeze(1).to(device)).cpu().numpy())
    return np.concatenate(out)


@contextlib.contextmanager
def device_path(device: str):
    """Around every --device encoding: on CUDA, nn.TransformerEncoderLayer's fused inference path off (see above)."""
    old = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(old and not device.startswith("cuda"))
    try:
        yield
    finally:
        torch.backends.mha.set_fastpath_enabled(old)


def parity(name: str, a: np.ndarray, b: np.ndarray, tol: float) -> None:
    d = float(np.abs(a - b).max())
    print(f"[parity] {name}: max |diff| = {d:.2e} over {len(a)} rows (tol {tol:g})", flush=True)
    if not d <= tol:
        raise SystemExit(f"[parity] {name} FAILED")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--parity", type=int, default=256, help="entries per set checked against the CPU path (0 = skip)")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--smoke", type=int, default=0, help="only the first N entries of the first shard / library")
    ap.add_argument("--compare", default="", help="smoke: an emb_dir made by the old path with the same checkpoint")
    ap.add_argument("--structures_only", action="store_true",
                    help="library structure embeddings only (a library without 30's clean renderings, e.g. COD)")
    args = ap.parse_args()

    from xrdclip.data.dataset import symops_twin
    from xrdclip.eval.retrieve import embed_library
    from xrdclip.models.clip import load_checkpoint

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.makedirs(args.emb_dir, exist_ok=True)
    model = load_checkpoint(args.ckpt).to(args.device).eval()
    cpu_model = load_checkpoint(args.ckpt).eval() if (args.parity and args.device != "cpu") else None
    shards = bench.shard_table(None)
    print(f"[d3_embed] {args.ckpt} symops={model.symops} device={args.device} -> {args.emb_dir}; "
          f"{len(shards)} shards, {sum(n for _, n, _ in shards):,} entries", flush=True)
    if model.symops:
        missing = [p for p, _, _ in shards if not os.path.exists(symops_twin(p))]
        if missing:
            raise SystemExit(f"[d3_embed] symops checkpoint but no operator twin for {missing}")
    if args.smoke:
        shards = shards[:1]

    t0 = time.time()
    for p, n, _ in shards:
        out = os.path.join(args.emb_dir, emb_name(p, n))
        if os.path.exists(out) and not args.smoke:
            print(f"[d3_embed] have {out}", flush=True)
            continue
        m = args.smoke or n
        recs = bench.load_records(p, 0, m, bench.EMBED_FIELDS)
        if model.symops and any(r["ops"] is None for r in recs):
            raise SystemExit(f"[d3_embed] {p}: entries without operators")
        store = {str(i): r for i, r in enumerate(recs)}
        ids = [str(i) for i in range(m)]
        with device_path(args.device):
            emb = embed_library(model, store, ids, args.device, batch=args.batch)
        if emb.shape[0] != m or not np.isfinite(emb).all():
            raise SystemExit(f"[d3_embed] {p}: bad embeddings {emb.shape}")
        if cpu_model is not None:
            k = min(args.parity, m)
            parity(os.path.basename(p), emb[:k], embed_library(cpu_model, store, ids[:k], "cpu", batch=128), args.tol)
        if args.compare:
            ref = np.load(os.path.join(args.compare, emb_name(p, n)), mmap_mode="r")[:m]
            parity(f"{os.path.basename(p)} vs {args.compare}", emb, np.asarray(ref), args.tol)
        if args.smoke:
            print(f"[d3_embed] smoke: {m} structure embeddings OK ({time.time() - t0:.0f} s)", flush=True)
            break
        save(out, emb)
        print(f"[d3_embed] {out} ({time.time() - t0:.0f} s)", flush=True)

    if args.structures_only:
        print(f"[d3_embed] structures only ({time.time() - t0:.0f} s) DONE", flush=True)
        return 0
    lib = np.load(gap.LIB, mmap_mode="r")
    n_lib = sum(n for _, n, _ in bench.shard_table(None))
    if lib.shape[0] != n_lib:
        raise SystemExit(f"[d3_embed] {gap.LIB} has {lib.shape[0]} rows, the library {n_lib}")
    path = os.path.join(args.emb_dir, gap.PEMB)
    N = args.smoke or lib.shape[0]
    if os.path.exists(path) and not args.smoke:
        print(f"[d3_embed] have {path}", flush=True)
        return 0
    if cpu_model is not None:
        k = min(args.parity, N)
        x = np.asarray(lib[:k])
        with device_path(args.device):
            pg = encode_patterns(model, x, args.device, args.batch)
        parity("lib patterns", pg, encode_patterns(cpu_model, x, "cpu", 256), args.tol)
    step = 16384
    pe = None
    for i in range(0, N, step):
        with device_path(args.device):
            z = encode_patterns(model, np.asarray(lib[i:min(i + step, N)]), args.device, args.batch)
        pe = np.empty((N, z.shape[1]), np.float32) if pe is None else pe
        pe[i:i + len(z)] = z
        print(f"[pemb] {min(i + step, N):,}/{N:,} {(time.time() - t0) / 60:.1f} min", flush=True)
    if not np.isfinite(pe).all():
        raise SystemExit("[d3_embed] non-finite pattern embeddings")
    if args.compare:
        parity(f"lib patterns vs {args.compare}", pe, np.asarray(np.load(os.path.join(args.compare, gap.PEMB),
                                                                          mmap_mode="r")[:N]), args.tol)
    if args.smoke:
        print(f"[d3_embed] smoke: {N} pattern embeddings OK", flush=True)
        return 0
    save(path, pe)
    print(f"[d3_embed] {path} ({(time.time() - t0) / 60:.1f} min) DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
