# Full training run — RTX 5090 (Vast.ai)

Reference settings for the real training run. This is a markdown note, not a parsed config, so the
rationale travels with the numbers.

## One-time box setup
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # 5090 needs cu128
pip install -r requirements.txt                                        # no ccdc — not needed on GPU
# ship cache/sticks/ + refcodes/ from the licensed machine (NOT the CSD, just derived arrays)
# verify Inet >= 200 Mbit/s before moving the ~GB of stick shards
```

## Command
```bash
python -m xrdclip.train \
    --sticks cache/sticks \
    --train_refs refcodes/scaffold_train.txt \
    --val_refs   refcodes/scaffold_val.txt \
    --batch_size 512 \
    --epochs 50 \
    --lr 3e-4 \
    --weight_decay 0.05 \
    --workers 12          # CAP this — Vast boxes report hundreds of cores; 12 is plenty
```

## Why these values
- **batch_size 512** — in contrastive training the batch *is* the negative pool; embeddings are
  256-d and tiny, so 512 fits easily. If VRAM ever binds, add a MoCo memory queue rather than
  shrinking the batch (SPEC §2.3).
- **workers 12** — the dataloader renders patterns on the fly (the augmentation is the point), so it
  is CPU-bound; 12 keeps the GPU fed without the many-hundred-core thrash Vast boxes invite.
- **epochs 50** — estimated ≲3 GPU-hr total at this scale. Watch `val_top1`; `best.pt` is saved on
  the best validation in-batch top-1.
- **lr 3e-4 / OneCycle** — standard for this model size; warmup is baked into the schedule.

## Sanity gates before trusting a run
1. `--overfit 48 --epochs 60 --batch_size 48` must reach train top-1 ≈ 1.0. If not, stop.
2. First real epoch: train top-1 should climb off chance (1/512 ≈ 0.002) within a few epochs.
3. τ should settle around 0.02–0.05, not slam into the 0.01 clamp.
