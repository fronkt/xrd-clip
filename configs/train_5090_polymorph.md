# Full training run (POLYMORPH PIVOT) — RTX 5090 (Vast.ai)

Reference settings for the pivot run: the same dual-encoder, trained with **polymorph hard-negative
batches** and evaluated on **within-family retrieval** (docs/polymorph-pivot-plan.md). Supersedes
`train_5090.md` as the run to launch — that file's rationale (batch size, workers, τ) still holds and
is not repeated here except where the pivot changes it. Markdown note, not a parsed config.

## One-time box setup
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # 5090 needs cu128
pip install -r requirements.txt                                        # no ccdc — not needed on GPU
# Ship from the licensed machine (derived arrays only, NEVER the CSD / no CIFs):
#   cache/sticks/            simulated stick patterns (~GB)
#   refcodes/scaffold_*.txt  polymorph-SAFE splits (respect_polymorphs:true — regenerated 2026-07-17)
#   refcodes/polymorph_families.json   the family map (509 KB)
# Verify Inet >= 200 Mbit/s BEFORE moving the stick shards.
# Sanity on-box: PYTHONPATH=src python -m pytest tests/ -q   # must be 21 passed
```

## 1. Overfit gate (do this first, ~1 min, catches broken plumbing)
```bash
python -m xrdclip.train --overfit 48 --epochs 60 --batch_size 48 --sticks cache/sticks
# train top-1 must reach ~1.0. If not, STOP — the encoder/plumbing is broken, scaling won't help.
```

## 2. Full training run (with polymorph hard negatives)
```bash
python -m xrdclip.train \
    --sticks cache/sticks \
    --train_refs refcodes/scaffold_train.txt \
    --val_refs   refcodes/scaffold_val.txt \
    --poly_families refcodes/polymorph_families.json \   # <-- the pivot: sibling polymorphs as in-batch hard negatives
    --batch_size 512 \
    --epochs 60 \
    --lr 3e-4 \
    --weight_decay 0.05 \
    --embed_dim 256 \
    --workers 12 \        # CAP this — Vast boxes report hundreds of cores; 12 keeps the GPU fed
    --out runs/xrdclip_poly
```
On start it prints `polymorph hard-negative batches: 8,656 families / 21,550 structs, 482 batches/epoch`
(~18 families / ~45 sibling structs per 512-batch) — confirm the family count is non-zero (else
`--poly_families` didn't load).

## 3. Evaluation (produces the paper figures)
```bash
# HEADLINE — within-family (polymorph) retrieval + degradation sweep, model vs baselines vs chance
python scripts/08_polymorph.py \
    --sticks cache/sticks --test_refs refcodes/scaffold_test.txt \
    --families refcodes/polymorph_families.json \
    --ckpt runs/xrdclip_poly/best.pt --embed_dim 256 \
    --out results/polymorph.json --fig results/polymorph.png

# SECONDARY — global retrieval robustness (the original xrd-clip result, still worth reporting)
bash scripts/run_eval.sh cache/sticks refcodes/scaffold_test.txt runs/xrdclip_poly/best.pt 256
```

## Why the pivot changes the numbers you watch
- **`--poly_families` makes the batch harder on purpose.** Each 512-batch now carries ~18 polymorph
  families (~45 sibling structures) as the hardest possible distractors, plus singletons. Expect the
  reported **in-batch train/val top-1 to sit LOWER** than the vanilla `train_5090.md` run at the same
  epoch — that is the hard negatives biting, not a regression. The number that matters is **within-family
  top-1 from `08_polymorph.py`**, not in-batch top-1.
- **`best.pt` is still selected on general val in-batch top-1** (a fine proxy). If you want to select
  directly on the pivot metric, run `08_polymorph.py` against `--test_refs refcodes/scaffold_val.txt`
  across a couple of checkpoints and pick the best — optional, not required.
- **embed_dim 256, batch 512, ~3 GPU-hr** — unchanged; the sampler is CPU-side, the model is identical.
  `--epochs 60` (up from 50): the harder objective can use a little more.

## Sanity gates before trusting the pivot result
1. Overfit gate (§1) reached ~1.0.
2. First few real epochs: train top-1 climbs off chance (1/512 ≈ 0.002). τ settles ~0.02–0.05, not
   the 0.01 clamp.
3. **The pivot gate:** `08_polymorph.py` within-family top-1 must **clear chance (0.380 on this test
   set — 1,077 families / 2,835 targets) by a wide margin** AND beat the composition-blind dashed line
   decisively (composition/symmetry methods are pinned at chance here). If within-family
   top-1 ≈ chance, the model is not learning packing→pattern — diagnose in this order: is the sampler
   actually active (start-of-run family count > 0)? is augmentation so aggressive it erases the subtle
   polymorph peak differences (try a milder `Ranges`)? are the "siblings" real polymorphs (they are —
   dedup verified corr<0.99)?
4. Secondary check: the global robustness sweep should still show the model beating classical
   search-match under shift/broadening (the original result must survive the pivot).

## Notes
- Push checkpoints/results back to the repo after the run (or at least `history.json` + `results/`),
  per the "push after large additions" rule — a dead box mid-run should not lose the training curve.
- `refcodes/_pre_polymorph_backup/` holds the pre-pivot splits; ignore it on the GPU box.
