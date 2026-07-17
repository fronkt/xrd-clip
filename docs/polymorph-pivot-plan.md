# xrd-clip polymorph pivot — plan (2026-07-17, DECIDED)

**Why:** the lit review (`docs/competitive-landscape-2026.md`) found xrd-clip's method pre-empted by
**XCCP** (npj 2026, inorganic MP) and **XtalNet-CPCP** (Adv Sci 2025, MOF). Both lean on
**composition + space-group** fingerprints. The pivot attacks exactly where that signal dies.

## Thesis (reframed)
**Polymorph-resolving PXRD retrieval.** Discriminate *polymorphs of the same molecule* — identical
formula, often identical space group — where the ONLY signal is packing → pattern. Composition- and
symmetry-based methods (XCCP, XtalNet, classical search-match on cell/SG) are structurally blind to this.
First such study on organic molecular crystals.

## Feasibility — VERIFIED (not assumed)
Ran `dedup.py`'s own correlation split (redeterminations corr≥0.99 collapsed; polymorphs corr<0.99 kept)
over the full simulated set (308,862 structures):
- **10,795 true polymorph families** (90.2% of the 11,973 multi-member CSD stems — only ~1,178 were pure
  redeterminations).
- **28,251 distinct polymorph structures** = retrieval targets. Distribution: 8,344 families ×2 polymorphs,
  1,359 ×3, 449 ×4, long tail. Largest stem family 78 entries.
- Artifact: **`refcodes/polymorph_families.json`** (stem → representative refcode per distinguishable
  polymorph; refcodes only, no CIFs) — the eval-set skeleton.

## Evaluation protocol (the scientific core)
1. **Within-family retrieval (the headline test).** Query = one polymorph's (augmented) pattern;
   library = its sibling polymorphs only. Can the model pick the right packing among same-compound
   siblings? Report top-1 accuracy vs #siblings. This is the test XCCP/XtalNet *cannot run* (same composition).
2. **Global retrieval with polymorph hard-negatives.** Query pattern → retrieve correct structure from the
   full dedup'd library (which now contains sibling polymorphs as hard negatives). top-1/top-k.
3. **Degradation sweep on within-family retrieval (the crux).** Polymorph patterns differ subtly — does the
   discrimination survive peak shift / Caglioti+Scherrer broadening / preferred orientation / noise? Plot
   within-family top-1 vs severity. If it survives realistic degradation, that's the paper. If it washes
   out, that's an honest, publishable boundary (and still novel).
4. **Baselines:** classical correlation + peak-list search-match on within-family retrieval (expected to
   degrade fast — subtle peak moves); a composition/SG-only retriever (expected ≈ chance within-family, by
   construction — the killer contrast figure).

## Training adjustment (modest; architecture UNCHANGED)
- **Polymorph-aware hard-negative mining:** ensure each contrastive batch contains sibling polymorphs of
  anchors, so the loss forces fine packing→pattern distinctions, not coarse composition. This is a
  sampler change to `dataset.py`, not a model change → the dual-encoder from the built pipeline stays.
- On-the-fly augmentation stays (it's what makes retrieval robust).

## Splits — family-level holdout (leakage guard)
Whole polymorph families go to a single split; hold out a family-level test set for the within-family
task so no sibling is seen in training. (Existing Bemis–Murcko scaffold split already blocks analogue
leakage; family holdout additionally blocks sibling leakage.)

## What changes vs the current build
- KEEP: dual-encoder, InfoNCE, augmentation suite, simulation, dedup.
- ADD: polymorph family holdout split; polymorph-aware negative sampler; within-family + degradation eval;
  the two baselines. Optionally adopt XCCP's low-angle/wide-angle dual-branch XRD encoder (cite XCCP).
- The full 5090 training run should launch with the negative-mining sampler — this is why we gated
  BEFORE the GPU spend.

## Timeline to ~Aug 3 (archival URTC)
- Wk1 (now): family split + eval harness (within-family + global + degradation) on the CPU pilot;
  negative-mining sampler. Data is done (`polymorph_families.json`).
- Wk2: full 5090 run with polymorph negatives; within-family + degradation results; baselines.
- Wk3: figures (within-family vs #siblings; degradation survival; killer composition-blind contrast),
  5-page write-up. Real-pattern zero-shot (opXRD/SDPD) as stretch.
- Floor unchanged: if the pivot underdelivers, [[batterycv]] backstops the URTC slot; or downgrade to poster.

## Build status — CPU-side DONE + verified (2026-07-17)
All implemented, tested, and smoke-run on CPU. The full 5090 run is the only remaining gated step.
- **Eval-set skeleton:** `refcodes/polymorph_families.json` — 10,795 families / 28,251 structures.
- **Family-safe splits:** `splits.py` gained `merge_polymorph_groups` + `--poly_families`; splits
  regenerated (`respect_polymorphs: true`), **verified 0 / 10,795 families cross a split** (was 1,665).
  Old splits backed up to `refcodes/_pre_polymorph_backup/`. New test split holds **1,077 polymorph
  families / 2,835 targets**.
- **Hard-negative sampler:** `data/polymorph.py::PolymorphBatchSampler` — atomic families, exact batch
  size, epoch reshuffle, LR-schedule-safe `__len__`. Wired into `train.py` via `--poly_families`.
- **Within-family eval:** `eval/polymorph_retrieval.py` (`within_family_retrieval` + `within_family_sweep`,
  model vs correlation vs peak-list vs chance) + `scripts/08_polymorph.py` (json + figure).
- **Tests:** `tests/test_polymorph.py` (6) green; full suite **21 passed**. Sampler bug caught+fixed
  (capped-family overflow now spills to singletons instead of being dropped).
- **Smoke run:** pilot ckpt on 30 test families → within-family top-1 = chance (0.429), as expected for
  an undertrained, polymorph-blind model. Plumbing proven; real number awaits the full run.

### Next (GPU): launch the full 5090 run with the sampler
`python -m xrdclip.train --sticks cache/sticks --poly_families refcodes/polymorph_families.json
--embed_dim 256 --batch_size 512 ...` then
`python scripts/08_polymorph.py --sticks cache/sticks --test_refs refcodes/scaffold_test.txt
--ckpt runs/xrdclip/best.pt --embed_dim 256`.

## Relationship to pxrd-flow
xrd-clip (polymorph-resolving retriever) is the natural **candidate ranker** for pxrd-flow's PXRD-conditioned
generator (XtalNet's CPCP-ranks-CCSG pattern, but molecular). Keep the pipelines/splits shared so a future
combined molecular-crystal paper is a merge, not a rebuild.
