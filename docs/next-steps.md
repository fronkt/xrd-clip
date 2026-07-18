# xrd-clip — post-run next steps (toward IEEE URTC 2026)

**Status as of 2026-07-17.** The polymorph-pivot training run + full degradation sweep are **done**.
The experiment is ~90% complete; what remains is *hardening the result* and *writing*, not new capability.
This doc is the live next-steps ledger. Pre-run rationale lives in `polymorph-pivot-plan.md` +
`competitive-landscape-2026.md` (unchanged); the results/figure are committed at `results/polymorph.{json,png}`.

## What the run produced

- **Training:** 60 epochs, clean convergence, best global in-batch `val_top1 = 0.914`, τ 0.070→0.017,
  no embedding collapse. Checkpoint is CSD-trained → **stays local** (`runs/xrdclip_poly/best.pt`), never pushed.
- **Headline (clean within-family retrieval, 1,077 test families / 2,835 same-molecule siblings):**
  top-1 = **0.561** vs composition-only chance **0.380** (1.48×); top-2 0.843, top-5 0.949, MRR 0.738.
  This is the task XCCP / XtalNet (composition + symmetry) are pinned at chance on **by construction** —
  the wedge for the whole paper.

## What the degradation sweep actually shows (be honest about this)

The sweep perturbs the **query** pattern only (library stays clean-simulated = the realistic
sim-library / real-query setup) and pits **model** vs **full-profile correlation** vs **peak-list** vs chance.

- **① Peak-POSITION degradations = the model's home turf** (real-world killers: sample displacement,
  thermal expansion, zero/wavelength error).
  - 2θ zero-offset: model **flat** 0.561→0.523 while correlation **collapses** 1.0→0.298; model overtakes
    correlation at ≥0.15° and beats *everything* by 0.2° (0.535 vs corr 0.316 vs peak 0.125).
  - lattice strain: model **flat** 0.561→0.546 vs correlation 1.0→0.323, peak-list 0.899→0.370; model wins from ≥0.5%.
- **② Position-PRESERVING degradations = honest limitations** (peaks don't move → full-profile correlation stays strong).
  - **Crystallite broadening is the one genuine loss** — at 10 nm the model drops to **0.472** vs correlation 0.884.
  - March-Dollase texture + Poisson noise: model stable (~0.53–0.56) but second to correlation's ~1.0.
- **Clean caveat:** at zero degradation the model (0.561) loses to trivial baselines because clean
  within-family retrieval is self-matching an identical simulated pattern (correlation = 1.0 trivially).
  That is the *context*, not the result. Robustness here is **learned** (shift/strain were in the training
  augmentation suite), not emergent — state this plainly before a reviewer does.

## Next steps (priority order)

1. **[CPU, ~10 min, checkpoint in hand] Combined realistic-degradation operating point.** Stack moderate
   shift + strain + broadening + noise at once (the actual field condition) at one point. Converts the
   "wins on 2 of 5 axes, one-at-a-time" figure into a single decisive headline where the learned embedding
   is the *only* method well above chance. **Highest-value remaining experiment.**
2. **[GPU, ~2.6 hr, needs a fresh box] No-augmentation ablation.** Train a twin *without* shift/strain
   augmentation; show it matches on clean but collapses under degradation. The exact answer to
   "isn't this just simulated-fingerprint leak?" — the most convincing single result for an archival paper.
   The only remaining item that needs the GPU.
3. **Own the broadening weakness.** Either harden it (heavier broadening in the aug suite → folds into the
   #2 retrain) or frame it honestly as the method's boundary. Decide while writing.
4. **Write the 5-pager.** Related Work anchored on XCCP/XtalNet (composition/symmetry → pinned at chance =
   our wedge); Methods; the sweep figure reordered to lead with shift/strain; ablation table.
5. **⚠️ GATING RISK — confirm URTC 2026 eligibility + dates NOW.** Open question: Purdue affiliation vs
   URTC's HS-needs-a-collegiate-program rule; the ~Aug 3 date is unconfirmed. This is a Frank-action
   (email organizers) and it is **on the critical path** — if xrd-clip can't be a paper, the fallback is a
   poster or rerouting the finished batterycv write-up into the URTC slot. Resolve before investing in the writeup.

## Scheduling note

xrd-clip's GPU run is complete and the Vast box is destroyed. Step #2 is the only xrd-clip item still
needing the GPU; everything else is CPU/writing. **The GPU is therefore free now** — pxrd-flow Phase-B
(previously blocked until ~Aug 4) is unblocked ~2.5 weeks early. See `pxrd-flow/tasks/todo.md`.
