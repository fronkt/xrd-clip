# xrd-clip competitive landscape (lit review 2026-07-17)

**Why this doc exists:** a rigorous lit sweep before the full 5090 training run turned up two 2025–2026
papers that **pre-empt xrd-clip's method**. Read this before spending GPU-weeks on the current framing.
xrd-clip is the **archival** paper (IEEE URTC ~Aug 3, Xplore copyright transfer) → novelty bar is real
and permanent. This is a STOP-and-re-plan moment, not a keep-pushing one.

## The two that hit hardest

### XCCP — *KAN-Enhanced Contrastive Learning* (arXiv [2511.04055](https://arxiv.org/abs/2511.04055); npj Comput. Mater., Feb 2026)
- **Exactly xrd-clip's method**: contrastive alignment of PXRD ↔ crystal-graph in a shared embedding for
  **structure retrieval**. Dual-expert XRD encoder (low-angle long-range branch + high-angle
  symmetry branch) + KAN projection head; crystal-graph encoder; dot-product similarity.
- Retrieval **88.98%**, space-group ID **93.39%**. KAN head concentrates on physically meaningful Bragg
  reflections → **robustness to peak-shape variation** (xrd-clip's exact selling point).
- **Dataset: 155,003 Materials Project CIFs → INORGANIC.** Generalizes to MPEA alloys; **zero-shot to
  experimental** patterns (mineral/inorganic).
- Authors: Xu, Su, Xiong, Wu, Dong, Jiang, He, Chen, Tong-Yi Zhang.

### XtalNet — *End-to-End CSP from PXRD* (arXiv [2401.03862](https://arxiv.org/abs/2401.03862); Adv. Science 2025)
- Two modules: **CPCP** (Contrastive PXRD-Crystal Pretraining = xrd-clip's retrieval) + **CCSG**
  (conditional generation = pxrd-flow's job). "First equivariant end-to-end CSP from PXRD."
- Transformer PXRD encoder + equivariant GNN; dot-product similarity. **Dataset: hMOF** (hypothetical
  MOFs); top-10 match rate 90.2% (hMOF-100), 79% (hMOF-400). Simulated, MOF — **not small-molecule organic.**

## What survives, what doesn't
- **Method novelty: GONE.** "Contrastive PXRD↔structure retrieval, robust to peak shape, attends to
  physical peaks" is XCCP + XtalNet-CPCP. A reviewer will cite both and ask "what's new?"
- **Domain survives: nobody does this on organic molecular crystals (CSD).** XCCP = MP inorganic;
  XtalNet-CPCP = hMOF. But "same method, organic domain" is a **weak** claim for an archival paper.

## The pivot that restores a real contribution: POLYMORPH discrimination
Inorganic/MOF retrieval leans on **composition + space-group** fingerprints. That signal **vanishes for
molecular polymorphs**: same molecule, same formula, often same space group — the *only* discriminating
signal is the packing → the pattern. XCCP's space-group-centric approach structurally cannot do
same-composition polymorph retrieval. **Retrieving/discriminating among molecular polymorphs of one
molecule is genuinely un-done, molecular-specific, and hard** — and it's where a pure pattern↔packing
contrastive model must earn its keep. It also connects directly to the existing `dedup.py` (which already
separates polymorphs from redeterminations). This reframes xrd-clip from "me-too contrastive XRD" to
"polymorph-resolving retrieval where composition is uninformative."

## Lessons to adopt regardless of pivot
- **Dual-branch low/high-angle encoder** (XCCP) — the 1D ResNet likely benefits from an explicit
  low-angle (long-range/lattice) vs wide-angle (symmetry) split. Cite XCCP either way.
- **Frame the real-pattern eval as "zero-shot transfer to experimental"** (XCCP's headline) — opXRD /
  SDPD real patterns, same framing.
- **Cite XtalNet's CPCP+CCSG synergy**: their retrieval model *ranks* the generator's candidates. That is
  the natural bridge to pxrd-flow (see below).

## Strategic synthesis (both projects)
xrd-clip (retrieval) and pxrd-flow (conditioned generation) are the **CPCP and CCSG halves of XtalNet,
but for molecular crystals.** XtalNet proved (on MOFs) they work best together — retrieval ranks
generation. Nobody has done the molecular-crystal version. Two standalone papers now; a combined
molecular-crystal XtalNet is the future journal story.

## Options for xrd-clip (decide before the GPU run)
1. **Polymorph pivot** — reframe eval around molecular-polymorph retrieval; architecture mostly unchanged,
   eval set becomes polymorph-family-focused. Restores novelty; tight but feasible by ~Aug 3.
2. **Downgrade to poster / non-archival** — sidesteps the novelty bar; keeps the work visible; frees the
   archival slot ([[batterycv]] already backstops URTC).
3. **Fold into the combined story** — hold xrd-clip as the ranker inside a later molecular-XtalNet journal paper.
