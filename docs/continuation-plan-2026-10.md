# xrd-clip v2: continuation plan (2026-10-10)

Written the day xrd-clip was presented at URTC 2026 (paper ID-69). Built from a four-track literature
sweep plus two adversarial fact-check passes. The raw notes, including findings that were wrong and
their corrections, are in `docs/lit-2026-10/`:

| File | Content |
|---|---|
| `lit_A_ml_retrieval.md` | 50 ML-for-PXRD papers: retrieval, phase ID, generative, agents |
| `lit_B_classical.md` | Classical search-match and similarity measures, with baseline pseudo-code |
| `lit_C_realdata.md` | Real organic PXRD sources, sim-to-real methods, thermal-expansion sizes |
| `lit_D_levers.md` | Analogues from MS/NMR/IR retrieval, rerank architectures, fine-grained contrastive tricks |
| `refute_1.md`, `refute_2.md` | Fact-check verdicts. Nothing fabricated; 8 claims corrected (see Section 9) |

---

## 1. Recommendation in one paragraph

Don't try to make the embedding beat the library by itself. Make xrd-clip the **fast, robust first
stage of a polymorph-identification pipeline**: the embedding shortlists about 20–100 candidates in
milliseconds, and a physics-based (later learned) reranker compares the query against those
candidates' simulated patterns. Measure it on **real organic patterns** against the **strongest**
classical pipeline, which is a de Gelder prefilter followed by a variable-cell VC-GPWDF re-rank. It
is not plain Pearson. The win we can realistically get is the same accuracy as the best classical
pipeline at 100× or more lower cost, plus outright wins on texture, impurity and noisy real data.
The first step is free and decides everything else: rerun the existing simulated sweeps against the
strong baselines (Phase 0).

## 2. What "beat the library method" has to mean

A reviewer from the powder or CSP community will accept a win only under these conditions:

- **Query:** real measured patterns are the headline. Simulated stress tests are supporting evidence.
- **Library:** simulated patterns of CSD structures. Classical tools use the same library, so the
  comparison is fair.
- **Opponents:** the ladder in Section 4 (Phase 0). The URTC baselines (fixed-grid Pearson, naive
  peak list) are strawmen for the shift and strain claim; `lit_B` Section 2 documents why.
- **Metrics:**
  - within-polymorph-family top-1 (headline; no published learned-retrieval paper reports it);
  - global hit@k against the full 31k library;
  - **accuracy against wall-clock** (Pareto curve), because speed is where the learned stage
    honestly wins;
  - all metrics with bootstrap CIs over families.
- **Scope:** identifying **known** forms only. CSP plus PXRD matching (Glick et al. 2025,
  arXiv 2507.16218) can find forms absent from any library, and retrieval cannot.

## 3. What the literature changed

### 3a. Threats, fact-checked

| Finding | Source | What it means for us |
|---|---|---|
| The strongest classical method is **VC-GPWDF**. It computes de Gelder similarity on peak lists, globally optimised over a 6-parameter cell strain (±10% lengths, ±5° angles), with no indexing. On a degraded blind-test pattern it ranked the true structure 1st of 310. It costs 31 s on average (102 s max) per candidate per core. Open source in critic2 (`COMPAREVC`, GPL-3). | Otero-de-la-Roza, J. Appl. Cryst. 57:1401 (2024), 10.1107/S1600576724007489 | It would very likely erase our shift/strain advantage (that is an inference; Phase 0 tests it). It has no zero-shift parameter, but the paper says one can be applied in pre-processing. At about 86 core-hours per query over 10⁴ candidates, a full search is impractical, so it is used as a re-ranker. |
| On real pharmaceutical polymorphs, **de Gelder WCC is as good as a learned model**. On 3,750 experimental patterns of 16 compounds, the Siamese network wins AUROC (0.98 vs 0.92) but loses accuracy (0.89 vs 0.92) and F1 (0.80 vs 0.93). | SMolNet, arXiv 2411.03308 | WCC is a mandatory baseline. It is also the only real organic polymorph benchmark found, and its data is not public. |
| On measured inorganic patterns, **plain correlation beat a contrastive model**: Recall@10 of 82.8% vs 5.3%, or vs **34.4%** with physics-based corruption training (the fair comparison). Library of 492 candidates. | RealBind, ChemRxiv 10.26434/chemrxiv.15009668 (Sep 2026, single author, preprint) | Learned-only retrieval breaks on real data. Our failed ibuprofen spot check (rank 1,872 of 30,887) fits that pattern. |
| A **non-learned** peak-feature vector index beat JADE, 89.0% vs 68.7% top-1 on 875 real patterns. Caveat: JADE was run with elemental priors and a different library. | MatDiffract, arXiv 2607.20880 | "Embedding retrieval beats search-match" is not evidence for *learning*. Add a hand-crafted peak-vector baseline. |
| Full-spectrum encoders scored under 1% on RRUFF real data without background/noise augmentation, and roughly matched peak-based encoders with it. The metric is a generative match rate, not retrieval. | XRDiff, arXiv 2606.14003 (corrected in `refute_1` #4) | Our encoder reads full profiles. Real-data transfer depends on background/noise realism. That is a fix, not a redesign. |
| **PXtal** replaces InfoNCE with unbalanced optimal transport and gets 67.26% → 74.45% Acc@1 on MP-100. The gain is concentrated where distinct structures have near-identical patterns. Inorganic and QMOF only, simulated only, no shift/strain robustness tests. | arXiv 2610.10653 (7 Oct 2026, Rosseinsky group) | A cheap loss swap aimed at polymorph confusion. The field is moving fast and this group also works on organics. |

### 3b. Gaps that still appear open

Searched in arXiv, OpenAlex, Crossref, Europe PMC and the open web. The query lists are in
`lit_A` Section 4 and `refute_1` #13. Semantic Scholar was rate-limited. Scopus, WoS and paywalled
full texts were not searched.

1. **G1:** contrastive PXRD↔structure retrieval trained on CSD organics and evaluated against
   search-match. Every contrastive model found (XtalNet, XCCP, PXRDGen encoder, MatBind, UniMat,
   PXtal) is inorganic or MOF.
2. **G2:** within-polymorph-family *learned retrieval* as a metric. A classical precedent exists:
   Glick et al. match predicted structures against PXRD.
3. **G3:** controlled 2θ-offset and strain sweeps comparing a learned embedding with correlation on
   organics.
4. **G4:** an **open**, labelled, real organic PXRD test set. None found. If we build one and release
   it, that is a contribution in its own right; it can also be a separate data paper.

### 3c. Lessons from other spectroscopies

- In MassSpecGym, ranking by **simulate-then-compare** beats learned joint embeddings for exact
  identity: FraGNNet 46.64% vs MIST 14.64% hit@1. Both figures come from the benchmark's own tables
  (corrected in `refute_1` #10).
- The published winners are **two-stage**:
  - MS2Query: learned shortlist, then a random-forest reranker;
  - GLMR: a learned rerank over a contrastive shortlist.
- **Cheap metadata gates** add a lot:
  - CReSS: 91.6% → 98.4% recall@10 with a molecular-weight filter;
  - VibraCLIP: 98.9% top-25 once molecular mass is added.
- PXRD's forward model is exact physics, which strengthens the case for reranking in pattern space.

## 4. Plan: phases with gates

Phases 0–2 need no GPU and no money, and can start now. Phase 2 runs in parallel with 0–1.

### Phase 0: strong baselines on the existing simulated evaluation

Runs on CPU with the existing `runs/xrdclip_poly/best.pt`.

Add to `src/xrdclip/eval/baselines.py`:

- **B1: de Gelder WCC plus a zero-shift scan.**
  - Triangle half-widths l = 0.5°, 1°, 2° (2° is the CSD API/Mercury default).
  - Scan δ out to the largest offset in the test set.
  - Validate against the CSD Python API's WCC on about 20 pairs.
  - Also report "Pearson + δ scan" as an ablation.
- **B2: zero-shift scan plus cross-correlation over lag in ln q.**
  - Isotropic strain becomes an exact uniform shift in ln q. It is only invariant when the score is
    maximised over lag, and peak widths and range edges still break it slightly.
  - A 2θ zero offset is *not* uniform in ln q (−cot θ·δ/2), which is why both scans are needed.
  - This baseline is our own construction (`refute_2` #11).
- **B3: VC-GPWDF.**
  - Reimplement the closed form in torch: I_AB = √z ΣΣ I_i I_j exp(−πz(θ_i − θ_j)²), with z = 1/(4πσ² + α²).
  - Optimise 6 strain parameters plus δ with multi-start L-BFGS, batched on GPU or CPU.
  - Validate against critic2 `COMPAREVC` on 20–50 pairs (critic2 is Linux-first, so use WSL).
  - A batched, validated reimplementation is itself a small contribution, because it makes the
    strongest baseline fast.
- **B4: hand-crafted peak-feature vector index** (MatDiffract-style): peak positions, intensities
  and widths, embedded and searched with nearest neighbours.

Then:

- Rerun `scripts/08_polymorph.py` and `scripts/09_combined.py` across **all** axes for every method:
  - shift, strain (isotropic and **anisotropic per-axis**), broadening, texture, noise;
  - field average and sign decomposition.
- Record per-query wall-clock for every method.

**Gate G0:**

| Outcome | Meaning | Next step |
|---|---|---|
| The embedding still beats B1/B2 on some axis at equal cost | Part of the robustness story survives | Keep that axis in the paper and go to P1 |
| B1–B3 match or beat the embedding everywhere (expected) | The URTC robustness edge was a weak-baseline artefact | Say so plainly in v2. The learned stage must earn its place through **shortlist speed** (P1) and **real-data robustness** (P2–P4) |

### Phase 1: training-free hybrid

Runs on CPU.

- Measure the embedding's recall@k ceiling for k = 5…200, within-family and against the full library.
- Rerank the shortlist with B1, B2 and B3, plus a learned fusion scorer (Stage 2b in `lit_D`
  Section 3). The fusion scorer is a gradient-boosted model over embedding cosine and rank, the B1–B3
  scores, the fitted shift and strain, and cell-volume/density deltas, trained on training-split
  shortlists.
- Plot the accuracy-against-wall-clock Pareto curve for: embedding alone, B1 over the full library,
  B1→B3, embedding→B3, embedding→fusion.

**Gate G1:**

| Outcome | Meaning | Next step |
|---|---|---|
| embedding→B3 matches B1→B3 accuracy (within CI) at lower cost, **or** the embedding's recall@k beats B1's at the same k | The learned shortlist is justified | Continue to P3/P4 |
| Neither holds | On simulated data the learned part adds nothing | It must win on real data (P2), otherwise **stop** (Section 5) |

### Phase 2: real organic test set

Runs in parallel with P0–P1.

**Tier A (open; can start immediately):**
- Download opXRD (Zenodo 10.5281/zenodo.15298026, CC BY 4.0) and take its `CNRS/` subset of 1,052
  COD patterns. Filter to molecular organics: C and H present, only non-metal elements, atomic
  coordinates present.
- Match the entries to CSD refcodes with the CSD API (formula plus Niggli-reduced cell, then packing
  similarity).
- Add the DASH example `.xye` files (MIT licence) and the SDPD round-robin organic pattern.

**CSD powder-study check (about 1 hour):** the CSD holds 10,408 powder-diffraction studies. Check
whether their deposited CIFs carry `_pd_meas_counts` / `_pd_proc_intensity` loops. If they do, this
becomes the largest pool by far.

**Tier C (requests):**
- I draft the emails and **Frank sends them**:
  - pdj@icdd.com for raw data from Kaduk's pharmaceutical structures in *Powder Diffraction*;
  - the Johnson group for the 7-compound VC-xPWDF patterns and the ROY high-throughput patterns.
- Scrape the IUCr Rietveld pdCIF supplements.

**Tier B — DEFERRED (Frank, 2026-10-10): no Purdue diffractometer until after college apps and STS.** G2 must be met from Tier A + the CSD powder-study check + Tier C alone; if those give <100 patterns or <5 polymorph families, run G2 on what exists and report it as a pilot, then add Tier B later as the confirmatory set.

**Tier B (later; needs a diffractometer):**
- Self-measure 60–80 patterns of polymorph families at Purdue.
- Candidates:
  - from Mayo 2023 (refcodes verified): acetaminophen HXACAN, D-mannitol DMANTL, caffeine NIWFEE,
    urea, uracil, progesterone;
  - unverified additions: carbamazepine, glycine α/β/γ, ROY, ibuprofen.
- Protocol:
  - two preparations each (back-loaded and top-pressed) to vary preferred orientation;
  - fast and slow scans;
  - LaB6 or Si once per session for the instrument function;
  - one or two compounds at 100–150 K if a cold stage exists, which gives a controlled thermal-shift
    measurement.

**Ibuprofen failure diagnosis (do first; cheap):** `scripts/11_real_pattern_eval.py` max-normalises
and z-scores the real pattern with **no background subtraction and no Kα2 handling**. The structure
in the CSD is low-temperature. Bond 2021 (10.1107/S2052520621003309) gives a mean volumetric
expansion of 161–173 ppm/K, which is about 3% in volume, or about 1% linear strain, from 100 K to
295 K. That is about 0.2° of shift at 2θ = 20°, and the shift differs by axis. Rerun the spot check
under four conditions:
1. as is;
2. with background subtracted;
3. with B1/B3 instead of the embedding;
4. with the embedding followed by B3.

**Gate G2:** at least 100 labelled real patterns covering at least 5 polymorph families. Evaluate
every method; these are the first real numbers. Each pattern is tagged with its source, its
temperature gap and whether its exact refcode is in the library.

### Phase 3: model v2 retrain

Runs on GPU. Vast 5090 runs take about 2.6 h each at $0.3–0.7/h, so about $1–2 per run; a cap of
8 runs keeps the total under about $25. Get Frank's OK before renting, and every box queue ends in a
self-stop plus idle watchdog.

Change one thing per A/B comparison, in this order:

1. **Realism augmentation:**
   - Kα2 doublet;
   - instrument function measured from LaB6;
   - varied background shapes, including an amorphous hump;
   - impurity phases, which Schopmans 2023 found essential;
   - **anisotropic per-axis strain** sized to Bond 2021: per-axis centre about 33 ppm/K, about 30%
     of structures with a contracting axis.
2. **Hard-negative polymorph mining from the model's own top-k**, with COMPACK/VC-PWDF duplicates
   merged into multi-positives so they are not treated as false negatives (the RocketQA lesson).
3. **Loss:** SigLIP (best in SMolNet) vs unbalanced OT (PXtal) vs InfoNCE.
4. **A peak / q-space input branch** next to the profile CNN, because peak-based encoders transfer
   better to real data (XRDiff, RealPXRD-Solver).
5. **Embedding size:** 512–1024-d or Matryoshka nested embeddings. This targets recall@k, not top-1.

**Gate G3:** v2 raises **real-set recall@k** (shortlist quality) with a bootstrap CI that excludes
zero.

### Phase 4: learned reranker

This is the research contribution. Build a pattern-to-pattern cross-encoder, comparing the query
with each candidate's simulated pattern:

- a shared 1D CNN trunk;
- cross-attention between the two patterns;
- an optional soft-DTW alignment layer;
- listwise loss over de-duplicated shortlists.

It cannot beat Pearson on clean simulated queries, so it is aimed at the regimes where classical
scores degrade: **preferred orientation, impurity peaks, noise and real patterns**.

On preferred orientation specifically: VC-xPWDF still ranked d-mannitol correctly, scoring 0.096–0.099
against a 0.1 match threshold. So "VC methods break under texture" is overstated (corrected in
`refute_2` #2); the margin is thin, not gone.

**Gate G4:** the cross-encoder beats the B3 rerank on the real set, or on a texture/impurity stress
set.

### Phase 5: paper

- **Story:** "Fast, robust polymorph identification from powder XRD: a learned shortlist plus physics
  re-ranking, benchmarked on the first open real organic test set."
- **Venue (recommended): Digital Discovery.**
  - It has published the deCIFer real-data paper and Schopmans 2023.
  - Alternatives: J. Appl. Cryst. (powder audience; pxrd-diff's HAT5032 is already under review
    there) or Cryst. Growth Des. / CrystEngComm (polymorph audience).
  - Optional early checkpoint: an AI4Mat 2027 workshop paper.
- **IEEE copyright:** the URTC paper's copyright is with IEEE. v2 must be substantially new work (new
  method, real data, new baselines). Cite the URTC paper and don't reuse its figures or text. Check
  the target journal's prior-publication policy before submitting.

## 5. Kill and pivot criteria

| Condition | Action |
|---|---|
| G1 fails **and** G2's real numbers show the embedding's real recall@100 is no better than B1's | Stop the learned-retrieval line. Salvage the real organic test set as a data/benchmark paper (it fills gap G4), plus the fast torch VC-GPWDF. |
| G3 fails (retraining doesn't move real recall) | Ship P1 + P2 as the paper: hybrid on real data, honest that the learned stage is a speed component. |
| A competitor publishes CSD-organic contrastive retrieval with a real-data evaluation first | Re-scope to the benchmark plus hybrid. Search arXiv for "PXRD contrastive organic" monthly. |

## 6. Claims v2 must not make

- "The learned embedding beats search-match" without B1–B3 on every axis.
- "Robust to shift/strain" as a unique advantage. Classical shift scans and variable-cell fitting
  also handle it.
- "Only method above chance" (already banned in the URTC framing rules).
- Anything about new or unknown polymorphs. That is the job of CSP plus PXRD matching.
- "No open organic PXRD benchmark exists" without the list of places searched. We searched arXiv,
  OpenAlex, Crossref, Europe PMC and the open web; the Figshare API, NOMAD, MDF and ChemRxiv's own
  search were **not** searched.

## 7. Decisions needed from Frank

1. ~~Diffractometer access at Purdue~~ — answered 2026-10-10: not now; revisit after college apps + STS.
2. **OK to run Phase 0–1 now.** Free, CPU-only, existing checkpoint.
3. **GPU budget for Phase 3** (about $25 cap) once G0–G2 have read out.
4. **Keep v2 separate from pxrd-flow.** My recommendation is yes. The combined "molecular XtalNet"
   story stays as a later option.

## 8. Order of work

1. P0 baselines.
2. In parallel: P2 Tier A, the ibuprofen diagnosis, the CSD powder-study check, and drafting the
   Tier C emails.
3. P1 hybrid.
4. G0 / G1 / G2 readout, then go/no-go.
5. P3 retrain.
6. P4 cross-encoder.
7. P5 paper.

No calendar is attached on purpose. College applications (Nov 1) and NeurIPS camera-ready (Nov 9)
come first, and every phase is resumable.

## 9. Fact-check corrections

The original claim is kept next to each correction.

| Claim as first reported | Correction |
|---|---|
| ICEBERG 46% vs MIST 9.6–14.6% vs JESTR 11.85% (MassSpecGym) | Different challenges and candidate sets. Cite FraGNNet 46.64% vs MIST 14.64% from the benchmark's own tables. |
| JESTR hard negatives give +6–37% rank@1 | That covers 3 datasets. Across all datasets it is 0.51–37.05% (mean 5.72%), with about no gain on MassSpecGym. |
| XRDiff: "full-spectrum encoders collapse on real data" | Only without background/noise augmentation. With it, they are roughly equal to peak-based. The metric is a generative match rate. |
| VC-xPWDF "breaks under preferred orientation"; references at 12–200 K | d-mannitol was still ranked correctly (0.096–0.099 vs a 0.1 threshold). References run from 12 K to ambient, plus up to 3.1 GPa. |
| RealBind 82.8% vs 5.3% | The fair comparison is 82.8% vs 34.4% (corruption-trained). Inorganic data, single-author preprint. |
| pyxtal Similarity = de Gelder WCC with a 2° triangle | pyxtal defaults to a cosine weight. The 2° triangle default belongs to the CSD API/Mercury. |
| 57% of CSD entries are non-room-temperature (Bond 2021) | The figure is from the CCDC 2026 statistics (57.1%) and covers all entries. Bond 2021's expansion statistics are confirmed. |
| Szymanski 2021: JADE 71.4% | The paper prints 53/76 = 71.4%, but 53/76 = 69.7%. Cite the fraction. |
| VC-PWDF: 44,939 pairs, 2.84% vs 12.35% disagreement with COMPACK | Search-snippet only (the RSC page returned 403). Don't cite the numbers until read. |
| MS2Query mean Tanimoto 0.67 | It is 0.63. |

---

## 10. Gate G0 readout (2026-10-10)

> **Correction (2026-10-12, Section 12c):** 35 condition rows, but `zero_offset=0.0` repeats `clean`, so 34 are
> distinct. The verdict is unchanged.

Full run: 1,077 families / 2,835 queries, 35 conditions, every method scoring identical rendered
queries. Ran on a rented RTX 3090 box (~$0.25), destroyed after the pull. Table: `results/g0_table.md`.
Figure: `results/strong_baselines.png`. Raw: `results/sb_w{1..4}.json` + per-query ranks `.npz`.

**Verdict: the model loses to the best strong baseline in all 35 conditions, with non-overlapping
95 % family-bootstrap CIs.** The URTC robustness edge came entirely from weak baselines.
- **2θ zero offset:** Pearson with a zero-shift scan scores 1.000 at every offset; the model scores 0.52–0.56. The 0.88–0.91 dips at 0.05° and 0.15° are sub-grid artefacts of a 0.02° scan step, and interpolating would remove them.
- **Isotropic strain:** the ln q scan holds 0.86–0.91 out to 1.2 %; the model holds 0.55.
- **Broadening, texture and noise:** shift-scanned Pearson stays at 0.78–1.00.
- **Anisotropic thermal strain** (the realistic sim-to-real case) is where the gap is smallest. Every classical method decays with ΔT (WCC l = 2: 0.84 → 0.67 from 50 K to 200 K), while the model stays flat at about 0.55. The gap shrinks from −0.29 to −0.12.
- **Field-thermal harsh:** −0.12 (model 0.523 vs WCC 0.644). No crossover yet at the strains tested.
- **VC reimplementation (B3):** it underperforms the cheap baselines on these within-family sets: 0.73 clean, 0.64–0.66 under thermal strain. Even with the strain prior, its cell freedom lets siblings fit. Not quotable until it is validated against critic2.

**What this means for the plan:**
1. "The learned embedding is more robust than search-match" is dead for simulated queries. v2 must not claim it.
2. The model's bottleneck is **discrimination**: a clean ceiling of 0.56, against 1.0 for any profile comparison. Its invariance is real (it is the flattest curve on every axis), but that does not help when the starting point is this low.
3. **The speed argument is weaker than the plan assumed.** Shift-scanned Pearson/WCC costs about 100 µs per pair on CPU, so about 3 s per query over the 31k library, and plain Pearson over the full library is a single matrix multiply. Only the VC-class methods (tens of seconds per candidate) are slow enough for a learned shortlist to matter.
4. **Real data is now the deciding test.**
   - The one regime that trends the model's way is large anisotropic strain combined with a degraded profile, which is exactly what real low-temperature-CSD vs room-temperature-pattern queries look like.
   - Tier A data is in hand: **196 molecular-organic patterns in opXRD CNRS** (164 with H atoms, 32 with H omitted). About 100 are Cu Kα and the rest synchrotron, convertible through d-spacing. Extracted to `C:/Users/frank/xrd-clip-data/opxrd/opxrd_cnrs_organic.json`, outside git; CC BY 4.0.
   - **Next:** match these to CSD refcodes, then run model vs B1/B2 on real queries. The kill criterion in Section 5 applies directly. If the model's real recall is no better than WCC's, stop the learned-retrieval line and salvage the benchmark.

## 11. Real-data readout (2026-10-10): the kill criterion is met

> **Correction (2026-10-11, Section 12).** The labels behind this section were checked by packing
> similarity against each pattern's own opXRD structure. 12 of the 21 "other-origin" queries were
> labelled with the WRONG compound (the matcher preferred corpus entries, and an unrelated
> compound with the same element set and a similar cell won), 1 cannot be identified, and the 8
> correct ones are powder studies under a different refcode. The "other-origin split is the open
> problem (classical ~0.33)" reading below is therefore a labelling artefact, not a finding. The
> model-vs-classical verdict stands: the powder-study labels were 73/74 right. The original text
> is kept unchanged below.

**Test set.** 95 real measured patterns (92 distinct targets) from opXRD CNRS.
- All 196 organic patterns matched a CSD entry by primitive Niggli reduced cell (CCDC defaults: 1.5 % lengths, 2°) with an identical non-H element set; 95 of those entries are in the xrd-clip corpus.
- Library: the targets, every corpus polymorph-family member of a target, and 2,000 random corpus decoys (2,108 in total).
- A 20,000-decoy run was stopped by Claude Code's low-memory protection and was not rerun. More decoys can only lower every method's rank, so the decision does not depend on it.
- Files: `results/real_eval_table.md`, `results/real_eval_2k.json` (per-query ranks plus metadata). Scripts: `scripts/15_…` (cell reduction), `scripts/16_…` (CSD match), `scripts/17_real_eval.py`.

**Result** (background-subtracted variant; the raw variant is the same story):

| | top-1 | top-10 | within-family top-1 (n = 31) |
|---|---|---|---|
| model | **0.09** | 0.27 | 0.48 |
| Pearson (URTC baseline) | 0.78 | 0.81 | 0.94 |
| cosine + zero scan (B1, l = 0) | **0.81** | 0.86 | 1.00 |
| ln q scan (B2) | 0.81 | 0.85 | 0.94 |
| model top-1000 → ln q rerank | 0.77 | 0.82 | n/a |

- **The model is worse than every classical method,** including the URTC fixed-grid Pearson, even though **85 of 95 targets are in its training split.** It was trained on these very structures.
- **The hybrid only loses accuracy.** The model's shortlist drops targets that classical scoring would have found.
- **On the 21 harder queries** (structure not refined from this pattern): classical 0.29–0.38, model 0.05. That is the realistic case, and it has a lot of headroom for *any* method.
- Matches the literature: RealBind (correlation 82.8 % vs contrastive 34.4 % recall@10 on measured inorganic patterns) and our earlier ibuprofen spot check.

**Decision (Section 5 kill criterion):** stop the learned-retrieval line. G0 failed on simulated data and the model's real recall is far below WCC/cosine. Phases 3–4 (retrain, cross-encoder) are cancelled unless a new idea addresses the discrimination ceiling directly.

**What is worth salvaging:**
1. **An open real organic PXRD retrieval benchmark.** 196 opXRD (CC BY) patterns matched to CSD refcodes, split by powder-study vs other-origin structure. The "other" split is where classical scores fall to about 0.33, so that is the open problem. Refcodes plus a matching script are shareable; CSD structures are not.
2. **The strong-baseline study itself.** de Gelder WCC, ln q and VC on within-family polymorph sets, with anisotropic thermal strain. It is an honest, reusable result: simple shift-corrected profile matching is the bar any learned PXRD retrieval has to clear.
3. The 99 powder-study refcodes and the two unsent Tier C request drafts still serve item 1.

## 12. Benchmark hardening (2026-10-11): ground truth by packing similarity

Frank chose the negative-result paper (strong-baseline study + real organic test set) over a
benchmark-only release, parking, or a new learned method. Step 1 is making the real-data numbers
benchmark-grade before drafting.

**Ground truth (scripts/18_opxrd_truth_sets.py, CCDC interpreter).** opXRD ships each pattern's
labelled phase as a full P1 atom basis, so identity is checked directly instead of trusted from a
cell match.
1. Reduced-cell search (CCDC defaults) with the element-set filter and no cap on the hit list (the
   10-10 matcher kept 10, and 41 patterns had more).
2. Anchors: CCDC PackingSimilarity (15-molecule shell, H positions ignored) of the opXRD structure
   against every hit. Strict tier first. If nothing passes, a lenient tier ignores bond types and
   counts, H counts and the smallest components. Structures rebuilt from bare P1 coordinates
   perceive bonds differently from curated CSD molecules, so H-omitted refinements, salts and
   hydrates only match there (strict 154, lenient 35). "Allow molecular differences" is not used:
   it ran to 3.7 GB on one H-omitted pattern.
3. Same-form set: anchors against their whole refcode family, CSD vs CSD, strict. 15/15 = same
   form at any temperature or pressure (ibuprofen check: every redetermination from 25 K to 296 K
   is 15/15; form II 1–2/15; S-ibuprofen 5/15). Entries that cannot be compared are ignored, never
   scored as errors.

**Outcome.** 189/196 identified. The 7 that are not: 3 Si frameworks or silsesquioxanes that passed
the organic filter (NECTUN, ECOROH, RIXNES) and 4 others (INUFIH, OYALOR, ABEDUJ, IODYBZ01).
pattern_706 is excluded separately because its 2θ axis runs 0.01–0.83 (probably radians); no unit
guess is made. That leaves **188 queries**.
- **13 of the 10-10 labels named the wrong compound.** All 13 were corpus entries chosen over the
  true match.
- **185 of 189 patterns are the very data their CSD structure was refined from** (a powder-study
  anchor with RMSD < 0.05 Å against the opXRD structure). opXRD CNRS is a Rietveld collection, so
  "find the entry" is mostly self-retrieval.
- **The realistic hard test is single-crystal references only** ("sc": same-form powder-study
  entries are ignored, so the answer must be an independent single-crystal determination). **59
  patterns qualify**, against 21 (mostly mislabelled) "other-origin" queries before.

**Library.** All 308,862 cached corpus sticks, plus 218 simulated family members of the targets
(cache/sticks_extra; 246 extracted, 28 have no coordinates in the CSD) = 309,080. The stage-2
simulator gained a fallback for Z′ < 1 entries: the CSD writes the whole molecule and symmetry
regenerates half of it, so coincident sites are merged. It only fires when the strict parse fails,
so corpus simulation is unchanged. Check: with a zero-offset scan, merged anchors score like
normally parsed ones against their own patterns (median 0.72 vs 0.74).

**Eval (scripts/19_real_bench.py).** Exact ranks over the full library.
- A pre-pass scores every truth, distractor and ignored entry and fixes each query's threshold.
- Library chunks stream through 4 workers, which store one bit per (query, entry): "scores above
  the threshold". Library-size curves are popcounts over random subsets.
- Model shortlist → classical rerank (hybrid) is recorded alongside.
- Bootstrap CIs resample compounds (refcode stems).
- Smoke test (3,218-entry library, `--verify`): streamed ranks equal brute-force ranks for all
  11 methods × 2 truth definitions × 188 queries.
- Smoke-scale read (not quotable; small library): single-crystal split, lnq 0.62 and cos + zero
  scan 0.55 top-1 against the model's 0.02.

Full run launched 2026-10-11 (4 workers, ~5 h, resumable per chunk, log `cache/real_bench/run.log`)
→ `results/real_bench.json`.

### 12a. Full-library readout (2026-10-12)

Run: the classical chunks ran on a rented 128-core Vast box (render-only sticks, no atoms or
checkpoint; the CSD-derived upload was approved in manual mode). The model and pre-pass ran locally,
then a local merge. 113/113 chunks, md5-verified, box destroyed. Cost $2.53 (credit $10.09 -> $7.56): the job is
memory-bandwidth-bound, so 118 workers were worth about 2 laptops (Vast lesson #22).
Table: `results/real_bench_table.md`; numbers: `results/real_bench.json`.

188 patterns vs 309,080 entries, top-1 [95 % compound-bootstrap CI]:

| | std (n = 188) | single-crystal refs (n = 59, 30 compounds) | within-family (n = 51) |
|---|---|---|---|
| Pearson fixed grid [bkg] | 0.82 [0.75, 0.88] | 0.34 [0.15, 0.48] | 0.98 |
| cos + zero scan [bkg] | 0.72 [0.56, 0.88] | 0.37 [0.17, 0.58] | 0.98 |
| WCC l = 1° [bkg] | 0.68 [0.51, 0.85] | 0.44 [0.19, 0.71] | 0.96 |
| ln q + zero scan [bkg] | 0.72 [0.55, 0.90] | 0.44 [0.20, 0.71] | 0.96 |
| xrd-clip model [bkg] | 0.01 [0.00, 0.03] | 0.00 [0.00, 0.00] | 0.71 |
| model top-1000 → ln q | 0.26 | 0.07 | n/a |

Findings:
1. **The learned model is unusable at library scale.** Top-1 0.01, median rank ~6,000 (std);
   single-crystal 0.00, median rank ~15–19k. A model shortlist destroys the classical rerankers
   (0.26 vs 0.72). Within-family it still discriminates partly (0.71 vs 0.96–0.98).
2. **Background subtraction matters more than the choice of method.** WCC raw → bkg: 0.29 → 0.68
   (l = 1°) and 0.21 → 0.57 (l = 2°).
3. > **Correction (2026-10-12, Section 12b): wrong as stated.** All 20 GUSRIV wins (and all of EFOXED's 10)
   > are patterns of ONE query compound, ZZZUWK (20 of the 188 patterns). This is one confusable compound
   > counted 20 times, not a decoy that beats many compounds. Original text kept below.

   **Hubness drives the full-library drop.** cos + zero scan falls 0.86 → 0.72 between 100k decoys and
   the full library, while Pearson falls 0.86 → 0.82. One decoy (GUSRIV) is the top-1 winner in 20 of
   cos + zero scan's 52 misses; EFOXED in 10 of ln q's. Hubs are sparse patterns (~10 strong
   reflections, one dominant peak at 22–23°) that a ±0.5° zero scan aligns to many queries' strongest
   peak. A hubness correction is an open method point.
4. **Independent-reference retrieval is the hard, open case:** best top-1 ≈ 0.44, top-10 ≈ 0.5.
   Caveat: 59 patterns over 30 compounds, with ZZZIZM (11), NAPHTA (10) and MEHNIY (7) supplying
   28; report compound-weighted numbers too.
5. The library-size curve: every classical method loses 0.1–0.2 top-1 from 2k to 309k decoys. The
   10-10 2k-library numbers overstated classical accuracy at realistic scale.

### 12b. Hubness pass (2026-10-12)

Frank 10-12: venue = Digital Discovery; run the hubness fix on a Vast GPU before drafting.

**Method.** 19 kept only rank bits, so every classical score was recomputed in full on one RTX 4090
(box 55604630, destroyed; 761 s wall; $0.26, credit $7.56 → $7.30). The scorer is a torch port of
WCCLibrary/Pearson (`src/xrdclip/eval/wcc_torch.py`): the triangle weight is folded into the reference,
so each zero-scan shift is one matrix product. It agrees with the CPU scorer to ≤ 4e-6 and reproduces
19's uncorrected ranks with no top-1 change on any key: 118 rank mismatches (first written as 59, a miscount; corrected 10-12) over 9 classical keys × 2
truth sets, all |Δrank| ≤ 3, smallest affected rank 68 (float32 near-ties). Probes: 4000 clean
simulated + 4000 augmented-then-background-subtracted library patterns, drawn only from stems no query
belongs to; same-stem pairs are masked. The corrections and the primary (CSLS, augmented probes,
k = 10) were fixed in commit 2a2ad8a before any corrected number existed.
Code: `scripts/21_hubness.py` (prep / score / eval / check). Table: `results/real_bench_hub_table.md`.

Top-1, background-subtracted queries, 95 % compound-bootstrap CIs. Macro = mean over compounds.

| | per pattern, none → CSLS (n = 188) | macro, none → CSLS (123 compounds) | macro paired diff | single-crystal macro, none → CSLS (30) |
|---|---|---|---|---|
| Pearson fixed grid | 0.82 → 0.87 | 0.83 → 0.85 | +0.02 [−0.02, +0.06] | 0.30 → 0.32 |
| cos + zero scan | 0.72 → 0.88 | 0.84 → 0.88 | +0.04 [−0.01, +0.10] | 0.40 → 0.41 |
| WCC l = 1° | 0.68 → 0.70 | 0.83 → 0.84 | +0.01 [−0.02, +0.05] | 0.50 → 0.52 |
| WCC l = 2° | 0.57 → 0.63 | 0.71 → 0.74 | +0.03 [−0.01, +0.08] | 0.47 → 0.52 |
| ln q + zero scan | 0.72 → 0.76 | 0.87 → 0.89 | +0.02 [−0.02, +0.05] | 0.49 → 0.55 |
| xrd-clip model | 0.01 → 0.02 | 0.02 → 0.02 | +0.01 [0.00, +0.02] | 0.00 → 0.00 |

Findings:
1. **The full-library drop is one compound, not hubness.** ZZZUWK contributes 20 of 188 patterns. Every
   dominant wrong top-1 entry wins only ZZZUWK patterns: GUSRIV x20 (cos + zero scan), ZZZLWW01 x18
   (WCC l = 1°), SOSBEK/MITMOU x9 each (WCC l = 2°), EFOXED x10 and VOXSIO x5 (ln q). The same holds for the
   model (CPRDCA x18). ZZZUWK top-1, uncorrected: Pearson 12/20, every zero-scan method 0/20. The confusers
   share two strong low-angle reflections (GUSRIV: 23.45° and 28.56°; ZZZLWW01: 23.5° and 28.53°).
   Hypothesis, not tested: the ±0.5° zero scan lets such decoys line up with ZZZUWK's peaks, and the fixed
   grid does not.
   > **Tested 10-12 (scripts/24_case_zzzuwk.py, results/case_zzzuwk.json): mostly wrong.** ZZZUWK is
   > propan-1-aminium chloride (P4/nmm), a 303-398 K variable-temperature powder series, one pattern per CSD
   > entry ZZZUWK01-20. The main decoys need no shift: GUSRIV, SOSBEK and ZZZLWW01 beat the own entry at zero
   > offset in 20/20 patterns under WCC l = 1°, GUSRIV in 9/20 under cos. Only some (MITMOU -0.37°, CIQBIO and
   > VOXSIO -0.2°) need the scan. The causes are (1) a sparse pattern: 24 reflections in 5-50° vs a median of
   > 130 for the benchmark's entries (fewer than 99 % of them); (2) texture: measured (001) at 12.0° is 0.36 of
   > the maximum vs 0.07 in the simulated own entry, (002) 0.13 vs 0.03, (110) 0.11 vs 0.53, consistent with
   > plates lying on c. So the own entry matches poorly (WCC l = 1° median 0.79 vs 0.97 for the other patterns),
   > and the more peak-tolerant the score, the more decoys pass it (1 decoy for cos, 7 for WCC l = 1°).
   > **Corrected 10-12 (Section 12c): (2) is wrong -- the ratios are not monotonic in the angle to c, so not texture;
   > the own entries are disordered and simulated through a fallback. Intensity mismatch, origin unresolved.**
2. **Hubness in the probe population is real but depends on the method.** N10 skew with augmented probes:
   cos + zero scan 18.7, ln q 12.7, Pearson 8.0, WCC 3.6 (both l). The triangle weighting of WCC resists hubs.
3. **CSLS removes the ZZZUWK confusion for cos + zero scan** (0/20 → 20/20). Per pattern that is 0.72 → 0.88
   (+0.15 [+0.01, +0.34]), and the library curve flattens (0.90 at 100k decoys, 0.88 at the full library).
   Compound-weighted, the gain is small for every classical method (+0.01 to +0.04) and no CI excludes 0.
   Fixed:broken patterns = 34:5 (cos), 11:2 (Pearson), 6:2 (WCC 1°), 12:2 (WCC 2°), 9:2 (ln q).
   With WCC and ln q, CSLS only swaps which decoy wins ZZZUWK (CUMCEV x20; CIQBIO x17).
4. **The probe domain matters.** Clean simulated probes give cos 0.79 per pattern (augmented: 0.88).
   Library-side z-norm hurts WCC l = 2° (0.57 → 0.36). Mutual proximity is neutral to slightly positive.
   No setting changes the conclusions.
5. **Independent references are not a hubness problem.** Single-crystal macro top-1 is 0.30–0.50 uncorrected and 0.32–0.55 with CSLS; no paired CI excludes 0.
   The strain-tolerant methods lead there (ln q, WCC ≈ 0.5 vs Pearson 0.30), as thermal-expansion
   differences between determinations would predict.
6. **The model's failure is not hubness:** 0.01 → 0.02 with CSLS.
7. **Report compound-weighted numbers first.** Per-pattern averages mostly measure ZZZUWK (20), ZZZIZM (12)
   and NAPHTA (10). Uncorrected macro order: ln q 0.87, cos 0.84, Pearson 0.83, WCC 1° 0.83, WCC 2° 0.71,
   model 0.02. The per-pattern lead of Pearson (0.82) comes from ZZZUWK.

Paper consequences (for the outline check-in): headline = macro top-1 with per-pattern secondary;
"hubness drives the drop" is out; CSLS is a cheap optional add-on with a small effect; ZZZUWK becomes a
case study in what shift tolerance costs.

### 12c. Draft and internal review (2026-10-12)

Draft v1 of the Digital Discovery paper is `paper/dd/manuscript.md` (+ generated ESI, figures, `build.sh` -> .docx);
every number is checked by `scripts/verify_claims.py` against `results/paper_numbers.json`. Two internal referee
passes (powder diffraction, ML benchmarking) changed these readings of the data:

1. **The model's failure is the measured-to-simulated step, not discrimination** (`scripts/26_model_probe_check.py`).
   Clean simulated probes vs all 309,080 entries: top-1 0.42, top-10 0.84, median rank 2; degraded probes 0.28 /
   0.67 / 4; clean renderings of the benchmark's own entries with the measured answer sets: 0.21 / 0.56 / 5; the
   measured patterns themselves: 0.01 / 0.03 / ~6,000.
2. **Shift/strain tolerance pays only for independent single-crystal references**: paired compound bootstrap,
   ln q - Pearson +0.18 [0.02, 0.35], WCC 1° - Pearson +0.20 [0.03, 0.36]; on the any-same-form set no score
   differs from fixed-grid Pearson except WCC 2° (-0.12). Single-crystal failures track cell agreement: largest
   cell-length difference <= 1 %: ln q top-1 0.71 (34 patterns); > 1 %: 0.08 (25). Recorded temperatures differ by a
   median of 3 K, so thermal expansion explains few.
3. **ZZZUWK is not texture** (correction to 12b finding 1's 10-12 note): the measured/simulated intensity ratios are
   not monotonic in the angle to c. All 20 own entries are disordered, H-omitted, lenient-tier. Disordered own
   entries (28 patterns, 9 compounds) score 0.56-0.78 macro vs 0.71-0.88 for the rest.
4. **G0 has 34 distinct conditions**, not 35 (clean == zero_offset 0.0). Every strong score alone beats the model:
   WCC 0.5/1/2 and ln q in 34/34, Pearson + zero scan in 33/34.
5. **Ground truth holds**: of 454 wrong top-1 entries, 1 is the same form under another refcode (GOFWUV,
   pattern_100/QQQAUJ) (`scripts/27_wrong_top1_audit.py`).
6. **ln q implementation flaw**: its 16 zero offsets skip 0 and its axis is coarse at high angle; it handicaps ln q.

Open revision options R1-R6 (cost + scope) are listed in tasks/todo.md Step 2; none started without Frank.

## 13. Revision R1-R5 (Frank 10-12: "Do 1-5")

Plan, pre-declared analyses and predictions: tasks/todo.md Step 3 (commit 4d4dae3, before any number).
Code: scripts/28_rescore.py (R1, R2), 29_critic2_rerank.py (R3), 30_model_gap.py (R4a/b), 31/32 (R5).

### 13a. R1 ln q fix (2026-10-12)

v2 = offset grid -0.32..+0.32 in 0.04 steps (17 offsets, contains 0; v1's 16 skipped it) + bin-averaged
resampling onto the ln q axis. Checks: tests/test_strong_baselines.py (self-score 1.0 at offset 0, 1 % strain
found, torch == numpy < 1e-5, peak area kept). scripts/28 tag v1 reproduces 19's smoke ranks exactly (every key,
model and hybrid included). Full library, 188 queries, local CPU 31 min (results/rescore_r1.json):

| ln q + zero scan | v1 | v2 | paired diff (compound bootstrap) |
|---|---|---|---|
| any same form, macro top-1 (123 cpds) | 0.87 | 0.87 | 0.00 [0.00, 0.00] |
| single-crystal, macro top-1 (30 cpds) | 0.49 | 0.45 | -0.03 [-0.10, 0.00] |
| single-crystal, macro top-10 | 0.59 | 0.61 | +0.02 [0.00, +0.05] |

One top-1 changes (VELWIV, single-crystal set, lost). 38/188 std ranks move, mostly down by a few places.
Prediction ("real data moves little, within +-0.05") held. v2 is the pre-declared primary; v1 stays in the ESI.
Found on the way: the G0 sweep seeded its draws with Python's salted `hash()`, so the 10-10 draws cannot be
reproduced; scripts/12 now seeds with zlib.crc32 and the sweep is rerun locally (results/strong_baselines_v2.json,
both ln q versions on the same draws).

### 13b. R2 measured-aware library and comparison (2026-10-12)

Package (pre-declared): overall Debye-Waller B = 4.45 A^2 (8 pi^2 x median non-H U_eq of 5,000 library CIFs at
283-303 K, results/r2_bvalue.json); Lorentz-polarisation conversion of the 68 non-Cu queries; K-alpha2 scanned as a
nuisance for the 125 lab-tube queries (max over K-alpha1 and K-alpha1 + K-alpha2 libraries); reference norms over
each query's measured range (46 partial-coverage queries). Full library, local CPU 117 min
(results/rescore_r2.json). Macro top-1 vs v1, paired compound bootstrap:

| score | any same form | diff | single-crystal | diff |
|---|---|---|---|---|
| Pearson, fixed grid | 0.81 | -0.02 [-0.07, +0.02] | 0.27 | -0.04 [-0.11, 0.00] |
| cosine + zero scan | 0.88 | +0.04 [+0.01, +0.08] | 0.44 | +0.04 [-0.01, +0.13] |
| WCC l = 1 deg | 0.81 | -0.02 [-0.05, +0.02] | 0.52 | +0.02 [-0.05, +0.10] |
| WCC l = 2 deg | 0.69 | -0.02 [-0.07, +0.03] | 0.50 | +0.03 [0.00, +0.10] |
| ln q v2 | 0.86 | -0.02 [-0.05, +0.02] | 0.49 | 0.00 [-0.13, +0.13] |

**Prediction mostly wrong:** I predicted every classical single-crystal score would rise. They move -0.04 to
+0.04 and only cosine (any form) clears zero. The model gap is unchanged (model 0.00-0.02). Reading: the
classical numbers are robust to these library-fidelity choices; the single-crystal failures are not a
simulation-fidelity problem. Non-Cu single-crystal patterns (25, mostly synchrotron temperature series) stay at
top-1 0.04-0.24 under every variant. One-off ablations run on the GPU box (R2 package minus one switch).

R2 ablations (box A, 10-13; paired macro top-1 of "package minus one switch" against the full package): only two
clear zero. Without K-alpha2, cosine (any form) -0.03 [-0.07, -0.01], so the K-alpha2 nuisance is what lifts cosine;
without the measured-range mask, WCC l = 2 deg (any form) +0.05 [+0.02, +0.09], so the mask costs it. B and LP are
null everywhere. Against the main benchmark (ln q v2; 28's own paired_vs_base is against v1): -0.04 to +0.04, only
cosine (any form) +0.04 [+0.01, +0.08] resolved. Manuscript Section 4.6 + Table 2, ESI S12.

### 13c. R3 critic2 VC-GPWDF re-rank (2026-10-13/14)

critic2 1.4.155 COMPAREVC GLOBAL QUICK ALPHA 1 on the ln q v2 top 50 of the 59 sc patterns (2,950 comparisons,
4-h cap, failures = DIFF 1). K: declared 100; ln q v2 recall is 36/59 at both 50 and 100, so K = 50 (only
competitors dropped). A two-point cost extrapolation first cut K to 10 (lesson 10-13); the K = 10 run measured 76
CPU-h for 589 comparisons, so K went back to 50 on the same rented 256-core box (Vast 55845149, ~$4.4, destroyed).
Cost: median 2.2 CPU-min, max 2.96 CPU-h, 400 CPU-h; 1 capped (pattern_730/GOGZOU, answer not in its list).
results/critic2_rerank.json:

| sc set (30 compounds) | critic2 re-rank | ln q v2 | paired |
|---|---|---|---|
| macro top-1 | 0.50 [0.32, 0.67] | 0.45 [0.28, 0.63] | +0.05 [-0.11, +0.20] |
| macro top-10 | 0.59 | 0.61 | -0.02 [-0.15, +0.09] |
| per pattern top-1 | 26/59 | 25/59 | |
| cells differ > 1 % (25) | 4 | 2 | |
| cells within 1 % (34) | 22 | 23 | |

Gains 4 (ln q ranks 2, 3, 7, 9 -> 1), losses 3 (-> 2, 22, 50; for pattern_1006 ln q's correct first choice gets
the worst DIFF of its 50). K = 10 on all 59: 0.533, paired +0.08 [-0.05, +0.21]. **Prediction partly right:** it
lifts the > 1 % group (2 -> 4 of 25) but is capped by recall (23/59 answers outside the list); no resolved gain.

### 13d. R4 learned side (2026-10-12/13)

R4a effect ladder (results/model_gap.json), compound-weighted top-1 / median rank, any same form: own clean
0.27 / 5; single effects 0.18-0.27 (K-alpha2 0.18 / 33 and background 0.24 / 130 the costly ones); all six 0.15 /
176; + B 0.05 / 744; + B + jitter 0.04 / 746; measured 0.02 / 5,988. Fixed-grid Pearson's ladder ends at 0.83 =
its measured 0.83. R4b pattern-to-pattern (measured vs clean library patterns in the pattern space): 0.13 / 1,622.
R4c realism retrain (design + prediction fixed in 3b77818 before training; best val 0.911): own entries all six
+ B 0.05 -> 0.23 (median 744 -> 34), K-alpha2 alone 0.18 -> 0.11; measured 0.02 [0, 0.06] vs 0.02 [0, 0.04],
sc 0.00, median 5,988 -> 2,981; p2p 0.21 / 267. **Prediction wrong** (per-pattern measured top-1 0.05-0.20
predicted; 0.016), so by the pre-declared branch the remaining gap is not the modelled effects. Manuscript
Section 4.5 + Fig. 6, ESI S11.

### 13e. R5 open replica on the COD (2026-10-13)

COD library 108,716 selected + 854 query candidates = 109,570, 108,902 simulated; truth 191/196 identified (189
as in the CSD + 2 COD-only); 2,670 other-compound candidates kept as library entries, 23 same-formula unknowns
set aside. 190 patterns scored (results/rescore_main_cod.json). Any same form (125 compounds): classical
0.77-0.90 (ln q 0.90 [0.85, 0.95]), model 0.02 [0, 0.04]. Single crystal: 35 patterns but 9 compounds; classical
0.26-0.58, model 0. **Prediction held** (same ordering, classical >> model). Manuscript Section 4.7 + Table 2,
ESI S13. Draft v2 (all five folded in): commits 2803c54, 7a4c833 and the ESI commit; verify_claims 169/169.

### 13f. Internal referee pass on draft v2 (2026-10-14)

Two independent referees (powder diffraction, ML/benchmarking). Both: the headline (classical >> model on measured
organic patterns) stands; the new single-crystal and learned-side readings were overstated. Verified and fixed:
(1) the cell-mismatch metric used sorted conventional lengths (setting-dependent; DEHKUA 21 % -> 3.8 % reduced) and
could take an sc anchor as the "own" entry; now primitive Niggli cells, own = powder-study anchor, dT for the same
entry. (2) 22 of the 59 sc patterns (5 compounds: NAPHTA, MEHNIY, EPIVEF, HOJQII, LILRIJ) are room-T compression
series (CSD pressure fields + publications; scripts/34); with dT > 50 K and one supercell they explain 24 of the
25 cells-differ patterns. "Temperature explains few" was wrong. (3) critic2: the ROHWIX "loss" was a disorder
input bug (CIFs carry no occupancies); major-component inputs give DIFF 0.006 -> re-rank 0.53 vs ln q 0.45 (+0.08
[-0.05, +0.21]). Positive control: own Rietveld entry beats every wrong shortlist entry for 29/36 ambient but 1/22
pressure-series patterns (peak fits 11-49 vs 66-296 peaks): the automated pipeline fails on the DAC data.
Injection: 2 of 23 missing answers would rank first. Ambient-only (secondary, post hoc): re-rank 0.64 vs ln q 0.54.
(4) Ladder: the Debye-Waller factor alone is the model's largest single cost (0.27 -> 0.07; Pearson 1.00); the
retrain fixes it (0.34). Background-everywhere rerun: rungs move <= 0.05, Kα2 result holds. (5) R4c prediction now
reported in full (top-1 AND median-rank parts failed); conclusion softened to "differences our simulation does not
reproduce", one seed / clean-val checkpoint caveats. (6) COD sc set = 20 ZZZUWK (1950 structure the CSD sets aside)
+ 8 NAPHTA pressure + 7 singletons: cannot test the CSD sc result; the replica now claims the any-same-form result
only. Open for Frank: headline sc set (pre-declared 59 kept primary; ambient split secondary), further compute
(DAC peak refits, zero-offset sensitivity, bin-averaged Cu-grid conversion for all scores).

### 13g. Frank's extra checks X1-X3 (2026-10-14/15; designs + predictions in tasks/todo.md C8, fixed before any number)

Frank 10-14: keep all 59 sc patterns as the headline; run all three checks. All three are sensitivity checks; none
changes a headline number.
- X1 bin-averaged Cu grid: classical macro top-1 moves 0.00 to +0.03 (one first place at most), model -0.004;
  below the pre-declared 0.05 rule (Section 4.6, ESI S21b). Prediction (+-0.02) held except sc WCC 2 deg (+0.03).
- X2 answer-blind zero offset for the critic2 re-rank (36 answer-holding shortlists, 1,800 comparisons): paired top-1
  0.00 [0.00, 0.00], no first place changes, 4 answers move down. Predicted lab-Cu median |delta| <= 0.03 deg FAILED
  (0.07 deg); predicted top-1 change within +-0.05 HELD.
- X3 (amended before the grid run) default-setting refits of the 10 pressure-series patterns that begin with
  zero-intensity points: peaks 11-29 -> 40-60, own entry first 1 -> 7 of 10, injected answer first 1 -> 5 of 9, but
  paired top-1 0.00 [0.00, 0.00] because only one of the ten shortlists holds the answer (rank 22 -> 2). So the DAC
  "fit failure" of 13c was mostly zero padding for these 10; the shortlist, not the fit, now bounds them. The other 12
  pressure-series patterns (NAPHTA, HOJQII) have no padding and still fail the control (0 of 12). Both predictions
  (control improves for a majority; top-1 rises <= 0.03) HELD.
- Run note: X2 and X3 ran side by side and collided on pattern_594's work files (lessons.md 10-14); its 100 pairs
  were rerun and recovered from the stopped box through a relay. Manuscript Sections 3.8, 4.2, Discussion and
  Limitations, ESI S18c/S18d; verify 196/196.
