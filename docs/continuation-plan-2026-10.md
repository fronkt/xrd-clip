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

**Tier B (Frank decides; needs a diffractometer):**
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

1. **Diffractometer access at Purdue** for Tier B. This is the biggest lever on G2.
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
