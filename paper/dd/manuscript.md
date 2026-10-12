---
title: "Shift-tolerant profile matching is the bar for learned powder-diffraction retrieval: a real organic benchmark against 309,080 CSD entries"
bibliography: refs_dd.bib
csl: royal-society-of-chemistry.csl
link-citations: true
---

<!-- DRAFT 2026-10-12. Every number in this file is checked by scripts/verify_claims.py against
results/paper_numbers.json. Citation keys resolve against refs_dd.bib, built from
docs/lit-2026-10/refs_dd_verified.md. Items marked [FRANK] need his decision before submission. -->

Frank Cai^a^

^a^ Purdue University, West Lafayette, Indiana, USA. E-mail: cai485@purdue.edu. ORCID: 0009-0003-0041-1459
<!-- [FRANK] Purdue e-mail kept for the Purdue-RSC open-access agreement (runs to 31 Dec 2026, applied at
acceptance); see tasks/todo.md Step 2. -->

## Abstract

Contrastive models that place powder X-ray diffraction (PXRD) patterns and crystal structures in one
embedding space are proposed as fast, shift-robust alternatives to search–match, but they are tested mostly
on simulated queries and against baselines that cannot absorb a peak shift. We test one such model, trained
on 247,089 molecular crystals from the Cambridge Structural Database (CSD), against classical profile
matching with a 2θ zero-offset scan: cosine and Pearson correlation, de Gelder weighted cross-correlation
(WCC), and a ln q lag scan for isotropic strain. On simulated queries the model loses to the best classical
score in all 35 degradation conditions tested, by 0.12 to 0.48 in within-family top-1. We then build a real
benchmark: 188 measured patterns of 123 organic compounds from the open opXRD database, with the correct
answers fixed by CCDC packing similarity and ranked against 309,080 simulated CSD entries. Compound-weighted
top-1 is 0.71–0.87 for the classical scores and 0.02 for the model, whose median rank for the correct entry
is about 6,000. When only independent single-crystal determinations count as correct, the classical scores
fall to 0.30–0.50 (30 compounds); this is the open problem. A hubness correction (CSLS) changes
compound-weighted top-1 by +0.01 to +0.04, within noise. We release the benchmark as code and refcode
lists.

## 1. Introduction

Powder X-ray diffraction is the routine way to identify which crystalline phase, and for molecular
materials which polymorph, a sample contains. The standard approach is search–match: compare the measured
pattern with reference patterns, measured or simulated, from a database such as the Powder Diffraction
File [@gatesrector2019pdf] or the Cambridge Structural Database (CSD) [@groom2016csd]. For organic
crystals the polymorph matters as much as the compound. Different forms of one drug differ in solubility
and stability, and an unexpected form can take a product off the market [@bauer2001ritonavir;
@cruzcabeza2015polymorph].

Machine learning offers another route. Contrastive models train a pattern encoder and a structure encoder
so that a pattern lands near the structure that produced it, after which retrieval is a nearest-neighbour
search [@lai2025xtalnet; @xu2025xccp; @pxtal2026]. Generative models go further and build a structure from
the pattern [@johansen2025decifer]. These models are trained on simulated patterns, and most are evaluated
on simulated patterns of inorganic or framework materials. Where the comparison has been made on measured
data, simple methods have held up. Plain correlation beat a contrastive model on measured inorganic
patterns [@realbind2026]; de Gelder's weighted cross-correlation matched a Siamese network on experimental
patterns of pharmaceutical polymorphs [@smolnet2024]; and a non-learned peak-feature index beat a commercial
search–match program on measured patterns [@matdiffract2026]. The same pattern appears in other fields:
simulate-then-compare methods outperform learned joint embeddings for molecule retrieval from tandem mass
spectra [@bushuiev2024massspecgym], and careful baselines erased much of the reported progress in deep
metric learning [@musgrave2020reality] and neural recommendation [@dacrema2019progress].

We previously trained a contrastive model of this kind on molecular crystals from the CSD
(xrd-clip) [@cai2026urtc]. Within families of polymorphs it was more robust to a 2θ zero offset and to
lattice strain than fixed-grid Pearson correlation and a peak-list match. Neither of those baselines can
absorb a peak shift, so the comparison says little about the methods crystallographers actually use. A
2θ zero-offset scan, de Gelder's triangle-weighted cross-correlation [@degelder2001wcc] and a lag scan on a
ln q axis each absorb a shift or a strain at almost no cost, and variable-cell methods go further
[@oterodelaroza2024vcpwdf].

This paper asks whether the learned model beats these methods, first on simulated queries and then on
measured ones, and answers no on both. We make four contributions.

1. A simulated stress test of the model against shift-tolerant classical scores over 35 degradation
   conditions (Section 4.1).
2. An open real benchmark for organic PXRD retrieval: 188 measured opXRD patterns [@opxrd2025] matched to
   the CSD by packing similarity [@chisholm2005compack], with two definitions of the correct answer and a
   library of 309,080 simulated entries (Sections 2 and 4.2–4.3). It is released as code and refcode
   lists, since CSD structures cannot be redistributed.
3. A check of whether hubness, the tendency of a few library entries to be the nearest neighbour of many
   queries [@radovanovic2010hubs], limits classical retrieval at this library size, with a pre-declared
   correction [@conneau2018csls] (Sections 3.5 and 4.4).
4. An analysis of where the learned model fails (Section 4.5) and a list of what a learned method should be
   compared against (Section 5).

## 2. Benchmark

### 2.1 Measured patterns

The opXRD database collects experimental powder patterns with their labelled phases under a CC BY 4.0
licence [@opxrd2025]. Its CNRS contribution holds 1,052 patterns from Rietveld studies, each with the
refined structure as a full atom list. We kept single-phase patterns whose phase contains C and H and no
element outside {C, H, N, O, S, P, F, Cl, Br, I, B, Si}, which left 196 patterns. Of the 188 patterns
used in the end (Section 2.2), 120 were measured with Cu Kα radiation and 68 at other wavelengths between
0.34 and 1.79 Å, mostly synchrotron. For 3 of the 120 opXRD records no wavelength, and we assumed Cu Kα. Every pattern was converted to Cu Kα1 2θ through its d-spacings and
interpolated onto the grid used throughout this work: 5–50° 2θ in 0.02° steps (2,251 points). In 31 of
the 188 refined structures the H atoms are omitted.

### 2.2 Correct answers by packing similarity

A cell match is not enough to identify the compound. In an earlier version of this benchmark, a matcher
that compared reduced cells and element sets assigned 13 patterns to the wrong compound, each time an
unrelated entry with a similar cell. We therefore identified every pattern from its refined structure
(Fig. 1).

1. **Candidates.** A CSD reduced-cell search on the opXRD structure's primitive Niggli cell (1.5 % in
   lengths and 2° in angles, the CCDC defaults), keeping hits with the same non-H element set and with no
   cap on the number of hits.
2. **Anchors.** CCDC packing similarity [@chisholm2005compack] between the opXRD structure and each
   candidate, over a 15-molecule shell with H positions ignored. A candidate that matches 15 of 15
   molecules is an anchor. Structures rebuilt from bare coordinates sometimes perceive bonds differently
   from curated CSD molecules, so when no candidate passes, a lenient pass ignores bond types, H counts and
   the smallest components. 154 patterns were anchored in the strict pass and 35 in the lenient pass.
3. **Same-form set.** Each anchor compared with every entry of its CSD refcode family, CSD against CSD
   with the strict settings. An entry that matches 15 of 15 molecules is the same solid form, whatever its
   temperature or pressure; entries that cannot be compared are set aside and never scored as errors.

![](figs/fig1_benchmark.pdf)

**Fig. 1** Construction of the benchmark. Top: each measured pattern is identified from its own Rietveld
structure. CSD candidates from a reduced-cell search are compared with that structure by packing
similarity, and the matching anchor entry is compared with its refcode family to collect every entry of
the same solid form. Bottom: the measured pattern is scored against all 309,080 simulated library
patterns, and the rank of the first correct entry is recorded under two answer sets.

189 of the 196 patterns were identified. The other seven are three Si frameworks or silsesquioxanes
that passed the element filter and four patterns with no packing match. One identified pattern was excluded because its
2θ axis runs from 0.01 to 0.83, probably radians. That leaves **188 patterns of 123 compounds**. Most
compounds have one pattern (110 of 123), but a few have many: ZZZUWK has 20 (a variable-temperature
series), ZZZIZM 12 and NAPHTA 10.

### 2.3 Two definitions of the correct answer

opXRD's CNRS patterns are Rietveld data, and 184 of the 188 patterns are the very data that their CSD
entry was refined against. Finding that entry tests whether a method can read a measured pattern, but it
is close to self-retrieval. We therefore score two answer sets.

- **Any same-form entry** (188 patterns, 123 compounds): any entry in the pattern's same-form set is
  correct.
- **Single-crystal entry only** (59 patterns, 30 compounds): same-form entries that are powder studies
  are set aside, so the answer must be an independent single-crystal determination, often at a different
  temperature. This is the realistic case of a new measurement checked against a database.

We also report **within-family** retrieval, the rank of the correct form among the compound's own
polymorphs in the library (51 patterns of 35 compounds), which is the polymorph-identification question in
isolation.

### 2.4 Library

The library holds 309,080 simulated patterns: the 308,862 molecular crystals of the xrd-clip corpus
(organic CSD entries without disorder, R ≤ 7.5 %, Z′ ≤ 2, after removing duplicate redeterminations)
[@cai2026urtc] and 218 same-form or polymorph entries of the benchmark compounds that the corpus filter had
removed. Of 246 such entries, 28 have no coordinates in the CSD. Stick patterns (Cu Kα1) were computed with
pymatgen [@ong2013pymatgen] and rendered with a pseudo-Voigt profile (Caglioti U = 0.002, V = −0.001,
W = 0.004 deg², η = 0.5) without texture, background or noise.

### 2.5 Metrics

For each pattern and score we compute the exact rank of the first correct entry over the full library:
the number of entries that are neither correct nor set aside and that score higher than the best correct
entry. Top-k is the fraction of patterns with rank below k. Because some compounds contribute many
patterns, per-pattern averages mostly measure those compounds, so our headline is the **compound-weighted**
average: top-k is computed per compound and then averaged over compounds. Confidence intervals (95 %) come
from 2,000 bootstrap resamples of compounds [@field2007cluster], so patterns of one compound move together.
Per-pattern numbers are given in Table 1 and the ESI.†

## 3. Methods

### 3.1 Preprocessing

The measured patterns were scored in two forms: *raw*, shifted to a zero minimum and scaled to a unit
maximum, and *background-subtracted*, in which a rolling minimum over 2° smoothed by a 2° moving average is
removed before scaling. The Kα2 component was left in. Library patterns are scored as simulated.

### 3.2 Classical scores

- **Pearson, fixed grid.** The Pearson correlation of the two patterns on the common grid. This was the
  main baseline of our earlier work [@cai2026urtc].
- **Zero scan.** Each of the scores below is maximised over a 2θ zero offset of ±0.5° in 0.02° steps,
  which absorbs a sample-displacement or zero error.
- **WCC.** de Gelder's weighted cross-correlation [@degelder2001wcc] with a triangular weight of
  half-width l = 1° or 2°, which tolerates peak shifts smaller than l without a scan, plus the zero scan.
  With l = 0 the weight is a delta and WCC reduces to the cosine of the two patterns; we call this *cosine
  + zero scan*. In the simulated sweep (Section 4.1) the l = 0 variant was applied to standardised
  patterns, which makes it Pearson + zero scan.
- **ln q lag scan.** Both patterns are resampled onto a ln q axis, on which an isotropic lattice strain is
  a pure translation. The score is a WCC with l = 0.01 in ln q, maximised over lags up to ±3 % strain and
  over 16 zero offsets from −0.3° to +0.3° applied before resampling.

These are the scores in the strong-baseline ladder we set before the experiments; a variable-cell method
[@oterodelaroza2024vcpwdf] was also tried in the simulated sweep but is not reported, because our
reimplementation has not been validated against the reference code.

### 3.3 Exact scores on a GPU

The full-library ranks were first computed on CPUs, with fast Fourier transform cross-correlations and
one bit per pattern–entry pair stored. The hubness analysis (Section 3.5) needs every score, not only
ranks, so we ported the scores to PyTorch [@paszke2019pytorch]. Folding the triangle weight into the
reference turns every zero offset into one matrix product: for weights w_k and shift d,
Σ_k w_k Σ_x q(x) r(x + d + k) = Σ_x q(x) r_w(x + d), with r_w the reference convolved with w. TF32 matrix
arithmetic was disabled, because its rounding reorders near-ties. On one RTX 4090 the scores of 8,188
rows (benchmark patterns and probes) against 309,080 entries for five scores took 761 s. The GPU scores
agree with the CPU scores to 4 × 10⁻⁶ and reproduce the CPU ranks with no top-1 change. Over nine score
variants and both answer sets, 118 rank values differ, all by at most 3 positions and none at a rank
better than 68, which is float32 rounding between near-tied entries.

### 3.4 Learned model

The model is the xrd-clip checkpoint described in ref. [@cai2026urtc]. A one-dimensional residual
convolutional network encodes the pattern on the grid above; a Transformer over atoms with a lattice token
encodes the structure; both map to a 256-dimensional unit vector. It was trained with a symmetric InfoNCE
loss [@oord2018cpc; @radford2021clip] on 247,089 CSD structures of a scaffold- and polymorph-family split,
on patterns simulated on the fly with random zero offset, isotropic strain, Caglioti and crystallite-size
broadening [@caglioti1958], March–Dollase texture [@dollase1986], background and Poisson noise. A measured
pattern is scored against the library by the cosine of its embedding with each entry's structure
embedding. We also test it as a shortlist: the model's top 100 or 1,000 entries re-ranked by a classical
score.

### 3.5 Hubness diagnosis and correction

A *hub* is a library entry that scores highly against many unrelated queries. We measured hubness
with probe patterns rather than the benchmark patterns, so that nothing was tuned on the test set: 4,000
library entries simulated cleanly and 4,000 simulated with the training-time degradations and then
background-subtracted like the measured patterns, drawn only from compounds outside the benchmark, with
pairs from the same compound excluded. For each score we report the skewness of N₁₀, the number of probes
that have an entry among their ten best matches [@radovanovic2010hubs].

The correction was fixed before any corrected number was computed (repository commit 2a2ad8a):
cross-domain similarity local scaling (CSLS) [@conneau2018csls] on the library side,
s′(q, r) = s(q, r) − h₁₀(r)/2, where h₁₀(r) is the mean of entry r's ten highest scores against the
degraded probes. The query-side term of CSLS is the same for every entry and does not change ranks. As
sensitivity checks we report k = 1–100, clean probes, library-side z-normalisation and mutual proximity
[@schnitzer2012mp] (ESI†); none was selected after seeing the results.

### 3.6 Simulated sweep

The sweep reuses the within-family test of ref. [@cai2026urtc]: 1,077 held-out polymorph families with
2,835 query structures, where the query is one form's simulated and degraded pattern and the candidates
are the clean patterns of the family's forms (composition-only chance 0.38). Each condition varies one
degradation: zero offset up to 0.3°, isotropic strain up to 1.2 %, crystallite size down to 10 nm,
March–Dollase r down to 0.5 and counting noise; or an anisotropic thermal strain, with per-axis linear
expansion coefficients drawn around 55 × 10⁻⁶ K⁻¹ (after the statistics of ref. [@bond2021thermal]) for a
temperature change of 50–200 K; or combinations of these. Every method scores the same rendered queries,
and CIs come from a bootstrap over families.

## 4. Results

### 4.1 Simulated queries: the model loses in every condition

Across the 35 conditions the model scores 0.47–0.56 within-family top-1 and the best classical score
0.64–1.00 (Fig. 2). The model loses every time, by 0.12 to 0.48, and the 95 % CIs never overlap.

The fixed-grid Pearson baseline of our earlier work behaves as reported there: it falls to chance by a
0.15° zero offset or a 0.8 % strain, where the model is flat (Fig. 2a and b). The zero scan removes that
weakness: Pearson + zero scan scores 1.00 at every offset that is a multiple of the 0.02° step (the dips at
0.05° and 0.15° are offsets that fall between grid points, which the scan cannot reach). The ln q scan
holds 0.86 at 1.2 % isotropic strain. The model's advantage is that its curves are flat, but they are flat
at about 0.55: its clean within-family top-1 is 0.56, where any profile score reaches 1.00. The smallest
gap, −0.12, comes from large anisotropic thermal strain combined with a degraded profile, the condition
closest to comparing a room-temperature pattern with a low-temperature CSD structure. Even there, WCC
(l = 1°) leads, 0.64 against 0.52.

![](figs/fig2_simulated_sweep.pdf)

**Fig. 2** Simulated stress test: within-family top-1 over 1,077 polymorph families (2,835 queries).
Bands are 95 % family-bootstrap CIs; the dashed line is composition-only chance (0.38). (a) 2θ zero
offset, (b) isotropic lattice strain, (c) anisotropic thermal strain, (d) crystallite size,
(e) March–Dollase preferred orientation, (f) counting statistics. The dips of the zero-scan scores at
0.05° and 0.15° in (a) are offsets that fall between grid points.

### 4.2 Measured patterns against the full library

Table 1 and Fig. 3 give the results on the 188 measured patterns. With background subtraction, the
classical scores retrieve a correct entry first for 71–87 % of compounds; the ln q scan is highest at 0.87,
followed by cosine + zero scan (0.84), Pearson and WCC l = 1° (0.83) and WCC l = 2° (0.71). The CIs of
the first four overlap widely, and we do not rank them. Top-10 is 0.88–0.94 for all but WCC l = 2°. The learned model reaches
0.02, and 0.04 at top-10; its median rank for the first correct entry is 5,988.

**Table 1** Retrieval of 188 measured opXRD patterns from 309,080 simulated CSD entries
(background-subtracted queries, no hubness correction). Compound-weighted top-1 and top-10 with 95 %
compound-bootstrap CIs; per-pattern top-1 and median rank of the first correct entry.

| Score | Any same-form entry (123 compounds): top-1 | top-10 | per-pattern top-1 | median rank | Single-crystal entry only (30 compounds): top-1 | top-10 | per-pattern top-1 | median rank |
|---|---|---|---|---|---|---|---|---|
| ln q lag + zero scan | 0.87 [0.82, 0.93] | 0.93 | 0.72 | 1 | 0.49 [0.31, 0.67] | 0.59 | 0.44 | 6 |
| Cosine + zero scan | 0.84 [0.78, 0.90] | 0.94 | 0.72 | 1 | 0.40 [0.23, 0.58] | 0.46 | 0.37 | 31 |
| Pearson, fixed grid | 0.83 [0.77, 0.89] | 0.88 | 0.82 | 1 | 0.30 [0.15, 0.47] | 0.32 | 0.34 | 131 |
| WCC, l = 1° + zero scan | 0.83 [0.76, 0.89] | 0.89 | 0.68 | 1 | 0.50 [0.33, 0.68] | 0.57 | 0.44 | 50 |
| WCC, l = 2° + zero scan | 0.71 [0.63, 0.78] | 0.81 | 0.57 | 1 | 0.47 [0.30, 0.65] | 0.53 | 0.42 | 350 |
| Learned embedding | 0.02 [0.00, 0.04] | 0.04 | 0.01 | 5,988 | 0.00 [0.00, 0.00] | 0.00 | 0.00 | 18,725 |

**Background subtraction matters more than the choice of score.** Without it, compound-weighted top-1
drops to 0.78 for Pearson, 0.72 for cosine + zero scan, 0.39 for WCC l = 1° and 0.27 for WCC l = 2°.
The loss is much larger with a triangle weight (0.44 for both widths) than without one (0.12 for
cosine + zero scan). We did not isolate the cause; a plausible one is that the triangle spreads each
comparison over neighbouring points, so a slowly varying background overlaps with everything.

**Per-pattern and compound-weighted rankings differ.** Per pattern, fixed-grid Pearson leads (0.82) and
the zero-scan scores trail (0.57–0.72). That order comes from one compound, ZZZUWK, which contributes 20 of
the 188 patterns and which Pearson ranks first 12 times and every zero-scan score never (Section 4.4).

**Library size.** All classical scores lose 0.06–0.12 in compound-weighted top-1 between a 2,000-entry
library and the full one (Fig. 3b): Pearson 0.93 → 0.83, cosine + zero scan 0.93 → 0.84, WCC l = 1°
0.90 → 0.83, WCC l = 2° 0.83 → 0.71 and ln q 0.94 → 0.87. Results on small libraries overstate accuracy at
database scale.

**Within families,** with only the compound's own forms as candidates, the classical scores reach
0.94–0.98 top-1 and the model 0.71 (51 patterns).

![](figs/fig3_real_benchmark.pdf)

**Fig. 3** Retrieval of 188 measured patterns from 309,080 simulated CSD entries (background-subtracted
queries). (a) Compound-weighted top-1 with 95 % compound-bootstrap CIs when any same-form entry is
correct (filled circles, 123 compounds) and when only independent single-crystal entries are correct
(open squares, 30 compounds); vertical bars mark the value with the CSLS hubness correction.
(b) Compound-weighted top-1 against library size (any same-form entry; random decoy subsets, mean ±
s.d. over five draws).

### 4.3 Independent single-crystal references are the open problem

When the pattern's own powder-study entry is set aside and only independent single-crystal determinations
count, the classical scores fall to 0.30–0.50 compound-weighted top-1 (Table 1, Fig. 5b). For 34–47 % of
these patterns the first correct entry lies beyond rank 1,000. The scores that tolerate strain do best:
WCC l = 1° (0.50), ln q (0.49) and WCC l = 2° (0.47), against 0.30 for fixed-grid Pearson. The ln q scan
has the best median rank, 6. This fits the physics: a single-crystal structure is usually determined at a
different temperature from the powder measurement, so its peaks are displaced by thermal expansion, which
is anisotropic in molecular crystals [@bond2021thermal]. The 30 compounds give wide intervals, and no
difference between the strain-tolerant scores is resolved.

### 4.4 One compound, twenty patterns, and the hubness correction

**Hubness.** In the probe population, hubness depends on the score. The skewness of N₁₀ over degraded
probes is 18.7 for cosine + zero scan, 12.7 for ln q, 8.0 for Pearson and 3.6 for both WCC widths; the
triangle weight resists hubs.

**ZZZUWK.** In an earlier per-pattern reading we attributed the drop of cosine + zero scan at full library
size to a hub: one entry, GUSRIV, was the wrong first answer for 20 of its 52 misses. All 20 are patterns
of ZZZUWK, propan-1-aminium chloride, measured from 303 to 398 K in 5 K steps; each pattern is the
refinement data of its own CSD entry (ZZZUWK01–20). The confusion is a property of this compound, not of a
hub. Two things make it hard (Fig. 4a).

- *A sparse pattern.* The tetragonal salt has 24 reflections between 5° and 50° 2θ, where the median
  same-form entry of the benchmark has about 130; fewer than 1 % of those entries have fewer.
- *Texture.* Relative to the simulated pattern of its own entry, the measured (001) reflection at 12.0° is
  five times stronger (0.36 of the maximum against 0.07) and (110) at 20.1° five times weaker (0.11 against
  0.53), consistent with plate-like crystallites lying on their c face. The library patterns carry no
  texture, so the own entry matches poorly: WCC l = 1° gives it a median score of 0.79, against 0.97 for the
  other patterns' own entries.

Unrelated entries with strong reflections near ZZZUWK's two strongest measured peaks (23.5° and 28.6°),
such as GUSRIV and ZZZLWW01, then outscore the own entry. The zero scan contributes but is not the root
cause. Without it, cosine already prefers GUSRIV for 9 of the 20 patterns (with it, all 20), and GUSRIV's
best offset is close to zero (median +0.03°); under WCC l = 1°, three decoys beat the own entry at zero
offset in all 20 patterns. The more peak-tolerant the score, the more decoys pass the own entry: one decoy
beats it in all 20 patterns under cosine + zero scan and seven do under WCC l = 1°. Fixed-grid Pearson,
the least tolerant score, ranks the own entry first for 12 of the 20 patterns (Fig. 4b).

![](figs/fig4_case_zzzuwk.pdf)

**Fig. 4** The ZZZUWK case. (a) A measured pattern of propan-1-aminium chloride at 303 K (background
removed), the simulated pattern of its own Rietveld entry ZZZUWK01, and the unrelated entry GUSRIV that
cosine + zero scan ranks first. The measured (001) reflection at 12.0° is much stronger, and (110) at
20.1° much weaker, than in the simulation. (b) Number of the compound's 20 patterns for which each score
ranks a same-form entry first, without and with CSLS.

**CSLS.** The pre-declared correction fixes ZZZUWK for cosine + zero scan (0 → 20 of 20) and raises it
for Pearson (12 → 17) and ln q (0 → 3); for WCC it only exchanges one decoy for another. Per pattern, cosine + zero
scan rises from 0.72 to 0.88. Compound-weighted, the gain is small for every classical score, from +0.01
to +0.04, and no 95 % CI of the paired difference excludes zero (ESI†). It does not help with
single-crystal references (+0.01 to +0.06, all CIs include zero) or with the learned model (0.02 →
0.02). CSLS is a cheap option that removes hub-like confusions, but hubness does not limit classical
retrieval on this benchmark.

### 4.5 Why the learned model fails

The model ranks the correct entry at a median of 5,988 of 309,080 (Fig. 5a). It is not a matter of
unseen structures: for the 81 patterns whose own structure was in its training set, the median rank is
6,396 and top-1 is 0.01; for the 101 whose structures were not in the corpus, 5,617 and 0.00. Within
families, where it only has to separate a compound's forms, it reaches 0.71 against 0.94–0.98 for the
classical scores.

Using it as a shortlist does not help either. Re-ranking its top 1,000 with the ln q scan gives 0.26
per-pattern top-1, against 0.72 for the ln q scan on the full library; with a top-100 shortlist every
re-ranker gives 0.08–0.09. The shortlist loses the correct entry before the classical score sees it.

We see two gaps, and our experiments do not separate them. The first is discrimination. Even on clean
simulated queries the model stops at 0.56 within families, where every profile score reaches 1.00, so
its embedding does not keep the fine differences between patterns that correlation uses; training against
511 in-batch negatives may not push it to separate one structure from 300,000 others. The second is the
measured-to-simulated gap: the training simulation covers zero offset, strain, broadening, texture,
background and noise, but not Kα2 or real instrument profiles and background shapes, and the model has
to transfer a learned representation across that gap. A correlation score compares the patterns directly
and needs no such transfer.

![](figs/fig5_rank_distributions.pdf)

**Fig. 5** Cumulative distribution of the rank of the first correct entry over the full library when
(a) any same-form entry is correct and (b) only independent single-crystal entries are correct.
Background-subtracted queries, no hubness correction.

## 5. Discussion

**What a learned retrieval method has to beat.** On these data, a correlation with a 2θ zero scan or a
ln q lag scan, on background-subtracted patterns, retrieves the correct entry first for 84–87 % of compounds
from 309,080 candidates. It is not slow: on one GPU, all five classical scores of 8,188 patterns against
the full library took 761 s (Section 3.3). We suggest that
learned PXRD retrieval be reported (i) on measured patterns, (ii) against a library of realistic size,
(iii) against shift- and strain-tolerant classical scores with background subtraction rather than
fixed-grid correlation or peak lists, (iv) with compound-weighted metrics and a compound-level bootstrap
whenever compounds repeat, and (v) separately for queries whose reference is an independent determination.
The benchmark released with this paper covers all five.

**Where learning could still help.** Independent single-crystal references remain hard: a third to a half
of those patterns have their first correct entry beyond rank 1,000 for every score tested. The cause is
physical (thermal expansion between the two determinations, texture, impurities) and the classical scores
handle it only partly. Variable-cell comparison [@oterodelaroza2024vcpwdf] and texture-aware scoring are
the natural next classical steps; a learned method that targets these residuals, rather than the whole
retrieval, is the more promising use of learning.

**Limitations.** The benchmark has 123 compounds, all organic, from one contributor to opXRD, and most
patterns are Rietveld data whose own entry is in the library. The single-crystal set is 30 compounds, so
its intervals are wide. The correct answers rely on packing similarity and treat entries that cannot be
compared as unknown. We tested one learned model; a model trained with measured-pattern augmentation or a
different loss might do better, but it would have to gain more than 0.8 in top-1 to match the classical
scores. Kα2 was not stripped, and no score models texture.

## 6. Conclusions

On simulated queries and on 188 measured organic patterns searched against 309,080 CSD entries, a
contrastive PXRD–structure model is far behind classical profile matching with a zero scan or a ln q lag
scan: 0.02 against 0.71–0.87 compound-weighted top-1. The robustness to shifts that motivated the model is
cheap to obtain classically. The hard case is a measured pattern whose only reference is an independent
single-crystal structure, where the best scores reach about 0.5. We release the benchmark so that learned
methods can be compared with these numbers on measured data.

## Data availability

<!-- [FRANK] Digital Discovery requires code with a DOI and, for restricted data, "a representative dataset
that enables similar conclusions"; see tasks/todo.md Step 2. Draft below assumes a public code release +
Zenodo deposit; the open stand-in dataset is not built yet. -->

The code, the benchmark definition and all results are available at [GitHub URL] and archived at Zenodo
([DOI]). The deposit contains: the opXRD pattern identifiers and their CSD refcodes, same-form sets and
packing-similarity results; the per-pattern ranks for every score and correction; the scripts that build
the library from a CSD installation, compute the scores (CPU and GPU) and regenerate every table and
figure; and results/paper_numbers.json, from which every number in this article is checked by
scripts/verify_claims.py. The measured patterns are from opXRD [@opxrd2025] under CC BY 4.0. CSD
structures and the simulated patterns derived from them cannot be redistributed; users with a CSD licence
regenerate them from the refcode lists. The model checkpoint was trained on CSD data and is not
distributed. [Representative open dataset: FRANK]

## Author contributions

F. C. conceived the study, wrote the code, ran the computations and analysis, and wrote the manuscript.

## Conflicts of interest

There are no conflicts to declare.

## Acknowledgements

The author thanks the opXRD contributors and the CNRS crystallographers whose Rietveld data made the
benchmark possible, and the CCDC for the CSD and its Python API. GPU time was rented from commercial cloud
providers. This research received no specific grant from any funding agency in the public, commercial, or
not-for-profit sectors. [AI-use statement: FRANK, in his own words; RSC asks for it here.]

## Notes and references
