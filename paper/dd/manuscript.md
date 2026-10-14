---
title: "Classical profile matching is the bar for learned powder-diffraction retrieval: a real organic benchmark against 309,080 CSD entries"
bibliography: refs_dd.bib
csl: royal-society-of-chemistry.csl
link-citations: true
---

<!-- DRAFT v1 2026-10-12 (v0 = commit 25b11e3; v1 applies two internal referee reports and the reference
ledger docs/lit-2026-10/refs_dd_verified.md). Every number is checked by scripts/verify_claims.py against
results/paper_numbers.json; citation keys resolve against refs_dd.bib. [FRANK] marks his decisions. -->

Frank Cai^a^

^a^ Purdue University, West Lafayette, Indiana, USA. E-mail: cai485@purdue.edu. ORCID: 0009-0003-0041-1459
<!-- [FRANK] Purdue e-mail kept for the Purdue-RSC open-access agreement (runs to 31 Dec 2026, applied at
acceptance); see tasks/todo.md Step 2. -->

## Abstract

Contrastive models that embed powder X-ray diffraction (PXRD) patterns and crystal structures in a shared
space have been proposed for structure retrieval, but they are mostly tested on simulated queries and
against baselines that cannot absorb a peak shift. We test one such model, trained on 247,089 molecular
crystals from the Cambridge Structural Database (CSD), against classical profile matching on 188 measured
patterns of 123 organic compounds from the open opXRD database. The correct answers are fixed by CCDC
packing similarity, and every pattern is ranked against 309,080 simulated CSD entries. When the answer must
be an independent single-crystal determination, which is the realistic case, the classical scores reach a
compound-weighted top-1 of 0.30–0.50 (30 compounds) and the model 0.00; there, the scores that tolerate
peak shifts and strain beat fixed-grid Pearson correlation by 0.18–0.20. When the pattern's own powder-study
entry also counts, the classical scores reach 0.71–0.87 and the model 0.02. The model is not blind: on
clean simulated patterns of the same targets its top-1 is 0.21 and its median rank 5, so it is the step
from simulated to measured patterns that defeats it. In 34 simulated degradation conditions, each of the
shift-tolerant classical scores beats it in at least 33. A hubness correction (CSLS) changes
compound-weighted top-1 by +0.01 to +0.04, which the data cannot distinguish from zero. We release the
benchmark as code and refcode lists.

## 1. Introduction

Powder X-ray diffraction is the routine way to identify which crystalline phase, and for molecular
materials which polymorph, a sample contains. The standard approach is search–match: the measured pattern
is compared with reference patterns, measured or simulated, from a database such as the Powder Diffraction
File [@gatesrector2019pdf] or the Cambridge Structural Database (CSD) [@groom2016csd]. For organic
crystals the polymorph matters as much as the compound: one in three compounds in the CSD is polymorphic,
and the appearance of a new, much less soluble form once threatened the supply of a marketed drug
[@bauer2001ritonavir; @cruzcabeza2015polymorph].

Machine learning offers another route. Contrastive models train a pattern encoder and a structure encoder so
that a pattern lands near the structure that produced it; retrieval is then a nearest-neighbour search
[@lai2025xtalnet; @xu2026xccp; @yang2026pxtal]. Generative models build the structure from the pattern
[@johansen2026decifer; @segal2026xrdiff]. These models are trained on simulated patterns, mostly of
inorganic or framework materials, and evaluated mostly on simulated patterns. Tests on measured data are
recent and limited: two framework case studies [@lai2025xtalnet], a zero-shot test on 773 inorganic opXRD
patterns [@xu2026xccp], and four inorganic samples for a generative model [@johansen2026realworld].
Where learned and simple methods have been compared on measured patterns, the simple ones have held up. On
measured patterns that are mostly minerals and inorganic or hybrid entries, with a minority of molecular
crystals, a model-free correlation reached 82.8 % recall@10 against 34.4 % for a contrastive model trained
on simulated patterns with corruption augmentation [@srivastava2026realbind]. On experimental patterns of
pharmaceutical solid forms, de Gelder's weighted cross-correlation gave a better accuracy and F1 score than
a Siamese network, which won on ranking metrics [@lange2024smolnet]. A hand-crafted peak-feature index
outperformed element-constrained commercial search–match on measured mineral patterns, with different
reference libraries [@wang2026matdiffract], and a classifier trained on synthetic crystals lost much of its
accuracy on measured diffractograms [@schopmans2023synthetic]. Other fields report similar results: simple
heuristics often outperformed six of the seven neural recommenders that could be reproduced
[@ferraridacrema2019progress], fair comparison left little of the reported progress in deep metric
learning [@musgrave2020reality], and in a mass-spectrometry benchmark, ranking candidates by their
forward-simulated spectra exceeded the best spectrum-to-fingerprint retrieval model
[@bushuiev2024massspecgym].

We previously trained a contrastive model of this kind on molecular crystals from the CSD (xrd-clip)
[@cai2026urtc]. Within families of polymorphs it was more robust to a 2θ zero offset and to lattice strain
than fixed-grid Pearson correlation and a peak-list match. Neither of those baselines can absorb a peak
shift, so the comparison says little about the methods crystallographers use. A 2θ zero-offset scan, de
Gelder's triangle-weighted cross-correlation [@degelder2001wcc] and a lag scan on a ln q axis each absorb a
shift or an isotropic strain at little cost, and variable-cell methods also absorb anisotropic strain
[@mayo2022vcpwdf; @mayo2023vcxpwdf; @oterodelaroza2024vcgpwdf].

This paper asks whether our model beats these methods, first on simulated and then on measured patterns.
It does not, on either. We make four contributions.

1. A simulated stress test of the model against shift-tolerant classical scores over 34 degradation
   conditions (Section 4.1).
2. A real benchmark for organic PXRD retrieval: 188 measured opXRD patterns [@hollarek2025opxrd] identified
   in the CSD by packing similarity [@chisholm2005compack; @macrae2008mercury], two definitions of the
   correct answer, and a library of 309,080 simulated entries (Sections 2, 4.2 and 4.3). It is released as
   code and refcode lists, since CSD structures cannot be redistributed.
3. A diagnosis of the model's failure: it retrieves simulated versions of the same targets but not the
   measured patterns (Section 4.5).
4. A check of whether hubness, the tendency of a few library entries to be the nearest neighbour of many
   queries [@radovanovic2010hubs], limits classical retrieval at this library size (Section 4.4).

## 2. Benchmark

### 2.1 Measured patterns

The opXRD database collects experimental powder patterns with their labelled phases (CC BY 4.0; the source
data from the Crystallography Open Database are CC0) [@hollarek2025opxrd; @opxrd2025zenodo]. Its CNRS
contribution holds 1,052 patterns taken from COD entries that deposit powder data, most with the refined
structure as a full atom list. We kept patterns whose phase contains C and H and no element outside {C, H,
N, O, S, P, F, Cl, Br, I, B, Si}, which left 196. Of the 188 patterns used in the end (Section 2.2), 120 were
measured with Cu Kα radiation, 61 with synchrotron radiation (0.34–1.30 Å) and 7 at Mo, Co or other
wavelengths. opXRD records no wavelength for 3 of the 120, and we assumed Cu Kα. Every pattern
was converted to a Cu Kα1 2θ axis through its d-spacings and interpolated onto the grid used throughout this
work: 5–50° 2θ in 0.02° steps (2,251 points). The Kα2 component was not removed. In 31 of the 188 refined
structures the H atoms are omitted.

### 2.2 Correct answers by packing similarity

A cell match is not enough to identify a compound: unrelated compounds with the same elements can have
similar reduced cells. We therefore identified every pattern from its deposited structure (Fig. 1).

1. **Candidates.** A CSD reduced-cell search on the deposited structure's primitive Niggli cell (1.5 % in
   lengths and 2° in angles, the CCDC defaults), keeping every hit with the same non-H element set.
2. **Anchors.** CCDC packing similarity [@chisholm2005compack; @macrae2008mercury] between the deposited
   structure and each candidate, over a 15-molecule shell (distance tolerance 20 %, angle tolerance 20°,
   H positions ignored). A candidate that matches 15 of 15 molecules is an anchor. Structures rebuilt from
   bare coordinates sometimes perceive bonds differently from curated CSD molecules, so when no candidate
   passes, a lenient pass ignores bond types, H counts and the smallest components. 154 patterns were
   anchored in the strict pass and 35 in the lenient pass.
3. **Same-form set.** Each anchor compared with every entry of its CSD refcode family, CSD against CSD
   with the strict settings. An entry that matches 15 of 15 molecules is the same solid form, whatever its
   temperature or pressure. Entries that cannot be compared are set aside and never scored as errors (134
   entries, affecting 57 patterns).

![](figs/fig1_benchmark.pdf)

**Fig. 1** Construction of the benchmark. Top: each measured pattern is identified from its own deposited
structure. CSD candidates from a reduced-cell search are compared with that structure by packing
similarity, and the matching anchor entry is compared with its refcode family to collect every entry of
the same solid form. Bottom: the measured pattern (on a Cu Kα1 2θ axis, Kα2 retained) is scored against all
309,080 simulated library patterns, and the rank of the first correct entry is recorded under two answer
sets.

189 of the 196 patterns were identified. The other seven are three Si frameworks or silsesquioxanes that
passed the element filter and four patterns with no packing match. One identified pattern was excluded
because its 2θ axis runs from 0.01 to 0.83, probably radians. That leaves **188 patterns of 123 compounds**.
Most compounds have one pattern (110 of 123), but a few have many: ZZZUWK has 20 (a variable-temperature
series), ZZZIZM 12 and NAPHTA 10.

Same-form sets were built within the anchor's refcode family, so a same-form entry deposited under another
refcode would be scored as an error. To check, we compared every wrong first-ranked entry of the classical
scores (454 cases over both answer sets) with the pattern's anchors by packing similarity. One entry,
GOFWUV, is the same form as the anchor of one QQQAUJ pattern; correcting it would raise every classical
score's per-pattern top-1 by 1/188. We leave the labels as built and report this here.

### 2.3 Two definitions of the correct answer

For 184 of the 188 patterns, the CSD contains a powder-study entry that reproduces the deposited structure
to within 0.05 Å RMSD (150 strict and 34 lenient anchors), so the pattern is the data that entry was
refined against [@rietveld1969]. Finding that entry tests whether a method can read a measured pattern, but
it is close to self-retrieval. We therefore score two answer sets.

- **Single-crystal entry only** (59 patterns, 30 compounds): same-form entries that are powder studies are
  set aside, so the answer must be an independent single-crystal determination. This is the realistic case of a new measurement checked against a database, and we treat it as the
  headline.
- **Any same-form entry** (188 patterns, 123 compounds): any entry in the pattern's same-form set is
  correct. This is an upper bound on what a method can do with these patterns.

We also report **within-family** retrieval, the rank of the correct form among the compound's own forms in
the library (51 patterns of 35 compounds; chance 0.40), which is the polymorph-identification question in
isolation.

### 2.4 Library

The library holds 309,080 simulated patterns: the 308,862 molecular crystals of the xrd-clip corpus
(organic CSD entries without disorder, R ≤ 7.5 %, Z′ ≤ 2, after removing duplicate redeterminations)
[@cai2026urtc] and 218 same-form or polymorph entries of the benchmark compounds that the corpus filter had
removed; 28 further entries have no coordinates in the CSD. Stick patterns (Cu Kα1) were computed with the
XRDCalculator module of pymatgen [@ong2013pymatgen]; recomputing 296 randomly drawn entries with pymatgen
2026.4.16 reproduces the stored sticks exactly. The sticks were rendered with one pseudo-Voigt profile
(Caglioti-type width, U = 0.002, V = −0.001, W = 0.004 deg², η = 0.5) [@caglioti1958], without
Debye–Waller factors, texture, background or noise; a library closer to the measurements is tested in
Section 3.7. For disordered entries whose CIF fails a strict parse, coincident sites are merged by
rescaling occupancies; the own entry of 28 patterns (9 compounds) is disordered.

### 2.5 Metrics

For each pattern and score, the rank of the first correct entry is 1 plus the number of entries that are
neither correct nor set aside and that score strictly higher than the best correct entry; ties therefore
favour the correct entry. Top-k is the fraction of patterns with rank ≤ k. Because some compounds
contribute many patterns, per-pattern averages mostly measure those compounds, so our headline is the
**compound-weighted** top-k: the mean over compounds of each compound's top-k rate. Confidence intervals
(95 %) come from 2,000 bootstrap resamples of compounds [@field2007cluster], so the patterns of one
compound move together; for the single-crystal set only its 30 compounds are resampled. Differences
between scores and the effect of a correction are paired: both are computed on the same resampled
compounds.

### 2.6 An open replica

The CSD licence does not allow the library or the answer sets to be redistributed. So that the results can
be checked without a licence, we rebuilt the benchmark from the Crystallography Open Database (COD; CC0)
[@grazulis2009cod; @vaitkus2021cod], the source of the opXRD patterns themselves. The queries are
unchanged. The library applies the corpus filter of Section 2.4 to the COD metadata (C-containing entries
with coordinates, no disorder flag, R ≤ 7.5 %, Z′ of 1–2, no element outside H, C, N, O, F, S, Cl, Br, I
and P), which keeps 108,716 entries, and adds 854 candidate entries of the queries (below) that the filter
removes. Of these 109,570 entries, 108,902 gave stick patterns, computed and rendered exactly as for the
CSD library.

The answer sets follow Section 2.2 with two changes. First, the COD has no refcode families. Candidates
are COD entries with the same non-H element set whose primitive reduced cell lies within 5 % of the deposited
structure's in each length and within 6 % in volume, a wider window than in the CSD search so that
temperature series are not lost. The two closest anchors are then compared with every other candidate, COD
entry against COD entry with the strict settings, to give the same-form set. Second, the COD candidates
include other compounds, which the CSD family search never returned. A candidate whose reduced non-H formula
differs from that of every same-form entry stays an ordinary library entry, and so counts as an error when it
ranks above the answer; setting the 2,670 such candidates aside, as the rule of Section 2.2 would, would
remove the hardest decoys and flatter every score. Same-formula candidates that cannot be compared (23) are
set aside as before. Powder or single-crystal is read from the COD method field, else from the powder tags of
the CIF. This rule was fixed before any replica result was computed.

191 of the 196 patterns are identified in the COD: all 189 identified in the CSD and two more. 35 patterns
have a single-crystal same-form entry, against 59 in the CSD, so the replica's single-crystal set is smaller
and its intervals are wider. Compounds are labelled as in the main benchmark. The replica is scored with the
scores of Section 3.2 and the model, without the corrections of Sections 3.5 and 3.7.

## 3. Methods

### 3.1 Preprocessing

The measured patterns were scored in two forms: *raw*, shifted to a zero minimum and scaled to a unit
maximum, and *background-subtracted*, in which a rolling minimum over 2° smoothed by a 2° moving average is
removed before scaling. Library patterns are scored as simulated.

### 3.2 Classical scores

- **Pearson, fixed grid.** The Pearson correlation of the two patterns on the common grid, the main
  baseline of our earlier work [@cai2026urtc].
- **Zero scan.** Each score below is maximised over a constant 2θ offset of ±0.5° in 0.02° steps. This
  absorbs a zero error exactly and a sample-displacement error, which varies as cos θ, approximately. For
  converted synchrotron patterns a native zero error becomes a slightly angle-dependent shift on the Cu
  axis.
- **WCC.** de Gelder's weighted cross-correlation [@degelder2001wcc] with a triangular weight of
  half-width l = 1° or 2°, which makes the score degrade gradually rather than abruptly with peak shifts,
  plus the zero scan. With l = 0 the weight is a delta and WCC reduces to the cosine of the two patterns
  (*cosine + zero scan*). These widths are large compared with the simulated peak width (FWHM about 0.06°);
  the simulated sweep also includes l = 0.5°.
- **ln q lag scan.** Both patterns are resampled onto a ln q axis with 2,251 points, on which an isotropic
  lattice strain is a translation; each ln q bin takes the mean of the pattern over its width, so peak areas
  are kept where the bins are wider than the 2θ step. The score is a WCC with l = 0.01 in ln q, maximised over
  lags up to ±3 % strain and over 17 zero offsets from −0.32° to +0.32° in 0.04° steps, applied before
  resampling. A first version of this score sampled the ln q axis at points and used 16 offsets that skipped
  zero (nearest ±0.02°), which cost it accuracy on clean simulated queries; we found both problems in an
  internal review, corrected them, and report the first version in the ESI.† The offset grid is still 0.04°, so on simulated
  queries the score depends on whether an applied offset falls on the grid (Section 4.1).

In the simulated sweep the l = 0 variant was applied to standardised patterns, which makes it Pearson +
zero scan; on the measured patterns it is cosine + zero scan. A variable-cell method
[@oterodelaroza2024vcgpwdf] was also tried in the sweep but is not reported, because our reimplementation
has not been validated against the reference code; on the measured patterns we run the reference code
itself (Section 3.8).

### 3.3 Exact scores on a GPU

The full-library ranks were first computed on CPUs. The hubness analysis (Section 4.4) needs every score,
not only ranks, so we ported the scores to PyTorch [@paszke2019pytorch]. Folding the triangle weight into
the reference turns every zero offset into one matrix product: for weights w_k and shift d,
Σ_k w_k Σ_x q(x) r(x + d + k) = Σ_x q(x) r_w(x + d), with r_w the reference convolved with w. On one RTX 4090
the scores of 8,188 rows (benchmark patterns and probes) against all 309,080 entries took 761 s. The GPU
scores agree with the CPU scores to 4 × 10⁻⁶ and reproduce the CPU ranks with no top-1 change (ESI†).

### 3.4 Learned model

The model is the xrd-clip checkpoint of ref. [@cai2026urtc] (6.1 million parameters). A one-dimensional
residual convolutional network encodes the pattern on the grid above; a Transformer over the atoms of the
asymmetric unit, with a lattice token, encodes the structure; both map to a 256-dimensional unit vector. It
was trained once (one seed) for 60 epochs with a symmetric InfoNCE loss [@oord2018cpc; @radford2021clip],
batch size 512 and polymorph hard-negative batches (sibling polymorphs placed in the same batch), on 247,089
CSD structures of a scaffold- and polymorph-family split. Patterns were simulated on the fly with random
zero offset, isotropic strain, Caglioti and crystallite-size broadening, March–Dollase texture
[@dollase1986], background and Poisson noise, but without Kα2. The checkpoint was chosen by in-batch
validation top-1. A measured pattern is scored against the library by the cosine between its pattern
embedding and each entry's structure embedding. We also test the model as a shortlist, re-ranking its top
100 or 1,000 entries with a classical score. Further details are in the ESI.†

Three diagnostics probe the failure reported in Section 4.5. (i) **Effect ladder.** Each pattern's own
library entry is rendered cleanly and then with one effect of the measurement at a time, each estimated from
the measured pattern: its 2θ coverage, its peak width, Kα2 (lab Cu patterns), its own background followed by
the background subtraction of Section 3.1, noise at its noise level, and the zero offset that best aligns it
with its entry; then with all six, and with all six plus an overall Debye–Waller factor (B = 4.45 Å², from
the library, Section 3.7) or log-normal intensity jitter per reflection. Every variant is scored with the
answer sets of the measured pattern. Two of the effects use the own entry, so the ladder diagnoses the model
and is not a method. (ii) **Pattern to pattern.** The clean library patterns are passed through the model's
pattern encoder, so measured patterns are matched against simulated patterns in the learned space without
the structure encoder. (iii) **Realism retrain.** The model was retrained once with the design and a
prediction fixed before training (repository commit 3b77818): the same split, architecture,
hyperparameters, epochs, hard-negative batches and checkpoint rule, with the training simulation extended by
a Kα2 copy (probability 0.6, I₂/I₁ = 0.5), an overall Debye–Waller factor with B drawn uniformly from 0–8 Å²,
log-normal intensity jitter per reflection with σ drawn uniformly from 0–0.5, a cut of the 2θ range (probability 0.25;
start 5–12°, end 30–50°), and the background subtraction applied to the measured patterns. The retrained
model is scored like the original, including the ladder.

### 3.5 Hubness diagnosis and correction

A *hub* is a library entry that scores highly against many unrelated queries. We measured hubness with
probe patterns rather than the benchmark patterns: 4,000 library entries simulated cleanly and 4,000
simulated with the training-time degradations and then background-subtracted like the measured patterns,
drawn only from compounds outside the benchmark, with pairs from the same compound excluded. For each score
we report the skewness of N₁₀, the number of probes that have an entry among their ten best matches
[@radovanovic2010hubs]. With 4,000 probes and 309,080 entries the mean N₁₀ is 0.13, so these skewness values
are not comparable with those of the hubness literature.

The analysis is exploratory: it was prompted by one entry that was the wrong first answer for many patterns
(Section 4.4). The correction itself was fixed before any corrected number was computed (repository commit
2a2ad8a): cross-domain similarity local scaling (CSLS) [@conneau2018muse] on the library side,
s′(q, r) = s(q, r) − h₁₀(r)/2, where h₁₀(r) is the mean of entry r's ten highest scores against the degraded
probes. The query-side term of CSLS is the same for every entry and does not change ranks. As sensitivity
checks we report k = 1–100, clean probes, library-side z-normalisation and mutual proximity
[@schnitzer2012mp] (ESI†); none was selected after seeing the results.

### 3.6 Simulated sweep

The sweep reuses the within-family test of ref. [@cai2026urtc]: 1,077 held-out polymorph families with
2,835 query structures, where the query is one form's simulated and degraded pattern and the candidates are
the clean patterns of the family's forms (composition-only chance 0.38). In the clean condition the query is
the candidate's own rendering, so every profile score that can align it exactly scores 1.00; the sweep
measures how far each method falls from that starting point. Each condition varies one degradation: zero
offset up to 0.3°, isotropic strain up to 1.2 %, crystallite size down to 10 nm, March–Dollase r down to 0.5
and counting noise; or an anisotropic thermal strain for a temperature change of 50–200 K, with per-axis
expansion coefficients drawn so that the mean volumetric coefficient (about 1.65 × 10⁻⁴ K⁻¹) matches the
CSD survey of ref. [@bond2021thermal]; or combinations with random offsets and strains. Every method scores
the same rendered queries, and CIs come from a bootstrap over families. The sweep has 34 distinct
conditions (a zero offset of 0° is the clean condition). Each condition's random draws are seeded from a
checksum of its name, so the sweep can be regenerated exactly.

### 3.7 A library closer to the measurements

The library of Section 2.4 leaves out several features of the measured patterns. As a robustness check, all
classical scores were recomputed with one package of changes, fixed before scoring (repository commit
4d4dae3). (a) An overall Debye–Waller factor on the library intensities, I exp(−2Bs²) with s = sin θ/λ and
B = 4.45 Å², which is 8π² times the median mean non-H U_eq of 5,000 random library entries measured at
283–303 K. (b) For the 68 patterns not measured with Cu Kα, the Lorentz–polarisation factor converted from
the measurement to Cu Kα1, with polarisation (1 + cos² 2θ)/2 for laboratory tubes and none for synchrotrons
(an assumption). (c) For the 125 laboratory-tube patterns, Kα2 as a nuisance: the score is the higher of
those against a Kα1 library and a Kα1 + Kα2 library (I₂/I₁ = 0.5), because opXRD does not record whether
the beam was monochromated. (d) For the 46 patterns that do not cover 5–50°, comparison over the measured
range only (reference norms over the query's coverage, not shifted with the zero scan). Each change was also
removed on its own to give one-off ablations. None of the four was selected after seeing results.

### 3.8 Variable-cell re-rank

Single-crystal references fail mainly when their cell differs from the measured one (Section 4.2), which
variable-cell comparison is designed to absorb. We re-ranked shortlists with VC-GPWDF
[@oterodelaroza2024vcgpwdf] as implemented in critic2 1.4 [@oterodelaroza2014critic2], which compares a
candidate structure with a measured pattern while varying the candidate's cell (±10 % in lengths and ±5° in
angles in its quick mode). We ran the published pipeline: background fit with 20 knots, peak fit with the
default settings, and the global quick comparison with α = 1 as in the paper (the code's default is 0.5) and
the measurement wavelength; it applies no zero-offset correction. One step was added: each pattern was cut
to the d-spacing window that the other scores see (Cu Kα1 2θ of 5–50°). The shortlist of each single-crystal
pattern is the 50 best entries of ln q, with set-aside entries excluded; a pattern whose answer is not in
its shortlist counts as a miss, and a comparison that fails or runs past a 4-hour limit counts as maximally
dissimilar. We had planned a shortlist of 100, but the shortlist holds the answer for the same 36 of the 59
patterns at 50 as at 100, so the further entries could only have added competitors; the shorter list is, if
anything, favourable to the re-rank. A comparison took a median of TBD(c2_cpu_median) CPU-minutes and up to
TBD(c2_cpu_max) CPU-hours for the largest cells (TBD(c2_capped) reached the limit).

## 4. Results

### 4.1 Simulated queries: the model loses in every condition

Across the 34 conditions the model scores 0.47–0.56 within-family top-1 and the best classical score
0.64–1.00 (Fig. 2). The best classical score leads by 0.12 to 0.48, and the 95 % CIs never overlap. The
result does not depend on picking the best score per condition: WCC (each of l = 0.5°, 1° and 2°) and the
ln q scan beat the model with non-overlapping CIs in all 34 conditions, and Pearson + zero scan in 33.

The fixed-grid Pearson baseline of our earlier work behaves as reported there: it falls to chance by a
0.15° zero offset or a 0.8 % strain, while the model is flat (Fig. 2a and b). The zero scan removes that
weakness: Pearson + zero scan scores 1.00 at every offset that is a multiple of the 0.02° step; the dips at
0.05° and 0.15° are offsets that fall between grid points. The ln q scan holds 0.86 at 1.2 % isotropic
strain. The model's curves are flat, but they are flat at about 0.55; its clean within-family top-1 is 0.56.
The smallest gap, −0.12, comes from large anisotropic thermal strain combined with a degraded profile. Even
there, WCC (l = 1°) leads, 0.64 against 0.52.

![](figs/fig2_simulated_sweep.pdf)

**Fig. 2** Simulated stress test: within-family top-1 over 1,077 polymorph families (2,835 queries).
Bands are 95 % family-bootstrap CIs; the dashed line is composition-only chance (0.38). (a) 2θ zero
offset, (b) isotropic lattice strain, (c) anisotropic thermal strain, (d) crystallite size,
(e) March–Dollase preferred orientation, (f) counting statistics. The l = 0 score is Pearson + zero scan
here (cosine + zero scan on the measured patterns). The dips of the zero-scan scores at 0.05° and 0.15° in
(a) are offsets that fall between grid points; the ln q score stays below 1.00 because its offset grid
skips zero (Section 3.2).

### 4.2 The realistic case: independent single-crystal references

When the pattern's own powder-study entry is set aside and only independent single-crystal determinations
count, the classical scores reach 0.30–0.50 compound-weighted top-1 (Table 1, Fig. 3a) and the model 0.00.
For 34–47 % of these patterns the first correct entry lies beyond rank 1,000 for every classical score.

Here shift and strain tolerance pays. Paired over compounds, the ln q scan beats fixed-grid Pearson by 0.18
[0.02, 0.35] and WCC (l = 1°) by 0.20 [0.03, 0.36]; cosine + zero scan (+0.10 [−0.05, 0.24]) and WCC
(l = 2°, +0.16 [−0.03, 0.35]) are not resolved. The ln q scan has the best median rank, 6.

What separates success from failure is how well the cells agree. Where the closest single-crystal entry
has every cell length within 1 % of the powder-study entry's, the ln q scan ranks it first for 0.71 of
patterns (34 patterns); where they differ by more, for 0.08 (25 patterns). Thermal expansion is the obvious
suspect, since a typical molecular crystal expands by about 3 % in volume between 100 K and room
temperature [@bond2021thermal], but the recorded temperatures of the two entries differ by a median of 3 K
for the 36 patterns where both are known, so it explains few of these mismatches. Pressure, refinement
differences and the low precision of some early determinations are other candidates; we did not resolve
them. The ln q scan absorbs isotropic strain up to 3 %, so its failures beyond 1 % suggest anisotropic
cell differences, which none of the scores tested here absorbs.

**Table 1** Retrieval of 188 measured opXRD patterns from 309,080 simulated CSD entries
(background-subtracted queries, no hubness correction). Compound-weighted top-1 and top-10, with 95 %
compound-bootstrap CIs for top-1; per-pattern top-1; median rank of the first correct entry over patterns.

| Score | Single-crystal entry only (30 compounds): top-1 | top-10 | median rank | Any same-form entry (123 compounds): top-1 | top-10 | per-pattern top-1 | median rank |
|---|---|---|---|---|---|---|---|
| ln q lag + zero scan | 0.49 [0.31, 0.67] | 0.59 | 6 | 0.87 [0.82, 0.93] | 0.93 | 0.72 | 1 |
| WCC, l = 1° + zero scan | 0.50 [0.32, 0.67] | 0.57 | 50 | 0.83 [0.76, 0.89] | 0.89 | 0.68 | 1 |
| WCC, l = 2° + zero scan | 0.47 [0.30, 0.63] | 0.53 | 350 | 0.71 [0.63, 0.78] | 0.81 | 0.57 | 1 |
| Cosine + zero scan | 0.40 [0.24, 0.57] | 0.46 | 31 | 0.84 [0.78, 0.90] | 0.94 | 0.72 | 1 |
| Pearson, fixed grid | 0.30 [0.15, 0.48] | 0.32 | 131 | 0.83 [0.77, 0.89] | 0.88 | 0.82 | 1 |
| Learned embedding | 0.00 [0.00, 0.00] | 0.00 | 18,725 | 0.02 [0.00, 0.04] | 0.04 | 0.01 | 5,988.5 |

### 4.3 When the pattern's own entry counts

With every same-form entry correct, the classical scores reach a compound-weighted top-1 of 0.71–0.87: the
ln q scan 0.87, cosine + zero scan 0.84, Pearson and WCC (l = 1°) 0.83, and WCC (l = 2°) 0.71. Paired over
compounds, only WCC (l = 2°) differs from fixed-grid Pearson (−0.12 [−0.21, −0.03]); the others are
within ±0.05 of it with CIs spanning zero. Top-10 is 0.88–0.94 for all but WCC (l = 2°). The learned model
reaches 0.02, and 0.04 at top-10; its median rank for the first correct entry is about 6,000.

![](figs/fig3_real_benchmark.pdf)

**Fig. 3** Retrieval of 188 measured patterns from 309,080 simulated CSD entries (background-subtracted
queries). (a) Compound-weighted top-1 with 95 % compound-bootstrap CIs when any same-form entry is
correct (filled circles, 123 compounds) and when only independent single-crystal entries are correct
(open squares, 30 compounds); vertical bars mark the value with the CSLS hubness correction.
(b) Compound-weighted top-1 against library size (any same-form entry; random decoy subsets; the s.d. over
five draws is below 0.02).

**Background subtraction matters.** Without it, compound-weighted top-1 drops to 0.78 for Pearson, 0.72
for cosine + zero scan, 0.39 for WCC l = 1° and 0.27 for WCC l = 2°. The loss is much larger with a
triangle weight (0.44 for both widths) than without one (0.12 for cosine + zero scan), presumably because
a triangle averages each comparison over neighbouring points, so a slowly varying background contributes
to every overlap. Our background removal is a simple rolling minimum, and a better one may help further.

**Per-pattern and compound-weighted rankings differ.** Per pattern, fixed-grid Pearson leads (0.82) and
the zero-scan scores trail (0.57–0.72). Pearson ranks 28 patterns first that cosine + zero scan misses, and
cosine + zero scan 10 that Pearson misses; 12 of the net difference of 18 come from one compound, ZZZUWK,
which contributes 20 of the 188 patterns (Section 4.4). Without ZZZUWK, Pearson still leads per pattern
(0.85 against 0.81 for cosine + zero scan and the ln q scan).

**Disordered entries are harder.** For the 28 patterns (9 compounds) whose own entry is disordered, the
classical scores reach 0.56–0.78 compound-weighted top-1, against 0.71–0.88 for the other 160 patterns.
Our simulation treats these entries only approximately (Section 2.4). Converted non-Cu patterns are not
harder than Cu patterns (ESI†).

**Library size.** All classical scores lose 0.06–0.12 in compound-weighted top-1 between a 2,000-entry
library and the full one (Fig. 3b): Pearson 0.93 → 0.83, cosine + zero scan 0.93 → 0.84, WCC l = 1°
0.90 → 0.83, WCC l = 2° 0.83 → 0.71 and ln q 0.94 → 0.87. Results on small libraries overstate accuracy at
database scale.

**Within families,** with only the compound's own forms as candidates (chance 0.40), the classical scores
reach 0.94–0.98 top-1 and the model 0.71 (51 patterns).

### 4.4 One compound, twenty patterns, and the hubness correction

**Hubness.** In the probe population, hubness depends on the score. The skewness of N₁₀ over degraded
probes is 18.7 for cosine + zero scan, 12.7 for ln q, 8.0 for Pearson and 3.6 for both WCC widths.

**ZZZUWK.** One entry, GUSRIV, was the wrong first answer of cosine + zero scan for 20 of its 52 misses,
which looks like a hub. All 20 are patterns of ZZZUWK, propan-1-aminium chloride, measured as a
variable-temperature series; each pattern belongs to its own CSD powder-study entry, ZZZUWK01–20. The
confusion is a property of this compound, not of a hub. Two things make it hard (Fig. 4a).

- *A sparse pattern.* The tetragonal salt has 24 reflections between 5° and 50° 2θ, where the median
  same-form entry of the benchmark has about 130; fewer than 1 % of those entries have fewer.
- *Poorly reproduced intensities.* The measured relative intensities differ strongly from the simulated
  pattern of the pattern's own entry: (001) at 12.0° is 0.36 of the maximum against 0.07 simulated, (101)
  at 18.6° 0.32 against 1.00, and (110) at 20.1° 0.11 against 0.53. The changes do not follow the angle to
  the c axis, as March–Dollase texture along c would require, so preferred orientation alone does not explain
  them. All 20 own entries are
  disordered, omit H atoms and were anchored only in the lenient pass, so our simulation of them is the more
  likely culprit. The own entry consequently matches poorly: WCC l = 1° gives it a median score of 0.79,
  against 0.97 for the other patterns' own entries.

Unrelated entries with strong reflections near ZZZUWK's two strongest measured peaks (23.5° and 28.6°),
such as GUSRIV and ZZZLWW01, then outscore the own entry. The zero scan contributes but is not the root
cause. Without it, cosine already prefers GUSRIV for 9 of the 20 patterns (with it, all 20), and GUSRIV's
best offset is close to zero (median +0.03°); under WCC l = 1°, three decoys beat the own entry at zero
offset in all 20 patterns. The more peak-tolerant the score, the more decoys pass the own entry: one decoy
beats it in all 20 patterns under cosine + zero scan and seven do under WCC l = 1°. Fixed-grid Pearson,
the least tolerant score, ranks the own entry first for 12 of the 20 patterns (Fig. 4b).

![](figs/fig4_case_zzzuwk.pdf)

**Fig. 4** The ZZZUWK case. (a) The measured pattern of propan-1-aminium chloride whose CSD entry,
ZZZUWK01, was refined at 303 K (background removed), the simulated pattern of that entry, and the
unrelated entry GUSRIV that cosine + zero scan ranks first. The measured and simulated relative intensities
of the main reflections differ strongly. (b) Number of the compound's 20 patterns for which each score
ranks a same-form entry first, without and with CSLS.

**CSLS.** The correction fixes ZZZUWK for cosine + zero scan (0 → 20 of 20) and raises it for Pearson
(12 → 17) and ln q (0 → 3); for WCC it only exchanges one decoy for another. Per pattern, cosine + zero
scan rises from 0.72 to 0.88. Compound-weighted, the change is +0.01 to +0.04 for the classical scores,
and no 95 % CI of the paired difference excludes zero; gains of up to 0.10 are not excluded (ESI†). For
single-crystal references the change is +0.01 to +0.06, with upper CI bounds up to 0.17, and for the
learned model it is 0.02 → 0.02. We detect no compound-level gain from the correction; it is a cheap option
that removes hub-like confusions when they occur.

### 4.5 Why the learned model fails

The model ranks the first correct entry at a median of about 6,000 of 309,080 (Fig. 5a). Two diagnostics
locate the failure.

**It retrieves simulated patterns.** For the 4,000 clean simulated probes of Section 3.5, the model ranks
the probe's own entry first for 0.42 of probes and within the top 10 for 0.84 (median rank 2); with the
training-time degradations, for 0.28 and 0.67 (median rank 4). For the clean simulated patterns of the
benchmark's own entries, scored with the same answer sets as the measured patterns, it reaches top-1 0.21
and top-10 0.56, with a median rank of 5. On the measured patterns of the same targets it reaches 0.01 and
0.03. (A classical score would rank a clean rendering of a library entry first by construction, so this
test only diagnoses the model.) The model can discriminate among 309,080 entries; what it cannot do is
carry that ability from simulated to measured patterns.

**Seeing the structure in training does not help.** For the 81 patterns that have a same-form entry in the
model's training split, the median rank is 6,396 and top-1 is 0.01; for the 101 patterns whose same-form
entries are all outside the corpus, 5,617 and 0.00 (the other 6 have entries in the validation or test
split).

Within families, where it only has to separate a compound's forms, the model reaches 0.71 against
0.94–0.98 for the classical scores. As a shortlist it caps any re-ranker: its top 1,000 contains a correct
entry for only 0.28 of patterns, and re-ranking that list with a classical score gives 0.19–0.26 per-pattern
top-1, against 0.57–0.82 for the same scores on the full library.

The training simulation covers zero offset, strain, broadening, texture, background and noise, but not Kα2,
real instrument profiles or real background shapes, and correlation-based scores do not have to transfer a
learned representation across that gap. Full-profile generative models have shown the same sensitivity to
how realistic the training patterns are [@segal2026xrdiff]. Our experiments do not say which missing
ingredient matters most.

![](figs/fig5_rank_distributions.pdf)

**Fig. 5** Cumulative distribution of the rank of the first correct entry over the full library when
(a) any same-form entry is correct and (b) only independent single-crystal entries are correct.
Background-subtracted queries, no hubness correction.

## 5. Discussion

**What a learned retrieval method has to beat.** On measured organic patterns searched against the CSD,
the bar is set by simple classical scores on background-subtracted patterns: a compound-weighted top-1 of
about 0.5 when the reference is an independent single-crystal structure, reached by strain-tolerant scores
(ln q scan, WCC), and of 0.83–0.87 when the pattern's own powder-study entry is in the library, where even
fixed-grid Pearson correlation reaches 0.83. We suggest that learned PXRD retrieval be reported (i) on
measured patterns, (ii) against a library of realistic size, (iii) against shift- and strain-tolerant
correlation scores with background subtraction, (iv) with compound-weighted metrics and a compound-level
bootstrap whenever compounds repeat, and (v) separately for queries whose reference is an independent
determination. The benchmark released with this paper supports all five.

**Speed is the model's one advantage.** With precomputed library embeddings, the model searches 309,080
entries in about 5 ms per query on a CPU. From the per-pair costs of the simulated sweep, one CPU core
needs about 0.5 s per query for fixed-grid Pearson, about 30 s for a zero scan and about 8 min for the ln q
scan. On one GPU, all five classical scores of 8,188 patterns against the full library took 761 s. Speed
only matters once accuracy does, and a shortlist from this model loses most correct entries (Section 4.5).

**Where learning could still help.** Two problems remain open, and both concern the gap between simulated
and measured patterns. For independent single-crystal references, retrieval fails once the cells differ by
more than about 1 %; variable-cell comparison [@mayo2023vcxpwdf; @oterodelaroza2024vcgpwdf] is the natural
classical next step, and we did not test it. For the learned model, the failure is the transfer from
simulated to measured patterns, not the size of the library. A learned method aimed at that transfer, for
example trained on more realistic simulations or adapted with measured patterns, is the more promising use
of learning than a stand-alone retriever.

**Limitations.** We tested one learned model, trained once; another architecture, loss or training
simulation could do better, and our conclusions about learning are about this model. The benchmark has 123
compounds, all organic, from one contributor to opXRD, and most patterns are the data their own entry was
refined against; the single-crystal set has 30 compounds, so its intervals are wide. The library is
simulated with one peak profile, without Debye–Waller factors or Kα2, and disordered entries only
approximately; the measured patterns keep Kα2 and have a simple background removal. The ln q score is
handicapped by its offset grid (Section 3.2). The correct answers rely on packing similarity, set aside
entries that cannot be compared, and contain one known error (Section 2.2). Retrieval identifies known
forms only; crystal structure prediction combined with pattern matching can identify forms that no
database contains [@glick2025csp].

## 6. Conclusions

On 188 measured organic patterns searched against 309,080 simulated CSD entries, a contrastive
PXRD–structure model reaches a compound-weighted top-1 of 0.02, against 0.71–0.87 for classical profile
matching; with independent single-crystal references, 0.00 against 0.30–0.50, where shift- and
strain-tolerant scores beat fixed-grid correlation. The model retrieves clean simulated patterns of the
same targets, so its failure lies in the step from simulated to measured data. The hard open case is a
measured pattern whose only reference is an independent single-crystal structure with a cell that differs
by more than about 1 %. We release the benchmark so that learned methods can be compared with these numbers
on measured data.

## Data availability

<!-- [FRANK] Digital Discovery requires code with a DOI and, for restricted data, "a representative dataset
that enables similar conclusions"; see tasks/todo.md Step 2. Draft below assumes a public code release +
Zenodo deposit; the open stand-in dataset is not built yet. -->

The code, the benchmark definition and all results are available at [GitHub URL] and archived at Zenodo
([DOI]). The deposit contains: the opXRD pattern identifiers with their CSD refcodes, same-form sets and
packing-similarity results; the corpus refcode list and the training splits; the per-pattern ranks of
every score and correction, including the learned model's; the scripts that build the library from a CSD
installation, compute the scores (CPU and GPU) and regenerate every table and figure; and
results/paper_numbers.json, from which every number in this article is checked by scripts/verify_claims.py.
The measured patterns are from opXRD [@opxrd2025zenodo] under CC BY 4.0. CSD structures and the simulated
patterns derived from them cannot be redistributed; users with a CSD licence regenerate them from the
refcode lists. The model checkpoint was trained on CSD data and is not distributed. [Representative open
dataset: FRANK]

## Author contributions

F. C. conceived the study, wrote the code, ran the computations and analysis, and wrote the manuscript.

## Conflicts of interest

There are no conflicts to declare.

## Acknowledgements

The author thanks the opXRD and COD contributors whose deposited powder data made the benchmark possible,
and the CCDC for the CSD and its Python API. GPU time was rented from commercial cloud providers. This
research received no specific grant from any funding agency in the public, commercial, or not-for-profit
sectors. [AI-use statement: FRANK, in his own words; RSC asks for it here.]

## Notes and references
