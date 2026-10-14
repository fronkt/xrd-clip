---
title: "Classical profile matching is the bar for learned powder-diffraction retrieval: a real organic benchmark against 309,080 CSD entries"
bibliography: refs_dd.bib
csl: royal-society-of-chemistry.csl
link-citations: true
---

<!-- DRAFT v2 2026-10-14 (v0 = commit 25b11e3; v1 = 3ecd84f, two internal referee reports and the reference
ledger docs/lit-2026-10/refs_dd_verified.md; v2 folds in revision R1-R5, tasks/todo.md Step 3). Every number is
checked by scripts/verify_claims.py against results/paper_numbers.json; citation keys resolve against
refs_dd.bib. [FRANK] marks his decisions. -->

Frank Cai^a^

^a^ Purdue University, West Lafayette, Indiana, USA. E-mail: cai485@purdue.edu. ORCID: 0009-0003-0041-1459
<!-- [FRANK] Purdue e-mail kept for the Purdue-RSC open-access agreement (runs to 31 Dec 2026, applied at
acceptance); see tasks/todo.md Step 2. -->

## Abstract

Contrastive models embedding powder X-ray diffraction (PXRD) patterns and crystal structures in a shared
space have been proposed for structure retrieval, but are mostly tested on simulated queries against
baselines that cannot absorb a peak shift. We test one such model, trained on 247,089 molecular crystals
from the Cambridge Structural Database (CSD), against classical profile matching on 188 measured patterns of
123 organic compounds from the opXRD database. Correct answers are fixed by packing similarity, and
every pattern is ranked against 309,080 simulated CSD entries. When the answer must be an independent
single-crystal determination, the classical scores reach a compound-weighted top-1 of 0.30–0.50 (30
compounds) and the model 0.00; a variable-cell re-rank (VC-GPWDF) reaches 0.53, not resolved from the ln q
scan it re-ranks (0.45). Most failures involve references determined at other pressures or temperatures.
When the pattern's own powder-study entry also counts, the classical scores reach 0.71–0.87 and the model
0.02. Yet on clean simulations of the same targets it reaches 0.27 (median rank 5). Retrained with
five modelled measurement effects, it handles them in simulation but stays at 0.02 on measured patterns, so
the gap lies in what our simulation does not reproduce. In 34 simulated degradation conditions, each
of the shift-tolerant classical scores beats it in at least 32. A more realistic library and a hubness
correction (CSLS) change none of this, and an open replica on
the Crystallography Open Database reproduces the any-same-form result. We release the benchmark as code,
refcode lists and the open replica.

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
It does not, on either. We make five contributions.

1. A simulated stress test of the model against shift-tolerant classical scores over 34 degradation
   conditions (Section 4.1).
2. A real benchmark for organic PXRD retrieval: 188 measured opXRD patterns [@hollarek2025opxrd] identified
   in the CSD by packing similarity [@chisholm2005compack; @macrae2008mercury], two definitions of the
   correct answer, and a library of 309,080 simulated entries (Sections 2, 4.2 and 4.3). It is released as
   code and refcode lists, since CSD structures cannot be redistributed, and as an open replica built from
   the Crystallography Open Database (Sections 2.6 and 4.7).
3. A test of the published variable-cell method on the hard single-crystal case (Section 4.2).
4. A diagnosis of the model's failure: it retrieves simulated versions of the same targets but not the
   measured patterns, and neither the measurement effects we could simulate nor a retrain that includes
   them closes that gap (Section 4.5).
5. Checks of whether hubness, the tendency of a few library entries to be the nearest neighbour of many
   queries [@radovanovic2010hubs], or the fidelity of the simulated library limits classical retrieval
   (Sections 4.4 and 4.6).

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
  headline; 22 of these patterns come from variable-pressure studies (Section 4.2).
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
Sections 3.7 and 4.6. For disordered entries whose CIF fails a strict parse, coincident sites are merged by
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
compounds. Our primary comparison is the learned model against the classical scores in compound-weighted
top-1. Comparisons among classical scores, and among the variants of Sections 3.5 and 3.7, are exploratory and
are not corrected for multiple comparisons.

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

191 of the 196 patterns are identified in the COD: all 189 identified in the CSD and two more. Without the
pattern with a radian axis (Section 2.2), 190 patterns of 125 compounds are scored. 35 patterns of 9
compounds have a single-crystal same-form entry, against 59 patterns of 30 compounds in the CSD, so the
replica's single-crystal set is small and its intervals are wide. Compounds are labelled as in the main
benchmark. The replica is scored with the scores of Section 3.2 and the model, without the corrections of
Sections 3.5 and 3.7.

## 3. Methods

### 3.1 Preprocessing

The measured patterns were scored in two forms: *raw*, shifted to a zero minimum and scaled to a unit
maximum, and *background-subtracted*, in which a rolling minimum over 2° smoothed by a 2° moving average is
removed before scaling. Library patterns are scored as simulated.

### 3.2 Classical scores

- **Pearson, fixed grid.** The Pearson correlation of the two patterns on the common grid, the main
  baseline of our earlier work [@cai2026urtc].
- **Zero scan.** Each score below is maximised over a constant 2θ offset of ±0.5° in 0.02° steps. This
  absorbs a zero error exactly and a sample-displacement error, which varies as cos θ, approximately. For a
  pattern converted from a short wavelength, a native zero error is magnified by about λ_Cu/λ (up to about
  4.5 times) and becomes weakly angle-dependent on the Cu axis. Specimen transparency, which shifts peaks in
  proportion to sin 2θ and is often the largest aberration for weakly absorbing organic samples in reflection
  geometry, is not absorbed by a constant offset.
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
  zero (nearest ±0.02°), which cost it accuracy on clean simulated queries (within-family top-1 0.88 rather
than 1.00); we found both problems in an
  internal review, corrected them, and report the first version in the ESI.† The corrected version was
  declared primary before it was scored; it lowers the single-crystal top-1 from 0.49 to 0.45 (ESI†). The offset grid is still 0.04°, so on simulated
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
scores agree with the CPU scores to 4 × 10⁻⁶ and reproduce the CPU ranks with no top-1 change (ESI†). The
corrected ln q scan of Section 3.2 was rescored alone (405 s) and reproduces its CPU ranks with no top-1
change.

### 3.4 Learned model

The model is the xrd-clip checkpoint of ref. [@cai2026urtc] (6.1 million parameters). A one-dimensional
residual convolutional network encodes the pattern on the grid above; a Transformer over the atoms of the
asymmetric unit, with a lattice token, encodes the structure; both map to a 256-dimensional unit vector. It
was trained once (one seed) for 60 epochs with a symmetric InfoNCE loss [@oord2018cpc; @radford2021clip],
batch size 512 and polymorph hard-negative batches (sibling polymorphs placed in the same batch), on 247,089
CSD structures of a scaffold- and polymorph-family split. Patterns were simulated on the fly with random
zero offset, isotropic strain, Caglioti and crystallite-size broadening, March–Dollase texture
[@dollase1986], background and Poisson noise, but without Kα2. The checkpoint was chosen by in-batch top-1 on
clean, unaugmented validation patterns. A measured pattern is scored against the library by the cosine
between its pattern embedding and each entry's structure embedding. We also test the model as a shortlist,
re-ranking its top 100 or 1,000 entries with a classical score. Further details are in the ESI.†

Three further diagnostics probe the failure reported in Section 4.5. (i) **Effect ladder.** Each pattern's
own library entry is rendered cleanly and then with one effect of the measurement at a time: its 2θ
coverage, its peak width and its noise level, each estimated from the measured pattern; Kα2, added to every
laboratory Cu pattern because monochromatisation is not recorded; its own background followed by the
background subtraction of Section 3.1; the zero offset that best aligns it with its entry; and an overall
Debye–Waller factor (B = 4.45 Å², from the library, Section 3.7). It is then rendered with the first six
together, and with those six plus the Debye–Waller factor, log-normal intensity jitter per reflection
(σ = 0.3 or 0.5, not estimated from the data), or both (σ = 0.3); the Debye–Waller and jitter rungs were
added after the first ladder. Every variant is scored with the answer sets of the measured pattern. One
effect, the zero offset, uses the own entry, so the ladder diagnoses the model and is not a method.
(ii) **Pattern to pattern.** The clean library patterns are passed through the model's pattern encoder, so
measured patterns are matched against simulated patterns in the learned space without the structure encoder.
(iii) **Realism retrain.** The model was retrained once with the design and a prediction fixed before
training (repository commit 3b77818): the same split, architecture, hyperparameters, epochs, hard-negative
batches and checkpoint rule, with the training simulation extended by a Kα2 copy (probability 0.6,
I₂/I₁ = 0.5), an overall Debye–Waller factor with B drawn uniformly from 0–8 Å², log-normal intensity jitter
per reflection with σ drawn uniformly from 0–0.5, a cut of the 2θ range (probability 0.25; start 5–12°, end
30–50°), and the background subtraction applied to the measured patterns. The retrained model is scored like
the original, including the ladder; since it was trained only on background-subtracted patterns, its ladder
was also rerun with the background subtraction applied to every rung (ESI†).

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
checksum of its name, so the sweep can be regenerated exactly. (A first run seeded them from Python's salted
string hash, which cannot be regenerated; every sweep number here is from the rerun.)

### 3.7 A library closer to the measurements

The library of Section 2.4 leaves out several features of the measured patterns. As a robustness check, all
classical scores were recomputed with one package of changes, fixed before scoring (repository commit
4d4dae3). (a) An overall Debye–Waller factor on the library intensities, I exp(−2Bs²) with s = sin θ/λ and
B = 4.45 Å², which is 8π² times the median mean non-H U_eq of 5,000 random library entries measured at
283–303 K, applied to every pattern whatever its temperature. (b) For the 68 patterns not measured with Cu
Kα, the Lorentz–polarisation factor converted from the measurement to Cu Kα1, with polarisation
(1 + cos² 2θ)/2 for laboratory tubes and none for synchrotrons, the two patterns at other wavelengths being
treated as synchrotron data. No polarisation holds for a vertical scattering plane with a point detector;
area-detector data integrated without a polarisation correction would carry the laboratory form. (c) For the 125 laboratory-tube patterns, Kα2 as a nuisance: the score is the higher of
those against a Kα1 library and a Kα1 + Kα2 library (I₂/I₁ = 0.5), because opXRD does not record whether
the beam was monochromated. (d) For the 46 patterns that do not cover 5–50°, comparison over the measured
range only (reference norms over the query's coverage, not shifted with the zero scan). Each change was also
removed on its own to give one-off ablations. None of the four was selected after seeing results. Changes
(b) and (d) act on the queries rather than on the library.

### 3.8 Variable-cell re-rank

Single-crystal references fail mainly when their cell differs from the measured one (Section 4.2), which
variable-cell comparison is designed to absorb. We re-ranked shortlists with VC-GPWDF
[@oterodelaroza2024vcgpwdf] as implemented in critic2 1.4.155 [@oterodelaroza2014critic2], which compares a
candidate structure with a measured pattern while varying the candidate's cell (±10 % in lengths and ±5° in
angles in its quick mode, so larger cell differences cannot be reached). We ran the published pipeline:
background fit with 20 knots, peak fit with the default settings, and the global quick comparison with α = 1
as in the paper (the code's default is 0.5) and the measurement wavelength; it applies no zero-offset
correction, and the peak fit does not model Kα2. One step was added: each pattern was cut to the d-spacing
window that the other scores see (Cu Kα1 2θ of 5–50°). The shortlist of each single-crystal pattern is the
50 best entries of ln q, with set-aside entries excluded; a pattern whose answer is not in its shortlist
counts as a miss, and a comparison that fails or runs past 4 hours of wall time counts as maximally
dissimilar. Entries with a disorder model are given as their major component; the first run passed every
component at full occupancy, which affected one answer and is corrected here (ESI†). We had planned a shortlist of 100, but the shortlist holds the answer for the same 36
of the 59 patterns at 50 as at 100, so the further entries could only have added competitors; the shorter
list is, if anything, favourable to the re-rank. Because the cost of a comparison was first unknown, we ran
the top 10 before extending to 50 (ESI†). A comparison took a median of 2.2 CPU-minutes and up to 3.0
CPU-hours for the largest cells; one of the 2,950 comparisons reached the limit, in a shortlist that does
not hold the answer, so it changes no result. Two checks were added after an internal review (ESI†): a
positive control, in which each pattern is compared with its own powder-study entry, the structure refined
against that very pattern; and an injection test, in which each answer missing from a shortlist is compared
with its pattern to see where it would rank.

## 4. Results

### 4.1 Simulated queries: the model loses in every condition

Across the 34 conditions the model scores 0.47–0.57 within-family top-1 and the best classical score
0.64–1.00 (Fig. 2). The best classical score leads by 0.12 to 0.48, and the 95 % CIs never overlap. The
result does not depend on picking the best score per condition: WCC (each of l = 0.5°, 1° and 2°) and the
ln q scan beat the model with non-overlapping CIs in all 34 conditions, and Pearson + zero scan in 32.

The fixed-grid Pearson baseline of our earlier work behaves as reported there: it falls to chance by a
0.15° zero offset or a 0.8 % strain, while the model is flat (Fig. 2a and b). The zero scan removes that
weakness: Pearson + zero scan scores 1.00 at every offset that is a multiple of the 0.02° step; the dips at
0.05° and 0.15° are offsets that fall between grid points. The ln q scan, whose offsets are 0.04° apart,
also dips at 0.1° and 0.3° (0.88 at both). It holds 0.89 at 1.2 % isotropic strain. The model's curves are
flat, but they are flat at about 0.55; its clean within-family top-1 is 0.56. The smallest gap, −0.12, comes
from large anisotropic thermal strain combined with a degraded profile. Even there, WCC (l = 0.5°) leads,
0.64 against 0.52, with WCC (l = 1°) and the ln q scan within 0.01 of it.

![](figs/fig2_simulated_sweep.pdf)

**Fig. 2** Simulated stress test: within-family top-1 over 1,077 polymorph families (2,835 queries).
Bands are 95 % family-bootstrap CIs; the dashed line is composition-only chance (0.38). (a) 2θ zero
offset, (b) isotropic lattice strain, (c) anisotropic thermal strain, (d) crystallite size,
(e) March–Dollase preferred orientation, (f) counting statistics. The l = 0 score is Pearson + zero scan
here (cosine + zero scan on the measured patterns). In (a), the zero-scan scores dip at offsets that fall
between the points of their 0.02° offset grid (0.05° and 0.15°), and the ln q score at offsets off its
0.04° grid (0.05°, 0.1°, 0.15° and 0.3°).

### 4.2 The realistic case: independent single-crystal references

When the pattern's own powder-study entry is set aside and only independent single-crystal determinations
count, the classical scores reach 0.30–0.50 compound-weighted top-1 (Table 1, Fig. 3a) and the model 0.00.
Depending on the classical score, the first correct entry lies beyond rank 1,000 for 34–47 % of these
patterns.

The shift- and strain-tolerant scores all lie above fixed-grid Pearson, but the intervals are wide. Paired
over compounds, and without correction for the four comparisons, WCC (l = 1°) beats
fixed-grid Pearson by 0.20 [0.03, 0.36]; the ln q scan (+0.15 [0.00, 0.31]), cosine + zero scan
(+0.10 [−0.05, 0.24]) and WCC (l = 2°, +0.16 [−0.03, 0.35]) are not resolved. Of the full-library scores, the ln q scan has the best median rank, 7.

What separates success from failure is how well the cells agree, and why they differ. We compare the
primitive Niggli-reduced cells of each pattern's own powder-study entry and of its closest single-crystal
same-form entry (58 patterns; one has no powder-study entry). Where every reduced cell length agrees within
1 %, the ln q scan ranks a single-crystal same-form entry first for 22 of 33 patterns; where the cells differ
by more, for 2 of 25, one of which is a supercell description of the same lattice. Most of the differences
have identifiable causes. 22 of the 59 single-crystal patterns (5 compounds) come from variable-pressure
studies, room-temperature compression series whose single-crystal references were determined at other
pressures, and they account for 16 of the 25 patterns with differing cells. In 7 of the other 9, the
recorded temperatures of the two entries differ by more than 50 K; overall, where both temperatures are
recorded (35 patterns), they differ by more than 50 K for 15, and those 15 are retrieved less often (0.20
against 0.45). Thermal expansion of molecular crystals is generally anisotropic
[@bond2021thermal], and so is their compression; the isotropic ln q lag cannot absorb either, but
variable-cell comparison is designed to.
The pressure-series patterns are not the routine case of a new measurement checked against a database.
Without them (37 patterns, 25 compounds; a split defined after an internal review), the classical scores
reach a compound-weighted top-1 of 0.35–0.60 and the model 0.00 (ESI†).

**Variable-cell re-rank.** Re-ranking the ln q top 50 with VC-GPWDF (Section 3.8) gives a compound-weighted
top-1 of 0.53 [0.37, 0.70], against 0.45 [0.28, 0.63] for the ln q scan on the same patterns; the paired
difference, +0.08 [−0.05, 0.21], does not resolve them, and at top-10 the two are level (0.62 against
0.61). Per pattern, the re-rank ranks the answer first for 27 of the 59 patterns and the ln q scan for 25. Of
the 36 patterns whose shortlist holds the answer, the re-rank moves it to first place for 4 (from ln q ranks
2, 3, 7 and 9) and moves ln q's correct first choice down for 2 (to ranks 2 and 22). Where the cells differ
by more than 1 %, it ranks the answer first for 4 of the 25 patterns, against 2 for the ln q scan; where
they agree, for 22 of 33, against 22. Its reach is limited in two ways. First, the shortlist: for 23 of the
59 patterns the answer is not among the 50 entries re-ranked, and compared directly, only 2 of those 23
answers would rank first. Second, the peak fits. On the pressure-series patterns, measured at short
wavelengths at high pressure, the fit finds 11–49 peaks (median 37), against 66–296 (median 154) on the
others. Compared with its own powder-study entry, the structure refined against the same data, the pipeline
ranks that entry first among the shortlist for 20 of 29 ambient-pressure patterns but for none of the 16
pressure-series patterns (median rank 30.5), so on those patterns the re-rank carries little information.
Without the pressure-series patterns, the re-rank reaches 0.64 [0.46, 0.80] against 0.54 [0.34, 0.72] for
the ln q scan (paired +0.10 [−0.04, 0.26]).

**Table 1** Retrieval of 188 measured opXRD patterns from 309,080 simulated CSD entries
(background-subtracted queries, no hubness correction). Compound-weighted top-1 and top-10, with 95 %
compound-bootstrap CIs for top-1; per-pattern top-1; median rank of the first correct entry over patterns.
The variable-cell re-rank exists for the single-crystal set only; a pattern whose answer is not in its
shortlist counts as a miss, and its median rank falls inside the shortlist. The model's 0.00 [0.00, 0.00]
on the single-crystal set is a degenerate bootstrap interval: none of the 30 compounds is retrieved, and an
exact one-sided 95 % bound is 0.10.

| Score | Single-crystal entry only (30 compounds): top-1 | top-10 | median rank | Any same-form entry (123 compounds): top-1 | top-10 | per-pattern top-1 | median rank |
|---|---|---|---|---|---|---|---|
| VC-GPWDF re-rank of the ln q top 50 | 0.53 [0.37, 0.70] | 0.62 | 3 | – | – | – | – |
| ln q lag + zero scan | 0.45 [0.28, 0.63] | 0.61 | 7 | 0.87 [0.82, 0.93] | 0.93 | 0.72 | 1 |
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
(open squares, 30 compounds); vertical bars mark the value with the CSLS hubness correction. The top row is
the variable-cell re-rank of the ln q top 50, which exists for the single-crystal set only.
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
0.90 → 0.83, WCC l = 2° 0.83 → 0.71 and ln q 0.93 → 0.87. Results on small libraries overstate accuracy at
database scale.

**Within families,** with only the compound's own forms as candidates (chance 0.40), the classical scores
reach 0.94–0.98 top-1 and the model 0.71 (51 patterns).

### 4.4 One compound, twenty patterns, and the hubness correction

**Hubness.** In the probe population, hubness depends on the score. The skewness of N₁₀ over degraded
probes is 18.7 for cosine + zero scan, 12.6 for ln q, 8.0 for Pearson and 3.6 for both WCC widths.

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
(12 → 17) and ln q (0 → 2); for WCC it only exchanges one decoy for another. Per pattern, cosine + zero
scan rises from 0.72 to 0.88. Compound-weighted, the change is +0.01 to +0.04 for the classical scores,
and no 95 % CI of the paired difference excludes zero; gains of up to 0.10 are not excluded (ESI†). For
single-crystal references the change is +0.01 to +0.10, with upper CI bounds up to 0.20, and for the
learned model it is 0.02 → 0.02. We detect no compound-level gain from the correction; it is a cheap option
that removes hub-like confusions when they occur.

### 4.5 Why the learned model fails

The model ranks the first correct entry at a median of about 6,000 of 309,080 (Fig. 5a). Five diagnostics
locate the failure.

**It retrieves simulated patterns.** For the 4,000 clean simulated probes of Section 3.5, the model ranks
the probe's own entry first for 0.42 of probes and within the top 10 for 0.84 (median rank 2); with the
training-time degradations, for 0.28 and 0.67 (median rank 4). For the clean simulated patterns of the
benchmark's own entries, scored with the same answer sets as the measured patterns, it reaches a per-pattern
top-1 of 0.21 and top-10 of 0.56, with a median rank of 5. On the measured patterns of the same targets it
reaches 0.01 and 0.03. (A classical score would rank a clean rendering of a library entry first by
construction, so this test only diagnoses the model.) The model can discriminate among 309,080 entries; what
it cannot do is carry that ability from simulated to measured patterns.

**Seeing the structure in training does not help.** For the 81 patterns that have a same-form entry in the
model's training split, the median rank is 6,396 and top-1 is 0.01; for the 101 patterns whose same-form
entries are all outside the corpus, 5,617 and 0.00 (the other 6 have entries in the validation or test
split).

**No single modelled effect explains the gap.** Rendering each pattern's own entry with the effects of its
measurement (Section 3.4) shows where the model loses ground (Fig. 6). Clean, it reaches a compound-weighted
top-1 of 0.27 with a median rank of 5. The largest single cost is the Debye–Waller factor (0.07, median rank
46), which fixed-grid Pearson does not notice (1.00). Kα2, which we add to every laboratory Cu pattern because
monochromatisation is not recorded, costs 0.18 (median rank 33), and the background with its subtraction
0.24 (median rank 130); coverage, peak width, noise and the zero offset cost little. With all six effects
the model reaches 0.15 (median rank 176), and with the Debye–Waller factor added 0.05 (median rank 744),
against 0.02 and about 6,000 on the measured patterns. With every modelled effect, including intensity
jitter, fixed-grid Pearson reaches 0.83, its value on the measured patterns (0.83), and the model 0.04
against 0.02; the difference shows in the model's median rank, about 750 on those simulations against about
6,000 on the measured patterns. The jitter σ was not estimated from the data, and the Debye–Waller and jitter
rungs were added after the first ladder.

**A retrain with these effects handles them in simulation only.** We retrained the model with Kα2, a
Debye–Waller factor, intensity jitter, range cuts and the background subtraction in its training simulation
(Section 3.4). On its own entries it now tolerates the Debye–Waller factor (0.34 alone, median rank 4; 0.23
together with all six effects, median rank 34 against 744 before); Kα2 alone stays costly (0.11, against 0.18 before,
with overlapping intervals). On the measured patterns the retrained model reaches a compound-weighted top-1 of
0.02 [0.00, 0.06], against 0.02 [0.00, 0.04] before, and 0.00 for single-crystal references; its median rank
falls from about 6,000 to about 3,000. Before training we predicted a measured per-pattern top-1 of
0.05–0.20 and a median rank of 50–1,000, and that a top-1 below 0.03 would mean the remaining gap is not the
modelled effects. Both predictions failed: the per-pattern top-1 is 0.02 and the median rank about 3,000.
The predicted recovery on the simulated patterns held in part (0.20 with every modelled effect, against 0.27
clean). The measured patterns therefore differ from our simulations in a way that barely affects fixed-grid
Pearson correlation but moves this model's embeddings far. We have not identified it. Candidates include
instrument aberrations that a constant offset does not absorb (axial divergence, specimen transparency),
anisotropic broadening and real preferred orientation. One retrain, with one seed and a checkpoint chosen
on clean validation patterns, does not show that no training on these effects would transfer.

**Pattern to pattern.** Matching the measured patterns to the simulated library patterns in the model's
pattern space, without the structure encoder, reaches a compound-weighted top-1 of 0.13 (median rank about
1,600), and 0.21 (about 270) with the retrained encoder. The retrained encoder matches its own entries
rendered with every modelled effect to their clean versions at 0.92. Fixed-grid Pearson, which like the
encoder compares the two patterns point by point without an alignment search, does the measured task at
0.83. The pattern encoder itself, not only the step to the structure encoder, places measured patterns away
from simulations of the same structures.

Within families, where it only has to separate a compound's forms, the model reaches 0.71 against
0.94–0.98 for the classical scores. As a shortlist it caps any re-ranker: its top 1,000 contains a correct
entry for only 0.28 of patterns, and re-ranking that list with a classical score gives 0.19–0.26 per-pattern
top-1, against 0.57–0.82 for the same scores on the full library.

![](figs/fig5_rank_distributions.pdf)

**Fig. 5** Cumulative distribution of the rank of the first correct entry over the full library when
(a) any same-form entry is correct and (b) only independent single-crystal entries are correct.
Background-subtracted queries, no hubness correction.

![](figs/fig6_effect_ladder.pdf)

**Fig. 6** Effect ladder. Each pattern's own library entry is rendered cleanly, with one effect of its
measurement at a time, with all six, and with an overall Debye–Waller factor (B), log-normal intensity
jitter per reflection (σ = 0.3 or 0.5) or both added; the last point is the measured patterns themselves
(Section 3.4). Answer set: any same-form entry (188 patterns, 123 compounds). (a) Compound-weighted top-1 for
fixed-grid Pearson, the original model and the model retrained with these effects; (b) median rank of the
first correct entry for the two models.

### 4.6 A library closer to the measurements

With the package of Section 3.7, the classical scores change by −0.04 to +0.04 in compound-weighted top-1,
paired against the main benchmark (Table 2). Only one change is resolved: cosine + zero scan rises by
0.04 [0.01, 0.08] when any same-form entry counts. The one-off ablations, each change removed alone and
paired against the full package, point to the Kα2 nuisance for that gain (−0.03 [−0.07, −0.01] without it)
and show that restricting the comparison to the measured range costs WCC (l = 2°) when any same-form entry
counts (+0.05 [0.02, 0.09] without it). Three of these 50 intervals exclude zero, about as many as chance
would give, so we do not attribute effects to single changes. We had expected every single-crystal score to
rise; they move by −0.04 to +0.04, with every interval including zero. Two of the changes act on the queries
rather than the library (the Lorentz–polarisation conversion and the measured range), which is why the
learned model, whose library embeddings are unchanged, scores 0.00 with the package on both sets. The
classical results are robust to these choices of library simulation, which do not explain the single-crystal
failures.

### 4.7 The open replica

Against the 108,902 COD entries, the 190 patterns reproduce the main result when any same-form entry counts
(Table 2): the classical scores reach a compound-weighted top-1 of 0.77–0.90 (125 compounds; ln q
0.90 [0.85, 0.95]) and the model 0.02 [0.00, 0.04]. The single-crystal set has 35 patterns but only 9
compounds; there the classical scores reach 0.26–0.58 and the model 0.00. This set cannot test the CSD
single-crystal result: 20 of its 35 patterns are ZZZUWK, scored against a 1950 room-temperature structure
that the CSD benchmark sets aside as not comparable, 8 are the naphthalene compression series, and the other
compounds have one pattern each; its 5 % cell window also leaves out, by construction, the large cell
differences that dominate the CSD failures (Section 4.2). The COD library is about a third of the size of
the CSD library, and smaller libraries favour every score (Fig. 3b). The classical numbers in this section
can be regenerated from open data alone; the model's need the CSD-trained checkpoint, which is not
distributed, and can only be checked against the released ranks.

**Table 2** Compound-weighted top-1 on the main benchmark (CSD library, Table 1), with the library of
Section 3.7 (CSD, realistic) and on the open replica (COD library, Section 2.6). The single-crystal set has
30 compounds in the CSD and 9 in the COD; the any-same-form set has 123 and 125. Intervals are in the ESI.†

| Score | Single-crystal: CSD | CSD, realistic | COD | Any same-form: CSD | CSD, realistic | COD |
|---|---|---|---|---|---|---|
| ln q lag + zero scan | 0.45 | 0.49 | 0.56 | 0.87 | 0.86 | 0.90 |
| WCC, l = 1° + zero scan | 0.50 | 0.52 | 0.56 | 0.83 | 0.81 | 0.88 |
| WCC, l = 2° + zero scan | 0.47 | 0.50 | 0.44 | 0.71 | 0.69 | 0.77 |
| Cosine + zero scan | 0.40 | 0.44 | 0.58 | 0.84 | 0.88 | 0.89 |
| Pearson, fixed grid | 0.30 | 0.27 | 0.26 | 0.83 | 0.81 | 0.87 |
| Learned embedding | 0.00 | 0.00 | 0.00 | 0.02 | 0.00 | 0.02 |

## 5. Discussion

**What a learned retrieval method has to beat.** On measured organic patterns searched against the CSD,
the bar is set by simple classical scores on background-subtracted patterns. When the reference is an
independent single-crystal structure, the shift- and strain-tolerant scores and the variable-cell re-rank
reach a compound-weighted top-1 of 0.40–0.53 (0.47–0.64 without the pressure series); when the pattern's
own powder-study entry is in the library, they reach up to 0.87, and even fixed-grid Pearson correlation
reaches 0.83. We suggest that learned PXRD retrieval be reported (i) on measured patterns, (ii) against a
library of realistic size, (iii) against shift- and strain-tolerant correlation scores with background
subtraction, (iv) with compound-weighted metrics and a compound-level bootstrap whenever compounds repeat,
and (v) separately for queries whose reference is an independent determination, with their measurement
conditions recorded. The benchmark released with this paper supports all five, and its open replica lets
the any-same-form results be checked without a CSD licence.

**Speed is the model's one advantage.** With precomputed library embeddings, the model searches 309,080
entries in about 5 ms per query on a CPU. From the per-pair costs of the simulated sweep, one CPU core
needs about 0.4 s per query for fixed-grid Pearson, about 20 s for a zero scan and about 6 min for the ln q
scan. On one GPU, all five classical scores of 8,188 patterns against the full library took 761 s, and the
corrected ln q scan alone 405 s. The variable-cell re-rank is much slower: a median of 2.2 CPU-minutes per
comparison and about 400 CPU-hours for the 2,950 comparisons of 59 shortlists. Speed only matters once
accuracy does, and a shortlist from this model loses most correct entries (Section 4.5).

**Where learning could still help.** Two problems remain open, and both concern the gap between simulated
and measured patterns. For independent single-crystal references, retrieval fails when the reference was
determined at another pressure or temperature and the cells differ anisotropically by more than about 1 %.
The published variable-cell method [@oterodelaroza2024vcgpwdf] gives no resolved gain here, is bounded
by the shortlist it re-ranks, and in its automated form fits peaks poorly on the high-pressure patterns that
make up most of the mismatched cells, so this case stays open for classical methods too, and it is where a
learned method could show its value. For the learned model, the failure is the transfer from simulated to measured patterns,
not the size of the library, and it survives a retrain with the five modelled effects of Section 3.4.
Full-profile generative models have shown the same sensitivity to how realistic the training patterns are
[@segal2026xrdiff]. A learned method aimed at that transfer, for example one adapted with measured patterns
rather than with more simulated effects, is the more promising use of learning than a stand-alone
retriever.

**Limitations.** We tested one learned model, trained once and retrained once with a more realistic
simulation, each with one seed and a checkpoint chosen on clean validation patterns; another architecture,
loss or training strategy could do better, and our conclusions about learning are about this model. The
benchmark has 123 compounds, all organic, from one contributor to opXRD, and most patterns are the data
their own entry was refined against. The single-crystal set has 30 compounds, so its intervals are wide;
22 of its 59 patterns come from variable-pressure studies, and the open replica's single-crystal set has only
9 compounds. The main library is simulated with one peak profile and without Debye–Waller factors or Kα2; a
package of such changes moved little (Section 4.6) but does not exhaust them, uses one Debye–Waller factor
for every temperature, and disordered entries are simulated only approximately. The measured patterns have a
simple background removal and are interpolated onto the Cu grid, which can miss the tops of narrow
synchrotron peaks. The zero scans use fixed offset grids (0.02°, and 0.04° for the ln q scan). The effect
ladder adds Kα2 to every laboratory Cu pattern and does not estimate the intensity jitter. The variable-cell
re-rank covers a shortlist of 50 with a 4-hour limit per comparison, uses no zero-offset correction, reaches
only cell differences within ±10 % and ±5°, and fits peaks poorly on the high-pressure synchrotron patterns.
The correct answers rely on packing similarity, which can count entries as one form across a phase
transition that preserves the packing, set aside entries that cannot be
compared, and contain one known error (Section 2.2). Retrieval identifies known forms only; crystal
structure prediction combined with pattern matching can identify forms that no database contains
[@glick2025csp].

## 6. Conclusions

On 188 measured organic patterns searched against 309,080 simulated CSD entries, a contrastive
PXRD–structure model reaches a compound-weighted top-1 of 0.02, against 0.71–0.87 for classical profile
matching; with independent single-crystal references, 0.00 against 0.30–0.50, and 0.53 for a
variable-cell re-rank, which is not resolved from the ln q scan it re-ranks. The model retrieves clean
simulated patterns of the same targets, and a retrain with five modelled measurement effects handles those
effects in simulation but not on the measured patterns, so its failure lies in differences between
measured and simulated patterns that our simulation does not reproduce. A more realistic library leaves the
classical results unchanged, and an open replica built from the COD reproduces the any-same-form result.
The hard open case is a measured pattern whose only reference was determined at another temperature or
pressure, so that the cells differ anisotropically by more than about 1 %; in this benchmark most such
patterns come from compression series. We release the benchmark so that learned methods can be compared
with these numbers on measured data.

## Data availability

<!-- [FRANK] GitHub URL and Zenodo DOI after the deposit (tasks/todo.md, after Step C). Digital Discovery
asks, for restricted data, for "a representative dataset that enables similar conclusions": the COD replica. -->

The code, the benchmark definition and all results are available at [GitHub URL] and archived at Zenodo
([DOI]). The deposit contains: the opXRD pattern identifiers with their CSD refcodes, same-form sets and
packing-similarity results; the corpus refcode list and the training splits; the per-pattern ranks of
every score and correction, including the learned model's and the variable-cell re-rank's; the
critic2 inputs derived from the opXRD patterns (cut patterns, background and peak fits, command files) and
the measurement conditions of the single-crystal patterns; the scripts that build the library from a CSD
installation, compute the scores (CPU and GPU) and regenerate every table and figure; and
results/paper_numbers.json, from which every number in this article is checked by scripts/verify_claims.py.
The open replica of Section 2.6 is included in full: the COD identifiers, the simulated COD library, its
answer sets and the ranks of every score, from which the classical results of Section 4.7 and the COD
columns of Table 2 can be reproduced without a CSD licence. The CSD library was built from the May 2026
release (1,451,367 entries; CSD Python API 3.7.1) and the measurement conditions were read from a later
update (1,469,017 entries; API 3.8.0); the COD was downloaded on 12–13 October 2026; the opXRD patterns are
from the Zenodo record 10.5281/zenodo.15298026. The measured patterns are from opXRD [@opxrd2025zenodo] under CC BY
4.0, and the COD structures are CC0. CSD structures and the simulated patterns derived from them cannot be
redistributed; users with a CSD licence regenerate them from the refcode lists. The model checkpoints, the
original and the retrained one, were trained on CSD data and are not distributed.

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
