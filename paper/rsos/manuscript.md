---
title: "Retrieving organic crystal structures from measured powder diffraction patterns: a benchmark of classical profile matching and a symmetry-aware contrastive model"
bibliography: refs_rsos.bib
csl: royal-society-open-science.csl
link-citations: true
---

<!-- RSOS DRAFT 2026-11-19 (resubmission after RSC Advances RA-ART-10-2026-010272; tasks/todo.md D-rest F4).
Every number is checked by scripts/verify_claims_rsos.py against results/rsos_numbers.json (scripts/48).
{{name}} = number still being computed; none may remain at build. [FRANK] marks his decisions. -->

Frank Cai

Purdue University, West Lafayette, Indiana, USA. E-mail: cai485@purdue.edu. ORCID: 0009-0003-0041-1459
<!-- [FRANK] Purdue e-mail as in the RSC Advances submission; confirm for RSOS. -->

## Abstract

Contrastive models that embed powder X-ray diffraction patterns and crystal structures in a shared space are
proposed for structure retrieval but mostly evaluated on simulated patterns. We built a benchmark of 188
measured patterns of 123 organic compounds, identified each pattern's crystal form in the Cambridge Structural
Database by packing similarity, and ranked every pattern against 309,080 simulated entries. When the
correct answer must be an independent single-crystal structure determined at ambient pressure (38 patterns of 26
compounds), shift-tolerant classical profile scores rank it first for a compound-weighted fraction of 0.45–0.62. A
contrastive model with a space-group-aware structure encoder, trained with every measurement effect we could
simulate, ranks it first for none of these compounds. When the entry refined against the pattern itself also
counts, the classical scores reach 0.71–0.88 and the model reaches 0.02.
Simulations of these patterns with their own measurement effects are retrieved by the model at a median rank of 8,
whereas the measured patterns sit at a median rank of about 5,600 (4 on an external set of 19 profiles), so on these
patterns it fails on differences our simulation does not reproduce. We release the benchmark, an open replica and every
per-pattern rank.

**Keywords:** powder X-ray diffraction; crystal structure retrieval; search–match; contrastive learning;
Cambridge Structural Database; polymorphism

## 1. Introduction

Powder X-ray diffraction (PXRD) is the routine way to identify which crystalline phase a sample contains and, for
molecular materials, which polymorph. The standard procedure is search–match: the measured pattern is compared with
reference patterns, measured or simulated, from a database such as the Powder Diffraction File
[@gatesrector2019pdf] or the Cambridge Structural Database (CSD) [@groom2016csd]. For organic crystals the form
matters as much as the compound, because polymorphs can differ in solubility and stability, and the late appearance
of a less soluble form once interrupted the supply of a marketed drug [@bauer2001ritonavir; @cruzcabeza2015polymorph].

Machine learning offers another route, in which contrastive models train a pattern encoder and a structure encoder so
that a pattern lands near the structure that produced it, and retrieval becomes a nearest-neighbour search
[@lai2025xtalnet; @xu2026xccp; @yang2026pxtal]. Generative models instead build the structure from the pattern
[@johansen2026decifer; @segal2026xrdiff]. Both kinds are trained on simulated patterns, mostly of inorganic or
framework materials, and are evaluated mostly on simulated patterns. Tests on measured data are recent and limited:
two framework case studies [@lai2025xtalnet], a zero-shot test on 773 inorganic patterns from the opXRD database
[@xu2026xccp] and four inorganic samples for a generative model [@johansen2026realworld]. Where learned and simple
methods have been compared on measured patterns, the simple methods have held up. On measured patterns that are
mostly minerals and inorganic or hybrid materials, a model-free correlation reached 82.8 % recall at 10 against
34.4 % for a contrastive model trained on corrupted simulated patterns [@srivastava2026realbind]. On experimental
patterns of pharmaceutical solid forms, de Gelder's weighted cross-correlation gave a better accuracy than a Siamese
network, although the network ranked better [@lange2024smolnet]. A hand-crafted peak-feature index outperformed
element-constrained commercial search–match on measured mineral patterns [@wang2026matdiffract], and a classifier
trained on synthetic crystals lost much of its accuracy on measured diffractograms [@schopmans2023synthetic]. In other
fields too, simple heuristics often outperformed the neural recommender systems that could be reproduced
[@ferraridacrema2019progress], a fair comparison left little of the reported progress in deep metric
learning [@musgrave2020reality], and in a mass-spectrometry benchmark ranking candidates by their simulated spectra
beat the best spectrum-to-fingerprint retrieval model [@bushuiev2024massspecgym].

We previously trained a contrastive model of this kind on molecular crystals from the CSD [@cai2026urtc]. Within
families of polymorphs it was more robust to a 2θ zero offset and to lattice strain than fixed-grid Pearson
correlation and a peak-list match. Neither of those baselines can absorb a peak shift, however, so the comparison
says little about the methods crystallographers actually use. A scan over the 2θ zero offset, de Gelder's
triangle-weighted cross-correlation [@degelder2001wcc] and a lag scan on a ln q axis each absorb a shift or an
isotropic strain at little cost, and variable-cell methods also absorb anisotropic strain [@mayo2022vcpwdf;
@mayo2023vcxpwdf; @oterodelaroza2024vcgpwdf].

This paper asks whether such a model beats these methods, first on simulated and then on measured patterns. Because
a weak model would settle little, we tested the strongest version of our model that we could build: a structure
encoder that sees the space group through its symmetry operators, a training simulation that includes every
measurement effect we could model, checkpoints chosen on degraded rather than clean validation patterns, and two
training seeds. Our conclusions about learning are conclusions about this model. The main contribution is a
benchmark for organic PXRD retrieval on measured patterns: 188 patterns from opXRD [@hollarek2025opxrd] whose
correct answers in the CSD are fixed by packing similarity [@chisholm2005compack; @macrae2008mercury], scored against
a library of 309,080 simulated entries under answer sets that separate a realistic search from self-retrieval. It is
released as code and refcode lists, because CSD structures cannot be redistributed, together with an open replica
built from the Crystallography Open Database (COD). On this benchmark we compare the model with shift-tolerant
classical scores and with a published variable-cell method, and we locate the model's failure with a ladder of
simulated measurement effects.

## 2. Benchmark

### 2.1 Measured patterns

The opXRD database collects experimental powder patterns with their labelled phases (CC BY 4.0; the source data
from the COD are CC0) [@hollarek2025opxrd; @opxrd2025zenodo]. Its CNRS contribution holds 1,052 patterns taken from
COD entries that deposit powder data, most with the refined structure as a full atom list. We kept the patterns
whose phase contains C and H and no element outside {C, H, N, O, S, P, F, Cl, Br, I, B, Si}, which left 196. Of the
188 patterns used in the end (Section 2.2), 120 were measured with Cu Kα radiation, 61 with synchrotron radiation
(0.34–1.30 Å) and 7 at Mo, Co or other wavelengths. opXRD records no wavelength for 3 of the 120, and a fit of the
peak positions to each pattern's deposited cell places all three at Cu Kα (Supplementary Section S1). Every pattern
was converted to a Cu Kα1 2θ axis through its d-spacings and interpolated onto the grid used throughout this work,
5–50° 2θ in 0.02° steps (2,251 points), without removing the Kα2 component. In 31 of the 188 refined structures the
H atoms are omitted.

### 2.2 Correct answers by packing similarity

A cell match alone does not identify a compound, because unrelated compounds with the same elements can have similar
reduced cells. We therefore identified every pattern from its deposited structure (figure 1) in three steps.

First, a CSD reduced-cell search on the deposited structure's primitive Niggli cell (1.5 % in lengths and 2° in
angles, the CCDC defaults) collects every hit with the same non-H element set as a candidate. Second, CCDC packing
similarity [@chisholm2005compack; @macrae2008mercury] compares the deposited structure with each candidate over a
15-molecule shell (distance tolerance 20 %, angle tolerance 20°, H positions ignored), and a candidate that matches 15
of 15 molecules is an anchor. Structures rebuilt from bare coordinates sometimes perceive
bonds differently from curated CSD molecules, so when no candidate passes, a lenient pass ignores bond types, bond
counts, H counts and the smallest components; 154 patterns were anchored in the strict pass and 35 in the lenient
pass. Third, the two anchors closest to the deposited structure are compared with every entry of their CSD refcode
family, CSD against CSD with the strict settings, and an entry that matches 15 of 15
molecules is the same solid form whatever its temperature or pressure. Entries that cannot be compared are excluded
from ranking and are never scored as errors (134 entries, affecting 57 patterns).

![](figs/fig1_benchmark.pdf)

**Figure 1.** Construction of the benchmark. Top: each measured pattern is identified from its own deposited
structure. CSD candidates from a reduced-cell search are compared with that structure by packing similarity, and the
matching anchor entries are compared with their refcode family to collect every entry of the same solid form.
Bottom: the measured pattern, on a Cu Kα1 2θ axis with Kα2 retained, is scored against all 309,080 simulated library
patterns, and the rank of the first correct entry is recorded under each answer set.

Of the 196 patterns, 189 were identified; the other seven are three Si frameworks or silsesquioxanes that passed the
element filter and four patterns with no packing match. One identified pattern records its 2θ axis in radians (0.01
to 0.83) and was excluded, which leaves 188 patterns of 123 compounds. Most compounds have one pattern (110 of 123),
but a few have many: ZZZUWK has 20, a variable-temperature series, ZZZIZM 12 and NAPHTA 10.

Three checks test these labels (Supplementary Section S2). Because same-form sets were built within the anchors'
refcode family, an entry of the same form deposited under another refcode would be scored as an error. We therefore
compared every wrong first-ranked entry of the classical scores (455 cases) with the pattern's anchors, and one
family, GOFWUV, matches the anchor of one QQQAUJ pattern 15 of 15; it is added to that pattern's answers in every
result reported here. Comparing each family member with every anchor, rather than the two closest, adds one entry
(EPIVEF, for one pattern), which matches the third anchor 15 of 15 at an RMSD of 0.50 Å but the two closest anchors
only 11 of 15; we keep the labels as built and report the variant as a sensitivity check. Finally, in all 35
lenient patterns the deposited structure and its anchors contain the same molecular components, so the lenient pass
never pairs a solvate with an anhydrate or a salt with its free form. The relaxation absorbs omitted H atoms, bond
perception and charge assignment in salts, and disorder sites listed at full occupancy; for 20 of the 35 patterns,
all from the H-omitted ZZZUWK series, ignoring bond counts alone is enough.

### 2.3 Answer sets

For 184 of the 188 patterns the CSD contains a powder-study entry that reproduces the deposited structure to within
0.05 Å RMSD, so the pattern is the data that entry was refined against [@rietveld1969], and we call this entry the
pattern's self-entry. Finding the self-entry tests whether a method can read a measured pattern, but it is close to
self-retrieval, so we score three answer sets.

- **Independent single-crystal entries at ambient pressure** (38 patterns, 26 compounds). Same-form entries that are
  powder studies are excluded, so the answer must be an independent single-crystal determination, and patterns from
  variable-pressure studies are reported separately. This is the realistic case of a new measurement checked against
  a database, and it is our headline.
- **Pressure series** (22 patterns, 5 compounds). Patterns from room-temperature compression studies whose
  single-crystal references were determined at other pressures, scored like the headline set. Together the two sets
  form the single-crystal set (60 patterns, 31 compounds).
- **Any same-form entry** (188 patterns, 123 compounds). Every entry in the pattern's same-form set is correct,
  including the self-entry. This is an upper bound on what a method can do with these patterns.

We also report within-family retrieval, the rank of the correct form among the compound's own forms in the library
(51 patterns of 35 compounds; chance 0.40), which isolates the polymorph-identification question.

### 2.4 Library

The library holds 309,080 simulated patterns: the 308,862 molecular crystals of the corpus of ref. [@cai2026urtc]
(organic CSD entries without disorder, R ≤ 7.5 %, Z′ ≤ 2, after removing duplicate redeterminations) and 218
same-form or polymorph entries of the benchmark compounds that the corpus filter had removed; 28 further entries
have no coordinates in the CSD. We did not use the whole CSD, because its inorganic and metal–organic entries cannot
be the answer to an organic query and the corpus filter keeps the organic search space that a crystallographer would
search. Stick patterns (Cu Kα1) were computed with the XRDCalculator module of pymatgen [@ong2013pymatgen], and
recomputing 296 randomly drawn entries with pymatgen 2026.4.16 reproduces the stored sticks exactly. The sticks were
rendered with one pseudo-Voigt profile (Caglioti-type width, U = 0.002, V = −0.001, W = 0.004 deg², η = 0.5)
[@caglioti1958], without Debye–Waller factors, texture, background or noise; a library closer to the measurements is
tested in Section 4.7. Disordered entries whose CIF fails a strict parse have coincident sites merged by rescaling
occupancies, and the self-entry of 28 patterns (9 compounds) is disordered.

### 2.5 Metrics

For each pattern and score, the rank of the first correct entry is 1 plus the number of entries that are neither
correct nor excluded and that score strictly higher than the best correct entry, so ties favour the correct entry.
Top-k is the fraction of patterns with rank ≤ k. Because some compounds contribute many patterns, per-pattern
averages mostly measure those compounds, and our headline metric is the compound-weighted top-k, the mean over
compounds of each compound's top-k rate. Confidence intervals (95 %) come from 2,000 bootstrap resamples of compounds
[@field2007cluster], so the patterns of one compound move together, and for each answer set only its own compounds
are resampled. Differences between scores and the effect of a correction are paired, both being computed on the same
resampled compounds. Our primary comparison is the learned model against the classical scores in compound-weighted
top-1. Comparisons among classical scores and among the variants of Section 4.7 are exploratory and are not
corrected for multiple comparisons; where an interval includes zero we call the two not statistically
distinguishable.

### 2.6 An open replica

The CSD licence does not allow the library or the answer sets to be redistributed. So that the results can be
checked without a licence, we rebuilt the benchmark from the COD (CC0) [@grazulis2009cod; @vaitkus2021cod], the source
of the opXRD patterns themselves. The queries are unchanged, and the library applies the corpus filter of Section
2.4 to the COD metadata (C-containing entries with coordinates, no disorder flag, R ≤ 7.5 %, Z′ of 1–2, no element outside
H, C, N, O, F, S, Cl, Br, I and P), which keeps 108,716 entries, and adds 854 candidate entries of the queries that the
filter removes. Of these 109,570 entries, 108,902 gave stick patterns, computed and rendered as for the CSD library.

The answer sets follow Section 2.2 with two changes. The COD has no refcode families, so candidates are COD entries
with the same non-H element set whose primitive reduced cell lies within 5 % of the deposited structure's in each
length and within 6 % in volume, a wider window than in the CSD search so that temperature series are not lost; the
two closest anchors are then compared with every other candidate to give the same-form set. The COD candidates also
include other compounds, which the CSD family search never returned. A candidate whose reduced non-H formula differs
from that of every same-form entry stays an ordinary library entry and counts as an error when it ranks above the
answer, because excluding the 2,670 such candidates would remove the hardest decoys and flatter every score.
Same-formula candidates that cannot be compared (23) are excluded as before, and powder or single-crystal is read from
the COD method field, else from the powder tags of the CIF. These rules were fixed before any replica result was
computed. Of the 196 patterns, 191 are identified in the COD, all 189 identified in the CSD and two more, and 190
patterns of 125 compounds are scored. Only 35 patterns of 9 compounds have a single-crystal same-form entry, so the
replica tests the any-same-form result. The replica is scored with the classical scores of Section 3.2 and the
model, without the corrections of Sections 3.6 and 3.8.

## 3. Methods

### 3.1 Preprocessing

The measured patterns were scored in two forms: raw, shifted to a zero minimum and scaled to a unit maximum, and
background-subtracted, in which a rolling minimum over 2° smoothed by a 2° moving average is removed before scaling;
library patterns are scored as simulated. The conversion to the Cu axis moves peak positions but not intensities, so
for the 68 patterns not measured with Cu Kα it leaves the Lorentz–polarisation factor of the original wavelength and
the intensity scale of the original axis. We tested this convention by rescoring every pattern with the
Lorentz–polarisation factor converted to Cu Kα1 and the intensities divided by the Jacobian of the axis change, which
conserves peak areas (Section 4.7).

### 3.2 Classical scores

Five classical scores were computed for every pattern against the whole library.

- **Pearson, fixed grid.** The Pearson correlation of the two patterns on the common grid, the main baseline of our
  earlier work [@cai2026urtc].
- **Zero scan.** Each score below is maximised over a constant 2θ offset of ±0.5° in 0.02° steps. This absorbs a
  zero error exactly and a sample-displacement error, which varies as cos θ, approximately. For a pattern converted
  from a short wavelength, a native zero error is magnified by about λ_Cu/λ (up to about 4.5 times) and becomes weakly
  angle-dependent on the Cu axis. Specimen transparency, which shifts peaks in proportion to sin 2θ and is often the
  largest aberration for weakly absorbing organic samples in reflection geometry, is not absorbed by a constant offset.
- **WCC.** de Gelder's weighted cross-correlation [@degelder2001wcc] with a triangular weight of half-width l = 1° or
  2°, which makes the score degrade gradually rather than abruptly as peaks shift, plus the zero scan. With l = 0 the
  weight is a delta and the score reduces to the cosine of the two patterns, which we call cosine + zero scan. These
  widths are large compared with the simulated peak width (full width at half maximum about 0.06°); the simulated
  sweep also includes l = 0.5°.
- **ln q lag scan.** Both patterns are resampled onto a ln q axis with 2,251 points, on which an isotropic lattice
  strain is a translation. Each ln q bin takes the mean of the pattern over its width, which keeps peak areas where
  the bins are wider than the 2θ step. The score is a WCC with l = 0.01 in ln q, maximised over lags up to ±3 % strain
  and over 17 zero offsets from −0.32° to +0.32° in 0.04° steps, applied before resampling. An earlier version of this
  score sampled the ln q axis at points and used 16 offsets that skipped zero, and it is reported in Supplementary
  Section S9.

In the simulated sweep the l = 0 score was applied to standardised patterns, which makes it Pearson + zero scan; on
the measured patterns it is cosine + zero scan.

### 3.3 Exact scores on a GPU

The full-library ranks were first computed on CPUs. The hubness analysis (Section 4.5) needs every score rather than
only ranks, so we ported the scores to PyTorch [@paszke2019pytorch]. Folding the triangle weight into the reference
turns every zero offset into one matrix product: for weights w_k and shift d, Σ_k w_k Σ_x q(x) r(x + d + k) =
Σ_x q(x) r_w(x + d), with r_w the reference convolved with w. On one RTX 4090 the scores of 8,188 rows (benchmark
patterns and probes) against all 309,080 entries took 761 s, and the GPU scores agree with the CPU scores to
4 × 10⁻⁶ with no change of any top-1 (Supplementary Section S5).

### 3.4 Learned model

The model is a dual encoder with 6.2 million parameters (figure 2). A one-dimensional residual convolutional network
encodes the standardised pattern on the grid above. A Transformer encodes the structure from three kinds of token.
One lattice token carries the logarithms of the cell lengths and volume, the cosines of the cell angles and the atom
count. Each atom of the asymmetric unit gives one token, an element embedding plus the sine and cosine of 2π times
each fractional coordinate. Each symmetry operator x′ = Rx + t of the entry's CIF gives one token, the nine entries of
R plus the sine and cosine of 2πt. The operator tokens are new in this work. The asymmetric unit and the cell alone do
not determine a pattern, because systematic absences, multiplicities and the number of molecules in the cell all follow
from the space group, and an encoder without the operators cannot know them. Both encoders map to a 256-dimensional
unit vector, and a pattern is scored against the library by the cosine between its pattern embedding and each entry's
structure embedding.

![](figs/fig_model.pdf)

**Figure 2.** The learned model. The pattern encoder maps a powder pattern to a unit vector z_p; the structure
encoder maps one lattice token, one token per asymmetric-unit atom and one token per symmetry operator to a unit
vector z_s. Training pulls the two embeddings of the same crystal together against the other crystals in a batch;
retrieval ranks every library entry by their cosine similarity s. The operator tokens (orange) are added in this work.

The model was trained for 60 epochs with a symmetric InfoNCE loss [@oord2018cpc; @radford2021clip] (batch size 512,
AdamW, learning rate 3 × 10⁻⁴, weight decay 0.05, 3 warm-up epochs), with the sibling polymorphs of each family placed
in the same batch as hard negatives, on 247,089 CSD structures of a scaffold- and polymorph-family split
[@cai2026urtc]. Patterns were simulated on the fly with a random zero offset, isotropic strain, Caglioti and
crystallite-size broadening, March–Dollase texture [@dollase1986] averaged over the symmetry-equivalent reflections of
each peak, background and Poisson noise. The simulation also included the measurement effects of the benchmark
patterns that we could model. A Kα2 copy of every reflection (probability 0.6, I₂/I₁ = 0.5) keeps its own wavelength
under strain; an overall Debye–Waller factor has B drawn uniformly from 0–8 Å²; log-normal intensity jitter per
reflection has σ drawn uniformly from 0–0.5; the 2θ range is cut with probability 0.25 (start 5–12°, end 30–50°); and
the background subtraction of the measured patterns is applied to every training pattern. The checkpoint was chosen by in-batch top-1 on a
fixed draw of these degraded patterns for the validation structures, not on clean ones, because a clean criterion
selects against robustness; the best epochs reach 0.864 (seed 0) and 0.862 (seed 1). We report seed 0 as the model
and seed 1 beside it. As an ablation, the same training without the operator tokens reaches 0.842. The earlier model
of ref. [@cai2026urtc], without the operator tokens, the measurement effects and the degraded criterion, is compared
in Supplementary Section S7, which also lists how this version differs from the one we submitted elsewhere.

### 3.5 Diagnostics of the learned model

Four diagnostics probe the model's failure (Section 4.6). The first, an effect ladder, renders each pattern's
self-entry cleanly and then with one effect of the measurement at a time. Three effects are estimated from the measured
pattern: its 2θ coverage, its peak width and its noise level. Kα2 is added to every laboratory Cu pattern, because
opXRD does not record monochromatisation, and the pattern's own background is added and then removed by the background
subtraction of Section 3.1. The zero offset is the one that best aligns the pattern with its self-entry, and the
Debye–Waller factor is an overall one (B = 4.45 Å², from the library, Section 3.8). The self-entry is then rendered with
the first six effects together, and with those six plus the Debye–Waller factor, log-normal intensity jitter per
reflection (σ = 0.3 or 0.5, not estimated from the data), or both (σ = 0.3). Every
variant is scored with the answer sets of the measured pattern. One effect, the zero offset, uses the self-entry, so
the ladder diagnoses the model and is not a retrieval method. The second diagnostic matches patterns to patterns:
the clean library patterns are passed through the model's pattern encoder, so measured patterns are compared with
simulated patterns in the learned space without the structure encoder. The third scores 4,000 library entries
simulated cleanly and 4,000 simulated with the training degradations (the probes of Section 3.6) against the whole
library, which tests whether the model can discriminate at library scale. The fourth splits the measured patterns by
whether their crystal form is in the model's training split. If the model had memorised its training structures,
patterns whose form it was trained on would be retrieved better than the rest, so this split is a control for
overfitting and not a test of generalisation.

### 3.6 Hubness diagnosis and correction

A hub is a library entry that scores highly against many unrelated queries [@radovanovic2010hubs]. We measured
hubness with probe patterns rather than with the benchmark patterns: 4,000 library entries simulated cleanly and
4,000 simulated with the training-time degradations and then background-subtracted like the measured patterns,
drawn only from compounds outside the benchmark, with pairs from the same compound excluded. For each score we report
the skewness of N₁₀, the number of probes that have an entry among their ten best matches. With 4,000 probes and
309,080 entries the mean N₁₀ is 0.13, so these skewness values are not comparable with those in the hubness
literature. The analysis is exploratory, prompted by one entry that was the wrong first answer for many patterns
(Section 4.5), but the correction was fixed before any corrected number was computed: cross-domain similarity local
scaling (CSLS) [@conneau2018muse] on the library side, s′(q, r) = s(q, r) − h₁₀(r)/2, where h₁₀(r) is the mean of
entry r's ten highest scores against the degraded probes. The query-side term of CSLS is the same for every entry and
does not change ranks. As sensitivity checks we report k = 1–100, clean probes, library-side z-normalisation and
mutual proximity [@schnitzer2012mp] (Supplementary Section S3).

### 3.7 Simulated sweep

The sweep reuses the within-family test of ref. [@cai2026urtc]: 1,077 held-out polymorph families with 2,835 query
structures, in which the query is one form's simulated and degraded pattern and the candidates are the clean patterns
of the family's forms (composition-only chance 0.38). In the clean condition the query is the candidate's own
rendering, so every profile score that can align it exactly scores 1.00, and the sweep measures how far each method
falls from that starting point rather than how well it retrieves. Each condition varies one degradation:

- a zero offset up to 0.3°, isotropic strain up to 1.2 %, crystallite size down to 10 nm, or counting noise;
- March–Dollase texture along (001) for every structure, with r from 0.5 to 0.9;
- March–Dollase texture along each structure's own Bravais–Friedel–Donnay–Harker axis, the normal of its largest-d
  allowed reflection [@donnay1937bfdh], for plates (r = 0.5–0.9) and needles (r = 1.1–2.0);
- an anisotropic thermal strain for a temperature change of 50–200 K, with per-axis expansion coefficients drawn so that
  the mean volumetric coefficient (about 1.65 × 10⁻⁴ K⁻¹) matches the CSD survey of ref. [@bond2021thermal];
- or combinations of these with random offsets and strains.

Texture is averaged over the symmetry-equivalent reflections of each peak. Every method scores the same rendered
queries, intervals come from a bootstrap over families, and each condition's random draws are seeded from a checksum
of its name, so the sweep can be regenerated exactly. The sweep has 42 distinct conditions, a zero offset of 0° being
the clean condition.

### 3.8 A library closer to the measurements

The library of Section 2.4 leaves out several features of the measured patterns. As a robustness check, all scores
were recomputed with one package of changes, fixed before scoring. (a) An overall Debye–Waller factor on the library
intensities, I exp(−2Bs²) with s = sin θ/λ and B = 4.45 Å², which is 8π² times the median mean non-H U_eq of 5,000
random library entries measured at 283–303 K, applied to every pattern whatever its temperature. (b) For the 68
patterns not measured with Cu Kα, the Lorentz–polarisation factor converted from the measurement to Cu Kα1, with
polarisation (1 + cos² 2θ)/2 for laboratory tubes and none for synchrotrons. (c) For the 125 laboratory-tube
patterns, Kα2 as a nuisance: the score is the higher of those against a Kα1 library and a Kα1 + Kα2 library
(I₂/I₁ = 0.5), because opXRD does not record whether the beam was monochromated. (d) For the 46 patterns that do not
cover 5–50°, comparison over the measured range only. Each change was also removed on its own to give one-off
ablations. Separately, two practitioner-style variants were scored: every library entry re-rendered at 295 K with an
isotropic strain from its recorded temperature and the mean expansion coefficient of ref. [@bond2021thermal], and the
measured patterns put on the Cu grid by averaging over each grid step instead of interpolating, which keeps the area
of narrow synchrotron peaks.

### 3.9 Variable-cell re-rank

Single-crystal references fail mainly when their cell differs from the measured one (Section 4.2), which
variable-cell comparison is designed to absorb. We re-ranked shortlists with VC-GPWDF [@oterodelaroza2024vcgpwdf] as
implemented in critic2 1.4.155 [@oterodelaroza2014critic2], which compares a candidate structure with a measured
pattern while varying the candidate's cell (±10 % in lengths and ±5° in angles in its quick mode). We ran the
published pipeline: background fit with 20 knots, peak fit with the default settings, and the global quick comparison
with α = 1, as in the paper, and the measurement wavelength. It applies no zero-offset correction, and the peak fit
does not model Kα2. One step was added: each pattern was cut to the d-spacing window that the other scores see (Cu
Kα1 2θ of 5–50°). The shortlist of each single-crystal pattern is the 50 best entries of the ln q scan, with excluded
entries left out; a pattern whose answer is not in its shortlist counts as a miss, and a comparison that fails or runs
past 4 hours of wall time counts as maximally dissimilar. Entries with a disorder model are given as their major
component. Two checks accompany the re-rank: a positive control, in which each pattern is compared with its own
self-entry, and an injection test, in which each answer missing from a shortlist is compared with its pattern to see
where it would rank. Two variants were run as sensitivity checks: the peaks of each pattern shifted by an answer-blind
zero offset, the one that best aligns the pattern with its ln q top-1 entry, and new fits of the 10 patterns whose
measured range begins with zero-intensity points, after dropping those points (Supplementary Section S10).

## 4. Results

### 4.1 Simulated patterns

Across the 42 distinct conditions of the sweep the model reaches a within-family top-1 of 0.47–0.59 and the best
classical score 0.64–1.00 (figure 3). The best classical score leads in every condition, by 0.10 to 0.47. The result
does not depend on picking the best score per condition, because WCC with l = 0.5° or 1° and the ln q scan beat the
model with non-overlapping 95 % intervals in all 42 conditions, WCC with l = 2° in 41 and Pearson + zero scan in 40.

The fixed-grid Pearson baseline of our earlier work falls to chance by a 0.15° zero offset or a 0.8 % strain, while the
model's curves stay between 0.53 and 0.59, and the model's clean within-family top-1 is 0.59. A zero scan removes
Pearson's weakness: Pearson + zero scan scores 1.00 at every offset that is a multiple of its 0.02° step, and the dips
at 0.05° and 0.15° are offsets that fall between grid points. The ln q scan, whose offsets are 0.04° apart, dips to
0.87–0.88 at every offset off that grid (0.05°, 0.1°, 0.15° and 0.3°) and holds 0.89 at 1.2 % isotropic strain.
Texture lowers the scores that compare peaks over a window rather than point by point. Along each structure's
Bravais–Friedel–Donnay–Harker axis, WCC with l = 1° falls from 1.00 to 0.75 for plates with r = 0.5 and to 0.82 for
needles with r = 2, much as it does along (001) (0.73 at r = 0.5), while fixed-grid Pearson stays at 1.00 and
Pearson + zero scan at 0.98; the model stays at 0.57–0.59 throughout. Even at r = 0.5 every classical score remains
above the model, the lowest being WCC with l = 2° at 0.66. The smallest gap, 0.10, comes from the largest anisotropic
thermal strain (ΔT = 200 K), where WCC with l = 2° reaches 0.67 and the model 0.57.

![](figs/fig3_simulated_sweep.pdf)

**Figure 3.** Simulated stress test: within-family top-1 over 1,077 polymorph families (2,835 queries). Bands are
95 % family-bootstrap intervals; the dashed line is composition-only chance (0.38). (a) 2θ zero offset, (b) isotropic
lattice strain, (c) anisotropic thermal strain, (d) crystallite size, (e) March–Dollase texture along each structure's
Bravais–Friedel–Donnay–Harker axis (plates r < 1, needles r > 1; r = 1 is the clean condition), (f) counting
statistics. The texture conditions along (001) are in Supplementary Section S8. In (a) the zero-scan scores dip at
offsets between the points of their 0.02° grid (0.05° and 0.15°), and the ln q score at offsets off its 0.04° grid.

### 4.2 Independent single-crystal references at ambient pressure

When the self-entry is excluded and only independent single-crystal determinations at ambient pressure count, the
shift-tolerant classical scores reach a compound-weighted top-1 of 0.45–0.62 and fixed-grid Pearson 0.33 (table 1,
figure 4a). The model reaches 0.00, and its second seed {{amb_s1}}. The first correct entry lies beyond rank 1,000 for
{{amb_frac_gt1000_range}} of these patterns under the classical scores and for {{amb_model_frac_gt1000}} under the
model. Paired over compounds, WCC with l = 1° beats fixed-grid Pearson by {{amb_paired_wcc1_pearson}}, and the ln q
scan beats it by {{amb_paired_lnq_pearson}}; cosine + zero scan and WCC with l = 2° are not statistically
distinguishable from it. Among the shift-tolerant scores, only WCC with l = 1° and cosine + zero scan differ
({{amb_paired_wcc1_coszs}}).

Re-ranking the ln q top 50 with VC-GPWDF gives {{c2_amb}}, against {{c2_amb_lnq}} for the ln q scan on the same
patterns, and the paired difference, {{c2_amb_paired}}, does not distinguish them. The re-rank's reach is limited by
its shortlist, which holds the answer for {{c2_amb_in_list}} of the 38 patterns. Compared with its own self-entry, the
structure refined against the same data, the pipeline ranks that entry above every wrong entry of the shortlist for
{{c2_ctl_amb}} ambient patterns that have one.

What separates success from failure is how well the cells agree. We compared the primitive Niggli-reduced cells of
each pattern's self-entry and of its closest single-crystal same-form entry over the whole single-crystal set
({{cm_n}} patterns with a self-entry). Where every reduced cell length agrees within 1 %, the ln q scan ranks a
single-crystal entry first for {{cm_small_hits}} of {{cm_n_small}} patterns; where the cells differ by more, it does
so for {{cm_large_hits}} of {{cm_n_large}}. Most of the differences have identifiable causes: {{cm_large_pressure}} of
the {{cm_n_large}} patterns with differing cells come from the pressure series, and in {{cm_large_dT}} of the others
the recorded temperatures of the two entries differ by more than 50 K. Thermal expansion of molecular crystals is
generally anisotropic [@bond2021thermal], and so is their compression, so the isotropic ln q lag cannot absorb either
difference, although variable-cell comparison is designed to. Rescaling every library entry to 295 K with an isotropic
strain does not help (Section 4.7).

**Table 1.** Retrieval of measured opXRD patterns from 309,080 simulated CSD entries (background-subtracted queries,
no hubness correction). Compound-weighted top-1 with 95 % compound-bootstrap intervals, compound-weighted top-10, and
the median rank of the first correct entry over patterns. The variable-cell re-rank exists for the single-crystal set
only; a pattern whose answer is not in its shortlist counts as a miss. An interval of [0.00, 0.00] is degenerate:
none of the compounds is retrieved, and for 26 compounds an exact one-sided 95 % upper bound is 0.11.

{{table1}}

![](figs/fig4_real_benchmark.pdf)

**Figure 4.** Retrieval of 188 measured patterns from 309,080 simulated CSD entries (background-subtracted queries).
(a) Compound-weighted top-1 with 95 % compound-bootstrap intervals for the ambient single-crystal set (open squares,
26 compounds) and the any-same-form set (filled circles, 123 compounds); vertical bars mark the value with the CSLS
hubness correction. (b) Compound-weighted top-1 against library size (any same-form entry; random decoy subsets; the
standard deviation over five draws is below 0.02).

### 4.3 Pressure series

The 22 pressure-series patterns of 5 compounds are hard for every score. The classical scores reach a
compound-weighted top-1 of {{hp_range}} and the variable-cell re-rank {{hp_c2}}, with the model at {{hp_model}}. These
patterns were measured at high pressure in diamond-anvil cells at short synchrotron wavelengths, and their
single-crystal references were determined at other pressures, so their cells differ anisotropically by more than the
isotropic lag or a constant offset can absorb. The variable-cell re-rank is designed for this case but is limited by
its peak fits: on these patterns the automated fit finds {{c2_peaks_hp}} peaks (median {{c2_peaks_hp_med}}), against
{{c2_peaks_amb}} (median {{c2_peaks_amb_med}}) on the ambient patterns, and the pipeline ranks the self-entry first
for only {{c2_ctl_hp}} of the 22. New fits of the 10 patterns whose measured range begins with zero-intensity points
find 40–60 peaks instead of 11–29 and rank the self-entry first for 7 of them instead of 1, but they change no first
place, because only one of those ten shortlists holds the answer (Supplementary Section S10).

### 4.4 When the self-entry counts

With every same-form entry correct, the classical scores reach a compound-weighted top-1 of 0.71–0.88 (table 1): the
ln q scan 0.88, cosine + zero scan 0.85, fixed-grid Pearson 0.84, WCC with l = 1° 0.84 and with l = 2° 0.71. Paired
over compounds, only WCC with l = 2° differs from Pearson ({{std_paired_wcc2_pearson}}). The model reaches 0.02, with
a median rank of about 5,600, and its second seed reaches {{s1_std}}. Within families, with only the compound's own
forms as candidates (chance 0.40), the classical scores reach 0.94–0.98 top-1 and the model 0.69 (51 patterns).

Background subtraction matters for the classical scores. Without it, compound-weighted top-1 drops to
{{raw_pearson}} for Pearson, {{raw_coszs}} for cosine + zero scan, {{raw_wcc1}} for WCC with l = 1° and {{raw_wcc2}}
for l = 2°. The loss is larger with a triangle weight than without one, presumably because the triangle averages each
comparison over neighbouring points, so a slowly varying background contributes to every overlap. Our background
removal is a simple rolling minimum, and a better one may help further. All classical scores also lose 0.06–0.12 in
compound-weighted top-1 between a 2,000-entry library and the full one (figure 4b), so results on small libraries
overstate accuracy at database scale.

### 4.5 One compound, twenty patterns, and the hubness correction

One entry, GUSRIV, was the wrong first answer of cosine + zero scan for 20 of its {{gusriv_misses}} misses, which looks
like a hub. All 20 are patterns of ZZZUWK, propan-1-aminium chloride, measured as a variable-temperature series, and
each belongs to its own powder-study entry, ZZZUWK01–20. The confusion is a property of this compound rather than of a
hub. The tetragonal salt has 24 reflections between 5° and 50° 2θ, where the median same-form entry of the benchmark
has about 130, and its measured relative intensities differ strongly from those simulated for its self-entry (figure
5a). A March–Dollase texture fitted along each of 49 low-index axes, with r from 0.25 to 4, raises the median Pearson
correlation of the 20 patterns with their self-entries only from 0.72 to 0.79, and the (101) and (202) reflections,
which share a plane normal and so must be scaled alike by any texture, have measured-to-simulated ratios of 0.32 and
2.49. Preferred orientation therefore cannot explain the mismatch. All 20 self-entries are disordered and omit H
atoms, so our simulation of them is the more likely cause. Unrelated entries with strong reflections near ZZZUWK's two
strongest measured peaks, such as GUSRIV and ZZZLWW01, then outscore the self-entry, and the more peak-tolerant the
score, the more such decoys pass it. Fixed-grid Pearson, the least tolerant score, ranks the self-entry first for 12
of the 20 patterns (figure 5b).

![](figs/fig5_case_zzzuwk.pdf)

**Figure 5.** The ZZZUWK case. (a) The measured pattern of propan-1-aminium chloride whose CSD entry, ZZZUWK01, was
refined at 303 K (background removed), the simulated pattern of that entry, and the unrelated entry GUSRIV that cosine
+ zero scan ranks first. (b) Number of the compound's 20 patterns for which each score ranks a same-form entry first,
without and with CSLS.

Hubness itself depends on the score. The skewness of N₁₀ over the degraded probes is 18.7 for cosine + zero scan,
12.6 for the ln q scan, 8.0 for Pearson, 3.6 for both WCC widths and 3.1 for the model. CSLS fixes ZZZUWK for cosine
+ zero scan (0 of 20 → 20 of 20) and raises the compound-weighted top-1 of the classical scores by 0.01–0.04 when any
same-form entry counts and by 0.01–0.08 on the single-crystal set; no 95 % interval of these paired differences
excludes zero. CSLS leaves the model unchanged. We detect no compound-level gain from the correction, which is a
cheap option that removes hub-like confusions when they occur.

### 4.6 Why the learned model fails

The model does not fail for lack of discrimination. For the clean simulated probes of Section 3.5 it ranks the probe's
own entry first for {{probe_sim_top1}} of probes (median rank {{probe_sim_med}}), and for the probes with the training
degradations for {{probe_aug_top1}} (median rank {{probe_aug_med}}). For clean simulated patterns of the benchmark's
self-entries, scored with the answer sets of the measured patterns, it reaches a compound-weighted top-1 of 0.25 and
a median rank of 5, against 0.02 and about 5,600 on the measured patterns of the same targets. A classical score would
rank a clean rendering of a library entry first by construction, so this test only diagnoses the model. Nor is the
failure about unseen structures: patterns whose crystal form is in the model's training split have a median rank of
{{ov_train_med}} and patterns whose form is in no split {{ov_none_med}}.

The effect ladder shows that the model has learned the effects we simulated (figure 6). Rendered with each effect of
its measurement, the self-entries stay near the clean result: with Kα2 the model reaches 0.25 (median rank 6), with
the Debye–Waller factor 0.39 (median rank 4), and with the first six effects together 0.22 (median rank 12). With
those six plus the Debye–Waller factor it reaches 0.23 at a median rank of 8, and with intensity jitter added 0.21 at
a median rank of 12. The earlier model of ref. [@cai2026urtc], trained without these effects and without the
operator tokens, falls to a median rank of 744 on the rendering with the six effects and the Debye–Waller factor. Yet on the measured patterns the retrained model
stays at 0.02 with a median rank of about 5,600. Fixed-grid Pearson, by contrast, reaches 0.84 on the measured
patterns, close to its value on the fully rendered self-entries. The measured patterns therefore differ from our
simulations in a way that barely affects fixed-grid correlation but moves this model's embeddings far. We have not
identified it; candidates include instrument aberrations that a constant offset does not absorb (axial divergence,
specimen transparency), anisotropic broadening and real preferred orientation, each of which leaves the peak
positions of a pattern, and so a correlation with its simulation, largely intact.

Matching patterns to patterns places the failure in the pattern encoder. Matched against the clean library patterns
in the model's pattern space, without the structure encoder, the measured patterns reach a compound-weighted top-1 of
0.17 (median rank 156), whereas fixed-grid Pearson, which also compares the two patterns point by point without an
alignment search, reaches 0.84. The encoder places measured patterns away from simulations of the same structures
before the structure side is involved. The seeds and the ablation agree with this reading. The second seed reaches
{{s1_std}} on the measured patterns with a median rank of about 3,300, the model without operator tokens 0.03 with a
median rank of about 4,100, and none of the three is statistically distinguishable from the earlier model (paired
differences {{paired_vs_orig_range}}). The operator tokens raise the model's within-family top-1 in the simulated
sweep by about 0.02 over the ablation (in all 42 conditions for seed 0 and 41 for seed 1, although in no single
condition do the intervals separate), but they leave the measured benchmark unchanged. As a shortlist for a classical re-rank the model also fails: its top 1,000 holds a correct
entry for {{model_recall_1000}} of the patterns, and re-ranking that list with the ln q scan gives a compound-weighted
top-1 of {{hybrid_lnq}}, against 0.88 for the ln q scan on the whole library.

The model's ranks also depend on where the patterns come from, a split we examined only after seeing the external set
of Section 4.8. On the opXRD patterns its median rank is {{inst_labcu_med}} for the {{inst_labcu_n}} laboratory Cu
patterns, {{inst_syncamb_med}} for the {{inst_syncamb_n}} ambient synchrotron patterns and {{inst_syncp_med}} for
the {{inst_syncp_n}} pressure-series patterns, whereas fixed-grid Pearson ranks the answer first for
{{inst_pearson_range}} of the patterns in each of these groups. On the {{kad_sync_n}} synchrotron profiles of the
external set, all from one group's Rietveld studies, the model's median rank is {{kad_sync_med}} and it ranks
{{kad_sync_top1}} of them first, while its {{kad_lab_n}} laboratory Cu profiles rank beyond {{kad_lab_min}}.
Of those {{kad_sync_n}} self-entries, {{kad_sync_train}} are in the model's training split, but on the opXRD
patterns membership made no difference, and the {{kad_sync_nontrain_n}} that are not rank {{kad_sync_nontrain}}. We have not
identified what separates these profiles from the opXRD synchrotron patterns; their counting statistics and angular
resolution, and the way their structures were refined, are candidates.

![](figs/fig6_effect_ladder.pdf)

**Figure 6.** Effect ladder. Each pattern's self-entry is rendered cleanly, with one effect of its measurement at a
time, with the first six together, and with an overall Debye–Waller factor (B), log-normal intensity jitter per
reflection (σ = 0.3) or both added; the last point is the measured patterns themselves (Section 3.5). Answer set:
any same-form entry (188 patterns, 123 compounds). (a) Compound-weighted top-1 for fixed-grid Pearson, the model (two
seeds) and the earlier model of ref. [@cai2026urtc]; (b) median rank of the first correct entry for the models.

### 4.7 Robustness of the benchmark

None of the choices we tested changes the comparison (table 2; Supplementary Section S12). Rescoring every pattern
with the Lorentz–polarisation factor and the area-conserving Jacobian of the conversion to Cu changes the classical
compound-weighted top-1 by {{conv_range}} and the ln q scan not at all, and leaves the model unchanged. Removing the
3 patterns without a recorded wavelength changes no classical score by more than {{nolam_range}}. Removing the 35
lenient-anchor patterns changes the classical scores by {{strict_range}}, and the model scores {{strict_model}}.
Counting EPIVEF as a correct answer for its pattern, as the every-anchor comparison would, moves that pattern into the
single-crystal set and changes no compound-weighted top-1 of the ambient set by more than {{ea_change}}. The library
package of Section 3.8 changes the classical scores by {{r2_range}}; {{r2_nexcl}} of the {{r2_n}} paired intervals of
the package and its ablations, over the single-crystal and any-same-form sets, exclude zero, about as many as chance
would give, and the model, whose library embeddings are unchanged, keeps its scores. Rescaling the library to 295 K,
which we scored with the converted intensities, changes the classical scores by {{dT_range}} against the converted
intensities alone, and averaging over each grid step instead of interpolating by {{binavg_range}}.

**Table 2.** Compound-weighted top-1 of each score under the main benchmark and the robustness variants of Section
4.7: ambient single-crystal set / any same-form entry. The recorded-wavelength and strict-anchor rows drop 3 and 35
patterns, and counting EPIVEF moves one pattern into the ambient set. Intervals and paired differences are in
Supplementary Sections S2 and S12.

{{table2}}

### 4.8 The open replica and an external set

Against the 108,902 COD entries, the 190 patterns reproduce the main result when any same-form entry counts: the
classical scores reach a compound-weighted top-1 of {{cod_std_range}} (125 compounds) and the model {{cod_model}}. The
replica's single-crystal set has 35 patterns but only 9 compounds, 20 of its patterns are ZZZUWK, scored against a
1950 room-temperature structure that the CSD benchmark excludes as not comparable, and its 5 % cell window leaves out,
by construction, the large cell differences that dominate the CSD failures, so it cannot test the single-crystal
result. Two of its queries contain Si, which the COD library filter excludes, so their answers enter the library only
as candidate entries; without these two queries no compound-weighted top-1 changes by more than {{cod_si_change}}. The
classical numbers of this section can be regenerated from open data alone.

The raw profiles that Kaduk and co-workers deposited with 18 structure reports in Powder Diffraction (19 patterns, 16
of them synchrotron) [@kaduk2015tramadol; @kaduk2015folic; @kaduk2015fingolimod; @kaduk2015levalbuterol;
@kaduk2015ziprasidone; @kaduk2015rilpivirine; @kaduk2015solifenacin; @kaduk2015lacosamide; @kaduk2015nilotinib;
@kaduk2015sitagliptin; @kaduk2016paliperidone; @kaduk2017palmitate; @kaduk2017hydrocortisone; @wheatley2018estradiol;
@reid2015curcumin; @kaduk2015saxagliptin; @reid2016trandolapril; @reid2021donepezil] form an external set, scored
with a design fixed before any result. When any same-form entry counts, the classical scores reach a compound-weighted
top-1 of {{kaduk_cls_range}} and the model {{kaduk_model}}, with a median rank of {{kaduk_model_med}}; this is
generous, because each pattern's own Rietveld entry is in the library. With those entries excluded, only 3 compounds
keep an independent single-crystal reference; the classical scores place it first for {{kaduk_sc_range}} of them and
the model for {{kaduk_sc_model}}, so this set is too small to test the separation seen on the opXRD single-crystal
patterns. The model's much better ranks on these profiles than on the opXRD patterns are examined in Section 4.6.

## 5. Discussion

On these measured organic patterns, all from one contributor to opXRD and searched against the CSD, simple classical
scores on background-subtracted patterns set the level that a learned retrieval method would have to reach. When the
reference is an independent single-crystal structure determined at ambient pressure, the shift-tolerant scores reach
a compound-weighted top-1 of 0.45–0.62 over 26 compounds, and when the pattern's self-entry is in the library they
reach up to 0.88, with even fixed-grid Pearson correlation at 0.84. Whether the same ordering holds for other sources
of measured patterns is untested beyond the small external set of Section 4.8. We suggest that learned PXRD retrieval
be reported on measured patterns, against a library of realistic size, against shift- and strain-tolerant correlation
scores with background subtraction, with compound-weighted metrics and a compound-level bootstrap whenever compounds
repeat, and separately for queries whose reference is an independent determination, with the measurement conditions
and the source of the patterns recorded. The benchmark released with this paper supports each of
these, and its open replica lets the any-same-form results be checked without a CSD licence.

The model's one advantage is speed. With precomputed library embeddings it searches 309,080 entries in about 5 ms per
query on a CPU, whereas one CPU core needs about 0.4 s per query for fixed-grid Pearson, about 20 s for a zero scan
and about 6 min for the ln q scan; on one GPU, all five classical scores of 8,188 patterns against the full library
took 761 s. The variable-cell re-rank is much slower, at a median of {{c2_cpu_min}} CPU-minutes per comparison. Speed
matters only once accuracy does, however, and a shortlist from this model loses most correct entries.

Two problems remain open, and both concern the gap between simulated and measured patterns. For independent
single-crystal references, retrieval fails when the reference was determined at another pressure or temperature and
the cells differ anisotropically by more than about 1 %. The published variable-cell method gives no gain that we can
distinguish here, is bounded by the shortlist it re-ranks, and in its automated form fits peaks poorly on the
high-pressure patterns that make up most of the mismatched cells. This case remains open for classical methods too,
and it is where a learned method could show its value. For the learned model, the failure is the transfer from
simulated to measured patterns, not the size of the library or the absence of symmetry information, and it survives
a training simulation that reproduces every effect we could model. How far the model falls differs between sources of
measured patterns: on high-resolution synchrotron profiles from one external source it ranks the self-entry at a median of
{{kad_sync_med}}, against thousands on the opXRD patterns, and which property of those profiles matters is open.
Full-profile generative models have shown the same
sensitivity to how realistic the training patterns are [@segal2026xrdiff]. A learned method aimed at that transfer,
for example one adapted with measured patterns rather than with further simulated effects, seems a more promising use
of learning than a stand-alone retriever.

Our study has several limitations. We tested one model architecture, trained with two seeds and one ablation; another
architecture, loss or training strategy could do better, and our conclusions about learning concern this model. The
benchmark has 123 compounds, all organic, from one contributor to opXRD, and most patterns are the data their own
entry was refined against. The ambient single-crystal set has 26 compounds, so its intervals are wide, and the open
replica's single-crystal set has only 9. The main library is simulated with one peak profile and without
Debye–Waller factors or Kα2; a package of such changes moved little but does not exhaust them, uses one
Debye–Waller factor for every temperature, and disordered entries are simulated only approximately. The measured
patterns have a simple background removal and are interpolated onto the Cu grid, and the zero scans use fixed offset
grids (0.02°, and 0.04° for the ln q scan). The effect ladder adds Kα2 to every laboratory Cu pattern and does not
estimate the intensity jitter. The only practitioner baseline beyond the classical scores is the isotropic rescaling
of reference cells to room temperature; indexing followed by a reduced-cell search, and Pawley or Le Bail fits on a
shortlist, were not tested. The variable-cell re-rank covers a shortlist of 50 with a 4-hour limit per comparison,
uses no zero-offset correction, reaches only cell differences within ±10 % and ±5°, and fits peaks poorly on the
high-pressure patterns. The correct answers rely on packing similarity, which can count entries as one form across a
phase transition that preserves the packing and excludes entries that cannot be compared. Retrieval identifies known
forms only; crystal structure prediction combined with pattern matching can identify forms that no database contains
[@glick2025csp].

## 6. Conclusions

We searched 188 measured organic powder patterns against 309,080 simulated CSD entries. When the answer is an
independent single-crystal structure determined at ambient pressure, shift-tolerant classical profile scores rank it
first for a compound-weighted 0.45–0.62 of the 26 compounds. A contrastive model with a space-group-aware structure
encoder ranks it first for none of them. When the pattern's self-entry also counts, the classical scores reach
0.71–0.88 and the model reaches 0.02. The model retrieves simulated patterns of the same targets, including patterns
rendered with every measurement effect we could model, so its failure lies in differences between measured and
simulated patterns that our simulation does not reproduce, and its pattern encoder already places measured patterns
away from their simulations. These controls establish the discrepancy for this model on the opXRD patterns, not its
cause or how common it is elsewhere: on high-resolution synchrotron profiles from one external source the model does
far better, for reasons we have not identified. A more realistic library, a different intensity convention and the label checks leave
the classical results unchanged, and an open replica built from the COD reproduces the any-same-form result. The hard
open case is a measured pattern whose only reference was determined at another temperature or pressure, so that the
cells differ anisotropically by more than about 1 %. We release the benchmark so that learned methods can be compared
with these numbers on measured data.

## Ethics

This work did not involve human participants, animals or field work.

## Data accessibility

The code, the benchmark definition and all results are available at https://github.com/fronkt/xrd-clip and archived
at Zenodo ({{zenodo_doi}}). The deposit contains the opXRD pattern identifiers with their CSD refcodes, same-form sets,
packing-similarity results and label corrections; the corpus refcode list and the training splits; the per-pattern
ranks of every score, correction and variant on the measured patterns, including the learned models' and the
variable-cell re-rank's; the simulated sweep and the effect ladder as summary results; the critic2 inputs derived
from the opXRD patterns; the scripts that build the library from a CSD installation, compute the scores and regenerate
every table and figure; and results/rsos_numbers.json, from which every number in this article is checked by
scripts/verify_claims_rsos.py. The open replica of Section 2.6 is included in full: the COD identifiers, the simulated
COD library with its symmetry operators, its answer sets and the ranks of every score, from which the classical
results of Section 4.8 can be reproduced without a CSD licence. The CSD library was built from the May 2026 release
(1,451,367 entries; CSD Python API 3.7.1); the COD was downloaded on 12–13 October 2026 and its CIFs again on
19 November 2026 for the symmetry operators; the opXRD patterns are from the Zenodo record 10.5281/zenodo.15298026
under CC BY 4.0, and the COD structures are CC0. CSD structures and the simulated patterns derived from them cannot be
redistributed, and users with a CSD licence regenerate them from the refcode lists. {{checkpoint_statement}} The
external set of Section 4.8 uses the measured profiles in the supplementary CIFs of the cited Powder Diffraction
reports, which can be downloaded from Cambridge Core, and, for donepezil hydrochloride form III, the CIF deposited
with the ICDD; these profiles are not redistributed. Supplementary material is available online.

## Declaration of AI use

<!-- [FRANK] RSOS requires AI use to be declared. Draft below states what was done in this project; edit to your
wording. -->
The author used a large language model (Claude, Anthropic), under the author's direction, to write and run analysis
code, to cross-check the statistical analyses and internal audits, and to draft and edit text. The author designed the
study, reviewed all code, results and text, and takes full responsibility for the content. All code, data and results
are deposited (see Data accessibility) so that every number can be regenerated independently.

## Authors' contributions

F.C.: conceptualization, methodology, software, formal analysis, investigation, data curation, visualization,
writing – original draft, writing – review and editing.

## Conflict of interest declaration

The author declares no competing interests.

## Funding

This research received no specific grant from any funding agency in the public, commercial or not-for-profit sectors.
<!-- [FRANK] Anvil time is ACCESS allocation CHE260157 (NSF): acknowledge it below, and check whether ACCESS asks for a
funding line. -->

## Acknowledgements

The author thanks the opXRD and COD contributors whose deposited powder data made the benchmark possible, the CCDC for
the CSD and its Python API, and the ICDD for providing the deposited donepezil hydrochloride CIF. This work used the
Anvil supercomputer at Purdue University through allocation CHE260157 from the Advanced Cyberinfrastructure
Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by U.S. National Science Foundation
grants 2138259, 2138286, 2138307, 2137603 and 2138296. GPU time was also rented from commercial cloud providers.

## References
