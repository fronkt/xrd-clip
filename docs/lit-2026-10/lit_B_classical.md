# Track B: classical (non-deep-learning) PXRD comparison and search-match

Compiled 2026-10-10 for the xrd-clip follow-up. Sources: open pages only (OpenAlex API, Crossref API, Semantic Scholar API, Europe PMC full-text XML, arXiv, ChemRxiv/university repository postprints, GitHub, vendor pages). No browser, no proxy, no keyed APIs. RSC, IUCr and Nature landing pages returned 403 or a login redirect, so their abstracts were read through OpenAlex or Europe PMC instead.

Legend for "Verified":
- **V-full**: I read the full text.
- **V-abs**: I read the abstract or API record.
- **V-snip**: only a search-engine snippet or vendor marketing text. These items are repeated in Section 4.

**Correction to the brief.** VC-PWDF is by **Mayo, Otero-de-la-Roza and Johnson** (Dalhousie / Oviedo), not Tuckerman.

---

## 1. Methods table

| # | Method / citation | ID | Distortions handled | Cost per comparison | Open implementation | Reported accuracy (as stated in the source) | Verified |
|---|---|---|---|---|---|---|---|
| 1 | **de Gelder weighted cross-correlation (WCC)**. de Gelder, Wehrens, Hageman, *J. Comput. Chem.* 22:273 (2001) | 10.1002/1096-987X(200102)22:3<273::AID-JCC1001>3.0.CO;2-0 | Small peak shifts of any origin, up to roughly the triangle half-width *l*. This covers a small zero offset and small strain, but nothing is corrected explicitly. Normalized, so invariant to intensity scale. | Numerical: O(n_points × window). No timing in the abstract. Otero-de-la-Roza (2024) reports his analytic GPWDF is about 4.5× faster than WCC. | **Yes**: CSD Python API `PowderPattern.similarity(width=2.0, use_esds=True)` (licensed CSD); R package `ptw::wcc(pattern1, pattern2, trwdth)`; pip `pyxtal` `XRD.Similarity(l=2.0)` (triangle or cosine weight); critic2 `COMPARE … POWDER` | Abstract: the criterion "properly recognizes shifted but otherwise similar details". Applied to classifying cephalosporin structures from calculated patterns. No numeric accuracy in the abstract. | V-abs (Radboud page + OpenAlex); CSD API docstring V-full; pyxtal source V-full; ptw docs V-full |
| 2 | **CCDC Mercury / CSD-Materials PXRD similarity** | CSD Python API docs | Same as WCC. Default width 2.0° with ESD weighting reproduces the Mercury "PXRD Similarity (ESDs)" column; width 0.1 without ESDs reproduces the plain column. Mercury reportedly also adjusts the simulated pattern to an indexed room-temperature cell. | Fast (WCC) | Licensed (CSD-Materials) | Mayo 2022 (snippet): 12.35% disagreement with COMPACK on 44,939 CSD pairs, against 2.84% for VC-PWDF | API docstring V-full; cell adjustment and Mercury UI V-snip |
| 3 | **FIDEL** (FIt with DEviating Lattice parameters). Habermehl, Mörschel, Eisenbrandt, Hammer, Schmidt, *Acta B* 70:347 (2014) | 10.1107/S2052520613033994 | **Wrong or deviating lattice parameters** (temperature, pressure, isostructural analogues), unindexed data, low-quality data, non-phase-pure samples. Optimizes the cell plus molecular position, orientation and selected torsions to maximize de Gelder similarity, then runs Rietveld. | **Expensive**: "a single FIDEL local optimization takes between a few minutes and half an hour" (quoted by Otero-de-la-Roza 2024) | Not open. AutoFIDEL is distributed inside CCDC software (V-snip). | Applications only. Abstract gives no success rate. | V-abs (OpenAlex) |
| 4 | **FIDEL-GO**. Habermehl, Schlesinger, Schmidt, *Acta B* 78:195 (2022) | 10.1107/S2052520622001500 | Global optimization from random structures with random cells in many space groups. No indexing, cell or space group needed. Works on nanocrystalline and low-crystallinity patterns. | Very expensive (global structure solution) | Not open | Solved quinacridone derivatives and a Cu coordination polymer, "some analyzed from patterns containing merely 14–20 peaks" | V-abs |
| 5 | **Hofmann–Kuleshova similarity index**. *J. Appl. Cryst.* 38:861 (2005) | 10.1107/S0021889805023484 | "Valid in cases of large deviations in cell constants" | n/r | n/r | Solved un-indexed pigments PY111, PR181 and Me-PR170 | V-abs |
| 6 | **van de Streek & Motherwell CSD polymorph search**. *Acta B* 61:504 (2005) | 10.1107/S0108768105020021 | Compares simulated patterns "with special provisions to deal with differences in unit-cell volumes caused by temperature or pressure" | Ran over 35,000 pairs, so cheap | No (CCDC internal) | 35,000 same-compound pairs compared, 7,300 polymorph pairs identified, 154 previously unknown | V-abs |
| 7 | **VC-PWDF** (variable-cell powder difference). Mayo, Otero-de-la-Roza, Johnson, *CrystEngComm* 24:8326 (2022) | 10.1039/D2CE01080A | **Lattice distortion** between two structures. Niggli-reduces both cells, enumerates candidate cell bases coincident with the reference, deforms the candidate onto the reference cell, then scores with de Gelder WCC. Needs **both cells**, so an experimental pattern must be indexed. Cannot handle disordered structures. | Cheap per pair (ran on ~45k pairs; no timing found) | **Yes**: critic2 `COMPAREVC VCPWDF` (GPL-3). Defaults MAXELONG 30%, MAXANG 20°, MAXVOL 50%. | Thesis abstract: on about 45,000 CSD pairs it is "as successful as COMPACK" at telling the same form under different conditions from a different polymorph, and it found CSP matches that COMPACK missed. Snippet: 2.84% total disagreement with COMPACK on 44,939 pairs, against 12.35% for the CCDC PXRD tool. | V-abs (OpenAlex, Semantic Scholar, Dal thesis); numbers V-snip |
| 8 | **VC-xPWDF** (experimental variant). Mayo, Marczenko, Johnson, *Chem. Sci.* 14:4777 (2023) | 10.1039/D3SC00168G | Experimental pattern against candidate structures. Lattice distortion is handled, **but indexing is required**. Normalization: subtract the minimum, scale the maximum to 100. Preferred orientation is NOT handled (d-mannitol scores rose to 0.096–0.099). Zero shift is not explicitly handled. | Not stated | critic2 (developers version at publication) | 7 organics (urea, 1,4-dicyanobenzene, uracil, caffeine, acetaminophen, d-mannitol, progesterone): "the CSD structures corresponding to the same polymorph … give the smallest VC-xPWDF score" for all 7. Matches scored 0.0114–0.0772; non-matching polymorphs 0.2087–0.7293; suggested match threshold < 0.1. 2-minute scans gave results "nearly indistinguishable" from 3-hour scans. AutoFIDEL found the match for 6 of 7 at default settings and fails when no ambient-temperature structure exists. | **V-full** (Europe PMC XML) |
| 9 | **GPWDF / VC-GPWDF**. Otero-de-la-Roza, *J. Appl. Cryst.* 57:1401 (2024) | 10.1107/S1600576724007489 | **Strongest open classical baseline.** GPWDF is de Gelder's index computed analytically on (θ_i, I_i) peak lists with Gaussian peaks (σ) and a Gaussian weight (α). VC-GPWDF globally minimizes GPWDF over a **6-parameter lattice strain** of the candidate, with fractional coordinates fixed. Default bounds: cell lengths ±10%, angles ±5°. **No indexing.** Experimental patterns are pre-processed by automatic background fitting and pseudo-Voigt peak fitting, which removes peak shape, background and noise from the comparison. The paper says zero-point and preferred-orientation corrections "can be applied during pre-processing" but were not used. | GPWDF local minimizations take "a fraction of a second". VC-GPWDF averaged **31 s on one processor, max 102 s**, per candidate against an experimental pattern. The global search stops after 5000 evaluations without improvement (NLopt MLSL + SLSQP). Peak-fit pre-processing takes seconds to minutes, once per pattern. | **Yes**: critic2 `COMPAREVC` (GPL-3, Fortran, cmake; NLopt optional). `XRPD FIT` turns a .xy file into a .peaks list. Defaults: ALPHA 0.5 (the paper used α = 1), MAXELONG 10%, MAXANG 5°. No pip or conda package (checked: GitHub README). | On the 7th CSP blind-test molecule XXIX (deliberately degraded experimental pattern, 310 DFT candidates), the experimental structure ranked **1st** (score 0.052). Against the experimental structure: 0.0024, against 0.047 for the next candidate. Plain fixed-cell indices gave the true DFT structure only de Gelder 0.27 / GPWDF 0.30 dissimilarity; with the experimental cell, 0.002. On Mayo's 7 molecules against CPOSS candidates, the experiment-derived structures were the top ranks, without indexing. | **V-full** (Oviedo postprint, CC-BY) |
| 10 | **PXRD-assisted evolutionary CSP** (XtalOpt + VC similarity). Racioppi, Otero-de-la-Roza, Hajinazar, Zurek, *Digital Discovery* (2024); arXiv:2407.05394 | 10.1039/D4DD00269E | Multi-objective search on enthalpy plus a similarity computed after distorting locally optimized cells, which "transcends" temperature and strain mismatch | n/r | XtalOpt + critic2 (open) | Qualitative in the abstract ("improves the accuracy … reduces the time") | V-abs |
| 11 | **Industrial CSP with cell and peak-shape refinement against PXRD**. Glick, Metcalf, Swarthout et al., arXiv:2507.16218 (2025) | arXiv:2507.16218 | "Simulate the powder pattern at the experimentally reported wavelength and refine the unit cell parameters and peak shape parameters to maximize a cross correlation-based similarity score" | n/r | No (proprietary Lavo pipeline) | Matched patent patterns for omaveloxolone, deucravacitinib and zuranolone. A tolbutamide Form IV pattern matched Form IIb at "almost 95% similarity". | V-full (PDF text) |
| 12 | **COMPACK** (structure-space reference). Chisholm & Motherwell, *J. Appl. Cryst.* 38:228 (2005) | 10.1107/S0021889804027074 | Compares a 3D cluster of molecules by interatomic distances. Uses no cell or space group, so it is cell-independent by construction. Needs structures, not patterns. | Can be "extremely costly" for highly branched molecules (Mayo 2022, snippet) | Licensed (CSD Python API packing similarity, Mercury) | Ground-truth reference in Mayo 2022 | V-abs (Crossref + CCDC docs) |
| 13 | **SNAP-1D / PolySNAP** full-profile clustering. Gilmore, Barr, Paisley, *J. Appl. Cryst.* 37:231 (2004); Barr, Dong, Gilmore, *J. Appl. Cryst.* 37:658 (2004) | 10.1107/S002188980400038X; 10.1107/S0021889804011173 | Point-by-point **Pearson + Spearman** full-profile correlation after wavelet pre-processing. An "optimal shift in 2θ" correction is provided (three variants, snippet). Clustering via dendrograms, PCA and MDS for polymorph screening. | Cheap (correlation matrices over more than 1000 patterns) | Commercial (Bruker-AXS); PolySNAP 1.7 manual downloadable from Glasgow | "over 1000 powder diffraction patterns … in less than one week" (throughput context only) | V-abs; shift correction V-snip |
| 14 | **ICDD SIeve+ (PDF-4+/PDF-4/Organics)** | ICDD SIeve Technical Bulletin (2014) | Hanawalt (8 strongest lines, rotated), Fink, Long8, Integral Index, Electron Diffraction, and a "Similarity Index (Normalized R-index)" that is a point-by-point full-pattern comparison. Goodness of Match (GOM) over 8 lines with max 8000, using a 2θ search window "SW". The user sets search and match windows (window tolerance absorbs small shifts). No cell refinement. | Index methods fast; the R-index is "visibly slower" | Commercial | No accuracy numbers; case studies only (vitamin pill) | V-full (bulletin PDF text) |
| 15 | **HighScore (Plus)**. Degen et al., *Powder Diffr.* 29:S13 (2014) | 10.1017/S0885715614000840 | Search-match on peaks and/or the full net profile, combined in one step; candidate clustering; cluster or similarity analysis of many patterns | n/r | Commercial | None stated | V-full (paper text), but the paper gives no algorithmic detail on zero-shift handling |
| 16 | **Full-profile search-match by Rietveld (FPSM)**. Lutterotti et al., *J. Appl. Cryst.* 52:587 (2019) | 10.1107/S160057671900342X | Automatic Rietveld fit of **every database candidate**. Refines background and scale, then **cell (volume expansion/contraction)**, then crystallite size and microstrain. | "an average of 30 full automatic Rietveld refinements per second, per thread". Whole searches: 1 s (garnet) to about 3,300 s (complex organic polymorphs). | Free web service fpsm.radiographema.com; MAUD (free); uses COD | Quantitative-standard errors within 2% | V-full (Europe PMC XML) |
| 17 | **Dissimilarity-measure benchmark incl. DTW**. Iwasaki, Kusne, Takeuchi, *npj Comput. Mater.* 3:4 (2017) | 10.1038/s41524-017-0006-2 | Peak-height change and peak shift from composition-driven lattice change (Fe–Co–Ni) | DTW is costlier (not quantified in the abstract) | Measures are standard (scipy, dtw-python, tslearn) | Abstract: with unknown shift magnitude, **cosine, Pearson and Jensen–Shannon performed best**; with known maximum shift, **constrained normalized DTW** was best. Nine measures in total. | V-abs |
| 18 | **RealBind benchmark** (DL vs classical). Srivastava, ChemRxiv (2026) | 10.26434/chemrxiv.15009668 | Simulated against measured PXRD queries | n/r | n/r | On a 492-candidate library a contrastive model reached 78.5% Recall@10 on simulated queries and **5.3%** on measured queries. A **model-free correlation baseline reached 82.8%** Recall@10. | V-abs (OpenAlex) |
| 19 | **7th CSP blind test** (PXRD-assisted exercise). Hunnisett et al., *Acta B* 80 (2024) | 10.1107/S2052520624007492 | A deliberately low-quality PXRD pattern of XXIX (methyl anthranilate) | — | Data on the CCDC website | The exercise "demonstrated the use of CSP in successfully determining a crystal structure from a low-quality PXRD pattern" | Record V-abs; wording V-snip |

n/r = not reported in the pages I could read.

Supporting, not baselines: Segal et al., *Digital Discovery* (2026), 10.1039/D6DD00017G, "XRD similarity metrics create a highly non-convex, ill-posed loss landscape". This is why the VC methods use global multi-start search rather than plain gradient descent (V-abs). Also Shtukenberg et al., *Chem. Sci.* (2017), 10.1039/C7SC00168A, used PXRD plus CSP to identify four new coumarin polymorphs (V-abs).

---

## 2. Baselines a rigorous follow-up MUST beat

The current baselines (fixed-grid Pearson, naive peak-list match) are strawmen for the shift and strain claim. A reviewer from the CSP or powder community will name de Gelder WCC and VC-(G)PWDF straight away, because both are in critic2, Mercury and the CSD API. Recommended ladder, from cheapest to strongest:

### B1. Shift-tolerant WCC (de Gelder 2001), plus an explicit zero-shift scan
Simple, standard and nearly free. Implement it in numpy: no reliance on licensed CCDC, and you can check against `pyxtal.XRD.Similarity` or `ptw::wcc`.

```
input: f(2θ), g(2θ) on a common grid (Δ = 0.02°), 2θ ∈ [5°, 40°] (or the xrd-clip range)
preprocess: subtract the minimum, scale to unit max (same normalization for both)
for δ in arange(-δmax, δmax, Δ):          # δmax ≥ largest test zero offset (e.g. 0.5°)
    g_δ = g shifted by δ
    c_fg = correlate(f, g_δ, lags −L..L)  # L = l/Δ points, l = triangle half-width
    w(r) = max(0, 1 − |r|/l)
    S(δ) = Σ w·c_fg / sqrt(Σ w·c_ff · Σ w·c_gg)
score = max_δ S(δ)
```
- Parameters: report results for l = 0.5°, 1.0° and 2.0° (2.0° is the CSD API / Mercury default), with δmax set by the zero-offset range in your test set. Also report plain Pearson after the δ scan ("Pearson + zero scan") as an ablation.
- Cost: microseconds to milliseconds per pair (inferred). Retrieval over the full library is trivially fast.

### B2. Zero-offset + isotropic-strain search in log-d (log-q) space (cheap, cell-free)
**This is my construction, not taken from a paper.** It follows from Bragg's law. An isotropic strain ε rescales every d-spacing by (1+ε), so in **ln d** (equivalently ln q) it becomes a **pure translation**. A zero offset is a pure translation in 2θ. So a 2-parameter search neutralizes both exactly for isotropic strain:
```
for δ in zero-offset grid (2θ space):
    map g(2θ − δ) → ln q grid (q = 4π sinθ/λ), interpolate
    FFT cross-correlate with f on the ln q grid, weighted by a triangle (WCC) over lags ≤ ln(1+εmax)
score = max over (δ, lag)
```
- εmax ≥ the largest strain tested (e.g. 3–5%). For **anisotropic** strain (unequal a, b, c, monoclinic β), B2 only approximates, so B3 is needed.
- Small-shift sense check (inferred, Cu Kα): Δ2θ ≈ −2 tanθ·ε. At 2θ = 30° and ε = 1% that is about 0.31°, so plain WCC with l ≥ 1° already absorbs about 1% isotropic strain over 5–40°.

### B3. VC-GPWDF (Otero-de-la-Roza 2024): the gold-standard classical baseline
Library candidates are CIFs, so each candidate's hkl list and structure factors are known. VC-GPWDF can therefore strain the candidate's cell to fit the query **without indexing the query**. This is exactly the "invariance to strain" the paper claims.
```
query: pattern → critic2 `XRPD BACKGROUND` + `XRPD FIT` → (2θ_i, I_i) peaks   [once per query]
   (synthetic queries: use the simulated peak list directly, or fit to mimic experiment)
for candidate CIF in library (or top-K from B1/B2):
    critic2: COMPAREVC candidate.cif query.peaks  GLOBAL  ALPHA 1.0 (paper) or 0.5 (default)
             MAXELONG 0.10  MAXANG 5   [widen if the test strain exceeds 10%/5°]
rank by the VC-GPWDF score (lower is better)
```
- Zero offset: VC-GPWDF has no zero-shift parameter. Correct it in pre-processing (the paper says this "can be applied"): scan δ as in B1, or add δ as a 7th parameter in a re-implementation. A reimplementation is easy: GPWDF has the closed form I_AB = √z Σ_i Σ_j I_i I_j exp(−πz(θ_i − θ_j)²), with z = 1/(4πσ² + α²) and G = 1 − I_AB/√(I_AA·I_BB). Optimize ε (6 parameters) plus δ with multi-start L-BFGS or NLopt MLSL.
- Cost: average 31 s per candidate (max 102 s) on one core, as reported. Exhaustive retrieval is impractical (10⁴ candidates ≈ 86 core-hours per query; inferred). The honest protocol is **B1/B2 prefilter → VC-GPWDF re-rank of the top-K** (K = 50–100). Report accuracy against wall-clock for xrd-clip versus this pipeline.
- Note on scores: VC-GPWDF and VC-xPWDF react strongly to **preferred orientation**, because intensities enter the score (Mayo 2023, d-mannitol). That is a texture regime where a learned model could honestly win.

Optional fourth baseline for a powder-diffraction audience: **FPSM-style Rietveld search-match** (Lutterotti 2019). Refine scale, background, cell, size and microstrain per candidate, about 30 refinements/s/thread. It is closest to what an expert would actually do. If too heavy, GSAS-II or a Pawley/Le Bail fit on the top-K would also serve.

Reporting requirement: xrd-clip loses to plain Pearson on clean, broadened, textured and noisy data (1.0 against 0.56 within-family top-1). The follow-up must show B1–B3 on **every** perturbation axis, not only shift and strain, and should consider a **hybrid**: xrd-clip retrieval followed by VC-GPWDF re-ranking.

---

## 3. Would these neutralize the "invariance to zero-offset and strain" advantage?

**Very likely yes**, based on design and reported results (inferred, not tested here):
- **Zero offset**: an explicit δ scan (B1) or WCC's shift tolerance removes it entirely, at negligible cost. A zero offset is the easiest distortion to correct classically. Any vendor search-match or SNAP-1D shift correction already handles it.
- **Isotropic strain or thermal expansion**: B2 is exact by construction. WCC with l ≈ 1–2° already tolerates about 1% (inferred estimate above).
- **Anisotropic lattice distortion** (the realistic case across temperatures or DFT-vs-experiment): VC-PWDF matches COMPACK on about 45k CSD pairs. VC-GPWDF ranks the true structure 1st from a degraded experimental pattern without indexing, within ±10% cell / ±5° angle by default.
- **Where xrd-clip can still claim ground** (each needs a measured comparison):
  1. **Speed and amortization**: one embedding lookup against seconds-to-minutes per VC candidate.
  2. **Preferred orientation and texture**: VC scores degrade, as Mayo 2023 documents.
  3. **Strains beyond the VC search box**, or many-phase mixtures.
  4. **Weak or noisy patterns**, where peak picking fails. Here GPWDF's pre-processing is the weak link, but RealBind (2026) found plain correlation beats a contrastive model on *measured* queries, 82.8% against 5.3% Recall@10, so this is not a safe assumption.

---

## 4. UNVERIFIED (V-snip or not fetched; do not cite without checking)

1. VC-PWDF exact figures: **44,939 pairs, 2.84% disagreement with COMPACK, 12.35% for the CCDC PXRD tool**. These come from a search-engine snippet of the RSC abstract; the RSC page returned 403. The ~45,000 figure is confirmed by Mayo's Dalhousie thesis abstract.
2. CCDC Mercury: the de Gelder similarity score in the powder dialog, room-temperature cell adjustment from an indexed cell, and AutoFIDEL distribution with preferred-orientation support. CCDC blog pages would not render. Only the CSD API docstring (de Gelder, width 2.0) is verified.
3. SNAP-1D "provides three possible corrections" for 2θ shift (snippet of the Gilmore 2004 J. Appl. Cryst. paper; PDF 403).
4. Bruker DIFFRAC.EVA "Profile Fit Residual Search (PFRS)" and its zero-error handling (vendor page via snippet only).
5. Rigaku SmartLab Studio II / PDXL "Hybrid Search/Match" (peak position plus profile, preferred-orientation detection). Vendor marketing snippet only.
6. MDI JADE profile-based search/match with preferred orientation (ICDD product snippet only).
7. HighScore zero-shift or specimen-displacement handling during search-match. The Degen 2014 text I read does not describe it; the WebFetch summariser's claim that it does was not supported by the extracted text.
8. Wording of the 7th blind test PXRD-assisted exercise (snippet). The DOI record itself is verified.
9. PolySNAP3, *J. Appl. Cryst.* 42:965 (2009): DOI not fetched.
10. Faber & Blanton (2008) normalized R-index: cited in the ICDD bulletin, not fetched.
11. DTW cost relative to Pearson for XRD: not quantified in any source I read.
12. Original brief items **not found**: a 2022 "FIDEL fit with deviating lattice parameters" paper under that exact title (the 2022 paper is FIDEL-GO, row 4), and any Tuckerman-group VC-PWDF paper.

Searches behind item 12 and the "no pip package" note: WebSearch "Habermehl Schlesinger Schmidt FIDEL fit with deviating lattice parameters Acta Cryst B 2022"; OpenAlex search "FIDEL powder pattern deviating lattice parameters"; WebSearch "variable-cell powder difference VC-PWDF Mayo Tuckerman …" (all VC-PWDF hits were Mayo / Johnson / Otero-de-la-Roza); critic2 GitHub README for pip or conda.
