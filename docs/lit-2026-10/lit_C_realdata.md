# Track C: real organic PXRD test data, sim-to-real methods, thermal-expansion effect sizes

Compiled 2026-10-10 for the xrd-clip continuation. Sources: WebSearch/WebFetch on open pages only (arXiv, Europe PMC REST, Crossref API, Zenodo, GitHub API, publisher abstract pages, ICDD/CCDC public PDFs). No browser, proxy or keyed API was used. PMC, IUCr journal pages, RSC article HTML and nature.com returned 403/captcha/login redirects, so for those papers the numbers come from Europe PMC full text, arXiv or author/repository PDFs. "Inference" marks my own reasoning or arithmetic, not a stated number.

---

## 1. Ranked candidate sources of real organic PXRD with structure labels

Ranked by expected usefulness for a CSD-retrieval test: organic content × structure label × access × effort.

| # | Source | URL / DOI | # patterns (as stated) | Structure label? | License / access | Effort | Notes |
|---|---|---|---|---|---|---|---|
| 1 | **opXRD, CNRS (COD) subset** | Zenodo 10.5281/zenodo.15298026 (v11, 2025-04-28; 1.4 GB opxrd.zip); paper arXiv 2503.05577, also Adv. Intell. Discov. 2026, 10.1002/aidi.202500044 | 1,052 COD raw patterns. 85% have space group and atomic coordinates, 100% have composition. "organic, inorganic, and hybrid structures, and 75 different elements" | Y (COD CIF; ~85%) | CC BY 4.0 | **Low** | **The organic count is NOT reported.** The paper gives no organic/inorganic split. You have to filter it yourself (C+H present, no metals), then map COD IDs to CSD refcodes (many COD organics are also in CSD; inference). Median Δ2θ 0.013°, about 9,190 points per pattern. COD has 5,432 entries with any pdCIF tag, but most hold metadata only, not raw profiles (opXRD paper). The other opXRD contributors are inorganic: USC 338 alloys, HKUST 520, Empa 770 perovskites, INT 19,796, IKFT 64, LBNL 70,012 unlabeled. Total 92,552 patterns, 2,179 at least partially labeled. |
| 2 | **IUCr journals: Rietveld pdCIF supplements** (Acta Cryst B/C/E, J. Appl. Cryst., IUCrData) | Requirement: https://journals.iucr.org/e/services/powder.html (same text for a/b/c/j and IUCrData) | Not stated. Upper-bound context: the CSD holds **10,408 powder-diffraction studies (0.7%)** as of 1 Jan 2026 (CCDC summary statistics PDF) | Y (CCDC deposition number → refcode) | Profile data are "made available to readers through Crystallography Journals Online". Acta E is open access | Medium-High | IUCr requires "the numerical intensity or counts at each measured point" in CIF form for Rietveld papers. Kaduk-type DFT-optimised API structures appear regularly in Acta E (search hits only). Needs a scraper over supplementary CIFs. This agent could not open journals.iucr.org (403), but the files are public. |
| 3 | **Powder Diffraction journal: Kaduk API series** (ICDD grant project, synchrotron, 298 K) | e.g. repotrectinib 10.1017/s0885715625100882 (CC BY 4.0, verified). Crossref: 162 Kaduk-authored items in ISSN 0885-7156. In a 40-item 2015–2025 sample, 33 titles were "Crystal structure of …", mostly APIs | ~100+ APIs (inference from the Crossref sample) | Y (Rietveld CIF, DFT-optimised) | Articles are OA (CC BY). **Raw data are NOT attached.** Article text: "CIF files … (including the raw data) … were deposited with the ICDD. The data can be requested at pdj@icdd.com" | Medium (email request) | Synchrotron λ, not Cu Kα, so patterns need conversion to d-space or Cu-2θ. Measured at room temperature, which is useful as a temperature-matched control. Many are structures new to the literature and may not be in the 247k CSD set; check refcodes. |
| 4 | **CSD powder-study entries** (licensed CSD) | CCDC stats PDF: https://www.ccdc.cam.ac.uk/media/CSD-Entries-Summary-Statistics-2026.pdf | 10,408 powder studies; 45,729 polymorphic structures; 817,929 (57.1%) non-room-temperature studies | Y (refcode) | CSD licence (Frank has access) | Medium | UNVERIFIED whether deposited CIFs retrievable through CSD tools contain `_pd_meas_counts` / `_pd_proc_intensity` loops. Worth a one-hour check with the CSD Python API: filter for powder studies and grep the original CIFs. |
| 5 | **ICDD PDF-5+ / PDF-4/Organics** | https://www.icdd.com/assets/files/2026-Product-Summary.pdf | 2026 summary: "Experiments Digital Patterns" ~22–25k for PDF-5+ (column alignment ambiguous in the extracted text). Search snippet: 2025 = 20,833. Organic share of the raw patterns NOT stated. Pharmaceuticals subfile 6,370 (Fawcett PPXRD-11 slides, ~2012) | Partial (CSD refcode cross-references exist) | Commercial licence; terms for ML use UNVERIFIED | High ($, licence) | Largest pool of experimental organic patterns, but closed. Purdue library licence UNVERIFIED. |
| 6 | **Mayo, Marczenko & Johnson 2023, VC-xPWDF** | Chem. Sci. 10.1039/d3sc00168g (Europe PMC PMC10171065) | 7 compounds, lab PXRD: urea UREAXX, 1,4-dicyanobenzene TEPNIT, uracil URACIL, caffeine NIWFEE, acetaminophen HXACAN, D-mannitol DMANTL, progesterone PROGST. 3-h "moderate" and 2-min "low" quality scans | Y (refcodes) | "provided in the ESI or … available from the authors upon reasonable request" | Low-Medium | Reference structures span 12–200 K and up to 3.1 GPa; patterns are at room temperature. Directly relevant to the thermal-cell problem. |
| 7 | **Sparrow, Mayo & Johnson 2024, ROY high-throughput PXRD** | CrystEngComm 10.1039/D4CE00700J (Crossref verified) | Not retrieved (article page 403) | Y (ROY polymorph refcodes) | UNVERIFIED | Medium | The best open lead for a multi-polymorph family. Contact the authors. |
| 8 | **DASH (CCDC open source) examples and tutorials** | https://github.com/ccdc-opensource/dash (MIT) | ~8 .xye files: `examples/Telmisartan.xye`, `Example.xye`, tutorials 01–06 (some also .raw) | Partial (the .mol2 is a molecule, not a cell; compounds must be identified from the DASH docs) | MIT | Low | Small but free, and these are real lab and synchrotron organic data. |
| 9 | **SDPD round robins** (Le Bail & Cranswick) | https://www.cristal.org/sdpdrr2/ , https://cristal.org/SDPDRR3/index.html | SDPDRR-1 (1998): one organic pharmaceutical sample. SDPDRR-3 (2008): calcium tartrate tetrahydrate (metal-organic) plus an inorganic | Y (published solutions) | Free download | Low | Fewer than 5 patterns usable. |
| 10 | RRUFF | e.g. whewellite R050240/R050526/R070655 | Minerals. Schopmans et al. used 2,952 RRUFF samples (942 usable); RealPXRD used 269 | Y | Free | — | **Confirmed mostly irrelevant.** The only organics are a handful of organic minerals such as oxalates (3 whewellite entries), which are ionic salts, not molecular crystals. |
| 11 | NIST SRMs (640e Si, 660c LaB6, 674a/b oxides, 676 Al2O3, 1976b Al2O3 plate) | tsapps.nist.gov SRM certificates | — | — | — | — | **No organic SRMs found.** Use LaB6 660c or Si 640e to measure the instrument resolution function (Caglioti U, V, W) for the simulator. |
| 12 | Linus Pauling File / MPDS | (opXRD paper, via search snippet) | "21,700 corresponding experimental powder diffraction patterns" | Y | Commercial | High | Inorganic-focused. Listed only for completeness. |
| 13 | **Self-measured patterns** (Purdue diffractometer) | — | Target 60–100 | Y (choose compounds with known refcodes) | Own | Medium | The only route that gives controlled polymorph families, sample preparation and temperature. See the plan in Section 2. |

Not found despite searching: an open, labelled benchmark of experimental organic or pharmaceutical PXRD (2023–2026), or a Zenodo/Figshare high-throughput pharma PXRD dataset with structure labels. Searches run:
- "zenodo dataset raw powder X-ray diffraction patterns pharmaceutical polymorphs high-throughput crystallization screening" (extended)
- "open dataset experimental PXRD patterns of drugs active pharmaceutical ingredients Mendeley Data figshare hundreds of compounds" (extended)
- "experimental powder XRD benchmark organic molecular crystals CSD refcode machine learning 2024 2025" (extended)
- "machine learning PXRD organic crystals real experimental patterns test set CSD refcode sim-to-real OR domain gap molecular crystals 2025 2026" (extended)
- "ROY polymorphs carbamazepine paracetamol powder diffraction raw data deposited open repository xy file"

These searches cover web-indexed pages only. Zenodo/Figshare were not searched through their own APIs, so a small dataset could still exist. The only open experimental pattern sets they surfaced were opXRD (inorganic-heavy) and single-paper deposits (e.g. Zenodo 15655614, "Raw diffraction data for CCDC 2462782-2462784", which is a per-paper single-crystal/raw-frame deposit).

Other existing work also skips organics. RealPXRD-Solver (arXiv 2603.00965) explicitly excluded "purely organic and hybrid materials" from its CNRS (126) and RRUFF (269) test sets. XCCP's experimental test is inorganic (MP-trained).

## 2. Recommended real test-set assembly plan (≥100 patterns, several polymorph families)

**Tier A: open, fast (week 1).** Expected yield: tens to a few hundred patterns. The organic count is unknown until you filter (inference).
1. Download opXRD v11 (CC BY), take `CNRS/` (1,052 JSON). Keep entries whose composition is molecular organic: C and H present, only {C,H,N,O,S,P,F,Cl,Br,I,B,Si}, and atomic coordinates present.
2. Match each COD CIF to the 247k CSD set: CSD Python API, same formula plus cell match after Niggli reduction, or packing similarity. Record the refcode and whether the exact entry is in the xrd-clip database. Unmatched entries are still usable for a "COD structure added to the gallery" evaluation.
3. Add the DASH .xye files (~8) and the SDPD round-robin organic pattern(s).

**Tier B: self-measured polymorph families (weeks 1–3).** Target 60–80 patterns. Pick cheap compounds with several well-characterised forms already in the CSD:
- From Mayo 2023 (refcodes verified): acetaminophen HXACAN, D-mannitol DMANTL, caffeine NIWFEE, urea UREAXX, uracil URACIL, progesterone PROGST, 1,4-dicyanobenzene TEPNIT.
- Suggested additions (refcodes and polymorph preparation routes UNVERIFIED, check in CSD): carbamazepine (CBMZPN family), glycine α/β/γ (GLYCIN family), ROY (QAXMEH family), aspirin (ACSALA), ibuprofen (IBPRAC; the failed spot check), sulfathiazole, indomethacin γ/α.
- Measurement protocol: two preparations per sample, back-loaded and top-pressed, to vary preferred orientation. One fast scan and one slow scan. Measure LaB6/Si once per session for the instrument function. If a cryostream or Peltier stage exists, also collect 1–2 compounds at 100–150 K. That gives a direct, controlled measurement of the thermal-shift effect at matched temperature.

**Tier C: requests and scraping (weeks 2–6, in parallel).**
- Email pdj@icdd.com for the Kaduk Powder Diffraction API raw CIFs, starting with ~30 recent ones.
- Scrape the IUCr Acta E/C/B Rietveld supplementary CIFs of organic structures.
- Ask Johnson's group for the 7-compound and ROY high-throughput patterns.
- Check whether CSD powder-study CIFs carry profiles.

**Evaluation design.**
- Rank against the full 247k gallery.
- Report hit@k for (i) the exact refcode, (ii) any refcode of the same compound (any temperature or polymorph), and (iii) a polymorph-discrimination accuracy within each family.
- Stratify by data-collection-temperature gap between the CSD entry and the pattern, and by source (lab Cu vs synchrotron converted to Cu-2θ).
- Keep a fine-tuning split: never fine-tune on the patterns used for test.
- Report each result with and without a cell-refinement or reranking step (Section 3).

## 3. Sim-to-real methods and reported gains (verified sources)

| Paper | Domain | Realism gap addressed | Method | Reported result (as stated) |
|---|---|---|---|---|
| Szymanski et al. 2021, Chem. Mater. 10.1021/acs.chemmater.1c01071 (arXiv 2103.16664; code github njszym/XRD-AutoAnalyzer) | Inorganic, Li-Mn-Ti-O-F | Strain, texture, domain size, off-stoichiometry; tested on noise, amorphous background, Kapton, thick-sample shift | Physics-informed augmentation: cells strained up to ±4%; texture rescaling intensities by up to ±50%; domain-size broadening; 150 augmented spectra per phase; hypothetical solid solutions; ensemble CNN + branching | Simulated test: CNN 94% vs JADE 78%. **Experimental (Table 1, ~76 spectra): 93.4% vs JADE 71.4%.** Thick-sample 2θ shift 5/6 vs 2/6. Training with ±6% strain gave 86% on the ±4% test set (over-augmentation hurts) |
| Schuetzke et al. 2021, IUCrJ 8, 408, 10.1107/S2052252521002402 | Iron ores (28 phases), cements (76) | Unit cell, preferred orientation (March–Dollase), crystallite size, air scatter | Ablation of augmentation groups | "variation of unit-cell parameters leads to the largest improvement with a plus of over 40%" (F1). PO and crystallite size give "only small enhancements". PO-only variation *hurt* the cement set. All variations combined performed best. ~100k (ores) and ~250k (cements) synthetic mixtures needed |
| Schopmans, Reiser & Friederich 2023, arXiv 2303.11699 (Digital Discovery) | Space group from synthetic crystals; RRUFF test | Background, noise, impurity phases | Gaussian-process background, additive and multiplicative noise, random impurity phase | RRUFF 942 patterns: **top-1 25.2%, top-10 >60%**. "it is essential to include impurity phases in the training data" |
| Salgado et al. 2023, npj Comput. Mater. 9, 214, 10.1038/s41524-023-01164-8 | Space group, inorganic | Small real fine-tune | 50% of RRUFF added to simulated training | +11 percentage points 230-way space-group accuracy on the other half. Quoted from the opXRD paper; the primary text could not be fetched |
| Xu et al. 2025, XCCP, arXiv 2511.04055 | Contrastive XRD↔crystal-graph retrieval, MP 155,003 structures, PyXtal sims | None explicit (intensity normalisation only) | Zero-shot on "773 experimental XRD patterns form the HKUST-subset of opXRD" | Text: top-1 56.14%, top-3 84.61%, top-5 93.40%, top-10 99.74%. Table 2: 57.18/84.73/93.66/99.61. **Caveats:** opXRD lists HKUST as 520 patterns (4% labeled); the gallery size and whether the composition prefilter was applied are unclear; 599/773 are triclinic. Treat as weak evidence |
| Li et al. 2026, RealPXRD-Solver, arXiv 2603.00965 | Structure determination, inorganic only | Background, broadening, 2θ range and step | Universal d–I encoder applied identically to sim and preprocessed exp; ±0.1° shifts, noise, intensity ×0.8–1.2 | CNRS (126): top-1/top-20 77.9/91.9%. RRUFF (269): 78.8/92.9%. Lattice-conditioned 77.9% vs lattice-free 52.4% top-1 on CNRS |
| Mayo, Marczenko & Johnson 2023, Chem. Sci. 10.1039/d3sc00168g | **Organic**, polymorph ID | **Low-T or high-P reference vs RT pattern (cell mismatch)** | Variable-cell matching (VC-xPWDF): deform each candidate cell to the indexed experimental cell, then compare | The correct structure is identified for all 7 compounds at moderate and low quality. Match scores typically <0.1 (caffeine 0.0114) vs wrong polymorph e.g. 0.2087. Preferred orientation (mannitol) raises scores to ~0.096–0.099 |
| Johansen et al. 2026, Digital Discovery 10.1039/d6dd00088f (deCIFer, arXiv 2502.02189) | Generative CSP, inorganic | Noise, background, peak asymmetry, Scherrer broadening | Robustness tests; training with noise and broadening | "adapts smoothly to signal distortions". Experimental Si and CeO2 recovered; Fe2O3 and nano-CeO2 fail |
| Wang et al. 2026, Xtalyst, arXiv 2608.22400 (**WITHDRAWN** by the authors) | Retrieval + agent | Peak-position drift | Drift correction, real-spectrum fine-tuning, peak-aligned reranking | Abstract: "synthetic denoising gives no measurable lift on real spectra, whereas correcting a small peak-position drift more than doubles median retrieval correlation". Withdrawn, so this is a direction only |
| Wu et al. 2024, UstcUnfold, arXiv 2411.06062 | Organic and hybrid (CCDC, >1.1M sim pairs) | Missing peaks, noise | Peak masking in pretraining; patterns "pre-processed to mitigate … noise" | Only 2 experimental examples; unit-cell MAE ~0.115. Needs 2θ max ≥ 60° |
| Lee et al. 2020, Nat. Commun. 10.1038/s41467-019-13749-3 | Sr-Li-Al-O | — | Synthetic multiphase training | "nearly 100% accuracy for phase identification on experimental XRD data". From a search snippet; not opened |

**Realism gaps ranked for xrd-clip (inference from the evidence above):**
1. Peak position / cell mismatch, including thermal expansion. This is the largest single effect in Schuetzke (+40%) and Xtalyst (2×), it is the explicit motivation for VC-xPWDF, and it fits the ibuprofen failure.
2. Impurity phases and background (Schopmans).
3. Preferred orientation, which matters most for plate- and needle-forming organics (mannitol).
4. Kα2 doublet. Not quantified in any paper fetched here. Cu Kα1/Kα2 ≈ 2:1 doublet physics is textbook and not cited here; check that the simulator includes it.
5. Pure denoising has little value (Xtalyst, withdrawn).

**Gaps in the evidence (searches listed):**
- Test-time augmentation for PXRD: no paper found. Search: `"test-time augmentation" powder X-ray diffraction neural network`.
- Adversarial or unsupervised domain adaptation, sim→exp PXRD, with numbers: none verified. Search: "domain adaptation unsupervised X-ray diffraction deep learning simulated to experimental patterns adversarial". It returned Bai et al. imitation refinement (Cornell) and an MRS 2025 abstract, neither opened.

**Implication (inference).** For a retrieval model, the cheapest high-value fixes are:
- Anisotropic cell-strain augmentation sized to real thermal expansion (Section 4), rather than isotropic strain.
- Impurity-phase superposition and Kα2.
- A test-time variable-cell rerank of the top-k candidates (VC-xPWDF style, or a Pawley/Le Bail fit against each candidate).
- Fine-tuning on the opXRD unlabeled and labelled patterns.

## 4. Thermal expansion: low-temperature CSD structures vs room-temperature patterns

**Primary numbers:**
- **Bond 2021**, Acta Cryst. B, 10.1107/S2052520621003309 (Cambridge repository 10.17863/CAM.72939). 6,201 CSD structure families with a room-temperature structure plus at least one lower temperature down to 90 K.
  - Volumetric α_V at 298 K: reliable subset (210 families) mean **173 ppm/K, SD 47**; full set mean **161 ppm/K, SD 51**.
  - V(T) is approximately linear over 90–300 K.
  - Principal linear coefficients α_L: positively skewed, modelled as two half-normals centred on **33 ppm/K** (SD 40 below, 56 above).
  - Uniaxial NTE ≈ 30%; biaxial NTE < 5%.
  - Anisotropy (α_max − α_min)/α_V: half-normals with common mean **0.44** (SD 0.24 below).
- **van der Lee & Dumitrescu 2021**, Chem. Sci., 10.1039/d1sc01076j. 4,719 structures (4,093 compounds plus 626 polymorphs).
  - Volumetric: mean 168.6, median 168.3, SD 72.5 MK⁻¹ (n = 745, R² > 0.90). Global mean 156.0.
  - Uniaxial: mean 71.4, median 62.6, SD 69.9 MK⁻¹ (n = 1,129).
  - ~34% of structures may have at least one axis with negative thermal expansion; 127 candidates for negative volumetric expansion.
- **CSD 2026 statistics:** 57.1% of CSD entries are non-room-temperature studies. The typical xrd-clip gallery entry is therefore a low-temperature cell (inference).

**Effect size for a 100 K → 295 K gap (ΔT = 195 K; my arithmetic, linear approximation):**

| Quantity | Value | Peak shift Δ2θ (Cu Kα), from Δ2θ ≈ 2·ε·tanθ |
|---|---|---|
| ΔV/V, mean (161–173 ppm/K) | **+3.1% to +3.4%** (±1 SD: ~2.2% to 4.3%) | — |
| Isotropic-equivalent linear strain | ~1.0–1.1% | 2θ = 10°: ~0.11°; 20°: ~0.22°; 30°: ~0.34° |
| Typical principal axis (33 ppm/K) | +0.6% | 20°: ~0.13° |
| Upper-side axis (33 + 56 = 89 ppm/K) | +1.7% | 20°: ~0.35° |
| NTE axis (e.g. −20 ppm/K) | −0.4% | Shifts the opposite way |
| Axis spread at anisotropy 0.44 (α_max − α_min ≈ 73 ppm/K) | ~1.4% differential between axes | Different reflections move by different amounts |

Interpretation (inference):
- Lab Cu FWHM is typically ~0.05–0.1° 2θ (not sourced here). A room-temperature query therefore displaces mid-angle peaks by **several FWHM**, and does so *anisotropically*.
- A model trained with small isotropic strain, or with strain sized to measurement error, would see a room-temperature pattern of a low-temperature CSD entry as out of distribution.
- Recommended augmentation: per-axis strain on the metric tensor, drawn from the half-normal α_L distribution × a random ΔT in 0–200 K, with occasional negative axes. Roughly ε_i ∈ [−0.5%, +2%] per axis, plus an isotropic component matching ΔV/V up to ~4–5%.
- Also report hit@k stratified by the CSD data-collection temperature.

## 5. UNVERIFIED list (not confirmed from a fetched primary page, or ambiguous)

- Organic-pattern count inside opXRD CNRS/COD: **not stated anywhere I could fetch**; must be computed from the files.
- Whether licensed CSD powder-study CIFs include raw profile loops (`_pd_meas_counts`).
- ICDD PDF-5+ 2026 experimental digital pattern count: the extracted table is misaligned (22,238 or 25,064). The 2025 figure of 20,833 comes from a search snippet. ICDD ML-use licence terms and Purdue licence status are unchecked.
- Salgado et al. 2023 primary numbers. The +11 pp is quoted via the opXRD paper. XCCP's intro cites Salgado as "67% … crystal systems and only 36% for space groups on 2,253 test samples" (secondary). nature.com was blocked.
- Lee et al. 2020 "nearly 100% … experimental" is a search snippet only.
- Linus Pauling File: 21,700 experimental patterns (search snippet of the opXRD paper).
- Data availability of the Sparrow et al. 2024 ROY high-throughput patterns.
- Refcode families for carbamazepine (CBMZPN), glycine (GLYCIN), ROY (QAXMEH), aspirin (ACSALA), ibuprofen (IBPRAC), and preparation routes for their polymorphs. These are from memory; verify in the CSD.
- Count of organic Rietveld papers with pdCIF profiles in IUCr journals (journals.iucr.org returned 403 to this agent).
- Kaduk Powder Diffraction API count (~100+) is inferred from a 40-item Crossref sample of 162 Kaduk-authored items. The ICDD request route is verified for repotrectinib only.
- Kα2 ratio and typical lab FWHM values are textbook, not fetched.
- Xtalyst (arXiv 2608.22400) is withdrawn; cite only as a direction.
- SimXRD-4M (arXiv 2406.15469) claims real-data generalisation in secondary summaries, but the abstract I fetched does not state it.
