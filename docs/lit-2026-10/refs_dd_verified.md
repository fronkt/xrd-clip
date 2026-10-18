# Verified reference ledger: xrd-clip Digital Discovery manuscript

Compiled 2026-10-12. Every field below was read from a primary record: Crossref API (`api.crossref.org/works/<doi>`), arXiv export API, OpenAlex (no key), Europe PMC REST, Zenodo API, doi.org resolution, publisher or proceedings landing pages (PMLR, JMLR, proceedings.neurips.cc, neurips.cc), and open full text (arXiv HTML/PDF, Europe PMC full-text XML, the Oviedo CC-BY postprint, the RealBind Zenodo release). No browser automation, no proxy, no keyed API. Paywalled full texts were not read. A field marked **UNVERIFIED** could not be confirmed from a primary record.

Verdict key: **SUPPORTED** = the source says it. **PARTLY** = the source supports the gist, but a number, a scope or a name must change (the fix is given). **NOT SUPPORTED** = the source does not say it. **CANNOT CHECK** = no open text available to check the claim.

Quotes are verbatim from the source named next to them. "[sic]" marks source typos.

## Summary

| # | Suggested key | Verdict | Action needed |
|---|---|---|---|
| 1 | groom2016csd (+ ccdc2026stats) | SUPPORTED | Cite the count to the CCDC statistics sheet: 1,431,347 structures on 1 Jan 2026 |
| 2 | gatesrector2019pdf | SUPPORTED | none |
| 3 | bauer2001ritonavir, cruzcabeza2015polymorph | SUPPORTED | none |
| 4 | degelder2001wcc | SUPPORTED | The triangle form was confirmed through a secondary open source (primary full text is paywalled) |
| 5 | oterodelaroza2024vcgpwdf | PARTLY | The method in this paper is **VC-GPWDF**, not VC-PWDF/VC-xPWDF (those are Mayo et al. 2022/2023) |
| 6 | lai2025xtalnet, xu2026xccp | PARTLY | Both include some experimental testing; XCCP has a new journal title and year (2026) |
| 7 | johansen2026decifer, johansen2026realworld | PARTLY | The TMLR deCIFer paper is synthetic-only; the experimental tests are in a separate *Digital Discovery* 2026 paper |
| 8 | yang2026pxtal | SUPPORTED | none |
| 9 | srivastava2026realbind | PARTLY | Not "inorganic" only: the measured set includes CNRS/COD MOFs and molecular crystals. The bare DOI does not resolve: cite `/v2` |
| 10 | lange2024smolnet | SUPPORTED | Add scope: pairwise same/different-form classification; SMolNet wins AUROC/AUPRC |
| 11 | wang2026matdiffract, segal2026xrdiff | MatDiffract PARTLY, XRDiff SUPPORTED | MatDiffract: JADE ran with element priors and a different library; "non-learned" is our reading, not the paper's word |
| 12 | bushuiev2024massspecgym | PARTLY | MIST is a fingerprint predictor, not a joint embedding; the subsets differ; a joint-embedding paper (JESTR) beat CFM-ID |
| 13 | ferraridacrema2019progress | PARTLY | "Most" must become "6 of the 7 reproducible (of 18 examined)"; the first author's surname is Ferrari Dacrema |
| 14 | musgrave2020reality | SUPPORTED | LNCS volume number UNVERIFIED |
| 15 | radovanovic2010hubs | SUPPORTED | none |
| 16 | conneau2018muse | SUPPORTED | none |
| 17 | schnitzer2012mp | SUPPORTED | none |
| 18 | hollarek2025opxrd (+ opxrd2025zenodo) | PARTLY | The paper does not say "Rietveld-refined"; COD-sourced data are CC0, the Zenodo deposit is CC BY 4.0 |
| 19 | chisholm2005compack | PARTLY | "15 molecules" is the CCDC software default, not in the COMPACK abstract |
| 20 | ong2013pymatgen | PARTLY | Library citation is fine; cite the XRD calculator as the module and version used |
| 21 | radford2021clip, oord2018cpc | SUPPORTED | CPC is for InfoNCE only, not dual encoders |
| 22 | glick2025csp | SUPPORTED | Still an arXiv preprint (no journal version found) |
| 23 | schopmans2023synthetic | SUPPORTED | one-line claim given below |
| 24 | bond2021thermal | PARTLY | Bond gives coefficients, not percentages; 100 K to 298 K implies about 3% in volume (about 1% per axis) |
| 25 | rietveld1969 | SUPPORTED | none |
| 26 | paszke2019pytorch | SUPPORTED | Pages UNVERIFIED (the official NeurIPS BibTeX has none) |
| 27 | field2007cluster (preferred), davison1997bootstrap | SUPPORTED (Field & Welsh) | Davison & Hinkley section number UNVERIFIED |
| 28 | dollase1986, caglioti1958 | Dollase SUPPORTED, Caglioti CANNOT CHECK | Caglioti primary text not open; the attribution is standard |
| 29 | recency sweep | No organic-specific hit | RealBind is a partial hit (some molecular crystals in its measured test set) |

---

## 1. Cambridge Structural Database

- **Key:** `groom2016csd`
- **Authors:** Groom, Colin R.; Bruno, Ian J.; Lightfoot, Matthew P.; Ward, Suzanna C.
- **Title:** The Cambridge Structural Database
- **Venue:** *Acta Crystallographica Section B: Structural Science, Crystal Engineering and Materials* **72**(2), 171–179 (2016; online 1 Apr 2016). Open access, CC BY 2.0 UK.
- **DOI:** 10.1107/S2052520616003954
- **Claim:** the CSD is the curated repository of experimentally determined organic and metal–organic crystal structures, with more than 1 million entries.
- **Verdict:** SUPPORTED.
- **Quotes (abstract):** "The Cambridge Structural Database (CSD) contains a complete record of all published organic and metal–organic small-molecule crystal structures." "All structures are processed both computationally and by expert structural chemistry editors prior to entering the database."
- **Current count:** the CCDC sheet "CSD Entries: Summary Statistics, 1 January 2026" gives "Total No. of structures 1,431,347". URL: https://www.ccdc.cam.ac.uk/media/CSD-Entries-Summary-Statistics-2026.pdf. The same sheet gives powder diffraction studies 10,408 (0.7%), non-room-temperature studies 817,929 (57.1%), polymorphic structures 45,729 (3.2%) and organic structures 655,284 (45.8%).
- **Suggested key for the count:** `ccdc2026stats` (@misc, CCDC, 2026, the URL above, accessed 2026-10-12).
- **Note:** the abstract says "published" structures and also mentions CSD Communications (direct deposits). "Experimentally determined" is accurate for CSD content, but the abstract's own word is "published".

## 2. Powder Diffraction File

- **Key:** `gatesrector2019pdf`
- **Authors:** Gates-Rector, Stacy; Blanton, Thomas
- **Title:** The Powder Diffraction File: a quality materials characterization database
- **Venue:** *Powder Diffraction* **34**(4), 352–360 (2019; online 6 Nov 2019). CC BY 4.0.
- **DOI:** 10.1017/S0885715619000812
- **Claim:** classical phase identification searches reference-pattern databases (search-match).
- **Verdict:** SUPPORTED.
- **Quotes (abstract):** "The ICDD's Powder Diffraction File™ (PDF®) is a database of inorganic and organic diffraction data used for phase identification and materials characterization by powder diffraction." "With data mining and phase identification software available in PDF products … the 1 000 000+ published PDF entries serve a wide range of disciplines".
- **Note:** the full-text phrase "phase identification by search-match processing" (and a Hanawalt-method passage) came through a WebFetch summary of the Cambridge Core HTML. Re-check the exact wording before quoting it; the abstract alone supports the claim.

## 3. Polymorphism matters

**3a.**
- **Key:** `bauer2001ritonavir`
- **Authors:** Bauer, John; Spanton, Stephen; Henry, Rodger; Quick, John; Dziki, Walter; Porter, William; Morris, John
- **Title:** Ritonavir: An Extraordinary Example of Conformational Polymorphism
- **Venue:** *Pharmaceutical Research* **18**(6), 859–866 (2001). PMID 11474792.
- **DOI:** 10.1023/A:1011052932607
- **Verdict:** SUPPORTED (polymorph identity matters for pharmaceuticals).
- **Quote (abstract, Europe PMC):** "In the summer of 1998, Norvir semi-solid capsules supplies were threatened as a result of a new much less soluble crystal form of ritonavir."

**3b.**
- **Key:** `cruzcabeza2015polymorph`
- **Authors:** Cruz-Cabeza, Aurora J.; Reutzel-Edens, Susan M.; Bernstein, Joel
- **Title:** Facts and fictions about polymorphism
- **Venue:** *Chemical Society Reviews* **44**(23), 8619–8635 (2015). PMID 26400501.
- **DOI:** 10.1039/C5CS00227C
- **Verdict:** SUPPORTED (polymorphism is common).
- **Quote (abstract, Europe PMC):** "we found that one in three compounds in the CSD are polymorphic whilst at least one in two compounds from the Roche and Lilly set display polymorphism with a higher estimate of up to three in four when compounds are screened intensively."

## 4. de Gelder weighted cross-correlation (WCC)

- **Key:** `degelder2001wcc`
- **Authors:** de Gelder, R.; Wehrens, R.; Hageman, J. A.
- **Title:** A generalized expression for the similarity of spectra: application to powder diffraction pattern classification
- **Venue:** *Journal of Computational Chemistry* **22**(3), 273–289 (2001).
- **DOI:** 10.1002/1096-987X(200102)22:3<273::AID-JCC1001>3.0.CO;2-0
- **Claim:** WCC with a triangular weight of half-width *l* compares patterns tolerantly to peak shifts.
- **Verdict:** SUPPORTED. The shift tolerance comes from the primary abstract. The triangle form was confirmed through a secondary open source; the primary full text is paywalled and was not read.
- **Quote (abstract, via OpenAlex):** "A generalized expression is given for the similarity of spectra, based on the normalized integral of a weighted crosscorrelation function. … The benefits of this criterion are that it properly recognizes shifted but otherwise similar details in spectra and that the resulting similarity measure is normalized."
- **Triangle form (Otero-de-la-Roza 2024 postprint, Eq. 4):** "In de Gelder's work a triangle function function [sic] was proposed as weight function (de Gelder et al., 2001)": w_dG(r) = 1 − |r|/l for |r| < l, and 0 for |r| ≥ l. So *l* is the half-width of the triangle's base.
- **Note:** the paper's application is classification of *calculated* patterns of cephalosporin complexes.

## 5. Variable-cell powder similarity (Otero-de-la-Roza 2024)

- **Key:** `oterodelaroza2024vcgpwdf`
- **Author:** Otero-de-la-Roza, Alberto (single author)
- **Title:** Powder-diffraction-based structural comparison for crystal structure prediction without prior indexing
- **Venue:** *Journal of Applied Crystallography* **57**(5), 1401–1414 (2024; online 30 Aug 2024).
- **DOI:** 10.1107/S1600576724007489
- **Open copies:** preprint ChemRxiv 10.26434/chemrxiv-2024-hdt4m. Postprint https://digibuo.uniovi.es/dspace/handle/10651/76234, which is the copy read here.
- **Claim:** variable-cell comparison that optimizes cell strain; implemented in critic2; costs tens of seconds per candidate.
- **Verdict:** PARTLY, for naming only. The substance is SUPPORTED.
- **Naming fix:** this paper introduces **GPWDF** and **VC-GPWDF**. **VC-PWDF** is Mayo, Otero-de-la-Roza & Johnson, *CrystEngComm* **24**(47), 8326–8338 (2022), DOI 10.1039/D2CE01080A. **VC-xPWDF** is Mayo, Marczenko & Johnson, *Chem. Sci.* **14**(18), 4777–4785 (2023), DOI 10.1039/D3SC00168G. Those two records were checked on Crossref; their content was not re-read here. Suggested keys: `mayo2022vcpwdf`, `mayo2023vcxpwdf`.
- **Quotes:**
  - Abstract: "a variable-cell similarity index (VC-GPWDF) is also proposed that assigns a high similarity score to structures that differ only by a lattice deformation"; "They do not require indexing of the experimental pattern or a guess of the space group".
  - Postprint: "It is implemented in the open-source critic2 program".
  - Postprint: "The average run time for the VC-GPWDF calculations in Figure 6a is 31 seconds on a single processor, with a maximum run time of 102 seconds."
  - Postprint, default search box: "maximum cell length elongation of up to 10% and maximum cell angle deformation of 5°".
- **Note:** the 31 s timing is for comparing a candidate against a peak-fitted experimental pattern (Figure 6a).

## 6. Contrastive PXRD–structure models (XtalNet, XCCP)

**6a. XtalNet**
- **Key:** `lai2025xtalnet` (refs.bib key fine)
- **Authors (journal, 12):** Lai, Qingsi; Xu, Fanjie; Yao, Lin; Gao, Zhifeng; Liu, Siyuan; Wang, Hongshuai; Lu, Shuqi; He, Di; Wang, Liwei; Zhang, Linfeng; et al. (Wang, Cheng; Ke, Guolin). arXiv v3 lists 11 authors, without Linfeng Zhang. Use the journal list.
- **Title:** End-to-End Crystal Structure Prediction from Powder X-Ray Diffraction
- **Venue:** *Advanced Science* **12**(8), 2410722 (2025; online 4 Jan 2025). CC BY 4.0.
- **DOI:** 10.1002/advs.202410722. arXiv:2401.03862v3.

**6b. XCCP**
- **Key:** `xu2026xccp` (refs.bib `xu2025xccp` needs updating)
- **Authors (9):** Xu, Chenlei; Su, Tianhao; Xiong, Jie; Wu, Yue; Dong, Shuya; Jiang, Tian; He, Mengwei; Chen, Shuai; Zhang, Tong-Yi
- **Journal title:** "KAN-enhanced contrastive learning: the accelerator of crystal structure identification from XRD patterns". This differs from the arXiv title used in refs.bib.
- **Venue:** *npj Computational Materials* **12**, 144 (2026; online 28 Feb 2026). CC BY-NC-ND 4.0.
- **DOI:** 10.1038/s41524-026-02015-y. arXiv:2511.04055v1 (6 Nov 2025).

**Assessment of both**
- **Claim:** contrastive PXRD–structure retrieval models trained and evaluated on simulated inorganic/MOF data.
- **Verdict:** PARTLY. Training is simulated for both, but both report some experimental testing.
- **XtalNet quotes:** "PXRD … from known crystal structures using GSAS software as our training data"; the evaluation uses "two MOF datasets (hMOF-100 and hMOF-400)". The retrieval test is a "database retrieval task" on simulated data. However, Section 2.5 is "Application of XtalNet to Real Experimental PXRD Patterns": two MOF case studies, for generation, not retrieval.
- **XCCP quotes:** "The dataset contains 155,003 crystallographic information files (CIFs) from the Materials Project"; "Powder XRD patterns were simulated from CIFs using PyXtal". However: "A broader zero-shot evaluation on 773 experimental XRD patterns form [sic] the HKUST-subset of opXRD" (average top-1 57.18%, inorganic).
- **Fix:** "trained on simulated inorganic (XCCP, Materials Project) or MOF (XtalNet, hMOF) patterns; experimental tests are limited to two MOF case studies (XtalNet) and a zero-shot test on 773 inorganic opXRD patterns (XCCP)". Also note that XtalNet is a generative CSP model whose contrastive module (CPCP) is one component.

## 7. deCIFer

**7a. Method paper**
- **Key:** `johansen2026decifer` (refs.bib `johansen2025decifer` needs updating)
- **Authors (6):** Johansen, Frederik Lizak; Friis-Jensen, Ulrik; Dam, Erik Bjørnager; Jensen, Kirsten Marie Ørnsbjerg; Mercado, Rocío; Selvan, Raghavendra
- **Title:** deCIFer: Crystal Structure Prediction from Powder Diffraction Data using Autoregressive Language Models
- **Venue:** *Transactions on Machine Learning Research* (2026).
  - arXiv:2502.02189v4 (13 Apr 2026) carries the comment "v4: accepted to TMLR (https://openreview.net/forum?id=LftFQ35l47)".
  - The year 2026 is from ML Anthology (secondary).
  - OpenReview returned a challenge page, so the TMLR month and volume are **UNVERIFIED**. A web snippet said "Vol. 2026-April"; that is unconfirmed.

**7b. Experimental follow-up**
- **Key:** `johansen2026realworld`
- **Authors (5):** Johansen, Frederik Lizak; Sapnik, Adam F.; Dam, Erik Bjørnager; Selvan, Raghavendra; Jensen, Kirsten M. Ø.
- **Title:** Tackling real-world crystal structure prediction from powder X-ray diffraction data
- **Venue:** *Digital Discovery* **5**(9), 3871–3882 (2026). CC BY-NC 3.0. PMC13470798. Preprint ChemRxiv 10.26434/chemrxiv.15000236.
- **DOI:** 10.1039/d6dd00088f

**Assessment of both**
- **Claim:** autoregressive crystal-structure generation conditioned on PXRD, evaluated including experimental patterns.
- **Verdict:** PARTLY. The TMLR paper is synthetic-only. The experimental evaluation is in 7b.
- **TMLR quote (abstract):** "Validated across diverse synthetic datasets representative of challenging inorganic materials, deCIFer achieves a 94% structural match rate."
- **DD quote (abstract):** "Experimental PXRD tests recover the known structures of Si and CeO 2 and expose the expected limitations for lower-symmetry Fe 2 O 3 and nanocrystalline CeO 2."
- **Fix:** cite 7a for the method and 7b for the experimental test (four inorganic samples). Since the target venue is *Digital Discovery*, 7b is the natural citation.

## 8. PXtal

- **Key:** `yang2026pxtal`
- **Authors (6):** Yang, Zhuoran; Collins, Christopher M.; Peng, Bei; Daniels, Luke M.; Rosseinsky, Matthew J.; Gusev, Vladimir V.
- **Title:** PXtal: Learning to Align Powder X-Ray Diffraction and Crystal Structures under Information Asymmetry across Modalities
- **Venue:** arXiv:2610.10653v1 (7 Oct 2026). Preprint, not peer reviewed.
- **Claim:** an optimal-transport replacement for InfoNCE in PXRD–structure retrieval; MP-100 Acc@1 67.26 to 74.45; simulated only.
- **Verdict:** SUPPORTED.
- **Quotes:**
  - "PXtal-CLIP is the matched InfoNCE baseline."
  - "On MP-100, it ranks the designated crystal first for 74.45% of PXRD patterns, compared with 67.26% for PXtal-CLIP" (Table 2; the variant is PXtal-UOT-GKL, i.e. unbalanced OT with coupling-level generalized KL).
  - On being simulated only: the patterns come from a GSAS-II forward model (Appendix A), and "Future work will extend this approach to measured, multiphase, and open-set PXRD."

## 9. RealBind

- **Key:** `srivastava2026realbind`
- **Author:** Srivastava, Preyansh (independent researcher; single author)
- **Title:** RealBind: a leakage-controlled benchmark for experimental PXRD and Raman structure retrieval
- **Venue:** ChemRxiv preprint, not peer reviewed. v1 29 Sep 2026 (DOI 10.26434/chemrxiv.15009668/v1); v2 7 Oct 2026 (DOI **10.26434/chemrxiv.15009668/v2**).
- **DOI warning:** the unversioned DOI 10.26434/chemrxiv.15009668 returns 404 at doi.org and at Crossref. Cite the `/v2` DOI.
- **Data and code:** Zenodo 10.5281/zenodo.23142866 (concept 10.5281/zenodo.23140153), 4 Oct 2026. Record licence CC BY 4.0; the datasheet puts code under MIT.
- **Claim:** on measured inorganic patterns, plain correlation beat a contrastive model (Recall@10 82.8% vs 34.4% for the corruption-trained model).
- **Verdict:** PARTLY. The numbers are exact; "inorganic" is wrong.
- **Quote (abstract, v2 via OpenAlex):** "On a 492-candidate prototype-held-out library, a contrastive model trained on simulated PXRD reaches 78.5% Recall@10 on simulated queries but 5.3% on measured queries; physics-inspired corruption raises measured-query Recall@10 to 34.4%. A model-free correlation against simulated candidate patterns, with no preprocessing beyond the common Q-grid resampling and the measured[-]window mask, achieves 82.8% Recall@10 on the same test library."
- **Chemistry (release DATASHEET.md):** "3,288 crystal structures (698 from opXRD, 2,590 from RRUFF)"; "the opXRD structures are dominated by one contributor (CNRS: metal-organic frameworks and molecular crystals), the RRUFF structures by minerals". The simulation (training) corpus is "92,597 Materials Project entries".
- **Our count from the released manifests:** `queries_pxrd.csv` and `structures.csv` show the 492-candidate library's 221 measured test queries split as 116 RRUFF and 105 opXRD. By an elemental heuristic (C and H present, no metal or metalloid), that is 23 organic, 20 C,H plus metal, and 178 other. Across all 1,468 measured patterns, 164 are organic by the same heuristic, all from opXRD-CNRS. This is our inference from the files, not a number stated by the author. The expected_results file confirms 82.8% = "Table 3 (492 candidates, 221 queries)" and model-free R@1 = 55.7%.
- **Fix:** "on measured patterns (mostly minerals and inorganic or hybrid COD entries, with a minority of molecular crystals), a model-free correlation reached 82.8% Recall@10 against 34.4% for a contrastive model trained on simulated Materials Project patterns with corruption augmentation (5.3% without)".
- **Novelty impact:** this corrects refute_1 #13a and refute_2 #5, which called RealBind inorganic. RealBind does put some real molecular-crystal patterns into a learned-vs-correlation retrieval test. It is not organic-specific, and its learned model never saw organic training data.

## 10. SMolNet

- **Key:** `lange2024smolnet`
- **Authors (5):** Lange, Julius; Komissarov, Leonid; Lang, Rene; Enkelmann, Dennis Dimo; Anelli, Andrea (Roche)
- **Title:** Automatic solid form classification in pharmaceutical drug development
- **Venue:** arXiv:2411.03308v1 (5 Nov 2024). Poster at *AI4Mat-2024: NeurIPS 2024 Workshop on AI for Accelerated Materials Design* (https://neurips.cc/virtual/2024/107633). The PDF footer reads "38th Conference on Neural Information Processing Systems (NeurIPS 2024)". It is a workshop paper, not in the main proceedings.
- **Claim:** on experimental patterns of pharmaceutical polymorphs, de Gelder WCC matched or beat a Siamese network on accuracy and F1.
- **Verdict:** SUPPORTED, with scope caveats.
- **Table 1 (AUROC / AUPRC / Acc / F1):**
  - De Gelder: 0.92 / 0.46 / 0.92 / 0.93
  - SMolNet variants: Acc 0.87–0.89, F1 0.79–0.89
  - Best SMolNet (SigLIP): 0.98 / 0.65 / 0.89 / 0.80
- **Quotes:**
  - "Hard classifier metrics, i.e. Accuracy and F1 score are best for these models [naive and de Gelder]".
  - "Although the de Gelder method produces the best F1 score, we argue that the increased AU{ROC,PRC} scores of the machine learning models is particularly significant".
  - Data: "proprietary and contains 3750 experimental XRPD patterns measured across 16 organic compounds … for a total of 24 different solid forms".
- **Caveats to state:** the task is pairwise same/different-form classification, not structure retrieval; SMolNet wins AUROC and AUPRC; the data are proprietary.

## 11. MatDiffract and XRDiff

**11a. MatDiffract**
- **Key:** `wang2026matdiffract`
- **Authors (8):** Wang, Hongqing; Chen, Mingwei; Luo, Hongjie; Yin, Wen; Qi, Fazhi; Chai, Xuqing; Liu, Miao; Hou, Fengyao
- **Title:** MatDiffract: A Material-Informed Automated Analysis Platform for X-ray Powder Diffraction
- **Venue:** arXiv:2607.20880v2 (v1 23 Jul 2026; v2 26 Jul 2026)
- **Claim:** a non-learned peak-feature index beat JADE on real patterns.
- **Verdict:** PARTLY. The numbers are right, but the comparison is not like-for-like, and "non-learned" is our reading.
- **Quotes:**
  - "The peak features of position, intensity, width, and local-profile are extracted from each simulated pattern and stored as indexable vectors." No encoder network is described.
  - "JADE achieved a Top-1 accuracy of 68.7% and a Top-10 accuracy of 83.1%. The initial vector retrieval stage of MatDiffract achieved a Top-1 accuracy of 89.0% and a Top-10 accuracy of 97.2%."
  - "JADE was operated with prior elemental information".
- **Other caveats:** the 875 test patterns are RRUFF minerals (inorganic), and the reference library is different (Atomly DFT, perturbation-augmented).
- **Fix:** "a hand-crafted peak-feature vector index (89.0% top-1) outperformed element-constrained JADE (68.7%) on 875 RRUFF patterns, with different reference libraries".

**11b. XRDiff**
- **Key:** `segal2026xrdiff`
- **Authors (4):** Segal, Nofit; Li, Mingda; Miller, Benjamin Kurt; Gómez-Bombarelli, Rafael
- **Title:** XRDiff: Crystal Structure Prediction from Powder X-Ray Diffraction Data Using Diffusion Models
- **Venue:** arXiv:2606.14003v1 (12 Jun 2026)
- **Claim:** full-profile encoders need background/noise augmentation for real data.
- **Verdict:** SUPPORTED.
- **Quotes:**
  - "Using the full PXRD spectrum, performance drops sharply relative to computational benchmarks, with match rates below 1% at a single attempt across both training datasets."
  - "augmentations affecting only peak appearance provide limited improvements over training on unmodified spectra, whereas introducing background noise leads to substantial gains in match rate. Under OQMD-Poly training, performance obtained using the full augmentation pipeline becomes largely comparable to that achieved using the proposed peak-based representation".
- **Caveats:** the metric is generative match rate with stoichiometry given; the test set is 148 RRUFF structures (inorganic); the abstract says peak descriptors outperform even augmented full spectra.

## 12. MassSpecGym

- **Key:** `bushuiev2024massspecgym`
- **Authors (30):** Bushuiev, Roman; Bushuiev, Anton; de Jonge, Niek F.; Young, Adamo; Kretschmer, Fleming; Samusevich, Raman; Heirman, Janne; Wang, Fei; Zhang, Luke; Dührkop, Kai; et al. (last author Pluskal, Tomáš)
- **Title:** MassSpecGym: A benchmark for the discovery and identification of molecules
- **Venue:** *Advances in Neural Information Processing Systems 37* (NeurIPS 2024), Datasets and Benchmarks Track, pp. 110010–110027 (pages per Crossref).
- **DOI:** 10.52202/079017-3491. arXiv:2410.23326v3.
- **Claim:** in molecule retrieval from MS/MS, simulate-then-compare beats learned joint embeddings.
- **Verdict:** PARTLY. The numbers check out; the framing does not.
- **Numbers:**
  - **Table 7** (spectrum-simulation challenge, "we restrict the task to spectra with the most abundant adduct ( A= [M+H] + )"): FraGNNet hit@1 **46.64%** (mass-based candidates) and 31.93% (formula-based bonus).
  - **Table 5** (molecule-retrieval challenge, all adducts): MIST hit@1 **14.64%** (mass) and 9.57% (formula).
- **Ranking procedure (quote):** "For each molecule G_i ∈ C, we predict a spectrum x̂_i and rank all candidates by decreasing cosine similarity between x̂_i and x."
- **Problems:**
  1. MIST is not a joint embedding: it "predicts a molecular fingerprint … and finally ranks the candidates by cosine similarity between the fingerprints". MassSpecGym benchmarks no joint-embedding model.
  2. The spectrum subsets differ ([M+H]+ only against all adducts).
  3. Counter-evidence: JESTR, a joint-embedding model (Kalia, Chen, Krishnan, Hassoun; arXiv:2411.14464v3), reports MassSpecGym rank@1 15.62% (mass) and 11.85% (formula), still far below FraGNNet. But it also states: "On carefully selected subsets of the MassSpecGym dataset, JESTR outperforms SIRIUS and CFM-ID on rank@1 by 31% and 238%". CFM-ID is a forward simulator.
- **Fix:** "Within MassSpecGym's own baselines, ranking candidates by similarity to forward-simulated spectra (FraGNNet, 46.64% hit@1, Table 7) exceeded the best spectrum-to-fingerprint retrieval model (MIST, 14.64%, Table 5), although on different adduct subsets." Do not generalize to "learned joint embeddings".

## 13. Neural recommenders reality check

- **Key:** `ferraridacrema2019progress`
- **Authors:** Ferrari Dacrema, Maurizio; Cremonesi, Paolo; Jannach, Dietmar. The surname is "Ferrari Dacrema"; cite as "Ferrari Dacrema et al."
- **Title:** Are we really making much progress? A worrying analysis of recent neural recommendation approaches
- **Venue:** *Proceedings of the 13th ACM Conference on Recommender Systems (RecSys '19)*, pp. 101–109 (2019).
- **DOI:** 10.1145/3298689.3347058. arXiv:1907.06902.
- **Claim:** well-tuned simple baselines matched or beat most neural recommenders tested.
- **Verdict:** PARTLY. The count needs to be stated exactly.
- **Quote (abstract):** "we considered 18 algorithms … Only 7 of them could be reproduced with reasonable effort. For these methods, it however turned out that 6 of them can often be outperformed with comparably simple heuristic methods, e.g., based on nearest-neighbor or graph-based techniques. The remaining one clearly outperformed the baselines but did not consistently outperform a well-tuned non-neural linear ranking method."
- **Fix:** "six of the seven neural recommenders that could be reproduced (of 18 examined) were often outperformed by simple nearest-neighbour or graph-based heuristics".

## 14. Metric learning reality check

- **Key:** `musgrave2020reality`
- **Authors:** Musgrave, Kevin; Belongie, Serge; Lim, Ser-Nam
- **Title:** A Metric Learning Reality Check
- **Venue:** *Computer Vision – ECCV 2020*, Lecture Notes in Computer Science, Springer, pp. 681–699 (2020). Editors Vedaldi, Bischof, Brox, Frahm. The LNCS volume number is **UNVERIFIED**: Crossref omits it (ISBN 978-3-030-58595-2).
- **DOI:** 10.1007/978-3-030-58595-2_41. arXiv:2003.08505.
- **Claim:** under fair comparison, the reported deep-metric-learning gains largely vanished.
- **Verdict:** SUPPORTED.
- **Quote (abstract):** "We find flaws in the experimental methodology of numerous metric learning papers, and show that the actual improvements over time have been marginal at best."

## 15. Hubness

- **Key:** `radovanovic2010hubs`
- **Authors:** Radovanović, Miloš; Nanopoulos, Alexandros; Ivanović, Mirjana
- **Title:** Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data
- **Venue:** *Journal of Machine Learning Research* **11**(86), 2487–2531 (2010). https://jmlr.org/papers/v11/radovanovic10a.html
- **Verdict:** SUPPORTED.
- **Quotes:**
  - "hubness, that affects the distribution of k-occurrences: the number of times a point appears among the k nearest neighbors of other points in a data set. … this distribution becomes considerably skewed as dimensionality increases, causing the emergence of hubs".
  - "To characterize the asymmetry of Nk we use the standardized third moment of the distribution of k-occurrences, S_Nk = E(Nk − μNk)^3 / σ^3_Nk".

## 16. CSLS

- **Key:** `conneau2018muse`
- **Authors:** Conneau, Alexis; Lample, Guillaume; Ranzato, Marc'Aurelio; Denoyer, Ludovic; Jégou, Hervé
- **Title:** Word Translation Without Parallel Data
- **Venue:** *International Conference on Learning Representations (ICLR)* 2018. The PDF says "Published as a conference paper at ICLR 2018". arXiv:1710.04087v3 (30 Jan 2018). The OpenReview ID is UNVERIFIED.
- **Verdict:** SUPPORTED.
- **Quotes:**
  - "We introduce a cross-domain similarity adaptation to mitigate the so-called hubness problem".
  - r_T(Wx_s) = (1/K) Σ_{y_t ∈ N_T(Wx_s)} cos(Wx_s, y_t)
  - "CSLS(W xs, yt) = 2 cos(W xs, yt) − rT(W xs) − rS(yt)."
- **Precision note:** CSLS subtracts the mean similarity to the K nearest neighbours on *both* sides: the query's neighbourhood among candidates, and each candidate's neighbourhood among queries.

## 17. Mutual proximity

- **Key:** `schnitzer2012mp`
- **Authors:** Schnitzer, Dominik; Flexer, Arthur; Schedl, Markus; Widmer, Gerhard
- **Title:** Local and Global Scaling Reduce Hubs in Space
- **Venue:** *Journal of Machine Learning Research* **13**(92), 2871–2902 (2012). https://jmlr.org/papers/v13/schnitzer12a.html
- **Verdict:** SUPPORTED.
- **Quotes:**
  - "we propose a global variant … to define a concept we call Mutual Proximity"
  - "MP(dx,y) = P(X > dx,y ∩ Y > dy,x)"
  - "Both methods lead to a strong decrease of hubness in these data sets, while at the same time improving properties like classification accuracy."

## 18. opXRD

**Paper**
- **Key:** `hollarek2025opxrd`
- **Authors (24):** Hollarek, Daniel; Schopmans, Henrik; Östreicher, Jona; Teufel, Jonas; Cao, Bin; Alwen, Adie; Schweidler, Simon; Singh, Mriganka; Kodalle, Tim; Hu, Hanlin; et al. (last author Friederich, Pascal)
- **Title:** opXRD: Open Experimental Powder X-Ray Diffraction Database
- **Venue:** *Advanced Intelligent Discovery* **2**(2), e202500044 (online 20 Jun 2025; issue dated Apr 2026). CC BY 4.0.
- **DOI:** 10.1002/aidi.202500044. arXiv:2503.05577v2.

**Data**
- **Key:** `opxrd2025zenodo`
- **Record:** Zenodo 10.5281/zenodo.15298026, published 2025-04-28, `opxrd.zip` 1.40 GB, creators Hollarek & Friederich, **licence CC BY 4.0**.
- **Version:** the concept record 10.5281/zenodo.14254270 resolves to 15298026 (API `relations.version` index 10, `is_last: true`). The RealBind datasheet calls this "version 11".

**Assessment**
- **Claim:** an open database of experimental powder patterns, including a CNRS contribution of Rietveld-refined patterns with structures; licence CC BY 4.0.
- **Verdict:** PARTLY.
- **What the paper says about CNRS:** "Experimental pXRD data was extracted from the Crystallography Open Database (COD)". There are 1,052 patterns, with "✔(85%)" for space group and atomic coordinates. The paper does not say the patterns are Rietveld-refined: they are raw profiles found in COD CIFs that carry powder (CIF_POW) tags.
- **Licence:**
  - The deposit is CC BY 4.0, with no separate licence for the CNRS subset.
  - The paper adds: "The data in the COD are placed in the public domain and licensed under the CC0 License".
  - The paper also says: "the opXRD database is both free and imposes no restrictions on how its data is used."
- **Chemistry quote:** "They span a wide chemical space, including organic, inorganic, and hybrid structures, and 75 different elements".
- **Fix:** "1,052 raw experimental patterns from COD entries with deposited powder data (about 85% with atomic coordinates); distributed under CC BY 4.0 (source COD data are CC0)". Drop "Rietveld-refined" unless you verify it per entry.

## 19. COMPACK

- **Key:** `chisholm2005compack`
- **Authors:** Chisholm, James Alexander; Motherwell, Sam
- **Title:** COMPACK: a program for identifying crystal structure similarity using distances
- **Venue:** *Journal of Applied Crystallography* **38**(1), 228–231 (2005).
- **DOI:** 10.1107/S0021889804027074
- **Claim:** crystal packing similarity by comparing molecular clusters (e.g. 15 molecules); the basis of CCDC packing similarity.
- **Verdict:** PARTLY.
- **Quote (abstract):** "A method is presented for comparing crystal structures to identify similarity in molecular packing environments. The relative position and orientation of molecules is captured using interatomic distances, which provide a representation of structure that avoids the use of space-group and cell information."
- **The "15 molecules":** this is not in the abstract (the full text is paywalled). The CSD Python API docs give `packing_shell(packing_shell_size=15)` (https://downloads.ccdc.cam.ac.uk/documentation/API/modules/crystal_api.html).
- **The "basis of CCDC packing similarity":** UNVERIFIED. The API page does not cite Chisholm & Motherwell.
- **Fix:** "COMPACK compares clusters of molecules via interatomic distances (we used the CCDC implementation's default 15-molecule shell)".

## 20. pymatgen

- **Key:** `ong2013pymatgen`
- **Authors (10):** Ong, Shyue Ping; Richards, William Davidson; Jain, Anubhav; Hautier, Geoffroy; Kocher, Michael; Cholia, Shreyas; Gunter, Dan; Chevrier, Vincent L.; Persson, Kristin A.; Ceder, Gerbrand
- **Title:** Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis
- **Venue:** *Computational Materials Science* **68**, 314–319 (2013).
- **DOI:** 10.1016/j.commatsci.2012.10.028
- **Claim:** a Python library including an XRD pattern calculator.
- **Verdict:** PARTLY.
- **Library citation:** SUPPORTED. pymatgen.org says: "If you use pymatgen in your research, please consider citing the following work: … Computational Materials Science, 2013, 68, 314–319".
- **Whether the 2013 paper describes the XRD calculator:** CANNOT CHECK. No open copy was reachable (OSTI 403; eScholarship and MIT DSpace are script-rendered). The pymatgen.org feature list does not name XRD.
- **The module exists:** `pymatgen/analysis/diffraction/xrd.py` begins "This module implements an XRD pattern calculator."
- **Fix:** "simulated with pymatgen's XRDCalculator (pymatgen vX.Y; Ong et al. 2013)".

## 21. CLIP and InfoNCE

**21a. CLIP**
- **Key:** `radford2021clip`
- **Authors (12):** Radford, Alec; Kim, Jong Wook; Hallacy, Chris; Ramesh, Aditya; Goh, Gabriel; Agarwal, Sandhini; Sastry, Girish; Askell, Amanda; Mishkin, Pamela; Clark, Jack; et al. (Krueger, Gretchen; Sutskever, Ilya)
- **Title:** Learning Transferable Visual Models From Natural Language Supervision
- **Venue:** *Proceedings of the 38th International Conference on Machine Learning*, PMLR **139**, 8748–8763 (2021). https://proceedings.mlr.press/v139/radford21a.html. arXiv:2103.00020.
- **Verdict:** SUPPORTED.
- **Quotes:** "CLIP jointly trains an image encoder and a text encoder to predict the correct pairings of a batch of (image, text) training examples"; "which we call CLIP, for Contrastive Language-Image Pre-training".

**21b. CPC / InfoNCE**
- **Key:** `oord2018cpc`
- **Authors:** van den Oord, Aaron; Li, Yazhe; Vinyals, Oriol
- **Title:** Representation Learning with Contrastive Predictive Coding
- **Venue:** arXiv:1807.03748 (v1 10 Jul 2018; v2 22 Jan 2019)
- **Verdict:** SUPPORTED, for the InfoNCE loss only.
- **Quote:** "Both the encoder and autoregressive model are trained to jointly optimize a loss based on NCE, which we will call InfoNCE."
- **Note:** CPC is not a cross-modal dual encoder. Cite it for the loss, and CLIP for the dual-encoder setup.

## 22. CSP plus PXRD matching for pharmaceuticals

- **Key:** `glick2025csp`
- **Authors:** Glick, Zachary L.; Metcalf, Derek P.; Swarthout, Scott F.
- **Title:** Toward Routine CSP of Pharmaceuticals: A Fully Automated Protocol Using Neural Network Potentials
- **Venue:** arXiv:2507.16218v1 (22 Jul 2025). No journal version was found by a Crossref bibliographic query on 2026-10-12.
- **Claim:** CSP combined with PXRD matching identifies forms absent from databases.
- **Verdict:** SUPPORTED.
- **Quotes:**
  - "Three current drug molecules for which no experimental crystal structures have been published, but PXRD patterns are available."
  - "we simulate the powder pattern at the experimentally reported wavelength and refine the unit cell parameters and peak shape parameters to maximize a cross correlation-based similarity score"
  - "a semi-blinded challenge that successfully identifies and ranks polymorphs of three modern drugs from powder X-ray diffraction patterns alone"
- **Caveats:** Z′ = 1 only; about 8.4k CPU-h per CSP; semi-blinded.

## 23. Synthetic-crystal training (Schopmans et al.)

- **Key:** `schopmans2023synthetic`
- **Authors:** Schopmans, Henrik; Reiser, Patrick; Friederich, Pascal
- **Title:** Neural networks trained on synthetically generated crystals can extract structural information from ICSD powder X-ray diffractograms
- **Venue:** *Digital Discovery* **2**(5), 1414–1424 (2023). CC BY 3.0. arXiv:2303.11699v3.
- **DOI:** 10.1039/D3DD00071K
- **Verdict:** SUPPORTED, for this one-line claim: "Space-group classifiers trained only on synthetic random-coordinate crystals reached 79.9% on unseen ICSD structure types (vs 56.1% for training on ICSD), but on 942 experimental RRUFF patterns reached only 25.2% top-1 (>60% top-10), and only when impurity phases were included in training."
- **Quotes:** "we achieved a test accuracy of 79.9% on unseen ICSD structure types … This surpasses the 56.1% accuracy"; "it is essential to include impurity phases in the training data. By doing so, we obtain a top-1 accuracy of 25.2% and a top-10 accuracy of over 60%." The SI states "This left 942 diffractograms for our analysis."

## 24. Thermal expansion of organic crystals

- **Key:** `bond2021thermal`
- **Author:** Bond, Andrew D.
- **Title:** A survey of thermal expansion coefficients for organic molecular crystals in the Cambridge Structural Database
- **Venue:** *Acta Crystallographica Section B* **77**(3), 357–364 (2021; online 13 May 2021). CC BY 4.0. PMC8182801.
- **DOI:** 10.1107/S2052520621003309
- **Claim:** molecular crystals typically expand by about 1% or more in cell volume between about 100 K and room temperature.
- **Verdict:** PARTLY. Bond states coefficients, not percentages, and our arithmetic gives about 3% in volume, so "~1%" understates it.
- **Quotes (abstract):**
  - "For a selected subset of 210 structures showing an optimal linear fit, the volumetric expansion coefficient at 298 K has mean 173 p.p.m. K−1 and standard deviation 47 p.p.m. K−1."
  - "The full set of 6201 structures … mean 161 p.p.m. K−1 and standard deviation 51 p.p.m. K−1".
  - "For structure families comprising four or more temperature points in the range 90–300 K, a linear relationship between unit-cell volume and temperature is shown to be a reasonable approximation."
  - Principal (linear) coefficients: "two half normal distributions centred on 33 p.p.m. K−1".
- **Our arithmetic (not Bond's):** ΔV/V ≈ (161 to 173) × 10⁻⁶ K⁻¹ × 198 K ≈ 3.2 to 3.4% from 100 K to 298 K. That is about 1.1% isotropic-equivalent linear strain.
- **Fix:** "volumetric thermal expansion coefficients average 161 p.p.m. K⁻¹ (SD 51; Bond 2021), i.e. roughly 3% in cell volume, about 1% per axis, between 100 K and room temperature".

## 25. Rietveld refinement

- **Key:** `rietveld1969`
- **Author:** Rietveld, H. M.
- **Title:** A profile refinement method for nuclear and magnetic structures
- **Venue:** *Journal of Applied Crystallography* **2**(2), 65–71 (1969).
- **DOI:** 10.1107/S0021889869006558
- **Verdict:** SUPPORTED.
- **Quote (abstract, via OpenAlex):** "A structure refinement method is described which does not use integrated neutron powder intensities, single or overlapping, but employs directly the profile intensities obtained from step-scanning measurements of the powder diagram."

## 26. PyTorch

- **Key:** `paszke2019pytorch`
- **Authors (21):** Paszke, Adam; Gross, Sam; Massa, Francisco; Lerer, Adam; Bradbury, James; Chanan, Gregory; Killeen, Trevor; Lin, Zeming; Gimelshein, Natalia; Antiga, Luca; et al. (last author Chintala, Soumith)
- **Title:** PyTorch: An Imperative Style, High-Performance Deep Learning Library
- **Venue:** *Advances in Neural Information Processing Systems 32* (NeurIPS 2019), Curran Associates. https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html. arXiv:1912.01703.
- **Pages:** the official NeurIPS BibTeX has `pages = {}`. The commonly cited "8024–8035" is UNVERIFIED; omit pages.
- **Verdict:** SUPPORTED (software citation).

## 27. Cluster (compound-level) bootstrap

**27a. Preferred citation**
- **Key:** `field2007cluster`
- **Authors:** Field, C. A.; Welsh, A. H.
- **Title:** Bootstrapping Clustered Data
- **Venue:** *Journal of the Royal Statistical Society Series B* **69**(3), 369–390 (2007).
- **DOI:** 10.1111/j.1467-9868.2007.00593.x
- **Verdict:** SUPPORTED.
- **Quote (abstract):** "the cluster bootstrap gives consistent estimates under both the transformation and the random-effect model". This is the cleaner citation for resampling compounds with replacement.
- **Caveat:** the theory covers one-way arrays (sample means, between/within sums of squares). Applying it to hit@k is your extension; say so.

**27b. Alternative**
- **Key:** `davison1997bootstrap`
- **Authors:** Davison, A. C.; Hinkley, D. V.
- **Title:** *Bootstrap Methods and their Application*
- **Publisher:** Cambridge University Press (1997)
- **DOI:** 10.1017/CBO9780511802843 (record verified)
- **Status:** the book was not read. The section on hierarchical data (often cited as Section 3.8) is **UNVERIFIED**.

## 28. Peak-shape and texture models

**28a. March–Dollase**
- **Key:** `dollase1986`
- **Author:** Dollase, W. A.
- **Title:** Correction of intensities for preferred orientation in powder diffractometry: application of the March model
- **Venue:** *Journal of Applied Crystallography* **19**(4), 267–272 (1986).
- **DOI:** 10.1107/S0021889886089458
- **Verdict:** SUPPORTED.
- **Quotes (abstract, via OpenAlex):** "Diffracted intensities from axially symmetric flat-plate or capillary specimens, composed of effectively rod- or disk-shaped crystallites, can be corrected for preferred orientation with a single pole-density profile." "the March distribution is proposed as an advantageous pole-density profile function for general use."

**28b. Caglioti**
- **Key:** `caglioti1958`
- **Authors:** Caglioti, G.; Paoletti, A.; Ricci, F. P.
- **Title:** Choice of collimators for a crystal spectrometer for neutron diffraction
- **Venue:** *Nuclear Instruments* **3**(4), 223–228 (1958).
- **DOI:** 10.1016/0369-643X(58)90029-X
- **Verdict:** CANNOT CHECK. There is no abstract or open text. Attributing H² = U tan²θ + V tanθ + W to this paper is standard, but it was seen only in secondary sources.
- **Caveat:** the paper derives the form for neutron collimator resolution. For lab X-ray patterns it is an empirical convention; Cheary & Cline (1994) criticised it (secondary source, not checked).
- **Suggested wording:** "a Caglioti-type angular dependence of the Gaussian width".

---

## 29. Recency sweep (2026-08-13 to 2026-10-12)

**Target:** any new paper on learned retrieval or identification of organic/molecular crystals from PXRD evaluated on real measured patterns, or any new open real organic PXRD benchmark.

**Result:** none found that is organic-specific. One partial hit, RealBind, must be cited and changes the novelty wording (see item 9).

### Hits in the window, with relevance

- **RealBind** (ChemRxiv 10.26434/chemrxiv.15009668/v1, 29 Sep; /v2, 7 Oct 2026). **Partial hit.** A learned contrastive model is compared with correlation on measured PXRD. The measured test set includes opXRD-CNRS molecular crystals and MOFs: about 23 of 221 test queries are organic by our elemental count. The learned model was trained on Materials Project (inorganic) simulations. Not organic-specific.
- **Gan Jiang agent** (arXiv:2610.07862, 6 Oct 2026). **Near hit.** Real-pattern identification on RRUFF and opXRD (opXRD top-1 40.83% without composition). The organic content of its opXRD subset is not stated in the text checked. Not organic-specific.
- **PXtal** (arXiv:2610.10653, 7 Oct 2026). Simulated only; inorganic plus QMOF. Not a hit.
- **GALAXI** (arXiv:2609.06908, 7 Sep 2026). Inorganic COD; curated experimental patterns. Not a hit.
- **X2SBench** (arXiv:2609.29751, 24 Sep 2026; **withdrawn** 29 Sep 2026). 591 measured patterns; organic content not stated. Not a hit.
- **PhaseMatcher** (arXiv:2610.05844, 5 Oct 2026). Multiphase identification on synthetic mixtures and mixtures built from measured single-phase patterns; no organic focus. Not a hit.
- **Knowledge-Driven XRD Phase Identification** (arXiv:2609.31888, 25 Sep 2026). Autoencoder retrieval on synthetic data only. Not a hit.
- **RINSE** (Fellowes, Evans, Cruz-Cabeza; ChemRxiv 10.26434/chemrxiv.15008615/v1, 10 Sep 2026). A non-learned reciprocal-space fingerprint "computed from a crystal's diffraction pattern" for structure comparison and approximate retrieval; no measured-pattern evaluation in the abstract. **Related work worth citing** (organic-relevant authors); not a hit.
- **"An unlearnable target: diagnosing the orientation gauge in PXRD-conditioned molecular-crystal flow matching"** (Cai; ChemRxiv 10.26434/chemrxiv.15008251/v1, 3 Sep 2026). Frank's own preprint; held-out structures, generative, no real patterns per the abstract. Not a hit (self-citation candidate).
- **XRD Pro, COD-derived XRPD Database** (Zenodo 10.5281/zenodo.23002197, 27 Sep 2026). *Simulated* (pymatgen) patterns for 531,286 COD entries. Not a real-pattern benchmark.
- **Hybrid quantum NN** (2609.22376), **AutoXRD** (2609.00070), **ERAF4XRD** (2609.18583), **CG-OMatG molecular CSP** (2609.39773; no PXRD), **PowderLine** (2608.17009). Not relevant.

### Queries run (all on 2026-10-12)

**arXiv export API** (`search_query`, sorted by submittedDate, max 100). Every query was ANDed with `submittedDate:[202608130000 TO 202610122359]`.

| Query | Hits |
|---|---|
| `abs:PXRD` | 6 |
| `abs:XRPD` | 0 |
| `abs:"powder diffraction"` | 12 |
| `abs:"powder X-ray"` | 16 |
| `abs:diffraction AND (abs:organic OR abs:molecular OR abs:pharmaceutical OR abs:polymorph OR abs:polymorphs)` | 37 |
| `abs:"Cambridge Structural Database"` | 2 |
| `abs:XRD AND (abs:retrieval OR abs:contrastive OR abs:embedding OR abs:identification)` | 3 |

All titles were screened. Abstracts were read for 2610.05844, 2609.29751, 2609.22376, 2609.00070, 2609.18583, 2609.31888, 2609.39773 and 2608.17009.

**OpenAlex** `works?search=…&filter=from_publication_date:2026-08-13,to_publication_date:2026-10-12` (top 25 screened each; this covers ChemRxiv):

| Query | Count |
|---|---|
| "powder X-ray diffraction machine learning organic molecular crystals" | 326 |
| "PXRD deep learning polymorph identification" | 6 |
| "powder diffraction retrieval contrastive crystal structure" | 100 |
| "experimental powder diffraction dataset benchmark organic" | 205 |
| "powder diffraction pharmaceutical machine learning experimental patterns" | 136 |
| "XRPD solid form classification neural network" | 3 |

**Zenodo API** (`q=(…) AND publication_date:[2026-08-13 TO 2026-10-12]`):

| Query | Hits |
|---|---|
| `powder diffraction AND (organic OR pharmaceutical OR polymorph)` | 41 |
| `PXRD AND (dataset OR benchmark)` | 2 |
| `XRPD` | 6 |

**WebSearch** (extended, no date filter available):
1. "2026 machine learning identification of organic molecular crystals or pharmaceutical polymorphs from experimental powder X-ray diffraction patterns retrieval benchmark"
2. "open dataset experimental PXRD patterns organic compounds polymorphs released September October 2026 Zenodo OR figshare OR "Scientific Data""

**Not searched** (residual risk): ChemRxiv's own search (only through OpenAlex), Figshare API, Semantic Scholar, journal tables of contents (*Digital Discovery*, *J. Appl. Cryst.*, *CrystEngComm*, *Cryst. Growth Des.*, *Mol. Pharmaceutics*), and paywalled full texts.

---

## refs.bib corrections (paper/refs.bib, URTC bib)

- `xu2025xccp`: published as *npj Comput. Mater.* **12**, 144 (2026), DOI 10.1038/s41524-026-02015-y. The journal title is "KAN-enhanced contrastive learning: the accelerator of crystal structure identification from XRD patterns". Rename to `xu2026xccp`.
- `lai2025xtalnet`: add volume 12, issue 8, article 2410722, DOI 10.1002/advs.202410722. The journal author list has 12 names (it adds Linfeng Zhang).
- `johansen2025decifer`: now TMLR 2026 (`johansen2026decifer`). Full authors: Johansen, Friis-Jensen, Dam, Jensen, Mercado, Selvan. Add `johansen2026realworld` (*Digital Discovery* 5, 3871–3882, DOI 10.1039/d6dd00088f).
- `radford2021clip`: add PMLR volume 139, pages 8748–8763.
- `groom2016csd`, `ong2013pymatgen`, `dollase1986`, `caglioti1958`, `cruzcabeza2015polymorph`, `bauer2001ritonavir`, `gatesrector2019pdf`: all fields match Crossref.

## Corrections to the earlier sweep files

- **refute_1 #13a and refute_2 #5:** RealBind is **not** inorganic-only. Its measured set includes opXRD-CNRS MOFs and molecular crystals (release DATASHEET.md).
- **refute_2 #6:** the Zenodo `relations.version` index for 15298026 is 10, with `is_last: true`. The RealBind datasheet calls this "version 11", consistent with lit_C's "v11". Both notes describe the same record.
- **lit_B row 9 and the item-5 brief:** the J. Appl. Cryst. 2024 method is VC-GPWDF; VC-PWDF and VC-xPWDF are the Mayo et al. papers.

## Kaduk external set (added 2026-10-18, RSC Advances retarget)

Each entry checked against its Crossref record (api.crossref.org/works/<DOI>): authors, title, volume, issue, pages, print year.

| key | anchor refcode | DOI | record | checked |
|---|---|---|---|---|
| kaduk2015tramadol | BUFPAV01 | 10.1017/s088571561500041x | Powder Diffraction 2015, 30(3), 242-249; 4 authors | Crossref API 2026-10-18 |
| kaduk2015folic | FOLCAH03 | 10.1017/s0885715614000815 | Powder Diffraction 2015, 30(1), 52-56; 3 authors | Crossref API 2026-10-18 |
| kaduk2015fingolimod | HUCHEU02 | 10.1017/s0885715615000317 | Powder Diffraction 2015, 30(3), 205-210; 4 authors | Crossref API 2026-10-18 |
| kaduk2015levalbuterol | OMAJIX | 10.1017/s0885715615000172 | Powder Diffraction 2015, 30(2), 158-162; 4 authors | Crossref API 2026-10-18 |
| kaduk2015ziprasidone | OMAKAQ01 | 10.1017/s0885715615000305 | Powder Diffraction 2015, 30(3), 192-198; 4 authors | Crossref API 2026-10-18 |
| kaduk2015rilpivirine | OMAKOE01 | 10.1017/s0885715615000196 | Powder Diffraction 2015, 30(2), 170-174; 3 authors | Crossref API 2026-10-18 |
| kaduk2015solifenacin | URATAK | 10.1017/s0885715615000329 | Powder Diffraction 2015, 30(3), 211-217; 5 authors | Crossref API 2026-10-18 |
| kaduk2015lacosamide | URATIS | 10.1017/s0885715615000378 | Powder Diffraction 2015, 30(3), 224-230; 4 authors | Crossref API 2026-10-18 |
| kaduk2015nilotinib | URAVUG | 10.1017/s0885715615000512 | Powder Diffraction 2015, 30(3), 270-277; 4 authors | Crossref API 2026-10-18 |
| kaduk2015sitagliptin | URAWUH | 10.1017/s0885715615000755 | Powder Diffraction 2015, 30(4), 349-356; 4 authors | Crossref API 2026-10-18 |
| kaduk2016paliperidone | YAGRIJ01 | 10.1017/s0885715616000087 | Powder Diffraction 2016, 31(2), 135-141; 4 authors | Crossref API 2026-10-18 |
| kaduk2017palmitate | NOKFUR | 10.1017/s0885715617000896 | Powder Diffraction 2017, 32(4), 222-227; 4 authors | Crossref API 2026-10-18 |
| kaduk2017hydrocortisone | ZZZGCU01 | 10.1017/s0885715616000713 | Powder Diffraction 2017, 32(1), 31-34; 3 authors | Crossref API 2026-10-18 |
| wheatley2018estradiol | DOVGEC02 | 10.1017/s0885715618000039 | Powder Diffraction 2018, 33(1), 70-71; 4 authors | Crossref API 2026-10-18 |
| reid2015curcumin | BINMEQ09 | 10.1017/s0885715614001018 | Powder Diffraction 2015, 30(1), 67-75; 4 authors | Crossref API 2026-10-18 |
| kaduk2015saxagliptin | HAQQUN02 | 10.1017/s0885715615000548 | Powder Diffraction 2015, 30(3), 296-297; 5 authors | Crossref API 2026-10-18 |
| reid2016trandolapril | IQISAE | 10.1017/s0885715616000294 | Powder Diffraction 2016, 31(3), 205-210; 3 authors | Crossref API 2026-10-18 |
| reid2021donepezil | EDUPUS | 10.1017/s0885715621000415 | Powder Diffraction 2021, 36(4), 233-240; 2 authors | Crossref API 2026-10-18 |
