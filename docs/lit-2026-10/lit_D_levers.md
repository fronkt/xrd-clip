# Track D: methods and analogues to borrow for xrd-clip continuation (2018 to Oct 2026)

Compiled 2026-10-10. Sources: arXiv export API, Europe PMC REST, OpenAlex API, institutional pages, WebSearch result listings. No browser, no proxy, no keyed API.
Verification tag: **[A]** = abstract or API record fetched and read; **[M]** = existence confirmed by API record (title/DOI/venue) but numbers taken from a search-result snippet or institutional page, not from the abstract itself. Anything weaker is in Section 4 (UNVERIFIED).
"INFERENCE" marks my own reasoning, not something a paper states.

---

## 1. Analogue papers

### 1a. Spectrum to molecule retrieval in other spectroscopies

| # | Citation | ID | Modality | Learned vs library / key result (as stated) | Tag |
|---|---|---|---|---|---|
| 1 | Li, Kind, ... Fiehn. "Spectral entropy outperforms MS/MS dot product similarity for small-molecule compound identification." Nat. Methods 2021 | doi:10.1038/s41592-021-01331-z | MS/MS | **Classical, not learned.** Entropy similarity "outperformed 42 alternative similarity algorithms, including dot product" searching 434,287 spectra against NIST20. FDR < 10% at entropy score 0.75 on 37,299 natural-product spectra. Robust to added noise ions. | [A] |
| 2 | Huber et al. "Spec2Vec: improved mass spectral similarity scoring through learning of structural relationships." PLoS Comput Biol 2021 | doi:10.1371/journal.pcbi.1008724 | MS/MS | Unsupervised Word2Vec-style peak embeddings. Scores "correlate better with structural similarity than cosine-based scores". The abstract claims advantages in library matching and analogue search. | [A] |
| 3 | Huber et al. "MS2DeepScore: a novel deep learning similarity measure to compare tandem mass spectra." J Cheminform 2021 | doi:10.1186/s13321-021-00558-4 | MS/MS | **Siamese network** predicting Tanimoto from a spectrum pair. RMSE about 0.15, or about 0.1 on an uncertainty-filtered subset (MC-dropout). "outperforms classical spectral similarity measures in retrieving chemically related compound pairs". Note: the target is *analogue* similarity, not exact identity. | [A] |
| 4 | de Jonge, Louwen, ... van der Hooft. "MS2Query: reliable and scalable MS2 mass spectra-based analogue search." Nat Commun 2023 | doi:10.1038/s41467-023-37446-4 | MS/MS | **Hybrid retrieve-then-rerank.** A learned embedding shortlist (Spec2Vec + MS2DeepScore) is reranked by a random forest over 5 features, including precursor mass (snippet). Abstract: "improved reliability and scalability" over modified-cosine analogue search. Snippet: reliable analogues for 35% of spectra, mean Tanimoto 0.67. | [A] abstract; RF and 35% numbers [M] |
| 5 | Goldman et al. MIST, "Annotating metabolite mass spectra with domain-inspired chemical formula transformers." Nat Mach Intell 2023 | bioRxiv doi:10.1101/2022.12.30.522318 | MS/MS | Peak-formula transformer that predicts a fingerprint, then retrieves. Compared against CSI:FingerID (snippet: "retrieves 66% of metabolites with equal or improved accuracy, 29% strictly better"). | [M] |
| 6 | Bushuiev et al. "MassSpecGym: a benchmark for the discovery and identification of molecules." NeurIPS D&B 2024 | arXiv:2410.23326 | MS/MS | Retrieval hit@1 (mass-based candidates): random 0.37%, DeepSets 1.47%, FP-FFN 2.54%, DeepSets+Fourier 5.24%, **MIST 14.64%** (hit@20 59.15%). Formula-based candidates (harder isomer sets): MIST 9.57% hit@1. | [A] (HTML tables) |
| 7 | Kalia, Chen, Krishnan, Hassoun. "JESTR: joint embedding space technique for ranking candidate molecules..." 2024 | arXiv:2411.14464 | MS/MS ↔ molecule (CLIP-style) | Contrastive spectrum and molecule embeddings, ranked by cosine. Rank@1: NPLIB1 45.76% vs MIST 27.96%; NIST2020 38.62% vs 20.95%; **MoNA 26.56% vs MIST 32.30% (loses)**; MassSpecGym-formula 11.85% vs MIST 9.57%, SIRIUS 10.41%. **Hard-candidate regularization** (the 32 most similar same-formula PubChem candidates per batch, final 3% of epochs, β=0.1). Removing it costs 6.0% to 37.1% relative rank@1. | [A] |
| 8 | Wang, Manjrekar, ... Goldman, ... Coley. "Neural spectral prediction for structure elucidation with tandem mass spectrometry" (ICEBERG). bioRxiv 2025 | doi:10.1101/2025.05.28.656653 (method: arXiv:2304.13136) | MS/MS, **forward simulation then spectrum-to-spectrum ranking** | Ranks candidates by the similarity between the experimental spectrum and *predicted* spectra. **46% top-1 / 86% top-10 on MassSpecGym**; 40% top-1 / 92% top-10 on NIST'20 [M+H]+. Outperforms SIRIUS on a new uncharacterized test set. INFERENCE: this is far above MIST 14.6% and JESTR 11.9% hit@1. Whether the candidate-set protocol matches exactly (mass vs formula sets) is not stated in the abstract, so treat the comparison as indicative. | [A] |
| 9 | Zhang et al. "Breaking the modality barrier: generative modeling for accurate molecule retrieval from mass spectra" (GLMR). AAAI 2026 | arXiv:2511.06259 | MS/MS | **Two-stage**: a contrastive pre-retrieval shortlist, then a generative model conditioned on the spectrum plus candidates, whose output reranks the shortlist. ">40% improvement in top-1 accuracy" on MassSpecGym and MassRET-20k. | [A] |
| 10 | De Waele et al. "Small molecule retrieval from tandem mass spectrometry: what are we optimizing for?" 2026 | arXiv:2602.16507 | MS/MS (theory) | "fundamental trade-off between (1) fingerprint similarity and (2) molecular retrieval" objectives. Which one wins depends on how similar the candidates are. | [A] |
| 11 | Jürgens et al. "When should we trust the annotation? Selective prediction for molecular structure retrieval from mass spectra." 2026 | arXiv:2603.10950 | MS/MS | "retrieval-level total uncertainty provides the overall strongest rejection criterion" for abstaining. | [A] |
| 12 | Bushuiev et al. DreaMS. Nat Biotechnol (Europe PMC lists 2026; online 2025) | doi:10.1038/s41587-025-02663-3 | MS/MS foundation model | Transformer over **peaks as tokens**, self-supervised on millions of unlabeled spectra (masked-peak and retention-order prediction). After fine-tuning, SOTA on spectral similarity and other tasks. | [A] |
| 13 | CLERMS. "Contrastive learning-based embedder for the representation of tandem mass spectra." Anal Chem 2023 | doi:10.1021/acs.analchem.3c00260 | MS/MS | Existence confirmed. Abstract numbers not read. | [M] (existence only) |
| 14 | Dührkop et al. "SIRIUS 4..." Nat Methods 2019 (contains CSI:FingerID) | doi:10.1038/s41592-019-0344-8 | MS/MS | The fingerprint-prediction-then-database-retrieval baseline. Existence confirmed only. | [M] (existence only) |
| 15 | Yang et al. "Cross-modal retrieval between 13C NMR spectra and structures for compound identification using deep contrastive learning" (CReSS). Anal Chem 2021 | doi:10.1021/acs.analchem.1c04307 | 13C NMR ↔ structure (CLIP-style) | 41,494 spectra searched against a 10.4M-structure library: recall@10 91.64%, 0.114 s/query. **Adding a 5 Da molecular-weight filter raised recall@10 to 98.39%.** | [A] |
| 16 | Sun et al. "Cross-modal retrieval between 13C NMR spectra and structures based on focused libraries." Anal Chem 2024 | doi:10.1021/acs.analchem.3c04294 | 13C NMR | States "a redundant library reduces the structure-elucidation accuracy". A generated focused library gave top-10 54.03% vs 9.17% with a random reference library (6,471 spectra). | [A] |
| 17 | Rocabert-Oriols et al. "Multi-modal contrastive learning for chemical structure elucidation with VibraCLIP." Digital Discovery 2025, 4, 3818 | doi:10.1039/d5dd00269a | IR + Raman ↔ graph (tri-modal CLIP) | Top-1 81.7%; **top-25 98.9% with molecular mass integrated**. Includes a lightweight fine-tuning protocol from theoretical to experimental spectra. | [M] (ICIQ institutional page; publisher page returned 403) |
| 18 | Kanakala et al. "Spectra to structure: contrastive learning framework for library ranking and generating molecular structures for infrared spectra" (SMEN). Digital Discovery 2024 | doi:10.1039/d4dd00135d | IR ↔ molecule | Contrastive library ranking. Snippet only: "Top 1 accuracy of ~81%". | [M] |
| 19 | Li, Schmidt, Alstrøm. "Raman spectrum matching with contrastive representation learning." 2022 | arXiv:2202.12549 | Raman, spectrum ↔ spectrum library | Siamese/contrastive matching with no preprocessing, one reference per class. "significantly improves or is on par with the state of the art". Adds conformal prediction sets. | [A] |
| 20 | Alberts et al. "Unraveling molecular structure: a multimodal spectroscopic dataset for chemistry." 2024 | arXiv:2407.17492 | IR/NMR/MS dataset | Simulated multimodal benchmark. Usable for checking ideas, not a retrieval result. | [A] (title) |
| 21 | Gao et al. "Robust small-molecule identification from incomplete, degraded, and inconsistent spectra using multimodal mixed-condition training." 2026 | arXiv:2609.14360 | MS + IR + NMR | Mixed-condition (degraded-input) training: +7.67% top-1. | [A] |

### 1b. XRD and crystal analogues (polymorph-relevant)

| # | Citation | ID | Topic | Key result (as stated) | Tag |
|---|---|---|---|---|---|
| 22 | Mayo, Marczenko, Johnson. "Quantitative matching of crystal structures to experimental powder diffractograms" (VC-xPWDF). Chem Sci 2023 | doi:10.1039/d3sc00168g | Polymorph ID from PXRD, classical | Variable-cell powder difference matches experimental PXRD to CSD and CSP (CPOSS) structures. It correctly identified the most similar structure for 7 organic compounds at moderate and "low" quality. Has trouble with preferred orientation. Beats FIDEL *when the pattern can be indexed*. | [A] |
| 23 | Mayo et al. "Development and assessment of an improved powder-diffraction-based method for molecular crystal structure similarity" (VC-PWDF). CrystEngComm 2022 | doi:10.1039/d2ce01080a | Simulated-PXRD crystal similarity | Snippet: total disagreement with COMPACK 2.84% vs 12.35% for other PXRD-based tools. | [M] |
| 24 | Iwasaki, Kusne, Takeuchi. "Comparison of dissimilarity measures for cluster analysis of X-ray diffraction data from combinatorial libraries." npj Comput Mater 2017 (just before the window; kept because it is directly on point) | doi:10.1038/s41524-017-0006-2 | XRD similarity under peak shift | 9 measures compared. Cosine, Pearson and JS divergence are best when the shift magnitude is unknown. **Constrained, normalized DTW is best when the maximum shift is known.** | [M] (OpenAlex record + snippet) |
| 25 | Utimula et al. "Machine learning clustering technique applied to powder X-ray diffraction patterns to distinguish alloy substitutions." 2018 | arXiv:1810.03972 | DTW on PXRD | DTW filters out intensity variation and uniform shifts. About 90% success. | [A] |
| 26 | Lange et al. "Automatic solid form classification in pharmaceutical drug development" (SMolNet). NeurIPS 2024 workshop | arXiv:2411.03308 | **Siamese XRPD comparison for pharma polymorph screening** | Self-supervised contrastive losses gave a "substantial boost" in separability and precision for phases of *unseen* compounds. | [A] |
| 27 | Lai et al. XtalNet, "End-to-end crystal structure prediction from powder X-ray diffraction." 2024 | arXiv:2401.03862 | PXRD ↔ structure contrastive pretraining + diffusion | Contrastive alignment module plus generation. Top-10 match about 90% (small) and 79% (large) on MOF sets. | [A] |
| 28 | Xu et al. XCCP, "KAN-enhanced contrastive learning..." 2025 | arXiv:2511.04055 | PXRD ↔ crystal graph CLIP | Dual-expert XRD encoder (a low-angle branch and a high-angle branch). Retrieval accuracy 0.89, space-group accuracy 0.93. | [A] |
| 29 | Das et al. "Learning self-supervised representations of powder-diffraction patterns." Crystals 2025 | doi:10.3390/cryst15050393 | Contrastive PXRD encoder | Contrastive pretraining beat supervised models on robustness and generalization (symmetry prediction). | [M] |
| 30 | Cao et al. XMatcher. 2026 | arXiv:2607.17162 | Classical PXRD phase ID | Open framework: theoretical libraries plus **angular-shift correction and peak matching**. | [A] |
| 31 | Martin, Ceriotti, Day. "An adapted similarity kernel and generalized convex hull for molecular crystal structure prediction." Cryst Growth Des 2025 | doi:10.1021/acs.cgd.5c01220 | Molecular-crystal similarity kernel (SOAP-based) | A physically motivated SOAP kernel for molecular crystals. Landscape analysis "sensitive to kernel design choices". | [A] |
| 32 | Kuroshima et al. "Machine learning classification of local environments in molecular crystals." 2024 | arXiv:2404.00155 | Polymorph classification from structure | GCN and symmetry-function models reach high accuracy on urea and nicotinamide polymorphs. | [A] |
| 33 | Gharakhanyan et al. FastCSP. 2025 | arXiv:2508.02641 | CSP with universal MLIP (UMA) | 28 semi-rigid and 10 flexible molecules, 74 experimental polymorphs: all recovered, ranked within 9 kJ/mol of the global minimum. | [A] |

### 1c. Retrieve-and-rerank and fine-grained contrastive methods

| # | Citation | ID | Relevance | Tag |
|---|---|---|---|---|
| 34 | Nogueira & Cho. "Passage re-ranking with BERT" (monoBERT). 2019 | arXiv:1901.04085 | Canonical cross-encoder reranker over a first-stage shortlist | [A] |
| 35 | Khattab & Zaharia. ColBERT. 2020 | arXiv:2004.12832 | Late interaction (MaxSim over per-token vectors) between bi- and cross-encoder | [A] |
| 36 | Humeau et al. Poly-encoders. 2019 | arXiv:1905.01969 | Bi/poly/cross-encoder trade-off | [A] |
| 37 | Lin, Nogueira, Yates. "Pretrained transformers for text ranking: BERT and beyond." 2020 | arXiv:2010.06467 | Survey of the rerank vs dense-retrieval trade-offs | [A] |
| 38 | Reimers & Gurevych. Sentence-BERT. 2019 | arXiv:1908.10084 | Siamese bi-encoder: 65 h → 5 s search cost vs cross-encoding | [A] |
| 39 | Thakur et al. Augmented SBERT. 2020 | arXiv:2010.08240 | A cross-encoder labels pairs to train the bi-encoder: up to +6 points in-domain | [A] |
| 40 | Qu et al. RocketQA. 2020 | arXiv:2010.08191 | Cross-batch negatives, **denoised hard negatives** (a cross-encoder removes false negatives), data augmentation | [A] |
| 41 | Xiong et al. ANCE. 2020 | arXiv:2007.00808 | Hard negatives mined from the model's own ANN index, refreshed during training | [A] |
| 42 | Robinson et al. "Contrastive learning with hard negative samples." 2020 | arXiv:2010.04592 | Hardness-weighted negative sampling for InfoNCE | [A] |
| 43 | Wang & Liu. "Understanding the behaviour of contrastive loss." 2020 | arXiv:2012.09740 | Temperature controls the penalty on hard negatives (a uniformity-tolerance dilemma) | [A] |
| 44 | Khosla et al. Supervised contrastive learning. 2020 | arXiv:2004.11362 | Multi-positive InfoNCE | [A] |
| 45 | Tian et al. StableRep. 2023 | arXiv:2306.00984 | Multi-positive contrastive loss over several views of one item | [A] |
| 46 | Yuksekgonul et al. NegCLIP / ARO. ICLR 2023 | arXiv:2210.01936 | CLIP behaves like a bag-of-words; **composition-aware hard negatives** fix fine-grained failures | [A] |
| 47 | Yao et al. FILIP. 2021 | arXiv:2111.07783 | Token-wise late interaction inside a CLIP dual encoder | [A] |
| 48 | Kusupati et al. Matryoshka representation learning. 2022 | arXiv:2205.13147 | Nested embedding dims: short vector to shortlist, full vector to rerank, one model | [A] |
| 49 | Cuturi & Blondel. Soft-DTW. 2017 | arXiv:1703.01541 | Differentiable alignment loss and layer for 1D signals | [A] |
| 50 | Zhang. "Making convolutional networks shift-invariant again." 2019 | arXiv:1904.11486 | Anti-aliased pooling for true shift invariance and equivariance in CNNs | [A] |
| 51 | Lee et al. Set Transformer. 2018 | arXiv:1810.00825 | Permutation-invariant attention over sets (a peak-list encoder) | [A] |

---

## 2. Lessons for xrd-clip, ranked by (expected impact / cost)

**L1. Physics-simulated library + spectrum-to-spectrum comparison is the strong baseline, and it usually wins on exact identity. Use the embedding as a shortlist, not as the final judge.** (Impact: very high. Cost: near zero.)
- Evidence: ICEBERG [8] ranks candidates by predicted-spectrum vs experimental-spectrum similarity: 46% top-1 on MassSpecGym, vs 9.6 to 14.6% for MIST [6] and 11.9% for JESTR [7] (INFERENCE: the protocols may differ). In MS the forward model is learned and imperfect. In PXRD it is exact physics, which is why Pearson reaches 1.0 on clean patterns. Spectral entropy [1] shows a well-designed classical score can beat 42 alternatives.
- Action: rerank the embedding's top-k (k = 5 to 20; top-5 = 0.949 already bounds the ceiling) by full-profile similarity against *simulated patterns of the candidate structures*. Report the ceiling (recall@k) and the reranked top-1 separately.

**L2. Make the classical rerank score shift-tolerant, so it keeps the embedding's main advantage.** (Impact: high. Cost: low.)
- Evidence: Iwasaki [24] found constrained, normalized DTW best when the maximum shift is bounded; Pearson and cosine are best when it is unknown. VC-PWDF/VC-xPWDF [22, 23] adjust the cell before comparing and agree with COMPACK far better than fixed-cell PXRD tools (snippet numbers). XMatcher [30] does angular-shift correction before peak matching. The XRD literature also widely uses de Gelder's cross-correlation-weighted similarity (pre-2018, not fetched).
- Action: rerank candidates by a *max over* {Pearson, Pearson after optimal global 2θ shift and scale, constrained DTW with a band set by the realistic lattice strain, cell-adjusted comparison (VC-PWDF-like) when the pattern can be indexed}. This is cheap because it only runs on k candidates.

**L3. Feature-fusion reranker (MS2Query pattern).** (Impact: high. Cost: low.)
- Evidence: MS2Query [4] combines learned-embedding scores, a classical feature and precursor-mass features in a random forest over a learned shortlist, and beats modified-cosine analogue search. CReSS [15] (MW filter: 91.64% → 98.39% recall@10) and VibraCLIP [17] (mass integration, top-25 98.9%) show that cheap metadata gates add a lot.
- Action: a gradient-boosted or logistic scorer on the shortlist over [embedding cosine, embedding rank, Pearson, shift-aligned Pearson, DTW distance, fitted-shift magnitude, density/cell-volume difference, same-molecule flag]. Train it on hard shortlists from the training split. If the user knows the molecule (a polymorph-screening setting), gate the library by molecule identity first; this mirrors the MW/formula filters in [15, 17].

**L4. Hard-negative training with same-molecule polymorphs, and de-duplicate first.** (Impact: medium to high on within-family top-1. Cost: medium, needs retraining.)
- Evidence: JESTR [7]: regularizing against the most-similar same-formula candidates gave 6 to 37% relative rank@1. ANCE [41], Robinson [42] and NegCLIP [46] show hard negatives fix fine-grained failures that random in-batch negatives miss. RocketQA [40]: hard negatives are often *false* negatives, so denoise them. In PXRD, false negatives are CSD duplicates or redeterminations of the same form (COMPACK-identical; [23] gives a PXRD-based way to detect them).
- Action: build batches from polymorph families (all forms of molecule M in one batch), mine negatives from the model's own top-k (refreshed each epoch), and drop or merge pairs whose structures are COMPACK/VC-PWDF-identical. Treat duplicates as multi-positives [44, 45] instead of negatives.

**L5. Temperature and objective choice.** (Impact: medium. Cost: low.)
- Evidence: Wang & Liu [43]: lower temperature concentrates the loss on hard negatives, at the cost of semantic tolerance. De Waele [10]: optimizing similarity and optimizing retrieval trade off against each other, depending on how similar the candidates are.
- Action: sweep or learn τ. Consider a second fine-tuning stage with a listwise loss over the hard shortlist only, which optimizes within-family discrimination without destroying the global shift-invariant geometry used for the shortlist.

**L6. Pattern-to-pattern cross-encoder (learned reranker).** (Impact: potentially highest on noisy or experimental patterns where Pearson breaks. Cost: medium to high.)
- Evidence: the monoBERT/cross-encoder line [34, 37] reranks a shortlist with joint attention. Augmented SBERT [39] and RocketQA [40] use the cross-encoder to label and denoise data for the bi-encoder. Siamese spectral matchers work in MS [3], Raman [19] and pharma XRPD [26], with self-supervised losses helping generalization to unseen compounds [26]. GLMR [9] shows a learned second stage over a contrastive shortlist gives large top-1 gains (>40%).
- Caveat (INFERENCE): on *clean simulated* queries the cross-encoder cannot beat Pearson = 1.0. Its value appears only under perturbation (shift, broadening, preferred orientation, noise, impurity peaks). The evaluation must therefore include a perturbed or experimental test set, or this stage has nothing to show.

**L7. Peak-token / late-interaction encoders to cut the information loss of a single 256-d vector.** (Impact: medium. Cost: medium.)
- Evidence: DreaMS [12] tokenizes peaks (Set-Transformer style [51]). ColBERT [35] and FILIP [47] keep per-token vectors and score with MaxSim. Matryoshka [48] gives a short vector for the shortlist and a longer one for scoring, from one model.
- Action: the cheapest version is Matryoshka (or simply 512 to 1024-d) on the existing encoder. Next is multi-vector late interaction over 2θ-window tokens, where MaxSim with a small positional tolerance window gives shift tolerance by construction (INFERENCE).

**L8. Abstention / confidence.** (Impact: medium for practical use. Cost: low.)
- Evidence: Jürgens [11]: retrieval-level uncertainty is the best rejection criterion. Raman matching [19] provides conformal sets.
- Action: report the reranker margin (top-1 minus top-2 score) and return a conformal set of polymorphs when the margin is small.

**Low priority / not supported here:** test-time augmentation for retrieval. No fetched paper in this track reports gains for it on spectra. It is the trivial version of L2 (averaging scores over shifted queries).

---

## 3. Recommended rerank-stage architecture

**Stage 0 (optional gate):** filter the library by molecule identity, or by cell volume/density when known (from CReSS [15] and VibraCLIP [17]).

**Stage 1 (shortlist):** the existing xrd-clip bi-encoder, top-k with k ≈ 10 to 20. The top-5 recall of 0.949 suggests k = 10 to 20 pushes the ceiling to ≥ 0.97 (INFERENCE; measure it). Optionally retrain with L4/L5 to raise the ceiling, not top-1.

**Stage 2a (training-free, ship first):** a classical shift-tolerant profile score between the query pattern and each candidate's simulated pattern: max of [Pearson, globally shift- and scale-aligned Pearson, band-constrained DTW, VC-PWDF-style cell-adjusted difference] (from [1], [8], [22 to 24], [30]).

**Stage 2b (learned fusion, MS2Query-style [4]):** a GBDT or logistic model over {embedding cosine and rank, Stage 2a scores, fitted shift, metadata deltas}. Train it on shortlists produced by Stage 1 on the training split, with perturbation augmentations on the query.

**Stage 3 (pattern-to-pattern cross-encoder, research contribution):** input the pair (query pattern, candidate simulated pattern). Encode both with a shared 1D CNN trunk (anti-aliased [50]), followed by cross-attention between the two token sequences (monoBERT-style joint encoding [34]). Optionally add a soft-DTW alignment layer [49] that warps the candidate onto the query before the cross-attention. Train with a listwise softmax over the Stage 1 shortlist (hard negatives by construction), with false negatives removed by COMPACK/VC-PWDF de-duplication [23, 40]. Initialize from the bi-encoder's pattern branch, and optionally pretrain contrastively on pattern augmentations [26, 29].

**Why this design:**
1. In every spectroscopy where a good forward model exists, comparing in spectrum space beats the joint embedding at exact identity [1, 8]. The joint embedding stays the best tool for robust recall [7, 15, 17].
2. Hybrid two-stage systems are the published winners in MS [4, 9].
3. Shift tolerance is solved classically when bounded [24, 22, 30], so the learned stage should target what classical scores cannot handle: intensity distortions, preferred orientation (VC-xPWDF's stated weak point [22]), impurity peaks and noise.
4. Pattern-to-pattern (Stage 3) is preferable to pattern-to-structure because the library can always be simulated, and the model then needs to learn only *distortion invariance*, not crystallography. This matches the Siamese spectral-matching results [3, 19, 26].

**Minimum experiment set:**
- (a) bi-encoder alone vs + Stage 2a vs + 2b vs + 3,
- (b) on clean, shifted, broadened, preferred-orientation and noisy query sets,
- (c) within-polymorph-family top-1 as the headline metric, with recall@k reported as the ceiling.

---

## 4. UNVERIFIED (not cited as fact above, or only partly verified)

- VC-PWDF "2.84% vs 12.35% disagreement with COMPACK": from a search-result snippet of the RSC page. The abstract was not read; the OpenAlex inverted index lacked the numbers.
- VibraCLIP numbers: read on the ICIQ institutional page, not the publisher abstract (403).
- SMEN "~81% top-1": search snippet only.
- MIST "66% equal or improved, 29% strictly better": search snippet only. The NMI DOI was not fetched.
- MS2Query details "random forest over 5 features", "35% reliable analogues, mean Tanimoto 0.67": search snippet / PMC listing, not the fetched abstract.
- CLERMS and SIRIUS 4 / CSI:FingerID: existence confirmed via OpenAlex, but no results read.
- Whether ICEBERG's MassSpecGym 46% top-1 uses the same candidate sets as the MassSpecGym MIST and JESTR numbers: unverified, so the cross-paper comparison is indicative only.
- de Gelder cross-correlation-weighted PXRD similarity and COMPACK (Chisholm & Motherwell 2005): pre-2018 background, not fetched.
- MS2DeepScore 2.0, CSU-MS2 and "ESP" (ensemble spectral prediction): mentioned in JESTR's comparison table but not independently fetched.
- No fetched source reported test-time augmentation gains for spectral retrieval. Searches run: none specifically for TTA (out of budget), so this is **not** a claim that none exist.
- Raman polymorph CNN work (e.g., the TiO2 Raman polymorph PMC9763332) was seen in a search listing only. It is inorganic and was not fetched.
- Searches run for "learned COMPACK-like packing similarity": "neural network molecular crystal packing similarity COMPACK replacement learned embedding polymorph clustering CSP" and "machine learning crystal structure similarity duplicate removal crystal structure prediction landscape molecular crystals learned descriptor SOAP". They found only [31] and [32] (classification and kernels), no learned COMPACK replacement. Two searches are not exhaustive.
