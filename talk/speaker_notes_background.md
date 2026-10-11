# PA69 speaker notes: background blocks

Appended under the timed script in each slide's notes in the Google Slides deck
(id 11z8gjIEaTM8Tjn_aDjVJsMbI3pieUkQhMV_MIDghDrI). Not for reading aloud; for
understanding and Q&A. Facts are taken from paper/main.tex, plus standard textbook physics.

## Slide 1

---- BACKGROUND (not for reading aloud) ----
One-sentence version: I trained a model that looks at a powder diffraction pattern and picks which crystal packing of a molecule produced it, even when all candidates are the same molecule.
PXRD in 30 seconds: you shine X-rays (here Cu K-alpha-1, wavelength 1.5406 Angstrom) on a powder made of millions of tiny, randomly oriented crystals. Each set of lattice planes with spacing d diffracts at the angle given by Bragg's law, n*lambda = 2 d sin(theta). Because the crystallites point every direction, you get rings, which are collapsed into a 1D plot of intensity vs 2-theta. Peak POSITIONS come from the unit cell (its size and shape). Peak INTENSITIES come from where the atoms sit inside the cell (the structure factor).
Why it matters: PXRD is the cheapest, fastest standard check of "which solid form is this?" in pharma QC, materials synthesis and geology.
If asked "who did this": me alone, as an independent project, using the licensed Cambridge Structural Database.

## Slide 2

---- BACKGROUND (not for reading aloud) ----
Polymorph: the same molecule crystallizing in more than one arrangement. Different packing means different density, melting point, solubility, dissolution rate and stability, so for a drug the polymorph changes how much of it reaches the blood.
ROY = 5-methyl-2-[(2-nitrophenyl)amino]-3-thiophenecarbonitrile, named for its Red, Orange and Yellow crystals. It has more than ten structurally characterized polymorphs, which made it the standard example of polymorphism. Colour changes because the twist between the two rings changes the conjugation.
Space group: the symmetry rules of the crystal (230 types). P2(1)/n is a monoclinic group (the same group number 14 as P2(1)/c in a different setting) and is one of the most common for organic molecules. Same space group means symmetry cues cannot tell Y from OP.
CSD refcodes: six letters name the compound (QAXMEH = ROY); the two digits number its separate determinations, which can be redeterminations or different polymorphs.
Ritonavir (Abbott's Norvir, an HIV protease inhibitor): launched in 1996 as Form I. In 1998 a more stable, much less soluble Form II appeared in manufacturing. The capsules failed dissolution tests, were pulled, and the drug had to be reformulated (Bauer et al., 2001). The standard example of why polymorph ID matters.

## Slide 3

---- BACKGROUND (not for reading aloud) ----
Search-match: the classical method. Compare the measured pattern against a reference database (ICDD Powder Diffraction File) by full-profile correlation or by matching peak lists, then rank by a figure of merit. It works when candidates differ a lot (different compound, different symmetry). It is fragile when peaks are shifted, because a shifted peak no longer lines up with its reference.
XCCP (Xu et al., 2025): contrastive PXRD-structure model on 155,003 inorganic Materials Project crystals. It reports 93.4% space-group ID and has a symmetry-oriented encoder branch, so symmetry is part of what it uses.
XtalNet (Lai et al., 2025): its contrastive module (CPCP) works on hypothetical MOFs and feeds a structure generator.
Why those cues fail here: polymorphs share the formula, so composition says nothing, and they often share the space group too.
Chance 0.380: for a family of k siblings, random guessing is right 1/k of the time (2 siblings = 0.5, 3 = 0.33, and so on). Averaging 1/k over all 2,835 test queries gives 0.380. Anything that only knows formula or symmetry cannot beat this, by construction.
Within-family vs full library: within-family = choose among the siblings only (the hard, fine-grained task). The real-data check on slide 10 searches the full 30,887-structure test library.

## Slide 4

---- BACKGROUND (not for reading aloud) ----
CLIP (Radford et al., OpenAI, 2021): trains an image encoder and a text encoder together so a matching image and caption land close together in one shared vector space. I use the same idea with pattern and crystal structure in place of image and text.
Pattern encoder: a 1D residual CNN (ResNet). Convolutions slide the same filter along 2-theta, so a peak is detected the same way wherever it sits. Global average pooling then averages over all positions, so the output barely changes if peaks move slightly. That is why shift tolerance comes "for free" (slide 9). Residual connections (skip connections) let deep CNNs train stably.
Input: 5 to 50 degrees 2-theta, 0.02 degree steps = 2,251 points.
Structure encoder: a Transformer over the atoms. It treats atoms as a set (no order, so it is permutation invariant). Each atom gets an element embedding plus a sin/cos encoding of its fractional coordinates; sin/cos is periodic, so an atom at x and at x+1 (the next cell over) look identical, which matches crystal periodicity. A lattice token (cell lengths a, b, c and angles) is added up front, because peak positions are set by the cell.
Shared 256-d embedding: both outputs are normalized to unit length, so cosine similarity is just a dot product.
Symmetric InfoNCE: for each batch of B = 512 pairs, compute all 512 x 512 similarities. Each pattern must pick its own structure out of 512 (cross-entropy), and each structure must pick its own pattern; the two losses are averaged. The other 511 act as negatives. A learnable temperature sets how sharp the softmax is; it is clamped so it cannot collapse.
On-the-fly simulation: patterns are computed from structures with pymatgen (diffraction "sticks") and then passed through the random measurement model, so no pattern is seen twice.

## Slide 5

---- BACKGROUND (not for reading aloud) ----
What each augmentation models:
- Zero offset / sample displacement: the sample surface sits slightly above or below the focusing circle, shifting every peak. The shift scales roughly with cos(theta).
- Isotropic lattice strain: the whole cell is a bit bigger or smaller (thermal expansion, slight composition change). Expansion moves peaks to LOWER 2-theta, by an amount that grows with tan(theta). The two shifts have different angular dependence and can point in opposite directions, which is the basis of slide 8.
- Caglioti broadening: instrument peak width vs angle, FWHM^2 = U tan^2(theta) + V tan(theta) + W.
- Scherrer broadening: small crystallites give broader peaks, width = K lambda / (L cos theta) with K about 0.9 and L the crystallite size. 10 nm is very small (nanocrystalline).
- March-Dollase preferred orientation: plate- or needle-shaped crystals lie flat in the holder, so some reflections get stronger and others weaker. This changes intensities, not positions.
- Background: a smooth Chebyshev polynomial plus a broad amorphous "hump" (non-crystalline material, sample holder, air scatter).
- Poisson noise: detector counts are photon counts, so the noise variance equals the count.
Hard negatives: if a batch has random crystals, telling them apart is easy (different formulas). Putting siblings in the same batch forces the model to learn the subtle packing differences.
Scaffold split (Bemis-Murcko): a molecule's scaffold is its ring systems plus the linkers between them. Splitting by scaffold keeps chemically similar molecules on the same side, so test molecules are new chemistry. On top of that, a whole polymorph family stays in one split, so no sibling of a test structure is seen in training.
Data filters: organic, no disorder, R <= 7.5% (R-factor = refinement quality), Z' <= 2 (at most two molecules in the asymmetric unit). 315,060 crystals out of 1.45 million CSD entries. Batch 512, 60 epochs.

## Slide 6

---- BACKGROUND (not for reading aloud) ----
Top-k: the true structure is among the model's k highest-ranked candidates. Top-1 is the strict "picked it" number.
MRR (mean reciprocal rank): average of 1/rank of the correct answer (rank 1 = 1.0, rank 2 = 0.5, rank 3 = 0.33). 0.738 means the right answer is usually first or second.
Lift: 0.561 / 0.380 = 1.48x.
How to read 0.561: within a family, it picks the exact packing more than half the time, and 84% of the time it is in the top two. In practice that cuts the candidates to two, which a quick Rietveld fit can settle.
Why correlation scores 1.0 on clean queries: the clean query is computed exactly like the library entry, so it is a perfect self-match. This tells you nothing about real samples, which is why the robustness slides are the real result.
Validation note: in-batch validation top-1 was 0.914, but that is the easy random-batch task, not the within-family one.
Likely question, "Is 56% good enough?": it is the first number for this task (no prior baseline exists), it is well above chance, and it holds under realistic shifts where classical methods fall to chance.

## Slide 7

---- BACKGROUND (not for reading aloud) ----
Setup: only the QUERY gets degraded, one physical effect at a time; the library stays clean. This copies reality: a reference database is clean simulated or ideal data, and your measured sample is not.
Baselines: full-profile correlation (Pearson correlation of the whole curve, red) and peak-list matching (compare the lists of peak positions, green). Grey dashed = 0.380 chance.
(a) Zero offset: model 0.561 -> 0.523; correlation 1.0 -> 0.298, falling below the model past about 0.15 degrees.
(b) Lattice strain: model 0.561 -> 0.546; correlation 1.0 -> 0.323.
(c) Scherrer broadening: peaks get wider but stay in place, so correlation still lines up (0.884 at 10 nm) while the model drops (0.472). This is the honest limitation.
Why shifts kill correlation: diffraction peaks are narrow. Shift one by more than its width and it no longer overlaps its reference peak, so the correlation drops toward that of an unrelated pattern.
Real-world size of these effects: a few hundredths to a tenth of a degree of displacement error is routine on lab instruments, and temperature differences between the reference and your sample cause strain.

## Slide 8

---- BACKGROUND (not for reading aloud) ----
The key idea: sample displacement and lattice expansion both shift peaks, and depending on sign they can push the same way (add) or opposite ways (cancel).
(a) All effects together (broadening, texture, noise, background, both shifts) with random shift signs per crystal: the model stays near 0.55 (0.554 -> 0.521, mild to harsh), correlation falls 0.782 -> 0.599, peak-list 0.797 -> 0.601.
(b) Control: fix the shift sizes (0.10 degree displacement, 0.6% strain) and flip only their relative sign. Cancelling: correlation 0.807, peak-list 0.784. Adding: correlation 0.334, peak-list 0.340, both at chance. Model: 0.555 -> 0.534.
Why this matters: a naive "everything at once" test with fixed signs can make classical methods look robust purely because the shifts happen to cancel. In a real sample you don't know the signs in advance, so classical methods are only robust about half the time, by luck. The model doesn't depend on the sign.
"Mid-angle regime": because the two shifts scale differently with angle (cos theta vs tan theta), they reinforce or cancel most strongly over the middle of the 5-50 degree range.

## Slide 9

---- BACKGROUND (not for reading aloud) ----
Ablation = retrain the model with one ingredient removed and see what breaks. Here, three twins with identical architecture, data and 60-epoch schedule; only the augmentation differs.
Twin 1 (no shift/strain augmentation): within about 1% of the main model everywhere, even under shifts. So shift tolerance does not come from training on shifted examples. It comes from the architecture: convolution plus global average pooling is naturally tolerant to translation, and small strain looks locally like a translation. "Emergent" = it appears without being trained in.
Twin 2 (no augmentation, clean patterns only): still fine on clean queries (0.557) and pure shifts (0.552), but falls to exactly chance (0.380/0.379) as soon as realistic broadening, texture, noise and background appear. The crystallite-size sweep alone walks it down 0.557 -> 0.410 -> 0.395 -> 0.390 (100/50/20 nm).
The objection this answers: "The query is simulated from the target structure, so isn't the model just memorizing simulated fingerprints (leakage)?" If it were memorizing, the clean-trained twin would still match. It can't, so the augmentation is teaching real generalization to measurement effects. "Learned" = only present because of the physics augmentation.
Red cells in the table = the clean-trained twin at chance.

## Slide 10

---- BACKGROUND (not for reading aloud) ----
Sim-to-real gap: everything else is evaluated on my own simulator, so the open question is whether it transfers to real instruments.
Where the real patterns came from: Rietveld refinement data that authors deposited as supporting information with the original structure papers (IUCr journals), digitized point by point. They are Cu K-alpha-1, matching my simulation.
Ibuprofen (IBPRAC04, Derollez et al., 2010): the structure is in the corpus, but the model has never seen this real measured pattern. Searching the FULL 30,887-structure test library (not just siblings), the right one ranks 1,872, top 6.1%. Far better than random (the midpoint is about 15,400), but not top-1.
Phenobarbital Forms I/II (Platteau et al., 2005): these specific determinations are not in the library (only sibling PHBARB entries are), so there is no correct answer to rank. Nearest neighbours were not chemically plausible matches, which is the correct behaviour: retrieval can only find what is in the library.
Rietveld refinement: least-squares fitting of a full calculated pattern to the measured one by adjusting structure and instrument parameters. Accurate, but it needs a good starting structure and expert setup. Retrieval supplies the candidates; Rietveld confirms them. That is the "Rietveld reranking" next step.
Low/high-angle split: low-angle peaks mostly reflect the cell, high-angle peaks finer atomic detail; separate encoders for the two ranges (as XCCP does) may help.
Inorganic and MOF materials are left out on purpose: XCCP and XtalNet already cover them, and composition cues work there.

## Slide 11

---- BACKGROUND (not for reading aloud) ----
Three claims to land: (1) a new task where formula and symmetry can't help; (2) the model beats chance and stays robust under the peak shifts that defeat search-match, whatever their sign; (3) ablations show why: shift tolerance from the architecture, profile robustness from the physics augmentation.
Future direction: generative models can propose candidate crystal structures from a PXRD pattern; a retriever like this one can rank and filter their outputs.
Likely Q&A, short answers:
- Why simulated training data? Labeled real PXRD for 300k molecular crystals doesn't exist; simulation from known structures is the only way to get scale, and the augmentation closes part of the gap (slide 9).
- Why not just Rietveld? It needs a candidate structure and expert time; retrieval gives a ranked shortlist in milliseconds.
- What about mixtures? Not tested; the model assumes a single phase. Mixtures are future work.
- Different wavelength (Mo, synchrotron)? Trained for Cu K-alpha-1 only; another wavelength would need re-simulation or conversion to d-spacing.
- Data? The CSD is a licensed database, so the extracted structures can't be redistributed.
