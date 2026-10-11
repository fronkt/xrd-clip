# PA69 speaker notes, paste-ready

Each slide's block below replaces that slide's existing notes. Copy everything under the slide heading and paste it into the notes box. SAY is the timed script, the same length as before (about 8 minutes total). IF ASKED is background for you, not to be read aloud.

## Slide 1

SAY [~20 s]
Hi, I'm Frank Cai. This talk is about identifying which crystal form of a molecule you have, straight from its powder X-ray diffraction pattern, including the hard case where every candidate is the same molecule.

IF ASKED
- One line: a model that reads a powder pattern and picks which packing of a molecule produced it.
- PXRD: X-rays (Cu K-alpha-1, 1.5406 Å) hit a powder of millions of randomly oriented crystallites. Planes with spacing d diffract at Bragg's law, n lambda = 2 d sin theta. You get rings, which are collapsed into intensity vs 2-theta.
- Peak positions come from the unit cell (size and shape). Peak intensities come from where the atoms sit inside the cell.
- It's the cheapest standard check of "which solid form is this?" in pharma QC and materials labs.

## Slide 2

SAY [~50 s]
Polymorphs are the same molecule packed differently. This is ROY, a molecule with more than ten known polymorphs, named for its red, orange and yellow crystals. The yellow and orange-plate forms have the same formula and the same space group. The molecule just twists differently, 105 versus 46 degrees, and packs differently. Their simulated powder patterns, at the bottom, are clearly different, and that is the only signal that separates them. This matters because packing controls solubility and stability: ritonavir's marketed form was withdrawn when a less soluble polymorph appeared in manufacturing.

IF ASKED
- Different packing changes density, melting point, solubility and stability, so for a drug it changes how much reaches the blood.
- The colour changes because the ring twist changes the conjugation.
- P2₁/n is monoclinic space group number 14 (the same group as P2₁/c in another setting), one of the most common for organics. Same space group means symmetry can't tell the forms apart.
- Structures: CSD QAXMEH01 (Y) and QAXMEH03 (OP). Six letters name the compound; the digits number its separate determinations.
- Ritonavir (Norvir, an HIV drug) launched in 1996 as Form I. A more stable, much less soluble Form II appeared in 1998, the capsules failed dissolution and were pulled, and the drug had to be reformulated (Bauer 2001).

## Slide 3

SAY [~50 s]
The usual tool is search-match: correlate the measured pattern against a reference library. Recent contrastive models such as XCCP and XtalNet do learned retrieval instead, but on inorganic crystals and MOFs, and they lean on composition and symmetry; XCCP reports 93% space-group ID. For polymorphs those cues are gone. So the task here is within-family retrieval: the query is one polymorph's pattern, and the library is only its siblings. Anything that uses only formula or symmetry scores 0.380, the family-size chance rate.

IF ASKED
- Search-match uses the ICDD Powder Diffraction File and ranks by full-profile correlation or peak-list matching. It breaks when peaks shift, because shifted peaks no longer line up with the reference.
- XCCP: Xu 2025, 155,003 inorganic Materials Project crystals, with a symmetry-oriented encoder branch. XtalNet: Lai 2025, hypothetical MOFs, and its contrastive module feeds a structure generator.
- Chance: with k siblings, a random pick is right 1/k of the time (2 siblings = 0.5, 3 = 0.33). Averaging 1/k over all 2,835 test queries gives 0.380.
- Within-family = choose among the siblings only. Slide 10's real-data check searches the full 30,887-structure library.

## Slide 4

SAY [~50 s]
The model is CLIP-style: two encoders map into one shared embedding space. The pattern side is a 1D residual CNN, because a diffractogram is a 1D signal where local peak shape carries the physics. The structure side is a Transformer over the set of atoms with a lattice token, because peak positions are set by the unit cell. Training uses CLIP's symmetric InfoNCE loss: each pattern should match its own structure and not the other structures in the batch. At test time we rank the candidate structures by cosine similarity.

IF ASKED
- CLIP (OpenAI, 2021) trains image and caption encoders so that matching pairs land close together. Same idea here, with pattern and structure.
- The CNN applies the same filters at every 2-theta, and global average pooling averages over position, so small peak shifts barely change the output. That's why shift tolerance comes for free (slide 9).
- Input: 5 to 50° 2-theta, 0.02° steps, 2,251 points.
- Transformer: atoms are an unordered set (permutation invariant). Each atom gets an element embedding plus a sin/cos encoding of its fractional coordinates, which is periodic, so x and x+1 look identical, matching crystal periodicity. The lattice token carries a, b, c and the angles.
- Both sides output 256-d unit vectors, so cosine similarity is a dot product.
- InfoNCE: batch of 512 pairs, a 512 x 512 similarity matrix, cross-entropy in both directions (pattern to structure and structure to pattern), averaged. The other 511 pairs are negatives. The learnable temperature is clamped so it can't collapse.
- Patterns are simulated on the fly with pymatgen, then passed through the random measurement model, so no pattern is seen twice.

## Slide 5

SAY [~45 s]
There are two design choices. First, augmentation: every training pattern is re-simulated through a randomized instrument model with sample displacement, lattice strain, peak broadening, preferred orientation, background and counting noise. So the query looks like a real measurement and not like the clean library entry. Second, polymorph-aware training: sibling polymorphs are put in the same batch as hard negatives, and whole families go into a single split, so no sibling of a test structure is ever seen in training. I checked: zero of 10,795 families leak across splits.

IF ASKED
- Displacement: the sample surface sits off the focusing circle, so all peaks shift, roughly in proportion to cos theta.
- Strain: the cell expands or contracts (temperature, composition). Expansion moves peaks to lower 2-theta, more at high angle (in proportion to tan theta).
- Caglioti: instrument width, FWHM² = U tan²θ + V tan θ + W. Scherrer: small crystals broaden peaks, width = K lambda / (L cos theta), K ≈ 0.9. 10 nm is nanocrystalline.
- March–Dollase: plates or needles lie flat in the holder, so some peaks get stronger and others weaker. It changes intensities, not positions.
- Background: a Chebyshev polynomial plus an amorphous hump. Poisson noise: photon counting, so variance = counts.
- Hard negatives: random batches are easy (the formulas differ). Siblings in one batch force the model to learn packing.
- Bemis–Murcko scaffold = ring systems plus linkers. A scaffold split makes the test molecules new chemistry, and on top of it whole families stay in one split.
- Data: organic, no disorder, R ≤ 7.5%, Z′ ≤ 2. That gives 315,060 of about 1.45M CSD entries. Batch 512, 60 epochs.

## Slide 6

SAY [~40 s]
On clean patterns the model picks the right polymorph 56% of the time against a 38% chance floor, a 1.48x lift, and it's in the top two 84% of the time. One caveat up front: on a perfectly clean query, plain correlation gets 100%, because the query is literally the library entry. That's not a realistic setting. What matters is what happens when the measured pattern carries real instrument effects.

IF ASKED
- Top-k = the true structure is in the k highest-ranked candidates. MRR = the mean of 1/rank (rank 1 = 1, rank 2 = 0.5). 0.738 means it's usually first or second.
- Lift: 0.561 / 0.380 = 1.48.
- Top-2 at 84% narrows the candidates to two, which a quick Rietveld fit can settle.
- In-batch validation top-1 was 0.914, but that's the easy random-batch task, not within-family.
- "Is 56% enough?" It's the first number for this task (there's no prior baseline), it's well above chance, and it holds where classical methods fall to chance.

## Slide 7

SAY [~55 s]
Here we degrade the query one dial at a time; the library stays clean, just like a real reference database. Blue is our model, red is full-profile correlation, green is peak-list matching, and dashed is chance. Under sample displacement and lattice strain, the peak shifts that break search-match in real labs, the model stays flat while correlation drops below chance. Panel c is the honest limitation: severe Scherrer broadening keeps the peak positions, and correlation handles that better than we do.

IF ASKED
- (a) Displacement: model 0.561 to 0.523; correlation 1.0 to 0.298, and below the model past about 0.15°.
- (b) Strain: model 0.561 to 0.546; correlation 1.0 to 0.323.
- (c) At 10 nm crystallites: correlation 0.884, model 0.472.
- Why shifts kill correlation: peaks are narrow, so a peak shifted by more than its width stops overlapping its reference.
- Displacement errors of hundredths to a tenth of a degree are routine on lab instruments. Temperature differences cause strain.

## Slide 8

SAY [~55 s]
With everything applied at once, in panel a, classical methods look decent. But that's partly luck: sample displacement and thermal expansion shift peaks in opposite directions, so they can cancel. Panel b is the control: the same shift sizes, with only the relative sign flipped. When the shifts add, correlation and peak-list both drop to chance, while the model barely moves. In a real sample you don't know in advance which case you're in, and roughly half the time it's the bad one. The model's invariance doesn't depend on the sign.

IF ASKED
- (a) Random shift signs per crystal on top of broadening, texture, noise and background: model 0.554 to 0.521 (mild to harsh), correlation 0.782 to 0.599, peak-list 0.797 to 0.601.
- (b) Fixed at 0.10° displacement and 0.6% strain. Cancelling: correlation 0.807, peak-list 0.784. Adding: 0.334 and 0.340, both at chance. Model: 0.555 to 0.534.
- The two shifts scale differently with angle (cos theta vs tan theta), so they reinforce or cancel most across the mid-angle range.
- Point to make: a fixed-sign stress test can flatter classical methods.

## Slide 9

SAY [~55 s]
Two ablations answer where this robustness comes from. Remove only the shift and strain augmentation, and nothing changes, so shift invariance comes from the architecture. Remove all augmentation, and the model still works on clean and pure-shift queries, but it falls to exactly chance once broadening, texture, noise and background appear. That answers the obvious objection: isn't this just memorizing simulated fingerprints? If it were, the clean-trained twin would still work. It doesn't, so the augmentation is teaching real generalization.

IF ASKED
- An ablation means retraining with one ingredient removed. All three twins share the architecture, data and 60-epoch schedule.
- Emergent = it appears without being trained in. Convolution plus global pooling tolerates translation, and small strain looks locally like a translation.
- The clean twin's crystallite-size sweep: 0.557, 0.410, 0.395, 0.390 at 100, 50 and 20 nm.
- The red cells mark the clean-trained twin at chance.
- The leakage worry: the query is simulated from the target structure itself. The clean twin shows that it can't just match fingerprints once real effects appear.

## Slide 10

SAY [~40 s]
As a first look at transfer, I took real published powder patterns. For ibuprofen, a real pattern the model never saw puts the true structure in the top 6% of a 30,000-structure library, which is good but not top-1. This is a spot check, not a benchmark, and most of the evaluation is still on the simulator. The model also concedes heavy broadening, and 56% top-1 leaves room to improve.

IF ASKED
- The real patterns are Rietveld data deposited as supporting information with the original structure papers, digitized point by point, Cu K-alpha-1.
- Ibuprofen IBPRAC04 (Derollez 2010): rank 1,872 of 30,887 in the full library, not within-family. The midpoint is about 15,400.
- Phenobarbital Forms I/II (Platteau 2005): those determinations aren't in the library, so there's no rank. The nearest neighbours weren't plausible matches, which is correct behaviour.
- Rietveld = least-squares fit of a calculated pattern to the measured one. It's accurate but needs a starting structure and expert setup. Retrieval proposes, Rietveld confirms.
- Low/high-angle split: low angles mostly reflect the cell, high angles carry finer atomic detail (XCCP uses a split).
- Inorganic and MOF materials are left out on purpose: XCCP and XtalNet cover them.

## Slide 11

SAY [~25 s]
To sum up: a new polymorph retrieval task where formula and symmetry don't help; a model that resolves polymorphs above chance and stays robust where classical search-match fails; and ablations that show why. Next, the same retriever could rank candidates for PXRD-conditioned crystal structure generation. Thank you, I'm happy to take questions.

IF ASKED
- Why simulated data? There's no labeled real PXRD for 300k molecular crystals. Simulation from known structures gives the scale, and the augmentation closes part of the gap.
- Why not just Rietveld? It needs a candidate and expert time; retrieval gives a ranked shortlist almost instantly.
- Mixtures? Not tested; the model assumes a single phase. That's future work.
- Other wavelengths (Mo, synchrotron)? It's trained on Cu K-alpha-1 only, so another wavelength needs re-simulation or conversion to d-spacing.
- Data sharing? The CSD is licensed, so the structures can't be redistributed.
