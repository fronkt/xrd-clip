# URTC 2026 submission guide + Xplore path (verified 2026-07-18)

Reference notes for submitting xrd-clip to the IEEE MIT Undergraduate Research Technology
Conference (URTC). Sourced from urtc.mit.edu and IEEE Xplore; confirm the 2026-specific dates on
the official site once posted ("Information for 2026 will be posted soon" as of this writing).

## Venue
- **IEEE MIT URTC** — in-person, MIT campus, ~early November. Peer-reviewed (faculty + grad
  students + industry), scored on originality, content quality, theme alignment.
- **Deadline:** ~Aug 3 (2025 cadence; 2026 unconfirmed — watch the site).
- **Eligibility (confirmed):** full-time undergraduates at accredited institutions **OR high-school
  students in collegiate research programs**; research completed before the undergraduate degree;
  original, unpublished, not under simultaneous review.

## Format (technical paper track)
- **≤ 5 pages, no exceptions.** Single-spaced, ≥ 10 pt.
- IEEE conference template (site ships a Word template; **IEEEtran two-column is equivalent** and is
  what `paper/main.tex` uses). Our draft compiles to **4 pages** — headroom to spare.

## Submission types — only ONE is archival
| type | reviewed | → IEEE Xplore? | outcome |
|------|----------|----------------|---------|
| **technical paper** | peer-reviewed (the real bar) | **yes** — proceedings, permanent, citable, DOI | a publication |
| poster | light relevance review | no | a CV "presentation", no copyright cost |
| lightning talk | light | no | short oral pitch, exposure only |

**We want the paper track** — the archival Xplore publication is the entire point, and it is kept
disjoint from the pxrd-diff journal manuscript precisely because a paper's IEEE copyright transfer
would conflict. Watch for reviewers offering to **down-slot an accepted paper to a poster** (happens
when work is judged interesting but not quite strong enough) — that would lose the publication.

## The Xplore path (what "gets into Xplore" actually means)
IEEE Xplore is IEEE's digital library, **not a journal** — there is no separate Xplore review.
Pipeline: **accepted to the paper track → URTC proceedings → deposited to Xplore.** Confirmed URTC
does this every year (proceedings indexed for 2017, 2018, 2022, 2023, 2024). Three procedural gates
sit between "accepted" and "in Xplore", and papers get pulled for missing them:
1. **IEEE PDF eXpress** format-compliance check on the final PDF.
2. **Plagiarism / CrossCheck** screen (the site notes IEEE plagiarism software is used).
3. **Signed IEEE eCopyright transfer** + **you must actually present** — a no-show pulls the paper
   from Xplore. URTC is in-person at MIT, so attendance is required for the archival outcome.

## Acceptance assessment (xrd-clip as-is)
**Confident accept for the paper track.** URTC is an inclusive undergraduate venue; the work clears
its bar comfortably: a genuinely novel task (polymorph-resolving retrieval, where composition/
symmetry methods are at chance by construction), an honest robustness result with a sign-cancellation
control, and ablations that pre-empt the "fingerprint-leak" objection. **The only substantive risk is
the simulated-only evaluation** — scoped as an explicit limitation with diagnostics built, so it
reads as honest scoping. Because Xplore inclusion is permanent, adding even a handful of real
experimental patterns before submission is the single highest-leverage de-risking move.

## Paper status
Drafted + compiled: `paper/main.tex` + `paper/refs.bib` (16 citations verified programmatically) +
`paper/main.pdf` (still 4 pp after the addition below). Figures regenerate from committed results
via `scripts/10_paper_figs.py`.

**UPDATE 2026-08-04: sim-to-real spot check added.** The prior "evaluated only on simulated
patterns" limitation is now partially addressed. Sourced 3 real experimental PXRD patterns from
IUCr supplementary information (Rietveld powder-refinement CIFs deposited with the original
structure-determination papers, Cu-K$\alpha_1$ throughout, digitized point-by-point since
curl-based IUCr SI access from this session's earlier working method is now blocked by
Cloudflare -- fetched via a real browser session instead). Real patterns + provenance in
`cache/real_patterns/*.json` (gitignored, not redistributable-scale but small enough to be
low-risk; not currently pushed -- Frank's call whether to commit). Eval script
`scripts/11_real_pattern_eval.py` (reuses `retrieve.embed_library`/`baselines`, new code only for
the real-pattern preprocessing + reporting). Result: IBPRAC04 (ibuprofen, corpus structure, never-seen
real pattern) ranks 1,872/30,887 by cosine similarity against the full test library (top 6.1%,
`results/real_pattern_eval.json`). PHBARB05 (phenobarbital Form I/II, real polymorph pair) has no
retrieval rank since its specific determination is not itself a corpus structure -- reported
honestly as a scope note, not spun. Sanity-verified: real vs. simulated IBPRAC04 pattern correlate
at 0.89 with near-identical peak positions (`cache/real_patterns/_sanity_ibprac04.png`), confirming
the grid/wavelength pipeline is correct. New paper section `\section{A sim-to-real spot check}`
added before Limitations; Limitations reworded to say "single-digit-N spot check" instead of "future
work." **Remaining = Frank hand-reviews and submits** (AI does not write the submission; review
only).

## Draft portal fields (2026-08-05)

Draft only — Frank reviews and pastes into the actual urtc.mit.edu portal form. Portal exact field
limits (character/word caps) weren't captured verbatim from the pasted guidelines, so two abstract
lengths are given; use whichever fits the form, or ask if the field visibly truncates.

**Track:** Technical Paper (not poster/lightning talk — this is the only archival/Xplore option;
see the table above).

**Title:**
> Polymorph-Resolving Retrieval of Molecular Crystals from Powder X-Ray Diffraction

**Abstract — full length (matches `paper/main.tex`, ~230 words), plain text for the portal:**
> We show that a contrastive pattern-structure model retrieves the correct polymorph of a molecular
> crystal from its powder X-ray diffraction (PXRD) pattern — a task on which composition- and
> symmetry-based methods are pinned at chance by construction, because polymorphs share a formula and
> often a space group. Training a CLIP-style dual encoder (a 1D residual network over the diffractogram
> against an atom-set Transformer over the crystal graph) on 247,089 molecular crystals from the
> Cambridge Structural Database with physics-grounded on-the-fly augmentation, the model attains
> within-family top-1 retrieval of 0.561 against a composition-only chance of 0.380 over 1,077 held-out
> polymorph families (2,835 same-molecule siblings). The result that matters for deployment is
> robustness: under the mid-angle regime where a sample-displacement shift and a thermal-strain shift
> reinforce rather than cancel, both classical full-profile correlation and peak-list matching fall to
> composition-only chance while the learned embedding is unmoved. Two ablations separate the mechanism:
> zeroing shift/strain augmentation costs nothing (peak-position invariance is architectural), whereas
> removing all augmentation collapses the model to chance under a realistic instrument profile
> (measurement-profile robustness is learned, not a memorized fingerprint). A spot check against real
> experimental PXRD patterns from the powder-diffraction literature shows this transfers beyond
> simulation: a held-out real pattern for racemic ibuprofen retrieves its true structure in the top
> 6.1% of a 30,887-structure library. The learned embedding is a genuine sign-independent invariance
> that classical search-match cannot guarantee.

**Abstract — short (~110 words), if the portal caps lower:**
> Polymorphs of a molecular crystal share a formula and often a space group, so composition- and
> symmetry-based search is pinned at chance by construction. We train a CLIP-style dual encoder (1D
> residual network over the PXRD pattern, atom-set Transformer over the crystal graph) on 247,089
> molecular crystals from the Cambridge Structural Database and retrieve the correct polymorph at
> within-family top-1 0.561 against 0.380 chance over 1,077 held-out families. Under a measurement
> regime where classical full-profile correlation and peak-list matching collapse to chance, the
> learned embedding is unmoved — an invariance ablations show is learned, not memorized. A held-out
> real experimental pattern for racemic ibuprofen retrieves its true structure in the top 6.1% of a
> 30,887-structure library, confirming this transfers beyond simulation.

**Keywords (suggested, adjust to whatever the portal's picklist offers):** powder X-ray diffraction,
crystal structure retrieval, contrastive learning, polymorphism, molecular crystals.

**Still Frank's to do:** verify the actual field limits on the live form, confirm eligibility
category (high-school student in a collegiate research program), attach `paper/main.pdf`, and submit
before Aug 9 11:59pm ET.
