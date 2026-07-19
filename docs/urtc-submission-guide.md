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
Drafted + compiled: `paper/main.tex` + `paper/refs.bib` (14 citations verified programmatically) +
`paper/main.pdf` (4 pp). Figures regenerate from committed results via `scripts/10_paper_figs.py`.
**Remaining = Frank hand-reviews and submits** (AI does not write the submission; review only).
