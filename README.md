# XRD-CLIP

**Contrastive PXRD ↔ crystal-structure retrieval for phase identification.**

A dual-encoder model — a 1D encoder over powder diffractograms and a graph encoder over
crystal structures — trained with a CLIP-style contrastive objective so that a measured
powder pattern retrieves its matching structure from a reference library by nearest neighbour
in a shared embedding space. Evaluated on top-k retrieval, robustness to realistic pattern
degradation (peak shift, broadening, preferred orientation, noise, two-phase mixtures), and
zero-shot phase identification against classical full-pattern search-match.

- **Target venue:** IEEE URTC 2026 (technical paper, 5 pp, archival → IEEE Xplore). Submission ~Aug 3, 2026.
- **Status:** scoping. See [`SPEC.md`](SPEC.md) for the full plan and week-by-week schedule.
- **Author:** Frank Cai, Purdue University (ORCID 0009-0003-0041-1459).

## Why this is distinct from `fronkt/pxrd-diff`

`pxrd-diff` is *generative*: it inverts a pattern to a novel structure with an E(3)-equivariant
diffusion model. XRD-CLIP is *discriminative retrieval* over a fixed reference library — a
different claim, different failure modes, and no archival conflict with the pxrd-diff journal
manuscript. The two are complementary: retrieval answers "which known phase is this?", and can
serve as a fast pre-filter or a re-ranker for the generative model.

## Data

Molecular crystals from the **Cambridge Structural Database (CSD)**. Patterns are simulated
from deposited structures with `pymatgen`'s `XRDCalculator` and a realistic instrument model.
**CSD compliance:** we redistribute *no* CIFs. The repo ships refcode lists, simulation
configs, and derived embeddings/features only; anyone with a CSD licence can regenerate the
structures. Cite the CSD and list all refcodes in the paper SI.

## Quick map

| Path | Purpose |
|---|---|
| `SPEC.md` | Full technical scope + week-by-week plan |
| `src/data/` | CSD pull, PXRD simulation, augmentation, splits |
| `src/models/` | Diffraction encoder, structure encoder, contrastive head |
| `src/eval/` | Top-k retrieval, robustness curves, search-match baseline |
| `configs/` | Training + simulation configs |
| `refcodes/` | CSD refcode lists per split (no CIFs) |

## Licence

Code: MIT (see `LICENSE`). Results, refcode lists and other data files written by this project: CC BY 4.0.
The archived data deposit for the article is https://doi.org/10.5281/zenodo.23446290.
