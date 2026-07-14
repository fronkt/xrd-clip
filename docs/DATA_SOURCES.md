# Data sources

## Primary (training) — Cambridge Structural Database

- **What:** filtered molecular-crystal subset, 315,060 structures (see `PLAN.md §2`).
- **Access:** local licensed install at `C:\Users\frank\CCDC`, CSD May-2026 release, `ccdc` 3.7.1,
  1,451,367 entries. Read via the CCDC miniconda interpreter only.
- **Licence discipline:** structures are **not redistributable**. We publish `refcodes/*.txt` and
  configs; a licence holder regenerates everything. `.gitignore` hard-blocks `*.cif` and `data/`.
- **Citation:** cite the CSD (Groom et al., Acta Cryst. B 2016) and deposit the full refcode list
  in the paper SI.

## Sim-to-real test — opXRD (open experimental PXRD)

- **What:** ~92,000 **real experimental** powder patterns, labeled + unlabeled, from high-throughput
  XRD facilities. The largest openly accessible experimental PXRD collection.
- **Access:** Zenodo record 15298026 (also 14254270); Python loader `pip install opxrd`
  (github.com/aimat-lab/opxrd) with PyTorch-tensor export, standardization, plotting; project site
  https://xrd.aimat.science/ ; paper arXiv:2503.05577.
- **Licence:** open, downloadable without restriction (verify exact CC variant on Zenodo before
  redistributing any derived array).
- **Role:** the sim-to-real bridge for both `xrd-clip` and `pxrd-flow`. A model trained on simulated
  CSD patterns and shown to retrieve/condition on *real* opXRD patterns is far more credible than
  one evaluated only on its own simulator's output.
- **CAVEAT (important):** opXRD skews **inorganic / materials-science**; our training data (CSD) is
  **molecular**. Testing molecular-trained encoders on inorganic real patterns confounds two
  distribution shifts at once (sim→real AND molecular→inorganic). Use opXRD's **labeled organic
  subset** for the clean experiment. If that subset is too small, the fallback real-data sources are:
    - **COD** (Crystallography Open Database) — open CIFs incl. organic; a handful carry raw patterns.
    - **Published journal SI** — IUCr / Acta Cryst papers often deposit a measured pattern + structure.
    - **A Purdue instrument** — a few in-house measurements of known compounds would be ideal and
      fully in our control.

## Simulation reference (not used, noted for related work)

- **SIMPOD** — 467,861 COD structures with *simulated* diffractograms (1D + 2D). Same idea as our
  stage-2 output but COD-sourced and inorganic-heavy; cite as related work, not a dependency.
