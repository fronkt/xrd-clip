# Real organic PXRD benchmark: 188 opXRD patterns vs 309,080 CSD entries

Ground truth: every CSD entry of the same solid form as the pattern's own opXRD structure (CCDC packing similarity, 15/15 molecules; scripts/18_...). Ranks are exact over the whole library; entries that could not be compared are ignored. top-1 with a 95 % compound-bootstrap CI in brackets.

- **std**: any same-form entry counts. 184 of 188 patterns are the data their own CSD structure was refined from, so this is mostly self-retrieval.
- **single-crystal refs**: same-form powder-study entries are ignored; the answer must be an independent single-crystal determination (n = 59). This is the realistic case.
- **within-family**: rank among the compound's own polymorphs only.
- Excluded: 8 patterns (covers 0.0 deg of the 5-50 deg grid (axis [0.0124, 0.8322]); not identified by packing similarity).

| Method | std top-1 | std top-10 | std median rank | single-crystal top-1 | single-crystal top-10 | single-crystal median rank | within-family top-1 (n) |
|---|---|---|---|---|---|---|---|
| Pearson, fixed grid (URTC baseline) [raw] | 0.76 [0.69, 0.83] | 0.86 | 1 | 0.32 [0.14, 0.47] | 0.36 | 175 | 0.94 (51) |
| Pearson, fixed grid (URTC baseline) [bkg] | 0.82 [0.75, 0.88] | 0.89 | 1 | 0.34 [0.15, 0.48] | 0.36 | 131 | 0.98 (51) |
| cosine + 2θ zero scan (B1, l = 0) [raw] | 0.57 [0.42, 0.74] | 0.68 | 1 | 0.34 [0.14, 0.55] | 0.36 | 3353 | 0.92 (51) |
| cosine + 2θ zero scan (B1, l = 0) [bkg] | 0.72 [0.56, 0.88] | 0.93 | 1 | 0.37 [0.17, 0.58] | 0.46 | 31 | 0.98 (51) |
| de Gelder WCC l = 1° + zero scan [raw] | 0.29 [0.20, 0.42] | 0.42 | 138 | 0.14 [0.05, 0.30] | 0.15 | 11215 | 0.84 (51) |
| de Gelder WCC l = 1° + zero scan [bkg] | 0.68 [0.51, 0.85] | 0.73 | 1 | 0.44 [0.19, 0.71] | 0.49 | 50 | 0.96 (51) |
| de Gelder WCC l = 2° + zero scan [raw] | 0.21 [0.13, 0.31] | 0.24 | 5562 | 0.10 [0.03, 0.24] | 0.12 | 26416 | 0.76 (51) |
| de Gelder WCC l = 2° + zero scan [bkg] | 0.57 [0.42, 0.74] | 0.66 | 1 | 0.42 [0.18, 0.70] | 0.46 | 350 | 0.94 (51) |
| ln q lag + zero scan (B2) [bkg] | 0.72 [0.55, 0.90] | 0.81 | 1 | 0.44 [0.20, 0.71] | 0.53 | 6 | 0.96 (51) |
| xrd-clip embedding (learned) [raw] | 0.01 [0.00, 0.03] | 0.03 | 5246 | 0.00 [0.00, 0.00] | 0.00 | 14506 | 0.73 (51) |
| xrd-clip embedding (learned) [bkg] | 0.01 [0.00, 0.03] | 0.03 | 5988 | 0.00 [0.00, 0.00] | 0.00 | 18725 | 0.71 (51) |

## Hybrid: model shortlist re-ranked by a classical score (bkg variant)

| Pipeline | std top-1 | single-crystal top-1 |
|---|---|---|
| model top-100 → Pearson, fixed grid (URTC baseline) [bkg] | 0.09 [0.04, 0.14] | 0.02 [0.00, 0.06] |
| model top-1000 → Pearson, fixed grid (URTC baseline) [bkg] | 0.24 [0.17, 0.33] | 0.03 [0.00, 0.10] |
| model top-100 → cosine + 2θ zero scan (B1, l = 0) [bkg] | 0.09 [0.04, 0.14] | 0.02 [0.00, 0.06] |
| model top-1000 → cosine + 2θ zero scan (B1, l = 0) [bkg] | 0.25 [0.16, 0.34] | 0.03 [0.00, 0.10] |
| model top-100 → de Gelder WCC l = 1° + zero scan [bkg] | 0.09 [0.04, 0.14] | 0.02 [0.00, 0.06] |
| model top-1000 → de Gelder WCC l = 1° + zero scan [bkg] | 0.21 [0.14, 0.31] | 0.05 [0.00, 0.13] |
| model top-100 → de Gelder WCC l = 2° + zero scan [bkg] | 0.08 [0.04, 0.14] | 0.02 [0.00, 0.06] |
| model top-1000 → de Gelder WCC l = 2° + zero scan [bkg] | 0.19 [0.11, 0.27] | 0.05 [0.00, 0.13] |
| model top-100 → ln q lag + zero scan (B2) [bkg] | 0.09 [0.04, 0.14] | 0.02 [0.00, 0.06] |
| model top-1000 → ln q lag + zero scan (B2) [bkg] | 0.26 [0.17, 0.34] | 0.07 [0.01, 0.17] |

## Library-size curve (std top-1, bkg variant; random decoy subsets, 5 repeats)

| Method | 2,000 decoys | 20,000 decoys | 100,000 decoys | full (309,080) |
|---|---|---|---|---|
| Pearson, fixed grid (URTC baseline) [bkg] | 0.93 | 0.89 | 0.86 | 0.82 |
| cosine + 2θ zero scan (B1, l = 0) [bkg] | 0.93 | 0.90 | 0.86 | 0.72 |
| de Gelder WCC l = 1° + zero scan [bkg] | 0.76 | 0.72 | 0.69 | 0.68 |
| de Gelder WCC l = 2° + zero scan [bkg] | 0.68 | 0.63 | 0.61 | 0.57 |
| ln q lag + zero scan (B2) [bkg] | 0.91 | 0.83 | 0.74 | 0.72 |
| xrd-clip embedding (learned) [bkg] | 0.06 | 0.03 | 0.02 | 0.01 |
