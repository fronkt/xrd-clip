# Real-pattern test: opXRD organics vs the CSD (library 2167: targets + family members + 2,000 random decoys)

95 real patterns (92 distinct CSD targets) from opXRD CNRS, matched by reduced cell + element set. 85/95 targets are in the model's TRAINING split. 74/95 are powder studies (structure refined against this pattern), the easy matched-temperature case. `model@K->X` = model top-K shortlist reranked by X.

| Method | Variant | top-1 | top-10 | median rank | within-family top-1 (n=31) | top-1, powder-study (n=74) | top-1, other (n=21) |
|---|---|---|---|---|---|---|---|
| model | raw | 0.08 | 0.33 | 54 | 0.55 | 0.09 | 0.05 |
| pearson | raw | 0.75 | 0.81 | 1 | 0.87 | 0.88 | 0.29 |
| cos_zs | raw | 0.68 | 0.73 | 1 | 0.87 | 0.78 | 0.33 |
| wcc_1 | raw | 0.51 | 0.59 | 1 | 0.65 | 0.55 | 0.33 |
| wcc_2 | raw | 0.33 | 0.45 | 45 | 0.58 | 0.35 | 0.24 |
| lnq | raw | 0.64 | 0.71 | 1 | 0.84 | 0.73 | 0.33 |
| model@100->wcc_2 | raw | 0.25 | 0.34 | 46 |  | 0.26 | 0.24 |
| model@100->lnq | raw | 0.42 | 0.46 | 29 |  | 0.45 | 0.33 |
| model@1000->wcc_2 | raw | 0.31 | 0.42 | 36 |  | 0.32 | 0.24 |
| model@1000->lnq | raw | 0.61 | 0.66 | 1 |  | 0.69 | 0.33 |
| model | bkg | 0.09 | 0.27 | 61 | 0.48 | 0.11 | 0.05 |
| pearson | bkg | 0.78 | 0.81 | 1 | 0.94 | 0.92 | 0.29 |
| cos_zs | bkg | 0.81 | 0.86 | 1 | 1.00 | 0.95 | 0.33 |
| wcc_1 | bkg | 0.74 | 0.83 | 1 | 0.87 | 0.84 | 0.38 |
| wcc_2 | bkg | 0.64 | 0.74 | 1 | 0.84 | 0.72 | 0.38 |
| lnq | bkg | 0.81 | 0.85 | 1 | 0.94 | 0.95 | 0.33 |
| model@100->wcc_2 | bkg | 0.45 | 0.57 | 5 |  | 0.49 | 0.33 |
| model@100->lnq | bkg | 0.55 | 0.58 | 1 |  | 0.61 | 0.33 |
| model@1000->wcc_2 | bkg | 0.61 | 0.71 | 1 |  | 0.68 | 0.38 |
| model@1000->lnq | bkg | 0.77 | 0.82 | 1 |  | 0.89 | 0.33 |
