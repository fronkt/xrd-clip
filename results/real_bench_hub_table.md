# Hubness correction: full-library real benchmark (188 patterns vs 309,080 entries)

Background-subtracted queries. Top-1 with 95 % compound-bootstrap CIs. Primary correction (pre-declared): `csls_aug_k10` = CSLS, k = 10, 4000 augmented library probes.

| Method | std, none | std, CSLS | single-crystal, none | single-crystal, CSLS | within-family, none | within-family, CSLS |
|---|---|---|---|---|---|---|
| Pearson, fixed grid | 0.82 [0.75, 0.88] | 0.87 [0.81, 0.91] | 0.34 [0.15, 0.48] | 0.36 [0.16, 0.50] | 0.98 | 0.96 |
| cosine + zero scan | 0.72 [0.56, 0.88] | 0.88 [0.81, 0.94] | 0.37 [0.17, 0.58] | 0.41 [0.19, 0.62] | 0.98 | 0.96 |
| WCC, l = 1 deg | 0.68 [0.51, 0.85] | 0.70 [0.54, 0.86] | 0.44 [0.19, 0.71] | 0.44 [0.19, 0.72] | 0.96 | 0.98 |
| WCC, l = 2 deg | 0.57 [0.42, 0.74] | 0.63 [0.47, 0.78] | 0.42 [0.18, 0.70] | 0.44 [0.19, 0.72] | 0.94 | 0.96 |
| ln q + zero scan | 0.72 [0.55, 0.90] | 0.76 [0.61, 0.91] | 0.44 [0.20, 0.71] | 0.46 [0.21, 0.74] | 0.96 | 0.92 |
| xrd-clip model | 0.01 [0.00, 0.03] | 0.02 [0.00, 0.04] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.71 | 0.73 |

## Compound-weighted view (the per-pattern numbers above mostly measure a few compounds)

Patterns per compound, largest: ZZZUWK 20, ZZZIZM 12, NAPHTA 10, MEHNIY 7, COUMAR 6, LILRIJ 6. Macro = mean over compounds of per-compound top-1. Paired differences vs uncorrected, 95 % compound-bootstrap CIs.

| Method | truth | macro, none | macro, CSLS | macro diff | per-pattern diff | fixed (by compound) | broken |
|---|---|---|---|---|---|---|---|
| Pearson, fixed grid | std (123 cpds) | 0.83 [0.77, 0.89] | 0.85 [0.79, 0.91] | +0.020 [-0.018, +0.060] | +0.048 [+0.000, +0.097] | 11: ZZZUWK 5, ZZZIZM 1, QOJSEP 1 ... | 2: PAWZIA 1, DOCMUE 1 |
| Pearson, fixed grid | sc (30 cpds) | 0.30 [0.15, 0.47] | 0.32 [0.17, 0.49] | +0.017 [+0.000, +0.054] | +0.017 [+0.000, +0.062] | 1: HXACAN 1 | 0:  |
| cosine + zero scan | std (123 cpds) | 0.84 [0.78, 0.90] | 0.88 [0.83, 0.93] | +0.043 [-0.010, +0.096] | +0.154 [+0.013, +0.336] | 34: ZZZUWK 20, NAPHTA 3, LILRIJ 3 ... | 5: MAMGUD 1, PAWZIA 1, HXACAN 1 ... |
| cosine + zero scan | sc (30 cpds) | 0.40 [0.23, 0.58] | 0.41 [0.23, 0.59] | +0.006 [+0.000, +0.017] | +0.034 [+0.000, +0.060] | 2: NAPHTA 1, ZZZIZM 1 | 0:  |
| WCC, l = 1 deg | std (123 cpds) | 0.83 [0.76, 0.89] | 0.84 [0.78, 0.90] | +0.012 [-0.024, +0.049] | +0.021 [-0.013, +0.066] | 6: LILRIJ 3, DOCNAL 1, WACDEN 1 ... | 2: MAMGUD 1, WIXDIS 1 |
| WCC, l = 1 deg | sc (30 cpds) | 0.50 [0.33, 0.68] | 0.52 [0.33, 0.70] | +0.017 [-0.048, +0.098] | +0.000 [-0.054, +0.054] | 1: AZUDOP 1 | 1: DEHKUA 1 |
| WCC, l = 2 deg | std (123 cpds) | 0.71 [0.63, 0.78] | 0.74 [0.66, 0.81] | +0.033 [-0.011, +0.076] | +0.053 [+0.000, +0.118] | 12: MEHNIY 5, LILRIJ 2, DOCNAL 1 ... | 2: TIWTAW 1, VOLROG 1 |
| WCC, l = 2 deg | sc (30 cpds) | 0.47 [0.30, 0.65] | 0.52 [0.34, 0.70] | +0.050 [-0.033, +0.154] | +0.017 [-0.040, +0.088] | 2: AKAGIG 1, CEHQEM 1 | 1: DEHKUA 1 |
| ln q + zero scan | std (123 cpds) | 0.87 [0.82, 0.93] | 0.89 [0.84, 0.94] | +0.017 [-0.015, +0.051] | +0.037 [-0.006, +0.082] | 9: LILRIJ 3, ZZZUWK 3, YASFUV 1 ... | 2: LAQSON 1, HXACAN 1 |
| ln q + zero scan | sc (30 cpds) | 0.49 [0.31, 0.67] | 0.55 [0.37, 0.72] | +0.062 [-0.009, +0.167] | +0.017 [-0.032, +0.093] | 2: TAQBIY 1, AZUDOP 1 | 1: MEHNIY 1 |
| xrd-clip model | std (123 cpds) | 0.02 [0.00, 0.04] | 0.02 [0.00, 0.06] | +0.008 [+0.000, +0.024] | +0.005 [+0.000, +0.018] | 1: QERNOT 1 | 0:  |
| xrd-clip model | sc (30 cpds) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 0:  | 0:  |

## Sensitivity (top-1, std truth, all patterns; not used to pick the primary)

| Correction | Pearson, fixed grid | cosine + zero scan | WCC, l = 1 deg | WCC, l = 2 deg | ln q + zero scan | xrd-clip model |
|---|---|---|---|---|---|---|
| none | 0.82 | 0.72 | 0.68 | 0.57 | 0.72 | 0.01 |
| csls_aug_k10 | 0.87 | 0.88 | 0.70 | 0.63 | 0.76 | 0.02 |
| csls_sim_k10 | 0.84 | 0.79 | 0.70 | 0.63 | 0.72 | 0.02 |
| csls_sim_k1 | 0.81 | 0.78 | 0.69 | 0.59 | 0.72 | 0.02 |
| csls_sim_k5 | 0.84 | 0.78 | 0.70 | 0.62 | 0.72 | 0.02 |
| csls_sim_k20 | 0.85 | 0.79 | 0.71 | 0.62 | 0.73 | 0.02 |
| csls_sim_k50 | 0.85 | 0.81 | 0.71 | 0.63 | 0.73 | 0.01 |
| csls_sim_k100 | 0.84 | 0.81 | 0.71 | 0.62 | 0.73 | 0.01 |
| csls_aug_k1 | 0.85 | 0.85 | 0.70 | 0.61 | 0.76 | 0.01 |
| csls_aug_k5 | 0.87 | 0.87 | 0.70 | 0.62 | 0.76 | 0.01 |
| csls_aug_k20 | 0.87 | 0.88 | 0.70 | 0.63 | 0.76 | 0.02 |
| csls_aug_k50 | 0.84 | 0.89 | 0.70 | 0.62 | 0.75 | 0.02 |
| csls_aug_k100 | 0.84 | 0.89 | 0.70 | 0.63 | 0.74 | 0.02 |
| znorm_sim | 0.77 | 0.68 | 0.60 | 0.36 | 0.68 | 0.00 |
| znorm_aug | 0.79 | 0.89 | 0.60 | 0.36 | 0.71 | 0.00 |
| mp_sim | 0.82 | 0.77 | 0.68 | 0.59 | 0.74 | 0.01 |
| mp_aug | 0.82 | 0.79 | 0.68 | 0.60 | 0.74 | 0.01 |

## Single-crystal split (top-1)

| Correction | Pearson, fixed grid | cosine + zero scan | WCC, l = 1 deg | WCC, l = 2 deg | ln q + zero scan | xrd-clip model |
|---|---|---|---|---|---|---|
| none | 0.34 | 0.37 | 0.44 | 0.42 | 0.44 | 0.00 |
| csls_aug_k10 | 0.36 | 0.41 | 0.44 | 0.44 | 0.46 | 0.00 |
| csls_sim_k10 | 0.32 | 0.36 | 0.46 | 0.42 | 0.42 | 0.00 |
| csls_sim_k1 | 0.31 | 0.36 | 0.46 | 0.41 | 0.42 | 0.00 |
| csls_sim_k5 | 0.34 | 0.36 | 0.46 | 0.42 | 0.44 | 0.00 |
| csls_sim_k20 | 0.36 | 0.36 | 0.42 | 0.42 | 0.42 | 0.00 |
| csls_sim_k50 | 0.34 | 0.34 | 0.42 | 0.44 | 0.44 | 0.00 |
| csls_sim_k100 | 0.34 | 0.36 | 0.44 | 0.44 | 0.44 | 0.00 |
| csls_aug_k1 | 0.36 | 0.39 | 0.44 | 0.42 | 0.46 | 0.00 |
| csls_aug_k5 | 0.36 | 0.39 | 0.44 | 0.44 | 0.47 | 0.00 |
| csls_aug_k20 | 0.34 | 0.41 | 0.42 | 0.44 | 0.46 | 0.00 |
| csls_aug_k50 | 0.34 | 0.41 | 0.42 | 0.44 | 0.44 | 0.00 |
| csls_aug_k100 | 0.34 | 0.41 | 0.44 | 0.44 | 0.44 | 0.00 |
| znorm_sim | 0.29 | 0.36 | 0.41 | 0.36 | 0.42 | 0.00 |
| znorm_aug | 0.36 | 0.44 | 0.41 | 0.32 | 0.42 | 0.00 |
| mp_sim | 0.34 | 0.36 | 0.44 | 0.42 | 0.44 | 0.00 |
| mp_aug | 0.34 | 0.46 | 0.44 | 0.42 | 0.44 | 0.00 |

## Library-size curve (top-1, std truth; mean over 5 random decoy subsets)

| Method | Correction | 2k | 20k | 100k | full |
|---|---|---|---|---|---|
| Pearson, fixed grid | none | 0.93 | 0.89 | 0.86 | 0.82 |
| Pearson, fixed grid | csls_aug_k10 | 0.94 | 0.90 | 0.88 | 0.87 |
| cosine + zero scan | none | 0.93 | 0.90 | 0.86 | 0.72 |
| cosine + zero scan | csls_aug_k10 | 0.94 | 0.92 | 0.90 | 0.88 |
| WCC, l = 1 deg | none | 0.76 | 0.72 | 0.69 | 0.68 |
| WCC, l = 1 deg | csls_aug_k10 | 0.86 | 0.76 | 0.72 | 0.70 |
| WCC, l = 2 deg | none | 0.68 | 0.63 | 0.61 | 0.57 |
| WCC, l = 2 deg | csls_aug_k10 | 0.70 | 0.68 | 0.66 | 0.63 |
| ln q + zero scan | none | 0.91 | 0.83 | 0.74 | 0.72 |
| ln q + zero scan | csls_aug_k10 | 0.90 | 0.88 | 0.84 | 0.76 |
| xrd-clip model | none | 0.06 | 0.03 | 0.02 | 0.01 |
| xrd-clip model | csls_aug_k10 | 0.06 | 0.03 | 0.02 | 0.02 |

## Library-size curve, compound-weighted (macro top-1, std truth; same subsets)

| Method | Correction | 2k | 20k | 100k | full |
|---|---|---|---|---|---|
| Pearson, fixed grid | none | 0.93 | 0.88 | 0.85 | 0.83 |
| Pearson, fixed grid | csls_aug_k10 | 0.93 | 0.89 | 0.86 | 0.85 |
| cosine + zero scan | none | 0.93 | 0.90 | 0.87 | 0.84 |
| cosine + zero scan | csls_aug_k10 | 0.95 | 0.93 | 0.91 | 0.88 |
| WCC, l = 1 deg | none | 0.90 | 0.87 | 0.84 | 0.83 |
| WCC, l = 1 deg | csls_aug_k10 | 0.92 | 0.89 | 0.86 | 0.84 |
| WCC, l = 2 deg | none | 0.83 | 0.77 | 0.74 | 0.71 |
| WCC, l = 2 deg | csls_aug_k10 | 0.84 | 0.81 | 0.78 | 0.74 |
| ln q + zero scan | none | 0.94 | 0.91 | 0.89 | 0.87 |
| ln q + zero scan | csls_aug_k10 | 0.94 | 0.93 | 0.91 | 0.89 |
| xrd-clip model | none | 0.07 | 0.04 | 0.02 | 0.02 |
| xrd-clip model | csls_aug_k10 | 0.07 | 0.04 | 0.03 | 0.02 |

## Hubness (k-occurrence over the probes, N10 = number of probes with the entry in their top 10)

| Method | probes | N10 skew | N10 max | top hubs (refcode, N10, reflections, strongest-peak share) |
|---|---|---|---|---|
| Pearson, fixed grid | sim | 6.7 | 12 | KEMVOO 12 (219, 0.38); EFOFIQ 12 (283, 0.25); LENLOJ 11 (205, 0.28) |
| Pearson, fixed grid | aug | 8.0 | 18 | XAXKIU 18 (922, 0.10); TIXXED 18 (146, 0.20); TIXXED01 18 (146, 0.20) |
| cosine + zero scan | sim | 12.2 | 28 | DOXTES 28 (624, 0.22); QOHPEK 25 (183, 0.40); CIMXOL 25 (113, 0.45) |
| cosine + zero scan | aug | 18.7 | 49 | EZEPOO 49 (402, 0.09); PIXGEK 38 (666, 0.11); KITLEI 36 (237, 0.17) |
| WCC, l = 1 deg | sim | 3.4 | 7 | ITEKUQ 7 (347, 0.18); REXNAK 6 (605, 0.17); CIJVEY 5 (827, 0.15) |
| WCC, l = 1 deg | aug | 3.6 | 7 | GAWQIH 7 (764, 0.09); EMOWEM 7 (935, 0.08); JAMYOM 7 (620, 0.04) |
| WCC, l = 2 deg | sim | 3.3 | 7 | RINGOK 7 (928, 0.04); KAGPOB 6 (452, 0.20); REQZES 6 (1068, 0.05) |
| WCC, l = 2 deg | aug | 3.6 | 8 | ODOHAQ 8 (1139, 0.06); ALOHIU 7 (1434, 0.04); IPOJAZ 7 (500, 0.09) |
| ln q + zero scan | sim | 4.5 | 15 | TACPOE 15 (683, 0.26); NOHQOS 10 (266, 0.13); JUCCES 9 (737, 0.29) |
| ln q + zero scan | aug | 12.7 | 40 | XAXKIU 40 (922, 0.10); UTATIU 32 (568, 0.16); YONZEK 27 (719, 0.07) |
| xrd-clip model | sim | 3.0 | 4 | WIJCUR 4 (356, 0.15); WENKEI 4 (284, 0.10); VUBCOK 4 (357, 0.12) |
| xrd-clip model | aug | 3.0 | 5 | AMEVET 5 (405, 0.08); XOJFOR 5 (174, 0.14); SOVBAK 5 (272, 0.10) |

Most frequent wrong top-1 entries (std truth), with the query compounds whose patterns they win:

- Pearson, fixed grid: none: GUSRIV x4 (1 cpd: ZZZUWK 4), ZZZLWW01 x3 (1 cpd: ZZZUWK 3), GOFWUV x1 (1 cpd: QQQAUJ 1) (of 34 misses); csls_aug_k10: ZZZLWW01 x3 (1 cpd: ZZZUWK 3), GOFWUV x1 (1 cpd: QQQAUJ 1), VIFXUH01 x1 (1 cpd: FEYZER 1) (of 25 misses)
- cosine + zero scan: none: GUSRIV x20 (1 cpd: ZZZUWK 20), QOGHOL x2 (1 cpd: NAPHTA 2), ISECEQ x2 (1 cpd: LILRIJ 2) (of 52 misses); csls_aug_k10: OSECEW x2 (1 cpd: LILRIJ 2), POJSEQ x2 (1 cpd: NAPHTA 2), PEFGIR x1 (1 cpd: MAMGUD 1) (of 23 misses)
- WCC, l = 1 deg: none: ZZZLWW01 x18 (1 cpd: ZZZUWK 18), GAQGUB x4 (1 cpd: NAPHTA 4), BEZPEE x2 (1 cpd: NAPHTA 2) (of 60 misses); csls_aug_k10: ZZZLWW01 x19 (1 cpd: ZZZUWK 19), GAQGUB x3 (1 cpd: NAPHTA 3), BEZPEE x2 (1 cpd: NAPHTA 2) (of 56 misses)
- WCC, l = 2 deg: none: SOSBEK x9 (1 cpd: ZZZUWK 9), MITMOU x9 (1 cpd: ZZZUWK 9), AFASIK x3 (1 cpd: NAPHTA 3) (of 80 misses); csls_aug_k10: CUMCEV x20 (1 cpd: ZZZUWK 20), ZOGSOD x4 (1 cpd: NAPHTA 4), GABGOF x4 (1 cpd: NAPHTA 4) (of 70 misses)
- ln q + zero scan: none: EFOXED x10 (1 cpd: ZZZUWK 10), VOXSIO x5 (1 cpd: ZZZUWK 5), NULXOH x4 (1 cpd: NAPHTA 4) (of 52 misses); csls_aug_k10: CIQBIO x17 (1 cpd: ZZZUWK 17), POJSEQ x5 (1 cpd: NAPHTA 5), QOGHOL x2 (1 cpd: NAPHTA 2) (of 45 misses)
- xrd-clip model: none: CPRDCA x18 (1 cpd: ZZZUWK 18), KOVSOF x10 (1 cpd: ZZZIZM 10), IVETOS x6 (2 cpd: MEHNIY 5, LILRIJ 1) (of 186 misses); csls_aug_k10: CPRDCA x18 (1 cpd: ZZZUWK 18), KOVSOF x10 (1 cpd: ZZZIZM 10), OPERAG x5 (1 cpd: NAPHTA 5) (of 185 misses)

Uncorrected ranks vs scripts/19_...: {"pearson_raw": {"max_abs_diff_vs_prepass": 2.086162567138672e-06, "std_rank_mismatches": 0, "std_top1_changes": 0, "sc_rank_mismatches": 1, "sc_top1_changes": 0}, "pearson_bkg": {"max_abs_diff_vs_prepass": 1.6689300537109375e-06, "std_rank_mismatches": 0, "std_top1_changes": 0, "sc_rank_mismatches": 0, "sc_top1_changes": 0}, "cos_zs_raw": {"max_abs_diff_vs_prepass": 3.039836883544922e-06, "std_rank_mismatches": 5, "std_top1_changes": 0, "sc_rank_mismatches": 5, "sc_top1_changes": 0}, "cos_zs_bkg": {"max_abs_diff_vs_prepass": 2.9802322387695312e-06, "std_rank_mismatches": 0, "std_top1_changes": 0, "sc_rank_mismatches": 1, "sc_top1_changes": 0}, "wcc_1_raw": {"max_abs_diff_vs_prepass": 3.993511199951172e-06, "std_rank_mismatches": 18, "std_top1_changes": 0, "sc_rank_mismatches": 8, "sc_top1_changes": 0}, "wcc_1_bkg": {"max_abs_diff_vs_prepass": 2.8014183044433594e-06, "std_rank_mismatches": 0, "std_top1_changes": 0, "sc_rank_mismatches": 7, "sc_top1_changes": 0}, "wcc_2_raw": {"max_abs_diff_vs_prepass": 4.172325134277344e-06, "std_rank_mismatches": 41, "std_top1_changes": 0, "sc_rank_mismatches": 14, "sc_top1_changes": 0}, "wcc_2_bkg": {"max_abs_diff_vs_prepass": 2.8014183044433594e-06, "std_rank_mismatches": 12, "std_top1_changes": 0, "sc_rank_mismatches": 5, "sc_top1_changes": 0}, "lnq_bkg": {"max_abs_diff_vs_prepass": 2.2649765014648438e-06, "std_rank_mismatches": 1, "std_top1_changes": 0, "sc_rank_mismatches": 0, "sc_top1_changes": 0}, "model_raw": {"max_abs_diff_vs_prepass": 3.5762786865234375e-07, "std_rank_mismatches": 0, "std_top1_changes": 0, "sc_rank_mismatches": 0, "sc_top1_changes": 0}, "model_bkg": {"max_abs_diff_vs_prepass": 3.8743019104003906e-07, "std_rank_mismatches": 0, "std_top1_changes": 0, "sc_rank_mismatches": 0, "sc_top1_changes": 0}}
