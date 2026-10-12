"""Check every number in the Digital Discovery manuscript against results/paper_numbers.json (Step 2, D1).

  python scripts/verify_claims.py [paper/dd/manuscript.md]

Each claim is a text template filled from the numbers file; the filled text must occur verbatim in the
manuscript (whitespace-normalised). If a result changes, its sentence stops matching and the check fails,
so the text cannot silently drift from the data. Exit code = number of failed claims.
"""
from __future__ import annotations

import json
import re
import sys

P = json.load(open("results/paper_numbers.json"))
B, D, G, CASE = P["bench"], P["dataset"], P["g0"], P["case"]
CLS = ["pearson_bkg", "cos_zs_bkg", "wcc_1_bkg", "wcc_2_bkg", "lnq_bkg"]


def f2(x):
    return f"{x:.2f}"


def ci(v):
    return f"[{v[0]:.2f}, {v[1]:.2f}]"


def b(key, td="std", corr="none"):
    return B[f"{corr}|{td}|{key}"]


def rng(vals):
    return f"{min(vals):.2f}–{max(vals):.2f}"


def n(x):
    return f"{int(round(x)):,}"


def macro_rng(td, keys=CLS, corr="none"):
    return rng([b(k, td, corr)["macro_top1"] for k in keys])


def row(name, key):
    s, c = b(key, "std"), b(key, "sc")
    return (f"| {name} | {f2(s['macro_top1'])} {ci(s['macro_top1_ci'])} | {f2(s['macro_top10'])} | {f2(s['top1'])} | "
            f"{n(s['median_rank'])} | {f2(c['macro_top1'])} {ci(c['macro_top1_ci'])} | {f2(c['macro_top10'])} | "
            f"{f2(c['top1'])} | {n(c['median_rank'])} |")


lc = P["library_curve_macro"]
pair = P["paired_csls"]
tex = CASE["analysis"]["context"]["texture"]
refl = CASE["analysis"]["context"]["reflections_5_50"]
own = CASE["analysis"]["context"]["wcc_1_bkg"]
dec_cos, dec_wcc = CASE["analysis"]["decoys"]["cos_zs"], CASE["analysis"]["decoys"]["wcc_1"]
t20 = CASE["top1_of_20"]
ov = P["model_training_overlap"]
hyb = P["hybrid"]
hub = P["hubness"]
wf = P["within_family"]
g0 = {r["condition"]: r for r in G["rows"]}

CLAIMS = [
    # abstract
    f"on 247,089 molecular crystals",
    f"in all {G['n_conditions']} degradation conditions tested, by {abs(G['delta_min_abs']):.2f} to {abs(G['delta_max_abs']):.2f} in within-family top-1",
    f"{D['n_queries']} measured patterns of {D['n_compounds']} organic compounds",
    f"ranked against {D['n_library']:,} simulated CSD entries",
    f"top-1 is {macro_rng('std')} for the classical scores and {f2(b('model_bkg')['macro_top1'])} for the model",
    f"fall to {macro_rng('sc')} ({b('lnq_bkg', 'sc')['n_compounds']} compounds); this is the open problem",
    f"compound-weighted top-1 by +{min(pair[f'std|{k}']['macro_diff'] for k in CLS):.2f} to +{max(pair[f'std|{k}']['macro_diff'] for k in CLS):.2f}, within noise",
    # benchmark
    f"left {D['n_opxrd_organic']} patterns",
    f"{D['n_lab_cu']} were measured with Cu Kα radiation and {D['n_synchrotron_or_other']} at other wavelengths between "
    f"{min(D['wavelengths_non_cu']):.2f} and {max(D['wavelengths_non_cu']):.2f} Å",
    f"In {D['n_h_omitted']} of the {D['n_queries']} refined structures the H atoms are omitted",
    f"For {D['n_wavelength_missing_assumed_cu']} of the {D['n_lab_cu']} opXRD records no wavelength",
    f"Its CNRS contribution holds {D['n_cnrs_total']:,} patterns",
    f"{D['anchor_tiers']['strict']} patterns were anchored in the strict pass and {D['anchor_tiers']['lenient']} in the lenient pass",
    f"the {D['n_corpus']:,} molecular crystals of the xrd-clip corpus",
    f"and {D['n_extra']} same-form or polymorph entries",
    f"{D['n_probe_per_set']:,}\nlibrary entries simulated cleanly and {D['n_probe_per_set']:,} simulated",
    f"{D['n_identified']} of the {D['n_opxrd_organic']} patterns were identified",
    f"**{D['n_queries']} patterns of {D['n_compounds']} compounds**",
    f"({D['n_compounds_single_pattern']} of {D['n_compounds']})",
    "ZZZUWK has 20 (a variable-temperature", "ZZZIZM 12 and NAPHTA 10",
    f"{D['n_own_powder_study']} of the {D['n_queries']} patterns are the very data",
    f"**Single-crystal entry only** ({D['n_sc_patterns']} patterns, {D['n_sc_compounds']} compounds)",
    f"({D['n_within_family_patterns']} patterns of {D['n_within_family_compounds']} compounds)",
    f"The library holds {D['n_library']:,} simulated patterns",
    # methods
    f"took {P['run_facts']['hub_wall_s']} s",
    f"agree with the CPU scores to {P['gpu_check']['max_abs_diff_vs_cpu_prepass'] * 1e6:.0f} × 10⁻⁶",
    f"{P['gpu_check']['rank_mismatches']} rank values differ",
    f"no top-1 change" if P["gpu_check"]["top1_changes"] == 0 else "TOP-1 CHANGED",
    f"1,077 held-out polymorph families with\n2,835 query structures" if (G["n_families"], G["n_queries"]) == (1077, 2835) else "G0 SIZE",
    f"(composition-only chance {f2(G['chance'])})",
    # 4.1
    f"the model scores {G['model_range'][0]:.2f}–{G['model_range'][1]:.2f} within-family top-1 and the best classical score\n"
    f"{min(r['best_top1'] for r in G['rows']):.2f}–{max(r['best_top1'] for r in G['rows']):.2f}",
    f"by {abs(G['delta_min_abs']):.2f} to {abs(G['delta_max_abs']):.2f}, and the 95 % CIs never overlap" if G["n_ci_overlap"] == 0 else "CI OVERLAP",
    f"holds {f2(g0['strain=0.012']['best_top1'])} at 1.2 % isotropic strain",
    f"clean within-family top-1 is {f2(g0['clean']['model'])}",
    f"The smallest\ngap, −{abs(G['delta_min_abs']):.2f}",
    f"leads, {f2(g0['field_thermal_harsh']['best_top1'])} against {f2(g0['field_thermal_harsh']['model'])}",
    # 4.2
    f"first for {round(100 * min(b(k)['macro_top1'] for k in CLS))}–{round(100 * max(b(k)['macro_top1'] for k in CLS))} % of compounds",
    f"ln q scan is highest at {f2(b('lnq_bkg')['macro_top1'])}",
    f"cosine + zero scan ({f2(b('cos_zs_bkg')['macro_top1'])}), Pearson and WCC l = 1° ({f2(b('pearson_bkg')['macro_top1'])})",
    f"WCC l = 2° ({f2(b('wcc_2_bkg')['macro_top1'])})",
    f"Top-10 is {rng([b(k)['macro_top10'] for k in CLS if k != 'wcc_2_bkg'])} for all but WCC l = 2°",
    f"The learned model reaches\n{f2(b('model_bkg')['macro_top1'])}, and {f2(b('model_bkg')['macro_top10'])} at top-10; its median rank for the first correct entry is {n(b('model_bkg')['median_rank'])}",
    row("ln q lag + zero scan", "lnq_bkg"), row("Cosine + zero scan", "cos_zs_bkg"),
    row("Pearson, fixed grid", "pearson_bkg"), row("WCC, l = 1° + zero scan", "wcc_1_bkg"),
    row("WCC, l = 2° + zero scan", "wcc_2_bkg"), row("Learned embedding", "model_bkg"),
    f"drops to {f2(b('pearson_raw')['macro_top1'])} for Pearson, {f2(b('cos_zs_raw')['macro_top1'])} for cosine + zero scan, "
    f"{f2(b('wcc_1_raw')['macro_top1'])} for WCC l = 1° and {f2(b('wcc_2_raw')['macro_top1'])} for WCC l = 2°",
    f"({f2(b('wcc_1_bkg')['macro_top1'] - b('wcc_1_raw')['macro_top1'])} for both widths) than without one "
    f"({f2(b('cos_zs_bkg')['macro_top1'] - b('cos_zs_raw')['macro_top1'])} for" if
    f2(b('wcc_1_bkg')['macro_top1'] - b('wcc_1_raw')['macro_top1']) == f2(b('wcc_2_bkg')['macro_top1'] - b('wcc_2_raw')['macro_top1'])
    else "RAW LOSS DIFFERS",
    f"fixed-grid Pearson leads ({f2(b('pearson_bkg')['top1'])}) and\nthe zero-scan scores trail "
    f"({rng([b(k)['top1'] for k in CLS if k != 'pearson_bkg'])})",
    f"which Pearson ranks first {t20['none|pearson_bkg']} times and every zero-scan score never" if
    all(t20[f"none|{k}"] == 0 for k in CLS if k != "pearson_bkg") else "ZZZUWK ZERO-SCAN",
    f"lose {rng([lc['none|' + k]['2000'][0] - lc['none|' + k]['full'][0] for k in CLS])} in compound-weighted top-1",
    "Pearson " + " → ".join(f2(lc["none|pearson_bkg"][s][0]) for s in ("2000", "full")) + ", cosine + zero scan " +
    " → ".join(f2(lc["none|cos_zs_bkg"][s][0]) for s in ("2000", "full")) + ", WCC l = 1°\n" +
    " → ".join(f2(lc["none|wcc_1_bkg"][s][0]) for s in ("2000", "full")) + ", WCC l = 2° " +
    " → ".join(f2(lc["none|wcc_2_bkg"][s][0]) for s in ("2000", "full")) + " and ln q " +
    " → ".join(f2(lc["none|lnq_bkg"][s][0]) for s in ("2000", "full")),
    f"reach\n{rng([wf[k]['top1'] for k in CLS])} top-1 and the model {f2(wf['model_bkg']['top1'])} ({wf['model_bkg']['n']} patterns)",
    # 4.3
    f"the classical scores fall to {macro_rng('sc')} compound-weighted top-1",
    f"For {round(100 * min(b(k, 'sc')['frac_rank_gt_1000'] for k in CLS))}–{round(100 * max(b(k, 'sc')['frac_rank_gt_1000'] for k in CLS))} % of\nthese patterns",
    f"WCC l = 1° ({f2(b('wcc_1_bkg', 'sc')['macro_top1'])}), ln q ({f2(b('lnq_bkg', 'sc')['macro_top1'])}) and WCC l = 2° "
    f"({f2(b('wcc_2_bkg', 'sc')['macro_top1'])}), against {f2(b('pearson_bkg', 'sc')['macro_top1'])} for fixed-grid Pearson",
    f"has the best median rank, {n(b('lnq_bkg', 'sc')['median_rank'])}",
    # 4.4
    f"probes is {hub['cos_zs_bkg']['N10_skew_aug']:.1f} for cosine + zero scan, {hub['lnq_bkg']['N10_skew_aug']:.1f} for ln q, "
    f"{hub['pearson_bkg']['N10_skew_aug']:.1f} for Pearson and {hub['wcc_1_bkg']['N10_skew_aug']:.1f} for both WCC widths" if
    f"{hub['wcc_1_bkg']['N10_skew_aug']:.1f}" == f"{hub['wcc_2_bkg']['N10_skew_aug']:.1f}" else "N10 WCC DIFFER",
    f"wrong first answer for {hub['cos_zs_bkg']['wrong_top1_none'][0][1]} of its {hub['cos_zs_bkg']['n_miss_none']} misses" if
    hub["cos_zs_bkg"]["wrong_top1_none"][0][0] == "GUSRIV" else "GUSRIV",
    f"has {refl['case_entry']} reflections between 5° and 50° 2θ",
    f"has about {round(refl['benchmark_entries_median'], -1):.0f}; fewer than {1 if refl['benchmark_entries_frac_fewer'] < 0.01 else 'X'} % of those entries have fewer",
    f"five times stronger ({f2(tex['001']['measured_mean'])} of the maximum against {f2(tex['001']['own_entry_mean'])}) and (110) at 20.1° five times weaker "
    f"({f2(tex['110']['measured_mean'])} against\n  {f2(tex['110']['own_entry_mean'])})" if
    round(tex["001"]["measured_mean"] / tex["001"]["own_entry_mean"]) == 5 and round(tex["110"]["own_entry_mean"] / tex["110"]["measured_mean"]) == 5 else "TEXTURE RATIO",
    f"median score of {f2(own['own_score_median_case'])}, against {f2(own['own_score_median_rest'])}",
    f"prefers GUSRIV for {dec_cos['GUSRIV']['beats_truth_at_offset0']} of the 20 patterns (with it, all {dec_cos['GUSRIV']['beats_truth_with_scan']})" if
    dec_cos["GUSRIV"]["beats_truth_with_scan"] == 20 else "GUSRIV SCAN",
    f"(median +{dec_cos['GUSRIV']['best_offset_deg_median']:.2f}°)",
    f"{sum(v['beats_truth_at_offset0'] == 20 for v in dec_wcc.values())} decoys beat the own entry at zero\noffset in all 20 patterns".replace("3 ", "three ", 1),
    f"{sum(v['beats_truth_with_scan'] == 20 for v in dec_cos.values())} decoy\nbeats it in all 20 patterns under cosine + zero scan and "
    f"{sum(v['beats_truth_with_scan'] == 20 for v in dec_wcc.values())} do under WCC l = 1°".replace("1 decoy", "one decoy", 1).replace("and 7 do", "and seven do", 1),
    f"first for {t20['none|pearson_bkg']} of the 20\npatterns (Fig. 4b)",
    f"({t20['none|cos_zs_bkg']} → {t20['csls_aug_k10|cos_zs_bkg']} of 20) and raises it\nfor Pearson ({t20['none|pearson_bkg']} → {t20['csls_aug_k10|pearson_bkg']}) "
    f"and ln q ({t20['none|lnq_bkg']} → {t20['csls_aug_k10|lnq_bkg']})",
    f"scan rises from {f2(b('cos_zs_bkg')['top1'])} to {f2(b('cos_zs_bkg', corr='csls_aug_k10')['top1'])}",
    f"from +{min(pair[f'std|{k}']['macro_diff'] for k in CLS):.2f}\nto +{max(pair[f'std|{k}']['macro_diff'] for k in CLS):.2f}, and no 95 % CI" if
    all(pair[f"std|{k}"]["macro_diff_ci"][0] <= 0 for k in CLS) else "CSLS CI EXCLUDES 0",
    f"(+{min(pair[f'sc|{k}']['macro_diff'] for k in CLS):.2f} to +{max(pair[f'sc|{k}']['macro_diff'] for k in CLS):.2f}, all CIs include zero)" if
    all(pair[f"sc|{k}"]["macro_diff_ci"][0] <= 0 for k in CLS) else "SC CSLS CI",
    f"learned model ({f2(b('model_bkg')['macro_top1'])} →\n{f2(b('model_bkg', corr='csls_aug_k10')['macro_top1'])})",
    # 4.5
    f"median of {n(b('model_bkg')['median_rank'])} of {D['n_library']:,} (Fig. 5a)",
    f"for the {ov['model_bkg_train']['n']} patterns whose own structure was in its training set, the median rank is\n"
    f"{n(ov['model_bkg_train']['median_rank'])} and top-1 is {f2(ov['model_bkg_train']['top1'])}; for the {ov['model_bkg_none']['n']} "
    f"whose structures were not in the corpus, {n(ov['model_bkg_none']['median_rank'])} and {f2(ov['model_bkg_none']['top1'])}",
    f"it reaches {f2(wf['model_bkg']['top1'])} against {rng([wf[k]['top1'] for k in CLS])} for the\nclassical scores",
    f"gives {f2(hyb['model@1000->lnq_bkg']['std']['top1'])}\nper-pattern top-1, against {f2(b('lnq_bkg')['top1'])} for the ln q scan",
    f"every re-ranker gives {rng([v['std']['top1'] for k, v in hyb.items() if k.startswith('model@100->')])}",
    f"the model stops at {f2(g0['clean']['model'])} within families",
    # discussion / conclusions
    f"first for {round(100 * b('cos_zs_bkg')['macro_top1'])}–{round(100 * b('lnq_bkg')['macro_top1'])} % of compounds\nfrom {D['n_library']:,} candidates",
    f"{f2(b('model_bkg')['macro_top1'])} against {macro_rng('std')} compound-weighted top-1",
    f"gain more than {f2(b('pearson_bkg')['macro_top1'] - b('model_bkg')['macro_top1'])[:3]} in top-1",
]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "paper/dd/manuscript.md"
    text = norm(open(path, encoding="utf-8").read())
    bad = [c for c in CLAIMS if norm(c) not in text]
    for c in bad:
        print("MISSING:", norm(c))
    print(f"{len(CLAIMS) - len(bad)}/{len(CLAIMS)} claims found verbatim")
    return len(bad)


if __name__ == "__main__":
    sys.exit(main())
