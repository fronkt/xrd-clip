"""Check every number in the Digital Discovery manuscript against results/paper_numbers.json (Step 2, D1).

  python scripts/verify_claims.py [paper/dd/manuscript.md]

Each claim is a text template filled from the numbers file; the filled text must occur verbatim in the
manuscript (whitespace-normalised). If a result changes, its sentence stops matching and the check fails,
so the text cannot silently drift from the data. A claim whose logical premise fails (e.g. "no CI excludes
zero") is replaced by a marker string that cannot match. Exit code = number of failed claims.
"""
from __future__ import annotations

import json
import re
import sys

P = json.load(open("results/paper_numbers.json"))
B, D, G, CASE, EX, RF = P["bench"], P["dataset"], P["g0"], P["case"], P["extra"], P["run_facts"]
SB = json.load(open("results/strong_baselines_v2.json"))
CLS = ["pearson_bkg", "cos_zs_bkg", "wcc_1_bkg", "wcc_2_bkg", "lnq_bkg"]


def f2(x):
    return f"{x:.2f}".replace("-", "−")


def ci(v):
    return f"[{f2(v[0])}, {f2(v[1])}]"


def b(key, td="std", corr="none"):
    return B[f"{corr}|{td}|{key}"]


def rng(vals, sep="–"):
    return f"{min(vals):.2f}{sep}{max(vals):.2f}"


def med(x):
    return f"{x:,.1f}" if x % 1 else f"{int(x):,}"


def pct(x):
    return f"{round(100 * x)}"


def macro_rng(td, keys=CLS, corr="none"):
    return rng([b(k, td, corr)["macro_top1"] for k in keys])


def need(cond, text):
    """The claim text, if its logical premise holds; otherwise a marker that cannot match."""
    return text if cond else f"PREMISE FAILED: {text[:60]}"


def row(name, key):
    s, c = b(key, "std"), b(key, "sc")
    return (f"| {name} | {f2(c['macro_top1'])} {ci(c['macro_top1_ci'])} | {f2(c['macro_top10'])} | {med(c['median_rank'])} | "
            f"{f2(s['macro_top1'])} {ci(s['macro_top1_ci'])} | {f2(s['macro_top10'])} | {f2(s['top1'])} | {med(s['median_rank'])} |")


def pdiff(td, k):
    v = EX["paired_methods"][f"{td}|{k}-pearson_bkg"]
    return v["diff"], v["ci"]


lc = P["library_curve_macro"]
pair = P["paired_csls"]
ctx = CASE["analysis"]["context"]
tex, refl, own = ctx["texture"], ctx["reflections_5_50"], ctx["wcc_1_bkg"]
dec_cos, dec_wcc = CASE["analysis"]["decoys"]["cos_zs"], CASE["analysis"]["decoys"]["wcc_1"]
t20 = CASE["top1_of_20"]
ov = P["model_training_overlap"]
hyb = P["hybrid"]
hub = P["hubness"]
wf = P["within_family"]
g0d = EX["g0_distinct"]
g0 = {r["condition"]: r for r in G["rows"]}
mp = EX["model_probe_check"]
sg = EX["subgroups"]
words = {1: "one", 3: "three", 7: "seven"}

lnq_sc, wcc1_sc = pdiff("sc", "lnq_bkg"), pdiff("sc", "wcc_1_bkg")
cos_sc, wcc2_sc = pdiff("sc", "cos_zs_bkg"), pdiff("sc", "wcc_2_bkg")
wcc2_std = pdiff("std", "wcc_2_bkg")
others_std = [pdiff("std", k) for k in ("lnq_bkg", "wcc_1_bkg", "cos_zs_bkg")]
beats = g0d["conditions_each_score_beats_model"]
hyb1000 = [v["std"]["top1"] for k, v in hyb.items() if k.startswith("model@1000->")]
hyb100 = [v["std"]["top1"] for k, v in hyb.items() if k.startswith("model@100->")]
csls_std = [pair[f"std|{k}"]["macro_diff"] for k in CLS]
csls_sc = [pair[f"sc|{k}"]["macro_diff"] for k in CLS]
dis, ordd = sg["disordered"], sg["ordered"]
cm = EX["sc_cell_mismatch"]
tg = EX["sc_temperature_gap"]
gap = EX["gap_pearson_vs_cos"]
wo = EX["per_pattern_without_case"]
cpu = EX["cpu_s_per_query_full_library"]

CLAIMS = [
    # ---- abstract
    f"trained on {RF.get('model_train_n', 247089):,} molecular\ncrystals",
    f"on {D['n_queries']} measured\npatterns of {D['n_compounds']} organic compounds",
    f"ranked against {D['n_library']:,} simulated CSD entries",
    f"compound-weighted top-1 of {macro_rng('sc')} ({D['n_sc_compounds']} compounds) and the model {f2(b('model_bkg', 'sc')['macro_top1'])}",
    need(lnq_sc[1][0] > 0 and wcc1_sc[1][0] > 0,
         f"beat fixed-grid Pearson correlation by {f2(min(lnq_sc[0], wcc1_sc[0]))}–{f2(max(lnq_sc[0], wcc1_sc[0]))}"),
    f"the classical scores reach {macro_rng('std')} and the model {f2(b('model_bkg')['macro_top1'])}",
    f"its top-1 is {f2(mp['own_entry_clean|std']['top1'])} and its median rank {med(mp['own_entry_clean|std']['median_rank'])}",
    f"In {g0d['n_conditions']} simulated degradation conditions, each of the\nshift-tolerant classical scores beats it in at least "
    f"{min(beats[s] for s in ('pearson_zs', 'wcc_0.5', 'wcc_1', 'wcc_2', 'lnq'))}",
    need(all(pair[f"std|{k}"]["macro_diff_ci"][0] <= 0 for k in CLS),
         f"compound-weighted top-1 by +{min(csls_std):.2f} to +{max(csls_std):.2f}, which the data cannot distinguish from zero"),
    # ---- introduction
    "82.8 % recall@10 against 34.4 %", "zero-shot test on 773 inorganic opXRD\npatterns",
    f"stress test of the model against shift-tolerant classical scores over {g0d['n_conditions']} degradation\n   conditions",
    # ---- benchmark
    f"contribution holds {D['n_cnrs_total']:,} patterns", f"which left {D['n_opxrd_organic']}",
    f"Of the {D['n_queries']} patterns used in the end (Section 2.2), {EX['wavelength_classes']['Cu']} were\nmeasured with Cu Kα radiation, "
    f"{EX['wavelength_classes']['synchrotron']} with synchrotron radiation",
    f"and {D['n_queries'] - EX['wavelength_classes']['Cu'] - EX['wavelength_classes']['synchrotron']} at Mo, Co or other\nwavelengths",
    "synchrotron radiation ({}–{} Å)".format(*(f"{x:.2f}" for x in (
        min(w for w in D["wavelengths_non_cu"] if w < 1.3 and not 0.70 < w < 0.72),
        max(w for w in D["wavelengths_non_cu"] if w < 1.3 and not 0.70 < w < 0.72)))),
    f"recomputing {RF['pymatgen_recheck_n']} randomly drawn entries with pymatgen\n{RF['pymatgen_recheck_version']} reproduces the stored sticks exactly",
    f"opXRD records no wavelength for {D['n_wavelength_missing_assumed_cu']} of the {D['n_lab_cu']}",
    f"In {D['n_h_omitted']} of the {D['n_queries']} refined\nstructures the H atoms are omitted",
    f"{D['anchor_tiers']['strict']} patterns were\n   anchored in the strict pass and {D['anchor_tiers']['lenient']} in the lenient pass",
    f"({EX['unknown_entries']['total']}\n   entries, affecting {EX['unknown_entries']['patterns_with_any']} patterns)",
    f"{D['n_identified']} of the {D['n_opxrd_organic']} patterns were identified",
    f"**{D['n_queries']} patterns of {D['n_compounds']} compounds**",
    f"({D['n_compounds_single_pattern']} of {D['n_compounds']})",
    "ZZZUWK has 20 (a variable-temperature", "ZZZIZM 12 and NAPHTA 10",
    f"({EX['wrong_top1_audit']['n_cases']} cases over both answer sets)",
    need(len({(c["stem"], c["wrong"]) for c in EX["wrong_top1_audit"]["same_form_cases"]}) == 1 and
         EX["wrong_top1_audit"]["same_form_cases"][0]["wrong"] == "GOFWUV", "One entry,\nGOFWUV, is the same form"),
    f"For {D['n_own_powder_study']} of the {D['n_queries']} patterns, the CSD contains a powder-study entry",
    f"({EX['own_powder_rmsd_lt_0p05']['strict']} strict and {EX['own_powder_rmsd_lt_0p05']['lenient']} lenient anchors)",
    f"**Single-crystal entry only** ({D['n_sc_patterns']} patterns, {D['n_sc_compounds']} compounds)",
    f"**Any same-form entry** ({D['n_queries']} patterns, {D['n_compounds']} compounds)",
    f"({D['n_within_family_patterns']} patterns of {D['n_within_family_compounds']} compounds; chance {f2(EX['within_family_chance'])})",
    f"The library holds {D['n_library']:,} simulated patterns: the {D['n_corpus']:,} molecular crystals",
    f"and {D['n_extra']} same-form or polymorph entries",
    f"the own entry of {dis['n_patterns']} patterns ({dis['n_compounds']} compounds) is\ndisordered",
    # ---- methods
    f"took {RF['hub_wall_s']} s",
    f"agree with the CPU scores to {P['gpu_check']['max_abs_diff_vs_cpu_prepass'] * 1e6:.0f} × 10⁻⁶",
    need(P["gpu_check"]["top1_changes"] == 0, "reproduce the CPU ranks with no top-1 change"),
    f"({RF['model_params_millions']} million parameters)",
    f"trained once (one seed) for {RF['model_epochs']} epochs", f"batch size {RF['model_batch']}",
    f"{D['n_probe_per_set']:,} library entries simulated cleanly and {D['n_probe_per_set']:,}\nsimulated",
    f"the mean N₁₀ is {D['n_probe_per_set'] * 10 / D['n_library']:.2f}",
    need((G["n_families"], G["n_queries"]) == (1077, 2835), "1,077 held-out polymorph families with\n2,835 query structures"),
    f"(composition-only chance {f2(G['chance'])})",
    f"The sweep has {g0d['n_conditions']} distinct\nconditions",
    f"they are the reason it\n  reaches only {f2(SB['conditions']['clean']['lnq']['top1'])} on clean simulated queries",
    # ---- 4.1
    f"Across the {g0d['n_conditions']} conditions the model scores {rng(g0d['model_range'])} within-family top-1 and the best classical score\n"
    f"{rng(g0d['best_range'])}",
    need(G["n_ci_overlap"] == 0, f"leads by {abs(g0d['delta_range'][0]):.2f} to {abs(g0d['delta_range'][1]):.2f}, and the 95 % CIs never overlap"),
    need(all(beats[s] == g0d["n_conditions"] for s in ("wcc_0.5", "wcc_1", "wcc_2", "lnq")),
         f"beat the model with non-overlapping CIs in all {g0d['n_conditions']} conditions, and Pearson + zero scan in {beats['pearson_zs']}"),
    f"holds {f2(g0['strain=0.012']['best_top1'])} at 1.2 % isotropic\nstrain",
    f"its clean within-family top-1 is {f2(g0['clean']['model'])}",
    f"The smallest gap, −{abs(g0d['delta_range'][0]):.2f}",
    f"leads, {f2(g0['field_thermal_harsh']['best_top1'])} against {f2(g0['field_thermal_harsh']['model'])}",
    # ---- 4.2
    f"the classical scores reach {macro_rng('sc')} compound-weighted top-1 (Table 1, Fig. 3a) and the model {f2(b('model_bkg', 'sc')['macro_top1'])}",
    f"For {pct(min(b(k, 'sc')['frac_rank_gt_1000'] for k in CLS))}–{pct(max(b(k, 'sc')['frac_rank_gt_1000'] for k in CLS))} % of these patterns",
    f"the ln q scan beats fixed-grid Pearson by {f2(lnq_sc[0])}\n{ci(lnq_sc[1])} and WCC (l = 1°) by {f2(wcc1_sc[0])} {ci(wcc1_sc[1])}",
    need(cos_sc[1][0] < 0 and wcc2_sc[1][0] < 0,
         f"cosine + zero scan (+{f2(cos_sc[0])} {ci(cos_sc[1])}) and WCC\n(l = 2°, +{f2(wcc2_sc[0])} {ci(wcc2_sc[1])}) are not resolved"),
    f"has the best median rank, {med(b('lnq_bkg', 'sc')['median_rank'])}",
    f"ranks it first for {f2(cm['lnq_top1_small'])} of\npatterns ({cm['n_small']} patterns); where they differ by more, for {f2(cm['lnq_top1_large'])} ({cm['n_large']} patterns)",
    f"differ by a median of {tg['median_dT_K']:.0f} K\nfor the {tg['n']} patterns where both are known",
    row("ln q lag + zero scan", "lnq_bkg"), row("WCC, l = 1° + zero scan", "wcc_1_bkg"), row("WCC, l = 2° + zero scan", "wcc_2_bkg"),
    row("Cosine + zero scan", "cos_zs_bkg"), row("Pearson, fixed grid", "pearson_bkg"), row("Learned embedding", "model_bkg"),
    # ---- 4.3
    f"compound-weighted top-1 of {macro_rng('std')}: the\nln q scan {f2(b('lnq_bkg')['macro_top1'])}, cosine + zero scan {f2(b('cos_zs_bkg')['macro_top1'])}, "
    f"Pearson and WCC (l = 1°) {f2(b('pearson_bkg')['macro_top1'])}, and WCC (l = 2°) {f2(b('wcc_2_bkg')['macro_top1'])}",
    need(wcc2_std[1][1] < 0 and all(ci_[0] < 0 < ci_[1] and abs(d) <= 0.05 for d, ci_ in others_std),
         f"only WCC (l = 2°) differs from fixed-grid Pearson ({f2(wcc2_std[0])} {ci(wcc2_std[1])})"),
    f"Top-10 is {rng([b(k)['macro_top10'] for k in CLS if k != 'wcc_2_bkg'])} for all but WCC (l = 2°)",
    f"The learned model\nreaches {f2(b('model_bkg')['macro_top1'])}, and {f2(b('model_bkg')['macro_top10'])} at top-10",
    f"drops to {f2(b('pearson_raw')['macro_top1'])} for Pearson, {f2(b('cos_zs_raw')['macro_top1'])}\nfor cosine + zero scan, "
    f"{f2(b('wcc_1_raw')['macro_top1'])} for WCC l = 1° and {f2(b('wcc_2_raw')['macro_top1'])} for WCC l = 2°",
    need(f2(b('wcc_1_bkg')['macro_top1'] - b('wcc_1_raw')['macro_top1']) == f2(b('wcc_2_bkg')['macro_top1'] - b('wcc_2_raw')['macro_top1']),
         f"({f2(b('wcc_1_bkg')['macro_top1'] - b('wcc_1_raw')['macro_top1'])} for both widths) than without one "
         f"({f2(b('cos_zs_bkg')['macro_top1'] - b('cos_zs_raw')['macro_top1'])} for cosine + zero scan)"),
    f"fixed-grid Pearson leads ({f2(b('pearson_bkg')['top1'])}) and\nthe zero-scan scores trail ({rng([b(k)['top1'] for k in CLS if k != 'pearson_bkg'])})",
    f"Pearson ranks {gap['pearson_only']} patterns first that cosine + zero scan misses, and\ncosine + zero scan {gap['cos_only']} that Pearson misses; "
    f"{gap['net_case']} of the net difference of {gap['net']} come from one compound, ZZZUWK",
    need(wo["pearson_bkg"]["top1"] > max(wo["cos_zs_bkg"]["top1"], wo["lnq_bkg"]["top1"]) and f2(wo["cos_zs_bkg"]["top1"]) == f2(wo["lnq_bkg"]["top1"]),
         f"Without ZZZUWK, Pearson still leads per pattern\n({f2(wo['pearson_bkg']['top1'])} against {f2(wo['cos_zs_bkg']['top1'])} for cosine + zero scan and the ln q scan)"),
    f"For the {dis['n_patterns']} patterns ({dis['n_compounds']} compounds) whose own entry is disordered, the\nclassical scores reach "
    f"{rng([dis[k]['macro_top1'] for k in CLS])} compound-weighted top-1, against {rng([ordd[k]['macro_top1'] for k in CLS])} for the other {ordd['n_patterns']} patterns",
    f"lose {rng([lc['none|' + k]['2000'][0] - lc['none|' + k]['full'][0] for k in CLS])} in compound-weighted top-1",
    "Pearson " + " → ".join(f2(lc["none|pearson_bkg"][s][0]) for s in ("2000", "full")) + ", cosine + zero scan " +
    " → ".join(f2(lc["none|cos_zs_bkg"][s][0]) for s in ("2000", "full")) + ", WCC l = 1°\n" +
    " → ".join(f2(lc["none|wcc_1_bkg"][s][0]) for s in ("2000", "full")) + ", WCC l = 2° " +
    " → ".join(f2(lc["none|wcc_2_bkg"][s][0]) for s in ("2000", "full")) + " and ln q " +
    " → ".join(f2(lc["none|lnq_bkg"][s][0]) for s in ("2000", "full")),
    need(max(v[1] for k, cv in lc.items() for v in cv.values()) < 0.02, "the s.d. over\nfive draws is below 0.02"),
    f"(chance {f2(EX['within_family_chance'])}), the classical scores\nreach {rng([wf[k]['top1'] for k in CLS])} top-1 and the model {f2(wf['model_bkg']['top1'])} ({wf['model_bkg']['n']} patterns)",
    # ---- 4.4
    f"probes is {hub['cos_zs_bkg']['N10_skew_aug']:.1f} for cosine + zero scan, {hub['lnq_bkg']['N10_skew_aug']:.1f} for ln q, "
    f"{hub['pearson_bkg']['N10_skew_aug']:.1f} for Pearson and {hub['wcc_1_bkg']['N10_skew_aug']:.1f} for both WCC widths",
    need(f"{hub['wcc_1_bkg']['N10_skew_aug']:.1f}" == f"{hub['wcc_2_bkg']['N10_skew_aug']:.1f}" and hub["cos_zs_bkg"]["wrong_top1_none"][0][0] == "GUSRIV",
         f"wrong first answer of cosine + zero scan for {hub['cos_zs_bkg']['wrong_top1_none'][0][1]} of its {hub['cos_zs_bkg']['n_miss_none']} misses"),
    f"has {refl['case_entry']} reflections between 5° and 50° 2θ",
    need(refl["benchmark_entries_frac_fewer"] < 0.01,
         f"has about {round(refl['benchmark_entries_median'], -1):.0f}; fewer than 1 % of those entries have fewer"),
    f"(001) at 12.0° is {f2(tex['001']['measured_mean'])} of the maximum against {f2(tex['001']['own_entry_mean'])} simulated, (101)\n"
    f"  at 18.6° {f2(tex['101']['measured_mean'])} against {f2(tex['101']['own_entry_mean'])}, and (110) at 20.1° {f2(tex['110']['measured_mean'])} against {f2(tex['110']['own_entry_mean'])}",
    f"median score of {f2(own['own_score_median_case'])},\n  against {f2(own['own_score_median_rest'])}",
    need(dec_cos["GUSRIV"]["beats_truth_with_scan"] == 20,
         f"prefers GUSRIV for {dec_cos['GUSRIV']['beats_truth_at_offset0']} of the 20 patterns (with it, all 20)"),
    f"(median +{dec_cos['GUSRIV']['best_offset_deg_median']:.2f}°)",
    f"{words[sum(v['beats_truth_at_offset0'] == 20 for v in dec_wcc.values())]} decoys beat the own entry at zero\noffset in all 20 patterns",
    f"{words[sum(v['beats_truth_with_scan'] == 20 for v in dec_cos.values())]} decoy\nbeats it in all 20 patterns under cosine + zero scan and "
    f"{words[sum(v['beats_truth_with_scan'] == 20 for v in dec_wcc.values())]} do under WCC l = 1°",
    f"first for {t20['none|pearson_bkg']} of the 20 patterns (Fig. 4b)",
    f"({t20['none|cos_zs_bkg']} → {t20['csls_aug_k10|cos_zs_bkg']} of 20) and raises it for Pearson\n({t20['none|pearson_bkg']} → {t20['csls_aug_k10|pearson_bkg']}) "
    f"and ln q ({t20['none|lnq_bkg']} → {t20['csls_aug_k10|lnq_bkg']})",
    f"scan rises from {f2(b('cos_zs_bkg')['top1'])} to {f2(b('cos_zs_bkg', corr='csls_aug_k10')['top1'])}",
    need(all(pair[f"std|{k}"]["macro_diff_ci"][0] <= 0 for k in CLS),
         f"the change is +{min(csls_std):.2f} to +{max(csls_std):.2f} for the classical scores,\nand no 95 % CI of the paired difference excludes zero; "
         f"gains of up to {max(pair[f'std|{k}']['macro_diff_ci'][1] for k in CLS):.2f} are not excluded"),
    f"the change is +{min(csls_sc):.2f} to +{max(csls_sc):.2f}, with upper CI bounds up to {max(pair[f'sc|{k}']['macro_diff_ci'][1] for k in CLS):.2f}",
    f"for the\nlearned model it is {f2(b('model_bkg')['macro_top1'])} → {f2(b('model_bkg', corr='csls_aug_k10')['macro_top1'])}",
    # ---- 4.5
    f"at a median of about {round(b('model_bkg')['median_rank'], -3):,.0f} of {D['n_library']:,} (Fig. 5a)",
    f"ranks\nthe probe's own entry first for {f2(mp['sim']['top1'])} of probes and within the top 10 for {f2(mp['sim']['top10'])} "
    f"(median rank {med(mp['sim']['median_rank'])}); with the\ntraining-time degradations, for {f2(mp['aug']['top1'])} and {f2(mp['aug']['top10'])} (median rank {med(mp['aug']['median_rank'])})",
    f"it reaches top-1 {f2(mp['own_entry_clean|std']['top1'])}\nand top-10 {f2(mp['own_entry_clean|std']['top10'])}, with a median rank of {med(mp['own_entry_clean|std']['median_rank'])}",
    f"On the measured patterns of the same targets it reaches {f2(mp['measured_bkg|std']['top1'])} and\n{f2(mp['measured_bkg|std']['top10'])}",
    f"For the {ov['model_bkg_train']['n']} patterns that have a same-form entry in the\nmodel's training split, the median rank is "
    f"{med(ov['model_bkg_train']['median_rank'])} and top-1 is {f2(ov['model_bkg_train']['top1'])}; for the {ov['model_bkg_none']['n']} patterns whose same-form\n"
    f"entries are all outside the corpus, {med(ov['model_bkg_none']['median_rank'])} and {f2(ov['model_bkg_none']['top1'])} (the other "
    f"{D['n_queries'] - ov['model_bkg_train']['n'] - ov['model_bkg_none']['n']} have entries in the validation or test\nsplit)",
    f"the model reaches {f2(wf['model_bkg']['top1'])} against\n{rng([wf[k]['top1'] for k in CLS])} for the classical scores",
    f"its top 1,000 contains a correct\nentry for only {f2(EX['model_recall']['1000'])} of patterns",
    f"gives {rng(hyb1000)} per-pattern\ntop-1, against {rng([b(k)['top1'] for k in CLS])} for the same scores on the full library",
    # ---- discussion / conclusions
    f"of {rng([b(k)['macro_top1'] for k in ('pearson_bkg', 'lnq_bkg')])} when the pattern's own powder-study entry is in the library, where even\n"
    f"fixed-grid Pearson correlation reaches {f2(b('pearson_bkg')['macro_top1'])}",
    f"searches {D['n_library']:,}\nentries in about {RF['model_query_ms_cpu']} ms per query on a CPU",
    f"about {cpu['pearson']:.1f} s per query for fixed-grid Pearson, about {round(cpu['wcc_1'], -1):.0f} s for a zero scan and about "
    f"{round(cpu['lnq'] / 60):.0f} min for the ln q\nscan",
    f"all five classical scores of 8,188 patterns against the full library took {RF['hub_wall_s']} s",
    f"reaches a compound-weighted top-1 of {f2(b('model_bkg')['macro_top1'])}, against {macro_rng('std')} for classical profile\nmatching; "
    f"with independent single-crystal references, {f2(b('model_bkg', 'sc')['macro_top1'])} against {macro_rng('sc')}",
]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    path = sys.argv[1] if len(sys.argv) > 1 else "paper/dd/manuscript.md"
    text = norm(open(path, encoding="utf-8").read())
    bad = [c for c in CLAIMS if norm(c) not in text]
    for c in bad:
        print("MISSING:", norm(c))
    print(f"{len(CLAIMS) - len(bad)}/{len(CLAIMS)} claims found verbatim")
    return len(bad)


if __name__ == "__main__":
    sys.exit(main())
