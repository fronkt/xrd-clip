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


REV = P["revision"]


def row2(name, key):
    """Table 2: compound-weighted top-1, main CSD | R2 realistic library | R5 COD replica, for sc then std."""
    v = lambda src, td: f2(src["macro"][f"{td}|{key}"]["top1"]["value"])
    cells = [part for td in ("sc", "std") for part in (f2(b(key, td)["macro_top1"]), v(REV["r2"], td), v(REV["r5_cod"], td))]
    return f"| {name} | " + " | ".join(cells) + " |"


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

# ---- revision R1-R5 (draft v2)
C2, c2r = EX["critic2"], REV["r3_critic2"]
c2t1, c2t10, c2p = c2r["critic2_vcgpwdf"]["top1"], c2r["critic2_vcgpwdf"]["top10"], c2r["paired_critic2_minus_lnq"]
lq_sc = b("lnq_bkg", "sc")
GA, GR = REV["r4_gap"]["results"], REV["r4c_realism"]["results"]
R2p, R2a, COD, CODX = REV["r2"]["paired_vs_main"], REV["r2_ablations"], REV["r5_cod"], EX["cod"]


def gm(res, rung, meth="model", td="std"):
    """Effect-ladder entry: (compound-weighted top-1, its CI, median rank, per-pattern top-1)."""
    r = res[f"{rung}|{meth}|{td}"]
    return r["macro_top1"]["value"], r["macro_top1"]["ci"], r["per_pattern"]["median_rank"], r["per_pattern"]["top1"]


def mr(x):            # a median rank as the text gives it (half ranks rounded)
    return f"{round(x):,}"


def sgn(x):           # a signed difference: "+0.04", "−0.04", "0.00"
    return ("+" if round(x, 2) > 0 else "") + f2(x)


def seq(v):           # [2, 3, 7, 9] -> "2, 3, 7 and 9"
    v = [str(x) for x in v]
    return ", ".join(v[:-1]) + " and " + v[-1]


def cod_m(td, key):
    return COD["macro"][f"{td}|{key}"]["top1"]


r2_cls = {td: [R2p[f"{td}|{k}"]["top1"] for k in CLS] for td in ("std", "sc")}
r2_all = r2_cls["std"] + r2_cls["sc"]
r2_resolved = [k for td in ("std", "sc") for k in CLS if R2p[f"{td}|{k}"]["top1"]["ci"][0] > 0 or R2p[f"{td}|{k}"]["top1"]["ci"][1] < 0]
abl_ka2, abl_mask = R2a["r2_noka2"]["paired_vs_r2"]["std|cos_zs_bkg"], R2a["r2_nomask"]["paired_vs_r2"]["std|wcc_2_bkg"]
c2_failed_misses = all(c2r["rank_by_pattern"][f[0]]["lnq_shortlist"] >= C2["k"] for f in c2r["meta"]["failed"])
# referee-pass additions (10-14): ambient-pressure split, causes of cell differences, critic2 control/injection/peaks
AS = EX["sc_ambient_split"]
CD = AS["cells_differ"]
TOL = ["cos_zs_bkg", "wcc_1_bkg", "wcc_2_bkg", "lnq_bkg"]                 # shift- and strain-tolerant scores
AMB_CLS = ["pearson", "cos_zs", "wcc_1", "wcc_2", "lnq"]                 # keys of the ambient split
AMB_TOL = ["cos_zs", "wcc_1", "wcc_2", "lnq", "critic2"]
INJ, CTL, PK = C2["injected_summary"], C2["control_summary"], C2["n_peaks_range"]
X1 = REV["x1_binavg"]["paired_vs_main"]                                  # bin-averaged Cu grid (Frank's X1)


def mci(v):           # a compound-weighted value with its CI
    return f"{f2(v['value'])} {ci(v['ci'])}"


r2_ints = [R2p[f"{td}|{k}"]["top1"]["ci"] for td in ("std", "sc") for k in CLS] + \
          [R2a[t]["paired_vs_r2"][f"{td}|{k}"]["ci"] for t in ("r2_noB", "r2_nolp", "r2_noka2", "r2_nomask")
           for td in ("std", "sc") for k in CLS]
n_r2_int, n_r2_excl = len(r2_ints), sum(1 for c in r2_ints if c[0] > 0 or c[1] < 0)

CLAIMS = [
    # ---- abstract
    f"trained on {RF.get('model_train_n', 247089):,} molecular\ncrystals",
    f"on {D['n_queries']} measured\npatterns of {D['n_compounds']} organic compounds",
    f"ranked against {D['n_library']:,} simulated CSD entries",
    f"compound-weighted top-1 of {macro_rng('sc')} ({D['n_sc_compounds']} compounds) and the model {f2(b('model_bkg', 'sc')['macro_top1'])}",
    need(c2p["top1"]["ci"][0] < 0 < c2p["top1"]["ci"][1],
         f"a variable-cell re-rank (VC-GPWDF)\nreaches {f2(c2t1['value'])}, not resolved from the ln q scan it re-ranks "
         f"({f2(lq_sc['macro_top1'])})"),
    need(gm(GR, "measured")[3] < 0.03,     # the pre-declared criterion (Section 4.5)
         f"stays at {f2(gm(GR, 'measured')[0])} on measured patterns, so\nthe gap lies in what our simulation does not reproduce"),
    f"the classical scores reach {macro_rng('std')} and the model {f2(b('model_bkg')['macro_top1'])}",
    f"Yet on clean simulations of the same targets it reaches {f2(gm(GA, 'own_clean')[0])} (median rank "
    f"{mr(gm(GA, 'own_clean')[2])})",
    need(sum(1 for r in C2["per_pattern"] if r["lnq_rank"] != 1 and (r["pressure_series"] or (r["dT_K"] or 0) > 50))
         > 0.5 * sum(1 for r in C2["per_pattern"] if r["lnq_rank"] != 1),
         "Most failures involve references determined at other pressures or temperatures"),
    f"In {g0d['n_conditions']} simulated degradation conditions, each of the\nshift-tolerant classical scores beats it in at least "
    f"{min(beats[s] for s in ('pearson_zs', 'wcc_0.5', 'wcc_1', 'wcc_2', 'lnq'))}",
    need(all(pair[f"std|{k}"]["macro_diff_ci"][0] <= 0 for k in CLS), "a hubness\ncorrection (CSLS) change none of this"),
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
    f"(within-family top-1 {f2(SB['conditions']['clean']['lnq_v1']['top1'])} rather\nthan {f2(SB['conditions']['clean']['lnq']['top1'])})",
    # ---- 4.1
    f"Across the {g0d['n_conditions']} conditions the model scores {rng(g0d['model_range'])} within-family top-1 and the best classical score\n"
    f"{rng(g0d['best_range'])}",
    need(G["n_ci_overlap"] == 0, f"leads by {abs(g0d['delta_range'][0]):.2f} to {abs(g0d['delta_range'][1]):.2f}, and the 95 % CIs never overlap"),
    need(all(beats[s] == g0d["n_conditions"] for s in ("wcc_0.5", "wcc_1", "wcc_2", "lnq")),
         f"beat the model with non-overlapping CIs in all {g0d['n_conditions']} conditions, and Pearson + zero scan in {beats['pearson_zs']}"),
    need(g0["zero_offset=0.15"]["pearson_fixed"] < G["chance"] + 0.01 and g0["strain=0.008"]["pearson_fixed"] < G["chance"] + 0.01,
         "it falls to chance by a\n0.15° zero offset or a 0.8 % strain"),
    need(all(g0[f"zero_offset={o}"]["pearson_zs"] > 0.995 for o in ("0.1", "0.2", "0.3")),
         "Pearson + zero scan scores 1.00 at every offset that is a multiple of the 0.02° step"),
    need(g0["strain=0.012"]["best"] == "lnq", f"holds {f2(g0['strain=0.012']['best_top1'])} at 1.2 % isotropic\nstrain"),
    need(f2(g0["zero_offset=0.1"]["lnq"]) == f2(g0["zero_offset=0.3"]["lnq"]) and g0["zero_offset=0.2"]["lnq"] > 0.995,
         f"also dips at 0.1° and 0.3° ({f2(g0['zero_offset=0.1']['lnq'])} at both)"),
    f"its clean within-family top-1 is {f2(g0['clean']['model'])}",
    f"The smallest gap, −{abs(g0d['delta_range'][0]):.2f}",
    need(g0d["delta_range"][0] == g0["field_thermal_harsh"]["delta"] and g0["field_thermal_harsh"]["best"] == "wcc_0.5"
         and max(abs(g0["field_thermal_harsh"][s] - g0["field_thermal_harsh"]["best_top1"]) for s in ("wcc_1", "lnq")) <= 0.01,
         f"Even there, WCC (l = 0.5°) leads,\n{f2(g0['field_thermal_harsh']['best_top1'])} against {f2(g0['field_thermal_harsh']['model'])}, "
         f"with WCC (l = 1°) and the ln q scan within 0.01 of it"),
    # ---- 4.2
    f"the classical scores reach {macro_rng('sc')} compound-weighted top-1 (Table 1, Fig. 3a) and the model {f2(b('model_bkg', 'sc')['macro_top1'])}",
    f"Depending on the classical score, the first correct entry lies beyond rank 1,000 for "
    f"{pct(min(b(k, 'sc')['frac_rank_gt_1000'] for k in CLS))}–{pct(max(b(k, 'sc')['frac_rank_gt_1000'] for k in CLS))} % of these\npatterns",
    need(wcc1_sc[1][0] > 0 and lnq_sc[1][0] <= 0 and cos_sc[1][0] <= 0 and wcc2_sc[1][0] <= 0,
         f"WCC (l = 1°) beats\nfixed-grid Pearson by {f2(wcc1_sc[0])} {ci(wcc1_sc[1])}; the ln q scan (+{f2(lnq_sc[0])} {ci(lnq_sc[1])}), "
         f"cosine + zero scan\n(+{f2(cos_sc[0])} {ci(cos_sc[1])}) and WCC (l = 2°, +{f2(wcc2_sc[0])} {ci(wcc2_sc[1])}) are not resolved"),
    f"has the best median rank, {med(b('lnq_bkg', 'sc')['median_rank'])}",
    need(cm["n"] == D["n_sc_patterns"] - 1, f"({cm['n']} patterns; one has no powder-study entry)"),
    need(len(cm["supercell"]) == 1 and cm["supercell"][0]["lnq_rank"] == 1 and cm["lnq_hits_large"] >= 1,
         f"the ln q scan ranks a single-crystal same-form entry first for {cm['lnq_hits_small']} of {cm['n_small']} patterns; "
         f"where the cells differ\nby more, for {cm['lnq_hits_large']} of {cm['n_large']}, one of which is a supercell description of the same lattice"),
    f"{AS['n_pressure_series_patterns']} of the {D['n_sc_patterns']} single-crystal patterns ({AS['n_pressure_series_compounds']} compounds) "
    f"come from variable-pressure\nstudies",
    f"they account for {CD['pressure_series']} of the {CD['n']} patterns with differing cells. In {CD['dT_gt_50_not_pressure']} of the other "
    f"{CD['n'] - CD['pressure_series']}, the\nrecorded temperatures of the two entries differ by more than 50 K",
    f"where both temperatures are\nrecorded ({tg['n']} patterns), they differ by more than 50 K for {tg['n_dT_gt_50']}, and those "
    f"{tg['n_dT_gt_50']} are retrieved less often ({f2(tg['top1_dT_gt_50'])}\nagainst {f2(tg['top1_dT_le_50'])})",
    f"Without them ({AS['n_ambient_patterns']} patterns, {AS['n_ambient_compounds']} compounds; a split defined after an internal review), "
    f"the classical scores\nreach a compound-weighted top-1 of {rng([AS[k]['macro_top1']['value'] for k in AMB_CLS])} and the model "
    f"{f2(AS['model']['macro_top1']['value'])}",
    f"headline; {AS['n_pressure_series_patterns']} of these patterns come from variable-pressure studies (Section 4.2)",
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
    f"it reaches a per-pattern\ntop-1 of {f2(mp['own_entry_clean|std']['top1'])} and top-10 of {f2(mp['own_entry_clean|std']['top10'])}, "
    f"with a median rank of {med(mp['own_entry_clean|std']['median_rank'])}",
    f"On the measured patterns of the same targets it reaches {f2(mp['measured_bkg|std']['top1'])} and\n{f2(mp['measured_bkg|std']['top10'])}",
    f"For the {ov['model_bkg_train']['n']} patterns that have a same-form entry in the\nmodel's training split, the median rank is "
    f"{med(ov['model_bkg_train']['median_rank'])} and top-1 is {f2(ov['model_bkg_train']['top1'])}; for the {ov['model_bkg_none']['n']} patterns whose same-form\n"
    f"entries are all outside the corpus, {med(ov['model_bkg_none']['median_rank'])} and {f2(ov['model_bkg_none']['top1'])} (the other "
    f"{D['n_queries'] - ov['model_bkg_train']['n'] - ov['model_bkg_none']['n']} have entries in the validation or test\nsplit)",
    f"the model reaches {f2(wf['model_bkg']['top1'])} against\n{rng([wf[k]['top1'] for k in CLS])} for the classical scores",
    f"its top 1,000 contains a correct\nentry for only {f2(EX['model_recall']['1000'])} of patterns",
    f"gives {rng(hyb1000)} per-pattern\ntop-1, against {rng([b(k)['top1'] for k in CLS])} for the same scores on the full library",
    # ---- discussion / conclusions
    f"reach a compound-weighted top-1 of {rng([b(k, 'sc')['macro_top1'] for k in TOL] + [c2t1['value']])} "
    f"({rng([AS[k]['macro_top1']['value'] for k in AMB_TOL])} without the pressure series); when the pattern's\nown powder-study entry is in the "
    f"library, they reach up to {f2(max(b(k)['macro_top1'] for k in TOL))}, and even fixed-grid Pearson correlation\nreaches "
    f"{f2(b('pearson_bkg')['macro_top1'])}",
    f"searches {D['n_library']:,}\nentries in about {RF['model_query_ms_cpu']} ms per query on a CPU",
    f"about {cpu['pearson']:.1f} s per query for fixed-grid Pearson, about {round(cpu['wcc_1'], -1):.0f} s for a zero scan and about "
    f"{round(cpu['lnq'] / 60):.0f} min for the ln q\nscan",
    f"all five classical scores of 8,188 patterns against the full library took {RF['hub_wall_s']} s",
    f"reaches a compound-weighted top-1 of {f2(b('model_bkg')['macro_top1'])}, against {macro_rng('std')} for classical profile\nmatching; "
    f"with independent single-crystal references, {f2(b('model_bkg', 'sc')['macro_top1'])} against {macro_rng('sc')}",

    # ==== revision R1-R5 (draft v2) ==========================================================================
    # ---- 2.6 open replica
    need(CODX["n_identified_both"] == D["n_identified"] and CODX["n_identified_cod_only"] == 2,
         f"{CODX['n_identified']} of the {CODX['n_patterns']} patterns are identified in the COD: all "
         f"{CODX['n_identified_both']} identified in the CSD and two more"),
    f"which keeps {CODX['n_selected']:,} entries, and adds {CODX['n_added_candidates']} candidate entries",
    f"Of these {CODX['n_all']:,} entries, {COD['meta']['n_library']:,} gave stick patterns",
    f"setting the {CODX['n_other_compound']:,} such candidates aside",
    f"Same-formula candidates that cannot be compared ({CODX['n_unknown_same_formula']})",
    f"{COD['meta']['n_queries']} patterns of {cod_m('std', 'lnq_bkg')['n_compounds']} compounds are scored. "
    f"{COD['per_pattern']['sc|lnq_bkg']['n']} patterns of {cod_m('sc', 'lnq_bkg')['n_compounds']}\ncompounds have a single-crystal "
    f"same-form entry, against {D['n_sc_patterns']} patterns of {D['n_sc_compounds']} compounds in the CSD",
    # ---- methods 3.3, 3.4, 3.7, 3.8
    need(P["gpu_check"]["lnq_v2_vs_cpu"]["top1_changes"] == 0,
         f"rescored alone ({RF['hub_lnq2_wall_s']} s) and reproduces its CPU ranks with no top-1\nchange"),
    f"(B = {REV['r2']['meta']['B']:.2f} Å², from\nthe library, Section 3.7)",
    f"B = {REV['r2']['meta']['B']:.2f} Å², which is 8π² times",
    f"For the {D['n_queries'] - D['n_lab_cu']} patterns not measured with Cu Kα",
    f"For the {sum(v for k, v in REV['r2']['meta']['tube_counts'].items() if k != 'sync')} laboratory-tube patterns",
    f"For the {REV['r2']['meta']['n_partial_coverage']} patterns that do not cover 5–50°",
    need(C2["n_answer_in_lnq_top100"] == C2["n_answer_in_shortlist"] and C2["k"] == 50,
         f"holds the answer for the same {C2['n_answer_in_shortlist']} of the {C2['n_patterns']}\npatterns at 50 as at 100"),
    f"median of {C2['cpu_min_median']:.1f} CPU-minutes and up to {C2['cpu_h_max']:.1f} CPU-hours",
    need(C2["n_failed"] == 1 and c2_failed_misses,
         f"one of the {C2['n_comparisons']:,} comparisons reached the limit, in a shortlist that does not hold the\nanswer"),
    # ---- 4.2 variable-cell re-rank
    need(c2p["top1"]["diff"] > 0 and c2p["top1"]["ci"][0] < 0 < c2p["top1"]["ci"][1],
         f"top-1 of {f2(c2t1['value'])} {ci(c2t1['ci'])}, against {f2(lq_sc['macro_top1'])} {ci(lq_sc['macro_top1_ci'])} for the "
         f"ln q scan on the same patterns; the paired\ndifference, {sgn(c2p['top1']['diff'])} {ci(c2p['top1']['ci'])}, does not resolve them"),
    need(c2p["top10"]["ci"][0] < 0 < c2p["top10"]["ci"][1] and abs(c2t10["value"] - lq_sc["macro_top10"]) < 0.02,
         f"at top-10 the two are level ({f2(c2t10['value'])} against\n{f2(lq_sc['macro_top10'])})"),
    f"ranks the answer first for {C2['hits']} of the {C2['n_patterns']} patterns and the ln q scan\nfor {C2['lnq_hits']}",
    f"Of\nthe {C2['n_answer_in_shortlist']} patterns whose shortlist holds the answer, the re-rank moves it to first place for "
    f"{C2['gains']} (from ln q ranks\n{seq(C2['gain_lnq_ranks'])}) and moves ln q's correct first choice down for {C2['losses']} "
    f"(to ranks {seq(C2['loss_c2_ranks'])})",
    f"it ranks the answer first for {cm['critic2_hits_large']} of the {cm['n_large']}\npatterns, against {cm['lnq_hits_large']} "
    f"for the ln q scan; where\nthey agree, for {cm['critic2_hits_small']} of {cm['n_small']}, against {cm['lnq_hits_small']}",
    need(INJ["ambient"]["n"] + INJ["pressure_series"]["n"] == C2["n_patterns"] - C2["n_answer_in_shortlist"],
         f"for {C2['n_patterns'] - C2['n_answer_in_shortlist']} of the\n{C2['n_patterns']} patterns the answer is not among the "
         f"{C2['k']} entries re-ranked, and compared directly, only {INJ['ambient']['rank1'] + INJ['pressure_series']['rank1']} of those "
         f"{C2['n_patterns'] - C2['n_answer_in_shortlist']}\nanswers would rank first"),
    f"the fit finds {PK['pressure_series'][0]}–{PK['pressure_series'][1]} peaks (median {PK['pressure_series'][2]:.0f}), against "
    f"{PK['ambient'][0]}–{PK['ambient'][1]} (median {PK['ambient'][2]:.0f}) on the\nothers",
    need(CTL["pressure_series"]["rank1"] <= 0.1 * CTL["pressure_series"]["n"] and CTL["ambient"]["rank1"] >= 0.75 * CTL["ambient"]["n"],
         f"ranks that entry above every wrong entry of the shortlist for {CTL['ambient']['rank1']} of {CTL['ambient']['n']} "
         f"ambient-pressure patterns but for\nonly {CTL['pressure_series']['rank1']} of the {CTL['pressure_series']['n']} pressure-series "
         f"patterns (median rank {med(CTL['pressure_series']['median_rank'])})"),
    f"Without the pressure-series patterns, the re-rank reaches {mci(AS['critic2']['macro_top1'])} against "
    f"{mci(AS['lnq']['macro_top1'])} for\nthe ln q scan (paired {sgn(AS['paired_critic2_minus_lnq']['diff'])} "
    f"{ci(AS['paired_critic2_minus_lnq']['ci'])})",
    f"declared primary before it was scored; it lowers the single-crystal top-1 from "
    f"{f2(EX['lnq_v1']['sc']['v1']['macro_top1']['value'])} to {f2(lq_sc['macro_top1'])}",
    f"| VC-GPWDF re-rank of the ln q top {C2['k']} | {f2(c2t1['value'])} {ci(c2t1['ci'])} | {f2(c2t10['value'])} | "
    f"{med(C2['median_rank'])} | – | – | – | – |",
    # ---- 4.5 effect ladder, realism retrain, pattern to pattern
    f"Clean, it reaches a compound-weighted\ntop-1 of {f2(gm(GA, 'own_clean')[0])} with a median rank of {mr(gm(GA, 'own_clean')[2])}",
    need(gm(GA, "own_dw")[0] < min(gm(GA, r)[0] for r in ("own_mask", "own_width", "own_ka2", "own_bkg", "own_noise", "own_zero"))
         and gm(GA, "own_dw", "pearson")[0] > 0.995,
         f"The largest single cost is the Debye–Waller factor ({f2(gm(GA, 'own_dw')[0])}, median rank\n{mr(gm(GA, 'own_dw')[2])}), "
         f"which fixed-grid Pearson does not notice ({f2(gm(GA, 'own_dw', 'pearson')[0])})"),
    f"costs {f2(gm(GA, 'own_ka2')[0])} (median rank {mr(gm(GA, 'own_ka2')[2])}), and the background with its subtraction\n"
    f"{f2(gm(GA, 'own_bkg')[0])} (median rank {mr(gm(GA, 'own_bkg')[2])})",
    need(all(gm(GA, "own_clean")[0] - gm(GA, r)[0] <= 0.05 for r in ("own_mask", "own_width", "own_noise", "own_zero")),
         "coverage, peak width, noise and the zero offset cost little"),
    f"With all six effects\nthe model reaches {f2(gm(GA, 'own_all')[0])} (median rank {mr(gm(GA, 'own_all')[2])}), and with the "
    f"Debye–Waller factor added {f2(gm(GA, 'own_all_dw')[0])} (median rank {mr(gm(GA, 'own_all_dw')[2])}),\nagainst "
    f"{f2(gm(GA, 'measured')[0])} and about {round(gm(GA, 'measured')[2], -3):,.0f} on the measured patterns",
    need(f2(gm(GA, "own_all_dw_int30", "pearson")[0]) == f2(gm(GA, "measured", "pearson")[0]),
         f"fixed-grid Pearson reaches {f2(gm(GA, 'own_all_dw_int30', 'pearson')[0])}, its value on the measured patterns "
         f"({f2(gm(GA, 'measured', 'pearson')[0])}), and the model {f2(gm(GA, 'own_all_dw_int30')[0])}\nagainst "
         f"{f2(gm(GA, 'measured')[0])}; the difference shows in the model's median rank, about "
         f"{round(gm(GA, 'own_all_dw_int30')[2], -1):,.0f} on those simulations against about\n"
         f"{round(gm(GA, 'measured')[2], -3):,.0f} on the measured patterns"),
    f"({f2(gm(GR, 'own_dw')[0])} alone, median rank {mr(gm(GR, 'own_dw')[2])}; {f2(gm(GR, 'own_all_dw')[0])}\ntogether with all six "
    f"effects, median rank {mr(gm(GR, 'own_all_dw')[2])} against {mr(gm(GA, 'own_all_dw')[2])} before)",
    need(gm(GR, "own_ka2")[1][1] >= gm(GA, "own_ka2")[1][0],
         f"Kα2 alone stays costly ({f2(gm(GR, 'own_ka2')[0])}, against {f2(gm(GA, 'own_ka2')[0])} before,\nwith overlapping intervals)"),
    f"the retrained model reaches a compound-weighted top-1 of\n{f2(gm(GR, 'measured')[0])} {ci(gm(GR, 'measured')[1])}, against "
    f"{f2(gm(GA, 'measured')[0])} {ci(gm(GA, 'measured')[1])} before, and {f2(gm(GR, 'measured', td='sc')[0])} for single-crystal "
    f"references; its median rank\nfalls from about {round(gm(GA, 'measured')[2], -3):,.0f} to about {round(gm(GR, 'measured')[2], -3):,.0f}",
    need(gm(GR, "measured")[3] < 0.05 and gm(GR, "measured")[2] > 1000,
         f"Both predictions failed: the per-pattern top-1 is {f2(gm(GR, 'measured')[3])} and the median rank about "
         f"{round(gm(GR, 'measured')[2], -3):,.0f}"),
    f"held in part ({f2(gm(GR, 'own_all_dw_int30')[0])} with every modelled effect, against {f2(gm(GR, 'own_clean')[0])}\nclean)",
    f"reaches a compound-weighted top-1 of {f2(gm(GA, 'measured', 'p2p')[0])} (median rank about\n"
    f"{round(gm(GA, 'measured', 'p2p')[2], -2):,.0f}), and {f2(gm(GR, 'measured', 'p2p')[0])} (about "
    f"{round(gm(GR, 'measured', 'p2p')[2], -1):,.0f}) with the retrained encoder",
    f"rendered with every modelled effect to their clean versions at {f2(gm(GR, 'own_all_dw_int30', 'p2p')[0])}",
    f"does the measured task at\n{f2(gm(GA, 'measured', 'pearson')[0])}",
    f"Answer set: any same-form entry ({D['n_queries']} patterns, {D['n_compounds']} compounds)",
    # ---- 4.6 library closer to the measurements
    f"the classical scores change by {sgn(min(x['diff'] for x in r2_all))} to {sgn(max(x['diff'] for x in r2_all))} in compound-weighted top-1",
    need(r2_resolved == ["cos_zs_bkg"],
         f"Only one change is resolved: cosine + zero scan rises by\n{f2(R2p['std|cos_zs_bkg']['top1']['diff'])} "
         f"{ci(R2p['std|cos_zs_bkg']['top1']['ci'])} when any same-form entry counts"),
    need(abl_ka2["ci"][1] < 0, f"point to the Kα2 nuisance for that gain ({f2(abl_ka2['diff'])} {ci(abl_ka2['ci'])} without it)"),
    need(n_r2_excl == 3 and n_r2_int == 50, "Three of these 50 intervals exclude zero"),
    need(abl_mask["ci"][0] > 0, f"counts ({sgn(abl_mask['diff'])} {ci(abl_mask['ci'])} without it)"),
    need(all(x["ci"][0] <= 0 <= x["ci"][1] for x in r2_cls["sc"]),
         f"they move by\n{sgn(min(x['diff'] for x in r2_cls['sc']))} to {sgn(max(x['diff'] for x in r2_cls['sc']))}, with every interval including zero"),
    need(REV["r2"]["macro"]["std|model_bkg"]["top1"]["value"] == REV["r2"]["macro"]["sc|model_bkg"]["top1"]["value"],
         f"whose library embeddings are unchanged, scores {f2(REV['r2']['macro']['std|model_bkg']['top1']['value'])} with the package "
         f"on both sets"),
    need(CODX["sc_composition"].get("ZZZUWK") == 20 and CODX["sc_composition"].get("NAPHTA") == 8
         and all(v == 1 for k, v in CODX["sc_composition"].items() if k not in ("ZZZUWK", "NAPHTA"))
         and sum(CODX["sc_composition"].values()) == COD["per_pattern"]["sc|lnq_bkg"]["n"],
         f"20 of its {COD['per_pattern']['sc|lnq_bkg']['n']} patterns are ZZZUWK"),
    f"{AS['n_pressure_series_patterns']} of its {D['n_sc_patterns']} patterns come from variable-pressure studies",
    need(max(X1[f"{td}|{k}"]["n_top1_changed"] for td in ("std", "sc") for k in CLS) <= 1
         and max(abs(X1[f"{td}|model_bkg"]["top1"]["diff"]) for td in ("std", "sc")) < 0.01,
         f"changes the classical\nscores by {sgn(min(X1[f'{td}|{k}']['top1']['diff'] for td in ('std', 'sc') for k in CLS))} to "
         f"{sgn(max(X1[f'{td}|{k}']['top1']['diff'] for td in ('std', 'sc') for k in CLS))} (at most one first place per score and "
         f"answer set) and the model by less than 0.01"),
    # ---- 4.7 open replica + Table 2
    f"Against the {COD['meta']['n_library']:,} COD entries, the {COD['meta']['n_queries']} patterns reproduce the main result",
    f"the classical scores reach a compound-weighted top-1 of {rng([cod_m('std', k)['value'] for k in CLS])} "
    f"({cod_m('std', 'lnq_bkg')['n_compounds']} compounds; ln q\n{f2(cod_m('std', 'lnq_bkg')['value'])} {ci(cod_m('std', 'lnq_bkg')['ci'])}) "
    f"and the model {f2(cod_m('std', 'model_bkg')['value'])} {ci(cod_m('std', 'model_bkg')['ci'])}",
    f"The single-crystal set has {COD['per_pattern']['sc|lnq_bkg']['n']} patterns but only {cod_m('sc', 'lnq_bkg')['n_compounds']}\n"
    f"compounds; there the classical scores reach {rng([cod_m('sc', k)['value'] for k in CLS])} and the model "
    f"{f2(cod_m('sc', 'model_bkg')['value'])}",
    need(0.3 < COD["meta"]["n_library"] / D["n_library"] < 0.4, "The COD library is about a third of the size of the CSD library"),
    f"The single-crystal set has\n{D['n_sc_compounds']} compounds in the CSD and {cod_m('sc', 'lnq_bkg')['n_compounds']} in the COD; "
    f"the any-same-form set has {D['n_compounds']} and {cod_m('std', 'lnq_bkg')['n_compounds']}",
    row2("ln q lag + zero scan", "lnq_bkg"), row2("WCC, l = 1° + zero scan", "wcc_1_bkg"),
    row2("WCC, l = 2° + zero scan", "wcc_2_bkg"), row2("Cosine + zero scan", "cos_zs_bkg"),
    row2("Pearson, fixed grid", "pearson_bkg"), row2("Learned embedding", "model_bkg"),
    # ---- discussion / conclusions
    f"and the\ncorrected ln q scan alone {RF['hub_lnq2_wall_s']} s",
    f"a median of {C2['cpu_min_median']:.1f} CPU-minutes per\ncomparison and about {round(C2['cpu_h_total'], -2):,.0f} CPU-hours for the "
    f"{C2['n_comparisons']:,} comparisons of {C2['n_patterns']} shortlists",
    f"the open replica's single-crystal set has only\n{cod_m('sc', 'lnq_bkg')['n_compounds']} compounds",
    need(c2p["top1"]["ci"][0] < 0, f"and {f2(c2t1['value'])} for a\nvariable-cell re-rank, which is not resolved from the ln q scan it re-ranks"),
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
