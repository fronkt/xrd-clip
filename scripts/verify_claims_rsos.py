"""Check every number in the RSOS manuscript against results/rsos_numbers.json (tasks/todo.md D-rest F5).

  python scripts/verify_claims_rsos.py                         # check paper/rsos/manuscript.md
  python scripts/verify_claims_rsos.py --fill                  # first replace each {{name}} by its value, then check
  python scripts/verify_claims_rsos.py --numbers X.json --md Y.md --out Z.md --fill   # dry runs

Each claim is a text template filled from the numbers file (scripts/48); the filled text must occur verbatim in the
manuscript (whitespace-normalised). If a result changes, its sentence stops matching and the check fails, so the text
cannot drift from the data. A claim whose logical premise fails (e.g. "no interval excludes zero") is replaced by a
marker that cannot match. --fill substitutes the placeholders the draft left for numbers still being computed
(PH below, and the two tables). Exit code = number of failed claims + placeholders left (the Zenodo DOI excepted
until Frank reserves it).
"""
from __future__ import annotations

import argparse
import json
import re
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--numbers", default="results/rsos_numbers.json")
ap.add_argument("--md", default="paper/rsos/manuscript.md")
ap.add_argument("--out", default=None, help="where --fill writes (default: --md itself)")
ap.add_argument("--fill", action="store_true")
ARGS = ap.parse_args()

P = json.load(open(ARGS.numbers))
B, D, G, CASE, EX, RF = P["bench"], P["dataset"], P["g0"], P["case"], P["extra"], P["run_facts"]
AS, MOD, ROB, LAD, C2 = EX["sc_ambient_split"], P["models"], P["robustness"], P["ladder"], EX["critic2"]
M, S1, FIX, ORIG = "d3_sym_s0", "d3_sym_s1", "d3_fix", "poly"
CLS = ["pearson_bkg", "cos_zs_bkg", "wcc_1_bkg", "wcc_2_bkg", "lnq_bkg"]
TOL = ["cos_zs_bkg", "wcc_1_bkg", "wcc_2_bkg", "lnq_bkg"]
AMB_CLS = ["pearson", "cos_zs", "wcc_1", "wcc_2", "lnq"]
AMB_TOL = ["cos_zs", "wcc_1", "wcc_2", "lnq"]
R2META = json.load(open(f"results/d3/rescore_r2_{M}.json"))["meta"]


# ---- formatting (as scripts/verify_claims.py) -----------------------------------------------------------------------
def f2(x):
    return "0.00" if round(x, 2) == 0 else f"{x:.2f}".replace("-", "−")


def f3(x):
    return f"{x:.3f}"


def ci(v):
    return f"[{f2(v[0])}, {f2(v[1])}]"


def rng(vals, sep="–"):
    return f"{min(vals):.2f}{sep}{max(vals):.2f}"


def med(x):
    return f"{x:,.1f}" if x % 1 else f"{int(x):,}"


def mr(x):            # a median rank as the text gives it (half ranks rounded)
    return f"{round(x):,}"


def about(x, nd=-2):  # "about 5,600"
    return f"{round(x, nd):,.0f}"


def pct(x):
    return f"{round(100 * x)}"


def sgn(x):           # a signed difference: "+0.04", "−0.04", "0.00"
    return ("+" if round(x, 2) > 0 else "") + f2(x)


def mci(v):           # a compound-weighted value with its CI
    return f"{f2(v['value'])} {ci(v['ci'])}"


def pdc(v):           # a paired difference with its CI
    return f"{f2(v['diff'])} {ci(v['ci'])}"


def incl0(c):
    return c[0] <= 0 <= c[1]


def need(cond, text):
    """The claim text, if its logical premise holds; otherwise a marker that cannot match."""
    return text if cond else f"PREMISE FAILED: {text[:70]}"


def b(key, td="std", corr="none"):
    return B[f"{corr}|{td}|{key}"]


def gm(x, rung, meth="model", td="std"):
    """Effect-ladder entry of checkpoint x: (compound-weighted top-1, median rank)."""
    r = LAD[x]["results"][f"{rung}|{meth}|{td}"]
    return r["macro_top1"]["value"], r["per_pattern"]["median_rank"]


def words(n):
    return {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
            10: "ten"}.get(n, str(n))


# ---- derived quantities ---------------------------------------------------------------------------------------------
amb_tol = [AS[k]["macro_top1"]["value"] for k in AMB_TOL]
std_cls = [b(k)["macro_top1"] for k in CLS]
CTL, INJ, PK = C2["control_summary"], C2["injected_summary"], C2["n_peaks_range"]
CM, CD = EX["sc_cell_mismatch"], AS["cells_differ"]
amb_in_list = sum(1 for r in C2["per_pattern"] if not r["pressure_series"] and r["lnq_rank"] <= C2["k"])
hub, pair, wf = P["hubness"], P["paired_csls"], P["within_family"]
csls_std = [pair[f"std|{k}"]["macro_diff"] for k in CLS]
csls_sc = [pair[f"sc|{k}"]["macro_diff"] for k in CLS]
mp, ov = EX["model_probe_check"], P["model_training_overlap"]
T2 = ROB["table2"]
lc = P["library_curve_macro"]
G0 = {r["condition"]: r for r in G["rows"]}
SE = P["symops_sweep_effect"]
X3 = EX["x3_dac_refit"]
X3_IN = [r for r in X3["control"] if r["old_rerank_rank"] is not None]
X3_PK_OLD, X3_PK_NEW = [r["old_n_peaks"] for r in X3["control"]], list(X3["n_peaks"].values())
CODR = ROB["cod"]
CODX = CODR["construction"]
KD = ROB["kaduk"]
cpu = EX["cpu_s_per_query_full_library"]
AC = ROB["anchor_checks"]
tex = CASE["analysis"]["context"]["texture"]
refl = CASE["analysis"]["context"]["reflections_5_50"]
TF = CASE["texture_fit"]
RFL = {r["hkl"]: r["ratio_measured_over_sim"] for r in TF["reflections"]}
CASE_T1, CASE_FULL = CASE["top1_of_20"], CASE["top1_of_20_full_occupancy"]
CSLS_EXCL = [(td, k) for td in ("std", "sc") for k in CLS if not incl0(pair[f"{td}|{k}"]["macro_diff_ci"])]


def cod_m(td, key):
    return CODR["macro"][f"{td}|{key}"]["top1"]


def kd_m(td, key):
    return KD["macro"][f"{td}|{key}"]["top1"]


def cnt_rng(fracs):   # compound-weighted values on the 3 single-crystal Kaduk compounds, as counts: "1–2", "2"
    n = KD["summary"]["n_sc_compounds"]
    c = [round(v * n) for v in fracs]
    return str(c[0]) if min(c) == max(c) else f"{min(c)}–{max(c)}"


def seq(v):           # [1, 38, 111] -> "1, 38 and 111"
    v = [f"{x:,}" for x in v]
    return v[0] if len(v) == 1 else ", ".join(v[:-1]) + " and " + v[-1]


ISP, KK = EX["instrument_split"], ROB["kaduk_by_instrument"]
INST_G = ("lab_cu", "sync_ambient", "sync_pressure")


def t2diff(var, ref="main", keys=CLS):
    return [T2[var][k][s] - T2[ref][k][s] for k in keys for s in ("ambient", "std")]


def drange(d):        # "−0.04 to +0.04"; "0.00" when nothing moves
    lo, hi = min(d), max(d)
    return "0.00" if round(lo, 2) == 0 and round(hi, 2) == 0 else f"{sgn(lo)} to {sgn(hi)}"


r2_int = ROB["r2_n_intervals"]
r2_exc = ROB["r2_n_intervals_excluding_zero"]
conv_pp = ROB["conv"]["paired_vs_main"]
pvo = {x: MOD[x]["paired_std_vs_original"] for x in (M, S1, FIX)}
cod_si = CODR["without_si_queries"]
cod_si_change = max(abs(cod_si["macro_top1"][k] - cod_m("std", k.split("|", 1)[1] if "|" in k else k)["value"])
                    for k in cod_si["macro_top1"] if k.startswith("std|"))
pt_tol = AS["paired_among_tolerant"]

# ---- placeholder values (the draft's {{name}}) ---------------------------------------------------------------------
PH = {
    "amb_s1": f2(AS[f"model_{S1}"]["macro_top1"]["value"]),
    "amb_frac_gt1000_range": f"{pct(min(AS[k]['frac_rank_gt_1000'] for k in AMB_CLS))}–"
                             f"{pct(max(AS[k]['frac_rank_gt_1000'] for k in AMB_CLS))} %",
    "amb_model_frac_gt1000": f"{pct(AS['model']['frac_rank_gt_1000'])} %",
    "amb_paired_wcc1_pearson": pdc(AS["paired_wcc_1_minus_pearson"]),
    "amb_paired_lnq_pearson": pdc(AS["paired_lnq_minus_pearson"]),
    "amb_paired_wcc1_coszs": f"{f2(-AS['paired_among_tolerant']['cos_zs-wcc_1']['diff'])} "
                             f"{ci([-AS['paired_among_tolerant']['cos_zs-wcc_1']['ci'][1], -AS['paired_among_tolerant']['cos_zs-wcc_1']['ci'][0]])}",
    "c2_amb": mci(AS["critic2"]["macro_top1"]),
    "c2_amb_lnq": mci(AS["lnq"]["macro_top1"]),
    "c2_amb_paired": f"{sgn(AS['paired_critic2_minus_lnq']['diff'])} {ci(AS['paired_critic2_minus_lnq']['ci'])}",
    "c2_amb_in_list": str(amb_in_list),
    "c2_ctl_amb": f"{CTL['ambient']['rank1']} of the {CTL['ambient']['n']}",
    "cm_n": str(CM["n"]), "cm_small_hits": str(CM["lnq_hits_small"]), "cm_n_small": str(CM["n_small"]),
    "cm_large_hits": str(CM["lnq_hits_large"]), "cm_n_large": str(CM["n_large"]),
    "cm_large_pressure": str(CD["pressure_series"]), "cm_large_dT": str(CD["dT_gt_50_not_pressure"]),
    "hp_range": rng([AS[k]["pressure_series_macro_top1"]["value"] for k in AMB_CLS]),
    "hp_c2": f2(AS["critic2"]["pressure_series_macro_top1"]["value"]),
    "hp_model": f2(AS["model"]["pressure_series_macro_top1"]["value"]),
    "c2_peaks_hp": f"{PK['pressure_series'][0]}–{PK['pressure_series'][1]}", "c2_peaks_hp_med": f"{PK['pressure_series'][2]:.0f}",
    "c2_peaks_amb": f"{PK['ambient'][0]}–{PK['ambient'][1]}", "c2_peaks_amb_med": f"{PK['ambient'][2]:.0f}",
    "c2_ctl_hp": str(CTL["pressure_series"]["rank1"]),
    "std_paired_wcc2_pearson": pdc(EX["paired_methods"]["std|wcc_2_bkg-pearson_bkg"]),
    "s1_std": f2(MOD[S1]["std_macro_top1"]["value"]),
    "raw_pearson": f2(b("pearson_raw")["macro_top1"]), "raw_coszs": f2(b("cos_zs_raw")["macro_top1"]),
    "raw_wcc1": f2(b("wcc_1_raw")["macro_top1"]), "raw_wcc2": f2(b("wcc_2_raw")["macro_top1"]),
    "case_sim_full": f"{P['simulation_check']['pre_occ_groups']['zzzuwk']['median']:.3f}",
    "case_full_pearson": str(CASE_FULL["none|pearson_bkg"]),
    "case_full_hub": CASE["wrong_top1_cos_zs_full_occupancy"][0],
    "case_full_hub_n": str(CASE["wrong_top1_cos_zs_full_occupancy"][1]),
    "case_pearson": str(CASE_T1["none|pearson_bkg"]), "case_wcc2": str(CASE_T1["none|wcc_2_bkg"]),
    "case_r101": f2(RFL["(1, 0, 1)"]), "case_r202": f2(RFL["(2, 0, 2)"]),
    "case_tex0": f2(TF["median_pearson_no_texture"]), "case_tex1": f2(TF["median_pearson_best_texture"]),
    "csls_std_range": rng(csls_std), "csls_sc_range": rng(csls_sc),
    "csls_wcc2_std": f"{f2(pair['std|wcc_2_bkg']['macro_diff'])} {ci(pair['std|wcc_2_bkg']['macro_diff_ci'])}",
    "csls_wcc2_zzzuwk": f"{dict(pair['std|wcc_2_bkg']['fixed'])['ZZZUWK']} of the "
                        f"{sum(n for _, n in pair['std|wcc_2_bkg']['fixed'])}",
    "probe_sim_top1": f"{pct(mp['sim']['top1'])} %", "probe_sim_med": med(mp["sim"]["median_rank"]),
    "probe_aug_top1": f"{pct(mp['aug']['top1'])} %", "probe_aug_med": med(mp["aug"]["median_rank"]),
    "ov_train_med": med(ov["model_bkg_train"]["median_rank"]), "ov_none_med": med(ov["model_bkg_none"]["median_rank"]),
    "paired_vs_orig_range": f"{sgn(min(v['diff'] for v in pvo.values()))} to {sgn(max(v['diff'] for v in pvo.values()))}",
    "model_recall_1000": f"{pct(EX['model_recall']['1000'])} %",
    "hybrid_lnq": f2(P["hybrid"]["model@1000->lnq_bkg"]["std"]["macro_top1"]),
    "conv_range": f"at most {max(abs(conv_pp[f'{td}|{k}']['top1']['diff']) for td in ('std', 'sc') for k in CLS):.2f}",
    "nolam_range": f"{max(abs(x) for x in t2diff('known_wavelength')):.2f}",
    "strict_range": drange(t2diff("strict")),
    "strict_model": f"{f2(T2['strict']['model_bkg']['ambient'])} on the ambient set and {f2(T2['strict']['model_bkg']['std'])} "
                    f"on any same-form entry",
    "ea_change": f"{max(abs(T2['every_anchor'][k]['ambient'] - T2['main'][k]['ambient']) for k in CLS + ['model_bkg']):.2f}",
    "r2_range": drange(t2diff("r2")), "r2_nexcl": str(r2_exc), "r2_n": str(r2_int),
    "dT_range": drange(t2diff("dT", "conv")), "binavg_range": drange(t2diff("binavg")),
    "cod_std_range": rng([cod_m("std", k)["value"] for k in CLS]), "cod_model": f2(cod_m("std", "model_bkg")["value"]),
    "cod_si_change": f"{cod_si_change:.2f}",
    "kaduk_cls_range": rng([kd_m("std", k)["value"] for k in CLS]), "kaduk_model": mci(kd_m("std", "model_bkg")),
    "kaduk_model_med": mr(KD["per_pattern"]["std|model_bkg"]["median_rank"]),
    "kaduk_sc_range": cnt_rng([kd_m("sc", k)["value"] for k in CLS]),
    "kaduk_sc_model": cnt_rng([kd_m("sc", "model_bkg")["value"]]),
    # exploratory instrument / source split (Section 4.6)
    "inst_labcu_med": mr(ISP[M]["lab_cu"]["median_rank"]), "inst_labcu_n": str(ISP["groups"]["lab_cu"]["n"]),
    "inst_syncamb_med": mr(ISP[M]["sync_ambient"]["median_rank"]), "inst_syncamb_n": str(ISP["groups"]["sync_ambient"]["n"]),
    "inst_syncp_med": mr(ISP[M]["sync_pressure"]["median_rank"]), "inst_syncp_n": str(ISP["groups"]["sync_pressure"]["n"]),
    "inst_pearson_range": f"{pct(min(ISP['pearson_bkg'][g]['top1_frac'] for g in INST_G))}–"
                          f"{pct(max(ISP['pearson_bkg'][g]['top1_frac'] for g in INST_G))} %",
    "kad_sync_n": str(KK[M]["synchrotron"]["n"]), "kad_sync_med": mr(KK[M]["synchrotron"]["median_rank"]),
    "kad_sync_top1": str(KK[M]["synchrotron"]["top1"]), "kad_lab_n": str(KK[M]["lab Cu"]["n"]),
    "kad_lab_min": f"{(min(KK[M]['lab Cu']['ranks']) // 100) * 100:,}",
    "kad_sync_train": str(KK["sync_self_entry_split"].get("train", 0)),
    "kad_sync_nontrain_n": words(len(KK["sync_not_train_ranks"])), "kad_sync_nontrain": seq(KK["sync_not_train_ranks"]),
    "c2_cpu_min": f"{C2['cpu_min_median']:.1f}",
    # pooled random-ordering median given each pattern's number of correct entries (scripts/48)
    "random_med": f"{round(EX['random_median_rank_std'], -3):,.0f}",
    "binavg_wcc2_sc": (lambda d: f"{sgn(d['diff'])} {ci(d['ci'])}")(ROB["binavg"]["paired_vs_main"]["sc|wcc_2_bkg"]["top1"]),
    "n_resim": str(P["simulation_check"]["n_resimulated"]), "n_major": str(P["simulation_check"]["n_major_component"]),
    "sim_rand_med": f"{P['simulation_check']['groups']['random']['median']:.3f}",
    "sim_zzz_med": f"{P['simulation_check']['groups']['zzzuwk']['median']:.3f}",
    "sim_min": f2(P["simulation_check"]["min_except_worst"]),
    "len_stoich_ok": str(ROB["label_sensitivity"]["n_lenient_patterns"] - len(AC.get("lenient_stoich_mismatch", []))),
    "gpu_s_per_pattern": f"{RF['hub_wall_s'] / (D['n_queries'] + 2 * D['n_probe_per_set']):.2f}",
    "x3_inj_rank1": str(X3["injected_rank1"]), "x3_inj_n": str(len(X3["injected"])),
    "lib2000_model": f2(lc["none|model_bkg"]["2000"][0]),
    "amb_n_compounds": str(AS["n_ambient_compounds"]), "amb_cp_bound": f2(1 - 0.05 ** (1 / AS["n_ambient_compounds"])),
    "sc_n": str(D["n_sc_patterns"]), "sc_nc": str(D["n_sc_compounds"]),
    "sc_all_range": rng([b(k, "sc")["macro_top1"] for k in CLS]), "sc_all_model": f2(b("model_bkg", "sc")["macro_top1"]),
    "checkpoint_statement": "The trained model checkpoints are derived from CSD structures and are not distributed; the "
                            "learned models' results can be checked against the released ranks.",
}

NAMES = {"lnq_bkg": "ln q lag + zero scan", "wcc_1_bkg": "WCC, l = 1° + zero scan", "wcc_2_bkg": "WCC, l = 2° + zero scan",
         "cos_zs_bkg": "Cosine + zero scan", "pearson_bkg": "Pearson, fixed grid", "model_bkg": "Learned model (seed 0)"}
SPLIT = {"lnq_bkg": "lnq", "wcc_1_bkg": "wcc_1", "wcc_2_bkg": "wcc_2", "cos_zs_bkg": "cos_zs", "pearson_bkg": "pearson",
         "model_bkg": "model"}


def table1() -> str:
    # pipe-table column widths follow the dashes of the separator line (pandoc), so the score names do not wrap
    head = ("| Score | Top-1 [95 % CI] | Top-10 | Median rank | Top-1 [95 % CI] | Top-10 | Median rank |\n"
            "|" + "-" * 26 + "|" + "-" * 17 + "|" + "-" * 7 + "|" + "-" * 9 + "|" + "-" * 17 + "|" + "-" * 7 + "|" + "-" * 9 + "|\n"
            "| | *Ambient single-crystal* | | | *Any same-form entry* | | |")
    rows = []
    for k in ["lnq_bkg", "wcc_1_bkg", "wcc_2_bkg", "cos_zs_bkg", "pearson_bkg"]:
        a, s = AS[SPLIT[k]], b(k)
        rows.append(f"| {NAMES[k]} | {mci(a['macro_top1'])} | {f2(a['macro_top10']['value'])} | {med(a['median_rank'])} | "
                    f"{f2(s['macro_top1'])} {ci(s['macro_top1_ci'])} | {f2(s['macro_top10'])} | {med(s['median_rank'])} |")
    a = AS["critic2"]
    rows.append(f"| VC-GPWDF re-rank of the ln q top {C2['k']} | {mci(a['macro_top1'])} | {f2(a['macro_top10']['value'])} | – | – | – | – |")
    a, s = AS["model"], b("model_bkg")
    rows.append(f"| {NAMES['model_bkg']} | {mci(a['macro_top1'])} | {f2(a['macro_top10']['value'])} | {med(a['median_rank'])} | "
                f"{f2(s['macro_top1'])} {ci(s['macro_top1_ci'])} | {f2(s['macro_top10'])} | {med(s['median_rank'])} |")
    return head + "\n" + "\n".join(rows)


T2_ROWS = [("main", "Main benchmark"), ("conv", "Lorentz–polarisation and Jacobian of the Cu conversion"),
           ("known_wavelength", "Recorded wavelength only"), ("strict", "Strict anchors only"),
           ("every_anchor", "EPIVEF counted as correct (every-anchor comparison)"),
           ("r2", "Library package of Section 3.8"), ("dT", "Library at 295 K, with Lorentz–polarisation and Jacobian"),
           ("binavg", "Measured patterns averaged over each grid step")]
T2_COLS = ["lnq_bkg", "wcc_1_bkg", "wcc_2_bkg", "cos_zs_bkg", "pearson_bkg", "model_bkg"]


def table2() -> str:
    head = ("| Variant | ln q scan | WCC, l = 1° | WCC, l = 2° | Cosine + zero scan | Pearson, fixed grid | Learned model |\n"
            "|" + "-" * 34 + "|" + "|".join("-" * 11 for _ in range(6)) + "|")
    rows = [f"| {lab} | " + " | ".join(f"{f2(T2[v][k]['ambient'])} / {f2(T2[v][k]['std'])}" for k in T2_COLS) + " |"
            for v, lab in T2_ROWS]
    return head + "\n" + "\n".join(rows)


PH["table1"], PH["table2"] = table1(), table2()

# ---- claims ---------------------------------------------------------------------------------------------------------
own_all_dw = gm(M, "own_all_dw")
measured = gm(M, "measured")
p_sh = [AC["lenient_single_flag"].get(k, 0) for k in ("ignore_bond_counts",)]
ea_new = AC["new_same_form"]
wcl = EX["wavelength_classes"]
sync_w = [w for w in D["wavelengths_non_cu"] if w < 1.3 and not 0.70 < w < 0.72]
other_tol = [EX["paired_methods"][f"std|{k}-pearson_bkg"] for k in ("lnq_bkg", "wcc_1_bkg", "cos_zs_bkg")]
zo_off = ["zero_offset=0.05", "zero_offset=0.1", "zero_offset=0.15", "zero_offset=0.3"]
zo_strain_thermal = [G0[c]["model"] for c in G0 if c.split("=")[0] in ("zero_offset", "strain", "thermal_dT")]
bf05, bf2, m05 = G0["march_bfdh=0.5"], G0["march_bfdh=2"], G0["march=0.5"]
bfdh_model = [G0[c]["model"] for c in G0 if c.startswith("march_bfdh")]
th200 = G0["thermal_dT=200"]
dmin = min(G["rows"], key=lambda r: abs(r["delta"]) if r["condition"] != "zero_offset=0.0" else 9)
lc_drop = [lc[f"none|{k}"]["2000"][0] - lc[f"none|{k}"]["full"][0] for k in CLS]

CLAIMS = [
    # ---- abstract
    f"built a benchmark of {D['n_queries']} measured patterns of {D['n_compounds']} organic compounds",
    f"ranked every pattern against {D['n_library']:,} simulated entries",
    f"determined at ambient pressure ({AS['n_ambient_patterns']} patterns of {AS['n_ambient_compounds']} compounds)",
    f"compound-weighted fraction of {rng(amb_tol)}",
    need(AS["model"]["macro_top1"]["value"] == 0, "ranks it first for none of these compounds"),
    f"the classical scores reach {rng(std_cls)} and the model reaches {f2(b('model_bkg')['macro_top1'])}",
    f"are retrieved by the model at a median rank of {mr(own_all_dw[1])}, whereas the measured patterns sit at a median "
    f"rank of about {about(measured[1])}, and the {KD['summary']['n_patterns']} profiles of a small external set at "
    f"{PH['kaduk_model_med']}",
    need(round(AS[f"model_{S1}"]["macro_top1"]["value"] * AS["n_ambient_compounds"]) == 1,
         "ranks it first for none of these compounds, and a second training seed for one"),
    # ---- introduction (literature numbers)
    "82.8 % recall at 10 against 34.4 %", "a zero-shot test on 773 inorganic patterns",
    # ---- 2.1
    f"Its CNRS contribution holds {D['n_cnrs_total']:,} patterns", f"which left {D['n_opxrd_organic']}",
    f"Of the {D['n_queries']} patterns used in the end (Section 2.2), {wcl['Cu']} were measured with Cu Kα radiation, "
    f"{wcl['synchrotron']} with synchrotron radiation ({min(sync_w):.2f}–{max(sync_w):.2f} Å) and "
    f"{D['n_queries'] - wcl['Cu'] - wcl['synchrotron']} at Mo, Co or other wavelengths",
    f"opXRD records no wavelength for {D['n_wavelength_missing_assumed_cu']} of the {D['n_lab_cu']}",
    f"In {D['n_h_omitted']} of the {D['n_queries']} refined structures the H atoms are omitted",
    # ---- 2.2
    f"{D['anchor_tiers']['strict']} patterns were anchored in the strict pass and {D['anchor_tiers']['lenient']} in the lenient pass",
    f"({EX['unknown_entries']['total']} pattern–entry pairs of {EX['unknown_in_library']['distinct_all']} entries over "
    f"{EX['unknown_entries']['patterns_with_any']} patterns; {EX['unknown_in_library']['pairs']} pairs\nof "
    f"{EX['unknown_in_library']['distinct']} entries over {EX['unknown_in_library']['patterns']} patterns concern library entries",
    need(AS['answer_pressure']['ambient_with_pressured_answer'] == 0,
         f"none of the {AS['answer_pressure']['ambient_answers']} answer entries of these patterns carries one"),
    need(D["n_opxrd_organic"] - D["n_identified"] == 7, f"Of the {D['n_opxrd_organic']} patterns, {D['n_identified']} were "
         f"identified; the other seven"),
    need(D["n_identified"] - D["n_queries"] == 1, f"which leaves {D['n_queries']} patterns of {D['n_compounds']} compounds"),
    f"Most compounds have one pattern ({D['n_compounds_single_pattern']} of {D['n_compounds']})",
    need([tuple(x) for x in D["patterns_per_compound_top"][:3]] == [("ZZZUWK", 20), ("ZZZIZM", 12), ("NAPHTA", 10)],
         "ZZZUWK has 20, a variable-temperature series, ZZZIZM 12 and NAPHTA 10"),
    need({(c["stem"], c["wrong"]) for c in EX["wrong_top1_audit"]["same_form_cases"]} == {("QQQAUJ", "GOFWUV")},
         f"In the first audit ({EX['wrong_top1_audit']['n_cases']} cases) one family, GOFWUV, "
         f"matches the anchor of one QQQAUJ pattern 15 of 15"),
    need(not EX["wrong_top1_audit_final"]["same_form_cases"],
         f"the audit repeated on the final scores ({EX['wrong_top1_audit_final']['n_cases']} cases) finds no further same-form entry"),
    need(AC["every_anchor"]["patterns_with_new_same_form"] == 1 and len(ea_new) == 1,
         f"adds one entry ({ea_new[0]['refcode']}, for one pattern), which matches the third anchor 15 of 15 at an RMSD of "
         f"{ea_new[0]['rmsd']:.2f} Å"),
    need(AC["lenient_component_mismatches"] == 0, f"in all {ROB['label_sensitivity']['n_lenient_patterns']} lenient patterns "
         f"the components of the deposited structure and of its anchors contain the same non-H elements"),
    need(sorted(r["id"] for r in AC.get("lenient_stoich_mismatch", [])) == ["pattern_1008.json", "pattern_158.json", "pattern_914.json"],
         f"and in {PH['len_stoich_ok']} they also have the same non-H stoichiometry. In the other three"),
    need(all(r["anchor"].startswith("ZZZUWK") and r["h_omitted"] for r in AC["lenient_rows"]
             if r["single_flag_15"] == ["ignore_bond_counts"])
         and sum(r["single_flag_15"] == ["ignore_bond_counts"] for r in AC["lenient_rows"]) == AC["lenient_single_flag"]["ignore_bond_counts"],
         f"for {AC['lenient_single_flag']['ignore_bond_counts']} of the {ROB['label_sensitivity']['n_lenient_patterns']} patterns, "
         f"all from the H-omitted ZZZUWK series, ignoring bond counts alone is enough"),
    # ---- 2.3
    f"For {D['n_own_powder_study']} of the {D['n_queries']} patterns the CSD contains a powder-study entry",
    f"**Independent single-crystal entries at ambient pressure**, the ambient single-crystal set ({AS['n_ambient_patterns']} patterns, "
    f"{AS['n_ambient_compounds']} compounds)",
    f"**Pressure series** ({AS['n_pressure_series_patterns']} patterns, {AS['n_pressure_series_compounds']} compounds)",
    f"form the single-crystal set ({D['n_sc_patterns']} patterns, {D['n_sc_compounds']} compounds)",
    f"**Any same-form entry** ({D['n_queries']} patterns, {D['n_compounds']} compounds)",
    f"({D['n_within_family_patterns']} patterns of {D['n_within_family_compounds']} compounds; chance {f2(EX['within_family_chance'])}, the "
    f"mean over patterns of the fraction of candidates that are correct)",
    f"except that two families whose same-form sets share an entry are one compound (two such pairs)",
    # ---- 2.4
    f"The library holds {D['n_library']:,} simulated patterns: the {D['n_corpus']:,} molecular crystals",
    f"and {D['n_extra']} same-form or polymorph entries",
    f"recomputing {RF['pymatgen_recheck_n']} randomly drawn entries with pymatgen {RF['pymatgen_recheck_version']} "
    f"reproduces the stored sticks exactly",
    need(P["simulation_check"]["worst"][0][0] == "QQQAUJ07",
         f"our simulation agrees with the CCDC's with a median correlation of {PH['sim_rand_med']} for 100 randomly drawn "
         f"library entries, {PH['sim_zzz_med']} for the 20 ZZZUWK self-entries and at least {PH['sim_min']} for every "
         f"re-simulated entry but one, an other-form decoy"),
    f"({PH['n_resim']} entries, {PH['n_major']} as the major component)",
    f"The self-entry of {EX['own_entry_disordered']['n_patterns']} patterns ({EX['own_entry_disordered']['n_compounds']} "
    f"compounds) is disordered",
    # ---- 2.6
    f"which keeps {CODX['n_selected']:,} entries, and adds {CODX['n_added_candidates']} candidate entries",
    f"Of these {CODX['n_all']:,} entries, {CODR['meta']['n_library']:,} gave stick patterns",
    f"excluding the {CODX['n_other_compound']:,} such candidates",
    f"Same-formula candidates that cannot be compared ({CODX['n_unknown_same_formula']})",
    need(CODX["n_identified_both"] == D["n_identified"] and CODX["n_identified_cod_only"] == 2,
         f"Of the {CODX['n_patterns']} patterns, {CODX['n_identified']} are identified in the COD, all {CODX['n_identified_both']} "
         f"identified in the CSD and two more, and {CODR['meta']['n_queries']} patterns of {cod_m('std', 'lnq_bkg')['n_compounds']} "
         f"compounds are scored"),
    f"Only {CODR['per_pattern']['sc|lnq_bkg']['n']} patterns of {cod_m('sc', 'lnq_bkg')['n_compounds']} compounds have a "
    f"single-crystal same-form entry",
    # ---- 3.1, 3.3, 3.4
    f"for the {D['n_queries'] - D['n_lab_cu']} patterns not measured with Cu Kα it leaves",
    f"against all {D['n_library']:,} entries took {RF['hub_wall_s']} s",
    need(P["gpu_check"]["top1_changes"] == 0,
         f"agree with the CPU scores to {P['gpu_check']['max_abs_diff_vs_cpu_prepass'] * 1e6:.0f} × 10⁻⁶ with no change of any top-1"),
    f"dual encoder with {RF['model_params_millions']} million parameters",
    need(all(MOD[x]["train"]["epochs"] == 60 and MOD[x]["train"]["batch_size"] == 512 and abs(MOD[x]["train"]["lr"] - 3e-4) < 1e-12
             and MOD[x]["train"]["weight_decay"] == 0.05 and MOD[x]["train"]["warmup"] == 3 and MOD[x]["train"]["val_mode"] == "degraded"
             for x in (M, S1, FIX)) and MOD[M]["train"]["symops"] and MOD[S1]["train"]["symops"] and not MOD[FIX]["train"]["symops"],
         "trained for 60 epochs with a symmetric InfoNCE loss [@oord2018cpc; @radford2021clip] (batch size 512, AdamW, learning "
         "rate 3 × 10⁻⁴, weight decay 0.05, 3 warm-up epochs)"),
    f"on {ov['split_sizes']['train']:,} CSD structures",
    f"the best epochs reach {f3(MOD[M]['best_degraded_val_top1'])} (seed 0) and {f3(MOD[S1]['best_degraded_val_top1'])} (seed 1)",
    f"without the operator tokens reaches {f3(MOD[FIX]['best_degraded_val_top1'])}",
    # ---- 3.5, 3.6, 3.7, 3.8
    f"(B = {R2META['B']:.2f} Å², from the library, Section 3.8)",
    f"{D['n_probe_per_set']:,} library entries simulated cleanly and {D['n_probe_per_set']:,} degraded ones",
    f"the mean N₁₀ is {D['n_probe_per_set'] * 10 / D['n_library']:.2f}",
    need((G["n_families"], G["n_queries"]) == (1077, 2835), "1,077 held-out polymorph families with 2,835 query structures"),
    f"(composition-only chance {f2(G['chance'])})",
    f"The sweep has {G['n_conditions']} distinct conditions",
    f"B = {R2META['B']:.2f} Å², which is 8π² times",
    f"For the {D['n_queries'] - D['n_lab_cu']} patterns not measured with Cu Kα, the Lorentz–polarisation factor",
    f"For the {sum(v for k, v in R2META['tube_counts'].items() if k != 'sync')} laboratory-tube patterns",
    f"For the {R2META['n_partial_coverage']} patterns that do not cover 5–50°",
    f"The shortlist of each single-crystal pattern is the {C2['k']} best entries of the ln q scan",
    f"new fits of the {len(X3['control'])} patterns whose measured range begins with zero-intensity points",
    # ---- 4.1
    f"Across the {G['n_conditions']} distinct conditions of the sweep the model reaches a within-family top-1 of "
    f"{rng(G['model_range'])} and the best classical score {rng(G['best_range'])}",
    need(G["n_model_wins"] == 0 and G["n_ci_overlap"] == 0,
         f"The best classical score leads in every condition, by {abs(G['delta_min_abs']):.2f} to {abs(G['delta_max_abs']):.2f}"),
    need(all(G["conditions_each_score_beats_model"][s] == G["n_conditions"] for s in ("wcc_0.5", "wcc_1", "lnq")),
         f"WCC with l = 0.5° or 1° and the ln q scan beat the model with non-overlapping 95 % intervals in all {G['n_conditions']} "
         f"conditions, WCC with l = 2° in {G['conditions_each_score_beats_model']['wcc_2']} and Pearson + zero scan in "
         f"{G['conditions_each_score_beats_model']['pearson_zs']}"),
    need(G0["zero_offset=0.15"]["pearson_fixed"] < G["chance"] and G0["strain=0.008"]["pearson_fixed"] < G["chance"],
         "falls to chance by a 0.15° zero offset or a 0.8 % strain"),
    f"while the model's curves stay between {f2(min(zo_strain_thermal))} and {f2(max(zo_strain_thermal))}",
    f"the model's clean within-family top-1 is {f2(G0['clean']['model'])}",
    need(all(G0[f"zero_offset={o}"]["pearson_zs"] > 0.995 for o in ("0.1", "0.2", "0.3")),
         "Pearson + zero scan scores 1.00 at every offset that is a multiple of its 0.02° step"),
    need(G0["zero_offset=0.2"]["lnq"] > 0.995,
         f"dips to {rng([G0[c]['lnq'] for c in zo_off])} at every offset off that grid (0.05°, 0.1°, 0.15° and 0.3°)"),
    f"holds {f2(G0['strain=0.012']['lnq'])} at 1.2 % isotropic strain",
    f"WCC with l = 1° falls from 1.00 to {f2(bf05['wcc_1'])} for plates with r = 0.5 and to {f2(bf2['wcc_1'])} at r = 2, much as it does along (001) ({f2(m05['wcc_1'])} at r = 0.5)",
    f"while fixed-grid Pearson stays at {f2(bf05['pearson_fixed'])} and Pearson + zero scan at {f2(bf05['pearson_zs'])}",
    f"the model stays at {rng(bfdh_model)} throughout",
    need(min(bf05[s] for s in ("pearson_fixed", "pearson_zs", "wcc_0.5", "wcc_1", "wcc_2", "lnq")) == bf05["wcc_2"] > bf05["model"],
         f"the lowest being WCC with l = 2° at {f2(bf05['wcc_2'])}"),
    need(dmin["condition"] == "thermal_dT=200" and th200["best"] == "wcc_2",
         f"The smallest gap, {abs(th200['delta']):.2f}, comes from the largest anisotropic thermal strain (ΔT = 200 K), where "
         f"WCC with l = 2° reaches {f2(th200['wcc_2'])} and the model {f2(th200['model'])}"),
    # ---- 4.2
    f"reach a compound-weighted top-1 of {rng(amb_tol)} and fixed-grid Pearson {f2(AS['pearson']['macro_top1']['value'])}",
    f"The model reaches {f2(AS['model']['macro_top1']['value'])}, and its second seed {PH['amb_s1']}",
    f"beyond rank 1,000 for {PH['amb_frac_gt1000_range']} of these patterns under the classical scores and for "
    f"{PH['amb_model_frac_gt1000']} under the model",
    need(AS["paired_wcc_1_minus_pearson"]["ci"][0] > 0 and AS["paired_lnq_minus_pearson"]["ci"][0] == 0
         and incl0(AS["paired_vs_pearson"]["cos_zs"]["ci"]) and incl0(AS["paired_vs_pearson"]["wcc_2"]["ci"]),
         f"WCC with l = 1° beats fixed-grid Pearson by {PH['amb_paired_wcc1_pearson']}; the interval of the ln q scan's "
         f"gain reaches zero ({PH['amb_paired_lnq_pearson']}), and cosine + zero scan and WCC with l = 2° are not "
         f"statistically distinguishable from it either"),
    need([k for k, v in pt_tol.items() if not incl0(v["ci"])] == ["cos_zs-wcc_1"],
         f"Among the shift-tolerant scores, only WCC with l = 1° and cosine + zero scan differ ({PH['amb_paired_wcc1_coszs']})"),
    need(incl0(AS["paired_critic2_minus_lnq"]["ci"]),
         f"Re-ranking the ln q top 50 with VC-GPWDF gives {PH['c2_amb']}, against {PH['c2_amb_lnq']} for the ln q scan on the "
         f"same patterns, and the paired difference, {PH['c2_amb_paired']}, does not distinguish them"),
    f"which holds the answer for {amb_in_list} of the {AS['n_ambient_patterns']} patterns",
    f"for {PH['c2_ctl_amb']} ambient patterns that have one",
    need(CM["n"] == D["n_sc_patterns"] - 1, f"over the whole single-crystal set ({CM['n']} patterns with a self-entry)"),
    f"the ln q scan ranks a single-crystal entry first for {CM['lnq_hits_small']} of {CM['n_small']} patterns; where the cells "
    f"differ by more, it does so for {CM['lnq_hits_large']} of {CM['n_large']}",
    f"{CD['pressure_series']} of the {CD['n']} patterns with differing cells come from the pressure series, and in "
    f"{CD['dT_gt_50_not_pressure']} of the others",
    need(AS["model"]["macro_top1"]["ci"] == [0.0, 0.0],
         f"treating the {PH['amb_n_compounds']} compounds as independent trials, a one-sided 95 % Clopper–Pearson upper bound "
         f"on the per-compound probability is {PH['amb_cp_bound']}"),
    f"On the whole single-crystal set of {PH['sc_n']} patterns of {PH['sc_nc']} compounds, which joins the ambient and "
    f"pressure-series patterns, the classical scores reach a compound-weighted top-1 of {PH['sc_all_range']} and the model "
    f"{PH['sc_all_model']}",
    need(X3["injected_rank1"] > X3["injected_rank1_old"],
         f"refitted, the pressure-series patterns rank an injected answer first for {PH['x3_inj_rank1']} of {PH['x3_inj_n']}"),
    need(lc["none|model_bkg"]["2000"][0] < 0.2,
         f"it persists against a library of 2,000 entries (compound-weighted top-1 {PH['lib2000_model']})"),
    table1(),
    # ---- 4.3
    f"The {AS['n_pressure_series_patterns']} pressure-series patterns of {AS['n_pressure_series_compounds']} compounds are hard",
    f"compound-weighted top-1 of {PH['hp_range']} and the variable-cell re-rank {PH['hp_c2']}, with the model at {PH['hp_model']}",
    f"the automated fit finds {PH['c2_peaks_hp']} peaks (median {PH['c2_peaks_hp_med']}), against {PH['c2_peaks_amb']} "
    f"(median {PH['c2_peaks_amb_med']}) on the ambient patterns",
    need(CTL["pressure_series"]["n"] == AS["n_pressure_series_patterns"],
         f"ranks the self-entry first for only {CTL['pressure_series']['rank1']} of the {CTL['pressure_series']['n']}"),
    need(not X3["gains"] and not X3["losses"] and len(X3_IN) == 1,
         f"New fits of the {len(X3['control'])} patterns whose measured range begins with zero-intensity points find "
         f"{min(X3_PK_NEW)}–{max(X3_PK_NEW)} peaks instead of {min(X3_PK_OLD)}–{max(X3_PK_OLD)} and rank the self-entry first "
         f"for {X3['control_rank1']} of them instead of {X3['control_rank1_old']}, but they change no first place, because only "
         f"{words(len(X3_IN))} of those {words(len(X3['control']))} shortlists holds the answer"),
    # ---- 4.4
    f"the classical scores reach a compound-weighted top-1 of {rng(std_cls)} (table 1): the ln q scan "
    f"{f2(b('lnq_bkg')['macro_top1'])}, cosine + zero scan {f2(b('cos_zs_bkg')['macro_top1'])}, fixed-grid Pearson "
    f"{f2(b('pearson_bkg')['macro_top1'])}, WCC with l = 1° {f2(b('wcc_1_bkg')['macro_top1'])} and with l = 2° "
    f"{f2(b('wcc_2_bkg')['macro_top1'])}",
    need(EX["paired_methods"]["std|wcc_2_bkg-pearson_bkg"]["ci"][1] < 0 and all(incl0(v["ci"]) for v in other_tol),
         f"only WCC with l = 2° differs from Pearson ({PH['std_paired_wcc2_pearson']})"),
    f"The model reaches {f2(b('model_bkg')['macro_top1'])}, with a median rank of about {about(b('model_bkg')['median_rank'])}, "
    f"far from the median of about {PH['random_med']} that a random ordering would give for these patterns' numbers of correct entries, and its second seed reaches {PH['s1_std']}",
    f"(chance {f2(EX['within_family_chance'])}), the classical scores reach {rng([wf[k]['top1'] for k in CLS])} top-1 and "
    f"the model {f2(wf['model_bkg']['top1'])} ({wf['model_bkg']['n']} patterns)",
    f"drops to {PH['raw_pearson']} for Pearson, {PH['raw_coszs']} for cosine + zero scan, {PH['raw_wcc1']} for WCC with l = 1° "
    f"and {PH['raw_wcc2']} for l = 2°",
    need(min(b(k)["macro_top1"] - b(k.replace("_bkg", "_raw"))["macro_top1"] for k in ("wcc_1_bkg", "wcc_2_bkg"))
         > b("cos_zs_bkg")["macro_top1"] - b("cos_zs_raw")["macro_top1"],
         "The loss is larger with a triangle weight than without one"),
    f"All classical scores also lose {rng(lc_drop)} in compound-weighted top-1 between a 2,000-entry library and the full one",
    need(max(v[1] for k, cv in lc.items() for v in cv.values()) < 0.02, "the standard deviation over five draws is below 0.02"),
    # ---- 4.5
    f"agree with the CCDC's simulation at a median correlation of only {PH['case_sim_full']}",
    need(CASE_FULL["none|cos_zs_bkg"] == CASE_FULL["none|wcc_1_bkg"] == CASE_FULL["none|wcc_2_bkg"] == 0,
         "Cosine + zero scan and both WCC widths then rank a same-form entry first for none of the 20 patterns"),
    need(CASE["wrong_top1_cos_zs_full_occupancy"][1] == CASE["n_patterns"]
         and CASE["wrong_top1_cos_zs_full_occupancy"][3] == [["ZZZUWK", CASE["n_patterns"]]]
         and not CASE["wrong_top1_cos_zs_full_occupancy"][0].startswith("ZZZUWK"),
         f"one unrelated entry, {PH['case_full_hub']}, is the first answer of cosine + zero scan for all {PH['case_full_hub_n']}"),
    need(CASE_T1["none|cos_zs_bkg"] == CASE_T1["none|wcc_1_bkg"] == CASE_T1["none|lnq_bkg"] == CASE["n_patterns"] == 20,
         "cosine + zero scan, WCC with l = 1° and the ln q scan rank a same-form entry first for all 20 patterns"),
    need(CASE_T1["none|model_bkg"] == 0, "the model ranks a same-form entry first for none of them"),
    need(refl["benchmark_entries_frac_fewer"] < 0.01,
         f"The salt has {refl['case_entry']} reflections between 5° and 50° 2θ, where the median same-form entry of the "
         f"benchmark has about {round(refl['benchmark_entries_median'], -1):.0f}"),
    need(abs(RFL["(1, 0, 1)"] - RFL["(2, 0, 2)"]) > 0.3,
         f"have measured-to-simulated peak-height ratios of {f2(RFL['(1, 0, 1)'])} and {f2(RFL['(2, 0, 2)'])}"),
    f"March–Dollase texture fitted along each of {TF['axes_tested']} low-index axes, with r from {TF['r_grid'][0]:g} to "
    f"{TF['r_grid'][1]:g}, raises the median Pearson correlation of the {TF['n_patterns']} patterns with their self-entries only "
    f"from {f2(TF['median_pearson_no_texture'])} to {f2(TF['median_pearson_best_texture'])}",
    f"The skewness of N₁₀ over the degraded probes is {hub['cos_zs_bkg']['N10_skew_aug']:.1f} for cosine + zero scan, "
    f"{hub['lnq_bkg']['N10_skew_aug']:.1f} for the ln q scan, {hub['pearson_bkg']['N10_skew_aug']:.1f} for Pearson, "
    f"{hub['wcc_1_bkg']['N10_skew_aug']:.1f} for both WCC widths and {hub['model_bkg']['N10_skew_aug']:.1f} for the model",
    need(f"{hub['wcc_1_bkg']['N10_skew_aug']:.1f}" == f"{hub['wcc_2_bkg']['N10_skew_aug']:.1f}", "for both WCC widths"),
    f"CSLS raises the\ncompound-weighted top-1 of the classical scores by {rng(csls_std)} when any same-form entry counts and by\n"
    f"{rng(csls_sc)} on the single-crystal set",
    need(min(csls_std + csls_sc) > 0, "CSLS raises the"),
    need(CSLS_EXCL == [("std", "wcc_2_bkg")] and pair["std|wcc_2_bkg"]["macro_diff_ci"][0] > 0,
         "Of these paired differences only the gain of WCC with l = 2° on the\nany-same-form set excludes zero"),
    need(dict(pair["std|wcc_2_bkg"]["fixed"])["ZZZUWK"] == CASE_T1["csls_aug_k10|wcc_2_bkg"] - CASE_T1["none|wcc_2_bkg"],
         f"and {PH['csls_wcc2_zzzuwk']} patterns it gains are ZZZUWK's"),
    need(pair["std|model_bkg"]["macro_diff"] == 0 and pair["sc|model_bkg"]["macro_diff"] == 0, "CSLS leaves the model unchanged"),
    # ---- 4.6
    f"it ranks the probe's own entry first for {PH['probe_sim_top1']} of probes (median rank {PH['probe_sim_med']}), and for "
    f"the degraded probes\nfor {PH['probe_aug_top1']} (median rank {PH['probe_aug_med']})",
    f"it reaches a compound-weighted top-1 of {f2(gm(M, 'own_clean')[0])} and a median rank of {mr(gm(M, 'own_clean')[1])}, "
    f"against {f2(measured[0])} and about {about(measured[1])} on the measured patterns of the same targets",
    f"split have a median rank of {PH['ov_train_med']}, and patterns whose form is in no split {PH['ov_none_med']}",
    f"with Kα2 the model reaches {f2(gm(M, 'own_ka2')[0])} (median rank {mr(gm(M, 'own_ka2')[1])}), with the Debye–Waller "
    f"factor {f2(gm(M, 'own_dw')[0])} (median rank {mr(gm(M, 'own_dw')[1])}), and with the first six effects together "
    f"{f2(gm(M, 'own_all')[0])} (median rank {mr(gm(M, 'own_all')[1])})",
    f"With those six plus the Debye–Waller factor it reaches {f2(own_all_dw[0])} at a median rank of {mr(own_all_dw[1])}, and "
    f"with intensity jitter added {f2(gm(M, 'own_all_dw_int30')[0])} at a median rank of {mr(gm(M, 'own_all_dw_int30')[1])}",
    f"falls to a median rank of {mr(gm(ORIG, 'own_all_dw')[1])} on the rendering with the six effects and the Debye–Waller factor",
    f"stays at {f2(measured[0])} with a median rank of about {about(measured[1])}",
    need(gm(M, "own_all_dw_int30", "pearson")[0] - gm(M, "measured", "pearson")[0] <= 0.05,
         f"Fixed-grid Pearson, by contrast, reaches {f2(gm(M, 'measured', 'pearson')[0])} on the measured patterns, against "
         f"{f2(gm(M, 'own_all_dw_int30', 'pearson')[0])} on the fully rendered self-entries"),
    f"the measured patterns reach a compound-weighted top-1 of {f2(gm(M, 'measured', 'p2p')[0])} (median rank "
    f"{mr(gm(M, 'measured', 'p2p')[1])}), whereas fixed-grid Pearson",
    f"alignment search, reaches {f2(gm(M, 'measured', 'pearson')[0])} on the same patterns",
    need(all(incl0(v["ci"]) for v in pvo.values()),
         f"The second seed reaches {PH['s1_std']} on the measured patterns with a median rank of about "
         f"{about(MOD[S1]['std_median_rank'])}, the model without operator tokens {f2(MOD[FIX]['std_macro_top1']['value'])} with a "
         f"median rank of about {about(MOD[FIX]['std_median_rank'])}, and none of the three is statistically distinguishable from "
         f"the earlier model (paired differences {PH['paired_vs_orig_range']})"),
    need(SE[M]["n_ci_separated"] == 0 and SE[S1]["n_ci_separated"] == 0 and SE[M]["n_positive"] == G["n_conditions"],
         f"The operator tokens raise the model's within-family top-1 in the simulated sweep by about "
         f"{f2((SE[M]['mean_diff'] + SE[S1]['mean_diff']) / 2)} over the ablation (in all {SE[M]['n_conditions']} conditions for "
         f"seed 0 and {SE[S1]['n_positive']} for seed 1, although in no single condition do the intervals separate)"),
    f"its top 1,000 holds a correct entry for {PH['model_recall_1000']} of the patterns, and re-ranking that list with the ln q "
    f"scan gives a compound-weighted top-1 of {PH['hybrid_lnq']}, against {f2(b('lnq_bkg')['macro_top1'])} for the ln q scan",
    # ---- 4.7
    need(all(conv_pp[f"{td}|lnq_bkg"]["top1"]["diff"] == 0 and conv_pp[f"{td}|model_bkg"]["top1"]["diff"] == 0 for td in ("std", "sc")),
         f"changes the classical compound-weighted top-1 by {PH['conv_range']} and the ln q scan not at all, and leaves the model unchanged"),
    f"Removing the {D['n_wavelength_missing_assumed_cu']} patterns without a recorded wavelength changes no classical score by "
    f"more than {PH['nolam_range']}",
    f"Removing the {ROB['label_sensitivity']['n_lenient_patterns']} lenient-anchor patterns changes the classical scores by "
    f"{PH['strict_range']}, and the model scores {PH['strict_model']}",
    f"changes no compound-weighted top-1 of the ambient set by more than {PH['ea_change']}",
    need(all(T2["r2"]["model_bkg"][s] == T2["main"]["model_bkg"][s] for s in ("ambient", "std")) and r2_exc <= 0.1 * r2_int,
         f"The library package of Section 3.8 changes the classical scores by {PH['r2_range']}; {r2_exc} of the {r2_int} paired "
         f"intervals of the package and its ablations"),
    f"changes the classical scores by {PH['dT_range']} against the converted intensities alone, and averaging over each grid "
    f"step instead of interpolating by {PH['binavg_range']}",
    need(round(T2['binavg']['wcc_2_bkg']['ambient'] - T2['main']['wcc_2_bkg']['ambient'], 2) == 0.06
         and max(abs(d) for k in CLS if k != 'wcc_2_bkg' for d in t2diff('binavg', keys=[k])) <= 0.01
         and abs(T2['binavg']['wcc_2_bkg']['std'] - T2['main']['wcc_2_bkg']['std']) <= 0.01
         and incl0(ROB['binavg']['paired_vs_main']['sc|wcc_2_bkg']['top1']['ci']),
         f"Only WCC with l = 2° moves by more than 0.01: by +0.06 on the ambient set and {PH['binavg_wcc2_sc']} on the\n"
         "single-crystal set"),
    table2(),
    # ---- 4.8
    f"Against the {CODR['meta']['n_library']:,} COD entries, the {CODR['meta']['n_queries']} patterns reproduce the main result",
    f"the classical scores reach a compound-weighted top-1 of {PH['cod_std_range']} ({cod_m('std', 'lnq_bkg')['n_compounds']} "
    f"compounds) and the model {PH['cod_model']}",
    need(CODX["sc_composition"].get("ZZZUWK") == 20, f"single-crystal set has {CODR['per_pattern']['sc|lnq_bkg']['n']} patterns "
         f"but only {cod_m('sc', 'lnq_bkg')['n_compounds']} compounds, 20 of its patterns are ZZZUWK"),
    need(cod_si["n_si_queries"] == 2, f"without these two queries no compound-weighted top-1 changes by more than {PH['cod_si_change']}"),
    f"deposited with {KD['summary']['n_compounds']} structure reports in Powder Diffraction ({KD['summary']['n_patterns']} patterns, "
    f"{sum(r['kind'] == 'synchrotron' for r in KD['summary']['patterns'])} of them synchrotron)",
    f"the classical scores reach a compound-weighted top-1 of {PH['kaduk_cls_range']} and the model {PH['kaduk_model']}, "
    f"with a median rank of {PH['kaduk_model_med']}",
    f"only {KD['summary']['n_sc_compounds']} compounds keep an independent single-crystal reference; the classical scores "
    f"place it first for {PH['kaduk_sc_range']} of them and the model for {PH['kaduk_sc_model']}",
    # exploratory instrument / source split (4.6, discussion, conclusions, abstract)

    f"On the opXRD patterns its median rank is {PH['inst_labcu_med']} for the {PH['inst_labcu_n']} laboratory Cu patterns, "
    f"{PH['inst_syncamb_med']} for the {PH['inst_syncamb_n']} ambient synchrotron patterns and {PH['inst_syncp_med']} for the "
    f"{PH['inst_syncp_n']} pressure-series patterns, whereas fixed-grid Pearson ranks the answer first for "
    f"{PH['inst_pearson_range']} of the patterns in each of these groups",
    need(sum(ISP["groups"][g]["n"] for g in ISP["groups"]) == D["n_queries"] and ISP["groups"]["sync_pressure"]["n"] == AS["n_pressure_series_patterns"],
         f"On the {PH['kad_sync_n']} synchrotron profiles of the external set, all from one group's Rietveld studies, the model's "
         f"median rank is {PH['kad_sync_med']} and it ranks {PH['kad_sync_top1']} of them first, while its {PH['kad_lab_n']} "
         f"laboratory Cu profiles rank beyond {PH['kad_lab_min']}"),
    need(ov["model_bkg_train"]["median_rank"] > 1000,
         f"Of those {PH['kad_sync_n']} self-entries, {PH['kad_sync_train']} are in the model's training split, but on the opXRD "
         f"patterns membership made no difference, and the {PH['kad_sync_nontrain_n']} that are not rank {PH['kad_sync_nontrain']}"),
    need(min(ISP[M][g]["median_rank"] for g in INST_G) >= 1000,
         f"it ranks the self-entry at a median of {PH['kad_sync_med']}, against thousands on the opXRD patterns"),
    need(KK[M]["synchrotron"]["median_rank"] < 0.01 * min(ISP[M][g]["median_rank"] for g in INST_G),
         "on high-resolution synchrotron profiles from one external source the model does far better, for reasons we have "
         "not identified"),
    # ---- discussion, conclusions
    f"the shift-tolerant scores reach a compound-weighted top-1 of {rng(amb_tol)} over {AS['n_ambient_compounds']} compounds, and "
    f"when the pattern's self-entry is in the library they reach up to {f2(max(std_cls))}, with even fixed-grid Pearson "
    f"correlation at {f2(b('pearson_bkg')['macro_top1'])}",
    f"the similarity product of one query with {D['n_library']:,} entries\ntakes about {RF['model_query_ms_cpu']} ms on a CPU",
    f"about {cpu['pearson']:.1f} s per query for fixed-grid Pearson, about {round(cpu['wcc_1'], -1):.0f} s for a zero scan and "
    f"about {round(cpu['lnq'] / 60):.0f} min for the ln q scan",
    f"one GPU needs about {PH['gpu_s_per_pattern']} s per pattern for all five classical scores together "
    f"({D['n_queries'] + 2 * D['n_probe_per_set']:,} patterns against the full library in {RF['hub_wall_s']} s)",
    f"at a median of {PH['c2_cpu_min']} CPU-minutes per comparison",
    f"We searched {D['n_queries']} measured organic powder patterns against {D['n_library']:,} simulated CSD entries",
    f"first for a compound-weighted {rng(amb_tol)} of the {AS['n_ambient_compounds']} compounds",
    f"the classical scores reach {rng(std_cls)} and the model reaches {f2(b('model_bkg')['macro_top1'])}. The model retrieves",
]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    raw = open(ARGS.md, encoding="utf-8").read()
    if ARGS.fill:
        def sub(mo):
            k = mo.group(1)
            return PH[k] if k in PH else mo.group(0)
        body_start = raw.index("## Abstract")          # leave the header comment's {{name}} legend alone
        raw = raw[:body_start] + re.sub(r"\{\{([A-Za-z0-9_]+)\}\}", sub, raw[body_start:])
        out = ARGS.out or ARGS.md
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(raw)
        print(f"filled -> {out}")
    body = raw[raw.index("## Abstract"):]
    left = sorted(set(re.findall(r"\{\{([A-Za-z0-9_]+)\}\}", body)) - {"zenodo_doi"})
    text = norm(body)
    bad = [c for c in CLAIMS if norm(c) not in text]
    for c in bad:
        print("MISSING:", norm(c)[:400])
    for k in left:
        print("PLACEHOLDER LEFT:", k)
    if "{{zenodo_doi}}" in body:
        print("note: {{zenodo_doi}} awaits the reserved DOI of the new Zenodo version (Frank's OK)")
    print(f"{len(CLAIMS) - len(bad)}/{len(CLAIMS)} claims found verbatim; {len(left)} placeholders left")
    return len(bad) + len(left)


if __name__ == "__main__":
    sys.exit(main())
