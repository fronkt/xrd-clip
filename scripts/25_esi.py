"""Electronic Supplementary Information for the Digital Discovery manuscript, generated from the numbers file.

  python scripts/25_esi.py            -> paper/dd/esi.md

Every table is written from results/paper_numbers.json (22), results/case_zzzuwk.json (24) and the revision results
(results/model_gap*.json, 30), so the ESI cannot drift from the data. Prose here is short on purpose; the
manuscript carries the argument.
"""
from __future__ import annotations

import json
import re

P = json.load(open("results/paper_numbers.json"))
CZ = json.load(open("results/case_zzzuwk.json"))
TITLE = re.search(r'^title: "(.*)"$', open("paper/dd/manuscript.md", encoding="utf-8").read(), re.M).group(1)
B, D = P["bench"], P["dataset"]
NAMES = {"lnq_bkg": "ln q lag + zero scan", "cos_zs_bkg": "Cosine + zero scan", "pearson_bkg": "Pearson, fixed grid",
         "wcc_1_bkg": "WCC, l = 1° + zero scan", "wcc_2_bkg": "WCC, l = 2° + zero scan", "model_bkg": "Learned embedding",
         "pearson_raw": "Pearson, fixed grid", "cos_zs_raw": "Cosine + zero scan", "wcc_1_raw": "WCC, l = 1° + zero scan",
         "wcc_2_raw": "WCC, l = 2° + zero scan", "model_raw": "Learned embedding"}
CORR = {"none": "none", "csls_aug_k10": "CSLS, k = 10, degraded probes (primary)",
        "csls_sim_k10": "CSLS, k = 10, clean probes", "znorm_sim": "z-norm, clean probes", "znorm_aug": "z-norm, degraded probes",
        "mp_sim": "mutual proximity, clean probes", "mp_aug": "mutual proximity, degraded probes"}
for ps, nm in (("aug", "degraded"), ("sim", "clean")):
    for k in (1, 5, 20, 50, 100):
        CORR[f"csls_{ps}_k{k}"] = f"CSLS, k = {k}, {nm} probes"


def f2(x):
    return "n/a" if x is None else f"{x:.2f}"


def ci(v):
    return f"[{v[0]:.2f}, {v[1]:.2f}]"


def table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]


LADDER = [("own_clean", "clean"), ("own_mask", "+ coverage"), ("own_width", "+ peak width"), ("own_ka2", "+ Kα2"),
          ("own_bkg", "+ background"), ("own_noise", "+ noise"), ("own_zero", "+ zero offset"), ("own_dw", "+ B"),
          ("own_all", "all six"),
          ("own_all_dw", "all six + B"), ("own_all_int30", "all six + jitter 0.3"), ("own_all_int50", "all six + jitter 0.5"),
          ("own_all_dw_int30", "all six + B + jitter 0.3"), ("measured", "measured patterns")]
CLS = ["lnq_bkg", "cos_zs_bkg", "pearson_bkg", "wcc_1_bkg", "wcc_2_bkg"]
SETS = (("std", "any same-form"), ("sc", "single-crystal"))


def sg(x):            # signed, without a sign on zero
    s = f"{x:+.2f}"
    return "0.00" if s in ("+0.00", "-0.00") else s


def dci(v):           # a paired difference with its CI
    return f"{sg(v['diff'])} [{sg(v['ci'][0])}, {sg(v['ci'][1])}]"


def mci(v):           # a compound-weighted value with its CI
    return f"{f2(v['value'])} {ci(v['ci'])}"


def revision_sections() -> list:
    """S9-S13: the revision analyses (corrected ln q, variable-cell re-rank, effect ladder, library realism, open replica)."""
    ex, rev, B_ = P["extra"], P["revision"], P["bench"]
    L = ["", "## S9 The first version of the ln q scan", "",
         "The first version sampled the ln q axis at points and used 16 zero offsets that skipped zero (nearest ±0.02°); the "
         "corrected version averages each ln q bin over its width and uses 17 offsets from −0.32° to +0.32° (0.04° steps). The "
         "article reports the corrected version throughout.", "",
         "**Table S17** ln q lag + zero scan, first and corrected versions, on the same patterns (background-subtracted, no "
         "hubness correction). Paired differences over compounds.", ""]
    rows = []
    for td, nm_ in SETS:
        v = ex["lnq_v1"][td]
        cur = B_[f"none|{td}|lnq_bkg"]
        rows.append([nm_, mci(v["v1"]["macro_top1"]), f"{f2(cur['macro_top1'])} {ci(cur['macro_top1_ci'])}",
                     dci(v["paired_v2_minus_v1"]["top1"]), dci(v["paired_v2_minus_v1"]["top10"]),
                     v["n_rank_changed"], v["n_top1_changed"]])
    L += table(["answer set", "first version, macro top-1", "corrected, macro top-1", "corrected − first, top-1",
                "corrected − first, top-10", "patterns whose rank changes", "top-1 changes"], rows)

    c2, c2r = ex["critic2"], rev["r3_critic2"]
    sp = ex["sc_ambient_split"]
    L += ["", "## S10 Variable-cell re-rank (critic2 VC-GPWDF)", "",
          f"critic2 1.4.155 COMPAREVC GLOBAL QUICK, α = 1, measurement wavelength, on the ln q top {c2['k']} of each of the "
          f"{c2['n_patterns']} single-crystal patterns ({c2['n_comparisons']:,} comparisons, set-aside entries excluded). "
          f"Median {c2['cpu_min_median']:.1f} CPU-minutes per comparison, maximum {c2['cpu_h_max']:.2f} CPU-hours, "
          f"{c2['cpu_h_total']:.0f} CPU-hours in total; {c2['n_failed']} comparison reached the 4-hour wall-time limit and counts as "
          f"maximally dissimilar (" + "; ".join(f"{p} against {c}" for p, c, _ in c2r["meta"]["failed"]) + ", whose shortlist "
          f"does not hold the answer). The answer is in the shortlist for {c2['n_answer_in_shortlist']} patterns, the same number "
          f"as in the ln q top 100 ({c2['n_answer_in_lnq_top100']}). Because the cost of a comparison was unknown, the top 10 was "
          "run first and then extended to 50.", "",
          "Disordered inputs: the CSD extraction writes CIFs without occupancies, so the first run passed every disorder "
          "component of the few disordered entries at full occupancy. One answer was affected (ROHWIX01 for pattern_1006: DIFF "
          "0.261, the worst of its shortlist). All structures with a disorder model were replaced by their major component and "
          "the affected comparisons rerun; ROHWIX01 then scores 0.006, the best of its shortlist.", "",
          f"Compound-weighted top-1: re-rank {mci(c2r['critic2_vcgpwdf']['top1'])}, ln q {mci(c2r['lnq_v2_full_library']['top1'])}, "
          f"paired {dci(c2r['paired_critic2_minus_lnq']['top1'])}; top-10: {mci(c2r['critic2_vcgpwdf']['top10'])} against "
          f"{mci(c2r['lnq_v2_full_library']['top10'])}, paired {dci(c2r['paired_critic2_minus_lnq']['top10'])}.", "",
          "Positive control: each pattern against its own powder-study entry (the structure refined against that pattern); its "
          "DIFF is ranked among the wrong entries of the shortlist. Injection: each answer missing from a shortlist (the single-crystal same-form "
          "entry closest in cell) compared with its pattern and ranked the same way. Pressure series: the pattern's own entry "
          "carries a CSD pressure, or another entry of its family from the same publication does, or it is flagged by hand "
          "with the evidence (scripts/34_sc_conditions.py; HOJQII01/02, compressed cells at 293 K from a pressure study). The "
          "cell difference is the largest relative difference of the primitive Niggli-reduced cell lengths of the own entry and "
          "the closest single-crystal entry; ΔT is the difference of their recorded temperatures.", "",
          "**Table S18** The single-crystal patterns. Ranks are 1-based; '–': the answer is not in the ln q top "
          f"{c2['k']}. Peaks: critic2 peak-fit count inside the d window.", ""]

    def cell(v, fmt):
        return "n/a" if v is None else format(v, fmt)
    L += table(["opXRD pattern", "CSD family", "pressure series", "cell difference (%)", "ΔT (K)", "peaks", "ln q rank",
                "rank after the re-rank", "own entry, rank among wrong entries", "injected answer, rank"],
               [[r["id"], r["stem"], "yes" if r.get("pressure_series") else "", cell(r["cell_diff_pct"], ".2f"),
                 cell(r.get("dT_K"), ".0f"), r["n_peaks"], f"{r['lnq_rank']:,}",
                 "–" if r["critic2_rank"] is None else r["critic2_rank"],
                 cell((r.get("control") or {}).get("rank_in_shortlist"), "d"),
                 cell((r.get("injected") or {}).get("rank_if_shortlisted"), "d")]
                for r in sorted(c2["per_pattern"], key=lambda r: (r["stem"], int(r["id"].split("_")[1])))])
    cs, isu = c2["control_summary"], c2["injected_summary"]
    L += ["", f"Positive control: own entry ranked first for {cs['ambient']['rank1']} of {cs['ambient']['n']} ambient-pressure and "
          f"{cs['pressure_series']['rank1']} of {cs['pressure_series']['n']} pressure-series patterns (median rank "
          f"{cs['ambient']['median_rank']:.0f} and {cs['pressure_series']['median_rank']:.1f}). Injection: the missing answer would "
          f"rank first for {isu['ambient']['rank1']} of {isu['ambient']['n']} ambient-pressure and {isu['pressure_series']['rank1']} of "
          f"{isu['pressure_series']['n']} pressure-series patterns. Fitted peaks: {c2['n_peaks_range']['ambient'][0]}–"
          f"{c2['n_peaks_range']['ambient'][1]} (ambient) and {c2['n_peaks_range']['pressure_series'][0]}–"
          f"{c2['n_peaks_range']['pressure_series'][1]} (pressure series).", "",
          f"**Table S18b** Single-crystal set without the pressure-series patterns ({sp['n_ambient_patterns']} patterns, "
          f"{sp['n_ambient_compounds']} compounds; a secondary split defined after the internal review): compound-weighted top-1 "
          "and patterns ranked first; the pressure-series column counts first places among the "
          f"{sp['n_pressure_series_patterns']} pressure-series patterns.", ""]
    names = {"lnq": "ln q lag + zero scan", "lnq_v1": "ln q, first version", "critic2": "VC-GPWDF re-rank",
             "wcc_1": "WCC, l = 1° + zero scan", "wcc_2": "WCC, l = 2° + zero scan", "cos_zs": "Cosine + zero scan",
             "pearson": "Pearson, fixed grid"}
    L += table(["score", "macro top-1, ambient", "first places, ambient", "first places, pressure series"],
               [[nm_, mci(sp[k]["macro_top1"]), sp[k]["hits"], sp[k]["hits_pressure_series"]] for k, nm_ in names.items()] +
               [["Learned embedding", mci(sp["model"]["macro_top1"]), "", ""]])
    cd = sp["cells_differ"]
    L += ["", f"Patterns whose cells differ by more than 1 % ({cd['n']}): pressure series {cd['pressure_series']}; recorded "
          f"temperatures more than 50 K apart (not pressure series) {cd['dT_gt_50_not_pressure']}; supercell description "
          f"{cd['supercell']}; none of these {cd['none_of_these']} ({', '.join(cd['none_of_these_ids'])})."]

    x2, x3 = ex.get("x2_zero_offset"), ex.get("x3_dac_refit")
    if x2 and x3:
        dz = x2["delta"]
        L += ["", "### Sensitivity of the re-rank to the zero offset and to the peak fits", "",
              "Zero offset: each single-crystal pattern gets an answer-blind zero offset δ, the native shift (±0.5°, steps of "
              "0.02°, both scaled by λ/λ(Cu)) that maximises the cosine between the background-subtracted measured pattern on "
              "the Cu grid and the clean rendering of its ln q top-1 entry, which is never the answer. The fitted peak positions "
              f"are shifted by −δ and the {c2['n_answer_in_shortlist']} shortlists that hold the answer are rerun "
              f"({x2['n_comparisons_rerun']:,} comparisons; a shortlist without the answer stays a miss). Median |δ| "
              f"{dz['median_abs_lab_cu_deg']:.2f}° for the {dz['n_lab_cu']} laboratory Cu patterns and "
              f"{dz['median_abs_cu_equiv_deg']:.2f}° (Cu-equivalent) over all {dz['n']}; {dz['n_at_edge']} offsets lie at the "
              "edge of the scan.", "",
              f"Peak fits: {len(x3['control'])} patterns, all from pressure series, start their measured range with a run of "
              "zero-intensity points, which can drive the 20-knot background into large oscillations. Those points are dropped "
              "and the patterns refitted with critic2's default settings; their shortlists "
              f"({x3['n_comparisons_rerun']:,} comparisons), own-entry controls and answer injections are rerun. A grid search "
              "over the fit settings, judged by the own-entry DIFF, was set aside before it ran: that criterion rewards sparse "
              "peak lists and would have picked degenerate fits.", "",
              "Rerun comparisons past the 4-hour limit, counted as maximally dissimilar: " +
              "; ".join(f"{p} against {c} ({nm_}{', a correct answer' if is_t else ', not an answer'})"
                        for nm_, v in (("zero offsets", x2), ("refits", x3)) for p, c, is_t in v["failed_rerun"]) + ".", "",
              "**Table S18c** The re-rank with answer-blind zero offsets and with the refitted pressure-series patterns, against "
              f"the main re-rank (paired differences over {x2['top1']['n_compounds']} compounds).", ""]
        L += table(["re-rank", "comparisons rerun", "macro top-1", "− main, top-1", "macro top-10", "− main, top-10",
                    "first places", "patterns whose rank changes"],
                   [["main", "", mci(c2r["critic2_vcgpwdf"]["top1"]), "", mci(c2r["critic2_vcgpwdf"]["top10"]), "",
                     x2["hits_main"], ""]] +
                   [[nm_, f"{v['n_comparisons_rerun']:,}", mci(v["top1"]), dci(v["paired_minus_main"]["top1"]), mci(v["top10"]),
                     dci(v["paired_minus_main"]["top10"]), v["hits"], v["n_rank_changed"]]
                    for nm_, v in (("zero offsets", x2), ("refitted pressure series", x3))])
        L += ["", "**Table S18d** The refitted patterns, before → after. Ranks as in Table S18 ('–': the answer is not in "
              "the shortlist; 'n/a': no injection, the answer is in the shortlist).", ""]

        def ba(a, b, fmt="d"):
            f_ = (lambda v: "n/a" if v is None else format(v, fmt))
            return f"{f_(a)} → {f_(b)}"
        inj = x3["injected"]
        L += table(["opXRD pattern", "CSD family", "own entry", "peaks", "own entry DIFF", "own entry, rank among wrong entries",
                    "injected answer, rank", "rank after the re-rank"],
                   [[r["id"], r["stem"], r["own"], ba(r["old_n_peaks"], x3["n_peaks"][r["id"]]), ba(r["old_diff"], r["diff"], ".3f"),
                     ba(r["old_rank_in_shortlist"], r["rank_in_shortlist"]),
                     ba(inj[r["id"]]["old_rank_if_shortlisted"], inj[r["id"]]["rank_if_shortlisted"]) if r["id"] in inj else "n/a",
                     " → ".join("–" if v is None else str(v) for v in (r["old_rerank_rank"], r["rerank_rank"]))]
                    for r in sorted(x3["control"], key=lambda r: (r["stem"], int(r["id"].split("_")[1])))])
        L += ["", f"Own entry ranked first: {x3['control_rank1_old']} → {x3['control_rank1']} of {len(x3['control'])}; median "
              f"own-entry DIFF {x3['control_median_diff_old']:.3f} → {x3['control_median_diff']:.3f}. Injected answer ranked "
              f"first: {x3['injected_rank1_old']} → {x3['injected_rank1']} of {len(inj)}."]

    GA, GR = rev["r4_gap"]["results"], rev["r4c_realism"]["results"]
    est = rev["r4_gap"]["estimates"]
    L += ["", "## S11 Effect ladder", "",
          f"Each pattern's own library entry, rendered with effects estimated from its measurement: its 2θ coverage "
          f"({est['n_partial_coverage']} patterns do not cover 5–50°), its peak width (median FWHM {est['fwhm_deg_median']:.2f}°), "
          f"Kα2 for the {est['n_cu_lab']} Cu patterns, its own background followed by the background subtraction, noise at its "
          f"level and its best-aligning zero offset (median |offset| {est['zero_deg_abs_median']:.2f}°); B = 4.45 Å². Scored with "
          "the measured pattern's answer sets; 'p2p' matches the variant against the clean library patterns in the model's "
          "pattern space.", ""]
    for td, nm_ in SETS:
        L += [f"**Table S{19 if td == 'std' else 20}** Effect ladder, {nm_} answer set: compound-weighted top-1 [95 % CI] / "
              "per-pattern top-1 / median rank.", ""]
        rows = []
        for rung, lab in LADDER:
            cell = lambda res, m: (f"{mci(res[f'{rung}|{m}|{td}']['macro_top1'])} / {f2(res[f'{rung}|{m}|{td}']['per_pattern']['top1'])}"
                                   f" / {res[f'{rung}|{m}|{td}']['per_pattern']['median_rank']:,.1f}")
            rows.append([lab, cell(GA, "model"), cell(GR, "model"), f2(GA[f"{rung}|pearson|{td}"]["macro_top1"]["value"]),
                         f"{f2(GA[f'{rung}|p2p|{td}']['macro_top1']['value'])} / {f2(GR[f'{rung}|p2p|{td}']['macro_top1']['value'])}"])
        L += table(["variant", "original model", "realism retrain", "Pearson (macro)", "p2p macro, original / retrain"], rows) + [""]
    # the retrain saw only background-subtracted inputs; the single-effect rungs rerun with the subtraction on every rung
    ba, br = (json.load(open(f))["results"] for f in ("results/model_gap_bkgall.json", "results/model_gap_realism_bkgall.json"))
    L += ["**Table S20b** Single-effect rungs with the background subtraction applied to every rung (the realism retrain saw "
          "only background-subtracted patterns in training); compound-weighted top-1 / median rank, any same-form entry.", ""]
    L += table(["variant", "original model", "original, subtraction everywhere", "realism retrain",
                "retrain, subtraction everywhere"],
               [[lab] + [f"{f2(res[f'{rung}|model|std']['macro_top1']['value'])} / {res[f'{rung}|model|std']['per_pattern']['median_rank']:,.1f}"
                         for res in (GA, ba, GR, br)]
                for rung, lab in LADDER if f"{rung}|model|std" in ba]) + [""]

    r2, r2a = rev["r2"], rev["r2_ablations"]
    L += ["## S12 A library closer to the measurements", "",
          f"Package (Section 3.7 of the article): Debye–Waller B = {r2['meta']['B']:.2f} Å², Lorentz–polarisation conversion, "
          "Kα2 as a nuisance, comparison over the measured range. Ablations remove one change from the package.", "",
          "**Table S21** Compound-weighted top-1. Main = the article's benchmark; package − main and the ablations are paired over "
          "compounds (an ablation's difference is against the full package).", ""]
    rows = []
    for k in CLS + ["model_bkg"]:
        for td, nm_ in SETS:
            key = f"{td}|{k}"
            rows.append([NAMES[k], nm_, f2(B_[f"none|{key}"]["macro_top1"]), mci(r2["macro"][key]["top1"]),
                         dci(r2["paired_vs_main"][key]["top1"])] +
                        [dci(r2a[t]["paired_vs_r2"][key]) if key in r2a[t]["paired_vs_r2"] else "n/a"     # the model was
                         for t in ("r2_noB", "r2_nolp", "r2_noka2", "r2_nomask")])                      # not ablated
    L += table(["score", "answer set", "main", "package", "package − main", "− B", "− LP", "− Kα2", "− range"], rows)
    x1 = rev["x1_binavg"]
    L += ["", "**Table S21b** Measured patterns put on the Cu grid by averaging over each 0.02° grid step instead of "
          "interpolating (the design and a decision rule were fixed before the run: a change above 0.05 would have made it "
          "the primary preprocessing). Compound-weighted top-1, paired against the main benchmark; first places that "
          "change.", ""]
    L += table(["score", "answer set", "main", "bin-averaged", "difference", "first places changed"],
               [[NAMES[k], nm_, f2(B_[f"none|{td}|{k}"]["macro_top1"]), mci(x1["macro"][f"{td}|{k}"]["top1"]),
                 dci(x1["paired_vs_main"][f"{td}|{k}"]["top1"]), x1["paired_vs_main"][f"{td}|{k}"]["n_top1_changed"]]
                for k in CLS + ["model_bkg"] for td, nm_ in SETS])

    cod, cx = rev["r5_cod"], ex["cod"]
    L += ["", "## S13 The open replica (COD)", "",
          f"Library: {cx['n_selected']:,} COD entries pass the corpus filter and {cx['n_added_candidates']} candidate entries of the "
          f"queries are added ({cx['n_all']:,}); {cod['meta']['n_library']:,} gave stick patterns. Identified: {cx['n_identified']} "
          f"of {cx['n_patterns']} patterns ({cx['n_identified_both']} also identified in the CSD, {cx['n_identified_cod_only']} "
          f"only in the COD). Other-compound candidates kept as library entries: {cx['n_other_compound']:,}; same-formula "
          f"candidates set aside as not comparable: {cx['n_unknown_same_formula']}. Scored: {cod['meta']['n_queries']} patterns. "
          "ln q is the corrected version; no hubness correction and no package of Section S12.", "",
          "**Table S22** Retrieval on the open replica (background-subtracted queries).", ""]
    rows = []
    for k in CLS + ["model_bkg"]:
        for td, nm_ in SETS:
            pp, mm = cod["per_pattern"][f"{td}|{k}"], cod["macro"][f"{td}|{k}"]
            rows.append([NAMES[k], nm_, f"{pp['n']} / {mm['top1']['n_compounds']}", mci(mm["top1"]), f2(mm["top10"]["value"]),
                         f2(pp["top1"]), f"{pp['median_rank']:,.0f}"])
    L += table(["score", "answer set", "patterns / compounds", "macro top-1", "macro top-10", "per-pattern top-1",
                "median rank"], rows)

    kd = rev.get("x4_kaduk")
    if kd is not None:
        ks = kd["summary"]
        pp_ = ks["patterns"]
        n_sync = sum(r["kind"] == "synchrotron" for r in pp_)
        L += ["", "## S14 An external set: raw profiles deposited by Kaduk and co-workers", "",
              f"Queries: the measured profiles in the supplementary CIFs of Kaduk and co-workers' Powder Diffraction structure "
              f"reports (public Cambridge Core supplements; donepezil hydrochloride form III from the ICDD deposit): "
              f"{ks['n_patterns']} patterns of {ks['n_compounds']} compounds, {n_sync} synchrotron "
              f"({sum(r['kind'] == 'synchrotron' and r['wavelength'] < 0.5 for r in pp_)} at 11-BM, λ ≈ 0.414 Å; one at "
              f"0.6888 Å) and {ks['n_patterns'] - n_sync} lab Cu Kα1. None was used in training, tuning or any earlier "
              "analysis. The design, predictions and a decision rule were fixed before scoring; preprocessing, scores, "
              "checkpoint and library are those of the main benchmark (corrected ln q, no hubness correction). Correct answers "
              "follow Section S1, with the article's refcode as the anchor.", "",
              "Every anchor is the powder-study entry refined against the query pattern itself, so the any-same-form answer "
              "set is an upper bound, not a test. With powder-study entries ignored, only "
              f"{ks['n_sc_compounds']} compounds keep an independent single-crystal entry in the library; "
              f"{len(ks['sc_entries_outside_library'])} same-form single-crystal entries lie outside the corpus and were not "
              "added, so that the library is identical to the main benchmark.", "",
              "**Table S23** Retrieval on the external set (background-subtracted queries).", ""]
        rows = []
        for k in CLS + ["model_bkg"]:
            for td, nm_ in SETS:
                pp, mm = kd["per_pattern"][f"{td}|{k}"], kd["macro"][f"{td}|{k}"]
                rows.append([NAMES[k], nm_, f"{pp['n']} / {mm['top1']['n_compounds']}", mci(mm["top1"]),
                             f2(mm["top10"]["value"]), f2(pp["top1"]), f"{pp['median_rank']:,.0f}"])
        L += table(["score", "answer set", "patterns / compounds", "macro top-1", "macro top-10", "per-pattern top-1",
                    "median rank"], rows)
        rk = lambda r, key: "–" if r["ranks"][key] < 0 else f"{r['ranks'][key] + 1:,}"
        L += ["", "**Table S24** Rank of the best correct entry per pattern (1 = first place; '–': no correct entry in the "
              "library).", ""]
        L += table(["compound", "anchor", "article DOI", "source", "Cu-grid coverage", "ln q, any", "Pearson, any", "model, any",
                    "ln q, single-crystal", "model, single-crystal"],
                   [[r["compound"], r["refcode"], r["doi"], f"{r['kind']}, {r['wavelength']:.4f} Å", f2(r["cu_grid_coverage"]),
                     rk(r, "std|lnq_bkg"), rk(r, "std|pearson_bkg"), rk(r, "std|model_bkg"), rk(r, "sc|lnq_bkg"),
                     rk(r, "sc|model_bkg")] for r in pp_])
    return L


def main() -> int:
    L = ["# Electronic Supplementary Information", "",
         f"{TITLE}. F. Cai.", "",
         "All tables are generated by `scripts/25_esi.py` from `results/paper_numbers.json`.", ""]

    # S1 dataset
    L += ["## S1 Benchmark construction", "",
          f"opXRD CNRS patterns: {D['n_cnrs_total']:,}; molecular-organic: {D['n_opxrd_organic']}; identified by packing "
          f"similarity: {D['n_identified']} (strict {D['anchor_tiers']['strict']}, lenient {D['anchor_tiers']['lenient']}); "
          f"used: {D['n_queries']} patterns of {D['n_compounds']} compounds. Library: {D['n_corpus']:,} corpus entries + "
          f"{D['n_extra']} added same-form or polymorph entries = {D['n_library']:,}.", "",
          "**Table S1** Patterns not used.", ""]
    why = lambda r: ("2θ axis runs 0.01–0.83, probably radians; no unit is guessed" if r.startswith("covers")
                     else "no packing-similarity match (incl. Si frameworks/silsesquioxanes)")
    L += table(["opXRD pattern", "reason"], sorted((k, why(v)) for k, v in D["excluded"].items()))
    L += ["", "**Table S2** Compounds with more than two patterns.", ""]
    L += table(["CSD refcode family", "patterns"], [r for r in D["patterns_per_compound_top"] if r[1] > 2])
    L += ["", f"{D['n_compounds_single_pattern']} of the {D['n_compounds']} compounds have a single pattern.", ""]

    # S2 full results
    for td, title in (("std", "any same-form entry is correct"), ("sc", "only independent single-crystal entries are correct")):
        rows = []
        for k in ["lnq_bkg", "cos_zs_bkg", "pearson_bkg", "wcc_1_bkg", "wcc_2_bkg", "model_bkg",
                  "pearson_raw", "cos_zs_raw", "wcc_1_raw", "wcc_2_raw", "model_raw"]:
            x = B[f"none|{td}|{k}"]
            rows.append([NAMES[k], "bkg" if k.endswith("bkg") else "raw", f"{f2(x['macro_top1'])} {ci(x['macro_top1_ci'])}",
                         f"{f2(x['macro_top10'])} {ci(x['macro_top10_ci'])}", f"{f2(x['top1'])} {ci(x['top1_ci'])}",
                         f"{f2(x['top10'])} {ci(x['top10_ci'])}", f"{x['median_rank']:,.0f}", f2(x["frac_rank_gt_1000"])])
        L += ["", f"## S2{'a' if td == 'std' else 'b'} Full-library results, {title}", "",
              f"**Table S3{'a' if td == 'std' else 'b'}** {B[f'none|{td}|lnq_bkg']['n']} patterns, "
              f"{B[f'none|{td}|lnq_bkg']['n_compounds']} compounds, {D['n_library']:,} entries, no hubness correction. "
              "Compound-weighted (macro) and per-pattern top-k with 95 % compound-bootstrap CIs.", ""]
        L += table(["Score", "query", "macro top-1", "macro top-10", "per-pattern top-1", "per-pattern top-10",
                    "median rank", "fraction rank > 1,000"], rows)
    L += ["", "**Table S4** Within-family top-1 (rank among the compound's own forms in the library).", ""]
    L += table(["Score", "query", "patterns", "top-1"],
               [[NAMES[k], "bkg" if k.endswith("bkg") else "raw", v["n"], f2(v["top1"])] for k, v in P["within_family"].items()])
    L += ["", "**Table S5** Learned model as a shortlist, re-ranked by a classical score (per-pattern top-1, bkg).", ""]
    L += table(["Pipeline", "any same-form entry", "single-crystal entry only"],
               [[k.replace("model@", "model top-").replace("->", " → ").replace("_bkg", ""),
                 f"{f2(v['std']['top1'])} {ci(v['std']['top1_ci'])}", f"{f2(v['sc']['top1'])} {ci(v['sc']['top1_ci'])}"]
                for k, v in P["hybrid"].items()])
    L += ["", "**Table S6** Library-size curve (compound-weighted top-1, any same-form entry; mean ± s.d. over five random "
          "decoy subsets).", ""]
    rows = []
    for k in ["pearson_bkg", "cos_zs_bkg", "wcc_1_bkg", "wcc_2_bkg", "lnq_bkg", "model_bkg"]:
        for c in ("none", "csls_aug_k10"):
            cv = P["library_curve_macro"][f"{c}|{k}"]
            rows.append([NAMES[k], CORR[c]] + [f"{cv[s][0]:.2f} ± {cv[s][1]:.2f}" if s != "full" else f"{cv[s][0]:.2f}"
                                               for s in ("2000", "20000", "100000", "full")])
    L += table(["Score", "correction", "2,000", "20,000", "100,000", f"{D['n_library']:,}"], rows)

    # S3 hubness
    L += ["", "## S3 Hubness and its correction", "", "**Table S7** k-occurrence skewness (N₁₀) over the probe sets.", ""]
    L += table(["Score", "skew, degraded probes", "skew, clean probes", "max N₁₀ (degraded)"],
               [[NAMES[k], f"{v['N10_skew_aug']:.1f}", f"{v['N10_skew_sim']:.1f}", v["N10_max_aug"]] for k, v in P["hubness"].items()])
    L += ["", "**Table S8** Paired change in top-1 from the primary correction (CSLS, k = 10, degraded probes); 95 % "
          "compound-bootstrap CIs. Fixed/broken = patterns that the correction moves to or from rank 1.", ""]
    rows = []
    for key, v in P["paired_csls"].items():
        td, k = key.split("|")
        rows.append([NAMES[k], "any same-form" if td == "std" else "single-crystal",
                     f"{v['macro_diff']:+.3f} [{v['macro_diff_ci'][0]:+.3f}, {v['macro_diff_ci'][1]:+.3f}]",
                     f"{v['pattern_diff']:+.3f} [{v['pattern_diff_ci'][0]:+.3f}, {v['pattern_diff_ci'][1]:+.3f}]",
                     sum(c for _, c in v["fixed"]), sum(c for _, c in v["broken"])])
    L += table(["Score", "answer set", "macro difference", "per-pattern difference", "fixed", "broken"], rows)
    L += ["", "**Table S9** Sensitivity: every correction computed (none was selected after the fact). Top-1, "
          "background-subtracted queries.", ""]
    rows = []
    for c, nm in CORR.items():
        rows.append([nm] + [f"{f2(P['sensitivity'][f'{c}|{k}']['macro_top1'])} / {f2(P['sensitivity'][f'{c}|{k}']['top1'])} / "
                            f"{f2(P['sensitivity'][f'{c}|{k}']['sc_macro_top1'])}"
                            for k in ["pearson_bkg", "cos_zs_bkg", "wcc_1_bkg", "wcc_2_bkg", "lnq_bkg", "model_bkg"]])
    L += ["Each cell: macro top-1 (any same-form) / per-pattern top-1 (any same-form) / macro top-1 (single-crystal).", ""]
    L += table(["Correction", "Pearson", "Cosine + zs", "WCC 1°", "WCC 2°", "ln q", "Model"], rows)

    # S4 ZZZUWK
    cx = CZ["context"]
    L += ["", "## S4 The ZZZUWK case", "",
          f"ZZZUWK (propan-1-aminium chloride, P4/nmm) contributes {CZ['n_patterns']} patterns, a 303–398 K series; each is the "
          f"refinement data of its own entry ZZZUWK01–20. Own-entry score, median over its patterns vs the other "
          f"benchmark patterns: " + "; ".join(f"{k.replace('_bkg', '')} {v['own_score_median_case']:.2f} vs "
                                             f"{v['own_score_median_rest']:.2f}" for k, v in cx.items()
                                             if k in ("pearson_bkg", "cos_zs_bkg", "wcc_1_bkg")) + ".", "",
          "**Table S10** Relative peak heights (fraction of the maximum, mean over the 20 patterns): measured vs the simulated "
          "pattern of each pattern's own entry.", ""]
    L += table(["hkl", "measured", "own entry (simulated)"],
               [[h, f"{v['measured_mean']:.2f}", f"{v['own_entry_mean']:.2f}"] for h, v in cx["texture"].items()])
    L += ["", "**Table S11** Decoys that win ZZZUWK patterns under any score: patterns (of 20) in which the decoy outscores "
          "every same-form entry, with the zero scan, at zero offset only, and under fixed-grid Pearson; median best offset.", ""]
    rows = []
    for m, per in CZ["methods"].items():
        for rc, v in per.items():
            if v["role"] == "decoy":
                rows.append([m.replace("cos_zs", "cosine + zero scan").replace("wcc_1", "WCC l = 1°"), rc,
                             cx["reflections_5_50"]["decoys"][rc], v["beats_truth_with_scan"], v["beats_truth_at_offset0"],
                             v["beats_truth_pearson"], f"{v['best_offset_deg_median']:+.2f}"])
    L += table(["score", "decoy", "reflections 5–50°", "with scan", "zero offset", "Pearson", "best offset (°)"], rows)

    # S5 GPU + G0
    g = P["gpu_check"]
    L += ["", "## S5 GPU and CPU scores", "",
          f"Maximum |GPU − CPU| score over all benchmark patterns and their special entries: {g['max_abs_diff_vs_cpu_prepass']:.1e}. "
          f"Rank values that differ: {g['rank_mismatches']} (all ≤ 3 positions); top-1 changes: {g['top1_changes']}. "
          f"The corrected ln q scan (Section 3.2 of the article), rescored on the GPU alone, against its CPU ranks: "
          f"{g['lnq_v2_vs_cpu']['rank_mismatches']} rank value differs; top-1 changes: {g['lnq_v2_vs_cpu']['top1_changes']}.", "",
          "## S6 Simulated sweep, all conditions", "",
          f"**Table S12** Within-family top-1 over {P['g0']['n_families']:,} families ({P['g0']['n_queries']:,} queries); "
          f"95 % family-bootstrap CIs; composition-only chance {P['g0']['chance']:.2f}. The last column is the first version "
          "of the ln q scan (Section S9) on the same draws.", ""]
    L += table(["Condition", "model", "best classical", "best score", "Δ", "Pearson, fixed grid", "ln q", "ln q, first version"],
               [[r["condition"], f"{r['model']:.2f} {ci(r['model_ci'])}", r["best"], f"{r['best_top1']:.2f} {ci(r['best_ci'])}",
                 f"{r['delta']:+.2f}", f"{r['pearson_fixed']:.2f}", f"{r['lnq']:.2f}", f"{r['lnq_v1']:.2f}"] for r in P["g0"]["rows"]])
    L += ["", "The condition zero_offset=0.0 repeats the clean condition, so 34 conditions are distinct. Score names: "
          "pearson_zs = Pearson + zero scan; wcc_0.5, wcc_1, wcc_2 = WCC with l = 0.5°, 1°, 2° + zero scan; lnq = ln q lag + "
          "zero scan. Conditions in which each score beats the model with non-overlapping CIs: " +
          ", ".join(f"{k} {v}" for k, v in P["extra"]["g0_distinct"]["conditions_each_score_beats_model"].items()) + ". "
          "Each condition's random draws are seeded from a CRC-32 checksum of its name. A first run of the sweep seeded them "
          "from Python's salted string hash and cannot be regenerated; its draws differ, and no number here comes from it."]

    # S7 learned model
    RF, ex = P["run_facts"], P["extra"]
    mp = ex["model_probe_check"]
    L += ["", "## S7 Learned model", "",
          f"Model card: dual encoder (1D residual CNN over the 2,251-point pattern; Transformer over the atoms of the asymmetric "
          f"unit with a lattice token), 256-dimensional unit embeddings, {RF['model_params_millions']} million parameters. "
          f"Symmetric InfoNCE with a learnable temperature, AdamW (learning rate 3 × 10⁻⁴, weight decay 0.05), batch "
          f"{RF['model_batch']}, {RF['model_epochs']} epochs, one seed; polymorph hard-negative batches (sibling polymorphs in "
          "the same batch); scaffold- and polymorph-family split 247,089 / 30,887 / 30,886; checkpoint chosen by in-batch "
          "validation top-1 (0.914). Training patterns simulated on the fly with zero offset, isotropic strain, Caglioti and "
          "Scherrer broadening, March–Dollase texture, Chebyshev and amorphous background and Poisson noise; no Kα2.", "",
          # runs/xrdclip_realism/history.json: best val_top1 0.9105 at epoch index 54 (10-13, box A)
          "Realism retrain (Section 3.4 of the article; design and prediction fixed in repository commit 3b77818): the same "
          "split, architecture, hyperparameters, 60 epochs, hard-negative batches and checkpoint rule, one seed, with Kα2, an "
          "overall Debye–Waller factor, per-reflection log-normal intensity jitter, 2θ range cuts and the measured patterns' "
          "background subtraction added to the training simulation; best in-batch validation top-1 0.911. Neither checkpoint "
          "is distributed (CSD-derived).", "",
          "**Table S13** The model on simulated and measured queries against all "
          f"{mp['n_library']:,} entries. Probes: library entries outside the benchmark compounds; their own refcode family is "
          "set aside. Own entries: the clean simulated pattern of each benchmark pattern's own entry, scored with the "
          "measured pattern's answer sets.", ""]
    rows = [["clean simulated probes", mp["sim"]["n"], f2(mp["sim"]["top1"]), f2(mp["sim"]["top10"]), mp["sim"]["median_rank"]],
            ["degraded simulated probes", mp["aug"]["n"], f2(mp["aug"]["top1"]), f2(mp["aug"]["top10"]), mp["aug"]["median_rank"]]]
    for nm_, k in (("own entries, clean simulated", "own_entry_clean"), ("measured patterns", "measured_bkg")):
        for td in ("std", "sc"):
            v = mp[f"{k}|{td}"]
            rows.append([f"{nm_} ({'any same-form' if td == 'std' else 'single-crystal'})", v["n"], f2(v["top1"]),
                         f2(v["top10"]), v["median_rank"]])
    L += table(["query", "n", "top-1", "top-10", "median rank"], rows)
    ov = P["model_training_overlap"]
    L += ["", f"Training-split overlap (any same-form entry in): {ov['patterns_with_same_form_entry_in']} patterns; "
          f"model top-1 / median rank for patterns with a same-form entry in the training split "
          f"{f2(ov['model_bkg_train']['top1'])} / {ov['model_bkg_train']['median_rank']:,.0f} (n = {ov['model_bkg_train']['n']}), "
          f"with none in the corpus {f2(ov['model_bkg_none']['top1'])} / {ov['model_bkg_none']['median_rank']:,.0f} "
          f"(n = {ov['model_bkg_none']['n']}). Recall of the model's top 100 / 1,000: {f2(ex['model_recall']['100'])} / "
          f"{f2(ex['model_recall']['1000'])}.", "",
          "**Table S14** Search cost per query over the full library. CPU: per-pair wall time of the simulated sweep (one core) "
          f"times {D['n_library']:,}; the model's figure there includes encoding every pair and is not representative of a "
          f"library search, which with precomputed embeddings takes about {RF['model_query_ms_cpu']} ms per query.", ""]
    L += table(["score", "CPU s per query (estimate)"],
               [[k, f"{v:,.1f}"] for k, v in ex["cpu_s_per_query_full_library"].items() if k != "model"])

    # S8 subgroups and paired differences
    L += ["", "## S8 Subgroups, paired differences and ground-truth checks", "",
          "**Table S15** Paired differences in compound-weighted top-1 against fixed-grid Pearson (95 % compound-bootstrap CI).", ""]
    L += table(["score − Pearson", "answer set", "compounds", "difference"],
               [[NAMES[k.split("|")[1].split("-")[0]], "any same-form" if k.startswith("std") else "single-crystal", v["n_compounds"],
                 f"{v['diff']:+.2f} {ci(v['ci'])}"] for k, v in ex["paired_methods"].items()])
    L += ["", "**Table S16** Subgroups (any same-form entry): per-pattern / compound-weighted top-1.", ""]
    sub_names = {"tier_strict": "strict anchors", "tier_lenient": "lenient anchors", "corpus_entry": "a correct entry in the corpus",
                 "extra_only": "correct entries only among the added entries", "cu": "Cu Kα", "non_cu": "other wavelengths",
                 "ordered": "own entry ordered", "disordered": "own entry disordered"}
    rows = []
    for g, nm_ in sub_names.items():
        d = ex["subgroups"][g]
        rows.append([nm_, f"{d['n_patterns']} / {d['n_compounds']}"] +
                    [f"{f2(d[k]['top1'])} / {f2(d[k]['macro_top1'])}" for k in
                     ["lnq_bkg", "cos_zs_bkg", "pearson_bkg", "wcc_1_bkg", "wcc_2_bkg", "model_bkg"]])
    L += table(["subgroup", "patterns / compounds", "ln q", "cosine + zs", "Pearson", "WCC 1°", "WCC 2°", "model"], rows)
    cm, tg, au = ex["sc_cell_mismatch"], ex["sc_temperature_gap"], ex["wrong_top1_audit"]
    L += ["", f"Single-crystal set, cell agreement: largest relative cell-length difference between the own powder-study entry and "
          f"the closest single-crystal same-form entry, median {cm['median_max_axis_rel'] * 100:.2f} %. Patterns ranked first "
          f"(per pattern) for ≤ 1 % (n = {cm['n_small']}) / > 1 % (n = {cm['n_large']}): " +
          "; ".join(f"{nm_} {cm[k + '_hits_small']} / {cm[k + '_hits_large']}" for k, nm_ in
                    (("lnq", "ln q"), ("lnq_v1", "ln q, first version"), ("critic2", "VC-GPWDF re-rank"), ("wcc_1", "WCC 1°"),
                     ("wcc_2", "WCC 2°"), ("cos_zs", "cosine + zs"), ("pearson", "Pearson"))) + ". "
          f"Recorded temperatures known for both entries: n = {tg['n']}, median |ΔT| {tg['median_dT_K']:.0f} K; ln q top-1 "
          f"{f2(tg['top1_dT_le_50'])} for |ΔT| ≤ 50 K (n = {tg['n_dT_le_50']}) and {f2(tg['top1_dT_gt_50'])} above (n = {tg['n_dT_gt_50']}).", "",
          f"Ground-truth audit: {au['n_cases']} wrong first-ranked entries (five classical scores, both answer sets) compared with "
          f"the anchors by packing similarity ({au['n_pairs']} pairs); molecules matched (max over anchors): "
          f"{au['matched_distribution']} (None = no comparable molecules). Same form: " +
          "; ".join(sorted({f"{c['wrong']} for {c['id']} ({c['stem']})" for c in au["same_form_cases"]})) + ".", "",
          f"Entries set aside as not comparable: {ex['unknown_entries']['total']} over {ex['unknown_entries']['patterns_with_any']} "
          f"patterns. Own powder-study entry reproducing the deposited structure within 0.05 Å RMSD: "
          f"{ex['own_powder_rmsd_lt_0p05']['strict']} strict + {ex['own_powder_rmsd_lt_0p05']['lenient']} lenient anchors. "
          f"Within-family chance: {f2(ex['within_family_chance'])}."]
    L += revision_sections()
    with open("paper/dd/esi.md", "w", encoding="utf-8") as fh:
        # typographic minus for negative numbers, as in the article
        fh.write(re.sub(r"(?<![\w.])-(?=\d)", "−", "\n".join(L)) + "\n")
    print(f"wrote paper/dd/esi.md ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
