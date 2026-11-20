"""Supplementary material for the RSOS manuscript, generated from the numbers file (tasks/todo.md D-rest F4).

  python scripts/49_rsos_esi.py            -> paper/rsos/esi.md

The RSOS version of scripts/25 (which stays the record of the RSC Advances ESI). Every table is written from
results/rsos_numbers.json (48) and results/case_zzzuwk.json (24), so the supplement cannot drift from the data. Tables
are numbered in order of appearance by one counter. Prose is short on purpose; the article carries the argument.
"""
from __future__ import annotations

import json
import os
import re

P = json.load(open("results/rsos_numbers.json"))
CZ = json.load(open("results/d3/case_zzzuwk.json"))
TITLE = re.search(r'^title: "(.*)"$', open("paper/rsos/manuscript.md", encoding="utf-8").read(), re.M).group(1)
B, D, EX, ROB, M = P["bench"], P["dataset"], P["extra"], P["robustness"], P["models"]
MODEL = P["model_name"]
NAMES = {"lnq_bkg": "ln q lag + zero scan", "cos_zs_bkg": "Cosine + zero scan", "pearson_bkg": "Pearson, fixed grid",
         "wcc_1_bkg": "WCC, l = 1° + zero scan", "wcc_2_bkg": "WCC, l = 2° + zero scan", "model_bkg": "Learned model",
         "pearson_raw": "Pearson, fixed grid", "cos_zs_raw": "Cosine + zero scan", "wcc_1_raw": "WCC, l = 1° + zero scan",
         "wcc_2_raw": "WCC, l = 2° + zero scan", "model_raw": "Learned model"}
CLS = ["lnq_bkg", "cos_zs_bkg", "pearson_bkg", "wcc_1_bkg", "wcc_2_bkg"]
SETS = (("std", "any same-form"), ("sc", "single-crystal"))
def ref_number(key: str, md: str = "paper/rsos/manuscript.md") -> int:
    """The Vancouver number citeproc gives `key` in the article: its place in the order of first citation."""
    order: list[str] = []
    for grp in re.findall(r"\[(?:[^\]]*?)@[^\]]+\]", open(md, encoding="utf-8").read()):
        for k in re.findall(r"@([A-Za-z0-9_:-]+)", grp):
            if k not in order and not k.startswith(("fig:", "tbl:", "sec:", "eq:")):
                order.append(k)
    return order.index(key) + 1


REFN = ref_number("cai2026urtc")
CKPTS = (("d3_sym_s0", "symmetry-aware, seed 0 (the model)"), ("d3_sym_s1", "symmetry-aware, seed 1"),
         ("d3_fix", "without operator tokens"), ("poly", f"earlier model [ref. {REFN}]"))
CORR = {"none": "none", "csls_aug_k10": "CSLS, k = 10, degraded probes (primary)",
        "csls_sim_k10": "CSLS, k = 10, clean probes", "znorm_sim": "z-norm, clean probes", "znorm_aug": "z-norm, degraded probes",
        "mp_sim": "mutual proximity, clean probes", "mp_aug": "mutual proximity, degraded probes"}
for ps, nm in (("aug", "degraded"), ("sim", "clean")):
    for k in (1, 5, 20, 50, 100):
        CORR[f"csls_{ps}_k{k}"] = f"CSLS, k = {k}, {nm} probes"
LADDER = [("own_clean", "clean"), ("own_mask", "coverage"), ("own_width", "peak width"), ("own_noise", "noise"),
          ("own_ka2", "Kα2"), ("own_bkg", "background + subtraction"), ("own_zero", "zero offset"), ("own_dw", "Debye–Waller"),
          ("own_all", "first six together"), ("own_all_dw", "six + Debye–Waller"),
          ("own_all_int30", "six + jitter σ = 0.3"), ("own_all_int50", "six + jitter σ = 0.5"),
          ("own_all_dw_int30", "six + Debye–Waller + jitter σ = 0.3"), ("measured", "measured patterns")]
_T = [0]


def T() -> str:
    _T[0] += 1
    return f"**Table S{_T[0]}**"


def f2(x):
    return "n/a" if x is None else f"{x:.2f}"


def ci(v):
    return f"[{v[0]:.2f}, {v[1]:.2f}]"


def table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]


def sg(x):
    s = f"{x:+.2f}"
    return "0.00" if s in ("+0.00", "-0.00") else s


def dci(v):
    return f"{sg(v['diff'])} [{sg(v['ci'][0])}, {sg(v['ci'][1])}]"


def mci(v):
    return f"{f2(v['value'])} {ci(v['ci'])}"


def s1_benchmark() -> list:
    L = ["## S1 Benchmark construction", "",
         f"opXRD CNRS patterns: {D['n_cnrs_total']:,}; molecular-organic: {D['n_opxrd_organic']}; identified by packing "
         f"similarity: {D['n_identified']} (strict {D['anchor_tiers']['strict']}, lenient {D['anchor_tiers']['lenient']}); "
         f"used: {D['n_queries']} patterns of {D['n_compounds']} compounds. Library: {D['n_corpus']:,} corpus entries + "
         f"{D['n_extra']} added same-form or polymorph entries = {D['n_library']:,}. Wavelengths: {D['n_lab_cu']} Cu Kα, "
         f"{D['n_synchrotron_or_other']} other; {D['n_wavelength_missing_assumed_cu']} patterns have no recorded wavelength. "
         "For those three, the peak positions fitted against each pattern's deposited cell give Cu Kα (validated on 21 "
         "patterns of known wavelength, median error 0.05 %).", "",
         f"{T()} Patterns not used.", ""]
    why = lambda r: ("2θ axis runs 0.01–0.83, probably radians" if r.startswith("covers")
                     else "no packing-similarity match (incl. Si frameworks and silsesquioxanes)")
    L += table(["opXRD pattern", "reason"], sorted((k, why(v)) for k, v in D["excluded"].items()))
    L += ["", f"{T()} Compounds with more than two patterns.", ""]
    L += table(["CSD refcode family", "patterns"], [r for r in D["patterns_per_compound_top"] if r[1] > 2])
    L += ["", f"{D['n_compounds_single_pattern']} of the {D['n_compounds']} compounds have a single pattern. Refcode "
          "families whose same-form sets share an entry count as one compound: "
          + "; ".join(" and ".join(fs) for fs in D["compound_merges"]) + ".", ""]
    # independent check of the library simulation (scripts/51-53)
    sx = P["simulation_check"]
    gname = {"random": "randomly drawn library entries", "zzzuwk": "ZZZUWK self-entries",
             "disordered_self": "other disordered self-entries", "resimulated": "re-simulated disordered added entries"}
    L += ["Library simulation check. The CIFs written by the CSD Python API carry no site occupancies, so the disordered "
          f"entries among the {D['n_extra']} added entries were first simulated with every disorder site at full occupancy. "
          "Each disordered added entry was re-simulated twice, (A) with the occupancies of the CSD's asymmetric unit, right "
          "for disorder generated by symmetry, and (B) as its major component at full occupancy, right when the CSD gives "
          "coordinates for one component only, and the variant that agrees better with the CCDC's own powder simulation "
          f"(Mercury engine, CSD Python API) was kept: {sx['n_resimulated']} entries, {sx['n_major_component']} as the major "
          "component. Both simulations were smoothed with a Gaussian of σ = 0.15° 2θ before comparison. The atom lists that "
          "the learned model encodes are unchanged (the encoder has no occupancy input).", "",
          f"{T()} Correlation of our simulation with the CCDC's, before (every disorder site at full occupancy) and after "
          "the re-simulation.", ""]
    pre = sx["pre_occ_groups"]
    L += table(["entries", "n", "median r, before", "min r, before", "median r, after", "min r, after"],
               [[gname[g], v["n"], f"{pre[g]['median']:.3f}" if g in pre else "–", f"{pre[g]['min']:.3f}" if g in pre else "–",
                 f"{v['median']:.3f}", f"{v['min']:.3f}"] for g, v in sx["groups"].items()])
    L += ["", "Lowest agreement after the re-simulation: " + "; ".join(f"{r} {x:.3f}" for r, x in sx["worst"]) + ". "
          f"{sx['worst'][0][0]}, an other-form decoy, lists only one of its disorder components in the CSD.", ""]
    return L


def s2_labels() -> list:
    au, ac = EX["wrong_top1_audit"], ROB["anchor_checks"]
    ls = ROB["label_sensitivity"]
    L = ["## S2 Ground-truth checks", "",
         f"Wrong first-ranked entries: {au['n_cases']} cases (five classical scores, both answer sets, final score matrices) "
         f"were compared with the pattern's anchors by packing similarity ({au['n_pairs']} pairs). Same form: " +
         "; ".join(sorted({f"{c['wrong']} for {c['id']} ({c['stem']})" for c in au["same_form_cases"]})) + ". The label "
         "correction applied throughout the article: " + "; ".join(f"{k}: {v}" for k, v in D["label_corrections"].items()) + ".",
         "",
         f"Every anchor: the same-form step compares each family member with the two anchors closest to the deposited structure. "
         f"Comparing every non-same-form member with every remaining anchor ({ac['every_anchor']['comparisons']:,} comparisons "
         f"over the {ac['every_anchor']['patterns_checked']} patterns with more than two anchors) adds "
         + "; ".join(f"{x['refcode']} for {x['id']} (15/15 against {x['anchor']}, RMSD {x['rmsd']:.2f} Å; previously "
                     f"{x['was'].replace('_', ' ')})" for x in ac["new_same_form"]) +
         ". The labels are kept as built; the variant that counts it is a sensitivity check (Table "
         f"S{_T[0] + 2}).", "",
         f"Lenient anchors: for each of the {len(ac['lenient_rows'])} patterns anchored in the lenient pass, the molecular "
         "components of the deposited (P1, bare-coordinate) structure and of its first anchor, and which single relaxation of "
         f"the strict comparison already gives 15/15. Patterns whose component element sets differ from an anchor's: "
         f"{ac['lenient_component_mismatches']}, so no lenient anchor pairs a solvate with an anhydrate or a salt with its free "
         "form.", "", f"{T()} The lenient-anchor patterns.", ""]
    flag = {"ignore_bond_counts": "bond counts", "ignore_hydrogen_counts": "H counts", "ignore_bond_types": "bond types",
            "ignore_smallest_components": "smallest components"}
    L += table(["opXRD pattern", "H omitted", "deposited structure components", "first anchor (of n)", "anchor components",
                "single relaxation giving 15/15"],
               [[r["id"].replace(".json", ""), "yes" if r["h_omitted"] else "", "; ".join(r["opxrd_components"]),
                 f"{r['anchor']} ({r['n_anchors']})", "; ".join(r["csd_components"]),
                 ", ".join(flag[f] for f in r["single_flag_15"]) or "none alone (more than one needed)"]
                for r in ac["lenient_rows"]])
    L += ["", f"{T()} Label sensitivity, compound-weighted top-1 (single-crystal set / any same-form): all patterns; "
          f"without the {ls['n_lenient_patterns']} lenient-anchor patterns; without the 3 patterns of unrecorded wavelength; "
          f"ambient-pressure single-crystal patterns only.", ""]
    rows = []
    for k in CLS + ["model_bkg"]:
        rows.append([NAMES[k]] + [f"{f2(ls['subsets'][s][f'sc|{k}']['top1'])} / {f2(ls['subsets'][s][f'std|{k}']['top1'])}"
                                  for s in ("all", "strict_only", "known_wavelength", "ambient")])
    L += table(["score", "all", "strict anchors only", "known wavelength", "ambient"], rows)
    ea = ROB["every_anchor_variant"]["macro"]
    L += ["", f"{T()} Every-anchor variant (EPIVEF counted as a correct answer for its pattern), compound-weighted top-1 "
          "[95 % CI].", ""]
    L += table(["score", "single-crystal", "any same-form"],
               [[NAMES[k], mci(ea[f"sc|{k}"]), mci(ea[f"std|{k}"])] for k in CLS + ["model_bkg"]])
    U = EX["unknown_in_library"]
    L += ["", f"Entries that cannot be compared: {EX['unknown_entries']['total']} pattern–entry pairs of "
          f"{U['distinct_all']} entries over {EX['unknown_entries']['patterns_with_any']} patterns; {U['pairs']} pairs of "
          f"{U['distinct']} entries over {U['patterns']} patterns concern library entries and are removed from those rankings, "
          f"the rest concern entries outside the library. Self-entry reproducing the deposited structure within "
          f"0.05 Å RMSD: {EX['own_powder_rmsd_lt_0p05']['strict']} strict + {EX['own_powder_rmsd_lt_0p05']['lenient']} "
          f"lenient anchors. Within-family chance: {f2(EX['within_family_chance'])}.", ""]
    return L


def s3_hubness() -> list:
    L = ["## S3 Hubness and its correction", "", f"{T()} k-occurrence skewness (N₁₀) over the probe sets.", ""]
    L += table(["score", "skew, degraded probes", "skew, clean probes", "max N₁₀ (degraded)"],
               [[NAMES[k], f"{v['N10_skew_aug']:.1f}", f"{v['N10_skew_sim']:.1f}", v["N10_max_aug"]] for k, v in P["hubness"].items()])
    L += ["", f"{T()} Paired change in top-1 from the primary correction (CSLS, k = 10, degraded probes); 95 % "
          "compound-bootstrap CIs. Fixed/broken: patterns that the correction moves to or from rank 1.", ""]
    rows = []
    for key, v in P["paired_csls"].items():
        td, k = key.split("|")
        rows.append([NAMES[k], "any same-form" if td == "std" else "single-crystal",
                     f"{v['macro_diff']:+.3f} [{v['macro_diff_ci'][0]:+.3f}, {v['macro_diff_ci'][1]:+.3f}]",
                     f"{v['pattern_diff']:+.3f} [{v['pattern_diff_ci'][0]:+.3f}, {v['pattern_diff_ci'][1]:+.3f}]",
                     sum(c for _, c in v["fixed"]), sum(c for _, c in v["broken"])])
    L += table(["score", "answer set", "macro difference", "per-pattern difference", "fixed", "broken"], rows)
    L += ["", f"{T()} Every correction computed (none was selected after the fact). Each cell: compound-weighted top-1 (any "
          "same-form) / per-pattern top-1 (any same-form) / compound-weighted top-1 (single-crystal).", ""]
    rows = [[nm] + [f"{f2(P['sensitivity'][f'{c}|{k}']['macro_top1'])} / {f2(P['sensitivity'][f'{c}|{k}']['top1'])} / "
                    f"{f2(P['sensitivity'][f'{c}|{k}']['sc_macro_top1'])}"
                    for k in ["pearson_bkg", "cos_zs_bkg", "wcc_1_bkg", "wcc_2_bkg", "lnq_bkg", "model_bkg"]]
            for c, nm in CORR.items()]
    L += table(["correction", "Pearson", "Cosine + zs", "WCC 1°", "WCC 2°", "ln q", "model"], rows)
    return L


def s4_results() -> list:
    L = []
    for td, title in (("std", "any same-form entry is correct"), ("sc", "only independent single-crystal entries are correct")):
        rows = []
        for k in CLS + ["model_bkg", "pearson_raw", "cos_zs_raw", "wcc_1_raw", "wcc_2_raw", "model_raw"]:
            x = B[f"none|{td}|{k}"]
            rows.append([NAMES[k], "bkg" if k.endswith("bkg") else "raw", f"{f2(x['macro_top1'])} {ci(x['macro_top1_ci'])}",
                         f"{f2(x['macro_top10'])} {ci(x['macro_top10_ci'])}", f"{f2(x['top1'])} {ci(x['top1_ci'])}",
                         f"{f2(x['top10'])} {ci(x['top10_ci'])}", f"{x['median_rank']:,.0f}", f2(x["frac_rank_gt_1000"])])
        L += ["", f"## S4{'a' if td == 'std' else 'b'} Full-library results, {title}", "",
              f"{T()} {B[f'none|{td}|lnq_bkg']['n']} patterns, {B[f'none|{td}|lnq_bkg']['n_compounds']} compounds, "
              f"{D['n_library']:,} entries, no hubness correction. Compound-weighted (macro) and per-pattern top-k with 95 % "
              "compound-bootstrap CIs; 'bkg' = background-subtracted query.", ""]
        L += table(["score", "query", "macro top-1", "macro top-10", "per-pattern top-1", "per-pattern top-10",
                    "median rank", "fraction rank > 1,000"], rows)
    sp = EX["sc_ambient_split"]
    L += ["", f"{T()} Single-crystal set split into ambient-pressure patterns ({sp['n_ambient_patterns']} patterns, "
          f"{sp['n_ambient_compounds']} compounds; the article's headline) and pressure-series patterns "
          f"({sp['n_pressure_series_patterns']} patterns, {sp['n_pressure_series_compounds']} compounds): compound-weighted "
          "top-1 [95 % CI], first places, median rank.", ""]
    names = {"lnq": "ln q lag + zero scan", "lnq_v1": "ln q, first version", "critic2": "VC-GPWDF re-rank of the ln q top 50",
             "wcc_1": "WCC, l = 1° + zero scan", "wcc_2": "WCC, l = 2° + zero scan", "cos_zs": "Cosine + zero scan",
             "pearson": "Pearson, fixed grid"}
    rows = [[nm_, mci(sp[k]["macro_top1"]), sp[k]["hits"], "n/a" if sp[k]["median_rank"] is None else f"{sp[k]['median_rank']:,.1f}",
             mci(sp[k]["pressure_series_macro_top1"]), sp[k]["hits_pressure_series"]] for k, nm_ in names.items()]
    rows += [[f"Learned model, {lab}", mci(sp[f"model_{x}"]["macro_top1"]), sp[f"model_{x}"]["hits"],
              f"{sp[f'model_{x}']['median_rank']:,.1f}", mci(sp[f"model_{x}"]["pressure_series_macro_top1"]), ""]
             for x, lab in CKPTS]
    L += table(["score", "ambient macro top-1", "ambient first places", "ambient median rank", "pressure-series macro top-1",
                "pressure-series first places"], rows)
    tn = {"cos_zs": "cosine + zero scan", "wcc_1": "WCC 1°", "wcc_2": "WCC 2°", "lnq": "ln q"}
    L += ["", f"Paired over ambient compounds: ln q − Pearson {dci(sp['paired_lnq_minus_pearson'])}; WCC 1° − Pearson "
          f"{dci(sp['paired_wcc_1_minus_pearson'])}; cosine + zero scan − Pearson {dci(sp['paired_vs_pearson']['cos_zs'])}; "
          f"WCC 2° − Pearson {dci(sp['paired_vs_pearson']['wcc_2'])}; re-rank − ln q {dci(sp['paired_critic2_minus_lnq'])}. "
          "Among the shift-tolerant scores: " + "; ".join(f"{tn[k.split('-')[0]]} − {tn[k.split('-')[1]]} {dci(v)}"
                                                          for k, v in sp["paired_among_tolerant"].items()) + ".", "",
          f"{T()} Within-family top-1 (rank among the compound's own forms in the library).", ""]
    L += table(["score", "query", "patterns", "top-1"],
               [[NAMES[k], "bkg" if k.endswith("bkg") else "raw", v["n"], f2(v["top1"])] for k, v in P["within_family"].items()])
    L += ["", f"{T()} The learned model as a shortlist re-ranked by a classical score: compound-weighted / per-pattern "
          "top-1 (background-subtracted).", ""]
    L += table(["pipeline", "any same-form", "single-crystal"],
               [[k.replace("model@", "model top-").replace("->", " → ").replace("_bkg", ""),
                 f"{f2(v['std']['macro_top1'])} / {f2(v['std']['top1'])}", f"{f2(v['sc']['macro_top1'])} / {f2(v['sc']['top1'])}"]
                for k, v in P["hybrid"].items()])
    L += ["", f"{T()} Library-size curve (compound-weighted top-1, any same-form entry; mean ± s.d. over five random decoy "
          "subsets).", ""]
    rows = []
    for k in ["pearson_bkg", "cos_zs_bkg", "wcc_1_bkg", "wcc_2_bkg", "lnq_bkg", "model_bkg"]:
        for c in ("none", "csls_aug_k10"):
            cv = P["library_curve_macro"][f"{c}|{k}"]
            rows.append([NAMES[k], CORR[c]] + [f"{cv[s][0]:.2f} ± {cv[s][1]:.2f}" if s != "full" else f"{cv[s][0]:.2f}"
                                               for s in ("2000", "20000", "100000", "full")])
    L += table(["score", "correction", "2,000", "20,000", "100,000", f"{D['n_library']:,}"], rows)
    L += ["", f"{T()} Paired differences in compound-weighted top-1 against fixed-grid Pearson (95 % compound-bootstrap CI).", ""]
    L += table(["score − Pearson", "answer set", "compounds", "difference"],
               [[NAMES[k.split("|")[1].split("-")[0]], "any same-form" if k.startswith("std") else "single-crystal",
                 v["n_compounds"], dci(v)] for k, v in EX["paired_methods"].items()])
    L += ["", f"{T()} Subgroups (any same-form entry): per-pattern / compound-weighted top-1.", ""]
    sub_names = {"tier_strict": "strict anchors", "tier_lenient": "lenient anchors", "corpus_entry": "a correct entry in the corpus",
                 "extra_only": "correct entries only among the added entries", "cu": "Cu Kα", "non_cu": "other wavelengths",
                 "ordered": "self-entry ordered", "disordered": "self-entry disordered"}
    L += table(["subgroup", "patterns / compounds", "ln q", "cosine + zs", "Pearson", "WCC 1°", "WCC 2°", "model"],
               [[nm_, f"{EX['subgroups'][g]['n_patterns']} / {EX['subgroups'][g]['n_compounds']}"] +
                [f"{f2(EX['subgroups'][g][k]['top1'])} / {f2(EX['subgroups'][g][k]['macro_top1'])}"
                 for k in ["lnq_bkg", "cos_zs_bkg", "pearson_bkg", "wcc_1_bkg", "wcc_2_bkg", "model_bkg"]]
                for g, nm_ in sub_names.items()])
    return L


def s5_s6() -> list:
    g = P["gpu_check"]
    L = ["", "## S5 GPU and CPU scores", "",
         f"Maximum |GPU − CPU| score over all benchmark patterns and their special entries: {g['max_abs_diff_vs_cpu_prepass']:.1e}. "
         f"Rank values that differ: {g['rank_mismatches']} (all ≤ 3 positions); top-1 changes: {g['top1_changes']}. The "
         f"corrected ln q scan rescored on the GPU against its CPU ranks: {g['lnq_v2_vs_cpu']['rank_mismatches']} rank values "
         f"differ; top-1 changes: {g['lnq_v2_vs_cpu']['top1_changes']}.", ""]
    cx = CZ["context"]
    tf = P["case"]["texture_fit"]
    L += ["## S6 The ZZZUWK case", "",
          f"ZZZUWK (propan-1-aminium chloride, P4/nmm) contributes {CZ['n_patterns']} patterns, a 303–398 K series; each is "
          f"the refinement data of its self-entry ZZZUWK01–20. Self-entry score, median over its patterns vs the other "
          "benchmark patterns: " + "; ".join(f"{k.replace('_bkg', '')} {v['own_score_median_case']:.2f} vs "
                                            f"{v['own_score_median_rest']:.2f}" for k, v in cx.items()
                                            if k in ("pearson_bkg", "cos_zs_bkg", "wcc_1_bkg")) + ".", "",
          "The cation is disordered over four orientations by symmetry, each at occupancy 0.25. Patterns (of 20) for which "
          "each score ranks a same-form entry first, with the self-entries simulated with every disorder site at full "
          "occupancy (the record of the pipeline before the re-simulation of Section S1) and with their site occupancies, "
          "without / with CSLS: " + "; ".join(
              f"{NAMES[k]} {P['case']['top1_of_20_full_occupancy'].get(f'none|{k}', '–')} / "
              f"{P['case']['top1_of_20_full_occupancy'].get(f'csls_aug_k10|{k}', '–')} before, "
              f"{P['case']['top1_of_20'][f'none|{k}']} / {P['case']['top1_of_20'][f'csls_aug_k10|{k}']} after"
              for k in CLS + ["model_bkg"]) + ". With the full-occupancy simulation, "
          f"{P['case']['wrong_top1_cos_zs_full_occupancy'][0]} was the wrong first answer of cosine + zero scan for "
          f"{P['case']['wrong_top1_cos_zs_full_occupancy'][1]} patterns. The ln q scan in its present version and the model "
          "were not scored against the full-occupancy library.", "",
          f"Texture fit: March–Dollase along each of {tf['axes_tested']} low-index axes, r from {tf['r_grid'][0]} to "
          f"{tf['r_grid'][1]} ({tf['r_grid'][2]} values), zero offset fitted per pattern. Median Pearson correlation with the "
          f"self-entry {tf['median_pearson_no_texture']:.2f} without texture and {tf['median_pearson_best_texture']:.2f} with "
          f"the best axis and r per pattern (best axes: " + ", ".join(f"{k} ×{v}" for k, v in tf["best_axis_counts"].items()) +
          f"; median best r {tf['median_best_r']}).", "",
          f"{T()} Relative peak heights (fraction of the maximum, mean over the 20 patterns): measured, simulated for the "
          "self-entry without texture and with the fitted texture; angle between the reflection normal and c*.", ""]
    L += table(["hkl", "2θ (°)", "angle to c* (°)", "measured", "simulated", "simulated, textured", "measured / simulated"],
               [[r["hkl"], f"{r['two_theta']:.2f}", f"{r['angle_to_c_star_deg']:.1f}", f"{r['measured_mean']:.2f}",
                 f"{r['sim_mean']:.2f}", f"{r['sim_textured_mean']:.2f}", f"{r['ratio_measured_over_sim']:.2f}"]
                for r in tf["reflections"]])
    rf = {r["hkl"]: r["ratio_measured_over_sim"] for r in tf["reflections"]}
    L += ["", "(101) and (202) share a plane normal, so any texture model scales them alike; their measured/simulated ratios "
          f"differ by a factor of about {rf['(1, 0, 1)'] / rf['(2, 0, 2)']:.1f}, so the mismatch is not preferred orientation "
          "alone.", "",
          f"{T()} Decoys that win ZZZUWK patterns: patterns (of 20) in which the decoy outscores every same-form entry, with the "
          "zero scan, at zero offset only and under fixed-grid Pearson; median best offset.", ""]
    rows = []
    for m, per in CZ["methods"].items():
        for rc, v in per.items():
            if v["role"] == "decoy":
                rows.append([m.replace("cos_zs", "cosine + zero scan").replace("wcc_1", "WCC l = 1°"), rc,
                             cx["reflections_5_50"]["decoys"][rc], v["beats_truth_with_scan"], v["beats_truth_at_offset0"],
                             v["beats_truth_pearson"], f"{v['best_offset_deg_median']:+.2f}"])
    L += table(["score", "decoy", "reflections 5–50°", "with scan", "zero offset", "Pearson", "best offset (°)"], rows)
    return L


def s7_model() -> list:
    m0 = M[MODEL]
    tr = m0["train"]
    mp = EX["model_probe_check"]
    ov = P["model_training_overlap"]
    L = ["", "## S7 Learned model", "",
         f"Model card. Dual encoder (figure 2 of the article): a one-dimensional residual CNN over the standardised 2,251-point "
         "pattern; a Transformer (4 layers, width 256, 8 heads, pre-norm) over one lattice token, one token per "
         "asymmetric-unit atom and one token per CIF symmetry operator; 256-dimensional unit embeddings; "
         f"{P['run_facts']['model_params_millions']} million parameters. Symmetric InfoNCE with a learnable temperature, AdamW "
         f"(learning rate {tr['lr']:g}, weight decay {tr['weight_decay']:g}, {tr['warmup']} warm-up epochs), batch "
         f"{tr['batch_size']}, {tr['epochs']} epochs; sibling polymorphs in the same batch; scaffold- and polymorph-family "
         "split 247,089 / 30,887 / 30,886. Training simulation: zero offset, isotropic strain, Caglioti and Scherrer "
         "broadening, March–Dollase texture averaged over the symmetry-equivalent reflections of each peak, background and "
         "Poisson noise, plus Kα2 (probability 0.6, I₂/I₁ = 0.5, each copy at its own wavelength under strain), an overall "
         "Debye–Waller factor (B uniform in 0–8 Å²), log-normal intensity jitter per reflection (σ uniform in 0–0.5), 2θ "
         "range cuts (probability 0.25) and the background subtraction of the measured patterns. Checkpoint: the epoch with "
         "the best in-batch top-1 on a fixed degraded draw of the validation structures. The checkpoints were trained on CSD "
         "data.", "", f"{T()} The four checkpoints. Real benchmark: compound-weighted top-1 [95 % CI] and median rank; "
         "ladder: the self-entries rendered with the first six effects and the Debye–Waller factor; sweep: within-family "
         "top-1 range over the 42 distinct conditions.", ""]
    rows = []
    for x, lab in CKPTS:
        r = M[x]
        lad = r["ladder"]
        rows.append([lab, "–" if x == "poly" else f"{r['best_degraded_val_top1']:.3f} (epoch {r['best_epoch']})",
                     mci(r["sc_macro_top1"]), f"{f2(r['ambient_sc_macro_top1'])}", mci(r["std_macro_top1"]),
                     f"{r['std_median_rank']:,.1f}", "–" if x == "poly" else dci(r["paired_std_vs_original"]),
                     f"{f2(lad['own_all_dw|model|std']['macro_top1'])} / {lad['own_all_dw|model|std']['median_rank']:,.1f}",
                     f"{f2(lad['measured|p2p|std']['macro_top1'])}",
                     f"{f2(r['sweep_model_range'][0])}–{f2(r['sweep_model_range'][1])}"])
    L += table(["checkpoint", "degraded validation top-1", "single-crystal, macro top-1", "ambient single-crystal",
                "any same-form, macro top-1", "median rank", "− earlier model (any same-form)", "ladder: six + B",
                "pattern to pattern", "sweep"], rows)
    L += ["", f"{T()} The model (symmetry-aware, seed 0) on simulated and measured queries against all {mp['n_library']:,} entries. Probes: "
          "library entries outside the benchmark compounds, their own refcode family excluded. Self-entries: the clean "
          "simulated pattern of each benchmark pattern's self-entry, scored with the measured pattern's answer sets.", ""]
    rows = [["clean simulated probes", mp["sim"]["n"], f2(mp["sim"]["top1"]), f2(mp["sim"]["top10"]), mp["sim"]["median_rank"]],
            ["degraded simulated probes", mp["aug"]["n"], f2(mp["aug"]["top1"]), f2(mp["aug"]["top10"]), mp["aug"]["median_rank"]]]
    for nm_, k in (("self-entries, clean simulated", "own_entry_clean"), ("measured patterns", "measured_bkg")):
        for td, tl in SETS:
            v = mp[f"{k}|{td}"]
            rows.append([f"{nm_} ({tl})", v["n"], f2(v["top1"]), f2(v["top10"]), v["median_rank"]])
    L += table(["query", "n", "top-1", "top-10", "median rank"], rows)
    ns = ov["patterns_with_same_form_entry_in"]
    se = P["symops_sweep_effect"]
    L += ["", f"Operator tokens in the simulated sweep (each seed minus the ablation, within-family top-1 over the "
          f"{se[MODEL]['n_conditions']} distinct conditions): seed 0 mean {sg(se[MODEL]['mean_diff'])} (range "
          f"{sg(se[MODEL]['min_diff'])} to {sg(se[MODEL]['max_diff'])}, higher in {se[MODEL]['n_positive']} conditions), seed 1 mean "
          f"{sg(se['d3_sym_s1']['mean_diff'])} (range {sg(se['d3_sym_s1']['min_diff'])} to {sg(se['d3_sym_s1']['max_diff'])}, higher in "
          f"{se['d3_sym_s1']['n_positive']}); " + ("in no condition do the 95 % family-bootstrap intervals separate"
          if se[MODEL]['n_ci_separated'] == se['d3_sym_s1']['n_ci_separated'] == 0 else
          f"the 95 % family-bootstrap intervals separate in {se[MODEL]['n_ci_separated']} and {se['d3_sym_s1']['n_ci_separated']} "
          "conditions") + "; mean absolute difference between the two seeds "
          f"{se['seed_mean_abs_diff']:.3f}.", ""]
    isp = EX["instrument_split"]
    gl = {"lab_cu": "laboratory Cu", "sync_ambient": "synchrotron, ambient", "sync_pressure": "synchrotron, pressure series",
          "other": "Mo, Co and other"}
    L += [f"{T()} By instrument (exploratory; prompted by the external set of Section S14), any same-form entry: patterns ranked "
          "first / within the top 10 / within the top 100 / median rank for the learned models; patterns ranked first and "
          "median rank for the classical scores.", ""]
    L += table(["checkpoint or score"] + [f"{gl[g]} ({isp['groups'][g]['n']} / {isp['groups'][g]['n_compounds']})" for g in gl],
               [[lab] + [f"{isp[x][g]['top1']} / {isp[x][g]['top10']} / {isp[x][g]['top100']} / {isp[x][g]['median_rank']:,.1f}"
                         for g in gl] for x, lab in CKPTS] +
               [[NAMES[k]] + [f"{isp[k][g]['top1']} / {isp[k][g]['median_rank']:,.1f}" for g in gl]
                for k in ("lnq_bkg", "cos_zs_bkg", "pearson_bkg", "wcc_1_bkg", "wcc_2_bkg")]) + [""]
    L += [f"Training-split control: the crystal form of {ns.get('train', 0)} patterns has an entry in the training split, "
          f"{ns.get('val', 0)} in the validation split, {ns.get('test', 0)} in the test split and {ns.get('none', 0)} in no split. "
          f"Model top-1 / median rank for patterns whose form is in the training split "
          f"{f2(ov['model_bkg_train']['top1'])} / {ov['model_bkg_train']['median_rank']:,.0f} (n = {ov['model_bkg_train']['n']}) and "
          f"in no split {f2(ov['model_bkg_none']['top1'])} / {ov['model_bkg_none']['median_rank']:,.0f} (n = {ov['model_bkg_none']['n']}). "
          "Memorised training structures would show up as better retrieval of the first group; the control finds none. "
          f"Recall of the model's top 100 / 1,000: {f2(EX['model_recall']['100'])} / {f2(EX['model_recall']['1000'])}.", ""]
    L += ["", f"{T()} Search cost per query over the full library. CPU: per-pair wall time of the simulated sweep (one core) "
          f"times {D['n_library']:,}; with precomputed embeddings the model searches the library in about "
          f"{P['run_facts']['model_query_ms_cpu']} ms per query.", ""]
    cpu_names = {"pearson": "Pearson, fixed grid", "pearson_zs": "Pearson + zero scan", "wcc_1": "WCC, l = 1° + zero scan",
                 "wcc_2": "WCC, l = 2° + zero scan", "lnq": "ln q lag + zero scan"}
    L += table(["score", "CPU s per query (estimate)"],
               [[cpu_names[k], f"{v:,.1f}"] for k, v in EX["cpu_s_per_query_full_library"].items() if k != "model"])
    return L


CONV_MAX = max(abs(ROB["conv"]["paired_vs_main"][f"{td}|{k}"]["top1"]["diff"]) for td in ("std", "sc") for k in CLS)
DEVIATIONS = [
    ["learned model", f"earlier model [ref. {REFN}]: no symmetry information, one seed, checkpoint on clean validation",
     "operator tokens for the space group; two seeds; checkpoint on degraded validation; ablation without the tokens",
     "the encoder could not know absences, multiplicities or Z; a clean criterion selects against robustness"],
    ["Kα2 in the training simulation", "a realism retrain whose Kα2 copies collapsed back onto Kα1 whenever strain was "
     "applied (every draw)", "each Kα2 copy keeps its own wavelength under strain (unit test through the training path)",
     "code defect found while re-checking the training simulation"],
    ["March–Dollase texture", "applied to the stored representative reflection of each merged peak; (001) axis for every "
     "structure, r ≤ 1", "averaged over the symmetry-equivalent reflections of each peak; (001) and each structure's BFDH axis; "
     "plates and needles (r 0.5–2)", "the single-reflection factor is wrong for higher symmetry (cubic {100}, r = 0.5: "
     "0.35 instead of 2.90); referee request"],
    ["labels", "as built", "GOFWUV family added to one QQQAUJ pattern (wrong-top-1 audit on the final score matrices); "
     "every-anchor and lenient-anchor checks reported", "a same-form entry under another refcode was scored as an error"],
    ["headline answer set", "all 59 single-crystal patterns", "ambient single-crystal patterns (38); pressure series "
     "reported separately", "the pressure series are not the routine case of a new measurement checked against a database"],
    ["intensity convention", "positions converted to Cu, intensities not", "unchanged; the Lorentz–polarisation and "
     "Jacobian conversion is reported as a sensitivity", f"changes no classical score by more than {CONV_MAX:.3f}"],
    ["variable-cell re-rank", "59 shortlists", "60 (the label correction added one single-crystal pattern)", "label correction"],
    ["practitioner baseline", "none", "library rescaled to 295 K by its recorded temperatures",
     "references determined at other temperatures are commonly rescaled to room temperature"],
    ["compound (unit of weighting and of the bootstrap)", "CSD refcode family (123 compounds)",
     "refcode families whose same-form sets share an entry are one compound (GOFWUV with QQQAUJ, IYESEO with QIQVAQ; "
     f"{D['n_compounds']} compounds; also applied to the COD replica)", "two compounds were counted twice"],
    ["disordered added library entries", "simulated with every disorder site at full occupancy (the CIFs written by the CSD "
     "Python API carry no occupancies)", f"re-simulated with the CSD site occupancies or as the major component, whichever "
     f"agrees better with the CCDC simulation ({P['simulation_check']['n_resimulated']} entries); every score cache and "
     "evaluation recomputed", "an independent check against the CCDC simulation (ZZZUWK median r "
     f"{P['simulation_check']['pre_occ_groups']['zzzuwk']['median']:.2f} before, "
     f"{P['simulation_check']['groups']['zzzuwk']['median']:.2f} after)"],
    ["ZZZUWK case", "attributed to texture (relative intensities far from the simulation)", "attributed to the full-occupancy "
     "simulation; with occupancies cosine + zero scan and WCC with l = 1° rank a same-form entry first for all 20 patterns "
     "instead of none", "the disordered entries above"],
    ["open replica", "model embeddings of the earlier model", "COD symmetry operators from each CIF (from the space-group "
     "symbol for CIFs without an operator list; operator lists quoted inside comment fields are ignored)",
     "the model needs the operators; a quoted list had been read for 3 library entries, none of them an answer"],
]


def s8_sweep() -> list:
    g = P["g0"]
    L = ["", "## S8 Simulated sweep, all conditions", "",
         f"{T()} Within-family top-1 over {g['n_families']:,} families ({g['n_queries']:,} queries); 95 % family-bootstrap "
         f"CIs; composition-only chance {g['chance']:.2f}; the model is the symmetry-aware encoder, seed 0. Other checkpoints: model top-1 only. The last "
         "column is the first version of the ln q scan on the same draws.", ""]
    oth = {x: {r["condition"]: r for r in P["sweep_rows_other"][x]} for x, _ in CKPTS[1:]} if "sweep_rows_other" in P else {}
    rows = []
    for r in g["rows"]:
        rows.append([r["condition"], f"{r['model']:.2f} {ci(r['model_ci'])}"] +
                    [f"{oth[x][r['condition']]['model']:.2f}" if oth else "" for x, _ in CKPTS[1:]] +
                    [r["best"], f"{r['best_top1']:.2f} {ci(r['best_ci'])}", f"{r['delta']:+.2f}", f"{r['pearson_fixed']:.2f}",
                     f"{r['lnq']:.2f}", f"{r['lnq_v1']:.2f}"])
    L += table(["condition", "model", "seed 1", "no operator tokens", "earlier model", "best classical score", "best top-1",
                "Δ", "Pearson, fixed grid", "ln q", "ln q, first version"], rows)
    sn = {"pearson_zs": "Pearson + zero scan", "wcc_0.5": "WCC 0.5°", "wcc_1": "WCC 1°", "wcc_2": "WCC 2°", "lnq": "ln q",
          "pearson": "Pearson, fixed grid"}
    L += ["", "A zero offset of 0° repeats the clean condition, so 42 conditions are distinct. Texture along each structure's "
          "Bravais–Friedel–Donnay–Harker axis: r < 1 plates, r > 1 the opposite sense along the same axis. Conditions in which "
          "each score beats the model with non-overlapping intervals: " +
          ", ".join(f"{sn[k]} {v}" for k, v in g["conditions_each_score_beats_model"].items()) + ". Each condition's random draws "
          "are seeded from a CRC-32 checksum of its name, and every checkpoint scores identical draws (checked by the numbers "
          "script)."]
    return L


def s9_s10() -> list:
    L = ["", "## S9 The first version of the ln q scan", "",
         "The first version sampled the ln q axis at points and used 16 zero offsets that skipped zero (nearest ±0.02°); the "
         "corrected version averages each ln q bin over its width and uses 17 offsets from −0.32° to +0.32°. The article "
         "reports the corrected version throughout. The corrected version was declared primary before it was scored.", "",
         f"{T()} ln q lag + zero scan, first and corrected versions, on the same patterns. Paired differences over compounds.", ""]
    rows = []
    for td, nm_ in SETS:
        v = EX["lnq_v1"][td]
        cur = B[f"none|{td}|lnq_bkg"]
        rows.append([nm_, mci(v["v1"]["macro_top1"]), f"{f2(cur['macro_top1'])} {ci(cur['macro_top1_ci'])}",
                     dci(v["paired_v2_minus_v1"]["top1"]), dci(v["paired_v2_minus_v1"]["top10"]), v["n_rank_changed"],
                     v["n_top1_changed"]])
    L += table(["answer set", "first version, macro top-1", "corrected, macro top-1", "corrected − first, top-1",
                "corrected − first, top-10", "patterns whose rank changes", "top-1 changes"], rows)
    c2, c2r = EX["critic2"], P["critic2_rerank"]
    L += ["", "## S10 Variable-cell re-rank (critic2 VC-GPWDF)", "",
          f"critic2 1.4.155 COMPAREVC GLOBAL QUICK, α = 1, measurement wavelength, on the ln q top {c2['k']} of each of the "
          f"{c2['n_patterns']} single-crystal patterns ({c2['n_comparisons']:,} comparisons, excluded entries left out). Median "
          f"{c2['cpu_min_median']:.1f} CPU-minutes per comparison, maximum {c2['cpu_h_max']:.2f} CPU-hours; "
          f"{c2['n_failed']} {'comparison' if c2['n_failed'] == 1 else 'comparisons'} reached the 4-hour wall-time limit and "
          "count as maximally dissimilar. The answer is "
          f"in the shortlist for {c2['n_answer_in_shortlist']} patterns ({c2['n_answer_in_lnq_top100']} in the ln q top 100). "
          "Entries with a disorder model are passed as their major component.", "",
          f"All single-crystal patterns: re-rank {mci(c2r['critic2_vcgpwdf']['top1'])}, ln q "
          f"{mci(c2r['lnq_v2_full_library']['top1'])}, paired {dci(c2r['paired_critic2_minus_lnq']['top1'])}.", "",
          "Positive control: each pattern against its own self-entry; its DIFF is ranked among the wrong entries of the "
          "shortlist. Injection: each answer missing from a shortlist (the single-crystal same-form entry closest in cell) "
          "compared with its pattern and ranked the same way. Pressure series: the self-entry carries a CSD pressure, or "
          "another entry of its family from the same publication does, or it is flagged by hand with the evidence. Cell "
          "difference: the largest relative difference of the primitive Niggli-reduced cell lengths of the self-entry and the "
          "closest single-crystal entry; ΔT: the difference of their recorded temperatures.", "",
          f"{T()} The single-crystal patterns. Ranks are 1-based; '–': the answer is not in the ln q top {c2['k']}.", ""]

    def cell(v, fmt):
        return "n/a" if v is None else format(v, fmt)
    L += table(["opXRD pattern", "CSD family", "pressure series", "cell difference (%)", "ΔT (K)", "peaks", "ln q rank",
                "rank after the re-rank", "self-entry, rank among wrong entries", "injected answer, rank"],
               [[r["id"], r["stem"], "yes" if r.get("pressure_series") else "", cell(r["cell_diff_pct"], ".2f"),
                 cell(r.get("dT_K"), ".0f"), r["n_peaks"], f"{r['lnq_rank']:,}", "–" if r["critic2_rank"] is None else r["critic2_rank"],
                 cell((r.get("control") or {}).get("rank_in_shortlist"), "d"),
                 cell((r.get("injected") or {}).get("rank_if_shortlisted"), "d")]
                for r in sorted(c2["per_pattern"], key=lambda r: (r["stem"], int(r["id"].split("_")[1])))])
    cs, isu = c2["control_summary"], c2["injected_summary"]
    sp = EX["sc_ambient_split"]
    cd = sp["cells_differ"]
    L += ["", f"Positive control: self-entry ranked first for {cs['ambient']['rank1']} of {cs['ambient']['n']} ambient and "
          f"{cs['pressure_series']['rank1']} of {cs['pressure_series']['n']} pressure-series patterns. Injection: the missing "
          f"answer would rank first for {isu['ambient']['rank1']} of {isu['ambient']['n']} ambient and "
          f"{isu['pressure_series']['rank1']} of {isu['pressure_series']['n']} pressure-series patterns. Fitted peaks: "
          f"{c2['n_peaks_range']['ambient'][0]}–{c2['n_peaks_range']['ambient'][1]} (ambient) and "
          f"{c2['n_peaks_range']['pressure_series'][0]}–{c2['n_peaks_range']['pressure_series'][1]} (pressure series). "
          f"Patterns whose cells differ by more than 1 % ({cd['n']}): pressure series {cd['pressure_series']}; recorded "
          f"temperatures more than 50 K apart (not pressure series) {cd['dT_gt_50_not_pressure']}; supercell description "
          f"{cd['supercell']}; none of these {cd['none_of_these']}" +
          (f" ({', '.join(cd['none_of_these_ids'])})" if cd["none_of_these_ids"] else "") + "."]
    x2, x3 = EX["x2_zero_offset"], EX["x3_dac_refit"]
    dz = x2["delta"]
    L += ["", "### Sensitivity of the re-rank to the zero offset and to the peak fits", "",
          "Zero offset: each single-crystal pattern gets an answer-blind zero offset δ, the native shift (±0.5°, steps of 0.02°, "
          "both scaled by λ/λ(Cu)) that maximises the cosine between the background-subtracted measured pattern on the Cu grid "
          "and the clean rendering of its ln q top-1 entry. The fitted peak positions are shifted by −δ and the shortlists that "
          f"hold the answer are rerun ({x2['n_comparisons_rerun']:,} comparisons). Median |δ| {dz['median_abs_lab_cu_deg']:.2f}° "
          f"for the {dz['n_lab_cu']} laboratory Cu patterns and {dz['median_abs_cu_equiv_deg']:.2f}° (Cu-equivalent) over all "
          f"{dz['n']}; {dz['n_at_edge']} offsets lie at the edge of the scan.", "",
          f"Peak fits: {len(x3['control'])} patterns, all from pressure series, start their measured range with a run of "
          "zero-intensity points. Those points are dropped and the patterns refitted with critic2's default settings; their "
          f"shortlists ({x3['n_comparisons_rerun']:,} comparisons), self-entry controls and answer injections are rerun.", "",
          f"{T()} The re-rank with answer-blind zero offsets and with the refitted pressure-series patterns, against the main "
          "re-rank (paired over compounds).", ""]
    L += table(["re-rank", "comparisons rerun", "macro top-1", "− main, top-1", "first places", "patterns whose rank changes"],
               [["main", "", mci(c2r["critic2_vcgpwdf"]["top1"]), "", x2["hits_main"], ""]] +
               [[nm_, f"{v['n_comparisons_rerun']:,}", mci(v["top1"]), dci(v["paired_minus_main"]["top1"]), v["hits"],
                 v["n_rank_changed"]] for nm_, v in (("zero offsets", x2), ("refitted pressure series", x3))])
    L += ["", f"{T()} The refitted patterns, before → after ('–': the answer is not in the shortlist; 'n/a': no injection).", ""]

    def ba(a, b, fmt="d"):
        f_ = (lambda v: "n/a" if v is None else format(v, fmt))
        return f"{f_(a)} → {f_(b)}"
    inj = x3["injected"]
    L += table(["opXRD pattern", "CSD family", "self-entry", "peaks", "self-entry DIFF", "self-entry, rank among wrong entries",
                "injected answer, rank", "rank after the re-rank"],
               [[r["id"], r["stem"], r["own"], ba(r["old_n_peaks"], x3["n_peaks"][r["id"]]), ba(r["old_diff"], r["diff"], ".3f"),
                 ba(r["old_rank_in_shortlist"], r["rank_in_shortlist"]),
                 ba(inj[r["id"]]["old_rank_if_shortlisted"], inj[r["id"]]["rank_if_shortlisted"]) if r["id"] in inj else "n/a",
                 " → ".join("–" if v is None else str(v) for v in (r["old_rerank_rank"], r["rerank_rank"]))]
                for r in sorted(x3["control"], key=lambda r: (r["stem"], int(r["id"].split("_")[1])))])
    return L


def s11_ladder() -> list:
    lad = P["ladder"]
    est = P["ladder_estimates"]
    L = ["", "## S11 Effect ladder", "",
         f"Each pattern's self-entry, rendered with effects estimated from its measurement: its 2θ coverage "
         f"({est['n_partial_coverage']} patterns do not cover 5–50°), its peak width (median FWHM {est['fwhm_deg_median']:.2f}°), "
         f"Kα2 for the {est['n_cu_lab']} Cu patterns, its own background followed by the background subtraction, noise at its "
         f"level and its best-aligning zero offset (median |offset| {est['zero_deg_abs_median']:.2f}°); B = 4.45 Å². Scored with "
         "the measured pattern's answer sets. 'p2p' matches the variant against the clean library patterns in the model's "
         "pattern space.", ""]
    for td, nm_ in SETS:
        L += [f"{T()} Effect ladder, {nm_} answer set: compound-weighted top-1 / median rank.", ""]
        rows = []
        for rung, lab in LADDER:
            row = [lab]
            for x, _ in CKPTS:
                res = lad[x]["results"]
                k = f"{rung}|model|{td}"
                row.append("n/a" if k not in res else f"{f2(res[k]['macro_top1']['value'])} / {res[k]['per_pattern']['median_rank']:,.1f}")
            k = f"{rung}|pearson|{td}"
            row.append(f2(lad[MODEL]["results"][k]["macro_top1"]["value"]) if k in lad[MODEL]["results"] else "n/a")
            k = f"{rung}|p2p|{td}"
            row.append(f2(lad[MODEL]["results"][k]["macro_top1"]["value"]) if k in lad[MODEL]["results"] else "n/a")
            rows.append(row)
        L += table(["rendering"] + [lab for _, lab in CKPTS] + ["Pearson", "pattern to pattern, seed 0"], rows) + [""]
    return L


def s12_robustness() -> list:
    n_nc = D["n_queries"] - D["n_lab_cu"]
    L = ["## S12 Robustness of the benchmark", "",
         "Compound-weighted top-1 [95 % CI] and the difference from the reference, paired over compounds. Lorentz–polarisation "
         f"and Jacobian: the Lorentz–polarisation factor and the area-conserving Jacobian of the conversion to Cu applied to the "
         f"{n_nc} non-Cu patterns. Grid-step averaging: measured patterns put on the Cu grid by averaging over each grid step "
         "(a rule fixed before the run: a change above 0.05 in any classical score would lead us to consider it as the "
         "primary analysis; only WCC with l = 2° crosses it, by +0.06 on the ambient set, with an interval that reaches zero, "
         "and interpolation was kept, Section 4.7 of the article). Library at 295 K: every "
         "library entry re-rendered with an isotropic strain from its recorded temperature to 295 K, scored with the "
         "Lorentz–polarisation and Jacobian queries. Library package: the changes of Section 3.8 of the article; its one-off "
         "ablations are paired against the full package. The label variants (strict anchors, recorded wavelengths, every-anchor "
         "comparison) are in Section S2.", ""]
    amb_t = ROB["table2"]
    t2_lab = {"main": "main benchmark", "conv": "Lorentz–polarisation and Jacobian", "known_wavelength": "recorded wavelength only",
              "strict": "strict anchors only", "every_anchor": "EPIVEF counted (every-anchor comparison)",
              "r2": "library package", "dT": "library at 295 K (with Lorentz–polarisation and Jacobian)",
              "binavg": "grid-step averaging"}
    L += [f"{T()} The ambient single-crystal set under every variant (table 2 of the article): compound-weighted top-1. "
          f"Patterns: {ROB['table2_n']['ambient_patterns']} ({ROB['table2_n']['strict_ambient_patterns']} with strict anchors, "
          f"{ROB['table2_n']['known_wavelength_ambient_patterns']} with a recorded wavelength, "
          f"{ROB['table2_n']['every_anchor_ambient_patterns']} with EPIVEF counted).", ""]
    L += table(["variant"] + [NAMES[k] for k in CLS + ["model_bkg"]],
               [[t2_lab[v]] + [f2(amb_t[v][k]["ambient"]) for k in CLS + ["model_bkg"]] for v in t2_lab]) + [""]
    labs = {"conv": "Lorentz–polarisation and Jacobian", "binavg": "grid-step averaging", "dT": "library at 295 K",
            "r2": "library package", "main": "main benchmark"}
    for tag, ref in (("conv", "main"), ("binavg", "main"), ("dT", "conv"), ("r2", "main")):
        v = ROB[tag]
        L += [f"{T()} {labs[tag].capitalize()} (reference: {labs[ref]}).", ""]
        L += table(["score", "answer set", "top-1", "− reference", "first places changed"],
                   [[NAMES[k], nm_, mci(v["macro"][f"{td}|{k}"]["top1"]), dci(v[f"paired_vs_{ref}"][f"{td}|{k}"]["top1"]),
                     v[f"paired_vs_{ref}"][f"{td}|{k}"]["n_top1_changed"]] for k in CLS + ["model_bkg"] for td, nm_ in SETS]) + [""]
    L += [f"{T()} One-off ablations of the package: compound-weighted top-1 difference from the full package.", ""]
    ab = ROB["r2_ablations"]
    L += table(["score", "answer set", "− B", "− LP", "− Kα2", "− range"],
               [[NAMES[k], nm_] + [dci(ab[t]["paired_vs_r2"][f"{td}|{k}"]) for t in ("r2_noB", "r2_nolp", "r2_noka2", "r2_nomask")]
                for k in CLS for td, nm_ in SETS])
    L += ["", f"Paired intervals of the package and its ablations that exclude zero: {ROB['r2_n_intervals_excluding_zero']} of "
          f"{ROB['r2_n_intervals']}."]
    return L


def s13_s14() -> list:
    cod = ROB["cod"]
    cx = cod["construction"]
    L = ["", "## S13 The open replica (COD)", "",
         f"Library: {cx['n_selected']:,} COD entries pass the corpus filter and {cx['n_added_candidates']} candidate entries of "
         f"the queries are added ({cx['n_all']:,}); {cod['meta']['n_library']:,} gave stick patterns. Identified: "
         f"{cx['n_identified']} of {cx['n_patterns']} patterns ({cx['n_identified_both']} also identified in the CSD, "
         f"{cx['n_identified_cod_only']} only in the COD). Other-compound candidates kept as library entries: "
         f"{cx['n_other_compound']:,}; same-formula candidates excluded as not comparable: {cx['n_unknown_same_formula']}. "
         f"Scored: {cod['meta']['n_queries']} patterns. The model's structure embeddings use each entry's symmetry operators, "
         "read from its CIF, or generated from the Hall (else H-M) symbol for CIFs that give no operator list.", "",
         f"{T()} Retrieval on the open replica (background-subtracted queries).", ""]
    rows = []
    for k in CLS + ["model_bkg"]:
        for td, nm_ in SETS:
            pp, mm = cod["per_pattern"][f"{td}|{k}"], cod["macro"][f"{td}|{k}"]
            rows.append([NAMES[k], nm_, f"{pp['n']} / {mm['top1']['n_compounds']}", mci(mm["top1"]), f2(mm["top10"]["value"]),
                         f2(pp["top1"]), f"{pp['median_rank']:,.0f}"])
    L += table(["score", "answer set", "patterns / compounds", "macro top-1", "macro top-10", "per-pattern top-1",
                "median rank"], rows)
    ws = cod["without_si_queries"]
    L += ["", f"Without the {ws['n_si_queries']} Si-containing queries ({', '.join(ws['si_ids'])}; the COD library filter "
          "excludes Si, so their answers enter only as candidate entries), compound-weighted top-1: " +
          "; ".join(f"{k.split('|')[0]} {NAMES[k.split('|')[1]]} {v:.2f}" for k, v in ws["macro_top1"].items()) + "."]
    kd = ROB["kaduk"]
    ks = kd["summary"]
    pp_ = ks["patterns"]
    n_sync = sum(r["kind"] == "synchrotron" for r in pp_)
    L += ["", "## S14 An external set: raw profiles deposited by Kaduk and co-workers", "",
          f"Queries: the measured profiles in the supplementary CIFs of Kaduk and co-workers' Powder Diffraction structure "
          f"reports and the ICDD deposit for donepezil hydrochloride form III: {ks['n_patterns']} patterns of "
          f"{ks['n_compounds']} compounds, {n_sync} synchrotron and {ks['n_patterns'] - n_sync} laboratory Cu Kα1. None was used "
          "in training, tuning or any earlier analysis; the design was fixed before scoring, and preprocessing, scores and "
          "library are those of the main benchmark. Every anchor is the powder-study entry refined against the query itself, "
          f"so the any-same-form set is an upper bound. With powder studies excluded, {ks['n_sc_compounds']} compounds keep an "
          "independent single-crystal entry in the library.", "",
          f"{T()} Retrieval on the external set (background-subtracted queries).", ""]
    rows = []
    for k in CLS + ["model_bkg"]:
        for td, nm_ in SETS:
            pp, mm = kd["per_pattern"][f"{td}|{k}"], kd["macro"][f"{td}|{k}"]
            rows.append([NAMES[k], nm_, f"{pp['n']} / {mm['top1']['n_compounds']}", mci(mm["top1"]), f2(mm["top10"]["value"]),
                         f2(pp["top1"]), f"{pp['median_rank']:,.0f}"])
    L += table(["score", "answer set", "patterns / compounds", "macro top-1", "macro top-10", "per-pattern top-1",
                "median rank"], rows)
    rk = lambda r, key: "–" if r["ranks"][key] < 0 else f"{r['ranks'][key] + 1:,}"
    L += ["", f"{T()} Rank of the best correct entry per pattern (1 = first place; '–': no correct entry in the library).", ""]
    L += table(["compound", "anchor", "article DOI", "source", "Cu-grid coverage", "anchor's split", "ln q, any", "Pearson, any",
                "model, any", "ln q, single-crystal", "model, single-crystal"],
               [[r["compound"], r["refcode"], r["doi"], f"{r['kind']}, {r['wavelength']:.4f} Å", f2(r["cu_grid_coverage"]),
                 r["self_entry_split"], rk(r, "std|lnq_bkg"), rk(r, "std|pearson_bkg"), rk(r, "std|model_bkg"), rk(r, "sc|lnq_bkg"),
                 rk(r, "sc|model_bkg")] for r in pp_])
    kk = ROB["kaduk_by_instrument"]
    L += ["", f"{T()} The learned models on the external set by instrument, any same-form entry: patterns ranked first / "
          "within the top 10 / median rank. The anchor's split is the model's scaffold split (training, validation, test); "
          f"of the {kk[MODEL]['synchrotron']['n']} synchrotron anchors, " + ", ".join(f"{v} in {k}" for k, v in
                                                                                       sorted(kk["sync_self_entry_split"].items()))
          + ".", ""]
    L += table(["checkpoint", "synchrotron", "laboratory Cu"],
               [[lab, *(f"{kk[x][kd_]['top1']} / {kk[x][kd_]['top10']} / {kk[x][kd_]['median_rank']:,.1f} (n = {kk[x][kd_]['n']})"
                        for kd_ in ("synchrotron", "lab Cu"))] for x, lab in CKPTS if x in kk])
    return L


def main() -> int:
    L = ["# Supplementary material", "", f"{TITLE}. F. Cai.", "",
         "All tables are generated from the deposited results file, so they cannot drift from the data. Corrected labels "
         "throughout (Section S2). 'The model' is the symmetry-aware encoder, seed 0.", ""]
    for part in (s1_benchmark, s2_labels, s3_hubness, s4_results, s5_s6, s7_model, s8_sweep, s9_s10, s11_ladder,
                 s12_robustness, s13_s14):
        L += part()
    with open("paper/rsos/esi.md", "w", encoding="utf-8") as fh:
        fh.write(re.sub(r"(?<![\w.])-(?=\d)", "−", "\n".join(L)) + "\n")
    print(f"wrote paper/rsos/esi.md ({len(L)} lines, {_T[0]} tables)")
    # the change list against the RSC Advances submission belongs with the response to the previous reviews, which is
    # kept outside the public repo (board review 11-19: no submission history in the article or its supplement)
    priv = os.environ.get("XRD_RSOS_PRIVATE", "D:/xrd-clip/rsos_private")
    if os.path.isdir(priv):
        with open(os.path.join(priv, "ChangesFromPreviousSubmission.md"), "w", encoding="utf-8") as fh:
            fh.write("# How this version differs from the version submitted to RSC Advances (RA-ART-10-2026-010272)\n\n"
                     + "\n".join(table(["item", "submitted version", "this version", "reason"], DEVIATIONS)) + "\n")
        print(f"wrote {priv}/ChangesFromPreviousSubmission.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
