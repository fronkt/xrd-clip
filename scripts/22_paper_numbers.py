"""Every number the Digital Discovery manuscript quotes, in one JSON (tasks/todo.md Step 2, B1).

  python scripts/22_paper_numbers.py            -> results/paper_numbers.json

Sources: results/real_bench.json (19), results/real_bench_hub_lnq2.json (21 eval with ln q v2), results/
strong_baselines_v2.json (13, crc32-seeded sweep), results/rescore_*.json + results/critic2_rerank.json +
results/model_gap*.json (revision R1-R5), cache/real_bench_lnq2/hub_ranks.npz (21 eval, gitignored), the opXRD truth
+ pattern files and the scaffold splits. ln q v2 (revision R1) is the primary ln q score; v1 (real_bench_hub.json,
cache/real_bench/hub_ranks.npz) is kept in `lnq_v1` for the ESI. The two hub files differ only in the ln q keys.
Rank conventions follow 19: r = number of wrong entries scoring above the best correct one, -1 = no correct entry
in the library. Per-pattern numbers are recomputed from the rank cache and asserted equal to the committed JSONs,
so the manuscript, the tables and the caches cannot drift apart silently.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from importlib import import_module

import numpy as np

sys.path.insert(0, "scripts")
bench = import_module("19_real_bench")

KEYS = ["pearson_bkg", "cos_zs_bkg", "wcc_1_bkg", "wcc_2_bkg", "lnq_bkg", "model_bkg"]
RAW = ["pearson_raw", "cos_zs_raw", "wcc_1_raw", "wcc_2_raw", "model_raw"]
CORRS = ["none", "csls_aug_k10"]
STRONG = ["pearson_zs", "wcc_0.5", "wcc_1", "wcc_2", "lnq"]   # G0 strong baselines; VC excluded (not validated)
CASE = "ZZZUWK"
CLS_KEYS = [k for k in KEYS if k != "model_bkg"]
HUB, HUB_RANKS = "results/real_bench_hub_lnq2.json", "cache/real_bench_lnq2/hub_ranks.npz"
HUB_V1, HUB_RANKS_V1 = "results/real_bench_hub.json", "cache/real_bench/hub_ranks.npz"
SWEEP = "results/strong_baselines_v2.json"     # both ln q versions on the same draws ('lnq' = v2, 'lnq_v1')
LNQ_V2 = "results/rescore_r1.json"             # 28 tag r1: ln q v2 per pattern + its model hybrids
# Facts from run logs, not from a results file (plan Sections 12a-12b).
RUN_FACTS = {"bench_cost_usd": 2.53, "hub_cost_usd": 0.26, "hub_gpu": "RTX 4090", "hub_wall_s": 761,
             "g0_cost_usd": 0.25, "g0_gpu": "RTX 3090",
             "model_query_ms_cpu": 5,        # 309,080 x 256 float32 product for one query, numpy, laptop CPU (10-12)
             "model_params_millions": 6.1, "model_epochs": 60, "model_batch": 512,
             # 10-12: 300 random cache/csd entries re-simulated with simulate_one under pymatgen 2026.4.16;
             # 296 were in the stick cache and all matched exactly (same reflections, |d2theta| = |dI| = 0)
             "pymatgen_recheck_n": 296, "pymatgen_recheck_version": "2026.4.16",
             # 10-13: ln q v2 rescored alone on an RTX 4090 (box A; logs/hub_lnq2_retry.log "SCORE_DONE 405s wall")
             "hub_lnq2_wall_s": 405}


def rd(x):          # 4 decimals: the manuscript rounds once, to 2
    return None if x is None else round(float(x), 4)


def ci(v):
    return [rd(v[0]), rd(v[1])]


def per_compound(hit: np.ndarray, ok: np.ndarray, stems: np.ndarray, uniq: np.ndarray) -> np.ndarray:
    """Per-compound hit rate over the patterns that have a correct entry in the library (NaN if none)."""
    return np.array([hit[ok & (stems == s)].mean() if (ok & (stems == s)).any() else np.nan for s in uniq])


def boot_idx(n: int, seed: int = 0, B: int = 2000) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, (B, n))


def macro(r: np.ndarray, kk: int, stems: np.ndarray, cboot: np.ndarray) -> dict:
    """Compound-weighted top-k. Any-same-form: the same compound bootstrap as 21 eval (rng(0), 2000 resamples).
    Single-crystal: only the compounds that have a single-crystal entry are resampled (referee 10-12), so every
    resample has exactly that many compounds."""
    uniq = np.unique(stems)
    ok = r >= 0
    p = per_compound((r < kk) & ok, ok, stems, uniq)
    p = p[np.isfinite(p)]
    if len(p) < len(uniq):
        cboot = boot_idx(len(p))
    bs = p[cboot].mean(axis=1)
    return {"n_compounds": int(len(p)), "value": rd(p.mean()),
            "ci": [rd(np.percentile(bs, 2.5)), rd(np.percentile(bs, 97.5))]}


def paired_macro(ra: np.ndarray, rb_: np.ndarray, stems: np.ndarray, kk: int = 1) -> dict:
    """Compound-weighted top-k of score a minus score b on the same patterns, compound-bootstrap CI over the
    compounds that have a correct entry under both."""
    uniq = np.unique(stems)
    ok = (ra >= 0) & (rb_ >= 0)
    pa = per_compound((ra < kk) & ok, ok, stems, uniq)
    pb = per_compound((rb_ < kk) & ok, ok, stems, uniq)
    keep = np.isfinite(pa) & np.isfinite(pb)
    d = pa[keep] - pb[keep]
    bs = d[boot_idx(len(d))].mean(axis=1)
    return {"n_compounds": int(keep.sum()), "diff": rd(d.mean()),
            "ci": [rd(np.percentile(bs, 2.5)), rd(np.percentile(bs, 97.5))]}


def rnd(o):
    """Round every float in a nested results object to 4 decimals."""
    if isinstance(o, float):
        return rd(o)
    if isinstance(o, dict):
        return {k: rnd(v) for k, v in o.items()}
    if isinstance(o, list):
        return [rnd(v) for v in o]
    return o


def revision(stems: np.ndarray, R) -> dict:
    """R1-R5 (tasks/todo.md Step 3). A block appears once its results file exists; nothing is recomputed here
    except paired macro differences from the 28 rank caches: the R2 package against the main benchmark (R: its ranks,
    ln q v2; 28's own paired_vs_base is against v1, so its ln q entry mixes in the v2 fix) and the R2 one-off ablations
    against the full package."""
    import os
    load = lambda p: json.load(open(p)) if os.path.exists(p) else None
    slim = lambda r: None if r is None else rnd({k: r[k] for k in ("meta", "per_pattern", "macro", "paired_vs_base",
                                                                     "groups") if k in r})
    rev = {"r1_real": slim(load("results/rescore_r1.json")), "r2": slim(load("results/rescore_r2.json")),
           "r2_ablations": {}, "r3_critic2": rnd(load("results/critic2_rerank.json")),
           "r4_gap": rnd(load("results/model_gap.json")), "r4c_realism": rnd(load("results/model_gap_realism.json")),
           "r5_cod": slim(load("results/rescore_main_cod.json"))}
    full = "cache/real_bench/rescore_r2/ranks.npz"
    if rev["r2"] is not None and os.path.exists(full):
        f = np.load(full)
        rev["r2"]["paired_vs_main"] = {k: {f"top{kk}": paired_macro(f[k], R[f"none|{k}"], stems, kk) for kk in (1, 10)}
                                       for k in f.files if "@" not in k}
    for tag in ("r2_noB", "r2_nolp", "r2_noka2", "r2_nomask"):
        res, rk = load(f"results/rescore_{tag}.json"), f"cache/real_bench/rescore_{tag}/ranks.npz"
        if res is None:
            continue
        rev["r2_ablations"][tag] = {"macro": rnd(res["macro"])}
        if os.path.exists(rk) and os.path.exists(full):
            a, b = np.load(rk), np.load(full)
            assert len(a[a.files[0]]) == len(stems)
            rev["r2_ablations"][tag]["paired_vs_r2"] = {k: paired_macro(a[k], b[k], stems) for k in a.files if k in b.files}
    return rev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default=f"{bench.DATA}/opxrd/opxrd_truth.json")
    ap.add_argument("--patterns", default=f"{bench.DATA}/opxrd/opxrd_cnrs_organic.json")
    ap.add_argument("--out", default="results/paper_numbers.json")
    args = ap.parse_args()

    rb = json.load(open("results/real_bench.json"))
    hub = json.load(open(HUB))
    sb = json.load(open(SWEEP))
    R = np.load(HUB_RANKS)
    r1 = json.load(open(LNQ_V2))
    # 19's committed per-pattern numbers, with ln q v2 (and its hybrids) from 28's r1 run
    rb_sum = {**rb["summary"], **{f"{k.split('|')[0]}|all|{k.split('|', 1)[1]}": v for k, v in r1["per_pattern"].items()}}
    meta = rb["meta"]
    stems = np.array([m["stem"] for m in meta])
    nq = len(meta)
    boot = bench.compound_bootstrap(stems)
    n_uniq = len(np.unique(stems))
    cboot = np.random.default_rng(0).integers(0, n_uniq, (2000, n_uniq))
    out: dict = {"run_facts": RUN_FACTS}

    # ---- dataset -------------------------------------------------------------------------------------------
    truth = {t["id"]: t for t in json.load(open(args.truth))}
    pats = {o["id"]: o for o in json.load(open(args.patterns))["organic"]}
    wl = [float(truth[m["id"]]["wavelength"] or 1.5406) for m in meta]
    sc_ok = R["none|sc|lnq_bkg"] >= 0
    cnt = Counter(stems.tolist())
    out["dataset"] = {
        "n_opxrd_organic": len(truth),
        "n_identified": sum(bool(t["true_stems"]) for t in truth.values()),
        "n_excluded": len(rb["excluded"]), "excluded": rb["excluded"],
        "n_queries": nq, "n_compounds": n_uniq,
        "n_lab_cu": int(sum(1.53 < w < 1.55 for w in wl)),
        "n_synchrotron_or_other": int(sum(not (1.53 < w < 1.55) for w in wl)),
        "wavelengths_non_cu": sorted({round(w, 4) for w in wl if not (1.53 < w < 1.55)}),
        "n_h_omitted": int(sum(bool(truth[m["id"]]["h_omitted"]) for m in meta)),
        "n_own_powder_study": int(sum(m["own_powder"] for m in meta)),
        "n_sc_patterns": int(sc_ok.sum()), "n_sc_compounds": int(len(set(stems[sc_ok]))),
        "n_within_family_patterns": int(sum(m["n_other"] > 0 for m in meta)),
        "n_within_family_compounds": int(len({m["stem"] for m in meta if m["n_other"] > 0})),
        "patterns_per_compound_top": cnt.most_common(8),
        "n_compounds_single_pattern": int(sum(v == 1 for v in cnt.values())),
        "n_library": rb["n_library"],
        "n_temp_given": int(sum(pats[m["id"]]["temp_K"] not in (None, "None") for m in meta)),
        "n_wavelength_missing_assumed_cu": int(sum(truth[m["id"]]["wavelength"] in (None, "None", "") for m in meta)),
        "n_cnrs_total": json.load(open(args.patterns))["stats"]["n_total"],
        "anchor_tiers": dict(Counter(str(t["anchor_tier"]) for t in truth.values())),
        "n_corpus": json.load(open("refcodes/splits_manifest.json"))["n_structures"],
        "n_extra": rb["n_library"] - json.load(open("refcodes/splits_manifest.json"))["n_structures"],
        "n_probe_per_set": hub["meta"]["n_probe"],
    }

    # ---- training-split overlap of the model (scaffold splits; extras were never in the corpus) ---------------
    split = {}
    for sp in ("train", "val", "test"):
        for ln in open(f"refcodes/scaffold_{sp}.txt"):
            split[ln.strip()] = sp
    q_split = []
    for m in meta:
        t = truth[m["id"]]
        sp = {split.get(x["refcode"], "none") for x in t["same_form"]}
        q_split.append("train" if "train" in sp else "val" if "val" in sp else "test" if "test" in sp else "none")
    qs = Counter(q_split)
    comp_train = {s for s, q in zip(stems, q_split) if q == "train"}
    out["model_training_overlap"] = {"patterns_with_same_form_entry_in": dict(qs),
                                     "compounds_with_same_form_entry_in_train": len(comp_train),
                                     "split_sizes": json.load(open("refcodes/splits_manifest.json"))["scaffold_counts"]}
    qsa = np.array(q_split)
    for sp in ("train", "none"):                  # model ranks for patterns whose structure it trained on, and not
        r = R[f"none|std|model_bkg"][qsa == sp]
        out["model_training_overlap"][f"model_bkg_{sp}"] = {"n": int(len(r)), "top1": rd((r == 0).mean()),
                                                           "top10": rd((r < 10).mean()), "median_rank": float(np.median(r) + 1)}

    # ---- real benchmark, per pattern + macro, uncorrected and CSLS ---------------------------------------------
    bm = {}
    for key in KEYS + RAW:
        for td in ("std", "sc"):
            for corr in CORRS if key in KEYS else ["none"]:
                r = R[f"{corr}|{td}|{key}"]
                s = bench.summarise(r, np.ones(nq, bool), boot)
                ref = hub["summary"][f"{corr}|{td}|all|{key}"]
                for f in ("n", "top1", "top10", "median_rank"):
                    assert abs(s[f] - ref[f]) < 1e-9, (key, td, corr, f, s[f], ref[f])
                m1, m10 = macro(r, 1, stems, cboot), macro(r, 10, stems, cboot)
                assert abs(m1["value"] - round(hub["macro"][f"{corr}|{td}|{key}"]["top1"], 4)) < 1e-9
                ok = r >= 0
                bm[f"{corr}|{td}|{key}"] = {
                    "n": s["n"], "top1": rd(s["top1"]), "top1_ci": ci(s["top1_ci"]),
                    "top10": rd(s["top10"]), "top10_ci": ci(s["top10_ci"]), "median_rank": s["median_rank"],
                    "macro_top1": m1["value"], "macro_top1_ci": m1["ci"], "n_compounds": m1["n_compounds"],
                    "macro_top10": m10["value"], "macro_top10_ci": m10["ci"],
                    "frac_rank_gt_100": rd((r[ok] >= 100).mean()), "frac_rank_gt_1000": rd((r[ok] >= 1000).mean()),
                }
                # against 19's committed numbers (uncorrected; ln q v2 from 28): top-1 must be identical
                if corr == "none":
                    assert abs(rb_sum[f"{td}|all|{key}"]["top1"] - s["top1"]) < 1e-9, (key, td)
    out["bench"] = bm
    out["paired_csls"] = {f"{td}|{k}": {kk: (rd(v) if isinstance(v, float) else ci(v) if kk.endswith("_ci") else v)
                                        for kk, v in hub["paired"][f"csls_aug_k10|{td}|{k}"].items()}
                          for k in KEYS for td in ("std", "sc")}
    for key, v in out["paired_csls"].items():     # macro diff: same compound-bootstrap rule as macro() above
        td, k = key.split("|")
        pm = paired_macro(R[f"csls_aug_k10|{td}|{k}"], R[f"none|{td}|{k}"], stems)
        assert abs(pm["diff"] - v["macro_diff"]) < 1e-3, (key, pm, v["macro_diff"])
        v["macro_diff"], v["macro_diff_ci"] = pm["diff"], pm["ci"]
    out["within_family"] = {k: {"n": rb["within_family"][k]["n"], "top1": rd(rb["within_family"][k]["top1"])}
                            for k in KEYS + RAW}
    for name in ("library_curve", "library_curve_macro"):     # {size: [mean, sd over 5 decoy subsets]}
        out[name] = {k: {n: ci(v) for n, v in cv.items()} for k, cv in hub[name].items()}
    out["hybrid"] = {k: {td: {"top1": rd(rb_sum[f"{td}|all|{k}"]["top1"]),
                              "top1_ci": ci(rb_sum[f"{td}|all|{k}"]["top1_ci"])} for td in ("std", "sc")}
                     for k in rb["keys"] if "->" in k and k.endswith("_bkg")}

    # ---- sensitivity of the correction (std, per pattern + macro) -------------------------------------------
    corr_all = hub["meta"]["corrections"]
    out["sensitivity"] = {f"{c}|{k}": {"top1": rd(hub["summary"][f"{c}|std|all|{k}"]["top1"]),
                                        "macro_top1": rd(hub["macro"][f"{c}|std|{k}"]["top1"]),
                                        "sc_macro_top1": rd(hub["macro"][f"{c}|sc|{k}"]["top1"])}
                          for c in corr_all for k in KEYS}

    # ---- hubness diagnosis + the case compound -----------------------------------------------------------------
    hz = {}
    for k in KEYS:
        h = hub["hubness"][k]
        hz[k] = {"N10_skew_aug": rd(h["probe_aug"]["N10_skew"]), "N10_skew_sim": rd(h["probe_sim"]["N10_skew"]),
                 "N10_max_aug": h["probe_aug"]["N10_max"],
                 "wrong_top1_none": h["query_top1_wrong|none"]["most_common"][:3],
                 "wrong_top1_csls": h["query_top1_wrong|csls_aug_k10"]["most_common"][:3],
                 "n_miss_none": h["query_top1_wrong|none"]["n_miss"]}
    out["hubness"] = hz
    rows = stems == CASE
    out["case"] = {"stem": CASE, "n_patterns": int(rows.sum()),
                   "formula": truth[meta[int(np.nonzero(rows)[0][0])]["id"]]["same_form"][0]["formula"],
                   "top1_of_20": {f"{c}|{k}": int(((R[f"{c}|std|{k}"] == 0) & rows).sum())
                                  for c in CORRS for k in KEYS}}
    cz = json.load(open("results/case_zzzuwk.json"))                    # 24: per-offset scores + texture check
    out["case"]["analysis"] = {"context": cz["context"],
                               "decoys": {m: {rc: {f: v[f] for f in ("beats_truth_with_scan", "beats_truth_at_offset0",
                                                                    "beats_truth_pearson", "best_offset_deg_median")}
                                              for rc, v in per.items() if v["role"] == "decoy"}
                                          for m, per in cz["methods"].items()}}

    # ---- rank-equivalence checks of the GPU scorer -------------------------------------------------------------
    # v1 run: every score against 19's CPU prepass. The v2 file's ln q check compares against the v1 CPU ranks, so
    # ln q v2 is checked here against 28's CPU torch ranks (rescore_r1) instead.
    rc = json.load(open(HUB_V1))["rank_check"]
    out["gpu_check"] = {"max_abs_diff_vs_cpu_prepass": max(v["max_abs_diff_vs_prepass"] for v in rc.values()),
                        "rank_mismatches": sum(v["std_rank_mismatches"] + v["sc_rank_mismatches"] for v in rc.values()),
                        "top1_changes": sum(v["std_top1_changes"] + v["sc_top1_changes"] for v in rc.values())}
    cpu2 = np.load("cache/real_bench/rescore_r1/ranks.npz")
    out["gpu_check"]["lnq_v2_vs_cpu"] = {
        "rank_mismatches": int(sum((R[f"none|{td}|lnq_bkg"] != cpu2[f"{td}|lnq_bkg"]).sum() for td in ("std", "sc"))),
        "top1_changes": int(sum(((R[f"none|{td}|lnq_bkg"] == 0) != (cpu2[f"{td}|lnq_bkg"] == 0)).sum() for td in ("std", "sc")))}

    # ---- simulated sweep (G0) ---------------------------------------------------------------------------------
    rows_g0, wins = [], 0
    for name, res in sb["conditions"].items():
        best = max(STRONG, key=lambda k: res[k]["top1"])
        m, b = res["model"], res[best]
        wins += m["ci95"][0] > b["ci95"][1]
        rows_g0.append({"condition": name, "model": rd(m["top1"]), "model_ci": ci(m["ci95"]), "best": best,
                        "best_top1": rd(b["top1"]), "best_ci": ci(b["ci95"]), "delta": rd(m["top1"] - b["top1"]),
                        "ci_overlap": not (m["ci95"][1] < b["ci95"][0] or m["ci95"][0] > b["ci95"][1]),
                        "pearson_fixed": rd(res["pearson"]["top1"]),
                        **{s: rd(res[s]["top1"]) for s in STRONG + ["lnq_v1"]}})
    d = [x["delta"] for x in rows_g0]
    out["g0"] = {"n_families": sb["n_families"], "n_queries": sb["n_queries"], "chance": rd(sb["chance_top1"]),
                 "n_conditions": len(rows_g0), "n_model_wins": int(wins),
                 "n_ci_overlap": int(sum(x["ci_overlap"] for x in rows_g0)),
                 "delta_min_abs": min(d, key=abs), "delta_max_abs": max(d, key=abs),
                 "model_range": [min(x["model"] for x in rows_g0), max(x["model"] for x in rows_g0)],
                 "us_per_pair_clean": {k: rd(sb["conditions"]["clean"][k]["us_per_pair"])
                                       for k in ["model", "pearson", "pearson_zs", "wcc_1", "wcc_2", "lnq"]},
                 "rows": rows_g0}

    # ---- referee-driven checks (10-12) -------------------------------------------------------------------------
    ex = {}
    # sweep: 'clean' and 'zero_offset=0.0' are the same condition; count distinct ones and per-score losses
    rows_d = [r for r in rows_g0 if r["condition"] != "zero_offset=0.0"]
    ex["g0_distinct"] = {"n_conditions": len(rows_d),
                         "delta_range": [min((x["delta"] for x in rows_d), key=abs), max((x["delta"] for x in rows_d), key=abs)],
                         "model_range": [min(x["model"] for x in rows_d), max(x["model"] for x in rows_d)],
                         "best_range": [min(x["best_top1"] for x in rows_d), max(x["best_top1"] for x in rows_d)],
                         "conditions_each_score_beats_model": {
                             s: int(sum(sb["conditions"][x["condition"]][s]["ci95"][0] > sb["conditions"][x["condition"]]["model"]["ci95"][1]
                                        for x in rows_d)) for s in STRONG + ["pearson"]}}
    # paired method differences (macro top-1, compound bootstrap)
    ex["paired_methods"] = {f"{td}|{a}-{b_}": paired_macro(R[f"none|{td}|{a}"], R[f"none|{td}|{b_}"], stems)
                            for td in ("std", "sc") for a in ("lnq_bkg", "wcc_1_bkg", "cos_zs_bkg", "wcc_2_bkg")
                            for b_ in ("pearson_bkg",)}
    # per pattern without the case compound, and the Pearson-vs-cosine gap
    nz = stems != CASE
    hit = {k: R[f"none|std|{k}"] == 0 for k in CLS_KEYS}
    ex["per_pattern_without_case"] = {k: {"hits": int(hit[k][nz].sum()), "n": int(nz.sum()),
                                          "top1": rd(hit[k][nz].mean())} for k in CLS_KEYS}
    g = hit["pearson_bkg"] & ~hit["cos_zs_bkg"]
    ex["gap_pearson_vs_cos"] = {"pearson_only": int(g.sum()), "cos_only": int((hit["cos_zs_bkg"] & ~hit["pearson_bkg"]).sum()),
                                "pearson_only_case": int((g & ~nz).sum()),
                                "net": int(hit["pearson_bkg"].sum() - hit["cos_zs_bkg"].sum()),
                                "net_case": int(hit["pearson_bkg"][~nz].sum() - hit["cos_zs_bkg"][~nz].sum())}
    # by subgroup: anchor tier, whether a correct entry is a corpus entry, wavelength class
    tier = np.array([str(truth[m["id"]]["anchor_tier"]) for m in meta])
    in_corpus = np.array([q != "none" for q in q_split])
    wlc = np.array(["Cu" if 1.53 < w < 1.55 else "Mo" if 0.70 < w < 0.72 else "Co" if 1.78 < w < 1.80
                    else "synchrotron" if w < 1.3 else "other" for w in wl])
    ex["wavelength_classes"] = dict(Counter(wlc.tolist()))
    groups = {"tier_strict": tier == "strict", "tier_lenient": tier == "lenient",
              "corpus_entry": in_corpus, "extra_only": ~in_corpus, "cu": wlc == "Cu", "non_cu": wlc != "Cu"}
    ex["subgroups"] = {}
    for gname, sel in groups.items():
        d = {"n_patterns": int(sel.sum()), "n_compounds": int(len(set(stems[sel])))}
        for k in KEYS:
            r = np.where(sel, R[f"none|std|{k}"], -1)
            d[k] = {"top1": rd((r[sel] == 0).mean()), "macro_top1": macro(r, 1, stems, cboot)["value"]}
        ex["subgroups"][gname] = d
    # set-aside ("cannot compare") entries and the own-powder RMSD check
    ex["unknown_entries"] = {"patterns_with_any": int(sum(bool(truth[m["id"]]["unknown"]) for m in meta)),
                             "total": int(sum(len(truth[m["id"]]["unknown"]) for m in meta))}
    ex["own_powder_rmsd_lt_0p05"] = {t_: int(sum(m["own_powder"] for m, tt in zip(meta, tier) if tt == t_))
                                     for t_ in ("strict", "lenient")}
    # model recall at the shortlist sizes (upper bound of any rerank), CPU cost per query over the full library
    rm = R["none|std|model_bkg"]
    ex["model_recall"] = {str(k): rd((rm < k).mean()) for k in (100, 1000)}
    us = sb["conditions"]["clean"]
    ex["cpu_s_per_query_full_library"] = {k: rd(us[k]["us_per_pair"] * 1e-6 * rb["n_library"])
                                          for k in ("pearson", "pearson_zs", "wcc_1", "wcc_2", "lnq", "model")}
    # temperatures of the anchor vs the single-crystal entries (CSD strings like '323 K' / 'at 120 K')
    def tK(s):
        mm = __import__("re").search(r"([\d.]+)\s*K", str(s or ""))
        return float(mm.group(1)) if mm else None
    dts = []
    for m, r in zip(meta, R["none|sc|lnq_bkg"]):
        t = truth[m["id"]]
        anc = [tK(x["temperature"]) for x in t["same_form"] if x["is_anchor"]]
        scs = [tK(x["temperature"]) for x in t["same_form"] if not x["is_powder_study"]]
        if r >= 0 and anc and anc[0] and any(scs):
            dts.append((min(abs(anc[0] - s) for s in scs if s), int(r)))
    if dts:
        dt = np.array(dts)
        ex["sc_temperature_gap"] = {"n": len(dts), "median_dT_K": float(np.median(dt[:, 0])),
                                    "top1_dT_le_50": rd((dt[dt[:, 0] <= 50, 1] == 0).mean()) if (dt[:, 0] <= 50).any() else None,
                                    "n_dT_le_50": int((dt[:, 0] <= 50).sum()),
                                    "top1_dT_gt_50": rd((dt[dt[:, 0] > 50, 1] == 0).mean()) if (dt[:, 0] > 50).any() else None,
                                    "n_dT_gt_50": int((dt[:, 0] > 50).sum())}
    # own entry disordered (simulated through the occupancy fallback when the strict parse fails)
    dis = np.array([any(x["has_disorder"] for x in truth[m["id"]]["same_form"] if x["is_anchor"]) for m in meta])
    ex["own_entry_disordered"] = {"n_patterns": int(dis.sum()), "n_compounds": int(len(set(stems[dis])))}
    for gname, sel in (("disordered", dis), ("ordered", ~dis)):
        ex["subgroups"][gname] = {"n_patterns": int(sel.sum()), "n_compounds": int(len(set(stems[sel])))}
        for k in KEYS:
            r = np.where(sel, R[f"none|std|{k}"], -1)
            ex["subgroups"][gname][k] = {"top1": rd((r[sel] == 0).mean()), "macro_top1": macro(r, 1, stems, cboot)["value"]}
    # cell mismatch between the own (powder) entry and the closest single-crystal same-form entry
    hubm = import_module("21_hubness")
    shards = bench.shard_table(None)
    lib_rc = hubm.library_refcodes(shards)
    gidx = {rc: i for i, rc in enumerate(lib_rc)}
    need, rows_c = set(), []
    for m in meta:
        t = truth[m["id"]]
        anc = [x["refcode"] for x in t["same_form"] if x["is_anchor"] and x["refcode"] in gidx]
        scs = [x["refcode"] for x in t["same_form"] if not x["is_powder_study"] and x["refcode"] in gidx]
        rows_c.append((anc[:1], scs))
        need |= set(anc[:1]) | set(scs)
    need = sorted(need)
    cell = {rc: np.asarray(r["cell"], float) for rc, r in zip(need, hubm.records_at(shards, [gidx[rc] for rc in need], ("cell",)))}
    dq = np.full(nq, np.nan)                      # per query: max relative cell-length difference, closest sc entry
    for qi, (anc, scs) in enumerate(rows_c):
        if anc and scs:
            a = cell[anc[0]]
            dq[qi] = min(float(np.max(np.abs(np.sort(cell[s][:3]) - np.sort(a[:3])) / np.sort(a[:3]))) for s in scs)
    c2 = json.load(open("results/critic2_rerank.json"))
    qpos = {m["id"].replace(".json", ""): i for i, m in enumerate(meta)}
    rank_c2 = np.full(nq, -1)
    for pid, v in c2["rank_by_pattern"].items():
        rank_c2[qpos[pid]] = v["critic2"]
    split_ranks = {"lnq": R["none|sc|lnq_bkg"], "lnq_v1": np.load(HUB_RANKS_V1)["none|sc|lnq_bkg"], "critic2": rank_c2,
                   **{k.replace("_bkg", ""): R[f"none|sc|{k}"] for k in CLS_KEYS if k != "lnq_bkg"}}
    sel = ~np.isnan(dq) & (R["none|sc|lnq_bkg"] >= 0)
    assert ((rank_c2 >= 0) == (R["none|sc|lnq_bkg"] >= 0)).all()      # critic2 covers exactly the sc queries
    small, large = sel & (dq <= 0.01), sel & (dq > 0.01)
    ex["sc_cell_mismatch"] = {"n": int(sel.sum()), "median_max_axis_rel": rd(np.median(dq[sel])),
                              "n_small": int(small.sum()), "n_large": int(large.sum())}
    for name, r in split_ranks.items():
        ex["sc_cell_mismatch"][f"{name}_top1_small"] = rd((r[small] == 0).mean())
        ex["sc_cell_mismatch"][f"{name}_top1_large"] = rd((r[large] == 0).mean())
        ex["sc_cell_mismatch"][f"{name}_hits_small"] = int((r[small] == 0).sum())
        ex["sc_cell_mismatch"][f"{name}_hits_large"] = int((r[large] == 0).sum())
    # ln q v1 (the first version, ESI) on the same queries, and the paired v2 - v1 change
    R1 = np.load(HUB_RANKS_V1)
    hub1 = json.load(open(HUB_V1))
    lv1 = {}
    for td in ("std", "sc"):
        a, b_ = R[f"none|{td}|lnq_bkg"], R1[f"none|{td}|lnq_bkg"]
        ok = b_ >= 0
        lv1[td] = {"v1": {"macro_top1": macro(b_, 1, stems, cboot), "macro_top10": macro(b_, 10, stems, cboot),
                          "top1": rd((b_[ok] == 0).mean()), "median_rank": float(np.median(b_[ok]) + 1)},
                   "paired_v2_minus_v1": {f"top{kk}": paired_macro(a, b_, stems, kk) for kk in (1, 10)},
                   "n_rank_changed": int((a[ok] != b_[ok]).sum()), "n_top1_changed": int(((a[ok] == 0) != (b_[ok] == 0)).sum())}
        assert abs(lv1[td]["v1"]["macro_top1"]["value"] - round(hub1["macro"][f"none|{td}|lnq_bkg"]["top1"], 4)) < 1e-9
    ex["lnq_v1"] = lv1
    # variable-cell re-rank: median rank over the 59 sc patterns (a miss lies beyond the shortlist, so the median is
    # defined only if it falls inside it) and per-pattern hits against ln q v2 on the same patterns
    k_c2 = c2["meta"]["k"]
    rc = rank_c2[rank_c2 >= 0]
    med_c2 = float(np.median(np.where(rc >= k_c2, 10 ** 9, rc)) + 1)
    assert med_c2 <= k_c2
    rl = R["none|sc|lnq_bkg"][rank_c2 >= 0]
    in_list = rl < k_c2
    ex["critic2"] = {"k": k_c2, "median_rank": med_c2, "n_patterns": int(len(rc)), "hits": int((rc == 0).sum()),
                     "lnq_hits": int((rl == 0).sum()), "n_answer_in_shortlist": int(in_list.sum()),
                     "n_answer_in_lnq_top100": int((rl < 100).sum()),
                     "gains": int(((rc == 0) & (rl != 0)).sum()), "losses": int(((rc != 0) & (rl == 0)).sum()),
                     # 1-based: ln q's rank of the answers the re-rank moves to first; the re-rank's rank of the
                     # answers ln q had first and it moves down
                     "gain_lnq_ranks": sorted(int(x) + 1 for x in rl[(rc == 0) & (rl != 0)]),
                     "loss_c2_ranks": sorted(int(x) + 1 for x in rc[(rc != 0) & (rl == 0)]),
                     "per_pattern": [{"id": meta[i]["id"].replace(".json", ""), "stem": str(stems[i]),
                                      "cell_diff_pct": None if np.isnan(dq[i]) else rd(100 * dq[i]),
                                      "lnq_rank": int(R["none|sc|lnq_bkg"][i]) + 1,
                                      "critic2_rank": int(rank_c2[i]) + 1 if rank_c2[i] < k_c2 else None}
                                     for i in np.nonzero(rank_c2 >= 0)[0]],
                     "n_comparisons": c2["meta"]["n_comparisons"], "n_failed": c2["meta"]["n_failed_comparisons"],
                     "cpu_min_median": rd(c2["meta"]["cpu_s_median"] / 60), "cpu_h_max": rd(c2["meta"]["cpu_s_max"] / 3600),
                     "cpu_h_total": rd(c2["meta"]["cpu_h_total"])}
    # chance level of the real within-family task: one random score per candidate
    wf_rows = [m for m in meta if m["n_other"] > 0]
    ex["within_family_chance"] = rd(np.mean([m["n_same_in_lib"] / (m["n_same_in_lib"] + m["n_other"]) for m in wf_rows]))
    ex["model_probe_check"] = json.load(open("results/model_probe_check.json"))
    ex["wrong_top1_audit"] = {k: v for k, v in json.load(open("results/wrong_top1_audit.json")).items()}
    # open replica (R5): library selection and identification counts from 31's outputs in the data dir
    cdir = f"{bench.DATA}/cod"
    sel_c = json.load(open(f"{cdir}/library_select.json"))
    n_all = sum(1 for ln in open(f"{cdir}/library_all_ids.txt") if ln.strip())
    tc = json.load(open(f"{cdir}/opxrd_truth_cod.json"))
    ident = [x for x in tc if x["true_stems"]]
    ex["cod"] = {"n_selected": sel_c["n_library"], "n_all": n_all, "n_added_candidates": n_all - sel_c["n_library"],
                 "n_patterns": len(tc), "n_identified": len(ident),
                 "n_identified_both": sum(1 for x in ident if truth[x["id"]]["true_stems"]),
                 "n_identified_cod_only": sum(1 for x in ident if not truth[x["id"]]["true_stems"]),
                 "n_other_compound": sum(x["n_other_compound"] for x in tc),
                 "n_unknown_same_formula": sum(len(x["unknown"]) for x in tc)}
    out["extra"] = ex
    out["revision"] = revision(stems, R)

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {args.out}: {nq} queries, {n_uniq} compounds; GPU check {out['gpu_check']}; "
          f"G0 model wins {wins}/{len(rows_g0)} rows (34 distinct conditions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
