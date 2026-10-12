"""Every number the Digital Discovery manuscript quotes, in one JSON (tasks/todo.md Step 2, B1).

  python scripts/22_paper_numbers.py            -> results/paper_numbers.json

Sources: results/real_bench.json (19), results/real_bench_hub.json (21 eval), results/strong_baselines.json (13),
cache/real_bench/hub_ranks.npz (21 eval, gitignored), the opXRD truth + pattern files and the scaffold splits.
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
# Facts from run logs, not from a results file (plan Sections 12a-12b).
RUN_FACTS = {"bench_cost_usd": 2.53, "hub_cost_usd": 0.26, "hub_gpu": "RTX 4090", "hub_wall_s": 761,
             "g0_cost_usd": 0.25, "g0_gpu": "RTX 3090"}


def rd(x):          # 4 decimals: the manuscript rounds once, to 2
    return None if x is None else round(float(x), 4)


def ci(v):
    return [rd(v[0]), rd(v[1])]


def per_compound(hit: np.ndarray, ok: np.ndarray, stems: np.ndarray, uniq: np.ndarray) -> np.ndarray:
    """Per-compound hit rate over the patterns that have a correct entry in the library (NaN if none)."""
    return np.array([hit[ok & (stems == s)].mean() if (ok & (stems == s)).any() else np.nan for s in uniq])


def macro(r: np.ndarray, kk: int, stems: np.ndarray, cboot: np.ndarray) -> dict:
    """Compound-weighted top-k with the same compound bootstrap as 21 eval (rng(0), 2000 resamples)."""
    uniq = np.unique(stems)
    ok = r >= 0
    p = per_compound((r < kk) & ok, ok, stems, uniq)
    bs = np.nanmean(p[cboot], axis=1)
    return {"n_compounds": int(np.isfinite(p).sum()), "value": rd(np.nanmean(p)),
            "ci": [rd(np.nanpercentile(bs, 2.5)), rd(np.nanpercentile(bs, 97.5))]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default=f"{bench.DATA}/opxrd/opxrd_truth.json")
    ap.add_argument("--patterns", default=f"{bench.DATA}/opxrd/opxrd_cnrs_organic.json")
    ap.add_argument("--out", default="results/paper_numbers.json")
    args = ap.parse_args()

    rb = json.load(open("results/real_bench.json"))
    hub = json.load(open("results/real_bench_hub.json"))
    sb = json.load(open("results/strong_baselines.json"))
    R = np.load("cache/real_bench/hub_ranks.npz")
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
                # against 19's committed numbers (uncorrected): top-1 must be identical
                if corr == "none":
                    assert abs(rb["summary"][f"{td}|all|{key}"]["top1"] - s["top1"]) < 1e-9, (key, td)
    out["bench"] = bm
    out["paired_csls"] = {f"{td}|{k}": {kk: (rd(v) if isinstance(v, float) else ci(v) if kk.endswith("_ci") else v)
                                        for kk, v in hub["paired"][f"csls_aug_k10|{td}|{k}"].items()}
                          for k in KEYS for td in ("std", "sc")}
    out["within_family"] = {k: {"n": rb["within_family"][k]["n"], "top1": rd(rb["within_family"][k]["top1"])}
                            for k in KEYS + RAW}
    for name in ("library_curve", "library_curve_macro"):     # {size: [mean, sd over 5 decoy subsets]}
        out[name] = {k: {n: ci(v) for n, v in cv.items()} for k, cv in hub[name].items()}
    out["hybrid"] = {k: {td: {"top1": rd(rb["summary"][f"{td}|all|{k}"]["top1"]),
                              "top1_ci": ci(rb["summary"][f"{td}|all|{k}"]["top1_ci"])} for td in ("std", "sc")}
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
    rc = hub["rank_check"]
    out["gpu_check"] = {"max_abs_diff_vs_cpu_prepass": max(v["max_abs_diff_vs_prepass"] for v in rc.values()),
                        "rank_mismatches": sum(v["std_rank_mismatches"] + v["sc_rank_mismatches"] for v in rc.values()),
                        "top1_changes": sum(v["std_top1_changes"] + v["sc_top1_changes"] for v in rc.values())}

    # ---- simulated sweep (G0) ---------------------------------------------------------------------------------
    rows_g0, wins = [], 0
    for name, res in sb["conditions"].items():
        best = max(STRONG, key=lambda k: res[k]["top1"])
        m, b = res["model"], res[best]
        wins += m["ci95"][0] > b["ci95"][1]
        rows_g0.append({"condition": name, "model": rd(m["top1"]), "model_ci": ci(m["ci95"]), "best": best,
                        "best_top1": rd(b["top1"]), "best_ci": ci(b["ci95"]), "delta": rd(m["top1"] - b["top1"]),
                        "ci_overlap": not (m["ci95"][1] < b["ci95"][0] or m["ci95"][0] > b["ci95"][1]),
                        "pearson_fixed": rd(res["pearson"]["top1"])})
    d = [x["delta"] for x in rows_g0]
    out["g0"] = {"n_families": sb["n_families"], "n_queries": sb["n_queries"], "chance": rd(sb["chance_top1"]),
                 "n_conditions": len(rows_g0), "n_model_wins": int(wins),
                 "n_ci_overlap": int(sum(x["ci_overlap"] for x in rows_g0)),
                 "delta_min_abs": min(d, key=abs), "delta_max_abs": max(d, key=abs),
                 "model_range": [min(x["model"] for x in rows_g0), max(x["model"] for x in rows_g0)],
                 "us_per_pair_clean": {k: rd(sb["conditions"]["clean"][k]["us_per_pair"])
                                       for k in ["model", "pearson", "pearson_zs", "wcc_1", "wcc_2", "lnq"]},
                 "rows": rows_g0}

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {args.out}: {nq} queries, {n_uniq} compounds; GPU check {out['gpu_check']}; "
          f"G0 model wins {wins}/{len(rows_g0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
