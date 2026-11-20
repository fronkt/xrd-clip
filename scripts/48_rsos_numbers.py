"""Every number the Royal Society Open Science manuscript quotes, in one JSON (tasks/todo.md D-rest F5).

  PYTHONPATH=src python scripts/48_rsos_numbers.py          -> results/rsos_numbers.json

The RSOS version of scripts/22 (which stays the record of the RSC Advances numbers, results/paper_numbers.json).
Differences from 22:
  * the paper's model is the D3 retrain d3_sym_s0 (space-group-aware structure encoder, fixed K-alpha2 and
    March-Dollase, checkpoint chosen on degraded validation); its second seed d3_sym_s1, the ablation without the
    symmetry encoder d3_fix and the submitted model (poly) are reported beside it;
  * corrected labels everywhere (refcodes/truth_corrections.json, applied by 19.build_queries and by
    corrected_truth() here), so every model-dependent and label-dependent result is read from its re-evaluated file
    (results/d3/*, cache/*/ranks_<model>.npz, 21 eval --ranks_out);
  * query metadata come from 19.build_queries, not from 19's committed real_bench.json (old labels); the
    cross-check is against 28 eval --tag main of the same model (an independent scorer pipeline);
  * new blocks: models (the four checkpoints side by side), label and anchor sensitivities (43, 47), the intensity
    convention (conv), the every-anchor variant and the BFDH texture sweep.
Every input must exist: a missing file is an error, never a silently absent block.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter
from importlib import import_module

import numpy as np

sys.path.insert(0, "scripts")
bench = import_module("19_real_bench")
pn = import_module("22_paper_numbers")
rd, ci, rnd, macro, paired_macro = pn.rd, pn.ci, pn.rnd, pn.macro, pn.paired_macro
KEYS, RAW, CORRS, STRONG, CASE, CLS_KEYS = pn.KEYS, pn.RAW, pn.CORRS, pn.STRONG, pn.CASE, pn.CLS_KEYS

MODEL, SEED2, ABLATION, ORIGINAL = "d3_sym_s0", "d3_sym_s1", "d3_fix", "poly"
CKPT = {"d3_sym_s0": "runs/d3_sym_s0/best.pt", "d3_sym_s1": "runs/d3_sym_s1/best.pt", "d3_fix": "runs/d3_fix/best.pt",
        "poly": "runs/xrdclip_poly/best.pt"}
GPU_CHECK = "results/real_bench_hub.json"      # the scorer check (GPU vs 19's CPU pre-pass), label-independent
RUN_FACTS = {**pn.RUN_FACTS, "model_params_millions": 6.2}


def load(p: str):
    if not os.path.exists(p):
        raise SystemExit(f"[48] missing input {p}")
    return json.load(open(p))


def npz(p: str):
    if not os.path.exists(p):
        raise SystemExit(f"[48] missing input {p}")
    return np.load(p)


def slim(r):
    return rnd({k: r[k] for k in ("meta", "per_pattern", "macro", "groups") if k in r})


def corrected_truth(path: str) -> dict:
    """The opXRD truth with refcodes/truth_corrections.json applied exactly as 19.build_queries applies it."""
    truth = {t["id"]: t for t in json.load(open(path))}
    if os.environ.get("XRD_NO_TRUTH_CORRECTIONS"):
        return truth
    cp = os.environ.get("XRD_TRUTH_CORRECTIONS") or "refcodes/truth_corrections.json"
    for pid, c in json.load(open(cp)).items():
        if pid.startswith("_") or pid not in truth:
            continue
        t = truth[pid]
        for x in c.get("add_same_form", []):
            if x["refcode"] not in {y["refcode"] for y in t["same_form"]}:
                t["same_form"].append(x)
            t["unknown"] = [y for y in t["unknown"] if y["refcode"] != x["refcode"]]
            t["other_form"] = [y for y in t["other_form"] if y["refcode"] != x["refcode"]]
    return truth


def sweep(m: str) -> dict:
    """12's per-group files of one model merged into one {n_families, n_queries, chance_top1, conditions}."""
    files = sorted(f for f in glob.glob(f"results/d3/sbv3_{m}_*.json"))
    if not files:
        raise SystemExit(f"[48] no sweep files for {m}")
    out = None
    for f in files:
        d = json.load(open(f))
        if out is None:
            out = {k: d[k] for k in ("n_families", "n_queries", "chance_top1")} | {"conditions": {}}
        assert d["n_queries"] == out["n_queries"]
        out["conditions"].update(d["conditions"])
    want = set(import_module("12_strong_baselines").conditions())
    assert set(out["conditions"]) == want, (m, len(out["conditions"]))
    return out


def main_ranks(m: str, tag: str = "main") -> dict:
    p = f"cache/real_bench/rescore_{tag}/ranks.npz" if m == ORIGINAL else f"cache/real_bench/rescore_{tag}/ranks_{m}.npz"
    z = npz(p)
    return {k: z[k] for k in z.files}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default=f"{bench.DATA}/opxrd/opxrd_truth.json")
    ap.add_argument("--patterns", default=f"{bench.DATA}/opxrd/opxrd_cnrs_organic.json")
    ap.add_argument("--out", default="results/rsos_numbers.json")
    args = ap.parse_args()
    m = MODEL

    hub = load(f"results/d3/real_bench_hub_lnq2_{m}.json")
    R = npz(f"cache/real_bench_lnq2/hub_ranks_{m}.npz")
    r1 = load(f"results/d3/rescore_r1_{m}.json")
    main28 = load(f"results/d3/rescore_main_{m}.json")
    sb = sweep(m)

    shards = bench.shard_table(None)
    hubm = import_module("21_hubness")
    lib_rc = hubm.library_refcodes(shards)
    gidx = {rc: i for i, rc in enumerate(lib_rc)}
    qinfo, excluded, _ = bench.build_queries(args.truth, args.patterns, gidx)
    meta = [{"id": q["id"], "stem": q["stem"], "own_powder": q["own_powder"], "n_other": len(q["other"]),
             "n_same_in_lib": len(q["sets"]["std"][0])} for q in qinfo]
    stems = np.array([x["stem"] for x in meta])
    nq = len(meta)
    assert nq == hub["meta"]["n_queries"] == main28["meta"]["n_queries"]
    boot = bench.compound_bootstrap(stems)
    n_uniq = len(np.unique(stems))
    cboot = np.random.default_rng(0).integers(0, n_uniq, (2000, n_uniq))
    out: dict = {"run_facts": RUN_FACTS, "model_name": m}

    # ---- dataset -------------------------------------------------------------------------------------------
    truth = corrected_truth(args.truth)
    pats = {o["id"]: o for o in json.load(open(args.patterns))["organic"]}
    wl = [float(truth[x["id"]]["wavelength"] or 1.5406) for x in meta]
    sc_ok = R["none|sc|lnq_bkg"] >= 0
    cnt = Counter(stems.tolist())
    manifest = json.load(open("refcodes/splits_manifest.json"))
    out["dataset"] = {
        "n_opxrd_organic": len(truth),
        "n_identified": sum(bool(t["true_stems"]) for t in truth.values()),
        "n_excluded": len(excluded), "excluded": excluded,
        "n_queries": nq, "n_compounds": n_uniq,
        "n_lab_cu": int(sum(1.53 < w < 1.55 for w in wl)),
        "n_synchrotron_or_other": int(sum(not (1.53 < w < 1.55) for w in wl)),
        "wavelengths_non_cu": sorted({round(w, 4) for w in wl if not (1.53 < w < 1.55)}),
        "n_h_omitted": int(sum(bool(truth[x["id"]]["h_omitted"]) for x in meta)),
        "n_own_powder_study": int(sum(x["own_powder"] for x in meta)),
        "n_sc_patterns": int(sc_ok.sum()), "n_sc_compounds": int(len(set(stems[sc_ok]))),
        "n_within_family_patterns": int(sum(x["n_other"] > 0 for x in meta)),
        "n_within_family_compounds": int(len({x["stem"] for x in meta if x["n_other"] > 0})),
        "patterns_per_compound_top": cnt.most_common(8),
        "n_compounds_single_pattern": int(sum(v == 1 for v in cnt.values())),
        "n_library": len(lib_rc),
        "n_temp_given": int(sum(pats[x["id"]]["temp_K"] not in (None, "None") for x in meta)),
        "n_wavelength_missing_assumed_cu": int(sum(truth[x["id"]]["wavelength"] in (None, "None", "") for x in meta)),
        "n_cnrs_total": json.load(open(args.patterns))["stats"]["n_total"],
        "anchor_tiers": dict(Counter(str(t["anchor_tier"]) for t in truth.values())),
        "n_corpus": manifest["n_structures"], "n_extra": len(lib_rc) - manifest["n_structures"],
        "n_probe_per_set": hub["meta"]["n_probe"],
        "label_corrections": {k: v["reason"] for k, v in json.load(open("refcodes/truth_corrections.json")).items()
                              if not k.startswith("_")},
    }

    # ---- training-split overlap (scaffold splits; extras were never in the corpus) -------------------------------
    split = {}
    for sp in ("train", "val", "test"):
        for ln in open(f"refcodes/scaffold_{sp}.txt"):
            split[ln.strip()] = sp
    q_split = []
    for x in meta:
        sp = {split.get(y["refcode"], "none") for y in truth[x["id"]]["same_form"]}
        q_split.append("train" if "train" in sp else "val" if "val" in sp else "test" if "test" in sp else "none")
    qsa = np.array(q_split)
    out["model_training_overlap"] = {"patterns_with_same_form_entry_in": dict(Counter(q_split)),
                                     "compounds_with_same_form_entry_in_train": len({s for s, q in zip(stems, q_split) if q == "train"}),
                                     "split_sizes": manifest["scaffold_counts"]}
    for sp in ("train", "none"):
        r = R["none|std|model_bkg"][qsa == sp]
        out["model_training_overlap"][f"model_bkg_{sp}"] = {"n": int(len(r)), "top1": rd((r == 0).mean()),
                                                           "top10": rd((r < 10).mean()), "median_rank": float(np.median(r) + 1)}

    # ---- real benchmark (21 eval, ln q v2 work dir), checked against 28 eval --tag main of the same model ----------
    bm = {}
    for key in KEYS + RAW:
        for td in ("std", "sc"):
            for corr in (CORRS if key in KEYS else ["none"]):
                r = R[f"{corr}|{td}|{key}"]
                s = bench.summarise(r, np.ones(nq, bool), boot)
                ref = hub["summary"][f"{corr}|{td}|all|{key}"]
                for f in ("n", "top1", "top10", "median_rank"):
                    assert abs(s[f] - ref[f]) < 1e-9, (key, td, corr, f, s[f], ref[f])
                m1, m10 = macro(r, 1, stems, cboot), macro(r, 10, stems, cboot)
                assert abs(m1["value"] - round(hub["macro"][f"{corr}|{td}|{key}"]["top1"], 4)) < 1e-9
                if corr == "none" and key in KEYS:   # the two pipelines (GPU hub pass, 28 CPU rescore) must agree
                    t28 = main28["per_pattern"][f"{td}|{key}"]
                    assert abs(t28["top1"] - s["top1"]) < 1e-9 and t28["n"] == s["n"], (key, td, t28["top1"], s["top1"])
                ok = r >= 0
                bm[f"{corr}|{td}|{key}"] = {
                    "n": s["n"], "top1": rd(s["top1"]), "top1_ci": ci(s["top1_ci"]),
                    "top10": rd(s["top10"]), "top10_ci": ci(s["top10_ci"]), "median_rank": s["median_rank"],
                    "macro_top1": m1["value"], "macro_top1_ci": m1["ci"], "n_compounds": m1["n_compounds"],
                    "macro_top10": m10["value"], "macro_top10_ci": m10["ci"],
                    "frac_rank_gt_100": rd((r[ok] >= 100).mean()), "frac_rank_gt_1000": rd((r[ok] >= 1000).mean())}
    out["bench"] = bm
    out["paired_csls"] = {}
    for td in ("std", "sc"):
        for k in KEYS:
            v = {kk: (rd(x) if isinstance(x, float) else ci(x) if kk.endswith("_ci") else x)
                 for kk, x in hub["paired"][f"csls_aug_k10|{td}|{k}"].items()}
            pm = paired_macro(R[f"csls_aug_k10|{td}|{k}"], R[f"none|{td}|{k}"], stems)
            assert abs(pm["diff"] - v["macro_diff"]) < 1e-3, (td, k, pm, v["macro_diff"])
            v["macro_diff"], v["macro_diff_ci"] = pm["diff"], pm["ci"]
            out["paired_csls"][f"{td}|{k}"] = v
    out["within_family"] = {k: {"n": hub["within_family"][f"none|{k}"]["n"], "top1": rd(hub["within_family"][f"none|{k}"]["top1"])}
                            for k in KEYS + RAW}
    for name in ("library_curve", "library_curve_macro"):
        out[name] = {k: {n: ci(v) for n, v in cv.items()} for k, cv in hub[name].items()}
    out["hybrid"] = {k.split("|", 1)[1]: {} for k in r1["per_pattern"] if "->" in k}
    for k, v in r1["per_pattern"].items():
        if "->" in k:
            td, key = k.split("|", 1)
            out["hybrid"][key][td] = {"top1": rd(v["top1"]), "top1_ci": ci(v["top1_ci"]),
                                      "macro_top1": rd(r1["macro"][k]["top1"]["value"]),
                                      "macro_top1_ci": ci(r1["macro"][k]["top1"]["ci"])}
    out["sensitivity"] = {f"{c}|{k}": {"top1": rd(hub["summary"][f"{c}|std|all|{k}"]["top1"]),
                                        "macro_top1": rd(hub["macro"][f"{c}|std|{k}"]["top1"]),
                                        "sc_macro_top1": rd(hub["macro"][f"{c}|sc|{k}"]["top1"])}
                          for c in hub["meta"]["corrections"] for k in KEYS}

    # ---- hubness + the case compound --------------------------------------------------------------------------
    out["hubness"] = {k: {"N10_skew_aug": rd(hub["hubness"][k]["probe_aug"]["N10_skew"]),
                          "N10_skew_sim": rd(hub["hubness"][k]["probe_sim"]["N10_skew"]),
                          "N10_max_aug": hub["hubness"][k]["probe_aug"]["N10_max"],
                          "wrong_top1_none": hub["hubness"][k]["query_top1_wrong|none"]["most_common"][:3],
                          "wrong_top1_csls": hub["hubness"][k]["query_top1_wrong|csls_aug_k10"]["most_common"][:3],
                          "n_miss_none": hub["hubness"][k]["query_top1_wrong|none"]["n_miss"]} for k in KEYS}
    rows = stems == CASE
    cz = load("results/case_zzzuwk.json")
    out["case"] = {"stem": CASE, "n_patterns": int(rows.sum()),
                   "formula": truth[meta[int(np.nonzero(rows)[0][0])]["id"]]["same_form"][0]["formula"],
                   "top1_of_20": {f"{c}|{k}": int(((R[f"{c}|std|{k}"] == 0) & rows).sum()) for c in CORRS for k in KEYS},
                   "analysis": {"context": cz["context"],
                                "decoys": {mm: {rc: {f: v[f] for f in ("beats_truth_with_scan", "beats_truth_at_offset0",
                                                                      "beats_truth_pearson", "best_offset_deg_median")}
                                                for rc, v in per.items() if v["role"] == "decoy"}
                                           for mm, per in cz["methods"].items()}},
                   "texture_fit": rnd(load("results/zzzuwk_texture.json"))}

    # ---- the GPU scorer against 19's CPU pre-pass (label-independent; from the submitted run) -----------------------
    rc = load(GPU_CHECK)["rank_check"]
    out["gpu_check"] = {"max_abs_diff_vs_cpu_prepass": max(v["max_abs_diff_vs_prepass"] for v in rc.values()),
                        "rank_mismatches": sum(v["std_rank_mismatches"] + v["sc_rank_mismatches"] for v in rc.values()),
                        "top1_changes": sum(v["std_top1_changes"] + v["sc_top1_changes"] for v in rc.values())}
    cpu2 = main_ranks(m, "r1")
    out["gpu_check"]["lnq_v2_vs_cpu"] = {
        "rank_mismatches": int(sum((R[f"none|{td}|lnq_bkg"] != cpu2[f"{td}|lnq_bkg"]).sum() for td in ("std", "sc"))),
        "top1_changes": int(sum(((R[f"none|{td}|lnq_bkg"] == 0) != (cpu2[f"{td}|lnq_bkg"] == 0)).sum() for td in ("std", "sc")))}

    # ---- simulated sweep: all 43 conditions (34 distinct + 8 BFDH texture + the duplicate zero offset) -------------
    def g0(sb_: dict) -> dict:
        rows_, wins = [], 0
        for name, res in sb_["conditions"].items():
            best = max(STRONG, key=lambda k: res[k]["top1"])
            mo, b = res["model"], res[best]
            wins += mo["ci95"][0] > b["ci95"][1]
            rows_.append({"condition": name, "model": rd(mo["top1"]), "model_ci": ci(mo["ci95"]), "best": best,
                          "best_top1": rd(b["top1"]), "best_ci": ci(b["ci95"]), "delta": rd(mo["top1"] - b["top1"]),
                          "ci_overlap": not (mo["ci95"][1] < b["ci95"][0] or mo["ci95"][0] > b["ci95"][1]),
                          "pearson_fixed": rd(res["pearson"]["top1"]), **{s: rd(res[s]["top1"]) for s in STRONG + ["lnq_v1"]}})
        dist = [r_ for r_ in rows_ if r_["condition"] != "zero_offset=0.0"]
        d_ = [x["delta"] for x in dist]
        return {"n_families": sb_["n_families"], "n_queries": sb_["n_queries"], "chance": rd(sb_["chance_top1"]),
                "n_conditions": len(dist), "n_model_wins": int(wins), "n_ci_overlap": int(sum(x["ci_overlap"] for x in dist)),
                "delta_min_abs": min(d_, key=abs), "delta_max_abs": max(d_, key=abs),
                "model_range": [min(x["model"] for x in dist), max(x["model"] for x in dist)],
                "best_range": [min(x["best_top1"] for x in dist), max(x["best_top1"] for x in dist)],
                "conditions_each_score_beats_model": {
                    s: int(sum(sb_["conditions"][x["condition"]][s]["ci95"][0] > sb_["conditions"][x["condition"]]["model"]["ci95"][1]
                               for x in dist)) for s in STRONG + ["pearson"]},
                "us_per_pair_clean": {k: rd(sb_["conditions"]["clean"][k]["us_per_pair"])
                                      for k in ["model", "pearson", "pearson_zs", "wcc_1", "wcc_2", "lnq"]},
                "rows": rows_}
    out["g0"] = g0(sb)
    sweeps = {x: sweep(x) for x in (m, SEED2, ABLATION, ORIGINAL)}
    # identical draws: the classical columns of every model's sweep must agree exactly (same platform, crc32 seeds)
    for x, s_ in sweeps.items():
        for c, res in s_["conditions"].items():
            for k in STRONG + ["pearson"]:
                assert res[k]["top1"] == sb["conditions"][c][k]["top1"], (x, c, k)

    # ---- the four checkpoints side by side ----------------------------------------------------------------------
    models = {}
    for x in (m, SEED2, ABLATION, ORIGINAL):
        rm = load("results/rescore_main.json" if x == ORIGINAL else f"results/d3/rescore_main_{x}.json")
        gap = load(f"results/d3/model_gap_{x}.json")["results"]
        ls = load("results/label_sensitivity.json" if x == ORIGINAL else f"results/d3/label_sensitivity_{x}.json")
        rk = main_ranks(x)
        row = {"std_macro_top1": rnd(rm["macro"]["std|model_bkg"]["top1"]), "sc_macro_top1": rnd(rm["macro"]["sc|model_bkg"]["top1"]),
               "std_median_rank": rm["per_pattern"]["std|model_bkg"]["median_rank"],
               "sc_median_rank": rm["per_pattern"]["sc|model_bkg"]["median_rank"],
               "std_top1_patterns": int((rk["std|model_bkg"] == 0).sum()), "sc_top1_patterns": int((rk["sc|model_bkg"] == 0).sum()),
               "hybrid_1000_lnq_std": rnd(rm["macro"]["std|model@1000->lnq_bkg"]["top1"]),
               "ambient_sc_macro_top1": ls["subsets"]["ambient"]["sc|model_bkg"]["top1"],
               "ladder": {k: {"macro_top1": rd(v["macro_top1"]["value"]), "median_rank": v["per_pattern"]["median_rank"]}
                          for k, v in gap.items() if k.endswith("|std")},
               "sweep_model_range": out["g0"]["model_range"] if x == m else g0(sweeps[x])["model_range"],
               "sweep_clean": rd(sweeps[x]["conditions"]["clean"]["model"]["top1"])}
        if x != ORIGINAL:
            import torch
            c = torch.load(CKPT[x], map_location="cpu", weights_only=False)
            hist = json.load(open(os.path.join(os.path.dirname(CKPT[x]), "history.json")))
            row["train"] = {k: c["args"][k] for k in ("realism", "symops", "val_mode", "seed", "epochs", "batch_size", "lr",
                                                      "weight_decay", "warmup")}
            row["best_epoch"] = int(c["epoch"]) + 1
            row["best_degraded_val_top1"] = rd(max(h["val_top1"] for h in hist))
            pm = paired_macro(rk["std|model_bkg"], main_ranks(ORIGINAL)["std|model_bkg"], stems)
            row["paired_std_vs_original"] = pm
        models[x] = row
    out["models"] = models
    out["sweep_rows_other"] = {x: g0(sweeps[x])["rows"] for x in (SEED2, ABLATION, ORIGINAL)}
    # what the operator tokens do in the sweep: each seed against the ablation, condition by condition
    dist_c = [c for c in sb["conditions"] if c != "zero_offset=0.0"]
    out["symops_sweep_effect"] = {}
    for x in (m, SEED2):
        d_ = np.array([sweeps[x]["conditions"][c]["model"]["top1"] - sweeps[ABLATION]["conditions"][c]["model"]["top1"] for c in dist_c])
        out["symops_sweep_effect"][x] = {
            "mean_diff": rd(d_.mean()), "min_diff": rd(d_.min()), "max_diff": rd(d_.max()), "n_positive": int((d_ > 0).sum()),
            "n_conditions": len(dist_c),
            "n_ci_separated": int(sum(sweeps[x]["conditions"][c]["model"]["ci95"][0] > sweeps[ABLATION]["conditions"][c]["model"]["ci95"][1]
                                      for c in dist_c))}
    d_ = np.array([sweeps[m]["conditions"][c]["model"]["top1"] - sweeps[SEED2]["conditions"][c]["model"]["top1"] for c in dist_c])
    out["symops_sweep_effect"]["seed_mean_abs_diff"] = rd(np.abs(d_).mean())
    out["seed_spread"] = {"std_macro_top1": rd(abs(models[m]["std_macro_top1"]["value"] - models[SEED2]["std_macro_top1"]["value"])),
                          "paired_s1_minus_s0": paired_macro(main_ranks(SEED2)["std|model_bkg"], main_ranks(m)["std|model_bkg"], stems)}

    # ---- referee-driven checks (carried over from 22, on the corrected labels and this model) ---------------------
    ex = {}
    ex["paired_methods"] = {f"{td}|{a}-{b_}": paired_macro(R[f"none|{td}|{a}"], R[f"none|{td}|{b_}"], stems)
                            for td in ("std", "sc") for a in ("lnq_bkg", "wcc_1_bkg", "cos_zs_bkg", "wcc_2_bkg") for b_ in ("pearson_bkg",)}
    nz = stems != CASE
    hit = {k: R[f"none|std|{k}"] == 0 for k in CLS_KEYS}
    ex["per_pattern_without_case"] = {k: {"hits": int(hit[k][nz].sum()), "n": int(nz.sum()), "top1": rd(hit[k][nz].mean())}
                                      for k in CLS_KEYS}
    g = hit["pearson_bkg"] & ~hit["cos_zs_bkg"]
    ex["gap_pearson_vs_cos"] = {"pearson_only": int(g.sum()), "cos_only": int((hit["cos_zs_bkg"] & ~hit["pearson_bkg"]).sum()),
                                "pearson_only_case": int((g & ~nz).sum()),
                                "net": int(hit["pearson_bkg"].sum() - hit["cos_zs_bkg"].sum()),
                                "net_case": int(hit["pearson_bkg"][~nz].sum() - hit["cos_zs_bkg"][~nz].sum())}
    tier = np.array([str(truth[x["id"]]["anchor_tier"]) for x in meta])
    in_corpus = np.array([q != "none" for q in q_split])
    wlc = np.array(["Cu" if 1.53 < w < 1.55 else "Mo" if 0.70 < w < 0.72 else "Co" if 1.78 < w < 1.80
                    else "synchrotron" if w < 1.3 else "other" for w in wl])
    ex["wavelength_classes"] = dict(Counter(wlc.tolist()))
    dis = np.array([any(y["has_disorder"] for y in truth[x["id"]]["same_form"] if y["is_anchor"]) for x in meta])
    ex["own_entry_disordered"] = {"n_patterns": int(dis.sum()), "n_compounds": int(len(set(stems[dis])))}
    groups = {"tier_strict": tier == "strict", "tier_lenient": tier == "lenient", "corpus_entry": in_corpus,
              "extra_only": ~in_corpus, "cu": wlc == "Cu", "non_cu": wlc != "Cu", "disordered": dis, "ordered": ~dis}
    ex["subgroups"] = {}
    for gname, sel in groups.items():
        d = {"n_patterns": int(sel.sum()), "n_compounds": int(len(set(stems[sel])))}
        for k in KEYS:
            r = np.where(sel, R[f"none|std|{k}"], -1)
            d[k] = {"top1": rd((r[sel] == 0).mean()), "macro_top1": macro(r, 1, stems, cboot)["value"]}
        ex["subgroups"][gname] = d
    ex["unknown_entries"] = {"patterns_with_any": int(sum(bool(truth[x["id"]]["unknown"]) for x in meta)),
                             "total": int(sum(len(truth[x["id"]]["unknown"]) for x in meta))}
    ex["own_powder_rmsd_lt_0p05"] = {t_: int(sum(x["own_powder"] for x, tt in zip(meta, tier) if tt == t_)) for t_ in ("strict", "lenient")}
    rmod = R["none|std|model_bkg"]
    ex["model_recall"] = {str(k): rd((rmod < k).mean()) for k in (100, 1000)}
    # per-pair CPU time: the median over the sweep's conditions (the first condition timed, clean, carries warm-up)
    ex["cpu_s_per_query_full_library"] = {k: rd(float(np.median([c[k]["us_per_pair"] for c in sb["conditions"].values()]))
                                             * 1e-6 * len(lib_rc))
                                          for k in ("pearson", "pearson_zs", "wcc_1", "wcc_2", "lnq", "model")}

    def tK(s):
        mm = re.search(r"([\d.]+)\s*K", str(s or ""))
        return float(mm.group(1)) if mm else None

    # cell mismatch between the own powder-study entry and the closest single-crystal same-form entry (22's method)
    need, rows_c = set(), []
    for x in meta:
        t = truth[x["id"]]
        ps = [y["refcode"] for y in t["same_form"] if y["is_powder_study"] and y["is_anchor"] and y["refcode"] in gidx]
        own = [min(ps, key=lambda rc_: (t["anchors"][rc_]["rmsd"], rc_))] if ps else []
        scs = [y["refcode"] for y in t["same_form"] if not y["is_powder_study"] and y["refcode"] in gidx]
        rows_c.append((own, scs))
        need |= set(own) | set(scs)
    need = sorted(need)
    recs_c = dict(zip(need, hubm.records_at(shards, [gidx[rc_] for rc_ in need], ("cell", "spacegroup"))))
    red = {rc_: pn.reduced_cell(np.asarray(r["cell"], float), str(r["spacegroup"])) for rc_, r in recs_c.items()}
    dq, dvol, dT = np.full(nq, np.nan), np.full(nq, np.nan), np.full(nq, np.nan)
    closest = {}
    tK_of = lambda t, rc_: next((tK(y["temperature"]) for y in t["same_form"] if y["refcode"] == rc_), None)
    for qi, (own, scs) in enumerate(rows_c):
        if own and scs:
            a = red[own[0]]
            best = min(scs, key=lambda s: (pn.cell_mismatch(red[s], a), s))
            dq[qi] = pn.cell_mismatch(red[best], a)
            dvol[qi] = red[best].volume / a.volume
            closest[meta[qi]["id"].replace(".json", "")] = best
            ta, ts = tK_of(truth[meta[qi]["id"]], own[0]), tK_of(truth[meta[qi]["id"]], best)
            if ta and ts:
                dT[qi] = abs(ta - ts)
    c2 = load("results/critic2_rerank.json")
    qpos = {x["id"].replace(".json", ""): i for i, x in enumerate(meta)}
    rank_c2 = np.full(nq, -1)
    for pid, v in c2["rank_by_pattern"].items():
        rank_c2[qpos[pid]] = v["critic2"]
    R1 = npz(f"cache/real_bench/hub_ranks_{m}.npz")              # ln q v1 hub pass, corrected labels
    split_ranks = {"lnq": R["none|sc|lnq_bkg"], "lnq_v1": R1["none|sc|lnq_bkg"], "critic2": rank_c2,
                   **{k.replace("_bkg", ""): R[f"none|sc|{k}"] for k in CLS_KEYS if k != "lnq_bkg"}}
    sel = ~np.isnan(dq) & (R["none|sc|lnq_bkg"] >= 0)
    assert ((rank_c2 >= 0) == (R["none|sc|lnq_bkg"] >= 0)).all(), "critic2 must cover exactly the sc queries"
    small, large = sel & (dq <= 0.01), sel & (dq > 0.01)
    ex["sc_cell_mismatch"] = {"n": int(sel.sum()), "median_max_axis_rel": rd(np.median(dq[sel])),
                              "n_small": int(small.sum()), "n_large": int(large.sum())}
    for name, r in split_ranks.items():
        ex["sc_cell_mismatch"][f"{name}_top1_small"] = rd((r[small] == 0).mean())
        ex["sc_cell_mismatch"][f"{name}_top1_large"] = rd((r[large] == 0).mean())
        ex["sc_cell_mismatch"][f"{name}_hits_small"] = int((r[small] == 0).sum())
        ex["sc_cell_mismatch"][f"{name}_hits_large"] = int((r[large] == 0).sum())
    vr = np.where(dvol >= 1, dvol, 1 / dvol)
    superc = sel & (np.round(vr) >= 2) & (np.abs(vr - np.round(vr)) < 0.05)
    ex["sc_cell_mismatch"]["supercell"] = [{"id": meta[i]["id"].replace(".json", ""), "stem": str(stems[i]),
                                            "volume_ratio": rd(float(vr[i])), "lnq_rank": int(R["none|sc|lnq_bkg"][i]) + 1}
                                           for i in np.nonzero(superc)[0]]
    ex["sc_cell_mismatch"]["closest_sc_entry"] = {k: v for k, v in closest.items() if sel[qpos[k]]}
    ex["sc_cell_mismatch"]["per_query_pct"] = {meta[i]["id"].replace(".json", ""): rd(100 * dq[i]) for i in np.nonzero(sel)[0]}
    tsel = sel & ~np.isnan(dT)
    hot, cold = tsel & (dT > 50), tsel & (dT <= 50)
    rl_ = R["none|sc|lnq_bkg"]
    ex["sc_temperature_gap"] = {"n": int(tsel.sum()), "median_dT_K": float(np.median(dT[tsel])),
                                "n_dT_gt_50": int(hot.sum()), "n_dT_le_50": int(cold.sum()),
                                "top1_dT_gt_50": rd((rl_[hot] == 0).mean()), "top1_dT_le_50": rd((rl_[cold] == 0).mean()),
                                "n_dT_gt_50_cells_differ": int((hot & large).sum()), "n_dT_gt_50_cells_agree": int((hot & small).sum()),
                                "n_cells_differ_with_T": int((tsel & large).sum()), "n_cells_agree_with_T": int((tsel & small).sum())}
    hub1 = load(f"results/d3/real_bench_hub_v1_{m}.json")
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

    # variable-cell re-rank (29 eval on the corrected sc set) + control, injection, pressure-series split, X2/X3
    k_c2 = c2["meta"]["k"]
    rcv = rank_c2[rank_c2 >= 0]
    med_c2 = float(np.median(np.where(rcv >= k_c2, 10 ** 9, rcv)) + 1)
    assert med_c2 <= k_c2
    rl = R["none|sc|lnq_bkg"][rank_c2 >= 0]
    in_list = rl < k_c2
    ex["critic2"] = {"k": k_c2, "median_rank": med_c2, "n_patterns": int(len(rcv)), "hits": int((rcv == 0).sum()),
                     "lnq_hits": int((rl == 0).sum()), "n_answer_in_shortlist": int(in_list.sum()),
                     "n_answer_in_lnq_top100": int((rl < 100).sum()),
                     "gains": int(((rcv == 0) & (rl != 0)).sum()), "losses": int(((rcv != 0) & (rl == 0)).sum()),
                     "gain_lnq_ranks": sorted(int(x) + 1 for x in rl[(rcv == 0) & (rl != 0)]),
                     "loss_c2_ranks": sorted(int(x) + 1 for x in rcv[(rcv != 0) & (rl == 0)]),
                     "per_pattern": [{"id": meta[i]["id"].replace(".json", ""), "stem": str(stems[i]),
                                      "cell_diff_pct": None if np.isnan(dq[i]) else rd(100 * dq[i]),
                                      "dT_K": None if np.isnan(dT[i]) else rd(float(dT[i])),
                                      "lnq_rank": int(R["none|sc|lnq_bkg"][i]) + 1,
                                      "critic2_rank": int(rank_c2[i]) + 1 if rank_c2[i] < k_c2 else None}
                                     for i in np.nonzero(rank_c2 >= 0)[0]],
                     "n_comparisons": c2["meta"]["n_comparisons"], "n_failed": c2["meta"]["n_failed_comparisons"],
                     "cpu_min_median": rd(c2["meta"]["cpu_s_median"] / 60), "cpu_h_max": rd(c2["meta"]["cpu_s_max"] / 3600),
                     "cpu_h_total": rd(c2["meta"]["cpu_h_total"])}
    for row in ex["critic2"]["per_pattern"]:
        d_ = f"cache/critic2/{row['id']}"
        row["n_peaks"] = sum(1 for ln in open(f"{d_}/fit.peaks") if ln[:1].isdigit())
        m_ = re.search(r"RMS of the fit \(final\)\s*=\s*([0-9.eE+-]+)", open(f"{d_}/fit.cro", errors="replace").read())
        row["fit_rms"] = float(m_.group(1)) if m_ else None
    dd = {}
    for ln in open("cache/critic2/diffs.jsonl"):
        r_ = json.loads(ln)
        dd[(r_["pid"], r_["cand"])] = r_["diff"] if r_["diff"] is not None else 1.0
    plan_ = json.load(open("cache/critic2/plan.json"))
    ctl = {}
    for ln in open("cache/critic2/control.jsonl"):
        r_ = json.loads(ln)
        od = r_["diff"] if r_["diff"] is not None else 1.0
        sl = [dd[(r_["pid"], c)] for c in plan_[r_["pid"]]["shortlist"][:k_c2] if c not in plan_[r_["pid"]]["truth"]]
        ctl[r_["pid"]] = {"own": r_["cand"], "diff": rd(od), "rank_in_shortlist": int(sum(x < od for x in sl)) + 1,
                          "shortlist_best_diff": rd(min(sl))}
    inj = {}
    for ln in open("cache/critic2/inject.jsonl"):
        r_ = json.loads(ln)
        od = r_["diff"] if r_["diff"] is not None else 1.0
        sl = [dd[(r_["pid"], c)] for c in plan_[r_["pid"]]["shortlist"][:k_c2]]
        inj[r_["pid"]] = {"answer": r_["cand"], "diff": rd(od), "rank_if_shortlisted": int(sum(x < od for x in sl)) + 1}
    for row in ex["critic2"]["per_pattern"]:
        row["control"], row["injected"] = ctl.get(row["id"]), inj.get(row["id"])
    scc = load("results/sc_conditions.json")
    hp = np.array([scc.get(x["id"].replace(".json", ""), {}).get("pressure_series", False) for x in meta])
    amb = (R["none|sc|lnq_bkg"] >= 0) & ~hp
    spl = {"n_pressure_series_patterns": int((hp & (R["none|sc|lnq_bkg"] >= 0)).sum()),
           "n_pressure_series_compounds": int(len(set(stems[hp & (R["none|sc|lnq_bkg"] >= 0)]))),
           "n_ambient_patterns": int(amb.sum()), "n_ambient_compounds": int(len(set(stems[amb])))}
    hpsel = hp & (R["none|sc|lnq_bkg"] >= 0)
    for name, r in split_ranks.items():
        spl[name] = {"macro_top1": macro(np.where(amb, r, -1), 1, stems, cboot), "hits": int((r[amb] == 0).sum()),
                     "macro_top10": macro(np.where(amb, r, -1), 10, stems, cboot),
                     "median_rank": float(np.median(r[amb]) + 1) if name != "critic2" else None,
                     "frac_rank_gt_1000": rd((r[amb] >= 1000).mean()) if name != "critic2" else None,
                     "hits_pressure_series": int((r[hpsel] == 0).sum()),
                     "pressure_series_macro_top1": macro(np.where(hpsel, r, -1), 1, stems, cboot)}
    for x in (m, SEED2, ABLATION, ORIGINAL):
        rr = main_ranks(x)["sc|model_bkg"]
        spl[f"model_{x}"] = {"macro_top1": macro(np.where(amb, rr, -1), 1, stems, cboot), "hits": int((rr[amb] == 0).sum()),
                             "macro_top10": macro(np.where(amb, rr, -1), 10, stems, cboot),
                             "median_rank": float(np.median(rr[amb]) + 1), "frac_rank_gt_1000": rd((rr[amb] >= 1000).mean()),
                             "pressure_series_macro_top1": macro(np.where(hpsel, rr, -1), 1, stems, cboot)}
    for td in ("std", "sc"):   # 28 (CPU) and 21 (GPU) break float ties differently: at most one place apart
        a_, b_ = main_ranks(m)[f"{td}|model_bkg"], R[f"none|{td}|model_bkg"]
        assert np.abs(a_ - b_).max() <= 1 and ((a_ == 0) == (b_ == 0)).all(), td
    spl["model"] = spl[f"model_{m}"]
    for a, b_ in (("lnq", "pearson"), ("wcc_1", "pearson"), ("critic2", "lnq")):
        spl[f"paired_{a}_minus_{b_}"] = paired_macro(np.where(amb, split_ranks[a], -1), np.where(amb, split_ranks[b_], -1), stems)
    tol_ = ["cos_zs", "wcc_1", "wcc_2", "lnq"]
    spl["paired_among_tolerant"] = {f"{a}-{b_}": paired_macro(np.where(amb, split_ranks[a], -1), np.where(amb, split_ranks[b_], -1), stems)
                                    for i, a in enumerate(tol_) for b_ in tol_[i + 1:]}
    spl["paired_vs_pearson"] = {a: paired_macro(np.where(amb, split_ranks[a], -1), np.where(amb, split_ranks["pearson"], -1), stems)
                                for a in tol_}
    sup = np.zeros(nq, bool)
    sup[[qpos[s["id"]] for s in ex["sc_cell_mismatch"]["supercell"]]] = True
    hot_ = ~np.isnan(dT) & (dT > 50)
    spl["cells_differ"] = {"n": int(large.sum()), "pressure_series": int((large & hp).sum()),
                           "dT_gt_50_not_pressure": int((large & ~hp & hot_).sum()), "supercell": int((large & sup).sum()),
                           "none_of_these": int((large & ~hp & ~hot_ & ~sup).sum()),
                           "none_of_these_ids": [meta[i]["id"].replace(".json", "") for i in np.nonzero(large & ~hp & ~hot_ & ~sup)[0]]}
    for row in ex["critic2"]["per_pattern"]:
        row["pressure_series"] = bool(scc.get(row["id"], {}).get("pressure_series", False))
    ex["sc_ambient_split"] = spl
    pp_ = ex["critic2"]["per_pattern"]
    for key, field in (("control", "rank_in_shortlist"), ("injected", "rank_if_shortlisted")):
        summ = {}
        for tag, want in (("ambient", False), ("pressure_series", True)):
            rows_ = [r_ for r_ in pp_ if r_.get(key) and r_["pressure_series"] == want]
            summ[tag] = {"n": len(rows_), "rank1": sum(r_[key][field] == 1 for r_ in rows_),
                         "top3": sum(r_[key][field] <= 3 for r_ in rows_),
                         "median_rank": float(np.median([r_[key][field] for r_ in rows_])) if rows_ else None}
        ex["critic2"][f"{key}_summary"] = summ
    npk = {t_: [r_["n_peaks"] for r_ in pp_ if r_["pressure_series"] == (t_ == "pressure_series")] for t_ in ("ambient", "pressure_series")}
    ex["critic2"]["n_peaks_range"] = {t_: [int(min(v)), int(max(v)), float(np.median(v))] for t_, v in npk.items()}
    main_r = {pid: v["critic2"] for pid, v in c2["rank_by_pattern"].items()}
    for tag, suf in (("x2_zero_offset", "_zs"), ("x3_dac_refit", "_x3")):
        v_ = load(f"results/critic2_rerank{suf}.json")
        var_r = {pid: x["critic2"] for pid, x in v_["rank_by_pattern"].items()}
        changed = v_["meta"]["patterns_changed_rank"]
        rerun = {(r_["pid"], r_["cand"]): r_ for r_ in map(json.loads, open(f"cache/critic2/diffs{suf}.jsonl"))}
        c2mod = import_module("29_critic2_rerank")
        truth_c2 = {pid: set(pp["truth"]) for pid, pp in json.load(open("cache/critic2/plan.json")).items()}
        ex[tag] = {"top1": rnd(v_["critic2_vcgpwdf"]["top1"]), "top10": rnd(v_["critic2_vcgpwdf"]["top10"]),
                   "paired_minus_main": rnd(v_["paired_variant_minus_main"]),
                   "hits": sum(r_ == 0 for r_ in var_r.values()), "hits_main": sum(r_ == 0 for r_ in main_r.values()),
                   "gains": sorted(p for p in changed if var_r[p] == 0), "losses": sorted(p for p in changed if main_r[p] == 0),
                   "n_rank_changed": len(changed), "n_failed": v_["meta"]["n_failed_comparisons"],
                   "n_comparisons_rerun": len(rerun),
                   "failed_rerun": sorted([r_["pid"], r_["cand"], r_["cand"] in truth_c2[r_["pid"]]]
                                          for r_ in rerun.values() if r_["diff"] is None),
                   "cpu_h_rerun": rd(sum(c2mod.cpu_seconds(r_["cpu"]) for r_ in rerun.values() if r_["cpu"]) / 3600)}
    zo = load("results/critic2_zero_offsets.json")
    lab = [abs(z["delta_native_deg"]) for z in zo.values() if abs(z["wavelength"] - 1.5418) < 0.01]
    ex["x2_zero_offset"]["delta"] = {"n": len(zo), "n_lab_cu": len(lab), "median_abs_lab_cu_deg": rd(np.median(lab)),
                                     "median_abs_cu_equiv_deg": rd(np.median([abs(z["delta_cu_equiv_deg"]) for z in zo.values()])),
                                     "n_at_edge": sum(z["at_edge"] for z in zo.values())}
    # X3: the refitted pressure-series patterns, own-entry control and injection before -> after (22's block)
    dd3 = {}
    for f_ in ("diffs.jsonl", "diffs_x3.jsonl"):
        for ln in open(f"cache/critic2/{f_}"):
            r_ = json.loads(ln)
            dd3[(r_["pid"], r_["cand"])] = r_["diff"] if r_["diff"] is not None else 1.0
    old = {row["id"]: row for row in ex["critic2"]["per_pattern"]}
    v3 = load("results/critic2_rerank_x3.json")["rank_by_pattern"]
    rows3 = []
    for ln in open("cache/critic2/control_x3.jsonl"):
        r_ = json.loads(ln)
        od = r_["diff"] if r_["diff"] is not None else 1.0
        sl = [dd3[(r_["pid"], c)] for c in plan_[r_["pid"]]["shortlist"][:k_c2] if c not in plan_[r_["pid"]]["truth"]]
        o_ = (old.get(r_["pid"]) or {}).get("control") or {}
        nr = v3[r_["pid"]]["critic2"]
        rows3.append({"id": r_["pid"], "stem": old[r_["pid"]]["stem"], "own": r_["cand"], "diff": rd(od),
                      "rank_in_shortlist": int(sum(x < od for x in sl)) + 1, "old_diff": o_.get("diff"),
                      "old_rank_in_shortlist": o_.get("rank_in_shortlist"), "rerank_rank": nr + 1 if nr < k_c2 else None,
                      "old_rerank_rank": old[r_["pid"]]["critic2_rank"], "old_n_peaks": old[r_["pid"]]["n_peaks"]})
    inj3 = {}
    for ln in open("cache/critic2/inject_x3.jsonl"):
        r_ = json.loads(ln)
        od = r_["diff"] if r_["diff"] is not None else 1.0
        sl = [dd3[(r_["pid"], c)] for c in plan_[r_["pid"]]["shortlist"][:k_c2]]
        o_ = (old.get(r_["pid"]) or {}).get("injected") or {}
        inj3[r_["pid"]] = {"answer": r_["cand"], "diff": rd(od), "rank_if_shortlisted": int(sum(x < od for x in sl)) + 1,
                           "old_rank_if_shortlisted": o_.get("rank_if_shortlisted")}
    x3 = ex["x3_dac_refit"]
    x3["control"] = sorted(rows3, key=lambda r_: r_["id"])
    x3["control_rank1"] = sum(r_["rank_in_shortlist"] == 1 for r_ in rows3)
    x3["control_rank1_old"] = sum(r_["old_rank_in_shortlist"] == 1 for r_ in rows3)
    x3["control_median_diff"] = rd(np.median([r_["diff"] for r_ in rows3]))
    x3["control_median_diff_old"] = rd(np.median([r_["old_diff"] for r_ in rows3 if r_["old_diff"] is not None]))
    x3["injected"] = inj3
    x3["injected_rank1"] = sum(v["rank_if_shortlisted"] == 1 for v in inj3.values())
    x3["injected_rank1_old"] = sum(v["old_rank_if_shortlisted"] == 1 for v in inj3.values())
    x3["n_peaks"] = {r_["pid"]: r_["n_peaks"] for r_ in json.load(open("cache/critic2/x3_refits.json"))}
    wf_rows = [x for x in meta if x["n_other"] > 0]
    ex["within_family_chance"] = rd(np.mean([x["n_same_in_lib"] / (x["n_same_in_lib"] + x["n_other"]) for x in wf_rows]))
    ex["model_probe_check"] = load(f"results/d3/model_probe_check_{m}.json")
    ex["wrong_top1_audit"] = load("results/wrong_top1_audit.json")
    out["extra"] = ex
    out["critic2_rerank"] = rnd(c2)

    # ---- robustness of the headline: labels, anchors, intensity convention, library package, external sets ------
    rob = {}
    rob["label_sensitivity"] = rnd(load(f"results/d3/label_sensitivity_{m}.json"))
    ac = load("results/anchor_checks.json")
    rob["anchor_checks"] = {"every_anchor": {k: ac["every_anchor"][k] for k in ("patterns_checked", "comparisons",
                                                                                "patterns_with_new_same_form")},
                            "new_same_form": [dict(id=r_["id"], **x) for r_ in ac["every_anchor"]["rows"] for x in r_["new_same_form"]],
                            "lenient_single_flag": dict(Counter(
                                "+".join(sorted({f for a in r_["anchors"] for f in (a["single_flag_15"] or ["more_than_one"])}))
                                for r_ in ac["lenient"]["rows"])),
                            "lenient_rows": [{"id": r_["id"], "h_omitted": r_["h_omitted"],
                                              "opxrd_components": sorted(set(r_["opxrd_components"])),
                                              "anchor": r_["anchors"][0]["refcode"], "n_anchors": len(r_["anchors"]),
                                              "csd_components": sorted(set(r_["anchors"][0]["csd_components"])),
                                              "single_flag_15": r_["anchors"][0]["single_flag_15"]} for r_ in ac["lenient"]["rows"]]}

    def esets(comps):
        return {frozenset(e for e in re.findall(r"([A-Z][a-z]?)\d*", c) if e != "H") for c in comps} - {frozenset()}
    rob["anchor_checks"]["lenient_component_mismatches"] = sum(
        esets(r_["opxrd_components"]) != esets(a["csd_components"]) for r_ in ac["lenient"]["rows"] for a in r_["anchors"])
    ea = load(f"results/d3/rescore_main_every_anchor_{m}.json")
    rob["every_anchor_variant"] = {"n_queries": ea["meta"]["n_queries"],
                                   "macro": {k: rnd(ea["macro"][k]["top1"]) for k in ea["macro"] if "@" not in k}}
    base = main_ranks(m)
    for tag, ref in (("conv", "main"), ("binavg", "main"), ("r2", "main"), ("dT", "conv")):
        f = main_ranks(m, tag)
        rr = base if ref == "main" else main_ranks(m, ref)
        res = load(f"results/d3/rescore_{tag}_{m}.json")
        rob[tag] = {"macro": {k: rnd(res["macro"][k]) for k in res["macro"] if "@" not in k},
                    "per_pattern": {k: rnd(res["per_pattern"][k]) for k in res["per_pattern"] if "@" not in k},
                    f"paired_vs_{ref}": {k: {"top1": paired_macro(f[k], rr[k], stems),
                                             "n_top1_changed": int(((f[k] == 0) != (rr[k] == 0)).sum())}
                                         for k in f if "@" not in k}}
    rob["r2_ablations"] = {}
    f2 = main_ranks(m, "r2")
    for tag in ("r2_noB", "r2_nolp", "r2_noka2", "r2_nomask"):
        a = main_ranks(m, tag)
        res = load(f"results/d3/rescore_{tag}_{m}.json")
        rob["r2_ablations"][tag] = {"macro": {k: rnd(res["macro"][k]) for k in res["macro"] if "@" not in k},
                                    "paired_vs_r2": {k: paired_macro(a[k], f2[k], stems) for k in a if "@" not in k}}
    n_int = sum(1 for tag in rob["r2_ablations"] for k, v in rob["r2_ablations"][tag]["paired_vs_r2"].items()
                if v["ci"][0] > 0 or v["ci"][1] < 0)
    n_int += sum(1 for k, v in rob["r2"]["paired_vs_main"].items() if v["top1"]["ci"][0] > 0 or v["top1"]["ci"][1] < 0)
    rob["r2_n_intervals_excluding_zero"] = n_int
    rob["r2_n_intervals"] = sum(len(v["paired_vs_r2"]) for v in rob["r2_ablations"].values()) + len(rob["r2"]["paired_vs_main"])
    # Table 2: compound-weighted top-1 on the ambient single-crystal set and the any-same-form set under every variant
    known_wl = np.array([truth[x["id"]]["wavelength"] not in (None, "None", "") for x in meta])
    ea_r = main_ranks(f"every_anchor_{m}")
    amb_ea = (ea_r["sc|lnq_bkg"] >= 0) & ~hp
    variants = {"main": (base, amb, None), "conv": (main_ranks(m, "conv"), amb, None), "r2": (f2, amb, None),
                "dT": (main_ranks(m, "dT"), amb, None), "binavg": (main_ranks(m, "binavg"), amb, None),
                "strict": (base, amb, tier == "strict"), "known_wavelength": (base, amb, known_wl),
                "every_anchor": (ea_r, amb_ea, None)}
    t2 = {}
    for vname, (rk_, amb_, sel_) in variants.items():
        keep = np.ones(nq, bool) if sel_ is None else sel_
        t2[vname] = {k: {"ambient": macro(np.where(amb_ & keep, rk_[f"sc|{k}"], -1), 1, stems, cboot)["value"],
                         "std": macro(np.where(keep, rk_[f"std|{k}"], -1), 1, stems, cboot)["value"]} for k in KEYS}
    lsub = rob["label_sensitivity"]["subsets"]
    for vname, sub in (("main", "all"), ("strict", "strict_only"), ("known_wavelength", "known_wavelength")):
        for k in KEYS:   # 43 computes the same subsets independently
            assert abs(t2[vname][k]["std"] - lsub[sub][f"std|{k}"]["top1"]) < 6e-4, (vname, k)
    for k in KEYS:
        assert abs(t2["main"][k]["ambient"] - lsub["ambient"][f"sc|{k}"]["top1"]) < 6e-4, k
    rob["table2"] = t2
    rob["table2_n"] = {"ambient_patterns": int(amb.sum()), "every_anchor_ambient_patterns": int(amb_ea.sum()),
                       "strict_ambient_patterns": int((amb & (tier == "strict")).sum()),
                       "known_wavelength_ambient_patterns": int((amb & known_wl).sum())}
    # Kaduk external set: classical rows are model-independent; the model rows are this model's
    kd = load(f"results/d3/kaduk_main_{m}.json")
    ks = load("results/kaduk_summary.json")
    kr = npz(f"cache/real_bench_kaduk/rescore_main/ranks_{m}.npz")
    for i, row in enumerate(ks["patterns"]):
        row["ranks"] = {k: int(kr[k][i]) for k in kr.files if "@" not in k}
    rob["kaduk"] = {**slim(kd), "summary": rnd(ks)}
    # open COD replica: classical scores from the R5 run, the model from this model's COD embeddings
    cod = load(f"results/d3/rescore_main_cod_{m}.json")
    rob["cod"] = slim(cod)
    cdir = f"{bench.DATA}/cod"
    tc = json.load(open(f"{cdir}/opxrd_truth_cod.json"))
    sel_c = json.load(open(f"{cdir}/library_select.json"))
    n_all = sum(1 for ln in open(f"{cdir}/library_all_ids.txt") if ln.strip())
    ident = [x for x in tc if x["true_stems"]]
    rob["cod"]["construction"] = {
        "n_selected": sel_c["n_library"], "n_all": n_all, "n_added_candidates": n_all - sel_c["n_library"],
        "n_patterns": len(tc), "n_identified": len(ident),
        "n_identified_both": sum(1 for x in ident if truth[x["id"]]["true_stems"]),
        "n_identified_cod_only": sum(1 for x in ident if not truth[x["id"]]["true_stems"]),
        "n_other_compound": sum(x["n_other_compound"] for x in tc), "n_unknown_same_formula": sum(len(x["unknown"]) for x in tc),
        "sc_composition": dict(Counter((x["true_stems"] or ["?"])[0] for x in tc if any(not s["is_powder_study"] for s in x["same_form"])))}
    cover = {k for k, v in excluded.items() if v.startswith("covers")}   # 19 drops these whatever the library
    cq = [x for x in ident if x["id"] not in cover]                       # 19's query order for the replica
    zc = npz(f"cache/real_bench_cod/rescore_main/ranks_{m}.npz")
    assert len(cq) == len(zc["std|lnq_bkg"]) == cod["meta"]["n_queries"], (len(cq), cod["meta"]["n_queries"])
    st_c = np.array([x["true_stems"][0] for x in cq])
    si = np.array([bool({"B", "Si"} & set(pats[x["id"]]["elements"])) for x in cq])
    cb = np.random.default_rng(0).integers(0, len(np.unique(st_c[~si])), (2000, len(np.unique(st_c[~si]))))
    rob["cod"]["without_si_queries"] = {"n_si_queries": int(si.sum()), "si_ids": [x["id"] for x, s in zip(cq, si) if s],
                                        "macro_top1": {k: macro(np.where(~si, zc[k], -1), 1, st_c, cb)["value"]
                                                       for k in zc.files if "@" not in k}}
    out["robustness"] = rob

    # ---- effect ladder of every checkpoint (30 eval; corrected labels) ---------------------------------------------
    out["ladder"] = {x: rnd(load(f"results/d3/model_gap_{x}.json")) for x in (m, SEED2, ABLATION, ORIGINAL, "realism")}
    out["ladder_estimates"] = out["ladder"][m]["estimates"]

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {args.out}: {nq} queries, {n_uniq} compounds, model {m}; sweep {out['g0']['n_conditions']} distinct "
          f"conditions, model wins {out['g0']['n_model_wins']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
