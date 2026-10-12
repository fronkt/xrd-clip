# Negative-result paper — 2026-10-11 (Frank chose this path over benchmark-only / park / new idea)

Story: simple shift-corrected profile matching is the bar learned PXRD retrieval has to clear.
Evidence = the strong-baseline study (plan §10) + an open real organic test set (plan §11).
Step 1 hardens the real-data numbers before any drafting. All CPU, no spend.

### Step 1 — harden the real-data benchmark
- [x] 18 (CCDC interp): ground-truth sets. DONE 10-11: 189/196 identified (strict 154, lenient 35); 13 old labels were the wrong compound; 185/189 are self-retrieval (own Rietveld structure); 59 have a single-crystal same-form entry. Per pattern, enumerate the CSD refcode family of every
      cell-match candidate; CCDC PackingSimilarity vs the matched entry; 15/15 = same form
      (probe 10-11: IBPRAC redeterminations 25–296 K all 15/15, form II 1–2/15, S-ibuprofen 5/15).
      compare() = None → "unknown" → ignore list (never scored as an error). Extract CIFs of every
      family member outside the stick cache (gitignored cache/).
- [x] simulate the extras → cache/sticks_extra: 218/246 (28 have no CSD coordinates); Z′<1 duplicate-site merge fallback added
- [x] 19 (scripts/19_real_bench.py): DONE 10-12 on Vast (box 55451580, destroyed; $2.53) + local merge with `--chunk 2800`. Readout in plan Section 12a; table results/real_bench_table.md.
      95 in-corpus). Exact ranks via a pre-pass over family members (thresholds + within-family),
      then chunked streaming counts with 4 un-throttled workers, resumable per chunk.
      Splits: all / powder-study / independent; plus leave-own-refinement-out (LORO: powder-study
      same-form entries ignored, only single-crystal same-form entries count as truth) — this
      should grow the hard split well beyond n = 21.
      Measured cost (1 core, 196 q × 310k): cos_zs 0.37 h, wcc_1 0.63 h, wcc_2 0.82 h, lnq 7.0 h
      per variant → ~3–4 h wall on 4 workers.
- [x] compound-level (stem) bootstrap 95 % CIs; library-size curve (2k / 20k / 100k / full)
- [x] compound-weighted (macro) metrics: DONE 10-12 in 21 eval for std + sc, every correction (plan Section 12b)
- [x] hubness: diagnose + test a correction — DONE 10-12 (Step 1b; plan Section 12b): the "hub" was one compound (ZZZUWK)
- [x] commit table + aggregate json; plan Section 11 corrected, 12/12a added

### Step 1b — hubness correction (Frank 10-12: Digital Discovery; hubness fix on a Vast GPU before drafting)
Why a rerun: 19 kept only rank bits, so any score correction needs the full score rows again.
Design (pre-declared, so nothing is tuned on the 188 test patterns):
- primary = CSLS (Conneau et al. 2018), library side only: s'(q,r) = s(q,r) − ½·h_k(r), h_k(r) = mean of r's
  k = 10 highest scores against a PROBE set (the query-side term is constant per query, so ranks ignore it).
- probes = library entries, never the test patterns: 4000 clean simulated ("sim") + 4000 augmented with the
  training ranges and then bkg-preprocessed like the real queries ("aug"); sampled from entries whose stem is
  not a query family; same-stem probe/entry pairs are masked out of h.
- sensitivity (reported, not selected): k ∈ {1, 5, 20, 50}; library-side z-norm; mutual proximity (Gauss).
- hubness diagnosis: k-occurrence skew (N_10 over probes), top hubs + their peak counts.
- model gets the same correction locally (probe patterns → pattern encoder; library embeddings cached).
Steps:
- [x] `src/xrdclip/eval/wcc_torch.py`: torch port of WCCLibrary + Pearson as lag-sliced GEMMs (triangle
      folded into the reference, so 51 lags for cos/WCC, 16 × 63 for lnq); fp32, TF32 off
- [x] validate locally on CPU (max |diff| 9.5e-7, 9 keys x 397 entries): torch vs `strong_baselines.WCCLibrary` and vs prepass.npz special-entry scores
- [x] `scripts/21_hubness.py prep | score | eval | check`; 19's query/truth build + summary shared (19 smoke output identical after refactor)
- [x] smoke: `score --limit 200` on CPU, `eval` uncorrected ranks == cache/real_bench_smoke/ranks.npz (0 mismatches, 11 keys x 2)
- [x] box runner `scripts/box/hub_box.sh` (md5, self-stop, watchdog); upload = existing box_upload.tar + small ctx
- [x] Frank switched to MANUAL mode → RTX 4090 box 55604630: upload 8 min, score 761 s, pull 9 min (md5 OK), destroyed; $0.26
- [x] eval: uncorrected ranks reproduce ranks.npz (0 top-1 changes; 59 |Δrank| ≤ 3 near-ties); table + json written
- [x] plan Section 12b (+ correction banner on 12a finding 3) + commit/push

#### Review — Step 1b (2026-10-12)
- Worked: folding the WCC triangle into the reference turns the zero scan into 51 GEMMs, so 8,188 rows x 309k
  entries x 5 methods took 12 min on one 4090. CPU-equivalence was checked three times: prepass (≤ 1e-6), the
  smoke run against 19's brute-force ranks (exact), and the full-library ranks (no top-1 change).
- Finding that changes the paper: the 12a "hub" (GUSRIV x20) was ONE query compound, ZZZUWK, with 20 patterns.
  The CSLS per-pattern gain for cos (+0.15) is mostly ZZZUWK; compound-weighted the gains are +0.01–0.04 and none
  is significant. Lead the paper with macro top-1. On that measure ln q is best (0.87), not Pearson.
- Lesson -> tasks/lessons.md (count distinct query compounds before calling a decoy a hub).

### Step 2 — draft (after Step 1 numbers are in)
- [x] venue: Digital Discovery (Frank 10-12)
- [ ] outline check-in with Frank

---

# xrd-clip v2 continuation — 2026-10-10 (plan: docs/continuation-plan-2026-10.md)

Status (10-10): **G0 FAILED + real-data test FAILED → kill criterion met; learned-retrieval line STOPPED** (plan §10–11). Salvage = real organic benchmark + strong-baseline study.

- [x] P0 B1 de Gelder WCC + zero-shift scan (l = 0.5/1/2°) — built + tested (CSD API cross-check still open)
- [x] P0 B2 ln-q lag scan + 2θ zero scan — built + tested
- [~] P0 B3 torch VC-GPWDF — built + tested; unregularised it bends sibling cells onto the query (clean 0.71 vs fixed-cell 1.0, 40 fams); strain prior λ=0.003 → thermal 0.72. critic2 validation OPEN
- [ ] P0 B4 hand-crafted peak-feature vector index
- [x] P0 full sweep (35 conditions, Vast 3090 ~$0.25, destroyed) → **G0: model loses everywhere, non-overlapping CIs; smallest gap = anisotropic thermal (−0.12)**. See plan §10 + results/g0_table.md
- [ ] P1 recall@k ceiling (k 5–200) + rerank (B1/B2/B3/fusion GBDT) + accuracy-vs-time Pareto → Gate G1
- [ ] P2 ibuprofen diagnosis (bkg subtraction, B1/B3, emb→B3)
- [x] P2 Tier A: 196 opXRD organics matched to CSD (95 in corpus) → real eval: model t1 0.09 vs cosine+zero-scan 0.81 (results/real_eval_table.md)
- [x] P2 CSD powder-study CIFs: API CIFs carry NO raw profiles; 99 powder studies in test lib (36 in families, 93 DOIs) → refcodes/powder_study_test_refs.json = IUCr/SI scrape targets
- [x] P2 Tier C: Gmail DRAFTS made 10-10 (ICDD addressed; Johnson draft needs recipient) — Frank sends
- [ ] P2 Tier B: Purdue measurement — DEFERRED until after apps + STS (Frank 10-10); G2 runs on Tier A/C + CSD check
- [-] P3 retrain — CANCELLED (kill criterion)
- [-] P4 cross-encoder — CANCELLED unless a new idea targets the discrimination ceiling
- [ ] P5 paper (Digital Discovery recommended)

---

# xrd-clip URTC paper + pxrd-flow Phase-B startup — 2026-07-18

## Track 1 — URTC 5-pager (priority; deadline ~Aug 3)
Format facts (verified urtc.mit.edu): ≤5 pages hard, IEEE conference template, peer-reviewed
(originality/content/theme), eligibility incl. HS students in collegiate research programs (confirmed).

- [ ] Load ml-paper-writing skill guidance + IEEEtran template
- [ ] Assemble source material: next-steps.md findings, polymorph-pivot-plan.md,
      competitive-landscape-2026.md, results/*.json numbers
- [ ] Draft paper.tex (IEEEtran, two-column, ≤5pp):
      - Intro: polymorph-resolving retrieval wedge (XCCP/XtalNet pinned at chance by construction)
      - Related Work: XCCP, XtalNet, classical search-match; molecular-domain gap
      - Method: dual encoder (1D ResNet ↔ GNN), InfoNCE, physics-grounded augmentation,
        polymorph-aware sampler, family-safe splits
      - Results: clean headline (0.561 vs 0.380 chance) framed as context; LEAD with robustness:
        sign-decomposition panel; shift/strain sweeps; field average; ablation table
        (emergent vs learned = central defense); broadening = stated limitation
      - Conclusion + limitations (simulated-only eval, honest scope)
- [ ] Figures: polymorph.png sweep panels + combined.png (reuse; check legibility at column width)
- [ ] Verify EVERY citation via search (no hallucinated refs — skill's hard rule)
- [ ] Install tectonic, compile PDF, check ≤5pp
- [ ] Self-review pass (reviewer attacks: leak, clean-loss, sim-only)

## Track 2 — pxrd-flow Phase-B startup (after paper draft; decisions LOCKED 2026-07-18)
- **Corpus = symmc-flow csd_export (~8k)** (Frank, 2026-07-18) — fastest to conditioned run #1;
  Tier-1 family exclusion on top; 315k upgrade only if time.
- **GPU box = after paper draft** (Frank, 2026-07-18) — no idle billing.
- [ ] Inspect symmc-flow state contract (crystal-state parameterization x_1 = L,{t_i},{R_i},conformer)
- [ ] Build structure-state loader + Dataset composing PXRDConditioner (c_pxrd) + state encoder (x_1)
- [ ] Apply tier1 filter_training_refcodes() at family level to csd_export
- [ ] Then: provision box (Frank approves) → backbone baseline → E_p port → FiLM/CFG → run #1

## Review
(to fill as work completes)
