# Negative-result paper — 2026-10-11 (Frank chose this path over benchmark-only / park / new idea)

Story: simple shift-corrected profile matching is the bar learned PXRD retrieval has to clear.
Evidence = the strong-baseline study (plan §10) + an open real organic test set (plan §11).
Step 1 hardens the real-data numbers before any drafting. All CPU, no spend.

### Step 1 — harden the real-data benchmark
- [ ] 18 (CCDC interp): ground-truth sets. Per pattern, enumerate the CSD refcode family of every
      cell-match candidate; CCDC PackingSimilarity vs the matched entry; 15/15 = same form
      (probe 10-11: IBPRAC redeterminations 25–296 K all 15/15, form II 1–2/15, S-ibuprofen 5/15).
      compare() = None → "unknown" → ignore list (never scored as an error). Extract CIFs of every
      family member outside the stick cache (gitignored cache/).
- [ ] simulate the extras with the existing stage-2 code → cache/sticks_extra
- [ ] 19 (scripts/19_real_bench.py): full-library eval. Library = 308,862 cached sticks + extras. 196 patterns (not just the
      95 in-corpus). Exact ranks via a pre-pass over family members (thresholds + within-family),
      then chunked streaming counts with 4 un-throttled workers, resumable per chunk.
      Splits: all / powder-study / independent; plus leave-own-refinement-out (LORO: powder-study
      same-form entries ignored, only single-crystal same-form entries count as truth) — this
      should grow the hard split well beyond n = 21.
      Measured cost (1 core, 196 q × 310k): cos_zs 0.37 h, wcc_1 0.63 h, wcc_2 0.82 h, lnq 7.0 h
      per variant → ~3–4 h wall on 4 workers.
- [ ] compound-level (stem) bootstrap 95 % CIs; library-size curve (2k / 20k / full)
- [ ] commit table + aggregate json; update plan §11

### Step 2 — draft (after Step 1 numbers are in)
- [ ] venue + outline check-in with Frank (Digital Discovery vs J. Appl. Cryst. vs AI4Mat workshop)

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
