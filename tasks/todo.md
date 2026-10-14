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
- [x] eval: uncorrected ranks reproduce ranks.npz (0 top-1 changes; 118 |Δrank| ≤ 3 near-ties; first written as 59, a miscount); table + json written
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
- [x] outline check-in with Frank (10-12: "its good"; outline = plan 12b consequences + the 5-part outline sent 10-12)

Format: RSC markdown like mlip-dynstab (`paper/dd/manuscript.md` -> .docx via the journal-submission skill).
`paper/main.tex` stays the URTC paper. The URTC paper's copyright is IEEE's: cite it and reuse no text or figures from it.
Headline metric = compound-weighted (macro) top-1; per-pattern is secondary (lessons.md 10-12).
- [x] A1 Digital Discovery guidelines (rsc.org pages 403 -> many items search-snippet only; re-check before upload):
      Full paper, no page limit; abstract ~50-250 words; TOC graphic at revision; "Fig. 1"; ESI cited with a dagger;
      refs numbered superscripts, "Notes and references"; DAS + conflicts required, CRediT encouraged; AI use declared
      in Acknowledgements; code needs a DOI (Zenodo) and a DATA REVIEWER checks it reproduces the findings;
      **restricted data: "a representative dataset that enables similar conclusions ... must be made publicly
      available" -> refcode lists alone may not pass (Frank decision: open stand-in library, e.g. COD decoys + the
      opXRD structures)**; preprints OK; URTC proceedings: no explicit RSC wording -> disclose + cite in cover letter;
      APC GBP 2,200; Purdue RSC agreement runs to 2026-12-31 (Purdue e-mail, CC BY) -> acceptance likely after;
      first decision ~46 days
- [x] A2 reference ledger `docs/lit-2026-10/refs_dd_verified.md` + recency sweep: 16/28 fully supported, rest fixed in
      the text (RealBind is NOT inorganic-only: ~23 opXRD-CNRS organics in its test set; XCCP has a 773-pattern opXRD
      zero-shot test; VC-GPWDF naming; opXRD never says "Rietveld"). No organic-specific scoop through 10-12.
- [x] B1 `scripts/22_paper_numbers.py` -> `results/paper_numbers.json`: every manuscript number comes from
      results/*.json or hub_ranks.npz (macro top-1/top-10 with compound CIs, per-pattern, median ranks, model
      rank distribution, ZZZUWK case, compound counts, training-split overlap of the query compounds)
- [x] B2 `scripts/23_paper_figs.py` -> `paper/dd/figs/` (vector PDF + 300 dpi PNG):
      Fig 1 benchmark construction (schematic); Fig 2 simulated sweep, model vs best classical over 35 conditions;
      Fig 3 real benchmark, macro top-1 std + single-crystal with CIs, plus the library-size curve;
      Fig 4 ZZZUWK case (measured pattern vs its own entry vs the winning decoys); Fig 5 rank distributions, model vs classical
- [x] C1 `paper/dd/manuscript.md`: Abstract; 1 Introduction; 2 Benchmark; 3 Methods; 4 Results (simulated sweep;
      full library; single-crystal references; ZZZUWK + CSLS; why the model fails); 5 Discussion (what a learned
      method has to beat; limits); 6 Conclusions; Data and code availability; Author contributions; Conflicts;
      Acknowledgements; References
- [x] C2 `paper/dd/esi.md` (generated by `scripts/25_esi.py`): S1-S8 incl. model card, subgroups, paired diffs, audit
- [x] D1 `scripts/verify_claims.py`: 112/112 manuscript numbers verbatim from paper_numbers.json (premise checks too)
- [x] D2 two internal referee passes (powder + ML) -> draft v1 (commit 3ecd84f). Changed by the review:
      model fails on the measured-to-simulated step (scripts/26: clean own entries top-1 0.21 / median 5 vs measured
      0.01 / ~6,000; clean probes 0.42); single-crystal set leads (ln q, WCC 1° beat fixed Pearson +0.18/+0.20,
      paired CIs > 0); ZZZUWK texture claim withdrawn (intensities not monotonic in the angle to c; own entries
      disordered); sweep = 34 distinct conditions (clean == offset 0); ground-truth audit (scripts/27): 1 mislabel
      in 454 wrong top-1s (GOFWUV for pattern_100/QQQAUJ); sc failures track cell mismatch (≤1 %: 0.71, >1 %: 0.08)
- [x] D3 `bash paper/dd/build.sh` -> manuscript.docx + esi.docx (pandoc + RSC CSL, 39 refs)
- [ ] D4 check in with Frank (draft v1 + the decisions below)

Revision options the referees raised that change numbers or scope (Frank's call; none started):
- R1 ln q bug: its 16 zero offsets skip 0 (nearest ±0.02°), and the ln q grid is coarse at high angle -> handicaps
     ln q (clean sweep 0.88). Fix = odd offset grid + finer ln q axis; re-run 19 for lnq (local CPU, hours) and the
     sweep lnq column (local, minutes); CSLS for lnq would need the GPU rescoring again (Vast, manual mode, ~$0.1).
- R2 library fidelity: Debye-Waller factor, wavelength-correct Lorentz-polarisation for the 68 converted patterns,
     Kα2 stripping, better background; re-score the full library (Vast GPU ~$0.3). Likely raises classical numbers.
- R3 VC-xPWDF / critic2 re-rank of the 59 single-crystal patterns (top-100: ~59 x 100 x 31 s = ~51 CPU-h) -> tests
     whether the "open problem" is open for the best classical method (B3 critic2 validation is the same task).
- R4 learned side: pattern-to-pattern retrieval with the existing encoder (local, ~1 h), and/or a retrain with Kα2 +
     instrument-profile realism (Phase 3, ~$2-25). Without it, every claim stays scoped to this one model (as in v1).
- R5 Digital Discovery restricted-data policy: an open stand-in (opXRD queries vs a COD-simulated organic library;
     the opXRD CNRS structures ARE COD entries) so the data reviewer can reproduce without a CSD licence.
- R6 statements: AI-use wording (his own), corresponding e-mail (Purdue OA agreement ends 2026-12-31, applied at
     acceptance), checkpoint release (ask CCDC?), ChemRxiv preprint, final title.

### Step 3 — revision R1–R5 (Frank 10-12: "Do 1-5")
Rule for every item: the analysis and a written prediction are committed here BEFORE its numbers exist
(lessons 10-12). Components of a package are reported as one-off ablations, never selected. v1 numbers stay
reproducible (old code paths kept behind a flag). Readouts go to plan Section 13; numbers to paper_numbers.json.
Local machine: 8 cores, 32 GB, no CUDA (measured 10-12: render 11 ms/entry/core; torch-CPU scoring of 188 queries
vs 309k: cos/WCC 4-6 min per variant, ln q v1 75 min). No CSD-derived upload anywhere without manual mode.

R1 ln q fix
- [ ] zero offsets -0.32..+0.32 step 0.04 (17, includes 0; v1: 16, skipped 0); area-preserving resampling onto
      the ln q axis (bin-average via the cumulative integral; v1 point-sampled a 0.05 deg-step axis at high angle
      and could miss peak tops by up to ~16 %). Same l = 0.01, +-3 % strain. v1 kept as `lnq_v1`.
- [ ] checks: self-score 1.0 at offset 0; a 1 % strained copy is found at lag ln(1.01); torch == numpy (<1e-5)
- [ ] 188 queries x full library locally (scripts/28) -> ln q ranks; hybrid (model top-K -> ln q) recomputed
- [ ] G0 sweep ln q column recomputed (same seeds, ln q only)
- [ ] CSLS for ln q: needs the 8k probe rows -> box session (B)
- Prediction: the clean sweep rises (v1 0.88); real data moves little (an 0.02 deg offset error sits inside the
  triangle tolerance), macro top-1 changes within +-0.05.

R2 measured-aware library and comparison (one pre-declared package + one-off ablations)
- (a) overall Debye-Waller factor on library intensities, I x exp(-2B s^2), s = sin(theta)/lambda; B = 8 pi^2 x
      median non-H mean U_eq over 5,000 random library CIFs measured at 283-303 K, query families excluded
      (computed from cache/csd before any scoring)
- (b) Lorentz-polarisation: non-Cu queries are converted to Cu K-alpha1 geometry, I x LP_Cu(theta_Cu) /
      LP_orig(theta_orig); P = (1 + cos^2 2theta)/2 for lab tubes, 1 for synchrotron (assumption, stated)
- (c) K-alpha2 as a scanned nuisance for lab-tube queries (Cu, Co, Mo; 3 unlabelled -> Cu): score = max over a
      K-alpha1 library and a K-alpha1 + K-alpha2 library (I2/I1 = 0.5, the tube's lambda2/lambda1), because
      monochromatisation is not recorded; synchrotron queries use K-alpha1 only
- (d) compare over each query's measured range only (21 % of queries do not cover 5-50 deg): reference norms
      are taken over the query's coverage mask (unshifted mask under the zero scan; stated approximation)
- [ ] B from the CIFs; [ ] scorer (scripts/28) with (a)-(d) switches; [ ] smoke vs v1 with all switches off (exact)
- [ ] full package, all classical scores, local; [ ] ablations (package minus one) local if time, else box session
- Prediction: classical sc macro top-1 rises for every score; (b) helps the 68 non-Cu patterns most; (a) small;
  the classical-vs-model gap is unchanged in sign and size.

R3 critic2 VC-GPWDF re-rank (the published variable-cell method)
- Method fixed 10-12 after the research + refute pass: critic2 1.4.155 (official Windows build, NLopt on) ships
  VC-GPWDF (Otero-de-la-Roza, J. Appl. Cryst. 57, 1401, 2024), which supersedes VC-xPWDF (needs indexing; not
  in 1.4 for experimental patterns). Pipeline as published: XRPD BACKGROUND (20 knots) -> XRPD FIT (defaults)
  -> COMPAREVC cand.cif pattern.peaks GLOBAL QUICK ALPHA 1 (the paper's alpha; code default 0.5) LAMBDA =
  measurement wavelength. No zero-offset correction (none in COMPAREVC; the paper used none). One added
  step: each measured pattern is cut to the benchmark's d-spacing window (Cu K-alpha1 5-50 deg, d = 1.82-17.7 A),
  which every other score sees too (first test: synchrotron naphthalene fit RMS 1644 -> 29 after the cut).
- [x] critic2 on Windows: release 1.4 zip, SHA256 OK, features line lists NLopt
- [ ] validate on own-entry + wrong pairs; time each
- [ ] shortlist = top-100 of fixed ln q for the 59 sc patterns (truth outside the shortlist = miss; report
      recall@100); critic2 score for each (pattern, candidate) locally; re-rank
- Prediction: lifts the > 1 % cell-mismatch group (ln q 0.08) but is capped by recall@100; may promote sibling
  polymorphs (B3 note: free cells bend a sibling onto the query).

R4 learned side
- [ ] R4a gap ladder (cheapest mechanism test first, local): the 188 own entries rendered clean (top-1 0.21) ->
      add ONE measured effect at a time (coverage mask, query background removal, K-alpha2, wavelength
      conversion, measured-like profile width, noise) -> which step takes the model to 0.01
- [ ] R4b pattern-to-pattern retrieval with the existing pattern encoder (library rendered clean, local)
- [x] R4a ladder DONE 10-12 (results/model_gap.json). Model, std per-pattern top-1 / median rank: own clean 0.21/5;
      one effect: mask 0.20/6, width 0.22/7, K-alpha2 0.14/33, bkg removal 0.18/130, noise 0.17/11, zero 0.21/6;
      all six 0.11/176; + Debye-Waller B 4.45 0.04/744; + jitter 0.3 0.10/162; + DW + jitter 0.03/746; measured
      0.01/5,988. Fixed-grid Pearson: all + DW + jitter 0.78 vs measured 0.82 (the modelled effects reproduce its
      measured level; for the model they explain about a third of the log-rank gap). p2p measured 0.09/1,622.
- [x] R4b pattern-to-pattern with the existing encoder: measured std top-1 0.09 (macro 0.13), sc 0.03
- [ ] R4c realism retrain (box A), DESIGN FIXED 10-12 before training: same split, architecture, hyperparameters,
      60 epochs, polymorph batches, checkpoint = best clean val in-batch top-1, one seed (all as the original).
      `--realism` (augment.Realism): K-alpha2 copy p 0.6 (I2/I1 0.5); overall B ~ U(0, 8) A^2; per-reflection
      log-normal intensity jitter sigma ~ U(0, 0.5); 2theta range crop p 0.25 (start U(5, 12), end U(30, 50));
      input = the queries' own background removal (eval.real_patterns.preprocess), on top of the original Ranges.
      Eval: measured std/sc (macro + per pattern, median rank), the R4a ladder, p2p with the new pattern encoder.
- Prediction R4c (written before training): measured std per-pattern top-1 rises from 0.01 to 0.05-0.20 and the
  median rank falls from ~6,000 to 50-1,000; macro top-1 stays <= 0.3, far below every classical score
  (0.83-0.87); own entries with all effects + DW + jitter recover to about the clean level (~0.2). If measured
  top-1 stays < 0.03, the remaining gap is not the modelled effects.

R5 open stand-in (Digital Discovery restricted-data clause)
- [ ] COD route (research agent) -> COD organic CIFs + metadata (open data, CC0: a rented box is fine)
- [ ] same organic filter + simulator as the CSD corpus -> COD sticks
- [ ] truth: per opXRD pattern, COD cell + element search, packing similarity (CCDC API, local) -> same-form
      sets; powder vs single-crystal from CIF tags
      - [x] 2,631 candidate CIFs fetched (10-13); scripts/32 done (1,312 s) -> 31 truth: 191/196 identified
        (all 189 CSD-identified + pattern_53, pattern_466 COD only); anchors strict 187, lenient 4, none 5;
        own powder entry reproduced (rmsd < 0.05) 170; single-crystal same-form 35 patterns (CSD 59);
        other_form 17 (CSD 315: COD polymorphs fall outside the 5 % cell search and stay library entries);
        2,670 other-compound candidates back in the library, 23 same-formula unknowns ignored
      - Fixed 10-13 before any replica number: 18's "unknown = ignored" assumed family-built sets (all 134 CSD
        unknowns share the stem); COD candidates include OTHER compounds that PackingSimilarity cannot compare
        (partial: 273 of 281 unknowns have another formula). Rule (31 `truth`): candidate whose reduced non-H COD
        sum formula matches none of the same-form entries' = ordinary library entry (an error when it ranks
        above); same-formula non-comparable = unknown (ignored, as in 18). Compound labels for the bootstrap/
        macro = the main benchmark's CSD stems (formula labels merge isomers: 114 vs 124 groups); COD-only ->
        'cod:<id>'. Scored with 28 `--tag main` (ln q v2, other scores as v1, no R2 switches) + the model.
- [ ] classical + model scores -> open-replica table; release COD IDs, simulated library, truth sets, ranks
- Prediction: same ordering (classical >> model); absolute numbers differ.

B. Box session (manual mode, one CSD-derived upload): R4c training; ln q CSLS rescore; R2 ablations if not done.
   Bundles cache/box_rev/rev_A.tar (md5 c72e83e9...) + rev_B.tar (md5 4972cdac...); A = rev_gpu_box.sh,
   B = rev_cpu_box.sh then cod_box.sh. After the pull (all local, PYTHONPATH=src):
   - [ ] A out.tar -> logs, results/sbv2_*.json, runs/xrdclip_realism, cache/real_bench_realism,
         cache/real_bench/full_lnq2, cache/real_bench/rescore_r2_*/ranks.npz; touch PULLED; destroy A
   - [ ] sweep merge: G0_STEM=results/strong_baselines_v2 python scripts/13_g0_report.py results/sbv2_w*.json
         (-> _v2.json/_ranks.npz/.png + g0_table_v2.md; v1 files untouched). Draws differ from v1 (crc32 seeds;
         v1 used salted hash, irreproducible - disclosed), so compare v2 to its own lnq_v1 column, not to v1
   - Box session 10-13: A = 27542310 (4090 WA, 128-core cgroup, $0.52/h, instance 55706679), B = 54197769
     (5060 Ti NO, $0.40/h, instance 55706682); both launched 08:32 / 09:31 UTC. On A the 21 ln q v2 score hit CUDA
     OOM at chunk 1 (R4c train ~17 GB + 6 sweep workers on the 24 GB card); sweep, ablations and training
     unaffected. Partial cache/real_bench/full deleted on the box (so out.tar has no full_lnq2); scripts/box/
     rev_gpu_retry.sh reruns 21 score alone after RUN_DONE into cache/real_bench_lnq2r -> /workspace/
     out_retry.tar; pull it, then touch RETRY_PULLED (else it self-stops 2 h after RETRY_DONE).
   - OUTCOME 10-13 (nobody pulled: the manual-mode session sat on an approval ~11 h; lessons.md):
     A: train (best val 0.911), embeddings, sweep, 4 ablations all EXIT 0; the watchdog then stopped A at 11:24Z
     (its pattern "[e]mbed_lib" never matched the heredoc `python -`, so the 30 idle minutes ran during the
     embedding) before out.tar/RUN_DONE, so the ln q retry never ran. B: COD rsync 109,570/109,570 + simulation
     done (COD_DONE 14:40Z, cod_out.tar 426 MB); cod_box.sh's own 2-h self-stop then stopped B at 16:40Z with
     critic2 `run` still going (two queues on one box, each able to stop it). Both GPUs re-rented; restarts
     queued -> cancelled. Recovery: relay 55811910 (RTX 2070 ON, $0.07/h, 3-h self-stop) + sequential
     `vastai copy` (8 concurrent copies failed with Permission denied). Credit $1.66 after the boxes.
     Pulled + merged locally: sweep v2 (strong_baselines_v2.*, g0_table_v2.md), rescore_r2_no*.json.
     STILL OPEN: CSLS ln q v2 needs a GPU rerun (CSD upload -> manual mode); critic2 run to resume (local
     Windows critic2 is validated; 29 `run` skips done pairs).
   - 10-13 evening (Frank: "go ahead on critic2 and ln q CSLS"): box A's machine freed up -> restarted it,
     scripts/box/lnq2_box.sh ran 21 score alone (405 s) with a local pull loop armed at launch; all outputs
     md5-verified, box A destroyed. B outputs via relay (copies one at a time; copied files read as ????????
     until a relay stop/start, lesson #20), md5-verified; relay + B destroyed. No instances left; credit $1.23.
     CSLS ln q v2 done (86def71). critic2 on B ran 0 comparisons: the peak file is fit.peaks on Linux AND
     Windows, the code looked for expt-bg.peaks (fixed); 55/59 box fits reused (inputs md5-identical; a
     Windows refit of pattern_655 gives RMS 0.556 vs 0.567, same 1.4.155) + 4 fitted locally. Local run:
     2,750 + 200 comparisons, 6-8 critic2 processes, unthrottled children (19 unthrottle(pid)); resumable.
     R5: COD sticks 108,902/109,570 (99.4 %); scripts/33_cod_embed.py + 28 main score/eval running locally.
   - 10-13 ~19:00: Claude Code's memory reaper killed all four local jobs (R4c pemb at 172k/309k, R5 embed after
     shards 000-001, critic2 run after 2 comparisons). Cause found on restart: 33 reused one worker for every
     shard (~1.8 GB growth per shard; 19 uses maxtasksperchild=1) -> fixed 4fd49c1. Fits: 57/59 have fit.peaks;
     pattern_887/949 hit the 1-h cap (they add ~1,000-1,400 peaks one at a time; pattern_180 took 1 h 45 min).
     20:40 restart (Frank "Go ahead"): chain R5 embed -> score -> eval -> R4c pemb -> eval, one torch job at a
     time (scratchpad chain_r5_r4c.sh, log cache/chain_r5_r4c.log); critic2 run --procs 3 (2,848 comparisons,
     cache/critic2_run.log); fit retry 887/949 --fit_timeout 14400 (cache/critic2_fit_retry.log). When the two
     fits land: stop the run (driver + critic2 children) and restart it so rank-major order picks them up.
     21:24 R5 DONE (250632d): results/rescore_main_cod.json, 190 queries vs 108,902 COD entries. std (125
     compounds) ln q top-1 0.901 [0.849, 0.952], cos_zs 0.885, model 0.016 [0, 0.040]; sc only 9 compounds (ln q
     0.556 [0.222, 0.889], model 0). Same picture as the CSD benchmark (CSD r1 std ln q 0.875). pattern_949 fit
     landed (RMS 1.57). critic2 COST PROBLEM: comparevc global cost ~ V^2.15 (16 s at 223 A^3, 230 s at 770,
     QOGHOL 5,264 A^3 > 40 min); K = 50 ~ 3,100 CPU-h uncapped -> asked Frank: K = 10 + 4-h cap on a rented
     CPU box (~490 CPU-h) vs K = 5 + 1-h cap locally (28 % capped). Local run continues rank-major meanwhile.
     Frank topped up Vast ($11.22) -> K = 10 box route (6702b7e: 29 --k 10 default, --timeout 14400, --pids;
     scripts/box/critic2_k10_box.sh). ln q v2 sc recall@10 0.542 (32/59) vs @50 0.61 (36/59), top-1 0.424.
     Local run stopped (3 comparisons kept). Upload cache/box_rev/c2k10.tar (13.7 MB, md5 c1f0d1b5..., 420
     CIFs, 59 peak lists: 887 landed locally 21:38, RMS 11.06; 587 comparisons). Launch = scratchpad launch_c2k10.sh <id> (upload, verify,
     launch, pull loop -> cache/box_rev/pullC/). Needs MANUAL mode (CSD CIFs). Offer seen: 51326865, 128-core
     EPYC 7B12, $0.31/h; onstart guard: stop if not launched in 2 h, hard stop at 24 h.
     LAUNCHED 10-13 21:42 local (01:42Z), manual mode: 51326865 was gone -> offer 49717811 = instance 55845149
     (EPYC 7W83, 256-core cgroup, 251 GB, Taiwan, $0.681/h), root@211.21.106.81 -p 46075. Hard stop lowered to
     14 h (credit $11.22 lasts ~16 h). Box: critic2 1.4.155, 0 missing fits, 254 critic2 processes. Pull loop =
     background launch_c2k10.sh (log cache/box_rev/launch_c2k10.log, polls every 10 min) -> cache/box_rev/pullC/
     out.tar; the box self-stops on PULLED. AFTER: back up cache/critic2/diffs.jsonl, take the box's (a
     superset), 29 eval --k 10 -> results/critic2_rerank.json, destroy 55845149.
     10-13 23:55: K = 10 done but 1 (pattern_730/GOGZOU; answer EPIVEF not in its top 10, so it moves no metric).
     589 rows all ok, 76 CPU-h total (median 2.3 CPU-min, max 1.95 h), ~6x below my two-point estimate (refit cost
     ~V^1.8). Preview eval (58 patterns, cache/box_rev/preview/): critic2 sc macro top-1 0.533 [0.35, 0.70] vs ln q
     0.455 [0.276, 0.633], paired +0.079 [-0.055, +0.212]; per pattern 27 vs 25 of 58. Local diffs.jsonl backed up
     to cache/critic2/diffs_local_3.jsonl and replaced by the preview.
     Frank (AskUserQuestion): "Extend to K = 50". 00:01 K = 50 extension on the SAME box (8bad8fa): K = 10 runner
     shells (PIDs 431/432) killed, its python + GOGZOU left running; c2k50.tar (1,603 more CIFs, md5 b44ac0c7...)
     + scripts/box/critic2_k50_box.sh (log /workspace/box50.log); ~2,360 comparisons, predicted ~260 CPU-h, ~1 h +
     4-h-cap tail. Pull loop = launch_c2k50.sh (cache/box_rev/launch_c2k50.log -> pullC/). AFTER: take the box
     diffs.jsonl, 29 eval (K = 50 default) -> results/critic2_rerank.json; fill Section 3.8 TBDs (c2_cpu_median,
     c2_cpu_max, c2_capped); destroy 55845149.
     22:05 R4c DONE (7e61008): results/model_gap_realism.json. Measured: std macro top-1 0.024 [0, 0.057] vs
     original 0.016 [0, 0.041]; sc 0 vs 0. Own-pattern nuisances: own_all_dw std 0.231 vs 0.053, sc 0.294 vs 0;
     own_clean sc 0.317 vs 0.150; own_ka2 std 0.109 vs 0.178 (worse). Reading: the retrain fixes the simulated
     nuisances it saw (B damping most) and leaves measured patterns at ~0 -> the gap is not the modelled ones.
     10-14 04:05 R3 DONE: K = 50 pulled (out.tar md5 845dee12... OK), 2,951 rows = all 2,950 planned pairs + one
     duplicate (pattern_730/GOGZOU: the K = 50 run re-queued the K = 10 straggler still in flight; both hit 4 h).
     Run end 08:01Z (exactly the cap). 29 eval now writes cpu_s_median/max/total + the failed list. results/
     critic2_rerank.json (K = 50, 59 patterns, recall 0.61): critic2 sc macro top-1 0.500 [0.317, 0.667] vs ln q
     0.455 [0.276, 0.633], paired +0.045 [-0.110, +0.195]; top-10 0.587 vs 0.611, paired -0.025 [-0.146, +0.092];
     per pattern 26 vs 25 of 59. Inside the 36 shortlists holding the answer: both top-1 22; critic2 gains 4
     (77, 487, 655, 1008: ln q ranks 6, 1, 8, 2), loses 3 (247 -> 1; 594 -> 21; 1006 -> 49, where ln q's correct
     first choice gets the WORST DIFF of the 50, 0.261 vs best 0.047); neither 7, of which 297/511/816 jump from ln q
     43/27/39 to 2/1/2 and 16/82/96/880 drop (6->27, 6->22, 22->46, 3->15). K = 10 on all 59 (cache/box_rev/
     critic2_rerank_k10_full.json): 0.533, paired +0.079 [-0.055, +0.212] -> the longer list costs the re-rank a
     little, as Section 3.8 says. Cost: 2,949 completed, median 2.2 CPU-min, max 2.96 CPU-h, 400 CPU-h total
     (forecast 260); 1 capped (no answer in that shortlist). Section 3.8 TBDs filled. Box 55845149 self-stopped on
     PULLED, then DESTROYED; Vast credit left $6.84 (R3 box total ~$4.4). Previous local diffs ->
     cache/critic2/diffs_k10_preview.jsonl.
   - [x] CSLS ln q v2 (86def71): mkdir cache/real_bench_lnq2; copy hub_ctx.npz + ranks.npz; out_retry.tar's
         cache/real_bench_lnq2r/full -> cache/real_bench_lnq2/full (check its DONE file);
         LNQ_VERSION=2 python scripts/21_hubness.py eval --work cache/real_bench_lnq2 --emb_dir cache/real_bench
         --out results/real_bench_hub_lnq2.json --table results/real_bench_hub_lnq2_table.md
         CHECK: its none|*|lnq_bkg top-1 == rescore_r1 (same v2 scores, GPU vs CPU torch)
   - [x] R2 ablations: results/rescore_r2_no*.json; 22 computes paired diffs vs r2
   - [x] R4c (7e61008): 30 pemb + eval with --ckpt runs/xrdclip_realism/best.pt --emb_dir cache/real_bench_realism
         -> results/model_gap_realism.json; check train.log best epoch / clean val vs the original run
   - [x] box out.tar -> cache/critic2/diffs.jsonl; python scripts/29_critic2_rerank.py eval -> results/critic2_rerank.json
   - [x] B cod_out.tar -> cod/sticks; touch PULLED + COD_PULLED; destroy B
   - [x] R5 (250632d): 31 truth; 19's _embed_shard over shard_table (orig ckpt; the rev_gpu_box.sh snippet) with
         XRD_STICK_DIRS=cod/sticks into cache/real_bench_cod; XRD_STICK_DIRS=cod/sticks
         28 score + eval --tag main --work cache/real_bench_cod --truth <data>/cod/opxrd_truth_cod.json --base ""
         --out results/rescore_main_cod.json
C. Paper: paper_numbers.json -> manuscript/ESI/figs -> verify_claims -> internal referee pass -> build -> check-in.
   Started 10-14 after R3. Methods (Sections 2.6, 3.2, 3.4, 3.7, 3.8) already describe R1-R5; Abstract, Results,
   Table 1, Discussion, Conclusions, Data availability are still v1. real_bench_hub_lnq2.json differs from v1 only
   in lnq_bkg keys and matches rescore_r1 (checked 10-14) -> it becomes 22's main source (ln q v2 primary, v1 in ESI).
   - [x] C1 22: hub source -> real_bench_hub_lnq2 (+ cache/real_bench_lnq2/hub_ranks.npz); lnq_v1 block kept for
         the ESI; G0 from strong_baselines_v2 (v1 sweep kept); 29 eval saves per-query critic2 ranks -> 22 adds the
         cell-mismatch split for ln q v2 AND critic2 (R3 prediction: critic2 lifts the > 1 % group); R2 ablations;
         regenerate results/paper_numbers.json, every existing assertion passing
   - [x] C2 figs (23; + new Fig. 6 effect ladder): Fig. 2 from sweep v2, Figs. 3/5 with ln q v2 (+ critic2 point in Fig. 3a)
   - [x] C3 manuscript (draft v2, abstract 250 words, new Table 2): Abstract; Section 1 contributions; 2.6 replica counts vs rescore_main_cod (190 queries,
         9 sc compounds); 4.1 sweep v2; 4.2 Table 1 ln q v2 + variable-cell re-rank + cell-mismatch split; 4.3; 4.4
         CSLS with ln q v2; 4.5 ladder, pattern-to-pattern, realism retrain; new 4.6 library realism (R2 + ablations)
         and 4.7 open replica (R5); Discussion (variable-cell now tested; limitations); Conclusions; Data availability
   - [x] C4 ESI (25; S9-S13): ln q v1 vs v2, R2 ablations, critic2 per-pattern + cost, ladder table, COD replica table
   - [x] C5 verify_claims (169/169; coverage scan: uncovered numbers are design constants/labels): templates for every new number; 100 % verified
   - [x] C6 internal referee pass (powder + ML subagents), fixes applied (10-14; verify 186/186, build OK). Reports in 10-14 session (summarised
         here). Both: main result stands; the new single-crystal and learned-side readings are overstated.
         Analyses (local, no cost; claims verified before use):
         - [x] cell metric was setting-dependent (sorted conventional lengths; DEHKUA 21 % -> 3.8 % reduced) and its
               "own entry" could be an sc anchor (BAPLOT 0 %): now primitive Niggli cells, own = powder-study anchor
               with lowest RMSD; dT for the SAME closest-cell entry (was closest-temperature). Groups 33/25 (n 58):
               ln q 22/2, critic2 21/4; CEHQEM (one of ln q's 2) is a supercell (V ratio 1.98). dT > 50 K for 15 of
               35 with both T, 9 of them in the cells-differ group -> "explains few" was wrong.
         - [x] pressure: powder referee says ~22 of 59 sc patterns are room-T compression series (MEHNIY, EPIVEF,
               HOJQII, LILRIJ, NAPHTA). Verify with CSD pressure fields; tag every sc pattern (pressure / dT > 50 K /
               setting / other); sc metrics with and without high-pressure queries (secondary, CIs)
         - [x] critic2: positive control vs own Rietveld entry (running); composition check of every CIF given to
               critic2 (ROHWIX01: disorder read at full occupancy?); peaks + fit residual per pattern (0.41 A fits
               13-31 peaks?); answer injection for the 23 missing answers (~1 h); ln q best-offset distribution
         - [x] ladder: retrain rungs with background removal everywhere (running; Kα2 "exception" may be an
               artefact); B-alone rung; disclose B/jitter rungs were added after the first ladder
         - [x] COD sc set composition (20/35 ZZZUWK vs a 1950 structure the CSD sets aside; 8 naphthalene HP)
         Text: soften "gap lies in effects we did not model" (one seed, clean-val checkpoint, rungs off the
         retrain's input distribution); report the full R4c prediction (median-rank part failed too); "robust"
         with multiplicity; data availability (model rows need the CSD checkpoint; versions); Kα2 assumed not
         estimated; LP/package is query-side; Section 3.4 only the zero offset uses the own entry; 34-47 % wording;
         K 10 -> 50 history in Section 3.8; critic2 build 1.4.155; Fig. 2 lacks WCC 0.5; S12 caption pointer.
         RESULTS (10-14): pressure series CONFIRMED by CSD/COD: 22 of 59 sc patterns, 5 compounds (scripts/34 ->
         results/sc_conditions.json; 20 by CSD pressure/same-publication rule + HOJQII01/02 by hand); of the 25 cells-
         differ patterns 16 pressure series, 7 dT > 50 K, 1 supercell (CEHQEM), 1 unexplained (TAQBIY). critic2:
         ROHWIX01 loss was a disorder-input bug (all components at full occupancy; 4+1 disordered inputs fixed to the
         major component; DIFF 0.261 -> 0.006) -> re-rank 0.53 [0.37, 0.70] vs ln q 0.45, paired +0.08 [-0.05, +0.21];
         positive control (own Rietveld entry vs wrong shortlist entries): rank 1 for 29/36 ambient but 1/22 pressure
         series (median 26); peak fits 11-49 peaks on pressure series vs 66-296; injection: 2 of 23 missing answers
         would rank first. Ambient split (37 patterns, 25 cpds): classical 0.35-0.60, re-rank 0.64 vs ln q 0.54
         (paired +0.10 [-0.04, +0.26]), model 0. Ladder: B alone is the model's largest single cost (0.27 -> 0.07),
         Pearson 1.00; background-everywhere rerun changes rungs by <= 0.05 (Kα2 retrain 0.13, not an artefact).
         Text: abstract (250 w), 2.3, 2.5 primary comparison, 3.2, 3.4, 3.7, 3.8, 4.2, 4.5, 4.6, 4.7, discussion,
         limitations, conclusions, data availability (versions), Fig. 2 (+WCC 0.5), Fig. 6 (+B rung), ESI S10/S11.
         DECISIONS FOR FRANK (not taken): (1) headline sc set: keep the pre-declared 59 (incl. high-pressure) as
         primary + non-HP as secondary, or reframe; (2) more compute: critic2 refits of the DAC patterns,
         zero-offset sensitivity, bin-averaged Cu-grid conversion for all scores (np.interp aliasing of fine
         synchrotron steps, the R1 issue for every score; GPU rescore ~$0.5).
         FRANK 10-14 (AskUserQuestion): "Keep all 59" as headline; run ALL THREE: "Refit critic2 on DAC data,
         critic2 zero-offset test, Bin-averaged Cu grid".
   - C8 Frank's three extra checks. Designs + predictions fixed here BEFORE any number (10-14):
     - [ ] X1 bin-averaged Cu grid (sensitivity, pre-declared rule). Each measured pattern goes onto the Cu
           grid by area-preserving bin averaging (the mean of the native pattern over each Cu grid point's
           d-range, as R1 did for ln q) instead of np.interp; behind a flag, interp stays the default. Score
           every classical score + the model with 28 (new tag, ln q v2, no R2 switches) locally; paired macro
           top-1 against the main benchmark. RULE: if any classical score's compound-weighted top-1 moves by
           more than 0.05 (point estimate) in either answer set, propose to Frank that it become primary (GPU
           rescore of hubness/CSLS/library curves, ~$0.5); otherwise it is a sensitivity check (Section 4.6 +
           ESI). PREDICTION: every change within +-0.02; Cu lab patterns barely move.
     - [ ] X2 critic2 zero offset (sensitivity). Per sc pattern an ANSWER-BLIND native zero offset: delta =
           argmax over native shifts (+-0.5 deg x lambda/lambda_Cu, step 0.02 deg x lambda/lambda_Cu) of the
           cosine between the background-subtracted measured pattern, shifted by -delta and put on the Cu grid,
           and the clean rendering of its ln q top-1 entry (never the answer). fit.peaks positions shifted by
           -delta (equivalent to refitting a shifted pattern). Rerun only the 36 shortlists that hold the
           answer (1,800 comparisons; a miss stays a miss) on a rented CPU box (manual mode, CSD CIF upload,
           pull armed at launch, self-stop + watchdog). Report re-rank top-1/top-10 with paired diffs against
           the unshifted re-rank, and the delta distribution. PREDICTION: lab Cu median |delta| <= 0.03 deg;
           re-rank top-1 moves within +-0.05, unresolved.
     - [ ] X3 critic2 refits of the 22 pressure-series patterns. Tune XRPD BACKGROUND/FIT settings using ONLY
           the own-entry positive control as the criterion (never the sc answers); fix one setting for all 22;
           refit; rerun their answer-holding shortlists, the control and the injection (~300 comparisons,
           local). Report the re-rank with refitted pressure-series patterns as a sensitivity. PREDICTION: the
           control improves (own entry first for a majority), the re-rank's sc top-1 rises by <= 0.03 because
           most pressure-series answers are outside the shortlist (injection: 1 of 17 first with old fits).
   - [~] C7 build.sh -> docx (builds 10-14, after the referee fixes); plan Section 13c-f readouts; check in with Frank
After C (not R1-R5): R6 decisions, Zenodo deposit (code + open data; DD's data reviewer needs a DOI), cover
letter, TOC graphic (at revision), Frank submits.

Open decisions for Frank (flagged in the draft, not blocking):
- corresponding e-mail cai485@purdue.edu? (Purdue's RSC OA agreement is applied at acceptance and ends 2026-12-31; otherwise there is an APC)
- release: public code + Zenodo (refcode lists, opXRD pattern IDs, per-query ranks); the CSD-trained checkpoint is NOT released
- AI-use statement wording (his own, as in mlip-dynstab); title; ChemRxiv preprint or not

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
