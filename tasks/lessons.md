# Lessons (xrd-clip)

## 2026-10-12 — a repeated query compound looks like a hub
- **What happened:** plan Section 12a called GUSRIV a hub because it was the wrong top-1 for 20 of the 52 cos +
  zero-scan misses. All 20 were patterns of one compound, ZZZUWK, which has 20 patterns in the 188-pattern test
  set. The CSLS "fix" looked like +0.15 top-1 per pattern but is +0.04 (CI includes 0) per compound.
- **Rule:** before calling a library entry a hub, or attributing an effect to it, count the DISTINCT query
  compounds it wins. Report compound-weighted (macro) metrics next to per-pattern ones whenever the test set
  repeats compounds. Use a compound-level bootstrap for paired differences.
- **Rule:** fix the primary analysis in a commit before computing it (here 2a2ad8a). Sensitivity variants are
  reported, never selected.

## 2026-10-12 — a mechanism written before its cheapest test was run
- **What happened:** draft v0 explained two results with mechanisms I had not tested. (1) "The model fails on
  discrimination at library scale": one script (scripts/26) showed the opposite, clean simulated patterns of the same
  targets reach top-1 0.21 / median rank 5, so the failure is the measured-to-simulated step. (2) "ZZZUWK shows
  texture along c": the referee pass showed the intensity ratios are not monotonic in the angle to c, which March-
  Dollase texture requires; the own entries are disordered and simulated through a fallback. Both were caught by an
  internal referee pass, not by me. Also: "35 conditions" counted a duplicate (clean == zero offset 0).
- **Rule:** before writing why something fails, write the prediction the mechanism makes and run the cheapest test
  of it (here: the same model on clean simulated versions of the same queries; intensity ratio vs the angle to the
  texture axis). If it is not run, the sentence says "hypothesis".
- **Rule:** count distinct conditions by comparing their results, not by counting names.
- **Rule:** run the adversarial referee pass BEFORE showing a draft; it found ~10 must-fix items here.

## 2026-10-13 — box results not pulled before the self-stop (manual mode)
- **What happened:** both revision boxes finished, waited their 2 h for a pull, and stopped themselves (correctly:
  no idle billing). The pull never came because in manual mode every tool call waits for Frank's approval and he
  was away ~11 h. Their GPUs were re-rented, restarts failed, and the outputs had to come off the stopped disks
  through a relay box. Credit went $7.30 -> $1.66, ~$1.8 of it the unused pull windows.
- **Rule:** in manual mode, the command that launches a box queue also starts the local pull loop (wait for
  RUN_DONE, pull, md5, touch PULLED), so nothing after launch needs an approval.
- **Rule:** budget GPU memory per concurrent job on a box (21 score OOMed beside a 17 GB train on 24 GB).

## 2026-10-13 — local jobs reaped; a copied worker lost its pool setting
- **What happened:** four local jobs ran at once (R4c pemb, R5 embed, critic2 fits + run) and Claude Code's memory
  reaper killed all of them. 33_cod_embed called 19's _embed_shard through a plain ProcessPoolExecutor, so one
  worker embedded every shard and grew ~1.8 GB per shard; 19 runs the same function with maxtasksperchild=1.
- **Rule:** when a new script reuses a worker function, copy the pool settings its original caller uses (and read
  the function's docstring for them).
- **Rule:** on the laptop, run one heavy torch job at a time; chain them in one script, CPU-light jobs alongside.

## 2026-10-13 — a two-point cost extrapolation cut the critic2 shortlist for nothing
- **What happened:** two finished comparisons (16 s, 230 s) gave cost ~V^2.15 in cell volume and priced K = 50 at
  ~3,100 CPU-h, so K was cut to 10. The K = 10 run measured 76 CPU-h for 589 comparisons (~6x below the estimate;
  refit ~V^1.8 on 589 points), and K went back to 50 on the same box.
- **Rule:** before cutting a declared protocol on cost, time a stratified sample (e.g. 20 jobs across the size
  range) on the target hardware; never extrapolate an exponent from two points, and say how many points it rests on.
