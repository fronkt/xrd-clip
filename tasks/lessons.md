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
