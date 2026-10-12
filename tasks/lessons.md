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
