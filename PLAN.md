# XRD-CLIP — Execution Plan & Status

Living status doc. `SPEC.md` is the technical scope; this is what is built, what the numbers say,
and what happens next. Updated 2026-07-14.

---

## Progress log

**2026-07-14** — Pipeline built + validated on pilot; week-0 gate PASSED; overfit plumbing check
top-1=1.000; 315,060 structures extracted. (Details throughout.)

**2026-07-15** — Day 2. First real training attempt + three fixes:
- **OOM bug found & fixed:** the structure encoder was being fed the *full unit cell* (all Z
  symmetry copies, up to 1000+ atoms), exploding the Transformer's O(atoms²) attention to ~17 GB
  and thrashing swap on the 31 GB box. Now feeds the **asymmetric unit** (median 42 atoms, capped
  256). Correct *and* compact. Fixes the pending full run too.
- **Repo bug found & fixed:** `.gitignore`'s unanchored `data/` rule was silently excluding the
  source package `src/xrdclip/data/` — six core modules (extract/simulate/augment/splits/dedup/
  dataset) were never committed. Anchored the ignore rules to root and recovered the source.
- **CPU training reality:** ~104 s/epoch for the 5.7M-param model on this CPU. A pilot run
  (884 train / 111 val scaffold split, 50 epochs) is in flight to answer the week-1 gate at small
  scale; the paper-scale run needs the Vast 5090. opXRD confirmed as the sim-to-real set.

---

## 0. Where this stands right now

**Week-0 gate: PASSED.** The whole project was conditional on one question — is there real headroom
between classical search-match and what a learned model could do? Answered empirically on day 1, on
a 1,098-structure pilot, and the answer is an emphatic yes. Details in §3.

The entire shared data pipeline is built, tested, and validated on a pilot. The full dataset
(**315,060 CSD molecular crystals**) is extracted and ready to simulate. What remains is mechanical:
run the full simulation, train on GPU, run the eval battery, write.

| Component | State |
|---|---|
| Stage 1 — CSD extraction (`csd_extract.py`) | **DONE**, full run complete: 315,060 structures |
| Stage 2 — pattern simulation (`simulate.py`) | **DONE + validated** on pilot; full run pending (~5 GPU-free hrs) |
| Physics augmentation (`augment.py`) | **DONE**, 15 physics tests green |
| Scaffold + random splits (`splits.py`) | **DONE**, leakage-verified |
| PXRD-degeneracy dedup (`dedup.py`) | **DONE** — found & fixed a hidden metric ceiling (§3.2) |
| Torch dataset + collate (`dataset.py`) | **DONE** |
| Classical baselines (`baselines.py`) | **DONE + validated**, gate passed |
| Dual encoders (`encoders.py`) | **DONE**, 6.1M params, forward verified on CPU |
| Contrastive model + loss (`clip.py`) | **DONE** |
| Training loop (`train.py`) | **DONE**, runs end-to-end on CPU; GPU run pending |
| Full-library retrieval eval | TODO (week 1) |
| Robustness-curve figure harness | TODO (week 1–2) |
| opXRD sim-to-real test | TODO (week 2, §5) |
| Paper | TODO (week 2) |

---

## 1. The pipeline, and why it is split the way it is

```
  [CCDC miniconda, licensed]          [main python / GPU box, NO licence]
  ┌─────────────────────┐            ┌──────────────────────────────────────────┐
  │ csd_extract.py      │  shards    │ simulate.py → sticks → augment.py (on the │
  │ 1.45M → 315k        │──jsonl.gz─▶│ fly) → dataset.py → train.py → eval        │
  └─────────────────────┘            └──────────────────────────────────────────┘
        stage 1                                    stage 2+
```

The hard separation is deliberate and load-bearing: **the GPU box has no CSD licence**, so nothing
downstream may import `ccdc`. Stage 1 emits everything later stages need (CIF-derived structure +
SMILES + metadata) into gitignored shards; only `refcodes/*.txt` is publishable.

**The stick/render split** (simulate once, degrade on the fly) is what makes strong augmentation
affordable, and strong augmentation is the entire source of the model's robustness. Never cache
rendered patterns.

---

## 2. Dataset facts (measured, not estimated)

- CSD scanned: 1,451,367 entries → **315,060 kept (21.7%)** after filters.
- Rejections dominated by: not-organic (785k), disorder (128k), R-factor (57k), Z′ (55k), element (47k).
- Extraction wall time: **44.7 min**, 6 workers, local machine.
- Simulation: **~0.43 s/structure** (pymatgen structure factors) → **~5 hours** full set on 8 procs,
  a one-time overnight cost. 98.2% simulate successfully.
- Pilot: 1,126 extracted → 1,106 simulated → 1,098 PXRD-distinct after dedup.

---

## 3. Week-0 gate results (the go/no-go, already run)

Ran `scripts/04_baseline_gate.py` on the 1,098-structure pilot library. Top-1 retrieval accuracy of
each classical baseline as a single degradation is dialed up from clean:

| Degradation | Correlation top-1 | Peak-list top-1 |
|---|---|---|
| **clean** | **1.000** | 0.987 |
| zero-offset 0.05° | 0.817 | 0.972 |
| zero-offset 0.10° | **0.003** | 0.987 |
| zero-offset 0.20° | 0.000 | 0.000 |
| strain 0.005 | 0.084 | 0.954 |
| strain 0.010 | 0.001 | 0.130 |
| broadening 20 nm | 0.995 | 0.878 |
| **broadening 10 nm** | **0.972** | **0.311** |
| preferred-orientation r=0.5 | 0.760 | 0.837 |
| noise 2e3 counts | 1.000 | 0.973 |

### 3.1 What this says — and it is a *better* story than the SPEC predicted

The two classical methods fail on **complementary axes**:

- **Correlation** dies under any peak shift (a 0.1° offset takes it from 1.000 to 0.003) because it
  compares intensities bin-by-bin and a shifted peak lands in the wrong bin. It is, however, robust
  to broadening and noise.
- **Peak-list matching** is the mirror image: it shrugs off shift (its tolerance window absorbs
  offsets) but collapses under 10 nm broadening (0.311), because heavy broadening destroys the peak
  *extraction* it depends on.

Real experimental patterns carry **all of these effects at once**. A pattern with both a modest
shift *and* heavy broadening defeats correlation (shift) *and* peak-list (broadening) simultaneously.
That is precisely the regime where a model trained to be jointly invariant to every effect should
win — and it sharpens the paper's thesis from the generic "learned beats classical" to the specific
and defensible **"classical methods are each robust to some distortions and fragile to others; only
a learned joint embedding is robust to the combination that real data actually presents."**

The clean-pattern row also confirms the harness is sane (correlation = 1.000), which is the other
thing the gate exists to check.

### 3.2 A hidden ceiling, found and removed

Clean self-retrieval initially scored 0.977, not 1.000. Every "failure" was a CSD **redetermination**
of the same compound (ABAMEX01 → ABAMEX). Two determinations of one structure give near-identical
patterns, so no PXRD method can separate them — scoring them as errors imposes a spurious ceiling
below 1.0 on every number in the paper, drifting with however many redeterminations land in the
library. `dedup.py` collapses PXRD-degeneracy classes (correlation ≥ 0.99 among entries sharing a
CSD stem) while deliberately keeping **polymorphs** apart (same compound, genuinely different
patterns). This had to be caught before training; it would have quietly corrupted the headline.

---

## 4. Week-by-week (updated against actual progress)

### Week 0 — Jul 14–20: data + baselines — **effectively DONE on day 1**
- [x] CSD extraction, full: 315,060 structures
- [x] Simulation validated on pilot; bug fixed (4-index Miller–Bravais → 3-index)
- [x] Physics augmentation + 15 tests
- [x] Scaffold splits with leakage verification
- [x] PXRD-degeneracy dedup
- [x] Both classical baselines + **gate PASSED**
- [ ] Kick off full-set simulation (the ~5-hour run) — *do this next, it blocks training*
- [ ] Build scaffold splits on the full set

### Week 1 — Jul 21–27: model + first real results
- [x] Encoders, contrastive model, training loop (built early), 6.1M params
- [x] **Overfit-tiny-subset plumbing check PASSED**: 48 structures → train top-1 = 1.000 at
      epoch 48 (loss 2.78 → 0.64). Architecture + loss + data plumbing all correct.
- [x] Full-library retrieval eval (`retrieve.py`) runs end to end; untrained model retrieves at
      ~chance (0.015 vs 1/200), confirming the harness has no leak.
- [ ] **Full training run on the Vast 5090** (B=512, ~50 epochs, est. ≲3 GPU-hr)
- [ ] Full-library retrieval eval (build reference embeddings, top-k on scaffold test)
- [ ] Robustness sweep harness → the money figure
- [ ] Hard-negative mining if clean numbers are healthy
- [ ] **Gate (Jul 27): does XRD-CLIP beat both baselines under the combined-degradation regime?**

### Week 2 — Jul 28–Aug 3: experiments, figures, write, submit
- [ ] Distractor-inflated library; random-vs-scaffold gap
- [ ] Mixture (two-phase) experiment
- [ ] Ablations: untrained encoder, no-augmentation, no-lattice-in-structure-encoder
- [ ] **opXRD sim-to-real test** (§5) — the credibility multiplier
- [ ] Figures (schematic, robustness curves, mixtures, ablations)
- [ ] Write 5-page IEEEtran paper; refcode list → SI
- [ ] **Submit**

---

## 5. opXRD — the sim-to-real bridge (found; not yet integrated)

The central weakness of a sim-only paper is the sim-to-real gap. **opXRD** (open experimental PXRD
database, ~92k real patterns, Zenodo, `pip install opxrd`) is the antidote. See `docs/DATA_SOURCES.md`.

**The honest caveat:** opXRD skews inorganic/high-throughput-materials, while our training set (CSD)
is molecular. Testing molecular-trained encoders on inorganic real patterns confounds two shifts
(sim→real *and* molecular→inorganic). The clean experiment uses opXRD's **labeled organic subset**
against a CSD-derived library. Even 20–50 such patterns retrieved correctly would elevate the paper
from "promising on simulation" to "works on real data," which is the difference between a poster and
a talk. Scope this in week 2; do not let it block the simulation-based core.

---

## 6. What could still go wrong (live risks)

- **Full simulation reveals a failure mode the pilot didn't** (exotic cells, huge Z′=2 structures).
  Mitigation: the 98.2% pilot success rate is reassuring, and `simulate.py` already logs a warning
  if failure exceeds 15%.
- **The model overfits scaffolds anyway.** Mitigation: scaffold split + report the random-vs-scaffold
  gap so the effect is measured, not hidden.
- **opXRD organic subset is too small or too dissimilar.** Mitigation: it is a bonus, not the core;
  the simulation robustness result stands on its own.
- **Aug 3 deadline slips earlier.** Mitigation: the pipeline is done; a poster or MoML redirect loses
  no work (SPEC §8).

---

## 7. Commands (reproducibility)

```bash
# stage 1 — under the CCDC miniconda (licensed), one time
<CCDC>/ccdc-software/csd-python-api/miniconda/python.exe -m xrdclip.data.csd_extract --out cache/csd --workers 6

# stage 2 — main python, no licence (the ~5-hour run)
python -m xrdclip.data.simulate --in cache/csd --out cache/sticks --workers 8

# stage 3 — splits
python -m xrdclip.data.splits --sticks cache/sticks --out refcodes

# gate (re-run any time)
python scripts/04_baseline_gate.py --sticks cache/sticks --n 5000

# train (Vast 5090)
python -m xrdclip.train --sticks cache/sticks --batch_size 512 --epochs 50 --workers 12

# sanity: overfit a tiny set (top-1 must approach 1.0)
python -m xrdclip.train --sticks cache/sticks --overfit 48 --epochs 60 --batch_size 48
```
