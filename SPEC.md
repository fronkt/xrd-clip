# XRD-CLIP — Technical Specification

**Contrastive PXRD ↔ structure retrieval for phase identification**

| | |
|---|---|
| **Venue** | IEEE URTC 2026, MIT Stata Center, Oct 9–11 2026 |
| **Format** | Technical paper, **5 pages hard limit** (submission *and* publication) |
| **Archival** | **Yes** — accepted + presented papers go to IEEE Xplore; copyright transfers to IEEE |
| **Deadline** | Expected **~Aug 3, 2026** (2025 cycle date; 2026 dates unposted as of Jul 14 — **recheck weekly**) |
| **Notification** | ~Aug 29, 2026 |
| **Written** | 2026-07-14 |

> **Archival gate.** Because URTC papers land in Xplore with copyright transfer, nothing that is
> bound for a journal may be submitted as a URTC *paper*. XRD-CLIP is designed from the ground up
> to be a standalone, non-journal-bound contribution so it can be published here cleanly. Keep it
> that way: do not fold XRD-CLIP results into the `pxrd-diff` journal manuscript.

---

## 1. Problem and motivation

Powder X-ray diffraction is the workhorse for identifying which crystalline phase you actually
made. The standard workflow is **search-match**: bin the measured pattern, compare it against a
reference database by correlation or peak-list overlap, and rank candidates. This works, and it
is brittle in exactly the ways real data is messy:

- **Peak shift** from thermal expansion, solid-solution composition, strain, or a mis-set zero offset.
- **Peak broadening** from small crystallite size and microstrain.
- **Preferred orientation**, which redistributes intensity and can make a correct match score worse than a wrong one.
- **Multi-phase mixtures**, where no single reference pattern matches the query.

Correlation-based matching treats the pattern as a raw vector, so every one of these physical
effects looks like "distance." A learned embedding can instead be trained to be *invariant* to
them — to map a degraded pattern and its clean parent structure to the same place.

**Claim.** A contrastively trained joint embedding of diffractograms and crystal structures
retrieves the correct phase more reliably than classical full-pattern search-match, and the
margin grows with the severity of physically realistic degradation.

That is a falsifiable, self-contained claim, testable in three weeks on one GPU. It is also
genuinely useful: retrieval is the operation practising experimentalists run every day.

### Novelty positioning

Prior work has applied CNNs to XRD for *classification* (predict space group / crystal system /
a fixed phase label) — a closed-set problem that cannot handle a library that grows. Retrieval
via a shared embedding is **open-set**: adding a new phase to the library means embedding one
more structure, with no retraining. The contribution is (i) the CLIP-style formulation for
diffraction, (ii) a physics-grounded augmentation suite as the source of invariance, and
(iii) an honest robustness benchmark against the classical baseline the field actually uses.

---

## 2. Method

### 2.1 Diffraction encoder `f_p`

Input: intensity on a fixed 2θ grid, **5–50° 2θ at 0.02° steps → 2251 points**, Cu-Kα1
(λ = 1.54056 Å). Normalise to unit max intensity, then to zero mean / unit variance per pattern.

Architecture: **1D ResNet**, ~6 residual blocks, channels 64→128→256, stride-2 downsampling,
kernel 7 (wide enough to see a peak plus its shoulders), GroupNorm, GELU. Global average pool →
projection MLP → embedding `z_p ∈ R^256`, L2-normalised.

*Why not a transformer first:* a diffractogram is a translation-sensitive 1D signal where local
peak shape carries the physics; a CNN with a decent receptive field is the right prior and trains
in minutes. Keep a patch-transformer variant as an ablation only if week 2 has slack.

### 2.2 Structure encoder `f_s`

Input: the periodic crystal graph. Nodes = atoms (element embedding + optionally Wyckoff-free
fractional coords are *excluded* — see below), edges = neighbours within a 5 Å cutoff with
Gaussian-expanded distances, under periodic boundary conditions.

Architecture: a **CGCNN/MEGNet-style message-passing GNN**, 4 layers, hidden 256, plus a global
readout that is *explicitly given the lattice*: concatenate a small MLP embedding of the reduced
cell parameters `(a, b, c, α, β, γ)` to the pooled node features before the projection head.
Project → `z_s ∈ R^256`, L2-normalised.

*Critical design note:* the diffraction pattern is determined by the lattice (peak **positions**)
and the atomic arrangement (peak **intensities**). The lattice term must reach the encoder or the
model cannot in principle place peaks. Feeding reduced-cell parameters directly is the cheap,
correct way to do this. Use a **Niggli-reduced** cell so that the same physical lattice always
gets the same parameters.

### 2.3 Contrastive objective

Standard symmetric InfoNCE (CLIP), with a learnable temperature τ initialised at 0.07 and clamped
to τ ≥ 0.01:

```
logits = (z_p @ z_s.T) / τ
L = 0.5 * ( CE(logits, arange(B)) + CE(logits.T, arange(B)) )
```

Positives are `(augmented pattern simulated from structure i, structure i)`. Negatives are the
other B−1 structures in the batch. **Batch size is the negative count and therefore matters a
lot** — target B ≥ 512 (embeddings are tiny; this fits easily on a 5090). If memory ever binds,
use a memory queue (MoCo-style) rather than shrinking B.

**Hard negatives (week 2, if time).** Random in-batch negatives are mostly trivially
distinguishable. Mine hard negatives by sampling, for each anchor, a structure whose *simulated
pattern* has high cosine similarity to the anchor's — i.e. genuine look-alike phases. This is the
single change most likely to move top-1, because it forces the model to learn what actually
separates confusable phases.

---

## 3. Data

### 3.1 Source

**Cambridge Structural Database** molecular crystals, via the CSD Python API (`ccdc`).

Filter to a clean, well-defined subset:
- organic, 3D coordinates determined, no disorder flag, no errors flag
- R-factor ≤ 0.075
- Z′ ≤ 2, no polymeric entries
- elements ⊆ {H, C, N, O, F, S, Cl, Br, I, P}
- exclude solvates only if they complicate the graph (decide in week 0; document either way)

Expected yield: **O(10⁵)** entries. Cap at ~100k for the 3-week build if the pull is slow.

> **Licence discipline.** No CIFs, no coordinate dumps in the repo, ever. Ship `refcodes/{train,val,test}.txt`,
> the simulation config, and derived arrays. Every result must be regenerable by a licence holder from
> the refcode list alone. Cite the CSD in the paper and put the full refcode list in the SI.

### 3.2 Pattern simulation

`pymatgen.analysis.diffraction.xrd.XRDCalculator` gives peak positions (2θ) and intensities.
Turn a stick pattern into a *measured-looking* one:

1. **Profile**: convolve each stick with a pseudo-Voigt, FWHM from the Caglioti relation
   `FWHM² = U tan²θ + V tanθ + W`, with (U, V, W) sampled per-pattern over an instrument-plausible range.
2. **Crystallite-size broadening**: additional Scherrer term, sampled domain size 20–500 nm.
3. **Background**: low-order Chebyshev polynomial, random coefficients, plus optional broad amorphous hump.
4. **Noise**: Poisson counting noise at a sampled peak-count level (controls effective SNR).
5. **Zero-offset / sample-displacement shift**: sampled ±0.1° 2θ.
6. **Preferred orientation**: March–Dollase along a randomly chosen low-index axis, ratio sampled in [0.7, 1.3].
7. **Lattice strain**: isotropic scaling of the cell by ±0.5 % *before* simulation → coherent peak shift.

Items 1–7 are the **augmentation suite**. Sample them fresh **on the fly, every epoch**, so the
model never sees the same degradation twice — this is where the invariance comes from and it is
the heart of the method. Precompute and cache only the *stick patterns* (positions + intensities),
which are the expensive part; profile/background/noise application is cheap and stays in the
dataloader.

### 3.3 Mixtures

Two-phase queries: `I_mix = w·I_A + (1−w)·I_B`, with w ∈ [0.3, 0.7], both normalised first. Used
**only at evaluation** in the headline experiment (train single-phase, test on mixtures = a
zero-shot generalisation claim). If time permits, an ablation that *trains* with mixtures and a
multi-positive loss is a natural extension — but do not let it eat the schedule.

### 3.4 Splits — the leakage trap

**Random splits will inflate every number and a reviewer will catch it.** The CSD is full of
near-duplicate entries: redeterminations of the same compound, polymorphs, close analogues. A
random split puts a redetermination of the test compound in train, and retrieval becomes trivial.

Split by **molecular scaffold** (Bemis–Murcko scaffold of the largest component, from the CSD
entry's SMILES via RDKit): all entries sharing a scaffold go to exactly one split. Target
80/10/10 by scaffold group.

Report **both** a random-split number and a scaffold-split number in the paper. The gap between
them *is a result* — it quantifies how much apparent performance in this area is memorisation.
That honesty is cheap to produce and is exactly what makes a benchmark paper credible.

---

## 4. Evaluation

### 4.1 Retrieval protocol

Build the reference library by embedding **all test-split structures** with `f_s` (this is the
"database"). Embed each query pattern with `f_p`. Rank by cosine similarity.

**Primary metric:** top-k retrieval accuracy, k ∈ {1, 5, 10}, plus **MRR**. Report library size
explicitly — accuracy is meaningless without it. Also report against a *distractor-inflated*
library (test structures + 50k train structures) to show it does not collapse as the library grows.

### 4.2 Baselines (non-negotiable — the paper dies without these)

1. **Full-pattern correlation search-match.** Bin query and every reference simulated pattern on
   the same grid, rank by Pearson correlation. This is the classical method; it is what we must beat.
2. **Peak-list / Hanawalt-style matching.** Extract top-N peak positions from the query, score
   references by matched-peak count within a 2θ tolerance. More shift-tolerant than (1) by
   construction — the honest strong baseline.
3. **Untrained-encoder ablation.** Random-init encoders, same retrieval protocol. Establishes that
   the contrastive training, not the architecture, is doing the work.

### 4.3 Robustness curves — the headline figure

Sweep **one degradation at a time**, holding the others at clean values, and plot top-1 accuracy
vs severity for XRD-CLIP and both baselines on the same axes:

| Axis | Sweep range |
|---|---|
| Zero-offset shift | 0 → ±0.3° 2θ |
| Crystallite-size broadening | 500 nm → 10 nm domain |
| Preferred orientation | March–Dollase ratio 1.0 → 0.5 |
| Noise | SNR high → low |
| Mixture fraction | pure → 50:50 two-phase |

**Expected shape (state this prediction now, so the result is a test and not a story told after the
fact):** correlation search-match degrades sharply with shift and orientation; XRD-CLIP degrades
gracefully because it saw those transforms in training. If XRD-CLIP does *not* win under shift,
the augmentation is not reaching the encoder and something is wrong — that is the debugging signal.

### 4.4 Mixture experiment

For a 50:50 two-phase query, does **top-2** retrieval recover *both* constituent phases? Report
"both-in-top-2" and "at-least-one-in-top-2" rates. Correlation search-match is known to be poor
here; a linear-superposition-friendly embedding may not be, and if it is, that is the most
practically interesting result in the paper.

---

## 5. Compute and environment

Single **RTX 5090** (Vast.ai). Both encoders are small; a full training run should be
**≲ 3 GPU-hours** at B = 512 for ~50 epochs over 100k structures. The expensive step is the
one-time CSD pull + stick-pattern precompute (CPU-bound, parallel, a few hours).

Environment notes carried over from prior projects:
- Install torch from the **cu128 index** for the 5090.
- **Cap dataloader workers** — Vast boxes advertise hundreds of cores and spawning that many
  workers will thrash. 8–16 is right.
- Check **Inet ≥ 200 Mbit/s** before moving the precomputed dataset.
- CSD pull runs **locally** (licence lives on the local machine); ship only the derived arrays to the GPU box.

---

## 6. Week-by-week plan

Today is **2026-07-14**. Assumed paper deadline **~Aug 3** → this is a **3-week sprint**. It is
tight, deliberately scoped to be survivable, and has an explicit bail-out.

### Week 0 — Jul 14–20: data pipeline and baselines
Everything downstream depends on this week; do not start modelling until it is done.
- [ ] **Confirm URTC 2026 dates and the paper-track eligibility question with the organizers** (do this on day 1 — the whole sprint is conditional on it)
- [ ] CSD pull with the §3.1 filters → refcode list + structures cached locally
- [ ] Stick-pattern precompute (positions + intensities) for all entries; cache as `.npz`
- [ ] Augmentation suite (§3.2) implemented as an on-the-fly dataloader transform, with unit tests: a known cell must produce peaks at the analytically correct 2θ
- [ ] Scaffold splits via RDKit Bemis–Murcko; verify no scaffold crosses splits
- [ ] **Both baselines implemented and scored** (correlation + peak-list). This is the target to beat, and knowing the number early tells you whether the paper is interesting.
- **Gate:** baselines produce a sane top-1 on clean patterns (expect high — clean simulated retrieval is easy) and a visibly degraded top-1 under shift. If not, the eval harness is broken, not the physics.

### Week 1 — Jul 21–27: model and first results
- [ ] Diffraction encoder + structure encoder + InfoNCE head
- [ ] Overfit-a-tiny-subset sanity check (100 structures → top-1 should hit ~100 %). If it cannot memorise 100, do not scale up — debug.
- [ ] Full training run; top-k on scaffold split; compare against baselines
- [ ] Robustness sweep harness (§4.3) automated end to end
- [ ] Hard-negative mining if the clean numbers look healthy
- **Gate (Jul 27, hard):** does XRD-CLIP beat correlation search-match on *at least* the shift and orientation sweeps? **If yes → write the paper. If no → stop and reroute** (see §8).

### Week 2 — Jul 28–Aug 3: experiments, figures, write, submit
- [ ] Distractor-inflated library experiment; random-vs-scaffold-split gap
- [ ] Mixture experiment (§4.4)
- [ ] Ablations: untrained encoder; no-augmentation training; no-lattice-in-structure-encoder
- [ ] Figures: (1) method schematic, (2) robustness curves — the money figure, (3) mixture result, (4) ablation table
- [ ] **Write the 5-page paper** (IEEE conference template, `\documentclass[conference]{IEEEtran}`)
- [ ] Full refcode list → SI; CSD citation; reproducibility statement
- [ ] **Submit by the deadline**

Paper outline (5 pp, IEEEtran two-column):
1. Intro + why search-match is brittle — 0.75 pp
2. Related work (XRD classification CNNs; search-match; contrastive/CLIP) — 0.5 pp
3. Method (encoders, InfoNCE, augmentation-as-physics) — 1.25 pp
4. Experiments (data, splits, baselines, robustness, mixtures, ablations) — 2 pp
5. Limitations + conclusion — 0.5 pp

---

## 7. Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| **Sim-to-real gap** — trained and tested only on simulated patterns | High | Be explicit and name it as the central limitation. If *any* real patterns are reachable (RRUFF, a Purdue instrument, published SI patterns), even a 20-pattern qualitative test elevates the paper enormously. **Try for this in week 2.** |
| **3-week crunch** | High | Week-0 gate + hard Jul 27 go/no-go. Baselines first means you always have *something*. |
| **URTC 2026 deadline lands earlier than assumed** | Medium | Recheck the site weekly. This is why the go/no-go is on Jul 27, not Aug 2. |
| **HS/undergrad paper-track eligibility** | Medium | Resolve with the organizers on day 1. Poster track is the fallback and requires no work to be wasted. |
| **Clean-pattern retrieval is trivially easy → no headroom** | Medium | Expected, and fine: the paper's claim lives in the *degraded* regime, not the clean one. Design the abstract around robustness, not clean top-1. |
| **CSD licence / redistribution** | Low but absolute | Refcodes only. Never commit a CIF. |

---

## 8. Reroute plan (if the Jul 27 gate fails)

Do not force it. Options, in order:
1. **URTC poster / lightning talk** — non-archival, later deadline (~Sep 5), no wasted work.
2. **Fold into MoML** as a short paper (non-archival, Sept 1) — retrieval over molecular crystals is
   squarely in MoML's scope, and 2–4 pages is a friendlier target than 5 polished ones.
3. **Fall back to `batterycv` for the URTC paper slot** — it is already finished; the paper is a
   pure write-up with zero experimental risk. *Keep this option warm regardless* — see §9.

---

## 9. Relationship to the other URTC submission

The URTC paper slot is not either/or. `batterycv` (fronkt/batterycv) is **already complete**
— closed detect→track→OCR→classify pipeline, .874 6-way / .899 chemistry accuracy, and a clean
negative result (OCR metadata adds zero sorting signal). Writing it up carries near-zero risk and
banks the IEEE Xplore publication.

**Recommended posture: submit both.** batterycv is the floor; XRD-CLIP is the upside. Multiple
submissions from one author are normal at URTC, and the two land in different tracks
(automation/sustainability vs. computation).

---

## 10. Related repos

- [`fronkt/pxrd-diff`](https://github.com/fronkt/pxrd-diff) — E(3)-equivariant conditional diffusion for PXRD → structure. Source of the PXRD simulation infrastructure. **Journal-bound — keep XRD-CLIP's claims disjoint.**
- [`fronkt/symmc-flow`](https://github.com/fronkt/symmc-flow) — flow-matching CSP; shares the CSD data plumbing.
- [`fronkt/pxrd-flow`](https://github.com/fronkt/pxrd-flow) — PXRD-conditioned flow matching (MoML 2026). The generative sibling of this work.
