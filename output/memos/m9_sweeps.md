# Memo — module m9_sweeps: our controlled two-corpus experiment — DRAFT (dry run; FineWeb pending)

Module owner: m9_sweeps (Claude). Date: 2026-09-24. Entry point: `code/analysis/m9_sweeps/run.py`.

**Binding plan.** `paper/notes/m9_preanalysis_plan.md` (commit bd5c0ad, 2026-09-24 03:12:31 +03, the only commit touching the file). Deviations are listed in Section 6.

**Status.**
- This is a DRAFT written during a dry run.
- The FineWeb-Edu main grid is complete (44 endpoints). The FineWeb main grid, the high-M runs and all seed replicates are still training on the GPU queues.
- Every estimate in Sections 1, 4 and 5 is **PRELIMINARY** and **FineWeb-Edu only**.

**Order of work.**
1. Section 0 (power, 15:52–16:44) was computed from the design alone and written into this memo **before** `run.py` was first executed on `results.jsonl`.
2. The coverage check in Section 0.3 was also run before any estimation.
3. Real-data estimation started after Section 0 was saved.

---

## 0. Power calculation (plan §4) — computed before any estimation on the sweep results

**Code and outputs.**
- Code: `code/analysis/m9_sweeps/m9_power.py` (`m9_coverage.py` for §0.3).
- Output: `output/tables/m9_sweeps_power.csv` (every cell and statistic) and `data/processed/m9_sweeps/power_replications.csv`.
- The stage never reads `results.jsonl`.

**Design.** Taken from `code/sweep/run_grid.py` and the architecture formulas:
- 8 widths with depth d/64;
- the D ladder 25M–800M, with caps of 400M at widths 448/512 and 200M at width 640;
- the trimmed high-M runs: FineWeb-Edu (128,4) and (256,4) at 0.2/0.8/1.6/3.2B, FineWeb (128,4) only. The (256,4) cells at 0.2B and 0.8B duplicate main-grid cells (§7.1) and are dropped.

That gives 44 main-grid and 6 new high-M endpoints for FineWeb-Edu, and 44 and 4 for FineWeb.

**Truth.**
- A Chinchilla form with Besiroglu et al.'s exponents (α = 0.3478, β = 0.3658, σ* = 0.737), in non-embedding N.
- Levels anchored so that at the design's geometric-mean compute (C = 3.95×10¹⁵) the w = 1 path sits at M* = 20 (N* = 5.7M, D* = 115M) with L = 3.8 nats and E = 1.9.
- FineWeb-Edu = FineWeb with a data-augmenting shift (ln B lower by χ). The tilt is χ = 0.22 (DataDecide's magnitude) or 0.

**Sensitivity cells.** A κ-family truth (σ*_κ = 0.70, κ = 0.6), M* = 60, and a trunk share ρ ∈ {0, 0.9}.

**Noise.**
- ln L error with s.d. s ∈ {0.002, 0.005, 0.01}.
- A common within-trunk component carries a share ρ = 0.5 of the variance (one trunk per architecture). This uses m6's `het_cluster_noise` with het = 0.
- A second validation set has the same truth and noise correlated 0.8 with the first.

**Replications and estimators.** 150 replications per cell. The estimators are exactly those of `run.py`.

### 0.1 Expected standard errors (Monte Carlo s.d. of the estimator), FineWeb-Edu design, χ = 0.22, ρ = 0.5

| Statistic | s = 0.002 | s = 0.005 | s = 0.01 |
|---|---|---|---|
| σ*_κ, P | 0.0025 | 0.0067 | 0.0140 |
| σ*_κ, T | 0.0038 | 0.0075 | 0.0141 |
| σ* (κ = 1), P | 0.0025 | 0.0064 | 0.0137 |
| Model-free σ*, P (FLOP-optimal path) | 0.0086 | 0.0187 | 0.0280 |
| Model-free σ*, P6 (w = 1 path) | 0.0054 | 0.0126 | 0.0201 |
| Model-free σ*, T | 0.0075 | 0.0184 | 0.0337 |
| Tilt χ̂, P | 0.0072 | 0.0181 | 0.0382 |
| Tilt χ̂, T | 0.0086 | 0.0197 | 0.0399 |
| Q3 slope, P (main + high-M) | 0.030 | 0.062 | 0.107 |
| Q3 slope, T (main + high-M) | 0.018 | 0.044 | 0.088 |
| Q3 slope, P, main grid only | 0.030 | 0.073 | 0.120 |
| Q3 slope, T, main grid only | 0.069 | 0.152 | 0.666 |

The FineWeb design gives the same σ* and tilt precision. Its Q3 slope s.e. is 0.024 / 0.059 / 0.093 (P), with fewer high-M points.

**Sensitivity at s = 0.005.**
- Tilt s.e. (P): 0.016 (ρ = 0), 0.019 (ρ = 0.9), 0.022 (M* = 60), 0.028 (κ truth).
- σ*_κ s.e. (P): 0.0074, 0.0054, 0.0066 and 0.0097 in the same cells.

### 0.2 Power for a tilt of 0.22

**Normal approximation.** With the Monte Carlo s.e., the power to reject χ = 0 on one validation set is 1.00 at every noise level. The minimum detectable tilt at 80 percent power is:

| Convention | s = 0.002 | s = 0.005 | s = 0.01 |
|---|---|---|---|
| P | 0.020 | 0.051 | 0.107 |
| T | 0.024 | 0.055 | 0.112 |

**The plan's actual procedure.** This is the 95 percent basic interval from the wild cluster bootstrap by width (Webb weights; 199 draws per replication, 60 replications) combined with the plan's two-validation-set decision rule, in convention P.

| | s = 0.002 | s = 0.005 | s = 0.01 |
|---|---|---|---|
| Pr("factor-biased" \| χ = 0.22) | 1.00 | 1.00 | 1.00 |
| Pr("factor-biased" \| χ = 0) | **0.17** | **0.08** | **0.12** |
| Pr("neutral" \| χ = 0) | 0.57 | 0.50 | 0.58 |
| Rejection of χ = 0 on one set, when χ = 0 (nominal 0.05) | **0.28** | **0.23** | **0.22** |
| Coverage of the nominal 95% interval | 0.72–0.73 | 0.75–0.77 | 0.78 |
| Mean bootstrap s.e. / Monte Carlo s.d. | 0.67 | 0.59–0.71 | 0.60–0.76 |
| Mean interval half-width | 0.009 | 0.020–0.023 | 0.043–0.046 |

**Reading.**
1. **Precision is not the constraint for the tilt.** A tilt of DataDecide's size is 6–30 s.e. from zero at any plausible noise level. The "neutral" branch of the decision rule (half-width < 0.10) is reachable unless s is about 0.02 or more.
2. **The pre-specified inference is badly anti-conservative on this design.**
   - The wild cluster bootstrap uses the residuals of the unrestricted fit ("WCU") with only 8 width clusters. Its s.e. is 0.6–0.76 times the true sampling s.d. Its 95 percent interval covers 72–78 percent of the time.
   - A "factor-biased" verdict under a true χ = 0 occurs 8–17 percent of the time, instead of the roughly 1–2 percent the two-set rule would have at nominal size.
   - The reason is that the width-specific trunk shocks are partly absorbed by the fitted N-terms, so the residuals understate them (MacKinnon and Webb 2017).
   - This was found before any estimation. §0.3 and Deviation D8 set out the pre-declared remedy.
3. **σ\* is precise but convention-dependent.**
   - σ*_κ has s.e. 0.003–0.014. The model-free σ* has 0.009–0.034, or 0.005–0.020 on the w = 1 path.
   - With no noise, the parametric forms fitted in convention T give σ* = 0.655 and the model-free T estimate 0.694, against 0.737 in P. The truth is the same technology in every case: in T the N axis is a nonlinear transform of N_nonemb, so its Hicks σ is a different number.
   - A P-versus-T difference of about 0.04–0.08 is therefore expected **by construction**. It is not evidence about the technology (Pearce–Song / Proposition M at this scale).
4. **Q3 is the weakest statistic, and its null value is not zero.**
   - If the Chinchilla form is true in non-embedding units, the restricted-fit slope is centred near −0.007 in P without noise. With noise it drifts to −0.02, −0.05 and −0.06 at s = 0.002, 0.005 and 0.01, because the corner derivatives of the CV-bandwidth local quadratic are nonlinear in the noise.
   - In convention T the slope is centred at **+0.11** (with the high-M runs) or **−0.08 to −0.10** (main grid only), even without noise, because the Chinchilla form is misspecified in total-N units when it holds in non-embedding units.
   - Against the κ-family truth, where the Chinchilla form really is misspecified, the P slope is −0.13 without noise and −0.25 (s.e. 0.15) with s = 0.005. So a κ-type departure of Farseer's size is detectable only at s ≤ 0.005, and marginally.
   - **Implication for reading Q3.** Compare the estimated slope with these design-specific null values, not with zero. Use convention P (the plan's primary) for the sign.

### 0.3 Size and coverage of the plan's inference, and the pre-declared remedy

This check (`m9_coverage.py`) was run before any estimation. It uses 40 replications at s = 0.005 and ρ = 0.5 with the Chinchilla truth, and 199 Webb draws per replication. It compares:
- the plan's WCU;
- the same wild cluster bootstrap on CR2 (Bell–McCaffrey) leverage-adjusted cluster residuals;
- for the tilt test, the restricted wild cluster bootstrap ("WCR"), which imposes χ = 0 through the Hicks-neutral corpus-pair fit.

**Setup.** 40 replications at s = 0.005, with 199 draws each (99 for "cr2cv"). The Monte Carlo error on a coverage rate is about ±0.07. Coverage is measured against the noise-free value of each estimator on the design, which strips out smoothing bias. Output: `output/tables/m9_sweeps_power_coverage.csv`. Schemes:
- **wcu**: the plan's scheme — Webb wild cluster bootstrap on the raw residuals.
- **cr2**: the same draws on CR2 leverage-adjusted cluster residuals.
- **cr2cv**: cr2 with the leave-one-out CV bandwidth re-selected in every draw (model-free statistics only).

| Statistic (FineWeb-Edu, P) | MC s.d. | Coverage, wcu | Coverage, cr2 | Coverage, cr2cv | MC s.d. / boot s.e., best scheme |
|---|---|---|---|---|---|
| σ*_κ | 0.006 | 0.83 | **0.88** | — | 1.12 |
| Model-free σ*, FLOP path | 0.021 | 0.40 | 0.58 | **0.65** | 1.73 |
| Model-free σ*, w = 1 path | 0.013 | 0.45 | 0.65 | **0.70** | 1.57 |
| Q3 slope | 0.074 | 0.55 | **0.75** | 0.68 | 1.39–1.75 |
| Tilt χ̂ (χ = 0.22) | 0.021 | 0.68 | **0.78** | — | 1.41 |
| Tilt χ̂ (χ = 0) | 0.015 | 0.80 | **0.93** | — | 0.95 |

**Tilt tests of χ = 0 at 5 percent.**

| Test | Size (χ = 0) | Power (χ = 0.22) |
|---|---|---|
| Plan interval (wcu) | **0.20** | 1.00 |
| cr2 interval | 0.075 | 1.00 |
| Restricted wild bootstrap (WCR), raw residuals | 0.075 | 0.98 |
| **WCR, CR2 residuals** | **0.050** | **0.83** |

**Pre-declared remedy (Deviation D8).** Written here at about 17:05, before `run.py` touched `results.jsonl`.
1. **The plan's scheme stays in every table**, labelled "wild (plan)", exactly as specified.
2. **A size-corrected companion, "cr2", is reported beside it.**
   - Parametric statistics resample CR2-adjusted cluster residuals of their own fit.
   - Model-free statistics and the Q3 slope resample CR2-adjusted residuals around the local-quadratic surface, with the bandwidth re-selected in every draw.
   - For the tilt, the test of χ = 0 is the restricted wild cluster bootstrap on CR2 residuals of the Hicks-neutral fit. It has correct size here.
   - In the decision rule, "excludes zero" means WCR-CR2 p < 0.05. "Neutral" needs p ≥ 0.05 and a cr2 interval half-width below 0.10 on both validation sets.
3. **Robustness standard.** A conclusion is called robust only if it holds under the plan's scheme, the cr2 companion and (once seeds exist) the plan's secondary seed-covariance bootstrap. This is stricter than the plan's "both".
4. **Model-free σ\* and the Q3 slope are not decided on their intervals alone.** No scheme reaches nominal coverage for them on this 8×6 grid (best 0.65–0.75). A statement about them needs the cr2 interval to exclude the relevant value even after its half-width is multiplied by the design calibration factor (1.7 for model-free σ* on the FLOP path, 1.6 on the w = 1 path, 1.4 for the Q3 slope). Q3 slopes are read against the design's null values in §0.2, not against zero.

**Timeline** (`data/processed/m9_sweeps/estimation_start.txt`):

| Time (24 Sep 2026) | Step |
|---|---|
| 15:52–16:44 | Power (m9_power) |
| 16:49–17:03 | Coverage checks (m9_coverage) |
| ~17:05 | §0 and D8 written into this memo |
| 17:12 | `run.py` first executed on `results.jsonl` (111 records) |
| 17:19 and 17:24 | Two further passes (output fixes only), with 113 and 114 records as the FineWeb grid kept growing |

The numbers below come from the 17:24 pass. They are byte-identical to the 17:12 pass for every FineWeb-Edu statistic: the run is deterministic, and the FineWeb-Edu grid was already complete.

---

## 1. Headline findings — PRELIMINARY, FineWeb-Edu only (FineWeb, high-M runs and seeds pending)

> Everything in this section is a dry-run estimate on one corpus. It must not be quoted in the paper until the final run. The between-corpus questions (Q2, and the σ* equality test in Q1) have not been estimated.
>
> "Plan" intervals are the pre-specified wild cluster bootstrap. {Braces} are the pre-declared CR2 companion (§0.3). Output: bits per byte on FineWeb-Edu validation.

**P1. σ\* is well below one and close to 0.66 in non-embedding units. The model-free estimate sides with κ free, not κ = 1.**

| Estimator (FineWeb-Edu, main grid) | Convention P | Convention T |
|---|---|---|
| Model-free σ*, FLOP-optimal path (P) / w = 1 path (T) | **0.666** [0.659, 0.673] {0.652, 0.683} | **0.607** [0.602, 0.614] {0.596, 0.626} |
| Model-free σ*, w = 1 path (P6) | 0.665 [0.660, 0.671] {0.651, 0.676} | — |
| σ*_κ (κ family, Huber) | **0.662** [0.635, 0.686] {0.634, 0.687} | **0.605** [0.588, 0.622] {0.578, 0.625} |
| σ*_κ on the FLOP-optimal path | 0.668 [0.641, 0.693] | — |
| σ*, Chinchilla form (κ = 1) | 0.720 [0.691, 0.746] {0.683, 0.757} | 0.654 [0.628, 0.681] {0.618, 0.687} |
| κ̂ | 0.36 [0.19, 0.50] | 0.46 [0.28, 0.63] |
| Gaussian NLS: κ family / κ = 1 | 0.669 / 0.717 | 0.602 / 0.647 |

- **σ\* < 1.** This holds under every estimator, convention, sample and bandwidth: the largest upper bound is 0.76, even after the design calibration factor.
- **κ free vs κ = 1.** The model-free and κ-free estimates agree to within 0.005 in both conventions. The κ = 1 form is 0.05 higher, as in ra1 on the public IsoFLOP designs. So κ = 1 is rejected here as well (κ̂ is 0.19–0.48 across samples and conventions).
- **Convention.** The P-versus-T gap of about 0.06 is the size the power calculation shows arises **by construction** from re-expressing N (§0.2 reading 3). It is not a finding about the technology.
- **Comparison with public designs.**
  - The model-free σ* on the public IsoFLOP designs is 0.695 [0.673, 0.717] (ra1; Chinchilla total N, Marin FLOP-implied N).
  - The FineWeb-Edu value in P (0.666) is about 0.03 lower. In T (0.607) it is about 0.09 lower.
  - It is well above Porian et al.'s unannealed 0.51, which is consistent with annealed endpoints.
  - Treat this comparison as descriptive. The model-free intervals under-cover on this design (§0.3), and σ*(C) drifts upward with compute (0.65 → 0.68 in P, 0.56 → 0.64 in T, `m9_sweeps_q1_levels.csv`).
- **Allocation exponent.** a = β/(α+β) is 0.64 (κ = 1) or 0.61 (κ) in P, and 0.48–0.49 in T. The model-free compute-optimal M* is 12–19 on the FLOP path (P), 16–35 on the w = 1 path, and 8–12 in T.
- **Pending.** The equal-σ* test across corpora (plan Q1) needs FineWeb.

**P2. Q3 (extrapolating the wedge). The Chinchilla form fitted on M ≤ 100 understates the local wedge at every FineWeb-Edu endpoint with M > 100. Whether the gap grows with M is not established.**

Convention P, main grid only (high-M runs pending):
- The mean of ln(w_{M≤100}/w_local) over the 11 endpoints with M > 100 is −0.69 [−0.75, −0.60].
- By M bin it is −0.58 (M 100–316), −0.51 (316–1,000) and −1.25 (> 1,000; one endpoint). So the restricted fit's wedge is 40–70 percent below the local one.
- **The slope is −0.29**, plan interval [−0.40, −0.09], companion {−0.44, −0.005}. The design's null value is −0.02 to −0.04 at s = 0.005.
- **The slope is not decided.** Under the pre-declared rule (§0.3 item 4) the companion's half-width, multiplied by 1.4, gives about [−0.53, +0.08]. That interval contains the null and zero, so the slope's sign is **not decided**.
- **One endpoint drives it.** The slope rests on the two-layer width-128 model at 800M tokens (M = 2,031). There the loss barely moves between 400M and 800M (local ε_D = 0.003, kernel n_eff = 2.7).
  - Dropping widths 128 and 192 leaves three evaluation points (M 127–254), with slope −0.12 [−0.15, −0.09] and level −0.47.
  - Dropping D = 25M gives −0.26 [−0.29, −0.19].
- **Comparators.**
  - The κ family fitted on M ≤ 100 *overstates* the local wedge (mean +0.35; slope −0.003 [−0.13, 0.20]). The parametric forms bracket the local estimate.
  - The full-support Chinchilla fit also understates it (mean −0.29).
- **Convention T.** The slope is −0.76 [−1.08, −0.21] {−1.17, +0.54}, against a design null of −0.08 to −0.10 for the main grid only. It is not decided.

**P3. Q5 (learning rate).**
- **Within FineWeb-Edu the rule is near-optimal at the high-M corner.** At width 128 the rule's LR is the best of {0.5×, 1×, 2×} at every budget from 25M to 800M (excess ln L ≤ 0.001).
- **It is too high at the large-width corner.** At width 640, 0.5× the rule beats it by 0.005 (25M) and 0.002 (50M) in ln L (0.013 and 0.007 against the quadratic minimum).
- **By corpus** (D = 50M calibration). The optimal LR is the same at widths 320 and 512 (|Δ ln LR\*| ≤ 0.02). At width 128 FineWeb's optimum is higher: the best tested LR is the top of the grid, 8×10⁻³, and the quadratic gives 9.7×10⁻³ against 5.8×10⁻³, so Δ ln LR\* = −0.51. FineWeb's width-128 LR corners confirm it: 2× beats the rule at 25–200M.
- **Drift.** d ln LR\*/d ln D at width 128 is −0.12 (s.e. 0.05) for FineWeb-Edu and −0.23 (0.12) for FineWeb.
- **Bound on a.** The flexible-input bound gives a_obs − a = −0.006 to −0.011 (P) and −0.008 to −0.014 (T). The plan applies it only if the corner excess exceeds the seed s.d. That s.d. is pending; the κ-fit residual s.d. (0.006) is an upper bound and is about the size of the width-640 excess, so the bound should be taken as relevant.

**P4. Q6 (functional dependence with real losses).**
- On the full FineWeb-Edu grid the κ-family profile pins σ*_κ to [0.65, 0.69] in P and [0.59, 0.61] in T.
- On the on-path subsample (one endpoint per doubling of compute, n = 9) the 95 percent set is [0.35, 0.95] in P and [0.23, 0.87] in T. That covers 65–69 percent of the grid over (0.05, 0.99).
- With two per bin (n = 17–18) it is [0.53, 0.71] in P and [0.41, 0.93] in T.
- The normalized condition number rises from 6.4×10³ to 3.4×10⁴.
- The ACF-type non-identification thus shows up in real losses: path data alone barely restrict σ*.

**P5. Cross-evaluation (descriptive; 15 matched main-grid cells, widths 128/192/256).**
- On FineWeb-Edu validation, FineWeb-Edu models have 4.0 percent lower loss (range 2.8–6.5). On FineWeb validation, FineWeb models have 3.6 percent lower loss (0.7–4.4).
- Each corpus wins on its own validation set. By the plan's classification the **level** difference is therefore "distribution match", not "quality", so far.
- On WikiText-103 the matched cells are only learning-rate-corner runs at widths 128 and 640. The median difference is −0.2 percent (interquartile range −0.6 to +0.6). The exceptions are three width-128 cells (+4 to +6 percent) where the two-layer models' late loss drop happened at different budgets in the two corpora (§7.3).
- The tilt, which is the object of Q2, is not yet estimable.

**Q4 (noise).** Pending seeds. The residual s.d. of the fitted technology on FineWeb-Edu is 0.0059 for the κ family (MAD-based 0.0023) and 0.0102 for the Chinchilla form (P). Seed noise is therefore at most about 0.006, so the power cell s = 0.005 is the relevant one.

---

## 2. Methods (as implemented; plan section in brackets)

**Output [§2].** Loss in bits per byte, bpb = (nats/token) / (ln 2 × bytes/token of the evaluation tokens). The evaluation tokens are the ones `train_sweep.evaluate` actually scores: the targets of the first 4,096 windows of 257 tokens (1,048,576 tokens; 654,848 on WikiText). Their bytes/token come from the tokenizer's byte lengths (`<eot>` = 0 bytes). The same map reproduces `meta.json`'s full-file values exactly: edu 3.88567, web 3.74264, wiki 3.71306. For the scored subsets the values are edu 3.88888, web 3.71255 and wiki 3.71305. The difference changes web-val bpb *levels* by 0.8 percent (deviation D3). It cancels in every exponent, σ*, tilt and wedge, because it multiplies every loss on one validation set by the same constant. Technology estimates (Q1, Q3, Q5, Q6) use each corpus's own validation set. The contrast (Q2) uses each validation set separately.

**Parameter conventions [§2].**
- **P**: non-embedding N (`N_nonemb`), with compute measured as actual training FLOPs (`C = fpt(N)·D`, which includes the unembedding matmul and attention). FLOPs per token relative to 6N_nonemb run from 3.83 at width 128 to 1.14 at width 640. The cost elasticity of non-embedding parameters along the width ladder is η(N) = d ln fpt/d ln N_nonemb, from 0.52 at width 128 to 0.93 at width 640. So under P the compute-optimal allocation satisfies **w = ε_N/ε_D = η(N), not 1**.
- **P6**: the same N with the textbook cost 6N_nonemb·D (path w = 1). This is the object that the closed forms 2/(2+α+β) and 2/(2+a₁+b₁) describe. It is reported for comparability.
- **T**: total N including the tied embedding, with C = 6N_total·D. Here FLOPs/(6N_total) is 1.03–1.05, so T is also nearly FLOP-exact.

M = D/N in each convention.

**Samples [§2].** The main sample is the tag-`main` endpoints (seed 0) of each corpus. Robustness samples drop D = 25M (`no25`) or widths 128 and 192 (`floor4`). For Q3, the full support adds the `hiM` endpoints that are not duplicates of main-grid cells (see §7, item 1).

**Model-free estimators [§2(i)].**
- **Local surface.** A Gaussian-kernel local quadratic of ln L in (ln N, ln D) (`m2_est.local_quad`, as in ra1). The bandwidth is h = m·sd, with (m_N, m_D) chosen by leave-one-out CV over {0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5}².
- **Compute levels.** 12 log-spaced levels from 3×10¹⁴ to 6×10¹⁶ FLOPs. A level is used only if its isocost crosses at least three widths inside their observed D ranges (5 percent slack).
- **Path point.** On each isocost, the root of ln w_local − ln w_path, with w_path = η(N) for P and 1 for P6 and T.
- **σ\* at the path point.** From the local gradient and Hessian, via the plan's identity generalized to the P cost:
  1/σ* − 1 = (L_nn|_C − η′·|dL*/d ln C|) / (η(1+η)·|dL*/d ln C|).
  This reduces to L_nn|_C/(2|dL*/d ln C|) at η = 1. Equivalently it is the Hicks elasticity between N and D at that point (`m2_est.sigma_from_derivs`). The code computes both forms and checks that they agree.
- **Model-free σ\*.** The mean over the valid compute levels.
- **"Slope" variant** (w = 1 paths only; ra1's first-derivative estimator). Regress ln w_local on u = ln(M/M*_local(C)) and u² through the origin over a fine grid inside the design; σ* = 1/(1+b₁).
- **Local wedge.** w_local = f_N/f_D at each endpoint.

**Parametric estimators [§2(ii)].**
- **Chinchilla form**: `sl.fit_chinchilla` with Huber δ = 10⁻³ on ln L. Starting values are sl's FAST grid (432 starts) plus m2's 48 level-preserving starts.
- **κ family**: `m2_est.fit_q`, L = E + (A N^−a₁ + B D^−b₁)^κ with σ*_κ = 2/(2+a₁+b₁), started from the Chinchilla solution at κ ∈ {1, 0.5, 0.75, 1.5, 2, 3}.
- **Robustness**: Gaussian NLS versions of both.
- **Under P**, the parametric σ is also evaluated on the fitted technology's own FLOP-optimal path at the same compute levels (`sig_kappa_Ppath`, `sig_chin_Ppath`), so that it can be compared with the model-free P estimate like for like.

**Q2 (neutrality) [§3].**
- **Corpus-pair model.** Common (α, β) and corpus-specific (ln A, ln B, ln E), estimated with Huber loss (`m2_est.fit_panel_ls("CE")`). It is started from separate fits embedded by `m2_est.embed`.
- **Tilt and implied ratios.** χ̂ = Δln(A/B), edu − web. The implied ratios are M*_edu/M*_web = e^{−2χ̂/(α+β)} on the w = 1 path, plus a numerical ratio on the P FLOP path at 10¹⁵, 10¹⁶ and 10¹⁷ FLOPs, and the ŵ factor e^{−χ̂}.
- **Tests.** (a) A Wald test of equal (α, β) from separate fits, with the bootstrap covariance. (b) The tilt interval.
- **Decision rule.** Exactly as written in the plan (`m9_stages.q2_decision`). It is run per validation set (edu, web) and on WikiText where a pair model is identified. Matched-cell differences on all three sets give the quality versus distribution-match sign check.

**Q3 (extrapolation) [§3].**
- **Fits.** The Chinchilla form (primary) and the κ family (secondary) are fitted by Huber loss on main-grid endpoints with M ≤ 100, and again on the full support (main + hiM).
- **Local wedge.** w_local comes from the local quadratic on the full support, with a CV bandwidth.
- **Statistic.** Δᵢ = ln(w_restricted/w_local) at every endpoint with M > 100. The reported slope is the OLS slope of Δ on ln M (with intercept, centred at M = 100), together with bin means (M in 100–316, 316–1,000 and > 1,000).
- **Comparators.** κ(M ≤ 100) vs local; full support vs local; restricted vs full support.
- **Per-endpoint table.** ε_N, ε_D and w under each fit.

**Q4–Q6 [§3].**
- **Q4.** Seed s.d. of ln L at each replicated cell (seed replicates plus the seed-0 run of the same cell), pooled. The within-trunk correlation is an equicorrelation estimate from seed deviations across budgets within a seed trunk. These are compared with the residual s.d. of the Huber fits.
- **Q5, calibration.** Quadratic in ln LR through the calibration LRs (fit_lr.py's rule, clipped to the tested range ± 0.35). The argmin is compared by corpus at each width.
- **Q5, corners.** LR multipliers {0.5, 1, 2}. The excess of the rule is ln L(rule) − min(quadratic minimum, best tested). The drift is d ln LR*/d ln D at width 128.
- **Q5, bound on a.** The flexible-input bound a_obs − a = −(ι_n − ι_d)E/[(α+β)R*]. The transverse gradient comes from the two corners at similar compute, (640, 25M) and (128, 800M), both at about 7–8×10¹⁵ FLOPs.
- **Q6.** The on-path subsample is the endpoint nearest the κ-family expansion path (P: FLOP path; T: w = 1) in each doubling bin of compute. Two per bin is a sensitivity. The Jacobian condition numbers of the Chinchilla form (KMW-normalized) come from m6's `jac_lse`. The κ-family profile likelihood ratio over σ* on {0.05, 0.07, …, 0.99} is taken relative to the unconstrained optimum (`mc_lib.profile_sigma`, the Section II machinery).

**Inference [§2].**
- **Primary: wild cluster bootstrap by trunk.** One cluster per architecture (d, L): eight per corpus, and nine with hiM (128,4). hiM (256,4) shares width 256's cluster because its trunk reproduces the main-grid trunk.
- **Draws.** Webb six-point weights, 999 draws. One weight per architecture is shared by both corpora and by every estimator within a draw. This keeps any cross-corpus dependence and gives joint draws for the differences.
- **Bootstrap population.** Parametric statistics resample the residuals of their own fit. Model-free statistics and Q3 resample residuals around the local-quadratic surface (as in ra1).
- **Intervals.** Basic intervals, est − (q(draws) − pop), where pop is the statistic on the noise-free population. Equality p-values are symmetric bootstrap p-values.
- **Secondary** (only once seeds exist). A Gaussian parametric bootstrap with the seed-estimated s.d. and within-trunk equicorrelation (`scheme = seedcov`), for Q1 (main), Q2 and Q3.
- **Robustness criterion.** A conclusion is called robust only if it holds under both schemes.

**Reproduction.** `.venv/bin/python code/analysis/m9_sweeps/run.py`. It uses seeds derived from 20260924 by stage key, runs CPU-only with at most 4 workers, and never imports MLX. `--power` re-runs the power calculation (about 50 min).


---

## 3. Inventory (at the 17:24 pass; `output/tables/m9_sweeps_inventory.csv`, `m9_sweeps_design.csv`)

| Tag | FineWeb-Edu completed / planned | FineWeb completed / planned |
|---|---|---|
| main | **44 / 44** | 15 / 44 (widths 128 and 192 complete, width 256 at 25–100M; running) |
| lrsweep (edu) / lrcal (web) | 12 / 12 | 12 / 12 |
| lrcorner | 16 / 16 | 15 / 16 (width 640 at 2×, 50M pending; running) |
| hiM | 0 / 8 | 0 / 4 |
| seedcorner | 0 / 3 | 0 / 3 |
| seeds | 0 / 18 | 0 / 18 |

**Checks.**
- No endpoint has a non-finite or diverged loss.
- There are no duplicate records.
- The architecture formulas reproduce `N_nonemb`, `N_total` and `flops_per_token` exactly, and C = fpt × tokens holds for every record.

**Wall-clock.** The sum over trunks of time to the last recorded endpoint is a lower bound, with two queues sharing the GPU:
- FineWeb-Edu: main 12.7 h, lrcorner 4.2 h, lrsweep 0.9 h.
- FineWeb: main 4.7 h (includes a sleep), lrcorner 4.6 h, lrcal 2.1 h.

**FineWeb-Edu main-grid design.**
- **Convention P:** M from 0.51 to 2,031. 11 endpoints have M > 100 and 2 have M > 1,000 (both at width 128). sd(ln M | ln C) = 2.02.
- **Convention T:** M from 0.46 to 555. 6 endpoints have M > 100 and none has M > 1,000. sd(ln M | ln C) = 1.74.
- **Compute:** 2.3×10¹⁴ to 7.3×10¹⁶ FLOPs.
- **Measurement ratios:** the embedding share is 10–73 percent; FLOPs/(6N_nonemb) is 1.14–3.83; FLOPs/(6N_total) is 1.03–1.05.

## 4. Claims for the paper (to be finalized; do not use before the final run)

For the TBD-m9 placeholders, and only if the final run confirms them:

1. **(Q1)**
   - "On our own annealed sweep the model-free σ* is about 0.67 with non-embedding parameters (0.61 with total parameters). It agrees with the κ-free estimate and lies below the κ = 1 value, as on the public IsoFLOP designs."
   - The σ*_edu = σ*_web test is pending.
   - Report the convention gap as expected by construction (§0.2).
2. **(Q2)** Pending. State the decision under both the plan rule and the size-corrected rule. If they disagree, state that the plan's interval is anti-conservative on this design (§0.2–0.3).
3. **(Q3)**
   - Only a *level* statement is supported so far: "a Chinchilla form fitted on M ≤ 100 understates the local wedge at M > 100 in our sweep".
   - Whether the understatement grows with M (the plan's slope statistic) currently rests on one two-layer endpoint and is not decided.
   - Keep the ex ante caveat that levels need not transfer.
4. **(Q4)** Pending (seed s.d.).
5. **(Q5)**
   - "The learning-rate rule is within 0.1 percent of the best of three LRs at the high-M corner for FineWeb-Edu. It is about 0.2–0.5 percent too high at the largest width. FineWeb's optimum at width 128 is higher than the rule."
   - The implied bias in a is below 0.015 in absolute value.
6. **(Q6)** "In our own losses, on-path runs leave σ* between 0.35 and 0.95. The full factorial grid pins it within ±0.02."

## 5. Robustness (FineWeb-Edu; `m9_sweeps_q1_sigma.csv`, `m9_sweeps_q1_bandwidth.csv`, `m9_sweeps_q3_extrap.csv`)

**σ\* by sample.**

| Sample | Model-free (P, FLOP path) | κ family (P) | Model-free (T) | κ family (T) |
|---|---|---|---|---|
| Main grid | 0.666 | 0.662 | 0.607 | 0.605 |
| Drop D = 25M | 0.673 | 0.675 | 0.629 | 0.619 |
| Four-layer floor | 0.660 | 0.651 | 0.619 | 0.617 |

**Other checks.**
- **Bandwidth.** The model-free σ* at 0.75×, 1×, 1.5× and 2× the CV bandwidth is 0.662 / 0.666 / 0.673 / 0.680 (P) and 0.606 / 0.607 / 0.617 / 0.622 (T). The extended-grid CV optimum gives 0.660 (P) and 0.605 (T).
- **CV boundary.** The CV optimum for ln N sits at the lower edge of both grids (§7.9).
- **Identity check.** The identity form and the Hessian form of the model-free σ* agree to machine precision.
- **Slope variant.** The first-derivative (slope) variant is lower: 0.631 (P, w = 1) and 0.586 (T).
- **Estimator.** Gaussian NLS moves σ*_κ by at most 0.007.
- **Q3 slope by sample** (P): −0.29 (main), −0.26 (drop 25M), −0.12 (four-layer floor, 3 points). In T: −0.76 and −1.09.
- **Q3 levels** are negative in every sample and convention except the T intercept at M = 100.
- **Secondary inference.** The seed-covariance bootstrap is not yet run (no seeds).

## 6. Deviations from the pre-analysis plan, with reasons

| # | Deviation | Reason | Effect |
|---|---|---|---|
| D1 | High-M runs trimmed: FineWeb-Edu (128,4),(256,4); FineWeb (128,4) only; the 3.2B continuation of (128,2) dropped | GPU time (recorded after the plan) | Less reach and replication for Q3, especially for FineWeb |
| D2 | A machine reboot at 15:34 interrupted both queues. Interrupted trunks were re-run from scratch | Hardware | Some trunks' early and late endpoints come from two runs of the same seed. Assumed identical up to kernel nondeterminism; unverified until the duplicate cells in §7.1 finish |
| D3 | Bits per byte use the bytes/token of the **scored** validation tokens, not `meta.json`'s full-file value | The plan's definition ("bytes/token of the evaluation set") read literally. `meta.json` differs by −0.8 percent for FineWeb | Levels on FineWeb validation only; exponents, σ*, tilts and wedges unaffected. Both versions are stored (`bpb_*_meta`) |
| D4 | Under P the compute-optimal path is w = η(N) (actual FLOPs), so the plan's identity is applied in its generalized form. A "P6" variant (w = 1 path, non-embedding N) and parametric σ on the FLOP path are added | The plan's formula holds only for C = 6ND; under P, C = fpt(N)·D with η = 0.52–0.93 | Clarification, not a change of object. P6 is reported alongside |
| D5 | Model-free details not fixed by the plan: LOO-CV bandwidth over {0.4…1.5}×sd; 12 compute levels; at least 3 widths per isocost; averaging over levels | Needed to implement §2(i) | Bandwidth sensitivity is reported (§5) |
| D6 | Intervals are basic bootstrap intervals; Q2 tilt equality uses joint draws with one weight per width shared by both corpora | Interval type and cross-corpus handling unspecified | None |
| D7 | Q2 on WikiText. Main-grid runs were never evaluated on WikiText, so a corpus-pair model on WikiText is not identified. WikiText enters through matched-cell sign checks (lrcorner, seedcorner, hiM, FineWeb seeds) | Design: evaluation on WikiText was added only for the new runs | The quality vs distribution-match classification on WikiText rests on few cells |
| D8 | **Size-corrected companion inference**, pre-declared before estimation (§0.3): CR2 cluster residuals; bandwidth re-selected in each draw; restricted wild bootstrap test of χ = 0; calibration factors for the model-free and Q3 intervals. The plan's intervals are still reported | The design Monte Carlo shows the plan's intervals cover 40–83 percent at nominal 95, and the tilt test rejects a true null about 20 percent of the time | Conclusions are called robust only under plan + companion (+ seeds) |
| D9 | Power DGP choices: Besiroglu exponents; anchor M* = 20 and 3.8 nats; ρ = 0.5; 199 draws in the power bootstrap (999 in estimation) | The plan fixes noise levels but not the DGP | Sensitivity cells reported (§0.1) |
| D10 | Q6 compute levels are doubling bins of compute. The on-path point is the one nearest the κ-family path (1 per bin; 2 per bin as sensitivity) | The plan does not define "each compute level" on a factorial grid | — |
| D11 | Q5 argmins: quadratic in ln LR clipped to the tested range ± 0.35 (the rule in `fit_lr.py`); corner excess measured against the lower of the quadratic minimum and the best tested LR | Implementation of "interpolated argmins" | Grid-based excesses are also reported |
| D12 | Clusters are architectures (d, L). hiM (256,4) shares width 256's cluster; hiM duplicates of main cells are excluded from fits | Trunk sharing (§7.1) | — |

## 7. Open issues and things in the design or data that look wrong

1. **Duplicated cells.**
   - `hiM` (256,4) has the same shape, seed, learning rate and data order as main-grid width 256 (the extension data are appended *after* the original permutation). Its 0.2B and 0.8B endpoints therefore reproduce main-grid cells up to GPU nondeterminism. Only 1.6B and 3.2B are new. The effective new high-M shape is (128,4), in both corpora.
   - Likewise `seedcorner` seed 1 at width 128 reproduces the `seeds` seed-1 trunk at 200M.
   - These cells are dropped from the fits and used as nondeterminism checks (`m9_sweeps_duplicate_cells.csv`, once present).
   - If GPU time is short, the (256,4) 0.2B/0.8B branches and the seedcorner 200M branch can be skipped (their trunks are still needed).
2. **The learning-rate calibration is not bracketed at two of six cells.**
   - At width 512 both corpora do best at the lowest tested LR (10⁻³).
   - At width 128 FineWeb does best at the highest (8×10⁻³).
   - The rule's width-512 optimum (1.66×10⁻³) is set by a quadratic dominated by the 8×10⁻³ blow-up. At width 640 the 0.5× corner beats the rule.
   - So the rule is likely too high at large widths. That is an N-gradient in inefficiency, which lowers the fitted a (bound in P3: |bias| < 0.015).
   - A width-128 FineWeb mistuning is corpus-specific, which is the confound R3 warned about for Q2.
3. **The two-layer width-128 model has an irregular D-response.**
   - Loss decrements are non-monotone in ln D. Main grid: −0.10 at 50→100M, −0.16 at 100→200M, −0.03 at 400→800M. At 0.5× LR: −0.17 at 200→400M.
   - This looks like a late, learning-rate-dependent drop in the two-layer model (the constant-LR trunk curve plateaus from about 160M to 600M tokens).
   - It (a) drives the Q3 slope through the (128, 800M) endpoint, (b) contaminates between-corpus contrasts at width 128 (matched-cell jumps of 4–6 percent on WikiText), and (c) is poorly fitted by any smooth form.
   - The four-layer-floor robustness sample exists for this reason. The (128,4) hiM runs will show whether the high-M corner behaves once depth ≥ 4.
4. **No WikiText on the main grid** (D7). The neutral-set check for Q2 can only be a sign check on off-rule-LR and seed cells. FineWeb-Edu seed runs (`seeds edu`) are also not evaluated on WikiText (`run_grid.py`: evals="edu,web"). Adding WikiText to them is cheap if the queue has not reached them.
5. **The plan's inference is anti-conservative on this design** (§0.2–0.3). This is the largest methodological risk for the paper's claims. The pre-declared companion fixes the size of the tilt test, but not the coverage of the model-free and Q3 intervals (best 0.65–0.75).
6. **Q3's null is not zero.** The design's null value is −0.01 to −0.05 in P and +0.11 or −0.08 to −0.10 in T (with and without the high-M runs), and it depends on noise. In T the Chinchilla form is misspecified by construction when it holds in non-embedding units. The paper should use P for the sign (as the plan says) and compare with the design null.
7. **The high-M cells have no learning-rate check.** The hiM runs use width 128's rule LR (calibrated on the two-layer model) for the four-layer model. The Q3 extrapolation leans on exactly these cells.
8. **Seed replicates vary both initialization and data order** (seed = data_seed), so Q4 measures their combined noise, which is the relevant object. The within-trunk correlation needs seeds at several budgets per trunk. The design has 3 per trunk, so ρ̂ will be noisy; it is clipped to [0, 0.99].
9. **The CV bandwidth sits on the grid's lower edge** for ln N (both conventions; also the extended grid in T). The model-free σ* moves by 0.02 across ×0.75–×2 bandwidths. The per-level σ*(C) jumps near C ≈ 10¹⁶ (the set of widths entering the local fits changes).
10. **Bytes per token.** `meta.json` records full-file values; the scored first 1,048,576 tokens of FineWeb validation have 0.8 percent fewer bytes per token (D3).
11. **Cosmetic.** The plan header says "Written 03:15" but the commit is 03:12:31.
12. **Still to do after the final data.** Run Q2 and the Q1 equality test. Run the secondary seed-covariance inference. Check the duplicate cells for nondeterminism. Fill Q4. Re-run Q3 with the hiM runs (the power calculation shows they matter for T). Then an independent review (`m9_sweeps_review.md`).

## Files

**Code** (`code/analysis/m9_sweeps/`):
- `run.py` (entry; `--power` re-runs the power and coverage stages; `--outputs-only` rebuilds tables and figures from the CSVs);
- `m9_common.py`, `m9_est.py`, `m9_stages.py`, `m9_outputs.py`, `m9_power.py`, `m9_coverage.py`.

**Tables** (`output/tables/`):
- `m9_sweeps_{power, power_coverage, inventory, design, crosseval, matched_cells, duplicate_cells, q1_sigma, q1_levels, q1_bandwidth, q2_tilt, q2_fits, q2_decision, q3_extrap, q3_points, q4_noise, q4_cells, q5_calibration, q5_corners, q5_drift, q5_abias, q6_onpath, q6_profiles, secondary_seedcov}.csv`;
- LaTeX (booktabs, `\footnotesize`, AEA `tablenotes`; compiled with AEA.cls without overfull boxes): `m9_sweeps_{power, design, sigma, tilt, extrap, noise_lr, onpath}.tex`.

**Figures** (`output/figures/`, PDF and PNG):
- `m9_sweeps_fig` (paper figure: (a) loss vs D by width on FineWeb-Edu validation with fitted curves; (b) isoquants and expansion path, total N);
- variants `m9_sweeps_fig_P` (non-embedding N) and `m9_sweeps_fig_webval` (FineWeb validation);
- `m9_sweeps_sigma`, `m9_sweeps_wedge`, `m9_sweeps_lr`, `m9_sweeps_profile`, `m9_sweeps_power`.

**Logs** (`data/processed/m9_sweeps/`): `run_log.txt`, `run_stdout*.txt`, `power_stdout.txt`, `coverage_*`, `estimation_start.txt`, `summary.json`.
