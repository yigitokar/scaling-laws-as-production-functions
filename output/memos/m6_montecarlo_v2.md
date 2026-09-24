# Memo — m6_montecarlo v2: Monte Carlo re-run with no estimator started at the truth

Module owner: mc-fix agent (Claude), 2026-09-24. Supersedes the Design A numbers of `m6_montecarlo.md` (v1, 2026-09-23)
and, where they differ, its Design B numbers. v1's methods (§2) and verification results (H5, H6) still apply
unless changed below. Entry point: `code/analysis/m6_montecarlo/run_v2.py`
(`--procs 4`; add `--design-b` to re-run the industry; `--skip-sim` rebuilds outputs in about 2 s).

**Why.** In v1, one of the nine starting values of the primal (Huber, κ = 1) estimator was the true parameter vector.
The dual frontier fit and the bootstrap check also used it, and every Design B least-squares estimator started at the
truth. Referees objected: R3 M12 and minor 13, R1 minor 2, R2 minor 7. R2 minor 8 asked for heteroskedastic and
clustered noise. R1 comment 9c and R4 M5b noted that the v1 grid [0.50, 0.95] bounded the κ-free confidence sets.

**What was done.**
1. *No start at the truth anywhere.*
   - Primal: nine random starts over Hoffmann et al.'s grid region (`ml.STARTS`). The first eight are v1's non-truth
     starts; the ninth is a new random draw.
   - Dual frontier fit: a 2 × 2 × 2 grid of starts.
   - Bootstrap check: warm start plus the nine random starts.
   - Design B nonlinear least squares: nine data-based starts (α, β ∈ {0.2, 0.5, 1}², levels matched to mean y),
     keeping the lowest cost. The pooled Huber fit uses the nine random starts. E1b holds E at the truth, as before.
2. *κ-free profile.* The grid is extended to 29 points on [0.05, 0.99]. The unrestricted optimum is searched over
   σ* ∈ [0.024, 0.995], and LR is measured against it. A second observationally equivalent start, built from the
   dual/path fit, is added at every grid point.
3. *New noise cells.* Six cells with heteroskedastic and clustered noise (details in §3).
4. *New diagnostics.* The objective at the truth is stored, never used as a start. This separates non-identification
   (estimate's objective ≤ the truth's) from optimizer failure. The summary also reports Monte Carlo standard errors
   of coverage rates.
5. *Robustness checks.* 64 random starts (`designA_startcheck.py`). Truth vs truth-free starts in Design B
   (`designB_startcheck.py`, `designB_startcheck2.py`).

**Unchanged.** Designs, noise, seeds, and replication counts: 500 per cell, 150 profile and bootstrap replications,
149 bootstrap draws, and a 60 × 49 bootstrap check. **No replications were cut.** Every v1 cell keeps its seed, so the
simulated data are identical to v1 (verified: the objective at the truth replays exactly, max |Δ| = 0).

**Runtime.** Four CPU processes, with the GPU sweep running alongside:

| Step | Wall-clock |
|---|---|
| Design A | 17.8 min |
| Bootstrap check | 2.9 min |
| Design B | 5.6 min |
| 64-start check | 14.8 min |
| Design B start checks | 1.3 min |
| **Total** | **≈ 42 min** |

An earlier attempt at 08:00 was interrupted at 45 min under heavy machine load, and a restart at 09:32 was lost to a
reboot. Their logs are `data/processed/m6_montecarlo/run_v2_interrupted_*.log`. The v1 outputs are archived in
`data/processed/m6_montecarlo/v1_backup/` (code, parquet, tables, figures).

Notation: the allocation-error s.d. is **v**, following Section II, because s is the expenditure share (w − 1)/w in v2.
Cell names in the CSVs still read `opt_s<v>`.

---------------------------------------------------------------------------------------------------

## 1. What changed, and by how much (Design A)

**Only near-path cells change (v ≤ 0.1, on-path, Kaplan).**
- Estimates are bit-identical to v1 for:
  - the IsoFLOP ±16× and ±4× designs, the factorial grid, and the digitized and noise-level cells;
  - v ≥ 0.2: max |Δσ̂*| = 0, max |Δ ln M̂*| ≤ 4 × 10⁻⁵.
- The dual estimator is identical in every cell. Replacing its truth start changed nothing.
- Sources: `m6_montecarlo_designA_v2_vs_v1.csv`, `m6_montecarlo_designA_summary.csv`, and the parquet merge in §6.

**Primal estimator, three start sets on identical data.** Source: Table C2 (`paper/tables/appC_starts.tex`), from
`v1_backup/tables/m6_montecarlo_designA_summary.csv`, `m6_montecarlo_designA_summary.csv` and
`m6_montecarlo_designA_startcheck.csv`.

Corner share (%):

| Design | v1: truth + 8 random | **v2: 9 random (baseline)** | 64 random |
|---|---|---|---|
| On path, v = 0.02 | 26 | **72** | 18 |
| v = 0 | 51 | **87** | 56 |
| v = 0.05 | 0.6 | **13** | 0.8 |
| v = 0.1 | 0 | **0.2** | 0 |
| v = 0.2 | 0 | **0** | 0 |
| Kaplan | 0 | **0** | 0 |

RMSE of σ̂* and of ln M̂*(10²⁴), reported as σ̂* / ln M̂*:

| Design | v1: truth + 8 random | **v2: 9 random (baseline)** | 64 random |
|---|---|---|---|
| On path, v = 0.02 | 0.185 / 16.9 | **0.222 / 23.8** | 0.189 / 15.5 |
| v = 0 | 0.282 / 24.0 | **0.272 / 27.7** | 0.302 / 24.8 |
| v = 0.05 | 0.067 / 5.87 | **0.089 / 10.2** | 0.071 / 5.78 |
| v = 0.1 | 0.0246 / 4.49 | **0.0251 / 4.58** | 0.0248 / 4.51 |
| v = 0.2 | 0.0147 / 2.99 | **0.0147 / 2.99** | 0.0147 / 3.00 |
| Kaplan | 0.101 / 4.97 | **0.102 / 4.97** | 0.104 / 5.00 |

- **Median error of σ̂*.**
  - On path: −0.030 (v1) → −0.119 (v2) → −0.026 (64 starts).
  - v = 0: −0.297 → −0.267 → −0.325.
- **MAE of σ̂*.** On path: 0.037 → 0.119 → 0.044.
- **Objective at the estimate above the objective at the truth** (optimizer failure), with nine starts:
  - 2.4% on path;
  - 0% at v = 0;
  - 7.0% at v = 0.05;
  - 0.2% at v = 0.1.

  With 64 starts it is 0 everywhere. At v = 0 every corner fit is at least as good as the truth: this is
  non-identification, not optimizer failure.

**Reading.** Removing the truth makes the near-path results worse.
- The v1 caveat that the truth start "favours near-truth solutions on the ridge" is confirmed, but the effect is small
  once the optimizer has enough random starts.
- With 64 random starts, on-path, v = 0 and v = 0.05 RMSEs are within 2–7% of v1.
- What the truth start hid is that on a ridge the estimate depends on the start set. The corner share at v = 0 is 51%,
  87% or 56% depending on the starts.
- The thesis does not depend on the start set. Near-path RMSEs are 20–80× the IsoFLOP 0.0036 in every column.
- **Baseline for the paper:** nine random starts, the pre-specified v1 count without the truth. The 64-start column is
  reported in Table C2.

**Inference (primal σ*).**
- Wald singular or corner on path: 35% (v1) → **79%** (computable 21.4%).
- Warm-start pairs bootstrap on path, 150 reps: coverage 67% → **44%** (MC s.e. 4.1 pp). For a: 32%; for ln M*: 43%.
- Bootstrap check (60 reps × 49 draws, same resamples):
  - on path, warm 57% → **33%**; multi-start 92% → **78%** (multi = warm + 9 random starts; v1 multi included the
    truth);
  - multi-start median width unchanged at 0.43 (v1 0.428; v2 0.426);
  - warm width 0.28 → 0.04;
  - at v = 0.3, unchanged: 100%, width 0.053.
- Unchanged:
  - IsoFLOP ±16×: Wald 96.0% (MC s.e. 0.9), bootstrap 96.7% (1.5), widths 0.015 / 0.016;
  - factorial: 94.6% / 96.7%, widths 0.019 / 0.019;
  - v = 0.3: 93.2% / 96.0%;
  - v = 1: 95.4% / 95.3%.

**κ-free profile.** Grid [0.05, 0.99] in v2 vs [0.50, 0.95] in v1; 150 reps. Source: `m6_montecarlo_designA_profile.csv`.

| Cell | Flat on (0.05, 0.99) | Contains all of [0.50, 0.95] (v1 "flat") | v1 flat | Median bounds | Covers truth |
|---|---|---|---|---|---|
| On path, v = 0.02 | 91% | 94% | 88% | [0.05, 0.99] | 100% |
| v = 0 | 95% | 97% | 97% | [0.05, 0.99] | 100% |
| v = 0.05 | 79% | 91% | 91% | [0.05, 0.99] | 96% |
| v = 0.1 | 21% | 93% | 93% | [0.08, 0.99] | 97% |
| v = 0.2 | 0% | 89% | 89% | [0.28, 0.99] | 98% |
| v = 0.3 | 0% | 65% | 65% | [0.42, 0.99] | 95% |
| v = 0.5 | 0% | 0.7% | 0.7% | [0.585, 0.98] | 93% |
| v = 1 | 0% | 0% | 0% | [0.691, 0.792] (width 0.104) | 94% |
| v = 2 | 0% | 0% | 0% | width 0.028 | 95% |
| IsoFLOP ±16× | 0% | 0% | 0% | [0.729, 0.745] (0.016) | 94% |
| IsoFLOP ±4× | 0% | 0% | 0% | [0.714, 0.760] (0.047) | 93% |
| Factorial | 0% | 0% | 0% | [0.727, 0.746] (0.019) | 95% |
| Kaplan | 21% | 84% | 77% | [0.08, 0.99] | 95% |

- **New finding: identification is asymmetric.** Near-optimal allocations first rule out poor substitutes; the upper
  end goes last.
  - The set excludes some σ* ≤ 0.5 in 72% of replications at v = 0.1 and in all replications at v ≥ 0.2.
  - It still contains every σ* ≥ 0.9 in 76% at v = 0.3 and 45% at v = 0.5.
- **Why.** An allocation error δ in ln M costs a Farrell loss of about (1 − σ*)δ²/(4σ*). That loss is large for low σ*
  and vanishes as σ* → 1. (It follows from Φ ≈ σ*(ln w)²/(4(1 − σ*)) and ln w = (1/σ* − 1)δ.)
- **The v1 "flat" measure is unchanged except in two cells.** On path it rose from 88% to 94%, and under Kaplan beliefs
  from 77% to 84%. The added dual-based equivalent start removes optimizer failures that v1 counted as rejections at
  the edges of [0.50, 0.95].
- Coverage is 93–100% in every cell (MC s.e. ≤ 2.1 pp).

**Noise level** (unchanged from v1 except that the whole-grid flat shares are now 0 at v = 0.3):
- IsoFLOP RMSE of σ̂*: 0.0024 / 0.0036 / 0.0078 at noise s.d. 0.005 / 0.0075 / 0.015.
- At v = 0.3, the set contains all of [0.50, 0.95] in 38% / 65% / 90% of replications.
- Median widths on the new grid: 0.46 / 0.55 / 0.70.

## 2. Heteroskedastic and clustered noise (new; R2 minor 8)

**Specification** (`ml.het_cluster_noise`):
- s.d._i ∝ (N_i D_i)^(−1/4), so noise is larger for small models and short runs. It is scaled so that the
  design-average variance equals 0.0075².
- Correlation 0.5 within clusters. Clusters are the nine compute budgets (path, optimizing and IsoFLOP designs; shared
  data order) or the nine model sizes (factorial grid; one trunk per size).
- At fixed compute the two size effects offset. Within an IsoFLOP budget the noise is therefore homoskedastic, and
  across budgets it falls 4.7× from the smallest budget to the largest (500^(1/4)).

**Results** (Table C3, `paper/tables/appC_noise.tex`):

| Cell | RMSE σ̂* (iid → het/cl) | RMSE ln M̂* | Wald cover | Profile covers truth | Flat, full grid / [0.50, 0.95] |
|---|---|---|---|---|---|
| On path | 0.222 → 0.247 | 23.8 → 25.2 | 100†→78† | 100 → 73% | 91/94 → 61/65% |
| IsoFLOP ±16× | 0.0036 → 0.0042 | 0.16 → 0.090 | 96 → 82% | 94 → 71% | 0 → 0 |
| Factorial | 0.0050 → 0.0052 | 0.28 → 0.27 | 95 → 74% | 95 → 82% | 0 → 0 |
| v = 0.1 | 0.025 → 0.037 | 4.58 → 4.16 | 99† → 74† | 97 → 92% | 21/93 → 11/69% |
| v = 0.3 | 0.013 → 0.022 | 1.97 → 1.47 | 93 → 61% | 95 → 91% | 0/65 → 0/42% |
| v = 1 | 0.012 → 0.019 | 0.59 → 0.43 | 95 → 60% | 94 → 87% | 0 → 0 |

(† = the Wald covariance is singular or at a corner in more than 5% of replications, as in Tables C1 and C3.)

- **The ranking of designs is intact.** Designed experiments remain 5–60× more precise for σ* than near-optimal data.
  The κ-free profile is still flat on path in 61% of replications.
- ln M* improves under the IsoFLOP design: the large budgets, which pin M*(10²⁴), are less noisy.
- **Inference that assumes independence fails.** Wald coverage is 60–82%; χ²-calibrated profile sets cover 71–91%.
- This supports cluster-robust inference (wild cluster bootstrap over trunks, as pre-registered for m9).
- Caveat: the flat shares fall partly because correlated noise inflates LR; they are not comparable with the iid
  cells as evidence of information.

## 3. Design B (industry) re-run without truth starts

**Source:** `m6_montecarlo_industry_v2_vs_v1.csv`, `m6_montecarlo_industry_summary.csv` (v2),
`m6_montecarlo_industry_startcheck.csv`, `m6_montecarlo_industry_startcheck2.csv`.

**Why the re-run was needed.** v1 started E2, E3, E4, E5 and E8 at the truth. `designB_startcheck2.py` compares least
squares from the truth, from a neutral start, and from a 9-start data-based grid on the first 100 replications of
rules (a)–(d):
- The grid reaches a cost at least as low as the truth start's in 99–100% of replications.
- It finds a strictly lower cost in 4–44%. Most are numerically tiny, but under the **target rule** the pooled criterion
  has several local optima.
- There the truth start had picked the optimum nearest the truth in about 15% of replications.

**Changes (v1 → v2, 400 replications).**
- γ bias: max |Δ| = 0.0024 over all scenarios and estimators.
  - Target rule, pooled NLS: −0.0511 → −0.0535, i.e. **−29% → −30% of γ**.
  - Lab FE: −0.021 → −0.022.
  - ACF with current c: −0.026 → −0.027.
- TFP growth bias: max |Δ| = 0.0002.
- Allocation exponent a, target rule: pooled NLS −0.032 → −0.060; Mundlak −0.019 → −0.031; ACF −0.037 → −0.055.
- ln M* bias, target rule: pooled NLS −0.18 → +0.10; ACF +0.31 → +0.52; Mundlak +0.04 → +0.18.
- ML-practice pooled Huber (E1), ln M* bias: +0.56 / +0.57 / +0.68 → **+0.53 / +0.55 / +0.64** (rules a / b / d).

**Unchanged** (to the printed precision):
- Pooled NLS γ bias: +0.023 (13%) under funding, +0.018 under predetermined budgets, +0.118 (66%) with flagships only.
  v1 printed +0.119 for the flagship cell; the value is 0.1185 in both runs.
- Within-family estimators: |bias| ≤ 0.002. ACF with lagged instruments: |bias γ| ≤ 0.002.
- FOC system ln M* bias: +1.25 to +1.28. With T proxied: −0.02 to −0.03.
- IV: median ln M* bias +1.27 to +1.32; median γ bias −0.002 to +0.005.
- ML practice: TFP growth −0.081 to −0.090. E1b misses 36–43% of TFP growth; γ bias +0.050 to +0.059.
- Selection: pooled −0.008 / +0.011; Heckman −0.003 / +0.006; predetermined +0.01 → +0.018 after Heckman.
- ME in D: within-family ln M* −0.33, lab FE −0.28.

## 4. Replacement numbers for the main text (integrator)

**`paper/sections/identification.tex`, lines marked `% m6v2`.** Remove the red TBD-m6v2 flags.

| Current text | v2 replacement |
|---|---|
| "ends on a corner … in 51 percent of replications" (v = 0) | **87 percent** (nine random starts; 56 percent with 64 starts; Table C2) |
| "falls from 0.067 to 0.015" (v = 0.05 → 0.2) | **from 0.089 to 0.015** |
| "4.5 at v = 0.1 and 0.59 at v = 1" | **4.6** and 0.59 |
| "IsoFLOP … 0.0036 and 0.16" | unchanged |
| "covers the entire grid searched, [0.50, 0.95], in 88 percent of on-path replications and in 65 percent at v = 0.3" | Grid is now [0.05, 0.99]. Suggested: "covers the entire grid searched, [0.05, 0.99], in 91 percent of on-path replications; at v = 0.3 it rules out low elasticities but still contains all of [0.50, 0.95] in 65 percent" |
| "against widths of 0.016 and 0.019" | unchanged |
| "falls to 38 percent when the noise s.d. is 0.005" | Under the [0.50, 0.95] definition, unchanged (38%). Say "the share of sets containing all of [0.50, 0.95]" |
| "Wald … singular, or … corner, in 35 percent" | **79 percent** |
| "warm-started … covers … 57 percent …, restarting … restores 92 percent coverage with intervals 0.43 wide" | **33 percent … raises coverage to only 78 percent, with intervals 0.43 wide** |

**Figure 2 notes (`fig:designs`).**
- Starts are "nine values drawn at random over the region of Hoffmann et al.'s initialization grid, none at the truth".
- Panel A: the dual estimator is the gray dashed line; the IsoFLOP ±16× (orange dashed) and factorial (green dotted)
  lines are primal.
- Panel B: median LR over a 29-point grid on [0.05, 0.99], relative to the unrestricted (κ, σ*) optimum, 150
  replications; curves for on-path (v = 0.02), v = 0.3 and IsoFLOP ±16×; horizontal line, the χ²₁ 95% value; dotted,
  the truth.
- The figure file is `output/figures/m6_montecarlo_fig2_v2.pdf`. The `\IfFileExists` switch in identification.tex will
  pick it up.

**`paper/sections/appendix_observational.tex`, line 258.** "understates it by 29 percent under a capability target"
→ **30 percent** (Design B v2; −0.0535/0.1783). The 13% and 66% figures are unchanged.

## 5. Files

**Figures** (`output/figures/`, PDF + PNG):
- `m6_montecarlo_fig2_v2`: **paper Figure 2**, two panels.
  - A: RMSE of σ̂* (top) and ln M̂*(10²⁴) (bottom) against v, for the primal (κ = 1) and dual estimators, with IsoFLOP
    ±16× and factorial reference lines.
  - B: median κ-free profile LR over σ* ∈ [0.05, 0.99], relative to the unrestricted optimum, for on-path, v = 0.3 and
    IsoFLOP.
- `m6_montecarlo_profile_ci_v2`: appendix Figure C1.
  - A: median bounds of the 95% κ-free set against v.
  - B: shares of sets that are the whole grid, contain [0.50, 0.95], contain all σ* ≥ 0.9, or contain all σ* ≤ 0.5.
- `m6_montecarlo_industry_bias`: regenerated with v2 Design B. It is not used in Appendix C v2; the table carries the
  numbers.
- The v1 files `m6_montecarlo_fig2_designs` and `m6_montecarlo_profile_ci` are left in place (v1 content).

**Tables** (`output/tables/`), overwritten with v2 content:
- `m6_montecarlo_designA.tex`, `_designA_summary.csv`, `_designA_profile.csv`, `_designA_diagnostics.csv`,
  `_designA_bootcheck.csv`.
- `m6_montecarlo_industry*.csv|.tex` (v2 Design B).

**New tables:**
- `m6_montecarlo_designA_v2_vs_v1.csv`: every summary, profile and bootcheck statistic, v1 vs v2.
- `m6_montecarlo_designA_startcheck.csv`: 9 vs 64 starts.
- `m6_montecarlo_industry_v2_vs_v1.csv`.
- `m6_montecarlo_industry_startcheck.csv` and `_startcheck2.csv`.

**Paper tables** (`paper/tables/`, generated by `code/paper/make_appendix_c_tables_v2.py`):

| File | Table | Content |
|---|---|---|
| `appC_designs.tex` | C1 | Design A: bias [RMSE]; corners, Wald, bootstrap, median bounds of the κ-free set, flat shares |
| `appC_starts.tex` | C2 | v1 vs 9 vs 64 starts |
| `appC_noise.tex` | C3 | Noise level, digitization, heteroskedastic/clustered |
| `appC_industry.tex` | C4 | Design B v2 |
| `appC_transmission.tex` | C5 | Formula check; label `prop:A-transmission` |

**Processed data** (`data/processed/m6_montecarlo/`):
- `designA_reps_v2.parquet` (12,000 rows = 24 cells × 500);
- `designA_bootcheck_v2.parquet`;
- `designB_reps_v2.parquet`;
- `designA_startcheck.parquet`, `designB_startcheck.parquet`, `designB_startcheck2.parquet`;
- logs.

**Code changes:**
- `mc_lib.py`: STARTS without the truth; `het_cluster_noise`; wider κ boxes; `th_from_path_dual`.
- `design_a.py`: v2 cells, grid, diagnostics, MC s.e.
- `design_b.py`: `start_grid`, `ls_multi`; no `start_norm` in the estimators.
- `outputs.py`: `fig2_v2`, `fig_ci_width_v2`, `v1v2_compare`, `make_design_a_v2`, `make_design_b_v2`.
- `run_v2.py`: `--design-b`, `--no-a`.
- New scripts: `designA_startcheck.py`, `designB_startcheck.py`, `designB_startcheck2.py`.
- `run.py` is now documented as the v1 entry point. Running it would write v2-estimator results under v1 file names.

## 6. Checks performed

- **Replay of simulated data.** The objective at the truth in `designA_startcheck` equals run_v2's in all 4,000
  replications (max |Δ| = 0). The 9-start refit reproduces run_v2's σ̂* exactly (max |Δ| = 0).
- **v1 vs v2 on identical data.** Merged 9,000 replications. Primal estimates differ only in the on-path, v ≤ 0.1 and
  Kaplan cells (share of |Δσ̂*| > 10⁻⁴: 62%, 60%, 19%, 1.0% and 0.2%). Dual estimates are identical everywhere.
- **Every number** quoted in `paper/sections/appendix_mc.tex` was checked against the CSVs above.
- **Tables.** Paper tables compile in `code/paper/test_section.sh appendix_mc` with no errors and no overfull boxes;
  the only undefined references point to other sections. Pages were rendered and inspected.

## 7. Open issues and caveats

1. **The near-path primal numbers depend on the optimizer's start set** (Table C2). The paper reports the 9-start
   baseline and the 64-start column. Hoffmann et al. used a 4,500-point grid. Our conclusions hold under every set.
2. **The heteroskedasticity specification is one choice.** Within an IsoFLOP budget, s.d. ∝ (ND)^(−1/4) is
   homoskedastic. A size-only rule (∝ N^(−1/4)) would add within-budget heteroskedasticity; it is not run.
3. **The m9 seed s.d.** will place the experiment within the noise range of Table C3. There is a
   `[TBD-m9: seed s.d.]` placeholder in Appendix C (R1 minor 16).
4. **Design B** remains an illustrative calibration. Only the target-rule allocation and ln M* numbers moved
   materially. The multiple local optima under the target rule are themselves a finding for the pooled estimators.
5. **Design B memo.** The v1 memo's Design B statements (H7–H9) remain valid, with §3's updates. Its "+0.119" becomes
   "+0.118" at three decimals (unchanged value 0.1185).
