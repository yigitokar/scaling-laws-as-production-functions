# Memo — module m6_montecarlo: Monte Carlo evidence on identification and IO estimators

Module owner: m6_montecarlo (Claude). Date: 2026-09-23. Entry point: `code/analysis/m6_montecarlo/run.py`.
Everything here is **simulated**; nothing is estimated from real data except the noise calibration (§2.1).
The truth is **assumed** (Besiroglu et al. 2024 Chinchilla parameters, from `sl.BESIROGLU`):
E = 1.8172, A = 482.01, B = 2085.43, α = 0.3478, β = 0.3658, hence
a = 0.5126, γ = 0.1783, σ* = 0.7370, ln M*(10^24) = 2.898 (M* = 18.1 tokens/parameter). These match the SYNTHESIS §2.1 ledger (a = 0.513, γ = 0.1783, σ* = 0.737; M* 18.4 at 5.76e23 and 16.5 at 3.8e25).
All bias numbers are Monte Carlo means over replications; "MC s.e." = sd/√R of the estimates. RMSE is around the truth.

**Revision note (independent review, 2026-09-23; details in `output/memos/m6_montecarlo_review.md`).** The module was re-run from scratch after these fixes:
- the κ-free profile LR is now measured against the unrestricted minimum;
- the σ* grid is refined next to the truth;
- the IV row reports medians, because 2SLS has no finite moments;
- a warm-start vs multi-start bootstrap check was added;
- a decomposition estimator for ML practice (E1b) was added.

All non-profile Design A numbers and all Design B numbers for the original estimators are unchanged (bit-identical). The profile-likelihood numbers (H2, claim 3), the on-path bootstrap interpretation (H4, claim 5), the IV numbers and several statements were corrected below.

---------------------------------------------------------------------------------------------------

## 1. Headline findings

**H1 — "The better labs optimize, the less their data reveal" (Design A, Figure 2A).** Every design has 90 runs and the same total training compute, Σ6ND = 5.1×10^22 FLOP. The designs differ only in how runs are allocated along their isocost lines. Primal estimator = Hoffmann/Besiroglu Huber-LSE with κ = 1 imposed. 500 replications per design. RMSE of σ̂* by design:
  - **On the expansion path** (s = allocation-error sd in ln(D/N)):
    - s = 0: RMSE **0.282**. 51% of fits land on a corner, with one power-law term switched off.
    - s = 0.02 (design (i)): RMSE **0.185**. Median |error| is 0.037, and the 5–95% range of σ̂* is [0.34, 0.77].
  - **Optimizing labs with larger errors:**

    | s | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 | 1 | 2 |
    |---|---|---|---|---|---|---|---|
    | RMSE of σ̂* | 0.067 | 0.025 | 0.015 | 0.013 | 0.012 | 0.012 | 0.0065 |

  - **Designed experiments:** IsoFLOP ±16× **0.0036**, factorial grid **0.0050**, IsoFLOP ±4× 0.0091.

  Without imposing optimality, the MRTS level M\* is far more demanding than σ*. (Under optimality M\* is trivially identified on the path: it is the observed D/N, and the dual estimator's RMSE of ln M̂\* is 0.009; see H3.) RMSE of the primal ln M̂\*(10^24):

  | Design | s = 0 | s = 0.02 | s = 0.1 | s = 0.3 | s = 1 | s = 2 | IsoFLOP ±16× | factorial |
  |---|---|---|---|---|---|---|---|---|
  | RMSE of ln M̂\* | 24.0 | 16.9 | 4.49 | 1.97 | 0.59 | 0.32 | **0.16** | 0.28 |

  At s = 0.3 the 5–95% range of ln M̂\* is [−0.40, 6.34], so M̂\* ranges over a factor of about 850 (truth 18.1). (File: `output/tables/m6_montecarlo_designA_summary.csv`.)

**H2 — With κ free, σ* is not identified by near-optimal allocations (s ≤ 0.3 at this noise level and design size; Figure 2B).** Profile likelihood over σ* in L = E + (A N^−a₁ + B D^−b₁)^κ with κ free (Kaplan-type outer exponent). 150 replications per design, 18-point σ* grid on [0.50, 0.95]. LR is measured against the unrestricted κ-free minimum (review fix; the builder's denominator understated LR by 0.001–0.22 on average, and the flat shares below moved by 0–4 pp).
  - **Flat profiles.** The 95% profile CI is the *entire* grid [0.50, 0.95] in:
    - 88% of on-path (s = 0.02) replications;
    - 97% / 91% / 93% / 89% / 65% of replications at s = 0 / 0.05 / 0.1 / 0.2 / 0.3;
    - 77% of Kaplan-belief replications.
    - In these cells the unrestricted κ-free optimum lies *outside* [0.50, 0.95] in 17–53% of replications, which is itself a symptom of a flat likelihood.
  - **Dependence on the noise level.** At s = 0.3 the flat share is 38% / 65% / 90% at noise sd 0.005 / 0.0075 / 0.015. What counts as "near-optimal" is s relative to the noise (and to the number of runs, 90 here).
  - **Informative profiles.**
    - s = 0.5: the CI still reaches a grid edge in 68% of replications (median width 0.32).
    - s = 1: median width 0.104 (median CI [0.69, 0.79]).
    - s = 2: median width 0.028.
    - Designed experiments: IsoFLOP ±16× **0.016**, factorial 0.019, IsoFLOP ±4× 0.047. These agree with the κ = 1 Wald and bootstrap widths (0.015–0.016 IsoFLOP, 0.019 factorial). The builder's 0.011/0.016 were interpolation artefacts of a grid coarser than the CI.
    - Coverage of the truth is 93–100% in every cell (IsoFLOP ±16× 94.0%, factorial 94.7%).
  - **Interpretation.** For 0.1 ≤ s ≤ 0.3, the κ = 1 primal estimator's apparently good σ* precision (RMSE 0.013–0.025) is functional-form information.
    - With κ = 1, α + β = γ/(a(1−a)), and a(1−a) is flat at a ≈ 1/2, so σ* = 2/(2 + γ/(a(1−a))) ≈ 2/(2 + 4γ) is pinned down by the on-path frontier elasticity γ.
    - This is why the dual estimator, which uses only the path (a, γ), is equally precise (RMSE 0.011–0.012).
    - Relaxing κ removes this information.
    - At s ≥ 1 the data carry genuine curvature information: the κ-free CI is informative, although about 2.4× wider than the κ = 1 Wald CI at s = 1 (0.104 vs 0.044).

**H3 — The dual (Approach-1/2) estimator nails σ* on the path, but only through κ = 1 plus optimality, and it is badly biased when labs optimize against wrong beliefs.** The dual estimator takes the slope a from the optima, γ from the frontier L*(C), then α = γ/a and β = γ/(1−a).
  - On-path RMSEs: a 0.0005, σ* 0.011, ln M* 0.009.
  - Kaplan-belief labs (N ∝ C^0.73) are the same estimator with a systematic allocation tilt. Bias in a is +0.217, in σ* −0.088, and in ln M* −3.92 (M* understated about 50-fold). The primal estimator on the same data is roughly unbiased but weakly identified: σ* RMSE 0.101, and the Wald CI is computable in only 77% of replications. (Table `m6_montecarlo_designA.tex`, Panel B.)

**H4 — Inference.** Nominal 95% CIs for σ* (Wald = LAD-sandwich + delta method, 500 reps; pairs-bootstrap percentile, 149 draws × 150 reps):

| Design | Wald coverage | Bootstrap coverage (warm start) | Median CI width (Wald / boot) |
|---|---|---|---|
| IsoFLOP ±16× | 96.0% | 96.7% | 0.015 / 0.016 |
| Factorial | 94.6% | 96.7% | 0.019 / 0.019 |
| Optimizing, s = 0.3 | 93.2% | 96.0% | 0.051 / 0.055 |
| Optimizing, s = 1 | 95.4% | 95.3% | 0.044 / 0.047 |
| On-path, s = 0.02 | 100%, but computable in only 65% of reps | **67%** (for σ*; 55% for a) | 0.76 / 0.31 |

On the path, the Wald covariance is numerically singular or the fit sits at a corner in 35% of replications.

**Review check: the on-path bootstrap undercoverage is largely an implementation artefact; the substantive failure is uninformativeness.** The main bootstrap warm-starts every draw at the replication's own estimate. `m6_montecarlo_designA_bootcheck.csv` compares, on 60 fresh on-path replications (49 draws each, same resamples), a bootstrap that also restarts every draw from the estimator's nine fixed starting values.
- Coverage for σ* rises from 57% to 92% (a: 50% → 90%; ln M*: 58% → 90%).
- But the intervals become nearly uninformative: median width 0.43 for σ* (vs 0.016 under IsoFLOP), 0.84 for a (the parameter lives in (0, 1)), and 45 for ln M*.
- At s = 0.3 the two bootstraps are identical: 100% (σ*) and 98% (a) coverage, width 0.053.

The correct statement is therefore: on the path, a carefully multi-started bootstrap is roughly honest but says almost nothing, and the warm-start bootstrap used in practice is far too narrow.

Condition numbers of J′J (A, B normalized at geometric means; one design draw at the truth):

| Design | Normalized cond(J′J) |
|---|---|
| IsoFLOP ±16× | 1.4×10³ |
| Factorial | 1.4×10³ |
| s = 1 | 6.0×10³ |
| s = 0.3 | 7.4×10⁴ |
| s = 0.1 | 6.6×10⁵ |
| s = 0.02 | 1.5×10⁷ |
| s = 0 | 4.4×10¹⁶ (singular) |

**H5 — Transmission-bias formulas verified (model_spec Prop. 2 / SYNTHESIS P5).** With n = 10^6 draws, the simulated OLS slope of y = −ln(L−E) on ln C matches the closed form γ + π₁Var(ω)/(π₁²Var(ω)+Var(η)) to within 4×10⁻⁴ in all 12 cases:
  - exogenous: 0.1781 vs 0.1783;
  - funding λ = 0.5 / 1 / 2 / 4: 0.2095 / 0.2370 / 0.2782 / 0.3034 vs 0.2091 / 0.2371 / 0.2783 / 0.3033;
  - predetermined λ = 1 / 2 / 4: 0.2254 / 0.2582 / 0.2783 vs 0.2253 / 0.2583 / 0.2783;
  - target with common target: 0.0000 vs 0; with target dispersion 0.25: 0.0892 vs 0.0891.

  The same holds with lifetime-optimal (T > 0) allocations (n = 3×10^5; |simulated − formula| ≤ 8×10⁻⁴ with simulation s.e. up to 5×10⁻⁴). The formula is exact for the exogenous, funding and predetermined rules. Under the target rule a small extra term −Cov(c, Δ)/Var(c) of −0.0009 to −0.0019 appears, where Δ is the loss from over-training, because high-T labs buy more compute to hit the same target.
  - **Finite-sample OLS CI coverage.** With n = 240 lab-generations, one model each, the 95% OLS CI covers the true γ in 95% of samples under the exogenous rule, 53% under funding λ = 0.5, ≤ 4% under funding λ ≥ 1, 16% under predetermined λ = 1, and 0% under target rules. *These coverage rates are calibration-specific:* they assume sd(η) = 1 for the ω-independent part of ln C (and sd(ω) = 0.25). More exogenous compute dispersion (real cross-lab compute spans orders of magnitude) shrinks the bias term λVar(ω)/Var(c) and raises coverage.
  - **TFP dispersion is understated exactly as predicted.** sd of the OLS residual is 0.229 vs a truth of 0.255 under funding λ = 2, and 0.050 under a common target, i.e. only the seed noise survives.
  - **Qualifier to P5.** Forward and reverse regressions bracket γ only when Cov(c, ω) ≤ 0; they do bracket it under exogenous budgets and under target rules with dispersed targets. With a common target, Cov(c, y) = 0 and the reverse regression is undefined (simulated value −31). Under funding or predetermined budgets *both* exceed γ. m7_theory gives the exact condition: the bracket holds iff −(1+V_ε/V_ω)/γ ≤ π₁ ≤ 0.

  (File: `output/tables/m6_montecarlo_transmission.tex`, `m6_montecarlo_transmission_check.csv`.)

**H6 — Selection (Prop. 3) verified.** Exogenous c, runs released only if y exceeds an absolute threshold. Slope of y on ln C:
  - 25% of runs dropped: 0.112; 50% dropped: 0.082 (−54%); 75% dropped: 0.057.
  - E[ω | c, released] has slope −0.093 when 50% are dropped.

  Combined with a funding rule (λ = 2), the net bias moves from +0.100 with no selection to −0.006 with 50% dropped and −0.046 with 75% dropped. This confirms the "sign ambiguous" statement. (`m6_montecarlo_selection_check.csv`.)

**H7 — Simulated AI industry (Design B; 400 replications per scenario; 40 labs × 6 generations × 1–4 models; ≈ 600 models per replication).** Bias of γ̂ (truth 0.178). MC s.e. of the mean bias is ≤ 0.001 for the NLS/FE/ACF/system rows, 0.001–0.002 for the pooled Huber-LSE. The IV row reports the **median** bias and the median absolute error (MAE), because just-identified 2SLS has no finite moments (under the target rule one draw of γ̂ is 2.7; the mean bias there is +0.026 but the median is −0.002).

| Estimator | (a) exogenous | (b) funding | (c) target | (d) predetermined | flagships only, funding |
|---|---|---|---|---|---|
| Pooled NLS + generation FE | −0.001 | **+0.023** (+13%) | **−0.051** (−29%) | +0.018 (+10%) | **+0.119** (+66%) |
| Lab FE | 0.000 | +0.008 | −0.021 | +0.004 | +0.113 |
| Lab × generation FE (within-family) | 0.000 | 0.000 | 0.000 | 0.000 | not identified |
| ACF GMM, c treated as predetermined | 0.000 | +0.007 | **−0.026** | 0.000 | not identified |
| ACF GMM, lagged lab-level instruments (no current c) | −0.001 | −0.001 | −0.002 | −0.001 | not identified |
| IV (compute price) + expansion path, median bias (MAE) | −0.002 (0.031) | +0.005 (0.031) | −0.002 (0.027) | −0.002 (0.028) | 0.000 (0.027) |
| Pooled Huber-LSE, E free (ML practice) | +0.018 | +0.027 | −0.004 | +0.025 | +0.117 |

  - **Within-family (lab × generation) variation is the observational analog of a designed experiment.**
    - It is unbiased for α, β, a, γ and σ* under every compute rule. RMSE is 0.002–0.004 for γ and 0.025–0.034 for a. Siblings share ω, and tier and inference-demand variation is exogenous.
    - It needs ≥ 2 released models per lab-generation.
    - Like the other fixed-effects estimators it is hurt by measurement error in D: with ME sd 0.15, ln M* bias is −0.33 (lab FE −0.28, Mundlak −0.25; the shifts relative to no ME are −0.31, −0.35 and −0.32).
  - **ACF with the correct timing** (no current lab-level compute among the instruments) removes the transmission bias under all four rules: |bias γ| ≤ 0.002 and |bias TFP growth| ≤ 0.0032.
    - Treating current compute as predetermined when labs actually respond to current ω (target rule) leaves a −0.026 bias.
    - ACF is imprecise for the input split: RMSE(a) is 0.12–0.13, against 0.026 for within-family.
    - *Caveat (review):* both ACF variants also use the **current within-lab-generation deviations** of n and d (with squares and product) as instruments. They are valid here by construction, because tier offsets and inference demand are independent of ω. Without them the lagged-instrument moment set has 8 moments for 10 parameters and is under-identified. So the ACF results borrow the same within-family variation as E5. They are not evidence that ACF works on flagship-only data.
  - **Transmission bias loads on γ, not on the allocation exponent.** The allocation regression of ln N on ln C has |bias(a)| ≤ 0.003 in all scenarios, with RMSE 0.004–0.005 (0.009 with flagships only).
  - **ML practice absorbs over half of algorithmic progress into the fitted scaling law.** Under the exogenous, funding and predetermined rules, the pooled Huber-LSE with E free and no time effects estimates TFP growth with bias −0.081 to −0.090 per generation (truth 0.150; −0.036 under the target rule). It also understates sd(ω) by 0.03–0.05 (truth 0.25) and overstates ln M* by +0.56 to +0.68 (≈ 0 under the target rule). Ê averages 1.756–1.762 under those three rules (1.783 under the target rule) vs a truth of 1.817. MC s.e. of the ln M* bias is about 0.07.
    - *Decomposition (review addition, estimator E1b = E1 with E held at the truth).* Even with E known, the pooled fit without time effects misses 36–43% of TFP growth (bias −0.054 / −0.065 / −0.062 under exogenous / funding / predetermined rules; MC s.e. 0.0005). It does so by steepening the frontier: γ̂ bias +0.050 / +0.059 / +0.057, i.e. +28–33%. This is the Sahal channel. Estimating E jointly lowers Ê by about 0.06, which pulls γ̂ back toward the truth (+0.018 to +0.027) but removes a further 0.025 per generation from measured progress. Roughly two-thirds of the absorbed progress is attributed to scale and one-third to the Ê level. The ln M* bias (+0.55 to +0.61 with E known) comes from the pooled design, not from E.
  - **TFP growth, pooled NLS with generation FE:** bias −0.024 (funding), +0.054 (target, +36%), −0.018 (predetermined), −0.130 (flagships only under funding, i.e. 86% of progress missed).

**H8 — The FOC ("GNR/system") estimator is contaminated by unobserved inference demand exactly as the wedge algebra predicts.** Stacking the within-family loss equation with the training-only FOC αu = βv (i.e. imposing w = 1):
  - The exponents stay essentially unbiased under the four main rules: |bias| ≤ 0.009 for α and β, and ≤ 0.005 for a, γ and σ*.
  - **ln M\* is biased by +1.24 to +1.28 in every scenario** (+1.25 under exogenous, funding and predetermined rules, +1.28 under the target rule; M* overstated about 3.5-fold; MC s.e. 0.002).
  - The closed form is bias(ln M̂*) = 2E[ln w]/(α+β) = 2 × 0.457/0.714 = **1.28**.

  The IV + expansion-path estimator, which reads M* off the observed path, has the same contamination (+1.28). Modeling T through a noisy usage proxy (log sd 0.3) removes it: bias −0.02 to −0.03, RMSE 0.05. The FOC residual *is* the inference wedge: ignoring it makes over-trained models look like evidence that data are cheap.

**H9 — Selection in the industry (≈ 18% of trained models unreleased; absolute "notability" threshold).**
  - Pooled γ̂ is biased by −0.008 (−4.5%) under the exogenous rule, and TFP growth by +0.011 (+7%).
  - The Heckman two-step correction (exclusion: an open-weight release policy) halves both, to −0.003 and +0.006.
  - Under the predetermined rule, selection (−) and transmission (+) partially offset: pooled +0.011. Heckman removes only the selection part and moves the estimate to +0.018. Correcting one IO problem can therefore raise measured bias.
  - The IV estimator becomes invalid under selection: median γ bias −0.022 (exogenous + selection) and −0.033 (predetermined + selection); mean biases −0.027 and −0.038.

---------------------------------------------------------------------------------------------------

## 2. Methods

### 2.1 Calibration (estimated from raw data)
- **Log-loss noise sd = 0.0075.** This is the residual sd of ln L around the Besiroglu parameters on Epoch's digitized Chinchilla sample: n = 240 after dropping the 5 highest-loss points; sd 0.0076, MAD-based sd 0.0049, i.e. heavy tails. It is computed in `run.py::calibrate` → `data/processed/m6_montecarlo/noise_calibration.csv`. It bundles seed noise, digitization error and misspecification, so it is an upper-end value for pure seed noise.
- Refit check: `fit_huber` on the same data returns α = 0.3473, β = 0.3672, E = 1.8172. This matches `sl.fit_chinchilla` and reproduces Besiroglu et al. up to their rounding.

### 2.2 Design A (experimental designs; `design_a.py`, `mc_lib.py`)
- **Budgets.** Chinchilla's nine IsoFLOP budgets (6e18, 1e19, 3e19, 6e19, 1e20, 3e20, 6e20, 1e21, 3e21) × 10 runs. The total, 5.106×10^22 FLOP, is identical in every design.
- **Designs.** The spread of the designs is summarized by the sd of ln(D/N) given c.
  - (i) On-path: s = 0.02.
  - (ii) IsoFLOP: 10 sizes log-spaced over [N*/16, 16N*] (M from 0.08 to 6,288; sd of ln(D/N) given c is 3.5), and ±4× as robustness.
  - (iii) Factorial: 9 N × 10 D grid, each spanning 256×, centered on the expansion path. corr(ln N, ln D) = 0, and compute is matched by scaling the centre.
  - (iv) Optimizing labs: same 90 budgets, ln(D/N) = optimum + s·z at fixed C, s ∈ {0, .05, .1, .2, .3, .5, 1, 2}.
  - (v) Kaplan-belief labs: N ∝ C^0.73, pivoting at the geometric-mean budget, plus s = 0.1. The tilt is systematic, not random: the rms distance from the true path is 0.87 in ln(D/N).
  - Robustness cells: noise sd 0.005 and 0.015 (IsoFLOP ±16× and s = 0.3), and Epoch-style digitization (L rounded to 0.012 nats, 2% pixel error in N and C, D imputed as C/6N).
- **Noise.** Multiplicative, ln L = ln L_true + N(0, 0.0075²). This is the ZKD case: noise is realized after inputs are chosen.
- **Primal estimator.** Exactly the Hoffmann/Besiroglu Huber(δ = 10⁻³) objective on the LSE parameterization (`sl._obj`, `sl._grad` reused read-only), minimized by L-BFGS-B from 9 starts (the truth plus 8 fixed dispersed starts). One deviation from `sl.fit_chinchilla`: a box α, β ∈ [10⁻³, 5], ln A, ln B ∈ [−10, 60], ln E ∈ [−3, 2.5]. It never binds off the path, but it stops divergence to A or B → 0 on on-path data. "Corner" = estimate on the box boundary, or one power-law term below 2% of reducible loss at every design point.
- **Dual estimator.**
  - On-path/optimizing designs: a = OLS slope of ln N on ln C over the (noisy) optima, and ln M*(10^24) from the same regression.
  - IsoFLOP designs (Approach 2): a quadratic in ln N at each budget, then the argmins and minimum losses.
  - γ comes from a 3-parameter Huber fit of the inverse cost function L*(C) = E + K(C/6)^−γ. Then α = γ/a and β = γ/(1−a).
- **Inference.**
  - Wald: V = (J′J)⁻¹/(4 f̂(0)²). With δ = 10⁻³ ≪ noise sd, the Huber estimator is LAD; f̂(0) is a Gaussian-kernel density of residuals at 0 with Silverman bandwidth. Delta method for derived objects. The Wald CI is flagged as not computable if cond(J′J) > 10¹² or the fit is a corner.
  - Pairs bootstrap (149 draws, warm start at the replication's estimate) in the first 150 replications of five cells.
  - Bootstrap-start check (review addition, `design_a.run_bootcheck_chunk`): on 60 fresh replications of the on-path and s = 0.3 cells, 49 pairs-bootstrap draws per replication, each fitted twice, once warm-started only and once also restarted from the estimator's nine fixed starting values. Output `m6_montecarlo_designA_bootcheck.csv`.
- **κ-generalized profile.** Fixed S = a₁ + b₁ on an 18-point σ* grid in [0.50, 0.95] (spacing 0.004–0.005 next to the truth). The other five parameters are optimized by L-BFGS-B, with Gaussian NLS on ln L and LR = n ln(SSR(σ*)/SSR_min).
  - SSR_min is the *unrestricted* minimum over (σ*, κ) and the other parameters, found by an extra L-BFGS-B fit with S free, started from the best and second-best grid solutions and from the κ = 1 fit. (Review fix: the original code used min(grid SSR, κ = 1 SSR), an upper bound that understated LR, and had only 14 grid points with spacing 0.013–0.017 around the truth, which made IsoFLOP/factorial CI widths interpolation artefacts. See `m6_montecarlo_review.md`.)
  - Starting values: the *analytically observationally equivalent* κ-model parameters (b₁ = aS, a₁ = (1−a)S, κ = γ/(a(1−a)S), K′ = K^{1/κ}, same G; this reproduces on-path loss to 10⁻¹⁵), plus the neighbouring grid solution.
  - Without the equivalence start, profiles on on-path data spuriously rose to LR ≈ 20–40 at σ* = 0.95 (an optimizer failure). CI endpoints are linear interpolations of LR.
- **Diagnostics** (`m6_montecarlo_designA_diagnostics.csv`):
  - cond(J′J) at the truth, raw and KMW-normalized. Raw cond is for the (ln A, ln B, ln E, α, β) LSE parameterization and is not comparable with Czech et al.'s 3.5×10^11 in levels; cond(J) = √cond(J′J).
  - Transverse sd of t = αn − βd given c (Lemma 2), sd of ln(D/N) given c, and rms distance from the true path.

### 2.3 Design B (simulated industry; `design_b.py`)
- **Technology and productivity.**
  - Technology (model_spec eq. T): y = ω_ft + F(n, d + ψ_D) − ε, with ε ~ N(0, 0.05²) in y units (on the frontier d ln L/dy = −R/L, so this is ≈ 0.011 in ln L at 10^21 and ≈ 0.005 at 10^23 FLOP). L = E + e^{−y}.
  - ω_ft = δ_t + x_ft, x_ft = 0.8 x_{f,t−1} + ξ_ft, stationary sd 0.25.
    - This sd is **assumed**, chosen to match Mertens et al.'s 41× p90/p10 in effective compute: γ ln 41/2.563 = 0.26 (SYNTHESIS §2.2 converts it to 1.78–1.94× in reducible-loss units).
    - δ_t = 0.15 (t−1), i.e. Hicks-neutral TFP growth of 0.15 per generation (assumed).
- **Compute.**
  - Lab-generation budget c_ft = c̄_t + h_f − p_ft + ν_ft, with c̄ growing ln 3 per generation from 10^21. h_f (sd 0.7) is lab size; p_ft (sd 0.4) is an observed compute-price shifter; ν_ft has sd 0.4.
  - Rules:
    - (a) exogenous;
    - (b) + 2 x_ft (funding responds to current productivity);
    - (c) the lab buys exactly the compute that reaches its target loss (target = what an average lab would reach with its nominal budget, ± sd 0.1; the target also responds to p);
    - (d) + 2 x_{f,t−1} (provisioned a generation ahead: ACF timing).
  - Tiers m = 1..M_ft, with M_ft ~ U{1..4}, use c − Δ_m, where Δ_m ≈ (m−1) ln 8 + N(0, 0.3²).
- **Allocation.** Lifetime-compute optimum (Sardana et al.): αu/(βv) = 1 + T/(3D) at the given c, solved by monotone Newton (h is increasing and concave in d).
  - T = 3D*(c)θ with ln θ ~ N(0, 0.8²); mean ln w = 0.457.
  - Plus an allocation error of sd 0.2 in ln(D/N).
  - Hicks-neutral ω does not enter the allocation (Lemma 1, verified to 10⁻¹⁴).
- **Variants.**
  - Selection: released iff y ≥ min(previous same-tier y + 0.5, generation median of y) − 0.25·(open-weight lab). All generation-1 models are released; overall ≈ 81–82% released.
  - Measurement error: classical ME in ln D (sd 0.15), with reported C = 6 N D_obs.
  - Factor bias: data-augmenting lab heterogeneity ψ_D,f (sd 0.3), reflected in the labs' allocation.
  - Flagships only: one model per lab-generation.
- **E.** It is treated as known (= truth) by all estimators except E1. This isolates IO identification from the E-level problem, and E1 shows what the E-level problem adds.
- **Estimators** (released models only; KMW-normalized parameterization; `scipy.optimize.least_squares`):
  - E1 pooled Huber-LSE on ln L (9 starts).
  - E1b (review addition): E1 with E fixed at the truth, still pooled with no time effects. It is a decomposition device that separates the scale-attribution channel from the Ê channel; it appears only in the appendix tables.
  - E2 pooled NLS in y with generation effects.
  - E3 lab FE (within-lab demeaning inside NLS).
  - E4 Mundlak (lab means of n, d).
  - E5 lab × generation FE. It is followed by TFP residuals y − F̂ for growth and dispersion.
  - E6 ACF/Wooldridge-type GMM.
    - Quasi-differenced ξ_ftm = (y − F − δ_t) − ρ(mean_{f,t−1}(y − F) − δ_{t−1}).
    - Instruments: generation dummies; within-lab-generation deviations of n and d, with squares and product; the lab's lagged mean n and d; the lab's twice-lagged mean y; and (E6 only) the current lab-generation mean c.
    - Level normalization E[y − F | t = 1] = 0. One-step NL2SLS weighting.
  - E7 IV-path:
    - â from the pooled OLS of n on c;
    - γ̂ from 2SLS of y on c with generation effects, instrument p (median first-stage F = 17–18; 13 under the target rule; 11 under selection; 40 for flagships). Just-identified 2SLS has no finite moments, so for this estimator tables and figures report median bias and median absolute error;
    - α = γ/a, β = γ/(1−a).
  - E8 Heckman two-step:
    - probit of release on the inputs of all trained generation ≥ 2 models, generation dummies and the open-weight policy (the exclusion restriction);
    - IMR added to E2. This assumes the inputs of unreleased models are observed, e.g. from compute registries.
  - E9 system: E5's within-family loss equation stacked with the training-only FOC ln(αA/βB) − αn + βd = 0. The residual variances are standardized at the E5 solution.
  - E10 = E9 with ln(1 + T_proxy/3D) subtracted in the FOC; T_proxy = T·exp(N(0, 0.3²)).
- **Reported objects.**
  - α, β, a, γ, σ* and ln M*(10^24).
  - sd(ω): the sd of lab-generation mean TFP residuals y − F̂ after removing generation means (estimand = population 0.25).
  - TFP growth: the mean per-generation change in generation means of y − F̂ (estimand 0.15). The same definition is used for every estimator, including E1, which has no time effects.

### 2.4 Verification (`verify_bias.py`)
- Prop. 2 / P5: n = 10^6 draws per case, on the path and with lifetime-optimal allocation. Finite-sample behavior: 2,000 samples of n = 240. Reverse regression γ_rev = Var(y)/Cov(c, y).
- Prop. 3: absolute thresholds dropping 0/25/50/75% of runs.
- Identities: Lemma 1, d* − n* = −2 ln G + (b − a)(c − ln 6) − 2aψ_D; Prop. 4, w = 1 + T/(3D) and T = 3D(w − 1).

### 2.5 Reproducibility and runtime
- Every (cell, chunk) task seeds its own generator from `SeedSequence([seed, cell, chunk])`. Results do not depend on the number of processes: two chunks recomputed after the run matched the saved replications to 0.0.
- Independent re-run (review): after all module outputs were deleted, `run.py` regenerated every file. All 40 non-profile Design A columns and all 44,000 original Design B rows are bit-identical to the builder's run (max abs difference 0.0), as are the verification and calibration CSVs.
- Full run after the review fixes: 44.8 minutes wall-clock with 6 processes on a heavily shared machine (load average 35–54 on 18 cores): Design A 27 min (the κ-free profile now has 18 grid points plus a free-S fit), Design B 9 min, bootstrap-start check 7 min, verification and outputs 1.5 min. The builder's run took 24.7 min at load 25–33. CPU only; no GPU/MLX.
- `run.py --skip-sim` rebuilds all tables and figures from the saved replications in about 10 s. `run.py --quick` is a smoke test that writes only to `data/processed/m6_montecarlo/quick/`.

---------------------------------------------------------------------------------------------------

## 3. Table and figure inventory

Tables (`output/tables/`):

| File | Caption |
|---|---|
| `m6_montecarlo_designA.tex` | **Paper table (designs).** Panel A: primal bias [RMSE] for α, β, a, γ, σ* and ln M* by design. Panel B: dual-estimator RMSE. Panel C: cond(J′J), corner share, Wald/bootstrap coverage for σ* (§ marks the warm-start bootstrap; the multi-start check is in the notes), κ-free profile-CI width and flat share. |
| `m6_montecarlo_designA_summary.csv` | All cells × {primal, dual} × parameters: mean, median, bias, MC s.e., sd, RMSE, median abs error, 5/95 percentiles, Wald/bootstrap coverage and widths, corner share. |
| `m6_montecarlo_designA_profile.csv` | κ-free profile likelihood by cell: coverage, flat share, edge share, empty-CI share, median CI and width, mean and median LR on the grid, and (review) the LR understatement of the original denominator, the flat share under it, and the share of replications whose unrestricted σ̂* lies off the grid. |
| `m6_montecarlo_designA_bootcheck.csv` | (Review addition) Warm-start vs multi-start pairs-bootstrap coverage and median width for all six objects, on-path and s = 0.3 cells, 60 fresh replications × 49 draws. |
| `m6_montecarlo_designA_diagnostics.csv` | Design geometry: M range, transverse sd, sd(ln D/N given c), rms distance from the path, raw and normalized condition numbers. |
| `m6_montecarlo_industry.tex` | **Paper table (industry Monte Carlo).** Bias and RMSE of γ, a, TFP growth and ln M*(10^24) for 9 estimators × 4 compute rules (IV row: median bias and MAE, ‡). |
| `m6_montecarlo_industry_full.tex` | Appendix: all 12 estimators (incl. the E1b decomposition); γ, a, σ*, TFP growth, sd(ω) and ln M*. The IV row reports median bias and MAE. |
| `m6_montecarlo_industry_variants.tex` | Appendix: selection (two rules), ME in D, ψ_D heterogeneity, flagships only. |
| `m6_montecarlo_industry_exponents.tex` | Appendix: α, β and σ* by estimator and rule. |
| `m6_montecarlo_industry_summary.csv` | All scenario × estimator × parameter statistics, including MC s.e. |
| `m6_montecarlo_industry_meta.csv` | Release rates, corr(c, ω), mean ln w, median first-stage F, mean IMR coefficient. |
| `m6_montecarlo_industry_failures.csv` | Share of failed or not-identified fits (0 everywhere except the within-family estimators with flagships only, which are not identified by design). |
| `m6_montecarlo_transmission.tex` / `_transmission_check.csv` | Prop. 2 closed form vs simulation (on path and T > 0), finite-sample mean and coverage, reverse regression, residual-sd check. |
| `m6_montecarlo_selection_check.csv` | Prop. 3 slopes under absolute-threshold selection. |
| `m6_montecarlo_identities_check.csv` | Lemma 1 and Prop. 4 identities, maximum absolute error ≈ 10⁻¹⁴. |

Figures (`output/figures/`, PDF and PNG):

| File | Caption |
|---|---|
| `m6_montecarlo_fig2_designs` | **Paper Figure 2.** (A) RMSE(σ̂*) vs allocation-error sd s for optimizing labs (primal and dual), with IsoFLOP and factorial reference lines. (B) median κ-free profile LR over σ* for on-path, s = 0.3, IsoFLOP and factorial designs. (C) RMSE of ln M̂*(10^24) vs s. (D) sampling distributions of σ̂* by design. |
| `m6_montecarlo_profile_ci` | Appendix: median κ-free profile-CI width and share of flat profiles vs s. |
| `m6_montecarlo_industry_bias` | Bias of γ̂, TFP growth and ln M̂* by estimator (rows) and compute rule (markers); median bias for the IV row. |
| `m6_montecarlo_transmission_check` | (A) simulated OLS slope vs the Prop. 2 closed form (filled: on path; hollow: T > 0). (B) slope among released runs vs the share dropped by an absolute threshold. |

Processed data (`data/processed/m6_montecarlo/`):
- `designA_reps.parquet`: 9,000 replications (18 cells × 500); includes 18 profile-LR columns plus `prof_shift` and `prof_sig_hat` for the first 150 replications of each cell.
- `designA_bootcheck.parquet`: 120 replications of the bootstrap-start check.
- `designB_reps.parquet`: 48,000 estimator-replication rows (10 scenarios × 400 × 12 estimators).
- `verify_*.csv`.
- `noise_calibration.csv`.

---------------------------------------------------------------------------------------------------

## 4. Claims for the paper

1. **On-path data do not identify the technology through the primal estimator. The failure is a ridge plus boundary solutions, not just imprecision.** With every run exactly compute-optimal:
   - the Huber-LSE fit ends on a corner (one power-law term switched off) in 51% of replications;
   - RMSE(σ̂*) is 0.28 and RMSE(ln M̂*) is 24.0.

   *Evidence:* `designA_summary.csv` (opt_s0, onpath), `designA.tex` Panel A/C. *Caveats:* the corner share depends on the parameter box and the multistart set; RMSEs of α, β and ln M* on the path are box-dependent, while σ* ∈ (0, 1) is not. One of the nine starting values is the truth, which favours near-truth solutions on a flat ridge, so on-path performance is if anything overstated (the claim is conservative). The underlying non-identification is analytic: on the path, one exponential in c is reproduced by the truth *and* by either term alone. Report medians and MAE next to RMSE: for σ* at s = 0 the median bias is −0.297 and the MAE 0.297 (the typical fit is a corner); at s = 0.02 they are −0.030 and 0.037 (the typical fit is fine, but the tails are extreme).

2. **"The better labs optimize, the less their data reveal" (Figure 2A/C).** For optimizing labs:
   - RMSE(σ̂*) falls from 0.28 at s = 0 to 0.067, 0.025 and 0.015 at s = 0.05, 0.1 and 0.2;
   - RMSE(ln M̂*) falls from 24 to 4.5 (s = 0.1), 2.0 (s = 0.3) and 0.59 (s = 1);
   - a Chinchilla-style IsoFLOP ±16× design at the same compute gives 0.0036 and 0.16.

   *Evidence:* `designA_summary.csv`, `fig2_designs`. *Caveat:* 90 runs and noise sd 0.0075; RMSE scales roughly linearly with noise (IsoFLOP σ*: 0.0024 / 0.0036 / 0.0078 at noise sd 0.005 / 0.0075 / 0.015).

3. **Without the κ = 1 functional form, near-optimal allocations reveal nothing about σ\*.** In the Kaplan-nesting model with κ free, the 95% profile CI for σ* is the whole range [0.50, 0.95]:
   - in 88–97% of replications for s ≤ 0.2 (on-path 88%), 65% at s = 0.3, and 77% for Kaplan-belief labs;
   - IsoFLOP and factorial designs give CIs of width 0.016–0.019, and IsoFLOP ±4× 0.047;
   - at s = 0.5 the CI still reaches a grid edge in 68% of replications; at s ≥ 1 it is informative (median width 0.104 at s = 1).

   For 0.1 ≤ s ≤ 0.3 the σ* precision of the κ = 1 primal estimator is functional-form information. With κ = 1, σ* = 2/(2 + γ/(a(1−a))) ≈ 2/(2+4γ) is pinned down by the on-path frontier elasticity γ, which is why the dual estimator (path only) is equally precise. This is the numerical counterpart of model_spec Prop. 1 and SYNTHESIS P2. *Evidence:* `designA_profile.csv`, `fig2_designs` panel B, `profile_ci`. *Caveats:* the profile is grid-based over [0.50, 0.95] (18 points; the unrestricted optimum lies off-grid in 17–53% of near-path replications). The flat share depends on s relative to the noise (38% / 65% / 90% at noise sd 0.005 / 0.0075 / 0.015 for s = 0.3) and on the design size (90 runs), so a threshold in s must always be quoted with the design.

4. **What on-path data *do* deliver is identification by optimality plus κ = 1, i.e. the dual (Approach 1/2) route.** It is precise on the path: RMSE of a 0.0005, σ* 0.011, ln M* 0.009. It fails when the optimality assumption fails systematically: under Kaplan-belief allocations, bias in a is +0.22, in σ* −0.09, and in ln M* −3.9. *Evidence:* `designA.tex` Panel B. *Caveat:* allocation errors here are mean-zero in ln(D/N) at every C (except the Kaplan cell); any C-dependent belief error biases the dual estimator.

5. **On the path, standard inference is either wrong or empty.**
   - The Wald covariance is singular or at a corner in 35% of on-path replications.
   - The warm-started pairs bootstrap (the usual implementation) covers σ* in only 67% of replications (a: 55%, ln M*: 66%).
   - A bootstrap that restarts every draw from the full start set covers 92% (a: 90%), but its intervals are nearly uninformative: median width 0.43 for σ* and 0.84 for a (60 fresh replications, 49 draws).
   - With designed variation all methods agree and are accurate: IsoFLOP 96% / 97%, factorial 95% / 97%, s = 1 95% / 95% (Wald / bootstrap); warm and multi-start bootstraps coincide at s = 0.3.

   *Evidence:* `designA.tex` Panel C, `designA_bootcheck.csv`. *Caveat:* the main bootstrap uses 149 draws × 150 replications (MC s.e. of a coverage rate ≈ 1.8 pp at 95%). The bootstrap-start check uses 60 replications (MC s.e. ≈ 3–6 pp). Do not cite "67% coverage" without the warm-start qualifier.

6. **The Prop. 2 transmission-bias formula is exact, and its sign follows the compute rule**:
   - exogenous budget: γ;
   - funding rule: γ + λVar(ω)/Var(c);
   - common capability target: 0;
   - predetermined budget: γ + λρVar(ω)/Var(c).

   Simulated slopes match the closed form within 4×10⁻⁴ (`transmission.tex`). OLS CIs from 240 lab-generations almost never cover γ under a funding rule with λ ≥ 1 (≤ 4%; 16% for a predetermined rule with λ = 1; 0% under target rules). *Caveats:* under a target rule with heterogeneous inference demand, an additional term −Cov(c, Δ)/Var(c) of about −0.002 appears. The n = 240 coverage rates assume sd(η) = 1 for the ω-independent part of ln C; with more exogenous compute dispersion the bias and the undercoverage shrink.

7. **In a realistic multi-model industry, transmission bias is diluted but not removed by common fixes.** Pooled NLS with generation FE:

   | Compute rule | Bias in γ | Bias in TFP growth |
   |---|---|---|
   | Funding | +0.023 (+13%) | −0.024 |
   | Target | −0.051 (−29%) | +0.054 |
   | Predetermined | +0.018 | −0.018 |
   | Flagships only, funding | +0.119 | −0.130 |

   The flagship-only number is the Prop. 2 formula applied to the industry DGP: with generation effects, Cov(c, ω)/Var(c) = 2·0.0625/(0.49 + 0.16 + 0.16 + 4·0.0625) = 0.118 (h, p, ν, 2x), against a simulated +0.119 (reviewer check). Lab FE removes 64% of the bias under funding and 80% under predetermined budgets, but not under the target rule (−0.021), and almost nothing with flagships only (+0.113, because ω is AR(1), not fixed, and the lab effects also absorb exogenous lab-size variation in compute). *Evidence:* `industry.tex`, `industry_variants.tex`. *Caveat:* magnitudes scale with λ = 2 and sd(ω) = 0.25 (assumed); the multi-model structure (1–4 models, ~8× compute steps) dilutes the bias relative to the one-model-per-lab formula.

8. **Within-family (lab × generation) estimation is the observational analog of an IsoFLOP experiment.** It is unbiased under every compute rule (|bias γ| ≤ 0.001, RMSE ≤ 0.004; |bias a| ≤ 0.002), and TFP growth is recovered from its residuals (bias ≤ 0.006). *Caveat:*
   - it requires ≥ 2 released siblings with exogenous within-family allocation;
   - it is not available for flagship-only data;
   - like lab FE and Mundlak, it is hurt by measurement error in D (ln M* bias −0.33 at ME sd 0.15, −0.37 when combined with selection and funding; lab FE −0.28, Mundlak −0.25).

9. **ACF-type GMM works when the timing assumption is right and fails when it is wrong.** Without current lab-level compute among the instruments, |bias γ| ≤ 0.002 and |bias TFP growth| ≤ 0.0032 under all four rules. Treating current compute as predetermined when labs respond to current ω biases γ̂ by −0.026 under the target rule and +0.007 under funding. *Caveats:* (i) ACF is 5× less precise for the input split than within-family (RMSE(a) 0.12–0.13 vs 0.026), because Hicks-neutral ω leaves no proxy (the OP/LP inversion is unavailable; SYNTHESIS §4.C). (ii) Both ACF variants use current within-lab-generation input deviations as instruments; without them the lagged moment set is under-identified (8 moments, 10 parameters). The ACF success therefore rests on the same exogenous within-family variation as claim 8 and says nothing about flagship-only panels.

10. **Transmission bias loads on the frontier elasticity γ, not on the allocation exponent a.** The allocation regression ln N on ln C has |bias| ≤ 0.003 under all ten scenarios, including selection and ME (RMSE 0.0035–0.0048; 0.0085 with flagships only). *Evidence:* `industry.tex` Panel B, E7 row. *Caveat:* this holds for Hicks-neutral ω. Factor-biased heterogeneity (ψ_D) tilts allocations but, being mean-zero here, did not bias the average a.

11. **FOC/"GNR" system estimation mistakes inference demand for technology.** Imposing w = 1 on over-trained models biases ln M* by +1.24 to +1.28 (M* overstated about 3.5×), matching the closed form 2E[ln w]/(α+β) = 1.28. The exponents stay essentially unbiased. A noisy usage proxy for T removes the bias (−0.02, RMSE 0.05). This is the link to the wedge (model_spec Prop. 4): the FOC residual is ln w. *Evidence:* `industry.tex` Panel D, E9 vs E10. *Caveat:* the proxy is simulated with log noise sd 0.3; real usage data (downloads, OpenRouter tokens) are much noisier and may be biased.

12. **ML practice (pooled Huber-LSE, E free, no time effects) absorbs more than half of algorithmic progress into the fitted scaling law, about two-thirds of it through a steeper frontier (scale) and one-third through a downward-biased Ê.** Under the exogenous, funding and predetermined rules:
    - TFP growth bias is −0.081 to −0.090 of a true 0.150 per generation;
    - γ bias is +0.018 to +0.027 (+0.117 with flagships only);
    - sd(ω) is understated by 13–21%;
    - Ê is biased down by 0.055–0.061 (0.034 under the target rule).

    With E held at the truth (E1b), the same pooled regression still misses 36–43% of progress (TFP growth bias −0.054 to −0.065) and overstates γ by +0.050 to +0.059. That is the scale-attribution (Sahal) channel on its own.

    Under the target rule these biases partly cancel (TFP growth −0.036, γ −0.004). This is Sahal-type bias [nagy2013statistical]. *Caveat:* this depends on compute growth being correlated with calendar time, which is true by construction here (c̄ grows ln 3 per generation).

13. **Selection on release biases returns to compute downward and progress upward, and corrections interact.** Exogenous rule + notability selection (18% unreleased): γ −0.008 and TFP growth +0.011; Heckman halves both. Predetermined rule + selection: the two biases offset (+0.011), and Heckman *raises* the bias to +0.018 by removing only the selection part (Prop. 3 "sign ambiguous", numerically). *Caveat:* Heckman relies on observing unreleased models' inputs and on an excluded release-policy shifter; neither is typically available.

---------------------------------------------------------------------------------------------------

## 5. Robustness and failures

- **Profile-likelihood denominator and grid (fixed in review).** The builder's LR used min(grid SSR, κ = 1 SSR) as SSR_min, an upper bound on the unrestricted κ-free minimum. That understated LR by 0.001–0.22 on average (90th percentile up to 0.6) and inflated flat shares by 0–4 pp. The 14-point grid was also coarser (0.013–0.017) than the IsoFLOP/factorial CIs, which made their widths (0.011/0.016) interpolation artefacts; the correct values are 0.016/0.019. Both are fixed (free-S fit; 18-point grid), and `designA_profile.csv` reports the old-denominator flat share for comparison.
- **On-path bootstrap starts (review check).** Warm-start bootstrap draws understate the spread on the ridge (coverage 57–67%). Restarting each draw from the full start set gives 92% coverage but median σ* width 0.43. See H4.
- **Profile-likelihood optimizer failure (fixed).** Naive starting values for the κ-free profile produced spurious LR of 20–40 at σ* = 0.95 on exactly on-path data. The analytic observational-equivalence start fixed it. Other modules profiling κ on real data (e.g. m1_chinchilla's `profile_sigma`) should use the same start, otherwise edge LRs can be optimizer artifacts.
- **Box constraints.** The primal estimator uses bounds (§2.2). With `sl.fit_chinchilla` unbounded, on-path fits drift to A or B → 0 or ∞; this is not a bug in `sl.py`, but on-path RMSEs of α, β and ln M* are not meaningful without bounds. **No bug found in sl.py.** Note that a warm start at the published Besiroglu values converges to A = 477.8, B = 2143 (objective marginally lower), i.e. the published values are rounded.
- **RMSE(σ̂*) vs s is not monotone-smooth.** It plateaus at ≈ 0.012 for s ∈ [0.3, 1] and drops to 0.0065 at s = 2. The plateau equals the dual estimator's error (Claim 3). The reason is analytic: with κ = 1, α + β = γ/(a(1−a)), and a(1−a) is flat at a ≈ 1/2 (0.2498 at the truth), so σ* = 2/(2 + γ/(a(1−a))) ≈ 2/(2 + 4γ) is pinned down by the on-path frontier elasticity γ alone. Even at s = 0.3, where the primal RMSE of a is 0.098, σ̂* has RMSE 0.013. So read the κ-free profile (H2), not the κ = 1 RMSE, as the measure of the information in the data.
- **Dual-estimator RMSE rises with s.** It goes from 0.012 at s ≤ 0.5 to 0.028 at s = 2, because noisy optima degrade the path regression. For M*: RMSE 0.48 at s = 1 and 0.91 at s = 2.
- **Wald CIs under near-singularity.** In the near-path cells (s ≤ 0.1 and on-path) coverage is 99–100% among computable replications because widths explode; computable shares are 11% (s = 0), 86% (s = 0.05), 93% (s = 0.1) and 65% (on-path). Kaplan-belief labs: 92% coverage among the 77% computable. Do not cite those coverage numbers as "correct inference".
- **Digitization.** Epoch-style digitization barely matters for an IsoFLOP design: σ* RMSE 0.0041 vs 0.0036, Wald coverage 93.6% (slight undercoverage).
- **IV (compute price).** First-stage F has a median of 17–18 (11 under selection; 13 under the target rule). Just-identified 2SLS has no finite moments, so means and RMSEs are dominated by rare extreme draws (target rule: max γ̂ error 2.54, mean bias +0.026, RMSE 0.18, but median bias −0.002 and MAE 0.027). Median absolute error of γ̂ is 0.027–0.031 (15–17% of γ) without selection. The instrument is valid but weak-ish; with selection, 2SLS becomes biased (median −0.022 to −0.033; mean −0.027 to −0.040).
- **ACF imprecision.** Only one-step NL2SLS with a fixed instrument set was tried; no optimal weighting or instrument-selection search. Its imprecision for a and M* may be partly implementation-specific.
- **Small but statistically non-zero biases.** Examples: E9/E10 γ −0.002/−0.001 (MC s.e. 0.0001); E10 ln M* −0.025 (s.e. 0.002; Jensen-type bias from the noisy T proxy); ACF-lag TFP growth +0.0025 (s.e. 0.0005). All are economically negligible.
- **Negative result.** Factor-biased heterogeneity ψ_D (sd 0.3, mean zero across labs) did *not* materially bias any average technology parameter (within-family ln M* −0.03; FOC system with T proxied −0.01). The wedge contamination in model_spec Prop. 4 (ŵ = w·exp(αψ_N − βψ_D)) is a model-level error of sd β × 0.3 ≈ 0.11 in ln w; this Monte Carlo did not evaluate it at the model level.
- **Runtime.** 44.8 minutes after the review additions, on a machine with load average 35–54 on 18 cores. This is at the edge of the 45-minute budget; unloaded, expect roughly half. `--skip-sim` rebuilds all outputs in about 10 s.

---------------------------------------------------------------------------------------------------

## 6. Open issues

1. **The industry calibration is illustrative, not estimated.** sd(ω) = 0.25 comes from the Mertens 41× figure (SYNTHESIS §2.2, suggestive only); λ = 2, g = 0.15, sd(ln θ) = 0.8, tier structure and selection margins are assumptions. Once m4_observational or m5 produces empirical moments (e.g. corr(c, TFP residual), within-family compute spreads, release shares), recalibrate and rerun with `run.py` (about 25 min).
2. **E known.** E is known in all industry estimators except E1. A module estimating E from sweeps should propagate its uncertainty. E1 shows that estimating E jointly in observational data biases Ê down by 0.03–0.06 (0.055–0.061 under the exogenous, funding and predetermined rules) and inflates other biases.
3. **Rate results.** A formal statement of convergence rates on or near the path was not pursued. The identified set on the path contains boundary rays, so standard asymptotics do not apply. The singular-information literature ([rotnitzky2000likelihood], added to `lit/bib/extra_m6_montecarlo.bib`) is the relevant reference if the paper wants a rate statement for small s.
4. **Missing designs.** Hierarchical-bootstrap inference with seed replicates, and designs mixing a few off-path runs into an on-path ladder, were not run. The latter answers "what fraction of runs must be off-ray?" (SYNTHESIS §5.12) and is cheap to add by editing `design_a.cells()`.
5. **Heckman assumptions.** The Heckman design assumes unreleased models' inputs are observed; a truncation-based (Lee-bounds) alternative would be more realistic for Epoch-type data.
6. **Citations.** Keys used, all in `lit/references.bib`: besiroglu2024chinchilla, hoffmann2022training, kaplan2020scaling, sardana2024beyond, mertens2026secret, marschak1944random, zellner1966specification, mundlak1961empirical, hoch1962estimation, olley1996dynamics, levinsohn2003estimating, ackerberg2015identification, wooldridge2009estimating, gandhi2020identification, klump2007factor, leonledesma2010identifying, diamond1978measurement, demirer2020production, kricheli2026tokens, czech2026problems, whitfill2025note, nagy2013statistical, andrews2012estimation. New keys (verified via Crossref) in `lit/bib/extra_m6_montecarlo.bib`: heckman1979sample, rotnitzky2000likelihood.
