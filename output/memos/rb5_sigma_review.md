# Independent review: module rb5_sigma (the sigma* re-analysis bundle of round 3)

Reviewer: Claude (independent replicator and skeptical referee; package WP2-review). Date: 2026-09-25.
Scope: `code/analysis/rb5_sigma/`, `output/tables/rb5_sigma_*`, `data/processed/rb5_sigma/`, `output/memos/rb5_sigma.md`, the figure code of T10 and T16 (previews), and the fills the builder proposes for the WP2 writer and for other packages. Specification: fix list `paper/notes/round3_fixlist.md`, item T1 (1 to 10) and the analysis part of T14.
Rules followed: CPU only under `nice -n 10`, at most two worker processes per job, no MLX or training, no git commit or push, no file in `paper/sections/` or `paper/tables/` edited, nothing from the experiment (m9) read. Changes to the module are marked `[review]` in code and memo.

---

## 1. Verdict

**Accept with revisions; the revisions are made here.** Every T1 number the builder proposes reproduces, both in a from-scratch re-run and in an independent re-implementation. One analysis (T14) was wrong and is corrected. Several proposed fills change or gain a qualifier (section 8).

**What holds.**
- **Re-run from scratch.** The whole pipeline, run into a scratch output root with no stage caches (47.6 minutes, 2 processes), reproduces all 32 CSV files byte for byte (`output/tables/rb5_sigma_review_rerun.csv`; log `data/processed/rb5_sigma/review_rerun_log.txt`).
- **Independent re-implementation.** `code/analysis/rb5_sigma/review_independent.py` imports none of the module's code, nor rb1's, ra1's or rb4's. It reads their published CSVs and re-implements every estimator: inverse-variance and DerSimonian-Laird pooling with the modified HKSJ interval, a three-level REML meta-regression, CR2 with Bell-McCaffrey degrees of freedom, the restricted wild cluster bootstrap-t by full enumeration of the Webb weights, and the linear statistics of the joint draws. **159 of 159 comparisons agree** within tolerance (`output/tables/rb5_sigma_review_independent.csv`). They cover T1.1, T1.2 (all nine specification-by-cluster rows, the subsamples and the pooled line), T1.3, T1.4, T1.5, T1.6 and T1.7.
- **T1.9 and T1.10.** The reviewer recomputed the decomposition from rb1's and ra1's evaluation points with separate code (`code/analysis/rb5_sigma/review_convexity.py`, `output/tables/rb5_sigma_review_convexity.csv`). Every point estimate the memo quotes is reproduced: b1, b2, the local first-derivative sigma*, and the convex part by bin (Farseer 0.169, Marin pooled 0.427, Llama 3 -0.134 and -0.006).
- **Specification.** Every T1 sub-item is delivered as the fix list specifies, with two interpretations the review accepts:
  - "five clusters" read as the IsoFLOP-only sample with Marin's corpora separate;
  - "compute-weighted centre" read literally as weights proportional to C.

**What does not hold as written.**
1. **T14 is wrong (corrected).** On the ray of final checkpoints the builder's fit is a local optimum.
   - For NLS and Huber alike, a second optimum has a lower objective. It swaps which power term decays fast: alpha 0.15 and beta 1.06 to 1.23, against alpha 0.86 to 0.94 and beta 0.09.
   - In that mode the recipes' tilts are ranked in the reverse order (Spearman -0.75 and -0.82 against the all-checkpoint tilts, against +0.75).
   - So the tilt is not identified on the ray. D4 must say so, and the final-checkpoint factor must not be marked on the breakdown frontier (section 2).
2. **T1.4's interval is Farseer's.** With the between-study s.d. at zero, Farseer's first-derivative entry carries 84 percent of the weight in 0.708 [0.690, 0.726], and its s.e. excludes smoothing bias. With Farseer's bandwidth range as uncertainty (the T1.6 treatment) the row is 0.701 [0.665, 0.738] (section 3).
3. **T1.8 answers the fix list's letter, not R1's point.** R1 minor 7 compares with the experiment's within-trunk sharing, where one trunk is one architecture shared across budgets. The public-design analogue is noise shared by model size across budgets, not by budget. The reviewer's Monte Carlo of that case (section 4) finds that the level interval still covers 90 to 97 percent of the time, but the within-design drift test rejects a true zero drift 14 to 19 percent of the time when a smooth size distortion is shared across budgets.
4. **Drift s.e.: 0.014, with a precise label.** The builder's 0.013 scales the joint-bootstrap s.e. by a dispersion factor estimated as if the budgets were independent. Done coherently it is 0.0136 to 0.0141 (section 5).
5. **Smaller corrections** (section 6):
   - the CR2 interval at 10^19 must not be quoted;
   - 0.598, not 0.599, and the memo's attribution of R4's number is wrong;
   - the Marin leave-corner-out convexity is a composition effect;
   - the "percentile" label of T1.1 is loose;
   - 7 of 51 plotted percentile intervals in Figure 5 still exclude their estimate;
   - independent-noise coverage at R = 500 is 93 to 95 percent, not 92 to 98;
   - two stale numbers in the fix list's T12(m) text (Wald 13.3 and s.e. 0.018 are pre-integration values).

---

## 2. T14: the DataDecide tilt is not identified on the final-checkpoint ray

**Finding.** `code/analysis/rb5_sigma/review_ddtilt.py` refits the common-exponent panel model with its own code (scipy least squares from 40 starts). Only m2's data loader is used. For both estimators it finds a lower objective than the builder's. m2's own `fit_panel_ls`, started in the second mode, confirms both optima to seven digits.

| Final checkpoints (950 runs) | Mode | Objective | alpha | beta | Tilt range | M* factor, own exponents | Spearman with all-checkpoint tilts |
|---|---|---|---|---|---|---|---|
| NLS | alpha > beta (builder's) | 0.0099262 | 0.859 | 0.094 | 0.698 | 4.33 | +0.75 |
| NLS | beta > alpha (**lower objective**) | 0.0098277 | 0.146 | 1.055 | 0.781 | 3.67 | -0.75 |
| Huber | alpha > beta (builder's) | 0.0026963 | 0.936 | 0.095 | 0.565 | 2.99 | +0.76 |
| Huber | beta > alpha (**lower objective**) | 0.0025923 | 0.155 | 1.228 | 0.496 | 2.05 | -0.82 |

**Why.** On one ray D is about 100 N, so both power terms are, to first order, power laws in N. The data then identify two exponents but not which input each belongs to. The recipe tilt ln(A_r/B_r) depends on that labelling, and its sign pattern flips between the modes. M varies from 85 to 110, so the modes are not exact mirror images; that is why their objectives differ slightly.

**What the builder's version got wrong.**
- The reported fit is the local optimum reached from m2's all-checkpoint start.
- The bootstrap started every draw in that mode only, so its intervals (factor 1.5 to 9.7) were conditional on the mode.
- The profile grid stopped at beta = 0.20 and never saw the other mode.

**Fix [review]** (`ddtilt.py`; stage re-run):
- every fit starts in both modes, and both local optima are reported (column `mode`; `global_optimum` flags the lower objective);
- the bootstrap starts each draw in both modes and records the share of draws that land in each;
- the profile grid covers both modes.

Re-run results (`rb5_sigma_dd_tilt.csv`):
- The point estimates are the two optima above, to seven digits.
- In the cluster bootstrap (B = 199), 60 percent of the NLS draws and 77 percent of the Huber draws land in the beta > alpha mode.
- Over draws from both modes the tilt range is [0.48, 1.32] (NLS) and [0.26, 1.67] (Huber). The own-exponent factor on M* is [1.9, 11.6] and [1.5, 6.8]; the reference-exponent factor is [3.0, 22.3] and [1.8, 49.7].
- In the Gaussian profile the builder's optimum lies 8.7 above the global one. The two basins are separated by a high barrier: the profile is at least 276 above the minimum for alpha between 0.25 and 0.55 (`rb5_sigma_dd_profile.csv`). A local solver started in one basin stays there, which is why the builder's fits and bootstrap never reached the other.

**For the paper (T12(r), D4).** Use this in place of the builder's recommendation: "On final checkpoints only, one ray at M of about 100, the recipes' tilt is not identified: the fit has two optima of almost equal fit that exchange the roles of the two power terms and reverse the ranking of the recipes' tilts, so we do not report a final-checkpoint allowance. The sign result's breakdown frontier (Online Appendix Figure E) does not depend on DataDecide." This is the branch T14 anticipates ("If the tilt is not identified on the ray ... say so in D4 and rely on the breakdown frontier (S3)").

**For WP4a (S3).** Do not mark "3.0 to 4.3 (bootstrap 1.5 to 9.7)" on the breakdown frontier. The all-checkpoint 3.4 and 1.84 are unaffected: they come from m2's fit on all 21,888 checkpoints, which spans many rays.

**Unaffected.** The builder's note for WP3 P6 (the all-checkpoint tilts convert to 1.82 with Huber and 1.68 with NLS under the reference exponents) is arithmetic on m2's published values and is correct.

---

## 3. T1.4: the budget-restricted study-level mean

The builder's construction is right and reproduces exactly: 0.708 [0.690, 0.726], Q = 1.02. The design means are DerSimonian-Laird on S_b with robust bootstrap variances, recomputed by the reviewer from the joint draws.

The heterogeneity that limits Farseer's weight in the all-budget mean (between-study s.d. 0.023) disappears below 3x10^20. The random-effects weights are then inverse-variance weights. Farseer's first-derivative entry (0.710, s.e. 0.006) gets **84 percent**, against 6, 5 and 5 percent for Hoffmann, Meta and Marin. The interval is therefore Farseer's sampling interval. R1 minor 3 already objects to that s.e. because it excludes smoothing bias.

| Version (`rb5_sigma_study_level.csv`, group T1.4) | Mean [95% CI] | Farseer's weight |
|---|---|---|
| Builder's primary (Farseer first derivative, s.e. 0.006) | 0.708 [0.690, 0.726] | 84% |
| **[review] Farseer's bandwidth range as uniform uncertainty (s.e. 0.019)** | **0.701 [0.665, 0.738]** | 35% |
| Same, range read as plus or minus 1.96 s.e. (reviewer's check) | 0.702 [0.667, 0.737] | 40% |
| Farseer's Hessian path, random effects over its levels up to 3e20 | 0.702 [0.668, 0.736] | 44% |
| IsoFLOP studies only | 0.697 [0.636, 0.758] | 0% |
| Budgets 10^19 to 3x10^20 only (reviewer's check) | 0.708 | |

"About 0.7 at 10^19 to 3x10^20 FLOP" holds under every version (0.697 to 0.708).
- **Table 1, panel C (T11(a)):** print 0.708 [0.690, 0.726] and add in the note "0.701 [0.665, 0.738] with Farseer's bandwidth range as uncertainty".
- **M9-6:** compare the experiment's estimate with the bandwidth-robust interval, [0.665, 0.738]. The narrow interval understates the public designs' uncertainty and would favour a finding of difference.

---

## 4. T1.8: coverage when noise is shared

**The builder's Monte Carlo reproduces exactly** in the re-run: ra1's R = 100 rows, the R = 500 rows and every dependence cell. It shows that a per-budget shock leaves the random-effects interval's coverage at 93 to 94 percent.

That is the fix list's design ("a per-budget random effect"). R1 minor 7, however, compares with the experiment's coverage study, where half the variance is shared within a **trunk**. There a trunk is one architecture trained once and branched across token budgets (m9 memo, section 2: "one trunk per architecture"). The public-design analogue is noise shared by **model size across budgets**. The per-budget design is the transpose of that: a common shift within a budget leaves each profile's curvature unchanged.

In all three MC designs sizes recur across budgets: 90 to 98 percent of runs belong to a size, matched within 3 percent in N, that appears at two or more budgets. `code/analysis/rb5_sigma/review_depmc.py` therefore adds two designs. It uses ra1's estimator and bootstrap unchanged, the builder's kappa-family truths, rho = 0.5 and R = 200 (`output/tables/rb5_sigma_review_mc_size.csv`):
- **per-size effect:** one shock per model size, shared across budgets;
- **smooth size distortion:** one Gaussian-process function of ln N, correlation length one log point, added to every run of the design.

| Design (truth sigma*) | Per budget (builder, R = 500) | Per budget (reviewer's seeds, R = 200) | **Per model size across budgets** | **Smooth size distortion, all budgets** | Smooth within budget (builder) |
|---|---|---|---|---|---|
| Chinchilla, total parameters (0.667) | 0.932 | 0.930 | **0.915** | **0.905** | 0.895 |
| Llama 3 (0.693) | 0.938 | 0.950 | **0.950** | **0.930** | 0.925 |
| Marin DCLM (0.663) | 0.942 | 0.935 | **0.965** | **0.915** | 0.895 |
| Drift test's rejection of a true zero drift (5 percent level) | 0.07 to 0.08 | 0.04 to 0.14 | 0.05 to 0.07 | **0.14 to 0.19** | 0.06 to 0.15 |

Monte Carlo s.e. of a coverage near 0.93 is 0.011 at R = 500 and 0.018 at R = 200. Bias of the random-effects mean is at most 0.008 in absolute value in every cell.

**Reading.**
- **The level interval survives every sharing structure: 90 to 97 percent coverage.** The experiment's 0.40 to 0.70 does not carry over. Its model-free estimator is a local-quadratic surface on an 8 x 6 grid with trunk-clustered resampling (m9 memo, section 2). Here budget-level IsoFLOP curvatures are pooled with a between-budget random effect, and a size shock enters every budget's profile as within-profile noise.
- **The within-design drift test does not survive.** When half the noise is one smooth function of model size shared across budgets (for example a digitization or architecture-quality distortion), a true zero drift is rejected 14 to 19 percent of the time at the 5 percent level. Llama 3's test is already at about 10 percent with independent noise (0.096 at R = 500). The builder's own smooth within-budget cells show 6 to 15 percent, which the memo did not report.
- **Relevance.** The drift at the top budgets rests on Chinchilla and Llama 3, both digitized from figures (R2 NM3(a)). This supports III.B's caution and the "not robust" framing of the decline.

**For the paper.**
- T3(b) and T12(i) can print the fix list's per-budget number, "93 to 94".
- T12(i) should add one sentence: "the interval covers the truth in 90 to 97 percent of replications also when the shared half attaches to model size across budgets or to a smooth distortion in size, but such a distortion makes the within-design drift test reject a true zero drift 14 to 19 percent of the time" (`rb5_sigma_review_mc_size.csv`).
- The independent-noise figure should become 93 to 95 percent (section 6, item 7).

---

## 5. T1.1: the drift's standard error and the top-budget interval

**Drift.** Point estimate, model-based s.e. (0.0126 with the dispersion factor 2.01) and joint-bootstrap s.e. (0.0095; 0.0089 independent) reproduce exactly. The builder recommends 0.013, which is 0.0095 times the square root of 2.01. But 2.01 is the weighted residual variance scaled as if the budgets were independent. Estimated coherently with the joint covariance of the draws:
- the moment estimator for this weighted estimator gives 2.06, and s.e. 0.0136;
- the GLS residual quadratic form gives 2.22, and s.e. 0.0141;
- the GLS slope with the joint covariance is -0.051.

**Use 0.014** and label it "from the joint bootstrap draws, scaled for the budgets' excess dispersion". Plain "bootstrap standard error" would describe 0.009. Table 1 note h then reads "... joint design-conditional draws, scaled for excess dispersion (model-based 0.013)".

**Top-budget mean.** 0.594 with joint s.e. 0.0119. The modified HKSJ interval with the joint variance, [0.561, 0.627], is confirmed. With the joint covariance Cochran's Q is 3.15 on 4 degrees of freedom, so the variance factor is still truncated at one. The within-design correlation is near zero except at Llama 3's two top budgets (0.39), as the memo says.

The memo's "percentile" interval [0.569, 0.615] is re-centred on the estimate. The plain percentile interval is [0.578, 0.623] (delta-method draws; [0.578, 0.625] on the sigma scale directly), centred on the bootstrap population's value 0.602. If D6 prints one, label it.

---

## 6. Smaller points (memo corrected where it is the memo's)

1. **T1.2, CR2 interval at 10^19.** [0.707, 0.719] has an s.e. one eighth of the model-based one: a four-cluster CR2 artifact. Quote only the point value 0.713. The 10^20 interval [0.638, 0.722] is fine and is wider than the model-based one.
2. **T1.2, CR2 construction.** The upper end of the p-value range, 0.49, is the design-FE inverse-variance row. There rb1's CR2 (whitened, symmetric square roots) is exactly unbiased for the slope under the working model, while the Cholesky-based adjustment's expected variance is 8 percent low (p = 0.44). rb1's is the right choice; the reviewer's code reproduces 0.4896.
3. **T1.2, five clusters.** The IsoFLOP-only five-cluster sample gives p from 0.013 to 0.23. "Between 0.03 and 0.49" is the range for the 44-budget regression with four and six clusters. It must not be presented as covering five.
4. **T1.3.** Substituting only the 6e20 budget gives 0.598, as the full re-estimate does. R4's 0.599 is approximate; print 0.598.
5. **T1.9, Marin without corner runs.**
   - The drop from 0.43 to 0.38 is a composition effect. Without each budget's largest-M run the bin holds only points at M of 256 to about 320. On those points the primary estimate's convex part is 0.28.
   - W10 must say that the leave-out ranges stop at M of about 320 and 595.
   - Farseer's 0.169 uses the path range down to the 2x10^18 root, as ra1's published first-derivative estimate (0.708) does. Without that root it is 0.164; both round to 0.16 or 0.17. Keep 0.17 for consistency with 0.708.
6. **Figure 5 (T16; note W13).** Percentile intervals still exclude their estimate in 7 of the 51 plotted parametric bins: Marin pooled at M < 16 and 64 to 256, and Farseer's Chinchilla form at 16 to 1,024. The note should say so, or the figure should show plus or minus 1.96 bootstrap s.e. None of the 14 linear-extrapolation bins is affected.
7. **T1.8, independent noise.** The R = 500 independent-noise rows give 93.6, 93.2 and 95.4 percent. The "92 to 98" in T3(b) and T12(i) comes from ra1's R = 100 run. The writer should print "93 to 95 percent with independent noise and 93 to 94 percent when half the noise variance is shared within a budget (500 replications)", so that the two numbers come from the same experiment.
8. **T12(m), stale numbers.** "Wald 13.3, p = 0.021" and Llama 3's "s.e. 0.018" are rb1's values with Chinchilla in total parameters (`output/sensitivity/rb1_chinchilla_T/...design_slopes.csv`). In the primary N_F convention they are **Wald 13.9, p = 0.016** and **0.017** (`output/tables/rb1_sigmaC_metareg_design_slopes.csv`, sample "budgets <= 3e20").
9. **T1.7, memo aside.** "0.70 is its rounding": 0.708 rounds to 0.71; 0.70 lies inside the interval. The "lower 95 percent limits" rows use normal limits; with t_4 the value at 10^24 is 0.31 rather than 0.32. Neither enters the paper.
10. **Figure 3 (T10).** Panel c meets T10(a). Panel d already has panel b's scale (0.40 to 1.07). The previews are fine to promote with `run.py --stages figures --final-figures`; panel c's numbers are unchanged by this review.

---

## 7. What the review changed

- `code/analysis/rb5_sigma/ddtilt.py`: both modes in fits, bootstrap and profile [review]; stage re-run.
- `code/analysis/rb5_sigma/study.py`: T1.4 bandwidth-robust row and a `weight_share_farseer` column [review]; stages study, pidset and tables re-run. The other rows are unchanged; only the new column is added.
- `code/analysis/rb5_sigma/run.py`: T14 mode diagnostics in the headline file; the determinism check skips the reviewer's `rb5_sigma_review_*` files [review].
- New reviewer scripts: `review_independent.py`, `review_convexity.py`, `review_ddtilt.py`, `review_depmc.py`. New outputs: `output/tables/rb5_sigma_review_{rerun,independent,convexity,mc_size}.csv`.
- `output/memos/rb5_sigma.md`: [review] notes in T1.1, T1.2, T1.3, T1.4, T1.7, T1.8, T1.9 and T14, and in section 4.
- The final figures were not written. `fig3_merged.*` and `rb1_sigmaC_extrap.*` are unchanged. The WP2 writer runs `run.py --stages figures --final-figures` after filling the notes.

---

## 8. Numbers the paper may use

Every number was verified against the CSV named, by the re-run and, where marked (I), by the reviewer's independent code. Rounding is as the paper prints it.

| Item and destination | Number to print | Source (`output/tables/`) | Check |
|---|---|---|---|
| T1.1 drift of Chinchilla and Llama 3, 17 budgets (T4, T12(m)) | -0.058 per decade | `rb5_sigma_joint_linear.csv` | R, I |
| T1.1 its s.e. (T4, T11 note h, T12(m)) | **0.014** "from the joint bootstrap draws, scaled for the budgets' excess dispersion" (0.009 from sampling error alone; model-based 0.013) | `rb5_sigma_review_independent.csv`; `_joint_linear.csv` | I |
| T1.1 top-budget mean, five budgets at 6x10^20 and above (T4, T11) | 0.594 [0.561, 0.627] (modified HKSJ with the joint variance), Q = 2.5, p = 0.65 | `rb5_sigma_headline.csv`; `_top_budget.csv` | R, I |
| T1.3 Llama 3's weight in it | 85 percent (46 at 6x10^20, 38 at 10^21) | `rb5_sigma_joint_linear.csv` | R, I |
| T1.2 pooled slope, three-level RE (T4) | -0.033 per decade; pooled line 0.713 at 10^19, 0.680 at 10^20, 0.647 at 10^21 | `rb5_sigma_metareg_slopes.csv`, `_predictions.csv` | R, I |
| T1.2 p-value range, 44 budgets, four and six clusters, every specification, weight, scale, test and design-FE row (T4, T11 note g) | between 0.03 and 0.49 (Marin as one cluster alone: 0.05 to 0.49) | `rb5_sigma_metareg_prange.csv` | R, I |
| T1.2 primary row with Marin as one cluster (T11 note g) | CR2 p = 0.11 on 2.8 df against wild p = 0.049: the two disagree | `rb5_sigma_metareg_slopes.csv` | R, I |
| T1.2 design-FE rows (D6) | inverse variance -0.021 (CR2 p 0.49, wild p 0.37); unweighted -0.026 (0.17 and 0.051 with Marin as one cluster; 0.065 and 0.073 with six) | `rb5_sigma_metareg_slopes.csv` | R, I |
| T1.2 pooled line at 10^20 with Marin as one cluster (H11, WP8) | 0.680 [0.638, 0.722]; no interval at 10^19 | `rb5_sigma_metareg_predictions.csv` | R, I |
| T1.2 subsamples (T4, D6) | budgets <= 3x10^20: -0.008 (p 0.56 to 0.87); each design's largest budget left out: -0.030 (p 0.19 to 0.29); hinge above 3x10^20: -0.152 (descriptive) | `rb5_sigma_metareg_slopes.csv` | R, I (hinge: R) |
| T1.3 Farseer's local path above 3x10^20 (T2, T4, T11) | 0.722 (0.008) at 5x10^20; 0.658 (0.011) at 10^21 | `rb5_sigma_top_budget.csv` | R, I |
| T1.3 top budgets with Farseer's 10^21 level (T4, T11) | random effects 0.606 [0.558, 0.655], Q = 21.1 (p < 0.001); fixed effect 0.625 | `rb5_sigma_top_budget.csv` | R, I |
| T1.3 top budgets without Chinchilla's five highest-loss runs (T12, D6) | **0.598** [0.569, 0.627] | `rb5_sigma_top_budget.csv` | R, I |
| T1.4 study-level mean, budgets <= 3x10^20 (T11(a); H11; M9-6) | 0.708 [0.690, 0.726], Q = 1.02; with Farseer's bandwidth range as uncertainty **0.701 [0.665, 0.738]** (print both; M9-6 compares with the latter) | `rb5_sigma_study_level.csv` (group T1.4) | R, I |
| T1.5 Cochran's Q over Chinchilla's windows, samples and memberships (T3(a)) | 4.4 to 14.8 | `rb5_sigma_study_ranges.csv` | R, I |
| T1.5 range of the study-level mean over all variants (T3(b), T12(m)) | 0.663 to 0.702; the lowest interval end is 0.596 ("about 0.60") | `rb5_sigma_study_ranges.csv` | R, I |
| T1.5 Chinchilla's counts, conventions, windows, samples (T11(a), T13(d)) | 0.674 to 0.702 | `rb5_sigma_study_ranges.csv` | R (the 26-row core: I) |
| T1.5 137-run window grid (audit check; T13(f)) | 0.685 to 0.702, Q 4.4 to 11.3 | `rb5_sigma_study_ranges.csv` | R |
| T1.6 Farseer's bandwidth range as uncertainty (T11(a)) | 0.680 [0.643, 0.717] | `rb5_sigma_study_level.csv` (group T1.6) | R, I |
| T1.7 lower bound at 10^23 and 10^24 (T5) | 0.48 and 0.42 at 10^21; **0.49 and 0.43** at the compute-weighted centre (1.5x10^21); 0.47 and 0.42 at the precision-weighted centre (8.0x10^20) | `rb5_sigma_pi_set.csv` | R, I |
| T1.7 identified sets (T5, H11, Figure 3c) | at 10^24: [0.42, 0.59] if the decline is real, [0.42, 0.70] drift-agnostic; at 10^23: [0.48, 0.59], [0.48, 0.70] | `rb5_sigma_pi_set.csv` | R, I |
| T1.8 coverage, half the noise shared within a budget (T3(b), T12(i)) | 93 to 94 percent; with independent noise **93 to 95** (both R = 500) | `rb5_sigma_mc_dependence.csv` | R |
| T1.8 other sharing structures (T12(i), new sentence) | 90 to 97 percent coverage; smooth size distortion shared across budgets: drift test rejects a true zero drift 14 to 19 percent of the time | `rb5_sigma_review_mc_size.csv`; `_mc_dependence.csv` | reviewer's MC |
| T1.9 convex part at M of 256 to 1,024 (W10, WP4b; Table E9) | Farseer 0.17; Marin 0.43 (0.38 and 0.46 without corner runs, whose ranges stop at M of about 320 and 595); Llama 3 -0.13 at M of 64 to 256, about zero at its two points near M = 290 (where the kappa-free gap, -0.38, is all rotation) | `rb5_sigma_extrap_decomposition.csv` | R, I (point estimates) |
| T1.9 local first-derivative sigma* (Figure 5, E9) | Farseer 0.708; Marin 0.666 to 0.676; Llama 3 0.629 | `rb5_sigma_extrap_lin.csv` | R, I |
| T1.9 Llama 3's gap at M < 16 (W13) | +0.27 (kappa free), +0.42 (Chinchilla form) | `rb5_sigma_extrap_delta_pct.csv` | R |
| T1.10 within Marin (T3(b), T11(e)) | 0.666 to 0.676 (local first derivative) against 0.700 to 0.713 (IsoFLOP curvature) | `rb5_sigma_marin_estimator_gap.csv` | R, I |
| T12(m) (not T1; stale in the fix list) | Wald 13.9, p = 0.016; Llama 3 -0.050 (s.e. 0.017), in N_F | `rb1_sigmaC_metareg_design_slopes.csv` | read |
| T14, D4 (T12(r)) | no number: "not identified on the ray" (section 2); all-checkpoint 3.4 and 1.84 unchanged | `rb5_sigma_dd_tilt.csv` | R, I |

R: reproduced by the from-scratch re-run (or by the stage re-run after the fixes). I: reproduced by the reviewer's independent code.

**Not for the paper.**
- The CR2 interval at 10^19.
- The final-checkpoint M* factors (3.0 to 4.3; 1.5 to 9.7).
- The builder's "percentile" top-budget interval without its label.
- "0.599".
- "92 to 98 percent" beside the R = 500 numbers.
- Wald 13.3 (p = 0.021) and s.e. 0.018 in the N_F text.
