# Memo: module rb5_sigma, the sigma* re-analysis bundle of round 3

Module owner: rb5_sigma (WP2, analysis step). Date: 2026-09-25. Entry point: `code/analysis/rb5_sigma/run.py`.
Status: builder's version. **An independent review is required before any number below enters the paper** (fix list T1).

> **[review] 2026-09-25 (package WP2-review; `output/memos/rb5_sigma_review.md`).** A from-scratch re-run into a scratch root reproduced all 32 CSV files byte for byte (`rb5_sigma_review_rerun.csv`), and an independent re-implementation of T1.1 to T1.7 (own REML, CR2, wild bootstrap and pooling code, no module code imported) agrees on 159 of 159 statistics (`rb5_sigma_review_independent.csv`). Changes made by the reviewer are marked [review] below. The two substantive ones: **T14 is corrected** (the reported final-checkpoint fit is a local optimum; the lower-objective optimum swaps the two power terms and reverses the recipes' tilt ranking, so the tilt is not identified on the ray), and **T1.4 gains a bandwidth-robust row** (Farseer carries 84 percent of the weight in 0.708 [0.690, 0.726]). The paper-facing list of numbers is section 8 of the review memo; it supersedes section 4 below where they differ.

**Scope.** Fix list `paper/notes/round3_fixlist.md`, item T1 (sub-items 1 to 10), the analysis part of T14 (DataDecide tilt from final checkpoints) and the new numbers behind the figure items T10 (Figure 3, panel c) and T16 (Figure 5). Referee requests: R1 New 2(b), (d), minors 3, 4, 7; R2 NM2 requests 1 and 3, NM3(b), (c) and its "smaller point", minor 17; R4 N3(b) to (f) and requests 2 and 3, minors 5 and 14; audit numbers_technology #4 ("better fix").

**Reproduction.**
- One command regenerates every output: `nice -n 10 .venv/bin/python code/analysis/rb5_sigma/run.py` (stages draws, metareg, convexity, study, pidset, depmc, ddtilt, tables, figures; 41 minutes on 2 CPU processes; no GPU, no MLX).
- Figures are written as previews (`output/figures/rb5_sigma_preview_fig3.*`, `rb5_sigma_preview_fig5.*`). After the review, `run.py --stages figures --final-figures` writes the paper's files `output/figures/fig3_merged.*` and `rb1_sigmaC_extrap.*`.
- Seeds: every reproduction uses the upstream module's own seeds (ra1: `iso|<design>`, `farseer|i`, `mc|<design>`; rb1: `wcr|<spec>`, `extrap|<design>|i`); new draws use `sigcommon.SEED = 20260927`, one stream per key.
- Determinism: `run.py --determinism` re-runs every stage into a scratch root and compares all 32 CSV files byte for byte: 32 of 32 identical (`data/processed/rb5_sigma/determinism_check.csv`).
- Inputs are read-only: ra1_modelfree (estimator, bootstraps, Farseer surface), rb1_sigmaC (meta-regression and local-wedge code, its bootstrap cache), rb4_chinflop (Chinchilla in N_F; override inputs, specification grid, review rows R11 to R13) and m2_techpanel (DataDecide loader and panel fits). None of their code or outputs is changed, except that rb1's own copy of Figure 5 is now saved as `rb1_sigmaC_extrap_rb1version.*`, so that re-running rb1 cannot overwrite the paper's Figure 5 (one line in `rb1_sigmaC/figures_rb1.py`).
- Licence: the Chinchilla and Llama 3 digitizations carry no redistribution licence; this module writes derived statistics only (budget-level estimates and their bootstrap draws).

**Conventions.** sigma* is in FLOP-effective parameters N_F throughout (Chinchilla rebuilt by rb4), as in rb1's primary outputs. "Per decade" is per unit of log10 C. Intervals are 95 percent. S = 2(1/sigma* - 1).

## 1. Findings

### T1.1 Joint-bootstrap standard errors (R2 NM3(b); R4 minor 5)

ra1's design-conditional wild bootstrap was re-run with ra1's seeds and B = 999 on Chinchilla (N_F, 137 runs), Llama 3 and Marin's three corpora; every per-budget sigma*_b, s.e. and interval reproduces the published inputs of rb1 exactly (`rb5_sigma_joint_check.csv`, max |diff| below 1e-15). The whole (999 x k) matrix of draws per design is kept (`rb5_sigma_joint_draws.csv`). Designs use disjoint data, so draws are independent across designs.

Within a design the draws are only weakly correlated (mean off-diagonal correlation between -0.012 and 0.016; `rb5_sigma_joint_corr.csv`), except at Llama 3's two top budgets (0.39), which sit at the edge of the frontier cubic's support.

| Statistic (`rb5_sigma_joint_linear.csv`) | Estimate | Model-based s.e. | Bootstrap s.e., independent | Bootstrap s.e., joint | Joint / independent |
|---|---|---|---|---|---|
| Chinchilla + Llama 3 drift, design FE, inverse variance (17 budgets) | -0.0576 | 0.0126 (scaled by s2 = 2.01; 0.0089 unscaled) | 0.0089 | 0.0095 | 1.07 |
| Same, unweighted | -0.0500 | | 0.0133 | 0.0143 | 1.07 |
| Chinchilla's own drift (9 budgets) | -0.0574 | 0.0204 | 0.0204 | 0.0219 | 1.08 |
| Llama 3's own drift (8 budgets) | -0.0576 | 0.0099 | 0.0099 | 0.0106 | 1.07 |
| Top-budget mean (five budgets >= 6e20) | 0.594 | 0.0103 | 0.0103 | 0.0119 | 1.16 |

- **Drift.** The joint draws raise the sampling s.e. of the Chinchilla-Llama 3 slope by 7 percent, to 0.0095. The published 0.013 is larger because it also scales by the budgets' excess dispersion around the design lines (s2 = 2.01). The joint-bootstrap s.e. scaled the same way is 0.0134. **Recommended: "bootstrap standard error 0.013"** (joint draws, allowing for the excess dispersion), with 0.009 from sampling error alone stated in Online Appendix D. Percentile interval from sampling error alone: [-0.077, -0.040].
  - [review] The dispersion factor 2.01 is estimated as if the budgets were independent. Estimated coherently with the joint covariance it is 2.06 (moment estimator for this weighted estimator) or 2.22 (GLS residuals), and the s.e. becomes 0.0136 or 0.0141. **Use 0.014**, labelled "from the joint bootstrap draws, scaled for the budgets' excess dispersion" (not "bootstrap standard error" alone, which is 0.009). 0.013 is within the imprecision of the dispersion estimate but mixes the two covariance models.
- **Top-budget mean.** The joint s.e. is 16 percent above the independent one. Intervals: the published modified HKSJ interval recomputed with the joint variance (t with 4 degrees of freedom) is **[0.561, 0.627]**; the direct percentile interval is [0.569, 0.615] and the normal interval [0.570, 0.617]. **Recommended for T4 and Table 1: [0.561, 0.627]**, the published method with the joint variance; the percentile interval belongs in D6. Its upper end, 0.627, replaces 0.622 wherever "the upper end of the top-budget interval" is used (IV.D Magnitudes, Table E8: for other packages).
  - [review] Confirmed: with the joint covariance Cochran's Q is 3.15 on 4 degrees of freedom, so the HKSJ variance factor is still truncated at one and [0.561, 0.627] stands. The "percentile" interval [0.569, 0.615] is re-centred on the estimate (estimate plus quantiles of the draws' deviations from the bootstrap population); the plain percentile interval is [0.578, 0.623] (delta-method draws, as for the other intervals; [0.578, 0.625] with the draws taken on the sigma scale), centred on the bootstrap population's value 0.602. Label the former "re-centred percentile" if it is printed.

### T1.2 Meta-regression with Marin as one cluster (R1 New 2(d); R4 N3(d), request 2)

rb1's six specifications (three-level RE with inverse-variance weights, unweighted LMM, design-balanced weights, the S scale, design fixed effects with inverse-variance and unweighted weights) and three new ones (study random intercepts, unweighted study RE, study fixed effects) were re-run with CR2 and a restricted wild cluster bootstrap-t clustered by study (Hoffmann, Meta, Marin, Farseer: four clusters) as well as by design (six). Every wild p-value is also computed by full enumeration of the 6^G Webb weight vectors (1,296 for four clusters, 46,656 for six), which removes Monte Carlo error. rb1's published design-cluster p-values reproduce to 1 in 10,000 (ties at the observed t are counted here; `rb5_sigma_metareg_check.csv`).

All 44 budgets (`rb5_sigma_metareg_slopes.csv`):

| Specification | Slope | Marin one cluster: CR2 p (df) | wild p (enumerated) | Six designs: CR2 p (df) | wild p (enumerated) |
|---|---|---|---|---|---|
| 3-level RE, inverse variance (primary) | -0.033 | 0.114 (2.8) | 0.049 | 0.097 (3.1) | 0.036 |
| Design RE, unweighted | -0.030 | 0.129 (2.7) | 0.054 | 0.050 (4.5) | 0.048 |
| Design-balanced weights | -0.034 | 0.064 (2.8) | 0.049 | 0.050 (3.3) | 0.033 |
| 3-level RE on the S scale | +0.137 (S units) | 0.150 (2.7) | 0.060 | 0.126 (3.2) | 0.062 |
| **Design FE, inverse variance** | **-0.021** | **0.490 (1.6)** | **0.370** | **0.489 (1.6)** | **0.366** |
| **Design FE, unweighted** | **-0.026** | **0.170 (2.7)** | **0.051** | **0.065 (4.5)** | **0.074** |
| Study RE, inverse variance | -0.032 | 0.118 (2.8) | 0.051 | | |
| Study RE, unweighted | -0.028 | 0.146 (2.7) | 0.054 | | |
| Study FE, inverse variance | -0.021 | 0.487 (1.6) | 0.384 | | |

- **p-value ranges** (`rb5_sigma_metareg_prange.csv`; CR2, random-draw and enumerated wild p-values; RE, balanced, S-scale and design-FE rows):
  - Marin as one cluster (four clusters): **0.049 to 0.490**.
  - Six design clusters: 0.032 to 0.489 (0.032 to 0.126 without the FE rows, which is rb1's published range).
  - Four to six clusters: **0.032 to 0.490**. **Recommended for T4 and Table 1 note g: "between 0.03 and 0.49"** (with Marin as one cluster alone: 0.05 to 0.49).
- **The CR2 interval and the wild p-value disagree about significance** in the primary row with Marin as one cluster (CR2 p = 0.11 on 2.8 df; wild p = 0.049), as Table 1 note g is to say. With four clusters the wild bootstrap has only 1,296 distinct draws, so its p-value moves in steps of 1/1,296 and ties matter; neither test is reliable (MacKinnon and Webb 2017).
- **Pooled line at 10^20 FLOP with Marin as one cluster** (`rb5_sigma_metareg_predictions.csv`): **0.680 [0.638, 0.722]** (CR2, 2.9 df); at 10^19 0.713 [0.707, 0.719]; at 10^21 0.647 [0.554, 0.741]. For the conclusion's "0.68 at 10^20 FLOP (95 percent confidence interval 0.638 to 0.722)" (H11: for WP8).
  - [review] Do not quote the CR2 interval at 10^19: its s.e. (0.0016) is one eighth of the model-based s.e. (0.0137), a small-sample artifact of CR2 with four clusters (the same happens with six: 0.0022). At 10^20 the CR2 interval is wider than the model-based one ([0.660, 0.701]) and is the conservative choice.
- Subsamples (both cluster levels): budgets <= 3e20, slope -0.008 (p 0.56 to 0.87 in the primary row); leaving out each design's largest budget, -0.030 (p 0.19 to 0.29); IsoFLOP designs only with Marin's corpora as five clusters (R3's round-2 sample), -0.048, p 0.017 to 0.022 (three study clusters: 0.043 to 0.096).
  - [review] Across all specifications the five-cluster IsoFLOP-only sample gives p from 0.013 to 0.23 (`rb5_sigma_metareg_prange.csv`, last row). That is a different sample (Farseer's path left out), so "between 0.03 and 0.49" is correct for the 44-budget regression only; if the text mentions five clusters it must name the sample.
  - [review] The upper end 0.49 is the design-FE inverse-variance row's CR2 p. It depends on the construction of the CR2 adjustment: rb1's (whitened, symmetric square roots) is exactly unbiased for the slope under the working model even with the fixed effects absorbed, while the Cholesky-based form gives 0.44 there (8 percent downward bias). rb1's construction is the right one; the reviewer's independent code reproduces 0.4896.

### T1.3 The top-budget set (R4 N3(b), (e))

`rb5_sigma_top_budget.csv`, `rb5_sigma_joint_linear.csv`:
- **Farseer's local path** above 3e20: 0.722 (0.008) at 5x10^20 and **0.658 (0.011) at 10^21**.
- **Top-budget mean** (Chinchilla 6e20, 1e21, 3e21; Llama 3 6e20, 1e21): 0.594 [0.565, 0.622] as published (Q = 2.46, p = 0.65). **Llama 3 carries 85 percent** of the inverse-variance weight (46 percent at 6e20 and 38 percent at 1e21).
- **With Farseer's 10^21 level**: fixed effect 0.625; random effects **0.606 [0.558, 0.655], Q = 21.1 (p < 0.001)**.
- **Without Chinchilla's five highest-loss runs** (132 runs; all nine Chinchilla budgets re-estimated with ra1's seed; RE sigma* 0.6812 reproduces rb4's review row R13): Chinchilla's 6e20 estimate moves from 0.550 to 0.665, and the top-budget mean from 0.594 to **0.598** [0.569, 0.627] (Q = 2.67; Llama 3's share 86 percent). R4's 0.599 substituted the 6e20 budget alone.
  - [review] Not so: substituting the 6e20 budget alone with this module's 132-run estimate also gives 0.598 (0.5978). R4's 0.599 is an approximate calculation; the paper should print 0.598.

### T1.4 Study-level mean on budgets at or below 3x10^20 (R4 N3(f); R1 New 2(b))

Each IsoFLOP design's random-effects mean over its budgets at or below 3e20 is computed exactly as ra1 computes the design mean (DerSimonian-Laird on S_b with bootstrap variances), from the joint draws; with all budgets the construction reproduces rb1's primary exactly (0.687259 [0.640215, 0.734302], Q = 8.2512; `rb5_sigma_study_check.csv`). Farseer enters with its first-derivative estimator on its path points at or below 3e20 (ra1's estimator and draws, regenerated exactly: section 3).

| Study (`rb5_sigma_study_parts_le3e20.csv`) | All budgets | Budgets <= 3e20 |
|---|---|---|
| Hoffmann (Chinchilla, N_F) | 0.660 (0.023), 9 budgets | 0.698 (0.023), 6 budgets |
| Meta (Llama 3) | 0.660 (0.023), 8 | 0.686 (0.026), 6 |
| Marin (three corpora, rho = 1) | 0.706 (0.026) | 0.706 (0.026) (all budgets <= 3e20) |
| Farseer (first derivative) | 0.708 (0.005) | 0.710 (0.006) |

- **Study-level mean, budgets <= 3x10^20: 0.708 [0.690, 0.726]**, Q = 1.02 (p = 0.80), between-study s.d. 0 (`rb5_sigma_study_level.csv`, group T1.4). Against 0.687 [0.640, 0.734] on all budgets, the difference comes from Chinchilla's and Llama 3's budgets above 3e20, and the heterogeneity disappears.
- Variants: Farseer's Hessian path over its levels <= 3e20, 0.702 [0.668, 0.736]; IsoFLOP studies only, 0.697 [0.636, 0.758].
- This supports "about 0.7 at 10^19 to 3x10^20 FLOP" as a statement about the level at those budgets. It is the number the m9 comparison (M9-6) should use beside 0.687.
  - [review] With the between-study s.d. at zero, Farseer's first-derivative entry (s.e. 0.006, which excludes smoothing bias: R1 minor 3) carries **84 percent** of the weight (column `weight_share_farseer`), so the interval [0.690, 0.726] is Farseer's. Given the same treatment as T1.6 (its bandwidth range as uniform uncertainty, s.e. 0.019), the row is **0.701 [0.665, 0.738]** (Q = 0.61; Farseer's share 35 percent; new row in `rb5_sigma_study_level.csv`, group T1.4, marked [review]). Restricting also the lower end to 10^19 (dropping the 3e18 and 6e18 budgets) gives 0.708. Recommended: Table 1 prints 0.708 [0.690, 0.726] with "0.701 [0.665, 0.738] with Farseer's bandwidth range as uncertainty" in the note, and M9-6 compares with the bandwidth-robust interval. "About 0.7" holds under every version (0.697 to 0.708).

### T1.5 Chinchilla's specification grid propagated (R4 N3 request 3; audit numbers_technology #4)

Every N_F row of `rb4_chinflop_specgrid.csv` (windows h = 0.6 to 1.5 and a global quadratic, cubic windows, E-free frontier smoothers, iterated centring; 137 and 132 runs) and rb4's review rows R11 to R13 replace Chinchilla's primary estimate; the other studies stay at their primary values (`rb5_sigma_study_level.csv`, groups "T1.5 grid" and "T1.5 R11-R13"; ranges in `rb5_sigma_study_ranges.csv`).

| Family | Rows | Study-level mean | Q (p) |
|---|---|---|---|
| Windows, order, frontier, centring; 137 runs; all nine budgets | 11 | 0.685 to 0.702 | 4.4 to 11.3 (0.010 to 0.22) |
| Same on the 132-run sample (h = 0.6 keeps nine budgets there) | 12 | 0.674 to 0.700 | 4.7 to 14.8 (0.002 to 0.20) |
| R11 to R13 (count-free N_F, T4-consistent membership, 132 runs) | 3 | 0.691 to 0.694 | 5.8 to 6.3 (0.10 to 0.12) |
| **Chinchilla's windows, samples and memberships (all of the above)** | 26 | **0.674 to 0.702** | **4.4 to 14.8 (0.002 to 0.22)** |
| Chinchilla only, adding rb4's counts and conventions | 38 | 0.674 to 0.702 | 4.4 to 14.9 |
| **All variants (rb1's 28 rows and the grids)** | 54 | **0.663 to 0.702** | 0.4 to 46.5 |

- The audit's approximate recomputation (0.685 to 0.702, Q 4.4 to 11.2) is confirmed on the 137-run grid. The maximum, 0.702 (Q = 4.4), is Chinchilla's h = 1.25 window (0.702, s.e. 0.011); the minimum of Chinchilla's variants, 0.674, is the 132-run sample with the narrowest window (0.596). The 137-run h = 0.6 window keeps seven of nine budgets (0.674, Q = 14.1) and is reported separately.
- **Recommended fills:** T3(a) "ranges over **4.4 to 14.8**"; T3(b) "between **0.663** and **0.702**"; Table 1 panel C row "Chinchilla's counts, conventions, windows, samples": **0.674 to 0.702**. Tables D4 to D6 (T13(f)) take these rows.

### T1.6 Farseer's bandwidth range as uncertainty (R1 minor 3)

Farseer's Hessian path mean moves between 0.664 (extended-grid CV bandwidth) and 0.727 (bandwidth x 2) across ra1's six bandwidth variants (`rb5_sigma_farseer_bandwidth.csv`). Treating the range as a uniform uncertainty added to sampling error, Farseer enters at the range's midpoint, 0.695, with s.e. 0.019 (against 0.005 in the headline), and the **study-level mean is 0.680 [0.643, 0.717]** (Q = 3.3, p = 0.35). Centring at the primary bandwidth (0.707) or at the first-derivative estimate (0.708) gives 0.684 [0.640, 0.727] and 0.684 [0.640, 0.728]; reading the range as +-1.96 s.e. gives 0.681 [0.645, 0.718]. Farseer's random-effects weight falls from 40 to 34 percent (its inverse-variance weight from 89 to 35 percent), and the between-study s.d. from 0.023 to 0.007.

### T1.7 The lower bound of sigma*(C) and the drift-agnostic set (R1 minor 4; R2 NM3(c))

The lower bound continues the Chinchilla-Llama 3 drift (-0.0576 per decade) from the top-budget value 0.594 (`rb5_sigma_pi_set.csv`; with the anchor at 10^21 it reproduces rb1's published bound to 2e-7).

| Anchor of the top-budget value | log10 C | Lower bound at 10^23 | at 10^24 |
|---|---|---|---|
| 10^21 (published) | 21.00 | 0.479 | 0.421 |
| Precision-weighted centre of the five budgets (8.0x10^20) | 20.90 | 0.473 | 0.415 |
| Compute-weighted centre (weights proportional to C; 1.5x10^21) | 21.19 | 0.489 | 0.432 |
| Unweighted centre of log10 C | 21.01 | 0.479 | 0.421 |
| 10^21, sigma_top and drift at their lower 95 percent limits (0.570; -0.084) | 21.00 | 0.403 | 0.319 |

- Under a linear drift an inverse-variance mean estimates sigma* at the inverse-variance-weighted centre of log10 C, so 8x10^20 is the coherent anchor; the referee's "compute-weighted centre" read literally (weights proportional to C) is 1.5x10^21. The two bracket 10^21 and move the bound by at most 0.011. **Recommended fill for T5: "(0.49 and 0.43 with it placed at the budgets' compute-weighted centre, 1.5x10^21 FLOP)"**, and D or E7 can add the precision-weighted 0.47 and 0.42.
- **Sets:** if the decline at the top budgets is real, [0.48, 0.59] at 10^23 and [0.42, 0.59] at 10^24; drift-agnostic (upper end 0.70), [0.48, 0.70] and [0.42, 0.70]. The level where the designs overlap is 0.708 [0.690, 0.726] by T1.4, so 0.70 is its rounding.
  - [review] 0.708 rounds to 0.71, not 0.70. Say instead that 0.70 lies inside the T1.4 interval (and inside its bandwidth-robust version, [0.665, 0.738]); rb1's 0.70 is kept as the drift-agnostic upper end.
- The lower bound is not a confidence bound: with sigma_top and the drift at their lower 95 percent limits (joint bootstrap; the drift's s.e. scaled for dispersion) it is 0.32 at 10^24.
  - [review] These rows use normal limits (0.570 for sigma_top); with the recommended t_4 limit (0.561) the value at 10^24 is 0.31. Memo-only statement; nothing in the paper uses it.

### T1.8 Coverage when noise is shared within a budget (R1 minor 7)

ra1's Monte Carlo was first reproduced exactly with ra1's seeds (R = 100): random-effects coverage 0.92 (Chinchilla, total parameters), 0.98 (Llama 3), 0.93 (Marin DCLM), with identical bias (`rb5_sigma_mc_check.csv`). The dependence designs keep the total noise variance and put a share rho of it into a per-budget random effect; R = 500 for rho = 0 and 0.5, R = 200 for the rest (`rb5_sigma_mc_dependence.csv`).

| Design (truth sigma*) | Independent noise, R = 500 | **rho = 0.5, per budget** | rho = 0.25 | rho = 0.75 | Smooth along ln N, half the variance |
|---|---|---|---|---|---|
| Chinchilla, total parameters (0.667) | 0.936 | **0.932** | 0.955 | 0.930 | 0.895 |
| Llama 3 (0.693) | 0.932 | **0.938** | 0.910 | 0.960 | 0.925 |
| Marin DCLM (0.663) | 0.954 | **0.942** | 0.940 | 0.885 | 0.895 |
| Chinchilla, N_F (0.656) | 0.936 | 0.918 | | | |

- **With half the noise variance shared within a budget, the random-effects interval covers the truth 93 to 94 percent of the time** (Monte Carlo s.e. 0.011; 92 percent for Chinchilla in N_F), with bias of at most 0.006 in absolute value. **Recommended fill for T3(b) and T12(i): "93 to 94"**.
- Why the shared part does little: a shift common to all runs of a budget leaves the profile's curvature unchanged and moves only the budget's minimum, and ra1's bootstrap already carries a budget-level shock on the frontier residual. A smooth distortion along each profile (a Gaussian process in ln N with a correlation length of one log point), the pattern of a digitized curve, does bend the profile: coverage is then 90 to 93 percent. The budget-level intervals that enter the meta-regression cover 89 to 90 percent under rho = 0.5 (90 to 92 percent with independent noise). The drift test rejects a true zero drift 7 to 8 percent of the time at the 5 percent level under rho = 0.5.
  - [review] Under the smooth within-budget distortion the same drift test rejects 6 to 15 percent of the time (Chinchilla 0.06, Llama 3 0.105, Marin 0.145; same file); the memo omitted this.
  - [review] R1 minor 7 compares with the experiment's within-trunk sharing, where a trunk is one architecture shared across token budgets. The public-design analogue is noise shared by **model size across budgets**, the transpose of the per-budget design. Sizes recur across budgets in all three designs (90 to 98 percent of runs belong to a size, within 3 percent in N, seen at two or more budgets). The reviewer's `review_depmc.py` (ra1's estimator, these truths, rho = 0.5, R = 200; `rb5_sigma_review_mc_size.csv`) gives:
    - **per-size shock:** random-effects coverage 0.915 (Chinchilla), 0.950 (Llama 3) and 0.965 (Marin DCLM);
    - **one smooth function of ln N shared by all budgets:** 0.905, 0.930 and 0.915;
    - **per-budget, the reviewer's seeds:** 0.930, 0.950 and 0.935, which replicates the rows above within Monte Carlo error.

    The level interval is therefore robust, at 90 to 97 percent across every sharing structure. But a smooth size distortion shared across budgets makes the within-design drift test reject a true zero drift 14 to 19 percent of the time (0.185, 0.190, 0.135). That matters for a drift carried by two digitized designs.
  - [review] With independent noise, the R = 500 rows give 0.936, 0.932 and 0.954. The paper's "92--98 percent" is ra1's R = 100 run and should become "93--95 percent" beside "93 to 94".

### T1.9 The extrapolation decomposition (R2 NM2 requests 1 and 3; R4 minor 14)

For each recipe the local log wedge is compared with the linear extrapolation from its own slope at the path, ln w_lin = b1 u (the tangent in Figure 5's panels d to f; 1/(1 + b1) is the local first-derivative sigma*). The convexity component is ln w_local - b1 u; the parametric gap splits exactly into it and a rotation-and-location component, ln w_param - b1 u. Bin means use the evaluation points inside the path's compute range. Draws: rb1's and ra1's wild cluster bootstraps regenerated exactly (every published draw of Delta, b1, b2 and the path reproduced to 1e-16; section 3), with per-point arrays kept. Intervals are percentile intervals.

Convexity component (log points; positive means the local wedge bends up beyond the linear extrapolation; `rb5_sigma_extrap_decomposition.csv`):

| Recipe | M 64-256 | **M 256-1,024** | M >= 1,024 | Rotation of the kappa-free gap at 256-1,024 |
|---|---|---|---|---|
| Farseer (404 runs; up to M = 2,570) | 0.021 [0.002, 0.024] | **0.169 [0.149, 0.175]** (42 points) | 0.373 [0.342, 0.464] | +0.034 [0.014, 0.041] |
| Marin, three corpora pooled (up to M about 1,110) | 0.093 [0.007, 0.094] | **0.427 [0.317, 0.445]** (19 points) | 1.021 [0.937, 1.207] (5 points) | -0.000 [-0.083, 0.013] |
| Marin, largest-M run of each budget left out (evaluated up to M about 320) | 0.094 [0.033, 0.100] | **0.377 [0.293, 0.388]** (8 points) | no support | -0.013 [-0.048, 0.006] |
| Marin, runs with M > 1,000 left out (up to M about 595) | 0.101 [0.025, 0.094] | **0.463 [0.286, 0.687]** (18 points) | no support | -0.028 [-0.121, -0.020] |
| Llama 3 (up to M about 290) | **-0.134 [-0.156, -0.061]** (16 points) | -0.006 [-0.130, 0.145] (2 points) | no support | -0.384 [-0.447, -0.256] |

- **Farseer and Marin:** measured against a linear extrapolation from the local slope at the path, the local wedge is convex; at M of 256 to 1,024 the convex part is 0.17 log points in Farseer and 0.43 in Marin. In Marin it is essentially the whole of the kappa-free form's gap (rotation 0.00), in Farseer more than all of it (the kappa-free form rotates the other way by +0.03).
- **Marin without its corner runs:** the convexity survives. Leaving out every budget's largest-M run lowers b2 from 0.029 to 0.018 and the convex part at 256 to 1,024 to 0.38 (the evaluated range then stops at M about 320); leaving out the two or three runs per corpus with M > 1,000 gives 0.46 (b2 0.026). The >= 1,024 bin exists only with those runs.
  - [review] The fall from 0.43 to 0.38 is a composition effect, not a smaller convexity: without the largest-M runs the 256 to 1,024 bin holds only points at M of 256 to about 320, where the primary estimate's convex part is 0.28 (4 points; reviewer's recomputation from rb1's evaluation points). Where both are evaluated, leaving the corner runs out does not reduce the convexity. W10 should say that the leave-out ranges stop at M of about 320 and 595. The reviewer's recomputation of the primary decomposition from rb1's and ra1's evaluation points reproduces every point estimate above (Farseer 0.169, Marin pooled 0.427, Llama 3 -0.134 and -0.006; b1, b2 and the local first-derivative sigma* of each recipe).
- **Llama 3:** the local wedge is concave (b2 = -0.029 [-0.035, -0.009]). The convex part is negative where Llama 3 has evidence, -0.13 at M of 64 to 256, and zero at its two points near M = 290. The kappa-free gap there (-0.38) is all rotation (-0.38): the steeper local slope at the path (local sigma* 0.629 against 0.69).
- **Recommended fill for W10 (WP4b):** "the convex part is 0.17 in Farseer and 0.43 in Marin (0.38 to 0.46 without its corner runs), at M of 256 to 1,024, and negative in Llama 3 (-0.13 at M of 64 to 256)".
- **Percentile intervals** for the published Delta (`rb5_sigma_extrap_delta_pct.csv`; for Farseer from ra1's published basic interval and population value, which is exact): of 140 bins, the percentile interval contains the estimate in 95 percent and the basic interval in 93 percent. [review] Of the 51 parametric bins plotted in Figure 5 (panels a to c), 7 percentile intervals exclude their estimate (Marin pooled at M < 16 and 64 to 256; Farseer's Chinchilla form at 16 to 1,024); none of the 14 plotted linear-extrapolation bins does. R4 minor 14 is therefore only partly met; the Figure 5 note (W13) should say so in one clause, or the figure should show plus or minus 1.96 bootstrap s.e. around each estimate instead. Example (R4 minor 14): Marin pooled, kappa free, 256 to 1,024: -0.427, percentile [-0.475, -0.359], basic [-0.549, -0.433]. The percentile interval can still exclude its estimate when the bootstrap population's statistic differs from the estimate by more than the draws' spread (Marin pooled, M < 16: -0.089 against [-0.087, -0.045]).
- **Local first-derivative sigma*** (`rb5_sigma_extrap_lin.csv`): Farseer 0.708 [0.705, 0.709]; Marin 0.666 / 0.667 / 0.676; Llama 3 0.629 [0.622, 0.645].

### T1.10 The within-Marin estimator gap (R2 NM3, smaller point)

IsoFLOP curvature gives 0.700, 0.713 and 0.705 (Comma, DCLM, Nemotron-CC), the local first-derivative estimator 0.666, 0.667 and 0.676: a gap of 0.029 to 0.046 (`rb5_sigma_marin_estimator_gap.csv`; from `rb1_sigmaC_extrap_convexity.csv`). For T3(b): "Within Marin, the local first-derivative estimator gives 0.666 to 0.676 against 0.700 to 0.713 from IsoFLOP curvature."

### T14 DataDecide's recipe tilt from final checkpoints (R2 minor 17)

m2's common-exponent panel model (recipe-specific A_r, B_r, E_r; common alpha, beta) fitted to the final checkpoint of every run whose schedule is complete: 950 of 1,100 runs (the 150 truncated auxiliary-seed runs of the 530M, 750M and 1B models dropped), 25 recipes, M = 85 to 110 (one ray). Cluster (size x seed cell, 38 cells) pairs bootstrap, B = 199, as in m2 (`rb5_sigma_dd_tilt.csv`, `_dd_profile.csv`, `_dd_recipes.csv`).

| | alpha | beta | Tilt range (95 percent) | M* factor under own exponents | M* factor under the reference exponents |
|---|---|---|---|---|---|
| Final checkpoints, NLS | 0.86 | 0.094 | 0.70 [0.49, 1.26] | 4.3 [1.9, 9.7] | 5.1 [3.2, 19.2] |
| Final checkpoints, Huber | 0.94 | 0.095 | 0.56 [0.27, 1.42] | 3.0 [1.5, 8.0] | 3.7 [1.9, 27.7] |
| All checkpoints (m2, published), NLS | 0.27 | 0.147 | 0.22 [0.13, 0.32] | 2.9 | 1.7 |
| All checkpoints (m2, published), Huber | 0.28 | 0.131 | 0.26 [0.18, 0.47] | 3.4 | 1.8 |

- **On the ray the tilt is identified only weakly.** The exponents move far from the all-checkpoint fit (alpha + beta 0.95 to 1.03 against 0.41) and are poorly pinned down (bootstrap alpha + beta 0.84 to 2.26); the Gaussian profile of the objective over (alpha, beta) is sharp, but it ignores the correlation of residuals within size-seed cells, which the cluster bootstrap carries. The recipes' ranking is stable (Spearman correlation of final-checkpoint and all-checkpoint tilts 0.75).
- **Implied allowance:** the M* factor across recipes under the fit's own exponents is 3.0 (Huber) to 4.3 (NLS), with bootstrap intervals from 1.5 to 9.7. The value 3.4 used in the sign sensitivity lies inside both intervals; 1.84 (the all-checkpoint tilt converted with the reference exponents) lies below both lower ends of the final-checkpoint factor under the reference exponents (3.2 and 1.9). Converting a tilt estimated with alpha near 0.9 and beta near 0.1 through the reference exponents is not meaningful, so the own-exponent factor is the relevant one.
- **Recommended for D4 (T12(r)):** "On final checkpoints only, one ray at M of about 100, the recipes' tilt is identified only weakly: the implied factor on M* is 3.0 to 4.3 under the fit's own exponents, with bootstrap intervals from 1.5 to 9.7, and the sign result's breakdown frontier (Online Appendix Figure E) does not depend on it."

> **[review] This section is superseded.** The fits above are the local optimum reached from m2's all-checkpoint start. For both estimators a second optimum with a LOWER objective exists. It exchanges the roles of the two power terms and reverses the recipes' tilt ranking. The reviewer found it with independent code (`review_ddtilt.py`) and confirmed it with m2's own `fit_panel_ls`. `ddtilt.py` now starts every fit and every bootstrap draw in both modes and reports both optima (`rb5_sigma_dd_tilt.csv`, columns `mode`, `global_optimum`, `share_draws_beta_gt_alpha`):
>
> | Final checkpoints | Mode | Objective | alpha | beta | Tilt range | M* factor, own exponents | Spearman with all-checkpoint tilts |
> |---|---|---|---|---|---|---|---|
> | NLS | alpha > beta (above) | 0.0099262 | 0.859 | 0.094 | 0.70 | 4.3 | +0.75 |
> | NLS | beta > alpha (lower objective) | 0.0098277 | 0.146 | 1.055 | 0.78 [0.48, 1.32] | 3.7 [1.9, 11.6] | -0.75 |
> | Huber | alpha > beta (above) | 0.0026963 | 0.936 | 0.095 | 0.56 | 3.0 | +0.76 |
> | Huber | beta > alpha (lower objective) | 0.0025923 | 0.155 | 1.228 | 0.50 [0.26, 1.67] | 2.0 [1.5, 6.8] | -0.82 |
>
> The bootstrap draws land in the beta > alpha mode 60 percent (NLS) and 77 percent (Huber) of the time; intervals are over draws from both modes. In the Gaussian profile the builder's optimum lies 8.7 above the global one, and a high barrier separates the two basins (at least 276 for alpha of 0.25 to 0.55; `rb5_sigma_dd_profile.csv`, now covering both), so a local solver never crosses it. The builder's statements that the ranking is stable, that the exponents are alpha near 0.9 and beta near 0.1, and that the factor is 3.0 to 4.3 (1.5 to 9.7) are therefore wrong. **The tilt is not identified on the ray**: its sign pattern depends on which power term is labelled N. **Use for D4 (T12(r)):** "On final checkpoints only, one ray at M of about 100, the recipes' tilt is not identified: the fit has two optima of almost equal fit that exchange the roles of the two power terms and reverse the ranking of the recipes' tilts, so we do not report a final-checkpoint allowance. The sign result's breakdown frontier (Online Appendix Figure E) does not depend on DataDecide." WP4a should not mark a final-checkpoint factor on the frontier. The all-checkpoint values (3.4; 1.84) are unaffected.

## 2. Figures (T10, T16)

- **Figure 5** (`figs_rb5.fig5`; preview `output/figures/rb5_sigma_preview_fig5.*`): rb1's layout; panels a to c now use percentile intervals and add a fourth series, the linear extrapolation from the local slope at the path (gray diamonds; Delta_lin = -convexity component); panels d to f keep the quadratic and relabel the tangent as that linear extrapolation. The note (W13, WP4b) should say "percentile intervals" and describe the fourth series.
- **Figure 3** (`code/paper/make_fig3_merged.py`, panel c now reads `rb5_sigma_pi_set.csv`; preview `rb5_sigma_preview_fig3.*`): the drift-agnostic upper line at 0.70 (dashed) beside 0.59, a lighter shade for the part of the set that is added if one is agnostic about the decline, the drift labelled "Chinchilla-Llama 3 drift (-0.058 per decade)", the values at 10^24 marked (0.42, 0.59, 0.70), shading from 10^21 with a tick at 3x10^21; the pooled-drift lower line is no longer drawn (the new note does not describe it). Panel d already had panel b's vertical scale (0.40 to 1.07), so T10(b) needs no change.

## 3. Checks

- Joint draws: per-budget sigma*_b, s.e. and intervals and the design summaries equal rb4's override inputs to 1e-15 (`rb5_sigma_joint_check.csv`); the 132-run Chinchilla RE sigma* equals rb4's review row R13 to its printed precision.
- Meta-regression: slopes, CR2 p (design and study clusters) equal rb1's to 1e-6; wild p to 1e-4 (ties; `rb5_sigma_metareg_check.csv`).
- Local-wedge draws: all 999 draws of rb1's four designs reproduce every published summary exactly (`rb5_sigma_extrap_check_rb1.csv`); ra1's Farseer draws reproduce the published s.e. of b1, b2, every Delta bin and every path level to 1e-16 (`rb5_sigma_extrap_check_farseer.csv`).
- Study level: the primary row equals rb1's (0.687259 [0.640215, 0.734302], Q = 8.2512); the identified set at the 10^21 anchor equals rb1's to 2e-7 (`rb5_sigma_pi_check.csv`).
- Monte Carlo: ra1's coverage and bias rows reproduced exactly (`rb5_sigma_mc_check.csv`).
- DataDecide: the all-checkpoint rows are m2's published values (no re-fit); the final-checkpoint fits start from m2's estimates and the bootstrap keeps the better of two starts with tight tolerances (a loose trust-region solve could stop at its start on this ray: development note).

## 4. Placeholder fills proposed for the writer (after the review)

| Placeholder or item | Value | Source |
|---|---|---|
| T3(a) [[T1.5: Q range]] | 4.4 to 14.8 | `rb5_sigma_study_ranges.csv` |
| T3(b) [[T1.5: 0.663]], [[T1.5: 0.702]] | 0.663, 0.702 | `rb5_sigma_study_ranges.csv` |
| T3(b), T12(i) [[T1.8]] | 93 to 94 ([review] and replace "92--98" for independent noise by "93--95", the R = 500 rows of the same file; add the per-size result of `rb5_sigma_review_mc_size.csv`, review memo section 4) | `rb5_sigma_mc_dependence.csv` |
| T4 [[T1.2: 0.03]], [[T1.2: 0.49]] | 0.03, 0.49 | `rb5_sigma_metareg_prange.csv` |
| T4, T12(m) [[T1.1]] (s.e. of -0.058) | 0.013 (0.009 from sampling error alone); [review] **0.014**, labelled "joint bootstrap draws, scaled for excess dispersion" | `rb5_sigma_joint_linear.csv`; `rb5_sigma_review_independent.csv` |
| T4 [[T1.1: [0.565, 0.622]]] | [0.561, 0.627] | `rb5_sigma_headline.csv` |
| T5 [[T1.7]] | 0.49 and 0.43 (compute-weighted centre, 1.5x10^21) | `rb5_sigma_pi_set.csv` |
| T11(a) Chinchilla's counts, conventions, windows, samples | 0.674 to 0.702 | `rb5_sigma_study_ranges.csv` |
| T11(a) Farseer's bandwidths | 0.680 [0.643, 0.717] | `rb5_sigma_study_level.csv` |
| T11(a) Study-level mean, budgets <= 3x10^20 | 0.708 [0.690, 0.726]; [review] add "0.701 [0.665, 0.738] with Farseer's bandwidth range as uncertainty" (Farseer carries 84 percent of the weight in 0.708) | `rb5_sigma_study_level.csv` |
| T11(a) top-budget row with Farseer 10^21 | 0.606 [0.558, 0.655], Q = 21.1 | `rb5_sigma_top_budget.csv` |
| T11(d) note g [[T1.2]] | 0.03 to 0.49 | `rb5_sigma_metareg_prange.csv` |
| H11 (WP8) [[T1.2: 0.645 to 0.716 recomputed]] | 0.638 to 0.722 | `rb5_sigma_metareg_predictions.csv` |
| W10 (WP4b) [[T1.9]] x 3 | 0.17 (Farseer), 0.43 (Marin), 0.38 to 0.46 (without corner runs; [review] those ranges stop at M of about 320 and 595) | `rb5_sigma_extrap_decomposition.csv` |
| T12(r) D4 | [review] the corrected T14 text (not identified on the ray; review memo section 2) | `rb5_sigma_dd_tilt.csv` |
| [review] T12 and D6, top-budget mean without Chinchilla's five high-loss runs | 0.598 (not 0.599) | `rb5_sigma_top_budget.csv` |
| [review] T12(m) Wald test of equal slopes, budgets <= 3x10^20; Llama 3's slope | Wald 13.9, p = 0.016; -0.050 (s.e. 0.017) in N_F (13.3, 0.021 and 0.018 are the Chinchilla-in-T values) | `rb1_sigmaC_metareg_design_slopes.csv` |

## 5. Caveats and open issues

1. The joint bootstrap is design-conditional: it carries the within-design dependence that ra1's bootstrap generates (shared frontier, budget shocks), not dependence in the published numbers themselves (digitization). T1.8's smooth-error design gives an idea of that channel (coverage 90 to 93 percent).
2. The choice between the joint-bootstrap interval of the top-budget mean with t_4 ([0.561, 0.627]) and without ([0.569, 0.615]) is a presentation choice; both are in `rb5_sigma_joint_linear.csv` and the headline file.
3. With four clusters, wild p-values move in steps of 1/1,296 and depend on how ties are counted; the tables count ties as at least as extreme.
4. The T1.4 mean restricts budgets but keeps each design's full-range frontier derivative, as the meta-regression's "budgets <= 3e20" subsample does.
5. The leave-corner-out checks re-estimate at the primary bandwidths; the evaluated range shrinks with the data (to M about 320 or 595).
6. The DataDecide final-checkpoint profile uses Gaussian independent errors; the bootstrap is the relevant uncertainty. B = 199 is enough for the percentile intervals quoted to two digits.
7. Nothing here uses the m9 experiment (no FineWeb estimate, sigma* for FineWeb or between-corpus statistic was computed or read).

## 6. Inventory

Code (`code/analysis/rb5_sigma/`): `run.py` (stages, caches `data/processed/rb5_sigma/stage_*.pkl`, log `run_log.txt`), `sigcommon.py` (paths, import plumbing, seeds), `jointboot.py` (T1.1, T1.3), `metareg.py` (T1.2), `convexity.py` (T1.9, Farseer's restricted estimator), `study.py` (T1.4 to T1.6, T1.10), `pidset.py` (T1.7), `depmc.py` (T1.8), `ddtilt.py` (T14), `figs_rb5.py` (figures). Also changed: `code/paper/make_fig3_merged.py` (panel c; `--out`), `code/analysis/rb1_sigmaC/figures_rb1.py` (its Figure 5 copy renamed).

Tables (`output/tables/rb5_sigma_*.csv`, 32 files): `headline` (every number above, with item and source), `joint_check`, `joint_budgets`, `joint_draws`, `joint_corr`, `joint_linear`, `top_budget`, `metareg_slopes`, `metareg_check`, `metareg_predictions`, `metareg_prange`, `study_level`, `study_parts_le3e20`, `study_ranges`, `study_check`, `farseer_bandwidth`, `farseer_restricted`, `marin_estimator_gap`, `pi_set`, `pi_check`, `mc_dependence`, `mc_check`, `extrap_decomposition`, `extrap_lin`, `extrap_delta_pct`, `extrap_corner`, `extrap_check_rb1`, `extrap_check_farseer`, `dd_tilt`, `dd_profile`, `dd_recipes`, `dd_sample`.
