# Module m1_chinchilla: what the canonical data identify

Chinchilla data, estimators, duality, functional dependence, and lab-own IsoFLOP technologies.

Author: m1 (Claude), 2026-09-23. Code: `code/analysis/m1_chinchilla/run.py`. Stages: horse, selection, duality, fdep, spec, labs, syslr, report. Seeds are fixed (`common.SEED = 20260923`). The run uses CPU only, with 6 processes.

**Independent review (2026-09-24).** The reviewer re-ran `run.py` from scratch into a clean output root: all stages ran without errors in 47 min (load average 22–40), and every CSV, `.npy` and `.npz` output was bit-identical to the builder's. The review added a `checks` stage (`review_checks.py`, about 4 min) and revised several claims. The main changes are in H3, H5, H6, H8, H9 and H11 and Claims 2–7, 11 and 12, each marked **[Rev]**. Full report: `output/memos/m1_chinchilla_review.md`.

**Wall time.** A full run (9 stages) took 47 + 4 min under load average 22–40. The most expensive step is the LAD bootstrap in the horse race; see Section 6.

**Notation.** Notation follows `paper/notes/model_spec.md`:
- a = β/(α+β), γ = αβ/(α+β), σ* = 2/(2+α+β);
- M* = D*/N*;
- w = ε_N/ε_D, evaluated at Chinchilla-70B (N = 70e9, D = 1.4e12).

**Standard errors.** "(x)" is a pairs-bootstrap SE and "[x]" a cluster-bootstrap SE, with clusters = the 9 reconstructed IsoFLOP budgets. For heavy-tailed objects (A, B, M*, w), the tables report robust SEs (IQR/1.349). Percentile CIs are in the CSVs.

---

## 1. Headline findings

### H1. Reproduction, and which estimator the published Besiroglu numbers come from

- **Our Huber-LSE estimate.** On the n = 240 Besiroglu sample, Huber-LSE (δ = 1e-3, 4,500-start grid) gives:
  - E = 1.8172 (0.027) [0.046];
  - A = 477.8, B = 2143.4;
  - α = 0.3473 (0.016) [0.024], β = 0.3672 (0.021) [0.046].
- **This is not exactly the published set.** Besiroglu et al. publish 482.01 / 2085.43 / 0.3478 / 0.3658.
  - Their published θ has a *higher* Huber(1e-3) objective than ours: 1.01864e-3 at the unrounded notebook values (1.0230e-3 at the rounded published values), vs 1.01827e-3 at ours.
  - Their own notebook's Huber grid code (L-BFGS-B, numerical gradients) returns our numbers (A = 477.6, B = 2142.9). This was a one-off check.
- **Where the published numbers come from.** They come from their notebook's Huber *log-likelihood with a free scale parameter* (BFGS from Hoffmann's values). The estimated scale is 4.7e-6, so the quadratic region is |r| < 5e-9: the objective is LAD in disguise.
  - Re-running that objective gives A = 482.10, B = 2076.2, α = 0.34783, β = 0.36564, with L1 objective 1.129507.
  - Our LAD started from the Huber optimum gives E = 1.8169, A = 482.01, B = 2085.2, α = 0.34781, β = 0.36585. This matches the published values to 4–5 digits, with the lowest L1 objective (1.129495).
  - The horse-race LAD (grid start) lands elsewhere on the flat L1 ridge: A = 480.9, B = 2083.6, α = 0.3477, β = 0.3658, with objective 1.129502.
  - All in `output/tables/m1_chinchilla_besiroglu_check.csv` (`horse_race.besiroglu_diagnostics`).
- **Size of the gap.** Huber and LAD differ by ≤ 0.07 SE, with small downstream consequences:

  | Object | Huber refit | Published values (≈ LAD) |
  |---|---|---|
  | w(70B) | 1.040 | 1.031 |
  | M*(5.76e23) | 17.9 | 18.4 |

### H2. Huber(δ = 1e-3) is LAD in practice

- 83.8% of log residuals lie in Huber's linear region, |r| > δ (84.9% at n = 245).
- On the same bootstrap draws, the SD of (Huber − LAD) is 0.0040 for β and 0.0033 for α. The pairs SEs are 0.021 and 0.016, and the correlation between the two estimators is 0.98.
- Source: `output/tables/m1_chinchilla_huber_vs_lad.csv`.

### H3. Estimator horse race (n = 240): robust estimators agree, least-squares estimators do not

| Estimator | β | Other |
|---|---|---|
| Huber | 0.367 (0.021) [0.046] | |
| LAD | 0.366 (0.021) [0.046] | |
| Gaussian NLS on log loss | 0.406 (0.025) [0.054] | |
| NLS in levels | 0.428 (0.026) [0.055] | α = 0.358 |

- α ranges 0.347–0.360, a 0.513–0.545 and γ 0.178–0.195.
- **[Rev] The robust vs least-squares gap in β is significant; the M\* gap is not.** All estimators share the same bootstrap draws, so the SD of the paired difference is the right s.e. (`m1_chinchilla_estimator_diffs.csv`, n = 240):

  | Comparison with Huber | Δβ (paired SD) [pairs / cluster z] | Δσ\* (SD) | ΔM\*(5.76e23) (SD) |
  |---|---|---|---|
  | LAD | −0.001 (0.004) | 0.000 (0.001) | 0.4 (1.6) |
  | Gaussian NLS, log | 0.039 (0.018) [z = 2.2 / 2.0] | −0.014 (0.005) | −3.7 (4.9) |
  | NLS, levels | 0.060 (0.020) [z = 3.0 / 2.9] | −0.019 (0.006) | −7.6 (5.3) |

  The marginal SEs overstate the uncertainty of these differences because the estimators are correlated across draws. The correlation is 0.67–0.75 for least squares vs Huber under the pairs bootstrap and 0.92–0.96 under the cluster bootstrap.
- **σ\* is the most stable object:** 0.737 (0.006) [0.010] under Huber, 0.718–0.723 under least squares.
- **The allocation extrapolates poorly.** M* at 1e21 is 21.4 (Huber) vs 18.3 (levels); at 5.76e23, 17.9 vs 10.4; at 1e26, 15.5 vs 6.5.
- **VPNLS (profile (E, A, B) by NNLS) equals direct NLS in levels to 7 digits.** Same objective; both find the global optimum.
- **KMW normalization is a pure reparameterization.** It gives identical estimates but:
  - cuts the Jacobian condition number from 2003 to 62 (Belsley-scaled: 220 to 33);
  - makes the level parameters precise. The normalized levels a′ = 0.379 (0.029) and b′ = 0.376 (0.023) have CVs of 8% and 6%. The raw levels do not: A = 478 (SE 129) and B = 2143 (SE 1628).
- With all 245 points, least squares blows up (β = 0.61–0.66, M*(5.76e23) = 1.2–2.0), while Huber/LAD give β = 0.452–0.453.
- Tables: `output/tables/m1_chinchilla_horse_race.csv|.tex`.

### H4. Cluster-robust inference matters and is imprecise

- Cluster-bootstrap SEs are 1.5–2.2× the pairs SEs for the exponents (β: 0.046 vs 0.021).
- For extrapolated objects the gap is much larger: the 95% CI for M*(5.76e23) is [7.8, 35.1] (pairs) vs [3.2, 126] (cluster).
- With only 9 clusters, the cluster bootstrap is itself noisy [cameron2008bootstrap].

### H5. Selection on the outcome (Besiroglu's "drop 5 worst")

**β falls almost monotonically in k,** the number of highest-loss runs dropped. **[Rev]** There are upticks of at most 0.0006 at k = 9, 12 and 13. α stays at 0.345–0.349 throughout; β does not plateau at k = 5. **[Rev]** Two runs tie at L = 3.4059 (ranks 6 and 7), so the k = 6 sample depends on the sort order; k ≤ 5 and k ≥ 7 are unaffected.

| k | 0 | 1 | 2 | 3 | 4 | 5 | 10 | 15 |
|---|---|---|---|---|---|---|---|---|
| β | 0.453 | 0.412 | 0.403 | 0.383 | 0.371 | 0.367 | 0.358 | 0.347 |

**The k = 0→5 swing.** A paired bootstrap applies the fixed truncation rule L < L̄ = 3.447 inside each of 400 draws. The swing is:

| Object | Swing | SE | Other |
|---|---|---|---|
| Δβ | 0.086 | 0.052 | 95% interval [0.006, 0.217]; > 0 in 98.8% of draws |
| Δa | 0.051 | 0.029 | |
| ΔM*(1e21) | −5.9 | 3.3 | |
| ΔE | 0.074 | 0.036 | |

**The five dropped runs are gross outliers in the data-starved corner.** They are the five most data-starved runs of the 1e19 IsoFLOP (D/N = 0.04–0.40).
- Their log residuals against the n = 240 fit are 0.06–0.31, i.e. 9–45 times the residual SD (σ_u = 0.0069).
- Runs 6 and 8 (D/N = 0.46, 0.51, same budget) are also under-predicted by about 0.05.

**It is not classical truncation bias.** A Hausman–Wise truncated-Gaussian MLE on n = 240 [hausman1977social] is identical to the untruncated Gaussian fit (β 0.4059 vs 0.4059). The largest truncation probability among the kept runs is 0.078. The swing is the leverage of a handful of gross outliers where the functional form fails (D/N → 0).

**Selecting on the regressor gives the same answer.** Dropping D/N < 0.4 (4 runs) gives β = 0.371 (0.024), essentially the outcome rule. Other rules:
- Dropping the whole 1e19 budget: β = 0.313 (0.022), a = 0.468, M*(1e21) = 27.4.
- Dropping the *lowest*-loss runs: β = 0.479. **[Rev]** This rule drops 6 runs, not 5, because of a tie at the 5th-lowest loss; the label now says so.

**The Chinchilla-70B verdict depends on excluding the gross outliers.** w(70B) is 1.54 (k = 0), 1.04 (k = 5) and 0.95 (k = 15).
- **[Rev]** The original wording, "rests on an arbitrary exclusion rule", overstated this. Every rule that removes the gross outliers gives w(70B) between 0.95 and 1.06: k = 4–15, D/N < 0.4, D/N < 1, D/N < 2, and the 5 largest residuals.
- Only including them (k ≤ 2: w ≥ 1.24) or discarding the whole 10^19 budget (w = 0.78) moves the verdict. The IsoFLOP-only sample with the 5 worst dropped gives 1.16.

Tables: `m1_chinchilla_selection_k.csv`, `m1_chinchilla_selection_rules.csv`, `m1_chinchilla_selection_swing.csv`, `m1_chinchilla_selection.tex`; Fig. `m1_chinchilla_selection`.

### H6. Duality: Approaches 1/2/3 = cost function / conditional factor demand / primal

**Approach 2 on the extraction.**
- a = 0.498, with classical OLS SE 0.025 (9 argmins, 7 d.f.) and stratified bootstrap SE 0.020.
- Hoffmann et al. report 0.49 (0.462, 0.534) for their Approach 2.
- The per-budget argmins scatter around the path with SD 0.15 in ln N.

**Approach 1 (Nerlove frontier through the 9 minima).**
- γ = 0.168 (0.008) with E free (E = 1.76), and 0.181 with E fixed at the Approach-3 value.
- The Approach-3 refit gives γ = 0.1785 (0.006).

**Factor-demand (path) restrictions.** These test the slope a and the level ln N*(C̄) of the expansion path. The main test is design-consistent: the null objects are the Approach-2/1 procedure applied to the primal's fitted values at the observed design, which removes the finite-grid parabola bias documented by [czech2026problems]. Covariance: within-budget stratified bootstrap, outlier-robust.

| Null technology | χ²₂ | p |
|---|---|---|
| Our refit | 2.9 | 0.24 |
| Besiroglu published | 4.3 | 0.12 |
| Hoffmann published A3 | 15.6 | 0.0004 |

- **Robustness across test variants.** There are 8 variants: {stratified, wild} bootstrap × {robust, SD} covariance × {analytic, design} null. At 5%, Hoffmann's A3 is rejected in 6 of 8. Besiroglu's published set and our refit are each rejected in 2 of 8, both times only under the naive analytic null with the stratified bootstrap. With plain SD covariances, the *design-consistent* Hoffmann test does not reject (stratified p = 0.16, wild p = 0.22).
- **Hoffmann's failure is mostly in the path *level*.** At the geometric-mean budget, the Approach-2 argmins put N* 12% above Hoffmann's A3: Δln N* = 0.113 (SE 0.031).
- The slope difference alone (a_A2 − a_H = 0.042) is marginal: OLS t = 1.65, p = 0.14.
- The same conclusion holds under the fixed-design wild bootstrap:

  | Null technology | p (wild bootstrap) |
  |---|---|
  | Hoffmann | 0.019 |
  | Besiroglu | 0.36 |
  | Refit | 0.36 |

**[Rev] The χ² p-values above are not robust inference; the Hoffmann rejection is suggestive, not decisive.** The χ² p-values depend heavily on which covariance is used (robust vs SD: 0.0004 vs 0.16 for Hoffmann), because resampled parabola argmins are heavy-tailed. The review adds two alternatives:
- **Bootstrap-calibrated p-values** (`m1_chinchilla_duality_bootcal.csv`): the same quadratic form, with critical values taken from its bootstrap distribution recentred at the estimate instead of from χ². These barely depend on the covariance choice.
- **A bootstrap-free classical test** (`m1_chinchilla_duality_classicalF.csv`): regress the 9 per-budget argmin gaps (data minus the null's design-consistent argmins) on [1, ln C_b] and F-test both coefficients, F(2, 7).

Design-consistent path tests, n = 240:

| Null technology | χ²₂ robust | χ²₂ SD | Boot-calibrated, stratified (robust / SD) | Boot-calibrated, wild (robust / SD) | Classical F(2,7) |
|---|---|---|---|---|---|
| Our refit | 0.24 | 0.36 | 0.23 / 0.25 | 0.70 / 0.61 | F = 0.45, p = 0.65 |
| Besiroglu published | 0.12 | 0.27 | 0.18 / 0.16 | 0.75 / 0.63 | F = 0.43, p = 0.66 |
| Hoffmann A3 | 0.0004 | 0.16 | 0.060 / 0.060 | 0.21 / 0.38 | F = 5.29, p = 0.040 |

- Hoffmann's A3 is the only technology near rejection. The classical F rejects at 5% (p = 0.04); the stratified bootstrap-calibrated test is just above 5% (p = 0.06); the wild bootstrap does not reject.
- The classical test puts Hoffmann's path level 0.122 (SE 0.046) above the null in logs; the slope gap, 0.042 (SE 0.023), is not significant.
- Summarize as: "Hoffmann's A3 is marginally inconsistent with its own IsoFLOP argmins (p ≈ 0.04–0.06 in the most defensible tests), mainly in the path level; Besiroglu's and our refit are consistent (p ≥ 0.16)."
- The analytic (naive) Hoffmann null is rejected more firmly: calibrated p = 0.010 stratified, 0.055–0.080 wild. But that comparison mixes finite-grid parabola bias into the test.

**Naive tests reject everything.** Tests that compare Approach 2 with the *analytic* A3 path (ignoring parabola bias) reject every technology, including our own refit (p = 0.033; **[Rev]** calibrated p = 0.047–0.053, wild 0.26–0.36). Duality tests must be design-consistent.

**Frontier (cost-function) restrictions are rejected for every A3 technology, including our own.** These test γ and L*(C̄); for our refit, χ²₂ = 20.7, p < 0.001.
- The Approach-1 frontier through the parabola minima lies 0.005–0.008 nats below every A3 frontier (SE 0.002).
- This is 0.2–0.3% of loss: a small but systematic lack of fit of the Chinchilla form at the IsoFLOP minima.

- **[Rev]** The frontier rejection survives bootstrap calibration under the stratified bootstrap: p = 0.007 (refit), 0.003 (Besiroglu) and 0.017–0.037 (Hoffmann). It does not survive under the wild bootstrap: p = 0.41, 0.15 and 0.70. Treat it as "rejected under the design-resampling bootstrap only".

**The all-restrictions Wald test (χ²₄) rejects for all technologies** because of the frontier component.

**With n = 245, our own primal is borderline on the path test.** χ²₂ = 11.6 (p = 0.003; OLS t on the slope = −2.8).
- **[Rev]** The original wording, "fails", overstated this. With bootstrap calibration p = 0.054 (robust) and 0.080 (SD); the SD χ² gives p = 0.51 and the wild bootstrap p = 0.73.
- The classical F(2,7) gives p = 0.041. The slope t that adds the primal's own s.e. to the OLS s.e. is −1.87 (p = 0.06).
- Reading: the outliers push the primal *towards* inconsistency with its own IsoFLOP argmins, but the evidence is marginal.

Tables: `m1_chinchilla_duality_objects.csv`, `m1_chinchilla_duality_tests.csv`, **[Rev]** `m1_chinchilla_duality_bootcal.csv`, `m1_chinchilla_duality_classicalF.csv`, `m1_chinchilla_duality.tex` (now with a bootstrap-calibrated p column and the classical F in the notes); Fig. `m1_chinchilla_duality`.

### H7. Revealed-preference test at Chinchilla-70B

If DeepMind's chosen model is cost-minimizing, then w = ε_N/ε_D = 1.

| Technology | w(70B) | 95% interval |
|---|---|---|
| Hoffmann A3, rounded | 0.62 | |
| Hoffmann A3, TeX precision | 0.71 | |
| Besiroglu published | 1.03 | |
| Our refit, pairs bootstrap | 1.04 | [0.82, 1.42] |
| Our refit, stratified bootstrap | 1.04 | [0.83, 1.41] |
| Our refit, 9-cluster bootstrap | 1.04 | [0.54, 2.01] |
| Laplace system | 1.03 | [0.84, 1.34] |

- Hoffmann's values lie outside the pairs and stratified intervals but *inside* the 9-cluster interval.
- Chinchilla-70B sits on the Approach-2 path extrapolated to its compute: the log gap is 0.025 and M*(A2) = 21.0.
- Source: `m1_chinchilla_revealed_pref_70b.csv`.

### H8. System estimator (Leon-Ledesma, McAdam and Willman 2010 logic)

The system combines the loss equation (Laplace errors, matching Huber/LAD) with the design-consistent argmin equation, under cross-equation restrictions. At n = 240:

| Object | Estimate |
|---|---|
| α | 0.348 (0.014) |
| β | 0.365 (0.016) |
| a | 0.512 (0.015) |
| σ* | 0.737 (0.005) |
| M*(1e21) | 21.6 |
| M*(5.76e23) | 18.5, 95% [9.2, 32.0] |

- **Nested LR test of the 2 restrictions** (the argmin equation may shift and tilt): 1.01, p = 0.60, at n = 240; 7.8, p = 0.020, at n = 245. The Gaussian version gives 3.3 (p = 0.19) and 14.2 (p < 0.001).
- **[Rev]** The argmin equation has only 9 observations, so the χ²₂ approximation is generous. An F-form calibration, F = (e^{LR/n₂} − 1)(n₂ − 2)/2 ~ F(2, n₂ − 2), attributes the whole LR to the argmin equation. It gives p = 0.67 (n = 240) and 0.048 (n = 245) for Laplace, and 0.27 and 0.004 for Gaussian (`m1_chinchilla_system_lr_smallsample.csv`).
- The precision gain over the primal alone is modest: SE(a) falls from 0.018 (primal, same stratified bootstrap) to 0.015. The pairs SEs of the primal are 0.020 for a and 0.021 for β, vs 0.016 for β in the system.
- Tables: `m1_chinchilla_system.csv`, `m1_chinchilla_system_lr.csv`, `m1_chinchilla_revealed_system.tex`.

### H9. Functional dependence (Proposition 1 / Lemma 2) in the data

**Design geometry (n = 240).** The variance of centered (ln N, ln D) is 1.83 along the expansion-path direction (β, α) and 0.90 in the transverse direction (α, −β): 33% transverse. With n = 245 the split is 1.84 / 1.12. The IsoFLOP-profile runs alone carry 37% transverse variance and the off-profile runs 21%.

**On-path subsample.** Take runs within ±0.15 log points of the fitted path (n = 41; transverse share 0.8%).
- *Jacobian collapse* (KMW-normalized parameters, evaluated at the full-sample θ, singular values ÷ √n):

  | Sample | 3rd singular value | 5th singular value | Condition number |
  |---|---|---|---|
  | Full design | 0.20 | 0.012 | 62 |
  | ±0.15 band | 0.025 | 0.0018 | 413 |
  | ±0.10 band | 0.018 | 0.0010 | 721 |
  | 9 argmins | 0.035 | 0.0008 | 915 |

  On an exactly optimal path the loss depends on (N, D) only through C, so the rank would be 3 and two singular values would vanish. In sorted order the two that collapse on-path are s₃ and s₅. s₄ (0.013 full vs 0.010 band) reflects the near-collinearity of E with the level parameters, which is present in every design. **[Rev]** The LaTeX table previously showed only s₄ and s₅, which hid the s₃ collapse; it now shows s₃–s₅ and a size-matched control row.
- *SE inflation* (full design → band). **[Rev]** The band has n = 41 against 240, so √(240/41) = 2.4 of each raw ratio is pure sample size. The review adds a size-matched control: 4 random n = 41 subsamples of the full design, 100 pairs draws each, median SE (`m1_chinchilla_fdep_se_ratios.csv`, `m1_chinchilla_fdep_sizecontrol.csv`):

  | Parameter | Full-design SE (n = 240) | Band SE (n = 41) | Random n = 41 SE | Raw ratio | **Band / random, same n** |
  |---|---|---|---|---|---|
  | α | 0.015 | 0.097 | 0.042 | 6.5× | **2.3×** |
  | β | 0.020 | 0.090 | 0.072 | 4.5× | **1.25×** |
  | a | 0.020 | 0.139 | 0.063 | 7.0× | **2.2×** |
  | γ | 0.005 | 0.018 | 0.019 | 3.6× | **0.94×** |
  | σ\* (κ = 1) | 0.006 | 0.011 | 0.020 | 1.9× | **0.54×** |

  Net of sample size, the on-path design inflates the SEs of α and a by about 2.2–2.3× and of β by only 1.25×. γ is estimated as precisely on-path as in a random subsample of the same size, as the theory says. The earlier "×4.5–7" figures conflated identification with n.
- **[Rev] The Jacobian collapse and the flat profile are genuine design effects.** The same random n = 41 subsamples have condition numbers 69–95 (vs 413 on-path) and s₃/√n ≈ 0.17 (vs 0.025). Their κ-free Gaussian profile LR over σ\* reaches 79–101, with 95% sets of width 0.04–0.05, vs ≤ 1.96 on-path.
- The point estimates wander. The band gives α = 0.414 and β = 0.251. The 9 argmins give SE(α) = 3.5, and 20% of their bootstrap draws have an exponent outside [0.05, 1.5].
- Under κ = 1, σ* stays fairly precise even on-path (SE 0.011): it is pinned by functional form through α + β = γ/(a(1−a)).

**Profile likelihood over σ\* with the outer exponent κ free.**
- *On-path:* flat. The Gaussian profile LR is ≤ 1.96 over σ* ∈ [0.50, 0.95], so every value is in the 95% set. The Huber objective is also flat.
- *Full design:* sharply curved. The LR reaches 424, and the 95% set is σ* ∈ [0.67, 0.69].
- This is the DMR-type non-identification of Proposition 1, shown on the canonical data.

Tables: `m1_chinchilla_fdep.csv`, `m1_chinchilla_design_variance.csv`, `m1_chinchilla_profile_sigma.csv`, `m1_chinchilla_identification.tex`; Figs. `m1_chinchilla_profile_sigma`, `m1_chinchilla_isoquants`.

### H10. Specification tests (n = 240)

**CES (α = β) is not rejected.**
- α − β = −0.020, with pairs SE 0.029 (p = 0.50) and cluster SE 0.066 (p = 0.76).
- The restricted CES has ρ = 0.357, σ = 0.737 and M* = 24.2 at every C.
- At n = 245 the result is borderline (p = 0.07 pairs).

**The outer exponent is below one (Kaplan direction), and κ = 1 is rejected.**
- κ̂ = 0.774 (Huber), with pairs SE 0.060 (95% CI [0.66, 0.89], p = 0.0002) and cluster SE 0.111 (p = 0.042).
- The Gaussian LR is 46.2 (p < 0.001); the Gaussian κ̂ = 0.737.
- Kaplan's κ = α_D = 0.103 is also far outside the CI.
- **This restriction is not innocuous.** With κ free, σ* = 0.700 (0.015) instead of 0.737. The inner exponents are nearly equal (a1 = 0.424, b1 = 0.431; a homothetic inner aggregator), and E = 1.757. The on-path objects barely move: a = 0.504, γ = 0.165.

**Rank-one curvature (translog in (ln N, ln D) on ln(L − E), with E fixed at 1.817).**
- The cross term is strongly negative, as predicted: b_ND = −0.043 (cluster t = −12.9).
- τ = b_NN b_DD − b_ND² = −5.1e-4 is not significantly different from the model-implied value (τ_model = −2.7e-6). The p-values are 0.06 (HC1), 0.09 (cluster delta method), 0.075 (pairs bootstrap with E re-estimated) and 0.27 (cluster bootstrap).
- But b_ND/√(b_NN b_DD) = −1.17 (vs −1), and **the test depends on E:**
  - E = 1.6934 gives τ > 0 (p = 0.09);
  - E = 1.8646 gives τ < 0 (p = 0.014).
- Read it as a joint test of (E, rank-one), not a clean test of the curvature restriction.

Tables: `m1_chinchilla_spec_ces.csv`, `m1_chinchilla_spec_kappa.csv`, `m1_chinchilla_spec_translog.csv`, `m1_chinchilla_spec_tests.tex`.

### H11. Lab-own technologies from IsoFLOPs

**Meta Llama 3** (133 digitized runs, 10 budgets, 6e18–1e22):
- *Approach 2 reproduces Meta's published law exactly:* D* = 0.2994·C^0.5368, vs 0.299·C^0.537 in the paper's Fig. 3. So a_N = 0.463 (OLS SE 0.018) and M*(3.8e25) = 40.9 (Meta: about 41–42).
- *The primal fitted to the same IsoFLOP points disagrees:*
  - α = 0.284 [0.050], β = 0.316 [0.024];
  - a = 0.527 [0.032], γ = 0.149, σ* = 0.770 [0.018];
  - M*(1e21) = 13.9 and M*(3.8e25) = 7.9. D/N *falls* with C, whereas Meta's A2 law has it *rising*.
- *The duality test rejects on the path slope:* OLS t = −3.5 (p = 0.008), design-consistent stratified Wald χ²₂ = 16.9 (p < 0.001), nested system LR = 7.5 (p = 0.024). It does *not* reject under the wild bootstrap (p = 0.51), which inflates the argmin variance.
- **[Rev] The rejection is weaker than stated.**
  - The OLS t treats the primal slope as known. Adding the primal's own stratified-bootstrap s.e. (0.024) to the OLS s.e. of the A2 slope (0.018) gives t = −2.1 (p = 0.035).
  - Bootstrap-calibrated p-values: 0.010–0.015 for the path, 0.050 for the slope alone; the wild bootstrap gives 0.78.
  - The F-form small-sample calibration of the system LR gives p = 0.050. The classical F on the 10 argmins gives p = 0.019, but it treats the primal as known.
  - The within-budget stratified bootstrap also understates the A2 slope uncertainty: SD 0.011 (robust 0.006) vs OLS s.e. 0.018. The registry now uses the OLS s.e. for this row.
  - Reading: the primal and the A2 path differ by 0.063 in slope, significant at about the 5% level under the more conservative calculations, not at 0.1%.
- *Under κ = 1, Approach 2+1 imply* α = γ/a = 0.295 and β = 0.254, with σ* = 0.785.
- *Caveat:* digitized points with unstated loss units. The primal and dual extrapolations to the 405B scale differ five-fold in M*.

**Marin 2026-03** (Llama-2 architecture; Comma, DCLM, Nemotron-CC; 85–88 runs, 7–8 budgets, 1.8e18–3e20):
- The primal gives α = 0.64–0.69 and β = 0.35–0.41, hence a = 0.34–0.39, γ = 0.23–0.26 and σ* = 0.65–0.66.
- Approach 2 gives a = 0.405–0.424.
- Design-consistent path tests pass (p = 0.33–0.53), while the naive analytic tests reject for Comma and DCLM (p = 0.03).
- **[Rev] The design explains the naive gap, but only about three-quarters of it is finite-grid parabola bias.** The rest is FLOP accounting: Marin's nominal budgets (3 × forward FLOPs) differ from 6ND by −7% to +35%, strongly and negatively correlated with N within each budget (within-budget correlation of ln N and ln(6ND/C) is −0.89 to −1.00).
  - Decomposition of the slope gap (`m1_chinchilla_labs_design_decomp.csv`): Comma 0.060 = 0.041 (grid) + 0.015 (FLOP accounting) + 0.005 (residual); DCLM 0.052 = 0.036 + 0.014 + 0.003; Nemotron 0.037 = 0.030 + 0.010 − 0.003.
  - The naive rejections are themselves fragile (bootstrap-calibrated p = 0.15–0.32).
  - Under the wild bootstrap the χ² design test rejects for Comma and Nemotron (p = 0.03–0.056), but not after calibration (p = 0.60–0.74).
- **Approach 1 (the frontier) is not identified:** with only 7 budgets over 2 decades, E and γ trade off (γ_A1 = 0.05–0.08, E = 0.1–1.0).
- M* extrapolations to 3.8e25 (57–707) are meaningless.

**(Mis)Fitting FineWeb/C4** (checkpoint-interpolated IsoFLOPs; reported with a caveat):
- α = 0.610, β = 0.705, σ* = 0.603.
- Approach 2 (a = 0.517) and the primal (a = 0.536) are consistent (p = 0.11 design).

Tables: `m1_chinchilla_labs_technology.csv`, `m1_chinchilla_labs_duality_tests.csv`, `m1_chinchilla_labs_system.csv`, `m1_chinchilla_labs.tex`.

---

## 2. Methods

### Data

**Chinchilla.** Epoch's digitization of Hoffmann et al. (2022), Fig. 4, via `sl.chinchilla_extraction`: 245 runs.
- N is total parameters.
- C is read off the figure's x-axis, and D = C/(6N) is constructed (not observed).
- L is MassiveText validation loss (nats/token), quantized to about ±0.01 by the 256-colour map.
- Besiroglu sample: the 5 highest-loss runs are dropped (n = 240).
  - Their rule is "L < 5th-highest loss". Note that only 4 of the 5 have D/N < 0.4 (the 5th has 0.4045); the data notes say "5 outliers at D/N < 0.4".

**IsoFLOP budget reconstruction.**
- Digitized log10 C sits systematically 0.018 dex *left* of the 9 nominal budgets. I estimated this offset as the median deviation of points within 0.1 dex of a budget.
- Cluster = the nearest nominal budget after removing the offset (all 245 runs get a cluster).
- IsoFLOP-profile membership requires |Δlog10 C| ≤ 0.045 (about 11%): 137 of 245 runs (132 of 240).
- The other runs are off-profile runs in Hoffmann's scatter (they form streaks of fixed D).
- The open-athena "ml_scalefit" Chinchilla series (Apple) is a 124-row snapped subset of the *same* Epoch extraction, not an independent source; I did not use it.

**IsoFLOP compilation.** open-athena/isoflop-experiments (Hugging Face, Apache-2.0, 814 rows; commands in `code/data/download_m1_chinchilla.sh`).
- *Llama 3:* 133 rows. These equal the eric-czech digitization row for row (checked). N = C/(6D) with C the nominal budget; D is read off the figure.
- *Marin:* Paloma macro loss; forward FLOPs × 3.
- *(Mis)Fitting:* D is derived from interpolated checkpoints.
- *Licensing:* the Llama extraction repo has no license and is not redistributed. The Epoch repo has no license, so Chinchilla data should be used, not redistributed.

### Estimators

All use the log-sum-exp parameterization θ = (ln A, ln B, ln E, α, β) unless stated. See `estimators.py`.

- **Huber:** `sl.fit_chinchilla` with the full DEFAULT_GRID (4,500 starts).
- **Gaussian NLS on log loss.**
- **LAD:** Huber(1e-6) from warm starts, then restarted Nelder–Mead on the exact L1 objective.
- **NLS in levels:** analytic gradient, grid of starts.
- **VPNLS in levels:** given (α, β), NNLS for (E, A, B) on scaled columns; a 0.01 grid on [0.02, 1.5]², then Nelder–Mead.
- **KMW:** inputs divided by their geometric means (N̄ = 8.49e8, D̄ = 1.69e10), with the Huber objective.

### Inference

**Pairs and cluster bootstraps** (horse race: B = 400).
- The same draws are used for every estimator, so estimator differences can be bootstrapped.
- Replications are warm-started from the full-sample estimate, Hoffmann and Besiroglu (the grid is not repeated).
- There were 0 failed replications in the horse race and 2 of 298 in the n = 245 duality stage.

**Duality.**
- *Approach 2:* parabola of L in ln N per budget; budgets need ≥ 3 distinct N and a convex parabola.
- *Approach 1:* bounded NLS in levels on the 9 minima.
- *Tests:*
  - Wald statistics on the difference vector (a, ln N*(C̄), γ, L*(C̄)), with C̄ = the geometric mean of the budgets.
  - Each is computed against both the analytic and the design-consistent null.
  - Covariance: robust (IQR scales + Spearman correlation mapped to Pearson), from a within-budget stratified bootstrap (B = 300; off-profile runs form their own stratum).
  - Sensitivity checks: SD-based covariance and a fixed-design wild bootstrap (400 Rademacher draws).
  - Classical OLS t-test on the path slope.
- *System:* concentrated quasi-likelihood,
  n₁·ln mean|r₁| + (n₂/2)·ln mean r₂² (Laplace), or the Gaussian analogue,
  with the argmin equation predicted design-consistently. SEs come from the stratified bootstrap. The nested LR frees a level and slope shift of the argmin equation (`duality.system_lr`).
- The first, non-nested LR in `m1_chinchilla_system.csv` (the "LR" column) compares with a free straight-line path. It can be negative and **should not be used**; use `m1_chinchilla_system_lr.csv`.

**Functional dependence.**
- On-path bands: |ln N − ln N*(C)| ≤ 0.15 and ≤ 0.10 around the Huber path.
- Jacobian of ln L̂ evaluated at the full-sample θ, in raw and KMW coordinates.
- Subsample fits use 75 starts; pairs bootstrap B = 200.
- Profile over σ* on a 0.01 grid, with κ, E, A, B and a1/(a1+b1) profiled out (forward and backward warm-start sweeps).

**Specification tests.**
- B = 300 pairs and cluster draws. Each draw re-estimates E, the translog τ, τ at the model's own fitted values, and κ (warm-started).
- CES restricted fit by Huber.

**[Rev] Review checks (`review_checks.py`, stage `checks`).**
- *Paired estimator differences:* SD across the shared bootstrap draws of θ_e − θ_Huber.
- *Bootstrap-calibrated Wald tests:* W = d̂′V⁺d̂, with V either robust or SD. The p-value is the share of draws with (d\*_b − d̂)′V⁺(d\*_b − d̂) ≥ W, i.e. critical values come from the bootstrap distribution, not from χ².
- *Classical F:* the per-budget argmin gaps e_b = ln N\*_b(data) − ln N\*_b(null, design-consistent) are regressed on [1, ln C_b − mean]. Both coefficients are F-tested with F(2, n_b − 2), with the null technology treated as known.
- *Slope t with primal variance:* d_a / √(se_OLS² + rse(a_A3)²).
- *Design decomposition:* the design-consistent null is evaluated at the actual design and at D = C_b/(6N) (exact 6ND = C_b).
- *Small-sample LR:* F = (e^{LR/n₂} − 1)(n₂ − 2)/2 ~ F(2, n₂ − 2).
- *Size control:* 4 random n = 41 subsamples of the n = 240 design (seeded), each with a pairs bootstrap (B = 100, same 75 starts), Jacobian singular values and the κ-free Gaussian profile over σ\*.
- *Warm-start check:* 40 + 40 horse-race draws re-fitted with the FAST_GRID added.

**Selection.**
- For each k = 0..15: Huber (FAST_GRID plus warm starts), LAD and Gaussian estimates, with a pairs bootstrap (B = 200).
- Paired swing bootstrap: B = 400.
- Truncated-Gaussian MLE conditional on L < L̄.

---

## 3. Tables and figures

### Paper-ready LaTeX tables

All are in `output/tables/`.

| File | Contents |
|---|---|
| `m1_chinchilla_horse_race.tex` | Table: estimator horse race on the Chinchilla data (6 estimators × {n = 240, n = 245}; pairs and cluster SEs). |
| `m1_chinchilla_selection.tex` | Table: selection on the outcome (k-worst path, alternative exclusion rules, Hausman–Wise). |
| `m1_chinchilla_duality.tex` | Table: dual objects (A2/A1 vs A3) and Wald tests of the cross-equation restrictions (**[Rev]** with a bootstrap-calibrated p column and the classical F in the notes). |
| `m1_chinchilla_revealed_system.tex` | Table: revealed-preference test at Chinchilla-70B and the system estimates. |
| `m1_chinchilla_identification.tex` | Table: functional dependence (full vs on-path samples: transverse share, singular values s₃–s₅, SEs, profile LR; **[Rev]** plus a size-matched random-subsample row). |
| `m1_chinchilla_spec_tests.tex` | Table: CES, outer exponent κ, rank-one translog tests. |
| `m1_chinchilla_labs.tex` | Table: lab-own technologies (Llama 3, Marin ×3, (Mis)Fitting): primal vs Approach 2+1, and tests. |

### CSVs behind the tables

`output/tables/m1_chinchilla_*.csv`:
- horse_race, huber_vs_lad, besiroglu_check;
- selection_k, selection_rules, selection_swing, selection_worst_points;
- duality_objects, duality_tests, a2_minima, revealed_pref_70b, system, system_lr;
- fdep, design_variance, profile_sigma;
- spec_ces, spec_kappa, spec_translog;
- labs_technology, labs_duality_tests, labs_duality_objects, labs_system, labs_a2_minima;
- **[Rev]** stage `checks` (`review_checks.py`):
  - estimator_diffs: paired-bootstrap estimator differences;
  - duality_bootcal: bootstrap-calibrated duality tests, including slope-only tests and the slope t with the primal's s.e.;
  - duality_classicalF: bootstrap-free fixed-design F tests of the path restrictions;
  - labs_design_decomp: finite-grid vs FLOP-accounting decomposition;
  - system_lr_smallsample: F-form calibration of the nested LR;
  - warmstart_check;
  - fdep_sizecontrol and fdep_se_ratios: size-matched control for the on-path band.

### Technology registry

`output/tables/technology_registry_m1.csv` has 29 rows:
- Chinchilla: 6 estimators × {240, 245}, plus 4 system rows.
- Literature: Hoffmann TeX, Hoffmann rounded, Besiroglu published.
- Labs: Llama 3, Marin ×3 and (Mis)Fitting, each with an A3 row and an A2A1 row.

Bootstrap covariances of θ = (ln A, ln B, ln E, α, β) are in `data/processed/m1_chinchilla/cov_<row_id>.npy` (and `_cluster`). Draws are in `boot_*.npy`; the column order is in `boot_columns.txt` (first 5 = θ). The A2A1 lab draws have columns a, γ, E_A1, ln K, ln G. Duality draws are in `duality_draws_*.npz`.

### Figures

In `output/figures/`, each as .pdf and .png.

| File | Contents |
|---|---|
| `m1_chinchilla_isoquants` | **Paper Figure 1.** The (log10 N, log10 D) plane: Chinchilla runs coloured by loss (the 5 dropped runs hollow), isoquants of the Huber refit, isocost lines C = 6ND, expansion paths (refit a = 0.514; Hoffmann A3 a = 0.456; Meta Llama 3 A2 a = 0.463), Chinchilla-70B, and along-path vs transverse arrows with design variances 1.83 / 0.90. |
| `m1_chinchilla_profile_sigma` | Profile likelihood ratio over σ* with κ free: full design (sharp minimum at 0.68–0.70) vs on-path band (flat, LR < 2). Gaussian and Huber panels. |
| `m1_chinchilla_selection` | β, α and M*(1e21) as the k highest-loss runs are dropped (Huber with 95% band, LAD, Gaussian). |
| `m1_chinchilla_duality` | Approach-2 argmins and Approach-1 frontier vs Approach-3 implied paths and frontiers. Chinchilla (with Hoffmann A3) and Llama 3. |

---

## 4. Claims for the paper

1. **The published Chinchilla replication numbers are an LAD fit, and Huber(1e-3) on log loss is LAD in practice.**
   - *Evidence:* H1 and H2; `m1_chinchilla_besiroglu_check.csv` (LAD: A 482.01, B 2085.2, α 0.34781, β 0.36585 vs published 482.01, 2085.43, 0.3478, 0.3658; Huber(1e-3) objective lower at our estimate); `m1_chinchilla_huber_vs_lad.csv` (83.8% of residuals in the linear region; SD(β_H − β_LAD) = 0.004).
   - *Caveat:* the difference from the Huber optimum is ≤ 0.07 SE. This is a numerical-practice point, not a substantive one.

2. **Robust and least-squares estimators give materially different data exponents on the same data.** β is 0.367 (Huber/LAD) vs 0.406 (Gaussian log) vs 0.428 (levels). a ranges 0.513–0.545 and M*(5.76e23) 10–18.
   - *Evidence:* `m1_chinchilla_horse_race.tex`; **[Rev]** `m1_chinchilla_estimator_diffs.csv`. The paired bootstrap of the difference (same draws) gives Δβ = 0.039 (SD 0.018; z = 2.2 pairs, 2.0 cluster) for Gaussian-log and 0.060 (0.020; z = 3.0, 2.9) for levels.
   - *Caveat:*
     - **[Rev]** The original caveat, "under 1 cluster SE", compared the differences with marginal SEs and was wrong: paired, they are 2–3 SDs.
     - The implied M\*(5.76e23) differences (−3.7 and −7.6) are *not* significant (z = −0.7 and −1.4). Do not claim that the estimator choice changes the frontier-scale allocation.
     - The robust estimators are preferred because the residuals are heavy-tailed (the gross outliers of H5).

3. **σ\* ≈ 0.72–0.74 is the most robust technology object on the Chinchilla data** (SE 0.006 pairs / 0.010 cluster). Across estimators its range is 2.6% of its midpoint, vs 6% for a and 55% for M\*(5.76e23).
   - *Evidence:* `m1_chinchilla_horse_race.csv` (σ* 0.718–0.737 at n = 240; the cluster CI for Huber is [0.718, 0.756]).
   - *Caveat:*
     - This holds only under κ = 1. With κ free, σ* = 0.700 (0.015); see Claim 8.
     - **[Rev]** "Robust" means small relative variation, not statistical equality. Because σ\* is so precisely estimated, the least-squares vs Huber differences (−0.014 and −0.019) are statistically significant in the paired bootstrap (z = −2.6 and −3.2).

4. **Dropping the highest-loss runs is outcome truncation in form, but here its effect is the leverage of a few gross outliers in the data-starved corner, not truncation bias.**
   - *Evidence:* `m1_chinchilla_selection_rules.csv` (Hausman–Wise β = Gaussian β = 0.4059; D/N < 0.4 rule β = 0.371 vs worst-5 rule 0.367); `m1_chinchilla_selection_swing.csv` (Δβ = 0.086, SE 0.052); `m1_chinchilla_selection_worst_points.csv` (log residuals 0.06–0.31 vs σ_u = 0.0069).
   - *Caveat:* the exclusion is still consequential. β keeps falling to 0.347 at k = 15, and w(70B) moves from 1.54 to 0.95 across k. Report sensitivity to k whenever the Chinchilla technology is used.
     - **[Rev]** Once the gross outliers are out (k ≥ 4, or D/N-based rules), w(70B) stays in [0.95, 1.06].
     - The Hausman–Wise comparison assumes Gaussian errors, which the 9–45σ outliers contradict. It shows that truncation *under the maintained Gaussian model* is ignorable, not that the model is right.

5. **[Rev, weakened] Approach 2 identifies the expansion path; tests against the primal must be design-consistent. Under such tests, Besiroglu's and our refit are consistent with the IsoFLOP argmins (p ≥ 0.16 in every variant). Hoffmann's published A3 is marginally inconsistent (p ≈ 0.04–0.06 in the most defensible tests), mainly through the path level.**
   - *Evidence:*
     - Classical fixed-design F(2,7) on the 9 argmin gaps: Hoffmann F = 5.29 (p = 0.040), vs p = 0.66 (Besiroglu) and 0.65 (refit) (`m1_chinchilla_duality_classicalF.csv`).
     - Bootstrap-calibrated design-consistent path test: Hoffmann p = 0.060 (stratified, either covariance), 0.21–0.38 (wild) (`m1_chinchilla_duality_bootcal.csv`).
     - χ²-based: robust stratified χ²₂ = 15.6 (p = 0.0004) vs 4.3 (0.12) and 2.9 (0.24) (`m1_chinchilla_duality_tests.csv`).
   - *Caveat:*
     - The builder's headline p = 0.0004 comes from pairing a heavy-tail-robust covariance with χ² critical values. With SD covariances p = 0.16; with bootstrap calibration p = 0.06. Do not report the χ² p-value as the test.
     - The gap is in the path *level*: N\* is 12–13% above Hoffmann's at C̄ (Δ ln N\* = 0.113–0.122, SE 0.031–0.046). The slope gap alone is not significant (OLS p = 0.14; calibrated p = 0.057).
     - The naive analytic comparison rejects everyone, including our refit, which is why design-consistency matters.
     - Suggested wording for the paper: "Hoffmann et al.'s published Approach-3 parameters imply a compute-optimal model size about 12% below what their own IsoFLOP argmins show (p ≈ 0.04–0.06); the replication parameters show no such gap."

6. **Revealed preference at Chinchilla-70B rejects Hoffmann's A3 (w = 0.62–0.71) but not the refit (w = 1.04, 95% [0.82, 1.42]).**
   - *Evidence:* `m1_chinchilla_revealed_pref_70b.csv`; horse race CIs.
   - *Caveat:* with the 9-cluster bootstrap the CI is [0.54, 2.01] and does not exclude Hoffmann. The test also depends on the exclusion rule (w = 1.54 at n = 245).
     - **[Rev]** "Rejects" here means that Hoffmann's implied w lies outside the refit's pairs/stratified 95% interval. This is a joint statement about Hoffmann's parameters and the maintained optimality of Chinchilla-70B (training compute only). With 9 clusters it is not a rejection.

7. **On-path data do not identify σ\* once the outer exponent is free, and they identify α and a much less precisely. They do identify γ.**
   - *Evidence:* `m1_chinchilla_fdep.csv`, `m1_chinchilla_identification.tex` and **[Rev]** `m1_chinchilla_fdep_se_ratios.csv` (size-matched control):
     - With κ free, the profile LR over σ\* ∈ [0.50, 0.95] is ≤ 1.96 on-path. Random full-design subsamples of the same size (n = 41) reach 79–101, and the full design (n = 240) reaches 424. This is the cleanest piece of evidence and it survives the size control.
     - Two of the five normalized Jacobian singular values collapse (s₃ ×8, s₅ ×7). The condition number goes from 62 (full) and 69–95 (random, same n) to 413–915 on-path.
     - Relative to random subsamples of the same size, on-path SEs are 2.3× (α), 2.2× (a), 1.25× (β), 0.94× (γ) and 0.54× (σ\* under κ = 1).
   - *Caveat:*
     - **[Rev]** The builder's "SEs ×4.5–7" compared n = 41 with n = 240. Net of sample size, the SE inflation is 2.2–2.3× for α and a and small for β. Report the size-matched ratios.
     - The band is an empirical approximation to exact optimality (transverse share 0.8%, not 0). Under κ = 1, functional form alone pins σ\* (SE 0.011 on-path, *smaller* than in random subsamples). That is the paper's point: on-path σ comes from the functional form.

8. **The Chinchilla outer-exponent restriction κ = 1 is rejected on the canonical data (κ̂ = 0.77), and relaxing it lowers σ\* from 0.737 to 0.700.**
   - *Evidence:* `m1_chinchilla_spec_kappa.csv`: pairs p = 0.0002, cluster p = 0.042, Gaussian LR = 46.
   - *Caveat:*
     - κ is identified only from the shape of the IsoFLOP profiles, which are digitized (loss quantized at about ±0.01).
     - E shifts from 1.817 to 1.757.
     - Not yet replicated on other sweeps (m2 should check it on Farseer).
     - The cluster p-value is borderline.

9. **CES (α = β) is not rejected on Chinchilla** (α − β = −0.020, p = 0.50 pairs, p = 0.76 cluster). The homothetic CES fit implies σ = 0.737 and M* = 24.2 at all C.
   - *Evidence:* `m1_chinchilla_spec_ces.csv`.
   - *Caveat:* the test has low power with the cluster SE of 0.066. At n = 245 it is borderline (p = 0.07).

10. **The translog cross-partial has the sign the Chinchilla form predicts** (b_ND < 0, t = −13). The rank-one restriction is not rejected at 5% at the estimated E.
    - *Evidence:* `m1_chinchilla_spec_translog.csv`.
    - *Caveat:* the restriction's verdict flips with E (p = 0.09 at E = 1.69, p = 0.014 at E = 1.86), and the curvature correlation is −1.17, not −1. Present it as weak support.

11. **Meta's own Approach-2 law is exactly reproduced by the digitized IsoFLOPs** (D* = 0.2994·C^0.5368). **The primal fitted to the same points implies a different allocation:** a = 0.527 vs 0.463, with M* falling rather than rising in C, and M*(3.8e25) = 7.9 vs 41.
    - *Evidence:* `m1_chinchilla_labs_technology.csv` and `m1_chinchilla_labs_duality_tests.csv` (OLS t = −3.5, p = 0.008; stratified Wald p < 0.001; nested LR p = 0.024).
    - *Caveat:*
      - **[Rev] Statistical strength is modest.** With the primal's own s.e. included, the slope t = −2.1 (p = 0.035). Bootstrap-calibrated p = 0.05 (slope) and 0.010–0.015 (path); the small-sample system LR p = 0.050; the wild bootstrap gives p = 0.51–0.78. Say "the primal and Meta's A2 law differ by 0.06 in the path slope (p ≈ 0.01–0.05)", not "p < 0.001".
      - The data are digitized with unknown loss units.
      - The frontier restriction is rejected (χ² = 28; bootstrap-calibrated p = 0.015–0.035 stratified, 0.13–0.30 wild), so the Chinchilla form may be misspecified for Meta's curves.
      - The five-fold M\* gap at 3.8e25 is an extrapolation 3.5 orders of magnitude beyond the largest budget.
    - *Implication:* the inference-wedge module should carry *both* Meta technologies. The "flagship on its own path" calculation (w = 0.97 in SYNTHESIS) uses the A2 law.

12. **[Rev, corrected] For the Marin ladders, the Approach-2 vs primal discrepancy is explained by the design.** About three-quarters of it is finite-grid parabola bias and a quarter is the budget's FLOP accounting (nominal C ≠ 6ND). Design-consistent p = 0.33–0.53, vs analytic p = 0.03. With 2 decades of compute, Approach 1 cannot separate E from γ.
    - *Evidence:* `m1_chinchilla_labs_duality_tests.csv`, `m1_chinchilla_labs_technology.csv`, **[Rev]** `m1_chinchilla_labs_design_decomp.csv`. Slope gap, Comma: 0.060 = 0.041 grid + 0.015 FLOP accounting + 0.005 residual.
    - *Caveat:*
      - The Marin α ≈ 0.65 is far from Chinchilla's; exponents are recipe-specific.
      - **[Rev]** The builder's "entirely finite-grid bias" wording was wrong.
      - The analytic rejections are fragile too (bootstrap-calibrated p = 0.15–0.32).
      - The frontier restriction is rejected for all three corpora (χ² 59–80; calibrated p = 0.005–0.065).

---

## 5. Robustness and failures

- **Reproducibility check.**
  - **[Rev]** The reviewer ran `run.py` (all 8 original stages) from scratch into an empty output root in 47 min, without errors. Every CSV, `.npy` and `.npz` was bit-identical to the builder's stage-by-stage outputs, so the pipeline is deterministic.
  - The final outputs in the project root are that run plus the new `checks` stage and a `report` re-run in the project root (for root-relative registry paths).
  - Full-size bootstrap counts are set in `run.py`: horse 400; selection 200 (swing 400); duality 300 + 400 wild; fdep 200; spec 300; labs 200.
- **Exact reproduction of Besiroglu's numbers failed with the Huber estimator.** The explanation (they used a Huber likelihood with a free scale, which is LAD) is documented in H1. `sl.py` is not buggy.
  - But its docstring claim that it "reproduces Besiroglu et al. (2024) exactly" is imprecise: `sl.fit_chinchilla` returns the true Huber optimum (A = 477.8, B = 2143.4, α = 0.3473, β = 0.3672), while `sl.BESIROGLU` holds the published LAD-type values.
  - Downstream modules should state which one they use.
- **The cluster bootstrap with 9 budgets is fragile.** Extrapolated objects have enormous cluster CIs: M*(5.76e23) [3.2, 126]; M*(1e26) cluster SD 183. Any headline M* at frontier scale from Chinchilla alone is not credible.
- **The fixed-design wild bootstrap misbehaves on these data.** Rademacher weights flip the sign of the gross outliers, creating runs 0.3 log points *below* the fit. The SD of the Approach-2 slope then rises from 0.020 to 0.096 (0.46 at n = 245).
  - I therefore use the stratified bootstrap with an outlier-robust covariance as the main test. The SD-based results are in the CSV (`cov == "sd"`).
  - The SD-based stratified test is *less* powerful for Hoffmann's design-consistent null (p = 0.16) than the robust one (p = 0.0004), because Hoffmann's off-centre optimum makes the resampled parabola argmins heavy-tailed. **The Hoffmann rejection is therefore sensitive to the covariance estimator.** The robust stratified and robust wild versions both reject at 5%.
  - **[Rev]** Pairing a heavy-tail-robust covariance with χ² critical values is anti-conservative when the heavy tails are real sampling variability. With critical values from the bootstrap distribution of the same statistic, the covariance choice no longer matters: Hoffmann p = 0.060 for both. The classical fixed-design F gives p = 0.040. See H6.
- **The frontier (cost-function) restrictions are rejected for every technology,** including the preferred refit. The Approach-1 frontier with free E is poorly determined: at n = 240, E_A1 = 1.76 and γ_A1 = 0.168, vs γ = 0.181 when E is fixed at the primal's E. The SE of E_A1 was not saved in the final run. I did not find a specification that passes the frontier-level restriction.
  - **[Rev]** After bootstrap calibration the frontier rejection holds under the stratified bootstrap (p = 0.003–0.037 across technologies), but not under the wild bootstrap (design nulls: p = 0.15–0.70).
- **The first system LR was non-nested and gave a negative statistic** (−0.90). It was replaced by a properly nested LR (`m1_chinchilla_system_lr.csv`); the old column remains in `m1_chinchilla_system.csv` for transparency.
- **The n = 245 sample fails most tests** (duality path test, system LR, least-squares estimates). Its results are reported, not hidden, but they are driven by 5 runs.
  - **[Rev]** The path-test and system-LR failures at n = 245 are borderline once calibrated: bootstrap p = 0.054–0.080, classical F p = 0.041, small-sample LR p = 0.048.
- **The on-path demonstration uses a band, not exact optimal points.** The 9-argmin "sample" has n = 9, so any 5- or 6-parameter fit is close to exact interpolation. The argmin-sample SEs (α: 3.5) partly reflect small n, not only functional dependence.
- **[Rev] Size-matched control for the on-path band** (`m1_chinchilla_fdep_sizecontrol.csv`): 4 random n = 41 subsamples of the full design, 100 pairs draws each.
  - The profile flatness and the Jacobian collapse are design effects: random subsamples have LR up to 79–101 and condition numbers 69–95.
  - The SE inflation is about 2.2–2.3× for α and a net of n, not 6.5–7×.
- **[Rev] Warm-started bootstrap replications are not a problem** (`m1_chinchilla_warmstart_check.csv`). Re-fitting 40 pairs and 40 cluster draws (Huber, n = 240) with the 432-start FAST_GRID added never found a lower objective, and the exponent SDs are unchanged.
- **The κ profile is non-smooth at σ\* > 0.9 in the Huber panel** (optimizer noise at large κ; κ → 20 at σ* = 0.95). This is irrelevant for inference: the LR there is > 100.
- **Llama 3 loss units are unknown.** The report calls it "negative log-likelihood on a held-out validation set"; values of 0.69–0.93 suggest a normalization other than nats/token. E and K for Llama 3 are therefore in unknown units; α, β, a and σ* are unit-free.
- **Units for Chinchilla.** N includes embeddings. D is constructed as C/(6N), where C is read from a figure with a 0.018-dex systematic offset. Loss is quantized.
  - All objects use the digitized C, consistent with D = C/(6N) as in Besiroglu. Budget clustering uses the offset-corrected C.
  - Relative to Hoffmann's nominal budgets, C is 4% lower. This shifts path levels ln N*(C) by about a × 0.04 ≈ 0.02, small next to the 0.113 gap in H6. It does not affect exponents.

---

## 6. Open issues

1. **Wall time.** A full from-scratch run took 47 min (8 original stages) + 4 min (`checks`) under load average 22–40.
   - The most expensive steps are the LAD bootstrap in the horse race (3 Nelder–Mead restarts × up to 40k evaluations per draw) and the system-estimator bootstraps in duality and labs.
   - Reducing the LAD restarts to 1 in the bootstrap should bring the idle-machine run under 30 min. This is not done because it would change the reported draws.
   - A `--quick` end-to-end smoke test of all stages into a scratch output root (`M1_OUTPUT_ROOT`) is described in Section 5.
2. **Replicate the κ < 1 finding on Farseer / Gadre (module m2).** If it holds, the paper's σ* headline should be reported with κ free, or κ = 1 should be presented as a maintained restriction.
3. **Decide which Meta technology the wedge module (H2) uses.** A2 (Meta's law: M*(3.8e25) = 41) and the primal on the same points (M* = 7.9) imply very different inference-demand numbers for Llama-3-8B/70B. At minimum, report both.
4. **Loss units of the Llama 3 IsoFLOP figure** should be checked against the report text before any cross-dataset comparison of E or K.
5. **SYNTHESIS ledger corrections:**
   - Besiroglu's 5 excluded points are "the 5 highest-loss runs"; only 4 have D/N < 0.4.
   - The Chinchilla design variance "1.12 vs 1.84" is the n = 245 sample. At n = 240 it is 0.90 transverse vs 1.83 along.
   - "Besiroglu refit" objects computed with `sl.BESIROGLU` (published, LAD-type) differ slightly from the Huber refit: w(70B) 1.031 vs 1.040; M*(5.76e23) 18.4 vs 17.9.
6. **Not done:**
   - PPML / Gamma-QMLE in levels (the log-of-gravity comparison);
   - Andrews–Cheng weak-identification CIs for E and A;
   - bootstrap of the profile-likelihood sets (the profile LR sets use asymptotic χ²₁ critical values).
7. **Marin and (Mis)Fitting M\* extrapolations** are shown only to illustrate primal/dual divergence and should not be used.
8. **[Rev] Open from the review:**
   - Which bootstrap is right for IsoFLOP argmins is unresolved. The within-budget pairs bootstrap resamples the experimenter's design, which creates duplicated N values and heavy-tailed parabola argmins; the wild bootstrap flips gross outliers. A parametric (residual) bootstrap from the fitted primal under the null, with the design held fixed, would give an exact-design calibration of the duality tests and of the nested system LR. It is not done.
   - The OLS s.e. of the A2 slope (from the between-budget scatter) exceeds the stratified-bootstrap s.e. for Llama 3 (0.018 vs 0.011). This suggests budget-level shocks that the within-budget bootstrap misses. The registry now uses the larger one for the A2A1 rows.

**Citation keys used:**
- besiroglu2024chinchilla, hoffmann2022training, czech2026problems, grattafiori2024llama;
- li2025misfitting, leonledesma2010identifying, klump2007factor, cameron2008bootstrap;
- kaplan2020scaling, koenker1978regression.

New keys, with verified entries in `lit/bib/extra_m1_chinchilla.bib`:
- `hausman1977social`: Econometrica 45(4):919–938, DOI 10.2307/1912682;
- `marin2026ladders`: W&B report, URL resolved; report date not verified;
- `czech2026llama3isoflop`: GitHub repository, created 2026-03-11.
