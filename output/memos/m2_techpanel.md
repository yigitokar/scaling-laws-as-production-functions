# Memo — module m2_techpanel: the parameter–data technology across public sweeps, and the neutrality of data quality

Module owner: m2_techpanel (Claude). Date: 2026-09-23. Entry point: `code/analysis/m2_techpanel/run.py`. One command regenerates every number, table and figure below from `data/raw/`, with fixed seeds. `--outputs-only` re-draws the tables and figures from the stage caches.

**Revised after independent review (2026-09-23; details in `output/memos/m2_techpanel_review.md`).** The reviewer re-ran the module from scratch (every table and bootstrap file reproduced byte for byte), then fixed the code and re-ran it in full. What changed:
- **Rank-one test.** The rank-one curvature test was re-done with E from the q family. The old "saddle-shaped, non-separable" result was an artifact of using the Chinchilla Ê (H1d, claim 5).
- **Out of sample.** The Farseer out-of-sample fits now start only from training-sample values.
- **New robustness variant.** A Muennighoff D/N ≥ 0.4 variant was added.
- **New summary files.** The local-σ slopes and the Farseer Eq. 3 M* are now written by the code rather than computed by hand.
- **Wald cross-check.** A Wald cross-check of the neutrality tests was added.
- **Softened claims.** Several claims were softened: the capital–labor comparison, the pointwise local-σ evidence, and the Lemma 1 reading of the DataDecide tilts.

Notation follows `paper/notes/model_spec.md`: L = E + A N^−α + B D^−β, a = β/(α+β), γ = αβ/(α+β), σ* = 2/(2+α+β), M = D/N, C = 6ND.
- **Estimated** = computed here from public run-level data.
- **Assumed** = a maintained functional form or convention, stated each time.
- **Literature** = a number taken from a paper, with its bib key.

---------------------------------------------------------------------------------------------------

## 1. Headline findings

### H1a. Under the Chinchilla functional form, σ* is 0.72–0.83 in every independent sweep

Under this form, parameters and data are gross complements, and closer to Cobb–Douglas than the conventional capital–labor estimates. H1b shows that this comparison depends on the maintained outer exponent q = 1. Reference estimator: Huber-LSE on log loss (δ = 10⁻³). Standard errors are from a pairs or cluster bootstrap. File: `output/tables/m2_table3_technology.csv`; LaTeX: `m2_table3_technology.tex`.

| Sweep (runs) | σ* (s.e.) [95% CI] | α | β | a (s.e.) | M*(10²¹) [95% CI] |
|---|---|---|---|---|---|
| Chinchilla, Epoch digitization (240) | **0.737** (0.006) [0.724, 0.747] | 0.347 | 0.367 | 0.514 (0.020) | 21.4 [15.1, 28.6] |
| Farseer (404) | **0.772** (0.010) [0.750, 0.790] | 0.281 | 0.312 | 0.526 (0.022) | 24.8 [18.8, 32.9] |
| Gadre et al., C4 (34) | **0.795** (0.020) [0.748, 0.827] | 0.253 | 0.262 | 0.509 (0.072) | 6.9 [1.3, 88] |
| Gadre et al., RedPajama (35) | **0.795** (0.013) [0.775, 0.824] | 0.243 | 0.273 | 0.529 (0.045) | 5.8 [1.2, 13.3] |
| Gadre et al., RefinedWeb (35) | **0.796** (0.015) [0.762, 0.823] | 0.223 | 0.290 | 0.566 (0.055) | 3.4 [1.2, 18.9] |
| Gadre, 3 corpora, common exponents (104) | **0.795** (0.011) [0.780, 0.819] | 0.245 | 0.270 | 0.524 (0.040) | 5.2 (median) |
| OLMo ladder (30) | **0.825** (0.034) [0.763, 0.895] | 0.267 | 0.157 | 0.370 (0.177) | 48 [5.9, 298] |
| Muennighoff et al., single epoch (33) | **0.735** (0.031) [0.657, 0.787] | 0.336 | 0.387 | 0.535 (0.126) | 60 [9.3, 281] |
| DataDecide, 25 recipes, common exponents (1,100 runs; 21,888 checkpoints) | 0.828 (0.005) [0.820, 0.838] ‡ | 0.284 | 0.131 | 0.315 (0.037) | ‡ |

‡ DataDecide levels are descriptive only (schedule artifact, Section 2.1).

- **Gaussian NLS gives the same picture** (Panel B): σ* runs from 0.723 (Chinchilla, Muennighoff) to 0.833 (OLMo ladder).
- **Local σ barely moves within a sample.** Over each sample's support it stays within ±0.02 of σ*; Farseer's range is 0.766–0.777.
- **Comparison with capital and labor.** Every Chinchilla-form point estimate lies above the conventional capital–labor range of 0.4–0.7 [chirinko2008sigma; oberfield2021micro; klump2007factor].
  - The margin is small at the bottom: Chinchilla 0.737 and Muennighoff 0.735, whose 95% CI [0.657, 0.787] overlaps the range.
  - The q-free estimates of H1b (0.51–0.71) lie *inside* the range. So "far above capital–labor" is not a robust statement.
- **Sanity check against the literature.** The Chinchilla reference row gives σ* = 0.737 and a = 0.514. The ledger's Besiroglu refit has 0.737 and 0.513 [besiroglu2024chinchilla]. Module m1 gets the same point estimates on the same data. Our α, β and E match Besiroglu's published values within 0.002, and A, B within 1–3%.
- **Second literature check.** Muennighoff et al.'s own C4 fit ties the exponents (α = β = 0.353 on 54 runs, ledger), which implies σ = 0.739 [muennighoff2023scaling]. On their 33 single-epoch runs we get σ* = 0.735 unrestricted and σ_CES = 0.719 with α = β imposed.
- **Gadre's α = β assumption holds here.** Gadre et al. impose α = β [gadre2024language]. Our unrestricted estimates for their corpora have |α − β| ≤ 0.07, never significant.
- **Muennighoff sample sensitivity (review addition).** Two single-epoch runs put 1.1B and 2.8B parameters on only 100M tokens (D/N = 0.04–0.09). Dropping them, as Besiroglu et al. drop D/N < 0.4 on Chinchilla, has these effects (`m2_robustness.csv`, row `datablations|single_epoch_M04`, 31 runs):
  - σ* moves from 0.735 to 0.726 (0.016);
  - a moves from 0.535 to 0.360 (0.086);
  - M*(10²¹) moves from 60 to 139 [28, 867].

### H1b. The level of σ is not robust to the maintained outer exponent

Every sweep rejects q = 1 (Chinchilla). With q free, σ* falls to 0.51–0.71.
- **The generalized form.** L = E + (A′N^−α′ + B′D^−β′)^q. It nests Chinchilla (q = 1) and the Kaplan form (q = α_D ≈ 0.1) [kaplan2020scaling].
- **Why σ*_q uses the inner exponents.** Isoquants depend only on the inner aggregator, so σ*_q = 2/(2+α′+β′). The wedge ε_N/ε_D = α′u′/(β′v′) also depends only on the inner aggregator, because q cancels.
- File: `m2_spec_tests.csv`; `m2_generalized_q.csv` has all generalized-form parameters with bootstrap SEs.

| Sweep | q̂ (s.e.) | σ*_q [95% CI] | LR(q = 1), Gaussian NLS |
|---|---|---|---|
| Chinchilla | 0.774 (0.060) | 0.701 [0.670, 0.722] | 46.2 |
| Farseer | 0.472 (0.014) | 0.710 [0.705, 0.716] | 999 |
| Gadre C4 | 0.353 (0.080) | 0.607 [0.471, 0.678] | 19.5 |
| Gadre RedPajama | 0.395 (0.139) | 0.623 [0.531, 0.746] | 16.0 |
| Gadre RefinedWeb | 0.355 (0.120) | 0.594 [0.489, 0.715] | 22.4 |
| OLMo ladder | 0.223 (0.022) | 0.544 [0.497, 0.577] | 77.0 |
| Muennighoff, 1 epoch | 0.422 (0.104) | 0.511 [0.435, 0.629] | 16.0 |

- **Every test rejects.** Each LR has p < 0.001 against χ²₁, and the bootstrap Wald tests of q = 1 also have p < 0.001 in all 7 sweeps.
  - The q bootstrap warm-starts each draw at q̂. The reviewer re-ran 100 draws with 6 additional starts that include q = 1, on Gadre C4 and RefinedWeb, the OLMo ladder and Muennighoff.
  - s.e.(q̂) and the σ*_q intervals were unchanged: s.e.(q̂) moved by at most 0.005.
  - No draw reached q ≥ 1.
- **The fit gains are large.** The RMSE of log loss falls from 0.0085 to 0.0025 on Farseer and from 0.0052 to 0.0014 on the OLMo ladder.
- **Cross-module consistency.** The Chinchilla row matches module m1 (κ̂ = 0.774, σ*_κ = 0.700).
- **Interpretation.** This is model_spec Proposition 1's DMR/κ point, seen in real data:
  - Designed off-path variation *does* identify q.
  - Where it does, it rejects the restriction that makes σ* look stable across sweeps.
  - The "σ ≈ 0.74 across refits" in the SYNTHESIS ledger is therefore partly a product of the maintained q = 1.

### H1c. On Farseer, the best-designed sweep, estimators that do not impose q = 1 agree on σ ≈ 0.69–0.71, below Chinchilla's 0.77

Files: `m2_farseer_local_sigma.csv`, `m2_farseer_local_sigma_summary.csv` (every summary below); figure `m2_farseer_local_sigma`.
- **Nonparametric local σ.** A kernel-weighted local quadratic in (ln N, ln D) on ln L; no E is needed, because σ is ordinal.
  - Median over 42 interior grid points: **0.690**; bootstrap 95% CI of the median [0.679, 0.699].
  - Range across points 0.61–0.99; 10th–90th percentile 0.64–0.83.
  - The median is stable to the bandwidth: 0.694 at 1.5× and 0.707 at 2× the CV bandwidth.
- **Farseer's own form.** Eq. 3 of [li2025predictableb] implies a local σ with median 0.706 (range 0.68–0.77).
- **q family.** σ*_q = 0.710.
- **Chinchilla vs the nonparametric median.** Chinchilla's σ (0.767–0.776 over the grid) lies outside the bootstrap CI of the nonparametric median [0.679, 0.699] and outside the q-family CI [0.705, 0.716].
  - It lies inside the pointwise 95% nonparametric band at only 4 of 42 points (10%).
  - *Review caveat.* The pointwise bands come from a pairs bootstrap at a fixed (CV, lower-edge) bandwidth, so they capture sampling variance but not smoothing bias.
  - The pointwise estimates move more across neighbouring model sizes than the bands allow. At D/N = 10: 0.72 at N = 0.8B, 0.78 at 1.6B, 0.69 at 3.2B, with bands of ±0.02 to ±0.05.
  - Read the pointwise count as descriptive; the median-based comparison is the robust one.
- **Where complementarity may be stronger (suggestive).** Local σ tends to fall with D/N.
  - Partial slope holding ln N fixed (joint OLS over the 42 grid points): −0.031 per unit of ln M [bootstrap −0.040, −0.024].
  - The simple slope is only −0.016 [−0.020, −0.012]. The slope on ln N is −0.048 [−0.069, −0.031].
  - The pattern is not monotone at every size: at N = 0.8B and 3.2B, local σ first rises with D/N.
  - The bootstrap intervals again ignore smoothing bias.
  - Farseer's own Eq. 3 implies a σ that falls with D/N only for small models (0.77 to 0.70 at N = 0.2B).
  - So "complementarity is stronger for over-trained models" is a hypothesis supported mainly at small N, not an established pattern.

### H1d. Specification tests

File: `m2_spec_tests.csv/.tex`.
- **CES (α = β) is not rejected by the bootstrap Wald test in any sweep.**
  - |α̂ − β̂| ≤ 0.11, and p ranges from 0.22 to 0.90.
  - The quasi-LR test (NLS, χ²₁) rejects on Chinchilla (p = 0.001) and is marginal on Farseer (p = 0.052) and the OLMo ladder (p = 0.067).
  - σ_CES = 1/(1+α) under the restriction is 0.72–0.85.
- **Rank-one curvature: close to rank one once E is taken from the q family (corrected in review).**
  - The prediction tested is that ln(L − E) has a rank-one Hessian whose null direction is the expansion path (model_spec Lemma 2). Both the Chinchilla and the q-family forms imply it, *each with its own E*.
  - Statistic: τ = det H/‖H‖²_F from a translog in ln(L − Ê).
  - *With the Chinchilla Ê* (the original test), the statistics look decisive:
    - Farseer τ̂ = −0.258 [−0.344, −0.175], null range [−0.055, 0.055], p ≤ 0.003;
    - Chinchilla τ̂ = −0.079, p = 0.007.
  - **This is an artifact, not evidence of non-separability.** An error in Ê adds a constant to reducible loss, and ln(R + c) is not rank one.
    - A noise-free q-family surface on the Farseer design, analysed with the Chinchilla Ê, gives τ = −0.273. That is the observed value, with no non-separability at all.
    - So the left-hand test only re-detects q ≠ 1.
  - *With Ê from the q family* and a residual-bootstrap null under the fitted q model (E re-estimated in every draw, 300 draws; `m2_spec_tests.csv`, columns `tau_q`, `p_rank_one_q`):
    - Chinchilla τ̂_q = −0.009, p = 0.69.
    - Farseer τ̂_q = 0.014 against a null range of [−0.015, 0.002], p ≤ 0.003. However, the pairs-bootstrap interval of τ̂_q, [−0.001, 0.026], includes 0.
    - OLMo ladder τ̂_q = 0.005, p = 0.053.
    - Gadre and Muennighoff: p = 0.21–0.82.
  - Reading: the log reducible-loss surface is close to rank one. On Farseer, the departure is statistically detectable against one bootstrap but an order of magnitude smaller than the old τ = −0.26. Its sign (τ > 0, a convex rather than saddle-shaped surface) is the opposite of what was reported.
  - This agrees with module m1, which reads its own translog test on Chinchilla as a joint test of (E, rank one) and does not reject at 5%.
  - Power in the 30–35-run sweeps is limited: their q-null ranges are about ±0.06 (Gadre) and [−0.12, 0.10] (Muennighoff). OLMo's ladder is the exception, with [−0.009, 0.005].
- **Farseer's own form wins in and out of sample.** File: `m2_farseer_forms.csv`.
  - Out-of-sample RMSE of log loss on the top compute decile: 0.0026 (Farseer Eq. 3), 0.0052 (q family), 0.0047 (translog with E free) and **0.022–0.025** (Chinchilla).
  - All out-of-sample fits start only from training-sample values. Before review, the Farseer and translog fits were started at the full-sample estimates, which include the test data; the correction changes those RMSEs by at most 0.00012.
  - Chinchilla's out-of-sample bias there is −0.020 to −0.023. It over-predicts loss at the largest budgets.
  - Holding out the 4 largest model sizes: 0.0026 (Farseer) vs 0.0126–0.0138 (Chinchilla).
  - BIC: −3,916 (Farseer) vs −2,670 (Chinchilla).

### H1e. The allocation exponent a and the compute-optimal ratio M* are fragile; σ is comparatively stable within a functional form (across forms, see H1b)

- **Spread across sweeps.** a runs from 0.370 (OLMo ladder) to 0.566 (Gadre RefinedWeb). M*(10²¹) runs from 3.4 to 60, and single-sweep 95% CIs span factors of 1.7 to 70.
- **Estimator sensitivity.** On Gadre RefinedWeb, M*(10²¹) is 3.4 under Huber and 10.4 under NLS.
- **Parameter-count convention.** Counting Farseer's two untied 65,536-row embedding matrices moves a from 0.526 to 0.411 and M*(10²³) from 19.5 to 45.0, but moves σ* only from 0.772 to 0.724 (`m2_robustness.csv`).
- **Output measure.** The OLMo ladder with task BPB as the output gives σ* = 0.763 (vs 0.825 with C4 cross-entropy) and a = 0.353.
- **Sample.** Dropping Muennighoff's two D/N < 0.4 runs moves a from 0.535 to 0.360 and M*(10²¹) from 60 to 139, but σ* only from 0.735 to 0.726.

### H2. Better data is close to Hicks-neutral in reducible loss plus a shift in the asymptote; the factor-biased part is small but moves allocation

Files: `m2_neutrality.csv/.tex`, `m2_neutrality_magnitudes.csv`, `m2_neutrality_groups_{gadre,datadecide}.csv`; figure `m2_neutrality_recipes`. All models are evaluated on the same C4 validation set; details in Section 2.

**Gadre et al.: 3 web corpora (104 runs, 37 (size, multiplier) clusters, 499 wild-bootstrap draws).**
- **Corpora differ.** The one-technology model is rejected: LR = 30.7, p ≤ 0.002.
- **An asymptote shift alone does not suffice.** E-shift only: p = 0.004; it explains 89.9% of the cross-corpus variation in fit.
- **Pure factor-augmenting stories with a common E are rejected.** Hicks, data-augmenting and parameter-augmenting models with common E: p ≤ 0.002 each.
- **Hicks-neutral plus corpus-specific E is not rejected** against the common-exponent model: LR = 0.02, p = 0.90. It explains 96.8% of the variation, against 96.9% for common exponents.
- **Low power.** Data-augmenting + E_r (p = 0.13) and parameter-augmenting + E_r (p = 0.15) are not rejected either.
- **Common exponents are not rejected** against fully separate fits: p = 0.09.
- **Wald cross-check (review addition; `m2_neutrality_wald_check.csv`).** Wald tests of the same restrictions on the common-exponent parameters, using their cell-level pairs-bootstrap covariance (200 draws). Resampling whole cells keeps the smooth misspecification intact, which the sign-flipping wild bootstrap does not.
  - Confirmed with both estimators: Hicks + E_r is not rejected (χ² p = 0.85 NLS, 0.74 Huber), and E-shift only is rejected (p < 0.001 NLS, 0.003 Huber).
  - Not robust: data-augmenting + E_r is *rejected* by the NLS Wald test (p = 0.018; bootstrap-t 0.030) but not by the Huber Wald test (p = 0.77) or the wild bootstrap (0.13).
  - So Gadre does not discriminate between Hicks-neutral and data-augmenting improvements with any confidence.
- **Allocation effect is negligible.** The implied factor tilt ln(A_r/B_r) varies by only 0.02–0.04 across corpora, so compute-optimal D/N differs by only 8–18% across corpora.

**DataDecide: 25 recipes (21,888 checkpoints, 44 (size, seed) clusters).**
- **Every restriction is rejected statistically,** at the smallest attainable wild-bootstrap p (≤ 0.01 with 99 draws; ≤ 0.05 with 19 draws for the common-E models):
  - exponents vary: LR = 705 on 48 df;
  - Hicks + E_r: LR = 145 on 24 df;
  - data-augmenting + E_r: LR = 69;
  - parameter-augmenting + E_r: LR = 738.
  - Every observed LR exceeds the largest bootstrap LR*. The tightest margin is data-augmenting + E_r: 69 vs a maximum of 57.
  - Uncapped refits of the largest LR* draws reproduce them to three decimals, so the 300-evaluation solver cap does not matter.
  - The Wald cross-check with the cell-level pairs-bootstrap covariance of the common-exponent fit (100 draws) also rejects Hicks + E_r (24 df: W = 141, p < 0.001 NLS; W = 51, p = 0.001 Huber), data-augmenting + E_r and parameter-augmenting + E_r (all p < 0.001).
- **The economic content is concentrated in E and a neutral scale.**
  - The asymptote shift alone explains **98.1%** of the cross-recipe variation in fit.
  - Adding a Hicks-neutral scaling of reducible loss brings this to **99.71%**.
  - Factor bias adds 0.05 points (99.76%). Exponent heterogeneity adds the last 0.24 points.
- **σ\* is homogeneous across recipes.** Separate per-recipe fits give σ*_r between 0.790 and 0.839, with a cross-recipe s.d. of 0.010 (Huber).
- **But the bias is systematic and it moves allocation.**
  - Under common exponents, the recipe tilt ln(A_r/B_r) spans 0.22 (NLS) to 0.26 (Huber). That implies compute-optimal D/N differing **2.9–3.4×** across recipes.
  - The bootstrap 95% intervals of the tilt range are [0.13, 0.32] (NLS) and [0.18, 0.47] (Huber) (`m2_neutrality_magnitudes.csv`).
  - The tilt covaries with the asymptote: Spearman(E_r, ln M*_r) = −0.85 [cluster-bootstrap −0.92, −0.55] (NLS); −0.84 [−0.93, −0.45] (Huber).
  - This is not a mechanical artifact of the E–B trade-off in estimation. Within a recipe, the bootstrap correlation of Ê_r with the tilt is *negative* (median −0.43 NLS, −0.79 Huber), the opposite sign to the cross-recipe relation, so estimation noise attenuates the correlation rather than creating it.
  - The aggressively quality-filtered DCLM-Baseline variants (QC FW 3%, QC 7% FW3) have the highest C4 asymptotes, a *data-augmenting* tilt, and the lowest M*. For example, M* for DCLM-Baseline (QC FW 3%) is 2.4× below the median recipe.
  - Their separate-fit allocation exponents are among the highest (a ≈ 0.37 vs 0.315 for the pooled Huber common-exponent fit). But per-recipe CIs are about ±0.08, and Dolma1.6++, the recipe with the *lowest* E_r, has an equally high a, so there is no systematic relation between a_r and E_r (figure panel b).
  - **How to read this against Lemma 1 (revised in review).**
    - Under common exponents, ln M*_r is a decreasing linear function of the tilt. So "data-augmenting tilt ⇒ lower M*" is an identity, not a test of Lemma 1.
    - The empirical content is *which* recipes carry the data-augmenting tilt: the heavily filtered DCLM variants.
    - Whether that is "better data" depends on the quality index. On the common C4 metric these recipes are *worse*: higher E_r, and recipes with lower E_r have higher M*.
    - So the evidence is consistent with Lemma 1 only if aggressive quality filtering is read as a data-augmenting improvement whose C4 asymptote is penalized by distribution mismatch.
    - Quality is not measured independently of the tilt here, and DeepSeek's finding concerns the exponent a [bi2024deepseek], which does not vary systematically here.
- **Robust to output and sample** (49 draws, p ≤ 0.02 at the minimum): the same conclusions hold with the mean cross-entropy over 11 validation sets as output, and on checkpoints with D/N ≥ 20. Shares explained by E + Hicks: 99.53% and 99.71%. Tilt ranges: 0.30–0.59.
- **Maintained assumption behind all DataDecide comparisons (review addition).** Every recipe shares the same sizes, steps and cosine schedule, so the *design* of the schedule artifact is identical across recipes. Its *effect on loss* need not be.
  - If annealing benefits some recipes more than others, part of the estimated tilt and E_r differences is a recipe × schedule interaction, not technology.
  - DataDecide cannot test this: there are no fully annealed runs at several D per size.

---------------------------------------------------------------------------------------------------

## 2. Methods

### 2.1 Data construction

All data are public run-level files. Conventions are recorded row by row in `technology_registry_m2.csv`.

| Dataset | Units / sample | N convention | D | Output (loss units) | Cluster unit |
|---|---|---|---|---|---|
| Chinchilla reference [hoffmann2022training; besiroglu2024chinchilla] | 240 digitized runs (Epoch; the 5 highest-loss points dropped as in Besiroglu) | total (Hoffmann) | C/(6N), imputed by Epoch | MassiveText validation, nats/token (SentencePiece 32k) | run |
| Farseer `1222_full.csv` [li2025predictableb] | 404 runs, 25 N × up to 19 D, one recipe, hyperparameters from Step Law | **non-embedding** (N_add_emb − N = 131,072·h, i.e. two untied 65,536-row embeddings) | tokens processed (single epoch) | **English bits per character** on IntelliValSet (`IntelliValSet_Raw\|en`, the average of web/paper/book), the output Farseer fits | run |
| Gadre et al. [gadre2024language] | 104 runs: C4 34, RedPajama 35, RefinedWeb 35 | **total**, incl. embeddings (`params`); D = 20·multiplier·params exactly | tokens | **C4 validation** loss (174M tokens) for every corpus, nats/token (GPT-NeoX tokenizer); in-distribution Paloma loss as robustness | run; (size, multiplier) cell for pooled and neutrality models |
| OLMo ladder [bhagia2024establishing] | 30 runs: 5 sizes × {0.5, 1, 2, 5, 10}× Chinchilla + 5 reruns; final logged step | OLMo `MODEL_PARAMS`, which excludes only the input embedding (checked: 3,169,537,280 = 16·16·3328² + 100,352·3328 + 65·3328; the last term is the 65 layer-norm weight vectors, missing from the pre-review arithmetic) | `throughput/total_tokens` at the final step | C4-en validation cross-entropy, nats/token (dolma2 tokenizer); task BPB as robustness | (size, multiplier) cell, so reruns cluster with their originals |
| Muennighoff datablations [muennighoff2023scaling] | 229 final losses from the authors' notebook (`return_alloc.ipynb`, `NAMES_TO_VAL_LOSSES`), parsed with the notebook's own code. **Primary: the 33 single-epoch runs (D = U)** | authors' `PARAMS_MAP` (convention not verified) | tokens processed | C4 validation, nats/token (GPT-2 tokenizer) | run |
| DataDecide ppl [magnusson2025datadecide] | 25 recipes × 14 sizes × 3 seeds; 21,888 checkpoints (D/N ≥ 5; step ≤ full-schedule step) | `MODEL_TO_PARAMS`, which excludes only the input embedding (vocab 50,304) | step × batch × 2,048 (batch sizes from `utils/constants.py`) | C4-en validation cross-entropy = ln(perplexity), nats/token; mean over 11 validation sets as robustness | run within recipe; (size, seed) cell for pooled and neutrality models |

Data quirks we handled:

- **Farseer.**
  - The column labeled `D/N` takes 5 values (3.52, 10.05, 28.2, 321.7, 454.9) that are unrelated to D/N. We recompute D/N from `D` and `N` (range 0.31–2,570).
  - Farseer's `C` is M·D, with M = 6N + attention FLOPs per token (M/N = 6.6–8.5). We use C = 6ND everywhere.
  - 4 (N, D) pairs are reruns; both copies are kept.
- **Gadre.**
  - The main output is the C4 validation loss, because the neutrality tests need a common output.
  - The reported token-level 95% CIs imply a sampling s.e. of ln L of only 0.0016 (median). The Chinchilla residual RMSE is 0.023–0.030, so residual variance is roughly 200–350 times the evaluation noise. The residuals are misspecification, not measurement error, which is why CI weights barely matter.
- **Datablations.**
  - ColPret's `tokens_per_epoch` and `epochs` columns are wrong; for example, "1b25" (1.25B unique tokens) is parsed as 25B. We rebuild N, D and U from the model names using the authors' own code.
  - Repetition is handled by restricting the primary sample to single-epoch runs. Robustness adds up to 4 epochs treated as fresh tokens. The decay model is not re-estimated.
- **DataDecide.**
  - Final checkpoints have D = 85–110·N, so ln N and ln D are almost collinear. Even with checkpoints, corr(ln N, ln D) = 0.91 and sd(ln M | ln C) = 0.71 on the unique (size, step) design points (0.73 over all checkpoints), the lowest of all sweeps. This is the ACF/Kricheli functional-dependence problem [ackerberg2015identification; kricheli2026tokens].
  - Only intermediate checkpoints break the collinearity, and they are evaluated **before the cosine learning-rate schedule has finished**. Schedule progress D/D_final is itself a function of D/N, so the data effect is not separable from the schedule effect without functional-form assumptions.
  - We therefore read DataDecide levels (α, β, σ*, a and especially M* in the hundreds) as descriptive, and use DataDecide for *across-recipe* comparisons. The schedule artifact is identical across recipes: same sizes, steps and schedules.
  - The truncated "small aux" seeds of the 530M, 750M and 1B models are kept.
- **Off-path design variation, sd(ln(D/N) | ln C).** Defined as the residual s.d. of an OLS of ln M on ln 6ND; figure `m2_design_planes` shows the designs.

  | Sweep | sd(ln M \| ln C) |
  |---|---|
  | Farseer | 1.97 |
  | Muennighoff (1 epoch) | 1.97 |
  | Gadre | 1.44 |
  | Chinchilla | 1.35 |
  | OLMo ladder | 0.94 |
  | DataDecide | 0.71 (unique (size, step) design points; 0.73 over all 21,888 checkpoints, as in Table 3) |
- **Not used.**
  - *Kricheli et al.* (HF `TPPIsCriticalFor/colinear_scaling_models`, GPL-2.0) was downloaded, but there are no validation losses. The `val_losses` arrays in `training_metrics/*.npz` are empty. `extracted_losses/*.csv` holds only per-epoch mean **training** losses from 3-epoch runs of 5–76M-parameter models. We report no estimates from it; the commands are in `code/data/download_m2_techpanel.sh`.
  - *open-sci-ref* (8 corpora) was not used, for lack of time. It is the natural next neutrality test.

### 2.2 Estimators
- **Reference estimator (Chinchilla form).** Hoffmann/Besiroglu Huber-LSE on log loss (δ = 10⁻³), run through `sl.fit_chinchilla` unmodified. The same machinery also runs Gaussian NLS on log loss.
  - **Starting values.** Chinchilla and Farseer use sl's DEFAULT grid (4,500 starts); the other sweeps use sl's FAST grid (432 starts). Every fit adds 48 "level-preserving" starts, which set u = v = (L̄ − E)/2 at the geometric-mean design point. In the development run, the FAST grid reproduced the DEFAULT-grid optimum for every primary sweep.
  - **Reported objects.** E, A, B, α, β. Normalized A_norm = A·N̄^−α and B_norm = B·D̄^−β, the reducible-loss contributions at the design center; these are far better identified than A and B. Then a, γ, σ*, the range of local σ over the sample points, and M* at C = 10²¹, 10²³ and 10²⁵ (extrapolated for most sweeps).
- **Generalized outer exponent.** L = E + (A′N^−α′ + B′D^−β′)^q, fitted by L-BFGS-B with an analytic gradient from 6 starting values of q around the Chinchilla fit.
- **CES.** α = β imposed, fitted with the panel machinery below.
- **Translog / rank-one.** OLS of ln(L − Ê) on a full quadratic in centered (ln N, ln D); τ = det H/‖H‖²_F. Computed twice: with Ê from the Chinchilla fit (a joint test of q = 1 and rank one) and with Ê from the q family (the test of rank-one curvature; review fix).
- **Farseer Eq. 3.** 9 parameters, NLS on log loss. Starting values follow the authors' multi-round logic: per-N fits of E_N + B_N D^−k_N, then power laws in N for ln E_N, ln B_N and ln k_N. Unlike the authors, we do not fit on finite differences.
- **Local nonparametric σ** [fan1996local]. A Gaussian-kernel local quadratic of ln L in (ln N, ln D).
  - It is evaluated at 42 interior grid points: 7 model sizes × log-spaced D/N, keeping points with effective n ≥ 25 that lie inside each size's observed D range.
  - Bandwidth by leave-one-out CV: h_N = 0.2·sd(ln N), h_D = 0.45·sd(ln D).
  - σ comes from the local gradient and Hessian: 1/σ = 1 − (f_nn f_d² − 2 f_nd f_n f_d + f_dd f_n²)/(f_n f_d (f_n + f_d)).
  - The formula is invariant to f → g(f), so E is not needed. Under Chinchilla it reproduces model_spec's 1/σ = 1 + sβ + (1 − s)α.
- **Neutrality (panel) models.**
  - For corpus or recipe r: L_r = E_r + A_r N^−α_r + B_r D^−β_r.
  - Nested models:
    - (0) one technology;
    - (iv) E_r only;
    - (i) Hicks-neutral: A_r = e^−ω_r A and B_r = e^−ω_r B;
    - (ii) data-augmenting: only B_r varies;
    - (iii) parameter-augmenting: only A_r varies;
    - (i′), (ii′), (iii′): the same with E_r free;
    - common exponents (CE): A_r, B_r and E_r free;
    - (v) all exponents free.
  - Each per-observation parameter is a sparse linear map of θ.
  - Solver: trust-region least squares with an analytic Jacobian, using the same objective for Huber (`loss='huber', f_scale = δ`) and NLS. It matched L-BFGS-B to 7 digits and is 5–30 times faster.

### 2.3 Inference
- **Bootstrap design.** Pairs or cluster bootstrap, warm-started at the full-sample estimate. Cluster units are listed in Section 2.1. Percentile 95% intervals throughout; M* intervals are very skewed.

  | Fits | Draws |
  |---|---|
  | Primary single-sweep fits | 400 |
  | Robustness variants, q family, DataDecide recipes, Gadre CE | 200 |
  | DataDecide CE | 100 |

- **Cluster check.** Clustering Farseer by model size (25 clusters) gives s.e.(σ*) = 0.009, against 0.010 when clustering by run.
- **Rank-one test.** The null distribution of τ̂ comes from a **residual bootstrap under the fitted model**: 300 draws, each re-estimating E and the translog. This is done once under the Chinchilla model and once under the q model (E re-estimated by the q family). A global quadratic only approximates these surfaces, so τ = 0 would be the wrong null. It is not wrong by much: the null medians are −0.001 to −0.04 (Chinchilla) and −0.045 to −0.001 (q family).
- **Neutrality tests.** Quasi-LR = n ln(SSR_restricted/SSR_unrestricted), from NLS on log loss. p-values come from a **wild cluster restricted bootstrap** [cameron2008bootstrap]:
  - y* = ŷ₀ + w_c ê₀, with Rademacher weights by cluster;
  - both models are re-fitted on y*, the unrestricted fit starting from the restricted solution;
  - Gadre: 37 (size, multiplier) cells, 499 draws. DataDecide: 44 (size, seed) cells, 99 draws for the key tests and 19 for the dominated common-E models (their refits are slow).
  - Bootstrap refits cap the solver at 300 evaluations. A cap could bias LR* in either direction: an unconverged restricted refit inflates it, an unconverged unrestricted refit deflates it. The pre-review claim that the cap makes p conservative was therefore not justified. In review, uncapped refits of the three largest LR* draws (DataDecide, data-augmenting + E_r and Hicks + E_r) reproduced them to three decimals, and tight refits of the common-exponent pairs-bootstrap draws reproduced the capped ones exactly.
  - Cross-check: Wald tests of the same linear restrictions with the cluster-pairs bootstrap covariance of the common-exponent fit (`m2_neutrality_wald_check.csv`; only where draws ≥ 3 × restrictions).
  - "≤" in the tables marks the smallest attainable value, 1/(B+1).

---------------------------------------------------------------------------------------------------

## 3. Table and figure inventory

All paths are under `output/`. Every table has a `.csv`; the paper-ready ones also have a `.tex` (booktabs + threeparttable).

| File | Content |
|---|---|
| `tables/m2_table3_technology.{csv,tex}` | **Paper Table 3.** The technology by dataset (E, α, β, a, γ, σ*, σ range, M*(10²¹), sd(ln M \| ln C); last two columns q̂ and σ*_q). Panel A Huber-LSE, Panel B NLS; bootstrap SEs. Landscape. |
| `tables/m2_spec_tests.{csv,tex}` | Specification tests per sweep: CES (Wald, LR, σ_CES); outer exponent q (Wald, LR, σ*_q); rank-one curvature τ with Ê from the Chinchilla fit and (review fix) τ_q with Ê from the q family, each with its residual-bootstrap null. Landscape. |
| `tables/m2_neutrality.{csv,tex}` | **Neutrality of data quality.** Nested models, share of cross-corpus variation explained, LR, wild-cluster-bootstrap p. Panels: Gadre, DataDecide. The CSV also has the DataDecide robustness panels (11-set mean output; D/N ≥ 20). |
| `tables/m2_neutrality_magnitudes.csv` | CE α, β, σ*, a; E_r range; tilt range (with cluster-bootstrap CI); M* ratio and wedge ratio exp(tilt range) across recipes; Spearman(L_ref, ln M*) and Spearman(E_r, ln M*) (with CI); cross-recipe s.d. of α_r, β_r, a_r, σ*_r. |
| `tables/m2_neutrality_wald_check.csv` | Review addition: Wald tests of the neutrality restrictions with the cluster-pairs bootstrap covariance of the common-exponent fit (Gadre; DataDecide where draws ≥ 3 × restrictions). |
| `tables/m2_neutrality_groups_*.csv` (gadre, datadecide, datadecide_avg11, datadecide_M20) | Group-level CE parameters (ln A_r, ln B_r, E_r, tilt, M*_r, reference loss) and separate-fit α_r, β_r, a_r, σ*_r. |
| `tables/m2_farseer_forms.{csv,tex}` | Farseer: Chinchilla vs q family vs translog vs Farseer Eq. 3 (in-sample RMSE, BIC, two out-of-sample splits). |
| `tables/m2_farseer_local_sigma.csv`, `m2_farseer_local_sigma_bw.csv`, `m2_farseer_local_sigma_summary.csv` | Nonparametric local σ at 42 grid points (bootstrap bands, elasticities, Chinchilla- and Farseer-implied σ); bandwidth sensitivity; summaries (median with CI, band coverage, OLS slopes on ln M and ln N with bootstrap CIs). |
| `tables/m2_farseer_eq3_Mstar.csv` | Review addition: compute-optimal N*, D*, M* implied by the fitted Farseer Eq. 3 at C = 10¹⁹ to 10²³. |
| `tables/m2_robustness.{csv,tex}` | Robustness: N incl. embeddings, training-loss output, clustering by size (Farseer); in-distribution eval, non-embedding N, CI-weighted NLS (Gadre); task-BPB output (OLMo); ≤4 epochs and D/N ≥ 0.4 (Muennighoff). |
| `tables/m2_generalized_q.csv` | Generalized-form (q free) parameters with bootstrap SEs, 7 sweeps. |
| `tables/m2_datadecide_recipes.csv` | Separate Chinchilla fits for the 25 DataDecide recipes (Huber and NLS; run-level bootstrap). |
| `tables/technology_registry_m2.csv` | **Registry**, 102 rows (incl. the review's Muennighoff D/N ≥ 0.4 variant): dataset, subset, estimator (huber / nls / huber_q), role, E, A, B, α, β and SEs, a, γ, σ*, A_norm, B_norm, σ range, M* with CIs, sd_offpath, n, clusters, N convention, D definition, loss units, eval set, bootstrap-draw file, q, notes. **Downstream rows for the inference-wedge module:** `farseer/all` and `gadre/RefinedWeb` (huber and nls; `huber_q` rows give the generalized form, for which w = α u/(β v) uses the inner parameters). |
| `data/processed/m2_techpanel/boot/*.npy` + `boot/index.csv` | Bootstrap draws. Columns: (E, A, B, α, β), or (…, q) for huber_q, or θ_CE for panel models. |
| `data/processed/m2_techpanel/panel_*.csv` | Harmonized panels. Local only: Farseer and analyzing-chinchilla have no license, so do not redistribute. |
| `figures/m2_fig3_sigma_forest.{pdf,png}` | **Paper Figure 3.** σ* with 95% CIs by sweep and estimator (Chinchilla form Huber/NLS; q free; Farseer nonparametric median). Panel (b): the 25 DataDecide recipes. The capital–labor range 0.4–0.7 is shaded. |
| `figures/m2_farseer_local_sigma.{pdf,png}` | Farseer local σ against D/N for 4 model sizes: nonparametric with bands vs Chinchilla- and Farseer-implied σ. |
| `figures/m2_neutrality_recipes.{pdf,png}` | (a) Recipe factor tilt (ln M*_r relative to the median) against asymptote E_r, for DataDecide and Gadre. (b) DataDecide separate-fit allocation exponents a_r against E_r. |
| `figures/m2_design_planes.{pdf,png}` | The six designs in the (log N, log D) plane with isocost lines and sd(ln M \| ln C). |

---------------------------------------------------------------------------------------------------

## 4. Claims for the paper

Each claim gives its evidence (file and number) and its caveat.

1. **Parameters and data are gross complements (σ < 1) in every public sweep and under every functional form we fit.**
   - *Evidence.* Chinchilla-form σ* = 0.735–0.828 (Huber) across 7 independent sweeps and 2 pooled panels, with bootstrap s.e. of 0.005–0.034 (Table 3, `m2_table3_technology.csv`). Every 95% CI lies above 0.7 except Muennighoff's single-epoch sweep (0.657–0.787). The q-free estimates are 0.51–0.71, and all upper CI bounds are below 0.75.
   - *Caveat.* Relative to capital and labor (0.4–0.7), the Chinchilla-form estimates sit just above the range, and the q-free estimates sit inside it. Do not write "far less complementary than capital and labor"; see claim 2.
2. **The level of σ depends on a functional-form restriction that the data reject.**
   - *Evidence.* The outer exponent q = 1 is rejected in all 7 sweeps (p < 0.001; q̂ = 0.22–0.77). The generalized σ*_q = 0.51–0.71 (`m2_spec_tests.csv`).
   - *Interpretation.* The paper should present σ as a range. Under the maintained Chinchilla form it is about 0.74–0.83. Under forms that let the data choose the curvature it is about 0.5–0.7, which overlaps the top of the capital–labor range. The DMR non-identification in model_spec Proposition 1 is empirically first-order.
   - *Caveat.* σ*_q is imprecise in the 30–35-run sweeps (CIs about ±0.1). Only Farseer pins it down: 0.710 [0.705, 0.716].
3. **On the best-designed sweep (Farseer; 404 runs, near-factorial), three estimators that do not impose q = 1 agree: σ ≈ 0.69–0.71.**
   - *Evidence.* Nonparametric local σ median 0.690 [0.679, 0.699]; Farseer Eq. 3 median 0.706; q family 0.710 (`m2_farseer_local_sigma.csv`, `m2_generalized_q.csv`).
   - The Chinchilla-implied 0.77 lies outside the bootstrap CI of the nonparametric median and of σ*_q.
   - *Caveat.* Local second derivatives are noisy at the design boundary; the pointwise range is 0.61–0.99. The median is stable to the bandwidth (0.690–0.707). The pointwise bands ignore smoothing bias, so the "inside the band at 4 of 42 points" statistic should not be used as a test.
4. **(Suggestive only) Complementarity may be stronger in the over-trained region.**
   - *Evidence.* Holding ln N fixed, Farseer's nonparametric local σ falls by 0.031 per unit of ln(D/N) [bootstrap −0.040, −0.024]. The simple slope is −0.016 [−0.020, −0.012] (`m2_farseer_local_sigma_summary.csv`).
   - *Relevance.* For the inference wedge, Chinchilla-form elasticities may overstate substitutability where released small models sit.
   - *Caveat.* One recipe, one tokenizer, BPC output. The profile is not monotone at every model size, the intervals ignore smoothing bias, and Farseer's Eq. 3 reproduces the decline only for small N. Do not state this as a finding without these qualifiers.
5. **CES (α = β) is a statistically acceptable restriction, and the log reducible-loss surface is close to rank one once E is estimated with a flexible outer exponent (revised in review).**
   - *Evidence.* The CES Wald p-values are 0.22–0.90 in all sweeps.
   - Rank-one with Ê from the q family (`m2_spec_tests.csv`, `tau_q`, `p_rank_one_q`): Chinchilla τ̂_q = −0.009 (p = 0.69); Gadre and Muennighoff p = 0.21–0.82; OLMo 0.005 (p = 0.053).
   - Farseer τ̂_q = 0.014 (p ≤ 0.003 against the residual-bootstrap null, but the pairs 95% interval [−0.001, 0.026] includes 0).
   - **Do not claim that separability or rank-one curvature is rejected, or that the surface is saddle-shaped.** The pre-review τ̂ = −0.26 (Farseer) and −0.08 (Chinchilla) came from using the Chinchilla Ê; they are a re-detection of q ≠ 1.
   - *Caveat.* The LR version of the CES test rejects on Chinchilla (p = 0.001). The small sweeps have little power for either test. τ is sensitive to Ê.
6. **Chinchilla extrapolates poorly in compute.**
   - *Evidence.* On Farseer's top compute decile, its out-of-sample log-loss RMSE is 0.022–0.025, with bias −0.02. It is 0.0026 for Farseer's own form and 0.005 for the q family (`m2_farseer_forms.csv`).
   - *Caveat.* Farseer's form has 9 parameters. Its advantage holds in two out-of-sample splits, but we did not test it on other sweeps.
7. **The allocation exponent and M\* are not portable across sweeps or conventions; σ is.**
   - *Evidence across sweeps.* a = 0.37–0.57, and M*(10²¹) = 3–60 with CIs up to 70-fold (Table 3).
   - *Evidence across conventions.* Counting embeddings moves Farseer's a by −0.115 and its M*(10²³) from 19.5 to 45, but moves σ* by only −0.048 (`m2_robustness.csv`).
   - *Caveat.* Parameter-count conventions differ across sweeps (Section 2.1), so cross-sweep comparisons of a mix technology and measurement.
8. **Across corpora and recipes, "better data" mostly shifts the asymptote and scales reducible loss neutrally.**
   - *Evidence.* E_r shifts explain 90% (Gadre) and 98% (DataDecide) of the cross-corpus variation; adding a Hicks-neutral scale brings this to 96.8% and 99.7% (`m2_neutrality.csv`).
   - Gadre does not reject Hicks-neutral + E_r (p = 0.90) or common exponents (p = 0.09).
   - *Caveat.* E_r shifts are relative to the C4 validation set, i.e. partly distribution match, not "quality" per se. Gadre has only 3 corpora.
9. **Recipe differences are measurably factor-biased in DataDecide, the heavily filtered DCLM variants carry the data-augmenting tilt, and the tilt matters for allocation and for the revealed wedge.**
   - *Evidence.* In DataDecide, pure Hicks + E_r is rejected (LR = 145, wild p ≤ 0.01; pairs-bootstrap Wald p ≤ 0.001).
   - Recipe tilts ln(A_r/B_r) span 0.22–0.26 (C4 output; 0.30–0.59 in the robustness panels). That implies M* varying 2.9–3.4× across recipes (DCLM-Baseline QC FW 3% has M* 2.4× below the median).
   - The same tilt range implies a revealed wedge ŵ = ε_N/ε_D that varies 1.25–1.29× across recipes at the same (N, D) (up to 1.8× in the robustness panels; Gadre: 1.02–1.04×).
   - *Implication for the paper.* Model_spec Proposition 4's warning is quantitatively relevant: recipe-specific factor bias can masquerade as roughly ±11–14% of w around the median recipe. Across Gadre's web corpora it is negligible.
   - *Caveat.* DataDecide M* levels are unreliable (schedule artifact, a ≈ 0.32–0.36). Only the relative tilts are interpretable, and only if the schedule artifact affects all recipes alike, which cannot be tested here.
   - Whether the tilted recipes are "better data" is not identified. They have the *worst* C4 asymptotes. Under common exponents, "data-augmenting tilt ⇒ lower M*" is an identity, so this is not independent support for model_spec Lemma 1.
   - Gadre cannot discriminate between Hicks-neutral and data-augmenting improvements (NLS vs Huber Wald tests disagree).
10. **Observational analogues: designed experiments are needed.**
    - *Evidence.* The design statistic sd(ln M | ln C) ranges from 1.97 (Farseer, Muennighoff) to 0.71–0.73 (DataDecide); DataDecide's final checkpoints are exactly collinear (D ≈ 100N).
    - Farseer, with both the most off-path variation and the most runs, delivers by far the tightest σ*_q (s.e. 0.003). DataDecide can only be used through within-run checkpoints.
    - Precision also depends on sample size and noise, not design alone. Muennighoff has Farseer's design spread (1.97) but 33 runs and the widest σ*_q interval [0.435, 0.629]. The OLMo ladder has little spread (0.94) but a fairly tight interval [0.497, 0.577].
    - *Caveat.* This is descriptive support for module m6's Monte Carlo, not a test. The six sweeps do not isolate the design effect.

---------------------------------------------------------------------------------------------------

## 5. Robustness and failures

- **Estimator choice (Huber vs NLS).** It moves σ* by at most 0.015 (Chinchilla 0.737 vs 0.723), but moves M* a lot (Gadre RefinedWeb: 3.4 vs 10.4; Muennighoff: 60 vs 87). The Huber point estimate is essentially a LAD fit.
- **N convention.** Non-embedding N in Gadre raises σ* by 0.01–0.015. Including embeddings in Farseer lowers σ* by 0.048 and a by 0.115.
- **Output.**
  - OLMo task BPB instead of C4 cross-entropy: σ* 0.763 vs 0.825.
  - Gadre RedPajama in-distribution eval: σ* 0.753 vs 0.795; other corpora ±0.01.
  - Farseer training loss (nats/token) instead of English BPC: σ* 0.785 vs 0.772.
  - σ is ordinal in theory, so these differences reflect sampling noise and functional-form misspecification interacting with the output transform.
- **Clustering.** Farseer by model size vs by run: SEs essentially unchanged. OLMo: reruns are clustered with their originals.
- **Repetition.** Muennighoff with ≤4 epochs treated as unique: σ* 0.732 vs 0.735 (single epoch).
- **q family.**
  - For the small sweeps the inner exponents are large (α′, β′ ≈ 0.6–1.1), A′ and B′ are huge (10⁶ in Gadre RefinedWeb), and E_q is close to the Chinchilla E. q and the inner scale are strongly correlated. σ*_q CIs are ±0.1 there.
  - Farseer's q-fit is tight: q̂ in [0.448, 0.503], and E_q = 0.27 against 0.37 under Chinchilla.
- **Rank-one test.**
  - The null is not exactly τ = 0, because a global quadratic misfits the Chinchilla surface; hence the residual-bootstrap null.
  - **Failure found in review.** The pre-review test computed τ with the Chinchilla Ê. Because q = 1 is rejected, that Ê is off (0.37 vs E_q = 0.27 on Farseer), and ln(R + c) is not rank one even when R is.
    - The large negative τ ("saddle") on Farseer and Chinchilla is reproduced exactly by a noise-free q-family surface: τ = −0.27.
    - With Ê_q the statistic is 0.014 (Farseer) and −0.009 (Chinchilla).
    - The E-free translog (7 parameters) is no help: it puts E at 0.13 and gives τ = +0.17. τ is very sensitive to Ê, so any rank-one statement is conditional on the E estimator.
  - The raw correlation statistic r = h_nd/√(h_nn h_dd) is undefined in 11–17% of null draws for the small sweeps, so we switched to τ. r is still reported in the CSV.
- **DataDecide failures.**
  - The per-recipe levels are contaminated by the unfinished LR schedule. a ≈ 0.27–0.37 and M*(10²¹) in the hundreds or thousands are not credible technology numbers.
  - The final-checkpoint-only design is exactly collinear, and we do not report a fit on it.
  - The common-E wild-bootstrap tests (19 draws) have coarse p-values (≤ 0.05). The key tests use 99 draws (≤ 0.01).
- **Gadre neutrality has little power.** With 3 corpora, data- and parameter-augmenting + E_r are not rejected either by the wild bootstrap (p = 0.13, 0.15). "Not rejecting Hicks" is not strong evidence for Hicks. The NLS Wald cross-check rejects data-augmenting + E_r (p = 0.018) while the Huber Wald test does not (p = 0.77), so the Gadre ranking of the factor-augmenting alternatives is estimator-dependent.
- **Wild bootstrap vs asymptotics.** Gadre's E-shift-only model is rejected by the wild bootstrap (p = 0.004) even though its LR is small (2.6 on 4 df). Residuals are dominated by misspecification that is common across corpora within a cell, so the bootstrap distribution of LR* is tight. Reading this against χ²₄ would be wrong. Worth a sentence in the paper.
  - Review check: a sign-flipping wild bootstrap breaks the smoothness of that common misspecification, which could make LR* too small.
  - The cell-level pairs-bootstrap Wald test keeps it intact and gives the same verdict: W = 42.6 on 4 df NLS, bootstrap-t p = 0.005; Huber W = 16.4, χ² p = 0.003, bootstrap-t p = 0.055.
- **Not done.**
  - open-sci-ref (8 corpora) neutrality.
  - A q-family or nonparametric σ for the pooled panels.
  - Weak-identification-robust (Andrews–Cheng) intervals for E and A.
  - Kricheli: no validation losses are public.
- **sl.py.** No bug found. One robustness note: `Chinchilla.theta` returns ln E = −inf if a fit drives E to exactly 0 (underflow), which breaks warm starts and emits a RuntimeWarning. We clip ln E ≥ −50 locally in `m2_est.fit_chin`. `sl.fit_chinchilla` on the Epoch data gives A = 477.8, B = 2143.4, α = 0.3473, β = 0.3672, against Besiroglu's published 482.01 / 2085.43 / 0.3478 / 0.3658. These are close, not identical; module m1 gets the same numbers as we do.

---------------------------------------------------------------------------------------------------

## 6. Open issues

1. **Headline for H1.** Given claims 2–3, the paper cannot say "σ ≈ 0.74 is stable across sweeps" without qualification. Suggested wording:
   - σ is between about 0.5 and 0.8;
   - it is about 0.74–0.83 under the Chinchilla restriction and about 0.7 on the best factorial design once curvature is freed;
   - it is always gross complements: above the capital–labor range of 0.4–0.7 under the Chinchilla restriction, inside it once curvature is freed.

   The writers and the lead author should decide how to frame this.
2. **Which σ feeds the inference wedge?** If the q-family / Farseer technology is used, the wedge uses the inner aggregator (registry rows `huber_q`). Claim 4 suggests, weakly, even lower σ at extreme D/N.
   - Farseer's own form is non-homothetic. The fitted Eq. 3 gives M* = 32, 25, 28, 43 and 82 at C = 10¹⁹ to 10²³ (`m2_farseer_eq3_Mstar.csv`): M* is non-monotone below 10²⁰ and rising above it, and the last two budgets are extrapolations beyond Farseer's 3.5 × 10²¹ maximum.
   - Rising M*(C) is a rival explanation for over-training that module H2 must handle.
3. **Output units.** Farseer is in bits/char and the others in nats/token with 4 different tokenizers. Only unit-free objects (α, β, a, γ, σ, q) should be compared across rows. M* and E are not comparable.
4. **DataDecide.** The schedule artifact could be removed by using WSD-style or fully annealed runs at several D per size, which DataDecide does not provide. The open-sci-ref logs (3 token budgets per size, separately trained) would give a clean second multi-recipe neutrality test.
5. **Unverified.** Muennighoff's `PARAMS_MAP` convention (total vs non-embedding) and the DataDecide tokenizer version. Both were inferred from code, not documentation.
6. **Farseer.** The form is fitted by NLS here, not by the authors' differential piecewise procedure. We did not reproduce their published constants, which are not in the repository in tabular form.
7. **Licenses.** Farseer (and Epoch's analyzing-chinchilla) have no license. Use them, but do not redistribute `data/processed/m2_techpanel/panel_farseer.csv` or `panel_chinchilla.csv`.
