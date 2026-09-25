# rb5_sign: sign identification beyond the designs, rebuilt (round 3, WP4a)

Module `code/analysis/rb5_sign/`, items S1 to S5 of `paper/notes/round3_fixlist.md` (decision D-3). Written 2026-09-25.

One command regenerates every output (deterministic; fixed seeds; one CPU process; about 20 seconds including the review's Monte Carlo):

```
nice -n 10 .venv/bin/python code/analysis/rb5_sign/run.py \
    [--units data/processed/rb5_units/units_primary.csv] [--units56 data/processed/rb2_decisions/units_primary.csv] \
    [--models data/processed/rb2_decisions/clean_models.csv] [--B 9999] [--mc-reps 1000] [--no-review] [--experiment-factor X]
```

- `--units` is the primary decision-unit file. [review] Its default is now WP4b's final file, `data/processed/rb5_units/units_primary.csv` (49 budget-level units, codes `readings_final.csv`). It is the same partition of the 77 models as rb2's `units_subgroup.csv` and `units_budget.csv`; two re-runs from scratch with it (25 September, from 14:26) gave every output file byte-identical to the 12:17 run, apart from the run log and the PDF's time stamp.
- `--experiment-factor` exists only for M9-8 (the experiment's implied factor for M\*, marked on the breakdown figure). It must not be set before Amendment 2 is committed and the Q2 statistic is reported. Nothing in this module reads a FineWeb or FineWeb-Edu estimate.
- Inputs are read-only: rb2's clean sample and unit files, ra1's per-budget IsoFLOP minima, rb4's Chinchilla minima in N_F, the open-athena compilation (Marin's configuration counts), ra2's technology cache (DeepSeek's law; every technology's anchor), rb1's sigma\*(C) set. rb2_decisions and ra2_wedge are imported read-only; no function that writes into another module's folder is called.

Review: `output/memos/rb5_sign_review.md`. [review] A separate reviewer (WP4a-review) re-ran the module from scratch and re-implemented S1 to S4 in standalone code (`code/analysis/rb5_sign/review_independent.py`, which imports no module of the project): 144 of 144 quantities agree (`output/tables/rb5_sign_review_independent.csv`). The module's own checks (`review_checks.py`, run at the end of `run.py`) pass 27 of 27.

---------------------------------------------------------------------------------------------------------

## 1. Headline

**H1. Over-training is identified for most models and most decisions, not for most compute.** Under the primary set (S1), w > 1 is identified beyond the designs for:

| | Share identified (tau = 1) | Weighted by compute |
|---|---|---|
| Models (77) | **0.78** (60 of 77) | **0.24** |
| Models below 15B (59) | 0.98 | 1.00 |
| Allocation decisions, budget level (49) | **0.69** (34 of 49) | **0.14** |
| Family-label units (56) | 0.71 | 0.15 |

- No model is identified as under-trained. That is mechanical: the smallest M in the clean sample is 19.5, and the lower end of the set at the models' compute runs from 4.0 to 11.4.
- The 17 unidentified models all have M below 250 (largest: Qwen2.5-72B, 247.6). They include Llama 2 70B, Llama 3.1 405B, Qwen2 72B, Falcon 180B and DeepSeek LLM 67B, and also Llama 3 70B, Apertus 70B, OLMo 2 32B, Olmo 3 32B and Qwen2.5 72B. Only one is below 15B (LLaMA 13B).
- Version 3's PI-1 gave 0.87 of models, 0.82 of the 56 units and 0.48 of compute. The claim "most models, though not most training compute" survives; the compute share halves.

**H2. The identified set is wider at the top and narrower at the bottom.** M\*(10^24) lies in [5.7, 248] and M\*(10^25) in [4.5, 468] (models' convention; version 3: [2.7, 88] and [1.9, 128]). DeepSeek's law (13.9 and 12.4 in its own convention) lies inside both.

**H3. Marin sets the upper end of the set.** Marin's pooled path, with one slope over its three corpora, rises by 0.198 in ln M\* per unit of ln C (95 percent interval [0.132, 0.265]); Chinchilla's and Llama 3's are flat (-0.003 and 0.008).
- Marin's band binds the upper end for 76 of the 77 models (DeepSeek's law for one). Llama 3's band binds the lower end for all 77.
- Marin's band alone identifies 0.78 of models; Chinchilla, Llama 3 and DeepSeek without Marin would identify 0.88 of models and 0.52 of compute ([review] now sourced: the comparison-only rows of `rb5_sign_shares.csv`).
- Implication for the text: the sign result is now as good as the extrapolation of Marin's path, fitted at 3x10^18 to 3x10^20 FLOP, over five more decades. With Marin's corpus levels averaged the shares at tau = 1 do not move (0.78, 0.69), and at tau = 3.4 the share of models rises from 0.53 to 0.60.
- [review] The minima paths are not straight inside the designs (`rb5_sign_review_curvature.csv`). A quadratic term in ln C is significant for Chinchilla (convex, t = 7.15, p < 0.001) and Marin (concave, t = -3.11, p = 0.008), and marginal for Llama 3 (convex, t = 2.39, p = 0.06). At their anchors the quadratic's local slopes are 0.40 (s.e. 0.06) for Chinchilla and 0.25 (0.10) for Llama 3, against -0.05 (0.08) for Marin: Marin's steep linear slope comes from its small budgets, and Chinchilla's and Llama 3's flat ones average a fall and a rise. The linear band does not cover this misspecification. Continuing Chinchilla's and Llama 3's paths above their anchors at those local slopes, beside the S1 bands, identifies 0.75 of models and 0.20 of compute (0.69 of decisions), with an upper end of 339 for M\*(10^24); with one standard error added to each local slope, 0.71, 0.17 and 0.65, and 477. The headline stays inside the S2 range, but the text should not present Chinchilla's and Llama 3's paths as flat beyond the designs.

**H4. Tilt allowances and the breakdown point.** Share of models identified (compute-weighted in brackets):

| tau | 1 | 1.3 | 1.84 | 3.4 | 4.4 | 8 | 10 |
|---|---|---|---|---|---|---|---|
| Models (77) | 0.78 [0.24] | 0.74 [0.23] | 0.66 [0.17] | 0.53 [0.16] | 0.47 [0.13] | 0.38 [0.08] | 0.32 [0.06] |
| Models below 15B (59) | 0.98 | 0.93 | 0.86 | 0.69 | 0.61 | 0.49 | 0.42 |
| Decisions (49) | 0.69 [0.14] | 0.67 [0.14] | 0.61 [0.11] | 0.47 [0.10] | 0.41 [0.10] | 0.31 [0.01] | 0.27 [0.01] |

- Breakdown tau (share falls below one half): 3.7 for all models, 7.5 for models below 15B, 2.7 for the 49 decisions. The compute share is below one half already at tau = 1.
- Under PI-1 the breakdown points were 8.3, 19.4 and 5.9; under PI-1's anchors with slopes widened to their 95 percent intervals (S2(c)) they are 2.7, 5.8 and 2.3.
- Version 3's sentence "Most small and mid-sized open models are over-trained under any allowance the evidence suggests" is too strong under S1: below 15B the share is 0.69 at DataDecide's 3.4 and 0.42 at 10.

**H5. Magnitudes (S4).** Medians over the 77 models of the bounds on s under S1:

| Curvature k = 1/sigma\* - 1 | tau = 1: lower, upper | tau = 1.84: lower | tau = 3.4: lower |
|---|---|---|---|
| [0.40, 0.52] (version 3's range) | 0.41, 0.90 | 0.24 | 0.03 |
| [0.40, 0.68] (to the top-budget value 0.594; R2 NM3(d)) | 0.41, 0.95 | 0.24 | 0.03 |
| sigma\* at the lower end of its identified set at each model's compute (Chinchilla-Llama 3 drift) | 0.41, 0.99 | 0.24 | 0.03 |
| same, pooled drift (three-level meta-regression) | 0.41, 0.98 | 0.24 | 0.03 |
| R1's drift-agnostic set [sigma_lin(C), 0.70] | 0.43, 0.99 | 0.26 | 0.03 |

- Version 3 (PI-1): 0.57 and 0.93; 0.45 at 1.84; 0.30 at 3.4 (reproduced exactly from S2(a)).
- At tau = 3.4 the median model is barely identified (53 percent identified), so its lower bound on s is close to zero (0.03). The introduction's "[[S4: 0.30]]" (H5) becomes 0.03.
- [review] Over the 49 decisions (a unit's bounds are the compute-weighted harmonic means of its members' bounds, since W rises with every member's w), the medians of the bounds are 0.33 and 0.88 at tau = 1 with k in [0.40, 0.52] (0.33 and 0.94 with k in [0.40, 0.68]), 0.15 at tau = 1.84 and -0.12 at tau = 3.4 (-0.16): at DataDecide's factor the median decision's sign is not identified (47 percent of decisions are). H5's sentence speaks of the median decision, so it should either name the median model ("lowers the median model's lower bound on s to 0.03") or say that the median decision's sign is then no longer identified. Rows `S4_dec49_*` of `rb5_sign_paper_numbers.csv`.
- With the reference zero points held fixed, the median share over the 49 decisions is 0.73 at the reference, 0.87 at a constant sigma\* of 0.60, 0.85 at 0.62 (0.848 at 0.622) and 0.96 with sigma\* at the lower end of its identified set. Over the 56 units: 0.74, 0.87, 0.85. The share of decisions with w > 1 does not move with sigma\* (0.98 over 49).

---------------------------------------------------------------------------------------------------------

## 2. Construction (S1; decision D-3)

**Paths.** One path per study, ln M\*_b = a_{s,g} + e_s (c_b - c_J), OLS on the logged per-budget minima at bracketed budgets (ra1's reviewed estimator, `ra1_modelfree_isoflop_budgets.csv`):

| Study | Budgets | c_J | Convention | Level at c_J [pointwise 95%] | Slope e [95%] | Residual s.d. (df) |
|---|---|---|---|---|---|---|
| Chinchilla | 9, 6x10^18 to 3x10^21 | 3x10^21 | total parameters, as digitized | 20.7 [13.8, 30.9] | -0.003 [-0.122, 0.115] | 0.271 (7) |
| Llama 3 | 8, 6x10^18 to 10^21 | 10^21 | FLOP-implied N_F = C/(6D) | 15.1 [11.2, 20.3] | 0.008 [-0.086, 0.102] | 0.185 (6) |
| Marin | DCLM 7, Nemotron-CC 7, Comma 5; 3x10^18 to 3x10^20 | 3x10^20 | FLOP-implied | DCLM 15.0 [12.2, 18.5]; Nemotron-CC 15.5 [12.5, 19.3]; Comma 20.8 [16.2, 26.8] | 0.198 [0.132, 0.265] (common) | 0.202 (15, pooled) |
| DeepSeek LLM | published law, point path | | non-embedding FLOPs per token | 20.6 at 3x10^20 | -0.049 | |

**Bands.** For each study, a simultaneous 95 percent band for its corpus paths over c, carrying level and slope error jointly:
- Joint wild bootstrap-t of (a, e): residuals rescaled by 1/sqrt(1 - h_ii), Webb's six-point weights, 9,999 draws, seeds 20260926 to 20260928.
- Studentized by the within-study pooled residual variance (classical OLS standard error). For Marin, the variance is pooled over the three corpora (15 degrees of freedom), which removes the instability of Comma's five-budget HC2 statistic (version 3's [0.1, 3,571]).
- Critical value: sup over corpora and over c of |t\*|, computed exactly. For a line, the squared t-ratio has one interior maximum, in the Working-Hotelling direction V^-1 z, so the supremum over an interval is that maximum or an end point. Checked against a grid of 401 points plus the models' computes.
- Small-sample rule (from the review's Monte Carlo): the band uses q = max(q_boot, q_normal), with q_normal the exact sup-t quantile under homoskedastic normal errors. With 8 or 9 budgets the bootstrap's q alone covers the true path over the whole domain in 91 to 93 percent of Monte Carlo samples; with the rule, 96 to 97 percent (review memo, Section 4; [review] the reviewer's own Monte Carlo gives 0.93 and 0.96 to 0.97). Critical values used: Chinchilla 2.73 (normal-theory; bootstrap 2.73), Llama 3 2.87 (bootstrap; normal-theory 2.76), Marin 2.77 (normal-theory; bootstrap 2.70). The bootstrap's q alone gives the same shares (a sensitivity row).
- Domain: c from the smallest model compute, 4.8x10^20 FLOP (Marin: from its anchor, 3x10^20), to 4.0x10^25, which covers every clean-sample model. The fix list says "from c_J"; [review] four models (SmolLM 135M and 360M, SmolLM2 135M and phi-1.5, 4.8x10^20 to 1.6x10^21 FLOP) lie below Chinchilla's c_J (the draft said three), so the domain starts lower. This makes the band slightly wider (more conservative); the reviewer's re-implementation on the models' total-parameter range alone gives the same shares.

**Set.** The identified set at c is the union of the bands (DeepSeek as a point), widened by ln tau (Proposition A6(vii)-(viii)). Each path places a model in its own convention; models' M is in total parameters for the three IsoFLOP studies (as in version 3) and in DeepSeek's count for its law. A decision is identified only if every member is.

**Anchor conventions (S1(5); R4 minor 7; numbers_wedge #9; numbers_conclusion #7).**
- Chinchilla: total parameters as digitized, the convention of the models' M. In N_F (rb4, Hoffmann et al.'s count) its level at 3x10^21 is 20.6 [13.6, 31.0] and its slope 0.022; the Chinchilla band never binds, so swapping it changes nothing (S2(f), third row).
- Marin: FLOP-implied. Marin's configuration count N_cfg is a total count (it includes both embedding matrices); the per-configuration ratio N_F/N_cfg = C_b/(6 N_cfg D) runs from 0.74 (157M) to 1.07 (above 5B) and is constant across budgets to 1e-5 (its within-budget elasticity is ra1's eta = 0.087). At the anchors' minima it is 1.03 to 1.05. Converting every budget's minimum to total parameters raises the top levels by 3 to 5 percent and steepens the slope to 0.247 [0.175, 0.318], because the ratio rises with size.
- Llama 3: FLOP-implied; Meta's FLOP accounting is not stated, so the conversion is bounded. For a Llama-3-shaped model of the anchor's size (Llama 3.2 3B: d = 3072, 28 layers, vocabulary 128,256), ln(N_F/N_total) ranges over [-0.247, 0.198] across three accountings (6 N_total D; non-embedding only; non-embedding plus head plus attention at Llama 3's 8,192-token context) and tied or untied embeddings. The total-parameter set shifts Llama 3's band by that range at every compute, which is conservative: embedding and attention shares fall with size.
- Sets in both conventions: M\*(10^24) in [5.7, 248] in the primary (mixed) convention, [4.4, 391] in total parameters, [5.7, 248] in FLOP-effective parameters (models then in Hoffmann's count at 4,096 tokens). At 10^25: [4.5, 468], [3.5, 828], [4.5, 468].

---------------------------------------------------------------------------------------------------------

## 3. Sensitivities (S2)

Share identified at tau = 1 and 3.4 (compute-weighted in brackets); decisions are the 49 budget-level units.

| Set | Models, tau = 1 | Models, 3.4 | Decisions, tau = 1 | Decisions, 3.4 |
|---|---|---|---|---|
| **S1 primary** | **0.78** [0.24] | 0.53 [0.16] | **0.69** [0.14] | 0.47 [0.10] |
| (a) PI-1 as published (point-slope hull [-0.156, 0.162], Comma excluded) | 0.87 [0.48] | 0.69 [0.23] | 0.82 [0.40] | 0.63 [0.14] |
| (a) PI-1 with Marin Comma (version 3's disclosure) | 0.08 [0.00] | 0.00 | 0.06 [0.00] | 0.00 |
| (b) t-intervals, k - 2 df, Comma with its own slope (slopes [-0.156, 0.294]) | 0.68 [0.17] | 0.43 [0.13] | 0.63 [0.12] | 0.39 [0.10] |
| (b') as (b), slopes at the hull of their t-intervals [-0.156, 0.459] | 0.44 [0.09] | 0.26 [0.03] | 0.41 [0.03] | 0.20 [0.01] |
| (c) PI-1 levels, slopes at the hull of their 95% intervals [-0.175, 0.353] | 0.71 [0.17] | 0.47 [0.13] | 0.65 [0.12] | 0.41 [0.10] |
| (d) normal inputs only, S1 levels, abs(e) <= 1 | 0.13 [0.00] | 0.10 | 0.14 [0.00] | 0.10 |
| (d) normal inputs only, PI-1 levels (version 3) | 0.14 [0.00] | 0.10 | 0.14 [0.00] | 0.10 |
| (e) every technology as an anchor, with the S1 bands (slopes [-0.131, 0.311]) | 0.23 [0.03] | 0.10 | 0.16 [0.00] | 0.08 |
| (e) PI-4 as published | 0.23 [0.03] | 0.10 | 0.16 [0.00] | 0.08 |
| (f) all anchors in total parameters | 0.71 [0.17] | 0.45 [0.13] | 0.65 [0.12] | 0.41 [0.10] |
| (f) anchors and models in FLOP-effective parameters | 0.75 [0.24] | 0.48 [0.15] | 0.67 [0.14] | 0.41 [0.10] |
| (f) Chinchilla's anchor alone in N_F | 0.78 [0.24] | 0.53 [0.16] | 0.69 [0.14] | 0.47 [0.10] |
| S1, HC2 studentization (as version 3's paths) | 0.75 [0.24] | 0.48 [0.15] | 0.67 [0.14] | 0.41 [0.10] |
| S1, pointwise instead of simultaneous bands | 0.78 [0.24] | 0.53 [0.16] | 0.69 [0.14] | 0.47 [0.10] |
| S1, bootstrap critical value alone | 0.78 [0.24] | 0.53 [0.16] | 0.69 [0.14] | 0.47 [0.10] |
| S1, Marin at the mean of its corpus levels | 0.78 [0.24] | 0.60 [0.16] | 0.69 [0.14] | 0.55 [0.11] |
| S1 without DeepSeek's law | 0.78 [0.24] | 0.53 [0.16] | 0.69 [0.14] | 0.47 [0.10] |
| S1, Marin without Comma (comparison only; re-creates the post hoc exclusion) | 0.86 [0.48] | 0.70 [0.23] | 0.80 [0.39] | 0.65 [0.14] |

Reading the table:
- Every variant that carries slope uncertainty, or admits Comma with an interval valid for few budgets, lands between 0.68 and 0.78 of models and 0.17 and 0.24 of compute, except (b'), which lets every slope reach its own t-interval, Comma's included (0.459): 0.44 of models. [review] PI-1 with Comma on version 3's HC2 bootstrap-t interval (0.08) also admits Comma; it is the post hoc disclosure, not a valid-interval variant. D-3's statement "62 to 87 percent of models, 17 to 48 percent of compute" should be read with S1 at 78 and 24; (b') is below that range.
- Two models sit within 3 percent of the envelope at tau = 1 (StableLM-Alpha 7B, margin 0.004 in ln M; Llama 2 13B, 0.030). They account for the drop from 0.78 to 0.75 under HC2 studentization and under FLOP-effective parameters.
- The referees' numbers are reproduced: R2 NM1 (0.68 and 0.17) by (b); R1 New 3(a) (0.71, 0.66 over the 56 units, 0.17) by (c); R4 N2 (72.7 and 19.5 percent) when DeepSeek's law is evaluated at the models' total M and C, as R4 did (the module uses DeepSeek's own convention, which gives (c)'s 0.714 and 0.173). R2's "0.62 and 0.17" for t-intervals with Comma and slope intervals is not reproduced exactly; its slope range is not stated (a reconstruction gives 0.58 and 0.16). [review] With every slope at its own t-interval, Comma's included, the same reconstruction is (b'): 0.44 and 0.09. R2's intermediate step, "0.84 of models (0.48 of compute)" for PI-1 plus Comma on its t-interval with PI-1's slopes, is reproduced (0.844 and 0.477).
- The 14 and 23 percent of the introduction (H6): 13 percent of models under normal inputs with the S1 levels (14 with version 3's), and 23 percent with every technology as an anchor (unchanged; the technologies' own anchors bind).
- Version 3's Comma interval was printed as [0.1, 3,571]; the same code on the current inputs gives [0.16, 3,586] (the share with Comma is unchanged at 0.078). The table note now prints [0.2, 3,586].

---------------------------------------------------------------------------------------------------------

## 4. By release year (S1, models, tau = 1)

| Year | n | Share identified | Compute-weighted |
|---|---|---|---|
| 2023 | 24 | 0.63 | 0.12 |
| 2024 | 40 | 0.88 | 0.19 |
| 2025 | 13 | 0.77 | 0.49 |

- 2023 remains the year in which the sign is least identified by compute (0.12; version 3: 0.20), which bears on the trend's base year (W19(b)).
- Over the 49 decisions: 0.50, 0.85 and 0.50 (compute-weighted 0.04, 0.08, 0.43).

---------------------------------------------------------------------------------------------------------

## 5. Outputs

| File | Content |
|---|---|
| `output/tables/rb5_sign_table.tex` | Appendix table (label `tab:app-sign`), for WP4b to copy into `paper/tables/appF_sign.tex`. Panels: models (primary, below 15B, S2(a)-(f)), decisions (49 and 56), primary by year; unweighted at tau = 1, 1.3, 1.84, 3.4, 4.4, 10 and compute-weighted at 1, 1.84, 3.4, 10. The note records the post hoc PI-1 rule, P6's "tau approximately 1.8 ... (we use 1.84)" wording and the anchor conventions. Test-compiled in the main preamble: no overfull box. |
| `output/figures/rb5_sign_breakdown.{pdf,png}` | Breakdown frontier (S3), four panels (all models, below 15B, decisions, compute), S1 solid and S2(c) dashed, tau from 1 to 20 on a log axis, tau = 1.3, 1.84, 3.4 marked. WP4b sets `\label{fig:app-breakdown}` where it inputs the figure. |
| `output/tables/rb5_sign_paper_numbers.csv` | Every number for W3, W11, W12, W19(b), W22(c), W23(m), H1, H5, H6, H11(g) and P13, with its slot, formatted value, raw value and source file. [review] A percentage that would round to 0 or 100 without being 0 or 1 is written "<1" or ">99"; decision-level S4 rows added (`S4_dec49_*`). |
| `output/tables/rb5_sign_shares.csv` | All sets x tau in {1, 1.3, 1.84, 3.4, 4.4, 8, 10} x level (models, below 15B, 49 and 56 decisions) x year; w > 1 and w < 1; unweighted and compute-weighted. |
| `rb5_sign_paths.csv`, `_paths_variants.csv` | Study fits: levels, slopes, intervals, critical values (bootstrap, normal-theory, used), domains. |
| `rb5_sign_mstar_sets.csv` | M\*(C) sets at 10^22 to 10^25 by set and convention, and each path alone. |
| `rb5_sign_unidentified.csv`, `rb5_sign_models.csv` | The 17 unidentified models with their M and the set at their compute; every model's bounds, set, binding path and breakdown tau. |
| `rb5_sign_breakdown.csv`, `_frontier.csv` | Breakdown tau by series; the frontier at the marked tau (full grid in `data/processed/rb5_sign/frontier_grid.csv`). |
| `rb5_sign_magnitude.csv`, `_fixed_zero_point.csv` | S4. [review] `rb5_sign_magnitude.csv` has a `level` column: models, and the 49 decisions (S1 only). |
| `rb5_sign_variants.csv`, `_tinterval_anchors.csv`, `_conventions.csv` | Definitions of S2's sets; S2(b)'s t-interval anchors; the Llama 3 conversion bound and Marin's conversion at each minimum. |
| `rb5_sign_review_checks.csv`, `_review_coverage.csv` | The module's own checks (Section 6). |
| `rb5_sign_review_curvature.csv`, `_review_independent.csv` | [review] Curvature of the paths inside the designs and the local-slope sensitivity; the reviewer's independent re-implementation against the module (`review_independent.py`, run separately). |
| `data/processed/rb5_sign/` | `model_bounds_S1.csv`, `model_bounds_all_sets.csv`, `units_sign.csv` (per-unit flags at every tau, for Table 2's Panels A to C), `bands_grid.csv`, `frontier_grid.csv`, `note_numbers.json`, `run_log.txt`. |

Suggested caption and note for WP4b (Appendix E, `\label{fig:app-breakdown}`):
> *Breakdown Frontier of the Sign Result.* Share of the clean sample for which $w>1$ is identified beyond the designs against the tilt allowance $\tau$ on $M^*$ (log scale): all 77 models (panel a), the 59 below 15 billion parameters (b), the 49 allocation decisions (c) and training compute $6ND$ over models (d). Solid: the primary set of Table~\ref{tab:app-sign}; dashed: PI-1's levels with path slopes anywhere in the hull of the designs' 95 percent slope intervals. Dotted vertical lines: $\tau=1.3$ (tokenizer differences), 1.84 and 3.4 (DataDecide's largest tilt under the reference exponents and under its own); dotted horizontal line: one half. The share of models falls below one half at $\tau=3.7$ (2.7 under the dashed set); the compute share is below one half at every $\tau$.

---------------------------------------------------------------------------------------------------------

## 6. Review and caveats

Independent checks (`rb5_sign_review.md`): refits by a separate implementation; S1 bounds recomputed with explicit loops (difference below 1e-15); exact and grid-based critical values agree; S2(a) reproduces version 3 to four decimals; the referees' recomputations reproduced; no em dash in any output. [review] A separate reviewer's standalone re-implementation agrees on all 144 quantities compared (shares exactly; critical values, set ends and the breakdown tau within Monte Carlo error).

Caveats:
1. **Marin drives the result.** Its pooled slope is estimated at small compute and extrapolated over five decades; its band sets the upper end for 76 of 77 models. A reader could argue for a narrower set (Chinchilla's and Llama 3's fitted lines are flat to 3x10^21 and 10^21; [review] but both paths rise at their top budgets, caveat 4) or a wider one (Marin in total parameters, slope 0.247). Both are reported.
2. **Coverage.** The union covers the developer's zero point with probability at least 0.95 only if the developer's technology lies within tau of one of the four paths (Proposition A6(viii)). The band's small-sample coverage is checked by simulation under the fitted designs (review memo, Section 4), not under misspecification of the linear path.
3. **Two marginal models.** StableLM-Alpha 7B and Llama 2 13B are identified with margins of 0.4 and 3 percent; state shares as "about three quarters of models".
4. [review] **Straight paths.** The minima paths curve inside the designs (H3). The band covers sampling error around a straight line, not the curvature; continuing Chinchilla's and Llama 3's top-budget local slopes gives 0.75 of models (0.71 with one standard error added).
5. **sigma\*(C) set.** S4's "lower end of the identified set" reads `rb1_sigmaC_pi_sigmaC.csv` (post-rb4). If WP2 revises the drift or the top-budget value (T items), re-run this module; the upper bound moves at the third decimal.
6. **Units.** [review] Done: the default is WP4b's final `data/processed/rb5_units/units_primary.csv`, the same partition as `units_subgroup.csv`; the outputs did not change.
