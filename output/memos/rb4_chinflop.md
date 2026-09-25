# Memo — module rb4_chinflop: Chinchilla's FLOP accounting rebuilt from Hoffmann et al.'s architecture table

Module owner: rb4_chinflop (Claude). Date: 2026-09-24. Entry point: `code/analysis/rb4_chinflop/run.py`.

**Scope.** Round-2 Referee 2, Major 1 request 4 (round-1 Major 9(b)): "Rebuild Chinchilla's FLOPs per token for each model in Hoffmann et al.'s architecture table (their Appendix F count) to obtain η by budget … it would turn the ±0.04 into a number." The module computes Chinchilla's model-free σ\* in the paper's primary convention, FLOP-effective parameters N_F = C/(6D), with C the actual training FLOPs. It also reports the non-embedding (P) and total (T) conventions. The rebuilt estimates are then propagated into rb1's study-level interval and σ\*(C) meta-regression. This memo covers the FLOP rebuild and its propagation only. The code of ra1, rb1, m1 and `sl.py` is imported unchanged, and `paper/sections/*.tex` is not edited.

**Reproduction.**
- One command regenerates every output: `.venv/bin/python code/analysis/rb4_chinflop/run.py`. It runs in about 29 s on at most 4 CPU processes, with no GPU ([review]: about 40 s since the reviewer's R11–R14 and specification grid were added to the review stage). The stages are arch → estimate → param → propagate → tables → review. `--stages tables` rebuilds the tables from the stage caches.
- Seeds:
  - For every accounting, the bootstrap uses ra1's own Chinchilla seed (`ra1_common.seed_of("iso|Chinchilla")`, B = 999). The accountings therefore share ra1's random numbers.
  - The parametric fits use ra1's parametric seed. The meta-regression uses rb1's wild-bootstrap seeds (B = 9,999).
- Inputs:
  - Epoch AI's digitization of Hoffmann et al.'s Figure 4, read through ra1/m1's loader.
  - ra1's and rb1's published CSVs.
  - The arXiv source of Hoffmann et al. (2022), 2203.15556, downloaded to `data/raw/rb4_chinflop/` (sha256 `6571e76a…`; public). The run downloads it if it is missing.
  - Tables A9 and A4 are transcribed into `rb4_arch.py`. At each run the transcription is checked cell by cell against the TeX (50/50 rows and 6/6 rows equal).
- **Licence.** No digitized row (a run's N, C or loss) is written to any committed file. Outputs, override CSVs and stage caches hold derived statistics only.
- **Hook into rb1.** rb1's `meta_c` reads ra1's inputs through `rb1common.RA1_BUDGETS` and `RA1_SUMMARY` at call time. The module writes override copies in which only Chinchilla's rows change (`data/processed/rb4_chinflop/override_*.csv`), points those attributes at them, and calls rb1's own `study_level`, `run_meta`, `top_budget_values`, `add_local_fd_variant` and `pid` functions. rb1's log goes to a temporary directory.
- **Validation of the hook.** The "old" run through the hook reproduces rb1's published study-level table, all 42 meta-regression rows, and the partial-identification and wedge-scenario tables, up to rb1's CSV rounding (%.6g).
- **Determinism.** Two runs into separate output roots are identical byte for byte (see Self-review).

**Notation.**
- T: total parameters, as Hoffmann et al. count them.
- P = T − Vd: non-embedding parameters.
- F: training FLOPs per token, forward plus backward (3 × forward).
- N_F = F/6.
- r = F/(6T).
- η = d ln r / d ln T within a budget.
- S = 2(1/σ\* − 1).
- "Nominal" means runs placed on their nominal budget C_b, as ra1 does.

---

## 1. Headline findings

### H1. Hoffmann et al.'s Appendix F formula, as printed, does not reproduce their own Table A4. The same formula without its two vocabulary terms reproduces all six ratios exactly

Files: `rb4_chinflop_tableA4.csv`, `_tableA4_summary.csv`, `_arch.csv`, `_totals.csv`.

**Parameter count.** Table A9's 50 reported counts are reproduced (max error 0.8 percent, mean 0.8M) by
T = L(4 d d_kv h + 2 d f) + L d d_kv h + V d.
- The first term is attention plus MLP.
- The second is one Transformer-XL relative-position projection W_R per layer, as in Gopher. Without it the counts miss by 7–8 percent.
- The third is **one** V×d vocabulary matrix, either tied or input only. With two such matrices the counts miss by up to 36 percent.
- So P = T − Vd, and there is no untied output matrix in T to remove. W_R is kept in P as a per-layer weight matrix. Dropping it changes nothing (0.691 against 0.691; R8).

**FLOPs per token.** Three counts are carried through the module:

| Count | Definition | Table A4 ratios reproduced (to 2 dp) | Max error | F/(6T) on the 35 profile models |
|---|---|---|---|---|
| **T4** | Appendix F **without** the embedding and final-logit terms | **6 of 6** | 0.005 | 0.98–1.15 |
| A | Appendix F as printed (embedding "matmul" + logits) | 0 of 6 | 0.55 | 1.00–1.65 |
| X | Executed: A minus the embedding term (a lookup costs no multiply-adds; the unembedding matmul does) | 0 of 6 | 0.27 | 0.99–1.34 |

- **T4 is the count Hoffmann et al. evidently implemented.** Their Table A4 ratios (1.03, 1.10, 1.08, 1.04, 1.03, 0.99) are F_T4/(6T) to two decimals.
  - Under the printed formula the 73M model's ratio would be 1.58, not 1.03.
  - The printed formula is also inconsistent with their statement that the differences from 6ND are "very small".
- **T4 is nearly 6T.** Two terms roughly cancel: the S² attention terms that 6N omits, and the embedding and W_R parameters that 6N counts but T4 gives no FLOPs.
- **One figure is not reproduced.** None of these counts gives Hoffmann et al.'s quoted 6.3×10²³ FLOP for Gopher (T4 4.74×10²³, A 4.75×10²³, 6TD 5.03×10²³; Rae et al. quote 5.76×10²³). This is an unexplained internal inconsistency in the source. Nothing here depends on it.

### H2. Which count set the budgets, and how D is recovered

Files: `rb4_chinflop_coords.csv` and `_match_summary.csv`; Table Panel A.

**Matching.**
- Every one of the 245 digitized runs matches a Table A9 model within 0.6 percent in N: median 0.03 percent; nearest-to-next-model ratio ≤ 0.076, so no match is ambiguous.
- The 137 profile runs use 35 of the 50 models. 43 models appear anywhere in the digitization.
- Seven models never appear: 44M, 3.53B, 3.80B, 4.08B, 12.3B, 13.7B and 14.9B.
- **Hoffmann et al.'s smallest model (44M) is absent from Figure 4.** This settles round-1 minor 10: the 57M floor is the extraction's, not an outlier drop.
  - [review] Strictly, the 44M model is absent from the *digitization* of Figure 4; whether the figure omits it or the extraction missed it is not tested here. Either way the 57M floor is not an outlier drop.

**Coordinate test.** Figure 4 draws contours of the fitted L̂(N, D) over (FLOPs, N), and its efficient frontier uses FLOPs ≈ 6ND. That requires the horizontal axis to be 6ND; the test below is consistent with this. If the tokens were set so that F(N)·D = C_b for some count F, each run sits at log₁₀C_dig − log₁₀C_b = −log₁₀ r_F(N). Within budgets, the residual must then be flat in N.

| Count that set the budgets | Within-budget slope of the residual on log₁₀N (CR1 s.e., 9 budgets) | p | Mean offset (dex) |
|---|---|---|---|
| **T4** | **−0.002 (0.007)** | **0.74** | −0.006 |
| A (as printed) | −0.065 (0.012) | < 0.001 | +0.045 |
| X (executed) | −0.036 (0.009) | 0.004 | +0.020 |
| 6T (tokens = C_b/(6T)) | +0.020 (0.004) [review: was "+0.021 (0.005)"; the CSV has 0.0205 (0.0045), and the table prints +0.020 (0.004)] | 0.002 | −0.023 |

- Only T4 is consistent with the figure's coordinates.
  - [review] The reviewer's re-implementation (own TeX parse, own budget assignment and matching) reproduces all four rows to four decimals. Further counts are rejected as well: 6P (non-embedding, budgets = 6PD) +0.065 (0.007); T4 at context 1,024 or with causal (halved) S² terms +0.028 (0.006); T4 at context 4,096 −0.049 (0.010). The test therefore also pins the attention accounting (full S² terms at 2,048). Using the digitized or reported N instead of the architecture count in r = F/(6N) changes no slope by more than 0.0003.
- It also explains most of m1's unexplained −0.018 dex offset of the digitized budgets (−0.023 before m1's offset correction; −0.006 under T4).
- Regressing the digitized deviation on −log₁₀ r_T4 with budget fixed effects gives 0.70 (s.e. 0.19). That is consistent with 1 and far from 0.
- The conclusion survives dropping the five highest-loss runs (R7).
- By budget, the 6T residual slope is positive in 8 of 9 budgets and the T4 residual slope in 3 of 9.

**How D is recovered, and why the circularity is harmless for N_F.**
- Epoch's D is C_dig/(6N). Under the supported reading, C_dig = 6·N·D_true, so Epoch's D is the true token count itself.
- Hoffmann-count FLOPs are then F_T4·D = C_b: each IsoFLOP profile is an **exact** isocost in the T4 count.
- The task's framing, D = C_b/(6N_reported) with C_actual = F·D, is the "6T" row of the test, and the data reject it.
- The circularity is that D is never observed, so recovering it requires assuming which count defined the budgets. For N_F this does not matter:
  - N_F = C/(6D) = F(N)·D/(6D) = F/6, so the token count cancels.
  - Every reading gives the same x = ln N_F.
  - The readings differ only in whether the runs of one nominal budget share one actual budget. Under T4 they do. Under any other count they differ by ρ_i = F_acc/F_budget, which varies along the profile.

### H3. Chinchilla in FLOP-effective parameters: σ\* = 0.660 (0.023), against 0.673 (0.027) in total parameters

Files:
- `rb4_chinflop_summary.csv` (design level) and `rb4_chinflop_budgets.csv` (every budget, with ra1's value beside it);
- `rb4_chinflop_eta.csv`, `rb4_chinflop_fixedwin.csv`, `rb4_chinflop_param.csv`, `rb4_chinflop_reference_fit.csv`;
- table Panel B.

The estimator is ra1's, unchanged: order 2, h = 1, path-centred windows, log-cubic frontier, and a design-conditional wild bootstrap. With x = ln T it reproduces ra1's Chinchilla rows exactly (all 9 σ\*_b, s.e., intervals and M\*_b; RE, FE, drift, Q; R1).

| Count (runs on) | RE σ\* (s.e.) | FE σ\* | Drift per decade (s.e.) | Q (bootstrap p) | Change vs. current | Equivalent η |
|---|---|---|---|---|---|---|
| Current: T, digitized (nominal) | 0.673 (0.027) | 0.684 | −0.072 (0.026) | 15.6 (0.025) | – | – |
| **N_F, T4 (nominal = exact isocosts)** | **0.660 (0.023)** | **0.666** | **−0.061 (0.025)** | **12.7 (0.060)** | **−0.014** | **−0.031** |
| N_F, A (nominal) | 0.648 (0.017) | 0.651 | −0.047 (0.024) | 9.5 (0.26) | −0.025 | −0.055 |
| N_F, X (nominal) | 0.658 (0.019) | 0.662 | −0.054 (0.025) | 9.7 (0.24) | −0.015 | −0.033 |
| P = T − Vd (nominal) | 0.690 (0.029) | 0.703 | −0.082 (0.027) | 17.0 (0.017) | +0.016 | +0.039 |
| T, architecture count (nominal) | 0.674 (0.027) | 0.684 | −0.072 (0.026) | 15.9 (0.023) | +0.000 | +0.001 |
| N_F, A, own isocosts (a) | 0.663 (0.019) | 0.667 | −0.053 (0.023) | 11.2 | −0.011 | |
| N_F, X, own isocosts (a) | 0.666 (0.020) | 0.670 | −0.057 (0.025) | 10.5 | −0.008 | |
| P, own 6PD isocosts (a) | 0.669 (0.027) | 0.678 | −0.070 (0.026) | 15.6 | −0.004 | |
| T, own 6TD isocosts (a) | 0.660 (0.026) | 0.664 | −0.059 (0.025) | 14.5 | −0.014 | |
| N_F, T4, if tokens had been C_b/(6T) (b) | 0.671 (0.024) | 0.680 | −0.071 (0.026) | 13.5 | −0.002 | |
| *Memo: current with η = −0.1 / +0.1 (rb1)* | *0.626 / 0.714* | | | | *−0.048 / +0.040* | |

(a) Losses moved from the observed T4 isocost to the count's own isocost at the same nominal budget, L_i + L̂(N_i, D_i/ρ_i) − L̂(N_i, D_i), with the κ-free surface fitted to the same runs. (b) The reading rejected by the coordinate test.

- **η by budget** (T4, OLS slope of ln r on ln T over each window's runs; `_eta.csv`), from 6×10¹⁸ to 3×10²¹ FLOP: −0.003, −0.020, −0.036, −0.028, −0.026, −0.030, −0.021, −0.019, −0.024. The mean is −0.023. Under count A, η runs from −0.12 to −0.05; under X, from −0.07 to −0.04; under P, from +0.09 to +0.01.
- **The rebuild is a small number inside the η = ±0.1 band.**
  - rb1 bracketed Chinchilla's accounting by [0.626, 0.714]. The rebuilt primary estimate is 0.660, an equivalent η of −0.031.
  - Every count and budget hypothesis lies in [0.648, 0.690] (mechanical).
  - Every N_F estimate under the supported budget reading lies in [0.648, 0.666]. Including the rejected 6T-token reading, the range is [0.648, 0.683].
    - [review] This holds for the architecture-based counts at the primary windows and sample. With the count-free N_F, T4-consistent membership or the 132-run sample (H3 [review]), T4 gives 0.668–0.681.
  - Placed on their own exact isocosts, all counts and conventions agree to within 0.010 (0.660–0.669).
- **Budget-level changes** (`_budgets.csv`, N_F T4 against T): +0.002, −0.006, **−0.066**, −0.009, −0.009, −0.016, −0.015, −0.008 and −0.009.
  - The one large change, at 3×10¹⁹ (0.791 → 0.725), is a window effect: under h = 1 in ln N_F, one extra run enters the window (13 → 14).
  - With the current windows held fixed, that budget moves −0.018, and every budget moves between −0.018 and +0.001 (`_fixedwin.csv`).
  - Holding windows fixed, about two thirds of the design-level change remains: plain mean −0.010 of −0.015; DL mean ≈ 0.668 against 0.678 with common variances (R9).
- **The largest budgets**: 0.550, 0.609 and 0.569 at 6×10²⁰, 10²¹ and 3×10²¹, against 0.564, 0.617 and 0.579.
- **Drift and heterogeneity within Chinchilla.**
  - The drift is smaller in N_F: −0.061 (0.025; bootstrap p = 0.015), against −0.072 (0.026; p = 0.009). Most of the difference comes from the 3×10¹⁹ budget, which is high in T (0.791) and falls to 0.725 through the window change. Under T4, η itself is nearly flat above 10¹⁹ (−0.02 to −0.04). Under count A, η is steepest at small budgets, which is why A's drift is smallest (−0.047).
  - Homogeneity across budgets is no longer rejected at 5 percent (Q = 12.7, bootstrap p = 0.060, against 15.6 and 0.025).
- **[review] Window and sample sensitivity of the N_F level** (`rb4_chinflop_review_checks.csv` R11–R13, `rb4_chinflop_specgrid.csv`). The direction of the change is robust; the level of 0.660 is at the low end of what nearby choices give.
  - *ra1's specification grid in N_F* (12 specifications, ra1's seeds and B = 199; the T cells reproduce `ra1_modelfree_isoflop_sens.csv` exactly):
    - 137 runs: N_F gives 0.656–0.702 against T's 0.665–0.714. N_F is below T in all 11 specifications with at least 8 budgets, by 0.008 to 0.018 (median −0.012).
    - 132 runs (the five highest-loss runs dropped): N_F − T runs from −0.024 to +0.004 (median −0.009; negative in 8 of 11).
  - *The five highest-loss runs.* They lie outside every window, but they enter the path-centre step. In T, dropping them leaves every window unchanged (ra1's "they do not affect the estimate" holds in T). In N_F, dropping them moves the path enough to change two windows: the 1.6B run leaves the 3×10¹⁹ window again, and one run enters at 6×10²⁰. The primary specification then gives **0.681 (0.020)** in N_F, against 0.674 in T (R13).
  - *Profile membership.* m1's membership rule (|dev − offset| ≤ 0.045 dex) was drawn on the uncorrected coordinates. Under the supported T4 reading, the coordinates of deep, narrow small models sit up to 0.06 dex left of their budget by construction. Applied to the T4-corrected coordinates, the same rule admits 14 runs and drops 4 (147 runs). The estimate is then **0.668 (0.027)** in N_F and 0.678 in T (R12).
  - *Count-free N_F.* Reading N_F = C_b/(6D) directly off the digitized coordinates, as is done for Llama 3, gives **0.676 (0.021)** (R11). Under the T4 reading this equals F_T4/6 times the run's coordinate scatter (s.d. 0.016 dex), which enters x as noise. Adding noise of that size to x raises the plain mean of σ\*_b by 0.007 on average (s.d. 0.007), which is consistent with the gap in plain means (0.670 against 0.660; 32 percent of draws are at least as high).
  - *Consequence for the propagation.* With Chinchilla at 0.668, 0.676 or 0.681, the study-level mean is 0.691–0.694 and Cochran's Q is 5.8–6.3 (p = 0.10–0.12). The Q = 8.3 (p = 0.041) of H4 therefore depends on the primary windows. All three means lie inside the "Chinchilla's counts and conventions" range 0.682–0.697. On identical window runs, the pooled residual s.d. of the budget quadratics is 0.0116 under T, N_F (any count) and P, equal to four digits (`_fitq.csv`). The convention is a measurement choice, fixed here by what defined the budgets.
- **Same-run parametric σ\* in N_F** (137 runs, Huber; ra1's code and seed; T reproduces ra1 exactly):
  - κ free: 0.656 (0.028), against 0.667 (0.023).
  - κ = 1: 0.690 (0.015), against 0.692 (0.013).
  - κ̂ = 0.771; the quasi-LR of κ = 1 is 12.4, against 7.8.
- **The reference technology is invariant to the accounting** (240 runs, Huber, point estimates; T reproduces m2's Table 3 row exactly):

  | | σ\*_κ | σ\* (κ = 1) | a | M\*(10²³), κ = 1 | M\*(10²³), κ free |
  |---|---|---|---|---|---|
  | T | 0.701 | 0.737 | 0.514 | 18.8 | 21.9 |
  | N_F | 0.700 | 0.740 | 0.520 | 14.9 | 20.4 |

  M\* in N_F is in tokens per FLOP-effective parameter.

### H4. Propagation: study-level mean 0.687 [0.640, 0.734], heterogeneity now significant at 5 percent, and the drift slope unchanged (−0.033 per decade)

Files: `rb4_chinflop_study_level.csv`, `_metareg_slopes.csv`, `_metareg_predictions.csv`, `_top_budget.csv`, `_pi_sigmaC.csv`, `_wedge_scenarios.csv`, `_headline.csv`; table Panel C. The "current" column is rb1's published run, reproduced through the hook.

| Statistic | Current (Chinchilla in T) | Rebuilt (Chinchilla in N_F, T4) |
|---|---|---|
| **Study-level mean [95% HKSJ]** (k = 4) | 0.693 [0.653, 0.733] | **0.687 [0.640, 0.734]** |
| τ; Q (p) | 0.017; 5.7 (0.13) | 0.023; **8.3 (0.041)** |
| FLOP-accounting row | η = ±0.1 for Chinchilla and Meta: 0.665–0.708 [0.578, 0.751] | η = ±0.1 for **Meta only**: 0.673–0.700 [0.598, 0.748] |
| Chinchilla's other counts and conventions (A, X, P, T; nominal or own isocosts; T4 under 6T tokens) | – | 0.682–0.697 |
| Local first-derivative estimator row (rb1 review) | 0.673 [0.607, 0.739] | 0.670 [0.603, 0.737] |
| IsoFLOP studies only (k = 3) | 0.678 [0.616, 0.740] | 0.673 [0.609, 0.737] |
| Range of variant means (a) | 0.663–0.708 | 0.663–0.700 |
| **Pooled drift, 3-level RE (44 budgets) [CR2 95%]** | −0.0325 [−0.075, 0.010] | **−0.0328 [−0.076, 0.011]** |
| Wild cluster p; CR2 p | 0.042; 0.096 | 0.039; 0.097 |
| p across weighting, scale and test | 0.03–0.12 | 0.03–0.13 |
| Chinchilla + Llama 3, FE (s.e.) | −0.058 (0.013) | −0.058 (0.013) |
| Budgets ≤ 3×10²⁰ | −0.007 (wild p 0.85) | −0.008 (0.83) |
| Leave out each design's largest budget | −0.029 (wild p 0.29) | −0.030 (0.29) |
| σ\* at 10¹⁹ / 10²⁰ / 10²¹ (pooled line) | 0.715 / 0.683 / 0.650 | 0.713 / 0.680 / 0.647 |
| σ\* at 10²¹, CR2 95% | [0.565, 0.736] | [0.560, 0.735] |
| σ\*_top, budgets ≥ 6×10²⁰ [HKSJ] | 0.596 [0.567, 0.624], Q = 1.9 | 0.594 [0.565, 0.622], Q = 2.5 (p = 0.65) |
| Identified-set lower bound, 10²³ / 10²⁴ | 0.480 / 0.422 | 0.479 / 0.421 |
| Median s: at σ\*_top / on the pooled line / at the lower bound | 0.888 / 0.906 / 0.978 | 0.890 / 0.910 / 0.978 |

(a) All of rb1's variant rows. For the rebuilt column the both-study η rows are replaced by the Meta-only rows, and the Chinchilla-count rows are added. Keeping rb1's both-study η rows, the rebuilt range is 0.660–0.707.

- **The mean moves by −0.006, and the interval widens downward (0.653 → 0.640).** Chinchilla's estimate falls and its s.e. shrinks (0.027 → 0.023), so the between-study τ rises from 0.017 to 0.023.
- **Chinchilla and Meta now agree** (0.660 and 0.660), and both lie below Marin (0.706) and Farseer (0.708). Q = 8.3 (p = 0.041) means the four studies are heterogeneous at 5 percent.
  - The two designs whose budgets reach 3×10²¹ and 10²² sit below the two that stop at or before 10²¹.
  - This is the compute drift of rb1 H1 showing up at the study level, not a new inconsistency.
  - [review] Two corrections. (i) Meta's estimate uses its eight bracketed budgets, which stop at 10²¹, as Farseer's local path does. The accurate contrast is therefore that Chinchilla and Llama 3 are the designs whose budget-level estimates fall above 3×10²⁰ FLOP, while Marin stops at 3×10²⁰ and Farseer's path does not fall (0.658 at 10²¹). (ii) The rejection is fragile. Under the window and membership sensitivities of H3 [review], Q is 5.8–6.3 (p = 0.10–0.12), so "heterogeneous at 5 percent" should not be stated without that qualification. The link to the drift is an interpretation, consistent with the data but not tested.
- **The accounting uncertainty now attached to the headline is smaller.** It is Meta's unobserved η (0.673–0.700) plus Chinchilla's alternative counts (0.682–0.697). The widest interval, the Meta η = −0.1 row [0.598, 0.748], is narrower than before [0.578, 0.751].
- **The drift is untouched.** Chinchilla's budget-level estimates fall roughly in parallel, so the pooled slope, its significance, the top-budget value, the identified set beyond 10²¹ and the revealed-demand medians change by at most 0.004.
- **Alternatives for Chinchilla in the propagation:**

  | Chinchilla in | Study-level mean | Drift |
  |---|---|---|
  | X, own isocosts | 0.688 | −0.032 |
  | A, own isocosts | 0.687 | −0.032 |
  | P | 0.697 | −0.033 |
  | T (architecture) | 0.693 | −0.033 |

---

## 2. What changes in the paper

These are recommended edits. No paper file was edited.

**Table 1 (`paper/tables/table1_designs.tex`), Panel A, Chinchilla row.**
```
Chinchilla, profiles$^{a}$ & 137 & 9/9 & 57M--16B & 0.04--294 & 1.65 & F & 0.660 & 0.656 & 0.690 \\
 & & & & & & & (0.023) & (0.028) & (0.015) \\
```
- The Conv. column changes from T to F.
- N_F runs from 57M to 15.8B, so the N range reads the same. M runs from 0.04 to 294 (D as digitized), and the spread is 1.65 (was 1.67).
- Note a: "without them, 132 runs, M from 0.44, spread 1.48" (was 0.46 and 1.49). Source: `rb4_chinflop_ranges.csv`.
- Panel B's "Chinchilla, all runs" row is the reference technology and stays in T. In N_F it is 0.700 (κ free) and 0.740 (κ = 1); a footnote could say so.

**Table 1, Panel C.**
- Study-level mean: 0.687 [0.640, 0.734]; k = 4, τ = 0.023, Q = 8.3 (p = 0.04).
- Replace the row "FLOP accounting, η = ±0.1 | 0.665–0.708 | [0.578, 0.751] | Chinchilla, Meta" by two rows:
  - "FLOP accounting, η = ±0.1 | 0.673–0.700 | [0.598, 0.748] | Meta"
  - "Chinchilla's FLOP counts and conventions | 0.682–0.697 | | Chinchilla"
- "Other parameter conventions": 0.668–0.673 (Marin, Farseer). The recomputed rows are 0.673, 0.670 and 0.668.
- "First-derivative estimator": 0.670 [0.603, 0.737].
- Slope per decade: −0.033 [−0.076, 0.011]; wild p = 0.039. Budgets ≤ 3×10²⁰: −0.008 [−0.042, 0.027]; wild p = 0.83.
- Chinchilla and Llama 3: −0.058 (0.013), unchanged.
- σ\* at 10²¹: 0.647 [0.560, 0.735].
- σ\*, budgets ≥ 6×10²⁰: 0.594 [0.565, 0.622]; Q = 2.5.
- Note e: "Over all variants computed, the mean lies in 0.663–0.700".
- Note f: "η = −0.1 for Meta; Chinchilla's accounting is rebuilt from Hoffmann et al.'s architecture table (Online Appendix …)".
- Notes g, h: p lies in 0.03–0.13.

**Section III (`paper/sections/technology.tex`).**
- **§III.A, last sentences of the designs paragraph.** Replace "N_F = C/(6D) for Llama 3 and Marin, the total count for Chinchilla, whose token counts are themselves constructed as C/(6N), and the non-embedding count for Farseer" with:
  > "N_F = C/(6D) for Chinchilla, Llama 3 and Marin, with C each study's own FLOP count, and the non-embedding count for Farseer. For Chinchilla we rebuild FLOPs per token from Hoffmann et al.'s architecture table: their Appendix F count without its two vocabulary terms reproduces their Table A4 exactly, and the digitized FLOP coordinates of their Figure 4 are consistent with budgets set in that count and with no other (Online Appendix …)."
- **Levels paragraph.**
  - "On Chinchilla's profiles σ\* = 0.673 (standard error 0.027)" becomes "On Chinchilla's profiles, in FLOP-effective parameters, σ\* = 0.660 (standard error 0.023)".
  - "the random-effects mean is 0.687, with a 95 percent confidence interval of [0.640, 0.734]".
  - Replace the Cochran sentence with:
    > "Cochran's statistic is 8.3 on three degrees of freedom (p = 0.04): Chinchilla and Meta, whose budgets reach 3×10²¹ and 10²² FLOP, both give 0.66, and Marin and Farseer, which stop at or below 10²¹, give 0.71; the designs average over different compute windows (below)."
  - [review] This sentence is inaccurate: Meta's bracketed budgets stop at 10²¹, as Farseer's path does. It also overstates the rejection (H3 [review]). Use instead:
    > "Chinchilla and Meta both give 0.66, and Marin and Farseer 0.71. Cochran's statistic is 8.3 on three degrees of freedom (p = 0.04), but 5.8 to 6.3 (p between 0.10 and 0.12) when Chinchilla's windows or profile membership are drawn differently, and the design means average over different compute windows: σ\* falls above 3×10²⁰ FLOP in Chinchilla and Llama 3, and Marin stops there."
- **Systematic-uncertainty sentences.** Replace "Chinchilla's and Meta's accounting is unobserved, and η = ±0.1 moves the mean to 0.665 or 0.708" with:
  > "Chinchilla's accounting is observed once its FLOPs are rebuilt from the architecture table, and its elasticity is small (η between −0.04 and 0.00 across budgets). Hoffmann et al.'s other printed counts, and the total and non-embedding conventions, move the mean within 0.682–0.697. Meta's accounting is unobserved, and η = ±0.1 for Meta moves the mean to 0.673 or 0.700."
  - "Across every variant we compute … between 0.663 and 0.700."
- **Scale paragraph.**
  - "a slope of −0.033 per decade … p between 0.03 and 0.13".
  - "σ\* = 0.713 at 10¹⁹ FLOP, 0.680 at 10²⁰ and 0.647 at 10²¹".
  - "Their five budgets … agree (Q = 2.5, p = 0.65) and give σ\* = 0.594, with a 95 percent confidence interval of [0.565, 0.622]".
  - The Chinchilla–Llama 3 drift (−0.058, s.e. 0.013) and "0.48 at 10²³ and 0.42 at 10²⁴" are unchanged to two decimals.
- **The sentence "Section IV therefore uses as its reference the κ-free Chinchilla technology, whose σ\* of 0.701 matches the study-level mean" stays true.** The reference refitted in N_F gives 0.700, inside [0.640, 0.734].

**Appendix B, Chinchilla extraction (`appendix_data.tex`, §"Chinchilla Extraction").** Replace "…the constructed D inherits an error correlated with N. Online Appendix … bounds its effect on σ\*; we have not rebuilt D from Hoffmann et al.'s architecture table." with:
> "We rebuild FLOPs per token for all 50 models of Hoffmann et al.'s Table A9. Their counts are reproduced by attention, MLP and one relative-position projection per layer plus one 32,000 × d vocabulary matrix. Their Appendix F formula as printed does not reproduce their Table A4; without its embedding and final-logit terms it reproduces all six ratios. In that count, FLOPs per token are 0.98–1.15 times 6N for the models in the profiles. Figure 4 plots runs at 6ND. The within-budget pattern of the digitized FLOP coordinates is flat once the budgets are read in that count (slope −0.002, s.e. 0.007), and not in the printed count (−0.065), in executed FLOPs (−0.036) or in 6ND (+0.021). So the digitization's D = C/(6N) is the true token count, each profile is an exact isocost in Hoffmann et al.'s count, and N_F = C/(6D) is FLOPs per token over six, whatever D. The 44M model of Table A9 does not appear in the figure. The digitized offset of log₁₀ C (−0.018; Appendix B) is mostly this accounting: net of it, −0.006 remains."
- [review] Three corrections to this text:
  - "(+0.021)" should read "(+0.020)".
  - "Figure 4 plots runs at 6ND" is inferred, not observed. Write "The pattern is what one expects if Figure 4 plots runs at 6ND".
  - "does not appear in the figure" should read "does not appear in the digitization".
- [review] Add a sentence on profile membership, since m1's ±0.045-dex rule was drawn on uncorrected coordinates:
  > "Drawn on the corrected coordinates, the same membership rule gives 147 profile runs instead of 137, and σ\* of 0.668 in FLOP-effective parameters (Online Appendix …)."

**Appendix D, model-free estimator (`appendix_additional.tex`, §"The Model-Free Estimator and the Elasticity by Compute").**
- Replace "…each design is analyzed in the convention in which its budgets equal 6ND … if Chinchilla's FLOPs per token varied with N with elasticity η, σ\* would move by about 0.44η, at most 0.04 for |η| ≤ 0.1" with:
  > "…each design is analyzed in FLOP-effective parameters, the convention in which its profiles are exact isocosts. For Chinchilla, FLOPs per token rebuilt from the architecture table have η between −0.04 and 0.00 by budget, and σ\* is 0.660 against 0.673 in total parameters. It is 0.648–0.666 across Hoffmann et al.'s printed, implemented and executed counts, and 0.690 in non-embedding parameters. Moved onto each count's own exact isocosts, every count and convention gives 0.660–0.669 (Table [rb4_chinflop])."
- Budget-level paragraph:
  - "Chinchilla gives 0.666 (fixed) against 0.660 (random)".
  - "Cochran's Q no longer rejects homogeneity within Chinchilla (Q = 12.7, bootstrap p = 0.06)".
  - The Chinchilla specification range "0.665–0.714" is ra1's grid in T. Either label it as in total parameters, or re-run the grid in N_F (not done here).
  - [review] The grid has now been re-run in N_F (`rb4_chinflop_specgrid.csv`). Replace "0.665–0.714 for Chinchilla" with "0.656–0.702 for Chinchilla (0.665–0.714 in total parameters)". Add the window sentence:
    > "In every specification σ\* in FLOP-effective parameters lies 0.008 to 0.018 below its value in total parameters. At the primary specification, one run that enters the 3×10¹⁹ window accounts for about a third of the difference. The five high-loss runs lie outside every window but move the window centres: without them the primary specification gives 0.681 in FLOP-effective parameters and 0.674 in total parameters."
- Study-level sentences: 0.687 [0.640, 0.734] and 0.663–0.700.

**Online Appendix table.** Add `output/tables/rb4_chinflop.tex` (label `tab:app-chinflop`; compiles in AEA.cls with no overfull boxes).

**Downstream uses of 0.693** (`appendix_wedge.tex` l.139; `wedge.tex` l.263: "the common model-free σ\* = 0.693" in the lab-own technologies). This becomes 0.687, so k rises from 0.443 to 0.456 (+3 percent). The lab-own median share (0.51) moves by roughly +0.005, but this was not recomputed here (owner: ra2/rb2 lab-own code). The revealed-demand medians under the top-budget and drift curvatures change by at most 0.004 (H4).
- [review] Two corrections:
  - k rises from 0.443 to **0.455**, not 0.456 (1/0.6873 − 1).
  - The reviewer recomputed the lab-own columns through a wrapper that feeds rb2's own code the rebuilt study-level row (0.6873, HKSJ s.e. 0.0148, k = 4) and reproduces rb2's published outputs exactly at the old row. The one-rule median share moves from 0.515 to **0.524** (all 22), not by +0.005, and the Llama 3 8B wedge from 8.27 to 8.75.
  - rb3 does not depend on the study-level σ\* and needs no re-run.
  - Full list: `rb4_chinflop_review.md` §5.
- [review] The paper edits listed in this section are incomplete. The complete list, with the introduction, conclusion, Online Appendix Tables D and F, Figure 3 and the lab-own numbers, is in `output/memos/rb4_chinflop_review.md` §6.

**Referee response (R2 Major 1.4).**
> "Done. FLOPs per token are rebuilt for all 50 models of Table A9 with Appendix F. The printed formula does not reproduce Hoffmann et al.'s own Table A4; the formula without its vocabulary terms does, exactly, and it is also the only count consistent with the digitized coordinates of their Figure 4. In that count, η by budget is −0.04 to 0.00. Chinchilla's model-free σ\* in FLOP-effective parameters is 0.660 (0.023), against 0.673 in total parameters: an equivalent η of −0.03, inside the ±0.1 band we had carried. Across every count and budget reading it is 0.648–0.690. The study-level mean becomes 0.687 [0.640, 0.734]. The pooled drift (−0.033 against −0.032 per decade), the top-budget value (0.594 against 0.596) and the identified set move by at most 0.003, and the revealed-demand median shares by at most 0.004. The η band now applies to Meta only."
- [review] Add to the response:
  > "Across our specification grid the rebuilt count lowers Chinchilla's σ\* by 0.008 to 0.018. The level is sensitive to window membership: with the five high-loss runs removed, or with profile membership drawn on the corrected coordinates, it is 0.681 or 0.668. The study-level mean is then 0.691–0.694."

---

## 3. Methods

- **Counts and FLOPs** (`rb4_arch.py`): the formulas in H1. Seq_len = 2,048 and V = 32,000 for every model. Softmax is 3·h·S² per layer. The attention S² terms are counted at full (non-causal) cost, as Appendix F does.
- **Matching** (`rb4_estim.load_runs`): nearest Table A9 model in log N. Architecture-based T, P and F for each run.
- **Coordinate test** (`coords_test`):
  - Outcome: log₁₀C_dig − log₁₀C_b, the deviation from m1's nominal-budget assignment before m1's offset removal, plus log₁₀ r_F.
  - Regression: on log₁₀N with budget fixed effects. HC1 and CR1-by-budget s.e.; t with 8 df.
  - Also reported: the coefficient on −log₁₀ r_F.
- **Estimation** (`run_variants`): ra1's `isoflop.design_estimate`, `boot_design` and `summarize_design`, called with the Chinchilla design laid out as `isoflop_designs` lays it out, with x = ln N_acc.
  - The order is fixed at 2, as in ra1. The pooled F test of the cubic term in each count has p = 0.45–0.83, so it would choose 2 anyway.
  - "Own isocosts" corrections use the κ-free Chinchilla surface fitted to the 137 runs, with N = T and D = C_b/F_budget.
  - First-order check: L + |s_b| ln ρ, with the primary frontier slope (R6).
- **Fixed-window check** (`fixed_membership`): quadratic in x_acc on exactly the runs of ra1's windows, with ra1's frontier.
- **η by budget** (`eta_by_budget`): (i) OLS slope of ln r on ln T over each primary window's runs; (ii) a kernel local-linear slope across Table A9 at the budget's argmin (bandwidth 0.35 in ln T). The two agree to within 0.010 for T4 and 0.015 for the other counts.
- **Parametric fits** (`parametric`, `reference_fit`): ra1's `parametric.fit_chin` and `fit_kappa` (Huber, δ = 10⁻³), and ra1's FHH wild bootstrap (`isoflop._par_one`, B = 399).
- **Propagation** (`rb4_prop.py`): the hook described above. The extra study-level rows (η for Meta only; Chinchilla's other counts) are produced by rb1's own `study_level` on further summary overrides, keeping its primary row.

---

## 4. Caveats

1. **Which count is "actual".**
   - The primary uses the count that set Hoffmann et al.'s budgets (T4). This is the analogue of Llama 3 and Marin, whose C are each study's own budget counts.
   - T4 omits real compute: the unembedding matmul (count X) and Transformer-XL relative-position attention.
   - With those added, the executed count gives:

     | Count | Nominal | Own isocosts |
     |---|---|---|
     | X | 0.658 | 0.666 |
     | X plus relative-position logits (plain means of σ\*_b; R10) | 0.649 | 0.664 |

   - "Actual FLOPs" in the engineering sense therefore puts Chinchilla at 0.65–0.67.
2. **The coordinate test identifies the budget count from within-budget slopes of about 0.02–0.07 dex per decade of N.**
   - It rejects the printed count, executed FLOPs and 6ND clearly. It cannot separate T4 from counts that differ from it by less than about ±0.015 in η.
   - The within-budget scatter of the digitized coordinates (s.d. 0.016 dex) is not explained by any count. Digitized N is precise to 0.03 percent, so this scatter is probably real in the figure (tokens rounded to steps, or plotting). It is small relative to the count differences.
3. **Mechanical versus own-isocost estimates are different objects.**
   - Nominal-budget ("mechanical") estimates in counts other than T4 re-express x on the observed isocosts. This is what ra1 does for Marin's configuration count and what the m9 amendment's P and T conventions mean.
   - Own-isocost estimates are σ\* on the expansion path of a hypothetical cost function, and they rely on a parametric surface to move runs by up to 0.5 log points in D (count A at 57M).
   - Their spread (0.660–0.669) is narrower than the mechanical spread (0.648–0.690). The first-order and κ-surface corrections differ by up to 0.014 in plain means (R6).
4. **Window membership.** The rule h = 1 applied in ln N_F admits one extra run at 3×10¹⁹. That accounts for about a third of the design-level change (H3). With the current windows held fixed, the rebuilt design mean is about 0.663–0.668.
5. **Architecture assumptions.** The rebuild assumes the IsoFLOP runs used Table A9's architectures ("all models trained as part of this work") at sequence length 2,048. If small models were trained at shorter contexts, T4's attention terms, and so r at small N, would be smaller.
6. **What is not redone in N_F.**
   - ra1's Chinchilla specification grid (windows, orders, smoothers), its Monte Carlo coverage and its window diagnostics table (`appD_budgets.tex`) remain in T. The budget-level rows in N_F are in `rb4_chinflop_budgets.csv`, with the same columns.
   - The reference technology stays in T. Its N_F refit gives the same σ\*_κ (H3), but M\* in N_F units is lower (20.4 against 21.9 at 10²³, κ free). Converting the wedge analysis to N_F is a separate change.
7. **Meta (Llama 3) is still unobserved.** Its digitized N = C/(6D) is already FLOP-implied, but Meta's own count is unknown. The η = ±0.1 row now applies to Meta alone.

---

## 5. Inventory

**Code** (`code/analysis/rb4_chinflop/`)

| File | Role |
|---|---|
| `run.py` | Stage runner; caches `data/processed/rb4_chinflop/stage_*.pkl` (derived statistics only); log `run_log.txt`, `run_stdout.txt` |
| `rb4common.py` | Paths, seeds, logging, import plumbing (rb1's outputs sent to a temp root), η helpers |
| `rb4_arch.py` | Tables A9/A4 (transcribed; checked against the TeX), parameter counts, FLOP counts T4/A/X/6T/6P, Table A4 reproduction, Gopher/Chinchilla totals, source download and checksum |
| `rb4_estim.py` | Matching, coordinate test, ra1 estimator under 12 accounting variants and 6 first-order checks, fixed windows, η by budget, fit quality, Table 1 ranges, same-run and reference parametric fits |
| `rb4_prop.py` | Override CSVs and rb1 propagation (study level, meta-regression, top budgets, identified set, wedge scenarios) |
| `rb4_tables.py` | CSVs and `rb4_chinflop.tex` |
| `rb4_review.py` | Self-review checks (R1–R10); `--compare` for the determinism check |

**Tables** (`output/tables/`)

| File | Content |
|---|---|
| `rb4_chinflop.tex` | Online Appendix table: Panel A (which count set the budgets), Panel B (Chinchilla σ\* by count), Panel C (propagation) |
| `rb4_chinflop_arch.csv` | 50 Table A9 models: counts, W_R and embedding shares, F under each count, r, N_F, runs matched |
| `rb4_chinflop_tableA4.csv`, `_tableA4_summary.csv`, `_totals.csv` | Table A4 reproduction; Gopher/Chinchilla totals |
| `rb4_chinflop_match_summary.csv`, `_coords.csv`, `_eta.csv` | Matching quality; coordinate test; η by budget and count |
| `rb4_chinflop_budgets.csv`, `_summary.csv`, `_fixedwin.csv`, `_fitq.csv`, `_ranges.csv` | Budget-level and design-level σ\* by variant (current values beside them); fixed-window check; fit quality; Table 1 row quantities |
| `rb4_chinflop_param.csv`, `_reference_fit.csv` | Same-run parametric σ\*; reference technology in T and N_F |
| `rb4_chinflop_study_level.csv`, `_metareg_slopes.csv`, `_metareg_predictions.csv`, `_top_budget.csv`, `_pi_sigmaC.csv`, `_wedge_scenarios.csv` | Propagation; `run` = old, new, alt_* |
| `rb4_chinflop_headline.csv` | Every number quoted in this memo, current against rebuilt |
| `rb4_chinflop_review_checks.csv` | Self-review checks. [review] R11–R14, the ra1-grid and rb1-symwin reproduction rows and the grid summaries were added by the reviewer |
| `rb4_chinflop_specgrid.csv` | [review] ra1's 12 specifications in T and N_F (T4), on 137 and 132 runs, with N_F − T by specification |
| `rb4_chinflop_design_top.csv` | [review] rb1's design-top table (each design's largest bracketed budget, raw σ\*_b, BLUP design line), old and rebuilt |

**Data.** `data/raw/rb4_chinflop/` holds the arXiv source (git-ignored). `data/processed/rb4_chinflop/` holds the override CSVs, stage caches and logs.

**Bibliography.** Existing keys only: `hoffmann2022training` and `besiroglu2024chinchilla` in the table. If the Gopher compute figures are cited, `rae2021scaling` is already in `references.bib`.

---

## 6. Referee comments addressed

- **R2 round 2, Major 1.4 / round 1, Major 9(b).** Done (H1–H4). η by budget: `rb4_chinflop_eta.csv`. The ±0.04 band becomes −0.014 for Chinchilla.
  - The round-1 request also asked for a, M\* and the Chinchilla-70B wedge under a rebuilt D. For the reference technology in N_F, a is 0.520 against 0.514, and M\*(10²³) is 14.9 against 18.8 (κ = 1) and 20.4 against 21.9 (κ free); σ\*_κ is unchanged.
  - The Chinchilla-70B wedge was not recomputed. It needs the reference technology in N_F and the 70B model's N_F = 67.1B (F_T4/6; r = 0.958).
- **R2 round 2, Major 1 (accounting term in the headline interval).** The study-level interval now carries Meta's η band only, plus Chinchilla's count and convention range (H4).
- **R2 round 1, minor 10 (57M against 44M).** The 44M model is absent from the digitization (H2).
- **R4 R2-M4.4 (FLOP-effective parameters throughout).** Chinchilla is now in N_F like Llama 3 and Marin. Table 1's Conv. column reads F for all three.

---

## Self-review

Reviewer: Claude, as its own skeptical referee. Date: 2026-09-24. Checks are in `output/tables/rb4_chinflop_review_checks.csv` (65 rows, 0 failed), produced by the `review` stage.

**1. Re-run from scratch.**
- The full pipeline was re-run into an empty scratch root (`RB4_OUTPUT_ROOT`). The stage caches in the project root were deleted before the final run.
- `rb4_review.py --compare` found 54 of 54 output files identical byte for byte (23 table files, 31 override CSVs).
  - [review] After the reviewer's changes it finds 56 of 56 identical: 25 table files, including the new `_specgrid.csv` and `_design_top.csv`, and 31 override CSVs. The project outputs were regenerated by one end-to-end run of the revised code (40 s). Against the builder's outputs, only three things change: the label of two 'old'-run rows in `_study_level.csv`, the rows added to `_review_checks.csv`, and the two new files. No builder number changed.
- The final project outputs come from one end-to-end run of the final code (`data/processed/rb4_chinflop/run_stdout.txt`, 29 s).

**2. Validation of the FLOP formula against Hoffmann et al.'s Table A4 (R4).**
- Printed Appendix F: 0 of 6 ratios; the 73M model gives 1.58 against 1.03.
- Executed count X: 0 of 6.
- T4: 6 of 6 exactly (max |error| 0.005). With Table A4's rounded parameter labels instead of the architecture counts, 4 of 6; the label "1.1B" is 1.143B in Table A9.
- Table A9 counts are reproduced to 0.8 percent.
- The validation therefore does not confirm Appendix F as printed; it identifies the implemented count. The independent evidence of the Figure 4 coordinates points to the same count.
- One check fails in the source itself: Hoffmann et al.'s 6.3×10²³ for Gopher is not reproduced by any count (4.7×10²³). Nothing in the module depends on it.

**3. Reproduction of upstream numbers.**
- R1: through this module's wrapper, the T variant reproduces ra1's nine Chinchilla σ\*_b, s.e., intervals, M\*_b, curvatures and frontier slopes to 10⁻¹⁶, and ra1's RE, FE, drift and Q exactly.
- R1: the same-run parametric T row reproduces ra1's `isoflop_param` row.
- R1: the reference refit in T reproduces m2's Table 3 row exactly (0.7368, 0.7006, M\* 18.81).
- R2: the "old" propagation reproduces rb1's 16 study-level rows, 42 meta-regression rows, identified-set table and wedge-scenario table to relative 5×10⁻⁶, rb1's %.6g rounding. The wild p-values match exactly.
- R3: the override CSVs differ from ra1's only in Chinchilla's rows.

**4. Adversarial checks and what they found.**
- **Is the −0.014 an accounting effect or a window artifact?** Partly the latter. One run enters the 3×10¹⁹ window. With ra1's windows held fixed, the design mean falls by about 0.010 instead of 0.014 (R9), and every budget moves by −0.018 to +0.001. The η approximation S_F/S_T ≈ (1+η)⁻² tracks the fixed-window ratios to within 0.035 per budget (R5). The residual comes from the non-linearity of ln r in ln T across architectures of different shape.
  - *Action:* both numbers are reported, and the caveat is in §4. The primary keeps ra1's window rule, as the task specified ("windows unchanged").
- **Is the coordinate test driven by the five outlying runs?** No (R7): 132 runs give T4 −0.001 (0.008), 6T +0.021 (0.005), A −0.065 (0.015) and X −0.036 (0.011). By budget, the slope under 6T is positive in 8 of 9 budgets. Under T4, 3 of 9 are positive, as expected for noise.
- **Are the own-isocost corrections model-driven?** Moderately (R6). The κ-surface and first-order (frontier-slope) corrections differ by −0.010 to +0.014 in plain means of σ\*_b. The corrected rows are therefore sensitivities. The primary needs no correction.
- **Does W_R belong in P?** It does not matter: P without W_R gives 0.691 against 0.691 (R8).
- **Does an executed count with relative-position attention change the range?** Its mechanical plain mean is 0.649, and on its own isocosts 0.664 (R10). Both are inside the N_F range, 0.648–0.666 under the supported reading.
- **Could the loss data themselves pick a convention?** No. The residual s.d. of the budget quadratics on identical runs is 0.0116 in every count (`_fitq.csv`).
- **Did the builder's first framing hold?** No, and it was changed.
  - The task framed D as C_b/(6N_reported) and C_actual = F·D. The coordinate test rejects that reading (6T row, p = 0.002), and that reading would give 0.671 for N_F (T4), within the reported range.
  - The memo states that N_F = F/6 does not depend on how D is recovered, which removes the circularity for the primary.
- **Numbers in this memo.** Every number was compared with the generated CSVs (`_headline.csv`, `_summary.csv`, `_study_level.csv`, `_metareg_*.csv`, `_eta.csv`, `_coords.csv`, `_review_checks.csv`).
  - One correction was made while writing: the within-Chinchilla homogeneity p is the bootstrap p (0.025 → 0.060), which the paper quotes, not the χ² p (0.048 → 0.12).
- **LaTeX.** `rb4_chinflop.tex` compiles in the paper's preamble (AEA.cls, tectonic) with no overfull or underfull boxes of its own. The first version was 79 pt too wide and was re-laid out: short row labels, η as a budget mean, σ\*_b at 10²¹ only.

**5. Open after review.**
- (i) ra1's Chinchilla specification grid and Monte Carlo in N_F.
  - [review] The grid is done: `rb4_chinflop_specgrid.csv` gives N_F 0.656–0.702 on 137 runs. The Monte Carlo and rb1's symmetric-window drift check remain in T.
- (ii) The lab-own technologies with σ\* = 0.687 instead of 0.693 (about +0.005 in the lab-own median share, not recomputed).
  - [review] Recomputed through a wrapper (review memo §5): +0.009 for all 22 models (0.515 → 0.524). rb2 itself still has to be re-run once rb1's study-level row is updated.
- (iii) Whether to move the reference technology and the wedges to N_F. The reference's σ\*_κ is unchanged (0.700), but M\* shifts. This is a paper-level decision.
- (iv) Meta's FLOP count remains unobserved.
