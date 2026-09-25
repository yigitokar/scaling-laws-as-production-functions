# Memo — module rb1_sigmaC: the elasticity of substitution as a function of compute, and more extrapolation checks

> **[rb4 integration, 2026-09-24]** The lead author adopted module rb4_chinflop (Chinchilla's FLOP accounting rebuilt from Hoffmann et al.'s architecture table; `output/memos/rb4_chinflop.md` and its review). rb1's primary outputs now put Chinchilla in FLOP-effective parameters N_F = F_T4/6 (0.660, s.e. 0.023) instead of total parameters (0.673, s.e. 0.027). Numbers in the body of this memo are the pre-integration values (Chinchilla in T) unless marked.
> - **Switch.** `RB1_CHINCHILLA` (in `rb1common.py`): `rb4` (default, primary) reads rb4's override inputs `data/processed/rb4_chinflop/override_new_isoflop_{budgets,summary}.csv`, in which only Chinchilla's rows differ from ra1's; `ra1` is the sensitivity with Chinchilla in T, as first published, and writes to `output/sensitivity/rb1_chinchilla_T/` (its 32 table files are byte-identical to the pre-integration outputs). In `rb4` mode the study-level variants carry η = ±0.1 for Meta only (Chinchilla's accounting is observed) and add rows for Chinchilla's nine other counts and conventions and for its three window/membership sensitivities (rb4 review R11–R13). Not switched: `driftmc.py` and `symwin.py` (still Chinchilla in T; rb4's review R14 gives symwin in N_F).
> - **Check.** The primary study-level, meta-regression, prediction, top-budget, design-top, identified-set and wedge-scenario CSVs equal rb4's `run = new` rows exactly (as written, %.6g); rb4's review stage re-checks this (R2b).
> - **What changed (old → new; `rb1_sigmaC_*.csv`).** Study-level mean 0.693 [0.653, 0.733] → **0.687 [0.640, 0.734]**; τ 0.017 → 0.023; Q 5.7 (p = 0.13) → 8.3 (p = 0.041), but 5.8–6.3 (p = 0.10–0.12) under Chinchilla's window/membership variants, so heterogeneity is borderline; range of variant means 0.663–0.708 → 0.663–0.700. Pooled drift −0.032 [−0.075, 0.010], wild p 0.042 → −0.033 [−0.076, 0.011], wild p 0.039 (p across weighting, scale and test 0.03–0.12 → 0.03–0.13); budgets ≤ 3×10²⁰ −0.007 (p 0.85) → −0.008 (p 0.83); Chinchilla + Llama 3 FE −0.058 (0.013), unchanged. Pooled line at 10¹⁹/10²⁰/10²¹: 0.715/0.683/0.650 → 0.713/0.680/0.647. σ\*_top 0.596 [0.567, 0.624], Q 1.9 → 0.594 [0.565, 0.622], Q 2.5. Identified-set lower bound at 10²³/10²⁴ 0.480/0.422 → 0.479/0.421. Median s at σ\*_top / pooled line / lower bound 0.888/0.906/0.978 → 0.890/0.910/0.978. Unchanged: extrap, tuning, driftmc and symwin outputs (they do not use Chinchilla's model-free estimate).
> - **Downstream.** rb2_decisions re-run (lab-own one rule now uses 0.687); rb3_econ2 does not read these files. Paper edits: `paper/notes/integration_log_rb4.md`.


Module owner: rb1_sigmaC (Claude). Date: 2026-09-24. Entry point: `code/analysis/rb1_sigmaC/run.py`.

**Scope.** Round-2 requests R1 New 2; R3 N1 (a)–(e); R4 R2-M4; R2 Major 1 (requests 1–6) and Major 5 (5.1, 5.2). Binding decisions: `paper/notes/revision_plan_v3.md` §1 (σ\* headline, "in one recipe" rule) and §2 (Table 1 row, Figure 3 panel, Figure 5).

**Reproduction.**
- One command, `.venv/bin/python code/analysis/rb1_sigmaC/run.py`, regenerates every number, table and figure below. It reads the reviewed outputs of ra1_modelfree (budget-level σ\*_b, Farseer path, design summaries), ra2_wedge (clean sample, reference zero points, per-model bounds on ln M − ln M\*(C)) and m8_measurement (Step Law inefficiency gradients), and `data/raw` through ra1's loaders. Code of ra1/ra2/m8 and `sl.py` are imported, never edited.
- Stages: meta → pid → extrap → tuning → driftmc → tables → figures (137 s on 4 CPU processes; no GPU). Seeds fixed (`rb1common.SEED = 20260925`, one stream per key). `RB1_QUICK=1` smoke test; `RB1_OUTPUT_ROOT` redirects all outputs.
- Determinism: a second end-to-end run of the same code into a scratch root reproduced all table files of that version byte for byte (the only difference was a subsample row added to the meta stage between the two runs). Final outputs come from one end-to-end run (`data/processed/rb1_sigmaC/run_stdout.txt`).
- **Independent review (2026-09-24; `output/memos/rb1_sigmaC_review.md`).** The builder's version was re-run from scratch and reproduced all 30 output tables byte for byte. The reviewer then corrected the code, outputs and this memo. The review file lists every change. Changes marked "[review]" below replace the builder's text. After the fixes, the module writes 32 table files (27 CSV and 5 .tex), including a new `symwin.py` stage and the `rb1_sigmaC_budgets.tex` table.

**Notation.** k = 1/σ\* − 1 = S/2 (the wedge scale: ln w = k ln(M/M\*) in the κ family); s = (w − 1)/w; budgets C in FLOP; "per decade" = per unit of log10 C.

---

## 1. Headline findings

### H1. σ\* falls with compute in the two designs that reach 10^21 FLOP. The pooled drift is borderline in significance and comes from budgets above ≈3×10^20 (task 1)
Meta-regression of the 44 budget-level model-free estimates (36 bracketed IsoFLOP budgets of Chinchilla, Llama 3, Marin ×3, plus Farseer's local path at 8 compute levels as a sixth design) on log10 C. Files: `rb1_sigmaC_metareg_slopes.csv`, `_metareg_predictions.csv`, `_metareg_design_slopes.csv`; table `rb1_sigmaC_table.tex` Panels A–C; figure `rb1_sigmaC_panel` (a).

| Specification (budgets) | Slope per decade | Model s.e. | CR2 s.e. by design [df] | Wild cluster p |
|---|---|---|---|---|
| **Six designs, 3-level RE, inverse variance (44)** | **−0.032** | 0.010 | 0.014 [3.2] | 0.042 |
| Six designs, design RE, unweighted (44) | −0.031 | 0.009 | 0.011 [4.5] | 0.034 |
| Six designs, design-balanced (CE-RVE) weights | −0.033 | – | 0.011 [3.2] | 0.039 |
| IsoFLOP designs only, design FE, IV (36) — R3's computation | −0.051 | 0.009 | (df 1.5) | 0.22 |
| Chinchilla + Llama 3, design FE (17) | −0.058 | 0.013 | 2 clusters | – |
| Marin (3 corpora), design FE (19) | −0.001 | 0.023 | 3 clusters | – |
| **All budgets ≤ 3×10^20 (37), 3-level RE** | **−0.007** | 0.015 | 0.012 [3.7] | 0.85 |
| Leave out each design's largest budget (38), 3-level RE | −0.029 | 0.013 | 0.018 [3.2] | 0.29 |
| Hinge at 3×10^20: slope above the knot (descriptive) | −0.151 | 0.038 | 0.028 [2.2] | 0.053 |
| Porian et al. (unannealed; not pooled), FE | **+0.040** | 0.006 | – | – |

- **[review] Significance is borderline and depends on the scale and the test.** The CR2 t-test p-values are 0.096 for the primary weighted fit and 0.043 unweighted. On the scale S = 2(1/σ\* − 1), which is the scale the wedge uses (k = S/2), the three-level RE slope is +0.137 per decade: CR2 p = 0.12, wild p = 0.065. With six clusters, the wild p (0.042) is the preferred test, but quote it together with these. Write "p between 0.03 and 0.12 across weighting, scale and small-sample test" (`rb1_sigmaC_metareg_slopes.csv`: `p_cr2_design`, `p_wcr_design`), not "p = 0.04".
- **Reading.** The pooled slope is zero where all designs overlap (≤ 3×10^20: −0.007, p = 0.85). The pooled drift of about −0.03 per decade comes from the budgets above 3×10^20 in Chinchilla and Llama 3 (their own drift −0.058, s.e. 0.013). It loses significance if either design, or each design's largest budget, is dropped: leave-one-design-out slopes run from −0.021 to −0.046, and the leave-top-out wild p is 0.29.
  - **[review] Design-specific slopes** (FE, IV weights, s.e. scaled by the residual variance s² = 2.47, the convention of every other FE row; the builder's version was unscaled): Chinchilla −0.060 (0.033), Llama 3 −0.058 (0.016), Farseer +0.001 (0.011), Marin −0.015 (0.086) / +0.018 (0.064) / −0.008 (0.052). Wald test of equal slopes 11.2 on 5 df, p = 0.047 (unscaled: 27.7, p < 0.001).
  - **[review] "No drift below 3×10^20" means no *common* drift.** In the common window the design-specific slopes are heterogeneous (Wald 13.3, p = 0.021). Llama 3 falls at −0.050 (0.018), the same rate as over its full range, driven by its precise 3×10^20 budget. Farseer rises at +0.038 (0.017), and Chinchilla (+0.018, 0.059) and Marin are flat. These slopes depend on the weights: unweighted OLS gives Llama 3 −0.025 and Farseer −0.009. `rb1_sigmaC_metareg_design_slopes.csv`, sample "budgets <= 3e20".
  - **[review] Farseer.** With inverse-variance weights its path is flat (+0.001). ra1's own path drift is −0.015 (s.e. 0.008, p = 0.048), and its largest level (10^21, 0.658) lies well below its 2–5×10^20 values (0.72–0.74). Farseer is therefore not evidence *against* the drift. Its top level agrees with the two IsoFLOP designs.
- **Reproduces the referees.** R3: IV weights, design FE, IsoFLOP designs −0.0507 (0.0094) ✓; Chinchilla + Llama 3 −0.058 (0.013) ✓; Marin −0.001 (0.023) ✓. R1's "about −0.05 per decade" is the IsoFLOP-only FE number. [review] Random effects alone change it little (IsoFLOP-only three-level RE: −0.046). Adding Farseer's path as a sixth design takes it to −0.032.
- **σ\* by compute level** (population line, CR2 95%): 10^19: **0.715** [0.705, 0.726]; 10^20: **0.683** [0.648, 0.717]; 10^21: **0.650** [0.565, 0.736] (unweighted: 0.721, 0.691, 0.660). [review] The model-based intervals are [0.689, 0.742], [0.663, 0.703] and [0.620, 0.680]. At 10^19 the CR2 interval is the narrower one, because the six designs happen to agree there; quote the wider interval.
  - Extrapolated: 0.618 at 10^22, 0.585 at 10^23, 0.553 at 10^24. The CR2 intervals ([0.49, 0.75], [0.41, 0.76], [0.34, 0.77]) are uninformative.
  - On the S scale (linear in 1/σ\*): 0.718, 0.684, 0.654, 0.626 at 10^19–10^22.
  - **[review] Per design at 10^20** (R3 N1(a), "a common compute level"; design line = BLUP intercept + common slope; `rb1_sigmaC_design_top_budgets.csv`): Chinchilla 0.681, Llama 3 0.672, Marin 0.682 / 0.685 / 0.686, Farseer 0.692. The raw budget estimates at 10^20 (Chinchilla 0.764, Llama 3 0.670, Farseer 0.713) or 9×10^19 (Marin DCLM 0.767, Nemotron-CC 0.722) are noisier.
- **σ\* at each design's largest bracketed budget** (raw σ\*_b, s.e.): Chinchilla 0.579 (0.054) at 3×10^21; Llama 3 0.609 (0.017) at 10^21; Farseer path 0.658 (0.011) at 10^21; Marin 0.686 (0.089), 0.737 (0.060), 0.733 (0.052) at 3×10^20.
- **Top-budget value.** The five budgets at 6×10^20 FLOP and above in Chinchilla and Llama 3 are homogeneous (Q = 1.9, p = 0.75, τ = 0) with inverse-variance mean **σ\*_top = 0.596 [0.567, 0.624]** (HKSJ). Adding Farseer's 10^21 level: RE 0.611 [0.564, 0.658] (heterogeneous, Q = 19).
- **The estimator neither invents nor erases a drift** (R2 Major 1.3; `rb1_sigmaC_driftmc.csv`). Truths built on each design's actual (N, D) grid by rescaling the fitted Chinchilla-form profiles so that σ\*(C) follows a known path (frontier and argmins unchanged); ra1's primary estimator, noise at each design's residual s.d., R = 300:

  | Design | True drift | Noise-free estimate | MC mean (s.d.) | Share below null 5th pct |
  |---|---|---|---|---|
  | Chinchilla | 0 / −0.03 / −0.06 | +0.005 / −0.025 / −0.055 | −0.001 (0.036) / −0.025 (0.035) / −0.048 (0.035) | 0.05 / 0.14 / 0.37 |
  | Llama 3 | 0 / −0.03 / −0.06 | +0.001 / −0.029 / −0.059 | −0.007 (0.026) / −0.037 (0.024) / −0.065 (0.021) | 0.05 / 0.23 / 0.72 |
  | Marin DCLM | 0 / −0.03 / −0.06 | +0.004 / −0.027 / −0.057 | −0.004 (0.049) / −0.029 (0.046) / −0.058 (0.047) | 0.05 / 0.14 / 0.31 |

  No spurious drift under constant σ\*; drifts are recovered with at most 20% attenuation (Chinchilla). A single design has little power against −0.06 per decade with an unweighted slope (0.31–0.72); in particular **Marin's "no drift" is weak evidence** (power 0.31 at −0.06, over two decades that stop at 3×10^20).
- **[review] Count-symmetric windows and a power-law frontier with E fitted** (R2 Major 1.3; new `symwin.py`, `rb1_sigmaC_driftmc_symwin.csv`; OLS drift with design FE, point estimates).
  - **Symmetric windows.** Each budget's window is trimmed to equal numbers of runs on the two sides of the fitted minimum, and the quadratic is refitted. Chinchilla + Llama 3: −0.054 (0.025), against −0.051 (0.017) for the primary windows. Chinchilla −0.035, Llama 3 −0.081.
  - **Power frontier.** E + K e^{−γc} with E fitted: −0.050 (0.016).
  - **Both:** −0.053 (0.025).
  - **Marin.** Symmetric windows leave only 2–3 runs per side, so the check is noisy there: +0.021 (0.030).
  - **Reading.** The drift in the two designs that carry it is not an artifact of asymmetric windows or of the log-cubic frontier's slope at the edge of its support. (ra1's grid already contained the power frontier at the design level: Chinchilla −0.072, Llama 3 −0.043. The builder's "not done" understated this.) Quartic fits remain untried; ra1's grid stops at cubic and h = ∞.

### H2. Study-level headline interval: 0.69 [0.65, 0.73]; 0.66–0.71 across accounting, bandwidth and convention (task 2)
One estimate per study (Hoffmann, Meta, Marin as one study, Farseer; k = 4), DerSimonian–Laird mean with the modified HKSJ interval (t_3). Marin = equal-weight mean of its corpora with the s.e. of perfectly correlated errors (shared code and FLOP accounting); Farseer = first-derivative path estimate 0.708 with the s.e. of its path mean (0.005, which does not treat compute levels as independent: R4 R2-M4 item 3). File `rb1_sigmaC_study_level.csv`; table Panel D.

| Variant | Mean [95% HKSJ] | τ | Q (p) |
|---|---|---|---|
| **Headline (each study in its primary convention)** | **0.693 [0.653, 0.733]** | 0.017 | 5.7 (0.13) |
| Marin's corpora independent (reproduces R1's ≈[0.66, 0.73]) | 0.696 [0.662, 0.730] | 0.014 | 5.7 (0.13) |
| IsoFLOP studies only (k = 3) | 0.678 [0.616, 0.740] | 0 | 1.9 (0.39) |
| Six designs, Marin separate (round-2 Table 1 C) | 0.695 [0.673, 0.717] | 0 | 4.1 (0.54) |
| FLOP accounting: Marin in configuration N (0.64–0.66) | 0.677 [0.621, 0.733] | 0.029 | 12.0 (0.007) |
| FLOP accounting: η = −0.1 / +0.1 for Chinchilla and Meta | 0.665 [0.578, 0.751] / 0.708 [0.693, 0.722] | | |
| Bandwidth: Farseer's Hessian path at the CV bandwidth (0.664) | 0.665 [0.651, 0.680] | 0 | 2.7 (0.43) |
| Convention: Farseer with embeddings (0.670) | 0.671 [0.656, 0.685] | 0 | 2.1 (0.54) |
| Convention: Marin config N and Farseer total N ("lowest") | 0.669 [0.654, 0.683] | 0 | 1.0 (0.81) |
| [review] Design-level fixed-effect means for Chinchilla (0.684) and Llama 3 (0.628) (R2 Major 1.1) | 0.681 [0.606, 0.755] | 0.044 | 44.2 (<0.001) |
| [review] Estimator: local first-derivative 1/(1+b₁) for Marin (0.669) and Llama 3 (0.629), as for Farseer | 0.673 [0.607, 0.739] | 0.036 | 14.5 (0.002) |

- The study-level mean moves over **0.663–0.708** across these rows; R4's "both" row (six designs) was 0.660 [0.638, 0.682].
- **[review] Estimator sensitivity.** The headline uses the local first-derivative estimator for Farseer (0.708) and ra1's windowed IsoFLOP estimator for the other three studies (Marin 0.700–0.713). Applied to Marin's local-wedge surfaces (task 4), the same first-derivative estimator gives 0.666–0.676. This is an estimator range of about 0.04 on Marin, on the same side as the convention range. With it, or with fixed-effect design means, the interval widens to about [0.61, 0.75] and the mean falls to 0.67–0.68. Both rows are in Panel D.
- **Direction of the accounting uncertainty.** In Marin, where it is observed, FLOP-implied N gives the *higher* σ\* (0.70–0.71 against 0.64–0.66 in configuration N). The estimand is an elasticity with respect to FLOP-effective parameters, N_F = C/(6D).

### H3. Beyond 10^21 FLOP σ\* is partially identified; with it the wedges' magnitudes are bounded below, not bracketed, by the reference (task 3)
Files `rb1_sigmaC_pi_sigmaC.csv`, `_wedge_scenarios.csv`, `_wedge_models.csv`, `_magnitude_bounds.csv`; table `rb1_sigmaC_wedge.tex`; figure `rb1_sigmaC_panel` (b).

- **Identified set** for C above the largest bracketed budgets, σ\*(C) ∈ [σ_lin(C), σ\_top]: upper bound σ\_top = 0.596 (σ\* does not rise with compute beyond the designs); lower bound = linear continuation of the Chinchilla–Llama 3 drift (−0.058 per decade) from 10^21, truncated to (0, 1).

  | C (FLOP) | 10^22 | 10^23 | 10^24 | 10^25 | 10^26 |
  |---|---|---|---|---|---|
  | σ\* bounds (Chinchilla–Llama drift) | [0.538, 0.596] | [0.480, 0.596] | [0.422, 0.596] | [0.363, 0.596] | [0.305, 0.596] |
  | σ\* lower bound, pooled drift (−0.032) | 0.563 | 0.531 | 0.498 | 0.466 | 0.433 |
  | k = 1/σ\* − 1 bounds | [0.68, 0.86] | [0.68, 1.09] | [0.68, 1.37] | [0.68, 1.75] | [0.68, 2.27] |
  | Memo: k at σ\* = 0.70 | 0.43 | 0.43 | 0.43 | 0.43 | 0.43 |

  If the drift is not real (Marin shows none up to 3×10^20), σ\* = 0.70 applies; the drift-agnostic set is [σ\_lin(C), 0.70]. R1's "0.45–0.55 at 10^24" lies inside it. [review] Farseer should not be cited as a no-drift design (see H1).
- **[review] What the bounds do not cover.**
  - *Sampling error.* The bounds are estimated bounds, not a confidence set. σ\_top has HKSJ interval [0.567, 0.624]. At its upper end, k = 0.60 (CSV `pi_sigmaC`: `sigma_upper_ci95_hi`, `k_lower_ci95`).
  - *An accelerating decline.* The lower bound is not conservative against one. The descriptive hinge gives −0.15 per decade above 3×10^20 (H1), so "falls no faster than the measured drift" is an assumption, not an implication of the data.
- **Revealed demand** (77 clean models, reference zero points held fixed; 76 of 77 train above 10^21, median 2×10^23 FLOP):

  | Curvature | Median w | Median s | Median s, decision units (56) | Llama 3 herd s_f | OLMo 2 7B w | SmolLM2 1.7B w |
  |---|---|---|---|---|---|---|
  | Reference (σ\* = 0.701) | 3.99 | **0.749** | **0.736** | 0.294 | 4.0 | 11.3 |
  | σ\* = 0.65 | 5.71 | 0.825 | 0.813 | 0.349 | 5.7 | 21.3 |
  | [review] σ\* = 0.624 (upper 95% limit of σ\_top) | 7.02 | 0.858 | 0.847 | 0.378 | 7.0 | 30.7 |
  | **Top budgets, σ\* = 0.60** | 8.64 | **0.884** | **0.874** | 0.405 | 8.6 | 44.2 |
  | σ\*(C_i), pooled meta-regression line | 10.7 | 0.906 | 0.886 | 0.519 | 10.7 | 57.5 |
  | σ\*(C_i) = lower bound, Chinchilla–Llama drift | 44.9 | 0.978 | 0.963 | 0.750 | 41.3 | 513 |
  | σ\*(C_i) = lower bound, pooled drift | 19.3 | 0.948 | 0.933 | 0.590 | 19.2 | 157 |

  The share with ŵ > 1 (0.974) does not depend on σ\*. Llama 3 8B: 6.7 (reference) → 19.5 (σ\* = 0.60), reproducing R2's table (19.6).
- **Magnitude bounds of Prop. 5(iv)** (median of per-model bounds on s; ra2's bounds on ln M − ln M\*(C), reproduced exactly):

  | Curvature range | PI-1 | PI-2 | PI-3 |
  |---|---|---|---|
  | k ∈ [0.40, 0.52] (round 2) | [0.589, 0.933] | [0.589, 0.887] | [0.545, 0.941] |
  | k ∈ [0.40, 0.67] (adds the top budgets) | [0.589, 0.969] | [0.589, 0.940] | [0.545, 0.974] |
  | k ∈ [0.40, k_lin(C_i)] (drift-agnostic) | [0.589, 0.998] | [0.589, 0.992] | [0.545, 0.998] |
  | k ∈ [0.68, k_lin(C_i)] (drift maintained) | **[0.776, 0.998]** | [0.776, 0.992] | [0.734, 0.998] |
  | [review] k ∈ [0.60, k_lin(C_i)] (drift maintained, σ\_top at its upper 95% limit 0.624) | [0.734, 0.998] | [0.734, 0.992] | [0.691, 0.998] |

- **What this means.** For w > 1, ln w rises with k. If σ\* falls with compute, the reference level (σ\* = 0.70) is a **lower bound** on the wedge and share, not a midpoint.
  - The reference median s of 0.75 becomes 0.88 at the top-budget curvature. [review] It is 0.86 even at the upper 95% limit of σ\_top (0.624), so the lower-bound reading survives sampling error in σ\_top.
  - The upper magnitude bound becomes uninformative (s → 1).
  - The ordinal results (sign; share > 1) are unaffected.
  - [review] All of this holds each model's reference zero point M\*_ref(C) fixed. "Lower bound" is conditional on that convention and on M\*_ref, not on σ\* alone.

### H4. A second and a third recipe agree: parametric extrapolation in M understates the wedge on Marin and Llama 3; the convexity replicates on Marin, not on Llama 3 (task 4)
Local wedge on IsoFLOP designs, w = (f_c − f_x)/(f_c + f_x) with f = ln L, x = ln M, c = ln C, from a kernel-weighted local quadratic in (x, c); parametric κ-free and Chinchilla-form (κ = 1) Huber fits on all runs and on M ≤ 100; wild cluster bootstrap by budget (999 draws), every fit recomputed. Files `rb1_sigmaC_extrap_*.csv`; table `rb1_sigmaC_extrap.tex`; figure `rb1_sigmaC_extrap`.

**[review] Where the local wedge is actually evaluated.** The designs' runs reach M = 3,706 (Marin) and M = 643 (Llama 3). With n_eff ≥ 8, the local wedge is evaluated only up to **M ≈ 1,110 on Marin and M ≈ 290 on Llama 3** (`rb1_sigmaC_extrap_bandwidths.csv`: `M_max_runs`, `M_max_evaluated`; `extrap_delta.csv`: `M_max_evaluated` by bin).
- Marin's "≥ 1,024" bin is 1–2 points at M ≈ 1,080–1,110.
- Llama 3's "256–1,024" bin is two points at M ≈ 288–290.
- The paper should say "up to M ≈ 1,100 (Marin) and M ≈ 290 (Llama 3)", not "M up to 3,706 / 643". Farseer's evaluated range reaches M ≈ 2,570.

Δ = ln(w_param/w_local) (95% basic interval); w_param/w_local in parentheses:

| Design | κ free, all runs, M 64–256 | κ free, all runs, M 256–1,024 | κ = 1 fitted on M ≤ 100, M 256–1,024 | Convexity b₂ |
|---|---|---|---|---|
| Farseer (ra1; 404 runs) | +0.00 (1.00) | −0.14 [−0.15, −0.13] (0.87) | −0.95 [−0.97, −0.93] (0.39) | +0.020 [0.019, 0.022] |
| Marin, Comma (85) | −0.09 | −0.43 [−0.62, −0.44] (0.65) | −1.09 [−1.26, −1.02] (0.34) | +0.032 [0.024, 0.042] |
| Marin, DCLM (84; 1 screened) | −0.09 | −0.45 [−0.62, −0.39] (0.63) | −0.95 [−1.18, −0.82] (0.39) | +0.027 [0.019, 0.038] |
| Marin, Nemotron-CC (88) | −0.08 | −0.39 [−0.53, −0.35] (0.68) | −1.04 [−1.14, −0.98] (0.35) | +0.027 [0.022, 0.033] |
| **Marin, three corpora pooled** | −0.09 [−0.16, −0.12] | **−0.43 [−0.55, −0.43] (0.65)** | **−1.03 [−1.15, −0.98] (0.36)** | |
| Llama 3 (133; digitized) | −0.08 [−0.15, −0.05] | −0.38 [−0.57, −0.23] (0.69; 2 points) | −0.90 [−1.02, −0.72] (0.41) | **−0.029 [−0.040, −0.014]** |

- **The understatement at high M replicates** in both new recipes, for every comparator, and grows with M. At M ≥ 1,024 (Marin only; 1–2 corner points per corpus, n_eff 8–11, fragile) Δ is about −1.0 (κ free) and −1.7 (κ = 1 on M ≤ 100). The κ-free fit on M ≤ 100 understates by −0.61 to −0.72 at 256–1,024 on Marin (Farseer: −0.42).
- **Convexity replicates on Marin** (b₂ = +0.027 to +0.032 against Farseer's +0.020), with the same consequence: ln w bends up beyond the path, which no linear-in-u form captures.
- **Llama 3 is concave** (b₂ = −0.029), and bandwidth-sensitive (−0.052 to +0.010). There the understatement at high M comes from a steeper local slope at the path: the local first-derivative σ\* is 0.63 against 0.69 (κ free) and 0.77 (κ = 1). Consistent with this, the parametric forms *overstate* w below M = 16 in Llama 3 (Δ = +0.27 κ free). [review] No other design shows this for the κ-free form (Farseer −0.02, Marin −0.09). Farseer's Chinchilla form does overstate below M = 16 (+0.16).
- **By-product.** The local first-derivative path σ\* = 1/(1 + b₁): Marin 0.666 / 0.667 / 0.676 (ra1's windowed estimator: 0.700 / 0.713 / 0.705), Llama 3 0.629 (ra1: 0.660).
- **Robustness** (`rb1_sigmaC_extrap_sensitivity.csv`): κ-free Δ at 256–1,024 is −0.37 to −0.53 on Marin across bandwidths ×0.75, ×1.5, h_x ×0.75, h_c ×1.5, n_eff ≥ 6/10 and own-corpus CV; −0.26 to −0.86 on Llama 3. Without the monotonicity screen DCLM gives −0.84. b₂ on Marin is 0.022–0.044 in every variant (0.016–0.034 restricted to M < 1,024).
- **Basic intervals** correct for the estimator's bootstrap bias (CSV `boot_bias`); on Marin's 84–88 runs the κ-free bias (+0.04 to +0.10) is of the order of the sampling spread and points toward a *larger* understatement, so several intervals lie below their point estimates. Run-level Rademacher intervals are similar (CSV `lo_run`, `hi_run`).
  - [review] **Source of the bias.** It comes from the parametric (Huber, δ = 10⁻³) refits, not from the local wedge. Across draws the local ln w in each bin moves by at most 0.02 from its bootstrap-world value, and no evaluation point is lost.
  - [review] **New columns.** The CSVs now carry the interval's centre, `delta_bc` = estimate − bias, and the pooled s.e.
  - [review] **What to quote.** For Marin pooled, κ free at 256–1,024: −0.43 (bias-corrected −0.49, s.e. 0.03) [−0.55, −0.43]. For κ = 1 on M ≤ 100: −1.03 (bias-corrected −1.07, s.e. 0.04). Avoid printing an interval next to a point estimate it excludes without saying why.

### H5. Mis-tuning explains at most a small part of Farseer's convexity, and D-biased inefficiency works against it (task 5)
Correcting Farseer's local elasticities for the measured inefficiency gradients (ε = ε^obs + ι), relocating the path and re-estimating b₂. File `rb1_sigmaC_tuning.csv`; table `rb1_sigmaC_tuning.tex`.

| Inefficiency gradients applied (m8, Step Law grid) | ι_n, ι_d (×10⁻³) | b₂ corrected | Tuning share of b₂ |
|---|---|---|---|
| Step Law rule (Farseer's policy), in sample | −0.48, +0.07 | 0.0200 | **+0.02** |
| Step Law rule, 95% box (four corners) | | 0.0188–0.0213 | **−0.04 to +0.08** |
| Random configuration, best of 1 (the "D-biased" inefficiency) | −4.38, −6.95 | 0.0381 | **−0.87** |
| Random, best of 4 / 16 | | 0.0235 / 0.0213 | −0.15 / −0.05 |
| Other rules: fixed, Porian base, Porian N-rule, DeepSeek, Bjorck | | | −0.80 to +0.22 |
| Stochastic-frontier-shaped gradients (Step Law / random level) | varies | 0.0205 / 0.0249 | −0.01 / −0.22 |
| **Breakdown**: ι_d needed to remove all convexity (ι_n = 0) | 0, **+26.0** | 0 | +1.00 |

- Inefficiency that falls with D (the measured D bias) raises the observed ε_D, lowers the observed w most where ε_D is small (high M), and so *hides* convexity: correcting for it makes ln w more convex.
- To manufacture all of the convexity, the policy's inefficiency would have to *rise* with D by 0.026 log-loss per log point: 47 times the upper 95% bound of Step Law's in-sample ι_d (0.00055), and the opposite sign of every untuned policy's D-gradient.
- D6's path formula gives the same sign as the pointwise path shift in every scenario, 2–7 times larger. The implied path-exponent bias is at most 0.006 for the Step Law rule (0.043 for Porian's N-rule).
- **Caveat.** The gradients are measured on Step Law's grid (M 19–466, N 0.21–1.07B). Farseer's convexity is driven by M up to 2,570 at 0.1–0.34B parameters, where the rule's inefficiency is unmeasured. The residual Bjorck–Step Law disagreement on the D-elasticity of the optimal learning rate (m8, 0.34–0.37) is the kind of out-of-sample mis-tuning that could produce ι_d > 0. It would need to be large.

---

## 2. Methods

**Task 1 — meta-regression** (`meta_c.py`).
- **Inputs.** ra1's per-budget σ\*_b with design-conditional bootstrap s.e. (Chinchilla 9, Llama 3 8, Marin 5/7/7 valid budgets) and Farseer's Hessian-based local-path σ\*(C) at 8 levels (5×10^18–10^21) with its bootstrap s.e. Porian's 24 budgets are analysed separately.
- **Primary model.** Three-level random-effects meta-regression (Konstantopoulos 2011), σ\*_bj = μ + β(log10 C − 20) + θ_j + e_bj, with θ_j ~ N(0, τ_d²) and e_bj ~ N(0, se² + τ_w²). REML; GLS for (μ, β).
- **Variants.**
  - Unweighted: linear mixed model with design random intercepts.
  - Design-balanced correlated-effects RVE weights 1/[k_j(v̄_j + τ²)] (Hedges, Tipton and Johnson 2010).
  - Design fixed effects with inverse-variance weights and scaled s.e. (R3's computation), and unweighted.
  - Linear in S = 2(1/σ\* − 1).
  - Hinge at 3×10^20 (descriptive; data-chosen knot).
  - Subsamples: Chinchilla + Llama 3; Marin; IsoFLOP-only; ≤ 3×10^20; leave one design out; leave each design's top budget out.
  - Design-specific slopes with a Wald test of equality.
- **Inference.**
  - CR2 cluster-robust s.e. by design (Bell and McCaffrey 2002) under the working covariance, with Bell–McCaffrey/Satterthwaite degrees of freedom computed under the working model (Pustejovsky and Tipton 2018; Tipton 2015). Moore–Penrose square roots handle absorbed fixed effects.
  - CR2 by study (Marin = one cluster) as a check.
  - Restricted wild cluster bootstrap-t by design with Webb six-point weights (9,999 draws; 1,999 for subsamples), holding the working covariance fixed.
  - CR2 and wild p are reported only with ≥ 4 designs and ≥ 2 df.
- **Predictions.** Population-average predictions at 10^19–10^24 FLOP with model-based and CR2 intervals; design lines = BLUP intercept + common slope.
- **Top-budget value.** Inverse-variance/DL mean of the budgets ≥ 6×10^20 in Chinchilla and Llama 3, with an HKSJ interval.

**Task 2 — study-level interval** (`meta_c.study_level`). Uses ra1's meta routine (DL + modified HKSJ, t_{k−1}).
- **Marin aggregation.** Mean of the three corpora, with s.e. under ρ ∈ {1, 0.5, 0}; ρ = 1 is primary.
- **Farseer.** First-derivative estimate 1/(1 + b₁) = 0.708, s.e. = max(its wild cluster s.e., the plain-mean path s.e. 0.005).
- **Variants.** Configuration-N values for Marin (ra1 sensitivity rows); FLOP-accounting elasticity η = ±0.1 for Chinchilla and Meta via σ\*_eff = 2/[2 + S/(1 + η)²]; Farseer at the extended-grid CV bandwidth and with embeddings. The convention table is `rb1_sigmaC_conventions.csv`.

**Task 3 — partial identification and propagation** (`pid.py`).
- **Bounds.** σ\*(C) ∈ [σ\_lin(C), σ\_top] beyond 10^21, with maintained assumptions of monotone non-increase and drift no faster than the measured within-design drift.
- **Wedges.** ln w_i = k(C_i) ln(M_i/M\*_ref(C_i)), holding each model's reference zero point fixed (the reference slope 0.42744 is recovered exactly from ra2's wedges).
- **Decision units.** 11 family-level W_f = [Σ ω_j/w_j]⁻¹ (common-D families, ω = compute shares), recomputed under each curvature, plus 45 members of size-specific families and singletons. This reproduces R3's 0.736.
- **Magnitude bounds.** ra2's `pi_bounds_models.csv` (per-model dlo, dhi under PI-1..4) with ra2's formula and widened k ranges. The round-2 row reproduces ra2's medians to 10⁻¹⁶.

**Task 4 — local wedge on IsoFLOP designs** (`extrap.py`).
- **Data and convention.** ra1's loaders, in the FLOP-implied convention (runs lie exactly on 6ND isocosts).
- **Monotonicity screen.** Where a configuration recurs across budgets (Marin), a run whose loss exceeds the lowest loss of the same configuration on fewer tokens by more than 10⁻⁴ is dropped. On these data this drops one run: DCLM, 157M at 1.8×10^20, loss 3.769 against 3.605 at half the tokens.
- **Local surface.** Gaussian product-kernel local quadratic of ln L in (ln M, ln C).
- **Bandwidths.**
  - LOO-CV over multiples of the coordinate s.d. (0.15–3.0), with h_c ≥ half the largest adjacent-budget gap in ln C (so every kernel spans two budgets).
  - One bandwidth for Marin's three corpora (the pooled criterion; the smallest c-multiple within 1% of the CV minimum, since the criterion is flat beyond 1 s.d.): (m_x, m_c) = (0.45, 1.3).
  - Llama 3: (1.3, 0.3).
- **Evaluation points.** 14 log-spaced M on each budget line within its observed range; n_eff ≥ 8.
- **Local path.** M\*_local(C_b) is the root of ln w_local on each budget line.
- **Parametric comparators and statistics.** Parametric comparators are ra1's `parametric.fit_chin` and `fit_kappa` (Huber). Δ is averaged by ra1's M bins. b₂ comes from OLS of ln w_local on (u, u²).
- **Bootstrap.**
  - Primary: wild cluster by budget (Webb weights, B = 999) of leverage-adjusted residuals around the pilot surface, with every fit warm-started and recomputed.
  - Comparison: run-level Rademacher (B = 499).
  - Basic intervals are centred on the noise-free pilot population, as in ra1.
- **Marin pooled.** Equal-weight mean of the corpora, with intervals from index-wise averaged bootstrap deviations.

**Task 5 — tuning share** (`tuning.py`).
- **Surface and data.** ra1's Farseer local quadratic (primary CV bandwidth; farseer.py imported) with f_x and f_z at ra1's grid; ra1's b₂ = 0.0204 is reproduced exactly.
- **Correction.** Corrected ln w = ln(−f_x + ι_n) − ln(−f_z + ι_d). The path is relocated by Brent's method on each isocost with the corrected gradients, and ln w = b₁u + b₂u² is re-estimated on the same points (n_eff ≥ 15, inside the path's C range).
- **Scenarios.** m8's per-policy gradients (with cluster-robust s.e.), and SFA-shaped non-constant gradients (γ_N = −0.063, γ_D = −0.260 around Step Law's design centre).
- **Breakdown value.** Found by Brent's method.
- **D6 formula.** −(ι_n − ι_d)E/[(α+β)R\*(C)] on Farseer's Chinchilla-form fit (ra1 stage cache).

**Drift Monte Carlo** (`driftmc.py`). Described in H1.

---

## 3. Inventory

**Code** (`code/analysis/rb1_sigmaC/`)

| File | Role |
|---|---|
| `run.py` | Stage runner; caches `data/processed/rb1_sigmaC/stage_*.pkl`; logs `run_log.txt`, `run_stdout.txt` |
| `rb1common.py` | Paths (upstream inputs), seeds, logging, helpers (renamed from common.py to avoid m1's `common`) |
| `meta_c.py` | Task 1 (3-level REML, CR2 + BM df, wild cluster bootstrap, subsamples, S-scale, hinge) and task 2 |
| `pid.py` | Task 3 |
| `extrap.py` | Task 4 |
| `tuning.py` | Task 5 |
| `driftmc.py` | Drift Monte Carlo |
| `symwin.py` | [review] Drift under count-symmetric windows and an E-fitted power frontier (run inside the driftmc stage) |
| `tables_rb1.py`, `figures_rb1.py` | Outputs |

**Paper-ready LaTeX** (`output/tables/`; booktabs, `tabular*`, `\footnotesize`, AEA.cls `tablenotes`; each compiles without overfull boxes in the AER class, text width 385.5 pt)

| File | Proposed use |
|---|---|
| `rb1_sigmaC_table.tex` | **Main text (Table 1 addition).** Panel A: drift; B: σ\* at 10^19/10^20/10^21; C: largest-budget values and σ\*_top; D: study-level interval with accounting/bandwidth/convention rows. For the ≤ 8-exhibit budget, Panels B and D (≈12 rows) can be merged into Table 1 and the rest moved to the Online Appendix |
| `rb1_sigmaC_wedge.tex` | Online Appendix F (partial identification of σ\*(C); wedges and bounds under the widened curvature) |
| `rb1_sigmaC_extrap.tex` | Online Appendix F (companion of Figure 5) |
| `rb1_sigmaC_tuning.tex` | Online Appendix D (flexible inputs) |
| `rb1_sigmaC_budgets.tex` | [review] Online Appendix D. Every budget-level σ\*_b with runs left/right of the minimum, and FE beside RE pooled values per design (R2 Major 1.1) |

**Figures** (`output/figures/`, .pdf and .png)

| File | Proposed use |
|---|---|
| `rb1_sigmaC_panel` | **Figure 3 merge.** (a) σ\*_b by budget, six designs, pooled weighted/unweighted lines, CR2 band; (b) identified set for σ\*(C) to 10^26 with the clean sample's training compute as a rug |
| `rb1_sigmaC_extrap` | **Figure 5.** (a–c) Δ by M bin for Farseer, Marin (pooled, with corpora shown faintly) and Llama 3 under three comparators; (d–f) local ln w against u with quadratic fit and tangent |

**CSVs** (`output/tables/rb1_sigmaC_*.csv`)
- Inputs and meta-regression: `budgets_input`, `metareg_slopes`, `metareg_predictions`, `metareg_design_slopes`, `design_top_budgets`, `top_budget_sigma`.
- Study level: `study_level`, `conventions`.
- Partial identification: `pi_sigmaC`, `wedge_scenarios`, `wedge_models` (per model: w and s under reference, 0.60, the meta line and the PI lower bound), `magnitude_bounds`.
- Extrapolation: `extrap_delta` (by design × comparator × bin, with bias, cluster and run-level s.e. and intervals), `extrap_convexity`, `extrap_sensitivity`, `extrap_sigma`, `extrap_marin_pooled`, `extrap_grid` (every evaluation point), `extrap_bandwidth_cv`, `extrap_bandwidths`, `extrap_screened_runs`.
- Tuning: `tuning`, `tuning_breakdown`, `tuning_d6`.
- Drift Monte Carlo: `driftmc`, `driftmc_truth`; [review] `driftmc_symwin` (symmetric windows, power frontier).

**Bibliography.** `lit/bib/extra_round3_rb1_sigmaC.bib`. Keys: `konstantopoulos2011fixed`, `hedges2010robust`, `tipton2015small`, `pustejovsky2018small`, `bell2002bias`, `manski2003partial`. All were verified against Crossref (the DOI of each) or the publisher's copy (Bell and McCaffrey, Statistics Canada) on 2026-09-24. Existing keys used: `dersimonian1986meta`, `hartung2001tests`, `sidik2002simple`, `webb2023reworking`, `li2025predictablea/b`, `marin2026ladders`, `grattafiori2024llama`, `czech2026llama3isoflop`, `hoffmann2022training`, `besiroglu2024chinchilla`, `porian2024resolving`.

---

## 4. Claims for the paper (with caveats)

**C1. σ\* headline (abstract, p. 3, §III.B, conclusion).**

> "About 0.70 at 10^19–3×10^20 FLOP, where every design has budgets and the pooled estimates show no common drift. Declining at larger budgets, to about 0.60 at 6×10^20–3×10^21 FLOP in the two designs that reach them. In FLOP-implied parameter units."

- [review] The builder's "and none drifts" is not accurate. Below 3×10^20, Llama 3's own inverse-variance slope is −0.050 (s.e. 0.018) and Farseer's is +0.038 (0.017); see H1.

- **Evidence:**
  - Study-level interval 0.693 [0.653, 0.733] (k = 4); 0.663–0.708 across accounting, bandwidth and convention.
  - No common drift at ≤ 3×10^20 (pooled −0.007, p = 0.85; design slopes heterogeneous there).
  - Chinchilla–Llama 3 drift −0.058 (0.013); top-budget σ\* 0.596 [0.567, 0.624].
- **Caveat on the plan's "10^19–10^21".** Revision plan §1 writes "≈0.70 at 10^19–10^21". At 10^21 the pooled line is 0.650 [0.565, 0.736], the two IsoFLOP designs give 0.61–0.62 raw and Farseer's path 0.66. We recommend "up to 3×10^20" (or "10^19–10^20.5"), or "0.65–0.70 at 10^19–10^21".
- **Other caveats:**
  - Two of the four studies are digitized (Chinchilla, Llama 3).
  - The drift rests on two designs and on budgets above 3×10^20: it is not significant without either design or without each design's largest budget.
  - [review] The pooled drift's p-value lies between 0.03 and 0.12, depending on the weights, the scale (σ\* or S) and the small-sample test.
  - [review] The study-level mean falls to 0.67–0.68 with an interval of about [0.61, 0.75] under two further sensitivities: the local first-derivative estimator for Marin and Llama 3, and fixed-effect design means.
  - A single design has low power against −0.06 per decade (MC 0.31–0.72), so Marin's flat profile is weak evidence of no drift.

**C2. Report the drift as a meta-regression in the main text** (Table 1 row or text).

> [review] "Pooling the 44 budget-level estimates with design random effects, σ\* falls by 0.03 per decade of compute (cluster-robust s.e. 0.014 on about three degrees of freedom; wild cluster bootstrap p = 0.04, cluster-robust t-test p = 0.10). The fall comes from the budgets above 3×10^20 FLOP in the two designs that reach them. At or below 3×10^20 the pooled slope is zero (−0.007), though the designs' own slopes differ there."

- **Caveats.**
  - Digitized per-budget s.e.: in ra1's design-level fixed-effect pooling (S scale), Llama 3's largest budget carries 62% of that design's weight. In this σ-scale meta-regression it carries 18% of Llama 3's inverse-variance weight, and 15% once the within-design variance τ_w² is added.
  - Marin's shorter range.
  - The CR2 degrees of freedom are about 3.
  - [review] On the S scale (the wedge's scale) the drift has CR2 p = 0.12 and wild p = 0.065.

**C3. σ\* at frontier compute is partially identified, and the reference wedge is a lower bound** (§IV.C/D, Prop. 5(iv), Table 2 notes).

> "Beyond 10^21 FLOP we bound σ\*(C) between the value at the largest budgets (0.60) and a linear continuation of the drift (0.48 at 10^23, 0.42 at 10^24). With σ\* at 0.60 the clean-sample median s is 0.88 (0.87 over decision units), against 0.75 (0.74) at the reference 0.70. If σ\* falls with compute, the reference levels are lower bounds."

- **Magnitude bounds.** Replace [0.59, 0.93] (PI-1) with [0.59, 0.97] when the top-budget curvature is added, and [0.78, 1.00] when the drift is maintained. [review] With σ\_top at its upper 95% limit, the drift-maintained bounds are [0.73, 1.00]. At σ\* = 0.624 the median s is 0.86, which is still above the reference 0.75.
- **Caveat.** Holding the reference zero points fixed is the round-2 convention. A drifting σ\* also violates the quasi-homotheticity used in the closed forms, so these are scale propagations, not re-estimations. Sign results are unaffected.

**C4. "Parametric extrapolation is conservative" now holds in three recipes** (§IV, Figure 5; revision plan: say "in one recipe" unless Marin/Llama 3 agree).

> "In Farseer, Marin's three corpora and Llama 3, parametric forms fitted to the whole design understate the model-free wedge at M = 256–1,024. On Marin the κ-free wedge is 0.65 of the local one, and a Chinchilla-form fit to M ≤ 100 gives 0.36. The gap grows with M. The convexity of ln w that drives it on Farseer replicates on Marin (b₂ = 0.027–0.032). On Llama 3 the gap comes from a steeper local slope instead."

- **Caveats:**
  - Three recipes, at 10^18–10^22 FLOP. [review] The local wedge is evaluated up to M ≈ 2,570 (Farseer), ≈ 1,100 (Marin) and ≈ 290 (Llama 3). Marin's runs reach 3,706 and Llama 3's 643, but there is too little kernel support at those M.
  - Llama 3's high-M bin has two evaluation points (at M ≈ 290), and its local ln w is concave.
  - [review] The Marin κ-free intervals are bias-corrected basic intervals. Quote "−0.43 (s.e. 0.03)", or the bias-corrected −0.49 with [−0.55, −0.43].
  - The M ≥ 1,024 bin on Marin rests on corner points.
  - The magnitude is bandwidth-sensitive; the sign is not.
  - [TBD-m9: the experiment's check under Amendment 1] should be added when available. The paper may state the direction as "in three recipes (Farseer, Marin, Llama 3)", not in general.

**C5. Tuning does not produce Farseer's convexity** (App. D6; R2 Major 5.2).

> "Applying the measured inefficiency gradients of Step Law's own rule changes the convexity by −4% to +8%. The D-biased inefficiency of untuned configurations would make it larger, not smaller. Explaining the whole convexity would need inefficiency rising with training tokens at 47 times the rule's measured bound."

- **Caveat.** The gradients come from Step Law's grid (M ≤ 466), not from Farseer's high-M corner.

**C6. Recommendation to economists** (conclusion, "What economists should use").

> "For substitution between FLOP-effective parameters and tokens: 0.70 at 10^19–3×10^20 FLOP, about 0.60 at 10^21 in the two designs that reach it, and partially identified beyond, in [0.42, 0.60] at 10^24 if the measured drift continues."

- **Caveat.** State the unit (FLOP-implied parameters) and that total- or configuration-parameter conventions give 0.04–0.07 less.

**C7. Porian et al.** Their unannealed profiles drift the other way (+0.040 per decade, s.e. 0.006), consistent with a schedule artifact that shrinks with training length. [TBD-m9: the annealed experiment's σ\* by budget, exploratory per R3 N1(e)].

---

## 5. Robustness (beyond Section 1)

- **Meta-regression.**
  - Weights: IV-weighted, unweighted and design-balanced give the same slope (−0.031 to −0.033).
  - Fixed effects dilute the drift when Farseer's path enters (−0.021; its IV weights are large and its slope is flat).
  - Leave-one-design-out slopes (3-level RE): −0.026 (no Chinchilla), −0.021 (no Llama 3), −0.034 to −0.035 (no Marin corpus), −0.046 (no Farseer).
  - The drift survives dropping each design's largest budget in point estimate (−0.029) but not in significance.
  - CR2 by study (4 clusters) gives s.e. 0.013–0.014 (df ≈ 2.7).
- **Estimator artifacts.**
  - The drift MC shows no spurious drift and ≤ 20% attenuation.
  - ra1's specification grid gives Chinchilla drifts of −0.034 to −0.096 and Llama 3 −0.043 to −0.116 across windows, orders and frontier smoothers (`ra1_modelfree_isoflop_sens.csv`).
  - Porian's opposite drift shows that the estimator does not force a negative slope.
  - [review] Count-symmetric windows and an E-fitted power frontier leave the Chinchilla + Llama 3 drift at −0.050 to −0.054 (`driftmc_symwin`).
- **Study level.** Every variant's mean lies in 0.663–0.708; τ ≤ 0.05. [review] The widest interval is the η = −0.1 row, [0.578, 0.751]. Among the other rows the widest are the review's fixed-effect-means row ([0.606, 0.755]) and local first-derivative row ([0.607, 0.739]); the builder's "IsoFLOP-only interval [0.616, 0.740] is the widest" was not correct.
- **Wedge propagation.** Reproductions: ra2's PI medians (exact), R3's decision-unit median (0.736), R2's σ\*-sensitivity table (0.88 at 0.60; Llama 3 8B 19.5).
- **Extrapolation.** Bandwidth, n_eff, screening and own-CV variants in H4. Parametric κ-free σ\* on the same Marin runs: 0.668–0.678 (all runs), 0.695–0.709 (M ≤ 100). Failed fits: none in 5,992 draws.
- **Tuning.** The SFA-shaped non-constant gradients agree with the constant-gradient rows. The D6 formula and the pointwise path shifts agree in sign in all 13 constant-gradient scenarios.

**Development failures (superseded, documented).**
1. **Name clash.** A module file named `common.py` shadowed m1's `common` (imported by ra1's loaders). Renamed to `rb1common.py`.
2. **DCLM c-bandwidth.** The unconstrained LOO-CV chose a c-bandwidth covering a single budget (0.21 in ln C), driven by DCLM's anomalous 1.8×10^20 small-model runs. f_c is then not identified. Fixed by the h_c floor and the monotonicity screen; the unscreened result is reported.
3. **First drift-MC truth.** Letting the N-exponent vary with ln(ND) moved the level of σ\* as well as its slope, and could not be calibrated. It was replaced by the profile-rescaling truth, which fixes argmins and frontier and sets σ\*(C) exactly.
4. **Table widths.** The AER text width is 385.5 pt; tables were re-laid out with estimates over intervals and short labels. AEA.cls renders `\scriptsize` larger than `\footnotesize`, so it is not used.

---

## 6. Referee comments addressed (comment → response)

- **R1 New 2.1 (meta-regression with design random effects; σ\*(C) beyond 10^21–10^22 as partial identification).**
  - Done (H1, H3). Three-level RE meta-regression, weighted and unweighted, with design-robust inference. The referee's −0.05 is the IsoFLOP-only FE value, reproduced.
  - Identified set [σ\_lin(C), 0.596] beyond 10^21, truncated to (0, 1).
- **R1 New 2.2 (propagate into Prop. 5(iv) and the level of s).** Done (H3; `rb1_sigmaC_wedge.tex` Panels B–C). k range widened from [0.40, 0.52] to [0.40, 0.67] and to [0.68, k_lin(C)]. Median s under σ\* = 0.60 is 0.88 (0.87 over decision units).
- **R1 New 2.3 (qualify abstract, p. 3, conclusion).** Wording in C1, C6. We flag that "10^19–10^21" in the revision plan should read "up to 3×10^20" or "0.65–0.70 at 10^19–10^21".
- **R1 New 2.4 (study-level summary, Marin one study; convention range).** Done (H2): 0.693 [0.653, 0.733]; R1's [0.66, 0.73] reproduced with independent Marin corpora; convention and accounting rows beside it.
- **R3 N1 (a) (σ\* at a common compute level and at each design's largest budget; the meta-regression in the main text with caveats).** Done (H1; table Panels B–C; figure panel a).
- **R3 N1 (b) (conditional headline).** C1.
- **R3 N1 (c) (revealed demand under the top-budget σ\*; reference as lower bound).** Done (H3).
- **R3 N1 (d) (FLOP-accounting and bandwidth uncertainty beside the interval).** Done (H2, Panel D).
- **R3 N1 (e) (experiment's own drift, exploratory).** [TBD-m9]; to be labelled exploratory (not in the pre-analysis plan).
- **R3 minors 10, 20.** "Weak" evidence of non-homotheticity is replaced by "significant within the two designs that reach 10^21, absent in the pooled estimates below 3×10^20" (H1). The conclusion's σ\* is conditioned on compute (C6).
  - [review] The t-statistics are 2.8 (Chinchilla) and 5.4 (Llama 3) by ra1's bootstrap, and 4.4 for the two pooled (FE, scaled s.e.). The builder's "t ≈ 3 and 6" used the unscaled IV s.e. With the scaled convention they are 1.8 and 3.7.
- **R4 R2-M4.1 (alternative rows in Table 1 C).** Done (Panel D rows).
- **R4 R2-M4.2 (widen or state convention range).** State "0.66–0.71 across conventions and FLOP accounting" beside 0.69 [0.65, 0.73].
- **R4 R2-M4.3 (one Farseer estimator; s.e. not treating levels as independent).** The first-derivative estimate 0.708 is used throughout the study-level interval, with the path-mean s.e. (0.005). The Hessian-based path is used only for the budget-level meta-regression, where levels are the unit.
  - [review] Figure 3's panel and Table 1's Panel C therefore still show the Hessian path. Its plain mean (0.707) equals the first-derivative estimate, and the paper should say so in one sentence.
  - [review] The same first-derivative estimator applied to Marin gives 0.666–0.676, against 0.700–0.713 from ra1's estimator. This row is now in Panel D.
- **R4 R2-M4.4 (FLOP-effective parameters).** C1 wording.
- **R4 R2-M4.5 ("annealed" in abstract).** Recommended; Porian's opposite drift reported separately (C7).
- **R2 Major 1.1 (budget-level estimates; fixed vs random effects).**
  - The builder provided a budget-level CSV (`rb1_sigmaC_budgets_input.csv`, from ra1), the top-budget fixed-effect mean (0.596), and FE against RE rows for the *slope* in Panel A.
  - [review] That did not answer the request, which asks for an appendix table and for FE beside RE for the *level*. Added:
    - `rb1_sigmaC_budgets.tex`: every budget with its runs left/right and M\*_b, and per design the FE and RE pooled σ\* (Llama 3: FE 0.628 against RE 0.660; Chinchilla 0.684 against 0.673);
    - a study-level row with FE design means (0.681 [0.606, 0.755]).
- **R2 Major 1.2 (pooled meta-regression; σ\* at 10^22–10^24 with intervals).** Done (H1): 0.618/0.585/0.553, with CR2 intervals that span 0.34–0.77.
- **R2 Major 1.3 (estimator artifact checks).**
  - Monte Carlo under a drifting truth: done (H1).
  - Leave-the-largest-budget-out: done.
  - Window, order, frontier-smoother variants: ra1's grid.
  - [review] Symmetric windows and a power-law frontier with E fitted: done (`symwin.py`; H1). The drift survives: Chinchilla + Llama 3 −0.054 and −0.050, against −0.051. Quartic fits are not done.
- **R2 Major 1.4 (Chinchilla FLOP rebuild).** Not done (Open issues). The η = ±0.1 band is carried into the interval.
- **R2 Major 1.5, 1.6 (conditional headline; lower curvature in the bounds).** C1, H3.
- **R2 Major 5.1 (second recipe; Marin M ≤ 3,706 and Llama 3 M ≤ 643; fit-on-M ≤ 100; convexity).** Done (H4). [review] The local wedge is evaluated only to M ≈ 1,100 (Marin) and ≈ 290 (Llama 3), so the high-M comparison R2 asked for is partial on Llama 3.
- **R2 Major 5.2 (tuning share of Farseer's convexity via D6 and Step Law).** Done (H5).
- **R2 minors 2, 20.** The homogeneity statement should add that the designs average over different compute windows and that the test has low power (H1 table; MC power).

---

## 7. Open issues

1. **The m9 experiment** (pending tomorrow). Two items to fill:
   - [TBD-m9: exploratory drift of σ\* across the experiment's six budgets at 10^15–10^17 FLOP (R3 N1(e)); run the budget-level estimates through `meta_c` as a seventh design, reported separately].
   - [TBD-m9: local-wedge extrapolation check per Amendment 1; `extrap.py` can take the experiment's IsoFLOP runs directly].
   - Cite the pre-analysis plan (commit bd5c0ad), Amendment 1 (a5f47f8; wording fix b2e9bf8) and the public repository (github.com/yigitokar/scaling-laws-as-production-functions, first push 2026-09-24 18:10 +03) in §III.E.
2. **Chinchilla FLOP accounting** (R2 Major 1.4): rebuilding FLOPs per token from Hoffmann et al.'s architecture table would turn the η = ±0.1 band into a number. Not done.
3. **R2 Major 1.3 residual items.** [review] Symmetric windows and the E-fitted power frontier are now run (`symwin.py`; point estimates without bootstrap s.e.). Quartic fits remain undone.
4. **The partial-identification lower bound assumes linear drift.** A hinge fit (data-chosen knot at 3×10^20) gives −0.15 per decade above the knot (wild p = 0.053) and would put σ\* far lower at frontier compute. The drift-maintained set is therefore not conservative against an accelerating decline. Nor is its upper bound conservative against a reversal (Farseer's path is flat to 5×10^20 and falls only at 10^21).
5. **Reference zero points held fixed.** With σ\* varying in C, quasi-homotheticity fails. A re-estimation of M\*(C) jointly with σ\*(C) (a non-homothetic technology) is beyond this module. The ordinal results and the PI sign results are unaffected.
6. **Llama 3's high-M evidence is thin** (3 runs with M > 200). Its concavity is bandwidth-sensitive (b₂ from −0.052 to +0.010).
7. **Marin's M ≥ 1,024 bin** rests on 1–2 corner evaluation points per corpus (n_eff 8–11), where ε_D is near zero. Quote 256–1,024 as the headline bin.
8. **Out-of-sample tuning.** Step Law's rule is unmeasured at Farseer's high-M corner. The Bjorck–Step Law learning-rate disagreement (m8) is the channel that could produce ι_d > 0.
