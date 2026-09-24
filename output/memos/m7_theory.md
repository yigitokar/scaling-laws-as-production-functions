# Memo — module m7_theory: verifying and proving the theory

Date: 2026-09-23. Author: m7_theory agent, for the writers of "Scaling Laws as Production Functions".

> **Reviewer note (2026-09-23).** An independent review re-ran the module from scratch and corrected several items. They are marked *[review]* below; details are in `output/memos/m7_theory_review.md`. The substantive changes are:
> - the common-allocation-rule CEG statement is "only if", not "iff";
> - the information bound is for S = α+β, not directly for σ*;
> - the se(ln M̂*) curve in `m7_theory_information` was the se of ln M* at C = 6 FLOP and is now at the design's central compute;
> - the Chinchilla transverse spread is 0.48 in the 240-run estimation sample (0.53 used all 245 raw rows);
> - the partial-identification band at C = 7.2e23 extrapolates in compute as well as in M;
> - the nonparametric drift identity (finding 3) is now also verified outside the κ-family.

What this module produced:
- a verification suite that runs end to end: `code/analysis/m7_theory/run.py`, 61 checks after the independent review (57 original + 4 reviewer additions), **61 PASS, 0 FAIL**, about 3.5 minutes on one CPU;
- a formal appendix in LaTeX: `paper/sections/appendix_proofs.tex`, which compiles with the AEA class after the `main.tex` fix in §6.1;
- a proposed main-text version with a numbering map: `paper/notes/theory_main_text.md`.

Everything below is either **verified** (true as stated), **corrected** (true only with the change given), **false** (with a counterexample and the correct statement), or **new** (a result not in the drafts).

- Nothing here is estimated from new data, with one exception: a bootstrap *illustration* of the partial-identification result. It uses the Epoch Chinchilla extraction and the shared `sl.fit_chinchilla`.
- Parameter values used in checks are taken from the literature ledger (SYNTHESIS §2): Hoffmann rounded and TeX, Besiroglu, and Muennighoff.

---

## 1. Headline findings

1. **Every formal claim in `model_spec.md` and SYNTHESIS P1–P8 was checked twice: symbolically (sympy) and by a method that does not use the closed form** (brute-force optimization, Monte Carlo with n ≥ 4×10⁵, direct Fisher-information computation).
   - Checks by type (after review): 31 VERIFIED, 21 NEW, 6 IMPRECISE (true only with the correction given), 3 FALSE (with counterexample). The three FALSE checks all concern SYNTHESIS P4: a symbolic and a numeric check of the frontier comparison, and *[review]* a numeric check of the common-allocation-rule comparison.
   - As statements, the tally is:
     - one **false** claim, SYNTHESIS P4 (constant compute-equivalent gains);
     - one **substantively wrong** claim, "the sign of the Hicks bias is not identified" (model_spec Prop. 5; SYNTHESIS P3 and Summary item 3);
     - about six **imprecise** statements;
     - one task conjecture, 2(a), that holds for a different object than stated.
   - Register: `output/tables/m7_theory_claims.csv`.

2. **New: interior compute optima reveal gross complementarity (σ < 1)** (Lemma A2, Corollary A1).
   - For any smooth two-input technology, σ = P/(P+Q) in log coordinates, with P = F_nF_d(F_n+F_d) > 0 and Q = −(F_nn F_d² − 2F_nd F_nF_d + F_dd F_n²).
   - Under the multiplicative cost C = 6ND, a point with ε_N = ε_D is a strict local cost minimum iff 0 < σ < 1 *[review: the lower bound was missing here; the appendix states it correctly]*.
   - For σ > 1 the same point is a cost *maximum*. Numerically, loss curvature along the isocost is +0.127 for Besiroglu and −0.090 for a CES with σ = 1.43.
   - Economic content: the implicit price of parameters (6D) rises with the other input, so tangency in log space needs complementarity.
   - Consequence: observational-equivalence arguments can never push σ* above 1.

3. **Correction of the DMR statement: the *sign* of biased technical change is identified; its magnitude and σ are not** (Proposition A3).
   - Nonparametric identity: ∂ln M*(C,t)/∂t = −[σ*/(1−σ*)]·B_t. Here B_t is the Hicks bias (the change in the MRTS at fixed inputs) and σ* ∈ (0,1) by finding 2. *[review]* Verified numerically outside the κ-family: a non-separable three-term technology with σ* ≈ 0.79, 3 progress patterns × 3 budgets, maximum relative error 1.3×10⁻⁶ (check P5.drift.gen).
   - So the drift of the compute-optimal tokens-per-parameter ratio signs the bias, and neutrality (no drift) is testable.
   - In the κ-family with factor augmentation, g_N and g_D are identified *separately*: frontier drift γ(g_N+g_D), path drift ag_D − bg_N, and a, γ are identified.
   - σ* and |B| = S·|ag_D − bg_N| are not. An explicit family is identical to 10⁻¹⁴ in ln R* across 4 members (σ* = 0.33–0.89), 6 dates and 7 budgets (`m7_theory_dmr_family.csv/.tex`).
   - The model_spec formula (1−1/σ)(g_N−g_D) is the α = β special case; in general B = βg_D − αg_N.

4. **The wedge is a sufficient statistic, exactly, for any α, β** (Lemma A4(iii), Lemma A5, Corollary A3).
   - ln w = (1/σ* − 1)·ln(M/M*(C)), where M*(C) is the training-optimal ratio at the model's *own* compute. So T/D = 3[(M/M*(C))^{1/σ*−1} − 1] generalizes the homothetic closed form (ρ becomes (α+β)/2). Numeric error ≤ 3×10⁻¹⁵ at 160 random points.
   - Farrell allocative loss: C/C_min = ((α+βw)/(α+β))^{1/γ} w^{−1/α}. It is free of A, B, E and ω, and reduces to cosh(ρ/2·ln(M/M*))^{2/ρ} when α = β.
   - Harberger form: ln(C/C_min) = σ*(ln w)²/(4(1−σ*)) + O(|ln w|³).
   - Accuracy of the Harberger form (Besiroglu exponents): exact vs approximation is 1.389 vs 1.400 at w = 2; 2.03 vs 2.08 at w = 0.36; 5.52 vs 6.78 at w = 5.22. The approximation overstates at large wedges.
   - The formulas reproduce the ledger: Llama-3-8B w = 5.22, T/D = 12.65, C/C_min = 5.51; Gopher (w = 0.362) C/C_min = 2.01. C/C_min is steep at small w: at w = 0.36 exactly it is 2.03.

5. **"The better labs optimize, the less their data reveal" is now a theorem, but for M*, not for σ as conjectured** (Proposition A2).
   - Exact decomposition: y_i = γ(c_i − ln 6) − ln K − γΦ_i − ε_i, where Φ_i = ln(C_i/C_min,i) is the run's allocative loss. The curvature parameter enters *only* through Φ_i.
   - Information about the location of the expansion path (ln M*): its score is γ(1−w_i)/(b + a w_i) ≈ −γ ln w_i, and the efficient information is (γ/s)² times the residual dispersion of log wedges.
   - Information about the curvature parameter S = α+β is bounded, to leading order, by (γ/(Ss))²ΣΦ_i², which is **fourth order** in the wedges. *[review]* For σ* = 2/(2+S), multiply by (dS/dσ*)² = 4/σ*⁴; the order is unchanged. The original wording attached the bound to σ* directly.
   - Numerical log-log slopes in a 9-budget × 5-offset factorial design: **4.00 (σ*) and 2.00 (ln M*)**. Exact over leading-order information is 1.000 and 1.000 at the smallest dispersion. The on-path information matrix is singular (singular-value ratio 6×10⁻³¹).
   - *[review]* The ln M* information above is for the path intercept ln G, i.e. ln M* at C = 6 FLOP, about 44 log-units outside the design. For ln M* at the design's central compute (10²⁰ FLOP), the slope is also 2.00 and the leading-order ratio 1.000, but the standard error is about 15 times smaller (3.4 vs 51 at sd(ln w) = 0.50, unit noise). The figure now plots the central-compute version.
   - The task conjecture ("Fisher information for σ ∝ variance of transverse deviations") is therefore true for M*, not for σ.

6. **Transmission and selection signs are regime-dependent, with explicit conditions** (Propositions A5 and A6).
   - The OLS slope bias has the sign of π₁ (the response of compute to productivity). Monte Carlo (n = 400k, γ = 0.178):
     - target rule: 0.078 vs formula 0.079;
     - exogenous budgets: 0.179 vs 0.178;
     - funding (π₁ = 2): 0.248 vs 0.247.
   - **The forward–reverse bracket contains γ iff −(1+V_ε/V_ω)/γ ≤ π₁ ≤ 0.** It fails under funding, where the reverse estimate is 0.403 and the forward 0.248, both above γ.
   - Outcome-based release (y ≥ ȳ):
     - with exogenous compute, the slope is ≤ γ for any distribution and in [0, γ] under log-concavity (normal case: 0.178 → 0.086 → 0.044 as the release rate falls from 100% to 50% to 10%);
     - a bimodal, non-log-concave case gives −0.023;
     - under funding (Gaussian case), the net bias is positive iff the retained outcome-variance share r exceeds r* = γV_{c|y}/(δV_y(1−γδ)). In the calibration r* = 0.499, and both signs occur (b = 0.246 at r = 0.99; b = 0.147 at r = 0.36).
   - The reverse regression is invariant to selection on y.

7. **Revealed inference demand: rival wedges have opposite signs** (Proposition A8).
   - Binding data constraint: w = 1/(θ_D + μ) < 1 + T/(3D). With T = 0: w = 0.847, 0.563, 0.132 at D̄ = 0.8, 0.5, 0.2 × D*; μ = −∂ln C*/∂ln D̄ = 0.18, 0.78, 6.58.
   - Binding memory constraint: w = (1+ν)(1 + T/(3D)); w = 1.18 and 1.78 at N̄ = 0.8 and 0.5 × N*.
   - Factor bias contaminates the econometrician's wedge: ŵ = w·e^{−χ}, where χ = βψ_D − αψ_N.
   - All verified by brute-force constrained optimization (formula errors < 4×10⁻⁸).

8. **Partial identification of T** (Proposition A9).
   - Parametric: at fixed compute, Var(ln ŵ) is exactly quadratic in ln M. Bands widen linearly in log-extrapolation distance with slope sd(α̂+β̂)/2.
   - Chinchilla extraction (Huber refit via `sl`, pairs bootstrap B = 300), at M = 1,875 and C = 7.2×10²³:
     - T/D = **12.83, 95% band [9.11, 18.88]**;
     - sd(ln ŵ) = **0.161**, of which M* alone contributes **0.143** and α+β alone **0.047**.
   - So M*, not curvature, drives the uncertainty. This is formal support for the within-family flagship calibration.
   - *[review]* The design reaches M = 341 only at compute ≤ 1.3×10²² FLOP. At C = 7.2×10²³ every M, including M ≤ 341, is an extrapolation in compute: M*(C) itself is extrapolated 55× beyond the largest run, which is why sd(ln M̂*) = 0.40 there. The "design support" shading in the figure is relabeled accordingly.
   - *[review]* Bootstrap health: no replicate is a gross outlier (0 beyond 6 MADs in S or ln M*; S ranges from 0.654 to 0.783). Re-fitting 30 of the 300 draws from the 432-start FAST_GRID reproduced the warm-start optimum (objective differences ≤ 1.4×10⁻¹⁰, |ΔS| ≤ 5×10⁻⁵).
   - Nonparametric: if F_nn, F_dd ≤ 0 and F_nd ≥ 0 (true throughout the κ-family), then w is monotone. So T₀ ≥ 3D₀[w_b − 1] for any support point with more parameters and fewer tokens than the model.
   - There is no upper bound without parametric structure.
   - **Honest caveat:** on the Chinchilla support this bound is uninformative for Llama-3-8B. The best dominating support point has w_b = 0.76, so the bound is negative and says nothing beyond T ≥ 0. It needs over-trained runs at N ≥ the model's N.
   - *[review]* w_b is now generated by run.py (`m7_theory_pi_summary.csv`: `wb_max_besiroglu` = 0.763, `wb_max_refit` = 0.766, 10 dominating points). It is evaluated with the parametric fit at the support points, standing in for a local nonparametric estimate.

9. **All 53 ledger numbers that are pure applications of the formulas reproduce** within rounding (`m7_theory_ledger_checks.csv`). These cover a, γ, σ*, M* at three budgets for three parameter sets; the wedge table; the homothetic table; Meta's-law M*(C) and T/D; Sahal factors; and 41^γ.
   - Ledger precision note: "41^γ ≈ 1.78 (Hoffmann)" uses the **TeX-precision** γ = 0.1548. With the rounded γ = 0.1535 the value is 1.768.

---

## 2. Methods

**Symbolic** (`code/analysis/m7_theory/symbolic.py`, sympy 1.14).
- Each identity is reduced to zero by `simplify`. Where simplification of symbolic powers is slow, the residual is also evaluated at 10–20 random rational points with 30-digit precision (max error ≤ 10⁻¹⁶⁴).
- Checks include:
  - the Hicks σ from first principles (bordered formula in levels) and its invariance to an arbitrary monotone transformation g(·);
  - the general log-coordinate identity σ = P/(P+Q) with an abstract F(n,d);
  - the FOC/SOC, duality objects and both forms of K;
  - Lemma 1 with productivity, including the fixed-target variant;
  - the Hessian, null vector and decomposition;
  - the wedge FOC, its invariances, the contamination, the homothetic and general closed forms, and the Harberger series;
  - KKT with data and memory constraints;
  - the DMR drifts and S-invariance, CEG, Sahal, Hall;
  - the transmission covariance algebra, the information decomposition, and the proxy.

**Numeric** (`numeric.py`). Closed forms are re-derived *without using them*:
- compute-optimal allocations by bounded scalar minimization at fixed C;
- lifetime-compute optima by brute force over n with D solved from the loss constraint;
- constraint multipliers from central differences of ln C* in ln D̄ or ln N̄;
- σ from numerically traced isoquants;
- the DMR family by brute-force optimization of each member at 6 dates × 7 budgets;
- on-path Jacobian rank by finite differences plus truncated-SVD row-space tests;
- efficient Fisher information as the Schur complement of Z'Z, with Z from central differences, for a 9-budget × 5-offset factorial design with transverse offsets scaled by v ∈ [0.05, 2].

**Monte Carlo** (seeds 3–5; n = 400k–1M; Gaussian unless stated). Transmission regimes, outcome-based selection (normal and bimodal), and Sahal with compute noise around trend.

**Bootstrap illustration** (seed 7).
- Data: `sl.chinchilla_extraction` (n = 240; the 5 highest-loss points dropped as in Besiroglu et al.). Estimator: `sl.fit_chinchilla` (Huber δ = 10⁻³ on log loss, L-BFGS).
- Full-sample start from both the published Besiroglu θ and `FAST_GRID`; both converge to the same optimum. Bootstrap replicates are warm-started from it: pairs bootstrap, B = 300.
- The refit has a slightly *lower* objective than the published parameters (1.01827×10⁻³ vs 1.02284×10⁻³). Refit values: E = 1.8172, A = 477.8, B = 2143.4, α = 0.3473, β = 0.3672; published: 482.0, 2085.4, 0.3478, 0.3658.
- This is a rounding or optimizer difference in the published values, not a bug in `sl.py`.
- Output units are MassiveText tokens, so the numbers are illustrative for other corpora and tokenizers.

**What is assumed vs estimated.**
- Assumed: the technology family, cost-minimization, and Gaussian errors for the information calculations and the closed-form selection results.
- Taken from the literature: all parameter values except the bootstrap refit.
- Estimated: only the bootstrap illustration.

---

## 3. Table and figure inventory

Tables in `output/tables/`:

| File | Content |
|---|---|
| `m7_theory_claims.csv` | Full register: 61 checks (after review) with ID, source, claim tested, verdict, method, pass flag, error metric, correction note |
| `m7_theory_claims.tex` | Paper-facing condensed register (31 statements in 4 groups): "Verification of the formal claims" |
| `m7_theory_dmr_family.csv/.tex` | Observationally equivalent technologies (S, σ*, κ, Hicks bias, off-path loss). *[review]* The last column is now the % of reducible loss above the common frontier at M = 5M*, C = 10²²: 1.4, 4.1, 8.4 and 18.2 for σ* = 0.89, 0.74, 0.57, 0.33. |
| `m7_theory_wedge.tex` | Top: C/C_min exact vs Harberger at 8 wedges. Bottom: T/D with bootstrap bands at M ∈ {20, 100, 341, 1000, 1875, 5000} |
| `m7_theory_harberger.csv` | Numbers behind the top panel of the wedge table |
| `m7_theory_pi_band.csv`, `m7_theory_pi_summary.csv` | T/D point estimate, 95% band and se(ln ŵ) on an M grid at C = 7.2e23; refit parameters and variance decomposition |
| `m7_theory_fisher_information.csv` | Efficient information and asymptotic s.e. for σ* and ln M* vs transverse dispersion. *[review]* Columns `se_lnMstar_at_C6` (the old `se_lnMstar`, which is ln M* at C = 6 FLOP) and `se_lnMstar_center` (ln M* at 10²⁰ FLOP) |
| `m7_theory_transmission_mc.csv`, `m7_theory_selection_mc.csv` | Monte Carlo vs formulas |
| `m7_theory_constraints.csv` | Data/memory-constraint wedges and multipliers (brute force) |
| `m7_theory_ledger_checks.csv` | 53 ledger numbers recomputed |

Figures in `output/figures/`, each as .pdf and .png:

| File | Content |
|---|---|
| `m7_theory_geometry` | (a) Isoquants, isocosts, expansion path (Besiroglu) and an 8B/15T run's equal-compute deviation (w = 5.2). (b) Three observationally equivalent technologies (σ* = 0.60, 0.74, 0.85): identical on the path, divergent off it |
| `m7_theory_information` | s.e. of σ̂* (slope −2) and of ln M̂* at the design's central compute (slope −1) vs sd(ln w), unit noise. The Chinchilla extraction's sd(ln w) = 0.48 (240-run estimation sample) is marked *[review: was 0.53 from all 245 raw rows, and the ln M* curve was at C = 6 FLOP]* |
| `m7_theory_wedge` | (a) T/D vs M/M*(C) for σ* ∈ {0.70, 0.737, 0.78}. (b) C/C_min exact vs Harberger. (c) Bootstrap band for T/D vs M at C = 7.2e23, with the design's M-range shaded (reached only at C ≤ 1.3e22) |
| `m7_theory_bias` | (a) plim forward OLS/γ vs γπ₁ for three selection severities. (b) Forward vs reverse, with the region where the bracket contains γ shaded |

Processed data in `data/processed/m7_theory/`: `wedge_bruteforce_cases.csv` and `chinchilla_transverse_sd.txt`.

Paper files:
- `paper/sections/appendix_proofs.tex`: 9 propositions, 5 lemmas, 5 corollaries, 3 definitions and 7 remarks, with complete proofs.
- `paper/notes/theory_main_text.md`: main-text statements, intuition, numbering map, corrections list.

---

## 4. Claims for the paper

Each claim gives the evidence, then the caveat.

1. **Chinchilla's approaches are cost-function, conditional-factor-demand and primal estimators, linked by a = β/(α+β) and γ = αβ/(α+β).**
   - Evidence: claims P1.path and P1.path.num (max relative error 1.9×10⁻⁷).
   - Caveat: the interpretation of the approaches is ours; the algebra is Hoffmann et al.'s eq. 4.
2. **σ = (αu+βv)/(αu(1+β)+βv(1+α)); σ(w) = (1+w)/(1+α+w(1+β)); σ* = 2/(2+α+β); σ is ordinal.**
   - Evidence: MS.SIG, MS.SIGW, MS.SIGORD, MS.SIG.num (finite-difference isoquants, error 1.4×10⁻¹⁰).
   - Caveat: the σ formula itself is in `hao2026theory`. Claim σ(w), σ*, ordinality and the Kaplan contrast (0.50–0.575).
3. **Interior compute optima under C = 6ND require σ < 1: compute-optimal training reveals gross complementarity.**
   - Evidence: SOC (general sympy identity), SOC.num.
   - Caveat: local statement at the optimum; σ could exceed 1 far off the path in a general technology (not in the κ-family).
4. **On-path data identify the path (a, G) and, from outcomes, (E, γ, K). α and β only by functional form (κ = 1). A/B, hence the technology's own M*, only by assuming optimality. σ* not at all without κ = 1 (any σ* ∈ (0,1) fits).**
   - Evidence: P2.rank (Jacobian rank 3; the row-space residual of ∇σ* is 0.022 for Besiroglu and 0.165 for Hoffmann, growing with |α−β|, and 7×10⁻¹¹ for α = β), P5.DMR.num, `m7_theory_dmr_family.tex`.
   - Caveat: "first-order vs second-order identification" is a local statement; I did not derive the non-standard convergence rates.
5. **The better labs optimize, the less their data reveal.** Information about ln M* ∝ dispersion of log wedges; information about σ* ≤ const × ΣΦ_i² (fourth order).
   - Evidence: INFO.dec (exact decomposition), INFO.num (slopes 4.00, 2.00; exact/leading ratios 1.000), INFO.onpath.
   - Caveat: Gaussian noise on y with E known. The variant with E unknown and noise on ln L was not computed.
6. **Time series of compute-optimal models identify augmentation rates g_N, g_D and the sign of the Hicks bias. They identify neither σ* nor the bias magnitude. Neutrality ⇔ no drift in the compute-optimal D/N at given compute.**
   - Evidence: P5.rates, P5.ident, P5.DMR.num; appendix eq. (A-drift).
   - Caveat: needs a cross-section of budgets at each date to identify γ and a. Pure frontier time series with a single budget per date do not suffice.
   - The drift must be measured on training-optimal allocations (IsoFLOP minima or published compute-optimal laws), not on deployed models, whose D/N also moves with inference demand.
7. **Constant compute-equivalent gains ⇔ equal E and equal γ.** Factor augmentation → constant CEG = e^{ψ_N+ψ_D}.
   - Evidence: P4S.CEG and P4S.CEG.num. Counterexample: (α,β) = (0.3,0.4) vs (0.4,0.3) gives sd(ln f) = 3×10⁻¹⁵.
   - Caveat: frontier-to-frontier comparison. At a fixed allocation rule, equal E and equal exponent sets are necessary but **not sufficient** *[review]*. The gain is constant only if the new law is the old one with N rescaled. Factor augmentation with ψ_N ≠ ψ_D, or Hicks-neutral change with α ≠ β, gives a non-constant gain on a common ray (check P4S.CEG.rule: sd(ln f) = 0.028 and 0.025, against 5×10⁻¹⁵ when ψ_N = ψ_D). The appendix's original "iff" was wrong and is corrected.
8. **Revealed inference demand: T/D = 3[(M/M*(C))^{1/σ*−1} − 1], exact for any α, β. C/C_min is a function of w alone.**
   - Evidence: W1, W1.num, W2, W2.num, HOM, ledger checks.
   - Caveats: requires lifetime-cost minimization, no binding constraints, common technology, and a correct M*(C). The Harberger approximation overstates by 23% at w = 5.2.
9. **Rival wedges have known signs:** data scarcity lowers ŵ (T̂ < T); memory or latency constraints raise it; factor bias multiplies it by e^{−χ}.
   - Evidence: D.constr, D.constr.num, P4.contam(.num).
   - Caveat: sign results only; magnitudes need the multipliers.
10. **Transmission bias has the sign of the compute response to productivity. The forward–reverse bracket holds iff −(1+V_ε/V_ω)/γ ≤ π₁ ≤ 0. Hicks-neutral transmission biases γ but not the allocation exponent a.**
    - Evidence: P2.plim, P2.bounds, P2.MC, P2.a.
    - Caveat: on-path, Hicks-neutral, linear compute rule. *[review]* In the dynamic-panel remedy, c_t is a valid instrument only if budgets do not respond to the previous run's evaluation noise. Otherwise use (1, c_{t−1}, c_{t−2}, y_{t−2}); the appendix now says so.
11. **Outcome-based release attenuates the compute slope (always ≤ γ; in [0, γ] under log-concavity). Under funding, the net sign flips at r* (closed form). Reverse regression is immune to selection on y (Gaussian).**
    - Evidence: P3.sel, P3.net, P3.cex.
    - Caveat: the closed forms are Gaussian; the inequalities are distribution-free.
12. **D/N is a valid (invertible) proxy for factor-biased productivity χ, but not for Hicks-neutral productivity, and even for χ the proxy step is subject to ACF functional dependence.**
    - Evidence: PROXY, PROXY.num (coefficient on ω is −10⁻⁹; on χ it is −2.8027, matching theory). *[review]* The zero ω coefficient holds by construction, because a Hicks-neutral shift does not move the argmin; the symbolic check is the proof. ω is now passed into the brute-force objective, but the numerical check adds little.
    - Caveat: requires common wedges; with heterogeneous T the mix reveals only ln w − χ.
13. **Farrell decomposition: log excess compute = allocative Φ(w) + technical (Ω_F − Ω_i)/γ + noise. Kaplan→Chinchilla is allocative (Gopher: 2.0× under Besiroglu).**
    - Evidence: appendix Prop. A4; W2; ledger (Gopher C/C_min 2.01).
    - Caveat: uses the lab's true wedge; with common-technology wedges, contamination applies.
14. **Sahal: naive frontier-release exponents are γ[1 + (g_A/g)R²_{c,t}] (= γ/(1−s_A) with deterministic trends; ×1.05–1.67). Hall: the equal-weight residual is biased by (ε_N−ε_D)(Δn−Δd)/2.**
    - Evidence: P7, P7.MC, P8, P8.num (second-order error).
    - Caveat: first-order (Divisia) for Hall; deterministic-trend version for the headline factor.

---

## 5. Robustness, failures, and what is fragile

- **Initial failures, all fixed and none substantive:**
  - a sympy series artifact (fixed by expanding the log first);
  - over-tight tolerances on the argmin of a flat objective (n* matches to 4×10⁻⁷, ln R* to 10⁻¹⁴);
  - a least-squares row-space test that needed rank truncation;
  - a first attempt at the bimodal selection example whose truncation points did not fall where m′ > 1.
- **Partial-identification slope:** my first test (asymptotic slope of se(ln ŵ) vs ln M) failed, 0.005 vs 0.010. The reason is that the asymptote is not reached for M ≤ 10⁴: se is a hyperbola dominated by M* uncertainty. The test was replaced by the exact quadratic identity. The substantive conclusion (M* dominates) came from this failure.
- **The nonparametric lower bound (Prop. A9(ii)) is uninformative on Chinchilla support for Llama-scale models** (w_b ≤ 0.76 among dominating support points). It needs over-trained designs at large N.
- **Harberger approximation:** accurate for |ln w| ≤ 0.7 (within 1%), poor beyond (23% at w = 5.2). Use the exact formula.
- *[review]* **Checks that hold by construction.** Some numeric checks restate their symbolic counterparts rather than testing them independently:
  - PROXY.num and L1.num (ω never moves the argmin);
  - PI.band (the quadratic form of Var(ln ŵ) is an identity);
  - P2.a (n is an exact function of c in the simulation).

  They are correct but add little evidence; the symbolic proofs carry the weight.
- *[review]* **Numbers not previously regenerated.** w_b = 0.76 is now produced by run.py (`m7_theory_pi_summary.csv`). Before the review it was quoted in the memo and appendix without any code producing it.
- **Fisher-information results:** local (small deviations), Gaussian, E known. At large dispersion the exact/leading ratio drifts (0.91 for σ* and 0.90 for M* at sd(ln w) ≈ 0.5).
- **Selection closed forms rely on joint normality.** The distribution-free statements are only inequalities.
- **Nothing in this module tests the Chinchilla functional form against data.** All results are conditional on the (generalized) family. The rank-one Hessian restriction holds for the *whole* κ-family, so a translog test of b_NN·b_DD = b_ND² tests additive power separability, not κ = 1. This is a note for the estimation modules.

---

## 6. Open issues (please route to the lead author)

1. **`paper/main.tex` does not compile as is.** AEA.cls defines a `proof` environment, and `\usepackage{amsthm}` then stops with "Command \proof already defined".
   - Tested fix: put `\let\proof\relax\let\endproof\relax` on the line before `\usepackage{amsthm}`.
   - With it, a copy of the preamble plus this appendix and the three tables compiles with `tools/tectonic`.
   - I did not edit `main.tex`.
2. **`lit/references.bib` breaks BibTeX under `aea.bst`.** Internal `note` fields contain raw underscores and URLs (for example `besiroglu2024chinchilla`: "svg_extracted_data.csv"), giving "Missing $ inserted" in the .bbl.
   - Tested fix: strip `note` fields in the compile copy of the bib (or escape them).
3. **Duplicate key:** `rotnitzky2000likelihood` also exists in `extra_m6_montecarlo.bib`. I cite it but did not duplicate it. My new keys are `bagnoli2005logconcave`, `efron1965increasing` and `klepper1984consistent`, in `lit/bib/extra_m7_theory.bib`, verified via Crossref.
   - Rotnitzky and Klepper–Leamer end pages come from the published articles; Crossref returned only first pages.
4. **Wording to propagate:**
   - SYNTHESIS §1 items 3 and 6(iii) and hypothesis H5 need the corrected DMR and CEG statements.
   - The main text must not say "on-path data cannot sign the bias".
5. **Not done:**
   - convergence rates under singular information (on-path α, β with κ = 1);
   - the information calculation with E unknown;
   - a formal treatment of the Kaplan-era belief wedge (only noted as a source of w < 1);
   - non-homothetic Farseer-type technologies. The wedge sufficient-statistic result is proven for the κ-family only. For Farseer, compute w directly from the fitted elasticities.
6. **Numbering:** the appendix uses A1, A2, …. The map to proposed main-text numbers is in `paper/notes/theory_main_text.md` §0. When the main-text sections are written, restatements in the appendix can cite main-text labels.
7. *[review]* **Suggested extension, not done.** Under κ = 1, on-path *outcome* data alone (without using the path slope a) bound σ* from above: the set with fixed (E, K, γ) has S ≥ 4γ, with equality at α = β, so σ* ≤ 1/(1+2γ), which is 0.737 for Besiroglu. This explains why ∇σ* lies in the Jacobian row space exactly when α = β. A one-line remark could go in Prop. A1; it is not in the appendix and is not verified by the suite.
