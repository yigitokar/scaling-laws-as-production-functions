# Proposed main-text theory: Framework and Identification sections

Module m7_theory, 2026-09-23. This note gives the main-text statements, with intuition and no proofs. The proofs are in `paper/sections/appendix_proofs.tex`, where results are numbered A1, A2, ...

Every statement here was verified symbolically (sympy) and numerically by `code/analysis/m7_theory/run.py`: 61 checks after the independent review, all pass (see `output/memos/m7_theory_review.md`). Notation is that of `paper/notes/model_spec.md`.

The main change from `model_spec.md` is that four statements there, and three in SYNTHESIS P1–P8, need correcting (listed at the end). Several new results sharpen the story. The two biggest:

- Compute-optimality itself reveals σ < 1.
- The sign of biased technical change *is* identified from compute-optimal models; only its magnitude and σ are not.

---

## 0. Recommended numbering

| Main text | Content | Appendix | model_spec draft | Status |
|---|---|---|---|---|
| Definition 1 | Technology (T) and the generalized κ-family | Def. A1, A2 | §1 | verified; wording fixes |
| Lemma 1 | Elasticity of substitution: σ formula, σ(w), σ*, ordinality | Lemma A1 | §1 derived objects | verified; σ(w) new |
| Proposition 1 | Compute-optimal training reveals gross complementarity (strict SOC ⇔ 0 < σ < 1) | Lemma A2, Cor. A1 | — | **new** |
| Lemma 2 | Duality and conditional factor demands, with productivity | Lemma A3 | Lemma 1 | verified; precision added |
| Lemma 3 | Geometry: rank-one curvature; transverse coordinate = −ln w; allocative loss is a function of w | Lemmas A4, A5 | Lemma 2 | verified; decomposition, C/C_min(w) new |
| Proposition 2 | Functional dependence: what on-path data identify | Prop. A1 | Prop. 1 | verified; sharpened |
| Proposition 3 | Information: the better labs optimize, the less their data reveal | Prop. A2 | (SYNTHESIS corollary) | **new**; task conjecture corrected |
| Proposition 4 | Technical change: time series identify augmentation rates and the *sign* of the bias, not σ* or its magnitude | Prop. A3, Cor. A2, Prop. A4 | Prop. 5 + "Decomposition" | **corrected** |
| Proposition 5 | Observational data: transmission bias by regime, selection, bounds | Props. A5, A6 | Props. 2, 3 | verified; bounds and selection corrected |
| Corollary 1 | Allocation-based proxies (Levinsohn–Petrin invertibility for factor bias; failure for Hicks-neutral) | Prop. A7 | — | **new** |
| Proposition 6 | Revealed inference demand: w = 1 + T/(3D) and its sufficient statistics | Prop. A8, Cor. A3 | Prop. 4 | verified; generalized closed form new |
| Proposition 7 | Partial identification of T outside the design support | Prop. A9 | (SYNTHESIS P6) | **new** |
| Corollaries 2–3 | Frontier-release (Sahal) bias; Hall-type growth-accounting bias | Cors. A4, A5 | (SYNTHESIS P7, P8) | verified with conditions |

If space is tight:
- Proposition 1 can be folded into Lemma 1 as part (iv).
- Corollary 1 can move to the appendix, with one sentence in the text.
- Corollaries 2–3 belong in the algorithmic-progress application rather than in Section III.

---

## Section II. Scaling laws as production functions

**Definition 1 (Technology).** A run with N parameters and D tokens attains

  L = E + e^{−ω}[A(e^{ψ_N}N)^{−α} + B(e^{ψ_D}D)^{−β}]e^{ε}.

Compute is C = 6ND. With lower case for logs, y ≡ −ln(L − E) = ω + F(n + ψ_N, d + ψ_D) − ε holds exactly, where F(n, d) = −ln(Ae^{−αn} + Be^{−βd}).
- Output elasticities: ε_N = αu/(u+v) and ε_D = βv/(u+v).
- The wedge is w ≡ ε_N/ε_D.
- The generalized family replaces the bracket by [·]^κ. Chinchilla is κ = 1. Kaplan's joint law is κ = 0.103 with inner exponents (0.738, 1).

*Writing note.* Say "u + v", not "L − E", in the elasticity formulas, because L − E = e^{−ω+ε}(u + v).

**Lemma 1 (Elasticity of substitution).**
- σ = (αu + βv)/(αu(1+β) + βv(1+α)).
- σ depends on the input mix only through the wedge: σ(w) = (1 + w)/(1 + α + w(1 + β)).
- On the compute-optimal path (w = 1), σ* = 2/(2 + α + β).
- σ is ordinal: it is unchanged by any monotone relabeling of output (loss, bits per byte, a benchmark link), and it does not depend on E, ω or κ.

*Intuition.* Iso-loss curves are isoquants. Their curvature is a property of the technology, not of how output is measured. Only the ratio of the two marginal terms matters, so every model with the same over-training wedge has the same local σ. For example, Llama-3-8B under the Besiroglu parameters has w = 5.22 and σ = 0.734, against σ* = 0.737.

*Credit.* The σ formula is in Hao and Merrill (2026, `hao2026theory`). New here: σ(w), σ*, ordinality, and the Kaplan contrast (σ ∈ [0.50, 0.575], σ* = 0.535).

**Proposition 1 (Compute-optimal training reveals gross complementarity).** Under the multiplicative cost C = 6ND, a point where ε_N = ε_D is a strict local cost minimum if and only if 0 < σ < 1 there (σ ≤ 0 would mean non-convex isoquants).
- In log coordinates, σ = P/(P + Q) for any technology, with Q > 0 exactly when the log-isoquant is convex.
- With σ > 1 the same first-order condition picks the *worst* allocation, and optimal training is a corner.
- Hence observing interior compute-optimal allocations reveals σ < 1.

*Intuition.* With linear costs the isocost is a straight line in levels, and any convex isoquant (σ > 0) gives an interior tangency. With C = 6ND the isocost is a straight line in logs. The implicit price of a parameter is 6D, which rises with the other input, so the isoquant must be convex in logs, and that requires complementarity. The existence of Chinchilla-style frontiers is therefore itself evidence that parameters and data are gross complements. It also means an observational-equivalence argument can never push σ* above 1 (this matters for Proposition 4).

**Lemma 2 (Duality and conditional factor demands).** The cost-minimizing allocation equates output elasticities (αu = βv): no price data are needed.
- Factor demands: N* = G(C/6)^a and D* = G^{−1}(C/6)^b.
- Inverse cost function: L*(C) = E + K(C/6)^{−γ}, with the two equivalent forms of K; the cost function is C*(R) = 6(K/R)^{1/γ}.
- Chinchilla's three approaches estimate, respectively, the cost function (Approach 1), the conditional factor demands (Approach 2) and the primal technology (Approach 3), linked by a = β/(α+β) and γ = αβ/(α+β).

With productivity:
- n* = ln G + a(c − ln 6) + χ/(α+β), with χ ≡ βψ_D − αψ_N.
- y* = ω + γ(c − ln 6 + ψ_N + ψ_D) − ln K.

Two consequences:
- Hicks-neutral ω never moves the mix at given compute. Only the "bias index" χ does.
- Better data (ψ_D ↑) mean more parameters, fewer tokens, and fewer tokens per parameter at given compute (∂ ln M*/∂ψ_D = −2a). This remains true with inference demand.

*Precision to state in the text.* These signs are at given compute. At a given loss target, better data leave N* unchanged and lower D* one for one. Hicks-neutral ω then does move the chosen mix whenever α ≠ β, because the technology is non-homothetic.

**Lemma 3 (Geometry).**
- The Hessian of ln R in (ln N, ln D) is rank one. Its null direction is the expansion path. ln R = −(αn + βd)/2 + g(αn − βd), with g convex.
- A run's transverse coordinate is minus its log wedge. At the run's own compute, ln w = (1/σ* − 1)·ln(M/M*(C)).
- The Farrell allocative loss depends on w alone: C/C_min = ((α + βw)/(α+β))^{1/γ} w^{−1/α}.
- This loss is approximately a Harberger triangle, ln(C/C_min) ≈ σ*(ln w)²/(4(1 − σ*)).

*Intuition.* Loss falls log-linearly along every ray parallel to the expansion path. All curvature, and hence everything about σ, is transverse to it. The distance of a run from the path is measured by its wedge, and the compute it wastes is second order in that distance, with a curvature governed by σ*. For example, Gopher's w = 0.36 means 2.0× wasted compute (Besiroglu parameters).

---

## Section III. What scaling-law data identify

**Proposition 2 (Functional dependence).** Suppose every run minimizes training compute (T = 0) under a common technology with Hicks-neutral productivity. Then:
1. (n, d) are deterministic affine functions of c. The choice data reveal the path slope a and intercept G (the observed M*(C)), and nothing else about the technology.
2. E[y | c] = γ(c − ln 6) − ln K + E[ω | c]. So γ is identified only if E[ω | c] is known up to a constant (a single-lab sweep, not a cross-lab sample).
3. With κ = 1 and ω ⊥ c, outcomes identify E, γ, K, α = γ/a and β = γ/b, but only by functional form:
   - the on-path Jacobian of the five Chinchilla parameters has rank 3;
   - α and β are identified from outcomes only at second order, unless the observed path is imposed to be optimal;
   - σ* is first-order identified only when α = β;
   - the level A/B, and with it the technology's own cost-minimizing ratio, is identified only by *assuming* the path is optimal.
4. With κ free, the explicit family a₁ = (1−a)S, b₁ = aS, κ = γ/(a(1−a)S), with (A, B) matched to (G, K), is observationally equivalent for every S > 0. Every σ* ∈ (0, 1) fits. σ* ≥ 1 does not, by Proposition 1.

*Intuition.* Optimizing labs sit on a ray, and a ray has no curvature. This is the Ackerberg–Caves–Frazer problem, created by behavior rather than by design. It also explains the published record: exponents are stable (s.e. 0.02), while A and B (s.e. 125 and 1,293), and with them M*, are fragile.

*Table.* `output/tables/m7_theory_dmr_family.tex` shows four members with σ* from 0.33 to 0.89. They share an identical frontier to 1e−14 (and the same path up to optimizer tolerance), but at five times the optimal tokens per parameter their reducible loss lies between 1.4% (σ* = 0.89) and 18.2% (σ* = 0.33) above the common frontier (4.1% for Chinchilla, σ* = 0.74).

**Proposition 3 (The better labs optimize, the less their data reveal).** In a sweep with noise variance s², write each run's output as

  y_i = γ(c_i − ln 6) − ln K − γΦ_i − ε_i,

where Φ_i = ln(C_i/C_min,i) is the run's allocative loss. This decomposition is exact. The curvature parameter enters only through Φ_i.
- The score for the path level (and hence for ln M*) is γ(1 − w_i)/(b + a w_i) ≈ −γ ln w_i. Its efficient information is (γ/s)² times the residual dispersion of the runs' log wedges.
- The score for the curvature parameter S = α + β is ≈ −(γ/S)Φ_i. To leading order its information is at most (γ/(S s))² Σ Φ_i², which is fourth order in the wedges. For σ* = 2/(2+S), multiply by (dS/dσ*)² = 4/σ*⁴; the order is unchanged.
- If transverse deviations are scaled by v, the information about M* scales as v² and the information about σ* as v⁴.

*Intuition.* Data reveal the location of the expansion path to the extent that runs are off it. They reveal its curvature only to the extent that runs waste compute. The waste is a Harberger triangle, second order in the deviation. Labs that train compute-optimally, and share objectives, generate no information about either.

*Numbers.* In a 9-budget × 5-offset factorial design, the numerical log-log slopes of the information are 4.00 (σ*) and 2.00 (ln M*, both for the path intercept and for ln M* at the design's central compute). The Chinchilla extraction's transverse spread is sd(ln w) = 0.48 across the 240-run estimation sample (0.53 if the 5 dropped highest-loss runs are kept; Figure `m7_theory_information`). The figure's standard errors assume unit noise and are meaningful only as slopes.

*Correction to the conjecture in the task list.* "Fisher information for σ ∝ variance of transverse deviations" is true for the *location* of the path (M*), not for σ. For σ the relevant moment is the dispersion of squared deviations.

**Proposition 4 (Technical change over time).**
1. *Nonparametric sign result.* For any smooth technology with interior compute optima, the compute-optimal mix at fixed compute drifts as

     ∂ ln M*(C,t)/∂t = −[σ*/(1 − σ*)]·B_t,

   where B_t is the Hicks bias (the change in the MRTS at fixed inputs) and σ* ∈ (0,1) by Proposition 1. The sign of the bias is therefore identified by the sign of the drift in the compute-optimal tokens-per-parameter ratio. Neutrality is testable (no drift).
2. In the generalized family with factor-augmenting progress (g_N, g_D):
   - frontier and path drifts identify g_N + g_D and ag_D − bg_N, hence g_N and g_D separately;
   - σ* and the magnitude of the bias, B = S(ag_D − bg_N), are not identified (explicit equivalent family, every date).
   - The formula (1 − 1/σ)(g_N − g_D) holds only for α = β; in general B = βg_D − αg_N.
3. Neutrality ⇔ αg_N = βg_D: a proportional shift of both A and B, not a shift in E.
4. The compute-equivalent gain between two technologies on their own frontiers is constant in C iff they share E *and* γ = αβ/(α+β); the individual exponents need not be equal. Factor augmentation gives a constant gain e^{ψ_N+ψ_D}. At a common allocation rule D = mN, equal E and exponents are necessary but *not* sufficient: the gain is constant only if the new law is the old one with N rescaled, which fails for factor augmentation with ψ_N ≠ ψ_D and for Hicks-neutral change with α ≠ β.
5. Farrell decomposition: ln(C_i/C_min,F(L_i)) = Φ(w_i) [allocative] + (Ω_F − Ω_i)/γ [technical] + noise. The Kaplan→Chinchilla reallocation is purely allocative (Gopher: 2.0× under Besiroglu).

*Relation to DMR.* Diamond–McFadden–Rodriguez non-identification remains for σ* and for magnitudes. But with fixed FLOP accounting and the multiplicative cost, the *signs* are identified. State this carefully: "on-path time series cannot sign the bias" is wrong in this setting. The result holds relative log-cost weights fixed (FLOP-only costs). If data-acquisition costs, a tightening data constraint, or inference demand change the effective price of tokens over time, the drift of M* mixes price responses with bias and the DMR problem returns. Measure the drift on IsoFLOP minima, not on deployed models.

**Proposition 5 (Observational data: transmission and selection).** Suppose runs are on-path with Hicks-neutral ω and compute follows c = π₀ + π₁ω + η.

(a) Transmission:
- The OLS slope of y on c tends to γ + π₁V_ω/(π₁²V_ω + V_η). The bias has the sign of π₁.
  - A common loss target (π₁ = −1/γ) attenuates the slope toward 0. If labs hit the target exactly, cross-lab data carry no information.
  - Exogenous budgets give γ.
  - Funding that responds to productivity overstates γ.
- TFP dispersion is understated by the factor V_η/Var(c).
- The allocation exponent a is not biased by Hicks-neutral ω. With factor bias it is: â → a + Cov(χ,c)/((α+β)Var c).
- Forward and reverse regressions bracket γ iff −(1 + V_ε/V_ω)/γ ≤ π₁ ≤ 0. They bracket it under target or exogenous behavior, not under funding.

(b) Selection on the outcome (release iff y ≥ ȳ):
- With exogenous compute, the released-sample slope is at most γ for any distribution, and lies in [0, γ] under log-concavity.
- In the Gaussian case:
  - the reverse regression is unaffected by the selection;
  - the forward slope falls with the retained outcome variance r;
  - under a funding rule, the net bias is positive iff r > r*, which has a closed form.

  Mild selection leaves transmission dominant; severe selection reverses the sign. Both signs occur in simulation (r* = 0.50 in the illustrative calibration).
- The forward–reverse bracket survives outcome-based selection.

*Correction.* E[ω | c, released] is decreasing in c only for exogenous compute. Under funding it can increase, and does in simulation. What is always decreasing is the selection component.

*Panel remedies.* With a Markov ω and predetermined compute, use the lagged-outcome (dynamic-panel) form: instruments (1, c_t, c_{t−1}, y_{t−2}). c_t is valid only if budgets do not respond to the previous run's evaluation noise; otherwise use (1, c_{t−1}, c_{t−2}, y_{t−2}). The Olley–Pakes/Levinsohn–Petrin first-stage inversion is unavailable because of Corollary 1.

**Corollary 1 (Allocation-based proxies).** Given compute, ln(D/N) = m₀(c) + (2/(α+β))(ln w − χ).
- With a common known wedge, the mix inverts the factor bias χ exactly (Levinsohn–Petrin invertibility with a known linear inverse). It is flat in Hicks-neutral ω, so no allocation-based proxy exists for neutral TFP.
- Even for χ, the map (n, d) ↔ (c, ln M) is one-to-one, so a first-stage control function absorbs the whole technology (ACF). Identification must come from timing.
- With heterogeneous inference demand, the mix reveals only ln w − χ: inference demand and factor bias are confounded.

---

## Revealed inference demand (Section on H2; theory part)

**Proposition 6 (Revealed inference demand).** A lab that minimizes lifetime compute 6ND + 2NT chooses w = 1 + T/(3D), the ratio of lifetime to training compute.
- Invariance: w is invariant to ω, ε, E and κ.
- Factor bias contaminates the econometrician's wedge: ŵ = w·e^{−χ}.
- A binding data constraint gives w = 1/(θ_D + μ) < 1 + T/(3D), and w < 1 if T = 0, where μ = −∂ln C*/∂ln D̄.
- A binding memory or latency constraint gives w = (1 + ν)(1 + T/(3D)).
- So T̂ = 3D(ŵ − 1) understates T under data scarcity or data-augmenting productivity, and overstates it under memory constraints or parameter-augmenting productivity.
- w < 1 cannot come from inference demand.

*Sufficient statistics* (Corollary A3; generalizes the homothetic closed form):
- T/D = 3[(M/M*(C))^{1/σ*−1} − 1] holds exactly for any α, β. M*(C) is the training-optimal ratio at the model's *own* compute.
- C/C_min = ((α + βw)/(α+β))^{1/γ} w^{−1/α}, which is cosh(ρ/2·ln(M/M*))^{2/ρ} when α = β.
- This justifies evaluating a family's own IsoFLOP law at each model's compute, with ρ = (α+β)/2.
- It also shows that M*(C) and σ* are the only technology objects that matter.

**Proposition 7 (Partial identification of T).**
1. *Parametric.* Projecting a confidence set for (ln(A/B), α, β) gives a valid confidence set for T. At fixed compute, Var(ln ŵ) is exactly quadratic in ln M. Bands widen linearly in the log-extrapolation distance, with slope sd(α̂+β̂)/2.
   - Near the design support, uncertainty about M* dominates. At a Llama-3-8B-sized point (M = 1,875, C = 7.2e23), sd(ln ŵ) = 0.161: 0.143 from M* alone and 0.047 from α+β alone (Chinchilla extraction, B = 300). The implied T/D is 12.8 with a 95% band of [9.1, 18.9].
   - The Chinchilla design reaches M = 341 only at compute ≤ 1.3e22 FLOP, so at 7.2e23 FLOP every ratio is an extrapolation in compute as well as in M.
2. *Nonparametric lower bound.* Suppose F is concave in each log input and supermodular (F_nd ≥ 0), which holds throughout the generalized family. Then w falls with n and rises with d. Hence T₀ ≥ 3D₀[w(n_b, d_b) − 1] for any support point with more parameters and fewer tokens than the model.
   - No finite upper bound follows from shape restrictions alone (a saturating data term sends w → ∞).
   - Supermodularity is needed: a concave translog with F_nd < 0 violates the bound.

*Intuition and link to evidence.* Extrapolated wedges are conservative in the direction Sardana et al. document: fitted laws overstate the value of extra tokens at extreme ratios. The within-family flagship calibration pins M*, which removes the dominant source of uncertainty.

---

## Algorithmic-progress corollaries (application section)

**Corollary 2 (Frontier-release regressions, Sahal).**
- With compute and effective compute trending, the naive exponent is γ[1 + (g_A/g)R²_{c,t}].
- With deterministic trends this equals γ/(1 − s_A), an inflation factor of 1.05–1.67 for s_A ∈ [0.05, 0.40].
- A time trend fixes it only if compute deviates from trend.

**Corollary 3 (Hall-type accounting bias).**
- With equal (compute-share) weights, the residual is biased by (ε_N − ε_D)(Δn − Δd)/2 = ε̄·(w−1)/(w+1)·(Δn − Δd).
- A shift toward over-training understates progress.
- This holds to first order and assumes the equal weights sum to the true scale elasticity.

---

## Corrections relative to earlier drafts (for the writers)

1. **SYNTHESIS P4 / Summary item 6(iii)** ("constant CEG ⇔ equal exponents and E"). **False** for frontier comparisons. The correct condition is equal E and equal γ = αβ/(α+β). At a fixed common allocation rule, equal exponent sets are necessary but not sufficient: the change must be a common rescaling of N. So P4's "factor augmentation always yields a constant CEG" holds on frontiers, not on a common ray unless ψ_N = ψ_D.
2. **model_spec Prop. 5 and SYNTHESIS P3 / Summary item 3** ("time series do not identify … the sign of the Hicks bias"). **Corrected.** The sign is identified (Proposition 4(1)); g_N and g_D are identified separately in the family; only σ* and the magnitude of the bias are not.
3. **SYNTHESIS P5** ("forward and reverse regressions bound γ under classical noise"). **Imprecise.** They bound γ iff −(1+V_ε/V_ω)/γ ≤ π₁ ≤ 0, and fail under funding.
4. **model_spec Prop. 3** ("E[ω | c, released] is decreasing in c"). True only for exogenous compute. The net-sign condition is now explicit (r > r*).
5. **model_spec Lemma 1** signs hold at given compute, not at a given loss target (see the precision note under Lemma 2).
6. **model_spec §1**: y = ω + F − ε is exact, not an approximation. Write ε_N = αu/(u+v).
7. **model_spec Prop. 2**: the bias formula holds for any sign of π₁. The ACF moments require the lagged-outcome form, because no proxy exists for neutral TFP.
8. **Task conjecture 2(a)** (Fisher information for σ ∝ variance of transverse deviations): true for M*, not for σ (see Proposition 3).

## Citations used (all in `lit/references.bib` unless noted)

- `hoffmann2022training`, `besiroglu2024chinchilla`, `kaplan2020scaling`, `sardana2024beyond`, `hao2026theory`
- `ackerberg2015identification`, `olley1996dynamics`, `levinsohn2003estimating`, `mundlak1961empirical`, `marschak1944random`
- `diamond1978measurement`, `ho2024algorithmic`, `nagy2013statistical`, `hall1988relation`, `farrell1957measurement`
- `deloecker2012markups`, `raval2023testing`, `demirer2020production`, `gandhi2020identification`
- New in `lit/bib/extra_m7_theory.bib`, verified via Crossref: `bagnoli2005logconcave`, `efron1965increasing`, `klepper1984consistent`.
- `rotnitzky2000likelihood` is in `extra_m6_montecarlo.bib`.
