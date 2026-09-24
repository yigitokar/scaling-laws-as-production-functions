# Strand notes: CES, elasticity of substitution, biased technical change, duality / cost functions, learning curves, growth accounting, frontiers

Project: "Scaling Laws as Production Functions" (AER-style). Strand owner notes, written for the paper-writing stage.
BibTeX: `/Users/yigitokar/scaling-laws-pf/lit/bib/io_ces_duality.bib` (62 entries, all verified this session; uncertain fields carry `note = {UNVERIFIED: ...}`).
Derivation scripts (sympy/numpy, re-runnable) are reproduced in Appendix A at the end of this file; every algebraic claim marked [SYMPY] was checked symbolically, [NUM] numerically.

Notation. $N$ = parameters, $D$ = training tokens, $C \approx 6ND$ training FLOPs, $L$ = per-token cross-entropy loss (nats), $E$ = irreducible loss, $R \equiv L-E$ = reducible loss, lower-case $n=\ln N$, $d=\ln D$, $c=\ln C$. Chinchilla form (Hoffmann et al. 2022, Eq. 2): $L = E + A N^{-\alpha} + B D^{-\beta}$. Write $u \equiv A N^{-\alpha}$, $v \equiv B D^{-\beta}$ (so $R=u+v$). Output elasticities of reducible loss: $\varepsilon_N \equiv -\partial \ln R/\partial n = \alpha u/R$, $\varepsilon_D = \beta v/R$. Relative-elasticity share $s \equiv \varepsilon_N/(\varepsilon_N+\varepsilon_D) = \alpha u/(\alpha u+\beta v)$.

---------------------------------------------------------------------------------------------------

## 0. Executive summary (what this strand contributes to the paper)

1. **Chinchilla is CES exactly when $\alpha=\beta$; otherwise it is a non-homothetic, additively separable technology whose elasticity of substitution is not constant.** [SYMPY] The Hicks/Allen/Morishima elasticity (all coincide with two inputs) is
   $$\sigma(N,D) = \frac{\alpha u+\beta v}{\alpha u(1+\beta)+\beta v(1+\alpha)}, \qquad \frac{1}{\sigma} = 1 + s\,\beta + (1-s)\,\alpha .$$
   With $\alpha=\beta$: $\sigma = 1/(1+\alpha)$ (verifies the starting hypothesis). On the compute-optimal expansion path the FOC forces $\alpha u = \beta v$ ($s=1/2$), so $\sigma^* = 2/(2+\alpha+\beta)$ is constant along the path even when $\alpha\neq\beta$.
   Numbers: Hoffmann et al. (rounded $\alpha=.34,\beta=.28$) $\sigma^*=0.763$ (range over input space [0.746, 0.781]); Besiroglu et al. (2024) replication ($\alpha=.3478,\beta=.3658$) $\sigma^*=0.737$; Muennighoff et al. (2023) C4 fit ($\alpha=\beta=.353$) $\sigma=0.739$. **Parameters and data are gross complements with $\sigma\approx 0.74$–$0.76$** — strikingly inside the range of capital–labor estimates in the macro/IO literature (Chirinko 2008: 0.4–0.6; Klump–McAdam–Willman 2007: 0.5–0.7; Oberfield–Raval 2021: 0.5–0.7; Doraszelski–Jaumandreu 2018 materials–labor 0.45–0.64 in the simplified model).
2. **Duality.** Minimizing $6ND$ s.t. $L\le \bar L$ gives the (inverse) cost function $L^*(C) = E + K (C/6)^{-\gamma}$ with $\gamma = \alpha\beta/(\alpha+\beta) = (1/\alpha+1/\beta)^{-1}$ (harmonic combination) and $K=(\alpha+\beta)(A/\beta)^{\beta/(\alpha+\beta)}(B/\alpha)^{\alpha/(\alpha+\beta)}$ [SYMPY]. Cost elasticity of reducible loss $\partial\ln C^*/\partial\ln R = -1/\gamma$ = −6.5 (Hoffmann) / −5.6 (Besiroglu). The n-input generalization reproduces Kaplan et al.'s (2020, Eq. 1.8) $\alpha_C^{min} = (1/\alpha_S+1/\alpha_B+1/\alpha_N)^{-1}$. Shephard's lemma holds in **log-input / log-cost-weight** space: with $\ln C = \theta_N n + \theta_D d$, $\partial V/\partial\theta_N = n^*$, $\partial V/\partial\theta_D=d^*$, with symmetry [SYMPY]. The compensated response of the optimal tokens-per-parameter ratio to relative log-cost weights is $\sigma/(1-\sigma) = 2/(\alpha+\beta)$ ≈ 2.8–3.2 at the symmetric point [SYMPY].
3. **Geometry of identification (new, clean).** In $(\ln N,\ln D)$, the Hessian of $\ln R$ is $s(1-s)\begin{pmatrix}\alpha^2 & -\alpha\beta\\ -\alpha\beta & \beta^2\end{pmatrix}$ — **rank one** [SYMPY], with null direction $(\beta,\alpha)$ = exactly the compute-optimal expansion direction. Along the path $\ln R$ is linear in $\ln C$ (a pure power law); **all curvature — hence all information about $\sigma$ — is transverse to the path.** This is the scaling-law version of functional dependence (Ackerberg–Caves–Frazer 2015) and of Diamond–McFadden–Rodriguez (1978). On-path-only data: Jacobian of the 5 Chinchilla parameters has rank 3 [NUM] (two singular values ~1e-16).
4. **DMR impossibility, scaling-law edition.** FLOPs give constant relative log-prices ($\theta_N=\theta_D=1$): there is never "factor-price variation". Compute-optimal models over time identify (i) effective-compute growth $g_N+g_D$ (from the frontier shift) and (ii) the difference in augmentation rates (from the drift of $D^*/N^*$), but **not $\sigma$, and therefore not the sign of the Hicks bias $(1-1/\sigma)(g_N-g_D)$**, unless one imposes Chinchilla's restriction that the outer exponent on the additive aggregator is 1 ($\kappa=1$), which ties substitution to scale (identification by functional form). With a Kaplan-style outer exponent $\kappa$ free, any $\sigma\in(0,1)$ rationalizes the same on-path data [SYMPY].
5. **Chinchilla's Approaches 1–3 = FOC equation vs. production function vs. system.** Approaches 1–2 estimate the FOC/expansion path ($a=\beta/(\alpha+\beta)$); Approach 3 estimates the production function. Klump–McAdam–Willman / León-Ledesma–McAdam–Willman show that jointly estimating production function + FOCs with **normalization** is what identifies $\sigma$ and biases in practice. Hoffmann's Approach 3 violates the cross-equation restriction ($a=.46$ vs .49–.50); Besiroglu's corrected fit restores it ($a=.513$). Under Hoffmann's own Approach-3 parameters Chinchilla itself sits off its predicted frontier (elasticity ratio 0.62; optimal $D/N\approx 93$ at Chinchilla compute), whereas under Besiroglu's it is on the frontier (ratio 1.03; $D/N\approx 18$).
6. **Normalization matters numerically.** On a Chinchilla-like IsoFLOP design, the raw parametrization has corr$(\ln A,\alpha)=0.9992$, corr$(\ln B,\beta)=0.9997$, condition number 3,465; normalizing inputs at geometric means (KMW 2007) cuts the condition number to 59 and the SD of the level parameter five-fold (still corr ≈ −0.98 because of $E$) [NUM]. This rationalizes Besiroglu et al.'s SE of 1,293 on $B=2085$.
7. **Inference wedge = DLW/Hsieh–Klenow-type wedge with a clean interpretation.** With lifetime cost $6ND + 2NT$ ($T$ inference tokens; Sardana & Frankle 2024 objective), optimality requires $\varepsilon_N/\varepsilon_D = 1 + T/(3D)$ = **(training + inference compute)/(training compute)**. So the observed output-elasticity ratio of an over-trained model reveals its planned lifetime compute multiple — the "markup" analog. Llama-3 8B (15T tokens): ratio 2.5 (Hoffmann params) or 5.2 (Besiroglu), i.e., implied lifetime inference ≈ 69T or 190T tokens; it uses 2.0× / 5.5× the training compute a frontier model would need for its loss (Farrell input-oriented cost efficiency 0.50 / 0.18 — allocative, not technical). Heavy caveat: extrapolation to $D/N=1875$ vs Chinchilla sample max $D/N=341$ (95th pct 159) [NUM].
8. **Factor-augmenting progress.** Ho et al. (2024) Eq. 2 literally is factor-augmenting technical change: $N_{eff}=Ne^{g_N t}$, $D_{eff}=De^{g_D t}$, with $\alpha_{year}=\alpha g_N$, $\beta_{year}=\beta g_D$. On the frontier any such change is observationally equivalent to compute-augmenting change at rate $g_N+g_D$ (Ho et al.'s $g_C=g_N+g_D$) [SYMPY]; Hicks-neutral (MRTS-preserving, multiplicative) change requires $\alpha g_N=\beta g_D$, i.e., a proportional shift of **both $A$ and $B$**, not of $E$. Ho et al.'s $\alpha_{year}$ and $\beta_{year}$ are individually insignificant and negatively correlated (95.2% of bootstrap draws have $\alpha_{year}+\beta_{year}>0$) — DMR in the data.
9. **Learning curves.** The data term $B D^{-\beta}$ is a Wright (1936) curve in cumulative tokens. Wright's "eighty percent curve" has exponent 0.322; Chinchilla β = 0.28 (progress ratio 82%), Besiroglu β = 0.366 (78%), Levitt–List–Syverson (2013) defects halve per 10× cumulative output (exponent 0.301, 81%). Muennighoff et al.'s repeated-data formula is literally geometric depreciation of an experience stock (perpetual inventory; cf. Benkard 2000 "forgetting"). Sahal's identity (Nagy et al. 2013) implies a naive time-series "scaling exponent" from frontier releases equals $\gamma/(1-s_A)$ where $s_A$ = algorithmic share of effective-compute growth; with Ho et al.'s 5–40% range, naive exponents are inflated by 5–67%.

---------------------------------------------------------------------------------------------------

## 1. Literature digest (IO / macro / production-theory side)

Format per entry: *What it does* — *key equations / estimates* — *identifying assumption / device* — **Mapping to scaling laws**.

### 1.1 CES: origins, estimation, generalizations

**Arrow, Chenery, Minhas & Solow (1961), REStat 43(3):225–250.** Introduce CES $V=\gamma[\delta K^{-\rho}+(1-\delta)L^{-\rho}]^{-1/\rho}$, $\sigma=1/(1+\rho)$. Derived from the empirical log-linear relation between value added per worker and the wage across countries (the "ACMS regression" $\ln(V/L)=a+\sigma\ln w$, i.e., estimation from the labor FOC under competition). KMW (2007, WP text) cite the ACMS US estimate as σ ≈ 0.57. **Mapping:** Chinchilla with $\alpha=\beta=\rho$ is exactly this CES in (N, D) with output index $Y=R^{-1/\rho}$. An "ACMS regression" for LLMs would regress $\ln(D^*/N^*)$ on log relative effective input cost (e.g., inference-driven shadow price of parameters) across labs: slope $=\sigma/(1-\sigma)$ in the log-cost formulation (Section 2.2).

**Uzawa (1962), RES 29(4):291–299.** Characterizes production functions with constant (Allen–)Uzawa partial elasticities of substitution; with more than two factors, constant pairwise elasticities essentially force a common σ (CES) or Cobb–Douglas nests of CES groups (standard statement of the result; not re-derived here). Reformulates Allen's partial elasticity via the cost function ($\sigma^{AU}_{ij}=C C_{ij}/(C_iC_j)$). **Mapping:** for multi-input scaling laws (params, tokens, test-time compute; or active vs. total params in MoE) the additively separable power form does **not** have equal/constant pairwise AES — [SYMPY/NUM] for $F=-(\sum_i c_i x_i^{-e_i})$ with $e=(.3,.4,.5)$ at an interior point, AES$_{12}$=0.770, AES$_{13}$=0.719, AES$_{23}$=0.668. So multi-input scaling laws need either nested CES (Sato 1967) or flexible forms.

**Kmenta (1967), IER 8:180–189.** Second-order Taylor expansion of the CES around $\rho=0$ (Cobb–Douglas): $\ln Y \approx \ln\gamma + \nu\delta\ln K+\nu(1-\delta)\ln L - \tfrac{\rho\nu}{2}\delta(1-\delta)[\ln K-\ln L]^2$. Known limits: only 2 inputs, reliable only for $\rho$ near 0 (micEconCES documentation). LLMW (2010) find Kmenta-based estimates perform poorly in Monte Carlo (e.g., for true σ=0.5, Kmenta $\hat\sigma$ = 0.347 with reported Monte Carlo interval [−0.603, 1.173] vs. system 0.541 [0.502, 0.601], Table 3a, T=50). **Mapping:** [SYMPY] $\ln(s_A e^{-\rho n}+(1-s_A)e^{-\rho d}) = -\rho(s_A n+(1-s_A)d)+\tfrac{\rho^2}{2}s_A(1-s_A)(n-d)^2+O(\rho^3)$. For LLM data the expansion variable is $\rho\,\Delta\ln(D/N)$; with ρ≈0.35 and $D/N$ spanning 0.04–341 in Chinchilla's sample (≈ ±4 log points around the center), $\rho\Delta\approx1.4$ — the approximation is poor unless centered (normalized) and the design is narrow. More useful: the exact local translog curvature of Chinchilla is **rank one** (Section 2.1c), which gives a testable restriction in a translog of $\ln(L-E)$ on $(\ln N,\ln D)$: $b_{NN}b_{DD}=b_{ND}^2$ with $b_{ND}<0$.

**Sato, K. (1967), RES 34(2):201–218.** Two-level (nested) CES. **Mapping:** natural for (i) parameters × data nested with test-time compute; (ii) "effective parameters" aggregating active and total MoE parameters; (iii) data mixtures (quality tiers) as an inner CES of token types.

**Hanoch (1971), Econometrica 39(5):695–712 (CRESH).** Homothetic implicitly-defined family $\sum_i (d_i/b_i)(x_i/Y)^{b_i}=1$ with constant *ratios* of elasticities. **Mapping:** Chinchilla with $\alpha\ne\beta$ is **not** CRESH (CRESH is homothetic; Chinchilla's MRTS along a ray $D=kN$ is $\propto N^{\beta-\alpha}$). Chinchilla is "directly additive, non-homothetic".

**Sato, R. (1975), Econometrica 43(5–6):999–1004** ("most general class of CES functions") and **Comin, Lashkari & Mestieri (2021), Econometrica 89(1):311–374** (implicitly defined non-homothetic CES $\sum_i\Omega_i^{1/\sigma}(x_i/Y^{\epsilon_i})^{(\sigma-1)/\sigma}=1$, constant σ but input-specific "income" elasticities). **Mapping:** Chinchilla with $\alpha\ne\beta$ is not in the CLM class either (CLM has a common exponent on inputs, Chinchilla has input-specific exponents and constant-$R$ isoquants); its σ varies with the input mix (Section 2.1). The CLM form is a candidate *alternative* scaling-law specification with constant σ but non-homothetic expansion paths (scale-dependent optimal $D/N$), worth testing against Chinchilla.

**Klump, McAdam & Willman (2007), REStat 89(1):183–192; (2012) J. Econ. Surveys 26(5):769–799 — normalized CES.** Normalized CES: $Y_t = Y_0\,\xi\,[\pi_0 (\Gamma^K_t K_t/K_0)^{\frac{\sigma-1}{\sigma}}+(1-\pi_0)(\Gamma^N_t N_t/N_0)^{\frac{\sigma-1}{\sigma}}]^{\frac{\sigma}{\sigma-1}}$ with baseline values = sample geometric means, scaling parameter ξ (≈1), and Box-Cox time paths for factor augmentation $g_i(t)=\frac{\gamma_i t_0}{\lambda_i}[(t/t_0)^{\lambda_i}-1]$ (λ=1 linear/exponential-level, 0 log-linear, <0 hyperbolic). "Supply-side system" = production function + two FOCs (profit shares with a markup μ). US 1953–1998: σ significantly below one, **0.5–0.7** (WP abstract); labor-augmenting progress ~exponential, capital-augmenting hyperbolic/logarithmic; markup ≈4%. Motivation for normalization: the CES parameters (distribution, efficiency) are not deep unless tied to a baseline point; otherwise σ and distribution parameters are confounded. **Mapping:** (i) *Normalized scaling laws*: write $L = E + a(N/\bar N)^{-\alpha}+b(D/\bar D)^{-\beta}$ at geometric means; $b/(a+b)$ at the normalization point is the reducible-loss share of the parameter term, and on the compute-optimal path with α=β it equals ½ (the analog of the distribution parameter $\pi_0$). Numerics in Section 2.7. (ii) *System estimation*: Chinchilla's Approaches 1–3 map to FOC-only, FOC-only (experimental), and production-function-only; the system estimator is the natural fix.

**León-Ledesma, McAdam & Willman (2010), AER 100(4):1330–1357.** Monte Carlo (T=25–100, 5,000 draws) on identifying σ with biased technical change. Key result (abstract): "jointly modeling the production function and first-order conditions is superior to single-equation approaches especially when merged with 'normalization'." Single-equation FOC or share regressions and Kmenta are unsuitable; direct nonlinear CES alone does not solve identification (esp. high σ). Example (Table 3a, $\gamma_N=.015,\gamma_K=.005$, T=50, true σ=0.5): nonlinear CES $\hat\sigma$=0.422 [0.109, 1.110]; system 0.541 [0.502, 0.601]. **Mapping:** direct template for estimating scaling laws: loss equation + allocation FOC (IsoFLOP minima) + cross-equation restrictions, normalized.

**Chirinko (2008), J. Macroeconomics 30(2):671–686.** Survey of 75 years of σ estimates; tension between short-run data and long-run parameter; weight of evidence σ ∈ [0.40, 0.60]. **Mapping:** benchmark: parameters–data σ ≈ 0.74 is "more substitutable than K–L but still complements". Also: short-run vs long-run distinction ↔ within-run (checkpoint) vs. across-run (fully re-tuned) substitution.

**Antràs (2004), Contributions to Macroeconomics 4(1).** Wrongly assuming Hicks-neutral technical change biases σ estimates toward unity; allowing constant-rate factor augmentation yields σ significantly below 1 (US private sector 1948–1998). **Mapping:** a cautionary parallel: pooled cross-lab regressions that absorb heterogeneous, drifting recipes into a common neutral term may bias the implied σ toward 1. (Suggestive: Ho et al.'s cross-lab exponents α=0.068, β=0.040 imply on-path σ*=2/(2.108)=0.95, vs. 0.74–0.76 from controlled experiments — but benchmarks/model scales differ, so this is at most a hypothesis.)

### 1.2 Definitions of the elasticity of substitution

**Hicks (1932), *The Theory of Wages*.** Introduces the elasticity of substitution and the classification of inventions (labor-saving, capital-saving, neutral). **Allen (1938), *Mathematical Analysis for Economists*.** Partial (Allen) elasticity for n inputs (primal, bordered-Hessian form); Uzawa (1962) cost-function form. **Morishima (1967), Keizai Hyoron 16:144–150** (in Japanese): asymmetric elasticity $M_{ij}=\varepsilon_{ji}-\varepsilon_{ii}$ (response of $x_i/x_j$ to $p_j$). **Blackorby & Russell (1989), AER 79(4):882–888**: the Allen/Uzawa elasticity is uninformative with >2 inputs (it is not a measure of curvature, does not give comparative statics of relative shares); Morishima is the natural generalization of Hicks' two-factor concept; with two inputs all coincide.
**Verified here [SYMPY]:** for Chinchilla, the primal Allen elasticity equals the Hicks direct elasticity; with two inputs $M_{ND}=s_N\sigma^{AU}_{ND}-s_N\sigma^{AU}_{NN}=\sigma$ using $\sigma^{AU}_{NN}=-(s_D/s_N)\sigma^{AU}_{ND}$. **Caveat:** dual (cost-function) definitions presume a linear cost $p\cdot x$; FLOP cost $6ND$ is multiplicative. Use the primal (technological) definitions, or the log-cost formulation of Section 2.2, where "prices" are log-cost weights θ.

### 1.3 Duality

**Shephard (1953), *Cost and Production Functions* (Princeton UP)**: cost function ↔ production function duality; Shephard's lemma $x_i^*=\partial C/\partial p_i$. **Uzawa (1964), IER 5(2):216–220**: duality principles. **McFadden (1978)**, ch. in Fuss & McFadden (eds.), *Production Economics: A Dual Approach*, North-Holland: cost/revenue/profit functions, McFadden duality theorem. **Diewert (1971), JPE 79(3):481–507**: generalized Leontief cost function; derived demands linear in parameters; attains any set of partial elasticities with minimal parameters ("flexible functional form"). **Christensen, Jorgenson & Lau (1973), REStat 55(1):28–45**: translog. **Mapping:** the compute-optimal frontier $L^*(C)$ is the (inverse) cost function; IsoFLOP curves are isocost curves; iso-loss contours are isoquants; Chinchilla's "efficient frontier" (their Fig. 4: "the curve goes through each iso-loss contour at the point with the fewest FLOPs") is the expansion path. Shephard's lemma in log-cost-weight form (Section 2.2). Translog in $(\ln N,\ln D)$ is the natural flexible nesting model; Chinchilla implies rank-one translog curvature.

### 1.4 Cost-function estimation of returns to scale

**Nerlove (1963)**, in C. Christ (ed.), *Measurement in Economics* (Grunfeld memorial volume): 145 US electric utilities, 1955; Cobb–Douglas cost function $\ln C = \kappa + \frac1r\ln Q + \sum_i\frac{a_i}{r}\ln p_i$; marked returns to scale at the firm level, declining with output; notes variations in returns to scale may be "neutral" (scale affects RTS but not MRS at given factor ratios). **Christensen & Greene (1976), JPE 84(4):655–676**: translog cost function, 1955 and 1970 cross-sections; in 1955 significant scale economies for nearly all firms; by 1970 the bulk of generation came from firms in the flat part of the AC curve. **Mapping:** Chinchilla Approach 1 (lower envelope of training curves: loss vs FLOPs) and Kaplan's $L(C_{min})$ are *cost-function* (dual) estimates, Approach 3 is a *production-function* (primal) estimate. The power-law frontier has a constant cost elasticity $-1/\gamma$ in reducible loss (no minimum efficient scale); Christensen–Greene-style curvature ↔ "curved"/broken frontiers and Chinchilla's own note of concavity of $\log N_{opt}$ in $\log C$ at large budgets. Nerlove's "neutral variation in returns to scale" ↔ exponent changes that leave the optimal $D/N$ unchanged. Nerlove/C-G data are in the `hayashir` R package (data leads).

### 1.5 Biased technical change and its identification

**Diamond, McFadden & Rodriguez (1978)**, ch. IV.2 in Fuss & McFadden vol. 2, pp. 125–147. Without assumed structure on technology, the elasticity of substitution and the factor biases of technical change cannot be separately identified from time series of output, inputs and marginal products (Oberfield & Raval 2021 phrase it: "cannot be identified from time series data on output, inputs, and marginal products alone. Instead, identification requires factor price movements that are independent of the bias of technical change"). Standard workaround: parametric (exponential) augmentation paths. **Mapping:** Section 2.4 (formal analog; FLOP prices never move).

**Acemoglu (2002), RES 69(4):781–809.** Directed technical change: price effect (innovation toward scarce factors) vs market-size effect (toward abundant factors); with σ<1 relative-bias results. **Mapping (speculative):** as unique data becomes scarce (data wall; Muennighoff et al.), the price effect predicts data-augmenting innovation (synthetic data, repetition-robust training, curation); abundant compute (market-size effect) predicts compute/parameter-using innovation (e.g., MoE). Testable with Ho-type panels by splitting $g_D$ over time.

**Klump–McAdam–Willman (2007/2012); León-Ledesma–McAdam–Willman (2010)** — see 1.1.

**Doraszelski & Jaumandreu (2018), JPE 126(3):1027–1084.** Firm-level production $Y_{jt}=F(K_{jt},e^{\omega_{Ljt}}L_{jt},M_{jt})e^{\omega_{Hjt}}e^{e_{jt}}$ (labor-augmenting ω_L and Hicks-neutral ω_H), CES with scale ν and substitution σ; Spanish manufacturing (1990s–2000s). **Identification key (quote):** "Hicks-neutral technological change scales input usage but, in contrast to labor-augmenting technological change, does not change the mix of inputs that a firm uses. A change in the input mix therefore contains information about the bias of technological change, provided we control for the relative prices." OLS with time trends gives σ>1 (biased up by correlation of ω_L with wages); GMM σ ∈ [0.45, 0.64] in the simplified model; both labor-augmenting and Hicks-neutral components raise output ≈1.5%/yr. **Mapping:** in LLMs the "input mix" is tokens-per-parameter $D/N$. Along compute-optimal paths, drift in $D^*/N^*$ at fixed FLOP prices is exactly the DJ signal of bias: $d\ln(D^*/N^*)/dt = 2(\alpha g_N-\beta g_D)/(\alpha+\beta)$ [SYMPY]. But because relative prices never vary, σ itself is not identified from this (Section 2.4).

**Raval (2019), RAND 50(1):147–167.** US manufacturing plants: large persistent variation in capital shares inconsistent with Cobb–Douglas; σ estimated from local wage variation (and instruments) **0.3–0.5**; labor-augmenting productivity persistent and correlated with exports, size, growth. **Mapping:** identification via exogenous cross-sectional price variation — the missing ingredient in FLOP-denominated LLM data; LLM analogs of "local wages": heterogeneous inference-demand expectations, memory/hardware constraints, data-access constraints across labs (all shift effective relative input costs without directly shifting the loss technology).

**Oberfield & Raval (2021), Econometrica 89(2):703–732.** Aggregate (industry) σ from micro: $\sigma^{agg}=(1-\chi)\sigma+\chi\varepsilon$, $\chi=\sum_i\frac{(\alpha_i-\alpha)^2}{\alpha(1-\alpha)}\theta_i$ (cost-weighted variance of capital shares; within-plant substitution plus reallocation across plants; demand elasticity ε). US manufacturing aggregate elasticity stable: "0.5–0.7" (published abstract) / "about 0.7" (2014 NBER WP abstract) since 1970. Builds on Houthakker (Leontief micro → Cobb–Douglas macro) and Sato (1975). **Mapping (loose):** an ecosystem-level "tokens vs parameters" substitution elasticity includes reallocation across heterogeneous models (small over-trained vs large Chinchilla-style) — the model-portfolio analog of χ.

**Demirer (forthcoming Econometrica 2026; MIT JMP Jan 2020), "Production Function Estimation with Factor-Augmenting Technology: An Application to Markups."** Extends Olley–Pakes to labor-augmenting productivity under a homothetic-separability restriction; identifies output elasticities nonparametrically via a control-variable approach plus optimality of input expenditures; standard (Hicks-neutral Cobb–Douglas) models underestimate capital elasticity by up to 70% and overestimate labor elasticity by up to 80%; markups overestimated by ~20 pp; markup growth about half of recent estimates. **Proposition 4.5 (JMP):** σ between effective labor and materials is *not identified*; "first-order conditions are only informative about the first derivatives of the production function, whereas the elasticity of substitution depends on the second derivatives"; "extends the impossibility theorem of Diamond et al. (1978) to a setup with firm-level data"; input-price variation restores identification (App. B.1). **Mapping:** *exactly* our on-path result: compute-optimal choices reveal first derivatives (ε_N=ε_D), σ needs curvature → off-path (IsoFLOP) variation. Also: output elasticities (and hence wedges) are identified only on the support of observed choices — relevant for extreme over-training.

**Gandhi, Navarro & Rivers (2020), JPE 128(8):2973–3016.** Proxy methods for gross output require extra variation in flexible-input demand (e.g., prices); nonparametric identification via transformation of the flexible input's FOC (share equation). **Mapping:** another no-price-variation non-identification result; the FOC-share route is the analog of using IsoFLOP optima (Approach 2) as "share equations". Also gross output vs value added ↔ distillation/synthetic data as intermediate inputs.

### 1.6 Growth accounting and TFP

**Solow (1957), REStat 39(3):312–320.** $Q=A(t)F(K,L)$, residual measure of Hicks-neutral shift; US 1909–1949: gross output per man-hour doubled, "87½ per cent of the increase attributable to technical change and the remaining 12½ per cent to increased use of capital"; shift ≈1%/yr first half, ≈2%/yr second half; technical change neutral on average. **Jorgenson & Griliches (1967), RES 34(3):249–283.** Correctly measured (quality-adjusted) inputs explain most of measured productivity change — the residual shrinks when inputs are measured as services. **Nadiri (1970), JEL 8(4):1137–1177.** Survey of TFP measurement approaches. **Hulten (2001),** in Hulten, Dean & Harper (eds.), *New Developments in Productivity Analysis* (NBER/U. Chicago Press): residual = Hicksian efficiency shift only under CRS, marginal-productivity pricing, Hicks neutrality; otherwise a "measure of our ignorance" (Abramovitz) with path dependence; index-number foundations. **Hall (1988), JPE 96(5):921–947.** With market power, cost-share-weighted Solow residuals are biased (price > marginal cost).
**Mapping:**
- Ho et al. (2024) Shapley decomposition is growth accounting for loss: "60–95% of the performance gains stem from compute scaling, while algorithms contribute only 5–40%"; e.g., RNN(2012)→GPT-3(2021): parameter scaling 48.6%, data scaling 32.4%, parameter efficiency 2.1%, data efficiency 16.8% (Table 1). **Contrast with Solow's 87.5% residual share**: LM progress is overwhelmingly input accumulation. (Metric caveat: shares of a nonlinear loss depend on the metric and ordering; Shapley handles ordering.)
- Jorgenson–Griliches ↔ Ho et al.'s admitted inability to distinguish data *quality* from algorithmic data *efficiency*: quality-adjusting tokens (effective-data services) moves "data-augmenting progress" into "input growth".
- Index-number growth accounting for LLMs is possible along compute-optimal paths because the FOC equalizes the output elasticities: $\varepsilon_N=\varepsilon_D=\gamma$ on the path [derived: $\varepsilon_N=\alpha u/(u+v)$ with $\alpha u=\beta v$ gives $\alpha\beta/(\alpha+\beta)$]. Hence $d\ln R = -\gamma(d\ln N+d\ln D) - d\omega$ requires only γ. Hall-type bias: if models are over-trained (ε_N>ε_D) but one weights by cost shares (equal log-weights), $\hat\omega-\omega=\tfrac{\varepsilon_N-\varepsilon_D}{2}(\Delta n-\Delta d)$: a shift toward over-training ($\Delta d>\Delta n$) makes the residual understate algorithmic progress (over-credits data growth).

### 1.7 Learning curves

**Wright (1936), J. Aeronautical Sciences 3(4):122–128.** Average labor cost vs quantity is a straight line in log–log; the "eighty percent curve" corresponds to exponent .322 (cost falls to 80% with each doubling); raw material 95% curve (.0732), purchased material 88% (.184) [verified from the PDF]. **Arrow (1962), RES 29(3):155–173.** Learning by doing: experience indexed by cumulative gross investment, technical change embodied; learning is an externality → competitive investment below optimum. **Benkard (2000), AER 90(4):1034–1054.** L-1011 cost data reject simple learning (cost must fall with cumulative output); strong support for organizational forgetting — experience depreciates. **Thompson (2010)** Handbook of the Economics of Innovation vol. 1 ch. 10, pp. 429–476; **Thompson (2012), JEP 26(3):203–224.** Evidence for organizational learning-by-doing is weaker than the unit-cost/cumulative-quantity correlation suggests; of Wright's three explanations only one is unambiguously learning, others are consistent with static scale economies. **Levitt, List & Syverson (2013), JPE 121(4):643–681.** Auto assembly plant: each ten-fold increase in cumulative production halves defect rates (⇒ exponent log10 2 = 0.301); knowledge embodied in the plant/process more than in workers. **Nagy, Farmer, Bui & Trancik (2013), PLoS ONE 8(2):e52669.** 62 technologies (Performance Curve Database): Wright $C=a_1Q^{-b_1}$ vs Moore $C=a_2e^{-b_2t}$; Sahal (1979): if $Q=Q_0e^{gt}$ then Wright holds with $w=m/g$; "Wright's law produces the best forecasts, but Moore's law is not far behind" — nearly indistinguishable because production grows exponentially.
**Mapping:**
- Data term = learning curve within a training run: exponents β ≈ 0.28–0.37 vs Wright 0.322 and LLS 0.301 — "the 80% regularity" (progress ratios 78–82%). Parameter term α ≈ 0.34–0.35 (79%).
- Muennighoff et al. (2023) effective data $D'=U_D+U_DR_D^*(1-e^{-R_D/R_D^*})$ is derived from geometric decay of each repeated token's value by δ per repetition, $R^*_D=(1-\delta)/\delta$, fitted $R^*_D=15.4$, $R^*_N=5.3$ (182 runs) → perpetual-inventory "experience stock" with depreciation (Benkard) and the Jorgenson capital-services logic; saturation at $U(1+R^*)$.
- Thompson's critique ↔ separating "learning" (D, experience) from "scale" (N, capacity) requires independent variation — IsoFLOP designs provide it.
- Sahal/Nagy ↔ scaling vs algorithmic progress: if frontier reducible loss falls at rate $m$ and physical compute grows at $g$, a naive regression of log loss on log compute across frontier vintages recovers $w=m/g=\gamma(g+g_A)/g=\gamma/(1-s_A)$, $s_A=g_A/(g+g_A)$. With Ho et al.'s 5–40% algorithmic share: bias factor 1.05–1.67. This is the Marschak–Andrews/omitted-TFP bias in closed form.

### 1.8 Frontiers and efficiency

**Farrell (1957), JRSS-A 120(3):253–290**: deterministic efficient frontier; overall (cost) efficiency = technical × price (allocative) efficiency. **Aigner, Lovell & Schmidt (1977), J. Econometrics 6(1):21–37; Meeusen & van den Broeck (1977), IER 18(2):435–444**: stochastic frontier with composed error $v-u$, $v$ normal, $u\ge0$ half-normal (ALS), ML estimation.
**Mapping (careful):** the *frontier* is the whole best-practice scaling law $f(N,D)$; $E$ is its asymptote (the entropy floor of the text distribution — an absolute technological bound like a thermodynamic limit), not the frontier itself. SFA analog: $\ln L_i = \ln f(N_i,D_i;\theta) + v_i + u_i$, $u_i\ge0$ = hyperparameter/recipe inefficiency (loss can only be worse than best practice). Chinchilla's Approach 1 (lower envelope over runs at each FLOP) is a Farrell/FDH-style deterministic frontier; Approach 3 with Huber δ=1e-3 on log loss is essentially median regression — a central-tendency, not frontier, estimator. Porian et al. (2024) show the Kaplan–Chinchilla discrepancy is driven by last-layer compute accounting, warmup, and **scale-dependent optimizer tuning** — i.e., inefficiency $u$ correlated with scale, which biases slope (exponent) estimates exactly as input-correlated inefficiency biases SFA/OLS slopes. Farrell decomposition for over-trained models: $C_{min}(L_i)/C_i$ is *allocative* (deliberate, inference-driven), not technical inefficiency (Section 2.6).

### 1.9 Wedges and markups (cross-strand anchors used in the math)

**De Loecker & Warzynski (2012), AER 102(6):2437–2471**: $\mu = \theta^V/\alpha^V$ (output elasticity of a variable input over its revenue share) under cost minimization. **Hsieh & Klenow (2009), QJE 124(4):1403–1448**: factor-market wedges $1+\tau_K$ = (MRPK/MRPL)/(r/w); reallocation gains 30–50% (China), 40–60% (India). **Foster, Haltiwanger & Syverson (2008), AER 98(1):394–425**: TFPQ vs TFPR ("physical productivity is inversely correlated with price while revenue productivity is positively correlated with price"). **Ackerberg, Caves & Frazer (2015), Econometrica 83(6):2411–2451**: functional dependence in OP/LP first stages. **Marschak & Andrews (1944), Econometrica 12(3–4):143–205**: simultaneity of inputs and productivity.
**Mapping:** Section 2.6 (inference wedge); benchmark accuracy as bounded transform of latent loss ↔ TFPR-type contamination (loose).

---------------------------------------------------------------------------------------------------

## 2. Derivations (verified)

### 2.1 Elasticity of substitution of Chinchilla

(a) **σ formula.** For two inputs the Hicks direct elasticity is
$\sigma = -\dfrac{f_Nf_D(Nf_N+Df_D)}{ND(f_{NN}f_D^2-2f_{ND}f_Nf_D+f_{DD}f_N^2)}$, invariant to any monotone transform of output (isoquants unchanged). With $f=-R$: $f_N=\alpha u/N$, $f_{NN}=-\alpha(1+\alpha)u/N^2$, $f_{ND}=0$, etc., giving
$$\sigma=\frac{\alpha u+\beta v}{\alpha u(1+\beta)+\beta v(1+\alpha)}\;\Longleftrightarrow\;\frac1\sigma = 1+s\beta+(1-s)\alpha,\quad s=\frac{\alpha u}{\alpha u+\beta v}. \quad\text{[SYMPY]}$$
Alternative derivation along the isoquant: $u\,d\ln u+v\,d\ln v=0\Rightarrow dd=-(\alpha u/\beta v)dn$; $\ln \text{MRTS}=\ln(\alpha u/\beta v)+\ln(D/N)$; $\sigma = d\ln(D/N)/d\ln\text{MRTS}$ gives the same expression (checked by hand).
- $\alpha=\beta=\rho$: $\sigma=1/(1+\rho)$ exactly (CES). [SYMPY]
- Range: $\sigma\in[1/(1+\max(\alpha,\beta)),\,1/(1+\min(\alpha,\beta))]$; $s\to1$ (parameter term dominates at the margin, data abundant) gives $1/(1+\beta)$.
- Compute-optimal path: FOC $\alpha u=\beta v$ ⇒ $s=\tfrac12$ ⇒ $\sigma^*=2/(2+\alpha+\beta)$, constant along the path. [SYMPY]
- Allen primal (bordered Hessian) = Hicks direct with two inputs [SYMPY]; Morishima coincides (Section 1.2).

(b) **Is Chinchilla a (generalized) CES after an output transform?**
- If $\alpha=\beta=\rho$: yes. $Y\equiv R^{-1/\rho}=(AN^{-\rho}+BD^{-\rho})^{-1/\rho}$ is CRS CES; $Y^{\nu}$ gives any degree of homogeneity ν. Normalized: $Y=Y_0[\pi_0(N/N_0)^{-\rho}+(1-\pi_0)(D/D_0)^{-\rho}]^{-1/\rho}$, $\pi_0=AN_0^{-\rho}/(AN_0^{-\rho}+BD_0^{-\rho})$ = share of the parameter term in reducible loss at the baseline (= ½ on the compute-optimal path).
- If $\alpha\ne\beta$: **no.** (i) Any monotone output transform leaves σ unchanged, but σ varies with $s$ ⇒ not constant ⇒ not CES. (ii) Non-homothetic: along $D=kN$, MRTS $=(\alpha A/\beta B)k^{\beta+1}N^{\beta-\alpha}$, constant iff α=β. (iii) It *is* CES in the power-transformed input $\tilde N=N^{\alpha/\beta}$ with $\sigma_{(\tilde N,D)}=1/(1+\beta)$ ("effective parameters"), but σ is not invariant to input reparametrization. (iv) Class: directly additive, non-homothetic; not CRESH (homothetic), not CLM non-homothetic CES (common input exponent).
- **Kaplan et al. (2020) Eq. (1.5):** $L(N,D)=[(N_c/N)^{\alpha_N/\alpha_D}+D_c/D]^{\alpha_D}$; Table 2 fit $\alpha_N=0.076$, $\alpha_D=0.103$, $N_c=6.4\times10^{13}$, $D_c=1.8\times10^{13}$ (headline Eq. 1.1–1.2: 0.076, 0.095, 8.8e13, 5.4e13). Inner aggregator: additively separable with exponents 0.738 on N and 1 on D; outer power $\kappa=\alpha_D$ and $E=0$. Hence the starting hypothesis "Kaplan's joint form is CES-type" is **partially right**: same additive family, but unequal inner exponents (σ ∈ [0.50, 0.575], on-path 0.535) and — crucially — an **outer exponent that decouples scale from substitution**. Caveat: this $L(N,D)$ is for early-stopped, possibly multi-epoch training where D is dataset size; its implied $N\propto C^{0.575}$ under $6ND$ is not comparable to Kaplan's headline $N\propto C^{0.73}$, which comes from $L(N,S)$ (Eq. 1.6–1.8).

(c) **Rank-one curvature.** $\nabla^2_{(n,d)}\ln R = s_u(1-s_u)\begin{pmatrix}\alpha^2&-\alpha\beta\\-\alpha\beta&\beta^2\end{pmatrix}$, $s_u=u/R$; determinant 0; null vector $(\beta,\alpha)$ [SYMPY]. The null direction is the compute-optimal path direction ($dn/dd = a/b=\beta/\alpha$). Consequences: (i) $\ln R$ is exactly linear in $\ln C$ along the path; (ii) all curvature is transverse; (iii) on-path data cannot inform curvature/σ except through functional form; (iv) testable restriction in a translog of $\ln(L-E)$: $b_{NN}b_{DD}=b_{ND}^2$, $b_{ND}<0$ locally.

### 2.2 Cost function, expansion path, Shephard's lemma

Problem: $\min_{N,D}6ND$ s.t. $E+AN^{-\alpha}+BD^{-\beta}\le\bar L$ (equivalently $\min L$ s.t. $6ND=C$). FOC: $\alpha AN^{-\alpha}=\beta BD^{-\beta}$ (equal output elasticities; the log-cost elasticities of $6ND$ w.r.t. N and D are both 1). Solution (Hoffmann et al. Eq. 4, verified [SYMPY]):
$N^*=G(C/6)^a$, $D^*=G^{-1}(C/6)^b$, $G=(\alpha A/\beta B)^{1/(\alpha+\beta)}$, $a=\beta/(\alpha+\beta)$, $b=\alpha/(\alpha+\beta)$.
Frontier / inverse cost function: $L^*(C)=E+K(C/6)^{-\gamma}$, $\gamma=\alpha\beta/(\alpha+\beta)$, $K=\frac{\alpha+\beta}{\beta}AG^{-\alpha}=(\alpha+\beta)(A/\beta)^{\frac{\beta}{\alpha+\beta}}(B/\alpha)^{\frac{\alpha}{\alpha+\beta}}$ [SYMPY; numeric check K=813.68 both forms for Hoffmann params]. Cost function: $C^*(L)=6\,[K/(L-E)]^{1/\gamma}$.
- Cost elasticities: $\partial\ln C^*/\partial\ln R=-1/\gamma$; $\partial\ln C^*/\partial\ln L=-(1/\gamma)\,L/(L-E)$ → explodes as $L\to E$.
- On-path output elasticities $\varepsilon_N=\varepsilon_D=\gamma$; scaling both inputs by λ lowers R by $\lambda^{-2\gamma}$.
- **n-input generalization:** $R=\sum_iA_ix_i^{-\alpha_i}$, $C=\kappa\prod_ix_i$ ⇒ $\gamma=(\sum_i1/\alpha_i)^{-1}$ and $x_i^*\propto C^{\gamma/\alpha_i}$ (log-compute shares $\gamma/\alpha_i$). Reproduces Kaplan Eq. (1.7)–(1.8): $N\propto C^{\alpha_C^{min}/\alpha_N}$, $\alpha_C^{min}=(1/\alpha_S+1/\alpha_B+1/\alpha_N)^{-1}=1/(1/0.76+1/0.21+1/0.076)=0.052$ vs. their empirical 0.050.
- **Shephard's lemma in log space.** Generalize cost to $\ln C=\text{const}+\theta_Nn+\theta_Dd$ (e.g., inference or memory costs change the weights). Value $V(\bar R;\theta)=\min\{\theta_Nn+\theta_Dd: R(n,d)\le\bar R\}$. Then $\partial V/\partial\theta_N=n^*$, $\partial V/\partial\theta_D=d^*$, and $\partial n^*/\partial\theta_D=\partial d^*/\partial\theta_N$ [SYMPY]. Compensated response: $\partial\ln(D^*/N^*)/\partial\ln\theta_N=(\theta_N+\theta_D)/(\alpha\theta_D+\beta\theta_N)=\sigma/(1-\sigma)$ evaluated at the optimum; $=2/(\alpha+\beta)$ at $\theta_N=\theta_D$ [SYMPY]. (In levels, implicit prices are $p_N=\theta_NC/N$, $p_D=\theta_DC/D$, so $d\ln(p_N/p_D)=d\ln(\theta_N/\theta_D)+d\ln(D/N)$, which converts σ into σ/(1−σ).)
- Numbers [NUM]:

| Parameter set | a (N) | γ | −∂lnC/∂lnR | σ* | σ range | σ*/(1−σ*) | N*, D*, D/N at Chinchilla compute (5.88e23) |
|---|---|---|---|---|---|---|---|
| Hoffmann 2022 rounded (E 1.69, A 406.4, B 410.7, α .34, β .28) | .452 | .1535 | 6.51 | .763 | [.746,.781] | 3.23 | 32.5B, 3.02T, 93 |
| Hoffmann TeX precision (α .3392, β .2849; per Besiroglu) | .456 | .1548 | 6.46 | .762 | [.747,.778] | 3.20 | 40.7B, 2.41T, 59 |
| Besiroglu 2024 (E 1.817, A 482.0, B 2085.4, α .3478, β .3658) | .513 | .1783 | 5.61 | .737 | [.732,.742] | 2.80 | 73.0B, 1.34T, 18 |
| Muennighoff 2023 C4 (E 1.87, A 521, B 1488, α=β .353) | .500 | .1765 | 5.67 | .739 | — | 2.83 | 70.8B, 1.38T, 20 |

Consistency check: TeX-precision Hoffmann parameters reproduce the paper's 40B-parameter projection for the Gopher budget; rounded ones give 32B (rounding sensitivity flagged by Besiroglu et al.).

### 2.3 Factor-augmenting technical change

$N\to e^{g_Nt}N$, $D\to e^{g_Dt}D$ ⇒ $A_t=Ae^{-\alpha g_Nt}$, $B_t=Be^{-\beta g_Dt}$ (Ho et al. 2024 Eq. 3: $\alpha_{year}=\alpha g_N$, $\beta_{year}=\beta g_D$). Results [SYMPY]:
- Frontier: $d\ln K_t/dt=-\gamma(g_N+g_D)$ ⇒ $L_t^*(C)=E+K(Ce^{(g_N+g_D)t}/6)^{-\gamma}$: **observationally equivalent to compute-augmenting progress at rate $g_N+g_D$ at every scale** (matches Ho et al.'s $g_C=g_N+g_D$).
- Allocation: $d\ln G_t/dt=(\beta g_D-\alpha g_N)/(\alpha+\beta)$; $d\ln(D^*/N^*)/dt=2(\alpha g_N-\beta g_D)/(\alpha+\beta)$.
- Hicks bias at fixed inputs: $d\ln\text{MRTS}_{ND}/dt=\beta g_D-\alpha g_N$; CES case $=\rho(g_D-g_N)=(1-1/\sigma)(g_N-g_D)$ — standard: with σ<1, faster parameter augmentation is data-*using* (raises relative marginal product of data).
- **Hicks-neutral (multiplicative) change** ⇔ MRTS unchanged ⇔ $A_t/B_t$ constant ⇔ $\alpha g_N=\beta g_D$ (NOT equal augmentation rates unless α=β) ⇔ $R_t=\lambda_tR_0$; with output index $Y=R^{-1/\rho}$ this is a TFP multiplier $\lambda_t^{-1/\rho}$. So the starting hypothesis is refined: Hicks-neutral change is a common proportional shift in A and B, not a shift in E.
- **Shift in E** also leaves MRTS unchanged (isoquant map relabeled — "additively neutral"), but it is not a TFP multiplier on any homogeneous output index, and its compute-equivalent value is scale dependent: $(\,(L-E_{new})/(L-E_{old})\,)^{-1/\gamma}$ depends on L. Conceptually E is the entropy of the evaluation distribution; estimated E moving with "technology" signals misspecification (E acting as an architecture-specific asymptote) or a change in the evaluation data/tokenizer (an output-measurement change).
- Exponent changes (Ho et al. models 13–15; transformer vs LSTM) are scale-biased: compute-equivalent gains depend on scale (Nerlove-type non-neutral variation in returns to scale).
- Ho et al. point estimates (Table 2, model 7): $\alpha_{param}=0.068$ [0.045, 0.127], $\beta_{data}=0.040$ [0.023, 0.062], $\alpha_{year}=0.004$ [−0.058, 0.032], $\beta_{year}=0.036$ [−0.002, 0.080] ⇒ $g_N\approx0.06$/yr, $g_D\approx0.9$/yr, Hicks bias $\beta_{year}-\alpha_{year}=+0.032$/yr (parameter-using) but not significant; effective compute doubling 8.4 months [4.5, 14.3]. The strong negative correlation of $(\alpha_{year},\beta_{year})$ with a well-identified sum is precisely the DMR/collinearity pattern.

### 2.4 DMR impossibility in the scaling-law setting

Setup: at each date t, observe compute-optimal models across budgets C: $\{C, N^*_t(C), D^*_t(C), L^*_t(C)\}$. FLOP log-prices fixed ($\theta_N=\theta_D=1$).

**Generalized technology:** $L=E+(A_tN^{-a_1}+B_tD^{-b_1})^{\kappa}$ (Chinchilla: κ=1; Kaplan-type: κ free). σ depends only on $(a_1,b_1)$ (outer transforms don't change isoquants [SYMPY]).
On-path observables per date: allocation exponent $a=b_1/(a_1+b_1)$, frontier exponent $\gamma=\kappa a_1b_1/(a_1+b_1)$, levels $G_t,K_t$, and E.

**Proposition (non-identification of σ).** For any $S=a_1+b_1>0$ set $b_1=aS$, $a_1=(1-a)S$, $\kappa=\gamma/(a(1-a)S)$ [SYMPY]; this reproduces the same $(a,\gamma)$ and (with suitable $A_t,B_t$) the same levels, while $\sigma^*=2/(2+S)$ sweeps (0,1). Hence **σ is not identified from on-path data**. Numerically, the on-path-only Jacobian for the five Chinchilla parameters has rank 3 [NUM].

**What is identified.** (i) With κ=1: $\alpha=\gamma/a$, $\beta=\gamma/(1-a)$, then A, B from $G,K$ — full identification, but *entirely by functional form* (the restriction that the same exponent governs scale and curvature). Check: Hoffmann $a=.4516$, $\gamma=.1535$ ⇒ $\alpha=.34$, $\beta=.28$. (ii) Regardless of κ: $g_N+g_D$ from the drift of $K_t^\kappa$ (since γ is observed) and $-(1-a)g_N+ag_D$ from the drift of $G_t$ ⇒ **augmentation rates are identified, σ is not**, so the Hicks bias $(1-1/\sigma)(g_N-g_D)$ — including its **sign** (it flips at σ=1) — is not identified. With the CES special case ($a_1=b_1=\rho$) this is transparent: $D^*/N^*=(B_t/A_t)^{1/\rho}$ reveals $g_N-g_D$ directly, and $L^*$ reveals $\kappa\rho/2$ and $g_N+g_D$; ρ alone is free.

**Sources of identifying variation (all "price-like" shifters excluded from the loss technology):** (1) designed off-path experiments — IsoFLOP sweeps (Chinchilla's extracted sample: $D/N$ from 0.04 to 341, variance transverse to the path 1.12 vs 1.84 along it [NUM]); (2) inference-cost wedges that vary across labs/models (shift $\theta_N$); (3) data-availability constraints (Muennighoff: unique-token budgets); (4) hardware memory/throughput limits that constrain N; (5) the Kaplan-era allocation (beliefs-driven departures, $N\propto C^{0.73}$). These play the role of Raval's local wages / Oberfield–Raval's factor-price movements / Demirer's input-price variation.

Link to ACF (2015): along the path $n=\ln G+a(c-\ln6)$, $d=-\ln G+b(c-\ln6)$ — both deterministic in c (functional dependence); lab-specific $G$ (recipe "TFP" that tilts the allocation) breaks collinearity only if G-shifters are excluded from the loss equation — an exclusion restriction analogous to ACF's timing/optimization-error assumptions.

### 2.5 E and frontier analysis

Model: $\ln L_i=\ln[E+AN_i^{-\alpha}+BD_i^{-\beta}]+v_i+u_i$, $u_i\ge0$. Remarks: (i) E is the asymptote, poorly identified jointly with exponents: design correlations corr(E,α)=0.66, corr(E,β)=0.82 on a Chinchilla-like IsoFLOP grid [NUM]; published E estimates 1.69 (Hoffmann) vs 1.82 (Besiroglu) vs 1.87 (Muennighoff, C4). (ii) Approach 1's lower envelope is a Farrell/FDH deterministic frontier estimator; Approach 3 (Huber on log loss, δ=1e-3) ≈ LAD — a central-tendency fit whose intercept/levels absorb $E[u]$ (COLS logic). (iii) Scale-dependent tuning quality (Porian et al. 2024) = $u$ correlated with inputs ⇒ biased exponents. (iv) Farrell decomposition applies to $C_{min}(L_i)/C_i$ (Section 2.6).

### 2.6 Inference wedge (DLW / Hsieh–Klenow analog)

Lifetime objective (Sardana & Frankle 2024, Eq. 2 with 6N per training token and 2N per inference token): $\min_{N,D}6ND+2NT$ s.t. $L(N,D)=\ell$. FOC: $\dfrac{\varepsilon_N}{\varepsilon_D}=\dfrac{6D+2T}{6D}=1+\dfrac{T}{3D}=\dfrac{\text{lifetime compute}}{\text{training compute}}\equiv w.$
Interpretation: $w$ is to scaling laws what $\mu=\theta^V/\alpha^V$ is to DLW (ratio of output elasticity to cost elasticity revealing an unobserved wedge), and what $1+\tau$ is to Hsieh–Klenow (marginal-product ratio relative to price ratio). Unlike a markup it is a shadow cost of a second use of the same input. $w<1$ (under-training) means a shadow *subsidy* to parameters — e.g., GPT-3/Gopher-era models, rationalizable as optimizing against Kaplan-era beliefs.

[NUM] (loss from Chinchilla law; C ratio = actual training FLOPs / minimal FLOPs on the training-only frontier for the same loss):

| Model (N, D) | D/N | w (Hoffmann) | implied T/D | C/C_min | w (Besiroglu) | implied T/D | C/C_min |
|---|---|---|---|---|---|---|---|
| Chinchilla 70B, 1.4T | 20 | 0.62 | −1.14 | 1.20 | 1.03 | 0.09 | 1.00 |
| GPT-3 175B, 300B | 1.7 | 0.30 | — | 2.98 | 0.43 | — | 1.65 |
| Gopher 280B, 300B | 1.1 | 0.25 | — | 3.94 | 0.36 | — | 2.01 |
| Llama-2 7B, 2T | 286 | 1.50 | 1.51 | 1.14 | 2.62 | 4.85 | 1.86 |
| Llama-2 70B, 2T | 29 | 0.69 | — | 1.12 | 1.17 | 0.52 | 1.02 |
| Llama-3 8B, 15T | 1875 | 2.52 | 4.57 | 1.99 | 5.22 | 12.65 | 5.51 |
| Llama-3 70B, 15T | 214 | 1.21 | 0.62 | 1.03 | 2.45 | 4.36 | 1.72 |

(Model sizes/tokens: GPT-3, Gopher, Chinchilla from Hoffmann et al. Table 1; Llama-2 2T and Llama-3 15T tokens as stated in Sardana & Frankle.) Local σ stays in 0.734–0.771 across these points. Caveats: different data/tokenizers than MassiveText; extrapolation far outside the Chinchilla support (max D/N 341); Sardana & Frankle report that laws fitted at typical ratios **overestimate** the value of extra tokens at extreme ratios — so true $\varepsilon_D$ at D/N≈1875 is likely lower and true $w$ higher. Present as illustrative "revealed inference demand", with ranges across parameter sets.

### 2.7 Normalization numerics (KMW 2007 applied to scaling laws)

Design: 12 log-spaced model sizes (7e7–1.6e10) × 9 IsoFLOP budgets (6e18–3e21), keeping 5e9 ≤ D ≤ 5e11 → 61 points; parameters at Besiroglu values; Gauss–Newton covariance of NLS on log loss [NUM]:
- raw $(E,\ln A,\ln B,\alpha,\beta)$: corr(lnA,α)=0.9992, corr(lnB,β)=0.9997, corr(α,β)=0.17, condition number 3,465;
- normalized at geometric means $(E,\ln a,\ln b,\alpha,\beta)$: corr(ln a,α)=−0.979, corr(ln b,β)=−0.984, condition number 59; SD of level parameter 37.6 → 7.6 (same σ²).
- on-path-only design (30 compute-optimal points, 1e18–1e24): singular values (22.3, 1.28, 0.064, 8.8e−16, 6.3e−17) ⇒ rank 3.
Takeaway: report normalized levels; the residual α–level correlation stems from E (asymptote) — a reason to use system estimation (FOC adds information on $\alpha A N^{-\alpha}/\beta BD^{-\beta}$).

### 2.8 Train-time vs test-time compute isoquant

Jones (2021, AlphaZero on Hex): "the trade-off is linear in log-compute: for each additional 10× of train-time compute, about 15× of test-time compute can be eliminated"; Fig. 9 fit $\log(\text{test})=-1.2\log(\text{train})+0.004\cdot\text{Elo}+29$. ⇒ Cobb–Douglas isoquants in (train, test) with **σ=1** and output-elasticity ratio train:test = 1.2 (Elo ≈ 300 log10 train + 250 log10 test + const, Elo acting as log output). Under a linear per-episode cost the cost-minimizing training share would be 1.2/2.2 = 0.545; with amortization over Q queries, test cost is multiplied by Q ⇒ training share rises — the same logic as the inference wedge. Snell et al. (2024): compute-optimal test-time strategies improve test-time efficiency ">4x" vs best-of-N and, FLOPs-matched, "outperform a 14x larger model" on problems where the small model has non-trivial success; effectiveness depends on prompt difficulty ⇒ substitution elasticity between test-time compute and parameters is heterogeneous across "tasks" (Oberfield–Raval-style aggregation over prompt difficulty). Note floors (1-node search) = corner solutions; Elo/accuracy bounded ⇒ σ estimates depend on output cardinalization.

---------------------------------------------------------------------------------------------------

## 3. ML ↔ IO dictionary (this strand)

| ML concept | IO/econometrics concept | Strength | Note |
|---|---|---|---|
| Iso-loss contours | Isoquants | exact | Chinchilla Fig. 4 left. |
| IsoFLOP curves ($6ND$=C) | Isocost curves | exact | Isocost is linear in logs: n+d=const. |
| Compute-optimal "efficient frontier" $N^*(C),D^*(C)$ | Expansion path | exact | Closed form Hoffmann Eq. 4. |
| $L^*(C)$ (loss vs compute envelope) | (Inverse) cost function; Nerlove (1963) | exact | $C^*(L)=6[K/(L-E)]^{1/\gamma}$. |
| Chinchilla with α=β | CES, σ=1/(1+α) | exact | Y=R^{-1/α}. |
| Chinchilla with α≠β | Non-homothetic, directly additive technology; variable σ | close | $1/\sigma=1+s\beta+(1-s)\alpha$; σ*=2/(2+α+β). |
| Kaplan L(N,D) Eq. 1.5 | Generalized CES with outer exponent (scale decoupled from substitution) | close | inner exps 0.738/1; κ=0.103; different training regime. |
| Frontier exponent γ=αβ/(α+β) | Cost elasticity / returns to scale in loss units | exact | n-input harmonic rule reproduces Kaplan Eq. 1.8. |
| Approaches 1–2 (allocation) vs 3 (parametric loss) | FOC / share-equation vs production-function vs system estimation (KMW 2007, LLMW 2010) | close | Cross-equation restriction a=β/(α+β) testable; Hoffmann fails it, Besiroglu passes. |
| IsoFLOP sweeps | Designed exogenous "relative price" variation that breaks functional dependence | close | Transverse variation identifies curvature (σ). |
| On-path collinearity of ln N, ln D, ln C | Functional dependence (ACF 2015) | close | Rank-3 Jacobian on path. |
| σ and bias from compute-optimal models over time | Diamond–McFadden–Rodriguez impossibility | exact | FLOP log-prices constant; sign of Hicks bias unidentified. |
| Parameter-/data-efficiency progress (Ho et al. Eq. 2) | Factor-augmenting technical change | exact | α_year=αg_N, β_year=βg_D. |
| Effective-compute doubling (8.4 months) | TFP growth in compute-equivalent units | close | Equivalent to g_N+g_D on frontier; residual depends on maintained law. |
| Hicks-neutral algorithmic progress | Proportional shift of A and B (αg_N=βg_D) | exact | Not an E shift. |
| Shift in irreducible loss E | Output-measurement change / additive neutral shift; asymptote change | loose | E should be invariant (entropy). |
| Tokens-per-parameter drift at fixed FLOP prices | Input-mix signal of bias (Doraszelski–Jaumandreu 2018) | close | d ln(D*/N*)/dt=2(αg_N−βg_D)/(α+β). |
| Normalized scaling law at geometric-mean inputs | Normalized CES (Klump–McAdam–Willman) | exact | cond. number 3,465→59. |
| Over-training for inference (Llama) | DLW-type wedge / Hsieh–Klenow factor wedge; Farrell allocative inefficiency | close | w=ε_N/ε_D=lifetime/training compute. |
| Under-training (GPT-3 era) | Belief-driven wedge (optimizing against Kaplan's law) | loose | w≈0.3–0.4. |
| Shapley decomposition of loss gains | Solow growth accounting | close | Algorithms 5–40% vs Solow's 87.5% residual. |
| Data quality vs data-efficiency ambiguity | Jorgenson–Griliches quality-adjusted inputs | close | Quality-adjusted tokens absorb "progress". |
| Over-training bias in cost-weighted progress accounting | Hall (1988) markup bias in Solow residual | close | bias=(ε_N−ε_D)(Δn−Δd)/2. |
| Loss vs tokens (data term) | Wright learning curve | close | β 0.28–0.37 vs Wright 0.322, LLS 0.301. |
| Repeated-data decay (Muennighoff R*) | Depreciation / perpetual inventory; Benkard forgetting | close | R*=(1−δ)/δ; R*_D=15.4. |
| Scaling exponent from frontier-release time series | Wright vs Moore (Sahal identity; Nagy et al.) | exact | naive w=γ/(1−s_A); 5–67% upward bias. |
| Scale-dependent hyperparameter mis-tuning (Porian et al.) | SFA inefficiency correlated with inputs | close | Biases exponents. |
| Irreducible loss E | Stochastic frontier (ALS/MvdB) | loose | E is the asymptote; the frontier is the whole law. |
| Envelope over runs (Approach 1) | Farrell/FDH deterministic frontier | close | |
| Train vs test-time compute (Jones 2021) | Cobb–Douglas isoquant (σ=1) | close | 10× train ↔ 15× test. |
| Multi-input laws (N, D, test-time; MoE active/total) | Nested CES (Sato 1967); Allen vs Morishima (Blackorby–Russell) | loose | Additive power law gives unequal AES across pairs. |
| Ecosystem-level substitution with model-mix reallocation | Oberfield–Raval micro-to-macro σ | loose | σ_agg=(1−χ)σ+χε. |
| Direction of algorithmic progress vs data scarcity | Directed technical change (Acemoglu 2002) | loose | Testable prediction. |
| "Returns to scale" of loss | Returns to scale | breaks-down | Not invariant to output cardinalization; loss is bounded; only isoquant map and loss-denominated cost function are meaningful. |
| Dual (Allen/Morishima) elasticities with FLOP costs | Cost-function elasticities with linear prices | breaks-down | 6ND is multiplicative; use primal or log-cost-weight formulation. |

---------------------------------------------------------------------------------------------------

## 4. Contribution ideas for the paper (from this strand)

1. **Proposition: scaling-law DMR theorem.** Formalize Section 2.4 (σ unidentified on-path; augmentation rates identified; sign of Hicks bias unidentified; Chinchilla's κ=1 identifies by functional form). Corollary: what IsoFLOP sweeps identify (transverse curvature).
2. **Rank-one curvature test.** Translog in $(\ln N,\ln D)$ for $\ln(L-E)$; test $b_{NN}b_{DD}=b_{ND}^2$ (Chinchilla) and $b_{NN}=b_{DD}=-b_{ND}$ (α=β CES). Use Epoch's extracted Chinchilla points (245) and Muennighoff's 400+ runs.
3. **System (KMW/LLMW) estimator for scaling laws.** Stack loss equation + IsoFLOP-minimum FOC with cross-equation restrictions, normalized at geometric means; Wald/Hausman test of Approach-3-vs-Approach-2 consistency; explains Hoffmann vs. Besiroglu.
4. **Report σ with CIs** (≈0.74–0.76) and test α=β (CES) — Besiroglu's α≈β and Muennighoff's α=β suggest CES is not rejected; compare with capital–labor σ literature.
5. **Generalized Chinchilla with outer exponent κ** (nests Kaplan): test κ=1; quantify how σ inference changes when scale and substitution are decoupled.
6. **Revealed lifetime-compute multiples** (w) for open-weight model families (Llama, etc.) from their (N, D) choices — DLW-style; bounds that account for extrapolation (partial identification over parameter sets); relate to observed usage/pricing.
7. **Stochastic-frontier scaling laws.** Composed error with one-sided recipe inefficiency; lab-level efficiency dispersion (Syverson-style 90/10 in compute-equivalent units); correct for scale-dependent tuning (Porian).
8. **Index-number growth accounting of LLM progress** using on-path elasticity equality (weights γ), with Hall-type correction for over-training wedges; compare with Ho et al. Shapley shares.
9. **Sahal-bias correction** for naive "scaling from frontier releases" regressions: $w=\gamma/(1-s_A)$.
10. **Directed technical change test**: is data-augmenting progress accelerating as unique data becomes scarce?
11. **Learning-curve meta-regularity**: place scaling exponents in the Performance Curve Database distribution of Wright exponents (Nagy et al.); the "80% regularity".
12. **Replicate Nerlove (1963)/Christensen–Greene (1976)** cost-function estimation on the LLM frontier as a didactic bridge (same code on both datasets).

---------------------------------------------------------------------------------------------------

## 5. Data leads

- Epoch AI `analyzing-chinchilla` repo — `data/svg_extracted_data.csv` (245 points: Model Size, Training FLOP, loss; extracted from Chinchilla Fig. 4). https://github.com/epoch-research/analyzing-chinchilla (downloaded and inspected: N 5.7e7–1.6e10, D 2.5e8–3.2e11, C 1.4e18–1.3e22, D/N 0.04–341).
- Muennighoff et al. `datablations` — 400+ runs, `utils/parametric_fit.ipynb` with fitting data (182-sample fit), Apache 2.0. https://github.com/huggingface/datablations
- Epoch AI "Data on AI Models" (3,600+ models; params, training compute, dataset size, date, organization; CC-BY 4.0; CSV). https://epoch.ai/data/ai-models (page: https://epoch.ai/data/notable-ai-models)
- Ho et al. (2024) algorithmic-progress dataset (~231 models on WikiText-2/103, PTB, 2012–2023). https://epoch.ai/publications/algorithmic-progress-in-language-models (data availability not verified).
- Sardana & Frankle (2024) 47 models, 150M–6B, 10–10,000 tokens/param. https://arxiv.org/abs/2401.00448 (data release not verified).
- Nerlove (1963) 145-firm electricity data, R `hayashir::nerlove`. https://lachlandeer.github.io/hayashir/reference/nerlove.html
- Christensen–Greene (1976) data, R `hayashir::greene`. https://lachlandeer.github.io/hayashir/reference/greene.html
- LLMW (2010) replication data (openICPSR 112363). https://www.openicpsr.org/openicpsr/project/112363/version/V1/view
- Demirer replication package (Zenodo 21142853). https://zenodo.org/records/21142853
- Performance Curve Database (62 technologies, Nagy et al. 2013). pcdb.santafe.edu (URL from paper; not fetched).

---------------------------------------------------------------------------------------------------

## 6. Warnings, disagreements, uncertain claims

1. **Chinchilla parameter uncertainty dominates.** Hoffmann's published (rounded) α=.34, β=.28 vs TeX .3392/.2849; Besiroglu et al. re-fit (E 1.8172 (.03), A 482.01 (124.58), B 2085.43 (1293.23), α .3478 (.02), β .3658 (.02); 240 points after dropping 5 outliers; Huber δ=1e-3; 4,000 bootstraps) differs materially; Hoffmann's Approach-3 bootstrap intervals for a, b ([.454,.455], [.542,.543]) are implausibly narrow ("would require over 600,000 experiments"). Besiroglu attribute the misfit to rounding and early optimizer termination (averaging vs summing Huber losses). σ* moves only from .763 to .737, but the optimal D/N at Chinchilla compute moves from 93 to 18 — second-order (σ) quantities are more robust than allocation levels.
2. **σ identification is by functional form on the path** (κ=1). Any claim "σ≈0.74" must be stated as conditional on the Chinchilla aggregator or estimated from off-path (IsoFLOP) variation.
3. **Wedge numbers are extrapolations** (D/N up to 1875 vs sample max 341; different datasets/tokenizers). Direction of likely bias: understated wedges (Sardana & Frankle).
4. **Multiplicative cost.** Standard dual elasticity formulas presume linear costs; use the primal/log-cost formulations derived here.
5. **Returns to scale / "diminishing returns to compute"** statements are cardinalization-dependent (loss bounded, not a quantity). Only isoquants (ordinal) and loss-denominated cost functions are invariant objects.
6. **Kaplan comparisons:** Kaplan's joint L(N,D) (early-stopped) and headline $N\propto C^{0.73}$ (from L(N,S), non-embedding params) are different objects. Pearce & Song (2024) attribute much of the 0.73 vs 0.50 gap to non-embedding parameter counting at small scale (input measurement error correlated with scale); Porian et al. (2024) to last-layer FLOPs, warmup, and scale-dependent optimizer tuning.
7. **Ho et al. exponents** (α_param .068, β_data .040) are from small models on WikiText/PTB across labs — not comparable to Chinchilla; the implied σ*≈0.95 is only suggestive of Antràs-type bias toward 1.
8. **Muennighoff α=β=0.3526596**: text says the fit "results in" α=β; whether equality was imposed is unclear (flag before claiming "independent evidence for CES").
9. **Oberfield–Raval:** published abstract "0.5–0.7" vs 2014 NBER WP "about 0.7" — cite the published version.
10. **Demirer:** cite as forthcoming Econometrica (2026) per author website; results quoted from the Jan 2020 JMP (Prop. 4.5 numbering may differ in the published version).
11. **Starting hypotheses checked:** (i) σ=1/(1+α) when α=β — confirmed; (ii) Kaplan joint form CES-type — partially (non-homothetic inner aggregator + outer exponent); (iii) Hicks-neutral as shift in A,B or E — refined (common proportional A,B shift; E-shift is a different, measurement-type object); (iv) functional dependence ⇒ α,β not identified on-path — refined: *not identified without* the κ=1 restriction; identified *with* it plus allocation data; σ never identified nonparametrically on-path; (v) DMR — confirmed and sharpened (augmentation rates identified, σ and bias sign not); (vi) over-training = markup-like wedge — confirmed with a closed-form interpretation (lifetime/training compute), but closer to a factor wedge/allocative inefficiency than to market power; (vii) AES equal across pairs under additive separability — my own conjecture, **refuted** numerically.
12. **Not verified this session (do not cite without checking):** Sahal (1979) original; Houthakker (1955); Young (1995); Abramovitz (1956) (only as quoted by Hulten); any exact Benkard depreciation rates (scanned PDF not machine-readable); Wright's data beyond the quoted curve exponents.

---------------------------------------------------------------------------------------------------

## 7. Verification log (what was actually read)

Full text parsed: Hoffmann et al. 2022 (arXiv PDF), Kaplan et al. 2020 (PDF), Muennighoff et al. 2023 (PDF), Sardana & Frankle 2024 (PDF), Jones 2021 (PDF), Ho et al. 2024 (arXiv HTML), Besiroglu et al. 2024 (arXiv HTML), León-Ledesma–McAdam–Willman ECB WP 1001 (PDF), Klump–McAdam–Willman ECB WP 367 (PDF), Doraszelski–Jaumandreu 2018 (JPE PDF), Raval 2019 (author PDF), Oberfield–Raval NBER WP 20452 (PDF), Demirer JMP 2020 (PDF), Solow 1957 (PDF), Hulten 2001 (NBER chapter PDF), Wright 1936 (PDF, OCR text). Abstract/landing pages fetched: Hall 1988, FHS 2008, GNR 2020, Nagy et al. 2013, Erdil & Besiroglu 2022, Hernandez & Brown 2020, Pearce & Song 2024, Porian et al. 2024, Snell et al. 2024, AER page of LLMW 2010, Demirer website. Remaining bibliographic records verified via search-result metadata (journal, volume, pages) — see BibTeX notes.

---------------------------------------------------------------------------------------------------

## Appendix A. Reproduction script (sympy/numpy) for all [SYMPY]/[NUM] claims

Run with: `uv run --with sympy --with numpy python derive.py` (≈1 min). Output reproduced in Sections 0–2.

```python
import sympy as sp
import numpy as np

print("=" * 70)
print("(1) Elasticity of substitution of Chinchilla reducible loss")
N, D, A, B, a, b, E = sp.symbols('N D A B alpha beta E', positive=True)
R = A * N**(-a) + B * D**(-b)
f = -R  # output increasing in inputs (any monotone transform leaves isoquants unchanged)
fN, fD = sp.diff(f, N), sp.diff(f, D)
fNN, fDD, fND = sp.diff(f, N, 2), sp.diff(f, D, 2), sp.diff(f, N, D)
# Hicks direct elasticity for two inputs
sigma = -fN * fD * (N * fN + D * fD) / (N * D * (fNN * fD**2 - 2 * fND * fN * fD + fDD * fN**2))
sigma = sp.simplify(sigma)
u, v = sp.symbols('u v', positive=True)
sig_uv = sp.simplify(sigma.subs({A: u * N**a, B: v * D**b}))
print("sigma(u,v) =", sp.factor(sig_uv))
claimed = (a * u + b * v) / (a * u * (1 + b) + b * v * (1 + a))
print("matches claimed formula:", sp.simplify(sig_uv - claimed) == 0)
print("alpha=beta case:", sp.simplify(claimed.subs(b, a)))
# on the compute-optimal path alpha*u = beta*v
print("on path (alpha u = beta v):", sp.simplify(claimed.subs(u, b * v / a)))

# invariance to monotone transform of output: take Y = R^(-1/rho) etc.
k = sp.symbols('kappa', positive=True)
g = -(R**k)
gN, gD = sp.diff(g, N), sp.diff(g, D)
gNN, gDD, gND = sp.diff(g, N, 2), sp.diff(g, D, 2), sp.diff(g, N, D)
sig_g = -gN * gD * (N * gN + D * gD) / (N * D * (gNN * gD**2 - 2 * gND * gN * gD + gDD * gN**2))
print("sigma invariant to outer power transform kappa:", sp.simplify(sig_g - sigma) == 0)

# Allen (primal, bordered Hessian) elasticity for 2 inputs
H = sp.Matrix([[0, fN, fD], [fN, fNN, fND], [fD, fND, fDD]])
detH = H.det()
F12 = H.cofactor(1, 2)
allen = (N * fN + D * fD) / (N * D) * F12 / detH
print("Allen primal == Hicks direct (2 inputs):", sp.simplify(allen - sigma) == 0)

print("=" * 70)
print("(2) Cost function under C = 6 N D")
C = sp.symbols('C', positive=True)
G = (a * A / (b * B))**(1 / (a + b))
Nopt = G * (C / 6)**(b / (a + b))
Dopt = G**-1 * (C / 6)**(a / (a + b))
print("budget check 6 N D - C =", sp.simplify(6 * Nopt * Dopt - C))
foc = sp.simplify(a * A * Nopt**(-a) - b * B * Dopt**(-b))
print("FOC alpha*A*N^-alpha - beta*B*D^-beta at optimum =", sp.simplify(sp.powsimp(sp.expand_power_base(foc, force=True), force=True)))
Rstar = sp.simplify(A * Nopt**(-a) + B * Dopt**(-b))
gam = a * b / (a + b)
K = (a + b) / b * A * G**(-a)
print("R*(C) - K (C/6)^-gamma =", sp.simplify(sp.powsimp(Rstar - K * (C / 6)**(-gam), force=True)))
Ksym = (a + b) * (A / b)**(b / (a + b)) * (B / a)**(a / (a + b))
print("K symmetric form check:", sp.simplify(sp.powsimp(sp.expand_power_base(K - Ksym, force=True), force=True)))
# numerically verify K symmetric form
vals = {A: 406.4, B: 410.7, a: 0.34, b: 0.28}
print("K numeric:", float(K.subs(vals)), float(Ksym.subs(vals)))

print("--- Shephard's lemma in log-cost weights ---")
# generalized log-linear cost: lnC = th_N n + th_D d (+const). minimize subject to R(n,d)=Rbar
n, d, thN, thD, Rbar = sp.symbols('n d theta_N theta_D Rbar', positive=True)
lam = sp.symbols('lambda', positive=True)
Rn = A * sp.exp(-a * n) + B * sp.exp(-b * d)
# FOCs: th_N = lam * a A e^{-a n}, th_D = lam * b B e^{-b d}
# => A e^{-a n} = th_N/(lam a), B e^{-b d} = th_D/(lam b); constraint gives lam
lam_sol = sp.solve(sp.Eq(thN / (lam * a) + thD / (lam * b), Rbar), lam)[0]
n_sol = -sp.log(thN / (lam_sol * a * A)) / a
d_sol = -sp.log(thD / (lam_sol * b * B)) / b
V = thN * n_sol + thD * d_sol
print("dV/dth_N - n* =", sp.simplify(sp.diff(V, thN) - n_sol))
print("dV/dth_D - d* =", sp.simplify(sp.diff(V, thD) - d_sol))
print("symmetry dn*/dth_D - dd*/dth_N =", sp.simplify(sp.diff(n_sol, thD) - sp.diff(d_sol, thN)))
# compensated elasticity of log input ratio wrt log relative weights at th_N = th_D
x = d_sol - n_sol
el = sp.simplify(sp.diff(x, thN) * thN)  # d(d-n)/d ln th_N holding th_D, Rbar
print("d ln(D/N)/d ln(th_N) at th_N=th_D=1:", sp.simplify(el.subs({thN: 1, thD: 1})))
print("  compare sigma*/(1-sigma*) = 2/(alpha+beta):", sp.simplify(2 / (a + b)))
# Note: el should equal (1/a)*? check general
print("  general el:", sp.simplify(el))

print("=" * 70)
print("(3) Factor-augmenting technical change")
t, gN_, gD_ = sp.symbols('t g_N g_D', real=True)
At = A * sp.exp(-a * gN_ * t)
Bt = B * sp.exp(-b * gD_ * t)
Kt = Ksym.subs({A: At, B: Bt})
print("d ln K_t / dt =", sp.simplify(sp.diff(sp.log(Kt), t)))
print("  should be -gamma (g_N+g_D) =", sp.simplify(-gam * (gN_ + gD_)))
Gt = G.subs({A: At, B: Bt})
print("d ln G_t / dt =", sp.simplify(sp.diff(sp.log(Gt), t)))
print("d ln (D*/N*)/dt = -2 dlnG/dt =", sp.simplify(-2 * sp.diff(sp.log(Gt), t)))
MRTS = (a * At * N**(-a - 1)) / (b * Bt * D**(-b - 1))
print("Hicks bias d ln MRTS_ND/dt at fixed inputs =", sp.simplify(sp.diff(sp.log(MRTS), t)))
rho = sp.symbols('rho', positive=True)
print("  CES case (alpha=beta=rho): bias =", sp.simplify(sp.diff(sp.log(MRTS), t).subs({a: rho, b: rho})),
      " vs (1-1/sigma)(g_N-g_D) with sigma=1/(1+rho):", sp.simplify((1 - (1 + rho)) * (gN_ - gD_)))

print("=" * 70)
print("(4) DMR-type non-identification with outer exponent kappa")
a1, b1, kap = sp.symbols('a1 b1 kappa', positive=True)
# on-path observables: allocation exponent a_obs = b1/(a1+b1), loss exponent gam_obs = kappa a1 b1/(a1+b1)
aobs, gobs, S = sp.symbols('a_obs gamma_obs S', positive=True)
sol = {b1: aobs * S, a1: (1 - aobs) * S}
kap_sol = sp.solve(sp.Eq(kap * a1 * b1 / (a1 + b1), gobs).subs(sol), kap)[0]
print("kappa(S) =", sp.simplify(kap_sol), "; sigma* = 2/(2+S) spans (0,1) as S ranges over (0,inf)")
print("Under kappa=1 (Chinchilla): alpha = gamma/a_obs =", sp.simplify(gobs / aobs), "; beta = gamma/(1-a_obs) =", sp.simplify(gobs / (1 - aobs)))
# check: with kappa=1, a_obs = beta/(alpha+beta), gamma = alpha*beta/(alpha+beta)
print("   check with Hoffmann: a=", 0.28 / 0.62, "gamma=", 0.34 * 0.28 / 0.62, "-> alpha=gamma/a=", (0.34 * 0.28 / 0.62) / (0.28 / 0.62), "beta=gamma/(1-a)=", (0.34 * 0.28 / 0.62) / (1 - 0.28 / 0.62))

print("=" * 70)
print("(5) Numbers")
sets = {
    'Hoffmann2022 (rounded)': dict(E=1.69, A=406.4, B=410.7, al=0.34, be=0.28),
    'Hoffmann2022 (TeX, per Besiroglu)': dict(E=1.6934, A=406.4, B=410.7, al=0.3392, be=0.2849),
    'Besiroglu2024 replication': dict(E=1.8172, A=482.01, B=2085.43, al=0.3478, be=0.3658),
    'Muennighoff2023 C4': dict(E=1.87, A=521.0, B=1488.0, al=0.353, be=0.353),
}
for name, p in sets.items():
    al, be = p['al'], p['be']
    gm = al * be / (al + be)
    aa = be / (al + be)
    sig_path = 2 / (2 + al + be)
    sig_lo, sig_hi = 1 / (1 + max(al, be)), 1 / (1 + min(al, be))
    Gv = (al * p['A'] / (be * p['B']))**(1 / (al + be))
    Kv = (al + be) * (p['A'] / be)**(be / (al + be)) * (p['B'] / al)**(al / (al + be))
    # tokens per parameter at C = Chinchilla compute (6*70e9*1.4e12)
    Cc = 6 * 70e9 * 1.4e12
    No = Gv * (Cc / 6)**aa
    Do = (Cc / 6) / No
    print(f"{name}: a={aa:.3f}, b={1-aa:.3f}, gamma={gm:.4f}, dlnC/dlnR=-{1/gm:.2f}, sigma*={sig_path:.3f}, sigma range=[{sig_lo:.3f},{sig_hi:.3f}], "
          f"sigma*/(1-sigma*)={sig_path/(1-sig_path):.2f}, G={Gv:.4f}, K={Kv:.2f}, N*(C_chinchilla)={No:.3e}, D*={Do:.3e}, D/N={Do/No:.1f}")

print("Kaplan 2020 Table 2 inner exponents: N exponent alpha_N/alpha_D =", 0.076 / 0.103, ", D exponent 1; outer kappa = alpha_D=0.103")
aK, bK = 0.076 / 0.103, 1.0
print("  Kaplan sigma on path =", 2 / (2 + aK + bK), " range [", 1 / (1 + bK), ",", 1 / (1 + aK), "]")
print("  Kaplan L(N,D) compute exponent if C=6ND:", 0.103 * aK * bK / (aK + bK), "; allocation N ~ C^", bK / (aK + bK))

print("=" * 70)
print("(6) Inference wedge for over-trained models (Hoffmann rounded & Besiroglu)")
models = [('Chinchilla 70B/1.4T', 70e9, 1.4e12), ('GPT-3 175B/300B', 175e9, 300e9), ('Gopher 280B/300B', 280e9, 300e9),
          ('Llama-2 7B/2T', 7e9, 2e12), ('Llama-2 70B/2T', 70e9, 2e12), ('Llama-3 8B/15T', 8e9, 15e12), ('Llama-3 70B/15T', 70e9, 15e12)]
for name, p in [('Hoffmann', sets['Hoffmann2022 (rounded)']), ('Besiroglu', sets['Besiroglu2024 replication'])]:
    print(" params:", name)
    for mn, Nv, Dv in models:
        eN = p['al'] * p['A'] * Nv**(-p['al'])
        eD = p['be'] * p['B'] * Dv**(-p['be'])
        w = eN / eD
        # training-only optimum requires w=1; with inference tokens T: w = 1 + T/(3D)
        T = 3 * Dv * (w - 1)
        Lv = p['E'] + p['A'] * Nv**(-p['al']) + p['B'] * Dv**(-p['be'])
        # compute-equivalent inefficiency: FLOPs needed at optimum to reach same loss vs actual
        al, be = p['al'], p['be']
        gm = al * be / (al + be)
        Kv = (al + be) * (p['A'] / be)**(be / (al + be)) * (p['B'] / al)**(al / (al + be))
        Cmin = 6 * (Kv / (Lv - p['E']))**(1 / gm)
        Cact = 6 * Nv * Dv
        s_share = eN / (eN + eD)
        sig_loc = 1 / (1 + s_share * p['be'] + (1 - s_share) * p['al'])
        print(f"   {mn}: D/N={Dv/Nv:.0f}, wedge eps_N/eps_D={w:.3f}, implied lifetime inference tokens T={T:.3e} ({T/Dv:.2f} x D), "
              f"L={Lv:.3f}, C_actual/C_min={Cact/Cmin:.3f}, local sigma={sig_loc:.3f}")

print("=" * 70)
print("(7) Normalization (KMW) and parameter correlation on a Chinchilla-like design")
rng = np.random.default_rng(0)
Ns = np.exp(np.linspace(np.log(7e7), np.log(1.6e10), 12))
Cs = np.exp(np.linspace(np.log(6e18), np.log(3e21), 9))
pts = []
for Cv in Cs:
    for Nv in Ns:
        Dv = Cv / (6 * Nv)
        if 5e9 <= Dv <= 5e11:
            pts.append((Nv, Dv))
pts = np.array(pts)
print(" design points:", len(pts))
def jac(theta, Nn, Dn):
    Ev, lA, lB, al, be = theta
    tA = np.exp(lA) * Nn**(-al)
    tB = np.exp(lB) * Dn**(-be)
    L = Ev + tA + tB
    # derivative of log L
    J = np.column_stack([1 / L, tA / L, tB / L, -np.log(Nn) * tA / L, -np.log(Dn) * tB / L])
    return J
p = sets['Besiroglu2024 replication']
theta_raw = np.array([p['E'], np.log(p['A']), np.log(p['B']), p['al'], p['be']])
J = jac(theta_raw, pts[:, 0], pts[:, 1])
cov = np.linalg.inv(J.T @ J)
sd = np.sqrt(np.diag(cov)); corr = cov / np.outer(sd, sd)
names = ['E', 'lnA', 'lnB', 'alpha', 'beta']
print(" raw parametrization corr(lnA,alpha)=%.4f corr(lnB,beta)=%.4f corr(alpha,beta)=%.4f corr(E,alpha)=%.3f corr(E,beta)=%.3f" % (corr[1, 3], corr[2, 4], corr[3, 4], corr[0, 3], corr[0, 4]))
print(" cond number raw:", np.linalg.cond(J))
N0 = np.exp(np.mean(np.log(pts[:, 0]))); D0 = np.exp(np.mean(np.log(pts[:, 1])))
theta_norm = np.array([p['E'], np.log(p['A']) - p['al'] * np.log(N0), np.log(p['B']) - p['be'] * np.log(D0), p['al'], p['be']])
Jn = jac(theta_norm, pts[:, 0] / N0, pts[:, 1] / D0)
covn = np.linalg.inv(Jn.T @ Jn)
sdn = np.sqrt(np.diag(covn)); corrn = covn / np.outer(sdn, sdn)
print(" normalized (geo-mean) corr(ln a,alpha)=%.4f corr(ln b,beta)=%.4f corr(alpha,beta)=%.4f corr(E,alpha)=%.3f corr(E,beta)=%.3f" % (corrn[1, 3], corrn[2, 4], corrn[3, 4], corrn[0, 3], corrn[0, 4]))
print(" cond number normalized:", np.linalg.cond(Jn))
print(" relative sd of lnA vs ln a (same sigma^2):", sd[1], sdn[1])

# on-path only design: collinearity
Cpath = np.exp(np.linspace(np.log(1e18), np.log(1e24), 30))
aa = p['be'] / (p['al'] + p['be'])
Gv = (p['al'] * p['A'] / (p['be'] * p['B']))**(1 / (p['al'] + p['be']))
Np = Gv * (Cpath / 6)**aa
Dp = Cpath / (6 * Np)
Jp = jac(theta_raw, Np, Dp)
print(" on-path-only design: singular values", np.linalg.svd(Jp, compute_uv=False))
print(" on-path-only cond number:", np.linalg.cond(Jp))

print("=" * 70)
print("(8) Three-input additively separable power law: frontier exponent & Allen elasticities")
x1, x2, x3, c1, c2, c3, e1, e2, e3 = sp.symbols('x1 x2 x3 c1 c2 c3 e1 e2 e3', positive=True)
F = -(c1 * x1**(-e1) + c2 * x2**(-e2) + c3 * x3**(-e3))
X = [x1, x2, x3]
fx = [sp.diff(F, xi) for xi in X]
Hb = sp.zeros(4, 4)
for i in range(3):
    Hb[0, i + 1] = fx[i]; Hb[i + 1, 0] = fx[i]
    for j in range(3):
        Hb[i + 1, j + 1] = sp.diff(F, X[i], X[j])
dH = sp.simplify(Hb.det())
num = sum(X[i] * fx[i] for i in range(3))
def AES(i, j):
    return sp.simplify(num / (X[i] * X[j]) * Hb.cofactor(i + 1, j + 1) / dH)
s12, s13, s23 = AES(0, 1), AES(0, 2), AES(1, 2)
print(" AES12 == AES13 ?", sp.simplify(s12 - s13) == 0, "; AES12 == AES23 ?", sp.simplify(s12 - s23) == 0)
# numeric illustration
subsn = {c1: 1, c2: 1, c3: 1, e1: 0.3, e2: 0.4, e3: 0.5, x1: 1.3, x2: 0.7, x3: 2.1}
print(" numeric AES12, AES13, AES23:", float(s12.subs(subsn)), float(s13.subs(subsn)), float(s23.subs(subsn)))
# frontier exponent with C = x1 x2 x3
print(" frontier exponent 1/sum(1/e_i) e.g. Kaplan (1.8): alpha_C_min = 1/(1/0.76 + 1/0.21 + 1/0.076) =", 1 / (1 / 0.76 + 1 / 0.21 + 1 / 0.076))

print("=" * 70)
print("(9) Kmenta approximation of CES (alpha=beta=rho) reducible loss")
eps = sp.symbols('epsilon')
sA = sp.symbols('s_A', positive=True)
expr = sp.log(sA * sp.exp(-eps * n) + (1 - sA) * sp.exp(-eps * d))
ser = sp.series(expr, eps, 0, 3).removeO()
print(" ln(s_A e^{-rho n} + (1-s_A) e^{-rho d}) ≈", sp.factor(sp.simplify(ser)))
print(" check equals -rho(s_A n + (1-s_A) d) + rho^2/2 s_A(1-s_A)(n-d)^2 :",
      sp.simplify(ser - (-eps * (sA * n + (1 - sA) * d) + eps**2 / 2 * sA * (1 - sA) * (n - d)**2)) == 0)

print("=" * 70)
print("(10) Jones (2021) train/test isoquant: log(test) = -1.2 log(train) + 0.004 Elo + 29")
print(" implied Cobb-Douglas: Elo = 250 log10(test) + 300 log10(train) + const; output-elasticity ratio train:test = 1.2; sigma=1")
print(" cost-min share of train in (train+test) compute under linear cost = 1.2/2.2 =", 1.2 / 2.2)
```

Rank-one Hessian check:

```python
import sympy as sp, numpy as np, csv
n,d,A,B,a,b=sp.symbols('n d A B alpha beta',positive=True)
lnR=sp.log(A*sp.exp(-a*n)+B*sp.exp(-b*d))
H=sp.hessian(lnR,(n,d))
print("det Hessian of lnR in (ln N, ln D):",sp.simplify(H.det()))
s=A*sp.exp(-a*n)/(A*sp.exp(-a*n)+B*sp.exp(-b*d))
print("H11 - alpha^2 s(1-s):",sp.simplify(H[0,0]-a**2*s*(1-s)),"; H12 + alpha beta s(1-s):",sp.simplify(H[0,1]+a*b*s*(1-s)))
# null direction
print("H*(beta,alpha):",sp.simplify(H*sp.Matrix([b,a])))
```
