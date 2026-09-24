# Strand notes: IO production-function estimation (endogeneity, proxy/control-function and panel methods, markups) and what each implies for neural scaling laws

Project: "Scaling Laws as Production Functions" (AER-style). Strand owner file. BibTeX: `lit/bib/io_controlfn.bib`.
Compiled 2026-09-23. Every factual claim about a paper is tagged with how it was verified:
**[read]** = text of paper (or official working-paper version) extracted and read this session;
**[abs]** = abstract / publisher / NBER / arXiv landing page or search-result metadata seen this session;
**[recalled]** = standard textbook knowledge NOT re-verified this session (treat as UNVERIFIED; check before the paper cites specifics).
Derivations marked **[derived]** are ours (this strand), not claims of any cited paper.

---------------------------------------------------------------------------------------------------

## 0. Conventions and the translation template

### 0.1 IO canonical model (the object all methods below share)
Firm `i`, period `t`, logs lower case:
`y_it = f(k_it, l_it, m_it; β) + ω_it + ε_it`
- `ω_it`: productivity known to the firm when (some) inputs are chosen ("transmitted"; Marschak–Andrews) → source of simultaneity/transmission bias.
- `ε_it`: ex-post shock / output measurement error, not in the firm's information set when inputs are chosen ("non-transmitted").
- `ω` Markov: `ω_it = h(ω_{i,t-1}) + ξ_it`, `E[ξ_it | I_{i,t-1}] = 0` (exogenous) or controlled `g(ω_{i,t-1}, r_{i,t-1})` (Doraszelski–Jaumandreu).
- Timing: `k_it` predetermined (chosen at t−1 via investment), `l_it` variable (or chosen at t−b), `m_it` fully flexible (chosen at t after observing ω_it).
- Identification comes from (i) timing/information-set restrictions (lagged inputs orthogonal to ξ), (ii) proxy inversion (a monotone policy function reveals ω), (iii) first-order conditions (cost/revenue shares), (iv) excluded input-price/cost shifters, or (v) fixed effects/panel structure.

### 0.2 Scaling-law translation used throughout [derived]
Unit `j` = one training run (a model), belonging to lab/model family `f`, generation/date `t`.
Inputs: parameters `N_j` (n = log N), training tokens `D_j` (d = log D), training compute `C_j ≈ 6 N_j D_j` (so `c = log 6 + n + d`, a *deterministic multiplicative* cost identity, unlike IO's additive `Σ p_x x`). Optional: data quality/mix `q`, architecture `a`, post-training compute, distillation/teacher tokens, inference compute.
Output: validation loss `L_j` (a "bad"), preferably bits-per-byte on a common corpus; or benchmark accuracy / Elo (bounded or demand-side transforms).
Technology (Chinchilla-type with Hicks-neutral lab productivity on the reducible loss):
`L_j = E + exp(−ω_j) · [A N_j^{−α} + B D_j^{−β}] · exp(ε_j)`            (T1)
equivalently, with reducible loss `R = L − E`:
`y_j ≡ −log(L_j − E) = F(n_j, d_j) + ω_j − ε_j`, `F(n,d) = −log(A e^{−αn} + B e^{−βd})`.   (T1')
- `ω_j` ↔ recipe/data/architecture quality known to the lab at design time (lab "TFP", Mundlak's "management").
- `ε_j` ↔ seed noise, training instabilities, evaluation noise: not known at design (ZKD's disturbance).
- Output elasticities of the reducible loss: `θ_N ≡ ∂y/∂n = αA N^{−α}/R`, `θ_D ≡ ∂y/∂d = βB D^{−β}/R`; scale elasticity `θ_N + θ_D`.
- NB: E enters non-log-additively; most IO estimators need log-additive ω. With E known/estimated (asymptote), (T1') is an IO-style log production function with a flexible (non-Cobb–Douglas) F and Hicks-neutral ω. If one term dominates, F is locally Cobb–Douglas (Kaplan-style single power laws).

### 0.3 Verified ML anchor facts used in the analog paragraphs
- **Hoffmann et al. 2022 (Chinchilla)** [read, arXiv 2203.15556]: >400 LMs, 70M–16B+ params, 5–500B tokens (abstract). Eq. (2): `L̂(N,D) = E + A/N^α + B/D^β`, fit by Huber loss (δ=10^{-3}) on log loss with L-BFGS over a grid of initialisations (Eq. 3; App. D.2 Eq. 11 uses log-sum-exp). App. D.2 Eq. (10): `E=1.69, A=406.4, B=410.7, α=0.34, β=0.28` (more precise TeX-source values reported by Besiroglu et al.: E=1.6934, A=406.4, B=410.7, α=0.3392, β=0.2849). Table 2 (`N_opt ∝ C^a`, `D_opt ∝ C^b`, 10th–90th pct bootstrap): Approach 1 (min over training curves) a=0.50 (0.488,0.502), b=0.50 (0.501,0.512); Approach 2 (IsoFLOP profiles) a=0.49 (0.462,0.534), b=0.51 (0.483,0.529); Approach 3 (parametric) a=0.46 (0.454,0.455), b=0.54 (0.542,0.543); Kaplan a=0.73, b=0.27. Approach 3 down-weights low-compute runs via Huber loss and "predicts even smaller models being optimal"; negative curvature of the frontier noted (App. E).
- **Besiroglu, Erdil, Barnett, You 2024** [read, arXiv 2404.10102]: re-fit on 240 points reconstructed from Chinchilla Fig. 4: `L = 1.8172 + 482.01/N^0.3478 + 2085.43/D^0.3658`; bootstrap SEs: A 124.58, B 1293.23, E 0.03, α 0.02, β 0.02; a = β/(α+β) = 0.5126 (SE 0.018). Hoffmann's reported CI for a (0.454–0.455) is ~50× too narrow; would require ~600,000 runs; lead author later attributed it to averaging (not summing) Huber losses → early L-BFGS termination. Hoffmann's Approach-3 parameters imply ≈70 tokens/parameter, inconsistent with Chinchilla's own ≈20; Besiroglu's imply ≈20 (point estimate 25.6 in one place).
- **Kaplan et al. 2020** [read, arXiv 2001.08361]: Eq. (1.5) `L(N,D) = [(N_c/N)^{α_N/α_D} + D_c/D]^{α_D}`; single-input fits α_N≈0.076 (N_c≈8.8e13 non-embedding params), α_D≈0.095; joint fit α_N=0.076, α_D=0.103; compute-efficient `N ∝ C_min^{0.73}`, `D ∝ C^{0.27}`; `L ∝ C_min^{−0.050}`.
- **Pearce & Song 2024** [abs, arXiv 2406.12907, TMLR]: Kaplan-vs-Chinchilla discrepancy largely due to Kaplan counting *non-embedding* rather than total parameters, at small scale → an **input measurement** story.
- **Porian et al. 2024** [abs, arXiv 2406.19146, NeurIPS]: discrepancy explained by last-layer compute cost, warmup duration, scale-dependent optimizer tuning; after correction Kaplan-style runs agree with Chinchilla.
- **Sardana, Portes, Doubov, Frankle 2024** [read, arXiv 2401.00448, ICML 2024]: `N*, D_tr* = argmin_{N,D_tr | L(N,D_tr)=ℓ} 6 N D_tr + 2 N D_inf` (Eq. 3); inference demand ~1B requests ⇒ train smaller and longer than Chinchilla-optimal; 47 MPT models 150M–6B at 10–10,000 tokens/param; loss keeps improving up to 10,000 tokens/param; fitting Chinchilla coefficients only on typical ratios overestimates the value of extra tokens at extreme ratios.
- **Ho et al. 2024** [read, arXiv 2403.05812]: >200 LM evaluations (WikiText, Penn Treebank) 2012–2023; effective inputs `N_eff = N exp(α'(Y−Y0))`, `D_eff = D exp(β'(Y−Y0))` (Eq. 2) ⇒ `L = E + A N^{−α_param} e^{−α_year(Y−Y0)} + B D^{−β_data} e^{−β_year(Y−Y0)}` (Eq. 3), α' = α_year/α_param, β' = β_year/β_data; compute needed for fixed performance halves every ~8 months (95% CI ~5–14; preferred model 8.4 months, CI 4.5–14.3).
- **Llama 3 (Grattafiori et al. 2024)** [read, arXiv 2407.21783]: flagship 405B params, 15.6T tokens, 3.8×10^25 FLOPs; IsoFLOP experiments 6×10^18–10^22 FLOPs, fit (for optimal *tokens*) `N*(C) = A C^α` with (α, A) = (0.53, 0.29); extrapolation to 3.8e25 → 402B params on 16.55T tokens; "we also train our smaller models for much longer than is compute-optimal"; flagship used to improve smaller models in post-training (distillation-like intermediate input); two-stage downstream prediction: FLOPs → normalized NLL of correct answer → accuracy via a sigmoid; BF16 MFU 38–43%. (Per-model token counts for 8B/70B not verified here; corpus "about 15T" tokens.)
- **Muennighoff et al. 2023** [abs, arXiv 2305.16264]: up to 4 epochs of repeated data ≈ unique data at fixed compute; diminishing returns beyond; 400 runs released.
- **Ruan, Maddison, Hashimoto 2024** [abs, arXiv 2405.10938, NeurIPS]: ~100 public models; performance is a function of a low-dimensional capability space; *families differ only in how efficiently they convert compute to capabilities* (= family-specific Hicks-neutral TFP in IO language).
- **Gadre et al. 2024** [abs, arXiv 2403.08540]: 104 models (0.011B–6.9B), over-training scaling laws, perplexity→downstream power law.
- **Biderman et al. 2023 (Pythia)** [abs, arXiv 2304.01373]: 16 LLMs 70M–12B trained on public data in the exact same order, 154 checkpoints each.

### 0.4 Derived frontier algebra used below [derived]
Training-only cost minimization `min 6ND s.t. L(N,D)=ℓ`. Because `log C = log 6 + n + d`, both inputs have unit *cost* elasticity, so the FOC is the ML "share equation":
**`θ_N = θ_D`** on the compute-optimal frontier (output elasticities equal).                                   (F1)
For Chinchilla form: `αA N^{−α} = βB D^{−β}` ⇒ `N* = G (C/6)^a`, `D* = G^{−1}(C/6)^b`, `a = β/(α+β)`, `b = α/(α+β)`, `G = (αA/βB)^{1/(α+β)}`;
`L*(C) = E + K (C/6)^{−γ}`, `K = A G^{−α} + B G^{β}`, **`γ = αβ/(α+β)`**, and on the frontier `θ_N = θ_D = γ` (the scale elasticity of reducible loss w.r.t. compute).  (F2)
Numbers (computed this session from the verified coefficients):
| parameter set | a | γ | tokens/param at C=5.76e23 | at C=3.8e25 |
|---|---|---|---|---|
| Hoffmann Eq.4 precise (0.3392, 0.2849) | 0.457 | 0.155 | 59 | 85 |
| Besiroglu et al. (0.3478, 0.3658) | 0.513 | 0.178 | 18 | 17 |
With α=β=ρ the reducible part is an exact CES aggregator: `Q ≡ R^{−1/ρ} = (A N^{−ρ} + B D^{−ρ})^{−1/ρ}`, CRS in (N,D), **σ = 1/(1+ρ)** (σ≈0.75 for ρ≈0.34; 0.73–0.78 over the verified α, β) — gross complements. With α≠β, σ varies along the isoquant.
Hicks-neutral ω in (T1) scales both terms by e^{−ω}, so **the expansion path (N/D mix at given C) is ω-invariant**; ω only moves a lab along the common path. [derived]

---------------------------------------------------------------------------------------------------

## 1. Early simultaneity, fixed effects and the behavioral model of input choice

### 1.1 Marschak & Andrews (1944), Econometrica 12(3/4):143–205 [abs; content via ZKD 1966 and ACF WP, read]
**Statement.** Output and inputs are jointly determined by the production function plus the firm's optimality conditions ("random simultaneous equations"). In ZKD's restatement of the "traditional" (Marschak–Andrews/Klein/Hoch) model with Cobb–Douglas `X = A L^{α1} K^{α2}`: `x0i − α1 x1i − α2 x2i = α0 + v0i` (production), `x0i − x1i = λ1 + v1i`, `x0i − x2i = λ2 + v2i` (FOCs, λ's functions of common prices and Hoch's systematic-error parameters R1, R2). Since each input depends on all disturbances, including the "technical efficiency" disturbance v0i (which, in Marschak–Andrews' words as quoted by ZKD, depends on the entrepreneur's technical knowledge, will, effort and luck), OLS of output on inputs is biased and inconsistent ("transmission bias"). Marschak–Andrews also allow that entrepreneurs differ in "urge, or ability, or luck" in choosing profitable input combinations (quoted in ACF WP fn. 3) — i.e., optimization error, which later becomes the only thing that can identify labor in LP (ACF).
**Identification.** Requires restrictions on the covariance of (v0, v1, v2) or on the behavioral model; with common prices, cross-sectional input variation arises only from disturbances.
**Scaling-law analog.** Cross-lab regressions of loss on (N, D) or C (e.g., Ho et al. 2024's >200 published evaluations; Epoch "notable models") are exactly Marschak–Andrews data: inputs are chosen by labs that know their recipe quality ω_f. The sign of the bias depends on the behavioral rule [derived]: with `r ≡ log(L−E) = −ω + log K − γ c + ε` along the (ω-invariant) path,
- (R1) *capability-target* regime (lab trains until a target ℓ̄ is hit): `c = const − ω/γ` → high-ω labs use less compute; OLS slope on c is `−γ + γ = 0` when all labs share the same target (complete attenuation), partial attenuation otherwise → **returns to compute understated**;
- (R2) *exogenous budget* (C set by funding/hardware unrelated to ω): OLS consistent (ZKD case);
- (R3) *funding responds to ω* (`c = const + λω`): OLS slope `−γ − λ Var(ω)/Var(c)` → **returns to compute overstated** (classic upward transmission bias).
Because Hicks-neutral ω does not tilt the expansion path (§0.4), transmission bias falls on the *scale* elasticity γ, not on the N/D split; only factor-biased ω (e.g., data quality augmenting D) biases the split.

### 1.2 Hoch (1958), Econometrica 26(4):566–578; Hoch (1962), Econometrica 30(1):34–53 [abs]
**Statement.** 1958: with a Cobb–Douglas and competitive firm decision functions, two extreme disturbance models: in one no simultaneous-equation bias, in the other bias occurs; LS elasticity sums tend toward one "regardless of the true value of the elasticities" [abs]. 1962: estimate CD parameters by analysis of covariance on combined time-series/cross-section farm data with firm-specific and time-specific constants; moving from LS to covariance lowers elasticity sums [abs].
**Scaling-law analog.** Two-way effects = lab (or model family) × vintage constants: `y_j = F(n_j,d_j) + η_f + λ_t + ε_j`. Within a family-generation (e.g., the Llama 3 herd 8B/70B/405B trained with one recipe), ω_ft is common across sizes, so size variation within family-generation identifies F free of lab TFP (multi-plant firm with common TFP). Hoch's "elasticity sums fall under covariance" warns that within-family variation is narrow (mostly N at ≈fixed D), so returns-to-scale estimates from within-family data can be attenuated by noise (see Griliches–Mairesse, §1.5).

### 1.3 Mundlak (1961), J. Farm Econ. 43(1):44–56 ("Empirical Production Function Free of Management Bias") [abs; content recalled + ACF WP]
**Statement.** `y_it = x_it'β + m_i + u_it`, with time-invariant "management" m_i known to the farmer and correlated with inputs → OLS "management bias". Covariance (within/fixed-effects) estimation removes m_i. ACF WP: FE requires `ω_it = ω_{i,t−1} ∀t` ("a strong assumption and in some cases can result in worse estimates than OLS", citing Griliches–Hausman 1986) [read].
**Critiques.** Removes all between variation; amplifies measurement error; fails with time-varying productivity; typically yields implausibly low capital coefficients and returns to scale (Griliches–Mairesse [recalled]).
**Scaling-law analog.** "Management bias" ↔ "recipe bias". Family/lab FE absorb persistent recipe quality (data pipeline, tokenizer, architecture tricks). ω is clearly *not* time-invariant in ML (algorithmic progress ~8-month halving; Ho et al.), so FE must be at lab×generation level, which leaves only within-generation size variation (§1.2). FE with mismeasured D (undisclosed or estimated token counts; repetition; tokenizer units) will attenuate θ_D badly.

### 1.4 Zellner, Kmenta & Drèze (1966), Econometrica 34(4):784–795 [read]
**Statement.** Replace deterministic profit maximization by maximization of *expected* profit with a stochastic production function `X_i = A L_i^{α1} K_i^{α2} e^{u0i}`; if u0i is realized after input choice (not in the firm's information set), inputs are independent of u0i and **classical least squares is consistent** (and ML/unbiased under normality) [read, §1 and §4]. They note that interpreting v0i as firm-specific "technical efficiency" (A varying across firms) reintroduces correlation; the variance components (permanent vs transitory) need panel data.
**Scaling-law analog.** This is the cleanest justification for OLS/NLS fits of scaling laws *inside a controlled sweep*: in an IsoFLOP or size sweep run by one lab with one recipe, ω is fixed by design and the residual (seed noise, eval noise) is realized after N and D are set → ZKD conditions hold → NLS consistent. It fails for pooled cross-lab data where the residual contains ω_f known at design. The ZKD/ω-vs-ε split maps to "recipe (known) vs run noise (unknown)".

### 1.5 Griliches & Mairesse (1998), "Production Functions: The Search for Identification", in Strøm (ed.), Econometrics and Economic Theory in the 20th Century (Ragnar Frisch Centennial Symposium), CUP, ch. 6, pp. 169–203 [abs; Cambridge lists the volume online with 1999 date — see bib note]
**Statement (survey).** Traces production-function econometrics from Cobb–Douglas to micro panels; persistent failure to find credible identification: simultaneity, selection, measurement error; FE/within estimates give low capital coefficients and decreasing returns (errors-in-variables amplified by differencing); input prices rarely usable instruments (little variation, quality contamination). Summarized by ACF and GNR as the reference statement of transmission bias [read in ACF/GNR].
**Scaling-law analog.** The ML literature is at the pre-OP stage of this history: pooled NLS fits of parametric laws with little attention to what variation identifies which parameter. GM's diagnosis — "the more you difference, the more you estimate measurement error" — applies to within-family estimation with poorly measured D and C.

### 1.6 Nerlove (1963), "Returns to Scale in Electricity Supply", in Christ (ed.), Measurement in Economics, Stanford UP, pp. 167–198 [abs for citation; content recalled]
**Statement (recalled).** Estimate the *cost function* `log C = f(log Q, log input prices)` for US electric utilities (1955 cross-section) instead of the production function: utilities must meet demand at regulated prices, so output is exogenous and inputs are chosen by cost minimization; duality recovers technology (returns to scale declining with output). GNR cite Nerlove (1963) as an early user of FOC cross-equation restrictions [read in GNR §6.2].
**Scaling-law analog.** The compute-optimal frontier `L*(C)` is the inverse cost function; `C*(ℓ) = 6 (K/(ℓ−E))^{1/γ}` is the cost function [derived from F2]. Nerlove's identification logic transfers directly: *which side is exogenous?* If labs choose capability targets (R1 regime), the right regression is cost-on-output (`c` on `log(ℓ−E)`), and the conventional regression of loss on compute is the "reverse regression" (attenuated); with noise, the two regressions bracket the truth [derived]. Chinchilla's Approach 1/2 estimate the expansion path (a, b) = the input-demand side of the dual.

---------------------------------------------------------------------------------------------------

## 2. Dynamic panel (internal-instrument) methods

### 2.1 Arellano & Bond (1991), REStud 58(2):277–297 [abs; details recalled]
**Statement (recalled).** `y_it = ρ y_{i,t−1} + x_it'β + η_i + v_it`; first-difference away η_i; with serially uncorrelated v, moments `E[y_{i,t−s} Δv_it] = 0`, `s ≥ 2` (and analogous for predetermined/endogenous x). Specification tests: m1/m2 tests for AR(1)/AR(2) in differenced residuals, Sargan overidentification test [abs: "a test of serial correlation based on the GMM residuals ... compare with Sargan tests ... and Hausman"].
**Scaling-law analog.** Lab panels by generation (Llama 1→2→3, GPT, Qwen, Gemma...) with lab effect η_f + AR(1) recipe shock. Very short T (3–5 generations), few labs (tens) → asymptotics in N_labs are poor; lagged levels are weak instruments for differences when input growth is explosive.

### 2.2 Blundell & Bond (1998), J. Econometrics 87(1):115–143; Blundell & Bond (2000), Econometric Reviews 19(3):321–340 [2000: read (IFS WP 99/04 version); 1998: abs]
**Statement (BB 2000, read).** `y_it = β_n n_it + β_k k_it + γ_t + (η_i + v_it + m_it)`, `v_it = ρ v_{i,t−1} + e_it`, `|ρ|<1`, `e, m ~ MA(0)` (2.1). Dynamic common-factor form (2.2)–(2.3): `y_it = π1 n_it + π2 n_{i,t−1} + π3 k_it + π4 k_{i,t−1} + π5 y_{i,t−1} + γ*_t + (η*_i + w_it)` with COMFAC restrictions `π2 = −π1π5`, `π4 = −π3π5` imposed by minimum distance. DIF-GMM uses lagged levels as instruments for first differences; SYS-GMM adds levels equations with lagged differences as instruments, `E[Δx_{i,t−s}(η*_i + w_it)] = 0` for s=1 (MA(0)) or s=2 (MA(1)) (3.6), valid under a mean-stationarity/initial-conditions restriction (3.10)/(3.12) (first moments of (n,k,y) time-invariant conditional on year dummies).
**Data/estimates (BB 2000, Table 2, restricted model, one-step GMM):** balanced panel of 509 R&D-performing US manufacturing firms, 1982–89 (sales as output).
| | OLS levels | Within | DIF t−2 | DIF t−3 | SYS t−2 | SYS t−3 |
|---|---|---|---|---|---|---|
| β_n | .538 | .488 | .583 | .515 | .773 | .479 |
| β_k | .266 | .199 | .062 | .225 | .231 | .492 |
| ρ | .964 | .512 | .377 | .448 | .509 | .565 |
| CRS p-value | .000 | .000 | .000 | .006 | .922 | .641 |
DIF-GMM estimates sit close to Within (weak instruments bias 2SLS toward OLS-on-orthogonal-deviations = Within); t−2 instruments rejected (measurement error); SYS t−3 accepted, CRS accepted. Series highly persistent (AR ≈ 0.9 for capital and sales).
**BB 1998 [abs]:** introduced the system estimator; large finite-sample bias of DIF-GMM with persistent series, removed by the extra level moments under stationary initial conditions.
**Scaling-law analog.** (i) The "stationarity of initial conditions" assumption is the one that most clearly **breaks down** for ML: compute, N and D grow by orders of magnitude across generations and the correlation between lab effects and input levels is not time-invariant. SYS-GMM level moments are not credible. (ii) The COMFAC idea is useful: an AR(1) recipe process implies testable restrictions linking current and lagged family losses and inputs. (iii) See Ackerberg et al. 2023 (§6.3): persistence of regressors creates multiple roots in BB-type moments — acute with near-deterministic compute growth.

---------------------------------------------------------------------------------------------------

## 3. Proxy / control-function methods

### 3.1 Olley & Pakes (1996), Econometrica 64(6):1263–1297 [read]
**Model.** Value added `y_it = β0 + β_a a_it + β_k k_it + β_l l_it + ω_it + η_it` (6). Capital accumulates with investment chosen at t−1; labor variable. Dynamic program ⇒ exit rule `χ_t = 1 iff ω_t ≥ ω̲_t(a_t, k_t)` (4) and investment `i_t = i_t(ω_t, a_t, k_t)` (5), strictly increasing in ω for i>0 (Pakes 1994 Thm 27) ⇒ `ω_t = h_t(i_t, a_t, k_t)` (7).
**Stage 1** (partially linear, Robinson 1988): `y_it = β_l l_it + φ_t(i_it, a_it, k_it) + η_it` (8)–(9) identifies β_l (4th-order polynomial; separate polynomials by regulatory period).
**Selection.** Survival probability `P_t = Pr{χ_{t+1}=1 | ω̲_{t+1}(k_{t+1},a_{t+1}), J_t} = p_t(i_t, a_t, k_t)` (10) (probit/kernel on polynomial in (i,a,k)).
**Stage 2**: `y_{t+1} − β_l l_{t+1} = β_a a_{t+1} + β_k k_{t+1} + g(P_t, φ_t − β_a a_t − β_k k_t) + ξ_{t+1} + η_{t+1}` (12); k_{t+1} mean-independent of ξ_{t+1} by timing.
**Selection bias mechanism.** Firms with larger k continue at lower ω, so `E[ω_t | k_t, survive]` is decreasing in k ⇒ **negative bias on the capital coefficient** in balanced panels.
**Results (Table VI; US telecommunications-equipment plants, regulatory periods 1974–86):** balanced panel OLS (l .851, k .173), Within (.728, .067); full sample OLS (.693, .304), Within (.629, .150); OP labor .608 (.027), capital .342 (.035) series / .355 (.058) kernel. Going from balanced to full sample more than doubles β_k; OP's β_k is twice the full-sample Within.
**Assumptions.** Scalar unobservable (only ω in the investment rule), strict monotonicity, k chosen exactly at t−1, labor non-dynamic, common input prices (absorbed in t-index).
**Scaling-law analog.** (i) *Proxy:* the ML analog of investment is the lab's *next* scale-up / research-compute decision, which depends on current recipe quality: a lab that sees a good small-scale result invests in a bigger run. OP-style inversion of "scale-up investment" is conceivable with lab-level panels but monotonicity is doubtful (lumpy, strategic, funding-constrained — exactly LP's criticism of investment). (ii) *Timing:* N (architecture) is fixed before training and hard to change mid-run → the "capital" role; D (train length) can be extended/cut after observing early loss curves → the "variable input" role [derived]. So the OP/ACF capital moment `E[ξ | n] = 0` is plausible for within-run innovations. (iii) *Selection:* the ML analog is **publication/notability selection**, not exit: Epoch's "notable" database includes a model only if it is SOTA on a recognized benchmark, has >1,000 citations, historical relevance, or >1M monthly users [abs, epoch.ai]. Selection is on the *outcome*, so `E[ω | N, included]` depends on N. Conjecture (sign theory-dependent): small models are cheap and published only when unusually good, large models are published regardless ⇒ E[ω|N, included] decreasing in N ⇒ the scaling exponent is *understated* in notability-selected samples — the exact OP sign pattern. OP's propensity-score correction (condition on P̂ = Pr(included | observables)) is implementable if inclusion rules are known. Also: Chinchilla Approach 1 takes the *minimum* over training curves at each FLOP level — an envelope (frontier) estimator that selects favorable (ω, ε) draws; with more runs per compute level, the envelope is mechanically lower (extreme-value selection) [derived].

### 3.2 Levinsohn & Petrin (2003), REStud 70(2):317–341 [abs; model read via ACF WP]
**Model.** Gross output `y_it = β1 k_it + β2 l_it + β3 m_it + ω_it + ε_it`; intermediate input demand `m_it = f_t(ω_it, k_it)` monotone ⇒ `ω_it = f_t^{-1}(m_it, k_it)`. Stage 1: `y = β2 l + Φ_t(m,k) + ε` identifies β2 (claimed). Stage 2 moments: `E[ξ_it(β1,β3) | k_it] = 0`, `E[ξ_it(β1,β3) | m_{i,t−1}] = 0`. Motivation: investment is lumpy with many zeros (monotonicity fails), intermediates are smooth.
**Critiques.** ACF functional dependence (next); Bond–Söderbom (2005); GNR (2020) show the gross-output version is not identified without extra variation.
**Scaling-law analog — is "D given N" an LP proxy?** [derived] With N predetermined (the "state") and ω Hicks-neutral:
- capability-target regime: D solves L(N,D;ω)=ℓ̄ ⇒ D strictly *decreasing* in ω (better recipe needs fewer tokens) → invertible, `ω = f^{-1}(d, n)`;
- marginal rule "train until marginal loss reduction = marginal cost": `−∂L/∂D = e^{−ω} βB D^{−β−1}` is decreasing in ω ⇒ D decreasing in ω (unless value of loss reduction is convex near capability thresholds) → invertible, direction model-dependent (either direction suffices for a control function);
- fixed-budget regime with N fixed: D = C/(6N) carries no information about ω → proxy fails.
Scalar-unobservability is the weak point: token budgets also respond to data availability (unique-token limits; Muennighoff), deadlines, hardware outages — i.e., "optimization error" in the proxy (use Hu–Huang–Sasaki, §5.3). Other candidate proxies: post-training (SFT/RL) compute, number of annealing tokens, distillation tokens — chosen after the base model's quality is realized; monotonicity sign unclear (compensation vs complementarity).

### 3.3 Ackerberg, Caves & Frazer (2015), Econometrica 83(6):2411–2451 [abs]; ACF (2005) WP "Structural Identification of Production Functions" [read]
**Functional dependence — exact statement (WP §3.1, read).** In LP, if l_it is chosen at the same time as m_it, labor has its own demand `l_it = f_{2t}(ω_it, k_it)`; substituting the inverted proxy, `l_it = f_{2t}(f_t^{-1}(m_it,k_it), k_it) = g_t(m_it, k_it)`, so l_it is a *deterministic function* of the arguments of the nonparametric first-stage term and **β_l is not identified** in `y = β1 k + β2 l + β3 m + f_t^{-1}(m,k) + ε`. Observing common input prices does not help (they enter both demands). If l is chosen before m, then `m_it = f_t(ω_it, k_it, l_it)` and l enters the control function → again unidentified. If l is chosen after m and ω moves in between, the inversion of m no longer controls for ω. Only two stories save LP: (a) optimization error in l but *none* in m (identification "purely a function of the extent of this optimization error"); (b) l chosen after m with ω constant but an i.i.d., firm-specific, non-persistent labor-price shock in between. Measurement error in l does not help (β_l → 0). A *parametric* first stage does not rescue CD: with price-taking, the FOC for m makes `y = ln(1/β3) + ln(p_m/p_y) + m + ε` (11), β2 drops out; same with Leontief-in-materials (13).
OP is less exposed: if l is chosen at t−b (0<b<1) with ω evolving between t−1, t−b, t, then `l_it = f_t(ω_{i,t−b}, k_it)` varies independently of `f_t^{-1}(i,k)`; this works for OP but not LP because m (unlike i) would depend on l.
**ACF estimator (WP §5; published version same logic).** Stage 1 identifies no coefficients: `y_it = φ_t(m_it, k_it, l_it) + ε_it` (only purges ε). Stage 2: `ω_it(β) = φ̂_it − β_k k_it − β_l l_it`, `ξ_it(β) = ω_it(β) − E[ω_it | ω_{i,t−1}(β)]` (nonparametric regression on lagged ω(β)); moments **`E[ξ_it(β) | k_it] = 0`** (19) and **`E[ξ_it(β) | l_{i,t−1}] = 0`** (20). Robust to l chosen at t−b, dynamic labor (`φ_t(m,k,l_{t−1})`, moments (24) `E[ξ | l_{t−1}, k_t]=0`). Gross-output variant with `E[ξ | k_t, l_{t−1}, m_{t−1}] = 0` noted (fn. 7) — later shown problematic by GNR. A lagged-labor instrument requires serially correlated input prices (DLW 2012 fn.).
**Bond & Söderbom (2005), IFS WP 05/04 [abs]:** CD parameters are not identified from cross-section variation when inputs are perfectly flexible, optimally chosen, and input prices are common; adjustment costs restore identification — fragile if deterministic, more useful if stochastic.
**Scaling-law analog (focus a) — see §7.A for the full proposition.** The compute-optimal expansion path is the ML version of `l = g_t(m,k)`: on-path runs have `n = a c + κ_n`, `d = (1−a) c + κ_d` — both inputs are affine functions of one state c, so (n, d) are perfectly collinear (functional dependence). Chinchilla's Approaches 1–2 break this with **designed off-path variation** (IsoFLOP sweeps: many N at fixed C) — the ML analog of the "i.i.d. firm-specific price shock" or "stochastic adjustment cost" that ACF/Bond–Söderbom need, but here created by the experimenter. Observational cross-lab data do *not* have it: with Hicks-neutral ω all labs share the same path (§0.4), so cross-lab heterogeneity does not break functional dependence; only factor-biased ω, lab-specific cost shifters (inference demand, data constraints, hardware), or optimization errors/belief differences (Kaplan-era vs Chinchilla-era allocations) do.

### 3.4 Wooldridge (2009), Economics Letters 104(3):112–114 [abs; equations recalled — verify numbering]
**Statement.** Implement OP/LP moments jointly as one-step GMM on two equations with the same dependent variable and different instrument sets:
(i) `y_it = α + β_l l_it + β_k k_it + g(k_it, m_it) + e_it`, `E[e_it | l_it, k_it, m_it, and all lags] = 0`;
(ii) `y_it = α + β_l l_it + β_k k_it + f(g(k_{i,t−1}, m_{i,t−1})) + u_it`, `E[u_it | k_it, l_{i,t−1}, k_{i,t−1}, m_{i,t−1}, ...] = 0` (l_it instrumented by l_{i,t−1}).
Benefits: efficiency via cross-equation restrictions, correct GMM standard errors (no bootstrap), robust to ACF critique because β_l is identified from (ii). Cost: joint search over polynomial coefficients (DLW fn. 20 [read]).
**Scaling-law analog.** A natural one-step NLS/GMM for families: equation (i) within a family-generation (sizes share ω_ft), equation (ii) across generations with lagged-generation inputs as instruments. Gives proper standard errors — relevant given Besiroglu et al.'s finding that Chinchilla's CIs were ~50× too narrow.

### 3.5 Ackerberg, Benkard, Berry & Pakes (2007), Handbook of Econometrics 6A, ch. 63, pp. 4171–4276 [abs]
**Content.** Survey: demand estimation, production functions (OP, LP, dynamic panel, selection, timing and information sets), dynamic parameters. Central for (a) the "timing and information set" language, (b) caution that input prices are rarely valid instruments (GNR §6.1 cites ABBP and GM for this [read]).
**Scaling-law analog.** Template for the paper's econometrics section: specify for every input *when* it is chosen relative to which shock and *what* the lab knows.

---------------------------------------------------------------------------------------------------

## 4. Gross output, first-order conditions and endogenous productivity

### 4.1 Gandhi, Navarro & Rivers (2020), JPE 128(8):2973–3016 [read, authors' final full PDF]
**Setup.** A1: `y_jt = f(k_jt, l_jt, m_jt) + ν_jt`, f differentiable, strictly concave in m; ν = ω + ε. A2: ω known at t, Markov `ω_jt = h(ω_{j,t−1}) + η_jt`; ε independent of I_jt (so `E ≡ E[e^{ε}]` constant). A3 (proxy): `m_jt = M_t(k_jt, l_jt, ω_jt)` strictly monotone, scalar unobservable. A4: price-taking in output and intermediate markets, common prices ρ_t, P_t. FOC (4): `P_t ∂F/∂M e^{ω} E = ρ_t` ⇒ `m_jt = M(k,l,ω − d_t)`, `d_t = ln(ρ_t/P_t) − ln E`.
**Theorem 1 (non-identification).** First stage (6) `E[y | k,l,m,d_t] = φ(k,l,m) + d_t` identifies no elasticities. Second stage (7)/(9) uses `E[η + ε | Γ_jt] = 0` with Γ = (k_t, l_t, lagged y,k,l,m, d_{t−1}, ...). By (8), `m_jt = M(k, l, h(M^{-1}(k_{t−1},l_{t−1},m_{t−1}) + d_{t−1}) + η − d_t)`, so after conditioning on predetermined variables the only variation left in m is η (orthogonal by construction) and d_t. **Without time-series variation in relative prices (d_t = d), there is a continuum of observationally equivalent** `f̃ = (1−a) f0 + a φ`, `h̃(x) = a d + (1−a) h0((x − a d)/(1−a))`, `a ∈ (0,1)`. Extends to dynamic panel (Online App. O1).
**Monte Carlo 1.** CD with k 0.25, m 0.65; AR(1) ω with ρ 0.8; AR(1) log intermediate price with coefficient 0.6 and baseline innovation variance 0.0001 (Chile, Food 311); 200/500 firms, T ∈ {3,...,50}: time-series price variation gives badly biased estimates even with T=50 unless variation is 10× baseline.
**Theorem 2 (share regression).** Log FOC minus production function: `s_jt ≡ ln(ρ_t M_jt / P_t Y_jt) = ln E + ln D(k,l,m) − ε_jt` (11); `E[s | k,l,m] = ln D^E(k,l,m)` (12) identifies the flexible-input elasticity `D = D^E/E` (13)–(14) nonparametrically, using only ε-moments.
**Theorem 3 (integrate up).** `∫ ∂f/∂m dm = f + C(k,l)` (15); `𝒴_jt ≡ y − ε − ∫D dm = −C(k,l) + ω` (16); `E[𝒴_jt | k_t, l_t, 𝒴_{t−1}, k_{t−1}, l_{t−1}] = −C(k_t,l_t) + h(𝒴_{t−1} + C(k_{t−1},l_{t−1}))` (18) identifies C up to a constant under a support condition (A5). Estimation: degree-2 polynomials for D^E and C, cubic h; bootstrap SEs.
**Results.** Colombian and Chilean plants: OLS vs GNR elasticities differ materially (paper reports an average difference of 34% — attributed to transmission bias); OLS understates productivity dispersion: "All industries" 95/5 ratio 21% (Colombia) and 16% (Chile) larger under GNR; productivity persistence corr 0.64 (GNR) vs 0.53 (OLS); importer premia 8%/13% (GNR) vs 1%/6% (OLS). §6.2: FOC-based identification goes back to Marschak–Andrews, Klein, Solow, Nerlove; nonparametric generalization of Griliches–Ringstad (1971); DJ (2013, 2018) and GLZ (2016) use parametric FOC restrictions with observed prices. GNR also cite a companion WP comparing gross output and value added (Gandhi, Navarro, Rivers 2017, "How heterogeneous is productivity?" — not independently verified).
**Scaling-law analog (focus b) — see §7.B.** Two separate lessons. (1) *Non-identification without FOC information:* if an input (D, post-training compute, distillation tokens) is chosen flexibly with ω known and no lab-specific cost shifter moves it, observational loss/input data plus Markov timing cannot separate its elasticity from ω — the ML literature's pooled NLS fits implicitly rely on functional form. (2) *The FOC fix is unusually clean in ML*: because `log C = log 6 + n + d`, the training-cost "shares" are known by accounting (each input's cost elasticity = 1), so the FOC `θ_N = θ_D` (or `θ_N/θ_D = 1 + D_inf/(3D)` with inference; §7.C) is the exact analog of GNR's share equation — *but* it holds only if labs optimize against the true technology. Pre-2022 allocations optimized against Kaplan's (mismeasured) frontier (N ∝ C^0.73) violate it — a documented case of systematic "optimization error" (belief shock) that biases FOC-based estimators but supplies the off-path variation that breaks functional dependence. (3) *Gross output vs value added:* distillation and synthetic data are intermediate inputs embodying teacher compute (Llama 3 uses the 405B to improve smaller models); a "value-added-like" productivity measure that ignores teacher compute overstates the TFP of distilled students.

### 4.2 Doraszelski & Jaumandreu (2013), REStud 80(4):1338–1383 [read]
**Model.** CD gross output `y_jt = β0 + β_t t + β_k k_jt + β_l l_jt + β_m m_jt + ω_jt + e_jt` (1); capital `K_jt = (1−δ)K_{j,t−1} + I_{j,t−1}`; Bellman (2) with state `s_jt = (t, k, ω, w, p_M, d)` and cost functions `C_i(i)`, `C_r(r)` for physical and knowledge investment. **Controlled first-order Markov process** `P(ω_jt | ω_{j,t−1}, r_{j,t−1})` ⇒ `ω_jt = g(ω_{j,t−1}, r_{j,t−1}) + ξ_jt` (3): the firm anticipates only the *expected* effect of R&D; ξ bundles chance in discovery, applicability, implementation.
**Parametric inversion via the labor FOC** (static labor and materials, monopolistic competition with demand elasticity η(p,d)): labor demand (4); inverse labor demand `h_l(·) = λ_l − β_t t − β_k k + (1−β_l−β_m) l + (1−β_m)(w−p) + β_m(p_M−p) − ln(1 − 1/η(p,d))` (5) (closed forms exist for CES and generalized linear, not translog; requires Hicks neutrality).
**Estimating equation** `y_jt = β0 + β_t t + β_k k + β_l l + β_m m + g(h_{l,j,t−1}, r_{j,t−1}) + ξ_jt + e_jt` (6); moments `E[A(z_jt)(ξ_jt + e_jt)] = 0` with z = lagged variables incl. lagged prices and r_{t−1}; `g = 1(R_{t−1}=0)[g00 + g01(h−λ)] + 1(R_{t−1}>0)[g10 + g11(h−λ, r)]`, cubic polynomials, |η| = 1+exp(q(p,d)); 27 parameters; NL2SLS then optimal two-step GMM. Parametric inversion avoids the curse of dimensionality of nonparametric OP/LP/ACF inversion when prices are firm-specific. Zero R&D in 43–83% of observations ⇒ OP-type inversion of R&D impossible.
**Data/findings.** ESEE, unbalanced panel of >1,800 Spanish manufacturing firms, nine industries, 1990s. Non-linearity: complementarity between current productivity and R&D in most industries; uncertainty: **25–75% of the variance of productivity is explained by innovations unpredictable when R&D is chosen**; return to R&D often twice that of physical capital; performers' expected-productivity distribution stochastically dominates non-performers'; performers account for 65–90% of productivity growth in intermediate/high-innovation industries; higher elasticities w.r.t. R&D and lower persistence than the knowledge-capital model (Griliches 1979), which is rejected in most industries.
**Scaling-law analog (focus d) — see §7.D.** Algorithmic progress as endogenous, lab-specific, uncertain productivity: `ω_ft = g(ω_{f,t−1}, r_{f,t−1}, s_{t−1}) + ξ_ft`, r = research inputs (experiment compute, researchers), s = spillovers from the public frontier. Contrast: Ho et al. treat algorithmic progress as a common deterministic exponential trend (an exogenous, non-controlled process) — DJ show such averaging "blurs important differences" between performers and non-performers.

### 4.3 Doraszelski & Jaumandreu (2018), JPE 126(3):1027–1084 [abs]
**Statement.** Measure the bias of technological change at the firm level with multidimensional productivity: CES with labor-augmenting and Hicks-neutral components, identified from FOCs for labor and materials with observed firm-level prices (GNR §6.2 [read]); Spain: labor-augmenting and factor-neutral components each cause output to grow ≈1.5%/yr [abs].
**Scaling-law analog.** Ho et al.'s `N_eff = N e^{α'(Y−Y0)}`, `D_eff = D e^{β'(Y−Y0)}` is *literally* factor-augmenting technical change. [derived] Hicks-neutral progress on the reducible loss ⇔ equal time-decay rates of the two terms in Ho et al. Eq. (3): **`α_year = β_year`** (then `e^{−g t}(A N^{−α} + B D^{−β})`). This is a one-line test of neutrality. DJ 2018's identification uses FOC/price variation; the ML analog of the FOC is (F1): under data-augmenting progress the optimal tokens-per-parameter ratio drifts over time, under Hicks-neutral progress it does not.

### 4.4 De Loecker (2011), Econometrica 79(5):1407–1451; De Loecker (2013), AEJ: Micro 5(3):1–21; De Loecker, Goldberg, Khandelwal & Pavcnik (2016), Econometrica 84(2):445–510 [abs]
**Statements.** 2011: combine a demand system with the production function to separate price, scale and productivity effects when only revenue is observed; multiproduct firms; Belgian textiles and trade liberalization [abs]. 2013: standard methods assume ω evolves exogenously; allowing the Markov process to depend on export status (learning by exporting) reveals substantial productivity gains on export entry (Slovenia) [abs]. 2016: markups and marginal costs from production data with quantity and price information for multiproduct firms, without assumptions on market structure or input allocation across products; India's trade liberalization lowered factory-gate prices; pro-competitive output-tariff effects [abs].
**Scaling-law analog.** (i) *Learning by deploying*: user interaction/feedback data from deployed models feed the next generation's recipe — a controlled Markov process `g(ω_{t−1}, deployed_{t−1})`; ignoring it, as in 2013, both mis-states the technology and hides the effect. (ii) Revenue-based "output" (API revenue, Arena Elo) mixes demand and technology; De Loecker 2011's demand-system correction is the template if one uses such outputs. (iii) Multiproduct: one pretraining run feeds several products (base, chat, code, distilled students) — input allocation across products unobserved, as in DGKP.

---------------------------------------------------------------------------------------------------

## 5. Markups from production data and their critiques

### 5.1 De Loecker & Warzynski (2012), AER 102(6):2437–2471 [read, NBER WP 15198 revised Oct 2010]
**Derivation.** Cost-minimizing producer, `Q = Q(X^1..X^V, K)`, Lagrangian (1) `L = Σ P^X X + r K + λ(Q − Q(·))`; FOC for a variable input free of adjustment costs (2) `P^X − λ ∂Q/∂X = 0`; λ = marginal cost; rearranging (3): output elasticity `θ^X = (1/λ) P^X X / Q`. Define markup `μ = P/λ` ⇒
**`μ_it = θ^X_it (α^X_it)^{-1}`** (4), where `α^X = P^X X / (P Q)` is the expenditure share of input X in *sales*. Needs: cost minimization, one flexible input, price-taking in that input, an estimate of θ^X. θ^X from a translog `y = β_l l + β_k k + β_ll l² + β_kk k² + β_lk l k + ω + ε` (6) estimated ACF-style: stage 1 `y = φ_t(l,k,m) + ε` (7) (materials proxy, no coefficients identified), stage 2 moments (10) `E[ξ_it(β) · (l_{t−1}, k_t, l²_{t−1}, k²_t, l_{t−1}k_t)'] = 0`; θ^L = β_l + 2β_ll l + β_lk k (11). Correct the observed share for ε: use `Q/exp(ε̂)`. Lagged labor is a valid instrument only if wages are serially correlated (verified in their data). Monotonicity of m in ω under imperfect competition requires more productive firms not to have "inordinately" higher markups (Melitz–Levinsohn).
**Data/results.** All Slovenian manufacturing firms 1994–2000. WP Table 2 (median markups): Hall-type growth regression 1.03 (0.004), Klette algorithm 1.12 (0.020); DLW specifications I–VI: 1.17, 1.10, 1.23, 1.28, 1.23, 1.26 (within-spec SD ≈ 0.5); common-markup variants 1.16 (0.006) and 1.11 (0.007, first differences). Markups higher when controlling for productivity (first-differenced Hall/Klette estimators biased down by measurement error); exporters charge higher markups; markups rise on export entry. DLW fn. 28 cites Bond–Söderbom on the difficulty of identifying a freely chosen input's coefficient under Cobb–Douglas.
**Scaling-law analog (focus c) — see §7.C.** In ML there is no revenue share; the analog of (3) is the cost-minimization FOC with the *multiplicative* compute identity. Over-training for inference is an input-specific wedge on N, recoverable as `θ_N/θ_D = 1 + D_inf/(3D)` — structurally closer to Hsieh–Klenow's input distortion (1+τ_K) than to DLW's output markup, but estimated exactly à la DLW: (estimated output elasticities) ÷ (observed cost elasticities).

### 5.2 De Loecker, Eeckhout & Unger (2020), QJE 135(2):561–644 [abs]
**Statement.** DLW applied to US public firms (Compustat) since 1955 with cost of goods sold as the variable input; aggregate markups rise from 21% above marginal cost in 1980 to 61%; average profit rate from 1% to 8% [abs].
**Scaling-law analog.** A direct (non-analogical) application: markups on AI inference APIs using inference compute as the flexible input (price per token vs 2N FLOPs/token × $/FLOP ÷ utilization). Not the core of this paper but a natural extension.

### 5.3 Bond, Hashemi, Kaplan & Zoch (2021), JME 121:1–14 [abs, NBER w27002]
**Statement.** The ratio estimator (θ^X/α^X) requires the *output* elasticity; when only revenue is observed and a revenue elasticity is substituted, "the estimand underlying the ratio estimator does not contain any information about the markup"; further issues when inputs are used to shift demand, when inputs are neither fully fixed nor flexible, and consistent elasticity estimation is impossible without quantity data under market power and heterogeneous markups.
**Scaling-law analog.** Output-concept discipline: elasticities of *benchmark accuracy* or *Arena Elo* (demand/preference-based, bounded) with respect to compute are not technology parameters. Any wedge calculation (§7.C) must use elasticities of a quantity-like output (bits-per-byte on a fixed corpus), never of Elo or accuracy.

### 5.4 Raval (2023), REStud 90(5):2592–2611 [abs] (task list had 2592–2627; publisher/RePEc say 2592–2611); Raval (2019), RAND 50(1):147–167 [abs]
**Statement (2023).** Under the production approach, markups computed from any two flexible inputs must have the same distribution. In manufacturing data from Chile, Colombia, India, Indonesia, the US and Southern Europe, plus store-level data from a US retailer, this is overwhelmingly rejected: labor-based and materials-based markups are *negatively* correlated, labor-based ones more dispersed, with opposite time trends [abs]. (Attribution of the failure to non-neutral, labor-augmenting technology — recalled, verify.) **2019:** micro elasticity of substitution with non-neutral technology [abs; estimates not verified].
**Scaling-law analog.** An over-identification test. With ≥3 "flexible" inputs (N, D, plus post-training compute or data-quality spend), wedges computed from different input pairs must agree under Hicks-neutral technology; disagreement signals factor-biased recipes. Within a family (8B/70B/405B trained with one recipe) the implied D_inf must be consistent with one demand model — a Raval-style test of the scaling-law parameters themselves (see §7.C, the Chinchilla self-consistency check).

### 5.5 Flynn, Gandhi & Traina (2019), "Measuring Markups with Production Data", SSRN 3358472 [abs]
**Statement.** Standard production-function methods do not identify markups; non-identification creates spurious skewness in markup distributions; ex-ante structure on returns to scale solves it; CRS "performs remarkably well" and halves skewness among US public firms [abs].
**Scaling-law analog.** Cost-min FOCs identify only *ratios* of elasticities; levels need the scale elasticity. In ML the scale elasticity is directly observable from the frontier: `d log R*/d log C = −γ` (F2). Frontier returns-to-scale + FOC ⇒ `θ_N = θ_D = γ` — the ML version of FGT's "impose RTS" fix, except RTS is estimable rather than assumed.

### 5.6 Klette & Griliches (1996), J. Applied Econometrics 11(4):343–361 [abs]
**Statement.** When output is deflated sales with a common deflator and firms have heterogeneous prices under imperfect competition, the unobserved firm-specific price enters the error and is correlated with inputs ⇒ scale estimators are inconsistent (downward biased) [abs]. (Remedy via industry demand/output term — recalled.)
**Scaling-law analog.** *Per-token loss is a "deflated" output with a common deflator that is not common*: tokenizers differ in bytes/token, so per-token cross-entropy and token counts D are in lab-specific units. If larger labs/models use larger vocabularies (more bytes per token), per-token comparisons bias the estimated returns to scale — Klette–Griliches in ML. Fix: measure both output and D in bytes (bits-per-byte; bytes of text).

### 5.7 Foster, Haltiwanger & Syverson (2008), AER 98(1):394–425 [read, working-paper version]
**Statement.** Eleven homogeneous products (e.g., ready-mixed concrete), Census of Manufactures 1977–1997 (quinquennial), plant-level prices and physical quantities. TFPQ (physical) vs TFPR (revenue = price × TFPQ): both disperse by >20 log points within product-years; TFPQ is *more* dispersed than TFPR because TFPQ and price are strongly negatively correlated (efficient plants charge less); young plants charge lower prices, so revenue-based measures understate entrants' productivity advantage and entry's contribution; selection is on profitability, not just efficiency.
**Scaling-law analog.** Bits-per-byte on a fixed corpus ≈ TFPQ; benchmark accuracy (bounded, sigmoidal in log-likelihood — Llama 3's two-stage method) and Elo/revenue ≈ TFPR-like transforms confounded by demand, style, contamination and ceiling effects. "Emergence" in accuracy can be a TFPR artifact of a smooth TFPQ process. Build the production function on TFPQ-like loss; treat accuracy as a known monotone measurement map.

---------------------------------------------------------------------------------------------------

## 6. Measurement error, input prices, timing, heterogeneity

### 6.1 Collard-Wexler & De Loecker (2016), NBER WP 22437 (rev. 2020) [abs]
**Statement.** Capital measurement error + simultaneity: instrument the mismeasured capital stock with lagged investment while conditioning on persistent productivity; nests ACF. Monte Carlo: 40% capital measurement error halves the capital coefficient; China, India, Chile: capital coefficients ≈ twice those from simultaneity-only methods [abs].
**Scaling-law analog.** ML inputs are measured with error in ways that correlate with scale: (i) N: total vs non-embedding parameters (Pearce & Song: this drove Kaplan's 0.73), MoE total vs active parameters, tied embeddings; (ii) D: tokens vs unique tokens (repetition), tokenizer units, undisclosed counts imputed by trackers; (iii) C: often *imputed* as 6ND (so C inherits N and D errors), or reported GPU-hours whose effective FLOPs depend on utilization (Llama 3 MFU 38–43%). Instruments in CWDL's spirit: hardware counts/cluster size × training days ("investment") for effective compute; architecture hyperparameters (d_model, layers) for N.

### 6.2 Grieco, Li & Zhang (2016), IER 57(2):665–690 [abs]
**Statement.** When intermediate quantities are unobserved and input prices disperse, deflated expenditure as a quantity proxy biases estimates; use FOCs for labor and intermediates (with observed wages) to recover unobserved input prices and productivity; finds significant input-price dispersion and even wider productivity dispersion than proxy methods [abs; GNR §6.2 read].
**Scaling-law analog.** Training "compute" is frequently measured as *dollars* (cost estimates) — expenditure = price × quantity with large dispersion in $/effective-FLOP across labs (owned TPUs vs rented GPUs; hardware generation; utilization). Productivity measured per dollar confounds recipe quality with compute prices; FOC-based recovery of lab-specific compute prices is the GLZ analog.

### 6.3 Hu, Huang & Sasaki (2020), J. Econometrics 215(2):375–398 [abs]
**Statement.** Relax the scalar-unobservable assumption on proxies; identification with **two conditionally independent proxy variables**; robust to optimization errors, idiosyncratic cost shocks, measurement error in proxies, and to the functional dependence problem; GMM; Chilean food-products plants [abs].
**Scaling-law analog.** Multiple noisy indicators of the same latent lab/model quality: loss on two disjoint held-out corpora, or two benchmark families; Ruan et al.'s low-dimensional capability space is a measurement model of the same type. Two conditionally independent evals (independent eval noise) allow deconvolution of ω from ε.

### 6.4 Ackerberg (2023), J. Industrial Economics 71(3):644–674 [abs]; Ackerberg, Frazer, Kim, Luo & Su (2023), J. Econometrics 237(1) [abs; arXiv 2303.15170]
**Statements.** 2023 JIE: proxy and dynamic-panel methods both rest on timing and information-set assumptions that can be strengthened or weakened almost continuously; empirically, strengthening/weakening them trades off efficiency across plant-level datasets [abs]. AFKLS: under-identification in Blundell–Bond-type models — moment conditions can have *multiple discrete solutions*, one at the main-equation persistence parameter and one at the regressor's persistence parameter; conditions under which it disappears; sign restrictions as remedy [abs].
**Scaling-law analog.** Explicit timing table for ML [derived]:
| input | when chosen | knows ω_f? | knows within-run shock? | IO role |
|---|---|---|---|---|
| N (width/depth) | design | yes | no | predetermined "capital" |
| D (train length, anneal) | design, adjustable mid-run | yes | partly (loss curve) | variable/flexible |
| data mix/quality | design (+annealing) | yes | partly | factor-augmenting ω or input |
| post-training compute, distillation | after pretraining | yes | yes | intermediate / proxy candidate |
| inference/test-time compute | deployment | yes | yes | output-side flexible input |
Weakest defensible assumption: `E[ξ | n] = 0` (N set before within-run surprises). AFKLS's multiplicity warning is acute because lab inputs grow almost deterministically (regressor persistence ≈ 1).

### 6.5 Kasahara, Schrimpf & Suzuki (2023), arXiv 2305.12067 [abs]
**Statement.** Nonparametric identification of production functions with heterogeneity beyond Hicks-neutral terms via a finite mixture of latent technology types; each type's production function identified with four periods of panel data; Japanese plants show large differences in input elasticities and productivity-growth processes across latent types within narrow industries; ignoring elasticity heterogeneity biases productivity-growth estimates substantially and systematically [abs].
**Scaling-law analog.** Latent architecture/recipe types (dense vs MoE vs state-space; tokenizer regimes) with type-specific exponents. Ho et al. impose common exponents across all models; KSS predict that this biases the algorithmic-progress rate if exponents differ across types. Finite-mixture scaling laws are a concrete methodological import.

---------------------------------------------------------------------------------------------------

## 7. Special-focus derivations [derived unless tagged]

### 7.A ACF functional dependence vs the Chinchilla compute-optimal frontier
**Proposition A1 (on-path collinearity).** If a sample contains only runs on one expansion path, `n_j = a c_j + κ_n`, `d_j = (1−a) c_j + κ_d` (with `κ_n + κ_d = −log 6`), then (n, d) is an affine function of the scalar c: rank-one design, exactly ACF's `l_it = g_t(m_it, k_it)`.
**Proposition A2 (what on-path data identify).** Nonparametrically, only the restriction `ℓ(c) = L(n(c), d(c))` is identified; the MRTS and all off-path losses are not. Parametrically (T1), `ℓ(c) = E + A' e^{−αa c} + B' e^{−β(1−a) c}` with `A' = A e^{−ακ_n}`, `B' = B e^{−βκ_d}`.
- If the path is *not* optimal (αa ≠ β(1−a)), ℓ(c) is a sum of two exponentials in c with distinct rates: (α, β, A, B) are identified only through that curvature — weakly when rates are close.
- If the path *is* optimal, αa = β(1−a) = γ and ℓ(c) = E + K e^{−γ c}: γ and hence α = γ/a, β = γ/(1−a) are pinned (a observed), but **A and B are not separately identified**: any (Ã, B̃) with `Ã G^{−α} + B̃ G^{β} = K` fits. The level of the MRTS — hence the optimal tokens-per-parameter ratio `D*/N* = G^{−2}(C/6)^{b−a}` — is identified only by *imposing* the FOC `G^{α+β} = αA/(βB)`, i.e., by assuming optimality.
**Corollary.** *The better labs optimize, the less their production data reveal about the technology* (the ACF/Bond–Söderbom insight). Without off-path variation, "the compute-optimal allocation is assumed, not estimated," and optimality is untestable.
**What breaks it.** (i) IsoFLOP sweeps (Chinchilla Approach 2; Llama 3 IsoFLOPs 6e18–1e22): designed variation in n at fixed c = experimenter-created "i.i.d. price shocks"; (ii) cross-lab cost shifters: inference demand (Sardana), unique-data constraints (Muennighoff), hardware/export constraints — valid only if orthogonal to the recipe innovation (DJ/GNR §6.1 logic on price instruments); (iii) belief shocks: the Kaplan→Chinchilla revision moved allocations independently of ω (a natural experiment in the input mix); (iv) factor-biased ω (moves the path, but is itself confounding).
**Empirical fingerprints.** Hoffmann's Approach-3 a = 0.46 vs Approaches 1–2 a ≈ 0.49–0.50 and Besiroglu's a = 0.513 (SE 0.018): the parametric fit's MRTS level is the fragile object, exactly as A2 predicts; A (482 ± 125) and B (2085 ± 1293) have huge SEs in Besiroglu's refit while α, β have SE 0.02 — the A/B split is weakly identified relative to the exponents.

### 7.B GNR non-identification for scaling laws, and the FOC/share fix
**Setting.** Observational panel of labs; N predetermined; D (or post-training compute) flexible, chosen with ω_f known; labs face common relative costs (training-only 6ND), no lab-specific cost shifters.
**Proposition B1 (non-identification; GNR Thm 1 analog).** Under Hicks-neutral ω, the flexible input's demand is `d = M(n, ω; ℓ̄ or budget)`; conditional on predetermined variables, the only residual variation in d is the recipe innovation ξ (orthogonal to all instruments by construction). Hence Markov/lag moments cannot separate θ_D from the ω-component: a continuum of `(F̃, h̃)` fits (mixtures of the true F and the first-stage φ). Pooled NLS fits of (T1) on observational data identify θ_D only via functional form.
**Proposition B2 (share-equation identification; GNR Thm 2 analog).** Under cost minimization, `θ_D/θ_N = (∂log C/∂d)/(∂log C/∂n)`; with `C = 6ND`, both cost elasticities are 1, so `θ_D = θ_N` at every chosen point — an *exact accounting share*, no price data needed. With observed N-elasticity (e.g., from within-family size variation at fixed D) the D-elasticity follows. Integrating up (GNR Thm 3) recovers F up to a function of the predetermined input, then Markov moments recover the rest.
**Caveat (why FOCs are riskier in ML than in manufacturing).** The FOC must hold w.r.t. the *true* technology. Documented violations: Kaplan-era allocations (N ∝ C^0.73) were optimal only for a mismeasured technology (non-embedding counts, small scale, warmup/tuning artifacts — Pearce & Song; Porian et al.); inference-aware over-training (Sardana; Llama 3 smaller models) adds a known wedge. So use B2 only with the wedge modeled (§7.C) and for post-2022 vintages; use pre-2022 vintages as off-path variation instead.
**Returns to scale.** FGT's "impose CRS" becomes "estimate γ from the frontier": along the frontier `θ_N = θ_D = γ` ≈ 0.155 (Hoffmann precise) or 0.178 (Besiroglu).

### 7.C DLW-style "markup" for the inference-cost wedge of over-trained models
**Derivation.** Lab minimizes lifetime compute `C_tot = 6 N D + 2 N D_inf` (Sardana Eq. 3) s.t. `L(N,D) = ℓ`. Cost elasticities: `∂log C_tot/∂n = 1`, `∂log C_tot/∂d = 6ND/(6ND + 2N D_inf) = [1 + D_inf/(3D)]^{−1}`. FOC (DLW eq. 3 analog: output elasticity ∝ cost elasticity):
**`μ̂_j ≡ θ_N,j / θ_D,j = 1 + D_inf,j / (3 D_j)`  ⇒  `D̂_inf,j = 3 D_j (θ̂_N,j/θ̂_D,j − 1)`**   (W)
with, for (T1), `θ_N/θ_D = (αA N^{−α}) / (βB D^{−β})`.
**Properties.** (1) Hicks-neutral lab TFP (e^{−ω}) and the irreducible loss E *cancel* — the estimator is robust to recipe-quality differences across labs, exactly as DLW is robust to Hicks-neutral ω given common technology parameters (DLW eq. 5 `Q = F(X;β)exp(ω)`). (2) It is *not* robust to factor-biased recipes (data quality changes B or β) — the Raval (2023) caveat. (3) It requires the MRTS level (A/B) — the object §7.A shows is weakly identified from on-path data — so the wedge must use experimentally identified (IsoFLOP-based) technology. (4) Interpretation: the wedge is input-specific (each parameter carries a shadow inference cost 2D_inf FLOPs), i.e., a Hsieh–Klenow τ on N, not an output markup; a common output wedge (all θ_x/s_x equal) is not identified without a revenue measure. (5) With real costs, D_inf is price-weighted: `D_inf^{eff} = D_inf × (p_inf/p_train)` (inference hardware price/utilization vs training).
**Illustration (computed this session; D per model for Llama 3 8B/70B assumed ≈15T — per-model counts not verified).**
| model (N, D) | θ_N/θ_D (Hoffmann precise) | implied D_inf/D | θ_N/θ_D (Besiroglu) | implied D_inf/D |
|---|---|---|---|---|
| 8B @ 15T | 2.92 | 5.8 (8.7e13 tok) | 5.22 | 12.7 (1.9e14 tok) |
| 70B @ 15T | 1.40 | 1.2 | 2.45 | 4.4 |
| 405B @ 15.6T | 0.78 | −0.7 (infeasible) | 1.35 | 1.1 |
| Chinchilla 70B @ 1.4T | 0.71 | −0.9 (infeasible) | 1.03 | 0.1 |
**Revealed-preference specification test.** Chinchilla-70B was chosen by its authors as compute-optimal (training-only), so (W) should give ≈1. Hoffmann's own Approach-3 parameters give 0.71 (implying *negative* inference demand, i.e., the model is "under-trained" relative to those parameters), Besiroglu's give 1.03. The FOC thus independently rejects Hoffmann's Approach-3 MRTS and corroborates Besiroglu et al. — a DLW/Raval-style use of choices to test estimated technology. Sensitivity: implied D_inf for the 8B differs by ≈2.2× across the two parameter sets — the wedge is only as good as the identified MRTS.
**Data needed.** Open-weight families with disclosed (N, D) (Llama, Qwen, Gemma, OLMo, Pythia), per-family IsoFLOP/size sweeps or a common experimental technology, and external usage data (API token volumes, downloads) to validate implied D_inf.

### 7.D Doraszelski–Jaumandreu controlled Markov process as a model of algorithmic R&D
**Model.** `y_j = F(n_j, d_j; θ) + ω_{f(j),t} − ε_j` (T1'); `ω_ft = g(ω_{f,t−1}, r_{f,t−1}, s_{t−1}) + ξ_ft`, where r = log research inputs (experiment compute, researchers), s = spillover index (best public recipe at t−1, open-weight releases), ξ = unpredictable research outcome.
**Recovering ω_{f,t−1}.** DJ invert a static FOC with observed prices; the ML analog is simpler: the lab's previous-generation models give `ω̂_{f,t−1}(θ) = y_{j'} − F(n_{j'}, d_{j'}; θ) + ε̂_{j'}`, with ε purged by averaging over sizes/seeds/evals within the family-generation (ACF first-stage role).
**Estimating equation (DJ eq. 6 analog).** `y_j = F(n_j, d_j; θ) + g(ω̂_{f,t−1}(θ), r_{f,t−1}, s_{t−1}) + ξ_ft − ε_j`; moments `E[A(z)(ξ − ε)] = 0`, z = (n_j if predetermined w.r.t. ξ, lagged inputs, r_{t−1}, s_{t−1}); sieve g with a dummy for "no disclosed research input".
**Testable implications.** (1) Exogenous vs controlled progress (Ho et al.'s common exponential trend is g = ω_{t−1} + const); (2) share of Var(ω) due to ξ (DJ: 25–75% in Spanish manufacturing); (3) complementarity ∂²g/∂ω∂r > 0 (DJ: yes in most industries) — do frontier labs gain more from research compute?; (4) persistence ∂g/∂ω (DJ: lower than the knowledge-capital model; ML: fast diffusion of recipes implies low private persistence, high spillover loading); (5) neutrality: estimate separate augmentation rates for N and D (DJ 2018) and test α_year = β_year (§4.3).
**Data reality.** Research inputs are mostly unobserved; proxies: headcount, publications, disclosed compute, funding; family generations are few (T≈3–6). DJ-style parametric inversion is preferable to nonparametric sieves given tiny samples.

---------------------------------------------------------------------------------------------------

## 8. Productivity dispersion and misallocation

### 8.1 Syverson (2004a) REStat 86(2):534–550; (2004b) JPE 112(6):1181–1222; (2011) JEL 49(2):326–365 [abs; 2011 WP read for the numbers]
**Statements.** 2004a: 443 US manufacturing industries; higher product substitutability → less within-industry productivity dispersion, higher median productivity. 2004b: ready-mixed concrete; spatial substitutability truncates the productivity distribution from below. 2011 (citing 2004a): average within-4-digit-SIC 90–10 log TFP gap 0.651 ⇒ **TFP ratio 1.92** (SD across industries 0.173); Hsieh–Klenow find 90–10 ratios over 5:1 in China and India; TFP AR(1) coefficients 0.6–0.8.
**Scaling-law analog.** Benchmark for "compute-efficiency dispersion" across model families (Ruan et al.: families differ in compute-to-capability efficiency). A headline number for the paper: the 90–10 ratio of lab "TFP" (compute multiplier at fixed loss) vs 1.92 in US manufacturing. Syverson's substitutability mechanism predicts that open-weight competition (high substitutability) truncates low-efficiency recipes.

### 8.2 Hsieh & Klenow (2009), QJE 124(4):1403–1448 [abs]
**Statement.** Plant-level wedges in marginal revenue products of capital and labor within narrow industries; equalizing to US-level dispersion would raise manufacturing TFP 30–50% (China), 40–60% (India) [abs]. Under CD + CES demand, TFPR ∝ (MRPK)^α (MRPL)^{1−α} is equalized absent distortions [recalled].
**Scaling-law analog.** (i) The inference wedge (§7.C) is an input-specific distortion τ_N; dispersion of τ_N across labs reflects heterogeneous inference demand (not misallocation). (ii) Dispersion in marginal loss-reduction per FLOP across labs under compute constraints (export controls, capital constraints) is a compute-misallocation object — loose analog, since loss reductions in different labs are not a common good.

### 8.3 De Loecker & Syverson (2021), "An Industrial Organization Perspective on Productivity", Handbook of IO vol. 4, ch. 3, pp. 141–223 [abs]
**Content.** Productivity concepts, facts, measurement and estimation, markets and productivity interactions, open questions. Use as the umbrella citation for the IO measurement framework (TFPQ/TFPR, production approach to markups, determinants of productivity).

### 8.4 Demirer (2020), "Production Function Estimation with Factor-Augmenting Technology: An Application to Markups" (MIT JMP) [abs]; Diamond, McFadden & Rodriguez (1978), in Fuss & McFadden (eds.), Production Economics: A Dual Approach, vol. 2, North-Holland, pp. 125–147 [abs]
**Statements.** Demirer: nonparametric production functions with factor-augmenting technology, identified via a control variable and optimality of input expenditures; output elasticities and markups for US and four developing countries [abs]. DMR: the elasticity of substitution and the bias of technical change cannot be separately identified from time series of output, inputs and marginal products alone (impossibility theorem) [abs].
**Scaling-law analog.** DMR applied to scaling: the drift over time in compute-optimal tokens/parameter (Kaplan-era ~C^0.73 for N; Chinchilla ~0.5; later over-training) cannot, from time series of frontier allocations alone, be attributed to data-augmenting progress vs a changing substitution elasticity vs changing "relative prices" (inference demand). Cross-sectional IsoFLOP experiments at each date identify local σ (like Raval 2019's cross-sectional wage variation), after which the time dimension identifies bias — a clean identification narrative for the paper. Ho et al.'s separation of α' and β' rests entirely on the additive power-law functional form.

---------------------------------------------------------------------------------------------------

## 9. Synthesis: which IO estimator for which ML dataset [derived]
| ML dataset | variation | main threat | IO estimator | assumptions that do the work |
|---|---|---|---|---|
| Single-lab IsoFLOP/size sweep (Chinchilla, Llama 3 IsoFLOPs, Pythia, Gadre, Muennighoff, Sardana) | designed | ε only; schedule artifacts | NLS/Huber (ZKD); proper bootstrap/GMM SEs | ω fixed by design; residual realized after choice |
| Open-weight family herds (Llama 3 8B/70B/405B; Qwen; Gemma) | within family-gen N; across-gen N, D | ω_ft; D mismeasured | Mundlak/Hoch FE at family×gen; Wooldridge one-step GMM | common recipe within generation |
| Cross-lab public evals (Ho et al.; Epoch) | observational | transmission bias (sign regime-dependent), notability selection, output units | ACF-style with N as predetermined, D flexible; OP propensity score; KG/BHKZ output discipline | timing; scalar ω; known inclusion rule |
| Frontier allocations over vintages | belief/price shifts | functional dependence; DMR | FOC/share identification with wedge (GNR/DLW); natural experiment (Kaplan→Chinchilla) | optimality w.r.t. correct beliefs (post-2022) |
| Lab panels with research inputs | generation × lab | endogenous progress | DJ controlled Markov | r_{t−1} predetermined; ω recoverable from prior models |

---------------------------------------------------------------------------------------------------

## 10. ML ↔ IO equivalence list (strength: exact / close / loose / breaks-down)
1. Iso-loss curve ↔ isoquant — exact.
2. IsoFLOP curve (ND = const) ↔ isocost line — exact, but cost is multiplicative (log-linear) not additive.
3. Compute-optimal frontier L*(C) ↔ (inverse) cost function; Nerlove duality — exact.
4. Chinchilla allocation rule N* ∝ C^a ↔ conditional factor demand / expansion path — exact.
5. On-path collinearity of (n, d) ↔ ACF functional dependence — close.
6. IsoFLOP sweep ↔ exogenous input-mix variation (ACF i.i.d. price shock; Bond–Söderbom stochastic adjustment cost) — close (experimental rather than natural).
7. FOC θ_N = θ_D ↔ cost-share = output-elasticity condition (DLW eq. 3; GNR share regression) — exact (with unit cost elasticities from C = 6ND).
8. Frontier exponent γ = αβ/(α+β) ↔ returns to scale (FGT) — exact as the scale elasticity of reducible loss in compute.
9. Chinchilla reducible loss with α = β ↔ CES with σ = 1/(1+α) — exact; α ≠ β ↔ CES-type, non-constant σ — close.
10. Over-training wedge θ_N/θ_D = 1 + D_inf/(3D) ↔ DLW markup θ/α — loose (no revenue; input-specific); ↔ Hsieh–Klenow input distortion τ_K — close.
11. Lab recipe quality ↔ Hicks-neutral TFP ω / Mundlak "management" — close.
12. Seed/eval noise ↔ non-transmitted ε (ZKD, OP η) — close.
13. Cross-lab loss–input regressions ↔ Marschak–Andrews transmission bias — close (sign depends on behavioral regime).
14. Family×generation fixed effects ↔ Mundlak/Hoch covariance estimators — close.
15. N fixed at design, D adjustable ↔ capital predetermined vs flexible input (OP/ACF timing) — close.
16. D given N as proxy for ω ↔ LP intermediate-input proxy — loose (valid under target/marginal rules, fails under fixed budget; scalar-unobservable doubtful).
17. N/D split as proxy for Hicks-neutral ω ↔ LP/OP proxy — breaks-down (split is ω-invariant); works only for factor-biased ω.
18. Notability/publication selection ↔ OP exit selection — loose (selection on outcome, not survival threshold).
19. Envelope "min over training curves" ↔ frontier/extreme-value selection — loose.
20. Algorithmic progress ↔ TFP growth — close; Ho et al. N_eff/D_eff ↔ factor-augmenting technical change (DJ 2018, Demirer) — exact functional form.
21. Hicks-neutral progress ↔ α_year = β_year restriction in Ho et al. Eq. 3 — exact [derived].
22. Research compute/researchers driving recipe ↔ DJ controlled Markov with R&D — close.
23. Learning from deployment data ↔ learning by exporting (De Loecker 2013) — close.
24. Distillation/synthetic data/teacher compute ↔ intermediate inputs; gross output vs value added (GNR) — close.
25. MoE total vs active parameters ↔ capital stock vs capital services/utilization — loose.
26. Data repetition (≤4 epochs ≈ unique) ↔ depreciation / effective input — loose.
27. Bits-per-byte loss ↔ TFPQ; accuracy/Elo/revenue ↔ TFPR — close for Elo/revenue, loose for accuracy (bounded transform, not price).
28. Per-token loss & token counts across tokenizers ↔ Klette–Griliches common-deflator bias — close.
29. Non-embedding vs total params; 6ND-imputed compute; MFU ↔ capital measurement error (CWDL); unobserved input-price dispersion (GLZ) — close.
30. Multiple held-out evals as indicators ↔ Hu–Huang–Sasaki two proxies — close.
31. Architecture/recipe types with different exponents ↔ Kasahara–Schrimpf–Suzuki finite mixture — close.
32. Drift in optimal tokens/param over vintages ↔ Diamond–McFadden–Rodriguez non-identification — close.
33. Compute-efficiency dispersion across families ↔ Syverson 90–10 TFP dispersion (1.92) — close.
34. Blundell–Bond mean-stationarity/initial-conditions ↔ ML input growth — breaks-down.
35. Blundell–Bond COMFAC AR(1) productivity ↔ AR(1) lab recipe process — close.
36. Kaplan-era allocations ↔ optimization error / belief shock (Marschak–Andrews "urge, ability, luck"; ACF) — close.

---------------------------------------------------------------------------------------------------

## 11. Contribution ideas (for the paper)
1. **Functional-dependence theorem for scaling laws** (§7.A): on-path data identify exponents at best through curvature; the MRTS level (optimal tokens/param) is identified only by assuming optimality; formalizes why IsoFLOP designs are necessary and why parametric Approach 3 disagrees with Approaches 1–2. Simulation: fit (T1) on on-path vs IsoFLOP designs; show variance explosion of A/B.
2. **Revealed inference demand** (§7.C): estimator (W) with IsoFLOP-identified technology; robust to Hicks-neutral lab TFP; Raval-style over-identification across sizes within a family; revealed-preference test that rejects Hoffmann Approach-3 and corroborates Besiroglu et al.
3. **ACF/OP-style cross-lab production function** with N predetermined, D flexible, family×generation effects, notability-selection correction; report lab-TFP dispersion (90–10) vs Syverson's 1.92.
4. **Algorithmic progress as a controlled Markov process** (DJ 2013): endogenous, lab-specific, uncertain; test neutrality (α_year = β_year), estimate the unpredictable share of recipe variance, and spillovers.
5. **Sign of transmission bias depends on the lab objective** (§1.1): capability-target (attenuation, Nerlove reverse regression), budget-exogenous (ZKD consistency), funding-responds (upward); bounds via forward and reverse regressions.
6. **Output and input measurement discipline**: bits-per-byte and bytes of data (Klette–Griliches); total parameters (Pearce–Song) and hardware-based compute instruments (CWDL/GLZ).
7. **Kaplan→Chinchilla belief shock as a natural experiment** that breaks functional dependence in observational data (shifted input mix independently of ω).
8. **Inference versus training as a two-input cost function**: estimate the full Sardana cost function and the implied lab-specific shadow prices; link to API pricing for a DLW markup on inference (DLEU-style).
9. **Finite-mixture scaling laws** (KSS) to allow architecture-type-specific exponents; show bias in common-exponent algorithmic-progress estimates.
10. **Inference for scaling-law parameters**: IO-style GMM/bootstrap with clustering by lab/family; replicate Besiroglu's CI critique within a GMM framework.

---------------------------------------------------------------------------------------------------

## 12. Data leads (verified existence this session)
- Epoch AI models database (>3,600 models; training compute, parameters, dataset size, cost, dates, organization; "notable" inclusion criteria stated; CC-BY 4.0; CSV): https://epoch.ai/data/notable-ai-models (downloads via https://epoch.ai/data/ai-models).
- Reconstructed Chinchilla (N, D, loss) points from Fig. 4 (240 points), with extraction/analysis notebooks: https://github.com/epoch-research/analyzing-chinchilla (`data/svg_extracted_data.csv`).
- Ho et al. algorithmic-progress dataset (>200 evaluations; Google Sheet linked; MIT-licensed code): https://github.com/epoch-research/lm-algorithmic-progress.
- Pythia suite (16 models 70M–12B, 154 checkpoints, same data order): arXiv 2304.01373 / EleutherAI.
- Muennighoff et al. data-constrained runs (400 runs released): arXiv 2305.16264.
- Gadre et al. over-training testbed (104 models): arXiv 2403.08540.
- Sardana et al. 47 over-trained MPT models (150M–6B, up to 10,000 tokens/param): arXiv 2401.00448.
- Ruan et al. observational scaling (~100 public models, benchmark matrix): arXiv 2405.10938.
- Llama 3 herd disclosures (N, D, FLOPs, IsoFLOP fit, MFU): arXiv 2407.21783.

---------------------------------------------------------------------------------------------------

## 13. Warnings, disagreements and uncertain claims
- **Log-additivity.** IO estimators need `y = f(x) + ω + ε`; scaling laws have an additive irreducible term E, so ω is not log-additive in L. Work with log(L − E) (requires E) or estimate E jointly by nonlinear GMM; E is weakly identified (asymptote) and differs across corpora/tokenizers (1.69 vs 1.82 across the two Chinchilla fits).
- **Sample sizes.** IO methods are built for thousands of plants; public ML data are hundreds of models, few per lab, T ≈ 3–6 generations: nonparametric first stages are infeasible; favor parametric (DJ-style) inversions and experimental designs.
- **Optimality is questionable.** Pre-2022 allocations followed Kaplan's frontier (now attributed to parameter-count measurement, small scale, warmup/tuning artifacts); FOC-based methods (GNR, DLW, §7.B–C) are biased for those vintages.
- **Chinchilla parameters disputed.** Hoffmann Approach 3 (α .3392, β .2849, A 406.4, B 410.7, E 1.6934) vs Besiroglu (α .3478, β .3658, A 482, B 2085, E 1.817); CIs in Hoffmann Table 2 for Approach 3 implausibly narrow (optimizer artifact). All wedge numbers move by ≈2× between sets.
- **Checkpoints are not independent observations** of `L(N, D)`: intermediate checkpoints of a cosine schedule are not models trained to that D (Chinchilla Approach 1 uses an envelope; Porian et al. stress warmup/decay/tuning). Using Pythia checkpoints as a "panel" violates the maintained technology unless schedule-adjusted.
- **Cost identity.** C ≈ 6ND ignores attention/last-layer FLOPs (Porian et al.), MoE activation, and hardware utilization; "compute" in trackers is often imputed from N and D, making C a deterministic function of mismeasured inputs.
- **Selection sign.** The OP-style sign of notability/publication selection (§3.1) is a conjecture; it depends on how inclusion probability varies with N.
- **Timing claims** (N predetermined, D flexible) are stylized; some labs grow models mid-training or restart runs.
- **Bibliographic caveats.** Griliches–Mairesse: commonly cited 1998; Cambridge Core lists the volume (Strøm ed.) with 1999 date, ch. 6, pp. 169–203. Raval (2023) pages are 2592–2611 (not 2627). ACF equations are quoted from the 2005 working paper (published version reorganized; logic identical). Blundell–Bond (2000) estimates quoted from IFS WP 99/04 (may differ slightly from the Econometric Reviews version). DLW formula from NBER WP 15198 (Oct 2010 revision); published AER uses the same formula. Wooldridge (2009) equations and Mundlak (1961), Nerlove (1963), Arellano–Bond (1991) details are recalled, not re-read. Kasahara–Schrimpf–Suzuki has no journal reference on arXiv as of this check. Flynn–Gandhi–Traina (SSRN) and Demirer (JMP) are working papers. Collard-Wexler–De Loecker later circulated as "Production Function Estimation with Measurement Error in Inputs". Ackerberg et al. (2023, J. Econometrics 237(1)) page/article number not verified.
- **Llama 3 per-model token counts** for 8B/70B are not verified here; the illustrative wedge table assumes ≈15T.

---------------------------------------------------------------------------------------------------

## 14. BibTeX keys in `lit/bib/io_controlfn.bib`
IO: marschak1944random, hoch1958simultaneous, hoch1962estimation, mundlak1961empirical, zellner1966specification, nerlove1963returns, griliches1998production, arellano1991some, blundell1998initial, blundell2000gmm, olley1996dynamics, levinsohn2003estimating, ackerberg2005structural, ackerberg2015identification, bond2005adjustment, wooldridge2009estimating, ackerberg2007econometric, gandhi2020identification, doraszelski2013rd, doraszelski2018measuring, deloecker2011product, deloecker2013detecting, deloecker2016prices, deloecker2012markups, deloecker2020rise, bond2021some, raval2023testing, raval2019micro, flynn2019measuring, klette1996inconsistency, foster2008reallocation, collardwexler2016production, grieco2016production, hu2020estimating, ackerberg2023timing, ackerberg2023underidentification, kasahara2023identification, syverson2004product, syverson2004market, syverson2011what, hsieh2009misallocation, deloecker2021industrial, demirer2020production, diamond1978measurement.
ML anchors: hoffmann2022training, kaplan2020scaling, besiroglu2024chinchilla, sardana2024beyond, ho2024algorithmic, grattafiori2024llama, muennighoff2023scaling, pearce2024reconciling, porian2024resolving, ruan2024observational, gadre2024language, biderman2023pythia.
Data: epochai2026notable.
