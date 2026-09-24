# ML core strand: neural scaling laws — foundations, functional forms, extensions

Notes for the AER paper "Scaling Laws as Production Functions". Compiled 2026-09-23.

Source discipline: everything below was read from the paper PDFs (arXiv full text pulled and parsed in this session) unless marked **[DERIVED]** (my own algebra/computation, reproducible from the closed-form formulas in §4 applied to the published parameters) or **[UNVERIFIED]**. Equation/table numbers refer to the arXiv version read. BibTeX keys refer to `lit/bib/ml_core.bib`.

Notation used throughout: N = parameters, D = training tokens, C = training FLOPs (≈6ND), L = held-out cross-entropy (nats/token unless noted), E = irreducible loss, ℓ ≡ L − E = reducible loss. "Output" in the IO sense is any decreasing transform of ℓ (e.g., Y = ℓ^{-1/ρ}); I write ε_N ≡ −∂ln ℓ/∂ln N for output elasticities of the reducible loss.

---

## 0. Top-line takeaways for the paper (short)

1. **The Chinchilla law is (almost exactly) a CES production function.** ℓ = A N^{-α} + B D^{-β}. If α = β = ρ, Y ≡ ℓ^{-1/ρ} = (A N^{-ρ} + B D^{-ρ})^{-1/ρ} is textbook CES (ACMS 1961 form) with σ = 1/(1+ρ). With α ≠ β, the isoquants are "generalized CES" and the Allen/Hicks elasticity of substitution is **σ(N,D) = 1/(1 + β s_N + α s_D)**, s_N = εN/(εN+εD) [DERIVED]. Because s ∈ [0,1], σ is bounded in [1/(1+max(α,β)), 1/(1+min(α,β))]. Hoffmann TeX parameters: σ ∈ [0.747, 0.778]; Besiroglu refit: [0.732, 0.742]; Muennighoff (α=β imposed): 0.739. **N and D are gross complements (σ≈0.74–0.78), robustly across refits** — a much more stable object than the allocation exponent a, which moved 0.46→0.51 between Hoffmann and Besiroglu. Kaplan's L(N,D) implies much stronger complementarity: σ ∈ [0.50, 0.58].
2. **The compute constraint is Cobb–Douglas, not linear.** C = 6ND means ln C is linear in (ln N, ln D) with equal "log-prices". Cost-minimization therefore equates *output elasticities*, εN = εD (not marginal products/prices). This is exactly the De Loecker–Warzynski/GNR first-order condition with cost shares replaced by the elasticities of the cost function (both = 1). It gives: N* ∝ C^{β/(α+β)}, D* ∝ C^{α/(α+β)}, ℓ*(C) ∝ C^{-αβ/(α+β)} (Hoffmann Eq. 4), and at the optimum the share of reducible loss due to finite N equals a = β/(α+β) [DERIVED].
3. **Functional dependence (ACF 2015) is literal here.** On the compute-optimal frontier, ln N and ln D are deterministic linear functions of ln C, so a regression of L on (N,D) using frontier models alone cannot separately identify α and β. Identification comes either from **designed off-frontier variation (IsoFLOP sweeps = randomized input ratios)** or from **imposing the optimality FOC** (then α = γ/a, β = γ/(1−a), where γ is the loss–compute exponent and a the allocation exponent) [DERIVED]. The second route is the GNR/Solow cost-share route and is invalid when labs deliberately deviate from cost minimization (over-training).
4. **Hyperparameters are flexible inputs; L(N,D) is a "concentrated" production function.** Porian et al. show that the Kaplan–Chinchilla gap in the allocation exponent (0.73–0.88 vs 0.50) is produced by mismeasured inputs (last-layer FLOPs, embedding parameters) plus non-optimized flexible inputs (warmup, LR, batch size, AdamW β2): a = 0.835 → 0.706 → 0.602 → 0.571 → 0.497 as each is fixed. DeepSeek's η_opt(C), B_opt(C) laws are literally flexible-input demand functions. This is the scaling-law version of the omitted flexible-input / transmission-bias problem.
5. **Estimated exponents are fragile to (i) output cardinalization (E), (ii) the support of the sample, (iii) the optimizer.** Kaplan's α_N = 0.076 vs Chinchilla α = 0.34 is mostly the E = 0 vs E > 0 choice (log-slope of total loss vs reducible loss) [DERIVED check: Chinchilla-implied total-loss log-slope ≈ 0.05 at N=1e9, D=2e10]. Sardana et al. Table 1: refitting the same functional form on runs with ≤100 vs all tokens/param moves (α,β) from (0.08,0.13) to (0.18,0.24). Besiroglu et al.: the Chinchilla A3 fit was an optimizer failure (averaged instead of summed Huber loss; early termination) with CIs 50× too narrow.
6. **Technical change taxonomy maps cleanly.** Hestness et al.: architecture improvements "only shift the error but do not appear to affect the power-law exponent" = Hicks-neutral (in log-log). Ho et al.: N_eff = N e^{α'(Y−Y0)}, D_eff = D e^{β'(Y−Y0)} = factor-augmenting technical change (effective compute doubling 8.4 months, 95% CI 4.5–14.3). Within the Chinchilla family **any factor-augmenting change leaves the expansion-path exponents (a,b) unchanged and only shifts the intercept G** [DERIVED]; so DeepSeek's finding that better data raises a from 0.450 to 0.578 is evidence of *non-augmenting* (curvature/β-changing) technical change, or of misspecification.
7. **Over-training is a measurable wedge.** Sardana & Frankle's objective min 6N D_tr + 2N D_inf s.t. L(N,D_tr)=ℓ implies εN/εD = 1 + D_inf/(3 D_tr) [DERIVED]. Given a production function, observed (N,D) back out the implied lifetime inference/training token ratio — a DLW-style markup. Illustration with Besiroglu parameters: Llama-3-8B (15T tokens) ⇒ D_inf/D_tr ≈ 12.7; Llama-3-70B ≈ 4.4; 405B ≈ 1.1; GPT-3 ⇒ negative (≈ −1.7, i.e., under-trained/misallocated relative to any cost-minimizing rationale). With Hoffmann (TeX) parameters they are much smaller (8B: 5.8; 70B: 1.2; 405B: −0.7; GPT-3: −2.0) — the wedge is only as identified as the elasticities (the DLW critique carries over one-for-one). Farseer's finding that optimal D/N *rises* with C (non-homothetic technology) is an alternative explanation for over-training that the wedge interpretation must rule out.
8. **Output measurement is a first-order issue.** Loss per token depends on tokenizer (Tao et al. use unigram-normalized loss; MiniCPM and DeepSeek use bits-per-byte), downstream accuracy is a bounded, sigmoidal transform of latent loss (Gadre Err = ε − k e^{−γL}; Llama 3 two-stage NLL→sigmoid; Ruan logistic link + low-rank capability factors), and "emergence" is largely a metric artifact (Schaeffer). Downstream scaling is predictable only in 39% of cases (Lourie et al.). This is the TFPQ vs TFPR / latent-output-with-noisy-indicators problem.
9. **Micro-foundations match Houthakker (1955)/Jones (2005).** (Two independent ML instances: Michaud et al. 2023 for pretraining loss; Schaeffer et al. 2025 for repeated-sampling inference, where per-problem exponential scaling aggregates to a power law through a heavy-tailed difficulty distribution.) Michaud et al.'s quantization model derives power-law loss from a Zipf (Pareto) distribution of discrete "quanta" learned in frequency order (α_N = α, α_D = α/(α+1)); Houthakker/Jones derive Cobb–Douglas aggregates from Pareto-distributed Leontief techniques/ideas. Same mathematics: aggregation over Pareto-distributed micro units ⇒ power-law aggregate production. The quantization model also implies a *cross-equation restriction* β = α/(1+α) and a = 1/(α+2) testable on sweep data.

---

## 1. Foundations (pre-Chinchilla)

### 1.1 Hestness et al. (2017), "Deep Learning Scaling is Predictable, Empirically" — `hestness2017deep` (arXiv:1712.00409)
- Form: generalization error ε(m) = α m^{β_g} (+γ irreducible), m = training-set size; best-fit model size s(m) ∝ m^{β_p}.
- Estimates (best-fit across model sweeps per data shard): word LM β_g ≈ −0.066 (−0.0656 ± 1% across architectures; "ε(m) = 11.9 m^{-0.066}"), char LM β_g = −0.0936 (SGD)/−0.0954 (Adam); NMT β_g ≈ −0.128 (projected from best-fit composite curve; single models −0.36/−0.30); ImageNet top-1 β_g = −0.309, top-5 −0.488, cross-entropy −0.35; speech (DS2 and attention models, very different encoders/decoders) β_g = −0.299 ± 0.7% — same exponent across architectures. Model size exponent β_p ≈ 0.69 ± 5% (LSTM/RHN word LM), 0.78 (4-layer LSTM), char LM 0.78/0.92, ImageNet 0.573; general claim β_p ∈ [0.5, 1).
- Theory benchmark cited: β_g = −0.5 or −1 (learning theory); empirical exponents far smaller in magnitude.
- **IO-relevant claim:** "model improvements only shift the error but do not appear to affect the power-law exponent" (abstract) ⇒ Hicks-neutral technical change in log–log space; exponent = technology-invariant "elasticity", intercept = TFP. Also three regions (small-data, power-law, irreducible) = sigmoidal/bounded output.
- Method: fit per data shard, choose best model per shard via hyperparameter search; power law fit in log space. Data: Baidu internal + public (WMT, LM1B, ImageNet); not released.

### 1.2 Rosenfeld et al. (2020, ICLR), "A Constructive Prediction of the Generalization Error Across Scales" — `rosenfeld2020constructive` (arXiv:1909.12673)
- Joint form (their Eq. 4), n = data, m = model size: ε̃(m,n) = a n^{−α} + b m^{−β} + c∞ — **the Chinchilla functional form, two years earlier**. With envelope for the random-guess transition (Eq. 5): ε̂ = ε0 |ε̃/(ε̃ − iη)|, ε0 = random-guess error, η = transition pole; a normalized to 1.
- Criteria C1–C5 (limits in m, n; no dependence of the constant on either).
- Estimation: least squares on relative divergence δ = (ε̂ − ε)/ε, 10-fold CV; 42–49 (m,n) configurations per dataset; |θ| ≤ 6.
- Table 2 estimates (α = data exponent, β = model exponent): ImageNet α 0.75, β 0.61; CIFAR10 0.66/0.53; CIFAR100 0.70/0.51; DTD 0.40/1.16; Aircraft 1.10/0.83; UCF101 0.93/0.54; LM: PTB α 0.81, β 0.34; WikiText-2 1.01/0.22; WikiText-103 0.74/0.56. Fit μ < 1%, σ < 5%; extrapolation divergence μ = 4.5% (σ = 4.7%).
- Note exponents here are on top-1 *error* after the envelope transform — not comparable to loss exponents (output cardinalization).

### 1.3 Kaplan et al. (2020), "Scaling Laws for Neural Language Models" — `kaplan2020scaling` (arXiv:2001.08361)
- Setup: WebText2 (2.29e10 tokens; 6.6e8 test), N = **non-embedding** params 768 to 1.5e9; D from 22M to 23B tokens; Adam (Adafactor >1B), 2.5e5 steps, batch 512×1024 tokens, 3000-step warmup + cosine to zero; C ≈ 6NBS non-embedding compute (PF-days).
- Univariate laws (Eq. 1.1–1.3): L(N) = (N_c/N)^{α_N}, α_N ≈ 0.076, N_c ≈ 8.8e13; L(D) = (D_c/D)^{α_D}, α_D ≈ 0.095, D_c ≈ 5.4e13 (early-stopped, large models); L(C_min) = (C_c^min/C_min)^{α_C^min}, α_C^min ≈ 0.050, C_c^min ≈ 3.1e8 PF-days. Critical batch B_crit(L) = B*/L^{1/α_B}, B* ≈ 2e8 tokens, α_B ≈ 0.21 (Eq. 1.4).
- **Joint law (Eq. 1.5 / 4.1):** L(N,D) = [(N_c/N)^{α_N/α_D} + D_c/D]^{α_D}. Table 2 fit of Eq. 1.5: α_N = 0.076, α_D = 0.103, N_c = 6.4e13, D_c = 1.8e13. Principles: (1) vocabulary rescales loss by overall factor; (2) correct univariate limits; (3) analytic at D = ∞ (1/D expansion with integer powers) — principle 3 is what makes the D-term enter linearly inside the bracket. Footnote 4 considers the alternative [(N_c/N)^{α_N} + (D_c/D)^{α_D}]^β and rejects it only because it lacks a 1/D expansion.
- **Crucial measurement caveat:** L(N,D) is *early-stopped test loss on a dataset of D tokens with 10% dropout* — i.e., potentially multi-epoch; compute is not 6ND. Overfitting rule D ≳ (5×10³) N^{0.74} (Eq. 4.4) from seed noise ≈ 0.02.
- L(N,S) = (N_c/N)^{α_N} + (S_c/S_min(S))^{α_S}, S_c ≈ 2.1e3, α_S ≈ 0.76 (Eq. 1.6). Compute-efficient allocation (Eq. 1.7–1.8): N ∝ C^{α_C^min/α_N}, B ∝ C^{α_C^min/α_B}, S ∝ C^{α_C^min/α_S}, α_C^min = 1/(1/α_S + 1/α_B + 1/α_N); empirically N ∝ C_min^{0.73}, B ∝ C^{0.24}, S ∝ C^{0.03}, D ∝ C^{0.27} (Fig. 14: N = 1.3e9 C^{0.73}).
- Self-identified inconsistency (§6.3): L(C_min) ∝ C^{−0.050} vs data-limited L(D(C_min)) ∝ C^{−0.03}; curves intersect at C* ~ 1e4 PF-days, N* ~ 1e12, D* ~ 1e12, L* ~ 1.7 nats/token.
- Estimation: curve fits in log space; no standard errors reported; data not released.
- **[DERIVED] What Kaplan's own L(N,D) implies for allocation if one (mis)applies C = 6ND:** isoquants satisfy (N_c/N)^{α'} + D_c/D = k with α' = α_N/α_D; cost-min in logs gives D ∝ N^{α'}, hence N ∝ C^{1/(1+α')} = C^{0.575} (Table 2 values) or C^{0.556} (§3 values). So the 0.73 comes from L(N,S)+B_crit, not from the joint N–D law; Kaplan's joint law is internally closer to Chinchilla than Kaplan's headline allocation. Caveat: because D is early-stopped multi-epoch data, the premise C = 6ND is false for these fits — flag rather than headline.

### 1.4 Henighan et al. (2020), "Scaling Laws for Autoregressive Generative Modeling" — `henighan2020scaling` (arXiv:2010.14701)
- L(x) = L∞ + (x0/x)^{α_x}, x ∈ {N, C, D} (Eq. 1.1); L∞ interpreted as entropy S(True), reducible loss as D_KL(True‖Model) (Eq. 1.2).
- Domains: language, images 8×8/16×16/32×32/64×64, video, math, image↔text. N_opt(C) ∝ C^{β} with β ≈ 0.7 "for all domains" (language 0.73, image 8×8 0.64, 16×16 0.75, ...). Language row: L(N) ∝ N^{−0.070}, L(C) ∝ C^{−0.048} (no L∞ for language).
- IO note: the irreducible-loss/reducible-loss split = choice of output measure; the "universal" 0.7 allocation exponent was later overturned (Chinchilla) — shows common-mode methodological bias across domains (same training protocol ⇒ same bias).

### 1.5 Hernandez et al. (2021), "Scaling Laws for Transfer" — `hernandez2021scaling` (arXiv:2102.01293)
- Effective data transferred (Eq. 1.1): D_T = k (D_F)^{α} (N)^{β}; D_F = fine-tuning data; total effective data D_E = D_F + D_T. Table 1: text→python k = 1.9e4, α = 0.18, β = 0.38; 50% text + 50% non-python code→python k = 2.1e5, α = 0.096, β = 0.38. Effective-data multiplier (D_F + D_T)/D_F ≈ k N^β D_F^{α−1} in the low-data regime.
- Unified fine-tuning law (Eq. 6.1): L ≈ [(N_C/N)^{α_N/α_D} + D_C/(k D_F^{α} N^{β})]^{α_D} — i.e., pre-training enters as an *input-augmenting* factor on data.
- **IO mapping:** "effective input" / quality-adjusted input (human-capital-augmented labor); pre-training = a durable intermediate input/knowledge capital whose services augment downstream data; α = "directed proximity" of distributions ~ transferability of capital across products (scope economies). Strength: close.

---

## 2. Compute-optimal scaling: Chinchilla and its econometric aftermath

### 2.1 Hoffmann et al. (2022), "Training Compute-Optimal Large Language Models" (Chinchilla) — `hoffmann2022training` (arXiv:2203.15556; NeurIPS 2022 per Semantic Scholar venue field)
- Problem (Eq. 1): N_opt(C), D_opt(C) = argmin_{N,D s.t. FLOPs(N,D)=C} L(N,D). The paper itself states that "the computational budget C is a deterministic function FLOPs(N,D)" — the functional-dependence premise.
- Data: >400 models, 70M to >16B params, 5B to 500B tokens (abstract; intro says "over 400B"); MassiveText; FLOPs counted *including* embeddings, factor 2 per MAC (App. F); parameters include embeddings (contrast Kaplan).
- **Approach 1 (training-curve envelope):** fixed N (70M to >10B), 4 cosine horizons spanning 16× (decay 10×); smooth/interpolate curves; at 1500 log-spaced FLOP values take the min-loss run; fit N_opt ∝ C^a, D_opt ∝ C^b: a = 0.50 (0.488, 0.502), b = 0.50 (0.501, 0.512). All selected points lie in last 15% of training.
- **Approach 2 (IsoFLOP profiles):** 9 budgets {6e18, 1e19, 3e19, 6e19, 1e20, 3e20, 6e20, 1e21, 3e21}; N up to 16B; cosine length matched to tokens; parabola fit of final (smoothed) loss vs log N per budget → argmin; a = 0.49 (0.462, 0.534), b = 0.51 (0.483, 0.529).
- **Approach 3 (parametric):** L̂(N,D) = E + A/N^α + B/D^β (Eq. 2), motivated by risk decomposition (App. D.2, Eq. 9: Bayes risk + approximation error + single-epoch stochastic-optimization error). Estimation (Eq. 3/11): min Σ_i Huber_δ(LSE(a − α log N_i, b − β log D_i, e) − log L_i), δ = 1e−3, L-BFGS, grid of initializations α,β ∈ {0,0.5,…,2}, e ∈ {−1,−0.5,…,1}, a,b ∈ {0,5,…,25}; A,B,E = exp(a,b,e). Reported (Eq. 10): E = 1.69, A = 406.4, B = 410.7, α = 0.34, β = 0.28 (TeX-source precise values per Besiroglu: E = 1.6934, α = 0.3392, β = 0.2849). a = 0.46 (0.454, 0.455), b = 0.54 (0.542, 0.543). CIs = 10th–90th percentiles from 100 bootstraps of 80% subsamples.
- Efficient frontier (Eq. 4): N_opt(C) = G (C/6)^a, D_opt(C) = G^{−1}(C/6)^b, G = (αA/(βB))^{1/(α+β)}, a = β/(α+β), b = α/(α+β).
- Low-compute points (C ≤ 1e21) have larger residuals and are effectively down-weighted by Huber; "negative curvature" of the C→N_opt frontier (App. E; linear fits on first/middle/last thirds of frontier points differ) ⇒ A3 predicts smaller N_opt.
- Table 3 (Approach 1): 400M → 1.92e19 FLOPs, 8.0B tokens; 1B → 1.21e20, 20.2B; 10B → 1.23e22, 205.1B; 67B → 5.76e23, 1.5T; 175B → 3.85e24, 3.7T; 280B → 9.90e24, 5.9T; 520B → 3.43e25, 11.0T; 1T → 1.27e26, 21.2T; 10T → 1.30e28, 216.2T (≈20 tokens/param: origin of the rule of thumb). Chinchilla: 70B params, 1.4T tokens, same compute as Gopher (5.76e23).
- Kaplan discrepancy attributed (§2) to fixed LR schedule/training length across runs (intermediate losses with a 130B-token cosine overestimate the loss at D' ≪ 130B) and small models (Kaplan mostly <100M; Chinchilla mostly >500M).
- Public data: none (raw runs not released). Figure 4 digitized by Epoch (below).

### 2.2 Besiroglu, Erdil, Barnett, You (2024), "Chinchilla Scaling: A replication attempt" — `besiroglu2024chinchilla` (arXiv:2404.10102)
- Data: 245 points digitized from Hoffmann Fig. 4 SVG (x = FLOP, y = N, loss from color on a log scale 2–5; ~0.01 loss precision); 5 outliers (tokens/param < 0.4 at ~1e19 FLOP) dropped → 240.
- Refit (same Huber δ = 1e−3, LSE, grid; BFGS for bootstraps): **L = 1.8172 + 482.01/N^{0.3478} + 2085.43/D^{0.3658}**; bootstrap SEs: A 124.58, B 1293.23, E 0.03, α 0.02, β 0.02, a = β/(α+β) = 0.5126 (SE 0.018). With outliers (Table 3): E 1.89 (0.044), A 463.3 (145.0), B 12530 (61650), α 0.345 (0.018), β 0.452 (0.054), a 0.512 (0.032); implied 25.6 tokens/param.
- Tests: χ² test of parameter equality p < 1e−51 (excl. outliers); E and β individually p = 2.6e−6 and 1.1e−4; LR test vs unrounded Hoffmann p = 1.2e−16. Rounded published values fit badly: β = 0.28 vs 0.2849 biases the data term by ≈ (1e11)^{0.0049} − 1 ≈ 13% at D = 1e11.
- **Diagnosis confirmed by a Hoffmann author (Borgeaud 2024):** Huber losses were *averaged* rather than summed → high loss scale → L-BFGS early termination, both in the point estimate and in bootstraps → implausibly narrow CIs (0.454–0.455 would require ~600,000 runs vs < 500 actual; the refit's 80% CI is ~50× wider).
- Hoffmann A3 implies ~70 tokens/param at Chinchilla scale, inconsistent with the 20 used; refit implies ~20 (80% CI of optimal D/N at ≥1e26 FLOP: 4 to 40).
- Data/code: https://github.com/epoch-research/analyzing-chinchilla (verified 200) — **primary public dataset for the paper's estimation exercise.**
- Econometric lessons: optimizer convergence/scale of the objective; rounding of reported parameters matters because exponents multiply log-inputs of ~25; standard errors never reported in the original; digitized data carry measurement error in the dependent variable (color quantization) and regressors.

### 2.3 Pearce & Song (2024, TMLR), "Reconciling Kaplan and Chinchilla Scaling Laws" — `pearce2024reconciling` (arXiv:2406.12907)
- Claim: most of the Kaplan (N ∝ C^{0.73}) vs Chinchilla (C^{0.50}) gap comes from Kaplan counting **non-embedding** parameters/compute (N\E, C\E = 6N\E D) instead of totals (N_T = N_E + N_\E), combined with small scale (768 to 1.5B). Simulating Chinchilla's law in Kaplan's regime and variables yields a *local* coefficient 0.74–0.78; in total parameters 0.51 (Epoch constants).
- Recommendation: use total parameters and total compute. Code: github.com/TeaPearce/Reconciling_Kaplan_Chinchilla_Scaling_Laws.
- **IO mapping:** non-classical input measurement error whose magnitude is a decreasing function of scale (embedding share falls with N) ⇒ biased elasticities; analogous to measuring capital excluding a component whose share varies systematically with firm size.

### 2.4 Porian, Wortsman, Jitsev, Schmidt, Carmon (2024, NeurIPS), "Resolving Discrepancies in Compute-Optimal Scaling of Language Models" — `porian2024resolving` (arXiv:2406.19146)
- >900 training runs; OpenWebText2 and RefinedWeb; FLOP grid {1.25e16·2^i}, i=0..11; IsoFLOP approach with noise-and-interpolate bootstrap (Akima interpolation) for N*(C_i) and its log-SD; weighted log-linear regression (weights ∝ 1/var) = log-space Gaussian ML; CIs from bootstrap quantiles.
- **Sequence of corrections (RefinedWeb, a with 95% CI):** reproduce Kaplan 0.835 (0.82, 0.85) [OWT2 0.864]; + count last-layer (unembedding) FLOPs 0.706 (0.69, 0.72); + warmup scaled with model size 0.602 (0.59, 0.62); + cosine decay matched to budget 0.571 (0.56, 0.59) [little effect — contradicts Hoffmann's conjecture]; + per-size tuning of LR, batch size, AdamW β2 (constant LR, no decay) 0.497 (0.49, 0.50). Implied N*(5.88e23): 3T → 787B → 292B → 183B → 77B.
- Also derive power laws for optimal LR and batch size; β2 = 0.95 suboptimal at batch ≤128.
- Data, code, checkpoints: github.com/formll/resolving-scaling-law-discrepancies; huggingface.co/formll/resolving-scaling-law-discrepancies.
- **IO mapping (strong):** hyperparameters = flexible inputs chosen conditional on the state (N, D). The object "L(N,D)" is the production function *with flexible inputs concentrated out at their optimum*. If flexible inputs are set by a rule that is suboptimal and scale-dependent, the residual "productivity" of a run is correlated with N ⇒ transmission-type bias in the allocation exponent (0.50 → 0.60–0.84). Measurement error in C (last-layer FLOPs) is additional. This is the cleanest empirical demonstration in the literature that "input choices correlated with unobserved efficiency" bias scaling-law parameters.

### 2.5 Choshen, Zhang, Andreas (2024/2025, ICML), "A Hitchhiker's Guide to Scaling Law Estimation" — `choshen2024hitchhiker` (arXiv:2410.11840)
- Dataset: losses/downstream evals for **485 pretrained models, >40 families, 1.9M training steps** (Pythia, OPT, OLMo, Amber, K2, Mamba, RedPajama, ModuleFormer MoE, Gadre over-trained, Bloom, T5-Pile, Muennighoff, Gopher, GPT-3...). Repo: https://github.com/IBM/ColPret (link in PDF; verified 200).
- Findings: seed-to-seed variation changes loss by up to 4% (most published pretraining-decision effects are 4–50%); using intermediate checkpoints improves fits; families differ in scaling shape but a single target-family model + other-family exponents often predicts well (≈ "common slopes, family-specific intercepts" = fixed-effects production function); ARE thresholds 15/10/5%.
- **IO mapping:** unbalanced panel of "firms" (families) × "sizes" × "time" (checkpoints); seed variance = output measurement error/idiosyncratic productivity shock; pooling exponents across families = homogeneous-slope assumption testable via random coefficients.

### 2.6 Hägele et al. (2024, NeurIPS spotlight), "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations" — `hagele2024scaling` (arXiv:2405.18392)
- Constant LR + short cooldown (linear or 1−sqrt, ~20% of steps) matches cosine; cooldown can be launched retroactively from any checkpoint; stochastic weight averaging improves the trajectory without cooldown.
- Scaling-law experiments with reusable runs: spacing token ratios 10/20/30 saves ~half the FLOPs and GPU hours.
- Code: github.com/epfml/schedules-and-scaling.
- **IO mapping:** changes what an "observation" is: with cosine schedules, intermediate checkpoints are *off* the production function (flexible input — LR schedule — not optimized for that D); with WSD/cooldown, one run yields many on-frontier observations (a within-unit panel along D). This removes the Kaplan-type bias Hoffmann identified and cuts the cost of IsoFLOP designs.

### 2.7 Sardana, Portes, Doubov, Frankle (2024, ICML), "Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws" — `sardana2024beyond` (arXiv:2401.00448)
- Objective (Eq. 3): N*(ℓ, D_inf), D*_tr(ℓ, D_inf) = argmin_{N,D_tr | L(N,D_tr)=ℓ} 6N D_tr + 2N D_inf (FLOPs); dollar version (Eq. 6): 6N D_tr C_tr/MFU_tr + 2N C_inf (D_inp/MFU_inp + D_out/MFU_out). No closed form with D_inf > 0; solved by Newton root-finding. Inference MFU can be ~1% vs 40–60% training.
- Examples: 7B-Chinchilla-quality with 1e11 inference tokens ⇒ 6B model on 1.18× data; 30B-Chinchilla-quality with 1.5B requests ⇒ 16B on 3.35T tokens, −17% cost.
- Experiments: **47 models** (MPT arch), 150M–6B, 10 to 10,000 tokens/param; loss keeps falling to 10,000 tok/param; lines of loss vs FLOPs for ratios ≥20 are nearly parallel.
- **Table 1 (sample-support dependence of the parametric fit; Chinchilla procedure, δ = 1e−3):** ≤100 tok/param: α 0.08, β 0.13, A 7.199, B 25.97, E 0.17; ≤250: 0.13, 0.16, 14.23, 39.54, 0.98; ≤500: 0.13, 0.16, 17.07, 35.80, 0.95; all data: 0.18, 0.24, 33.66, 138.9, 1.45; vs Chinchilla 0.34, 0.28, 406.4, 410.7, 1.69. None fits the 150M long-ratio runs well.
- **[DERIVED] wedge:** FOC in logs: ∂ℓ/∂ln N ÷ ∂ℓ/∂ln D_tr = ∂cost/∂ln N ÷ ∂cost/∂ln D_tr = (6ND_tr + 2ND_inf)/(6ND_tr) ⇒ **εN/εD = 1 + D_inf/(3D_tr)**. Training-only cost-minimization is the special case εN = εD. The ratio of output elasticities to "cost elasticities" is a markup-like wedge (De Loecker–Warzynski 2012 logic: markup = output elasticity / expenditure share of a flexible input).

### 2.8 Gadre et al. (2024), "Language models scale reliably with over-training and on downstream tasks" — `gadre2024language` (arXiv:2403.08540)
- Testbed: **104 models**, 0.011B–6.9B params, token multipliers M = D/N from 20 to 640, three datasets (C4, RedPajama, RefinedWeb). Hyperparameters tuned only at M = 20.
- Assumes α = β (from Chinchilla's near-equal exponents) ⇒ reparameterization (Eq. 4): **L(C, M) = E + (a M^{η} + b M^{−η}) C^{−η}**, η = α/2, a = A(1/6)^{−η}, b = B(1/6)^{−η}. Interpretation: parallel lines in log L′ vs log C for different M; the offset aM^η + bM^{−η} is minimized at M* = (b/a)^{1/(2η)}.
- Downstream (Eq. 5): average top-1 error Err(L) = ε − k exp(−γL) (equivalently Err(PP) = ε − k PP^{−γ}), 17-task subset of 46 LLM-foundry tasks (tasks ≥10 points above chance at 0.154B).
- Estimation: SciPy curve_fit (Levenberg–Marquardt nonlinear least squares). Predict 1.4B/900B-token (32× over-trained) and 6.9B/138B from 300× less compute; downstream from 20× less compute.
- Data: github.com/mlfoundations/scaling (verified 200).
- **IO mapping:** imposing α = β is imposing an exact CES with σ = 1/(1+α); the bracket aM^η + bM^{−η} is the CES unit-isoquant "cost of deviating from the optimal factor ratio". Err(L) is a bounded, monotone transform of latent output — revenue-vs-quantity (TFPR/TFPQ) style.
- Venue: Semantic Scholar lists ICLR (year field 2024 = arXiv year; proceedings year likely 2025) [UNVERIFIED year].

### 2.9 Muennighoff et al. (2023, NeurIPS), "Scaling Data-Constrained Language Models" — `muennighoff2023scaling` (arXiv:2305.16264)
- >400 runs, 10M–9B params, up to 900B tokens, up to 1500 epochs; C4 (+ code, filtering ablations). Up to 4 epochs of repetition ≈ no loss change vs unique data; value of repetition decays toward zero.
- **Effective inputs (Eq. 5–6):** D′ = U_D + U_D R*_D (1 − e^{−R_D/R*_D}); N′ = U_N + U_N R*_N (1 − e^{−R_N/R*_N}); U_D = min(D_C, D) unique tokens, R_D = D/U_D − 1 repetitions; U_N = compute-optimal N for U_D, R_N = N/U_N − 1. For R_D ≪ R*_D, D′ ≈ D. L = E + A/N′^α + B/D′^β.
- Base law refit on C4 (App. B, Eq. 19–20) **with α = β tied** (because Chinchilla's C4 isoFLOPs gave a = b = 0.5): L(N,D) = 1.87 + 521/N^{0.353} + 1488/D^{0.353} (a = 6.255414, b = 7.3049974, e = 0.6254804, α = β = 0.3526596; 54 samples; δ = 1e−3; L-BFGS; Chinchilla grid).
- Decay constants (Eq. 16–17): fixing base parameters, fit R*_N, R*_D on 182 runs (7M–9B, 1–500 epochs; outliers where excess parameters/epochs *raise* loss removed): **R*_N = 5.309743, R*_D = 15.387756**. Final: L = 521/(U_N + 5.3 U_N(1 − e^{−R_N/5.3}))^{0.35} + 1488/(U_D + 15.4 U_D(1 − e^{−R_D/15.4}))^{0.35} + 1.87, with U_N = 0.051 U_D. R*_D ≈ 15 ⇒ repeated tokens lose 1/e of value around 16 epochs ("half-life of epochs"); R*_N < R*_D ⇒ excess parameters decay faster ⇒ spend extra compute on epochs before parameters.
- Data/models: github.com/huggingface/datablations (verified 200).
- **IO mapping:** repetition = depreciation / diminishing effective services of a reused input (effective capital services vs gross stock); data constraint = quantity-constrained factor (shadow price); outlier removal on the dependent variable = selection on outcomes (truncation). Strength: close (depreciation here is in *use*, not in time — closer to "utilization-dependent depreciation").

### 2.10 Bi et al. / DeepSeek-AI (2024), "DeepSeek LLM: Scaling Open-Source Language Models with Longtermism" — `bi2024deepseek` (arXiv:2401.02954)
- Hyperparameter scaling laws (Eq. 1), fitted on 1e17–2e19 FLOPs (near-optimal = within 0.25% of min generalization error): **η_opt = 0.3118 · C^{−0.1250}, B_opt = 0.2920 · C^{0.3271}**.
- Model scale measured as non-embedding FLOPs/token **M = 72 n_layer d_model² + 12 n_layer d_model l_seq** (Eq. 2) instead of 6N₁ = 72 n_layer d² (non-embedding) or 6N₂ (+6 n_vocab d); C = MD. Table 3: 6N₁/M from 0.43 (8 layers, d = 512) to 0.92 (80 layers, d = 8192); 6N₂/M from 1.32 to 0.94 — up to 50% mismeasurement at small scale.
- IsoFLOP: 8 budgets 1e17–3e20, ~10 allocations each; bits-per-byte on 100M-token validation set: **M_opt = 0.1715 · C^{0.5243}, D_opt = 5.8316 · C^{0.4757}** (Eq. 4); predicted 7B and 67B performance from 1000× smaller budgets.
- **Data quality changes the allocation exponent (Table 4):** early in-house data a = 0.450, b = 0.550; current in-house 0.524/0.476; OpenWebText2 0.578/0.422 (vs OpenAI-OWT2 0.73/0.27, Chinchilla-MassiveText 0.49/0.51). Higher quality ⇒ more compute to model size. Authors propose the allocation exponent as an indirect data-quality measure.
- **IO mapping:** (i) η_opt(C), B_opt(C) = flexible-input demand functions; (ii) M vs 6N = input measurement with scale-dependent error; (iii) quality → exponent = *non-neutral* technical change: within the Chinchilla family, a = β/(α+β) depends only on exponents, so a pure data-augmenting quality shift D → qD changes only B (hence G) not a [DERIVED]. A change in a requires quality to change β (or α) — i.e., quality alters the curvature/elasticity, not just efficiency units. This is a testable restriction ("is data quality factor-augmenting?").

### 2.11 Grattafiori et al. / Llama Team (2024), "The Llama 3 Herd of Models" — `grattafiori2024llama` (arXiv:2407.21783; often cited as Dubey et al. 2024)
- §3.2.1: IsoFLOP experiments at 6e18 to 1e22 FLOPs, models 40M–16B, cosine schedule, 2000-step warmup, peak LR 2e−4 to 4e−4; loss on separate validation set; second-degree polynomial per budget → minimum.
- Power law fit **N*(C) = A C^{α} with (α, A) = (0.53, 0.29)** (Fig. 3 label: α = 0.537, A = 0.299). **Careful: the text defines N*(C) as the optimal number of *training tokens*** and Fig. 3 plots training tokens; so D* ∝ C^{0.53}, N* ∝ C^{0.47}. Extrapolation to 3.8e25 FLOPs → 402B params on 16.55T tokens; chose 405B (trained on 15.6T text tokens; corpus ~15T).
- "IsoFLOPs curves become flatter around the minimum as the compute budget increases" ⇒ performance robust to allocation near optimum (envelope theorem; also means argmin weakly identified at high C).
- Downstream: two-stage — (1) linear relation between normalized NLL of the correct answer and log FLOPs (models up to 1e22); (2) sigmoidal map NLL → accuracy using both scaling-law models and Llama 2 models (different data mix & tokenizer). Extrapolates ~4 orders of magnitude on ARC-Challenge, slightly underestimating the flagship.
- Scaling-law experiments also used for data-mix selection; annealing on small high-quality data used to assess data quality.

### 2.12 Hu et al. (2024), "MiniCPM" — `hu2024minicpm` (arXiv:2404.06395)
- WSD (warmup–stable–decay) LR schedule; decaying from stable checkpoints at 10N…60N tokens gives O(mC) instead of O(m²C) scaling experiments. 6 model sizes 0.04B–2B; loss in bytes (tokenizer-comparable).
- Fit (Eq. 2, scipy curve_fit): L = C_N N^{−α} + C_D D^{−β} + L0; N_opt/D_opt = K²(C/6)^{η}, K = (αC_N/(βC_D))^{1/(α+β)}, η = (β−α)/(α+β). Average across 5 eval corpora: **α = 0.29, β = 0.23, K² = 0.01, η = −0.10**.
- **Compute-optimal D/N ≈ 192 on average (vs ~20 Chinchilla)**; rough re-estimate of Llama 2's implied ratio 70–100.
- **IO mapping:** 10× dispersion in optimal factor ratios across labs/recipes at similar exponents = heterogeneity in factor-augmenting efficiencies (A vs B), i.e., lab-specific "labor/capital-augmenting productivity" (Doraszelski–Jaumandreu), not just Hicks-neutral TFP.

### 2.13 DeepSeek-AI (2024), "DeepSeek-V3 Technical Report" — `deepseekai2024deepseekv3` (arXiv:2412.19437)
- MoE: 671B total, 37B activated per token; 14.8T tokens; FP8 mixed precision; multi-token prediction; 2.788M H800 GPU hours total ($5.576M at $2/GPU-hr; pre-training 2.664M hrs), excluding prior research/ablation costs.
- No new scaling law disclosed; useful as a data point: tokens per *active* param ≈ 400, per *total* param ≈ 22; training FLOPs ≈ 6 × 37e9 × 14.8e12 ≈ 3.3e24 [DERIVED, 6·N_active·D approximation].
- **IO mapping:** total vs active parameters = capital stock vs capital services (utilization); the "right" N in the production function is ambiguous for MoE — see Abnar/Clark.

### 2.14 Li et al. / StepFun (2025), "Predictable Scale: Part II, Farseer: A Refined Scaling Law in Large Language Models" — `li2025predictable` (arXiv:2506.10972)
- ~1,000 LLMs, ~3M H100 GPU hours; **L(N,D) = exp(a₃N^{γ} + b₃) + exp(a₂N^{β} + b₂) · D^{−exp(a₁N^{α} + b₁)}** (Eq. 3) — the data exponent is itself a function of N (non-separable; cross terms). General decomposition L = E + U(N) + V(D) + H(N,D). Claims large extrapolation-error reductions vs Chinchilla (their "433%"/"232%" phrasing).
- **Optimal D/N rises steadily with compute** (Chinchilla's ~20 valid only near 1e20–1e21), consistent with recent practice (Llama 3.1, Qwen3).
- Data/logs/code: github.com/Farseer-Scaling-Law/Farseer (verified 200) — large public sweep.
- **IO mapping:** translog-like interaction; non-homothetic technology (expansion path not a ray). Important confound for the "inference wedge": rising D/N can be compute-optimal.

---

## 3. Inputs beyond (N, D): architecture, data, precision, vocabulary, distillation

### 3.1 Clark et al. (2022, ICML), "Unified Scaling Laws for Routed Language Models" — `clark2022unified` (arXiv:2202.01169; ICML 2022 pp. 4057–4086 per Semantic Scholar)
- All models trained on 130B tokens (fixed D); N from ~15M to large, E experts up to 512; three routing methods (S-BASE, RL-R, HASH).
- **Law (Eq. 1): log L(N,E) = a log N + b log Ê + c log N log Ê + d**, with saturating transform 1/Ê = 1/(E − 1 + (1/E_start − 1/E_max)^{−1}) + 1/E_max. Slopes affine in the other log input (Eq. 8): −∂logL/∂logN = a + c log E; −∂logL/∂logE = b + c log N. Generalized (Eq. 2) in F (TFLOPs/forward pass) and B = P/F (parameter utilization ratio).
- Table 3 (Eq. 1 solutions): S-BASE a = −0.082, b = −0.108, c = 0.009, d = 1.104, E_start = 1.847, E_max = 314.478; RL-R −0.083, −0.126, 0.012, 1.111, 1.880, 469.982; HASH −0.087, −0.136, 0.012, 1.157, 4.175, 477.741. Dense: α_N = 0.078, N_c = 3.568e13 (vs Kaplan 0.076, 8.8e13).
- Effective Parameter Count N̄(N,E) (Eq. 11); routing stops helping at N_cutoff = 937B (S-BASE), 85B (RL-R), 83B (HASH) at 130B tokens. E.g., N = 5M with E = 128 ≈ dense 55M.
- **IO mapping (exact):** Eq. 7/Eq. 1 is a **translog production function** (Christensen–Jorgenson–Lau) in (log N, log Ê) without own-quadratic terms; c > 0 = the two inputs are substitutes-in-log-output (routing benefit decreases with size). EPC = quality-adjusted input index (hedonic capital). Strength: exact (functional form).

### 3.2 Krajewski et al. (2024, ICML), "Scaling Laws for Fine-Grained Mixture of Experts" — `krajewski2024scaling` (arXiv:2402.07871; ICML 2024, PMLR pp. 33270–33288 per Semantic Scholar)
- >100 experiments; 129M–3.7B params; 16B–130B tokens; granularity G ∈ [1,16] (log-spaced), expansion E = 64 fixed.
- **L(N,D,G) = c + (g/G^{γ} + a)/N^{α} + b/D^{β}** (Eq. 9); Huber δ = 0.1, BFGS, weight decay 5e−4; RMSE 0.015 (validation 0.019). Table 1: MoE a = 18.1, α = 0.115, b = 30.8, β = 0.147, g = 2.1, γ = 0.58, c = 0.47; Dense a = 16.3, α = 0.126, b = 26.7, β = 0.127, c = 0.47.
- Claims: compute-optimal MoE at 1e20 FLOPs matches dense at 20× compute; >40× beyond 1e25; standard G = 1 almost never optimal.
- **IO mapping:** granularity is a parameter-augmenting technology shifter (multiplies the N-term) — factor-augmenting technical change; note exponents here (≈0.12–0.15) are far below Chinchilla's (≈0.3), illustrating non-comparability of exponents across setups (units, E, ranges).

### 3.3 Abnar et al. (2025), "Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models" — `abnar2025parameters` (arXiv:2501.12370) [venue UNVERIFIED]
- Sparsity S = (E − K)/E; IsoFLOP surfaces over (N, S) and (N_a, S); optimal N rises and N_a falls with S; optimal sparsity → 1 as N grows (for fixed C).
- **L(N,D,S) = a/N^{α} + b/D^{β} + c/(1−S)^{λ} + d/((1−S)^{δ} N^{γ}) + e** (Eq. 6); Huber δ = 1e−3, L-BFGS, grid (Table 2). Table 3: α = 0.5962, β = 0.3954, λ = −0.1666, δ = 0.1603, γ = 0.1595, a = 16612.50, b = 5455.67, c = 0.4598, d = 17.26, e = 0.94. MSE 0.00056 (fit) / 0.0058 (held-out S = 0.98); R² 99% fit vs **68% held-out**.
- **IO mapping:** total parameters = capital stock, active parameters/FLOPs per token = capital services/utilization; the optimal utilization rate is an interior choice at fixed C. Weak out-of-sample fit is a warning about extrapolating a 10-parameter surface.

### 3.4 Frantar et al. (2023/2024), "Scaling Laws for Sparsely-Connected Foundation Models" — `frantar2023scaling` (arXiv:2309.08520; ICLR per Semantic Scholar venue field)
- **L(S,N,D) = (a_S(1−S)^{b_S} + c_S)·(1/N)^{b_N} + (a_D/D)^{b_D} + c** (Eq. 1), N = non-zero parameters; ViT/JFT-4B and T5/C4. Optimal sparsity rises with D at fixed non-zero N.
- **IO mapping:** sparsity enters as a multiplicative efficiency on the parameter term (parameter-augmenting).

### 3.5 Kumar et al. (2024), "Scaling Laws for Precision" — `kumar2024scaling` (arXiv:2411.04330) [venue UNVERIFIED]
- 465 pretraining runs, 3–16-bit, up to 1.7B params / 26B tokens.
- **Effective parameters:** N_eff(N, P_w) = N(1 − e^{−P_w/γ_w}); general N_eff = N(1 − e^{−P_w/γ_w})(1 − e^{−P_a/γ_a})(1 − e^{−P_kv/γ_kv}) (Eq. 4). L = A N_eff^{−α} + B D^{−β} + E + δ_PTQ (Eq. 1, 3).
- Post-training quantization degradation (Eq. 2): δ_PTQ(N,D,P_post) = C_T (D^{γ_D}/N^{γ_N}) e^{−P_post/γ_post}, γ_D ≈ γ_N ⇒ ≈ power law in D/N: over-trained models degrade more; beyond a critical D, more pretraining data *raises* post-quantization loss. Fit R² = 0.90 on >1000 points.
- With C ∝ N D P jointly optimized, optimal pretraining precision is independent of compute, **P* ≈ 7–8 bits** (integer); with N fixed, P*(C) ∝ log C. Compute-optimal N*, D* scale with [1 − e^{−P/γ̄}]^{∓3α/(α+β)} P^{∓β/(α+β)} (Eq. 6).
- **IO mapping:** precision = input quality; N_eff = quality-adjusted capital (quality/quantity trade-off with cost ∝ quantity × quality); "compute-optimal precision independent of scale" = homotheticity of the quality choice; δ_PTQ rising in D/N = interaction between over-training and downstream compression (complementarity between training intensity and fragility).

### 3.6 Tao et al. (2024, NeurIPS), "Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies" — `tao2024scaling` (arXiv:2407.13623)
- 33M–3B params, up to 500B characters; N = N_nv + N_v (N_v = V·d); data measured in characters H (tokens D = H f(V)); C ≈ 6(N_nv + Vd) H f(V).
- Loss normalized by unigram model (Eq. 4): L_u (vocabulary-comparable). Three approaches (IsoFLOP, derivative, parametric). Parametric: **L_u = −E + A₁/N_nv^{α₁} + A₂/N_v^{α₂} + B/H^{β}**, with α₁ = β imposed: A₁ = 1.831, A₂ = 0.196, B = 2.124, E = 5.533, α₁ = β = 0.447, α₂ = 0.671.
- N_v^opt ∝ N_nv^{γ}, γ ≈ 0.83 < 1; Llama2-70B optimal vocabulary ≥216K vs 32K used; 32K→43K raised ARC-C 29.1→32.0 at 2.3e21 FLOPs.
- Code: github.com/sail-sg/scaling-with-vocab (verified 200).
- **IO mapping (strong):** tokenizer defines the *unit of output and of data input*; per-token loss is not comparable across tokenizers ⇒ output-unit normalization (bits per byte / unigram-normalized) is like converting revenue to physical quantity with a common deflator. Also: vocabulary is a third input with its own elasticity.

### 3.7 Busbridge et al. (2025, ICML), "Distillation Scaling Laws" — `busbridge2025distillation` (arXiv:2502.08606)
- **Student loss (Eq. 8):** L_S(N_S, D_S, L_T) = L_T + L_T^{−c₀} (1 + (L_T/(L̃_S d₁))^{1/f₁})^{−c₁ f₁} (A/N_S^{α′} + B/D_S^{β′})^{γ′}, where L̃_S = supervised loss of the same student; capacity gap: too-strong teachers hurt (broken power law in L_T; transition at L_T/L̃_S = d₁). Fits ≲1% relative error.
- Cost (Eq. 9): FLOPs ≈ 3F(N_S)D_S + F(N_T)(δ^{Lgt}_T D_S + δ^{Pre}_T 3D_T), δ ∈ {0,1} for counting teacher logits/teacher training.
- Findings: with an existing teacher or many students, distillation beats supervised learning up to a compute level that scales with student size; if a teacher must be trained for a single student, supervised learning is generally preferable.
- **IO mapping (strong):** teacher = intermediate input (gross-output production; GNR 2020); δ^{Pre} toggles whether the intermediate is produced in-house (vertical integration) or treated as sunk/public — make-or-buy; many students sharing one teacher = economies of scope/fixed-cost amortization. Capacity gap = non-monotone input-quality effect.

### 3.8 Goyal et al. (2024, CVPR), "Scaling Laws for Data Filtering — Data Curation cannot be Compute Agnostic" — `goyal2024scaling` (arXiv:2404.07177)
- CLIP-style pretraining on web pools of differing quality. Base: y = a n^{b} + d; "utility" = exponent b (more negative = higher quality). Repetition decays utility (Eq. 2): b_{k+1} = b (1/2)^{k/τ} = b δ^{k}, τ = half-life; loss after k passes (Eq. 3): y_k = a n₁^{b₁} Π_{j=2..k} (n_j/n_{j−1})^{b_j} + d. Mixture of p pools (Thm. 1, Eq. 4): τ̂ = pτ; b_eff^{(k)} = Σ_i b_i δ̂_i^{k}/p.
- Quality–quantity trade-off (QQT): the best filtering level depends on compute (aggressive filtering good at low compute, bad at high compute).
- **IO mapping:** input quality enters the *exponent* (non-neutral), and depreciation acts on the exponent (not on effective quantity as in Muennighoff) — two competing parameterizations of input depreciation that are observationally distinguishable only with repetition variation.

### 3.9 Sorscher et al. (2022, NeurIPS outstanding paper), "Beyond neural scaling laws: beating power law scaling via data pruning" — `sorscher2022beyond` (arXiv:2206.14486)
- Theory (teacher–student perceptron, statistical mechanics) + ResNets on CIFAR-10/SVHN/ImageNet: with a good pruning metric, error vs *pruned* dataset size can fall faster than a power law (toward exponential); benchmarking 10 pruning metrics on ImageNet; new self-supervised metric.
- **IO mapping:** selecting the highest-quality units of an input changes the functional form (curvature), not just efficiency units — again non-augmenting quality.

### 3.10 Data mixtures — Ye et al. (2024/2025, ICLR) `ye2024data` (arXiv:2403.16952) and Shukor et al. (2025) `shukor2025scaling` (arXiv:2507.09404)
- Ye et al.: "data mixing laws" predicting loss as a function of domain proportions (read via arXiv metadata only; details not extracted — [UNVERIFIED specifics]).
- Shukor et al.: L(N, D, h) with domain-weight vector h; *additive* law (Eq. 2.4) implies optimal h* independent of N and D; *joint* law (Eq. 2.5) lets the N and D contributions depend on h so optimal mixture is compute-dependent. Validated for LLMs, native multimodal models and vision models (models 106M–8B).
- **IO mapping:** data mixture = composition of a composite input (a CES/Leontief aggregate of data "varieties"); additive vs joint law = separability test (is the optimal composition homothetic?).

### 3.11 Magnusson et al. (2025, ICML), "DataDecide: How to Predict Best Pretraining Data with Small Experiments" — `magnusson2025datadecide` (arXiv:2504.11393)
- Open suite: **25 data recipes (corpora) × 14 model sizes (4M–1B) × 3 seeds, up to 100B tokens, >30K checkpoints**, 10 downstream tasks. Single-scale ranking at 150M predicts 1B pairwise winners ~80% of the time; no scaling-law method among 8 baselines beats the compute–decision frontier of single-scale predictions; continuous likelihood proxies make MMLU/ARC/HellaSwag/MBPP/HumanEval >80% predictable at 0.01% of compute.
- Data: huggingface.co/collections/allenai/datadecide-67edb1d2bacba40b5d3ed633; github.com/allenai/DataDecide (both verified 200).
- **IO mapping:** a *balanced experimental panel* of "technologies" (data recipes) × scale × seed — ideal for estimating recipe-specific TFP and testing Hicks-neutrality (do recipes shift intercepts or slopes?).

---

## 4. CES interpretation — derivations and numbers [DERIVED unless cited]

### 4.1 Chinchilla form
Let ℓ(N,D) = A N^{−α} + B D^{−β}, u = αAN^{−α} = −∂ℓ/∂ln N, v = βBD^{−β} = −∂ℓ/∂ln D, s_N = u/(u+v).
- MRTS_{N,D} = (∂ℓ/∂N)/(∂ℓ/∂D) = (u/v)(D/N). Along an isoquant u d ln N + v d ln D = 0.
- d ln(D/N) = −(1 + u/v) d ln N; d ln MRTS = −[(1+α) + (1+β)(u/v)] d ln N.
- **σ = d ln(D/N)/d ln MRTS = (u+v)/[(1+α)v + (1+β)u], i.e. 1/σ = 1 + β s_N + α s_D.**
- Special case α = β = ρ: σ = 1/(1+ρ) everywhere; Y = ℓ^{−1/ρ} = (AN^{−ρ} + BD^{−ρ})^{−1/ρ} is CES, HD1. ρ > 0 ⇒ σ < 1 (complements).
- Bounds: σ ∈ [1/(1+max(α,β)), 1/(1+min(α,β))]. At the compute optimum (u = v): σ* = 1/(1 + (α+β)/2).
- Homotheticity: along a ray D = kN, MRTS ∝ N^{β−α} ⇒ homothetic iff α = β. With α ≠ β the expansion path is D*/N* = G^{−2}(C/6)^{b−a}: rising in C if α > β (Hoffmann A3), falling if α < β (Besiroglu).
- Returns to scale depend on the cardinalization of output (loss is ordinal for isoquant purposes). With Y = 1/ℓ, local RTS = εN + εD = (αAN^{−α} + βBD^{−β})/ℓ ∈ [min(α,β), max(α,β)] ≈ 0.28–0.37 (strongly decreasing). With Y = ℓ^{−1/ρ} and α = β: exactly CRS. σ is invariant to monotone transforms of output; RTS and "TFP levels" are not ⇒ **σ and the expansion path are the robust, cardinalization-free objects; returns to scale are not identified without a cardinal output (e.g., downstream value).**
- Cost function: min_{N,D} 6ND s.t. ℓ(N,D) = ℓ̄. Log-linear "prices" ⇒ FOC u = v ⇒ N* = G(C/6)^{a}, D* = G^{−1}(C/6)^{b} (Hoffmann Eq. 4) and **ℓ*(C) = (AG^{−α} + BG^{β})(C/6)^{−γ}, γ = αβ/(α+β)** ⇒ inverse cost function C(ℓ̄) ∝ ℓ̄^{−1/γ}; cost elasticity w.r.t. "output" Y = 1/ℓ is 1/γ ≈ 6 (Hoffmann) — massive diseconomies in loss units, which again is cardinalization-dependent.
- At the optimum, reducible-loss share from finite N = AN^{−α}/ℓ = β/(α+β) = a.
- **Identification from frontier + FOC:** α = γ/a, β = γ/b. Check: Hoffmann γ = 0.1548, a = 0.456 ⇒ α = 0.339 ✓.

### 4.2 Numbers (computed from published parameters)
| Spec | α | β | a = β/(α+β) | γ = αβ/(α+β) | σ* | σ range |
|---|---|---|---|---|---|---|
| Hoffmann A3 rounded | 0.34 | 0.28 | 0.452 | 0.1535 | 0.763 | 0.746–0.781 |
| Hoffmann A3 TeX | 0.3392 | 0.2849 | 0.456 | 0.1548 | 0.762 | 0.747–0.778 |
| Besiroglu refit | 0.3478 | 0.3658 | 0.513 | 0.1783 | 0.737 | 0.732–0.742 |
| Muennighoff C4 (α=β) | 0.3527 | 0.3527 | 0.500 | 0.1763 | 0.739 | 0.739 |
| Krajewski dense | 0.126 | 0.127 | 0.502 | 0.0632 | 0.888 | 0.887–0.888 |
| Sardana all-data | 0.18 | 0.24 | 0.571 | 0.1029 | 0.826 | 0.806–0.847 |
| MiniCPM avg | 0.29 | 0.23 | 0.442 | 0.1283 | 0.794 | 0.775–0.813 |
| Tao (α₁=β) | 0.447 | 0.447 | 0.500 | 0.2235 | 0.691 | 0.691 |
| Abnar MoE | 0.5962 | 0.3954 | 0.399 | 0.2377 | 0.669 | 0.626–0.717 |
| Kaplan L(N,D) Tab. 2 (isoquant exps 0.738, 1) | — | — | 0.575* | — | 0.535 | 0.500–0.575 |
(*Kaplan "a" if C = 6ND misapplied; see §1.3.) Cross-study σ ranges 0.54–0.89: σ is *not* a universal constant, but within the Chinchilla-style single-epoch regime with E estimated it clusters at 0.74–0.78.

Inference-wedge illustration [DERIVED; εN/εD evaluated at the released (N, D); implied D_inf/D_tr = 3(εN/εD − 1); heroic because the parameters were estimated on MassiveText/Chinchilla tokenizer and are applied to other labs' models — use only to show sensitivity]:
| Model (N / D) | D/N | εN/εD (Hoffmann TeX) | implied D_inf/D_tr | εN/εD (Besiroglu) | implied D_inf/D_tr |
|---|---|---|---|---|---|
| GPT-3 175B / 300B | 1.7 | 0.34 | −1.99 | 0.43 | −1.72 |
| Gopher 280B / 300B | 1.1 | 0.29 | −2.14 | 0.36 | −1.91 |
| Chinchilla 70B / 1.4T | 20 | 0.71 | −0.86 | 1.03 | 0.09 |
| Llama 2 70B / 2T | 28.6 | 0.79 | −0.63 | 1.17 | 0.52 |
| Llama 3 8B / 15T | 1875 | 2.92 | 5.77 | 5.22 | 12.65 |
| Llama 3 70B / 15T | 214 | 1.40 | 1.20 | 2.45 | 4.36 |
| Llama 3 405B / 15.6T | 38.5 | 0.78 | −0.66 | 1.35 | 1.06 |
| DeepSeek-V3 37B active / 14.8T | 400 | 1.73 | 2.20 | 3.05 | 6.14 |
Negative values = allocations that no training+inference cost-minimization rationalizes (the Kaplan-era "under-trained" models), i.e., a misallocation wedge. Llama 3 8B/70B token counts use the ~15T corpus figure (only the 405B's 15.6T is stated explicitly in the text read).

Compute-optimal allocations: Hoffmann TeX: C = 1e21 → N* = 2.2e9, D/N = 34; 5.76e23 → N* = 4.0e10, D/N = 59; 3.8e25 → N* = 2.7e11, D/N = 85. Besiroglu: 1e21 → 2.8e9, D/N = 21.6; 5.76e23 → 7.2e10, D/N = 18.4 (≈ actual Chinchilla 70B/1.4T); 3.8e25 → 6.2e11, D/N = 16.5. Muennighoff: D/N ≡ 19.6.

### 4.3 Kaplan form and a generalized nesting
- Kaplan: L = [(N_c/N)^{α_N/α_D} + D_c/D]^{α_D}. Isoquants: (N_c/N)^{α′} + D_c/D = k, α′ = α_N/α_D — the Chinchilla isoquant family with exponents (α′, 1) and E = 0. σ ∈ [1/2, 1/(1+α′)] = [0.50, 0.575] (Tab. 2) — markedly more complementary than Chinchilla.
- **Nesting family:** L = E + [(A N^{−α})^{1/q} + (B D^{−β})^{1/q}]^{q}.
  - q = 1: Chinchilla/Rosenfeld.
  - E = 0, q = α_D, α = α_N, β = α_D, A = N_c^{α_N}, B = D_c^{α_D}: Kaplan Eq. 1.5 exactly.
  - E = 0, general q: Kaplan's footnote-4 alternative [(N_c/N)^{α_N} + (D_c/D)^{α_D}]^{β}.
  - q → 0: L → E + max(AN^{−α}, BD^{−β}) (pure bottleneck/Leontief in the two "shortfall" terms).
  - Isoquants depend only on (α/q, β/q) ⇒ σ ∈ [1/(1+max(α,β)/q), 1/(1+min(α,β)/q)]; q governs how the two bottlenecks combine, separately identified only from strongly unbalanced (N,D) designs (very over- and under-trained runs). Formally this is a CES aggregator (with parameter 1/q) of two power-law "shortfall" terms plus a location shift E — a "generalized CES" in the sense of nested CES with power-transformed inputs.
- Output cardinalization matters for exponents: Kaplan's α_N ≈ 0.076 is the log-slope of *total* loss (E = 0); under Chinchilla parameters the total-loss log-slope at N = 1e9, D = 2e10 is α·AN^{−α}/L = 0.34 × 0.354 / 2.58 ≈ 0.047 [DERIVED; rounded Hoffmann parameters], i.e., the same order as Kaplan's — "0.076 vs 0.34" is mainly an output-measure (E) difference, not a technology difference.

### 4.4 Farseer, Clark, Abnar as translog-type generalizations
- Clark: exact translog without own-quadratic terms in (log N, log Ê).
- Farseer: log-elasticity of loss w.r.t. D equals exp(a₁N^{α} + b₁), a smooth function of N — a varying-coefficient (random-coefficient-like) production function.
- Abnar: additive terms plus an N×(1−S) interaction.
- All three break additive separability of ℓ, i.e., allow cross-partial ∂²ℓ/∂N∂D ≠ 0, which Chinchilla rules out (in Chinchilla complementarity comes only through the level/curvature of isoquants, not cross-partials of ℓ).

---

## 5. Test-time/inference compute and RL compute (a second pair of inputs)

### 5.1 Jones (2021), "Scaling Scaling Laws with Board Games" — `jones2021scaling` (arXiv:2104.03113)
- AlphaZero on Hex (board sizes up to 9×9); compute frontier Elo(C, board size): plateau = m_plateau·boardsize + c; incline = m_incline·boardsize + m_flops·log C + c (Table III: incline m_flops = 510, m_boardsize = −430, c = −4400; plateau m_boardsize = −270, c = 570); slope ≈ 500 Elo per 10× compute (2× compute to win 2/3 of games); perfect-play compute 7× per board-size increment; takeoff compute 4× per increment.
- **Train/test isoquant (Fig. 9): log10(test) = −1.2 · log10(train) + 0.004 · Elo + 29** (9×9 board): "for each additional 10× of train-time compute, about 15× of test-time compute can be eliminated", down to a 1-node floor.
- **IO mapping (exact):** isoquants linear in logs with constant slope ⇒ **Cobb–Douglas in (train, test) compute, σ = 1**, output Elo = 250·log10(test) + 300·log10(train) + const (rearranging) ⇒ train elasticity/test elasticity = 1.2; corner solution at 1 node. Code/data: github.com/andyljones/boardlaw (verified 200; the paper's footnote link).

### 5.2 Snell et al. (2024), "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters" — `snell2024scaling` (arXiv:2408.03314; ICLR 2025 Oral per OpenReview, retitled "...than Scaling Parameters for Reasoning")
- PaLM-2 variants fine-tuned for revision and PRM verification on MATH (12k train/500 test). Compute-optimal per-prompt allocation using difficulty bins (5 bins from base-model pass@1) improves test-time efficiency >4× vs best-of-N. FLOPs-matched: small model + test-time compute can beat a ~14× larger model on easy/medium questions; with R = D_inference/D_pretrain ∈ {0.16, 0.79, 22}, pretraining wins on hard questions or high inference load.
- **IO mapping:** substitution between pretraining and test-time compute is *heterogeneous across "products"* (question difficulty); R plays the role of relative price/quantity; the aggregate elasticity is a mixture — Houthakker-type aggregation again.

### 5.3 Brown et al. (2024), "Large Language Monkeys: Scaling Inference Compute with Repeated Sampling" — `brown2024large` (arXiv:2407.21787)
- Coverage (pass@k) vs samples k often log-linear over 4 orders of magnitude; fit **c ≈ exp(a k^{b})** ("exponentiated power law"). SWE-bench Lite with DeepSeek-Coder-V2-Instruct: 15.9% (1 sample) → 56% (250), beating single-sample SOTA 43%; CodeContests Gemma-2B 0.02% → 7.1% at 10,000 samples. Without verifiers, majority vote/reward models plateau after a few hundred samples. On MATH, a smaller model can maximize coverage at fixed FLOPs.
- Data: huggingface.co/datasets/ScalingIntelligence/monkey_business; code github.com/ScalingIntelligence/large_language_monkeys (from paper).
- **IO mapping:** coverage is a bounded output (probability) — a "success probability production function"; verification = a complementary input without which sample-scaling saturates.

### 5.4 Wu et al. (2025, ICLR), "Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for Problem-Solving with Language Models" — `wu2024inference` (arXiv:2408.00724)
- Greedy, majority voting, best-of-n, weighted voting, MCTS, and new REBASE tree search across Pythia/Llemma sizes; voting accuracy saturates at a limit set by model output probabilities; smaller models are compute-optimal at small inference budgets, larger ones once small models saturate (e.g., Llemma-7B beats 34B until ~128 samples); Llemma-7B + REBASE beats Llemma-34B at all budgets on MATH; ~2× fewer FLOPs for comparable accuracy.
- **IO mapping:** the optimal (model size, samples) mix shifts with the inference budget — an expansion path in the inference-input plane; saturation = bounded output.

### 5.5 Khatri et al. (2025), "The Art of Scaling Reinforcement Learning Compute for LLMs" — `khatri2025art` (arXiv:2510.13786)
- >400,000 GPU-hours (GB200); sigmoidal compute–performance curves **R_C − R₀ = (A − R₀)/(1 + (C_mid/C)^{B})** (Eq. 1): A = asymptotic pass rate (ceiling), B = efficiency exponent, C_mid = midpoint. Most design choices (loss aggregation, normalization, curriculum, off-policy algorithm) shift efficiency (B, C_mid) but not A; ScaleRL reaches A = 0.61; single run extrapolated from 50k to 100k GPU-hours (8B dense), 16k→45k (17B×16 MoE). Cites DeepSeek-R1 RL ≈ 100,000 H800 GPU-hours ≈ 3.75% of its pre-training compute.
- **IO mapping (strong):** input-augmenting technical change (shifts C_mid) vs frontier-shifting change (raises A) — the distinction between efficiency improvements and changes in the attainable frontier (stochastic-frontier language: inefficiency vs frontier).

---

## 6. Downstream capabilities and output measurement

- **Schaeffer, Miranda, Koyejo (2023, NeurIPS 36, pp. 55565–55581), "Are Emergent Abilities of Large Language Models a Mirage?"** — `schaeffer2023emergent` (arXiv:2304.15004; DOI 10.52202/075280-2425 via OpenAlex/proceedings.com): nonlinear/discontinuous metrics (exact match, multiple-choice grade) generate apparent emergence; linear/continuous metrics give smooth predictable curves (InstructGPT/GPT-3 family; BIG-Bench meta-analysis; induced "emergence" in vision). ⇒ Output-measurement artifact; the TFPR/TFPQ analogy (a nonlinear price/metric transform of latent quantity).
- **Ruan, Maddison, Hashimoto (2024, NeurIPS spotlight), "Observational Scaling Laws and the Predictability of Language Model Performance"** — `ruan2024observational` (arXiv:2405.10938): ~100 public models (core PCA set: 77 base models in 21 families); benchmark matrix B low-rank — top 3 PCs explain ~97% of variance, PC1 alone nearly 80% (PC1 "general", PC2 "reasoning", PC3 "programming"); **model:** σ^{−1}(E_m) ≈ βᵀS_m + α (Eq. 3); **S_m ≈ θ_f log C_m + ν_f (Eq. 4)** — "model families only vary in their efficiency in converting training compute to capabilities"; B_{i,m} ≈ γ_iᵀS_m (Eq. 5). "f-equivalent FLOPs" C̄_{m,f} maps any model into a reference family's compute units. Predicts emergent phenomena, agentic performance (GPT-4) from small models, and post-training (CoT, self-consistency) effects. Code/data: github.com/ryoungj/ObsScaling (verified 200).
  - **IO mapping (exact/strong):** Eq. 4 is a production function with family(firm)-specific TFP ν_f and possibly heterogeneous elasticities θ_f; S_m is latent output measured by multiple noisy indicators (factor model — cf. Cunha–Heckman–Schennach latent-skill measurement systems); f-equivalent FLOPs = TFP-adjusted input index ("effective compute"); identification uses within-family variation (fixed effects) — vulnerable to within-family simultaneity if larger siblings get better data/recipes.
- **Isik et al. (2025, ICLR), "Scaling Laws for Downstream Task Performance [in Machine Translation]"** — `isik2024scaling` (arXiv:2402.04177): downstream cross-entropy improves monotonically with pretraining data, but BLEU/COMET can fluctuate or worsen under moderate distribution misalignment; with alignment, a log-law predicts translation quality. ⇒ Divergence between latent (loss) and measured (task metric) outputs depends on "product–market fit" (alignment).
- **Bhagia et al. (2025, COLM), "Establishing Task Scaling Laws via Compute-Efficient Model Ladders"** — `bhagia2024establishing` (arXiv:2412.04403): two-step (N,D) → task loss via L = A/N^α + B/D^β + E (Eq. 1) → accuracy via sigmoid (Eq. 2); ladder of small models predicts 7B/13B OLMo-style over-trained targets.
- **Lourie, Hu, Cho (2025, EMNLP Findings), "Scaling Laws Are Unreliable for Downstream Tasks: A Reality Check"** — `lourie2025scaling` (arXiv:2507.00885): meta-analysis; predictable (linear after transformation) downstream scaling in only **39%** of tasks (18 of the tasks examined); benign setup changes can flip the scaling behavior.
- Gadre (Eq. 5) and Llama 3 (two-stage NLL→sigmoid) are the other standard latent-output links.

---

## 7. Theory of scaling exponents

- **Sharma & Kaplan (2020/2022), "A Neural Scaling Law from the Dimension of the Data Manifold"** — `sharma2020neural` (arXiv:2004.10802) [JMLR 2022 per common citation — UNVERIFIED from fetched metadata]: if networks perform regression on a data manifold of intrinsic dimension d, α ≈ 4/d for cross-entropy and MSE; verified in teacher–student settings, CNNs, GPT-type LMs (LM α ≈ 0.076 ↔ large d). ⇒ Exponent = a property of the "task/product", not of the lab — supports treating exponents as common technology parameters and intercepts as lab efficiency.
- **Bahri, Dyer, Kaplan, Lee, Sharma (2024, PNAS 121(27): e2311878121), "Explaining Neural Scaling Laws"** — `bahri2024explaining` (arXiv:2102.06701; DOI 10.1073/pnas.2311878121 from arXiv metadata): four regimes — variance-limited (exponent 1 in D or width-related P) and resolution-limited (α ∝ 1/d) for each of data and model size; duality between large-width and large-data resolution-limited exponents; random-feature/kernel spectrum derivations.
- **Michaud, Liu, Girit, Tegmark (2023, NeurIPS), "The Quantization Model of Neural Scaling"** — `michaud2023quantization` (arXiv:2303.13506): knowledge in discrete quanta with Zipfian use frequencies p_k ∝ k^{−(α+1)}; learning quanta in frequency order ⇒ L(N) − L∞ ∝ N^{−α} (α_N = α), L(D) − L∞ ∝ D^{−α/(α+1)} (α_D = α/(α+1)), α_S = α/(α+1) (so multi-epoch and single-epoch data exponents coincide). Explains both smooth aggregate scaling and emergence of individual skills.
  - **[DERIVED] implications:** cross-equation restriction β = α/(1+α); compute-optimal a = β/(α+β) = 1/(α+2) (≈ 0.43 for α = 0.34) and γ = α/(α+2). With Chinchilla α = 0.34, predicted β = 0.254 (Hoffmann 0.285; Besiroglu 0.366) — Besiroglu's refit rejects the restriction; Hoffmann's does not strongly.
  - **IO mapping (strong, novel):** Houthakker (1955) derives an aggregate Cobb–Douglas from Leontief activities with Pareto-distributed input coefficients; Jones (2005, QJE) derives Cobb–Douglas and long-run labor-augmenting change from Pareto-distributed ideas. The quantization model is the same aggregation theorem with "skills" in place of "activities/ideas".
- **Schaeffer et al. (2025), "How Do Large Language Monkeys Get Their Power (Laws)?"** — `schaeffer2025large` (arXiv:2502.17578): per problem, failure rate under k independent attempts falls *exponentially* in k; the aggregate −log(average success) is a *power law* in k because the distribution of single-attempt success probabilities across problems is heavy-tailed (a small fraction of very hard problems warps the aggregate); the distributional view forecasts the exponent with ~an order of magnitude lower relative error (≈2–4 orders of magnitude less inference compute). **IO mapping (strong):** a second, cleaner instance of Houthakker aggregation — exponential micro "production" + heavy-tailed heterogeneity ⇒ power-law macro production; the aggregate exponent is a property of the heterogeneity distribution, not of any micro unit.
- **Maloney, Roberts, Sully (2022), "A Solvable Model of Neural Scaling Laws"** — `maloney2022solvable` (arXiv:2210.16859): generative data model with power-law spectrum + random feature model; solved in the joint large-N, large-P limit; nonlinear feature maps extend the data spectrum's power law; **equiparameterization** (features ∝ training set size) optimal; finite latent-space size produces plateaus/breakdown of scaling.
- **Paquette, Paquette, Xiao, Pennington (2024, NeurIPS 37, pp. 16459–16537), "4+3 Phases of Compute-Optimal Neural Scaling Laws"** — `paquette2024phases` (arXiv:2405.15074; DOI 10.52202/079017-0526 via OpenAlex/proceedings.com): power-law random features with data complexity α, target complexity β, parameter count d; one-pass SGD; 4 phases (+3 subphases) in the (α,β) plane determined by model capacity, optimizer noise and feature embedding; over a large region d*(f) = f^{1/2} ("universal Chinchilla"); numerics d* ≍ f^{0.508}, f^{0.525} at the Phase II–III boundary. ⇒ The Chinchilla 0.5 has a theoretical attractor; deviations are phase-dependent (algorithm-constrained phases III/IV mean the optimizer, not the data, determines the frontier — "technology includes the optimizer").
- **Caballero, Gupta, Rish, Krueger (2023, ICLR), "Broken Neural Scaling Laws"** — `caballero2023broken` (arXiv:2210.14891): y = a + b x^{−c₀} Π_{i=1}^{n} (1 + (x/d_i)^{1/f_i})^{−c_i f_i}; smoothly broken power law; fits double descent, sharp inflections (arithmetic), many domains; SciPy curve fitting. ⇒ Piecewise-log-linear (spline-like) production functions; extrapolation risk from undetected breaks (structural-break analogy).
- **Alabdulmohsin, Neyshabur, Zhai (2022, NeurIPS), "Revisiting Neural Scaling Laws in Language and Vision"** — `alabdulmohsin2022revisiting` (arXiv:2209.06640): estimator families M1 ε = βx^c; M2 ε − ε∞ = βx^c; M3 ε = β(x^{−1} + γ)^{−c}; **M4 (ε_x − ε∞)/(ε₀ − ε_x)^{α} = βx^{c}**, fitted in log space; argue for *extrapolation loss* (out-of-sample) rather than in-sample fit; toy example: exponent estimates c from −0.09 (M1) to −0.57 (M4) on the same data; releases 90-task benchmark. ⇒ Functional-form dependence of the "elasticity"; out-of-sample validation as the model-selection criterion (cf. cross-validation in Ho et al.).

---

## 8. Progress/TFP measures (cross-strand; brief)

- **Ho et al. (2024), "Algorithmic progress in language models"** — `ho2024algorithmic` (arXiv:2403.05812) [venue UNVERIFIED]: ~231 models (WikiText, PTB; 2012–2023); augmented law (Eq. 3) L = E + A/(N^{α_param} e^{α_year(Y−Y0)}) + B/(D^{β_data} e^{β_year(Y−Y0)}) (E set to 0 in preferred variants because "poorly estimated"); N_eff = N exp(α′(Y−Y0)), D_eff = D exp(β′(Y−Y0)), α′ = α_year/α_param, β′ = β_year/β_data; T_C = (1/T_N + 1/T_D)^{−1} from g_C = g_N + g_D (because C ≈ 6ND). **Effective-compute doubling time 8.4 months (95% CI 4.5–14.3)**; ~90 specifications compared by leave-one-out CV; preferred model 7 (benchmark-specific A, B; common exponents and progress rates), held-out R² ≈ 0.91. Transformer: reduces reducible loss by 4.6% [3.0, 7.0]; compute-equivalent gain 7.2× [3.3, 45.7]. Explicit caveat: algorithmic improvements and scaling introduced concurrently ⇒ hard to disentangle (collinearity/simultaneity). Data/code: github.com/epoch-research/lm-algorithmic-progress.
  - **IO mapping (exact):** factor-augmenting technical change with constant exponential rates (Doraszelski–Jaumandreu 2018 labor-augmenting; Raval 2019/2023 non-neutral). Diamond–McFadden–Rodriguez: time-series data on frontier models alone cannot separate σ from bias — here, sweeps at a fixed date can.
- **Xiao et al. (2024/2025), "Densing Law of LLMs"** — `xiao2024densing` (arXiv:2412.04315; journal version Nature Machine Intelligence 7(11):1823–1833, 2025, DOI 10.1038/s42256-025-01137-0 per OpenAlex; arXiv version read): capability density ρ = N_eff/N, N_eff = parameters a *reference* model family needs to match performance on 5 benchmarks; **ln ρ_max = A t + B, A ≈ 0.007/day (R² ≈ 0.93) ⇒ max density doubles ≈ every 3.3 months** (open-source base LLMs since 2023). ⇒ Frontier (best-practice) parameter-augmenting productivity index relative to a reference technology; subject to benchmark contamination and output-measure issues.

---

## 9. Econometric pitfalls catalogue (for the "IO lens" section)

1. **Functional dependence on the frontier** (Chinchilla Eq. 1 & App. E; Approach 1 envelope): (ln N, ln D) collinear given C; identification requires IsoFLOP sweeps or the optimality FOC.
2. **Envelope/argmin weak identification:** IsoFLOP curves flatten at large C (Llama 3) ⇒ small loss differences, large uncertainty in N*; Approach 2's parabola argmin inherits noise ~ seed variance (≈0.02 nats per Kaplan; up to 4% per Choshen).
3. **Winner's-curse selection in Approach 1:** taking the min-loss run at each FLOP value over many curves selects favorable noise draws ⇒ downward-biased frontier and argmin biased toward noisier configurations (extreme-value selection; cf. DEA/frontier estimation).
4. **Selection on outcomes:** diverged/failed runs and "outliers" (Muennighoff: runs where more epochs hurt; Besiroglu: 5 low tokens/param points; Chinchilla: Huber down-weighting low-compute points) dropped ⇒ truncation.
5. **Input measurement error:** embedding vs non-embedding N (Pearce–Song), last-layer FLOPs (Porian), 6N vs M (DeepSeek; 6N₁/M ∈ [0.43, 0.92]), total vs active params (MoE), precision (N_eff), repetitions (D′), data quality (effective data). Errors are *scale-dependent* ⇒ non-classical ⇒ biased elasticities, not mere attenuation.
6. **Output measurement:** tokenizer/eval-set dependence of loss; E (irreducible) as a location parameter changes exponents; downstream metrics bounded/sigmoidal; seed noise; digitization noise (Besiroglu ±0.01).
7. **Flexible inputs (hyperparameters) not at optimum and correlated with scale** (Porian; Kaplan's fixed schedule per Hoffmann) ⇒ omitted-variable bias; the structural object is the concentrated L(N,D) = min_h L(N,D,h).
8. **Sample-support dependence / misspecification:** Sardana Table 1; Chinchilla App. E curvature; Farseer; Abnar held-out R² 68% ⇒ extrapolation beyond the support of token ratios is unreliable; exponents are local elasticities.
9. **Numerical optimization & inference:** non-convex NLS with LSE; grid of starts; averaged vs summed losses changes stopping (Besiroglu); Huber δ = 1e−3 on log loss ≈ LAD in logs for essentially all points (residuals ≫ 1e−3), δ = 0.1 (Krajewski) ≈ least squares; standard errors rarely reported; bootstrap over runs ignores within-family dependence.
10. **Observational (cross-lab) data:** simultaneity (better labs scale more), family effects, concurrent algorithmic progress (Ho et al. caveat), contamination; within-family identification (Ruan) requires within-family exogeneity.
11. **Non-comparability of exponents across studies:** units (tokens vs bytes vs characters), E handling, parameter definitions, schedules — e.g., α ranges 0.076 (Kaplan, E = 0) … 0.126 (Krajewski dense) … 0.34 (Chinchilla) … 0.596 (Abnar). Compare σ, a, or γ within a consistent protocol only.

---

## 10. ML ↔ IO dictionary (this strand's contribution; strength in brackets)

| ML concept | IO/econometrics concept | Strength | Note |
|---|---|---|---|
| Chinchilla reducible loss AN^{−α} + BD^{−β} with α = β | CES production function (Arrow–Chenery–Minhas–Solow), Y = ℓ^{−1/ρ}, σ = 1/(1+ρ) | exact | Gadre and Muennighoff impose α = β explicitly |
| Chinchilla with α ≠ β | non-homothetic generalized CES; σ = 1/(1 + βs_N + αs_D) ∈ [0.747, 0.778] | close | σ nearly constant; homotheticity fails |
| Kaplan L(N,D) Eq. 1.5 | nested generalized-CES (outer exponent q = α_D), σ ∈ [0.50, 0.58] | close | Kaplan's D is early-stopped multi-epoch data |
| Clark routed law log L = a log N + b log Ê + c log N log Ê + d | translog (Christensen–Jorgenson–Lau) | exact | no own-quadratic terms |
| IsoFLOP curve | isocost curve (Cobb–Douglas cost 6ND ⇒ linear in logs) | exact | "prices" are unit elasticities of the cost function |
| Iso-loss contour | isoquant | exact | |
| Compute-optimal frontier N*(C), D*(C) | expansion path / conditional factor demands | exact | power laws with exponents β/(α+β), α/(α+β) |
| L*(C) loss–compute law | (inverse) cost function; Nerlove-type cost-function estimation | exact | γ = αβ/(α+β) |
| C ≈ 6ND | Cobb–Douglas "budget" (compute = parameter-token pairs) | close | unusual: cost is multiplicative in inputs |
| Compute-optimal FOC εN = εD | cost-minimization FOC equating output elasticities to cost elasticities (GNR/DLW first-order condition) | exact | used for identification: α = γ/a, β = γ/b |
| Only frontier models observed | functional dependence (ACF 2015) / collinearity | exact | |
| IsoFLOP sweep | designed/randomized input-ratio variation (experiment) | close | breaks functional dependence; rare in IO |
| Hyperparameters (LR, batch, warmup, β2) | flexible/static inputs (materials) | close | L(N,D) is the concentrated production function |
| DeepSeek η_opt(C), B_opt(C) | flexible-input demand functions | close | |
| Mis-tuned hyperparameters correlated with scale (Porian) | transmission/omitted-variable bias | close | a moves 0.84 → 0.50 |
| Embedding vs non-embedding N; 6N vs M; last-layer FLOPs | non-classical input measurement error | exact | error share falls with scale |
| Irreducible loss E | location parameter / output cardinalization; entropy floor | loose | changes exponents drastically |
| Seed noise; eval-set noise | output measurement error / idiosyncratic shock | close | |
| Benchmark accuracy via sigmoid of loss | bounded nonlinear transform of latent output (TFPR vs TFPQ) | close | Gadre, Llama 3, Ruan, Bhagia |
| Multiple benchmarks → low-rank capabilities | latent-factor measurement system for output | close | Ruan |
| Tokenizer/vocabulary, bits-per-byte normalization | units of measurement of output (deflator) | close | Tao, MiniCPM, DeepSeek |
| Model-family "compute efficiency" ν_f (Ruan), lab differences in optimal D/N (MiniCPM 192 vs Chinchilla 20) | firm TFP heterogeneity; factor-augmenting productivity heterogeneity | close | |
| Architecture improvements shift intercept only (Hestness) | Hicks-neutral technical change | close | in log–log space |
| N_eff = N e^{α′t}, D_eff = D e^{β′t} (Ho) | factor-augmenting technical change | exact | Doraszelski–Jaumandreu |
| Data quality changes allocation exponent a (DeepSeek) | non-neutral technical change beyond factor augmentation | close | factor augmentation cannot move a in Chinchilla family |
| Effective data from repetition D′ (Muennighoff) | effective capital services with use-dependent depreciation | close | R*_D ≈ 15.4 |
| Effective data transferred D_T = k D_F^α N^β (Hernandez) | quality-adjusted input/knowledge capital spillover | close | |
| MoE total vs active parameters | capital stock vs capital services (utilization) | close | Abnar, Clark, DeepSeek-V3 |
| Precision → N_eff (Kumar) | input quality (quality-adjusted capital) | close | P* ≈ 7–8 bits independent of C |
| Distillation teacher | intermediate input (gross output); make-or-buy | close | Busbridge δ^{Pre} toggle |
| Inference-aware over-training (Sardana) | markup-like wedge εN/εD = 1 + D_inf/(3D_tr) (DLW) | close | wedge is conditional on production function |
| Train vs test-time compute (Jones) | Cobb–Douglas isoquant between two inputs (σ = 1) | exact | slope −1.2 in log10 |
| Test-time vs pretraining trade-off varies by difficulty (Snell) | heterogeneous substitution across products; aggregation | loose | |
| RL sigmoid: C_mid/B vs asymptote A (Khatri) | efficiency (input-augmenting) vs frontier-shifting change | close | |
| Quantization model (Zipf quanta) | Houthakker (1955)/Jones (2005) Pareto aggregation ⇒ Cobb–Douglas/power law | close | cross-equation restriction β = α/(1+α) |
| Per-problem exponential, aggregate power law in attempts (Schaeffer 2025) | aggregation over heavy-tailed micro heterogeneity (Houthakker) | close | exponent = tail index of difficulty distribution |
| Broken neural scaling laws | piecewise log-linear production / structural breaks | loose | |
| Densing law ρ_max(t) | frontier productivity growth index | loose | reference-model dependent |
| Choosing min-loss run per FLOP (Approach 1) | frontier estimation with selection on outcomes (winner's curse) | loose | |
| Economies of scale | returns to scale | breaks-down | not identified: loss is ordinal; RTS depends on cardinalization |
| Compute ≈ 6ND as "price" | market input prices | breaks-down | no market prices for N, D; dollar costs require MFU/hardware prices (Sardana Eq. 6) |

---

## 11. Disagreements / flags

- Allocation exponent a: Kaplan 0.73 (non-embedding, L(N,S)); Henighan ~0.7 all domains; Hoffmann 0.46–0.50; Besiroglu 0.513; DeepSeek 0.450–0.578 (data-dependent; C = MD); Llama 3 N* ∝ C^{0.47} (tokens ∝ C^{0.53}); Porian 0.835 → 0.497; MiniCPM η = (β−α)/(α+β) = −0.10 (a ≈ 0.44); Farseer: not a constant (D/N rises with C).
- Tokens/param at optimum: Chinchilla ~20; Hoffmann A3 ~70 (artifact); MiniCPM ~192; Farseer rising; Muennighoff α = β ⇒ constant ~20.
- Exponent levels are not comparable across papers (see §9 item 11).
- Llama 3's "N*(C)" notation denotes tokens — easy to misread.
- Hoffmann's A3 CIs are invalid (optimizer); use Besiroglu SEs for inference.
- Kaplan L(N,D) implies N ∝ C^{0.56–0.58} if C = 6ND is (inappropriately) applied — my derivation, not in the literature I read.
- Venues flagged UNVERIFIED in bib where arXiv metadata/PDF footer did not confirm.

---

## 12. Data leads (public)

1. Epoch Chinchilla reconstruction (245 points, N, D, L): github.com/epoch-research/analyzing-chinchilla.
2. Muennighoff datablations (400+ runs, repetition, code-mix): github.com/huggingface/datablations.
3. Gadre over-training testbed (104 models, M up to 640, 3 datasets, downstream evals): github.com/mlfoundations/scaling.
4. Porian et al. (>900 runs, OWT2/RefinedWeb, hyperparameter sweeps, checkpoints): github.com/formll/resolving-scaling-law-discrepancies.
5. Farseer (~1000 LLMs, logs): github.com/Farseer-Scaling-Law/Farseer.
6. DataDecide (25 recipes × 14 sizes × 3 seeds, 30K checkpoints, downstream): huggingface.co/collections/allenai/datadecide-67edb1d2bacba40b5d3ed633; github.com/allenai/DataDecide.
7. Choshen et al. ColPret (485 models, 40+ families, 1.9M steps): github.com/IBM/ColPret.
8. Ruan et al. observational scaling (~100 public models × benchmarks): github.com/ryoungj/ObsScaling.
9. Ho et al. algorithmic progress (231 models, WT/PTB): github.com/epoch-research/lm-algorithmic-progress.
10. Pearce–Song reconciliation code: github.com/TeaPearce/Reconciling_Kaplan_Chinchilla_Scaling_Laws.
11. Hägele schedules/scaling code: github.com/epfml/schedules-and-scaling.
12. Tao vocabulary: github.com/sail-sg/scaling-with-vocab.
13. Large Language Monkeys samples: huggingface.co/datasets/ScalingIntelligence/monkey_business.
14. Jones board-game laws: github.com/andyljones/boardlaw.
15. Frontier disclosures for a cross-lab "firm-level" table: Llama 3 (§3.2.1), DeepSeek LLM (Tables 3–4, Eq. 1, 4), DeepSeek-V3 (costs), MiniCPM (α, β, D/N), Chinchilla Table 3.

---

## 13. Contribution ideas generated from this strand

1. **Estimate σ(N,D) instead of a.** Show σ is stable (0.74–0.78) across Hoffmann/Besiroglu/Muennighoff while a is fragile; report σ with delta-method/bootstrap CIs on the Epoch and datablations data; test α = β (homotheticity/CES) formally.
2. **Nesting test Kaplan vs Chinchilla** with the q-family L = E + [(AN^{−α})^{1/q} + (BD^{−β})^{1/q}]^q on datasets with unbalanced designs (Gadre M up to 640; Sardana-style 10k tokens/param; Muennighoff multi-epoch); identify q.
3. **FOC-based vs production-function-based identification:** compare α, β recovered from frontier (γ/a, γ/b) with those from full-surface NLS; a Hausman-type test of cost minimization (i.e., "are published frontier models cost-minimizing?").
4. **Inference wedge ("markup") estimation** for released model families (Llama 1/2/3, Qwen, Gemma, DeepSeek) with uncertainty propagated from the production-function estimate; decompose over-training into (i) non-homotheticity (Farseer), (ii) inference demand wedge, (iii) data constraints (Muennighoff shadow price).
5. **Hyperparameters as flexible inputs:** a GNR-style two-step — use the flexible-input FOC (optimal LR/batch laws) to "concentrate out" h and show how Porian's bias arises; propose a control-function correction when h is set by fixed rules.
6. **Factor-augmenting vs curvature-changing progress:** using DataDecide (25 recipes) test whether recipe differences are Hicks-neutral (intercept), factor-augmenting (A vs B only; a unchanged), or exponent-changing (DeepSeek-type).
7. **Aggregation result:** formalize the equivalence Michaud quantization model ↔ Houthakker/Jones Pareto aggregation; derive the implied restriction β = α/(1+α) and test it.
8. **Output measurement:** treat benchmarks as noisy, bounded indicators of latent log-output (Ruan factor model) and estimate family TFP with a measurement system; contrast with loss-based TFP (TFPR vs TFPQ analog).
9. **Winner's-curse correction** for Approach 1 envelopes and IsoFLOP argmins (seed noise ~ up to 4%).
10. **Train/test compute Cobb–Douglas** (Jones) vs heterogeneous substitution (Snell): estimate σ between pretraining and inference compute from public test-time scaling datasets (monkey_business).

---

## 14. Reference keys in `ml_core.bib`
hestness2017deep, rosenfeld2020constructive, kaplan2020scaling, henighan2020scaling, hernandez2021scaling, hoffmann2022training, besiroglu2024chinchilla, pearce2024reconciling, porian2024resolving, choshen2024hitchhiker, hagele2024scaling, sardana2024beyond, gadre2024language, muennighoff2023scaling, bi2024deepseek, grattafiori2024llama, hu2024minicpm, deepseekai2024deepseekv3, li2025predictable, clark2022unified, krajewski2024scaling, abnar2025parameters, frantar2023scaling, kumar2024scaling, tao2024scaling, busbridge2025distillation, goyal2024scaling, sorscher2022beyond, ye2024data, shukor2025scaling, magnusson2025datadecide, jones2021scaling, snell2024scaling, brown2024large, wu2024inference, khatri2025art, schaeffer2023emergent, ruan2024observational, isik2024scaling, bhagia2024establishing, lourie2025scaling, sharma2020neural, bahri2024explaining, michaud2023quantization, schaeffer2025large, maloney2022solvable, paquette2024phases, caballero2023broken, alabdulmohsin2022revisiting, ho2024algorithmic, xiao2024densing, jones2005shape, houthakker1955pareto.
