# Strand notes: statistical practice, estimation controversies, and measurement in scaling-law fitting

Scope: how neural scaling laws are estimated (estimators, inference, sensitivity), how inputs (N, D, C) and output (loss, accuracy) are measured, and how hyperparameters enter as scale-dependent choices. Every issue is mapped to an IO / applied-econometrics analog. BibTeX keys refer to `lit/bib/ml_estimation.bib`.

Notation: L = held-out cross-entropy (nats/token unless stated). N = parameters. D = training tokens. C = training FLOPs, with C ≈ 6ND unless stated. Chinchilla form: L(N,D) = E + A N^-α + B D^-β. Compute-optimal allocation: N* ∝ C^a, D* ∝ C^b. "[derived]" marks my own calculations from published numbers. These are not claims made by the cited papers, and the modeling strand should re-check them.

Sources were read on arXiv abs/HTML pages or in full-text PDFs (Hoffmann, Kaplan, Porian, Ivgi, Schaeffer 2023 and Gao/Pile were extracted and read directly). Numbers taken only from tool summaries, and not cross-checked in the full text, are tagged (summary-only).

---

## 0. Executive summary (the ten things the paper must use)

1. **The flagship parametric estimate was a numerical artifact.** Hoffmann et al. (2022) Approach 3 (`hoffmann2022training`) reported E=1.69, A=406.4, B=410.7, α=0.34, β=0.28, with an 80% CI for a of (0.454, 0.455). Besiroglu et al. (2024) (`besiroglu2024chinchilla`) digitized Fig. 4 (245 points) and refit the same estimator. They got E=1.8172 (0.03), A=482.01 (124.58), B=2085.43 (1293.23), α=0.3478 (0.02), β=0.3658 (0.02), giving a=0.5126±0.02 (SE on a is 0.018). That is about 20 tokens per parameter, consistent with Approaches 1 and 2. The Chinchilla authors confirmed the cause: they *averaged* rather than summed the Huber loss, and L-BFGS-B stopped early. The bootstrap replicates also stopped near their initialization, so the CIs were about 50x too narrow. Matching that width honestly would take about 600,000 runs instead of about 400. Rounding β from 0.2849 to 0.28 caused a further ~13% bias at D=1e11. Econometric analog: the numerical-convergence failures in BLP demand estimation (`knittel2014estimation`, `dube2012improving`).
2. **The Kaplan-vs-Chinchilla dispute (a=0.73 vs 0.50) is mostly input mismeasurement plus small-scale curvature, not a technology difference.** Three pieces of evidence:
   - Pearce & Song (`pearce2024reconciling`): counting only non-embedding parameters removes an input whose share varies with scale (embeddings ∝ N^{1/3}). This alone makes a Chinchilla technology *look* like a≈0.74–0.78 over Kaplan's range.
   - Porian et al. (`porian2024resolving`): going step by step from the Kaplan setup, a moves 0.864/0.835 → 0.699/0.706 (count last-layer FLOPs) → 0.603/0.602 (fix warmup) → 0.574/0.571 (cosine decay) → 0.518/0.497 (tune LR, batch and β2; no decay needed).
   - IO mapping: nonclassical, regressor-correlated measurement error in inputs (`collardwexler2016production`), plus the Nerlove (1963) / Christensen–Greene (1976) lesson. Local elasticities estimated over small firms do not extrapolate when the technology is non-homothetic.
3. **Estimator choice is not innocuous, and most papers do not report it.** Of 51 scaling-law papers surveyed (`li2025misfitting`), 45 fit power laws, 28 describe curve fitting, 19 release code, 23 omit the optimizer entirely and 15 do not say how they count FLOPs or parameters. Initialization, the loss function (Huber vs MSE, log vs level), the choice of which points to include and the range of model sizes all move the optimal D/N "substantially".
4. **The 5-parameter Chinchilla surface is weakly identified.** Czech et al. (`czech2026problems`) find a Hessian condition number κ≈3.5×10^11 (eigenvalues ~8×10^-6 to 3×10^6), with the flat directions along the linear coefficients A and B. Profiling out (E, A, B) by least squares (variable projection, `golub2003separable`) brings κ down to ~11. Choshen et al. (`choshen2024hitchhikers`) find A–α and B–β nearly linearly related across 40+ model families, and 3 principal components explain 99.49% of the variance of the 5 parameters. Porian's saturating fit of L*(C) has its intercept E drifting 2.01 → 1.78 → 1.68 as one and then two higher-compute points are added. IO analog: normalized CES (`klump2007factor`, `klump2012normalized`) and weak/semi-strong identification (`andrews2012estimation`).
5. **The Chinchilla estimator is effectively least absolute deviations (LAD) on log loss.** Huber with δ=10^-3 on log-loss residuals is quadratic only for residuals below 0.1%. Loss digitization alone has precision ~0.01, so almost every residual sits in the linear (L1) region. The estimator is therefore close to median regression (`koenker1978regression`) of log L, not NLS. Hoffmann et al. also note that the Huber loss treats low-compute points (C≤1e21) as outliers, and that this, together with the frontier's negative curvature, pulls N_opt down. Robust loss functions here choose which part of a misspecified surface to fit; they do not fix the misspecification (`white1982maximum`).
6. **Approach 2 (IsoFLOP parabolas) is biased whenever α≠β or the sampling grid is off-center.** Only the intercept is biased (by 0.3% for ±2x grids at Chinchilla-like exponents, up to 23% for asymmetric surfaces with ±16x grids), not the exponent (`czech2026problems`). Applied to digitized Llama 3 IsoFLOP data, this implies 6.5% of a 3.8e25-FLOP budget misallocated, about $1.4M (90% CI $412K–$2.9M). IO analog: the Kmenta (1967) second-order approximation of CES (`kmenta1967estimation`).
7. **Hyperparameters are scale-dependent flexible inputs.** Leaving them untuned produces inefficiency that is correlated with scale, and that biases the estimated elasticities. Evidence:
   - Porian: optimal batch size BS=0.00037·N^0.703, optimal LR=3.7·N^-0.36, and β2 must rise to 0.99–0.999 at small batch sizes.
   - DeepSeek (`bi2024deepseek`): η_opt=0.3118·C^-0.1250 and B_opt=0.2920·C^0.3271.
   - Step Law (`li2025predictable`), from 3,700 runs: η=1.79·N^-0.713·D^0.307 and B=0.58·D^0.571.
   - Bjorck et al. (`bjorck2024scaling`): LR*=C·N^-0.23·D^-0.32. The **sign of the D-elasticity of LR disagrees with Step Law**; see §13.
   - Lourie et al. (2026) (`lourie2026small`): with 4 hyperparameter configurations per scale, no scaling law is visible at all. It takes about 256 configurations to recover an accurate one.

   IO analog: flexible inputs and the envelope/concentrated production function (`gandhi2020identification`, `ackerberg2015identification`), and scale-correlated inefficiency in stochastic frontiers (`aigner1977formulation`).
8. **Output is measured in tokenizer- and dataset-specific units.** Kaplan et al. state that the constants N_c, D_c, C_c "depend on the vocabulary size and tokenization". BPB = (L_T/L_B)·ℓ/ln2, with 0.29335 GPT-2 tokens per byte on the Pile (`gao2020pile`). So "20 tokens/parameter" ≈ 68 bytes/parameter for GPT-2 tokenization [derived]; the ratio is *not* unit-free. Exponents α and β are invariant to rescaling L or D, but E, A, B, G and the D/N ratio are not. Tao et al. (`tao2024scaling`) normalize against a unigram model because raw per-token cross-entropy is not comparable across vocabularies. Brandfonbrener et al. (`brandfonbrener2024loss`) show the losses on different datasets are linked by shifted power laws, L1 = K(L0−E0)^κ + E1, under which the compute-optimal size is invariant. That is a monotone relabeling of the output index that preserves isoquants. IO analog: TFPR vs TFPQ (`foster2008reallocation`) and bias from unobserved output prices (`klette1996inconsistency`).
9. **Downstream metrics are bounded, nonlinear and noisy transforms of latent output.** Most "emergence" is created by the choice of metric: over 92% of emergent abilities on BIG-Bench appear under just two metrics, Multiple Choice Grade and Exact String Match (`schaeffer2023emergent`). Along the chain log p → p → p_choices → accuracy, the share of samples whose score correlates >0.75 with compute falls from about 90% to about 40% (`schaeffer2024predicting`). Only 39% of 46 downstream tasks scale predictably (`lourie2025scaling`). Seed standard deviations are 0.57 pp on MMLU and 2.15 pp on COPA (`madaan2024quantifying`). The best achievable loss-extrapolation error is about 4% ARE (`choshen2024hitchhikers`).
10. **Observational cross-family scaling laws are production functions with family (firm) fixed effects, and no one addresses simultaneity.**
    - Ruan et al. (`ruan2024observational`): S_m ≈ θ_f log C_m + ν_f over 77 models from 21 families. ν_f is family "compute efficiency", i.e. TFP.
    - Sloth (`polo2024sloth`): θ_{i,k} = α_{i,k} + β_k^T[log s, log t, log s·log t]. This is a translog with family intercepts and common slopes.
    - Ho et al. (`ho2024algorithmic`): algorithmic progress is a time-trend TFP term. Effective compute halves every ~8 months (95% CI ~5–14).

    None of these papers treat input choices as endogenous to family productivity. IO analogs: Marschak–Andrews (`marschak1944random`), Mundlak (`mundlak1961empirical`), OP/LP/ACF. The within-lab IsoFLOP sweep is the ZKD (`zellner1966specification`) case: inputs are fixed before the output shock (the seed) is realized, so the fit is consistent.

---

## 1. The canonical estimators: exact definitions

### 1.1 Kaplan et al. (2020) (`kaplan2020scaling`)
- **Inputs:** N = *non-embedding* parameters; C ≈ 6NBS (non-embedding compute), where B is batch size and S is steps; C ≈ 6N FLOPs per token. Model sizes run from 768 to 1.5B non-embedding parameters; data is WebText2.
- **Single-factor laws:**
  - L(N) = (N_c/N)^{α_N}, with α_N≈0.076 and N_c≈8.8e13.
  - L(D) = (D_c/D)^{α_D}, with α_D≈0.095 and D_c≈5.4e13 tokens (early stopping).
  - L(C_min) = (C_c^min/C_min)^{α_C^min}, with α_C^min≈0.050 and C_c^min≈3.1e8 PF-days.
- **Joint law (Eq. 1.5):** L(N,D) = [(N_c/N)^{α_N/α_D} + D_c/D]^{α_D}. The fit in Table 2 gives α_N=0.076, α_D=0.103, N_c=6.4e13, D_c=1.8e13.
  - This is a CES-type aggregator: L^{1/α_D} = a·N^{-α_N/α_D} + b·D^{-1}, a sum of power terms with *different* exponents (α' = 0.738 on N, 1 on D).
  - Applying the Chinchilla-style formula at a 6ND cost optimum would give σ* = 2/(α'+1+2) ≈ 0.535 [derived]. This is only illustrative: Kaplan's L(N,D) is an early-stopped loss, not the loss of a compute-constrained run.
- **Learning curve (Eq. 1.6):** L(N,S) = (N_c/N)^{α_N} + (S_c/S_min)^{α_S}, with S_c≈2.1e3 and α_S≈0.76.
- **Critical batch size (Eq. 1.4):** B_crit(L) = B*/L^{1/α_B}, with B*≈2e8 tokens and α_B≈0.21 (building on `mccandlish2018empirical`).
- **Allocation (Eqs. 1.7–1.8):** N ∝ C^{α_C^min/α_N}, B ∝ C^{α_C^min/α_B}, S ∝ C^{α_C^min/α_S}. Empirically N ∝ C^0.73, B ∝ C^0.24, S ∝ C^0.03.
- **Note:** Kaplan et al. state the constants N_c, D_c, C_c "do not have a fundamental meaning" because they depend on vocabulary and tokenization. That is an explicit admission that the output and data units are measurement-specific.

### 1.2 Hoffmann et al. (2022), "Chinchilla" (`hoffmann2022training`)
- Over 400 models, 70M to 16B parameters, 5B to 500B tokens, on MassiveText. N counts *all* parameters, including embeddings. FLOPs are counted in full, including embeddings, attention and logits (App. F). The ratio of their FLOP count to 6ND is reported to be "very small" and not to matter.
- **Approach 1 (lower envelope of training curves):**
  - Train each N with 4 cosine horizons spanning 16x. Smooth and interpolate the loss curves. At 1,500 log-spaced FLOP values, take the run with the minimum loss. Then fit N_opt ∝ C^a and D_opt ∝ C^b.
  - Result: a=0.50 (0.488, 0.502), b=0.50 (0.501, 0.512).
  - All selected points fall within the last 15% of their cosine cycle.
  - IO analog: a nonparametric *minimum-cost frontier* (envelope). Taking the minimum over noisy curves biases the frontier downward, and the bias grows with the number of competing runs at each C (extreme-value selection).
- **Approach 2 (IsoFLOP profiles):**
  - 9 budgets from 6e18 to 3e21. At each budget, vary N (up to 16B) and set D = C/(6N) with a cosine length matched to D. Fit a parabola in log N to the final (smoothed) loss and take its minimum.
  - Result: a=0.49 (0.462, 0.534), b=0.51 (0.483, 0.529).
  - IO analog: each IsoFLOP curve is loss along an **isocost line**, and the parabola vertex estimates the cost-minimizing input mix.
- **Approach 3 (parametric):** Eq. (2) is L̂ = E + A/N^α + B/D^β. Eq. (3) is the estimator: min Σ_i Huber_δ(log L̂(N_i,D_i) − log L_i), computed through the LSE parameterization (Eq. 11): log L̂ = LSE(a−α log N, b−β log D, e), with A,B,E = exp(a,b,e).
  - Optimizer: L-BFGS started from a grid of α∈{0,0.5,…,2}, β∈{0,0.5,…,2}, e∈{−1,−0.5,…,1}, a∈{0,5,…,25}, b∈{0,5,…,25}. That is 5·5·5·6·6 = 4,500 starting points [derived count].
  - δ=10^-3. Per the paper, larger δ overfits the small-compute regime, and smaller δ does not change predictions.
  - Result (Eq. 10): E=1.69, A=406.4, B=410.7, α=0.34, β=0.28. This implies a=0.46 (0.454, 0.455) and b=0.54 (0.542, 0.543).
  - The 10th–90th percentiles come from "bootstrapping" by drawing an 80% subsample 100 times. That is subsampling, not the standard bootstrap (`politis1994large`).
- **Closed-form frontier (Eq. 4):** N_opt(C) = G (C/6)^a and D_opt(C) = G^{-1}(C/6)^b, where G = (αA/(βB))^{1/(α+β)}, a = β/(α+β) and b = α/(α+β).
- **Curvature (App. E, Fig. A5):** the FLOP-minimal loss frontier bends negatively. Separate fits on the first, middle and last thirds of the frontier give different slopes, so "projections from very small models lead to different predictions than those from larger models". Approach 3 predicts smaller N_opt than Approaches 1 and 2 because the Huber loss downweights the low-C points.
- **Learning-rate schedule (App. B):** a cosine cycle that overshoots the step count by more than 25% hurts. This was Hoffmann's hypothesis for the Kaplan discrepancy, and Porian et al. (2024) show it is *not* the main cause.

### 1.3 Numerical check on the published Approach-3 values [derived: my computation from Eq. 4 closed forms]
| Spec | a | D/N at 5.76e23 (Gopher) | N* at Gopher C | σ* = 2/(α+β+2) | loss–compute exponent αβ/(α+β) |
|---|---|---|---|---|---|
| Hoffmann, published rounded (α=.34, β=.28) | 0.452 | 92.6 | 32.2B | 0.763 | 0.154 |
| Hoffmann, TeX unrounded (α=.3392, β=.2849) | 0.457 | 59.1 | 40.3B (paper: "40B") | 0.762 | 0.155 |
| Besiroglu et al. refit | 0.513 | 18.4 | 72.3B (Chinchilla = 70B) | 0.737 | 0.178 |

- The unrounded Hoffmann values reproduce the paper's own "40B at Gopher budget". The rounded ones give 32B, so rounding alone moves the optimal size by 20%.
- The Epoch refit implies D/N of 18.4 at 5.76e23, falling slowly to 16.1 at 1e26 (b<a).
- The loss–compute exponents 0.155 and 0.178 match Pearce & Song's quoted values (L−E ∝ C^-0.155 for Chinchilla, C^-0.178 for Epoch).
- **σ\* derivation [derived].** Let s_N = A N^-α/R and s_D = B D^-β/R be the shares of reducible loss R. Along an isoquant, σ = (1+r)/((α+1)+(β+1)r), where r = αs_N/(βs_D). The cost-minimization first-order condition under C=6ND implies αA N^-α = βB D^-β, so r=1 and σ* = 2/(α+β+2). With α=β this reduces to the CES value 1/(1+α). **Away from the optimal expansion path σ varies**, because the additive-power form is CRESH/CDE-like (`hanoch1971cresh`), not CES.

---

## 2. The Chinchilla replication controversy (Besiroglu, Erdil, Barnett, You 2024) (`besiroglu2024chinchilla`)

- **Data extraction.** Fig. 4 of Hoffmann was converted from PDF to SVG and the scatter-point coordinates parsed. Axes were mapped from tick labels; loss was decoded from a log color scale spanning 2.00–5.00. Result: 245 points (240 after dropping 5 outliers). The outliers sit at ~1e19 FLOP with D/N<0.4 and have ~70% higher loss.
- **Digitization noise.** Imprecise y-coordinates (no tick marks). Loss precision is ~0.01 because the colormap has only 256 colors.
- **Data file:** `data/svg_extracted_data.csv` in github.com/epoch-research/analyzing-chinchilla. I checked it: 245 data rows with columns x, y, color, Model Size, Training FLOP, hex_color, loss. **D is not in the file.** It must be imputed as C/(6N). If Hoffmann's plotted FLOPs are their own full count (App. F) rather than 6ND, the imputed D carries a size-dependent measurement error [my flag].
- **Estimator.** Same as Hoffmann (Huber δ=10^-3 on LSE log loss, same initialization grid), but minimized with BFGS rather than L-BFGS-B. The bootstrap replicates skip the grid.
- **Bootstrap.** Nonparametric: n=240 points resampled with replacement 4,000 times, giving the covariance of (log A, log B, log E, α, β).
- **Re-estimates (SEs in parentheses):** A=482.01 (124.58), B=2085.43 (1293.23), E=1.8172 (0.03), α=0.3478 (0.02), β=0.3658 (0.02).
  - A χ² test against Hoffmann's values gives p<10^-51 (excluding outliers). Individually, E differs at p=2.6e-6 and β at p=1.1e-4.
  - Allocation: a=0.5126±0.02 and b=0.4874, versus Hoffmann's 0.454/0.546.
  - Tokens/parameter: about 20 (point estimate 25.6 when the outliers are included). The 80% CI at ≥1e26 FLOP spans 4–40 tokens/parameter. Hoffmann's parametric fit implied about 70, inconsistent with the 20 used to train Chinchilla itself.
- **Likelihood comparison (Table 2).** The log-likelihood uses a Huber-based density, p_δ(x) ∝ exp(−Huber_δ(x)), with free location μ and scale σ.
  - 240 points: Hoffmann rounded 562.25, Hoffmann unrounded 837.78, refit 879.77. The LR test gives p=1.22e-16.
  - 245 points: 531.89, 714.43 and 757.80, with p=3.23e-17.
  - The refit is better for 90% of observations, and 98% of its Huber losses fall below Hoffmann's median.
- **Root causes, confirmed by the Chinchilla authors after v1.** (i) The Huber loss was averaged rather than summed over points. The resulting objective scale made L-BFGS-B terminate early, both in the main fit and in the bootstrap replicates, which stayed near their initialization and so had artificially low variance. (ii) Parameters were rounded in print; the TeX source has β=0.2849 and E=1.6934, and rounding causes a ~13% bias at D=1e11.
- **The "600,000 experiments" argument.** The refit SE of a (0.018) implies an 80% CI width of about 0.05, 50 times Hoffmann's 0.001. Matching Hoffmann's width would need 50² = 2,500 times the sample, i.e. about 600k runs.
- **IO mapping.**
  - (a) Optimizer tolerances and objective scaling that stop estimation early are the same failure documented for BLP by Knittel & Metaxoglou (2014), who found different optima and elasticities across optimizers and starting values, and addressed by Dubé, Fox & Su (2012) with tight inner-loop tolerances / MPEC. Strength: close.
  - (b) A bootstrap in which the replicates do not re-converge produces *pseudo-precision*. Standard practice for NLS is to check convergence in every replicate, or to use analytic sandwich SEs (`jennrich1969asymptotic`, `white1982maximum`).
  - (c) Rounding bias is a reporting issue. It matters because of the **extreme collinearity**: small changes in β shift B and G enormously.

---

## 3. Reconciling Kaplan and Chinchilla: input measurement

### 3.1 Pearce & Song (TMLR 2024) (`pearce2024reconciling`)
- **Decomposition.** N_T = N_\E + N_E, where N_E = (h+v)d is the embedding count (h = context length, v = vocabulary, d = width). With a fixed aspect ratio A≈40 and N_\E = (12/A)d³, this becomes N_T = N_\E + ω·N_\E^{1/3} (Eq. 11). Fitting to Chinchilla configurations (44M–16B) gives ω=47,491 and an exponent of ≈0.34, confirming the 1/3.
- **Local elasticity (Eq. 20).** At small scale, N*_\E ∝ C^{β/(α/3+β)}; at large scale it tends to C^{β/(α+β)}, the Chinchilla value. The transition sits at N_\E ≈ ω^{3/2} ≈ 1e7, where embeddings are ~50% of parameters.
  - Simulating the Chinchilla law over Kaplan's range (790 to 1.58B non-embedding parameters) and fitting a local power law gives N*_\E ∝ C^0.78 with Epoch's parameters and C^0.74 with Chinchilla's. Kaplan reported 0.73.
- **Experiments.** BookCorpus, total sizes 0.8M–4.6M, GPT-2 vocabulary, context 16.
  - Measuring with N_T and C_T gives a=0.49, b=0.51. Measuring with N_\E and C_\E gives a=0.74, b=0.26.
  - "The change moving from N_T to N_\E has a much larger effect than moving between optimization schemes." A single LR per size with no annealing gives the same coefficients as a multi-cosine setup.
- **Loss–compute relationship.** Kaplan's L ∝ C^-0.057 without an offset, versus Chinchilla's L−E ∝ C^-0.155 (C^-0.178 for Epoch). Simulating the Kaplan form over Kaplan's range gives C^-0.066 to -0.069. The gap comes from (non-)embedding accounting, curvature, and Kaplan's omission of E.
- **Recommendation.** Use total parameters and total compute, and include an offset E.
- **IO mapping.**
  - Embedding parameters behave like a **quasi-fixed input whose share falls with scale** (N_E/N_T ∝ N^{-2/3}). Dropping it is a nonclassical, regressor-correlated input measurement error, so the local elasticity is biased away from the global one. This is exactly the pattern Nerlove (1963) found for electricity (`nerlove1963returns`): large apparent scale economies among small firms that vanish at scale. Christensen & Greene (1976) (`christensen1976economies`) fixed it with a translog (`christensen1973transcendental`) that lets scale elasticity vary with output. Strength: close.
  - The lesson from the parallel: **a local Cobb-Douglas elasticity estimated on a restricted size range is not a structural parameter.**

### 3.2 Porian, Wortsman, Jitsev, Schmidt, Carmon (NeurIPS 2024) (`porian2024resolving`)
- **Setup.** 16 models from 5M to 901M parameters (the main grid runs 55M–901M), trained on OpenWebText2 (~30B tokens) and RefinedWeb (~600B). Sequence length 2048. FLOP grid {1.25e16·2^i}, i=0..11. 160M held-out tokens. Total cost 22.3K GPU-hours and 3.03e21 FLOPs.
- **Measurement.** N counts all linear layers including the head (the output/logit layer), excluding embeddings. N_eff = N + n·d·l for attention (n=2048); the ratio N_eff/N is ~1.1–1.2. Kaplan's N_Kaplan = N − dv undercounts FLOPs by ~10% at large scale and up to ~90% at small scale.
- **Step-by-step exponents (Table 1; a with 95% CI and R²), OpenWebText2 / RefinedWeb:**
  - Reproduce Kaplan: 0.864 (0.82, 0.90) / 0.835 (0.82, 0.85). This is close to Kaplan's 1.6e9·(C/8.64e19)^0.88.
  - + count last-layer FLOPs: 0.699 (0.66, 0.72) / 0.706 (0.69, 0.72).
  - + correct warmup: 0.603 (0.57, 0.63) / 0.602 (0.59, 0.62). Kaplan warmed up for 3000·2^19 ≈ 1.57B tokens, which is longer than or close to the optimal D of small models. Porian's rule is warmup tokens = N, and the optimal D always ends up ≥5x the warmup.
  - + cosine decay: 0.574 (0.54, 0.61) / 0.571 (0.56, 0.59). Decay alone is *not sufficient*.
  - + tune batch size, LR and β2 (no decay): **0.518 (0.49, 0.54) / 0.497 (0.49, 0.50)**. This matches 0.5 to within 0.6% and Chinchilla's size to within 15%, using a *constant* LR.
  - Reverting the head-FLOP and warmup fixes with tuned hyperparameters reproduces Kaplan's adjusted 0.73 (they get 0.717).
- **Hyperparameter laws (Fig. 3).** BS = 0.00037·N^0.703 (R²=0.986) and LR = 3.7·N^-0.36 (R²=0.997). Rounded: BS = 160·(N/108e6)^{2/3} and LR = 0.0047·(N/108e6)^{-1/3}.
  - **Batch size is measured in sequences of 2048 tokens.** Table 3's default is 256 sequences = 2^19 tokens. One tool summary said "tokens"; that is wrong.
  - β2 = 0.99 up to 220M parameters and 0.95 above.
  - Compared with DeepSeek: batch-size exponents differ by <0.05 and predictions by <60%; LR exponents differ by 0.11, a factor of 2–3.
  - They find an *optimal* batch size below which loss degrades, contradicting the critical-batch-size folklore that all small batches are equally good.
- **Inference procedure.**
  - The IsoFLOP optimum N*(C_i) and its uncertainty come from a "noise-and-interpolate" bootstrap: Gaussian noise of empirically calibrated magnitude plus Akima interpolation, with the median of the bootstrap as the point estimate.
  - The power law is then fit by weighted least squares in logs, weighting by inverse squared log-SDs (Gaussian MLE in logs).
  - CIs come from bootstrap quantiles. Accuracy and CI width improve with experiment budget (Fig. 5).
- **Loss frontier.** Tuned hyperparameters give a saturating L*(C) with exponent about −0.1, twice Kaplan's.
  - Fits: L = 128.93·C^-0.106 + 2.01 → adding one higher-C point gives 97.27·C^-0.096 + 1.78 → two points gives 87.16·C^-0.092 + 1.68.
  - **E is poorly identified** from sub-saturation data. A 901M model at C+≈8e19 reaches L+=2.943, on the predicted trend.
  - Takeaway: a predictable L*(C) trend is an *indicator of locally optimal hyperparameters*.
- **Cost.** Fixed-LR experiments cost 1.54e20 FLOPs versus 2.99e20 with varying-length cosine. The hyperparameter sweep cost 2.04e20 (it could have been done for 1.44e19 using models ≤57M).
- **IO mapping.**
  - Missing head FLOPs: measurement error in the *cost* variable that shrinks with N. This is systematic, like omitting a capital component whose share varies with firm size (`collardwexler2016production`). Strength: close.
  - Warmup and mis-tuned hyperparameters: **technical inefficiency that is larger at small scale**. It inflates small-N losses and so biases a upward. This is the stochastic-frontier issue of inefficiency correlated with inputs (`aigner1977formulation`). Strength: close.

### 3.3 DeepSeek LLM (Bi et al., 2024) (`bi2024deepseek`)
- **Model scale is measured as M = non-embedding FLOPs/token = 72·n_layer·d² + 12·n_layer·d·l_seq** (it includes attention), and compute as C = M·D. The alternatives 6N1 = 72·n_layer·d² and 6N2 = 72·n_layer·d² + 6·n_vocab·d are off by up to 50% at small scale; for an 8-layer model the ratios to M are 0.43 and 1.32.
- **Hyperparameters.** η_opt = 0.3118·C^-0.1250 and B_opt = 0.2920·C^0.3271, fit on 1e17–2e20 FLOPs. A "near-optimal" region is defined as within 0.25% of the minimum generalization error and is wide.
- **Allocation.** IsoFLOP over 8 budgets from 1e17 to 3e20, about 10 allocations per budget, with loss measured in bits-per-byte on a 100M-token validation set. Result: M_opt = 0.1715·C^0.5243 and D_opt = 5.8316·C^0.4757.
- **Data quality changes the exponent:** a = 0.450 (early in-house data), 0.524 (current in-house data), 0.578 (OpenWebText2).
  - Porian et al. get a ≈ 0.5 on OpenWebText2 and conjecture that DeepSeek *repeated* that dataset (it has only ~30B tokens). **This is an open disagreement.**
- **IO mapping.** If data quality were purely D-augmenting (D_eff = qD in the Chinchilla form), it would move only the level G, not a. A change in a requires a change in α/β, i.e. **factor-biased change in the elasticity itself**, not factor augmentation. Separating the two runs into the Diamond–McFadden–Rodriguez identification problem (`diamond1978measurement`, `leonledesma2010identifying`). Strength: close.

### 3.4 Schaeffer, Levi, Kirsch, et al. (2025), robustness of Chinchilla to the parameter definition (`schaeffer2025evaluating`)
- There are three notions of N for Chinchilla's Table A9 (50 models, 44M–16B): as reported, a standard formula, and a best-fit formula with attention ×5. Standard vs reported differs by 7.4% on average (3.6% min, 15.2% max). The best-fit formula brings the maximum error down to 8.7%.
- None of the five parameters differ significantly across the three N notions, and D/N stays ≈20. The slope of D/N per decade of compute is −1.248 (reported), −1.049 (best-fit) and −0.572 (standard).
- **Perturbation experiments** (these map exactly onto the measurement-error taxonomy):
  - (i) Multiplicative Ñ = c_m·N: absorbed into A ≈ A·c_m^α, α unchanged, ratio shifted by a power of c_m. This is like a units change or classical proportional error.
  - (ii) Additive Ñ = c_a + N: α rises linearly from 0.199 to 0.481, A grows ~2.5x and E moves from 1.565 to 1.897. D/N becomes compute-dependent. This is like a fixed component, e.g. embeddings.
  - (iii) Systematic Ñ = μ·(N/μ)^s: α ≈ 10^-0.46·s^-1, and the sign of the ratio's compute slope flips with s≶1.
  - (iv) Log-normal noise exp(δ)·N: 80% CIs widen by about an order of magnitude. Â falls roughly polynomially and α̂ roughly logarithmically (attenuation), pushing D/N up.
- Venue: arXiv. It was submitted to ICLR 2026 and appears in the rejected-submissions list.
- **IO mapping.**
  - Case (i) is the log-linear units invariance of Cobb-Douglas: only the intercept moves.
  - Case (iv) is classical errors-in-variables with attenuation (`griliches1986errors`, `hausman2001mismeasured`).
  - Cases (ii)–(iii) are nonclassical errors (`schennach2016recent`).
  - Strength: exact, as algebra.

---

## 4. Estimator choice, conditioning and identification

### 4.1 (Mis)Fitting survey (Li, Kudugunta, Zettlemoyer; ICLR 2025) (`li2025misfitting`)
- **Reporting.** Of 51 papers: 45 quantify with power laws, 40 describe setup, 36 define variables, 29 describe evaluation, 28 describe curve fitting, 19 provide code, 17 provide checkpoints or scores. 23 omit the optimizer and 15 omit FLOP/parameter counting. 29 of 51 specify checkpoint selection. Very few report CIs or goodness of fit; Porian and Alabdulmohsin are the exceptions named, and Ivgi uses the bootstrap.
- **Taxonomy.** Functional form; training setup; data collection (which checkpoints, interpolation); fitting algorithm (objective, optimizer, initialization, hyperparameters). L-BFGS is the most common optimizer; others use grid search, Adam, scipy `curve_fit` and assorted losses (Huber, MSE, MAE, log-Huber).
- **Case studies.**
  - Initializing at Hoffmann's reported values reproduces Hoffmann almost exactly, even though a full grid search finds a *lower* objective.
  - On their own sweeps, the full grid "frequently yields the power law which diverges most" from Hoffmann: many local minima.
  - Excluding embeddings in Porian's data changes the laws substantially, with the gap growing at scale.
  - Removing the largest models shifts the predicted optimal D/N a lot.
  - A fixed, suboptimal LR can match Hoffmann *better* than a proper sweep. Agreement with Chinchilla is not evidence of correctness.
  - The choice of loss function matters less than the other factors, but still gives a wide range of optimal N.
  - They note the log transform "exaggerat[es] the effects of errors at small values" and is "generally not advised", yet common.
- **Checklist:** hypothesis/form; training setup (number of models, size range, data, FLOP formula, hyperparameters); data collection (checkpoints, evaluation data, metric); fitting (objective, algorithm, initialization, data coverage, validation).
- **Title caveat:** the arXiv title is "(Mis)Fitting: A Survey of Scaling Laws". The ICLR 2025 title is "(Mis)Fitting Scaling Laws: A Survey of Scaling Law Fitting Techniques in Deep Learning".
- **IO mapping.** This is the scaling-law counterpart of the reporting and replication discussions for production-function estimation. Every IO production-function paper states the estimator, instruments, timing assumptions and SEs; scaling-law papers mostly do not.

### 4.2 Czech, Xu, Elmatad, Wang, Held (2026), "Problems with Chinchilla Approach 2" (`czech2026problems`)
- **Source of bias.** The IsoFLOP parabola is a second-order Taylor expansion in log N around the optimum. When α=β, the odd-order terms cancel by symmetry; when α≠β they do not, and the fitted vertex shifts. Uncentered sampling grids add further bias.
  - The error in the intercept is 10^{δw} − 1, independent of compute. The exponent is unbiased; the intercept absorbs all the error.
  - Chinchilla-like surface (α=0.34, β=0.28): 0.3% error with a ±2x grid, 4.1% with ±16x.
  - Asymmetric surface (α=0.46, β=0.15): up to 23% at ±16x.
- **VPNLS.** For fixed (α,β), solve (E,A,B) by OLS or NNLS, then run L-BFGS-B over (α,β) using analytic gradients, starting from a 32² grid. The Hessian condition number falls from ~3.5e11 to ~11.
  - They note that Huber on log L is "slightly misspecified" for additive Gaussian noise, and that MSE-in-levels VPNLS is the MLE under that noise model.
- **Llama 3 application.** Uses digitized IsoFLOP points from Grattafiori et al. (2024) Fig. 2. Approach 2 overestimates tokens by ~60% relative to Approach 3 (VPNLS). The implied deadweight compute is 6.5% of 3.8e25 FLOPs, about $1.4M (90% CI $412K–$2.9M) at 50% H100 MFU. The inferred b=0.5368 matches the published 0.537.
- **Code and data:** github.com/Open-Athena/vpnls; github.com/eric-czech/llama3_isoflop_extraction; huggingface.co/datasets/open-athena/isoflop-experiments (814 rows, Apache-2.0).
- **IO mapping.**
  - VPNLS is concentrating out the linear parameters (`golub2003separable`), the same device used in BLP, where linear taste parameters are concentrated out given the nonlinear σ. Strength: exact.
  - The parabola bias is the same kind of error as Kmenta's (1967) second-order (translog-type) approximation to CES (`kmenta1967estimation`). Strength: close.

### 4.3 Choshen, Zhang, Andreas (ICML 2025), "A Hitchhiker's Guide" (`choshen2024hitchhikers`)
- **Data.** 485 pretrained models from 40+ families (Pythia, OPT, OLMo, Amber, K2, Mamba, RedPajama, ModuleFormer, T5-Pile, GPT-3, Gopher, …), with 1.9M training-step losses. Released at github.com/IBM/ColPret (MIT).
- **Form.** L̂ = e^E + e^A/N^α + e^B/D^β, fit with scipy `curve_fit` (squared loss). An L-BFGS solver was "less stable"; Huber was tested in an appendix with similar results.
- **Error metric.** Absolute relative error, ARE = mean |L − L̂|/L on the target family's ≥30%-trained models.
- **Findings.**
  - About 4% ARE is typically the best attainable, because of seed variance. Up to 20% ARE can still distinguish design choices.
  - Fitting on *intermediate checkpoints* while dropping the first ~10B tokens improves accuracy markedly; for OPT and Pythia, ARE falls from >15% to 4–10%. Final losses alone do worse.
  - About 5 models are "a safe bet", and more small models can beat one larger model because of seed noise. Adding one bad seed can raise error.
  - A single run of a new family plus exponents borrowed from other families is sometimes enough. For example, OLMo 7B can be predicted from 1B checkpoints at <1% error.
  - Extrapolation errors grow with distance: OPT-175B predicted from 8.7B, 13B and 30B models gives 37%, 25% and 15% ARE.
- **Identification.** Across families, A vs α and B vs β are nearly linear (a ridge in the objective), and 3 PCs explain 99.49% of the variance of the 5 parameters, so the model is over-parameterized. Families trained with multiple epochs, and encoder-decoder models, deviate from the pattern.
- **IO mapping.**
  - Pooling families with shared slopes and family-specific intercepts is a panel production function with firm fixed effects (`mundlak1961empirical`). Strength: close.
  - The A–α ridge is the classic CES identification problem between level and curvature. Normalizing around a sample point (A N^-α = A_0 (N/N̄)^-α) orthogonalizes them (`klump2007factor`, `klump2012normalized`). Strength: exact.
  - Using intermediate checkpoints means using non-steady-state observations. A checkpoint partway through a cosine schedule is not the loss of a run trained to that D, so it adds systematic error in exchange for more observations: a bias–variance trade. The panel analog is the adjustment-cost concern in `griliches1998production`.

### 4.4 Other estimator-design papers
- **Alabdulmohsin, Neyshabur, Zhai (NeurIPS 2022)** (`alabdulmohsin2022revisiting`) argue for evaluating scaling-law estimators by *extrapolation loss* rather than in-sample fit. They propose an estimator (M4) and release a benchmark of 90 tasks. IO analog: out-of-sample validation of production-function specifications. Strength: loose.
- **Ivgi, Carmon, Berant (Findings of EMNLP 2022)** (`ivgi2022scaling`) fit OLS of ln y on ln x over M scales × T seeds, report R², and compute CIs with a *hierarchical bootstrap* (B=1000): resample scales with replacement, then seeds within each scale. This gives much wider CIs than the flat bootstrap, which ignores pretraining-seed variance.
  - Other findings: careful hyperparameter tuning is essential, especially at small scale, and model selection works when in-sample R²>0.95.
  - IO analog: the cluster bootstrap with scale as the cluster (`cameron2008bootstrap`). Strength: close.
- **Gadre et al. (ICLR 2025)** (`gadre2024language`) use scipy Levenberg–Marquardt NLS. The over-training law is L(C,M) = E + (a·M^η + b·M^-η)·C^-η, with M = D/N, derived from the Chinchilla form with α=β. The downstream law is Err(L) = ε − k·exp(−γL).
  - 104 models (0.011B–6.9B), M∈{5,…,640}, on C4, RedPajama and RefinedWeb.
  - Predictions (abstract, verified): validation loss of a 1.4B/900B-token run (32x over-trained) and a 6.9B/138B-token run (compute-optimal), each from experiments using 300x less compute; average downstream top-1 error for both models from experiments using 20x less compute.
  - Relative errors of 0.7% (loss) and 0.05% (downstream) appeared only in a tool summary (summary-only); check them in the paper before quoting.
  - Data: github.com/mlfoundations/scaling.
- **Farseer (Li et al., NeurIPS 2025)** (`li2025farseer`) fits L(N,D) = exp(a3·N^γ + b3) + exp(a2·N^β + b2)·D^{-exp(a1·N^α + b1)} using "differential piecewise" fitting on finite differences ΔL = L(N,D) − L(N,λD), with multiple rounds.
  - Fitted (summary-only): L = exp(−0.021·N^0.169 − 0.091) + exp(88.01·N^-0.1 − 6.287)·D^{-exp(−0.124·N^0.123 + 0.424)}.
  - About 1,000 models (201M–25.1B) using ~3M H100 hours. Extrapolation error is 0.50% vs 2.68% for Chinchilla.
  - Key point: the **data exponent depends on N (non-separable)**, and the optimal D/N rises with compute.
  - IO analog: rejecting additive separability in favor of a translog-type interaction (`christensen1973transcendental`). Strength: close.
- **Broken Neural Scaling Laws (ICLR 2023)** (`caballero2022broken`): y = a + (b·x^-c0)·Π_i(1 + (x/d_i)^{1/f_i})^{-c_i f_i}. A flexible, piecewise functional form, analogous to spline or threshold production functions. Risk: overfitting and extrapolation fragility.
- **Tissue et al. (NeurIPS 2025)** (`tissue2024scaling`): loss depends on the entire LR path, L(s) = L0 + A·S1^-α − C·S2, where S1 = Σηᵢ (forward area) and S2 is a momentum-weighted annealing area with λ≈0.99–0.999. They report ~0.2% error on unseen schedules fit from one or two curves (summary-only).
- **Luo et al. (ICLR 2025)** (`luo2025multipower`): a multi-power law for loss curves across LR schedules.
- **Zhang, Wen, Ma (2026)** (`zhang2026configuration`) parameterize the whole configuration→performance map with an LLM (NCPL). They report 20–40% lower error than Chinchilla and extrapolation to 10x compute. IO analog: a flexible nonparametric production function over a high-dimensional input vector. Strength: loose.

### 4.5 Consolidated view: what a scaling-law fit identifies (for the paper)
- **Log vs levels.** Hoffmann, Epoch and most follow-ups fit in *log* space (Huber on log L). If log L = log f(N,D) + u with E[u]=0, a levels prediction needs a retransformation correction, e.g. smearing (`duan1983smearing`). If the error is additive in levels and heteroskedastic, log-space estimators are inconsistent for the level parameters. This is the "log of gravity" problem (`santossilva2006log`, `manning2001estimating`), and Poisson or gamma quasi-ML in levels is the standard fix. Nobody in ML has run this comparison → **contribution opportunity.**
- **Huber with δ=1e-3 ≈ LAD.** Hoffmann's estimator targets the *conditional median* of log L, not the mean. Inference for LAD depends on the error density at zero. Seed noise is roughly symmetric, so the median and mean may coincide, but digitization noise is discrete (colormap bins).
- **Weak identification of E and A.** Near saturation (large N, D), the reducible terms are small and E is well identified. Far from saturation, E trades off against A and B, with the α and B–β ridges. In Andrews–Cheng terms (`andrews2012estimation`), in y = β·g(x,π)+u the nonlinear parameter π is weakly identified when β is small. Here, α is weakly identified when A·N^-α is small relative to noise, i.e. at large N. Standard Wald CIs, and the bootstrap, can then have poor coverage.
- **Boundaries.** E ≥ 0 and A, B ≥ 0 (NNLS in VPNLS). When E hits zero, which is Kaplan's no-offset form, the bootstrap is inconsistent (`andrews2000inconsistency`), and testing E=0 is a boundary test (`andrews2001testing`).
- **Misspecification.** Hoffmann's frontier curvature (App. E), Farseer's non-separability and Pearce & Song's embedding term all say the 5-parameter form is misspecified over wide ranges. The fitted parameters are pseudo-true values whose meaning depends on how the loss weights the design points (`white1982maximum`). **Robust loss choice (Huber) is then effectively a choice of weighting**, not robustness to outliers.

---

## 5. Inference practice: who does what

| Paper | Uncertainty method | Unit resampled | Comment |
|---|---|---|---|
| Hoffmann 2022 | 80% subsample ×100 | final-loss points | Replicates did not re-converge, so CIs about 50x too narrow |
| Besiroglu 2024 | nonparametric bootstrap ×4000 | digitized points | Ignores digitization error and within-run dependence |
| Ivgi 2022 | hierarchical bootstrap B=1000 | scales, then seeds | Cluster-bootstrap analog |
| Porian 2024 | noise-and-interpolate bootstrap + weighted log-OLS | IsoFLOP curves | Noise calibrated empirically; closest to proper two-step inference |
| Ho et al. 2024 | bootstrap ×100; LOO-CV over ~90 specifications | models | Specification search with CV; post-selection inference not addressed |
| Nezhurina 2025 (`nezhurina2025scaling`) | delta method σ²=JᵀCov(θ̂)J, t-intervals | — | Standard NLS asymptotics; form L(C) = A_c(C+B_c)^-α_c + E_c |
| Step Law 2025 | 1000 bootstrap fits of the LR/BS laws | runs | Data released |
| Cai et al. 2025 (`cai2025latent`) | latent-variable model with prediction intervals | families | Statistical (stat.AP) treatment |
| Choshen 2024 | ARE on held-out large models | — | Predictive, not inferential |

Missing everywhere, and a contribution opportunity:
- Clustering by run, since checkpoints and WSD cooldown branches from the same run share a seed.
- Joint uncertainty for derived quantities such as a, G, D/N and σ, beyond Epoch's bootstrap.
- Weak-ID-robust CIs.
- Separating sampling uncertainty from extrapolation or specification uncertainty.

---

## 6. Noise: seeds, benchmark variance, signal-to-noise

- **Madaan et al. (2024)** (`madaan2024quantifying`): 10 Llama-2-7B-architecture models trained from scratch with different seeds for 210B tokens, 21 checkpoints each, across 13 benchmarks and 280 models in total.
  - Seed SD vs bootstrap CI (pp): MMLU 0.57 vs 0.72; HellaSwag 0.21 vs 0.93; COPA 2.15 vs 8.30; HumanEval 1.11 vs 3.98 (summary-only).
  - Monotonicity over training: MMLU 0.09 vs MMLU-cloze 0.95.
  - Continuous metrics have much higher SNR (ARC-C 45.9 discrete vs 381.6 continuous).
  - IRT-based methods increase seed variance.
- **PolyPythias (van der Wal et al., ICLR 2025)** (`vanderwal2025polypythias`): 50 pretraining runs (5 sizes from 14M to 410M × 10 seeds). Two outlier seeds at 410M (seeds 3 and 4) show loss spikes. Seed-level variation is small but has fat tails.
- **Bhagia et al. (COLM 2025)** (`bhagia2024establishing`): the noise of the last 10 checkpoints (SD_10) correlates with accuracy-prediction error at r=0.821 (p=0.004). Noisy tasks (Winogrande, BoolQ) were excluded.
- **Heineman et al. (NeurIPS 2025)** (`heineman2025signal`): signal = spread across models; noise = variability across checkpoints and seeds. 30 benchmarks, 375 models (60M–32B), 900K evaluations. Lower noise → lower scaling-law prediction error. Fixes: perplexity over accuracy, dropping noisy subtasks, averaging checkpoints.
- **Choshen et al.**: roughly 4% ARE floor.
- **IO mapping.**
  - Seed noise is a *pure idiosyncratic output shock realized after the inputs are chosen*. This is the ZKD (1966) condition (`zellner1966specification`), under which OLS/NLS on designed experiments is consistent. There is no transmission bias within a lab's sweep, unlike observational cross-lab data.
  - Benchmark item sampling adds *output measurement error* on top. That error is heteroskedastic across scale, largest near chance and near ceiling.

---

## 7. Hyperparameters as scale-dependent flexible inputs

| Source | Optimal LR | Optimal batch | Notes |
|---|---|---|---|
| Kaplan 2020 | fixed schedule, 3000-step warmup | B_crit(L)=2e8/L^{1/0.21} tokens | Fixed hyperparameters bias a upward (Porian) |
| DeepSeek 2024 | 0.3118·C^-0.1250 | 0.2920·C^0.3271 | Near-optimal band within 0.25% |
| Porian 2024 | 3.7·N^-0.36 | 0.00037·N^0.703 sequences | β2 0.99–0.999 at small batch |
| Bjorck 2024 (ICLR 2025) | LR*(N,D) = 1.55e-3·N^-0.23·D^-0.32 (N≥760M); per size LR*(D) ∝ D^-β, β=0.70 (50M), 0.38 (350M), 0.32 (1.3B), 0.42 (2.7B) (constants summary-only) | not the focus | >250 runs, 50M–2.7B, up to 800B tokens. Llama-1's LR 3e-4 is >2.5x too large; predicted 1.15e-4 at 1T tokens; loss penalty ≤0.027 at 100B tokens |
| Step Law 2025 | 1.79·N^-0.713·D^0.307 | 0.58·D^0.571 | 3,700 runs, 100T tokens, ~1M H800 hours. Convex landscape in (lr, bs) with a broad optimum; 0.094% from the grid-search optimum; covers MoE and dense |
| Everett 2024 (ICML) | per-layer LR prescriptions by parameterization (SP, NTK, μP, MF) × optimizer | — | Tuned per-layer SP beats μP; SP 15.3B beats μP 26.8B (summary-only). Adam ε underflow → Adam-atan2 |
| Yang 2022 (μP, TP-V) | width-independent optimal hyperparameters under μP | — | Zero-shot transfer from small proxy models |

- **Hägele et al. (NeurIPS 2024)** (`hagele2024scaling`): a constant LR plus a cooldown (WSD; 1-sqrt shape; about 20% cooldown is enough) matches cosine. Cooling down from checkpoints of one run per model size gives many (N,D) points. This halves scaling-experiment FLOPs; applied to Chinchilla's design it would cost 2.36e23 instead of 5.59e23 (summary-only).
- **MiniCPM (Hu et al., COLM 2024)** (`hu2024minicpm`): WSD cuts scaling-law cost from O(m²)C to O(m)C. Six sizes from 0.04B to 2B, each with decays from 10N to 60N tokens. The Chinchilla form with `curve_fit` gives α=0.29, β=0.23 and **192 tokens/parameter** (summary-only; re-check the constants). Loss is computed per byte for tokenizer comparability.
- **Wortsman et al. (ICLR 2024)** (`wortsman2023small`): small models become unstable at high LR. "LR sensitivity" is how the final loss varies across LRs, and mitigations such as qk-layernorm and z-loss let loss stay similar across orders of magnitude of LR.
- **Lourie, Cho, Ullrich, Lotfi (2026)** (`lourie2026small`): hyperparameter sensitivity fades with scale, and the effective number of hyperparameters falls toward 1, driven more by N than by D. Scaling laws exist even at small scale (from ~4M parameters), but only with extensive tuning: absent with 4 configurations per scale, still incomplete with 16, accurate with 256.
  - How parameters are counted (six variants) matters little *relative to tuning*.
  - LR decay cuts test MSE by 98%. Tying exponents makes little difference.
  - Irreducible-error estimates are unreliable before saturation.
  - This result partially conflicts with Porian and Pearce & Song on the *importance* of parameter counting. Reconciliation: counting matters for the *allocation exponent a over small ranges*, less for predictive MSE of the joint law.
- **Porian's "no decay" result vs Hägele/Tissue.** A constant LR suffices for the *allocation exponent* (Porian), but annealing shifts *loss levels* (Tissue: the −C·S2 term). Allocation is identified from *relative* losses along isocost lines, which are invariant to a common level shift. Hyperparameter inefficiency that is *scale-neutral* does not bias a; *scale-dependent* inefficiency does.
- **IO mapping.**
  - Learning rate, batch size, β2, warmup and schedule are **flexible inputs**: chosen after (N,D) and costless at the margin, but they affect output (`gandhi2020identification`, `ackerberg2015identification`). The scaling law people want is the *concentrated* (envelope) function L*(N,D) = min_h L(N,D,h). The envelope theorem means first-order mis-tuning has only second-order effects on L (Step Law's convex landscape and broad optimum; DeepSeek's 0.25% band). But the error is not zero, and it is scale-dependent (Lourie 2026; Porian).
  - Fitting L on runs with fixed hyperparameters estimates a *frontier contaminated by scale-correlated inefficiency*. That biases α and hence a: Kaplan's 0.73 → 0.5 once the hyperparameters are tuned.
  - Stochastic-frontier analog: `aigner1977formulation`, with inefficiency depending on N.
  - The disagreement over the D-elasticity of LR is a disagreement over the *flexible-input demand function*, conditional on how the other flexible input (batch size) is chosen.

---

## 8. Measurement of output

- **Which loss?** Pretraining-distribution validation loss is the norm (MassiveText held out, WebText2, OWT2/RefinedWeb held out). Gadre, Lourie 2025 and Brandfonbrener show the validation set changes *both* the level and the ranking of pretraining corpora. Lourie et al. (Findings of EMNLP 2025): on HellaSwag, C4 and RedPajama have identical scaling laws when validated on C4 but diverge on a 100-programming-language validation set; on CoQA the ranking reverses.
- **Tokenizer and BPB.**
  - BPB = (L_T/L_B)·ℓ/ln2, where L_T/L_B = 0.29335 GPT-2 tokens per UTF-8 byte on the Pile (`gao2020pile`). BPB is preferred for its invariance to tokenization.
  - DeepSeek reports BPB; MiniCPM computes loss per byte.
  - Tao et al. (NeurIPS 2024) (`tao2024scaling`) use unigram-normalized loss, L_u = −(1/T)·Σ log[p(wᵢ|context)/p(wᵢ)], so that different vocabularies can be compared. Their optimal vocabulary law is N_v^opt ∝ N_nv^0.83, with fits N_v = 0.20·C^0.42 and N_nv = 0.08·C^0.50 (summary-only). Data is measured in characters, D = H·f(V). Llama2-70B's vocabulary "should have been at least 216K" (vs 32K).
  - Implication [derived]: the tokenizer is both a *measurement* choice (units of L and D) and a *technology* choice (it changes N through embeddings, and changes the prediction task).
- **Compression as the latent output.** Huang et al. (COLM 2024) (`huang2024compression`): average benchmark score across 31 LLMs is almost linear in BPC compression on external corpora.
- **Loss-to-loss translation.** Brandfonbrener et al. (TMLR) (`brandfonbrener2024loss`): L1 = K(L0 − E0)^κ + E1, across 6 pretraining datasets and 528 models (20M–1.7B, 2e17–4.84e19 FLOPs), extrapolating to 20x compute. With 8 models on a new dataset, the translated fit reaches R² 0.988–0.991, against 0.987–0.992 for the full 88-model skyline and 0.45–0.975 for independent fits. The compute-optimal N is invariant under the translation. Mayilvahanan et al. (ICML 2025) (`mayilvahanan2025llms`): loss-to-loss curves are determined mainly by the *pretraining data*, and only weakly by architecture, size, hyperparameters or tokenizer.
  - IO analog: a monotone transformation of the output index preserves isoquants and hence cost-minimizing input ratios, so allocation results are *ordinal*. Strength: close.
- **Downstream accuracy.**
  - Mirage (`schaeffer2023emergent`, NeurIPS 2023): if L_CE(N) = (N/c)^α, then p(token correct) = exp(−(N/c)^α), and exact-match accuracy on L tokens ≈ exp(−L·(N/c)^α). That is sharp in log N. Token edit distance ≈ L·(1 − exp(−(N/c)^α)) is smooth.
  - Elusive downstream prediction (`schaeffer2024predicting`, ICML 2025): 5 families (Pythia, Cerebras-GPT, OLMo, INCITE, LLM360) and 70 benchmarks. The share of samples with score–compute correlation above 0.75 falls from about 90% to about 40% along the chain. Mass on *incorrect* choices varies by orders of magnitude and does not average out.
  - Owen (2024) (`owen2024predictable`): BBH average extrapolated one order of magnitude has 6 pp mean absolute error; individual BIG-Bench tasks have 18 pp.
  - Two-step prediction (Bhagia; Llama 3), N,D → task loss → sigmoid accuracy, lands within 2 pp on 4 of 8 tasks for OLMo-2 7B/13B, with mean errors of 3.8 and 4.2 pp. The ladder costs 3.2% of the target's compute.
  - Krajewski et al. (ICLR 2026) (`krajewski2025revisiting`): a *direct* power law in log accuracy on budget, at fixed D/N, beats the two-step approach, for models up to 17B trained on 350B tokens.
  - Lourie et al. (2025) (`lourie2025scaling`): only 18 of 46 tasks (39%) are predictable. The rest are inverse, nonmonotonic, noisy, trendless or breakthrough.
  - DataDecide (`magnusson2025datadecide`, ICML 2025): 25 corpora × scales up to 1B parameters/100B tokens × 3 seeds. Single-scale ranking at 150M predicts the 1B winner about 80% of the time, and **none of 8 scaling-law methods beat that baseline**. Continuous likelihood metrics make benchmarks more than 80% predictable at 0.01% of the compute.
- **IO mapping.**
  - Accuracy is a bounded (0 to 1, with a chance floor), nonlinear, discretized measure of latent output. That is analogous to measuring output through revenue (TFPR) instead of physical quantity (TFPQ) (`foster2008reallocation`), or to IRT-type latent output. The "sharpness" of emergence is created by the measurement function, not the technology.
  - The tokenizer and validation set act as the *price or units* in which output is measured (`klette1996inconsistency`). If they are chosen jointly with inputs (e.g. new labs adopt larger vocabularies as they scale), cross-lab comparisons of per-token loss are biased.

---

## 9. Functional form: what the forms imply economically

- **Chinchilla additive form.** R = A N^-α + B D^-β. With α=β, Y ≡ R^{-1/α} = (A N^-α + B D^-α)^{-1/α} is CES with ρ=α and σ=1/(1+α), homogeneous of degree 1 (`arrow1961capital`). With α≠β it is non-homothetic and additively separable (CRESH/CDE-like, `hanoch1971cresh`). σ varies along isoquants but equals 2/(α+β+2) on the cost-minimizing path [derived].
- **Kaplan form.** R^{1/α_D} is a sum of power terms: a generalized CES with unequal exponents. It describes the early-stopped loss.
- **Farseer.** Non-separable, with the D-exponent depending on N: a translog-like interaction.
- **Sloth.** Explicit translog in logs (log s, log t, log s·log t), with family intercepts.
- **BNSL.** Piecewise power laws.
- **Gadre.** L(C,M) re-expresses Chinchilla (α=β) in terms of compute and the token multiplier M = D/N, with an M-dependent coefficient (a·M^η + b·M^-η). Over-training is a movement *along the isocost line*.
- **Hoffmann App. E.** Curvature in the log-log frontier means Cobb-Douglas-in-C is only a local approximation.
- **Test agenda** [derived; contribution]: test homotheticity, separability (translog interaction = 0) and α=β (the CES restriction). Report σ(N,D) across the observed input range rather than a single number.

---

## 10. Observational vs experimental data; confounding

- **Experimental (within-lab sweeps):** Hoffmann, Porian, Gadre, DeepSeek, Farseer, Step Law, DataDecide, and the Epoch digitization of Hoffmann. Inputs are set by design, so ZKD applies and seed noise is exogenous. The remaining threats are *measurement* (N, C, tokenizer), *flexible-input mis-tuning* and *misspecification*.
- **Observational (cross-lab public models):**
  - Ruan et al. (NeurIPS 2024) (`ruan2024observational`): σ^{-1}(E_m) ≈ βᵀS_m + α (Eq. 3); S_m ≈ θ_f·log C_m + ν_f (Eq. 4); B_{i,m} ≈ γ_iᵀS_m (Eq. 5). 77 models, 21 families; 3 PCs explain ~97% (PC1 ~80%). f-equivalent FLOPs are measured relative to Llama-2. Validation: train on 47 weaker models (≤ Llama-2-7B FLOPs), test on 30 stronger ones. **No causal interpretation is offered.**
  - Sloth (NeurIPS 2025) (`polo2024sloth`): family intercepts α_{i,k} are described as "efficiency" in converting compute, absorbing data quality and post-training. Slopes are common. Identified up to rotation (Thm A.2). Estimated by Huber loss with Adam, about 69+3f parameters. Uses 12 Open LLM Leaderboard benchmarks.
  - Cai et al. (2025) (`cai2025latent`): family latent variables and statistical properties of the estimator.
  - Ho et al. (NeurIPS 2024) (`ho2024algorithmic`): L = E + (A/N^α_param)·e^{−α_year(Y−Y0)} + (B/D^β_data)·e^{−β_year(Y−Y0)}. Over 200 models (~231 usable) on WikiText-103/2 and PTB, 2012–2023, capped at 3 models per paper.
    - Estimation: NLS in log space; LOO-CV over ~90 specifications (the selected model has R²≈0.91); 100 bootstrap draws.
    - Results: effective-compute doubling time 8.4 months (95% CI 4.5–14.3). The Transformer is worth 7.2x compute (3.3x–45.7x). Shapley decomposition: 60–95% of gains from compute, 5–40% from algorithms.
    - Limitations they note themselves: tokenization, preprocessing and stride differ across papers; algorithmic progress cannot be separated from data quality; possible scale-dependence.
  - Nezhurina et al. (2025) (`nezhurina2025scaling`): **scaling curves cross.** CLIP beats MaMMUT at small compute and MaMMUT wins above ~1e10–1e11 GFLOPs, so single-scale comparisons can reverse. IO analog: crossing isoquants / technology rankings that depend on scale. Strength: close.
- **Threats in observational data (IO framing):**
  - (i) *Simultaneity* (`marschak1944random`): labs with better recipes (higher ν_f) choose larger C.
  - (ii) *Selection*: only released models are observed, and releases are selected on quality. This is like Olley–Pakes attrition (`olley1996dynamics`).
  - (iii) *Measurement*: reported N and D are sometimes estimated; C is often imputed as 6ND; losses sit on different tokenizers and validation sets.
  - (iv) *Unobserved input quality* (data), as in Ho et al.
- **Candidate designs:**
  - Family fixed effects (Mundlak), which Ruan and Sloth already use.
  - Within-family variation in N at fixed recipe, which is close to experimental.
  - Proxy approaches (OP/LP/ACF). For example, lab compute stocks or inference-serving choices could serve as proxies for ν_f. **Not yet done anywhere.**

---

## 11. Effective inputs: data repetition (brief; the detail belongs to another strand)
- Muennighoff et al. (NeurIPS 2023; JMLR 2025) (`muennighoff2023scaling`): D' = U_D + U_D·R_D*·(1 − e^{−R_D/R_D*}), with R_D* ≈ 15.4, and the symmetric N' formula with R_N* ≈ 5.3. Fit by Huber/L-BFGS on 182 runs with the Chinchilla coefficients held fixed (R²≈0.77). Up to 4 epochs is ≈ unique data; the maximum effective data is about 16.4x the unique data [derived: 1+R_D*].
- IO analog: an effective-input/depreciation (perpetual-inventory) correction. Measuring D as raw tokens seen overstates input when data is repeated. The same measurement issue explains the DeepSeek–Porian OWT2 disagreement (§3.3).

---

## 12. Master equivalence table (ML ↔ IO/econometrics), strand-specific

| ML concept | IO/econometrics analog | Strength | Key refs |
|---|---|---|---|
| Chinchilla Approach 3 (Huber-log NLS on 5 parameters) | NLS estimation of a CES-type production function | close | hoffmann2022training; kmenta1967estimation; jennrich1969asymptotic |
| Huber δ=1e-3 on log L | LAD / median regression on log output | exact (as math) | huber1964robust; koenker1978regression |
| Log-space fitting of a level-additive model | Log-linearization bias under heteroskedasticity (log of gravity) | close | santossilva2006log; manning2001estimating; duan1983smearing |
| L-BFGS early stop from averaging the objective; non-reconverged bootstrap | BLP numerical-convergence failures | close | besiroglu2024chinchilla; knittel2014estimation; dube2012improving |
| VPNLS (profile out E, A, B) | Concentrating out linear parameters | exact | czech2026problems; golub2003separable |
| A–α / B–β ridge; κ≈3.5e11 | Normalized CES; weak/semi-strong identification | exact / close | choshen2024hitchhikers; klump2012normalized; andrews2012estimation; rothenberg1971identification |
| E≥0, Kaplan no-offset form | Parameter on the boundary; bootstrap inconsistency | exact | andrews2000inconsistency; andrews2001testing |
| IsoFLOP parabola vertex (Approach 2) | Cost-minimizing input mix on an isocost line; 2nd-order approximation bias | close | czech2026problems; kmenta1967estimation |
| Approach 1 envelope of training curves | Minimum-cost / production frontier (min over noisy runs → downward bias) | close | hoffmann2022training; aigner1977formulation |
| Non-embedding N (Kaplan) | Omitted quasi-fixed input with scale-varying share → nonclassical input error; spurious scale economies | close | pearce2024reconciling; nerlove1963returns; christensen1976economies; collardwexler2016production |
| Missing head/attention FLOPs in C | Systematic mismeasurement of the cost variable | close | porian2024resolving; bi2024deepseek; collardwexler2016production |
| Perturbation experiments on N (multiplicative/additive/systematic/noise) | Units change / fixed-component error / nonclassical error / classical EIV attenuation | exact | schaeffer2025evaluating; griliches1986errors; schennach2016recent |
| Local exponent varies with scale (Kaplan 0.73 vs 0.50) | Scale-varying returns (Nerlove → translog) | close | pearce2024reconciling; christensen1976economies |
| Farseer N-dependent data exponent; Sloth log s·log t | Translog / non-separable technology | close | li2025farseer; polo2024sloth; christensen1973transcendental |
| LR, batch, β2, warmup, schedule | Flexible inputs; concentrated (envelope) production function | close | porian2024resolving; li2025predictable; gandhi2020identification; ackerberg2015identification |
| Mis-tuning that is worse at small N | Stochastic frontier with inefficiency correlated with inputs | close | porian2024resolving; lourie2026small; aigner1977formulation |
| μP / hyperparameter transfer | Scale-invariant optimal flexible-input rule (a "managerial practice") | loose | yang2022tensor; everett2024scaling |
| Seed variance | Idiosyncratic shock realized after inputs are chosen (ZKD) → no transmission bias in designed sweeps | close | zellner1966specification; madaan2024quantifying; vanderwal2025polypythias |
| Hierarchical bootstrap by scale/seed | Cluster bootstrap | close | ivgi2022scaling; cameron2008bootstrap |
| Hoffmann "80% × 100" resampling | Subsampling (m-out-of-n without replacement) | exact | hoffmann2022training; politis1994large |
| Delta-method CIs | Standard NLS asymptotics | exact | nezhurina2025scaling; jennrich1969asymptotic |
| Per-token loss depends on tokenizer; BPB normalization | Output measured in lab-specific units/prices (TFPR vs TFPQ; Klette–Griliches) | close | gao2020pile; tao2024scaling; foster2008reallocation; klette1996inconsistency |
| Loss-to-loss shifted power law | Monotone relabeling of the output index; isoquants preserved | close | brandfonbrener2024loss; mayilvahanan2025llms |
| Accuracy / emergence | Bounded nonlinear measurement of latent output; threshold artifacts | close | schaeffer2023emergent; schaeffer2024predicting; lourie2025scaling |
| Model family efficiency ν_f / Sloth intercepts | Firm TFP fixed effects (management bias) | close | ruan2024observational; polo2024sloth; mundlak1961empirical |
| Cross-family observational scaling | Marschak–Andrews simultaneity; OP attrition/selection | close (unaddressed in ML) | ruan2024observational; marschak1944random; olley1996dynamics |
| Algorithmic progress as a time trend in effective N and D | Factor-augmenting technical change; DMR non-identification with σ | close | ho2024algorithmic; diamond1978measurement; leonledesma2010identifying |
| Data quality changes a (DeepSeek) | Change in elasticity (biased), not pure factor augmentation | close | bi2024deepseek |
| Data repetition → effective D | Depreciation / effective input stock | loose–close | muennighoff2023scaling |
| Crossing scaling curves across architectures | Crossing isoquants; scale-dependent technology choice | close | nezhurina2025scaling |
| Intermediate checkpoints / WSD branches | Non-steady-state observations; within-run correlated panel | loose | choshen2024hitchhikers; hagele2024scaling |
| Extrapolation-based model validation | Out-of-sample specification testing | loose | alabdulmohsin2022revisiting |
| Where the analogy breaks | Scaling-law "inputs" are designed, not chosen by optimizing firms facing prices, so there is no market equilibrium and no demand side. Output is a loss, not a market good. The noise is simulation noise. | breaks-down | — |

---

## 13. Disagreements and open controversies (flag in paper)

1. **Optimal D/N.** Estimates range widely:
   - Chinchilla Approaches 1/2 and the Epoch refit: ~20 (Epoch: 18.4 falling to 16.1 as C goes from 5.8e23 to 1e26 [derived]).
   - Porian: 14–16.
   - Hoffmann's own Approach 3: ~59–93 depending on rounding [derived]. Besiroglu say ~70.
   - MiniCPM: 192.
   - Farseer: rising with compute.
   - Llama 3: tokens exponent 0.53–0.537.

   Part of the spread is units (tokenizer), part is estimator, part is data (quality or repetition), and part is hyperparameter policy.
2. **LR vs token horizon.** Bjorck: LR* ∝ D^-0.32 (decreasing). Step Law: η ∝ D^+0.307 (increasing) jointly with B ∝ D^0.571. The likely reconciliation is that Step Law co-scales batch size, but this is unverified.
3. **Does LR decay matter?** Hoffmann: yes (hypothesis). Porian: not for a (constant LR suffices once warmup and hyperparameters are fixed). Tissue and Hägele: annealing shifts loss levels. Lourie 2026: decay cuts test MSE by 98%. Reconciliation: decay affects levels (and E), not the allocation exponent, unless its effect is scale-dependent.
4. **Does parameter counting matter?** Pearce & Song and Porian: yes, for local a. Schaeffer 2025 and Lourie 2026: little effect on the fitted joint law or D/N≈20 for plausible counting variants. These are consistent once you separate the *local allocation exponent over a small-scale range* from the *joint-law fit over Chinchilla's range*.
5. **OWT2 exponent.** DeepSeek a=0.578 vs Porian ≈0.50 on the same dataset. Porian conjectures data repetition.
6. **Downstream predictability.** Optimistic: Gadre, Krajewski, Bhagia, Ruan, Owen (aggregate benchmarks). Pessimistic: Lourie 2025 (39%), Schaeffer 2024, DataDecide (scaling-law methods don't beat single-scale ranking).
7. **Two-step vs direct downstream prediction:** Bhagia and Llama 3 use two steps; Krajewski argues for a direct fit.
8. **Venue metadata.** Lourie, Hu & Cho is Findings of EMNLP 2025 per the ACL Anthology (the HTML template summary said ICLR 2025, which is wrong).

---

## 14. Contribution ideas (estimation/measurement strand)

1. **Estimator horse race on the Epoch Chinchilla data (245 points) and the open-athena IsoFLOP set (814 rows):**
   - Hoffmann Huber-log with the same grid.
   - LAD on log L.
   - NLS in levels.
   - Poisson/Gamma quasi-ML in levels (Santos Silva–Tenreyro).
   - VPNLS profile NLS.
   - Normalized-CES reparameterization, (N/N̄), (D/D̄).

   Report a, G, D/N at 1e24/1e26, σ* and E, with sandwich, bootstrap (converged-replicate check), cluster-by-IsoFLOP-budget and Andrews–Cheng weak-ID-robust CIs. Headline question: *how much of the D/N uncertainty is sampling vs specification?*
2. **Measurement-error decomposition of the Kaplan–Chinchilla gap.** Build a Collard-Wexler–De Loecker-style bias formula for the local elasticity when N is measured without a component whose share is s(N) = ωN^{-2/3}. Show it reproduces 0.73–0.78, then *quantify in IO units*: the apparent returns-to-scale bias is the analog of Nerlove's small-firm scale economies.
3. **Hyperparameters as flexible inputs (Step Law's 3,700 runs + Porian's data):**
   - Estimate the concentrated function L*(N,D) via the envelope.
   - Estimate the flexible-input demand functions η*(N,D) and B*(N,D).
   - Fit a stochastic-frontier model in which inefficiency depends on N and on the hyperparameter policy (fixed vs tuned).
   - Show analytically and empirically that scale-dependent inefficiency ∂u/∂logN biases α by a computable amount.
4. **Inference for derived economic quantities** (σ, returns to scale, D/N, the markup-like wedge from over-training) with proper clustering (run/seed) and delta-method or bootstrap. Report which are well identified.
5. **Optimal experimental design for scaling studies.** Treat the choice of (N, D) grid, the number of seeds and the IsoFLOP width as D-optimal design for (α, β, E) under a compute budget. Compare against Choshen's "5 models" heuristic and Porian's budget-accuracy curve (Fig. 5).
6. **Units-free reporting.** Convert all losses to BPB, and D to bytes, before comparing across labs. Show that "tokens per parameter" varies with tokenizer compression (about 68 bytes/param for the GPT-2 tokenizer ≡ 20 tokens/param).
7. **Specification tests:** α=β (CES), additive separability vs translog/Farseer interaction, homotheticity, and a Nerlove-style test for scale-varying elasticities by size bins.
8. **Observational cross-family panel** (Ruan/Sloth/Open LLM Leaderboard): estimate family TFP dispersion with fixed effects. Test for simultaneity by comparing within-family (quasi-experimental) and between-family elasticities (the Mundlak/Hausman logic). Candidate proxies for ACF-style control include release date and lab compute.

---

## 15. Data leads (verified this session unless noted)

- **Epoch Chinchilla digitization:** github.com/epoch-research/analyzing-chinchilla. File `data/svg_extracted_data.csv` has 245 rows with columns including Model Size, Training FLOP and loss; D must be imputed.
- **Open-Athena IsoFLOP compilation:** huggingface.co/datasets/open-athena/isoflop-experiments. 814 rows. Sources: Epoch Chinchilla, a second Chinchilla extraction, Llama 3, Marin 2026 (Llama-2 architecture on Comma/DCLM/Nemotron), and (Mis)fitting sweeps. Columns: source, dataset, model, experiment, tokens, params, budget, loss. Apache-2.0.
- **Porian et al.:** github.com/formll/resolving-scaling-law-discrepancies. File `data/experiment_results.pickle.xz` has per-run losses and hyperparameters (width, depth, batch size, LR, warmup, decay, dataset). Checkpoints at huggingface.co/formll/resolving-scaling-law-discrepancies.
- **Gadre et al.:** github.com/mlfoundations/scaling. 104 models, 8 validation losses each, 46 downstream tasks, token multipliers.
- **Choshen et al. (ColPret):** github.com/IBM/ColPret. 485 models, 1.9M loss observations, 40+ families; MIT license.
- **Step Law:** github.com/step-law/steplaw. 3,700 runs (N, D, LR, batch size, smooth loss), dense and MoE; 1,000 bootstrap fits. License not stated.
- **Llama 3 IsoFLOP digitization:** github.com/eric-czech/llama3_isoflop_extraction. VPNLS code: github.com/Open-Athena/vpnls.
- **Lourie et al. 2025 downstream failures:** github.com/nicholaslourie/scale-fails.
- **Sloth:** github.com/felipemaiapolo/sloth.
- **Pythia:** github.com/EleutherAI/pythia. PolyPythias adds 50 seed runs (release location not verified).
- **Compression vs intelligence:** github.com/hkust-nlp/llm-compression-intelligence.
- **Hoffmann et al. Table A9:** 50 model configurations (used by Schaeffer 2025). Table A4: FLOP ratio vs 6ND.
- **Not verified this session:** repositories for Ruan et al. (observational scaling), DataDecide, Farseer, Madaan et al. seeds, Muennighoff (the paper says its 400 models are released on GitHub), and Krajewski et al. (the paper says its data is released).

---

## 16. Warnings and uncertain claims

- **Tool-summary errors I caught and corrected:**
  - Porian's batch size is in sequences, not tokens.
  - PolyPythias is 50 runs (5 sizes × 10 seeds), not "99 seeds per size".
  - Lourie 2025's venue is Findings of EMNLP.
- **Numbers that are summary-only and should be re-checked before quoting:**
  - Madaan's per-benchmark SDs.
  - Everett's 15.3B vs 26.8B comparison.
  - Farseer's fitted constants.
  - The Step Law comparison table (it lists DeepSeek's LR coefficient as 0.3188; DeepSeek's own paper says 0.3118).
  - Hitchhiker's specific numbers beyond 485 models, 1.9M steps, 4% ARE, 3 PCs = 99.49%, and the OPT 37/25/15% errors.
- **Llama 3 rounding.** The paper states (α, A) = (0.53, 0.29) for tokens-optimal vs C. Plugging those in at 3.8e25 gives ~10.5T tokens, not the 16.55T the paper uses. Using α≈0.537 (the value Czech et al. report as published) gives ~16T [derived]. So **the rounded exponent in the text does not reproduce the reported extrapolation.** This is another instance of rounding sensitivity; re-check against the PDF before citing.
- **The Epoch data is digitized.** Loss precision is ~0.01 and y-coordinates are imprecise. D is imputed. Outliers (5 points) change the D/N point estimate (20 → 25.6).
- **The Chinchilla-authors confirmation** (averaging vs summing, early stopping) is reported *by Besiroglu et al. v2*. I did not find a separate erratum from Hoffmann et al.
- **σ\* = 2/(α+β+2)** and all "[derived]" numbers are my own algebra and computation. The modeling strand should re-derive them.
- **Venues:**
  - Hoffmann 2022 is cited as arXiv; NeurIPS 2022 was not verified via OpenReview this session.
  - Tensor Programs V is "NeurIPS 2021" per its arXiv comment.
  - Schaeffer 2025 (robustness) is arXiv only.
  - Czech 2026, Cai 2025, Nezhurina 2025, Step Law, Owen, Madaan and Zhang 2026 are arXiv preprints.
- **Collard-Wexler & De Loecker (2016)** is NBER WP 22437 titled "Production Function Estimation and Capital Measurement Error"; the SSRN version is titled "…with Measurement Error in Inputs". Use the NBER title.
- **Nerlove (1963)** is a chapter in C. Christ (ed.), *Measurement in Economics*, Stanford UP. Its existence is confirmed via Nerlove's "Reminiscences" chapter in the Handbook of Production Economics (Crossref); the original page numbers are unverified.
- **Web-search budget ran out mid-session.** Verification after that point used arXiv API, OpenReview API and Crossref API queries plus direct page fetches.
