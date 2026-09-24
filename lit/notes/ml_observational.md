# Strand notes: Observational (cross-model / cross-lab) scaling, benchmarks as output, algorithmic progress

Project: "Scaling Laws as Production Functions" (AER-style). Strand compiled 2026-09-23.
BibTeX: `/Users/yigitokar/scaling-laws-pf/lit/bib/ml_observational.bib` (keys given in [brackets] below).

How these notes were checked. Every number below comes from the arXiv abstract or HTML page, an official repository, or an Epoch/HF documentation page fetched this session, unless it carries the tag **(my calc)** (my own arithmetic on reported numbers) or **(hypothesis)** (my interpretation, still to be checked). Web search ran out partway through, so a few venue details could not be confirmed. Those are flagged in the bib and in Section 8.

---

## 0. Headline takeaways for the paper

1. **The observational literature already uses fixed-effects production functions, but never IO estimators.**
   - Ruan et al. (2024): family-specific intercepts and slopes on log compute.
   - Sloth (Maia Polo et al.): a translog form taken explicitly "from stochastic frontier analysis", with family-by-skill intercepts read as "efficiency".
   - Mertens et al. (2026): developer and period fixed effects, estimated by OLS on 809 models.
   - Rosetta Stone (Ho et al. 2025): the output elasticity k comes from within-family variation, and algorithmic progress from the trend in the frontier intercept.
   - None of them uses proxy-variable or control-function methods (OP/LP/ACF), dynamic-panel GMM, or selection corrections. Only Whitfill (2025) writes down the endogeneity problem formally. He proves a sign theorem and proposes experiments or instruments, but cites no IO work (no Marschak–Andrews, OP, LP, ACF).
   - **Positioning:** we are not the first to call these "production functions" (Sloth) or to add firm fixed effects (Mertens). What is new is the systematic IO diagnosis and the IO estimators.
2. **The logit-linear benchmark model is a Cobb–Douglas function for the odds of success.**
   - Ruan, Mertens and Rosetta all use a version of logit(score) = a + b·log C + FE. So the odds are e^a·C^b.
   - The family or developer effect is Hicks-neutral TFP on the odds scale.
   - The "compute-equivalent factor" M = 10^{ν/β} (Mertens eq. 3) is the standard conversion of TFP into input units, TFP^{1/(returns to scale)}.
3. **Using compute alone as the input imposes an aggregation restriction.** C = 6ND gives N and D equal unit output elasticities. That is only innocuous on the compute-optimal expansion path.
   - Bhagia et al. (2024): FLOPs "cannot distinguish between compute-optimal and overtrained models".
   - Sloth adds log s, log t and their interaction (a translog) for this reason.
   - Owen (2024) aggregates (N, D) through a fitted Chinchilla loss, which works like a known-technology quantity index.
4. **Within-family designs usually hold D fixed, or set D proportional to N.** Examples in the ObsScaling data: Llama-2 at 2T tokens, Llama-3/3.1 at 15T, Pythia at 0.3T, OPT at 0.18T; Cerebras-GPT has D = 20N exactly.
   - With family fixed effects, the N-elasticity is identified from within-family variation.
   - The D-elasticity rests on a few families where D varies (Gemma-2, Qwen1.5, StarCoder2, Falcon, H2O-Danube, …).
   - Cerebras-GPT is literal ACF-style functional dependence: log D ≡ log N + log 20.
5. **Algorithmic progress = TFP growth, and estimates run from about 2× to 9× a year.** They depend on the output measure, the reference algorithm, the scale, and the fixed effects used.
   - Ho et al.: effective compute doubles every 8.4 months (95% CI 4.5–14.3), about 2.7×/yr (my calc).
   - Rosetta: about 6×/yr (range 1–50×).
   - Mertens: 7.5× over about 1.5–2 years from period effects, about 3–4×/yr (my calc).
   - Gundlach et al. (Price of Progress): about 3×/yr from inference prices.
   - Gundlach et al. (Origin): 6,930× over 2012–2023 at the frontier, against less than 100× at small scale.
   - Densing law: capacity density doubles every 3.3 months (arXiv) or 3.5 months (Nature Machine Intelligence).
   - Scale dependence (Gundlach) means **non-neutral technical change**. It also creates an **index-number problem**: measured progress depends on the reference algorithm.
6. **Ho et al. show the Diamond–McFadden–Rodriguez identification problem in their own estimates.**
   - α_year and β_year are each insignificant and move in opposite directions across bootstrap draws. Only the combined effective-compute rate is pinned down.
   - Their observational β_data = 0.040 is far below experimental values (about 0.28–0.37). This gap drives Whitfill's "overstated by around a factor of nine" argument, which is essentially attenuation plus simultaneity bias.
7. **Output measurement is the TFPR-versus-TFPQ problem.**
   - Benchmarks are bounded, with guessing floors and ceilings.
   - They are contaminated (GSM1k drops of up to 8 points).
   - Models are trained on the test task. Dominguez-Olmedo et al. estimate a post-Nov-2023 advantage θ ≈ 7 points on MMLU and 19 on GSM8K at equal compute. It vanishes once every model is fine-tuned on the same task data, and the Pareto-frontier improvement area shrinks sixfold.
   - Scores are also sensitive to prompt format (up to 76 points).
   - So period effects ("shared algorithmic progress") partly measure benchmark-specific shifts in demand or targeting, not general capability.
8. **Selection problems.**
   - Epoch's notability rule includes "state of the art performance", which is selection on the outcome.
   - Leaderboard entries are self-submitted.
   - Unreleased or failed runs are never observed. This is survivorship, the analog of exit in OP.
   - Frontier statistics (densing ρ_max, Mertens' minimum compute needed to reach a threshold, Rosetta's highest-b frontier) are extreme-value statistics that rise with the number of entrants.
9. **Input measurement error is documented and can be quantified.**
   - Epoch labels its compute estimates with 90% CIs of about ±3× (Confident), ±10× (Likely) and ±31× (Speculative). In log10 terms that is an error SD of about 0.29, 0.61 and 0.91 orders of magnitude (my calc). This gives known reliability ratios for errors-in-variables corrections.
   - Closed-model compute in Mertens et al. comes from lifearchitect.ai, which they call "subject to substantial uncertainty".
   - Within-family (fixed-effects) estimators make attenuation worse (Griliches–Hausman logic).
10. **Productivity dispersion looks like Syverson's facts.**
    - Mertens: model-level compute efficiency differs 41× between the 90th and 10th percentiles, conditional on developer and date (186× on MATH Level 5).
    - Developer compute factors range from 2.3× (DeepSeek) to about 61× (Microsoft, Nvidia) relative to small developers.
    - **(hypothesis)** The Microsoft and Nvidia outliers reflect distillation, pruning or synthetic data: teacher compute is an unmeasured intermediate input. That is the gross-output versus value-added issue (Gandhi–Navarro–Rivers).

---

## 1. A common econometric template that nests the strand

For model m, family or developer f(m), release period t(m), and benchmark b:

```
Measurement:  y_{mb} = γ_b + (1-γ_b) · σ( a_b (θ_m − d_b) ) + u_{mb}                (IRT / sigmoid link; floor γ_b)
Production:   θ_m   = h(N_m, D_m; β) + ω_{f(m)} + δ_{t(m)} + ε_m                      (latent capability = log output)
Inputs:       C_m ≈ 6 N_m D_m  (observed with error: log C^obs = log C + v_m)
Choice:       (N_m, D_m) chosen by the lab after observing (ω_f, δ_t, part of ε_m)      <- simultaneity (Marschak–Andrews)
Selection:    model m is observed if released / notable / submitted: 1{θ_m − c(C_m) > τ}  <- selection on outcome
```

Where each paper sits in this template:

| Paper | Output y (link) | h(·) | Family/firm effect | Time effect | Estimator | Sample | Endogeneity discussed? |
|---|---|---|---|---|---|---|---|
| Ruan, Maddison, Hashimoto 2024 | 8 benchmarks → PCA (K=3); logit of normalized error, ceiling h∈[0.8,1] | θ_f log C + ν_f | family slope and intercept (both vary) | none (families span time) | PCA + per-family linear regression | 77 base models, 21 families (repo: 211 rows) | No. Contamination and within-family heterogeneity listed as limitations |
| Maia Polo et al. (Sloth) | 12 Open LLM Leaderboard v1/v2 benchmarks; floor γ_j + monotone σ_j (sigmoid or neural net); d=3 latent skills | translog: β_k'(log s, log t, log s·log t) | α_ik family×skill intercept ("efficiency"); slopes shared | none | Huber loss (δ=0.01), Adam; Geomin rotation | 30 families (53 counting base and instruct separately), 164 models | No |
| Owen 2024 | BBH, MMLU, BIG-Bench tasks; sigmoid in "scaled compute" (reducible-loss equivalent) | Chinchilla L(N,D) index | none (pooled) | none | NLS, backtest | 11 families; 44 model sizes on BBH | Mentions algorithmic-progress confounding |
| Mertens, Fischl-Lanzoni, Thompson 2026 | MMLU-Pro (guess-normalized, logit); MATH L5 robustness | β_c log(6ND) | developer FE (10 majors plus size-grouped "other") | 3 period FE (2022q4–2023q3, 2023q4–2024q2, 2024q3–2025q1) | OLS, SEs clustered by developer; Shapley R² decomposition | 809 models (Oct 2022–Mar 2025); 122 from majors | No |
| Ho et al. 2024 (LM algorithmic progress) | perplexity (log loss) on WT103/WT2/PTB | Chinchilla with N_eff = N e^{α'(Y−Y0)}, D_eff = D e^{β'(Y−Y0)} | none (dataset dummies only) | exponential factor-augmenting trends | NLS; LOOCV over about 90 specs; bootstrap | about 231 models, 2012–2023 | Mentions data quality vs algorithms and compute–time correlation; no lab FE |
| Whitfill 2025 | same as Ho | same, plus lab-specific ε_N, ε_D | latent lab quality | same | theory plus Monte Carlo | — | **Yes**: sign theorem; no fix beyond experiments or IV |
| Ho, Denain, Atanasov, Albanie, Shah 2025 (Rosetta) | 38 benchmarks, 2PL-style σ(α_b(C_m − D_b)) | k log F + b | within-family k (Llama, Llama 2, Llama 3.1) | trend in frontier b | least squares with L2 penalty (0.1) | 179 models, 1,324 scores | Acknowledges the compute–algorithm correlation |
| Xiao et al. 2024/25 (densing) | 5 benchmarks via loss→sigmoid | reference-model L(N, D=1T) | none | trend in ln ρ_max | OLS on frontier | 29 open base models | Mentions contamination only |
| Erdil & Besiroglu 2022 (vision) | ImageNet top-1, σ^{-1}(σ(C)σ(D)) | logistic in log compute and log data | none | factor-augmenting year terms | MAP with priors; bootstrap | 124 models | Threshold sensitivity; reimplementation bias |
| Dominguez-Olmedo, Dorner, Hardt 2025 | MMLU, GSM8K accuracy | α·max(0, log C − c_e) + θ·1{post Nov-2023} + r | none | post-Nov-2023 dummy | OLS; "treatment" = equal fine-tuning | 56 base models (70M–70B) | **Yes, on output**: training on the test task confounds progress |

---

## 2. Paper-by-paper notes

### 2.1 Ruan, Maddison & Hashimoto (2024), "Observational Scaling Laws and the Predictability of Language Model Performance" [ruan2024observational]

- arXiv 2405.10938 (v1 May 17 2024; v3 Oct 1 2024). **NeurIPS 2024 spotlight** (arXiv comments).
- **Model (eqs. 3–7):**
  - (3) σ^{-1}(E_m) ≈ β'S_m + α, where E_m is normalized error and S_m ∈ R^K are latent capabilities.
  - (4) S_m ≈ θ_f log C_m + ν_f, family-specific.
  - (5) B_{i,m} ≈ γ_i' S_m, with orthonormal γ_i (PCA).
  - (6) E_m ≈ h·σ(β'S_m + α), with h ∈ [0.8, 1.0] to handle the ceiling or floor mismatch.
  - (7) Within a family, P_m := β*'S_m + α* = w_f log C_m + b_f, with w_f = β*'θ_f and b_f = β*'ν_f + α*.
- **"f-equivalent FLOPs" (Sec. 3.4):** log C̄_{m,f} := (1/w_f*)(β*'S_m + α* − b_f*). This maps every model to the compute a reference family (Llama-2) would need to match it. It is the same object as "TFP expressed in input units".
- **Capability space:** PCA on the benchmark×model matrix (PCA imputation for missing entries). The top 3 PCs explain about 97% of variance and PC-1 about 80%. The PCs read as general, reasoning and programming.
- **Heterogeneity:** "Model families only vary in their efficiency in converting training compute to capabilities." Both slope θ_f and intercept ν_f vary by family, which makes this a heterogeneous-coefficients production function. Sloth criticizes it because it needs 3–5 models per family.
- **Data:** 77 pretrained base models in 21 families, plus instruct and proprietary models. Benchmarks: MMLU, ARC-C, HellaSwag, Winogrande, GSM8K, HumanEval, TruthfulQA, XWinograd, from the Open LLM Leaderboard, EvalPlus and LM Eval Harness. Compute is C ≈ 6ND.
- **Validation:** Training cutoff is FLOPs ≤ Llama-2-7B (8.4×10^22), giving 47 training and 30 test models. There was preregistration: forms were frozen in May 2024 and tested on 20 later models (Llama-3.1-405B, Qwen2-72B, …).
- **Family facts:** Phi is "a clear outlier in compute efficiency". It benefits from chain-of-thought earlier but its capabilities scale less steeply with compute. DeepSeek-Coder shows a similar pattern. No table of family multipliers is given.
- **Limitations (their words):** the approach does not account for "potential benchmark contamination" or "heterogeneity within model families". It is "primarily applicable to post-training scaling analyses". There is no discussion of endogeneity or selection.
- **Data:** https://github.com/ryoungj/ObsScaling (Apache-2.0).
  - `eval_results/base_llm_benchmark_eval.csv` has 211 rows. Columns: Model, Model Family, Model Size (B), Pretraining Data Size (T), FLOPs (1E21), and the 8 benchmarks.
  - About 45 families, including Llama, Llama-2, Llama-3, Llama-3.1, Qwen, Qwen1.5, Qwen2, Gemma, Gemma-2, Yi, Yi-1.5, DeepSeek-*, Falcon, Phi, Pythia, BLOOM, OPT, XGLM, CodeLlama, StarCoder(2), OLMo, Cerebras-GPT, SmolLM, StableLM, RWKV, Mixtral, …
- **IO reading:**
  - Eq. (4) is a random-coefficients or heterogeneous-technology production function (in the spirit of Mairesse–Griliches, "are there stable production functions?").
  - The f-equivalent FLOPs are productivity stated in input units.
  - Estimating family by family with OLS assumes log C within a family is uncorrelated with model-level shocks. That is plausible if sizes are fixed menus chosen ex ante for deployment, and implausible if later sizes respond to early results.
- **Within-family variation in the repo CSV (my read via WebFetch; to be recomputed from the CSV):**
  - D is constant across sizes for Llama-2 (2T), Llama-3 and Llama-3.1 (15T), OPT (0.18T), Pythia (0.3T), XGLM (0.5T), StarCoder (1T), MPT (1T), Gemma (6T), Yi (3T), Yi-1.5 (3.6T), DeepSeek-LLM and DeepSeek-Coder (2T).
  - Cerebras-GPT has D = 20N exactly (0.111B/2.22B … 6.7B/134B tokens), so log D is perfectly collinear with log N.
  - D varies independently only in a handful of families: Gemma-2 (2B/2T, 9B/8T, 27B/13T), Qwen1.5 (2.4T vs 4T), StarCoder2, Falcon, H2O-Danube, Llama-1, GPT-Neo/J, StableLM, SmolLM, OLMo.
  - D is missing for Qwen2, Mistral, Mixtral, RWKV, GPT-2, InternLM2 and others.

### 2.2 Maia Polo, Somerstep, Choshen, Sun & Yurochkin, "Sloth: scaling laws for LLM skills to predict multi-benchmark performance across families" [maiapolo2024sloth]

- arXiv 2412.06540 (v5 Dec 1 2025). Venue not confirmed; an OpenReview PDF exists (id 9GN5Jsa3lv).
- **Model:**
  - (3.1) η_i(s,t) = Λ θ_i(s,t) + b, with loadings Λ ∈ R^{J×d} and d ≪ J.
  - (3.2) θ_ik(s,t) = α_ik + β_k' x(s,t), with x = (log s, log t, log s·log t), where s = parameters and t = tokens.
  - Link: μ_ij = γ_j + (1−γ_j) σ_j(η_ij), where γ_j is the lower asymptote (e.g. 0.25 for 4-option MMLU) and σ_j is a sigmoid or a monotone neural net.
- **Explicit economics link:** "Inspired by models used in Economics, we use the family of translog production functions from stochastic frontier analysis" (citing Kumbhakar & Lovell). And: "In economic terms, the intercept term α_ik can be interpreted as an efficiency measure of the family i in converting compute to performance for skill k". α_ik "will absorb all hidden factors specific to family i such as data quality, post-training factors, etc."
- **Heterogeneity:** family-by-skill intercepts (firm × output fixed effects); slopes shared across families. Instruct and base versions are treated as different families, and cross-validation holds out the twin family. **This is a fixed-effects frontier model in the Schmidt–Sickles (1984) sense.** It has no one-sided error, so it is "SFA" only in functional form.
- **Identification (Thm A.2):** parameters are identified up to an invertible d×d rotation M: Λ̃ = ΛM, B̃ = B(M')^{-1}. It is orthogonal if the skills are uncorrelated. The authors apply a Geomin oblique rotation. Estimation minimizes Huber loss (δ=0.01) with Adam.
- **Data:** Open LLM Leaderboard v1 and v2. 12 benchmarks: GSM8K, MATH Lvl 5, MMLU, MMLU-PRO, BBH, GPQA, MUSR, TruthfulQA, HellaSwag, Winogrande, ARC, IFEval. 30 families (28 in v1, 17 in v2, 15 overlapping); 164 models listed in Appendix G. d=3 fits best (reasoning, knowledge, instruction following).
- **Results:** leave-one-family-out MAE is about 1–1.5 points per benchmark when only the smallest model (or two) of a new family is seen. It is competitive with or better than a "Size and Tokens" baseline and better than a pooled FLOPs model. An interaction term is motivated by Pythia (few tokens, flat slope in log s) versus Qwen2 (many tokens, steep slope), i.e. **complementarity between N and D**.
- **Compute-optimal allocation (Sec. 4.6, Table 1):** max over (u, v) of α + β0·u + β1·v + β2·uv s.t. u + v = l (log compute), with u and v kept inside the observed support (box constraints).
  - Budgets (×10^19 FLOPs) 100 / 578 / 3346 / 19360 / 112005 / 648000 give params (B) 0.16 / 0.30 / 0.72 / 2.15 / 12.44 / 72.0 and tokens (T) 1.04 / 3.24 / 7.78 / 15 / 15 / 15.
  - The 15T cap is a corner solution at the edge of the data support, not a modeled data constraint (my reading). Optimal allocations do not depend on family because slopes are shared, so families are Hicks-neutral shifters.
- **Instruction tuning as treatment:** it raises instruction following for all families, lowers reasoning moderately, and has mixed effects on knowledge. This makes post-training a multi-output shifter.
- **Missing:** no discussion of endogeneity, selection or contamination.
- **Data:** https://github.com/felipemaiapolo/sloth (MIT). The `data/` folder contains data_v1.csv, data_v2.csv, open-llm-leaderboard_old.csv / _new.csv, training_tokens.csv, tokens.csv, base_llm_benchmark_eval.csv, instruct_llm_benchmark_eval.csv, lower_bounds.csv, and others.
- **IO reading:** a multi-output translog with firm×output fixed effects and common technology, in the spirit of multi-product production functions. The shared-slope restriction is testable (a Chow or Wald test of family-specific β).

### 2.3 Owen (2024), "How predictable is language model benchmark performance?" [owen2024predictable]

- arXiv 2401.04757 (Epoch).
- **Form:** a sigmoid (1+exp(−k(x−x0)))^{-1}, where x is reducible loss expressed as "scaled compute". (N, D) are mapped through a Chinchilla L(N,D) and converted to compute-optimal-equivalent compute. Offset-sigmoid and clipped-linear variants were also tried. It is fit both pooled across tasks and per task.
- **Data:** about 11 families (PaLM, PaLM-2, GPT-3, Gopher, BIG-G, Chinchilla, GLM, BloombergGPT, GPT-NeoX, OPT, BLOOM; also LLaMa, Falcon, MPT, Yi). 44 model sizes on BBH, 36 on full BIG-Bench. Range: 2M–540B parameters, 1e18–1e24 FLOPs.
- **Backtests:**
  - Aggregate BBH error is about 6 points over one order of magnitude of compute (about 4 points over 0.33 OOM, 12.5 over 2 OOM).
  - Individual BIG-Bench tasks: 18 points on average.
  - MMLU: 10–20 points; about 17 points per OOM when starting above 2e23.
  - A constant baseline has about twice the error.
- **Heterogeneity:** pooled, no family effects. Owen notes "we may be underestimating reducible loss for older models and overestimating reducible loss for newer models" (algorithmic progress is omitted). Fitting on raw training compute instead of reducible loss is "similar, but slightly worse". Pre-Chinchilla models such as PaLM fall below trend.
- **IO reading:** using a pre-estimated technology (Chinchilla) to aggregate inputs is the index-number approach: a quantity index built with known weights. Omitting TFP growth biases the result in a direction that depends on vintage.

### 2.4 Mertens, Fischl-Lanzoni & Thompson (2026), "Is there 'Secret Sauce' in Large Language Model Development?" [mertens2026secret]

- arXiv 2602.07238 (v1 Feb 6 2026; v2 May 3 2026). cs.AI, cross-listed econ.GN. **The closest precedent to our cross-lab empirics.**
- **Eq. (1):** Y_i = β0 + β_c log(c_i) + δ_t + ν_j + ε_i.
  - Y is the logit of MMLU-Pro, normalized for random guessing; MATH Lvl 5 is a robustness check.
  - c = 6ND.
  - δ_t is 3 release periods; ν_j is developer dummies (10 majors plus size-grouped "other").
  - Estimated by OLS with SEs clustered by developer.
- **Compute factor (eq. 3):** M = 10^{δ/β_c}, applied to period, developer and residual effects. This is exactly TFP converted to input-equivalents.
- **Table C.1:**
  - log FLOPs: 0.789 (SE 0.192), full sample N=809, R²=0.526. Major developers only: 0.885 (0.142), N=122, R²=0.637.
  - Developer coefficients (vs small "other"): DeepSeek 0.282 (0.224); Qwen 0.667*** (0.198); Meta 0.535* (0.278); Google 0.627*** (0.198); Microsoft 1.407*** (0.086); OpenAI 0.805 (0.692); Anthropic 0.740 (0.663); xAI 0.890 (0.745); 01-AI 1.071*** (0.159); Nvidia 1.412*** (0.070).
  - Implied compute factors 10^{ν/0.789} **(my calc)**: DeepSeek 2.3× (matches the paper's 2.3×), Meta 4.8×, Google 6.2×, Qwen 7.0×, Anthropic 8.7×, OpenAI 10.5×, xAI 13.4×, 01-AI 22.8×, Microsoft 60.7× (the paper reports 60.5×), Nvidia 61.6×.
- **Headline results:**
  - At the frontier, 80–90% of performance differences come from compute (Shapley decomposition of R²). Frontier compute grew about 5,000× from the earliest to the latest top model.
  - Shared algorithmic progress: "achieving a given MMLU-Pro score in early 2023 would have required 7.5x more compute than achieving the same score in late 2024". That is roughly 3.2–3.8×/yr **(my calc)**, depending on the exact span.
  - Within-developer dispersion: model-level compute efficiency at p90 is 41× that at p10, conditional on developer and period; 186× on MATH L5.
  - Compute needed to reach 15% MMLU-Pro fell 50× (major developers) to 8,000× (including small developers).
  - Variance shares, full sample: scaling 32%, shared progress 3–10%, developer 14–34%, model-specific 32–47%. Among majors, scaling is about 45% and developer about 34%.
  - Period×log-compute interactions are positive but insignificant, so the data do not reject Hicks-neutral time effects.
- **Data sources:** HF Open LLM Leaderboard (4,000+ models, harmonized MMLU-Pro, parameters, dates); Epoch AI; TIGER-Lab for proprietary scores; hand-collected token counts; lifearchitect.ai for proprietary compute (results robust to excluding it, per their App. D).
- **Data availability:** dataset and code released (a Google Drive link in the paper).
- **What is missing:**
  - Only 9 references and no productivity literature: no OP, LP, ACF, De Loecker, Syverson.
  - No discussion of endogeneity, selection or contamination.
  - No split of log C into log N and log D.
  - No handling of derivative models (community fine-tunes) or distillation.
- **(hypothesis)** Microsoft (Phi: synthetic textbook data) and Nvidia (Minitron/Nemotron: pruning and distillation) have the largest effects. If so, the "secret sauce" is partly an **omitted intermediate input** (teacher compute and synthetic data). Check the model list.
- **IO reading:**
  - A textbook within-industry productivity-dispersion regression (Syverson 2004/2011), with firm and year fixed effects.
  - The 41× p90/p10 figure is directly comparable to the TFP p90/p10 ratios Syverson reports for US manufacturing (about 1.9× for 4-digit industries; check against the IO strand).
  - The "15% threshold" analysis traces the input-requirement function over time: a cost-function-dual measure of technical change.

### 2.5 Ho, Besiroglu, Erdil, Owen, Rahman, Guo, Atkinson, Thompson & Sevilla (2024), "Algorithmic progress in language models" [ho2024algorithmic]

- arXiv 2403.05812 (Mar 2024). The Epoch blog cites it as arXiv; NeurIPS 2024 publication is not confirmed.
- **Specification:**
  - Eq. (1): L = E + A/N^α + B/D^β.
  - Eqs. (2–3): N_eff ≡ N exp(α'(Y−Y0)), D_eff ≡ D exp(β'(Y−Y0)), so L = E + A N^{−α_param} e^{−α_year(Y−Y0)} + B D^{−β_data} e^{−β_year(Y−Y0)}.
  - Fitted form, eq. (8): L = exp[α'_const − α_year(Y−Y0) − α_param log(N/N0)] + exp[β'_const − β_year(Y−Y0) − β_data log(D/D0)], with dataset dummies (PTB, WT2) in the constants. E is dropped in the preferred spec.
  - Transformer variant, eq. (7): reducible loss multiplied by σ(γT) when the model is a transformer.
- **Model selection:** about 90 specifications compared by LOOCV. The preferred "Model 7" has benchmark-specific A and B and shared exponents (test MSE 0.0486). "Model 10" is marginally better (0.0485) but rejected for parsimony and because it gives dataset-specific progress rates. Held-out R² is about 0.91.
- **Table 2 (95% bootstrap CIs):**
  - α_const 0.913 [0.000, 1.208]; α_year 0.004 [−0.058, 0.032]; α_param 0.068 [0.045, 0.127]*.
  - β_const 0.771 [0.233, 1.293]; β_year 0.036 [−0.002, 0.080]; β_data 0.040 [0.023, 0.062]*.
  - Doubling times: T_N = ln2·α_param/α_year, T_D = ln2·β_data/β_year, T_C = (1/T_N + 1/T_D)^{-1}.
  - From the point estimates **(my calc)**: T_N ≈ 11.8 years and T_D ≈ 9.2 months, giving T_C ≈ 8.7 months. The point estimates therefore attribute almost all progress to **data-augmenting** efficiency, but that split is not identified; see the next bullet.
- **Diamond–McFadden–Rodriguez in the paper's own words:** "when α_year is positive, β_year is negative and vice versa, such that the overall estimated effective compute doubling time is always positive". Also, "the confidence intervals for α_year and β_year are not statistically significant at the 5% significance level, while α_param and β_data are." **Only the Hicks-like aggregate rate is identified. The factor bias is not.**
- **Headline:** effective-compute doubling time 8.4 months [4.5, 14.3]. An alternative numerical procedure (compute-optimal C = 6ND) gives 8.6 [4.5, 14.5]; the top-10 CV models give 7.8 [1.5, 17.6]. The abstract says "halved approximately every 8 months (95% CI 5–14 months)". In annual terms, 8.4 months is about 2.7×/yr (range 1.8–6.3×) **(my calc)**. Epoch's later summary table lists "3× per year (95% CI 1.5× to 64×)".
- **Decomposition:**
  - Effective compute "expanded by about 22-billion-fold since 2014, with slightly under two-thirds of the scaling being due to increased use of actual, physical computing resources".
  - Shapley shares (Table 1). RNN 2012 → GPT-3: parameter scaling 48.6%, data scaling 32.4%, parameter efficiency 2.1%, data efficiency 16.8%. Transformer 2018 → GPT-3: 56.8 / 35.9 / 0.8 / 6.4%.
  - The blog summarizes this as 60–95% of gains from compute and data and 5–40% from algorithms.
- **Transformer and Chinchilla:** the transformer's compute-equivalent gain is 7.2× [3.3, 45.7] (6.6× at 1e22 FLOPs), and it cuts reducible loss by 4.6% [3.0, 7.0]. That is about 20% of algorithmic gains, or about 2 years of progress. The Chinchilla re-balancing gives a 2–4× compute reduction at the frontier, 8–16 months of progress (blog).
- **Data:**
  - About 410 models from 226 papers, of which about 231 are usable. At most 3 models per paper (autocorrelation control, which drops about 50).
  - 103 transformer and 127 non-transformer models.
  - Exclusions: retrieval, compression/pruning, NAS, distillation, cache models, non-standard tokenization.
  - A vocabulary fixed effect as a robustness check gives 8.0 months [4.0, 17.1].
- **Threats they acknowledge:**
  - They cannot separate data quality from data-use efficiency.
  - Some innovations are scale-dependent.
  - Evaluation inconsistencies: word vs subword tokenization (about 30%), preprocessing (about 10%), stride (about 10%).
  - Later models are larger, which "can make it hard to disentangle their relative contributions".
  - Context length is not controlled.
  - **No lab or organization effects, and no treatment of selection.**
- **Data:** https://github.com/epoch-research/lm-algorithmic-progress (MIT repo; data in a linked Google Sheet).
- **IO reading:**
  - Factor-augmenting technical change with exponential trends inside a CES-like additive loss aggregator, i.e. normalized CES with factor-augmenting progress (Klump–McAdam–Willman; see IO strand).
  - Dataset dummies are output-measurement fixed effects, like industry-specific deflators.
  - Missing firm effects plus endogenous inputs is the Marschak–Andrews bias (Whitfill 2025).
  - β_data = 0.04 against experimental values of about 0.28–0.37 (Hoffmann; Besiroglu et al. replication) is the classic pattern of attenuated input elasticities in observational panels (Griliches–Mairesse 1998).

### 2.6 Whitfill (2025), "Note on Selection Bias in Observational Estimates of Algorithmic Progress" [whitfill2025note]

- arXiv 2508.11033 (econ.GN; v2 Aug 18 2025).
- **Model:** L = E + A/(N q_N)^α + B/(D q_D)^β, with q_N(Y) = exp(α'(Y−Y0) + ε_N) and q_D(Y) = exp(β'(Y−Y0) + ε_D). The ε terms are "within-year, across-lab algorithmic heterogeneity". Compute is chosen by labs "endogenously after observing their algorithmic quality (say on smaller experiments)".
- **Theorem 1:** sign(bias(β̂_year/β̂)) = −sign(Cov(ln D, ε_D)). He argues the covariance is likely negative (better algorithms need less data), which would bias estimated progress upward.
- **Illustration:** Ho's β ≈ 0.04 against an experimental β ≈ 0.37. "Then a first-pass estimate is that algorithmic progress is overstated by around a factor of nine."
- **Monte Carlo:** true progress of 45%/yr is estimated at 16.5% when the correlation is +0.5 and at 93% when it is −0.5.
- **Remedies proposed:** experiments with randomized compute or algorithms, or an instrument for compute. No fixed effects or control functions. **No IO citations.** Epoch's summary reports the claim that growth could be "as much as 9× slower".
- **IO reading:** exactly the Marschak–Andrews (1944) transmission bias applied to a time-trend coefficient. Our paper can supply the OP/LP/ACF and dynamic-panel machinery, together with timing assumptions (small-scale experiments reveal ω before the big run is sized; a proxy is an input chosen after ω is observed).

### 2.7 Erdil & Besiroglu (2022/23), "Algorithmic progress in computer vision" [erdil2022algorithmic]

- arXiv 2212.05153 (v1 Dec 2022; v4 Aug 2023).
- **Model (eqs. 1–3):** σ^{-1}(P) = σ^{-1}(σ(C)·σ(D)) + ε, with C = α1 + α_Year(Year−2012) + α_compute log(compute) and D = β1 + β_Year(Year−2012) + β_data log(data). This is approximately 1−P ≈ Ã C^{−α_compute} + B̃ D^{−β_data}. The product of logistic terms is a complementarity (bottleneck) structure.
- **Data:** 124 ImageNet-1k models (2012–2023), at most 3 per paper, excluding NAS and reimplementations (extends Thompson et al. 2020).
- **Estimation:** MAP with Normal priors (α_year, α_compute, β_year, β_data ~ N(0, 0.09)), prior variance chosen by CV, 100 bootstrap draws. There are multiple local maxima, and 90% intervals are reported because the bootstrap distribution is multimodal.
- **Results:** compute-augmenting progress doubles effective compute every 8.95 months [3.55, 25.40]; the abstract says "every nine months (95% CI 4 to 25 months)". Table 1 has α̂_Year = 0.159 and α̂_compute = 0.154, implying about 8.1 months **(my calc)**. The reported median likely comes from the bootstrap.
  - Compute-augmenting innovations explain 4–11× more variation than data-augmenting ones.
  - AlexNet → ResNet-50 Shapley shares: 64.9% algorithms, 35.1% compute.
  - The halving time is very sensitive to the chosen performance threshold (7.72 down to 0.67 months for accuracy 0.60–0.88).
- **IO reading:** factor-augmenting technical change estimated by Bayesian shrinkage. Weak identification appears as multimodality. No economics citations.

### 2.8 Hernandez & Brown (2020), "Measuring the Algorithmic Efficiency of Neural Networks" [hernandez2020measuring]

- arXiv 2005.04305 (OpenAI).
- **Method:** hold output fixed (AlexNet-level, 79.1% top-5) and measure the minimum training compute over time. This traces the **input-requirement set / isoquant shift**, the dual of productivity growth.
- **Results:**
  - 44× less compute in 2019 (EfficientNet, 0.069 pfs-days) than 2012 (AlexNet, 3.1 pfs-days). Doubling every 16 months (15.4 months **(my calc)** from 44× over 7 years). Moore's law would give 11×.
  - ResNet-50 level (92.9% top-5): 10× (2015 → 2019).
  - Translation: Seq2Seq level (34.8 BLEU, WMT-14 En-Fr) 61× from 2014 to 2017 (Transformer Base); GNMT level 9× (2016 → 2017).
  - Data: github.com/openai/ai-and-efficiency.
- **Caveats (Erdil & Besiroglu):** reimplementations benefit from software improvements, and the estimate depends on the threshold.
- **IO reading:** a fixed-output input-requirement index, i.e. cost-function-dual TFP. Because it is the minimum over entrants, it is an extreme-value statistic.

### 2.9 Gundlach, Fogelson, Lynch, Trišović, Rosenfeld, Sandhu & Thompson (2025), "On the Origin of Algorithmic Progress in AI" [gundlach2025origin]

- arXiv 2511.21622 (Nov 2025, MIT FutureTech).
- **Compute-equivalent gain (CEG) function (their definition):** f such that algorithm A with compute C and algorithm A′ with compute C/f(C) reach equal performance, for all C. A CEG multiplier is f evaluated at one point.
- **Ablations** on a 3.6M-parameter transformer at NLL = 5.3:
  - SwiGLU 1.17×, rotary 1.44×, pre-RMSNorm 1.87×, LR schedules < 1.05×, Adam 1.87×.
  - Modern transformer vs LSTM: 6.28×, against 16.66× if the gains multiplied, so the innovations are **sub-multiplicative**.
  - Overall, "we are able to account for less than 10× of these gains … yielding a total under 100×", compared with Ho et al.'s 22,000× for 2012–2023.
- **Scale dependence:**
  - LSTM → Transformer has **different scaling exponents**. The gain is 6.28× at small scale but extrapolates to about 725–846× at the frontier.
  - Post-2017 transformer tweaks show "near-identical scaling exponents", so they are scale-invariant.
  - Kaplan → Chinchilla re-balancing: about 10× at the frontier, with a closed-form CEG (App. B).
  - Together, LSTM → Transformer and Chinchilla account for 91% of the 6,930× total at the 2025 frontier. Other contributions: scale-invariant innovations about 2.6×, MoE about 2×, data about 3×, tokenizers about 1.6×.
  - The 6,930× total is about 2.23×/yr; the unexplained gap to Ho et al. is about 3.18×.
- **Reference dependence:** "Progress may appear exponential relative to one baseline algorithm, yet be zero relative to another". With MoE measured against LSTMs, growth is about 63%/yr; against dense transformers it is 0% (a constant 2×). "Model trainers with a large computational budget see much greater improvements with the introduction of new algorithms than smaller builders."
- **IO reading:**
  - Scale-dependent CEG is **non-neutral technical change** that changes an output elasticity, i.e. a non-homothetic or scale-biased technology shift.
  - Reference dependence is the **index-number problem** of productivity measurement (Laspeyres vs Paasche; Caves–Christensen–Diewert).
  - A precise statement we can prove **(my derivation)**. Within the Chinchilla family, L*(C) − E = G·C^{−γ} with γ = αβ/(α+β). A factor-augmenting change (A → Aλ_N^{−α}, B → Bλ_D^{−β}) only rescales G, so the CEG is constant in C. A scale-dependent CEG requires a change in the exponents (α, β, hence γ) or in E. **So "Hicks-neutral / factor-augmenting within the family" is equivalent to "constant CEG", and Gundlach's finding amounts to rejecting common exponents across architectures.**

### 2.10 Sanderson, Foley, Guo, Qu & Josephson (2025), "Rethinking LLM Advancement: Compute-Dependent and Independent Paths to Progress" [sanderson2025rethinking]

- arXiv 2505.04075 (v2 Jun 2025).
- Classifies innovations as compute-dependent (Transformer, MoE, sparse attention) or compute-independent (RoPE, FlashAttention, LayerNorm).
- nanoGPT experiments at 50M and 110M parameters:
  - RoPE CEG 1.7×, LayerNorm 1.67×, FlashAttention a 2–4× practical speedup; combined up to 3.5×.
  - Compute-dependent innovations are below 1 at small scale and rise with scale (MQA 0.673 → 0.931×; sparse attention 0.515 → 0.964×).
  - Historical estimates: Transformer 20–50×, MoE 7–11×, sparse attention 4.8–7×.
- **IO reading:** the same non-neutrality point as Gundlach.

### 2.11 Ho, Denain, Atanasov, Albanie & Shah (2025), "A Rosetta Stone for AI Benchmarks" [ho2025rosetta]; Epoch Capabilities Index [epoch2025eci]

- arXiv 2512.00193 (Nov 28 2025; commissioned by Google DeepMind).
- **Eq. (1):** score(m,b) = σ(α_b(C_m − D_b)). It is 2PL-IRT-like at the benchmark level, without a guessing floor in this form. Normalization: α_WinoGrande = 1 and D_WinoGrande = 0, which works like a base-period price index.
- **Estimation:** least squares (scipy least_squares, TRF) with an L2 penalty of 0.1. Data: 179 models, 38 benchmarks, 1,324 scores; each model has at least 4 benchmarks.
- **Algorithmic efficiency (eq. 2):** C_m = k log F_m + b.
  - k is estimated **within families with identical recipes** (Llama, Llama 2, Llama 3.1) and averaged: k = 0.168. Llama 3.1 alone gives 0.12, which they are "more suspicious" of. Note that within Llama 3.1, D is fixed at 15T, so k reflects only the N-elasticity.
  - b is fit for every model. The frontier trend is Δb/yr = 0.297, so "training compute needed to reach a certain capability has been declining at exp(0.297/0.168) ≈ 6× per year" (range 1–50×). The highest-b models trace about 9×/yr (range 3–40×).
- **Capability growth:** 0.55 units/yr [0.45, 0.67]. They detect about a 1.95× acceleration with a breakpoint around April 2024 (slope 0.352 → 0.689/yr); a synthetic-data false-positive rate of 38% limits the claim.
- **Threats they acknowledge:**
  - "High correlation between training compute and algorithmic innovations". Same-recipe families are their identification device.
  - Constant k across scales is untested ("algorithmic progress could also involve changing k").
  - Saturated or new benchmarks sit on the flat part of the sigmoid, which inflates difficulty.
  - Model selection favors "highly cited" and notable models and avoids models "heavily optimized for a particular benchmark".
  - A single dimension is not enough: Claude is strong at coding, Gemini at multimodal tasks.
- **ECI:** built from over 50 benchmarks. The rescaled display anchors Claude 3.5 Sonnet at 130 and GPT-5 at 150. CSVs of scores and benchmark difficulty parameters are downloadable under CC BY 4.0 (https://epoch.ai/benchmarks/eci).
- **IO reading:**
  - IRT is the measurement model for **latent output** from multiple noisy, bounded indicators. The anchoring is an index-base normalization.
  - The within-family k is literally the **within (fixed-effects) estimator** of the output elasticity. The frontier-b trend is **frontier TFP growth** (a Schmidt–Sickles-style panel frontier).
  - Multiple benchmarks per model act as repeated measurements, allowing measurement-error-robust estimation (factor-model or Kotlarski logic).
  - Selective reporting of benchmarks (labs report where they do well) is missing-not-at-random output data and biases b upward for strategic reporters **(hypothesis)**.

### 2.12 Xiao et al. (2024/2025), "Densing law of LLMs" [xiao2025densing]

- arXiv 2412.04315 (Dec 2024). **Nature Machine Intelligence 7, 1823–1833 (2025)**, published Nov 6 2025, cover article.
- **Definition:** ρ(M) = N̂(S_M)/N_M, where N̂ is the "effective parameter size", i.e. the parameters a reference model would need to reach the same performance.
  - Two steps: L = aN^{−α} + bD^{−β} (fit on reference models of 0.005B–0.8B trained on {10, 15, 20, 30, 40, 60}×N tokens), then S = c/(1+e^{−γ(L−l)}) + d.
  - **The inversion uses D = D0 = 1T tokens for the reference.**
- **Law:** ln(ρ_max) = A·t + B, with A ≈ 0.007/day and R² ≈ 0.93, so doubling takes about 3.3 months (ln2/A ≈ 95–99 days). NMI states about 3.5 months.
  - Sample: 29 open base models since Llama-1 (Feb 2023); MMLU, BBH, MATH, HumanEval, MBPP.
  - A rises from 0.0048 to 0.0073 after ChatGPT.
  - Corollary: inference cost halves about every 2.6 months (GPT-3.5 at $20/M tokens in Dec 2022 vs Gemini-1.5-Flash at $0.075/M in Aug 2024, 266.7×).
- **Caveats they give:** contamination. Also, pruned or distilled models have *lower* density than their parents (only Gemma-2-9B exceeds its source). The fitted trend uses only frontier (maximum-density) models.
- **IO critique:**
  - Density is a **single-factor (partial) productivity measure**: output per parameter, like labor productivity.
  - Holding the reference at D = 1T while target models are trained on 15T tokens means **over-training (data deepening) mechanically raises "density"**. This is the partial-productivity bias from input-mix changes, a form of capital deepening. The paper does not discuss over-training.
  - The NMI abstract says density gains were "primarily driven by the expansion of training data scale and enhancement of data quality", which is consistent with this critique.
  - ρ_max is a maximum over entrants, so it rises with the number of models (an extreme-value statistic).
  - There is a sign conflict with Mertens (distilled models high "TFP" there, low density here). It arises because the input is measured differently: parameters here, 6ND compute in Mertens.

### 2.13 Compute, cost and price trends (inputs and cost functions)

- **Sevilla, Heim, Ho, Besiroglu, Hobbhahn & Villalobos (2022), "Compute Trends Across Three Eras of Machine Learning"** [sevilla2022compute]. IJCNN 2022; arXiv 2202.05924.
  - Doubling times (95% bootstrap CI): Pre-DL (1952–2010, n=19) 21.3 months [17.0, 29.3]; DL era (2010–2022, n=72) 5.7 months [4.3, 9.0]; large-scale (Sept 2015–2022, n=16) 9.9 months [7.7, 17.1].
  - Notability criteria: more than 1,000 citations, historical importance, significant SOTA advance, or deployment. "Large-scale" means Z ≥ 0.76 relative to contemporaries.
  - Acknowledged selection biases: toward academic, English-language and "notable" work, with larger and recent models overrepresented.
- **Sevilla & Roldán (2024), Epoch, "Training compute of frontier AI models grows by 4–5x per year"** [sevilla2024training].
  - Notable models 4.1×/yr (90% CI 3.7–4.6; 333 estimates, 2010–May 2024); frontier (running top-10) 5.3×/yr [4.9–5.7]; since 2018 4.2× [3.6–4.9].
  - Language models 9.5×/yr [7.4–12.2] (Jun 2017–May 2024); frontier LMs post-2020 5.0× (80% CI 3.1–7.3).
  - By company: OpenAI 5.3×, Google DeepMind 4.9×, Meta 7.1×.
- **Cottier, Rahman, Fattorini, Maslej, Besiroglu & Owen (2024), "The rising costs of training frontier AI models"** [cottier2024rising]. arXiv 2405.21015 (v2 Feb 2025).
  - Three methods: (i) amortized hardware capex plus energy, with depreciation r = 0.14 OOM/yr from GPU price-performance, cost ≈ start value per chip × chip-hours/(8760) × r·ln10; (ii) cloud rental price × chip-hours; (iii) full development cost including R&D staff.
  - Growth: 2.4×/yr (90% CI 2.0–2.9) since 2016 for amortized cost (3.0× excluding TPUs); cloud 2.5× [2.1–3.1].
  - Sample: 41 frontier models (top-10 compute at release, after Oct 2015), from the Epoch Notable database.
  - Costs: GPT-4 about $40M amortized vs about $800M hardware acquisition; Gemini Ultra about $30M.
  - Shares: R&D staff including equity 29–49% of full development cost; hardware 47–64%; energy 2–6%. The abstract also gives server components 15–22% and interconnect 9–13%.
  - Total development compute is 1.2–4.0× the final run (median 2.2×).
  - Projection: more than $1B by 2027.
  - **IO use:** Nerlove-style cost-function data. Staff costs are a non-compute input missing from 6ND. The R&D experiment compute is the analog of an "intangible capital" input.
- **Cottier, Snodin, Owen & Adamczewski (2025), Epoch data insight, "LLM inference prices have fallen rapidly but unequally across tasks"** [cottier2025llm]. The price to reach a fixed performance level fell 9× to 900× per year across 6 benchmarks (e.g. GPQA-Diamond at GPT-4-Turbo level: $15 → $0.12 per M tokens, Nov 2023 → Dec 2024, about 125×/yr). Method: cheapest model above the threshold.
- **Gundlach, Lynch, Mertens & Thompson (2025/26), "The Price of Progress: Price Performance and the Future of AI"** [gundlach2025price]. arXiv 2511.23455 (v2 Mar 2026).
  - Price for a given performance falls 5–10×/yr at the frontier.
  - "Isolating out open models to control for competition effects and dividing by hardware price declines, we estimate that algorithmic efficiency progress is around 3× per year."
  - The price of running frontier models rises 3–18×/yr.
  - **IO reading:** prices = marginal cost × markup, and they purge markups by using open models (a competitive-fringe assumption), which points toward De Loecker-type markup work.
- **Pilz, Heim & Brown (2023/2025), "Increased Compute Efficiency and the Diffusion of AI Capabilities"** [pilz2025increased]. arXiv 2311.15377; AAAI-25 proceedings (per search result).
  - An "access effect" (more actors can reach a fixed capability) versus a "performance effect" (every actor reaches higher capability). Large investors can keep their lead.
  - **IO reading:** falling costs move both the extensive margin (entry) and the intensive margin.

### 2.14 Davidson, Denain, Villalobos & Bas (2023), "AI capabilities can be significantly improved without expensive retraining" [davidson2023capabilities]

- arXiv 2312.07413.
- **CEG definition:** CEG = C′/C, where C′ is the training compute an unenhanced model would need to match the enhanced model's performance. It is computed at the scale used in the source paper.
- **Estimates:**
  - Chain-of-thought about 9×; few-shot more than 26×; Toolformer more than 20× (about 0.1% of training cost); WebGPT more than 13× (about 0.01%); Minerva data more than 67× (about 10%); InstructGPT RLHF more than 3900× (about 0.31%); LATS more than 10×.
  - "Most surveyed enhancements improve benchmark performance by more than a 5x increase in training compute, some by more than 20x."
  - Fine-tuning usually costs less than 1% of training.
- **Caveats:** non-experimental; bounds or few-point extrapolations; benchmark-dependent (authors choose favorable metrics); scale-dependent.
- **IO reading:**
  - CEG is an **equivalent-input (distance-function) measure** of a complementary intangible input (post-training, scaffolding), missing from the 6ND input vector.
  - For observational scaling: instruct and chat variants and CoT evaluations differ in unmeasured post-training inputs. Pooling them with base models is an omitted-input problem; Sloth handles it by treating them as separate "families".

### 2.15 Emergence, bounded output metrics and predictability

- **Wei et al. (2022), "Emergent Abilities of Large Language Models"** [wei2022emergent]. TMLR 2022; arXiv 2206.07682.
  - An ability is emergent if it "is not present in smaller models but is present in larger models". Examples: 3-digit arithmetic at about 2.3e22 FLOPs (13B), MMLU at about 3.1e23 (175B), TruthfulQA at about 5.0e23 (280B), WiC at about 2.5e24 (540B).
  - "Emergent abilities also crucially depend on other factors such as … data, its quality". PaLM-62B reaches above-random performance with fewer FLOPs, i.e. a TFP shift lowers the threshold.
- **Schaeffer, Miranda & Koyejo (2023), "Are Emergent Abilities of Large Language Models a Mirage?"** [schaeffer2023emergent]. arXiv 2304.15004; NeurIPS 2023 venue not confirmed.
  - Nonlinear or discontinuous metrics produce apparent emergence; continuous metrics give smooth, predictable curves.
  - **IO reading:** a bounded nonlinear transform of latent output. Thresholds are artifacts of the measurement function, as in TFPR vs TFPQ.
- **Schaeffer, Schoelkopf, Miranda, Mukobi, Madan, Ibrahim, Bradley, Biderman & Koyejo (2024/25), "Why Has Predicting Downstream Capabilities of Frontier AI Models with Scale Remained Elusive?"** [schaeffer2024predicting]. arXiv 2406.04391.
  - The transformations from log-likelihood to accuracy progressively degrade the correlation with scale, because probability mass on the incorrect choices matters.
  - Data: 5 families (Pythia, Cerebras-GPT, OLMo, INCITE, LLM360) and 12 benchmarks plus 57 MMLU subjects. About 90% of samples have correlation above 0.75 with log-likelihood, but only about 40% after renormalizing over the available choices.
- **Dominguez-Olmedo, Dorner & Hardt (2025), "Training on the Test Task Confounds Evaluation and Emergence"** [dominguezolmedo2025training]. ICLR 2025 oral; arXiv 2407.07890.
  - Specification: A = α·max(0, log C − c_e) + θN + r + ε, where N is an indicator for post-Nov-2023 models. Data: 56 base models (70M–70B); R² > 0.9.
  - θ̂ is about 7 points on MMLU and 19 on GSM8K. **After fine-tuning every model on the same task data (about 100k MMLU-style examples; about 600k MetaMathQA plus Orca-Math examples), θ̂ becomes small and insignificant.** Older models gain more from that fine-tuning.
  - Emergence moves to smaller scale (MMLU: 1e22 → 6e20 FLOPs); the log-linear R² rises from 0.63 to 0.95; the Pareto-frontier improvement area shrinks sixfold.
  - **IO reading:** the equal fine-tuning is a *deflator* or quality adjustment: it equalizes a benchmark-specific intermediate input before comparing productivity. Period effects in Mertens and Ruan-type regressions are contaminated by this "test-task" input.
- **Fogelson, Brown, Gundlach, Lynch & Thompson (2026), "Two AI Metrics Diverged: Will it Make All the Difference?"** [fogelson2026metrics]. arXiv 2607.00913; ICML 2026 TAIG workshop.
  - "Bounded performance metrics always" favor accessible smaller models; unbounded metrics suggest concentration. Many bounded metrics have unbounded counterparts.
  - **IO reading:** conclusions about convergence or dispersion depend on the output transform (ceiling compression), just as productivity dispersion depends on TFPR vs TFPQ.
- **Gundlach, Lynch & Thompson (2025), "Meek Models Shall Inherit the Earth"** [gundlach2025meek]. arXiv 2507.07931; ICML 2025 TAIG workshop. Diminishing returns to compute under fixed-distribution next-token objectives imply capabilities converge.

### 2.16 Task and downstream scaling from controlled ladders (experimental designs)

- **Bhagia et al. (2024), "Establishing Task Scaling Laws via Compute-Efficient Model Ladders"** [bhagia2024establishing]. COLM 2025; arXiv 2412.04403.
  - Two steps: task loss (bits-per-byte on the correct answer) L(N,D) = A/N^α + B/D^β + E per task, then Acc(L) = a/(1+exp(−k(L−L0))) + b.
  - Ladder: 4 sizes (190M, 370M, 760M, 1.3B non-embedding) × 4 data multipliers (1, 2, 5, 10× Chinchilla) = 16 models, 5.2e21 FLOPs, **under 1% of the targets** (OLMo 2 7B on 4T tokens, 13B on 5T).
  - Errors are within about 2 points on MMLU, HellaSwag, PIQA and SIQA, and about 4 points on average including ARC and OBQA.
  - Using FLOPs as the input gives higher fitting error on all tasks because it "cannot distinguish between compute-optimal and overtrained models".
  - Task-level prediction error correlates with checkpoint-to-checkpoint noise (r = 0.821).
  - **IO reading:** the ladder is a designed factorial experiment over (N, D/N) that breaks functional dependence.
- **Gadre et al. (2024), "Language models scale reliably with over-training and on downstream tasks"** [gadre2024language]. arXiv 2403.08540. 104 models (0.011B–6.9B) with varying over-training; scaling laws that extrapolate in both over-training and N; a power law from perplexity to downstream error.
- **Chen, Huang, Gao, Wang, Yang & Ji (2024/25), "Scaling Laws for Predicting Downstream Performance in LLMs"** [chen2024scaling]. TMLR; arXiv 2410.08527. Two-stage FLOPs → loss → performance (FLP). Predicts 7B and 13B within 5% and 10% using sampling models of 3B or less. FLP-M handles data mixtures with domain-specific losses and a 2-layer net.
- **Isik, Ponomareva, Hazimeh, Paparas, Vassilvitskii & Koyejo (2024/25), "Scaling Laws for Downstream Task Performance of Large Language Models"** [isik2024scaling]. ICLR 2025; arXiv 2402.04177.
  - Translation fine-tuning. Cross-entropy follows L(D_p) = E + A/D_p^α. BLEU and COMET follow a log-law f(D_p) = (log(A·D_p^α))^β when pretraining data is aligned with the task.
  - With misalignment, "downstream cross-entropy monotonically improves [while] translation scores fluctuate or get worse".
  - **IO reading:** the proxy output (loss) and the true output (task quality) diverge. Using loss as output is like TFPR when the "demand" for a particular capability is misaligned.
- **Magnusson et al. (2025), "DataDecide: How to Predict Best Pretraining Data with Small Experiments"** [magnusson2025datadecide]. ICML 2025; arXiv 2504.11393. 25 corpora × sizes up to 1B × 3 seeds, up to 100B tokens. Single-scale (150M) ranking predicts the 1B winner about 80% of the time; no scaling-law baseline does better. **A randomized "recipe × scale" experiment that can test recipe×scale interactions, i.e. non-neutral recipe (TFP) effects.**
- **Heineman et al. (2025), "Signal and Noise"** [heineman2025signal]. arXiv 2508.13144. 30 benchmarks, 375 open-weight models (60M–32B), 900k evaluation results (200M instances), public. Signal-to-noise matters; perplexity metrics beat accuracy. **Useful for measuring error in the output variable.**

### 2.17 Data scarcity (input supply)

- **Villalobos, Ho, Sevilla, Besiroglu, Heim & Hobbhahn (2024), "Position: Will we run out of data? Limits of LLM scaling based on human-generated data"** [villalobos2024run]. ICML 2024, PMLR 235:49523–49544; arXiv 2211.04325.
  - The stock of public human text is about 300T effective tokens (90% CI roughly 100T–1,000T per Epoch's summary; the paper's appendix gives indexed-web and whole-web figures of 510T and 3,100T).
  - Full use arrives between 2026 and 2032 (median about 2028), or about a year earlier with 5× over-training. Historical dataset growth is 0.38 OOM/yr (about 2.4×/yr).
  - Multi-epoch training (Muennighoff) can add up to 3–15× effective data (5× assumed in the realistic case). 10–40% of deduplicated web data is usable.
  - **IO reading:** a quasi-fixed factor with a rising shadow price. Over-training and synthetic data substitute along the isoquant; repetition acts as depreciation of effective data.

### 2.18 Agentic and time-horizon output measures

- **Kwa et al. (2025), "Measuring AI Ability to Complete Long Software Tasks"** [kwa2025measuring]. NeurIPS 2025; arXiv 2503.14499.
  - p_success = σ((log h_agent − log t_task)·β_agent), where h is the 50% time horizon.
  - 12 frontier models (plus 4 near-frontier) on 170 tasks (97 HCAST, 7 RE-Bench, 66 SWAA).
  - Doubling every 207 days [166, 240] (about 7 months; about 3.4×/yr **(my calc)**), from OLS of log h on release date. Evidence for a 2024–25 speed-up is weak ("only seven frontier models").
  - An IRT with human task time as the difficulty scale, which gives an **unbounded (cardinal) output measure**.
- **Pimpale, Højmark, Scheurer & Hobbhahn (2025), "Forecasting Frontier Language Model Agent Capabilities"** [pimpale2025forecasting]. arXiv 2502.15850. Six methods backtested on 38 models from the Open LLM Leaderboard v2. The chosen pipeline is release date → Elo → benchmark, i.e. **time, not compute, as the regressor**.
- **Whitfill, Snodin & Becker (2025), "Forecasting AI Time Horizon Under Compute Slowdowns"** [whitfill2025forecasting]. arXiv 2511.19492. Models time horizon as a function of compute and algorithms, with compute spilling over into algorithmic progress. Constant growth rates imply time-horizon growth ∝ compute growth.
  - **IO reading:** endogenous TFP growth driven by R&D inputs (Griliches knowledge-capital logic).

### 2.19 Contamination and measurement noise in outputs

- **Sainz et al. (2023), "NLP Evaluation in trouble: On the Need to Measure LLM Data Contamination for each Benchmark"** [sainz2023nlp]. arXiv 2310.18018. Position paper that defines contamination levels. The venue is Findings of EMNLP, but the fetched pages disagree on the year (2023 vs 2024).
- **Oren et al. (2023), "Proving Test Set Contamination in Black Box Language Models"** [oren2023proving]. arXiv 2310.17623. Exchangeability test comparing the canonical ordering with shuffled orderings; finds little evidence of widespread contamination in 5 public models.
- **Zhang et al. (2024), "A Careful Examination of Large Language Model Performance on Grade School Arithmetic"** [zhang2024careful]. NeurIPS 2024 Datasets & Benchmarks; arXiv 2405.00332.
  - GSM1k is a fresh twin of GSM8k. Accuracy drops up to 8 points; "several families of models showing evidence of systematic overfitting". Spearman r² = 0.36 between the probability of generating GSM8k items and the performance gap. Frontier models show little overfitting.
  - **IO reading:** family-specific output mismeasurement correlated with family fixed effects.
- **Xu, Guan, Greene & Kechadi (2024), "Benchmark Data Contamination of Large Language Models: A Survey"** [xu2024benchmark]. arXiv 2406.04244.
- **Madaan et al. (2024), "Quantifying Variance in Evaluation Benchmarks"** [madaan2024quantifying]. arXiv 2406.10229. Seed variance and monotonicity; IRT and item analysis help less than recasting multiple-choice as completion.
- **Sclar, Choi, Tsvetkov & Suhr (2024), "Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design…"** [sclar2024quantifying]. ICLR 2024; arXiv 2310.11324. Up to 76 accuracy points from formatting alone, and format effects correlate only weakly across models. This is **non-classical, model-specific output measurement error**.
- **Burnell, Hao, Conway & Hernández-Orallo (2023), "Revealing the structure of language model capabilities"** [burnell2023revealing]. arXiv 2306.10062. 29 LLMs × 27 tasks; factor analysis finds 3 factors (reasoning, comprehension, core language modeling). Supports the low-dimensional latent output of Ruan and Sloth.
- **Maia Polo et al. (2024), "tinyBenchmarks"** [maiapolo2024tinybenchmarks]. ICML 2024; arXiv 2402.14992. About 100 curated items suffice to estimate MMLU (14K items) via IRT.

### 2.20 Other related items

- **Besiroglu, Erdil, Barnett & You (2024), "Chinchilla Scaling: A replication attempt"** [besiroglu2024chinchilla]. arXiv 2404.10102.
  - Hoffmann Approach 3 re-fit: A = 482.01 (124.58), B = 2085.43 (1293.23), E = 1.8172 (0.03), α = 0.3478 (0.02), β = 0.3658 (0.02), versus Hoffmann's 406.4 / 410.7 / 1.6934 / 0.3392 / 0.2849.
  - About 20 tokens per parameter is optimal, vs about 70 implied by Hoffmann's numbers. Hoffmann's CIs are "implausibly narrow", requiring more than 600,000 experiments.
  - This is the experimental benchmark behind Whitfill's β ≈ 0.37.
- **Biderman et al. (2023), "Pythia"** [biderman2023pythia]. arXiv 2304.01373. 16 LLMs from 70M to 12B trained on the same data in the same order, with 154 checkpoints each. A within-family experiment with fixed D. The ObsScaling CSV lists 0.3T tokens for all Pythia models.
- **Douglas & Verstyuk (2025), "Progress in Artificial Intelligence and its Determinants"** [douglas2025progress]. arXiv 2501.17894 (econ.GN). Patents and publications double about every 10 years versus about 2 years for compute (a 5:1 ratio); builds an aggregate SOTA (ASOTA) index; argues researcher inputs matter. **Knowledge production function.**

---

## 3. Comparison table: estimates of algorithmic progress (TFP growth)

| Source | Domain / output | Measure | Rate | Identification device |
|---|---|---|---|---|
| Hernandez & Brown 2020 | ImageNet, fixed AlexNet level | min compute to reach target | 44× over 2012–19; doubling 16 months | fixed output, min over entrants |
| Erdil & Besiroglu 2022 | ImageNet top-1 | compute-augmenting | doubling 8.95 months [3.55, 25.4] (abstract: 9 [4, 25]) | factor-augmenting year terms; MAP |
| Ho et al. 2024 | LM perplexity | effective compute | doubling 8.4 months [4.5, 14.3]; about 2.7×/yr (my calc); 22-billion-fold effective compute since 2014, under 2/3 from physical compute | factor-augmenting year terms; no FE |
| Whitfill 2025 | critique of Ho | — | possibly overstated by up to about 9× (sign depends on Cov(ln D, ε_D)) | theory |
| Mertens et al. 2026 | MMLU-Pro logit | period FE → compute factor | 7.5× from early 2023 to late 2024 (about 3–4×/yr, my calc); threshold compute ↓50× (majors) to 8,000× (all) | developer and period FE, OLS |
| Ho et al. 2025 (Rosetta) | IRT capability over 38 benchmarks | frontier b trend / within-family k | about 6×/yr (1–50×); frontier-b about 9×/yr | within-family slope, frontier intercept |
| Gundlach et al. 2025 (Origin) | LM loss, ablations and scaling experiments | CEG | under 100× total at small scale; 6,930× at the 2025 frontier (about 2.23×/yr) | experiments; scale-dependent CEG |
| Gundlach et al. 2025 (Price) | $ per benchmark level | price-adjusted | about 3×/yr algorithmic (after hardware and competition adjustment) | open models = competitive fringe |
| Xiao et al. 2025 (Densing) | 5 benchmarks via loss | effective params / actual params | ρ_max doubles in 3.3 months (arXiv) or 3.5 months (NMI) | frontier max, reference D = 1T |
| Dominguez-Olmedo et al. 2025 | MMLU, GSM8K | post-Nov-2023 dummy | 7 / 19 points before adjustment → insignificant after; frontier area ÷6 | equalizing treatment |
| Davidson et al. 2023 | various | post-training CEG | 5–30×, some more than 20× | non-experimental |
| Epoch summary (gradient update) | various | — | Ho 2024: 3×/yr [1.5, 64]; Ho 2025: 6×/yr [2, 50]; Whitfill–Snodin–Becker 2025: 3×/yr [2, 40]; Scher 2025: about 16–20×/yr | — |
| Kwa et al. 2025 (output, not TFP) | 50% time horizon | log h on date | doubling 207 days [166, 240] | OLS on frontier |
| Sevilla & Roldán 2024 (input growth) | training compute | — | 4–5×/yr (frontier 5.3×); LMs 9.5×/yr | log-linear |
| Cottier et al. 2024 (cost growth) | $ amortized | — | 2.4×/yr [2.0, 2.9] | Epoch frontier-41 |

---

## 4. ML ↔ IO dictionary for this strand (strength ratings)

| ML concept | IO concept | Strength | Note |
|---|---|---|---|
| Family/developer intercept (Ruan ν_f; Sloth α_ik; Mertens ν_j; Rosetta b) | Firm fixed effect / Hicks-neutral TFP level (Schmidt–Sickles panel frontier) | close | Sloth says so explicitly ("efficiency"). But a family is closer to a "product line or vintage" than a firm; developers own several families. |
| Compute-equivalent factor M = 10^{ν/β} (Mertens), f-equivalent FLOPs (Ruan), CEG (Davidson; Gundlach) | TFP expressed in input units (TFP^{1/RTS}); input-distance-function / equivalent-input measure | exact | For log-linear specs this is an identity. For scale-dependent CEG (Gundlach) it is a function, not a scalar. |
| Time trend in N_eff, D_eff (Ho) | Factor-augmenting technical change (Acemoglu; Doraszelski–Jaumandreu) | exact (as specification) | Exponential factor-augmenting trends inside a CES-like aggregator. |
| α_year and β_year jointly unidentified (Ho) | Diamond–McFadden–Rodriguez non-identification of bias vs substitution | close | Ho report the negative co-movement themselves. Identification rests entirely on the imposed Chinchilla exponents. |
| Algorithmic progress (effective-compute doubling) | TFP growth (Solow residual in input units) | close | But the output measure is bounded or benchmark-specific, and the reference algorithm matters. |
| Scale-dependent CEG; "reference dependence" (Gundlach) | Non-neutral (scale-biased) technical change; index-number problem (Laspeyres/Paasche; Caves–Christensen–Diewert) | close | Constant CEG ⇔ same exponents and E (my derivation). |
| Logit(score) = a + b log C | Cobb–Douglas in the odds of success; S-curve / diffusion-type output | exact (algebraic) | The odds ratio is the "output"; b is an output elasticity on the odds scale. |
| Sloth translog in (log s, log t) with interaction | Translog production function (Christensen–Jorgenson–Lau), SFA form (Kumbhakar–Lovell) | exact | Sloth itself makes this mapping. |
| Sloth multi-skill model (d=3) | Multi-output / multi-product production function | close | Skills are latent outputs sharing inputs; allocation of inputs across outputs is not modeled. |
| PCA / IRT capability index (Ruan, Rosetta, ECI) | Latent output from multiple noisy indicators; quality-adjusted output index; hedonic index | close | Anchoring = index base normalization; repeated measures allow errors-in-variables fixes. |
| Benchmark accuracy with floor and ceiling | Bounded, nonlinear transform of latent output; TFPR vs TFPQ | close | Ceiling compression causes spurious convergence (Fogelson et al.); floors cause spurious emergence (Schaeffer). |
| Contamination / training on the test task | Output mismeasurement correlated with the firm; "demand shock" in TFPR | close | Dominguez-Olmedo's equal fine-tuning works like a deflator. |
| Compute-only regressor (C = 6ND) | Aggregation restriction: equal unit elasticities, valid only on the expansion path | exact | Violated by over-trained models; ACF-type functional dependence on the Chinchilla path. |
| Within-family designs (fixed D, or D = 20N in Cerebras-GPT) | Within estimator with no within-firm variation in one input; perfect collinearity | exact | The D-elasticity cannot be identified with family FE if D is constant within families. |
| Model ladders / IsoFLOP / DataDecide | Designed experiments; exogenous input variation | close | Ends simultaneity but has external-validity limits (small scale, one lab's recipe). |
| Labs size runs after small-scale experiments reveal recipe quality (Whitfill) | Marschak–Andrews transmission bias; OP/LP timing assumptions | exact | Whitfill's Theorem 1 is the omitted-variable sign rule. |
| Epoch notability (SOTA criterion); released-only models; leaderboard self-submission | Selection on outcomes; survivorship / exit selection (OP) | close | Unobserved failed runs are the analog of exit. |
| Frontier statistics (ρ_max, min compute to threshold, highest-b models) | Frontier estimation (DEA/SFA); extreme-value statistics sensitive to the number of firms | close | They rise with entry even when the distribution does not shift. |
| Epoch compute confidence (±3×, ±10×, ±31× at 90%) | Classical input measurement error with known variance; reliability ratio | close | Errors may be non-classical (proprietary models guessed from rumors). |
| Distillation / synthetic data / pruning (Phi, Minitron, Gemma-2 distilled) | Intermediate inputs; gross output vs value added (Gandhi–Navarro–Rivers) | close (hypothesis for the Mertens outliers) | Teacher compute is missing from 6ND, so TFP is overstated. |
| Post-training enhancements (CEG 5–30×) and instruction tuning | Complementary intangible inputs / organizational capital | loose | Not in the input vector; shifts specific outputs (skills). |
| Capacity density (Densing law) | Single-factor (partial) productivity: output per parameter | exact | Over-training (data deepening) raises it with no TFP change. |
| Inference price declines (Cottier 2025; Gundlach Price of Progress) | Price = MC × markup; quality-adjusted price index; hedonic deflator | close | Gundlach uses open models to strip markups. |
| Training cost data (Cottier 2024) | Cost-function data (Nerlove) with factor prices (chips, energy, staff) | close | Staff is 29–49% of development cost and not in compute. |
| Data stock limit (Villalobos) | Quasi-fixed factor / resource constraint; rising shadow price | loose | Motivates over-training as input substitution. |
| Time horizon (METR) | Cardinal output measured in human labor time (labor-equivalent) | loose | Ties output to labor substitution, which is closer to task-based models (Acemoglu–Autor). |

---

## 5. Identification problems in this strand: IO diagnosis and proposed remedies

1. **Simultaneity (Marschak–Andrews).**
   - Cross-lab observational regressions (Ho, Mertens, Rosetta frontier, Owen) treat log C as exogenous.
   - Whitfill's timing story (small-scale experiments reveal ε, then compute is chosen) is exactly OP/ACF timing.
   - Remedies:
     - (a) Firm/family FE (Mertens, Sloth) handle time-invariant recipe quality only.
     - (b) Dynamic panel over **family generations** (Llama → 2 → 3 → 3.1; Qwen → 1.5 → 2 → 2.5; Gemma → 2 → 3; Yi → 1.5; DeepSeek LLM → V2 → V3), with ω_{f,g} = ρω_{f,g−1} + ξ and lagged-generation inputs as instruments (Blundell–Bond).
     - (c) Proxy / control function: an input chosen after ω is observed and monotone in ω, e.g. D given N (tokens are extended when the recipe works), post-training compute, or the decision to release larger sizes.
     - (d) Instruments: hardware-supply shocks (export controls, GPU availability by lab or country), cloud price changes, chip generation at training time.
2. **Functional dependence and collinearity.**
   - On the compute-optimal path (Cerebras-GPT: D = 20N), log N and log D are collinear.
   - Within families, D is often constant, so the N and D elasticities are identified from different variation.
   - With family FE, the D-elasticity is identified only from the few families with within-family D variation.
   - Report "first-stage" diagnostics: within-family SD of log(D/N), condition numbers.
3. **Aggregation into compute.**
   - C = 6ND forces equal elasticities. Test the restriction β_N = β_D in logit(score) = a + β_N log N + β_D log D + FE.
   - Over-trained small models (Llama-3-8B at 15T, Qwen 0.5B at multi-trillion tokens) are where it fails.
   - Bhagia shows FLOPs fit worse; Sloth includes the interaction term.
4. **Input measurement error.**
   - Epoch confidence classes give heteroskedastic error variances. Use them for reliability-weighted corrections (Fuller) or SIMEX.
   - Two compute measures (6ND vs hardware-time) serve as repeated measurements, one instrumenting the other.
   - Fixed effects amplify attenuation (Griliches–Hausman). Ho's β_data = 0.04 vs 0.37 experimental fits this pattern.
5. **Output measurement.**
   - Bounded outputs with floors and ceilings; contamination (GSM1k); training on the test task (period effects biased upward); prompt format (up to 76 points); seed noise.
   - Remedies:
     - IRT latent output with guessing floors (Sloth, Rosetta).
     - Continuous metrics: loss or bits-per-byte (Schaeffer; Heineman).
     - Unbounded outputs (time horizon).
     - "Deflate" by equalized fine-tuning (Dominguez-Olmedo).
     - Contamination-robust twins (GSM1k).
6. **Selection.**
   - Only released or notable models are observed. Epoch notability includes SOTA performance, which is selection on Y. Leaderboards are self-submitted. Labs choose which benchmarks to report (missing not at random).
   - Remedies:
     - Compare the full Epoch database with the Notable subset.
     - Lee/Manski bounds.
     - A release-propensity model using compute and developer size.
     - Treat benchmark reporting as selection (a Heckman-type model on the model × benchmark matrix).
7. **Time effects and factor bias.**
   - Ho's factor-augmenting split is not identified (Diamond–McFadden–Rodriguez).
   - Identify factor bias from within-year variation in D/N, i.e. over-training choices across labs, combined with a flexible (translog) technology and time×input interactions.
   - Mertens' period×compute interactions are insignificant, so Hicks-neutral time effects are not rejected for MMLU-Pro.
8. **Frontier vs mean.**
   - ρ_max (densing), min-compute-to-threshold (Mertens, Hernandez & Brown) and frontier-b (Rosetta) are order statistics.
   - Use panel stochastic frontier or quantile regression, and adjust for the number of entrants per period.
9. **Heterogeneous technologies.**
   - Ruan allows family-specific slopes; Sloth shares slopes; Gundlach finds exponent changes across architectures.
   - Test slope homogeneity (Mairesse–Griliches), and consider architecture-specific technologies (dense vs MoE vs SSM/RWKV), since RWKV and Jamba are in the ObsScaling data.

---

## 6. Contribution ideas (for our paper)

1. **An IO re-estimation of the Mertens/Ruan cross-lab production function.** Compare:
   - pooled OLS;
   - family FE;
   - developer FE;
   - Blundell–Bond GMM over family generations;
   - an ACF-style control function using D-given-N or post-training as the proxy.
   Report how the output elasticity, the implied returns to scale and the algorithmic-progress rate move. The prior from IO: OLS elasticities are biased up under simultaneity; FE estimates are biased down under measurement error.
2. **Identification audit of the public data.** Tabulate within-family variation in log N, log D and log(D/N) in the ObsScaling and Sloth CSVs. Show which elasticities are identified under which fixed effects. Cerebras-GPT (D = 20N) and Llama-3.1 (D = 15T for all sizes) are clean motivating examples, and they explain Rosetta's "suspicious" k = 0.12 for Llama 3.1.
3. **Test the C = 6ND aggregation restriction** (β_N = β_D on the logit-odds scale), and measure the misspecification bias in compute-only regressions from over-training.
4. **Errors-in-variables correction** using Epoch confidence classes (90% CI ±3×, ±10×, ±31×) and dual compute measures. Quantify how much of the gap between Ho's β_data = 0.04 and the experimental 0.37 is attenuation and how much is simultaneity.
5. **Intermediate inputs.** Add teacher compute to the input vector for distilled, pruned or synthetic-data models (Phi, Minitron/Nemotron, Gemma-2 distilled, DeepSeek-R1-distill). Test whether the Microsoft and Nvidia "secret sauce" (about 60×) shrinks. This is the gross-output vs value-added correction.
6. **Output-measurement robustness.** Re-estimate developer and period effects with:
   - (a) the IRT latent index (ECI);
   - (b) Dominguez-Olmedo-style "deflated" scores;
   - (c) a contamination-robust benchmark (GSM1k vs GSM8k);
   - (d) an unbounded output (METR time horizon).
   Show how much of the measured "algorithmic progress" is benchmark-specific, i.e. TFPR rather than TFPQ.
7. **Scale-biased technical change and the index-number problem.** Formalize Gundlach's CEG function. Constant CEG holds iff the technologies share exponents and E. Propose a Törnqvist/translog-based algorithmic-progress index, evaluated at the mean log compute of the two periods, as a superlative index that removes reference dependence to second order.
8. **Selection-corrected productivity dispersion.** Recompute the within-developer p90/p10 (41×) with bounds for unobserved failed or unreleased runs. Compare with Syverson-style manufacturing dispersion.
9. **Frontier vs average technical change.** Contrast Rosetta's frontier-b (about 9×/yr), the densing ρ_max and Mertens' threshold minima with mean-shift estimates, controlling for the number of entrants (extreme-value correction).
10. **Cost-function dual.** Use Cottier's amortized costs and hardware price series to estimate a Nerlove cost function for frontier runs. Compare returns to scale with the primal elasticities.

---

## 7. Data leads (see the structured output for URLs)

- ObsScaling CSV: 211 rows; family, N, D, FLOPs, 8 benchmarks; Apache-2.0.
- Sloth data folder: Open LLM Leaderboard v1/v2 compiled, plus training tokens; MIT.
- Mertens et al. dataset: 809 models, released on Google Drive per the paper.
- Epoch AI Models database: about 3,620 models, 1,895 language models since 2018. Includes compute confidence levels and the notability flag; CC BY.
- Epoch ECI and Benchmarking Hub CSVs: CC BY 4.0.
- Epoch lm-algorithmic-progress: Ho et al. data, about 231 models; GitHub plus a Google Sheet.
- Open LLM Leaderboard results dataset: HF `open-llm-leaderboard/results`; v2 benchmarks IFEval, BBH, MATH Lvl 5, GPQA, MuSR, MMLU-Pro.
- Signal & Noise (Heineman et al.) and DataDecide: controlled "recipe × scale" experiments.
- OLMo ladder (Bhagia et al.): designed 4×4 N × D/N grid.
- METR time-horizon data.
- Epoch inference-price data and Artificial Analysis prices (used by Gundlach et al.).
- OpenAI ai-and-efficiency repo: Hernandez & Brown tables.

---

## 8. Warnings, open issues and conflicting claims

- **Venues not fully confirmed:**
  - Ho et al. 2024 at NeurIPS: not confirmed; cited as arXiv.
  - Schaeffer et al. 2023 at NeurIPS 2023: not confirmed.
  - Sloth: venue unknown (OpenReview record exists).
  - Oren et al.: arXiv only confirmed.
  - Sainz et al.: Findings of EMNLP, year conflict (2023 vs 2024).
  - Biderman et al. (Pythia) at ICML 2023: not confirmed.
- **Densing law** doubling is 3.3 months in the arXiv version and 3.5 months in the NMI version. Cite the NMI version and note the change.
- **Erdil & Besiroglu:** the text says 90% intervals because the bootstrap is multimodal, but the abstract reports a 95% CI of 4–25 months. The Table 1 point ratio implies about 8.1 months against the reported 8.95.
- **Ho et al. headline:** the abstract says 8 months [5, 14]; the text says 8.4 [4.5, 14.3]; Epoch's later summary says 3×/yr [1.5, 64]. "22-billion-fold since 2014" (total effective compute) is different from Gundlach's "22,000× over 2012–2023" (algorithmic part). Do not mix them up.
- **Mertens et al.:** OpenAI, Anthropic and xAI effects are imprecise (SE about 0.66–0.75). Proprietary compute comes from lifearchitect.ai. The 7.5× is over about 1.5–2 years (annualizing is my calculation). The paper has 9 references and does not engage with the productivity literature.
- **Cottier et al.:** the fetched shares for energy disagree (abstract 2–6% vs a fetched "9% average"). Use the abstract numbers.
- **Sloth compute-optimal table:** the 15T token cap reflects the support restriction in their optimization, not a data-scarcity model (my reading).
- **ObsScaling within-family D table** was read by a summarizing fetch, not computed. Recompute from the CSV before citing. Some entries look off: BTLM shows "627T" (probably 0.627T); the DeepSeek-Coder classification is inconsistent.
- **Hypotheses flagged in the text** (not from the sources):
  - Microsoft and Nvidia effects reflect distillation or synthetic data.
  - Selective benchmark reporting biases Rosetta/ECI.
  - Constant CEG ⇔ equal exponents and E (a derivation, but it should be checked formally in the paper).
- **Web search** ran out during this strand. Some secondary items were verified only via arXiv abstract pages: Wei et al. (TMLR, confirmed), Kwa et al. (NeurIPS 2025, confirmed on arXiv), and others.
- **Not included in this bib** (covered by the core-scaling strand): Kaplan et al. 2020 (arXiv 2001.08361), Hoffmann et al. 2022 (arXiv 2203.15556), Muennighoff et al. 2023, Sardana & Frankle 2023. Kumbhakar & Lovell (the SFA text Sloth cites) belongs to the IO strand.
