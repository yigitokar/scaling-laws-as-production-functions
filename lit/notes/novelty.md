# Adversarial prior-work / novelty search

Project: "Scaling Laws as Production Functions". This strand asks one question: what already exists that does what we plan to do?
Search date: 2026-09-23. Compiled for the paper-writing stage. BibTeX: `lit/bib/novelty.bib`.

---

## 0. How the search was run, and its limits

- **Sources queried.** WebSearch (about 40 queries before the session-wide web-search budget ran out). arXiv full-text and abstract search through `arxiv.org/search` and the `export.arxiv.org` API: about 30 Boolean queries, including econ-category-restricted queries. OpenReview API search. Crossref, for bibliographic verification of the IO anchors. LessWrong and Alignment Forum through greaterwrong.com search. Epoch AI blog and data pages. NBER and AEA pages.
- **Unavailable.** The Semantic Scholar API returned HTTP 429 on every call, so it was never queried. DuckDuckGo served a CAPTCHA, which was not attempted. Bing returned only irrelevant results.
- **Full-text reads.** Every Tier-1 paper below was downloaded as a PDF and read or grepped in full. Equations and numbers come from those texts, not from abstracts. The exceptions are the Epoch blog posts, which were read through WebFetch summaries.
- **Coverage caveat.** Blogs, Substack posts and SSRN-only working papers are under-covered because the WebSearch budget ran out. Anything that appeared only there after about June 2026 may have been missed. Re-run a targeted Scholar/SSRN/Substack sweep before submission.

---

## 1. Verdict table, one row per planned contribution

| # | Planned contribution | Verdict | Closest prior work | What is still ours |
|---|---|---|---|---|
| C1 | **Dictionary** between the IO production-function literature and scaling laws | **PARTIALLY DONE (piecemeal). A systematic dictionary is NOVEL** | Hao & Merrill 2026 (Leontief, sigma, MRTS, homogeneity, Mussa-Rosen); Ho et al. 2024 (effective inputs, a "Hicks-neutral" model variant); Erdil & Besiroglu 2022 (compute- vs data-augmenting progress); Whitfill 2025 (selection bias); Mertens et al. 2026 (developer FE as "secret sauce"); Konig et al. 2026 (causal-validity framing); Erdil & Besiroglu 2025 (f(T,I) proportional to I·T^c) | No one maps the *econometric* IO toolkit onto scaling laws: Marschak-Andrews, OP/LP/ACF, GNR, DLW, DMR, Nerlove duality, SFA/Farrell, FHS TFPR/TFPQ. Each entry has at most a one-sentence precursor. |
| C2 | **CES / elasticity-of-substitution reading** of Chinchilla | **DONE (derivation) by Hao & Merrill 2026**. It was also stated informally in a 2022 LessWrong post. | Hao & Merrill 2026, App. B: sigma = (aα d^β + bβ n^α)/(aα(β+1)d^β + (α+1)bβ n^α); sigma = 1/(1+α) when α = β; sigma ≈ 0.762 for Chinchilla | (i) Estimate sigma with standard errors, across datasets and families. (ii) New result: on the compute-optimal path, **sigma\* = 2/(2+α+β)** exactly. This explains why Hao & Merrill's averaging of α and β "works". (iii) New result: the Kaplan (2020) law implies **sigma ∈ [0.50, 0.575]** versus Chinchilla's **0.74–0.76**. (iv) Test CES against non-CES forms (translog; Farseer's N–D coupling term). (v) sigma is ordinal and invariant to loss versus benchmark output; returns to scale are cardinal. (vi) Use sigma to discipline over-training. Hao & Merrill *replace* Chinchilla with Leontief (sigma = 0), which rules out over-training by construction. |
| C3 | **ACF-style functional dependence** on the compute-optimal frontier; IsoFLOPs as the experiments that break it | **PARTIALLY DONE (statistics side) by Kricheli et al. 2026**. Related estimator critiques: Besiroglu et al. 2024; Czech et al. 2026. | Kricheli et al. 2026 prove that D = kN designs make Chinchilla/Kaplan/Muennighoff/Droppo-Elibol ill-conditioned, with condition number Θ(\|α−β\|^-2). They also give a closed-form threshold for tokens-per-parameter diversity. | Their collinearity is **design-induced**. Ours is **behaviour-induced**: optimizing labs lie on N\*(C), D\*(C), as in ACF. Also ours: (i) the sources of identifying deviation in observational data (inference demand, data scarcity, hardware and memory limits), the analogue of ACF's optimization errors or input-price shocks; (ii) the reading of Chinchilla Approaches 1/2/3 as cost-function / conditional-factor-demand / primal estimation, where Approach 2 identifies only a = β/(α+β); (iii) the GNR first-order-condition route. |
| C4 | **Simultaneity / selection** in observational scaling (Marschak-Andrews) | **PARTIALLY DONE**: Whitfill 2025 (formal omitted-variable/selection theorem for Ho et al. 2024, plus Monte Carlo); Konig et al. 2026 (confounding and effect heterogeneity in the "observational approach"); Dominguez-Olmedo et al. 2024 (training on the test task confounds). | Whitfill proposes only RCTs or unspecified IVs. Konig et al. frame the problem in Rubin/causal-validity terms. | Not done by anyone: the explicit Marschak-Andrews / Olley-Pakes structure (timing, compute chosen after small-scale ablations reveal ω); **proxy-variable or control-function**, dynamic-panel (Blundell-Bond) and timing-based identification; signing the bias in the Mertens and Ruan fixed-effects regressions; bounds. |
| C5 | **Lab TFP dispersion** | **PARTIALLY DONE by Mertens, Fischl-Lanzoni & Thompson 2026** (809 LLMs, developer and period FE, OLS). Ruan et al. 2024 (family-specific compute efficiency). | Mertens et al.: developer factors from 2.3x (DeepSeek) to 60.5x (Microsoft) relative to small developers; within-developer 90/10 of 41x in effective compute. They cite no IO work and do not correct for endogeneity. | IO-grade TFP: correct for simultaneity and selection (released or evaluated models only); treat TFPQ versus TFPR-like output measurement (logit of a saturating benchmark; contamination); treat **intermediate inputs** (distillation or synthetic data from teachers, e.g. Phi, which may explain Microsoft's 60.5x) through a GNR gross-output vs value-added logic; compare the magnitude with manufacturing dispersion. |
| C6 | **Algorithmic progress as factor-augmenting TFP growth**; Hicks-neutral vs biased; DMR | **PARTIALLY DONE (substance)**: Erdil & Besiroglu 2022 (compute- vs data-augmenting); Ho et al. 2024 (N_eff, D_eff with exponential year terms; Model 12 is labelled "Hicks-neutral"); Gundlach et al. 2025 ("scale-dependent" progress, i.e. exponent change); Sanderson et al. 2025 (compute-dependent vs compute-independent progress). | None of these cites the economics of biased technical change (Acemoglu, DMR 1978, Klump-McAdam-Willman, Doraszelski-Jaumandreu 2018). | Still ours: (i) the DMR non-identification point, since Ho et al.'s α_year and β_year are individually insignificant; (ii) the reclassification of **Kaplan→Chinchilla re-balancing as allocative-efficiency gain (Farrell 1957), not technical change**, which Gundlach et al. and Ho et al. count as "algorithmic progress"; (iii) scale-dependence as non-neutral change in output elasticities; (iv) selection-corrected growth rates. |
| C7 | **DLW-style inference wedge / revealed inference demand** from over-training | **NOVEL (none found)** | The forward problem is done. Given inference demand, the optimal (N, D) is in Sardana et al. 2024, Bian et al. 2025 (latency), Roberts et al. 2026 (test-time sampling), de Vries 2023 (compute overhead). Hao & Merrill 2026 include inference cost 2nt but assume Leontief, so no wedge. | The **inversion**: T = 3D(ε_N/ε_D − 1). The analogy with a DLW markup, and the Raval (2023) and Bond et al. critiques, carry over (see §5.3). |
| C8 | **IO estimators applied to scaling data** | **PARTIALLY DONE (simple tools only)**: OLS with FE (Mertens et al. 2026); first-order-condition-based CES with firm FE and an IV on a lab panel of *R&D* inputs, not the scaling law (Whitfill & Wu 2025); quantile-regression frontier (Zhang, Jin, Syrgkanis & Kakade 2026); DEA/Malmquist on *inference prices* (Du 2026) | — | Not done: OP/LP/ACF, GNR, dynamic panel, SFA, DLW, Nerlove-type cost-function estimation on training-scaling data. |
| C9 | **Train vs test-time compute as an isoquant** | **PARTIALLY DONE**: Jones 2021; Villalobos & Atkinson 2023 (Epoch); Erdil 2024 (optimal training share α/(α+β), a Cobb-Douglas-type expenditure-share rule); Erdil & Besiroglu 2025 (f(T,I) proportional to I·T^c, increasing returns); Snell et al. 2024; Roberts et al. 2026 | Epoch already uses production-function language here. | Estimate sigma between train-time and test-time compute; tie it to the inference wedge (C7). |
| C10 | MoE capital stock vs services; data repetition as depreciation; distillation as intermediate input; SFA / Farrell technical inefficiency | Mixed. **Depreciation: the functional form is already exact** in Muennighoff et al. 2023, D′ = Σ_k (1−δ)^k U, a perpetual-inventory sum, although the paper never uses the word "depreciation". **SFA: PARTIAL** (Lourie et al. 2026, where hyperparameters are the confounder and scaling laws hold only on the tuned frontier; Zhang et al. 2026 quantile frontiers). MoE and distillation analogies: **NOVEL** as economics mappings, with ML precursors in Clark et al. 2022, Krajewski et al. 2024 and Busbridge et al. 2025. | — | The naming and econometric consequences are ours: vintage/efficiency-unit capital, the gross-output problem, inefficiency decomposition. |

### Recommended positioning statement (draft)

> "A small but fast-growing literature has begun to read neural scaling laws through an economic lens. Hao and Merrill (2026) note that the Chinchilla loss surface implies an elasticity of substitution between parameters and tokens of about 0.76 and study profit-maximizing training under a Leontief approximation. Mertens, Fischl-Lanzoni and Thompson (2026) document developer-specific 'compute efficiency' with fixed-effects regressions. Ho et al. (2024) and Erdil and Besiroglu (2022) measure parameter-, data- and compute-augmenting algorithmic progress. Whitfill (2025) and Konig et al. (2026) warn that observational estimates may be confounded, and Kricheli et al. (2026) show that fixed tokens-per-parameter designs leave scaling-law parameters practically unidentified. What is missing is the recognition that each of these observations is an instance of a *known* problem in empirical IO, together with that literature's solutions: transmission bias (Marschak and Andrews 1944; Olley and Pakes 1996), functional dependence (Ackerberg, Caves and Frazer 2015), gross-output identification (Gandhi, Navarro and Rivers 2020), the production approach to wedges (De Loecker and Warzynski 2012), non-identification of biased technical change (Diamond, McFadden and Rodriguez 1978), and cost-function duality (Nerlove 1963). This paper supplies that mapping, derives its implications for scaling-law estimation, and applies IO estimators to public scaling data."

**Claims to avoid.**
- "First to interpret scaling laws as production functions." This is false: Hao & Merrill 2026; Erdil & Besiroglu 2025 (blog); the 2022 LW post.
- "First to compute the elasticity of substitution." This is false: Hao & Merrill 2026.
- "First to note endogeneity or selection in observational scaling." This is false: Whitfill 2025; Konig et al. 2026.
- "First to measure developer heterogeneity." This is false: Mertens et al. 2026; Ruan et al. 2024.
- "First to model factor-augmenting algorithmic progress." This is false: Erdil & Besiroglu 2022; Ho et al. 2024.
- "First to show collinearity kills identification." This is false: Kricheli et al. 2026.

**Claims that appear defensible** (none found in searches):
- The systematic IO-econometrics dictionary.
- The ACF behavioural functional-dependence reading, including Approaches 1/2/3 as cost-function vs factor-demand vs primal estimation.
- The DLW/GNR duality and revealed inference demand.
- Control-function or proxy estimation on scaling data.
- The DMR point.
- Allocative vs technical decomposition of "algorithmic progress".
- sigma\* = 2/(2+α+β); the Kaplan vs Chinchilla elasticity contrast.
- The gross-output / distillation problem in lab TFP.

---

## 2. Tier-1 overlaps (must cite; position explicitly)

### 2.1 Hao & Merrill (2026), "A Theory of Training Profit-Optimal LLMs", arXiv:2605.16430 (v1 14 May 2026; v3 11 Jun 2026). Boston U. / AI2.
- **What it does.** Microeconomic model of a monopoly LLM firm choosing (n, d, t).
  - Profit: π = ω t f(q(n,d)) − δ t² − (6nd + 2nt)/E. Here E is hardware efficiency in FLOP/$. Data are free. Inverse demand comes from a consumer quality-threshold distribution proportional to 1 − q^(−γ), with γ < −1 (Mussa-Rosen/Spence style).
  - **Technology:** a "Leontief scaling law" q(n,d) = min{a n^α, b d^β}. They call a "parameter efficiency" and b "data efficiency".
  - **App. B** derives the elasticity of substitution of Chinchilla with q = 1/(ℓ − ℓ\*):
    - σ = (aα d^β + bβ n^α) / (aα(β+1) d^β + (α+1) bβ n^α).
    - This reduces to σ = 1/(1+α) when α = β.
    - Plugging in α = 0.3392, β = 0.2849 (averaged to 0.31205) gives **σ ≈ 0.7622**. So n and d are complements.
  - They state that with α ≈ β "LLM scaling is homogeneous of degree α".
  - Results:
    - Compute-bound regime: n\* and d\* track E near-linearly, and training cost is sub-quadratic in E.
    - Data-bound regime: expenditure is proportional to D²/E.
    - Current spending trends are consistent only with permissive variants.
  - Cites Leontief 1941, Mussa-Rosen 1978, Spence 1975, Acemoglu 2025, Bergemann-Bonatti-Smolin 2026.
- **Overlap.**
  - C2: high. They derive σ.
  - C1: medium.
  - C7: low. Their Leontief assumption makes profit-optimal points satisfy a n^α = b d^β (Lemma 2), so **over-training is ruled out by construction**.
  - C3–C6, C8: none. There is no estimation or econometrics.
- **Use in our paper.** Cite as the first formal σ derivation. Extend it:
  - σ\* = 2/(2+α+β) on the expansion path. Note that this equals their averaged-α value exactly.
  - Standard errors for σ.
  - Show that σ > 0 is what makes observed over-training rational, which their Leontief approximation discards.
  - Footnote 3 of their paper reports α, β for three OLMo-Hybrid families: (0.25, 0.21), (0.23, 0.22), (0.18, 0.23). Via σ\* = 2/(2+α+β) these imply σ\* ≈ 0.82–0.83. This is a heterogeneity data point, secondary source: Merrill et al. 2026, arXiv:2604.03444.

### 2.2 Mertens, Fischl-Lanzoni & Thompson (2026), "Is there 'Secret Sauce' in Large Language Model Development?", arXiv:2602.07238 (v2 3 May 2026). MIT FutureTech. Mertens is a productivity/IO economist.
- **Data.**
  - 809 LLMs, Oct 2022–Mar 2025.
  - HF Open LLM Leaderboard, plus Epoch AI and TIGER-Lab for proprietary models.
  - Hand-collected tokens. Proprietary N and D come from lifearchitect.ai, and they flag this as uncertain.
  - Reasoning models are excluded.
  - The dataset is said to be published; the link was not recoverable from the PDF text.
- **Specification.**
  - logit(MMLU-Pro)_i = β0 + β_c log(c_i) + δ_t (3 period FE) + ν_j (developer FE) + ε_i, with c = 6ND.
  - Effective-compute factor M = 10^(δ/β_c).
- **Results.**
  - β_c = 0.79 (SE 0.19) log-odds per 10x compute.
  - Shared progress: 7.5x effective compute between 2022q4–2023q3 and 2024q3–2025q1.
  - Developer factors relative to small others range from DeepSeek 2.3x to **Microsoft 60.5x**.
  - Within-developer model residuals: **90/10 ratio of 41x** in effective compute.
  - Shapley R² decomposition:
    - Compute: 32–45%.
    - Period: 3–10%.
    - Developer: 14–34%.
    - Residual: 32–47%.
  - At the frontier, 80–90% of performance growth is explained by compute.
  - There is weak evidence that developers of smaller models are more efficient (not significant).
- **What they do NOT do.**
  - No IO citations: no OP/LP/ACF, Syverson or Hsieh-Klenow.
  - No endogeneity or selection correction.
  - No N/D separation, since compute is the only input.
  - Output is a logit of a saturating benchmark.
  - No treatment of distillation or synthetic data as intermediate inputs.
- **Overlap.** C5: high. C8: medium, but only OLS with FE. C4: low; they discuss time-effect robustness only.
- **Our angle.**
  - The FE estimate of β_c is biased upward if ν or ε correlates with c (transmission bias). The sign could go either way; cf. Whitfill.
  - Selection on releasing models.
  - Microsoft's 60.5x likely reflects Phi-style distillation or synthetic data. That is value-added TFP contaminated by purchased intermediate inputs (the GNR gross-output problem).
  - The "smaller models more efficient" pattern is predicted by the inference wedge (C7).
  - **Magnitude benchmark (ours, illustrative).** Converting the 41x 90/10 effective-compute ratio into reducible-loss units with the Chinchilla compute elasticity γ = αβ/(α+β) gives 41^0.155 ≈ 1.78x (Hoffmann) to 41^0.178 ≈ 1.94x (Besiroglu). This is the same order as within-industry TFP 90/10 ratios in US manufacturing (~1.9; Syverson; *number not re-verified this session*). The conversion assumes that loss-based scaling applies to MMLU-Pro, which is heroic. Present it as suggestive only.

### 2.3 Whitfill (2025), "Note on Selection Bias in Observational Estimates of Algorithmic Progress", arXiv:2508.11033 [econ.GN] (v2 18 Aug 2025). MIT.
- **Setup.**
  - Starts from Ho et al. (2024), L = E + A/(N q_N)^α + B/(D q_D)^β with q_N = exp(α′(Y−Y0)) and q_D = exp(β′(Y−Y0)).
  - Adds within-year lab heterogeneity: q_D = exp(β′(Y−Y0) + ε_D), with Cov(Y, ε) = 0.
  - Under N → ∞ and E ≈ 0: ln L = ln B − β ln D − β_year (Y−Y0) − β ε_D, an omitted-variable problem.
- **Theorem 1.** Assume Cov(ln D, Y) > 0, β > 0, β_year > 0, and Cov(ln D, ε_D) > −Var(residualized ln D). Then **sign(bias(β̂_year/β̂)) = −sign(Cov(ln D, ε_D))**. The closed-form plims are derived.
  - Arguments are given for both signs. Positive: data pipelines co-move with know-how. Negative: labs that are ahead need less data.
- **Evidence.**
  - Experimental β ≈ 0.37 (Hoffmann; Besiroglu) versus Ho et al.'s β ≈ 0.04. A "first-pass" reading is that progress is overstated about 9x.
  - Monte Carlo on Ho's actual (D, Y), with β = 0.37 and β′ = 0.45:
    - corr = 0.5 gives 16.5%/yr.
    - corr = −0.5 gives 93%/yr.
- **Remedies suggested.** RCTs, or "some plausible instrument". No control function, no proxy, no timing assumptions.
- **Overlap.** C4: high (conceptual plus formal). C6: medium. It does **not** cite Marschak-Andrews, Olley-Pakes or any IO work.
- **Our angle.** Recast as Marschak-Andrews transmission bias. Add IO remedies. Note also the functional-form channel: Ho et al.'s β ≈ 0.04 versus 0.37 may reflect the attenuation expected with measurement error in D (epochs, tokenization) and collinearity with Y, not only selection.

### 2.4 Kricheli, Reid, Sarkar, Gandikota & Shakarian (2026), "Tokens-per-Parameter Coverage Is Critical for Robust LLM Scaling Law Extrapolation", arXiv:2605.08541 (v2 12 May 2026). Syracuse / Amazon AGI.
- **Result.**
  - With designs on a single ray D = kN, the Gauss-Newton (J^T J) condition number grows as **Θ(ε^−2), with ε = |α−β|**.
  - For |α−β| ≈ 0.06 the condition number is about 10²–10³, and the scale coefficients (A, B) become "practically unidentifiable": confidence intervals inflate 10x or more, making it a "sloppy" model.
  - Proved for Chinchilla, Muennighoff (repeated data), Kaplan and Droppo-Elibol.
  - Closed-form necessary and sufficient TPP-diversity threshold.
  - Empirically, non-collinear designs win 97.3% of held-out comparisons.
  - The degeneracy is geometric: any smooth objective inherits it.
  - D-optimal allocation discussed in an appendix.
  - Data: HF dataset `TPPIsCriticalFor/colinear_scaling_models`.
- **Relatives cited there.**
  - Volkova et al. 2026 (optimizer-axis ill-conditioning; arXiv:2602.07712).
  - Schaeffer et al. 2026 ("compute-envelope" reparameterizations; arXiv:2509.24012).
  - Farseer (Li et al. 2025), about 1,000 models on a 2-D grid with an N–D coupling term.
  - Bergsma et al. 2025 (hyperparameters obey power laws in D/N).
- **Overlap.** C3: high on statistics. They cite only generic multicollinearity references, not ACF, and treat collinearity as a design choice.
- **Our angle.**
  - In observational data the ray arises from *optimizing behaviour*: Chinchilla-rule labs, or labs over-training at a firm-specific ratio. This is ACF's functional-dependence problem.
  - Identification then needs *economically generated* deviations: heterogeneous inference demand, data scarcity (Muennighoff), memory and latency constraints, hardware lumpiness. These are ACF/GNR analogues.
  - IsoFLOP sweeps are experiments that move along the isocost. They recover the MRTS, not α and β separately, unless several compute levels are used.

### 2.5 Ho, Besiroglu, Erdil, Owen, Rahman, Guo, Atkinson, Thompson & Sevilla (2024), "Algorithmic progress in language models", arXiv:2403.05812; NeurIPS 37:58245–58283. Also Erdil & Besiroglu (2022), "Algorithmic progress in computer vision", arXiv:2212.05153.
- **Ho et al.**
  - N_eff = N·exp(α′(Y−Y0)) and D_eff = D·exp(β′(Y−Y0)) are plugged into Chinchilla (eq. 3).
  - Around 231 models on WT103, WT2 and PTB over 2012–2023.
  - About 90 specifications compared by leave-one-out CV. Model 7 is preferred, with benchmark-specific A and B and R² ≈ 0.91.
  - Table 2 (95% bootstrap CI, 100 iterations):
    - α_param = 0.068 [0.045, 0.127].
    - β_data = 0.040 [0.023, 0.062].
    - α_year = 0.004 [−0.058, 0.032] (n.s.).
    - β_year = 0.036 [−0.002, 0.080] (n.s.).
  - Compute to reach fixed performance halves every **~8 months (95% CI ~5–14)**.
  - Shapley shares: 60–95% from compute and data scaling, 5–40% from algorithms.
  - Transformer compute-equivalent gain: 7.2x [3.3x, 45.7x].
  - **Model 12 is explicitly labelled "Hicks-neutral"** (one efficiency term multiplying both loss terms).
  - Model 18 is compute-only. Model 20 corrects for epochs.
  - They acknowledge that "algorithmic improvements and scaling have historically been introduced concurrently", and discuss sources of input and output measurement error (tokenization, epochs, preprocessing) in Table 4. They do not formally address endogeneity.
  - Implied loss–compute exponent is about C^(−1/20), versus about −0.3 in Chinchilla. They attribute this to the dataset's compute range.
  - Data and code: github.com/epoch-research/lm-algorithmic-progress.
- **Erdil & Besiroglu (2022).**
  - Logit of ImageNet accuracy modelled as a product of sigmoids in effective compute C = α1 + α_Year·(Year−2012) + α_compute·log(compute) and effective data D (analogous).
  - Approximates 1 − P ≈ Ã/C^α + B̃/D^β.
  - MAP estimation with Normal priors, because "parameters about data scaling end up being poorly identified". That is an identification admission.
  - Around 124 models (extending Thompson et al. 2020), with **at most 3 top models per paper**, which is selection on the outcome.
  - Compute-augmenting progress halves requirements every 9 months [4, 25].
  - Progress is mostly compute-augmenting, not data-augmenting.
- **Overlap.** C6: high. Factor-augmenting progress and a Hicks-neutral variant exist; they cite Davidson 2023's economic model but not the biased-technical-change literature. C1: medium. C4: low.
- **Our angle.**
  - DMR: the bias and σ are identified only by the imposed separable power form, which fits the insignificant individual year terms.
  - Selection (Whitfill).
  - Top-3-per-paper truncation.
  - Allocative vs technical efficiency (see §5.5).

### 2.6 Konig, Pawelczyk, von Luxburg & Bordt (2026), "Validity Threats for Foundation Model Research", arXiv:2606.05029 (3 Jun 2026). Tubingen / Vienna.
- Casts foundation-model research as causal inference with four validity types: statistical, internal, external, construct.
- **The observational approach**, i.e. fitting scaling laws on public models (Ruan, Owen, Choshen, Ho, Mertens), "faces confounding and effect heterogeneity".
- Calendar time is named as a prominent confounder, together with test-task-tuned data mixes (Dominguez-Olmedo et al. 2024; Zhang et al. 2025).
- Exchangeability is untestable. Pooling across families trades precision against external validity.
- Proxy (small-scale) experiments trade external and construct validity for internal validity.
- Cites Whitfill 2025.
- **Overlap.** C4: high (conceptual). C1: low-medium. There are no IO or econometric production-function tools.
- **Our angle.** Supply the structural model: why input choices correlate with productivity, and which timing and proxy assumptions identify the model.

### 2.7 Whitfill & Wu (2025), "Will Compute Bottlenecks Prevent an Intelligence Explosion?", arXiv:2507.23181 [econ.GN]. MIT / Yale.
- CES *research* production F(K_res, L) = [γK^((σ−1)/σ) + (1−γ)L^((σ−1)/σ)]^(σ/(σ−1)).
- Identification from the FOC-ratio regression ln(K/H) = σ ln(w/r) + const, with firm FE.
- Panel of 4 labs (OpenAI, DeepMind, Anthropic, DeepSeek), 2014–2024, **N = 27 firm-years**.
- Estimates:
  - Baseline σ = 2.58 (SE 0.34): substitutes.
  - "Frontier experiments" specification with ln K_train as a control: σ = −0.10 (0.18), i.e. complements.
- Appendix IV: local exchange-rate-adjusted wages as instrument.
- They acknowledge endogeneity and simultaneity.
- **Overlap.** C8: medium. It is IO-style estimation on AI labs, but for the R&D (idea) production function, not the training scaling law. C1: low.
- **Use.** A precedent that econ.GN readers accept this genre. Their relative-price FOC method is the GNR/Oberfield-Raval-style route we can mirror for (N, D).

### 2.8 Gundlach, Fogelson, Lynch, Trisovic, Rosenfeld, Sandhu & Thompson (2025), "On the Origin of Algorithmic Progress in AI", arXiv:2511.21622. MIT FutureTech.
- Small-scale ablations account for under 10x. Surveyed extras add under 10x. The total is under 100x, against the 22,000x (2012–2023) of Ho et al.
- Scaling experiments find **scale-dependent** gains:
  - LSTM→Transformer shows an exponent difference.
  - The **Kaplan→Chinchilla re-balancing** is "a scale-dependent algorithmic change that is not an exponent shift". They give a closed-form compute-equivalent-gain (CEG) function.
- With these, they account for 6,930x. The two scale-dependent changes are 91% of the total.
- "Measures of algorithmic efficiency are strongly reference-dependent."
- Code: github.com/hansgundlach/Experimental_Progress.
- **Overlap.** C6: high. It documents non-neutral, scale-biased progress.
- **Our angle.**
  - In IO language, an exponent change is a change in output elasticities, i.e. non-neutral technical change.
  - The **Kaplan→Chinchilla re-balancing is not technical change at all**. It is a movement from an off-expansion-path input mix to the cost-minimizing one: *allocative efficiency* (Farrell 1957).
  - "Reference dependence" is the index-number problem (Laspeyres vs Paasche) familiar from TFP measurement.

### 2.9 Epoch AI on the training vs inference trade-off (blogs)
- **Villalobos & Atkinson (28 Jul 2023), "Trading off compute in training and inference".**
  - Five techniques: model scaling, MCTS (replicates and extends Jones 2021), pruning, resampling, chain-of-thought.
  - Roughly 1 OOM can be saved in either training or inference by spending somewhat over 1 OOM on the other.
  - Examples: MCTS 1.6/1.6 OOM; resampling with cheap verification 3 OOM for 6 OOM.
  - They do not use the words isoquant, production function or elasticity.
- **Erdil (29 Mar 2024), "Optimally allocating compute between inference and training".**
  - If β OOM of training trades for α OOM of inference, spend a fraction α/(α+β) on training.
  - This is exactly the Cobb-Douglas expenditure-share rule, though not named as such.
- **Erdil & Besiroglu (7 Mar 2025), "Train once, deploy many: AI and increasing returns".**
  - f(T,I) proportional to I·T^c, with c ≈ 1.
  - This gives increasing returns to scale, and hence hyperbolic growth in a macro model.
- **Overlap.** C9: high. Production-function language already exists for train vs inference. C7: low; there is no inversion.

### 2.10 Inference-aware scaling (the forward problem of C7)
- **Sardana, Portes, Doubov & Frankle (2024), "Beyond Chinchilla-Optimal", arXiv:2401.00448 (ICML 2024).**
  - Minimize 6ND + 2N·D_inf subject to L(N,D) = ℓ.
  - Example: with 2T inference tokens, a 13B→7B switch saves 1.7e22 FLOPs (17%).
  - With about 1e9 requests, train smaller and longer.
  - They also find that Chinchilla *over-predicts* gains at extreme tokens per parameter (training runs up to about 10,000 tok/param), which is misspecification in the extrapolation region.
  - Forward problem only.
- **Bian, Yan & Venkataraman (2025), arXiv:2501.18107 (ICML 2025).** Adds architecture (shape) and latency; 63 models; Morph-1B.
- **Roberts et al. (2026), "Test-Time Scaling Makes Overtraining Compute-Optimal", arXiv:2604.01411.** T2 laws jointly optimize N, D and pass@k samples. The optimal decisions move "radically" into over-training.
- **de Vries (2023), "Go smol or go home" (blog).** The critical model size is about 30% of Chinchilla-optimal, at about 100% compute overhead.
- **Overlap with C7.** These papers give the model we invert. None backs out demand or shadow prices from observed (N, D) choices.

### 2.11 Zhang, Jin, Syrgkanis & Kakade (2026), "Prescriptive Scaling Reveals the Evolution of Language Model Capabilities", arXiv:2602.15327 (v2 6 Jun 2026). Harvard / Stanford. Syrgkanis is an econometrician.
- 5k existing and 2k newly evaluated checkpoints (2022–2026), 6 benchmarks.
- **Capability boundaries** are high conditional quantiles (τ ≈ 0.98) of accuracy given log pre-training FLOPs. They use smoothed quantile regression (Koenker-Bassett) with a monotone sigmoid, i.e. a *quantile production frontier*.
- Out-of-distribution coverage error is below 2% on 4 of 6 tasks, but the MATH boundary keeps advancing.
- Attainable accuracy at 10^24 FLOPs: IFEval 0.83, MATH L5 0.54.
- Contamination probes.
- I-optimal sampling recovers the frontier with about 20% of the evaluation budget.
- Data: HF `hlzhang109/proteus-2k`.
- **Overlap.** C10 (SFA/frontier): medium. C8: low-medium. There is no production-function or efficiency-decomposition language; they do not cite Aigner-Lovell-Schmidt or Aragon-Daouia-Thomas-Agnan.

### 2.12 Lourie, Cho, Ullrich & Lotfi (2026), "Small-Scale Experiments: Are We There Yet?", arXiv:2608.11859 (FAIR/NYU).
- "The confounding factor is hyperparameters." Scaling laws "only emerge on the fully tuned frontier" and exist down to 4M parameters.
- The sensitivity to hyperparameters fades with scale.
- **Overlap.** C10: technical inefficiency. Untuned runs sit inside the frontier. In SFA terms, observed loss = frontier + u (u ≥ 0) + v, and u shrinks with scale. The paper uses no econometric framing.

### 2.13 Ruan, Maddison & Hashimoto (2024), "Observational Scaling Laws and the Predictability of Language Model Performance", arXiv:2405.10938 (NeurIPS 2024).
- 77 models; PCA "principal capabilities" S.
- **S_m ≈ θ_f log(C_m) + ν_f**. θ_f is read as the "compute efficiency" of family f. The link to benchmarks is a sigmoid.
- Code: github.com/ryoungj/ObsScaling.
- This is a random-coefficients production function with family-specific TFP (ν_f) and elasticity (θ_f). There is no endogeneity discussion.
- **Overlap.** C5: medium. C1: low.

---

## 3. Tier-2 related work (cite as needed; low to medium overlap)

**Estimation critiques of scaling laws (ML statistics, no IO).**
- **Besiroglu, Erdil, Barnett & You (2024), arXiv:2404.10102.**
  - Refit of Chinchilla Approach 3 on 240 points reconstructed from the plots.
  - Estimates (bootstrap SE): E = 1.8172 (0.03), A = 482.01 (124.58), B = 2085.43 (1293.23), α = 0.3478 (0.02), β = 0.3658 (0.02), a = β/(α+β) = 0.5126 (0.02).
  - Implies 25.6 tokens/param.
  - Hoffmann's own values are E = 1.6934, A = 406.4, B = 410.7, α = 0.3392, β = 0.2849. These imply about 70 tok/param, inconsistent with the 20 actually used.
  - Hoffmann's CIs would need more than 600,000 experiments.
  - Data: epochai.org/code/analyzing-chinchilla-repo.
- **Czech, Xu, Elmatad, Wang & Held (2026), arXiv:2603.22339.**
  - Approach 2 (IsoFLOP parabola) is biased even without noise, through grid width, off-centre sampling, and α ≠ β asymmetry.
  - For Llama 3: 6.5% parameter under-allocation of 3.8e25 FLOPs, costing $1.4M (90% CI $412K–$2.9M).
  - This is a textbook allocative-inefficiency cost, but not framed that way.
- **Schaeffer, Levi, Kirsch, Guenais, Miranda, Obbad & Koyejo (2025), arXiv:2509.23963.** There are three possible definitions of N, differing by up to 15.2%, yet Chinchilla's key results are robust. This is input measurement error.
- **Pearce & Song (2024, TMLR), arXiv:2406.12907.** Reconciles Kaplan and Chinchilla through non-embedding parameters and small scale. This is input definition.
- **Porian et al. (2024, NeurIPS), arXiv:2406.19146.** The discrepancies come from last-layer cost, warm-up and scale-dependent tuning.
- **Choshen, Zhang & Andreas (2024), arXiv:2410.11840.** 485 models, over 1,000 fitted laws; the dataset is released.
- **Li, Kudugunta & Zettlemoyer (2025), "(Mis)Fitting: A Survey of Scaling Laws", arXiv:2502.18969.** 50+ studies; fitted laws shift with the D/N range covered.
- **Volkova et al. (2026), arXiv:2602.07712.** Ill-conditioning across optimizers.
- **Hu et al. (2026), arXiv:2601.19831.** Neural extrapolators.
- **Lourie, Hu & Cho (2025), arXiv:2507.00885.** Downstream scaling laws are unreliable.

**Output measurement (TFPR/TFPQ-like; contamination; selection of what is reported).**
- Schaeffer, Miranda & Koyejo (2023), arXiv:2304.15004. Emergence is an artefact of nonlinear metrics.
- Owen (2024), arXiv:2401.04757.
- Chen et al. (2024), arXiv:2410.08527. Two-stage loss→accuracy mapping.
- Dominguez-Olmedo, Dorner & Hardt (2024), arXiv:2407.07890. Training on the test task confounds evaluation and emergence. This is output mismeasurement correlated with time.
- Zhang, Dominguez-Olmedo & Hardt (2025), arXiv:2507.05195.
- Singh et al. (2025), "The Leaderboard Illusion", arXiv:2504.20879. Private testing and selective disclosure, i.e. selection or attrition in the output measure.
- Epoch Capabilities Index (IRT-based latent capability index; CSV, CC-BY; "A Rosetta Stone for AI Benchmarks").

**Effective-input and TFP-like indices.**
- Hernandez & Brown (2020), arXiv:2005.04305. Input-requirement measure of algorithmic efficiency.
- Davidson, Denain, Villalobos & Bas (2023), arXiv:2312.07413. "Compute-equivalent gain" (CEG), an input-denominated productivity index.
- Xiao et al. (2025), "Densing law of LLMs", Nature Machine Intelligence; arXiv:2412.04315. Capability per parameter doubles about every 3.5 months, i.e. parameter-augmenting productivity.
- Sanderson et al. (2025), arXiv:2505.04075. Compute-dependent vs compute-independent innovations.
- Thompson et al. (2020), arXiv:2007.05558.
- Barnett (2025), arXiv:2507.10618.

**Inputs with economic structure (ML side; our mapping is new).**
- **Muennighoff et al. (2023), arXiv:2305.16264 (NeurIPS 2023; JMLR 26).**
  - D′ = U + (1−δ)U(1−(1−δ)^(R_D))/δ, a geometric-decay perpetual inventory.
  - Fitted R\*_D ≈ 15.4 (±7.3) and R\*_N ≈ 5.3 (±0.1). R\* = (1−δ)/δ is the "half-life" of repeated data.
  - Code: github.com/huggingface/datablations.
- Clark et al. (2022), arXiv:2202.01169, and Krajewski et al. (2024), arXiv:2402.07871. Routed and MoE scaling with effective parameter counts, i.e. capital services.
- Busbridge et al. (2025), arXiv:2502.08606. Distillation scaling laws, where the teacher is an intermediate input.
- Villalobos et al. (2022), arXiv:2211.04325. Data stock limits.
- Gadre et al. (2024), arXiv:2403.08540. Reliable scaling under over-training.
- Li et al. (2025, "Farseer"), arXiv:2506.10972. An N–D coupling term, i.e. non-separability (translog-like).
- Bergsma et al. (2025), arXiv:2505.13738.
- Magnusson et al. (2025, DataDecide), arXiv:2504.11393.

**Economics of LLMs (low overlap; context).**
- Korinek & Vipra (2025), "Concentrating intelligence", Economic Policy 40(121):225–256. Qualitative economies of scale and scope; no production-function estimation.
- Demirer, Fradkin, Tadelis & Peng (2025), NBER w34608, and Demirer, Fradkin & Tadelis (2026), JEP 40(3):23–46. The inference *market* (OpenRouter): prices, a roughly 1000x decline, and short-run price elasticity just above 1. No training production function.
- Bergemann, Bonatti & Smolin (EC'25; arXiv:2502.07736, now "Menu Pricing of LLMs"). Pricing and mechanism design.
- Du (2026), arXiv:2603.28576. DEA/Malmquist on *inference* prices; 318 OpenRouter and 3,237 Epoch models; TFP residual ≈ 103.7% of cost decline.
- Merali (2024), arXiv:2409.02391, and Merali (2025), arXiv:2512.21316. "Scaling laws for economic productivity" in RCTs:
  - 10x compute gives 12.3% faster tasks and 16.1% higher earnings per minute.
  - Each model-year gives 8% less task time, split 56% compute and 44% algorithms.
  - This is downstream output valuation, i.e. the TFPR analogue.
- Narayanan & Pace (2025), arXiv:2502.00909 and arXiv:2503.05816. Elasticity of substitution between AI and *labour* (CES/VES) with log-capability scaling; theory only.
- Zhang & Zhang (2026), arXiv:2601.12339. "Digital intelligence capital"; an augmented CES in compute and effective data inside an agent-based model; calibrated, not estimated.
- Press (2025), arXiv:2511.11572. Descriptive cost summary.
- Douglas & Verstyuk (2025), arXiv:2501.17894. The ASOTA index.
- Erdil, Besiroglu & Ho (2024), arXiv:2405.10494. Idea-production estimation survey. It discusses endogeneity and *explicitly does not correct for it* in the software-R&D case.
- Erdil et al. (2025, GATE), arXiv:2503.04941.
- Besiroglu, Emery-Xu & Thompson (2022), arXiv:2212.08198.
- Ludwig, Mullainathan & Rambachan (2024/25), NBER w33344. LLMs *as tools* for econometrics; not related.
- LessWrong (anonymous, deleted account, 16 Dec 2022), "AI overhangs depend on whether algorithms, compute and data are substitutes or complements". It reads Chinchilla visually as "closer to a perfect complement production function"; the elasticity calculation was begun and abandoned. This is the earliest informal CES reading found.

---

## 4. Negative results: searched for, found nothing

Each of these searches returned nothing on neural scaling. That supports the claims in C1, C3 (ACF framing), C7 and C8, subject to the coverage caveat in §0.

- **"Cobb-Douglas" + scaling law + neural** (arXiv all fields): zero results.
- **isoquant / isocost + scaling laws**: none. IsoFLOP and "IsoLoss contours" (Muennighoff) are used without economic naming.
- **Production-function estimators applied to AI models** (Olley-Pakes, Levinsohn-Petrin, Ackerberg / ACF, control function, proxy variable + neural/LLM/AI models): none.
- **Wedges from over-training** (markup / wedge / De Loecker + over-training, inference-optimal, tokens-per-parameter), and **revealed or implied inference demand**: none. Only the forward problem exists (§2.10).
- **Duality** (Nerlove / cost-function duality / Shephard + compute-optimal): none.
- **DMR impossibility** (Diamond-McFadden-Rodriguez + algorithmic progress): none.
- **"Hicks-neutral"**: appears only as a model label in Ho et al. 2024.
- **SFA and data envelopment** (stochastic frontier / DEA + training scaling law): none. DEA appears only for inference prices (Du 2026).
- **Gross output vs value added, and distillation as an intermediate input**: none.
- **Economics-restricted arXiv search.** All 64 econ-category hits for "scaling law" were checked. None estimates a *training* production function with IO methods.
- **OpenReview** ("scaling laws production function"): no economics-framed paper.

---

## 5. New analytical observations produced during this search

All of these are our own derivations. None was found in prior work. Verify each in the modelling strand.

### 5.1 Elasticity on the expansion path
Chinchilla cost minimization with C = 6ND equalizes the loss elasticities, αA N^−α = βB D^−β. Substituting into Hao & Merrill's σ gives

**σ\* = 2/(2+α+β)**

This is constant along the whole compute-optimal path.
- Hoffmann parameters: σ\* = **0.762**.
- Besiroglu parameters: σ\* = **0.737**.
- OLMo-Hybrid (α, β) from Hao & Merrill fn. 3: about 0.82–0.83.

This explains why Hao & Merrill's "average α and β" gives exactly 0.7622. Off the path, for example under over-training with ε_N/ε_D = r, the local elasticity is σ = (1+r)/(r(1+β)+1+α). For Llama-3-8B this is about 0.77 (Hoffmann) or 0.73 (Besiroglu), so it is nearly constant.

### 5.2 Kaplan vs Chinchilla elasticity
Kaplan eq. (1.5) is L = [(N_c/N)^(α_N/α_D) + D_c/D]^α_D, with α_N = 0.076, α_D = 0.103, N_c = 6.4e13, D_c = 1.8e13 (Kaplan Table 2 joint fit). Isoquants are those of (N_c/N)^p + D_c/D with p = 0.738. This gives **σ ∈ [0.500, 0.575]**, much lower than Chinchilla's 0.74–0.76. So the two canonical laws disagree not only on the expansion path but on curvature. This is a testable hypothesis for the estimation.

### 5.3 Revealed inference demand (DLW analogue)
Minimize p(6ND + 2N·T) subject to L(N, D) = L̄. The FOC ratio gives

ε_N/ε_D = 1 + T/(3D), so **T = 3D(ε_N/ε_D − 1)**

where ε_X = −∂L/∂ln X. Illustration: ε from the two published Chinchilla fits; token counts from Sardana et al. and Meta, with the Llama-3.1-405B figure of 15.6T *not verified this session*.

| Model | D/N | Hoffmann (E4): ε_N/ε_D | T/D | Besiroglu (E3): ε_N/ε_D | T/D |
|---|---|---|---|---|---|
| Chinchilla-70B | 20 | 0.71 | −0.86 | 1.03 | 0.09 |
| Llama-2-7B | 286 | 1.72 | 2.17 | 2.62 | 4.85 |
| Llama-3-8B | 1875 | 2.92 | 5.8 (T ≈ 8.7e13) | 5.22 | 12.7 (T ≈ 1.9e14) |
| Llama-3-70B | 214 | 1.40 | 1.2 | 2.45 | 4.4 |
| Llama-3.1-405B | 39 | 0.78 | −0.66 | 1.35 | 1.06 |

- **Lesson.** The "wedge" is extremely sensitive to the estimated elasticities. Under Hoffmann's own parameters, Chinchilla-70B and Llama-405B get *negative* implied demand; they are "under-trained". This is the exact analogue of Raval (2023) and Bond, Hashemi, Kaplan & Zoch (2020/21) on DLW markups: the wedge is only as good as θ̂.
- **Confounds.**
  - Unmodelled costs: data scarcity, memory, latency, serving batch size.
  - Chinchilla's documented misspecification at over 100 tok/param (Sardana et al.; Gadre et al.).
  - Tokens counted as "training" may include repeated or synthetic data.
- **The DLW–GNR duality.**
  - DLW direction: IsoFLOP-identified ε plus observed shares give the wedge (T).
  - GNR direction: assuming no wedge (T = 0, Chinchilla-optimal labs), observed D/N identifies the elasticity ratio.
- **Dependence on the quantity condition.** For example, dollars per inference token differ from dollars per training FLOP because of hardware utilization. Parameterize T in *FLOP-equivalent* units.

### 5.4 Chinchilla's three approaches as the three classic IO estimation routes
- **Approach 1** (envelope of training curves, i.e. minimum loss at each FLOP) estimates the **cost / frontier function** L\*(C) = E + K·C^−γ, with γ = αβ/(α+β). Nerlove-style. The value of γ is 0.155 (Hoffmann) or 0.178 (Besiroglu).
- **Approach 2** (IsoFLOP minima N_opt(C)) estimates **conditional factor demands** (the analogue of Shephard's lemma). It identifies only a = β/(α+β): 0.456 (Hoffmann) versus 0.513 (Besiroglu).
- **Approach 3** (Huber fit of L(N, D)) is **primal production-function estimation**.
- **Their disagreement.** Hoffmann's Approach 3 implies about 70 tok/param while Approaches 1 and 2 imply about 20. This is the classic primal–dual inconsistency: it signals misspecification or non-optimizing data, as in the IO debates on cost vs production function estimates.

### 5.5 Allocative vs technical efficiency in "algorithmic progress"
The Kaplan→Chinchilla re-balancing (Gundlach et al.) is a movement to the cost-minimizing input mix at the same technology. That is Farrell (1957) allocative efficiency. Ho et al.'s effective-compute doubling time therefore mixes technical change with improvements in allocative efficiency. A Farrell or Kumbhakar-style decomposition is new.

### 5.6 What depends on the choice of output measure
The multiplicative compute constraint makes IsoFLOPs rectangular hyperbolae in (N, D), which are lines in logs. As a result:
- **σ is ordinal.** It is invariant to any monotone transform of output (loss, 1/(L−E), or a benchmark that is a monotone function of loss).
- **Returns to scale are cardinal.** Hao & Merrill's "homogeneous of degree α" holds only for q = 1/(L−E) with α = β.

So claims about RTS need a valuation-based output, which is the TFPR-like problem (cf. Merali's productivity scaling).

### 5.7 Magnitude of lab dispersion
See the benchmark in §2.2: within-firm effective-compute 90/10 of 41x becomes about 1.8–1.9x in reducible-loss units. This is suggestive only.

---

## 6. Equivalences with strength ratings (for the dictionary section)

| ML concept | IO concept | Strength | Note |
|---|---|---|---|
| L(N,D) = E + AN^−α + BD^−β, with q = 1/(L−E) | CES aggregator (ρ = −α) homogeneous of degree α when α = β; generalized (non-homothetic) CES otherwise | exact (α = β), close (α ≠ β) | σ = 1/(1+α) (Hao & Merrill); on the expansion path σ\* = 2/(2+α+β) |
| Irreducible loss E | Output ceiling / satiation level; "q" is a translated output | close | The choice of E changes returns to scale, not σ |
| IsoLoss contours | Isoquants | exact | Muennighoff uses "IsoLoss" |
| IsoFLOP curves (6ND = C) | Isocost curves | close | Hyperbolic, not linear. Linear only in log inputs, with equal "log-prices" |
| Compute-optimal allocation (N\*(C), D\*(C)) | Cost minimization; expansion path; conditional factor demands | exact | The FOC equalizes loss-elasticities |
| Compute-optimal frontier L\*(C) | (Inverse) cost function; Nerlove | exact | γ = αβ/(α+β) |
| Chinchilla Approaches 1/2/3 | Cost-function / factor-demand / primal estimation | close | Their disagreement is a primal–dual inconsistency |
| Fixed-TPP designs; labs on the optimal ray | Functional dependence (ACF 2015); multicollinearity | close | Kricheli et al. prove the statistics; the behavioural source is ours |
| IsoFLOP sweeps / controlled grids | Experiments or exogenous input-price variation that break functional dependence | close | They identify the MRTS at each C |
| Observational scaling on public models | Transmission bias (Marschak-Andrews); OP/LP setting | close | Whitfill and Konig flag it, with no IO remedy |
| Developer / family efficiency (ν_j, θ_f) | Firm TFP (Hicks-neutral) and heterogeneous elasticities (random coefficients) | close | Mertens et al.; Ruan et al. |
| Effective compute / CEG / "compute factor" | TFP expressed in input units (input-requirement distance) | close | Index-number and reference dependence (Gundlach) |
| Algorithmic progress in N_eff, D_eff | Factor-augmenting technical change | exact (as specified) | Ho et al.; Erdil & Besiroglu. DMR limits identification |
| Scale-dependent progress (exponent change) | Non-neutral change in output elasticities | close | Gundlach et al. |
| Kaplan→Chinchilla re-balancing | Allocative-efficiency gain (Farrell) | exact | Currently misclassified as technical progress |
| Over-training for inference (Llama) | Wedge between the output-elasticity ratio and the training cost-share ratio (DLW logic) | close | It is a shadow cost, not a markup. Invert to get T |
| Train vs test-time compute trade-off | Isoquant between two inputs; Cobb-Douglas share rule | close | Epoch (Erdil 2024; Erdil & Besiroglu 2025) |
| Repeated data, effective D′ | Depreciation / perpetual inventory (geometric δ) | exact | Muennighoff eq. 8–10; R\* = (1−δ)/δ |
| MoE total vs active parameters | Capital stock vs capital services / utilization | loose | Routing efficiency is not a utilization rate |
| Distillation / synthetic data from teachers | Intermediate inputs; gross output vs value added (GNR) | close | May inflate developer "TFP" (Microsoft 60.5x) |
| Benchmark accuracy (sigmoid of capability) | Bounded nonlinear output index; TFPR vs TFPQ | loose | Not a price-times-quantity decomposition; closer to quality-index measurement |
| Test-set contamination / training on the test task | Output measurement error correlated with inputs and time | close | Dominguez-Olmedo et al. |
| Leaderboard selective disclosure | Selection / attrition (OP exit correction) | close | Singh et al. 2025 |
| Hyperparameter mis-tuning | Technical inefficiency u ≥ 0 (SFA) | close | Lourie et al. 2026 |
| Quantile capability boundary | Quantile / order-m frontier (Aragon-Daouia-Thomas-Agnan 2005) | exact | Zhang, Jin, Syrgkanis & Kakade 2026 |
| Non-embedding vs total parameter counts; FLOP estimates | Input measurement error | close | Pearce & Song; Schaeffer et al. 2025 |
| Research compute vs researchers (AI R&D) | CES R&D / idea production function | exact | Whitfill & Wu 2025 (not the training law) |

---

## 7. Data leads surfaced by this strand
1. **Mertens et al. 2026.** 809 LLMs: MMLU-Pro, N, D, developer, date. The dataset is "published", but the link was not recoverable from the PDF; check the arXiv HTML page.
2. **Proteus-2k (Zhang et al. 2026)**: https://huggingface.co/datasets/hlzhang109/proteus-2k. About 2.4k open-weight checkpoints on 6 benchmarks, plus pre-training FLOPs.
3. **Kricheli et al. 2026**: https://huggingface.co/datasets/TPPIsCriticalFor/colinear_scaling_models. Controlled collinear vs non-collinear (N, D) grids.
4. **Ho et al. 2024**: https://github.com/epoch-research/lm-algorithmic-progress. About 231 LMs with perplexity, N, D and date.
5. **Besiroglu et al. 2024**: https://epochai.org/code/analyzing-chinchilla-repo. 240 reconstructed Chinchilla Approach-3 points.
6. **Muennighoff et al. 2023**: https://github.com/huggingface/datablations. 400+ runs varying epochs and parameters, i.e. variation off the ray.
7. **Ruan et al. 2024**: https://github.com/ryoungj/ObsScaling.
8. **Epoch AI models database**: https://epoch.ai/data/ai-models. Compute, parameters, tokens, organization, date; CC-BY.
9. **Epoch Capabilities Index**: https://epoch.ai/benchmarks/eci. IRT latent capability plus benchmark difficulty; CSV, CC-BY.
10. **Gundlach et al. 2025**: https://github.com/hansgundlach/Experimental_Progress. LSTM vs Transformer scaling experiments.
11. **Choshen et al. 2024.** 485-model loss and evaluation dataset, released; URL not captured.
12. **OpenRouter inference prices and usage** (as used by Demirer et al. 2025/2026; Du 2026). Serves as a cost-side "price" for T.

---

## 8. Warnings and open uncertainties
- **Budget-limited coverage.** WebSearch ran out; Semantic Scholar was rate-limited. Before submission, re-check Google Scholar and SSRN for 2026 working papers by IO economists (Mertens, Demirer, Raval, De Loecker, Syverson, Syrgkanis) on "AI production functions", and for Epoch "Gradient Updates" posts after mid-2026.
- **Mertens et al. is the most dangerous overlap for C5 and C8.** Mertens is a production-function economist. A follow-up applying ACF or DLW to the same 809-model data is plausible. Consider contacting the authors or framing our work as complementary.
- **Hao & Merrill are the most dangerous overlap for C2.** Never claim the σ derivation. Claim the estimation, σ\*, and the Kaplan contrast.
- **Kricheli et al. own the statistics of C3.** Claim the behavioural and ACF reading and the Approaches 1/2/3 mapping only.
- **Numbers not re-verified this session.**
  - Llama-3.1-405B's 15.6T tokens.
  - Syverson's 90/10 ≈ 1.9.
  - The Nerlove (1963) chapter pages.
  - Hsieh-Klenow's Crossref record resolved to both SSRN and QJE; the QJE 124(4):1403–1448 record was found.
  - Bond et al. appear in Crossref only as NBER w27002 (2020); the JME 2021 version is not verified.
  - A "Demirer (2020) factor-augmenting production function" working paper was *not found* by Crossref. Do not cite it without verification.
- **The inference-wedge illustration (§5.3) is fragile.** It depends on ε estimates extrapolated far outside the Chinchilla support (Llama-3-8B is at 1,875 tok/param, versus Chinchilla's support of about 100 tok/param or less). Present it as an identification argument plus bounds, not a point estimate.
- **Mixed output units.** Ho et al. use perplexity, Mertens a logit of MMLU-Pro, Ruan PCA capabilities. Cross-paper comparisons of "effective compute" are not like-for-like.
- **Other language to avoid.** Do not call the inference wedge a "markup". It is a shadow price of the parameter input from a future cost (inference), not market power. The DLW analogy is methodological only.
