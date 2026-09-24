# Referee Report — Referee 2 (ML scaling laws)

**Manuscript:** "Scaling Laws as Production Functions" (Okar and Claude), AER submission draft of 2026-09-24
**Referee perspective:** machine-learning researcher who has trained and fit scaling laws at a frontier lab. I concentrate on the ML content: Chinchilla and Kaplan facts, FLOP and parameter conventions, learning-rate schedules and tuning, MoE and distillation, token accounting, the design of the authors' own experiment, whether the wedge can be extrapolated to released models, and what practitioners would say about over-training. I leave the IO econometrics to the other referees, except where it depends on ML facts.

**Recommendation:** Major revision.

Page numbers refer to the compiled `main.pdf` (133 pages). Section, equation, table and figure numbers are the printed ones. Where I cite numbers I computed myself, they come from `output/tables/m3_wedge_models.csv` (Sample B core, n = 173) and the reference technology column `w_chin`.

---

## 1. Summary

The paper maps neural scaling laws onto production theory and uses the mapping in three ways.

1. **Identification.** Chinchilla's three approaches are read as a cost-function estimator, a conditional-factor-demand estimator and a primal estimator, linked by duality (Prop. 1). Cost-minimizing labs place runs on the expansion path. On that path, curvature (σ*) and the compute-optimal ratio M* are not identified (Prop. 3). The information about them grows with the second and fourth powers of runs' log deviations from the path (Prop. 4). The sign of biased technical change is identified from the drift of M* (Prop. 5). The paper also signs transmission bias in cross-lab regressions (Prop. 6) and bounds revealed inference demand (Prop. 7).
2. **The technology.** The Chinchilla form gives σ* ≈ 0.73–0.83 in every sweep. The outer exponent restriction κ = 1 is rejected everywhere, and with κ free σ*_κ ≈ 0.51–0.71, about 0.70 on Farseer. Data quality acts mostly like an asymptote shift plus a neutral scaling, with a factor-biased remainder that moves M* about 3-fold across DataDecide recipes. The Kaplan–Chinchilla gap in the allocation exponent splits into head-counting (39–47 percent) and flexible inputs (53–61 percent). A two-lab experiment (FineWeb-Edu vs FineWeb) is still running.
3. **Revealed inference demand.** Inverting the first-order condition of lifetime-compute minimization gives w = ε_N/ε_D = 1 + T/(3D) (Prop. 2). Under a Chinchilla refit, the median of 173 open-weight models has ŵ = 3.19, that is, "lifetime inference compute ≈ 2.2× training compute." The section adds trends, within-family calibration, a usage validation and rival explanations.
4. **Observational production functions.** A LaLonde-style benchmark finds no detectable bias in cross-lab returns to compute. The paper replicates Ho et al.'s algorithmic-progress estimate and finds that the published 8.4 months is not the optimum of its own objective. It argues that the Kaplan→Chinchilla rebalancing is an allocative, not a technical, gain.

## 2. Overall assessment

This is an ambitious paper. It is careful in many places and unusually candid about fragility. Much of Sections II–IV will be useful to ML practitioners:

- the design statistic sd(ln M | ln C);
- the precise second- and fourth-order information rates, a sharp statement of why fixed-ratio ladders and compute-optimal ladders cannot identify curvature;
- the test of the outer exponent;
- clustering by IsoFLOP budget;
- duality tests that are consistent with the design and absorb the finite-grid parabola bias;
- the finding that Huber with δ = 10⁻³ on log loss is LAD in practice;
- the outlier-leverage analysis of Besiroglu et al.'s exclusion rule;
- the Porian decomposition, with a closed-form bias formula that reconciles it with Pearce and Song;
- the optimizer early-stopping problem in Ho et al.

I would use several of these in my own work.

The paper's most prominent claim is the headline of the abstract and conclusion: that the median open-weight model is trained as if lifetime inference will use about 2.2 times its training compute. That claim is not yet credible to an ML audience, for reasons that go beyond the caveats the authors already state:

- **(a) Inconsistent curvature.** Every headline wedge uses the κ = 1 curvature that Section IV rejects. Over the range of σ the paper itself recommends (0.5–0.8), the median ŵ ranges from 0.7 to 8 times its reported value (Major 1).
- **(b) FLOPs are not costs.** The wedge prices a serving FLOP like a training FLOP. For open-weight models, the developer does not pay for most serving (Major 2).
- **(c) Robustness comes from extrapolation.** The "robust sign" is produced by extrapolating one to two orders of magnitude beyond every design. Within the Chinchilla support the median is 1.83, and the technology band lies above 1 for only 58–65 percent of models. The ordinal findings are a monotone transform of M: the Spearman correlation between ŵ and M is 0.999 (Major 3).
- **(d) Inconsistent conventions.** Parameter conventions differ between the reference technology and the models: embeddings, MoE active versus total parameters, distillation and pruning compute, multimodal tokens (Major 4).
- **(e) Rivals are under-tested.** The rival explanations practitioners would raise first are fixed-D families, product tiers, on-device targets, the small stakes of over-training small models, latency, and internal RL/distillation compute. The tests offered have little power against them (Major 5).

The authors' own experiment is the natural place to fix (c). As designed, it cannot. In the parameter convention the wedge uses, it reaches M = 555. M > 1,000 is reached only by the two-layer model. And the learning rate, weight decay and warmup are not re-optimized across D (Major 7).

**My recommendation for the revision** is to take one of two routes:

- **Route (i).** Keep the cardinal claims and do the work that makes them credible: a consistent σ, the expenditure interpretation with calibrated serving costs, consistent N conventions, MoE and distillation handled explicitly, and validation at high M.
- **Route (ii).** Present the wedge as a diagnostic that is ordinal and conditional on a technology, remove "2.2 times" from the abstract, and move the level results to a clearly labelled illustrative section.

Either way, the experiment needs to be redesigned before its results are interpreted.

---

## 3. Major comments

### Major 1. The headline wedges use a curvature that the paper itself rejects, and curvature uncertainty is understated by an order of magnitude

- Section IV rejects κ = 1 in all seven sweeps (p. 33–35, Table 4). The conclusion recommends σ ≈ 0.7, with a range of 0.5–0.8 (p. 58–59).
- Every technology in the Section V reference and band imposes κ = 1, except "Farseer with κ free" (p. 42–43: median 3.85, against 2.41–2.67 with κ = 1).
- By Prop. 2(ii) and eq. (11), ln ŵ = (1/σ* − 1) ln(M/M*(C)). The wedge is therefore highly elastic with respect to σ at the large M of released models.

Holding the reference M*(C) fixed and changing only σ*, I obtain (my computation from `m3_wedge_models.csv`):

| σ* | exponent (1/σ* − 1) | median ŵ, Sample B | Llama 3 8B | Qwen3 0.6B |
|---|---|---|---|---|
| 0.80 | 0.25 | 2.3 | 3.2 | 7.5 |
| 0.737 (reference) | 0.357 | 3.19 | 5.3 | 17.9 |
| 0.71 (Farseer κ free) | 0.41 | 3.8 | 6.7 | 27 |
| 0.69 (Farseer nonparametric) | 0.45 | 4.3 | 8.1 | 38 |
| 0.60 | 0.67 | 8.7 | 22 | 219 |
| 0.50 | 1.00 | 26 | 105 | >3,000 |

The claim in Section II.E (p. 23) and Section V.A (p. 41), that "uncertainty in curvature [contributes] little" (sd 0.047 against 0.143 for ln M*), is correct only within the sampling distribution of the κ = 1 Chinchilla fit. Across specifications the paper considers plausible, it is badly wrong.

This matters in two directions:

- **Consistency.** Using the paper's preferred σ ≈ 0.69–0.71 raises the median wedge to about 4 and Llama 3 8B's to 7–8. The headline "2.2×" should then be about "3×".
- **Credibility.** Section V.B (p. 43) reports that Farseer's local elasticity falls with M. If so, the relevant σ at M in the thousands is lower still. The implied inference volumes for sub-billion models then become implausible (Major 3). I read that as evidence against the first-order-condition interpretation at extreme M, not as a larger demand.

**Requests.**
1. Compute all Section V results under a technology whose curvature matches Section IV's recommendation: the Chinchilla κ-free fit, Farseer κ-free, and a nonparametric σ at the relevant M where available. Propagate σ uncertainty over the recommended range into the band.
2. Rewrite the Prop. 7 discussion (p. 23) and V.A (p. 41) to distinguish sampling uncertainty given κ = 1 from specification uncertainty.
3. Revise the abstract and conclusion accordingly.

### Major 2. What the wedge measures: FLOP pricing, who pays for inference, and what "T" contains

**(a) FLOPs are not costs.**
- Prop. 2 (p. 12) and eq. (10) (p. 40) price a serving FLOP like a training FLOP. Section V.A acknowledges this in one sentence (p. 41: w = 1 + pT/(3D)). Everywhere else, w − 1 is read as "inference compute relative to training compute," and T̂ as a count of tokens (Table 6; ≈1.9×10¹⁴ tokens for Llama 3 8B on p. 13 and p. 43).
- In practice p is far from 1:
  - Decode is memory-bandwidth bound.
  - Serving utilization is typically far below training utilization. Sardana et al.'s own dollar objective, their Eq. 6, has separate utilizations for training, input tokens and output tokens.
  - Prefill (compute-bound) and decode (memory-bound) have different costs per FLOP.
  - Serving uses different hardware and precision (FP8/INT4).
- As an order of magnitude: list prices of roughly $0.1–0.2 per million tokens for 8B-class open models on third-party providers, set against about $2 per H100-hour at 40 percent training utilization, imply a cost per serving FLOP several times the cost per training FLOP. Prices include margins, so this is an upper bound on p. It is enough to make T̂ in tokens uninterpretable without calibration.

**Requested fix.** Under cost minimization, w − 1 identifies the ratio of planned lifetime serving *expenditure* to training expenditure, whatever p is. That is the natural economic object and it needs no FLOP-to-dollar conversion.
1. Restate every headline in expenditure terms.
2. Report T only with an explicit calibration of p: a range from p = 1 to p of about 5–10, using public API prices (the bib already contains `artificialanalysis2026api`) and rental prices.

**(b) Open-weight developers do not pay for most inference.**
- For Llama, Qwen, Gemma, SmolLM and OLMo, serving is largely borne by third parties.
- The lifetime objective therefore needs a reason why the developer internalizes users' serving costs: competition for adoption, ecosystem value, its own serving through Meta AI, Alibaba Cloud or AI Studio. The interpretation of the open-versus-closed premium (p. 46) depends on this.
- One observation weighs against reading this as the developer's own inference demand. The developer with the highest median ŵ in Section V.D is Hugging Face (6.6). Its inference business is the smallest of the developers listed, and its SmolLM models are explicitly on-device models. That is more consistent with a product-tier or memory constraint (Prop. 2(iv), ν > 0) than with the lab's own planned serving (see Major 5).

**(c) Internal uses of the model are part of "T".**
- In 2024–25, RL post-training rollouts (≈2N FLOP per rollout token), synthetic-data generation, distillation-teacher forward passes and evaluation are all compute that scales with N and is known when pretraining is planned.
- For 2025 releases, where the median ŵ is 4.68, RL post-training compute is not negligible.
- Discuss this, and ideally separate "deployment demand" from "internal N-proportional compute." Conversely, the training side should include post-training compute where it is material.

### Major 3. Extrapolation: the sign's robustness is manufactured by the functional form, and the levels need an external plausibility check

- **The median model is itself extrapolated.** The median Sample B model has M ≈ 477 and C ≈ 7.8×10²², beyond the Chinchilla design in both ratio (341) and compute (1.3×10²²).
- **Robustness grows with extrapolation.** In my recomputation:
  - Within the design (M ≤ 341, n = 78): median ŵ = 1.83; the technology band lies above 1 for 58 percent.
  - Within both Chinchilla bounds (M ≤ 341 and C ≤ 1.3×10²², n = 23): median 1.83; band above 1 for 65 percent.
  - Beyond the design (M > 341, n = 95): median 4.56; band above 1 for 96 percent.
- The headline "79 percent have the entire band above one" (p. 4, p. 42) is therefore driven by the models furthest outside every design. This follows from the family: for every member, ŵ → ∞ as M → ∞ at fixed C. The agreement of technologies at extreme M reflects a shared functional form, not information in the data. The paper should say so plainly.
- **The ordinal findings are nearly tautological.** The Spearman correlation between the reference ŵ and M in Sample B is 0.999. "Small models more over-trained than large ones" and "ranks are robust (Spearman 0.95–1.00)" (p. 42, p. 50) restate the ranking by M = D/N, which needs no technology. That modern small models are trained far beyond 20 tokens per parameter is well known; the Llama 3 paper says so. The technology adds the cardinal translation, which is the part the authors concede is fragile.
- **The Sardana et al. argument (p. 23, p. 43) conflates two things.**
  - (i) Level and slope. Sardana et al. report that the fitted law mispredicts loss *levels* at extreme ratios. The wedge needs the local *slope* ε_D at the model's (N, D). The two are related but not the same.
  - (ii) True and perceived technology. Revealed preference recovers T under the technology the lab *believed* when it chose D. If labs plan with fitted Chinchilla-type laws that overstate the value of tokens, their choices embed the overstated ε_D. "Correcting" ŵ toward the true technology then moves it *away* from the lab's first-order condition. The claim that "extrapolated inference demand is conservative" does not follow.
  - Sardana et al.'s Table 1 also shows that fitted exponents change sharply with the range of M used in the fit (≤100 against all data). Since ŵ depends on α + β, this bears directly on the wedge.
- **Plausibility audit of levels.**
  - Implied lifetime T for Qwen3 0.6B is 1.8×10¹⁵ tokens. Summed over Sample B, implied T is about 1.6×10¹⁶ tokens.
  - Compare these with public aggregate volumes: the largest providers disclosed on the order of 10¹⁵ tokens per month across all products in 2025, and OpenRouter publishes weekly token counts by model.
  - Fleet-level disclosures (e.g., Patterson et al. 2022 on the inference share of Google's ML energy) give an order-of-magnitude benchmark for the aggregate in eq. (13).
  - None of these is decisive, but they discipline levels that currently span a factor of 5–10 per model.

**Concrete high-M validation, feasible at modest cost.**
1. *Cool down released stable-phase checkpoints.* OLMo 2 (1B/7B), SmolLM2/3, LLM360 Amber/K2 and TinyLlama release intermediate checkpoints. Short cooldowns (Hägele et al. 2024) at the 1B scale over a few tens of billions of tokens cost about 10²⁰–10²¹ FLOP. They would give annealed losses along D at fixed N, at M in the thousands, on a common held-out set such as Paloma. From these, ε_D at high M can be read off directly, at the scale that matters. Alternatively, adjust intermediate checkpoints with the learning-rate-annealing law of Tissue et al. (2024).
2. *Out-of-sample tests on public sweeps.* Fit the Chinchilla form on M ≤ 100 in Marin (M up to 2,751) and Farseer (M up to 2,570 non-embedding), and compare the predicted ε_D and ŵ at M > 1,000 with the full-grid and nonparametric values.
3. *The authors' experiment, redesigned* (Major 7).

### Major 4. The model's inputs are measured in conventions that do not match the reference technology

**(a) Embeddings.**
- The reference technology counts total parameters on Chinchilla's 32K SentencePiece vocabulary (Table B1, p. 89). Sample B applies it to total parameters of models with vocabularies of 128K–262K.
- Among sub-2B models the embedding share is large: Gemma 3 270M 0.63; Gemma 3 1B 0.30; Qwen2.5 0.5B 0.28; Qwen3 0.6B 0.26; Llama 3.2 1B 0.21; Pythia-70M 0.73.
- The paper's own Section IV.D and eq. (9) show that counting conventions move a by 0.04–0.17, and Farseer's M*(10²³) from 19.5 to 45 (p. 35, p. 38).
- For the loss technology, an embedding parameter is not a transformer-block parameter (Kaplan et al.; Pearce and Song). For cost, the input lookup is free and the output head costs 2dV per token.
- **Request.**
  1. Compute ŵ with a consistent non-embedding technology (the Chinchilla non-embedding refit of p. 38, and Farseer non-embedding) evaluated at N_nonemb, with the output head counted in FLOPs.
  2. Report the median, the sign shares and the band under each convention.
  3. At minimum, note that ∂ln ŵ/∂ln N = −α, so the convention alone moves ŵ by about 10–40 percent for sub-billion models. That is larger than the 4–7 percent tokenizer and D effects discussed on p. 27 and p. 41.

**(b) Mixture of experts.**
- The paper includes 12 MoE models in the Sample B core and many of the largest 2024–25 models in the production-scale universe. Using active N in a dense technology treats DeepSeek-V3 (671B total) as a 37B dense model in its loss.
- At given active parameters and D, MoE loss is lower, and the optimal tokens per active parameter differ (Clark et al. 2022; Krajewski et al. 2024; Abnar et al. 2025). Serving memory scales with total parameters.
- The bounds are wide. With total N, DeepSeek-V3 has M ≈ 22 and Kimi-K2 M ≈ 15, which gives ŵ ≈ 1.1 and ≈ 1.0 under the reference technology. With active N, the paper reports 3.09 and 3.30.
- **Request.**
  1. Exclude MoE from headline statistics, or use an MoE scaling law.
  2. Report [total-N, active-N] bounds per MoE model in Table 6 and Figure 6.
  3. Show the open-weight premium and the trend regressions (p. 46) without MoE.

**(c) Distillation and pruning.**
- The affected models are Gemma 2 2B/9B, all Gemma 3 sizes including the 27B flagship, Llama 3.2 1B/3B, and Llama 4 Maverick. Llama 3.2 1B/3B were pruned from Llama 3.1 8B and then distilled with logits from 8B and 70B.
- With logit distillation, teacher forward passes add 2N_T·D to training cost. The first-order condition becomes
  w = (1 + T/(3D)) / (1 + N_T/(3N)).
- For a 1B student with a teacher of ≥27B, the denominator is ≥ 10. The observed ŵ then implies a T an order of magnitude larger, if the technology were otherwise unchanged. It is not: see Busbridge et al. (2025) on distillation scaling laws.
- For pruned models, D omits the ~15T tokens embedded in the initialization.
- The dummy-variable test (p. 48; 0.02, s.e. 0.23) has no power, as the authors say.
- **Request.** Exclude distilled and pruned models from the headline statistics and the family analysis, or model them explicitly.

**(d) Multimodal tokens.**
- Llama 4 Scout and Maverick (the model cards give ~40T and ~22T *multimodal* tokens) and Gemma 3 4B–27B were pretrained on text and images.
- D therefore includes non-text tokens. This contradicts the Sample A rule that drops multimodal models "whose D would include other tokens" (App. B, p. 96).

**(e) Tokenizer and M\* units.**
- Section III (p. 26) states that M* is never compared across units. Section V nevertheless applies Chinchilla's M*, in SentencePiece-32K tokens and total N, to Qwen, Gemma and Llama tokens.
- The 7 percent correction (p. 41) covers only the compression of D. It misses the vocabulary-driven differences in N and in the loss technology (Tao et al. 2024).

### Major 5. The behavioral model and the rivals practitioners will raise first

Identification assumes that each model's N and D are chosen jointly to minimize that model's lifetime cost. Several features of how labs actually decide are observationally close to "inference demand" and are not tested with power.

**(a) Fixed-D families and shared pipelines.**
- In my count, 20 families contributing 70 of the 173 models train every size on a single D: Qwen2.5 (7), Qwen3 (6), Pythia (8), OPT (8), RWKV (6), XGLM (4), Llama 2/3, Yi, and others.
- The corpus, tokenizer, curriculum and annealing mix are set once per family, and sizes are set by deployment tiers. Under this model, each small sibling's ŵ reflects the family's D, not its own first-order condition.
- The "smaller siblings are more over-trained" result (p. 46, p. 117) is mechanical, as the paper partly concedes. The over-identification rejection (p. 47) is what this model predicts.

**(b) The stakes are small where ŵ is largest.**
- Qwen3 0.6B's 1.3×10²³ FLOP is a small absolute cost: tens of thousands of GPU-hours at most.
- With any fixed cost of customizing D per size (data mixing, schedule tuning, evaluation), reusing the family's D is rational whatever T is. The first-order-condition inversion is least reliable exactly where ŵ is extreme.

**(c) Size classes as product segmentation and memory tiers.**
- Leaderboards and users stratify by parameter count, and competition within a class raises D at fixed N. This is Prop. 2(iv) with ν > 0 and T = 0.
- The bunching test (p. 47–48, Table D8) uses a single 16-bit tier (6.5–9.5B). Local deployment is mostly quantized, and at 4/8-bit the tiers are roughly ≤3B (phones), 7–9B (8 GB GPUs and laptops), 12–14B (12–16 GB), 27–32B (24 GB) and 70B (48 GB or more). Note also that an 8B model in bf16 has 16 GB of weights and does not fit a 16 GB card with its KV cache.
- **Request.** A multi-tier bunching test at quantized tier boundaries. Also code stated deployment targets (on-device: Llama 3.2 1B/3B, Gemma 3 270M/1B, SmolLM/SmolLM2, Phi-3-mini) and test whether ŵ differs given C and year.

**(d) Latency service-level objectives.** Time per output token depends on shape (depth), not only on N (Bian et al. 2025, in the bib but not cited). Latency constraints produce memory-type wedges.

**(e) Sunk compute.** Clusters are provisioned for the flagship, and small models are often trained on capacity that would otherwise be idle. The shadow price of training compute is then below its rental price, which gives large w with small T. The expenditure interpretation of Major 2 absorbs part of this; the token interpretation does not.

**(f) Capability is not a function of loss alone.**
- The claim that ŵ is ordinal and "unchanged by ... a monotone benchmark link" (p. 8) requires downstream capability to be a function of pretraining loss alone across (N, D).
- There is supporting evidence: Gadre et al. find downstream error is a function of loss across M. There are also known departures: knowledge capacity per parameter, trainability by RL, and robustness to post-training quantization. Kumar et al. (2024) show that over-trained models degrade more when quantized, which cuts against extreme over-training for quantized deployment.
- Discuss these and cite them.

**(g) Data constraints on flagships** push flagship ŵ down (Prop. 2(iv)). The normalization by the flagship (eq. 12) inherits this.

**Requests.**
1. Define an "inference-demand sample" that excludes fixed-D research suites (Pythia, OPT, BLOOM, RWKV-Pile, XGLM, Cerebras-GPT, GPT-Neo/J), MoE, distilled/pruned, multimodal and instruct-only checkpoints. In my rough recomputation, excluding research suites, MoE, distilled and synthetic-data models leaves 93 models with median ŵ = 2.96 and the band above 1 for 77 percent.
2. Run the tests in (c), and a developer-level test: does ŵ rise with the developer's own serving footprint, as the demand story predicts, or with an on-device focus, as the tier story predicts?
3. Report which story each test can reject.

### Major 6. The choice of reference technology, and the circularity of the stated-intent and revealed-preference checks

**(a) The Chinchilla reference tilts M\* the wrong way.**
- The reference is a 2021 recipe: MassiveText, cosine schedule, SentencePiece-32K. Its refit has β > α, so M*(C) *falls* with compute: 21.4 at 10²¹ and 17.6 at 10²⁴ (p. 42).
- Modern published evidence points the other way or elsewhere:
  - Meta's Llama 3 law puts the frontier optimum at about 402B parameters and 16.55T tokens at 3.8×10²⁵ FLOP (M ≈ 41).
  - Farseer's M* rises with compute (p. 49).
  - MiniCPM, with WSD, reports compute-optimal ratios on the order of 10² tokens per parameter.
  - DeepSeek LLM finds that the allocation depends on data quality: higher quality shifts compute toward model size.
- Most Sample B models sit at 10²²–10²⁵ FLOP, so the extrapolated M*(C) dominates the levels. A tenfold change in M* changes ŵ by a factor of 10^0.357 ≈ 2.3.
- **Requests.**
  1. Add the published modern laws (DeepSeek LLM, MiniCPM, Llama 3 for all models, Marin's three corpora, OLMo ladder for all models) to the band.
  2. Justify the choice of reference. I would make a modern, open, high-M sweep (Farseer or Marin) the reference and treat Chinchilla as historical.

**(b) The stated-intent checks are close to circular (p. 45, Table D8).**
- Chinchilla-70B and Cerebras-GPT were sized with the 20-tokens-per-parameter rule derived from the very runs used as the reference technology. That their ŵ ≈ 1 confirms that the refit's M* ≈ 20 matches Hoffmann et al.'s. It does not show that the wedge reads intentions.
- Likewise, the "revealed-preference test" at Chinchilla-70B (Table 3, Panel C; p. 31) is a duality check at one extrapolated point. DeepMind's choice of 70B and 1.4T was computed from Approaches 1–2 on these same runs.
- Reframe both as internal-consistency checks.

### Major 7. The authors' controlled experiment: as designed, it cannot answer the questions it is meant to answer

I comment on the design only; the results are pending. Sources: Section III.B (p. 26–27), Section IV.E (p. 39–40), App. B (p. 93–96), Table B2 (p. 94), and `paper/notes/m9_spec.md`.

**(a) The range of M does not reach the target region, and high M coincides with the smallest model.**
- The motivation (p. 26) is that 55 percent of released models exceed M = 341. In the total-N convention used by the reference technology and by Sample B, the design's maximum M is 555: width 128 at 800M tokens. That is one endpoint per lab above 341.
- In non-embedding units the maximum is 2,031, but M > 1,000 is reached only by width 128 at 400M and 800M tokens. That model has two layers, 0.39M non-embedding parameters, and 73 percent of its parameters in embeddings (Table B2).
- The planned extrapolation check at "M ≈ 1,000–2,000" (p. 40, p. 43) therefore rests on two endpoints per lab from a two-layer network. High M coincides exactly with the smallest, shallowest model, so any departure from the law could be a small-model artifact; Kaplan et al. document deviations for very shallow models.
- **Redesign.**
  1. Extend D to about 3–25B tokens for widths 128–384, using more shards of FineWeb and FineWeb-Edu (sample-100BT or the full releases), so that M spans roughly 20 to ≥10⁴ *in the total-N convention* at three or more widths. Tiny models are cheap: a 5M-parameter model at 10B tokens is about 3×10¹⁷ FLOP.
  2. Put a floor of four layers on the main fits and treat the two-layer model as a robustness check.

**(b) The flexible inputs are not concentrated out, and the resulting bias points toward the answer the paper expects.**
- The peak learning rate was calibrated with four learning rates at three widths, at a single D (50M), for one lab (FineWeb-Edu), with extrapolation allowed (p. 95). It is then applied at every D from 25M to 800M and to both labs.
- The optimal peak learning rate falls with the training horizon (Bjorck et al. 2024).
- The weight-decay timescale should scale with D (Bergsma et al. 2025, in the bib; Wang and Aitchison 2024). With a fixed λ = 0.1 and a learning rate tuned only to width, the AdamW EMA timescale is fixed in steps while runs range from about 1.5K to 49K steps.
- The fixed 250-step warmup is 16 percent of the 25M-token runs and 0.5 percent of the 800M-token runs. That is the same kind of scale-correlated warmup the paper identifies in Porian et al.'s step 3.
- By the paper's own formula (p. 38), a transverse gradient of inefficiency (ι_d ≠ ι_n) biases a, and with it the curvature. Mis-tuning that worsens with D inflates losses at high M. That makes extra tokens look less valuable, which biases the extrapolation check toward the Sardana direction that Section V hopes to confirm.
- Lourie et al. (2026, in the bib) find that scaling laws appear only on a well-tuned frontier; four configurations per scale were not enough in their study.
- **Fix.** Run three trunks per width at learning-rate multipliers {0.5, 1, 2}. The trunk-and-branch design keeps the cost linear. Optionally cross them with two weight-decay values. Report the per-cell envelope (the concentrated technology) and the share of cells whose optimum is interior. Calibrate the learning rate separately for the FineWeb lab; a lab-specific mis-tuning would masquerade as factor bias, which is the object of the neutrality test.

**(c) Noise, dependence and power.**
- Endpoints of one width share a trunk up to their branch points, and data order is common across widths. The effective sample is therefore 8 clusters per lab, not 44 runs.
- Seed replicates exist only for FineWeb-Edu, at low M (50–200M tokens at widths 128/256/384). The high-M corners and the FineWeb lab have no estimate of noise.
- **Requests.**
  1. Add seeds at the high-M corners and for both labs.
  2. Base inference on a wild cluster bootstrap over widths, or a parametric bootstrap with the seed-estimated covariance.
  3. Before interpreting results, report a Monte Carlo power calculation with the Section II machinery calibrated to this design. Two targets: the precision of σ*_κ (Farseer needed 404 runs to reach ±0.006; the 30–35-run sweeps give ±0.1), and the power to detect a factor bias of the size in DataDecide (tilts of 0.22–0.26).
- Note also that "all widths share the initialization seed" (p. 26) does not create common random numbers in weight space across different shapes. Only the data order is common.

**(d) The "common output" is not neutral.**
- FineWeb-Edu is a classifier-filtered subset of FineWeb. On edu-val the edu lab has a home advantage, and on web-val the web lab does.
- Asymptote shifts will partly measure distribution match, as the paper notes for DataDecide on p. 36.
- **Requests.**
  1. Add a third held-out set that is neutral with respect to both labs (e.g., Paloma subsets, Wikipedia, books) and pre-specify the primary output.
  2. Report D in bytes as well as tokens. The labs' bytes per token differ (3.89 vs 3.74), so ψ_D per token mixes compression with quality.

**(e) Pre-registration.** The placeholders already anticipate directions ("whether ε_D is overstated," "sign relative to Lemma A..."). Before the results arrive, fix and record:
- the primary N convention, and why;
- the output;
- the estimator and treatment of E;
- the clustering;
- the exact extrapolation statistic: finite-difference ε_D at fixed width, or fitted.

**(f) Scope.**
- The runs use at most 10¹⁷ FLOP, a context of 256 tokens, and a batch of 16K tokens, far below the critical batch size. The token efficiency of this regime is not that of industry's large-batch runs.
- State ex ante which conclusions are expected to transfer (the sign of σ − 1, neutrality) and which are not (levels).
- A handful of anchor runs at 100–300M non-embedding parameters on rented GPUs, at M ≈ 20 and M ≈ 1,000, would add a great deal.

### Major 8. The evidence on σ is narrower than presented

- **Headline "σ ≈ 0.7".** It rests mainly on Farseer: 404 runs, non-embedding N, bits per character, hyperparameters set by the Step Law rules. The only other support is a κ-free Chinchilla fit to digitized, quantized losses, for which the paper says it "rel[ies] on the replications" (p. 31). The other κ-free estimates (0.51–0.62) have intervals of about ±0.1 and κ̂ between 0.22 and 0.42.
- **"Seven sweeps".** This counts Gadre et al.'s three corpora (one codebase, design and evaluation) as three sweeps. Say five sources.
- **κ may absorb other misspecifications.**
  - Parameter convention: Farseer's Chinchilla-form σ* moves from 0.772 to 0.724 when embeddings are counted. What is σ*_κ?
  - Curvature among the smallest models: drop the smallest sizes.
  - The E–κ trade-off, which is severe at small n. Irreducible-loss estimates before saturation are unreliable (Lourie et al.).
  - Tuning quality that varies with D: Farseer uses the Step Law rules, whose inefficiency the paper finds to be data-biased (p. 39).
  - **Request:** a robustness table of σ*_κ over these choices.
- **DataDecide.** Its σ* = 0.828 sets the top of the headline range "0.73–0.83" (abstract logic, p. 4, p. 33; Figure 4). It is identified only from checkpoints taken before the cosine schedule ends. Appendix B (p. 91) concedes that levels are uninterpretable, but the checkpoint effect is a function of D/N and so contaminates curvature as well. Remove it from the headline range, or correct it with an annealing-aware law (Tissue et al. 2024) and show the correction.
- **Practitioner translation.** ML readers will use σ only through what it implies for over-training costs. Tabulate the compute overhead C/C_min of training at k × M* for k = 2, 10, 100, 1,000 under σ* = 0.6, 0.7, 0.74 and 0.8, using Prop. 2(iii), and relate it to de Vries's (2023) compute-overhead curve. This would be the paper's most-used table in ML.

### Major 9. Chinchilla and Kaplan facts that need correction or qualification

**(a) Approach 1 is mislabelled and is not implemented as Approach 1** (p. 10, Table 1, p. 31).
- In Hoffmann et al., Approach 1 fixes model sizes and trains each with several cosine cycle lengths. It takes the envelope of the training curves over FLOPs and reads off, for each FLOP count, both the minimal loss *and the argmin N and D*. Its reported output is the allocation exponents (a = b = 0.50). It is as much a factor-demand estimator as Approach 2.
- The paper's "Approach 1" is a frontier fitted through the nine IsoFLOP parabola minima of Approach 2 (memo m1, H6). The intermediate-checkpoint envelopes that define Approach 1 are not in Epoch's extraction of final losses.
- Rename it (e.g., "IsoFLOP-minima frontier") and correct the classification in the discussion of Prop. 1 and in Table 1.

**(b) D = C/(6N) in the Chinchilla extraction** (p. 24, p. 92).
- Hoffmann et al. describe a more detailed FLOP count (their Appendix F) that includes embedding and attention terms, and its ratio to 6ND varies with model size.
- If the budgets in Figure 4 use that count, the constructed D inherits an error correlated with N. That is exactly the measurement problem of eq. (9).
- As a robustness check, rebuild D from Hoffmann et al.'s architecture table and FLOP formula, and report a, M* and the Chinchilla-70B wedge.
- The same concern applies to the digitized Llama 3 profiles, where N = C/(6D) assumes nominal budgets equal to 6ND (p. 92). For Marin the paper itself finds nominal budgets 7 percent below to 35 percent above 6ND, correlated with N.

**(c) Kaplan's joint law** (p. 7, p. 11, Table 1).
- Kaplan et al.'s L(N, D) was fitted to *early-stopped* runs on datasets of D unique tokens, in the overfitting regime, with non-embedding N.
- Their N ∝ C^0.73 allocation came from L(N, S) and critical-batch arguments, not from L(N, D).
- The statement that "the two canonical laws disagree about the curvature of the isoquants" (p. 11) compares a data-constrained, converged law with a single-epoch, compute-bounded law. Qualify it, and qualify the use of Kaplan's κ ≈ 0.103 as a member of the same family.

**(d) Terminology.** "Chinchilla's Approaches 1 and 2 recover (a, G) from choices" (p. 16). IsoFLOP argmins are computed from the outcomes of designed runs, not from optimizing behavior. The distinction matters for the paper's own argument about behavior.

### Major 10. Data handling in Sample B and the production-scale universe

- **Duplicates.**
  - Meta-Llama-3-8B/Llama-3.1-8B and Meta-Llama-3-70B/Llama-3.1-70B enter as separate observations with identical N and D (Table 6 notes). The 3.1 checkpoints are continued pretraining of the same runs, not new allocation decisions.
  - Qwen-72B/Qwen1.5-72B and OpenLlama v1/v2 at 3B and 7B also have identical (N, D).
  - In the 2024 aggregate, the two 70B Llama entries each carry about 5 percent of compute. Deduplicate by pretraining run.
- **Instruction-tuned checkpoints in a "base" sample.** Phi-3-mini/small/medium-instruct, Phi-3.5-mini-instruct, Phi-3.5-MoE-instruct and Phi-4-mini-instruct are included, although App. B says instruction-tuned models are dropped.
- **Corpus size versus tokens processed.**
  - The source note for Qwen2.5's D reads "18T tokens pre-training corpus"; whether every size processed 18T is not documented.
  - In Sample A, missing epochs are set to 1 (p. 96), which turns corpus sizes into tokens seen by assumption.
  - The Qwen1.5 token counts come from ObsScaling; give primary sources or drop them, as was done for Qwen1.5-110B.
  - Report the share of repeated or synthetic tokens where it is disclosed (e.g., phi-1.5's 150B tokens are several epochs over about 30B). Consider the effective-data correction of Muennighoff et al.
- **Research suites.** 44 of the 173 models come from suites whose D is fixed by research design. The paper calls Pythia "a useful warning" (p. 45) but keeps these suites in the median.
- **Non-transformer and hybrid architectures.** RWKV in Sample B, and Nemotron-H and Falcon-H1 in the universe, have different FLOP accounting and serving costs (no growing KV cache). Flag or exclude them.
- **Request.** A table of the median ŵ, the share with ŵ > 1 and the share with the band above 1 under each cleaning step. My rough recomputations: excluding research suites 3.44; excluding MoE 2.88; excluding MoE, distilled and synthetic 2.62; the clean 93-model sample 2.96.

### Major 11. The validation against usage is weak and uses noisy proxies

- Hugging Face download counts mix CI and test pulls, fine-tuning and research use. Local inference mostly runs on community quantizations (GGUF/AWQ/EXL2) hosted in third-party repositories, disproportionately for models of 7B and above. Base-repository downloads (even summed with official instruct repositories) therefore under-count serving demand for larger models, which biases toward the reported sign.
- The coefficient on ln M at fixed C (0.97) is equivalent to −1.94 on ln N (p. 48). The paper acknowledges this size confound.
- **Better outcomes.**
  - Weekly token volumes by model, which OpenRouter publishes; the paper uses only listings and provider counts.
  - Counts of quantizations and fine-tunes from the Hugging Face "model tree".
  - Third-party provider counts and prices.
- **A test with bite.** Relate ŵ to the developer's own serving footprint versus an on-device product focus (Major 5).

### Major 12. The paper misses directly relevant ML literature, most of it already in `references.bib`

The following are in the bibliography file but not cited in the text, or cited only in an appendix table.

- **The training–inference trade-off, i.e., the forward problem of the wedge:**
  - Villalobos and Atkinson (2023);
  - Erdil (2024), "Optimally allocating compute between inference and training";
  - Erdil (2025) on inference economics;
  - de Vries (2023) on the compute overhead of smaller models;
  - Bian et al. (2025) on latency-aware scaling;
  - Roberts et al. (2026), "Test-time scaling makes overtraining compute-optimal", which offers a distinct mechanism for over-training.
- **MoE:** Clark et al. (2022), Krajewski et al. (2024), Abnar et al. (2025).
- **Distillation:** Busbridge et al. (2025).
- **Schedules and tuning:** Tissue et al. (2024) on learning-rate annealing; Lourie et al. (2026) on tuning in small-scale experiments; Bergsma et al. (2025) on weight decay and batch size.
- **Precision:** Kumar et al. (2024), on over-training and quantization.
- **Data quality and allocation:** DeepSeek LLM (Bi et al. 2024), which directly studies how data quality changes the compute-optimal allocation and replaces 6N with non-embedding FLOPs per token for small-scale accuracy; MiniCPM (Hu et al. 2024) on WSD and high M*; Goyal et al. (2024) on data filtering that depends on compute.

The novelty claim for the *inversion* is fine. But the discussion of what the wedge means, of rivals and of the experiment's contribution must engage with these papers. DeepSeek LLM is a direct precedent for the data-quality question in Section IV.C and in the experiment.

---

## 4. Minor comments

1. **Abstract, p. 1.** Qualify "trained as if its lifetime inference compute will be about 2.2 times its training compute" with "under a Chinchilla-form reference technology." Replace "verified parameter and token counts" (also p. 4) with "documented": token counts are self-reported by developers.
2. **P. 2 and p. 33.** "Seven public sweeps": say five sources (Gadre et al.'s three corpora are one sweep).
3. **P. 8, I.A.** "A deployed model spends about 2N FLOPs per token." This is the forward matmul cost and ignores attention (2·n_layer·n_ctx·d_attn per token at long context). In decode the binding cost is often weight and KV-cache memory traffic, not FLOPs. One sentence would help.
4. **P. 8, I.A.** "M* counts tokens and so depends on the tokenizer, while the exponents do not." The exponents depend on the parameter-count convention (the paper's own p. 35 and p. 38), and may depend on vocabulary size (Tao et al. 2024).
5. **P. 13.** "Roughly 1.9×10¹⁴ tokens over its life": put the p caveat (Major 2) here, where the number first appears.
6. **P. 13, Harberger.** "Overstates the loss by 23 percent at w = 5.2": say it overstates C/C_min by 23 percent (about 12 percent in logs).
7. **P. 19, Fig. 2 notes.** One of nine starting values is the truth. Add a column of results with random starts only.
8. **P. 19 and App. C, Monte Carlo.** Noise is iid homoskedastic Gaussian. Real seed noise is larger at small N and short runs, and correlated within budgets through shared data order. Add a heteroskedastic, clustered-noise variant.
9. **P. 23, Prop. 7 discussion.** State that the decomposition (0.143 against 0.047) is conditional on κ = 1 (Major 1).
10. **P. 24.** The Chinchilla sample has "N 57M–16B" in Table 2, but Hoffmann et al.'s smallest model is 44M. State whether the smallest runs are among the dropped outliers or absent from the extraction.
11. **P. 26.** Common random numbers: say that only the data order is common across widths (Major 7c).
12. **P. 27.** "A 10 percent error in D moves the revealed wedge by about 4 percent": add that the embedding convention moves it by α·ln(N_total/N_nonemb), about 10–40 percent for sub-billion models with large vocabularies.
13. **P. 31.** "The Approach-1 minima lie 0.005–0.008 nats below each Approach-3 frontier": rename (Major 9a). Given that the digitization is quantized to about ±0.01, discuss whether a 0.005–0.008 gap is detectable.
14. **P. 36, DataDecide.** The assumption that "the unfinished schedule affects all recipes alike" is partly testable. Estimate recipe tilts from final checkpoints only, where the schedule is complete (a single ray, M ≈ 100, which still identifies relative A_r/B_r under common exponents), and compare them with the all-checkpoint tilts.
15. **P. 38.** "At 2.5×10¹⁶ FLOP the untuned runs' excess loss at the tuned optimum is 1.57 nats." This coincides numerically with the "1.57-billion-token warmup" a sentence earlier. Please verify it is not a transcription error.
16. **P. 37–38, Porian decomposition.** Step 1 uses training loss and the Kaplan count; steps 2–5 use validation loss. State whether the 39–47 percent "measurement" share includes the switch from training to validation loss, or report it separately.
17. **P. 41.** The 7 percent tokenizer calculation ignores the change in N through the vocabulary. See Major 4(e).
18. **P. 43.** "Qwen3 0.6B ... its level is not credible, but its rank is": its rank is its rank in M. Say so.
19. **P. 45–46, trends.** Closed models' N and D in Epoch are often estimates with "likely" or "speculative" confidence. Show the trends and the open–closed premium using only rows with confident N and D.
20. **P. 46.** The open-weight premium "conditional on ... MoE architecture" depends on the active-N convention. Report it with MoE excluded and with total N.
21. **P. 47–48, hardware tiers.** Define several tiers at 4- and 8-bit precision (Major 5c). At bf16, an 8B model's weights alone take 16 GB.
22. **P. 48, distillation and synthetic dummies.** Drop them or replace them with a model of teacher compute (Major 4c). Phi models' synthetic tokens are also multi-epoch.
23. **P. 49, aggregate.** Show a leave-one-out analysis. Llama 3.1 405B alone is about 30 percent of 2024 open-weight compute. Deduplicate the Llama 3/3.1 entries.
24. **P. 51–52, LaLonde benchmark.**
    - Say how "log odds of above-chance accuracy" is handled when accuracy is at or below chance, which is common for the smallest designed runs on HellaSwag.
    - ObsScaling and Sloth use different evaluation harnesses and few-shot settings. Document the harmonization.
    - Classifier-filtered corpora (DCLM-style) partly target benchmark-like text, which is a form of output contamination.
25. **P. 54.** "Where Epoch reports independent compute estimates, the implied reliability of ln C is at least 0.988." Many Epoch compute entries are derived from 6ND using the same N and D. Restrict to estimates based on hardware and time, or explain the sense in which they are independent.
26. **Fig. 1 (p. 12).** The Llama 3 8B point uses total N. Add a marker for non-embedding N.
27. **Fig. 4 (p. 33).** The legend still says "q"; switch to κ (flagged in the integration log). Mark the DataDecide estimate as identified from checkpoints.
28. **Fig. 6 (p. 42).** Plot MoE models at both active and total N, joined by a segment. Mark distilled, pruned and multimodal models.
29. **Table 6 (p. 44).**
    - Add, for DeepSeek-V3, ŵ at total N.
    - Flag Gemma 3 27B as a distilled flagship; its † is easy to miss.
    - Flag Llama 3.2 1B/3B as pruned from Llama 3.1 8B.
    - Add the embedding share as a column.
30. **Table 7 (p. 48).** Report the specification with outcomes in the instruct repositories alone, and with counts of quantized derivatives.
31. **Section III.B (p. 26) and App. B (p. 93–95).** State the cooldown shape (1 − sqrt) and the warmup in the main text. They matter for comparison with Hägele et al. and Porian et al.
32. **App. B, Llama 3 profiles (p. 92).** The loss units are "unstated" (0.69–0.93). If these are normalized or per-character losses, κ and γ from them are not comparable with the others. Say this where "Meta's law" enters Section V.
33. **Section V.E (p. 46).** Flagships after 2024 have a median ŵ of 3.14 under the reference, and 1.64–2.55 under Meta’s law (Table D7). This is as consistent with a higher modern M* as with over-trained flagships (Major 6a). Say so.
34. **Replication.** Release per-model wedges under every technology and convention (total/non-embedding N; active/total N for MoE), with token-count sources. ML readers will want to check individual models.
35. **Terminology.** "Tokens processed" versus "tokens seen" versus "corpus size": use one term consistently, define it once (p. 7), and flag every model where D is a corpus size.

---

## 5. Would ML readers find the paper credible and useful?

- **Sections II–IV and VI.** Largely yes, after the corrections in Majors 8–9. They are useful to practitioners: the design statistic, the κ test, budget clustering, duality tests consistent with the design, the explanation of Kaplan–Chinchilla through measurement and flexible inputs, and the Ho et al. convergence finding. Practitioners will ask for σ translated into compute overheads (Major 8).
- **Section V.** In its current form, ML readers will read it as a restatement of M = D/N (Spearman 0.999) with a fragile cardinal overlay. They will immediately raise FLOP-versus-cost pricing, who pays for open-weight inference, MoE/embedding/distillation conventions, fixed-D families and product tiers.
- **The own experiment.** As designed, it will not persuade: the M range, the two-layer confound and the tuning.

**Priority list for the revision.**
1. A consistent σ in the wedge (Major 1).
2. The expenditure interpretation and calibration of p (Major 2).
3. Consistent conventions for N (embeddings, MoE) and D (multimodal, corpus), with distilled and pruned models excluded or modelled (Major 4).
4. Out-of-sample validation at high M, on public sweeps and on released checkpoints after a cooldown (Major 3).
5. A redesigned experiment: M ≥ 10⁴ in total-N units at several widths, learning-rate and weight-decay envelopes, seeds at the corners, a neutral output, pre-registration (Major 7).
6. A clean inference-demand sample and sharper rival tests: multiple tiers, on-device intent, developer serving footprint (Major 5).
7. Modern lab laws in the band (Major 6).
8. Corrections of Chinchilla and Kaplan facts (Major 9).
9. Reframing of the abstract and conclusion.
