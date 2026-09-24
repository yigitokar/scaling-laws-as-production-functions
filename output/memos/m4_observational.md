# Module m4_observational: Observational (cross-lab) production functions vs experimental benchmarks

**What this module does.** It runs a LaLonde-style test of IO production-function estimators on LLMs and measures lab TFP dispersion. Date: 2026-09-23.
- **Code:** `code/analysis/m4_observational/run.py` is the single entry point. It takes 25–32 minutes on 1 CPU, depending on machine load (average load 25–45). Seeds are deterministic: repeated full runs give byte-identical tables.
- **Extra raw data:** `code/data/download_m4_observational.sh`.

> **Revised after independent review (2026-09-23).** The review found and fixed two data errors: 5 duplicated Pythia rows, and parameter counts inflated by stored attention-mask buffers. It also added few-cluster (wild-bootstrap) inference and corrected several claims. All numbers below are from the corrected run. Details are in `output/memos/m4_observational_review.md`.
>
> **What changed for the writers:**
> 1. The Table 6 compute result survives: OLS bias −0.001 (0.038).
> 2. The "N/D split is biased" claim is weaker. The signs are not robust to the benchmark surface or to the output measure. Under few-cluster inference, only the family-FE biases are near significance: θ_N wild p = 0.05; θ_D wild p = 0.02.
> 3. The regime conclusion is restated. The data reject both simulated target and funding rules at the observed TFP dispersion.
> 4. The Pythia output-measurement effect is 0–13% and its sign varies. It is not "10–15%".
> 5. A strong-looking but implausible China × export-control IV result (overlap sample) was previously unreported. It is now reported.

**Notation.** This module follows `paper/notes/model_spec.md`, with one change: the output is benchmark log-odds, not loss.
- Output: y = logit((acc − chance)/(1 − chance)), the "log odds of above-chance success".
- Model: y = a + θ_N ln N + θ_D ln D + ω + e. Here θ_N and θ_D are the elasticities of the success odds with respect to parameters and tokens.
- On the odds scale, this is Cobb–Douglas, and ω is Hicks-neutral TFP (SYNTHESIS §4.E).
- θ_C(ray) = (θ_N + θ_D)/2 is the elasticity with respect to ln C along a proportional (D ∝ N) expansion.
- θ_C (C-only) is the slope from a regression on ln C = ln 6ND alone, which is what Ruan- and Mertens-style papers run.
- **Units:**
  - N = total parameters. For observational models this is the HF floating-point tensor count where available, else the reported size.
  - D = pretraining tokens processed, including epochs and parent-model tokens for continued pretraining.
  - C = 6ND.

---

## 1. Headline findings (exact numbers)

1. **The experimental benchmark: global log-linear fits.** Designed sweeps give θ_N ≈ 0.46–0.51 and θ_D ≈ 0.35–0.51 for HellaSwag log-odds (HC1 SEs):

   | Design | θ_N | θ_D | θ_C(ray) |
   |---|---|---|---|
   | OLMo ladder, 30 runs (190M–3B, 0.5–10× Chinchilla) | 0.481 (0.028) | 0.441 (0.029) | 0.461 (0.016) |
   | + OLMo-2 7B/13B targets | 0.464 (0.045) | 0.352 (0.044) | 0.408 (0.027) |
   | Gadre et al., N ≥ 0.1B (56 runs, corpus FE) | 0.513 (0.047) | 0.511 (0.027) | 0.512 (0.022) |
   | DataDecide, 525 final runs on the D = 100N ray, recipe FE | — | — | 0.464 (0.008) |

   - The composite output (HellaSwag, ARC-C, Winogrande) gives θ_C(ray) = 0.43–0.53.
   - Pythia (D fixed at 300B; 15 final models with released evals) identifies θ_N only. It is 0.68–0.74 on ARC-C and 0.59–0.67 on Winogrande (0- and 5-shot), using corrected parameter counts.

2. **The technology is not log-linear, and this matters for the comparison.**
   - HellaSwag log-odds are strongly concave in (ln N, ln D). The F-test for the quadratic terms is 352 in the OLMo ladder + OLMo-2 sample and 70 in Gadre et al. (both p < 0.001).
   - A quadratic surface fitted to 88 designed runs (R² = 0.994) implies that **θ_D falls and θ_N rises with tokens per parameter.**
     - At N = 1B: (θ_N, θ_D) = (0.39, 0.58) at D/N = 20; (0.59, 0.23) at D/N = 200; (0.67, 0.09) at D/N = 500.
     - At N = 7B, θ_N is lower: 0.15 at D/N = 50 and 0.35 at D/N = 500. Near 7B the surface rests on only 5 runs (three Gadre 6.9B runs at D/N = 20, plus OLMo-2 7B and 13B), so treat the 7B curve as indicative.
   - Observational base models are heavily over-trained: median D/N = 213, p75 = 597, p90 = 1,715.
   - So the global experimental slopes are the wrong target for a cross-lab regression. This is the Chinchilla structure (ε_D = βv/R falls with D/N) showing up in benchmark units.

3. **The naive (global) comparison looks like attenuation of the data elasticity.**
   - Sample: all 128 dense base models, 38 families, 22 developers. SEs are clustered by developer.
   - Pooled OLS: θ_N = 0.470 (0.047), θ_D = 0.257 (0.042).
   - Family FE: θ_N = 0.490 (0.063), θ_D = 0.189 (0.159).
   - Year FE: θ_D = 0.186 (0.075).
   - Against global experimental θ_D of 0.35–0.51, observational θ_D (0.19–0.26) is 27–63% lower. The ledger's Ho et al. contrast (β_data 0.04 vs 0.28–0.37) shows the same qualitative pattern.

4. **Design-matched LaLonde test (Table 6): no detectable bias in the compute elasticity for HellaSwag. It is less clean for ARC-C and Winogrande.**
   - Sample: 57 observational models inside the experimental (ln N, ln D) support (30 families, 19 developers).
   - Method: each estimator is applied both to the observed outputs and to counterfactual outputs generated by the experimental technology on the same models.

   | Estimator (HellaSwag) | Parameter | Obs. | Exp., same design | Bias (CR1 SE) | Wild-cluster p |
   |---|---|---|---|---|---|
   | Pooled OLS | θ_C (C-only) | 0.328 (0.037) | 0.329 (0.009) | **−0.001 (0.038)** | 0.99 |
   | 6 estimators (OLS, year, developer, family, lab×period FE, OP-style) | θ_C (C-only) | — | — | −0.029 to +0.037 | ≥ 0.36 (OLS and the 4 FE estimators; OP-style not tested) |
   | Pooled OLS | θ_N | 0.502 | 0.423 | +0.079 (0.091) | 0.37 |
   | Pooled OLS | θ_D | 0.223 | 0.273 | −0.050 (0.083) | 0.58 |
   | Family FE | θ_N | 0.648 | 0.445 | **+0.203 (0.086)** | 0.052 |
   | Family FE | θ_D | 0.087 (0.161) | 0.348 (0.018) | **−0.262 (0.162)** | 0.023 |
   | Developer FE / Lab×period FE | θ_N | — | — | +0.141 / +0.165 | 0.12 / 0.12 |

   - The OLS θ_C 95% CI is [−0.075, +0.074], i.e. about ±23% of the benchmark. The test can therefore rule out large biases but not biases below about 0.07.
   - Other outputs (OLS θ_C bias):
     - composite −0.018 (0.057);
     - ladder-only surface −0.016 (0.054);
     - ARC-C −0.098 (0.127);
     - Winogrande −0.160 (0.102) (Year FE −0.161 (0.096)).

     The ARC-C and Winogrande point estimates imply 18–31% attenuation of the compute elasticity, though neither is significant.
   - **Reading.** Once the design is matched, most of the apparent attenuation of θ_D in item 3 is the technology itself: over-trained models sit where θ_D is small. The experimental technology on the observational design gives θ_D = 0.27–0.36.
   - What remains in the N/D split is fragile (see Claim 3). It is clearest under family FE, where the design has little within-family D variation.

5. **Behavioural regime (model_spec Proposition 2): the data reject both the simulated capability-target rule and the simulated funding rule at the observed TFP dispersion.** They are consistent with a budget regime (compute roughly orthogonal to TFP) or with offsetting mechanisms.
   - **Semi-synthetic benchmark.** DataDecide's 25 real recipes act as "labs", with real evaluated outcomes and simulated choice rules. Truth θ_C = 0.464, and SD of recipe TFP = 0.10. Pooled-OLS bias by regime:

     | Regime | Pooled OLS bias | Recipe FE bias |
     |---|---|---|
     | Budget | 0.000 | −0.001 |
     | Funding | +0.044 | +0.010 |
     | Capability target | −0.183 (−40%) | not identified |
     | Release-only-if-good selection | −0.074 | −0.062 |

   - **Observational bias.**
     - Full hull: −0.001 (0.038).
     - "Clean" subsample (49 models): excludes Qwen1.5 (per-size D imputed), distilled, synthetic-data and code models. Bias is +0.045 (0.023). This is significant under CR1-normal inference (p = 0.03) but not under the wild-cluster bootstrap (p = 0.10, 16 clusters).
   - **Scaling to the observed dispersion.** Cross-family TFP dispersion is about 5× DataDecide's (SD of family effects 0.49 vs 0.10 logit). The simulated funding rule assigns sizes by TFP rank, so its bias scales roughly linearly with SD(ω): about +0.2 at the observed dispersion. The target rule attenuates even more than −0.18. The observed CIs exclude both.
   - **Caveats.**
     - This is a stylized comparison. The simulated target lab releases one model, while real families release size menus.
     - The test is joint with "same technology up to Hicks-neutral shifts".
     - Offsetting funding (+) and release selection (−) cannot be excluded.
   - **Mundlak.** Between − within for ln C is −0.100 (0.046) on the full sample. On the hull, the experimental technology alone implies a within-family C-only slope 0.080 above the pooled one (0.409 vs 0.329), so this gap is a design effect, not evidence of target behaviour.

6. **Standard IO remedies are either uninformative or implausible here.**
   - **Frontier FLOP/$ IV is weak.** The instrument is the FLOP/s per $ of the best data-centre accelerator at the training date.
     - First-stage F = 2.3 on the full sample and 0.9 on the hull. It is 1.7 with developer FE and 0.002 with developer FE + trend.
     - On the overlap sample: 0.001, 0.24 and 3.8 respectively.
     - Anderson–Rubin 95% sets are unbounded.
     - The instrument takes only 4 distinct values in the sample: it is a step function of calendar time, collinear with vintage.
   - **Own-hardware FLOP/$ IV is "strong" but is effectively a vintage dummy.**
     - Full sample: F = 10.7, n = 24 from 8 developers, θ_C = 0.276 (0.083), AR [−0.055, 0.425]. Overlap: F = 19.2, n = 19, θ_C = 0.17 (0.17).
     - It takes 3 values. All identifying variation comes from 6 models: 2 AI2 models on MI250X, and Llama-3 8B/70B, StarCoder2-7B and MPT-30B on H100.
     - After deduplication, fewer than 12 hull models have it, so it is not in Table 6.
   - **China × post-export-control IV (developer + year FE).**
     - Full sample: F = 2.4.
     - **Overlap sample (N ≤ 14B): F = 10.6, θ_C = 0.90 (0.25), AR [0.49, 2.23].** This is 1.7–2.3× the OLS and FE estimates on the same sample (0.40–0.54).
     - The exclusion restriction (Chinese labs' post-Oct-2023 models differ from their earlier models only through compute) is implausible, because recipes and data changed at the same time. Treat this as a failed instrument, not as evidence.
   - **Dynamic panel is infeasible and fragile.**
     - There are only 30 consecutive-generation pairs in 10 product lines, and 6 cells with three generations, so Arellano–Bond / Blundell–Bond are not estimable.
     - FD: θ_N = −0.001 (0.486), θ_D = 0.065 (0.124).
     - ACF-style (AR(1) ω, N predetermined, D flexible): θ_N = 0.56 [−0.87, 0.79], θ_D = 0.75 [−0.36, 1.25], with profiled ρ = 1.17 (explosive).
     - The review's data fix moved these a lot (before: FD θ_N 0.16, ACF θ_D 0.11, ρ = 0.75). They are uninformative.
   - **EIV corrections are negligible.**
     - On the full sample, independent Epoch compute estimates deviate from 6ND with SD 0.195 in ln units (14 models). Reliability is 0.995 pooled and 0.988 within families.
     - On the overlap sample only 6 such models exist (SD 0.025).
     - Epoch's "Confident" class (±3×, σ = 0.67) over-corrects under family FE (0.86 vs benchmark 0.41 on the hull).
     - Exact (HF float-tensor) vs reported parameter counts differ by +2.6% on average (SD 6.3%, 86 models).
   - **Reverse (Nerlove) regression** gives 0.56 vs 0.35 on the hull. Forward and reverse bracket the truth, and the truth sits at the forward end.

7. **Selection checks change little.**
   - Notable-only (19 models, 7 developers): θ_C = 0.281 (0.059), vs 0.371 (0.023) for all.
   - An OP-style control for the notability propensity gives θ_C = 0.376. Matched bias −0.015 (0.051).
   - Selection into *release* is unobservable in public data.

8. **Lab TFP dispersion is large in compute-equivalent units.** It looks Syverson-sized only after converting to a loss index, and that conversion depends on the choice of output units. Ratios are compute-equivalent 90/10, with developer-bootstrap 90% intervals; θ is held fixed in the "experimental θ" intervals.

   | Measure (HellaSwag) | 90/10 ratio |
   |---|---|
   | Family effects, experimental θ_C = 0.329, all 128 models / 38 families | **23.7× [9.1, 197]** |
   | ... netting inputs with the same θ_C (ω_f = family mean of y − θ_C ln C) | 19.9× [7.6, 161] |
   | ... dropping distilled (Gemma-2 2B/9B) and synthetic-data (Phi, SmolLM) families | 18.3× (17.6× netted by θ_C) |
   | ... also dropping code families | **7.0× [4.7, 168]** (9.1× [5.2, 11.6] netted by θ_C) |
   | Within-developer model residuals (Mertens design: developer + year FE), own θ_C = 0.43 | **11.1× [4.4, 18.8]** |
   | Within-developer model residuals, experimental θ | 22.4× [6.3, 34.6] |

   - Mertens et al. report 41× within developer (MMLU-Pro).
   - **Like-for-like comparison with Syverson.** With returns to scale near one, Syverson's manufacturing 90/10 of 1.92 is also an input-equivalent ratio. So LLM family dispersion, at **7–24× in compute units, is roughly 3–5 times larger in log terms than manufacturing**.
   - **In reducible-loss units** (R^γ with an assumed Chinchilla γ = 0.178), the family figure is 1.76 → 1.41 after the exclusions (1.71 → 1.48 netted by θ_C), and the within-developer figure is 1.54–1.74. These are close to Syverson's 1.92. This conversion is an output-cardinalization choice, not an estimate.
   - Phi and SmolLM (synthetic data) rank 1st and 3rd of 38 families; distilled Gemma-2 ranks 10th. This is the gross-output contamination.
   - **Frontier sample (Epoch ECI on Epoch compute, 104 models, 23 developers):**
     - θ_C = 4.76 ECI points per ln C (1.02).
     - Within-developer 90/10 = 8.1× [3.4, 19.2]; 3.5× [2.6, 5.6] for "Confident"-compute models only.

9. **Output measurement moves elasticities by 0–13%, and the sign varies.**
   - Comparison: the same 11 Pythia models, with corrected N, measured by the leaderboard vs by the Pythia team's own evals.
   - ARC-C θ_N: 0.785 (0.064) with leaderboard 25-shot evals, vs 0.692 (0-shot) and 0.754 (5-shot) with Pythia's evals.
   - Winogrande θ_N: 0.677 (0.064) with the leaderboard, vs 0.682 / 0.734 with Pythia's evals.

**Sanity checks against the ledger and literature.**
- Mertens' 0.789 per log10 FLOP (MMLU-Pro logit) is 0.343 per ln C. Our θ_C (HellaSwag logit per ln C) is 0.33–0.46, the same order.
- The loss-unit dispersion (1.4–1.8) matches the ledger's 41^γ = 1.78–1.94 convention and sits below Syverson's 1.92. See item 8 for why the input-unit comparison is the like-for-like one.

---

## 2. Methods

### 2.1 Observational panel (`panel.py` → `data/processed/m4_observational/obs_panel.csv`)

**Sources.**
- ObsScaling `base_llm_benchmark_eval.csv`: 148 base models, Open LLM Leaderboard v1 metrics.
- Sloth `data_v2.csv`: base models absent from ObsScaling (Pythia standard, Yi-9B, RecurrentGemma), plus fills for missing scores.
  - **Review fix:** Sloth's pythia-70m/1b/1.4b/2.8b/6.9b rows are exact copies of ObsScaling's deduped scores (all six benchmarks, to 6 decimals). They are dropped as duplicates; 5 of the 10 additions remain.
- Hugging Face model API: metadata for 158 ids.
  - **Review fix:** parameter counts use floating-point tensors only. `safetensors.total` also counts the U8 causal-mask buffers of GPT-NeoX/GPT-Neo checkpoints: +36% for Pythia-70m, +33% for 160m, +23% for 410m, +20% for GPT-Neo-125M; 17 models were affected.
  - Upload dates also come from this source.
- Epoch `all_ai_models.csv`, via a hand-built crosswalk: 102 of 153 rows matched, 90 of the 128 main-sample models. Fields used: publication date, confidence, estimation method, hardware, notability, C.
- Epoch `ml_hardware.csv` (for the instruments).

**Outputs.** The same harness and prompts apply to every model:
- HellaSwag 10-shot acc_norm (chance 0.25);
- ARC-C 25-shot acc_norm (0.25);
- Winogrande 5-shot (0.5);
- MMLU 5-shot, GSM8K, TruthfulQA (these three only for PC1).

The composite is the logit of the mean chance-adjusted accuracy of the three core tasks. Chance-adjusted accuracy is clipped to [0.01, 0.99].

**Corrections to source data** (all in `panel.FIXES`, each with its source):
- BTLM D 627T → 0.627T.
- open_llama_7b_v2 N 13B → 7B.
- OpenLLaMA-13B D 0.6T → 1.0T.
- Gemma-2B D 6T → 3T.
- Qwen2 D filled (7T; 0.5B: 12T).
- InternLM2-20B D = 2.6T.
- RWKV D = 0.33T.
- Llama-3 ARC-C filled from Sloth.
- MoE models use active N and are excluded from the main sample.

**Samples.**
- **Main:** dense transformers with N, D and all three core outputs: 128 models, 38 families, 22 developers, 2021-03 to 2024-07.
- **Overlap:** N ≤ 14B and D ≤ 5T, 90 models.
- **Hull:** models inside the convex hull of the experimental (ln N, ln D) points, 57 models. This is Dehejia–Wahba-style common-support trimming.

**Release date.** Epoch where matched, else a manual override (HF migrations), else the HF upload date. In the main sample: 90 Epoch, 12 manual, 26 HF upload.

**Family, developer, product line and generation** are coded in `panel.FAM`. Families are ObsScaling families (product line × generation). For the dynamic panel, the unit is (line, size tier) and t = generation.

**Flags.**
- Distilled: Gemma-2 2B/9B.
- Synthetic data: Phi-1.5/2, SmolLM.
- Code-specialised: CodeLlama, StarCoder(2), DeepSeek-Coder.
- Continued pretraining: CodeLlama, Yi-1.5, Yi-200K, Yi-9B.

**Identification audit** (Table `design_audit`):
- Within-family SD of ln D is 0.42, vs 1.29 for ln N.
- Only 18 families (66 models) have any within-family D variation, and Cerebras-GPT has D = 20N exactly.
- Corr(ln N, ln D) = 0.46.
- C = 6ND in 100% of rows.

### 2.2 Experimental designs (`experiments.py`)

- **OLMo ladder:** final checkpoints of 30 runs, plus the OLMo-2 7B (925k steps, 3.9T tokens) and 13B (596k steps, 5.0T) targets from the same repository.
  - N for ladder runs = OLMo's logged training FLOPs/(6D). OLMo's FLOP counter charges 2·N(excl. input embedding) + 4·N_total + 12·L·d·seq per token. This "N" therefore differs from the total parameter count by a few percent; the exact configs were not verified.
  - OLMo-2 N = HF float-tensor counts.
  - Outputs: OLMES 5-shot "rc" len-norm accuracies.
- **Gadre et al.:** 104 models with LLM-foundry evals (HellaSwag 10-shot, ARC-C 10-shot, Winogrande 0-shot). `params` includes embeddings. Corpus FE.
- **Pythia:** lm-eval-harness JSONs per checkpoint (0/5-shot ARC-C acc_norm, Winogrande). There are 15 size×dedup models; pythia-1b standard has no released evals.
  - N = HF float-tensor counts.
  - θ_D uses only checkpoints ≥ 27B tokens and is confounded by the cosine schedule.
- **DataDecide:** 1,410,750 eval rows, filtered to 5 tasks: acc_per_char for ARC/HellaSwag (≈ lm-eval acc_norm, but char- rather than byte-normalised) and acc_raw for Winogrande.
  - Final checkpoints with N ≥ 60M.
  - D = 100N, so only θ_C is identified.
  - Recipe FE; SEs clustered by recipe.

### 2.3 Design-matched LaLonde test (`lalonde.design_matched`, Table 6)

- **Surface.** f̂ is a quadratic (translog-in-odds) surface in (ln N, ln D) with dataset intercepts. It is fitted to 32 OLMo ladder/OLMo-2 runs and 56 Gadre runs with N ≥ 0.1B.
- **Counterfactual outputs.** For each observational model inside the hull, y*_i = f̂(ln N_i, ln D_i).
- **Comparison.** Every estimator is run on y and on y* over the same models with the same fixed effects and controls. Bias = est(y) − est(y*).
- **Benchmark uncertainty.** 300 bootstrap refits of the surface, with runs resampled within design; all 300 succeeded.
- **Observational uncertainty.** Developer-clustered CR1 SEs. Because there are only 19 clusters, the review added restricted wild-cluster bootstrap-t p-values (Webb six-point weights, 999 draws) for every bias, holding the benchmark fixed: `table6_lalonde_wildboot.csv` and `table6_lalonde_clean_wildboot.csv`.
- **What the test measures.** It tests the joint null "observational technology = experimental technology up to Hicks-neutral shifts, output formats map one-to-one, and no transmission or selection bias".
- **Robustness:**
  - ladder-only surface (39 models);
  - composite, ARC-C and Winogrande outputs;
  - a "clean" subsample (49 models).

### 2.4 Global estimators (`observational.py`, Table 6b)

All estimators use developer-clustered CR1 SEs and are run on the overlap and full samples:
- OLS, year FE, developer FE, family FE, Mundlak (family means), developer×year FE.
- FD across generations.
- ACF-style GMM: the quasi-difference y_t − ρ y_{t−1} with instruments (1, n_t, n_{t−1}, d_{t−1}), ρ profiled on a grid over [−0.5, 1.5], and a product-line cluster bootstrap.
- IV via 2SLS, with cluster-robust first-stage F and Anderson–Rubin sets by grid inversion. Instruments:
  - frontier FLOP/s per $ (≥ $5k accelerators released ≥ 6 months earlier);
  - own-hardware FLOP/s per $;
  - China × post-Oct-2023.
- EIV reliability correction.
- Nerlove reverse regression.
- Notable-only subsample.
- OP-style control: a quadratic in the logit propensity of Epoch notability on (ln N, ln D, date).

### 2.5 Semi-synthetic regimes (`semisynth.py`)

- Setup: DataDecide finals with N ≥ 60M (7 sizes × 25 recipes × 3 seeds). ω_r = recipe FE.
- Each replication draws one seed per chosen (recipe, size). 400 replications.
- Choice rules:
  - budget: 3 random sizes;
  - funding: a 3-size window whose position rises with the lab's ω *rank*;
  - target: the smallest size with y ≥ the median lab's y at the middle size; one model per lab, so FE is not identified;
  - selection: 3 random sizes, released only if y ≥ the 40th percentile.

### 2.6 TFP dispersion (`tfp.py`)

- **Family effects:** ω_f from y ~ ln N + ln D + family FE.
  - Review addition: a variant nets inputs with the benchmark θ_C instead (ω_f = family mean of y − θ_C ln C). This way the netting and the conversion use one technology.
- **Mertens design:** y ~ ln C + developer FE + year FE, then residual dispersion. Review fix: single-model developers, whose residuals are identically 0, are excluded; 125 of 128 residuals remain.
- **Compute-equivalent 90/10:** exp((ω_90 − ω_10)/θ_C), with θ_C either the regression's own or the design-matched experimental value (0.329 HellaSwag; 0.388 composite).
- **Inference:** developer-cluster bootstrap (300 draws), reporting 90% intervals. In the "experimental θ" columns θ_C is held fixed, so those intervals omit benchmark uncertainty.
- **Loss-unit crosswalk:** R^γ, with γ = 0.178 (Besiroglu) or 0.155 (Hoffmann), both taken from the ledger (an ASSUMED Chinchilla technology).
- **PC1:** the first PC of 6 standardized benchmark logits explains 73% of variance (TruthfulQA loading 0.20; the others 0.43–0.44).
- **ECI:** `eci_scores.csv` joined by exact name to Epoch models (104 with C); developer = first listed organization. Some developer-bootstrap draws have collinear developer and year dummies. statsmodels then uses a pseudo-inverse; θ_C and the residuals are unaffected.

---

## 3. Table and figure inventory

Tables are in `output/tables/`; figures are in `output/figures/`, each as .pdf and .png.

| File | Content |
|---|---|
| `m4_observational_table6_lalonde.{csv,tex}` | **Paper Table 6.** Design-matched LaLonde test, HellaSwag: 6 estimators × (θ_N, θ_D, θ_C), plus compute-only rows (IV, EIV, reverse). Obs / Exp / Bias with SEs. |
| `m4_observational_table6_lalonde_wildboot.csv`, `..._clean_wildboot.csv` | **New (review).** Wild-cluster bootstrap p-values for every Table 6 bias (main and clean samples). |
| `m4_observational_table6_lalonde_{core,arc,wino,ladderonly,clean}.{csv,tex}` | Table 6 robustness: composite; ARC-C; Winogrande; ladder-only surface; clean subsample. |
| `m4_observational_table6b_global_{overlap,main}.{csv,tex}` (+`_long.csv`) | Appendix: all estimators on the overlap/full samples next to the global experimental fits. **Not design-matched.** |
| `m4_observational_exp_benchmarks.{csv,tex}` | Experimental elasticities by design and benchmark (global log-linear fits). |
| `m4_observational_curvature.csv` | Tests of log-linearity in the experimental designs. |
| `m4_observational_design_audit.{csv,tex}` | Identifying variation in the observational panel; also exact-vs-reported N statistics. |
| `m4_observational_iv_diagnostics_{main,overlap}.csv` | First stages (F), 2SLS, AR sets for all instruments. |
| `m4_observational_regimes_semisynthetic.csv` | Semi-synthetic bias by behavioural regime (DataDecide). |
| `m4_observational_tfp_table.tex`, `m4_observational_tfp_dispersion.csv`, `m4_observational_eci_dispersion.csv` | TFP dispersion: compute-equivalent 90/10, the θ_C-netted variant, loss units, exclusions, ECI frontier sample. |
| `m4_observational_pythia_output_check.csv` | Output comparability: the same 11 Pythia models, leaderboard vs the experimenters' evals. |
| `m4_observational_fig5_lalonde` | **Paper Figure 5 (LaLonde plot).** Observational estimate (blue) vs experimental technology on the same design (orange), 95% CIs (CR1 for Obs.), three panels. Gray band = range of global experimental fits. IV rows are omitted (F < 1). |
| `m4_observational_support` | (ln N, ln D) support: observational models, experimental runs, convex hull, iso-D/N lines. |
| `m4_observational_elasticity_by_M` | Local experimental elasticities vs tokens per parameter at N = 1B and 7B (inside the support only). |
| `m4_observational_tfp_families` | Family effects in log10 compute-equivalents (general / code / distilled-synthetic). |
| `m4_observational_regimes` | Semi-synthetic θ_C by regime and estimator vs experimental truth. |
| `data/processed/m4_observational/headline.json` | All headline numbers in machine-readable form. |

---

## 4. Claims for the paper

1. **In cross-lab data, returns to compute show no detectable bias against the experimental benchmark once the design is matched (HellaSwag).**
   - **Evidence** (`table6_lalonde.csv`, `_wildboot.csv`):
     - Pooled OLS θ_C = 0.328 (0.037) vs 0.329 (0.009) from the experimental technology on the same 57 models. Bias −0.001 (0.038), wild p = 0.99.
     - Six estimators: biases −0.03 to +0.04.
     - Composite −0.018 (0.057). Ladder-only surface −0.016 (0.054).
   - **Caveats:**
     - The CI (±0.075) cannot exclude biases of ±20%.
     - ARC-C and Winogrande point estimates are −0.10 (0.13) and −0.16 (0.10), i.e. 18–31% attenuation.
     - Coverage is limited to ≤ 9B models inside the experimental support.
     - The benchmark is recipe-specific (AI2/OpenLM).
     - The test is joint with the output-format mapping (claim 9).
     - Clean subsample: +0.045 (0.023), wild p = 0.10.

2. **The observational "data elasticity gap" is mostly a design effect, not an endogeneity bias.**
   - **Evidence:**
     - Global experimental θ_D = 0.35–0.51 (`exp_benchmarks.csv`). The experimental technology evaluated on the observational design gives 0.27 (OLS) to 0.36 (lab×period FE) (`table6_lalonde.csv`).
     - Local θ_D at N = 1B falls from 0.58 (D/N = 20) to 0.09 (D/N = 500) (`elasticity_by_M`), and the median observational D/N is 213.
   - **Caveats:**
     - This rests on a quadratic surface. At the high-D/N edge of the support it turns slightly negative (θ_D ≈ −0.04 at 7B, D/N = 500), which is an approximation artifact.
     - Near 7B only 5 experimental runs anchor it.
     - The analogy to Ho et al.'s β_data is suggestive only: their output is perplexity, and their model drops E.

3. **(Weakened by the review.) The observational split between N and D is unreliable, most clearly under family fixed effects.**
   - **Evidence:**
     - Family FE on HellaSwag: θ_N bias +0.203 (0.086), wild p = 0.052; θ_D bias −0.262 (0.162), wild p = 0.023.
     - With the ladder+Gadre surface, θ_D biases are negative for every estimator and output.
     - Under family FE, the design has little within-family D variation: within SD of ln D is 0.42 vs 1.29 for ln N, and only 18 families have any D variation.
   - **What does NOT hold:**
     - "θ_N overstated and θ_D understated in every specification". With the ladder-only surface, developer-FE and OP-style θ_N biases are negative (−0.11, −0.09) and the developer-FE θ_D bias is positive (+0.10).
     - For ARC-C and Winogrande, OLS and year-FE θ_N biases are negative.
     - Pooled-OLS N/D biases are insignificant (wild p = 0.37 and 0.58).
   - **Caveats:**
     - We cannot separate measurement error in D, factor-biased (data-quality) recipe changes that coincide with D changes within families, and distillation.
     - In the clean subsample the OLS θ_D bias vanishes (−0.013 (0.062)); the FE θ_D bias remains (−0.344 (0.223), wild p = 0.04).

4. **Compute-only regressions impose a false restriction, in both worlds.**
   - **Evidence:** pooled OLS θ_N − θ_D = 0.213 (0.079) on the full sample (`table6b_global_main_long.csv`). The experimental technology at the hull design also implies θ_N > θ_D (0.423 vs 0.273).
   - **Caveat:** this is a feature of over-trained designs. It is consistent with the inference wedge w = ε_N/ε_D > 1 of module H2, but it is not a test of it.

5. **(Restated.) The cross-lab data are inconsistent with strong compute-on-TFP choice rules of either sign** (model_spec Proposition 2).
   - **Evidence:**
     - The semi-synthetic target rule attenuates θ_C by −0.183 (−40%), and the funding rule inflates it by +0.044, at DataDecide's recipe dispersion (SD 0.10).
     - Observed cross-family dispersion is about 5× larger (SD 0.49). Under the simulated rules, the implied biases are ≤ −0.18 and ≈ +0.2.
     - The observational CIs are [−0.075, +0.074] (hull) and [0.000, +0.090] (clean).
   - **Interpretation:** a budget regime (compute roughly orthogonal to TFP) or offsetting mechanisms fit best.
   - **Caveats:**
     - The simulated rules are stylized (one model per target lab; rank-based funding).
     - The test is joint with common technology.
     - Offsetting funding (+) and release selection (−) cannot be excluded, and release selection is unobservable in public data.

6. **Standard IO remedies are not usable on public cross-lab LLM data.**
   - **Evidence** (`iv_diagnostics_*.csv`, `table6b_*`):
     - The frontier FLOP/$ IV has F ≤ 3.8 in every specification and unbounded AR sets.
     - Own-hardware FLOP/$ is strong (F = 10.7–19.2) but has 3 values identified by 6 models, making it effectively a vintage dummy.
     - China × export-control (overlap sample) is strong (F = 10.6) but gives an implausible θ_C = 0.90 (0.25). The exclusion is not credible.
     - There are 30 generation pairs and 6 triples, so AB/BB is infeasible. FD/ACF estimates are imprecise and moved materially under a small data correction (ACF ρ went from 0.75 to 1.17).
     - EIV reliability is ≥ 0.988.
   - **Caveat:** the failure is informative about the data, not about the estimators.

7. **Lab TFP dispersion is about 7–24× in compute-equivalent units, far above manufacturing's roughly 1.9× input-equivalent. In a reducible-loss index it is 1.4–1.8.**
   - **Evidence** (`tfp_table.tex`, `tfp_dispersion.csv`):
     - Family 90/10 is 23.7× [9.1, 197] with the experimental θ, 19.9× when inputs are netted with the same θ, and 7.0–9.1× after dropping code, distilled and synthetic families.
     - Within-developer residual 90/10 is 11.1× [4.4, 18.8] (Mertens design, own θ), vs Mertens' 41×.
   - **Caveats:**
     - The loss-unit numbers apply an assumed Chinchilla γ to benchmark log-odds. They depend on the output index and are illustrative; only the compute-equivalent comparison is unit-free.
     - Upper bootstrap bounds are very wide.
     - Intervals hold θ fixed.
     - Code families dominate the bottom decile.

8. **Gross-output contamination is visible.**
   - **Evidence:** the synthetic-data families (Phi, SmolLM) rank 1st and 3rd of 38 in family TFP; distilled Gemma-2 ranks 10th (`tfp_families`). Dropping distilled/synthetic families lowers the family 90/10 from 23.7× to 18.3×.
   - **Caveats:**
     - No teacher compute is available, so no GNR correction was estimated.
     - Minitron/Nemotron are not in the base-model panel.

9. **(Revised.) Output measurement moves elasticities by up to about 13%, with a sign that varies by task.**
   - **Evidence** (`pythia_output_check.csv`; same 11 Pythia models, corrected N):
     - ARC-C θ_N is 0.785 (0.064) with leaderboard 25-shot evals, vs 0.692 (0-shot) and 0.754 (5-shot) with the Pythia team's evals.
     - Winogrande: 0.677 vs 0.682 / 0.734.
   - **Caveat:** no HellaSwag in the released Pythia evals, and n = 11.

---

## 5. Robustness and failures

- **Failed or uninformative estimators (reported, not hidden):**
  - IV with the frontier FLOP/$ shifter (weak, F ≤ 3.8).
  - China × export-control IV: weak on the full sample (F = 2.4); strong on the overlap sample (F = 10.6) but implausible (θ_C = 0.90, AR [0.49, 2.23]).
  - Arellano–Bond / Blundell–Bond: not estimable, only 6 three-generation cells.
  - FD and ACF-style estimates on generations are imprecise and unstable (ACF ρ = 1.17 on the corrected data).
  - The ACF-style "proxy = reported training compute" was infeasible because C = 6ND by construction in every row.
  - Notable-only has 19 models.
- **Surface choice matters for precision, and for the N/D signs.** With the ladder-only surface (39 models), the OLS θ_C bias is −0.016 (0.054) and the θ_D bias −0.010 (0.131). The developer-FE θ_N/θ_D biases reverse sign (−0.11/+0.10).
- **Few-cluster inference.** CR1 with 16–19 developer clusters is anti-conservative for some rows. Two cases matter:
  - Developer-FE and lab×period θ_N biases are not significant under the wild bootstrap (p = 0.12), although CR1-normal gives p = 0.06–0.08.
  - The clean-sample θ_C bias has wild p = 0.10, vs CR1-normal 0.03.
- **Negative local θ_D.** The quadratic surface gives mildly negative local θ_D at the high-D/N edge (7B, D/N ≥ 500). Treat local elasticities there as ≈ 0.
- **Composite and floor effects.** Many models below about 0.3B sit at chance on ARC-C and Winogrande, where logits are clipped. HellaSwag is the cleanest output and is the headline.
- **EIV sensitivity.** Epoch's "Confident" class (σ = 0.67 in ln C) implies within-family reliability 0.86. It pushes the family-FE θ_C to 0.51 (full) or 0.86 (hull). Given that independent compute estimates deviate by only 0.195, this is an over-correction.
- **TFP dispersion is fragile.**
  - The family 90/10 depends on sample composition.
  - PC1-based family dispersion is much larger (336× with its own θ). It is driven by MMLU/GSM8K, where test-task training favours newer families, so it is a TFPR-like artifact.
  - The ECI developer-level 90/10 (934×) is dominated by developers with "Speculative" compute. It falls to 118× with "Confident" compute only.
- **Data corrections made by the review** (numbers above are post-fix):
  - 5 duplicate Pythia rows removed: main sample 133 → 128, hull 63 → 57.
  - Mask-buffer-inflated N corrected for 17 GPT-NeoX/GPT-Neo models (observational and Pythia experimental).
  - The headline OLS θ_C bias moved from +0.003 to −0.001. The ACF/FD estimates moved materially.
- **Runtime.** 25–32 minutes with B = 300, depending on load. The TFP bootstrap is the bottleneck. `--fast` uses B = 100.

---

## 6. Open issues

1. **Scale overlap.** The matched test covers 57/128 models, all ≤ 9B. Llama-3/3.1, Qwen2 7B/72B, Gemma-2 9B/27B and all ≥ 20B models lie outside the experimental support. A larger experimental ladder (e.g. open-sci-ref, Marin, Proteus-2k) would extend the test.
2. **Recipe heterogeneity vs bias.** "Truth" is the AI2/OpenLM technology. Non-neutral technology differences across labs (data quality ψ_D) are observationally equivalent to transmission bias in this test. DataDecide's recipe × scale design could test neutrality directly.
3. **Output-format mapping.** Experimental outputs (OLMES 5-shot cloze; LLM-foundry) differ from the leaderboard formats. A common re-evaluation of a few observational models in the OLMES harness would pin this down.
4. **Mertens et al. 809-model data** was not used (Google Drive access UNVERIFIED).
5. **Release dates** come from HF uploads for 26 main-sample models. Year FE and lab × period FE depend on these.
6. **Instruments.** Model-specific hardware and cluster-access shocks (e.g. Epoch `gpu_clusters.csv`, not downloaded) could give a stronger cost shifter.
7. **Unverified parameter counts.**
   - XGLM-4.5B's HF count (5.08B vs a nominal 4.5B) may double-count the tied embedding; it was not verified.
   - OLMo ladder N from logged FLOPs is within a few percent of total parameters, but the configs were not checked.
8. **No bug found in `sl.py`.** This module does not call it.
9. **Citations.**
   - Keys used, all in `lit/references.bib`: cameron2008bootstrap (added to the list by the review), ruan2024observational, maiapolo2024sloth, mertens2026secret, ho2024algorithmic, whitfill2025note, konig2026validity, olley1996dynamics, ackerberg2015identification, levinsohn2003estimating, blundell1998initial, blundell2000gmm, marschak1944random, mundlak1961empirical, syverson2004product, syverson2011determines, nerlove1963returns, griliches1998production, griliches1986errors, gandhi2020identification, bhagia2024establishing, gadre2024language, magnusson2025datadecide, biderman2023pythia, teamolmo2024olmo, epochai2026data, epochai2026capabilities, zellner1966specification, hoffmann2022training, besiroglu2024chinchilla, dominguezolmedo2025training, owen2024predictable, ho2025rosetta.
   - Entries in `lit/bib/extra_m4_observational.bib`: lalonde1986evaluating, dehejia1999causal, anderson1949estimation, staiger1997instrumental, arellano1991some, and webb2023reworking (Webb six-point weights for the wild-cluster bootstrap; added by the review, verified via Crossref).
