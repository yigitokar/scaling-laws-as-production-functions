# Memo — module m3_wedge: over-training reveals anticipated inference demand (headline H2)

Module owner: m3_wedge (Claude). Date: 2026-09-24. Entry point: `code/analysis/m3_wedge/run.py` (about 6 seconds on CPU; deterministic, no random draws of its own: bands use the saved bootstrap draws of modules m1 and m2). One-time data pull: `bash code/data/download_m3_wedge.sh` (Hugging Face API and model cards, OpenRouter endpoints, LMArena; about 1 minute).

Notation follows `paper/notes/model_spec.md`: u = A N^−α, v = B D^−β, w = ε_N/ε_D = αu/(βv), M = D/N, C = 6ND, a = β/(α+β), γ = αβ/(α+β), σ* = 2/(2+α+β). Proposition 4: with lifetime cost 6ND + 2NT the FOC is w = 1 + T/(3D), so **T/D = 3(w − 1)** and w is lifetime/training compute. Cost efficiency CE = C_min(L)/(6ND) = [((α+βw)/(α+β))^{1/γ} w^{−1/α}]^{−1} (m7 Lemma A5). Under the Chinchilla form ln w = (α+β) ln G − α ln N + β ln D with G = (αA/(βB))^{1/(α+β)}, equivalently ln w = ((α+β)/2)·ln(M/M*(C)) (m7): only (α, β, G) matter.

Labels: **Estimated** = computed here. **Upstream** = technology estimated by m1/m2 (registries and bootstrap draws). **Assumed** = maintained assumption. **Literature** = number taken from a paper (bib key).

**[Rev]** marks text changed by the independent review (2026-09-24; details in `output/memos/m3_wedge_review.md`). The review fixed six Epoch token counts, merged developer-name variants into single clusters, and applied the 10²¹-FLOP production-scale threshold to both samples. It also corrected several statements. The Sample-B headline numbers (H1, H2, H5 flagship results, H6 downloads) are unchanged.

"Reference technology" = m1's Huber refit on the Besiroglu n = 240 Chinchilla sample (α = 0.347, β = 0.367, M*(10²¹) = 21.4), with its 400 pairs-bootstrap draws. Besiroglu's published values give wedges within 1–2% of it (Chinchilla-70B: 1.031 vs 1.040).

---------------------------------------------------------------------------------------------------

## 1. Headline findings

### H1. Open-weight base models are over-trained, and the verdict survives technology uncertainty for about four in five of them
Sample B: 173 general-purpose open-weight pretrained models, 2021–2025, documented token counts, exact parameter counts. File `output/tables/m3_wedge_models.csv`.
- **Reference technology.** Median ŵ = **3.19**, so median T/D = 6.6. 95% of models have ŵ > 1; 75% have ŵ > 2.
- **Bootstrap.** For 87% of models the lower 95% bootstrap bound is above 1 (pairs); 80% with m1's 9-cluster bootstrap.
- **Partial-identification band.** The band is the union of 95% intervals (or points) over six technologies: Chinchilla refit, Besiroglu published, Hoffmann A3 (TeX), Farseer non-embedding N, Farseer incl.-embedding N, Gadre et al. RefinedWeb. For Meta and AI2 models, their own IsoFLOP technologies are added. **The whole band lies above 1 for 79% of models** (136/173; median band [1.77, 6.92]). [Rev] It was 78% (135) before the OLMo-ladder fix for OLMoE: the builder applied that technology to OLMoE's 6.9B total rather than its 1.3B active parameters, so OLMoE's band fell from [0.45, 12.0] to [1.05, 12.0].
- **Dispersion across technologies** (median ŵ over Sample B, from `m3_wedge_models.csv`):

  | Technology | Median ŵ |
  |---|---|
  | Hoffmann A3 | 1.91 |
  | Meta law (A2/A1) | 2.12 |
  | Farseer incl. emb. | 2.41 |
  | Farseer non-emb. | 2.67 |
  | Meta A3 | 3.08 |
  | Besiroglu | 3.16 |
  | Chinchilla refit | 3.19 |
  | Farseer q-family | 3.85 |
  | Gadre RefinedWeb | 4.03 |

  Estimator choice on the same Chinchilla data matters too: NLS in levels gives 4.11 and keeping the 5 gross outliers (n = 245) gives 4.78.
- **Extrapolation [Rev].**
  - *In M.* 55% have M outside the Chinchilla design (M ≤ 341). 29% have M outside Farseer's (incl. emb., ≤ 1,227). With non-embedding N (≤ 2,570) it is 21% of the 146 models where that technology is defined. The builder's 17% divided by all 173.
  - *In compute.* 75% of Sample B exceed the Chinchilla design's largest budget (1.3×10²²), which is the largest budget of any band technology. The shares are 81% for Gadre's design and 85% for Farseer's (incl. emb.). The builder's "every model lies beyond every design in compute" is false: the small models (Pythia-70M–410M, Cerebras-GPT, SmolLM-135M, OPT-125M/350M, …) are inside the designs.

### H2. Canonical models
`output/tables/m3_wedge_table5.csv|.tex`.

| Model | M | ŵ reference [95% CI] | T/D | T (tokens) | PI band | ŵ, Meta law |
|---|---|---|---|---|---|---|
| Llama 3 8B | 1,868 | **5.27** [3.97, 8.12] | 12.8 [8.9, 21.4] | 1.9×10¹⁴ | [2.48, 12.0] | 3.09 (T/D = 6.3) |
| Llama 3/3.1 70B | 213 | 2.48 [1.81, 3.81] | 4.4 | 6.7×10¹³ | [1.02, 7.55] | 1.63 |
| **Llama 3.1 405B (flagship)** | 38 | 1.37 [0.99, 2.22] | 1.1 | | [0.42, 5.73] | **0.98** |
| Llama 2 70B (flagship) | 29 | 1.19 [0.93, 1.67] | | | | 0.98 |
| Gemma 3 27B | 518 | 3.37 [2.52, 5.20] | | | | |
| DeepSeek-V3, 37B active | 400 | 3.09 [2.29, 4.73] | | | | |
| Qwen2.5-72B | 248 | 2.62 [1.91, 4.07] | | | | |
| Qwen3-0.6B | 60,398 | 17.9 | 51 | | | |
| SmolLM2-135M | 14,868 | 10.4 | | | | |

- **CE.** Farrell cost efficiency is 0.18 for Llama 3 8B, i.e. 5.6× the training-only minimum compute for its loss, and 0.93 for the 405B.
- **Qwen3-0.6B.** Its M is 23× beyond Farseer's largest M. Its level is not credible; only its rank is.
- **Ledger check** (SYNTHESIS §2.3, Besiroglu row). We match within rounding:

  | Model | Ours | Ledger |
  |---|---|---|
  | Chinchilla-70B | 1.031 | 1.03 |
  | Gopher | 0.362 | 0.36 |
  | GPT-3 | 0.427 | 0.43 |
  | Llama-2-7B | 2.65 | 2.62 |
  | Llama-3-8B | 5.21 | 5.22 |
  | Llama-3-70B | 2.45 | 2.45 |
  | 405B | 1.35 | 1.35 |
  | DeepSeek-V3 | 3.05 | 3.05 |

  The small differences come from exact HF parameter counts (Llama-2-7B has 6.74B, not 7B). The homothetic grid also reproduces the ledger: Llama-3-8B T/D is 9.5 at (M* = 16, ρ = 0.30) and 11.9 at (20, 0.3527) (`m3_wedge_homothetic.csv`).

### H3. The wedge reproduces developers' stated allocation rules
`m3_wedge_stated_intent.csv`; bottom of `m3_wedge_rivals.tex`.

| Case | Stated rule | ŵ (reference) |
|---|---|---|
| Chinchilla | compute-optimal [hoffmann2022training] | 1.04 |
| Cerebras-GPT, 7 sizes | "compute-optimal", 20 tokens/parameter [dey2023cerebras] | 0.92–1.00 [Rev] |
| Pythia 12B | fixed 300B tokens by research design [biderman2023pythia] | 1.09 |
| LLaMA-1 7B/13B | best performance at given inference budgets [touvron2023llama] | 1.65–2.07 |
| Llama 3 8B/70B | small models trained far beyond compute-optimal for inference [grattafiori2024llama] | 2.48–5.27 |

Compute-optimal designs sit at w ≈ 1 and inference-motivated designs at w > 1. The Pythia siblings (1.09–6.49) show that w > 1 can also be a mechanical by-product of a fixed-D research design, not demand.

[Rev] The builder's 0.87 lower end for Cerebras-GPT came from an Epoch double count for the 13B model: its 257.1B is already the number of tokens seen, and it was multiplied by 0.69 epochs, giving M = 13.6. With the stated 257B tokens (M = 19.8), the 13B has ŵ = 1.00.

### H4. Trends: from under-training to 3–5× over-training within three years, driven by open-weight labs
Production-scale universe [Rev]: 335 models with 6ND ≥ 10²¹ FLOP, released 2019–2026 (271 open, 64 closed). It combines 176 Sample-A rows (Epoch, not speculative, not code, not matched to Sample B) with 159 Sample-B models. The builder applied the 10²¹ threshold to Sample A only, so 14 small Sample-B models (Pythia-70M, Cerebras-GPT-111M, …) sat in the "production-scale" core (n = 349). `m3_wedge_trends.csv|.tex`, figure `m3_wedge_trends`.
- **Share with ŵ < 1 [Rev]:** 47% (2021), 41% (2022), 18% (2023), 0% (2024), 2% (2025).
- **Median ŵ [Rev]:** 1.06 (2022), 1.83 (2023), 3.45 (2024), 4.68 (2025).
- **Median M [Rev]:** 23, 107, 612, 1,303.
- **Open vs closed [Rev].** The open-weight premium is +0.57 log points (SE 0.09) conditional on year and MoE, and +0.44 (0.09) conditional also on ln C. SEs are clustered by developer: 94 clusters after merging name variants of the same developer. The builder had 106 clusters, from "Meta"/"Meta AI", "Cerebras"/"Cerebras Systems", Google's units, and so on. MoE shows no clear difference: +0.01 (0.08), and +0.13 (0.09) given ln C, with active N.
- **Robustness [Rev].** Drop the 11 Sample-A rows whose D is still inconsistent with Epoch's own compute (|ln C/(6ND)| > ln 1.5; `m3_wedge_trends_robust.csv`). The open premium is then 0.59 (0.09) and 0.43 (0.10). For 2021–2025 the by-year medians move by at most 0.14 in ŵ and the shares by at most 2 points. In 2020 (n = 10) the median moves by 0.25.
- **Developers with ≥ 4 production-scale models [Rev]** (median ŵ, `m3_wedge_labs.csv`; a selection, the file has all 25):

  | Developer | Median ŵ |
  |---|---|
  | Hugging Face | 6.61 |
  | Ant Group | 6.34 |
  | Apple | 5.32 |
  | IBM | 5.28 |
  | Alibaba | 4.84 |
  | LG AI Research | 4.22 |
  | Google (incl. DeepMind, Brain) | 2.27 |
  | Meta | 1.83 |
  | EleutherAI | 1.68 |
  | Cerebras | 1.00 |
  | Tsinghua | 0.80 |

  Google and Meta fall from the builder's 3.53 and 2.24 for two reasons. The merged labels now include DeepMind's 2021–22 models (Gopher, Chinchilla) and Meta AI's earlier Epoch rows. And the small models below 10²¹ FLOP (e.g., OPT-125M/350M) drop out.

### H5. Within-family revealed preference (GNR direction) and over-identification
Setup: the flagship is assumed on its own path (w_f = 1), so w_i^rel = (N_i/N_f)^−α (D_i/D_f)^β. This depends on the exponents only; the family's A/B cancels. Files `m3_wedge_family_*.csv`, `m3_wedge_family.tex`, figure `m3_wedge_family`.
- **Siblings.** 95% of the 105 non-flagship siblings in 44 families have w^rel > 1, i.e. smaller siblings are more over-trained. The only exceptions are the five smaller Cerebras-GPT models (constant M = 20, w^rel = 0.92–0.98).
- **The flagship-on-path normalisation holds before 2024 and fails after it** under the common technologies:

  | Flagships | n | Median ŵ | 95% CI contains 1 | CI entirely above 1 | PI band entirely above 1 |
  |---|---|---|---|---|---|
  | Pre-2024 | 18 | 1.14 | 50% | 39% | 28% |
  | 2024+ | 26 | 3.14 | 8% | 92% | 69% |

  Examples: Qwen3-32B 4.46 [3.22, 7.33], Gemma-3-27B 3.37, SmolLM2-1.7B 8.05.
  - Meta is the exception: its flagships Llama 2 70B (2023) and Llama 3.1 405B (2024) have ŵ = 0.98 under Meta's own IsoFLOP law, and the 405B's reference CI [0.99, 2.22] contains 1. So m1's "flagship on its own path" check passes for Meta with Meta's technology.
  - For post-2024 families, the flagship-normalised T are **lower bounds**.
- **Raval-style over-identification: a single demand model T_i = κ_f N_i^−η is rejected.**
  - Absolute T_i, all families: common η̂ = 0.53 (0.07); slope homogeneity F = 2.55 (p = 0.001; 138 models, 42 families).
  - Flagship-normalised T_i: F = 2.77 (p = 0.004).
  - Restricted to families where D varies with size (the only informative ones; with a common D, T_i/D is mechanical in N_i) **[Rev]**:
    - absolute T: p = 0.047 (58 models, 18 families);
    - normalised T: p = 0.122 (33 models, 11 families).
    - "D varies" now requires max D/min D > 1.10. The builder's rule (any difference) counted Llama 3.1 (15T vs 15.6T) and BLOOM (341B vs 366B) as size-varying. With that rule the p-values were 0.031 and 0.124 (70 and 41 models).
  - Families that scale D with N imply planned T *rising* with N (negative family slopes). These are Gemma 2 (−0.74), SmolLM2 (−0.55), Phi-3.5 (−0.53), Gemma (−0.38) and Phi-4 (−0.29). Phi-3's slope is positive.
  - Reading: either demand for large siblings is higher in tokens, or those labs set D by per-size budget rules rather than by lifetime-cost minimisation.

### H6. Validation: at fixed compute and date, more over-trained models are used more
Sample B with HF metadata: n = 164, 26 developers. `m3_wedge_validation.csv|.tex`, figure `m3_wedge_validation`.
- **Outcomes:** HF downloads summed over the base repository and the developer's official post-trained releases; OpenRouter listing and providers; LMArena votes.
- **Why condition on C.** At fixed C, higher M means a smaller, cheaper-to-serve model. *Under a common technology* it also has higher loss than the compute-optimal model. So a positive coefficient is not the quality channel implied by the common technology.
  - **[Rev]** It can still reflect quality differences across developers or vintages (ω, ψ) that correlate with M. The developer-FE column only partly addresses this. The builder's "cannot come from quality" overstates it.
- **Coefficient on ln M, controlling for ln C, release year and log age** (SEs clustered by developer):

  | Outcome | Coefficient (SE) | With developer FE |
  |---|---|---|
  | All-time downloads | **0.97** (0.18) | 0.72 (0.14) |
  | 30-day downloads | **1.13** (0.19) | 0.94 (0.14) |
  | OpenRouter providers [Rev] | 0.08 (0.04), p = 0.03 | 0.06 (0.04) |
  | OpenRouter listing (LPM) | 0.055 (0.030), p = 0.07 | |
  | LMArena votes (n = 46) | 0.29 (0.16), p = 0.08 | |
  | HF likes | 0.27 (0.16), p = 0.10 | |

- **[Rev] Provider counts** are now the number of *distinct* providers across all of a model's listed releases, which is what the table labels them. The builder summed per-release counts, double counting providers that serve several releases. The coefficient moved from 0.083 to 0.081.
- **Unconditional correlations mislead.** ln ŵ is unrelated to likes (0.13, SE 0.25) and to arena votes (−0.17, SE 0.40) without controls, and strongly related once size is held fixed. This is the size confound: w is itself a function of (N, D), so only the conditional coefficients are informative.

### H7. Rival wedges
`m3_wedge_rivals.tex`, `m3_wedge_rival_*.csv`, figure `m3_wedge_rivals`.
- **(a) Data scarcity or Kaplan-era beliefs (w < 1).** Almost all w < 1 models are 2020–2023 releases, mostly closed giants: GPT-3 0.43, Gopher 0.37, MT-NLG 0.28, PaLM 0.41. Among open models there are BLOOM 0.46, OPT-175B 0.43 and GLM-130B 0.53 [Rev: 0.37 before the D correction to 400B tokens]. Post-2024 there is essentially none: 1 of 171 production-scale models (StepFun's Step-1). m7 shows a binding data constraint lowers w, so data scarcity cannot generate the modern w > 1.
- **(b) Memory and hardware tiers are a real but second-order rival [Rev].**
  - Bunching: 26% of 2023–2026 open-weight production-scale models have 6.5–9.5B total parameters (one 16–24 GB GPU in 16-bit), vs 9% under a log-normal fit.
  - Conditional on ln C and year, these tier models have ln ŵ higher by 0.11 (SE 0.06, p = 0.06; n = 238, developer-clustered). That is about a tenth of the median ln ŵ (1.16). The builder reported 0.13 (0.06), p = 0.03 (n = 249) before the universe and cluster fixes, so the effect is now only marginally significant.
  - m7's brute-force result: a binding memory cap at N̄ = 0.5N* alone produces w = 1.78 with T = 0.
- **(c) Non-homotheticity can explain the large models' token counts, not the small ones' [Rev].**
  - Farseer's own Eq. 3 (refit here with m2's code; identical parameters to m2's cache) implies M*(C) of 25 (10²⁰), 28 (10²¹), 43 (10²²), 82 (10²³) and 195 (10²⁴, extrapolated). It is not monotone below 10²⁰ (32 at 10¹⁹).
  - *Within Farseer's N support* (non-embedding N ≤ 6.4B, n = 66), Eq. 3 wedges are *higher* than the Chinchilla-form wedges on the same sweep: median 4.12 vs 3.44. Non-homotheticity therefore does not explain away the small models' over-training.
  - *Above the support* (n = 75), Eq. 3 gives median 1.54 vs 2.12. Under Eq. 3, Llama-3-70B would be *under*-trained (0.62). Llama-3-8B, whose non-embedding N of 7.0B is just above the support, stays over-trained (3.60 vs 4.14).
  - The builder counted models up to 1.5× the largest Farseer model as "in range" (n = 99: Eq. 3 3.11 vs 2.96). Its "above" group (n = 47: 0.84 vs 2.00) also contained 5 models *below* Farseer's smallest N; without them it is n = 42, 0.71 vs 1.76. Both splits are in `m3_wedge_rival_farseer_eq3.csv` (`*_1p5` columns).
  - This is an extrapolation of a 9-parameter form, 1–2 orders of magnitude beyond its N support. Eq. 3 even gives a negative w for the 405B.
- **(d) Distillation and synthetic data.** No detectable difference conditional on ln C and year: distilled +0.02 (0.23) [Rev: SE was misquoted as 0.21], synthetic-data-heavy +0.06 (0.18), n = 173. See Claims for the sign logic.
- **(e) Factor-biased lab productivity** (model_spec Prop. 4: ŵ = w·e^{αψ_N − βψ_D}).
  - m2's recipe tilts move the wedge by a factor of at most 1.25–1.29 across 25 DataDecide recipes (primary), 1.80 in m2's largest robustness variant, and 1.02–1.04 across Gadre's corpora.
  - 88% (79%) of Sample B have ŵ above 1.29 (1.80). 69% (49%) have the entire PI band above it.
  - So recipe bias of the size m2 measures cannot turn the typical modern model into a training-optimal one. It can for about half the models if the extreme 1.8 bound is used together with the widest technology band.

### H8. Aggregate: planned lifetime inference compute of the open-weight ecosystem
Definition: Σ(w_i − 1)C_i / ΣC_i, the compute-weighted mean of w − 1. Sample [Rev]: 271 open-weight production-scale models (6ND ≥ 10²¹; the builder's 285 included 14 small Sample-B models, which carry negligible compute). The numbers below are unchanged to two digits. `m3_wedge_aggregate.csv|.tex`, figure `m3_wedge_aggregate`.

| Releases | Reference [95% CI] | PI band | Notes |
|---|---|---|---|
| All 2019–2026 | **2.0** [1.2, 3.7] | [0.4, 7.2] | |
| 2025 | **3.2** [2.1, 5.7] | [1.05, 9.6] | |
| 2024 | 1.2 [0.6, 2.3] | | |
| 2022 | −0.40 | | 0.03 with T truncated at 0, because of BLOOM, OPT-175B and GLM-130B (GLM-130B now with its 400B tokens [Rev]) |

- **It is technology-dependent** (all years): Hoffmann 0.67, Farseer (incl. emb.) 0.81, Meta law 0.88, Besiroglu 1.97, Gadre RW 3.26.
- **It is dominated by the largest, most extrapolated models.**
- **No comparison with public inference volumes.** I found no verifiable per-model inference-token disclosure that maps to lifetime T. Platform totals such as OpenRouter's (demirer2025emerging) cover a slice of the market. No comparison is reported.

---------------------------------------------------------------------------------------------------

## 2. Methods

### 2.1 Choices data
Full detail in `data/processed/m3_wedge/data_appendix.md`; counts in `output/tables/m3_wedge_data_audit.csv`.

**Sample A (Epoch universe; open and closed)**
- *Source.* Epoch AI `all_ai_models.csv`, snapshot 2026-09-23. 2,070 language-domain rows.
- *Kept.* 342 decoder-only pretrained models, 2019-01 to 2026-09, not derived from another model, with N and D recoverable.
- *Excluded, with reasons.* 1,728 rows. `choices_excluded.csv`:
  - fine-tunes/continued pretraining (non-empty `Base model`) 531;
  - N missing 361; release date outside 2019-01..2026-09 234 [Rev: previously omitted from this list];
  - encoders 34 and encoder-decoders 24; post-trained variants by name 16; non-text by name 2;
  - multimodal 131;
  - benchmark-scale research runs 100;
  - D unrecoverable 224;
  - MoE without active parameters 27;
  - implausible epochs or D/N 15;
  - 29 manual exclusions (post-training updates, systems-paper runs, a diffusion LM, leaked GPT-4 figures, and others).
- *D units.* D = dataset size × epochs. Epoch's older entries are in words; the conversion rule is in the appendix (7 rows converted; 67 rows take D = C/(6N) from operation-counted compute).
- *Primary-source fixes [Rev].* The builder made one: GPT-3 175B uses 300B tokens. The review added five, found by comparing D with Epoch's own operation-counted or reported compute. Each is documented in Epoch's own notes (`build_choices.EPOCH_D_FIX`):
  - GLM-130B: 152B → 400B tokens;
  - Cerebras-GPT-13B: 177B → 257B (size × epochs double-counted tokens already seen);
  - BloombergGPT: 455B → 569B (same double count);
  - Polyglot-Ko-12.8B: 96B words → 167B tokens;
  - GPT-SW3 3.5B: 509B → 102B (× 5 epochs over-counted).
  - 11 production-scale Sample-A rows still have |ln C/(6ND)| > ln 1.5, often because of Epoch compute issues (e.g., CPM-Ant/Bee's compute uses 10× the parameter count). They are flagged in `D_C_mismatch`, and trends are reported without them as a robustness check.
- *MoE.* Active parameters from the name, Epoch's notes, or operation-counted compute.
- *Core.* 6ND ≥ 10²¹, Epoch confidence not "Speculative", not code: 287 models (222 open). [Rev] Trends, rivals (a)–(b) and the aggregate use the *production-scale universe*: Sample-A core rows not matched to Sample B, plus Sample-B general-purpose models with 6ND ≥ 10²¹. That is 335 models in 2019–2026, with 94 developer clusters.

**Sample B (verified open-weight)**
- *Size.* 188 base models in 73 families (173 general-purpose).
- *Sources.*
  - ObsScaling base models, with corrections.
  - The standard Pythia suite (Sloth).
  - 57 curated 2024–2026 additions. For 47 of them, the model card's token statement is matched by a saved regular expression (47/47 match).
- *Corrections to ObsScaling.*
  - OPT D: 0.18T → 0.30T (0.18T is the corpus).
  - TinyLlama v1.1: 3T → 2T.
  - Llama-3.1-405B: 15.6T.
  - Gemma-2B: 3T.
  - BTLM: 0.627T.
  - Qwen2 D filled.
  - Also the m4 fixes.
- *17 dropped.* Undisclosed D: Mistral, Mixtral, Jamba, GPT-2, Qwen1.5-110B, DeciLM. Ambiguous D: InternLM2. Plus the upcycled Qwen2 MoE and duplicates.
- *N.* Exact HF safetensors float-tensor counts (143 models); Gemma 3's vision tower is removed. MoE uses active N.
- *Embedding counts.* Taken from config.json, via ungated mirrors for Meta/Google.
- *Cross-check.* 119 models match an Epoch row. Only 5 differ from Epoch's D by more than 0.05 dex, all documented: Phi-2, OLMo-1B, Llama 4 ×2, Phi-3-mini June update.

**Usage data (Sample B)**
- HF 30-day and all-time downloads and likes, summed over the base repository and the developer's official post-trained releases, including later post-training updates of the same base (Llama 3.3 70B, Qwen3-2507, DeepSeek-V3-0324/R1, Kimi-K2-0905).
- OpenRouter listing, distinct providers across the model's listed releases [Rev], and price (2026-09-23 snapshot plus endpoints API).
- LMArena text-arena votes: maximum over all published leaderboards, 46 matched models.

### 2.2 Technologies
All reduced to (α, β, ln G) plus bootstrap draws. `m3_wedge_technologies.csv|.tex`; code in `technologies.py`.

**Upstream from m1**
- Chinchilla refit (Huber, n = 240): pairs and 9-cluster draws, 400 each. This is the reference technology.
- NLS in levels, and n = 245: sensitivity only.
- Meta Llama 3 IsoFLOPs:
  - Approach 2/1 (reproduces Meta's law D* = 0.299 C^0.537), with α = γ/a and β = γ/(1−a) under κ = 1;
  - the primal (A3) on the same points.

**Upstream from m2**
- Farseer: non-embedding N (headline row), incl.-embedding N, and q-family inner exponents.
- Gadre et al.: RefinedWeb, C4, RedPajama.
- OLMo ladder: N excludes the input embedding.

**Literature points**
- Besiroglu published; Hoffmann A3 at TeX precision (sl.py).

**Homothetic CES grid**
- M* ∈ {16, 20, 24.2, 41, 192} × ρ ∈ {0.30, 0.3527, 0.40}.

**Farseer Eq. 3 (non-homothetic)**
- Refitted with m2's estimator on m2's harmonised panel (parameters identical to m2's cache).
- w = (∂L/∂ln N)/(∂L/∂ln D) by central differences. M*(C) by 1-D minimisation.

### 2.3 Wedges, inference and bands
- **Point estimates.** For every model × technology, the full-sample point estimate gives ŵ, T/D = 3(ŵ−1), T = 3D(ŵ−1) and CE.
- **Intervals.** 95% percentile intervals come from the technology's bootstrap draws, which carry the joint uncertainty in α, β and G. Literature technologies have no interval.
- **PI band.** The union of intervals and points over BAND_SET = {Chinchilla refit, Besiroglu, Hoffmann, Farseer ×2, Gadre RW}, plus lab-own technologies: Meta (A2/A1, A3) and AI2 (OLMo ladder). A "core" band without Hoffmann and Gadre and a "no Hoffmann" band are in `wedge_band.csv`. For Sample B both give the same share with band_lo > 1 (79%). [Rev] Hoffmann or Gadre set the band minimum for 111 of 173 models, but for every model whose band reaches below 1, the Farseer or Chinchilla-refit intervals also reach below 1. The median band_lo is 1.77 (full), 1.83 (no Hoffmann) and 1.86 (core).
- **Extrapolation flags.** Models are flagged when M is outside the technology's design M range, and separately when 6ND exceeds its largest budget. [Rev] The builder wrote "(always)". In fact 75% of Sample B exceed the Chinchilla budget (`share_Cx` in `m3_wedge_sampleB_by_tech.csv`).

### 2.4 Families, over-identification, trends, validation, aggregate
- **GNR.** Families are Sample-B groups from one developer and one data release. The flagship is the largest by total parameters. w_i^rel carries bootstrap bands from each technology's (α, β) draws.
- **Over-identification.** OLS of ln T_i on family effects + ln N_i (η̂ with family-clustered SE) vs family-specific slopes (classical, homoskedastic F test). Run on absolute and flagship-normalised T, and on all families vs families with size-varying D (max/min D > 1.10 [Rev]). Only T_i > 0 enter, which is selection on the outcome.
- **Trends.** Medians and shares by year, openness, MoE and size class. OLS of ln ŵ on year effects, open and MoE (± ln C), clustered by developer. [Rev] 94 clusters after mapping name variants to the parent developer (`analysis.DEV_MAP`; Alphabet units → Google); the builder had 106.
- **Validation.** OLS with developer-clustered SEs; specifications in H6.
- **Aggregate.** Compute-weighted mean of w − 1, raw and truncated at T ≥ 0, by release year. Reference-technology bootstrap interval, plus a band over six technologies.

---------------------------------------------------------------------------------------------------

## 3. Table and figure inventory
All in `output/tables/` and `output/figures/` (figures as `.pdf` + `.png`).

| File | Content |
|---|---|
| `m3_wedge_table5.tex/.csv` | **Paper Table 5**: revealed inference demand by family and size. 43 rows across 13 families/flagships. Columns: N, D, M (with extrapolation marks), ŵ [95% CI], T/D, T, CE, PI band, ŵ under Meta's law, w^rel. |
| `m3_wedge_fig4.pdf/.png` | **Paper Figure 4**: ŵ (log) vs M (log). Contents: verified open-weight models with PI bands (gray), other Epoch open-weight models (light gray), flagships of the main families (orange), MoE (hollow squares), Meta models under Meta's own law (aqua diamonds), the sufficient-statistic curve at C = 10²⁴, and the design M-ranges of Chinchilla, Gadre and Farseer. |
| `m3_wedge_technologies.tex/.csv` | Technologies used: α, β, a, σ*, M*(10²¹), M*(10²⁴), design M range, N convention, number of draws. |
| `m3_wedge_family.tex`, `m3_wedge_family_members.csv`, `m3_wedge_family_summary.csv`, `m3_wedge_overid_demand.csv`, `m3_wedge_overid_siblings_wrel_lt1.csv` | Within-family GNR and the over-identification tests. |
| `m3_wedge_family.pdf` | Small multiples, 9 families: ŵ vs N and w^rel vs N. |
| `m3_wedge_trends.tex/.csv`, `m3_wedge_trends_reg.csv`, `m3_wedge_labs.csv`, `m3_wedge_trends_robust.csv` [Rev], `m3_wedge_trends.pdf` | Trends and heterogeneity (production-scale universe); the robust file and the second half of `trends_reg` drop the 11 Sample-A rows with D/compute inconsistencies. |
| `m3_wedge_validation.tex/.csv`, `m3_wedge_validation.pdf` | Usage validation (FWL partial plots). |
| `m3_wedge_rivals.tex`, `m3_wedge_rival_*.csv`, `m3_wedge_stated_intent.csv`, `m3_wedge_rivals.pdf` | Rival wedges and stated-intent checks: size bunching; M*(C) under Chinchilla refit, Farseer Eq. 3, Farseer Chinchilla-form and Meta's law. |
| `m3_wedge_aggregate.tex/.csv`, `m3_wedge_aggregate_by_tech.csv`, `m3_wedge_aggregate.pdf` | Aggregate planned-inference multiple by year. |
| `m3_wedge_homothetic.tex/.csv` | T/D over the (M*, ρ) grid for five models (reproduces the ledger's homothetic table). |
| `m3_wedge_models.csv`, `m3_wedge_models_long.csv` | Every model × every technology: w, CI, CE, T/D, extrapolation flags, bands, Farseer Eq. 3 wedge. |
| `m3_wedge_sampleB_inputs.csv`, `m3_wedge_data_audit.csv` | Sample B inputs with sources; exclusion counts. |
| `m3_wedge_sampleB_by_tech.csv`, `m3_wedge_flagships_by_period.csv` | Summaries quoted in H1 and H5: Sample-B median ŵ, share > 1, share of lower bounds > 1, extrapolation share and interval ratios by technology; flagship wedges before vs after 2024. |
| `data/processed/m3_wedge/` | `choices.csv` (clean choices dataset), `sampleA_epoch.csv`, `sampleB_verified.csv`, `choices_excluded.csv`, `sampleB_dropped.csv`, `data_appendix.md`, `headline.json` (numbers quoted here). |

---------------------------------------------------------------------------------------------------

## 4. Claims for the paper

1. **Modern open-weight base models reveal planned lifetime inference compute of several times their training compute.**
   - *Evidence:* Sample B median ŵ = 3.19 (T/D = 6.6). 95% have ŵ > 1; 87% have a bootstrap lower bound > 1; 79% [Rev] have the whole six-technology PI band > 1 (`m3_wedge_models.csv`, `headline.json`).
   - *Caveat:*
     - The *level* of T is technology-dependent: median ŵ is 1.9 (Hoffmann) to 4.0 (Gadre RW).
     - [Rev] 75% of models are extrapolations in compute beyond every band design, and 55% are outside the Chinchilla design's M range. The builder wrote "every model", which is not true for the small models.
     - D is in each model's own tokens (a 20% tokenizer difference moves ŵ by about 7%).
     - Report ranges and ranks, not point T.

2. **Llama 3 8B's allocation rationalises roughly 13 training-token-equivalents of lifetime inference per training token under the Chinchilla refit, and 6 under Meta's own law.** The 405B flagship is on Meta's own training-optimal path.
   - *Evidence:*
     - Reference: T/D = 12.8 [8.9, 21.4]. Meta law: ŵ = 3.09, T/D = 6.3. PI band for T/D: [4.4, 33].
     - 405B: ŵ = 0.98 (Meta law) and 1.37 [0.99, 2.22] (reference) (`m3_wedge_table5.csv`).
   - *Caveat:*
     - Meta's law imposes κ = 1 on Approach-2/1 objects. m1 shows the primal on the same points disagrees (A3: ŵ = 4.82).
     - **Ledger correction:** SYNTHESIS's "Meta's own law gives about 9.6×" imposes ρ = 0.35. Meta's implied curvature is (α+β)/2 = 0.27, which gives 6.3.

3. **The wedge ranks developers' stated objectives correctly.**
   - *Evidence:* stated compute-optimal (Chinchilla 1.04; Cerebras-GPT 0.92–1.00 [Rev]) vs stated inference-oriented (LLaMA-1 small 1.65–2.07; Llama 3 2.48–5.27) (`m3_wedge_stated_intent.csv`).
   - *Caveat:* five cases chosen because the rule is stated. This is a sanity check, not a test with power.

4. **The industry switched from under- to over-training around 2023–2024, led by open-weight developers.**
   - *Evidence:*
     - [Rev] Share ŵ < 1: 47% (2021) → 0% (2024). Median ŵ: 1.1 (2022) → 4.7 (2025). Production-scale universe (6ND ≥ 10²¹ in both samples), n = 335.
     - [Rev] Open-weight premium +0.57 (0.09) log points given year; +0.44 (0.09) also given ln C (94 developer clusters; `m3_wedge_trends_reg.csv`). It is unchanged when the 11 Sample-A rows with D/compute inconsistencies are dropped (0.59, 0.43).
   - *Caveat:*
     - Closed-model N and D are rarely disclosed (64 closed production-scale models; 3 in 2025). The closed sample is selected on disclosure. [Rev] So is the open sample: developers that do not disclose D (Mistral, Mixtral, Jamba, DeciLM) are absent from Sample B.
     - [Rev] Near-duplicate observations: Llama 3 and Llama 3.1 8B/70B have identical (N, D), and so do Qwen-72B and Qwen1.5-72B. Each pair is counted twice in the trends.
     - Early w < 1 reflects Kaplan-era beliefs or data limits, not negative demand.

5. **Smaller siblings are more over-trained than their flagships in essentially every family (GNR direction), but the flagship-on-path normalisation fails for post-2024 families under common technologies.**
   - *Evidence:*
     - 95% of 105 siblings have w^rel > 1.
     - Post-2024 flagships: median ŵ = 3.14, CI above 1 for 92%. Pre-2024: 1.14, with CI containing 1 for 50%.
     - Meta's flagships are at 0.98 under Meta's law (`m3_wedge_family_summary.csv`).
   - *Caveat:* the flagship-normalised T are lower bounds when w_f > 1, which is the post-2024 norm. The family A/B cancels only under the Chinchilla form with common exponents within the family.

6. **A single power-law demand for model size across families is rejected (Raval-style over-identification), mainly because families that scale D with N imply planned T rising with size.**
   - *Evidence:* homogeneity F = 2.55 (p < 0.001) with absolute T. [Rev] Within size-varying-D families (max/min D > 1.10): p = 0.047 with absolute T (58 models, 18 families) and p = 0.12 normalised (`m3_wedge_overid_demand.csv`).
   - *Caveat:*
     - With a common D the test is partly mechanical: T_i/D is a nonlinear function of N_i, so slopes differ even under the null.
     - The informative subsample is small (11–18 families [Rev]). The rejection there is marginal (p = 0.047) and disappears with normalised T. The F test is classical (homoskedastic).
     - Read as evidence against lifetime-cost minimisation with a common downward demand curve, *or* for per-size budget rules. Not a rejection of the wedge itself.

7. **Revealed over-training predicts usage at fixed compute and release date.** Log downloads rise about one-for-one with log M (0.97, SE 0.18; 0.72 with developer effects).
   - *Evidence:* `m3_wedge_validation.csv`: n = 164, 26 clusters. Also OpenRouter providers 0.08 (0.04) and LMArena votes 0.29 (0.16).
   - *Caveat:*
     - Downloads are not inference tokens (they include CI and automated pulls, and local use).
     - OpenRouter is a current snapshot with survivorship (21 of 164 listed).
     - Smaller models also get more downloads because more users can run them (the hardware channel of Claim 8b).
     - The coefficient is a correlation consistent with anticipated demand, not an estimate of T.
     - [Rev] At fixed C, lnM is (ln D − ln N)/2 plus a constant, so the result is equivalently "smaller models at given compute are downloaded more". Developer or vintage quality (ω, ψ) correlated with M is not ruled out. The developer-FE estimate (0.72) is the more conservative number.

8. **Rival wedges.**
   - (a) **Data scarcity** (w < 1) is a pre-2024 phenomenon.
   - (b) **Hardware tiers matter for sizing, less for the wedge [Rev]:** 26% of open production-scale models sit at 6.5–9.5B vs 9% under a log-normal fit, and tier models have ŵ 11% higher given C (SE 6%, p = 0.06).
   - (c) **Non-homotheticity** (Farseer Eq. 3: M* rising from about 25 to about 195 over 10²⁰–10²⁴) can rationalise the largest models' M but not the small models'. [Rev] Within Farseer's N support, Eq. 3 gives *larger* wedges than the Chinchilla form (4.12 vs 3.44, n = 66).
   - (d) **Distillation and synthetic data** show no detectable wedge difference.
   - (e) **Factor-biased recipes** of the size m2 measures (≤ 1.29×, at most 1.8×) cannot bring 79–88% of Sample B to w ≤ 1.
   - *Evidence:* `m3_wedge_rivals.tex` and the CSVs.
   - *Caveat:*
     - (c) rests on a 9-parameter form extrapolated 1–2 orders of magnitude beyond its N support.
     - (d) The sign of the distillation bias is theoretically ambiguous. If distillation is data-augmenting (ψ_D > 0), ŵ *understates* the lab's true wedge (ŵ = w·e^{−βψ_D}). If it is parameter-augmenting, ŵ overstates it. With 10 distilled models the test has no power.
     - (e) uses recipe variation within one lab's sweep (DataDecide) as the scale of cross-lab factor bias.

9. **The open-weight ecosystem planned roughly 2× (all years; [0.4, 7.2] across technologies) and 3× (2025 releases; [1.05, 9.6]) as much lifetime inference compute as training compute.**
   - *Evidence:* `m3_wedge_aggregate.csv`.
   - *Caveat:*
     - Dominated by the largest models, which are the most extrapolated.
     - Strongly technology-dependent: 0.67–3.26 over technologies (all years).
     - Not validated against any inference-volume data.
     - Present as an illustration of the object, not a measurement.

---------------------------------------------------------------------------------------------------

## 5. Robustness and failures

- **Technology dispersion is the dominant uncertainty, as m7 predicted.** M*, not curvature, drives the level: the median ŵ across technologies ranges 1.9–4.0, while the within-technology bootstrap interval is about ×(0.80, 1.39) around the point (median over Sample B).
  - Hoffmann's A3 is marginally inconsistent with its own IsoFLOP argmins (m1 H6). Gadre RW's M* (1.0–3.4 at 10²¹–10²⁵) is fragile; m2 finds 3.4 (Huber) vs 10.4 (NLS).
  - Both are kept in the band to be conservative, but a "core" band without them is in `wedge_band.csv`.
- **Estimator and sample on the Chinchilla data.** Median ŵ over Sample B is 3.19 (Huber), 4.11 (NLS levels) and 4.78 (all 245 points).
  - The m1 9-cluster bootstrap widens the intervals: the share with lower bound > 1 falls from 87% to 80%.
- **Parameter conventions.** The Farseer technology with non-embedding N (applied to N − embeddings) gives median 2.67 (n = 146). The incl.-embedding version gives 2.41 (n = 173). For small models with 150–260k vocabularies (Gemma, Qwen) the two differ most.
  - The q-family (m2's preferred, q = 0.47) gives higher wedges: median 3.85, since the inner exponents are larger.
- **Input measurement.** d ln ŵ/d ln D = β ≈ 0.37 and d ln ŵ/d ln N = −α ≈ −0.35. "Up to 9T" (Llama 3.2) or "over 15T" (Llama 3) moves ŵ by a few percent.
  - Five Sample-B D values differ from Epoch by > 0.05 dex (documented).
  - ObsScaling errors corrected here: OPT D was the corpus, not tokens seen; TinyLlama v1.1 is 2T; Sloth's Gemma-2 token counts are shifted by one size (2B listed at 8T) and are not used.
- **Epoch Sample A** relies on heuristics for word/token units and on compute-derived D (67 rows). It is used only for trends and the aggregate; the Sample-B results do not depend on it.
  - [Rev] The review found five Sample-A D errors that the heuristics missed, by checking D against Epoch's own compute: GLM-130B, Cerebras-GPT-13B, BloombergGPT, Polyglot-Ko-12.8B and GPT-SW3. They are corrected from Epoch's own notes. 11 production-scale rows remain inconsistent (|ln C/(6ND)| > ln 1.5, flag `D_C_mismatch`). Dropping them leaves the trends and the open premium unchanged (`m3_wedge_trends_robust.csv`, `m3_wedge_trends_reg.csv` rows with sample = "drop …").
  - [Rev] Epoch's 'Training dataset size' is sometimes the tokens *seen* and sometimes the corpus size. Multiplying by `Epochs` double-counts in the first case. Any row with Epochs ≠ 1 is at risk; the 16 production-scale such rows were read by hand. The four wrong ones (GPT-3 and three of the new fixes) are corrected; ruGPT-3.5 (3 epochs, notes silent) remains flagged.
- **MoE.** Wedges use active parameters with dense-model technologies. No MoE-specific technology is used; the Abnar et al. exponents in the ledger are for a different form.
- **Things that did not work or were not done.**
  - The HF API returns authentication errors for MosaicML and Cerebras repositories, so there are no downloads for those 9 models.
  - OpenRouter covers only 21 Sample-B releases (current snapshot); the Wayback history was not used.
  - The LMArena match is manual and limited (46 models).
  - The nonparametric monotone bound of m7 (Prop. A9) was not applied: m7 shows it is uninformative for these models, since no support point dominates Llama-3-8B.
  - Farseer Eq. 3 wedges for N far above Farseer's range are unusable (negative for the 405B) and are reported only as flagged.
  - [Rev] `data/processed/m3_wedge/data_appendix.md` is hand-written, not produced by `run.py`: a clean rebuild of `data/processed/m3_wedge/` deletes it. Keep a copy, or move it next to the code.
- **sl.py.** No bug found. As m1 noted, `sl.fit_chinchilla` returns the Huber optimum while `sl.BESIROGLU` holds the published (LAD-type) values. This module uses m1's Huber refit as reference and carries Besiroglu's published values as a separate technology.

---------------------------------------------------------------------------------------------------

## 6. Open issues

1. **Tokenizer and output harmonisation.** Technologies are in their sweeps' tokens and industry D in each model's tokens. A bytes-based D (tokens × bytes/token of each tokenizer) would put all models on one scale. Not done: it needs tokenizer compression rates per model.
2. **Extrapolation beyond Farseer's M range [Rev].** 50 Sample-B models exceed it with N incl. embeddings (M ≤ 1,227), and 30 of 146 with non-embedding N (M ≤ 2,570); the builder's "47" matched neither. Qwen3-0.6B reaches M = 60,000. The self-trained over-training sweep proposed in SYNTHESIS §8.5 (M 5–2,000+) is the natural check on the direction of the extrapolation bias. Sardana et al. [sardana2024beyond] report that extra tokens are overvalued at extreme M, which would mean ŵ is *understated*.
3. **Lab-own technologies beyond Meta and AI2** (e.g., StepFun's Step Law sweeps, DeepSeek's reported scaling fits) would tighten the band for those labs. Under AI2's OLMo ladder, OLMo 2 wedges are 0.91 (32B) to 2.01 (1B). That is close to on-path for the flagship, like Meta, but the OLMo technology is imprecise (its band reaches 0.23).
4. **Demand data.** Inference-token volumes per model (e.g., Demirer et al.'s OpenRouter panel [demirer2025emerging], access unverified) would turn H6 from a correlation with downloads into a test of T levels.
5. **Survivorship in usage data.** A panel of OpenRouter snapshots from Wayback would remove the "listed today" selection.
6. **Paper wording.** Do not call w a markup (SYNTHESIS §3 H6). It is an input-specific wedge: the ratio of lifetime to training compute.

**Citation keys used** (all in `lit/references.bib` or `lit/bib/extra_m3_wedge.bib`):
- sardana2024beyond, raval2023testing, gandhi2020identification, deloecker2012markups, hsieh2009misallocation;
- hoffmann2022training, besiroglu2024chinchilla, grattafiori2024llama, li2025predictableb, gadre2024language, bhagia2024establishing, muennighoff2023scaling;
- biderman2023pythia, ruan2024observational, maiapolo2024sloth, epochai2026data, lmarena2026leaderboard, openrouter2026models, demirer2025emerging, busbridge2025distillation, kaplan2020scaling, teamolmo2024olmo, deepseekai2024deepseekv3.

New keys in `lit/bib/extra_m3_wedge.bib`, verified against the arXiv API on 2026-09-24:
- touvron2023llama, touvron2023llama2, dey2023cerebras, brown2020language, zhang2022opt;
- yang2024qwen2, qwen2024qwen25, yang2025qwen3;
- gemmateam2024gemma, gemmateam2024gemma2, gemmateam2025gemma3;
- abdin2024phi3, benallal2025smollm2.
