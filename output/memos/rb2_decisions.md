# Memo: module rb2_decisions, the value of compactness revealed by allocation decisions

> **[rb4 integration, 2026-09-24]** rb1_sigmaC's study-level σ\* changed from 0.6929 (HKSJ s.e. 0.0126) to **0.6873 (s.e. 0.0148)** because Chinchilla is now in FLOP-effective parameters with FLOPs per token rebuilt from Hoffmann et al.'s architecture table (module rb4_chinflop, adopted by the lead author). This module reads that row at run time (`rb2common.common_sigma`), so it was re-run end to end (code unchanged; 216 s). Numbers in the body are pre-integration unless marked.
> - **What changed (old → new).** Only the lab-own one-rule columns and what is built from them. One-rule median s over the 22 lab-own models 0.515 → **0.524**; without AI2 0.632 → 0.642; Meta 0.569 → 0.579; AI2 0.435 → 0.444. Meta 10-budget law column 0.479/0.535/0.458/0.435 → 0.488/0.544/0.467/0.444; OLMo κ = 1 path column 0.441/0.632/0.569/0.314 → 0.449/0.642/0.579/0.321. Llama 3 8B one-rule w 8.27 [4.51, 15.91] → **8.75 [4.58, 18.13]**; Llama 3.1 405B 1.46 → 1.47. Each lab-own model's s rises by 0.002–0.010. "Lab-own where available": median s over models 0.737 → 0.738, over decision units 0.726 (unchanged), interval [0.649, 0.776] → [0.649, 0.777]. Table 2 lab-own column: Llama 3 herd 0.37 → 0.38, Llama 2 0.33 → 0.34, OLMo 2 7B 0.52 → 0.53, Marin 8B 0.80 → 0.81. Trend table lab-own column: 2023 units 0.51 → 0.52, 2024 0.79 → 0.80, aggregate 2025 0.80 → 0.81. The ECDF figure's lab-own curve moves accordingly.
> - **Unchanged (diffed):** every reference-technology number, the 32 technologies other than the lab-own row, readings, serving, sign identification, second output, post-training and the lab-curvature and earlier-rule columns. Changed files: `rb2_decisions_{headline,labown_models,labown_summary,table2,trend_by_tech,trend_monotone}.csv`, `rb2_decisions_{labown,table2,trend}.tex`, `rb2_decisions_ecdf.{pdf,png}`, `data/processed/rb2_decisions/{clean_models,decision_units}.csv` (columns `w_lab1*`, `s_lab1`, `w_primary1` only), `headline.json`, `exhibits_cache.pkl` and the run logs. rb3_econ2 reads only the reference-technology columns and needs no re-run. Details: `paper/notes/integration_log_rb4.md`.


Module owner: rb2_decisions (Claude). Date: 2026-09-24. **Revised after the independent review** (`rb2_decisions_review.md`): numbers below are from the reviewed run; changes are marked [review]. The round-2 requests covered are R1 New 1, 3, 4, 5; R2 Majors 2, 4, 6, 7; R3 N2, N3, N4; and R4 R2-M1 to M3. The binding plan is `paper/notes/revision_plan_v3.md`.

**How to run it.** The entry point is `code/analysis/rb2_decisions/run.py`.
- The run is deterministic. It uses the CPU only, with at most 4 processes, and takes 3.5 to 4.5 minutes.
- Two consecutive full runs gave byte-identical CSV and .tex outputs (40 files; checked by md5).
- `--exhibits-only` redraws the tables and figure from `data/processed/rb2_decisions/exhibits_cache.pkl`.
- `--rebuild-techs` rebuilds the cached ra2 technology registry. The seeds are fixed, so the rebuilt draws are the same set, possibly in a different order. [review] A from-scratch run with `--rebuild-techs` reproduced every CSV and .tex output byte for byte.
- Primary sources are downloaded once by `fetch_sources.py`. `run.py` never downloads anything. It checks every quoted passage against the saved sources and stops if one is not found.

**Inputs (read-only).**
- ra2_wedge code, imported rather than copied. Its two side-effect writes are redirected to this module's folder.
- m1/m2/m3 registries and draws.
- ra1's reviewed per-budget IsoFLOP minima (`ra1_modelfree_isoflop_budgets.csv`).
- The common curvature: rb1_sigmaC's study-level σ* (`rb1_sigmaC_study_level.csv`, row "PRIMARY"), which is **0.6929 (HKSJ s.e. 0.0126, k = 4 studies)**. It is read at run time; ra1's 0.695 is the fallback if the rb1 file is absent. [review] Curvature draws are σ* + s.e. × t₃, which reproduces rb1's HKSJ interval [0.653, 0.733]; the builder's normal draws gave only ±0.025.
- `sl.py` was not edited.

**Notation.**
- w = ε_N/ε_D, M = D/N, C = 6ND.
- s = (w − 1)/w = m_N/(1 + m_N), where m_N is the value of compactness per unit of training cost (Prop. A8).
- A **decision unit** is one choice of (N, D).
- Family share of a common-D family: W_f = [Σ_j ω_j/w_j]⁻¹ and s_f = 1 − 1/W_f = Σ_j ω_j s_j, with ω_j the members' training-compute shares (Prop. A9).
- Labels: [E] estimated here; [U] upstream module; [L] taken from a primary source and quote-verified.

---------------------------------------------------------------------------------------------------

## 1. Headline findings

### H1. Decision units: the headline survives, and the Llama 3 showcase does not [E/L]
The clean sample has 77 models. They form **56 decision units**:
- 11 families that trained every size on one token budget (max/min D ≤ 1.10);
- 30 members of families with size-specific budgets;
- 15 singletons.

Files: `rb2_decisions_headline.csv`, `data/processed/rb2_decisions/decision_units.csv`.

**Headline statistics.** Brackets are joint wild-bootstrap intervals of one technology: the reference's 399 wild draws, each used three times (B = 1,197) so that lab-own path draws can be paired [review: B = 999 had cycled the 399 draws unevenly].

| Units | n | Median s (reference) | Median s (lab-own, one rule) | Share w > 1 (reference) |
|---|---|---|---|---|
| Decision units (primary) | 56 | **0.736 [0.666, 0.784]** | 0.726 [0.649, 0.776] | **0.964 [0.945, 1.000]** |
| ... under ra1's budget-plus-trunk cluster bootstrap (44 clusters) [review] | 56 | 0.736 **[0.526, 0.808]** | | 0.964 [0.839, 1.000] |
| Reading-consistent units (cap members as lower bounds; cap+menu families dropped) | 66 | 0.752 [0.703, 0.804] | 0.752 | 0.970 |
| ... without cap members inside a tier window (bounds conflict) [review] | 57 | 0.749 [0.691, 0.794] | | 0.965 |
| Sub-group units (size-specific families split where members share one D) | 49 | 0.735 [0.657, 0.782] | 0.717 | 0.980 |
| Models (the v2 unit) | 77 | 0.749 [0.689, 0.794] (cluster: [0.575, 0.827]) | 0.737 | 0.974 |
| Members of size-specific families and singletons (where Prop. 2(i) applies) | 45 | 0.749 | | 0.956 |
| ... of which outside memory-tier windows | 19 | 0.848 | | 0.947 |

- **Across the 32 ex-ante technologies**, the decision-unit median s runs from **0.06 to 0.81**, and the share of units with w > 1 from 0.57 to 1.00 (`rb2_decisions_technologies_units.csv`).
- **[review] Conservative inference (R4 R2-M3(d)).** Under ra1's budget-plus-trunk cluster bootstrap of the reference (`cluster_ref.py`; 44 clusters, B = 999, same point estimates), M\*(5.76×10²³) widens from [14.0, 33.9] (wild) to [6.9, 67.0], and the decision-unit median's interval from [0.67, 0.78] to **[0.53, 0.81]**. The lower end is just above one half, so "exceeded training cost for the median decision" (w > 2, s > 0.5) holds under this scheme only barely.
- **Robustness (reference):**
  - without Alibaba: 0.737 (47 units);
  - without Qwen2.5/Qwen3, whose per-member D is not audited: 0.736;
  - without the synthetic-data models: 0.699;
  - with ra2's uncorrected token counts: 0.736.

**Readings of the 11 common budgets.** Sources are the technical reports and release posts. The quotes, URLs and retrieval times are in `data/processed/rb2_decisions/family_readings.csv`, and every quote is verified against the saved text.

| Family | Reading | Family share s_f [95%] | Evidence (short) |
|---|---|---|---|
| Llama 3 herd (8B, 70B, 3.1 405B) | cap | **0.29 [0.08, 0.46]** | "corpus of about 15T"; 8B/70B "continued to improve log-linearly after we trained them on up to 15T tokens"; flagship "approximately compute-optimal size for our training budget" |
| Qwen3 (5 base sizes) | cap | 0.89 [0.85, 0.92] | all models on the 36T dataset; data expanded by PDF extraction and synthetic generation |
| Qwen2.5 (7 sizes) | cap (weak) | 0.72 [0.63, 0.78] | "all models ... encompassing up to 18 trillion tokens"; tokens per member not stated |
| DeepSeek LLM (7B, 67B) | cap+menu | 0.18 [0.00, 0.32] | dataset "currently consists of 2 trillion tokens"; sizes are "two prevalent used open-source configurations" |
| Yi (6B, 34B) | cap+menu | 0.49 [0.38, 0.59] | 34B chosen for the 24G memory of an RTX 4090; "not saturated at 3.1T" |
| MPT (7B, 30B) | menu | 0.24 [0.10, 0.35] | 30B "specifically chosen to make it easy to deploy on a single GPU"; 1T "like LLaMA", sampled from a larger mix |
| Llama 2 (7B, 13B, 70B) | choice | 0.23 [0.06, 0.36] | "2 trillion tokens ... good performance–cost trade-off" |
| Granite 3.0 (2B, 8B) | choice | 0.85 [0.81, 0.89] | dense models on 12T; the MoE siblings on 10T of the same data |
| Apertus (8B, 70B) | choice | 0.65 [0.54, 0.73] | 15T in five stages; no statement that the corpus was exhausted |
| Olmo 3 (7B, 32B) | choice | 0.63 [0.53, 0.70] | 6T mix drawn from a 9T pool; the 32B schedule was truncated at 5.5T |
| StableLM-Alpha (3B, 7B) | choice | 0.53 [0.44, 0.59] | 800B tokens of a 1.5T dataset |

- **What each reading licenses:**
  - **cap** (binding data cap; Prop. A8): each member is its own decision given the cap, and its w − 1 is a **lower** bound on m_N;
  - **choice**: only the π-weighted family share is revealed (Prop. A9);
  - **menu**: only an upper bound (A9(iv));
  - **cap+menu**: nothing is revealed.
- **Llama 3 herd (R1 New 1a, R2 M2, R3 N3b).** The herd reveals one family share of **0.29 [0.08, 0.46]** under the reference and 0.37 under Meta's own 8-budget path.
  - The flagship's sign is **not identified** beyond the designs (Llama 3.1 405B: lab-own w = 1.46 [0.71, 3.01]).
  - Under the cap reading, which the report supports, the members' wedges are lower bounds, and so is the family share [review: with a binding data cap the family condition becomes Σπᵢ(1 + m_N,i) = (1 + μ)W̄_H ≥ W̄_H, so 0.29 bounds the herd's π-weighted value from below]:
    - Llama 3 8B: s ≥ 0.85 under the reference (w = 6.73 [5.16, 8.84]) and s ≥ 0.88 under Meta's path (w = 8.27 [4.51, 15.91]);
    - 70B: s ≥ 0.63 (reference) and ≥ 0.68 (lab-own; w = 3.13 [1.65, 6.06]).
  - [review] Both the 8B (8.0B) and the 70B (70.6B) sit in memory-tier windows (6.5–9.5B, 65–72.9B). A binding tier cap makes w − 1 an upper bound (Prop. A8(iii)(b)) and a binding data cap a lower bound (A8(iii)(d)); if both bind, the member wedge bounds nothing. The lower-bound statement therefore needs the premise that the sizes were not set by a tier. 9 of the 15 cap members are in tier windows; dropping them from the reading-consistent units gives 0.749 (57 units).
  - The within-Meta contrast ("flagship for training efficiency, 8B for serving") is not a revealed-preference statement. Recast it as "the herd reveals ≈ 0.3; the small members' wedges bound their value of compactness from below."
- **Token audit.** StableLM-Alpha's training tokens are 800B (Stability-AI/StableLM README), not the 1.5T dataset size that m3 and ra2 used. The parameter counts are 3.64B and 7.87B, not the nominal 3B and 7B. Both are corrected (`d_audit.csv`). The member wedges move from 3.79 to 2.66 (3B) and from 2.65 to 1.92 (7B). The medians do not move.

### H2. Serving coded at the model level: the serving group is small and concentrated [E/L]
File: `data/processed/rb2_decisions/serving_model_level.csv`. Each row gives the channel and the evidence origin. [review correction] Not every row has its own dated quote: 48 rows carry the model's own evidence (URL, date, lag, retrieval time, verified quote), 13 inherit a sibling's from the same release, 2 a same-developer page (no date, so no lag), and **14 rows, all coded 0** (TII ×4, Stability AI ×4, AI2 ×4 including OLMo 2 1B, TinyLlama, M-A-P), rest on ra2's developer-level coding with a note and no URL. Absence of first-party serving is not documented by a dated page for these; all 26 served = 1 codes have a dated, quote-verified source.

**The coding rule.** A model is coded 1 if a first-party per-token API or consumer product offered **this size** (any post-trained variant of the same run) within 180 days of release. The evidence is dated first-party pages, Wayback captures of Alibaba's DashScope/Model Studio catalogs, or dated press. Two kinds of availability are coded 0 in the primary code and kept as separate flags:
- self-deploy catalogs (Vertex AI Model Garden; Azure AI Studio catalog), where the customer rents the accelerators;
- research demos (Hugging Face Spaces; Ai2 Playground).

**Result.**
- **26 of 77** models were served, against 33 under ra2's developer-level code.
- Recoded from 1 to 0 (7): Gemma 2B/7B (Vertex self-deploy only), DeepSeek LLM 7B (the chat product served only the 67B), phi-1.5 (research release), phi-2 (Azure catalog), and Llama 3 8B/70B.
- Llama 3 8B/70B are ambiguous: "Meta AI, built with Llama 3 technology" does not name a size. They are served = 1 in a sensitivity; the Llama API came only in April 2025.
- Yi-34B is also ambiguous, via an unnamed early-access API.
- All 19 Alibaba sizes, including Qwen3 0.6B and Qwen2.5 0.5B, were in Alibaba's own catalogs. Qwen3 appeared on its release day with per-token prices; Qwen2.5 within 11–15 days; Qwen2 within 40; Qwen 1 by the earliest capture, 64–127 days after release. So Qwen3 0.6B is coded as served under the model-level rule [review: this records first-party availability at release; it does not contradict R3's point that most of its use may be local, which the catalog cannot measure]. Qwen2.5 0.5B–3B were listed as "限时免费" (free for a limited time): served, at a zero price.
- **19 of the 26 served models are Alibaba's.** 24 of 26 are common-D members or sit in a memory-tier window. Only Qwen2 0.5B and 1.5B are neither.

**Tests.** The outcome is ln ŵ given ln C and year effects, with a WCR p-value by developer (B = 9,999). File: `rb2_decisions_conduct.csv`.

| Specification | Coef. (CRV1 s.e.) | WCR p | Treated clusters |
|---|---|---|---|
| Models, developer-level code (ra2) | 0.43 (0.27) | 0.49 | 7 of 18 |
| Models, **model-level code** | 0.56 (0.24) | **0.42** | 6 of 18 |
| Models, model-level, ambiguous sizes = 1 | 0.62 (0.24) | 0.23 | 7 |
| Models, model-level, **without Alibaba** | **−0.10** (0.28) | 0.74 | 5 of 17 |
| **Decisions**, any member served | 0.31 (0.25) | 0.64 | 6 |
| Decisions, all members served | 0.48 (0.14) | 0.13 | 4 |
| Decisions, without Alibaba | −0.19 (0.30) | 0.59 | 5 |

**Tier windows.** The windows are 2.4–3.3, 6.5–9.5, 11.5–14.9, 26–32.9 and 65–72.9B parameters. There, w − 1 bounds m_N from **above** (Prop. A8(iii)(b)).
- 49 of 77 models sit in a window; 18 of the 26 served models do.
- Median s is 0.68 inside the windows and 0.86 outside them.
- Under the task's literal windows (≤ 3.3B, ...), 61 of 77 models are in a window.

**On-device contrast (descriptive; 2 developers).** Median s is **0.90** (IQR 0.84–0.92) for the 9 on-device-target models, against 0.70 for server or unspecified targets. At the decision level it is 0.90 against 0.68. [review] R3 reported 0.74 for the 65 server/unspecified models; the difference is the StableLM-Alpha token audit, which moves two models across a gap in the distribution (the 65-model median is fragile: 0.70–0.74).

**Descriptive medians by serving status:**
- served (model level): 0.83;
- served without Alibaba: **0.55** (7 models, 5 developers);
- decisions with any member served: 0.70, against 0.74 for decisions with none served.

**Open-weight premium with common-D families entered once** (production-scale universe, 165 units, 67 developers): **0.46 (0.17; WCR p = 0.018)**, and 0.63 (0.15; p = 0.001) for 2023 and later. At the model level it is 0.51, p = 0.012.

**The conduct reading the data favor.** The evidence points to adoption and memory-tier mechanisms (Prop. A8 cases (a′) and (b)), not to serving-cost internalization:
- coded by model, serving has no detectable relation to over-training: the point estimate is positive (0.56 log points, CRV1 t = 2.3) but the few-cluster p-value is 0.42 with 6 treated developers, and the estimate is Alibaba's (−0.10 without Alibaba) [review: the builder's "does not predict" overstated a null with a positive point estimate];
- the sign of the serving coefficient reverses without Alibaba;
- over-training is highest for on-device targets;
- open weights carry a premium of about half a log point at the decision level.

A developer that bears all of its serving cost would over-train less, not more.

### H3. Lab-own technologies under one rule [E/U]
The rule is the lab's own expansion path plus the common model-free σ* = 0.693 (rb1 study-level). Files: `rb2_decisions_labown_models.csv` and `_labown_summary.csv`.

The paths:
- **Meta**: the 8 bracketed IsoFLOP budgets (a = 0.496; M*(10²¹) = 15.1). Meta's 10-budget planning law is a sensitivity row.
- **Marin**: the Nemotron-CC ladder (7 bracketed budgets).
- **AI2**: the OLMo ladder path, κ free.
- **DeepSeek**: the published law.

The path intervals come from a wild bootstrap-t on the **between-budget** residuals (HC2, Webb weights, B = 9,999). Curvature draws are 0.693 + 0.0126 × t₃ [review; the builder used N(0.693, 0.0126), which understated rb1's HKSJ interval].

| Median s | n | Reference | One rule | Lab curvature | Meta 10-budget law | v2 (ra2) rule |
|---|---|---|---|---|---|---|
| All lab-own models | 22 | 0.65 | **0.51** | 0.67 | 0.48 | 0.61 |
| **Without AI2** | 13 | 0.56 | **0.63** | 0.69 | 0.53 | 0.59 |
| Meta | 10 | 0.49 | 0.57 | 0.62 | 0.46 | 0.51 |
| AI2 | 9 | 0.70 | 0.44 | 0.66 | 0.44 | 0.66 |

- **AI2 drives the gap.** Under one rule the κ-free ladder path puts M* at 82–145 at these models' computes, 4–7 times the reference. With the common curvature, AI2's shares fall from 0.66 (the ladder's own σ*_κ = 0.544) to 0.44.
- **Meta.** On the 8 bracketed budgets, Llama 3 8B has w = 8.27 [4.51, 15.91] and Llama 3.1 405B has 1.46 [0.71, 3.01]. The 405B is "on Meta's path" (w = 0.97) only under the 10-budget planning law, which uses two unbracketed budgets.

### H4. Second output (Bond et al.): task bits per byte against C4 loss, on the same runs [E]
The data are the OLMo ladder's 30 runs, with technologies fitted to each output. The statistic is ln(w_task/w_C4) for the 77 clean models, in OLMo's parameter convention. Inference is a pairs bootstrap by (size, multiplier) cell, with the same resample for both outputs (B = 399). File: `rb2_decisions_second_output_summary.csv`.

| Output | Form | Median ln(w_task/w_C4) [95%] | 10th–90th across models | Median s: C4 → task |
|---|---|---|---|---|
| Task BPB, all tasks | κ = 1 | **−0.11 [−0.78, 0.60]** | [−0.40, 0.21] | 0.22 → 0.13 |
| Task BPB, all tasks | κ free | **−0.37 [−1.17, 1.26]** | [−0.75, 0.05] | 0.75 → 0.64 |
| MMLU (knowledge) | κ = 1 / free | 0.01 / 0.14 | | |
| ARC, OBQA (science QA) | κ = 1 / free | 0.17 / −0.42 | | |
| Commonsense (5 tasks) | κ = 1 / free | −0.07 / −0.34 | | |
| BoolQ | | degenerate κ = 1 fit (M* in the millions); not interpretable | | |

- σ*: C4 0.825 (κ = 1) and 0.544 (κ free); task 0.763 and 0.512.
- **Reading.**
  - The output-concept contamination is not detectable on this single recipe. Every 95% interval of the cross-model median includes 0, and only 4% of models have a model-level interval excluding 0 (all-task BPB; 10% for the science-QA and commonsense families under κ = 1 [review]).
  - The point estimates say that a task-optimizing developer's wedge is 11–31% smaller than the C4-based one: ŵ_C4 overstates w_task by e^{0.11}–e^{0.37}.
  - This moves the median s by about 0.1 and the share with w > 1 by 0.12–0.14.
  - It varies with task family, and neither its sign nor its size is stable across forms.
  - It is one recipe (AI2's) at small scale (190M–3.2B, M ≤ 200). It bounds the size of the critique's plausible effect; it does not validate the output concept.

### H5. Sign identification (Prop. 5 / PI-1) with honest anchor intervals [E]
Files: `rb2_decisions_anchors.csv`, `_sign_identified.csv`, `_sign_magnitude.csv`, `_mstar_bounds.csv`.

**Anchors.** Each is the path of ra1's bracketed per-budget minima at the design's largest bracketed budget, with a wild bootstrap-t on the between-budget residuals.

| Design | Budgets | Anchor | M* [95%] |
|---|---|---|---|
| Chinchilla | 9 | 3×10²¹ | 20.7 [12.4, 34.4] |
| Llama 3 | 8 (R1 minor 7) | 10²¹ | 15.1 [9.7, 23.1] |
| Marin DCLM | 7 | 3×10²⁰ | 13.9 [9.9, 19.6] |
| Marin Nemotron-CC | 7 | 3×10²⁰ | 13.8 [9.5, 20.0] |
| DeepSeek (published law) | | | point |

- For comparison, the ra2 intervals were 18.2–29.4 (Chinchilla) and 20.8–23.7 (Llama 3 at 10²²); ra5's bootstrap-t gave [12.2, 40.2] for the Llama 3 anchor at 10²².
- e ∈ [−0.156, 0.162].
- Identified set: M*(10²⁴) ∈ **[2.7, 88]** (ra2: [2.3, 89]); M*(10²⁵) ∈ [1.9, 128].
- Marin Comma is excluded. It has only 5 bracketed budgets, and its bootstrap-t interval is [0.1, 3,571]. The ≥ 6-budget rule was set after seeing this interval, so it is disclosed; including Comma leaves nothing identified (8%).

**Share with w > 1 identified, by tilt allowance τ on M\*:**

| | τ = 1 | 1.3 | 1.84 | 3.4 | 4.4 |
|---|---|---|---|---|---|
| Models, unweighted | **0.87** | 0.86 | 0.82 | 0.69 | 0.68 |
| Models, compute-weighted | **0.48** | 0.48 | 0.34 | 0.23 | 0.23 |
| Decision units, unweighted | 0.82 | 0.80 | 0.77 | 0.63 | 0.61 |
| Decision units, compute-weighted | 0.41 | 0.40 | 0.32 | 0.14 | 0.14 |
| Models, compute-weighted, 2023 / 2024 / 2025 | 0.20 / 0.40 / 1.00 | 0.13 / 0.40 / 1.00 | 0.12 / 0.31 / 0.62 | 0.05 / 0.19 / 0.49 | 0.05 / 0.19 / 0.49 |

**Other assumption sets:**
- **Normal inputs only** (e ∈ [−1, 1]), harmonized sample: **0.14** of models (0.16 of decision units); compute-weighted 0.003.
- **e widened** to the path slopes' 95% intervals: 0.71.
- **Own-lab anchors**: 0.87.
- [review] **PI-4, every ex-ante technology's in-support M\* as an anchor** (e ∈ [−0.156, 0.311]): 0.23 of models (ra2: 23%), 0.20 of decision units, 0.03 of compute; 0.14 of models at τ = 1.84 (R2 Major 4.3 asked for this number beside the 87%).
- **Meta's 10-budget anchor**: unchanged, because Chinchilla's anchor sets the upper envelope.

**Magnitude.** The median lower bound on s is 0.57 (τ = 1), 0.52 (1.3), **0.45 (1.84)**, **0.30 (3.4)** and 0.23 (4.4). The upper bound runs from 0.93 to 0.97 (k ∈ [0.40, 0.52]), and reaches 0.97–0.99 with k up to 0.667 (σ* = 0.60). rb1_sigmaC carries σ*(C) into these bounds.

### H6. The trend from 2023, under one set of rules [E]
Files: `rb2_decisions_trend_by_tech.csv`, `_trend_monotone.csv`, `_trend_lodo.csv`.

| | 2023 | 2024 | 2025 | Rises in 2023→24 and 2024→25 under |
|---|---|---|---|---|
| Median s, models (reference; lab-own) | 0.55; 0.56 | 0.82; 0.82 | 0.87; 0.85 | 31 of 32 technologies |
| Median s, decision units | 0.49; 0.51 | 0.81; 0.79 | 0.84; 0.80 | 28/32 |
| Compute-weighted s, models | 0.27; 0.30 | 0.58; 0.61 | 0.83; 0.82 | 32/32 |
| Compute-weighted s, decision units | 0.25; 0.28 | 0.54; 0.56 | 0.81; 0.80 | 32/32 |
| Range over 32 technologies (weighted, models) | [0.01, 0.58] | [0.05, 0.74] | [0.24, 0.96] | |
| Leave one developer out (weighted, models) | [0.23, 0.40] | [0.48, 0.75] | [0.70, 0.87] | |
| n models (decision units) | 24 (19) | 40 (30) | 13 (7) | |
| **Median M** | **146** | **1,217** | **2,438** | |

- The **rise holds under every one of the 32 technologies** for the compute-weighted shares, and under 28–31 of 32 for the medians.
- The exceptions are flat from 2024 to 2025: the three Marin primal fits and OLMo κ = 1 for decision units; Marin Comma primal for models.
- **The trend largely restates the rise in M:** the median M rose 17-fold from 2023 to 2025.
- The 2024 weighted share is dominated by Llama 3.1 405B (55% of 2024 compute). Its sign is not identified.
- The compute-weighted identified share rises from 0.20 to 0.40 to 1.00 (H5), so part of the weighted rise is a shift in composition.
- The 2025 cohort is dense-only, with 13 models in 7 decision units.
- [review] **Leave one developer out.** The compute-weighted shares rise in both steps in all 18 leave-one-developer-out samples. The medians do not: the model median is flat from 2024 to 2025 without Alibaba (0.82 → 0.81), and the decision-unit median falls from 2024 to 2025 without Alibaba (0.82 → 0.75), Hugging Face (0.76 → 0.75) or Marin (0.81 → 0.76); 2025 then has 6 decision units. The robust statement is the 2023 → 2024 rise and the compute-weighted rise; the 2024 → 2025 step of the medians rests on one or two developers.

### H7. Before 2023: does s ≈ 0 for the right reason? [E]
File: `rb2_decisions_pre2023.csv`. The sample is the 2019–2022 open-weight production-scale universe that ra2/ra3 used (8 models).

| Sample | Technology | Median s | Compute-weighted s |
|---|---|---|---|
| All 8 | Reference (the v2 "near zero") | 0.22 | **0.04** |
| All 8 | Kaplan belief (Kaplan path N ∝ C^0.73 anchored at GPT-3; reference curvature) | 0.52 | **0.19** |
| All 8 | Kaplan belief, Kaplan joint-law curvature (σ* = 0.535) | 0.77 | 0.43 |
| OPT-175B and BLOOM-1.7B dropped (clean rule b1) | Reference / Kaplan belief | 0.22 / 0.52 | 0.06 / 0.29 |

- Under the believed technology of Prop. 2(iii), **OPT-175B has w_K = 1.00 (s = 0)** "for the right reason", but by construction: it replicates GPT-3, the anchor of Kaplan's path.
- GLM-130B has w_K = 1.28 (s = 0.22). The four small models (Grover, AraGPT2, EMDR, BLOOM-1.7B) are over-trained under either technology.
- **The v2 "near zero" was the truncation at w = 1 of two Kaplan-era flagships** (92% of the period's compute), not a measured absence of value.
- Under the belief correction the period's weighted share is 0.19–0.29, not ≈ 0. **Drop the pre-2023 baseline** (R3 N2a; R4 R2-M1).

### H8. Post-training compute acts like δ > 0 and is small [E/L]
File: `rb2_decisions_posttrain_*.csv`.

**Disclosed budgets.** x_P is post-training compute divided by the same model's pre-training compute.
- DeepSeek-V3: **0.0019** (5K of 2,664K H800 GPU hours).
- Tülu 3 SFT on Llama 3.1 8B and 70B: 0.0004 and 0.0007 (FLOP estimate).
- Olmo 3 Think 32B:
  - RL alone: **0.026** (5 days on 224 GPUs);
  - with the 21-day Olmo 3.1 continuation: **0.135**;
  - upper bound for all post-training: **0.21** (9 days of the whole 1,024-GPU cluster).

**Effect.** Since w = 1 + x_P + m_N:
- median s falls from 0.749 to 0.740 (x_P = 0.135) and to 0.735 (0.21);
- the share with m_N > 0 falls from 0.974 to 0.948 and 0.922;
- with 0.135 for 2025 releases and 0.002 before, the 2025 compute-weighted share falls from 0.83 to 0.81.

It matters only for flagships with w near 1, and it does not change the shape of the trend.

[review] Two qualifications. (i) The 0.21 "upper bound" covers the ~9 post-training days before the initial Olmo 3 release; the 21-day Olmo 3.1 RL continuation came after them, so Olmo 3.1 Think 32B's total is at most (9 × 1,024 + 21 × 224)/43,264 GPU-days = **0.32**. At x_P = 0.32 the median s is 0.727 (models) and 0.711 (decision units), the share with m_N > 0 is 0.91, and the 2025 compute-weighted share is 0.77. (ii) The aggregate divides by (1 + m + x_P), the share of compactness in pre- plus post-training cost plus compactness; the medians use m/(1 + m). With m/(1 + m) throughout, the 2025 aggregate at x_P = 0.32 is 0.82. Either way the conclusion stands: post-training compute is second order for the median and trims the 2025 aggregate by at most about 0.06.

### H9. The "ex ante" wording: the facts
- **Rules.** The cleaning steps (a)–(h) implement the round-1 referee lists: R1 comments 1–2 and minors 24–25; R2 Majors 4–5 and 10; R3 M5.5–6. The docstring of `ra2_wedge/sample.py` cites them.
  - The round-1 referee reports were committed at 03:12 on 2026-09-24 (bd5c0ad). [review correction] The revision plan (`paper/notes/revision_plan.md`) was **not** in that commit: it was first committed at 07:48 with ra2 (acc8286). Its file modification time is 03:13:37, which is consistent with its having been written right after the reports, but a modification time is not a commit record. The plan names "clean inference-demand sample" and "all technologies under an ex-ante rule" without enumerating the steps.
  - [review] `ra2_wedge/sample.py`, which encodes the rules, was last modified at 06:42:32, before the logged ra2 run (07:09:53). Earlier, unlogged ra2 runs may have computed wedges; the file times cannot exclude that.
  - ra2's code and outputs were committed together at 07:48 (acc8286). Its logged full run started with the sample stage at 07:09:53, and the wedges followed at 07:10:24.
- **What git cannot show.** Git cannot establish the order within the ra2 session. The lead author's statement that the rules were fixed before the v2 wedges were recomputed is consistent with the record but not provable from it.
- **Version 1** (m3, b4866cd, 01:40) had already published wedges for all 173 verified models (median 3.19, κ = 1).
- **Technologies added after round 1: 20 of the 32** (`rb2_decisions_technologies_units.csv`, column `in_version1`). They are:
  - Chinchilla κ-free (the reference) and the Chinchilla non-embedding refit;
  - Gadre ×3 κ-free, OLMo κ-free, Muennighoff κ = 1 and κ-free;
  - the Meta model-free path;
  - Marin ×9;
  - the DeepSeek and MiniCPM laws.
- **The 12 in version 1:** Chinchilla (κ = 1), Besiroglu, Hoffmann, Farseer ×3, Gadre ×3 (κ = 1), OLMo (κ = 1), and Meta A2 and A3.
- **Suggested sentence:** "The sample rules implement the round-1 referee reports and were fixed before the version-2 wedges under the new reference were computed; version 1 had reported wedges for all 173 verified models under 12 of the 32 technologies."

---------------------------------------------------------------------------------------------------

## 2. Methods
- **Base.** ra2's `sample.build`, `modelfree.run_all` and `techs.build` are imported. The reference is κ-free Chinchilla with a design-conditional wild bootstrap (B = 399). The replication is exact: the median w on the 77 models is 3.986234596660721, and the maximum absolute difference from `ra2_wedge_models.csv` is 3.6×10⁻¹⁵ before the token audit.
- **Readings** (`evidence.py`). Primary sources are saved with a manifest (URL, retrieval time, sha256). Quotes are matched after removing whitespace. Each reading follows the definitions above; evidence on both margins gives cap+menu.
- **Serving** (`evidence.py`). The rule is described in H2. Every row carries its evidence origin: the model's own page, the same release, the same developer, or the developer coding. An assertion stops the run if any served = 1 row has a lag above 180 days or below −60 days. DeepSeek LLM's sample date (2024-01-05) is the report date; the weights came out on 2023-11-29.
- **Decision units** (`decisions.py`).
  - Family objects are recomputed in every bootstrap draw, so the intervals of family shares and unit medians are joint.
  - Lab-own draws: the path bootstrap-t "adjusted draws" θ̂ − t*·ŝe reproduce the equal-tailed bootstrap-t interval and keep the cross-model dependence. They are combined with normal curvature draws.
  - Where lab-own and reference draws are mixed, draw indices are aligned modulo the number of draws. [review] The joint B is 3 × 399 = 1,197, an exact multiple of the reference's draws, so every reference draw has equal weight.
- **Conduct.** ra2's `wcr` (restricted wild cluster bootstrap-t, Webb weights, B = 9,999, clustered by developer). Decision units carry:
  - the family ln W_f and log total compute;
  - the first release year;
  - served = any, all, or flagship.

  For the open-weight premium, ra2's production-scale universe (with the token audit applied) is used, with each clean common-D family collapsed to one row over its members in the universe.
- **Sign** (`paths.py`, `signid.py`).
  - Path: OLS of ln M*_k on ln C_k over ra1's bracketed budgets, with wild bootstrap-t (HC2 residuals and s.e., Webb weights, B = 9,999). This is copied from `ra5_theory/partial_id.py`, where simulated coverage was about 90%.
  - The union of anchors is taken with ra2's `pi_bounds`.
  - Tilt allowance: w > 1 is identified iff (ln M − upper bound) > ln τ.
  - Magnitude bounds widen the set by ln τ on both sides.
- **Trend.** Clean sample, 2023–2025:
  - medians over models and decision units (units dated by first release);
  - compute-weighted s = Σ(w⁺ − 1)C/Σw⁺C;
  - leave one developer out;
  - all 32 ex-ante technologies.

  Kaplan belief: ln w_K = k[ln M − ln M*_K(C)], with M*_K(C) = C/(6N_K²) and N_K = 174.6B·(C/3.14×10²³)^0.73 (m5_progress's anchor).
- **Second output.** m2's OLMo-ladder loader. Task-family BPB is the mean of the validation splits. Fits use m2's Huber estimator (κ = 1, DEFAULT grid plus level starts) and κ-free `fit_q`. The bootstrap is a pairs bootstrap by cell with the same indices for all outputs, warm-started.
- **Post-training.** x_P from disclosed GPU time or FLOPs; m_N = w − 1 − x_P.

## 3. Inventory
| File | Content |
|---|---|
| `output/tables/rb2_decisions_table2.tex` (+ `_table2.csv`) | **Main Table 2**, rebuilt around decisions. Panel A: 11 common-budget families (reading, served k/n, tier k/n, M/M*, s_f [95%], lab-own, band over 32 technologies, sign identified at τ = 1/1.84). Panel B: OLMo 2 and SmolLM2 members. Panel C: 5 singletons. Panel D: medians and shares over decision units. Label `tab:wedge`. |
| `output/figures/rb2_decisions_ecdf.pdf/.png` | **Replaces Figure 4.** (a) ECDF of s over the 56 decision units: 32 technologies in gray, reference in blue, lab-own (one rule) in orange. (b) Reference ECDFs for models (77), decision units (56) and reading-consistent units (66). |
| `output/tables/rb2_decisions_readings.tex` | Appendix: readings of the common budgets, with one verified quote each. |
| `output/tables/rb2_decisions_conduct.tex` (+ `_conduct.csv`, `_open_premium.csv`, `_serving_descriptive.csv`, `_serving_composition.csv`) | Appendix: serving tests (model and decision level), open-weight premium at the decision level, descriptive medians (served, tier, on-device). |
| `output/tables/rb2_decisions_sign.tex` (+ `_sign_identified.csv`, `_anchors.csv`, `_sign_magnitude.csv`, `_mstar_bounds.csv`) | Appendix: sign identification by τ, unweighted and compute-weighted, models and decisions, by year; anchors; magnitude bounds; M*(C) sets. |
| `output/tables/rb2_decisions_trend.tex` (+ `_trend_by_tech.csv`, `_trend_monotone.csv`, `_trend_lodo.csv`, `_pre2023.csv`, `_pre2023_models.csv`) | Appendix: the trend from 2023 and the pre-2023 robustness check. |
| `output/tables/rb2_decisions_second_output.tex` (+ `_second_output_{summary,technologies,models}.csv`) | Appendix: the Bond et al. second-output test. |
| `output/tables/rb2_decisions_labown.tex` (+ `_labown_summary.csv`, `_labown_models.csv`) | Appendix: lab-own technologies under one rule. |
| `output/tables/rb2_decisions_posttrain.tex` (+ `_posttrain_disclosed.csv`, `_posttrain_effect.csv`) | Appendix: post-training compute. |
| `output/tables/rb2_decisions_headline.csv`, `_technologies_units.csv` | Headline statistics and variants; decision-unit medians under each technology, with provenance (version 1 or added later). |
| `data/processed/rb2_decisions/family_readings.csv` | **Evidence**: 11 families × up to 3 quotes, URLs, retrieval times, reading, D margin, N margin, what is revealed. |
| `data/processed/rb2_decisions/serving_model_level.csv` | **Evidence**: 77 models, with model-level code, developer code, channel, demo flag, ambiguity flag, evidence date, lag, URL, retrieval time, verified quote. |
| `data/processed/rb2_decisions/decision_units.csv`, `units_{primary,reading,subgroup}.csv`, `clean_models.csv`, `pi_bounds_models.csv`, `d_audit.csv`, `headline.json`, `run_log.txt`, `techs_cache.pkl`, `exhibits_cache.pkl` | Working files. |
| `data/raw/rb2_decisions/` (+ `manifest.json`) | 65 saved primary sources (PDF/HTML/MD plus text renderings): technical reports, release posts, model cards, Wayback captures of Alibaba's catalogs, the DeepSeek API docs, and VentureBeat on DeepSeek Chat. |
| `code/analysis/rb2_decisions/` | `run.py`, `rb2common.py`, `base.py`, `sources.py`, `fetch_sources.py`, `evidence.py`, `labown.py`, `paths.py`, `decisions.py`, `serving.py`, `signid.py`, `trend.py`, `second_output.py`, `posttrain.py`, `exhibits_rb2.py`, `cluster_ref.py` [review]. |
| `lit/bib/extra_round3_rb2_decisions.bib` | New verified references: young2024yi, apertus2025apertus, olmo2025olmo3, lambert2024tulu3, granite2024granite, mosaicml2023mpt7b, mosaicml2023mpt30b, stability2023stablelm. |

## 4. Claims for the paper (with caveats)
1. **Over allocation decisions, the median open-weight release was built as if the value of compactness were about three times training cost (w − 1 ≈ 2.8), under the reference.** In share terms, s = 0.74 [0.67, 0.78] over 56 decision units (wild; [0.53, 0.81] under ra1's budget-plus-trunk cluster bootstrap [review]); 0.73 under lab-own technologies; 0.75 over reading-consistent units. For non-serving developers, do not call this a share of "lifetime cost" (R3 N2c).
   - *Caveat:* the level is conditional on the technology (0.06–0.81 across the 32 technologies). It is not an abstract claim.
2. **Families that train every size on one budget reveal one number.** Llama 3 herd 0.29 [0.08, 0.46]; its flagship's sign is not identified.
   - *Caveat:* the technical report supports a data-cap reading. Under it, 0.29 is a lower bound on the herd's π-weighted value [review], and the 8B's s ≥ 0.85–0.88 is a lower bound, not a revealed value, provided the 8B's size was not set by the 6.5–9.5B memory tier it sits in [review].
3. **Showcase (replaces Llama 3 8B): SmolLM2.** It has size-specific budgets (135M/2T, 360M/4T, 1.7B/11T), no tier window, was not served, and has an on-device target.
   - Reference s = **0.94 [0.92, 0.95], 0.93 [0.91, 0.95], 0.91 [0.88, 0.93]**. The sign is identified for all three, even at τ = 1.84.
   - Band over 32 technologies: [0.51, 0.996], [0.48, 0.99], [0.42, 0.99]: w > 1 under every ex-ante technology.
   - [review] *Caveat:* SmolLM2 carries the synthetic-data flag (`f_synthetic`; Cosmopedia, FineMath, Stack-Edu), and the paper's own robustness row drops such models (median 0.70). The reference technology is estimated on natural text, so SmolLM2's level is more exposed to a data-quality tilt of M\* than OLMo 2's; its sign is not (identified at τ = 1.84, positive under all 32 technologies). The table marks it with a dagger. Use SmolLM2 for the sign and the ordering and OLMo 2 (natural data, lab-own path) for levels.
   - Secondary showcase, **OLMo 2** (1B/4.05T, 7B/4.05T, 13B/5.15T, 32B/6.15T):
     - reference s = 0.87, 0.75, 0.70, 0.61;
     - AI2's own ladder path with the common σ* gives 0.80, 0.52, 0.40, 0.12 (w 4.90 [3.60, 6.90], 2.10 [1.74, 2.65], 1.66 [1.39, 2.04], 1.14 [0.97, 1.38]).
   - *Caveats:*
     - For both families s is the internalized value of compactness, not planned serving expenditure: neither Hugging Face nor AI2 served these sizes; AI2 offered a research demo.
     - OLMo 2 7B, 13B and 32B sit in tier windows, so their w − 1 is an upper bound.
     - OLMo 2 1B and 7B share 4.05T but were released five months apart.
4. **The serving reading has no support.** Coded by model, 26 models are served, 19 of them Alibaba's. Serving has no detectable relation to ŵ: 0.56, p = 0.42 (positive point estimate, 6 treated clusters); −0.10 without Alibaba; decision level 0.31, p = 0.64. [review] Write "no detectable relation", not "does not predict".
   - Over-training is highest for on-device targets (0.90 against 0.70), and open weights carry a premium (+0.46, p = 0.018, with families entered once).
   - The data favor adoption and tier mechanisms. **Remove "for developers that serve them, lower serving cost" from the abstract.**
5. **Sign.** w > 1 is identified beyond the designs for 87% of models (82% of decisions) with no tilt allowance.
   - With allowances: 82% at 1.84 and 69% at 3.4.
   - It covers only 48% of training compute (34% at 1.84).
   - By year, compute-weighted: 20% (2023), 40% (2024), 100% (2025).
   - Normal inputs alone identify 14%; admitting every ex-ante technology as an anchor (PI-4), 23% [review].
   - *Caveat:* do not write "weak assumptions". Name the anchors, the e range and τ. The sign-identified 2023–2024 compute is small.
6. **Trend.** From 2023 to 2025, on the clean sample, the value of compactness rose steeply under all 32 technologies (compute-weighted, both units) and under 28–31 of 32 for the medians.
   - Reference: 0.27 → 0.58 → 0.83 (compute-weighted models); 0.49 → 0.81 → 0.84 (median decisions).
   - *Caveat:*
     - This largely restates the rise in M (median 146 → 2,438).
     - 2024 is dominated by the 405B, whose sign is unidentified.
     - 2025 has 7 decision units and is dense-only.
     - Do not quote "near zero before 2023".
7. **Output concept.** On the OLMo ladder's own runs, a task-bits-per-byte technology implies wedges 11–31% smaller than the C4-loss technology at the median. None of these differences is statistically detectable.
   - *Caveat:* one recipe, small scale, M ≤ 200. It bounds the critique; it does not validate the output concept.
8. **Post-training compute** is 0.2–13.5% of pre-training compute in disclosed budgets; 21% bounds Olmo 3's initial post-training and 32% Olmo 3.1's including the RL continuation [review]. It lowers the median s by at most 0.02 (models; 0.03 over decision units) and the 2025 compute-weighted share by at most about 0.06.

## 5. Robustness
- **Unit definitions.** The decision-unit median s is 0.72–0.75 across the primary, reading-consistent and sub-group definitions, under the reference and lab-own. The model-level median is 0.75.
- **Serving code.**
  - With ambiguous sizes coded as served, the coefficient is 0.62 (p = 0.23).
  - With an on-device control, 0.58 (p = 0.38).
  - Without Qwen2.5/Qwen3, 0.17 (p = 0.61).
  - Decision level with "all members served": 0.48 (p = 0.13; only 4 treated clusters).
- **Anchors.**
  - Including Marin Comma (5 bracketed budgets) destroys identification (8%). It is excluded by the ≥ 6-budget rule, which was chosen after inspection and is disclosed.
  - Meta's 10-budget anchor at 10²² changes nothing, because Chinchilla sets the envelope.
  - Widening e to the path-slope 95% intervals gives 71% of models (τ = 1).
- **Lab-own.** Medians are 0.44–0.69 across the rules (H3). The one-rule median without AI2 is 0.63.
- **Common σ*.** It is read from rb1 at run time (0.6929). With ra1's 0.695 the lab-own values move by about 2%. An early run used ra1's value; the Llama 3 8B lab-own w was 8.12 against 8.27 now.
- **Token audit.** StableLM-Alpha's corrected counts change no median. The audit table is kept.
- **Tier windows.** Under the literal windows, 61 of 77 models are in a window (49 under ra2's windows), and 23 of 26 served models.

## 6. Referee comments addressed
**R1 (IO econometrician)**
- **New 1(a) Llama 3 showcase.** Recast as the herd's family share 0.29, with the flagship's sign unidentified. Under the cap reading, the 8B's w − 1 is a lower bound (H1). Showcase replaced by SmolLM2, with OLMo 2 as secondary (Claim 3).
- **New 1(b) Table 2 flags.** Table 2 is rebuilt around decision units. It shows the reading, served k/n, tier k/n, family shares, lab-own values, the band and sign identification.
- **New 1(c) serving group.** Model-level code: 26 served (19 Alibaba). 24 of 26 are common-D or tier-window; only Qwen2 0.5B/1.5B are neither. Tier-window wedges are upper bounds. Results are shown without Alibaba (H2).
- **New 1(d) model-level serving.** Coded with dated evidence (`serving_model_level.csv`). Tests re-run at the model and decision levels.
- **New 1 request 2.** Headline on size-specific members and singletons: 0.75 (45). Outside tier windows: 0.85 (19), matching R1's count.
- **New 1 request 4.** The reading the data favor (adoption and tiers, not serving-cost internalization) is stated (H2).
- **New 3.** Anchors widened by wild bootstrap-t on the between-budget residuals. Share identified reported by τ ∈ {1, 1.3, 1.84, 3.4, 4.4}: unweighted, compute-weighted, by year, and for decisions. Normal-inputs-only share on the harmonized sample: 14% (H5).
- **New 4.** Trend from 2023 on the clean sample under one set of rules: unweighted, decision-level, compute-weighted, leave one developer out, 32 technologies. Pre-2023 only as robustness under the Kaplan belief (H6, H7). Recommended abstract language in Claims 5–6.
- **New 5.** Second-output test on the OLMo ladder, including task families (H4).
- **Minor 7.** Llama 3 anchor moved to the largest bracketed budget, 10²¹ (8 budgets). The 10-budget law is a flagged sensitivity.
- **Minor 10.** Facts behind "ex ante"; the 20 technologies added after round 1 are listed (H9).
- **Minor 16.** Qwen token counts. Qwen3 is documented (all sizes ≥ 30T in S1 of a 36T corpus). Qwen2.5 states only "up to 18T", flagged D_unaudited, with robustness rows that drop both families. The audit also found and fixed StableLM-Alpha's counts.

**R2 (ML scaling)**
- **Major 2.** The cap case is applied: lower bounds (Prop. A8 with a data cap). Every common-D family has a documented reading (H1). Table 2 is rebuilt. The headline is over decision units.
- **Major 4.1–4.3, 4.5.** The level claim is conditioned. "Weak assumptions" is dropped, and the sign result names its assumptions (H5). The trend is ordinal and restates M (H6).
- **Major 4.4.** "Ex ante" facts (H9).
- **Major 6.** One rule for lab-own technologies: lab path plus common σ*; lab curvature as sensitivity; median without AI2 = 0.63. Meta's path is on the 8 bracketed budgets, with the 10-budget law flagged (H3).
- **Major 7.** Post-training compute bounded from disclosed budgets (H8). The loss-versus-capability point is partly covered by the second-output test (H4). The quantization paragraph belongs to the writers.
- **Minor 16.** Robustness row without Qwen2.5/Qwen3 (0.736).

**R3 (editor)**
- **N2.** Pre-2023 baseline explained (Kaplan-era truncation) and dropped. The trend is stated within 2023–2025 under all 32 technologies (H6, H7).
- **N3(a–c).** Table 2 rebuilt around decisions. Llama 3 narrative replaced. Conduct tests re-run at the decision level (H1, H2).
- **N4(a).** Model-level serving code and tests (H2).
- **N4(b).** Remove the serving parenthetical (Claim 4).
- **N4(d).** On-device contrast reported descriptively (0.90 against 0.70; 2 developers).
- **Minor 14.** Common-D flag, family share and model-level serving in Table 2.

**R4 (auditor)**
- **R2-M1.** The eight-model baseline is examined. Dropping the suites gives 0.058. Under the Kaplan belief it is 0.19–0.29, with OPT-175B on Kaplan's path by construction (H7).
- **R2-M2.** Level claims are conditioned. The robust claims are the rise (32/32) and the sign (87%, with its assumptions).
- **R2-M3(1–2).** The anchors and lab-own path intervals are re-run with between-budget residuals (wild bootstrap-t). Llama 3 lab-own 8B is 8.27 [4.51, 15.91], and the 405B 1.46 [0.71, 3.01].
- **R2-M3(c).** Llama 3 anchor at its largest bracketed budget.
- **R2-M3(d) / request 3** [review]. Headline under ra1's budget-plus-trunk cluster bootstrap: 0.736 [0.526, 0.808] over decision units (Table 2, Panel D).

## 7. Open issues
1. ~~R4 R2-M3(d)~~ [review: done]. `cluster_ref.py` refits κ-free Chinchilla on ra1's 44 budget-plus-trunk clusters (B = 999, no failed fits): decision-unit median s 0.736 [0.526, 0.808], models 0.749 [0.575, 0.827], share of units with w > 1 [0.839, 1.000]. Table 2, Panel D shows the interval.
2. **MoE in the 2025 aggregate** (R3 N2, smaller point). Not added. The trend is labelled dense. R3's own calculation moves 2025 only to 0.78–0.82.
3. **Qwen2.5 tokens per member** remain unaudited: the report says "up to 18T".
4. **Nominal parameter counts** remain for MPT-7B (6.7B per the blog), DeepSeek LLM 7B/67B, BTLM (2.6B) and others. They affect ln w by about α·Δln N, at most 3%.
5. **Serving ambiguity.** Llama 3 8B/70B in Meta AI and Yi-34B's early-access API are unresolved. Both codings are reported.
6. **The "ex ante" order** within the ra2 session cannot be proven from git. The writers should use the suggested sentence only with the lead author's confirmation.
7. **The ≥ 6-budget anchor rule** was set after inspecting Marin Comma's interval. It is disclosed, with the sensitivity reported.
8. **Second output.** Single recipe; BoolQ is degenerate. A second recipe with per-task losses, such as DataDecide final checkpoints, would help.
9. **rb1 dependency.** The common σ* is read at run time. If rb1 is re-run with a different study-level value, re-run this module. Only the lab-own columns change.
10. **Hugging Face serving.** HuggingChat and the serverless Inference API may host SmolLM models. They are not documented at release and are coded 0; this is a platform, not first-party serving of the developer's own product.
11. [review] **Lab-own paths are anachronistic for Llama 1/2.** Meta's Llama 3 IsoFLOP path (2024, Llama 3 data) is applied to Llama 1 and Llama 2 (2023). Under Prop. 2(iii) the believed technology at the time was not this path; the lab-own medians for Meta mix a recipe correction with a belief assumption. ra2's convention, kept here; flag it in the text.
12. [review] **Cap-and-tier conflicts.** 9 of the 15 cap members sit in tier windows, where the two constraints bound m_N in opposite directions. Reported as a variant (57 units, 0.749); the readings of Qwen2.5 and Qwen3 as caps are judgment calls (the primary units use their family shares either way).

**Citation keys used**
- **Existing:** grattafiori2024llama, yang2025qwen3, qwen2024qwen25, bi2024deepseek, touvron2023llama2, touvron2023llama, yang2024qwen2, benallal2025smollm2, gemmateam2024gemma, teamolmo2024olmo, deepseekai2024deepseekv3, bhagia2024establishing, kaplan2020scaling, bond2020unpleasant (Bond, Hashemi, Kaplan and Zoch 2021; in paper/references.bib) [review: the builder listed bond2021some from lit/bib/io_controlfn.bib, which is not an allowed bib file; use bond2020unpleasant].
- **New** (`lit/bib/extra_round3_rb2_decisions.bib`, verified): young2024yi, apertus2025apertus, olmo2025olmo3, lambert2024tulu3, granite2024granite, mosaicml2023mpt7b, mosaicml2023mpt30b, stability2023stablelm.
