# Memo: module rb5_units, budget-level decision units and the revealed value of compactness (round 3, WP4b analysis)

Module owner: WP4b analysis agent (coder A). Date: 2026-09-25. Binding specification: `paper/notes/round3_fixlist.md`, items W1, W2, W14, W18, W19 (analysis parts) and W24 (data corrections), decisions D-1 and D-2. No paper section or paper table was edited; the paper text is WP4b's second part. [review] The independent review is `output/memos/rb5_units_review.md` (package WP4b-review); passages it changed are marked [review].

**Status of the readings.** [review] Every number below now uses the **resolved codes** (`data/processed/rb5_units/readings_final.csv`, written by `code/analysis/rb5_units/resolve.py`): the blind second coding (coder B, `readings_coderB.csv`) agrees with coder A on 15 of the 17 readings (Cohen's kappa 0.83), and the six unit-field disagreements were resolved under protocol Section 11 (review memo, Section 2). The resolved readings equal coder A's, so no identification class count changes; the rationale flags of DeepSeek LLM (R-CO 1 to 0) and StableLM 2 1.6B (R-INF 0 to 1) change. Both modules were re-run from scratch on the resolved codes (twice; outputs identical) and record `readings_final.csv` as their codes file.

**How to run** (CPU only, one or two processes, deterministic):
1. `nice -n 10 .venv/bin/python code/analysis/rb5_units/verify_readings.py` (coding frame codes, quote check, `readings_coderA.csv`); [review] then `resolve.py` (both coders' files to `readings_final.csv`);
2. `nice -n 10 .venv/bin/python code/analysis/rb2_decisions/run.py` (about 10 minutes on 2 processes; `--exhibits-only` redraws Table 2, Figure 4 and the appendix tables from the cache in seconds);
3. `nice -n 10 .venv/bin/python code/analysis/rb5_units/run.py` (about 6 minutes with B = 9,999; `--reuse` reuses the cached randomization-inference and bootstrap draws).

Notation: w = ε_N/ε_D; s = (w − 1)/w, the compactness share; v = w − 1, the virtual value of compactness (decision D-1); a decision unit is one choice of (N, D).

---------------------------------------------------------------------------------------------------

## 1. Findings

### F1. Budget-level units (W1(a)): 49 decisions, and the headline does not move
- Members of one family whose token budgets are within 10 percent form one decision (the rule of D-2). The clean sample's 77 models form **49 units**: 17 common-budget units, 17 size-specific members and 15 singletons. The partition is identical to version 3's "sub-group" units (checked in code).
- The six new common-budget units are LLaMA 7B and 13B (1.0T), LLaMA 30B and 65B (1.4T), Qwen 14B and 72B (3T), Qwen2 1.5B, 7B and 72B (7T), SmolLM 135M and 360M (0.6T), and OLMo 2 1B and 7B (4.05T).
- Under the reference technology the median decision has **w = 3.77 and s = 0.735**, with a joint wild-bootstrap interval of **[0.657, 0.782]** and a budget-plus-trunk cluster-bootstrap interval of **[0.512, 0.806]**. Stated as a marginal (D-1): the median decision was made as if a model one percent smaller at the same loss were worth **2.8 percent** of its training cost (w − 1 = 2.77).
- **98.0 percent** of decisions are over-trained (w > 1), with a wild interval of [0.937, 1.000].
- The unit definition hardly moves the median (`rb2_decisions_headline.csv`):

  | Units | n | Median s [95%, wild] |
  |---|---|---|
  | Budget level (primary) | 49 | 0.735 [0.657, 0.782] |
  | Family label (version 3) | 56 | 0.736 [0.666, 0.784] |
  | Cap-type budgets split into members (new readings) | 66 | 0.751 [0.694, 0.798] |
  | Version 3's reading-consistent units | 66 | 0.752 [0.703, 0.804] |
  | OLMo 2 1B as its own unit | 50 | 0.736 [0.666, 0.784] |
  | Models | 77 | 0.749 [0.689, 0.794] |
  | Lab-own technologies where available, reference elsewhere | 49 | 0.717 |

### F2. Readings coded under a written protocol (W1(b)-(d))
- The protocol (`paper/notes/rb5_reading_protocol.md`) was written before any unit was coded. It defines the evidence for a cap (C1: the whole available data; C2: repetition; C3: data limited the budget), for a choice (K1: a cost, compute or schedule reason; K2: a larger usable pool; K3: other models trained longer on the same data), and "cap or choice" when only the corpus size is stated. "Still improving", data expansion and "trained beyond compute-optimal" are not evidence on the token margin. Three dated amendments were added during coding and before any outcome was computed (own caps from C1/C3 statements, prompted by the Falcon report; the frame's budget when the saved source gives only the corpus size; stated benefits of size count as rationales).
- All **139 quotes** are verbatim and verified against the saved sources in `data/raw/rb2_decisions/` (whitespace removed, case folded); URLs and retrieval times are in `readings_coderA.csv`.
- Readings of the 17 common budgets (reference s_f; the class follows from protocol Section 8):

  | Unit | Reading (rule) | Class | s_f [95%] |
  |---|---|---|---|
  | LLaMA 7B, 13B | choice (K3, K2) | point | 0.46 [0.37, 0.55] |
  | LLaMA 30B, 65B | cap or choice (size only) | lower (point if choice) | 0.08 [−0.11, 0.23] |
  | Llama 2 | choice (K1) | point | 0.23 [0.06, 0.36] |
  | MPT | menu (K2; single GPU) | point | 0.25 [0.11, 0.36] |
  | StableLM-Alpha | choice (K2) | point | 0.53 [0.44, 0.59] |
  | Qwen 14B, 72B | cap or choice | lower (point if choice) | 0.31 [0.15, 0.43] |
  | Yi | menu (K1, K2; 24G GPU) | point | 0.49 [0.38, 0.59] |
  | DeepSeek LLM | cap+menu (C1; configurations) | lower | 0.18 [0.00, 0.32] |
  | Llama 3 herd | cap or choice | lower (point if choice) | 0.29 [0.08, 0.46] |
  | Qwen2 1.5B, 7B, 72B | menu (K1, K2; portable devices) | point | 0.51 [0.38, 0.61] |
  | SmolLM 135M, 360M | cap or choice (C2 and K1) | lower [review: every member repeated its corpus, so lower also if chosen] | 0.86 [0.83, 0.88] |
  | Qwen2.5 | cap or choice | lower (point if choice) | 0.72 [0.63, 0.78] |
  | Granite 3.0 | cap or choice | lower (point if choice) | 0.85 [0.81, 0.89] |
  | OLMo 2 1B, 7B | choice (K1, K3) | point | 0.77 [0.71, 0.81] |
  | Qwen3 | cap or choice | lower (point if choice) | 0.89 [0.85, 0.92] |
  | Apertus | choice (K2) | point | 0.65 [0.54, 0.73] |
  | Olmo 3 | choice (K2, K1) | point | 0.63 [0.53, 0.70] |

- **Changes from version 3's readings** (all from applying the protocol, not from outcomes):
  - Llama 3 herd and Qwen2.5: cap to **cap or choice**, as the fix list anticipated (R2 minor 3): the sources state only the corpus size.
  - Qwen3: cap to **cap or choice**: data expansion (PDF extraction, synthesis) is corpus building, not a statement that the corpus bound (R1 minor 12).
  - Granite 3.0: choice to **cap or choice**: the dense budget (12T, the dense quote of citations_4 #4) is stated without a pool or a reason; the MoE siblings trained on fewer tokens, which does not show that the dense budget was interior.
  - Yi: cap+menu to **menu** (a point estimate): the report states that the budget was raised to 3.1T "to compensate for the decreased compute flops" and that 3T of filtered data was preferred to 10T unfiltered (K1, K2); "not saturated" is not cap evidence.
  - MPT's choice rests on "1T tokens sampled according to this mix" and the 30B's different proportions of the same subsets; the saved texts do not give the subsets' sizes, so this is the weakest choice code.
- **Own caps (W1(d)).** 12 size-specific members and singletons have their own binding cap: OLMo 2 13B and 32B (1.2 and 1.5 epochs), OLMo 1B and 7B (a second epoch over a 2T corpus), SmolLM 1.7B (1T tokens of a 252B-token corpus), SmolLM2 1.7B (about two epochs), phi-1.5 (150B tokens of a 30B dataset), phi-2 ("multiple passes"), StableLM-3B-4E1T (4 epochs), TinyLlama v1.1 (2T over about 950B), and Falcon 40B and 180B (budgets "partially motivated by constraints over data availability"; Amendment 1).

### F3. Classes under the virtual value (W2)
- **29 decisions are point estimates and 20 are lower bounds** (8 cap-type budgets and 12 members with an own cap). [review] If the six "cap or choice" budgets without repetition were choices, 35 would be point estimates (SmolLM 135M and 360M stay lower bounds: they repeated their 252B-token corpus about 2.4 times, so their own caps bind under either reading; version 1 of this memo counted 36). Under coder B's readings (Granite 3.0 and the Llama 3 herd as choices) 31 are point estimates.
- Median s (reference; wild interval; cluster interval for the point estimates):

  | Units | n | Median s |
  |---|---|---|
  | Point-identified | 29 | **0.737** [0.672, 0.790]; cluster [0.539, 0.824] |
  | Lower bounds | 20 | 0.710 [0.634, 0.770] |
  | ... cap-type budgets | 8 | 0.512 [0.390, 0.620] |
  | ... members with an own cap | 12 | 0.721 [0.659, 0.772] |
  | Point-identified if "cap or choice" is a choice [review] | 35 | 0.735 [0.657, 0.788] |
  | Point-identified under coder B's readings [review] | 31 | 0.737 [0.672, 0.790]; cluster [0.539, 0.824] |
  | Point-identified without MPT and Yi, the weakest agreed choice codes [review] | 27 | 0.745 [0.691, 0.793]; cluster [0.575, 0.830] |
  | Point-identified, without synthetic-flagged units (protocol rule) | 18 | 0.675 [0.607, 0.731] |
  | Point-identified, without the fix list's synthetic families | 25 | 0.678 [0.610, 0.734] |
  | Units outside tier windows (interpretation check) | 17 | 0.837 [0.794, 0.872] |
  | Point-identified and outside tier windows | 9 | 0.837 [0.794, 0.872] |
  | Size-specific members and singletons outside tier windows | 15 | 0.837 (R1 reported 0.84 on these 15) |

- Under D-1 no unit is an upper bound: a binding tier or menu is part of v. The point-identified median equals the headline to two decimals, so the level does not rest on the lower bounds.

### F4. Serving (W14)
- The model-level coefficient is 0.56 (CRV1 s.e. 0.24; restricted wild cluster bootstrap p = 0.41, was 0.42 before the MPT-30B correction; 6 treated developers). Without Alibaba it is −0.09 (p = 0.75). Over the 49 decisions it is 0.28 (s.e. 0.29; p = 0.67).
- **Randomization inference as specified** (the model-level code permuted within release-year by compute-tercile strata, B = 9,999): p < 0.001 for the coefficient and **0.06 for its clustered t** (0.07 with two compute bins or year strata only).
- This permutation treats models, not developers, as the units of assignment; 19 of the 26 served models are Alibaba's. **Permuting developers' serving shares across developers** (the assignment unit of MacKinnon and Webb 2020; residuals of ln w on ln C and year effects) gives **p = 0.10** (0.28 unweighted), and **0.94 without Alibaba**. Over decisions the stratified RI gives 0.17 (coefficient) and 0.38 (t).
- Reading: the relation is Alibaba's; the evidence does not establish a relation that survives treating developers as the units of assignment. The model-level RI p-value should not be quoted without this caveat.

### F5. The trend (W19)
- Developer-cluster bootstrap (18 developers, B = 9,999), 49 units, reference:

  | | 2023 | 2024 | 2025 |
  |---|---|---|---|
  | Compute-weighted s, decisions | 0.24 [0.09, 0.51] | 0.54 [0.34, 0.79] | 0.81 [0.65, 0.89] |
  | Median s, decisions | 0.49 [0.24, 0.59] | 0.77 [0.73, 0.85] | 0.74 [0.63, 0.89] |
  | Compute-weighted s, models | 0.27 [0.10, 0.53] | 0.58 [0.40, 0.80] | 0.83 [0.67, 0.90] |
  | Median s, models | 0.55 [0.41, 0.62] | 0.82 [0.75, 0.85] | 0.87 [0.74, 0.92] |

  Changes of the compute-weighted share over decisions: 2023 to 2024, 0.30 [0.02, 0.65]; **2024 to 2025, 0.28 [−0.12, 0.44]** (10 percent of draws at or below zero); **2023 to 2025, 0.57 [0.19, 0.72]**. The rise from 2023 to 2025 and from 2023 to 2024 is established; the step from 2024 to 2025 is not (R4 N4).
- **New with budget-level units: the median decision share falls from 2024 to 2025** (0.77 to 0.74; change −0.03 [−0.19, 0.13]). It rises in both steps under only 1 of the 32 technologies (28 of 32 over version 3's 56 units). Without Alibaba, Hugging Face or Marin it falls from 0.80, 0.76 and 0.77 to 0.65. The compute-weighted share over decisions still rises in both steps under 32 of 32 technologies and in all 18 leave-one-developer-out samples. Sentences that say the value "rose in both years" must name the compute-weighted statistic.
- **Within developers** (W19(ii)): the unit wedge grows **1.59-fold a year** with developer fixed effects [0.91, 2.61], 2.70 given log compute [1.24, 4.54], against 1.86 pooled [1.25, 2.80]. Six developers have decisions in more than one year.
- **Mixture of experts** (W19(iii)): adding the six 2025 MoE models moves the 2025 compute-weighted share from 0.81 to **0.82 at active and 0.73 at total parameters** (2024: 0.54 to 0.56 and 0.52). MoE models carry 53 percent of 2025 compute when added.
- **Synthetic data** (W19(iv)): under the protocol's one rule (a report documents teacher-generated data in pretraining or mid-training), **21 of the 49 units are flagged, including all six 2025 decisions**, so no 2025 share without flagged units exists under that rule. Beyond the fix list's list (SmolLM 1-3, phi-1.5, phi-2, Qwen2.5, Qwen3, OLMo 2), the rule flags Qwen2 ("these models are utilized to synthesize high-quality pre-training data"), Granite 3.0 (synthetic corpora in stage 2), H2O-Danube3 ("synthetic texts"), Olmo 3 (Qwen2.5-generated QA in mid-training), Marin 8B (MathCoder2 Synthetic in its cooldown data) and Apertus (a synthetic instruction set in pretraining, by label only). With the fix list's families dropped, the 2025 share is **0.67** (median 0.65; three decisions).
- **Stated rationales** (W19(v); `tab:app-rationale`): [review] the share of decisions citing a compute-optimal target falls from 0.12 (2023) to 0.04 (2024) and 0.00 (2025); weighted by compute it is 0.45, 0.65 and 0.00 (the Llama 3 herd's flagship dominates 2024). The three decisions that cite one (Llama 3 herd, Falcon 40B and 180B) have a median s of 0.04. (Version 1: 0.12, 0.07, 0.00 and four decisions with DeepSeek LLM, whose R-CO code was resolved to 0: its report attributes the sizes to "prevalent used open-source configurations" and uses its scaling laws for hyperparameters and performance forecasts.) **Among decisions with a deployment rationale the median share rises from 0.48 (2023, 6 decisions) to 0.83 (2024, 18) and 0.89 (2025, 3)**; compute-weighted 0.35, 0.53, 0.88. Among decisions that state no rationale: 0.55, 0.75, 0.65 [review: n = 8, 9, 3; DeepSeek LLM now among them].
- Composition: tier windows hold 47, 44 and **92 percent** of 2023, 2024 and 2025 clean-sample compute; Llama 3.1 405B is 55 percent of 2024 compute; median M is 146, 1,217 and 2,438.
- Before 2023 (unchanged): 0.19 to 0.29 under the Kaplan belief with the reference curvature (0.43 to 0.57 with Kaplan's curvature), against 0.04 under the reference.

### F6. Post-training (W18)
- The disclosed budgets now include Olmo 3.1 Think 32B's post-training with the 21-day RL continuation, **x_P = 0.3217**. At x_P = 0.32 the median model share falls from 0.749 to 0.727 and the **median decision share by 0.025** (0.735 to 0.710); the share with m_N > 0 is 0.91. The 2025 compute-weighted share over models falls from 0.83 to **0.82** (m/(1 + m)) or to **0.77** with post-training compute in the cost base (both columns are in `rb2_decisions_posttrain_effect.csv`).

### F7. Data corrections and appendix inputs (W24)
- **MPT-30B: D = 1.05T** (1T at 2k context plus 50B at 8k; quote verified). Its w moves from 1.198 to 1.22 and the MPT family share from 0.24 to 0.25; the budget stays common (1.05). `d_audit.csv` records the change and six checked conventions that change nothing: OLMo 2's accounting (pretraining plus the mean mid-training budget of the averaged runs; the report's totals 5.6T and 6.6T add every run; the report's 3.90T pretraining stage for the 7B would give 3.95T, we keep the release post's 4T), Yi at 3T (the report also gives 3.1T) and Qwen2.5's unaudited per-member budgets.
- **Table E3 on audited counts** (ra2's own functions on the audited sample; ra2's files untouched): `rb2_decisions_technologies_models.csv`. The median share over models runs from 0.14 to 0.84 across the 32 technologies (0.04 to 0.80 over the 49 decisions); the reference, Farseer κ free, DeepSeek's law and the paths with model-free curvature give 0.67 to 0.79 (0.62 to 0.75 over decisions); the κ-free fits of the smaller designs 0.74 to 0.84 (0.66 to 0.80). All 32 technologies put w > 1 for 0.62 of models; the union of intervals for 0.18.
- **Table E4** (`rb2_decisions_conventions.csv`): Farseer total-count 2.50 / 0.88 / 0.60; MiniCPM 1.17 / 1.24 (head-FLOP) / 0.62 / 0.14.
- **Repetition** (R2 minor 16): with Muennighoff et al.'s effective data (R* = 15.39), StableLM-3B-4E1T's w moves from 5.96 to 5.78 and TinyLlama's from 6.57 to 6.52; the median decision share does not move (0.735).
- **Reference level** (R2 minor 7): with the curvature of the κ-free fit to Chinchilla's 137 profile runs (σ\* 0.656 in N_F; 0.667 in T) and the reference zero points held fixed, the median decision share is 0.80 (0.79). Only σ\* of that fit is available (`rb4_chinflop_param.csv`), so this is a curvature sensitivity, not a refitted technology.

### F8. Lab-own technologies without Llama 1 and 2 (W16(a), light re-run)
- Llama 1 and 2 were allocated before Meta's 2024 path existed. Without them the lab-own set has 15 models: median s **0.53 under the developers' own technologies against 0.70 under the reference**. Without AI2 as well (6 models) the order reverses: **0.71 against 0.65** (version 3's set without AI2: 0.64 against 0.56). Over the 49 decisions, with developers' own technologies where they exist (Llama 1 and 2 on the reference) and the reference elsewhere, the median is 0.717, as with Llama 1 and 2 on Meta's path. File: `rb5_units_labown_rerun.csv`.

---------------------------------------------------------------------------------------------------

## 2. Exhibits produced (not yet copied into `paper/tables`)
- `output/tables/rb2_decisions_table2.tex` (Table 2, `tab:wedge`): Panel A all 17 common-budget decisions with reading, class (pt, lb, lb^c) and embedding share; Panel B SmolLM2 and OLMo 2 13B, 32B; Panel C the five singletons with the largest training compute (stated rule); Panel D the 49-unit median with both intervals, the point-identified and lower-bound medians, the 56- and 66-unit rows, the 77 models, and identified shares at τ = 1 and τ = 1.84. The sign columns read WP4a's `data/processed/rb5_sign/model_bounds_S1.csv` when it exists (it did at the last run: 0.69 and 0.61 of decisions, 0.14 and 0.11 of their compute) and otherwise version 3's PI-1; the source is written in the file's first line.
- `output/figures/rb2_decisions_ecdf.{pdf,png}` (Figure 4): 49 units; legend, median box and the "s < −1 shown at −1" note moved to the empty upper-left area; panel b now shows the 66-unit split.
- `output/tables/rb2_decisions_readings.tex` (`tab:app-readings`): all 17 common budgets with one decisive quote, D in member order, reading and class.
- `output/tables/rb5_units_trend.tex` (`tab:app-trend-boot`, new label) and `output/tables/rb5_units_rationale.tex` (`tab:app-rationale`).
- `output/tables/rb2_decisions_posttrain.tex` with the 0.32 rows; `rb2_decisions_conduct.tex`, `_sign.tex` and `_trend.tex` regenerated on the 49 units.
- The generated tables compile without errors or overfull boxes with the paper's preamble (scratch build); two citation keys are missing from `references.bib` (Section 4).

## 3. Files
- Code: `code/analysis/rb5_units/` (`r5common.py`, `frame.py`, `coderA.py`, `verify_readings.py`, `readings.py`, `analyses.py`, `agreement.py`, `run.py`); edited `code/analysis/rb2_decisions/` (`base.py`, `decisions.py`, `run.py`, `exhibits_rb2.py`, `posttrain.py`, `rb2common.py`: 2 processes).
- Data: `data/processed/rb5_units/coding_frame.csv` (what coder B sees), `readings_coderA.csv`, **`units_primary.csv`** (77 models with unit, class, flags and wedges; the final unit file), `headline.json`; `data/processed/rb2_decisions/units_budget.csv` (49; same partition as `units_subgroup.csv`), `units_split.csv`, `units_olmo1b.csv`, `decision_units.csv` (now the 49 primary units), `decision_units_family56.csv`, `d_audit.csv`. **`units_primary.csv` in `rb2_decisions` keeps version 3's 56 family-label units**, because `rb5_sign` reads it as its 56-unit row.
- Numbers: `output/tables/rb5_units_paper_numbers.csv` (172 rows, each with its source file) and the other `rb5_units_*.csv`.

## 4. Caveats for the reviewer
1. Coder A had seen version 3's wedges and readings (protocol Section 4). The protocol and the verified quotes limit the room for outcome-driven coding, but the blind coder B is the real check. The readings most open to dispute are MPT (choice rests on "sampled according to this mix"), Granite 3.0 (cap or choice), Yi (menu rather than cap+menu), Apertus's synthetic flag (by label) and TinyLlama's own cap (Amendment 2). [review] Coder B agreed on MPT, Yi, Apertus and TinyLlama and disagreed on Granite 3.0 (resolved as cap or choice); without MPT and Yi the point-identified median is 0.745 (27 decisions).
2. The amendments were written during coding. They were prompted by specific sources (Falcon, OLMo 1, TinyLlama) but written before any class count or median was computed.
3. The model-level randomization inference is the specified test, but it ignores developer-level assignment; the developer-level version is the one that respects the few-treated-clusters problem.
4. 2025 has six decisions; every 2025 statistic is thin.
5. Randomization inference, the trend bootstrap and the within-developer intervals use B = 9,999 (4,999 for within-developer); rerunning with `--reuse` reproduces them from the cache.
