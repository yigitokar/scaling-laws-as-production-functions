# Independent review: module rb2_decisions (revealed value of compactness at the level of allocation decisions)

Reviewer: independent replicator and skeptical referee (Claude), 2026-09-24. The binding plan is `paper/notes/revision_plan_v3.md`. The builder's memo is `output/memos/rb2_decisions.md`; it has been revised in place, and every change is marked [review].

**Verdict.** The module replicates exactly. Its central findings survive, but several numbers, claims and pieces of inference needed fixing:
- the decision-unit median s is 0.74 with 11 family shares;
- the Llama 3 herd share is 0.29 and the flagship's sign is not identified;
- the serving code shows no detectable relation to ŵ;
- sign identification is 87% of models but only 48% of compute;
- the rise from 2023 holds under all 32 technologies (compute-weighted).

I fixed these problems in the code, re-ran the module, and updated the memo. The largest substantive addition is R4 R2-M3(d), which the builder had left open. Under ra1's conservative budget-plus-trunk cluster bootstrap, the headline interval widens to **[0.53, 0.81]**, against [0.67, 0.78] under the wild bootstrap.

---------------------------------------------------------------------------------------------------

## 1. Replication

- **From scratch.** `run.py --rebuild-techs` (ra2 registry rebuilt, 227 s, CPU only, 4 processes) reproduced all 31 `output/tables/rb2_decisions_*` files and every CSV in `data/processed/rb2_decisions/` **byte for byte**. `headline.json` differed only in its runtime, and the PDF only in its timestamp.
- **Against ra2.** The reference wedges equal `ra2_wedge_models.csv` (`w_chin_q`) to 1e-15 on 75 of 77 models. The two exceptions are StableLM-Alpha 3B and 7B, which this module deliberately corrects.
- **After the fixes.** Two further full runs are identical, apart from the intended move of one Table 2 row between them. Runtime is about 3.7 minutes.
- **Independent re-implementations:**
  - The four anchor bootstrap-t intervals, re-implemented from scratch (OLS on ra1's bracketed minima, HC2, Webb weights, B = 20,000), match to rounding. Chinchilla: 20.66 [12.39, 34.66] against the module's [12.38, 34.42]. Llama 3: 15.06 [9.68, 23.61] against [9.72, 23.15]. Marin: identical to within 0.1.
  - Checked by hand:
    - Llama 3 8B lab-own w = 8.27 (M* = 15.9 at 7.2e23; k = 0.443);
    - Llama 3.1 405B w = 1.46;
    - the post-training ratios (DeepSeek-V3 0.0019; Tülu 3 SFT 0.0004/0.0007; Olmo 3 RL 0.026/0.135/0.213, from the Olmo 3 report's GPU-day breakdown, pre-training 43,264 GPU-days);
    - the Kaplan-belief wedges (OPT-175B w_K = 1.00 by construction; GLM-130B 1.28).
- **Primary sources.** Eleven sources were re-downloaded, and each matches the manifest's sha256 byte for byte:
  - the four Alibaba Wayback captures (2024-02-02, 2024-07-17, 2024-09-30, 2025-04-29);
  - the reports for Qwen3, DeepSeek LLM, Llama 3, Olmo 3 and Yi;
  - the SmolLM2 card and the Qwen2.5 blog.

  Live pages were fetched as well, and these quotes were confirmed:
  - VentureBeat, 2023-12-01: "Only the 67B version is available through this interface."
  - Llama 3 blog: "continued to improve log-linearly … up to 15T tokens"; "Meta AI, built with Llama 3 technology".
  - Llama 3.1 blog: "Try Llama 3.1 405B … at meta.ai".
  - MPT-30B blog: the single-GPU sizing sentence, and the hosted endpoints "with standard pricing per-1K-tokens".
  - Gemma 2 blog: AI Studio "at 27B" (the page does not name the 9B).
  - IBM Granite 3.0 announcement: watsonx availability.
  - Qwen3 catalog: the Model Studio capture lists per-token prices for qwen3-0.6b through qwen3-14b.
  - StableLM README: the "Training Tokens" column of the Alpha table reads 800B, so the builder's token audit is correct. The contemporaneous press (TechCrunch, Voicebot) repeats the 1.5T dataset size, which is the source of the old error.
- **Citations.** Every key used in the .tex tables is in `paper/references.bib` or `lit/bib/extra_round3_rb2_decisions.bib`. The eight new entries match their sources.
- **LaTeX.** All eight tables compile with tectonic under AEA.cls.

## 2. Problems found and fixed

### Code and inference

| # | Severity | Problem | Fix |
|---|---|---|---|
| 1 | major | **R4 R2-M3(d) not addressed.** The referee asked for the headline under ra1's budget-plus-trunk cluster scheme; the builder left it open ("needs ra1's cluster draws"). | New `cluster_ref.py`: loads m1's Chinchilla loader under a private name, applies ra1's trunk rule (44 clusters = 9 budgets + 35 trunks; matches ra1), and refits κ-free Chinchilla by cluster resampling (m2 `fit_q` from the ra2 point, B = 999, 0 failures). The point estimate equals ra2's to 1e-8. Results: M\*(5.76e23) [6.9, 67.0] vs [14.0, 33.9] under the wild bootstrap. Decision-unit median s: 0.736 **[0.526, 0.808]**; models 0.749 [0.575, 0.827]; share of units with w > 1 [0.839, 1.000]. Now in `headline.csv` and in Table 2, Panel D. |
| 2 | minor | **Uneven bootstrap weights.** The joint bootstrap reported B = 999 but cycled the reference's 399 wild draws: draws 0–200 got weight 3, draws 201–398 weight 2. | B = 3 × 399 = 1,197, an exact multiple. Headline interval [0.663, 0.785] → [0.666, 0.784]; share of units with w > 1 [0.946, 1] → [0.945, 1]. |
| 3 | minor | **Curvature uncertainty understated.** The lab-own curvature draws were N(0.693, 0.0126). rb1's study-level interval is HKSJ, mean ± t₃ × s.e. = [0.653, 0.733] with k = 4 studies, so the draws covered only ±0.025 of its ±0.040. | `rb2common.common_sigma` returns df = k − 1, and `labown._k_draws` draws σ* + s.e. × t₃. Lab-own intervals widen: Llama 3 8B [4.85, 14.64] → [4.51, 15.91]; OLMo 2 1B [3.99, 6.18] → [3.60, 6.90]; 405B [0.71, 3.01]. Point estimates and medians are unchanged. |
| 4 | minor | **PI-4 missing.** R2 Major 4.3 asks for the all-technologies set beside the 87%; the module did not recompute it with the new anchors. | PI-4 added (widened IsoFLOP anchors plus ra2's in-support anchors for all 32 technologies; e ∈ [−0.156, 0.311]). It identifies w > 1 for 0.23 of models (ra2: 23%), 0.20 of decision units, and 0.03 of compute; 0.14 of models at τ = 1.84. New row in the sign table. |
| 5 | minor | **Leave-one-developer-out fragility not shown.** The trend table reported LODO ranges only. | The table now counts LODO samples in which each statistic rises. Compute-weighted: 18/18 for models and for units. Model median: 17/18 (flat without Alibaba). Decision-unit median: 15/18; it falls from 2024 to 2025 without Alibaba (0.82 → 0.75), Hugging Face (0.76 → 0.75) or Marin (0.81 → 0.76). |
| 6 | minor | **Showcase uses synthetic-data models.** SmolLM2 (and SmolLM3), the recommended showcase, carries `f_synthetic`, which is a robustness exclusion elsewhere in the paper; nothing in Table 2 flagged it. | Dagger and table note added in Table 2; the memo caveat recommends SmolLM2 for sign and ordering, and OLMo 2 for levels. |
| 7 | minor | **Table 2 notes incomplete.** They did not say that under a binding data cap the family share is also a lower bound, or that the cap bound fails where a tier cap also binds. | Notes extended. New variant: reading-consistent units without the 9 cap members that sit in tier windows, 57 units, s = 0.749 [0.691, 0.794]. |
| 8 | minor (privacy) | **E-mail in request headers.** `fetch_sources.py` sent the user's e-mail address in the User-Agent to arXiv, Wayback, Meta, IBM and other sites. | Removed. Nothing was re-fetched with it. |
| 9 | cosmetic | **Stale runtime.** The run.py docstring said "about 10 minutes". | Now "about 4 minutes"; cluster_ref is listed. |

### Memo: facts and wording

| # | Severity | Problem | Fix |
|---|---|---|---|
| 10 | major | **Wrong commit for the revision plan.** H9 said "the round-1 reports and revision plan were committed at 03:12–03:13 (bd5c0ad)". In fact `git log -- paper/notes/revision_plan.md` shows the plan was first committed at **07:48 in acc8286**, together with ra2's code and outputs. Its *file* modification time is 03:13:37, which is where the builder's "03:13" came from. | H9 corrected. It now states the modification times as consistent with, but not proof of, the order: plan 03:13, `ra2_wedge/sample.py` 06:42, logged ra2 run 07:09. The suggested "ex ante" sentence still needs the lead author's confirmation. |
| 11 | minor | **Serving null overstated.** "Serving does not predict over-training" misstates a null with a positive point estimate (0.56 log points, CRV1 t = 2.3, WCR p = 0.42, 6 treated clusters). | Now "no detectable relation; the positive estimate is Alibaba's (−0.10 without)". |
| 12 | minor | **"Contradicts R3" overclaims.** A catalog listing of Qwen3 0.6B shows availability, not that most use is served. | Reworded. Also noted that Qwen2.5 0.5B–3B were listed as free for a limited time. |
| 13 | minor | **Evidence coverage overstated.** The memo said every serving row has its own dated quote. In fact 48 rows have the model's own evidence and 13 a sibling's from the same release. Two rest on a same-developer page with no date, and **14 rows (all coded 0)** rest on ra2's developer coding with no URL. | Corrected. All 26 served = 1 codes do have dated, verified sources. |
| 14 | minor | **Post-training upper bound incomplete.** The 0.21 bound excluded Olmo 3.1's 21-day RL continuation, which came after the ~9 post-training days; the bound for Olmo 3.1 Think 32B is **0.32**. The aggregate also used a different share definition, m/(1 + m + x), from the medians, m/(1 + m). | Added: at x_P = 0.32, median s is 0.727 (models) and 0.711 (units), the share with m_N > 0 is 0.91, and the 2025 aggregate is 0.77 (0.82 under m/(1 + m)). Conclusion unchanged: second order. |
| 15 | minor | **Unapproved citation key.** The memo listed `bond2021some` from `lit/bib/io_controlfn.bib`, which is not an allowed bib file. | Replaced with `bond2020unpleasant`, the same paper, which is in `paper/references.bib`. |
| 16 | cosmetic | **Rounding inconsistencies.** Examples: compute-weighted identified share 0.49 in the memo against 0.4847 in the file and 0.48 in the table; the lab-own table's 0.54 against the file's 0.53; an s.e. of 0.16 against 0.155. | Harmonized with the outputs. |
| 17 | cosmetic | **On-device contrast.** R3's 0.74 for the 65 server/unspecified models is now 0.70. | Explained: the StableLM token audit moves two models across a gap in the distribution, so this median is fragile. |

## 3. Numbers verified against outputs

- **Every table in the memo** was checked row by row against the CSVs:
  - H1 headline and robustness rows, family shares with intervals, the token audit;
  - H2 serving counts, conduct table, tier and on-device medians, open-weight premium;
  - H3 lab-own table;
  - H4 second-output table;
  - H5 anchors, the τ table by level, weight and year, M\* sets and magnitude bounds;
  - H6 trend table, including 31/32, 28/32 and 32/32, and the LODO ranges;
  - H7 pre-2023 table;
  - H8 post-training;
  - H9 the 12 version-1 technologies. The v1 registry was checked in commit b4866cd: chin, besi, hoff, farseer ×3, gadre ×3, olmo, meta_a2, meta_a3. The v1 median of 3.19 over 173 models was also checked.
- **Discrepancies** were fixed as listed in Section 2. After the fixes, the memo's numbers match the final outputs.

## 4. Referee comments: are they actually addressed?

| Comment | Status after review |
|---|---|
| R1 New 1(a) Llama 3 showcase | Addressed. The herd share is 0.29 and the flagship's sign is unidentified; the showcase is replaced by SmolLM2 and OLMo 2. [review] Caveats added: the 8B lower bound needs the tier cap to be slack, and SmolLM2 carries the synthetic-data flag. |
| R1 New 1(b), R2 M2 req 3, R3 N3(a) Table 2 | Addressed (`rb2_decisions_table2.tex`, label tab:wedge). |
| R1 New 1(c)–(d), R3 N4(a) model-level serving | Addressed, but the evidence is incomplete for 14 served = 0 rows (Section 2, item 13). |
| R1 New 1 req 2 (Prop. 2(i) sample; 19 outside tier windows) | Addressed: 0.749 (n = 45) and 0.848 (n = 19). The latter matches R1's own count and value. |
| R1 New 1 req 4, R3 N4(b) conduct reading and abstract | Addressed. Wording softened (Section 2, item 11). |
| R1 New 3 (tilt allowance, weighting, normal inputs, "weak assumptions") | Addressed. PI-4 added [review]. |
| R1 New 4, R3 N2, R4 R2-M1 (trend from 2023; pre-2023 baseline) | Addressed. [review] LODO fragility of the 2024 → 2025 medians is now reported. |
| R1 New 5 (second output) | Addressed, on a single recipe. |
| R1 minor 7, R4 R2-M3(c) (Llama 3 anchor at a bracketed budget) | Addressed. |
| R2 Major 2 req 1 (add the cap case to Prop. 2(iv)/A9 and Table F1) | **Theory text not done.** This module is not the place for it, but the result needed is now written in the memo. Under a binding data cap the family condition becomes Σπᵢ(1 + m_N,i) = (1 + m_D + μ) W̄_H ≥ (1 + m_D) W̄_H, so W̄_H − 1 bounds the π-weighted value of compactness from below (with m_D = 0). A cap member's own wedge is a lower bound only if no tier cap also binds. Writers or theory must add a part (vi) to Prop. A9 and a Table F1 row. |
| R2 Major 2 req 2 and 4 (readings; headline over decision units) | Addressed. The Qwen2.5 and Qwen3 "cap" readings are judgment calls, but the primary units use their family shares either way. |
| R2 Major 4.3 (name the assumptions; give PI-4) | Addressed [review: PI-4 = 23%]. |
| R2 Major 4.4 ("ex ante") | Addressed, after correcting the plan-commit fact (Section 2, item 10). |
| R2 Major 4.5, R4 R2-M2 (trend restates M; level conditional) | Addressed in the claims. The abstract wording is for the writers. |
| R2 Major 6 (one lab-own rule) | Addressed. [review] Curvature draws fixed. Remaining concern: Meta's 2024 Llama 3 path is applied to Llama 1 and 2 (open issue 11). |
| R2 Major 7 (post-training) | Addressed. [review] The Olmo 3.1 bound is corrected to 0.32. The quantization paragraph belongs to the writers. |
| R3 N3(b)–(c), N4(d) | Addressed. |
| R4 R2-M3(1–2) (between-budget anchor and path intervals) | Addressed. Replicated independently. |
| **R4 R2-M3(d) / request 3** (headline under the budget-plus-trunk cluster scheme) | **Was open; now addressed** [review]: [0.53, 0.81]. |

## 5. Remaining concerns (not fixed here)

1. **The headline level under conservative inference.** With ra1's recommended conservative scheme, the lower end of the decision-unit median is 0.53. "Exceeded training cost for the median decision" (s > 0.5) survives only barely. Writers should quote both intervals.
2. **The model-level serving code rests on ra2's developer coding for 14 served = 0 models,** with no dated evidence of absence. Coding any of them 1 is unlikely: TII, Stability AI, AI2, TinyLlama and M-A-P had no first-party per-token LLM product at those dates. But the file does not document it.
3. **Cap readings are judgment calls.**
   - For the Llama 3 herd, the flagship's D (15.6T) exceeded the stated "about 15T" corpus, and the flagship was sized by the compute-optimal law. For it the cap need not bind.
   - The Qwen3 reading rests on data expansion, not a statement that the corpus binds.
   - 9 of the 15 cap members sit in memory-tier windows, where the two constraints bound m_N in opposite directions.
4. **Anachronistic lab-own paths.** The Llama 3 path is applied to Llama 1 and 2 (a ra2 convention, kept). The Meta lab-own medians mix a recipe correction with a belief assumption.
5. **Second-output test.** One recipe (OLMo ladder, M ≤ 200), and BoolQ is degenerate. Under κ = 1, 10% of model-level intervals exclude zero for science QA and commonsense.
6. **Fragile medians.** With 7 (or, leaving one developer out, 6) decision units in 2025, the 2024 → 2025 step of the medians rests on one or two developers. The robust trend statements are the compute-weighted ones, which rise under 32/32 technologies and 18/18 LODO samples, and the 2023 → 2024 rise.
7. **The "ex ante" order within the ra2 session** cannot be established from git. The file modification times are consistent with the rules being fixed first, but they are not proof.
8. **Upstream dependency.** If rb1's study-level σ* (0.6929, k = 4) changes, re-run this module; only the lab-own columns move.

## 6. Files changed by the review

- **Code:**
  - `code/analysis/rb2_decisions/cluster_ref.py` (new);
  - `run.py` (B = 1,197; cluster rows; cap-in-tier variant; PI-4; docstring);
  - `labown.py` (t₃ curvature draws);
  - `rb2common.py` (df from rb1);
  - `exhibits_rb2.py` (Table 2 cluster row, synthetic dagger, cap/tier note; PI-4 row; LODO counts);
  - `fetch_sources.py` (no e-mail in the User-Agent).
- **Outputs (re-generated):** all `output/tables/rb2_decisions_*`, `output/figures/rb2_decisions_ecdf.{pdf,png}`, and `data/processed/rb2_decisions/*`.
- **Memos:** `output/memos/rb2_decisions.md` (revised, [review] marks) and this file.
