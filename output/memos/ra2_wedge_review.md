# Independent review: module ra2_wedge ("The re-specified inversion")

Reviewer: Claude, acting as independent replicator and skeptical referee. Date: 2026-09-24.

**Scope.**
- Re-run `code/analysis/ra2_wedge/run.py` from scratch.
- Audit the code: units, conventions, formulas, estimators, bootstrap and inference.
- Integrate module ra1_modelfree's final outputs where the ra2 spec calls for them: Meta's (and Marin's) model-free σ*, and the Farseer extrapolation test.
- Check every memo number against regenerated outputs, check referee-comment coverage and citations, fix problems in code, outputs and memo, and re-run.

## 0. Verdict

The module is broad, well organized and largely correct. It reproduces byte for byte.

The following survive audit and replication:
- the clean sample (77 models);
- the κ-free reference headline (median ŵ 3.99 [3.21, 4.86], median s 0.75 [0.69, 0.79]);
- the 32-technology dispersion (median s 0.17–0.85; union band > 1 for 18%);
- the family-level first-order condition;
- partial identification (86% / 77% / 23% signed);
- the open-weight premium, tier bunching, validation and cost-sensitivity results.

Five problems needed fixing, three of them material for reported numbers:

| # | Severity | Problem | Resolution |
|---|---|---|---|
| 1 | Major | **ra1 was never integrated.** `techs.ra1_sigma()` looked for `output/tables/ra1_modelfree_sigma.csv`, a file ra1 never writes; ra1's σ* is in `ra1_modelfree_isoflop_summary.csv`. The lab-own Meta and Marin technologies therefore silently used this module's own estimator. That estimator (global per-budget quadratics, including Llama 3's two unbracketed budgets, plus a 10–13% simulation-based "bias correction") differs from ra1's reviewed one by 0.02–0.05 in σ*. | Loader rewritten to read ra1's RE σ* and s.e.; the joint draws keep this module's path draws and rescale the curvature draws to ra1's s.e. The builder's estimates are kept as sensitivity rows (`*_mfbc`, `*_mfraw`). **Llama 3 8B lab-own w: 7.07 [6.24, 8.02] → 8.36 [5.67, 12.76] (s 0.86 → 0.88). Marin 8B: 6.66 → 5.08. Lab-own subsample median s: 0.59 → 0.61.** Llama 3.1 405B unchanged (0.97). |
| 2 | Major | **Serving-footprint coding contradicted its own rule and the memo.** The memo says Llama 1 is coded 0 (a research release before Meta AI), and the rule is "served first-party at the model's release". The code used a year-level threshold (Meta ≥ 2023), which coded Llama 1 (Feb 2023) and Llama 2 (Jul 2023) as 1, although Meta AI launched on 2023-09-27. | Date-level thresholds (`sample.serve_flag`). **Conduct (b): serving 0.37 (p = 0.53) → 0.45 (0.26; p = 0.39).** Clean sample: 33 models from serving developers, not 40. |
| 3 | Major | **Spec task 6 was incomplete.** The direction of the extrapolation error was not stated; the memo waited for ra1. | New `analysis_ra2.extrapolation_check` → `ra2_wedge_extrapolation_check.csv`, using ra1's Farseer gaps by M bin. Inside Farseer's M range every parametric form understates the model-free local wedge at high M. κ-free: −0.14 (M 256–1,024), −0.36 (M ≥ 1,024). The 57 clean models inside that range have median κ-free Farseer w 3.10, which becomes 3.56 when the local gap is applied (3.49 at ra1's CV bandwidth). Beyond the range, and in C, nothing is observed. |
| 4 | Major | **Convention claim misattributed.** "For sub-1B models the convention moves w by 40–55% (15.1 vs 9.1 vs 6.8)" compares the κ-free total-N row with κ = 1 non-embedding rows, so it mixes curvature with convention. | Like-for-like comparison: Chinchilla κ = 1, 9.96 vs 9.13 (−8%); Farseer κ = 1, 10.67 vs 6.85 (−36%). Memo H5 corrected. |
| 5 | Major (exhibits) | **The "paper-ready" .tex tables did not fit the page.** Compiled with `tools/tectonic` and `paper/AEA.cls` (the builder reported no TeX engine; `tools/tectonic` exists), four tables overflowed the 135.5 mm text width by 130–145 pt, i.e. about 2 inches: models, cleaning, conduct and model-free. PI overflowed by 16 pt. | Tables restructured: short labels, intervals in `\scriptsize`, lab-own w and s merged, a note column for the conduct table. A box-measure safety net (`_fit`) shrinks a table only if it is still wider than `\textwidth`. All six now compile without overfull boxes. Four are scaled slightly: models 0.94, model-free (appendix) 0.89, cleaning/conduct/PI 0.97–0.98; cost sensitivity is unscaled. |

Smaller items, all fixed:
- **Few treated clusters.** The on-device attribute is carried by 2 developers and "local" by 3. With so few treated clusters the WCR p-values are uninformative (mackinnon2018wild). Treated-cluster counts G₁/G₀ were added to every binary test.
- **Wrong comparison claim.** "Model-free σ* below every κ = 1 fit" is false for Marin: ra1's same-run κ = 1 fits (0.673–0.690) are close to model-free (0.700–0.713). The claim holds only against m1's A2/A1 objects.
- **MiniCPM caveat.** MiniCPM's M* = 192 lies outside its own design (D/N 10–60), so it is itself extrapolated.
- **Inconsistent range.** The memo's §4 placebo range (0.61–1.00) disagreed with its own universe values (0.37).
- **Hard-coded curvature range.** The PI table note had the curvature range 0.41–0.51 hard-coded; with ra1 it is 0.40–0.52, now generated.
- **Rounding.** W_f for Granite is 6.68, not 6.69.
- **Bunching windows.** They were described as "≤ 3.3B"; the code uses 2.4–3.3B and counts 0.1-dex bins whose midpoints fall in each window.

**Additions requested by the spec or the referees and missing from the build:**
- The vintage of each lab-own law, contemporaneous vs ex post (R1 minor 37). 11 of 22 are ex post: Llama 1/2, OLMo 1/1.7, Marin 8B.
- A Meta path on ra1's 8 bracketed budgets (`meta_mf8`).
- The headline by developer serving footprint (R1 c1: "confine the planned-inference interpretation to developers that serve"). Serving developers: median s 0.80, n = 33. Others: 0.74, n = 44.
- A synthetic-data robustness row (R2 Major 10, minor 22): n = 68, median ŵ 3.80, s 0.74.
- The headline excluding common-D families (R1 c2b): n = 45, median ŵ 3.99, s 0.75.

## 1. Replication

| Step | Result |
|---|---|
| From-scratch run of the builder's code (131–133 s; ≤ 5 processes, CPU only) | Exit 0. All 29 `output/tables/ra2_wedge_*` files and all `data/processed/ra2_wedge/*` files (CSV, pickle) are **byte-identical** to the builder's installed outputs. Only `headline.json` (runtime field) and `run_log.txt` differ. |
| Two consecutive full runs of the reviewed code (148 s and 145 s) | Byte-identical CSV and .tex outputs. Two later full runs, which added the serving split, the non-common-D row and the conduct-table layout, changed only those outputs. |
| `--outputs-only` from the cache | Byte-identical .tex outputs. |
| Independent recomputations, sharing no code with the module | Reference median ŵ 3.9862, share > 1 0.974, median s 0.7491 (from N, D and the κ-free point). Llama 3 8B lab-own w 8.3557 (from ra1's σ*, the path a and ln G). Qwen3 W_f 9.126. Synthetic-free row: n = 68, median 3.798, band 0.147. Serving regression: CRV1 coefficient and s.e. match statsmodels to 4 decimals (0.4526, 0.2568). A separate WCR implementation (Webb weights, B = 4,999, different seed) gives p = 0.386 vs 0.390. R4's 3.78 is reproduced (3.7787). |

## 2. Code audit

### 2.1 Verified correct
- **Wedge algebra.**
  - ln w = (α+β)ln G − α ln N + β ln D with G = (αA/βB)^{1/(α+β)}.
  - ln M*(C) = −2 ln G + (1 − 2a) ln(C/6).
  - Hybrids: α = S(1 − a), β = Sa.
  - The κ family: the inner aggregator is used and κ cancels. m2's κ-free draw layout (E, A, B, α, β, κ) is confirmed: the draw medians match the registry point estimates for Gadre RefinedWeb, OLMo and Farseer. The module's κ-free Chinchilla refit reproduces m2's row (0.42436/0.43053).
- **Published laws, verified against the PDFs in `data/raw/ra2_papers/`.**
  - DeepSeek LLM: M_opt = 0.1715 C^0.5243, D_opt = 5.8316 C^0.4757 (Eq. 4); Table 4 exponents 0.450/0.524/0.578. The conversion N_ds = M/6 gives ln G = ln(0.1715/6) + 0.5243 ln 6, and N·D = C/6 holds (0.1715 × 5.8316/6 = 0.1667). M* at 10²¹ = 19.46, checked by hand.
  - MiniCPM: α = 0.29, β = 0.23, D/N = 191.87 at 10²¹ ("Average" box; the text says "192 times"). The implied 1 − 2a = 0.115 is close to the published η = −0.10.
  - Llama 3: the path slope a = 0.4632 reproduces D* ∝ C^0.537.
- **Family first-order condition.** Σ_j ω_j ℓ_j/w_j = 1 was re-derived by hand from the Lagrangian with a shared D. The brute-force check holds (|LHS − 1| ≤ 2.2×10⁻⁸). W_f equals ra5's π-weighted identity (Prop. A9, compute-weighted harmonic mean). The memo now says so.
- **Cost sensitivity.** Minimizing 6N^{1+δ}D + 2pN^ηT subject to L gives w = (1+δ) + ηK_inf/K_tr, which is the case the code implements. R1's multiplicative case is η = 1 + δ.
- **Head FLOPs for non-embedding technologies.** With the head fixed, the lifetime/training ratio is w(1 + N_head/N_ne); checked from the first-order conditions.
- **OLMo and DeepSeek parameter conventions** match m3's rules and DeepSeek's Eq. 2.
- **Joint bootstrap (R1 9f).** One technology draw is shared by all models, and the sample median and share are recomputed in each draw.
- **Partial identification.**
  - Union over anchors, with the anchor's own 95% interval and the elasticity range applied on the correct side of C₀.
  - The ambiguous-model list (11 under PI-1, all with M ≤ 96; 18 under PI-3) was re-derived from `pi_bounds_models.csv`.
  - No model's w < 1 is identified.
- **WCR** (`ra2common.wild_cluster_boot`). Restricted residuals, Webb six-point weights, CRV1 small-sample factor, symmetric p with floor 1/(B+1).
- **Placebo tier menus and the polynomial bunching counterfactual** are as described (after the window-wording fix).
- **Citations and numbers verified.**
  - Patterson et al.: "about ⅗ of ML energy use is for inference"; Crossref metadata Computer 55(7):18–28.
  - Wu et al.: 10:20:70 power capacity; LM footprint 65/35.
  - aubakirova2026state, Table 1: DeepSeek 14.37, Qwen 5.59, Meta 3.96, Z-AI 1.18, MoonshotAI 0.92, Google 0.82 T tokens.
  - OpenRouter Terms of Service (last updated 2026-08-31), §7(5), prohibits scripts and crawlers that scrape the Site.
  - The public `/api/v1/models` endpoint returns 18 fields and none is usage.
  - All 24 citation keys in the memo exist in `paper/references.bib`, `lit/references.bib` or `lit/bib/*.bib`.

### 2.2 Problems found (beyond Section 0)
- **Upstream draws are not design-conditional.** Apart from the reference, the Chinchilla non-embedding row and the model-free paths, technology intervals use m1/m2 pairs or stratified draws. That is contrary to R1 9a's preference. Documented in the memo (§2.2, §6); not re-estimated.
- **The Meta path uses two unbracketed IsoFLOP budgets** (3×10²¹ and 10²², 6 runs each; ra1 excludes them). The 10-budget path is Meta's published law, so it is the right lab-own object. Refitted on the 8 bracketed budgets, however, a = 0.501 and M*(10²⁴) = 15 instead of 31, and Llama 3 8B's w is 12.2 (405B: 1.64). Now a sensitivity row and a memo caveat.
- **The two model-free estimators disagree on Marin by 0.03–0.05.** ra1: 0.700–0.713. ra2: 0.63–0.65 raw, 0.66–0.68 corrected. Both are within ra1's own specification range for Marin (0.64–0.72). ra1's is used because it was reviewed and its Monte Carlo shows negligible bias.

## 3. Numbers checked against regenerated outputs

Every number in memo sections 1, 4 and 5 was checked against the CSVs and `headline.json` after the final run. Rounding slips were corrected (Granite W_f; local-target coefficient 0.04, not 0.05).

| Block | Status |
|---|---|
| H1 cleaning table (10 rows × 6 statistics), reference headline and joint intervals, κ = 1 comparison, band shares | pass |
| H2 dispersion ranges (1.20–6.87; 0.17–0.85; 0.64–1.00), clustered rows, estimator variants (4.58, 5.50, [1.83, 6.48]) | pass; model-free rows updated |
| H3 lab-own | rewritten from the reviewed run |
| H4 family classes and W_f (11 families) | pass (Granite 6.68 fixed) |
| H5 conventions | medians pass; sub-1B claim corrected |
| H6 PI anchors, elasticity bounds, M* bounds, sign shares, ambiguous list | pass; curvature range updated to 0.40–0.52 |
| H7 in-support splits | pass; ra1 direction added |
| H8 conduct | (a) and (c) pass; (b) updated after the coding fix; G₁/G₀ added |
| H9 OpenRouter ratios, HF regressions, s_agg by year, leave-one-out, universe | pass |
| H10 cost sensitivity; H11 mechanical R², Spearman, σ grid | pass |
| H12 model-free | rewritten (ra1 values; ra2's own values as a sensitivity table) |

## 4. Referee coverage

| Comment | Status after review |
|---|---|
| R1 c1 | **Partly addressed.** Open vs closed, serving footprint and deployment target are tested. The headline is now reported for serving developers separately. The three-conduct-model derivation belongs to ra5, and durability is not modeled. The on-device test is uninformative (2 treated clusters). |
| R1 c2a–b, R2 Major 5a/5c | Addressed: multi-tier bunching, placebo menus, family W_f, and the headline excluding common-D families. |
| R1 c3 | Lab laws are in the set. **The Raval second-margin test is not done.** |
| R1 c4a–c, R2 Majors 1 and 6a, R3 M4, R4 M2 | Addressed: ex-ante set of 32, κ-free reference, lab laws, R4's 3.78 reproduced. |
| R1 c4e | Addressed: PI-1 to PI-4. The Llama 3 402B/16.55T extrapolation is not used. |
| R1 c4f, R3 M3e, M12 | Addressed: ECDF exhibit and trend figure with counts. |
| R1 c5 | Addressed: δ, η and p table. |
| R1 9a | **Partly addressed.** Only the reference and the model-free rows use wild draws. |
| R1 9f | Addressed. |
| R2 Major 2 | Addressed: expenditure share; T only with p. 2c (RL rollouts and other internal compute) is discussed, not measured. |
| R2 Major 3 | Addressed: in-support splits and the ra1 direction. **The "perceived vs true technology" point (3(ii)) is not resolved.** |
| R2 Major 4a–d | Addressed: 4a with the corrected claim. 4e (tokenizer) is not done. |
| R2 Major 10 | Addressed: cleaning table, synthetic robustness row, dedupe. **"Corpus size vs tokens processed" (Qwen2.5's 18T for every size) is not audited.** |
| R2 Major 11, R3 M3c | Addressed as far as public data allow: OpenRouter blocked by the ToS; HF model tree; aggregates vs Patterson and Wu. |
| R3 M3a, b, d | Addressed; lab-own vintage flagged. |
| R3 M5.1, 5.2, 5.5, 5.6 | Addressed. M5.3–4 (data cost, test-time compute) are not measured. |

## 5. Files changed by the review

- **Code** (`code/analysis/ra2_wedge/`):
  - `ra2common.py`: ra1 paths.
  - `techs.py`: ra1 loader and draws, `*_mfbc` and `meta_mf8` rows, `LAB_LAW_FROM`, lab-own alternatives.
  - `sample.py`: date-level serving thresholds; Meta dated 2023-09-27.
  - `analysis_ra2.py`: G₁/G₀ in `wcr`, `extrapolation_check`, synthetic robustness row.
  - `run.py`: vintage, extrapolation check, serve split, non-common-D row, headline fields.
  - `tables_ra2.py`: layouts that fit the page, `_fit`, dynamic notes, ra1 column.
  - `figures_ra2.py`: ECDF clipping note, ra1 values in the model-free legend.
- **Outputs**:
  - regenerated: all `output/tables/ra2_wedge_*`, `output/figures/ra2_wedge_*` and `data/processed/ra2_wedge/*`;
  - new: `ra2_wedge_extrapolation_check.csv` and `ra2_wedge_serve_split.csv`.
- **Memo**: `output/memos/ra2_wedge.md`, updated in place with a review paragraph at the top.
- The builder's pre-review outputs and code are archived in the reviewer's scratchpad (not part of the repository).

## 6. Remaining concerns (for writers and the lead author)

1. **Levels are technology-dependent.**
   - Median s runs 0.17–0.85 over the 32 ex-ante technologies, and the union band lies above 1 for 18% of clean models.
   - The reference interval, [0.69, 0.79] for median s, is one technology's sampling interval.
   - Only the ordinal content is robust, and it is mechanical (R² = 0.99994 on ln M).
2. **The lab-own Meta numbers depend on the path as much as on σ*.** The 8B's w ranges over 3.09–12.2 across Meta's own alternatives; the "405B on the path" result holds only for the 10-budget published law.
3. **The extrapolation direction comes from one recipe, and in M only.** It suggests the parametric levels are conservative at high M. Frontier-scale M*(C) remains partially identified only (PI).
4. **Conduct tests have little power.** The WCR p for serving is 0.39 at t = 1.76 with 18 clusters. On-device and local cannot be tested with 2–3 treated developers. Only the open-weight premium (p = 0.011) and bunching are informative.
5. **Serving-footprint coding.** Microsoft's phi-1.5 (coded 1) and 01.AI's 2024 date are judgment calls; the rest of `ra2_wedge_coding.csv` was not re-audited.
6. **Not done:** the Raval second-margin test, tokenizer/byte normalization of D, durability and test-time compute, and an audit of corpus size vs tokens processed.
7. **Table sizes.** The .tex tables fit only after mild scaling (0.94 for the models table, 0.89 for the appendix model-free table). Writers condensing them for the main text should drop the "D (T)" column or Panel C.
