# Independent review — module m3_wedge (revealed inference demand, headline H2)

Reviewer: independent replicator and referee (Claude), 2026-09-24. The module memo `output/memos/m3_wedge.md` has been updated; every changed passage is marked **[Rev]**.

## 0. Bottom line

- **The core Sample-B results replicate exactly and are correct.** This covers the median ŵ, the shares above 1, the bands, the canonical models, the within-family flagship results and the download regressions.
- **I verified the closed forms by hand.** The wedge ln w = (α+β)ln G − α ln N + β ln D, the CE formula, M*(C) and the homothetic cosh form are all correct. I recomputed Llama-3-8B independently from the registries:
  - 5.271 under the reference technology;
  - 3.090 under the Meta A2/A1 law;
  - 5.21 under Besiroglu's published values.
- **The problems were in the secondary layers:**
  - Sample-A (Epoch) token counts;
  - the definition of the trend universe;
  - developer clustering;
  - the Farseer Eq. 3 range split;
  - a few summary denominators;
  - several memo statements that the outputs did not support.
- **Fixing them changes the numbers in H3, H4, H5 (over-identification), H7(b)/(c) and the lab table.** The qualitative conclusions are unchanged, with one exception: the "within Farseer's range, Eq. 3 ≈ Chinchilla form" statement reverses. Eq. 3 gives *larger* wedges there, which strengthens the conclusion that non-homotheticity does not explain small-model over-training.

## 1. Replication

- **Clean rebuild.** I deleted all module outputs (`output/tables/m3_wedge_*`, `output/figures/m3_wedge_*`, `data/processed/m3_wedge/*`) and ran `run.py`.
  - All 36 tables, 12 figure files and the processed data regenerated. The builder's tables were byte-identical to the regenerated ones.
  - Runtime is about 5 s on CPU. There are no own random draws; the bands use m1/m2's saved bootstrap draws.
- **After the fixes,** two consecutive clean runs give byte-identical CSV/TeX outputs, so the pipeline is deterministic. There are now 37 tables; `m3_wedge_trends_robust.csv` is new.
- **Not regenerated.** `data/processed/m3_wedge/data_appendix.md` is hand-written, so a clean rebuild deletes it. I restored and updated it; this is documented in the memo.
- **Upstream objects checked:**
  - bootstrap draw column orders: m1 θ = (ln A, ln B, ln E, α, β); m1 A2/A1 = (a, γ, E, ln K, ln G); m2 = (E, A, B, α, β). The draw medians match the point estimates.
  - the G convention for Meta's law, N* = G (C/6)^a: G = 0.5574·6^0.463 = 1.2776, which reproduces D* = 0.299 C^0.537.
  - the Farseer Eq. 3 refit: M*(C) matches `m2_farseer_eq3_Mstar.csv` to 7 significant digits.
  - `sl.HOFFMANN` and `sl.BESIROGLU` constants.
  - No sl.py bug found. The module uses only sl.py's constants, not `fit_chinchilla`.
- **Memo numbers.** A scripted comparison of about 100 quoted numbers against the final outputs passes.
- **Citations.** Every citation key exists in `lit/references.bib` or `lit/bib/extra_m3_wedge.bib`. The 13 new entries have correct arXiv IDs.
- **LaTeX.** All eight tables have consistent column counts and balanced braces. They were not compiled: there is no TeX engine on the machine.
- **Figures.** All six were inspected; no label collisions, and each panel has at most 3 colours plus grays.

## 2. Issues found and fixes

| # | Severity | Issue | Fix / effect |
|---|---|---|---|
| 1 | Major | **Developer clusters were split across name variants.** Sample A uses Epoch's `Organization` and Sample B a curated lab name. As a result "Meta" / "Meta AI", "Cerebras" / "Cerebras Systems", "Google" / "Google DeepMind" / "DeepMind" / "Google Brain" / "Google Research", Microsoft / MSR, IBM / IBM Research, TII / Technology Innovation Institute, RWKV / RWKV Foundation, Zhipu / Z.ai and the Huawei units were separate clusters. This inflated the cluster count (106) and split the lab table. | `analysis.DEV_MAP` / `developer()` maps labels to the parent developer; used for the trend regressions, the lab table and the tier regression. Now 94 clusters. Open premium: 0.60 (0.11) → 0.57 (0.09). Lab medians: Google 3.53 → 2.27, Meta 2.24 → 1.83. |
| 2 | Major | **Sample-A token counts wrong for five production-scale rows.** The builder's words/token heuristic missed them; I found them by comparing D with Epoch's own compute. GLM-130B 152B (its paper and Epoch's notes say 400B); Cerebras-GPT-13B and BloombergGPT (Epoch's dataset size is already tokens *seen*, so × epochs < 1 double-counted); Polyglot-Ko-12.8B (96B *words*; compute notes: 167B tokens); GPT-SW3 (tokens seen × 5 epochs). The Cerebras error produced the memo's "Cerebras-GPT 0.87–0.99" stated-intent range. | Added to `build_choices.EPOCH_D_FIX`, each with its source quote. New diagnostic columns `C_D_ratio` and `D_C_mismatch`. Trend regressions and by-year medians/shares are re-reported without the 11 remaining flagged rows; results are unchanged. Cerebras-GPT → 0.92–1.00; GLM-130B 0.37 → 0.53. |
| 3 | Major (claim) | **"Every model lies beyond every design in compute" is false.** It appeared in H1, Claim 1's caveat, §2.3 ("always") and the wedge.py docstring. Only 75% of Sample B exceed the Chinchilla budget, and 81–85% exceed Gadre's and Farseer's. The Pythia, Cerebras, SmolLM-135M and OPT-125M/350M models are inside. | Text corrected. The shares are now written to `m3_wedge_sampleB_by_tech.csv` (`share_Cx`) and `headline.json`. |
| 4 | Moderate | **The "production-scale core" was defined inconsistently.** 6ND ≥ 10²¹ was imposed on Sample A but not on Sample B, so 14 small Sample-B models entered the trends, rival (a)/(b) and the aggregate universe (n = 349). | New `prod` flag with the same threshold in both samples, used for trends, rivals and the aggregate (n = 335; 271 open). Sample-B-only analyses keep all 173. Share ŵ < 1: 45/39/21% → 47/41/18% (2021/22/23). Median ŵ in 2022: 1.12 → 1.06. The aggregate is unchanged to 2 digits. |
| 5 | Moderate | **Farseer Eq. 3 "in range" used 1.5× the largest Farseer model,** which put 33 7–8B models "in range". The "above" group was `~in_range`, so it also contained 5 models *below* Farseer's smallest N. The memo said "within range, Eq. 3 is close to the Chinchilla form (3.11 vs 2.96)". | Strict support (N_nonemb ≤ 6.37B) with separate above/below groups. Within range, Eq. 3 = 4.12 vs 3.44 (n = 66); above, 1.54 vs 2.12 (n = 75). The 1.5× variant is kept in the CSV. The conclusion "Eq. 3 can rationalise large models only" is strengthened. |
| 6 | Moderate (claim) | **The tier effect's significance depended on issues 1 and 4:** 0.13 (0.06), p = 0.03, n = 249. | Now 0.11 (0.06), p = 0.06, n = 238; bunching 26% vs 9%. The memo says "marginal". |
| 7 | Minor | **The "D varies within family" flag used `nunique() > 1`,** so Llama 3.1 (15T vs 15.6T) and BLOOM (341B vs 366B) entered the "informative" over-identification subsample. | Now requires max/min > 1.10. The rejection in that subsample weakens: p = 0.031 → 0.047 (absolute T); normalised p = 0.12 (unchanged conclusion). |
| 8 | Minor | **Shares in `m3_wedge_sampleB_by_tech.csv` divided by all 173 models** even for technologies defined on a subset (Farseer non-emb 146, OLMo 10), counting NaN as "no". The memo's "17% outside Farseer's non-emb M range" came from this. | Denominators are now the defined models: 21% (30/146). Farseer non-emb share ŵ > 1 is 99%, not 83%. |
| 9 | Minor | **The OLMo-ladder technology used N_total** (OLMoE: 6.9B) instead of active N, contrary to the module's convention for every other technology. | `wedge.n_for_tech` now starts from N (active). OLMoE's band [0.45, 12.0] → [1.05, 12.0]. Share with band_lo > 1: 78% → 79% (136/173). |
| 10 | Minor | **OpenRouter "number of distinct providers" summed per-release provider counts,** double counting providers that serve several releases. | Now distinct providers across releases. Coefficient 0.083 → 0.081 (SE 0.036). |
| 11 | Minor | **The Table 5 footnote defined the "e" mark by the non-embedding Farseer bound (M ≤ 2,570),** while the code used the incl.-embedding flag (M ≤ 1,227). It also repeated the "every model" claim. | Footnote corrected. |
| 12 | Minor (memo) | **Other memo errors:** distillation SE 0.21 (output: 0.23); "47 Sample-B models exceed Farseer's M range" (actually 50 incl. emb., or 30 of 146 non-emb); an exclusion list missing N-missing (361) and date (234); negative-slope families listed as "Gemma 2, SmolLM2, Phi" (actually Phi-3.5 and Phi-4, while Phi-3's slope is positive; Gemma 1 is also negative); "a positive coefficient cannot come from quality" (overclaim); the lab table presented as if complete. | All corrected in the memo. |
| 13 | Minor (repro) | **`data_appendix.md` is hand-written,** so a clean rebuild deletes it. | Restored, updated (six D fixes, `prod` universe, D/C diagnostic) and flagged in memo §5. |

**Code files changed:** `build_choices.py`, `analysis.py`, `wedge.py`, `tables.py`, `figures.py`, `run.py`. Each change carries a `[Review m3]` comment. sl.py was not touched.

## 3. Checks that passed (no change needed)

- **Formulas.** The Prop. 4 FOC w = 1 + T/(3D) holds for any technology, including the Farseer Eq. 3 numerical w. The CE formula C/C_min = w^{−1/α}((α+βw)/(α+β))^{1/γ} is correct and reduces to the cosh form when α = β. M* = G^{−2}(C/6)^{1−2a} is correct. The Figure-4 curve (M/M*(C))^{(α+β)/2} is exact. The q-family wedge uses the inner parameters (q cancels).
- **Units:**
  - N is the total count including embeddings for the Chinchilla, Gadre and Meta technologies, matching their registries.
  - N_nonemb = N − V·H·(1 or 2), consistent with Farseer's exclusion of both embeddings.
  - MoE models use active N.
  - D is tokens processed including epochs.
  - Sample-B inputs were spot-checked against primary sources I know: Llama 1/2/3/3.1, Qwen/Qwen2/2.5/3, Gemma 1/2/3, OLMo 1/2, SmolLM/SmolLM2, Pythia, OPT (300B = 180B × 1.67), GPT-NeoX/Neo/J, BLOOM, Falcon, DeepSeek, Phi, Granite, TinyLlama v1.1.
  - Exact HF float-tensor counts are used, with Gemma 3's vision tower removed.
- **Ledger reproduction.** Chinchilla 1.031, Gopher 0.362, GPT-3 0.427, Llama-3-8B 5.21, 405B 1.35 and DeepSeek-V3 3.05 under the Besiroglu row. The Meta-law correction (T/D = 6.3, not 9.6) is right *given κ = 1*: the ledger imposed ρ = 0.35, while (α+β)/2 = 0.274.
- **Bootstrap use.** The draws are joint over (α, β, ln G), so the intervals carry the correlation. Percentile intervals are used, and failed draws are filtered (none present).
- **m7 cross-reference.** A memory cap at N̄ = 0.5N* gives w = 1.78, as stated.
- **SYNTHESIS §8.4/§9 consistency.** The Sardana et al. direction (ŵ understated at extreme M) matches the literature notes.

## 4. Remaining concerns (not fixed; flag for the writers)

1. **The "PI band" is a sensitivity envelope, not an identified set.** It is a union of marginal 95% intervals and literature *points*, with no coverage statement. Upstream bootstrap intervals reflect sampling error in the sweeps only, not extrapolation error in M and C, which is the dominant uncertainty. The paper should call it a technology-sensitivity band.
2. **Meta's A2/A1 law** is used as the "lab-own" technology for *all* Meta rows in Sample B, including OPT, XGLM, LLaMA-1 and CodeLlama. These were trained 2021–23 on different data and recipes. The m1 memo also notes that the stratified A2/A1 draws may understate slope uncertainty, so Meta-law intervals may be too narrow.
3. **Tokenizer units are not harmonised,** and MoE models use active N with dense technologies. Mixed dense/MoE families affect the GNR normalisation: Phi-3.5's flagship is the MoE, and Llama 4's flagship is chosen by total N.
4. **Llama 4 Scout/Maverick D includes multimodal tokens,** and they are not excluded from the core (2 models).
5. **Near-duplicate observations:** Llama 3 vs 3.1 8B/70B (identical N, D) and Qwen-72B vs Qwen1.5-72B (identical N, D). Qwen1.5 token counts rest on a developer's GitHub reply (via Epoch), not a report.
6. **Selection on disclosure** affects both the closed and the open samples (Mistral, Mixtral, Jamba and DeciLM are absent).
7. **The validation is weak evidence of demand.** At fixed C, the ln M coefficient equals "smaller models at the same compute are downloaded more", which the hardware-accessibility channel also predicts. Downloads are not inference tokens, and OpenRouter coverage is small (21 of 164) and survivor-selected.
8. **The over-identification test** is a homoskedastic F on log T for T > 0 only (selection on the outcome). It is mechanical with common D, and marginal (p = 0.047) in the informative subsample.
9. **Sample A** still has 11 production-scale rows whose D disagrees with Epoch's compute (flagged; results are robust to dropping them), plus heuristic word/token conversions (7) and D-from-compute rows (67).
10. **The aggregate multiple** is dominated by the largest, most extrapolated models and ranges from 0.67 to 3.26 across technologies. It is illustrative only.

## 5. Confidence in the headline claims (after review)

| Claim | Confidence | Why |
|---|---|---|
| H1: modern open base models have ŵ > 1 (median 3.19; 95% > 1; 79% with the whole band > 1) | **High** for the sign and ranking; **medium** for levels | Reproduced exactly; the sign survives all technologies for about 4/5 of models. Levels span 1.9–4.0 across technologies. |
| H2: canonical models (Llama-3-8B 5.27 [3.97, 8.12]; 405B ≈ 1 under Meta's law) | **High** for the computation; **medium** for interpretation | Recomputed by hand. Meta's law rests on κ = 1; the primal gives 4.82. |
| H3: the wedge matches stated allocation rules (Chinchilla 1.04; Cerebras-GPT 0.92–1.00; LLaMA-1 1.65–2.07; Llama 3 2.48–5.27) | **Medium–high** as a sanity check | Now cleaner after the Cerebras-13B D fix. Five hand-picked cases, no power. |
| H4: switch from under- to over-training in 2023–24; open premium +0.57 (0.09), +0.44 (0.09) given ln C | **Medium–high** | Robust to the universe, cluster and D fixes and to dropping flagged rows. Sample A relies on heuristics, and the closed sample is selected on disclosure. |
| H5a: siblings are more over-trained than flagships (95% w^rel > 1) | **High**, but largely mechanical | It follows from the exponents given D. |
| H5b: flagship-on-path normalisation fails after 2024 (median 3.14; 92% CI > 1) | **High** | Unchanged by the review. |
| H5c: common demand elasticity rejected | **Low–medium** | p < 0.001 overall but partly mechanical; p = 0.047 in the informative subsample; not rejected with normalised T. |
| H6: more over-trained models used more at fixed C and year (0.97 all-time; 0.72 with developer FE) | **Medium–low** | Correlational; the hardware-accessibility confound is observationally equivalent. |
| H7a: w < 1 is pre-2024 (1 of 171 after 2024) | **High** | |
| H7b: hardware-tier bunching (26% vs 9%) | **High** | |
| H7b: tier effect on ŵ (0.11, p = 0.06) | **Medium–low** | |
| H7c: non-homotheticity explains large models only | **Medium** | Reinforced under the strict support, but it rests on extrapolating Eq. 3. |
| H7d: no distillation or synthetic-data effect | **Low** (no power) | |
| H7e: m2-scale recipe bias leaves 79–88% of models above it | **Medium** | |
| H8: ecosystem lifetime inference ≈ 2× training (all years), ≈ 3× (2025) | **Low** — illustrative | Band [0.44, 7.2]. |

## 6. Files

- **Code:** `code/analysis/m3_wedge/{build_choices,analysis,wedge,tables,figures,run}.py`.
- **Memo (updated, [Rev] marks):** `output/memos/m3_wedge.md`.
- **Appendix (updated):** `data/processed/m3_wedge/data_appendix.md`.
- **New output:** `output/tables/m3_wedge_trends_robust.csv`.
- **New columns:**
  - `choices.csv`: `C_D_ratio`, `D_C_mismatch`;
  - `m3_wedge_rival_farseer_eq3.csv`: `above_farseer_N`, `*_1p5`;
  - `m3_wedge_sampleB_by_tech.csv`: `share_Cx`;
  - `m3_wedge_trends_reg.csv`: `sample`.
