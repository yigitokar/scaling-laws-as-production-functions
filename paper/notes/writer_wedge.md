# Writer notes: Section V (wedge), "Over-Training Reveals Anticipated Inference Demand"

Written 2026-09-24 by the section writer. Files:
- `paper/sections/wedge.tex`: main text, about 3,100 words by a crude count (target 2,800). Every plan item V.A–H is covered.
- `paper/tables/table6_wedge.tex` (`tab:wedge`): 17 rows in 3 panels. Condensed from `m3_wedge_table5.tex` and `m3_wedge_models.csv`.
- `paper/tables/table7_validation.tex` (`tab:validation`): built from `m3_wedge_validation.csv`.
- Figures: `fig:wedge` = `m3_wedge_fig4`; `fig:trends` = `m3_wedge_trends`.
- The section compiles alone with `code/paper/test_section.sh wedge`. There are no errors and no overfull boxes; the only undefined references point to other sections or the appendix.

Preamble: nothing new is needed. `\textcolor` comes from xcolor.

## 1. Numbers used, with sources
All CSVs are in `output/tables/`. "models" means `m3_wedge_models.csv` restricted to Sample B general-purpose models (`sample=='B' & ~code`, n = 173). I checked each number against the CSV, not only the memo.

| Number in text | Value | Source |
|---|---|---|
| Reference technology | α 0.347, β 0.367, σ* 0.737, M*(1e21) 21.4, M*(1e24) 17.6; 400 draws | m3_wedge_technologies.csv (chin) |
| σ* range, six band technologies | 0.72–0.80 (chin, besi, hoff, farseer, farseer_emb, gadre_rw) | m3_wedge_technologies.csv |
| M*(1e21) range | 3.4 (gadre_rw) to 34.0 (hoff) | m3_wedge_technologies.csv |
| Sample B | 173 models, 68 families, 2021–2025 | models (family nunique = 68; year min/max) |
| Tokenizer effect | β·ln1.2 = 0.067, about 7% | memo H1 caveat |
| Share with M > 341 | 55% | m3_wedge_sampleB_by_tech.csv share_Mx (chin) 0.549 |
| Share with M > 1,227 | 29% | share_Mx (farseer_emb) 0.289 |
| Share with C > 1.3e22 | 75% | share_Cx (chin) 0.746 |
| sd(ln ŵ) at 8B size | 0.161 total; 0.143 from M*; 0.047 from α+β | m7_theory_pi_summary.csv; appendix_proofs remark |
| Median ŵ | 3.19; T/D 6.6; inference/training 2.2 | models (w_chin median 3.186) |
| Shares | ŵ>1 95%; ŵ>2 75%; lower bound >1 87% (cluster 80%); band >1 79% (136/173); median band [1.77, 6.92] | sampleB_by_tech.csv; models |
| Median CE 0.41, i.e. 2.4× | 0.414, and 1/0.414 = 2.42 | **computed by me** from models CE_chin (median over 173). Not in the memo. |
| Technology medians | Hoffmann 1.91, Meta 2.12, Farseer 2.41/2.67, Besiroglu 3.16, Gadre RW 4.03, q-family 3.85, NLS 4.11, n = 245: 4.78 | sampleB_by_tech.csv |
| Bootstrap interval ratio | 0.80–1.39 | sampleB_by_tech.csv median_lo/hi_ratio |
| Spearman, reference vs each alternative | 0.95–1.00 (min 0.954, Farseer incl. emb.) | **computed by me** from models (w_chin vs w_{besi,hoff,farseer,farseer_emb,farseer_q,gadre_rw,meta_a2,meta_a3,chin_nls,chin245}). Not in the memo. |
| Llama 3 8B | M 1,868; ŵ 5.27 [3.97, 8.12]; T/D 12.8 [8.9, 21.4]; T 1.9e14; Meta law 3.09 (T/D 6.3); Meta A3 4.82; band for T/D [4.4, 33]; CE 0.18 (5.6×) | m3_wedge_table5.csv; models (w_meta_a3); memo H2 |
| Llama 3.1 405B | Meta law 0.98; reference 1.37 [0.99, 2.22] | table5.csv |
| Llama 2 70B | Meta law 0.98; reference 1.19 [0.93, 1.67] | models |
| Qwen3 0.6B | D 36T; M 60,398; ŵ 17.9. "More than 20×" the largest design M (Farseer non-emb 2,570: 23.5×) | table5.csv; technologies.csv |
| Local σ falls with M on Farseer | qualitative, "suggestive" | m2 memo claim 4 (flagged suggestive only) |
| Nonparametric bound | dominating support points have ŵ ≤ 0.77 (0.766 refit; 0.763 Besiroglu) | m7_theory_pi_summary.csv |
| Stated intent | Chinchilla 1.04 [0.82, 1.42]; Cerebras 0.92–1.00; LLaMA-1 1.65–2.07; Llama 3 2.48–5.27; Pythia 1.09–6.49 | m3_wedge_stated_intent.csv; models |
| Trends universe | 335 models (271 open, 64 closed), 2019–2026 | m3_wedge_trends.csv |
| Share ŵ<1 | 47% (2021), 41% (2022), 18% (2023), 0% (2024), 2% (2025) | trends.csv |
| Median ŵ | 1.06 (2022), 1.83, 3.45, 4.68 (2025); median M 23 → 1,303 | trends.csv |
| Closed giants | GPT-3 0.43, Gopher 0.37, MT-NLG 0.28, PaLM 0.41 | models (Sample A rows) |
| Open premium | 0.57 (0.09); 0.44 (0.09) given ln C; 94 clusters; robust 0.59/0.43 | m3_wedge_trends_reg.csv |
| Developer medians | HF 6.6, Alibaba 4.8, Google 2.3, Meta 1.8, Cerebras 1.0 | m3_wedge_labs.csv |
| Closed 2025 | 3 models | trends.csv year_x_open |
| Siblings | 95% of 105 in 44 families; Cerebras exceptions 0.92–0.98 | memo H5; m3_wedge_overid_siblings_wrel_lt1.csv |
| Flagships | pre-2024 n 18, median 1.14, 50% CI contains 1; 2024+ n 26, median 3.14, 92% CI above 1; Qwen3 32B 4.46 [3.22, 7.33] | m3_wedge_flagships_by_period.csv; table5.csv |
| Over-identification | η̂ 0.53 (0.07); F 2.55, p 0.001, 138 models / 42 families; informative subsample p 0.047 (58 / 18), normalized p 0.12 (33 / 11) | m3_wedge_overid_demand.csv |
| Validation | see Table 7: 0.97 (0.18), 1.13 (0.19), 0.72 (0.14), 0.94 (0.14), providers 0.081 (0.036), LMArena 0.29 (0.16), likes unconditional 0.13 (0.25), votes −0.17 (0.40); n 164 / 26 developers; FE n 158 / 20; LMArena 46 / 12, FE 42 / 8; OpenRouter lists 21 of 164 | m3_wedge_validation.csv; memo H6 |
| Post-2024 ŵ<1 | 1 of 171 (StepFun Step-1) | trends.csv (2024–26 n = 106 + 61 + 4); memo H7a |
| Hardware tier | 26% vs 9%; 0.11 (0.06), n 238; median ln ŵ 1.16 | m3_wedge_rival_regressions.csv; models (median ln w_chin 1.159) |
| Memory cap | w = 1.78 at N̄ = 0.5N*, T = 0 (Besiroglu parameters) | m7_theory_constraints.csv (m7 numeric.py uses BES) |
| Farseer Eq. 3 | M* 25 (1e20), 82 (1e23), 195 (1e24); in support 4.12 vs 3.44 (n 66); above 1.54 vs 2.12 (n 75); Llama 3 70B 0.62 | m3_wedge_rival_Mstar_curves.csv; rivals.tex; models w_eq3.1 |
| Distillation / synthetic | 0.02 (0.23), 0.06 (0.18) | m3_wedge_rival_regressions.csv |
| Factor bias | 1.25–1.29; 1.80; Gadre 1.02–1.04; shares 88% / 79%, band 69% / 49% | m3_wedge_rival_factor_bias_m2.csv, _shares.csv |
| Aggregate | all 2.0 [1.2, 3.7], band [0.4, 7.2]; 2025 3.2 [2.1, 5.7], band [1.05, 9.6]; 2024 1.2 [0.6, 2.3]; 2022 −0.40 (truncated 0.03); by technology 0.67–3.26 | m3_wedge_aggregate.csv; m3_wedge_aggregate_by_tech.csv |

Table 6 rows come from `m3_wedge_models.csv` (w_chin, wlo/whi_chin, CE_chin, band_lo/hi, w_meta_a2, w_olmo), generated by script and cross-checked against `m3_wedge_table5.csv`. Three rows are **not** in table5: Llama 2 70B, Chinchilla and GPT-3 175B. They are taken from the models CSV (Sample A for Chinchilla and GPT-3).

## 2. Claims and the caveats attached to them
- **Levels vs ranks.** Levels depend on the technology (1.91–4.03). Ranks are robust (Spearman ≥ 0.95).
- **Band.** Called a "technology band" and described as a sensitivity envelope, not an identified set, with no coverage guarantee (review §4.1).
- **Extrapolation.** 55% / 29% outside the design M ranges; 75% beyond every design's largest budget, stated as the correct share and not "every model" (review issue 3). Direction: Sardana et al. imply ŵ is understated. The Farseer local-σ evidence is flagged as suggestive.
- **Meta's law.** It imposes κ = 1; the primal gives 4.82 for the 8B.
- **Qwen3-0.6B.** The level is not credible; only its rank is.
- **Stated intent.** Five hand-picked cases, with no statistical power.
- **Trends.** Selection on disclosure affects both the closed and the open samples.
- **Siblings.** The sibling pattern is "close to mechanical". Post-2024 flagship-normalized T are lower bounds.
- **Over-identification.** Mechanical with a common D; marginal in the informative subsample; a classical F test on T > 0. It does not reject the wedge.
- **Validation.** Weak evidence. It is equivalent to "smaller models at equal compute are downloaded more" (the hardware-accessibility confound), and quality correlated with M is not ruled out. The FE estimate is the conservative one. Downloads are not tokens, and OpenRouter covers 21 of 164. **I removed the claim "cannot come from quality" from the original table note** (review issue 12).
- **Rivals.**
  - Tier effect: second-order.
  - Eq. 3: an extrapolated 9-parameter form.
  - Distillation: no power, and the sign is ambiguous.
  - Factor bias: uses DataDecide within-lab recipe variation as the scale.
- **Aggregate.** Presented as illustrative only: dominated by the largest models, technology-dependent, and not validated against inference volumes.
- **Wording.** The text says explicitly that w is not a markup. w is lifetime/training compute, w − 1 is inference/training compute, and T/D = 3(w − 1).

**Addition of mine to verify.** The measurement subsection says: "if serving costs p times as much per FLOP, the first-order condition becomes w = 1 + pT/(3D), so T̂ is in training-cost-equivalent tokens." This is a one-line derivation of mine (θ_D = 6D/(6D + 2pT)), not an m7 result. It is correct, but the integrator may drop it or add it to Proposition A8 as a remark.

## 3. Cross-references assumed to exist elsewhere
- Sections: `sec:framework` (CE formula and rival-wedge signs; Farrell CE), `sec:ident` (M* weakly identified), `sec:data` (Sample B, Epoch universe, usage data), `sec:tech` (Chinchilla Huber refit, 9-cluster bootstrap, κ = 1 rejection, DataDecide recipe tilts 1.25–1.29).
- Propositions: `prop:wedge` (main-text Prop 2), `prop:pi` (Prop 7).
- Appendix labels: `cor:suff` (Online Appendix A, sufficient statistics).
- "Online Appendix D" is assumed to hold the m3 tables: technologies, family, rivals, aggregate, homothetic.

## 4. Placeholders
- One `\textcolor{red}{[TBD-m9: ...]}` sentence, at the end of the "Where does extrapolation push the levels?" paragraph in V.B.
  - Fill in: the M reached by our sweep (X); the fitting range (Y); the direction and size of the misstatement of the marginal value of tokens at M ≈ 1,000–2,000 by the Chinchilla form (Z); and the implied bias in the reference wedges (W%).
- There are no other placeholders.

## 5. New bib keys
None. Every key used is already in `paper/references.bib`:
- sardana2024beyond, deloecker2012markups, besiroglu2024chinchilla, hoffmann2022training, li2025predictableb, gadre2024language, grattafiori2024llama, bhagia2024establishing;
- epochai2026data, dey2023cerebras, touvron2023llama, biderman2023pythia, kaplan2020scaling;
- gandhi2020identification, raval2023testing, demirer2025emerging, openrouter2026models, lmarena2026leaderboard, brown2020language.

## 6. Open issues for the integrator
1. **Label collision.** `appendix_proofs.tex` already defines `\label{prop:wedge}`, `prop:pi`, `prop:info`, `prop:dmr` and `prop:transmission` for its A-numbered results. These are the labels the main text is told to use. One set must be renamed, e.g. the appendix to `prop:A-wedge`; otherwise LaTeX gives "multiply defined" and references resolve to the wrong number.
2. **Significance stars.** AEA style discourages them, so Table 7 has none (the source .tex had them). Other writers may differ; harmonize.
3. **Table numbering.** In the full paper these should come out as Tables 6–7 and Figures 5–6 if the earlier sections place their floats in plan order.
4. **Plan vs memo.** I found no numeric discrepancies; the plan's V numbers all match the memo and CSVs after the review. Two small differences:
   - Table 6 includes Llama 2 70B, Chinchilla and GPT-3, which are not in `m3_wedge_table5`.
   - The "own law" column is filled only for Meta (Meta A2/A1) and OLMo 2 (OLMo ladder: 2.01 for 1B, 0.91 for 32B).
5. **Llama naming.** Text and Table 6 say "Llama 3 8B/70B" (identical N and D to 3.1 8B/70B, as noted in the table note), while `m3_wedge_table5` labels them Llama-3.1. The 405B is "Llama 3.1 405B".
6. **Word count.** About 300 words over target. If space is tight, the first candidates to cut are the over-identification paragraph (V.E, third paragraph) and the distillation sentence in V.G.
7. **Figure 5 labels.** `m3_wedge_fig4` has in-figure labels such as "Cerebras-GPT 6.7B" and "BLOOM 176B". Its legend says "Other open-weight models (Epoch, unverified D)"; the figure note matches this.
