# Writer notes: Section III "Data" and Online Appendix B "Data Construction"

Writer: section writer "data", 2026-09-24.

## Files written

| File | Content |
|---|---|
| `paper/sections/data.tex` | Section III, about 1,200 words of prose. The target was about 1,000. |
| `paper/sections/appendix_data.tex` | Online Appendix B, about 3,500 words. The target was about 2,500; see open issue 6. |
| `paper/tables/table2_data.tex` | Table 2, label `tab:data`. |
| `paper/tables/tableB1_conventions.tex` | Table B1, label `tab:app_conventions`. |
| `paper/tables/tableB2_sweep_design.tex` | Table B2, label `tab:app_sweepdesign`. |
| `lit/bib/extra_writing_data.bib` | New bib entries (listed below). |

The appendix also includes Figure B1, `m2_design_planes`, with label `fig:app_designs`.

**Test compiles.**
- `code/paper/test_section.sh data appendix_data` compiles with no LaTeX errors. Its only warnings are undefined cross-references to other sections and my 6 new bib keys, which are not yet in references.bib.
- A stub build in my scratchpad defines stub labels for the other sections and adds the new bib. It compiles with zero overfull boxes, zero undefined references and zero duplicate hyperref anchors. I checked every page as a PNG: all tables fit the text width, and Figure B1 sits on its own float page.

## Numbers used, with sources

### Section III text and Table 2, Panel A (public sweeps)

**Chinchilla**
- 245 digitized runs, 240 after dropping the 5 highest-loss runs.
- Loss quantized to about ±0.01; 9 nominal budgets.
- Source: m1 memo §2 (Data); m1 review.

**Design statistic sd(ln M | ln C)** (m2 memo §2.1 table; `output/tables/m2_table3_technology.csv`, column `sd_offpath`, which is an OLS residual SD with ddof = 2; I reproduced it from the `data/processed/m2_techpanel/panel_*.csv` files):
- Chinchilla 1.35
- Farseer 1.97 (1.9671)
- Gadre, pooled 104 runs: 1.44
- OLMo ladder 0.94
- Muennighoff, single epoch: 1.97 (1.9706)
- DataDecide: 0.73 over all 21,888 checkpoints (the estimation sample, as in m2 Table 3). It is 0.71 over unique (size, step) points, which is the figure used in m2 memo §2.1 and in Figure B1.
- **Discrepancy handled.** The m2 memo quotes 0.71; Table 3 and the technology section use 0.73. I use 0.73 in the text and in Table 2, and note 0.71 in footnote c and in the Figure B1 notes.

**sd(ln w) = 0.48 on Chinchilla**
- Source: `paper/notes/theory_main_text.md` (m7). The identification section also uses it.
- Consistency check: 1.349 × (0.347 + 0.367)/2 = 0.482.

**N and M ranges** (computed from m2's harmonized panels `data/processed/m2_techpanel/panel_*.csv`):

| Sweep | N | M |
|---|---|---|
| Chinchilla (n = 240) | 5.73e7–1.62e10 | 0.456–341.1 |
| Farseer | 9.96e7–6.37e9 | 0.314–2,570 |
| Gadre | 1.06e7–6.89e9 | 5–640 |
| OLMo | 1.9e8–3.17e9 | 10–200 |
| Muennighoff, epochs = 1 (33 runs) | 7.1e6–8.67e9 | 0.0356–303 |
| DataDecide | 3.74e6–1.18e9 | 5.12–110 |

**DataDecide structure**
- 1,100 runs = 25 recipes × 44 (size, seed) cells; 14 sizes × 3 seeds, with 5 seeds at 1B.
- Checked in the panel file.

**Llama 3 and Marin** (computed from `data/raw/isoflop_experiments/isoflop_experiments.csv`, Apache-2.0 compilation):

| Experiment | Runs | Budgets | N | M | Other |
|---|---|---|---|---|---|
| Llama 3 | 133 | 10 | 5.87e7–1.68e10 | 0.436–642.5 | Loss 0.694–0.932; C 6e18–1e22 |
| Marin (3 corpora) | 85 + 85 + 88 = 258 | 7–8 | 1.57e8–1.18e10 | 0.101–2,751 | C 1.8e18–3e20 |

- The (Mis)Fitting series has 176 points.
- The Marin budget mismatch of −7% to +35% is from the m1 review.

**Porian**
- 975 runs: RefinedWeb 772, OWT2 203; base 264, tuned 61, sweep 566, seed 84.
- 16 architectures, width 96–1,504, depth 3–30, vocabulary 50,432, sequence length 2,048.
- Budgets 1.25e16·2^i for i = 0..11.
- Reproduced within 0.003.
- Source: m8 memo §2.1; m8 review.
- N standard count (`params`) 5.17M–902M; non-embedding 0.33M–826M.
- Source: `data/processed/m8_measurement/porian_architectures.csv`.

**Step Law**
- 1,911 runs in 17 cells.
- N {215, 268, 429, 537, 1,074}M; D 4B–100B; M 18.6–466.
- Up to 12 LRs × 10 batch sizes per cell; one cell has 5 LRs and 47 runs.
- 9.5% of runs diverged.
- Sources: m8 memo §1.5, §2.3; `data/processed/m8_measurement/steplaw_cells.csv`.

### Table 2, Panel B and Appendix B.4 (our experiment; design numbers)

The design numbers come from `code/sweep/{run_grid.py, gpt_mlx.py, train_sweep.py, prep_data.py}`, `data/processed/sweep/meta.json` and `data/processed/sweep/lr_fit.json`. I computed the derived numbers with a scratch script, `scratchpad/data_writer/design.py`. These are deterministic properties of the fixed design, **not results**. m9 task 1 will recompute them; the integrator should check them against m9's design table.

**Parameter counts**
- Non-embedding N = L(12d² + 2d) + d. The values are 393,856 / 1,328,448 / 3,148,032 / 6,147,520 / 10,621,824 / 16,865,856 / 25,174,528 / 49,165,440, which match the `N_nonemb` field in `results.jsonl` for the widths already run.
- Total N adds 8,192d: 1.44M–54.4M.
- Embedding share falls from 0.73 to 0.10. The m9 spec says "20–73%"; my computation gives 10% at d = 640.

**Endpoints and ratios**
- 44 endpoints per lab: 5 × 6 + 2 × 5 + 1 × 4.
- Actual tokens are ceil(D/16,384) × 16,384.
- M with non-embedding N: 0.509–2,031. With total N: 0.460–555.

**Off-path spread**
- 1.961 with non-embedding N; 1.742 with total N.
- Computed as the OLS residual SD with ddof = 2, the same as m2.

**Compute**
- 6ND with non-embedding N: 5.9e13–6.0e16. With total N: 2.2e14–7.1e16.
- Hence "under 10^17 FLOP", and "five orders below" the Chinchilla maximum of 1.3e22.

**FLOPs per token** (including attention and the unembedding; `flops_per_token` in gpt_mlx.py)
- 1.14–3.83 × 6N_nonemb.
- 1.03–1.05 × 6N_total.

**Learning-rate rule and calibration**
- Peak LR by width from lr* = 3.066988e-3 · (d/256)^−0.8997: 5.72, 3.97, 3.07, 2.51, 2.13, 1.85, 1.64 and 1.35 × 10⁻³.
- Calibration optima: 5.75e-3 (d = 128), 2.47e-3 (d = 320), 1.66e-3 (d = 512). Source: `lr_fit.json`.
- Calibration grid: LR {1, 2, 4, 8} × 10⁻³ at D = 50M, FineWeb-Edu. Extrapolation clipped at 0.35 log points. Source: `fit_lr.py`.

**Corpora and tokenizer**
- Validation tokens: 24,646,462 (edu), 16,672,041 (web).
- Training tokens: 1.76B (edu), 1.70B (web).
- Bytes per token on validation: 3.8857 (edu), 3.7426 (web).
- The first 20,000 documents of each corpus are held out.
- The tokenizer is trained on about 300M characters per corpus.
- Source: `meta.json`; `prep_data.py`.

**Optimization and evaluation**
- Warmup 250 steps = 4,096,000 tokens; stability constant 1e-8; clip 1.0; weight decay 0.1; decay rates 0.9 and 0.99.
- Evaluation on 1,048,576 tokens = 4,096 sequences of 256.
- Seeds: widths {128, 256, 384} × seeds {1, 2} × D {50, 100, 200}M = 18 endpoints, FineWeb-Edu only.
- LR calibration: 12 runs.

### Table 2, Panel C and Appendix B.5–B.7 (observational data)

**Sample B**
- 173 general-purpose models (188 − 15 code), 68 families, dates 2021-03 to 2025-11.
- N 7.04e7–4.06e11 (active N for MoE); M 2.08–60,398.
- Sources: `output/tables/m3_wedge_models.csv` (sample B, code = False); m3 memo; `data/processed/m3_wedge/headline.json`.

**Production-scale universe**
- 335 models: 176 from Sample A plus 159 from Sample B; 271 open; 94 developers.
- N 1.35e8–6e11; M 0.199–60,398; 6ND 1.13e21–3.8e25; 2019-05 to 2026-06.
- Source: `m3_wedge_models.csv` with the flag `prod = core & Cmp ≥ 1e21`, as in `analysis.py`.

**Choice-data construction** (m3 data appendix `data/processed/m3_wedge/data_appendix.md`; audit `output/tables/m3_wedge_data_audit.csv`)
- Sample A exclusions: 2,070 rows → 342 kept, with the reasons summing to 1,728.
- Word-to-token conversions (7) and D = C/(6N) (67).
- Six primary-source D corrections; 11 rows flagged as mismatched.
- Sample B:
  - 148 ObsScaling models, 9 fixes, 17 dropped;
  - 57 curated models, 47 verified against the card by regex;
  - 143 exact parameter counts; 156 embedding counts;
  - 119 Epoch matches, of which 5 differ from Epoch.
- Flags: code 15, distilled 10, synthetic 16, continued 8, multimodal 5, MoE 13, RNN 6.
- Gemma 3 vision encoder: 417M. The causal-mask fix affects up to +36% (Pythia-70m), from m4.

**Usage data** (`output/tables/m3_wedge_validation.csv`; m3 memo H6)
- 164 models from 26 developers with Hub metadata.
- 46 LMArena matches.

**Tokenizer and input-error sensitivity** (m3 data appendix, Units)
- 20% tokenizer difference → about 0.07 in ln w.
- ∂ln w/∂ln D = β ≈ 0.37; ∂ln w/∂ln N = −α ≈ −0.35.
- A 10% error in D moves w by about 4%.

**"55% of Sample B have M > 341"**
- Source: m3 memo H1 (Rev); `headline.json` `sampleB_share_Mx_chin` = 0.549.

**Benchmark panel** (m4 memo §2.1; `output/tables/m4_observational_design_audit.csv`; `data/processed/m4_observational/obs_panel.csv`, `main` = True)
- Samples: 128 models, 38 families, 22 developers, 2021-03 to 2024-07. Overlap sample 90, hull 57.
- Design: off-path spread 1.64; within-family SD of ln D 0.42 vs 1.29 for ln N; 18 families (66 models) vary D; corr(ln N, ln D) = 0.46; C = 6ND in 100% of rows.
- Ranges: N 7.04e7–1.8e11; M 2.08–24,290.
- Construction: 5 duplicate Pythia rows dropped; 17 models with inflated counts corrected.

**Ho et al.**
- 408-row sheet → 231 rows, 144 papers: WT103 103, PTB 80, WT2 48.
- Epochs known for 155 rows.
- Ranges: N 2e6–2.8e11; dataset size 8.9e5–1.4e12; D/N 0.00135–203.5; publication years 2012–2023.
- Sources: m5 memo §2.1; `code/analysis/m5_progress/common.load_ho()`, which I re-ran.

**Epoch allocative sample**
- Funnel: 792 → 544 → 454 → 428 → 339 → 247.
- N 6.5e7–5.4e11; M 0.30–24,000.
- Sources: m5 memo §2.1; `data/processed/m5_progress/epoch_lm_sample.csv`.

**Licenses**
- Source: `lit/notes/data_sources.md` and the m2 and m1 memos.

### Appendix B.1–B.3 (public-sweep details)

**Farseer**
- 25 sizes × up to 19 D values.
- Embedding check: N including embeddings exceeds N by exactly 131,072h.
- Compute column / N = 6.6–8.5; 4 reruns; corr(ln N, ln D) = −0.04.
- Sources: m2 memo §2.1; m2 review.

**Gadre**
- C4 34 / RedPajama 35 / RefinedWeb 35 runs; 174M validation tokens.
- Sampling SE of ln L is 0.0016, against a residual RMSE of 0.023–0.030.
- Source: m2 §2.1.

**OLMo, Muennighoff, DataDecide**
- OLMo: 30 runs.
- Muennighoff: 229 losses parsed, 33 single-epoch runs, 2 runs with D/N < 0.4.
- DataDecide: D ≈ 85–110 N; corr(ln N, ln D) = 0.91.
- Source: m2 §2.1.

**Chinchilla**
- Nominal budgets 6e18–3e21.
- Offset −0.018 dex; membership window ±0.045.
- 137/245 runs on IsoFLOP profiles (132/240).
- Transverse share 33%; the ml_scalefit series is a 124-row subset.
- 4 of the 5 dropped runs have D/N < 0.4 (the fifth 0.4045); all 5 are in the 10¹⁹ budget.
- Sources: m1 §2, H5; `m1_chinchilla_design_variance.csv`.

## Claims made, with the caveats included

1. **The off-path spread times (α+β)/2 equals the transverse spread (dispersion of log wedges).** It governs information: second order for M*, fourth order for σ*.
   - Caveat implicit in the text: this holds under the Chinchilla form, around the best-fitting log-linear path.
2. **Cross-sweep comparisons use only unit-free objects** (α, β, a, γ, σ, κ), never E, A, B or M*.
   - This follows m2 open issue 3. Parameter-count conventions still matter; see Section IV.D.
3. **DataDecide is used only for across-recipe comparisons.** The schedule artifact is stated, and the untestable equal-effect assumption is stated in the appendix (m2 review item 13).
4. **Our experiment.**
   - Limitations stated in the text: under 10¹⁷ FLOP; the LR is tuned to width only, not to D or the lab, which is a flexible-input bias (m8).
   - Also stated in the appendix: the LR is calibrated on one lab at one budget; branch endpoints share their trunk, so noise is correlated within a width and inference should cluster by width.
   - No results are reported.
5. **Observational data do not lack off-path variation (spread 1.64); they lack exogeneity.**
6. **Disclosure selection.** Mistral and Mixtral drop out.
7. **The Chinchilla "ml_scalefit" series is not independent**, so it is not used.
8. **Llama 3 N = C/(6D)** assumes Meta's budgets are 6ND, and the loss units are unknown (m1 review).
9. **The observational benchmarks were evaluated with their developers' own harnesses.** The mapping between formats is part of the maintained null hypothesis in Section VI (m4 §2.3).

## Cross-references assumed to exist in other sections

- **Section labels:** `sec:framework`, `sec:ident`, `sec:tech`, `sec:wedge`, `sec:obs`.
- **Subsection labels:** `sec:tech:chinchilla`, `sec:tech:measurement`, `sec:tech:experiment`. These exist now in technology.tex; if they are renamed, update data.tex and appendix_data.tex.
- **Proposition 4:** `prop:info`.
- **My own labels:** `sec:data`, `tab:data`, `app:data`, `app:sweeps`, `app:chinchilla`, `app:isoflop`, `app:sweep`, `app:choices`, `app:obspanel`, `app:ho`, `app:licenses`, `tab:app_conventions`, `tab:app_sweepdesign`, `fig:app_designs`.
- **The ln w relation** ln w = ((α+β)/2) ln(M/M*(C)) is assumed stated in Section I (framework, wedges subsection). It is equivalent to (1/σ* − 1) ln(M/M*).

## Placeholders (TBD-m9)

1. data.tex, end of the training paragraph: `[TBD-m9: completed endpoints and seed replicates; total training hours.]`
2. appendix_data.tex, Hardware paragraph: `[TBD-m9: completed endpoints per lab, seed replicates completed, any diverged or failed runs, and total wall-clock hours.]`

If m9's design table differs from my computed design numbers, update:
- Table 2 Panel B (2 × 44; 0.39–49M; 0.51–2,031; 1.96; footnote a: 1.4–54M, 0.46–555, 1.74);
- Table B2.

## New bib keys (in `lit/bib/extra_writing_data.bib`, verified)

| Key | Reference |
|---|---|
| `penedo2024fineweb` | Penedo et al., The FineWeb Datasets, NeurIPS 2024 Datasets and Benchmarks, arXiv 2406.17557 |
| `loshchilov2019decoupled` | AdamW, ICLR 2019 |
| `su2024roformer` | RoFormer, Neurocomputing 568:127063 |
| `zhang2019root` | RMSNorm, NeurIPS 2019 |
| `hannun2023mlx` | MLX, GitHub 2023 |
| `sennrich2016neural` | BPE, ACL 2016 |

Existing keys used: hoffmann2022training, besiroglu2024chinchilla, li2025predictableb, li2025predictablea, gadre2024language, bhagia2024establishing, muennighoff2023scaling, magnusson2025datadecide, grattafiori2024llama, czech2026llama3isoflop, czech2026problems, marin2026ladders, porian2024resolving, li2025misfitting, kricheli2026tokens, hagele2024scaling, ruan2024observational, maiapolo2024sloth, epochai2026data, ho2024algorithmic, openrouter2026models, lmarena2026leaderboard, biderman2023pythia, pearce2024reconciling.

## Open issues for the integrator

1. **Duplicate labels.** appendix_proofs.tex defines `prop:info`, `prop:wedge`, `prop:dmr`, `prop:transmission` and `prop:pi`, and the main text uses the same labels (identification.tex, framework.tex). These collide in the full build. My `\ref{prop:info}` in data.tex should resolve to main-text Proposition 4.
2. **Appendix numbering.** appendix_data.tex resets the table and figure counters to B1, B2, … and sets `\theHtable` and `\theHfigure` to avoid hyperref clashes. Appendices C and D must reset their own counters (C1…, D1…); otherwise they will continue as B3, B4, ….
   - `\ref` to an appendix *subsection* renders as "B.B1" in AEA.cls, so I avoided such references. Other writers should do the same, or the integrator should redefine `\p@subsection` inside the appendix.
3. **Plan versus assignment on Table 2 columns.** The plan asks for a "license" column; the assignment asks for "role in the paper". I used role in the paper and moved licenses to Appendix B.8.
4. **"Exact HF parameter counts" for Sample B** (plan §III.C) holds for 143 of 188 models; the rest use reported sizes. The text says "where available".
5. **Figure B1 is not in the plan's figure list.** It is m2_design_planes, used in Online Appendix B, and it adds no main-text figure. It shows 0.71 for DataDecide (unique design points), while Table 2 shows 0.73 (all checkpoints); both notes explain the difference.
6. **Lengths.** The main text is about 1,200 words against a target of about 1,000. The appendix is about 3,500 against about 2,500. Candidates to cut:
   - main text: the observational-data paragraph (usage sentence) and the Chinchilla paragraph;
   - appendix: "Sweeps we did not use", the Farseer quirks, and the details of the Sample B corrections.
7. **Rendered years differ from key years.** The bib years for gadre2024language, bhagia2024establishing, magnusson2025datadecide and maiapolo2024sloth render as 2025. That is fine, but "Gadre et al. (2025)" appears while the key says 2024. Check this is intended (published versions).
8. **m9 spec says the embedding share is "20–73%"**; the correct range for the tied embedding is 10–73% (d = 640: 5,242,880 / 54,408,320 = 0.096). I use 10–73%.
