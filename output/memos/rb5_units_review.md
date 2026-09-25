# Review memo: module rb5_units (budget-level decision units, readings, trend and serving inference)

Package WP4b-review, 2026-09-25. Independent reviewer of `output/memos/rb5_units.md` (coder A, the WP4b analysis agent) and adjudicator of the disagreements between coder A and the blind coder B. Binding specification: `paper/notes/round3_fixlist.md` (Section 0, decisions D-1 and D-2, items W1, W2, W14, W18, W19, W24). Protocol: `paper/notes/rb5_reading_protocol.md` (version 1, Amendments 1-3, and the resolution record added by this review). No paper section or paper table was edited. No MLX, training or evaluation job was started; every run was CPU-only under `nice -n 10`. No FineWeb estimate, sigma* for FineWeb or tilt was computed.

## Verdict

**Accept, with four corrections that this review made (marked [review]) and one open dependency.**
- The headline numbers reproduce exactly in an independent re-implementation: 49 decision units, median w 3.77 and s 0.735, 29 point estimates with a median of 0.737, the trend points, and the serving coefficient. Their bootstrap and randomization intervals agree within Monte Carlo error.
- The blind second coding agrees with coder A on 15 of 17 readings (Cohen's kappa 0.83). The adjudication keeps coder A's readings, so no identification class changes: 29 point estimates and 20 lower bounds.
- Three re-runs from scratch on the resolved codes are identical, apart from run time, PDF metadata and this review's own edits between runs. The last run also rebuilt the technology registry. Every numerical table that does not depend on the codes is identical to coder A's run.
- The corrections:
  - the "point if choice" sensitivity now keeps SmolLM 135M and 360M as lower bounds (35 units, not 36);
  - the stated-rationale numbers follow the resolved codes: the compute-optimal share in 2024 is 0.04, not 0.07;
  - randomization-inference p-values below 0.001 now print as "< 0.001";
  - a technology rebuild now reproduces the cached reference draws bit for bit with any number of worker processes (Section 3).
- The open dependency: Table 2's sign columns still read WP4a's `model_bounds_S1.csv` of 12:17. They must be redrawn after WP4a's final S1 run (`code/analysis/rb2_decisions/run.py --exhibits-only`).

## 1. What was checked

1. **Adjudication:** all six unit-field disagreements in `data/processed/rb5_units/readings_disagreements.csv`, decided from the saved primary sources under protocol Sections 5-9 (Section 2 below). The disagreements file has 9 rows because derived fields repeat: reading and D margin for Granite 3.0 and the Llama 3 herd, and R-NONE for DeepSeek LLM.
2. **Agreement:** recomputed with my own confusion-matrix implementation of Cohen's kappa. Every quote of both coders (291) was re-verified against the saved text. `verify_readings.py` accepts both files and `readings_final.csv`. Coder B's script, now shipped as `code/analysis/rb5_units/coderB.py`, regenerates `readings_coderB.csv` byte for byte. `compare_AB.py` regenerates `readings_disagreements.csv` byte for byte.
3. **Re-runs:** `code/analysis/rb2_decisions/run.py` (about 11 minutes, 2 worker processes) and then `code/analysis/rb5_units/run.py --B 9999` (about 6.5 minutes, no cache), each run twice, compared file by file (Section 3).
4. **Independent re-implementation** in my own code, which reads only `clean_models.csv`, the resolved codes and the reference technology's parameters and draws (Section 4). It covers:
   - the unit construction (both greedy directions);
   - the reference wedge from the M-form of the technology;
   - the unit wedge;
   - the classes and the point-identified median with its joint wild-bootstrap interval;
   - the developer-cluster trend bootstrap;
   - the model-level and developer-level serving randomization inference;
   - the within-developer growth points.
5. **Data corrections of W24 checked against the sources:** MPT-30B, StableLM-Alpha, OLMo 2, Yi and Qwen2.5, the Table E3/E4 recomputation and the cleaning steps (Section 5).
6. **Technology cache:** the ra2 registry (`techs_cache.pkl`, 45 technologies with draws) was rebuilt in memory and compared with the cache without overwriting it (Section 3).

## 2. Agreement and adjudication

**Agreement between coder A and the blind coder B** (`output/tables/rb5_units_agreement.csv`; my recomputation is identical):

| Field | Units | n | Raw agreement | Cohen's kappa |
|---|---|---|---|---|
| Reading | common-budget | 17 | 0.882 (15/17) | 0.83 |
| Token margin (D) | common-budget | 17 | 0.882 | 0.78 |
| Size margin (N) | common-budget | 17 | 1.000 | 1.00 |
| Own cap | all | 49 | 0.959 | 0.90 |
| Own cap | size-specific members and singletons | 32 | 1.000 | 1.00 |
| R-CO (compute-optimal) | all | 49 | 0.980 | 0.85 |
| R-INF (inference cost) | all | 49 | 0.980 | 0.96 |
| R-DEV (device or memory) | all | 49 | 1.000 | 1.00 |
| R-NONE | all | 49 | 0.980 | 0.96 |
| Deployment rationale | all | 49 | 1.000 | 1.00 |
| Synthetic flag | all | 49 | 1.000 | 1.00 |

**Resolutions.** They are recorded in `data/processed/rb5_units/readings_final.csv`, with both original codes of every field, the decision, the deciding rule and the reason. The file is written by `code/analysis/rb5_units/resolve.py`.

| Unit | Field | A | B | Resolved | Reason (protocol rule) |
|---|---|---|---|---|---|
| Granite 3.0 | reading | cap or choice | choice (K2) | **cap or choice** | See the Granite 3.0 note below the table (5.1: no C and no K evidence; K2 not established). |
| Llama 3 herd | reading | cap or choice | choice (K1) | **cap or choice** | See the Llama 3 herd note below the table (5.1, 7; fix list W1(c)). |
| DeepSeek LLM | own cap | 0 | 1 | 0 | 6, last bullet: when every member of a common budget is capped, the cap is coded in the reading (C1 here). Class unchanged (cap+menu, lower bound). |
| SmolLM 135M, 360M | own cap | 0 | 1 | 0 | Same convention (C2 in the reading). The class is set by the mapping correction below. |
| DeepSeek LLM | R-CO | 1 | 0 | **0** | 7: "Under the guidance of our scaling laws, we build ... models" does not say that the sizes or the 2T budget were set by an allocation. The report calls the sizes "two prevalent used open-source configurations" and names hyperparameters and performance forecasts as the uses of its laws. This is consistent with both coders leaving LLaMA's "inspired by the Chinchilla scaling laws" uncoded. |
| StableLM 2 1.6B | R-INF | 0 | 1 | **1** | 7 with Amendment 3: "available directly on-device without the computational overhead of larger models" states an inference benefit of the size. The deployment rationale is unchanged, because R-DEV = 1 for both coders. |

- **Granite 3.0.** Coder B infers K2 from source sizes: FineWeb has "more than 15T tokens" against a 10T stage 1. That size is before IBM's own processing. Section 3.1 of the report applies exact and fuzzy deduplication, HAP filtering and Gopher and KenLM quality filters to the curated web data, which include FineWeb and DCLM. The usable pool is therefore not stated, and the stage-1 web share is read off a figure. With no C and no K evidence, the residual code is "cap or choice".
- **Llama 3 herd.** Coder B's K1 is the flagship's compute-optimal sizing ("suggests training a 402B parameter model on 16.55T tokens", then "decided to train a flagship model with 405B parameters"). The protocol's own example of the R-CO rationale is this report's wording, "approximately compute-optimal size for our training budget". Both coders coded that rationale. The quote does not show that the token budget lay below the data available: the law's 16.55T exceeds the corpus of "about 15T", and the flagship trained on 15.6T, about the corpus size. So the evidence is size only. Fix list W1(c) specifies "cap or choice" for the herd unless the sources state that the corpus bound, and they do not.

**Mapping correction (Section 8; disclosed in the protocol's resolution record).** SmolLM 135M and 360M trained on 600B tokens of a stated 252B-token corpus: 28B Cosmopedia v2, 4B Python-Edu and 220B FineWeb-Edu, about 2.4 passes. Their reading is "cap or choice" because K1 (a stated stopping rule) is also present. Every member repeated the corpus, however, so each member's own cap binds whatever the reading of the token margin. The unit is now classed "lower bound" rather than "lower bound (point if choice)"; `readings.id_class` implements this. Coder A's own note says the same ("the repetition itself makes the share a lower bound"), but the code did not implement it.

**What changes.**
- No count of point estimates or lower bounds changes: 29 and 20, of which 14 are plain lower bounds and 6 are "cap or choice" lower bounds.
- The "point if choice" sensitivity has 35 units, not 36: median 0.735 [0.657, 0.788], against 0.736 before.
- The rationale flags change for DeepSeek LLM (a 2024 unit) and StableLM 2 1.6B (Section 6).
- Coder B's readings would make Granite 3.0 and the Llama 3 herd point estimates: 31 point estimates, median 0.737 [0.672, 0.790]. That is a robustness row (protocol Section 11), not the primary.

**Disclosure.**
- The adjudicator had seen version 3's and coder A's wedges before resolving.
- Both reading decisions keep the more conservative class (lower bound) and are required by the protocol's text and by fix list W1(c), which was written before any coding.
- Coder B's blindness exposure (the round-2 Llama 3 herd reading, seen in `evidence.py`) did not pull coder B toward that reading. Coder B coded "choice", not the round-2 "cap".
- Coder B's log (`round3_WP4b_blind_log.md`, Section 1) says "15 size-specific members, 17 singletons". The coding frame and both CSVs have 17 size-specific members and 15 singletons. This is a slip in the log only.

**Agreed codes that remain the weakest** (not changed, because both coders agree):
- **MPT, choice (K2).** It rests on "1T tokens sampled according to this mix" and on the 30B's different proportions of the same subsets. The subsets' sizes are not in the saved texts.
- **Yi, menu (K1).** It rests on "we increase the pretrain data scale to 3.1T tokens to compensate for the decreased compute flops". Coder B notes that this is close to the "beyond compute-optimal for inference" rationale, which Section 5.1 says is not token-margin evidence.
- Without these two, 27 point estimates remain, with a median of 0.745 [0.691, 0.793] (cluster [0.575, 0.830]). This row is now in `rb2_decisions_headline.csv` and `rb5_units_paper_numbers.csv`.

## 3. Re-runs and determinism

- **Coder A's run against run 1 on the resolved codes.** Only the expected files differ:
  - `decision_units.csv`: SmolLM's class, DeepSeek LLM's R-CO, StableLM 2 1.6B's R-INF, and the codes file name;
  - the unit files, which gain a `point_if_choice` column;
  - `rb2_decisions_headline.csv`: the corrected "point if choice" row and four new rows (coder B's readings, and without MPT and Yi, each with the wild and cluster intervals);
  - Table 2 (SmolLM moves from lb^c to lb) and the readings table;
  - `rb5_units_classes.csv`, `_common_budget_units.csv`, `_rationale.csv` and `.tex`, `units_primary.csv`, `rb5_units_paper_numbers.csv`.
- **Unchanged.** Every other rb2 and rb5 table is byte-identical to coder A's run: conduct, sign, trend, LODO, technologies, conventions, cleaning, post-training, trend bootstrap, randomization inference, within-developer, MoE, synthetic, repetition and level sensitivity. That is a determinism check across two sessions of the same code.
- **Run 1 against run 2** (same code and codes, both from scratch; rb5 without `--reuse`). 71 of 75 files are byte-identical. The other four are:
  - the two `headline.json` files (only `runtime_s` differs);
  - the ECDF PDF (creation-date metadata; the PNG is identical);
  - the readings table, whose note I edited between the runs.
- **Technology cache.** I rebuilt ra2's registry in memory with 2 workers (model-free designs, B = 999; kappa-free wild bootstraps, B = 399) and compared it with `techs_cache.pkl`.
  - All 45 technologies came back with identical parameters and identical sets of draws.
  - Three chunked bootstraps (the reference `chin_q`, `chin_ne`, and DeepSeek's law, which takes the `chin_q` draws) returned their draws in a different order. Each draw has a fixed seed, but the chunks `seeds[i::n_proc]` are concatenated in an order that depends on the number of workers, and the cache was built with 4.
  - The order leaves every single-technology interval unchanged. It does change the pairing of reference and lab-own draws in the "lab-own where available" rows, so a replicator rebuilding with another worker count would get slightly different intervals there.
  - [review] `code/analysis/rb2_decisions/base.py` now restores the 4-worker order after a rebuild. With 2 workers the rebuild then matches the cache bit for bit.
  - Run 3 (`run.py --rebuild-techs`, then `rb5_units/run.py` without a cache) is the final state of every output. The rebuilt `techs_cache.pkl` has the same draws, in the same order, for all 45 technologies. Against run 2, 71 of 75 files are byte-identical; the other four are the two `headline.json` run times, the PDF metadata, and the three RI p-values in `rb5_units_paper_numbers.csv` that now print "< 0.001".

## 4. Independent re-implementation

| Quantity | Module | Independent | Note |
|---|---|---|---|
| Units (partition) | 49: 17 common-budget, 17 size-specific members, 15 singletons | same partition; identical under ascending and descending greedy clustering, and every group has max/min D <= 1.10 | the 10 percent rule gives a unique partition here |
| Reference w (77 models) | `w_ref` | max relative error 2e-15 | from ln w = (S/2)[ln M - ln M*(C)] |
| Unit wedge (harmonic, compute-weighted) | `decision_units.csv` | max relative error 2e-15 | |
| Median s, 49 units | 0.7345; w 3.767; share over-trained 0.980 | 0.7345; 3.767; 0.980 | |
| Wild interval, 49 units | [0.6575, 0.7822] | [0.6576, 0.7822] | module cycles the 399 draws 3 times; same draws |
| Classes | 29 point, 14 lower, 6 lower (point if choice) | same | |
| Point-identified median | 0.7373 [0.6717, 0.7899] | 0.7373 [0.6718, 0.7898] | |
| Lower-bound median; point if choice | 0.7104; 0.7345 (35) | 0.7104; 0.7345 (35) | |
| Compute-weighted s by year, 49 units | 0.242 [0.09, 0.51], 0.538 [0.34, 0.79], 0.814 [0.64, 0.89] | same points; [0.10, 0.51], [0.35, 0.80], [0.64, 0.89] | own RNG stream, B = 9,999 |
| Change 2024 to 2025 (weighted) | 0.276 [-0.12, 0.44] | 0.276 [-0.12, 0.45] | |
| Change 2023 to 2025 (weighted) | 0.573 [0.19, 0.72] | 0.573 [0.19, 0.72] | |
| Median decision share by year | 0.487, 0.774, 0.745 (change 2024 to 2025: -0.029 [-0.19, 0.13]) | same points; [-0.19, 0.13] | the median falls from 2024 to 2025 |
| Serving coefficient (models) | 0.560, CRV1 t 2.35 | 0.560, t 2.35 | |
| RI p, model-level code, year x compute terciles | coefficient < 0.001; t 0.065 | < 0.001; t 0.059 | tercile edges assigned slightly differently |
| RI p, year strata | < 0.001; t 0.066 | 0; t 0.066 | |
| Developer-level RI (weighted; unweighted; without Alibaba) | 0.104; 0.278; 0.945 | 0.096; 0.273; 0.948 | Monte Carlo s.e. about 0.003 |
| Within-developer growth, points (pooled; pooled with ln C; FE; FE with ln C) | 1.858; 2.138; 1.591; 2.697 | 1.858; 2.138; 1.591; 2.697 | |

**Reading of the serving test.** The specified permutation (the model-level code permuted within year x compute strata) treats models as the units of assignment. 19 of the 26 served models are Alibaba's, so the coefficient's permutation distribution is too narrow, and p < 0.001 is not credible evidence. The clustered t version (0.06) and the developer-level version (0.10; 0.94 without Alibaba) are the relevant ones. This agrees with decision D-8's sentence, "with six treated developers the test cannot detect a relation". The paper should report the specified RI with this caveat, or lead with the developer-level RI, as coder A proposed.

## 5. Data corrections (W24) against the sources

| Item | Source check | Status |
|---|---|---|
| MPT-30B, D = 1.05T | `mpt30b_blog`: "we first pre-trained on 1T tokens using sequences that were 2k tokens long, and continued training for an additional 50B tokens using sequences that were 8k tokens long" | correct. w 1.198 to 1.224; the MPT budget stays common (1.05/1.00 <= 1.10); family share 0.25 |
| StableLM-Alpha 3B, 7B: D = 0.8T, N = 3,638,525,952 and 7,869,358,080 | `stablelm_github` table rows (800B) | correct |
| OLMo 2 accounting: 1B and 7B 4.05T, 13B 5.15T, 32B 6.15T | report: 7B "averaging three training runs on 50B"; 13B and 32B "three runs on 100B tokens and one run on 300B tokens" (mean 150B); release posts: 7B one epoch up to 4T, 13B up to 5T, 32B 1.5 epochs up to 6T | convention correct and logged. Two source inconsistencies are not logged in `d_audit.csv` (see the notes below the table) |
| Yi, 3T | report: "we overtrain the model on more tokens (3T) than the compute optimal (around 1T)"; also "3.1T" | kept at 3T per X8(m); logged |
| Qwen2.5 per-member budgets | blog: "up to 18 trillion tokens" | flag logged. Without Qwen2.5 and Qwen3 the median decision share is 0.735 (47 units) |
| Table E3 model-level columns (audited counts) | compared cell by cell with the audit's corrected values (numbers_appE M2): Hoffmann A3 2.08/0.88/0.52 ... MiniCPM 1.17/0.62/0.14; range 0.14-0.84 | all 18 changed rows match to the printed precision |
| Table E4 (Farseer total; MiniCPM) | 2.503/0.883/0.600; 1.168/1.242 (head-FLOP)/0.623/0.144 | match (numbers_appE M4) |
| Cleaning: (b1) share and the all-points share | 0.9750 (117/120; prints 0.98 under half-up rounding); all-points 0.6234 (48/77), band 0.1818 | match (numbers_appE M3, minor) |
| Repetition (W24(e)) | StableLM-3B-4E1T: "1 trillion (1T) tokens for 4 epochs"; TinyLlama README: "Combined Dataset Size, Around 950B tokens" | see the TinyLlama note below the table |
| x_P = 0.3217 (W18) | (9 x 1,024 + 21 x 224)/(9.5 x 512 + 37.5 x 1,024) GPU-days; quote verified in `olmo3_report` | correct |

- **OLMo 2 13B epochs.** The 32B release post says the 13B trained "1.3 epochs"; the November post says 1.2. Fix list W8 uses 1.2.
- **OLMo 2 32B pretraining.** The report's mid-training section says the pretraining checkpoints were "trained on ... 7 trillion (32B)" tokens, against its own 6.06T and the post's 6T. Neither inconsistency changes a class, because the 13B and 32B are own-cap lower bounds under any of these counts.
- **TinyLlama.** The 950B figure describes the v1.0 run (3T tokens). The v1.1 corpus is not in the saved sources, so TinyLlama's own cap (Amendment 2) and its effective-data row rest on v1.0's dataset size. The median does not move.

## 6. Changes made by this review ([review])

| File | Change |
|---|---|
| `code/analysis/rb5_units/resolve.py` (new) | resolution of the six disagreements; writes and verifies `readings_final.csv` |
| `code/analysis/rb5_units/readings.py` | reads `readings_final.csv` first; a C2 "cap or choice" budget is a lower bound; accepts coder B's file (for the robustness row) |
| `code/analysis/rb5_units/agreement.py` | adds N margin, own cap (non-common units), R-NONE and deployment rationale |
| `code/analysis/rb5_units/run.py` | agreement numbers and the resolution in the paper numbers; new W2 rows; the rationale table note accepts the final file; RI p-values below 0.001 print "< 0.001" |
| `code/analysis/rb5_units/coderB.py`, `compare_AB.py` (new) | coder B's scripts copied from the session scratchpad (only path lines changed); they reproduce coder B's CSV and the disagreement list byte for byte |
| `code/analysis/rb2_decisions/decisions.py`, `run.py` | `point_if_choice` column; W2 rows "if 'cap or choice' is a choice" (35), "under coder B's readings" (31), "without MPT and Yi" (27) |
| `code/analysis/rb2_decisions/base.py` | a rebuild restores the cache's draw order (4 workers) for the chunked bootstraps, so `--rebuild-techs` reproduces `techs_cache.pkl` with any worker count |
| `code/analysis/rb2_decisions/exhibits_rb2.py` | Table 2 and readings-table notes: "cap or choice" means only the corpus or budget size is stated, or evidence of both is present (SmolLM's repetition explained) |
| `paper/notes/rb5_reading_protocol.md` | resolution record (Section 11) with the mapping correction disclosed |
| `output/memos/rb5_units.md` | status, W2, rationale and caveat passages updated |
| Outputs | all rb2_decisions and rb5_units outputs regenerated on `readings_final.csv`; `data/processed/rb5_units/units_primary.csv` and `output/tables/rb5_units_paper_numbers.csv` (190 rows) are final |

## 7. Numbers the paper text must take from the final files (all in `output/tables/rb5_units_paper_numbers.csv`)

- **W1(c), W5(b), W23(h).** The coders agreed on 15 of 17 readings (kappa 0.83); kappas for the other fields are in the table of Section 2. Disagreements were resolved by the protocol. Granite 3.0 and the Llama 3 herd are "cap or choice". Under the second coder's readings, 31 decisions are point estimates, median 0.737.
- **W2 and W6** are unchanged:
  - 29 point estimates and 20 lower bounds;
  - point-identified median 0.737 [0.672, 0.790], cluster [0.539, 0.824];
  - the headline 0.735 [0.657, 0.782], cluster [0.512, 0.806];
  - 0.980 of decisions over-trained.
- **W2 sensitivity.** 35 decisions (not 36) are point estimates if every "cap or choice" budget without repetition was chosen, median 0.735.
- **W19(d), W22 and `tab:app-rationale`.**
  - The share citing a compute-optimal target is 0.12 (2023), 0.04 (2024) and 0.00 (2025); compute-weighted 0.45, 0.65 and 0.00.
  - Three decisions cite one (the Llama 3 herd, Falcon 40B and Falcon 180B), with a median s of 0.04.
  - Among deployment-rationale decisions the median share is 0.48, 0.83 and 0.89 (n = 6, 18, 3), unchanged.
- **W14.** Model-level RI: coefficient p < 0.001, t-statistic p 0.06. Developer-level RI: 0.10 (0.94 without Alibaba). Bootstrap p 0.41; decisions 0.28 (p 0.67).
- **W19 trend, unchanged.**
  - Compute-weighted shares 0.24, 0.54 and 0.81.
  - Changes 0.30 [0.02, 0.65], 0.28 [-0.12, 0.44] and 0.57 [0.19, 0.72].
  - The median decision share falls from 2024 to 2025 (0.77 to 0.74). A sentence that says the value "rose" in both steps must name the compute-weighted share.

## 8. Remaining concerns (ranked)

1. **Table 2's sign columns** use WP4a's in-progress S1 file (0.69 and 0.61 of decisions; 0.14 and 0.11 of compute at tau 1 and 1.84). Redraw with `--exhibits-only` after WP4a's final S1.
2. **The synthetic-data flag moves the point-identified median most**: 0.675 [0.607, 0.731] without the protocol's flagged units (18 decisions), and 0.678 without the fix list's families (25). The protocol flags every 2025 decision. The text should report the point-identified median with and without flagged units, as W2 asks, and say that no 2025 share without flagged units exists under the protocol's rule.
3. **The weakest agreed choice codes** (MPT, Yi) are disclosed with their robustness row (27 decisions, 0.745).
4. **Coder A was not blind**, and the adjudicator saw the wedges. The blind coder's readings are reported as a robustness row. They change no median at printed precision.
5. **The rb2 file-name hazard.** `data/processed/rb2_decisions/units_primary.csv` still holds version 3's 56 family-label units, because `rb5_sign` reads it. The final primary unit file is `data/processed/rb5_units/units_primary.csv`.
