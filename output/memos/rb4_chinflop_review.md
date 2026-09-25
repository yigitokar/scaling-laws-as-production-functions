# Independent review — module rb4_chinflop (Chinchilla's FLOP accounting rebuilt)

Reviewer: Claude (independent replicator and skeptical referee). Date: 2026-09-24.
Scope: `code/analysis/rb4_chinflop/`, `output/tables/rb4_chinflop*`, `data/processed/rb4_chinflop/`, `output/memos/rb4_chinflop.md`, and the builder's recommended paper edits.
Rules followed: no MLX, at most 4 CPU processes, no commits, and no code edited outside rb4. The reviewer's scratch scripts live outside the project, in the session scratchpad. Changes to rb4 are marked `[review]` in code and memo.

---

## 1. Verdict

**Accept, with revisions.** The revisions to rb4's code, outputs and memo are made here. The paper revisions are listed in §7.

**What holds.**
- Every headline number reproduces byte for byte.
- Every key fact survives an independent re-derivation that uses none of the builder's code:
  - Table A4: 6 of 6 under T4 and 0 of 6 as printed.
  - The coordinate test: T4 is flat and every other count is rejected.
  - Matching and the missing 44M model.
  - The budget-level σ\*_b and the design-level 0.660 (0.023).
- T4 is the right primary count, for three reasons:
  - it is the count Hoffmann et al. implemented;
  - it is the count that set their budgets;
  - it treats Chinchilla as Llama 3 and Marin are treated, in each study's own budget count.

**What does not hold as written.**
1. **The level 0.660 is at the low end of what nearby, equally defensible choices give.** The builder did not examine this sensitivity.
   - The direction of the change is robust. Across ra1's 12 specifications, N_F lies 0.008–0.018 below T in every specification on the 137 runs (median −0.012).
   - The level is not robust. Drop the five high-loss runs, which lie outside every window but move the window centres, and the primary specification gives 0.681. Draw profile membership on the T4-corrected coordinates, and it gives 0.668.
   - As a result, the study-level heterogeneity the memo emphasizes (Q = 8.3, p = 0.041) falls to Q = 5.8–6.3 (p = 0.10–0.12) under these perturbations.
2. **The recommended Cochran sentence is factually wrong.** Meta's bracketed budgets stop at 10²¹, as Farseer's path does, so they do not "reach 10²²".
3. **The recommended paper-edit list covers about half of what changes.** It misses:
   - the introduction and conclusion;
   - Online Appendix Tables D (`appD_sigmaC.tex`, `appD_budgets.tex`) and F (`appF_sigmaC.tex`, `appF_labown.tex`);
   - Figure 3;
   - several numbers in §III (the configuration-count and Farseer means, the common-window slope, the top-budget value 0.60 → 0.59, and the symmetric-window drift range);
   - the lab-own numbers in §IV and Appendix F.
4. **Small slips.**
   - The 6T slope is +0.020 (0.004), not +0.021 (0.005).
   - k = 1/σ\* − 1 rises to 0.455, not 0.456.
   - The lab-own share moves by +0.009, not "+0.005 (not recomputed)".
   - "44M absent from Figure 4" should read "absent from the digitization".
   - The η-for-Meta rows of the 'old' run were labelled "Chinchilla rebuilt".

---

## 2. Reproduction

| Step | Result |
|---|---|
| Builder's code, from scratch, into an empty root (`RB4_OUTPUT_ROOT`), before any change | **23 of 23** table files and **31 of 31** override CSVs byte-identical to the committed outputs; 65 self-checks, 0 failed; 29 s |
| After the reviewer's changes: one end-to-end run into the project root, one into a fresh scratch root | `rb4_review.py --compare`: **56 of 56** identical (25 tables, 31 overrides); 81 checks, 0 failed; 40 s |
| Revised outputs against the builder's | Only three things change: two relabelled 'old'-run rows in `_study_level.csv`, rows added to `_review_checks.csv`, and two new files (`_specgrid.csv`, `_design_top.csv`). **No builder number changed.** |

---

## 3. Independent checks, without the builder's code

The reviewer's own scripts:
- parse the arXiv TeX (`data/raw/rb4_chinflop/src/main.tex`, lines 982–1034 and 1352–1413);
- re-implement the parameter and FLOP formulas;
- re-implement m1's budget assignment and membership from the raw Epoch CSV;
- re-implement the matching and the clustered regression.

ra1's `isoflop` is imported only for the estimator.

### (a) Table A9, Appendix F and Table A4

**Parameter counts.** T = L(4d·d_kv·h + 2df) + L·d·d_kv·h + Vd reproduces all 50 Table A9 counts:
- maximum error 0.80 percent, mean absolute error 0.78M;
- without W_R: 7.7 percent;
- with two vocabulary matrices: 36.4 percent (31.7 percent also without W_R).

The code docstring said "31"; it now says 36 [review].

**Table A4.**

| Count | Architecture T: ratios exact to 2 dp | Max error | Rounded-label N |
|---|---|---|---|
| **T4** (Appendix F without the embedding and logit terms) | **6 of 6** | **0.0047** | 4 of 6 |
| A (as printed) | 0 of 6 | 0.552 (73M: 1.58 against 1.03) | 0 of 6 |
| X (executed; no embedding term) | 0 of 6 | 0.274 | 0 of 6 |
| Non-embedding N in 6ND (any count) | 0 of 6 | — | — |

**Gopher.**
- Computed: T4 4.74×10²³, A 4.75×10²³, 6TD 5.03×10²³.
- Neither 6.3×10²³ nor 5.76×10²³ is reproduced, as the builder says.

### (b) Matching

- m1's rule, re-implemented, gives an offset of −0.0184 dex and **137** profile runs.
- Every run matches a Table A9 model:
  - maximum log error 0.0058, median 0.00028;
  - nearest-to-next ratio at most 0.076.
- 43 models are used anywhere and 35 on the profiles.
- Never used: 44M, 3,530M, 3,802M, 4,084M, 12,295M, 13,735M and 14,940M. The smallest digitized N is 57.3M. All confirmed.

### (c) Coordinate test

Within-budget slope of the digitized deviation plus log₁₀ r_F on log₁₀N; budget fixed effects; CR1 standard errors by budget; t with 8 degrees of freedom.

| Count that set the budgets | Slope (s.e.) | p | Without the 5 high-loss runs | Budgets with a positive slope |
|---|---|---|---|---|
| **T4** | **−0.0024 (0.0069)** | **0.735** | −0.0014 (0.0076) | 3 of 9 |
| A | −0.0651 (0.0121) | 0.0007 | −0.0654 | 0 of 9 |
| X | −0.0363 (0.0090) | 0.004 | −0.0361 | 1 of 9 |
| 6T | +0.0205 (0.0045) | 0.002 | +0.0206 | 8 of 9 |
| *Reviewer additions:* | | | | |
| 6P (non-embedding) | +0.065 (0.007) | < 0.001 | | |
| T4, context 1,024 (equivalently, causal half-S² terms) | +0.028 (0.006) | 0.001 | | |
| T4, context 4,096 | −0.049 (0.010) | 0.001 | | |

- All four builder rows reproduce to four decimals.
- The test also pins down the attention accounting: full S² terms at 2,048.
- With r = F/(6·N_dig) or F/(6·N_reported) instead of the architecture count, the slopes move by at most 0.0003.
- **One caveat on the logic.** The test identifies the ratio "plotted coordinate / budget count" as 6T/F_T4. The reading "plotted at 6ND, budgets in T4" is the natural one, but the plotting convention is an inference from Figure 4's construction, not an observation. The recommended paper text should say "consistent with" (§7).

### (d) Budget-level σ\*_b in N_F with ra1's estimator (imported)

**Point estimates.** The reviewer's own design data frame, passed to `isoflop.design_estimate(order 2, h = 1, path)`, matches the builder's nine σ\*_b to within 5×10⁻⁷ in four conventions: T, N_F T4, N_F A and P. For N_F T4:

| Budget | 6e18 | 1e19 | 3e19 | 6e19 | 1e20 | 3e20 | 6e20 | 1e21 | 3e21 |
|---|---|---|---|---|---|---|---|---|---|
| σ\*_b, N_F (T4) | 0.672 | 0.698 | **0.725** | 0.663 | 0.755 | 0.695 | 0.550 | 0.609 | 0.569 |
| σ\*_b, T (ra1) | 0.670 | 0.704 | 0.791 | 0.672 | 0.764 | 0.712 | 0.564 | 0.617 | 0.579 |
| window runs | 9 | 13 | **14** (T: 13) | 14 | 14 | 10 | 9 | 14 | 8 |

At 3×10¹⁹ the extra run is the 1,609M model.

**Bootstrap.** Run in ra1's row order (m1's order, sorted by loss), with ra1's seed and B = 999, it reproduces the builder exactly:
- RE **0.6596 (0.0228)**, FE 0.6655;
- drift −0.0613 (0.0249);
- Q 12.69 (bootstrap p 0.060).

In the raw file's row order the same estimator gives 0.6591 (0.0229), Q 12.96. That is Monte Carlo noise from the Rademacher-to-run mapping, not an error.

**η by budget.** The window OLS slopes are −0.003, −0.020, −0.036, −0.028, −0.026, −0.030, −0.021, −0.019 and −0.024, as reported.

### (e) Numbers quoted in the memo

Checked against the CSVs: H1–H4, the Table-1 row quantities (`_ranges.csv`) and the propagation (`_study_level`, `_metareg_*`, `_top_budget`, `_pi_sigmaC`, `_wedge_scenarios`). All are correct except the slips in §5.

---

## 4. Logic

### T4 as the primary

**Defensible, and consistent with the paper.**
- Section III and the m9 amendment define N_F = C/(6D) with C the actual training FLOPs.
- For Llama 3 and Marin the paper uses each study's own budget count: Meta's budgets, and Marin's 3 × forward count. That count is what makes each profile an exact isocost, which the model-free estimator needs.
- For Chinchilla that count is T4:
  - Table A4 is reproduced exactly;
  - the Figure 4 coordinates are flat only under T4;
  - under T4 the nominal budgets are exact isocosts.

**One nuance: the m9 amendment's "actual training FLOPs".**
- For the paper's own experiment this is the *executed* count (`code/sweep/gpt_mlx.flops_per_token`: non-embedding plus unembedding matmul plus attention), the analogue of count X.
- For Chinchilla, X gives 0.658 on the nominal budgets and 0.666 on its own isocosts, within 0.006 of T4.
- The paper should say "each study's own FLOP count" (the builder's text does) and cite X as the engineering-FLOP check.

### The window change at 3×10¹⁹

- **In the memo:** transparent. See H3, caveat 4, R9, the fixed-window CSV and the ≈ one-third share of the change.
- **In the builder's recommended paper text:** absent. §7 adds it.

### New finding: the level depends on window membership (review checks R11–R13, specification grid)

| Variant (ra1's estimator, seed and B = 999 unless noted) | N_F (T4) | T | N_F − T |
|---|---|---|---|
| Primary: 137 runs, h = 1, path windows | **0.660** (0.023) | 0.673 (0.027) | −0.014 |
| **R13**: five highest-loss runs dropped (132 runs; Besiroglu's sample) | **0.681** (0.020) | 0.674 (0.025) | +0.008 |
| **R12**: m1's membership rule applied to the T4-corrected coordinates (147 runs: 14 enter, 4 leave) | **0.668** (0.027) | 0.678 (0.028) | −0.010 |
| **R11**: count-free N_F = C_b/(6·D_dig), as for Llama 3 | **0.676** (0.021) | — | — |
| Current windows held fixed (builder R9; DL, approximate) | 0.668 | 0.678 | −0.010 |
| ra1's grid, 137 runs, 11 specifications with at least 8 budgets (B = 199) | 0.656–0.702 | 0.665–0.714 | −0.018 to −0.009 (median −0.012; 11 of 11 negative) |
| ra1's grid, 132 runs (B = 199) | 0.657–0.698 (excluding h = 0.6) | 0.671–0.714 | −0.024 to +0.004 (median −0.009; 8 of 11 negative) |

**R13: the five high-loss runs.**
- They sit on the 10¹⁹ profile at 2.0–6.8B, outside every window, but they enter ra1's preliminary global quadratics, which set the path centres.
- Dropping them moves the pooled path (in N_F its slope goes from 0.449 to 0.481). In T every window's membership stays the same. ra1's statement that they "do not affect the model-free estimate" is true in T.
- In N_F the same move changes two windows:
  - the 1.6B run leaves the 3×10¹⁹ window again, so σ\*_b goes from 0.725 back to 0.773;
  - one run enters at 6×10²⁰, so σ\*_b goes from 0.550 to 0.665.
- Table 1's note a ("fall outside every model-free window") is therefore true but no longer implies irrelevance.

**R12: membership.**
- m1 drew the ±0.045-dex profile band on uncorrected coordinates, after removing a common −0.018 offset.
- Under the builder's own T4 reading, runs of deep, narrow small models are plotted up to 0.06 dex left of their budget by construction (r_T4 up to 1.146). Some are therefore excluded as off-profile, and one run is included 4.7 s.d. from its T4 budget (217M at 6×10¹⁸).
- Applying the same rule after the T4 correction gives 147 runs. This is the internally consistent membership under the supported reading. The builder did not consider it.

**R11: the count-free version.**
- Under T4, C_b/(6D_dig) equals F_T4/6 multiplied by each run's coordinate scatter (s.d. 0.0375 in ln), which acts as noise in x.
- Simulation: adding N(0, 0.0375) to x_T4 raises the plain mean of σ\*_b by 0.007 on average (s.d. 0.007; 200 draws).
- The count-free plain mean (0.670, against 0.660) is at the 68th percentile of those draws. That is consistent with errors-in-x, so the architecture-based N_F is the better measurement. The count-free value is still the direct analogue of the Llama 3 treatment and belongs among the sensitivities.

**Propagation of these variants** (through rb4's hook).

| Chinchilla at | Study-level mean [HKSJ] | Q (p) |
|---|---|---|
| 0.668 (R12) | 0.691 [0.649, 0.734] | 6.2 (0.10) |
| 0.676 (R11) | 0.692 [0.653, 0.731] | 6.3 (0.10) |
| 0.681 (R13) | 0.694 [0.657, 0.730] | 5.8 (0.12) |

- All three means lie inside the builder's "Chinchilla's counts and conventions" range 0.682–0.697 and inside the variant range 0.663–0.700. The Table 1 ranges therefore stand.
- The "significant at 5 percent" heterogeneity does not stand.

**rb1's symmetric-window and power-frontier drift check, with Chinchilla in N_F (R14).**
- The T run reproduces `rb1_sigmaC_driftmc_symwin.csv` to 5×10⁻⁸.
- Chinchilla and Llama 3, design fixed effects:
  - primary windows −0.050;
  - symmetric −0.064;
  - power frontier −0.049;
  - both −0.064 (in T: −0.051, −0.054, −0.050, −0.053).
- The paper's "−0.050 to −0.054" becomes "−0.049 to −0.064".

**Recommendation.**
- Keep 0.660 as the primary. It uses the same pre-specified estimator, windows and 137-run sample as every other design and every upstream module.
- Report the accounting effect as "about −0.01 (−0.009 to −0.018 across specifications)".
- Report the level's sensitivity (0.656–0.702 across the grid; 0.668–0.681 under membership and sample perturbations).
- Stop presenting Q = 8.3 (p = 0.04) as a finding without that qualification.

---

## 5. Issues found and fixes

| # | Issue | Severity | Fix |
|---|---|---|---|
| 1 | Level fragility to window membership and sample (R11–R13, grid) not examined; Q = 8.3 (p = 0.041) presented as a finding | **Major** (interpretation) | Added R11–R14 and ra1's 12-specification grid in N_F to the review stage (`rb4_review.py`), writing `rb4_chinflop_specgrid.csv` and review-check rows. Memo H3 and H4: [review] bullets. §7 paper text hedged. |
| 2 | Recommended Cochran sentence says Meta's budgets "reach 10²²"; Meta's bracketed estimate stops at 10²¹, like Farseer's path | **Major** (factual) | Memo §2: [review] replacement sentence. §7 below. |
| 3 | Paper-edit list incomplete: introduction, conclusion, `appD_sigmaC.tex`, `appD_budgets.tex`, `appF_sigmaC.tex`, `appF_labown.tex`, Figure 3, technology.tex l.94–95, 115, 124, 131; appendix_additional l.66–67, 76–78, 85–86, 89; appendix_wedge l.142–145, 223, 228; wedge.tex l.265–267 | **Major** (completeness) | Complete list in §7. The BLUP design lines needed for `appD_sigmaC` Panel C were not in rb4's outputs; `rb4_prop.py` now keeps rb1's `tops` and `rb4_tables.py` writes `rb4_chinflop_design_top.csv` [review]. |
| 4 | rb2's lab-own columns not recomputed ("+0.005") | Moderate | Recomputed through a wrapper (§6): +0.009 (0.515 → 0.524); Llama 3 8B w from 8.27 to 8.75. Memo corrected. |
| 5 | 6T coordinate slope quoted as +0.021 (0.005); the CSV and table give 0.0205 (0.0045) → +0.020 (0.004) | Minor | Memo H2 table and Appendix B text [review]. |
| 6 | k = 1/σ\* − 1 "0.443 to 0.456"; it is 0.455 | Minor | Memo [review]. |
| 7 | 'old'-run study-level rows labelled "η for Meta only (Chinchilla rebuilt)" although Chinchilla is in T there | Minor (label) | `rb4_prop.py`: label depends on the run [review]. Only these two labels in `_study_level.csv` change. |
| 8 | "44M absent from Figure 4"; the digitization is what is tested | Minor | Memo [review]; §7 wording. |
| 9 | Docstring "two vocabulary matrices miss by up to 31 percent"; with W_R it is 36 (the memo already said 36) | Trivial | `rb4_arch.py` docstring [review]. |
| 10 | Log line "19 CSVs"; 22 are written | Trivial | `rb4_tables.py` [review]. |

**Not changed, but flagged for their owners.**
- **ra1's statement** "Chinchilla's 5 high-loss runs lie outside every window, so they do not affect the model-free estimate" (`ra1_modelfree.md` l.582). It holds in T but not in N_F, because the runs move the path centres. Once Chinchilla is in N_F, Table 1's note a needs the qualifier in §7.
- **Still in T** (not redone): ra1's Monte Carlo coverage and bias for Chinchilla, and rb1's drift-power Monte Carlo. The text can label them as computed in total parameters. The shift in x is small (η ≈ −0.02), so no material change is expected.

---

## 6. Downstream modules

### rb2_decisions: lab-own columns

**Method.**
- rb2 reads rb1's study-level PRIMARY row at run time, from `rb1_sigmaC_*.csv` only.
- The wrapper (scratchpad `rb2wrap/labown_wrap.py`) imports rb2's own `base`, `labown` and `decisions` unchanged and patches only `common_sigma` and the log functions. Nothing is written to the project.
- At the published row (0.6929, s.e. 0.0126, t₃) it reproduces rb2's outputs **exactly**:
  - lab-own medians: difference 6×10⁻¹⁷;
  - model w and intervals: 4×10⁻¹⁶;
  - headline medians and intervals: identical.
- At the rebuilt row (0.6873, s.e. 0.0148, t₃; k = 0.4551 against 0.4432):

| Statistic | Published | Rebuilt |
|---|---|---|
| One-rule median s, all lab-own models (22) | 0.515 | **0.524** |
| — without AI2 (13) | 0.632 | **0.642** |
| — Meta (10) | 0.569 | **0.579** |
| — AI2 (9) | 0.435 | 0.444 |
| Meta 10-budget law column (all / w/o AI2 / Meta / AI2) | 0.479 / 0.535 / 0.458 / 0.435 | 0.488 / 0.544 / 0.467 / 0.444 |
| OLMo κ = 1 path column | 0.441 / 0.632 / 0.569 / 0.314 | 0.450 / 0.642 / 0.579 / 0.321 |
| Lab curvature, earlier rule, reference columns | — | unchanged (they do not use the common σ\*) |
| Llama 3 8B one-rule w [95%] | 8.27 [4.51, 15.91] | **8.75 [4.58, 18.13]** |
| Llama 3.1 405B one-rule w | 1.46 | 1.47 |
| "Lab-own where available", median s: models (77) / decision units (56) | 0.737 / 0.726 | 0.738 / 0.726 |
| — 95% interval, decision units | [0.649, 0.776] | [0.649, 0.777] |
| Each lab-own model's s | — | +0.002 to +0.010 |

**Action.** rb2 must be re-run once rb1's study-level CSV carries the rebuilt row. rb2's glob does not read rb4's file. The paper numbers are listed in §7.

### rb3_econ2

**No recomputation needed.**
- `RB1_STUDY` and `RB1_PRED` are defined in `rb3common.py` but never read.
- The σ\* sensitivities are the constants 0.60, 0.70 and 0.74.
- The inputs are rb2's reference-technology wedges, which do not use the common σ\*. rb2's trend file is read only to check the reference rows.
- The label "σ\* = 0.60, the top-budget value" becomes a round value of 0.594. That is harmless.

### rb1_sigmaC

Its published CSVs are unchanged. If the paper adopts N_F for Chinchilla, rb1's own tables (Online Appendix D) must be replaced by rb4's rebuilt columns, listed below, or regenerated.

---

## 7. Paper edits (file:line → new text)

Line numbers refer to the files as of this review. "Unchanged" items are listed where a reader might expect a change.

### paper/tables/table1_designs.tex

| Line | Now | Replace with |
|---|---|---|
| 4–16 (source comments) | "Panel A model-free: ra1_…summary.csv"; "headline 0.6929 [0.6528, 0.7329]; conventions … 0.6770, 0.6709, 0.6688; eta = -0.1: 0.6648 [0.5781, 0.7514]; eta = +0.1: 0.7076; local first derivative 0.6732 [0.6074, 0.7391]" | "Chinchilla, Panel A: rb4_chinflop_summary.csv (NF_T4), rb4_chinflop_param.csv (N_F (T4)), rb4_chinflop_ranges.csv. Panel C: rb4_chinflop_study_level.csv (run = new): headline 0.6873 [0.6402, 0.7343]; conventions 0.6735, 0.6703, 0.6683; eta = -0.1 (Meta) 0.6729 [0.5980, 0.7478]; eta = +0.1 (Meta) 0.6997; Chinchilla counts 0.6816-0.6972; local first derivative 0.6698 [0.6027, 0.7369]; rows 5-10: rb4_chinflop_metareg_slopes.csv / _predictions.csv / _top_budget.csv (run = new)" |
| 31–32 | `Chinchilla, profiles$^{a}$ & 137 & 9/9 & 57M--16B & 0.04--294 & 1.67 & T & 0.673 & 0.667 & 0.692 \\` / `(0.027) & (0.023) & (0.013)` | `Chinchilla, profiles$^{a}$ & 137 & 9/9 & 57M--16B & 0.04--294 & 1.65 & F & 0.660 & 0.656 & 0.690 \\` / `(0.023) & (0.028) & (0.015)` |
| 62 | `0.693 & [0.653, 0.733] & $k=4$, $\tau=0.017$, $Q=5.7$ ($p=0.13$)` | `0.687 & [0.640, 0.734] & $k=4$, $\tau=0.023$, $Q=8.3$ ($p=0.04$)` |
| 63 | `Other parameter conventions & 0.669--0.677` | `… & 0.668--0.673 & & Marin, Farseer` |
| 64 | `FLOP accounting, $\eta=\pm0.1$ & 0.665--0.708 & [0.578, 0.751]$^{f}$ & Chinchilla, Meta` | Two rows: `FLOP accounting, $\eta=\pm0.1$ & 0.673--0.700 & [0.598, 0.748]$^{f}$ & Meta \\` and `\quad Chinchilla's FLOP counts, conventions and windows & 0.682--0.697 & & Chinchilla \\` |
| 65 | `First-derivative estimator & 0.673 & [0.607, 0.739]` | `… & 0.670 & [0.603, 0.737] & Marin, Meta` |
| 66 | `$-$0.032 & [$-$0.075, 0.010] & 44 budgets, wild $p=0.042$` | `$-$0.033 & [$-$0.076, 0.011] & 44 budgets, wild $p=0.039$` |
| 67 | `$-$0.007 & [$-$0.041, 0.027] & 37 budgets, wild $p=0.85$` | `$-$0.008 & [$-$0.042, 0.027] & 37 budgets, wild $p=0.83$` |
| 68 | `$-$0.058 & (0.013)` | unchanged |
| 69 | `0.650 & [0.565, 0.736]` | `0.647 & [0.560, 0.735]` |
| 70 | `0.596 & [0.567, 0.624] & Chinchilla, Llama~3; $Q=1.9$` | `0.594 & [0.565, 0.622] & Chinchilla, Llama~3; $Q=2.5$` |
| 76 (note, Conv.) | "T, total; F, FLOP-implied $C/(6D)$" | "T, total; F, FLOP-implied $C/(6D)$, with $C$ the study's own FLOP count (Chinchilla: FLOPs per token over six, rebuilt from Hoffmann et al.'s architecture table; Online Appendix Table~\ref{tab:app-chinflop})" |
| 81 (note a) | "Five high-loss runs fall outside every model-free window; without them, 132 runs, $M$ from 0.46, spread 1.49." | "Five high-loss runs fall outside every model-free window but enter the window centres; without them, 132 runs, $M$ from 0.44, spread 1.48, and model-free $\sigma^*=0.681$. Panel B's Chinchilla row stays in total parameters; in $N_F$ it is 0.700 ($\kappa$ free) and 0.740 ($\kappa=1$)." |
| 87 (note e) | "the mean lies in 0.663--0.708" | "the mean lies in 0.663--0.700" |
| 88 (note f) | "For $\eta=-0.1$, the elasticity of FLOP per token with respect to the measured parameter count." | "For $\eta=-0.1$ applied to Meta, the elasticity of FLOP per token with respect to the measured parameter count; Chinchilla's accounting is rebuilt." |
| 91 (note g) | "$p$ lies in 0.03--0.12" | "$p$ lies in 0.03--0.13" |

### paper/sections/technology.tex

| Line | Now | Replace with |
|---|---|---|
| 41–43 | "$N_F=C/(6D)$ for Llama~3 and Marin, the total count for Chinchilla, whose token counts are themselves constructed as $C/(6N)$, and the non-embedding count for Farseer." | "$N_F=C/(6D)$ for Chinchilla, Llama~3 and Marin, with $C$ each study's own FLOP count, and the non-embedding count for Farseer. For Chinchilla we rebuild FLOPs per token from Hoffmann et al.'s architecture table: their Appendix F count without its two vocabulary terms reproduces their Table A4 exactly, and the digitized FLOP coordinates of their Figure~4 are consistent with budgets set in that count and with no other we test (Online Appendix Table~\ref{tab:app-chinflop})." |
| 84 | "On Chinchilla's profiles $\sigma^*=0.673$ (standard error 0.027)" | "On Chinchilla's profiles, in FLOP-effective parameters, $\sigma^*=0.660$ (standard error 0.023)" |
| 87–89 | "the random-effects mean is 0.693, with a 95 percent confidence interval of [0.653, 0.733] (…). Cochran's statistic is 5.7 on three degrees of freedom ($p=0.13$), but the test has little power, and the design means average over different compute windows." | "the random-effects mean is 0.687, with a 95 percent confidence interval of [0.640, 0.734] (…). Chinchilla and Meta both give 0.66, and Marin and Farseer 0.71. Cochran's statistic is 8.3 on three degrees of freedom ($p=0.04$), but 5.8 to 6.3 ($p$ between 0.10 and 0.12) when Chinchilla's windows or profile membership are drawn differently, and the design means average over different compute windows: $\sigma^*$ falls above $3\times10^{20}$ FLOP in Chinchilla and Llama~3, and Marin stops there." |
| 92–93 | "…shifts by about $0.44\eta$; Chinchilla's and Meta's accounting is unobserved, and $\eta=\pm0.1$ moves the mean to 0.665 or 0.708." | "…shifts by about $0.44\eta$. Chinchilla's accounting is observed once its FLOPs are rebuilt from the architecture table, and its elasticity is small ($\eta$ between $-0.04$ and 0.00 across budgets); Hoffmann et al.'s other counts, the total and non-embedding conventions, and alternative windows and profile memberships move the mean within 0.682--0.697. Meta's accounting is unobserved, and $\eta=\pm0.1$ for Meta moves the mean to 0.673 or 0.700." |
| 94–95 | "…and the mean to 0.677, and counting Farseer's embeddings lowers the mean to 0.671." | "…and the mean to 0.673, and counting Farseer's embeddings lowers the mean to 0.670." |
| 96 | "the mean lies between 0.663 and 0.708" | "the mean lies between 0.663 and 0.700" |
| 109–111 | "a slope of $-0.032$ per decade … standard error of 0.014 … between 0.03 and 0.12" | "a slope of $-0.033$ per decade … standard error of 0.014 … between 0.03 and 0.13" |
| 112 | "$\sigma^*=0.715$ at $10^{19}$ FLOP, 0.683 at $10^{20}$ and 0.650 at $10^{21}$" | "$\sigma^*=0.713$ at $10^{19}$ FLOP, 0.680 at $10^{20}$ and 0.647 at $10^{21}$" |
| 115 | "the pooled slope is $-0.007$" | "the pooled slope is $-0.008$" |
| 116 | "falls by 0.058 per decade (standard error 0.013)" | unchanged |
| 117–118 | "($Q=1.9$, $p=0.75$) and give $\sigma^*=0.596$, with a 95 percent confidence interval of [0.567, 0.624]" | "($Q=2.5$, $p=0.65$) and give $\sigma^*=0.594$, with a 95 percent confidence interval of [0.565, 0.622]" |
| 124 | "leave the Chinchilla--Llama~3 drift between $-0.050$ and $-0.054$ per decade" | "leave the Chinchilla--Llama~3 drift between $-0.049$ and $-0.064$ per decade" (R14) |
| 131 | "and the top-budget value, 0.60" | "and the top-budget value, 0.59" |
| 132 | "0.48 at $10^{23}$ FLOP and 0.42 at $10^{24}$" | unchanged (0.479, 0.421) |
| 162 | "whose $\sigma^*$ of 0.701 matches the study-level mean" | unchanged; 0.701 is inside [0.640, 0.734]. Optionally "is close to". |
| 16–17 (optional) | "It falls to about 0.60 at the largest budgets" | may stay ("about"); 0.594 |

**Figure 3** (`fig3_merged`, generated by `code/paper/make_fig3_merged.py`). It must be regenerated with Chinchilla rebuilt. The script reads ra1's and rb1's CSVs, so either point it at rb4's outputs or re-run the upstream modules.

| Panel | New inputs |
|---|---|
| a | Chinchilla circle 0.660 (0.023), diamond 0.656 (0.028), square 0.690 (0.015); study-level mean 0.687 [0.640, 0.734]; gray band 0.663–0.700 |
| b | Chinchilla's nine N_F σ\*_b (`rb4_chinflop_budgets.csv`, NF_T4) and the pooled line and band (`_metareg_predictions.csv`, run = new) |
| c | `_pi_sigmaC.csv`, run = new |

### paper/sections/introduction.tex

| Line | Now | Replace with |
|---|---|---|
| 51–52 | "$\sigma^*$ is 0.69, with a 95 percent confidence interval of $[0.65, 0.73]$; … move the mean between 0.66 and 0.71." | "$\sigma^*$ is 0.69, with a 95 percent confidence interval of $[0.64, 0.73]$; … move the mean between 0.66 and 0.70." |
| 53 | "$\sigma^*$ falls to about 0.60" | optional: "to about 0.59" |
| 70 | "it is 0.51 for those models, against 0.65 under the reference" | "it is 0.52 for those models, against 0.65 under the reference" |
| 71 | "at 0.60 the median share is 0.87" | unchanged (constant σ\* = 0.60 scenario) |

### paper/sections/conclusion.tex

| Line | Now | Replace with |
|---|---|---|
| 33 | "(95 percent confidence interval 0.65 to 0.73)" | "(95 percent confidence interval 0.64 to 0.73)" |
| 34 | "move it between 0.66 and 0.71. At $6\times10^{20}$ to $3\times10^{21}$ FLOP, 0.60." | "move it between 0.66 and 0.70. At $6\times10^{20}$ to $3\times10^{21}$ FLOP, 0.59." |
| 35 | "it lies between 0.42 and 0.60 at $10^{24}$ FLOP" | "it lies between 0.42 and 0.59 at $10^{24}$ FLOP" |

### paper/sections/appendix_data.tex (§ Chinchilla Extraction)

**Lines 62–65.** Now: "…and if the figure's budgets count FLOPs differently from $6ND$ (…), the constructed $D$ inherits an error correlated with $N$. Online Appendix~\ref{app:add-modelfree} bounds its effect on $\sigma^*$; we have not rebuilt $D$ from Hoffmann et al.'s architecture table."

Replace with:

> "…one for one into $\ln D$. We rebuild FLOPs per token for all 50 models of Hoffmann et al.'s Table A9. Their parameter counts are reproduced, to 0.8 percent, by attention, MLP and one relative-position projection per layer plus one $32{,}000\times d$ vocabulary matrix. Their Appendix F formula as printed does not reproduce their Table A4; without its embedding and final-logit terms it reproduces all six ratios. In that count, FLOPs per token are 0.98--1.15 times $6N$ for the models in the profiles. The within-budget pattern of the digitized FLOP coordinates is flat once the budgets are read in that count (slope $-0.002$, s.e.\ 0.007) and not in the printed count ($-0.065$), in executed FLOPs ($-0.036$) or in $6ND$ ($+0.020$). This is what one expects if Figure~4 plots runs at $6ND$ with token counts set in Hoffmann et al.'s count. Then the digitization's $D=C/(6N)$ is the true token count, each profile is an exact isocost in that count, and $N_F=C/(6D)$ is FLOPs per token over six, whatever $D$. The 44M model of Table A9 does not appear in the digitization. The digitized offset of $\log_{10}C$ below the nominal budgets ($-0.018$; below) is mostly this accounting: net of it, $-0.006$ remains. Drawn on the corrected coordinates, the membership rule below gives 147 profile runs instead of 137, and $\sigma^*=0.668$ (Online Appendix~\ref{app:add-modelfree})."

### paper/sections/appendix_additional.tex (§ The Model-Free Estimator and the Elasticity by Compute)

| Line | Now | Replace with |
|---|---|---|
| 66 | "ranges over 0.665--0.714 for Chinchilla" | "ranges over 0.656--0.702 for Chinchilla (0.665--0.714 in total parameters)" |
| 67 | "an independent re-implementation gives 0.669--0.694" | "an independent re-implementation gives 0.669--0.694 (Chinchilla in total parameters)" |
| 67–69 | Monte Carlo sentence | add "(Chinchilla's in total parameters)", or re-run |
| 69–72 | "The estimator needs exact isocosts, so each design is analyzed in the convention in which its budgets equal $6ND$. On Marin, … if Chinchilla's FLOPs per token varied with $N$ with elasticity $\eta$, $\sigma^*$ would move by about $0.44\eta$, at most 0.04 for $|\eta|\le0.1$." | see the replacement text below |
| 76 | "Chinchilla gives 0.684 (fixed) against 0.673 (random)" | "Chinchilla gives 0.666 (fixed) against 0.660 (random)" |
| 77–78 | "Cochran's $Q$ rejects homogeneity within Chinchilla ($p=0.025$) and Llama~3 ($p=0.001$), but not within Marin's corpora." | "Cochran's $Q$ rejects homogeneity within Llama~3 ($p=0.001$), but not within Chinchilla ($Q=12.7$, bootstrap $p=0.06$) or Marin's corpora." |
| 82–83 | "The slope is $-0.032$ per decade; … between 0.03 and 0.12" | "The slope is $-0.033$ per decade; … between 0.03 and 0.13" |
| 85 | "the pooled slope is $-0.007$ ($p=0.85$)" | "the pooled slope is $-0.008$ ($p=0.83$)" |
| 86 | "an inverse-variance mean of 0.596" | "an inverse-variance mean of 0.594" |
| 89–90 | "leave the Chinchilla and Llama~3 drift at $-0.050$ to $-0.054$" | "leave the Chinchilla and Llama~3 drift at $-0.049$ to $-0.064$" |
| 91–92 | "a mean of 0.693 … $[0.653,0.733]$; … over 0.663--0.708" | "a mean of 0.687 … $[0.640,0.734]$; … over 0.663--0.700" |

Replacement text for lines 69–72:

> "The estimator needs exact isocosts, so each design is analyzed in FLOP-effective parameters, the convention in which its profiles are exact isocosts. On Marin, whose accounting is observed, the configuration count gives 0.04 to 0.07 less. For Chinchilla, FLOPs per token rebuilt from the architecture table have an elasticity with respect to $N$ between $-0.04$ and 0.00 by budget, and $\sigma^*$ is 0.660 against 0.673 in total parameters. It is 0.648--0.666 across Hoffmann et al.'s printed, implemented and executed counts, and 0.690 in non-embedding parameters; moved onto each count's own exact isocosts, every count and convention gives 0.660--0.669 (Table~\ref{tab:app-chinflop}). In every specification of the grid the rebuilt count gives 0.008 to 0.018 less than the total count. At the primary specification about a third of the difference comes from one run that enters the $3\times10^{19}$ window. The five high-loss runs lie outside every window but move the window centres: without them the primary specification gives 0.681 (0.674 in total parameters). With profile membership drawn on the corrected coordinates it gives 0.668, and with $N_F$ read directly off the digitized coordinates 0.676."

### paper/tables/appD_sigmaC.tex (Online Appendix Table D; a copy of rb1's table, so replace with the rebuilt values)

**Panel A.**

| Line | Row | New values |
|---|---|---|
| 14 | Six designs, 3-level RE | `44 & $-$0.033 & 0.010 & 0.014 [3.1] & 0.039` |
| 15 | RE, unweighted | `44 & $-$0.030 & 0.008 & 0.011 [4.5] & 0.046` |
| 16 | Balanced weights | `44 & $-$0.034 & -- & 0.011 [3.3] & 0.032` |
| 17 | IsoFLOP designs, FE | `36 & $-$0.050 & 0.009 & -- & 0.226` |
| 18 | Chinchilla and Llama 3, FE | unchanged (`$-$0.058 & 0.013`) |
| 19 | Marin | unchanged |
| 20 | Budgets ≤ 3×10²⁰ | `37 & $-$0.008 & 0.014 & 0.012 [3.7] & 0.827` |
| 21 | Porian | unchanged |

**Panel B.**

| Lines | Row | New values |
|---|---|---|
| 26–27 | 3-level RE | `0.713 & 0.680 & 0.647 & 0.615`; intervals `[0.707, 0.719] & [0.645, 0.716] & [0.560, 0.735]` |
| 28–29 | RE, unweighted | `0.718 & 0.688 & 0.659 & 0.629`; intervals `[0.713, 0.723] & [0.657, 0.720] & [0.596, 0.722]` |

**Panel C** (source: `rb4_chinflop_design_top.csv`, run = new).

| Line | Row | New values |
|---|---|---|
| 34 | Chinchilla | `$3\times10^{21}$ & 0.569 & (0.054) & 0.624` |
| 35 | Llama 3 | `0.609 & (0.017) & 0.636` |
| 36 | Marin, Comma | `0.686 & (0.089) & 0.664` |
| 37 | Marin, DCLM | `0.737 & (0.060) & 0.668` |
| 38 | Marin, Nemotron-CC | `0.733 & (0.052) & 0.669` |
| 39 | Farseer | `0.658 & (0.011) & 0.659` |
| 40 | Chinchilla and Llama 3, top budgets | `5 & 0.594 & [0.565, 0.622]` |

**Note, line 45.** "CR2 $t$-test $p$-values of the first two rows: 0.096 and 0.043; on the scale $S$ … CR2 $p$ 0.122 and wild $p$ 0.065" becomes "… 0.097 and 0.050; … CR2 $p$ 0.126 and wild $p$ 0.063".

**Panel D** (source: `rb4_chinflop_study_level.csv`, run = new).

| Line | Row | New values |
|---|---|---|
| 63 | Headline | `4 & 0.687 & [0.640, 0.734] & 0.023 & 8.3 (0.041)` |
| 64 | Marin's corpora independent | `0.691 & [0.649, 0.732] & 0.019 & 8.3 (0.041)` |
| 65 | IsoFLOP studies only | `3 & 0.673 & [0.609, 0.737] & 0.010 & 2.3 (0.311)` |
| 66 | Six designs, Marin separate | `6 & 0.691 & [0.668, 0.715] & 0.008 & 5.7 (0.336)` |
| 67 | Marin in configuration N | `0.673 & [0.615, 0.732] & 0.032 & 14.3 (0.003)` |
| 68 | η = −0.1 | relabel "$\eta=-0.1$ (Meta)": `0.673 & [0.598, 0.748] & 0.043 & 21.1 ($<$0.001)` |
| 69 | η = +0.1 | relabel "$\eta=+0.1$ (Meta)": `0.700 & [0.669, 0.730] & 0.011 & 4.3 (0.230)` |
| 70 | Farseer, Hessian path, CV h | `0.665 & [0.650, 0.679] & 0.000 & 2.7 (0.439)` |
| 71 | Farseer, total N | `0.670 & [0.656, 0.685] & 0.000 & 2.4 (0.500)` |
| 72 | Marin config. N, Farseer total N | `0.668 & [0.654, 0.683] & 0.000 & 1.1 (0.783)` |
| 73 | Fixed-effect means | `0.676 & [0.600, 0.752] & 0.045 & 46.5 ($<$0.001)` |
| 74 | Local first-derivative | `0.670 & [0.603, 0.737] & 0.037 & 16.8 ($<$0.001)` |
| new | Chinchilla's other counts and conventions (9 variants) | `4 & 0.682--0.697 & & &` |
| new | Chinchilla windows and membership (R11–R13) | `4 & 0.691--0.694 & & & 5.8--6.3 (0.10--0.12)` |

**Note b, line 79.** "Chinchilla 0.673 (0.027), total $N$" becomes "Chinchilla 0.660 (0.023), FLOP-effective $N_F=F/6$ with Hoffmann et al.'s FLOPs per token rebuilt from their architecture table (Table~\ref{tab:app-chinflop})". In the same note, "the mean ranges over 0.663--0.708" becomes "0.663--0.700", and $\eta$ now applies to Meta only.

### paper/tables/appD_budgets.tex (Chinchilla block, lines 12–22; source `rb4_chinflop_budgets.csv`, variant NF_T4)

| Budget | σ\*_b (s.e.) | Runs left/right | M\*_b |
|---|---|---|---|
| 6×10¹⁸ | 0.672 (0.064) | 3/6 | 26.1 |
| 10¹⁹ | 0.698 (0.060) | 4/9 | 21.1 |
| 3×10¹⁹ | 0.725 (0.045) | 3/11 | 15.7 |
| 6×10¹⁹ | 0.663 (0.043) | 7/7 | 14.9 |
| 10²⁰ | 0.755 (0.060) | 7/7 | 13.9 |
| 3×10²⁰ | 0.695 (0.062) | 5/5 | 14.9 |
| 6×10²⁰ | 0.550 (0.047) | 5/4 | 21.7 |
| 10²¹ | 0.609 (0.039) | 10/4 | 21.3 |
| 3×10²¹ | 0.569 (0.054) | 3/5 | 28.3 |
| Pooled, FE / RE | 0.666 / 0.660 (0.018 / 0.023) | Q 12.7 | p 0.060 |

**Note, line 38.** "Parameters: total $N$ (Chinchilla), FLOP-implied $N_F=C/(6D)$ (Llama~3, Marin)" becomes "Parameters: FLOP-implied $N_F=C/(6D)$ (Chinchilla, with FLOPs per token rebuilt from Hoffmann et al.'s architecture table; Llama~3; Marin); $M^*_b$ in tokens per FLOP-effective parameter".

### paper/tables/appF_sigmaC.tex

**Panel A** (source: `rb4_chinflop_pi_sigmaC.csv`, run = new).

| Line | Row | New values |
|---|---|---|
| 14 | Upper bound | 0.594 in all six columns |
| 15 | Lower bound, Chinchilla–Llama 3 drift (−0.058) | `0.594 & 0.536 & 0.479 & 0.421 & 0.363 & 0.306` |
| 16 | Lower bound, pooled drift | relabel "($-$0.033)": `0.594 & 0.561 & 0.528 & 0.495 & 0.463 & 0.430` |
| 17 | θ at the upper bound | 0.684 in all six columns |
| 18 | θ at the lower bound | `0.684 & 0.865 & 1.089 & 1.375 & 1.751 & 2.269` |

**Panel B** (source: `rb4_chinflop_wedge_scenarios.csv`, run = new).

| Line | Row | New values |
|---|---|---|
| 29 | Top budgets, 95% upper | relabel "($\sigma^*=0.622$)": `7.14 & 0.860 & 0.849 & 0.380 & 7.1 & 31.6` |
| 32 | σ\*(C_i), pooled line | `11.07 & 0.910 & 0.890 & 0.525 & 11.1 & 61.1` |
| 33 | σ\*(C_i), lower bound (Chin.–Llama) | `45.50 & 0.978 & 0.963 & 0.750 & 41.8 & 526.0` |
| 34 | σ\*(C_i), lower bound (pooled) | `19.80 & 0.949 & 0.935 & 0.594 & 19.8 & 165.6` |
| 30 | Top budgets (σ\* = 0.60) | constant-0.60 row, unchanged; optionally relabel "$\sigma^*=0.60$ (top budgets, rounded)" |

**Note.** "$\sigma_{\rm top}=0.596$" becomes "0.594".

### paper/tables/appF_labown.tex (source: §6 wrapper; rb2 to be re-run)

| Line | Row | New values |
|---|---|---|
| 13 | all lab-own models | `22 & 0.65 & 0.52 & 0.67 & 0.49 & 0.45 & 0.61` |
| 14 | without AI2 | `13 & 0.56 & 0.64 & 0.69 & 0.54 & 0.64 & 0.59` |
| 15 | Meta | `10 & 0.49 & 0.58 & 0.62 & 0.47 & 0.58 & 0.51` |
| 16 | AI2 | `9 & 0.70 & 0.44 & 0.66 & 0.44 & 0.32 & 0.66` |
| 21 | Note | "common model-free $\sigma^*=0.693$" becomes "$\sigma^*=0.687$" |

### paper/sections/appendix_wedge.tex

| Line | Now | Replace with |
|---|---|---|
| 139 | "$\sigma^*=0.693$" | "$\sigma^*=0.687$" |
| 142–143 | "the median share is 0.51 under the one rule against 0.65 under the reference, and 0.63 against 0.56 without AI2's models" | "… 0.52 … 0.65 …, and 0.64 against 0.56 without AI2's models" |
| 144 | "falls from 0.66 to 0.44" | unchanged |
| 145 | "Llama~3 8B has $w=8.27$, with a 95 percent interval of $[4.51,15.91]$, and Llama~3.1 405B has 1.46" | "$w=8.75$, … $[4.58,18.13]$, … 405B has 1.47" |
| 223 | "largest budgets, 0.596" | "largest budgets, 0.594" |
| 223–224 | "(0.48 at $10^{23}$ FLOP and 0.42 at $10^{24}$)" | unchanged |
| 227–228 | "at $\sigma^*=0.60$ the median share is 0.88 over models (0.87 over decision units), against 0.75 (0.74) at the reference" | unchanged |
| 228 | "it is 0.86 at 0.624, the upper end" | "it is 0.86 at 0.622, the upper end" |

### paper/sections/wedge.tex

| Line | Now | Replace with |
|---|---|---|
| 263 | "$\sigma^*=0.693$" | "$\sigma^*=0.687$" |
| 265 | "the median share is 0.51, against 0.65 under the reference" | "the median share is 0.52, against 0.65 under the reference" |
| 266–267 | "without AI2 the medians are 0.63 and 0.56" | "without AI2 the medians are 0.64 and 0.56" |
| 268–269 | "Over all 56 decisions … is 0.73" | unchanged (0.726) |
| 200–202 | "at 0.62, the upper end …, the median share is 0.85" | unchanged (0.622 → 0.849) |

### Online Appendix

- Add `\input{…/rb4_chinflop.tex}` (label `tab:app-chinflop`) next to Table D.
- Optionally add one line in its notes citing `rb4_chinflop_specgrid.csv` for the grid range 0.656–0.702.

### Unchanged (checked)

- Abstract ("about 0.7 … declining toward 0.6").
- Table 3 and Section V (rb3; constant σ\* 0.60/0.70/0.74).
- Llama 3 numbers (0.660; κ = 1 0.769; κ free 0.693).
- The reference technology (0.701) and all reference-technology wedges and shares.
- "0.73 when … developers' own technologies replace the reference" (wedge.tex l.98–99).
- The hinge decline "0.15 per decade".
- The "loses significance without either design or each design's largest budget" claims (leave-one-out wild p 0.14–0.29).

---

## 8. Files touched by the review

**Code** (all changes marked `[review]`):
- `code/analysis/rb4_chinflop/rb4_review.py`: R11–R14, the specification grid, the ra1-grid and rb1-symwin reproduction checks.
- `rb4_prop.py`: the 'old'-run label; keeps rb1's design-top table.
- `rb4_tables.py`: writes `rb4_chinflop_design_top.csv`; log count.
- `rb4_arch.py`: docstring.

**Outputs** (regenerated by one end-to-end run):
- new: `output/tables/rb4_chinflop_specgrid.csv`, `rb4_chinflop_design_top.csv`;
- extended: `rb4_chinflop_review_checks.csv` (81 rows, 0 failed);
- relabelled: `rb4_chinflop_study_level.csv` (two labels).

**Memo.** `output/memos/rb4_chinflop.md`, with `[review]` bullets in H1–H4, §2, the inventory and the self-review.

**Reviewer scratch scripts** (not in the project): independent A9/A4, matching, coordinate, σ\*_b and bootstrap re-derivations; the membership, count-free and five-run sensitivities; the rb2 lab-own wrapper; the rb1 hook for design lines and study-level alternatives.
