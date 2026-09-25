# rb5_sign: review

Review of module `code/analysis/rb5_sign/`, its outputs and memo `output/memos/rb5_sign.md` (round 3, WP4a; fix list items S1 to S5, decision D-3). Reviewer: package WP4a-review, 25 September 2026, in a session separate from the builder's. Changes made by the review are marked "[review]" in the code, the memo and this file.

**Verdict: accept, with seven corrections (all made) and four points for the author.** The construction follows D-3 and S1 to S5. A from-scratch re-run with WP4b's final unit file reproduces every output byte for byte. A standalone re-implementation that imports no project module agrees on 144 of 144 quantities: every share exactly; critical values, the ends of the set and the breakdown points within Monte Carlo error. The headline numbers stand: w > 1 is identified for 78 percent of models, 69 percent of the 49 decisions and 24 percent of training compute. The substantive new finding is that the minima paths are not straight inside the designs (Section 5, point 2). It does not move the headline outside the range of the S2 variants, but the text should not call Chinchilla's and Llama 3's paths flat beyond the designs.

---------------------------------------------------------------------------------------------------------

## 1. Re-run from scratch and determinism

- Inputs at the time of the review: rb2's `clean_models.csv` of 14:15 (WP4b's final re-run; MPT-30B at 1.05T tokens, as the builder already had); WP4b's final unit file `data/processed/rb5_units/units_primary.csv` (49 units, codes `readings_final.csv`); rb2's `units_primary.csv` for the 56 family-label units (unchanged since version 3); `techs_cache.pkl` rebuilt by WP4b at 14:02; rb1's sigma\* files of 00:02 (WP2 has not changed them).
- The unit file is the same partition of the 77 models as rb2's `units_subgroup.csv`, which the builder used (checked by the reviewer and now by a module check).
- Two runs from scratch (`__pycache__` removed; `--units data/processed/rb5_units/units_primary.csv`), from 14:26: all 24 output files are byte-identical to the builder's 12:17 run, apart from the run log and the PDF's creation time. The PNG is identical.
- After the review's changes (Section 6) the module was run again. Only the intended files changed: `rb5_sign_shares.csv` and `model_bounds_all_sets.csv` (two comparison-only sets added), `rb5_sign_magnitude.csv` (a `level` column and decision-level rows), `rb5_sign_paper_numbers.csv` (217 rows, was 205), `rb5_sign_table.tex` (note), `rb5_sign_review_checks.csv` (27 checks, all pass); new: `rb5_sign_review_curvature.csv`, `rb5_sign_review_independent.csv`. Unchanged: `model_bounds_S1.csv`, `units_sign.csv`, the figure, the paths, the sets, the breakdown and frontier files, the fixed-zero-point file.
- So Table 2's sign columns, which WP4b's generator reads from `model_bounds_S1.csv`, are final as drawn from the 12:17 file.

## 2. Independent re-implementation (`code/analysis/rb5_sign/review_independent.py`)

The script reads only raw inputs: ra1's per-budget minima, the clean sample, the two unit files, Marin's raw IsoFLOP runs, rb4's Chinchilla minima in N_F, rb1's sigma\* files and version 3's PI-1 anchors (git commit 69c9ae8, the round-3 base). It takes DeepSeek's law from its published constants. It fits the paths by QR least squares, draws its own wild bootstrap (Webb weights, HC2-rescaled residuals, pooled studentization, other seeds), and takes the supremum on a grid of 2,001 points where the module computes it exactly. It writes `output/tables/rb5_sign_review_independent.csv` (144 rows; about 25 seconds; not called by `run.py`).

| Quantity | Module | Independent |
|---|---|---|
| Slopes: Chinchilla; Llama 3; Marin (common) | -0.0033; 0.0083; 0.1985 | identical to 1e-9 |
| Levels at c_J: Chinchilla; Llama 3; DCLM, Nemotron-CC, Comma | 20.66; 15.06; 15.03, 15.52, 20.80 | identical |
| Normal-theory sup-t q: Chinchilla; Llama 3; Marin | 2.726; 2.764; 2.767 | 2.722; 2.765; 2.765 |
| Bootstrap sup-t q (seeds differ) | 2.726; 2.871; 2.698 | 2.762; 2.869; 2.749 |
| Share of models, tau = 1, 1.3, 1.84, 3.4, 4.4, 8, 10 | 0.78, 0.74, 0.66, 0.53, 0.47, 0.38, 0.32 | identical at every tau |
| Compute-weighted; 49 decisions; 56 decisions (tau = 1) | 0.237; 0.694; 0.714 | identical (also at every tau, cw included) |
| By year, compute-weighted (2023, 2024, 2025) | 0.119, 0.192, 0.486 | identical |
| Unidentified models; largest M; identified as w < 1 | 17; 247.6; 0 | identical |
| M\*(10^24) set; M\*(10^25) set | [5.68, 248.2]; [4.53, 468.4] | [5.69, 248.0]; [4.54, 467.9] |
| Breakdown tau: models; below 15B; 49 decisions; compute | 3.662; 7.536; 2.725; 1 | 3.664; 7.539; 2.727; 1 |
| SmolLM2 1.7B factor (P13) | 47.10 | 47.14 |
| S2(a) PI-1 as published: models, compute, 49 decisions | 0.870, 0.485, 0.816 | identical |
| S2(a) with Comma at [0.1, 3,571] | 0.078 | identical |
| S2(b); S2(b'); S2(c): models (compute) | 0.675 (0.172); 0.442 (0.093); 0.714 (0.173) | identical |
| S2(d) with S1 levels; with version 3's levels | 0.130; 0.143 | identical |
| S2(f) all in total parameters; all in N_F | 0.714 (0.173); 0.753 (0.235) | identical |
| Marin slope in total parameters; Llama 3 conversion bound | 0.247; [-0.247, 0.198] | identical |
| S4 model medians, k in [0.40, 0.52]: lower, upper (tau = 1); lower at 1.84, 3.4 | 0.405, 0.899; 0.241; 0.029 | 0.405, 0.899; 0.241; 0.030 |
| S4, k in [0.40, 0.684]: upper; sigma\* at the lower end of its set: upper | 0.951; 0.994 | 0.951; 0.994 |
| Fixed zero points, 49 decisions: reference; sigma\* 0.60; 0.62; sigma_lin(C_i) | 0.7345; 0.8736; 0.8507; 0.9571 | identical; the reference equals WP4b's unit median (0.7345) |

Not re-implemented: S2(e) (every technology as an anchor), which needs ra2's in-support anchors of 32 technologies. Its value, 0.234 of models and 0.028 of compute, equals version 3's PI-4 in rb2's committed file (0.2338 and 0.0275; the fourth decimal of compute moves with MPT-30B's corrected tokens).

## 3. The referees' variants (fix list D-3, S2)

| Referee's number | Variant | Module | Status |
|---|---|---|---|
| R1 New 3(a): 0.71 models, 0.66 decisions (56), 0.17 compute | PI-1 levels, slopes at the hull of their 95 percent intervals | 0.714, 0.661, 0.173 | reproduced |
| R1 New 3(c): breakdown about 8.3 under PI-1 | PI-1, share of models below one half | 8.3 | reproduced |
| R2 NM1: 0.84 models, 0.48 compute | PI-1 plus Comma on its t-interval, PI-1's slopes | 0.844, 0.477 | reproduced [review: new check] |
| R2 NM1: 0.68, 0.17 | t-interval levels, Comma with its own slope (S2(b)) | 0.675, 0.172 | reproduced |
| R2 NM1: 0.62, 0.17 | as above, "with the path slopes' intervals" | 0.584, 0.159 (version 3's bootstrap-t slope hull); 0.442, 0.093 (every slope at its own t-interval, S2(b')) | not reproduced; R2 does not state the slope range |
| R4 N2: 72.7 percent, 19.5 percent | slope hull [-0.175, 0.353] with DeepSeek's law at the models' total M and C | 0.727, 0.195 | reproduced; in DeepSeek's own convention (the module's) 0.714 and 0.173 |
| R2 NM1: Comma t-interval about [16.5, 39.5] at 3x10^20 | Comma's own line, 3 df | 25.9 [16.1, 41.7] | close; R2's minima differ slightly |

D-3's range, "62 to 87 percent of models, 17 to 48 percent of compute", contains S1 (78, 24) and every variant except (b') (44, 9) and the disclosures (PI-1 with Comma at 8 percent; normal inputs at 13; every technology at 23).

## 4. Coverage of the band

- The builder's Monte Carlo (`rb5_sign_review_coverage.csv`; 1,000 samples, 999 draws each, fitted paths as truth) shows that the wild bootstrap's q alone covers the whole path in 0.919 (Chinchilla), 0.911 (Llama 3) and 0.949 (Marin) of samples under normal errors. With the rule q = max(q_boot, q_normal) coverage is 0.961 to 0.964.
- The reviewer's own Monte Carlo (400 samples, grid supremum) gives 0.935, 0.930 and 0.925 for the bootstrap alone and 0.970, 0.963 and 0.945 with the rule. It confirms the undercoverage of the bootstrap alone with few budgets. It suggests Marin's may be larger than the builder's run shows (0.925 against 0.949, about two Monte Carlo standard errors). The rule is therefore the right choice; it changes no share on the fitted data (sensitivity row "S1 bootstrap critical value alone").
- The rule was added by the builder after its own Monte Carlo, that is, after the data were fitted. Because it raises the critical value and changes no reported share, it cannot have been chosen for its effect on the result. The memo discloses it.
- Both Monte Carlo exercises take the straight line as the truth. Section 5, point 2 shows that inside the designs it is not.

## 5. Findings

1. **The construction is correct.** The union over studies is taken over each model's placement in each path's convention (total parameters for the three IsoFLOP studies, DeepSeek's count for its law). The ln tau widening, the decision rule (a decision is identified only if every member is), the compute weights (sum of members' 6ND) and the exact breakdown point (the share is a step function of tau with drops at exp(dlo)) are all right. The magnitude bounds take the correct corners of the (k, d) box.
2. **The minima paths curve inside the designs** ([review]; `rb5_sign_review_curvature.csv`). A quadratic term in ln C is significant for Chinchilla (convex, t = 7.15, p < 0.001) and Marin (concave, t = -3.11, p = 0.008), and marginal for Llama 3 (convex, t = 2.39, p = 0.06). At the anchors the local slopes are 0.40 (s.e. 0.06) for Chinchilla, 0.25 (0.10) for Llama 3 and -0.05 (0.08) for Marin. So Marin's steep line reflects its small budgets, while Chinchilla's and Llama 3's flat lines average a fall and a rise. The band covers sampling error around a line, not this misspecification.
   - Sensitivity: add Chinchilla's and Llama 3's paths continued above their anchors at the local slopes, starting from the upper ends of their S1 bands. The set then identifies 0.75 of models, 0.20 of compute and 0.69 of decisions, and the upper end of M\*(10^24) is 339. With one standard error added to each local slope: 0.71, 0.17, 0.65 and 477.
   - Restricting the fit to the upper budgets is not a usable alternative: with budgets from 3x10^19 up, Comma keeps two budgets and Marin's critical value jumps to 8.5, so the set becomes uninformative for reasons of degrees of freedom, not curvature.
   - For the text: the headline survives, inside the S2 range. But "Chinchilla and Llama 3 are flat" should not appear as a statement about the paths beyond the designs.
3. **Marin sets the upper end and Llama 3 the lower end.** Marin's band alone would identify w < 1 for six flagships: Llama 2 70B, LLaMA 65B, Llama 3.1 405B, Falcon 180B, Falcon 40B and DeepSeek LLM 67B (7.8 percent of models, 46.5 percent of compute; comparison-only rows). The set is wide at the top because the steep path (Marin) and the flat one (Llama 3) disagree. "No model is identified as under-trained" rests on the union, not on any one study.
4. **Decision-level magnitudes** ([review]). H5 speaks of the median decision, but S4 reported medians over models. Over the 49 decisions, the median lower bound on s is 0.33 at tau = 1, 0.15 at 1.84 and -0.12 at 3.4 (k in [0.40, 0.52]; -0.16 with k up to 0.68). At DataDecide's factor the median decision's sign is not identified. Over models the figures are 0.41, 0.24 and 0.03.
5. **Two marginal models** (confirmed): StableLM-Alpha 7B (margin 0.004 in ln M against Marin's band) and Llama 2 13B (0.030). Any change of 3 percent in the envelope moves the share from 78 to 75 percent.
6. **Rounding in the paper numbers.** A compute share of 0.9954 was formatted "100", and several shares of 0.001 to 0.004 were formatted "0". They are now "<1" and ">99" (text: "less than 1 percent", "more than 99 percent"). The table keeps two decimals (1.00, 0.00), which is standard in a table.
7. **Small errors in the memo** (corrected): four models lie below Chinchilla's c_J, not three (SmolLM2 135M at 1.6x10^21 FLOP was missing); the check count read "23 of 23"; the sentence "every variant that ... admits Comma lands between 0.68 and 0.78" overlooked PI-1 with Comma's HC2 interval (0.08); cross-references to the review's sections were stale.

## 6. Changes made by the review

1. `rb5common.py`: the default `--units` is WP4b's final `data/processed/rb5_units/units_primary.csv`, so the one command reproduces the final state. The outputs are unchanged by the switch.
2. `run.py`: two comparison-only sets ("S1 without Marin", "S1 Marin alone"), which source the memo's statements on Marin's role. Comparison-only sets are excluded from the table note's "no model is identified as w < 1 in any row" check, since Marin alone does identify w < 1. The percentage formatting follows point 6 above. Decision-level S4 rows are added (`S4_dec49_*`).
3. `magnitude.py`: `unit_bounds_table`, the decision-level medians of the bounds on s (a unit's bounds are the compute-weighted harmonic means of its members'); `curvature_ranges` factored out.
4. `exhibits.py`: the table note says that the brackets are pointwise 95 percent intervals and that Marin's band starts at 3x10^20. The table was test-compiled with the main preamble (tectonic, session scratchpad): no error, no overfull box, one page.
5. `review_checks.py`: three new checks. R2's 0.84 and 0.48 are reproduced; the primary units are the same partition as `units_subgroup.csv`; the curvature diagnostics are written to `rb5_sign_review_curvature.csv`. The w < 1 check now covers the identified sets only. 27 of 27 checks pass.
6. `review_independent.py` (new, standalone): Section 2.
7. Memo: the corrections in Section 5, points 2, 4 and 7, each marked [review].

## 7. Points for the author and for the text packages

1. **Say which path sets each end.** Marin's pooled path (slope 0.198, fitted at 3x10^18 to 3x10^20 FLOP) sets the upper end for 76 of 77 models, and Llama 3's band the lower end for all 77. The upper end of M\*(10^24), 248, is about three times version 3's 88.
2. **Curvature.** One clause in IV.D or Appendix E, for example: "inside the designs the minima paths curve (convex in Chinchilla and Llama 3, concave in Marin); continuing Chinchilla's and Llama 3's top-budget slopes identifies 75 percent of models." It uses `rb5_sign_review_curvature.csv`.
3. **H5 (WP8).** With the median decision as the subject, the sentence's slot "[[S4: 0.30]]" has no positive value: at tau = 3.4 the median decision's lower bound on s is below zero. Either name the median model ("to 0.03") or say that an allowance of 3.4 leaves the median decision's sign unidentified.
4. **"About three quarters of models"** is safer than "78 percent" in the abstract-level prose (point 5 of Section 5); 78 is right in IV.D.

Unchanged from the builder's report and confirmed: v3's Comma interval is printed as recomputed, [0.2, 3,586] (v3 printed [0.1, 3,571]; the 8 percent is unchanged); the median decision share at the reference is 0.7345, which rounds to 0.73 (the fix list's "0.74" is version 3's); SmolLM2 1.7B's M is 47 times the upper end of the set at its compute; Llama 3.1 405B lies inside the set ([4.0, 677], M = 38.4).
