# Independent review: module rb1_sigmaC (σ\* as a function of compute; more extrapolation checks)

Reviewer: Claude, acting as replicator and skeptical referee. Date: 2026-09-24. Builder's memo: `output/memos/rb1_sigmaC.md`. The reviewer edited it in place, and every change there is marked "[review]". The builder's original code, memo and tables are archived in the session scratchpad and were not committed.

**Verdict.** The module is sound, and it answers the referee requests it targets. Five findings needed fixes:
- The significance of the drift was reported selectively.
- The design-specific drift test used unscaled standard errors.
- "No design drifts below 3×10^20" and "Farseer shows no drift" are not accurate.
- The extrapolation check's M range was overstated (Marin 3,706 and Llama 3 643, against evaluated ranges of about 1,100 and 290).
- R2 Major 1.1 (the budget-level appendix table with FE beside RE) and part of R2 Major 1.3 (symmetric windows, E-fitted power frontier) were missing.

All five are fixed in code, outputs and memo. No headline number of the builder changes. The review adds qualifications, three new sensitivity rows, one new check and one new table.

---

## 1. Reproduction

- **Builder's version.** `code/analysis/rb1_sigmaC/run.py`, re-run end to end from scratch into a scratch output root (`RB1_OUTPUT_ROOT`), reproduced all 30 table files byte for byte against the committed outputs. The run took 124 s on at most 4 CPU processes, with no GPU.
- **Corrected version.**
  - The project outputs come from one end-to-end run: `data/processed/rb1_sigmaC/run_stdout.txt`, followed by `--stages tables` re-runs for formatting-only changes to the tables code, appended to the same log. The builder's log is kept as `run_stdout_builder.txt`.
  - A second end-to-end run into a fresh scratch root reproduced every output file (see §7).
  - There are 32 table files (27 CSV, 5 .tex) and 2 figures.
- **LaTeX.** All five .tex tables compile in AEA.cls with `tools/tectonic`, with no overfull boxes.

## 2. Code audit (units, conventions, formulas, inference)

| Component | Checked | Result |
|---|---|---|
| Three-level REML (`meta_c.reml`) | V = diag(se² + τ_w²) + τ_d² design blocks; REML criterion ½[log\|V\| + log\|X'V⁻¹X\| + r'V⁻¹r] | Correct |
| CR2 (`meta_c.cr2`) | A_j = Φ_j^{1/2}(Φ_j^{−1/2}M_jjΦ_j^{−1/2})^{−1/2}Φ_j^{−1/2} satisfies A_j M_jj A_j' = Φ_j; Satterthwaite df = tr(Q)²/tr(Q²) with Q = PΦP' | Correct (symmetric-root variant of Pustejovsky and Tipton) |
| Wild cluster bootstrap-t | Restricted residuals; the restricted fit has zero slope by construction; Webb weights; CR2 t; working V held fixed | Correct. With 6 clusters the p-value is itself fragile |
| S-scale meta-regression | se_S = 2 se/σ² (delta method); map back S → σ swaps the bounds | Correct. **Fixed:** the S-scale prediction rows carried s.e. in S units beside σ-scale estimates; now NaN |
| Design-specific slopes and Wald test | FE, IV weights | **Error (fixed):** the covariance was not scaled by the residual variance (s² = 2.47), unlike every other FE-IV row. Scaled Wald: 11.2 (p = 0.047), not 27.7 (p < 0.001) |
| Hinge | Regressor (log10 C − 20.5)₊; the prediction coordinates map back correctly | Correct; knot chosen after seeing the data (the memo says "descriptive") |
| Study level | DL + modified HKSJ (ra1 `meta`); Marin with ρ = 1 s.e.; Farseer s.e. = max(wild s.e., path-mean bootstrap s.e.) | Correct. η formula σ\*_eff = 2/[2 + S/(1+η)²] follows from n_meas = (1+η)n with the frontier slope unchanged; it gives Marin's configuration-to-FLOP shift in sign and size (η = 0.087 gives 0.689 against the observed 0.706) |
| PI bounds and wedge propagation | ln w = k ln(M/M\*_ref), k = 1/σ\* − 1 (k_ref = 0.42744 recovered); family W_f = [Σω/w]⁻¹; magnitude bounds min/max of k·d | Correct. Reproduces ra2's PI medians (to 10⁻¹⁶), R3's 0.736 and R2's 19.6 |
| Local wedge (`extrap`) | x = ln M, c = ln C: f_n = f_c − f_x and f_d = f_c + f_x, so w = f_n/f_d; C = 6ND exactly in the FLOP-implied convention; parametric wedge αA N^−α/(βB D^−β), with κ cancelling | Correct |
| Extrapolation bootstrap | Basic intervals centred on the pilot-population statistic | Correct, but the bias (up to 0.10 on Marin κ-free) comes entirely from the Huber parametric refits. The local ln w moves by at most 0.02 and no evaluation point is lost in any draw. **Added:** `delta_bc` (the interval centre) and the pooled s.e. to the CSVs |
| Tuning (`tuning`) | ε = ε^obs + ι from L_obs = L\*e^ι; ln M\* = ln(C/6) − 2 ln N\*; D6: −(ι_n − ι_d)E/[(α+β)R\*] | Correct. D6 matches m8 Proposition F (a_obs − a = −∂[Δ/f'']/∂ln C, constant-gradient case) |
| Drift Monte Carlo | Profile rescaling L\* + λ(C)(L_ch − L\*) with λ = S_target/S_ch leaves argmin and frontier unchanged and sets S = λS_ch exactly | Correct |
| Figure 3 panel (a) | CR2 band | **Fixed:** below 10^19 the band was clamped by `np.interp`; the prediction grid now starts at 10^18.5 |

## 3. Every memo number checked against outputs

- Every number in H1–H5, C1–C7 and §5 was compared with the CSVs.
- **All match**, except the following, which are corrected in the memo:
  1. "Design-specific slopes differ (Wald 27.7 on 5 df, p < 0.001): Chinchilla −0.060 (0.021), Llama 3 −0.058 (0.010), Farseer +0.001 (0.007)". The s.e. were unscaled. Scaled: (0.033), (0.016), (0.011); Wald 11.2, p = 0.047.
  2. R3 minor 10: "t ≈ 3 in Chinchilla, 6 in Llama 3". These are unscaled IV t-values. ra1's bootstrap gives 2.8 and 5.4; the scaled FE values are 1.8 and 3.7.
  3. "The IsoFLOP-only interval [0.616, 0.740] is the widest". It is not: the η = −0.1 row, [0.578, 0.751], is wider.
  4. "R1's −0.05 ... adding Farseer's path and using random effects halves it". Random effects alone give −0.046 (IsoFLOP-only). Adding Farseer is what takes the slope to −0.032.
  5. The C2 caveat "Llama 3's largest budget carries 62% of that design's fixed-effect weight" is true of ra1's S-scale FE pooling. In the σ-scale meta-regression it carries 18% of Llama 3's inverse-variance weight, and 15% with τ_w². Clarified.
  6. Inventory counts (26 CSV / 28 files). Updated to 27 CSV and 5 .tex.
- Recomputed independently outside the module code (statsmodels WLS on the ra1 CSV):
  - design slopes in the ≤ 3×10^20 window;
  - the Farseer path slope: IV +0.001, OLS −0.015, matching ra1's −0.015;
  - the Llama 3 weight shares;
  - the screened DCLM run (157M: 3.605 at 9×10^19, 3.769 at 1.8×10^20, 3.597 at 3×10^20).

## 4. Substantive findings (adversarial)

**F1. The drift's significance was reported selectively (major; fixed in the memo and table note).**
- The memo and the proposed main-text sentence (C2) quote "wild cluster bootstrap p = 0.04".
- The CR2 t-test for the same primary fit gives p = 0.096 (df 3.2); unweighted, p = 0.043.
- On the scale S = 2(1/σ\* − 1), on which the wedge's k = S/2 is linear, the slope has CR2 p = 0.12 and wild p = 0.065.
- These are now in the memo (H1, C1, C2) and generated into the Table 1 note. The suggested wording is "p between 0.03 and 0.12".

**F2. "Below 3×10^20 no design drifts" is false. What is true is that the pooled slope is zero (major; fixed).**
- In the common window the design slopes are heterogeneous (scaled Wald 13.3, p = 0.021):
  - Llama 3: −0.050 (0.018), the same rate as over its full range;
  - Farseer: +0.038 (0.017);
  - Chinchilla and Marin: flat.
- The slopes depend on the weights: OLS gives Llama 3 −0.025 and Farseer −0.009.
- The builder's statements "the drift sits entirely above 3e20", "where every design has budgets and none drifts" (C1) and "below it no design drifts" (C2) are rewritten.
- The design slopes for both windows are in `rb1_sigmaC_metareg_design_slopes.csv`, with scaled and unscaled s.e.

**F3. Farseer was cited as a no-drift design (moderate; fixed).**
- Its IV-weighted path is flat. But ra1's own Farseer drift is −0.015 (s.e. 0.008, p = 0.048), and its largest level (10^21, 0.658) is far below its 2–5×10^20 values (0.72–0.74).
- The H3 text "If the drift is not real (Marin, Farseer)" now cites Marin only.

**F4. The evaluated M range of the extrapolation check was overstated (major for the paper's wording; fixed).**
- The designs' runs reach M = 3,706 (Marin) and 643 (Llama 3). With n_eff ≥ 8, the local wedge exists only to **M ≈ 1,112 (Marin)** and **M ≈ 290 (Llama 3)**:
  - Marin's "≥ 1,024" bin is 1–2 points at M ≈ 1,080–1,110;
  - Llama 3's "256–1,024" bin is two points at M ≈ 288–290.
- The direction ("parametric extrapolation understates at high M in three recipes") stands. The paper must not say it was checked up to 3,706 or 643.
- New columns `M_max_runs` and `M_max_evaluated` (`extrap_bandwidths.csv`) and `M_max_evaluated` by bin (`extrap_delta.csv`); a sentence in the table note, generated from the data; memo H4, C4 and R2 5.1.

**F5. R2 Major 1.1 was not answered (moderate; fixed).**
- R2 asked for an appendix *table* of the budget-level σ\*_b with window diagnostics, and the FE pooled estimate beside the RE estimate. The builder pointed to a CSV and to FE/RE *slope* rows.
- Added:
  - `output/tables/rb1_sigmaC_budgets.tex`: 44 budgets with runs left/right and M\*_b; per design, FE and RE pooled σ\* with Q (Llama 3: FE 0.628 against RE 0.660);
  - a study-level row with FE design means: 0.681 [0.606, 0.755].

**F6. R2 Major 1.3 was partly open. It is now done except quartic fits (moderate; fixed).**
- New `symwin.py`, run in the driftmc stage; output `rb1_sigmaC_driftmc_symwin.csv`.
- Count-symmetric windows trim each budget's window to equal runs per side and refit.
- Chinchilla + Llama 3 drift (design FE, OLS):

  | Variant | Drift per decade (s.e.) |
  |---|---|
  | Primary | −0.051 (0.017) |
  | Symmetric windows | −0.054 (0.025) |
  | E-fitted power frontier | −0.050 (0.016) |
  | Both | −0.053 (0.025) |

- On Marin, symmetric windows leave 2–3 runs per side, and the result is +0.021 (0.030).
- The builder also missed that ra1's specification grid already contained the E-fitted power frontier (Chinchilla −0.072, Llama 3 −0.043).
- The drift is not an artifact of asymmetric windows or of the log-cubic frontier.

**F7. The estimator range for the headline was incomplete (moderate; new sensitivity row).**
- The headline uses Farseer's local first-derivative estimator (0.708), per R4 R2-M4.3. The builder computed the same estimator on Marin and Llama 3 as a by-product: 0.666–0.676 on Marin (against ra1's 0.700–0.713) and 0.629 on Llama 3.
- Used in the study-level interval, it gives 0.673 [0.607, 0.739]. It is now a row of Panel D (`meta_c.add_local_fd_variant`, run in the tables stage).
- The variant means still lie within 0.663–0.708, but the intervals of these rows reach 0.61. The paper's "0.69 [0.65, 0.73]; 0.66–0.71 across conventions, accounting and bandwidth" should add "and estimators".

**F8. The partial-identification set is an estimated set (minor; documented and extended).**
- The upper bound σ\_top = 0.596 has HKSJ interval [0.567, 0.624].
- New: a revealed-demand row at σ\* = 0.624 (median s 0.858, decision units 0.847) and a magnitude-bound row with k_L = k(0.624) (PI-1 [0.734, 0.998]).
- The "reference is a lower bound" conclusion survives sampling error in σ\_top: 0.86 > 0.75.
- The lower bound is not conservative against an accelerating decline (the hinge gives −0.15 per decade). The builder listed this as an open issue; it is now also stated in H3 and in the table note.

**F9. Bias-corrected basic intervals exclude their own point estimates (minor; documented).**
- Marin κ-free, 256–1,024: −0.43 [−0.55, −0.43]. The bias comes from the Huber refits under noise, not from the local wedge, and it points toward a larger understatement.
- The CSVs now give `delta_bc` (e.g., pooled −0.49) and the s.e.
- Recommendation: quote "−0.43 (s.e. 0.03)", or the bias-corrected value with its interval.

**F10. Checked and not an issue.**
- CR2 at 10^19 is narrower than the model-based interval (s.e. 0.004 against 0.014), because the six designs agree there. The memo now gives both.
- Wild cluster p with six clusters: disclosed.
- Farseer's 8 overlapping-kernel levels are treated as conditionally independent in the three-level model. This affects the weights, not the CR2 inference. Leave-Farseer-out (−0.046) is reported.
- The Llama 3 wording on overstatement below M = 16 is corrected: it holds for the κ-free form only, since Farseer's Chinchilla form also overstates there (+0.16).

## 5. Referee comments: are they actually addressed?

| Comment | Status after review |
|---|---|
| R1 New 2.1 (meta-regression with design RE; PI of σ\*(C)) | Addressed. Significance qualified (F1). The PI set is an estimated set (F8) |
| R1 New 2.2 (Prop. 5(iv) bounds; level of s) | Addressed. Sampling-conservative row added |
| R1 New 2.3 (qualify abstract, p. 3, conclusion) | Wording in C1/C6. The plan's "10^19–10^21" should read "up to 3×10^20" or "0.65–0.70 at 10^19–10^21" (builder's flag, endorsed) |
| R1 New 2.4 (study-level interval, Marin one study; convention range) | Addressed. Estimator row added (F7) |
| R3 N1(a) (common compute level; largest budgets; meta-regression with caveats) | Addressed. Per-design values at 10^20 added to the memo |
| R3 N1(b), (d) | Addressed (wording; Panel D) |
| R3 N1(c) (revealed demand at σ\* ≈ 0.60; reference a lower bound) | Addressed. Robust to σ\_top's sampling error |
| R3 N1(e) | [TBD-m9]; `meta_c` and `extrap` can take the runs |
| R4 R2-M4.1–4.5 | Addressed. The Hessian path remains in the meta-regression, Figure 3 and Panel C (the only level-specific estimator; its mean 0.707 equals the first-derivative 0.708). The paper should say so |
| R2 Major 1.1 | Now addressed (F5) |
| R2 Major 1.2 | Addressed (predictions at 10^22–10^24; uninformative CR2 intervals disclosed) |
| R2 Major 1.3 | Addressed except quartic fits (F6) |
| R2 Major 1.4 (Chinchilla FLOP rebuild) | **Not done.** The η = ±0.1 band stands in (open) |
| R2 Major 1.5, 1.6 | Addressed |
| R2 Major 5.1 | Addressed, with the evaluated-M qualification (F4). Llama 3's high-M evidence is two points at M ≈ 290 |
| R2 Major 5.2 | Addressed. D6 formula verified against m8; Step Law gradients measured only to M = 466 (disclosed) |
| R2 minors 2, 20; R3 minors 10, 20 | Addressed (t-statistics corrected, §3 item 2) |

## 6. Bibliography and web evidence

**Bibliography.** All six new keys in `lit/bib/extra_round3_rb1_sigmaC.bib` were re-verified by the reviewer: authors, title, journal, year, volume, issue and pages.
- Crossref: `konstantopoulos2011fixed` (10.1002/jrsm.35), `hedges2010robust` (10.1002/jrsm.5), `tipton2015small` (10.1037/met0000011), `pustejovsky2018small` (10.1080/07350015.2016.1247004; vol. 36(4) 2018, online 2017) and `manski2003partial` (10.1007/b97478).
- `bell2002bias`: the Statistics Canada PDF (Survey Methodology 28(2), 169–181).
- The existing keys used are present in `paper/references.bib` or `lit/bib/extra_*.bib`. The new keys still have to be merged into `paper/references.bib`.

**Web evidence for coded items.** This module uses ra2's serving and family codes; it does not code them. Sample checks:
- SmolLM2-1.7B model card: "lightweight enough to run on-device" and 11T tokens, matching `deploy = ondevice` and D = 1.1×10^13.
- OLMo 2 7B card: 4T tokens in stage 1 plus 50B in stage 2, i.e. 4.05T, matching D. Minor: the 13B card lists 5T plus a 100B/300B merged stage 2, i.e. 5.1–5.3T, against the coded 5.15T. The size-specific-D classification is unaffected.
- Llama 3 8B/70B card: "15T+" tokens for both, matching the common-D herd.
- Meta AI assistant launched at Connect on 27 September 2023 (Meta newsroom; TechCrunch), matching ra2's date-level serving threshold.

## 7. Files changed by the review

**Code** (`code/analysis/rb1_sigmaC/`):
- `meta_c.py`: scaled design-slope covariance and Wald test, run in both windows; prediction grid from 10^18.5; S-scale s.e. columns set to NaN; FE-means variant; `add_local_fd_variant`.
- `pid.py`: σ\_top confidence interval in `pi_table`; σ\* = σ\_top,hi scenario; sampling-conservative magnitude bounds.
- `extrap.py`: `delta_bc` and `M_max_evaluated` columns.
- `tables_rb1.py`:
  - generated CR2/S-scale p-values in the Table 1 note;
  - Panel D rows c and d;
  - evaluated-M sentence in the extrapolation note;
  - new rows in the wedge table;
  - new `table_budgets`;
  - `M_max` columns in the bandwidths CSV; `delta_bc`/s.e. in the pooled CSV.
- `run.py`: passes σ\_top's CI to the pid stage; adds the symwin stage and the tables-stage variant.
- **New** `symwin.py`.

**Outputs.**
- New: `rb1_sigmaC_budgets.tex`, `rb1_sigmaC_driftmc_symwin.csv`.
- Changed: `metareg_design_slopes`, `metareg_predictions`, `study_level`, `pi_sigmaC`, `wedge_scenarios`, `magnitude_bounds`, `extrap_delta`, `extrap_marin_pooled`, `extrap_bandwidths` (CSV); `rb1_sigmaC_table.tex`, `_wedge.tex`, `_extrap.tex`; `rb1_sigmaC_panel.{pdf,png}`.
- All other outputs are byte-identical to the builder's.

**Memo.** `output/memos/rb1_sigmaC.md`, with "[review]" edits in the reproduction note, H1–H4, C1–C4, §3, §5, §6 and §7.

**Determinism.** A final end-to-end run into a fresh scratch root reproduced all 32 project table files byte for byte.

## 8. Remaining concerns (not fixable in this module)

1. **Chinchilla FLOP accounting (R2 Major 1.4).** It is the largest single term in the convention range: η = −0.1 moves the study-level mean to 0.665 with interval [0.578, 0.751].
2. **Quasi-homotheticity.** Holding M\*_ref(C) fixed while σ\* drifts is a scale propagation, not a re-estimation. "Reference = lower bound" holds conditional on M\*_ref.
3. **Few clusters.** The drift rests on two digitized designs and on six budgets above 3×10^20. Its significance lies between p = 0.03 and 0.12. A frontier-compute σ\* of about 0.60 is a within-design statement, not a pooled one.
4. **Llama 3's high-M check** is two evaluation points at M ≈ 290, and its local ln w is concave and bandwidth-sensitive. "In three recipes" is accurate for the direction only.
5. **Quartic window fits** (R2 1.3) are not run. Neither is a bootstrap for the symmetric-window variant (point estimates only).
6. **Exhibit budget.** Table 1 Panel D now has 12 rows. For the main text, keep the headline, Marin-independent, configuration N, η ±0.1, Farseer total N and local-FD rows, and move the rest to the Online Appendix.
