# Independent review of module m1_chinchilla

Reviewer: independent replicator and referee (Claude), 2026-09-23/24.

Scope:
- Re-run the module from scratch.
- Audit the code line by line.
- Check every number in `output/memos/m1_chinchilla.md` against regenerated outputs.
- Fix problems and document them here.

Revised text in the module memo is marked **[Rev]**.

---

## 1. Verdict

**The module replicates, and its point estimates are right.**
- `run.py` ran from scratch into an empty output root without errors in 47 min (load average 22–40).
- Every CSV, `.npy` and `.npz` it produced was bit-identical to the builder's.
- Formulas match `model_spec.md`. I checked the derived objects (M\*, L\*, σ\*, w) against brute-force numerical optimization; they agree to 1e-7.
- Key data facts check out: the 5 dropped runs, the −0.018 dex budget offset, the 137/132 IsoFLOP runs, and Llama 3's N = C/(6D).
- The Besiroglu-reconciliation claim is correct and verified against their notebook.

**Several headline inferential claims were overstated.**

1. **The duality test on Hoffmann's A3** (χ²₂ = 15.6, p = 0.0004) came from pairing an ad hoc heavy-tail-robust covariance with χ² critical values. Other versions of the test:
   - bootstrap-calibrated: p = 0.060 (stratified, either covariance), 0.21–0.38 (wild);
   - bootstrap-free classical F on the 9 argmins: p = 0.040;
   - SD-covariance χ²: p = 0.16.

   The evidence is marginal (p ≈ 0.04–0.06), not decisive.
2. **The on-path "SE explosion"** (×4.5–7) compared n = 41 with n = 240. Against random subsamples of the same size the inflation is 2.3× (α), 2.2× (a) and 1.25× (β), and none for γ. The profile-likelihood flatness and the Jacobian collapse are, by contrast, confirmed as genuine design effects.
3. **The estimator horse-race caveat was wrong.** The paired bootstrap on the shared draws shows the robust vs least-squares β gap is significant (z = 2–3); the M\* gap is not.
4. **The Llama 3 primal-vs-dual rejection** (p < 0.001) falls to p ≈ 0.01–0.05 once three things are accounted for: the primal's own sampling error, the understated stratified-bootstrap variance of the A2 slope, and small-sample calibration.
5. **The Marin gap was misattributed.** It is not "entirely finite-grid parabola bias": about a quarter is FLOP accounting, because the nominal budgets differ from 6ND by −7% to +35%.

All five are fixed in the memo, the tables, the registry and new code. There were also several minor bugs: a LaTeX-breaking string, a table/memo inconsistency, a mislabeled rule and column, hard-coded numbers in table notes, and figure label collisions. All are fixed.

---

## 2. Replication

| Step | Result |
|---|---|
| Full run of the original 8 stages into a scratch root (`M1_OUTPUT_ROOT`) | Exit 0. Stage times (cumulative s): horse 1141, selection 1528, duality 1761, fdep 2240, spec 2295, labs 2772, syslr 2825, report 2827. |
| Comparison with the builder's outputs | All 24 module CSVs: max relative difference 0. All `.npy`/`.npz`: byte-identical. `.tex`: identical, except `selection.tex` (my label fix). Registry: identical except root-relative paths. |
| New `checks` stage (`review_checks.py`) plus `report` | Exit 0, 4 min. |
| Final outputs in the project root | Old m1 outputs deleted. Replaced by the verified regeneration, the checks, and a `report` re-run in the project root (so registry paths are root-relative). |
| LaTeX | All 7 table fragments compile with `tools/tectonic` (booktabs, threeparttable, natbib): no errors, no overfull boxes. |

**Code state versus outputs.** After the regeneration started I changed three module files:
- `selection.py`: labels and column name only, applied before that stage ran, so the change is included in the run;
- `duality.py`: log text only;
- `boot.py`: docstring only.

Everything else (`review_checks.py`, `tables_tex.py`, `figures.py`, `registry.py`, `run.py`) runs in the `checks`/`report` stages, which were re-run with the final code. So every output corresponds to the current code. The builder's pre-review code is backed up in the reviewer scratchpad (`m1rev/code_before_review/`).

---

## 3. Issues found and fixes

Severity key: **major** = changes a headline claim or its inference; **moderate** = changes a secondary claim or a registry value; **minor** = presentation, labels or robustness.

| # | Severity | Issue | Fix | Status |
|---|---|---|---|---|
| 1 | major | **Duality headline (H6, Claim 5).** "Hoffmann's A3 violates the factor-demand restrictions (χ²₂ = 15.6, p = 0.0004)". This p-value pairs a robust covariance with χ² tails and is not robust inference. The SD covariance gives p = 0.16; the builder noted this but kept the robust version as the main test. | Added bootstrap-calibrated p-values (the same statistic, critical values from its bootstrap distribution) and a bootstrap-free classical F(2,7) on the argmin gaps. Hoffmann: calibrated p = 0.060 (stratified), 0.21–0.38 (wild); F p = 0.040 (level gap 0.122, SE 0.046). Refit and Besiroglu: p ≥ 0.16 everywhere. H6, Claim 5, the duality table (new column and notes) and the registry note are rewritten as "marginal (p ≈ 0.04–0.06), mainly the path level". | fixed |
| 2 | major | **On-path SE explosion (H9, Claim 7)** confounded identification with sample size (band n = 41 vs full n = 240, √(240/41) = 2.4). | Size-matched control: 4 random n = 41 subsamples of the full design, B = 100 each. Band/random SE ratios: α 2.3, a 2.2, β 1.25, γ 0.94, σ\* (κ = 1) 0.54. Random subsamples keep profile LR 79–101 (band ≤ 1.96) and condition numbers 69–95 (band 413), so the non-identification of σ\* with κ free is a genuine design effect. H9, Claim 7 and the identification table (new control row) are updated. | fixed |
| 3 | major | **Claim 2 caveat.** "Differences are 1.5–2.5 pairs SEs and under 1 cluster SE" compared differences with marginal SEs, ignoring the correlation across estimators on the same draws. It is also factually wrong: the levels − Huber Δβ = 0.060 exceeds the cluster SE of 0.046–0.055. | New `m1_chinchilla_estimator_diffs.csv` (paired bootstrap). Δβ: Gaussian 0.039 (SD 0.018, z = 2.2/2.0 pairs/cluster); levels 0.060 (0.020, z = 3.0/2.9). Δσ\* is significant too (z = −2.6, −3.2). ΔM\*(5.76e23) is not (z = −0.7, −1.4). H3, Claims 2–3 and the horse-race table notes are updated. | fixed |
| 4 | major | **Llama 3 duality rejection (H11, Claim 11)** reported with p < 0.001 / OLS p = 0.008. The OLS t treats the primal slope as known. The within-budget stratified bootstrap understates the A2 slope s.e. (SD 0.011, robust 0.006, vs OLS 0.018). The system LR uses χ² with 10 argmins. | Slope t with the primal's s.e. added: −2.1 (p = 0.035). Calibrated slope p = 0.050, path p = 0.010–0.015. Small-sample system LR p = 0.050. Wild p = 0.51–0.78. The claim is reworded to "p ≈ 0.01–0.05". The registry `se_a` for A2A1 rows is now max(robust bootstrap s.e., OLS s.e.): Llama 3 0.011 → 0.018. | fixed |
| 5 | major | **Marin (H11, Claim 12).** "The Approach-2 discrepancy is entirely the finite-grid parabola bias." The open-athena Marin budgets are 3 × forward FLOPs; 6ND/C_b ranges from −7% to +35%, strongly correlated with N within budgets (ρ = −0.89 to −1.00). The design-consistent null absorbs this too. | New decomposition (`m1_chinchilla_labs_design_decomp.csv`). Slope gap, Comma: 0.060 = 0.041 grid + 0.015 FLOP accounting + 0.005 residual (DCLM 0.036 + 0.014; Nemotron 0.030 + 0.010). The claim is corrected; the registry `D_convention` and the labs table notes document the mismatch. | fixed |
| 6 | moderate | "With n = 245, even our own primal fails the path test (p = 0.003)." | Calibrated p = 0.054–0.080; SD χ² 0.51; wild 0.73; classical F 0.041. Reworded to "borderline". | fixed |
| 7 | moderate | Frontier restrictions "rejected for every technology". This holds only under the stratified bootstrap. | Calibrated: stratified p = 0.003–0.037 (rejects); wild p = 0.15–0.70 for the design nulls (does not). Qualified in H6 and Section 5. | fixed |
| 8 | moderate | Nested system LR with n₂ = 9–10 argmins uses χ²₂. | F-form small-sample calibration added (`m1_chinchilla_system_lr_smallsample.csv`, with a row in each system table): n = 245 Laplace p = 0.048 (χ²: 0.020); Llama 3 p = 0.050 (0.024). | fixed |
| 9 | moderate | Registry A2A1 rows' `se_a` used the stratified SD: understated for Llama 3 (0.011) and dominated by failed parabolas for Marin Comma (2.81). The Hoffmann row note said "Fails the duality/revealed-preference tests". | `se_a` = max(robust bootstrap s.e., OLS s.e.), documented in `se_scheme`/notes. The Hoffmann note now states the computed test p-values and the cluster-CI caveat. | fixed |
| 10 | minor | The identification table showed s₄, s₅ while the memo cited s₃, s₅. s₄ does not collapse (0.013 → 0.010), so the table hid the collapse the memo described. | The table shows s₃–s₅; the notes explain that rank 3 implies two vanishing singular values (s₃ and s₅ in sorted order). | fixed |
| 11 | minor | LaTeX bug: the identification notes emitted `$|\\Delta\\ln N|\\le 0.15$` (double backslashes, i.e. line breaks inside math). | Raw-string fix; all tables compile. | fixed |
| 12 | minor | Hard-coded numbers in LaTeX notes: pairs/cluster w(70B) CIs, Llama OLS s.e. 0.018, "7–26 clusters". | Computed from the CSVs. | fixed |
| 13 | minor | Rule "drop 5 lowest loss" drops 6 runs (tie at the 5th-lowest loss). Column `dropped_minM` holds the *max* D/N of the dropped runs. Two runs tie at L = 3.4059 (ranks 6/7), so k = 6 depends on the sort order. | Label computed ("drop the 6 lowest-loss runs (L ≤ 5th lowest; ties)"); column renamed `dropped_maxM`; tie documented in code and memo. | fixed |
| 14 | minor | "β falls monotonically in k": there are upticks of ≤ 0.0006 at k = 9, 12 and 13. | "Almost monotonically". | fixed |
| 15 | minor | "'Chinchilla-70B was compute-optimal' rests on an arbitrary exclusion rule" is an overclaim. | Every rule that removes the gross outliers (k = 4–15, D/N < 0.4/1/2, 5 largest residuals) gives w(70B) ∈ [0.95, 1.06]. Only keeping them (k ≤ 2) or dropping the whole 10^19 budget (0.78) moves it. Reworded. | fixed |
| 16 | minor | Figure 1: the Chinchilla-70B label sat about 1.5 log-units from its star with no connector, on top of the Hoffmann path. Selection figure: the legend hid the Gaussian curve at k = 0–1. | Leader line and opaque box; figure-level legend below the panels. PNGs checked visually. | fixed |
| 17 | minor | The `boot.py` docstring claimed a warm-start diagnostic that was never implemented. | Implemented as `review_checks.warmstart_check`: 40 pairs + 40 cluster draws re-fitted with the 432-start grid added; 0/80 found a lower objective and the exponent SDs are identical. Docstring corrected. | fixed |
| 18 | minor | The duality log printed se(ln G) in parentheses next to a. | Label fixed (log only). | fixed |
| 19 | minor | The `sl.py` docstring says it "reproduces Besiroglu et al. (2024) exactly": `fit_chinchilla` returns the true Huber(1e-3) optimum, while `sl.BESIROGLU` holds their LAD-type published values. | Not edited (shared file, instructions). Reported here and in the memo (the builder had already flagged it). | reported |

---

## 4. Memo number audit

**Result.** Every numeric statement in the builder's memo was checked against the regenerated CSVs. The numbers themselves are correct to the stated precision. The problems were in interpretation and inference (Section 3), not in transcription.

**Spot checks with independent computation:**
- Huber(1e-3) objective:
  - published rounded: 1.0230e-3;
  - notebook-unrounded: 1.01864e-3;
  - ours: 1.01827e-3.
- Besiroglu notebook:
  - cell 24 (Huber likelihood with free scale, log-scale −12.27, s = 4.7e-6) produces the published values;
  - cell 18 (grid Huber) gives A = 477.6, matching our refit.
- M\*(C), L\*(C) and σ\* agree with numerical minimization of L(N, C/6N) and with a finite-difference EOS on the isoquant.
- Budget offset −0.018 dex; C is 4.2% below nominal; 137/132 IsoFLOP runs.
- The 5 dropped runs have D/N = 0.036–0.4045.
- Llama 3: N = C/(6D) exactly; loss 0.694–0.932; 10 budgets 6e18–1e22.
- Meta M\*(3.8e25) = 40.9 vs 16.55T/402B = 41.2.
- Kaplan's κ = 0.103 matches the SYNTHESIS ledger.

**Statements found wrong or imprecise** (all corrected):
- "under 1 cluster SE" (Claim 2);
- "monotonically" (H5);
- "5 lowest loss" (actually 6);
- "fails" (n = 245 path test);
- "entirely finite-grid bias" (Marin);
- "SEs ×4.5–7" (confounded with n);
- "3rd/5th singular value" in the memo vs "s₄, s₅" in the table;
- "Fails the duality … tests" (registry note).

**Citation keys:**
- All keys used exist in `lit/references.bib` or `lit/bib/extra_m1_chinchilla.bib`: besiroglu2024chinchilla, hoffmann2022training, czech2026problems, grattafiori2024llama, li2025misfitting, leonledesma2010identifying, klump2007factor, cameron2008bootstrap, kaplan2020scaling, koenker1978regression, hausman1977social, marin2026ladders, czech2026llama3isoflop.
- The Hausman–Wise entry (Econometrica 45(4), 919–938) is correct.

---

## 5. Code audit: checked and found correct

**Estimators** (`estimators.py`):
- LSE parameterization and gradients (via `sl`); Huber and Gaussian objectives.
- LAD: Huber(1e-6) followed by 3 restarted Nelder–Mead runs on the exact L1 objective.
- NLS in levels with an analytic gradient; VPNLS with NNLS on scaled columns (identical to NLS to 7 digits).
- KMW mapping `to_norm`/`from_norm`; Jacobians.

**Bootstrap** (`boot.py`):
- Pairs, cluster and within-budget stratified resampling units.
- Draws are generated in the parent process with fixed seeds, so the pipeline is deterministic (verified).
- Failed replications are recorded (2/300 at n = 245 in duality; 0 elsewhere).
- Warm starts are harmless (issue 17).

**Selection:**
- The fixed-threshold paired swing bootstrap is correct.
- The Hausman–Wise likelihood is correct (truncated normal, ln L < ln L̄).

**Duality:**
- A2 parabola argmin, and D\* via C_b/(6N\*).
- A1 NLS with E bounded below the minima.
- Design-consistent nulls; Wald subsets.
- System concentrated likelihoods: Laplace n ln mean|r|, Gaussian (n/2) ln mean r². The nested LR (level and slope shift) is correctly nested.

**fdep:**
- σ\* = 2/(2 + a₁ + b₁) is the on-path EOS of the inner aggregator for any κ.
- Profile parameterization at fixed S = a₁ + b₁, with forward and backward sweeps.
- Variance decomposition along (β, α)/‖·‖ and (α, −β)/‖·‖.

**spec:**
- CES σ = 1/(1 + ρ); Gaussian LR for κ (SSR = 2 × objective in both fits).
- Translog τ and its delta-method gradient; design-consistent τ_model.

**labs:**
- α = γ/a and β = γ/(1 − a) under κ = 1.
- M\* = exp((1 − 2a) ln(C/6) − 2 ln G).
- Meta law reproduction (the slope of ln D\* on ln C is 1 − a by construction).

**Registry:**
- θ-column order of every draws file; 29 rows; every cov/draws path exists; conventions are documented.

**Units:**
- Chinchilla: N is total parameters incl. embeddings (Hoffmann convention); D = C/(6N) from the digitized C; loss in nats (MassiveText).
- Llama 3: loss units unknown; flagged.
- Marin: nominal C_b ≠ 6ND (issue 5).

---

## 6. Remaining concerns (not fixed)

1. **Which bootstrap is right for IsoFLOP argmins is unresolved.** Within-budget pairs resampling perturbs the experimenter's design (duplicated N values, heavy-tailed parabola argmins); the wild bootstrap flips gross outliers.
   - The calibrated and classical tests bracket the answer, but a parametric bootstrap under the null (design held fixed, residuals from the fitted primal) would be the cleanest calibration for both the duality Wald tests and the nested system LR. It is not done (time).
2. **The size-matched control is small** (R = 4 subsamples, B = 100). Across subsamples, the random-subsample SE of β ranges from 0.036 to 0.125. The ratios (2.2–2.3 for α and a; 1.25 for β) carry ±30% noise. The qualitative conclusion (α and a lose precision on-path; γ does not; σ\* is unidentified with κ free) is robust.
3. **κ < 1** (κ̂ = 0.774; cluster p = 0.042) rests on digitized, quantized IsoFLOP profiles and is not replicated. The warm-start check covered the Huber primal only, not the κ bootstrap.
4. **Llama 3.** Loss units are unknown, and N = C/(6D) assumes Meta's budgets are 6ND (Meta's FLOP formula may include attention). The primal-vs-A2 slope gap is significant only at about 5% under conservative calculations.
5. **The Chinchilla cluster bootstrap has 9 clusters.** No wild-cluster or CR2 small-cluster correction was tried. M\* at frontier scale is effectively unidentified from Chinchilla alone ([3.2, 126]).
6. **The frontier-level rejection** (A1 minima 0.005–0.008 nats below every A3 frontier) is not robust to the wild bootstrap. It may reflect parabola-minimum bias in L\* rather than misspecification of the Chinchilla form; not investigated.
7. **CSV headline columns.** `m1_chinchilla_duality_tests.csv` and `m1_chinchilla_labs_duality_tests.csv` still carry the χ² p-values. Writers must read the calibrated and classical tests in the new CSVs, the revised memo, and the new duality-table column.
8. **`sl.py`** docstring imprecision (issue 19) is left to the lead author.

---

## 7. Confidence in each headline claim (after revision)

| Claim (memo numbering) | Confidence | Note |
|---|---|---|
| H1 / C1: Besiroglu's published values are an LAD-type optimum (Huber likelihood with a free scale); Huber(1e-3) gives A = 477.8, B = 2143.4 | **High** | Verified against their notebook cells 18 and 24. |
| H2: Huber(1e-3) ≈ LAD (83.8% linear region; SD of paired Δβ 0.004) | **High** | |
| H3 / C2: robust vs least-squares β differ (paired z = 2–3); M\* differences not significant | **High** (revised form) | The original caveat was wrong. |
| C3: σ\* ≈ 0.72–0.74 is the most stable object under κ = 1 | **Medium-high** | Stable in relative terms; still statistically different across estimators; depends on κ = 1. |
| H4: cluster SEs 1.5–2.2× pairs; M\* at 5.76e23 unidentified under the cluster bootstrap | **High** (numbers), **medium** (9-cluster inference) | |
| H5 / C4: the k = 0→5 swing is outlier leverage, not truncation bias; Δβ = 0.086 (SE 0.052) | **Medium-high** | Hausman–Wise relies on Gaussian errors; the regressor-based rule gives the same answer. |
| H6 / C5: Hoffmann's A3 is inconsistent with its own IsoFLOP argmins | **Medium-low** as originally stated (p = 0.0004); **medium** as revised (marginal, p ≈ 0.04–0.06, path level ~12%) | Depends on the test. |
| H6: Besiroglu and refit consistent with the argmins | **High** | p ≥ 0.16 in all variants. |
| H6: frontier restriction rejected for all technologies | **Medium-low** | Stratified only. |
| H7 / C6: w(70B) = 0.71 (Hoffmann) vs 1.04 [0.82, 1.42] (refit) | **High** (numbers), **medium** ("rejects": the cluster CI contains 0.71) | |
| H8: system estimator; nested LR 1.01 (p = 0.60) at n = 240 | **Medium-high** | n = 245 rejection is borderline under the small-sample calibration. |
| H9 / C7: on-path data cannot identify σ\* with κ free; α and a lose precision; γ identified | **High** for the profile flatness and Jacobian collapse (survive the size control); **medium** for the SE-inflation magnitudes | |
| H10 / C8: κ = 1 rejected (κ̂ = 0.774), and σ\* falls to 0.700 | **Medium** | Digitized data; cluster p = 0.042; not replicated. |
| H10 / C9: CES not rejected | **Medium-high** | Low power. |
| H10 / C10: rank-one translog | **Low-medium** | Depends on E. |
| H11 / C11: Meta's A2 law reproduced exactly | **High** | |
| H11 / C11: the primal on Meta's points implies a = 0.527 vs 0.463 (significant) | **Medium-low** | p ≈ 0.01–0.05; not under the wild bootstrap; unknown loss units. |
| H11 / C12: the Marin gap is a design artifact (grid + FLOP accounting) | **Medium-high** (revised) | |

---

## 8. Files added or changed by the review

**Code** (`code/analysis/m1_chinchilla/`):
- `review_checks.py`: **new**, stage `checks`.
- `run.py`: adds the `checks` stage.
- `tables_tex.py`:
  - duality table: calibrated-p column and classical F;
  - identification table: s₃–s₅ and a control row; LaTeX fix;
  - computed notes;
  - paired differences in the horse-race notes;
  - small-sample LR rows;
  - labs Panel C additions;
  - `≤` in the selection labels.
- `figures.py`: 70B leader line; selection legend.
- `registry.py`: A2A1 `se_a`; Marin `D_convention`; computed Hoffmann note.
- `selection.py`: rule label and column name; tie comment.
- `boot.py`: docstring.
- `duality.py`: log label.

**New outputs** (`output/tables/`):
- `m1_chinchilla_estimator_diffs.csv`
- `m1_chinchilla_duality_bootcal.csv`
- `m1_chinchilla_duality_classicalF.csv`
- `m1_chinchilla_labs_design_decomp.csv`
- `m1_chinchilla_system_lr_smallsample.csv`
- `m1_chinchilla_warmstart_check.csv`
- `m1_chinchilla_fdep_sizecontrol.csv`
- `m1_chinchilla_fdep_se_ratios.csv`

**Regenerated outputs:**
- all `m1_chinchilla_*` tables (`.csv`/`.tex`), figures (`.pdf`/`.png`), `technology_registry_m1.csv`;
- `data/processed/m1_chinchilla/*`.

**Memos:**
- `output/memos/m1_chinchilla.md`: revised, with [Rev] marks;
- `output/memos/m1_chinchilla_review.md`: this file.
