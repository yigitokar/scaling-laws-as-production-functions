# Independent review — module m5_progress ("algorithmic progress" as TFP growth)

Reviewer: Claude (independent replicator and referee). Date: 2026-09-23.

Scope:
- re-ran the module from scratch;
- audited all code (`common.py`, `s1`–`s6`, `run.py`) line by line against `paper/notes/model_spec.md`, Ho et al.'s notebook (commit 29c7d85) and `sl.py`;
- checked every number in `output/memos/m5_progress.md` against regenerated outputs;
- fixed what was wrong and re-ran.

**Bottom line.**
- The module is careful, well documented and largely correct. The replication of Ho et al. (2024) is exact, and every output reproduces bit-for-bit from raw data.
- I found **no bug that overturns a headline**.
- I found:
  - one inference error that affects a headline CI (issue 1);
  - one overstated interpretation of the headline "optimizer" finding (issue 2);
  - a silently broken cross-module input and an undocumented run-order dependency (issues 3–4);
  - a partly circular validation (issue 5);
  - a set of smaller reporting inaccuracies.
- All are fixed in code and/or memo.

---

## 1. Re-run

| Run | Code | Result |
|---|---|---|
| Run 1 | Builder's code, all module outputs deleted first | Exit 0; 36 min wall, 6,173 CPU-s. Every s1/s2/s3/s5 output is **bit-identical** to the builder's files (all `.npy` bootstrap draws, profiles, MC, tables; `.tex` identical except Table 7). Panel A registry rows Rm20/Rm21 differ by ≤ 4e-4, because m2's registry was regenerated after the builder's run. The s4/s6 outputs *lacked the m2 technologies*, because m2's registry had been deleted by a concurrent m2 rerun (issue 4). |
| Run 2 | Reviewer-fixed code, all module outputs deleted first | Exit 0; 35 min wall, 6,142 CPU-s. s3/s5 CSV and `.tex` bit-identical to run 1. m2's registry was absent during s2 (deleted by m2's rerun), so the refresh below followed. |
| Run 3 | `run.py --only s2,s4,s6` after m2 finished and its registry was stable | Exit 0; 17 min. Final Table 7 includes Rm10–Rm15 (m1 registry, 22:46) and Rm20–Rm25 (m2 registry, 23:42, which added a DataDecide row). Panel B is bit-identical to run 2 (per-cell RNG). s1 reran (it always runs) and reproduced run 2 exactly. |

Replication facts verified independently (not by re-running the builder's code):
- **Objective values.** I re-implemented Ho et al.'s `model_7` + `residuals` directly from their notebook (`check_obj.py`). Objective at their published θ: 0.0518129 (theirs: 0.05181292650905238). At the builder's converged θ: 0.0507227.
- **Optimizer trace.** SciPy's default SLSQP from zero gives **nit = 18, nfev = 209**, exactly the optimizer printout in their notebook. The builder's memo said 17 iterations, an off-by-one from reading the callback path index.
- **Bootstrap median.** The Ho-protocol bootstrap median is 8.44382859 vs their printed 8.44382833.
  - Their notebook runs with `use_cached_bootstraps = True`, i.e. the printed percentiles come from an earlier cached file. That is the likeliest source of the small tail gap ([4.39, 14.05] vs [4.52, 14.27]).
- **Ho appendix figure.** Ho et al.'s appendix figure "95.2% of draws above the anti-diagonal" is in `appendices.ipynb` (cell 51 prints 0.952).

---

## 2. Issues found and fixes

Severity scale:
- **major**: changes a headline number or its interpretation;
- **moderate**: affects a robustness or secondary result, or reproducibility;
- **minor**: presentation or wording.

| # | Severity | Issue | Fix | Status |
|---|---|---|---|---|
| 1 | major | **Invalid T_C bootstrap intervals when some draws have g_C ≤ 0.** `s1` took percentiles of T_C = 12 ln2/g_C directly. A draw with g_C ≤ 0 (infinite doubling time) became a *negative* T_C and sorted as the smallest value, pulling the lower bound down. The converged-estimator cluster CI, a headline number, was reported as [2.81, 21.26]; the correct g_C-mapped interval is **[3.05, 22.67]** (4 of 400 draws have g_C ≤ 0). The iid-1000 Ho-code interval [4.18, 15.64] becomes [4.34, 15.99]. | New `common.tc_q()` maps g_C percentiles (g_C ≤ 0 → ∞). It is used for every column except Ho's own B = 100 protocol, which has no g_C ≤ 0 draws and where direct percentiles reproduce their method. The table note was also corrected ("columns 5, 6 and 8", not "5–7"). | fixed (code, table, memo) |
| 2 | major (interpretation) | **The "published point is not the minimizer" finding is correct but was framed as if 6.1 months were the better estimate.** The objective gap (0.0518129 → 0.0507227) is almost entirely the L1 penalty: MSE 0.046416 → 0.046291 (−0.27%), Σ\|θ\| 2.159 → 1.773. The converged point sets all α_const terms to zero. Those constants depend on the arbitrary normalization (N₀, D₀, t₀ = sample minima), so the penalty selects the ridge point for reasons unrelated to fit. | Added MSE and Σ\|θ\| rows to the replication table. Headline 2 and Claim 2 now say that the point and CI are artefacts of the stopping rule and of a normalization-dependent penalty on a flat ridge (6.1–10.2 months across estimators), not that 6.1 months is correct. | fixed (code, table, memo) |
| 3 | moderate | **m1 technology registry silently ignored.** `registry_technologies` (s2) and `registry_chinchillas` (s4) filtered on `estimator == 'huber'`. m1's registry has no `role` column and labels the estimator `Huber-LSE delta=1e-3 …`, so it contributed nothing even when present. The builder's claim that "s2 and s4 pick up registries automatically" was false for m1. | New `common.read_registry()` handles both formats, logs a WARNING when a registry is missing, and records file mtime and SHA-256 under `registries_used` in `dmr_summary.json` and `allocative_summary.json`. m1 now contributes Table 7 rows Rm10–Rm15 (Chinchilla, Llama 3, Marin ×3, (Mis)Fitting) and 5 CE technologies. `max_rows` raised from 5 to 6, so m2's new sixth dataset (DataDecide) enters as Rm25. | fixed |
| 4 | moderate | **Undocumented run-order dependency, and RNG coupling across rows.** Table 7 Rm rows, panel B "m1:/m2:" rows, the Kaplan table and the named-model CE table exist only if the other modules' registries exist at run time. During this review, m2 deleted and regenerated its registry twice, and m1 created one mid-review. In s4 a single RNG stream ran across all (sample × technology) cells, so adding or removing a registry technology shifted the bootstrap CIs of *all* later cells, including the Besiroglu/Hoffmann rows (the builder noted this). | Per-cell seeded RNG, `default_rng([21, crc32(sample\|technology)])`: core rows are now invariant to registry presence. The dependency is documented at the top of the memo, with the command to refresh (`run.py --only s2,s4,s6`). Registry snapshots are recorded. | fixed |
| 5 | moderate | **Sahal "match" partly circular and selectively reported.** The memo said the inflation factors match γ/(1−s_A) "with s_A ≈ 0.13–0.37", but 0.13 and 0.31 are *backed out of* the inflation factors. The only independent check (rate-based s_A) was reported for record-setters (predicts 1.58 vs observed 1.44), not for all rows, where it predicts 1.77 vs observed 1.15. | Headline 9 and Claim 12 rewritten. The formula is supported for record-setters (corr(ln C, year) = 0.87). It fails on all rows (corr 0.62), where the ordinary omitted-variable formula applies. Also noted: compute = 6N × dataset size ignores epochs. | fixed (memo) |
| 6 | minor | **Profile-likelihood CI reported at grid points and called interior at "step 0.15".** The upper T_C bound (37.0) is also resolution-limited (the true crossing lies between g_C = 0.200 and 0.225). The grid step there is 0.025, not 0.15. | Added `lr_crossing()` (linear interpolation of the LR): T_C **[4.1, 40.5]**, φ [−0.39, 3.37]. Grid-point intervals are kept in the JSON. | fixed |
| 7 | minor | **Ridge-slope evidence incomplete.** The memo reported theory −1.20 and bootstrap −0.98, but not the valley-floor slope of the 2-D profile (−0.40), computed and stored by the code. "T_C rises monotonically along the valley floor" is false at grid resolution: there is a branch jump near α_year ≈ 0. | Memo now reports all three slopes and describes the floor accurately. | fixed (memo) |
| 8 | minor | **Iteration count** 17 → 18 (scipy `nit`, matching Ho's printout). | `s2` records `slsqp_default_nit`. | fixed |
| 9 | minor | **F-test for Hicks neutrality under the L1 estimator** is computed from MSEs at penalized optima, so it is not an F statistic. | Flagged `F_valid = False` in the JSON; memo says descriptive only. The bootstrap p = 0.29 stands. | fixed |
| 10 | minor | **"Tightest intervals in the module"** for the Hicks-neutral rows. A4 is narrowest in months, but A11 (imposed γ + neutrality) is tighter in relative terms (hi/lo 1.74 vs 2.28), and the data reject A11. | Reworded: "tightest among specifications the data do not reject". | fixed (memo) |
| 11 | minor | **E channel.** Ho's own data *prefer* E = 0 within the common-E form (LR 4.0 at E = 0.8, 13.5 at E = 1.5, not reported). And in the most realistic MC scenario (S2: E dropped, D = dataset size) Ho's procedure gives T_C 4.6 vs 12. The memo's "Whitfill's 9× correction does not apply to the E channel" needed the qualifier "at the global optimum". | Qualifiers added to Headline 6 and Claim 5. | fixed (memo) |
| 12 | minor | **Allocative sample contains duplicates and questionable entries** in era 2 (C ≥ 1e23): Gemma 1.1 7B Instruct, Qwen2-VL-72B, Qwen1.5-72B (= Qwen-72B values), Granite 3.1 (= 3.0 values), MegaScale 530B/175B (systems-paper runs), Falcon Mamba (SSM), FragLlama (molecular), and Amazon Titan/Hunyuan with D/N *exactly* 20 (likely imputed, CE ≈ 1 by construction). | Added the robustness sample "C ≥ 1e23, cleaned" (13 era-2 models removed) to s4, panel B and Table 7. Allocative gain moves 1.05 → 1.09 (Besiroglu), 1.91 → 1.99 (Hoffmann-TeX), 1.58 → 1.65 (Farseer). The headline is robust. | fixed (robustness row) |
| 13 | minor | **Figure labels.** Fig. 6 panel (c) "Hoffmann" was Hoffmann-TeX while panel (b) plots Hoffmann-rounded. The CEG figure called Ho's 7.2× "constant CEG", but Ho also report 6.6× at 1e22. The Gundlach 6.28× mark sits at an *assumed* compute (6N·20N, N = 3.6M) not stated anywhere. The attenuation figure's baseline row shows the *converged* estimator (α̂ 0.138, β̂ 0.025), not Ho's published values, and this was not labelled. | Labels fixed ("Hoffmann-TeX", "Ho et al. (2024): 7.2x", "small scale (C assumed)", "(a) Data variants (converged L1)"); memo inventory documents all four. | fixed |
| 14 | minor | **Epoch dataset-size parsing.** 24 rows store "a,b" pairs and are coerced to missing. Only Chameleon-34B and BlueLM 7B (era 2, C ≥ 1e23) would otherwise enter a sample. | Documented; not changed (negligible). | documented |
| 15 | minor | **Stated run time** in `run.py` ("15–25 min") and in the memo (51 min). | Updated: ~6,200 CPU-s; 36 min under moderate load. | fixed |
| 16 | minor | Docstring: the MC transmission shock ψ_D is row-level, not "paper-level". | Comment fixed. | fixed |

Things I checked and found **correct**:
- data pipeline equals Ho's cell by cell (including the WT103-perplexity quirk and `include == NaN` kept);
- α_year = α·g_N, β_year = β·g_D, g_C = g_N + g_D, and T_C = 12 ln2/g_C equals Ho's (1/T_N + 1/T_D)^−1;
- Hicks neutrality ⇔ α_year = β_year, identical to Ho's model 12 (single penalized parameter);
- g_C- and φ-profile parameterizations; `HoModel.technology()` (A = exp(α_const − α_year·t + α ln N₀));
- paper-cluster bootstrap (resampling of papers; full-sample normalization constants as in Ho's code);
- Muennighoff effective data D′ = U[1 + R*(1 − e^{−(e−1)/R*})], R* = 15.4;
- the Griliches–Hausman reliability formula λ = ρ + (1 − ρ)R² with R² from the *observed* ln D;
- CE = C_min(L(N, D))/6ND via `sl.Chinchilla.cost_efficiency` (E cancels);
- Kaplan counterfactual anchoring; CEG formula with E = 0; the MC truth construction (λ = γ g_C, g_N = λ/α, g_D = λ/β ⇒ g_N + g_D = g_C);
- selection rules; the chinchilla E = 0 refits (their bootstrap median equals the point, and the CI is right-skewed but correct);
- citation keys (all 17 exist in `lit/references.bib`);
- the literature numbers (Ho 8.4 [4.5, 14.3], 7.2× [3.3, 45.7]; Whitfill "≈9×"; Gundlach 10× of 6,930×, 6.28× small scale; GPT-3 C/C_min = 1.64).

No bug found in `sl.py`. I agree with the builder's minor note on `fit_chinchilla(E_fixed=…)` tolerances.

---

## 3. Independent verification beyond re-running

1. **Objective re-implementation** (`check_obj.py`). Confirms 0.0518129 (published θ) and 0.0507227 (converged θ). MSE 0.046416 vs 0.046291; gradient at the published point up to 9.6e-3 in α_year.
2. **Profile-likelihood robustness** (`check_profile.py`: 60 global random starts plus 10 local perturbations per point, 12 boundary points).
   - Every LR is reproduced to three decimals (e.g. g_C = 2.00: 3.824; 2.15: 4.414; 0.200: 3.887; 0.225: 3.681; φ = 3.25: 3.786; φ = −0.25: 2.982).
   - The unpenalized NLS minimum 0.0453029538 is global.
   - The profile CI claims are therefore numerically sound, conditional on the iid-Gaussian calibration.
3. **A7 degeneracy.** E_b (WT103, PTB, WT2) = (1.9675, 2.0376, 1.6004), each exactly at its upper bound (min y_b − 0.001), as the memo says.
4. **Allocative cleaning** (issue 12): gains 1.09 / 1.99 / 1.65 (Besiroglu / Hoffmann-TeX / Farseer) vs 1.05 / 1.91 / 1.58.

Scripts: `code/analysis/m5_progress/review_checks/` (`check_obj.py`, `check_nit.py`, `check_profile.py`, `check_tex.py`; see its README). They are not called by `run.py` and write no outputs.

---

## 4. Memo number audit

Every number in headline findings 1–9, the Methods, Claims 1–12 and §5 was checked against the regenerated CSV/JSON files.
- Numbers that were wrong or misleading and are now corrected: see issues 1, 2, 5–11 and 15.
- All other numbers matched to the stated precision. Examples:
  - 8.68 point; [4.07, 16.93] cluster; objective 0.0507227; 300/300 starts below the published objective; 53% within 1e-5;
  - corr −0.74/−0.73/−0.82; s̄ 0.545; a = 0.37; φ grid CI [−0.25, 3.25]; T_C 5.4–38 over φ;
  - neutrality p = 0.29/0.94; A3/A4 11.6 [6.6, 17.6] / 9.2 [6.3, 14.3]; A7 MSE 0.113 with g_C < 0; A8 54 [40, 99]; A6 76.5 [46, ∞);
  - γ 0.178 → 0.0525 [0.0524, 0.0538]; E profile 10.2 → 12.4; MC S1 11.1 / 11.2 [8.8, 15.5], S1h 4.2/4.4, S2 4.6 vs 10.6, S4 4.6 vs 9.0, S0 median 9.9;
  - a1–a3 and b/d variants; reliability 0.80, noise s.d. 1.8, R² 0.78, VIF 4.6;
  - CE GPT-3 0.61 / Chinchilla 1.00 / Llama-3-8B 0.18; era n 7/92, median M 1.7 → 92;
  - Kaplan 1.6 / 5.2 / 11.7× and 2.4 / 11.5 / 32×; C at 10×: 3.2e26 / 1.2–3.3e25;
  - Δγ 0.008 [−0.012, 0.034] p = 0.46; CEG at 1e19 0.45 vs 14.5; Sahal 1.15 [1.10, 1.21], 1.44 [1.24, 1.78].
- Allocative CIs (panel B) changed in the second decimal because of the per-cell RNG (issue 4). The memo now quotes the final values (§6).

---

## 5. Confidence in each headline claim (after fixes)

| Headline | Claim | Confidence | Why |
|---|---|---|---|
| 1 | Exact replication of Ho model 7; published 8.4 is a bootstrap median, point 8.68 | **High** | Independently re-implemented; nit/nfev identical; median to 3e-7. |
| 2 | Published point is not the minimizer of Ho's objective; converged T_C 6.1 [3.0, 22.7] | **High** (numerical fact) / **Medium** (interpretation) | The optimizer fact is robust (300/300 starts). The *economic* content is the fragility of T_C on a flat ridge under a normalization-dependent penalty, not the value 6.1. |
| 3 | DMR ridge; T_C weakly identified, profile CI [4.1, 40.5] | **Medium-high** | LRs verified with heavy multistart. The χ²(1)-iid calibration is not justified under paper clustering; the paper-cluster bootstrap under NLS gives [4.1, 27.6]. The lower bound sits on a flat stretch (LR ≈ 3) and is sensitive to the critical value. |
| 4 | Hicks neutrality not rejected; imposing it gives 9–12 months | **Medium** | Non-rejection reflects low power. The MC shows neutrality inside the E = 0 form is misspecified if E > 0 (pseudo-true 4.2–4.4 vs 12). |
| 5 | Imposing experimental exponents is rejected and gives g_C ≤ 0 or very slow progress | **High** | Holds for Besiroglu, Hoffmann, all 6 m2 and all 6 m1 technologies (14 in total; MSE ×1.3–4.8 relative to the unrestricted E_b fit; every registry-row CI includes g_C ≤ 0). |
| 6 | β_data gap mostly specification/measurement; T_C much less affected than exponents; Whitfill's 9× does not apply to the E channel | **Medium** | The first-order argument is correct and the E profile supports it. But the MC is uncalibrated, Ho's data prefer E = 0, and Ho's own procedure can overstate progress ~2.6× through local optima (S2). The γ-gap "2–2.5×" compares word- and token-level losses. |
| 6a | Effective data (Muennighoff) is the single most consequential data choice (T_C 25.7, CI to ∞) | **Medium-low** | 33% of epochs imputed; R* from unregularized LLMs; CI unbounded. |
| 7 | Realized allocative gains 1.05–2.4× for C ≥ 1e23 (2–40% of the Ho-rate gain); Gundlach's 10× is a counterfactual at 2025-frontier compute | **Medium** (Kaplan counterfactual: **High**) | Robust to sample cleaning. But: era 1 has n = 7; the result depends on the technology by ~2× among the main technologies, and by 0.55–4.3× across the m1/m2 sweep technologies (Llama 3's own gives 0.74×, i.e. not even the sign is robust); CE counts inference-motivated over-training as inefficiency; "share of Ho-rate gain" uses Ho's point rate (8.7 months). With the converged (6.1) or NLS (10.2) rate the denominator is 22.5× or 6.4×. The Ho rate comes from 2012–2023 small models. The Kaplan-counterfactual comparison is solid. |
| 8 | CEG scale dependence (P4) not identified in Ho's data | **High** | Δγ CI spans zero under both estimators; CEG levels span > 6 orders of magnitude. |
| 9 | Sahal inflation 1.15× (all) to 1.44× (records) | **High** for the estimates; **Medium** for "consistent with γ/(1−s_A)" | Only the record-setter comparison is an independent check (1.58 predicted vs 1.44 [1.24, 1.78]). |

---

## 6. Final regeneration record

- **Final outputs** = run 2 (full, from raw data) + run 3 (`--only s2,s4,s6`, after the m1/m2 registries were stable). Every file under `output/tables/m5_progress_*`, `output/figures/m5_progress_*` and `data/processed/m5_progress/` was regenerated by the reviewer-fixed code.
  - Registry snapshots used: m1 `technology_registry_m1.csv` mtime 2026-09-23 22:46:35, SHA-256 47832eca…; m2 mtime 23:42:40, SHA-256 a490883a… (recorded in `dmr_summary.json` / `allocative_summary.json`).
  - m1 started a new full run at 23:27 that may rewrite its registry. If it does, rerun `run.py --only s2,s4,s6`. Only rows Rm1x and the "m1:" CE technologies can change.
- **LaTeX.** All four fragments pass a brace, environment, math-mode and column-count check (`check_tex.py`); they were not compiled (no TeX installed).
- **Figures.** All four PNGs were inspected after regeneration. Labels were fixed (issue 13); no collisions that impair reading.
- **Numbers that changed relative to the builder's memo:**

  | Quantity | Builder | Final |
  |---|---|---|
  | Converged T_C, cluster CI | [2.8, 21.3] (direct) / [3.0, 22.7] (mapped) | **[3.0, 22.7]**, median 6.8 |
  | Ho-code iid-1000 T_C interval (JSON only) | [4.18, 15.64] | [4.34, 15.99] |
  | Profile-LR CI for T_C | [4.2, 37.0] | **[4.1, 40.5]** (grid [4.2, 37.0]) |
  | φ profile CI | [−0.25, 3.25] | [−0.39, 3.37] (grid [−0.25, 3.25]) |
  | SLSQP default iterations | 17 | 18 (nfev 209) |
  | Allocative gain, C ≥ 1e23, Besiroglu | 1.05 [0.56, 1.68] | 1.05 [0.57, 1.72] |
  | Hoffmann-TeX / Farseer | 1.91 [1.52, 2.43] / 1.58 [1.24, 2.01] | 1.91 [1.49, 2.43] / 1.58 [1.27, 2.04] |
  | Top-5 Besiroglu; frontier flag | 1.27 [0.93, 1.74]; 1.56 [1.09, 2.20] | 1.27 [0.96, 1.78]; 1.56 [1.08, 2.22] |
  | Top-5 Besiroglu share | 10% [−3%, 30%] | 10% [−3%, 32%] |
  | Experimental-technology rows | 2 + 5 (m2) | 2 + 6 (m1) + 6 (m2); all reject (MSE ×1.3–4.8), all registry CIs include g_C ≤ 0 |
  | Registry CE range, C ≥ 1e23 | 0.60–2.52 (m2) | 0.60–2.52 (m2), 0.55–4.31 (m1; Llama 3 0.74) |
  | New: cleaned-sample allocative gains | — | 1.09 [0.58, 1.74] / 1.99 [1.58, 2.57] / 1.65 [1.33, 2.12] |

- **Unchanged** (bit-identical): replication point estimates and bootstrap draws, multistart, all of Table 7 A1–A12, profiles, attenuation panels A–C, E profile, reliability, MC, CEG, Sahal, the Kaplan counterfactual for the four main technologies, and the named-model CE.

---

## 7. Remaining concerns (not fixed; for the writers)

1. **Inference calibration.** Profile-LR intervals are iid-Gaussian. A paper-cluster calibration (e.g. bootstrap of the LR statistic, or a cluster-robust score test) would make Claim 3 publication-grade. The paper-cluster NLS bootstrap interval [4.1, 27.6] is the safer number to quote alongside.
2. **Allocative "share of algorithmic gain".** The denominator uses Ho's *published-code* point rate. Given Claim 2, the paper should show the share under all three Ho estimators (8.7, 6.1 and 10.2 months give 8.9×, 22.5× and 6.4× over 2.28 years) or use the neutral estimate. The share is also not a decomposition in the accounting sense: Ho's model attributes N and D movements to the exponents, not to the trend (Claim 10).
3. **Cross-module dependency.** Rows based on the m1/m2 registries will change whenever those modules regenerate. Rerun `run.py --only s2,s4,s6` after the final m1/m2 runs and check `registries_used`.
4. **L1 penalty on normalization-dependent constants.** Ho's penalty (and hence their published point) is not invariant to the choice of N₀, D₀, t₀. The paper could note this in one sentence when presenting Claim 2.
5. **The MC is uncalibrated** (simulated log-ppl dispersion 0.8–8 vs observed 1.6–5.1). It demonstrates mechanisms only; the builder says so.
6. **LaTeX fragments were not compiled** (no TeX on this machine). I checked the regenerated `.tex` for brace and environment balance only.

## 8. Collateral note

At the start of this review I copied m5 backups into the shared scratchpad folder `…/scratchpad/orig/code/`. That folder already held another reviewer's (apparently m6's) backups, and my copy **overwrote their backup `orig/code/run.py`**. I removed all m5 files from `orig/` (so nothing wrong will be silently restored) and used `…/scratchpad/m5_review/` instead. No project files were affected. The m6 reviewer should not rely on `orig/code/run.py`; the live copy in `code/analysis/m6_montecarlo/run.py` is untouched.
