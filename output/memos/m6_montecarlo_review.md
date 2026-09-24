# Review — module m6_montecarlo (independent replication and referee report)

Reviewer: independent replicator and skeptical referee (Claude), 2026-09-23. Scope:
- the code in `code/analysis/m6_montecarlo/` (`run.py`, `mc_lib.py`, `design_a.py`, `design_b.py`, `verify_bias.py`, `outputs.py`);
- every module output;
- the memo `output/memos/m6_montecarlo.md`, which I corrected in place. Its revision note summarizes the changes.

**Bottom line.**
- The module is careful, and its core results survive.
- It reproduces bit-for-bit from raw data.
- I found:
  - two estimator/design problems in the κ-free profile likelihood, both biased toward the paper's thesis but small in effect;
  - one mis-reported estimator (means and RMSEs for just-identified 2SLS);
  - one inference headline that was an implementation artefact (on-path bootstrap coverage);
  - about a dozen memo statements that were overclaimed, mis-stated or missing a caveat.
- All are fixed. Two checks became permanent `run.py` stages: a warm-start vs multi-start bootstrap check, and a decomposition estimator (E1b).
- The module was re-run from scratch after the fixes.

---

## 1. Replication

- **From scratch.** I deleted every module output (`output/tables/m6_montecarlo_*`, `output/figures/m6_montecarlo_*`, `data/processed/m6_montecarlo/*`) and ran `.venv/bin/python code/analysis/m6_montecarlo/run.py --procs 6`. It finished without error in 44.8 min wall-clock on a heavily shared machine (load average 35–54 on 18 cores; the builder's run took 24.7 min at load 25–33):
  - Design A: 27 min;
  - Design B: 9 min;
  - new bootstrap-start check: 7 min;
  - verification and outputs: 1.5 min.
  - A final `run.py --skip-sim` (8 s) re-rendered the figures and tables after two cosmetic edits to `outputs.py` made while the run was in progress: a label position in Fig. 2C and one table-note wording.
- **Bit-identical reproduction.** Compared with the builder's saved replications (merged on keys):
  - all 40 non-profile columns of `designA_reps.parquet` (9,000 rows) are identical, with max absolute difference 0.0 and identical NaN patterns;
  - all 44,000 original rows of `designB_reps.parquet` are identical (max absolute difference 0.0);
  - `verify_transmission.csv`, `verify_selection.csv`, `verify_identities.csv` and `noise_calibration.csv` are identical;
  - `designA_summary.csv`, `designA_diagnostics.csv`, `industry_meta.csv`, `transmission_check.csv`, `selection_check.csv` and `identities_check.csv` are byte-identical.
  - Only the profile-likelihood columns changed (by design, fixes 1–2 below), plus the new E1b rows and the bootstrap-check file.
- **Smoke test.** `run.py --quick` also runs end to end (3.5 min) and writes only to `data/processed/m6_montecarlo/quick/`, which I deleted afterwards.
- **LaTeX.** No TeX installation is available, so I checked all six `.tex` fragments structurally: balanced braces and environments, cell counts per row equal to the column spec, and $-parity. All pass.

## 2. Issues found

Severity: **major** = changes a headline number or claim the paper would print; **moderate** = a code or estimator error with a small effect, or a misleading table cell; **minor** = wording, precision or a missing caveat.

| # | Severity | Where | Issue | Status |
|---|---|---|---|---|
| 1 | moderate | `mc_lib.profile_sigma` | The profile LR was measured against min(grid SSR, κ = 1 SSR). That is only an *upper bound* on the unrestricted κ-free minimum, so every LR was understated. The mean understatement was 0.001–0.17 in near-path cells (90th percentile ≤ 0.44) and 0.05–0.22 in IsoFLOP/factorial cells (90th percentile ≤ 0.61). The unrestricted optimum lies between grid points, or outside [0.50, 0.95] in 17–53% of near-path replications. This inflated the flat-profile shares by 0–3.4 pp and the IsoFLOP coverage of the truth (96.0% → 94.0%), i.e. it was biased toward the thesis. | **Fixed**: new `kap_obj_free` with S free (analytic gradient checked against finite differences) and an L-BFGS-B fit from 3 starts. The residual gap to a brute-force 1-D search is ≤ 0.06 LR in 8 checks. `prof_shift` stores the old understatement; `designA_profile.csv` reports the flat share under the old denominator. |
| 2 | moderate | `design_a.SIG_GRID` | The 14-point σ* grid had spacing 0.013–0.017 around the truth, wider than the IsoFLOP/factorial profile CIs (half-width ≈ 0.008). Their reported widths (0.011 IsoFLOP ±16×, 0.016 factorial) were artefacts of linearly interpolating a convex LR across one grid step: a chord lies above a convex function, so widths were understated. They were also inconsistent with the Wald (0.015/0.019) and bootstrap (0.016/0.019) widths. | **Fixed**: 4 points added (0.728, 0.732, 0.742, 0.746; 18 in all). New widths: **0.016** (IsoFLOP ±16×), **0.019** (factorial), 0.047 (±4×), in line with Wald and bootstrap. |
| 3 | moderate | `outputs.industry_tex`, `fig_industry`, memo H7/H9 | Just-identified 2SLS (E7) has no finite moments, so its Monte Carlo mean and RMSE are driven by rare extreme draws. Under the target rule a single γ̂ error of 2.54 produced a mean bias of +0.026 and an RMSE of 0.177, against a median bias of −0.002 and a MAE of 0.027. The memo reported the mean bias as a finding. | **Fixed**: tables (‡ row plus note) and the figure report median bias and MAE for E7. The memo is updated: median IV bias is −0.002 to +0.005 without selection, and −0.022 / −0.033 with selection. |
| 4 | major | `design_a` bootstrap; memo H4, claim 5 | "Bootstrap coverage for σ* on the path is 67%" was presented as a failure of standard inference. It is largely an **implementation artefact**: every bootstrap draw was warm-started only at the replication's own estimate, which on the flat ridge understates the spread. | **Tested and added to `run.py`** (`design_a.run_bootcheck_chunk`, `m6_montecarlo_designA_bootcheck.csv`; 60 fresh replications × 49 draws, same resamples fitted both ways). On the path, coverage for σ* is 57% warm-start vs **92% multi-start** (a: 50% vs 90%; ln M*: 58% vs 90%). The multi-start intervals are nearly uninformative: median width 0.43 for σ* and 0.84 for a. At s = 0.3 the two bootstraps coincide (σ* 100%, width 0.053). H4 and claim 5 are rewritten: "on the path, standard inference is either wrong (warm start, Wald) or empty (honest multi-start bootstrap)". Table 1 carries a § note. |
| 5 | major (memo) | memo H2, claim 3 | "For 0.2 ≤ s ≤ 1 … relaxing κ removes [the κ = 1 precision]" is false at s = 1. There the κ-free CI is informative (median width 0.104, never flat); at s = 0.5 the claim holds only partly (edge-hitting 68%). The "functional-form" interpretation rested on a numerical coincidence (primal RMSE ≈ dual RMSE). | **Fixed**: the claim is restricted to 0.1 ≤ s ≤ 0.3, and the analytic reason is added. With κ = 1, σ* = 2/(2 + γ/(a(1−a))) ≈ 2/(2+4γ), because a(1−a) is flat at a ≈ ½ (0.2498 at the truth), so σ* is pinned by the on-path γ. This explains why the primal σ* RMSE is 0.013 at s = 0.3 while the RMSE of a is 0.098. |
| 6 | minor (memo) | memo H1 | "The object that on-path data cannot identify at all is the MRTS level M*" is wrong: under optimality M* is the observed D/N, and the dual estimator's RMSE is 0.009. It is the *primal*, optimality-free estimate that fails. | Fixed. |
| 7 | minor (memo) | memo H1 | "RMSE 1.97 means the 95% band spans a factor of roughly 50 in M*" is wrong. At s = 0.3 the 5–95% range of ln M̂* is [−0.40, 6.34], a factor of about 850. | Fixed. |
| 8 | minor (memo) | memo H7 | "MC s.e. ≤ 0.001 unless noted" was false for the pooled Huber-LSE (0.0011–0.0019) and for the IV (0.002–0.009). | Fixed. |
| 9 | minor (memo) | memo H5, claim 6 | The n = 240 OLS coverage rates (53%, ≤ 4%, 16%, 0%) depend on the assumed sd(η) = 1 of the ω-independent part of ln C. The memo stated them without that qualifier. | Caveat added. |
| 10 | minor (memo) | memo H5 | The forward/reverse bracket "holds under exogenous and target rules" fails under a *common* target, where Cov(c, y) = 0 and the reverse regression is undefined (simulated −31). | Fixed; cross-referenced to m7_theory's iff condition. |
| 11 | minor (memo, tables) | memo H7, claim 9; industry table notes | Both ACF variants use *current within-lab-generation input deviations* (with squares and product) as instruments. Without them the lagged moment set has 8 moments for 10 parameters and is under-identified. So "ACF works" borrows the within-family variation of E5 and says nothing about flagship-only panels. The label "lagged instruments only" hid this. | Caveat added to the memo and to all industry table notes; the row is relabelled in the memo. |
| 12 | minor (memo) | memo H7, claim 12 | The mechanism "ML practice attributes over half of progress to scale" was asserted without a decomposition. Ê is also biased and compresses measured progress. | **Tested and added** as estimator E1b (E fixed at the truth, still pooled with no time effects). With E known, TFP growth bias is −0.054 / −0.065 / −0.062 (exogenous / funding / predetermined) and γ̂ bias +0.050 / +0.059 / +0.057. So about two-thirds of E1's absorbed progress is the scale (Sahal) channel and one-third is the Ê level. The claim is reworded to "absorbs over half … into the fitted scaling law". |
| 13 | minor (memo) | memo H8, claim 11 | "\|bias\| ≤ 0.004 for a, γ, σ*" was false (a: −0.0044 under target). "+1.25 in every scenario" is really a range of +1.24 to +1.28. | Fixed. |
| 14 | minor (memo) | memo §5 | "In some cells (s = 0.1, on-path, Kaplan) Wald coverage is 99–100%" is wrong: Kaplan is 92%. | Fixed. |
| 15 | minor (memo) | memo H7, claim 8 | "Within-family is the estimator most hurt by ME in D" is an overclaim. Lab FE (−0.28; shift −0.35 relative to no ME) and Mundlak (−0.25; −0.32) are hurt about as much as E5 (−0.33; −0.31). | Fixed. |
| 16 | minor (memo) | memo claim 1 | The claim's numbers are for s = 0, but its "median bias −0.030, MAE 0.037" are for s = 0.02. At s = 0 they are −0.297 and 0.297. | Fixed; both reported. |
| 17 | minor (memo) | memo §2.3, `design_b` docstring | "ε sd 0.05 in y ≈ 0.01 in ln L at 10^21–10^23" holds only at 10^21. Since d ln L/dy = −R/L, it is ≈ 0.011 at 10^21 and ≈ 0.005 at 10^23. | Fixed. |
| 18 | minor (memo) | memo H7, claims 10 and 12, open issue 2 | "RMSE(a) = 0.004 in all scenarios" (it is 0.0035–0.0048, and 0.0085 with flagships only); "Ê biased down by about 0.05" (it is 0.055–0.061, and 0.034 under target); "Ê 1.756–1.783" mixed the target rule with the others. | Fixed. |
| 19 | minor | `design_a._interp_ci` | Behaviour was undefined when LR exceeds the critical value at every grid point, which becomes possible once LR uses the true minimum. | Fixed: returns an empty CI, counted in `share_ci_empty_on_grid` (at most 0.7%). |
| 20 | minor | `outputs.fig2` | The "exact at s = 0" label in Fig. 2C collided with the dual-estimator line. | Fixed (label moved; PNG inspected). |
| 21 | note | `mc_lib.fit_huber` | One of the 9 starting values is the truth. On a flat ridge this favours near-truth solutions, so on-path primal performance is optimistic and the H1 claim is conservative. It is disclosed in the table notes. | Added to the memo's claim 1 caveat. |
| 22 | note | `mc_lib.asym_cov` | The LAD sandwich 1/(4f(0)²) approximates the Huber(10⁻³) variance. At noise sd 0.0075 the exact Huber factor E[ψ²]/E[ψ′]² is about 7% smaller, so Wald SEs are about 3.5% too wide (nominal 95% becomes about 95.8%). | Documented; conservative and negligible (IsoFLOP Wald coverage 96.0%). |

## 3. Independent tests I ran

1. **Profile denominator.** In 8 replications at s = 0.3 I compared the builder's denominator min(grid, κ = 1) with a brute-force 1-D minimization of the κ-free profile over S (25-point log grid plus bounded Brent).
   - The old LR was understated by 0–0.37 in those replications.
   - The new free-S fit recovers the brute-force minimum to within 0.06 LR.
   - In the full run the mean understatement was 0.001–0.17 (near-path cells) and 0.05–0.22 (IsoFLOP/factorial).
   - Flat shares under the old vs new denominator: on-path 88.7% → 88.0%; s = 0.05: 92.7 → 91.3; s = 0.1: 95.3 → 93.3; s = 0.2: 92.7 → 89.3; s = 0.3: 65.3 → 65.3; Kaplan: 78.7 → 77.3.
2. **Bootstrap starts.** First a scratch test (30 replications × 39 draws: 77% vs 97% coverage), then the permanent `run.py` stage (60 × 49): on the path, σ* coverage is 57% warm-start vs 92% multi-start, with median widths 0.28 vs 0.43. At s = 0.3 both give 100% and width 0.053.
3. **ML-practice decomposition (E1b).** See issue 12. E1b also shows that the ln M* bias of pooled ML practice (+0.55 to +0.61 with E known) comes from pooling without time effects, not from E.
4. **Prop. 2 inside the industry DGP.** With flagships only and generation effects, Prop. 2 predicts Cov(c, ω)/Var(c) = 2·0.0625/(0.49 + 0.16 + 0.16 + 0.25) = 0.118 (variances of h, p, ν and 2x). Pooled NLS gives +0.119. The industry simulation and the closed form agree; this check is added to claim 7.
5. **IV tails.** Under the target rule the γ̂ error distribution has 1st, 99th percentiles and maximum of −0.068, 0.377 and 2.54, against a median of −0.002.
6. **E1b and IV code paths** were exercised in the smoke test and the full run with 0 failures. `industry_failures.csv` shows failures only where estimators are not identified by design: the within-family estimators with flagships only.

## 4. What I checked and found correct

**Formulas against `model_spec.md`.**
- `mc_lib.derived` computes:
  - a = β/(α+β), γ = αβ/(α+β), σ* = 2/(2+α+β);
  - ln M*(C) = (1−2a) ln(C/6) − 2 ln G, with G = (αA/(βB))^{1/(α+β)}.
  - This matches N* = G(C/6)^a and D* = (C/6)^{1−a}/G and gives M*(10^24) = 18.1, consistent with the ledger (18.4 at 5.76e23, 16.5 at 3.8e25).
- `kappa_equivalent_start`: re-derived by hand. b₁ = aS, a₁ = (1−a)S, κ = γ/(a(1−a)S), K′ = K^{1/κ}, A′ = K′aG^{a₁} and B′ = ((1−a)/a)A′G^{−S} reproduce the Chinchilla expansion path and frontier. The `kap_obj` gradient was derived and checked.
- `design_b.alloc_d`:
  - the FOC αu/(βv) = 1 + T/(3D) follows from min 6ND + 2NT s.t. L ≤ ℓ;
  - h(d) is increasing and concave, so Newton from the T = 0 root converges monotonically (the docstring's tangent argument is right);
  - the T = 0 root equals Lemma 1's d*, including the −aψ_D tilt;
  - inference demand moves labs to smaller N and more D, as in Sardana et al.
- The Lemma 1 and Prop. 4 identities hold to 10⁻¹⁴.
- `verify_bias`: the closed forms for all four compute rules are right, including the target-rule slope γVar(ȳ)/(Var(ȳ)+Var(ω)) and the TFP-dispersion formula.
- The FOC-system bias 2E[ln w]/(α+β) follows from k̂ = k − E[ln w].

**Estimators.**
- Primal: the exact `sl._obj`/`sl._grad` Huber-LSE with L-BFGS-B, a box and 9 starts.
- Dual: the path OLS and the Huber frontier fit are correct, including ln M* = ln(C/6) − 2 ln N*.
- E2–E5: the within transformations of residual and Jacobian are the correct concentrated NLS. The lA = 0 normalization is harmless, because every reported object depends only on ln A − ln B.
- E6: the quasi-difference, instrument timing and level normalization are right.
- E7: the 2SLS, the first-stage F and the TFP residuals with generation effects added back are right.
- E8: the Heckman two-step computes the inverse Mills ratio only where selection is active.
- E9/E10: the FOC residual and its Jacobian are right.
- Selection rule, measurement error and ψ_D enter the DGP as the memo describes.

**Inference.**
- Pairs bootstrap with percentile CIs; no non-finite bootstrap draws in any replication.
- Coverage is computed over computable replications, and the computable shares are reported.

**Units.** Everything is simulated from the Besiroglu technology, so N, D and C are internally consistent: C = 6ND, and D is tokens processed. The one piece of real data is the calibration: the Epoch Chinchilla extraction with D = C/(6N), as in `sl.chinchilla_extraction`, with n = 240 after dropping the 5 worst points, which reproduces Besiroglu et al. (resid sd 0.0076).

**Sanity against the ledger (SYNTHESIS §2).** The truth values (a = 0.513, γ = 0.1783, σ* = 0.737) match. The Mertens-based sd(ω) = 0.25 conversion (γ ln 41/2.563 = 0.258) is correct.

**Citations.**
- All 23 keys in the memo exist in `lit/references.bib`.
- The two new keys are in `lit/bib/extra_m6_montecarlo.bib`, and their bibliographic data are correct:
  - heckman1979sample: Econometrica 47(1):153–161, doi 10.2307/1912352;
  - rotnitzky2000likelihood: Bernoulli 6(2):243–284, doi 10.2307/3318576.
- m7_theory cites rotnitzky2000likelihood from this file and does not duplicate it.

**`sl.py`.** Not edited. No bug found, which confirms the builder's statement.

**Numbers in the memo.** I checked every number in H1–H9, §2 and claims 1–13 against the regenerated CSVs. The ones that were wrong or needed a qualifier are items 3–18 above. All other numbers match the regenerated outputs to the printed precision.

## 5. Confidence in each headline claim (after fixes)

| Claim (memo numbering) | Confidence | Why |
|---|---|---|
| H1 / claims 1–2: on-path primal non-identification; RMSE(σ̂*) and RMSE(ln M̂*) fall steeply with s; IsoFLOP/factorial designs are much more precise at equal compute | **High** | Analytic (the Lemma 2 rank-one Hessian) and reproduced bit-for-bit. The truth-as-start and the box make the on-path numbers conservative. Exact RMSEs at s ≤ 0.05 depend on the box and the starts, so medians and MAE are reported. |
| H2 / claim 3: with κ free, near-optimal data leave σ* unidentified | **Medium-high** | Survives the denominator and grid fixes: flat in 88–97% of replications at s ≤ 0.2 and 65% at s = 0.3. Now correctly restricted to s ≤ 0.3, not s ≤ 1. The flat share depends strongly on s relative to the noise (38/65/90% at noise sd 0.005/0.0075/0.015 for s = 0.3) and on the design size, so "realistic error levels" must be read relative to the design. |
| H3 / claim 4: the dual estimator is precise on the path but biased under Kaplan-type beliefs | **High** | Mechanical: the bias equals the belief tilt (a: 0.73 − 0.513 = +0.217). |
| H4 / claim 5: inference fails on the path | **Medium** (after rewording) | The original "bootstrap covers 67%" was implementation-specific. The robust statement is: wrong (warm start, Wald) or uninformative (multi-start bootstrap: 92% coverage with σ* width 0.43). The multi-start result rests on 60 replications (MC s.e. ≈ 3.5 pp). |
| H5 / claim 6: Prop. 2 transmission formula | **High** | Matches the closed form within 4×10⁻⁴ (n = 10⁶). Also matches the industry flagship cell (0.118 vs 0.119). The n = 240 coverage rates are calibration-specific. |
| H6: Prop. 3 selection | **High** | Simple and exact; consistent with m7_theory. |
| H7 / claims 7–10: industry biases and which fixes work | **Medium** | Signs and rankings are robust within the DGP. Magnitudes scale with the assumed λ, sd(ω), compute dispersion and tiers. The ACF success rests on within-family instruments (now caveated). The IV row is now reported as medians. |
| H7 / claim 12: ML practice absorbs over half of TFP growth | **Medium-high** | The total (54–60% of 0.15 missed) is precise (MC s.e. 0.0005). The E1b decomposition supports the scale-attribution mechanism for about two-thirds of it. The rest depends on the DGP linking compute growth to calendar time. |
| H8 / claim 11: FOC-system M* contamination = 2E[ln w]/(α+β) | **High** | Algebraic given the DGP (simulated +1.24 to +1.28 vs 1.28). Its size scales with the assumed inference-demand distribution (mean ln w = 0.46). |
| H9 / claim 13: selection in the industry; Heckman interactions | **Medium-low** | The effects are small (−0.008), and the Heckman design needs observed inputs for unreleased models and an excluded shifter. Present it as an illustration of sign ambiguity, not as a recommended fix. |

## 6. Remaining concerns (not fixed; for the writers and the lead author)

1. **"Near-optimal" is relative to the design.** Whether near-optimal data identify σ* with κ free depends on s relative to the noise sd and on the number of runs (90 here). Quote the design whenever a threshold in s is quoted, and do not map s to real labs without a calibration. Released industry models are far off the path: for example Llama-3 8B at M ≈ 1,900 vs M* ≈ 20, a log deviation of about 4.5. The on-path problem concerns in-house ladders and compute-optimal sweeps.
2. **Truth-as-start and the parameter box** make the on-path primal results optimistic. This is conservative for the thesis, but on-path RMSEs of α, β and ln M* are box-dependent; cite medians and MAE alongside them.
3. **The chi-square calibration of the κ-free LR** is used near a non-identified ridge. Empirical coverage of the truth is 93–100% in every cell, so the practical consequence looks small, but a formal statement needs the singular-information results the memo already flags (rotnitzky2000likelihood).
4. **The industry calibration is assumed**: λ = 2, sd(ω) = 0.25, compute dispersion, tier structure, inference-demand dispersion and selection margins. Claim only signs and rankings until m4/m5 moments are used to recalibrate. The flagship transmission bias is exactly the Prop. 2 ratio at these variances, so it scales one-for-one with λVar(ω)/Var(c).
5. **ACF without within-family instruments was not run.** For example, lab-level lags with polynomial terms would reach identification. The paper should not claim that ACF works on flagship-only panels.
6. **Runtime is at the edge of the 45-minute budget (44.8 min)** on a machine at load 35–54. The added profile work (+4 grid points and a free-S fit) and the bootstrap-start stage cost about 10 min of wall time. `--skip-sim` rebuilds everything in about 10 s.
7. **Not re-checked:** the memo §5 aside that a warm start at the published Besiroglu values converges to A = 477.8, B = 2143. It does not enter any result.
8. **The bootstrap-start check uses fresh replications, not the main bootstrap's replications.** Its warm-start coverage (57%) differs from the main table’s 67% by about 1.3 MC s.e. of the difference, which is consistent with sampling noise.

## 7. Files changed by the review

- **Code:**
  - `mc_lib.py` (`kap_obj_free`, `fit_kappa_free`, profile denominator, `return_info`);
  - `design_a.py` (grid, `_interp_ci`, profile summary diagnostics, bootstrap-start stage);
  - `design_b.py` (E1b, docstring);
  - `outputs.py` (IV medians, notes, bootstrap-check CSV and § note, Fig. 2C label, E1b excluded from the figure);
  - `run.py` (bootstrap-start stage).
- **Outputs:** every module output was regenerated, and one is new: `output/tables/m6_montecarlo_designA_bootcheck.csv` plus `data/processed/m6_montecarlo/designA_bootcheck.parquet`.
- **Memo:** `output/memos/m6_montecarlo.md` corrected in place, with a revision note at the top.
- **Unchanged:** `sl.py`, `aer_style.py`, and the bib files.
