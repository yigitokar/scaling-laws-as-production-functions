# Writer notes: Conclusion (Section VII), Online Appendix C (Monte Carlo), Online Appendix D (additional results)

Writer: conclusion writer, 2026-09-24.

## Files written

- `paper/sections/conclusion.tex`: Section VII, `\label{sec:concl}`. About 1,000 words, against a target of about 900.
- `paper/sections/appendix_mc.tex`: Online Appendix C, `\label{app:mc}`.
  - Subsections: `app:mc-designA`, `app:mc-designB`, `app:mc-formulas`.
  - Tables: `tab:app-mc-designs`, `tab:app-mc-industry`, `tab:app-mc-transmission`.
  - Figures: `fig:app-mc-profile` (m6_montecarlo_profile_ci) and `fig:app-mc-industry` (m6_montecarlo_industry_bias).
- `paper/sections/appendix_additional.tex`: Online Appendix D, `\label{app:additional}`.
  - Subsections:
    - `app:add-chinchilla`
    - `app:add-tech`
    - `app:add-wedge`
    - `app:add-obs`
    - `app:add-progress`
    - `app:add-flexible`
    - `app:add-claims`
  - 19 tables `tab:app-*`.
  - 3 figures:
    - `fig:app-farseer-sigma` (m2_farseer_local_sigma)
    - `fig:app-tfp-families` (m4_observational_tfp_families)
    - `fig:app-allocative` (m5_progress_fig6_allocative)
- `paper/tables/appC_{designs,industry,transmission}.tex` and 19 `paper/tables/appD_*.tex`:
  - selection, duality, labs;
  - robust, neutrality, farseer;
  - family, rivals, aggregate, homothetic;
  - lalonde_robust, tfp;
  - replication, attenuation, ceg_sahal;
  - steplaw, sfa, demand;
  - claims.
- `code/paper/make_appendix_cd_tables.py`: generates **all three C tables** and four D tables (selection, homothetic, lalonde_robust, attenuation) directly from `output/tables/*.csv`.
  - The other D tables are hand-condensed from the reviewer-verified `output/tables/*.tex`. I spot-checked every quoted number in them against the CSVs.
  - **Warning:** rerunning the script overwrites only the seven generated files.

Compile status:
- `test_section.sh conclusion`, `appendix_mc` and `appendix_additional` each compile with no errors and no overfull boxes.
- A joint scratch build (conclusion + `\appendix` + C + D) numbers everything correctly: Tables C1–C3 and D1–D19, Figures C1–C2 and D1–D3. It has no errors, overfull boxes or float overflow.

## Numbers used (value → source)

### Conclusion

| Number | Source |
|---|---|
| median w = 3.19; inference/training = w − 1 ≈ 2.2; "factor of about three" (lifetime/training = w) | `data/processed/m3_wedge/headline.json` (sampleB_median_w 3.186) |
| median ŵ across technologies 1.9–4.0 | m3 memo H1 (Hoffmann 1.91, Gadre RW 4.03); `m3_wedge_sampleB_by_tech.csv` |
| σ 0.69–0.71 on Farseer (three q-free estimators) | `m2_farseer_local_sigma_summary.csv` (median 0.690, Eq. 3 0.706); `m2_table3_technology.csv` (σ*_q 0.710) |
| Chinchilla-form σ* 0.74–0.83 | `m2_table3_technology.csv` (Huber 0.735–0.828) |
| κ = 1 rejected in all 7 sweeps; relaxing it lowers σ* by 0.04–0.28 | `m2_table3_technology.csv` (σ* − σ*_q: Chinchilla 0.036 … OLMo 0.281) |
| γ about 0.09–0.18 under the Chinchilla form | `m2_table3_technology.csv` (DataDecide 0.090, OLMo 0.099 … Muennighoff 0.180) |
| M*(10²¹) 3–60 across sweeps | `m2_table3_technology.csv` (3.35–59.6; excludes DataDecide, whose levels are descriptive) |
| Chinchilla cluster CI for M*(5.76e23) [3.2, 126] | `m1_chinchilla_horse_race.csv` (lo/hi_cluster_Mstar_5.76e+23 = 3.17, 125.8) |
| IsoFLOP ±16× RMSE(σ*) 0.004 vs on-path (s = 0) 0.28 | `m6_montecarlo_designA_summary.csv` (0.0036; 0.2819) |
| Cluster/pairs SE ratio 1.5–2.2 | `m1_chinchilla_horse_race.csv` (α 0.0235/0.0156; β 0.0461/0.0212) |
| Counting conventions move a by 0.04–0.17 | Chinchilla +0.042 (m8 memo, `m8_measurement_chinchilla_nonembed.csv`); Farseer −0.115 (`m2_robustness.tex`); Porian 0.118–0.166 (m8 memo, `m8_measurement_porian_counting.csv`) |
| Condition number about 2,000 → 62 | `m1_chinchilla_horse_race.csv` (cond_raw 2003, norm 61.9) |
| 84% of residuals in Huber's linear region | `m1_chinchilla_horse_race.csv` (share_linear_huber 0.8375) |
| 1/γ ≈ 5.6 | 1/0.1785 (m1 refit γ) |
| three-quarters of Sample B beyond the largest design budget | `headline.json` (sampleB_share_Cx_chin 0.746) |

### Appendix C

All numbers are from the m6 CSVs:
- `m6_montecarlo_designA_summary.csv`, `_designA_profile.csv`, `_designA_diagnostics.csv`, `_designA_bootcheck.csv`;
- `m6_montecarlo_industry_summary.csv`, `_industry_meta.csv`;
- `m6_montecarlo_transmission_check.csv`, `_selection_check.csv`, `_identities_check.csv`;
- `data/processed/m6_montecarlo/noise_calibration.csv`.

| Result | Value |
|---|---|
| Truth (Besiroglu parameters) | a 0.513, γ 0.178, σ* 0.737, ln M*(1e24) 2.898 |
| Noise sd | 0.0075 (resid 0.00756; MAD sd 0.0049) |
| Corner share at s = 0 | 51% |
| Median bias at s = 0 | −0.297 |
| s = 0.02 | MAE 0.037; 5–95 range [0.34, 0.77] |
| RMSE ln M* | 24.0 / 4.49 / 1.97 / 0.59 at s = 0 / 0.1 / 0.3 / 1; 0.16 under IsoFLOP |
| s = 0.3 range of ln M* | [−0.40, 6.34] |
| Flat profile shares | 88% (on-path); 97 / 91 / 93 / 89 / 65% (s = 0, 0.05, 0.1, 0.2, 0.3); 77% (Kaplan) |
| Noise sensitivity (s = 0.3, flat share) | 38 / 65 / 90% at noise sd 0.005 / 0.0075 / 0.015 |
| κ-free CI widths | s = 1: 0.104, median CI [0.69, 0.79]; IsoFLOP 0.016; factorial 0.019; ±4× 0.047 |
| Dual estimator on the path, RMSE | a 0.0005, σ* 0.011, ln M* 0.009 |
| Dual estimator under Kaplan beliefs, bias | a +0.217, σ* −0.088, ln M* −3.92 |
| Bootstrap coverage | 67% (warm start); multi-start 92% with width 0.43 (bootcheck) |
| Industry: pooled NLS bias in γ | +0.023 / −0.051 / +0.018 |
| Industry: flagships only | +0.119 vs formula 0.118 (m6 review) |
| Industry: lab FE, share of bias removed | 64% / 80% (m6 memo; recomputed ≈ 65% / 78%) |
| Within-family ME bias in ln M* | −0.33 (lab FE −0.28) |
| ACF | −0.026 under the target rule; RMSE(a) 0.12–0.13 vs 0.026 |
| FOC system bias in ln M* | +1.25 to +1.28; closed form 1.28 |
| T proxy | −0.02 to −0.03 |
| IV, median bias (MAE) | −0.002 to +0.005 (0.027–0.031); under selection −0.022 / −0.033 |
| ML practice | TFP bias −0.081 to −0.090; ln M* +0.56 to +0.68 |
| E1b (E at truth) | TFP −0.054 to −0.065 (36–43%); γ +0.050 to +0.059 |
| Two-thirds / one-third split | m6 memo (review decomposition) |
| Selection | −0.008 / +0.011; Heckman halves; predetermined +0.011 → +0.018 |
| Transmission check | within 4e-4 (n = 10⁶); coverage 95 / 53 / ≤4 / 16%; residual sd 0.229 vs 0.255 |
| Slope under selection | 0.178 → 0.112 / 0.082 / 0.057; net bias +0.100 → −0.006 / −0.046 |

### Appendix D

Tables use the numbers in their source `.tex`/CSV files. The numbers quoted in the prose, with their sources:

- **Selection.**
  - β 0.453 → 0.347 and α 0.345–0.349 (`m1_chinchilla_selection_k.csv`).
  - w(70B) 0.95–1.06 under outlier-removing rules; 1.24 or more when the outliers are kept; 0.78 when the 1e19 budget is dropped (`selection_k.csv`, `selection_rules.csv`).
  - Hausman–Wise β 0.406 = Gaussian β 0.406.
- **Duality.**
  - Refit and Besiroglu p ≥ 0.12.
  - Hoffmann F-test p 0.040; calibrated p 0.060; path level about 12% (`m1_chinchilla_duality_bootcal.csv`, `_classicalF.csv`, memo H6).
  - Frontier: stratified calibrated p 0.003–0.037; wild p ≥ 0.15; gap 0.005–0.008 nats (`duality_objects.csv`, `duality_bootcal.csv`).
- **Labs.**
  - a 0.527 vs 0.463; t = −2.1.
  - Marin p 0.33–0.54 (`m1_chinchilla_labs.tex`).
  - Three-quarters of the Marin gap is grid bias (m1 memo [Rev] and `labs_design_decomp.csv`).
- **m2 robustness.** Farseer a 0.526 → 0.411, σ* 0.772 → 0.724; Muennighoff a 0.535 → 0.360, M* 60 → 139; OLMo 0.825 → 0.763 (`m2_robustness.tex`).
- **Neutrality.**
  - Gadre p 0.90, 96.8%.
  - DataDecide 98.1% → 99.7%.
  - Tilt 0.22–0.26, M* 2.9–3.4×, wedge 1.25–1.29× (`m2_neutrality.tex`, `m2_neutrality_magnitudes.csv`).
- **Farseer.**
  - Out-of-sample 8–10× (24.75 or 21.95 vs 2.58; `m2_farseer_forms.csv`).
  - Local σ 0.690 [0.679, 0.699], Eq. 3 0.706, q-family 0.710; bandwidth variants 0.694 / 0.707 (m2 memo); Eq. 3 M* 32/25/28/43/82 (`m2_farseer_eq3_Mstar.csv`).
- **Families.**
  - 95% of 105 siblings (`m3_wedge_family_members.csv`).
  - Flagship medians 1.14 / 3.14; 50% / 92% (`m3_wedge_flagships_by_period.csv`).
  - Over-identification p = 0.047 / 0.12 (`m3_wedge_family.tex`).
- **Rivals.** 26% vs 9%; +0.11 (0.06) (`m3_wedge_rival_regressions.csv`).
- **Aggregate.** Table from `m3_wedge_aggregate.csv`; by technology 0.67–3.26 (`m3_wedge_aggregate_by_tech.csv`).
- **LaLonde robustness.**
  - ARC −0.098 (0.127), Winogrande −0.160 (0.102); 18–31% attenuation.
  - Clean sample +0.045 (0.023), wild p 0.10 (`m4_*_arc/wino/clean.csv`, `_clean_wildboot.csv`).
- **TFP.**
  - 23.7 [9.1, 197], 18.3, 7.0, 11.1 vs 41; Phi and SmolLM rank 1st and 3rd.
  - 3–5× in logs (ln 7/ln 1.92 = 2.98; ln 23.7/ln 1.92 = 4.85).
  - Loss units 1.41–1.76 (`m4_observational_tfp_table.tex`).
- **Ho replication.**
  - 8.44 / 8.68 → 6.08 [3.05, 22.67]; objective down 2.1%, MSE down 0.27%.
  - Estimator range 6.1–10.2; profile CI [4.1, 40.5] (`m5_progress_replication.tex`, `data/processed/m5_progress/dmr_summary.json`).
- **Attenuation.**
  - γ 0.178 → 0.053; cross-lab γ 0.021–0.027 (`m5_progress_table7_panelA.csv`, rows A1–A5); gap 2–2.5×.
  - Monte Carlo 11.1 vs 12; authors' code 4.6; effective data 25.7 (12.7) (`m5_progress_attenuation_*.csv`).
- **CEG and Sahal.** Δγ 0.008 [−0.012, 0.034]; inflation 1.15 [1.10, 1.21] and 1.44 [1.24, 1.78] (`m5_progress_ceg.csv`, `m5_progress_sahal.csv`).
- **Allocative.**
  - 1.05 / 1.91 / 2.41 / 1.58; share 2–40%; registry range 0.55–4.3; Meta 0.74 (`m5_progress_table7_panelB.csv`).
  - Kaplan counterfactual 5.2 at 3.8e25 and 11.7 at 5e26 (`m5_progress_kaplan_counterfactual.csv`).
- **Step Law.**
  - a 0.76–0.45 over E.
  - Policy shifts −0.041 to +0.076, within 0.7 SE.
  - One random configuration: σ* −0.051 (−3.7 SE), a −0.3 SE.
  - γ_D −0.23 to −0.26; implied bias −0.012.
  - Conditioning closes 26–43% (m8 memo §1.5, `m8_measurement_steplaw.tex`, `_sfa.tex`, `_demand.tex`).
- **Claims register.** 61 checks, all pass (`m7_theory_claims.csv`).

## Claims with caveats (as written)

- **σ.** The conclusion gives σ as a range, 0.5–0.8 with a central value near 0.7. The Chinchilla-form values are presented as conditional on a rejected restriction. The paper never says that σ is more stable than a.
  - The m2 robustness note originally had a sentence comparing how much σ* and a move; I removed it (plan "Do not claim").
  - D6 states explicitly that one random configuration biases σ* and that the data "give no reason to regard σ* as the more robust object" (m8 review M1/M4).
- **Wedge levels.** Revealed demand is described as "ranks and ranges, not precise levels". Aggregate multiples are called illustrative and not validated. Validation is called correlational. "Markup" is never used.
- **IO remedies.** The Design B text states explicitly that the ACF success borrows within-family instruments and "does not show that proxy or timing estimators work on flagship-only panels". The industry calibration is illustrative; only signs and rankings should be read from it.
- **DMR sign, CEG and the bracket condition.** The corrected statements are used throughout (m7): the sign of the bias is identified; CEG is constant iff E and γ are equal; the bracket holds iff π₁ is in the stated range.
- **LaLonde benchmark.** The conclusion says "detects no bias ... for the main output"; D4 reports the ARC and Winogrande point estimates (18–31% attenuation, not significant).
- **Hoffmann duality.** Called "marginally inconsistent (p = 0.040 / 0.060)", not rejected at 0.0004.
- **DataDecide tilt.** Not presented as independent evidence for data-augmenting quality (m2 review item 3).

## Cross-references assumed to exist elsewhere

- **Sections:** `sec:framework`, `sec:ident`, `sec:tech`, `sec:wedge`, `sec:obs`.
- **Lemma and propositions:** `lemma:sigma`, `prop:duality`, `prop:fdep`, `prop:info`, `prop:dmr`, `prop:transmission`, `prop:wedge`, `prop:pi`.
- **Tables:** `tab:technology`, `tab:lalonde`, `tab:measurement`.
- **Figures:** `fig:designs`.
- **Appendix:** `app:proofs`.

## Placeholders (m9)

Both are in `conclusion.tex`, in red:
1. End of the opening paragraph: one sentence on what our controlled two-lab experiment adds (σ by lab with κ free; the neutrality verdict for FineWeb-Edu vs FineWeb).
2. Limitations, extrapolation caveat: the direction of the extrapolation bias in ŵ from the m9 high-M check.

## New bib keys

None. All 41 keys used are already in `paper/references.bib`, including the extra_* keys already merged: hausman1977social, czech2026llama3isoflop, marin2026ladders, heckman1979sample, caudill1995frontier, milgrom1996lechatelier.

## Open issues for the integrator

1. **Label collisions (verified 2026-09-24).** `appendix_proofs.tex` already defines `prop:info`, `prop:dmr`, `prop:transmission`, `prop:wedge` and `prop:pi` for results A2, A3, A5, A8 and A9. `identification.tex` and `framework.tex` define the same labels, so each is now defined twice. The plan assigns the same labels to main-text Propositions 4, 5, 6 and 2. My files reference them intending the **main-text** propositions. Rename the appendix labels (e.g. `prop:A-info`), or the references resolve to whichever is defined last.
2. **Two T_C intervals for the same baseline.** Ho baseline converged T_C has CI [3.0, 26.5] in Table D14 (attenuation; bootstrap draws start at zero only) and [3.05, 22.67] in Table D13 (replication; two starting values per draw). Both come from the m5 outputs. The D14 note says "each started at zero". The main-text progress section should quote [3.0, 22.7] (m5 memo).
3. **Possible duplicate aggregate table.** Table D9 (aggregate) reproduces `m3_wedge_aggregate`. If the wedge writer puts the same table in the main text, drop D9 and point Section D3's prose to the main-text table.
4. **Conclusion length.** About 1,000 words against a target of about 900. The five design recommendations plus the two numerical points were kept as plan-required content.
5. **Appendix layout settings.**
   - `appendix_mc.tex` and `appendix_additional.tex` set `\emergencystretch=2em` right after their `\section`. The online appendix is last in `main.tex`, so this affects only C and D (and anything after).
   - Appendix D uses `\clearpage` after each subsection to flush its 22 floats. Removing these risks "too many unprocessed floats".
6. **Notes lead-in.** Table notes use AEA.cls's default lead-in (`Note:`), with separate `[Source]` notes. Figure notes include the source in the text.
7. **Claims register (Table D19).** It was restructured from `m7_theory_claims.tex`: each statement is mapped to its main-text result, and the internal model_spec/SYNTHESIS source IDs were dropped. The "Corrected" status refers to "our working drafts or common readings of the literature". If referees could find that odd, drop the Status column.
8. **Figure D1.** `m2_farseer_local_sigma` includes a pointwise band that ignores smoothing bias. This is stated in its note.
9. **Unverifiable numbers.** The m6 memo's Ê bias (0.055–0.061) is not in any CSV. I did not quote it.
10. **Labels other writers depend on.** `observational.tex` already references `tab:app-lalonde-robust`, `tab:app-tfp`, `tab:app-mc-industry`, `tab:app-ho-replication`, `tab:app-attenuation` and `tab:app-ceg-sahal`, so keep these labels. A scan of all sections and tables on 2026-09-24 found no undefined `\ref`.
