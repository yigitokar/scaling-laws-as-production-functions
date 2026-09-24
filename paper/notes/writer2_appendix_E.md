# Writer notes: Online Appendix E (v2), "Observational Production Functions and Algorithmic Progress"

Writer: appendix_E, 2026-09-24. Replaces the v1 Section VI (`sections/v1/observational.tex`) and the observational and
algorithmic-progress parts of v1 Appendix D (`app:add-obs`, `app:add-progress`).

## Files

| File | What |
|---|---|
| `paper/sections/appendix_observational.tex` | Appendix E text, about 5,900 words. Test build: 23 pages including 9 tables and 3 figures. |
| `paper/tables/appE_matched.tex` | ra4 Table E1 (design-matched comparison). Label `tab:app-matched`. Note edited: "original Table 8" → "an earlier version of this paper"; "(formerly OP-style selection)" dropped. |
| `paper/tables/appE_inference.tex` | ra4 Table E3 (few-cluster inference, family FE). Label `tab:app-inference`. New caption; "Table 8" row labels → "earlier version". |
| `paper/tables/appE_reliability.tex` | ra4 Table E4 (reliability of ln C). Label `tab:app-reliability`. "Section VI.A set" → "Epoch `Hardware'/`Reported'"; "(Table 8)" dropped. |
| `paper/tables/appE_compute.tex` | ra4 Table E2 (compute-only estimators). Label `tab:app-compute`. Row "Table 8" → "$n-1$". **Note fixed:** the brackets are the CR1-studentized WCR with benchmark draws, not the conservative p of Table E1 as the ra4 note said. The reverse-regression identity is stated as an identity ("By construction"), not as "with no transmission bias". |
| `paper/tables/appE_tfp.tex` | ra4 Table E7 (TFP in output and input units). Label `tab:app-tfp-units`. Unchanged. |
| `paper/tables/appE_progress.tex` | ra4 Table E5 (Ho replication and ridge). Label `tab:app-progress`. T_C → τ_C, φ → g_N/g_C, Besiroglu named in the "experimental exponents" row; bimodality is disclosed in the note. |
| `paper/tables/appE_attenuation.tex` | v1 `appD_attenuation.tex` with τ_C. Label **renamed** `tab:app-obs-attenuation`. **Number fixed:** the Panel B E = 0 Huber interval is now [0.051, 0.055]; v1 had the stale [0.052, 0.054], and the post-rebuild CSV gives 0.0525 [0.0510, 0.0548]. |
| `paper/tables/appE_ceg_sahal.tex` | v1 `appD_ceg_sahal.tex`. Label **renamed** `tab:app-obs-ceg-sahal`. `prop:dmr` → `cor:ceg`. The slope "$\hat\theta$" → "Compute slope" (θ is reserved for the benchmark elasticities). |
| `paper/tables/appE_allocative.tex` | ra4 Table E6 (allocative gains). Label `tab:app-allocative`. Group labels rewritten for v2 (see below). `prop:A-wedge`(v), which does not exist, → (ii) and (iv). "Proposition 2" → `\ref{prop:wedge}`. The κ-free Chinchilla row is marked "(reference)". |
| `code/paper/make_appendix_e_ridge.py` → `output/figures/appE_ridge.{pdf,png}` | The m5 ridge figure redrawn with v2 notation: τ_C instead of T_C, g_N/g_C instead of φ (φ is the serving-cost elasticity in Prop. 2). Panel (c) uses ra4's refined profile (`ra4_obsfix_phi_profile.csv`) and shades the interval [−0.436, 3.367] read from `progress_headline.json`. No estimation is redone; the script only reads processed files. |

Figures used: `ra4_obsfix_fig_matched` (Fig. E1, `fig:app-obs-matched`), `appE_ridge` (Fig. E2, `fig:app-obs-ridge`), and
`ra4_obsfix_fig_allocative` (Fig. E3, `fig:app-obs-allocative`).

Table order in the build: E1 matched, E2 inference, E3 reliability, E4 compute-only, E5 TFP, E6 progress,
E7 attenuation, E8 CEG/Sahal, E9 allocative. These differ from the ra4 memo's E1–E7 names; the labels are what matter.

Section labels: `app:obs` (section), `app:obs-theory`, `app:obs-matched`, `app:obs-compute`, `app:obs-tfp`,
`app:obs-progress`, `app:obs-allocative`.

Compile: `bash code/paper/test_section.sh appendix_observational appendix_proofs` gives 0 errors, 0 undefined citations
and 0 overfull boxes in Appendix E (the 4 overfull boxes are in Appendix A). The only undefined references point to
main-text labels and to the section labels of Appendices B, C and D, which are absent from the test build. Every page
image was inspected.

## Numbers used, with sources

All ra4 numbers were read from `output/tables/ra4_obsfix_*.csv` or `data/processed/ra4_obsfix/*.json`; the ra4 memo
was not used alone.

| Claim in text | Value | Source |
|---|---|---|
| Panel | 128 models, 38 families, 22 developers, 2021-03 to 2024-07 | `m4_observational_design_audit.csv` |
| Median D/N; 90th percentile | 213; 1,715 | design_audit; recomputed from `obs_panel.csv` (213.4; 1715.2) |
| F for quadratic terms, Gadre N ≥ 0.1B | 70 | `m4_observational_curvature.csv` (70.28). The OLMo value of 352 includes OLMo-2, so it is not quoted. |
| Local θ_D at N = 1B, D/N 20 → 500, strict surface | 0.57 → 0.07 | Writer-computed from `ra4_obsfix_surfaces.csv` (strict, HellaSwag, OLS coefficients; ∂/∂d = b_d + 2b_dd d′ + b_nd n′). The same formula reproduces m4's legacy values 0.58/0.09. |
| Observational θ_D (global) | 0.19–0.26 | `m4_observational_table6b_global_main_long.csv` (0.186–0.257) |
| Experimental global θ_D | OLMo ladder 0.44; Gadre 0.51 | `m4_observational_exp_benchmarks.csv` (0.441; 0.511) |
| Strict surface | 86 runs, R² 0.994 | `ra4_obsfix_surfaces.csv` (0.9937) |
| Strict support | 20 models, 9 families (5 with ≥ 2 models: 16 models), 6 developers, 2021-03 to 2023-09, N ≤ 6.7B, D ≤ 627B | `matched_headline.json` |
| Legacy support | 57 models, 30 families (15 with ≥ 2 models: 42 models; 15 singletons), 19 developers, N ≤ 8.8B | `matched_headline.json` |
| Largest designed D without OLMo-2 | 921B (Gadre 1.4B); 634B above 3.8B | ra4 review §3 |
| G* | 2.5–6.7 of 19; 1.2–3.8 of 6 | `matched_long.csv` G_star |
| OLS θ_C, 57 models | 0.328 vs 0.329; bias −0.001 (CV3 0.041); conservative p 0.98 | matched_long row 107 |
| Six estimators, 57 models | biases −0.029 to +0.037; conservative p ≥ 0.36 | matched_long rows 105–122 |
| Strict surface on 57 | +0.014 (0.045) | row 233 |
| Strict support, five estimators | −0.024 to +0.037 | rows 0–14 |
| 95% interval with t(18) | [−0.088, 0.086]; ±26% of 0.329 | −0.0011 ± 2.101 × 0.0413 |
| MDE at 80% power | 0.12 (37%); 0.27 (62%) on 20 | mde80_cv3_tG1: 0.1225; 0.2658 |
| Runs at the floor | ARC-C 52, Winogrande 16, HellaSwag 2 of 86 | surfaces n_censored |
| ARC-C, OLS surface | −0.098 | matched_long row 161 (−0.0976) |
| ARC-C, Tobit | +0.036 (0.069), p 0.61 | row 179 |
| Winogrande, Tobit | −0.150 (0.077), p 0.068–0.084 (57); −0.179, conservative p 0.53 (20) | rows 215, 92 |
| Composite | −0.018 (0.050) | row 143 |
| Family FE, 57 models | θ_N +0.203 (0.059), p 0.048; θ_D −0.262 (0.106), p 0.027 | rows 114–115 |
| Leave-one-developer-out | signs unchanged | `inference_reconcile.csv` (θ_N [0.157, 0.224]; θ_D [−0.339, −0.193]) |
| Within-family sd of ln D vs ln N (128) | 0.42 vs 1.29 | design_audit |
| θ_D leverage | Cerebras-GPT 37% + Phi 28% = 65% | `leverage_familyFE.csv` |
| Strict family FE | +0.185, −0.227; conservative p 0.14, 0.15 | rows 9–10 |
| Reconciliation | (n−1)/(n−k) 2.24 vs 1.04; s.e. × 1.47; benchmark s.e. 0.020; p 0.01–0.03 (θ_D), 0.003–0.05 (θ_N) | `inference_reconcile.csv` |
| θ_N vs θ_D | 0.42 vs 0.27 (benchmark, 57); θ_N − θ_D 0.21 (0.08) on 128 | matched_long rows 105–106; `m4_observational_table6b_global_main_long.csv` (0.2130, 0.0791; traced by the ra4 review) |
| DataDecide regimes | +0.044; −0.18 (40%); dispersion about 5× | m4 memo item 5 (R4 PASS) |
| Industry Monte Carlo | +13%, −29%, +66%; within developer–generation unbiased | `m6_montecarlo_industry_summary.csv` (E2_pooled_nls: +0.0228, −0.0511, +0.1185 of 0.1783; E5_labgen_fe ≈ 0) |
| Hardware-time comparison | 27 models (12 families); sd 0.29 [0.20, 0.41] | `reliability_comparisons.csv` |
| Reliability, 57 models | 0.980 / 0.955 (0.961 / 0.912 at the upper bound) | `reliability_summary.csv` |
| EIV shifts | +0.007 pooled; +0.021 within family (0.042 at the upper bound) | `compute_only.csv` (0.3347 − 0.3280; 0.4614 − 0.4407; 0.4407/0.9121 − 0.4407) |
| Earlier version's EIV row | 0.858 vs 0.409; corrected +0.052 (0.049), p 0.30 | `compute_only.csv` |
| Reverse regression | 0.560 = 0.328/0.585; 0.580 = 0.441/0.760; experimental 0.352 | `compute_only.csv` |
| Frontier IV | F 0.90 (0.01 with developer FE); AR unbounded; on experimental outputs 0.012 | `compute_only.csv` |
| Own-hardware IV | n = 10, 7 developers, F 14.3, AR [−1.50, 0.34], 3 values | `compute_only.csv` |
| Export-control IV | 90 models, F 10.6, 0.90 (0.25), AR [0.49, 2.23]; 1.7–2.3× OLS/FE (0.40–0.54) | `m4_observational_iv_diagnostics_overlap.csv`, `table6b_global_overlap_long.csv` |
| Dynamic panel | six cells with three generations | m4 memo item 6 |
| Notability-propensity control | −0.015, p 0.80 | matched_long row 122 |
| TFP, odds | 1.9–3.0 | `ra4_obsfix_tfp_units.csv` (1.90–3.02, HellaSwag rows) |
| TFP, loss units | 1.3–1.8 | tfp_units (1.30–1.76) |
| TFP, compute units | 7–24 (θ_C 0.33); 4–13 (0.43); within developer 11.1 [4.1, 18.1] | tfp_units |
| Phi 1st and SmolLM 3rd of 38 (within netting) | | Writer-recomputed from `obs_panel.csv` (family-FE regression on n and d). R4 had this as UNTRACED. |
| Ho et al. | 231 obs., 144 papers; objective 0.0518129 → 0.0507227; params within 4e-7; median 8.44; point 8.68; 18 iterations; 300/300 starts below; 6.08 [3.0, 22.7]; MSE −0.27%; NLS 10.2 [4.1, 27.6] | `m5_progress_replication.csv`, `progress_headline.json`, Table E5 |
| Loss shares; a | 0.55/0.45; 0.37 | m5 memo claim 3 (s̄ = 0.545; a = 0.040/0.108) |
| Bootstrap correlations | −0.73 (Ho code, cluster), −0.82 (converged, cluster) | `progress_headline.json` |
| Profile interval for τ_C | [4.1, 40.5] | m5 `dmr_summary.json` via Table E5 |
| g_N/g_C interval; τ_C range | [−0.44, 3.37]; 5.3–39.6 | `progress_headline.json` (−0.436, 3.367; 5.35–39.57) |
| Second local minimum | near −0.28 | `ra4_obsfix_phi_profile.csv` (−0.275, LR 2.95) |
| Neutrality | p 0.29 / 0.94; 9.2 [6.3, 14.3] (NLS), 11.6 [6.6, 17.6] | `m5_progress_table7_panelA.csv`; Table E5 |
| Neutrality misspecified | 4.2–4.4 vs 12 | `m5_progress_attenuation_mc.csv` S1h |
| Whitfill | 76.5 months | panelA |
| Imposed exponents | MSE × 1.3–4.8 across 14 technologies | panelA (0.0605/0.0453 to 0.2187/0.0453) |
| E = 0 on Chinchilla | γ 0.178 → 0.053; total-loss elasticity 0.054 | `m5_progress_attenuation_panelB.csv` (0.0525; 0.0543) |
| Common E | τ_C 10.2 → 12.4; γ̂ doubles | `m5_progress_E_profile.csv` (R4 PASS) |
| Epochs; effective data | 62–140; ×1.5–4 | m5 memo claim 7 (R4 PASS) |
| CEG / Sahal | Δγ 0.008 [−0.012, 0.034]; 1.15 and 1.44; s_A 0.37 → 1.58; corr 0.62 | `m5_progress_ceg.csv`, `m5_progress_sahal.csv` |
| Allocative sample | 7 and 92 models; median D/N 1.7 → 92 | `ra4_obsfix_allocative.csv` |
| Truncated gains | 1.97 (κ-free reference), 1.68 [1.21, 2.44] (Besiroglu), 1.14–4.95 across 14 technologies | allocative.csv |
| Share of the era-1 gap closed | 0.80–0.94 | allocative.csv |
| Untruncated | 0.55–4.31 | allocative.csv |
| Marin | 3.52–4.95; exponents about twice Chinchilla's (0.64–0.69) | allocative.csv; panelA registry rows |
| Top five per year; frontier flag | 1.31; 1.54 | allocative.csv |
| Share of the Ho-rate gain | 6–73% (31% reference); 8.87× over 2.28 years; denominator 6.4–22.5 | allocative.csv; m5 memo claim 9 |
| Kaplan counterfactual (Besiroglu) | 1.6 / 5.2 / 11.7 | `ra4_obsfix_kaplan_counterfactual.csv` |
| Gopher | about twice the compute (Φ = ln 2.01) | Prop. A4 (`prop:farrell`) |

## Referee comments addressed

- **R1 8(f), minor 49 (DMR sign).** The "Technical change" paragraph states $\mathcal B=b_1g_D-a_1g_N$ (βg_D − αg_N when κ = 1) and gives the counterexample. g_N and g_D are identified only by functional form. The result is called methodological and is not taken to vintage data, with the reasons given. No "corrects the statement" sentence.
- **R1 8(i), R4.** "LaLonde" is gone. The exercise is called "a validation across populations" (Todd–Wolpin; Hotz–Imbens–Mortimer). LaLonde is cited only for the contrast and Dehejia–Wahba for common-support trimming.
- **R1 8(j).** Goldberger (1981) and Hausman–Wise are cited for truncation attenuation; the interaction with transmission is presented as the new part.
- **R1 9(e), R4 minor 36.** Table E2 (inference) and the reconciliation paragraph: the nested-FE degrees-of-freedom factor, not benchmark uncertainty. CV3 standard errors and conservative p-values are used throughout.
- **R1 10(a).** "Failure to reject in a low-power joint test"; the joint null is stated; the interval ±26%; MDE 37% (62% strict); N ≤ 8.8B. The family-FE split is demoted to "a within-family disagreement driven by two families, not a finding".
- **R1 10(b).** OLMo-2 is excluded from the benchmark; the strict support (20 models) is reported alongside, with the loss of support.
- **R1 10(c), minor 48.** Renamed "notability-propensity control (selection on observables)". The state variable, flexible input and timing are set out. Both proxy failures are given: the mix, and compute as an LP-type proxy (Prop. A7(iii)). Lemma A3(v) is cited for the target-rule co-movement.
- **R1 10(d).** The allocative gain is truncated at w = 1. Caveats: the sign is near-mechanical; w < 1 can come from a data constraint or data-augmenting productivity; the intervals hold the technology fixed except for Besiroglu.
- **R1 10(e), R4 M11.** Both unit systems are reported; the comparison with manufacturing is "a choice of cardinalization, not a finding"; contamination, teacher compute and θ_C uncertainty are covered.
- **R1 minor 18.** New paragraph on transmission with heterogeneous wedges: loads on the mix only if wedges covary with ω at given compute; otherwise equal shifts; with an equal-weight index, the Hall term.
- **R1 minor 42.** Cluster and family counts are given in the text for both supports.
- **R1 minor 43.** The Hall-type bias paragraph cites `cor:hall`.
- **R1 minor 44, R2 minor 25, R4 M7.** Only hardware-time records are independent. The reliabilities are recomputed, and the earlier EIV row is explained as two errors.
- **R1 minor 45, R4 minor 37–38.** Every IV carries an Anderson–Rubin set, including the export-control IV. The instrument applied to the experimental outputs returns 0.012.
- **R1 minor 46.** The ridge and the neutrality dependence come first; the stopping-rule narrative is one paragraph plus Table E6.
- **R1 minor 50.** Dynamic-panel moments (Prop. A5(v)) are stated as unusable for lack of consecutive generations.
- **R2 minor 24.** Chance handling (floor counts) and the Tobit benchmark are described; harness non-harmonization and classifier-filtered or contaminated corpora are stated as untested parts of the null.
- **R3 M10.** The material moves to the Online Appendix. The Ho finding is condensed and framed as numerical optimization, "not the order of magnitude of progress".
- **R4 M4(d), minor 41.** Table E9's group labels say where each technology comes from.
- **R4 M4(f).** g_N/g_C ∈ [−0.44, 3.37]; τ_C 5.3–39.6 over the interpolated interval; the earlier grid-point range is explained in the note.
- **R4 M4(c).** FOC-system bias: not quoted here (it belongs to Appendix C).
- **R4 M8(b).** Gundlach et al. are credited with defining the CEG and with the ~10× figure; our counterfactual "agrees"; our contribution is the realized-versus-counterfactual contrast.
- **R4 minor 28, 39.** τ_C replaces T_C, and g_N/g_C replaces φ (φ is the serving-cost elasticity in Prop. 2). The CEG/Sahal table no longer uses θ̂ for the slope.

## Open issues and notes for the integrator

1. **Labels other writers may need.** `app:obs`, `app:obs-matched`, `app:obs-progress`, `app:obs-allocative`, `tab:app-matched`, `tab:app-allocative`, `fig:app-obs-ridge`. The v1 labels `sec:obs`, `tab:lalonde`, `tab:progress`, `fig:lalonde`, `fig:ridge`, `tab:app-lalonde-robust`, `tab:app-tfp`, `tab:app-ho-replication`, `fig:app-allocative` and `fig:app-tfp-families` no longer exist. Two v1 labels were renamed: `tab:app-attenuation` → `tab:app-obs-attenuation` and `tab:app-ceg-sahal` → `tab:app-obs-ceg-sahal`.
2. **Appendix D must not `\input`** `appD_lalonde_robust`, `appD_tfp`, `appD_replication`, `appD_attenuation`, `appD_ceg_sahal`, or `m5_progress_fig6_allocative`. Appendix E supersedes them, and the main text must not `\input` `table8_lalonde` or `table9_progress`.
3. **Appendix D, R4 M4(d) and minor 18.** Table E9 says the Llama 3, Marin ×3 and (Mis)Fitting technologies are "Chinchilla-form fits reported in Online Appendix D". The D writer must report them, including a (Mis)Fitting row, or the group label must change.
4. **Introduction and conclusion writers.** Say "fails to reject in a low-power joint test" (R1 10(a)). Drop the "0.20 / 0.26" family-FE finding and "reliability ≥ 0.988". If an allocative number is quoted, use "2.0 under the reference technology (1.1–4.9 across technologies)", with the Gundlach framing above. R3 minor 8: if the Ho finding is mentioned, give 0.0518 vs 0.0507 and the 0.27% MSE change in the same sentence.
5. **R3 E5 (share the Ho et al. convergence finding with the authors).** This is the lead author's decision. The text claims nothing about contact. If the finding is shared, add a footnote in Section E5.
6. **TBD-m9 placeholder** at the end of the "Algorithmic Progress" subsection: the M* shift between the experiment's two corpora signs the Hicks bias of the data-quality difference. Delete it if m9 cannot answer.
7. **Carried over from the ra4 review, not fixed:**
   - the Tobit benchmark's bootstrap is unclustered, so ARC-C and Winogrande conclusions stay tentative;
   - registry-technology intervals hold the technology fixed (open issue 6c);
   - the g_N/g_C profile should be re-optimized with more starts in [−0.5, 0] (6b);
   - harness mapping is untested (R2 24).
8. **Figure fonts.** R3 said the v1 Figure 9 fonts were unreadable in print. `appE_ridge` keeps m5's layout (legend at 5.4 pt). If needed, enlarge the fonts in `code/paper/make_appendix_e_ridge.py`.
9. **Length.** About 5,900 words and 23 pages in the test build (9 tables, 3 figures). If the Online Appendix must shrink, cut in this order:
   - Table E8 (CEG/Sahal) and its paragraph, which supports Cors. A2 and A4 only;
   - the "How large could transmission bias be?" paragraph;
   - Table E4 (compute-only), moved to the replication package.
10. **Main-text references used:** `sec:framework`, `sec:tech`, `sec:wedge`, `lemma:sigma`, `prop:wedge`, `prop:modelfree`, and the appendix labels `app:proofs`, `app:data`, `app:mc`, `app:additional`. Appendix A labels used: `prop:A-transmission`, `prop:selection`, `prop:proxy`, `prop:A-dmr`, `prop:farrell`, `cor:ceg`, `cor:sahal`, `cor:hall`, `lem:ce`, `lem:alloc`, `prop:A-wedge`. No new bibliography entries; all 44 keys resolve in `paper/references.bib`.
