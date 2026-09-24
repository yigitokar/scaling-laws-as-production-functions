# Writer notes (v2): Online Appendices B, D and F (assignment "appendix_D_F")

Writer: Claude, 2026-09-24. This run continued an interrupted earlier run of the same assignment, which had drafted
Appendix D and the appD/appF tables but left `appendix_wedge.tex` as a stub and Appendix B at v1.

## Files

Sections:
- `paper/sections/appendix_additional.tex` — Online Appendix D, "The Technology: Additional Results". Completed and extended: new
  "Estimators" paragraph and table, new "Farseer's local expansion path" paragraph, Step Law tables merged, intro updated.
- `paper/sections/appendix_wedge.tex` — Online Appendix F, "The Revealed Value of Compactness: Robustness". Written from scratch
  (the file was a one-line stub).
- `paper/sections/appendix_data.tex` — Online Appendix B, rewritten for v2 (samples table, Chinchilla design, experiment additions and
  pre-registration, released-model samples, universes, usage, licenses).

Tables written or changed in this run (all in `paper/tables/`):
- New: `appB_data.tex` (tab:app-data), `appD_estimators.tex` (tab:app-estimators), `appD_hyper.tex` (tab:app-hyper; replaces
  `appD_sfa.tex` + `appD_demand.tex`, which are no longer input anywhere), `appF_farseer_w.tex` (tab:app-farseer-w).
- Restructured to fit `\textwidth` (all previously overfull by 15–180 pt): `appF_cleaning.tex`, `appF_technologies.tex` (intervals in
  their own columns; share-interval of the reference moved to the note), `appF_pi.tex`, `appF_labown.tex`, `appF_wedge_cases.tex`
  (column widths), `tableB2_sweep_design.tex` (d → d_model, L → n_layer, R4 minor 5).
- `appF_conduct.tex`: the aggregate-share block of the module table was dropped (Table 4, panel C, reports it; the module block used a
  per-model envelope for "range", Table 4 uses the technology range, so keeping both would show two different ranges); caption now
  "Conduct Tests and Derivative Counts"; notation fixes.
- `appF_wedge_cases.tex`: carries an alias label `tab:ra5-wedge-cases`, because `framework.tex` cites that label (see open issue 2).
- From the interrupted run, kept after checking: `appD_dictionary`, `appD_chin_inference`, `appD_selection`, `appD_duality`,
  `appD_labs`, `appD_finite_grid`, `appD_mf_sens`, `appD_spec_tests`, `appD_kappa_robust`, `appD_robust`, `appD_neutrality`,
  `appD_farseer`, `appD_overhead`, `appD_measurement`, `appD_steplaw`, `appF_technologies`, `appF_conventions`, `appF_costsens`,
  `appF_family`, `appF_pi`, `appF_labown`, `appF_cleaning`, `tableB1_conventions`.

Figures used: D — `ra1_modelfree_kappa_profile` (fig:app-kappa-profile), `ra1_modelfree_overhead` (fig:app-overhead),
`m8_measurement_fig1` (fig:app-measurement); F — `ra2_wedge_trends` (fig:app-trends, new); B — `m2_design_planes` (fig:app_designs).

Test compile: `bash code/paper/test_section.sh appendix_data appendix_additional appendix_wedge` builds with no errors and **no
overfull boxes**; pages rendered with pymupdf and inspected (tables, figures, text pages). Test build: B 16 pp., D 26 pp., F 19 pp.
(68 pages with references). Text words (excluding tables/figures): B 4,950; D 4,860; F 4,520.
A whole-paper test build (all sections, in `_build/appendix_wedge`) resolves every reference from these appendices; the only
unresolved refs are in `appendix_mc.tex` (`prop:fdep`, `prop:info`, `prop:transmission`, `sec:obs`; being redone by the MC-fix agent).

## Labels defined
- Sections: `app:data` (+ `app:sweeps`, `app:chinchilla`, `app:isoflop`, `app:sweep`, `app:choices`, `app:obspanel`, `app:ho`,
  `app:licenses`); `app:additional` (+ `app:add-dictionary`, `app:add-chinchilla`, `app:add-modelfree`, `app:add-tech`,
  `app:add-overhead`, `app:add-measurement`, `app:add-biasformulas`, `app:add-flexible`); `app:wedge` (+ `app:wedge-cases`,
  `app:wedge-sample`, `app:wedge-tech`, `app:wedge-conv`, `app:wedge-labown`, `app:wedge-pi`, `app:wedge-conduct`,
  `app:wedge-family`, `app:wedge-cost`, `app:wedge-trends`).
- Tables: `tab:app-data`, `tab:app_conventions`, `tab:app_sweepdesign`; `tab:app-dictionary`, `tab:app-estimators`,
  `tab:app-chin-inference`, `tab:app-selection`, `tab:app-duality`, `tab:app-labs`, `tab:app-finite-grid`, `tab:app-mf-sens`,
  `tab:app-spec-tests`, `tab:app-kappa-robust`, `tab:app-robust`, `tab:app-neutrality`, `tab:app-farseer`, `tab:app-overhead`,
  `tab:app-measurement`, `tab:app-steplaw`, `tab:app-hyper`; `tab:app-wedge-cases` (alias `tab:ra5-wedge-cases`),
  `tab:app-cleaning`, `tab:app-technologies`, `tab:app-conventions`, `tab:app-labown`, `tab:app-farseer-w`, `tab:app-pi`,
  `tab:app-conduct`, `tab:app-family-budgets`, `tab:app-costsens`.
- Figures: `fig:app_designs`, `fig:app-kappa-profile`, `fig:app-overhead`, `fig:app-measurement`, `fig:app-trends`.
- Equations: `eq:app-measbias`, `eq:app-flexbias`.
- Labels that no longer exist (v1): `tab:data` (Appendix B now uses `tab:app-data`), `tab:app-sfa`, `tab:app-demand` (merged into
  `tab:app-hyper`). Nothing in the v2 sections references them.

## External labels referenced
`sec:framework`, `sec:tech`, `sec:tech:measurement`, `sec:wedge`, `sec:wedge-conduct`, `sec:concl`, `tab:designs`, `tab:econ`,
`prop:duality`, `prop:wedge`, `prop:ident`, `prop:modelfree`, `prop:pi`, `lemma:sigma`; Appendix A: `app:proofs`, `prop:A-wedge`,
`prop:A-family`, `prop:A-modelfree`, `prop:proxy`, `cor:suff`, `rem:A-conduct`, `rem:A-pi-illus`, `rem:A-pi-param`; Appendix E:
`app:obs`. All exist in the current files.

## Numbers used, with sources (T = `output/tables/`). Every number below was checked against the CSV in this run.

**Appendix F**
- Cleaning steps and Panel B subsamples: T `ra2_wedge_cleaning.csv` (all ten rows; e.g. start 3.767/0.931/0.613/0.168/0.7345/3.186;
  clean 3.986/0.974/0.636/0.182/0.749/3.336), `ra2_wedge_family_split.csv` (3.898, 4.013, 3.925, 3.986; s 0.743–0.751; band 0.125,
  0.300, 0.067, 0.222), `ra2_wedge_serve_split.csv` (33/7: 4.955, 0.798; 44/12: 3.798, 0.737, 0.955; 12/3: 7.591, 0.864),
  `ra2_wedge_insupport.csv` (30: 2.28/0.56/0.00; 47: 6.57/0.85/0.30; 57: 3.10/0.68/0.035; 20: 10.8/0.91/0.60; 2 of 77 in Chinchilla's
  (M, C) design). Synthetic row `r_synthetic`: 68, 3.798, 0.737, band 0.147.
- 32 technologies: T `ra2_wedge_technologies.csv` (`in_set`), every cell of Table F3 checked (σ*, M*(1e24), median w with 2.5/97.5
  percentiles, share w>1, median s with percentiles). Range 1.20–6.87, 0.64–1.00, 0.17–0.85, M* 1.4–426.
- Estimator variants: same file, sensitivity rows: `chin_nls` 4.579, `chin245` 5.495, `chin_cl` [1.831, 6.477], `chin_q_pairs`
  [3.249, 6.039]; R4's 3.78 = memo H1 (3.779 with m1's κ-free point).
- Curvature sensitivity: T `ra2_wedge_sigma_sensitivity.csv` (σ* 0.60 → 8.643, s 0.884, Llama 3 8B 19.545; 0.80 → 2.245, 0.555, 3.049;
  0.70 → 4.001; share 0.974 throughout).
- Conventions and MoE: T `ra2_wedge_conventions.csv` (head-FLOP 3.745 → 3.74; sub-1B 9.965/9.127, 10.670/6.847, 15.104, 14.263);
  `ra2_wedge_models.csv` (`moe`, `w_chin_q`, `wtotalN_chin_q`, `N_total`, `D`): all 12 rows recomputed, medians 5.14 and 2.07, s 0.80
  and 0.52. ∂ln w/∂ln D = b₁ = 0.4305, ∂ln w/∂ln N = −a₁ = −0.4244 (memo, reference technology).
- Lab-own: T `ra2_wedge_labown.csv` (10 rows of Table F5 panel B; medians 0.606/0.646 over 22 and 0.614/0.666 over 11 contemporaneous),
  `ra2_wedge_modelfree_sigma.csv` (global quadratics 0.653/0.650/0.629/0.644/0.610 with intervals; bias-corrected 0.662–0.678; paths
  0.463/0.405/0.424/0.422/0.492), `ra2_wedge_pi_anchors.csv` (M* anchors exp(lnMs): 22.33 [20.81, 23.74], 12.41 [10.67, 13.87],
  9.42 [8.01, 10.60], 9.97 [8.71, 11.07], 22.66 [18.25, 29.43]); `meta_mf8` (6.469 median, slope 0.501, M*(1e24) 14.8; 8B 12.2,
  405B 1.64 from memo H3); Marin 8B 5.08 vs 6.66 (memo H3/§5); OLMo κ = 1 alternative 2.010.
- Farseer check: T `ra1_modelfree_farseer_delta.csv` and `_sensitivity.csv` (every cell of Table F6; ratios 0.701, 0.440, 0.257;
  CV-optimum −0.246/−0.696/−1.220; ×2 −0.672), `ra2_wedge_extrapolation_check.csv` (3.098 → 3.560 / 3.488; s 0.677 → 0.719 / 0.713;
  κ = 1 Farseer 2.294 → 3.734; share understated 0.561); quadratic term 0.020 (0.001), 0.018–0.026 across bandwidths (ra1 memo H4).
- Partial identification: T `ra2_wedge_pi_mstar.csv`, `ra2_wedge_pi_summary.csv` (86/86/77/23%; s bounds 0.589/0.933 etc.),
  `ra5_theory_pi_sign.csv` + `ra5_theory_pi_bounds.csv` (Panel C: 149/137 models; shares; M*(1e24) bounds). Ambiguous-model list:
  ra2 memo H6 (review-verified).
- Conduct: T `ra2_wedge_conduct.csv` (open 0.517 (0.180) p 0.0115; 2023+ 0.712 p 0.001; on s 0.294 p 0.097; serve 0.453 (0.257)
  p 0.390; clean+suites 0.467 p 0.070 and s 0.135 p 0.046; tier in-window −0.385 placebo 0.997; distance 0.464 vs placebo mean 0.455),
  `ra2_wedge_bunching.csv` (0.509 vs 0.134; 82.6 (9.8); 45.3 at 6.5–9.5B), `ra2_wedge_trends.csv` (closed 6 in 2024, 2 in 2025).
- Family: T `ra2_wedge_family_level_W.csv` (11 rows), `ra2_wedge_family_split.csv` (all 173: 83 models / 23 families).
- Cost side: T `ra2_wedge_costsens.csv` (all rows).
- Validation: T `ra2_wedge_validation_openrouter.csv` (4.6%, 0.7%, 0.5%, 0.2%, 0.08%, 0.04%), `ra2_wedge_validation_hf.csv`
  (0.605/0.869/0.898/0.795; T/D 0.603 p 0.086).
- Trends: T `ra2_wedge_trends.csv` (open medians 0.508/0.812/0.849; closed 2024 0.635), `ra2_wedge_aggregate.csv` (clean reference
  0.277/0.585/0.829; lab-own 0.273/0.555/0.824; 405B share 0.548; leave-out 0.734), `ra3_econ_inference_share_by_tech.csv` (the rise
  2023 → 2024 → 2025 holds for all 32 technologies on the clean sample; checked).

**Appendix D (added or re-checked in this run)**
- Estimators table: T `m1_chinchilla_horse_race.csv` (points, pairs SDs), `m1_chinchilla_system.csv` (system column),
  `m1_chinchilla_system_lr.csv` (LR 1.014, p 0.602), `m1_chinchilla_estimator_diffs.csv` (Δβ 0.039 (0.018), 0.060 (0.020);
  ΔM* −3.65 (4.95), −7.55 (5.32)), `m1_chinchilla_fdep.csv` + `_fdep_sizecontrol.csv` (0.330/0.008/0.29–0.42; cond. 61.9/413.5/
  68.6–95.1; raw 2,003/14,736), M* dispersion IQR/1.349 recomputed from `data/processed/m1_chinchilla/boot_*_pairs.npy`
  (6.92, 7.32, 4.95, 3.84); 83.8% of residuals in the linear region (m1 memo).
- Farseer path paragraph: ra1 memo H4 and §2.5; T `ra1_modelfree_farseer_sensitivity.csv` (path means 0.707, 0.726/0.727, 0.683,
  0.664), `ra1_modelfree_farseer_sigma_slope.csv` (0.708; 0.701–0.711; 0.670 with embeddings); bandwidths 0.237/0.745.
- Merged Step Law table: T `m8_measurement_sfa.csv`, `m8_measurement_demand.csv` (all cells re-checked).
- Spot checks of the interrupted run's text against memos/CSVs, all passing: 84%; MAD 0.0049 vs 0.0076; calibrated critical value
  20.2; LR 29.0 (p 0.003); band LR ≤ 1.21 on [0.25, 0.94]; outliers 9–45 residual SDs; 12 rules, 10 in [0.95, 1.06], 0.78 and 1.16;
  Llama 3 slope t −2.11, wild p 0.51–0.78; Marin three-quarters parabola bias; per-budget noise-free bias [−0.005, +0.002]
  (`ra1_modelfree_isoflop_bias.csv`); 2.7 decades; θ_X = 1/3 → 0.76, 1/2 → 0.68; 0.7–1.2 log points; 7.0 vs 2.8 SE; table cells of
  appD_labs, appD_duality, appD_selection, appD_kappa_robust, appD_overhead, appD_measurement against the memo tables.

**Appendix B**
- Chinchilla extraction: raw `data/raw/epoch_chinchilla/svg_extracted_data.csv` (245 runs; FLOP 1.397e18–1.296e22; N 57.3M–16.18B;
  five dropped runs 2.0–6.8B at ~1e19), `m1_chinchilla_design_variance.csv` (0.33/0.37/0.21), ra1 memo H5 (35 trunks, 137/108).
- Experiment: `code/sweep/run_grid.py` (tags lrcal, lrcorner, seedcorner, seeds, hiM, hiM2 dropped), `code/sweep/extras.sh`,
  `data/processed/sweep/meta.json` (extension shards 1.78B/1.72B tokens; WikiText-103 654,959 tokens, 125 docs), git commit
  `bd5c0ad` 2026-09-24 03:12:31 +0300 (pre-analysis plan). Status at writing: FineWeb-Edu main 44/44, lrsweep 12, lrcorner 16;
  FineWeb main 5, lrcal 12, lrcorner 5 (queues running) — the completed-endpoints sentence stays a TBD-m9 placeholder.
- Samples table: v1 `table2_data.tex` (sources in `writer_data.md`), Table 2 `table2_designs.tex` (Llama 3, Marin, Porian ranges and
  spreads), `ra2_wedge_models.csv` (clean sample: N 0.1345–405.9B, M 19.5–60,398), R4 minor 16 (universe minimum 134.5M).
- Universes: `code/analysis/ra2_wedge/sample.py::universe`, `run.py` (186 confident; 141 open; 220 open 2023–2026 for bunching;
  133 = open 2023–2025 for tier regressions).

## Referee comments addressed (where in B, D, F)
- **R1 c1, R2 Major 2b, R3 M5**: F.1 (what s measures by conduct; "planned serving" only for developers that serve), F.2/F.7 (estimand by
  serving footprint; serving coding with dates and judgment calls).
- **R1 c2, R2 Major 5a/5c**: F.7 (multi-tier bunching with placebo menus), F.8 (family budgets, Prop. A9), F.2 (results without
  common-D families).
- **R1 c3**: F.1 last row (factor-bias/output critique), F.5 (lab-own technologies with vintage); Raval second-margin test listed as
  not done (F.10).
- **R1 c4, R2 Major 1/6, R4 M2**: F.3 (ex-ante rule, 32 technologies, κ-free reference, curvature sensitivity, estimator variants,
  R4's 3.78 reproduced), F.6 (partial identification with anchors, PI-1..PI-4, ambiguous list).
- **R1 c5**: F.9 (δ, φ, p sensitivity; s independent of p).
- **R1 c6(b), c7, R1 minor 29**: D (Farseer local path: like-for-like σ* on the path; first-derivative estimator; SE excludes smoothing
  bias), D finite-grid bias.
- **R1 c8(a,d,e,g,h), R1 minors 10, R2 Major 9(a,d)**: D dictionary (Bond–Söderbom collinearity, assigned inputs not price shocks,
  Nerlove only as inverse cost function, measurement map not TFPR, tokenizer changes units of output and D, Approach 1 estimates
  factor demands); "IsoFLOP-minima frontier".
- **R1 c9(a–d), R4 M1, M5, M6(a–c)**: D (design-conditional wild bootstraps, trunk/singleton clusters, calibrated LR tests, profile
  over the full grid, both CES tests, rank-one counts under both E, bootstrap-resolution labels, A/B SEs as parameterization artifacts
  with raw vs normalized condition numbers — also R4 minor 42).
- **R1 c9(f)**: F (joint bootstrap for sample statistics).
- **R1 minor 21–25, R2 Major 4, R2 Major 10**: F.2 (dedupe, suites, MoE, distilled/pruned, multimodal, instruct-only, undocumented D),
  F.4 (conventions, head FLOPs, MoE active/total bounds, tokenizer effect), B (popularity selection of ObsScaling; Epoch D = C/(6N)
  rows only if Confident).
- **R1 minor 26–28, R3 minor 17–19**: D (Huber ≈ LAD; outlier leverage not truncation; Hausman–Wise in the appendix; 70B check without
  power).
- **R1 minor 31–34, R2 minor 14–16, R4 M8(c), minor 27–29**: D (tilt-based neutrality; DataDecide resolution; SFA framed as variance
  function only; θ_X sensitivity; 1.57 coincidence; loss switch ≤ 0.005; Björck range; β₂ renamed; "close to").
- **R1 minor 37**: F.5 (vintage of lab laws). **R1 minor 39, R2 Major 11, R3 M3c**: F.10 (OpenRouter ToS, author totals; HF model tree).
- **R2 Major 3, R1 c7.2**: F.6 (Farseer in-support check and its application to clean models; limits).
- **R2 Major 7, R1 c11, R4 M10, R3 M9 (design)**: B (corpora not labs; four-layer high-M runs to 3.2B tokens; FineWeb LR calibration and
  LR ×{0.5, 2} corners; seeds in both corpora incl. the high-M corner; WikiText-103 neutral output; both parameter conventions;
  trunk-sharing and CRN clarified; pre-registration with commit time and the one deviation).
- **R2 Major 8, R2 minor (practitioner)**: D (κ-robustness table; overhead table and de Vries comparison).
- **R2 minor 10, R4 M1(a), minor 15, minor 16, minor 17, minor 19, minor 5**: B.
- **R3 minor 33**: claims register (`appD_claims.tex`) not included; the merged Step Law table shortens D.
- **R4 M4(a), M4(d), minor 18**: D selection paragraph (0.78–1.16), `appD_labs` has Llama 3, Marin ×3 and (Mis)Fitting (resolves the
  Appendix E writer's open issue 3).
- **R4 D4** ("σ more stable than a"): D Step Law paragraph disavows it explicitly.

## Open issues (for the integrator and lead author)
1. **Label clash `app:wedge`.** `appendix_proofs.tex` line 777 labels its subsection "What over-training reveals: the generalized
   wedge" `\label{app:wedge}`; Appendix F defines `app:wedge` as assigned. The whole-paper build warns "Label `app:wedge' multiply
   defined" and, since F comes later, `\ref{app:wedge}` resolves to F (the intended target in introduction.tex and wedge.tex). Rename
   the proofs label (e.g. `app:A-wedge-sub`); nothing references it with the proofs meaning.
2. **`framework.tex` line 215–216** says "Online Appendix~\ref{app:proofs} tabulates eleven cases (Table~\ref{tab:ra5-wedge-cases})".
   The table is in Appendix F (`tab:app-wedge-cases`; the alias `tab:ra5-wedge-cases` makes the reference resolve). Change
   `app:proofs` → `app:wedge` and the label to `tab:app-wedge-cases`; then the alias can be deleted.
3. **wedge.tex (IV.F)**: "for 2025 it ranges from 0.24 to 0.96 across technologies" follows the 141-model universe numbers, but the
   0.24–0.96 range is computed on the 13-model clean sample (`ra3_econ_inference_share_clean.csv`; Table 4 note says so). Suggest
   "on the clean sample, the 2025 share ranges from 0.24 to 0.96".
4. **Unused table files** (safe to delete): `appD_sfa.tex`, `appD_demand.tex` (merged into `appD_hyper.tex`), `appD_claims.tex`,
   `appD_aggregate.tex`, `appD_family.tex`, `appD_homothetic.tex`, `appD_rivals.tex`, and the v1 files that Appendix E supersedes
   (`appD_lalonde_robust`, `appD_tfp`, `appD_replication`, `appD_attenuation`, `appD_ceg_sahal`), plus `table2_data.tex`.
5. **ra3 appendix exhibits** (`ra3_econ_compute_growth.tex`, `ra3_econ_inference_disclosures.tex`, figure `ra3_econ_inference_share`)
   were not placed; they are outside this assignment. Appendix F could host them if the integrator wants them (economics writer's
   issue 4).
6. **Not done (stated in the text where relevant):** Raval-type second-margin test; byte normalization of D; durability of T; audit of
   corpus-size token counts (e.g. Qwen2.5's 18T for every size); DataDecide tilts from final checkpoints only (R2 minor 14); rebuilding
   Chinchilla's D from Hoffmann et al.'s architecture table (R2 Major 9b; D bounds the effect at ±0.04 in σ* for |η_C| ≤ 0.1).
7. **Experiment deviations** beyond the one stated in the plan (for the m9 memo): R1's dose–response mixtures, R2's 100–300M anchor runs
   and weight-decay envelopes, and M ≥ 10⁴ in total parameters were not run; B describes only what was run.
8. **TBD-m9 placeholder** in B ("Hardware"): completed endpoints by corpus and tag, failures, wall-clock hours.
9. **Minor facts to confirm if time allows:** Hoffmann et al.'s smallest model (44M) is taken from R2 minor 10 and Hoffmann et al.'s
   Table A9 as recalled, not re-read from the PDF; the ~1,000 Farseer and ~3,700 Step Law totals are from R4 minor 19.
10. **Module table mismatch (R4 minor 43)**: `output/tables/m1_chinchilla_horse_race.tex` still reports M* SDs rather than IQR/1.349;
    the paper's `appD_estimators.tex` uses IQR/1.349 recomputed from the stored draws. Regenerate or delete the module .tex.
11. **Length.** The three appendices run to about 61 pages in the test build (B 16, D 26, F 19). Further cuts, if needed: D's
    `appD_robust` (overlaps `appD_kappa_robust`), `appD_steplaw` + `appD_hyper` and their paragraphs (to the replication package),
    and F's `appF_costsens` (its content is in one paragraph of F.9 and Section IV).
