# Writer notes: Section IV "The Technology" (paper/sections/technology.tex)

Writer: section writer "technology" (Claude), 2026-09-24.

**Files**
- Section: `paper/sections/technology.tex`
- Tables:
  - `paper/tables/table3_chinchilla.tex` (label `tab:chinchilla`)
  - `paper/tables/table4_technology.tex` (`tab:technology`)
  - `paper/tables/table5_measurement.tex` (`tab:measurement`)
- Figures:
  - Figure 3 = `m1_chinchilla_isoquants` (`fig:chinchilla`)
  - Figure 4 = `m2_fig3_sigma_forest` (`fig:sigma`)
  - Figure 9 is a red placeholder (`fig:experiment`)
- New bib file: `lit/bib/extra_writing_technology.bib`

**Compile status.** The section compiles with `code/paper/test_section.sh technology` with no errors and no overfull boxes. All three tables fit the 135.5 mm AER text width at `\scriptsize`; I checked the rendered pages. Undefined references in the isolated test are expected, since they point to other sections: `sec:data`, `sec:framework`, `sec:ident`, `sec:wedge`, `sec:concl`, `prop:duality`, `prop:fdep`, `prop:info`, `prop:wedge`, `lemma:sigma`. One citation is undefined, `penedo2024fineweb`, which is new (see below).

**Length.** The main text is about 3,450 words (tables, figure notes and placeholders excluded), against a target of about 3,000. If the paper runs long, these are the candidates to cut first:
- the cluster-inference paragraph of IV.A;
- the Step Law paragraph of IV.D;
- the Gadre power caveats of IV.C.

**Preamble.** Nothing beyond `main.tex` is needed. The tables use `booktabs`, `array` (`>{\raggedright\arraybackslash}p{}`) and `tabular*`, plus `amsmath` for `\eqref`.

---

## 1. Numbers used (value → source file)

### IV.A Chinchilla (module m1; memo `output/memos/m1_chinchilla.md` + `_review.md`)

**Data and design**

| Number | Source |
|---|---|
| 245 runs; 9 budgets 6e18–3e21; n = 240 after dropping the 5 highest-loss runs | m1 memo §2; `m1_chinchilla_a2_minima.csv` (C 5.6e18–2.9e21, digitized) |
| Design variance 0.90 transverse vs 1.83 along the path (33%) | `m1_chinchilla_design_variance.csv` (var_trans 0.9027, var_along 1.8325, share 0.330) |

**Estimators and inference**

| Number | Source |
|---|---|
| 84% of residuals in the linear region | `m1_chinchilla_horse_race.csv` share_linear_huber 0.8375 |
| SD(β_Huber − β_LAD) = 0.004 | `m1_chinchilla_huber_vs_lad.csv` 0.00396 |
| Besiroglu published values are an LAD-type optimum; reproduced to 4–5 digits | m1 memo H1; `m1_chinchilla_besiroglu_check.csv` |
| β = 0.367 (Huber) / 0.406 (Gaussian) / 0.428 (NLS levels) | horse race table/CSV |
| Paired Δβ = 0.039 (0.018) and 0.060 (0.020) | `m1_chinchilla_estimator_diffs.csv` |
| σ* 0.718–0.737; a 0.513–0.545 | horse race |
| ΔM*(5.76e23) = −3.7 (4.9) and −7.6 (5.3), not significant | `estimator_diffs.csv` |
| Cluster SEs 1.5–2.2× pairs; β SE 0.021 → 0.046 | horse race table |
| M*(5.76e23) CI [7.8, 35.1] pairs vs [3.2, 126] cluster | `m1_chinchilla_horse_race.csv` lo/hi (7.78, 35.12; 3.17, 125.8) |

**Selection (dropped runs)**

| Number | Source |
|---|---|
| β 0.453 → 0.367; swing 0.086 (0.052); 0.347 at k = 15 | `m1_chinchilla_selection_swing.csv`; memo H5 |
| Hausman–Wise β 0.406 = Gaussian 0.406 | `m1_chinchilla_selection_rules.csv` (0.405945 vs 0.405883) |
| D/N 0.04–0.40; residuals 9–45× σ_u | memo H5; `selection_worst_points.csv` |
| D/N < 0.4 rule gives β = 0.371 | selection_rules (0.3708) |
| w(70B) 1.54 at n = 245; 0.95–1.06 under every rule that removes the outliers | selection_rules; review item 15 |

**Duality and revealed preference**

| Number | Source |
|---|---|
| Approach 2: a = 0.498 (OLS s.e. 0.025); Hoffmann report 0.49 | `m1_chinchilla_duality_bootcal.csv` se_a_ols 0.0252; memo H6 |
| Refit / Besiroglu p ≥ 0.16 (classical and calibrated variants) | `m1_chinchilla_duality_classicalF.csv` (F p 0.65, 0.66); bootcal (0.23/0.25/0.70/0.61; 0.18/0.16/0.75/0.63) |
| Hoffmann: F p = 0.040; calibrated 0.060; wild 0.21–0.38 | classicalF; bootcal rows 6–7, 18–19 |
| Level gap 0.122 (0.046); slope gap 0.042 (0.023); refit gap 0.026 (0.046); Besiroglu 0.028 (0.046) | classicalF |
| Frontier minima 0.005–0.008 nats below the A3 frontiers; stratified rejection, not wild | memo H6 [Rev]; bootcal p_frontier_boot |
| w(70B): refit 1.04 [0.82, 1.42] pairs; cluster [0.54, 2.01]; Hoffmann 0.62/0.71; Besiroglu 1.03 | `m1_chinchilla_revealed_pref_70b.csv`; horse race lo/hi_cluster_w |

**System estimator**

| Number | Source |
|---|---|
| α 0.348 (0.014), β 0.365 (0.016), a 0.512 (0.015), γ 0.178 (0.005), σ* 0.737 (0.005), M*(1e21) 21.6 (2.7), M*(5.76e23) 18.5 (6.0), w 1.03 (0.12) | `m1_chinchilla_system.csv` (laplace n240); `revealed_pref_70b.csv` |
| Nested LR 1.01, p = 0.60; F-form p = 0.67 | `m1_chinchilla_system_lr.csv`, `_smallsample.csv` |
| SE(a) 0.018 → 0.015 | memo H8 |

**Specification tests**

| Number | Source |
|---|---|
| CES α − β = −0.020 (0.029) [0.066]; p = 0.50 [0.76] | `m1_chinchilla_spec_ces.csv` |
| κ̂ = 0.774 (0.060) [0.111]; p < 0.001 (0.00015) [0.042]; Gaussian LR 46.2 | `m1_chinchilla_spec_kappa.csv` |
| σ* with κ free 0.700 (0.015) [0.018] | `spec_kappa.csv` |
| With κ free: a = 0.504, γ = 0.165 | `spec_kappa.csv` |
| Rank one τ̂_κ = −0.009, p = 0.69 | `m2_spec_tests.csv` (tau_q, p_rank_one_q, chinchilla row). This is m2's version with Ê from the κ family. m1's own translog test (E fixed at the Huber Ê) gives p = 0.06–0.27 and depends on E; not used. |

**Functional dependence**

| Number | Source |
|---|---|
| n = 41; transverse share 0.8%; profile LR ≤ 2.0 (1.96); full design 424 (424.5), set [0.67, 0.69]; random n = 41: 79–101 | `m1_chinchilla_design_variance.csv`; `m1_chinchilla_identification.tex` notes; `m1_chinchilla_fdep_sizecontrol.csv` |
| Condition number 62 / 413 / 69–95 | `m1_chinchilla_fdep.csv` cond_norm; `fdep_sizecontrol.csv` |
| SE ratios: α 2.3, a 2.2, γ 0.94 (β 1.25 not quoted in the text); σ* on-path s.e. 0.011 | `m1_chinchilla_fdep_se_ratios.csv`; `fdep.csv` |

**Table 3 LAD note** (α 0.348, β 0.366, σ* 0.737, M*(1e21) 21.5, w 1.03, Δβ −0.001 (0.004)): horse race CSV and `estimator_diffs.csv`.

### IV.B Across sweeps (module m2; `m2_techpanel.md` + `_review.md`)

**Table 4**

| Number | Source |
|---|---|
| All Panel A entries (Huber rows; SEs; M* CIs; sd_offpath; q̂ and σ*_q with SEs) | `m2_table3_technology.csv` (q = κ). Gadre per-corpus s_M = 1.46/1.47/1.47 (the m2 memo quotes 1.44 for the pooled data). |
| Panel B: RMSE in-sample 0.0087 / 0.0025 / 0.0018; top-decile hold-out 0.025 / 0.0052 / 0.0026; four-largest-N hold-out 0.013 / 0.0036 / 0.0026; BIC −2,647 / −3,663 / −3,916 | `m2_farseer_forms.csv`: Chinchilla Huber row; Kaplan-q NLS row; Farseer Eq. 3 row |
| Panel B: σ*_κ 0.710 [0.705, 0.716] (Huber q-fit) | `m2_table3_technology.csv` |
| Panel B: median local σ 0.690 [0.679, 0.699]; Farseer Eq. 3 median 0.706 | `m2_farseer_local_sigma_summary.csv` |

**Text**

| Number | Source |
|---|---|
| σ* 0.735–0.828; SEs 0.005–0.034 | table3 CSV |
| κ = 1 rejected in all 7 sweeps, p < 0.001; κ̂ 0.22–0.77; σ*_κ 0.51–0.71 | `m2_spec_tests.csv` |
| RMSE 0.0085 → 0.0025 on Farseer | `m2_farseer_forms.csv` (Chinchilla NLS 0.00850; Kaplan-q NLS 0.00247) |
| Small-sweep σ*_κ intervals about ±0.1 | spec_tests lo/hi |
| Local σ range 0.61–0.99 | local_sigma_summary (min 0.606, max 0.987) |
| Out-of-sample 0.022–0.025 vs 0.0026 | `m2_farseer_forms.csv` |
| Capital–labor 0.4–0.7 | chirinko2008sigma (0.4–0.6), klump2007factor and oberfield2021micro (0.5–0.7), raval2019micro lower (0.3–0.5); `lit/notes/io_ces_duality.md`, SYNTHESIS §4 |
| CES Wald p 0.22–0.90 | `m2_spec_tests.csv` |
| Rank one not rejected at 5% in 6 of 7 sweeps; Farseer p ≤ 0.003 with pairs CI [−0.001, 0.026] including 0 | `m2_spec_tests.csv` (p_rank_one_q: 0.69, 0.003, 0.21, 0.82, 0.80, 0.053, 0.75) |
| a 0.37–0.57; M*(1e21) 3.4–60; CI factors 1.7–70; γ 0.10–0.18 (7 sweeps, DataDecide 0.090 excluded) | table3 CSV |
| Farseer embeddings: a 0.526 → 0.411; M*(1e23) 19.5 → 45; σ* 0.772 → 0.724 | m2 memo H1e / `m2_robustness.csv` (review-verified) |
| Muennighoff D/N ≥ 0.4: a 0.535 → 0.360; M* 60 → 139; σ* 0.735 → 0.726 | m2 memo H1a / `m2_robustness.csv` |
| Gadre RefinedWeb M* 3.4 (Huber) vs 10.4 (NLS) | table3 CSV |
| σ*²/2 ≈ 0.3 compression | derivative of 2/(2+S); m8 review M1 |
| Step Law: 7.0 SE vs 2.8 SE | `m8_measurement_stability.csv` (6.97, 2.85) |

### IV.C Data quality (m2)

| Number | Source |
|---|---|
| Gadre 104 runs; one technology p ≤ 0.002; E-shift only p = 0.004, 90% explained; Hicks + E_r p = 0.90, 96.8% vs CE 96.9%; daugE p = 0.13, paugE 0.15 | `m2_neutrality.csv` |
| NLS Wald rejects daugE (p = 0.018); Huber does not (0.77) | `m2_neutrality_wald_check.csv` |
| M* differs 8–18% across corpora | `m2_neutrality_magnitudes.csv` (Mstar21_ratio 1.08 / 1.18) |
| DataDecide 25 recipes, 21,888 checkpoints; Hicks + E_r LR 145 on 24 df, p ≤ 0.01; E-shift 98.1%; + Hicks 99.71%; factor bias +0.05 pts | `m2_neutrality.csv` |
| Tilt 0.22–0.26 → M* 2.9–3.4× (2.93/3.43); wedge 1.25–1.29× | `m2_neutrality_magnitudes.csv` |
| σ*_r 0.790–0.839 | `m2_neutrality_magnitudes.csv` (min 0.7897, max 0.8399 Huber/NLS). The m2 memo also quotes a cross-recipe s.d. of 0.010; not used in the text. |

### IV.D Measurement and flexible inputs (m8; `m8_measurement.md` + `_review.md`)

**Porian path (Table 5 Panels A–B)**

| Number | Source |
|---|---|
| Panel A: all â, CIs, published values, Δâ; shares 0.39 / 0.47 | `m8_measurement_porian_steps.csv`; `m8_measurement_porian.tex` |
| Panel B: all entries | `m8_measurement_porian_counting.csv`; `m8_measurement_porian.tex` (review-corrected total count) |
| 975 runs; all 10 exponents within 0.003 | m8 memo §1.1 |
| Head-count effect 0.12–0.17; tuning 0.19–0.23; total tuned 0.46–0.47; 5M–901M | m8 memo §1.2 |

**Measurement bias formula (equation `eq:measbias`)**

| Number | Source |
|---|---|
| a_m = β/(β + αφ − ζ), φ = 1 − s(1 − θ), ζ = (1 − θ)s(φ − θ)/φ | m8 memo §1.3; review re-derived |
| Predicted 0.792 / 0.757 vs Pearce–Song 0.78 / 0.74 | `m8_measurement_pearce_song_sim.csv` |
| Over-prediction 20–50% (semi-synthetic, actual design) | `m8_measurement_porian_meas_formula.csv` (ratios 1.22–1.52) |
| Chinchilla non-embedding a 0.514 (0.020) → 0.556 (0.023); Δ 0.042 (0.006) | `m8_measurement_chinchilla_nonembed.csv`; memo §1.3 |

Notation change: m8's κ and η in the formula are renamed φ and ζ here, because κ is the paper's outer exponent and η is compute noise in model_spec.

**Flexible-input formula**

| Number | Source |
|---|---|
| a_obs − a = −∂(Δ/f'')/∂ln C; f'' = (α+β)γR* | m8 memo §1.4 |
| 47–114% explained | `m8_measurement_porian_flex_formula.csv`; memo |
| δ = 1.57 nats at 2.5e16, gone by about 3e18 | memo §1.4 (review: 3.2e18) |
| 1.57B-token warmup | `m8_measurement_porian.tex` notes |

**Step Law (Table 5 Panel C)**

| Number | Source |
|---|---|
| 1,911 runs; 17 cells; up to 12 × 10 = 120 configurations | m8 memo §2.3 |
| SFA γ_D −0.231 [0.032], γ_N −0.038 [0.076]; exponential −0.260 [0.023], −0.063 [0.070] | `m8_measurement_sfa.csv`/`.tex` |
| LR demand −0.823 (0.127), 0.288 (0.071); published −0.713, 0.307; conditional 0.046 (0.015); Bjorck −0.32; gap closed 0.26 (0.02)–0.43 (0.13) | `m8_measurement_demand.csv`/`.tex` |
| SSR within 17% for E in [0.6, 1.7]; a 0.76 → 0.45 | m8 memo §1.5(i) |
| σ* 7.0 SE vs a 2.8 SE | `m8_measurement_stability.csv` |

### IV.E Controlled experiment (plan §III.B and `paper/notes/m9_spec.md`)

- Design facts only: 8 sizes (0.4M–49M non-embedding), D 25M–800M, shared 8,192-token BPE, WSD with cooldown branches, per-width LR, common random numbers, and evaluation on both validation sets.
- All results are TBD.

---

## 2. Claims with their required caveats (as written)

1. **Huber(1e-3) is LAD in practice; Besiroglu's published values are LAD-type.** This is a numerical-practice point.
2. **Robust vs least-squares β differ significantly** (paired z = 2.2–3.0). The M* differences are *not* significant; the text says so.
3. **Dropping the 5 highest-loss runs is outlier leverage, not truncation bias.**
   - Hausman–Wise assumes Gaussian errors, and the outliers contradict that; the caveat is in the text.
   - w(70B) is in [0.95, 1.06] under any rule that removes the outliers.
4. **Duality.** Design-consistent tests are required.
   - Refit and Besiroglu: p ≥ 0.16.
   - Hoffmann A3: *marginally* inconsistent (p ≈ 0.04–0.06; wild 0.21–0.38), in the path level (about 12%).
   - The χ² p = 0.0004 is NOT used.
   - The frontier rejection is stated as holding under the stratified bootstrap only.
5. **Revealed preference at 70B.** Worded as consistency of the refit, not rejection of Hoffmann: the cluster interval [0.54, 2.01] contains 0.62–0.71.
6. **κ = 1 rejected on Chinchilla** (cluster p = 0.042). The digitized/quantized caveat is in the text, with a pointer to the replications.
7. **Functional dependence.** Only size-matched SE ratios are used (2.3 / 2.2 / 0.94), not the raw 4.5–7×.
8. **σ < 1 everywhere.**
   - The level is stated as a range: 0.74–0.83 (κ = 1) and about 0.7 (curvature free; Farseer 0.69–0.71).
   - Capital–labor wording: "just above the range under the Chinchilla form; overlap its top once curvature is freed"; "complements of roughly the strength of capital and labor, if anything slightly weaker". Never "far above".
9. **No "σ is more stable than a".**
   - The text explicitly disclaims it, citing the σ*²/2 compression and the Step Law SE-unit result (m8 review M1).
   - The plan's sanctioned sentence is used verbatim: "The frontier elasticity and σ are comparable across sweeps in point estimate; M* is not."
10. **Rank-one / separability not rejected.**
    - With the κ-family Ê: 6 of 7 sweeps.
    - Farseer: a small departure against the residual-bootstrap null, but the bootstrap interval includes 0.
    - The retracted "saddle" result is not mentioned.
11. **Data quality.**
    - Gadre: Hicks + E_r not rejected (p = 0.90), with the low-power caveat and the Huber/NLS Wald disagreement.
    - DataDecide: everything rejected, yet E + Hicks explain 99.71%.
    - Tilts → M* 2.9–3.4× and wedge 1.25–1.29×.
    - Explicit caveats:
      - "better data ⇒ data-augmenting" is not identified (the filtered DCLM variants have the worst C4 asymptotes; lower M* given the tilt is an identity under common exponents);
      - schedule artifact: relative tilts only, assuming no recipe × schedule interaction;
      - E_r partly reflects distribution match.
    - The Lemma 1 reading is not claimed.
12. **Kaplan–Chinchilla decomposition, 39–47% / 53–61%.**
    - The order-dependence caveat is handled by Panel B (near-additivity).
    - Stated for 5M–901M-parameter models.
13. **Measurement formula.** "Gets sign and magnitude roughly right but over-predicts, by 20–50% on the actual design." The continuous closed form's 63–108% is not quoted; add it if a referee asks.
14. **Flexible-input formula.** Explains "about half to all (47–114%)", depending on f''.
15. **Bjorck–Step Law.** Conditioning removes the positive sign but closes only 26–43% of the gap; not "mostly explained".
16. **Step Law does not identify a** (flat E-profile).

---

## 3. Cross-references assumed to exist elsewhere

**Sections**
- `sec:framework`: the κ family L = E + [AN^{−a1} + BD^{−b1}]^κ; σ*, ψ_N, ψ_D; the geometry/rank-one statement.
- `sec:ident`: separability and geometry.
- `sec:data` and `tab:data`: the sweeps; our controlled experiment design.
- `sec:wedge`: the wedge; rival explanation from factor bias.
- `sec:concl`: reporting recommendations for a.

**Results**
- `prop:duality` (Prop 1: Approaches 1/2/3).
- `prop:wedge` (Prop 2: w = 1 + T/(3D) and ŵ = w e^{−χ}).
- `prop:fdep` (Prop 3).
- `prop:info` (Prop 4).
- `lemma:sigma` (Lemma 1: interior optimum ⇒ σ < 1).

**Appendices**
- Online Appendix A: Lemma A4 (geometry, rank one) and Lemma A3 (allocation; used only inside a TBD placeholder), cited by name.
- Online Appendix D (appendix_additional) is assumed to contain:
  - Gaussian-NLS rows, the pooled Gadre fit and the robustness tables for Table 4 (`m2_table3_technology` Panel B, `m2_robustness`);
  - the full neutrality tables and the Wald cross-check (`m2_neutrality.tex`, `m2_neutrality_wald_check.csv`);
  - **the derivation of equation `eq:measbias`** (m8 "Proposition M") and of the flexible-input formula (m8 "Proposition F"). See issue 2.

---

## 4. Placeholders (all `\textcolor{red}{[TBD-m9: ...]}`, in subsection IV.E only)

1. σ* and σ*_κ by lab, κ free, on a common output.
2. Neutrality of data quality: which restriction is rejected; χ; effects on M* and ŵ; sign relative to Lemma A3.
3. Extrapolation check: fit on M ≤ 100 (or 341), evaluate at M ≈ 1,000–2,000; w_extrap/w_full; direction of bias for Section V.
4. Seed noise vs residual s.d.
5. Figure 9 (`fig:experiment`): a placeholder box with caption and TBD notes. Replace the `\fbox` with `\includegraphics{m9_...}` when the file exists.

The four TBD sentences follow "Four results follow." When m9 arrives, each becomes one or two sentences with SEs clustered by width. m9_spec says to report both pairs and wild, with 8 clusters.

---

## 5. New bib keys

- `penedo2024fineweb` in `lit/bib/extra_writing_technology.bib`.
  - Verified: arXiv:2406.17557 and the NeurIPS 2024 Datasets and Benchmarks proceedings page.
  - The data-section writer may add the same paper under the same or another key. Deduplicate.
- Every other key used already exists in `paper/references.bib`:
  - hoffmann2022training, besiroglu2024chinchilla, grattafiori2024llama, cameron2008bootstrap, hausman1977social, czech2026problems, leonledesma2010identifying;
  - li2025predictableb, gadre2024language, bhagia2024establishing, muennighoff2023scaling, magnusson2025datadecide;
  - chirinko2008sigma, klump2007factor, oberfield2021micro, raval2019micro, kaplan2020scaling, porian2024resolving, pearce2024reconciling;
  - nerlove1963returns, christensen1976economies, collardwexler2016production, ackerberg2015identification, gandhi2020identification;
  - li2025predictablea, aigner1977formulation, caudill1995frontier, milgrom1996lechatelier, bjorck2024scaling, hagele2024scaling.

---

## 6. Open issues for the integrator

1. **Duplicate labels (important).** `appendix_proofs.tex` defines `prop:info`, `prop:dmr`, `prop:transmission`, `prop:wedge` and `prop:pi` on its A-numbered results. These are the same labels the plan assigns to the main-text Propositions 4, 5, 6, 2 and 7.
   - In the full build they will clash (duplicate-label warnings), and `\ref` will resolve to whichever is defined last, i.e. the appendix A-numbers.
   - The appendix labels should be renamed (e.g. `prop:A-info`).
2. **Measurement and flexible-input formulas lack an appendix proof.** The text says "derivation in Online Appendix D" for equation `eq:measbias`.
   - m8's Propositions M and F are derived and verified in the m8 memo (§1.3–1.4; the review re-derived both) but are not in `appendix_proofs.tex`.
   - The integrator (or the appendix_additional writer) must add a short derivation, or change the pointer to the replication package.
3. **Figure 4 legend says "q free".** The paper's notation is κ, and the figure note says "(labeled q in the legend)".
   - Better: regenerate `m2_fig3_sigma_forest` with κ via `code/analysis/m2_techpanel/m2_out.py` (plot labels only).
   - The x-axis label also says "Huber-LSE"; that is fine.
4. **Table numbering.** In the isolated test the tables show as 1–3 and the figures as 1–3. In the full paper they become Tables 3–5 and Figures 3, 4 and 9, provided the earlier sections define Tables 1–2 (`tab:dictionary`, `tab:data`) and Figures 1–2.
   - Figure 9 (`fig:experiment`) is defined in this section. If Section IV floats appear before Figures 5–8 (Sections V–VI), LaTeX will number the placeholder "Figure 5", not 9.
   - Consider moving the m9 figure environment to wherever Figure 9 should appear, or accept sequential numbering.
5. **Chinchilla SEs differ slightly across modules.** Table 3 uses m1's bootstrap (α s.e. 0.016); Table 4's Chinchilla row uses m2's (0.015). These are the same data with different bootstrap seeds. The point estimates are identical; the difference is harmless but a sharp-eyed referee might notice. Either keep it (and footnote it) or overwrite Table 4's Chinchilla SEs with m1's.
6. **Plan vs memo discrepancies** (the memo wins):
   - The plan says the cluster bootstrap "doubles SEs". The memo gives 1.5–2.2×, and the text uses that.
   - The plan says "σ*_q 0.51–0.71" and "Farseer three estimators 0.69–0.71"; both are confirmed.
   - The plan says "rank-one not rejected". This is confirmed only with the κ-family Ê; the text says so.
   - The plan says "ACF: Chinchilla extrapolates poorly (RMSE 0.022–0.025 vs 0.0026)"; confirmed.
7. **Omitted from the main text.** Candidates for Online Appendix D or a footnote if wanted:
   - the lab-own IsoFLOP technologies (Llama 3 primal vs Approach-2 slope gap, p ≈ 0.01–0.05; Marin grid/FLOP-accounting decomposition), which Section V uses;
   - the KMW normalization (condition number 2,003 → 62);
   - the "local σ falls with D/N" pattern (suggestive only);
   - the n = 245 results;
   - the Step Law policy shifts (≤ 0.74 SE).
8. **Meta Llama 3 citation in the Figure 3 notes** is `grattafiori2024llama` (the Llama 3 report). The digitization is `czech2026llama3isoflop`, if the integrator wants to credit it.
9. **Section length** is about 3,450 words, above the 3,000 target (see the top of this file for cut candidates).
