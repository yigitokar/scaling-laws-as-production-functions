# Writer notes (v2): Section IV "What Over-Training Reveals" (wedge)

Writer: Claude, 2026-09-24. Files written:
- `paper/sections/wedge.tex` (Section IV; replaces the v1 text, which remains in `paper/sections/v1/wedge.tex`)
- `paper/tables/table3_wedge.tex` (Table 3; condensed from `output/tables/ra2_wedge_models_selected.tex`)

Exhibits used: Figure 4 = `ra2_wedge_ecdf` (label `fig:wedge`); Table 3 = `tables/table3_wedge` (label `tab:wedge`);
Figure 5 = `ra1_modelfree_farseer_w` (label `fig:farseer`). The optional trend figure was not added (exhibit budget).

Test compile: `bash code/paper/test_section.sh wedge` builds with no errors and no overfull boxes. The only warnings are
undefined references to labels that live in other sections (listed below). Pages rendered with pymupdf and inspected.
Table 3 fits `\textwidth` without scaling (385.52pt against 385.53pt at `\tabcolsep` 1.9pt). A box-measure safety net
scales it only if it is still too wide.

Length: about 3,290 words of body text (plan: about 3,000), plus about 230 words of figure notes and 140 of table notes.

## Labels defined here
- Section and subsections: `sec:wedge`, `sec:wedge-measure` (A), `sec:wedge-data` (B), `sec:wedge-results` (C),
  `sec:wedge-pi` (D), `sec:wedge-conduct` (E), `sec:wedge-trends` (F).
- Equation: `eq:sIV-inv`. This is the inversion: ln w = (1/σ* − 1) ln(M/M*(C)) and s = 1 − (M/M*)^{−(1/σ*−1)}.
- Exhibits: `fig:wedge` (Figure 4, ECDF), `tab:wedge` (Table 3), `fig:farseer` (Figure 5).
- v1 labels that no longer exist: `sec:wedge-intent`, `sec:wedge-family`, `sec:wedge-validation`, `sec:wedge-rivals`,
  `sec:wedge-aggregate`, `fig:trends`, `tab:validation`, `eq:sV-foc`, `eq:sV-logw`, `eq:sV-rel`, `eq:sV-agg`. The
  introduction and conclusion writers must not reference them.

## External labels referenced (must exist in the integrated paper)
- `sec:tech`, `sec:econ`; `prop:wedge`, `prop:modelfree`, `prop:pi`, `lemma:sigma` (from Sections I and II)
- `cor:suff`, `prop:A-wedge`, `prop:A-family`, `rem:A-conduct`, `rem:A-pi-param` (already in `appendix_proofs.tex`)
- `app:data` (Appendix B), `app:wedge` (Appendix F); see integrator issue 1.

## Numbers used, with sources (every number checked against the CSV, not only the memo)
Abbreviations: T = `output/tables/`; P = `data/processed/ra2_wedge/`.

**IV.A**
- R² 0.9999 of ln ŵ (reference) on ln M in the clean sample. Recomputed from T `ra2_wedge_models.csv` (clean == True,
  `w_chin_q`): slope 0.4264, R² 0.99994, Spearman 0.9998 (memo H11).
- T/D at the median: 9.0 at p = 1, 0.9 at p = 10. T `ra2_wedge_costsens.csv`, row δ = 0, η = 1: `median_TD_p1` 8.96,
  `median_TD_p10` 0.896.

**IV.B**
- 173 → 77 models, 36 families (the `gen` field), 18 developers, 2023–2025; nine steps. T `ra2_wedge_cleaning.csv`
  (rows start … h_D_undocumented). Checked: `models.csv` clean n = 77, `gen` has 36 unique values, `dev` 18, years
  2023–2025.
- The median wedge along the cleaning steps stays between 3.77 and 4.46. T `ra2_wedge_cleaning.csv` `median_w`: min
  3.767 (start), max 4.457 (b2).
- 32 ex-ante technologies. T `ra2_wedge_technologies.csv`, `in_set` True: 32 rows.
- Reference σ*_κ = 0.701 (0.70055) and M*(10^21) = 22.7 (22.67). Same file, row `chin_q`.
- κ = 1 median 3.34 against 3.99. Same file, rows `chin` (3.336) and `chin_q` (3.986); `cleaning.csv` `median_w_kappa1`.
- Meta σ* = 0.660 and Marin Nemotron-CC σ* = 0.705. Same file, rows `meta_mf` (0.6595) and `marin_nemotron_mf` (0.7049).
  These are ra1's random-effects values.
- 22 lab-own models; 11 contemporaneous and 11 ex post (Llama 1/2, OLMo 1/1.7, Marin 8B). T `ra2_wedge_labown.csv`,
  `lab_vintage`.
- Embedding share at most 28 percent in the clean sample. `models.csv` `emb_share`: max 0.277, min 0.005.

**IV.C**
- Median ŵ 3.99; median s 0.75 [0.69, 0.79]; share ŵ > 1 = 97 percent. T `ra2_wedge_technologies.csv` row `chin_q`:
  `med_w` 3.986, `med_s` 0.749 [`med_s_lo` 0.689, `med_s_hi` 0.794], `share_gt1` 0.974. Joint wild bootstrap, B = 399.
- "A model one percent smaller at the same quality is worth three percent of its training cost": w − 1 = m_N = 2.99 at
  the median (Proposition A8(i)).
- Range of the median s over the 32 technologies is 0.17 (`minicpm`) to 0.85 (`muen_q`). The κ-free or model-free
  curvature set gives 0.69–0.79: chin_q 0.749, farseer_q 0.744, meta_mf 0.791, marin_*_mf 0.688/0.731/0.736, deepseek
  0.781. Marin A2 σ* is 0.87–0.90 (0.904/0.866/0.867). MiniCPM M*(10^21) = 191.87. The small-sweep κ-free σ*_κ is
  0.51–0.62 (muen_q 0.511, olmo_q 0.544, gadre_* 0.594/0.607/0.623), with medians of s up to 0.854.
- Point estimates above one under all 32 technologies: 64 percent. Union band above one: 18 percent. `models.csv` clean:
  `pt_min > 1` 0.636 and `band_lo > 1` 0.182 (also `cleaning.csv`).
- Curvature sensitivity at the reference zero point. T `ra2_wedge_sigma_sensitivity.csv`: σ* 0.80 gives a median of 2.245
  and s 0.555; σ* 0.60 gives 8.643 and s 0.884; `share_gt1` is 0.974 in every row.
- Lab-own subsample median s: 0.61 lab-own against 0.65 reference. `ra2_wedge_labown.csv`: median `s_lab` 0.606,
  median `s_ref` 0.646.
- Llama 3 8B lab-own ŵ 8.36 [5.67, 12.76], s 0.88, alternatives 3.09–12.16. Llama 3.1 405B lab-own 0.968. Same file.
  The alternatives' extremes are `meta_a2` (κ = 1 law) and `meta_mf8` (path on ra1's 8 bracketed budgets), per memo H3.
- Table 3: every cell comes from `models.csv` (columns listed in the table file's header comment). Panel B uses
  `wtotalN_chin_q` (total N) and `w_chin_q` (active N), with s = 1 − 1/w recomputed. The Serves column is `serve`.

**IV.D**
- 47 of 77 have M > 341; median s is 0.56 inside and 0.85 outside; 0 percent inside have the band above one; 2 models
  are inside both M and C. T `ra2_wedge_insupport.csv` (rows "Chinchilla M range only" and "Chinchilla design").
- Farseer M range up to 2,570 (non-embedding). `ra2_wedge_extrapolation_check.csv` `M_max` 2569.7.
- Local w at M ≥ 1,024 is 7.48 (ra1 memo H4, table `ra1_modelfree_farseer_w.tex`).
- Ratios w_param/w_local at M ≥ 1,024 (non-embedding). From T `ra1_modelfree_farseer_delta.csv`: κ free 0.701
  (`kappa_full`), Chinchilla 0.440 (`chin_full`), Chinchilla fitted on M ≤ 100 0.257 (`chin_M100`). At the
  extended-grid CV bandwidth (ra1 memo H4 and review M-A): 0.78, 0.50, 0.30. Hence 0.70–0.78, 0.44–0.50 and 0.26–0.30.
- Linearity rejected, b₂ = 0.020 (SE 0.001). T `ra1_modelfree_farseer_lin.csv` (cited qualitatively in the text).
- 57 clean models inside Farseer's M range: median s moves from 0.68 to 0.72 (primary bandwidth) or 0.71 (CV
  bandwidth). T `ra2_wedge_extrapolation_check.csv`, rows `farseer_q`: 0.677 → 0.719 / 0.713. 20 models lie outside.
- Smallest sizes at the corner, 0.10–0.34B. ra1 memo C6 caveat.
- 88 percent of the clean sample has C above 1.3×10^22. Recomputed from `models.csv` `Cmp`: 0.883.
- Anchors. T `ra2_wedge_pi_anchors.csv`, kind `iso`: Chinchilla exp(3.1206) = 22.7 at 3e21; Meta exp(3.1057) = 22.3 at
  1e22; Marin exp(2.2424 / 2.2994 / 2.5185) = 9.4 / 10.0 / 12.4 at 3e20; DeepSeek published law.
- e ∈ [−0.16, 0.19]. T `ra2_wedge_pi_summary.csv` `e_lo` −0.156, `e_hi` 0.190.
- M*(10^24): PI-1 [2.26, 88.8]; Meta anchor [10.1, 57.0]. T `ra2_wedge_pi_mstar.csv`.
- Sign identified: 86 / 77 / 23 percent (PI-1 / PI-3 / PI-4); no w < 1 identified. T `ra2_wedge_pi_summary.csv`
  (0.857 / 0.766 / 0.234; `share_w_lt1_identified` 0 everywhere).
- 11 ambiguous models under PI-1, all with M ≤ 96: Llama 2 70B, LLaMA 30B/65B, Llama 3.1 405B, Qwen-72B, Qwen2-72B,
  Yi-34B, Falcon 40B/180B, MPT-30B, DeepSeek LLM 67B. P `pi_bounds_models.csv` (set PI-1, sign == "ambiguous"; max M
  96.28).
- Curvature range 0.40–0.52 and the medians of the per-model s bounds, 0.59 and 0.93. `pi_summary.csv` `median_s_lo`
  0.589, `median_s_hi` 0.933; range from the `ra2_wedge_pi.tex` note.

**IV.E**
- Open-weight premium 0.52 (CRV1 SE 0.18), WCR p = 0.011, n = 186, 67 developers. T `ra2_wedge_conduct.csv` row 2
  (0.517, p_wcr 0.0115).
- Serving 0.45 (0.26), p = 0.39, 18 clusters, 7 treated. Same file, row 5 (0.453, 0.257, 0.3903, G_treated 7).
- On-device carried by two developers. Row 6 (G_treated 2).
- Bunching: 51 percent against 13 percent, n = 220. T `ra2_wedge_bunching.csv` (`share_in_windows` 0.509,
  `share_pred` 0.134).
- Placebo p-values 0.37–1.00. `conduct.csv` rows 21–24 (`p_placebo_tierstory` 0.610, 0.997, 0.369, 0.589).
- Serving split: 33 models (7 developers), median s 0.80, all ŵ > 1; 44 models (12 developers), 0.74. T
  `ra2_wedge_serve_split.csv` (0.798; 0.737). Developer lists come from `models.csv` `serve`. Meta appears in both groups:
  Llama 1/2 predate Meta AI.
- Common-D: 32 models in 11 families; median ŵ 3.90 against 4.01 (size-specific); excluding common-D, 45 models with
  median s 0.75. T `ra2_wedge_family_split.csv`.
- Family-level s_f from 0.18 (DeepSeek LLM) to 0.89 (Qwen3). T `ra2_wedge_family_level_W.csv` (0.1799, 0.8904).
- Cost sensitivity: s 0.72–0.77 for δ ∈ [−0.1, 0.1] (with η = 1 + δ); 0.70–0.86 for serving-cost elasticity
  φ ∈ [0.5, 1.25]. T `ra2_wedge_costsens.csv` (0.724–0.774; 0.705–0.857). The ra2 memo's "0.71–0.86" is a rounding
  slip: the CSV gives 0.7049. The costsens file calls this elasticity η; the text uses φ, as in Proposition A8(iii)(a).
- Synthetic-data robustness: 68 models, median s 0.74. `cleaning.csv` row `r_synthetic` (0.737).

**IV.F**
- OpenRouter: 0.04–4.6 percent. T `ra2_wedge_validation_openrouter.csv` `ratio` (0.00037–0.0459).
- HF derivatives: 0.80, p = 0.031, 74 models, 16 developers. T `ra2_wedge_validation_hf.csv` row `hf_total`/`lnM`
  (0.795, 0.0312).
- Trend: 0.04 / 0.35 / 0.54 / 0.82 on the 141-model universe. T `ra3_econ_inference_share.csv` (`s_ref` 0.0393, 0.3478,
  0.5353, 0.8191; n 8+53+56+24 = 141), identical to `ra2_wedge_aggregate.csv` (universe rows). 2025 technology range
  [0.24, 0.96] (`tech_min` 0.240, `tech_max` 0.956).
- 2024: the 405B has 39 percent of compute (`top_share_C` 0.388); without it the share is 0.63 (`loo` 0.629).
- 2019–22: OPT-175B and GLM-130B have 92 percent of compute. ra3 memo H6; checked as 0.462 (OPT, CSV) + 3.12e23/6.82e23
  (GLM) ≈ 0.92.

## Referee comments addressed in this section
- **R1 c1** (conduct): the estimand is stated as s under Proposition 2 and interpreted by who serves (IV.A, IV.E). Three
  conduct predictions are tested (open vs closed, serving footprint, tiers), and the split by serving footprint is
  reported. "Planned serving expenditure" is confined to developers that serve. Durability is not modeled (open).
- **R1 c2a/c2b, R2 Major 5a/5c, R3 M5.5**: multi-tier bunching and placebo menus; common-D families treated as
  family-level first-order conditions (Proposition A9), with family-level shares and the headline excluding them.
- **R1 c3** (DLW critiques): factor bias and the output-concept mismatch are named as rivals with known signs. Lab-own
  technologies and published lab laws (DeepSeek, MiniCPM) are in the set. The Raval second-margin test was not run (open).
- **R1 c4a–f, R2 Major 1, R2 Major 6a, R3 M4, R4 M2**: κ-free reference; ex-ante set of 32 technologies, including
  lab laws and κ-free members; ECDFs by technology; partial identification of M*(C) with sign shares by assumption set;
  ranges instead of pseudo-precise intervals; the κ = 1 comparison stated (3.34 against 3.99).
- **R1 c5, R3 M5.2**: cost-side sensitivity (δ, φ, p).
- **R1 c7.2–7.3, R2 Major 3**: in-support against extrapolated splits; the Farseer model-free wedge against parametric
  extrapolation (Figure 5); linearity rejected. R2 Major 3(ii), perceived against true technology, is stated as a limit,
  not resolved.
- **R1 9a, 9f**: design-conditional wild bootstrap for the reference; joint intervals for the median and the share.
- **R1 minor 3**: the 405B on Meta's path is "by construction".
- **R1 minors 24–25, R2 Major 4b–d, R2 Major 10**: MoE excluded, with bounds in Table 3 Panel B; distilled, pruned,
  multimodal and continued-pretraining models excluded; cleaning documented (Appendix F).
- **R1 minor 37**: vintage of the lab-own laws (11 of 22 ex post).
- **R1 minor 39, R2 Major 11, R3 M3c**: OpenRouter access documented (ToS); HF model tree; comparison with aggregates
  deferred to Section V.
- **R1 minor 40, R2 minor 23**: aggregate truncated at w ≥ 1, with a leave-one-out for the 405B and a technology range.
- **R1 minor 41**: the interval in Table 3 is labeled as one technology's interval, with the band next to it.
- **R2 Major 2a–c, R3 M3b**: s is the headline; T appears only with p; internal N-proportional compute is part of m_N.
- **R2 Major 2b, R3 M5.1**: the open-weight premium is read against the serving-cost model.
- **R2 Major 4a**: conventions (own convention per technology; head FLOPs). Details are in Appendix F.
- **R2 Major 5d, R3 M5.4**: latency (Bian et al.) and test-time compute (Roberts et al.) are cited.
- **R2 minors 18, 22, 28–29, 34**: rank = rank of M; synthetic-data robustness row; MoE bounds; per-model file in the
  replication package.
- **R3 M3a**: the object is stated as a monotone transformation of M/M*(C), with R² 0.9999.
- **R3 M3d**: lab-own technologies primary.
- **R3 M3e, M12**: ECDF exhibit replaces the w-against-M figure.
- **R3 minor 35, R4**: at most one interval per sentence; result first.
- **R4 M1**: Chinchilla's largest run is 1.3×10^22 FLOP.
- **R4 minor 33**: tier windows are described as quantized windows; there is no "16–24 GB" claim.
- **R2 Major 6b, R1 minor 36**: the v1 stated-intent subsection is dropped.

## Do-not-claim compliance
- T appears in tokens only with p (IV.A: T/D at p = 1 and p = 10).
- "Planned serving" is used only for developers that serve. For others the text says "internalized value of
  compactness".
- w levels at frontier scale are not claimed as identified: "a sign, not a level".
- No "LaLonde", "Nerlove spurious", "within 4e-4" or "exactly 2E[ln w]/(α+β)"; no "gross complements" as a finding.
- The Farseer direction is stated as "conservative relative to the true technology", with the believed-technology
  caveat from Remark A-pi-param. The v1 phrasing "extrapolated T̂ is conservative" is not used unconditionally.

## Open issues (for the integrator and lead author)
1. **Label clash: `app:wedge`.** `appendix_proofs.tex` line 777 labels its subsection "What over-training reveals: the
   generalized wedge" as `\label{app:wedge}`. The assignment reserves `app:wedge` for Online Appendix F
   (`appendix_wedge.tex`). This section refers to `app:wedge` three times, meaning Appendix F. Rename the proofs
   subsection label (e.g., `app:A-wedge-sub`) and update any `\ref{app:wedge}` inside `appendix_proofs.tex` that means
   the proofs subsection.
2. **Bibliography.** `aubakirova2026state` is an `@misc` entry with a `journal` field, so aea.bst prints no venue.
   Change it to `@article` (journal = {arXiv preprint arXiv:2601.10088}) or add `howpublished` in
   `lit/bib/extra_ra2_wedge.bib`. No new references were needed for this section, so there is no
   `extra_writing2_wedge.bib`.
3. **Obsolete table.** `paper/tables/table6_wedge.tex` (v1) is no longer used; Table 3 is `tables/table3_wedge.tex`.
4. **m9 placeholder** (IV.D, one sentence): the result of pre-registered question Q3, the extrapolation of the Chinchilla
   form fitted on M ≤ 100 against the model-free local wedge. Fill X (the largest non-embedding M reached) and Y.
5. **Appendix F** must contain the objects the text points to: the cleaning table with sources, the list of 32
   technologies, conventions (including the sub-1B like-for-like comparison: Farseer −36 percent, Chinchilla −8
   percent), the PI tables (`ra2_wedge_pi.tex`), the conduct table (`ra2_wedge_conduct.tex`), cost sensitivity,
   families, validation, and the lab-own alternatives. The appendix_wedge writer should use these numbers.
6. **Section V overlap.** IV.F reports the trend in the aggregate share and defers the comparison with fleet
   disclosures (Patterson et al.; Wu et al.) to Section V. Section V should use the same numbers: 0.04 / 0.35 / 0.54 /
   0.82 on the 141-model universe; 2025 range [0.24, 0.96].
7. **Substantive open items carried from ra2** (not resolvable in writing): the Raval second-margin test; tokenizer or
   byte normalization of D; durability of T and test-time compute as a separate measurement; the lab-own Meta path uses
   two unbracketed budgets (the 8B's range 3.1–12.2 is stated); the serving-footprint coding of phi-1.5 and 01.AI is a
   judgment call; the Farseer direction comes from one recipe, and the believed-versus-true point (R2 Major 3(ii)) is
   unresolved.
8. **Introduction and abstract writers.** Robust claims from this section: 86 percent sign-identified; the rise from
   0.04 to 0.82; median s 0.75 conditional on the reference, with a range of 0.17–0.85 across technologies and 0.69–0.79
   for the κ-free and model-free set. The lab-own median is 0.61 on 22 models.
