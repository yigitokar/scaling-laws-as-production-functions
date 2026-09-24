# Writer notes: Section VI, "Observational Production Functions and Algorithmic Progress"

Writer: Section VI writer (Claude), 2026-09-24.

## Files

- **Section:** `paper/sections/observational.tex`. Label `sec:obs`; subsections A–C have no labels.
- **Table 8:** `paper/tables/table8_lalonde.tex`, label `tab:lalonde`.
- **Table 9:** `paper/tables/table9_progress.tex`, label `tab:progress`.
- **Figure 7:** `m4_observational_fig5_lalonde`, label `fig:lalonde`.
- **Figure 8:** `m5_progress_dmr_ridge`, label `fig:ridge`.
- **Table generator:** `code/paper/make_tables_observational.py`. It reads only `output/tables/*.csv` and writes both table files. Rerun it if m4 or m5 regenerate.
- **New bib file:** `lit/bib/extra_writing_observational.bib` (see the bib keys section).
- **Test compile:** `bash code/paper/test_section.sh observational` compiles with no LaTeX errors and no overfull boxes. The only undefined items are:
  - cross-references to other sections, main-text propositions and appendix labels;
  - the new citation key `zellers2019hellaswag`, which is undefined until the bib is rebuilt.
- **Preamble:** no extra packages are needed. The files use `booktabs` (`\addlinespace`, `\cmidrule`), `amsmath` (`\text`) and `xcolor` (the placeholder).
- **Length:** about 2,500 words of main text (my count, excluding tables, figure notes and the placeholder). The target was ~2,200. If the integrator needs to cut, the best candidates are:
  - the industry Monte Carlo paragraph (A, last paragraph);
  - the Sahal sentence at the end of B;
  - the TFP-dispersion paragraph, which could move to the appendix and leave one sentence.

## Every number used (value → source)

### A. LaLonde benchmark (m4; all checked against the CSVs)

**Sample and design**
- **Sample:** 128 models, 38 families, 22 developers, 2021-03 to 2024-07. Sources: `m4_observational_design_audit.csv` and the m4 memo §2.1.
- **Experimental curvature:** F = 352 (OLMo ladder + OLMo-2) and 70 (Gadre N≥0.1B). Source: `m4_observational_curvature.csv` (351.6 and 70.3).
- **Local θ_D at N = 1B:** 0.58 at D/N = 20 and 0.09 at D/N = 500. Source: m4 memo, headline 2 (the review confirmed these).
- **Observational D/N:** median 213, p90 1,715. Sources: `design_audit.csv` (median_M 213.4) and the m4 memo (p90 = 1,715).
- **Global experimental θ_D:** 0.35–0.51 (0.352, 0.353, 0.441, 0.511). Source: `m4_observational_exp_benchmarks.csv`.
- **Observational θ_D:** 0.19–0.26 (pooled 0.257, family FE 0.189, year FE 0.186), described as 27–63% lower. Sources: m4 memo, headline 3, and `table6b_global_main_long.csv`.
- **Surface:** quadratic, 88 designed runs, R² = 0.994. Source: `headline.json` (surface.R2 0.99420).
- **Hull:** 57 models (30 families, 19 developers), all ≤ 9B. Sources: `table6_lalonde.csv` (n_obs, n_families, n_dev) and the m4 memo ("≤ 9B").

**Compute elasticity, design-matched**
- **OLS θ_C:** 0.328 (0.037) observed vs 0.329 (0.009) experimental. Bias −0.001 (0.038), wild p 0.99 (0.986). Sources: `table6_lalonde.csv` and `table6_lalonde_wildboot.csv`.
- **Six estimators:** θ_C biases −0.029 to +0.037, rounded in the text to "−0.03 to +0.04". Wild p ≥ 0.37 for OLS and the four FE estimators. The OP-style estimator has no wild p (CR1 z ≈ −0.29). Source: same CSVs.
- **Other outputs and surfaces (OLS θ_C bias):**
  - composite −0.018 (0.057);
  - ladder-only surface −0.016 (0.054), 39 models.

  Source: `table6_lalonde_{core,ladderonly}.csv`.
- **OLS 95% CI:** [−0.075, 0.074], about ±23%. Source: m4 memo, headline 4.
- **ARC-C and Winogrande OLS θ_C bias:** −0.098 (0.127) and −0.160 (0.102), i.e. 18–31% attenuation. Source: `table6_lalonde_{arc,wino}.csv` (benchmarks 0.547 and 0.511).
- **Clean subsample (49 models, 16 clusters):** OLS θ_C bias +0.045 (0.023), wild p 0.10 (0.099). Source: `table6_lalonde_clean.csv` and `_clean_wildboot.csv`.

**The N/D split**
- **Family FE:** θ_N bias +0.203 (0.086), wild p 0.052; θ_D bias −0.262 (0.162), wild p 0.023. The text rounds these to 0.20 (0.09; p = 0.05) and 0.26 (0.16; p = 0.02). Source: `wildboot.csv`.
- **Developer FE with the ladder-only surface:** θ_N −0.113, θ_D +0.099. The text says only "reverses the signs". Source: `table6_lalonde_ladderonly.csv`.
- **Within-family SD:** ln D 0.42 vs ln N 1.29; 18 of 38 families have any within-family D variation. Source: `design_audit.csv` (0.4167, 1.2903, 18).
- **Experimental technology on the hull design:** θ_N 0.423 vs θ_D 0.273 (text: 0.42 vs 0.27). Source: `table6_lalonde.csv`, OLS bench.
- **θ_N − θ_D on the full sample of 128 models:** 0.213 (0.079), pooled OLS. Source: `m4_observational_table6b_global_main_long.csv`.

**Behavioral regimes (semi-synthetic)**
- **Bias under each compute rule:** budget 0.000; funding +0.044; target −0.183 (−40%). Source: `m4_observational_regimes_semisynthetic.csv`.
- **Productivity dispersion:** recipe SD(ω) 0.10 (0.099) vs cross-family SD 0.49. The text says "about five times that of the recipes" and does not quote 0.49. Sources: the same CSV and the m4 memo.
- **Scaled biases:** funding ≈ +0.2 and target beyond −0.18 at the observed dispersion. Source: m4 memo, claim 5 (restated by the review).

**IO remedies**
- **Frontier FLOP/$ IV:** F ≤ 3.8 in every specification (overlap sample, developer FE + trend: 3.76). Source: `iv_diagnostics_{main,overlap}.csv`.
  - The main text now says only "step function of calendar time". The "four distinct values" and "unbounded AR sets" were cut for length but remain in the m4 memo.
- **Own-hardware IV:** F = 10.7 (10.67), three values, six identifying models. Sources: `iv_diagnostics_main.csv` and the m4 memo.
- **China × export-control IV (overlap sample, N ≤ 14B):** F = 10.6, θ_C = 0.90 (0.25), 1.7–2.3 times the OLS/FE estimates. Sources: `iv_diagnostics_overlap.csv` (10.56, 0.903, 0.253) and the m4 memo.
- **Dynamic panel:** six three-generation cells. Source: m4 memo, and `headline.json` (global_truth_main.n_triples = 6).
- **Measurement error in compute:** EIV reliability ≥ 0.988. Source: m4 memo, headline 6.

**Productivity dispersion (90–10, compute-equivalent)**
- **Family effects:** 23.7 [9.1, 197]. Sources: `m4_observational_tfp_table.tex` and `tfp_dispersion.csv` (fam_ce_bench 23.69).
- **Excluding code, distilled and synthetic-data families:** 7–9 (7.0 with the experimental θ; 9.1 netted by θ_C).
- **Within developers:** 11.1 [4.4, 18.8], vs Mertens et al.'s 41.
- **Manufacturing comparison:** Syverson's 1.92; LLM dispersion is 3–5 times larger in logs (ln 7/ln 1.92 = 2.98; ln 24/ln 1.92 = 4.87).
- **Gross-output contamination:** Phi and SmolLM rank 1st and 3rd of 38.

  Source for these rows: m4 memo, headline 8 and claims 7–8.

### A (last paragraph). Industry Monte Carlo (m6, claims 6–13; checked against `m6_montecarlo_industry_summary.csv`)

- **Pooled NLS with generation FE, bias in γ:**
  - funding +0.0228 (13% of γ = 0.178);
  - target −0.0511 (29%);
  - flagships only +0.1185 (66%), matching the closed form 0.118.

  The m6 memo says +0.119; the CSV value is 0.1185, and I quote only "66 percent".
- **Within-family (lab × generation FE):** unbiased under every rule; needs at least two siblings per lab-generation.
- **ACF:** works only with within-family instruments.
- **FOC/GNR system imposing w = 1:** ln M* bias +1.25 to +1.28, equal to 2E[ln w]/(α+β) = 1.28.
- **ML practice (E1: pooled Huber-LSE, E free, no time effects):** TFP growth bias −0.081 to −0.090 of 0.150, i.e. 54–60% of progress attributed to the scaling law.

### B. Algorithmic progress (m5; checked against the CSVs and `dmr_summary.json`)

**Replication and optimizer**
- **Replication:** 231 rows, 144 papers; objective 0.0518129; parameters within 4×10⁻⁷; bootstrap median 8.44; point estimate 8.68 (text: 8.7). Sources: `m5_progress_replication.csv` and the m5 memo.
- **Optimizer:** SLSQP stops after 18 iterations. Source: `dmr_summary.json` (slsqp_default_nit.nit = 18; `slsqp_default_stop_iteration` = 17 is the old off-by-one, not used).
- **Converged objective:** 0.0507227 (text: 0.0518 → 0.0507). All 300 random starts end below the published value. Source: m5 memo.
- **Converged T_C:** 6.08, cluster CI [3.05, 22.67]; "twice as wide as published" (published CI [4.5, 14.3]). Source: `replication.csv`.
- **MSE change:** 0.27% (0.046416 → 0.046291). Source: `replication.csv`.
- **NLS T_C:** 10.2, CI [4.1, 27.6]. Source: `table7.csv` (A2).
- **Ridge and profile likelihood:**
  - bootstrap correlation −0.73 to −0.82 (`replication.csv` / `dmr_summary.json`);
  - profile CI [4.1, 40.5] (TC_profile_ci_interp);
  - φ CI [−0.39, 3.37];
  - T_C over the φ grid 5.4–38.3 (text: "5 to 38").
- **Parameter shares:** mean share of the parameter term 0.545 (text: weights 0.55/0.45); on-path share a = 0.37 (a_onpath_Ho = 0.368). Source: `dmr_summary.json` (s_bar, a_onpath_Ho).

**Hicks neutrality and imposed exponents**
- **Neutrality tests:** bootstrap p = 0.29 (converged L1) and 0.94 (NLS). Imposing neutrality gives 11.6 [6.6, 17.6] and 9.2 [6.3, 14.3], i.e. "9–12 months". Source: `dmr_summary.json` and `table7.csv` (A3, A4).
- **Imposed experimental exponents:** MSE ×1.3–4.8 across 14 technologies (0.0605/0.0451 = 1.34; 0.2187/0.0451 = 4.85). Of the 12 registry rows, 9 have g_C < 0 and the other 3 have T_C of 42–122 months. Source: `table7.csv`.
- **Data exponents:** Ho β_data 0.040 vs experiments "about 0.37"; Whitfill ~9×. Source: m5 memo.

**Specification channels**
- **Frontier elasticity:** γ falls 0.178 → 0.0525 when E = 0 is imposed on the Chinchilla sweep; mean total-loss elasticity 0.054 (0.0543). Source: `m5_progress_attenuation_panelB.csv`.
- **Like-for-like gap:** 2–2.5× (Ho γ 0.021–0.025). Source: `table7.csv` / m5 memo.
- **E profile:** T_C moves 10.2 → 12.4 as a common E is imposed from 0 to 1.5 nats per word, and γ doubles. Ho's data prefer E = 0. Source: m5 memo, headline 6.
- **Epochs:** median 62–140 epochs for small-dataset models. Effective data raises T_C 1.5–4× (8.7 → 12.7 with Ho's code; 6.1 → 25.7 converged), CI [3.9, ∞). Sources: m5 memo and `attenuation_panelA.csv` (a3).
- **Sahal bias:** 1.15 [1.10, 1.21] on all rows and 1.44 [1.24, 1.78] on record-setters. Source: `m5_progress_sahal.csv`.

### C. Allocative vs technical (m5 panel B; `m5_progress_table7_panelB.csv`)

- **Sample:** C ≥ 1e23, with 7 models in 2020–21 and 92 in 2022–24.
- **Besiroglu technology:** geometric-mean CE 0.57 → 0.60, gain 1.05 [0.57, 1.72].
- **Other technologies:** Hoffmann TeX 1.91 [1.49, 2.43]; Hoffmann rounded 2.41 [1.92, 3.14]; Farseer 1.58 [1.27, 2.04].
- **Median D/N:** 1.7 → 92.
- **Ho-rate gain:** 8.9. Allocative share 2–40%. With the other Ho estimators the denominator is 22.5 (6.1 months) or 6.4 (10.2 months), from the m5 memo and review §7.
- **Nine sweep technologies:** gain 0.55–4.31; Llama 3 gives 0.74.
- **Robustness rows:** cleaned 1.09 [0.58, 1.74]; top five 1.27 [0.96, 1.77]; Ho technology 0.09.
- **Kaplan-rule counterfactual (Besiroglu):** 1.6 (GPT-3), 5.2 (3.8e25), 11.7 (5e26). Hoffmann TeX: 2.4, 11.5, 32. Source: `m5_progress_kaplan_counterfactual.csv`.
- **Ho technology CE:** 0.098 → 0.009 ("tenfold"). Source: panelB CSV.
- **Gopher:** allocative loss ×2.0 under Besiroglu. Source: m7 / `appendix_proofs.tex` Proposition A (farrell), ln 2.01.

**Discrepancy with the memo:** for "Besiroglu, top five per year" the m5 memo gives the CI as [0.96, 1.78]. The CSV gives [0.961, 1.7749], so Table 9 prints [0.96, 1.77].

## Claims and the caveats attached to them

1. **"No detectable bias" in returns to compute, not "unbiased".** Caveats stated in the text:
   - the CI spans ±23%;
   - ARC-C and Winogrande point estimates imply 18–31% attenuation;
   - the clean-sample bias is +0.045 (wild p 0.10);
   - coverage is limited to models ≤ 9B;
   - the benchmark is the AI2/OpenLM recipe technology;
   - the null is joint with common technology and output mapping.
2. **The N/D split is unreliable: a design/identification problem, not established transmission bias.** This follows the m4 review's weakening. Added reasoning: Hicks-neutral transmission loads on c, so it biases both elasticities in the same direction. That follows from Prop. 6(iv) / appendix Prop. A5(iv) together with the allocation algebra. Biases of opposite sign point to mix-correlated factors (measurement error in D, factor-biased data quality).
3. **Regimes:** the data reject both simulated strong rules; a budget regime or offsetting mechanisms fit. The rules are described as stylized.
4. **IO remedies are infeasible on public data,** framed as a data limitation.
5. **The Monte Carlo is assumed-calibration:** only signs and rankings carry over. The ACF success requires within-family instruments.
6. **Ho replication:** we never say 6.1 months is correct. The point is optimizer and penalty fragility on a flat ridge (m5 review, issue 2).
7. **Profile CI:** the iid assumption is stated; the paper-cluster bootstrap [4.1, 27.6] is quoted alongside (m5 review, §7.1).
8. **Neutrality:** non-rejection reflects low power. If E > 0, neutrality within the E = 0 form is misspecified and overstates progress in simulations (Monte Carlo S1h).
9. **E channel:** Ho's data prefer E = 0, so the argument rests on the sweep and the simulations (m5 review, issue 11).
10. **Allocative gain:** conditional on the technology; even the sign is not robust across sweep technologies. The denominator depends on the Ho estimator. The text also states:
    - cost efficiency counts over-training for inference as waste;
    - era 1 has only seven models, so the comparison is imprecise.
11. **Gundlach's 10×:** a counterfactual at 2025 frontier compute.
12. **"Do not claim" list respected:** no "first" claims; no "w is a markup"; no claim that IO remedies work on public data; no "σ more stable than a". The DMR statement uses the corrected form: time series identify the direction of the bias, not its size.

## Cross-references assumed to exist elsewhere

**Main-text sections:** `sec:ident`, `sec:data`, `sec:framework`, `sec:tech`, `sec:wedge`.

**Main-text propositions:**
- `prop:transmission` (Prop 6): transmission bias, sign = sign of the compute response; allocation exponent unbiased.
- `prop:dmr` (Prop 5): time series identify the sign of the bias, not its size or σ.

**Online Appendix A labels (in `appendix_proofs.tex`):**
- `prop:proxy`: allocation-based proxies (Hicks-neutral ω has no proxy);
- `prop:farrell`: Farrell decomposition;
- `cor:sahal`: frontier-release regressions.

**Online Appendix tables, created by other writers in `paper/tables/`:**
- `tab:app-lalonde-robust`;
- `tab:app-tfp`;
- `tab:app-mc-industry` (included in `appendix_mc.tex`);
- `tab:app-ho-replication`;
- `tab:app-attenuation`;
- `tab:app-ceg-sahal`.

`appendix_additional.tex` is still a stub. The `appD_*` tables must be `\input` there, or these references will be undefined.

**"Online Appendix C"** is written as literal text (it matches `\label{app:mc}` in `appendix_mc.tex`).

**Allocative figure:** the plan puts `m5_progress_fig6_allocative` in the appendix. I did not reference it from the main text. If the appendix writer includes it, a pointer could be added at the start of C.

## Label conflict (integrator must fix)

`appendix_proofs.tex` defines `\label{prop:transmission}`, `\label{prop:dmr}`, `\label{prop:info}`, `\label{prop:wedge}` and `\label{prop:pi}`. These are the same labels the plan assigns to the main-text Propositions 2, 4, 5, 6 and 7. When both are compiled, LaTeX will warn about multiply defined labels, and refs may resolve to the appendix numbers (A-numbered).
- **Intent of my references:** `prop:transmission` and `prop:dmr` should point to the main-text Propositions 6 and 5.
- **Suggested fix:** rename the appendix labels, e.g. `app:prop:transmission`.

## Placeholders

There is one placeholder, at the end of the section:

`\textcolor{red}{[TBD-m9: one sentence on what our two-corpus experiment (Section~\ref{sec:tech}) implies for the sign of the bias of a data-quality improvement, e.g., the shift in compute-optimal $M^*$ at fixed compute.]}`

- **What belongs there:** the m9 estimate of whether FineWeb-Edu vs FineWeb shifts the compute-optimal M* at fixed compute, which by Prop. 5 signs the Hicks bias.
- **If m9 cannot answer it:** delete the sentence.

## New bib keys

- `zellers2019hellaswag`, in `lit/bib/extra_writing_observational.bib`. Verified against the ACL Anthology record P19-1472 (ACL 2019, pp. 4791–4800, doi 10.18653/v1/P19-1472).

All other keys used already exist in `paper/references.bib`:
- ruan2024observational, maiapolo2024sloth, owen2024predictable, mertens2026secret, ho2024algorithmic
- whitfill2025note, konig2026validity, lalonde1986evaluating, dehejia1999causal, gadre2024language
- cameron2008bootstrap, webb2023reworking, magnusson2025datadecide, arellano1991some, blundell1998initial
- olley1996dynamics, levinsohn2003estimating, syverson2011determines, gandhi2020identification, besiroglu2024chinchilla
- diamond1978measurement, hoffmann2022training, knittel2014estimation, dube2012improving, muennighoff2023scaling
- nagy2013statistical, kaplan2020scaling, gundlach2025origin, bhagia2024establishing, teamolmo2024olmo
- epochai2026data

## Open issues for the integrator

1. **Length:** about 2,500 words vs the ~2,200 target. Candidate cuts are listed above.
2. **Duplicate proposition labels** between the main text and `appendix_proofs.tex` (see the label-conflict section).
3. **Table 8 is long** (three panels, nearly a full page at `\footnotesize`). If space is tight:
   - Panel C can be dropped, since it duplicates `tab:app-lalonde-robust` in part;
   - the text statement about the composite and ladder-only surfaces then points to the appendix table.
4. **Table 9, Panel A, row (12)** reports "g_C<0 or 42–122" in merged T_C/CI cells. The details are in the table notes.
5. **Table 9, Panel B, the "Ho et al. (2024)" row** has share −1.10. The meaning (CE fell) is explained in the notes. It could be dropped if it confuses; the text discusses the tenfold fall.
6. **Row numbering:** the rows in Table 9 are labeled (1)–(12), and so are the Table 8 columns. There is no clash, but the integrator may prefer letters.
7. **m5 registry dependence:** Table 9 rows (12) and the "nine sweep technologies" row depend on the m1/m2 technology registries of 2026-09-23. They are recorded in `data/processed/m5_progress/dmr_summary.json` (`registries_used`). If m1/m2 regenerate their registries, rerun:
   - `code/analysis/m5_progress/run.py --only s2,s4,s6`;
   - `code/paper/make_tables_observational.py`.
