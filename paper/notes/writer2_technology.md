# Writer notes: Section III, "The Technology from Designed Variation" (version 2)

Writer: section writer "technology", 2026-09-24. Plan: `paper/notes/paper_plan_v2.md` III.A–E.

## Files written
- `paper/sections/technology.tex`: Section III (label `sec:tech`), subsections A–E.
- `paper/tables/table2_designs.tex`: Table 2 (label `tab:designs`). Panels A, B and C.
- `lit/bib/extra_writing2_technology.bib`: two new, verified entries, `olken2015promises` and `merity2016pointer`.
- The v1 section is kept at `paper/sections/v1/technology.tex`. The v1 tables (`table3_chinchilla`, `table4_technology`,
  `table5_measurement`) are no longer `\input` by this section.

## Size, compile status and labels
- **Length.** About 3,100 words of text, excluding the table, the figure notes and the TBD-m9 placeholders. The plan's
  target is 2,800. The experiment subsection is about 750 words before its results arrive.
- **Test compile** (`code/paper/test_section.sh technology`):
  - No errors and no overfull boxes.
  - The only undefined references point to labels in other sections: `sec:ident`, `sec:framework`, `sec:wedge`,
    `sec:econ`, `prop:modelfree`, `prop:ident`, `prop:wedge`, `lemma:sigma`.
  - The two new citations are undefined until `references.bib` is rebuilt.
  - Compiled together with `appendix_proofs` and `appendix_additional`, `rem:A-grid` and `app:additional` resolve.
- **Labels defined:**
  - `sec:tech`;
  - `sec:tech:designs` (compatibility alias `sec:tech:chinchilla`);
  - `sec:tech:modelfree`;
  - `sec:tech:parametric` (alias `sec:tech:sweeps`);
  - `sec:tech:portable` (aliases `sec:tech:quality`, `sec:tech:measurement`);
  - `sec:tech:experiment`;
  - `eq:tech-mf` (equation 1 of the section, Proposition 4(ii));
  - `tab:designs`, `fig:sigma` (Figure 3 = `ra1_modelfree_sigma_by_design`) and `fig:experiment` (placeholder).
- **Labels used from theory v2 §0.** `prop:modelfree` items (ii), (iii) and (iv) are cited by item number, so the
  identification writer must keep that numbering. Also `prop:ident`, `prop:wedge`, `lemma:sigma` and, in the
  appendix, `rem:A-grid`.

## Numbers used, with sources
Every number was checked against the CSV. Rounding is to the precision shown.

### Designs and Table 2
| Number in text or table | Source |
|---|---|
| Chinchilla: 245 runs, FLOP 1.40e18–1.30e22; 137 on 9 profiles (budgets 6e18–3e21); 108 off-profile in 35 trunks; baseline n = 240 | ra1 memo H5; `ra1_modelfree_chinchilla_design.csv`; `ra1_common.load_chinchilla_all` (recomputed: 137 iso) |
| Llama 3: 133 runs, 10 budgets 6e18–1e22 (8 valid) | `ra1_modelfree_isoflop_summary.csv` (k_valid 8, k_budgets 10) |
| Marin: 85/85/88 runs, 7/7/8 budgets, up to 3e20 | same |
| Porian: 121/116 points, 16 architectures, 12 budgets 1.25e16–2.56e19 | same; ra1 memo 2.2 |
| Farseer: 404 released runs of about 1,000 trained | m2 memo 2.1; R4 minor 19 |
| Small sweeps: Gadre 34/35/35, OLMo 30, Muennighoff 33 | `m2_table3_technology.csv` n_runs |
| N, M ranges and off-path spreads (Panel A: 1.67, 1.38, 2.46, 2.46, 2.42, 2.29, 2.20, 1.97; Panel B: 1.35, 1.46, 1.47, 1.47, 0.94, 1.97) | Recomputed with the ra1 loaders (`ra1_common.isoflop_designs`, `sweep_panels`): scratch script, OLS residual s.d. of ln M on ln C with nominal budgets for the IsoFLOP designs. Sweep spreads equal m2's `sd_offpath` |
| Model-free σ*: 0.673 (0.027), 0.660 (0.023), 0.700 (0.027), 0.713 (0.027), 0.705 (0.023), 0.518 (0.017), 0.505 (0.014) | `ra1_modelfree_isoflop_summary.csv` (`sigma_re`, `se_sigma_re`) |
| Same-run κ free / κ = 1: 0.667 (0.023) / 0.692 (0.013); 0.693 (0.020) / 0.769 (0.006); 0.678 / 0.690; 0.663 / 0.673; 0.677 / 0.684; 0.345 (0.141) / 0.599 (0.032); 0.347 (0.160) / 0.609 (0.039) | `ra1_modelfree_isoflop_param.csv` |
| Farseer model-free 0.708, and 0.701–0.711 across bandwidths | `ra1_modelfree_farseer_sigma_slope.csv` (0.7078; 0.7008–0.7114) |
| Farseer Hessian path 0.703 (0.013) | `ra1_modelfree_heterogeneity_inputs_modelfree.csv` |
| Farseer κ free 0.710 (0.003) and κ = 1 0.772 (0.010); Panel B values and SEs | `m2_table3_technology.csv` (Huber rows: `sigma_star_q`, `se_sigma_star_q`, `sigma_star`, `se_sigma_star`) |
| Panel C: 0.695 [0.673, 0.717], between-design s.d. 0, I² 0.00, Q 4.1 (p 0.54); 0.646 [0.562, 0.730], 0.098, 0.96, 188.9; 0.672 [0.636, 0.707], 0.019, 0.47, 11.4 (0.077); 0.622 [0.549, 0.695], 0.068, 0.93, 82.1; 0.777 [0.745, 0.809], 0.030, 0.83, 36.3 | `ra1_modelfree_heterogeneity.csv`, rows "all technologies" |

### III.B, model-free σ*
| Number | Source |
|---|---|
| One estimate per study: 0.693 | heterogeneity CSV, row 44 |
| Low power below a between-design s.d. of about 0.02 | ra1 memo H3 |
| Farseer Hessian path at the cross-validated bandwidth 0.664; mean 0.679 | ra1 memo H4 and the Table 2 note; heterogeneity row 45 (0.679) |
| Finite-grid bias ≤ 0.006; Monte Carlo coverage 0.92 / 0.98 / 0.93 | ra1 memo H2 and 5.2; `ra1_modelfree_isoflop_mc.csv` |
| FLOP-accounting shift ≈ 0.44η; ±0.04 for \|η\| ≤ 0.1 | ra1 memo §7.2 |
| Marin configuration-N values 0.658 / 0.642 / 0.655 (quoted as 0.64–0.66); η = 0.087 | ra1 memo 5.3 |
| Porian window sensitivity 0.45–0.55 | ra1 memo 5.3 table (h 0.8–1.5) |
| Porian included: mean 0.646 | heterogeneity CSV |
| Drift per decade: −0.072 (0.026), −0.053 (0.010), −0.015 (0.008); Marin p ≥ 0.69 | isoflop_summary; ra1 memo H4 (Farseer path) |
| Chinchilla drift across windows: 0.034–0.096 per decade | ra1 memo C5 |
| Over-training overhead: 73% at σ* = 0.70 and 57% at 0.74 (k = 10); w = 2.68 and s = 0.627 at k = 10, σ* = 0.70 | `ra1_modelfree_practitioner.csv` |

### III.C, parametric forms
| Number | Source |
|---|---|
| κ̂ = 0.774; κ = 1 rejected at 1% in 5 of 6 schemes; budgets + trunks p = 0.004; original nine clusters p = 0.052 | ra1 memo H5 table |
| κ = 1 rejected in all seven sweep–corpus technologies (p < 0.001) | m2 memo H1b |
| Chinchilla profile-run QLR 7.8; other IsoFLOP designs 72–133 | `ra1_modelfree_isoflop_param.csv` (`qlr_kappa1`) |
| Llama 3: 4.8 SE and 1.5 SE | ra1 memo C3 (review-corrected values) |
| Calibrated κ-free profile set [0.68, 0.72]; σ* = 0.74 rejected at p = 0.003 (floor, B = 299) | ra1 memo H5 |
| CES on Chinchilla: robust p-values 0.06–0.40 (HC1 0.060, HC3 0.089, calibrated Gaussian LR 0.17, Koenker–Bassett 0.27, Laplace QLR 0.40); sweep Wald p 0.22–0.90 | `ra1_modelfree_chinchilla_tests.csv`; m2 memo H1d. The main text now says only "survives at 5 percent in every sweep under tests robust to heavy-tailed residuals"; the details go to Appendix D |
| Pairwise: Chinchilla–Farseer p = 0.51; z from 3.3 to 7.8 | ra1 memo H3 |
| Leave out OLMo: heterogeneity survives (Q = 22.3) | heterogeneity CSV row 6 |
| E–κ trade-off: Gadre RedPajama moves 0.08 (0.606–0.685); Muennighoff 0.05 (0.503–0.555); at most 0.006 on Chinchilla, Farseer and OLMo | `ra1_modelfree_kappa_robustness.csv` (E-set columns) |
| Dropping the smallest sizes moves Gadre by 0.05–0.11 | same file |
| OLMo with task bits per byte: 0.512 | same file |
| The Chinchilla form raises σ* by 0.036–0.281, to 0.735–0.825 | `m2_table3_technology.csv` |

### III.D, what is not portable
| Number | Source |
|---|---|
| a from 0.37 to 0.57 (seven sweep–corpus technologies) | `m2_table3_technology.csv` |
| a from 0.36 to 0.57 with lab laws and Farseer's own form | `ra3_econ_data_demand.csv` (farseer_eq3 0.357; gadre_rw 0.566) |
| M*(10²¹) from 3.4 to 60, an 18-fold spread | m2 (3.35 and 59.6) |
| Model-free path slopes: 0.50 (Chinchilla, Llama 3); 0.35–0.43 (Marin) | `ra3_econ_data_demand.csv`, `ra1_*` rows |
| Farseer embeddings: a 0.526 → 0.411; M*(10²³) 19.5 → 45; σ*_κ 0.710 → 0.668 | m2 memo H1e; `ra1_modelfree_kappa_robustness.csv` |
| Chinchilla non-embedding a: 0.514 → 0.556 | m8 memo (v1 text) |
| Porian: 975 runs; 0.837 → 0.496 (RefinedWeb); 39–47% measurement, 53–61% flexible inputs | m8 memo (post-review) |
| Gadre: p = 0.90, p = 0.004, M* 8–18% | m2 memo H2 |
| DataDecide: 98.1%, 99.71%, +0.05 points; tilt 0.22–0.26; M* 2.9–3.4×; wedge 1.25–1.29× | m2 memo H2 |
| DeepSeek allocation exponent 0.450 → 0.524 → 0.578 as quality improves | Verified in the arXiv PDF of `bi2024deepseek` (Table 4 and the text below it) |

### III.E, experiment design
| Number | Source |
|---|---|
| Widths 128–640, depth width/64, D 25M–800M with caps, 44 endpoints per corpus | `code/sweep/run_grid.py` |
| N: 0.394M–49.2M non-embedding and 1.44M–54.4M total; M: 0.508–2,031 and 0.459–555; off-path spread 1.961 (1.742 total) | Recomputed from the `gpt_mlx.count_params` formula (scratch script) |
| High-M runs: (128,4) and (256,4) to 3.2B tokens; (128,4) reaches M = 4,063 non-embedding, 1,743 total; the largest run is 1.06e17 FLOP (the (256,4) run at 3.2B) | Same formula |
| WSD, 1−sqrt cooldown over the last 20%, branch at 0.8 D_k, warmup 250, context 256, batch 64 × 256, AdamW wd 0.1, bf16 compute with fp32 master weights, LR rule 3.07e-3 (d/256)^−0.90 | `train_sweep.py`, `gpt_mlx.py`, `m9_spec.md`, pre-analysis plan |
| 1,048,576 evaluation tokens | `m9_spec.md` |
| Warmup is 16% of the 25M runs | 250 × 16,384 / 25M = 16.4% |
| Plan committed at 03:12 on 24 September 2026, before estimation | git commit `bd5c0ad`, 2026-09-24 03:12:31 +0300, "Pre-analysis plan … (committed before estimation)". The plan file is unchanged since (byte-compared with commit `acc8286` and the working tree) |
| Deviation: (256,4) run for FineWeb-Edu only; the hiM2 continuation dropped | Diff of `run_grid.py` between commits `bd5c0ad` and `acc8286` |
| Decision rules for Q2 and Q3; conventions; sample robustness; inference | `m9_preanalysis_plan.md` §2–3 |

## Referee comments addressed in Section III
- **R1**
  - c6(a): heterogeneity test and random-effects summary (Panel C; III.C).
  - c6(b): like objects on Farseer. The path σ* is 0.708 against κ free 0.710.
  - c6(c): the model-free estimator has no pseudo-true value. The same-run comparison is reported, and discrimination
    is claimed only for Llama 3 and Farseer.
  - c6(d): complementarity is stated as implied, with its empirical content (U-shaped profiles).
  - c6(e): the elasticity is between parameters and tokens processed, not unique data; the data wall is in Section V.
  - c7(1): model-free σ* on every IsoFLOP design, including Porian, with finite-grid bias checks.
  - c9(a): design-conditional wild bootstrap, reported in the text and table notes.
  - c9(b): nine-cluster versus budget + trunk clustering.
  - c9(c): calibrated profile set.
  - c11(a–g): experiment design.
    - Neutral set: WikiText-103.
    - Dose–response: **not done**; flagged below.
    - Learning-rate checks at the corners and on FineWeb.
    - Both conventions, with the primary one pre-specified.
    - Inference by width plus a seed-covariance bootstrap.
    - Seeds in both corpora.
    - Common random numbers limited to data order.
    - Power calculation and pre-analysis plan.
    - Model-free w in the experiment.
  - Minors 20 (design statistic), 26–29 and 31 (the tilt as the only ordinally relevant quality channel).
  - Minor 30: the capital–labor comparison is dropped.
- **R2**
  - Major 7(a–f):
    - (a) Four-layer high-M runs; the reach is stated honestly (M ≤ 1,743 total).
    - (b) Learning-rate envelope checks.
    - (c) Seeds at the corners and in both corpora; power calculation.
    - (d) Neutral output; bits per byte.
    - (e) Pre-registration of convention, output, estimator, clustering and extrapolation statistic.
    - (f) Scope stated.
  - Major 8: the narrower evidence on σ; "five studies"; the κ robustness facts; DataDecide out of the headline; the
    over-training cost translation, with the full table in Appendix D.
  - Major 9(a): "IsoFLOP-minima frontier".
  - Major 9(b): FLOP-accounting sensitivity of σ* (η).
  - Minors 2, 11 and 31: cooldown shape and warmup stated.
  - Minor 27: new figure, with κ in the legend.
  - Major 12: DeepSeek cited.
- **R3**
  - M6.1–6.5: complementarity; not a common 0.7 (heterogeneity and RE); κ family as a specification check; "five
    studies, seven sweep–corpus technologies"; the capital–labor band dropped.
  - M9(a–j): experiment design as above; "two corpora", not "labs".
  - E4: pre-analysis plan.
  - M12: Figure 4 panel (b) is gone; the old Table 4 is replaced by Table 2 (σ only; a, γ and M* move to the text and
    the appendix).
  - Minors 15 (context, batch, precision), 20 (the tilt is descriptive) and 35 (result first; one interval per
    sentence).
- **R4**
  - M1(a): correct Chinchilla design description.
  - M3(a–c): one sample definition; DataDecide separate; the "factor of two" contrast removed.
  - M5: point and set from the same (Huber) objective.
  - M6(a): the CES test is summarized; details in Appendix D.
  - M8(d): DeepSeek cited.
  - M10(a–g): experiment design.
  - Minors 5 (width written as d_model), 19 (Farseer 404 of about 1,000), 24 (0.701 used throughout) and 25 (κ
    legend).
  - D4: "σ better determined than a" is explicitly disavowed.

## Open issues and requests to the integrator and other writers
1. **Farseer model-free estimate.** Table 2 and the text report the first-derivative path estimate, 0.708, as the ra1
   review recommends. The random-effects summary (0.695) and Figure 3 use the Hessian-based 0.703 (0.013). No random
   effects have been computed with the first-derivative estimate: ra1 open item 9 asks for an SE that does not treat
   compute levels as independent. Both facts are stated in the table note and the text.
2. **Online Appendix D (`appendix_additional`) must contain** the material this section points to:
   - estimator details and specification sensitivity (`ra1_modelfree_isoflop_sens.csv`);
   - Monte Carlo and the independent re-implementation;
   - the Chinchilla inference-schemes table (`ra1_modelfree_chinchilla_inference.tex`);
   - the κ-robustness table (`ra1_modelfree_kappa_robustness.tex`);
   - the practitioner overhead table (`ra1_modelfree_practitioner.tex`);
   - the CES tests, reporting both the Gaussian LR and the robust tests (R4 M6a);
   - the Porian decomposition (old `table5_measurement.tex`) and the flexible-input formula;
   - the neutrality tables (`appD_neutrality.tex`);
   - the old Table 3 (Chinchilla estimators, duality, selection);
   - the Farseer a / M* robustness.
3. **Cross-section consistency.**
   - Section IV must use the κ-free Chinchilla technology as its reference, as stated here, and must report the
     convexity of ln w on Farseer (III.C points there).
   - Section V must carry a ∈ [0.36, 0.57] and the data-wall result (III.B and III.D point there).
   - The identification writer must keep Proposition 4's items (ii) the σ* identity, (iii) the ln w slope and (iv)
     what κ = 1 does.
4. **Bibliography.** `olken2015promises` and `merity2016pointer` are new, in `lit/bib/extra_writing2_technology.bib`.
   Rebuild `references.bib` with `code/paper/build_bib.py`. Several existing entries render without a venue
   (Bhagia et al., Czech et al.): R4 M9(a), for the bib pass.
5. **Pre-analysis-plan timestamp.** The plan's header says "Written 2026-09-24 03:15 (+03)", but git commit `bd5c0ad`
   is 03:12:31. The text uses the commit time. Someone should correct the header typo, or note it in the m9 memo; the
   committed content is unchanged since.
6. **Experiment deviations not stated in the text.**
   - The (256,4) high-M runs at 200M and 800M tokens replicate main-grid endpoints: same shape and same data order
     with `ext = 1`. m9 could report them as a determinism check.
   - R1's dose–response design (several mixtures) was not adopted; the text says the experiment tests one filter
     through "two corpora".
   - R2's 100–300M-parameter anchor runs were not run.
   - M ≥ 10⁴ in the total-parameter convention (R2) is not reached; the maximum is 1,743.
7. **TBD-m9 placeholders** remain: completed endpoints and power; results for Q1–Q6; the Figure `fig:experiment`
   placeholder and its notes. The power calculation must be dated before the results, as the plan requires.
8. **Length** is about 300 words over target. If the integrator must cut, the candidates are the FLOP-accounting
   paragraph and the Kaplan–Chinchilla sentences of III.D (both covered in Appendix D).
9. **Figure 3 (`ra1_modelfree_sigma_by_design`)** is used unchanged. Its panel-(b) subtitle is set in small type and
   its Farseer panel labels Eq. 3 "gray". Both are legible at textwidth; a font bump would help in print.
