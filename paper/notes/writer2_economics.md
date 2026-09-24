# Writer notes (v2): Section V "Economic Implications" and Section VI "Conclusion"

Writer: economics writer (v2), 2026-09-24. Binding inputs: paper_plan_v2.md (V, VI), revision_plan.md, theory_main_text_v2.md,
R1–R4, memos ra3_econ (+ review), ra1_modelfree, ra2_wedge. Every number below was checked against the CSVs in
`output/tables/` or `data/processed/ra3_econ/` (not only the memo).

## Files written

| File | Content |
|---|---|
| `paper/sections/economics.tex` | Section V, `\label{sec:econ}`; subsections `sec:econ-demand`, `sec:econ-wall`, `sec:econ-share`, `sec:econ-growth`. Inputs Table 4 and contains Figure 6. About 1,290 words of main text plus a 64-word footnote. |
| `paper/tables/table4_econ.tex` | Table 4, `\label{tab:econ}` ("Economic Implications under Alternative Technologies"). Built from `output/tables/ra3_econ_table.tex`: headers compressed to fit the 385.5pt text width (tabular*, tabcolsep 2pt); `\item` inside tablenotes removed (AEA.cls tablenotes is a minipage, `\item` would error); notes rewritten without module jargon. One row added: κ free under Muennighoff's full model (40.3%, r_max 9.5, shadow 3.76, σ_CU 0.49), from `wall_grid.csv`/`wall_rmax.csv`/`wall_sigma_CU.csv`. The κ = 1 full-model r_max is shown as 9.1 (ra3 table: 9). Panel C note adds the leave-one-out (R2 23). |
| Figure 6 | `\includegraphics[width=\textwidth]{ra3_econ_figure}`, `\label{fig:econ}`, figurenotes written from `code/analysis/ra3_econ/exhibits.py`. |
| `paper/sections/conclusion.tex` | Section VI, `\label{sec:concl}`; 676 words including 40 words of two TBD-m9 placeholders (limit 700). |

No new references: every key used exists in `paper/references.bib`, so no `lit/bib/extra_writing2_economics.bib` was needed.
Keys used: sevilla2024training, epochai2026data, webb2023reworking, gadre2024language, villalobos2022run, muennighoff2023scaling,
jones2020nonrivalry, reddit2024s1, patterson2022carbon, wu2022sustainable, nvidia2024q4call, you2025openai, denain2026final,
demirer2025emerging, erdil2025gate, snell2024scaling, brown2024large, sutton1991sunk.

Compile: `test_section.sh economics conclusion` compiles with no errors, no overfull boxes, no citation warnings; only
cross-section `\ref`s are undefined in isolation (expected). Pages rendered and inspected (table fits one page with notes).

## Numbers used (value → source)

### Section V
| Number in text | Source (file: row/column) |
|---|---|
| Frontier compute 5.1-fold/yr, 95% CI 4.3–5.9; 92 runs, 27 developers, 2018–2026; snapshot 2026-09-23 | `ra3_econ_compute_growth.csv` top10: growth 5.0618, lo 4.2903, hi 5.9349, n 92, clusters 27 |
| excl. speculative 4.8-fold | same, top10_nsp 4.841 |
| 2018–May 2024: 5.2 vs Epoch 4.2 | same, top10_2024 5.226; `ra3_econ_compute_growth_epoch_published.csv` 4.2 |
| a from 0.36 (Farseer own form) to 0.57 (Gadre RW), ten technologies | `ra3_econ_data_demand.csv` farseer_eq3 0.357, gadre_rw 0.566 (Table 4 Panel A rows) |
| data demand 2.0–2.8×/yr; 34–183× over 5 yrs; "more than a factor of five" | same, gD_hat 2.023/2.836; gD5_hat 33.9/183.4 (ratio 5.4) |
| frontier compute Sept 2026 9.4e26 FLOP | `ra3_econ_compute_growth.csv` C_trend_now 9.398e26 |
| D* at 2026 frontier 9–519T | `ra3_econ_data_demand.csv` Dstar_now_T gadre_rw 9.30, olmo 519.0 |
| stock 100T, 95% interval 22–490T; 320T; median 2028 | Villalobos et al. 2024 (verified in ra3 review §1) |
| four disclosed dense runs above 1e25 (Llama 3.1 405B, Nemotron-4 340B, Pangu Ultra, Aramco Metabrain; 2024–2025) | `data/processed/ra3_econ/frontier_runs_disclosed_D.csv` |
| 100T reached mid-2026 to late 2027 (2026.6–2027.8) across technologies at observed M | `ra3_econ_data_demand.csv` year_unique_obsOT, ten Panel A rows: min 2026.64 (farseer_eq3), max 2027.81 (gadre_rw) |
| 320T reached late 2027 to mid-2029 (2027.8–2029.6) | year_effective_obsOT: 2027.81–2029.58 |
| Chinchilla 5–95%: 2025.7–2029.4 | `ra3_econ_exhaustion_mc.csv` chin, unique, all, obsOT: p5 2025.66, p95 2029.36 |
| a moves dates by about a year | spread of year_unique_obsOT 1.17 yr |
| frontier runs process "about as many tokens as the stock"; 0.6–1.2 times | Dstar_now_T × data_multiple_obs / 100T: 0.56 (gadre_rw) – 1.24 (farseer_eq3) |
| compute-optimal r at 2026 frontier 0.09–5.2 | Dstar_now_T/100: 0.093 (gadre_rw) – 5.19 (olmo) |
| R*_D = 15.4; R*_N = 5.3; data-only 2.9 | Muennighoff et al. App. A / Table 1 (verified in ra3 review) |
| extra compute at r = 4: 7.2 / 7.4 / 7.7% | `wall_grid.csv` C = 1e26, D15: k1 0.07216, kq 0.07383, e60 0.07715 |
| hard wall 117 / 83 / 47 | `ra3_econ_wall_rmax.csv` D15: k1 116.9, kq 83.4, e60 46.96 |
| full model 43% (κ = 1), 40% (κ free), wall r ≈ 9 | `wall_grid.csv` DN r = 4: k1 0.4336, kq 0.4027; `wall_rmax.csv` DN 9.06 / 9.50 |
| σ_CU 0.28 → 0.58 for r 1.5 → 16 | `ra3_econ_wall_sigma_CU.csv` k1 D15: 0.2795 (1.5), 0.5764 (16) |
| γ_eff/γ 0.95 (r = 4), 0.82 (r = 16) | `wall_grid.csv` k1 D15 gamma_ratio 0.9477, 0.8202 |
| shadow value 0.43 × 6N; $3 per million at $1e-18/FLOP | `wall_grid.csv` k1 D15 r = 4: shadow_rel 0.4306, shadow_flop 3.142e12 × $1.015e-18 = $3.19/M |
| $88 per million at r = 16 | shadow_flop 8.710e13 × 1.015e-18 = $88.4/M |
| $1e-18/FLOP = median amortized cost of 2024–25 frontier runs | `ra3_econ_usd_per_flop.csv` (median 1.0154e-18) |
| Table note "one unit ≈ $7 per million" | 6·N_U at r = 4 = 6 × 1.216e12 × 1.015e-18 = $7.4/M |
| Reddit $203 million | `reddit2024s1` (verified in ra3 review) |
| D/D* = w^{σ*/[2(1−σ*)]}; 4.7 (σ* 0.74) vs 2.3 (0.60) at w = 3 | `ra3_econ_wedge_data_multiple.csv` k1 4.654, e60 2.280; closed form checked in `ra3_econ_checks.csv` |
| s aggregate 0.04 (2019–22), 0.54 (2024), 0.82 (2025) | `ra3_econ_inference_share.csv` s_ref 0.0393, 0.5353, 0.8191 (= `ra2_wedge_aggregate.csv`, 16 checks) |
| Google ~3/5; Meta 10:20:70; Nvidia ~40% | `ra3_econ_inference_disclosures.csv` |
| OpenAI 2024: ~a quarter; ~four-fifths | same: 0.2647, 0.7933 (press-reported via Epoch) |
| w − 1 = 4.5 (2025) | `ra3_econ_inference_share.csv` m_ref 4.528 |
| "roughly halves" (vintage factor 0.49, L = 1) | `ra3_econ_inference_share_flow.csv` phi 0.4948 |
| final runs 10–23% of R&D compute (ρ = 4.4–10.4) | `denain2026final` 9.6–22.6% (ra3 review) |
| fleet shares 0.18–0.34 at equal cost per FLOP; disclosed range only if R&D small (0.69 at ρ = 1) | flow CSV, 2025, frontier g, L = 1, p = 1: ρ 10.4 → 0.177, ρ 4.42 → 0.336, ρ 1 → 0.691 |
| 2025 technology range 0.24–0.96 (clean sample, 32 technologies) | `ra3_econ_inference_share.csv` tech_min 0.2404, tech_max 0.9556 |
| 2^{1/γ}: 49 (γ 0.178, κ = 1) to 161 (γ 0.136, Llama 3 A1); 2.4–3.1 years | `ra3_econ_growth_calibration.csv` 48.6 / 160.9; years 2.39 / 3.13 |

Table 4: every cell checked against `ra3_econ_data_demand.csv` (Panel A), `wall_grid.csv`, `ra3_econ_wall_rmax.csv`,
`ra3_econ_wall_sigma_CU.csv` (Panel B), `ra3_econ_inference_share.csv` (Panel C; LOO 2024: s_without_most_influential 0.629,
top_share_C 0.388). Rounding checks: gadre_rw compute-optimal year 2030.554 → 2030.6; kq γ_eff/γ 0.8046 → 0.80; e70 0.807 → 0.81;
D3 0.5552 → 0.56.

### Section VI
| Number | Source |
|---|---|
| information about curvature fourth order in allocation errors | Prop. 3(iii) (theory_main_text_v2) |
| three laboratories' designs agree on about 0.7 | `ra1_modelfree` H2: Chinchilla 0.673, Llama 3 0.660, Marin 0.700/0.713/0.705 |
| median s three-quarters under the reference technology (clean sample) | ra2 H1: median s 0.749 [0.689, 0.794] |
| 0.04 (2019–22) → 0.82 (2025), compute-weighted, open-weight | `ra3_econ_inference_share.csv` |
| model-free RE 0.70, 95% CI 0.67–0.72 | `ra1_modelfree_heterogeneity.csv` / ra1 H3: 0.695 [0.673, 0.717], τ = 0 |
| small-scale values 0.5–0.6, heterogeneous | ra1 H3: σ*_κ 0.511–0.623 across small sweeps (Q = 82.1, τ = 0.068); Porian model-free 0.505–0.518 |
| γ 0.14–0.18 on the large designs | `ra3_econ_growth_calibration.csv`: 0.136 (Llama 3 A1) – 0.178 (Chinchilla κ = 1); Farseer 0.148 |
| sign identified for 86% | `ra2_wedge_pi_summary.csv` PI-1: 0.857 |
| 88% of the clean sample above every design's compute | `ra2_wedge_models.csv` clean (77): Cmp > 1.3e22 for 68 (0.883); largest design budget = Chinchilla 1.3e22 |
| a quarter beyond Farseer's tokens-per-parameter range | `ra2_wedge_insupport.csv`: 20 of 77 outside non-emb. M ≤ 2,570 |
| parametric extrapolation understates w where checkable | ra1 H4 / ra2 H7 (inside Farseer) |
| repetition estimates from models ≤ 9B | ra3 memo claim 4 caveat |
| 50- to 160-fold per halving | growth calibration (48.6–160.9) |

## Referee comments addressed

- **R3 M7(1)** data-demand growth: own compute-growth estimate with few-cluster inference; ten technologies; 2.0–2.8×/yr, 34–183× over five years; exhaustion dates anchored at observed M; link to Villalobos et al. and "a least portable → economically consequential".
- **R3 M7(2)** σ*, M* and the data wall: extra compute and shadow value under κ = 1, κ free and σ* = 0.60 (equivalent member); "0.74 vs 0.70 barely matters" stated with numbers under both repetition models; M*'s role quantified (r 0.09–5.2); nonrivalry (Jones–Tonetti) and licensing (Reddit S-1) connected. Farboodi–Veldkamp not cited (space); could be added as a clause.
- **R3 M7(3)** inference share as a measurement: expressed as s, compute-weighted, with CI and technology range in Table 4 Panel C; conditional reading (serving vs value of compactness); link to Demirer et al.
- **R3 M7 final paragraph** γ in a growth calibration: 2^{1/γ} and years of compute growth; GATE caveat.
- **R3 M3(c) / R2 M3** aggregate levels vs Patterson/Wu (and Nvidia, OpenAI): loose agreement; mapping to fleet shares shows disagreement once R&D compute is counted.
- **R3 minor 29** quantified (γ_eff/γ 0.95/0.82).
- **R1 6(e)** σ for tokens processed vs unique data: compute–unique-token elasticity 0.28–0.58, set by repetition; γ is the growth-model elasticity; σ* enters via r_max and the serving-to-data map.
- **R2 23** leave-one-out and one row per pretraining run: Table 4 note.
- **R3 M1 / minor 28 / minor 30** (conclusion): recommendations cut to two sentences; guidance on choosing γ; Sutton reduced to a clause.
- **R1 6(a), R4 M3, R3 M6** (conclusion): "0.5 to 0.8" replaced by the random-effects summary 0.70 [0.67, 0.72]; σ < 1 described as implied by U-shaped profiles, not a finding (R1 6(d)).
- **R4 D8 / R2 abstract-conclusion headline**: "2.2 times"/"factor of three" removed; s stated as conditional/ordinal with the 86% sign result.
- **R4 M1(ii)**: the v1 recommendation "cluster by IsoFLOP budget" (drawn from data where 45% of runs have no budget) dropped; replaced by design-conditional intervals.

## Open issues / notes for the integrator

1. **Length.** Section V is about 1,290 words of text (+64-word footnote) against the plan's ≈1,000. If cuts are needed: the OpenAI sentence (≈25 words), the licensing clause, and the growth-model subsection (≈90 words, could be two sentences).
2. **Two TBD-m9 placeholders** in the conclusion (paragraph 1 and Limitations). They total 40 words; the replacements must keep the conclusion ≤ 700 words (currently 676).
3. **Section IV overlap.** Section IV.F (plan) quotes the same trend numbers (0.04 → 0.82). The wedge writer can point to Table~\ref{tab:econ}, panel C instead of adding a separate trend exhibit.
4. **Appendix exhibits not placed.** ra3's appendix tables (`ra3_econ_compute_growth.tex`: six frontier definitions; `ra3_econ_inference_disclosures.tex`: disclosures and implied fleet shares) and figure `ra3_econ_inference_share` are not assigned to any appendix in the plan. Section V is self-contained (footnote carries the robustness), but Appendix F would be the natural home if the integrator wants them.
5. **Bibliography.** `erdil2025gate` and `snell2024scaling` render with no venue (R4 M9(a)); fix in the bib rebuild.
6. **Figure 6 legibility.** At the 385.5pt text width the ra3 figure's fonts are about 5pt and panel (b)'s title nearly touches panel (c)'s. Acceptable, but a regeneration with larger fonts (exhibits.py, `figsize`) would help.
7. **Caveats kept in the text by design:** the data-wall numbers are conditional on Muennighoff et al.'s repetition estimates (≤ 9B); the 7% case is the most favorable one and is always paired with the full-model 43%; the fleet comparison is a loose check; the level of s is technology-dependent (2025: 0.24–0.96), the trend is robust.
8. main.tex already inputs `sections/economics` and `sections/conclusion` in the right order; no edit needed. The stale v1 `tables/table4_technology.tex` is not mine to remove.
