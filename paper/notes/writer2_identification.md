# Writer notes (v2): Section II "What Optimizing Labs' Data Identify" (`paper/sections/identification.tex`)

Writer: section "identification" (writer2), 2026-09-24. Replaces the v1 section (kept in `paper/sections/v1/identification.tex`).

**Compile status.** The section compiles cleanly when the missing Figure 2 file is replaced by the v1 figure
(`m6_montecarlo_fig2_designs`) in a scratch build. There are no LaTeX errors, no citation warnings and no overfull boxes.
The only undefined references point to other sections and appendices. Rendered pages were inspected at 90 dpi.

The official `code/paper/test_section.sh identification` stops only at
`File 'm6_montecarlo_fig2_v2' not found`. The mc-fix agent had not produced that file, or
`output/memos/m6_montecarlo_v2.md`, when this section was finished.

**Length.** About 2,480 words of main text by the integrator-style regex count: about 620 of them inside the three
propositions, plus 175 words of figure notes. The plan target is about 2,000. Candidate cuts, if the integrator needs them:
- the Chinchilla on-path profile paragraph (about 55 words) could move to Section III or Online Appendix D;
- the last two sentences of the Monte Carlo inference paragraph (about 60 words) could move to Online Appendix C;
- the closing "In sum" paragraph (about 45 words).

## 1. Structure and labels

| Item | Label | Content | Appendix counterpart |
|---|---|---|---|
| Section II | `sec:ident` | | |
| (no display) | uses framework's `eq:tech` (with ψ_N = ψ_D = 0) and `eq:wM` | v1's `eq:ident-family` display was dropped to avoid duplicating Section I; no v2 file references it | Def. A2 |
| II.A | `sec:ident-onpath` | Optimization removes the variation that identifies curvature | |
| Proposition 3 | `prop:ident` | (i) on-path identification; (ii) global vs local identification, fourth-root rate; (iii) O(v²)/O(v⁴) information | Props. A1 (`prop:fd`), A2 (`prop:A-info`), Lemma A4 |
| eq. (decomp) | `eq:ident-decomp` | y = γ(c − ln 6) − ln K − γΦ − ε | Prop. A2(i) |
| Figure 2 | `fig:designs` | `\includegraphics{m6_montecarlo_fig2_v2}`: two panels, (A) RMSE of σ* and ln M* against v, (B) κ-free profile | Online Appendix C |
| II.B | `sec:ident-modelfree` | Designed variation identifies curvature without a functional form | |
| Proposition 4 | `prop:modelfree` | Model-free w and σ*; test of the family; what κ = 1 imposes; designs | Prop. A10, Remark A-grid |
| eq. (mf) | `eq:ident-mf` | 1/σ* − 1 = L_nn\|_C / (2\|dL*/dc\|) | eq. `eq:A-mf` |
| II.C | `sec:ident-pi` | Beyond the design (label kept from v1) | |
| Proposition 5 | `prop:pi` | Partial identification of M*(C); sign(w − 1) = sign(ln M − ln M*(C)); magnitudes; s and T | Prop. A11 (`prop:A-pi`) |

The subsection letters and proposition numbers follow `theory_main_text_v2.md` §0. Proposition 5 is placed in II.C, as
`paper_plan_v2.md` requires; the theory note had suggested "Section IV (opening)".

**Cross-references assumed to exist elsewhere:**
- `sec:framework`, `eq:tech`, `eq:wM`, `prop:duality` (defines a, γ, G and K), `lemma:sigma`, `prop:wedge` (case (a):
  λ, p, T), `fig:geometry` (Figure 1, panel b, which shows equivalent members with σ* = 0.60, 0.74 and 0.85);
- `sec:tech`, `sec:wedge`;
- `app:proofs`, `app:additional` (the Chinchilla κ-free profile sets from ra1), `app:mc`.

## 2. Number ledger (every number in the section)

| Text | Value | Source (verified) |
|---|---|---|
| Off-path loss of the equivalent members drawn in Figure 1(b), at 5 × M* | 2.0% (σ* = 0.85), 4.1% (0.74), 7.6% (0.60) | Writer computation with the m7 method: pct = exp(γΦ) − 1, Φ from Lemma A5 with S = 2(1/σ* − 1), a = 0.5126, γ = 0.1783, ln w = (S/2) ln 5. It reproduces `output/tables/m7_theory_dmr_family.csv` exactly (1.449, 4.128, 8.441, 18.214 for σ* = 0.889, 0.737, 0.571, 0.333) and gives 2.046 (0.85) and 7.565 (0.60). This addresses R4 minor 6 |
| Rate MC: 40 on-path budgets; q75 exponent | 0.48 (α ≠ β), 0.49 (α = β); 0.99 restricted; 0.5 predicted | `ra5_theory_singular_mc_slopes.csv`: 0.4810, 0.4853, restricted 0.9920 / 0.9947; ra5 memo §1.4 |
| Information slopes, factorial design, E known or estimated | 2.00 (ln M*), 4.00 (σ*) | `ra5_theory_info_Eunknown.csv`: I_S ×9,986 and I_lnM* ×99.9 over a 10× range of v; appendix Prop. A2(iv) |
| Harberger approximation | Φ ≈ σ*(ln w)²/[4(1 − σ*)] | Lemma A5 (`lem:ce`) |
| DataDecide design | about 100 tokens per parameter | v1 `data.tex` l. 38 (D ≈ 100N); m2 memo l. 360 |
| Chinchilla on-path band | 41 runs within 0.15 log points of the fitted N*(C); κ-free 95% set [0.20, 0.99], hitting the grid edge (grid [0.05, 0.99]) | `ra1_modelfree_chinchilla_profile_sets.csv` (on-path band, Laplace QLR: set_lo 0.20, set_hi 0.99, set_hits_hi True); band definition in `code/analysis/ra1_modelfree/chin_inf.py` l. 312 |
| Chinchilla full design (240 runs) | calibrated set [0.68, 0.72] | ra1 memo H5 ("with the largest endpoint critical value the calibrated set is [0.68, 0.72]"); `ra1_modelfree_chinchilla_profile_set_calibrated.csv` |
| MC truth | σ* = 0.737; M*(10²⁴) = 18.1 | m6 memo header; `ra5_theory_modelfree.csv` (M* = 18.137 at 10²⁴) |
| MC noise | s.d. 0.0075 in ln L | m6 memo §2.1 |
| MC design | 90 runs; nine budgets 6×10¹⁸–3×10²¹; total 5.1×10²² FLOP; IsoFLOP ±16× with ten sizes; factorial 9 N × 10 D; 500 / 150 replications | `m6_montecarlo_designA_diagnostics.csv` (total_compute 5.106e22); m6 memo §2.2 |
| **m6v2** corner share at v = 0 | 51% | `m6_montecarlo_designA_summary.csv`, opt_s0 `corner_share` 0.510 |
| **m6v2** RMSE σ̂* at v = 0.05 and 0.2 | 0.067, 0.015 | same CSV: 0.06695, 0.01469 |
| **m6v2** RMSE ln M̂* at v = 0.1 and 1 | 4.5, 0.59 | same CSV: 4.491, 0.590 |
| **m6v2** IsoFLOP ±16×: RMSE σ̂*, ln M̂* | 0.0036, 0.16 | same CSV, iso16 primal: 0.00357, 0.1619 |
| **m6v2** κ-free profile: flat on path; flat at v = 0.3 | 88%, 65% | `m6_montecarlo_designA_profile.csv`, `share_flat_everywhere`: 0.880, 0.653 |
| **m6v2** κ-free set widths: IsoFLOP, factorial | 0.016, 0.019 | same CSV, `median_ci_width`: 0.0161, 0.0186 |
| **m6v2** flat share at v = 0.3, noise s.d. 0.005 | 38% | same CSV, opt_s0.3_sd0.005: 0.380 |
| **m6v2** Wald singular or corner, on path | 35% | summary CSV, onpath `wald_computable` 0.652 |
| **m6v2** bootstrap coverage of σ*, warm vs multi-start (60 reps × 49 draws) | 57% vs 92%; width 0.43 | `m6_montecarlo_designA_bootcheck.csv`, onpath sigma_star: 0.567, 0.917, 0.4285 |
| Finite grid (Chinchilla at 10²¹, nine points) | span 9×: 0.734; span 100×: 0.726; truth 0.737; quartic 0.737 | `ra5_theory_finite_grid.csv`: 0.73447, 0.72574, 0.73703, quartic 0.73704 / 0.73716 |
| Largest public IsoFLOP budget | 10²² FLOP (Llama 3) | ra2 memo H6; ra1 memo §2.2 (Llama 3 budgets 6×10¹⁸–10²²) |
| Clean sample beyond 10²² FLOP | 72 of 77 | `ra2_wedge_models.csv`, clean == True, `Cmp` > 1e22: 72 |
| Clean sample inside Chinchilla design (M ≤ 341, C ≤ 1.3×10²²) | 2 of 77 | `ra2_wedge_insupport.csv` (inside = True, n = 2) |
| Llama 3 8B | M = 1,868; C = 7×10²³; w > 1 identified under PI-1 to PI-3 (IsoFLOP anchors) | `ra2_wedge_models.csv` (M 1867.9, Cmp 7.23e23); `data/processed/ra2_wedge/pi_bounds_models.csv` (sign w>1 under PI-1, PI-2, PI-3; ambiguous under PI-4, hence the qualifier "with the path anchored at the largest IsoFLOP budgets") |
| Llama 3.1 405B | M = 38; C = 4×10²⁵; sign not identified under every assumption | same files (M 38.4, Cmp 3.80e25; ambiguous under PI-1 to PI-4). ra5's illustration agrees (Remark A-pi-illus). |

Not quoted in Section II: ra2's PI shares (86% / 77% / 23%) and the M*(10²⁴) bounds. They
belong to Section IV.D (see open issue 5).

## 3. Referee comments addressed

| Comment | Where / how |
|---|---|
| R1 c7 (model-free σ* and w) | Prop. 4 with intuition; the finite-grid bias and the quartic fix; Section III applies it |
| R1 c8a (Marschak–Andrews, Bond–Söderbom, GNR; ACF is a different point) | Positioning paragraph after Prop. 3 |
| R1 c8b (optimal-design literature; where the result binds) | Box–Lucas and Kiefer–Wolfowitz cited; scope paragraph (ladders, DataDecide, compute-optimal era vs released models) |
| R1 c8c (global vs local; the "iff" in (ii) too strong) | Prop. 3(ii) separates global identification (functional form) from local (fails; fourth-root rate; Wald invalid); (i) says "within the model" |
| R1 c8d (IsoFLOP sweeps are not price shocks) | "IsoFLOP sweeps are different: … neither a price nor a behavioral response" |
| R1 c8e (GNR analogy for A/B on the path) | Removed; (ii) says A/B is identified "only by imposing that the observed path is optimal" |
| R1 c4e (partial identification of M*(C) at frontier scale) | Prop. 5, with the Llama 3 8B / 405B illustration; the shares are left to Section IV.D |
| R1 c5 (cost-shifter heterogeneity = relative-price variation) | Positioning: "size-dependent prices of a FLOP" among the escape routes |
| R1 c9c; R4 M5(b) (profile sets truncated at the grid) | Chinchilla band: "runs from 0.20 to 0.99, the edge of the grid searched"; MC: "covers the entire grid searched" |
| R1 c9d (SEs of A and B are parameterization artifacts) | v1's 125 / 1,293 claim is dropped from the main text |
| R1 minor 2 (RMSE at s = 0 depends on the box and the starts) | Only the corner share is reported at v = 0; RMSEs are for v > 0 |
| R1 minor 14 (separate outcomes from choices) | Prop. 3(i): choices reveal the path; outcomes reveal γ; (ii) "from outcomes only by imposing optimality" |
| R1 minor 15 (E estimated) | "with E known or estimated" after Prop. 3(iii) |
| R1 minor 16 (noise calibration vs seed noise) | Noise sensitivity of the flat share (38% at 0.005) plus a TBD-m9 placeholder for the experiment's seed s.d. |
| R1 minor 17, 18 (DMR, transmission) | These results are now in Online Appendix E; not in Section II |
| R1 minor 19 (projection variance to a remark) | Not in Prop. 5 (Remark A-pi-param) |
| R2 Major 3(ii) ("extrapolated T̂ is conservative" conflates level and slope, and true and perceived technology) | Withdrawn. Closing sentences of II.C: the direction of extrapolation errors is empirical (Section IV), and revealed preference recovers the believed technology |
| R2 Major 9d (IsoFLOP argmins are outcomes of assigned inputs) | "their minima estimate the path from outcomes, not from choices" |
| R2 minor 7 (random starts only) | Figure 2 notes: "started from dispersed values that exclude the truth" (**to verify against the m6 re-run**) |
| R2 minor 9 (decomposition conditional on κ = 1) | Not quoted; II.C says a single-technology interval reflects "one design and one functional form" |
| R3 M8 (Kricheli reconciliation; Bond–Söderbom; Sargan; Rotnitzky; optimal design) | Part (ii) paragraph (Kricheli: equal decay rates along the design's ray; α = β on fixed-ratio rays, always on the path); citations |
| R3 M11 (scope of the theory; E vs 𝔼; fewer results) | Scope paragraph; `\E` = 𝔼; a₁, b₁ absent from the main text; v1 Props. 5–6 moved out (Appendix E) |
| R3 M12 (Figure 2: two panels, truth removed) | Figure call and notes written for the two-panel v2 figure |
| R3 minor 3 (functional-form caveat beside the MC precision) | "For 0.1 ≤ v ≤ 0.3 this precision is functional-form information, as part (ii) predicts" |
| R3 minor 4 ("entire grid searched") | Done |
| R3 minor 13 (MC s.e. of coverage rates) | "in a 60-replication check" (MC s.e. of a coverage rate there is about 3–6 pp) |
| R3 §2 item 6 (under optimality, choices *do* reveal the optimal mix) | Prop. 3(i): "choices reveal the path, that is, a and M*(C)" |
| R4 M4(e) (warm vs multi-start like for like) | 57% vs 92%, both from the 60-replication check |
| R4 M1 / minor 11 (Chinchilla budgets in the Figure 2 notes) | "nine budgets matching Chinchilla's nominal IsoFLOP budgets" |
| R4 minor 14 (s has three meanings) | The MC allocation-error s.d. is now v, the transverse spread of Prop. 3(iii); s is reserved for the expenditure share |
| R4 minor 6 (Figure 1 members vs quoted numbers) | The off-path losses are now quoted for the members plotted in Figure 1(b): 0.85, 0.74 and 0.60 |

## 4. Open issues for the integrator

1. **Monte Carlo v2 (blocking).** `output/memos/m6_montecarlo_v2.md` and `output/figures/m6_montecarlo_fig2_v2.pdf` did not
   exist when this section was finished.
   - Every MC number in the text is from the first m6 run (truth among the starting values). Each such line ends with
     the comment `% m6v2`: the corner share, the RMSEs, the κ-free flat shares and widths, the noise sensitivity, Wald
     computability, and bootstrap coverage.
   - Replace them with the re-run's numbers, and check the Figure 2 notes against the v2 design: that the starting values
     exclude the truth, the replication counts, the σ* grid and the panel contents.
   - If the re-run widens the σ* grid beyond [0.50, 0.95], update "the entire grid searched, [0.50, 0.95]".
   - Near-path primal numbers (51%, 0.067, 4.5, 35%) are the most likely to change. The dual, IsoFLOP and factorial
     numbers should not change.
2. **Figure 2 labels.** The text calls the allocation-error s.d. **v** (because v2 uses s = (w − 1)/w) and refers to
   "panel A" (RMSE of σ̂* and ln M̂* against v, with IsoFLOP reference lines) and "panel B" (κ-free profile LR). Ask the
   mc-fix agent, or edit the plotting script, so that the axis reads v and the panels are titled A and B. The v1 figure
   says "s" and has four panels.
3. **Figure 1 consistency (R4 minor 6): resolved.** The text quotes 2.0 / 4.1 / 7.6 percent for the plotted members
   σ* = 0.85 / 0.74 / 0.60. If the Figure 1 owner changes the plotted members, recompute them with the formula in the
   ledger.
4. **Stale labels in other files.** `prop:fdep` and `prop:info` (v1 Props. 3–4) no longer exist; use `prop:ident`.
   - `paper/sections/appendix_mc.tex` (v1 text, ll. 40 and 54) still references both.
   - v1 `data.tex`, `technology.tex` and `introduction.tex` did as well; v2 writers should use `prop:ident`.
   - Kept from v1: `fig:designs`, `sec:ident-pi`. Dropped: `eq:ident-family` (use `eq:tech`).
   - New: `sec:ident-onpath`, `sec:ident-modelfree`, `eq:ident-decomp`, `eq:ident-mf`.
5. **Partial-identification numbers (Section IV.D).** The ra5 reviewer (`ra5_theory_review.md` §7.7) flags ra2's anchor
   intervals as too narrow (Llama 3: ra2 [20.8, 23.7] vs ra5 [12.2, 40.2]).
   - ra2's review did not change them.
   - Section II quotes no PI share or bound, and its two Llama examples hold under either set of anchors.
   - The Section IV writer or integrator should settle the anchor inference before quoting 86% / 77% / 23%.
6. **Duplicate family definition: resolved.** The section now refers to framework's `eq:tech` (with ψ = 0) and
   `eq:wM`. A combined build of framework + identification + appendix_proofs resolves every label among the three files,
   with no multiply defined labels.
7. **Chinchilla on-path band.** The two sets come from ra1: the Laplace-QLR χ² set on the 41-run band, and the calibrated
   [0.68, 0.72] on the full design. They must also appear in Online Appendix D (`ra1_modelfree_chinchilla_inference.tex`),
   which the text cites through `app:additional`.
8. **Result count.** There are six main-text formal results (Prop. 1, Lemma 1, Props. 2–5), against revision plan A's
   "≤ 5" (ra5 open issue 7). This is the lead author's call.
9. **Not addressed here.** R2 minor 8 (a heteroskedastic, clustered-noise MC variant) is a Monte Carlo task.
10. **Placeholder.** `\textcolor{red}{[TBD-m9: seed standard deviation of $\ln L$]}` in II.A: fill it from the m9 seed
    replicates (pre-analysis plan Q4).
11. **No new bibliography keys.** Every key used is in `paper/references.bib`: marschak1944random, bond2005adjustment,
    gandhi2020identification, box1959design, kiefer1959optimum, ackerberg2015identification, sargan1983identification,
    rotnitzky2000likelihood, chen1995optimal, kricheli2026tokens, kaplan2020scaling, magnusson2025datadecide,
    besiroglu2024chinchilla, hoffmann2022training, czech2026problems and grattafiori2024llama.
