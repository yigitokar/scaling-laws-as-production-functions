# Independent review: module ra4_obsfix (Online Appendix E fixes)

**Scope.** This is a replicator and skeptical-referee review of the revision module that fixes the observational and algorithmic-progress material. The module addresses R4 M4(d)(f), M7, M8(a)(b)(c) and M11, and R1 8(g)(i), 9(e), 10(a)–(e) and minors 42–46. Date: 2026-09-24.
- **Code:** `code/analysis/ra4_obsfix/{run,step_matched,step_alloc,step_progress,step_tfp,tables_figs,infer,ra4common}.py`.
- **Memo:** `output/memos/ra4_obsfix.md`. It was edited in place, and every reviewer change is tagged "[review]".
- **Backups for comparison:** the builder's outputs are in the session scratchpad; they are not deliverables.

## 1. Replication

1. **Run 1: the builder's code, unchanged.**
   - `run.py` exited 0 in 710 s at load average ~40, on one CPU process plus a pool of 4 in step 3.
   - **Every CSV (13), every TeX table (7), `allocative_per_model.csv`, `bench_thetaC_draws.npz` and `besiroglu_boot_theta.npy` were byte-identical to the builder's.**
   - The JSON headlines differ only in runtime and mtime fields, plus one provenance field (issue 12).
   - The builder's pipeline is deterministic.
2. **Run 2: the corrected code.** Exit 0, 667 s. The builder's columns in `matched_long.csv`, `compute_only.csv`, `allocative.csv`, `tfp_units.csv`, `phi_profile.csv` and the reliability files are numerically unchanged, because the new tests use their own seeded streams. The only additions are new columns and rows. After run 2, `tables_figs.py` was re-run alone twice to shorten the Table E1 note (issue 3).
3. **Run 3: the final code, full run.** See §8.
4. **Independent re-implementation** (scratch script, not a deliverable):
   - Rebuilt the legacy (57-model) design and the quadratic surface with statsmodels.
   - Reproduced the point estimates for OLS and family FE × {θ_C, θ_N, θ_D}, for example family-FE θ_D 0.0866 against 0.3485, bias −0.2619.
   - Reproduced the statsmodels CR1 (all-k) s.e. 0.1614.
   - A brute-force restricted wild-cluster bootstrap loop (refitting by least squares in each draw, B = 1,999) gives p = 0.024 for family-FE θ_D. The vectorized `infer.wild_test` gives 0.0245 (builder) and 0.029 (another seed), within Monte Carlo error (s.e. ≈ 0.0035).
   - The vectorization identities (b*_j − θ₀ = h'E*; bootstrap residuals = M E*) are correct.

## 2. Issues found and fixes

Severity levels:
- **major**: changes a number, a claim, or the coherence of the inference the writers would use;
- **minor**: presentation, provenance, or a small numeric effect.

| # | Severity | Issue | Fix |
|---|---|---|---|
| 1 | **Major** | **Table E1 recreated the R1 9(e) inconsistency it was meant to fix.** It paired CV3 s.e. with p-values from a wild bootstrap *studentized by CR1*, and on the 20-model panel the two contradicted each other. OLS θ_C bias −0.024 had s.e. 0.076 (t = 0.32) but p = 0.23. Winogrande (20) bias −0.179 had s.e. 0.263 (t = 0.68) but p = 0.076. Family-FE θ_N (20) went the other way: CV3-t p = 0.010 against WCR p = 0.14. With G* = 1.2–3.8 the procedures disagree by up to 10×. | Added `infer.wild_test_cv3`, a restricted WCR on the bias outcome z = y − y* studentized by CV3 in the sample and in every draw, with independent benchmark draws. Its CV3 comes from vectorized leave-one-developer-out maps and matches `jackknife` to 7×10⁻¹² (`cv3_check`). Table E1 now reports the **conservative p**: the largest of CV3-t(G−1), WCR-CR1 and WCR-CV3. Rejecting only when all three reject is valid if any one is. All three p-values, and their minimum, are in `matched_long.csv`. Table E3 adds the CV3-studentized row: θ_D 0.011, θ_N 0.048. Key 57-model results are unchanged: family-FE θ_D conservative p = 0.027, θ_N 0.048; HellaSwag θ_C all p ≥ 0.36. On the 20-model panel the θ_C conservative p is now ≥ 0.77 (was "≥ 0.16"). |
| 2 | **Major** | **Table E1's "Obs." s.e. was not the s.e. of Obs.** It printed the CV3 of the *bias* (jackknife of y − y*, with the benchmark recomputed on each reduced design). The two differ a lot, e.g. 57-model developer FE: 0.054 against 0.018; 20-model OLS: 0.043 against 0.076. | Added `obs_se_cv3_y` (CV3 of the observational estimate) and display it under Obs. The note says the bias s.e. is the CV3 of Obs. − Exp. combined with the benchmark s.e. |
| 3 | Minor | The longer, corrected Table E1 note made the float 13 pt too tall for the page (tectonic warning). | Note tightened; it now compiles with no oversized float. |
| 4 | **Major** | **Experimental support misstated.** The memo said "OLMo-2 were the only experimental points with D above 634B", that removing them "removes every point above 3.8B × 634B" and that the extrapolated rows use "runs with D ≤ 634B". In fact one Gadre RedPajama run is at 1.44B × 921B (640 tokens per parameter). | Memo corrected in three places: the maximum D is 921B, the maximum at N ≥ 3.8B is 634B, and the 6.9B Gadre runs stop at 138B. |
| 5 | **Major** | **Bjorck exponents (R4 M8(c)) were partly mis-stated.** Checked against arXiv 2409.19913v3, Table 5 and Sec. 4: β ≈ 0.32 is a *joint* fit over 760M, 1.3B and 2.7B, whose per-size fits are 0.3155, 0.3171 and **0.4184**. The memo presented 0.32 as a per-size result for N ≥ 760M. Its range "16–43%" also used the 50M exponent (0.70), which lies below Step Law's 215M–1.07B range. | `step_progress.BJORCK` now lists all six per-size fits plus the joint 0.32, with a `brackets_steplaw_range` flag. The memo (H8, C12, §6) now uses the exponents that bracket Step Law (0.32 to 0.65), giving **17–43%**, and mentions the 2.7B fit. The correction of R4's "0.40–0.70" to 0.38–0.70 for small models is confirmed. |
| 6 | Minor | Wrong interval: the 57-model OLS θ_C bias interval was given as [−0.081, 0.079]. With bias −0.00112 and s.e. 0.04134, the normal interval is [−0.082, 0.080] and the t(18) interval [−0.088, 0.086]. The MDE used normal critical values only. | Memo corrected. Added `mde80_cv3_tG1`: 0.123 (37%) on 57 models and 0.266 (62%) on 20, against 0.116 (35%) and 0.213 (50%) with normal values. |
| 7 | Minor | "Table 8 s.e. × 1.50" is the ratio to *no* correction. Relative to the nested convention the variance is 2.24/1.04 = 2.16 times too large, and the s.e. 1.47 times (0.1614/0.1098). | Memo (H2, C2) corrected. |
| 8 | Minor | Rounding: "EIV raises the within-family elasticity by 0.020 (0.043 at the upper bound)". The exact values are 0.0207 and 0.0425. | Changed to 0.021 (0.042). |
| 9 | Minor | The H6 TFP table did not say that its intervals are **90%** (Table E7 did). It labelled the within-developer row "θ_C = 0.33", but that row converts with its own θ_C = 0.43. Table E7 had the same header ambiguity. | Memo relabelled. Table E7 header footnote ^a says the within-developer rows use column 6's θ. |
| 10 | Minor | **Overclaim on ARC-C.** "The apparent attenuation was a censoring artefact." The Tobit benchmark extrapolates a Gaussian, homoskedastic latent surface below the floor for 60% of runs. It has one level nuisance parameter and a parametric bootstrap that ignores the (design, size) clustering. Its benchmark (0.413) differs from the OLS one (0.547) by more than the observational s.e. | Reworded to "does not survive modelling the censoring (Gaussian Tobit)", with the caveats. Winogrande p-values are now reported across the three methods: 0.068–0.084 on 57 models; conservative 0.53 on 20, where the builder's WCR-CR1 gave 0.076. |
| 11 | Minor | **Allocative intervals and sign.** The memo said "every 95% interval excludes 1" without saying that 13 of 14 intervals hold the technology fixed (only Besiroglu draws parameters) and resample 7 era-1 models. The positive sign is also close to mechanical: every era-1 model has w < 1, and truncation can only raise era 2. Finally, by Prop. A-wedge(v), w < 1 can come from binding data constraints or data-augmenting lab productivity, not only allocative error. | Caveats added to H5, C8 and the Table E6 note, which states that intervals are percentile, that parameters are held fixed except for Besiroglu, and what w < 1 can reflect. The memo recommends leading with the size (1.1–4.9×) and the 80–94% of the gap closed. |
| 12 | Minor | **Provenance and staleness.** `step_alloc.besiroglu_draws` used its own cached copy whenever it existed, so a re-run recorded the copy (not m5's file) as the source and would never pick up regenerated m5 draws. | It now prefers m5's file, refreshes the copy and records m5's hash on every run. The copy is a fallback only. The two files are identical today (sha256 ca3ac3a3…). |
| 13 | Minor | **The φ profile is bimodal, which the memo did not mention.** There is a second local minimum at φ ≈ −0.28 (LR 2.95), and β_year jumps from 0.066 to 0.033 between φ = −0.100 and −0.075, so the optimizer is on different branches. | Disclosed in H7 and open issue 6b. The interval stays connected, and a missed optimum could only widen it. The number [−0.44, 3.37] and T_C 5.3–39.6 are unchanged. |
| 14 | Minor | C7 (renaming OP-style selection) explained why the N/D mix is not a proxy (Prop. proxy(ii)). It did not address R1's own candidate, per-model compute as an LP-type proxy. | Added: under a monotone budget rule the inputs on the path are functions of c alone, so a control function in c absorbs the technology (Prop. proxy(iii), ACF functional dependence). |
| 15 | Minor | C2's "transmission would bias θ_N and θ_D in the same direction" holds under the budget-rule transmission of Prop. 6. It is not a general property of two-regressor omitted-variable bias. | Qualified. |
| 16 | Minor | Table E4 had no header rows, so the writer emitted `\toprule\midrule` (a double rule). | `ra4common.write_aea_table` emits `\midrule` only after header rows. |
| 17 | Minor | Table E1 Panel C rows "ARC-Challenge, 57" and "Winogrande, 57" did not say that they use the benchmark with OLMo-2. | Relabelled "…, 57, with OLMo-2". |
| 18 | Minor | The memo's runtime "4–10 minutes" was exceeded under the current load (11–12 min). | Updated to 4–12 minutes. |

**Checked and found correct (no change):**
- *Reliability (R4 M7).*
  - The diagnosis is right. m4's `lalonde.py` (Table 8 path) computed `cc.var()` (n − 1) on family-demeaned ln C. m4's `observational.eiv_correction` used n − F. Reproduced: λ = 0.514 → 0.766 with σ_u = 0.668; θ_C 0.858 → 0.576.
  - The n − F formula is the correct within-family error variance: E Σũ² = σ²(n − F).
  - The hardware-time reconstruction is independent of reported N and D, and the peaks are right (A100 312 TFLOP/s, H100 SXM 989 TFLOP/s dense BF16).
  - Epoch's MFU field is back-calculated from its own compute estimate. Confirmed in Epoch's `Utilization notes` for Llama 2-70B (0.4192 = 8.1e23/1.932e24), MPT-7B (0.3727), Falcon-40B (0.3864) and Falcon-180B (0.1892).
  - Robustness: keeping one Pythia record per shared chip-hours value (24 records) gives s.d. 0.306 against 0.292.
- *Inference.*
  - CR1 "nested k" follows the fixest/reghdfe convention, and Cameron and Miller (2015) is the right citation.
  - A bootstrap-t is invariant to the constant, which explains the original p = 0.02 with t = 1.6.
  - G* follows Carter, Schnepel and Steigerwald (2017) with ρ = 0, γ_g = Σ_{i∈g} h_i².
  - The leverage shares are right: Cerebras 37% and Microsoft 28% for θ_D; 7 developers at 0 and 2 under 0.1%.
  - The leave-one-developer-out ranges are right.
  - The unrestricted variant is centred correctly, and the benchmark draws are centred at the point estimate.
- *IV and Anderson–Rubin.*
  - The AR sets are right: the grid inverts the CR1-nested t on z in y − θc, with t(G−1) and normal critical values.
  - Own-hardware IV: n = 10, 7 developers, 3 instrument values, F = 14.3, AR t(6) [−1.50, 0.34].
  - Frontier IV: F = 0.90 and 0.008, AR unbounded.
  - The reverse-regression "bracket" 1/b = θ̂/R² is an algebraic identity: 0.328/0.585 = 0.560.
- *Supports and counts (R1 minor 42).*
  - 57 models: 19 developers, 30 families, 15 families with ≥ 2 models (42 models), 15 singletons.
  - 20 models: 6 developers (BigScience, Cerebras, EleutherAI, Meta, Microsoft, TII), 9 families, 5 families with ≥ 2 models (16 models), N ≤ 6.7B, D ≤ 627B.
- *Allocative (R1 10(d)).*
  - CE_tr = CE(min(w, 1)) is implemented correctly.
  - The share of the gap closed, 1 − ln gm_tr2 / ln gm_tr1, is consistent between the point estimate and the bootstrap.
  - Kaplan allocations have w < 1 under every technology (w_kaplan 0.006–0.72), so the CEG is unaffected by truncation.
  - The Ho model uses HO_M7_X, which agrees with m5's fitted x to 4×10⁻⁷.
  - The era samples are 7 and 92 models (GPT-3, Jurassic-1, HyperCLOVA, MT-NLG, Yuan 1.0, Gopher, ERNIE 3.0 Titan).
- *Progress (R4 M4(f); R1 minor 46).*
  - The φ interval [−0.436, 3.367] comes from linear interpolation of LR crossings. T_C at the boundaries uses interpolated g_C, and the minimum of T_C is 5.35 at φ = −0.30.
  - m5's grid-point T_C range was 5.45–38.30, which confirms R4's diagnosis.
  - Table E5 entries match m5's CSVs and JSONs.
- *TFP (R4 M11).*
  - Reproduces m4's ratios: 23.7, 19.9, 7.0, 9.1, 11.1.
  - The unit conversions e^Δ, e^{Δ/θ_C} and e^{γΔ/θ_C} are right.
  - "Log ratio to Syverson 2.3–4.9" holds only in compute units.
- *Citations.*
  - All seven new entries in `lit/bib/extra_ra4_obsfix.bib` were re-verified on Crossref today: titles, authors, journal, volume, issue, pages and year all match.
  - Every other key used resolves in `paper/references.bib` or `lit/references.bib`.
  - The label targets `prop:proxy`, `cor:hall` and `prop:A-wedge` exist in `paper/sections/appendix_proofs.tex`.
  - θ_N − θ_D = 0.21 (0.08) traces to `m4_observational_table6b_global_main_long.csv`: 0.2130 (0.0791), OLS, 128 models. R4 had marked it "UNTRACED".
- *Compile.* All seven tables compile with `paper/AEA.cls` via `tools/tectonic`, with 0 errors, 0 overfull boxes, no oversized floats and no undefined citations or references. The page images were inspected.
- *Figures.* All five PNGs were inspected. They use at most 3 colours per panel and no twin axes, and the labels do not collide.

## 3. Independent verification beyond re-running

- **A brute-force WCR loop.** It re-estimates by least squares and CR1 in every draw, and matches the vectorized p-value within Monte Carlo error (§1.4).
- **The CV3-studentized WCR.**
  - Prototyped on six key rows before integration.
  - Its sample CV3 equals the builder's jackknife CV3 exactly, on all 357 rows (max |diff| 7×10⁻¹²).
- **Bjorck et al.** The per-size exponents were read from arXiv 2409.19913v3 (Table 5; Sec. 4 joint fit over 760M–2.7B; batch 0.5M tokens).
- **Epoch utilization.** The `Utilization notes` text of four records confirms that MFU is back-calculated.
- **The experimental support** was rebuilt from m4's `exp_surface_data`: strict maximum D = 921B (Gadre RPJ 1.44B); maximum N = 6.89B at D = 138B; OLMo-2 at (7.3B, 3.88T) and (13.7B, 5.0T).

## 4. Memo number audit (after fixes)

Every numbered claim in H1–H9, C1–C12 and §5 was traced to the regenerated CSV/JSON files.
- **Pass:** all reliability numbers; every row in H2's table; G*; leverage; leave-one-out; H3 point estimates and counts; H4 biases; every entry in H5's table; the 0.80–0.94 gap-closed range; the 6–73% (24% Besiroglu) Ho shares; Kaplan 11.7×; H6's point ratios and intervals; H7; H9's F, estimates and AR sets; C4; C5; the C9 provenance list.
- **Fixed:** the 95% interval (issue 6); ×1.50 (issue 7); 0.020 and 0.043 (issue 8); D ≤ 634B (issue 4); 16–43% (issue 5); H6 labels (issue 9); strict-panel p-values (issue 1).

## 5. Referee-comment coverage

| Comment | Where | Verdict |
|---|---|---|
| R4 M7 (two reliabilities) | H1, C3; Tables E2 and E4; `fig_reliability` | Addressed. Independent comparisons are identified, EIV rows are recomputed, and the within-family row is explained by two verified errors. |
| R4 M4(d) ("nine technologies") | C9; Table E6 groups | Addressed in the module. The paper still needs a (Mis)Fitting row in Table 2 or Appendix D (writers). |
| R4 M4(f) (φ and T_C) | H7, C11; Table E5 | Addressed. Refined grid; T_C 5.3–39.6; bimodality now disclosed. |
| R4 M8(a) Nerlove | C12 | Addressed (wording). |
| R4 M8(b) Gundlach | C8 | Addressed (wording and contrast with the realized gain). |
| R4 M8(c) Bjorck | H8, C12 | Addressed after review correction (17–43%; joint vs per-size exponents). |
| R4 M11 TFP units | H6, C10; Table E7; `fig_tfp_units` | Addressed. R4 asked for both unit systems "in the main text". The material now sits in Online Appendix E, so the writers must keep one sentence in the main text if any dispersion number stays there. |
| R1 8(g) Nerlove | C12 | Addressed. |
| R1 8(i) LaLonde-style | C1; new bib entries verified | Addressed. |
| R1 9(e) Table 8 p vs s.e. | H2, C2; Table E3 | Addressed. The diagnosis (nested-FE dof factor) is verified. The first version of Table E1 reintroduced the problem; it is fixed by the conservative p (issue 1). |
| R1 10(a) low-power joint test | C1 | Addressed. MDEs are now given with t critical values too. |
| R1 10(b) drop OLMo-2 | H3; Table E1 Panel A; `fig_support` | Addressed. |
| R1 10(c) OP-style misnomer; state/flexible input/timing | C7 | Addressed; strengthened at review (issue 14). |
| R1 10(d) truncated allocative measure | H5, C8; Table E6 | Addressed; caveats added (issue 11). |
| R1 10(e) TFP caveats | C10; Table E7 note | Addressed. |
| R1 minor 42 (counts) | C1 | Addressed. |
| R1 minor 43 (Hall-type bias) | C5 | Addressed; the number is traced. |
| R1 minor 44 / R2 minor 25 (independent comparisons) | H1 | Addressed. |
| R1 minor 45 (AR for every IV) | H9; Table E2 | Addressed for the Table 8 IVs. m4's overlap and trend IVs already carry AR sets in m4's CSVs; the writers must print them wherever those IVs appear. |
| R1 minor 46 (condense Ho) | C11; Table E5 | Addressed. |
| R2 minor 24 (chance floor, harness) | H4; §6 | Partly addressed: floor handling yes, harness mapping no (open issue 2, correctly disclosed). |

## 6. Confidence in each headline claim (after fixes)

- **H1 (reliability ≥ 0.98 pooled, ≥ 0.955 within family; the Table 8 within-family EIV row is wrong): high.** The formula error is verified in code. The bound is conservative.
- **H2 (the Table 8 p/s.e. conflict comes from the dof factor; the family-FE N/D split is significant on 57 models at p ≈ 0.01–0.05): high for the diagnosis, low-to-moderate for the substantive split.** G* ≈ 4; two families (Cerebras-GPT, Phi) carry about 65% of the θ_D leverage.
- **H3 (HellaSwag θ_C null on both designs): moderate.** The claim is only "fails to reject". On 57 models the interval is ±0.087 with t(18) (MDE 37%). On 20 models no inference procedure is reliable (MDE ≥ 50–62%).
- **H4 (ARC-C attenuation not robust; Winogrande borderline): low-to-moderate.** It is model-dependent (Gaussian Tobit).
- **H5 (truncated allocative gain 1.1–4.9×, closing 80–94% of the era-1 gap): moderate for the direction and size under each technology, low for the intervals.** There are 7 era-1 models, parameters are held fixed except for Besiroglu, and the sign is near-mechanical.
- **H6 (dispersion is Syverson-sized in output units and 4–24× in compute units): high as a statement about cardinalization.**
- **H7 (φ ∈ [−0.44, 3.37], T_C 5.3–39.6): moderate.** The profile is bimodal and uses iid LRs.
- **H8 (Bjorck 17–43%): high, after correction.**
- **H9 (instruments uninformative): high.**

## 7. Remaining concerns (not fixed; for the writers and lead author)

1. **The 20-model strict design cannot carry any inferential claim.** Report it only as "the point estimates do not move". The conservative p-values in Panel A are coherent but uninformative.
2. **The Tobit benchmark's s.e. is understated.** The parametric bootstrap is unclustered and conditional on the model (open issue 6a). ARC-C/Winogrande conclusions should stay in the robustness paragraph.
3. **Allocative intervals should carry technology uncertainty for the registry technologies too** (open issue 6c). Until then, quote only the Besiroglu interval.
4. **The φ profile** should be re-optimized with more starts in [−0.5, 0] (open issue 6b). It cannot shrink the interval.
5. **Harness mapping (R2 24)** is untested. It is part of the joint null.
6. **Writers:**
   - Drop the introduction's family-FE "0.20 / 0.26" finding and the "≥ 0.988" sentence.
   - Replace Table 9 panel B with Table E6.
   - Add a (Mis)Fitting row to Table 2 or Appendix D.
   - Use C1–C12 as corrected here, in particular the Bjorck sentence (17–43%) and "D ≤ 921B".

## 8. Final regeneration record

- **Run 3: the final code, full pipeline via `run.py`.** Exit 0, 616 s at load average ~40:
  - step_matched 300 s;
  - step_alloc 13 s;
  - step_progress 17 s (pool of 4);
  - step_tfp 276 s;
  - tables_figs 11 s.
- **Run 3 against the state before it** (run 2 plus the re-generated tables):
  - **all 21 CSV/TeX outputs are byte-identical**;
  - `allocative_per_model.csv`, `bench_thetaC_draws.npz` and `besiroglu_boot_theta.npy` are byte-identical;
  - the JSON headlines are identical apart from runtime and mtime fields;
  - the final code is deterministic and regenerates everything.
- **Every number in the revised memo and in this review comes from these outputs.**
- **The seven tables** compile with AEA.cls (tectonic): 0 errors, 0 overfull boxes, no oversized floats, all citations resolved.
- **Files changed by the reviewer:**
  - `code/analysis/ra4_obsfix/infer.py`: `loo_maps`, `cv3_from_maps`, `wild_test_cv3`.
  - `step_matched.py`: CV3-studentized WCR per row and in the reconciliation, `obs_se_cv3_y`, `mde80_cv3_tG1`, `p_conservative`, `p_min_of_three`.
  - `step_alloc.py`: Besiroglu draw provenance.
  - `step_progress.py`: all Bjorck per-size exponents and the Step Law range flag.
  - `tables_figs.py`: Table E1 Obs. s.e., conservative p, labels and note; Table E3 row; Table E6 note; Table E7 footnote.
  - `ra4common.py`: no double rule.
  - `output/memos/ra4_obsfix.md` (tagged "[review]"); this file.
