# Independent review: module m7_theory (theory verification and formal appendix)

Reviewer: independent replicator and referee. Date: 2026-09-23.

Scope:
- the code in `code/analysis/m7_theory/` (`run.py`, `symbolic.py`, `numeric.py`, `figures.py`, `common.py`);
- all tables, figures and processed files;
- the memo `output/memos/m7_theory.md`;
- the appendix `paper/sections/appendix_proofs.tex`;
- the main-text proposal `paper/notes/theory_main_text.md`;
- the new bib file.

**Bottom line.** The module is careful and mostly right. The mathematics holds up:
- every proof in the appendix was re-derived by hand;
- every headline number was reproduced, and the key ones were recomputed with independent code.

I found **no critical error**. I found **two major problems** and **about 18 minor ones**, all fixed:
- a false "if and only if" in an appendix corollary;
- a mislabeled standard-error curve in a paper figure.

After the fixes, the suite has 61 checks (57 original plus 4 reviewer additions). All pass. The module regenerates deterministically from scratch in about 3.5 minutes on one CPU.

---

## 1. Replication

- I backed up and deleted every module output (13 tables, 8 figure files, 2 processed files) and re-ran `run.py` from scratch.
  - The run completes in 229 s: 57/57 PASS, exit 0.
  - All CSV/TeX outputs are byte-identical to the builder's, except the "max|err| at random points" residuals (~1e-165) in three sympy rows.
  - Cause: `symbolic._zero` assigned random values by iterating a Python *set* of symbols, which depends on the hash seed. Verdicts are unaffected. **Fixed** by sorting symbols by name. Two consecutive post-fix runs now produce identical check lines.
- After all fixes, a final clean run gives 61/61 PASS in 208 s (`VERIFIED` 31, `NEW` 21, `IMPRECISE` 6, `FALSE` 3).
- Independent recomputation, with my own code rather than the module helpers:
  - Llama-3-8B under Besiroglu: w = 5.2175, T/D = 12.653. C/C_min = 5.5127 by direct constrained minimization.
  - Gopher: w = 0.3622, C/C_min = 2.0121.
  - Transmission limits: 0.0787, 0.1780, 0.2470. r* = 0.4991; reverse slope under funding 0.402.
  - Harberger overstatement at w = 5.22: 1.228.
  - Curvatures along the isocost: 0.1272 (Besiroglu; analytic ab(α+β)²) and −0.090 (CES; analytic −r²).
  - Refit objective 1.01827e−3 vs published parameters 1.02284e−3.
  - w_b = 0.763.

  All match the memo.
- **Bootstrap health** (not checked by the builder). Replicates are warm-started from the full-sample optimum.
  - I re-fitted 30 of the 300 draws (every 10th, seed 7) from the 432-start `FAST_GRID`: objective differences ≤ 1.4e−10 and |ΔS| ≤ 4.7e−5.
  - Across the 300 draws there are 0 outliers beyond 6 MADs in S or ln M*; S ranges from 0.654 to 0.783.
  - The resampling unit (pairs, n = 240) matches Besiroglu et al.
  - The full-sample fit from `FAST_GRID` equals the fit from the published start.
- **Compile test.** With the builder's proposed preamble fix (`\let\proof\relax\let\endproof\relax` before `amsthm`), the edited appendix and all three `.tex` tables compile under `AEA.cls` with `tools/tectonic`. There are 3 overfull boxes of at most 4.4pt. Rendered pages were inspected.

## 2. Issues found and fixes

Severity key: **major** = a false or misleading statement that would enter the paper; **minor** = imprecision, coverage gap, reproducibility or presentation.

| # | Severity | Issue | Fix |
|---|---|---|---|
| 1 | major | Appendix Corollary A2 (CEG) said that at a common allocation rule D = mN, f is constant "if and only if E and the exponent sets coincide". Only "only if" is true. Counterexamples: factor augmentation with ψ_N ≠ ψ_D, and Hicks-neutral change with α ≠ β, both give a non-constant f on a common ray. | Statement and proof corrected: constant iff the new law is the old one with N rescaled, f = λ². New check `P4S.CEG.rule`: sd(ln f) = 0.028 (ψ_N ≠ ψ_D) and 0.025 (Hicks-neutral, α ≠ β), against 5.5e−15 when ψ_N = ψ_D. Main-text note and memo claim 7 updated. |
| 2 | major | The "se(ln M̂*)" curve in `m7_theory_information` (column `se_lnMstar`) was 2/√I(ln G), the s.e. of ln M* at C = 6 FLOP, about 44 log-units outside the design. At the design's central compute the s.e. is about 15× smaller (3.45 vs 51.1 at sd(ln w) = 0.50, unit noise). The slope claim (−1) was unaffected, but the level comparison with se(σ̂*) in the figure was misleading. | Added the efficient information and s.e. of ln M* at 10²⁰ FLOP via the full inverse information matrix (`se_lnMstar_center`, `I_lnMstar_center`, with a leading-order check). Old column renamed `se_lnMstar_at_C6`. Figure replotted. INFO.num now also tests the central-compute slope (2.00) and ratio (1.000). |
| 3 | minor | Memo: "Information about σ* is bounded by (γ/(Ss))²ΣΦ²". The bound is for S = α+β; for σ* multiply by (dS/dσ*)² = 4/σ*⁴. The appendix statement was correct (for S). | Memo, claims table (`INFO.dec` row), main-text note and appendix remark reworded. |
| 4 | minor | SOC stated as "cost minimum iff σ < 1" in the memo, claims table and main-text note. The correct statement is 0 < σ < 1: σ < 0 also has Q < 0. The appendix Lemma A2 was correct. | Fixed in all three places. |
| 5 | minor | The Chinchilla transverse spread used all 245 raw rows (sd(ln w) = 0.534), while the docstring and the estimation/bootstrap sample are the 240-run sample (0.480). The processed text file said 245; the memo said 0.53. | Uses `sl.chinchilla_extraction` (240 runs): 0.480. Memo and main-text note updated. |
| 6 | minor | Partial-identification illustration: "design support ends at M = 341" and the shaded region in `m7_theory_wedge`(c) implied that M ≤ 341 is supported at C = 7.2e23. The design's maximum compute is 1.3e22, so every M there is a 55× compute extrapolation; sd(ln M̂*) = 0.40 at that C reflects this. | Figure label, table note, appendix remark and memo now say so. `C_design_max` added to `m7_theory_pi_summary.csv`. |
| 7 | minor | The memo and appendix number "w_b = 0.76" (nonparametric bound uninformative) was not produced by any code. | Now computed in `partial_id_illustration` (`wb_max_besiroglu` = 0.763, `wb_max_refit` = 0.766, 10 dominating support points). |
| 8 | minor | In `check_ceg`, the "E differs" branch never ran: E_new = 1.75 < E_old made L_new fall below E_old, so f was undefined. The necessity of equal E was never checked numerically. | E_new = 1.85; the branch now runs (sd(ln f) = 1.15), and the check requires it. |
| 9 | minor | Lemma A3(vi) (with inference demand, ∂ln M/∂χ = −2/(α+β+θ_T) at given compute) was not verified. `L1.T` checks a different comparative static, at a fixed loss target. | Added a sympy implicit-differentiation check (`L1.T.sym`) and a brute-force check at fixed training compute (`L1.T.num`: −1.45269 vs −1.45269). |
| 10 | minor | The headline correction (sign of the Hicks bias identified) rests on the *nonparametric* identity ∂ln M*/∂t = −[σ*/(1−σ*)]B_t. The suite verified it only inside the κ-family. | New check `P5.drift.gen`: a non-separable three-term technology outside the family (σ* ≈ 0.79 vs κ-family 0.738), 3 progress patterns × 3 budgets. Maximum relative error 1.3e−6. |
| 11 | minor | Dynamic-panel remedy (Prop. A5(v)): c_{ft} is a valid instrument only if compute does not respond to the previous run's noise ε_{f,t−1}. The proof asserted validity from "information set at t−1", which contains y_{f,t−1}. | Condition stated; fallback instrument set (1, c_{t−1}, c_{t−2}, y_{t−2}) given. Proof corrected. |
| 12 | minor | Some numeric checks hold by construction: PROXY.num and L1.num (ω never entered the optimizer), PI.band (fitting an exact quadratic), P2.a (n exact in c). | ω now enters the brute-force objective. The by-construction nature is documented in the memo §5; the symbolic proofs carry the weight. |
| 13 | minor | DMR table: the last column was "relative to the lowest member" (hard to interpret). The note said paths and frontiers coincide to 1e−14, but that is true only for ln R*; the argmin n* agrees to 4e−7, the optimizer tolerance. | Column now reports % above the common frontier at M = 5M*: 1.4 / 4.1 / 8.4 / 18.2. Note corrected. Main-text "16%" updated to the range. |
| 14 | minor | Prop. A1(iii): K is identified only under the normalization E[e^{−ω+ε}] = 1 (the proof already used K̃). | Stated. |
| 15 | minor | Hall corollary proof: "Törnqvist weights remove it". They reduce the error to third order, not to zero. | Reworded. |
| 16 | minor | `m7_theory_bias`(b) legend said the reverse regression is unchanged by "any selection on y". That holds only under joint normality. | Label now "selection-proof if Gaussian". The layout collision this created was fixed and checked visually. |
| 17 | minor | `m7_theory_wedge.tex` typeset negative numbers with a hyphen. | Math minus. |
| 18 | minor | DMR remark: the sign-identification argument presumes fixed relative cost weights over time. If the effective price of tokens moves (data acquisition, tightening data constraints, inference demand), the DMR problem returns. | Caveat added to the appendix remark and the main-text note. |
| 19 | minor | Prop. A3(ii): a reader may object that a Hicks-neutral trend undoes the separate identification of (g_N, g_D). It does not: in every member, ω_t = g_ω t equals augmentation at (a, b)g_ω/γ, which is S-invariant. | One sentence added to the proof (verified by hand: κ_S a₁ = γ/a, κ_S b₁ = γ/b). |
| 20 | note | Register verdict for `P5.ident` is `IMPRECISE`, while SYNTHESIS P3's "hence not the sign of the bias" is plainly false. model_spec Prop. 5's "the Hicks bias is not identified" is true for the magnitude. | Left as is; the memo explains it. Writers should treat SYNTHESIS P3's sign statement as false. |

Other items checked and found correct:
- all sympy identities, including the general σ = P/(P+Q), the Hicks formula and its ordinality, W1/W2/W3, the KKT wedge formulas, the information decomposition and the proxy;
- the proofs of Lemmas A1–A5, Props. A1–A9 and Corollaries A1–A5 (by hand);
- the transmission and selection algebra, including r* and the claim that b_sel increases in r when δ > 0;
- the Efron / Bagnoli–Bergstrom conditions;
- the κ-family construction (A_S, B_S);
- the rank-3 Jacobian and the α = β row-space condition;
- all 53 ledger numbers;
- the MC tolerances;
- units: N and D taken as in the Epoch extraction (D = C/6N); losses in nats on MassiveText; no unit conversions are needed in this module.

All citation keys exist (`lit/references.bib` or `lit/bib/extra_m6_montecarlo.bib` / `extra_m7_theory.bib`). The three new entries (Bagnoli–Bergstrom 2005, Efron 1965, Klepper–Leamer 1984) have correct metadata.

`sl.py` was not edited. No bug was found in it: the small gap between the published and refit Besiroglu parameters is a lower objective at the refit, not a bug.

## 3. Files changed by the review

- **Code:**
  - `code/analysis/m7_theory/symbolic.py`: deterministic `_zero`; new check `L1.T.sym`.
  - `code/analysis/m7_theory/numeric.py`: ω in the objective; `L1.T.num`; CEG fixes and `P4S.CEG.rule`; `P5.drift.gen`; Fisher information at the central compute; bootstrap diagnostics; w_b and `C_design_max`; the DMR %-above-frontier column.
  - `code/analysis/m7_theory/figures.py`: 240-run sample; central-compute s.e.; labels.
  - `code/analysis/m7_theory/run.py`: claims-table rows; DMR and wedge table formatting and notes; processed-file text.
- **Paper:** `paper/sections/appendix_proofs.tex` (issues 1, 3, 6, 11, 14, 15, 18, 19); `paper/notes/theory_main_text.md` (issues 3, 4, 5, 6, 13, 18, and the check count).
- **Memo:** `output/memos/m7_theory.md`. Corrections are marked *[review]*, with a reviewer note at the top.
- **Regenerated outputs:**
  - `output/tables/m7_theory_*` (13 files);
  - `output/figures/m7_theory_*` (8 files);
  - `data/processed/m7_theory/*` (2 files).

## 4. Remaining concerns (not fixable within this module)

1. **Everything is conditional on the (generalized) Chinchilla family and on cost minimization.** Nothing here tests the functional form. The rank-one-Hessian restriction holds for the whole κ-family, so a translog rank-one test checks additive power separability, not κ = 1. The builder notes this correctly.
2. **The Fisher-information results are local** (small transverse deviations), with Gaussian noise on y and E known.
   - Leading-order ratios drift to 0.91 (σ*) and 0.90 (ln M*) at sd(ln w) ≈ 0.5.
   - The case with E unknown and the non-standard convergence rates under singular information are not done.
   - The figure's s.e. levels assume unit noise; only the slopes should be quoted.
3. **The selection closed forms (b_sel, r*) require joint normality.** The distribution-free results are inequalities only.
4. **The partial-identification illustration is parametric and illustrative.** It uses MassiveText units and extrapolates 55× in compute and 5.5× in M beyond the Chinchilla design. The nonparametric lower bound is uninformative on this support (w_b = 0.76 < 1).
5. **Empirical use of the sign result (Prop. A3(i)) needs** training-optimal allocations (IsoFLOP minima) across dates, at comparable compute, under FLOP-only costs. Deployed models' D/N moves with inference demand and data constraints.
6. **For the lead author:**
   - `paper/main.tex` still does not compile as is: the `amsthm`/AEA `proof` clash; the tested fix is above.
   - `lit/references.bib` note fields still break `aea.bst` (raw underscores).
   - Neither was changed here.
7. **Possible sharpening (suggested, not implemented).** Under κ = 1, on-path outcome data alone imply σ* ≤ 1/(1+2γ), because S ≥ 4γ at fixed γ. That is 0.737 for Besiroglu. This explains the α = β row-space result and could be a one-line remark in Prop. A1.

## 5. Confidence in each headline claim (after fixes)

| Headline claim (memo §1) | Confidence | Comment |
|---|---|---|
| 1. All formal claims checked twice; tally of verified, corrected and false | High | 61/61 re-run; the proofs re-derived by hand. Tally updated for reviewer additions. |
| 2. Interior compute optima reveal 0 < σ < 1 (SOC lemma) | High | General P/(P+Q) identity proven; curvature numbers replicate analytically. |
| 3. Sign of Hicks bias identified; g_N, g_D identified; σ* and \|B\| not | High (math); medium (empirical use) | Identity now verified outside the κ-family. Needs fixed cost weights and IsoFLOP minima over time. |
| 4. Wedge sufficient statistic; C/C_min(w); Harberger form | High | Independent brute-force replication (Llama-3-8B 5.513, Gopher 2.012). |
| 5. Information: ln M* ∝ log-wedge dispersion (v²); curvature ≤ ΣΦ² (v⁴) | High (orders); medium (levels) | Object corrected: the bound is for S. The ln M* s.e. is now at a relevant compute. Local, Gaussian, E known. |
| 6. Transmission sign = sign(π₁); bracket condition; selection r* | High | Closed forms replicate; MC within tolerance. Gaussian for r*. |
| 7. Data-constraint and memory wedges; contamination e^{−χ} | High | Brute-force KKT multipliers match to 4e−8. |
| 8. Partial identification: quadratic Var(ln ŵ); T/D = 12.83 [9.11, 18.88]; M* dominates | Medium | Bootstrap sound (warm starts verified); the result is an illustration with heavy compute extrapolation. |
| 9. 53 ledger numbers reproduce | High | Recomputed. |
| CEG corrections (P4) | High | The frontier "iff E and γ" holds. The common-rule statement is now correctly "necessary, not sufficient". |
