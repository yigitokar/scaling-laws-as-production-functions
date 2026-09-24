# Independent review: module ra5_theory (Theory v2)

Reviewer: independent replicator and skeptical referee. Date: 2026-09-24.
Scope: `code/analysis/ra5_theory/*`, `paper/sections/appendix_proofs.tex`, `paper/notes/theory_main_text_v2.md`,
`output/memos/ra5_theory.md`, the module's tables and figures, and `lit/bib/extra_ra5_theory.bib`.

## 0. Verdict

The theory is sound. I re-derived every new formula by hand, and the brute-force checks are real checks: they solve the
developer problem or trace isoquants, and do not re-evaluate the closed form under test. The module reproduces exactly.

One inference problem needed a real fix. The partial-identification anchors used a raw-residual percentile wild
bootstrap that covers only about 80 percent at these designs. After the fix, every partial-identification number
changed; for example, the share of models with w > 1 identified fell from 85 percent to 77–81 percent.

Four statements were overstated or imprecise and are now qualified:
- v1's "iff α = β";
- the n^{-1/4} "no estimator" claim;
- the family-FOC wording;
- one wedge-table column.

After the fixes: **38/38 PASS** (one check added, one extended). The run takes about 2 minutes on 4 processes and is
deterministic.

## 1. Replication

- I deleted every module output and re-ran `run.py` from scratch before changing anything. The result was 37/37 PASS in
  111 s. All 19 tables, both figures and both processed files were **byte-identical** to the builder's.
- After the fixes I deleted the outputs and re-ran twice: **38/38 PASS** (32 NEW, 5 CORRECTED, 1 VERIFIED) in 47–60 s.
  All CSV, TeX and PNG outputs are byte-identical across the two runs. The two figure PDFs differ only in matplotlib's
  embedded `CreationDate`, which comes from the shared `aer_style.savefig`, not from this module.
- `appendix_proofs.tex` compiles with `code/paper/test_section.sh appendix_proofs` with no errors and no undefined
  citations.
- No label was removed relative to v1. The 12 new labels follow the `prop:A-`/`rem:A-`/`eq:A-` convention.

## 2. Code audit (bugs, conventions, estimators, inference)

| # | Severity | Issue | Fixed? |
|---|---|---|---|
| 1 | **Major** | **Anchor intervals for Prop. A11's illustration under-cover.** `partial_id.fit_path` fitted ln M\* on ln C through the 9–10 IsoFLOP minima and bootstrapped **raw** OLS residuals (Webb weights), then took **percentile** intervals for the intercept at the top budget. The top budget has leverage h ≈ 0.40, so raw residuals understate its error variance by (1 − h), and percentile intervals ignore small-sample thickening. The builder's interval [14.8, 28.5] was narrower even than the exact homoskedastic t₇ interval [13.0, 32.3]. **Simulated coverage at these exact designs** (normal errors, 1,500–2,000 replications, B = 999) for a nominal 95% interval: raw percentile 76–83%; HC2-rescaled percentile 81–87%; HC3-rescaled percentile 87–91%; **wild bootstrap-t with HC2 residuals and SEs 90–91%**, both homoskedastic and with noise rising toward the top budget; classical t 95% (homoskedastic) but 87–88% (heteroskedastic). | **Yes.** Switched to the equal-tailed wild bootstrap-t (HC2-rescaled residuals, HC2 SEs recomputed in each draw, Webb weights, B = 9,999). The residual ~5-point under-coverage is disclosed in the appendix remark, the notes and the memo. New anchors: Chinchilla M\* = 20.5 **[11.4, 36.8]** (was [14.8, 28.5]); Llama 3 22.3 **[12.2, 40.2]** (was [16.4, 30.5]). All downstream numbers were updated (§3). |
| 2 | Minor | The **wedge-case table's last column** ("T̂ vs λpT") mixed three comparisons. Row (a) gave ×λφp, which is relative to T, not λpT. The factor-bias row gave ×e^{−χ}, which applies to ŵ, not T̂. Row (f) gave "sign of δ", which is wrong when δ and δ_S have opposite signs. | **Yes.** The column now compares T̂ = 3D(ŵ − 1) with λpT throughout: (a) ×φ; (f) "+3Dδ + λδ_S pT" (the exact difference); factor bias "Over iff w_L > w_L′" (χ < 0 under factor bias). The same fix is in `theory_main_text_v2.md`. |
| 3 | Minor | **PI.sharp.num tested only an increasing continuation** (parameter augmentation ψ_N). The claim, and the proof of A11(ii), cover decreasing continuations through ψ_D. | **Yes.** A decreasing continuation (data augmentation) was added; it passes (argmin shift error 4.1e-7, positivity and single-peakedness hold, in-support argmins unchanged). |
| 4 | Minor | The FG.num register metric said "10x N-span" but reported the 9× row. | **Yes.** |
| 5 | Minor | **Latent crash in the Monte Carlo objective.** A bound branch of `_si_profile` becomes singular if a rate is pushed to ≈ 0. My extra-polish script hit it; the module's deterministic run does not. | **Yes.** It is guarded (returns ∞); outputs are unchanged. |
| 6 | Minor | **Paper tables.** `ra5_theory_claims.tex` was one 37-row float, which overflowed the page ("Float too large" by 422 pt). The finite-grid table printed "−0.0000". The rate table's note said "ln L technology" although the noise and model are on L, and it did not say that the rate is simulation-based. | **Yes.** The register is split into two floats (symbolic `tab:ra5-claims`, numerical `tab:ra5-claims-num`), both of which fit. The argmin bias now has 5 decimals, with a bias-only caveat. The rate note is fixed. All five tables were compiled and inspected. |
| 7 | Minor | Figure 1(a) legend said "Sample B" (the revision plan renames it "verified sample"). In the rate figure, dashed and solid legend handles were indistinguishable. | **Yes.** |
| 8 | Check | **Could optimizer tolerance fake the n^{-1/4} rate?** In a quartic valley, a stopping rule on the criterion (fatol ∝ nτ²) also produces errors ∝ τ^{1/2}. I re-fitted 30 replications per cell with eight extra Nelder–Mead restarts at 1e-14 tolerances. The 75th and 90th percentiles changed by < 0.1%, and the criterion fell by at most 7e-7 · nτ². The rate is statistical. Slopes carry Monte Carlo error: 0.57/0.50 (q75) on 30 replications against 0.48/0.49 on 100. | Not a bug. Disclosed in memo §5. |

Verified as correct, with no change needed:
- Every sympy identity: I re-derived the generalized FOC; cases (a)–(f); the family FOC; the model-free w and σ\* (via 1/σ − 1 = d ln w/dx along the isoquant); the DMR drift −(2/S)B; and the twin map.
- The brute-force problems: 12 cases; FOC residuals ≤ 1.9e-7.
- Isoquant tracing; the finite-grid moments formula; the Fisher-information parameterization, including the ln M\* gradient.
- The `lnw_bounds` branches for x above, below and inside the set.
- The Webb weights.
- Seeds, and determinism across runs.
- The inherited decomposition 0.161 / 0.143 / 0.047 (from `m7_theory_pi_summary.csv`).
- The cross-sweep slope range. a ∈ [0.370, 0.565] across `m3_wedge_technologies.csv`, rounded to [0.37, 0.57], gives e ∈ [−0.14, 0.26].

## 3. Numbers: memo against regenerated outputs

The builder's numbers were checked against the regenerated tables before any fix. Every number in the builder's memo,
appendix and notes matched its output file to the reported precision. Items checked include:
- the wedge-case errors;
- the family 0.86/0.91/1.06, 1.952 and 0.475, and +4%;
- the model-free σ\* values 0.7601/0.7573/0.7548 and the errors 1.8e-8, 1.7e-10 and 5.5e-8;
- the finite-grid values 0.7345/0.7257, 0.7370/0.7372, +1.3%/+5.9%;
- the MC slopes 0.48/0.49, 0.55/0.56, 1.00, 0.99, and 5.0e-4 against 4.9e-8;
- the INFO slopes 4.00/2.00, with ratio 1.00;
- the DMR −0.080;
- 336 points with 0 errors;
- all partial-identification counts: 126/149, 116/137, 2/3, 21/18, 53/69 and [1.72, 13.5];
- M\*(10²⁴) sets [6.5, 130] and [8.6, 101];
- 5.4 log points (ln(290.6/1.35) = 5.37).

Numbers that changed because of fix #1, all updated in the appendix (Remark A-pi-illus and Remark A-pi-param), the notes
and the memo:

| Quantity | Builder | After review |
|---|---|---|
| Chinchilla anchor M\*(2.9e21), 95% interval | 20.5 [14.8, 28.5] | 20.5 [11.4, 36.8] |
| Llama 3 anchor M\*(1e22) | 22.3 [16.4, 30.5] | 22.3 [12.2, 40.2] |
| In-sample slope e, Chinchilla / Llama 3 | [−0.09, 0.10] / [−0.003, 0.15] | [−0.16, 0.17] / [−0.09, 0.24] |
| M\*(10²⁴) set, cross-sweep slopes | [6.5, 130] / [8.6, 101] | [5.0, 168] / [6.4, 133] |
| w > 1 identified, cross-sweep | 126/149 (85%), 116/137 (85%) | **115/149 (77%), 111/137 (81%)** |
| w < 1 identified / not identified | 2, 3 / 21, 18 | 2, 2 / 32, 24 |
| w > 1 identified, normal inputs only | 53 (36%), 69 (50%) | 46 (31%), 47 (34%) |
| Medians of models' bounds on w (σ\* ∈ [0.6, 0.75]) | [1.72, 13.5] | 1.58 and 16.1 |
| Width of the identified set at 10²⁴ | 3.0 log points | 3.5 log points |

Unchanged: the qualitative conclusions. Llama 3.1 405B's sign is not identified under any set assumption, and the two
point extrapolations disagree. The nesting check (PI.illus) passes.

## 4. Claims: overstatements and precision (fixed in the appendix, notes and memo)

1. **v1's "σ\* first-order identified iff α = β" was "withdrawn", but it is correct conditionally.** With the split A/B
   known, or pinned by the intercept restriction G^{α+β} = αA/βB, the only first-order-flat direction keeps the split fixed.
   Along it σ\* moves at first order if α ≠ β and not at all if α = β. The builder's own proof hints at this ("holds only at
   the true A/B").
   - New numeric check **SI.split** (CORRECTED; PASS). Flat-direction criterion slope 4.000; other directions 2.000–2.006;
     d σ\* slope 1.000 (α ≠ β) and exactly 0 (α = β).
   - Prop. A1(iii)(b), Remark A-positioning (the Kricheli reconciliation), the notes and the memo now say this.
   - The substantive correction stands: with A/B estimated from outcomes, the rate is n^{-1/4} whether or not α = β.
2. **"No estimator converges faster than n^{-1/4}" was stated as a theorem in Prop. A1(iii)(c).** Only the proof called it
   heuristic. Chen (1995) is a result about finite-mixture densities, and Rotnitzky et al. do not apply because P/Q is not
   identified at the truth. The statement now reads "by the analogue of … should not be expected to converge faster …
   uniformly near the truth. This is a heuristic argument, not a theorem for this model". Attainment by least squares
   remains shown by simulation.
3. **Family FOC wording.** The memo and notes said the family's D-condition "identifies the compute-weighted harmonic mean
   w̄_H; equivalently 1 − 1/w̄_H = Σω_i s_i". But w̄_H is computed from the technology, and the second equation is an
   identity. What the condition pins is the π-weighted average of the members' values of compactness. The wording is fixed
   in A9(iii), the notes and the memo; the math was already right.
4. **Prop. A8(iv) signs.**
   - "T̂ overstates under δ > 0" needs δ_S ≥ 0.
   - "A wedge below one cannot be generated by serving demand" needs φ + η ≥ 0 (usage falling with size faster than serving
     cost rises would give m_N < 0).
   - Both qualifiers are added.
5. **Prop. A10(vi)** said on-path designs "identify neither". They identify the frontier slope but not the curvature.
   Fixed.
6. **Remark A-grid** reported bias only (noise-free). A bias–variance caveat is added: narrower grids and quartics raise
   variance.
7. **DMR remark:** "because σ\* > 0" should be "0 < σ\* < 1", since the sign mapping needs σ\* < 1. Fixed.
8. **Partial-identification illustration and the paper.** ra2_wedge runs a harmonized partial-identification analysis with
   different anchors, intervals and slope ranges: PI-1 86% of 77 models; PI-4 23%. The appendix remark previously read as
   a result, and the notes' suggested main-text sentence quoted "about 85 percent".
   - The remark is now labeled an illustration of the logic and points to Section IV.
   - The notes tell writers to quote ra2_wedge's numbers.
   - The table note says the same.

## 5. Referee comments: addressed or not

| Comment | Status | Where / gap |
|---|---|---|
| R1-1 (conduct) | Addressed (theory) | Prop. A8 (general V(L,N); η; direct size effects; T in present value); cases (a), (a′), (b); Remark A-conduct with Berry–Haile, BCS and DMSS. **Gap:** the conduct *tests* the referee asks for are empirical (ra2_wedge has `ra2_wedge_conduct`). |
| R1-2 (tiers, family D) | Addressed | A8(iii)(b), with any w rationalizable; A9 (family FOC, member wedges technological, knife-edge). The empirical split by common-D is in ra2_wedge. |
| R1-5 (cost side) | Addressed (theory) | "FLOP-accounting approximation"; case (f) with the δ_S = δ correction; φ; data costs; c_T as a shadow price; Bond–Söderbom price variation. The sensitivity table is in ra2 (`ra2_wedge_costsens`). |
| R1-7 (model-free) | Addressed (theory) | A10(i)–(vi) plus Remark A-grid. Estimation on the designs is ra2 (`ra2_wedge_modelfree_sigma.csv`) and ra1_modelfree. |
| R1-8a/b/d/e/f/j | Addressed | Remark A-positioning; A1(iii); A3 and remark; Remark A-pi-param; the remark after A6. |
| R1-8c (global vs local) | Addressed, **sharpened in review** | A1(iii)(a)–(c). The conditional validity of v1's statement is now explicit (SI.split); the rate claim is labeled heuristic. |
| R1-4e (PI of M\*(C)) | Addressed (theory); illustration only | A11. The empirical answer (harmonized units, lab-published anchors) is ra2_wedge's PI; ra5's counts are illustrative. |
| R1 minors 9, 10, 11, 12, 14, 15, 19, 47, 48, 49, 51 | Addressed | As listed in the memo §6; each checked in the appendix text. |
| R2 Major 2a/b/c | Addressed | Expenditure-ratio reading w − 1 = λφX_S/X_T (no p); λ; N-proportional internal compute. T in tokens requires p (ra2's calibration). |
| R2 Major 4c | Addressed | Verified exactly (sympy GW.e.sym and brute force, case e). Caveat kept: the student's technology is assumed unchanged by distillation. |
| R3 M5 | Addressed (model); tests are empirical | A8 cases, Remark A-conduct, and the notes' model section. |
| R3 M8 | Addressed | Hao–Merrill framing; Bond–Söderbom/ACF; Sargan, Rotnitzky, Chen; Box–Lucas, Kiefer–Wolfowitz; the Kricheli reconciliation (now with the known-A/B qualifier). |
| R3 M11 | Mostly addressed | Scope; notation (E vs 𝔼; a₁, b₁ appendix-only); v1 Props. 5–6 moved to Appendix E. **Open:** the map keeps six main-text results (Prop. 1, Lemma 1, Props. 2–5), while revision plan A says "≤ 5 formal results". This is the lead author's call. |

## 6. Files changed by the review

- `code/analysis/ra5_theory/partial_id.py`: wild bootstrap-t (HC2), new anchor fields, docstring.
- `code/analysis/ra5_theory/numeric_v2.py`:
  - new `check_singular_split` (SI.split);
  - PI.sharp.num tests both directions;
  - MC objective guarded;
  - FG metric label.
- `code/analysis/ra5_theory/run.py`: wedge-table last column and notes; PI table note (bootstrap-t, coverage, illustration
  only); writes `ra5_theory_singular_split.csv`.
- `code/analysis/ra5_theory/claims_tex.py`: SI.split entry; PI.sharp wording.
- `code/analysis/ra5_theory/figures_v2.py`: anchor band from the new interval; legend labels.
- `paper/sections/appendix_proofs.tex`:
  - A1(iii)(b)–(c) (known-A/B clause; heuristic rate);
  - Remark A-positioning (Kricheli sentence);
  - DMR remark (0 < σ\* < 1);
  - A8(iv) qualifiers;
  - A9(iii) identity wording;
  - A10(vi);
  - Remark A-grid (bias–variance);
  - Remark A-pi-illus (new numbers, inference, "illustration only", pointer to Section IV);
  - Remark A-pi-param (3.5 log points).
- `paper/notes/theory_main_text_v2.md`: wedge table; family bullets; A1 inference bullets; Kricheli bullet; illustration
  numbers; suggested main-text sentence (now points to ra2_wedge); corrections list; check count.
- `output/memos/ra5_theory.md`: post-review numbers; review-fix list; tolerance check; open issues 7–8.
- `code/analysis/ra5_theory/run.py` (tables): claims register split into two floats; finite-grid decimals and caveat; rate-table note.
- All module tables and figures were regenerated. `ra5_theory_singular_split.csv` is new.

## 7. Remaining concerns (not fixable here)

1. **The n^{-1/4} rate has no formal proof for this model.** It rests on simulation (least squares, compact space,
   Gaussian noise on L) plus a heuristic mixture analogy. The appendix now says so in the statement.
2. **The anchor intervals still under-cover slightly**, about 90% for nominal 95% at n = 9–10. They also assume a straight
   path within the design. Any main-text use of anchors should come from ra2_wedge, which has its own inference, and
   should carry the same caveat.
3. **Two partial-identification exercises exist** (ra5 illustration and ra2 application). They must not both be quoted as
   results; the main text should use ra2's.
4. **Six main-text formal results** against the revision plan's "≤ 5".
5. **Lab conduct.** Prop. A8 is the right general frame, but case (a′)'s "λ" is an accounting relabeling, not a
   structural parameter. The conduct tests remain empirical and live in ra2_wedge.
6. **Small overfull boxes** (≤ 6.9 pt) remain in Appendix A.
7. **Cross-module flag: ra2_wedge's anchor intervals look far too narrow.** This affects the paper's main
   partial-identification numbers.
   - For the same Llama 3 minima, the point estimate matches exactly (ln M\* = 3.1057). The intervals do not:
     - ra2 reports [20.8, 23.7], a log-width of 0.13;
     - this module reports [12.2, 40.2], a log-width of 1.19.
   - The cause is in `ra2_wedge/analysis_ra2.py::pi_anchors` and `modelfree.wild_boot`. Those intervals bootstrap only the
     residuals of the per-budget parabola fits.
   - The minima scatter around the fitted straight path with s.d. 0.27 (Llama 3) and 0.30 (Chinchilla). ra2's interval
     implies a per-budget argmin s.d. of about 0.05, so lack of fit across budgets is roughly 5× the within-profile noise.
     ra2's intervals omit it.
   - Unless ra2's reviewer shows that the scatter is digitization error that should be excluded, the PI-1 share (86%)
     rests on anchors that are too tight.
   - Recommendation: take the anchor uncertainty from the between-budget residuals, as here (wild bootstrap-t), or from
     both sources.
   - Not changed here (not this module's code).
