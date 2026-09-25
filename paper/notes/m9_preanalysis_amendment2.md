# Amendment 2 to the pre-analysis plan (m9): definitions, restated map, robustness conditions and calibration

**Time of writing.** Written on 2026-09-25 from 11:45 (+03). The numbers in items 1 and 6 were filled in from the
design-only power and coverage study v2 (item 6) at 16:10 (+03); that study reads no endpoint loss. First complete text
at 16:11 (+03). An independent review from 16:20 (+03) corrected the record of what had been seen, made the map exact
(order of classification, robustness, placeholder sentences, the abstract rule of item 3.1) and completed the operating
characteristics of the tilt rule (`paper/notes/round3_WP1_review.md`), before any FineWeb technology estimate; final
text completed at 16:55 (+03) on 2026-09-25. After 16:55 and before the commit, two edits were made: the integrator
added the last bullet of item 10 (placeholder wording only), and an editor replaced every dash used as punctuation by
other punctuation (the author's style rule). No rule, threshold, factor or statement of what had been seen changed. The
lead author confirmed the change of the tilt test in item 6 before the commit. The commit hash and the times of the commit and of the push to
the public repository are those recorded by git and by GitHub. They are cited in Online Appendix B4 and are deliberately
not written into this file, so that the committed text never needs a later edit (Amendment 1's committed header
misstated its own time of writing).

**Standing.** This amendment adds to the plan (`paper/notes/m9_preanalysis_plan.md`, commit `bd5c0ad`, 24 September
03:12), to deviation D8 (analysis memo, section 0.3; committed in `a5f47f8`, 17:44) and to Amendment 1 (`a5f47f8`, 17:44;
its stated time corrected in `b2e9bf8`, 17:45). Where it differs from them, it governs. It answers the round-3 reports
(Referee 1, New Majors 2 and 3, minors 14 and 15; Referee 2, New Majors 4 and 5 and section 4; Referee 3, comment F and
section 5; Referee 4, N1 and section 5) and implements items X1 to X3 of `paper/notes/round3_fixlist.md`.

## What had been seen when this was written

1. **Preliminary FineWeb-Edu estimates** from the pipeline's dry run (passes at 17:12, 17:19 and 17:24 on 24 September;
   `output/tables/m9_sweeps_q1_sigma.csv` 17:26:51, `_q1_levels.csv` 17:26:52, `_q3_extrap.csv` 17:28:44; analysis memo,
   sections 1 and 5), main grid, 44 endpoints:
   - model-free σ\* of 0.666 in convention P (plan interval [0.659, 0.673], CR2 interval [0.652, 0.683]), 0.665 in P6
     and 0.607 in T ([0.602, 0.614] and [0.596, 0.626]);
   - κ-family σ\* of 0.662 (P) and 0.605 (T); Chinchilla form (κ = 1) 0.720 and 0.654; κ̂ of 0.36 (P) and 0.46 (T);
   - per-level σ\*_b rising with compute, from 0.65 to 0.68 in P and from 0.56 to 0.64 in T;
   - the Q3 slope in P, −0.29 (plan [−0.40, −0.09], CR2 [−0.44, −0.005]; not decided under D8), and the level gap,
     −0.69 ([−0.75, −0.60]);
   - the other FineWeb-Edu results of memo sections 1 and 5 (samples, bandwidths, learning-rate corners, on-path
     profiles, residual s.d. 0.0059).
2. **FineWeb quantities that are not technology estimates**, from the same passes: the FineWeb learning-rate
   calibration (`_q5_calibration.csv`: FineWeb's optimum at width 128 lies above the rule, Δ ln LR\* = −0.51); the
   learning-rate corners and their drift with D for both corpora (`_q5_corners.csv`, `_q5_drift.csv`); and the
   between-corpus log-loss differences at 15 matched FineWeb main-grid cells (widths 128 to 256) and 15
   learning-rate-corner cells, on both validation sets and, for the corners, WikiText-103 (`_matched_cells.csv`,
   `_crosseval.csv`).
3. **Raw endpoint losses.** Since the dry run, the loss of every finished run of both corpora has been written to
   `data/processed/sweep/results.jsonl` and printed in the queue logs. When this was written these included all 44
   FineWeb main-grid endpoints (the last, width 640 at 200 million tokens, rerun after the second reboot, finished at
   12:19 on 25 September), the FineWeb high-M runs, the FineWeb-Edu high-M, learning-rate and seed runs, and the first
   retrained FineWeb-Edu endpoints of item 9; runs that finish before this amendment is committed are in the same
   position. Some of these lines were displayed while the queues were monitored, and some while this amendment was
   prepared and reviewed (log lines of FineWeb-Edu learning-rate and seed runs and of the rerun FineWeb width-640
   endpoint). No model has been fitted to any FineWeb loss, and none of the statistics of point 5 has been computed.
4. **The round-3 reports and audits** (25 September), which quote items 1 and 2.
5. **Not computed by anyone:** any FineWeb technology estimate (σ\*, exponents, allocation exponent, M\*, local or
   parametric wedge), the Q1 test of equal σ\*, any tilt or other corpus-pair statistic (Q2), and any FineWeb or
   between-corpus Q3 statistic. The files that would hold them hold FineWeb-Edu only (`_q1_sigma`, `_q1_levels`,
   `_q3_extrap`, `_q3_points`) or are empty (`_q2_tilt`; `_q2_decision` reads "pending: FineWeb main grid
   incomplete"). No m9 pipeline output has been written since 17:33 on 24 September except
   `data/processed/m9_sweeps/bytes_per_token.json` (00:49 on 25 September), which holds the bytes per token of the
   evaluation sets and no loss, and the design-only outputs of item 6 (`output/tables/m9_sweeps_power_v2*`), which read
   no loss.

## 1. Confirmatory and exploratory content

- In this design the FLOP-effective parameter count N_F = C/(6D), with C the actual training FLOPs, is 1.030 to 1.051
  times the total count on the main grid and 1.071 times at the (128,4) high-M shape (`m9_sweeps_design.csv`; Table B2
  note). Amendment 1 made N_F the comparison convention, in place of the plan's total count (T), and drew the band
  [0.60, 0.76] after FineWeb-Edu's estimate in T (0.607) had been seen. By construction, the noise-free model-free
  σ\* in N_F lies only 0.001 to 0.002 above the one in T, and 0.04 to 0.05 below the one in P, under the Chinchilla-form
  and both κ-family truths on both corpora's designs (item 6; `m9_sweeps_power_v2_offsets.csv`). FineWeb-Edu's position
  relative to the band was therefore effectively known when the band was drawn.
- **FineWeb-Edu's Q1 classification and its Q3 slope are exploratory.**
- **The confirmatory content of the experiment is the FineWeb arm of Q1, the Q1 test of equal σ\* across corpora, Q2,
  and the FineWeb arm of Q3.** None of their inputs had been estimated when this was written (see above).
- The registered map still applies to both corpora: a FineWeb-Edu estimate outside the band still triggers Q1.3. A
  FineWeb-Edu estimate inside the band is not reported as confirmation of anything.
- Amendment 1 states no rationale for the band. It stays as written, because redrawing it now would be a second choice
  made after FineWeb-Edu's estimate had been seen. Each corpus's estimate is also reported against the study-level
  interval [0.640, 0.734] of the public designs (in N_F) and against the study-level mean over budgets up to
  3×10^20 FLOP, with a sentence for each corpus on whether it lies inside that interval.

## 2. Inside, outside and undetermined (Q1)

- **Statistic.** Model-free σ\* in N_F: the w = 1 path of C = 6 N_F D, main-grid sample, mean over the valid compute
  levels, for each corpus.
- **Intervals.** The plan's wild cluster bootstrap interval; and the CR2-scheme interval (CR2 residuals around the
  local-quadratic surface, bandwidth re-selected in every draw) with its half-width multiplied, about its midpoint, by
  the model-free calibration factor of item 6, called the calibrated CR2 interval below. The same construction, with the
  factor of item 6 for each statistic, gives every calibrated interval in this amendment.
- **Classification of each corpus**, applied in this order:
  1. **Outside [0.60, 0.76]:** the plan's interval and the calibrated CR2 interval both lie entirely below 0.60, or both
     lie entirely above 0.76, whatever their width.
  2. **Undetermined:** otherwise, if the calibrated CR2 interval has a half-width above 0.10 (the neutrality threshold of
     Q2).
  3. **Inside:** otherwise, if the point estimate lies in [0.60, 0.76].
  4. **Undetermined:** in every other case (for example, a point estimate below 0.60 whose calibrated CR2 interval
     reaches 0.60).
- **Outcomes.** Q1.1 and Q1.2 need both corpora inside; Q1.3 needs either corpus outside; Q1.4 applies otherwise.
- **Robustness of a classification.** A corpus's classification is robust if steps 1 to 4 give the same result when the
  seed-covariance interval replaces the calibrated CR2 interval; a rejection of equality (below) is robust if the
  seed-covariance interval for the difference also excludes zero. If a corpus has no seed replicates, robustness is
  judged on the other two schemes and the text says so (D8: "once seeds exist").
- **Equality (Q1.1 against Q1.2).** "Rejected" means that the plan's interval for σ\*_edu − σ\*_web (joint draws, one
  weight per width shared by both corpora) and the CR2-scheme interval for the difference, with its half-width
  multiplied by the equality factor of item 6, both exclude zero. The minimum detectable difference at 80 percent power,
  before and after calibration, is reported beside the test.
- The seed-covariance interval is reported beside the other two, but the classification uses the two schemes that
  Amendment 1 names. An experimental number enters the abstract only if it is robust under all three schemes (plan, CR2
  scheme and seed covariance), as D8 requires: the abstract changes under Q1.1 or Q1.2 only if both corpora's
  classifications are robust, and under Q1.3 only if the classification of at least one corpus classified outside is
  robust. Otherwise the abstract stays as it is (subject to item 3.1), and the other sentences of the outcome apply with
  the words "(not robust to the seed-covariance bootstrap)" after the classification; the same words follow the Q1.2
  sentences if the rejection of equality is not robust.

## 3. Additional commitments on the headline

1. If FineWeb's N_F estimate lies below 0.640, the lower end of the study-level interval, under all three schemes (the
   plan's interval, the calibrated CR2 interval and the seed-covariance interval each lie entirely below 0.640), the
   abstract names the public designs and adds the experiment's values, whatever the band gives. This abstract replaces
   the abstract change of item 4 under every Q1 outcome, including Q1.3; the other sentences of the outcome still
   change as item 4 says. Exact change: "Annealed designed experiments give" becomes "Public annealed designs give"
   (word-neutral), the rest of the σ\* sentence is unchanged, and the clause of Q1.1 ("A controlled experiment at
   2×10^14–10^17 FLOP gives [x, y] for two corpora.", 12 words) follows it, with the five cuts of the fix list (M9-14,
   10 words), so that the abstract has 98 − 10 + 12 = 100 words by the project's count.
2. If either corpus's point estimate lies below 0.640 and its within-experiment drift (item 7.5) is positive, with a
   CR2-scheme interval widened by the model-free factor of item 6 that excludes zero, Section III.E adds: "Below the
   public designs' level and rising with compute inside the experiment, σ\*(C) is then non-monotone across studies or
   specific to the recipe." The abstract states only levels (item 3.3).
3. The within-experiment drift of σ\*_b is exploratory under every outcome. It never enters the abstract, the
   introduction or the conclusion's findings; it is reported in Section III.E (and Figure 3(d)) only.
4. No experimental value below 0.65 is described as "close to 0.7" or "about 0.7".

## 4. The map restated against the current text

"Current text" is the round-3 revision of the abstract, the introduction, Sections III.E and IV.D and the conclusion, as
specified in `paper/notes/round3_fixlist.md` (items H1, H4, H6, T9, W10 to W12 and H11). The numbers that other work
packages fill into those sentences do not affect the map. The abstract changes only under Q1; no Q2 or Q3 outcome
changes it.

**Q1** (statistic and definitions in item 2; x and y are FineWeb's and FineWeb-Edu's N_F values, to two decimals,
written "x and y" where the text has "[x, y]"). The introduction and conclusion placeholders take the sentences of
"Placeholder sentences" at the end of this item under every outcome.
- **Q1.1, both inside and equality not rejected.**
  - Abstract: after "... a parameter-data elasticity of substitution near 0.7 at 10^19–3×10^20 FLOP and about 0.6 at two
    designs' largest budgets." add "A controlled experiment at 2×10^14–10^17 FLOP gives [x, y] for two corpora." (12
    words), offset by the cuts listed in the fix list, M9-14.
  - Section III.E: the full report. Section IV.D: no change; its Q1.3 placeholder is deleted.
- **Q1.2, both inside and equality rejected.** As Q1.1, and in addition:
  - Section III.C, one sentence: "In our controlled experiment σ\* differs between FineWeb-Edu and FineWeb by [Δ]
    (calibrated interval [a, b]; on this design the test rejects a true zero difference [r] percent of the time), so
    curvature can depend on data quality."
  - Section IV.D, Magnitudes, after the sentence on the medians of the bounds: "With the curvature of each of our
    corpora instead, the medians are [A] and [B] on FineWeb and [C] and [D] on FineWeb-Edu."
- **Q1.3, either corpus outside.**
  - Abstract (unless item 3.1 applies): "near 0.7 at 10^19–3×10^20 FLOP and about 0.6 at two designs' largest budgets"
    is replaced by "between X and Y across designs, compute levels and our controlled experiment", where X and Y are the
    smallest and largest of 0.59 (the top-budget value), 0.70 and each corpus's N_F estimate, to two decimals.
  - Section IV.D, Magnitudes: the Q1.3 placeholder at the end of that paragraph reads "Because our experiment's σ\* lies
    outside the band of 0.60 to 0.76 on [FineWeb / FineWeb-Edu (exploratory) / both corpora], the curvature range widens
    to [X]–[Y] and the medians of the lower and upper bounds on s become [A] and [B]." The range ("1/σ\*−1 between 0.40
    and 0.68") is widened to include 1/σ\*−1 at the value of each corpus classified outside, and the medians are
    recomputed.
  - Conclusion: after "... about 0.6 at the largest budgets of the two IsoFLOP designs that go beyond" add "; our
    controlled experiment gives [x] at 2×10^14–10^17 FLOP" (x of each corpus classified outside, FineWeb-Edu's followed
    by "on FineWeb-Edu (exploratory)"), and in "What economists should use", after the recommendation for σ\*: "Our
    controlled experiment gives [x] at 2×10^14–10^17 FLOP, a level for small models only."
  - If only FineWeb-Edu is outside, the abstract changes in the same way (the map applies to both corpora), and every
    other sentence carries "on FineWeb-Edu (exploratory)".
- **Q1.4, otherwise.** Section III.E and Appendix B report it; Section IV.D's Q1.3 placeholder is deleted. Abstract
  unchanged (unless item 3.1 applies).

**Q2** (tilt χ̂ = Δ ln(A/B), FineWeb-Edu minus FineWeb, corpus-pair model with common exponents. Under the plan's
scheme, "excludes zero" means that the plan's interval excludes zero, and "neutral" that it contains zero with a
half-width below 0.10. Under the CR2 scheme, as revised in item 6, the same definitions apply to the calibrated CR2
interval; D8's restricted test is reported beside it. Each verdict needs both in-distribution validation sets, and
"factor-biased" needs the same sign on both. Q2.1 needs "factor-biased" under both schemes, Q2.2 "neutral" under both;
every other combination, including a disagreement between the schemes, is Q2.3.)
- **Under every outcome.** The implied M\* factor at 10^15, 10^16 and 10^17 FLOP, with its calibrated CR2 interval, is
  reported in Section III.E and marked on the breakdown frontier as a small-scale magnitude beside DataDecide's (the
  placeholder of the note to Online Appendix Figure `fig:app-breakdown`), and the tilt is never used as a
  production-scale allowance τ (fix-list decision D-4). If the Wald test of the plan's Q2(a) rejects common exponents,
  the registered decision is still made and reported, labeled "(common exponents rejected)", but no single tilt is
  quoted outside Section III.E: every sentence below that quotes χ̂ quotes instead the M\* factor at 10^16 FLOP and the
  model-free ratio M\*_edu/M\*_web of item 7.3.
- **Q2.1, factor-biased** on both validation sets with the same sign under both schemes.
  - Robust only under the conditions of item 5. If robust: Section III.E reports χ̂, the implied M\* factor at 10^15,
    10^16 and 10^17 FLOP and the ŵ factor e^(−χ̂); Section III.C adds "In our controlled experiment the quality filter
    is factor-biased: FineWeb-Edu's tilt of [χ̂] (calibrated interval [a, b]) changes M\* by a factor of [Y] at 10^16
    FLOP, [a quality effect / an effect of quality or distribution match]."; Section IV.D's Q2 placeholder reads "The
    experiment's tilt between its two corpora, χ̂ = X, implies an M\* factor of Y at 10^15–10^17 FLOP, a magnitude at
    small scale beside DataDecide's (Online Appendix Figure `fig:app-breakdown`)."
  - Not robust: Section III.E reports the main-grid verdict and each condition it fails; the introduction and
    conclusion placeholders call it "factor-biased on the main grid but not robust"; Section IV.D reports it as
    inconclusive, its Q2 placeholder reading "The experiment's test of the tilt between its two corpora is
    inconclusive: factor-biased on its main grid but not robust (χ̂ = X)."
  - Label: "quality" only if the FineWeb-Edu advantage has the same sign on C4 (where distribution match favors
    FineWeb) and on the other neutral sets where matched cells exist, and does not rest on width-128 cells alone;
    otherwise "distribution match or quality".
- **Q2.2, neutral.** Section III.C adds "In our controlled experiment the quality filter shifts the level of loss but
  not the optimal mix (tilt [χ̂], calibrated interval [a, b])."; Section IV.D's Q2 placeholder reads "The experiment's
  tilt between its two corpora, χ̂ = X, is neutral, with an M\* factor of Y at 10^15–10^17 FLOP, so recipe tilt remains
  bounded by DataDecide's (Online Appendix Figure `fig:app-breakdown`)."
- **Q2.3, inconclusive.** Section III.E reports it, with each scheme's verdict; Section IV.D's Q2 placeholder reads
  "The experiment's test of the tilt between its two corpora is inconclusive (χ̂ = X)." No other change.
- A verdict that the seed-covariance bootstrap does not reproduce carries "(not robust to the seed-covariance
  bootstrap)" wherever it is stated outside Section III.E.

**Q3** (statistic: the plan's slope of ln(w_{M≤100}/w_local) in ln M over endpoints with M > 100, Chinchilla form
fitted on M ≤ 100, main grid plus high-M runs, convention P, against the null of item 6 recomputed at the realized
noise).
- **Baseline** (fix-list decision D-5). Introduction: "Inside the designs, parametric forms understate the wedge at high
  M in all three recipes that allow the check, but the local log wedge is convex in two and concave in one, so the
  direction of extrapolation error in M beyond the designs is recipe-dependent." Section IV.D: "Inside the designs,
  then, parametric forms understate the wedge at high M in three recipes, but the direction of the error beyond them is
  recipe-dependent." Conclusion: "Parametric forms understate the wedge at high tokens per parameter inside three
  recipes' designs, but the direction beyond them is recipe-dependent and nothing is observed at frontier compute."
- **Definitions.** Below: the plan's interval and the calibrated CR2 interval both lie entirely below the null. Above:
  both lie entirely above it. Undetermined: otherwise. The map is applied to FineWeb's slope (confirmatory);
  FineWeb-Edu's slope is reported beside it (exploratory), and if the two lie on opposite sides of their nulls, Sections
  III.E and IV.D say so.
- **Placeholders under every outcome.** Section IV.D's Q3 placeholder is "Our experiment repeats the check under
  question Q3 of its pre-analysis plan and Amendments 1 and 2, up to M ≈ [X] in non-embedding parameters." followed by
  the sentence of the outcome below; if FineWeb-Edu's slope lies on the other side of its null, it adds "On FineWeb-Edu
  (exploratory) the slope lies [above / below] its null." Online Appendix E's placeholder reports the slope, its null,
  both intervals, the decision and the level gap by bin of M (descriptive) for each corpus.
- **Q3.1, below the null.** Introduction and Section IV.D: "in all three recipes that allow the check" and "in three
  recipes" become "in all four recipes that allow the check (three public recipes by level, ours by the slope of its
  pre-analysis plan)"; the conclusion's "inside three recipes' designs" becomes "inside four recipes' designs". The
  recipe-dependent clause on the direction beyond the designs stays. Section IV.D's placeholder sentence: "The
  extrapolation slope lies below its design-specific null, so the experiment is the fourth recipe in which parametric
  forms understate the wedge at high M." The conclusion's Limitations placeholder ("whether our high-M runs confirm this
  direction") is deleted.
- **Q3.2, undetermined.** The text is unchanged. Section IV.D's placeholder sentence: "The extrapolation slope is not
  distinguishable from its design-specific null, so our experiment does not decide." The conclusion's Limitations
  placeholder reads "Our high-M runs do not decide the direction inside our design."
- **Q3.3, above the null.** Introduction and Section IV.D: "... in all three public recipes that allow the check, while
  our experiment shows the opposite direction inside its design"; conclusion: "inside three public recipes' designs but
  not inside our experiment's". Section IV.D's placeholder sentence: "The extrapolation slope lies above its
  design-specific null, so inside its design our experiment shows the opposite direction, and the magnitude bounds below
  are widened." The conclusion's Limitations placeholder is deleted. The magnitude bounds of Section IV.D widen, as
  Amendment 1's Q3.3 requires: the lower end of the curvature range is lowered to the smallest model-free local
  elasticity ∂ln w/∂ln M at fixed compute that the experiment's surface gives at its endpoints with M > 100 (convention
  P), if that is lower, and the medians of the bounds are recomputed.
- **Which statistic supports which recipe** (Referee 1, New Major 2(d)): Farseer, Marin and Llama 3 by the level of
  ln(w_param/w_local) by bin of M (Figure 5); our experiment by its slope against the design-specific null. The
  experiment's level gap by bin of M is reported as descriptive, comparable with Figure 5, and does not by itself count
  as a fourth recipe.

**Placeholder sentences of the introduction and the conclusion** (every outcome; square brackets are filled, slashes
choose by outcome; one interval per sentence).
- **Introduction**, the placeholder after "... on two corpora that differ in quality.": "In FLOP-effective parameters its
  model-free σ\* is [x] on FineWeb (calibrated 95 percent interval [a, b]) and [y] on FineWeb-Edu (exploratory), [Q1
  phrase]; the quality filter [Q2 phrase]; and its extrapolation slope [Q3 phrase]."
  - Q1 phrase: Q1.1 "inside the band of 0.60 to 0.76 fixed before any FineWeb estimate, and equal curvature across
    the corpora is not rejected"; Q1.2 "inside the band of 0.60 to 0.76 fixed before any FineWeb estimate, but
    different across the corpora"; Q1.3 "outside the band of 0.60 to 0.76 fixed before any FineWeb estimate on
    [FineWeb / FineWeb-Edu / both corpora], so σ\* depends on the design and the scale"; Q1.4 "but the experiment does
    not determine where [FineWeb / FineWeb-Edu / both corpora] lie relative to the band of 0.60 to 0.76 fixed before any
    FineWeb estimate".
  - Q2 phrase: Q2.1, robust, "is factor-biased, [a quality effect / an effect of quality or distribution match]";
    Q2.1, not robust, "is factor-biased on the main grid but not robust"; Q2.2 "is neutral"; Q2.3 "has no determined
    effect on the optimal mix".
  - Q3 phrase (FineWeb): Q3.1 "lies below its design-specific null"; Q3.2 "is not distinguishable from its
    design-specific null"; Q3.3 "lies above its design-specific null".
- **Conclusion**, the placeholder after "... its level depends on the technology.": under Q1.1, Q1.2 and Q1.4, "Our
  controlled experiment at 2×10^14–10^17 FLOP gives a σ\* of [x] on FineWeb and [y] on FineWeb-Edu (exploratory), [Q1
  phrase], and its quality filter [Q2 phrase]."; under Q1.3, whose σ\* clause goes into the findings sentence (above),
  "In our controlled experiment the quality filter [Q2 phrase]."

## 5. Conditions for a robust factor-biased verdict (Q2)

- **Governing sample.** The registered decision uses the main grid of both corpora, as the plan specifies.
- **Robust** only if the seed-covariance bootstrap gives the same verdict on the main grid (D8; if both corpora have
  seed replicates), and, in addition, the same verdict (both validation sets, same sign, both schemes) holds:
  1. on the four-layer floor sample (widths 256 to 640), which removes the two- and three-layer models and the
     width-128 learning-rate problem (analysis memo, sections 7.2 and 7.3);
  2. with D measured in bytes: tokens times the bytes per token of the training stream the run used (`meta.json`:
     3.887 for FineWeb-Edu and 3.751 for FineWeb on the main stream, and the stored byte and token counts of the
     extension stream for the high-M runs), since a shared tokenizer compresses the corpora differently;
  3. under a flexible-input bound on the tilt: for each corpus, the inefficiency gradient ι_n − ι_d of its
     learning-rate policy is measured from its own learning-rate corners exactly as for the plan's Q5 bound on a
     (transverse gradient between the (640, 25M) and (128, 800M) corners, at about 7 to 8×10^15 FLOP); eq. (D2)
     (`eq:app-flexbias`) converts it into a first-order shift of that corpus's log argmin at fixed compute, δ_r =
     −Δ_r/f''_r; the spurious tilt is (α+β)(δ_edu − δ_web); the verdict is robust to tuning only if both intervals,
     moved toward zero by the absolute value of that spurious tilt, still exclude zero.
- **M\* factor.** The implied factor for M\* is reported at 10^15, 10^16 and 10^17 FLOP (the last just beyond the main
  grid's 7.3×10^16, from the corpus-pair model only), with the model-free ratio of item 7. If the Wald test of the plan's
  Q2(a) rejects common exponents, no single tilt is reported: the factor is reported by compute level only.
- The tilt is never used as a production-scale τ.

## 6. Calibration factors and the Q3 null

- **Design-only study v2** (`code/analysis/m9_sweeps/m9_power_v2.py`, `m9_coverage_v2.py`; outputs
  `output/tables/m9_sweeps_power_v2*.csv`; it reads no endpoint loss). Same data-generating process as the original
  power study (Chinchilla-form truth with Besiroglu et al.'s exponents in non-embedding N, anchored at M\* = 20; noise
  with a within-trunk share ρ; FineWeb-Edu = FineWeb with ln B lower by χ), plus κ-family truths and the N_F
  convention. Replication counts and Monte Carlo standard errors are in the output files.
- **Model-free σ\*.** The CR2-scheme interval (bandwidth re-selected in every draw) is widened by the largest ratio of
  Monte Carlo s.d. to mean bootstrap s.e. that v2 gives across noise levels (s.d. 0.002, 0.005 and 0.010), conventions
  (N_F, P, P6, T), within-trunk shares (0.5 and 0.9) and truths (Chinchilla form; κ family with κ = 0.36):
  **1.74** (Monte Carlo s.e. 0.13; attained at within-trunk share 0.9, N_F, FineWeb-Edu design), replacing D8's 1.6 to
  1.7. At noise s.d. 0.005 the CR2-scheme interval for the model-free σ\* in N_F covers 0.74 (MC s.e. 0.01) and the
  calibrated interval 0.92 (0.01); at within-trunk share 0.9 the calibrated interval covers 0.85 (0.02), so a statement
  that rests on it near its boundary is labeled as such (`m9_sweeps_power_v2_coverage.csv`, `_factors.csv`).
- **Why calibrated coverage stays below 0.95.** Coverage is measured against each estimator's noise-free value on the
  design, which excludes smoothing bias. Across the v2 cells the calibrated intervals cover 0.84 to 0.96 (both designs
  pooled; 0.81 to 0.95 for the difference between corpora), because a factor of this kind corrects the bootstrap s.e.,
  while the basic interval's half-width is only about 1.8 to 1.9 bootstrap s.e. (the wild bootstrap distribution over
  eight clusters has light tails) and the s.e. varies across replications. The v2 replications use 99 bootstrap draws
  (49 in the sensitivity cells) against 999 in estimation; rerun with 999 draws (eight tilt replications, three coverage
  replications), the ratio of half-width to s.e. rose by about 2 percent (1.85 against 1.81), so these rates apply to
  the estimation, if anything slightly pessimistically.
- **Q3 slope.** Widened by the larger of 1.75 (the upper end of the original study's range) and the largest v2 factor
  across noise levels and truths (1.50, MC s.e. 0.10): **1.75** (replacing D8's 1.4). The calibrated interval then
  covers 0.91 (0.01) at noise s.d. 0.005.
- **Q1 equality test.** The CR2-scheme interval for σ\*_edu − σ\*_web (model-free, N_F) is widened by **1.64** (MC
  s.e. 0.10), the largest v2 factor for the difference. At noise s.d. 0.005 the calibrated test rejects a true zero
  difference 10 percent of the time (MC s.e. 1.3 points), in part because the design's noise-free difference is 0.005,
  and 21 percent at within-trunk share 0.9; the uncalibrated CR2 test rejects 26 percent. The smallest difference
  detectable with 80 percent power is 0.068 for a correctly sized test, 0.054 for the uncalibrated CR2 test and 0.075 for
  the calibrated one. A rejection is reported with this size, and Q1.2's sentence in Section III.C says so.
- **Tilt test (a change to D8's test, on design-only evidence, before any tilt has been computed).** With 500
  replications per value of χ (noise s.d. 0.005, within-trunk share 0.5, two validation sets with noise correlated
  0.8), D8's restricted wild cluster bootstrap on CR2 residuals, which the 40-replication study had found correctly
  sized (0.050, with power 0.83), rejects a true χ = 0 on one validation set 9.2 percent of the time (Monte Carlo s.e.
  1.3 points) and a true χ = 0.22 only 65 percent of the time (s.e. 2.1 points): the residuals of the fit that imposes
  χ = 0 carry the tilt it imposes away, which widens the bootstrap distribution. Its two-set rule calls a true χ = 0
  "factor-biased" with probability 0.030 (0.008), but a true χ = 0.22 "factor-biased" with probability only 0.47 (0.02)
  and "neutral" with probability 0.15 (0.02). The CR2 interval with its half-width multiplied by **1.25**, the largest
  ratio of the Monte Carlo s.d. of χ̂ to its mean CR2 bootstrap s.e. (over χ in {0, 0.22} and both validation sets),
  excludes a true χ = 0 on one validation set 11.8 percent of the time (1.4 points), more often than D8's test. Its
  two-set rule calls a true χ = 0 "factor-biased" with probability 0.054 (0.010), against 0.030 for D8's rule, and
  "neutral" with probability 0.82 (0.02), and a true χ = 0.22 "factor-biased" with probability 1.00 (never "neutral").
  The change thus accepts a false "factor-biased" rate about 2.4 points higher to remove a 15 percent chance of calling
  a tilt of DataDecide's size neutral and a 38 percent chance of calling it inconclusive. **In the CR2 scheme,
  "excludes zero" for the tilt therefore means that the calibrated CR2 interval excludes zero, and "neutral" that it
  contains zero with a half-width below 0.10, on both in-distribution validation sets.** D8's restricted test is
  reported beside it with these operating characteristics. The plan's rule, which calls a true χ = 0 "factor-biased"
  with probability 0.13 (0.02), stays in every table as registered; a verdict is robust only if the plan's scheme, the
  CR2 scheme and the seed-covariance bootstrap agree (in the study, the plan's rule and the new CR2 rule both call a
  true χ = 0 "factor-biased" in 5.4 percent of replications). The minimum detectable tilt at 80 percent power is 0.048
  (normal approximation with the Monte Carlo s.d.). Source: `output/tables/m9_sweeps_power_v2_tilt.csv`.
- **Out-of-range noise.** If the seed s.d. measured in Q4 lies outside 0.002 to 0.010, or the within-trunk correlation
  above 0.9, the factors are recomputed with the v2 code at the measured values, and the larger of the recomputed and
  the fixed factor is applied. This is done and written into the analysis memo before any Q1 to Q3 statistic is
  computed.
- **Q3 null at the realized noise** (Referee 1, section 6 item 16; Referee 2, 4.E). For each corpus: fit the Chinchilla
  form on its endpoints with M ≤ 100 (convention P); simulate 500 replications of ln L at every design endpoint (main
  grid and high-M runs) from that fit plus noise with the seed s.d. and within-trunk correlation measured in Q4; compute
  the Q3 slope in each replication exactly as in the estimation. The null is the mean, reported with its Monte Carlo
  s.e.; a sensitivity uses the upper ends of the Q4 intervals. This is computed after Q4 and before the Q3 statistic. The
  design-only null (Chinchilla truth, within-trunk share 0.5, 500 replications, MC s.e. at most 0.007) is −0.006 on
  the FineWeb-Edu design and −0.004 on the FineWeb design without noise, and −0.023 and −0.014 at noise s.d. 0.002,
  −0.052 and −0.034 at 0.005, −0.092 and −0.045 at 0.010, and −0.107 and −0.056 at 0.015; it barely moves with the
  within-trunk share (−0.044 to −0.053 on the FineWeb-Edu design at 0.005 for shares 0 to 0.9). Under the κ-family
  truths the slope is −0.13 to −0.20 without noise and −0.21 to −0.34 at 0.005 (`m9_sweeps_power_v2_q3null.csv`).

## 7. Secondary analyses (labeled as secondary in every exhibit)

1. **κ-free restricted Q3 fit.** The Q3 statistic with the κ family, instead of the Chinchilla form, fitted on M ≤ 100.
   It governs the interpretation of a slope below the null: a Chinchilla-form slope below its null can reflect κ < 1
   rather than convexity (the power study's κ-truth row gives −0.13 without noise).
2. **Convexity decomposition** per corpus (Referee 2, New Major 2): regress the model-free local ln w on
   u = ln(M/M\*_local(C)) and u² through the origin over grid points inside the design (w = 1 path, N_F), and report b1,
   b2 and, by bin of M, the gap of the parametric wedge from a linear extrapolation of local ln w from its slope at the
   path, as for the public recipes (fix list, T1.9).
3. **Model-free ratio M\*_edu/M\*_web** at each compute level where both corpora's local paths exist, from the roots of
   the local wedge (the path points of Q1), with CR2-scheme intervals. It does not depend on the functional form of the
   tilt.
4. **Shape consistency of (128,4).** The difference between the (128,4) endpoints at 0.2 and 0.8 billion tokens and the
   prediction of the main-grid local-quadratic surface (main-grid endpoints only) at their (N, D), in conventions P and
   T, with CR2-scheme intervals. The shape is consistent with the ladder if the calibrated interval contains zero.
5. **Within-experiment drift** of σ\*_b (exploratory; item 3): the slope of the per-level σ\*_b in log10 C, only over
   compute levels at which the set of widths in the local fits is fixed.

## 8. Exploratory trunk-versus-endpoint comparison

- **Cells** (from `data/processed/sweep/results_trunk.jsonl`, verified on 25 September): unannealed trunk losses exist
  only for trunks launched after the trainer update of 17:44 on 24 September, and never at a trunk's last budget:
  - FineWeb main grid, widths 320 to 640: 25 to 400 million tokens at widths 320 and 384, 25 to 200 million at 448 and
    512, 25 to 100 million at 640;
  - FineWeb-Edu main grid, widths 384 and 512, from the retrained trunks of item 9 (25 to 400 and 25 to 200 million);
  - seed replicates at the rule's learning rate: FineWeb-Edu widths 128, 256 and 384 at 50 and 100 million tokens (two
    seeds each), the width-128 seed corner at 200 and 400 million, and FineWeb's seed replicates, which have not yet run,
    at the same cells;
  - high-M shapes: FineWeb-Edu (256,4) at 0.2, 0.8 and 1.6 billion tokens; FineWeb (128,4) at 0.2 billion (and 1.6
    billion from its rerun after the second reboot); the (128,4) learning-rate and weight-decay runs of FineWeb-Edu,
    which are not at the rule and are reported separately.
- **Statistic 1 (FineWeb).** Δσ\* = σ\*(unannealed) − σ\*(annealed), model-free, N_F (and P), on the identical cells of
  FineWeb widths 320 to 640, with a calibrated CR2-scheme interval for the difference (joint draws). Coverage: M up to
  65 in convention P (44 in N_F). It cannot test the high-M part of the explanation for Porian et al.'s estimates.
- **Statistic 2 (both corpora).** The annealing gap g = ln L_unannealed − ln L_annealed at every cell with both, regressed
  on ln N and ln D with CR2 inference by trunk; its elasticity in ln M at fixed compute shows whether annealing changes
  the curvature of isocost profiles. It covers FineWeb widths 320 to 640, FineWeb-Edu widths 384 and 512, the seed
  replicates (M up to 254 in P at width 128, and 1,016 at the seed corner) and the high-M shapes (M up to 508 at
  FineWeb-Edu (256,4), and 254, or 2,031 after the rerun, at FineWeb (128,4)).
- **Implication for the Porian exclusion** (Referee 3, 5(e)). If the calibrated interval for Δσ\* lies entirely above
  −0.15, the gap between Porian et al.'s 0.51 and the annealed designs, annealing cannot account for that gap on this
  grid, and the reason given for excluding Porian et al. from the study-level mean fails there: Table 1 Panel C gets a
  row with Porian et al. included, and Section III.B a sentence. If the interval lies entirely below zero and its lower
  end is at or below −0.15, the exclusion is supported on this grid; otherwise (the interval contains both zero and
  −0.15) the comparison is undetermined. Either way the statement is limited to M ≤ 65 in P.

## 9. Additions (X3)

- **Approved and launched: X3(a).** Queue E (`code/sweep/queue_e.sh`; `run_grid.py` kind "edure", tag "mainre")
  retrains the FineWeb-Edu main-grid widths 384 (six layers; 25 to 800 million tokens) and 512 (eight layers; 25 to 400
  million) at the rule's learning rates (2.13×10^-3 and 1.644×10^-3), with the original seed and data order (seed 0,
  data seed 0), evaluates every endpoint during training on all five sets (FineWeb-Edu, FineWeb, WikiText-103, C4,
  PG-19), logs the unannealed trunk loss at every branch point except the last, and saves the endpoint weights. It was
  approved by the author on 25 September, launched at 11:34:53 (+03), concurrently with queues A to D, and committed in
  `5599198` at 11:35:07. Its first endpoint (width 384, 25 million tokens) carries all five sets.
- **Pre-specified analyses of the retrained cells** (all exploratory; none enters the registered Q1 to Q3 fits, which
  keep the original endpoints, so no cell is counted twice):
  1. Nondeterminism: max and median |Δ ln L| between each retrained endpoint and the original on the FineWeb-Edu and
     FineWeb validation sets, against the seed s.d. of Q4.
  2. Neutral-set matched cells at the rule's learning rate beyond width 128: pairs of retrained FineWeb-Edu and FineWeb
     main-grid endpoints at widths 384 and 512 and every budget (11 cells; FineWeb's neutral-set losses come from its
     saved weights, deviation D13), with Δ ln L on WikiText-103, C4 and PG-19 per cell and their median. They enter the
     label "quality" versus "distribution match" of item 4 (Q2.1) as matched-cell sign checks.
  3. A corpus-pair model with common exponents on each neutral set, over the cells where both corpora have that set
     (FineWeb widths 320 to 640 and FineWeb-Edu widths 384 and 512, main grid), with a CR2-scheme interval. With two
     FineWeb-Edu widths its tilt is weakly identified and is reported as exploratory only.
  4. The FineWeb-Edu arm of item 8 (annealing gap at widths 384 and 512).
- **Declined by the author (25 September): X3(b) and X3(c).** No learning-rate corners are run at FineWeb's (128,4)
  high-M shape, the corners are not extended to 3.2 billion tokens on either corpus, and no rescaled-weight-decay run
  is made at 3.2 billion tokens. Consequently the original FineWeb-Edu main-grid endpoints carry no neutral-set loss, and
  the high-M tuning checks exist for FineWeb-Edu only and stop at 1.6 billion tokens. High-M statements about tuning
  therefore rest on FineWeb-Edu alone, and the flexible-input bound is applied before any Q3 direction is stated if the
  rule's excess loss at a tested corner exceeds the seed s.d. or the optimal learning rate drifts with D at width 128.

## 10. Naming and planned wording

- "The D8 companion" is renamed "the CR2 scheme" in the paper, the memo and every generated table.
- Placeholder wording in the paper changes as follows. Each change alters planned wording, not a result, and each keeps
  the placeholder red:
  - introduction: "one sentence per Amendments 1 and 2: model-free σ\* by corpus, [x, y]; neutrality of the filter;
    extrapolation slope against its null" (the "exploratory drift" is dropped, item 3);
  - conclusion: "one sentence on what our experiment adds: σ\* by corpus and the FineWeb-Edu versus FineWeb neutrality
    verdict" (the drift is dropped);
  - abstract: a new placeholder after the σ\* sentence for the Q1 outcomes of item 4;
  - Section III.E, Results: "companion" becomes "CR2 scheme", Q1 is in FLOP-effective parameters (primary; FineWeb-Edu
    exploratory), and Q2 is decided under the robustness conditions of this amendment;
  - Section IV.D: the Q3 placeholder takes the wording of item 4 ("... under question Q3 of its pre-analysis plan and
    Amendments 1 and 2 ..."), and the Q2 placeholder reports the M\* factor as a small-scale magnitude beside
    DataDecide's, without a share identified at it;
  - Figure 3(d) note and Appendix B: "companion" becomes "CR2 scheme"; Appendix B gains a placeholder for the endpoints
    with post hoc neutral-set losses.
  - [integration, 25 Sep] Appendix C: the sentence on the measured seed standard deviation of ln L becomes neutral; it
    compares that value with the noise standard deviations of 0.005 to 0.015 simulated there and gains a placeholder
    "below, inside or above that range". Appendix E: the local-wedge check refers to Amendments 1 and 2 (it referred to
    Amendment 1), and the note of the breakdown-frontier figure gains a placeholder for the experiment's implied factor
    for M\* at 10^15 to 10^17 FLOP, marked as a small-scale magnitude beside DataDecide's (item 5; no share identified
    at it).

## Commitments

Every Q1 to Q6 result, every secondary analysis of item 7 and every exploratory analysis of items 8 and 9 is reported
whatever its outcome. FineWeb-Edu's arms of Q1 and Q3 are labeled exploratory wherever they appear. The abstract quotes
no experimental number that is not robust under the plan's scheme, the CR2 scheme and the seed-covariance bootstrap.
Every departure from the plan, from D8, from Amendment 1 or from this amendment is listed, with its time, in the
analysis memo (section 6) and in Online Appendix Table B-deviations.
