# Amendment 1 to the pre-analysis plan (m9) — outcome-to-headline map

Written 2026-09-24 17:40 (+03); committed in a5f47f8 at 17:44. Referee R3 (round 2, N8(d)) asked for a dated statement of how the paper's headline
claims will change under each outcome of the experiment, written before the between-corpus results exist.

**What had been seen when this was written.** (i) The power and coverage study (m9 memo §0; computed from the design
alone, before estimation). (ii) Preliminary FineWeb-Edu-only estimates from the pipeline dry run (m9 memo §1, P1–P5;
estimation started 2026-09-24 ~17:05 per data/processed/m9_sweeps/estimation_start.txt): model-free σ* ≈ 0.666 (P),
0.607 (T); κ-free agrees, κ = 1 higher; the Q3 slope on FineWeb-Edu alone is not decided. (iii) No FineWeb (web)
estimate of any kind: its main grid is about one-third complete, and no between-corpus statistic (Q1 equality test,
Q2 tilt) has been computed. The high-M runs, seeds and LR checks are in progress.

**Comparison convention for σ\*.** Comparisons with the public IsoFLOP designs (Section III) use FLOP-equivalent
parameters N_F = C/(6D) with C the actual training FLOPs (Referee 1, round 2, New 2(d)); conventions P
(non-embedding) and T (total) are also reported. The power study shows conventions differ by 0.04–0.08 by
construction; such differences are not findings.

## Q1 — σ\* by corpus (primary statistic: model-free σ*, FLOP-equivalent convention; inference: plan + D8 companion)
1. **Both corpora in [0.60, 0.76] and equality not rejected.** Abstract and introduction: keep "about 0.7" for
   the public designs (conditioned on 10^19–10^21 FLOP, per round-2 N1) and add one clause: "a controlled
   experiment at 10^15–10^17 FLOP gives [x, y] for two corpora." Section III.E reports both conventions.
2. **Both in [0.60, 0.76] but equality rejected (both inference schemes).** Same as 1, plus a sentence in III.C that
   curvature differs with data quality by Δ, and the wedge section uses corpus-specific curvature as sensitivity.
3. **Either corpus outside [0.60, 0.76] (both inference schemes).** The abstract replaces "about 0.7" by the range
   spanned by the public designs and the experiment ("between 0.6 and 0.75 across designs and compute levels"); the
   introduction states that σ* is design- and scale-dependent; Section IV's magnitude bounds widen accordingly.
4. **Undetermined (intervals too wide under the companion scheme).** The experiment is reported in III.E and
   Appendix B only; headline unchanged from the public-design statement.

## Q2 — neutrality of the quality filter (tilt χ between corpora)
1. **"Factor-biased" (CIs exclude 0 on both in-distribution validation sets, same sign, under both schemes).**
   III.C: the quality filter is factor-biased with χ̂; the implied M* ratio and wedge factor e^{−χ̂} are reported;
   Section IV's discussion of recipe tilt uses χ̂ alongside DataDecide's. If the sign also holds on the neutral sets
   (WikiText, C4, PG-19) where available, call it "quality"; otherwise "distribution match or quality".
2. **"Neutral" (CIs include 0 on both; half-width < 0.10).** III.C: the filter shifts the asymptote/scale but not the
   optimal mix in this design; recipe tilt remains bounded by DataDecide's.
3. **Inconclusive.** Reported as such; no headline change.

## Q3 — extrapolation of the wedge (slope vs the design-specific null; convention P primary)
1. **Slope below the null (companion interval excludes the null and lies below it).** Introduction and IV.D:
   "parametric extrapolation understates the wedge at high M in two recipes (Farseer and our experiment)".
2. **Undetermined.** Introduction and IV.D keep "in one recipe (Farseer)"; III.E reports the level gap
   (mean ln(w_{M≤100}/w_local)) as descriptive only.
3. **Slope above the null.** The "conservative extrapolation" statement is dropped from the abstract/introduction;
   IV.D reports that the direction of extrapolation error is recipe-dependent and widens the magnitude bounds.

## Commitments
Every Q1–Q6 result is reported regardless of outcome (plan §5). The abstract never quotes an experiment number that is
not robust under both the plan's inference and the D8 companion. This amendment and the power/coverage outputs are
committed to git together; the paper cites the commit hash.
