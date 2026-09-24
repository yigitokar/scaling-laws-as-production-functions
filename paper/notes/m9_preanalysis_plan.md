# Pre-analysis plan — our controlled two-corpus experiment (module m9)

Written 2026-09-24 03:15 (+03), committed to git before any estimation on the sweep results. At this time the
FineWeb-Edu main grid is partly complete and its raw endpoint losses have appeared in run logs; no model has been
fitted to them and no statistic below has been computed. The FineWeb grid, the referee-requested additions and the
high-M runs have not started. This plan supersedes paper/notes/m9_spec.md where they differ.

## 1. Design (fixed; see code/sweep/*.py)
- Two corpora (not "labs"): FineWeb-Edu ("edu") and FineWeb ("web"); shared 8,192-token byte-level BPE.
- Main grid: 8 widths d ∈ {128,…,640}, depth d/64; D ∈ {25,50,100,200,400,800}M (≤400M at d ∈ {448,512};
  ≤200M at d = 640); WSD, 1−sqrt cooldown over the last 20% branched from a constant-LR trunk; LR rule
  lr*(d) = 3.07e-3 (d/256)^−0.90 calibrated on edu at D = 50M.
- Additions (tags): `lrcal` (web LR calibration, 3 widths × 4 LRs, D = 50M); `lrcorner` (LR × {0.5, 2} at
  d = 128 over the full D ladder and at d = 640, D ∈ {25, 50}M, both corpora); `seedcorner` (seed 1 at d = 128,
  D ∈ {200,400,800}M, both corpora); `seeds` (seeds 1–2 at d ∈ {128,256,384}, D ∈ {50,100,200}M, both corpora);
  `hiM` ((d,L) ∈ {(128,4),(256,4)}, D ∈ {0.2,0.8,1.6,3.2}B on extension data, both corpora); `hiM2` ((128,2) to
  3.2B, edu). New runs are also evaluated on a neutral set (WikiText-103 validation+test, 655k tokens).

## 2. Primary choices (committed)
- **Output.** Primary: loss in bits per byte (nats/token ÷ (ln 2 × bytes/token of the evaluation set)) on the
  corpus's OWN validation set for technology estimates, and on BOTH validation sets for the between-corpus contrast
  (the contrast is reported separately on edu-val, web-val, and — for runs that have it — WikiText). A corpus
  effect is called "quality" only if it has the same sign on the other corpus's validation set (and on WikiText
  where available); otherwise it is "distribution match".
- **Parameter convention.** Report every result under BOTH conventions: (P) non-embedding N with compute counted
  as actual training FLOPs incl. the unembedding head and attention (field `C`); (T) total N incl. the tied
  embedding with C = 6 N_total D. Primary for the high-M extrapolation test: (P), because (T) cannot reach M above
  ~555 at this scale and because embedding parameters are not transformer-block capacity (Kaplan et al. 2020;
  Pearce & Song 2024). Primary for comparisons with public technologies that count total N: (T).
- **Sample.** Main-grid endpoints (tag main) of both corpora. Robustness: drop D = 25M (fixed 250-step warmup is
  16% of those runs); drop the 2- and 3-layer widths (d ∈ {128,192}) — the "four-layer floor".
- **Estimators.** (i) Model-free: σ* from the curvature of isocost profiles and the frontier slope,
  1/σ* − 1 = L_{ln N, ln N}|_C / (2 |dL*/d ln C|), and local w = (L_c − L_x)/(L_c + L_x) from local polynomial
  (quadratic in (ln N, ln D) with Gaussian kernel weights) fits on the grid; (ii) parametric: Huber-LSE
  (δ = 1e-3) Chinchilla form and its κ-free generalization; Gaussian NLS as a robustness check.
- **Inference.** Wild cluster bootstrap by width (Webb six-point weights, 999 draws) on the residuals of each fit
  (clusters = trunks). Secondary: parametric residual bootstrap with the within-trunk covariance estimated from
  the seed replicates. Report both; conclusions must hold under both to be called robust.

## 3. Pre-specified questions, statistics and decision rules
Q1 (curvature). σ*_κ and model-free σ* per corpus with 95% CIs. Test H0: σ*_edu = σ*_web (wild bootstrap).
   Report whether σ* < 1 (implied by interior optima; the empirical content is U-shaped isocost profiles).
Q2 (neutrality of the quality filter). With a common output (each validation set separately), fit the corpus-pair
   model with common exponents and corpus-specific (E, A, B). Statistic: tilt χ̂ = Δln(A/B) between corpora
   (edu − web) and the implied factor ratios for M* and ŵ. Tests: (a) exponents equal (Wald); (b) tilt = 0.
   Decision rule: "factor-biased" if the tilt's 95% wild-bootstrap CI excludes 0 on BOTH validation sets with the
   same sign; "neutral" if it includes 0 on both and the CI half-width is below 0.10; otherwise "inconclusive".
   Lemma 1 prediction if edu is data-augmenting: lower M* for edu.
Q3 (extrapolation of the wedge). Fit the parametric technology on the Chinchilla-like support M ≤ 100 (convention P
   and T separately) and predict ε_D, ε_N and w at every grid and hiM endpoint with M > 100. Compare with (a) the
   full-support fit and (b) the model-free local w from the local polynomial on the full grid. Statistic:
   ln(w_restricted / w_local) as a function of ln M (report the slope and its CI). The sign answers whether
   Chinchilla-form extrapolation over- or under-states the wedge at high M in this regime. We commit to reporting
   the result whatever its sign; we state ex ante that levels need not transfer to production scale (10^22–10^25
   FLOP), while the sign of the extrapolation error is the object of interest.
Q4 (noise). Seed SD of log loss at each replicated cell; compare with the residual SD of the fitted technology;
   use it to calibrate the power calculation.
Q5 (flexible inputs). From lrcal (web) and lrcorner (both): whether the optimal LR differs by corpus (compare the
   interpolated argmins at each width) and whether it drifts with D at the high-M corner; report the excess loss of
   the rule relative to the best LR at each tested cell. If the rule's excess loss exceeds the seed SD at a corner,
   bound the implied bias in a with the flexible-input formula (Online Appendix, eq. for a_obs − a).
Q6 (functional dependence with real losses). On-path subsample (endpoints nearest the fitted expansion path at each
   compute level) vs full grid: Jacobian condition numbers and the κ-free profile over σ* (on a grid over (0.05,
   0.99), with the LR measured relative to the unconstrained optimum).

## 4. Power (computed BEFORE touching the results)
Using the Section II Monte Carlo machinery with this design (widths, D ladder, trunk structure as a common
within-width error component) and log-loss noise sd ∈ {0.002, 0.005, 0.01}, report the expected s.e. of σ*_κ,
of the tilt χ̂, and of the Q3 slope, and the power to detect a tilt of 0.22 (DataDecide's magnitude).

## 5. Reporting commitments
All of Q1–Q6 are reported in the paper or Online Appendix regardless of outcome. Deviations from this plan are
listed explicitly in the m9 memo with reasons.
