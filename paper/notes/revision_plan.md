# Revision plan after referee round 1 (lead author, 2026-09-24)

Reports: paper/referee/R1_io_econometrician.md (major revision), R2_ml_scaling.md (major revision),
R3_aer_editor.md (reject & resubmit; refocus), R4_auditor.md (major revision; ~395/420 numbers pass).
All four agree on the core diagnosis. Decisions below are binding for all revision modules and writers.

## A. Refocus: one paper, one frame
Title: **What Optimizing Labs Reveal: Scaling Laws as Production Functions**.
Frame (R3 M2): optimization is double-edged for measurement. It removes the (transverse) variation in outcomes that
identifies the technology's curvature and optimal mix; it makes choices informative about objectives. Designed
experiments break the circle: they supply the technology (curvature σ*, zero point M*), and labs' choices then reveal
the value they place on compact models (for developers who serve, planned inference expenditure).
Main text ≈ 14,000 words, ≤ 5 formal results, ≈ 7–8 exhibits:
- Intro (≤1,800): stakes; the double-edged idea; three results with ≤15 numbers; what is robust vs not; 5–6 closest
  papers (Hao & Merrill 2026; Kricheli et al. 2026; Mertens et al. 2026; Whitfill 2025; Sardana et al. 2024/Bian et
  al. 2025/Roberts et al. 2026; Bond & Söderbom 2005 / ACF 2015 / GNR 2020 / DLW 2012 + Raval 2023); roadmap.
- I. The Training Problem: technology (quasi-homothetic Chinchilla family; κ family), FLOP-accounting approximation
  (not an "identity"), duality (short; rename our "Approach 1" = IsoFLOP-minima frontier; Hoffmann's Approach 1
  also estimates factor demands), Lemma 1 (σ < 1 at interior optima; empirical content = U-shaped IsoFLOP profiles),
  Proposition 2 generalized wedge: under any objective V(L, N) − training cost, w − 1 = marginal value of
  compactness at fixed quality ÷ training cost; special cases (developer-borne serving with price p and
  internalization λ; tier/memory/latency; data cost; distillation teacher cost; MFU varying with N; family-level D).
  Report the expenditure share s = (w − 1)/w.
- II. What Optimizing Labs' Data Identify: merged Prop 3 (on-path identification + information rates, positioned vs
  Marschak–Andrews, Bond–Söderbom, GNR, optimal design; global vs local identification, Rotnitzky et al.);
  Prop 4 model-free σ* and w (NEW; R1 comment 7); scope: where the theory binds (compute-optimal ladders; designs
  like DataDecide) vs released models (off path: heterogeneous wedges = the variation that helps). Monte Carlo
  (2 panels; truth removed from start set).
- III. The Technology from Designed Variation: data (brief table); model-free σ* on all IsoFLOP designs; κ-free σ;
  heterogeneity test + random-effects summary; M* and a not portable; our controlled experiment (m9); one
  paragraph on conventions/flexible inputs (Porian decomposition → appendix).
- IV. What Over-Training Reveals: re-specified inversion (clean inference-demand sample; common-D families treated
  separately; lab-own technologies primary; κ-free reference; all technologies under an ex-ante rule; conventions;
  expenditure share; partial identification of M*(C) at frontier scale; in-support nonparametric w vs
  extrapolation; conduct tests (open vs closed, developer serving footprint vs on-device focus, multi-tier
  bunching); validation of levels where possible (OpenRouter token volumes; HF derivative counts; aggregates vs
  Patterson et al. 2022 / Wu et al. 2022)).
- V. Economic Implications (≈1,000 words, 1 exhibit; R3 M7): data-demand growth D* ∝ C^{1−a}; the data wall
  (shadow value of unique tokens with repetition, σ* matters?); inference share of AI compute.
- VI. Conclusion (≤700 words).
- Online Appendix: A proofs; B data; C Monte Carlo; D additional technology results (Chinchilla duality/selection,
  measurement & flexible inputs = old IV.D, sweeps robustness); E Observational production functions and
  algorithmic progress (old Section VI with fixes); F revealed-demand robustness.

## B. Front matter
- Byline: the user explicitly requested "Yigit Okar (Care AI) and Claude (Anthropic)". Keep it, with a disclosure
  footnote describing Claude's role; flag the AEA/COPE authorship policy to the user (not our decision).
- Running heads: AER typography (finalmode) but NO journal name/volume/issue placeholders — use a working-paper
  running head (e.g., "OKAR AND CLAUDE: WHAT OPTIMIZING LABS REVEAL" / "WORKING PAPER, SEPTEMBER 2026").
- Abstract ≤ 100 words; state only robust claims; cardinal numbers only with their conditioning technology.
- Replication statement: repository path (local) or "available from the authors"; no "[LOCATION TBD]".
- Bibliography: every arXiv/proceedings entry must render a venue (journal = {arXiv preprint arXiv:XXXX.XXXXX} /
  {Advances in Neural Information Processing Systems 35}, etc.). Fix the 37 entries listed in R4 M9(a).

## C. Things to fix everywhere (from R4 + R1 minors)
Chinchilla design: 245 runs, 137 on nine IsoFLOP profiles, 108 off-profile, FLOP 1.4e18–1.3e22. "Within 10^-3" (not
4e-4). "Close to" (not "exactly") 2E[ln w]/(α+β). Warm-start coverage 57% vs 92% like for like. 0.78–1.16 for the
Chinchilla-70B outlier rules. Nerlove = scale-dependent returns, not spurious. Gundlach et al. define the CEG; our
contribution is the contrast with realized gains. Bjorck exponent range. DeepSeek LLM (Bi et al. 2024) cited for data
quality → allocation. Notation: irreducible loss L∞ (not E, which clashes with expectation) — or keep E but use
𝔼 for expectation; allocation exponent a vs inner exponents: use a₁, b₁ only in appendix; T (tokens) vs T_C
(rename doubling time τ_C); "Sample B" → "verified sample". Consistent sweep counting: "seven sweep–corpus
technologies from five public studies" (DataDecide separately). One interval per sentence in the main text.
