# Referee Report 4 (Replication Auditor)

**Manuscript:** "Scaling Laws as Production Functions" (Okar and Claude), AER submission draft compiled 2026-09-24 (`paper/main.pdf`, 133 pp.; main text pp. 1–60, references pp. 60–70, Online Appendix pp. 71–133).

**Mandate.** I was asked to audit, mechanically: (a) every number in the main text and main-text tables against the module outputs (`output/tables/*.csv`) and memos/reviews (`output/memos/*.md`); (b) the most important citations; (c) internal consistency (a quantity quoted differently in two places, notation clashes, figure and table references); and (d) whether any claim on the "Do not claim" list of `paper/notes/paper_plan.md` §3 appears. The authors' controlled experiment (the "TBD-m9" placeholders) is still running, so I comment on its design, not on missing results.

**Method.** I read every main-text section in `paper/sections/*.tex` and every main-text table in `paper/tables/table*.tex`, and located each item in the compiled PDF (text extracted with PyMuPDF). I then traced numbers to the CSVs. Where a number is a summary over model-level data, I recomputed it: for example the Sample B statistics from `m3_wedge_models.csv`, the IQR-based standard errors from the stored bootstrap draws in `data/processed/m1_chinchilla/*.npy`, and the profile-likelihood sets from `m1_chinchilla_profile_sigma.csv`. I also recomputed the headline wedge under the κ-free Chinchilla technology. Cross-references were checked against the LaTeX log: there are no undefined references or citations. Citations were checked against `paper/references.bib`, the rendered `_build/main.bbl`, `lit/notes/*.md` and, where needed, the arXiv abstract or HTML full text.

Page numbers are PDF pages. "PASS" means the value matches its source after sensible rounding. "FAIL" means the value, or the verbal claim built on it, is contradicted by the source. "UNTRACED" means I could not find the number in any CSV or memo.

---

## 1. Summary of the paper

The paper argues that neural scaling laws are production functions and that fitting them is production-function estimation. It maps Chinchilla's Approaches 1–3 to cost-function, conditional-factor-demand and primal estimation (Prop. 1). It shows that interior compute optima under C = 6ND require 0 < σ < 1 (Lemma 1). It derives a revealed-inference-demand wedge w = ε_N/ε_D = 1 + T/(3D) (Prop. 2) and a set of identification results: functional dependence on the expansion path, information that is second order for M* and fourth order for σ*, the sign of Hicks bias identified by the drift of M*, signed transmission bias, and partial identification of T (Props. 3–7). A Monte Carlo quantifies these results.

Empirically, the paper re-estimates the Chinchilla technology with an estimator horse race, duality tests and κ tests; estimates σ* across public sweeps (0.73–0.83 under κ = 1, 0.51–0.71 with κ free, and 0.69–0.71 on Farseer); tests data-quality neutrality (Gadre, DataDecide); and decomposes the Kaplan–Chinchilla gap into measurement and flexible-input components (Porian, Step Law). It then inverts the first-order condition for 173 open-weight models (median w = 3.19) and validates the result against downloads. Finally, it benchmarks observational estimators against a design-matched experimental technology in a LaLonde-style exercise and re-analyzes Ho et al. (2024) and allocative efficiency.

## 2. Overall assessment

As a replication object, the paper is unusually good. Of the roughly 420 distinct numerical statements I checked in the main text and Tables 1–9 (ledger in Appendix A), about 395 match their sources exactly or after rounding. I recomputed the Sample B headline statistics from the model-level file and matched every one to the third decimal (median ŵ = 3.186; 94.8% > 1; 136/173 with the band above 1; and so on). All 17 rows of Table 6, all cells of Tables 7, 8 and 9, and the Huber, Gaussian, levels and system columns of Table 3 (including the IQR-based standard errors of M*) reproduce from the CSVs or the stored bootstrap draws. The citations I checked say what the text attributes to them, with one clear mischaracterization (Nerlove, below) and a few framing issues.

The audit nevertheless finds problems that must be fixed before the paper can be evaluated:

1. **The Chinchilla data are misdescribed.** The runs are said to lie on nine IsoFLOP budgets from 6×10^18 to 3×10^21 FLOP. In fact the data span 1.4×10^18 to 1.3×10^22 FLOP, and 108 of the 245 runs are not on IsoFLOP profiles. This matters for the budget-clustered inference that the paper recommends, and it contradicts other sections of the same paper.
2. **The headline wedge uses a specification the paper rejects, and an inference scheme the paper advises against.** The headline is computed under κ = 1 with a pairs bootstrap. The paper's own preferred κ-free Chinchilla technology raises the median wedge from 3.19 to 3.78 (my recomputation), i.e. inference/training compute of 2.8 rather than 2.2.
3. **The "σ ≈ 0.7" headline rests on a single sweep.** Ranges for σ*, a and γ are also quoted over different sets of sweeps in different paragraphs.
4. **Several verbal claims fail against the CSVs.** For example, "between 0.95 and 1.06 under every rule that removes them", "within 4×10^-4", "exactly 2E[ln w]/(α+β)" and "nine technologies estimated in Section IV".
5. **One profile-likelihood set excludes the reported point estimate**, and another profile has its minimum at the grid boundary.
6. **Notation collides repeatedly.** θ, φ, τ, η, s and d each carry two to four meanings.
7. **37 of 113 references are typeset with no venue or arXiv identifier.**
8. **The controlled experiment (m9) has design features that will confound its three stated purposes** unless they are addressed before the results are written up: the non-embedding convention with embedding shares up to 73%, a learning rate tuned on one lab at one D, a fixed warmup, and trunk-shared endpoints.

**Recommendation: major revision.** None of these problems is fatal. Several are one-line fixes. The two substantive ones (the reference technology for the wedge, and the Chinchilla design description with its clustering consequences) require re-running existing code, not new data.

---

## 3. Major comments

**M1. The Chinchilla extraction is not "245 runs on nine IsoFLOP budgets from 6×10^18 to 3×10^21 FLOP" (§IV.A, p. 28). This affects the clustered inference and contradicts §II.E and §V.A.**

- **Source data.** `data/raw/epoch_chinchilla/svg_extracted_data.csv` has training FLOP from 1.40×10^18 to 1.30×10^22. By the m1 memo's own classification (`m1_chinchilla.md`, "IsoFLOP budget reconstruction"), only 137 of 245 runs (132 of 240 in the baseline sample) lie within ±0.045 dex of a nominal IsoFLOP budget. The other 108 are off-profile runs, presumably the Approach-1 training-horizon runs that Hoffmann et al. pooled into Approach 3. `m1_chinchilla_design_variance.csv` confirms the split: 132 IsoFLOP-profile runs with a 37% transverse variance share, and 108 off-profile runs with 21%.
- **The paper contradicts itself.** §II.E (p. 23) says the design reaches "compute up to 1.3×10^22 FLOP". §V.A (p. 41) says "Chinchilla's 1.3×10^22 FLOP" is the largest budget of any design. §IV.A (p. 28) and Figure 2's notes (p. 19) say the budgets run from 6×10^18 to 3×10^21.
- **Why it matters beyond wording.**
  - (i) Every "nine-cluster" result assigns each off-profile run to the nearest nominal budget after an offset correction (memo lines 324–327). These include the budget-bootstrap SEs in Table 3, the [3.2, 126] interval for M*(5.76×10^23) repeated in the Conclusion, the [0.54, 2.01] wedge interval for Chinchilla-70B, the κ̂ cluster p = 0.042, and the "80 percent" cluster share in §V.B. A run from a 10^22 training-horizon trunk is not a member of the 3×10^21 IsoFLOP experiment. Off-profile runs that share a model-size trunk are the natural cluster.
  - (ii) The Conclusion's "Third" recommendation (cluster by IsoFLOP budget) is thus drawn from data in which 45% of runs have no IsoFLOP budget.
- **Requested.**
  - (a) Describe the data correctly in §III.A, §IV.A and the Figure 2 notes: 245 runs, 137 on nine IsoFLOP profiles, 108 off-profile, FLOP 1.4×10^18–1.3×10^22.
  - (b) Re-run the cluster bootstrap with two alternatives: off-profile runs clustered by model size (their trunk), and off-profile runs as singleton clusters. Report how Table 3's bracketed SEs, the M* interval, the Chinchilla-70B wedge interval and the κ cluster p-value change.
  - (c) State that the duality tests use only the 132 profile runs, if that is the case (the "nine argmin gaps" suggests so).

**M2. The headline revealed-demand number uses a technology the paper rejects (κ = 1) and a bootstrap the paper recommends against (pairs). Under the paper's own preferred specification the headline moves by about 20 percent.**

- **What the headline rests on.** The reference technology in §V (p. 41) is the κ = 1 Huber refit, with 400 pairs-bootstrap draws. Yet §IV.A rejects κ = 1 on these same data (p < 0.001), §IV.B rejects it in all seven sweeps, and the Conclusion tells scaling studies to cluster by budget and to test κ instead of imposing it.
- **My recomputation.** I used the κ-free Huber estimates in `m1_chinchilla_spec_kappa.csv` (ln A = 7.742, ln B = 9.214, a₁ = 0.4243, b₁ = 0.4309) and the inner-aggregator wedge that §V.A says is valid for any κ ("matters only through the inner exponents"). For the 173 Sample B models this gives:
  - median ŵ = 3.78, against 3.19 under the reference;
  - median ŵ − 1 = 2.78, against 2.19;
  - share with ŵ > 1 = 93.1%;
  - Llama 3 8B ŵ = 6.75, against 5.27.
  My recomputation of the reference itself reproduces the CSV exactly (median 3.186), so the code path is not in question.
- **Consequences for the abstract.** The abstract ("about 2.2 times") and the Conclusion ("a factor of about three") therefore quote the lower of two Chinchilla-based numbers, and the lower one is the specification the paper rejects. The direction is "conservative", which the authors may prefer. But the choice must be stated and justified, and the κ-free Chinchilla technology should enter the technology band. At present only the Farseer κ-free variant appears (median 3.85, p. 43).
- **Clustered intervals.** The same logic applies to intervals. With the budget-clustered bootstrap, the share of models whose lower bound exceeds 1 is 80%, not 87% (`m3_wedge_sampleB_by_tech.csv`). The paper mentions this in a parenthesis (p. 42) but uses the pairs bootstrap for Table 6 and every model-level interval. Either justify pairs resampling for this purpose, or make the clustered intervals primary, in line with the Conclusion's recommendation.

**M3. "σ ≈ 0.7" is a one-sweep result presented as a cross-sweep result, and ranges are quoted over different sets of sweeps in different places.**

- **(a) The headline.** The abstract (p. 1) says "In seven public training sweeps ... an elasticity of substitution near 0.7 once the data choose the curvature". But with κ free, σ*_κ is 0.511, 0.544, 0.594, 0.607 and 0.623 in five of the seven sweeps (Muennighoff, OLMo, and the three Gadre corpora; Table 4). It is about 0.70 only on Chinchilla and Farseer. The three "agreeing" estimators (0.69–0.71) are all fitted to the same 404 Farseer runs.
  - The honest summary is: 0.51–0.71 with κ free; about 0.70 on the two largest designs; and 0.69–0.71 across three functional forms on Farseer. The Conclusion's "0.5 to 0.8, central value near 0.7" is defensible. The abstract's sentence is not, as written.
- **(b) Inconsistent sample definitions.** "Seven public sweeps" counts the three Gadre corpora separately. DataDecide is an eighth row in Table 4 but is sometimes included and sometimes not:

  | Location | Quantity | Range quoted | Includes DataDecide? |
  |---|---|---|---|
  | Intro, p. 3 | σ* (κ = 1) and its SE | 0.73–0.83; SE 0.006–0.034 | No |
  | §IV.B ¶2, p. 32 | σ* and its SE | 0.735 (Muennighoff) to 0.828 (DataDecide); SE 0.005–0.034 | Yes |
  | §IV.B ¶"What differs", p. 35 | σ* | 0.735–0.825 | No |
  | Intro p. 3; §IV.B p. 35 | a | 0.37 to 0.57 | No (DataDecide a = 0.315) |
  | §IV.B p. 35; Conclusion p. 59 | γ | 0.10–0.18 | No (DataDecide γ = 0.090) |
  | Data §III.A, p. 24 | datasets | "five further sweeps", listing five datasets including DataDecide | — |

  Adopt one definition (for example, "seven sweep–corpus technologies from five public sweeps, plus DataDecide for recipe comparisons") and use it everywhere, including the abstract and the Intro footnote.
- **(c) The "factor of two" contrast is misleading.** On p. 35 the text says "σ* (0.735–0.825) and the frontier elasticity γ (0.10–0.18) vary by less than a factor of two", as a contrast with a and M*. But a (0.37–0.57, a ratio of 1.53) also varies by less than a factor of two, and γ's ratio (1.82) exceeds a's. Only M* varies by more than 2×. See also D4 in Appendix D.

**M4. Verbal claims that fail against the module outputs (each needs a one-line fix).**

- **(a) Chinchilla-70B wedge under outlier rules (p. 31).** The text says it is "1.54 with the outliers and between 0.95 and 1.06 under every rule that removes them". `m1_chinchilla_selection_rules.csv` shows two other rules that remove all five outliers:
  - dropping the 10^19 IsoFLOP budget gives w = 0.78;
  - IsoFLOP runs only, with the five highest-loss runs dropped, gives w = 1.16.
  The m1 memo (line 112) itself flags both. Replace "every rule" with the actual set of rules, or report 0.78–1.16.
- **(b) Transmission Monte Carlo (§II.D, p. 22).** The text says "Simulated slopes match the closed form to within 4×10^-4". In `m6_montecarlo_transmission_check.csv`, max |diff| is 4.24×10^-4 on the path and 8.49×10^-4 in the lifetime-optimal-plus-error cells. The m7 check (`m7_theory_transmission_mc.csv`) gives 5.8×10^-4 for the funding rule. Write "within 10^-3".
- **(c) FOC-system bias (§VI.A, p. 54).** The text says "overstates ln M* by 1.25–1.28, exactly 2E[ln w]/(α+β)". The closed form is 1.28 (m6 memo line 144), which matches only the target rule; the other three rules give 1.25 (and the memo's own range is 1.24–1.28). Write "close to the closed form 2E[ln w]/(α+β) = 1.28".
- **(d) "Nine technologies" and "12 technologies" said to be estimated in §IV.**
  - §VI.C (p. 58) says "across nine technologies estimated in Section IV the gain ranges from 0.55 to 4.3, and Meta's Llama 3 IsoFLOP runs give 0.74". Table 9's note (p. 56) says "the 12 technologies estimated in Section IV (two on the Chinchilla sweep)".
  - `m5_progress_table7_panelB.csv` shows that the nine are Llama 3, Marin ×3, (Mis)Fitting FineWeb/C4, and m2's Farseer, Gadre C4, OLMo and Muennighoff. Llama 3 is therefore one of the nine, not an addition.
  - Marin's three technologies and the (Mis)Fitting technology are never estimated or reported in §IV. (Mis)Fitting does not appear in Table 2 at all.
  - Either report these technologies (for example in Online Appendix Table D-x, with a Table 2 row for (Mis)Fitting) or say "nine technologies from the m1/m2 registries (Online Appendix …)".
- **(e) Warm-start versus multistart bootstrap coverage (§II.B, p. 18).** The text contrasts "covers the true σ* in 67 percent" (R = 500) with "restores coverage (92 percent in a check with 60 replications)". In that same 60-replication check, warm-start coverage of σ* was 57% (`m6_montecarlo_designA_bootcheck.csv`). Report 57% → 92% like for like, or give both samples.
- **(f) Parameter-augmenting share φ (§VI.B, p. 55).** The text says "φ … can lie anywhere in [−0.39, 3.37], over which T_C ranges from 5 to 38 months". The 5.4–38 range is over the grid-point interval [−0.25, 3.25] (m5 memo line 59), not over the interpolated interval.

**M5. Profile-likelihood sets: one excludes the reported point estimate, and one has its minimum at the grid boundary.**

- **(a) Full design (p. 32).** On the full Chinchilla design the text reports the 95% profile set for σ* as [0.67, 0.69]. Two paragraphs earlier and in Table 3 it reports the κ-free estimate as σ*_κ = 0.700 (0.015); Table 4 gives 0.701. The set excludes the point estimate. From `m1_chinchilla_profile_sigma.csv`, the profile is computed only for the Gaussian likelihood (the Huber column is all NaN), whose κ-free σ* is 0.680 (`m1_chinchilla_spec_kappa.csv`, `sigma_star_gauss`). State in the text and in the Panel D notes that the profile is Gaussian, or compute the Huber/LAD profile, so that point and set come from one objective.
- **(b) On-path band (p. 32; Table 3 Panel D).** In the on-path band (n = 41), the "flat" profile rises monotonically from LR = 0 at σ* = 0.50 (the lower edge of the grid) to 1.96 at σ* = 0.95. The minimum is therefore at the boundary: the MLE lies at or below 0.50, and the LRs are measured relative to a constrained minimum. The statement "every value lies in the 95 percent confidence set" is conditional on the grid. The MC has the same feature: `share_sig_hat_off_grid` = 0.5 for the on-path design in `m6_montecarlo_designA_profile.csv`. Extend the grid over (0, 1), or report the LR relative to the unconstrained optimum. The qualitative conclusion (no curvature information on the path) will almost surely survive; the reported numbers should be correct.

**M6. Specification tests are reported selectively.**

- **(a) CES.** The text says CES "is not rejected" (p. 31) and "not rejected by a Wald test in any sweep (p between 0.22 and 0.90)" (p. 35). These statements are true for the Huber-based Wald tests. But `m2_spec_tests.csv` reports a Gaussian LR test of α = β that rejects on Chinchilla (LR = 10.4, p = 0.0013) and is borderline on Farseer (p = 0.052). Since §IV.A argues that the choice of estimator matters (Huber ≈ LAD versus Gaussian), both tests belong in Table 3 Panel B and in the text.
- **(b) Rank-one curvature.** With the Chinchilla-form Ê, rank-one curvature is rejected on Chinchilla (p = 0.007) and Farseer (p = 0.003). It is not rejected only with Ê from the κ family. The text notes the sensitivity (p. 31), but the sweep-level summary on p. 35 ("not rejected at 5 percent in six of seven sweeps") should also give the κ = 1-Ê count, which is five of seven.
- **(c) Bootstrap p-values at their floor.** Several bootstrap p-values are reported at the minimum their number of draws allows, and should be labeled as such:
  - DataDecide Hicks neutrality, "p ≤ 0.01", uses B = 99.
  - The DataDecide pooled, Hicks, data-augmenting and parameter-augmenting tests use B = 19, so p = 0.05 is the smallest attainable value. "Every restriction is rejected" (p. 35) rests on p = 0.05 for four of them.
  - Farseer rank-one "p ≤ 0.003" uses B = 300, so 1/301 is the floor.
  Increase B, or say "at the resolution of B draws".

**M7. Two different "reliabilities" of ln C are used without reconciliation (§VI.A).**

- The text dismisses measurement error in compute because "the implied reliability of ln C is at least 0.988" (p. 53; m4 memo: 0.995 pooled, 0.988 within families).
- Table 8 Panel B nonetheless reports EIV-corrected estimators using reliabilities of 0.895 (pooled) and 0.514 (within family), implied by Epoch's "Confident" class (`m4_observational_table6_lalonde.csv`, `reliability` column).
- The within-family EIV-corrected estimate has the largest and most significant bias in the table: θ_C = 0.858, bias +0.448 (0.128). It is never discussed. The reverse-regression bias (+0.209, SE 0.126) is not discussed either.
- One of the two reliability figures is wrong for this purpose. If 0.988 is right, the EIV rows should use it (they would then barely move). If the class-based figures are right, the text's dismissal of measurement error does not hold within families.

**M8. Characterization of cited work: one misattribution and three framing issues.** Full citation audit in Appendix B.

- **(a) Nerlove (1963).**
  - *Misattribution.* The Intro (p. 3) attributes to Nerlove "the spurious small-firm scale economies". Nerlove estimated genuine (not spurious) returns to scale in electricity generation that decline with firm size. His lesson, which §IV.D states correctly on p. 38, is that a constant-elasticity form fitted over a size range misrepresents a varying local elasticity.
  - *Fix.* "Spurious" scale economies from scale-correlated input error is the paper's own argument (via Collard-Wexler and De Loecker). Rephrase to "in the sense of Nerlove's (1963) scale-dependent returns".
- **(b) Gundlach et al. (2025).**
  - *Framing.* The paper says that "the tenfold gain … is the counterfactual cost of keeping Kaplan's rule at 2025 frontier compute" (pp. 4, 58), as if correcting Gundlach et al. But Gundlach et al. define the compute-equivalent gain (CEG) at a reference compute level and report "about 10× at the frontier" for Kaplan→Chinchilla rebalancing (`lit/notes/ml_observational.md` §2.9). The two are describing the same counterfactual.
  - *Fix.* The paper's actual contribution is the contrast with realized between-era gains (1.0–2.4×). Say so, and credit Gundlach et al.'s definition.
- **(c) Bjorck et al. (2025).** The −0.32 learning-rate elasticity holds batch size fixed (confirmed) and applies to models of at least 760M parameters. For smaller models Bjorck et al. report 0.40–0.70 (arXiv 2409.19913, Sec. 3). Step Law's cells run from 215M to 1.07B non-embedding parameters, so the relevant comparison value, and hence the "26–43 percent" share of the gap closed (p. 39), depends on which exponent is used. Report the range.
- **(d) Missing citation: DeepSeek LLM (Bi et al. 2024).** This paper reports that data quality shifts the allocation exponent (higher quality moves compute toward model size). It is in the authors' own notes (`lit/notes/ml_core.md` line 137) and in the bib (`bi2024deepseek`). It is the closest precedent for §IV.C and should be cited there.

**M9. Front matter and bibliography are not compliant.**

- **(a) Bibliography.** 37 of the 113 rendered references have no venue, journal or arXiv identifier, because `aea.bst` drops `eprint`/`howpublished`/`booktitle` for these entry types. Examples:
  - Porian et al. (2024), Gadre et al. (2025), Hao and Merrill (2026), Mertens et al. (2026), Whitfill (2025), Kricheli et al. (2026), König et al. (2026), Cottier et al. (2024) and Erdil et al. (2025): title only.
  - Hoffmann et al. (2022) ("Vol. 35, 30016–30030") and Ho et al. (2024) ("Vol. 37, 58245–58283"): volume and pages with no proceedings name.
  - Sennrich et al. (2016): pages only.

  Full list: `bergemann2025economics bhagia2024establishing biderman2023pythia bjorck2024scaling brown2020language choshen2024hitchhikers cottier2024rising czech2026problems demirer2020production dey2023cerebras erdil2025gate erdil2022algorithmic gadre2024language gundlach2025origin hagele2024scaling hao2026theory ho2024algorithmic hoffmann2022training hu2024minicpm konig2026validity kricheli2026tokens li2025misfitting loshchilov2019decoupled maiapolo2024sloth mertens2026secret muennighoff2023scaling owen2024predictable penedo2024fineweb porian2024resolving ruan2024observational sennrich2016neural snell2024scaling teamolmo2024olmo touvron2023llama whitfill2025note zellers2019hellaswag zhang2019root`. Use `journal = {arXiv preprint arXiv:xxxx}` or `note`, and `booktitle` → `journal` for proceedings, as AEA style expects.
- **(b) Abstract length.** The abstract has 117 words; AEA journals limit abstracts to 100.
- **(c) Replication statement.** The package location is "[LOCATION TBD]" (p. 1, footnote). AEA's Data Editor will require it before acceptance.
- **(d) Authorship.** The author line lists an AI system as coauthor. The editorial office should confirm this against the AEA's current policy on generative AI. Most publishers following COPE guidance do not accept AI systems as authors and ask for disclosure in a footnote instead. I flag this for the editor and take no position.

**M10. Design of the controlled experiment (m9).** I comment on design only. The experiment is meant to deliver σ by lab, a neutrality test for data quality, and a high-M extrapolation check. As designed (§III.B, Table B2, `paper/notes/m9_spec.md`), several features will confound all three unless handled now.

- **(a) The parameter-counting convention is first order at this scale.**
  - The embedding share is 73% at width 128 and 54% at width 192 (Table B2).
  - FLOP per token relative to 6N_non-emb is 3.83 at width 128 and 1.14 at width 640. Relative to 6N_total it is 1.03–1.05. So C = 6N_ne·D understates true compute by a scale-correlated factor of 1.1–3.8. That is exactly the input-measurement bias §IV.D shows moves the allocation exponent by 0.04–0.17.
  - The headline design statistics in Table 2 Panel B use non-embedding counts (M 0.51–2,031, spread 1.96). With total counts they are M 0.46–555 and spread 1.74 (Table 2, note a).
  - The high-M extrapolation check targets "M ≈ 1,000–2,000". That range exists only under the non-embedding convention.
  - *Requested:* pre-specify total counts (with the tied embedding) as primary, or report every m9 result under both conventions. Treat the non-embedding M > 555 region as convention-dependent.
- **(b) High M is reached only by the smallest, shallowest models.** M > 600 occurs only at width 128 (2 layers) and width 192 (3 layers). Budget caps at widths 448–640 remove the high-D corner for large models. The extrapolation check therefore compares high M with small N inside a 2–3-layer architecture. A failure of the Chinchilla form there may reflect depth-2 architecture effects rather than token saturation.
  - *Requested:* add at least two over-trained runs at widths 320–384 at 1,600–3,200M tokens, which is feasible on the stated hardware budget; or restrict the extrapolation test to widths ≥ 256 and report its reach honestly.
- **(c) The learning rate is tuned on one lab at one D.**
  - `m9_spec.md`: the calibration sweep is "lrsweep: LR calibration, D = 50M, edu". The peak LR is therefore tuned on FineWeb-Edu only, at D = 50M, and applied to both labs and all D from 25M to 800M.
  - Bjorck et al. (2025) find LR* ∝ D^−0.32 or steeper at small scale. The LR is thus about 16^0.32 ≈ 2.4 times too high at 800M, and mis-tuned in a direction correlated with D.
  - By the paper's own formula (Eq. 5 and the text that follows, p. 38), a D-gradient in inefficiency biases the fitted β and a. A lab-specific gradient loads directly on the neutrality test, the experiment's central purpose.
  - The fixed 250-step warmup is also a larger share of short runs. This is the Kaplan-style fixed warmup that Porian et al. identify as a flexible-input bias (Table 5, step 3).
  - *Requested:* a small LR sweep (3 LRs) on the FineWeb lab at two D values and two widths, reporting whether the optimal LR differs by lab or by D. If it does, correct for it or bound the implied bias with the paper's inefficiency formula.
- **(d) Dependence structure.**
  - Within a width, all endpoints branch from one constant-LR trunk, and all widths share the initialization seed and data order ("common random numbers").
  - Endpoint residuals are therefore strongly correlated within a width (trunk sharing) and across widths at a given D (shared data order). There are effectively 8 independent trajectories per lab.
  - The m9 spec correctly proposes clustering by width (8 clusters), but with 8 clusters and 6–10 parameters (κ free, lab-specific A, B and E) inference will be fragile. The paper itself notes that "with nine clusters the cluster bootstrap is itself imprecise" (p. 29).
  - *Requested:* pre-specify the inference (wild cluster bootstrap with Webb weights, 8 clusters, reported alongside a parametric residual bootstrap that preserves the trunk structure). Run a power calculation with the paper's own MC machinery for the smallest χ worth detecting (for example the DataDecide tilt, 0.22–0.26).
- **(e) Seed noise is measured for only one lab and three widths.** The 18 seed replicates are FineWeb-Edu only, at widths 128/256/384. The neutrality test needs the noise variance of both labs. Add replicates for FineWeb at the same cells.
- **(f) The "common output" is not neutral.** FineWeb-Edu is a filtered subset of FineWeb, so the two validation sets nest. As the paper notes for DataDecide (p. 36), asymptote shifts on a common validation set partly measure distribution match. Add a third held-out set that neither lab is drawn from (for example C4 validation or Paloma), and report bits per byte (the bytes-per-token values are already in `meta.json`) so that results are tokenizer-free.
- **(g) Pre-registration.** The paper already exhibits several forks: Huber versus NLS, κ, counting convention, cluster versus pairs, and which Ê for rank-one. For the one dataset the authors control, write down before looking the primary estimator, primary convention, inference, and the decision rule for "neutral versus factor-biased". Commit the plan (for example in `paper/notes/m9_spec.md`) with a timestamp.

**M11. The productivity-dispersion comparison with manufacturing depends on the output units (§VI.A, p. 54).**

- The main text concludes that LLM dispersion is "three to five times larger in logs" than Syverson's 1.92. That holds in compute-equivalent (input-equivalent) units: 23.7, and 7–9 for the restricted families.
- Compute-equivalent units convert log-output gaps at the rate 1/θ_C ≈ 3 (HellaSwag) or 1/γ ≈ 5.6 (loss).
- The Online Appendix (Table D-TFP, "Loss units" column) shows that in reducible-loss units the 90/10 ratio is 1.76, below manufacturing's 1.92.
- The "returns to scale near one" justification (p. 54) makes Syverson's ratio input-equivalent. But it does not remove the fact that the LLM ratio looks large because returns to compute are small.
- *Requested:* report both unit systems in the main text and state that the comparison is a cardinalization choice, as the appendix note already does.

---

## 4. Minor comments

Numbered in page order. "→" gives the correction.

1. **p. 1, abstract.** "the median open-weight model is trained as if its lifetime inference compute will be about 2.2 times its training compute" → add "under a reference technology (0.9–3.0 across technologies)". See D8 in Appendix D and M2.
2. **p. 3, Intro.** "σ* lies between 0.73 and 0.83, with bootstrap standard errors between 0.006 and 0.034" versus p. 32, "0.005 and 0.034". See M3(b).
3. **p. 3, Intro.** "By contrast, the allocation exponent a … ranges from 0.37 to 0.57" → see Appendix D4. The juxtaposition implies the forbidden comparison. Compare in units of sampling error, as p. 35 does, or drop "By contrast".
4. **p. 4, Intro.** "so biases of about 20 percent cannot be ruled out" versus p. 53, "spans about ±23 percent". → "about 23 percent".
5. **p. 7, §I.A, notation.** d ≡ ln D, but §III.B (p. 26) uses d for model width ("width d ∈ {128,…,640}, depth d/64, lr*(d)"). → Use W or d_model for width.
6. **p. 12, Fig. 1 notes and p. 16, §II.A.** Figure 1(b) shows members with σ* = 0.60, 0.74, 0.85. The text on p. 16 then reports off-path losses for σ* = 0.89, 0.74 and 0.33 (`m7_theory_dmr_family.csv`). → Report the numbers for the plotted members, or plot the members whose numbers are reported.
7. **pp. 13–14, §I.E, Llama 3 8B.** The text gives M ≈ 1,875 (using N = 8×10^9), w ≈ 5.2, 12.7 tokens and 5.5× compute (Besiroglu parameters). §V.B and Table 6 give M = 1,868 (exact N = 8.03×10^9), ŵ = 5.27, 12.8 and 5.6× (reference technology). All are correct for their inputs. → Say "(nominal 8B; Besiroglu parameters)" in §I.E, or use the Table 6 inputs throughout.
8. **p. 14 versus p. 46, Gopher.** w = 0.36 (Besiroglu) versus 0.37 (reference); both correct (0.362, 0.365). → Name the technology in §V.D.
9. **p. 14, §I.E.** "roughly 1.9×10^14 tokens over its life" is a point level with no band. → Add the band (T̂/D ∈ [4.4, 33] across technologies, p. 43). See Appendix D8.
10. **p. 16, §II.A.** "standard errors of 0.02 in Besiroglu et al." → give the exact values: 0.02 for both α and β, 124.58 for A and 1,293.23 for B. The rounded 125 is fine.
11. **p. 19, Fig. 2 notes.** "Every design has 90 runs on Chinchilla's nine IsoFLOP budgets". → "on nine budgets matching Chinchilla's nominal IsoFLOP budgets" (see M1). Also note that the on-path RMSEs of ln M* (24 at s = 0) depend on the parameter box (m6 memo line 297).
12. **p. 23, §II.E.** "the design reaches M = 341 only at compute up to 1.3×10^22 FLOP" is garbled: the M = 341 run is at about 10^19 FLOP. → "the design's largest ratio is M = 341 and its largest compute is 1.3×10^22 FLOP".
13. **p. 23 versus p. 43.** w_b = 0.76 versus ŵ ≤ 0.77. The CSV (`m7_theory_pi_summary.csv`) has 0.766 under the refit (→ 0.77) and 0.763 under Besiroglu. → Use 0.77 in both places, or name the technology.
14. **p. 23, §III, notation.** "sd(ln M | ln C)" is called the off-path spread and written s_M in Table 4. But s is also the allocation-error s.d. in the MC (Fig. 2) and the omitted-parameter share in Eq. (5) (p. 38). → Rename two of the three.
15. **p. 24, §III.A.** "In these data D = C/(6N) is constructed". Since C is itself digitized, digitization error in C transmits one for one to d and is correlated with c. → Add a sentence noting this errors-in-variables structure; it is relevant to the κ identification caveat on p. 31.
16. **p. 25, Table 2.** Production-scale universe N range "0.14–600B": the minimum is 134.5M (SmolLM2-135M) → "0.13–600B".
17. **p. 25, Table 2.** Porian et al. are listed with Output "Train loss". Panel A of Table 5 uses validation loss for steps 2–5 (Table 5 notes). → "Train/val. loss".
18. **p. 25, Table 2.** Add a row, or a note, for the (Mis)Fitting FineWeb/C4 technology used in §VI.C and Table 9. See M4(d).
19. **p. 26, §III.B.** "Farseer … near-factorial grid of 404 runs". Farseer trained about 1,000 models (arXiv 2506.10972). → State that the 404 are the released single-recipe runs (`1222_full.csv`; m2 memo line 202). Similarly, Step Law trained about 3,700 models; 1,911 runs in 17 cells are the released subset used here.
20. **p. 28, §IV.A.** "Epoch's digitization of the runs in Figure 4 of Hoffmann et al." is fine. But see M1: "245 runs on nine IsoFLOP budgets from 6×10^18 to 3×10^21 FLOP" is wrong.
21. **p. 30, Table 3, Panel D and notes.** Say that the profile LR is Gaussian (M5a). Also, "rank one: τ̂ = det H/‖H‖²_F" is an unconventional statistic; give its null distribution's median (−0.003) in the note.
22. **p. 31.** "A truncated-Gaussian estimator … returns the untruncated estimate (β̂ = 0.406 in both)". "Untruncated" is ambiguous, since the n = 245 Gaussian estimate is 0.612. → "returns the estimate that ignores truncation (0.406)".
23. **p. 32, §IV.B.** "relaxing it … the root mean squared error of ln L on Farseer falls from 0.0085 to 0.0025". 0.0085 is the NLS value; Table 4 Panel B reports the Huber value, 0.0087 (`m2_farseer_forms.csv`). → Use one.
24. **pp. 30 versus 34.** Chinchilla σ*_κ is 0.700 (0.015) in Table 3 and 0.701 (0.014) in Table 4. The point estimates differ in the fourth decimal (0.70048 versus 0.70055) because the m1 and m2 optimizers differ. Table 4's note attributes only the SE difference to separate bootstrap draws. → Harmonize the point estimate.
25. **p. 33, Fig. 4 notes.** "(labeled q in the legend)". → Relabel the figure legend κ, consistent with the text.
26. **p. 36, §IV.C.** DataDecide "(bootstrap p ≤ 0.01)" rests on B = 99; see M6(c). Also, "every restriction is rejected" rests on B = 19 for four tests.
27. **p. 37, Table 5.** "tuned LR, batch, β₂": β₂ (Adam) collides with the data exponent β. → "Adam's second-moment decay".
28. **p. 38, notation.** θ ≡ d ln X/d ln N_m (Eq. 5) collides with:
    - θ_N, θ_D, θ_C (benchmark elasticities, §VI.A);
    - θ_D ≡ 6D/(6D+2T) (Prop. 2(iv));
    - θ = (ln(A/B), α, β) (Prop. 7);
    - θ_T (Appendix Prop. A8).

    φ ≡ 1 − s(1−θ) (Eq. 5) collides with φ = (S, a, γ, ln G, ln K) (Prop. 4) and φ = g_N/g_C (§VI.B). → Rename at least three of these.
29. **p. 39, §IV.D.** "The estimated demand functions replicate Step Law's learning-rate rule". The grid-argmin elasticities (N −0.82, D +0.29; `m8_measurement_demand.csv`) are close to Step Law's published values (N −0.713, D +0.307; m8 memo line 177) but not equal. → "are close to".
30. **p. 41, §V.A.** "six common technologies" (reference plus five alternatives) versus Intro p. 4, "spanned by six alternative estimates". → "six estimates".
31. **p. 43, §V.B.** "Qwen3 0.6B … more than 20 times the largest ratio in any design" is correct against Farseer (2,570; 23.5×) and Marin (2,751; 22.0×). Note that Marin's M range is not reported in §V.
32. **p. 46, §V.E, notation.** T_i = τ_f N_i^{−η}: τ is also the transverse coordinate (Eq. 7) and the rank-one statistic (Table 3), and η is the compute shock in Prop. 6. → Rename.
33. **p. 49, §V.G.** "6.5–9.5 billion parameters, which fit on one 16–24 GB accelerator at 16-bit precision". A 9.5B model needs 19 GB for weights alone at 16-bit, so it does not fit on 16 GB. → "on one 24 GB accelerator (the smaller ones on 16 GB)".
34. **p. 49, §V.G.** The Farseer Eq. 3 comparison ("within Farseer's parameter support (66 models) … above that support (75 models)") covers 141 models, but the Farseer non-embedding technology is defined for 146. → Say what happens to the other five.
35. **p. 49.** M*(C) under Farseer Eq. 3 is not monotone: 32.3 at 10^19 and 25.2 at 10^20 (`m3_wedge_rival_Mstar_curves.csv`). "Rising from 25 at 10^20" is correct but invites the question. → Note the minimum.
36. **p. 52, Table 8.** Family FE θ_D bias −0.262 (0.162) has z = 1.6 but a wild p-value of 0.02, while θ_N +0.203 (0.086) has z = 2.4 and p = 0.05. With 19 clusters this is possible, but readers will ask. → Add a sentence explaining that the two inference methods differ, and report the wild-bootstrap SE.
37. **p. 53.** "first-stage F at most 3.8 in every specification" is true only for the frontier-FLOP/$ instrument (max 3.76; `m4_observational_iv_diagnostics_overlap.csv`). The same paragraph reports F = 10.7 and 10.6 for other instruments. → "for this instrument".
38. **p. 52, Table 8 Panel B.** The IV row's Exp. entry is 0.012 (0.043): the same weak instrument applied to the experimental outputs recovers essentially zero, against a true design-matched elasticity of about 0.33. This is the clearest demonstration in the paper that the instrument, not the data, fails. It deserves one sentence in the text, which currently mentions only the first-stage F. The note's "F < 0.01 with developer fixed effects (not shown)" matches the CSV (F = 0.0054).
39. **p. 55.** "T_C = 12 ln 2/g_C" with T also the lifetime inference tokens. → Write τ_C or DT for the doubling time.
40. **p. 56, Table 9.** Row (8) is labeled "Impose γ = 0.0525" but lists α̂ = 0.058 and β̂ = 0.574. These satisfy γ = αβ/(α+β) = 0.0527, but readers will not see this. → Add "(α and β constrained to γ = 0.0525)".
41. **p. 58, §VI.C.** "Meta's Llama 3 IsoFLOP runs give 0.74" is one of the nine; see M4(d).
42. **p. 59, Conclusion.** "Normalizing inputs at their geometric means cuts the condition number of the Chinchilla Jacobian from about 2,000 to 62" (2,003 → 61.9; PASS). But Table 3 Panel D's "Condition number 62 / 413 / 69–95" rows are all normalized. → Add "raw" and "normalized" labels so readers do not compare 2,000 with 413.
43. **Replication-package consistency.** The module table `output/tables/m1_chinchilla_horse_race.tex` reports M*(5.76×10^23) SEs of (9.0)[25.6], (5.6)[16.3] and (3.9)[10.4]. The paper's Table 3 reports (6.9)[17.2], (4.9)[12.1] and (3.8)[9.2]. I recomputed IQR/1.349 from the stored bootstrap draws and confirm the paper's values; the module table uses a different dispersion measure. → Regenerate the module .tex, or delete it, so that the package does not contain two versions.
44. **Length.** The main text is about 21,400 words of LaTeX source (the plan's target is 16,000–18,000) plus 9 tables and 9 figures. The Monte Carlo detail in §II.B (p. 18, two paragraphs) and the Step Law paragraph (p. 39) are candidates for the Online Appendix.

---

## Appendix A. Number-by-number audit ledger

Format: **location — statement (value in text) — source — status**. Values are PASS unless marked otherwise. For brevity, several values sharing one source are grouped on one line.

### A.1 Abstract and Introduction (pp. 1–6)

- p. 1 — median w ⇒ inference/training ≈ 2.2 — `m3_wedge_models.csv` (recomputed median ŵ = 3.186; ŵ − 1 = 2.19) — PASS. Level is κ = 1-specific; see M2.
- p. 1 — σ "near 0.7" in seven sweeps — Table 4/m2 — PASS for Farseer and Chinchilla only; 0.51–0.62 elsewhere. **Misleading**; see M3(a).
- p. 1 — compute growth 4–5×/yr — Sevilla and Roldán (2024) — PASS.
- p. 1 — costs 2.4×/yr since 2016 — Cottier et al. (2024) — PASS.
- p. 2 — MC: 90 runs, equal compute; RMSE σ* 0.28 versus 0.004 — `m6_montecarlo_designA_summary.csv` (0.2819; 0.00357) — PASS.
- p. 2 — κ-free profile covers [0.50, 0.95] in 88–97% of reps for s ≤ 0.2 — `m6_…_profile.csv` (0.880, 0.973, 0.913, 0.933, 0.893) — PASS.
- p. 2 — 0.15 log points; LR ≤ 2.0; random 79–101; full design 424 — `m1_chinchilla_fdep_sizecontrol.csv`, `…profile_sigma.csv` (1.96; 79.0–101.4; 424.5) — PASS.
- p. 3 — σ* 0.73–0.83; SE 0.006–0.034 — `m2_table3_technology.csv` (seven sweeps excluding DataDecide: 0.7345–0.8250; 0.0062–0.0338) — PASS. Inconsistent with p. 32; M3(b).
- p. 3 — κ = 1 rejected in all seven — `m2_spec_tests.csv` (p_q1_wald ≤ 1.6×10^-4) — PASS.
- p. 3 — κ-free 0.51–0.71 — (0.511–0.710) — PASS.
- p. 3 — Farseer 0.69–0.71; 0.690 [0.679, 0.699] — `m2_farseer_local_sigma_summary.csv` — PASS.
- p. 3 — a 0.37–0.57 — (0.370–0.566; excludes DataDecide 0.315) — PASS with caveat M3(b).
- p. 3 — M*(10^21) 3–60 — (3.35–59.6) — PASS.
- p. 3 — Gadre p = 0.90 — `m2_neutrality.csv` (0.896) — PASS.
- p. 3 — DataDecide asymptote shifts 98% — (98.06) — PASS.
- p. 3 — M* differs 2.9–3.4× — `m2_neutrality_magnitudes.csv` (2.93; 3.43) — PASS.
- p. 3 — Kaplan exponent 0.84 → 0.50 on Porian's runs — Table 5 (0.837 → 0.496) — PASS.
- p. 3 — 39–47% measurement; 53–61% flexible inputs — Table 5 (0.39; 0.47) — PASS.
- p. 4 — 173 models, 2021–2025 — recomputed — PASS.
- p. 4 — median 3.19; 95% > 1; 79% band > 1 — recomputed (3.186; 94.8%; 78.6% = 136/173) — PASS.
- p. 4 — median across technologies 1.9–4.0 — recomputed (1.909 Hoffmann; 4.027 Gadre RW) — PASS.
- p. 4 — share w < 1: 47% (2021) → 0 (2024) — `m3_wedge_trends.csv` (0.474; 0.000) — PASS.
- p. 4 — median 1.06 (2022) → 4.68 (2025) — (1.064; 4.676) — PASS.
- p. 4 — Llama 3 8B 5.27 [3.97, 8.12]; ≈13 tokens (12.8); 6 under Meta's law (6.27) — `m3_wedge_models.csv` — PASS.
- p. 4 — 405B = 0.98 under Meta's law — (0.983) — PASS.
- p. 4 — downloads 0.97 (0.18); 0.72 with FE — `m3_wedge_validation.csv` — PASS.
- p. 4 — 57 models; bias −0.001 (0.038) — `m4_observational_table6_lalonde.csv` — PASS.
- p. 4 — "biases of about 20 percent cannot be ruled out" — CI ±0.075/0.329 = ±23% — **FAIL (inconsistent with p. 53)**.
- p. 4 — FE: +0.20 / −0.26 — (0.203; −0.262) — PASS.
- p. 4 — F ≤ 3.8 — `…iv_diagnostics_*.csv` (max 3.76) — PASS.
- p. 4 — 8.4 bootstrap median; 8.7 point; 6.1 [3.0, 22.7]; 6.1–10.2; [4.1, 40.5] — `m5_progress_replication.csv`, Table 9, m5 memo — PASS.
- p. 4 — allocative gains 1.0–2.4× — Table 9 B (1.05–2.41) — PASS.
- p. 4 — 2–40% — (0.02–0.40) — PASS.
- p. 5 — related-literature attributions — see Appendix B.

### A.2 Section I (pp. 6–14)

- p. 7 — Kaplan κ ≈ 0.103, inner exponents 0.738 and 1, E = 0 — Kaplan et al. (2020) Table 2 (α_N = 0.076, α_D = 0.103; ratio 0.738) — PASS.
- p. 8 — 18 correspondences in Table 1 — count 7+4+3+4 — PASS.
- p. 10 — Besiroglu γ = 0.178; 1% less loss needs ≈5.6% more compute — `m7_theory_ledger_checks.csv` (0.1783; 1/γ = 5.61) — PASS.
- p. 10 — Hoffmann a between 0.46 and 0.50 — Hoffmann et al. (A1 0.50, A2 0.49, A3 0.46) — PASS.
- p. 11 — σ* = 0.737; σ range 0.732–0.742 — recomputed (1/1.366, 1/1.348) — PASS.
- p. 11 — Hao and Merrill's 0.762 equals σ* for Hoffmann's exponents — arXiv 2605.16430 HTML (0.7622 from averaged α = 0.31205); ledger 0.762 — PASS.
- p. 11 — Kaplan σ ∈ [0.500, 0.575], σ* = 0.535 — recomputed — PASS.
- p. 12 — Fig. 1: orange dot w = 5.2 — ledger (5.2175) — PASS.
- p. 13 — Llama 3 8B M ≈ 1,875; w ≈ 5.2 (5.3 reference); 12.7 tokens per token; 1.9×10^14 tokens — ledger (5.218; 12.65) and models CSV (5.27) — PASS. Inconsistent inputs with §V; minor 7.
- p. 14 — 5.5× compute; Farrell CE 0.18 — ledger (5.513) — PASS.
- p. 14 — Harberger accurate within 1% for |ln w| ≤ 0.7; overstates by 23% at w = 5.2 — `m7_theory_harberger.csv` (0.84% at w = 2; 6.776/5.518 = 1.228) — PASS.
- p. 14 — Gopher w = 0.36 and 2.0×; GPT-3 w = 0.43 — ledger (0.362; 2.01; 0.4265) — PASS.

### A.3 Section II (pp. 14–23)

- p. 15 — 61 symbolic/numerical checks — `m7_theory_claims.csv` (61 rows) — PASS.
- p. 16 — Besiroglu SEs 0.02 (exponents), 125 and 1,293 (A, B) — `lit/notes/econ_ai.md` (124.58; 1,293.23) — PASS.
- p. 16 — off-path loss 1.4 / 4.1 / 18.2% at 5M* — `m7_theory_dmr_family.csv` (1.449; 4.128; 18.21) — PASS. Members differ from Fig. 1(b); minor 6.
- p. 17 — sd(ln w) = 0.48 over 240 runs — 1.349 × 0.357 = 0.482 — PASS.
- p. 17 — information slopes 4.00 (σ*) and 2.00 (ln M*) — recomputed from `m7_theory_fisher_information.csv` (4.00; 2.00) — PASS.
- p. 19 — Fig. 2 notes: σ* = 0.737; M* = 18.1 at 10^24 (Besiroglu, interpolated 18.14); noise s.d. 0.0075 (m6 memo); budgets and total compute 5.1×10^22 (5.106×10^22); 500/150 reps; Kaplan C^0.73, s = 0.1 — diagnostics CSV, memo — PASS. Budget description: see M1.
- p. 18 — corner solutions 51%; RMSE σ* 0.28 and RMSE ln M* 24 — summary CSV (0.510; 0.282; 23.995) — PASS.
- p. 18 — σ* RMSE 0.067 / 0.025 / 0.015 at s = 0.05 / 0.1 / 0.2 — (0.0670; 0.0246; 0.0147) — PASS.
- p. 18 — ln M* RMSE 4.5 / 2.0 / 0.59 at s = 0.1 / 0.3 / 1 — (4.49; 1.97; 0.590) — PASS.
- p. 18 — 5–95% range of M̂* spans ≈850× — q05 −0.399, q95 6.344; e^6.743 = 848 — PASS.
- p. 18 — IsoFLOP design: 0.0036 and 0.16 — (0.00357; 0.162) — PASS.
- p. 18 — RMSE 0.013–0.025 for 0.1 ≤ s ≤ 0.3; dual ≈0.012 — (0.0131–0.0246; 0.0117–0.0121) — PASS.
- p. 18 — flat profile share 88% on path, 65% at s = 0.3; s = 1 median width 0.10; IsoFLOP 0.016, factorial 0.019 — profile CSV (0.880; 0.653; 0.104; 0.0161; 0.0186) — PASS.
- p. 18 — flat share 38 / 65 / 90% by noise level — (0.380; 0.653; 0.900) — PASS.
- p. 18 — dual on path: RMSE 0.011 (σ*) and 0.009 (ln M*) — (0.0113; 0.0094) — PASS.
- p. 18 — Kaplan-belief labs: +0.22 in a, −0.09 in σ*, −3.9 in ln M* ("fiftyfold") — (0.217; −0.088; −3.917; e^3.917 = 50.3) — PASS.
- p. 18 — Wald singular or corner in 35% — 1 − 0.652 — PASS.
- p. 18 — warm-start coverage 67% — (0.673; R = 500) — PASS. Like-for-like issue: M4(e).
- p. 18 — multistart coverage 92% (60 reps); median width 0.43 versus 0.016 — `…bootcheck.csv` (0.917; 0.428) and summary (0.0158) — PASS.
- p. 18 — designed-variation coverage 95–97% — (0.946–0.973) — PASS.
- p. 19 — Llama 3 8B at ≈100× its compute-optimal ratio — 1,875/18.1 = 104 — PASS.
- p. 21 — equivalent members σ* 0.33–0.89; frontiers equal to 10^-14 in ln R*; bias 0.05–0.74/yr — dmr CSV (0.046–0.743), m7 review item 13 — PASS.
- p. 22 — simulated slopes within 4×10^-4 of the closed form — transmission CSVs (max 4.24×10^-4 on path; 8.49×10^-4 overall) — **FAIL**; M4(b).
- p. 22 — forward 0.248, reverse 0.403 (γ = 0.178, π₁ = 2) — `m7_theory_transmission_mc.csv` (0.2475; 0.4029) — PASS.
- p. 22 — funding net bias +0.100 / −0.006 / −0.046 — `m6_montecarlo_selection_check.csv` — PASS.
- p. 23 — 300 draws; sd(ln ŵ) = 0.161; parts 0.143 and 0.047 — `m7_theory_pi_summary.csv` — PASS.
- p. 23 — M = 341; compute 1.3×10^22; 55-fold extrapolation — (341.1; 1.2956×10^22; 55.6) — PASS. Wording: minor 12.
- p. 23 — w_b = 0.76 — (0.766 refit, 0.763 Besiroglu) — **minor FAIL/inconsistent** with p. 43; minor 13.

### A.4 Section III and Table 2 (pp. 23–27)

- p. 23 — off-path spread 1.35 ⇒ sd(ln w) 0.48 — PASS.
- p. 24 — 245 runs; 5 dropped → 240 — PASS. "Quantized to about ±0.01" — UNTRACED (not in m1 CSVs; plausible from digitization).
- p. 24 — Farseer 404 runs; M 0.31–2,570; spread 1.97, tied with Muennighoff's 33 runs — `m2_robustness.csv` (0.314–2,569.7; 1.967 versus 1.971) — PASS.
- p. 24 — Gadre 104; OLMo 30; DataDecide 25 × 14; Llama 3 133; Marin 258 (85+85+88); Porian 975; Step Law 1,911 in 17 cells — m1, m2 and m8 memos — PASS.
- p. 24 — spread range 0.73–1.97 — PASS.
- p. 26 — 55% of Sample B have M > 341 — recomputed (54.9%) — PASS.
- p. 26 — experiment: N_ne 0.39–49M; total 1.4–54M; 44 endpoints; M 0.51–2,031; lr formula; 1,048,576 eval tokens — Table B2, `m9_spec.md` — PASS.
- p. 26 — experiment off-path spread 1.96 (1.74 with total N) — UNTRACED (m9 not yet run; design-based figure).
- p. 27 — largest runs < 10^17 FLOP; ≈5 orders of magnitude below Chinchilla — 6.7×10^16 including attention; 1.3×10^22/6.7×10^16 ≈ 2×10^5 — PASS.
- p. 27 — 173; 335 (271 open); 128/38; 231 rows / 144 papers — m3, m4 and m5 memos — PASS.
- p. 27 — benchmark panel spread 1.64 — `m4_observational_design_audit.csv` (1.636) — PASS.
- p. 27 — 10% error in D ⇒ ≈4% change in w — 0.367 × ln 1.1 = 3.5% — PASS.
- Table 2 — Chinchilla 57M–16B, M 0.46–341, 1.35 — technologies CSV — PASS.
- Table 2 — Farseer 0.10–6.4B — PASS.
- Table 2 — Gadre 11M–6.9B, M 5–640, 1.44 — PASS.
- Table 2 — OLMo 0.19–3.2B, M 10–200, 0.94 — PASS.
- Table 2 — DataDecide 1,100 runs, spread 0.73 — PASS.
- Table 2 — Llama 3 59M–17B, M 0.44–643 — PASS.
- Table 2 — Porian 5M–0.90B — PASS.
- Table 2 — Step Law 0.21–1.07B, M 19–466 (memo 18.6–466) — PASS.
- Table 2 — Sample B 70M–406B, M 2.1–60,398 — recomputed — PASS.
- Table 2 — production-scale universe M 0.2–60,398 — PASS; **N "0.14–600B" FAIL (min 0.1345B)**.
- Table 2 — Epoch LMs 247 — m5 memo — PASS.
- Table 2 — Muennighoff N/M ranges; Marin ranges; benchmark panel N/M ranges; Ho N/M ranges — UNTRACED in CSVs (not audited in detail).

### A.5 Section IV and Tables 3–5 (pp. 27–40)

- p. 28 — "nine IsoFLOP budgets from 6×10^18 to 3×10^21" — raw data 1.4×10^18–1.3×10^22; 137/245 on profiles — **FAIL**; M1.
- p. 28 — design variance 0.90 transverse versus 1.83 along the path — `m1_chinchilla_design_variance.csv` — PASS.
- p. 28 — 84% of residuals in the linear region; Huber–LAD SD 0.004 — `m1_chinchilla_horse_race.csv` (0.8375); `…huber_vs_lad.csv` (0.00396) — PASS.
- p. 28 — β = 0.367 / 0.406 / 0.428 — PASS.
- p. 28 — paired differences 0.039 (0.018) and 0.060 (0.020) — `m1_chinchilla_estimator_diffs.csv` — PASS.
- p. 28 — σ* 0.718–0.737; a 0.513–0.545 — PASS.
- p. 28 — ΔM* −3.7 (4.9) and −7.6 (5.3) — PASS.
- p. 29 — cluster SE factor 1.5–2.2 (β 0.021 → 0.046) — horse race (α 1.51; β 2.17) — PASS.
- p. 29 — M* interval [7.8, 35.1] → [3.2, 126] — recomputed from `.npy` draws — PASS.
- p. 29 — β̂ 0.453 → 0.367; paired diff 0.086 (0.052); 0.347 at fifteen dropped — `…selection_swing.csv`, `…selection_k.csv` (0.3467) — PASS.
- p. 29 — Hausman–Wise β̂ = 0.406 in both — `…selection_rules.csv` — PASS.
- p. 29 — dropped runs have D/N 0.04–0.40 — `…worst_points.csv` — PASS.
- p. 29 — residuals 9–45× the residual s.d. — 0.060/0.0069 to 0.311/0.0069 — PASS.
- p. 29 — dropping D/N < 0.4 gives β̂ = 0.371 — PASS.
- p. 31 — Chinchilla-70B wedge 1.54 with the outliers — PASS.
- p. 31 — "0.95–1.06 under every rule that removes them" — **FAIL**; M4(a).
- p. 31 — Approach 2: a = 0.498 (0.025) — `m1_chinchilla_duality_objects.csv` (0.4977; OLS SE 0.0252) — PASS.
- p. 31 — design-consistent p ≥ 0.16 — `…duality_bootcal.csv` (min 0.156) — PASS.
- p. 31 — Hoffmann p = 0.040 / 0.060 / 0.21–0.38 — classicalF, bootcal — PASS.
- p. 31 — Δln N* = 0.122 (0.046); ≈12% smaller; slope gap 0.042 (0.023) — classicalF — PASS.
- p. 31 — Approach-1 minima 0.005–0.008 below the frontier; rejection does not survive the wild bootstrap — bootcal (p_frontier_boot wild 0.15–0.70) — PASS.
- p. 31 — Chinchilla-70B w 1.04 [0.82, 1.42]; Hoffmann 0.62 / 0.71; cluster [0.54, 2.01] — horse race, ledger — PASS.
- p. 31 — system LR 1.01, p = 0.60; SE of a 0.015 versus 0.018 — `…system_lr.csv`, duality objects — PASS.
- p. 31 — CES not rejected — `…spec_ces.csv` (p = 0.50 [0.76]) — PASS. Selective; M6(a).
- p. 31 — κ̂ 0.774 (0.060; 0.111); p < 0.001; cluster p = 0.042 — `…spec_kappa.csv` — PASS.
- p. 31 — σ* 0.737 → 0.700 (0.015); a 0.504; γ 0.165 — spec_kappa — PASS.
- p. 31 — rank-one p = 0.69 — `m2_spec_tests.csv` (p_rank_one_q 0.687) — PASS.
- p. 32 — 41 runs; transverse share 0.8% — fdep, design_variance — PASS.
- p. 32 — LR ≤ 2.0; full 424; set [0.67, 0.69] — profile CSV (Gaussian) — PASS numerically. **Inconsistent with 0.700**; M5.
- p. 32 — random subsamples 79–101 — PASS.
- p. 32 — SE ratios 2.3 (α), 2.2 (a), 0.94 (γ); on-path σ* SE 0.011 — `…fdep_se_ratios.csv` — PASS.
- p. 32 — spread 1.97 / 0.73 — PASS.
- p. 32 — σ* 0.735–0.828; SE 0.005–0.034; κ̂ 0.22–0.77 — m2 technology CSV — PASS. Inconsistent with pp. 3 and 35; M3(b).
- p. 32 — RMSE 0.0085 → 0.0025 — `m2_farseer_forms.csv` (NLS 0.0085; κ 0.0025) — PASS. **Inconsistent with Table 4 (0.0087, Huber)**; minor 23.
- p. 32 — σ*_κ 0.51–0.71; Farseer 0.710 [0.705, 0.716]; local 0.690 [0.679, 0.699] at 42 points; Farseer form 0.706; local range 0.61–0.99; Chinchilla-implied 0.77 — m2 CSVs — PASS.
- p. 32 — out-of-sample 0.022–0.025 versus 0.0026 — forms CSV (0.0220 NLS, 0.0247 Huber; 0.0026) — PASS.
- p. 35 — capital–labor 0.4–0.7 — citations — PASS.
- p. 35 — CES Wald p 0.22–0.90 — spec tests — PASS.
- p. 35 — rank-one: 6 of 7 not rejected; Farseer p ≤ 0.003 with CI containing 0 — spec tests (κ-Ê) — PASS. See M6(b).
- p. 35 — a 0.37–0.57; M* 3.4–60; single-sweep intervals 1.7–70× — m2 technology CSV (interval ratios 1.75–69.8) — PASS.
- p. 35 — σ* 0.735–0.825; γ 0.10–0.18 — PASS (excluding DataDecide). M3.
- p. 35 — Farseer embeddings: a 0.526 → 0.411; M*(10^23) 19.5 → 45; σ* 0.772 → 0.724 — `m2_robustness.csv` — PASS.
- p. 35 — Muennighoff D/N < 0.4 dropped: a 0.535 → 0.360; M* 60 → 139; σ* 0.735 → 0.726 — PASS.
- p. 35 — Gadre RW M*(10^21) 3.4 (Huber) versus 10.4 (LS) — PASS.
- p. 35 — σ*²/2 ≈ 0.3 — PASS.
- p. 35 — Gadre: p ≤ 0.002; E-shift p = 0.004 explaining 90%; Hicks p = 0.90 explaining 96.8% versus 96.9%; augmenting alternatives p = 0.13 / 0.15; M* differs 8–18% — `m2_neutrality*.csv` — PASS.
- p. 36 — DataDecide: 21,888 checkpoints; LR 145 on 24 df; 98.1 / 99.71 / +0.05; tilts 0.22–0.26; M* 2.9–3.4×; wedge 1.25–1.29×; σ*_r 0.790–0.839 — m2 CSVs — PASS. p-value resolution: M6(c).
- p. 38 — Porian: ten exponents reproduced within 0.003 (max 0.002); 0.837 → 0.496; 0.862 → 0.517; shares 39 / 47%; 0.12–0.17; 0.19–0.23; 5M–901M — Table 5 and m8 CSVs — PASS.
- p. 38 — Pearce–Song predictions 0.792 / 0.757 — `m8_measurement_pearce_song_sim.csv` — PASS.
- p. 38 — Pearce and Song's simulated 0.78 / 0.74 — arXiv 2406.12907 HTML — PASS.
- p. 38 — formula over-predicts by 20–50% on the actual design — `…porian_meas_formula.csv` (23–52%) — PASS.
- p. 38 — Chinchilla non-embedding a 0.514 → 0.556 (0.042, 0.006) — `…chinchilla_nonembed.csv`, m8 memo — PASS.
- p. 38 — inefficiency formula explains 47–114%; 1.57B-token warmup; 1.57 nats at 2.5×10^16 FLOP, gone by 3×10^18 — m8 memo lines 120–121 — PASS.
- p. 39 — Step Law: up to 120 configurations per cell; SFA −0.23 (0.03) and −0.04 (0.08) — `m8_measurement_sfa.csv` (cluster SEs 0.032 and 0.076) — PASS.
- p. 39 — Bjorck −0.32 — PASS (batch fixed; ≥760M models; M8(c)).
- p. 39 — batch fixed: 0.046 (0.015); closes 26–43% of the gap — demand CSV, memo — PASS.
- p. 39 — SSR varies < 17% for E 0.6–1.7; a 0.76 → 0.45; σ* moves 7.0 own SEs versus 2.8 for a — m8 memo, `…stability.csv` (6.97; 2.85) — PASS.
- Table 3 — all Panel A cells, including IQR-based M* SEs (recomputed: 6.92 [17.18], 4.95 [12.05], 3.84 [9.23]); Panel B; Panel C (0.65 / 0.66 / 0.040; 0.23 / 0.18 / 0.060; gaps; wedges 1.04 / 1.03 / 0.62–0.71); Panel D (0.33 / 0.008 / 0.34; 62 / 413 / 69–95; 424 / 2.0 / 79–101); notes (LAD 0.348 / 0.366 / 0.737 / 1.03; wild 0.70 / 0.75 / 0.21; column w 1.14 / 1.30 / 1.03) — PASS.
- Table 4 — all 8 rows × 10 columns and Panel B (RMSEs, BIC −2,647 / −3,663 / −3,916) — `m2_table3_technology.csv`, `m2_farseer_forms.csv` — PASS. Chinchilla σ*_κ 0.701 versus Table 3 0.700: minor 24.
- Table 5 — Panel A (all 20 estimates and CIs versus the m8 CSV and Porian's published values); Panel B (all 16 cells) — `m8_measurement_porian_counting.csv` — PASS.

### A.6 Section V and Tables 6–7 (pp. 40–50)

- p. 41 — reference α 0.347, β 0.367, σ* 0.737, M*(10^21) 21.4; 400 draws — `m3_wedge_technologies.csv` — PASS.
- p. 41 — band σ* 0.72–0.80; M*(10^21) 3.4–34 — PASS.
- p. 41 — 68 families; ∂ln ŵ/∂ln D = 0.37; 20% tokenizer difference ⇒ 7% — recomputed — PASS.
- p. 41 — 55% (> 341); 29% (> 1,227); 75% above 1.3×10^22 — recomputed; `…sampleB_by_tech.csv` (0.746) — PASS.
- p. 42 — median 3.19; T/D 6.6; ratio 2.2; 95% > 1; 75% > 2; 87% (80% cluster); 79% (136/173); median band [1.77, 6.92]; 2.4×; CE 0.41 — recomputed — PASS.
- p. 43 — technology medians 1.91 / 2.12 / 2.41–2.67 / 3.16 / 4.03 / 3.85 / 4.11 / 4.78 — recomputed — PASS.
- p. 43 — interval 0.80–1.39× the point estimate; Spearman 0.95–1.00; M* 21.4 at 10^21 and 17.6 at 10^24 — recomputed; technologies CSV — PASS.
- p. 43 — Llama 3 8B: 1,868; 5.27 [3.97, 8.12]; T̂/D 12.8 [8.9, 21.4]; T̂ ≈ 1.9×10^14; 3.09; 6.3; 4.82; T̂/D band [4.4, 33]; CE 0.18; 5.6× — models CSV — PASS.
- p. 43 — 405B 0.98 (reference 1.37 [0.99, 2.22]); Llama 2 70B 0.98 (1.19 [0.93, 1.67]) — PASS.
- p. 43 — Qwen3 0.6B: 36T tokens; M 60,398; ŵ 17.9; >20× the largest design ratio — PASS.
- p. 43 — Sardana: extrapolated laws overstate token gains — arXiv 2401.00448 abstract — PASS.
- p. 43 — Farseer local σ falls with M — `m2_farseer_local_sigma_summary.csv` (slope −0.031 [−0.040, −0.024]) — PASS.
- p. 45 — stated intent: 1.04 [0.82, 1.42]; seven Cerebras 0.92–1.00; LLaMA-1 1.65–2.07; Llama 3 2.48–5.27; Pythia 1.09–6.49 — `m3_wedge_stated_intent.csv` — PASS.
- p. 45 — 335 (271/64); share w < 1: 47 / 41 / 18 / 0 / 2%; medians 1.06 / 1.83 / 3.45 / 4.68; median M 23 → 1,303 — trends CSV — PASS.
- p. 46 — GPT-3 0.43; Gopher 0.37; MT-NLG 0.28; PaLM 0.41 — models CSV — PASS.
- p. 46 — open premium 0.57 (0.09) and 0.44 (0.09); 94 clusters; 11 dropped rows ⇒ 0.59 — `m3_wedge_trends_reg.csv` — PASS. (The second robustness figure, 0.43, was not displayed in my extract: UNTRACED.)
- p. 46 — developer medians HF 6.6, Alibaba 4.8, Google 2.3, Meta 1.8, Cerebras 1.0 — `m3_wedge_labs.csv` — PASS.
- p. 46 — 64 closed models, 3 from 2025 — trends CSV — PASS.
- p. 46 — 95% of 105 siblings in 44 families; five Cerebras at 0.92–0.98 — family CSVs (44 families, 149 − 44 = 105 siblings; five w_rel < 1) — PASS.
- p. 46 — flagships: 18 pre-2024 (median 1.14, half contain 1); 26 from 2024 (3.14, 92% above 1); Qwen3 32B 4.46 [3.22, 7.33] — `…flagships_by_period.csv` — PASS.
- p. 46 — η̂ 0.53 (0.07); F = 2.55, p = 0.001 (0.0007); 138 models / 42 families; 0.047 (58 / 18); 0.12 (33 / 11) — `m3_wedge_overid_demand.csv` — PASS.
- p. 47 — validation: 164 models / 26 developers; 0.97 / 1.13 / 0.72 / 0.94; 0.08 (0.04); 0.29 (0.16) on 46; likes 0.13 (0.25); votes −0.17 (0.40); OpenRouter lists 21 of 164 — validation CSV, memo — PASS.
- p. 48 — one of 171 post-2024 models has w < 1; tiers 26% versus 9%; +0.11 (0.06); median ln ŵ 1.16; memory cap ⇒ w = 1.78 — rival CSVs, `m7_theory_constraints.csv` (1.784) — PASS.
- p. 49 — "16–24 GB accelerator" for 9.5B at 16-bit — **factual FAIL** (19 GB); minor 33.
- p. 49 — Farseer Eq. 3: M* 25 / 82 / 195; 66 models 4.12 versus 3.44; 75 models 1.54 versus 2.12; Llama 3 70B 0.62 — `…rival_Mstar_curves.csv`, memo — PASS.
- p. 49 — distillation 0.02 (0.23); synthetic data 0.06 (0.18) — `…rival_regressions.csv` — PASS.
- p. 49 — recipe bias 1.25–1.29 (1.80; Gadre 1.02–1.04); 88 / 79 / 69 / 49% — m2 magnitudes; recomputed — PASS.
- p. 49 — aggregate 2.0 [1.2, 3.7], band [0.4, 7.2]; 2025 3.2 [2.1, 5.7], band [1.05, 9.6]; 2024 1.2 [0.6, 2.3]; 2022 −0.40 (0.03 truncated); Hoffmann 0.67 — aggregate CSVs — PASS. (Gadre 3.26 was not displayed in my extract: UNTRACED.)
- Table 6 — all 17 rows × 9 columns (N, D, M, ŵ and CI, T̂/D, CE, band, own law) — `m3_wedge_models.csv` — PASS.
- Table 7 — all 18 coefficients and SEs, and all N and developer counts — `m3_wedge_validation.csv` — PASS.

### A.7 Section VI and Tables 8–9 (pp. 50–58)

- p. 50 — 128 models / 38 families / 22 developers; March 2021–July 2024 — design audit — PASS.
- p. 51 — curvature F = 352 and 70 — `m4_observational_curvature.csv` — PASS.
- p. 51 — θ_D 0.58 → 0.09; median D/N 213; 90th percentile 1,715 — m4 memo — PASS.
- p. 51 — observed 0.19–0.26 versus experimental 0.35–0.51, i.e. 27–63% attenuation — m4 memo (arithmetic on the rounded endpoints gives 26–63%) — PASS.
- p. 51 — 88 runs, R² = 0.994; 57 models — PASS.
- p. 53 — 0.328 (0.037) versus 0.329 (0.009); −0.001 (0.038); p = 0.99 — Table 8 CSV — PASS.
- p. 53 — biases −0.03 to +0.04 over six estimators — PASS.
- p. 53 — CI [−0.075, 0.074]; ±23% — computed (−0.001 ± 1.96 × 0.038) — PASS.
- p. 53 — ARC/Winogrande 18–31%; subsample bias 0.045 (0.023), p = 0.10 — Online Appendix table (not re-traced to CSV) — UNTRACED in this audit.
- p. 53 — N ≤ 9B — Table 8 notes — PASS.
- p. 53 — family FE +0.20 (0.09; p = 0.05) and −0.26 (0.16; p = 0.02) — PASS. Inference oddity: minor 36.
- p. 53 — within-family s.d. 0.42 versus 1.29; 18 of 38 families vary D — design audit — PASS.
- p. 53 — 0.42 versus 0.27 — PASS. θ_N − θ_D = 0.21 (0.08) on 128 models — UNTRACED.
- p. 53 — DataDecide regimes: +0.044; −0.18 (40%); ≈5× dispersion ⇒ ≈+0.2 — `…regimes_semisynthetic.csv`; tfp CSV (0.491/0.099 = 4.96) — PASS.
- p. 53 — frontier-FLOP/$ F ≤ 3.8; own-hardware F = 10.7; export control F = 10.6 with 0.90 (0.25) — IV diagnostics CSVs — PASS.
- p. 53 — "three values identified by six models"; "1.7–2.3×"; "six product-line cells" — UNTRACED.
- p. 53 — reliability ≥ 0.988 — m4 memo — PASS. Conflicts with Table 8 EIV rows; M7.
- p. 54 — TFP 90/10: 23.7 [9.1, 197]; 7–9 restricted; 11.1 [4.4, 18.8]; Mertens et al. 41; Syverson 1.92; "three to five times in logs" — tfp CSV; lit notes (4.86 and 2.98 in logs) — PASS. Framing: M11.
- p. 54 — Phi 1st and SmolLM 3rd of 38 — UNTRACED.
- p. 54 — industry MC: +13%, −29%, +66% — m6 memo line 120 — PASS.
- p. 54 — FOC system bias 1.25–1.28, "exactly" the closed form — m6 memo — PASS for the range; **"exactly" FAIL**; M4(c).
- p. 54 — 54–60% of progress absorbed — memo (−0.081 to −0.090 of 0.150) — PASS.
- p. 54 — Ho et al.: 8.4; 231 / 144; loss shares 0.55 / 0.45 (s̄ = 0.545); a = 0.37 — m5 memo — PASS.
- p. 55 — parameters within 4×10^-7; 8.44; 8.7; 18 iterations; objective 0.0518 → 0.0507; 300 starts; 6.1 [3.0, 22.7], twice as wide; 0.27%; 10.2 [4.1, 27.6] — replication CSV, memo — PASS.
- p. 55 — correlation −0.73 to −0.82; profile [4.1, 40.5]; cluster [4.1, 27.6] — memo — PASS.
- p. 55 — φ [−0.39, 3.37] ⇒ T_C 5–38 — memo (5.4–38 over [−0.25, 3.25]) — **minor FAIL**; M4(f).
- p. 55 — neutrality p = 0.29 / 0.94; imposing it gives 9–12 months — memo; Table 9 — PASS.
- p. 57 — Ho data exponent 0.04 versus 0.37; Whitfill's ninefold — Table 9; Whitfill (2025) — PASS.
- p. 57 — MSE ×1.3–4.8 across 14 technologies — Table 9 A (0.060/0.0453 to 0.219/0.0453) — PASS.
- p. 57 — γ 0.178 → 0.0525; mean total-loss elasticity 0.054; gap 2–2.5× rather than 7–9× — `m5_progress_attenuation_panelB.csv` — PASS.
- p. 57 — common E from 0 to 1.5 doubles γ̂; T_C 10.2 → 12.4 — `m5_progress_E_profile.csv` (γ 0.0223 → 0.0463; 10.20 → 12.39) — PASS.
- p. 57 — 62–140 epochs; T_C raised 1.5–4× — memo claim 7 — PASS.
- p. 57 — Sahal inflation 1.15 [1.10, 1.21] and 1.44 [1.24, 1.78] — `m5_progress_sahal.csv` — PASS.
- p. 58 — 7 / 92 models; CE 0.57 → 0.60, gain 1.05 [0.57, 1.72]; 1.9–2.4; 1.6; median M 1.7 → 92; 8.9×; 2–40%; denominator 6.4–22.5 — Table 9 B CSV, memo — PASS.
- p. 58 — "nine technologies estimated in Section IV", range 0.55–4.3; Llama 3 0.74 — numbers PASS; **provenance FAIL**; M4(d).
- p. 58 — Kaplan-rule counterfactual 1.6 / 5.2 / 11.7 (Besiroglu) and 2.4 / 11.5 / 32 (Hoffmann); tenfold fall under Ho's exponents (gain 0.09) — `m5_progress_kaplan_counterfactual.csv` — PASS.
- Table 8 — all Panel A–C cells — `m4_observational_table6_lalonde.csv` — PASS (Panel C composite and ladder-only rows not re-traced).
- Table 9 — all Panel A rows (T_C recomputed from g_N + g_D; Hicks-neutral rows satisfy αg_N = βg_D to three decimals) and all Panel B rows (shares recomputed as ln(gain)/ln 8.9, 9.9, 8.2) — m5 CSVs — PASS.

### A.8 Section VII (pp. 58–60)

- p. 59 — σ range 0.5–0.8; 0.69–0.71; 0.73–0.83 — PASS. M3.
- p. 59 — γ 0.10–0.18 — PASS (excludes DataDecide).
- p. 59 — M* 3–60; cluster interval [3.2, 126] — PASS. Clustering: M1.
- p. 59 — RMSE 0.004 versus 0.28 — PASS.
- p. 59 — relaxing κ lowers σ* by 0.04–0.28 — Table 4 (0.036–0.281) — PASS.
- p. 59 — cluster SEs 1.5–2.2× — PASS.
- p. 59 — conventions move a by 0.04–0.17 — PASS.
- p. 59 — condition number 2,000 → 62; 84% — PASS.
- p. 59 — median understated by a factor of about three (w = 3.19), range 1.9–4.0 — PASS.
- p. 59 — three quarters beyond the largest budget — PASS.
- p. 59 — 1/γ ≈ 5.6 under the refit — (1/0.1785 = 5.60) — PASS.

**Tally.** About 420 numbers checked. About 395 PASS. 11 FAIL or inconsistent (the fails flagged above, plus the "0.14B" and "16–24 GB" slips). About 14 UNTRACED, mostly appendix-sourced or design-based figures.

---

## Appendix B. Citation audit (the most important attributions)

| # | Where | Attribution in text | Check | Verdict |
|---|---|---|---|---|
| 1 | p. 1 | Sevilla and Roldán (2024): frontier training compute grows 4–5×/yr | Epoch report title and notes | Correct |
| 2 | p. 1 | Cottier et al. (2024): amortized cost ≈2.4×/yr since 2016 | arXiv 2405.21015; notes (90% CI 2.0–2.9) | Correct |
| 3 | pp. 1, 7 | Kaplan et al. (2020): C ≈ 6ND; joint law exponents | Kaplan Table 2 (0.076, 0.103) | Correct |
| 4 | pp. 2, 10 | Hoffmann et al. (2022): Approaches 1–3; a between 0.46 and 0.50; Huber δ = 10^-3 | Notes on Hoffmann (A1 0.50, A2 0.49, A3 0.46) | Correct |
| 5 | pp. 2, 16 | Besiroglu et al. (2024): replication, SEs 0.02 / 125 / 1,293, dropping 5 runs | Notes (124.58; 1,293.23) | Correct |
| 6 | p. 2 | Ackerberg et al. (2015): functional dependence | Standard | Correct |
| 7 | pp. 2, 20 | Diamond, McFadden and Rodriguez (1978): non-identification of σ and bias | Standard | Correct |
| 8 | pp. 3, 5, 11 | Hao and Merrill (2026): derive Chinchilla σ, Leontief profit model; 0.762 by averaging exponents | arXiv 2605.16430 HTML: Leontief q = min{…}; σ ≈ 0.7622 from averaged α = 0.31205 | Correct |
| 9 | p. 5, 54 | Mertens et al. (2026): developer-FE compute efficiency; within-developer 90/10 = 41 | Notes: Eq. (1); 41× on MMLU-Pro | Correct |
| 10 | pp. 5, 22, 57 | Whitfill (2025): confounding; Ho's β 0.04 versus 0.37 ⇒ ninefold | Notes: Theorem 1; "overstated by around a factor of nine" | Correct |
| 11 | pp. 5, 22 | König et al. (2026): observational confounding | arXiv 2606.05029 abstract: "observational studies face confounding and effect heterogeneity" | Correct |
| 12 | pp. 5, 17 | Kricheli et al. (2026): fixed-TPP designs ill-conditioned | arXiv 2605.08541 abstract | Correct |
| 13 | pp. 5, 31 | Czech et al. (2026): finite-grid bias of IsoFLOP parabolas | Notes | Correct |
| 14 | pp. 5, 38 | Pearce and Song (2024): embedding counting and small scale; simulated 0.78 (Epoch) and 0.74 (Chinchilla) | arXiv 2406.12907 HTML | Correct |
| 15 | pp. 3, 37–38 | Porian et al. (2024): head FLOPs, warmup, decay, tuning; published exponents | Notes and Table 5 published column | Correct |
| 16 | pp. 4, 23, 43 | Sardana et al. (2024): 6ND + 2NT; up to 10,000 tokens/param; typical-ratio fits overestimate token gains at extreme ratios | arXiv 2401.00448 abstract, verbatim | Correct |
| 17 | pp. 4, 13, 41 | De Loecker and Warzynski (2012): markup = elasticity / revenue share | Standard | Correct (and the paper correctly denies that w is a markup) |
| 18 | pp. 4, 58 | Gundlach et al. (2025): ≈10× from Kaplan→Chinchilla rebalancing | Notes: "about 10× at the frontier, with a closed-form CEG" | Correct fact. **Framing** issue: M8(b) |
| 19 | p. 3 | Nerlove (1963): "spurious small-firm scale economies" | Nerlove estimated genuine size-dependent returns to scale | **Mischaracterized**: M8(a) |
| 20 | p. 59 | Trammell and Korinek (2023): scaling laws imply compute and data are gross complements | Notes: footnote 10, quoted | Correct |
| 21 | pp. 59–60 | Erdil et al. (2025, GATE): effective compute as a sufficient statistic; train–inference trade-off calibrated loosely | Notes | Correct |
| 22 | p. 39 | Bjorck et al. (2025): LR* elasticity −0.32 holding batch fixed | arXiv 2409.19913 HTML: β = 0.32 for ≥760M models, 0.40–0.70 for smaller; batch 0.5M fixed | Correct, but incomplete: M8(c) |
| 23 | p. 39 | Step Law (Li et al. 2025a): positive LR elasticity in D, batch adjusting | m8 memo (published +0.307) | Correct |
| 24 | pp. 24, 32 | Farseer (Li et al. 2025b): near-factorial grid, Eq. 3 nine-parameter form | arXiv 2506.10972: about 1,000 models trained | Correct, but 404 is a subset: minor 19 |
| 25 | pp. 23, 36 | Magnusson et al. (2025): 25 recipes × 14 sizes × 3 seeds | Notes and data | Correct |
| 26 | p. 45 | Dey et al. (2023) 20 tokens/param; Touvron et al. (2023) inference budgets; Grattafiori et al. (2024) beyond compute-optimal; Biderman et al. (2023) 300B tokens for all sizes | Standard | Correct |
| 27 | p. 54 | Syverson (2011): US manufacturing 90/10 ≈ 1.92 | Standard (Syverson 2004 via the JEL survey) | Correct. Comparison framing: M11 |
| 28 | p. 35 | Chirinko (2008), Klump et al. (2007), Oberfield and Raval (2021): 0.4–0.7; Raval (2019) lower | SYNTHESIS.md table | Correct |
| 29 | p. 22 | Klepper and Leamer (1984): forward and reverse regressions bound the truth under classical error | Standard | Correct |
| 30 | p. 31 | Hausman and Wise (1977): truncated-normal MLE | Standard | Correct |

**Omission:** Bi et al. (2024, DeepSeek LLM) on data quality shifting the allocation exponent; see M8(d).

**Typesetting:** 37 entries lack a venue or locator; see M9(a).

**Key-year mismatches.** Several keys say 2024 but render as 2025: `gadre2024language`, `bhagia2024establishing`, `choshen2024hitchhikers`, `bjorck2024scaling`, `brandfonbrener2024loss`. This is harmless for readers, but it confuses the replication package's citation registry.

---

## Appendix C. Internal-consistency register

| # | Quantity or item | Place 1 | Place 2 | Resolution |
|---|---|---|---|---|
| C1 | Chinchilla compute range and budgets | p. 28 and Fig. 2 notes: 9 budgets, 6×10^18–3×10^21 | p. 23 and p. 41: up to 1.3×10^22 | Data: 1.4×10^18–1.3×10^22; 137/245 on profiles (M1) |
| C2 | σ* range under κ = 1 | p. 3: 0.73–0.83, SE 0.006–0.034 | p. 32: 0.735–0.828, SE 0.005–0.034; p. 35: 0.735–0.825 | One sweep set (M3) |
| C3 | a and γ ranges | p. 3 and p. 35 exclude DataDecide | Table 4 includes DataDecide (0.315; 0.090) | State the sample |
| C4 | Chinchilla σ*_κ | Table 3: 0.700 (0.015) | Table 4: 0.701 (0.014) | Harmonize |
| C5 | Profile set versus point | p. 32: [0.67, 0.69] (Gaussian) | p. 31 and Table 3: 0.700 (Huber) | Label the objective (M5) |
| C6 | Farseer in-sample RMSE (κ = 1) | p. 32: 0.0085 | Table 4: 0.0087 | NLS versus Huber |
| C7 | Llama 3 8B inputs | p. 13: M ≈ 1,875, w ≈ 5.2, 12.7, 5.5× | p. 43 and Table 6: 1,868, 5.27, 12.8, 5.6× | Name the technology and N |
| C8 | Gopher w | p. 14: 0.36 | p. 46: 0.37 | Name the technology |
| C9 | Best dominating support point | p. 23: 0.76 | p. 43: ≤ 0.77 | 0.766 |
| C10 | Figure 1(b) members | Fig. 1: σ* = 0.60, 0.74, 0.85 | p. 16: 0.89, 0.74, 0.33 | Align |
| C11 | LaLonde bias bound | p. 4: about 20% | p. 53: ±23% | 23% |
| C12 | "Six alternative" versus "six common" technologies | p. 4 | p. 41 | Six technologies including the reference |
| C13 | Reliability of ln C | p. 53: ≥ 0.988 | Table 8 EIV rows: 0.895 and 0.514 | M7 |
| C14 | Technologies "estimated in Section IV" | §VI.C: nine; Table 9: twelve | §IV reports eight rows; Marin and (Mis)Fitting absent | M4(d) |
| C15 | Clustering recommendation | Conclusion: cluster by budget | §V headline uses pairs | M2 |
| C16 | κ = 1 | Rejected (§IV) | Used for the headline wedge (§V) | M2 |
| C17 | φ interval versus T_C range | p. 55 [−0.39, 3.37] | T_C 5–38 from [−0.25, 3.25] | M4(f) |
| C18 | Warm versus multistart coverage | 67% (R = 500) | 92% (R = 60, where warm-start = 57%) | M4(e) |
| C19 | Module versus paper Table 3 SEs of M* | `m1_chinchilla_horse_race.tex` (9.0)[25.6] | Paper (6.9)[17.2], recomputed correct | Regenerate the module table |

**Notation clashes (must fix):**

- d = ln D versus width d (§III.B).
- θ has four meanings (Eq. 5, §VI.A, Prop. 2(iv), Prop. 7).
- φ has three meanings (Eq. 5, Prop. 4, §VI.B).
- τ has three meanings (Eq. 7, §V.E, Table 3).
- η has two meanings (Prop. 6, §V.E).
- s has three meanings (MC error s.d., s_M, omitted share in Eq. 5).
- T (inference tokens) versus T_C (doubling time).
- β (data exponent) versus β₂ (Adam, Table 5).
- κ versus q (Fig. 4 legend).
- a (allocation exponent) versus a₁ (inner exponent) is fine but should be stated once in §I.

**Cross-references.** All `\ref` and `\cite` commands resolve (no undefined references in `_build/main.log`). The main-text to appendix map is correct: I checked each "Online Appendix A, Lemma/Proposition X" against the appendix statements (Lemma A1–A5, Propositions A1–A9, Corollaries A1–A5). The figure numbering shifts because Figure 5 is the m9 placeholder, so the wedge figure is Figure 6 and so on; the text uses `\ref` throughout, so nothing breaks.

---

## Appendix D. "Do not claim" list (paper_plan.md §3)

| # | Forbidden claim | Found? | Where / comment |
|---|---|---|---|
| D1 | "first to interpret scaling laws as production functions" | No | p. 5 disclaims it explicitly. |
| D2 | "first to compute σ" | No | p. 5 disclaims it; credits Hao and Merrill (pp. 5, 11). |
| D3 | "first to note confounding" | No | p. 5 disclaims it. |
| D4 | "σ is more stable than a" | **Implicitly, yes** | p. 35 explicitly disavows it ("We do not read these contrasts as evidence that σ is intrinsically better determined than a"). But: (i) Intro p. 3, "By contrast, the allocation exponent a … ranges from 0.37 to 0.57 … The object every allocation decision needs is the one the data pin down least", juxtaposed with σ*'s range; (ii) §IV opening p. 27, "the objects that allocation decisions need, a and M*, are the least portable"; (iii) p. 35, "σ* … and γ … vary by less than a factor of two", which is true of a as well (M3(c)). Rephrase all three in terms of M*, which does vary 18-fold, or in units of sampling error. |
| D5 | "on-path data cannot sign the bias" | No | Prop. 5(i) states the sign is identified. |
| D6 | "constant CEG iff equal exponents" | No | Prop. 5(iv) correctly says "iff they share E and γ". |
| D7 | "w is a markup" | No | Denied three times (pp. 4, 13, 41). |
| D8 | revealed T levels measured precisely | **Partially** | The abstract (p. 1) and Conclusion (p. 59) quote the point level ("about 2.2 times") without the range. §I.E (p. 14) gives "roughly 1.9×10^14 tokens" with no band. The Intro (p. 4) and §V (pp. 43, 49–50) do give ranges and caveats. Add ranges to the abstract and §I.E. Note also M2: the point level itself depends on κ = 1. |
| D9 | IO remedies work on public cross-lab data | No | pp. 4 and 53 state the opposite. |
| — | (Plan §2 II.C) P4 "constant CEG iff equal exponents" from SYNTHESIS | No | Not repeated. |

---

## Appendix E. Recomputations performed

All scripts are ad hoc and read only the project's CSV and `.npy` files. No project file was modified.

1. **Sample B statistics.** From `m3_wedge_models.csv`, filtered to `sample == 'B'` and `code == False`: n = 173, 68 families, median ŵ, shares, bands, technology medians, Spearman correlations, interval ratios, and shares above 1.29 and 1.80. All matched the text.
2. **κ-free Chinchilla wedge.** ln ŵ = ln(a₁/b₁) + ln A − ln B − a₁ ln N + b₁ ln D, with the κ-free Huber parameters from `m1_chinchilla_spec_kappa.csv`: median 3.78, share > 1 93.1%, Llama 3 8B 6.75. The same formula with the κ = 1 reference parameters reproduces the CSV median (3.186).
3. **IQR/1.349 SEs and percentile CIs of M*(5.76×10^23).** From `data/processed/m1_chinchilla/boot_chinchilla_n240_*_{pairs,cluster}.npy`: Huber 6.92 [17.18], CI [7.8, 35.1] and [3.2, 125.8]; Gaussian 4.95 [12.05]; NLS 3.84 [9.23]. These match Table 3; the module .tex does not (minor 43).
4. **Profile-likelihood sets.** From `m1_chinchilla_profile_sigma.csv`: full design, Gaussian, maximum LR 424.5, set [0.67, 0.69]; on-path band maximum LR 1.96 with the minimum at the grid edge (0.50); the Huber profile is all NaN.
5. **Information slopes.** From `m7_theory_fisher_information.csv`: log–log slopes 4.00 (I_S) and 2.00 (I_lnM*).
6. **Table 9 arithmetic.** T_C = 12 ln 2/(g_N + g_D) for every row; the Hicks-neutral rows satisfy αg_N = βg_D; shares = ln(gain)/ln(8.9, 9.9, 8.2).
7. **Chinchilla raw compute range and profile membership.** From `data/raw/epoch_chinchilla/svg_extracted_data.csv` and `m1_chinchilla_design_variance.csv`.
