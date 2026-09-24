# Round-2 Referee Report — Referee 1

**Manuscript:** "What Optimizing Labs Reveal: Scaling Laws as Production Functions" (Okar and Claude), revised version (v2), submitted to the *American Economic Review*
**Referee expertise:** empirical IO; production-function estimation (proxy variables and control functions, GNR, markups and wedges, biased technical change)
**Version reviewed:** compiled `main.pdf` (175 pp.; main text pp. 1–42, Online Appendices A–F pp. 54–175), the LaTeX sources, the revision plan, and the replication outputs cited below. Page numbers refer to the compiled PDF.

**The authors' controlled experiment is still running.** Section III.E and its eight "TBD-m9" placeholders are unfilled. I comment on its pre-analysis plan (committed at `bd5c0ad`), on the draft power memo (`output/memos/m9_sweeps.md`) and on how the paper describes the experiment. I do not comment on results.

**My own calculations.** Some comments rest on numbers I computed from the authors' replication files. Each one names its source file, and I label them "my calculation". They are rough checks, not replacements for the authors' estimates.

---

## 1. What the revision does

The revision is substantial and, on most of my round-1 points, responsive. Five changes stand out.

1. **A model-free estimator of σ\*.** Proposition 4 (pp. 16–17) implements the result I suggested: the curvature of the IsoFLOP profile divided by twice the frontier slope. It is applied to every public IsoFLOP design (Table 1, p. 20; Figure 3, p. 22), and the local wedge is used to test parametric extrapolation inside Farseer (Figure 6, p. 32). This is the best part of the revision. It also changes the technology section's headline: σ\* ≈ 0.70 now comes from three laboratories' annealed designs, not from one sweep.
2. **A re-specified inversion.**
   - Proposition 2 (pp. 8–10; Proposition A8, pp. 72–75) now covers any objective V(L, N).
   - The estimand is renamed: "the value of compactness", with "planned serving expenditure" confined to developers that serve their own models.
   - Family token budgets get their own first-order condition (Proposition A9).
   - Table F1 derives what the wedge identifies under eleven objectives.
3. **Technology uncertainty treated honestly.**
   - An ex-ante set of 32 technologies (Table F3).
   - A κ-free reference technology.
   - Partial identification of M\*(C) at frontier scale (Proposition 5, pp. 17–18; Table F7).
   - The cardinal token count removed from the abstract.
4. **The observational material moved to Online Appendix E.** The claims are corrected: "low-power failure to reject", validation across populations in the sense of Todd–Wolpin rather than LaLonde, OLMo-2 excluded from the strict benchmark, and the truncated allocative measure.
5. **The IO analogies tightened throughout.**

I thank the authors for a careful revision. The paper is now much closer to a publishable contribution.

## 2. Overall assessment

The technology half of the paper is now in good shape. Its identification argument is sharp:
- on-path data are uninformative about curvature;
- designed IsoFLOP variation identifies it without a functional form.

The empirical answer is transparent and replicable: σ\* ≈ 0.70 on annealed IsoFLOP designs, with the κ = 1 restriction distorting σ\* where the data can tell.

The revealed-demand half has improved as much, but it now has a different problem. **The paper's own new theory undercuts several of its headline uses of the data.** There are four instances.

1. **Proposition 2(iv) and A9.** Member wedges in families with a common token budget are *not* revealed preferences, and with menu-chosen sizes they carry no information. Yet:
   - the paper's sharpest example (Llama 3 8B) comes from such a family;
   - so do 10 of the 16 models in Table 2, Panel A;
   - so do 21 of the 33 models behind the "planned serving expenditure" reading (New Major Comment 1).
2. **Scale dependence of σ\*.** Section III reports that σ\* falls with compute within the two designs that reach the highest budgets. The conclusion nevertheless recommends σ\* ≈ 0.7 "for frontier-scale language models" (New Major Comment 2).
3. **What the sign result assumes.** The sign-identification result ("86 percent under weak assumptions") excludes cross-lab heterogeneity in M\*, the very factor-bias channel the paper documents in Section III.D. It is also unweighted, while the headline trend is compute-weighted (New Major Comment 3).
4. **The abstract.** The abstract's "most of lifetime cost … up from near zero before 2023" rests on a level the body calls non-robust. Its baseline is dominated by OPT-175B, a research suite that the clean sample excludes as uninformative (New Major Comment 4).

These are fixable. Most need reporting and reframing, not new data. Two items do need new work:
- a second-output (Bond et al.) test, which is feasible with data the authors already have;
- the experiment's results, which I cannot evaluate yet. As the draft power memo shows, the paper's description of the experiment's inference is already out of date.

**Recommendation: major revision.** At the top-journal bar I could not recommend acceptance with the pre-registered experiment unreported. Nor could I with the internal inconsistencies in Section IV. I expect to be able to support the paper if the requests in Sections 4 and 6 are met.

---

## 3. Status of round-1 major comments

| # | Round-1 comment | Status |
|---|---|---|
| 1 | Conduct model for the inversion | **Partially resolved** |
| 2 | Size tiers and family token budgets | **Partially resolved** |
| 3 | DLW analogy and its critiques | **Partially resolved** |
| 4 | M\*(C) at frontier scale; selective band | **Largely resolved** |
| 5 | Cost side is not an identity | **Resolved** |
| 6 | σ ≈ 0.7 from one sweep; heterogeneity | **Largely resolved** (new issue: drift with compute, see New 2) |
| 7 | Model-free σ\* and w | **Resolved** |
| 8 | Precision of IO analogies; novelty of formal results | **Resolved** (one residual on Prop. 3(ii)) |
| 9 | Inference | **Largely resolved** |
| 10 | Observational section | **Resolved** |
| 11 | Controlled experiment (design) | **Partially resolved**; results pending |
| 12 | Scope and focus | **Largely resolved** |

**Comment 1 (conduct). Partially resolved.**

*What is done.*
- Proposition A8 is exactly the generalization I asked for. It handles an objective V(L, N) with direct size effects, which covers my points 1(b) and 1(c). It also covers:
  - adoption-driven open release, case (a′);
  - tiers, latency, data costs and distillation;
  - size-dependent FLOP prices.
- The estimand is renamed, and "planned serving expenditure" is confined to developers that serve their models.
- Section IV.E (pp. 33–34) and Appendix F7 run conduct tests.

*What remains.*
- The paper concedes that the tests do not discriminate (p. 34).
- The one test with power is the open-weight premium: +0.52 log points, p = 0.011. It points *away* from serving-cost internalization. A developer bearing all of its serving cost over-trains less, not more.
- The paper states this implication in one sentence, then still reads s as planned serving expenditure for the serving group. As New Major Comment 1 shows, that group is almost entirely made up of models for which the paper's own Proposition 2(iv) or 2(iii)(b) rules out that reading.
- Durability (my 1(d)) is still not modeled (Appendix F10). This matters for the trend interpretation.
- I no longer insist on instrument-based conduct tests; with 18 developer clusters they are not feasible. What is needed instead is to apply the paper's own theory consistently (New 1).

**Comment 2 (tiers and family budgets). Partially resolved.**

*What is done.*
- Proposition 2(iv) and Proposition A9 derive the family first-order condition.
- Table F9 reports family-level shares.
- Research suites with D fixed by design are dropped.
- Common-D and size-specific families are reported separately.
- The bunching estimate is now proper: 51 percent against 13 percent, with an excess mass of 83 models.

*What remains.*
- **(a) Member wedges still used as revealed preferences** (New 1).
- **(b) The tier test is uninformative by construction.** Under one technology, ln ŵ is a deterministic function of (N, C). The authors say so in the Table F8 notes, and the text concludes that the tier model's prediction "is not visible." Given bunching this strong, the right response is the one the paper's own Proposition A8(iii)(b) prescribes: for tier-bunched models, w − 1 bounds the value of compactness from above.
  - That is harmless for the "value of compactness" reading, because a tier shadow price *is* a value of compactness.
  - It is not harmless for the "planned serving expenditure" reading.

**Comment 3 (DLW critiques). Partially resolved.**

*What is done.*
- The wedge is framed as Raval's ratio statistic with FLOP-accounting weights, not as a markup.
- Remark A8 has a paragraph on why experimental elasticities avoid the proxy circularity, at the cost of external validity.
- De Ridder, Grassi and Morzenti, and Doraszelski and Jaumandreu, are cited.
- Lab-own technologies are used where they exist.

*What remains.*
- **(a) No second-margin or second-output test.** Appendix F10 lists the Raval-type test as "not done." A second-*output* version is feasible with data already in the paper (New 5).
- **(b) Cross-lab factor bias is still not bounded.**
  - It is represented only by the spread over 32 technologies.
  - That spread is excluded from the sets (PI-1 to PI-3) behind the headline sign result (New 3).
  - Lab-own technologies address recipe bias for 22 models. They do not address the output-concept critique at all, because every lab-own technology is itself a validation-loss technology.

**Comment 4 (M\* at frontier scale). Largely resolved.**

*What is done.*
- The ex-ante set includes every sweep I listed: all three Gadre corpora, OLMo, Muennighoff, and κ-free versions. It also includes the lab laws (Llama 3, DeepSeek, MiniCPM).
- The reference is κ-free.
- M\*(C) at frontier scale is now the object of partial identification (Proposition 5), as I requested in 4(e).
- The union band is reported: only 18 percent of clean models have the whole band above one (p. 30), which is commendably frank.
- The cardinal T is gone from the abstract.
- Table 2 shows the band next to the single-technology interval.

*What remains.*
- The abstract still makes a level claim (New 4).
- The PI anchors omit recipe tilt (New 3).
- The Llama 3 anchor sits at an unbracketed budget (Minor 7).

**Comment 5 (cost side). Resolved.**
- "Identity" is replaced by "FLOP-accounting approximation."
- The δ and φ sensitivity is in Table F10, and s is shown not to depend on p.
- Data costs are in the objective.
- The Bond–Söderbom point about cost-shifter heterogeneity is made on p. 11.

**Comment 6 (σ heterogeneity). Largely resolved.**

*What is done.*
- Homogeneity is now tested (Table 1, Panel C).
- The claim is confined to annealed IsoFLOP designs, and small-sweep κ-free values are called design-specific.
- On Farseer, like objects are compared: the local-path first-derivative estimate is 0.708 and the κ-free estimate 0.710.
- "Gross complements" is gone.
- The tokens-processed versus unique-data distinction is now handled correctly in Section V.B.
- Growth models are told to use γ, not σ\* (p. 40).

*New issue.* The within-design drift of σ\* with compute is statistically strong. It is inconsistent with the constant-σ\* technology used in equation (8), and it undermines the conclusion's "frontier-scale" recommendation (New 2).

**Comment 7 (model-free estimator). Resolved.**
- Estimation of σ\* on all IsoFLOP designs, with finite-grid bias analysis (Table D7).
- The in-support local wedge compared with parametric extrapolation (Figure 6).
- The linearity test of ln w in ln(M/M\*): rejected, since ln w is convex.

These are exactly what I asked for, and well executed. Residual points are in Minor 4–6.

**Comment 8 (analogies and novelty). Resolved.**
- Proposition 3 is positioned against Marschak–Andrews, Bond–Söderbom, GNR and Box–Lucas/Kiefer–Wolfowitz, and distinguished from ACF's functional dependence.
- IsoFLOP sweeps are "designed variation, not price shocks."
- The GNR normalization is gone.
- The DMR sign is corrected (βg_D − αg_N) and demoted to a methodological remark.
- Nerlove is reworded.
- The TFPR analogy is gone.
- LaLonde is replaced by Todd–Wolpin / Hotz–Imbens–Mortimer.
- Goldberger is cited for truncation.

One residual: Proposition 3(ii) now contains a rate claim that the paper says is not proved (Minor 3).

**Comment 9 (inference). Largely resolved.**
- A design-conditional wild bootstrap with Feng–He–Hu weights.
- A joint bootstrap for sample statistics, my 9(f).
- Bootstrap-calibrated profile-likelihood sets.
- Three few-cluster p-values in Appendix E, with the most conservative reported.
- Table 8's inconsistency is gone.

Residuals are Farseer's standard errors (Minor 5) and the experiment's inference (New 6).

**Comment 10 (observational). Resolved.** Everything I asked for is done, and the material now sits in Appendix E, where it belongs.

**Comment 11 (experiment design). Partially resolved.**

*Adopted:*
- a pre-analysis plan committed before estimation, with the pre-commitment state disclosed honestly (p. 92);
- a power calculation done before estimation;
- both parameter conventions;
- learning-rate corner checks on both corpora;
- seed replicates on both corpora;
- a within-trunk covariance bootstrap;
- WikiText-103 as a neutral set;
- a model-free local wedge for the high-M test.

*Not adopted:*
- the dose–response design;
- WikiText evaluation of the main-grid endpoints (only "most added runs," p. 92).

See New 6.

**Comment 12 (scope). Largely resolved.**
- The main text is down to about 14,800 words.
- Section VI and the peripheral material are in the appendices.

The Online Appendix, however, is now 122 pages (pp. 54–175). Appendix E (22 pages) is still a companion paper inside this one.

---

## 4. New major comments

### New 1. The paper's own family-budget and tier results contradict how Section IV uses member-level wedges, including the "planned serving expenditure" reading

Proposition 2(iv) (p. 10) and Proposition A9 (pp. 75–77) establish three things for a family that trains every size on one token budget:
1. Member wedges differ only through the technology: ln w_i − ln w_j = −α(ln N_i − ln N_j).
2. Only the π-weighted average w̄_H − 1 is a revealed-preference object.
3. If the sizes come from a menu, even that average carries no information about the members' values of compactness beyond an upper bound (A9(iv)).

The paper draws the right conclusion in words (p. 11: "member-level wedges are not revealed preferences") but not in its exhibits or headline claims.

**(a) The sharpest example is a member wedge.**
- Section IV.C (p. 30) presents Llama 3 8B (ŵ = 8.4, s = 0.88) as "the sharpest case." It concludes that "Meta served Llama 3 … so this is a planned serving share" and that "the smallest model [was trained] as if for serving."
- Llama 3 8B, 70B and 3.1 405B form the "Llama 3 herd," a common-budget family in Table F9 (p. 173).
- Under Proposition A9 the herd reveals one number: a family share of **0.29**, dominated by the flagship.
- The 8B's large wedge is fixed by the technology given the herd's D. The within-Meta contrast between the 8B "trained for serving" and the 405B "trained for training efficiency" is exactly the inference Proposition 2(iv) says cannot be drawn.
- Section I.D previews the example (p. 12), and Table 2 lists it.
- **The flagship's own sign is not identified at frontier scale** (Llama 3.1 405B is among the 11 unidentified models, p. 33). So for Meta's herd, neither the sign nor the level of the family's value of compactness is identified.

**(b) Table 2 (p. 31) mixes the two kinds of wedge without flagging them.** Of its 16 Panel A models, 10 belong to common-budget families:
- Llama 3 8B, Llama 3 70B and Llama 3.1 405B;
- Llama 2 70B;
- Qwen2.5 7B and 72B;
- Qwen3 0.6B and 14B;
- DeepSeek LLM 7B and 67B.

Their member wedges are listed beside the others with no indication that they are not revealed preferences.

**(c) The serving group rests almost entirely on uninformative or upper-bound wedges.** The paper reads s as planned serving expenditure for the 33 clean models whose developers serve (median s = 0.80, p. 34). My tabulation from `output/tables/ra2_wedge_models.csv` (clean == True, serve == 1), using the five tier windows of Table F8:
- **21 of the 33** belong to common-budget families.
- **24 of the 33** have sizes inside the tier windows. There, by Proposition A8(iii)(b), w − 1 only bounds m_N from above.
- **Only 3 of the 33** (Qwen2-0.5B, Qwen2-1.5B and phi-1.5) are neither.
- **19 of the 33 are Alibaba Qwen models.** For Qwen2.5 and Qwen3 the documented D is a single corpus size stated for every member (18T and 36T). Appendix F10 (p. 174) concedes that these have "not been audited against the tokens each model processed."

The planned-serving reading therefore has almost no support in the sample to which it is applied.

**(d) The serving indicator is coded at the developer level** (Table F8 notes; p. 170). λ is a property of the model, not of the firm. Gemma models, for example, are coded as served because Google serves Gemini. Whether Alibaba serves the 0.5B–3B Qwen models through its API, or Meta served the 8B in Meta AI, should be documented model by model.

**Requests.**
1. In Table 2 and wherever member wedges appear:
   - flag common-budget members;
   - report the family-level share (Table F9) as the revealed-preference object;
   - drop the Llama 3 8B "planned serving share" interpretation, or recast it as a statement about the herd (0.29, with the flagship's unidentified sign).
2. Report the revealed-preference headline on the sample where Proposition 2(i) actually applies: size-specific-budget families and singletons, with tier-window models flagged. Give the family-level shares for common-budget families. By my count (same file), models that are neither in common-budget families nor in tier windows number 19. Their median reference s is 0.85, so the ordinal conclusion may survive. The point is to report the object the theory licenses.
3. Restate the serving-group result as an upper bound for tier-window models. Recode serving at the model level. Show the result without Alibaba, or with Alibaba's D audited.
4. State plainly in Section IV.E that the only conduct test with power, the open–closed premium, is evidence against serving-cost internalization as the main driver of over-training among open-weight developers. The adoption and tier readings (cases (a′) and (b)) are the ones the data favor.

### New 2. σ\* is measured at 10^19–10^21 FLOP and falls with compute where it can be checked. "About 0.7 for frontier-scale language models" is not supported, and equation (8)'s constant scale is rejected

**What the paper reports** (p. 23; Appendix D, p. 119): within designs, σ\* falls with compute:
- by 0.072 per decade in Chinchilla (s.e. 0.026);
- by 0.053 per decade in Llama 3 (s.e. 0.010).

**How significant the drift is.** From `output/tables/ra1_modelfree_isoflop_summary.csv`:
- The drift p-values are 0.009 and 0.001.
- The budget-level estimates are heterogeneous *within* these designs: Llama 3 has Q = 50.7 with I² = 0.86, and Chinchilla has a bootstrap p = 0.025.
- Marin shows no drift, but spans only 3×10^18 to 3×10^20.
- The two IsoFLOP designs that reach 10^21 FLOP are the two that drift.
- Farseer's local path drifts in the same direction, though weakly (−0.015 per decade, s.e. 0.008).

**A rough pooled estimate (my calculation).** I ran a weighted regression of the budget-level σ\*_b in `ra1_modelfree_isoflop_budgets.csv` on log₁₀ C with design effects, excluding Porian:
- The drift is about **−0.05 per decade**. The naive standard error is 0.009 and ignores within-design dependence.
- The designs' precision-weighted centers lie at 10^19.2–10^20.3 FLOP.
- So the headline 0.695 is σ\* at roughly 10^20 FLOP.

**Why this matters.**
1. **The recommendation to economists** (p. 41) — "for substitution between parameters and data in frontier-scale language models, an elasticity of about 0.7" — is an extrapolation of three to five decades. The paper refuses to make that extrapolation for M\*, and it deserves the same treatment for σ\*. Linear extrapolation of the drift is not credible (σ\* must stay in (0, 1)), but it gives a sense of scale: it would put σ\* near 0.45–0.55 at 10^24 FLOP.
2. **The scale of the inversion.** Equation (8) uses a constant 1/σ\* − 1. A drift of this size would roughly double ln ŵ for frontier-compute models. The magnitude bounds of Proposition 5(iv) use 1/σ\* − 1 ∈ [0.40, 0.52] (p. 33; Table F7). That range reflects in-design, cross-design variation only, and would have to widen substantially.
3. **The functional form.** Together with the convexity of ln w in ln(M/M\*) inside Farseer (p. 32), the drift means that two of the κ family's restrictions fail where they can be checked, and both failures bear on the inversion. The quasi-homotheticity of Lemma A4 fails because σ\* varies along the path. The linearity of Proposition 4(iii) fails because ln w is convex in ln(M/M\*).

**A second, smaller issue: the parameter convention.** The model-free σ\* is defined for the parameter count that makes each IsoFLOP profile an exact isocost, N = C/(6D) for Llama 3 and Marin.
- In Marin's configuration count it falls to 0.64–0.66 (p. 21; Appendix D, p. 118).
- The authors' own power memo shows that the non-embedding and total conventions differ by 0.04–0.08 *by construction*, because a Hicks elasticity is not invariant to a nonlinear re-measurement of one input.

The confidence interval [0.673, 0.717] excludes both sources of uncertainty. It also treats Marin's three corpora, which share one lab, one code base and one FLOP accounting, as independent designs. With one estimate per study (k = 4) and the Hartung–Knapp–Sidik–Jonkman interval, I get roughly [0.66, 0.73] (my calculation, from the Table 1 standard errors).

**Requests.**
1. Report a meta-regression of budget-level σ\* on log compute with design random effects. Make σ\*(C) beyond 10^21–10^22 FLOP an object of partial identification, as M\*(C) already is.
2. Propagate the range into the magnitude bounds of Proposition 5(iv) and into the level of s.
3. Qualify the abstract, the introduction (p. 3) and the conclusion: σ\* ≈ 0.70 at 10^19–10^21 FLOP, in FLOP-equivalent parameter units, with evidence of a decline at higher compute.
4. Report the study-level summary, with Marin as one study, as the headline interval, and add the convention range.

### New 3. The sign-identification result excludes cross-lab heterogeneity in M\*, and is unweighted while the trend it is paired with is compute-weighted

**What the result assumes.** "Over-training … identified at frontier scale under weak assumptions for 86 percent of the clean sample" (pp. 4, 33, 35, 41) is Proposition 5 under PI-1 (Table F7, p. 169). PI-1 assumes two things:
1. M\* equals one of five anchors (Chinchilla, Llama 3, three Marin corpora and DeepSeek's law) at their largest budgets.
2. The path elasticity over the next two to four decades stays within the range of in-design paths and published laws, [−0.16, 0.19].

**Why the assumptions are not weak.**
- **Assumption 1 rules out developer-specific tilts of A/B.** Proposition 1(iii) and Section III.D (p. 25) show that such tilts move M\*:
  - the DataDecide tilt moves M\* 2.9–3.4-fold;
  - DeepSeek's allocation exponent moves with data quality;
  - MiniCPM's M\* is 192 at 10^21 FLOP.

  That is precisely the Demirer–Raval channel. When all technologies are admitted as anchors (PI-4), the share falls to 23 percent.
- **Normal inputs alone identify far less.** Panel C of Table F7 shows that normal inputs alone identify the sign for only about a third of models, in unharmonized units.

A sign statement robust to the heterogeneity the paper documents needs an explicit allowance for it.

**How much the result moves (my calculation).** I took `data/processed/ra2_wedge/pi_bounds_models.csv` (PI-1). I then required ln M to exceed the upper bound of the identified set by an additional tilt allowance τ in ln M\*.

| M\*-factor allowance | Rationale | Share with w > 1 identified |
|---|---|---|
| 1 (the paper) | — | 86 percent |
| 1.3 | tokenizer differences | 84 percent |
| 1.84 | DataDecide tilt of 0.26 under the reference exponents | 78 percent |
| 3.4 | DataDecide's own M\* factor | 69 percent |
| 3.4 × 1.3 | both | 61 percent |

The ordinal conclusion — most small and mid-size open models are over-trained — survives. The number 86 should not be presented as resting on "weak assumptions."

**Weighting (my calculation).** The share is unweighted, but the trend headline (abstract; p. 35; Table 3, Panel C) is compute-weighted.
- Weighted by training compute, only **48 percent** of the clean sample's compute is in models whose sign is identified under PI-1.
- By year, the compute-weighted share with the sign identified is:

| Year | Compute share with sign identified |
|---|---|
| 2023 | 13 percent |
| 2024 | 40 percent |
| 2025 | 100 percent |

- So the compute-weighted aggregate in 2023–2024 is dominated by flagships whose sign is not identified: Falcon 180B, Llama 2 70B, Llama 3.1 405B and Qwen-72B. Llama 3.1 405B alone carries 55 percent of the clean sample's 2024 compute (p. 174).
- Part of the "rise" in the compute-weighted share from 2023 to 2025 is therefore a shift in composition: from flagships with unidentified signs to smaller models with identified ones.

**Requests.**
1. Add a PI set with an explicit bound on developer-specific tilt, taken from the paper's own cross-recipe evidence, and a harmonization allowance for tokenizers. Report the identified share as a function of the allowance.
2. Report the normal-inputs-only share on the harmonized clean sample in the main text.
3. Report the compute-weighted identified share next to the compute-weighted trend.
4. Drop "weak assumptions."

### New 4. The abstract, introduction and conclusion make level and trend claims the body calls non-robust

1. **"Accounted for most of lifetime cost"** (abstract) is a level claim, s > 0.5.
   - The body says the level "is not robust" (p. 4). The median share ranges from 0.17 to 0.85 across the ex-ante technologies. In 2025 the aggregate ranges from 0.24 to 0.96 (Table 3, Panel C).
   - It holds under the class of technologies that let the data choose the curvature, and for the median lower bound under PI-1 (0.59).
   - It does not survive the tilt allowance of New 3. With the same curvature range, the median lower bound on s falls to about 0.47 at an M\* factor of 1.84 and to about 0.32 at 3.4 (my calculation, same file).
2. **"For developers that serve them, lower serving cost"** (abstract) is not supported, for the reasons in New 1.
3. **"Up from near zero before 2023."**
   - The 2019–2022 value, 0.04, comes from eight models in the open-weight universe (Table 3, Panel C).
   - OPT-175B and GLM-130B carry 92 percent of that period's compute (p. 35). OPT-175B alone carries 46 percent, per `ra3_econ_inference_share.csv`.
   - OPT is a research suite whose D was fixed by design. The paper's own cleaning rule (b1) excludes it from the clean sample as uninformative about compactness.
   - The unweighted median s of the eight pre-2023 models is 0.22 (same file).
   - The clean-sample trend starts in 2023, at 0.28 [0.19, 0.40], with a technology range of 0.01–0.60.
   - The trend claim should be stated on one sample with one set of rules. It should be reported unweighted and at the family level (Proposition A9) as well as compute-weighted, and with leave-one-developer-out checks. There are only 13 clean models in 2025.
4. **Terminology.** "Planned" is used for all open-weight models in several places, contrary to the paper's own terminology:
   - the x-axis label of Figure 5 (p. 29), "planned serving share of lifetime cost";
   - p. 40, "the 2025 planned multiple, w − 1 = 4.5".

**Request.** A suggested abstract sentence:

> "Inverting first-order conditions with this technology, most open-weight models released since 2023 are trained on more tokens per parameter than training-cost minimization implies — robustly in sign, conditionally in size — and the implied value of compactness has risen steeply."

Put levels in the body, with their conditioning technology.

### New 5. The output-concept critique (Bond et al. 2021) can be tested with data already in the paper

- Proposition A8(ii) nests the critique: ŵ = w · w_L/w_{L′} when the developer optimizes an output L′ different from the econometrician's L.
- The paper's two defenses do not address it:
  - lab-own technologies are all validation-loss technologies;
  - the 32-technology spread varies recipes, not outputs.
- The ordinality result (Lemma A1) covers only monotone transformations of the *same* output.
- The OLMo ladder is fitted on C4 loss and on task bits per byte (σ\*_κ = 0.544 and 0.51; p. 24). That gives two technologies on the *same runs* with *different outputs*.

**Request.**
1. Compute ŵ for the clean sample under both OLMo-ladder technologies.
2. Report the distribution of ln(w_{task}/w_{C4}). This is a direct, if single-recipe, estimate of the output-concept contamination.
3. If Bhagia et al.'s task-specific losses allow it, report it by task family.

A Raval-type second *input* margin (MoE sparsity; KV heads) remains desirable. The second-output test is cheap and would do more for credibility than any of the current validation exercises.

### New 6. The experiment: reporting must match the authors' own pre-estimation findings

I cannot assess results, but three things about how the paper describes the experiment need attention now.

**(a) The inference described in the paper is superseded by the authors' own power analysis.** The main text (p. 26) and Appendix B4 (p. 92) state the plan's robustness standard: the wild cluster bootstrap by width plus the seed-covariance bootstrap. The draft power memo (`output/memos/m9_sweeps.md`, §0.2–0.3) was computed before estimation. It finds that the plan's wild bootstrap on unrestricted residuals with eight width clusters is badly anti-conservative:

| Statistic | Finding |
|---|---|
| Size of the tilt test | 0.20 |
| Coverage of the tilt interval | 0.68–0.80 |
| Coverage of the model-free σ\* interval | 0.40–0.45 |
| Pr("factor-biased" \| no tilt), under the two-set decision rule | 8–17 percent |

The memo pre-declares a remedy, "Deviation D8": CR2-adjusted residuals and a restricted wild bootstrap for the tilt, with design calibration factors for the model-free statistics.

This is exemplary practice. But D8 post-dates the compiled paper, which still lists "one deviation, made for time." The paper must:
- report the power and size results (for example as an Appendix B4 table);
- declare D8, with its timing relative to estimation;
- apply the stricter robustness standard in III.E.

**(b) The Q3 null is not zero.** The memo shows the extrapolation slope's null value is design-specific. It is −0.007 in the non-embedding convention without noise and drifts with noise. In the total convention it is +0.11 or −0.08 to −0.10, even without noise. The plan says the sign of the slope "answers" the question. Reading it against a design-specific null is a substantive change to a pre-registered decision rule. It should be declared as a deviation (I agree with it), and its derivation reported.

**(c) Two cheap additions, labelled as not pre-registered.**
- **WikiText for the main grid.** The neutrality rule's third leg (WikiText-103) is available only for "most added runs" (p. 92). If main-grid endpoint checkpoints exist, evaluating them on WikiText costs minutes and makes the rule applicable to the grid on which the tilt is estimated.
- **A direct test of the Porian explanation.** The constant-learning-rate trunks provide unannealed losses at every branch point. Computing the model-free σ\* on the trunks' pre-cooldown losses against the annealed endpoints, on the same grid, directly tests the explanation for excluding Porian et al. (pp. 21–22): annealed versus unannealed at similar scale. Without it, the exclusion remains post hoc.

**(d) Match conventions when comparing with 0.70.** Compare the experiment's σ\* with Table 1 in the same convention, N = C/(6D) with C the actual training FLOPs. The memo shows that the non-embedding and total conventions differ by construction.

---

## 5. Minor comments

**Front matter and introduction**

1. **Abstract and p. 3.** Add the compute range and the parameter convention to "about 0.7" (New 2). "No detectable heterogeneity across designs" (p. 3) should be accompanied by the within-design drift, which is detectable.
2. **Replication statement** (p. 1 footnote). "Available from the authors" does not meet the AEA Data and Code Availability Policy, which requires a deposit in the AEA repository at acceptance. Several inputs cannot be redistributed: Epoch's Chinchilla digitization, Farseer, Step Law and the digitized Llama 3 points. The editor and the Data Editor will need a plan for these.

**Sections I–II**

3. **Proposition 3(ii)** (p. 13) states that estimators of σ\* "should not be expected to shrink faster than the inverse fourth root." The text then says the rate "rests on … simulations and a heuristic argument, not on a limit theorem" (pp. 13–14). A proposition should contain only what is proved.
   - One option is to move the rate to a remark.
   - The better option is to prove it. The on-path criterion is of fourth order along one direction because the regression function's first derivative vanishes there. That is the setting of Rotnitzky et al. (2000) with a zero score in one direction, which gives n^{−1/4} under their non-degeneracy condition on the second-order term. Verifying that condition for the κ = 1 family should be feasible.
4. **Proposition 5(ii)**, "sharp identified set." Given only an anchor and e ∈ [e_L, e_U], sharpness is immediate. Say sharp with respect to which information, or drop the word. Part (i) (|e| ≤ 1 under normal inputs) is a one-line observation and could be a remark.

**Section III**

5. **Farseer standard errors.** In Table 1, Farseer's κ-free standard error of 0.003 and the first-derivative model-free standard error of 0.001 (Appendix D, p. 119) "treat the grid points as the design and exclude smoothing bias." The Hessian-based estimate moves between 0.66 and 0.73 with the bandwidth. These are not confidence intervals for σ\*.
   - Please say which estimate the (0.003) in Table 1 belongs to.
   - Report bias-aware or bandwidth-robust intervals, or present the bandwidth range as the uncertainty.
   - This was round-1 Minor 29.
6. **Coverage under dependent noise.** The model-free estimator's Monte Carlo coverage (92–98 percent, Appendix D) assumes independent noise across runs. The authors' m9 power memo shows how quickly coverage collapses with shared components. Two sources of shared error exist in the public designs:
   - digitization error, which is correlated along a plotted curve (Chinchilla, Llama 3);
   - Chinchilla's off-profile runs, which are truncations of trunks.

   Report coverage under a plausible dependent-noise design, for example a per-budget or per-size random effect.
7. **Llama 3 anchor and path.**
   - Llama 3's model-free σ\* uses eight bracketed budgets up to 10^21 FLOP.
   - The PI anchor is taken at 10^22 FLOP, a budget whose minimum is not bracketed (Table F5 notes, p. 166; Table F7, p. 169).
   - The lab-own path uses all ten budgets, including the two unbracketed ones, justified by "the law Meta planned with." That logic calls for Meta's *believed* curvature as well, which is not the model-free one.
   - Anchor at the largest bracketed budget and state the choice. The Chinchilla anchor drives the PI-1 upper bound at high compute, so the headline should not move much.
8. **Table 1, Panel A.** The Llama 3 row says ten budgets up to 10^22 (p. 19), but the model-free estimate uses eight up to 10^21. State the budgets used in each column.
9. **p. 23**, the worked example ("73 percent more compute … at σ\* = 0.70"). State that it assumes α = β and the κ family's linear ln w, which Section IV rejects inside Farseer.

**Section IV**

10. **"Defined ex ante, before any wedge was computed"** (p. 28; Appendix F3). Version 1 reported wedges under many of the same technologies (Chinchilla, OLMo, Gadre, Farseer, Meta). Say that the rule was fixed before the version-2 computations, and list which technologies were added after round 1.
11. **Within-technology sensitivity.** On the reference technology, keeping Chinchilla's five highest-loss runs raises the median wedge from 3.99 to 5.50 (p. 161), which is more than the κ = 1 contrast reported in the main text (3.34). Report it on p. 29.
12. **Clarify the unanimity statement.** "Every technology puts the point estimate above one for 64 percent of models" (p. 30; p. 159) reads as "each technology, 64 percent." Say "all 32 technologies agree on w > 1 for 64 percent."
13. **Vintage (round-1 Minor 37).** The trend applies 2022–2024 technologies to 2023–2025 models. By Proposition 1(iii):
    - data-augmenting progress lowers M\* over time, which would steepen the true trend;
    - parameter-augmenting architecture changes would flatten it.

    Lab-own contemporaneous technologies cover only 11 models. Say what the sign of vintage bias is under the paper's own evidence (DeepSeek's data-quality result).
14. **Validation sample selection.** The verified sample starts from the ObsScaling / Open LLM Leaderboard file, which selects on popularity (round-1 Minor 23). This matters for the Hugging Face derivative regressions (Table F8, Panel B), whose outcome is itself a popularity measure. Discuss it, or restrict to models not selected through the leaderboard.
15. **Byte normalization of D** (round-1 Minor 21) is still not done. Appendix F4 (p. 163) bounds its effect at about 8 percent of the wedge under the reference, which is reassuring. Tokenizer differences also enter the sign result through M\* (New 3), so include them in the PI allowance.
16. **Qwen token counts.** The corpus-size D stated for every Qwen2.5 and Qwen3 member (Appendix F10) matters more than the text suggests: Qwen models are 19 of the 33 serving-group models and four of Table 2's rows. Audit it, or flag these models in every table.

**Section V**

17. **Length.** Section V is informative but long relative to its identification content. V.C's two fleet adjustments (p. 40) could go to an appendix, since the comparison is "loose, not a validation." V.D is useful and short; keep it.
18. **Repetition model.** The data-wall numbers rest on Muennighoff et al.'s repetition parameters, estimated at ≤ 9B parameters (acknowledged on p. 41). State in V.B that R\*_D is the object driving the headline, and report sensitivity to it next to the σ\* sensitivity. Table 3, Panel B already has the ingredients.

**Online Appendix and presentation**

19. **Figure 5 axis label** (p. 29): "planned serving share of lifetime cost" should be "expenditure share s (value of compactness)." The same slip appears in Figure F1's note.
20. **Exhibit count.** The main text has ten exhibits, one a placeholder. Merging Figure 4 into Figure 3 once the experiment is in, as the integration log suggests, would help.
21. **Online Appendix length.** It runs to 122 pages. Appendix E (22 pages) is the natural candidate for a companion paper. If it stays, trim Tables E2–E9 to what Section II needs.
22. **Authorship.** Listing an AI system as a co-author remains an editorial question under the AEA's and COPE's policies (round-1 Minor 6). I note it again only so that it is resolved before acceptance.
23. **Pre-analysis plan header.** The header time (03:15) differs from the commit time (03:12). This is disclosed (p. 92); fixing the header is trivial. Also state in B4 when the power calculation was run (15:52–16:44 on 24 September, per the memo) relative to the first estimation.

---

## 6. What I would need to see in the next version

In priority order:

1. **Consistent use of Proposition 2(iv)/A9 and A8(iii)(b)** (New 1):
   - family-level shares for common-budget families;
   - tier-window models flagged as upper bounds;
   - the serving-group reading restated or dropped;
   - serving coded at the model level;
   - the Llama 3 8B example recast.
2. **σ\*(C)** (New 2): a meta-regression across budgets; σ\* at frontier compute treated as partially identified; wedge magnitude bounds widened accordingly; the conclusion's recommendation qualified by compute range and parameter convention.
3. **A sign result robust to cross-lab tilt and tokenizer differences** (New 3), with the normal-inputs-only share and the compute-weighted identified share reported.
4. **Abstract, introduction and conclusion aligned with the body** (New 4): no level claim, a consistent-sample trend, and "planned" used only where the theory licenses it.
5. **The second-output (Bond et al.) test** from the OLMo ladder (New 5).
6. **The experiment's results**, reported under the pre-registered procedures and D8, with the Q3 null values and D8 declared as deviations. Add WikiText for the main grid and the trunk-versus-endpoint σ\* comparison as labelled exploratory analyses (New 6).
7. **The minor comments above**, especially 3, 5, 7, 10, 11 and 16.

With these, the paper would make three contributions:
- a clean identification result;
- a transparent, model-free estimate of the technology's curvature over the range where designs exist;
- an honest, largely ordinal reading of what open-weight developers' choices reveal.

That is a contribution I would be glad to see in the *AER*.
