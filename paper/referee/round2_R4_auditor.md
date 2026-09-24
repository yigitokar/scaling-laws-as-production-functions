# Round-2 Referee Report 4 (Replication Auditor)

**Manuscript:** "What Optimizing Labs Reveal: Scaling Laws as Production Functions" (Okar and Claude), revised version 2, `paper/main.pdf` compiled 2026-09-24 17:02 (175 pp.; main text pp. 1–42, references pp. 42–53, Online Appendix pp. 54–175).

**Round-1 report:** `paper/referee/R4_auditor.md` (major revision).

**Mandate.** As in round 1: audit the main-text numbers and the main-text tables against the module outputs (`output/tables/*.csv`) and memos (`output/memos/*.md`); check the key citations; check internal consistency; and check the "Do not claim" list (now `paper/notes/paper_plan_v2.md` §3). The controlled experiment (m9, the red TBD-m9 blocks) is still running, so I comment on its design and its pre-registration record, not on results.

**Method.**
- I extracted the PDF with PyMuPDF and read the main text in full. I read the Online Appendix sections that the main text cites for its numbers.
- I traced every numerical statement in the main text and in Tables 1–3 to its CSV or memo. I recomputed wherever a figure is a summary:
  1. The random-effects σ* summary from ra1's inputs: DerSimonian–Laird plus HKSJ. It reproduces 0.695 [0.673, 0.717] and Q = 4.1 (p = 0.54). I also ran it under alternative inputs (R2-M4).
  2. The 2019–2022 open-weight "universe", rebuilt with ra2's own `sample.universe()` code. It reproduces the published shares 0.039, 0.348, 0.535 and 0.819, and it identifies the eight models behind the early value (R2-M1).
  3. The compute-weighted shares under each of the 32 ex-ante technologies, from `ra3_econ_inference_share_by_tech.csv` (R2-M2).
  4. Every cell of Table 2, against `ra2_wedge_models.csv`.
  5. The off-path losses at 5·M* for the three equivalent-family members in Figure 1(b). By direct minimization I get 2.05, 4.13 and 7.56 percent.
  6. The φ profile of Appendix E.
- I did **not** open the m9 estimation outputs that now exist in `output/tables/m9_sweeps_q*.csv`. The paper does not report them yet, and my design comments should not be conditioned on them.

Page numbers are PDF pages. "PASS" means the value matches its source after rounding.

---

## 1. Summary of the revision

The paper has been refocused around one idea: optimization is double-edged for measurement. It hides the technology's curvature and reveals the developer's objective.

- **Theory.** There are six formal results: Proposition 1, Lemma 1 and Propositions 2–5.
  - Proposition 2 generalizes the wedge to any objective V(L, N) and reports the expenditure share s = (w−1)/w.
  - Proposition 3 gives information that is fourth order in allocation errors.
  - Proposition 4 is new. It gives a model-free σ* equal to IsoFLOP curvature divided by twice the frontier slope.
  - Proposition 5 is new. It gives partial identification of M*(C) and of the sign of w − 1.
- **Technology.** The model-free σ* on the annealed IsoFLOP designs of three developers plus Farseer is 0.695 [0.673, 0.717]. The κ-free parametric fits agree on the large designs. Chinchilla's κ = 1 overstates σ* on Llama 3.
- **Revealed demand.** The authors invert first-order conditions on an ex-ante "clean inference-demand sample" of 77 models, with the κ-free Chinchilla reference. Under the reference the median s is 0.75. Across 32 ex-ante technologies it ranges from 0.17 to 0.85. The sign of over-training is identified for 86 percent of the sample. The paper also adds conduct tests and a trend in the compute-weighted share.
- **Economics.** A new Section V covers data demand, the data wall and serving shares.
- **What moved to the Online Appendix.** The observational/LaLonde material is now Appendix E. The Porian decomposition, duality and selection results are in Appendix D.
- **The experiment.** It is described and pre-registered in §III.E and Appendix B4.

## 2. Overall assessment

The revision fixes almost everything the round-1 audit raised, and it does so carefully.

- The Chinchilla design is now described correctly.
- The headline no longer rests on κ = 1 or on pairs resampling.
- The failed verbal claims are corrected.
- The profile-likelihood sets now come from the same objective as the point estimates.
- The bibliography renders a venue for all 127 entries. I parsed `main.bbl` and every entry has one.
- The two reliabilities of ln C are reconciled.

**Numerical accuracy is again high.** I checked about 480 numerical entries in the main text and Tables 1–3. Every entry I could trace matches its source after rounding, and I found no transcription or arithmetic errors. About 15 entries I did not trace to a file:
- the run counts and N/M ranges of Table 1 for Marin and Porian, which the table's source comment says are recomputed from loaders;
- a few cells of Table 3, Panel B;
- the 2.0/4.1/7.6 percent in §II.A, which I recomputed but which is in no output file.

The remaining problems are of a different kind. They concern claims whose robustness class, as the paper itself defines "robust", is misstated, and pieces of the evidence record that the authors' own internal reviews flagged but did not close:

1. **The abstract's trend baseline.** "Up from near zero before 2023" rests on eight unverified models. Two of them carry 92 percent of the compute, and one belongs to a suite that the paper's own ex-ante rule excludes. The paper elsewhere attributes the w < 1 of that era to Kaplan-law beliefs, not to a zero value of compactness (R2-M1).
2. **The abstract's level claim.** "Most of lifetime cost" for releases since 2024 fails under 16 of the 32 ex-ante technologies. That includes Meta's own published law and all three of Marin's model-free technologies (R2-M2).
3. **The partial-identification numbers.** The identified set [2.3, 89], Meta's [10.1, 57], and the lab-own Meta intervals rest on anchor intervals that the authors' own theory review found about nine times too narrow in log width. Appendix F reports both widths for the same Llama 3 anchor without reconciling them (R2-M3).
4. **The headline σ* "about 0.7" with interval [0.67, 0.72].** It is conditional on a parameter-counting convention and a kernel bandwidth. Under the alternatives the paper itself reports, the random-effects summary is 0.660 [0.638, 0.682] (R2-M4).
5. **The pre-registration record.** The paper says the one deviation "changes no estimator or decision rule". The authors' m9 memo records a later deviation, D8, that redefines the Q2 decision rule and the robustness standard. D8 and the power calculation are not under version control, and the power calculation shows that the pre-registered inference is badly anti-conservative. None of this is in the paper yet (R2-M5).

None of these requires new data. Items 1, 2 and 4 are rewording plus tables that the authors already have. Item 3 is a re-run of existing code. Item 5 is disclosure, and it must be settled before the m9 results are written up.

---

## 3. Status of round-1 major comments

| # | Round-1 comment | Status | What remains |
|---|---|---|---|
| M1 | Chinchilla design misdescribed; nine-cluster assignment of off-profile runs | **Resolved** | §III.A (p. 19) now reads "245 runs, 1.4×10¹⁸–1.3×10²², 137 on nine profiles, 108 from 35 trunks". The alternative cluster schemes were re-run (ra1 H5): with budgets plus trunks, the p-value of κ = 1 is 0.004, and the text reports this with the original scheme's 0.052 (p. 23). The Conclusion no longer recommends budget clustering. Residual: Table 1's "Chinchilla, IsoFLOP runs 137" row counts the five high-loss runs that fall outside every window (minor 3). |
| M2 | Headline wedge under κ = 1 with pairs bootstrap | **Largely resolved** | The reference is now κ-free Chinchilla with a design-conditional wild bootstrap, and the κ = 1 value is reported (3.99 → 3.34; p. 29). Remaining: the headline interval uses the narrowest of ra1's six schemes (R2-M3). Also, §V.D and Table 3 lead with the rejected κ = 1 refit (minor 10). |
| M3 | "σ ≈ 0.7" was one sweep; inconsistent sweep sets | **Resolved in substance** | Model-free σ* now spans six design estimates from three developers plus Farseer. Sweep counting is consistent ("seven sweep–corpus technologies from five studies; DataDecide separately"). The "factor of two" contrast is gone, and p. 24 disavows the D4 reading. New issue: the headline depends on the counting convention and the bandwidth (R2-M4). §V's "σ*, the best-measured object" (p. 35) re-approaches D4 (minor 11). |
| M4 | Six verbal claims failing against CSVs | **Resolved** | (a) Appendix D (p. 112): 0.95–1.06 for ten of twelve rules, 0.78 and 1.16 for the other two. (b) "Within 10⁻³" (Appendix C). (c) "Close to the closed form … 1.28" (Appendix C). (d) Table E-allocative lists all 14 technologies with provenance, including (Mis)Fitting labelled as the m1 registry. (e) Coverage 33 → 78 percent like for like (Appendix C); the main text reports 78 percent only. (f) The φ interval [−0.44, 3.37] with τ_C 5.3–39.6 months; I verified this against `ra4_obsfix_phi_profile.csv`. |
| M5 | Profile set excluded its point estimate; boundary minimum | **Resolved** | The full-design set is now Huber/Laplace, bootstrap-calibrated at [0.68, 0.72], and contains 0.701. The on-path grid now spans [0.05, 0.99] with LR relative to the unrestricted optimum; the set [0.20, 0.99] is reported as hitting the grid edge (p. 14). The Monte Carlo Panel B was redone the same way. |
| M6 | Selective specification tests; bootstrap p-values at their floor | **Partially resolved** | (a) The Gaussian CES LR is now calibrated (bootstrap p = 0.17; HC1/HC3 0.060/0.089; Appendix D). (b) Rank-one results are in the appendix. (c) Appendix D (p. 122) now says "at the resolution of the bootstrap", but B was not increased. The main text still says "every restriction is rejected" for DataDecide (p. 25), which rests on B = 19 (floor 0.05) (minor 12). |
| M7 | Two reliabilities of ln C | **Resolved** | Appendix E (p. 143) now uses a hardware-time bound (≥ 0.98 pooled, 0.955 within families) and explicitly withdraws the earlier 0.858 estimate. |
| M8 | Citations: Nerlove; Gundlach et al.; Bjorck et al.; DeepSeek | **Largely resolved** | Nerlove is now correctly "returns to scale … found to vary with scale" (p. 7). Appendix E (p. 153) credits Gundlach et al.'s definition, but its opening paragraph still frames the tenfold gain as "not a gain realized between eras" (minor 17). Bjorck is given only as "less against their steeper per-size estimates", with no number (minor 18). DeepSeek LLM is cited for the data-quality tilt (p. 25); I verified 0.450 → 0.524 → 0.578 against arXiv 2401.02954. |
| M9 | Bibliography; abstract length; replication location; authorship | **Mostly resolved** | (a) All 127 entries render a venue. (b) The abstract is 99 words. (c) "Available from the authors" is not a public archive, and the working tree that builds the PDF is uncommitted (minor 20). (d) The authorship line is unchanged, with a disclosure footnote. This is for the editor; I take no position. |
| M10 | m9 design | **Design largely resolved; new disclosure issues** | Addressed: (a) both parameter conventions, with actual-FLOP compute in convention P; (b) four-layer high-M runs; (c) a learning-rate calibration on FineWeb plus learning-rate corners over the D ladder; (d) pre-specified inference by width plus a secondary bootstrap; (e) seeds in both corpora; (f) a WikiText-103 evaluation and bits per byte; (g) a committed pre-analysis plan. Only width 128/256 was added at high M rather than 320–384, but the four-layer floor answers the depth-2 concern. New issues are in R2-M5. |
| M11 | TFP dispersion depends on units | **Resolved** | Appendix E (p. 146) reports output, reducible-loss and compute-equivalent units, and concludes that "the comparison with manufacturing is … a choice of cardinalization, not a finding." |

**Round-1 minors.** Nearly all are resolved or moot because the text moved. Examples:
- width is no longer d;
- the Figure 1(b) members now match the text: 0.60/0.74/0.85 with 7.6/4.1/2.0 percent;
- the Llama 3 8B and Gopher numbers name their technology;
- "404 of about 1,000" Farseer runs;
- the κ/q legend;
- the Chinchilla σ*_κ is 0.701 everywhere.

Two remain open:
- minor 43, the stale module table `m1_chinchilla_horse_race.tex` (minor 20 below);
- minor 36/26, the p-value resolution in the main text (minor 12 below).

---

## 4. New major comments

### R2-M1. The trend baseline "up from near zero before 2023" is not a measurement of the value of compactness, and it violates the paper's own sample rule (abstract p. 1; Intro p. 3; §IV.F p. 35; §V.C p. 37; §VI p. 40)

**The claim.** The abstract ends "…accounted for most of lifetime cost, up from near zero before 2023." The Introduction (p. 3) says: "the compute-weighted share under the reference rose from 0.04 in 2019–2022 to 0.82 in 2025, and it rises under every technology we consider." §IV.F lists "the steep rise between 2022 and 2025" as one of the section's two robust findings.

**What the audit finds.**

- **(a) The baseline is eight unverified rows, and two carry 92 percent of the weight.** I rebuilt the 2019–2022 universe with ra2's code (`sample.universe()` plus `open_closed()` filters) and reproduced s = 0.039. Its eight models are all Sample-A Epoch rows, none from the verified sample:

  | Model | Year | M | ŵ (reference) | Share of period compute |
  |---|---|---|---|---|
  | OPT-175B | 2022 | 1.7 | 0.34 | 46% |
  | GLM-130B | 2022 | 3.1 | 0.43 | 46% |
  | NeMo Megatron GPT-20B | 2022 | 17 | 0.90 | 6% |
  | BLOOM-1.7B | 2022 | 208 | 2.59 | 0.5% |
  | Grover-Mega, AraGPT2-Mega, EMDR, CPM-Ant | 2019–2022 | 3–1,580 | 0.44–6.1 | ~1.5% |

- **(b) The universe does not apply the paper's own exclusion rule to these rows.**
  - Step (b1) drops "research suites and replications whose D was fixed by design, such as Pythia, OPT and BLOOM" (p. 28). The clean-sample exclusions are applied only to *verified* rows (Appendix F, p. ~168; `sample.py::universe`).
  - OPT-175B and BLOOM-1.7B are therefore members of suites whose verified siblings (OPT-1.3B through 66B; BLOOM-560M through 176B) are excluded as not being choices of the kind Proposition 2 describes.
  - Dropping the two leaves six models and s = 0.058 (my recomputation). The baseline then becomes, in effect, GLM-130B alone, which carries about 85 percent of the remaining compute.
- **(c) Truncation at w = 1 turns under-training into "zero value of compactness", contrary to the paper's own reading of that era.**
  - Both heavy models have ŵ < 1.
  - The paper attributes wedges below one in this period to optimization against Kaplan et al.'s law. See §I.D on Gopher (p. 11), Proposition 2(iii) (ŵ = w·w_L/w_L′), and Appendix E (p. 153): "For 2020–21 models … a belief in the Kaplan law is the more plausible reading."
  - A developer with a positive value of compactness that planned with Kaplan's law would show ŵ < 1. The pre-2023 value is therefore evidence about beliefs, not about the absence of serving demand.
- **(d) Unweighted, the early period is not near zero.** The median model's s in 2019–2022 is 0.22 (`ra3_econ_inference_share.csv`, `median_model_s_ref`), and the three 2019–2021 models have s = 0.55, 0.69 and 0.84 (`ra2_wedge_aggregate.csv`).
- **(e) "It rises under every technology" is established only from 2023 to 2025 on the clean sample.** I confirmed that 32 of 32 technologies are monotone over 2023 < 2024 < 2025. No technology range exists for 2019–2022: the `tech_min` and `tech_max` columns are empty for that row. The sentence juxtaposes the unverified 0.04 with a robustness statement that does not cover it.
- **(f) The external benchmark the paper cites contradicts the baseline in the same window.** Patterson et al. (2022) put inference at about three-fifths of Google's ML energy in 2019–2021 (§V.C, p. 37). That is exactly when the planned share is 0.04.

**Requested.**
1. Remove "up from near zero before 2023" from the abstract, or restate the trend in its robust form. For example: "and this share rose steeply between 2023 and 2025 under every technology we consider."
2. Apply the clean-sample exclusions to unverified suite members, or report the sensitivity.
3. For every period, report the unweighted median alongside the compute-weighted share, and the range across the 32 technologies.
4. State in §IV.F and §V.C that the pre-2023 value reflects under-training attributed to Kaplan-era beliefs.
5. Make the same edits on p. 3, p. 35 ("between 2022 and 2025") and p. 40.

### R2-M2. The abstract's level claim ("most of lifetime cost" for releases since 2024) fails the paper's own robustness standard (abstract p. 1; cf. p. 4)

**The inconsistency.**
- The Introduction's "What is robust" paragraph (p. 4) says "The level of s is not robust … the median share ranges from 0.17 to 0.85 across the technologies we specify ex ante … We treat s as ordinal and conditional on a technology."
- The revision plan (§B) requires that "cardinal numbers [appear] only with their conditioning technology."
- Yet the abstract states a level ("most", i.e. s > 0.5) without naming the technology. "With this technology" in the abstract refers to σ* ≈ 0.7. But the level also depends on the zero point M*(C), which "no public design observes" (p. 35).

**The numbers.** From `ra3_econ_inference_share_by_tech.csv`, clean sample, compute-weighted:
- **2024 (below 0.5 under 16 of 32 technologies).** These include:
  - Meta's own published law (`meta_a2`): 0.32;
  - all three of Marin's model-free technologies, the curvature-free benchmark the paper favours: 0.38, 0.45 and 0.46;
  - Farseer κ = 1: 0.49;
  - Hoffmann: 0.27;
  - OLMo κ = 1: 0.06;
  - MiniCPM: 0.05.
- **2025 (below 0.5 under 5 of 32).** These are the three Marin Approach-2 laws, OLMo κ = 1 and MiniCPM.
- **Reference technology.** The 2024 universe interval is [0.41, 0.64] (Table 3, Panel C), which itself includes values below one half.
- **Non-serving developers.** For the 44 of 77 clean models whose developers do not serve (p. 34), s is not a share of lifetime cost at all.

**Requested.** Either condition the sentence explicitly ("under our reference technology, … most of lifetime cost"), or replace the level with the two claims the paper classifies as robust: the sign of over-training for 86 percent, and the rise from 2023 to 2025 under every technology.

### R2-M3. The partial-identification numbers rest on anchor intervals that the authors' own theory review found too narrow (§IV.D p. 33; Table 2 p. 31; Appendix F pp. 168–169)

**Where the numbers come from.** The identified set for M*(10²⁴), [2.3, 89], Meta's own set [10.1, 57], and the sign-identified shares (86/77/23 percent) use ra2's anchor intervals (`ra2_wedge_pi_anchors.csv`): Chinchilla [18.2, 29.4] and Llama 3 [20.8, 23.7].

**The internal review.** The independent review of module ra5 (`ra5_theory_review.md`, item 7) finds that these intervals bootstrap only the within-profile parabola residuals.
- The profile minima scatter around the fitted path with s.d. 0.27 (Llama 3) and 0.30 (Chinchilla). That is about five times the within-profile noise.
- ra5's wild bootstrap-t interval for the *same* Llama 3 anchor (same point, ln M* = 3.1057) is [12.2, 40.2]: a log-width of 1.19 against 0.13.
- The integration log (§4 and open issue 3) records an approximate re-run: the share stays at 86 percent, but the set widens to roughly [1.5, 110]. This was not implemented.

**Specific problems.**
- **(a) Appendix F reports both widths without reconciliation.** Table F7 and its notes give 22.3 [20.8, 23.7] in F6 (p. 168) and 22.3 [12.2, 40.2] in the Panel C notes (p. 169). They reconcile only the Chinchilla difference (22.7 against 20.5, attributed to eight versus nine budgets).
- **(b) The lab-own intervals likely share the problem.** The same path bootstrap (`ra2_wedge/modelfree.wild_boot`) produces the lab-own Meta and Marin intervals: Llama 3 8B 8.4 [5.7, 12.8] (p. 30) and Llama 3.1 405B 0.97 [0.88, 1.08] (`ra2_wedge_labown.csv`).
- **(c) The Llama 3 anchor sits on an unbracketed budget.** It is placed at the 10²² budget, which ra1 finds unbracketed. On ra1's eight bracketed budgets the path gives M* ≈ 15 (ra2 memo H3).
- **(d) The headline interval uses the narrowest bootstrap scheme.** The reference technology's own intervals (median s 0.75 [0.69, 0.79]; the brackets in Table 2) use the Feng–He–Hu wild scheme. In ra1 H5 that is the narrowest of six schemes: M*(5.76×10²³) is [10.5, 37.4], against [5.2, 106] under the budget-plus-trunk cluster scheme that ra1 recommends as "the conservative check". A back-of-envelope mapping: a factor of two in M* moves the median s between about 0.66 and 0.81.

**Requested.**
1. Re-run `pi_anchors`, and the lab-own path draws, with between-budget residuals (wild bootstrap-t, as in ra5), or with both sources of error.
2. Update §IV.D, the lab-own figures in §IV.C and Table 2, and Appendix F (F6/F7), and reconcile the two anchor intervals.
3. Report the headline median s under the budget-plus-trunk cluster scheme beside the wild interval.

### R2-M4. The headline σ* (about 0.7, 95 percent CI [0.67, 0.72]) is conditional on the counting convention and the bandwidth, and the CI omits a systematic error the paper itself quantifies (abstract; pp. 3, 20–21, 41)

**What the estimator requires.** Proposition 4 needs profiles that are exact isocosts. For Llama 3 and Marin, Table 1 therefore defines N = C/(6D). The paper discloses the consequences:
- in Marin's configuration count, the estimates fall to 0.64–0.66 (p. 21);
- Chinchilla's and Meta's FLOP accounting "adds uncertainty of about ±0.04" (p. 21);
- the Farseer input to Panel C (the Hessian-based 0.703) is 0.664 at the cross-validated bandwidth (p. 21).

**My recomputation** (DL/HKSJ, with ra1's inputs and standard errors):

| Inputs | RE mean [95% CI] | τ | Q (p) |
|---|---|---|---|
| As in Table 1, Panel C | 0.695 [0.673, 0.717] | 0.000 | 4.1 (0.54) |
| Marin in configuration N (0.658/0.642/0.655)* | 0.671 [0.642, 0.700] | 0.016 | 7.7 (0.18) |
| Farseer at the CV bandwidth (0.664) | 0.679 [0.656, 0.703] | 0.007 | 5.5 (0.36) |
| Both | **0.660 [0.638, 0.682]** | 0.000 | 0.8 (0.98) |

\*The configuration-N standard errors are set equal to the FLOP-implied ones, which is an approximation.

Under both alternatives the interval excludes 0.70. The consequence for the headline is not negligible. At σ* = 0.65 the reference median wedge is 5.7 and s = 0.82, against 4.0 and 0.75 at 0.70 (`ra2_wedge_sigma_sensitivity.csv`).

A related inconsistency: Table 1 enters Farseer's model-free estimate as the first-derivative 0.708, with no standard error, while Panel C and Figure 3 use the Hessian-based 0.703.

**Requested.**
1. Add the alternative rows to Table 1, Panel C.
2. Either widen the stated interval to include the ±0.04 FLOP-accounting band, or state the headline as "0.66–0.70 depending on the parameter-counting convention."
3. Use one Farseer estimator throughout. Give it a standard error that does not treat the eight compute levels as independent (ra1 open item 9).
4. State once, in §III.A, that σ* is an elasticity between FLOP-effective parameters and tokens.
5. Say in the abstract that the designs are annealed ("annealed IsoFLOP experiments"). Porian's unannealed 0.51 is excluded on that basis, and that exclusion is a hypothesis that the authors' own experiment is meant to test.

### R2-M5. Pre-registration record and disclosure for the controlled experiment (§III.E p. 26; Appendix B4 p. 92; `m9_preanalysis_plan.md`; `output/memos/m9_sweeps.md`)

The design responses to round-1 M10 are good. So is the decision to compute the power and size of the pre-registered inference before estimating anything. The paper's account of the record, however, does not match the authors' own memo.

- **(a) "One deviation … changes no estimator or decision rule" (p. 26; Appendix B4 p. 92) is contradicted by Deviation D8.** The m9 memo (§0.3) records D8, adopted after the plan was committed and after the FineWeb-Edu grid had finished, but before estimation. D8:
  1. adds a CR2 companion bootstrap to every table;
  2. redefines "excludes zero" in the Q2 decision rule as a restricted wild bootstrap p < 0.05 on CR2 residuals, and "neutral" as p ≥ 0.05 plus a cr2 half-width below 0.10 on both validation sets;
  3. raises the robustness standard from two schemes to three;
  4. inflates the model-free σ* and Q3 intervals by design calibration factors of 1.4–1.7 before a statement may be made.

  These changes alter the operational decision rule and the inference. List them as deviations, with their dates and reasons, in §III.E and Appendix B4.

- **(b) D8 and the power calculation are not under version control.**
  - `output/memos/m9_sweeps.md`, `code/analysis/m9_sweeps/` and the power and coverage CSVs are untracked in git.
  - `data/processed/m9_sweeps/estimation_start.txt` records the first estimation run at 17:12 on 24 September.
  - The only timestamped commitment is the plan itself (commit `bd5c0ad`, 03:12).

  Commit the power outputs, the coverage check and the D8 text now, with their file times, or use an external timestamp. Report every result under the plan's own scheme, labelled as such, next to the D8 schemes.

- **(c) The power study's findings govern how Q1–Q3 can be read, and they belong in the paper.**
  - Under the plan's wild cluster bootstrap with 8 width clusters:
    - coverage is 0.72–0.78;
    - the single-set size of the tilt test is 0.22–0.28;
    - a false "factor-biased" verdict occurs 8–17 percent of the time under χ = 0.
  - No scheme reaches nominal coverage for the model-free σ* (at most 0.70) or for the Q3 slope (at most 0.75) on the 8×6 grid.
  - The null value of the Q3 slope is not zero. It is −0.02 to −0.06 in convention P with noise, and +0.11 or −0.08 to −0.10 in convention T even without noise.

  Yet the TBD text says that the slope's "sign is the object" (p. 26), and the Introduction's placeholder promises "the sign of the Chinchilla form's extrapolation error" (p. 3). The design-specific null values must be stated where the Q3 result will be reported.

- **(d) WikiText-103.** §III.E says a corpus difference counts as quality only if it has the same sign "on the other corpus's validation set and on WikiText-103." But Appendix B4 says that "most added runs" are evaluated on WikiText, and the plan (§1) says "New runs are also evaluated". The main-grid endpoints are not. The plan's rule says "where available". State which endpoints enter the WikiText leg, and how the rule applies to the main grid.

- **(e) "The plan poses six questions, each with a decision rule" (p. 26) overstates the plan.** Only Q2 has a decision rule. Q1 is a test, Q3 is a signed statistic, and Q4–Q6 are descriptive.

- **(f) The plan header says 03:15, but the commit is at 03:12.** Appendix B4 explains this, but the plan's own claim to have been "written … committed to git before any estimation" should not carry a time later than its commit.

---

## 5. Minor comments

1. **p. 29, Figure 5.** Both panels label the x-axis "planned serving share of lifetime cost" for all models. This contradicts pp. 11 and 34 and the plan-v2 "do not claim" rule. Relabel it "expenditure share s = (w − 1)/w" and leave the serving reading to the notes.
2. **p. 32, Farseer extrapolation.** "0.70 to 0.78 times … (the range over kernel bandwidths)" for κ free and "0.44 to 0.50" for Chinchilla cover only the primary and cross-validated bandwidths. ra1 H4 also reports the ×1.5 and ×2 bandwidths: κ free gives −0.49 and −0.67 (0.61× and 0.51×), and Chinchilla gives −0.96 and −1.14 (0.38× and 0.32×). Eq. 3 gives 0.73–0.82. Report the full range. The sign is unchanged; the magnitude is not.
3. **p. 20, Table 1.**
   - (i) Farseer's model-free 0.708 has no standard error. ra1 gives a wild cluster SE of 0.001 and a range of 0.701–0.711 across bandwidths.
   - (ii) "Chinchilla, IsoFLOP runs, 137" includes the five high-loss runs that fall outside every estimation window, and they supply the M minimum of 0.04. Report 132, or add a footnote.
   - (iii) The standard errors are of different types across panels: Feng–He–Hu wild in Panel A; pairs or cell bootstrap for Farseer and Panel B. The κ-free sweep–corpus Q = 82.1 uses the pairs SEs; with design-conditional wild SEs it is 104.4 (ra1 H3). Add a footnote.
4. **p. 21.** "Biased by at most 0.006 on each design's own fitted technology." This excludes Porian's κ-family truth, which gives +0.047 to +0.050 (`ra1_modelfree_isoflop_bias.csv`). Qualify.
5. **p. 23.** "The quasi-likelihood-ratio statistic of 7.8 is borderline." Against χ²₁ it has p ≈ 0.005. State the critical value that makes it borderline (the bootstrap-calibrated one?).
6. **p. 33 and Appendix F.** "1/σ* − 1 between 0.40 and 0.52, the range of the κ-free and model-free estimates." This excludes the κ-free small sweeps, which imply 0.61–0.96. Name the four estimates used: κ-free Chinchilla, κ-free Farseer, model-free Meta and model-free Marin.
7. **p. 30.** "Technologies that let the data choose the curvature give medians of 0.69 to 0.79." Two sentences later, the κ-free small-sweep fits, which also let the data choose the curvature, give up to 0.85. Reword, for example "κ-free fits of the large designs and model-free curvatures".
8. **p. 34.** 33 models from 7 serving developers plus 44 models from 12 non-serving developers makes 19 developers, against 18 in the sample. Meta appears in both groups, because it is coded as serving from 27 September 2023. Say so, and say how the Meta cluster is coded in the serving regression.
9. **p. 36 and Table 3, Panel A.**
   - "Across ten technologies, a runs from 0.36 … to 0.57" omits ra1's model-free Marin Comma path, a = 0.35, which p. 24 reports.
   - The top of the growth range (2.8×) is set by two technologies. One is Farseer's own-form local slope, evaluated at 10²²–10²³ FLOP, beyond Farseer's design (about 3.5×10²¹). The other is the OLMo ladder, whose a has a bootstrap interval of [0.11, 0.82] (ra3 memo).
   - Note both, or add the range over the IsoFLOP model-free paths (0.35–0.50, i.e. 2.24–2.86× per year).
10. **p. 40 (§V.D) and Table 3.** The growth-model figure uses the κ = 1 refit (γ = 0.178, 49×), which §III rejects. The reference κ-free technology gives γ = 0.165 and 66× (`ra3_econ_growth_calibration.csv`). Table 3 Panels A and B also list κ = 1 first. Lead with the reference.
11. **p. 35.** "The curvature σ*, the best-measured object." Given the κ-free heterogeneity across sweeps (Q = 82) and R2-M4, write "the object on which the IsoFLOP designs agree."
12. **p. 25.** "Every restriction is rejected" for DataDecide rests on B = 19 (floor p = 0.05) and B = 99. Raise B to at least 999, as round-1 M6(c) asked, or carry "at the resolution of the bootstrap" into the main text.
13. **p. 26.** "Evaluation on WikiText-103 … address[es] the other threats." See R2-M5(d): not every endpoint is evaluated.
14. **Notation (main text).**
    - c ≡ ln C (p. 5) against c_T(N) and c_D (Proposition 2).
    - v, the data-loss term in eq. (2), against v, the allocation-error s.d. in Proposition 3(iii) and Figure 2.
    - e, the path slope in Proposition 5, against the exponential in eq. (1) and in ŵ = we^{−χ} (p. 10).
    - R_i, reducible loss in Proposition 3(iii), against R, repetitions, and R*_D in §V.B.
    - δ, the FLOP-price elasticity, against the Huber δ in the Figure 2 notes.
    - The price ratio p next to p-values (pp. 28, 34).
    - ε (elasticities) against ϵ (noise).
    - k in Table 1, Panel C against k_L and k_U in Proposition 5.
15. **p. 13.** "2.0, 4.1 and 7.6 percent" is correct: my recomputation gives 2.05, 4.13 and 7.56. But no output file contains these numbers; `m7_theory_dmr_family.csv` still holds the v1 members. Add them to the replication outputs.
16. **p. 29.** "(Llama 1 and 2, OLMo 1, Marin 8B)": the ex-post set also includes OLMo-7B-0424 (OLMo 1.7).
17. **Appendix E, opening paragraph (p. 134).** "The tenfold gain of Gundlach et al. … not a gain realized between eras" still reads as a correction. Align it with p. 153, which credits their definition and states that the paper's addition is the realized-gain contrast.
18. **Appendix D, p. 130.** "Less against their steeper per-size estimates" should give numbers. From `ra4_obsfix_bjorck_shares.csv`, conditioning closes 17–28 percent of the gap against Bjorck et al.'s 125M–350M exponents (0.38–0.65; smoothed optimum), against 26 percent at the headline 0.32.
19. **Appendix A, p. 56.** Corollary A1 is titled "Gross complementarity". In IO usage, with two inputs, Hicksian cross effects are positive. σ < 1 is complementarity in the sense of the Hicks elasticity, which is the macro usage of "gross complements". The main text avoids the phrase, and plan v2 lists "gross complements" among claims not to make. Rename it, for example "Complementarity (σ < 1)".
20. **Replication package.**
    - (i) `output/tables/m1_chinchilla_horse_race.tex` still reports SD-based errors for M*, (9.0) [25.6], against the paper's IQR-based (6.9) in Appendix Table D (`appD_estimators.tex`). Round-1 minor 43 is still open.
    - (ii) The revised sources, the m6 v2 outputs, the ra1–ra5 outputs and all m9 files are uncommitted or untracked (integration log §9, item 13). Tag the commit that builds the submitted PDF.
    - (iii) Footnote 1 says the terms of Epoch's Chinchilla digitization, the Farseer and Step Law run files, and Czech's Llama 3 digitization do not permit redistribution. Give the license basis for each; "no license granted" is not the same as an explicit prohibition.
    - (iv) "Available from the authors" must become a public, versioned archive (for example openICPSR) before acceptance.

---

## 6. Recommendation

**Major revision.** From the replication-audit standpoint alone, the paper is close:
- the numbers are accurate;
- the round-1 audit items are resolved;
- R2-M1, R2-M2 and R2-M4 are rewording plus tables the authors already have;
- R2-M3 is a re-run of existing code.

What keeps this at a major revision:
- The paper's central original evidence, the controlled experiment, is still a set of placeholders. They sit in the Introduction, §III.E, Figure 4, §IV.D, the Conclusion and three appendices.
- Its pre-registration record (R2-M5) must be completed and disclosed before those results can be evaluated at the standard the paper sets for itself.
- Two abstract claims currently fail the paper's own robustness classification.

---

## Appendix A. Audit ledger (main text and Tables 1–3)

Format: **location — statement — source — status.** Statements sharing a source are grouped.

### A.1 Abstract and Introduction (pp. 1–5)
- p. 1 — abstract is 99 words — `main.tex` — PASS.
- p. 1 — compute about fivefold a year since 2018; 2.4-fold cost growth — `ra3_econ_compute_growth.csv` (5.06); Cottier et al. — PASS.
- p. 1 — "most of lifetime cost … up from near zero before 2023" — ra3 by-technology and universe files — **FAIL on robustness class** (R2-M1, R2-M2).
- p. 3 — σ* about 0.70 [0.67, 0.72]; no detectable heterogeneity — `ra1_modelfree_heterogeneity.csv` (0.695 [0.673, 0.717], τ = 0) — PASS numerically; convention-conditional (R2-M4).
- p. 3 — 77 models, 2023–2025; s = 0.75; lab-own 0.61 — `ra2_wedge_cleaning.csv`, `ra2_wedge_labown.csv` (0.749; 0.606 against 0.646) — PASS.
- p. 3 — 0.04 (2019–2022) → 0.82 (2025); "rises under every technology" — `ra3_econ_inference_share*.csv` — numbers PASS; the juxtaposition is misleading (R2-M1).
- p. 3 — data demand 2.0–2.8×/yr — `ra3_econ_data_demand.csv` (2.02–2.84) — PASS.
- p. 3 — Porian about 0.51 — ra1 (0.518/0.505) — PASS.
- p. 4 — 86 percent sign-identified; none w < 1; range 0.17–0.85 — `ra2_wedge_pi_summary.csv`, `ra2_wedge_technologies.csv` — PASS (anchors: R2-M3).

### A.2 Section I (pp. 5–12)
- p. 7 — γ = 0.178, so a 1 percent loss cut needs 5.6 percent more compute — PASS.
- p. 8 — α = 0.348, β = 0.366; σ* = 0.737; σ between 0.732 and 0.742 — PASS.
- p. 11 — eleven cases in Table F1 — `appF_wedge_cases.tex` (11 rows) — PASS.
- p. 11 — Gopher w = 0.36 (Besiroglu) — PASS.
- p. 12 — Llama 3 8B: M ≈ 1,875, w ≈ 5.2, 4.2 percent, s ≈ 0.81 (Besiroglu, nominal 8B); Meta's own w ≈ 8.4 — PASS.

### A.3 Section II (pp. 12–18)
- p. 13 — 2.0/4.1/7.6 percent at 5·M* for σ* = 0.85/0.74/0.60 — recomputed (2.05/4.13/7.56) — PASS (not in any output file; minor 15).
- p. 14 — on-path band of 41 runs, set [0.20, 0.99]; full design [0.68, 0.72] — `ra1_modelfree_chinchilla_profile_sets.csv`, `_set_calibrated.csv` — PASS.
- pp. 15–16 — corner share 87 percent; RMSE of σ* 0.089 → 0.015; IsoFLOP 0.0036; RMSE of ln M* 4.6 and 0.59 against 0.16; 91 percent flat on the grid; 65 percent containing [0.50, 0.95] at v = 0.3; widths 0.016/0.019; 79 percent singular or corner; 78 percent multistart coverage — `m6_montecarlo_designA_{summary,profile,bootcheck}.csv` (0.868; 0.0887/0.0147; 0.00357; 4.580/0.590/0.162; 0.913; 0.653 [v1 grid]; 0.0161/0.0186; 1 − 0.214; 0.783) — PASS.
- p. 17 — largest IsoFLOP budget 10²²; 72 of 77 above it; two models inside Chinchilla's (M, C) — `ra2_wedge_models.csv`, `ra2_wedge_insupport.csv` — PASS.
- p. 18 — Llama 3 8B: M = 1,868 at 7×10²³; 405B: 38 at 4×10²⁵ — PASS.

### A.4 Section III and Table 1 (pp. 18–27)
- p. 19 — 245 runs, 1.4×10¹⁸–1.3×10²², 137/108, 35 trunks, n = 240 — `ra1_modelfree_chinchilla_design.csv` — PASS.
- p. 19 — Llama 3 ten budgets to 10²²; Marin to 3×10²⁰; Porian 16 architectures, 12 budgets to 2.6×10¹⁹; Farseer 404 of about 1,000 — ra1 memo §2.2 — PASS.
- p. 19 — off-path spread 1.4–2.5 in the IsoFLOP designs; 0.94 on the OLMo ladder — Table 1 — PASS.
- Table 1, Panel A:
  - model-free point estimates and SEs;
  - same-run κ-free and κ = 1 estimates with FHH SEs;
  - Farseer rows: 0.708 (first-derivative estimator), 0.710 (0.003), 0.772 (0.010).

  Sources: `ra1_modelfree_isoflop_summary.csv`, `ra1_modelfree_isoflop_param.csv`, `m2_table3_technology.csv`. PASS.
- Table 1, Panel B — the six rows' κ-free and κ = 1 estimates and SEs — `m2_table3_technology.csv` — PASS.
- Table 1, Panel C — all five rows (k, mean, CI, τ, I², Q, p) — `ra1_modelfree_heterogeneity.csv`; the first row recomputed — PASS.
- p. 21 — 0.673 (0.027); 0.660 (0.023); 0.700/0.713/0.705; 0.708 with 0.701–0.711 across bandwidths; RE 0.695; Q = 4.1 on 5 df, p = 0.54; one-per-study 0.693; power below 0.02; 0.664 at CV bandwidth giving mean 0.679 — PASS.
- p. 21 — bias ≤ 0.006, coverage 92–98 percent — `ra1_modelfree_isoflop_bias.csv`, `ra1_modelfree_isoflop_mc.csv` — PASS, with the Porian κ-truth exception (minor 4).
- p. 21 — 0.44η; Marin η = 0.087 gives 0.64–0.66; ±0.04 — ra1 memo (0.658/0.642/0.655) — PASS.
- p. 21 — Porian 0.518 (0.017) and 0.505 (0.014); 0.45–0.55 across windows; all-in mean 0.646 — PASS.
- p. 23 — drift −0.072 (0.026), −0.053 (0.010), −0.015 (0.008); Chinchilla 0.034–0.096 across windows — `ra1_modelfree_isoflop_summary.csv`, `_sens.csv` — PASS.
- p. 23 — 73 against 57 percent at 10·M*; w = 2.7, s = 0.63 — `ra1_modelfree_practitioner.csv` (1.727; 1.574) — PASS.
- p. 23 — κ̂ = 0.774; rejected at 1 percent under five of six schemes; p = 0.004 (budgets plus trunks) and 0.052 (original) — ra1 H5 — PASS.
- p. 23 — rejected in all seven technologies; QLR 7.8 on the Chinchilla profile runs — `ra1_modelfree_isoflop_param.csv` (7.81) — PASS (minor 5).
- p. 23 — Llama 3: 0.769 is 4.8 SE and 0.693 is 1.5 SE from the model-free estimate; Farseer 0.772/0.708/0.710; full design 0.737/0.701; [0.68, 0.72] excludes 0.74 (p = 0.003, at the floor with 299 draws) — PASS.
- p. 24 — κ-free 0.59–0.62, 0.544, 0.511; Q = 82.1, τ = 0.068, RE 0.622 [0.549, 0.695]; Chinchilla–Farseer p = 0.51; 3.3–7.8 SE; IsoFLOP κ-free p = 0.08; E–κ ≤ 0.08 against ≤ 0.006; OLMo task bits per byte 0.51; κ = 1 gives 0.735–0.825 — ra1 CSVs and memo H6 — PASS.
- p. 24 — a 0.37–0.57 and 0.36–0.57; M*(10²¹) 3.4–60 (18-fold); model-free path slopes 0.50 and 0.35–0.43 — `m2_table3_technology.csv`, `ra3_econ_data_demand.csv` — PASS.
- pp. 24–25 — Farseer embeddings: a 0.53 → 0.41, M*(10²³) 19.5 → 45, σ*_κ 0.710 → 0.668; Chinchilla 0.514 → 0.556; Porian 975 runs, 0.84 → 0.50, 39–47 and 53–61 percent — PASS.
- p. 25 — Gadre p = 0.90, 0.004, 8–18 percent; DataDecide +0.05 on 99.7 percent, tilt 0.22–0.26, 2.9–3.4× and 1.25–1.29×, carried by DCLM; DeepSeek 0.45 → 0.58 (arXiv 2401.02954, Table 4) — PASS (p-value resolution: minor 12).
- pp. 25–26 — experiment design: 0.39M–49M and 1.4M–54M parameters; 44 endpoints; M 0.51–2,031 and 0.46–555; spread 1.96; 4,063 and 1,743 at 3.2B tokens with four layers; about 10¹⁷ FLOP — Appendix B4, `m9_preanalysis_plan.md`; recomputed 4,063/1,743 — PASS.
- p. 26 — pre-registration description — **FAIL on "one deviation … changes no … decision rule"** (R2-M5).

### A.5 Section IV, Table 2, Figures 5–6 (pp. 27–35)
- p. 28 — R² = 0.9999; T/D = 9.0 at p = 1 and 0.9 at p = 10 — ra2 memo H11 and H10 — PASS.
- p. 28 — nine steps; 77 models, 36 families, 18 developers; median between 3.77 and 4.46 — `ra2_wedge_cleaning.csv` — PASS.
- pp. 28–29 — 32 technologies; σ*_κ = 0.701; M*(10²¹) = 22.7; κ = 1 gives 3.99 → 3.34; Meta σ* 0.660; Marin 0.705; 22 lab-own models (11 ex ante, 11 ex post); embeddings ≤ 28 percent — ra2 CSVs — PASS.
- pp. 29–30 — ŵ = 3.99, s = 0.75 [0.69, 0.79]; 97 percent above one; range 0.17–0.85; 0.69–0.79 cluster; Marin A2 σ* 0.87–0.90; MiniCPM 192; 64 percent; 18 percent; σ* 0.80 → 0.60 moves ŵ 2.25 → 8.64 and s 0.55 → 0.88; lab-own 0.61 against 0.65; Llama 3 8B 8.4 [5.7, 12.8], s 0.88; 405B 0.97; alternatives 3.1–12.2 — `ra2_wedge_*` CSVs — PASS (intervals: R2-M3).
- Table 2 — all 16 rows × 9 columns and all four MoE rows — `ra2_wedge_models.csv` (`w_chin_q`, `wlo_`/`whi_`, `s_ref`, `w_lab`, `band_lo`/`band_hi`, `wtotalN_chin_q`) — PASS.
- Figure 5 — medians 0.75/0.75 and 0.65/0.61 — PASS; axis label **inconsistent** (minor 1).
- p. 30 — 47 models above M = 341: median s 0.56 inside and 0.85 outside; no band above one inside — `ra2_wedge_insupport.csv` — PASS.
- p. 32 — local w = 7.5 at M ≥ 1,024; 0.70–0.78; 0.44–0.50; 0.26–0.30; 57 models: 0.68 → 0.71/0.72; 0.10–0.34B; 20 models beyond — ra1 H4, `ra2_wedge_extrapolation_check.csv` — PASS (bandwidth range: minor 2).
- p. 33 — 88 percent above 1.3×10²²; anchors 22.7/22.3/9.4–12.4; e ∈ [−0.16, 0.19]; [2.3, 89]; Meta [10.1, 57]; 86/77/23 percent; 11 models, all with M ≤ 96; bounds 0.59/0.93 — `ra2_wedge_pi_*.csv` — PASS numerically (R2-M3).
- p. 34 — open-weight premium 0.52 (p = 0.011) on 186 models from 67 developers; serving 0.45 (0.26), p = 0.39, 7 of 18 clusters; 51 against 13 percent of 220; placebo p 0.37–1.00; 33 models/7 developers at 0.80 and 44/12 at 0.74; 32 models in 11 families; 0.18–0.89; 0.75; 3.90 against 4.01; 0.72–0.77 and 0.70–0.86; synthetic 0.74 — `ra2_wedge_conduct.csv`, `_bunching.csv`, `_serve_split.csv`, `_family_split.csv`, `_costsens.csv`, `_cleaning.csv` — PASS (developer count: minor 8).
- p. 35 — 0.04–4.6 percent; elasticity 0.80, p = 0.031, 74 models/16 developers; 141 models; 0.04/0.35/0.54/0.82; 92 percent; 0.24–0.96; 39 percent and 0.63 — ra2/ra3 CSVs — PASS numerically (R2-M1).

### A.6 Section V, Table 3, Figure 7 (pp. 35–40)
- p. 36 — 5.1 [4.3, 5.9]; footnote: 92 runs, 27 developers; 4.8×; 5.2× against Epoch's 4.2× — `ra3_econ_compute_growth.csv` — PASS.
- p. 36 — a 0.36–0.57; 2.0–2.8×; 34–183×; 9.4×10²⁶ FLOP; 9–519T — `ra3_econ_data_demand.csv` — PASS (minor 9).
- p. 36 — 100T reached mid-2026 to late 2027; 320T late 2027 to mid-2029; 2025.7–2029.4 — `ra3_econ_data_demand.csv`, `ra3_econ_exhaustion_mc.csv` — PASS.
- Table 3, Panel A — all ten rows × seven columns — `ra3_econ_data_demand.csv` — PASS.
- pp. 36–37 — 7.2/7.4/7.7 percent; r_max 117/83/47; 43 percent and r ≈ 9; 43 against 40 percent; σ_CU 0.28–0.58; 0.95γ and 0.82γ; r 0.09–5.2; 0.6–1.2; 0.43, $3 and $88 per million; D/D* multiple 4.7 against 2.3 — `ra3_econ_wall_*.csv`, `ra3_econ_wedge_data_multiple.csv`, ra3 memo — PASS.
- Table 3, Panel B — the r_max, σ_CU, shadow-value and penalty cells I traced — PASS; the remaining cells were not traced.
- Table 3, Panel C — all cells — `ra3_econ_inference_share.csv` — PASS.
- p. 37 — Google about 3/5; Meta 70 percent; Nvidia about 40 percent — ra2/ra3 memos — PASS.
- p. 40 — w − 1 = 4.5; 10–23 percent; 0.18–0.34 — `ra3_econ_inference_share_flow.csv` and ra3 memo (0.34 at ρ = 4.4; 0.18 at ρ = 10.4; p = 1) — PASS.
- p. 40 — 49× and 161×; 2.4–3.1 years — `ra3_econ_growth_calibration.csv` — PASS (minor 10).

### A.7 Section VI (pp. 40–42)
- p. 41 — 0.70 [0.67, 0.72]; γ 0.14–0.18; 86 percent; 88 percent; "a quarter" (20 of 77) beyond Farseer's range; 50–160× — PASS.

---

## Appendix B. Citations checked in this round

| Where | Attribution | Check | Verdict |
|---|---|---|---|
| pp. 4, 8 | Hao and Merrill (2026): a profit model with training and serving; a Leontief approximation that rules out over-training | arXiv 2605.16430 HTML: π = ωt·f(q) − δt² − (6nd + 2nt)/E; Leontief, and "a·n^α = b·d^β" holds at the optimum | Correct |
| p. 25 | DeepSeek LLM (2024): a rises 0.45 → 0.58 with data quality | arXiv 2401.02954, Table 4 (0.450/0.524/0.578) and the quoted sentence | Correct |
| p. 11 | De Ridder, Grassi and Morzenti (2026): levels more fragile than ranks | Econometrica 94(1):137–168; trends and dispersion are well measured, levels are not | Correct |
| p. 13 | Chen (1995): n^{−1/4} rate in over-fitted mixtures | Annals of Statistics 23(1) | Correct |
| p. 7 | Nerlove (1963): returns to scale that vary with scale | Round-1 M8(a) | Now correct |
| p. 36 | Villalobos et al. (2024): 100T [22, 490]; 320T; median 2028 | ra3 review §B (checked against the PDF) | Consistent with the authors' check |
| p. 37 | Patterson et al. (2022); Wu et al. (2022); NVIDIA FY2024 call | ra2 H9, ra3 memo | Correct (but see R2-M1(f)) |
| p. 40 | You (2025); Denain and Wu (2026) | ra3 memo | Consistent |
| p. 26 | Olken (2015); Webb (2023) | Standard | Correct |

---

## Appendix C. "Do not claim" list (`paper_plan_v2.md` §3)

| Item | Found? | Where / comment |
|---|---|---|
| First to read scaling laws economically, compute σ, or note confounding | No | — |
| "Gross complements" as a finding | No in the main text | Appendix A, Corollary A1 is titled "Gross complementarity" (minor 19) |
| Cardinal T in tokens without p | No | T/D is always given with p |
| "Planned inference/serving" for developers who do not serve | **Yes, in Figure 5** | Axis labels on p. 29 (minor 1). The text is correct. |
| w levels identified at frontier scale | No in the text; **implicitly in the abstract** | "Most of lifetime cost" (R2-M2) |
| Observational benchmark shows no bias | No | Appendix E says "failure to reject … not evidence of unbiasedness" |
| "LaLonde" framing | No | Cited only as the contrast case (Appendix E) |
| Nerlove "spurious"; "within 4×10⁻⁴"; "exactly 2E" | No | — |
| (v1 D4) "σ more stable than a" | Borderline | p. 24 disavows it; p. 35 "best-measured" (minor 11) |

---

## Appendix D. Recomputations (ad hoc scripts; no project file was modified)

1. **Random-effects summary of the model-free σ*.** DerSimonian–Laird τ² with a modified HKSJ interval (t₅) on ra1's six estimates and SEs. It reproduces 0.695 [0.673, 0.717], τ = 0, Q = 4.1 (p = 0.54). The alternatives are in R2-M4.
2. **The 2019–2022 universe.** I ran ra2's `sample.build()` and `sample.universe()` with the `open_closed()` filters, and applied the reference wedge ln w = 0.4274·ln(M/M*(C)), with ln M*(C) backed out of `ra2_wedge_models.csv`. This reproduces 0.039/0.348/0.535/0.819. The eight 2019–2022 models and their weights are listed in R2-M1. Dropping OPT-175B and BLOOM-1.7B gives 0.058.
3. **Aggregate shares by technology.** From `ra3_econ_inference_share_by_tech.csv`, the share is monotone over 2023 < 2024 < 2025 for 32 of 32 technologies. It is below 0.5 for 16 technologies in 2024 and 5 in 2025.
4. **Equivalent-family off-path losses.** Members of the κ family sharing Besiroglu's path and frontier, at C = 10²², by direct minimization: 2.05/4.13/7.56 percent above the frontier at 5·M*.
5. **Table 2.** Every cell against `ra2_wedge_models.csv`. The MoE bounds come from `wtotalN_chin_q` and `w_chin_q`.
6. **Bibliography.** I parsed all 127 `\harvarditem` entries in `main.bbl`; every one has a venue or locator after the title.
7. **φ profile.** `ra4_obsfix_phi_profile.csv`: LR crosses 3.84 near −0.44 and 3.37; τ_C is 5.35–39.4 on the grid, and about 39.6 when interpolated to the bound.
