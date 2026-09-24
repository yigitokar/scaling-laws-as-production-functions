# Writer notes: introduction (paper/sections/introduction.tex)

Writer: section writer "introduction", 2026-09-24. The file compiles alone with
`bash code/paper/test_section.sh introduction`: no LaTeX errors, every citation resolves, no overfull boxes.
The only undefined references are labels that other sections define (listed in §5).

Length: about 2,450 words of main text, not counting parenthetical citations and the one footnote. About 2,800
words as rendered, counting citation text. The target was about 2,300; the related-literature block (three
paragraphs) plus the positioning paragraph comes to about 650 words, not counting citations.

Structure:
1. Hook.
2. Thesis.
3. Run-in paragraph "What scaling data identify.—", followed by a quantitative paragraph.
4. "The technology.—", followed by a debates paragraph with the TBD-m9 sentence.
5. "Revealed inference demand.—".
6. "Observational scaling and 'algorithmic progress.'—".
7. "Related literature.—", in three strands.
8. Positioning statement.
9. Roadmap.

The italic run-in heads follow AER's "Title.—Text" convention. If the lead author prefers plain paragraphs, delete
the four `\textit{...}---` heads; the prose still reads.

---

## 1. Proposed abstract (116 words; replaces the placeholder in main.tex)

> Neural scaling laws relate a language model's loss to its parameters, training data, and compute. We show that
> they are production functions and that fitting them is production-function estimation: standard practice
> estimates a cost function, conditional factor demands, and a primal technology, and labs that minimize compute
> generate data that identify neither the technology's curvature nor its optimal input mix. In seven public training
> sweeps, parameters and data are gross complements, with an elasticity of substitution near 0.7 once the data choose
> the curvature. Inverting labs' first-order conditions, we show that over-training reveals anticipated inference
> demand: the median open-weight model is trained as if its lifetime inference compute will be about 2.2 times its
> training compute.

Notes on the abstract:
- **"About 2.2 times"** is median w − 1 = 3.19 − 1 = 2.19, the ratio of inference compute to training compute. The
  plan's draft says "twice"; 2.2 is the precise figure.
- **When m9 confirms the experiment,** change "In seven public training sweeps" to "In seven public training sweeps and
  our own controlled experiment". This adds 4 words, for 120 in total.
- **"Near 0.7":** the q/κ-free σ* ranges from 0.51 to 0.71, and Farseer's three q-free estimators give 0.69–0.71.
  The Chinchilla-form range is 0.73–0.83. "Near 0.7 once the data choose the curvature" is the plan's framing.
  "About 0.7" would be defensible only for Farseer's precise estimate; the small sweeps' q-free values are 0.51–0.62,
  with CIs of about ±0.1. A more conservative alternative: "between 0.5 and 0.8, and about 0.7 in the best-designed
  sweep".
- **JEL:** D24, L86, O33, C51 (optional L11).
- **Keywords:** production function estimation; elasticity of substitution; scaling laws; large language models;
  identification; revealed preference.

## 2. Proposed title footnote (`\thanks{}` in main.tex)

> Okar: Care AI (email: okar.yigit@gmail.com). Claude: Anthropic. Claude is an AI system developed by Anthropic; it
> designed and executed the analysis and drafted the paper together with the first author. Data and code
> availability: all data used in this paper are publicly available. The replication package [LOCATION TBD]
> reproduces every table and figure from the raw downloads with fixed random seeds. For sources whose licenses do not
> permit redistribution (Epoch AI's reconstruction of the Chinchilla runs, the Farseer and Step Law run files, and the
> digitized Llama 3 IsoFLOP points), it provides download scripts in place of the data.

Notes on the footnote:
- **Licensing basis.** The m1, m2 and m8 memos state that these sources lack a license, so they are "used, not
  redistributed".
- **Optional acknowledgments** (keep minimal): thank the authors of the public sweeps (Farseer/Step Law, Gadre et
  al., OLMo ladder, DataDecide, Porian et al., Epoch AI) for releasing run-level data.
- **Courtesy note to Ho et al.** The m5 memo recommends notifying Ho et al. of the optimizer-tolerance finding before
  publication. That courtesy could be mentioned here once it has been done.

---

## 3. Every number in the introduction, with its source

The statistical numbers were checked against the CSV/JSON, not just the memos. "Lit." marks figures taken from
published sources.

| Number in text | Value used | Source |
|---|---|---|
| Frontier training compute growth | 4–5×/yr | Lit.: `sevilla2024training` (title; notes: notable 4.1×/yr, frontier 5.3×/yr) |
| Amortized cost growth of the largest runs since 2016 | about 2.4×/yr | Lit.: `cottier2024rising` (90% CI 2.0–2.9; `lit/notes/ml_observational.md`) |
| "Eighty years" of production-function estimation | 1944 → 2026 | `marschak1944random` |
| MC, RMSE of σ*, every run compute-optimal (standard κ = 1 primal estimator) | 0.28 | `m6_montecarlo_designA_summary.csv`, cell `opt_s0`, primal: 0.2819 |
| MC, RMSE of σ*, IsoFLOP design | 0.004 | same file, cell `iso16`, primal: 0.00357 |
| MC design size | 90 runs, equal total compute | m6 memo §2.2 |
| κ-free profile CI covers all of [0.50, 0.95] | 88–97% of replications for s ≤ 0.2 | `m6_montecarlo_designA_profile.csv`, `share_flat_everywhere`: onpath 0.880, s=0 0.973, s=0.05 0.913, s=0.1 0.933, s=0.2 0.893 |
| Chinchilla on-path band (\|Δ ln N\| ≤ 0.15): max κ-free profile LR | at most 2.0 (1.96) | `m1_chinchilla_profile_sigma.csv`, gauss, on-path band |
| Random same-size subsamples: profile LR | 79–101 | m1 memo H9 / m1 review issue 2 (`m1_chinchilla_fdep_sizecontrol.csv`) |
| Full design: profile LR | up to 424 | `m1_chinchilla_profile_sigma.csv`, gauss, full: 424.5 |
| Chinchilla-form σ*, seven sweeps (Huber) | 0.73–0.83 | `m2_table3_technology.csv`: min Muennighoff 0.7345, max OLMo 0.8250 (DataDecide pooled 0.828) |
| Bootstrap SEs of σ* | 0.006–0.034 | same file, `se_sigma_star`: Chinchilla 0.0062, OLMo 0.0338 |
| κ = 1 rejected in all seven | p < 0.001 everywhere | `m2_spec_tests.csv`, `p_q1_wald` and `p_q1_LR` |
| q/κ-free σ* | 0.51–0.71 | `m2_spec_tests.csv`, `sigma_star_q`: Muennighoff 0.511, Farseer 0.710 |
| Farseer: three q-free estimators | 0.69–0.71 | `m2_farseer_local_sigma_summary.csv` (median 0.690; Eq. 3 median 0.706) and `sigma_star_q` 0.710 |
| Farseer nonparametric median and CI | 0.690 [0.679, 0.699] | `m2_farseer_local_sigma_summary.csv` |
| Farseer runs | 404 | `m2_table3_technology.csv` `n_runs` |
| Allocation exponent a across sweeps | 0.37–0.57 | `m2_table3_technology.csv` Huber: OLMo 0.370, Gadre RW 0.566 |
| M*(10²¹) across sweeps | 3–60 | same file: Gadre RW 3.35, Muennighoff 59.6 (DataDecide excluded: levels descriptive only) |
| Gadre: Hicks + E_r not rejected | p = 0.90 | `m2_neutrality.csv`, gadre/hicksE, `p_wild` 0.896 |
| DataDecide: E shifts explain | 98% | `m2_neutrality.csv`, datadecide/Eshift `share_explained` 0.981 |
| DataDecide: compute-optimal ratios differ | 2.9–3.4× | `m2_neutrality_magnitudes.csv`, `Mstar21_ratio`: NLS 2.93, Huber 3.43 |
| Porian path, Kaplan → tuned (RW) | 0.84 → 0.50 | `m8_measurement_porian_steps.csv`: 0.837 → 0.496 |
| Share of the gap: input measurement | 39–47% | m8 memo §1.2; RW 0.132/0.341, OWT2 0.162/0.344 |
| Share of the gap: flexible inputs | 53–61% | same |
| Lifetime compute | 6ND + 2NT | model_spec; `sardana2024beyond` |
| Wedge FOC | w = 1 + T/(3D) | m7 Prop. A8 |
| Sufficient statistic | ln w = (1/σ* − 1) ln(M/M*(C)) | m7 Lemma A4(iii) / Cor. A3 |
| Sample B size | 173 models, 2021–2025 | `data/processed/m3_wedge/headline.json` (`n_sampleB_core`) and m3 memo H1 |
| Median w | 3.19 | `headline.json` `sampleB_median_w` = 3.186 |
| Inference / training compute at the median | about 2.2 | w − 1 = 2.19 (plan note) |
| Share w > 1 | 95% | `headline.json` 0.948 |
| Share with the whole band > 1 | 79% | `headline.json` `sampleB_share_bandlo_gt1` = 0.786 |
| Median w across technologies | 1.9–4.0 | `m3_wedge_sampleB_by_tech.csv`: Hoffmann 1.909, Gadre RW 4.027 |
| Share w < 1 by year | 47% (2021) → 0% (2024) | `m3_wedge_trends.csv`, year rows: 0.474, 0.000 |
| Median w by year | 1.06 (2022) → 4.68 (2025) | same file: 1.064, 4.676 |
| Production-scale threshold | 6ND ≥ 10²¹ | m3 memo H4 |
| Llama 3 8B w and CI | 5.27 [3.97, 8.12] | `headline.json` models/Meta-Llama-3-8B |
| Llama 3 8B T/D | about 13 (12.8) | same; T/D = 3(w − 1) |
| Llama 3 8B T/D under Meta's law | about 6 (6.3) | w_meta = 3.09 → 3 × 2.09 = 6.27 |
| 405B flagship under Meta's law | w = 0.98 | `headline.json` Llama-3.1-405B `w_meta` 0.983 |
| Downloads on ln M at fixed C | 0.97 (0.18); 0.72 with developer FE (SE 0.14, not quoted) | `m3_wedge_validation.csv`, ln_dlall `cond_C` / `cond_C_lab` |
| LaLonde sample | 57 models | `m4_observational_table6_lalonde.csv` `n_obs` |
| OLS θ_C bias (HellaSwag) | −0.001 (0.038) | same file, OLS theta_C |
| "Biases of about 20 percent cannot be ruled out" | CI ±0.075 on a benchmark of 0.329 | m4 memo claim 1 (the memo says ±23% / "±20%") |
| Family FE θ_N / θ_D bias | +0.20 / −0.26 | same file: +0.203 / −0.262 (wild p 0.052 / 0.023 in `_wildboot.csv`; not quoted) |
| Weak instruments | first-stage F ≤ 3.8 | m4 memo item 6 (frontier FLOP/$ IV, all specifications) |
| Ho et al. published doubling time (bootstrap median) | 8.4 months | `m5_progress_replication.csv`: 8.4438 |
| Ho et al. point estimate | 8.7 months | same: 8.6848 |
| Converged T_C and paper-cluster CI | 6.1 [3.0, 22.7] | `m5_progress_table7_panelA.csv` row 1: 6.08 [3.05, 22.67] |
| Estimator range | 6.1–10.2 months | same file: rows 1 and 2 (6.08, 10.20) |
| Profile-likelihood CI | [4.1, 40.5] | `data/processed/m5_progress/dmr_summary.json` `TC_profile_ci_interp` = [4.15, 40.46] |
| Realized allocative gains, C ≥ 10²³ | 1.0–2.4× | `m5_progress_table7_panelB.csv`: Besiroglu 1.047, Hoffmann-TeX 1.914, Hoffmann-rounded 2.412, Farseer 1.583 |
| Allocative share of the Ho-rate gain | 2–40% | same file, `allocative_share_of_algorithmic`: 0.021 to 0.403 |
| "Even its sign" depends on the technology | 0.55–4.3× across sweep technologies | same file, m1/m2 registry rows (Llama 3 0.74; FineWeb 0.55; Marin Comma 4.31) |
| Gundlach's 10× | counterfactual cost of keeping Kaplan's rule at 2025 frontier compute | m5 memo claim 9; `m5_progress_kaplan_counterfactual.csv` (11.65× at 5e26 under Besiroglu; not quoted) |
| Footnote: seven sweeps and 25 DataDecide recipes | | m2 memo §2.1 |

## 4. Claims made, with the caveats they carry

- **Identification (Props. duality / fdep / info / dmr; Lemma sigma).** These come from m7 and theory_main_text,
  with the m7 corrections applied:
  - "the sign of the bias is identified" is qualified with "measured on training-optimal allocations". The full
    caveat (fixed relative cost weights over time, FLOP-only costs) belongs in Section II.
  - The information result is phrased as "second order in the size of runs' log deviations" (M*) vs "fourth order"
    (curvature). This is correct for S = α+β and has the same order for σ*, per the m7 review.
  - The "γ identified" statement is qualified as "data from a single lab", since γ needs E[ω | c] known.
  - σ in (0, 1): "between zero and one", i.e. 0 < σ < 1, per the m7 review.
- **Monte Carlo.**
  - The design is stated (90 runs, equal total compute).
  - Per the m6 review, "near-optimal" depends on s relative to noise and on design size. The text gives the s
    threshold (≤ 0.2) and the design.
  - The m6 review says not to cite on-path bootstrap coverage without the warm-start qualifier; it is not cited.
- **Chinchilla on-path flatness.** The size-matched control (random subsamples 79–101) is included, per the m1 review.
- **σ.**
  - Stated as a range: 0.73–0.83 under the Chinchilla form, 0.51–0.71 with κ free, and 0.69–0.71 on Farseer.
  - Only the κ-free estimates are compared with the capital–labor range ("overlap the top"), per the m2 review.
  - **"σ more stable than a" is not claimed.** Instead: "a and M* are not portable". The word "stable" appears only
    in the related-literature sentence "exponents and the frontier are precisely estimated in published fits while
    the levels A and B, and with them M*, are not". That is m7's point about sampling precision in published
    Chinchilla fits, not a cross-sweep stability claim.
- **Data quality.**
  - "Do not reject" is used for Gadre, which has low power.
  - The DataDecide tilt magnitude is quoted; its levels, and the Lemma 1 reading, are not.
  - The schedule-artifact caveat is left to Section IV.
- **Kaplan–Chinchilla decomposition.** It is order-dependent and covers 5M–901M models only; these caveats are left
  to Section IV.
  - The bias formulas are described as getting "sign and order of magnitude right". The measurement formula
    over-predicts by 20–108%, and the flexible-input formula explains 47–114% (m8 review).
- **Wedge.**
  - "Not a markup" is stated.
  - "Technology-sensitivity band" is the m3 review's requested term.
  - The level is stated as technology-dependent (1.9–4.0).
  - The validation is read as "consistent with anticipated demand rather than a measurement of T", with the
    fixed-compute ≡ smaller-models equivalence stated (m3 review).
  - The aggregate ecosystem multiple (2.0 [1.2, 3.7]) is **not** used in the intro, because it is illustrative only
    (m3 review: low confidence).
- **LaLonde.**
  - "No detectable bias" is framed with the power caveat ("biases of about 20 percent cannot be ruled out").
  - The N/D split is called "unreliable", not "biased in a known direction" (m4 review).
  - The IO remedies are stated as unavailable in public data, not as failing in general.
- **Ho et al.**
  - Per the m5 review, 6.1 months is not presented as the right answer. The text reports the 6.1–10.2 range across
    estimators and the ridge.
  - The profile CI is iid-calibrated. The paper-cluster NLS bootstrap [4.1, 27.6] is the "safer number to quote
    alongside" and should appear in Section VI.
- **Allocative gains.** "Under the main technologies" (Besiroglu, Hoffmann ×2, Farseer), with "even its sign" for
  the registry technologies. Section VI should report the share under all three Ho estimators (m5 review).

## 5. Cross-references assumed to exist elsewhere

- **Sections:** `sec:framework`, `sec:ident`, `sec:data`, `sec:tech`, `sec:wedge`, `sec:obs`, `sec:concl`.
- **Results:**
  - `prop:duality` (Prop. 1)
  - `prop:fdep` (Prop. 3)
  - `prop:info` (Prop. 4)
  - `prop:dmr` (Prop. 5)
  - `prop:wedge` (Prop. 2)
  - `lemma:sigma` (Lemma 1: an interior optimum implies 0 < σ < 1)
- **Appendices:** Online Appendix A (proofs), B (data), C (Monte Carlo), D (additional), referred to in text by
  letter, without `\ref`.
- **No tables or figures are referenced from the introduction.**

## 6. Placeholders

- One sentence, in the "technology" block, second paragraph:
  `\textcolor{red}{[TBD-m9: one-sentence headline --- σ by lab with κ free, and whether data quality is Hicks-neutral or factor-biased]}`.
  - The preceding clause describes the design as "two 'labs' train the same model sizes on corpora of different
    quality (FineWeb-Edu and FineWeb) under a shared tokenizer". Adjust it if the final m9 design differs.
- **Optional, once m9 is in:**
  - add "and in our own controlled experiment" to the first sentence of "The technology.—";
  - update the abstract as noted in §1.

## 7. New bib keys

None. Every key used exists in `paper/references.bib`: sevilla2024training, cottier2024rising,
hestness2017deep, kaplan2020scaling, hoffmann2022training, marschak1944random, griliches1998production,
ackerberg2007econometric, ackerberg2015identification, diamond1978measurement, besiroglu2024chinchilla,
li2025predictableb, gadre2024language, bhagia2024establishing, muennighoff2023scaling, magnusson2025datadecide,
chirinko2008sigma, oberfield2021micro, porian2024resolving, nerlove1963returns, sardana2024beyond,
deloecker2012markups, lalonde1986evaluating, ho2024algorithmic, farrell1957measurement, gundlach2025origin,
li2025predictablea, choshen2024hitchhikers, li2025misfitting, czech2026problems, pearce2024reconciling,
kricheli2026tokens, hao2026theory, mertens2026secret, whitfill2025note, konig2026validity, erdil2022algorithmic,
ruan2024observational, maiapolo2024sloth, trammell2023economic, erdil2025gate, bergemann2025economics,
korinek2025concentrating, demirer2025emerging, mundlak1961empirical, olley1996dynamics, levinsohn2003estimating,
gandhi2020identification, bond2020unpleasant, raval2023testing, doraszelski2018measuring, demirer2020production,
klump2007factor, leonledesma2010identifying, collardwexler2016production, foster2008reallocation,
hsieh2009misallocation, syverson2011determines.

Keys in the plan's list that are **not** used (to save space): thompson2020computational, hoch1962estimation,
zellner1966specification. The plan's "bond2021some" does not exist; the correct key is `bond2020unpleasant`
(Journal of Monetary Economics, 2021).

## 8. Open issues for the integrator

1. **Label collision (important).** `appendix_proofs.tex` already defines `\label{prop:info}`, `prop:dmr`,
   `prop:transmission`, `prop:wedge` and `prop:pi` (and `lem:sigma`) for its A-numbered results. The main-text labels
   assigned to writers use the same names (`prop:info`, `prop:dmr`, `prop:transmission`, `prop:wedge`, `prop:pi`).
   In the integrated build these become multiply defined, and `\ref` will resolve to whichever label comes last,
   i.e. the appendix's "A2", "A3", and so on. Rename one set; the simplest fix is to prefix the appendix labels
   (`app:prop:info`, …).
2. **Plan vs CSV discrepancy.** The plan (§0 and the Section IV list) says the Chinchilla-form σ* is "0.74–0.83". The
   lowest value is Muennighoff's 0.7345 (`m2_table3_technology.csv`), which rounds to 0.73; the memo says
   0.735–0.828. The intro uses **0.73–0.83**. Other sections should use the same range, or write "0.735–0.83".
3. **Plan vs memo: the "twice" in the draft abstract.** The precise median inference/training ratio is 2.19, so the
   intro and the abstract use "about 2.2 times".
4. **Notation.** The intro uses κ for the outer exponent, per the plan; m2's memo and CSVs call it q. Sections III–IV
   should translate q → κ.
5. **Sweep count.** The intro says "seven public training sweeps": Chinchilla, Farseer, Gadre ×3, OLMo ladder and
   Muennighoff single-epoch, with DataDecide used for the neutrality tests. It says so in footnote 1. Sections III–IV
   and the conclusion should use the same count.
6. **Term.** The intro uses "technology-sensitivity band" for the m3 PI band, as the m3 review asks. Section V should
   use the same term, not "partial-identification band".
7. **Moved out of the intro for length; suggested for Section IV.A.** "Hoffmann et al.'s published parameters imply a
   compute-optimal model about 12 percent smaller than their own IsoFLOP minima indicate (p ≈ 0.04–0.06 in
   design-consistent tests); the replication parameters show no such gap" (m1 claim 5, suggested wording).
8. **Section VI** should carry the full caveats referenced in the intro:
   - the ±0.075 CI of the LaLonde test;
   - ARC-C / Winogrande attenuation of 18–31% (not significant);
   - the iid calibration of the profile CI, with the paper-cluster alternative [4.1, 27.6];
   - the allocative share under all three Ho estimators.
9. **Section II** must state the fixed-relative-cost-weights condition for the sign-identification result (m7 review,
   issue 18).
10. **The intro describes σ* on-path as "the on-path elasticity of substitution σ* = 2/(2+α+β)".** Section I should
    define it the same way, and credit hao2026theory for the general σ formula. The intro credits Hao and Merrill for
    "derive the elasticity of substitution implied by the Chinchilla form".
11. **AI-author disclosure.** The text is in §2 above. main.tex's current `\thanks` has only the affiliations and
    email.
