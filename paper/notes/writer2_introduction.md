# Writer notes (v2): introduction (paper/sections/introduction.tex)

Writer: section writer "introduction", 2026-09-24 (version 2, after referee round 1).

**Status.**
- `bash code/paper/test_section.sh introduction` compiles with no LaTeX errors, no citation warnings and no overfull
  boxes. I also compiled with `--keep-logs` to check this. Rendered pages were checked with pymupdf.
- The only undefined references are labels that other sections define: `sec:framework sec:ident sec:tech sec:wedge
  sec:econ sec:concl app:proofs app:obs app:wedge lemma:sigma prop:wedge prop:ident prop:modelfree prop:pi`.
  They all follow the numbering map in `theory_main_text_v2.md` §0 and the label list in the task.
- I also compiled the proposed front matter below (title, `\thanks`, abstract, JEL, keywords) with AEA.cls in a scratch
  copy of the main preamble. It renders on page 1 without errors.

**Length.**
- The limit is 1,800 words. The text runs about 1,750 words with all citations removed (each math chunk counts as
  one word, and the red TBD-m9 placeholder is included). Counting in-text `\citet` names as words (about three each)
  and leaving out parenthetical citations, it is about 1,800.
- When m9 is done, replace the red placeholder (about 40 words) with one sentence of the same length or shorter.

**Structure** (follows plan §2 and R3 M2):
1. Stakes: compute and cost growth, scaling laws as the planning tool, and why substitution and the value of
   compactness matter.
2. The question and the idea: optimization cuts both ways, and designed experiments break the circle.
3. *Theory.* Two paragraphs:
   - Lemma 1, with its intuition and empirical content;
   - Prop. 2, with its intuition and the reading of s;
   - Prop. 3, with the intuition for the fourth-order rate and a scope sentence;
   - Prop. 4, with its intuition.
4. *The technology.* Model-free σ*, the parametric comparison, what does not travel, and the m9 placeholder.
5. *What over-training reveals.* The inversion's two inputs, the headline s, the lab-own result and the trend.
6. The economic implications paragraph (Section V).
7. *What is robust.* One paragraph.
8. *Related literature.* Two paragraphs: the closest papers, then the IO antecedents. The pointer to Online
   Appendix E is in the first.
9. Roadmap.

---

## 1. Front matter for the integrator (main.tex; do not edit main.tex myself)

**Title.** `\title{What Optimizing Labs Reveal: Scaling Laws as Production Functions}`
**Short title.** `\shortTitle{What Optimizing Labs Reveal}`

**Running heads (R3 E2; plan §1).**
- Remove the journal volume and issue placeholders.
- The test build still prints "VOL. VOLUME NO. ISSUE" and "THE AMERICAN ECONOMIC REVIEW MONTH YEAR" in the heads.
- Use a working-paper head instead, e.g. "OKAR AND CLAUDE: WHAT OPTIMIZING LABS REVEAL" and "WORKING PAPER —
  SEPTEMBER 2026". This needs `\pubMonth{September}`, `\pubYear{2026}` and edits to the AEA.cls head macros.

**Author and `\thanks`** (verbatim; compiled and checked):
```
\author{Yigit Okar and Claude\thanks{Okar: Care AI (email: okar.yigit@gmail.com). Claude: Anthropic. Claude is an AI
system developed by Anthropic. It designed and carried out the analysis and drafted the paper together with the first
author, who takes responsibility for its content. Data and code availability: all data used in the paper are publicly
available. Code and data-construction scripts are available from the authors (replication package: the
scaling-laws-pf repository). The package reproduces the tables and figures from the raw downloads with fixed random
seeds. For sources whose terms do not permit redistribution (Epoch AI's digitization of the Chinchilla runs, the
Farseer and Step Law run files, and the digitized Llama~3 IsoFLOP points), it provides download scripts in place of the
data.}}
```
- The disclosure sentence is the plan's text (§1). "[LOCATION TBD]" is gone (R3 E3, R4 M9(c)).
- The byline follows the user's explicit request. Whether it is compatible with AEA/COPE authorship policy is the
  user's decision (R3 E1, R4 M9(d)); see open issue 3.

**Abstract** (99 words, counting hyphenated compounds as one word; 103 if they are split. Limit 100):
```
\begin{abstract}
Language-model developers choose model size and training data to minimize training and serving costs. This
optimization cuts both ways for measurement. It removes the variation that identifies the technology---information
about curvature is fourth order in allocation errors---but makes choices informative about objectives. Designed
experiments give, without functional-form restrictions, an elasticity of substitution between parameters and data of
about 0.7. Inverting first-order conditions with this technology shows that open-weight models released since 2024 were
trained as if compactness---for developers that serve them, lower serving cost---accounted for most of lifetime cost,
up from near zero before 2023.
\end{abstract}
```
Support for each claim:
- **"Fourth order in allocation errors."** Prop. 3(iii): information about σ* is O(v⁴) in log-wedge dispersion
  (theory_main_text_v2).
- **"About 0.7, without functional-form restrictions."** The model-free random-effects mean is 0.695, HKSJ 95% CI
  [0.673, 0.717], with τ = 0 and Q = 4.1 (p = 0.54). Source: `ra1_modelfree_heterogeneity.csv`, set "model-free
  sigma*, excluding Porian", all technologies.
- **"Since 2024 … most of lifetime cost."** These are compute-weighted planned shares under the reference (κ-free
  Chinchilla) in `ra3_econ_inference_share.csv`:
  - 2024: 0.535 [0.41, 0.64];
  - 2025: 0.819 [0.76, 0.86];
  - pooled 2024–2025: 0.657. I computed the pooled value from the file's `m_ref` and `C_total` columns as
    s = m/(1+m), with m the compute-weighted mean of the two periods' m_ref. It is not stored in any CSV.
  - Per-model medians: 0.81 (2024) and 0.85 (2025).
- **"Near zero before 2023."** s_ref for 2019–2022 is 0.039 [0.032, 0.049], from the same file. Only 8 models fall in
  this period; OPT-175B and GLM-130B hold 92% of its compute.
- **"With this technology."** This phrase carries the level caveat. The rising trend holds under every technology
  (ra3 H6). The level does not: the 2025 range across technologies is [0.24, 0.96].
- **The plan's "open-weight models released since 2024 … most … near zero in 2021".** I kept its structure but
  wrote "before 2023", because the earliest period in the CSV is 2019–2022.
- **A firmer alternative, if the lead author prefers one:** "open-weight models released in 2025 were trained as if
  compactness … accounted for about four-fifths of lifetime cost" (0.82 [0.76, 0.86]). It is precise, but the level
  is technology-dependent.

**JEL.** `\JEL{C51, D24, L86, O33}`. These are the plan's codes. R3 minor 32 suggests considering L63 or O30; I did
not add them.
**Keywords.** `\Keywords{production functions, identification, elasticity of substitution, scaling laws, large
language models, revealed preference}`

---

## 2. Numbers used in the introduction, with sources

There are 16 result numbers. Years and the FLOP-accounting constants 6ND and 2N are not counted. Every value was
checked against the CSV named.

| # | Text | Value in source | Source |
|---|---|---|---|
| 1 | frontier compute "about fivefold a year since 2018" | 5.06 [4.29, 5.93], 2018.1–2026.7, 92 runs | `ra3_econ_compute_growth.csv` (top10); consistent with `sevilla2024training` (4–5×) |
| 2 | amortized cost of largest runs "about 2.4-fold a year since 2016" | 2.4×/yr | `cottier2024rising` (R4 citation audit: PASS; `lit/notes/econ_ai.md`) |
| 3–5 | model-free σ* "about 0.70, 95% CI [0.67, 0.72], no detectable heterogeneity" | μ_RE 0.6950, HKSJ [0.6730, 0.7170], τ = 0, Q = 4.06 (p = 0.54) | `ra1_modelfree_heterogeneity.csv` (model-free, excluding Porian, k = 6: Chinchilla 0.673, Llama 3 0.660, Marin 0.700/0.713/0.705, Farseer path 0.703) |
| — | "Chinchilla form overstates σ* on Llama 3's profiles" (no number) | κ = 1 0.769 (0.006) vs model-free 0.660 (0.023), 4.8 SE | `ra1_modelfree_isoflop_param.csv`; ra1 memo H2/C3 |
| — | "free the outer curvature … come close" (no number) | κ free: 0.667–0.701 (Chinchilla), 0.693 (Llama 3), 0.663–0.678 (Marin), 0.710 (Farseer) vs model-free 0.660–0.713 | ra1 memo H2; `ra1_modelfree_isoflop_summary.csv` |
| — | "M* differs by more than an order of magnitude across designs" | M*(10²¹) 3.35–59.6 across the seven κ = 1 sweep technologies | `m2_table3_technology.csv` (Huber rows) |
| — | "allocation exponent … also differs" | a 0.36–0.57 across ten technologies | `ra3_econ_data_demand.csv`; ra3 H2 |
| 6 | Porian "about 0.51" | 0.518 (RefinedWeb), 0.505 (OpenWebText2) | `ra1_modelfree_heterogeneity_inputs_modelfree.csv` |
| 7 | clean inference-demand sample "77 open-weight base models released in 2023–2025" | n = 77, 36 families, 18 developers | `ra2_wedge_cleaning.csv` (row h_D_undocumented) |
| 8 | median s "three-quarters … (s = 0.75)" | 0.7491 [0.689, 0.794] | `ra2_wedge_cleaning.csv`; `ra2_wedge_technologies.csv` (Chinchilla κ free) |
| — | "almost every model is over-trained" | share ŵ > 1 = 0.974 | `ra2_wedge_cleaning.csv` |
| 9 | lab-own median "0.61, slightly below the reference" | 0.606 (n = 22) vs 0.646 under the reference | `ra2_wedge_labown.csv` (median of s_lab, s_ref) |
| 10–11 | compute-weighted share "0.04 in 2019–2022 to 0.82 in 2025"; "rises under every technology" | 0.039; 0.819 [0.76, 0.86] | `ra3_econ_inference_share.csv`; ra3 memo H6 |
| 12–13 | data demand "2.0 to 2.8 a year" | 2.02 (Gadre RW, a = 0.566) to 2.84 (Farseer Eq. 3, a = 0.357), at 5.06× compute growth | `ra3_econ_data_demand.csv` (gD_hat); ra3 H2 (ten-technology main set) |
| — | "σ* barely affects the cost of a binding cap; repetition matters far more" | r = 4: 7.2 / 7.4 / 7.7% for σ* = 0.74 / 0.70 / 0.60 (D′-only), 43% under Muennighoff's full model | ra3 memo H4; `ra3_econ_wall_*.csv` |
| 14 | "w > 1 identified for 86 percent … w < 1 for none" | PI-1: 0.857; w < 1: 0.0 | `ra2_wedge_pi_summary.csv` |
| 15–16 | "median share ranges from 0.17 to 0.85 across the technologies we specify ex ante" | 0.169 (MiniCPM law) to 0.854 (Muennighoff κ free), 32 technologies | `ra2_wedge_technologies.csv` (in_set) |
| — | "Inside Farseer's design … parametric forms understate w at high M" | M ≥ 1,024: w_param/w_local 0.44 (Chinchilla), 0.70 (κ free), 0.26 (Chinchilla fit on M ≤ 100) | `ra1_modelfree_farseer_delta.csv`; ra1 C6 |
| — | "open-weight models are more over-trained than closed ones at given compute" | +0.517 log points (CRV1 s.e. 0.180), WCR p = 0.0115, 67 developer clusters | `ra2_wedge_conduct.csv` |
| — | "serving one's own models does not significantly predict over-training" | 0.453 (0.257), WCR p = 0.39; 18 clusters, 7 treated | `ra2_wedge_conduct.csv` |

Theory statements are from `theory_main_text_v2.md`: Lemma 1, Props. 2–5, the scope sentence, the Kricheli
reconciliation and the positioning paragraph. The Cobb–Douglas intuition for Lemma 1 is mine (log-coordinate
geometry). It matches Lemma A2 (σ = P/(P+Q) in log coordinates).

---

## 3. Referee comments addressed in the introduction

- **R3 M2 (order, frame, density, title).**
  - The introduction now runs: stakes → question → double-edged idea → three results → economic implications → what
    is robust → closest papers → roadmap.
  - It carries 16 result numbers (about 110 in v1) and never more than one interval per sentence.
  - The new title follows R3's direction.
  - The IO citation catalog is gone; the IO literature is one paragraph of five antecedents.
- **R3 M2.4–5, M6.1; R1 c12; R4 D-list.**
  - "Gross complements" is no longer a finding. σ < 1 is stated as the implication of interior optima, and its
    empirical content is U-shaped IsoFLOP profiles.
  - "Near 0.7" is now the model-free, homogeneous estimate across three laboratories' annealed designs.
  - The Porian ladder (≈ 0.51) and the heterogeneity of the small sweeps are disclosed.
  - "2.2 times" is removed.
- **R3 M2.6.** The claim is now that on-path data carry no information about *curvature*, not about "the optimal
  input mix".
- **R3 M3(a)(b).** The text states that the inversion needs exactly the zero point M*(C) and the scale 1/σ* − 1.
  Results are reported as the share s, and no T appears in tokens (R2 Major 2).
- **R3 M4, R4 M2, R2 Major 1.** The reference technology is Chinchilla with κ free. Lab-own technologies are reported
  (R3 M3(d)).
- **R3 M5; R1 c1–2.**
  - s is read as the value of compactness in general, and as planned serving expenditure only for developers that
    serve their own models.
  - The conduct evidence is stated: the open-weight premium, and the low-power serving test.
- **R3 M7.** One paragraph on economic implications: data-demand growth, and the data wall, where σ barely matters
  and repetition matters far more.
- **R3 M8 (positioning).**
  - Hao & Merrill: we invert their kind of problem once σ > 0, and extend it to developers that do not bear serving
    costs.
  - Kricheli et al.: the reconciliation is that on the compute-optimal path the degeneracy arises whatever the
    exponents, and we add the rates.
  - Sardana, Bian, Roberts: they solve the forward problem; we solve the inverse.
  - Mertens and Whitfill: one sentence, pointing to Online Appendix E.
  - Bond & Söderbom and Box & Lucas are cited here. Sargan and Rotnitzky are left to Section II.
- **R3 M10.** Observational results are one sentence (Appendix E), not in the abstract.
- **R3 M11.** The scope sentence is in the introduction: "This binds in compute-optimal ladders run by a single
  laboratory; released models lie far off the path…".
- **R3 minors 1–4, 8.** Dropped: the "eighty years … solved" framing, the Monte Carlo and grid sentences, and the Ho
  et al. claims.
- **R3 E1–E3.** The `\thanks` has a disclosure footnote. The running-head fix goes to the integrator. "[LOCATION
  TBD]" is removed.
- **R1 c3.**
  - The DLW analogy is kept with its limits: w is not a markup.
  - Its elasticities come from experiments, which avoids the Doraszelski–Jaumandreu circularity at the cost of
    external validity.
  - The Bond et al. and Raval critiques are acknowledged and passed on to Section IV.
- **R1 c7.** The model-free σ* estimator leads the technology results.
- **R1 c8.**
  - (a)–(b): Prop. 3 is positioned as Marschak–Andrews / Bond–Söderbom, with the rates as design-theoretic orders
    generated by economic wedges.
  - (d): the "IsoFLOP sweep as a price shock" analogy is removed.
  - (g): Nerlove is removed from the introduction.
  - (i): "LaLonde" is removed.
- **R2 Major 3.**
  - The sign's robustness now rests on bounds on M*(C) (Prop. 5), not on the functional form.
  - Levels are stated as not robust.
  - The direction of the extrapolation error inside Farseer's design is reported.
- **R2 Major 12.** Bian et al. (2025) and Roberts et al. (2026) are cited.
- **R2 minor 1.** "Verified" is not used for token counts.
- **R2 minor 2.** The introduction gives no sweep count, so the five-versus-seven count problem does not arise.
- **R4 M8(a)–(b).** The Nerlove "spurious" language and the Gundlach framing are removed.
- **R4 M9(b).** The abstract is 99 words.
- **R4 M9(c).** The replication statement is in the `\thanks`.
- **R4 D4.** Portability claims avoid saying "σ is more stable than a". The text says that M* differs by more than
  an order of magnitude, and that a "also differs across designs and published laws".

**Do-not-claim list (plan §3), checked.** The introduction does not contain any of the following:
- "first";
- "gross complements";
- T in tokens;
- "planned inference" for developers that do not serve;
- w levels identified at frontier scale;
- "no bias" in the observational benchmark;
- "LaLonde";
- "spurious";
- "within 4e-4";
- "exactly 2E[ln w]".

**Terminology.** The text uses "clean inference-demand sample", "corpora" (not labs) for m9, E (irreducible loss, in
words only) and "loss–compute frontier". The terms 𝔼, τ_C and "verified sample" do not appear.

---

## 4. Open issues

1. **m9 headline.** The red placeholder is at the end of the "The technology.—" paragraph. Its content follows the
   pre-analysis plan (Q1–Q3): model-free and κ-free σ* by corpus; neutrality on both validation sets and on WikiText;
   the sign of the extrapolation error at high M. Replace it with one sentence of 40 words or fewer. If the result
   contradicts the "about 0.70" statement, the abstract's "about 0.7" needs revisiting too.
2. **Running heads.** The AEA.cls head macros need an integrator edit (see §1).
3. **Authorship.** The byline lists Claude, as the user requested. R3 E1 and R4 M9(d) note that COPE-following
   journals do not accept AI authors. This is the user's decision, not the writers'.
4. **Ho et al. critique.** R3 E5 asks that the critique (now in Appendix E) be shared with the original authors, and
   that the paper say so. The user must decide and act; the `\thanks` makes no such statement.
5. **Replication statement.**
   - The `\thanks` says the package "reproduces the tables and figures … with fixed random seeds". Every module memo
     reports deterministic single-command runs, but m9 is not finished.
   - R3 minor 34 asks for archived snapshots of time-varying sources: Hugging Face downloads and model trees, and the
     Epoch database snapshot of 2026-09-23. The data appendix should state them. I did not add a claim about
     checksums, which I could not verify.
6. **Consistency with other sections.** Please keep these numbers identical:
   - **Technology section.** σ* "about 0.70 [0.67, 0.72]" (RE, HKSJ, k = 6) and Porian ≈ 0.51. ra1 open issue 9
     suggests replacing Farseer's Hessian-based path σ* (0.703) with the first-derivative estimate (0.708) in the
     heterogeneity table. If the technology writer does that, the RE mean must be recomputed and the introduction
     updated. The narrower-bandwidth variant gives 0.679 [0.656, 0.703].
   - **Wedge section.** 77 / 0.75 / 0.61 / 86% / 0.17–0.85.
   - **Economics section.** 0.04 → 0.82 (the ra3 table built on ra2's clean open-weight universe of 141 models) and
     2.0–2.8×/yr.
7. **The pooled 2024–2025 share (0.66) behind the abstract's "most".** I computed it here and it is not in any CSV.
   If the integrator wants a stored number, ask ra3 to add a pooled 2024–2025 row with its CI.
8. **No figure or table is referenced** from the introduction. Exhibit labels were not yet final when I wrote it.
