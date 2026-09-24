# Integration log: "Scaling Laws as Production Functions"

Integrator pass, 2026-09-24. Build: `tools/tectonic -X compile main.tex` in `paper/`. The build has **0 errors, 0 undefined references or citations, 0 multiply defined labels, 0 duplicate hyperref anchors, and 0 floats too large for the page**. Three overfull boxes remain, all under 5pt and all in Online Appendix A. Page images at 60 dpi are in `paper/_build/pages/page_NNN.png`.

## 1. Bibliography

- `code/paper/build_bib.py` was rerun. `paper/references.bib` now holds 346 entries from 11 files.
- The writers added 7 keys:
  - `penedo2024fineweb`, defined in both `extra_writing_data.bib` and `extra_writing_technology.bib`. The data version wins because it sorts first, so no action is needed.
  - `loshchilov2019decoupled`, `su2024roformer`, `zhang2019root`, `hannun2023mlx`, `sennrich2016neural`, `zellers2019hellaswag`.
- I checked each key's authors, venue, year, pages and DOI against the published record I know. None looked suspicious, so I did not fetch them from the web.
- bibtex reports no warnings.

## 2. Front matter (`main.tex`)

- **Abstract:** 117 words. It is the introduction writer's proposal with one edit: "identify neither the technology's curvature nor its optimal input mix" became "the better labs optimize, the less their data reveal about the technology's curvature and its optimal input mix", to match the identification results. It says "about 2.2 times" (median w − 1 = 2.19).
- **`\JEL{C51, D24, L86, O33}` and `\Keywords{...}`:** AEA.cls supports both and prints them under the abstract.
- **`\thanks`:** affiliations, the AI-author disclosure in the plan's wording, and a data/code availability statement. The repository location is a red placeholder, `[LOCATION TBD]`. The statement lists sources without a redistribution license (download scripts are provided instead).
- **Preamble:**
  - `\E`, `\Var` and `\Cov` are defined with `\providecommand`.
  - Added `\usepackage[section]{placeins}` so floats stay within their section. Figure 5 had been drifting into Section V.
- **After `\appendix`:**
  - `\p@subsection` is emptied, so appendix subsection references print "B1", not "B.B1".
  - `\theHtable`, `\theHfigure` and `\theHequation` are set to the printed numbers, which makes the hyperref anchors unique.

## 3. Labels, numbering, cross-references

- **Label collision fixed.** In `appendix_proofs.tex`, `prop:info`, `prop:dmr`, `prop:transmission`, `prop:wedge` and `prop:pi` became `prop:A-info`, `prop:A-dmr`, `prop:A-transmission`, `prop:A-wedge` and `prop:A-pi`, and the internal references were updated. The main-text labels now point to Propositions 2 and 4–7. References in Appendices C and D and in the tables deliberately point to the main-text propositions.
- **Main text ↔ Appendix A correspondence:**
  - Each A-result header now names the main-text statement it proves, e.g. "Proposition A1 (Functional dependence; Proposition 3 in the main text)".
  - A paragraph at the start of Appendix A gives the full map:
    - Prop 1 = Lemma A3.
    - Lemma 1 = Lemmas A1–A2 and Corollary A1.
    - Prop 2 = Prop A8 with Lemmas A4–A5 and Corollary A3.
    - Prop 3 = Lemma A4 and Prop A1.
    - Prop 4 = A2.
    - Prop 5 = A3 and Corollary A2.
    - Prop 6 = A5–A7.
    - Prop 7 = A9.
  - `\theH...` was added for the A-numbered theorem counters, which removes the duplicate `proposition.N` anchors.
- **Appendix A notation aligned with the main text:**
  - The Hicks bias is now `\mathcal B_t`, which also removes a clash with the loss coefficient `B_t` inside Prop A3.
  - Noise variance is `V_\epsilon` in Prop A2 (was `s^2`).
  - Productivity persistence is `\rho_\omega` in the dynamic-panel remark.
  - One sentence now reconciles the appendix's R = u + v with the main text's R = L − E.
- **Appendix table and figure numbering was broken.** `appendix_data.tex` hard-coded `\thetable = B\arabic`, so every table in Appendices C and D printed as B1–B19. I removed the override, and the numbers now come from `\appendix`:
  - Appendix B: tables B1–B2, figure B1.
  - Appendix C: tables C1–C3, figures C1–C2.
  - Appendix D: tables D1–D19, figures D1–D3.
- **Hard-coded references replaced:** `Lemma~A4` → `\ref{lem:geometry}`; `Lemma~A3` → `\ref{lem:alloc}`; several bare "Online Appendix D" pointers → specific table references (`tab:app-neutrality`, `tab:app-rivals`, `tab:app-robust`, `tab:app-farseer`, `tab:app-duality`, `tab:app-sfa`/`-demand`/`-steplaw`, `tab:app-lalonde-robust`, `fig:app-allocative`).
- **Missing derivation added.** Section IV.D cited "derivation in Online Appendix D" for eq. (measbias) and the flexible-input formula, but neither existed. I added Online Appendix D6, "Bias Formulas for Input Measurement and Flexible Inputs" (`app:add-biasformulas`). It derives $a_m=\beta/(\beta+\alpha\varphi-\zeta)$ and $a_{\rm obs}-a=-\partial[\Delta/f'']/\partial\ln C$, with $f''=(\alpha+\beta)\gamma R^*$ and the constant-gradient case. Both come from m8 Propositions M and F; I re-derived them by hand.
- **Every figure and table is referenced and appears once.** The only unreferenced float, Figure C2, is now referenced, and `fig:experiment` now has a text reference in IV.E.
- **Main-text numbering, in order of appearance:**
  - Figures:
    1. Geometry
    2. Designs (Monte Carlo)
    3. Chinchilla
    4. σ forest
    5. Our experiment (placeholder in IV.E; the plan's "Fig 9" becomes Figure 5 by order of appearance)
    6. Wedge
    7. Trends
    8. LaLonde
    9. Ridge
  - Tables 1–9 are in the planned order.
- **Figure 1 panel labels:** the text and notes now say "panel (a)/(b)", matching the figure. Figure 2 keeps A–D, matching its own labels.

## 4. Harmonization of numbers and terms (checked against CSVs)

- **Chinchilla-form σ\* range:** "0.74–0.83" → **0.73–0.83** in technology.tex (twice) and conclusion.tex. The source is `m2_table3_technology.csv`, whose minimum is Muennighoff at 0.7345. The introduction already used 0.73–0.83.
- **IV.B ranges:** "σ\* (0.735–0.828) and γ (0.10–0.18) vary by less than a factor of two" mixed the seven sweeps with DataDecide.
  - Now σ\* is 0.735–0.825 and γ is 0.10–0.18, both over the seven sweeps.
  - With DataDecide's γ = 0.0897 the ratio is 2.004, which would break the claim.
  - The conclusion's "about 0.09 to 0.18" became "about 0.10 to 0.18".
- **Duality p-values:** Appendix D said "p ≥ 0.12 in every variant" and IV.A said "p ≥ 0.16 in every classical or bootstrap-calibrated variant". Both are correct; the 0.12 is a χ² test with robust covariance, which the m1 review says not to report. I aligned Appendix D to the IV.A wording.
- **Terms:**
  - "technology band" → "technology-sensitivity band (technology band for short)" in Section V, matching the introduction and the m3 review.
  - κ is used throughout; the q legend in Figure 4 is flagged in its notes.
- **Values checked in `m3_wedge_models.csv`:**
  - Sample B has 173 models in 68 families.
  - Median Farrell cost efficiency is 0.414, i.e. 2.4× the training-only minimum (writer-computed; confirmed).
  - Spearman correlation between the reference and alternative technologies is 0.95–1.00 (confirmed).
- **Other checks:**
  - Llama 3 8B: w = 5.27 [3.97, 8.12], T/D = 12.8, CE = 0.179, consistent across the introduction and Section V.
  - Framework's Besiroglu-parameter illustration: w ≈ 5.2, T/D ≈ 12.7, 5.5×. It is now labeled with a pointer: "(5.3 under the reference technology of Section V)".
- **Duplication removed:**
  - II.A no longer repeats the Chinchilla profile-LR numbers (≤ 2 / 79–101 / 424). They stay in the introduction and IV.A, and II.A points to IV.A.
  - V.A no longer repeats the Prop 7 variance decomposition (0.161 / 0.143 / 0.047); it points to II.E.
  - Section III no longer re-displays the ln w formula; it cites Prop 2.
- **Transitions:** opening bridge sentences were added to IV (from II: curvature needs off-path runs), V (from IV: turning the estimates around to read labs' choices) and VI (from V: the reverse question).

## 5. Tables resized

- **Font size (root cause).** In AEA.cls AER mode, `\scriptsize` is 9pt/11 (larger than `\footnotesize` at 8pt/9), and `\small` is 8pt/10. Tables 3, 4, 5 and D19 used `\scriptsize` and Table 7 used `\small`; all are now `\footnotesize`, so every table uses the same size.
- **Table 3:**
  - Dropped the M\*(10²¹) rows, which duplicate Table 4.
  - Dropped the Hausman–Wise row; it is in the text and Appendix D1.
  - Dropped the panel-A w row. Panel C keeps w with its interval, and the notes give the other estimators' w.
  - Moved n into the panel D header and condensed the notes.
- **Table 5:** removed the old panel C (non-embedding Chinchilla a, Step Law SFA, learning-rate demand). All of it is in the text and in Tables D16–D18; the text pointer is updated.
- **Table 8:**
  - Panel C now keeps only the composite and ladder-only rows. ARC-Challenge, Winogrande and the clean subsample are in Table D11, and the text pointers were fixed. The composite and ladder-only rows are not in D11, so they stay.
  - Wild-bootstrap p-values moved onto the standard-error rows.
- **Table 2:** the source note was shortened to point to Section III and Online Appendix B.
- **Table 4:** the notes now explain why the Chinchilla standard errors differ from Table 3 (separate bootstrap draws).
- **Table D19:** split into two floats ("…(continued)", same number, unique anchor).

## 6. Length

**Word counts.** These are my regex counts: comments, floats and display math stripped; each inline-math group and each citation counted as one token. The writers' own counts are about 2% lower.

| Section | Words | Target |
|---|---|---|
| Introduction | 2,430 | 2,300 |
| I Framework | 2,486 | 2,200 |
| II Identification | 3,500 | 2,800 |
| III Data | 1,213 | 1,000 |
| IV Technology | 3,621 | 3,000 |
| V Wedge | 3,164 | 2,800 |
| VI Observational | 2,625 | 2,200 |
| VII Conclusion | 1,023 | 900 |
| **Main text** | **20,062** | 16–18k |
| Appendix A (proofs) | 5,966 | |
| Appendix B (data) | 3,836 | |
| Appendix C (Monte Carlo) | 2,316 | |
| Appendix D (additional) | 2,741 | |

**Pages:** 133 in total. The main text runs pp. 1–60, the references pp. 60–70, and the Online Appendix pp. 71–133.

The main text exceeds about 19,000 words. The candidates below could move to an appendix, each with a pointer left in place; the words saved are estimates. I did not cut any of them.

1. **II.B:** the two Monte Carlo paragraphs "Two findings qualify…" and "Standard inference fails on the path…" could move to Appendix C, keeping one summary sentence. Saves about 450.
2. **II.D:** the illustrative calibration numbers (0.248 / 0.403) and the selection simulation numbers. Saves about 120.
3. **IV.A:** the truncation/outlier paragraph could move to Appendix D1, which already covers it, keeping two sentences. Saves about 180. The cluster-inference paragraph could be trimmed by about 60.
4. **IV.D:** the Step Law paragraph could move to Appendix D7, which already covers it. Saves about 230. The flexible-input formula paragraph could shrink now that D6 derives it. Saves about 120.
5. **V.E:** the over-identification (Raval) paragraph could move to Appendix D3. Saves about 200. V.H, the illustrative aggregate, could become two sentences plus Table D9. Saves about 150.
6. **VI.A:** the industry Monte Carlo paragraph could move to Appendix C, which already covers it. Saves about 200. The TFP-dispersion paragraph could move to Table D12. Saves about 120.
7. **Introduction:** the related-literature strand could be tightened. Saves about 200.
8. **I.E:** the Harberger approximation and the Gopher/GPT-3 illustration. Saves about 120.

Together these would bring the main text to about 18,000 words.

## 7. TBD-m9 placeholders (no m9 outputs exist yet)

1. introduction.tex, "The technology" block: one-sentence headline (σ by lab with κ free; neutral vs factor-biased). Also optionally add "and our own controlled experiment" to the abstract's "In seven public training sweeps".
2. data.tex III.B: completed endpoints, seed replicates, total training hours.
3. appendix_data.tex B4: completed endpoints per lab, replicates, failed runs, wall-clock hours.
4. technology.tex IV.E: four items:
   - σ\* and σ\*_κ by lab;
   - neutrality test, χ, effect on M\* and ŵ, and sign vs Lemma A3;
   - extrapolation check at M ≈ 1,000–2,000;
   - seed noise vs residual s.d.
5. technology.tex, Figure 5 (`fig:experiment`): the placeholder box and its notes. Replace with `\includegraphics{m9_...}`.
6. wedge.tex V.B: the X/Y/Z/W extrapolation sentence.
7. observational.tex, end of VI.C: sign of the data-quality bias from the two-corpus experiment.
8. conclusion.tex: the first-paragraph sentence and the extrapolation-bias clause in Limitations.
9. main.tex `\thanks`: `[LOCATION TBD]` for the replication package. This one is not m9.

The design numbers in Table 2 panel B and Table B2 are deterministic properties of the design; check them against the final m9 design table.

## 8. Remaining issues for the lead author

- **Figure 4** (`m2_fig3_sigma_forest`): the legend says "q free". Regenerate it with κ via `code/analysis/m2_techpanel/m2_out.py`. The figure notes flag it for now.
- **Notation:**
  - Section I describes the κ-family "with inner exponents α and β" (and Prop 2 uses α, β for the inner exponents). Section II and Appendix A write a₁, b₁.
  - IV.A now points to eq. (ident-family) for the a₁, b₁ form.
  - A full harmonization would restate Prop 2 in a₁, b₁; that is left to the author.
- **Writer additions not in the module memos** (kept):
  - V.A: w = 1 + pT/(3D) when an inference FLOP costs p training FLOPs.
  - V.G: the memory-cap example, w = 1.78 (from `m7_theory_constraints.csv`, not in Appendix A).
- **Table D19:**
  - The "Status" column uses "New", which reads as a novelty claim; consider dropping the column.
  - Its first part floats above the D8 heading.
- **T_C intervals:** D13 reports [3.05, 22.67] and D14 reports [3.0, 26.5] (different bootstrap starts). The main text quotes [3.0, 22.7].
- **Removed from the main tables:** Table 3 no longer shows M\*(10²¹) (it is in Table 4), and Table 5 has no panel C. Restore them if the author prefers fuller main tables; after the font fix, Table 3 has only about 15pt of spare height.
- **Remaining overfull boxes** (Appendix A, < 5pt): around Definition A1, a display near line 355, and Prop A6(ii).
- **Float-only pages:** placeins keeps floats in their sections, which leaves several float-only pages in Sections II–VI. This is acceptable for a submission draft.
