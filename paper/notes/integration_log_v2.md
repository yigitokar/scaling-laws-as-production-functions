# Integration log, version 2: "What Optimizing Labs Reveal: Scaling Laws as Production Functions"

Integrator pass, 2026-09-24 (after the machine reboot; resumes the interrupted first integrator run).
Build command: `tools/tectonic -X compile main.tex`, run in `paper/`.

**Build status**
- 0 errors.
- 0 LaTeX warnings: no undefined references or citations, no multiply defined labels, and no floats too large for the page.
- Four overfull boxes remain, all in Online Appendix A and all at most 6.9 pt.
- The PDF has 175 pages.
- Page images are in `paper/_build/pages_v2/`:
  - `page_NNN.png` at 70 dpi;
  - `hi_page_001.png` and `hi_page_002.png` at 110 dpi;
  - contact sheets `sheet_NNN.png`, eight pages each.
- I inspected the first pages and every main-text figure and table page. I also looked at the appendix sheets for Appendices C, D and F.

**State found at start.** The interrupted run had left `identification.tex` in the state the writer delivered it: red TBD-m6v2 flags, first-run Monte Carlo numbers on lines marked `% m6v2`, and an `\IfFileExists` switch for Figure 2. It had also left a 09:31 build of `main.pdf`. The earlier integrator made no other edits that I could detect. Before editing, I backed up all sources to the session scratchpad.

---

## 1. Bibliography (R4 M9a)

- I reran `code/paper/build_bib.py`. It reports 383 entries from 17 files.
- I audited all 130 cited entries against the rendered `main.bbl`. Every arXiv, proceedings, report and web entry now renders a venue:
  - arXiv entries as "arXiv preprint arXiv:…";
  - proceedings as journal-style venues;
  - working papers with their institution and number;
  - web sources with their URL.
- R4's 37 venue-less entries had already been converted in the source `.bib` files before this pass (commit 9cb7ec1).
- Two fixes were needed in this pass:
  - `aea.bst`: `format.editors.secondary` printed ", ed." after a `join ", "`, which gave ", , ed." (Diamond et al. 1978) and ". , ed." (Nerlove 1963). It now prints "ed.", and the `inbook` join uses ", ". The original file is saved in the scratchpad as `aea.bst.orig`.
  - `lit/references.bib`, entry `diamond1978measurement`: I dropped the `series` field, which duplicated the volume.
- bibtex reports no warnings.

## 2. Front matter (`main.tex`)

- **Title:** "What Optimizing Labs Reveal: Scaling Laws as Production Functions".
- **`\shortTitle`:** "What Optimizing Labs Reveal".
- **Authors:** Yigit Okar and Claude.
- **`\thanks`:** taken verbatim from `writer2_introduction.md`. It gives affiliations, the AI-author disclosure in the plan's wording, and a data/code statement. The "[LOCATION TBD]" placeholder is gone.
- **Abstract:** 99 words, taken from `writer2_introduction.md`. It makes robust claims only:
  - "fourth order in allocation errors";
  - "about 0.7", from the model-free random-effects estimate 0.695 [0.673, 0.717];
  - "since 2024 … most of lifetime cost, up from near zero before 2023", from compute-weighted shares of 0.54 (2024) and 0.82 (2025) against 0.04 (2019–22) in `ra3_econ_inference_share.csv`.
- **JEL:** C51, D24, L86, O33.
- **Keywords:** as in plan §1.
- **Running heads (R3 E2).**
  - Mechanism: `\pubVolume` and `\pubIssue` are removed; `\pubMonth{September}` and `\pubYear{2026}` are set; `\issueName{Working Paper}` is set. `\ps@headings` is redefined in the preamble, so `AEA.cls` itself is unchanged.
  - Odd pages read "OKAR AND CLAUDE | WHAT OPTIMIZING LABS REVEAL | page".
  - Even pages read "page | WORKING PAPER | SEPTEMBER 2026".
  - No journal name, volume or issue appears anywhere.
- **PDF metadata:** `\hypersetup` sets the title and authors.
- `code/paper/test_section.sh` still works with the new preamble; I tested it on `identification`.

## 3. Labels, numbering and cross-references

- **Label clash fixed.** In `appendix_proofs.tex`, the subsection label `app:wedge` is now `app:A-wedge-sub`. Every existing `\ref{app:wedge}` means Appendix F, and they now resolve uniquely.
- **Wedge-cases table.**
  - `framework.tex` now cites "Online Appendix Table F1" (`tab:app-wedge-cases`) instead of Appendix A.
  - I removed the alias label `tab:ra5-wedge-cases`.
- **Main-text numbering** follows `theory_main_text_v2` §0, and I checked it in the PDF:
  - Proposition 1 is duality (`prop:duality`).
  - Lemma 1 is `lemma:sigma`.
  - Proposition 2 is the generalized wedge (`prop:wedge`).
  - Proposition 3 is identification and rates (`prop:ident`).
  - Proposition 4 is model-free σ* and w (`prop:modelfree`).
  - Proposition 5 is partial identification (`prop:pi`).
- **Appendix A map.** The map paragraph at the start of Appendix A matches v2. The A-results keep their labels.
- **Fixed in Proposition 5(v).** "under case (a) of Proposition 2" becomes "under part (ii) of Proposition 2". The main-text proposition has no case (a).
- **Notation clash, Appendix A.** Definition A1 and Lemma A1(i) used s for the elasticity share, which clashes with the main text's s = (w−1)/w. The elasticity share is now ς (`\varsigma`), with a one-line disambiguation.
- **Duplicate display removed.** The technology section re-displayed the model-free formula as `eq:tech-mf`. The display is gone; the text, the Figure 3 notes and the Table 1 notes now cite `eq:ident-mf`, equation (7).
- **Figure 2.** The `\IfFileExists` switch is gone, and the figure uses `m6_montecarlo_fig2_v2` directly.
- **Exhibits.**
  - Every figure and table that is `\input` is referenced; I checked this by script. `fig:experiment` now has a reference in III.E.
  - Main-text exhibits, in order of appearance:
    1. Figure 1, geometry (p. 9).
    2. Figure 2, Monte Carlo (p. 15).
    3. Table 1, designs and σ* (p. 20).
    4. Figure 3, σ* by design (p. 22).
    5. Figure 4, experiment placeholder (p. 27).
    6. Figure 5, ECDF of s (p. 29).
    7. Table 2, selected models (p. 31).
    8. Figure 6, Farseer extrapolation (p. 32).
    9. Table 3, economics (p. 38).
    10. Figure 7, economics (p. 39).
  - That makes 10 exhibits, one of them a placeholder. The v2 dictionary table left the main text: R3's exhibit list moves it to the Online Appendix. The full dictionary is Table D1, and the "three analogies break down" paragraph stays in I.A with a pointer to it (R3 minor 10).
- **Obsolete files.**
  - `sections/data.tex` and `sections/observational.tex` moved to `sections/v1/obsolete_v1_main/`. They were identical to their v1 copies and were not `\input`.
  - 21 table files that nothing `\input`s moved to `tables/unused_v1/`: `table1_dictionary`, `table2_data`, the v1 `table3`–`table9` files, and `appD_{aggregate,attenuation,ceg_sahal,claims,demand,family,homothetic,lalonde_robust,replication,rivals,sfa,tfp}`.

## 4. Numbers updated or checked

**Monte Carlo re-run (m6 v2).** In `identification.tex`, every line marked `% m6v2` now carries re-run numbers, and the red flags are gone. Each number was checked against `output/tables/m6_montecarlo_designA_{summary,profile,bootcheck}.csv`.

| Quantity | Old value | New value | Source |
|---|---|---|---|
| Corner share at v = 0 | 51% | **87%** | `corner_share`, `opt_s0`, 0.868 |
| RMSE of σ̂* at v = 0.05 → 0.2 | 0.067 → 0.015 | **0.089** → 0.015 | 0.0887 → 0.0147 |
| RMSE of ln M̂* at v = 0.1 | 4.5 | **4.6** | 4.580 |
| RMSE of ln M̂* at v = 1 | 0.59 | 0.59 | unchanged |
| IsoFLOP benchmarks (σ̂*, ln M̂*) | 0.0036, 0.16 | unchanged | |
| Profile set is the whole grid, on the path | "entire grid [0.50, 0.95], 88%" | **"entire grid [0.05, 0.99], 91%"** | 0.913 |
| Set contains all of [0.50, 0.95] at v = 0.3 | 65% | 65% | 0.653 |
| Set widths, IsoFLOP and factorial designs | 0.016, 0.019 | unchanged | |
| Wald covariance singular or corner, on path | 35% | **79%** | 1 − 0.214 |
| Multi-start bootstrap coverage | 92% | **78%** | 0.783 |

- The main text no longer reports the warm-start bootstrap (33%), the 0.005-noise sensitivity (38%) or its TBD-m9 seed-noise placeholder. All three remain in Appendix C, with its own placeholder.
- The Figure 2 notes were rewritten to the v2 design:
  - nine random starts, none at the truth;
  - the primal and dual estimators;
  - horizontal lines for the IsoFLOP and factorial designs;
  - Panel B: a 29-point grid on [0.05, 0.99], LR relative to the unrestricted optimum, 150 replications, curves for on-path (v = 0.02), v = 0.3 and IsoFLOP.

**Other number changes and checks**
- `appendix_observational.tex`: the capability-target bias changed from "29 percent" to **30 percent**. Source: Design B v2, `m6_montecarlo_industry_summary.csv`, pooled NLS γ bias −0.0535 / 0.1783 = −0.300. The 13% and 66% figures were rechecked (0.128 and 0.665).
- `wedge.tex` IV.F:
  - The 141-model aggregate is now labelled "open-weight releases of production scale" (Table 3, panel C), not the "clean … universe".
  - The 2025 technology range 0.24–0.96 is attributed to the clean sample, where it is computed (D/F writer's open issue).
  - Checked against Table 3 panel C and Appendix F.
- **Reference M\*.** The reference technology's M\*(10²¹) = 22.7 and M\*(10²⁴) = 21.6 were checked against `ra2_wedge_technologies.csv` (`chin_q`).
- **Anchor-interval check (ra5 review §7.7).** I approximately reproduced PI-1 from `ra2_wedge_pi_anchors.csv` and `ra2_wedge_models.csv`: 87% of the clean sample is sign-identified, against the reported 86%.
  - Replacing the Chinchilla and Llama 3 anchor intervals with ra5's wild bootstrap-t intervals, [11.4, 36.8] and [12.2, 40.2], and widening Marin's anchors by the same amount leaves the share at **86%**. The headline is robust.
  - The identified set for M\*(10²⁴), reported as [2.3, 89], would widen to roughly [1.5, 110].
  - Appendix F already reports the bootstrap-t-anchor variant (77–81% on the verified sample). The main-text set was not changed; see open issue 3.
- **Verification of retained numbers.** Every other number in text I edited was already verified by its writer, and I changed none of them. I checked the arithmetic of two:
  - σ\* = 0.70 at 10× M\* gives w = 2.68 and s = 0.63.
  - T/D at the median wedge: 3 × 2.99 = 9.0 at p = 1.

## 5. Cuts, duplication and terminology (main text)

Everything cut from the main text already appears in the Online Appendix; no substance was lost.

**Framework (I)**
- Table 1 (dictionary) was removed; it is in Appendix D1.
- The DLW/Raval paragraph lost its duplicates of the introduction: Doraszelski–Jaumandreu, Hao–Merrill and Sardana.
- The part (iv) paragraph no longer repeats the 32/77/11 counts, which are in IV.E.
- The conduct-prediction sentence now points to IV.E.
- The Farrell 5.5× / 0.18 sentence was dropped; it is in Lemma A-ce.

**Identification (II)**
- Monte Carlo text condensed; the details are in Appendix C.
- The simulation exponents 0.48/0.49/0.99 were dropped; the rates are confirmed in the Appendix A proof of Prop. A1.
- The factorial slopes 2.00/4.00 were dropped; they are in Appendix A.
- The finite-grid numbers now point to Remark A-grid and Table D-finite-grid.
- The positioning paragraph was tightened to avoid repeating the introduction.
- The "believed technology" sentence is left to IV.D.

**Technology (III)**
- The model-free display was replaced by a reference.
- "Complementarity is not a finding" was shortened; the full statement is in I.C.
- The small-sweep E–κ details now point to Table D-kappa-robust.
- The DataDecide paragraph was condensed; the details are in Appendix D4.
- III.E (experiment) was condensed from about 900 to about 560 words. The design and pre-registration details (commit bd5c0ad, 03:12, and the deviation) are in Appendix B4.
  - Context length, batch size and bf16 stay in the text (R3 minor 15).
  - The six TBD-m9 blocks were merged into one, with a comment that points to `m9_preanalysis_plan.md`.

**Wedge (IV)**
- The s-reading paragraph now refers to I.D instead of repeating it.
- The rival-wedges list (in I.D) is replaced by a pointer.
- The family-budget interpretation (in I.D) is shortened.
- The OpenRouter details are in Appendix F10.
- The closing summary is shortened.

**Economics (V)**
- The Reddit licensing clause was dropped (R3 minor 30 spirit).
- The stock-timing sentence was shortened.
- The technology-range sentence now points to IV.F.

**Do-not-claim list (plan §3).** I grepped all sections and tables for each item:
- "gross complements";
- "LaLonde";
- Nerlove "spurious";
- "4e-4" / "within 4×10⁻⁴";
- "exactly 2E";
- "Sample B";
- T_C;
- "planned inference";
- "first to".

None occurs, except "spurious rejections" in Appendix C, which refers to optimizer failures and is fine. The doubling time is τ_C (Appendix E). The experiment speaks of corpora, not labs.

## 6. Tables and figures resized

- **Font sizes.**
  - In AER mode, `\scriptsize` is 9 pt, which is larger than `\footnotesize` at 8 pt.
  - Five tables used `\scriptsize`, as a table-wide size or for in-cell intervals: `appD_dictionary`, `appF_conduct`, `appF_wedge_cases`, `table3_wedge` (now main Table 2) and `appF_technologies`. All five are now `\footnotesize`.
  - Main Table 2 is still scaled by its `\resizebox` guard. I checked it at 130 dpi and it is legible.
- **Floats that were too large for the page.** Each fix moves text only where it is duplicated nearby.

| Table | Overflow | Fix |
|---|---|---|
| Table 3, `table4_econ` | 110 pt | Notes cut from about 350 to about 200 words, keeping definitions and sources; estimator details stay in the V text. |
| C1, `appC_designs` | 36 pt | The design paragraph in the notes now points to the text. |
| C4, `appC_industry` | 28 pt | Setup in the notes replaced by the rule list plus "setup in the text". |
| F3, `appF_technologies` | 35 pt | Ex-ante rule and bootstrap description replaced by "described in the text" (Appendix F3 text). |
| D1, `appD_dictionary` | 137 pt | Fixed by the font change alone. |

## 7. Length

**Word counts.** I counted with my own regex counter:
- comments, floats and display math are removed;
- each inline-math group, citation and `\ref` counts as one token;
- proposition statements and footnotes are included.

A cruder count that keeps table and figure text is given for comparison. The "about 17,000" in the brief matches neither measure; it lies between them. On a text-only count the paper is about 6 percent above 14,000 and below the 15,500 threshold that required further cuts.

| Section | Target (plan) | Before | After | After, incl. floats |
|---|---|---|---|---|
| Introduction | ≤1,800 | 1,785 | 1,785 | 1,767 |
| I Framework | ≈2,300 | 2,683 | 2,535 | 2,783 |
| II Identification | ≈2,000 | 2,538 | 2,390 | 2,676 |
| III Technology | ≈2,800 | 3,357 | 2,965 | 3,310 |
| IV Wedge | ≈3,000 | 3,284 | 3,133 | 3,491 |
| V Economics | ≈1,000 | 1,418 | 1,358 | 1,865 |
| VI Conclusion | ≤700 | 678 | 678 | 662 |
| **Main text** | **≈14,000** | **15,743** | **14,844** | 16,554 (before: 17,954) |

- About 225 of the 14,844 words are TBD-m9 placeholder text, so the substantive text is about 14,600.
- Appendix words, same counter:

| Appendix | Words |
|---|---|
| A, proofs | 10,536 |
| B, data | 4,950 |
| C, Monte Carlo | 2,988 |
| D, technology | 4,846 |
| E, observational | 5,803 |
| F, wedge | 4,557 |

**Pages.** The PDF has 175 pages, compared with 178 before this pass.

| Part | Pages |
|---|---|
| Main text | 1–42 |
| References | 42–53 |
| Appendix A | 54–81 |
| Appendix B | 82–96 |
| Appendix C | 97–104 |
| Appendix D | 105–133 |
| Appendix E | 134–155 |
| Appendix F | 156–175 |

If further cuts are wanted:
1. Merge Figure 4 (experiment) into Figure 3 once m9 is done. This reaches R3's and the plan's exhibit budget of 9 or fewer.
2. Shorten economics V.C (the OpenAI sentence and the two fleet adjustments), saving about 80 words.
3. Cut the framework's Besiroglu worked example (w ≈ 5.2), saving about 90 words.
4. Cut the Hicks/Allen/Morishima sentence and the Gopher sentence, saving about 50 words.

## 8. TBD-m9 placeholders (red), in reading order

1. `introduction.tex`, end of "The technology": a one-sentence headline (≤40 words).
2. `technology.tex` III.E: one merged block. It covers completed endpoints; the dated power calculation; Q1 (σ\* by corpus under both counts); Q2 (tilt on three validation sets, with the decision); Q3 (the sign of the extrapolation slope); and Q4–Q6 (seed noise, learning-rate excess loss and the bound on a, on-path versus full-grid profile).
3. `technology.tex`, Figure 4 (`fig:experiment`): two panels and their notes. The figure file is expected as `output/figures/m9_*`.
4. `wedge.tex` IV.D: the Q3 result (fill X, the maximum non-embedding M, and Y, the factor).
5. `conclusion.tex`: paragraph 1 (what the experiment adds) and Limitations (the high-M test).
6. `appendix_data.tex` B4 (Hardware): completed endpoints, failed runs, wall-clock hours.
7. `appendix_mc.tex` (Noise): the measured seed standard deviation of ln L.
8. `appendix_observational.tex` E5: the M\* shift between corpora, which signs the Hicks bias. Delete it if m9 cannot answer.

If m9 contradicts "about 0.7", the abstract, introduction and conclusion must change together.

## 9. Remaining issues (for the lead author)

1. **Authorship.** The byline lists Claude, as the user requested. R3 E1 and R4 M9(d) note that journals following COPE do not accept AI authors. This is the user's decision.
2. **Ho et al. critique (R3 E5).** Whether to share it with the original authors is the user's decision. The paper makes no statement about it.
3. **Anchor intervals for Prop. 5 in IV.D (ra5 review §7.7).**
   - The 86% share is robust to ra5's wider anchors (§4 above).
   - The set [2.3, 89] for M\*(10²⁴), and Meta's [10.1, 57], use ra2's narrower anchors.
   - ra2 should rerun `pi_anchors` with between-budget residuals (wild bootstrap-t) and update IV.D and Appendix F, or IV.D should add a clause. This needs an analysis rerun, not integration.
4. **Result count.** The main text has six formal results (Props 1–5 and Lemma 1). The plan allows this; revision plan A said five or fewer.
5. **Exhibit count.** There are 10 main-text exhibits, one a placeholder, against 9 or fewer in the plan and 7 in R3. See §7, item 1.
6. **Figure legibility in print.**
   - Figure 7 (`ra3_econ_figure`) and Figure 3 panel b have fonts of about 5 pt.
   - Figure 1 is unchanged from v1. R2 minor 26 asks for a non-embedding-N marker.
   - These need regeneration by the figure owners.
7. **Unplaced ra3 appendix exhibits.** The compute-growth definitions table, the inference-disclosures table and the inference-share figure are not placed. Section V's footnote carries the robustness. Appendix F could host them.
8. **Farseer σ\* in the random-effects summary.** Farseer's first-derivative estimate (0.708) is in Table 1, but the random-effects summary (0.695) uses the Hessian-based 0.703. Both are disclosed. ra1 open item 9, a standard error that does not treat compute levels as independent, is still open.
9. **Module table.** `output/tables/m1_chinchilla_horse_race.tex` still reports M\* standard deviations (R4 minor 43). The paper does not use it. Regenerate or delete it.
10. **Pre-analysis plan header.** The header says 03:15; the commit time is 03:12. Appendix B states both. The typo in the header itself is not fixed.
11. **Not done, and stated as such in the appendices** (D/F writer list):
    - a Raval second-margin test;
    - byte normalization of D;
    - durability of T;
    - rebuilding Chinchilla's D;
    - the m9 extensions R1 and R2 asked for: dose–response mixtures, 100–300M anchors, M ≥ 10⁴ in total parameters.
12. **`aea.bst` edited** to fix the editor comma bug (§1). It must be kept if the AEA template is refreshed.
13. **Nothing was committed to git.** The working tree has the changes, and the backups are in the session scratchpad.
