# Independent review: module ra3_econ (economic implications, Section V)

Reviewer: independent replicator and skeptical referee (Claude). Date: 2026-09-24.
Scope: R3 M7 (1)–(3) and its final paragraph; R3 minor 29; R1 6(e); R2 M3 / R3 M3(c); R2 23.
Inputs audited:
- code: `code/analysis/ra3_econ/*.py`;
- outputs: `output/tables/ra3_econ_*`, `output/figures/ra3_econ_*`, `data/processed/ra3_econ/`;
- memo: `output/memos/ra3_econ.md`;
- bib: `lit/bib/extra_ra3_econ.bib`;
- archived PDFs: `data/raw/ra3_econ_lit/`.

## 0. Verdict

**The module is now usable for Section V.** The builder's economics are sound, and the two numerically hardest parts replicate independently:
- the growth of frontier compute (wild cluster bootstrap);
- the data-wall optimization (dual problem, shadow values, hard-wall closed forms).

Four problems needed fixing:
1. The inference-share panel was the preliminary version (m3's universe under the rejected κ = 1 reference), although ra2's final reviewed inversion is now available.
2. The Monte Carlo for dates anchored at observed M was internally inconsistent.
3. The data-wall headline ("a binding cap is cheap, about 7%") rested on a repetition specification that the source paper never fitted.
4. The appendix disclosures table would not have compiled.

All four are fixed, the module was re-run from scratch, and the memo was updated. After the fixes, two runs give byte-identical CSV/TeX/processed outputs (33 files), in about 55 s on at most 5 processes.

## 1. Replication

- **Fresh rerun of the builder's code.** All builder outputs were backed up and deleted, then `run.py` was re-run. It reproduced every builder CSV/TeX byte for byte, except `ra3_econ_inference_share*.csv`. That file changed only because ra2 finalized after the builder ran: ra2's 2023 envelope minimum moved from 0.0080 to 0.0069.
- **Frontier compute growth (independent code).**
  - Method: my own running top-10 (ties handled differently), statsmodels CR1 and my own WCU loop (Webb weights, B = 9,999).
  - Result: 5.04× [4.26, 5.96], n = 91, 34 unmapped developer clusters. The builder has 5.06× [4.29, 5.93] (n = 92, 27 parent-mapped clusters).
  - The one-run difference is a same-date tie (GPT-3 6.7B rerun). **Pass.**
- **Data wall (independent code).**
  - Method: minimize cost over ln D, with N solved from the loss constraint and effective parameters inverted for the full model. The builder minimizes over ln N.
  - κ = 1: penalties 1.53 / 7.22 / 21.4 / 58.4 / 187% at r = 2 / 4 / 8 / 16 / 32. κ free: 1.56 / 7.38 / 22.2 / 63.4 / 233%.
  - Other rows: full model (N′ decay) 43.4% at r = 4; data-only fit 35.6%; no repetition 330%; shadow value at r = 4 is 3.14×10¹² FLOP.
  - All agree with the builder to the reported digits. The r_max closed forms were re-derived analytically and agree: D′-only (1 + R\*_D)(S/a₁)^(1/b₁) and the D′ + N′ form. The envelope formula for the shadow value was re-derived: λ = 6N[(1 + R\*)(e^(R/R\*) − 1) − R]. **Pass.**
- **Other formulas re-derived.**
  - Frontier constant K = [(S/b₁) A G^(−a₁)]^κ and γ = κ a₁b₁/S.
  - Equivalent-family construction.
  - Wedge-to-data multiple D/D\* = w^(1/S) = w^(σ/(2(1−σ))).
  - GATE slope m = |d ln I/d ln C| along an isoquant = 1/(w − 1).
  - Fleet vintage factor φ = (1 − e^(−gL))/(gL).
  - Exhaustion-date closed form.
  - The bootstrap-file column orders (m1 pairs: lnA, lnB, lnE, α, β; m2 κ-free: E, A, B, α, β, q) were checked against the registry points. **All pass.**
- **Literature numbers**, verified against the archived PDFs or source pages:
  - Villalobos et al. 2024: 100T [22T, 490T]; 320T [65T, 1,700T]; median 2028 at about 5×10²⁸ FLOP; 5× over-training = √5 data, one year earlier.
  - Muennighoff et al. 2023, App. A and Table 1: R\*_D = 15.387756, R\*_N = 5.309743; data-only fit R\*_D = 2.9157; U_N = compute-optimal N for U_D.
  - Sevilla and Roldán 2024: 4.1× [3.7, 4.6], 5.3× [4.9, 5.7], 4.2× [3.6, 4.9], 5.0× [80%: 3.1, 7.3].
  - Patterson et al.: "about 3/5 … inference". Wu et al.: 10:20:70 and LM 65/35.
  - GATE: m ≈ 1–2 (parameter range 1–4).
  - Epoch OpenAI spend: $1.8B inference, about $5B R&D, $386M + $83M final runs.
  - Denain and Wu: 9.6 / 12.3 / 22.6%.
  - Reddit S-1: $203.0M, 2–3-year terms.
  - Nvidia: about 40% of data-center revenue for inference, stated by the CFO on the 21 February 2024 call. It is **not** in the FY2024 10-K or the CFO commentary exhibit, so a claim I had added to that effect was removed. **Pass.**
- **Citation keys.** Every key cited in the memo exists in `paper/references.bib` or `lit/references.bib`. The six new keys in `extra_ra3_econ.bib` are real sources. patterson2022carbon and wu2022sustainable duplicate keys already in `paper/references.bib` (consistent content); a comment was added so writers do not merge them twice.

## 2. Issues found and fixes

| # | Severity | Issue | Fix |
|---|---|---|---|
| 1 | Major | **Panel C, H6–H7 and the GATE/H5 inputs used the preliminary inference share**: m3's 271-model universe under m3's κ = 1 reference (a restriction ra2 rejects), with a six-technology band. ra2's final reviewed outputs were only merged as side columns. | `share.py` was rewritten on ra2's final outputs: ra2's clean open-weight universe (141 models, rebuilt by calling ra2's sample code read-only), the κ-free reference, ra2's design-conditional wild bootstrap (B = 399; ra2's function and seeds; draws identical to ra2's to 4×10⁻¹⁶), a κ = 1 comparison, ra2's clean verified sample (77) with a technology-consistent range over ra2's 32 ex-ante technologies, and ra2's per-model envelope. 16 exact reproduction checks of `ra2_wedge_aggregate.csv` are written and enforced; `run.py` stops on failure. The builder's version is kept as `ra3_econ_inference_share_m3universe.csv` (superseded). New: s = 0.04 / 0.35 / 0.54 / 0.82 for 2019–22 / 2023 / 2024 / 2025 (the builder had 0.07 / 0.35 / 0.54 / 0.76). |
| 2 | Major | **The exhaustion MC under observed-M anchoring was inconsistent.** The anchoring multiple m_D = median D_i/D\*(C_i) was fixed at its point value while technology draws moved the level c₀. That re-introduced exactly the level uncertainty the anchoring removes and inflated the intervals. | m_D is recomputed for every technology draw. The technology component of the Chinchilla 100T interval falls from 1.69 to 0.45 years, Gadre RW's from 5.7 to 1.4 and OLMo's from 25 to 6.4. The total Chinchilla interval changes from [2025.5, 2029.5] to [2025.7, 2029.4], because the stock dominates. |
| 3 | Major (claims) | **The data-wall headline was framed on the most favourable case, which is not a fitted model.** The primary uses Muennighoff's jointly fitted R\*_D = 15.4 without their parameter decay. Muennighoff et al.'s own D-only fit has R\*_D = 2.9 (Table 1, lower R²), and their preferred full model implies 43% at r = 4 and a hard wall at r ≈ 9. The memo's Claim 4 led with "cheap … about 7%". | Claim 4 and H4 were rewritten: the full-model number sits alongside the D′-only one, with an explicit "do not quote 7% alone". Panel B's robustness rows were relabelled as Muennighoff's full model and data-only fit, and the notes say the primary is the most favourable case. The σ\* comparison R3 asked for is reported under both models: 7.2 vs 7.4%, and 43.4 vs 40.3%. |
| 4 | Major (exhibit) | **`ra3_econ_inference_disclosures.tex` would not compile.** "R&D" was unescaped inside a tabular (an extra column), and the "This paper" rows ended in `\\\\`, which gives empty rows. | Added `_tex()` escaping, fixed the row endings and en-dashed the periods. All three .tex files pass a column-count check (no TeX engine is installed here, so they were not compiled). They use `tablenotes`, not threeparttable. |
| 5 | Minor | The observed-M anchor included LongCat-2.0, a MoE model counted at its active N (M = 729). D/D\* under a dense technology is not comparable there (R1 minor 24; ra2 drops MoE). | Excluded by default (n = 4 dense runs). The with-MoE sensitivity is kept (`*_withMoE` columns). Chinchilla's m_D goes from 1.55 to 1.43; the dates move about 0.1 year later. |
| 6 | Minor | The ra1 auto-glob (any `ra1_modelfree_*.csv` with an `a` column) was a silent no-op. It would also have crashed on string-valued `a` columns (pairwise tables) had they carried a label column. The builder's open issue 2 was unresolved. | Replaced by explicit model-free Approach-2 path slopes from ra1's reviewed per-budget argmins (valid budgets only): 0.50 (Chinchilla), 0.50 (Llama 3, vs Meta's 0.46), 0.35–0.43 (Marin), 0.50–0.51 (Porian). These are growth-only rows. |
| 7 | Minor | "Scale invariance at machine precision" was overstated: the maximum difference is 2.5×10⁻⁷ (optimizer tolerance). | Wording corrected. |
| 8 | Minor | The fleet mapping used frontier compute growth (5.06×) as the vintage growth of all served models. Aggregate training compute probably grows more slowly. | Added illustrative 3× and 2× vintage-growth rows. The 2025 fleet shares rise from 0.34 to 0.42 at ρ = 4.4. |
| 9 | Minor | H5 plugged the aggregate multiple ("w = 4.1") into the convex per-model map D/D\* = w^(1/S) without comment. | Updated to ra2's aggregates (1 + m = 2.15 for 2024, 5.53 for 2025) and labelled illustrative (Jensen). A caveat was added that w is inferred from M with a given σ\*. |
| 10 | Minor | R3 M7(2) names M\* but the wall section never quantified M\*'s role. | Added: at today's frontier and the 100T stock, compute-optimal r is 0.09 (Gadre RW) to 5.2 (OLMo). M\*, not σ\*, decides whether the wall already binds; at observed data use frontier runs sit at r ≈ 0.6–1.2. |
| 11 | Minor | The GATE m mapping was presented as a calibration. GATE's m concerns runtime compute per task; ours is per-token serving cost at equal loss. | Caveat added. m is now 0.87 / 0.22 / 0.56 at ra2's 2024 / 2025 / pooled aggregates. |
| 12 | Minor | Figure overlaps: panel (a)'s legend covered data, and the unique-stock label was struck through by its own line. In the share figure, the legend covered the 2024 technology bar and labels crossed bars. | Relaid out; PNGs inspected. At most 3 colors plus gray per panel. |
| 13 | Minor | H6 robustness statements ("dedup ≤ 0.01", "LOO 2024 within [0.53, 0.60]") referred to the superseded sample. | Replaced: ra2's universe has no duplicate runs; LOO 2024 gives 0.54 → 0.63 without Llama 3.1 405B (0.58 → 0.73 on the clean sample). |
| 14 | Note | Calling ra2's `sample.build()` writes ra2's audit file. One exploratory call during this review rewrote `data/processed/ra2_wedge/audit_flags.csv` with **byte-identical** content (verified against the redirected copy). | The write is now redirected to `data/processed/ra3_econ/ra2_sample_rebuild/`. ra3 never writes into ra2's directories. |

## 3. Numbers checked against regenerated outputs

Every number in memo sections H1–H8, 4 and 6 was checked against the final CSVs:
- H1: all six samples, p-values, trend level, 19/92 speculative.
- H2: a, growth per year and over five years at 5.06× and 4.5×, bootstrap CI of growth, D\* and M\* at the frontier.
- H3: compute-optimal, observed-M and OT5 dates, MC quantiles by source.
- H4: penalties, r_max, shadow values, σ_CU, γ_eff/γ, bootstrap intervals, dollar scenarios, $/FLOP.
- H5: data multiples.
- H6: shares, CIs, κ = 1, clean sample, technology range, envelope, LOO, compute concentration.
- H7: disclosures and fleet shares.
- H8: 2^(1/γ), CIs, years per halving, GATE slope.

The table and figure were regenerated from the same run. Corrections made while checking my own new text: "CI 5–10× narrower than the range" became 3–7×; the high end of the technology range is Gadre RW κ = 1 (2023) / Muennighoff κ-free (2024–25), not "Gadre κ-free"; the 2019–22 composition is OPT-175B + GLM-130B = 92% of compute.

## 4. Referee comments: are they addressed?

| Comment | Status |
|---|---|
| R3 M7(1): data demand, a, exhaustion | **Addressed.** Own compute-growth estimate with few-cluster inference; ten technologies plus ra1 slopes; one-year and five-year growth; exhaustion dates (compute-optimal, observed M, 5× over-training) with a consistent MC; R3's 25× vs 114× reproduced. |
| R3 M7(2): σ\*, M\*, data wall, shadow value, economics of data | **Addressed.** Cost of a cap and shadow value in FLOP and $ at a stated $/FLOP, under κ = 1, κ free and σ\* = 0.60, and under three repetition models. "0.74 vs 0.70 barely matters" is shown under both the D′-only and the full model. M\*'s role is now quantified. Jones–Tonetti nonrivalry and data licensing (Reddit S-1) are connected qualitatively. **Partial:** Farboodi–Veldkamp is only cited, and no per-token licensing price comparison is possible (corpus sizes undisclosed). |
| R3 M7(3): inference share as a measurement | **Addressed, with a binding caveat.** Now built on ra2's final inversion, with a design-conditional CI and the range over 32 technologies. The 2025 range is [0.24, 0.96], so the level is not robust; the rising trend is. The external comparison is loose and becomes a disagreement once R&D compute is counted. The link to Demirer et al. (2025) is a citation only. |
| R3 M7, final paragraph (γ in growth calibration) | **Addressed** (2^(1/γ) = 49–161×; years per halving; GATE m with caveat). |
| R3 minor 29 | **Addressed** (γ_eff/γ = 0.95 / 0.82 / 0.71 at r = 4 / 16 / 32). |
| R1 6(e) | **Addressed.** σ_CU (compute vs unique data) = 0.28–0.58, set by repetition and nearly invariant to σ_ND; γ is the growth-model elasticity; σ_ND enters via r_max and the inference-to-data map. σ_CU is model-implied, not estimated, as the memo says. |
| R2 M3 / R3 M3(c) (aggregate vs Patterson/Wu) | **Addressed as far as public data allow** (H7 plus the fleet mapping), jointly with ra2. |
| R2 23 (dedupe; LOO) | **Addressed** for the share (no duplicates in ra2's universe; LOO reported). |

## 5. Remaining concerns (not fixed)

1. **The data wall is conditional on Muennighoff et al.'s repetition estimates** (models ≤ 9B parameters). The primary D′-only case is a lower bound on costs; frontier-scale R\* is unknown.
2. **The inference-share level is technology-driven** ([0.24, 0.96] in 2025 across ra2's 32 technologies), open-weight only, and dominated by a few large releases (the top model is 22–46% of period compute).
3. **Pairs bootstraps remain in the data-wall and path-slope intervals** (m1 B = 400; m2 B = 200), where the standards prefer design-conditional draws. These intervals are tiny next to specification uncertainty, so this was not changed. For the share, m2's pairs draws give wider upper tails than ra2's wild bootstrap (2024: [0.43, 0.71] vs [0.41, 0.64]).
4. **Extrapolation.** a and M\* are extrapolated four to five orders of magnitude beyond every design. Farseer's own-form a is the local slope of an extrapolated segment, and OLMo's a has CI [0.11, 0.82]. Exhaustion dates under compute-optimal allocation are therefore uninformative for some technologies, and anchoring at observed M rests on four dense runs.
5. **Epoch vintage.** Our 2018–May 2024 rate (5.23×) exceeds Epoch's published 4.2× for that window; this is unexplained, probably database vintage.
6. **Stock uncertainty is modelled crudely.** It is a lognormal fitted to Villalobos's 95% interval, with the stock growth rate independent of it.
7. **No TeX engine here.** The .tex files were checked structurally only; writers should compile once.

## 6. Files changed by the review

- Code:
  - `code/analysis/ra3_econ/share.py` (rewritten: ra2 final);
  - `demand.py` (MoE exclusion, MC re-anchoring, explicit ra1 slopes);
  - `run.py` (part order, part 3 on ra2, flow g-sensitivity, wedge multiples and GATE slope at ra2 aggregates, with-MoE sensitivity);
  - `exhibits.py` (Panel C, notes, TeX escaping, figure layout, summary).
- Outputs: all `output/tables/ra3_econ_*` and `output/figures/ra3_econ_*`, and `data/processed/ra3_econ/*`, regenerated.
  - New files: `ra3_econ_inference_share_{universe,clean,checks,m3universe}.csv`.
  - Removed: `ra3_econ_inference_share_ra2.csv`.
- Memo: `output/memos/ra3_econ.md` (review header; H2, H3, H4 caveat and M\* paragraph; H5–H7 rewritten; H8 GATE; methods; inventory; claims 3, 4, 7, 8; robustness; referee table; open issues).
- Bib: `lit/bib/extra_ra3_econ.bib` (comment on duplicate keys).
- Builder's original outputs: backed up in the session scratchpad (`ra3rev_builder_backup/`).
