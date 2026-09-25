# Independent review: module rb3_econ2 (Section V integrated with Section IV)

Reviewer: independent replicator and skeptical referee (Claude), 2026-09-24. The binding plan is `paper/notes/revision_plan_v3.md` §2 ("Section V integrates Section IV"). The builder's memo is `output/memos/rb3_econ2.md`. I revised it in place and marked the additions "(review addition)" or "[review]".

**Verdict.** The module replicates exactly and its computations are correct. I rebuilt the three central calculations independently and they agree to the last reported digit:
- compute growth by database vintage;
- the wall solver at observed allocations;
- the scenario arithmetic.

Two substantive problems were in the *reading* of the numbers, not in the code, and I fixed both in code, exhibits and memo:
1. **The "understatement" statistic mixes two effects.** ŵ/(1 + m_N) is the correct Proposition 1(iii) object. But much of its fall comes from 1 + m_N = 1 + Φ/(6D) *rising* when the capped developer processes fewer tokens. The trend prediction ("the measured trend flattens") needs ŵ/w, the measured wedge relative to the wedge the same developer would reveal without the wall. At D/U = 4, ŵ/(1 + m_N) is 0.92 / 0.67 / 0.14 (R\*_D = 15.4 / 2.9 / hard cap) while ŵ/w is 0.96 / 0.85 / 0.47. On the rising-wedge frontier path the flattening is 5% (2028) and 15% (2029), not 9% and 30%. The prediction survives, but at about half the size.
2. **The catch-up target and the fleet shares rest on one family.** The 2025 compute-weighted aggregate wedge (5.39) comes from seven decision units, and Alibaba's Qwen3, a cap-reading *lower bound* in rb2, carries most of their compute. Without Alibaba the 2025 aggregate is 3.13:
   - the "plausible" five-year catch-up demand falls from 172/320/469× to 114/170/216× (σ\* = 0.60/0.70/0.74);
   - one-year fleet shares fall from 0.17–0.33 to 0.09–0.19.

   Both are now reported.

One referee comment the builder listed as addressed was not: R4 minor 9. The Table 3 notes did not flag OLMo or give the model-free range. It is fixed now. The remaining fixes are minor: labels, exact grid points, citations, and the figure output path.

---------------------------------------------------------------------------------------------------

## 1. Replication

- **From scratch, before any change.** `.venv/bin/python code/analysis/rb3_econ2/run.py` with `RB3_OUTPUT_ROOT` set to a scratch folder took 4.6 s on the CPU, in one process. It reproduced all 32 `output/tables/rb3_econ2_*` files and `summary.json` byte for byte.
  - Side finding: the figure ignored `RB3_OUTPUT_ROOT` (`aer_style.savefig` defaults to the project folder), so a "scratch" run overwrote the project figure. Now fixed.
- **After the fixes.** Two further scratch runs and the project run are byte-identical: 33 table files (one new CSV) and `summary.json`.
- **ra3 and rb2 untouched.** ra3's output timestamps are unchanged (07:41 today). Its sandbox contains only empty directories, and rb2's processed files are only read.
- **Checks.** The wedge checks reproduce rb2's by-year aggregates exactly (6/6). The 13 wall checks pass, with maximum deviations of:
  - 3.4×10⁻⁷ for the closed-form optimum;
  - 1.4×10⁻⁹ for scale invariance;
  - 4.3×10⁻⁷ for μ = 1/η − 1.

### Independent re-implementations
- **Compute growth by vintage** (own running top-10 and OLS, `scratchpad/rb3rev/vint.py`). May 2024 file: 4.131×, 58 runs. September 2026 file, same window: 5.226×, 78 runs. 2018–2026: 5.062×, 92 runs. All three match the module.
  - A variant top-10 rule counts same-date rows on both sides, where ra3 counts only earlier rows in the sort order. It gives 5.20 and 5.04, so the rule is immaterial.
- **The wall at observed allocations** (own solver: bracketing for the no-wall wedge, a dense grid in ln N plus Brent, explicit corner for the hard cap; `scratchpad/rb3rev/wallcheck.py`). At six points the life penalty, ŵ/(1 + m_N), ŵ/w, epochs and N_c/N_w agree with `obs_point` to 5 digits:
  - (w, D/U, spec) = (5.39, 8, D15), (1.21, 3.46, D15), (5.27, 19.2, D15), (5.39, 4, D3), (5.39, 4, hard cap), (1, 4, D15).
- **Scenario arithmetic by hand.**
  - e(0.70) = 1.1667 and 1.91^1.1667 = 2.128; × 5.06^0.496 = 4.76; ^5 = 2,441.
  - Catch-up: (5.39/1.211)^1.1667 = 5.71, and 56 × 5.71 = 320. The catch-up takes ln 4.45 / ln 1.91 = 2.31 years.
  - Fleet: φ = (1 − e^(−gL))/(gL) = 0.495 / 0.296 / 0.204. The share 2.173/(2.173 + 4.4) = 0.331.
  - γ: 2^(1/0.1654) = 66.0, and ln 66 / ln 5.06 = 2.58 years.
- **The data-multiple identity.** Since ln w = k ln(M/M\*) and D/D\* = (M/M\*)^(1/2) at given compute, D/D\* = w^(1/2k) = w^(σ\*/[2(1−σ\*)]). The code uses this relation consistently (`trend.py`, `scenario.py`, `wallobs.uncapped`).

### Primary sources (web evidence)
| Item | Check | Result |
|---|---|---|
| Epoch file, 31 May 2024 (`epoch_all_systems_20240531.csv`) | Internet Archive CDX for `epochai.org/data/epochdb/all_systems.csv`, May–June 2024 | One capture, 20240531034617, digest `W5ISMX45PMW2CMS2ZU65D2USZOKGNO74`. The SHA-1 (base32) of the saved file is **identical**, and the sha256 matches the download script. The latest "Last modified" in the file is 2024-05-29 10:55 UTC. No row is flagged in the `Exclude` column. |
| Epoch's published rate | Live page epoch.ai/blog/training-compute-of-frontier-ai-models-grows-by-4-5x-per-year | Confirmed: "4.2 x/year (90% CI: 3.6x to 4.9x) after 2018" for models "in the top 10 of training compute when they were released". The post is dated 28 May 2024, excludes AlphaGo Master and Zero, and uses piecewise models with a break. |
| DeepSeek API prices | Saved HTML (sha256 matches) and the live page, both on 24 Sep 2026 | Confirmed: deepseek-flash (DeepSeek-V4.1-Flash) $0.6 / $1.2 per 1M output tokens off-peak / peak; deepseek-v4-pro (V4-Pro-0813) $1.98 / $3.96. |
| Reddit S-1 (memo only) | SEC filing and press coverage | Confirmed: "over one billion posts and over 16 billion comments"; aggregate contract value $203.0 million, terms of two to three years. |
| Vintage examples cited in the memo | Both Epoch files | Confirmed: Mistral Large 2×10²⁵ → blank; FLAN 137B 4.9×10²² → 2.05×10²⁴; GPT-4 (Mar 2023) matched by (date, org); GPT-4 (Jun 2023) and Aramco Metabrain added. Nuance added to the memo: the GPT-3.5 (text-davinci-003) estimate was *moved* to a separate "GPT-3.5 (davinci-002)" entry dated March 2022, which the decomposition counts as a withdrawal plus an addition. |

---------------------------------------------------------------------------------------------------

## 2. Audit of every memo number

Every number in H1–H7, the claims and the robustness section was checked against the regenerated CSVs. All matched to rounding except the items below, which were corrected:

| Where | Builder | Correct | Why |
|---|---|---|---|
| H2 anchor | "Llama 3.1 405B 1.29, Nemotron-4 1.10, Pangu Ultra 1.92, Aramco 1.13" called D/D\* | D/D\* = 1.35, 1.12, 2.15, 1.15 (those were the *wedges*) | mislabeled |
| H3(d) table, hard cap | 0.59 / 0.40 / 0.13 | 0.58 / 0.38 / 0.14 | the exhibit looked up the *nearest* grid point: "D/U = 2" was 1.94, "8" was 8.25, and so on |
| H3(d) table, R\*_D = 2.9 | 0.84 / 0.50 | 0.83 / 0.49 | same |
| Claim 4, robustness | "87% under a hard cap"; "0.40 at D/U = 2" | 86%; 0.38 | same |
| H3(a) column labels | r = 0.7, 1.2 | r = 0.71, 1.19 (grid 2^(k/4)) | labels |
| H3(a) crossing | "r ≈ 20" | between r = 19 and 23 (w = 5.39 vs 1) | precision |
| H5 drift | −0.001 per decade | −0.0006 | rounding hid the size |
| H4 model-free range at 4.2× | 2.05–2.53 | 2.04–2.53 (Chinchilla path 2.0446) | rounding |
| H6 | "at p = 2, 0.29–0.50" after a 4.2× sentence | those are 5.06× values (at 4.2×: 0.31–0.51) | ambiguity |
| H3 title | "5–10% of lifetime cost by 2029" | "5–10% … if repeated data decay slowly (23–48% under the data-only fit)" | quoted only the favourable repetition model, which the ra3 review warned against |
| H5 point 2 | E-profile "independently" confirms E | "separately" (same 240 runs, different identifying variation) | wording |

Verified without change, among others:
- g_w = 1.91 [1.03, 3.48] and every estimator row of H1;
- leave-one-developer-out 1.73–2.30;
- the 32 technologies (1.93–2.33 for the data multiple, 1.16–4.12 for w);
- all of Panel B;
- wall today: 0.49–1.09; at 22T, 1.8–8.6% (D15) and 3.3–32% (full model);
- the frontier path (2028–2030, three repetition models; $2bn and $21bn);
- shadow values;
- the vintage decomposition (64/36; 145/54/21/978);
- the γ table and E-profiles;
- fleet shares;
- the synthetic break-even values (1.47–2.86 at the frontier, 2.12–4.62 at 10²⁶, 3.48–4.79 for self-generation).

---------------------------------------------------------------------------------------------------

## 3. Issues found and what was done

### Major
1. **The trend prediction was sized with the wrong ratio (fixed).**
   - *The problem.* In `wallobs.obs_point`, `ratio_proc` = ŵ/(1 + m_N,c) with m_N,c = Φ/(6D_c). Under a cap the developer processes fewer tokens (D_c < D_w), so 1 + m_N,c > w: in 2029 on the rising path it is 6.42 against w = 5.27. The ratio is the Proposition 1(iii) object, and it is right to report it. But the memo used it to size "the measured trend should flatten". The flattening is ŵ(t)/w(t), where w(t) is what would be measured without the wall.
   - *Fix, code.* `wallobs.understatement_grid` and `frontier_path` now keep `ratio_proc_vs_w` (= ŵ/w) and `w_hat_proc`.
   - *Fix, exhibits.* `app_wall` Panel B shows "ŵ/(1 + m_N) / ŵ/w" in each cell. The Table 3 note gives ŵ/w on the rising path: 0.95 in 2028 and 0.85 in 2029. Figure panel (c) adds ŵ/w as dashed lines.
   - *Fix, memo.* H3(d) and Claim 4 carry both ratios, a wording instruction, and two new facts:
     - under R\*_D = 2.9 the measured wedge can *fall* on the catch-up path (2.94 → 2.09 in 2029–2030);
     - with the wedge held it drops below 1 by 2029 (0.91).
2. **The catch-up target and fleet m depend on Qwen3 (sensitivity added).**
   - *The problem.* 2025 has seven decision units. Leaving out each 2025 developer in turn, the 2025 aggregate runs from 3.13 (without Alibaba) to 7.24 (without Swiss AI). Qwen3 is itself a lower bound under rb2's cap reading.
   - *Fix, code.* `scenario.py` writes `rb3_econ2_scenarios_catchup_sensitivity.csv`, with the target, one-off multiple, catch-up years and five-year demand at both growth rates. `fleet.py` adds the "2025, decision units excl. Alibaba" rows (m = 2.13).
   - *Fix, exhibits and memo.* The Table 3 note (a) states the no-Alibaba target of 3.1. The memo's H2, H6 and Claims 1 and 8 carry the ranges.

### Moderate
3. **R4 round-2 minor 9 was not actually addressed in the table (fixed).** The memo said Farseer and OLMo were "flagged in the notes", but the Table 3 notes mentioned only Farseer's local slope. They now say that:
   - Farseer's slope lies beyond its design;
   - OLMo's a has a bootstrap interval of [0.11, 0.82];
   - the model-free IsoFLOP paths (a = 0.35–0.50, Porian excluded as in Section III) give 2.04–2.53× (4.2) and 2.24–2.86× (5.06).

   All three are generated from `dem_all`, not hard-coded.
4. **Exact grid points for the understatement table (fixed).** Panel B used the nearest point of a 41-point geometric grid. The grid now includes 1.5, 2, 3, 4, 8, 16 and 32 exactly, and the exhibit asserts an exact match.

### Minor (all fixed)
5. **Citations.**
   - The new key `epochai2024dataarchive` was not cited in any exhibit. `app_growth` now cites it, together with `epochai2026data` and `sevilla2024training`; before, it hard-coded "Sevilla and Roldán 2024" in the row label.
   - The decomposition line "ln(5.23/4.13)" was hard-coded. It is now generated.
   - All four exhibits compile in AEA.cls against `references.bib` plus the extra bib with **no undefined citations, no overfull boxes and no float overflow**. Table 3 still fits one page; its notes were rewritten to absorb the additions, so there is little room left.
6. **Figure output path.** The figure now honours `RB3_OUTPUT_ROOT`. The panel (c) legend has an opaque frame, and the ylabel names both ratios.
7. **Table 3 notes.**
   - Panel A's "100T reached" is stated to be at 5.06×.
   - Panel B's σ\* rows are explained: they hold the growth of the value of compactness fixed, while extrapolating tokens per parameter gives the σ\* = 0.70 row whatever σ\*.
   - Panel C's D is defined as the tokens processed at the projected multiple, without the wall.
   - The full-model definition is kept.
8. **Docstrings and memo text.**
   - Runtime: "1–2 minutes" is now 5 s.
   - `N_PROC`: "the wall grid uses a pool" is now "single-process".
   - `synth.py`: the phrase "upper bounds" is clarified; the break-even scarcities are lower bounds.
   - The memo states that the API price is one provider's low-end price.
   - The memo gives the figure's band definition (inherited from ra3: 22–490T, growing 0–10% a year).
   - Panel B is labelled as notional (no-wall) demand.
   - The σ\* half of R3 minor 20 is assigned to rb1_sigmaC.
   - A duplicated conditional was removed in `obs_point` (`Dpc`).

---------------------------------------------------------------------------------------------------

## 4. Referee comments: were they addressed?

| Comment | Status after review |
|---|---|
| R3 N6(1): rising-wedge scenario in Table 3, σ\* 0.60/0.70/0.74 | **Yes.** Panel B, with the catch-up variant, both growth rates, and the σ\*-free reading. R3's own arithmetic is reproduced: w = 3 gives D/D\* = 4.77 at σ\* = 0.74 and 2.28 at 0.60; the clean-sample aggregate gives 2.26×/yr. Qwen3 sensitivity added. |
| R3 N6(2): the wall at observed allocations; the understatement prediction | **Yes, after fix 1.** Observed-allocation costs are reported today, along the path, and by (w, r). The prediction is now sized with ŵ/w, alongside the Prop. 1(iii) ratio. |
| R3 N6(3), minor 19: service life 1/2/3 years; keep it short | **Yes.** Moved to the appendix with one sentence; the Qwen3 caveat is added. |
| R3 minor 17: 4.2× and 5.06× | **Yes, and reconciled.** 4.13× on the archived May 2024 file (verified capture), with a 64/36 revisions/additions decomposition. |
| R3 minor 18 (optional): $3/M beside an observed price | **Yes, partly.** A posted API price for generated tokens and the cost of self-generation. No per-token licensing price, which is correctly declined. |
| R3 minor 20: which γ at 10²⁵–10²⁷, and why | **Yes** for γ (0.165, 66×, four reasons, the range, the total-loss caveat). The σ\* part is rb1's. |
| R3 N5 (part): say where σ\* matters | **Yes.** It converts value into data, and it matters for wall costs under the data-only fit (51/40/35% at r = 4). |
| R1 minor 17: shorten Section V; fleet to the appendix | **Yes.** |
| R1 minor 18: R\*_D sensitivity next to σ\* | **Yes.** Table 3 Panel C. |
| R2 minors 14 and 20: E-profiles; units; compute window | **Yes.** |
| R4 minor 9: flag Farseer and OLMo, or give the model-free range | **Now yes** (fix 3). The builder's claim was premature. |
| R4 minor 10: lead with κ free (66×, not 49×) | **Yes.** |
| R2 Major 1 (σ\* ≈ 0.60 at the top budgets) as it bears on Section V | **Yes.** σ\* = 0.60 is carried through Panels B–C. |

---------------------------------------------------------------------------------------------------

## 5. Remaining concerns (not fixable within this module)

1. **The σ\* rows of Panel B mix measurement and conversion.** g_w is measured under the reference σ\* (0.70). If σ\* = 0.60 were true, the measured value trend would differ (1.16–4.12 across technologies), and the implied data trend would again be the σ\*-free 2.13. The rows therefore answer R3's question: *given* a value trend, σ\* converts it into data. They are not a forecast uncertainty band. The notes now say so; the text should too.
2. **The σ\* ordering of the compute-optimal wall is not robust.** It is 7.7/7.4/7.2% for κ-free equivalents and 7.2/7.4/7.7% for κ = 1 equivalents (ra3). Only "σ\* barely matters under slow decay" is robust, not its direction.
3. **A short, thin trend.** The trend covers two years (19/30/7 units). Its 95% CI runs from 1.03 to 3.48, and the 2025 level rests on Qwen3, a lower bound. Extrapolating an open-weight small-model trend to the frontier (w_F = 1.21, four disclosed runs) is a scenario, not a forecast.
4. **Repetition parameters and the value of compactness.** Muennighoff et al.'s R\* come from runs of at most 9B parameters. Wall costs vary four- to fivefold across repetition models, and the understatement prediction reverses near compute-optimality under the full model. Φ linear in N is an assumption.
5. **The γ recommendation.** It rests on a 2022 recipe (MassiveText) and extrapolates 3–5 decades. Loss is not capability, and reducible loss is about 4% of total loss at 10²⁶ FLOP.
6. **The synthetic-token comparison.** It uses one provider's price on one date; prices change, and the saved page is the record.
7. **Epoch's piecewise estimator is not replicated.** Our fixed-2018 OLS gives 4.13 against their 4.2, so the gap is small.
8. **Integration tasks for the writer.**
   - Table 3's structure changes: the old inference-share panel is dropped, so Section V.C must be rewritten around the fleet sentence.
   - Merge `epochai2024dataarchive` and `deepseek2026pricing` into `paper/references.bib`.
   - Section V.D should lead with 66×.
   - Table 3's notes now fill the page; any further additions need cuts elsewhere.

---------------------------------------------------------------------------------------------------

## 6. Files changed by the review
- **Code** (`code/analysis/rb3_econ2/`):
  - `wallobs.py`: exact understatement points; ŵ/w kept in the understatement grid and the frontier path; cleanup.
  - `scenario.py`: catch-up target sensitivity.
  - `fleet.py`: m excluding Alibaba.
  - `exhibits.py`: Table 3 notes; `app_growth` citations and generated decomposition; `app_wall` Panel B; figure panel (c) and output folder.
  - `run.py`: new CSV; runtime docstring.
  - `synth.py` and `rb3common.py`: docstrings.
- **Outputs regenerated:** all 33 `output/tables/rb3_econ2_*` files (new: `rb3_econ2_scenarios_catchup_sensitivity.csv`), `output/figures/rb3_econ2_figure.pdf/.png` and `data/processed/rb3_econ2/summary.json`.
- **Memo:** `output/memos/rb3_econ2.md`, revised in place.
- **Not changed:** ra3, rb2, `sl.py` and the bib file (its entries were verified).
