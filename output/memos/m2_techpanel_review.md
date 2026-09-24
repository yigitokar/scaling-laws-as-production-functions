# Review — module m2_techpanel (independent replication and referee report)

Reviewer: Claude, as independent replicator and skeptical referee. Date: 2026-09-23.
Scope: code in `code/analysis/m2_techpanel/` (`run.py`, `m2_data.py`, `m2_est.py`, `m2_out.py`), all outputs, and the memo `output/memos/m2_techpanel.md`.

## 0. Verdict

- **The module is reproducible and mostly sound.**
  - A from-scratch re-run of the builder's code regenerated all 20 tables and every bootstrap draw file byte for byte (23.3 min).
  - Units, parameter counts, token counts and the estimators check out against the raw data and the upstream repositories.
- **The headline technology numbers stand.** These are the Chinchilla-form σ* (0.735–0.828), the rejection of q = 1 in every sweep, and σ ≈ 0.69–0.71 on Farseer with q free.
- **One reported result was wrong, and several claims went beyond the evidence.**
  - **Wrong result: the rank-one curvature finding.** The memo reported a "saddle-shaped, non-separable" log reducible-loss surface (Farseer τ = −0.26, Chinchilla τ = −0.08). This is an artifact of computing the translog with the Chinchilla Ê. With Ê from the q family, the surface is close to rank one.
  - **Claims that went beyond the evidence:** "far above the capital–labor range", the Lemma 1 reading of the DataDecide tilts, the local-σ slope, and the pointwise-band comparison.
- **All issues were fixed in code and memo, and the fixed code was re-run in full from scratch** (log: `data/processed/m2_techpanel/run_log.txt`).

## 1. Replication (task step 1)

- **Clean run of the builder's code.** All module outputs were deleted: `output/tables/m2_*`, `technology_registry_m2.csv`, `output/figures/m2_*`, and `data/processed/m2_techpanel/`. Then `run.py` was re-run unchanged.
  - Exit 0, 23.3 min on 6 processes; the machine load average was 20–33 from other agents.
  - All 20 CSV/TeX tables, the registry and all 100 bootstrap `.npy` files were **byte-identical** to the builder's.
  - The stage caches (point estimates, bootstrap draws, q fits, CES fits, local σ, neutrality tests) were identical to machine precision.
  - Seeds are deterministic, as claimed.
- **Fixed code, run from scratch again.** Exit 0, 25.9 min; the run log is in `data/processed/m2_techpanel/run_log.txt`.
  - All quantities shared with the first run are unchanged: θ, bootstrap draws and q draws differ by 0.
  - The changes are only the additions and fixes listed in Section 2: new columns in `m2_spec_tests.csv` and `m2_neutrality_magnitudes.csv`; 2 new registry/robustness rows; the Farseer out-of-sample RMSE (+0.00012); corrected table notes; 3 new CSVs; 2 new bootstrap-draw files (102 in total, indexed in `boot/index.csv`).
  - All 4 figures and Table 3 are byte-identical.

## 2. Issues found and fixes

| # | Severity | Issue | Fix | Status |
|---|---|---|---|---|
| 1 | **Major** | **Rank-one test confounded with q = 1.** τ was computed from a translog of ln(L − Ê) with Ê from the Chinchilla fit. The q family also implies a rank-one Hessian, but with its own E. Once q ≠ 1 (rejected everywhere), the Chinchilla Ê is off: 0.369 vs E_q = 0.265 on Farseer. ln(R + c) is not rank one, so the test re-detects q ≠ 1. Demonstration: a noise-free q-family surface on the Farseer design, analysed with the Chinchilla Ê, gives τ = −0.273 (observed: −0.258). The memo's "saddle-shaped, i.e. non-separable, as in Farseer's N-dependent data exponent" and claim 5 ("rank-one (separable) curvature is not [acceptable]") were therefore unsupported. The Chinchilla rejection (p = 0.007) also conflicted with module m1, which does not reject. | Added `w_translog_null_q`: τ with Ê_q, and a residual-bootstrap null under the fitted q model with E re-estimated in each of 300 draws. `w_q` now also stores the pairs-bootstrap τ_q. New columns `E_q, tau_q, tau_q_lo/hi, tau_q_null_*, p_rank_one_q` in `m2_spec_tests.csv`, plus a new column pair and rewritten notes in `m2_spec_tests.tex`. Results: Chinchilla τ_q = −0.009 (p = 0.69); Farseer 0.014 (null [−0.015, 0.002], p ≤ 0.003, but pairs CI [−0.001, 0.026] includes 0); OLMo 0.005 (p = 0.053); others p = 0.21–0.82. Memo H1d, claim 5 and Section 5 rewritten. | Fixed |
| 2 | **Major (overclaim)** | "Far less complementary than capital and labor" / "far above the capital–labor range 0.4–0.7" (claim 1, H1a) and "at or above the top of the capital–labor range" (open issue 1). The module's own q-free estimates (0.51–0.71) lie *inside* the range, and the lowest Chinchilla-form estimates (0.735–0.737) barely clear it. | Memo reworded: gross complements under every form; above the range under q = 1, inside it once curvature is freed. | Fixed |
| 3 | **Major (interpretation)** | DataDecide: "aggressively filtered DCLM variants tilt in the data-augmenting direction (lower M*), matching Lemma 1 and DeepSeek." Under common exponents, ln M*_r = const − 2/(α+β)·tilt_r, so "data-augmenting tilt ⇒ lower M*" is an identity. Quality is not measured independently of the tilt. On the common C4 metric the tilted recipes are the *worst* (highest E_r), and recipes with lower E_r have higher M*. DeepSeek's result concerns the exponent a, and separate-fit a_r shows no systematic relation to E_r: Dolma1.6++, which has the lowest E_r, has a = 0.374, the same as the DCLM QC variants. | Memo H2 and claim 9 rewritten. The pattern itself is supported: Spearman(E_r, ln M*_r) = −0.85, cluster-bootstrap CI [−0.92, −0.55]. It is not an artifact of the E–B estimation trade-off: the within-recipe sampling correlation of Ê_r and the tilt has the *opposite* sign (median −0.43 NLS, −0.79 Huber). These statistics are now written by the code to `m2_neutrality_magnitudes.csv` (they were quoted without a source file). A recipe × schedule-interaction caveat was also added. | Fixed |
| 4 | Moderate | **Pointwise-band claim.** "The Chinchilla-implied σ lies inside the nonparametric pointwise band at only 10% of grid points." The bands are a pairs bootstrap at a fixed, lower-edge CV bandwidth and ignore smoothing bias. The local estimates jump across neighbouring sizes by more than the bands allow: at D/N = 10, 0.72 / 0.78 / 0.69 at N = 0.8B / 1.6B / 3.2B, with bands of ±0.02 to ±0.05. | Memo and table note reworded. The robust comparison is the median: its CI [0.679, 0.699] excludes 0.77, as does the q-family CI [0.705, 0.716]. | Fixed |
| 5 | Moderate | **Local-σ slope.** "Local σ falls by −0.031 per unit of ln M" was the partial slope from a joint OLS on ln M and ln N. The regression was not in the code and no SE was given. The simple slope is −0.016, and the profile is non-monotone at N = 0.8B and 3.2B. | New `m2_farseer_local_sigma_summary.csv` holds every summary quoted, with bootstrap CIs (partial −0.031 [−0.040, −0.024]; simple −0.016 [−0.020, −0.012]). Claim 4 is downgraded to "suggestive". | Fixed |
| 6 | Moderate | **Wild-bootstrap inference.** The table note and memo said capped solver iterations "make the p-values conservative". That is not justified: an unconverged unrestricted refit deflates LR*, which is anti-conservative. | Checked directly. Uncapped refits of the 3 largest-LR* draws for DataDecide data-augmenting + E_r and Hicks + E_r reproduce the capped LR* to 3 decimals. Tight refits of DataDecide CE pairs-bootstrap draws reproduce the capped ones exactly. Notes corrected. | Fixed |
| 7 | Moderate | **Robustness of the neutrality tests** to the bootstrap design was unexamined. The sign-flipping wild bootstrap breaks the smoothness of the common misspecification. Example: Gadre E-shift-only rejected (p = 0.004) with LR = 2.6 on 4 df and a max LR* of 2.59. | New `neutrality_wald` → `m2_neutrality_wald_check.csv`: Wald tests with the cluster-pairs bootstrap covariance of θ_CE.<br>**Confirms:**<br>• Gadre E-shift rejected (NLS W = 42.6, bootstrap-t p = 0.005; Huber χ² p = 0.003, bootstrap-t 0.055);<br>• Gadre Hicks + E_r not rejected (p = 0.85 NLS, 0.74 Huber);<br>• all DataDecide rejections (p ≤ 0.03).<br>**New finding:** Gadre data-augmenting + E_r *is* rejected by NLS Wald (p = 0.018) but not by Huber (0.77). Gadre's ranking of factor-augmenting alternatives is estimator-dependent; the memo says so. | Fixed / reported |
| 8 | Moderate | **Muennighoff sample.** The single-epoch sample includes 1.1B- and 2.8B-parameter models trained on 100M tokens (D/N = 0.04–0.09). Dropping them, as Besiroglu does for D/N < 0.4 on Chinchilla, moves a from 0.535 to 0.360 and M*(10²¹) from 60 to 139, while σ* only moves from 0.735 to 0.726. | New robustness variant `datablations|single_epoch_M04` (registry, `m2_robustness.csv/.tex`). Memo H1a/H1e updated. This strengthens the "a and M* fragile, σ stable" message. | Fixed |
| 9 | Minor | **Out-of-sample leakage.** Farseer Eq. 3 and the E-free translog were started from the full-sample estimates (which include the test observations). | Starting values are now training-sample only. The Farseer top-decile RMSE moves from 0.00246 to 0.00258; nothing else changes by more than 1e-4. | Fixed |
| 10 | Minor | **Claim 10.** "Sweeps with the most off-path variation deliver the tightest σ." Muennighoff has Farseer's spread (sd 1.97) but the widest σ*_q CI; OLMo has little spread (0.94) but a fairly tight one. | Reworded: precision also depends on n. | Fixed |
| 11 | Minor | **Numbers inconsistent across files or arithmetic.**<br>• DataDecide sd(ln M \| ln C): 0.71 in the memo (unique design points) vs 0.73 in Table 3 (all checkpoints).<br>• The OLMo parameter check 3,169,537,280 = 16·16·3328² + 100,352·3328 is off by 216,320 = 65·3328 (layer-norm weights). | Memo clarified and corrected. The constant itself is right: it matches the OLMo-ladder `MODEL_PARAMS` on GitHub. | Fixed |
| 12 | Minor | "Farseer's own form is non-homothetic (M*(C) rising in C)" was not computed anywhere. | New `m2_farseer_eq3_Mstar.csv`: M* = 32, 25, 28, 43, 82 at C = 10¹⁹ to 10²³. It is non-monotone below 10²⁰, and the last two budgets are extrapolations. Memo updated. | Fixed |
| 13 | Minor | **DataDecide identification assumption not stated.** The design of the schedule artifact is common to all recipes, but its *effect* could interact with the recipe. | Caveat added to memo H2 and claim 9. | Fixed (caveat) |

`sl.py`: no bug found; the builder's note on ln E = −inf at E = 0 is accurate and handled locally. No edits to `sl.py`.

## 3. Audit checks that passed

- **Units and data construction** (verified against raw files and upstream code):
  - *DataDecide.* Batch sizes, full-schedule last steps and non-embedding parameter counts match `allenai/DataDecide` `utils/constants.py` exactly (fetched with `gh api`). D = step × batch × 2048 matches the repo's own `tokens_trained`. The 1B count includes one 50,304 × 2,048 vocabulary matrix plus 33 × 2,048 norm weights, i.e. it excludes only the input embedding, as documented.
  - *DataDecide truncated seeds.* The "small aux" seeds of the 530M/750M/1B models follow the same LR schedule as the default seed (losses coincide at equal steps to about 0.003), so treating them as truncated runs is correct.
  - *OLMo ladder.* `MODEL_PARAMS` matches `allenai/OLMo-ladder` `src/scaling/utils.py`. The last CSV row is the final step, with a non-missing C4 loss and the maximum `total_tokens`; D/N = 20 × multiplier.
  - *Farseer.*
    - N_add_emb − N = 131,072·h for every run (two untied 65,536-row embeddings), so N is non-embedding.
    - The `D/N` column takes 5 unrelated values and is correctly recomputed.
    - The output `L` is `IntelliValSet_Raw|en` from the authors' `*_val_bpc` files (verified in `Farseer/scalinglaw_utils/.../read_data_big_exp.py`), i.e. bits per character.
    - C6 spans 1.19e18–3.49e21; D/N spans 0.31–2,570; corr(ln N, ln D) = −0.04; 4 duplicate (N, D) pairs.
  - *Gadre.* D = 20 × multiplier × params exactly; the C4-val loss and CI parsing is correct; N/N_ne = 1.03–1.85.
  - *Muennighoff.* The notebook's own parsing yields 229 runs, 33 with D = U.
- **Formulas** (numerical verification):
  - The E-free local-σ formula `sigma_from_derivs` reproduces the closed-form Chinchilla σ to 5 decimals when applied to ln L, ln(L − E) or −L. It is invariant to monotone transforms, as claimed.
  - 1/σ = 1 + sβ + (1 − s)α (model_spec) holds, and σ at the compute optimum equals 2/(2 + α + β).
  - The analytic gradient of the q-family objective matches finite differences (relative error 1e-6), and q = 1 reproduces sl's objective exactly.
  - The sparse Jacobians and the Huber gradient of all 10 panel models match finite differences.
  - scipy `least_squares(loss='huber', f_scale=δ)` cost equals Σ huber_δ(r), so the objective scale is consistent with `sl.fit_chinchilla`.
  - The panel parameterizations implement the nested models as stated (checked mapping by mapping).
- **Estimation and inference.**
  - No failed bootstrap draws in any variant; few E → 0 draws (≤ 9%, OLMo Huber).
  - CES and q LR statistics recomputed from the SSRs match the memo (Chinchilla CES LR = 10.4, p = 0.001; q LR 46.2 / 999 / 77.0).
  - The q-family bootstrap (single warm start) was re-run with 6 extra starts including q = 1 on 4 small sweeps (100 draws): s.e.(q̂) changed by ≤ 0.005, the σ*_q intervals were identical, and no draw reached q ≥ 1.
  - DataDecide LRs exceed every bootstrap LR*.
- **Memo numbers.** Every number in memo sections 1–6 was checked against the regenerated CSVs, and all match. The exceptions were items 5, 11 and 12 above, and the Spearman(E_r, ln M*) value, which was correct but had no source file (now added).
- **Literature and cross-module checks.**
  - The Chinchilla row (σ* 0.737, a 0.514, M*(10²¹) 21.4) matches the SYNTHESIS ledger's Besiroglu refit (0.737, 0.513, 21.6) and module m1 (κ̂ = 0.774, σ*_κ = 0.700).
  - Muennighoff's own α = β = 0.353 implies σ = 0.739, consistent with 0.735.
  - The Kaplan range 0.50–0.575 in the ledger is consistent with the q-free range.
- **Citations.** All 16 keys cited in the memo exist in `lit/references.bib`, or in `lit/bib/extra_m2_techpanel.bib` (`fan1996local`, with a DOI).
- **Registry.** Schema as specified; σ* = 2/(2 + α + β) and a = β/(α + β) are exact in every row. The downstream rows `farseer/all` and `gadre/RefinedWeb` exist for huber, nls and huber_q, with N convention and loss units.
- **Figures.** They respect aer_style (at most 3 colors + gray, no twin axes), and no label collisions were found in the PNGs.

## 4. Remaining concerns (not fixable within this module)

1. **The level of σ is form-dependent.** Chinchilla form: 0.72–0.83. q family: 0.51–0.71. Nonparametric (Farseer only): 0.69. The q family is one alternative among many; in the 30–35-run sweeps its inner parameters are weakly identified (A′, B′ of order 10⁶–10⁹, q strongly correlated with the inner scale).
2. **The rank-one conclusion is conditional on the E estimator.** An E-free 7-parameter translog puts E at 0.13 and τ at +0.17 on Farseer. τ is too sensitive to Ê to support a strong statement in either direction.
3. **Local nonparametric σ.** The CV bandwidth sits at the lower edge of the grid and the bands exclude smoothing bias. Only the median (0.690–0.707 across bandwidths) is robust.
4. **DataDecide.** Identification rests on unfinished-schedule checkpoints. Across-recipe comparisons assume no recipe × schedule interaction, which is untestable here. Levels (a ≈ 0.32–0.36, M* in the hundreds) are not technology numbers.
5. **Gadre neutrality.** Three corpora, low power; the ranking of the factor-augmenting alternatives depends on NLS vs Huber.
6. **Wild cluster bootstrap under misspecification.** The Wald cross-check agrees on every headline test, but with 37–44 clusters and misspecification common across groups, neither bootstrap is textbook-valid. Report both.
7. **Unverified conventions.** Muennighoff `PARAMS_MAP` (total vs non-embedding) and the DataDecide tokenizer version remain unverified.
8. **Shared scratchpad.** Before switching to a private subfolder (`scratchpad/m2_review/`), the reviewer copied the builder's outputs into the shared `scratchpad/orig/{tables,figures,proc}`. Files there named `panel_*.csv`, `run_log.txt`, `boot/`, `cache/` may have overwritten another agent's scratch backups with the same names (`/private/tmp/claude-501/-Users-yigitokar-scaling-laws-pf/0d2513c5-45da-44dc-af1d-80b8664ef0bc/scratchpad/orig/`). No project files were affected.

## 5. Confidence in each headline claim (after fixes)

| Claim | Confidence | Note |
|---|---|---|
| H1a: Chinchilla-form σ* = 0.735–0.828 in every sweep, with the reported SEs | **High** | Reproduced exactly; matches Besiroglu and m1; SEs robust to clustering (Farseer by size: 0.009 vs 0.010) |
| H1a: σ < 1 (gross complements) | **High** | Holds under every form and in every sweep |
| "Above the capital–labor range" | **Low–medium** | True only under q = 1; q-free estimates lie inside 0.4–0.7 |
| H1b: q = 1 rejected in all 7 sweeps | **High** | LR and bootstrap Wald; robust to multi-start bootstrap |
| H1b: σ*_q = 0.51–0.71 | **Medium** | Tight on Farseer (0.710 [0.705, 0.716]); weakly identified in small sweeps |
| H1c: Farseer σ ≈ 0.69–0.71 by three q-free methods | **Medium–high** | Median/q-family agree; pointwise details fragile |
| H1c: σ falls with D/N | **Low** | Partial slope significant only under a bootstrap that ignores smoothing bias; non-monotone |
| H1d: CES not rejected (Wald) | **Medium–high** | LR rejects on Chinchilla (p = 0.001) |
| H1d (original): rank one rejected / saddle surface | **Retracted** | Artifact of the Chinchilla Ê |
| H1d (revised): close to rank one with q-family E | **Medium** | Conditional on Ê; Farseer shows a small detectable departure |
| Chinchilla extrapolates poorly on Farseer (out-of-sample RMSE 0.022–0.025 vs 0.0026) | **High** | Leakage-free re-run confirms |
| H1e: a and M* fragile, σ comparatively stable | **High** | Strengthened by the Muennighoff D/N ≥ 0.4 variant |
| H2 Gadre: Hicks + E_r not rejected; E-shift-only rejected | **Medium** | Confirmed by the Wald cross-check; low power |
| H2 DataDecide: all restrictions rejected; E + Hicks explain 99.7% | **High** (statistical) / **medium** (economic) | Schedule caveat |
| H2 DataDecide: tilt range 0.22–0.26 ⇒ M* 2.9–3.4× and wedge 1.25–1.29× | **Medium** | Bootstrap CI of the tilt range [0.13, 0.32] (NLS); schedule caveat |
| H2: filtered data tilt ⇒ "matches Lemma 1" | **Low** | Tautological under common exponents; quality not independently measured |

## 6. Files changed by the reviewer

- **`code/analysis/m2_techpanel/m2_est.py`**
  - `w_q` also returns τ_q.
  - New `w_translog_null_q`.
- **`code/analysis/m2_techpanel/run.py`**
  - New variant `datablations|single_epoch_M04`.
  - q-family rank-one null jobs.
  - Leakage-free out-of-sample starting values.
- **`code/analysis/m2_techpanel/m2_out.py`**
  - Spec-tests table: τ_q columns and new notes.
  - New local-σ summary file.
  - New Farseer Eq. 3 M* file.
  - Magnitudes: Spearman(E_r, ln M*) and tilt-range CIs, plus the wedge ratio.
  - New `neutrality_wald`.
  - Corrected table notes (neutrality, Farseer).
  - Robustness table: the new variant.
- **Regenerated outputs.** All of `output/tables/m2_*`, `technology_registry_m2.csv` (now 102 rows), `output/figures/m2_*` and `data/processed/m2_techpanel/*`.
  - New tables: `m2_farseer_local_sigma_summary.csv`, `m2_farseer_eq3_Mstar.csv`, `m2_neutrality_wald_check.csv`.
- **`output/memos/m2_techpanel.md`**: revised as described above; a review banner lists the changes.
- **Reviewer scratch checks** (not deliverables) are in `/private/tmp/claude-501/-Users-yigitokar-scaling-laws-pf/0d2513c5-45da-44dc-af1d-80b8664ef0bc/scratchpad/m2_review/`: `check_formulas.py`, `check_qboot.py`, `check_oos.py`, `check_wald.py`, `check_rank1_q.py`, `check_ce_conv.py`, `check_wild_conv.py`.
