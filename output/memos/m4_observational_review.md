# Independent review: module m4_observational

**Scope.** Replicator and skeptical referee review of the observational vs experimental ("LaLonde") module. Date: 2026-09-23.
- **Code:** `code/analysis/m4_observational/{run,panel,experiments,observational,lalonde,semisynth,tfp,figs,common}.py`.
- **Memo:** `output/memos/m4_observational.md`, rewritten with post-fix numbers and a revision banner.
- **Backups used for comparison:** original outputs in the session scratchpad; not part of the deliverables.

## 1. Replication

1. **Run 1: the original code, from scratch.**
   - All module outputs were deleted first, including the DataDecide parquet cache and `hf_ids.txt`.
   - `run.py` completed (exit 0): 30.4 min wall at load average ~45, 869 s CPU, 1.5 GB RSS.
   - **Every table, the TFP/ECI outputs and `headline.json` (excluding runtime) were byte-identical to the builder's.** The original pipeline is deterministic and regenerates from raw data.
2. **Run 2: the corrected code, from scratch.** Exit 0, 32.1 min wall. All numbers in the revised memo come from this run.
3. **Run 3: final verification.**
   - Same code as run 2 plus note-text and `headline.json` additions only; from scratch, including rebuilding the DataDecide cache from the raw parquet shards.
   - The numeric tables must match run 2; see §6 for the check.

## 2. Issues found and fixes

Severity: **major** = changes a number or claim the writers would use; **minor** = presentation, reproducibility or small numeric effects.

| # | Severity | Issue | Fix |
|---|---|---|---|
| 1 | **Major** | **Duplicated observations.** Sloth `data_v2` rows for pythia-70m/1b/1.4b/2.8b/6.9b (standard) have scores identical to 6 decimals on all six benchmarks to ObsScaling's `*-deduped` rows: they are copies. Effects: 5 duplicates in the main sample (133 rows = 128 distinct models) and 4 in the hull (63 → 59); pseudo-replication; the "same 16 Pythia models" output check used 11 distinct models on the leaderboard side. | `panel.build_panel` drops Sloth additions whose score vector duplicates an ObsScaling row, and prints which. |
| 2 | **Major** | **Parameter counts inflated by non-parameter buffers.** HF `safetensors.total` counts the U8 causal-mask buffers (`attention.bias`, 2048×2048 per layer) in GPT-NeoX/GPT-Neo checkpoints: Pythia-70m +36%, 160m +33%, 410m +23%, 1b +7%, GPT-Neo-125M +20%, GPT-NeoX-20B +0.9%; 17 models. Effects: (a) standard vs deduped Pythia had different N with identical outputs; (b) the same bug hit the Pythia *experimental* θ_N (≈6% too high); (c) which of two counts was used for Pythia-1.4b depended on filesystem glob order. | `panel.hf_param_count` counts floating-point tensors only (F16/BF16/F32/F8). Used in both panel and experiments, with deterministic ordering. After the fix, exact vs reported N differ by +2.6% (SD 6.3%, 86 models), vs the memo's +4.6% (SD 8.4%). |
| 3 | **Major** | **Overclaim: "θ_N overstated and θ_D understated in every specification; family FE make it worse."** Counter-evidence: with the ladder-only surface, developer-FE θ_N bias is −0.11 and θ_D +0.10, and OP-style θ_N is −0.09; for ARC-C and Winogrande, OLS and year-FE θ_N biases are negative; pooled-OLS N/D biases are insignificant. CR1 inference with 19 clusters overstated precision. | Added a restricted wild-cluster bootstrap-t (Webb weights, 999 draws; `lalonde.wild_bias_table` → `table6_lalonde_wildboot.csv`, `table6_lalonde_clean_wildboot.csv`). Only the family-FE biases are near significance: θ_N +0.203, wild p = 0.052; θ_D −0.262, wild p = 0.023. Developer/lab×period θ_N biases have wild p = 0.12 (CR1-normal 0.06–0.08). Claim 3 was rewritten and marked "weakened". |
| 4 | **Major** | **Unreported result.** On the overlap sample the China × post-export-control IV (developer + year FE) has F = 10.8 (now 10.6), θ_C = 0.90 (0.25) and AR [0.49, 2.23]: a "strong" instrument giving an estimate 1.7–2.3× the OLS/FE estimates on that sample. The memo said "F = 2.4" (full sample only) and "frontier IV F < 3 everywhere", but the overlap developer FE + trend specification has F = 3.8. | Both are now reported, with an explicit judgement that the exclusion restriction is not credible. |
| 5 | **Major** | **Asymmetric regime argument.** The memo scaled the semi-synthetic *target* bias up by the 5× larger cross-family TFP dispersion, but compared the observed +0.049 to the *unscaled* funding bias (+0.044), concluding "budget or mild funding". The simulated funding rule is rank-based, so its bias scales ~linearly with SD(ω): ≈ +0.2 at the observed dispersion, which the CIs also reject. The clean-sample bias (now +0.045 (0.023)) has wild p = 0.10. | Claim 5 was restated: both simulated rules are rejected at the observed dispersion; budget (compute ≈ orthogonal to TFP) or offsetting mechanisms fit best; stylization caveats were added. |
| 6 | **Major** | **Output-comparability claim not supported after fixes 1–2.** On the same 11 models with correct N: ARC-C leaderboard θ_N 0.785 vs 0.692 / 0.754 (+4–13%); Winogrande 0.677 vs 0.682 / 0.734 (−1 to −8%). The memo claimed 10–15% (Winogrande 0.77 vs 0.63/0.71). | `run.pythia_output_check` restricts both sides to common models and records the model list. Claim 9 was revised to "0–13%, sign varies". |
| 7 | Minor | "Essentially unbiased" overstates a null. The OLS θ_C bias CI is ±0.075 (±23%). ARC-C (−0.098) and Winogrande (−0.160) point estimates imply 18–31% attenuation. | Rephrased as "no detectable bias for HellaSwag". The CI and the ARC-C/Winogrande estimates are now in the headline and in Claim 1. |
| 8 | Minor | **TFP: inconsistent netting and conversion.** The "experimental θ" family 90/10 nets inputs with the within-family (θ_N, θ_D), which Table 6 finds biased, but converts with the experimental θ_C. | Added a consistent variant: ω_f = family mean of y − θ_C ln C. HellaSwag: 19.9× [7.6, 161] vs 23.7×; after exclusions 9.1× vs 7.0×. New table column `exp. θ, ln C`; `famc_*` columns in the csv. |
| 9 | Minor | Within-developer residual dispersion included single-model developers, whose residuals are identically 0 under developer FE. The ECI version already excluded them. | Excluded (125 of 128 residuals remain). Own-θ ratio 11.7× → 11.1× (the duplicate fix also contributes). |
| 10 | Minor | "Syverson-sized in loss units" framing. R^γ is an output-cardinalization choice. With returns to scale near one, Syverson's 1.92 is also an input-equivalent ratio, so the like-for-like comparison is 7–24× (LLMs) vs 1.92× (manufacturing): roughly 3–5× larger in logs. | Headline 8, Claim 7 and the TFP table note were reframed. The loss-unit numbers are kept as the ledger's convention. |
| 11 | Minor | TFP table note said "400 bootstrap draws"; the run uses 300. The Table 6 note hard-coded "300". The exp_benchmarks table hard-coded "16 / 210" Pythia runs; there are 15 final models. | Counts are now passed from the code (`B`, `B_ok`, `n_obs`). |
| 12 | Minor | The TFP "exp. θ" bootstrap intervals hold θ_C fixed; this was undisclosed. | Disclosed in the table note and the memo. |
| 13 | Minor | The "reported vs exact N" statistic in the memo was not produced by `run.py`. | Added to `design_audit.csv` (`n_exact_N`, `exact_over_reported_*`). |
| 14 | Minor | The memo said ladder N = FLOPs/(6D) "equals total parameters incl. untied embeddings". OLMo's counter (`olmo/model.py`: `num_fwd_flops` + `num_bck_flops`) charges 2·N(excl. wte) + 4·N_total + 12·L·d·seq per token, so this N is within a few percent of the total, not equal to it. | Docstring and memo corrected; residual concern listed. Not re-engineered, because the configs of the ladder paper were not verified. |
| 15 | Minor | The memo said "above 4B the surface rests on only two experimental points". There are 5 (three Gadre 6.9B runs at 20 tokens/parameter, plus OLMo-2 7B and 13B). | Corrected. |
| 16 | Minor | Own-hardware IV: 3 distinct values; identifying variation from 6 models (OLMo-1B/7B on MI250X; Llama-3 8B/70B, StarCoder2-7B, MPT-30B on H100). Frontier IV: 4 values. Neither fact was disclosed. | Disclosed ("effectively a vintage dummy"). After fix 1 the own-hardware IV has fewer than 12 hull models, so it is not in Table 6. |
| 17 | Minor | EIV σ_u on the overlap sample rests on 6 models (σ_u = 0.025, reliability 1.000). The memo's "0.19 from 14 models" is the full-sample figure. | Both are now stated. The main-sample truth dict is added to `headline.json`. |
| 18 | Minor | Transcription errors: composite family-FE θ_N bias SE 0.094 (actual 0.096); Winogrande θ_D bias SE 0.14 (actual 0.155); "attenuated by 50–70%" (actual 29–68% pre-fix, 27–63% post-fix); "0.27 (OLS) to 0.34 (family FE)" (the max is 0.36, lab×period). | Superseded by the rewritten memo. |
| 19 | Minor | Fig. 5: the θ_C axis clipped the EIV-corrected family-FE interval at 1.0. The regime label used ">=". | Axis widened to 1.25; label uses "≥". PNGs inspected: no collisions. |
| 20 | Minor | A citation was needed for the new wild-bootstrap method. | `cameron2008bootstrap` exists in `references.bib`. `webb2023reworking` added to `lit/bib/extra_m4_observational.bib` (verified via Crossref: CJE 56(3):839–858, doi 10.1111/caje.12661). All other memo citation keys were checked and exist. |

**Checked and found correct** (no change needed):
- chance-adjusted logit and composite;
- OLS/FE/Mundlak/CR1 implementations;
- the design-matched logic (same estimator, same sample, y* from the surface; the bootstrap resamples runs within design, and all 300 draws succeeded);
- surface local-elasticity numbers (1B: (0.39, 0.58)@20, (0.59, 0.23)@200, (0.67, 0.09)@500; 7B: θ_N 0.15@50, 0.35@500, θ_D −0.04@500);
- curvature F-tests (352, 70);
- DataDecide construction (525 finals ≥ 60M, D = 100N);
- semi-synthetic regimes (identical before and after fixes);
- Epoch join (no NaN-key matches; 0 duplicate names);
- D values and FIXES against Epoch;
- IV first-stage F = t² (cluster-robust) and AR grid inversion;
- ACF profile-GMM (just-identified);
- ECI join and developer mapping (Hugging Face → BigCode applies only to StarCoder 2);
- all literature numbers in the memo against the SYNTHESIS ledger: Mertens 0.789/log10 = 0.343/ln; 41^γ = 1.78–1.94; Syverson e^0.651 = 1.92; Epoch "Confident" ±3× ⇒ σ = ln3/1.645.

`sl.py` is not used by this module, so no bug is reported there.

## 3. Effect of the fixes on the headline numbers

| Quantity | Builder | After review |
|---|---|---|
| Main sample / hull | 133 / 63 | 128 / 57 |
| Table 6 OLS θ_C bias (HellaSwag) | +0.003 (0.037) | −0.001 (0.038), wild p 0.99 |
| Family FE θ_N / θ_D bias | +0.184 (0.065) / −0.245 (0.147) | +0.203 (0.086) / −0.262 (0.162); wild p 0.052 / 0.023 |
| Clean-sample OLS θ_C bias | +0.049 (0.023) | +0.045 (0.023), wild p 0.10 |
| Global OLS θ_N, θ_D (main) | 0.490, 0.251 | 0.470, 0.257 |
| Pythia experimental θ_N, ARC-C (0/5-shot) | 0.72 / 0.79 | 0.68 / 0.74 |
| FD θ_N; ACF θ_D, ρ | 0.16; 0.11, 0.75 | −0.00; 0.75, 1.17 (fragile) |
| Family 90/10, exp. θ (all; after exclusions) | 25.7× ; 7.5× | 23.7× ; 7.0× (θ_C-netted: 19.9× ; 9.1×) |
| Within-developer 90/10, own θ | 11.7× | 11.1× |
| Loss units, family (Besiroglu γ) | 1.78 → 1.43 | 1.76 → 1.41 |
| Pythia output check (ARC-C; WG) | 0.83 vs 0.72/0.79; 0.77 vs 0.63/0.71 | 0.785 vs 0.69/0.75; 0.68 vs 0.68/0.73 |

## 4. Confidence in each headline claim (post-review)

1. **Experimental global elasticities (θ_N 0.46–0.51, θ_D 0.35–0.51, ray 0.41–0.51): high.** They reproduce exactly and the fixes did not affect them. Pythia θ_N is ~6% lower after the N fix.
2. **Strong concavity; θ_D falls with D/N: high qualitatively** (F = 352/70, R² 0.994). **Medium for exact local values near 7B**, where 5 runs anchor the surface and it goes negative at the edge.
3. **Naive observational θ_D looks 27–63% attenuated: high** as a description; it is a design effect (claim 2).
4. **No detectable θ_C bias once the design is matched: medium.**
   - Holds for HellaSwag, the composite and the ladder-only surface, and is robust to the fixes.
   - The power is limited (±23%).
   - ARC-C and Winogrande point estimates show 18–31% attenuation.
   - The test is joint with common technology and output-format equivalence.
   - Coverage is ≤ 9B models only.
5. **N/D split biased, worst under family FE: low–medium.** The sign pattern is not robust to the surface or the output. The FE biases are borderline under few-cluster inference. This is a design-identification problem (little within-family D variation) rather than established transmission bias.
6. **Regime (Prop. 2): low–medium.** The restated version (both simulated rules rejected at the observed dispersion; budget or offsetting fits best) is defensible but stylized. The builder's "funding fits" reading is not supported.
7. **IO remedies fail on public data: high.**
   - The frontier IV is weak; own-hardware is a vintage dummy; the China IV fails exclusion; AB/BB is infeasible; FD/ACF are fragile; EIV is negligible.
   - The paper should present these as data limitations.
8. **TFP dispersion 7–24× compute-equivalent (Mertens-design residuals 11×): medium** for order of magnitude; the upper intervals are very wide. **Low** for the "Syverson-sized in loss units" framing, which depends on the output index.
9. **Output measurement moves slopes by 0–13%, sign varies: low–medium** (n = 11, two tasks, no HellaSwag).

## 5. Remaining concerns (not fixed)

- **Identification.** The design-matched benchmark assumes the AI2/OpenLM technology is every lab's technology up to a Hicks-neutral shift. Factor-biased data-quality differences are indistinguishable from transmission bias. The output formats differ (OLMES 5-shot cloze and LLM-foundry vs leaderboard 10/25-shot). The Pythia check suggests up to ~13% slope differences, which is the same order as the Table 6 bias SE.
- **Few clusters.** 16–22 developer clusters dominate inference everywhere. The wild bootstrap was added only for Table 6 (main and clean). Table 6b and the TFP bootstraps still rely on CR1 or the plain cluster bootstrap.
- **Surface uncertainty.** The benchmark SEs (≈0.01) reflect sampling only. Functional-form uncertainty (quadratic vs other) is gauged only by the ladder-only variant.
- **Unverified parameter counts:**
  - XGLM-4.5B's HF count (5.08B vs nominal 4.5B) may double-count the tied embedding.
  - OLMo ladder N from logged FLOPs includes attention FLOPs and drops a third of the input embedding (a few-% effect, largest for the smallest ladder models).
- **Data quality not independently verified:** HF upload dates stand in for release dates for 26 models; Qwen1.5 per-size D is imputed (ObsScaling); several ObsScaling token counts (e.g. TinyLlama v1.1 and H2O-Danube2, both listed as 3T) were not checked against the model reports.
- **ACF-style GMM.** ρ is profiled on [−0.5, 1.5] with no stationarity restriction. The corrected-data estimate ρ = 1.17 is inadmissible. The estimator is reported as uninformative, not refined.
- **ECI developer bootstrap.** About half of the draws have collinear developer/year dummies (statsmodels pseudo-inverse; θ_C and residuals are unaffected).
- **Loss-unit crosswalk.** It applies Chinchilla γ to benchmark log-odds. It is kept only as the ledger's convention.

## 6. Final verification (run 3)

Run 3 used the final code, from scratch: all module outputs were deleted, including the DataDecide parquet cache, which was rebuilt from the four raw shards.
- **Status:** exit 0 in 24.9 min wall (906 s CPU) at load average ~25–30.
- **Tables:** all 25 CSV tables are byte-identical to run 2, as are the processed panels (`obs_panel.csv`, `exp_*.csv`).
- **`headline.json`:** identical except for the two new keys (`global_truth_main`, `global_main`). The main-sample EIV σ_u = 0.195 from 14 independent compute estimates, and there are 6 three-generation cells.
- **LaTeX:** the 7 .tex files that differ (the Table 6 family and the TFP table) differ only in their `tablenotes` lines, as intended.
- **Figures:** 10 files (5 figures × pdf/png) were regenerated and inspected.

The deliverables in `output/` and `data/processed/m4_observational/` are from run 3.
