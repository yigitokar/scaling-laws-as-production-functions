# Memo — revision module ra4_obsfix: fixes to the observational and algorithmic-progress material (Online Appendix E)

Module owner: Claude (ra4_obsfix). Date: 2026-09-24.
- **Code:** `code/analysis/ra4_obsfix/run.py` is the single entry point. It runs five steps as separate processes (`step_matched.py`, `step_alloc.py`, `step_progress.py`, `step_tfp.py`, `tables_figs.py`); the helpers are in `ra4common.py` and `infer.py`. Runtime is 4–12 minutes, depending on machine load (11–12 minutes at load average ~40 during review), on at most 5 CPU processes, with no GPU. Two full runs from scratch gave byte-identical CSV and TeX outputs; the reviewer's re-run of the builder's code reproduced every CSV and TeX file byte for byte, and two runs of the final code match (see the review).
- **Reuse (read-only):**
  - m4_observational code, imported: panel, experimental runs, hull, surface data.
  - m5_progress code and cached outputs: the Ho et al. model and its bootstrap draws, the Epoch LM sample, the Farseer fit, the Besiroglu draws, and Table 7 panel A.
  - m8_measurement `results_summary.json`.
  - The m1/m2 technology registries. Their SHA-256 hashes are recorded in `data/processed/ra4_obsfix/allocative_headline.json`: m1 `5416fb06…` and m2 `a490883a…`, the same files m5 used.
- **No new data were downloaded**, so there is no `download_ra4_obsfix.sh`.
- **Inference conventions used throughout:**
  - wild and bootstrap p-values use B = 9,999 draws; the floor is 1/B = 0.0001;
  - allocative intervals (95%) and TFP intervals (90%) use B = 999;
  - the Tobit surfaces use a parametric bootstrap with B = 499.
  - Table E1 reports a conservative few-cluster p-value: the largest of three procedures that all include the benchmark's uncertainty (added at review). These are (i) CV3 with t(G−1); (ii) the restricted wild-cluster bootstrap-t studentized by CR1; (iii) the same bootstrap studentized by CV3. All three are in `matched_long.csv`.
- **Independent review (2026-09-24):** see `output/memos/ra4_obsfix_review.md`. The reviewer's changes are marked "[review]" below.

---

## 1. Headline findings

**H1. Reliability of ln C (R4 M7; R1 minor 44; R2 minor 25).** Two reliabilities disagreed: "≥ 0.988" in §VI.A, and 0.895 / 0.514 in the EIV rows of Table 8. The comparison behind the 0.988 was not independent, and the Table 8 rows used the wrong error variance and a wrong formula. Details:

- **Which comparisons are independent of reported N and D.**
  - Epoch's compute figure for our models is usually itself 6ND ("operation counting"; 51 models, MAD s.d. 0.015).
  - Some figures are developer-reported FLOP counts, usually also 6ND (5 models, s.d. 0.033).
  - Others are geometric means of 6ND and hardware time.
  - The §VI.A set ("Hardware*/Reported*", 14 models, s.d. 0.195) mixes all three.
  - Only hardware time is independent. We rebuilt it for the 27 main-sample models (12 families) that have Epoch chip-hours: chip-hours × peak dense 16-bit FLOP/s × a common 0.3 utilization.
  - We did not use Epoch's utilization field. It is back-calculated from Epoch's own compute estimate: we checked Llama-2-70B (0.419), StarCoderBase, MPT-7B, Falcon-40B and Falcon-180B.
- **The bound on the measurement error.**
  - s.d.(ln C_hw − ln 6ND) = **0.292** (family-cluster bootstrap 95% interval [0.197, 0.409]; MAD s.d. 0.296). The within-family s.d. is 0.208 (6 families).
  - This is an **upper bound** on the error s.d. of ln 6ND, because it also contains the dispersion of true utilization. For example, XGLM-7.5B and Llama-3-8B lie at +0.66.
  - [review] Pythia accounts for 11 of the 27 records, and the deduplicated and standard variants share chip-hour records. Keeping one record per (family, chip-hours) leaves 24 records with s.d. 0.306, so the bound hardly moves.
- **Justified reliabilities** (σ_u = 0.292; `ra4_obsfix_reliability_summary.csv`):

  | Sample | Pooled | Within family |
  |---|---|---|
  | 57-model support of Table 8 | 0.980 | 0.955 |
  | 20-model strict support | 0.973 | 0.969 |
  | all 128 models | 0.988 | 0.973 |

  At the upper 95% bound σ_u = 0.409, the 57-model reliabilities are 0.961 pooled and 0.912 within family.
- **Why Table 8's within-family EIV row (0.858, "bias" +0.448) is wrong.** Two errors combine:
  - It used Epoch's "Confident" class (σ_u = ln3/1.645 = 0.668). That class describes the uncertainty of Epoch's own estimate, not the error of 6ND.
  - It computed the within-family variance of ln C with an n − 1 denominator instead of n − F. The within transformation removes 30 family means, 15 of them for singleton families.
  - With the correct variance, σ_u = 0.668 gives reliability 0.766 (not 0.514), θ_C = 0.576 and bias +0.166 (0.060), p = 0.12.
  - With the justified σ_u = 0.292: θ_C = 0.461 (0.047) against a benchmark of 0.409 (0.013); bias +0.052 (0.049), p = 0.30.
  - Pooled OLS with EIV: 0.335 vs 0.329, bias +0.006 (0.039), p = 0.90.
  - (m4's own global EIV code, `observational.eiv_correction`, used the correct n − F formula; only the design-matched code path behind Table 8 did not.)

**H2. Why Table 8's wild-cluster p-values contradicted its standard errors (R1 9(e)).** Benchmark uncertainty is *not* the explanation. For θ_D under family FE, the benchmark bootstrap s.e. is 0.020 against an observational s.e. of 0.161. The explanation is the small-sample factor, as follows.

- **The CR1 factor.** Table 8's CR1 used G/(G−1)·(n−1)/(n−k) with k = 32, which counts the 29 family dummies (15 families are singletons). The factor (n−1)/(n−k) is then 2.24 instead of 1.04 under the nested convention. The variance is therefore 2.16 times too large, and the s.e. 1.47 times too large (0.161 against 0.110) [review: was "× 1.50"].
  - The dummies should not be counted, because families are nested in developer clusters (the fixest/reghdfe convention; `cameron2015practitioner`).
  - A bootstrap-t is invariant to this constant; a normal reference is not.
- **Corrected inference for the same biases** (`ra4_obsfix_inference_reconcile.csv`; Table E3). Every few-cluster-robust method now gives p = 0.01–0.03 for θ_D and 0.003–0.05 for θ_N. The unrestricted WCU, which over-rejects with few clusters, gives < 0.0001 for θ_N:

  | Method | θ_D bias −0.262 | θ_N bias +0.203 |
  |---|---|---|
  | s.e. (CR1, nested k) | 0.110 | 0.055 |
  | s.e. (CV3 jackknife) | 0.104 | 0.052 |
  | p, t(18), CR1 nested / CV3 | 0.031 / 0.024 | 0.005 / 0.003 |
  | p, restricted wild-cluster bootstrap (WCR), benchmark fixed | 0.024 | 0.046 |
  | p, WCR including benchmark draws | 0.028 | 0.048 |
  | p, WCR with family-level (subcluster) weights | 0.024 | 0.031 |
  | p, WCR studentized by CV3, with benchmark draws [review] | 0.011 | 0.048 |
  | conservative p (largest of CV3-t, WCR-CR1, WCR-CV3; Table E1) [review] | 0.027 | 0.048 |

- **Fragility.**
  - The effective number of clusters is G* = 4.1–4.3 of 19. Seven developers contribute no identifying variation (they have only singleton families), and two more contribute under 0.1%.
  - The θ_D identifying variation comes from Cerebras-GPT (37%; D = 20N) and Phi (28%; synthetic data).
  - Dropping any one developer keeps the sign: θ_D bias ∈ [−0.339, −0.193]; θ_N bias ∈ [0.157, 0.224].

**H3. Excluding OLMo-2 7B/13B from the experimental benchmark (R1 10(b)) collapses the common support from 57 to 20 released models.** The strict support has 9 families and 6 developers, all released 2021-03 to 2023-09, with N ≤ 6.7B and D ≤ 627B. [review: corrected] Without OLMo-2 the largest experimental D is 921B (one Gadre RedPajama run at 1.4B parameters, 640 tokens per parameter). At N ≥ 3.8B the largest D is 634B (OLMo ladder), and the 6.9B Gadre runs stop at 138B. The draft said OLMo-2 were the only points with D above 634B.

- **HellaSwag on the strict support** (`ra4_obsfix_tabE1_matched.tex`, Panel A):
  - OLS θ_C = 0.406 against 0.430 from the experimental technology on the same models: bias −0.024 (CV3-based s.e. 0.076), conservative p = 0.77.
  - [review] The three few-cluster p-values disagree widely here: CV3-t 0.77, WCR studentized by CR1 0.23, WCR studentized by CV3 0.39. Pairing the CV3 s.e. with the CR1-studentized WCR p (as the first version of Table E1 did) recreated the R1 9(e) mismatch, so Table E1 now reports the conservative p.
  - The other four estimators give biases +0.008 to +0.037, with every conservative p ≥ 0.77 (≥ 0.16 for the CR1-studentized WCR alone).
  - With 6 clusters (G* = 1.2–3.8), the test has almost no power. The 80%-power minimum detectable bias is 0.21 (50% of the benchmark) with normal critical values, and 0.27 (62%) with t(5).
- **The old design with the strict surface.** Evaluating the strict surface on the 57-model design (an extrapolation for 37 models) gives OLS bias +0.014 (0.045), p = 0.75 (conservative 0.76). The HellaSwag null does not depend on anchoring the surface with OLMo-2.
- **The old 57-model comparison with corrected inference:** OLS θ_C bias −0.001 (0.041), p = 0.98. The 95% interval is [−0.082, 0.080] with normal critical values, or [−0.088, 0.086] with t(18) [review: was "[−0.081, 0.079]"]. The minimum detectable bias is 0.116 (35% of 0.329) with normal critical values, or 0.123 (37%) with t(18).

**H4. ARC-Challenge and Winogrande: the apparent ARC-C attenuation does not survive modelling the censoring.**

- **The censoring.** 52 of 86 experimental runs are at the logit floor on ARC-C (16 of 86 on Winogrande). The m4 surface is OLS on these clipped values.
- **The fix.** We fitted a Tobit surface (left-censored), matched its level to the observed outputs, and clipped the counterfactual at the same floor.
- **ARC-C.** The OLS θ_C bias becomes +0.036 (0.069), conservative p = 0.61, on 57 models, and −0.007 (0.248) on 20. The m4 value was −0.098.
  - [review] With 60% of the runs censored, the Tobit benchmark rests on a Gaussian, homoskedastic latent surface below the floor. Its level is matched through one nuisance parameter. Its parametric bootstrap (B = 499) draws errors independently by run, not by (design, size) cluster. The benchmark is 0.413 (Tobit) against 0.547 (OLS on clipped logits). Say "not robust to modelling the censoring", not "a censoring artefact".
- **Winogrande** keeps a borderline attenuation: −0.150 (0.077) on 57 models, with p = 0.068 (CV3-t and WCR-CR1) and 0.084 (WCR-CV3; conservative). On 20 models it is −0.179 (0.263): the CR1-studentized WCR gives 0.076, but CV3-t gives 0.53 and the conservative p is 0.53 [review].

**H5. Allocative gain with the wedge truncated at one (R1 10(d)): the realized Kaplan→Chinchilla rebalancing gain is positive under every technology.**

- **Sample:** Epoch models with C ≥ 10^23 FLOP; 7 in 2020–21 and 92 in 2022–24 (`ra4_obsfix_allocative.csv`; Table E6).
- **Every era-1 model has w < 1 under every technology.**
- **Truncated gains (counting only the elimination of under-training):**

  | Technology | Truncated gain [95% interval] |
  |---|---|
  | Besiroglu | 1.68 [1.21, 2.44] |
  | Hoffmann (TeX) | 2.34 [1.87, 3.03] |
  | Hoffmann (rounded) | 2.77 [2.16, 3.69] |
  | κ-free Chinchilla | 1.97 [1.58, 2.53] |
  | Farseer (total N) | 2.06 [1.66, 2.60] |
  | Farseer (non-embedding N) | 1.54 [1.32, 1.84] |
  | Meta's Llama 3 IsoFLOPs | 1.30 [1.16, 1.50] |
  | Gadre C4 | 1.14 [1.06, 1.26] |
  | OLMo ladder | 2.63 [2.07, 3.33] |
  | Muennighoff et al. | 2.33 [1.87, 2.97] |
  | (Mis)Fitting | 1.64 [1.31, 2.12] |
  | Marin ×3 | 3.5–4.9 |

  - The range across the 14 technologies is **1.14–4.95**, and every 95% interval excludes 1 (smallest lower bound 1.06).
  - [review] These intervals resample models only. Only the Besiroglu row also draws the technology's parameters; the other 13 hold their technology fixed. With 7 era-1 models, percentile intervals are rough.
  - [review] The sign is close to mechanical. Every era-1 model has w < 1, and truncation can only raise era-2 efficiency. The finding with content is the *size* of the gain and the 80–94% of the era-1 gap that was closed.
  - [review] Under Proposition A-wedge(v), w < 1 reveals a belief error (the Kaplan law), a binding data constraint, or data-augmenting lab productivity. The truncated measure attributes all of it to allocative error. That is plausible for 2020–21 models of about 300B tokens, but it is an assumption.
  - The untruncated gains range from 0.55 to 4.31, so the untruncated sign was not robust.
- **Share of the era-1 under-training gap closed** by era 2 (in logs): 0.80–0.94.
- **Share of the gain implied by Ho et al.'s rate** (8.87× over 2.28 years): 6% (Gadre) to 73% (Marin Comma); 24% under Besiroglu.
- **Unchanged pieces.**
  - The Kaplan-rule counterfactual (Gundlach et al.'s CEG) is unaffected by truncation: Kaplan allocations have w < 1 under every technology. Under Besiroglu it is 11.7× at 5×10^26 FLOP.
  - Under Ho et al.'s own attenuated technology every model has w > 1, so the truncated gain is 1 by construction.

**H6. Productivity dispersion in both unit systems (R4 M11; R1 10(e)).** HellaSwag family 90/10 ratios; `ra4_obsfix_tfp_units.csv`, Table E7. The intervals now also draw θ_C. Brackets are **90%** developer-cluster bootstrap intervals [review: label added].

| Sample | Output units (odds) | Compute units | Reducible-loss units (γ = 0.178) |
|---|---|---|---|
| All 128 models, θ_C-netted (θ_C = 0.33) | 2.68 [1.94, 5.27] | 19.9 [7.5, 161] | 1.70 [1.43, 2.47] |
| Without code/distilled/synthetic (θ_C = 0.33) | 2.07 | 9.1 [4.9, 12.7] | 1.48 |
| Within-developer residuals (own θ_C = 0.43 [review: the row does not use 0.33]) | 2.78 | 11.1 [4.1, 18.1] | 1.54 |

- With the strict benchmark θ_C = 0.43, the compute units are 13.1 (all models) and 6.9 (restricted).
- Manufacturing's 1.92 lies inside the output-unit range, above the loss-unit ratios, and far below the compute-unit ratios.
- "Three to five times larger in logs" holds only in compute units: log ratio to 1.92 = 2.3–4.9.

**H7. Algorithmic progress (R4 M4(f); R1 minor 46).**

- **The φ interval.** We re-optimized the φ = g_N/g_C profile on a 0.025 grid near its boundaries. The 95% interval is **[−0.44, 3.37]**; m5's coarser grid with interpolation gave [−0.39, 3.37].
- **The T_C range.** Over that interval T_C ranges from **5.3 to 39.6 months**, with its minimum at φ ≈ −0.3. The draft's "5 to 38" was computed over the grid points [−0.25, 3.25].
- [review] **The profile is bimodal.** It has a second local minimum at φ ≈ −0.28 (LR 2.95) besides the global one near φ = 0.5. The fitted β_year jumps from 0.066 to 0.033 between φ = −0.100 and −0.075, so the optimizer is on different branches there (`ra4_obsfix_phi_profile.csv`).
  - The interval stays connected, because every grid value in [−0.425, 3.35] has LR < 3.84.
  - But a missed global minimum can only overstate LR. The lower endpoint is therefore conservative in the sense that the true interval can only be wider.
  - Profile LRs are iid Gaussian, not clustered by paper.
- **Replication and ridge.** These are condensed into Table E5, with no re-estimation: 8.44 [4.4, 14.1] under the authors' protocol; converged 6.08 [3.0, 22.7]; NLS 10.2 [4.1, 27.6]; profile interval [4.1, 40.5]; bootstrap correlation −0.73 (their code) and −0.82 (converged).

**H8. Bjorck exponent range (R4 M8(c)).**

- **Bjorck et al.'s exponents.** They report LR* ∝ D^−β at a fixed 0.5M-token batch (arXiv 2409.19913v3, Table 5 and Sec. 4, re-verified at review).
  - β ≈ 0.32 is a *joint* fit over 760M, 1.3B and 2.7B.
  - The per-size fits are 0.3155 (760M), 0.3171 (1.3B) and 0.4184 (2.7B) [review: the 2.7B value was omitted].
  - For smaller models β is larger: 0.3799 at 350M, 0.6531 at 125M, 0.7029 at 50M.
- **Correction to R4.** R4's "0.40–0.70" for small models should read 0.38–0.70.
- **What changes.** Step Law's cells span 215M–1.07B non-embedding parameters. The Bjorck exponents that bracket that range run from about 0.32 (760M–1.3B) to 0.65 (125M). The 50M value (0.70) lies below Step Law's range [review]. Conditioning on the batch closes this share of the published gap:
  - 26% → 17% (smoothed optima) as β goes from 0.32 to 0.65 (16% at 0.70);
  - 43% → 28% (grid argmins) over the same range (27% at 0.70).
  - The single "26–43 percent" becomes **17–43 percent** [review: was "16–43", which used the out-of-range 50M exponent] (`ra4_obsfix_bjorck_shares.csv`, column `brackets_steplaw_range`).

**H9. IV rows and Anderson–Rubin sets (R1 minor 45).** AR sets are by grid inversion over θ_C ∈ [−3, 4], with t(G−1) critical values; normal-critical-value sets are also given.

| Sample | Instrument | First-stage F | Estimate | AR set |
|---|---|---|---|---|
| 57 models | Frontier FLOP/$ | 0.90 | 0.724 | unbounded |
| 57 models | Frontier FLOP/$, developer FE | 0.01 | 8.54 | unbounded |
| 57 models | Own-hardware FLOP/$ (n = 10, 7 developers) | 14.3 | −0.090 (benchmark 0.195) | [−1.50, 0.34]; normal [−0.94, 0.27] |
| 20 models | Frontier FLOP/$ (6 clusters) | 5.15 | 0.008 | unbounded under t(5); normal [−1.85, 0.27] |

None of these instruments is informative.

---

## 2. Methods

**Samples and benchmark (`step_matched.py`).**
- **Data.** m4's panel (128 dense base models) and experimental runs are rebuilt by importing m4's `panel.build_panel`, `experiments.build_olmo_ladder` and `build_gadre`.
- **Two surfaces**, both a quadratic in (ln N, ln D) with dataset intercepts:
  - "strict": the OLMo ladder (30 runs) plus Gadre N ≥ 0.1B (56 runs) = 86 runs;
  - "legacy" (m4/Table 8): the strict runs plus OLMo-2 7B/13B = 88 runs.
- **Supports.** Each surface's support is the convex hull of its runs. The strict hull is a subset of the legacy hull.
- **Three comparisons:**
  - strict surface on the strict support (20 models);
  - legacy surface on the legacy support (57 models; reproduces Table 8's point estimates exactly);
  - strict surface on the legacy support (an extrapolation for 37 models).
- **Outputs.** HellaSwag, ARC-C, Winogrande and the composite.
- **ARC-C and Winogrande surfaces.** Beyond m4's OLS surface on clipped logits, a Tobit likelihood left-censored at logit(0.01).
  - The latent surface's level is matched to the observed outputs by minimizing Σ(y − max(floor, μ + a))².
  - Levels are a Hicks-neutral nuisance, and the harness intercept is unknown.
  - The counterfactual is clipped at the same floor, so both sides are censored alike.
  - The Tobit fits converged (max |gradient| < 1e-4; scipy's "precision loss" flag is benign; checked with Powell and L-BFGS-B polish).

**Estimators.**
- **List:** OLS, year FE, developer FE, family FE, developer×year FE, and the notability-propensity control (m4's "OP-style selection").
- **Two specifications:** (ln N, ln D) and ln C only.
- **Same estimator on both sides.** Every estimator is linear in the outcome, so the benchmark estimate is exactly the same linear map applied to y* = f̂(n, d). Its bootstrap draws are that map applied to the surface draws.

**Inference (`infer.py`).**
- **Benchmark uncertainty:** a design-conditional wild bootstrap of the surface. It uses HC2-rescaled residuals and Webb weights, clustered by (design, model size) (17 clusters strict, 19 legacy), with 9,999 draws. A run-level Rademacher version is in the CSV (`bench_se_runlevel`). For HellaSwag the two agree within 0.005 (run-level/clustered ratio 0.79–1.08). For the OLS surfaces of ARC-C, Winogrande and the composite, the run-level s.e. is up to 1.7× the clustered one in a few fixed-effects θ_D rows, where the observational s.e. is larger by an order of magnitude.
- **Observational uncertainty**, all by developer:
  - CR1 counting all dummies (as in Table 8);
  - CR1 not counting FE nested in developer clusters;
  - the CV3 jackknife (MacKinnon, Nielsen and Webb 2023).
- **Test of zero bias:** a restricted wild-cluster bootstrap-t (Webb weights; developer clusters) of H0: plim obs = plim bench.
  - The statistic is T = (b − θ̂_b)/√(V_CR1 + V_b).
  - In each draw an independent benchmark draw enters, so both sources of uncertainty are included (R1 9(e)).
  - Variants reported: benchmark fixed (Table 8 protocol), unrestricted (WCU), and subcluster weights by family or by model (MacKinnon and Webb 2018).
  - [review] **A CV3-studentized variant** (`infer.wild_test_cv3`) runs a restricted WCR on the bias outcome z = y − y*, which imposes a zero coefficient.
    - Its statistic is T = h'z/√(CV3² + V_b), with CV3 recomputed in every draw from vectorized leave-one-developer-out maps. The sample CV3 reproduces `jackknife` to 7×10⁻¹²; this is logged as `cv3_check`.
    - It uses its own seeded stream, so all earlier draws are unchanged.
  - [review] **Table E1 shows the conservative p**, the largest of CV3-t(G−1), WCR-CR1 and WCR-CV3. It rejects only if all three reject, which is valid if any one of them is.
  - [review] **Table E1's "Obs." s.e. is now the CV3 of the observational estimate itself** (`obs_se_cv3_y`). The bias s.e. is still the CV3 of Obs. − Exp., with the benchmark recomputed on each reduced design; the first version printed that bias CV3 under "Obs.".
- **Diagnostics:** the effective number of clusters G* (Carter, Schnepel and Steigerwald 2017, ρ = 0); partial-leverage shares by developer and family; leave-one-developer-out biases; minimum detectable bias at 80% power (2.8 × s.e.).
- **The vectorized bootstrap relies on two identities.** In the restricted case, b*_j − θ_0 = h'E*. Bootstrap residuals are M·E*, so no refit is needed. Both follow from h = X(X'X)⁻¹e_j.

**Compute-only rows.**
- **EIV.** Reliability λ = 1 − σ_u²/Var(ln C). The within-family variance is Σc̃²/(n − F). Table 8's n − 1 version is reproduced for comparison.
- **Reverse regression.** Reported with its Nerlove/Hall bracket: 1/b = θ̂/R² under no transmission bias.
- **IV** (2SLS). Nested-CR1 first-stage F. AR sets by grid inversion over [−3, 4] (step 0.005), with t(G−1) and normal critical values.

**Reliability (`step_matched.reliability`).**
- **The comparison.** Epoch structured fields (`Training chip-hours`, or `Hardware quantity` × `Training time (hours)`; `Training hardware`), and peak dense BF16/FP16 tensor FLOP/s from `ml_hardware.csv`.
- **Utilization.** A common 0.3; the level cancels in the s.d.
- **Classes.** Comparisons are classified by Epoch's `Training compute estimation method`.
- **Uncertainty.** Family-cluster bootstrap of the s.d. (9,999 draws); a MAD-based s.d. as a robust check.

**Allocative (`step_alloc.py`).**
- **Reused m5 code:** m5's `s4_allocative` sample (`load_epoch`, 247 models), cleaning mask and Kaplan counterfactual.
- **Truncation.** CE_tr = CE(min(w, 1)): CE if w < 1, and 1 otherwise. Within the Chinchilla family CE depends on w alone.
- **Technologies.**
  - Besiroglu, Hoffmann-TeX and Hoffmann-rounded.
  - The κ-free Chinchilla fit. Its inner exponents come from m2's `generalized_form` row; CE and w depend only on the inner aggregator.
  - m5's Farseer fit (total N).
  - m2 primary Huber rows: Farseer (non-embedding N), Gadre C4, OLMo ladder, datablations. DataDecide is excluded because its pooled fit has recipe-specific levels.
  - m1 rows: Meta's Llama 3 IsoFLOPs, Marin (Comma, DCLM, Nemotron-CC), and (Mis)Fitting FineWeb/C4.
  - Ho et al.'s model 7 at each release date.
- **Bootstrap.** Models are resampled within era (999 draws, one seeded stream per sample–technology cell). For Besiroglu the draws also cycle through m5's 200 parameter draws. The Ho-rate shares use m5's paper-cluster g_C draws.

**Progress (`step_progress.py`).**
- **φ profile.** Re-optimized at 31 new φ values (m5's `HoModel(HoSpec(delta=0, phi_fixed=v))`, `fit(n_random=6, seed=3)`), merged with m5's grid. LR = n·ln(MSE/MSE_min), MSE_min = 0.0453030. The crossings are found by linear interpolation.
- **Bjorck shares.** m8's bootstrap summary, rescaled: the share of the gap closed is linear in the gap 0.307 + β, so the point, s.e. and quantiles rescale exactly.

**TFP (`step_tfp.py`).**
- **Three dispersion measures:** family effects netted with the benchmark θ_C; family effects netted with the within-family (θ_N, θ_D); and within-developer residuals (Mertens design). Code families, distilled and synthetic-data families are dropped as in m4.
- **Three unit systems:**
  - output units, odds: e^Δ;
  - input units, compute-equivalents: e^(Δ/θ_C);
  - reducible-loss units: e^(γΔ/θ_C), with γ = 0.178 (Besiroglu) and 0.155 (Hoffmann).
- **Uncertainty.** Developer-cluster bootstrap (999 draws). θ_C is drawn from the design-conditional benchmark draws; m4 held θ fixed.

---

## 3. Inventory

**Tables (`output/tables/`)**

| File | Content |
|---|---|
| `ra4_obsfix_tabE1_matched.tex` | **Online Appendix E, design-matched comparison.** Panel A: strict benchmark, 20 models. Panel B: benchmark with OLMo-2, 57 models (old Table 8). Panel C: OLS for the strict surface on 57 models, ARC-C and Winogrande (censored benchmark) on 20 and 57. CV3 s.e. (Obs.: of the observational estimate; bias: of Obs. − Exp. plus the benchmark's s.e.); conservative p (largest of CV3-t, WCR-CR1, WCR-CV3; [review]). |
| `ra4_obsfix_tabE2_compute.tex` | Compute-only estimators: EIV with the justified reliability (and the Table 8 row reproduced), reverse regression with its bracket, IV with first-stage F and AR sets (t(G−1) and normal). |
| `ra4_obsfix_tabE3_inference.tex` | Reconciliation of Table 8's family-FE inference: CR1 all-k / singletons dropped / nested k, CV3, seven wild-bootstrap variants (incl. the CV3-studentized WCR added at review), G*, leave-one-developer-out range, leverage shares (in the notes). |
| `ra4_obsfix_tabE4_reliability.tex` | How well compute is measured: the four comparison classes (s.d., interval, MAD) and the implied reliabilities (pooled, within family, and Table 8's n−1 formula) on 57, 20 and 128 models. |
| `ra4_obsfix_tabE5_progress.tex` | Ho et al. replication and ridge summary (condensed; R1 minor 46). |
| `ra4_obsfix_tabE6_allocative.tex` | Allocative gains, untruncated and truncated at w = 1, for 14 technologies plus Ho's, with provenance groups; era-1 gap closed; Kaplan-rule CEG at 5×10^26; other samples under Besiroglu. |
| `ra4_obsfix_tabE7_tfp.tex` | TFP 90/10 in output (odds, reducible loss) and input (compute) units for two θ_C benchmarks; manufacturing reference. |

**Supporting CSV files (`output/tables/ra4_obsfix_*.csv`):**
- `matched_long` has every sample × surface × output × estimator × parameter row, with all s.e. and p variants, G*, leave-one-out results and minimum detectable biases.
  - [review] It adds `obs_se_cv3_y` (CV3 of Obs.), `p_wcr_cv3_combined` (the CV3-studentized WCR), `p_conservative` and `p_min_of_three`, `mde80_cv3_tG1` (MDE with t(G−1)), and `cv3_check`.
- `compute_only`, `inference_reconcile`, `leverage_familyFE`.
- `reliability_models`, `reliability_comparisons`, `reliability_summary`.
- `surfaces`, `matched_hull_members`.
- `allocative`, `kaplan_counterfactual`.
- `phi_profile`, `bjorck_shares`, `tfp_units`.

**Figures (`output/figures/ra4_obsfix_*.{pdf,png}`)**
- `fig_support`: (ln N, ln D) support, strict vs with-OLMo-2 hulls, designed runs and released models (20 inside the strict support, 37 more inside the OLMo-2 support only).
- `fig_matched`: bias by estimator for θ_C and θ_D (strict vs with OLMo-2), and the OLS θ_C bias by output.
- `fig_reliability`: (a) ln(C_alt/6ND) for the 27 hardware-time models, hardware time vs Epoch's reported compute; (b) EIV-corrected θ_C against σ_u, with the Table 8 point marked.
- `fig_allocative`: untruncated vs truncated gains by technology.
- `fig_tfp_units`: the 90/10 ratio in three unit systems vs manufacturing's 1.92.

**Processed data (`data/processed/ra4_obsfix/`)**
- `matched_headline.json`: support counts (models, families, developers, families with two or more models, singletons, maximum N and D, dates), surface run counts and reliability comparisons.
- `allocative_headline.json`: registry hashes, the Ho rate and the technology provenance.
- `progress_headline.json`: φ interval, bootstrap correlations, the replication summary and file hashes.
- `tfp_headline.json`, `run_log.json`.
- `bench_thetaC_draws.npz`, `allocative_per_model.csv` (w and CE per model and technology), and `besiroglu_boot_theta.npy` (a copy of m5's draws).

**Bibliography.** `lit/bib/extra_ra4_obsfix.bib` holds seven new entries, each verified on Crossref 2026-09-24 (DOIs in the file): todd2006assessing, hotz2005predicting, mackinnon2023cluster, cameron2015practitioner, mackinnon2018wild, carter2017asymptotic, basu1997returns.

**Compile check.** The seven .tex tables compile with `paper/AEA.cls` via `tools/tectonic`, with no errors, no overfull boxes and no floats too large; the page images were inspected. Every citation key resolves against `paper/references.bib` plus the new entries.

---

## 4. Claims for the paper (Online Appendix E), with corrected wording

**C1. Design-matched validation across populations.**
- **Wording.** Replace "LaLonde-style" everywhere with: *"a design-matched validation across populations (cf. Todd and Wolpin 2006; Hotz, Imbens and Mortimer 2005): the benchmark technology comes from other labs' recipes, not from the treated population, so the comparison is joint with a common technology up to Hicks-neutral shifts."*
- **Citations.** LaLonde (1986) and Dehejia and Wahba (1999) may be cited only for the idea of benchmarking and common-support trimming.
- **Claim.** *"Once the design is matched, cross-lab returns to compute on HellaSwag do not differ detectably from the experimental technology. This is a failure to reject in a low-power joint test, not evidence of unbiasedness."*
- **Evidence.**
  - Table 8 design (57 models): OLS bias −0.001 (0.041), p = 0.98, 95% interval [−0.081, 0.079], minimum detectable bias 0.116 (35%).
  - Strictly experimental benchmark (20 models): bias −0.024 (0.076), p = 0.23, minimum detectable bias 0.21.
  - The strict surface on the 57 designs gives +0.014 (0.045).
  - Six estimators (57): biases −0.029 to +0.037, all conservative p ≥ 0.36. Five estimators (20): biases −0.024 to +0.037, conservative p ≥ 0.77 (the CR1-studentized WCR alone: ≥ 0.16).
  - [review] 95% interval on 57 models: [−0.088, 0.086] with t(18), or [−0.082, 0.080] with normal critical values. Minimum detectable bias 0.12 (37%) on 57 models; 0.27 (62%) on 20 models with t(5).
- **Caveats.**
  - Removing OLMo-2 leaves 20 models from 6 developers (G* ≤ 3.8), so there is almost no power.
  - Winogrande shows borderline attenuation: −0.150 (0.077), p = 0.068–0.084 across the three procedures.
  - The ARC-C attenuation in Table 8 (−0.098) does not survive modelling the censoring. With a Gaussian Tobit benchmark it is +0.036 (0.069). The Tobit extrapolates a latent surface below the floor for 60% of the runs.
  - The test covers N ≤ 8.8B (57) or ≤ 6.7B (20) only.
- **Counts to state (R1 minor 42):**
  - 57 models: 19 developer clusters; 30 families, of which 15 have two or more models (42 models) and 15 are singletons.
  - 20 models: 6 developers; 9 families, of which 5 have two or more models (16 models).

**C2. The parameter/data split under family FE.**
- **Evidence.** On the Table 8 design the biases are significant under inference that accounts for few clusters and for benchmark uncertainty:
  - θ_N +0.203 (CV3 0.059), WCR p = 0.048 (conservative p 0.048);
  - θ_D −0.262 (CV3 0.106), WCR p = 0.028 (conservative p 0.027);
  - the sign survives dropping any developer.
- **Caveats.**
  - The Table 8 s.e. were inflated by a factor of 1.47 by counting nested FE dummies; that is why "t = 1.6, p = 0.02" appeared [review: was "1.50"].
  - With the strict benchmark the biases are similar in size (+0.185, −0.227) but not significant: conservative p = 0.14 and 0.15 with 6 clusters. [review] The individual procedures range from 0.010 to 0.14 for θ_N and from 0.047 to 0.15 for θ_D, so no procedure is reliable there.
  - θ_D is identified by two families: Cerebras-GPT (D = 20N) and Phi (synthetic data).
  - Transmission of Hicks-neutral productivity through compute (the budget rule of Proposition 6, which moves n and d together along the path) would bias θ_N and θ_D in the same direction. Opposite signs point to factor-biased recipe changes within families or to error in D. [review] With off-path variation, the omitted-variable bias of a two-regressor model can have opposite signs, so this is a model-based argument, not a theorem.
- **Recommendation.** Drop the introduction's "overstate by 0.20 / understate by 0.26" as a finding. In Appendix E, report it as a within-family disagreement driven by two families.

**C3. Measurement error in compute is small, but "≥ 0.988" came from a non-independent comparison. Replace it with:**
> *"Only hardware-time records measure compute independently of reported parameters and tokens; Epoch's figures for these models are mostly 6ND itself. Across 27 models with hardware-time records, ln(C_hw/6ND) has standard deviation 0.29 (95% interval 0.20–0.41), an upper bound on the error in ln 6ND because it includes utilization differences. The implied reliability of ln C is at least 0.98 pooled and 0.955 within families on the design-matched sample (0.96 and 0.91 at the upper bound); EIV correction raises the pooled compute elasticity by 0.007 and the within-family one by 0.021 (0.042 at the upper bound)."* [review: was "0.020 (0.043)"; the exact values are 0.0207 and 0.0425.]
- The within-family EIV row of Table 8 (0.858) must be replaced: it used Epoch's "Confident" class and a within-variance formula that ignores the family means.

**C4. Reverse regression.**
- **Wording.** *"The inverse slope of ln C on output, 0.560, equals the forward slope 0.328 divided by the forward R² of 0.585: the Nerlove/Hall bracket. Its distance from the benchmark (0.352) measures output dispersion not explained by compute, not a bias."*
- **Evidence.** 0.328/0.585 = 0.560 exactly (Table E2). With family FE: 0.441/0.760 = 0.580.

**C5. Compute-only regressions impose a false restriction (R1 minor 43, Hall-type bias).**
- **Wording.** *"Both worlds give θ_N > θ_D on this design (0.42 vs 0.27 under the experimental technology; θ_N − θ_D = 0.21 (0.08) on all 128 models), so a regression on ln C = ln 6ND imposes equal weights that the data reject. The residual of such an equal-weight index misattributes (θ_N − θ_D)(Δn − Δd)/2 to productivity: the Hall-type bias of Corollary A5 (cor:hall), which with over-training (w > 1) and shifts toward tokens makes measured productivity understate progress."*

**C6. Instruments.**
- **Claim.** Every IV estimate must carry its AR set (Table E2).
- **Wording.** *"Frontier FLOP/$: F = 0.90, AR set unbounded; with developer FE, F = 0.01, unbounded. Own-hardware FLOP/$ is strong only on 10 models from 7 developers (F = 14.3; AR [−1.50, 0.34]) and takes three values."*
- The China × export-control IV (m4, overlap sample) already reports AR [0.49, 2.23]; keep it with its failed exclusion restriction.

**C7. Renaming "OP-style selection" (R1 10(c)).** Call it a **"notability-propensity control (selection on observables)"**, and set out the timing:
- **State variable:** the lab's compute capacity, i.e. the provisioned cluster, fixed before the run.
- **Flexible input:** the run's split of its compute budget between N and D, chosen at the start of the run given capacity and the known recipe quality ω.
- **Timing and selection:** release follows the observed outcome.
- **Why no OP/LP proxy exists:** Hicks-neutral ω does not move the N/D mix at given compute (Proposition prop:proxy(ii)), so the flexible input cannot proxy ω. An OP-type proxy would need capacity investment, which public data do not show.
- [review] **Compute as an LP-type proxy.** Per-run compute is the other candidate, as R1 suggests. It fails for a different reason: under a strictly monotone budget rule c = h(ω, z), the inputs on the path are functions of c alone. A control function in c therefore absorbs the technology, which is ACF functional dependence (Proposition prop:proxy(iii)). Identification then needs timing, e.g. compute committed before the recipe innovation. State both reasons.

**C8. Allocative vs technical change (R1 10(d)).**
- **Evidence.** Counting only the elimination of under-training (w truncated at one), the realized gain for C ≥ 10^23 FLOP between 2020–21 and 2022–24 is:
  - 1.68 [1.21, 2.44] under Besiroglu;
  - 1.30–2.77 across the Chinchilla-family, κ-free and lab-published technologies (Meta's Llama 3: 1.30 [1.16, 1.50]);
  - 1.14–4.95 across all 14 estimated technologies, every interval above 1. [review] Only the Besiroglu interval also draws technology parameters; the others hold the technology fixed and resample 7 era-1 models.
  - Era-2 releases closed 80–94% (in logs) of the era-1 under-training gap.
- **Contrast with Gundlach et al. (M8(b)).** *"Gundlach et al. (2025) define the compute-equivalent gain and report that the Kaplan→Chinchilla rebalancing is worth about 10× at the 2025 frontier; our Besiroglu counterfactual (11.7× at 5×10^26 FLOP) agrees. Our contribution is the contrast with the gain actually realized between eras, 1.1–4.9×, which is 6–73% of the effective-compute gain implied by Ho et al.'s rate over the same interval (24% under Besiroglu)."*
- **Caveats.**
  - Era 1 has 7 models.
  - Technologies from other sweeps use their own N and D conventions, so their levels are indicative.
  - Ho's attenuated technology rates every model as over-trained, so it cannot price rebalancing.
  - [review] Truncation attributes every w < 1 to allocative error. By Proposition A-wedge(v), w < 1 can also reveal a binding data constraint or data-augmenting lab productivity. Say so where the measure is defined.
  - [review] The positive sign is close to mechanical, since all era-1 models have w < 1 and truncation only raises era 2. Lead with the size (1.1–4.9×) and the share of the gap closed, not with "every interval above 1".

**C9. Provenance of the "nine technologies" (R4 M4(d), minor 18).** The original §VI.C's "nine technologies estimated in Section IV" are, in fact:
- 5 from the m1 registry, estimated only in Online Appendix D (m1), never in §IV: Llama 3, Marin × 3, (Mis)Fitting FineWeb/C4;
- 4 from m2, in §IV/Table 4: Farseer, Gadre C4, OLMo ladder, Muennighoff datablations.

Two further points:
- Table 9 note's "12 technologies (two on the Chinchilla sweep)" counted both registries' Chinchilla rows plus DataDecide in panel A only.
- **Recommendation.** Use Table E6's grouping, which labels where each technology is estimated. Add a Table 2 or Online Appendix D row for (Mis)Fitting.

**C10. TFP dispersion (R4 M11; R1 10(e)).**
- **Wording.** *"In output units, the 90–10 ratio of family productivity is 1.9–3.0 in HellaSwag odds and 1.3–1.8 in reducible loss, bracketing manufacturing's 1.92 (Syverson 2004). In compute-equivalent units it is 7–24 (θ_C = 0.33) or 4–13 (θ_C = 0.43). Because returns to compute are small, the two unit systems differ by the factor 1/θ_C in logs. The comparison with manufacturing is a cardinalization choice, not a finding."*
- **Caveats (R1 10(e)).** Family effects absorb benchmark contamination, data similarity to the benchmark and omitted teacher compute; the compute conversion rests on one cardinal elasticity.

**C11. Algorithmic progress (R4 M4(f); R1 minor 46).**
- **Wording.** *"The parameter-augmenting share φ = g_N/g_C can lie anywhere in [−0.44, 3.37] (95% profile interval), over which T_C ranges from 5.3 to 39.6 months."*
- **Emphasis.** Put the ridge (Figure) and the dependence of T_C on the neutrality restriction (9–12 months imposed vs 6.1–10.2 unrestricted) in front. Condense the SciPy stopping-rule narrative to one sentence plus Table E5.

**C12. Other wording fixes to carry.**
- **Nerlove (R4 M8(a); R1 8(g)):** *"a constant-elasticity form fitted over a size range averages scale-dependent returns, in the sense of Nerlove's (1963) finding that returns to scale decline with firm size."* Spurious scale economies from scale-correlated input error are the paper's own mechanism; cite Basu and Fernald (1997; utilization) and Collard-Wexler and De Loecker (2016). Nerlove is not a source for "spurious".
- **Bjorck (R4 M8(c)):** *"Bjorck et al. (2025) find LR* ∝ D^−0.32 in a joint fit over models of 760M–2.7B parameters at a fixed 0.5M-token batch, and steeper exponents for smaller models (0.38 at 350M, 0.65 at 125M); Step Law's cells span 215M–1.07B non-embedding parameters, so conditioning on the batch closes 17–43 percent of the gap between the two published elasticities, depending on the exponent and on how optima are located."* [review: "16–43" used the 50M exponent (0.70), which lies outside Step Law's range. The joint 0.32 is not a per-size result; the 2.7B per-size fit is 0.42.]

---

## 5. Robustness and failures

- **Loss of support (the main cost of R1 10(b)).**
  - Removing the two production models removes every experimental point with D > 921B, and every point with N > 3.8B and D > 138B [review: was "above 3.8B × 634B tokens"]. The strict test then covers only 2021–23 releases below 6.7B parameters and 627B tokens.
  - The CV3 s.e. are up to four times the CR1 s.e.: 0.076 vs 0.019 for OLS θ_C. Leave-one-developer-out moves the OLS θ_D bias as far as −0.76 when Cerebras is dropped. G* = 1.2–3.8.
  - We report these numbers but do not base conclusions on them.
- **Extrapolation.** The strict surface evaluated on the 57 designs agrees for HellaSwag, where the quadratic surface is well determined. It does not for Winogrande (Tobit −0.31; WCR-CR1 p = 0.011, conservative p = 0.08) or ARC-C with the OLS surface (−0.37; conservative p = 0.09). These rows extrapolate a quadratic fitted to runs with D ≤ 921B (≤ 634B at N ≥ 3.8B) up to 3.8T, and should not be used.
- **Censored outputs.** The Tobit surface fixes the experimental side. On the observational side, 4 of the 20 or 57 models are at the ARC-C floor (1 on Winogrande). The level match uses the observed outputs through one nuisance parameter.
- **Few clusters everywhere.** G* is 2.5–6.7 of 19 even on the 57-model design. The CR1-normal, CV3-t and wild-bootstrap p-values agree within about 0.02 for the key rows, except θ_N under family FE: t-based p = 0.003–0.005, restricted wild p = 0.010–0.048 across the variants.
- [review] **On the 20-model design the procedures disagree by a factor of up to 10.** For OLS θ_C: CV3-t 0.77, WCR-CR1 0.23, WCR-CV3 0.39. For family-FE θ_N: 0.010, 0.14, 0.11. For Winogrande θ_C: 0.53, 0.076, 0.42. With G* = 1.2–3.8 no procedure is reliable, so Table E1 reports the largest of the three.
- **Reliability bound.**
  - The hardware-time s.d. mixes utilization dispersion (true MFU ranges from about 0.17 to 0.6) with error in chip-hours and in 6ND. It is conservative.
  - The Epoch operation-counting deviations (s.d. 0.129, driven by a few definitional outliers such as OLMo-1B at −0.57; MAD s.d. 0.015) measure disagreement about D definitions, not independent error.
- **Allocative.**
  - The era-1 sample is 7 models.
  - Registry technologies use their own conventions: Marin's fitted N-exponents (0.64–0.69, about twice Chinchilla's) imply extreme CE levels, so the 3.5–4.9 gains are indicative.
  - Top-5-per-year and Epoch-frontier samples give smaller, noisier truncated gains (1.08–2.02; some lower bounds below 1).
- **Not re-estimated here.** m4's FD/ACF/dynamic-panel and China IV rows are unchanged (see m4 memo). m5's Table 7 panel A is read, not recomputed.
- **Reproduced exactly.**
  - Table 8's point estimates on the legacy design: OLS θ_C 0.328/0.329; family FE θ_D 0.087 vs 0.348; CR1 0.161; the EIV row 0.858 with λ = 0.514.
  - m4's TFP ratios: 23.7, 19.9, 7.0, 9.1, 11.1.
  - m5's untruncated allocative gains (1.047, 1.914, …) and the φ grid-point interval.
- **No bug found in `sl.py`.** Its `wedge` and `cost_efficiency` methods were used unchanged.

---

## 6. Referee comments addressed (comment → response)

- **R4 M7:** H1, C3; Tables E2 and E4; `fig_reliability`. The reliability is reconciled: independent comparisons are identified, the EIV rows are recomputed with σ_u = 0.29, and the within-family EIV row is explained as two errors.
- **R4 M4(d):** C9; Table E6 groups each technology by where it is estimated. Recommend a Table 2 or Online Appendix D row for (Mis)Fitting (minor 18).
- **R4 M4(f):** H7, C11. φ ∈ [−0.44, 3.37], T_C 5.3–39.6 over it, on a refined grid.
- **R4 M8(a):** C12 (Nerlove wording, with Basu–Fernald and Collard-Wexler–De Loecker).
- **R4 M8(b):** C8 (Gundlach et al. define the CEG; the contribution is the realized-vs-counterfactual contrast).
- **R4 M8(c):** H8, C12. Bjorck's small-model exponents are 0.38–0.70, not "0.40–0.70". The 0.32 is a joint fit over 760M–2.7B. Over the exponents that bracket Step Law's range (0.32–0.65), the share of the gap closed is 17–43% [review: was 16–43%].
- **R4 M11:** H6, C10; Table E7 reports output (odds, loss) and input (compute) units side by side; `fig_tfp_units`.
- **R1 8(g):** C12 (Nerlove).
- **R1 8(i):** C1 ("design-matched validation across populations", Todd–Wolpin; Hotz–Imbens–Mortimer).
- **R1 9(e):** H2, C2; Table E3. The cause is the CR1 small-sample factor counting nested FE, not benchmark uncertainty. The fix is combined wild bootstrap draws and CV3 in every table.
  - [review] The first version of Table E1 paired CV3 s.e. with CR1-studentized WCR p-values. On the 20-model panel these contradicted each other (s.e. 0.076 with p = 0.23 for a bias of −0.024; s.e. 0.263 with p = 0.076 for Winogrande), which is the same inconsistency R1 flagged.
  - Fixed: Table E1 now reports the conservative p over CV3-t, WCR-CR1 and a new CV3-studentized WCR. Table E3 adds the CV3-studentized WCR (θ_D 0.011, θ_N 0.048).
- **R1 10(a):** C1. "Fails to reject in a low-power joint test", with minimum detectable biases (35% on 57 models, 50% on 20); the N/D split is demoted (C2).
- **R1 10(b):** H3; Table E1 Panel A; `fig_support`. OLMo-2 7B/13B are excluded; the strict support has 20 models.
- **R1 10(c):** C7 (renamed; state, flexible input and timing set out).
- **R1 10(d):** H5, C8; Table E6; `fig_allocative`.
- **R1 10(e):** H6, C10 (caveats; θ_C uncertainty now in the intervals).
- **R1 minor 42:** C1 (19 developers; 30 families, 15 with two or more models; strict: 6, 9, 5).
- **R1 minor 43:** C5 (Hall-type bias sentence tied to Corollary A5).
- **R1 minor 44** (and **R2 minor 25**): H1, C3 (only hardware-time comparisons are independent).
- **R1 minor 45:** H9, C6; Table E2 (AR sets with t(G−1) and normal critical values for every IV).
- **R1 minor 46:** C11; Table E5.
- **R2 minor 24 (partly):** at-or-below-chance handling. Accuracy is chance-adjusted and clipped at 0.01. The share of experimental runs at the floor is reported (52 of 86 ARC-C runs, 16 Winogrande, 2 HellaSwag), and a censored benchmark is used. Harness harmonization is **not** addressed (open issue 2).

---

## 7. Open issues

1. **Experimental support.** A genuinely experimental ladder at 7B+ and ≥ 1T tokens with downstream evaluations would restore power without production models. Candidates are open-sci-ref and Marin evaluations, if released. The OLMo-2 medium-LR 7B run in `data/raw/olmo_ladder` is an unreleased ablation, but at production N and D. We did not use it: it raises the same objection.
2. **Harness mapping (R2 24).** Leaderboard formats (10/25-shot) differ from OLMES 5-shot cloze and LLM-foundry. Re-evaluating a few observational models in the OLMES harness would test the "one-to-one output" part of the joint null.
3. **A sharper compute-error bound.** Model-card GPU-hours paired with reported, not back-calculated, MFU would separate utilization dispersion from error in 6ND.
4. **Registry dependence.** Table E6 uses the m1/m2 registries as of the recorded hashes. Rerun `run.py` (or `step_alloc.py` then `tables_figs.py`) if a revision module regenerates them.
5. **Paper edits.** The corrected claims (C1–C12) still have to be written into Online Appendix E by the writers. The introduction's family-FE sentence and the "≥ 0.988" sentence should go. Table 9 panel B should be replaced by Table E6.
6. **Out of scope.** R3 E5 (share the Ho et al. convergence finding with the authors before publication) is a decision for the lead author, not an analysis.
6a. [review] **Tobit benchmark uncertainty.** The parametric bootstrap draws run-level errors, while the OLS surfaces' wild bootstrap clusters by (design, size). A clustered or residual-based bootstrap of the Tobit fit would be more coherent. The benchmark s.e. is small relative to the observational s.e. in every reported row.
6b. [review] **φ profile.** Re-optimize the region φ ∈ [−0.5, 0] with more random starts (or warm starts from both branches) to confirm the global profile. Note that a missed optimum could only widen the interval.
6c. [review] **Allocative intervals.** Draw registry technologies' parameters from their bootstrap files where they exist (m1/m2), so that all 14 intervals carry technology uncertainty, not just Besiroglu's.
7. **Citations used** (keys in `paper/references.bib`, `lit/references.bib` or the extras):
   - gadre2024language, ruan2024observational, maiapolo2024sloth, epochai2026data, bhagia2024establishing, teamolmo2024olmo;
   - cameron2008bootstrap, webb2023reworking;
   - besiroglu2024chinchilla, hoffmann2022training, whitfill2025note, muennighoff2023scaling, ho2024algorithmic, gundlach2025origin;
   - syverson2004product, mertens2026secret, nerlove1963returns, collardwexler2016production, hall1988relation;
   - lalonde1986evaluating, dehejia1999causal, bjorck2024scaling, li2025predictablea;
   - new: todd2006assessing, hotz2005predicting, mackinnon2023cluster, cameron2015practitioner, mackinnon2018wild, carter2017asymptotic, basu1997returns.
