# Memo — module m5_progress: "algorithmic progress" as TFP growth

Module owner: Claude (m5_progress). Date: 2026-09-23. Code: `code/analysis/m5_progress/run.py`, a single entry point that regenerates every output below from raw data with fixed seeds. A clean run (`run.py`, no cache) uses about 6,200 CPU-seconds on 6 processes: 36 min wall time in the reviewer's rerun, 51 min in the builder's run under heavier load. Notation follows `paper/notes/model_spec.md`.

**Reviewer note (independent review, 2026-09-23).** This memo was audited and corrected by an independent reviewer; see `output/memos/m5_progress_review.md` for every change. Numbers below are from the reviewer's final regeneration. Main corrections:
- the converged-estimator T_C interval ([3.0, 22.7], not [2.8, 21.3]);
- the profile-likelihood interval ([4.1, 40.5] months after interpolating the LR crossing; the grid-point interval was [4.2, 37.0]);
- the objective gap between published and converged points is almost all penalty, not fit;
- the Sahal "match" was partly circular;
- m1-registry technologies were silently ignored and are now used;
- Table 7 and panel B depend on other modules' registries at run time.

**Dependency on other modules.** Table 7 rows Rm1x/Rm2x and the "m1:/m2:" technologies in panel B exist only if `output/tables/technology_registry_m1.csv` / `technology_registry_m2.csv` exist when `run.py` runs. The files used (SHA-256, mtime) are recorded under `registries_used` in `data/processed/m5_progress/dmr_summary.json` and `allocative_summary.json`. Rerun `run.py --only s2,s4,s6` whenever m1 or m2 regenerate their registries. The core results (A1–A12, panel B for the Besiroglu/Hoffmann/Farseer technologies) do not depend on them. Since the reviewer fix, the panel B bootstrap uses one random stream per (sample, technology), so those rows are also invariant to which registries are present.

In Ho et al.'s parameterization, α_year = α·g_N and β_year = β·g_D, where g_N and g_D are the factor-augmentation rates. Effective-compute growth is g_C = g_N + g_D, and the doubling time is T_C = ln2/g_C. γ = αβ/(α+β) is the frontier elasticity. Hicks neutrality means α·g_N = β·g_D, that is, α_year = β_year.

Status labels used throughout:
- **[E]** estimated in this module.
- **[A]** assumed.
- **[L]** taken from the literature.

---

## 1. Headline findings

1. **Exact replication of Ho et al. (2024) model 7** [E]. `m5_progress_replication.tex`.
   - Their data pipeline is re-implemented line by line from their notebook (commit 29c7d85): n = 231 model–benchmark rows, 144 papers, 104 transformers.
   - Their estimator (MSE + 0.0025·Σ|θ|, SLSQP from θ = 0) returns objective 0.0518129, identical to theirs. All 10 parameters match to within 4.0e-7: α_param = 0.068, β_data = 0.040, α_year = 0.004, β_year = 0.036. SciPy reports nit = 18 and nfev = 209, the same as the optimizer output printed in their notebook.
   - With their bootstrap protocol (np.random.seed(0), 100 iid draws), the bootstrap median of T_C is **8.4438 months** (8.44382859), matching their printed 8.44382833.
   - Our percentile interval is [4.39, 14.05] against their [4.52, 14.27]. The tails differ slightly, most likely because of scipy-version differences in a few ill-conditioned draws.
   - The published "8.4 months" is a bootstrap *median*. The point estimate is **8.68 months**.
   - Paper-cluster bootstrap (400 draws): [4.07, 16.93].
   - A data quirk is reproduced on purpose: 4 WT2/PTB rows carry the model's WT103 perplexity. Fixing it changes T_C to 9.0 under their code and 5.7 when converged.

2. **Ho et al.'s published estimates are not the minimizer of their own objective** [E].
   - Their code uses scipy's default SLSQP stopping rule (ftol = 1e-6). SLSQP stops after 18 iterations (nfev 209), part-way along the flat DMR ridge, with gradient components up to 9e-3.
   - Run to convergence (same objective, algorithm and start; 196 iterations), the objective falls to 0.0507227 from 0.0518129. An independent re-implementation of their `model_7`/`residuals` code by the reviewer gives the same two values.
   - A 300-start random multistart (SLSQP to convergence) confirms this. **All 300 starts end below the published objective.** The best value, 0.0507223, is essentially the converged-from-zero value, with T_C = 6.08 there. 53% of starts reach within 1e-5 of it; the rest stop at nearby local minima. See `multistart_summary.json`.
   - At the true minimizer: α_param = 0.138, β_data = 0.025, α_year = −0.059, β_year = 0.045. **T_C = 6.08 months**, with paper-cluster CI **[3.0, 22.7]** (bootstrap median 6.8).
     - The interval maps the g_C percentiles. 4 of 400 draws have g_C ≤ 0, i.e. an infinite doubling time.
     - An earlier version reported [2.8, 21.3] from direct percentiles of T_C. That treats those draws as negative doubling times and is incorrect.
   - **Almost all of the objective gap is the penalty, not fit.** The MSE barely moves (0.046416 → 0.046291, −0.27%), while Σ|θ| falls from 2.159 to 1.773. The converged point sets all three α_const terms to (essentially) zero.
     - The constants depend on the arbitrary normalization (N₀, D₀, t₀ = sample minima), so the L1 penalty picks a point on a nearly flat MSE ridge for reasons unrelated to the data.
     - The unpenalized NLS minimum (MSE 0.045303) is at T_C = 10.2.
     - The substantive lesson is therefore not "the right answer is 6.1 months". It is that the reported point depends on the optimizer's stopping rule and on a normalization-dependent penalty.
   - The confidence interval is therefore about twice as wide as published. The default stopping rule acts as an implicit regularizer toward the early part of the optimizer path, which also narrows the bootstrap spread.
   - This resembles the failure mode Besiroglu et al. found in Hoffmann et al.'s Approach 3 (optimizer stopped early, CIs too narrow).

3. **DMR non-identification is visible in the data, and it also contaminates the effective-compute rate** [E]. `m5_progress_dmr_ridge.pdf`.
   - **Bootstrap correlation** of (α_year, β_year): −0.74 (iid, 1,000 draws), −0.73 (cluster, Ho code) and −0.82 (cluster, converged). 95% of draws lie above the anti-diagonal, as in Ho et al.'s appendix (95.2%).
   - **Ridge direction.** The theoretical ridge slope −s̄/(1−s̄) is −1.20, with s̄ = 0.545 the sample-mean share of the parameter term in loss (10th–90th percentile 0.49–0.60). The first principal component of the bootstrap cloud (Ho code, iid, 1,000 draws) has slope −0.98.
     - The valley floor of the unpenalized 2-D profile is much flatter: a linear fit over the 95% joint region gives slope −0.40.
     - This is because the region is banana-shaped (Figure panel a). The local-linear ridge slope therefore describes the neighbourhood of Ho's point only, not the whole valley.
   - **The sample is not on the expansion path.** The on-path share would be a = β/(α+β) = 0.37 under Ho's exponents. So the identified combination is not g_C, and T_C is only as well identified as the factor bias.
   - **Profile-likelihood 95% CI for T_C: [4.1, 40.5] months** (iid Gaussian LR, 1 df; boundaries by linear interpolation of the LR between grid points).
     - The grid-point interval is [4.2, 37.0]; the true crossings lie in [3.9, 4.2] and [37.0, 41.6].
     - A reviewer check with 60 global random starts per grid point reproduced every boundary LR to three decimals, and confirmed that the NLS minimum (MSE 0.045303) is the global minimum.
   - The g_C profile is itself bimodal: modes at T_C ≈ 10.4 (LR 0.0) and T_C ≈ 5.4 (LR 2.95).
   - **The split is barely constrained.** The parameter-augmenting share φ = g_N/g_C has 95% profile CI [−0.39, 3.37] (interpolated; grid points [−0.25, 3.25]). Over the grid-point range the implied T_C already runs from 5.4 to 38 months.
   - **Along the valley floor** of the 2-D profile (inside the 95% joint region), T_C rises from about 5 to about 18 months. The rise is not monotone at grid resolution, and the floor jumps between two branches near α_year ≈ 0 (β_year falls from 0.07 to 0.03), which is the same bimodality.
   - **The unpenalized objective has at least two local minima** on the ridge: T_C = 5.4 (MSE 0.045886) and T_C = 10.2 (global, MSE 0.045303).
   - **Point estimates across estimators of the *same* model:** 6.1 (converged L1), 8.7 (L1, default stop), 10.2 (unpenalized NLS) and 10.5 (with E_b estimated).

4. **Hicks neutrality is not rejected, and imposing it stabilizes T_C** [E]. `m5_progress_table7.tex`, rows A3/A4.
   - Test statistic α_year − β_year:
     - converged L1: −0.104 [−0.146, 0.065], bootstrap p = 0.29. An iid F of 1.19 is descriptive only, because both MSEs are evaluated at penalized optima.
     - unpenalized NLS: 0.013 [−0.185, 0.059], p = 0.94, F(1, 221) = 0.40 (iid).
   - Imposing neutrality (Ho's model 12) gives T_C = **11.6 [6.6, 17.6]** (converged L1) and **9.2 [6.3, 14.3]** (NLS).
   - Among specifications the data do not reject, these are the tightest intervals (A4 is the narrowest in months overall). That is what a DMR restriction buys: identification of the level of progress by assumption on the bias.
     - Row A11 (imposed γ plus neutrality) is tighter in relative terms (hi/lo 1.74 vs 2.28), but the data reject it (MSE 0.0512 vs 0.0453).
   - **Caveat (Monte Carlo S1h).** If the truth has E > 0 and genuinely Hicks-neutral progress (T_C = 12), imposing neutrality *inside Ho's E = 0 form* is itself misspecified: pseudo-true T_C = 4.2 (Ho procedure) and 4.4 (global). The neutral rows may therefore *understate* T_C.

5. **Imposing experimental exponents is rejected by the cross-lab data and yields nonsensical progress rates** [E]. Table 7 rows A6–A9.
   - With Besiroglu's α = 0.348 and β = 0.366 fixed and benchmark-specific E_b estimated:
     - the MSE rises from 0.0451 to 0.113 (×2.5);
     - E_b hits its upper bound (the best observed loss);
     - estimated g_C is negative (−0.12/yr; "effective compute fell");
     - under additional Hicks neutrality, T_C = 54 [40, 99] months.
   - Hoffmann exponents give the same picture: MSE 0.099, g_C < 0.
   - So do all six sweep technologies in module m2's registry (`technology_registry_m2.csv` of 23:42, primary Huber rows; Table 7 rows Rm20–Rm25):
     - Chinchilla, Farseer and datablations: MSE 0.089–0.113 and g_C < 0;
     - Gadre-C4: MSE 0.076 and T_C = 122 months;
     - OLMo ladder: MSE 0.064 and T_C = 54 months;
     - DataDecide (pooled CE; new in m2's regenerated registry): MSE 0.060 and T_C = 42 [17.5, ∞) months;
     - every 95% interval includes non-positive growth.
   - So do all six technologies in module m1's registry (`technology_registry_m1.csv`, primal Huber rows; Table 7 rows Rm10–Rm15: Chinchilla, Meta's Llama 3 IsoFLOPs, Marin ×3, (Mis)Fitting FineWeb/C4). Their MSEs are 0.090–0.219, g_C < 0 in every case, and every 95% interval includes non-positive growth. Llama 3's own technology fits best among them (MSE 0.090 vs 0.045 unrestricted).
   - Whitfill's (2025) "first pass" divides Ho's year coefficients by the experimental exponents without re-estimating. It gives **T_C = 76.5 months** [46, ∞), i.e. 8.8× slower progress. This matches his "overstated by ~9×".

6. **Why β_data is 0.04 and not 0.37: mostly specification and measurement; the progress rate is much less affected than the exponents** [E]. `m5_progress_attenuation.tex`, `m5_progress_attenuation.pdf`.
   - **(c) E dropped.** On the experimental Chinchilla sweep itself, fitting Ho's E = 0 form collapses one exponent (Huber: α 0.375, β **0.061**; least squares: α **0.061**, β 0.377). The frontier elasticity falls from γ = **0.178** [0.169, 0.188] to **0.0525** [0.0510, 0.0548] (post-rebuild with the sl.py E_fixed tolerance fix, 2026-09-24).
     - 0.0525 is close to the sweep-average elasticity of *total* loss, γ(L−E)/L = 0.0543.
     - The Huber and least-squares E = 0 fits collapse *different* exponents but give the same γ. In the E = 0 form only the frontier elasticity is well determined.
     - So "γ" in an E = 0 model measures the total-loss elasticity. The like-for-like gap between Ho (γ = 0.021–0.025) and experiments is about 2.1–2.5×, not 7–9×.
     - Crucially, dropping E rescales year coefficients and exponents by the same factor (to first order), so g = coefficient/exponent is roughly invariant.
     - Real-data profile over an imposed common E (0 → 1.5 nats/word): γ̂ doubles (0.022 → 0.046) and β̂ doubles (0.031 → 0.068), while **T_C moves only 10.2 → 12.4 months**.
       - Within this functional form, Ho's data *prefer* E = 0: the MSE is minimized at the boundary, and the iid LR rises to 4.0 at E = 0.8 and 13.5 at E = 1.5.
       - So the case that dropping E attenuates the exponents rests on the sweep evidence and the Monte Carlo, not on Ho's data.
     - Monte Carlo on Ho's actual design, truth = Besiroglu technology with T_C = 12 months: dropping E gives γ̂ = 0.041 (α̂ collapses to 0.046). The global-optimum T_C is 11.1 (pseudo-true), with MC median 11.2 [8.8, 15.5].
     - **Therefore Whitfill's 9× first-pass correction does not apply to the E channel** *if the E = 0 model is fitted to its global optimum*.
       - With Ho's own procedure (SLSQP from zero) and D measured as dataset size (MC S2, the most realistic scenario), the pseudo-true T_C is 4.6 against a true 12. That overstates the progress rate about 2.6×, through a local optimum in which β̂ collapses (see "Local optima" below).
   - **(a) Measuring D.** Ho's D is dataset size (unique tokens). Small-dataset models train for a median of 62–140 epochs; web-scale models for 1.
     - Using Muennighoff effective data raises β̂ to 0.067 (Ho code 0.052) and T_C to **25.7** (Ho code 12.7). The CI is wide: [3.9, ∞).
     - Using raw tokens seen *lowers* β̂ (0.018–0.035) and T_C (2.6–8.0 months), because repeated epochs have diminishing value.
     - The measurement of D is the single most consequential data choice for T_C.
     - Classical EIV would need reliability 0.80 in ln D to produce β̂/β = 0.108, given R²(ln D | year, ln N, benchmark) = 0.78 (VIF 4.6). That is a noise s.d. of 1.8 log points. The actual error is not classical: ln(D_eff/U) has s.d. 0.87 and correlation −0.77 with ln D; ln(epochs) has s.d. 1.84 and correlation −0.74 with ln D.
   - **(b) Collinearity/time.** corr(ln D, year) = 0.59 and corr(ln N, ln D) = 0.87. Dropping the year terms raises β̂ only to 0.084 [0.019, 0.24]. Post-2018 and transformer-only subsamples give β̂ = 0.077 and 0.039.
   - **(d) Selection.** Ho's top-3-per-paper rule is selection on the outcome.
     - All models: β̂ 0.064, T_C 15.3 (Ho code 7.6). Top-1: β̂ 0.025, T_C 6.7. Including flagged outliers: T_C 5.2.
     - In the MC, best-3 vs random-3 selection changes nothing material (global T_C 11.0 vs 10.6).
   - **Local optima.** In the MC, Ho's procedure (SLSQP from zero) lands in a local optimum where β̂ collapses to 0.048 and T_C to **4.6** (true 12). The global optimum on the same data gives β̂ 0.389 and T_C 10.6. The same happens with a negatively correlated data-quality shock (S4: Ho procedure 4.6, global 9.0).
   - **No channel alone brings β̂_data anywhere near 0.37 in the real data**: the maximum across 15 variants is 0.091 (excluding zero-shot rows).

7. **The Kaplan→Chinchilla rebalancing: realized allocative gains were modest under the Chinchilla-family and Farseer technologies, and strongly technology-dependent; Gundlach's 10× is a counterfactual at 2025-frontier scale** [E]. `m5_progress_fig6_allocative.pdf`, Table 7 panel B.
   - **Sample:** 247 Epoch language models (2018–2026) with internally consistent (N, D, C), dense, ≤ 4 epochs.
   - **Checks against the ledger (Besiroglu technology):**
     - GPT-3 CE = 0.61 (C/C_min = 1.64 in SYNTHESIS §2.3);
     - Gopher 0.55;
     - Chinchilla 1.00, LLaMA-65B 1.00, Llama-3.1-405B 0.94;
     - Llama-3-8B 0.18 (over-trained for inference, not an error).
   - **Large models (C ≥ 1e23; 7 in 2020–21, 92 in 2022–24).** Geometric-mean CE goes 0.57 → 0.60 (Besiroglu), an allocative gain of **1.05× [0.57, 1.72]**. The median tokens per parameter rises from 1.7 to 92: the correction of under-training is offset by deliberate over-training.
     - With other technologies: Hoffmann-TeX 1.91× [1.49, 2.43]; Hoffmann-rounded 2.41× [1.92, 3.14]; Farseer-fit 1.58× [1.27, 2.04].
     - Cleaned sample (reviewer robustness; 79 era-2 models): Besiroglu 1.09× [0.58, 1.74], Hoffmann-TeX 1.99× [1.58, 2.57], Farseer 1.65× [1.33, 2.12].
   - **Top-5 compute per year:** 1.27× [0.96, 1.78] (Besiroglu) to 1.49× (Hoffmann-rounded). **Epoch frontier flag:** 1.56× [1.08, 2.22] (Besiroglu) to 1.95× (Hoffmann-rounded).
   - **Registry technologies (indicative only: N/D units and loss corpora follow each sweep).**
     - m2 (Farseer, Gadre-C4, OLMo ladder, datablations): 0.60–2.52× for C ≥ 1e23, 1.00–1.51× for top-5, 1.08–1.99× for the frontier flag.
     - m1 (Llama 3 IsoFLOPs, Marin ×3, (Mis)Fitting): 0.55–4.31× for C ≥ 1e23, 1.08–1.51× for top-5, 1.23–2.04× for the frontier flag.
     - Meta's own Llama 3 technology gives 0.74× [0.62, 0.86] for C ≥ 1e23, i.e. era-2 large models *less* allocatively efficient than era 1.
     - **Across technologies, even the sign of the realized allocative gain for C ≥ 1e23 is not robust. The robust statement is that it is at most comparable to, and under the main technologies a small fraction of, the Ho-rate gain.**
   - **Share of algorithmic gain.** Against Ho's (published-code) rate over the same interval (8.9× over 2.28 yr for C ≥ 1e23; 9.9× over 2.40 yr for top-5), the allocative share is 2% (Besiroglu, C ≥ 1e23) to 30% (Hoffmann-TeX) and 40% (Hoffmann-rounded). Top-5 with Besiroglu: 10% [−3%, 32%].
     - The denominator depends on which Ho estimator is used (Claim 2): 8.9× at 8.7 months, 22.5× at the converged 6.1 months, 6.4× at the NLS 10.2 months (C ≥ 1e23, 2.28 yr). The shares scale accordingly.
   - **The Kaplan-rule counterfactual** (N ∝ C^0.73 anchored at GPT-3) costs:
     - 1.6× at GPT-3's compute, 5.2× at 3.8e25 and **11.7× at 5e26** (Besiroglu);
     - 2.4×, 11.5× and 32× (Hoffmann-TeX);
     - the 10× cost is reached at 3.2e26 (Besiroglu) or 1.2–3.3e25 (Hoffmann).
   - Gundlach et al.'s 10× is therefore consistent with Besiroglu at the 2025 frontier. It is a counterfactual of *continuing* the Kaplan rule, not the allocative gain actually realized between eras.
   - **Under Ho's own attenuated technology,** CE falls 0.098 → 0.009 (it rates Chinchilla at CE 0.03 vs GPT-3 at 0.12). An attenuated β makes data nearly worthless, so a regression with Ho's exponents *cannot* credit rebalancing to N and D. The time trend absorbs it instead.

8. **Compute-equivalent gains, transformer vs. other architectures: scale dependence is not identified in these data** [E]. `m5_progress_ceg_sahal.tex`, `m5_progress_ceg.pdf`.
   - Ho's model 13 (architecture-specific exponents) gives Δγ = γ_tr − γ_non-tr = 0.008 [−0.012, 0.034] (converged L1, p = 0.46) and 0.001 [−0.011, 0.028] (NLS, p = 0.53).
   - Point estimates of CEG at 1e19 FLOP are 0.45 (L1) vs 14.5 (NLS). The bootstrap intervals for CEG levels span more than six orders of magnitude.
   - Proposition P4 (constant CEG ⇔ equal exponents and E) can be neither confirmed nor rejected with Ho's data. Gundlach et al.'s scale-dependent LSTM→transformer gain [L] needs designed experiments.

9. **Sahal-type bias of naive frontier regressions** [E].
   - Regressing log loss on log compute *without* a time trend inflates the compute elasticity by **1.15× [1.10, 1.21]** on all 231 rows, and by **1.44× [1.24, 1.78]** on record-setting releases (n = 36).
   - Both factors lie inside the ledger's range γ/(1−s_A) ∈ [1.05, 1.67] for s_A ∈ [0.05, 0.40] (SYNTHESIS §2.2).
   - The "implied" s_A values (0.13 and 0.31) are backed out *from* the inflation factors, so they are not a test of the formula.
   - The independent check uses growth rates:
     - Among record-setters (corr(ln C, year) = 0.87, close to the formula's assumption that compute moves with time), physical compute grows 2.64 and algorithmic progress 1.54 log-points/yr. That gives s_A = 0.37 and a predicted factor of 1.58, consistent with the observed 1.44 [1.24, 1.78].
     - On all rows (corr 0.62) the same rate calculation predicts 1.77 against the observed 1.15. There the formula's assumption fails and the ordinary omitted-variable formula applies instead.
   - "Compute" here is 6·N·(dataset size), which ignores epochs [A].

---

## 2. Methods

### 2.1 Data
- **Ho et al. (2024) sheet** (`data/raw/ho2024/algorithmic_progress.csv`, 408 rows; Google-Sheet export gid=2087221150).
  - Their pipeline is re-implemented line by line in `common.load_ho()`, following their `section3.ipynb` (commit 29c7d85, MIT). The notebooks are saved in `data/raw/ho2024/code/` by `code/data/download_m5_progress.sh`.
  - Filters: numeric N and D > 0; `Include?` ≠ 0 (NaN kept); `Outlier?` ≠ 1; at least one perplexity; stack by benchmark; `uncertain` == 0; drop 4 named systems (µP and LoRA fine-tunes); sort by paper and perplexity; keep the top 3 per paper.
  - Result: **231 rows, 144 papers** (WT103 103, PTB 80, WT2 48).
  - D is the sheet's dataset size (unique tokens), **not tokens processed**. Loss is log perplexity (mostly nats per word).
- **Epochs** are known for 155/231 rows. Missing epochs are imputed by the median of known epochs in the same log10(D) bin; D ≥ 1e10 is set to 1 epoch.
  - Effective data follows Muennighoff et al. (2023): D′ = U[1 + R*(1 − e^{−(e−1)/R*})] with R* = 15.4 [L].
- **Chinchilla sweep:** Epoch extraction via `sl.chinchilla_extraction` (n = 240), estimated with Besiroglu et al.'s estimator (`sl.fit_chinchilla`). It reproduces E 1.817, α 0.347, β 0.367 and γ 0.178.
- **Farseer grid** (`1222_full.csv`, 404 runs; no license, so used but not redistributed): Chinchilla form fitted with the same estimator, with N = parameters including embeddings (`N_add_emb`) and L = training loss. Result: α 0.376, β 0.251, γ 0.150, a 0.40 [E].
- **Epoch all-models:** language-domain models released 2018-01-01 to 2026-09-23. Sample funnel:

  | Filter | Models remaining |
  |---|---|
  | Language models with N and C | 792 |
  | + dataset size | 544 |
  | + not fine-tunes | 454 |
  | + not MoE (name match) | 428 |
  | + C/(6N) within a factor 3 of dataset × epochs | 339 |
  | + ≤ 4 epochs | **247** |

  - D := C/(6N), so that 6ND = C exactly.
  - Frontier sets: C ≥ 1e23; top-5 by compute per calendar year among disclosed models; Epoch `Frontier model` flag.

### 2.2 Estimators and inference
- **Ho model:**
  - log ppl = E_b + exp(a_b − α_year(t−t0) − α ln(N/N0)) + exp(b_b − β_year(t−t0) − β ln(D/D0)).
  - E_b = 0 in Ho. N0, D0 and t0 are sample minima.
  - Class `HoModel` (`common.py`) supports:
    - E none, common, per benchmark or fixed;
    - fixed exponents;
    - fixed γ;
    - Hicks neutrality;
    - fixed (α_year, β_year), fixed g_C or fixed φ (profiles);
    - architecture-specific exponents (Ho model 13);
    - a vocabulary control.
- **Three optimizers, always labelled:**
  - "Ho code": SLSQP with scipy defaults on MSE + 0.0025·Σ|θ|, starting from 0. This is exactly their procedure.
  - "Converged": the same objective, ftol = 1e-13, starts at 0 and at the full-sample optimum.
  - "NLS": unpenalized `least_squares` (TRF, bounds 0 ≤ E_b < min log ppl_b and 0 < exponents ≤ 3), with deterministic and random multistarts. Nesting is verified.
- **Profiles:**
  - 2-D grid over (α_year, β_year) ∈ [−0.12, 0.08] × [−0.06, 0.16], step 0.005, other parameters profiled out;
  - 1-D profiles in g_C and φ;
  - Gaussian LR = n·ln(MSE/MSE_min). This assumes iid errors; papers are clustered, so read the profile CIs as indicative.
  - 1-D interval boundaries are located by linear interpolation of the LR between the last grid point inside and the first outside (`lr_crossing` in `s2_dmr.py`). Grid-point intervals are also stored in `dmr_summary.json`.
- **Inference:**
  - Paper-cluster bootstrap, 400 draws, fixed seeds.
  - Ho's iid protocol is used for the replication.
  - T_C intervals come from percentile intervals of g_C, mapped monotonically; they are ∞ when g_C ≤ 0 lies inside. This now holds everywhere, including the replication table.
    - The one exception is the column that reproduces Ho et al.'s own protocol (iid, B = 100), where percentiles of T_C are taken directly, as in their code. No draw there has g_C ≤ 0, so the two methods differ only through interpolation: [4.39, 14.05] direct vs [4.39, 13.94] mapped.
- **Monte Carlo on Ho's design** (261 rows before the top-3 rule):
  - Truth: Besiroglu technology with E, Hicks-neutral progress and T_C = 12 months, evaluated at the actual (N, effective D, date). Benchmark scale is matched to the observed mean log perplexity.
  - Noise s.d. 0.215 (Ho residual). 204 replications.
  - Channels switched on one at a time: E dropped; D observed as dataset size; best-3 vs random-3 per paper; ψ_D shock with corr(ψ_D, ln D) = ±0.5.
- **Allocative efficiency:** CE = C_min^T(L^T(N, D))/(6ND), computed with `sl.Chinchilla.cost_efficiency`.
  - Technologies: Besiroglu (200 bootstrap draws propagate uncertainty), Hoffmann TeX and rounded, Farseer fit, and Ho model 7 (evaluated at each release date).
  - Era allocative gain = ratio of geometric-mean CE (2022–24 vs 2020–21), with a bootstrap over models within era (400 draws). Each (sample, technology) cell has its own seeded random stream, so the cell is invariant to which other technologies are present.
  - Robustness sample "C ≥ 1e23, cleaned" (reviewer addition) drops:
    - derivative releases (Instruct/Chat/-VL-), MegaScale systems runs, Falcon Mamba, FragLlama;
    - exact re-releases (same Organization, N, C);
    - D/N exactly 20 (possibly a Chinchilla-rule imputation, which forces CE ≈ 1);
    - Epoch "Speculative" entries.
    - This removes 13 era-2 models.
  - Comparison with the Ho-rate gain exp(g_C·Δt).
  - Kaplan counterfactual: N = 174.6B·(C/3.14e23)^0.73.
- **CEG:** f(C) = C_non-tr(L*_tr(C))/C at 2023 on WT103. With E = 0, ln f is linear in ln C with slope γ_tr/γ_non-tr − 1.
- **Sahal regression:** OLS of ln L on ln(6ND) plus benchmark dummies, with and without a year trend. Paper-clustered SEs; paper-cluster bootstrap for the ratio.

---

## 3. Table and figure inventory

### Tables (`output/tables/`)

| File | Content |
|---|---|
| `m5_progress_replication.tex/.csv` | Replication of Ho et al. model 7: published vs our estimates (their code) with iid and cluster bootstrap CIs, plus the converged optimum of the same objective. The last rows split the objective into its fit (MSE) and penalty (Σ\|θ\|) parts. |
| `m5_progress_table7.tex/.csv` (**paper Table 7**) | Panel A: T_C, g_N, g_D and γ under 13 identifying assumptions (Ho code, converged, NLS, Hicks-neutral, E_b, effective data, experimental γ, Whitfill first pass, imposed experimental exponents). Also: imposed m1-registry technologies (Rm10–Rm15) and m2-registry technologies (Rm20–Rm25), present only if those registries exist at run time (files used: m1 of 2026-09-23 22:46, m2 of 23:42; see `registries_used`). Panel B: allocative gain 2020–21 → 2022–24 by technology and its share of the Ho-rate gain, plus the cleaned-sample Besiroglu row. Components are in `m5_progress_table7_panelA.csv` and `_panelB.csv` (all samples: C ≥ 1e23, top-5 per year, Epoch frontier flag, all LMs, C ≥ 1e23 cleaned). |
| `m5_progress_attenuation.tex` | Channels of attenuation. Panel A: 15 data variants (`_panelA.csv`). Panel B: Chinchilla sweep with and without E (`_panelB.csv`). Panel C: Monte Carlo on Ho's design (`_mc.csv`). |
| `m5_progress_E_profile.csv` | Estimates and T_C under an imposed common E = 0…1.5. |
| `m5_progress_reliability.csv` | Reliability ratios that classical measurement error in ln D would require. |
| `m5_progress_ceg_sahal.tex` | Panel A: CEG of transformers vs other architectures (`m5_progress_ceg.csv`). Panel B: Sahal regressions (`m5_progress_sahal.csv`). |
| `m5_progress_named_models_CE.csv` | CE and wedge w for 16 named flagship models under 4 technologies. |
| `m5_progress_kaplan_counterfactual.csv` | CE and CEG of the Kaplan allocation rule at C = 3.1e23 … 1e27. |

### Figures (`output/figures/`, .pdf and .png)

| File | Content |
|---|---|
| `m5_progress_dmr_ridge` | (a) 95% joint profile-LR region in (α_year, β_year), bootstrap cloud, SLSQP iterates from Ho's stopping point to the converged optimum, NLS optimum and Hicks-neutral line. (b) T_C along the valley floor. (c) Profile LR over the factor-bias share φ with implied T_C. |
| `m5_progress_attenuation` | (a) α̂ and β̂ with CIs across 15 data variants vs sweep values, from the *converged* L1 estimator (so the baseline row shows α̂ 0.138, β̂ 0.025, not Ho's published 0.068/0.040; Ho-code values are in the table's last column). (b) Exponents rise with imposed E. (c) T_C barely moves with E. |
| `m5_progress_fig6_allocative` (**paper Figure 6**) | (a) Farrell CE of Epoch LMs over time (Besiroglu technology) with era geometric means. (b) CEG of the Kaplan rule vs compute under 3 technologies (Besiroglu, Hoffmann *rounded*, Farseer), with Gundlach's 10×. (c) Allocative part of the Ho-rate gain by technology (Besiroglu, Hoffmann-TeX, Farseer) and sample. |
| `m5_progress_ceg` | CEG(C) of transformers under model 13, with 5th/95th bootstrap percentile lines and the non-transformer support shaded. Reference marks: Ho et al.'s 7.2× [L], and Gundlach et al.'s small-scale 6.28× [L]. The x-position of the Gundlach mark is *assumed* [A]: C = 6N·20N with N = 3.6M, since their ablations are reported at fixed loss, not fixed FLOP. |

### Processed data (`data/processed/m5_progress/`)
- `ho_df_head.csv`: Ho estimation sample.
- `epoch_lm_sample(_CE).csv`: Epoch analysis sample with CE.
- Bootstrap draws (`boot_m7_*.npy`).
- Profile grids (`dmr_profile_grid.csv`, `dmr_profile_1d.csv`, `dmr_valley_floor.csv`) and the SLSQP path (`slsqp_path.csv`).
- JSON summaries.
- Fitted Chinchilla/Farseer technologies.

---

## 4. Claims for the paper

1. **"Ho et al.'s estimates are reproducible exactly, and the headline is a bootstrap median."**
   - *Evidence:* `m5_progress_replication.tex`. Objective 0.0518129 identical; parameter differences ≤ 4e-7; bootstrap median T_C 8.4438 identical; point estimate 8.68.
   - *Caveat:* the bootstrap tails differ slightly from theirs ([4.39, 14.05] vs [4.52, 14.27]), probably because of scipy versions.

2. **"The published point is not the minimizer of the authors' own objective: scipy's default stopping rule halts SLSQP part-way along a DMR ridge. At the minimizer of the same objective T_C is 6.1 months with a paper-cluster CI of [3.0, 22.7], roughly twice as wide as published. Because the fit (MSE) changes by only 0.3%, and the move is driven by a normalization-dependent L1 penalty, the point estimate of T_C depends on estimator choices (6.1–10.2 months) that the data cannot arbitrate."**
   - *Evidence:* replication table ("Converged" columns) and Table 7 row A1c.
   - Objective 0.0507227 < 0.0518129 (reproduced by an independent re-implementation of the authors' code); 300-start multistart (`multistart_summary.json`, `multistart_ho_objective.csv`); 18 vs 196 SLSQP iterations (`dmr_summary.json`: `slsqp_default_nit`; `slsqp_path.csv`).
   - MSE 0.046416 (published) vs 0.046291 (converged); Σ|θ| 2.159 vs 1.773 (replication table, last rows).
   - *Caveats:*
     - The L1 penalty itself selects a point on the ridge, and it penalizes constants whose values depend on the arbitrary normalization (N₀, D₀, t₀). Unpenalized NLS gives 10.2 months.
     - Do not present 6.1 months as the "correct" rate. Present it as evidence that the published point and CI are optimizer artefacts on a flat ridge.
     - The claim is about numerical optimization, not about Ho et al.'s substantive conclusions (progress of order 0.5–2 years per doubling survives).
     - This echoes [besiroglu2024chinchilla] on [hoffmann2022training] and the BLP convergence literature [knittel2014estimation; dube2012improving].

3. **"Only a share-weighted combination of α_year and β_year is identified (DMR; [diamond1978measurement]); because cross-lab data are off the expansion path, even the effective-compute doubling time is weakly identified: the profile-likelihood 95% CI is [4.1, 40.5] months, and T_C ranges 5–38 months across factor-bias shares inside their CI."**
   - *Evidence:* `m5_progress_dmr_ridge.pdf`, `dmr_summary.json` (`TC_profile_ci_interp`). Bootstrap corr −0.73 to −0.82; ridge slopes −1.20 (local theory), −0.98 (bootstrap PC1) and −0.40 (valley floor).
   - *Caveats:*
     - The LR CIs assume iid Gaussian errors, and papers are clustered.
     - Bounds are interpolated between grid points: the lower crossing lies at g_C ∈ (2.00, 2.15), i.e. T_C ∈ (3.9, 4.2); the upper at g_C ∈ (0.200, 0.225), i.e. T_C ∈ (37.0, 41.6).
     - The profile LR stays near 3 over g_C ∈ [1.25, 1.85] (secondary mode), so the lower bound is sensitive to the critical value. Under paper-clustered errors the χ²(1) calibration is not justified.

4. **"Hicks neutrality (α·g_N = β·g_D) cannot be rejected; imposing it yields T_C ≈ 9–12 months, with the tightest intervals among specifications the data do not reject ([6.3, 14.3] NLS; [6.6, 17.6] converged L1)."**
   - *Evidence:* Table 7 rows A3/A4; neutrality tests in `dmr_summary.json` (bootstrap p = 0.29 and 0.94; iid F(1, 221) = 0.40 for NLS, the only valid F).
   - *Caveats:*
     - Non-rejection reflects low power (DMR), not evidence of neutrality.
     - When the truth has E > 0, neutrality imposed in the E = 0 form is misspecified. MC S1h (true T_C = 12) gives pseudo-true T_C = 4.2–4.4, so the neutral estimates may be biased toward faster progress.

5. **"Imposing experimental exponents in the cross-lab model is rejected (MSE ×1.3–4.8 across 14 experimental technologies; ×2.5 for Besiroglu) and yields negative or implausibly slow progress. Whitfill's (2025) first-pass 9× correction does not apply to the E=0 channel, because dropping E attenuates exponents and year coefficients proportionally."**
   - *Evidence:*
     - Table 7 rows A6–A9 and registry rows Rm10–Rm15 (six sweep technologies from m1) and Rm20–Rm25 (six from m2);
     - E profile: T_C 10.2 → 12.4 while γ̂ doubles;
     - Monte Carlo S1: T_C 11.1 vs 12 true, γ̂ 0.041 vs 0.178;
     - Chinchilla sweep E = 0 fit: γ 0.0525 ≈ sample-mean total-loss elasticity 0.0543.
   - *Caveats:*
     - The first-order argument holds locally. In the MC the progress bias reappears whenever the E = 0 fit collapses the *D*-exponent (Ho-procedure local optima). In the most realistic scenario (S2: E dropped, D = dataset size, Ho's procedure) the pseudo-true T_C is 4.6 vs 12, i.e. progress overstated about 2.6×. The sign and size of the bias in Ho's real data therefore remain uncertain.
     - Within Ho's data a common E is not supported (the LR against E = 0 is 13.5 at E = 1.5), so the E channel is argued from the sweep and the MC, not observed in Ho's data.
     - Whitfill's transmission mechanism [whitfill2025note] is a separate channel that we do not rule out.

6. **"The like-for-like observational–experimental gap in the frontier elasticity is about 2–2.5×, not 7–9×."**
   - *Evidence:* Ho γ̂ 0.021–0.025 (Table 7) vs 0.0525 [0.0510, 0.0548] (Chinchilla sweep in Ho's E = 0 form, attenuation Panel B).
   - Imposing γ = 0.0525 gives T_C 21.5 [9.9, 48] (A10), or 25.8 [20, 35] with neutrality (A11). The data reject it: MSE 0.0509 vs 0.0453.
   - *Caveat:* the Chinchilla sweep (MassiveText, subword tokens) and Ho's word-level WT103/PTB losses differ in regime and units, so R/L need not be comparable.

7. **"How D is measured drives the progress estimate: correcting for repeated epochs (effective data) raises T_C by 1.5× (Ho code, 8.7 → 12.7) to 4× (converged, 6.1 → 25.7), while raw tokens-seen lowers it in three of four estimator/imputation combinations (2.6, 4.7 and 4.7 months; 8.0 in the fourth)."**
   - *Evidence:* attenuation Panel A rows a1–a3.
   - *Caveats:*
     - 33% of epoch counts are imputed.
     - Muennighoff's R* = 15.4 was estimated on unregularized LLMs; heavily regularized small RNNs may extract more from repetition.
     - The CIs are very wide.

8. **"Selection by the top-3-per-paper rule is not the main source of attenuation."**
   - *Evidence:* attenuation rows d1–d4 (β̂ 0.023–0.091); MC S3 vs S3r.
   - *Caveat:* T_C still moves (5.2–16.6 months, converged) because every change of sample moves the ridge point.

9. **"Realized allocative gains from the Kaplan→Chinchilla rebalancing among large models (C ≥ 1e23) were 1.0–2.4× across the Besiroglu, Hoffmann and Farseer technologies (1.3–1.9× for compute leaders), i.e. 2–40% of the algorithmic gain implied by Ho's rate over 2020–21 → 2022–24; Gundlach et al.'s 10× is the counterfactual cost of keeping Kaplan's rule at 2025-frontier compute (11.7× at 5e26 under Besiroglu)."**
   - *Evidence:* Table 7 Panel B; `m5_progress_table7_panelB.csv`; `m5_progress_kaplan_counterfactual.csv`; Figure 6.
   - *Caveats:*
     - Era 1 has only 7 large models with disclosed (N, D).
     - CE is relative to the *training-only* frontier. Over-training for inference (Prop. 4) lowers CE by design, and it dominates era 2 among smaller large models.
     - Results depend on the technology: Besiroglu vs Hoffmann differ by a factor of 2.
     - Under the sweep technologies in the m1/m2 registries (indicative only, since their units differ) the C ≥ 1e23 gain ranges 0.55–4.3×, and Meta's Llama 3 technology gives 0.74×. Present the 1.0–2.4× range as conditional on the Chinchilla-family and Farseer technologies.
     - The "share" uses Ho's published-code rate (8.9× over the interval). With the converged or NLS Ho estimates (Claim 2) the denominator is 22.5× or 6.4×. Also, the share is a comparison of magnitudes, not an accounting decomposition (see Claim 10).
     - Robust to cleaning the era-2 sample of duplicates, derivatives, systems runs, non-transformers and possibly imputed D (gains 1.09 / 1.99 / 1.65).
     - D is inferred as C/(6N), e.g. Gopher gets 376B tokens vs the 300B reported.

10. **"An attenuated technology cannot price misallocation: under Ho's own estimates, CE falls tenfold from 2020–21 to 2022–24 (the move to more tokens looks harmful), so rebalancing gains are necessarily loaded onto the time trend."**
    - *Evidence:* Table 7 panel B CSV, technology "Ho model 7" (0.098 → 0.009); named-model CSV (Chinchilla CE 0.03 vs GPT-3 0.12).
    - *Caveat:* Ho's technology is in WT103 word units with D = dataset size, so the A/B ratio is not unit-comparable to Epoch token counts. Treat as qualitative.

11. **"Scale dependence of transformer CEG (P4) is not identified in Ho's data."**
    - *Evidence:* `m5_progress_ceg.csv`. Δγ 0.008 [−0.012, 0.034] and 0.001 [−0.011, 0.028]; CEG at 1e19 = 0.45 vs 14.5 across estimators.
    - *Caveat:* above 3e18 FLOP the non-transformer law is extrapolated.

12. **"Naive frontier regressions without time controls overstate the compute elasticity by 1.15× (all models) to 1.44× (record-setters). For record-setters this is consistent with the Sahal formula γ/(1−s_A) evaluated at independently measured growth rates (s_A = 0.37 predicts 1.58) [nagy2013statistical]."**
    - *Evidence:* `m5_progress_sahal.csv` (columns `inflation`, `sA_from_rates`, `inflation_formula`).
    - *Caveats:*
      - The elasticity is in total-loss units (E = 0), and record-setters are only n = 36 (and selected on the outcome).
      - The "implied" s_A values (0.13, 0.31) are backed out *from* the inflation factors, so they do not test the formula.
      - On all rows the rate-based prediction (1.77) does not match the observed 1.15. ln C and year are only moderately correlated there (0.62), so the collinearity behind the Sahal formula fails.
      - Compute is 6·N·(dataset size), which ignores epochs.

---

## 5. Robustness and failures

- **Local optima everywhere.**
  - The penalized objective (Ho code vs converged: 8.7 vs 6.1 months) and the unpenalized objective (local minima at 5.4 and 10.2) are both multimodal along the DMR ridge.
  - Bootstrap results depend on starting values. We start the converged bootstraps at 0 and at the full-sample optimum and keep the better one. In a development run (not part of the pipeline), a single start at the full-sample optimum gave a converged CI of [3.0, 14.6] instead of [3.0, 22.7].
  - Report estimator and optimizer settings with every number.
- **Word-level-vocabulary subsample (n = 134) is degenerate:** β̂ = 0 at the bound and T_C ≈ 0. It is reported for transparency; do not use it.
- **Experimental-exponent specifications (A7–A9) are degenerate:** E_b goes to its upper bound (the best observed loss), and g_C is negative. These are failures of the model on these data, not estimates to quote as "progress".
- **Hicks-neutral fits in an E = 0 model are misspecified if the truth has E > 0.**
  - A proportional shift of A and B with E > 0 does not imply equal year coefficients in the E = 0 pseudo-true fit.
  - MC scenario S1h (`m5_progress_attenuation_mc.csv`) gives T_C ≈ 4.2–4.4 vs 12 true.
  - So the neutral rows are an identifying *assumption*, not a validated fact.
- **Monte Carlo realism.** The truth (Besiroglu technology in token units, rescaled per benchmark) produces more cross-sectional dispersion in simulated log perplexity (about 0.8–8 in a development check) than observed (1.6–5.1). The MC shows channel mechanics, not a calibrated decomposition of Ho's 0.04.
  - The correct-specification MC (S0) shows finite-sample downward bias of T_C (median 9.9 vs 12): weak identification of E_b with noise.
- **Profile CI boundaries.** Both 1-D profile intervals are interior to their grids (g_C ∈ [0.15, 3.5]; φ ∈ [−1.5, 6]).
  - Grid-point intervals: T_C [4.2, 37.0] and φ [−0.25, 3.25].
  - Interpolated crossings: T_C [4.1, 40.5] and φ [−0.39, 3.37].
  - The g_C grid step is 0.025/yr below 0.6 and 0.15/yr above; the φ step is 0.25.
- **Reviewer checks (not part of the pipeline; scripts in `code/analysis/m5_progress/review_checks/`, results in the review memo):**
  - Re-optimizing the 1-D profile at 12 boundary points with 60 global random starts reproduced every LR to three decimals, and confirmed that MSE_min = 0.045303 is global.
  - In A7, all three E_b sit exactly at their upper bounds (best observed loss − 0.001).
  - An independent implementation of Ho's `model_7` objective reproduces 0.0518129 (published point) and 0.0507227 (converged point).
- **Allocative sample cleaning.** Dropping duplicates/derivatives, systems runs, non-transformers, D/N = 20 exactly and speculative entries (13 era-2 models) *raises* the allocative gains slightly (see panel B, 'C >= 1e23, cleaned'). The headline range is robust to this cleaning.
- **Epoch dataset-size parsing.** 24 Epoch rows store 'Training dataset size (total)' as a comma-separated pair (e.g. Chameleon-34B, BlueLM 7B). These are coerced to missing and dropped. Among 2018+ language models with N and C, only two would otherwise enter a sample: Chameleon-34B (mixed-modal) and BlueLM 7B, both era-2 with C ≥ 1e23 (2 of about 94). The other four fail the ≤ 4-epoch filter anyway. Not fixed; the effect on the era geometric means is negligible.
- **Allocative analysis:**
  - Era 1 is small: 7 models with C ≥ 1e23, or 10 in the top-5 set.
  - The Epoch frontier flag sets are 7 and 6 models.
  - All-LM geometric-mean CE *falls* from 0.70 to 0.59 (Besiroglu) because of over-training.
  - MoE exclusion is by name match and may miss some models.
- **Replication tails** (see Claim 1). Ho et al.'s notebook loads *cached* bootstrap results (`use_cached_bootstraps = True`), so their printed percentiles come from an earlier bootstrap file we cannot inspect. Library differences at generation time in a few ill-conditioned draws would explain the tail gap. The medians agree to 3e-7, so most draws are identical.
  - **Ho et al.'s CSV** was downloaded as a live Google-Sheet export. If the sheet changes, `s1` prints the objective/parameter comparison with the published values, which would reveal it.
- **sl.py:** no bug found. One minor note: `fit_chinchilla(E_fixed=...)` runs L-BFGS-B with default tolerances, unlike the free-E branch (ftol = 1e-15, gtol = 1e-12). For the E = 0 Chinchilla refit, an independent least-squares fit gives the same γ (0.0528 vs 0.0525), so it does not matter here.

---

## 6. Open issues

1. **Common output units.** Ho's losses are word-level WT103/PTB/WT2 perplexities, while the experimental technologies are subword-token losses on other corpora. A BPB bridge (OLMo ladder, DataDecide WikiText-103) would make the "total-loss γ" comparison exact.
2. **Epoch counts.** Hand-coding the 76 missing epochs, or dropping those rows, would firm up the effective-data result (Claim 7), which is currently the largest single driver of T_C.
3. **Transmission/selection with lab identities.** Organizations are missing for 223 of 231 rows, so lab fixed effects or control-function corrections are infeasible on this sheet. Linking to Epoch's `WikiText and Penn Treebank data` flag could recover organizations.
4. **Superlative (Törnqvist) progress index and Hall-type accounting bias (P8)** were not implemented.
5. **Vintage-specific cross-sectional σ.** Identifying the factor bias would require IsoFLOP sweeps at several vintages on a common output (Chinchilla 2022, Porian/Gadre 2024, Farseer 2025). This module does not attempt it.
6. **Coordination:**
   - `technology_registry_m1.csv` and `technology_registry_m2.csv` are used automatically when present, through `common.read_registry`, which handles both formats. The builder's original filter (`estimator == 'huber'`) silently ignored the m1 format; the reviewer fixed this.
     - `s2` imposes the exponents of the primary Huber row per dataset (Table 7 rows Rm1x, Rm2x).
     - `s4` adds the non-Chinchilla technologies as robustness technologies for CE (rows "m1: …", "m2: …" in `m5_progress_table7_panelB.csv`).
   - The registry files used (mtime, SHA-256) are recorded in `dmr_summary.json` and `allocative_summary.json` (`registries_used`). Rerun `run.py --only s2,s4,s6` after m1 or m2 regenerate their registries. A missing registry now triggers a logged WARNING instead of silently dropping rows.
   - Units of N and D in the registry technologies follow each sweep's conventions (see the `N_convention`/`D_definition` columns), so CE under them is only indicative for Epoch token counts.
7. **Notification to authors:** the optimizer-tolerance finding (Claim 2) should be communicated to Ho et al. before publication, as courtesy and to check for any unrecorded setting.

### Citations used (keys in `lit/references.bib`)
ho2024algorithmic, whitfill2025note, gundlach2025origin, diamond1978measurement, besiroglu2024chinchilla, hoffmann2022training, muennighoff2023scaling, kaplan2020scaling, li2025predictableb, farrell1957measurement, nagy2013statistical, knittel2014estimation, dube2012improving, griliches1986errors, leonledesma2010identifying, erdil2022algorithmic, sardana2024beyond.
