# m8_measurement — independent review (replication + referee)

Reviewer: independent replicator, 2026-09-23. Scope: `code/analysis/m8_measurement/*`, all module outputs, and the memo `output/memos/m8_measurement.md`.

## 0. Verdict in brief

- **The replication is clean.** I deleted every module output and re-ran `run.py`. All 16 CSV tables, 5 TeX tables, 20 processed files and `results_summary.json` came back **bit-identical** (26.8 min; load average about 30–45). The Porian port matches their released code step by step. All ten Porian Table 1 exponents, the Kaplan-adjusted 0.717 and the N*(5.88e23) values were checked against the paper text (arXiv:2406.19146). The Pearce–Song inputs (ω = 47,491; 0.78 / 0.74; range 790–1.58B) were checked against arXiv:2406.12907, and Bjorck's setup (fixed 0.5M-token batch; N ≥ 760M; C = 1.55e-3, exponents −0.23 / −0.32) against arXiv:2409.19913.
- **The theory is right.** I re-derived Proposition M (a_m = β/(β+ακ−η), κ = 1 − s(1−θ), η = (1−θ)s(κ−θ)/κ) and Proposition F (a_obs − a = −∂_{ln C}[Δ/f''], f'' = (α+β)γR*, constant-gradient form −(ι_n−ι_d)E/((α+β)R*)) by hand. Both agree with the code and the sympy/numerical checks.
- **I confirmed the builder's sl.py bug report independently.** From the free-E NLS start, `sl.fit_chinchilla(E_fixed=1.4)` stops at a = 0.565 (objective 1.238e-4) instead of the optimum 0.5735 (1.1447e-4).
- **Two headline interpretations were overclaimed, and one important caveat was missing.** I rewrote them (major issues M1–M4 below). The numerical content of the builder's other headline findings stands.
- Final state: code fixed and memo corrected. After the fixes, a final clean rerun from deleted outputs completed without errors in 17.4 min. It produced 18 table files, 20 processed files and 4 figure files. Every number now quoted in the memo matches these outputs.

## 1. Issues found and fixes

| # | Severity | Issue | Fix | Status |
|---|---|---|---|---|
| M1 | major | **"σ* is far more stable than a" (claim 7; design recommendation 2 says "σ* and γ are more stable") is not supported.** The comparison was in natural units. σ* = 2/(2+α+β) compresses α+β by about 0.3, so natural units are not comparable. Relative to α+β, a is about equally fragile along the E profile (half-range/midpoint 0.25 vs 0.27) and across N conventions (0.16 vs 0.16). **γ is more fragile than a** along the E profile (0.41; γ moves 0.076 → 0.181). In units of the frontier's own sampling SE, σ* is never more stable than a, and is 2–3× *less* stable along the E profile (7.0 vs 2.8 SE), across N conventions (4.4 vs 1.5 SE) and with one random configuration per cell (7.0 vs 2.5 SE). | Added `stability_comparison()` → `m8_measurement_stability.csv` and `results_summary.json:stability_comparison`. Rewrote claim 7, added memo §1.5(vii), and rewrote design recommendation 2 (removed the γ claim). Flagged the implication for the SYNTHESIS headline ("σ is more stable than a"). | fixed |
| M2 | major | **"The Bjorck–Step Law sign disagreement is mostly conditional vs unconditional demand" is an overclaim.** Conditioning on the batch closes only 26% (smoothed; SE 0.02) to 43% (grid; SE 0.13) of the 0.63 gap between the published +0.307 and −0.32. The "batch co-scaling explains ~90%" figure applies to Step Law's *own* unconditional elasticity, and it is sensitive to how optima are located: 0.91 (SE 0.15) with smoothed optima vs 0.56 (SE 0.24) with grid argmins, from 0.016 + 0.295 × 0.543 = 0.176 vs 0.288. No SE was reported. | Added `S.lechatelier_bootstrap()`, a cell-cluster bootstrap with 1,000 draws covering both optimum definitions, to `results_summary.json:lr_decomposition_boot` and demand table Panel D. Rewrote claim 11 and §1.5(vi). | fixed |
| M3 | major | **Step Law's N is the non-embedding count (Kaplan's convention), which the module's own Proposition M says biases a upward. This was not disclosed.** Counting the head (N + 65,536·h, head share 11–23%) moves the NLS frontier a from 0.563 to 0.465 with E free (σ* 0.786 → 0.752) and from 0.573 to 0.550 at E = 1.4. Also counting untied input embeddings gives 0.405 and 0.531. | Added `S.frontier_count_robustness()` → `m8_measurement_steplaw_count.csv`, plus caveats in the Step Law table note, memo §1.5(i), §5 and design recommendation 1. | fixed (disclosed) |
| M4 | major | **Claim 8 ("untuned configurations mainly cost precision, not accuracy") omitted that one random configuration per cell biases σ*.** The median σ* is 0.739 vs 0.790 on the frontier: −0.051, or −3.7 residual-bootstrap SE, while the median bias in a is −0.3 SE. This bears directly on the σ*-robustness narrative. | Memo §1.5(iii) and claim 8 revised; the median bias is reported in SE units in `_stability.csv`. | fixed |
| m1 | minor | The **"total" count** reused Porian's `params_all` = 12w²d + (S+2V)w. Its body term understates the precise SwiGLU body (`params_no_embed`) by up to 25% for small architectures (w = 128: −25%; w = 224: −16%). So "standard → total" mixed "add input embeddings" with "change the body formula". | `params_all_precise` = precise body + (S+2V)w is now used for the total count and the embedding shares. Effect ≤ 0.007: tuned 0.459 → 0.460 (RW) and 0.470 → 0.471 (OWT2); Kaplan-setup runs 0.668 → 0.661 and 0.671 → 0.670. | fixed |
| m2 | minor | The **residual bootstrap** of the 17-point Step Law frontier resampled raw NLS residuals (n = 17, p = 5) without centring or df rescaling, which understates SEs. | Residuals are now centred and scaled by √(n/(n−5)). SEs rose about 20%: a at E = 1.4: 0.089 → 0.108; σ*: 0.011 → 0.014; E-free a: 0.128 → 0.149; E: 0.41 → 0.50. The maximum policy shift is now 0.70 SE (a) and 0.74 SE (σ*), previously "≤ 0.9 SE". Conclusions are unchanged and stronger. | fixed |
| m3 | minor | **SFA SEs** came from the numerical Hessian, treating 1,730 runs as independent, although the inefficiency distribution is set by one designed grid per cell. | Added cluster-robust (by cell) sandwich SEs and a transparent cell-level regression (ln mean ι on ln N, ln D; n = 17). γ_D: half-normal −0.231 [0.032], exponential −0.260 [0.023], cell-level −0.266 (0.028). γ_N is insignificant in all variants. Conclusion unchanged. | fixed |
| m4 | minor | The **closed-form measurement-bias prediction** on Porian's ladder (`delta_pred`: 0.285 / 0.199 / 0.270 / 0.224, i.e. 63–108% over-prediction) was in the CSV but not the memo. Only the semi-synthetic 20–50% was reported. | Reported in memo §1.3 and claim 2, and in the bias-formula table note. | fixed |
| m5 | minor | **Proposition F share** was headlined as 83–114% (fitted-technology curvature), although the local-curvature version gives 47–62%. The memo said the f'' estimates differ by "about 3.5×"; the maximum is 4.4× (min f2_model/f2_local = 0.226). | Memo now says "about half to all (47–114%)". Table Panel B shows both shares; the factor is corrected to 4.4. | fixed |
| m6 | minor | **Numerical imprecisions in the memo**: "roughly 40% measurement" (RW 39%, OWT2 47%); flexible-input effect "0.20–0.23" (actual 0.194–0.231); local-curvature share "47–61%" (actual 47–62%); a-shift at E = 1.2/1.6 "+0.071" (0.070); λ-profile "1.0% higher at λ = 0" (RW only; OWT2 0.4%); smoothed LR "about 30% below" grid (geometric mean 34%). | Corrected. | fixed |
| m7 | minor | The **sl.py bug example** in the memo (a = 0.503, "SSR 1.197e-4") could not be reproduced from the documented start. Also, the "SSR" is the objective, which equals ½·SSR. | Replaced with a reproducible example (warm start at the free-E NLS estimate: 0.565 vs 0.5735, objective 1.238e-4 vs 1.1447e-4). | fixed |
| m8 | minor | **Smoothed optima**: the quadratic's minimum lies above the best observed run in 16/17 cells (by up to 0.29%), so the quadratic misfits the asymmetric optimum, and its within-cell SEs understate location uncertainty. **Grid coverage**: the N = 1.07B, D = 56.9B cell has 5 LRs and 47 runs; at batch 256 its best LR is the largest tried. | Disclosed in memo §1.5(v), claim 10 and the Step Law table note ("up to 12 LRs"). Batch N-elasticity reported for grid argmins too (−0.36, SE 0.18). | disclosed |
| m9 | minor | Claim 5 said "a penalty constant in log-loss units biases a down". The sign is set by −(ι_n − ι_d), not by constancy. | Reworded, with the explicit ι_n − ι_d = +0.0045. | fixed |
| m10 | minor | **Presentation.** The Porian table's p{}-column labels wrapped to three lines, detaching the CI rows from their estimates. At AER's 1-inch margins the Porian and Step Law tables overflowed the text width (the Step Law table by 44pt). The Kaplan-adjusted IO label read "Measurement only", although that run also has the long warmup. The demand and SFA tables used a hyphen for minus. Figure 1(b) did not say which comparison it shows. | Short one-line labels ("llccccccc") with the explanation moved into the note. The Kaplan-adjusted label is now "Mismeasured", and the note says it keeps the long warmup. Tighter `\tabcolsep` (new optional argument of `common.write_tex`). Proper minus signs throughout. Fig. 1(b) is annotated "RW: step 2 vs. step 5". All five tables compile with tectonic at 1-inch margins with no overfull boxes. | fixed |
| m11 | minor (efficiency) | `common.fit_efixed` iterated over the grid's E dimension even though E is fixed, so every start was optimized 3× (FAST_GRID) or 5× (DEFAULT_GRID). | Starts de-duplicated. Results are bit-identical (E-profiles, policy fits and Monte Carlo draws unchanged), and the run takes 18–24 min instead of 27 at similar load. | fixed |
| m12 | minor | The run.py docstring claimed "10–15 minutes". | Updated. | fixed |

### sl.py (not edited; reported)
- **Confirmed:** the `E_fixed` branch of `sl.fit_chinchilla` uses default L-BFGS-B tolerances. When |objective| < 1, the default ftol ≈ 2.2e-9 acts as an absolute tolerance, and optimization stops early. The module's workaround (`common.fit_efixed`) is correct. **Other modules that call `sl.fit_chinchilla(..., E_fixed=...)` should re-check their results.** Patching this centrally in sl.py is the lead author's decision.

## 2. Verification of memo numbers

I checked every number in memo §1 and §4 against the regenerated outputs.

**Checked and correct:**
- Porian path: all 10 exponents with CIs, and N*(5.88e23).
- Decomposition: shares and counting-convention table.
- Measurement: head-share map (θ = 0.358, ω = 48,210, s from 0.936 to 0.084), semi-synthetic biases (0.208 / 0.170 / 0.214 / 0.158; E = 1.8: 0.188 / 0.162 / 0.212 / 0.121), Pearce–Song simulation (0.792 / 0.757 / 0.752; a_m peak 0.981 / 0.990 at N_m ≈ 1.9M, s = 0.755; brute-force error < 2e-7).
- Chinchilla non-embedding refit: 0.514 (0.020) → 0.556 (0.023); paired difference +0.042 (0.006) [0.030, 0.050]; σ* 0.737 → 0.753; subsamples +0.070 / +0.082.
- Proposition F tables and per-budget mechanism (δ = 1.57 nats at 2.5e16; vanishes by 3.2e18). Simulation check: maximum error 0.0033.
- Step Law: Huber E 0.91 / a 0.689 / σ* 0.824; NLS E 1.43 / a 0.563 / σ* 0.786; E-profile SSR within 16%; policy shifts (−0.041 … +0.076); rule inefficiency 0.1–0.9%; Monte Carlo table.
- Inefficiency and SFA: OLS gradients (−0.0068, SE 0.0006; −0.0023, SE 0.0014); SFA γ's and log-likelihoods; implied IsoFLOP bias −0.012.
- Demand functions: all rows.
- Decomposition: 0.046 + 0.320 × 0.598 = 0.238 vs 0.210.

**Numbers that changed because of the fixes:** total-count exponents (m1), residual-bootstrap SEs and policy-shift SE ratios (m2). The memo has been updated.

**Citations:** all 17 keys in the memo exist (15 in `lit/references.bib`, plus `caudill1995frontier` and `milgrom1996lechatelier` in `lit/bib/extra_m8_measurement.bib`). Both extra entries have correct bibliographic details (JBES 13(1):105–111; AER 86(1):173–179).

## 3. Remaining concerns (not fixable within this module)

1. **E is not identified in either small-scale dataset.** Porian's tuned-run Huber fits put E at about the lowest observed loss (3.13 vs 3.10 for RW), and Step Law's E-profile is flat over [0.6, 1.7]. Everything that uses a fitted technology is a local approximation: the semi-synthetic measurement check, the model-based f'' in Proposition F, and every Step Law level of a, σ* and γ.
2. **The Proposition F share explained (47–114%) depends on the curvature estimate**, and the first-order formula operates outside its validity region at the smallest budgets (δ up to 1.6 nats; argmin shifts of about 1 log point). The *sign* and rough size are robust; the exact share is not.
3. **Sequential-decomposition order dependence.** The 40–47% / 53–61% split uses Porian's order. The near-additivity result (count effect 0.12–0.17 under both tuned and untuned runs) mitigates but does not remove this. Evidence covers 5M–901M-parameter models only.
4. **The Chinchilla non-embedding reanalysis uses the Pearce–Song map** (ω, θ from their fit), not Chinchilla's Table A9 architectures.
5. **The Step Law design has 17 cells, 5 N values and no IsoFLOPs.** Policy shifts are below 1 SE. The inefficiency distribution is designed, not behavioural. The embedding-tying and vocabulary assumptions in the count robustness are [A]/[L].
6. **The Bjorck residual (0.34–0.37) remains unexplained.** It cannot be tested within Step Law, whose warmup is fixed at 2,000 steps.
7. **Paper-level implication:** SYNTHESIS §1 claim 2 ("σ is more stable than the allocation exponent") must not cite this module as support. On the Step Law evidence, σ*'s apparent stability is a unit artifact. Other modules' evidence (cross-dataset σ clustering at 0.74–0.78) should be re-examined with the same normalization: relative to α+β, and in SE units.
8. **Porian CIs** come from their noise-and-interpolate bootstrap (seed noise s.d. 0.002 for RW). They reflect calibrated seed noise only, not misspecification of the IsoFLOP interpolation; they are narrow by construction.

## 4. Confidence in each headline claim (after review)

| Headline (builder) | Confidence | Note |
|---|---|---|
| Porian path reproduced within 0.003 from the raw runs | **High** | Bit-identical rerun; published values verified in the paper text. |
| Gap ≈ 40% measurement / 60% flexible inputs, roughly additive | **High (numbers) / medium (interpretation)** | Now stated as 39–47% / 53–61%; order-dependent; small models only. |
| Proposition M closed form; always upward; Pearce–Song 0.79 / 0.76 | **High** | Independently re-derived. On Porian's ladder it over-predicts by 20–108% (medium). |
| Chinchilla a rises by +0.042 (0.006) with non-embedding N | **High (estimate) / medium (interpretation)** | Relies on the Pearce–Song map. |
| Proposition F formula; explains flexible-input steps | **High (formula) / medium-low (share)** | The share is 47–114% depending on f''. |
| Step Law does not identify a | **High** | Plus the new N-convention caveat (M3). |
| Random configurations cost precision, not accuracy | **High for a; revised for σ*** | σ* is biased at k = 1 (M4). |
| Inefficiency is D-biased (SFA γ_D ≈ −0.23 to −0.26) | **High** | Robust to cluster-robust SEs and the cell-level check. |
| Demand functions replicate Step Law; batch has an N-elasticity | **High (LR) / medium (batch N term)** | Grid version of the batch N-elasticity: t ≈ −2. |
| Bjorck–Step Law disagreement is "mostly" conditional vs unconditional | **Low as stated → revised** | Conditioning explains 26–43% of the published gap; batch co-scaling explains 56–91% of Step Law's positive elasticity. |
| (Implicit) σ* more robust than a | **Not supported** | See M1. |
