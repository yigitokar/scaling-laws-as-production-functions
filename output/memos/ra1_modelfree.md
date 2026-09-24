# Memo — module ra1_modelfree: the technology from designed variation, without a functional form

Module owner: ra1_modelfree (Claude). Date: 2026-09-24. Entry point: `code/analysis/ra1_modelfree/run.py`.

**Reproduction.**
- One command, `.venv/bin/python code/analysis/ra1_modelfree/run.py`, regenerates every number, table and figure below from `data/raw/`.
- Seeds are fixed (`ra1_common.SEED = 20260924`, with one stream per stage key). The run uses the CPU only, with at most 5 worker processes.
- The final outputs come from a single end-to-end run of the reviewed code, including the new `review` stage (704 s on a shared machine). It was written to a scratch output root (`RA1_OUTPUT_ROOT`); all 48 table files are byte-identical to the installed ones. The independent review (`output/memos/ra1_modelfree_review.md`, 2026-09-24) is summarised in Section 5.5.
  - The reviewer first re-ran the builder's code from scratch (806 s): all 43 table files were byte-identical to the builder's installed outputs.
  - The review then added outputs without changing any pre-existing number, except the E-set columns of the κ-robustness table (finer E grid, H6).
  - Stage times (final run): isoflop 168 s, farseer 214 s, chinchilla 189 s, kappa 129 s, everything else under 10 s.
  - Logs in `data/processed/ra1_modelfree/`: `run_stdout.txt` (final run), `run_stdout_review_replication.txt` (the reviewer's replication of the builder's code), `run_stdout_builder_final.txt` (the builder's final run, 946 s).
  - Earlier development passes are kept in `run_stdout_pass1.txt` and `run_stdout_pass2.txt`.
- The module reuses m1's Chinchilla loader and budget reconstruction, m2's loaders and estimators (`fit_q`, the local quadratic, Farseer's Eq. 3) and m8's port of Porian et al.'s pipeline. All are imported unchanged, and `sl.py` is not edited.

**Referee comments addressed:** R1 c6, c7, c9; R2 Major 8, 9(a)(b); R3 M6; R4 M1, M3, M5, M6 (Section 6).

**Notation** (`paper/notes/model_spec.md`):
- n = ln N; C = 6ND; M = D/N; w = ε_N/ε_D; S ≡ 2(1/σ* − 1).
- σ* = 2/(2+α+β) in the Chinchilla family and 2/(2+a₁+b₁) in the κ family.

**"Model-free"** means no E, no outer exponent, no functional form and no loss units.

**Citations.** Methods cite `feng2011wild`, `dersimonian1986meta`, `cochran1954combination`, `higgins2002quantifying`, `hartung2001tests`, `sidik2002simple`, `mammen1993bootstrap`, `koenker1982tests` and, added in review, `hagemann2017cluster` (new; `lit/bib/extra_ra1_modelfree.bib`, all Crossref-verified). Existing keys: `webb2023reworking`, `devries2023go`, `czech2026problems`, `czech2026llama3isoflop`, `marin2026ladders`, `porian2024resolving`, `li2025predictableb`, `besiroglu2024chinchilla`, `hoffmann2022training`, `grattafiori2024llama`, `andrews2012estimation`.

## 1. Headline findings

**Summary.**
- **H1.** The identity is exact for any smooth technology, homothetic or not.
- **H2.** On every IsoFLOP design with annealed runs, the model-free σ* is 0.66–0.71. Where the same-run parametric estimates discriminate (Llama 3; Farseer), it sides with κ free against κ = 1. On Chinchilla's 137 profile runs and on Marin, κ = 1 and κ free are both within about 1–2 SE of it; on Marin κ = 1 is the closer. Within Chinchilla and Llama 3, σ* falls with compute.
- **H3.** The model-free σ* is homogeneous across those designs plus Farseer's local path: the random-effects (RE) mean is 0.695 [0.673, 0.717] with τ = 0 (0.679 [0.656, 0.703] with Farseer's path σ* at the narrower cross-validated bandwidth). The test has little power below a between-design s.d. of about 0.02. Across the seven sweep–corpus technologies, the κ-free σ*_κ is heterogeneous (Q = 82, τ = 0.068).
- **H4.** Inside Farseer's support, every parametric form understates the wedge at high M: w_param/w_local is 0.26–0.73 at M ≥ 1,024 at the primary bandwidth and 0.30–0.82 at the narrower cross-validated one. The sign is robust. The size of the gap for the κ-free and Eq. 3 forms is bandwidth-sensitive. ln w is convex in ln(M/M*), so the family's linearity restriction fails; that convexity is the source of the understatement.
- **H5.** For Chinchilla, the design is now described correctly, and κ = 1 is rejected under every inference scheme except the original nine clusters (p = 0.052). The Gaussian CES rejection does not survive calibration at 5%: restricted-bootstrap p = 0.17, sandwich Wald p = 0.06 (HC1) and 0.09 (HC3). It is borderline at 10%.
- **H6, H7.** The σ*_κ robustness results and the practitioner table follow.

### H1. The identity holds for any technology and needs no E, κ or functional form
**Statement:** 1/σ* − 1 = L_nn|_C / (2|dL*/d ln C|). That is, the curvature of the IsoFLOP profile in ln N at its minimum, over twice the slope of the loss–compute frontier. Section 2.1 gives the derivation.

It is verified numerically to below 10⁻⁸ (`ra1_modelfree_identity_check.csv`, `..._farseer_eq3.csv`) in three cases:
- **Chinchilla form** (Besiroglu et al.): 0.737028, matching the analytic 2/(2+α+β) at C = 10¹⁹, 10²¹ and 10²⁴.
- **κ family** (m2's κ-free Chinchilla fit): 0.700553, matching 2/(2+a₁+b₁).
- **Farseer's non-homothetic Eq. 3:** the identity equals the full two-input Hicks elasticity from all first and second derivatives. σ*(C) falls from 0.745 at 10¹⁸ to 0.725, 0.707, 0.690 and 0.672 at 10²².

Computing both terms on ln L instead of L gives the same number, which confirms monotone-transform invariance.

*Corollary (added in review).* At the compute-optimal point of any smooth technology, d ln w/d ln M|_C = 1/σ* − 1. This follows from differentiating w = f_n/f_d along the isocost where f_n = f_d. It is verified to 6 digits on all three technologies (column `dlnw_dlnM_at_path`). The slope of ln w at M* is therefore not a test of the Chinchilla family; only the linearity of ln w in ln(M/M*) is (H4).

### H2. Model-free σ* on every IsoFLOP design

| | σ* |
|---|---|
| Undigitized Marin ladders | ≈ 0.70 |
| Digitized Chinchilla and Llama 3 profiles (falling with compute) | 0.66–0.67 |
| Porian et al.'s unannealed profiles | ≈ 0.51 |

- **Primary specification:**
  - quadratic, h = 1;
  - path-centred windows;
  - log-cubic frontier (log-quadratic for Marin Comma, which has 5 valid budgets);
  - design-conditional wild bootstrap, B = 999.
- **Files:** `ra1_modelfree_isoflop_summary.csv`, `ra1_modelfree_isoflop_budgets.csv`; table `ra1_modelfree_sigma.tex`; figure `ra1_modelfree_sigma_by_design`.

| Design (runs) | Valid/total budgets | **Model-free σ*, RE (SE)** | τ across budgets; Q (bootstrap p) | Pooled, design-conditional [95%] | Drift per decade of C (SE; p) | κ = 1, same runs (SE) | κ free, same runs (SE) |
|---|---|---|---|---|---|---|---|
| Chinchilla (137) | 9/9 | **0.673** (0.027) | 0.056; 15.6 (0.025) | 0.684 [0.650, 0.720] | −0.072 (0.026; 0.009) | 0.692 (0.013) | 0.667 (0.023) |
| Llama 3 (133) | 8/10 | **0.660** (0.023) | 0.058; 50.7 (0.001) | 0.628 [0.610, 0.652] | −0.053 (0.010; 0.001) | 0.769 (0.006) | 0.693 (0.020) |
| Marin, Comma (85) | 5/7 | **0.700** (0.027) | 0; 0.9 (0.91) | 0.701 [0.657, 0.753] | −0.015 (0.056; 0.81) | 0.690 (0.015) | 0.678 (0.015) |
| Marin, DCLM (85) | 7/7 | **0.713** (0.027) | 0; 0.7 (1.00) | 0.712 [0.672, 0.772] | +0.019 (0.048; 0.69) | 0.673 (0.013) | 0.663 (0.013) |
| Marin, Nemotron-CC (88) | 7/8 | **0.705** (0.023) | 0; 1.1 (0.98) | 0.702 [0.667, 0.758] | −0.007 (0.034; 0.85) | 0.684 (0.014) | 0.677 (0.014) |
| Porian, RefinedWeb (121) | 12/12 | **0.518** (0.017) | 0.041; 27.1 (0.001) | 0.525 [0.506, 0.547] | +0.041 (0.009; 0.001) | 0.599 (0.032) | 0.345 (0.141) |
| Porian, OpenWebText2 (116) | 12/12 | **0.505** (0.014) | 0.018; 13.0 (0.24) | 0.502 [0.472, 0.530] | +0.034 (0.012; 0.001) | 0.609 (0.039) | 0.347 (0.160) |
| Farseer, local path (404) | 8 compute levels | **0.703** (0.013) | 0.033 | mean 0.707 [0.700, 0.718] | −0.015 (0.008; 0.048) | 0.772 (0.001) | 0.710 (0.001) |

Notes on the table:
- The parametric columns refit each form by Huber loss on the same runs in the same convention, with Feng–He–Hu (FHH) wild SEs (B = 399). For Farseer they are the 404-run fits with wild cluster SEs.
- Budget validity requires an interior minimum bracketed by at least two runs on each side. The invalid budgets are:
  - Llama 3 at 3×10²¹ and 10²²;
  - Marin Comma at 3×10¹⁹ (unbracketed) and 9×10¹⁹ (no interior minimum in the window);
  - Nemotron at 1.8×10¹⁸ (3 runs).

**Comparison with m1/m2's parametric σ* (Task 1).**
- **Chinchilla.**
  - On the same 137 profile runs the model-free 0.673 (0.027) does not discriminate between the parametric forms: κ free gives 0.667 (0.023) and κ = 1 gives 0.692 (0.013), both within 1 SE.
  - The model-free value lies 0.064 (2.3 SE) below m1/m2's κ = 1 value of 0.737 and close to m2's κ-free 0.701. Both of those come from the 240-run design, which adds 108 off-profile runs, so this is a comparison across samples, not a like-for-like test (review item M-C).
- **Llama 3.** The κ = 1 value 0.769 lies 4.8 SE above the model-free 0.660. The κ-free 0.693 is 1.5 SE away. This is the one IsoFLOP design where the same-run comparison discriminates.
- **Farseer.** The model-free path value 0.703 is close to κ free (0.710) and well below κ = 1 (0.772; 5 SE). The path value is bandwidth-sensitive, at 0.66–0.73 (H4). The first-derivative estimate 1/(1+b₁) is 0.701–0.711 across bandwidths, so the comparison holds.
- **Marin.** Both parametric forms lie 0.01–0.05 below the model-free values, and κ = 1 is the closer of the two on all three corpora (0.690/0.673/0.684 against κ-free 0.678/0.663/0.677), although κ = 1 is rejected against κ free by QLR 86–133.
- **Porian.** κ = 1 gives 0.60–0.61 and κ free gives 0.35 (SE 0.14–0.16). The model-free 0.51–0.52 lies between them.
- **Porian's 16 sizes** come from constant-learning-rate runs without cooldown. Their profiles are asymmetric and window-sensitive: 0.45–0.55 across h ∈ [0.8, 1.5], and 0.33 with a global quadratic. We treat them as IsoFLOP-like and keep them out of the headline range (Section 7).

**Finite-grid bias and coverage (Czech et al.'s concern).**
- Applied to each design's own fitted Chinchilla-form and κ-family technologies without noise, the estimator's bias is at most 0.006, except +0.047 to +0.050 under the strongly curved κ fit to Porian.
- In a Monte Carlo with R = 100 replications × B = 149 draws, the RE 95% interval covers at 0.92 (Chinchilla), 0.98 (Llama 3) and 0.93 (Marin DCLM), with |bias| ≤ 0.006.
- The pooled design-conditional interval under-covers (0.85 / 0.91 / 0.82) and is reported as secondary (Section 5.2).
- **Specification sensitivity** (Section 5.3):
  - Chinchilla ranges 0.60–0.71 across windows, orders, frontier smoothers and centring. The 0.60 comes from h = 0.6, which keeps only 7 budgets; excluding it, the range is 0.665–0.714.
  - Llama 3: 0.64–0.68.
  - Marin: 0.64–0.72.

### H3. Heterogeneity across designs and technologies (Task 3; `ra1_modelfree_heterogeneity.csv`, `ra1_modelfree_sigma.tex` Panel B)
Every set uses a DerSimonian–Laird random-effects mean with a modified Hartung–Knapp–Sidik–Jonkman (HKSJ) 95% interval (t_{k−1}), plus Cochran's Q, I² and a 95% prediction interval (PI).

| Set | k | RE mean [HKSJ 95%] | τ | I² | Q (df; p) | 95% PI |
|---|---|---|---|---|---|---|
| **Model-free**, IsoFLOP designs + Farseer path, excluding Porian | 6 | **0.695 [0.673, 0.717]** | 0.000 | 0.00 | 4.1 (5; 0.54) | [0.671, 0.719] |
| Same, one estimate per study (Hoffmann, Meta, Marin, Farseer) | 4 | 0.693 [0.660, 0.726] | 0.010 | 0.24 | 3.9 (3; 0.27) | [0.631, 0.755] |
| Model-free, all 8 including Porian | 8 | 0.646 [0.562, 0.730] | 0.098 | 0.96 | 188.9 (7; < 0.001) | [0.390, 0.902] |
| **σ*_κ, seven sweep–corpus technologies** (m2 Table 4 SEs) | 7 | **0.622 [0.549, 0.695]** | 0.068 | 0.93 | 82.1 (6; < 0.001) | [0.431, 0.813] |
| σ*_κ, seven, with design-conditional wild SEs | 7 | 0.620 [0.549, 0.692] | 0.068 | 0.94 | 104.4 (6; < 0.001) | [0.431, 0.809] |
| σ*_κ, five studies (one per study) | 5 | 0.625 [0.516, 0.735] | 0.069 | 0.95 | 82.0 (4; < 0.001) | [0.382, 0.869] |
| σ*_κ, leave out OLMo | 6 | 0.658 [0.582, 0.734] | 0.038 | 0.78 | 22.3 (5; < 0.001) | [0.537, 0.778] |
| σ* under κ = 1, seven technologies | 7 | 0.777 [0.745, 0.809] | 0.030 | 0.83 | 36.3 (6; < 0.001) | [0.693, 0.861] |
| σ* under κ = 1, leave out Chinchilla | 6 | 0.786 [0.764, 0.808] | 0.011 | 0.31 | 7.2 (5; 0.20) | [0.748, 0.824] |
| σ*_κ on the 7 IsoFLOP designs (same runs) | 7 | 0.672 [0.636, 0.707] | 0.019 | 0.47 | 11.4 (6; 0.077) | [0.616, 0.727] |

- **σ is technology-specific, not a common 0.7.**
  - With κ free, the seven sweep–corpus technologies are strongly heterogeneous.
  - Leaving out any one technology, or keeping one per study, never brings I² below 0.78.
  - Pairwise, Chinchilla and Farseer agree (difference −0.009, p = 0.51). Each differs from the OLMo ladder (z = 6.2 and 7.8) and from Muennighoff (z = 3.3 and 3.6) (`ra1_modelfree_heterogeneity_pairwise_*.csv`).
- **The model-free σ* is homogeneous.** On the designs that have IsoFLOP profiles from annealed or IsoFLOP-trained runs, it is homogeneous at 0.695 (τ = 0; 95% PI [0.671, 0.719]).
  - With within-design SEs of 0.013–0.027, the test has limited power against between-design s.d. below about 0.02. "Homogeneous" therefore means "within ±0.02–0.03 of 0.70".
  - Farseer's path input is bandwidth-sensitive (H4). With it taken at the extended-grid cross-validated bandwidth (0.664), the set is still homogeneous: RE 0.679 [0.656, 0.703], τ = 0.007, Q = 5.5 (p = 0.36). At h_N × 0.75 it is 0.686 [0.664, 0.708] (`ra1_modelfree_heterogeneity.csv`).
  - The designs span different compute ranges, and σ* drifts with compute in Chinchilla and Llama 3 (C5), so the design means average over different parts of the compute axis.
- **Where the heterogeneity sits.** It is concentrated in the small sweeps that have no IsoFLOP profiles: Gadre, OLMo and Muennighoff, with 30–35 runs each. There we cannot separate technology differences from the κ family's fit to small designs (H7).
- **Two interpretations are left open.**
  - Either recipes differ — OLMo's ladder, Muennighoff's multi-epoch runs and Gadre's over-trained design — or the κ family is misspecified for designs dominated by off-path runs.
  - Only IsoFLOP-profile designs identify σ* without a functional form.

### H4. Farseer: model-free local w against parametric extrapolation (Task 2; `ra1_modelfree_farseer_*.csv`, table `ra1_modelfree_farseer_w.tex`, figure `ra1_modelfree_farseer_w`)
Δ = ln(w_param/w_local), bin averages over grid points inside each model size's observed token range. Local ln w is in parentheses as w. The intervals are 95% basic intervals from a wild cluster bootstrap by model size (Webb, B = 999; every model is refitted in every draw). Non-embedding N (Farseer's convention):

| M bin | Local ln w (w) | Chinchilla, all runs | Chinchilla, fit on M ≤ 100 | κ free, all | κ free, M ≤ 100 | Farseer Eq. 3 |
|---|---|---|---|---|---|---|
| < 16 | −0.689 (0.50) | +0.158 [0.148, 0.166] | +0.079 | −0.021 | +0.023 | −0.027 |
| 16–64 | 0.059 (1.06) | +0.008 [0.000, 0.013] | −0.230 | +0.053 | −0.039 | +0.014 |
| 64–256 | 0.675 (1.96) | −0.195 [−0.207, −0.185] | −0.545 | +0.004 | −0.184 | −0.012 |
| 256–1,024 | 1.368 (3.93) | −0.487 [−0.501, −0.474] | −0.945 | −0.139 | −0.419 | −0.129 |
| **≥ 1,024** | **2.012 (7.48)** | **−0.821 [−0.896, −0.746]** | **−1.360 [−1.438, −1.276]** | **−0.355 [−0.431, −0.279]** | **−0.705 [−0.787, −0.643]** | **−0.314 [−0.383, −0.235]** |

- **Parametric forms understate the wedge at high M, and the gap grows with M.**
  - At M ≥ 1,024 the ratio w_param/w_local is 0.44 [0.41, 0.47] for Chinchilla (all runs), 0.26 [0.24, 0.28] for Chinchilla fitted on M ≤ 100, 0.70 [0.65, 0.76] for κ free and 0.73 [0.68, 0.79] for Eq. 3.
  - The κ = 1 fit on M ≤ 100 extrapolated to M ≈ 1,000–2,570 understates w by a factor of 3.9. This is the operation Section V performs for over-trained small models.
- **With embeddings counted, every gap widens.** At M ≥ 1,024 the local ln w is 2.74 (w = 15.5), and Δ = −1.17, −1.67, −0.70, −1.00 and −0.62 for the five comparators.
- **Robustness.**
  - The run-level Rademacher wild bootstrap gives the same SEs as the cluster bootstrap (±0.01).
  - The residual ICC is 0.27 by model size and 0.07 by D level.
  - At bandwidth ×1.5 and ×2 the M ≥ 1,024 gaps widen: Chinchilla −0.96 and −1.14; κ free −0.49 and −0.67; Eq. 3 −0.44 and −0.63.
  - **Narrower bandwidths shrink the gaps (review item M-A).** The primary h_N = 0.2 s.d. is the *smallest* multiple on the cross-validation grid, so it is a boundary solution. On an extended grid the leave-one-out optimum is h_N = 0.125 s.d. (0.148; 0.10 s.d. with embeddings), with h_D unchanged (`ra1_modelfree_farseer_bandwidth_cv_extended.csv`).
    - At that bandwidth the M ≥ 1,024 gaps are −0.70 (Chinchilla, all runs), −1.22 (Chinchilla, M ≤ 100), −0.25 (κ free) and −0.20 (Eq. 3): ratios 0.50, 0.30, 0.78 and 0.82.
    - At h_N × 0.75 they are −0.78, −1.32, −0.32 and −0.27.
    - The sign is robust at every bandwidth; the size of the κ-free and Eq. 3 gaps is not. The understatement is 18–30% between the CV-optimal and primary bandwidths, and up to 49% at twice the primary bandwidth. Because derivatives need more smoothing than the function-CV optimum, the primary bandwidth is kept, and the narrower values are the lower end of the range.
  - With n_eff ≥ 25, no non-embedding grid point reaches M ≥ 1,024. The 256–1,024 bin then gives −0.46 (Chinchilla) and −0.12 (κ free).
  - An independent finite-difference check agrees: within-size quadratics in ln D, and cross-size quadratics over three adjacent sizes at a common D, give ln w of 1.68–1.93 at M ≈ 1,270 for the 0.14–0.24B sizes. That is above every parametric comparator at the same points (κ free 1.59–1.60, Eq. 3 1.63–1.64, Chinchilla 1.14–1.15), by 0.09–0.33 for κ free (`ra1_modelfree_review_farseer_fd.csv`).
- **Linearity of ln w in ln(M/M*_local(C)) (R1 c7.3). Reframed in review (item M-B).**
  - *What is testable.* At a compute-optimal point of *any* smooth technology, d ln w/d ln M|_C = 1/σ* − 1. This is an identity, not a restriction of the Chinchilla family: differentiate w = f_n/f_d along the isocost at f_n = f_d and use H1. It is verified to 6 digits on Farseer's non-homothetic Eq. 3 (`ra1_modelfree_identity_check_farseer_eq3.csv`, column `dlnw_dlnM_at_path`). The family's testable content is that ln w is *linear* in u = ln(M/M*(C)).
  - *Linearity is rejected.* In ln w_local = b₁u + b₂u², b₂ = +0.020 (SE 0.001; bootstrap p at the floor of 0.001). b₂ is 0.018–0.026 at every bandwidth, and 0.033 with embeddings. The convexity adds about +0.37 to ln w at the largest u in the design (u ≈ 4.3, M ≈ 2,500), which is what a linear-in-u parametric form misses at high M (the H4 gaps).
  - *The earlier comparison.* The through-origin OLS slope (0.420, SE 0.003) against the Hessian-based path 1/σ* − 1 (0.434, SE 0.015; p = 0.36) compares two estimates of the same object. It is not a test of the family. It is also bandwidth-specific: at the cross-validated h_N the path value is 0.56 against a slope of 0.41. With embeddings the slope exceeds the path value (0.536 against 0.477; difference 0.059, p = 0.001).
- **Local path σ*(C) (R1 c6(b)).**
  - At 8 compute levels (5×10¹⁸ to 10²¹), σ*(C) on the local expansion path (w = 1) ranges 0.66–0.76, with M*_local = 25–45.
  - RE mean **0.703 (0.013)**, τ = 0.033; plain mean 0.707 [0.700, 0.718].
  - Drift −0.015 per decade (0.008; p = 0.048). Over the same range Eq. 3's σ*(C) goes from 0.730 to 0.690.
  - With embeddings: 0.674 (0.019).
  - **Bandwidth sensitivity (review item M-A).** σ on the path uses the local Hessian, and a local quadratic is not the natural order for second derivatives. The mean over the 8 levels is 0.707 at the primary bandwidth, 0.726–0.727 at ×1.5 and ×2, 0.683 at h_N × 0.75, and 0.664 at the extended-grid CV optimum (0.564 with embeddings). The bootstrap SEs above exclude this smoothing bias.
  - **First-derivative estimate (review addition).** By the identity above, σ*(C) = 1/(1 + d ln w/d ln M|_path). The slope b₁ at u = 0 from the quadratic regression gives **0.708** (wild cluster SE 0.001, interval [0.705, 0.710]; with embeddings 0.670). It is 0.701–0.711 across all bandwidth variants (0.665–0.674 with embeddings) (`ra1_modelfree_farseer_sigma_slope.csv`). This is the more reliable model-free path σ* on Farseer, and it is close to κ free (0.710; 0.668 with embeddings).
- **Parametric σ* on Farseer** (wild cluster SEs):

  | Convention | Chinchilla form | Chinchilla form, M ≤ 100 | κ free | κ free, M ≤ 100 |
  |---|---|---|---|---|
  | Non-embedding | 0.772 (0.001) | 0.820 (0.002) | 0.710 (0.001) | 0.745 (0.002) |
  | Including embeddings | 0.724 | 0.770 | 0.668 | 0.698 |

  - The design-conditional sampling error is tiny. The relevant uncertainty is the spread across forms, samples and conventions (0.67–0.82).

### H5. Chinchilla inference (Task 4; R4 M1, M5, M6; R1 c9; `ra1_modelfree_chinchilla_*.csv`, table `ra1_modelfree_chinchilla_inference.tex`, figure `ra1_modelfree_kappa_profile`)
**Design, corrected** (`ra1_modelfree_chinchilla_design.csv`):
- There are 245 digitized runs: **137 on the nine IsoFLOP profiles and 108 off-profile**.
- Training FLOP spans **1.40×10¹⁸–1.30×10²²**. The IsoFLOP runs span 5.3×10¹⁸–3.0×10²¹; the off-profile runs span the whole range.
- The baseline n = 240 drops the five highest-loss runs and has 132 profile and 108 off-profile runs.
- The off-profile runs form 35 model-size trunks, i.e. Approach-1 training-horizon streaks.

**Standard errors and intervals by scheme** (Huber on ln L, n = 240, B = 999 each, no failed draws):

| Scheme (clusters) | SE α | SE β | SE a | SE σ* | M*(5.76×10²³) 95% | w(70B) 95% | κ̂ = 0.774: SE [95%] | p(κ = 1), Wald | p(α = β), Wald | SE σ*_κ |
|---|---|---|---|---|---|---|---|---|---|---|
| Pairs (240) | 0.015 | 0.021 | 0.020 | 0.006 | [7.4, 35.4] | [0.82, 1.43] | 0.057 [0.67, 0.89] | < 0.001 | 0.49 | 0.014 |
| 9 budget clusters, original (9) | 0.026 | 0.050 | 0.049 | 0.011 | [3.7, 179] | [0.48, 1.89] | 0.116 [0.53, 0.97] | 0.052 | 0.77 | 0.018 |
| **Budgets + trunks (44)** | 0.018 | 0.042 | 0.040 | 0.009 | [5.2, 106] | [0.58, 1.65] | 0.079 [0.64, 0.94] | **0.004** | 0.72 | 0.018 |
| Budgets + singletons (117) | 0.018 | 0.041 | 0.039 | 0.009 | [5.9, 109] | [0.58, 1.56] | 0.076 [0.65, 0.94] | 0.003 | 0.71 | 0.018 |
| **Wild, Feng–He–Hu (240)** | 0.014 | 0.016 | 0.017 | 0.005 | [10.5, 37.4] | [0.80, 1.26] | 0.054 [0.67, 0.88] | < 0.001 | 0.40 | 0.012 |
| Wild cluster, Webb (44) | 0.020 | 0.029 | 0.021 | 0.011 | [9.2, 37.3] | [0.81, 1.32] | 0.062 [0.67, 0.90] | < 0.001 | 0.50 | 0.015 |

- **How Table 3 changes (R4 M1(b)).**
  - Clustering the off-profile runs by trunk instead of assigning them to the nearest budget lowers the β SE from 0.050 to 0.042.
  - It narrows the M*(5.76×10²³) interval from [3.7, 179] to [5.2, 106] and the w(70B) interval from [0.48, 1.89] to [0.58, 1.65].
  - It moves the κ = 1 p-value from 0.052 to 0.004. Singleton clusters give nearly the same answer.
  - Our nine-cluster run reproduces the paper's Table 3 up to Monte Carlo and starting-value differences: [3.7, 179] against [3.2, 126], [0.48, 1.89] against [0.54, 2.01], and p = 0.052 against 0.042.
- **Recommended procedure.**
  - We recommend the FHH wild bootstrap, which is design-conditional with LAD-appropriate weights (R1 c9(a)).
  - The budget + trunk cluster bootstrap is the conservative check.
  - Under every scheme κ = 1 is rejected at 1% except the original nine clusters (p = 0.052). CES is never rejected by a Huber-based Wald test.
  - The w(70B) interval always contains 1. It excludes Hoffmann et al.'s 0.62–0.71 only under the pairs scheme and the two wild schemes.
- **The wild population is the κ-free fit.**
  - A first pass drew from the κ = 1 fit, which centred κ* at 1 ([0.87, 1.14]) and excluded κ̂.
  - With the κ-free population the wild κ interval is [0.67, 0.88], consistent with the other schemes. The Chinchilla-form parameters are re-estimated in each draw, so their wild SEs are computed under the better-fitting surface.
  - *Review note (m-E).* Under that population the Chinchilla-form pseudo-true values differ from the estimates: w(70B) is 0.978 against 1.040, M*(5.76×10²³) is 21.3 against 17.9, and α − β is −0.011 against −0.020. The percentile intervals are therefore centred off the estimates. Basic intervals (in logs), which correct for this, are nearly identical for FHH: M* [10.2, 36.4] and w(70B) [0.81, 1.27], against the percentile [10.5, 37.4] and [0.80, 1.26]. For the Webb wild cluster scheme they are [10.2, 41.4] and [0.77, 1.25] (`ra1_modelfree_chinchilla_schemes.csv`, columns `pop_*`, `lo_basic_*`, `hi_basic_*`). No conclusion changes.

**Tests** (restricted wild FHH bootstrap under H₀, B = 999, p-floor 0.001; `ra1_modelfree_chinchilla_tests.csv`):

| H₀ | Laplace quasi-LR: stat; χ² p; boot p (crit.) | Gaussian LR: stat; χ² p; boot p (crit.) | Koenker–Bassett: stat; χ² p |
|---|---|---|---|
| κ = 1 | 31.2; < 0.001; 0.001 (8.6) | 46.2; < 0.001; 0.001 (13.4) | 20.7; < 0.001 |
| α = β (CES) | 1.84; 0.18; 0.40 (8.6) | **10.40; 0.0013; 0.171 (20.2)** | 1.20; 0.27 |

- **R4 M6(a).** The Gaussian LR rejects CES against χ², as the referee notes. But the residuals are heavy-tailed (MAD-based s.d. 0.0049 against s.d. 0.0076), and the restricted-bootstrap 95% critical value of the Gaussian LR is 20.2, not 3.84, giving a calibrated p-value of 0.17.
  - A second calibration that does not depend on the bootstrap population was added in review (m-D): heteroskedasticity-robust Wald tests of α = β for the same Gaussian fit give 3.54 (HC1, p = 0.060) and 2.90 (HC3, p = 0.089), against 10.64 under homoskedasticity (`ra1_modelfree_chinchilla_tests.csv`).
  - Both robust (Huber) tests agree: Laplace QLR p = 0.40 and Koenker–Bassett p = 0.27.
  - So CES is not rejected on Chinchilla at 5% by any heavy-tail-robust procedure, but the Gaussian evidence is borderline at 10%. "Artifact" overstates it; "not robust to heavy tails" is accurate.
  - κ = 1 is rejected by every statistic, at the bootstrap floor, and by the Gaussian sandwich Wald (HC1 40.7, HC3 34.5).

**Profile over σ* with κ free** (grid 45 points on [0.05, 0.99]; LR relative to the unconstrained optimum; `ra1_modelfree_chinchilla_profile_sigma.csv`, `..._profile_sets.csv`, `..._profile_calibration.csv`, `..._profile_set_calibrated.csv`):
- **Full design (n = 240).**
  - Huber gives σ̂ = 0.701. The Laplace-QLR χ² set is [0.69, 0.71] and the Koenker–Bassett set [0.69, 0.72]. Point and set now come from the same objective (R4 M5(a)).
  - Gaussian gives σ̂ = 0.680 with set [0.67, 0.69]. This is the set m1 reported: it came from the Gaussian profile.
  - No set touches the grid; the maximum LR is 817 (Laplace) and 718 (Gaussian).
- **Calibration** (restricted wild bootstrap of the profile QLR with σ* fixed, B = 299, floor 0.0033).
  - The 95% critical values are 11.4 at σ* = 0.69, 9.0 at 0.71 and 7.7 at 0.74, all above 3.84, so the χ² set is too narrow.
  - With the largest endpoint critical value the calibrated set is **[0.68, 0.72]**.
  - σ* = 0.74, the κ = 1 value, is rejected: LR 29.0, bootstrap p = 0.003 (the floor).
- **On-path band (n = 41; |ln N − ln N*(C)| ≤ 0.15; R4 M5(b), R1 c9(c)).**
  - Relative to the unconstrained optimum (σ̂ = 0.342, κ̂ = 0.17), the Laplace-QLR set is [0.20, 0.99] and **hits the upper grid boundary**. The Koenker–Bassett set is [0.15, 0.99] and also hits it. The Gaussian set is [0.15, 0.77] (σ̂ = 0.215).
  - The profile is flat: LR ≤ 1.21 on [0.25, 0.94], rising to 3.4 at 0.99. At σ* ≥ 0.98, κ reaches its upper bound (300). The largest LR, 24 (Laplace), occurs only at σ* ≤ 0.15.
  - Bootstrap critical values are 6.5 at σ* = 0.30 and 5.2 at 0.74, which widen the set further.
  - As Proposition 3 implies, on-path data carry no information on σ*. The profile is informative only with transverse variation.

### H6. σ*_κ robustness (Task 5; R2 Major 8; `ra1_modelfree_kappa_robustness.csv/.tex`, `ra1_modelfree_kappa_E_profile.csv`)
All SEs are design-conditional wild: FHH, or Webb clusters by size for Farseer and by cell for OLMo. B = 499 for baselines and 199 for variants.
- **Chinchilla (0.701, SE 0.012).**
  - Dropping N below the 10th percentile gives 0.690 (0.014); below the 25th, 0.648 (0.017).
  - Keeping the 5 outliers gives 0.664 (0.016).
  - With E fixed at the κ = 1 estimate (1.817), 0.703.
  - Over the 95% QLR set for E ([1.72, 1.79] on the refined grid), 0.697–0.703. Over the whole E grid, 0.641–0.711.
- **Farseer (0.710, SE 0.004).**
  - Dropping the 1, 2 or 5 smallest sizes gives 0.709–0.710.
  - **Counting both embeddings gives 0.668 (0.004).**
  - Dropping M > 1,000 gives 0.716; dropping M < 2 gives 0.694.
  - With E fixed at the Chinchilla-form value (0.369), 0.696. The E set is [0.258, 0.274] on the refined grid, over which σ*_κ is 0.709–0.711, so the E–κ trade-off does not bind.
- **Gadre (C4 0.607, RedPajama 0.623, RefinedWeb 0.594; SEs 0.045–0.052). Fragile.**
  - Dropping the smallest size gives 0.528, 0.574 and 0.545.
  - Dropping two sizes gives 0.493, 0.696 and 0.484, with only 18–19 runs.
  - Inside the E sets on the refined grid, the estimate moves by 0.014 (C4, 0.600–0.614) and 0.025 (RefinedWeb, 0.580–0.605). For RedPajama it moves by **0.079 (0.606–0.685)**. Over the full E grid it ranges from 0.52 to 0.91 (RefinedWeb).
  - *Review note (m-F).* The builder's 36-point grid resolved these sets with 2–7 points and reported at most 0.016. The refined grid adds 25 points around each set.
- **OLMo ladder (0.544, SE 0.019).**
  - Dropping sizes gives 0.548 and 0.518.
  - Task bits per byte as the output gives 0.512 (0.059).
  - The E set moves it by at most 0.004 (0.541–0.545, refined grid).
- **Muennighoff (0.511, SE 0.047).**
  - Dropping sizes gives 0.527 and 0.536; dropping D/N < 0.4 gives 0.566.
  - With E fixed at the Chinchilla-form value, 0.558. The E set (refined) gives 0.503–0.555.
- **Summary.**
  - The E–κ trade-off is immaterial inside the E confidence sets for Chinchilla, Farseer and the OLMo ladder (≤ 0.006). It is small for Gadre C4 and RefinedWeb (≤ 0.025). It is material for Gadre RedPajama (0.606–0.685) and Muennighoff (0.503–0.555). These are descriptive χ² sets; they are not bootstrap-calibrated.
  - On Farseer the convention matters (−0.042).
  - The smallest sizes matter on Chinchilla (−0.01 and −0.05) and Gadre (−0.05 to −0.11).
  - Only one variant moves a small sweep near 0.70: Gadre RedPajama without its two smallest sizes gives 0.696, from 19 runs with SE 0.097.

### H7. The compute cost of over-training (Task 6; R2 Major 8; `ra1_modelfree_practitioner.csv/.tex`, figure `ra1_modelfree_overhead`)
- C/C_min at M = k·M* in the homothetic case (α = β = ρ = 1/σ* − 1) is cosh(ρ ln k / 2)^{2/ρ}.
- Every entry is checked by direct minimization on the technology.
- Implied wedge: w = k^{1/σ* − 1}.

| σ* | k = 2 | k = 10 | k = 100 | k = 1,000 | w at k = 100 |
|---|---|---|---|---|---|
| 0.60 | 1.083 | 2.244 | 14.32 | 128.8 | 21.5 |
| 0.70 | 1.053 | 1.727 | 7.23 | 49.8 | 7.20 |
| 0.74 | 1.043 | 1.574 | 5.42 | 31.3 | 5.04 |
| 0.80 | 1.030 | 1.387 | 3.52 | 14.5 | 3.16 |

- **How much over-training costs.**
  - Training at 10× the compute-optimal ratio costs 39–124% extra compute across σ* ∈ [0.6, 0.8]. 100× costs 3.5–14.3× the minimum, and 1,000× costs 14–129×.
  - σ* matters more as k grows: at k = 100, σ* = 0.70 against 0.74 is 7.2× against 5.4×.
- **Non-homothetic splits at the same σ*** — Besiroglu's α/β = 0.951 and Hoffmann's 1.191 — change C/C_min by at most 4.5% and 19% for k ≤ 100 (k relative to M* at the model's own compute).
- **de Vries (2023; `devries2023go`).** His parameters are E 1.69, A 406.4, B 410.7, α 0.32, β 0.28, so σ* = 0.769.
  - A model at 30% of the compute-optimal size needs 6.63× the tokens and **99.0%** more compute (his "about 100%").
  - That is k = 22.1 relative to the optimal model's ratio, or 21.1 relative to M* at its own compute. Our formula gives exactly 99.0%; the homothetic formula at the same σ* gives 100.3%.
  - His whole curve, for 30–100% of the optimal size, lies between our σ* = 0.74 and 0.80 curves.

## 2. Methods

### 2.1 The identity (Task 1; R1 comment 7)
- For any smooth monotone loss index f(n, d), the two-input Hicks elasticity is 1/σ = 1 − (f_nn f_d² − 2 f_nd f_n f_d + f_dd f_n²)/(f_n f_d (f_n + f_d)) (m2's `sigma_from_derivs`).
- At a compute-optimal point f_n = f_d, because the isocost C = 6ND is the line n + d = const. The Hicks formula then reduces to 1/σ* − 1 = −(f_nn − 2f_nd + f_dd)/(2f_n).
- The numerator is the second derivative of the IsoFLOP profile in ln N, since dd = −dn along the isocost. By the envelope theorem, f_n = dL*/d ln C.
- Hence **1/σ* − 1 = L_nn|_C / (2|dL*/d ln C|)** for any smooth technology, homothetic or not.
- A monotone transform g of the loss multiplies both terms by g′(L*). The ratio therefore needs no E, no outer exponent, no functional form and no loss units, so the unknown units of the digitized Llama 3 losses do not matter.
- The identity holds on an *exact* isocost. Each design is therefore analysed in the parameter convention in which its profiles are exact 6ND isocosts.

### 2.2 Designs and conventions (Task 1)

| Design | Runs | Budgets | Exact-isocost N convention | Loss |
|---|---|---|---|---|
| Chinchilla (Epoch digitization) | 137 IsoFLOP-profile runs; the 5 high-loss runs are included and fall outside every window | 9, 6×10¹⁸–3×10²¹ | total N (Hoffmann); runs placed at their nominal budget | MassiveText, digitized |
| Meta Llama 3 (Czech digitization) | 133 | 10, 6×10¹⁸–10²² | N = C/(6D), as digitized | units unstated |
| Marin 2026-03: Comma / DCLM / Nemotron-CC | 85 / 85 / 88 | 7 / 7 / 8, 1.8×10¹⁸–3×10²⁰ | N_eff = C_b/(6D), FLOP-weighted; configuration N as robustness | Paloma macro average |
| Porian et al.: RefinedWeb / OpenWebText2 | 121 / 116 | 12 each, 1.25×10¹⁶–2.56×10¹⁹ | Porian's "standard" count (non-embedding body + head), C = 6ND | validation loss |
| Farseer | 404; factorial, not IsoFLOP | local path at up to 10 compute levels | non-embedding; incl. embeddings as robustness | English bits/char |

- **Marin.** Budgets are 3× forward FLOPs. C_b/(6 N_cfg D) ranges from 0.74 to 1.07 and rises with N; the within-budget elasticity is η = 0.087.
- **Porian et al.** The profiles come from their final configuration: tuned learning rate, batch size and β₂; short warmup; constant learning rate. At each FLOP budget the validation loss is read off each run's curve with their log-log interpolation (m8's port of `fetch_flop`), giving 16 sizes.
  - The losses are **unannealed**, so these profiles are IsoFLOP-like rather than IsoFLOP designs.
  - All budgets of a size come from one run. The bootstrap therefore adds run-level Gaussian shocks with Porian's seed-noise s.d.: 0.002 (RefinedWeb) and 0.01 (OpenWebText2).

### 2.3 Estimator (per budget, then per design)
1. **Window centre.** In each budget, fit a global quadratic in x = ln N and take its argmin (convex, interior) as a preliminary argmin. A Theil–Sen line of these argmins on ln C (a robust Approach-2 path) then gives the window centre for every budget.
   - Centring on a pooled path instead of each budget's own noisy minimum removes a "winner's curse" (Section 5).
2. **Local polynomial.** Fit L by a polynomial in x on the runs within |x − centre| ≤ h, with h = 1 as the primary window (N within a factor e of the centre).
   - The order is 2 unless a pooled F test of the cubic term across budgets rejects at 5%. It never does in the final specification.
   - κ_b = L″ at the interior minimum, and L*_b = the fitted minimum.
   - **Validity.** A budget is valid if the minimum is interior with at least two runs on each side and κ_b > 0. Validity is judged at the point estimate only.
3. **Frontier slope.** s_b = dL*/d ln C = L*_b · d ln L*/d ln C, from a cubic polynomial in ln C of ln L*_b (a quadratic with fewer than six valid budgets). The log is only a smoothing device, since the identity is invariant to it.
4. **Per-budget estimate.** S_b = κ_b/|s_b|, so σ*_b = 2/(2 + S_b).
5. **Design summaries.**
   - **Random effects (primary).** A DerSimonian–Laird mean of S_b across budgets, with within-budget variances from the bootstrap (IQR-based) and between-budget s.d. τ. It is mapped to σ* with a delta-method s.e.
   - **Pooled, design-conditional.** The ratio of weighted sums S̄ = Σω_bκ_b / Σω_b|s_b|, with ω_b = |s_b|/Var*(κ_b), and a basic bootstrap interval.
   - **Drift.** A WLS slope of S_b on log₁₀ C, reported in σ* units per decade (non-homotheticity).
   - **Homogeneity.** A test across budgets whose bootstrap null distribution accounts for the shared frontier.

### 2.4 Inference (design-conditional wild bootstrap)
- **Bootstrap population.** Within each budget, a Gaussian-kernel local quadratic of L in ln N (bandwidth 0.5) gives L̃. Its residuals are rescaled by the smoother's degrees-of-freedom factor sqrt(n/(n − 2tr H + tr H′H)).
- **Draws.** L*_i = L̃_i + v_i r̃_i + η_b ẽ^F_b, plus the Porian run shocks. Here v_i is a Rademacher weight, η_b a Webb six-point weight, and ẽ^F_b the leverage-adjusted residual of budget b's minimum around the frontier fit (budget-level shocks).
- **Re-estimation.** Every draw re-estimates the window centres, windows, polynomials and frontier on the (N, C) design held fixed.
- **Intervals.** Basic intervals are centred at the estimator applied to the noise-free population.
- **Number of draws.**
  - Primary: B = 999.
  - Sensitivity specifications: B = 199 each.
  - Parametric κ = 1 and κ-free fits on the same runs: B = 399 (Feng–He–Hu weights).
- **Monte Carlo check.**
  - Truth: each design's own κ-family fit, plus Gaussian noise with the robust residual s.d.
  - The full estimator and bootstrap (B = 149) are re-run in each of R = 100 replications, for Chinchilla, Llama 3 and Marin DCLM.
- **Bootstrap p-values.** With B = 999, the smallest attainable p is 0.001; reported values of 0.001 are at that floor.

### 2.5 Farseer: local wedge and local expansion path (Task 2)
- **Local surface.** A Gaussian product-kernel local quadratic of ln L in (ln N, ln D), m2's estimator. Local quadratic is the natural order for first derivatives (p − ν = 1).
  - Bandwidths come from leave-one-out cross-validation over multiples of the s.d. (0.2–1.0): non-embedding h_N = 0.237, h_D = 0.745; with embeddings 0.200, 0.993.
  - *Review:* h_N is at the smallest multiple on that grid. On an extended grid (0.08–1.0) the optimum is h_N = 0.125 s.d. = 0.148 (0.10 s.d. = 0.100 with embeddings), with h_D unchanged. Narrower-bandwidth results are reported as sensitivity; the primary bandwidth is kept because derivatives call for more smoothing than the function-CV optimum.
- **Local wedge.** w_local = f_n/f_d is a ratio of elasticities, so it is invariant to any monotone transform of L.
- **Evaluation grid.** Each of the 25 model sizes × 24 log-spaced M from 1 to 2,570.
  - Points are kept only inside each size's observed D range (±10%) and with a kernel effective sample size of at least 15; 25 is a sensitivity check.
  - The threshold of 15 is what reaches M ≥ 1,024, which only the 0.10–0.34B sizes attain.
- **Parametric comparators.** Each is refitted in the same convention and evaluated at the same (N, D):
  - the Chinchilla form (Huber), on all 404 runs and on the runs with M ≤ 100;
  - the κ family (Huber), all and M ≤ 100;
  - Farseer's Eq. 3 (NLS, m2's starts), all and M ≤ 100.
- **Comparison statistic.** Δ = ln w_param − ln w_local, averaged within M bins (< 16, 16–64, 64–256, 256–1,024, ≥ 1,024).
- **Local expansion path.** Along each isocost, the root of ln w_local = 0 is found by Brent's method inside the design. σ* there, from the local gradient and Hessian, is the model-free σ*(C).
  - Compute levels where the root is not found, either on the data surface or on the noise-free pilot surface, are dropped: 10¹⁸ and 2×10¹⁸, leaving 8 of 10 levels.
  - The path summary is a DerSimonian–Laird mean across levels with bootstrap variances, which treats the levels as independent. The bootstrap SE of the plain mean, which accounts for the overlapping kernels, is 0.005.
- **Linearity test.** OLS of ln w_local on u = ln(M/M*_local(C)) and u². M*_local(C) is interpolated along the local path. The testable restriction is b₂ = 0; the slope at u = 0 equals 1/σ* − 1 identically (review).
- **First-derivative path σ* (review).** σ*_slope = 1/(1 + b₁), from the same regression. Its interval is the bootstrap interval of b₁ mapped through the transformation.
- **Inference.** A wild cluster bootstrap by model size (25 clusters, Webb weights):
  - Residuals are the leverage-adjusted residuals of the pilot local surface. The pilot-residual ICC is 0.27 by size and 0.07 by D level.
  - Every parametric model is refitted (warm starts) and every local quantity recomputed in each draw. B = 999.
  - A run-level Rademacher wild bootstrap (B = 499) is the comparison.
  - Intervals are basic, centred at the estimators applied to the noise-free pilot surface.
  - The local estimator's smoothing bias is not in the intervals; bandwidths ×1.5 and ×2 are reported instead.

### 2.6 Chinchilla inference (Task 4)
- **Sample and reference fits.** n = 240 (Besiroglu). The reference Huber fit (δ = 10⁻³) reproduces m1: E 1.8172, A 477.8, B 2143.4, α 0.3473, β 0.3672, σ* 0.7368. The κ-free Huber fit gives κ̂ = 0.774 and σ*_κ = 0.701.
- **Six bootstrap schemes,** B = 999 each, all with the identical estimator. Warm starts are the own estimate, Besiroglu and Hoffmann; the κ fit starts from the own estimate and from κ = 1.
  1. pairs;
  2. m1's nine budget clusters (every run assigned to the nearest nominal budget);
  3. IsoFLOP runs by budget plus off-profile runs by model-size trunk (35 trunks, 44 clusters);
  4. budgets plus off-profile singletons (117 clusters);
  5. Feng–He–Hu wild: y* = ŷ + v|r̃|, v = ±1 with probability ½, r̃ = r + h_ii(½ − 1{r < 0})/f̂(0), as in quantreg's `boot.rq` "wild";
  6. wild cluster, with one Webb weight per budget/trunk cluster.
- **Wild population.** The wild population (ŷ, r) is the κ-free fit, which fits better; κ = 1 is rejected. The Chinchilla form is re-estimated on every draw.
- **Tests of κ = 1 and α = β.**
  - Wald with each scheme's SE.
  - Laplace quasi-LR, 2n ln(H_r/H_u), with H the mean Huber objective.
  - Gaussian LR, n ln(SSR_r/SSR_u).
  - Koenker–Bassett LR, 4f̂(0)(S₁^r − S₁^u), with f̂(0) a kernel estimate at zero of the unrestricted residual density.
  - Restricted wild (FHH) bootstrap under H₀: data generated by the restricted fit, B = 999, floor p = 0.001.
  - CES is tested within the κ = 1 family.
- **Profile over σ*.**
  - A 45-point grid on [0.05, 0.99]. The κ family is in normalized coordinates with analytic gradients (κ ∈ [10⁻³, 300]).
  - Forward and backward warm-started sweeps. The LR is relative to the unconstrained optimum: multi-start, then re-polished from the best profile point.
  - Objectives: Huber and Gaussian. Samples: the full design and the on-path band (|ln N − ln N*(C)| ≤ 0.15, n = 41).
- **Calibration.** A restricted wild (FHH) bootstrap of the profile Laplace QLR with σ* fixed, at the set endpoints and at σ* = 0.74 (full design), and at 0.30 and 0.74 (band). B = 299 each.

### 2.7 σ*_κ robustness and heterogeneity (Tasks 3 and 5)
- **Fits.** κ-free Huber fits for the seven sweep–corpus technologies (m2 loaders) and their variants.
- **SEs.** Design-conditional wild: FHH, or Webb clusters by size for Farseer and by (size, multiplier) cell for OLMo. B = 499 for baselines and 199 for variants.
- **E–κ trade-off.**
  - A profile over E on a 36-point grid up to 0.995 · min L, plus Ê, with the other parameters refitted. In review the grid was refined with 25 points between the first excluded neighbours of the χ² set.
  - Descriptively, the range of σ*_κ over the E values whose Laplace QLR is ≤ 3.84.
  - σ*_κ with E fixed at the Chinchilla-form Ê.
- **Heterogeneity statistics.**
  - Cochran's Q, DerSimonian–Laird τ, I², and the random-effects mean with the modified HKSJ t_{k−1} interval and the 95% prediction interval.
  - Leave-one-out and one-estimate-per-study variants (inverse-variance within study).
  - Inputs are point estimates with their bootstrap SEs treated as known.
- **Model-free sets.** These use the RE σ* per design and, for Farseer, the RE mean of σ*(C) along the local path.

### 2.8 Compute overhead (Task 6)
- **Chinchilla family.** C/C_min = ((α + βw)/(α + β))^{1/γ} · w^{−1/α}, with ln w = (1/σ* − 1) ln k and k = M/M* at the model's own compute.
- **Homothetic case (α = β = ρ).** This reduces to cosh(ρ ln k/2)^{2/ρ}, with ρ = 1/σ* − 1.
- **Check.** Every entry is verified by minimizing C subject to L(N, D) = L* on the technology (`practitioner.brute`).

## 3. Inventory

### Code
The code lives in `code/analysis/ra1_modelfree/`. `run.py` regenerates everything.
- `RA1_QUICK=1` runs a smoke test with tiny bootstraps.
- `RA1_OUTPUT_ROOT` redirects all outputs.
- `--stages` reruns selected stages from the caches.

| File | Role |
|---|---|
| `run.py` | Stages: porian → identity → isoflop → farseer → chinchilla → kappa → practitioner → meta → tables → figures. Caches `data/processed/ra1_modelfree/stage_*.pkl`; log `run_log.txt`, `run_stdout.txt` |
| `ra1_common.py` | Paths, seeds, weights (Rademacher, Webb six-point, Feng–He–Hu), spawn pool (≤ 5 processes), loaders, LaTeX writer. The loaders reuse m1's `common.load_chinchilla` budget reconstruction and m2's `m2_data`; off-profile trunks are defined here |
| `porian_points.py` | Porian et al.'s IsoFLOP-like points via m8's `porian.py`, unchanged (run in a separate process) |
| `isoflop.py` | Identity checks; model-free estimator (windows, frontier, pooling); design-conditional bootstrap; sensitivity; parametric comparison; noise-free bias; Monte Carlo |
| `conventions.py` | Marin FLOP-accounting elasticity; configuration-N profiles |
| `farseer.py` | Local surface, local w, parametric comparators, local path σ*(C), linearity test, wild cluster bootstrap, bandwidth sensitivity |
| `parametric.py` | Chinchilla, κ and Eq. 3 fits (m1/m2 machinery), wedges, Feng–He–Hu residual adjustment |
| `kprofile.py` | κ-family fits in normalized coordinates with analytic gradients; profiles over σ* and E |
| `chin_inf.py` | Chinchilla design facts, six bootstrap schemes, restricted wild tests, σ* profiles and their calibration |
| `kappa_rob.py` | σ*_κ robustness and wild SEs |
| `meta.py` | Heterogeneity: Cochran Q, DerSimonian–Laird, I², HKSJ |
| `practitioner.py` | Overhead table, brute-force check, de Vries check |
| `tables.py`, `figures.py` | Outputs (review: `sigma_slope`, updated notes) |
| `review_checks.py` | Review stage: independent re-implementation of the model-free estimator; finite-difference Farseer wedge |

### Paper-ready LaTeX tables (`output/tables/`; booktabs, `tabular*`, AEA.cls `tablenotes`)
| File | Exhibit |
|---|---|
| `ra1_modelfree_sigma.tex` | **Exhibit (ii).** Model-free σ* by design with random-effects SE, drift and τ; κ = 1 and κ-free σ* on the same runs; Panel B heterogeneity statistics for six sets of technologies |
| `ra1_modelfree_farseer_w.tex` | **Exhibit (iii).** Local ln w and ln(w_param/w_local) by M bin for five parametric comparators, in two parameter conventions |
| `ra1_modelfree_practitioner.tex` | **Exhibit (iv).** C/C_min and the implied w at k = 2, 10, 100, 1,000 for σ* = 0.60, 0.70, 0.74, 0.80 |
| `ra1_modelfree_chinchilla_inference.tex` | Chinchilla SEs under six bootstrap schemes; LR tests; σ* profile sets (R4 M1, M5, M6) |
| `ra1_modelfree_kappa_robustness.tex` | σ*_κ robustness table (R2 Major 8) |

### CSVs (`output/tables/ra1_modelfree_*.csv`)
| Group | Files |
|---|---|
| Identity | `identity_check`, `identity_check_farseer_eq3` |
| IsoFLOP | `isoflop_summary` (per design); `isoflop_budgets` (per budget); `isoflop_invalid`; `isoflop_order` (F tests); `isoflop_sens` (windows, orders, frontier smoothers, centring, Marin config-N); `isoflop_param` (same-run parametric fits); `isoflop_bias` (noise-free bias); `isoflop_mc` (Monte Carlo); `isoflop_marin_flop_accounting` |
| Farseer | `farseer_delta` (Δ by bin, spec, convention); `farseer_lnw`; `farseer_lin` (linearity); `farseer_path` (σ*(C) on the local path, with parametric comparators); `farseer_pool`; `farseer_sigma` (parametric σ* with wild cluster SEs); `farseer_grid` (every grid point); `farseer_sensitivity` (bandwidth incl. narrower variants, n_eff); `farseer_bandwidth_cv`, `farseer_bandwidth_cv_extended`, `farseer_bandwidth_icc`; `farseer_sigma_slope` (first-derivative path σ*) |
| Review checks | `review_independent_modelfree`, `review_independent_modelfree_budgets`, `review_farseer_fd` |
| Chinchilla | `chinchilla_design` (facts); `chinchilla_schemes` (all objects × schemes: SE, robust SE, percentile CI, p-values); `chinchilla_tests`; `chinchilla_profile_sigma` (full profiles); `chinchilla_profile_sets`; `chinchilla_profile_calibration`; `chinchilla_profile_set_calibrated` |
| κ robustness | `kappa_robustness`, `kappa_E_profile` |
| Practitioner | `practitioner`, `practitioner_devries`, `practitioner_devries_curve` |
| Heterogeneity | `heterogeneity` (all sets × variants: all, leave-one-out, one per study); `heterogeneity_inputs_sweeps`, `heterogeneity_inputs_modelfree`; `heterogeneity_pairwise_tableSE`, `heterogeneity_pairwise_wildSE` |

### Figures (`output/figures/`, .pdf and .png)
| File | Exhibit |
|---|---|
| `ra1_modelfree_sigma_by_design` | **Exhibit (i).** (a) σ* by design: model-free (random effects, 95%), κ free and κ = 1 on the same runs. (b) σ*_b by compute budget, in small multiples, with the design mean and the parametric lines; Farseer's local path with Eq. 3 |
| `ra1_modelfree_farseer_w` | **Exhibit (iii).** (a, b) ln(w_param/w_local) by M bin (Chinchilla all runs, Chinchilla M ≤ 100, κ free) in the two conventions. (c) ln w at grid points: local, Chinchilla and Eq. 3 |
| `ra1_modelfree_kappa_profile` | Profile LR over σ* (κ free) relative to the unconstrained optimum: full design vs on-path band; Laplace QLR, Koenker–Bassett and Gaussian |
| `ra1_modelfree_overhead` | C/C_min against k for σ* = 0.6, 0.7, 0.74, 0.8, with de Vries's (2023) curve |

### Other outputs
- **Data.** `data/processed/ra1_modelfree/`:
  - `porian_isoflop_points.csv` (237 points);
  - the stage caches `stage_*.pkl`;
  - the run log `run_log.txt` and the stdout of the final end-to-end run, `run_stdout.txt`;
  - the development passes, `run_stdout_pass1.txt` and `run_stdout_pass2.txt`.
- **Bibliography.** `lit/bib/extra_ra1_modelfree.bib`: feng2011wild, dersimonian1986meta, cochran1954combination, higgins2002quantifying, hartung2001tests, sidik2002simple, mammen1993bootstrap, koenker1982tests. All are Crossref-verified.
- **Downloads.** `code/data/download_ra1_modelfree.sh`: no new data; it checks the existing raw files and records the verification sources.

## 4. Claims for the paper (evidence; caveat)

**C1. Identity (a new proposition for Section II).** At every compute-optimal point, 1/σ* − 1 = L_nn|_C / (2|dL*/d ln C|). This holds for any smooth technology, homothetic or not, and needs no E, no outer exponent, no functional form and no loss units.
- *Evidence:* the algebra in Section 2.1, plus numerical checks to 10⁻⁸ on the Chinchilla form, the κ family and Farseer's non-homothetic Eq. 3 (H1).
- *Caveat:* it requires an exact isocost (C = 6ND in the parameter convention used) and a smooth, locally quadratic profile. Measured quantities inherit the design's FLOP-accounting convention (H2, Section 5).

**C2. Where there are IsoFLOP profiles, the model-free σ* is 0.66–0.71, and it is homogeneous across the five designs with annealed or IsoFLOP-trained runs plus Farseer's local path.**
- *Evidence:* Chinchilla 0.673 (0.027), Llama 3 0.660 (0.023), Marin 0.700/0.713/0.705 (0.023–0.027) and Farseer's local path 0.703 (0.013).
- *Random-effects mean:* **0.695**, HKSJ 95% CI [0.673, 0.717], τ = 0, Q = 4.1 (p = 0.54). With one estimate per study: 0.693 [0.660, 0.726].
- *Caveat:* two of these designs are digitized (Chinchilla, Llama 3). Llama 3 and Marin are measured in a FLOP-implied N convention; Marin's configuration-N convention gives 0.64–0.66. Porian et al.'s unannealed profiles give 0.51–0.52 and are excluded from the "0.70" statement; with them the mean falls to 0.646 and heterogeneity is large (Q = 189).
- *Caveat (review):* Farseer's Hessian-based path σ* is bandwidth-sensitive (0.66–0.73). With it at the cross-validated bandwidth the RE mean is 0.679 [0.656, 0.703], still homogeneous (Q = 5.5, p = 0.36). The first-derivative estimate 1/(1+b₁) is 0.701–0.711 at every bandwidth. The homogeneity test has little power below a between-design s.d. of about 0.02, and the designs cover different compute ranges while σ* drifts with compute (C5).

**C3 (revised in review, M-C). Where the same-run comparison discriminates, the model-free σ* sides with κ free against κ = 1: on Llama 3 and on Farseer. Elsewhere it does not discriminate.**
- *Discriminating evidence:*
  - Llama 3: κ = 1 on the same runs gives 0.769, against a model-free 0.660 (0.023), a gap of 4.8 SE. κ free gives 0.693 (1.5 SE).
  - Farseer: κ = 1 gives 0.772, against a path σ* of 0.703 (0.013) and a first-derivative estimate of 0.708. κ free gives 0.710.
- *Non-discriminating evidence:*
  - Chinchilla's 137 profile runs: κ = 1 gives 0.692 (0.013) and κ free 0.667 (0.023), against a model-free 0.673 (0.027). Both are within 1 SE.
  - The often-quoted contrast with the 240-run κ = 1 value of 0.737 (2.3 SE) compares different samples.
  - Marin: κ = 1 (0.673–0.690) is *closer* to the model-free values (0.700–0.713) than κ free (0.663–0.678).
  - Porian: the model-free 0.51–0.52 lies between κ = 1 (0.60–0.61) and a poorly determined κ free (0.35, SE 0.14–0.16).
- *Caveat:* do not write "the model-free estimates reject κ = 1 wherever the two differ". κ = 1 is rejected against κ free by the Laplace quasi-LR on the other IsoFLOP designs (72–133). On Chinchilla's 137 profile runs the QLR is 7.8, which is borderline: the restricted-bootstrap 95% critical value on the 240-run design is 8.6. Rejection is a statement about fit, not about σ*.

**C4. σ is technology-specific across the seven sweep–corpus technologies; "≈ 0.7" is not a common value.**
- *Evidence:* the κ-free σ*_κ has Q = 82.1 on 6 df (p < 0.001), I² = 0.93 and τ = 0.068. The RE mean is 0.622 [0.549, 0.695], with a 95% prediction interval of [0.43, 0.81].
- It stays heterogeneous with OLMo left out (Q = 22.3, p < 0.001) and with one estimate per study (0.625 [0.516, 0.735]).
- The recommended wording is "0.51–0.71 with κ free: about 0.70 on Chinchilla, Farseer and every IsoFLOP design with annealed runs; 0.51–0.62 on the small sweeps (Gadre, OLMo ladder, Muennighoff)".
- *Caveat:* the small sweeps have 30–35 runs and no IsoFLOP profiles. On Gadre, σ*_κ moves by 0.05–0.11 when the smallest one or two sizes are dropped. The heterogeneity may therefore partly reflect the κ family's fit to small designs rather than technology.

**C5. Non-homotheticity: σ* falls with compute in Chinchilla and Llama 3.**
- *Evidence:* the drift is −0.072 per decade (SE 0.026; p = 0.009) in Chinchilla and −0.053 (0.010; p = 0.001) in Llama 3.
- Farseer's local path drifts −0.015 per decade (0.008; p = 0.048), and Farseer's Eq. 3 implies σ*(C) falling from 0.745 at 10¹⁸ to 0.672 at 10²².
- Marin shows no drift over its 2 decades (p ≥ 0.69).
- Neither the Chinchilla form nor the κ family allows any drift: σ* is constant along their paths. Under those constant-σ* truths fitted to Chinchilla and Llama 3, the estimator's noise-free per-budget bias lies within [−0.005, +0.002], so the drift is not a finite-grid artifact.
- *Caveat:* the Chinchilla drift ranges from −0.034 to −0.096 per decade across windows, and its p-value from 0.005 to 0.385. It is significant at 10% in 10 of 12 specifications; the exceptions are h = 0.6, which keeps only 7 budgets, and a cubic with h = 1, where p = 0.10. Porian's profiles drift the other way (+0.03 to +0.04).

**C6. Parametric forms understate the high-M wedge inside Farseer's support.**
- *Evidence:* at M ≥ 1,024 (non-embedding N) the local wedge is w = 7.48 (ln w = 2.01). The parametric wedges are 0.44× (Chinchilla, all runs; Δ = −0.82 [−0.90, −0.75]), 0.26× (Chinchilla fitted on M ≤ 100; −1.36), 0.70× (κ free; −0.36 [−0.43, −0.28]) and 0.73× (Eq. 3; −0.31 [−0.38, −0.24]).
- The gap grows monotonically with M and is larger with embeddings counted.
- For Section V: extrapolating a κ = 1 fit from M ≤ 100 to M ≈ 1,000–2,500 understates w by a factor of about 4 on Farseer.
- *Caveat:* the local wedge at M ≥ 1,024 comes only from the 0.10–0.34B sizes at the corner of the design (effective kernel n 15–25). Its smoothing bias is not in the intervals.
  - Wider bandwidths make the gap larger.
  - Narrower ones make it smaller. The primary h_N sits at the edge of its cross-validation grid, and at the extended-grid CV optimum the gaps are −0.70 (Chinchilla), −1.22 (Chinchilla, M ≤ 100), −0.25 (κ free) and −0.20 (Eq. 3).
  - The sign is robust (a finite-difference check agrees). The magnitude for the flexible forms, an understatement of about 18–30% between the CV-optimal and primary bandwidths, should be quoted as a range.

**C7 (revised in review, M-B). The family's linearity of ln w in ln(M/M*(C)) is rejected. ln w is convex, and the convexity is what the parametric extrapolations miss.**
- *Theory:* at the path, d ln w/d ln M|_C = 1/σ* − 1 for any smooth technology. This is an identity, verified on Farseer's non-homothetic Eq. 3. The slope *at* the path therefore carries no test of the family; only linearity away from the path does.
- *Evidence (non-embedding):* ln w_local = b₁u + b₂u² with b₁ = 0.413 and b₂ = +0.020 (SE 0.001; bootstrap p at the floor 0.001). b₂ is 0.018–0.026 at every bandwidth, and 0.033 with embeddings. The convexity adds about +0.37 to ln w at the largest ratio in the design (u ≈ 4.3).
- *By-product:* 1/(1 + b₁) = 0.708 is a bandwidth-robust, first-derivative estimate of Farseer's path σ* (0.701–0.711 across bandwidths).
- *Do not claim* "linearity is not rejected (p = 0.36)". That comparison (through-origin slope 0.420 against the Hessian-based path value 0.434) compares two estimates of the same object, and it changes sign across bandwidths.

**C8. Chinchilla inference: correct the design description; use budget + trunk clusters or wild inference. κ = 1 is rejected under every scheme except the original nine clusters (p = 0.052). The Gaussian CES rejection is not robust to heavy tails.** Calibrated p = 0.17 (restricted bootstrap) and 0.06–0.09 (sandwich Wald). Both Huber-based tests give p ≥ 0.27. Do not call it an "artifact". See H5.
- *Caveat:* digitization and the unobserved FLOP accounting (Section 7).

**C9. σ*_κ robustness.**
- σ*_κ is stable on Farseer (0.694–0.716) except for the embedding convention (0.668).
- On Chinchilla it moves with the smallest sizes (0.648–0.701) and with the outliers (0.664).
- It is fragile on Gadre (0.48–0.70) and stable on OLMo (0.51–0.55) and Muennighoff (0.51–0.57).
- The E–κ trade-off is immaterial inside the E confidence sets for Chinchilla, Farseer and OLMo. It is material for Gadre RedPajama (0.606–0.685) and Muennighoff (0.503–0.555) (refined E grid; review item m-F). See H6.

**C10. Practitioner translation.** Training at 10× the compute-optimal ratio costs 39% (σ* = 0.8) to 124% (σ* = 0.6) extra compute; at 100× it costs 3.5–14.3× the minimum. Between σ* = 0.70 and 0.74 the overhead at k = 100 differs by 7.2× against 5.4×. This is consistent with de Vries's (2023) curve. See H7.
- *Caveat:* the table assumes the Chinchilla family's power structure and a homothetic split. Non-homothetic splits change it by ≤ 19% at k ≤ 100.

## 5. Robustness and failures (including superseded choices)

### 5.1 Development failures, all documented and superseded
1. **Frontier smoother.**
   - *First choice:* a quadratic in ln C of L* in levels. Under each design's own fitted technology (noise-free) it mis-estimated the end-budget slopes by up to 31% (Chinchilla, 3×10²¹), 18% (Llama 3, 10²²) and 138% (Porian's largest budget, with the wrong sign). That biased σ*_b by up to −0.27.
   - *Fix:* a cubic in ln C of ln L* (E-free) has noise-free per-budget errors within ±0.016 on every design.
   - *Sensitivity only:* the power law E + K C^−γ is exact under the Chinchilla truth but is not E-free.
2. **Window centring (winner's curse).**
   - *First choice:* a window iterated around each budget's own noisy minimum and held fixed in the bootstrap. In development Monte Carlo (R = 10–60) this gave a downward bias of 0.02–0.03 and 20–70% coverage.
   - *Fix:* path-centred windows, re-estimation in every draw, a ratio-of-sums pooled estimator and a random-effects summary. The iterated-window estimator is kept as a sensitivity row; it gives 0.701 (Chinchilla) against 0.673 primary.
3. **Budget validity is a selection step.** With Chinchilla-level noise, high-budget curvatures have relative s.d. 0.4–0.7. Validity is therefore judged once, at the point estimate, and non-convex bootstrap draws are kept. At least one budget is non-convex in 17.5% of draws for Marin Comma (mostly the 1.8×10²⁰ budget, at 15%) and in 5.3% for Marin DCLM; elsewhere the share is ≤ 1%.
4. **Bootstrap scale.** Without the smoother's degrees-of-freedom correction, the bootstrap SE of the pooled estimator was below the Monte Carlo s.d. in development runs. The correction closes part of the gap (Section 5.2).
5. **Chinchilla wild bootstrap population.**
   - *Pass 1* drew wild samples around the κ = 1 fit. The κ interval was then centred near 1 ([0.87, 1.14]) and excluded κ̂ = 0.774.
   - *Final version:* the population is the κ-free fit, and restricted tests use the restricted fits. Both passes are logged (`run_stdout_pass1.txt`, `run_stdout_pass2.txt`).
6. **Farseer.** With a minimum effective kernel size of 25, no non-embedding grid point reached M ≥ 1,024. The primary threshold is 15; 25 is reported as sensitivity.
7. **Practitioner brute-force check.** It initially mismatched the formula in non-homothetic cases because k was defined relative to M* at the reference compute. It is now defined relative to M* at the model's own compute, as in the paper's lemma.
8. **Koenker–Bassett CES statistic.** The pass-2 edit that switched the wild population briefly computed the CES Koenker–Bassett statistic against the κ-free fit (22.0), which is the wrong unrestricted model. It is now computed against the κ = 1 fit (1.20, p = 0.27), as in pass 1.
9. **Determinism check.** The final end-to-end run wrote to a scratch output root (`RA1_OUTPUT_ROOT`). Apart from the corrected statistic and one table note, its 38 CSVs are byte-identical to the in-place passes. Its outputs were installed unchanged.

### 5.2 Monte Carlo of the model-free estimator (final specification)
- *Truth:* each design's own κ-family fit, plus Gaussian noise at the design's robust residual s.d.
- *Replications:* R = 100, each with a full bootstrap of B = 149. The Monte Carlo s.e. of a coverage near 0.95 is about 0.022.

| Design (true σ*) | Bias, RE | Bias, pooled | Coverage, RE 95% | Coverage, pooled basic 95% |
|---|---|---|---|---|
| Chinchilla (0.667) | −0.005 | −0.004 | 0.92 | 0.85 |
| Llama 3 (0.693) | −0.001 | −0.000 | 0.98 | 0.91 |
| Marin DCLM (0.663) | +0.005 | +0.001 | 0.93 | 0.82 |

- The random-effects interval is close to nominal, and the pooled design-conditional interval under-covers by 5–13 points. This is why the random-effects summary is primary.
- Noise-free finite-grid bias under each design's own Chinchilla and κ fits is ≤ 0.006, except under the strongly curved κ fit to Porian's profiles (+0.047 to +0.050) (`ra1_modelfree_isoflop_bias.csv`).

### 5.3 Specification sensitivity of the model-free σ* (RE; `ra1_modelfree_isoflop_sens.csv`, B = 199 each)

| Design | Primary | h = 0.6 / 0.8 / 1.25 / 1.5 / ∞ | Cubic, h = 1 / 1.5 | Frontier: log-quad / cubic / power | Iterated window |
|---|---|---|---|---|---|
| Chinchilla | 0.672 | 0.603 (7 budgets) / 0.669 / 0.714 / 0.691 / 0.671 | 0.665 / 0.696 | 0.675 / 0.679 / 0.678 | 0.701 |
| Llama 3 | 0.660 | 0.655 / 0.677 / 0.657 / 0.654 / 0.642 | 0.655 / 0.653 | 0.662 / 0.660 / 0.663 | 0.666 |
| Marin, Comma | 0.700 | n.a. / 0.685 / 0.721 / 0.723 / 0.699 | 0.668 / 0.685 | 0.706 / 0.702 / 0.704 | 0.707 |
| Marin, DCLM | 0.712 | 0.637 / 0.703 / 0.691 / 0.696 / 0.679 | 0.688 / 0.686 | 0.710 / 0.713 / 0.711 | 0.682 |
| Marin, Nemotron-CC | 0.704 | n.a. / 0.691 / 0.696 / 0.703 / 0.689 | 0.662 / 0.690 | 0.704 / 0.705 / 0.703 | 0.692 |
| Porian, RefinedWeb | 0.519 | n.a. / 0.554 / 0.513 / 0.474 / 0.332 | 0.511 / 0.459 | 0.518 / 0.519 / 0.518 | 0.492 |
| Porian, OpenWebText2 | 0.505 | n.a. / 0.512 / 0.524 / 0.451 / 0.341 | 0.488 / 0.435 | 0.501 / 0.503 / 0.498 | 0.506 |

- **Marin in configuration N** (N from the model config instead of C_b/(6D)): 0.658 / 0.642 / 0.655.
  - Marin's nominal budgets are 0.74–1.07 × 6 N_cfg D, with a within-budget FLOP-accounting elasticity of η = 0.087.
  - The curvature rescaling S_cfg = S_eff(1+η)² predicts config-N values of 0.664, 0.677 and 0.669, against the observed 0.658, 0.642 and 0.655. The remainder arises because configuration-N profiles are not exact isocosts.
  - Either way, the convention moves σ* by 0.04–0.07 on Marin. This is the size of error that Chinchilla's unobserved FLOP accounting could induce (R2 Major 9(b)).
- **Budget-level shocks.** The Webb shocks on each budget's frontier residual roughly double the pooled SE on Llama 3 (0.006 → 0.011) and barely matter elsewhere (`se_sigma_fe_within_only`).
- **Chinchilla's 5 high-loss runs** lie outside every window, so they do not affect the model-free estimate.
- **Pooled F tests of the cubic term** never reject at 5%: p = 0.83, 0.74, 0.79, 0.83, 0.85, 0.12 and 0.055 (Porian OWT2 is borderline). Every design uses order 2.
- **Porian's profiles are window-sensitive** (0.33–0.55), because the constant-LR curves are asymmetric about their minimum. With h = ∞ (a global quadratic) the estimate collapses to 0.33–0.34.

### 5.4 Nulls and fragile results reported as such
- **Linearity of ln w (Farseer).** Linearity is rejected: the quadratic term is significant in both conventions (b₂ = 0.020 and 0.033, bootstrap p at the floor) at every bandwidth. The through-origin slope against the path value (p = 0.36) is not a test of the family (review item M-B).
- **CES (α = β) on Chinchilla.** Not rejected at 5% by any robust procedure: Huber-based QLR p = 0.40, Koenker–Bassett p = 0.27, restricted-bootstrap-calibrated Gaussian LR p = 0.17, and Gaussian sandwich Wald p = 0.060 (HC1) and 0.089 (HC3). It is rejected by the Gaussian LR against χ² (p = 0.0013), which is miscalibrated under heavy tails. It is borderline at 10%.
- **On-path band.** The profile over σ* is flat (LR ≤ 1.21 on [0.25, 0.94]). The confidence set reaches the upper grid boundary (0.99), and σ̂ sits at the lower end of the flat region. There is no curvature information on the path, as Proposition 3 predicts.
- **Gadre.** σ*_κ is fragile: 0.48–0.70 across small-size exclusions.
- **Farseer, bandwidth.** At bandwidth ×1.5 or ×2 the M ≥ 1,024 gap grows (Chinchilla −0.96 and −1.14; κ free −0.49 and −0.67). At narrower bandwidths it shrinks: at the extended-grid CV optimum, Chinchilla −0.70, κ free −0.25 and Eq. 3 −0.20. The builder's statement that "the primary CV bandwidth gives the smallest gap" held only because narrower bandwidths were not tried. The Hessian-based local-path σ* moves from 0.664 to 0.727 over the same range.

### 5.5 Independent review (2026-09-24): what changed
Full report: `output/memos/ra1_modelfree_review.md`.
- **Replication.** The builder's code was re-run from scratch into a scratch root (806 s). All 43 table files were byte-identical to the installed outputs, and the figures matched visually.
- **Code changes.** No pre-existing estimate changed except the E-set columns of the κ-robustness table. The changes:
  1. `farseer.py`: extended cross-validation grid (`..._farseer_bandwidth_cv_extended.csv`), plus narrower-bandwidth sensitivity variants (h_N × 0.75 and the extended-grid CV optimum) in `..._farseer_sensitivity.csv`, including path-σ* summaries.
  2. `tables.py`: the first-derivative path σ* 1/(1+b₁) (`..._farseer_sigma_slope.csv`).
  3. `isoflop.py`: a numerical check that d ln w/d ln M|_path = 1/σ* − 1 for every technology (columns `dlnw_dlnM_at_path` in the identity CSVs).
  4. `chin_inf.py`: Gaussian sandwich Wald tests (`..._chinchilla_tests.csv`), plus the wild-bootstrap population's pseudo-true values and basic intervals (`..._chinchilla_schemes.csv`).
  5. `kappa_rob.py`: an E-profile grid refined around the set (`..._kappa_E_profile.csv`, column `fine`).
  6. `meta.py`: model-free heterogeneity with Farseer's path at narrower bandwidths.
  7. `review_checks.py` (new stage `review`): an independent re-implementation of the model-free estimator (`..._review_independent_modelfree*.csv`) and a finite-difference Farseer wedge (`..._review_farseer_fd.csv`).
  8. Table notes updated.
- **Independent re-implementation.** The model-free σ* is 0.690 (Chinchilla), 0.669 (Llama 3), 0.690–0.694 (Marin) and 0.47–0.48 (Porian). It uses equal weights, windows iterated around each budget's own argmin, and shares no code with `isoflop.py`. That compares with the primary 0.673, 0.660, 0.700–0.713 and 0.51–0.52, and with the builder's iterated-window sensitivity row (0.701, 0.666, 0.682–0.707, 0.49–0.51). The drift is −0.066 per decade (Chinchilla) and −0.056 (Llama 3).

## 6. Referee comments addressed (comment → response)

- **R1 c7 (model-free σ* and w; the headline request).**
  - *Done.* The identity is derived and verified (H1). σ* is estimated on every IsoFLOP design: Chinchilla, Llama 3, Marin × 3 and Porian × 2. The per-budget local polynomials have their order chosen by pooled F test, with window and order sensitivity, and the finite-grid bias is quantified by noise-free and Monte Carlo checks (H2, Section 5).
  - *Result:* 0.66–0.71 on annealed designs, homogeneous; 0.51 on Porian's unannealed profiles.
- **R1 c7.2 / R1 c4 context (in-support w against extrapolation).**
  - *Done on Farseer* up to M = 2,570 in two conventions (H4).
  - The parametric forms understate w by 27–74% at M ≥ 1,024 at the primary bandwidth, and by 18–70% at the narrower cross-validated bandwidth. The κ = 1 fit on M ≤ 100 is the worst, at 0.26–0.30× the local wedge. The sign is robust to bandwidth and to an independent finite-difference estimate.
  - *Not done:* R2 Major 3's parallel test on Marin, whose designs are IsoFLOP rather than factorial. The m9 experiment (M up to 2,031) is not yet available (Section 7).
- **R1 c7.3 (linearity of ln w in ln(M/M*)).** *Done, reframed in review* (H4, C7).
  - The slope at the path equals 1/σ* − 1 for any smooth technology, so it is not a test of the family.
  - Linearity itself is rejected: convexity b₂ = +0.020 (SE 0.001), significant at every bandwidth and in both conventions. It accounts for the parametric understatement at high M.
- **R1 c6(a) / R3 M6.2 / R4 M3 (heterogeneity test; RE summary; ranges).** *Done* (H3; `ra1_modelfree_sigma.tex` Panel B).
  - σ*_κ across the seven technologies: Q = 82.1, p < 0.001, τ = 0.068, RE mean 0.622 [0.549, 0.695].
  - Model-free across the IsoFLOP designs with annealed runs: τ = 0, RE mean 0.695 [0.673, 0.717].
  - Suggested wording is in C4.
- **R1 c6(b) (Farseer: compare like objects; σ at w = 1).**
  - *Done.* The local-quadratic σ is now evaluated on the local expansion path (w = 1): RE 0.703 (0.013), τ = 0.033, against κ-free 0.710 and Eq. 3's path 0.69–0.73.
  - *Review caveat:* this Hessian-based value is bandwidth-sensitive (0.66–0.73). The first-derivative estimate 1/(1+b₁) = 0.708 is stable (0.701–0.711) and supports the like-for-like agreement with κ free.
  - The comparison uses one wild cluster bootstrap by model size (ICC of residuals by size 0.27).
  - The design-conditional SE of σ*_κ is 0.004 with residuals from the κ fit and 0.001 with residuals from the local surface; m2's pairs SE was 0.003. Either way sampling error is small relative to the spread across forms, conventions and samples (0.67–0.82), which is the relevant uncertainty.
- **R1 c6(c) (pseudo-true values depend on design).**
  - *Addressed.* The model-free estimator has no pseudo-true value.
  - On the same runs, κ = 1 overstates σ* by 0.11 on Llama 3 and by 0.08–0.10 on Porian. On Chinchilla's profile runs and on Marin, both parametric forms are within 0.01–0.05 of model-free, and on Marin κ = 1 is the closer (C3).
  - The model-free σ* is homogeneous across designs (τ = 0). This is consistent with, but cannot establish, the κ-free spread across the small sweeps partly reflecting design and specification: those sweeps have no IsoFLOP profiles.
- **R1 c9(a) (design-conditional inference; LAD wild weights).** *Done.*
  - Feng–He–Hu wild bootstrap for Chinchilla, and FHH or Webb wild cluster bootstraps for every σ*_κ.
  - A design-conditional wild bootstrap with full re-estimation for the model-free estimator.
  - Restricted wild bootstraps for tests.
  - Pre-specified procedure: FHH wild, with budget/trunk clusters as the robustness check. The κ = 1 rejection holds at 1% under five of six schemes; under the original nine clusters p = 0.052 (H5).
- **R1 c9(b) (rationale for nine clusters).** *Largely done* (H5). The explicit model of digitization error (interval-censored losses) that the referee suggested is not done.
  - The 108 off-profile runs are Approach-1 trunks of fixed model size. They are re-clustered by trunk (44 clusters) or as singletons (117 clusters).
  - Profile runs keep budget clusters, because a budget's runs share a digitized curve and its reconstructed nominal C.
  - Budget-level dependence is also built into the model-free bootstrap through Webb shocks on each budget's frontier residual.
- **R1 c9(c) (weak identification; Gaussian vs Huber profile; grid boundary).** *Done* (H5).
  - The profile runs over σ* ∈ [0.05, 0.99], with LR relative to the unconstrained optimum, for the Huber (Laplace QLR and Koenker–Bassett) and Gaussian objectives, on the full design and the on-path band.
  - The restricted-bootstrap calibration of the QLR widens the full-design set to [0.68, 0.72].
  - Band sets reach the upper grid boundary, as Proposition 3 implies.
- **R2 Major 8 (σ*_κ robustness; practitioner table; "five studies").**
  - *Done for the items assigned to this module.* σ*_κ robustness table covering conventions, small sizes, outliers and the E–κ trade-off (H6, `ra1_modelfree_kappa_robustness.tex`). On the refined E grid the E–κ trade-off matters for Gadre RedPajama and Muennighoff.
  - *Not addressed here:* removing DataDecide from the headline range or correcting it for the annealing schedule, and tuning quality that varies with D (Step Law). Both belong to the writing and measurement modules.
  - Practitioner table with the de Vries check (H7, `ra1_modelfree_practitioner.tex`).
  - The heterogeneity set with one estimate per study is reported in H3.
- **R2 Major 9(a) (Approach 1 relabel).** The model-free estimator uses the slope of the "IsoFLOP-minima frontier" (the frontier through the per-budget minima). We recommend that label in the paper; this is not Hoffmann et al.'s Approach 1.
- **R2 Major 9(b) (constructed D; FLOP accounting).**
  - *Partially addressed.* The identity needs an exact isocost. Llama 3 and Marin are analysed in the FLOP-implied convention N = C_b/(6D).
  - Marin's configuration-N profiles and FLOP-accounting elasticity η = 0.087 quantify the convention effect: −0.04 to −0.07 on σ*.
  - Chinchilla's σ* under a convention elasticity η is σ*_eff = 2/(2 + S/(1+η)²). The Chinchilla rebuild from Hoffmann's architecture table is not done (Section 7).
- **R3 M6 (what is shown vs claimed; counting).**
  - *Done.* Form-free evidence now exists on seven IsoFLOP designs plus Farseer's path.
  - The heterogeneity sets use "seven sweep–corpus technologies from five studies", with a one-per-study variant.
  - The κ-free fit is presented as a specification check against the model-free benchmark.
  - On R3 M6.1 (complementarity): by the identity, σ* < 1 exactly when the IsoFLOP profile is convex at its minimum and the frontier slopes down. The form-free content of "gross complements" is therefore U-shaped profiles (R1 c6(d)), not a separate finding.
- **R4 M1 (design description; alternative clusters).** *(a) and (b) done.* The design facts are confirmed: 245 runs; 137 on nine profiles; 108 off-profile; FLOP 1.40×10¹⁸–1.30×10²²; 35 trunks. Table 3's SEs, the M* interval, the w(70B) interval and the κ p-value are reported under the trunk and singleton clusterings (H5).
  - *(c) not addressed here:* whether the duality tests use only the 132 profile runs. That is for m1 and the writing module.
- **R4 M5 (profile sets: objective; boundary).** *Done.* Point and set now come from the same objective: Huber σ̂ = 0.701 inside [0.69, 0.71]. The band LR is computed relative to the unconstrained optimum, and grid-boundary hits are reported (H5).
- **R4 M6 (Gaussian LR CES test; bootstrap floors).** *(a) done.* The Gaussian LR (10.40, χ² p = 0.0013) is reported alongside the Wald and robust tests, with its restricted-bootstrap p = 0.17 and the Gaussian sandwich Wald p of 0.060 (HC1) and 0.089 (HC3), added in review.
  - *(c)* is done for this module's p-values: each is labelled with its floor, 1/(B+1).
  - *(b)*, the count of rank-one curvature tests under the κ = 1 Ê, is not addressed here; it is for m2 and the writing module.

## 7. Open issues
1. **Porian et al.'s σ* ≈ 0.51.**
   - Their profiles are unannealed constant-LR losses read along one curve per size. The loss gap to an annealed run plausibly varies with training progress, and so with D/N at fixed C. That would bend the profile and change its curvature relative to the frontier.
   - A cooldown-annealed IsoFLOP (the m9 experiment, or Porian's cosine runs, which have too few sizes per budget) is needed before this counts as technology.
   - Until then Porian is excluded from the "≈ 0.70" statement and shown separately.
2. **Chinchilla's FLOP accounting is not observed.**
   - If Hoffmann et al.'s FLOPs per token vary with N with elasticity η, the curvature rescales as S_eff = S/(1+η)². The model-free σ* then changes by about 2σ*(1 − σ*)·η ≈ 0.44η, i.e. up to ±0.04 for |η| ≤ 0.1 (`conventions.sensitivity_to_eta`).
   - Rebuilding D from Hoffmann's Table A9 architectures and Appendix F FLOP formula (R2 Major 9(b)) remains to be done.
3. **Llama 3's IsoFLOP points are digitized with N = C/(6D).** Meta's FLOP accounting is unknown; the loss units are irrelevant to the identity.
4. **Farseer's local wedge at M ≥ 1,024** comes only from the corner of the design (0.10–0.34B sizes; effective n 15–25). The smoothing bias is not in the intervals. The m9 experiment should be run through `farseer.py` once available.
5. **The design-conditional pooled interval under-covers** (0.82–0.91 in Monte Carlo). The paper should use the random-effects interval.
6. **No model-free σ* for Gadre, OLMo or Muennighoff.** They have no IsoFLOP profiles and only 30–35 runs. Nor is there one for DataDecide, whose checkpoints come before the schedule ends. Their σ*_κ remain parametric.
7. **The drift of σ* with compute (C5)** is estimated from nine (Chinchilla) or eight (Llama 3) budgets that span 2.7 decades, and it is window-sensitive in Chinchilla. It should be confirmed on an annealed design spanning more decades.
8. **Chinchilla's κ interval.** The wild bootstrap interval is centred on κ̂ = 0.774, but the percentile interval inherits the profile's skewness. A subvector-robust procedure (Andrews and Cheng 2012) was not implemented. Instead, the restricted bootstrap calibrates the QLR at the set endpoints.
9. **Farseer's Hessian-based path σ* (review item M-A).** A local quadratic is not the natural order for second derivatives. The path σ* moves from 0.664 to 0.727 across bandwidths, and the primary h_N sits at the edge of its cross-validation grid. A local cubic (the natural order for the Hessian), or the first-derivative estimate 1/(1+b₁) (0.708, stable at 0.701–0.711), should replace it as the reported Farseer model-free σ* in the paper. The heterogeneity table would then use the latter, with a bootstrap SE that does not treat compute levels as independent.
10. **Wild cluster bootstrap for LAD-type fits (review item n-H).** The Webb scheme draws y* = ŷ + η_g|r̃_i|, so every residual in a cluster has the same sign in each draw. That exaggerates within-cluster dependence relative to a standard wild cluster bootstrap (η_g r̃_i) or Hagemann's (2017; `hagemann2017cluster`, added to `lit/bib/extra_ra1_modelfree.bib`) wild gradient bootstrap. It is a sensitivity scheme only, and its conclusions match the others.
11. **Chinchilla window placement (review item n-K).** The windows are centred on a pooled Approach-2 path, so at the lowest budget the argmin sits 0.42 in ln N from the window centre. Around each budget's own argmin (the iterated sensitivity row), σ* is 0.701 against 0.673 (about 1 SE). The noise-free bias checks favour the primary choice, but the spec's "symmetric window around the argmin" is only approximately met.
