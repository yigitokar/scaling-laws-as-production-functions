# Independent review: module ra1_modelfree ("The technology from designed variation, model-free")

Reviewer: Claude, acting as independent replicator and skeptical referee. Date: 2026-09-24.

Scope: re-run `code/analysis/ra1_modelfree/run.py` from scratch; audit the code; check every memo number against regenerated outputs; check referee-comment coverage and citations; fix problems in code, outputs and memo, then re-run.

## 0. Verdict

The module is careful and mostly correct.

- **Replication is exact.** A from-scratch run reproduces all 43 table files byte for byte.
- **The core results survive audit.** These are the model-free identity, the IsoFLOP estimates (0.66–0.71 on annealed designs), the Chinchilla inference fixes, the heterogeneity statistics and the practitioner table.
  - An independent re-implementation of the estimator, sharing no code with `isoflop.py`, lands in the same range.

Three claims were overstated, and one inference choice was reported selectively:

| Item | Problem | Resolution |
|---|---|---|
| **M-A** | Farseer bandwidth. The primary h_N is the smallest multiple on its cross-validation grid, i.e. a boundary solution. Only wider bandwidths were reported. At the narrower CV optimum the κ-free and Eq. 3 high-M gaps shrink by about a third, and the Hessian-based local-path σ* drops from 0.707 to 0.664. | Sensitivity added. A bandwidth-robust first-derivative estimator of the path σ* (0.708; range 0.701–0.711) was added. Claims revised. |
| **M-B** | The "linearity test" (p = 0.36) is not a test of the family. At the path, d ln w/d ln M = 1/σ* − 1 for *any* smooth technology; this is an identity, now verified numerically. The testable content, linearity, is rejected (convexity b₂ = 0.020, SE 0.001). | Reframed in the memo. |
| **M-C** | "The model-free σ* agrees with κ free and rejects κ = 1 wherever they differ" holds only for Llama 3 and Farseer. On Chinchilla's own 137 runs, κ = 1 (0.692) is within 1 SE of model-free (0.673). On Marin, κ = 1 is closer than κ free. | Claim narrowed. |

Smaller items: the Gaussian CES "artifact" wording (sandwich Wald p = 0.06–0.09); the E–κ grid resolution (the trade-off is material for Gadre RedPajama and Muennighoff); the centring of wild percentile intervals; referee-coverage statements marked "done" that are partial; and three numeric slips. All are fixed.

No pre-existing number in the outputs changed except the E-set columns of the κ-robustness table.

## 1. Replication

| Step | Result |
|---|---|
| From-scratch run of the builder's code (`RA1_OUTPUT_ROOT` = scratch; 806 s; ≤ 5 processes, CPU only) | Exit 0. All **43** `output/tables/ra1_modelfree_*` files are **byte-identical** to the installed ones. The figures were regenerated and inspected (four PNGs). Log: `data/processed/ra1_modelfree/run_stdout_review_replication.txt`. |
| From-scratch run of the reviewed code before the `review` stage was added (scratch root; 775 s) | Exit 0. Every pre-existing CSV column and row is unchanged, except the E-set columns of `kappa_robustness` (checked programmatically). |
| `--stages tables,figures` from the installed caches | All 45 table files are byte-identical to the installed ones, so the caches and tables are consistent. |
| Final end-to-end run including the new `review` stage (scratch root; 704 s) | Exit 0. All **48** table files are byte-identical to the installed outputs. Its log (`run_stdout.txt`), stage caches and figures are installed. |

## 2. Code audit

### 2.1 Verified correct
- **Identity** (`isoflop.verify_identity`).
  - Derivation checked by hand for the Chinchilla form: L_nn = α²R and |dL*/dc| = αR/2 give 1/σ* − 1 = α = (α+β)/2 when α = β.
  - The numerical checks agree to below 10⁻⁸, including against the full two-input Hicks formula on the non-homothetic Eq. 3. Invariance to ln L was also checked.
- **Model-free estimator** (`isoflop.py`).
  - Curvature = 2c₂ of the window polynomial in ln N.
  - Frontier slope = L*·d ln L*/d ln C, from a cubic polynomial of ln L* in ln C.
  - S_b = κ_b/|s_b| and σ* = 2/(2 + S).
  - Budget validity (≥ 2 runs on each side) is judged at the point estimate.
  - The design-conditional wild bootstrap re-estimates the path centres, windows, polynomials and frontier in every draw. Budget-level Webb shocks enter through the leverage-adjusted frontier residuals.
  - Basic intervals are centred on the estimator applied to the population.
  - DerSimonian–Laird, Cochran Q, I², HKSJ (modified, q ≥ 1) and prediction intervals (t_{k−2}) follow the standard formulas (`meta.meta`).
- **Chinchilla design facts.**
  - 245 runs: 137 on the nine profiles, 108 off; FLOP 1.40×10¹⁸–1.30×10²²; 35 off-profile trunks.
  - The five highest-loss runs all sit on the 10¹⁹ profile at N ≥ 2.0B. None falls in any model-free window (checked).
  - Porian points: 16 sizes per corpus. `width` identifies the run, so the run-level shocks are correctly shared.
- **FHH wild bootstrap** (`parametric.fhh_residuals`, `ra1_common.fhh`). It matches `quantreg::boot.rq` "wild": r̃ = r + h(τ − 1{r<0})/f̂(0), and y* = ŷ + v|r̃| with v ∈ {−2τ, 2(1−τ)}.
- **Test statistics.** Laplace QLR 2n ln(H_r/H_u); Gaussian LR n ln(SSR_r/SSR_u); Koenker–Bassett 4f̂(0)(S₁^r − S₁^u). The restricted bootstraps generate the data from the restricted fit, and the null holds for both estimators' targets (E[v] = 0).
- **Practitioner formula.** The ratio C/C_min = ((α+βw)/(α+β))^{1/γ}w^{−1/α} was re-derived from the isoquant. The homothetic reduction cosh(ρ ln k/2)^{2/ρ} with ρ = 1/σ* − 1 was confirmed, and every entry was checked by brute force in code.
  - de Vries (2023): the post's parameters (E 1.69, A 406.4, B 410.7, **α = 0.32**, β 0.28) were verified at the source URL. The 99.0% overhead at 30% size and k = 21.1 reproduce.
  - His curve for 30–100% of the optimal size lies between the σ* = 0.74 and 0.80 curves at all 70 points (checked).
- **Marin FLOP-accounting prediction.** S_cfg = S_eff(1+η)², with η = 0.087, reproduces the predicted configuration-N values of 0.664, 0.677 and 0.669.
- **Conventions.** The identity requires exact isocosts. For Llama 3 (N = C/(6D)), the curvature in ln N at fixed C equals the curvature in ln D. The digitized x-axis (tokens) therefore enters directly, and only Meta's unknown FLOP accounting matters, as the memo says.

### 2.2 Issues found

| ID | Severity | Issue | Fix | Status |
|---|---|---|---|---|
| **M-A** | Major | **Farseer bandwidth and selective sensitivity.** The LOO-CV h_N equals the smallest multiple on the grid (0.2 s.d.) in both conventions. On an extended grid the optimum is 0.125 s.d. (h_N = 0.148; 0.10 s.d. with embeddings). The memo reported only ×1.5 and ×2, and said "wider bandwidths make the gap larger, not smaller" and "the primary CV bandwidth gives the smallest gap". At the extended CV optimum the M ≥ 1,024 gaps are −0.70 (Chinchilla), −1.22 (Chinchilla, M ≤ 100), **−0.25 (κ free) and −0.20 (Eq. 3)**, against −0.82, −1.36, −0.36 and −0.31. The Hessian-based local-path σ* mean is **0.664** (0.683 at h_N × 0.75), against 0.707. The path σ* is a second-derivative object estimated with a local quadratic, which is not the natural order for second derivatives. | `farseer.py`: extended CV table; sensitivity variants "h_N × 0.75" and "extended-grid CV optimum" with pooled-level path summaries. `tables.py`: **first-derivative path σ* = 1/(1 + b₁)**, where b₁ is the slope at u = 0 of ln w_local on u = ln(M/M*). It is 0.708 [0.705, 0.710] and **0.701–0.711 across all bandwidths** (emb 0.665–0.674). `meta.py`: model-free heterogeneity with the narrower-bandwidth Farseer input (RE 0.679 [0.656, 0.703], Q = 5.5, p = 0.36; still homogeneous). Memo H3, H4, C2, C6, 5.4 and 7.9 revised. | Fixed (reporting). Primary bandwidth kept, since derivatives warrant more smoothing than function-CV. Recommend the first-derivative estimate for the paper. |
| **M-B** | Major | **Linearity test misframed (R1 c7.3).** Differentiating w = f_n/f_d along the isocost at f_n = f_d gives d ln w/d ln M\|_C = −(f_nn − 2f_nd + f_dd)/(2f_n) = 1/σ* − 1 for *every* smooth technology. The comparison of the OLS slope (0.420) with the path 1/σ* − 1 (0.434), p = 0.36, compares two estimates of one object. It was presented as "the family's first-order implication is not rejected". It also flips across bandwidths (0.41 against 0.56 at the CV optimum). The family's testable restriction is linearity, which is rejected: b₂ = +0.020 (SE 0.001; bootstrap p at the floor) at every bandwidth, 0.033 with embeddings. The convexity adds about +0.37 to ln w at u ≈ 4.3, which is exactly what the parametric forms miss at high M. | `isoflop.py`: numerical check of the corollary (column `dlnw_dlnM_at_path` in both identity CSVs; equal to 1/σ* − 1 to 6 digits for the Chinchilla form, the κ family and Eq. 3 at every C). Memo H1, H4, C7, 5.4 and Section 6 rewritten. | Fixed |
| **M-C** | Major (claims) | **"Model-free agrees with κ free and rejects κ = 1 wherever they differ"** (H2 summary, C3, builder's headline). Same-run evidence: Chinchilla κ = 1 gives 0.692 (0.013) and κ free 0.667, against model-free 0.673 (0.027), both within 1 SE. Marin κ = 1 (0.673–0.690) is *closer* than κ free (0.663–0.678) to model-free (0.700–0.713). The Chinchilla contrast with 0.737 compares the 137 profile runs with the 240-run design (108 off-profile runs), across samples. Discriminating evidence exists only for Llama 3 (4.8 SE) and Farseer. | Memo H2, C3 and Section 6 (R1 c6(c)) revised. Also: "2.4 SE" corrected to 2.3; "1.4 SE" to 1.5; Llama 3 κ = 1 "0.770" to 0.769 (0.769475). | Fixed |
| m-D | Moderate | **Gaussian CES test "artifact".** A population-free calibration check gives heteroskedasticity-robust Wald statistics for α = β under the same Gaussian NLS fit of 3.54 (HC1, p = 0.060) and 2.90 (HC3, p = 0.089), against 10.64 homoskedastic. That is consistent with the restricted bootstrap (p = 0.17) but borderline at 10%. The Gaussian κ = 1 test survives (HC1 40.7). | `chin_inf.py`: `wald_gauss_{homosk,hc1,hc3}` and p columns in `chinchilla_tests.csv`; table note. Memo H5, C8 and 5.4: "not robust to heavy tails; borderline at 10%" in place of "artifact". | Fixed |
| m-E | Minor | **Wild percentile intervals centred on the population's pseudo-true values.** The wild population is the κ-free fit, whose Chinchilla-form pseudo-true values are w(70B) 0.978 (estimate 1.040), M*(5.76e23) 21.3 (17.9) and α − β −0.011 (−0.020). Percentile intervals of the Chinchilla-form objects are therefore not centred on the estimates. Basic intervals (in logs) are nearly identical for FHH: M* [10.2, 36.4] and w(70B) [0.81, 1.27], against [10.5, 37.4] and [0.80, 1.26]. For Webb they are [10.2, 41.4] and [0.77, 1.25]. | `chin_inf.py`: `pop_*`, `lo_basic_*` and `hi_basic_*` for the wild schemes; table note; memo H5 note. | Fixed; no conclusion changes |
| m-F | Moderate | **E–κ trade-off under-resolved.** The 36-point E grid resolved the χ² E sets with 2–7 points. On Gadre RedPajama, σ*_κ at the first excluded neighbour was 0.692. On a grid refined with 25 points around each set, σ*_κ over the set is Chinchilla 0.697–0.703, Farseer 0.709–0.711, OLMo 0.541–0.545, Gadre C4 0.600–0.614, Gadre RefinedWeb 0.580–0.605, **Gadre RedPajama 0.606–0.685** and **Muennighoff 0.503–0.555**. The claim "immaterial inside the E sets for every technology" was false for two technologies. | `kappa_rob.e_profile` refines the grid (column `fine`); memo H6 and C9. | Fixed |
| m-G | Moderate | **Referee coverage marked "done" that is partial.** R1 c9(b): the digitization-error model was not built. R2 Major 8: DataDecide and Step-Law tuning were not addressed in this module. R2 Major 3.2: the Marin M ≤ 100 test was not done. R4 M1(c): the 132-run duality statement was not addressed. R4 M6(b): the rank-one count was not addressed. | Memo Section 6 corrected. | Fixed (documentation) |
| m-I | Trivial | The `farseer.py` docstring said the statistic uses n_eff ≥ 25; the primary is 15. | Docstring fixed. | Fixed |
| n-H | Minor | The Webb wild cluster scheme uses y* = ŷ + η_g\|r̃_i\|, so all residuals in a cluster share a sign in each draw. That is more dependence than a standard wild cluster bootstrap (η_g r̃_i) or Hagemann's (2017) wild gradient bootstrap. It is a sensitivity scheme only, and its results sit between the FHH and cluster schemes. | Documented (memo 7.10; `hagemann2017cluster` added to `lit/bib/extra_ra1_modelfree.bib`, Crossref-verified, DOI 10.1080/01621459.2016.1148610). | Noted |
| n-J | Minor | Farseer path summaries differ in which compute levels they use. The pooled σ* (RE 0.703; mean 0.707) uses the 8 levels with a root on both the data and the population surfaces. The linearity row's "path 1/σ* − 1" averages 9 levels, including 2×10¹⁸ (σ* = 0.644, no population root). | The new sensitivity keys `path_sigma_mean_poolC` use the pooled levels. | Noted |
| n-K | Minor | Path-centred windows: at Chinchilla's lowest budget the argmin sits 0.42 in ln N from the window centre, so the window is asymmetric about the argmin (the spec asked for symmetric). Windows iterated around each argmin give 0.701 against 0.673 (about 1 SE). The noise-free bias checks (≤ 0.006) favour the primary choice. | Documented (memo 7.11). | Noted |
| n-L | Minor | The within-design DerSimonian–Laird RE treats per-budget S_b as independent, although they share the frontier fit. Monte Carlo coverage (0.92–0.98; R = 100, MC s.e. about 0.022) supports the RE interval, but only for three designs and under constant-σ* truths (no drift, τ = 0). Coverage under non-homothetic truths is untested. | — | Open (recommend a drift truth, e.g. Farseer Eq. 3, in the Monte Carlo) |
| n-M | Minor | The practitioner figure plots de Vries's curve against k relative to the optimal model's ratio at the reference compute (22.1 at 30%), while the σ* curves use k at the model's own compute (21.1). The difference is invisible at plot scale. | — | Noted |

## 3. Numbers in the memo checked against regenerated outputs

Every number below was compared with the regenerated CSVs; all match unless noted.

- **H1:** 0.737028, 0.700553; Eq. 3 0.745 → 0.672.
- **H2 table:** all 8 rows and every column. RE, SE, τ, Q and bootstrap p; pooled intervals; drift, SE and p; same-run parametric values. The one exception is Llama 3 κ = 1, 0.770 → 0.769, now fixed.
- **H2 other:** invalid budgets; noise-free bias (≤ 0.006; Porian κ +0.047/+0.050); Monte Carlo (bias and coverage for three designs); sensitivity ranges (Chinchilla 0.60–0.71 and 0.665–0.714; Llama 3 0.64–0.68; Marin 0.64–0.72).
- **H3 table:** all 10 rows (μ, HKSJ interval, τ, I², Q, p, PI); pairwise z (Chinchilla–Farseer p = 0.51; OLMo z = 6.2 and 7.8; Muennighoff 3.3 and 3.6); leave-one-out I² ≥ 0.78.
- **H4:** local ln w by bin; Δ for five comparators in two conventions; ratios and intervals; embedding gaps; run-level SE; ICC 0.27 and 0.07; ×1.5 and ×2 gaps; n_eff ≥ 25 values; linearity numbers; path σ*(C) range 0.66–0.76 and M* 25–45; RE 0.703 (0.013); mean 0.707 [0.700, 0.718]; drift; embedding path 0.674 (0.019); parametric Farseer σ* table.
- **H5:** design facts; the full six-scheme table (checked cell by cell); tests (31.2, 46.2, 20.7; 1.84, 10.40, 1.20; critical values 8.6, 13.4, 8.6, 20.2); profile sets; calibration (11.4, 9.0, 7.7; 29.0, p = 0.003; band 6.5, 5.2); band profile (LR ≤ 1.21 on [0.25, 0.94]; 3.4 at 0.99; κ bound at σ* ≥ 0.98).
- **H6:** every baseline and variant σ*_κ and SE.
- **H7:** the whole table, the non-homothetic deviations (4.5% and 19%), and the de Vries numbers.
- **Section 5.3 sensitivity table:** all 7 × 13 cells; configuration-N values; the η prediction.

## 4. Referee comments: are they addressed?

| Comment | Status after review |
|---|---|
| R1 c7 (model-free σ*) | **Addressed.** Estimated on all seven IsoFLOP designs requested, with finite-grid bias and Monte Carlo checks. Independently re-implemented (0.690/0.669/0.690–0.694/0.47–0.48 against primary 0.673/0.660/0.700–0.713/0.51–0.52). |
| R1 c7.2, R2 Major 3.2 (in-support w against extrapolation) | **Addressed on Farseer.** Sign robust; size for the flexible forms bandwidth-sensitive (now reported); a finite-difference check agrees (ln w 1.68–1.93 at M ≈ 1,270, above every parametric comparator: κ free 1.59–1.60, Eq. 3 1.63–1.64). **Not done:** Marin M ≤ 100 (IsoFLOP, not factorial); the m9 experiment is pending. |
| R1 c7.3 (linearity) | **Addressed after reframing** (M-B). Linearity rejected; convexity quantified. |
| R1 c6(a), R3 M6.2, R4 M3 (heterogeneity) | **Addressed.** Q, τ, I², HKSJ, PI, leave-one-out and one-per-study. |
| R1 c6(b) (like objects on Farseer) | **Addressed**, with the bandwidth caveat and the first-derivative estimator (0.708) added. |
| R1 c6(c) (pseudo-true values) | **Addressed**, with claims narrowed (M-C). |
| R1 c9(a) (design-conditional inference) | **Addressed.** |
| R1 c9(b) (nine clusters) | **Largely addressed.** Trunk and singleton clusters. The digitization-error model is not done. |
| R1 c9(c) (weak identification) | **Addressed** with a restricted-bootstrap calibration. Andrews–Cheng is not implemented (acceptable per the referee's "or at least"). |
| R2 Major 8 | **Addressed for this module's items** (conventions, small sizes, outliers, E–κ, practitioner table, five studies). DataDecide and Step-Law tuning belong elsewhere. |
| R2 Major 9(a) | Label recommended ("IsoFLOP-minima frontier"). |
| R2 Major 9(b) | **Partial**, as the memo says. Sensitivity formula and Marin convention done; Chinchilla architecture rebuild not done. |
| R3 M6 | **Addressed.** One addition: the identity shows that σ* < 1 exactly when the profile is U-shaped, so "complementarity" has that form-free content only. |
| R4 M1 | (a) and (b) **addressed**; (c) not addressed here. |
| R4 M5 | **Addressed.** |
| R4 M6 | (a) **addressed** (plus a sandwich check); (c) addressed for this module; (b) not addressed here. |

## 5. Citations
- The 8 new keys in `lit/bib/extra_ra1_modelfree.bib` were checked against my knowledge of the sources: authors, journal, volume, issue, pages and DOI are correct for Feng–He–Hu 2011, DerSimonian–Laird 1986, Cochran 1954, Higgins–Thompson 2002, Hartung–Knapp 2001, Sidik–Jonkman 2002, Mammen 1993 and Koenker–Bassett 1982.
- One key was added in review: `hagemann2017cluster` (verified via Crossref).
- Every existing key cited in the memo and tables is present in `paper/references.bib` or `lit/references.bib`, or in a module extra bib that is already in the paper bib.
- `devries2023go`: the post's parameters and the "about 100% overhead at about 30% size" statement were verified at the URL.

## 6. Changes made

**1. Code.** No existing estimator or seed changed. The changes are additions:
- `farseer.py`: `EXT_MULTS` and the extended CV; narrower sensitivity variants; pooled-level path summaries; docstring fix.
- `isoflop.py`: slope-identity check.
- `chin_inf.py`: Gaussian sandwich Wald tests; population pseudo-true values and basic intervals for the wild schemes.
- `kappa_rob.py`: refined E grid.
- `meta.py`: narrower-bandwidth model-free heterogeneity rows.
- `tables.py`: `sigma_slope()`; notes on bandwidth, linearity, sandwich tests and basic intervals; the Panel B row "Model-free, excl. Porian; Farseer h_N at CV opt.".
- `run.py`: new `review` stage, which calls the new `review_checks.py` (independent model-free re-implementation and finite-difference Farseer wedge).

**2. Outputs.**
- New CSVs: `farseer_bandwidth_cv_extended`, `farseer_sigma_slope`, `review_independent_modelfree`, `review_independent_modelfree_budgets`, `review_farseer_fd`.
- New columns and rows in `chinchilla_schemes`, `chinchilla_tests`, `identity_check*`, `farseer_sensitivity`, `heterogeneity` and `kappa_E_profile`.
- `kappa_robustness`: E-set columns only.
- Table notes updated in `ra1_modelfree_sigma.tex`, `..._farseer_w.tex`, `..._chinchilla_inference.tex` and `..._kappa_robustness.tex`.
- Figures are unchanged in content.

**3. Runs.**
- Final end-to-end run of the reviewed code (775 s) installed.
- `--stages review` run on the installed caches.
- A final full run including the `review` stage (scratch root, 704 s) was compared file by file with the installed outputs: all 48 table files are byte-identical.

**4. Memo** (`output/memos/ra1_modelfree.md`). Reproduction block; H1 corollary; the H2 summary and comparisons; H3, H4, H5 and H6 notes; methods 2.5 and 2.7; inventory; claims C2, C3, C6, C7, C8 and C9; Sections 5.4 and 5.5 (new); Section 6 coverage; Section 7 items 9–11.
- The builder's version is preserved at `data/processed/ra1_modelfree/ra1_modelfree_memo_builder_version.md` for diffing.

## 7. Remaining concerns and recommendations for the paper

1. **Farseer's model-free σ*.**
   - Report the first-derivative estimate 1/(1 + b₁) = 0.708 (0.701–0.711 across bandwidths), or a local-cubic Hessian, not the local-quadratic Hessian path value. That value is bandwidth-sensitive (0.664–0.727) and its SE excludes smoothing bias.
   - The b₁-based SE (0.001) is design-conditional and also excludes smoothing bias. Quote the bandwidth range as the relevant uncertainty.
2. **The high-M wedge (Section IV credibility check).**
   - Quote the understatement as a range: κ free 0.70–0.78× and Eq. 3 0.73–0.82× across the CV-optimal and primary bandwidths; Chinchilla form (all runs) 0.44–0.50×; Chinchilla form fitted on M ≤ 100, 0.26–0.30×.
   - It rests on the 0.10–0.34B corner of one design.
3. **Linearity.** Say "ln w is convex in ln(M/M*); the family's linearity is rejected, and the convexity is what makes parametric extrapolation understate w". Do not say "linearity not rejected".
4. **κ = 1 against model-free.** Only Llama 3 and Farseer discriminate. Do not use the 0.737 (240-run) against 0.673 (137-run) contrast as a like-for-like test.
5. **Homogeneity of the model-free σ*.** State it with its power limit (between-design s.d. below about 0.02 is undetectable). Note that the designs cover different compute ranges while σ* drifts with compute in Chinchilla and Llama 3.
6. **Monte Carlo** for the model-free RE interval under a non-homothetic truth (Farseer Eq. 3 fitted to each IsoFLOP design), since the drift finding (C5) is itself a headline.
7. **Not addressed in this module** (hand-offs): Chinchilla D rebuild from Hoffmann's architecture table (R2 9(b)); digitization-error model (R1 c9(b)); DataDecide and Step-Law items (R2 Major 8); R4 M1(c) and M6(b).
