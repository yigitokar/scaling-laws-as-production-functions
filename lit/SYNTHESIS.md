# SYNTHESIS — "Scaling Laws as Production Functions"

Synthesis of the eight literature strands (compiled 2026-09-23). Inputs: `lit/notes/{ml_core, ml_estimation, ml_observational, io_controlfn, io_ces_duality, econ_ai, novelty, data_sources}.md` (all read in full) and the merged bibliography `lit/references.bib` (309 works; old→new keys in `lit/bib_aliases.md`).

**Conventions.**
- Citations: `[key]` = a key in `lit/references.bib`. Cite only canonical keys. For example, `li2025predictable` is retired: use `li2025predictablea` (Step Law) or `li2025predictableb` (Farseer).
- **[D]** = derived by the strands and re-computed in code during synthesis (script logic in §2). **[UNV]** = unverified, to be checked. **[H]** = hypothesis.
- Notation:
  - N = parameters, D = training tokens, C ≈ 6ND = training FLOPs, M ≡ D/N = tokens per parameter.
  - L = held-out loss, E = irreducible loss, ℓ ≡ L − E = reducible loss.
  - Chinchilla form: ℓ = A N^−α + B D^−β.
  - Output elasticities: ε_N ≡ −∂ln ℓ/∂ln N = αA N^−α/ℓ, and ε_D likewise.
  - Allocation exponent a ≡ β/(α+β), so N* ∝ C^a.
  - Frontier exponent γ ≡ αβ/(α+β), so ℓ*(C) ∝ C^−γ.
  - On-path elasticity of substitution σ* ≡ 2/(2+α+β).
  - Compute-optimal ratio M* ≡ D*/N* at a given C.
  - Wedge w ≡ ε_N/ε_D.

---

## 1. Executive summary

1. **The mapping is real, mostly exact, and mostly unexploited.** Several pieces of the scaling-law workflow are exact identities of IO/production theory:
   - iso-loss curves are isoquants;
   - IsoFLOP curves are isocosts;
   - the compute-optimal allocation is the conditional factor demand;
   - the loss–compute frontier L*(C) is the inverse cost function;
   - Chinchilla's Approaches 1/2/3 are cost-function, factor-demand (FOC) and primal estimation.

   Because C = 6ND is *multiplicative* (log-linear, with unit log-cost weights), cost minimization equates **output elasticities**, ε_N = ε_D. So share-equation (GNR/DLW-type) identification needs *no price data*.

   ML has rediscovered almost every classic IO identification problem under other names:
   - functional dependence: "fixed tokens-per-parameter designs are ill-conditioned" [kricheli2026tokens];
   - input measurement error: embedding parameters, head FLOPs [pearce2024reconciling; porian2024resolving];
   - flexible-input mis-tuning [porian2024resolving; lourie2026small];
   - transmission/selection bias [whitfill2025note; konig2026validity];
   - DMR non-identification of factor bias, which appears as insignificant α_year and β_year with an identified sum [ho2024algorithmic];
   - TFPQ vs TFPR [schaeffer2023emergent].

   Nobody has applied the IO estimators (OP/LP/ACF, GNR, DLW, dynamic panel, SFA, Nerlove duality) or the IO remedies to training-scaling data (novelty strand, §6).

2. **Parameters and data are gross complements, σ ≈ 0.74–0.76, and σ is more stable than the allocation exponent.**
   - σ* is 0.763 (Hoffmann rounded), 0.762 (Hoffmann TeX), 0.737 (Besiroglu refit) and 0.739 (Muennighoff, α = β).
   - Over the same refits, a moves 0.452 → 0.513 and the compute-optimal M* at Gopher compute moves 93 → 18 [D].
   - Kaplan's joint law implies much stronger complementarity: σ ∈ [0.50, 0.575], 0.535 on-path [D]. So Kaplan and Chinchilla disagree about *curvature*, not only about the expansion path.
   - The σ *formula* is already in [hao2026theory]. What is new is the on-path closed form σ* = 2/(2+α+β), estimates with standard errors across independent sweeps, and a test of CES.

3. **Identification is the heart of the paper.**
   - On the compute-optimal path, the Hessian of ln ℓ in (ln N, ln D) is rank one, and its null direction is the expansion path [D]. On-path data therefore carry no curvature information.
   - On-path data identify γ and a. Under Chinchilla's outer-exponent restriction (κ = 1), they identify α = γ/a and β = γ/(1−a) *by functional form*.
   - They do **not** identify: σ nonparametrically (any σ ∈ (0,1) fits when κ is free); the MRTS level (A/B, hence M*) unless optimality is imposed; or the sign of the Hicks bias of technical change (DMR) [D; io_ces_duality §2.4; io_controlfn §7.A].
   - Corollary: *the better labs optimize, the less their data reveal.*
   - IsoFLOP sweeps and factorial ladders are the experimenter-made analog of ACF's i.i.d. input-price shocks.

4. **The object that on-path data cannot identify, M* (the MRTS level), is exactly the object the inference wedge needs.**
   - Sardana-type lifetime cost minimization gives **w = ε_N/ε_D = 1 + T/(3D) = lifetime/training compute** [D]; see [sardana2024beyond] for the forward problem.
   - Under homothetic CES (α = β = ρ), this collapses to a two-parameter sufficient statistic [D, new closed form]:
     - **T/D = 3[(M/M*)^ρ − 1]**;
     - training-cost overspend relative to the training-only frontier: **C/C_min = cosh(½ρ ln(M/M*))^{2/ρ}**.
   - For Llama-3-8B (M = 1,875), T/D ranges from 2.9 to 17 as (M*, ρ) span published values; M* ∈ {16, 20, 41, 192} matters far more than ρ.
   - Using Meta's *own* IsoFLOP law [grattafiori2024llama] (D* = 0.299·C^0.537, ρ = 0.35 assumed): the 405B flagship sits on its expansion path (w = 0.97, T/D ≈ −0.1), while the 8B and 70B imply T/D ≈ 9.6 and 2.6 [D, illustrative]. This is a within-family revealed-preference design: the flagship pins M* (GNR direction), and the siblings reveal inference demand (DLW direction).
   - This inversion is **novel** (novelty verdict C7).

5. **Cross-lab ("observational") scaling laws are production functions with firm fixed effects estimated by OLS/NLS**, with no endogeneity treatment [ruan2024observational; maiapolo2024sloth; mertens2026secret; ho2024algorithmic].
   - The public data carry functional dependence *by construction*:
     - FLOPs = 6ND in 100% of ObsScaling rows and in 53% of Epoch language models;
     - D is constant within Llama-2/3/3.1, Pythia and OPT;
     - D = 20N in Cerebras-GPT;
     - D = 100N in DataDecide final checkpoints.
   - The experimental–observational gap is large: Ho et al.'s cross-lab β_data = 0.040 [0.023, 0.062] vs about 0.28–0.37 in sweeps. Whitfill reads this as progress overstated by up to about 9×; attenuation plus simultaneity are the IO reading.
   - This invites a **LaLonde-style benchmark**: experimental (IsoFLOP/ladder) elasticities vs observational estimators (OLS, FE, control function, dynamic panel, IV).

6. **Algorithmic progress is TFP growth, with three IO corrections.**
   - (i) Factor-augmenting specifications [ho2024algorithmic; erdil2022algorithmic] identify only the sum g_N + g_D (DMR).
   - (ii) Hicks neutrality means α·g_N = β·g_D, a proportional shift of A and B. It is *not* an E shift. Factor augmentation can never move the allocation exponent a, so DeepSeek's quality-dependent a (0.450 → 0.578) [bi2024deepseek] is non-neutral, curvature-changing change [D].
   - (iii) The Kaplan→Chinchilla re-balancing, which [gundlach2025origin] counts as ~10× of 6,930× "algorithmic progress", is **Farrell allocative efficiency, not technical change**.
   - Constant compute-equivalent gain ⇔ equal exponents and E [D]. So scale-dependent gains (LSTM → Transformer) are non-neutral, and "reference dependence" is the index-number problem [diewert1976exact].

7. **Measurement is first-order and has exact IO analogs.**
   - Inputs: Kaplan's non-embedding N is Nerlove's small-firm spurious scale economy. Porian's step-by-step path is a 0.835 → 0.706 → 0.602 → 0.571 → 0.497 [porian2024resolving].
   - Output: per-token loss is a tokenizer-specific unit (Klette–Griliches). BPB ≈ TFPQ; accuracy, Elo and revenue are TFPR-like bounded transforms [foster2008reallocation; bond2020unpleasant].
   - σ and isoquants are ordinal, i.e. invariant to any monotone output transform. Returns to scale and TFP *levels* are cardinal.

8. **Estimation practice is fragile in exactly the ways IO learned about the hard way.**
   - The Chinchilla Approach-3 fit was an optimizer failure: averaged Huber loss and early L-BFGS stop, with CIs about 50× too narrow [besiroglu2024chinchilla]. This is the BLP convergence lesson [knittel2014estimation; dube2012improving].
   - Huber with δ = 10^−3 on log loss is effectively LAD [koenker1978regression].
   - The 5-parameter surface has condition number about 3.5×10^11, which variable projection reduces to about 11 [czech2026problems]. KMW normalization cuts it 3,465 → 59 on a Chinchilla-like design [D].
   - A–α and B–β ridges: 3 PCs explain 99.49% of the variance of the fitted parameters across families [choshen2024hitchhikers].

9. **Economists calibrate, never estimate, this technology.** Examples: "compute and data are gross complements" as a footnote [trammell2023economic]; productivity = D^η flagged as a "(substantial!) assumption" [farboodi2026datadriven]; homogeneous CES in inference tokens [bergemann2026menu]; log-linear automation [erdil2025gate]. Our σ, γ and wedge estimates feed these models directly.

10. **Recommended paper.** Two headline results, one supporting application.
    - **(H1)** An identification theory (duality + functional dependence + DMR), with the first standard-errored estimates of the parameter–data elasticity of substitution and of the technology across public sweeps.
    - **(H2)** "Over-training reveals anticipated inference demand": the DLW/GNR duality, with estimates for open-weight families and validation against usage and prices.
    - **(S)** Observational production functions and algorithmic progress: a LaLonde-style experimental benchmark for IO estimators, plus a Farrell/DMR decomposition of "algorithmic progress".
    - All three are feasible on a laptop with public data (§8).

---

## 2. Canonical numbers ledger (all re-computed in code during synthesis)

### 2.1 Parameter sets and derived technology objects

| Parameter set | E | A | B | α | β | a = β/(α+β) | γ = αβ/(α+β) | −∂lnC*/∂lnℓ = 1/γ | σ* | σ range | σ*/(1−σ*) | M* at 1e21 / 5.76e23 / 3.8e25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Hoffmann A3, rounded [hoffmann2022training] | 1.69 | 406.4 | 410.7 | 0.34 | 0.28 | 0.452 | 0.1535 | 6.51 | 0.763 | [0.746, 0.781] | 3.23 | 50 / 93 / 139 |
| Hoffmann A3, TeX precision (via [besiroglu2024chinchilla]) | 1.6934 | 406.4 | 410.7 | 0.3392 | 0.2849 | 0.457 | 0.1548 | 6.46 | 0.762 | [0.747, 0.778] | 3.20 | 34 / 59 / 85 |
| Besiroglu refit (SEs) [besiroglu2024chinchilla] | 1.8172 (0.03) | 482.0 (124.6) | 2085.4 (1293.2) | 0.3478 (0.02) | 0.3658 (0.02) | 0.513 (0.018) | 0.1783 | 5.61 | 0.737 | [0.732, 0.742] | 2.80 | 21.6 / 18.4 / 16.5 |
| Muennighoff C4, α = β [muennighoff2023scaling] | 1.87 | 521 | 1488 | 0.3527 | 0.3527 | 0.500 | 0.1763 | 5.67 | 0.739 | — | 2.84 | 19.6 (constant) |
| Kaplan joint L(N,D), Table 2 [kaplan2020scaling] | 0 | — | — | inner exponents 0.738 (N), 1 (D); outer κ = 0.103 | | 0.575* | — | — | 0.535 | [0.500, 0.575] | — | — |
| MiniCPM average [hu2024minicpm] | — | — | — | 0.29 | 0.23 | 0.442 | 0.128 | — | 0.794 | [0.775, 0.813] | — | ≈192 (UNV constants) |
| Krajewski, dense [krajewski2024scaling] | 0.47 | 16.3 | 26.7 | 0.126 | 0.127 | 0.502 | 0.063 | — | 0.888 | — | — | — |
| Tao (α1 = β) [tao2024scaling] | — | — | — | 0.447 | 0.447 | 0.500 | 0.224 | — | 0.691 | — | — | — |
| Abnar MoE [abnar2025parameters] | 0.94 | — | — | 0.596 | 0.395 | 0.399 | 0.238 | — | 0.669 | [0.626, 0.717] | — | — |
| OLMo-Hybrid (via [hao2026theory] fn. 3; secondary) | — | — | — | 0.18–0.25 | 0.21–0.23 | — | — | — | 0.82–0.83 | — | — | — |
| Ho et al. cross-lab [ho2024algorithmic] | (dropped) | — | — | 0.068 | 0.040 | — | — | — | 0.95 | — | — | — |

Notes on the table:
- \*Kaplan's "a" = 0.575 holds only if C = 6ND is misapplied to an early-stopped, multi-epoch law. Kaplan's headline N ∝ C^0.73 comes from L(N,S) with non-embedding N; it is a different object.
- σ ranges from 0.54 to 0.95 across setups, so σ is not universal. Within the single-epoch Chinchilla regime with E estimated, it clusters at 0.74–0.78.
- The Ho cross-lab σ* ≈ 0.95 is suggestive of Antràs-type bias toward 1 [antras2004aggregate] [H].
- Consistency checks:
  - TeX-precision Hoffmann parameters reproduce the paper's "40B at Gopher budget". Rounded parameters give 32B.
  - Chinchilla-70B/1.4T sits on the Besiroglu expansion path (w = 1.03).

### 2.2 Other anchors

- Frontier elasticity: on-path ε_N = ε_D = γ, i.e. 0.155 (Hoffmann) and 0.178 (Besiroglu). Pearce & Song quote the same L − E ∝ C^−0.155 / C^−0.178 [pearce2024reconciling].
- n-input harmonic rule γ = (Σ 1/α_i)^−1 reproduces Kaplan Eq. 1.8: 1/(1/0.76 + 1/0.21 + 1/0.076) = 0.052 vs 0.050 reported [D].
- Rounding sensitivity: β = 0.28 vs 0.2849 changes the data term by about 13% at D = 1e11 [besiroglu2024chinchilla].
- Llama 3: (α, A) = (0.53, 0.29) for optimal **tokens** gives 10.5T at 3.8e25. The figure label's (0.537, 0.299) gives 16.3T, close to the reported 16.55T. The text-rounded exponent does not reproduce the extrapolation [D] [grattafiori2024llama].
- Implied optimal model: Meta's law gives N* = 3.9e11, D* = 1.63e13, M* ≈ 42 at 3.8e25. Besiroglu gives M* ≈ 16.5; Hoffmann-TeX gives 85.
- Quantization-model restriction β = α/(1+α) [michaud2023quantization]:
  - Besiroglu: predicted 0.258 vs estimated 0.366 ± 0.02 (z ≈ 5.4, ignoring covariance) ⇒ rejected [D].
  - Hoffmann: 0.253 vs 0.285.
- Ho et al. point estimates: T_N = ln2·α_param/α_year ≈ 11.8 yr; T_D ≈ 9.2 months; T_C ≈ 8.7 months (reported 8.4 [4.5, 14.3]) [D].
- Erdil & Besiroglu point ratio: ≈ 8.1 months (reported 8.95) [D].
- Thompson et al.: 1 year of progress ≈ 10^(0.022/0.065) = 2.18× compute [thompson2020computational] [D].
- Mertens et al. 41× p90/p10 in effective compute: in reducible-loss units that is 41^γ ≈ 1.78× (Hoffmann) to 1.94× (Besiroglu). MATH L5's 186× becomes 2.25–2.54×. Syverson's manufacturing 90–10 TFP ratio is e^0.651 = 1.92 [syverson2011determines].
  - Suggestive only: it assumes loss-scaling applies to MMLU-Pro logits.
- Sahal/Wright–Moore bias of a naive frontier-release regression: γ/(1 − s_A). With s_A ∈ [0.05, 0.40] (Ho's algorithmic share), the inflation factor is 1.05–1.67 [nagy2013statistical] [D].

### 2.3 Inference wedge, reconciled across strands

The strands' numbers differ only because some used the *rounded* and some the *TeX* Hoffmann parameters. Both are shown. w = ε_N/ε_D; T/D = 3(w − 1); C/C_min is the ratio of actual training FLOPs to the training-only frontier FLOPs for the same loss (Farrell input-oriented cost inefficiency; allocative, not technical).

| Model (N, D) | M | w, Hoffmann rounded | w, Hoffmann TeX | T/D, TeX | w, Besiroglu | T/D, Besiroglu | C/C_min, Besiroglu | Local σ |
|---|---|---|---|---|---|---|---|---|
| Chinchilla 70B, 1.4T | 20 | 0.62 | 0.71 | −0.86 | 1.03 | 0.09 | 1.00 | 0.737–0.760 |
| Gopher 280B, 300B | 1.1 | 0.25 | 0.29 | −2.14 | 0.36 | −1.91 | 2.01 | ≈0.74–0.75 |
| GPT-3 175B, 300B | 1.7 | 0.30 | 0.34 | −1.99 | 0.43 | −1.72 | 1.64 | ≈0.74–0.75 |
| Llama-2 7B, 2T | 286 | 1.50 | 1.72 | 2.17 | 2.62 | 4.85 | 1.86 | 0.735–0.767 |
| Llama-2 70B, 2T | 29 | 0.69 | 0.79 | −0.63 | 1.17 | 0.52 | 1.02 | — |
| Llama-3 8B, 15T | 1,875 | 2.52 | 2.92 | 5.77 | 5.22 | 12.65 | 5.51 | 0.734–0.771 |
| Llama-3 70B, 15T | 214 | 1.21 | 1.40 | 1.20 | 2.45 | 4.36 | 1.72 | — |
| Llama-3.1 405B, 15.6T | 38.5 | 0.67 | 0.78 | −0.66 | 1.35 | 1.06 | 1.07 | — |
| Qwen1.5 0.5B, 2.4T | 4,800 | 3.88 | 4.44 | 10.33 | 7.00 | 18.0 | 9.84 | — |
| DeepSeek-V3, 37B active, 14.8T | 400 | 1.49 | 1.73 | 2.20 | 3.05 | 6.14 | 2.27 | — |

Closed form under homothetic CES [D]: T/D = 3[(M/M*)^ρ − 1] and C/C_min = cosh(½ρ ln(M/M*))^{2/ρ}. Values for Llama-3 8B / Llama-3 70B / Llama-3.1 405B, reported as T/D (C/C_min):

| | ρ = 0.30 | ρ = 0.3527 | ρ = 0.40 |
|---|---|---|---|
| M* = 16 (Porian) | 9.5 (4.8) / 3.5 / 0.9 | 13.1 (6.1) / 4.5 / 1.1 | 17.2 (7.3) / 5.5 / 1.3 |
| M* = 20 (Chinchilla) | 8.7 (4.2) / 3.1 / 0.7 | 11.9 (5.2) / 3.9 / 0.8 | 15.4 (6.2) / 4.7 / 0.9 |
| M* = 41 (Meta's own law at 3.8e25) | 6.4 (2.8) / 1.9 / −0.1 | 8.6 (3.3) / 2.4 / −0.1 | 10.8 (3.8) / 2.8 / −0.1 |
| M* = 192 (MiniCPM) | 2.9 (1.5) / 0.1 / −1.1 | 3.7 (1.6) / 0.1 / −1.3 | 4.5 (1.7) / 0.1 / −1.4 |

Meta's own law evaluated at each model's *own* compute gives M*(C) = 31.3 (8B), 36.7 (70B) and 41.9 (405B). With ρ = 0.35, the implied T/D is 9.6, 2.6 and −0.1 [D].

Caveats:
- Llama-3 8B/70B per-model token counts (about 15T) are [UNV]; only the 405B's 15.6T is stated in the text read.
- Extrapolation: Chinchilla's support tops out at M = 341. Farseer covers M up to 2,570, Gadre up to 640, Sardana up to 10,000 (not public).
- Sardana et al. report that laws fit at typical ratios *overestimate* the value of extra tokens at extreme M. If so, true ε_D is lower and w is **understated**.

---

## 3. Starting hypotheses: verdicts

| # | Hypothesis | Verdict | Refinement |
|---|---|---|---|
| H1 | Chinchilla reducible loss is CES-like; exact CES with σ = 1/(1+α) when α = β; Kaplan also CES-type | **Confirmed, refined** | With α ≠ β the technology is directly additive and **non-homothetic**, and σ varies: 1/σ = 1 + sβ + (1−s)α, s = ε_N/(ε_N+ε_D). It is not CRESH [hanoch1971cresh] and not CLM non-homothetic CES [comin2021structural]. On path s = ½, so σ* = 2/(2+α+β). Kaplan is the same additive family with *unequal inner exponents plus an outer exponent* κ = α_D that decouples scale from substitution, giving σ ∈ [0.50, 0.575]. |
| H2 | Compute-optimal = cost minimization; IsoFLOP = isocost; iso-loss = isoquant; L*(C) = inverse cost function (Nerlove) | **Confirmed (exact)** | The cost is multiplicative, so the FOC equates output elasticities, not MRTS to price ratios. Dual Allen/Morishima formulas assuming linear costs break down; use primal or log-cost-weight formulas (Shephard's lemma holds in log-cost-weight space [D]). |
| H3 | Functional dependence on the frontier ⇒ α, β not separately identified; IsoFLOPs break it | **Refined** | With κ = 1 and a observed, α = γ/a and β = γ/(1−a) *are* identified on-path, but only by functional form. Not identified on-path: σ nonparametrically; the MRTS level (A/B → M*) unless optimality is imposed; optimality itself (untestable); bias sign over time. The on-path Jacobian for the 5 Chinchilla parameters has rank 3 [D]. |
| H4 | Cross-lab data suffer Marschak–Andrews simultaneity | **Confirmed; sign is regime-dependent** | With Hicks-neutral lab ω the expansion path is ω-invariant, so bias falls on γ, not on a. Sign by lab behavior: capability-target ⇒ attenuation (Nerlove's reverse regression is right); exogenous budget ⇒ OLS consistent (ZKD); funding responds to ω ⇒ overstatement. Already flagged, without IO remedies, by [whitfill2025note; konig2026validity]. |
| H5 | Algorithmic progress = TFP growth; parameter- vs data-augmenting = factor-augmenting; DMR | **Confirmed, sharpened** | Ho et al.'s Eq. 3 is literally factor-augmenting, with α_year = α·g_N and β_year = β·g_D. Hicks-neutral ⇔ α·g_N = β·g_D, not equal rates and not an E shift. On the frontier any factor augmentation equals compute augmentation at g_N + g_D. Time series identify augmentation rates, not σ, so not the sign of the bias (1 − 1/σ)(g_N − g_D). |
| H6 | Over-training = DLW markup-like wedge | **Confirmed algebraically; reinterpreted** | w = 1 + T/(3D) = lifetime/training compute. It is **input-specific** (a Hsieh–Klenow τ_N, or the shadow price of a second use of N), not market power. It is robust to Hicks-neutral TFP and to E, but not to factor-biased recipes or to M* uncertainty. Do not call it a "markup". |
| H7 | MoE = capital stock vs services; repetition = depreciation; distillation = intermediate input; tokenizer = output measurement; accuracy = TFPR | **Mixed** | Repetition = geometric depreciation is **exact** (D′ = U + U·R*(1 − e^{−R/R*}), R* = (1−δ)/δ; R*_D = 15.4, R*_N = 5.3) [muennighoff2023scaling]. Distillation = intermediate input: close. Tokenizer = units/deflator: close. MoE: loose, since routing sparsity is structural, not utilization. Accuracy = TFPR: loose, since it is a bounded monotone transform, not price × quantity; Elo and revenue are closer to TFPR. |
| H8 | Train vs test-time compute trade-off = isoquant | **Confirmed** | Jones (Hex) is Cobb–Douglas, σ = 1, slope −1.2 [jones2021scaling]. Snell finds heterogeneity by difficulty [snell2024scaling]. Epoch already uses production language [erdil2024optimally; erdil2025train; villalobos2023trading], so novelty is partial. |

---

## 4. (a) Master dictionary

Strength: **exact** (identical mathematics), **close** (same structure, minor differences), **loose** (suggestive), **breaks-down** (analogy fails; say so in the paper). "Buys" = what the mapping delivers: a result, test, estimator or caution.

### 4.A Technology and functional form

| ML concept | IO / econometrics concept | Strength | Key citations (ML; IO) | What the mapping buys us |
|---|---|---|---|---|
| Iso-loss ("IsoLoss") contour | Isoquant | exact | [hoffmann2022training; muennighoff2023scaling]; [hicks1932theory; shephard1953cost] | Isoquants and σ are **ordinal**: invariant to loss vs BPB vs any monotone benchmark link. Allocation results transfer across output measures [brandfonbrener2024loss]. |
| Chinchilla ℓ = AN^−α + BD^−β with α = β | CES, Y = ℓ^{−1/ρ}, σ = 1/(1+ρ), CRS | exact | [muennighoff2023scaling; gadre2024language (impose α = β); hao2026theory]; [arrow1961capital; klump2007factor] | σ ≈ 0.739. CES test = Wald test of α = β (Besiroglu's α, β are within about 1 SE: [UNV] formal test). Benchmark against capital–labor σ: 0.4–0.6 [chirinko2008sigma], 0.5–0.7 [klump2007factor; oberfield2021micro], 0.45–0.64 [doraszelski2018measuring], 0.3–0.5 [raval2019micro]. |
| Chinchilla with α ≠ β | Directly additive non-homothetic technology; variable σ; CES in Ñ = N^{α/β} | close | [hoffmann2022training; besiroglu2024chinchilla]; [hanoch1971cresh; comin2021structural; sato1975most] | σ* = 2/(2+α+β) on path [D]; bounds [1/(1+max), 1/(1+min)]. The expansion path is not a ray: D*/N* ∝ C^{(α−β)/(α+β)}, rising for Hoffmann and falling for Besiroglu. |
| Kaplan L(N,D) = [(N_c/N)^{α_N/α_D} + D_c/D]^{α_D} | Generalized CES with outer exponent (scale ≠ substitution) | close | [kaplan2020scaling]; [sato1967twolevel] | Nesting family L = E + [(AN^−α)^{1/q} + (BD^−β)^{1/q}]^q (q = 1 Chinchilla; q = α_D Kaplan). Test κ(q) = 1 on unbalanced designs. Kaplan's σ is 0.50–0.575. |
| Clark routed law log L = a·logN + b·logÊ + c·logN·logÊ + d | Translog (no own-quadratic terms) | exact | [clark2022unified]; [christensen1973transcendental] | c > 0 means the routing benefit falls with size. Translog tests of separability. |
| Sloth θ = α_ik + β_k′(log s, log t, log s·log t) | Translog with firm×output FE ("efficiency"; Schmidt–Sickles frontier) | exact | [maiapolo2024sloth]; [christensen1973transcendental; aigner1977formulation] | Already borrowed by ML. Our additions are endogeneity treatment and the one-sided error. |
| Farseer: data exponent depends on N; optimal D/N rises with C | Non-separable, non-homothetic technology (translog-like interaction) | close | [li2025predictableb]; [christensen1976economies] | Rival explanation for over-training (non-homotheticity vs inference wedge). Rank-one-curvature test of Chinchilla: b_NN·b_DD = b_ND², b_ND < 0 [D]. |
| Hessian of ln ℓ in (ln N, ln D) = s(1−s)[[α², −αβ], [−αβ, β²]], rank 1, null vector (β, α) | Functional-form restriction; curvature orthogonal to expansion path | exact [D] | [io_ces_duality §2.1c] | All σ information is transverse to the path. Gives a testable translog restriction. |
| Quantization model (Zipf quanta) [michaud2023quantization]; per-problem exponentials aggregating to a power law [schaeffer2025large]; Pareto task complexity [korinek2024scenarios] | Houthakker (1955) / Jones (2005) Pareto aggregation to Cobb–Douglas | close (same math) | [michaud2023quantization; schaeffer2025large]; [houthakker1955pareto (UNV details); jones2005shape] | Cross-equation restriction β = α/(1+α), a = 1/(α+2). Rejected by Besiroglu (z ≈ 5.4) [D]. |
| Data term BD^−β ("learning within a run") | Wright learning curve / Arrow learning by doing | close | [hoffmann2022training]; [wright1936factors; arrow1962economic; levitt2013understanding; nagy2013statistical; thompson2012relationship] | β = 0.28–0.37 gives progress ratios of 78–82%, vs Wright's 80% (0.322) and LLS 0.301. Separating learning from scale needs independent variation, i.e. IsoFLOPs. |
| E + BD^−β with β = 1 | Bayesian forecast loss σ_a² + Ω^{−1} [farboodi2021model]; Amazon 1/√N + 1/√T [bajari2019impact] | close | — | LLM β ≈ 0.3 is far below the parametric rate. Bajari's time-FE confound is our scaling-vs-progress confound. |
| "Sum of inverse powers" | Jones weak-link CES (exponent −1, σ = ½) | close | [jones2026ai; jones2011intermediate] | The scarce input bottlenecks. Chinchilla is the weak-link form with exponents 0.28–0.37. |

### 4.B Cost side and duality

| ML concept | IO concept | Strength | Citations | Buys |
|---|---|---|---|---|
| IsoFLOP curve, 6ND = C | Isocost, but *multiplicative*: n + d = const, unit log-cost weights | exact (with caveat) | [hoffmann2022training]; [shephard1953cost; uzawa1964duality] | FOC ε_N = ε_D; **cost shares are known by accounting**, so no price data are needed. Linear-price dual elasticity formulas break down; use primal or log-cost-weight forms [blackorby1989real]. |
| N*(C) = G(C/6)^a, D*(C) = G^{−1}(C/6)^b | Conditional factor demands / expansion path | exact | [hoffmann2022training Eq. 4] | a = β/(α+β); G = (αA/βB)^{1/(α+β)}. |
| L*(C) = E + K(C/6)^{−γ} | Inverse cost function; C*(ℓ) = 6[K/(L−E)]^{1/γ} | exact | [hoffmann2022training; kaplan2020scaling]; [nerlove1963returns; christensen1976economies; mcfadden1978cost] | Cost elasticity of reducible loss −1/γ = −6.5 / −5.6. Nerlove's "which side is exogenous" gives reverse regressions and bounds. Christensen–Greene curvature corresponds to Hoffmann App. E frontier curvature. |
| Approach 1 / 2 / 3 | Cost function / factor demand (FOC) / primal production function; system estimation | close | [hoffmann2022training]; [klump2007factor; leonledesma2010identifying] | Cross-equation Hausman/Wald test: Hoffmann A3 fails (0.46 vs 0.49–0.50); Besiroglu passes (0.513). The normalized system estimator (KMW/LLMW) is the fix. |
| γ = αβ/(α+β) = (1/α + 1/β)^−1 | Scale elasticity along frontier (FGT's RTS fix: here estimable, not assumed) | exact in loss units | [flynn2019measuring] | On path ε_N = ε_D = γ, so index-number growth accounting is valid there. |
| ∂ln(D*/N*)/∂ln(θ_N/θ_D) = σ/(1−σ) = 2/(α+β) ≈ 2.8–3.2 | Shephard's lemma in log-cost-weight space; compensated factor-ratio response | exact [D] | [io_ces_duality §2.2] | Comparative statics of over-training with respect to inference-cost shocks. |
| "Compute is the price of N, D" | Market input prices | **breaks-down** | — | There are no market prices for N and D. Dollar costs need MFU and hardware prices (Sardana Eq. 6). |

### 4.C Identification and endogeneity

| ML concept | IO concept | Strength | Citations | Buys |
|---|---|---|---|---|
| Only compute-optimal runs; fixed-TPP designs (DataDecide D = 100N; Cerebras-GPT D = 20N; Pythia D ≈ 300B) | Functional dependence [ackerberg2015identification]; Bond–Söderbom non-identification [bond2005adjustment] | exact / close | [kricheli2026tokens] (condition number Θ(\|α−β\|^−2), about 10²–10³ at a gap of 0.06; non-collinear designs win 97.3%); [magnusson2025datadecide] | The collinearity is **behavioral**, not only a design choice: optimizing labs lie on firm-specific rays. Identification needs economically generated deviations: inference demand, data scarcity, memory/latency constraints, Kaplan-era beliefs. |
| IsoFLOP sweeps, factorial ladders (Farseer; OLMo ladder; Gadre) | Designed exogenous input-mix variation (ACF's i.i.d. price shock; stochastic adjustment costs) | close | [hoffmann2022training; li2025predictableb; bhagia2024establishing; gadre2024language] | These are experimental benchmarks for technology parameters, i.e. the "LaLonde" ground truth. |
| Seed/eval noise | Non-transmitted shock ε (ZKD; OP η) | close / exact | [madaan2024quantifying; vanderwal2025polypythias; choshen2024hitchhikers]; [zellner1966specification] | NLS within a single-lab sweep is consistent. Var(ε) is identified from seed replicates; about 4% ARE floor. |
| Family/lab recipe quality (ν_f, α_ik, ν_j) | Hicks-neutral TFP ω; Mundlak "management" | close | [ruan2024observational; maiapolo2024sloth; mertens2026secret]; [mundlak1961empirical] | Hicks-neutral ω **does not tilt the expansion path**, so (i) the N/D split cannot proxy neutral ω (the OP/LP inversion breaks down) and (ii) transmission bias loads on γ, not a. |
| Cross-lab regression of loss on C or (N, D) | Marschak–Andrews transmission bias; OP/ACF timing | close | [whitfill2025note] (sign theorem; ~9× first-pass); [konig2026validity]; [marschak1944random; olley1996dynamics] | Timing: N is predetermined ("capital"), D flexible (can be extended mid-run), post-training/distillation are intermediate. Remedies: control function, dynamic panel over family generations, IV (hardware/export shocks), forward/reverse bounds. |
| Family×generation FE (Llama-3 herd) | Mundlak/Hoch covariance estimators | close | [hoch1962estimation; griliches1998production] | Within-family variation is mostly in N at fixed D, so the D-elasticity comes from few families. FE amplifies measurement-error attenuation. |
| Compute-only regressor, C = 6ND | Aggregation restriction (equal elasticities), valid only on the expansion path; constructed input | exact | [bhagia2024establishing] ("FLOPs cannot distinguish compute-optimal from overtrained"); [mertens2026secret; ruan2024observational] | Test β_N = β_D. Never regress on C together with N and D when C was constructed from them. |
| Notable/SOTA inclusion; release-only; leaderboard self-submission; "≤3 models per paper" | Selection on outcome; OP attrition | close / loose | [epochai2026data; singh2025leaderboard; erdil2022algorithmic] | Lee/Manski bounds. Propensity correction (full Epoch vs Notable). Frontier/Large-scale subsets select on inputs, which is benign. |
| Hyperparameters (LR, batch, β2, warmup, schedule) | Flexible inputs; L(N,D) = min_h L(N,D,h) is the concentrated production function | close | [porian2024resolving; li2025predictablea; bi2024deepseek; bjorck2024scaling; lourie2026small]; [gandhi2020identification] | DeepSeek η_opt = 0.3118·C^−0.125, B_opt = 0.2920·C^0.3271 are flexible-input demands. The envelope theorem makes mis-tuning second-order, but scale-correlated mis-tuning biases a (Porian 0.835 → 0.497). |
| Mis-tuning worse at small N; laws "only emerge on the tuned frontier" | Stochastic frontier with inefficiency correlated with inputs | close | [lourie2026small; porian2024resolving]; [aigner1977formulation; meeusen1977efficiency] | SFA with u(N): scale-neutral inefficiency shifts levels only, scale-dependent inefficiency biases slopes. |
| Kaplan-era allocations (GPT-3, Gopher: w ≈ 0.25–0.43) | Optimization error / belief shock ("urge, ability, luck") | close | [kaplan2020scaling]; [marschak1944random; ackerberg2015identification] | Natural experiment: off-path variation that is independent of ω. Invalidates FOC methods for pre-2022 vintages. |
| Algorithmic R&D; learning from deployment | Controlled Markov ω = g(ω₋₁, r₋₁) + ξ; learning by exporting | close | [doraszelski2013rd; deloecker2013detecting] | Tests exogenous vs endogenous progress, the unpredictable share of Var(ω) (DJ: 25–75%), and complementarity. |
| Architecture types with different exponents | Finite-mixture technologies | close | [kasahara2023identification] | Common-exponent progress estimates are biased when types differ (LSTM vs Transformer). |
| Lab panels with explosive input growth | Blundell–Bond mean-stationarity of initial conditions | **breaks-down** | [blundell1998initial; ackerberg2023underidentification] | SYS-GMM level moments are not credible; there are multiple-root risks. Use DIF moments, parametric inversions or experiments. |
| Scaling-law "inputs" in sweeps | Inputs chosen by profit-maximizing firms | **breaks-down within sweeps** | — | IO endogeneity machinery applies to *cross-lab* data. Within-lab sweep problems are measurement, flexible inputs and misspecification. |

### 4.D Input measurement

| ML concept | IO concept | Strength | Citations | Buys |
|---|---|---|---|---|
| Non-embedding N; omitted head/attention FLOPs; 6N vs M (6N₁/M = 0.43–0.92) | Nonclassical, scale-correlated input error; Nerlove's spurious small-firm scale economies | close | [pearce2024reconciling (N_E ≈ ω·N^{1/3}, ω = 47,491 ⇒ local 0.74–0.78); porian2024resolving; bi2024deepseek]; [nerlove1963returns; christensen1976economies; collardwexler2016production] | A bias formula for the local allocation exponent. "Local Cobb–Douglas elasticities on narrow size ranges are not structural." |
| Perturbations Ñ = c·N / c + N / μ(N/μ)^s / e^δ·N | Units change / fixed component / nonclassical error / classical EIV (attenuation) | exact (algebra) | [schaeffer2025evaluating]; [griliches1986errors; hausman2001mismeasured; schennach2016recent] | Multiplicative error moves only A. Additive error moves α from 0.199 to 0.481. Noise widens CIs about 10×. |
| Epoch confidence classes (±3×, ±10×, ±31×) and a hardware-based second C (126 LMs; median log10(C_hw/6ND) = +0.21) | Known-variance measurement error; repeated indicators | close | [epochai2026data; epochai2026machine] | Reliability ratios (log10 SD about 0.29 / 0.61 / 0.91). IV with the second indicator. |
| Dollar compute; MFU (Llama 3: 38–43%) | Unobserved input-price dispersion | close | [grieco2016production; cottier2024rising] | Per-dollar productivity confounds recipe quality and compute prices. |
| MoE total vs active parameters (DeepSeek-V3: 671B / 37B) | Capital stock vs capital services / utilization | loose / close | [deepseekai2024deepseekv3; abnar2025parameters; li2025predictablea (MoE file, N_a/N 0.087–0.576)] | Column definitions are mixed in Epoch/ObsScaling (Mixtral-8x7B as total, 8x22B as active). Use active N for flows. |
| Repetition: D′ = U + U·R*(1 − e^{−R/R*}) | Geometric depreciation / perpetual inventory; organizational forgetting | exact (functional form) | [muennighoff2023scaling]; [benkard2000learning; jorgenson1967explanation] | R*_D = 15.4 (±7.3), R*_N = 5.3; effective data capped at 16.4·U. Explains DeepSeek (a = 0.578) vs Porian (≈0.50) on OWT2 [H: repetition]. |
| Transfer D_T = k·D_F^α·N^β; data quality/filtering | Quality-adjusted input services | close | [hernandez2021scaling (text→python k = 1.9e4, α = 0.18, β = 0.38); goyal2024scaling]; [jorgenson1967explanation] | Quality-adjusted tokens absorb "data-augmenting progress". |
| Precision: N_eff = N(1 − e^{−P/γ}) | Input quality with cost ∝ quantity × quality | close | [kumar2024scaling (P* ≈ 7–8 bits, independent of C)] | A homothetic quality choice. |
| Distillation teacher; synthetic data; pruning | Intermediate input; gross output vs value added | close | [busbridge2025distillation (δ^Pre toggle = make-or-buy); ruan2024observational (Phi outlier)]; [gandhi2020identification] | Compute-only "TFP" is overstated for distilled students. Mertens' Microsoft and Nvidia effects of about 60× [H]. |

### 4.E Output measurement

| ML concept | IO concept | Strength | Citations | Buys |
|---|---|---|---|---|
| Per-token loss depends on tokenizer and eval set; BPB | Output in firm-specific units; common-deflator bias | close | [tao2024scaling; gao2020pile; hu2024minicpm]; [klette1996inconsistency] | BPB = (L_T/L_B)·ℓ/ln 2. At 0.29335 GPT-2 tokens/byte on the Pile, "20 tokens/param" ≈ 68 bytes/param [D]. D/N, E, A, B and G are not unit-free; α and β are. |
| Irreducible loss E | Location parameter / output cardinalization | loose | [henighan2020scaling; kaplan2020scaling] | Kaplan 0.076 vs Chinchilla 0.34 is mostly the choice of E (the Chinchilla total-loss slope at N = 1e9 is 0.047 [D]). An E shift is not Hicks-neutral TFP. |
| Accuracy (sigmoid/logit of loss); "emergence" | Bounded nonlinear transform of latent output; TFPR vs TFPQ | close / loose | [schaeffer2023emergent (>92% of emergent abilities under 2 metrics); schaeffer2024predicting; lourie2025scaling (18/46 tasks predictable); wei2022emergent]; [foster2008reallocation] | Build technology on BPB (TFPQ). Treat accuracy as a measurement map. Thresholds are artifacts. |
| logit(score) = a + b·log C + FE | Cobb–Douglas in odds; FE = Hicks-neutral TFP on the odds scale; M = 10^{ν/β} = TFP in input units | exact (algebra) | [mertens2026secret; ruan2024observational; ho2025rosetta] | Converts "secret sauce" into TFP. Constant only under homotheticity. |
| PCA/IRT capability (3 PCs ≈ 97%; ECI) | Latent output from multiple noisy indicators; index base | close | [ruan2024observational; burnell2023revealing; ho2025rosetta; epochai2026capabilities]; [hu2020estimating] | Two conditionally independent evals deconvolve ω from ε. |
| Loss-to-loss shifted power law L1 = K(L0 − E0)^κ + E1 | Monotone relabeling of output; isoquants preserved | close | [brandfonbrener2024loss; mayilvahanan2025llms] | Compute-optimal N is invariant across validation sets. |
| Contamination; training on the test task | Output error correlated with vintage/firm; equal fine-tuning as a deflator | close | [dominguezolmedo2025training (post-Nov-2023 +7 MMLU / +19 GSM8K pts, vanishing after equal fine-tuning); zhang2024careful (GSM1k up to −8 pts)] | Period effects ("shared progress") are partly benchmark targeting. |
| Elo / Arena; API revenue | TFPR / demand-side output | close | [chiang2024chatbot; lmarena2026leaderboard]; [bond2020unpleasant; deloecker2011product] | Elasticities of Elo are not technology parameters. Never compute wedges from them. |
| Time horizon at 50% success | Cardinal output in labor-time units | loose | [kwa2025measuring] (doubling 207 days [166, 240]) | An unbounded output that avoids ceiling compression. |
| "Diminishing returns to compute" / RTS in loss | Returns to scale | **breaks-down** | — | Not invariant to cardinalization. With Y = 1/ℓ, RTS ≈ 0.28–0.37; with Y = ℓ^{−1/ρ}, CRS. Only isoquants and the loss-denominated cost function are invariant. |

### 4.F Productivity, technical change, frontiers

| ML concept | IO concept | Strength | Citations | Buys |
|---|---|---|---|---|
| Algorithmic progress (effective-compute doubling) | TFP growth in input units (Solow residual) | close | [ho2024algorithmic (8.4 mo [4.5, 14.3]); erdil2022algorithmic (≈9 mo); hernandez2020measuring (16 mo, 44×); ho2025rosetta (≈6×/yr); mertens2026secret (7.5× over ~1.5–2 yr); gundlach2025price (≈3×/yr dual)]; [solow1957technical; hulten2001total] | Rates span about 2–9×/yr depending on output measure, FE, reference and scale. The residual depends on the maintained law. |
| N_eff = N·e^{α′t}, D_eff = D·e^{β′t} | Factor-augmenting technical change | exact | [ho2024algorithmic; erdil2022algorithmic]; [doraszelski2018measuring; demirer2020production] | α_year = α·g_N, β_year = β·g_D. On the frontier this equals compute augmentation at g_N + g_D [D]. |
| α_year, β_year individually n.s., opposite signs, identified sum | Diamond–McFadden–Rodriguez impossibility | close | [ho2024algorithmic]; [diamond1978measurement; leonledesma2010identifying] | Identify bias from *cross-sectional* IsoFLOPs at each date (Raval-style local variation). |
| Hicks-neutral progress (Ho "Model 12") | Proportional shift of A and B: α·g_N = β·g_D | exact [D] | [ho2024algorithmic] | One-line neutrality test. An E shift is an output-measurement-type change, not TFP. |
| Data quality changes a (DeepSeek 0.450 → 0.524 → 0.578) | Curvature-changing (non-augmenting) technical change | close | [bi2024deepseek] | Factor augmentation cannot move a in the Chinchilla family [D]. Testable on DataDecide (25 recipes) and open-sci-ref (8 corpora). |
| Scale-dependent CEG (LSTM → Transformer 6.28× small vs ~725–846× frontier); reference dependence | Non-neutral / scale-biased change; index-number problem | close | [gundlach2025origin; sanderson2025rethinking]; [diewert1976exact] | Constant CEG ⇔ equal exponents and E [D]. Use superlative (Törnqvist/Fisher) progress indexes. |
| Kaplan → Chinchilla re-balancing counted as progress (≈10× of 6,930×; Ho blog 2–4×) | **Farrell allocative efficiency** | exact (reclassification) | [gundlach2025origin; ho2024algorithmic]; [farrell1957measurement] | A Farrell decomposition of "algorithmic progress" into technical and allocative parts is new. |
| Shapley decomposition (compute 60–95%, algorithms 5–40%) | Solow growth accounting (Solow residual share 87.5%); Hall bias | close | [ho2024algorithmic]; [solow1957technical; hall1988relation] | Cost-share weighting with over-trained models biases the residual by (ε_N − ε_D)(Δn − Δd)/2 [D]. |
| Densing law ρ_max doubling every 3.3 mo (arXiv) / 3.5 mo (NMI) | Single-factor (parameter) productivity | exact | [xiao2025densing] | The reference fixes D = 1T, so over-training raises "density" mechanically (capital deepening). |
| Frontier statistics (ρ_max; min compute to a threshold; highest-b models; Approach-1 envelope) | DEA/FDH frontier; order statistics rise with entry; winner's curse | close | [xiao2025densing; mertens2026secret; ho2025rosetta; hoffmann2022training]; [farrell1957measurement] | Correct for the number of entrants; use SFA or quantile frontiers. |
| Quantile capability boundary (τ ≈ 0.98) | Quantile / order-m frontier | exact | [zhang2026prescriptive]; [aragon2005nonparametric] | — |
| Family efficiency dispersion (41× p90/p10 in effective compute) | Syverson within-industry TFP dispersion (1.92) | close (suggestive) | [mertens2026secret]; [syverson2011determines; syverson2004product; hsieh2009misallocation] | ≈1.8–1.9× in reducible-loss units [D]. |
| Hardware vs software progress | Embodied vs disembodied technical change | exact | [erdil2025gate]; [jones2024framework] | — |

### 4.G Wedges, inference and markets

| ML concept | IO concept | Strength | Citations | Buys |
|---|---|---|---|---|
| Over-training: w = ε_N/ε_D = 1 + T/(3D) | DLW θ/α (loose: no revenue share); Hsieh–Klenow τ_N (close); Farrell allocative inefficiency (close) | close | [sardana2024beyond; roberts2026test; bian2025scaling]; [deloecker2012markups; hsieh2009misallocation; raval2023testing] | **Revealed inference demand** T̂ = 3D(ŵ − 1). Robust to Hicks-neutral TFP and E. Raval over-identification within families. |
| Under-training (w < 1) | Belief-driven wedge (optimizing against Kaplan's law) | loose | [kaplan2020scaling] | Not a "negative markup". |
| Train vs test-time compute | Cobb–Douglas isoquant (σ = 1, slope −1.2); expenditure-share rule α/(α+β) | close | [jones2021scaling; erdil2024optimally; villalobos2023trading; snell2024scaling; brown2024large] | σ between training and inference compute, tied to w. |
| Fixed training cost + low marginal inference cost; nonrival weights | Scale economies from fixed cost (not technological RTS) | close | [korinek2025concentrating; varian2019artificial]; [jones2020nonrivalry] | "Economies of scale in AI" is a cost-structure claim. Read as technological IRS it breaks down (loss ∝ C^−0.15 to C^−0.18). |
| Convex cost of capability C ∝ (ℓ−E)^{−1/γ} + demand for intelligence | Sutton endogenous sunk costs (concentration bounds) | close | [sutton1991sunk]; [demirer2025emerging] | Unused so far (econ_ai grep). |
| Price of fixed capability falling 5–10×/yr (≈3×/yr algorithmic); GPT-4-class ≈1000× in 2 yr | Dual (price-based) TFP; P = MC × markup | close | [gundlach2025price; demirer2025emerging; cottier2025llm]; [hsieh2002explains] | Primal minus dual gives markup change plus pass-through. |
| Inference API pricing | DLEU markups with inference compute as the flexible input | direct application | [deloecker2020rise] | Extension. |

---

## 5. (b) State of knowledge: what ML knows about each IO problem, under which name, and what it does not

Format for each problem: **Known** (ML name → key facts and numbers), **Not known**, **Implication**.

### 5.1 Functional dependence / collinearity along the expansion path
- **Known.**
  - Hoffmann: "C is a deterministic function FLOPs(N,D)".
  - [kricheli2026tokens]: fixed tokens-per-parameter designs give condition number Θ(|α−β|^−2), about 10²–10³ at a gap of 0.06. Scale coefficients are "practically unidentifiable" and CIs inflate 10× or more. Non-collinear designs win 97.3% of held-out comparisons. They give a closed-form diversity threshold.
  - Czech et al.: the 5-parameter Hessian has condition number ≈3.5×10^11, about 11 after variable projection.
  - Choshen: across families, 3 PCs explain 99.49% of the variance of (E, A, α, B, β).
  - Besiroglu: SE(A) = 125 and SE(B) = 1,293 vs SE(α, β) = 0.02, i.e. the A/B split is weak.
  - Designs with built-in collinearity: DataDecide final checkpoints (D = 100N), Pythia (D fixed), ObsScaling (FLOPs = 6ND, 100%), Epoch (53% of language models within ±12% of 6ND·epochs).
- **Not known.**
  - That the ray is *behavioral* (optimizing labs), as in ACF.
  - What on-path data identify (γ, a; α and β only under κ = 1; never σ nonparametrically; the MRTS level only if optimality is imposed).
  - That observational deviations (inference demand, data scarcity, beliefs) are the analog of ACF's input-price shocks, and what exclusion restrictions they need.
- **Implication.** Proposition on identification (P2), plus a Monte Carlo (on-path vs IsoFLOP vs factorial). This underpins H1 and H2.

### 5.2 Simultaneity / transmission bias (Marschak–Andrews)
- **Known.**
  - [whitfill2025note] Theorem 1: sign(bias(β̂_year/β̂)) = −sign(Cov(ln D, ε_D)). A Monte Carlo on Ho's data turns true 45%/yr into 16.5%/yr (corr +0.5) or 93%/yr (corr −0.5). A "first-pass" reading implies progress overstated about 9×.
  - [konig2026validity]: calendar time is a confounder; exchangeability is untestable.
  - Ho et al. acknowledge that algorithmic improvements and scaling were "introduced concurrently".
  - Rosetta uses same-recipe families to identify k (0.168; Llama 3.1 alone 0.12, since D is fixed at 15T).
- **Not known.**
  - The behavioral structure (timing: compute sized after small-scale ablations reveal ω).
  - Proxy/control-function, dynamic-panel or IV estimators; bias signs in the Mertens and Ruan FE regressions; bounds.
  - That Hicks-neutral ω biases only γ, not a [D].
  - The regime-dependence of the sign (target / budget / funding) [D].
- **Implication.** Contribution 3 (LaLonde benchmark) and P5.

### 5.3 Selection and attrition
- **Known.**
  - Epoch notability includes SOTA, >1,000 citations, >1M users and historical significance: partly selection on Y.
  - Erdil & Besiroglu keep ≤3 top models per paper.
  - Thompson et al.: only 91–114 of 785 models report compute.
  - "Leaderboard illusion" [singh2025leaderboard].
  - Closed labs rarely disclose N and D: since 2023, 21 API-only language models above 1e21 FLOP have N, D and C, vs 246 open-weights models.
  - Frontier statistics are order statistics.
  - Muennighoff drops runs where more epochs *raise* loss (truncation on Y).
- **Not known.** Any correction (OP propensity, Heckman, Lee/Manski); the sign of the bias on exponents. One conjecture: small models are released only when unusually good, so E[ω | N, released] falls in N and exponents are *understated* [H].
- **Implication.** Robustness module in contribution 3. Frontier/Large-scale subsets (selection on inputs) as the benign comparison.

### 5.4 Unobserved TFP heterogeneity across labs/families
- **Known.**
  - Ruan: S = θ_f·log C + ν_f, with family-specific slopes and intercepts ("families only vary in efficiency").
  - Sloth: family×skill intercepts called "efficiency".
  - Mertens: developer FE from DeepSeek 2.3× to Microsoft 60.5× (Nvidia 61.6× [D]); within-developer p90/p10 = 41× (186× on MATH L5); period×compute interactions insignificant (neutral time effects not rejected).
  - Optimal D/N varies across labs: ~20 (Chinchilla), 14–16 (Porian), ~192 (MiniCPM), rising with C (Farseer), ~42 at 3.8e25 (Meta's law).
  - DeepSeek: a depends on data quality.
  - Hestness: architectures shift intercepts, not exponents.
- **Not known.**
  - Endogeneity-corrected TFP; dispersion in comparable units (TFPQ vs TFPR).
  - Whether heterogeneity is neutral (intercept), factor-augmenting (A vs B) or curvature-changing (α, β).
  - Intermediate-input corrections (teacher compute).
- **Implication.** Contribution 6, and the DataDecide neutrality test in contribution 4.

### 5.5 Flexible inputs (hyperparameters) and the concentrated production function
- **Known.**
  - Porian: a moves 0.835 → 0.706 (head FLOPs) → 0.602 (warmup) → 0.571 (cosine) → 0.497 (tuned LR, batch, β2). Porian's laws: LR = 3.7·N^−0.36; batch = 0.00037·N^0.703, in 2,048-token sequences.
  - DeepSeek η_opt(C), B_opt(C).
  - Step Law: η = 1.79·N^−0.713·D^0.307 and B = 0.58·D^0.571 (3,700 runs; convex landscape; 0.094% from the grid optimum).
  - Bjorck: LR* ∝ D^−0.32. **The sign disagrees with Step Law** (likely because Step Law co-scales batch size [UNV]).
  - Lourie 2026: with 4 configurations per scale no law appears; with 256 the law is recovered.
- **Not known.**
  - A formal envelope/SFA treatment; the bias formula for a from scale-dependent inefficiency ∂u/∂ln N.
  - A GNR-style use of flexible-input FOCs.
- **Implication.** Contribution 5 (Step Law 1,911 dense runs = 17 (N,D) cells × LR×BS grid).

### 5.6 Input measurement error
- **Known.** Pearce–Song (non-embedding N: local 0.74–0.78 vs global 0.51); Porian (head FLOPs); DeepSeek (6N vs M off by up to 50%); Schaeffer 2025 (three N definitions differing up to 15.2%, with the perturbation taxonomy); Epoch confidence classes; hardware-vs-6ND measures; MoE mixing; dataset size ≠ tokens seen.
- **Not known.**
  - An IO framing: Nerlove's spurious scale economies; Collard-Wexler–De Loecker-style instruments.
  - EIV corrections in cross-lab data.
  - How much of Ho's β_data = 0.04 (vs experimental 0.28–0.37) is attenuation vs simultaneity.
- **Implication.** A measurement section plus the EIV module of contribution 3.

### 5.7 Output measurement
- **Known.**
  - Tokenizer dependence (Kaplan says the constants "depend on vocabulary"); BPB and unigram-normalized loss [tao2024scaling].
  - Emergence as a metric artifact; only 39% of downstream tasks predictable; single-scale ranking beats all 8 scaling-law methods in DataDecide.
  - Seed SD of 0.57 pp on MMLU.
  - Continuous metrics have higher SNR (ARC-C 45.9 vs 381.6).
  - Prompt format moves scores up to 76 pts.
  - Contamination (GSM1k); training on the test task.
  - ECI/IRT latent indices; loss-to-loss invariance of compute-optimal N.
- **Not known.** The ordinal/cardinal distinction (σ invariant, RTS not); a TFPQ/TFPR vocabulary; the implication that wedge and TFP calculations must use quantity-like output (BPB), never Elo or accuracy [bond2020unpleasant].
- **Implication.** A short output section. H1 is invariant to it; H2 must use BPB-based technology.

### 5.8 Functional form, separability, homotheticity, σ
- **Known.**
  - [hao2026theory] derive σ = 1/(1+α) for α = β and σ ≈ 0.762 for Chinchilla, then *replace* Chinchilla with Leontief, which rules out over-training.
  - Gadre and Muennighoff impose α = β.
  - Farseer rejects separability (N-dependent data exponent; optimal D/N rising in C). Abnar's held-out R² is 68%. Sardana's refits move (α, β) from (0.08, 0.13) to (0.18, 0.24) with the M support.
  - Hoffmann App. E: the frontier is curved.
  - Kricheli: single-ray designs.
  - A 2022 LessWrong post reads Chinchilla as near-Leontief [anonymous2022ai].
- **Not known.**
  - σ* = 2/(2+α+β) on path; the Kaplan contrast (0.50–0.575).
  - σ estimates with SEs across datasets; formal CES / translog / rank-one tests; a nonparametric local σ on orthogonal designs (Farseer).
  - Normalized-CES reparameterization (only VPNLS exists).
- **Implication.** Headline H1.

### 5.9 Cost-function duality and FOC-based identification
- **Known.** Chinchilla's three approaches, but not their dual meaning. Approach 2 bias for α ≠ β or off-center grids: 0.3% (±2×) to 4.1% (±16×), up to 23% for asymmetric surfaces, intercept only; for Llama 3, 6.5% of a 3.8e25 budget misallocated, ≈ $1.4M [$412K–$2.9M] [czech2026problems]. Besiroglu shows A3's ~70 tok/param is inconsistent with Chinchilla's own 20.
- **Not known.**
  - The primal–dual reading: A1 = cost function, A2 = conditional factor demand, A3 = primal.
  - The cross-equation Hausman test and system estimation.
  - The revealed-preference test: Chinchilla-70B must have w ≈ 1. It gives 0.62–0.71 under Hoffmann and 1.03 under Besiroglu, independently rejecting Hoffmann's MRTS.
  - GNR share identification without prices.
- **Implication.** P1, plus the system estimator in H1.

### 5.10 Technical change: neutral vs biased; DMR; index numbers
- **Known.** Ho et al.'s factor-augmenting model with a "Hicks-neutral" variant; Erdil–Besiroglu (compute-augmenting ≈101%/yr vs data ≈38%/yr n.s.; MAP priors because data parameters are "poorly identified"); Gundlach (scale-dependent, reference-dependent; <100× verifiable at small scale vs 22,000× claimed); Sanderson (compute-dependent vs independent innovations); Bajari et al. (data-size effects vanish with time FE).
- **Not known.**
  - DMR as the reason α_year and β_year are unidentified.
  - Hicks neutrality ⇔ α·g_N = β·g_D; factor augmentation leaves a unchanged; constant CEG ⇔ common exponents and E.
  - Allocative vs technical decomposition; superlative indexes.
  - The Sahal bias γ/(1 − s_A).
  - Directed technical change as data becomes scarce [acemoglu2002directed].
- **Implication.** Contribution 4.

### 5.11 Returns to scale and cardinalization
- **Known.** "Economies of scale" in AI IO means fixed costs [korinek2025concentrating; varian2019artificial; gans2024market]. Farboodi–Koh–Xia flag loss → productivity as a "(substantial!)" assumption.
- **Not known.** That RTS in loss units is not identified without a cardinal output; that the technological returns to compute are sharply *decreasing* (ℓ ∝ C^−0.155 to C^−0.178; C ∝ ℓ^{−6.5}); the Sutton reading.
- **Implication.** Discussion section; the link-function problem.

### 5.12 Estimation and inference practice
- **Known.**
  - Hoffmann's optimizer failure: averaged Huber, early L-BFGS stop, non-reconverged bootstrap; the "80% × 100" resampling is subsampling.
  - Besiroglu's 4,000-draw bootstrap. The (Mis)Fitting survey: 23 of 51 papers omit the optimizer and 15 omit FLOP/parameter counting.
  - Ivgi's hierarchical bootstrap (scales, then seeds); Porian's noise-and-interpolate bootstrap; delta-method CIs (Nezhurina); VPNLS (Czech).
  - The Huber δ = 1e−3 ≈ LAD point; log vs levels noted by Czech; Alabdulmohsin's extrapolation-loss criterion.
- **Not known.**
  - PPML / Gamma-QMLE in levels (log-of-gravity) [santossilva2006log; duan1983smearing].
  - Weak-ID-robust inference [andrews2012estimation]; boundary tests for E = 0 [andrews2001testing].
  - Cluster-by-run inference for checkpoints and WSD branches; joint inference for σ, M* and w.
  - D-optimal experimental design against compute budgets (Kricheli has an appendix).
- **Implication.** Estimation appendix and horse race (inside H1).

### 5.13 Frontiers and efficiency
- **Known.** Thompson's 10th-percentile quantile regression = best-practice frontier (log10 error on log10 compute −0.084); Zhang et al. quantile capability boundaries; Lourie 2026 "laws only on the tuned frontier"; Approach 1 = lower envelope.
- **Not known.** SFA with composed error; a winner's-curse correction for envelopes and argmins (seed noise ≈0.02 nats); Farrell allocative vs technical decomposition of over-training (C/C_min up to 5.5× for Llama-3-8B under Besiroglu, allocative by design).
- **Implication.** Contributions 5 and 2.

### 5.14 Intermediate inputs (gross output vs value added)
- **Known.** Distillation scaling laws (the student benefits when the teacher exists and is shared; a capacity gap exists); ObsScaling attributes Phi's efficiency to uncounted synthetic-data FLOPs; Densing finds distilled models *less* dense, while Mertens finds distillation-heavy developers *most* efficient. The conflict comes from the input definition (parameters vs 6ND).
- **Not known.** Any gross-output production function with teacher compute; the GNR logic.
- **Implication.** Contribution 6.

### 5.15 Demand side: inference demand, prices and markups
- **Known.**
  - Forward problem only [sardana2024beyond; bian2025scaling; roberts2026test; devries2023go].
  - OpenRouter: 100T+ tokens in 2025; provider-level price elasticity ≈ −1.1 (no short-run Jevons); hedonic slope 0.039–0.047 log price per intelligence point; open models ≈87% cheaper [demirer2025emerging].
  - Price per capability falling 9–900×/yr by benchmark [cottier2025llm].
- **Not known.** The inversion: revealed T from (N, D). Validation of revealed T against usage and prices.
- **Implication.** Headline H2.

### 5.16 Aggregation and micro-foundations
- **Known.** Quanta (Zipf) → power law; monkeys (exponential per problem, heavy-tailed difficulty → power law; exponent = tail index); Korinek–Suh Pareto tasks; the Paquette "4+3 phases" universal Chinchilla d* ∝ f^{1/2}; Sharma–Kaplan α ≈ 4/d.
- **Not known.** The Houthakker/Jones identity; cross-equation restrictions as tests (β = α/(1+α) is rejected by Besiroglu's refit).
- **Implication.** A short theory subsection or appendix (optional).

---

## 6. (c) Novelty verdicts per planned contribution (from the novelty strand, extended)

| # | Contribution | Verdict | Closest prior work | What remains ours |
|---|---|---|---|---|
| C1 | Systematic ML↔IO econometrics dictionary | Partially done piecemeal; **systematic version NOVEL** | [hao2026theory] (Leontief, σ, MRTS, Mussa–Rosen); [ho2024algorithmic] ("Hicks-neutral" model label); [erdil2022algorithmic]; [whitfill2025note]; [mertens2026secret]; [konig2026validity]; [erdil2025train] | Marschak–Andrews, OP/LP/ACF, GNR, DLW, DMR, Nerlove duality, SFA/Farrell, FHS TFPQ/TFPR, Klette–Griliches, KMW normalization. Each has at most a one-sentence precursor. |
| C2 | CES / σ reading | **DONE (derivation)** [hao2026theory]; informal [anonymous2022ai] | Hao & Merrill App. B: σ(n,d), 1/(1+α) if α = β, ≈0.762 | σ* = 2/(2+α+β) on path; the Kaplan contrast (0.50–0.575); **σ estimates with SEs across datasets**; CES/translog/rank-one tests; ordinal vs cardinal; σ > 0 is what rationalizes over-training (Hao & Merrill's Leontief rules it out). |
| C3 | ACF functional dependence; IsoFLOP as experiment | **Statistics DONE** [kricheli2026tokens]; related estimator critiques [besiroglu2024chinchilla; czech2026problems] | Kricheli's Θ(ε^−2) conditioning; D-optimal appendix | *Behavioral* source (optimizing labs); what on-path data identify (γ, a, not σ or M*); A1/A2/A3 = cost / factor demand / primal; GNR route; economic deviations as instruments. |
| C4 | Simultaneity / selection in observational scaling | **Partially done** [whitfill2025note; konig2026validity; dominguezolmedo2025training] | Whitfill proposes RCTs or unspecified IVs; Konig uses a Rubin framing | Marschak–Andrews/OP structure and timing; control function, dynamic panel, IV; bias signs; bounds; a LaLonde benchmark (not searched explicitly: re-check). |
| C5 | Lab TFP dispersion | **Partially done** [mertens2026secret; ruan2024observational] | OLS + FE on 809 models; no IO | Endogeneity- and selection-corrected TFP; TFPQ vs TFPR; gross-output correction (teacher compute); Syverson comparison. |
| C6 | Algorithmic progress = factor-augmenting TFP growth; neutral vs biased; DMR | **Partially done (substance)** [ho2024algorithmic; erdil2022algorithmic; gundlach2025origin; sanderson2025rethinking] | No biased-technical-change literature cited | DMR; neutrality ⇔ α·g_N = β·g_D; constant CEG ⇔ equal exponents and E; **Kaplan→Chinchilla = allocative efficiency (Farrell)**; index numbers; selection-corrected rates. |
| C7 | Revealed inference demand (DLW/GNR duality) | **NOVEL** (only forward problems exist) | [sardana2024beyond; bian2025scaling; roberts2026test; devries2023go]; [hao2026theory] (Leontief, so no wedge) | Inversion T = 3D(w − 1); homothetic closed form T/D = 3[(M/M*)^ρ − 1]; within-family flagship calibration; Raval over-ID; validation against usage and prices. |
| C8 | IO estimators on scaling data | **Partially done (simple tools only)**: OLS+FE [mertens2026secret]; FOC-CES on *R&D* inputs [whitfill2025compute]; quantile frontier [zhang2026prescriptive]; DEA/Malmquist on inference prices [du2026tiered] | — | OP/LP/ACF, GNR, dynamic panel, SFA, DLW and Nerlove cost functions on *training* data: none found. |
| C9 | Train vs test-time compute isoquant | **Partially done** [jones2021scaling; villalobos2023trading; erdil2024optimally; erdil2025train; snell2024scaling] | Epoch uses production language | Estimate σ(train, test) with SEs; tie it to the wedge. |
| C10 | MoE capital services; repetition = depreciation; distillation = intermediate input; SFA/Farrell | Mixed: the depreciation *form* is exact in [muennighoff2023scaling]; SFA partial [lourie2026small; zhang2026prescriptive]; MoE and distillation mappings novel as economics | [clark2022unified; krajewski2024scaling; busbridge2025distillation] | Naming plus econometric consequences. |
| C11 (new) | Duality test / system estimator reconciling Hoffmann vs Besiroglu | **NOVEL** (no duality/Nerlove/Shephard + compute-optimal hits) | [besiroglu2024chinchilla; czech2026problems] | Cross-equation Hausman test; KMW normalized system; the revealed-preference test at Chinchilla-70B. |
| C12 (new) | Log-of-gravity / PPML, LAD and weak-ID inference for scaling laws | **Likely novel** (no ML paper compares level vs log estimators; Czech notes misspecification) | [czech2026problems; li2025misfitting] | Estimator horse race; weak-ID-robust CIs. |
| C13 (new) | Sutton endogenous-sunk-cost reading of frontier-AI concentration | **Novel** (econ_ai grep of Korinek–Vipra, Gans, Varian and Demirer et al.: no Sutton) | [korinek2025concentrating; demirer2025emerging] | Theory extension, not core. |

**Claims to avoid** (novelty strand):
- "First to call scaling laws production functions" [hao2026theory; erdil2025train; anonymous2022ai].
- "First to compute σ" [hao2026theory].
- "First to note endogeneity or selection" [whitfill2025note; konig2026validity].
- "First to measure developer heterogeneity" [mertens2026secret; ruan2024observational].
- "First to model factor-augmenting progress" [ho2024algorithmic; erdil2022algorithmic].
- "First to show collinearity kills identification" [kricheli2026tokens].

**Most dangerous overlaps:**
- Mertens et al.: an IO economist is on that team; a follow-up applying ACF or DLW to their 809 models is plausible. Frame our work as complementary and consider contacting them.
- Hao & Merrill (the σ derivation).
- Kricheli et al. (conditioning statistics).

**Draft positioning sentence:** see novelty strand §1. Use it nearly verbatim in the introduction.

---

## 7. (d) Candidate contributions, ranked

Ranking criteria: (i) novelty; (ii) economic content for AER readers; (iii) value to ML practice; (iv) feasibility on one Apple M5 Max with public data; (v) risk. "Laptop feasibility" assumes all listed data are small (every file below is <100 MB), and NLS, GMM, bootstrap and SFA are trivial. Downloads require user approval under the session's safety policy.

### Rank 1 — Over-training reveals anticipated inference demand: the DLW/GNR duality for LLM training (headline H2)
- **Claim.** Labs that minimize lifetime compute choose (N, D) so that w = ε_N/ε_D = 1 + T/(3D) = lifetime/training compute. Given an experimentally identified technology, observed (N, D) reveal each model's planned inference volume T.
  - Under homothetic CES the statistic is (M, M*, ρ): T/D = 3[(M/M*)^ρ − 1], and C/C_min = cosh(½ρ ln(M/M*))^{2/ρ}.
  - Illustration: Llama-3-8B planned for roughly 4–13× its training tokens in lifetime inference (T ≈ 70–190T tokens). The range comes from the parameter set; Meta's own law gives about 9.6×.
  - Meta's 405B flagship sits on its own expansion path (w ≈ 0.97).
  - Pre-2022 models (GPT-3, Gopher) have w ≈ 0.25–0.43, revealing Kaplan-era beliefs rather than "negative demand".
- **Why economists care.**
  - A new revealed-preference object: firms' *ex ante* expectations of demand, recovered from production choices, as DLW recover markups.
  - It connects production to the emerging demand evidence on the inference market [demirer2025emerging].
  - The IO critiques transfer: the wedge is only as good as θ̂ [raval2023testing; bond2020unpleasant]; the input-specific wedge is a Hsieh–Klenow τ.
  - It disciplines AI growth models that calibrate train–inference trade-offs [erdil2025gate; cunningham2026economics].
- **Why ML researchers care.** It quantifies how "inference-optimal" released families are. It gives a consistency check on scaling-law estimates: Hoffmann's A3 fails it at Chinchilla-70B (w = 0.62–0.71), Besiroglu's passes (1.03). And it shows that M*, not the exponents, drives inference-aware sizing.
- **Formal results.**
  - (P6) the FOC inversion; invariance to Hicks-neutral ω and E (DLW eq. 5 logic);
  - the closed forms above;
  - a partial-identification proposition: bounds on T when (A/B, α, β) lie in a confidence set and M lies outside the design support (Demirer: elasticities are identified only on the support of choices);
  - GNR direction: within a family sharing a recipe, assume the flagship is training-optimal, which pins M*(C) through the family's own IsoFLOP law, then back out the siblings' T;
  - Raval-style over-identification: all siblings must be consistent with one demand model, e.g. T ∝ expected users × tokens/user, with memory/latency constraints as rival wedges;
  - non-homotheticity (Farseer) as a rival: compute w relative to the non-homothetic M*(C) at each model's own C.
- **Estimators.** Technology from designed sweeps (NLS/VPNLS/PPML with cluster bootstrap), then plug-in w with delta-method or bootstrap bands; a GMM system stacking family FOCs.
- **Data.**
  - Technology:
    - Farseer grid, `1222_full.csv` (404 runs, M 0.31–2,570, which covers Llama-3-8B's M = 1,875): https://github.com/Farseer-Scaling-Law/Farseer
    - Gadre testbed (M 5–640, 3 corpora): https://github.com/mlfoundations/scaling
    - Epoch Chinchilla extraction: https://github.com/epoch-research/analyzing-chinchilla
    - Llama 3 IsoFLOP digitization (Meta's own technology): https://github.com/eric-czech/llama3_isoflop_extraction and https://huggingface.co/datasets/open-athena/isoflop-experiments (814 rows)
    - OLMo ladder (AI2's own technology, 0.5–10× Chinchilla): https://github.com/allenai/OLMo-ladder
  - Choices (N, D):
    - Epoch all-models: https://epoch.ai/data/all_ai_models.csv
    - ObsScaling: https://github.com/ryoungj/ObsScaling
    - Sloth: https://github.com/felipemaiapolo/sloth
    - HF safetensors counts (exact N): https://huggingface.co/api/models/{id}
  - Validation:
    - HF downloads (same API);
    - OpenRouter prices and provider counts: https://openrouter.ai/api/v1/models, with Wayback history;
    - LMArena votes: https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset;
    - Demirer et al. volumes (NBER w34608; data access UNV).
- **Feasibility.** High. Everything is <10 MB except the LMArena parquet (1.14M rows, fine). Optional self-trained over-training sweep on the M5 Max at 1–30M parameters, M 5–2,000, to verify the closed form and the extrapolation direction (§8.4).
- **Risk.**
  - (i) Cross-lab transfer of technology (data mix, tokenizer): mitigate with lab-own sweeps (Meta, AI2, StepFun) and BPB units.
  - (ii) Extrapolation beyond the support; Sardana says w is understated.
  - (iii) Rival wedges: data scarcity (a Muennighoff shadow price pushes the *other* way), memory and latency, distillation targets, marketing.
  - (iv) Per-model token counts for Llama-3 8B/70B [UNV].
  - (v) Validation data are coarse; downloads ≠ tokens.
- **Novelty.** ★★★★★ (C7 novel).

### Rank 2 — What scaling-law data identify, and the parameter–data elasticity of substitution (headline H1)
- **Claim.**
  - (a) Duality: Chinchilla's A1/A2/A3 are cost-function, factor-demand and primal estimators, with testable cross-equation restrictions.
  - (b) Functional dependence: on-path data identify (γ, a) and, under κ = 1, α and β by functional form. They never identify σ nonparametrically or the MRTS level M* without assuming optimality. The rank-one Hessian's null direction is the expansion path.
  - (c) Estimates: σ ≈ 0.74 (gross complements) is stable across independent sweeps and estimators, while a and M* are fragile. CES (α = β) and Chinchilla's rank-one translog curvature are tested; Kaplan's curvature (σ ≈ 0.54) is rejected (or not) on unbalanced designs; Hoffmann's A3 fails the duality test and Besiroglu's passes.
- **Why economists care.** σ between "model capacity" and "data" is a key parameter for AI growth and automation models [trammell2023economic; farboodi2026datadriven; jones2026ai; bergemann2026menu]. It sits next to the capital–labor σ literature (0.4–0.7). The identification results are clean applications of ACF, DMR and Nerlove to a new domain with *designed* experiments, which IO rarely has.
- **Why ML researchers care.** It explains the Hoffmann/Besiroglu and Kaplan/Chinchilla disputes, says which reported numbers are structural, and yields design rules (fraction of runs off-ray; normalization; which estimator).
- **Formal results.** P1 (duality and cross-equation restrictions); P2 (on-path rank deficiency; non-identification of σ with free κ; M* identified only under the FOC); a corollary that on-path time series identify augmentation rates, not σ (DMR); Monte Carlo on on-path vs IsoFLOP vs factorial designs.
- **Estimators.**
  - Hoffmann Huber-LSE (reference); LAD; NLS in levels; PPML and Gamma-QMLE in levels; VPNLS; KMW-normalized NLS.
  - A system (loss equation + IsoFLOP-argmin FOC with cross-equation restrictions) [leonledesma2010identifying].
  - Translog in (ln N, ln D) on ln(L − E) with the rank-one test; a Farseer-form NLS; local nonparametric σ on Farseer's orthogonal grid.
  - Inference: cluster bootstrap by IsoFLOP budget, run or corpus; delta method for σ*, a, γ, M*; Andrews–Cheng weak-ID CIs for E and A.
- **Data.** Epoch Chinchilla (245); Farseer (404; corr(ln N, ln D) = −0.04); Gadre (104 models × 8 losses with token-level 95% CIs); Porian (tuned subset; https://github.com/formll/resolving-scaling-law-discrepancies); datablations via ColPret (229 contour rows; https://github.com/IBM/ColPret); OLMo ladder (30 runs, BPB); Step Law dense (frontier = min over hyperparameters; https://github.com/step-law/steplaw); Kricheli collinear vs diverse designs (https://huggingface.co/datasets/TPPIsCriticalFor/colinear_scaling_models); open-athena IsoFLOP compilation (Chinchilla, Llama 3, Marin 2026).
- **Feasibility.** High. Seconds to minutes per fit; bootstraps are minutes.
- **Risk.**
  - Digitization error in the Chinchilla data (±0.01 loss; D imputed as C/6N).
  - Heterogeneous units across datasets: compare σ, a and γ within dataset, never raw levels.
  - Farseer and Step Law have no license: cite, don't redistribute, ask the authors.
  - The σ derivation is Hao & Merrill's; the novelty is estimation, testing and identification.
- **Novelty.** ★★★★ (C2 estimation, C3 behavioral/duality, C11, C12).

### Rank 3 — Observational vs experimental production functions: a LaLonde test for IO estimators, applied to LLMs (supporting application S)
- **Claim.** Using sweeps as ground truth for the N- and D-elasticities of a common output (BPB/WikiText-103; benchmark logits on shared tasks), quantify the bias of cross-lab estimators and whether IO corrections recover the truth:
  - pooled OLS, developer/family FE, Mundlak;
  - ACF-type control function (N predetermined; proxies = D-given-N, post-training compute, reported LR/batch);
  - DIF-GMM over family generations;
  - IV (hardware release prices, export-control exposure, chip generation);
  - EIV corrections (Epoch confidence classes; two compute measures).

  Decompose Ho's β_data = 0.04 vs about 0.3 into attenuation, simultaneity and specification (e.g., E = 0).
- **Why economists care.** It is the first "LaLonde (1986)" benchmark for *production-function* estimators in any industry. It tests which IO fixes work when truth is known, a methods contribution with external value. It also shows that the sign of transmission bias depends on firm behavior (P5).
- **Why ML researchers care.** Observational scaling laws [ruan2024observational; maiapolo2024sloth; mertens2026secret; ho2025rosetta] and algorithmic-progress numbers are used for forecasting and policy; this says how far to trust them.
- **Formal results.**
  - P5: the transmission-bias sign by regime (target ⇒ attenuation; budget ⇒ consistent; funding ⇒ overstatement), with forward/reverse bounds.
  - Hicks-neutral ω biases γ, not a.
  - The D-elasticity is identified under family FE only from families with within-family D variation.
- **Data.**
  - Observational:
    - ObsScaling (148 base models, 123 with N and D, 39 families, 8 benchmarks);
    - Sloth `data_v2` (175 with N and D, 33 families, Open LLM Leaderboard v1/v2);
    - Epoch all-models (589 language models with N, D and C; confidence; hardware; 126 with hardware-based C);
    - Mertens 809 models (arXiv 2602.07238; Google Drive link [UNV access]);
    - Ho et al. sheet (408 rows, tokenizer and vocabulary columns): https://github.com/epoch-research/lm-algorithmic-progress
    - Epoch hardware: https://epoch.ai/data/ml_hardware.csv
  - Experimental on the same outputs:
    - DataDecide eval/ppl (OLMES tasks, WikiText-103; 25 recipes × 14 sizes × 3 seeds): https://huggingface.co/datasets/allenai/DataDecide-eval-results
    - OLMo ladder (task BPB and accuracy; WikiText-103);
    - Pythia/PolyPythias (lm-eval JSON; seeds): https://github.com/EleutherAI/pythia
    - Gadre (46 downstream tasks);
    - Proteus-2k (2.4k checkpoints, 6 benchmarks, FLOPs): https://huggingface.co/datasets/hlzhang109/proteus-2k
- **Feasibility.** High for estimation. Medium for harmonization: common tasks and units, and name joins across Epoch, ObsScaling and Sloth (the Epoch–ECI exact-name join is 219/268).
- **Risk.**
  - Scale overlap is limited: experiments mostly ≤13B, observational up to 405B+.
  - Recipes differ, so "truth" is itself heterogeneous.
  - Small N per family (T = 3–6 generations); weak instruments.
  - Output comparability (tokenizers; contamination).
- **Novelty.** ★★★★ (C4 and C8; the LaLonde framing was not searched explicitly: verify).

### Rank 4 — Decomposing "algorithmic progress": technical vs allocative, neutral vs biased, and the index-number problem
- **Claim.**
  - (i) Measured effective-compute growth mixes technical change with allocative-efficiency gains. The Kaplan→Chinchilla re-balancing (≈10× of 6,930× at the frontier [gundlach2025origin]; 2–4× per Ho) is Farrell allocative, not technical.
  - (ii) Only g_N + g_D is identified from time series (DMR). Cross-sectional sweeps at each vintage (Chinchilla 2022, Porian/Gadre 2024, Farseer 2025, Marin 2026) identify local σ and hence the bias.
  - (iii) Hicks neutrality ⇔ α·g_N = β·g_D. Constant CEG ⇔ equal exponents and E, so scale-dependent CEG (LSTM → Transformer) is non-neutral.
  - (iv) A superlative (Törnqvist) progress index removes reference dependence to second order.
  - (v) The Sahal bias in naive frontier-release regressions is γ/(1 − s_A).
  - (vi) Hall-type bias in cost-share growth accounting with over-trained models is (ε_N − ε_D)(Δn − Δd)/2.
- **Why economists care.** These are the TFP-measurement debates (Solow, Jorgenson–Griliches, Hulten, Diewert) in the fastest-moving industry. They bear on forecasting and compute governance.
- **Why ML researchers care.** They reconcile 2×/yr vs 9×/yr progress estimates and say which rates are structural.
- **Formal results.** Propositions P3, P4, P7, P8; Farrell decomposition C_actual/C_min = technical × allocative, with closed-form allocative loss under Chinchilla.
- **Data.** Ho et al. sheet; Gundlach experiments (https://github.com/hansgundlach/Experimental_Progress); vintage sweeps (above); DataDecide (25 recipes) and open-sci-ref (8 corpora × 4 sizes × 3 budgets: https://huggingface.co/datasets/open-sci/open-sci-ref-0.01-logs) for neutrality tests; ECI for a latent output.
- **Feasibility.** High (small data; NLS/bootstrap).
- **Risk.** Output units differ across vintages. Ho's data mix word- and subword-level perplexities; use their tokenizer columns and BPB conversions. Common-output bridges (WikiText-103 BPB) exist only for recent sweeps.
- **Novelty.** ★★★★ (C6 partially done; DMR, Farrell and index-number framings are new).

### Rank 5 — Hyperparameters as flexible inputs: a stochastic-frontier treatment of the Kaplan–Chinchilla discrepancy
- **Claim.** Scaling laws are concentrated (envelope) production functions L*(N,D) = min_h L(N,D,h). Scale-correlated inefficiency (∂u/∂ln N < 0) biases the allocation exponent upward by a computable amount. It reproduces Porian's 0.835 → 0.497 path together with the measurement corrections (head FLOPs = CWDL-type input error; non-embedding N = Nerlove's small-firm economies). Flexible-input demand functions η*(N,D) and B*(N,D) are estimated with SEs, and the Bjorck–Step Law sign disagreement is analyzed as conditional vs unconditional demands.
- **Why economists care.** A clean, experimental demonstration of transmission-type bias from unoptimized flexible inputs and of SFA inefficiency correlated with inputs, with the ground truth observed.
- **Why ML researchers care.** It tells practitioners how much tuning is needed and which quantities are robust to it (σ vs a).
- **Formal result.** Bias formula for a under u(ln N); the envelope theorem bounds second-order effects.
- **Data.** Step Law dense (1,911 runs; 17 cells × 26 LR × 13 BS) and MoE (708); Porian pickle (tuned vs untuned; several FLOP conventions); Lourie 2026 (data availability [UNV]).
- **Feasibility.** High.
- **Risk.** Step Law has no license and only 17 (N,D) cells. The Porian pickle row count needs checking. The novelty is mostly framing plus quantification.
- **Novelty.** ★★★.

### Rank 6 — Lab TFP dispersion with gross-output (distillation) and TFPQ/TFPR corrections
- **Claim.** Mertens-type "secret sauce" (Microsoft/Nvidia ≈60×; within-developer 41× p90/p10) shrinks once:
  - (i) teacher and synthetic-data compute enter as intermediate inputs (GNR gross output);
  - (ii) output is BPB or ECI rather than a saturating benchmark logit;
  - (iii) contamination and test-task training are deflated [dominguezolmedo2025training].

  Dispersion in reducible-loss units (≈1.8–1.9×) is then compared with Syverson's 1.92.
- **Why economists care.** It is the Syverson dispersion fact, and misallocation, in a new industry; gross output vs value added is a live debate.
- **Why ML researchers care.** It says which "efficiency" claims are real.
- **Formal result.** Value-added TFP bias = (teacher elasticity) × (log teacher compute), under GNR-type assumptions.
- **Data.**
  - Mertens 809 models (access [UNV]); Epoch all-models (`Base model`, `Finetune compute`, `Post-training compute` columns); Open LLM Leaderboard contents (Base Model lineage, merges); ECI.
  - Hand-coded teacher lineage (Phi, Minitron/Nemotron, Gemma-2 distilled, DeepSeek-R1-distill).
- **Feasibility.** Medium (hand-coding teacher compute).
- **Risk.** Teacher compute is rarely disclosed; highest overlap risk with Mertens et al.
- **Novelty.** ★★★ (C5 partially done).

### Rank 7 — The elasticity of substitution between training and inference (test-time) compute
- **Claim.** Estimate σ(train, test) and its heterogeneity by task difficulty. Jones's Hex isoquant is Cobb–Douglas (σ = 1, slope −1.2); LLM repeated sampling gives "coverage = exp(a·k^b)"; Snell finds difficulty dependence. Aggregating over difficulty (Houthakker/Oberfield–Raval) predicts σ_agg. Link this to w.
- **Why economists care.** It is the key parameter in AI growth models (GATE's m ≈ 1–2; Cunningham's γ ≈ 0.15–0.3), currently calibrated loosely.
- **Why ML researchers care.** It is the compute-allocation rule between training and inference.
- **Data.** Jones boardlaw SQLite (69.5 MB, MIT): https://github.com/andyljones/boardlaw ; monkey_business (Gemma 2B/7B, Llama-3 8B/70B; MIT): https://huggingface.co/datasets/ScalingIntelligence/monkey_business
- **Feasibility.** High.
- **Risk.** Bounded outputs (coverage, Elo) make σ cardinalization-dependent. Partially done by Epoch.
- **Novelty.** ★★★.

### Rank 8 — Estimation practice from IO: an estimator horse race and experimental design for scaling studies
- **Claim.**
  - Huber δ = 10^−3 on log L is LAD.
  - Log-space fitting with additive level noise raises the log-of-gravity issue: compare with PPML/Gamma-QMLE.
  - VPNLS equals concentrating out linear parameters; KMW normalization; Andrews–Cheng weak-ID CIs; E = 0 as a boundary test; cluster bootstrap by run.
  - D-optimal design: what fraction of runs must be off-ray?

  Report how much of the M* uncertainty is sampling vs specification.
- **Why it matters.** Credible SEs for every number in H1 and H2. Most ML papers report none [li2025misfitting].
- **Data.** Same as Rank 2.
- **Feasibility.** High.
- **Risk.** Better as an appendix or companion methods note than as a standalone AER contribution.
- **Novelty.** ★★★ (C12).

### Rank 9 — Sutton bounds for frontier AI
- **Claim.** The scaling-law convex cost of capability C(ℓ) ∝ (ℓ − E)^{−1/γ}, with 1/γ ≈ 5.6–6.5, combined with demand escalation in quality (hedonic slope 0.039–0.047 log price per Intelligence-Index point [demirer2025emerging]; Merali's productivity per 10× compute [merali2024scaling]) gives an endogenous-sunk-cost lower bound on concentration [sutton1991sunk].
- **Why it matters.** It connects the production function to market structure and policy.
- **Data.** OpenRouter shares; Artificial Analysis (the user must create an API key); Demirer et al.
- **Feasibility.** Theory plus light calibration.
- **Risk.** Far from the core estimation paper; better as the concluding extension or a separate paper.
- **Novelty.** ★★★★ (unused), but low fit to this paper.

(Also considered and folded into the above: the Houthakker/quantization aggregation restriction (a short subsection, since Besiroglu's refit rejects it); the MoE capital-services mapping (measurement subsection); repetition as depreciation (measurement subsection); Sahal bias (in Rank 4).)

---

## 8. (e) Recommended paper architecture and empirical plan

### 8.1 Title, pitch and headline results
**Title:** "Scaling Laws as Production Functions" (alternatives: "What Scaling Laws Identify"; "The Production Function of Machine Intelligence").

**Abstract skeleton (AER: short, about 100 words [UNV current limit]).**

> Neural scaling laws are production functions: models are produced from parameters and data at a cost in compute. We show that the canonical estimation approaches are cost-function, factor-demand and production-function estimators, and that optimizing behavior destroys the variation needed to identify the technology — the functional-dependence and Diamond–McFadden–Rodriguez problems of empirical IO. Using designed experiments from public training sweeps, we estimate an elasticity of substitution between parameters and data of about 0.74. Inverting labs' first-order conditions, we show that over-training reveals anticipated inference demand: small open-weight models were built for several times their training compute in lifetime inference.

**Headline results.**
- **H1.** The technology (σ ≈ 0.74 [CI]; frontier elasticity γ ≈ 0.16–0.18; CES tested; Kaplan's curvature rejected), plus the identification theorems that say why only designed variation delivers it.
- **H2.** Revealed inference demand (the lifetime/training compute multiple w) for open-weight families, with partial-identification bands, within-family revealed-preference checks and validation against usage.
- **S.** A LaLonde benchmark for observational estimators plus the Farrell/DMR decomposition of "algorithmic progress". This can be trimmed to one section if length binds.

**JEL (suggested):** D24, L86, O33, C51. **Keywords:** production function estimation, elasticity of substitution, scaling laws, large language models, revealed demand.

### 8.2 Sections

1. **Introduction** (≈6 pp). Pitch; three results with numbers; relation to [hao2026theory; mertens2026secret; whitfill2025note; kricheli2026tokens; ho2024algorithmic; ruan2024observational] and to IO [olley1996dynamics; ackerberg2015identification; gandhi2020identification; deloecker2012markups; diamond1978measurement; nerlove1963returns].
2. **Scaling laws and the production problem.**
   - Setup (inputs, cost 6ND, output ℓ, BPB); the dictionary (Table 1, condensed from §4).
   - What is ordinal (isoquants, σ, expansion path) vs cardinal (RTS, TFP levels).
   - Facts: the Kaplan–Chinchilla dispute as measurement (Nerlove/CWDL); the Chinchilla replication as an optimizer failure (BLP lesson).
3. **Identification.**
   - 3.1 Duality: A1 = inverse cost function, A2 = conditional factor demands, A3 = primal; cross-equation restrictions (P1).
   - 3.2 Functional dependence from optimizing behavior (P2: rank-one Hessian; what on-path data identify; M* only under optimality; "the better labs optimize, the less their data reveal").
   - 3.3 Technical change over time: DMR (P3), neutrality and CEG (P4).
   - 3.4 Observational data: transmission bias by behavioral regime (P5), selection, measurement error.
   - 3.5 Over-training FOC and revealed inference demand (P6), with closed forms under homothetic CES.
4. **Data.**
   - (i) Experimental panel: Chinchilla extraction, Farseer, Gadre, Porian, datablations, Step Law, OLMo ladder, DataDecide, Kricheli, Llama 3 IsoFLOPs; about 3–4k runs; BPB harmonization where possible.
   - (ii) Observational panel: Epoch ∩ ObsScaling ∩ Sloth ∩ ECI ∩ HF API; about 600 language models; lineage; confidence classes.
   - (iii) Demand side: HF downloads, OpenRouter prices and provider counts with Wayback panel, LMArena votes.
   - Table 2 (datasets, n, design, off-ray variation sd(ln(D/N) | ln C), license).
5. **Estimating the technology (H1).**
   - Estimator horse race and inference; σ, γ, a, M* by dataset (Table 3); CES / translog / rank-one / Kaplan-κ tests.
   - Duality tests (Table 4; Hoffmann vs Besiroglu reconciled).
   - On-path vs off-path Monte Carlo (Fig. 2); σ forest plot vs the K–L literature (Fig. 3).
6. **Over-training reveals inference demand (H2).**
   - Implied w and T by family (Table 5; Fig. 4).
   - Within-family flagship calibration (Meta, AI2/OLMo-2, Qwen, Gemma).
   - Raval over-ID test; rival wedges (data scarcity, memory/latency, non-homotheticity via Farseer); validation (downloads, prices, provider counts).
   - Farrell cost of over-training: C/C_min.
7. **Observational production functions and algorithmic progress (S).**
   - LaLonde table (Table 6: estimator × elasticity vs experimental truth); EIV and selection bounds.
   - Farrell/DMR decomposition of effective-compute growth (Fig. 6; Table 7); neutrality tests on DataDecide and open-sci-ref.
8. **Discussion and conclusion.** What economists should use (σ, γ, w; not raw exponents); design recommendations for ML scaling studies (off-ray share; normalization; PPML/LAD; clustered inference); the link-function problem (loss → value); Sutton and market structure as an extension.

**Appendices.** A: proofs. B: data construction and harmonization (BPB conversion; Farseer column fix; ColPret datablations rebuild). C: estimators and inference. D: robustness (units, outliers, Huber δ, support restrictions). E: optional self-trained sweep.

### 8.3 Key figures and tables

| # | Content | Data |
|---|---|---|
| Fig. 1 | (ln N, ln D) plane: Chinchilla points coloured by loss; isoquants from the fitted law; isocost lines; expansion paths (Hoffmann vs Besiroglu vs Meta); transverse vs along-path variance (1.12 vs 1.84) | Epoch Chinchilla; Llama 3 IsoFLOPs |
| Fig. 2 | Monte Carlo: sampling distributions of σ, M*, α and β under on-path, IsoFLOP-only and factorial designs (same compute budget) | Simulated from the Besiroglu truth; Farseer-design resampling |
| Fig. 3 | Forest plot of σ* (and local σ) with CIs across datasets and estimators; band for capital–labor σ (0.4–0.7) | Rank-2 datasets |
| Fig. 4 | Revealed lifetime/training compute multiple w vs M for open-weight models; bands from technology uncertainty and M* heterogeneity; Farseer/Gadre/Chinchilla support shaded; flagship points at w ≈ 1 | Epoch/ObsScaling + technology |
| Fig. 5 | LaLonde plot: observational estimates (by estimator) vs experimental benchmark for ε_N and ε_D | Rank-3 datasets |
| Fig. 6 | Decomposition of effective-compute growth 2012–2025: input growth, technical change (neutral/biased) and allocative efficiency | Ho sheet + Gundlach + sweeps |
| Tab. 1 | Condensed dictionary (≈20 rows) | §4 |
| Tab. 2 | Datasets | §8.4 |
| Tab. 3 | Technology estimates by dataset × estimator (E, α, β, a, γ, σ*, M* at 1e21/1e24/1e26; CES p-value; κ) | Rank 2 |
| Tab. 4 | Duality and revealed-preference tests (A1 vs A2 vs A3; Chinchilla-70B w) | Chinchilla, Llama 3 |
| Tab. 5 | Revealed inference demand by family and size | Rank 1 |
| Tab. 6 | Cross-lab production functions by estimator; bias vs experiment | Rank 3 |
| Tab. 7 | Algorithmic progress under alternative identifying assumptions | Rank 4 |

### 8.4 Empirical plan: results → datasets → estimators

| Result | Datasets (files) | Estimator / computation | Identifying assumption | Compute on laptop |
|---|---|---|---|---|
| P1 duality test | Epoch Chinchilla `svg_extracted_data.csv`; open-athena `isoflop-experiments` (Chinchilla, Llama 3, Marin) | A1 envelope fit (Nerlove cost function); A2 parabola/VPNLS argmins; A3 NLS; Wald/Hausman on a = β/(α+β), γ = αβ/(α+β); KMW-normalized system | ZKD (inputs fixed before seed shock); correct functional form | seconds |
| P2 on-path non-identification | Same + Farseer | Subsample to per-C argmins; Jacobian rank; profile likelihood of σ with κ free; condition numbers; bootstrap SE ratios | — | minutes |
| σ, γ, a, M* estimates | Chinchilla (245); Farseer (404); Gadre (104×8 losses, CI weights); Porian (tuned runs); datablations (ColPret 229); OLMo ladder (30); Step Law (min over h: 17 cells); Kricheli | Huber-LSE, LAD, NLS-levels, PPML/Gamma-QMLE, VPNLS, normalized NLS; translog + rank-one test; nested-q (Kaplan) NLS; local σ via kernel-weighted translog on Farseer; cluster bootstrap (budget/run/corpus); delta method | Designed variation; separability tested | minutes |
| CES and neutrality tests across recipes | DataDecide ppl (25 recipes × 14 sizes × 3 seeds; final D = 100N, so use checkpoints plus a WSD caveat); open-sci-ref (8 corpora × 4 sizes × 3 budgets); Gadre (3 corpora) | Recipe-specific (A_r, B_r, α_r, β_r); test intercept-only (Hicks), A/B-only (augmenting), exponent change | Common architecture within suite | minutes |
| H2 revealed inference demand | Technology: Farseer, Gadre, Llama 3 IsoFLOPs, OLMo ladder, Chinchilla. Choices: Epoch all-models (N, D, epochs, open weights, base model), ObsScaling, Sloth, HF safetensors (exact N). Validation: HF downloads, OpenRouter models/endpoints + Wayback, LMArena votes | ŵ = (αA·N^−α)/(βB·D^−β) with bootstrap bands; homothetic closed form; family GMM (flagship FOC = 1 pins M*); Raval over-ID; regress log T̂ on log downloads, provider counts and price | Lifetime-compute minimization; technology transferable within lab (or BPB-harmonized); Hicks-neutral lab TFP | minutes; API pulls are small |
| Transmission bias / LaLonde | Observational: ObsScaling, Sloth, Epoch, Mertens [access UNV], Ho sheet. Experimental: DataDecide eval, OLMo ladder, Pythia/PolyPythias, Gadre, Proteus-2k | OLS; family/developer FE; Mundlak; CF (ACF 2-step with proxy D-given-N or post-training compute); DIF-GMM over generations; IV (hardware release price and FLOP/$ at training date; export-control exposure); EIV (confidence-class reliability; two compute measures); Lee bounds (Notable vs all) | Timing (N predetermined); exclusion of hardware shocks; classical error in the second measure | minutes |
| Algorithmic progress decomposition | Ho sheet (408 rows); Gundlach experiments; vintage sweeps; ECI | Re-estimate Ho with the neutrality restriction α·g_N = β·g_D and a vintage-specific cross-sectional σ; Farrell decomposition; Törnqvist index; Sahal correction | Common-output harmonization (BPB/word-level) | minutes |
| Flexible inputs / SFA | Step Law dense + MoE CSVs; Porian pickle | Frontier = min over h; half-normal/exponential SFA with u(ln N, ln D); flexible-input demand regressions; bias-in-a simulation | Tuned frontier = concentrated technology | minutes |
| Train/test isoquant | boardlaw SQLite; monkey_business | Isoquant slopes; CES in (train, test) by board size / difficulty bin; aggregation | Elo/coverage monotone in latent output | minutes |

### 8.5 Optional self-trained experiments on the M5 Max
Purpose: fill gaps that public data cannot.
- (i) An **over-training sweep in one recipe**: N ∈ {1, 2, 4, 8, 16}M × M ∈ {5, 20, 80, 320, 1280} × 2 seeds, BPB on a fixed corpus. It verifies w = (M/M*)^ρ, checks the extrapolation direction (is ε_D overstated at extreme M, as Sardana reports?) and gives a within-support benchmark.
- (ii) A **semi-synthetic "lab market"**: 2–3 recipes (data-quality tiers = TFP) × the grid above. "Labs" choose (N, D) from real runs under the target / budget / funding rules, and IO estimators are applied to the selected subsamples with known truth (the LaLonde logic under control).

Budget: the largest cell is 16M × 20B tokens ≈ 2e18 FLOPs, which is too big. Cap at 16M × 5B tokens ≈ 5e17. Total ≈ 1–2e18 FLOPs, a few days at 10–20 TFLOP/s effective [UNV: M5 Max sustained throughput].

Tooling: MLX or PyTorch-MPS, a nanoGPT-style model, WSD schedule so that one run yields many D values (with the Hägele caveat [hagele2024scaling]); corpora such as FineWeb-Edu or TinyStories subsets (download approval needed). This is optional: every headline result can be produced from public data.

### 8.6 Order of work (suggested)
1. Freeze data snapshots, after user approval for downloads:
   - Chinchilla, Farseer, Gadre, OLMo ladder, Porian, Step Law, ColPret;
   - Epoch all-models, hardware and benchmarks zip;
   - ObsScaling, Sloth, the Ho sheet;
   - HF API and OpenRouter pulls.
2. Re-derive all [D] results symbolically; write proofs P1–P6.
3. H1 estimation plus the Monte Carlo.
4. H2 wedge.
5. LaLonde and progress decomposition.
6. Write in AER format (AEA LaTeX template, `aea.cls` [UNV exact class name]; compile `references.bib` with natbib author-year).

---

## 9. (f) Open questions and things to verify next

**Bibliographic and factual items to verify before citing:**
1. Llama 3 (a)–(c):
   - (a) 8B/70B per-model training tokens (≈15T assumed; only the 405B's 15.6T is in the text read);
   - (b) the exponent: 0.53 in the text vs 0.537 in the Fig. 3 label (only 0.537 reproduces 16.55T);
   - (c) confirm that "N*(C)" denotes optimal **tokens** [grattafiori2024llama].
2. Whether Muennighoff's α = β = 0.3526596 was *imposed* or estimated; ml_core says tied. This matters for claiming independent CES evidence [muennighoff2023scaling].
3. Hoffmann TeX-precision values (E 1.6934, α 0.3392, β 0.2849) come via Besiroglu; confirm against the arXiv TeX source. Confirm the NeurIPS 2022 pages and DOI (the ml_core entry has pages 30016–30030 and DOI 10.52202/068431-2176 from Semantic Scholar; other strands flag them [UNV]).
4. The [hao2026theory] σ formula and footnote-3 OLMo-Hybrid exponents (secondary source [merrill2026olmo]).
5. Syverson's 90–10 = 1.92 (e^0.651) [syverson2011determines], and the Mertens Table C.1 coefficients and data link [mertens2026secret].
6. Other citation items:
   - [houthakker1955pareto]: end page, year, given name. io_ces_duality says "do not cite without checking".
   - [demirer2020production]: Econometrica publication status.
   - [bond2020unpleasant]: JME 121:1–14.
   - [griliches1998production]: year 1998 vs 1999.
   - [nerlove1963returns]: pages.
   - [diamond1978measurement]: volume and pages.
7. Unresolved numerical disagreements:
   - Densing law: 3.3 (arXiv) vs 3.5 months (NMI); cite NMI.
   - Ho et al.: 8 months [5, 14] in the abstract vs 8.4 [4.5, 14.3] in the text.
   - Erdil–Besiroglu: 8.95 reported vs ≈8.1 from the point ratio.
   - Cottier et al.: 90% CI 2.0–2.9 vs 95% CI 2.0–3.1.
   - Step Law's table lists DeepSeek's LR coefficient as 0.3188; DeepSeek's paper says 0.3118.
8. Venue years flagged in `references.bib` notes: gadre2024language (ICLR 2025?), choshen2024hitchhikers (ICML 2025?), ho2024algorithmic (NeurIPS 2024 pp. 58245–58283 per novelty/data; other strands UNV), schaeffer2023emergent, biderman2023pythia, maiapolo2024sloth.

**Data checks (require downloading after user approval):**

9. Recompute from the CSVs:
   - the within-family D variation table for ObsScaling (the WebFetch summary had errors, e.g. BTLM "627T");
   - the Farseer true D/N (the `D/N` column is mislabeled);
   - Porian pickle row counts;
   - datablations N/D/U rebuilt from model names (ColPret's `tokens_per_epoch` and `flops` look wrong);
   - the DataDecide seed structure;
   - ECI's IRT specification.
10. Licenses: Farseer, Step Law and Epoch's analyzing-chinchilla have **no license file**. Use for research, don't redistribute raw files, consider asking the authors. OpenRouter and Artificial Analysis terms are unread. Artificial Analysis needs an account and API key that **the user** must create.
11. Access to the Mertens et al. 809-model dataset (Google Drive link in the paper), Demirer et al. OpenRouter volumes, and Lourie 2026 runs.

**Substantive open questions for the modelling stage:**

12. **Output for H2.** Should the wedge be computed with loss/BPB-based technology only (yes, per Bond et al.)? How should BPB be harmonized across tokenizers for choices and technologies from different labs?
13. **Rival wedges.** How do we separate inference demand from (i) non-homotheticity (Farseer: M*(C) rising in C), (ii) data scarcity (a Muennighoff shadow price that *lowers* w), (iii) memory/latency/hardware-fit constraints (e.g., 8B fits one GPU), (iv) distillation targets? Within-family variation plus validation data is the proposed route. Is a structural demand model needed?
14. **Is σ well defined off the Chinchilla regime?** Farseer's non-separable form implies σ varies with (N, D). Report local σ on the grid, and decide whether the headline is "σ* on the expansion path".
15. **Common output for the LaLonde test.** WikiText-103 BPB (lm-eval-harness `bits_per_byte`) vs benchmark logits. Is the scale overlap between experiments (≤13B) and observational data sufficient?
16. **Sign of notability-selection bias** (OP-style conjecture: exponents understated). Derive it under an explicit release rule and test with Notable vs All.
17. **Whether Hicks-neutral lab ω is plausible.** DeepSeek's quality-dependent a says data quality is non-neutral. If ω is factor-biased, both the wedge and the N/D-split results change. Test on DataDecide and open-sci-ref first.
18. **Kaplan κ test power.** Can q/κ be identified from existing unbalanced designs (Gadre M = 640; Farseer M = 2,570; datablations up to 1,500 epochs)?
19. **Novelty re-sweep before submission.** Google Scholar, SSRN and Substack for 2026 working papers by IO economists (Mertens, Demirer, Raval, De Loecker, Syverson, Syrgkanis) and Epoch "Gradient Updates" after mid-2026. Specifically search "LaLonde"/"experimental benchmark" + production function + scaling, and "revealed inference demand" / "over-training" + "first-order condition".
20. **Self-trained sweep.** Confirm M5 Max sustained throughput under MLX/MPS before committing to §8.5.
21. **AER formatting.** Fetch the current AEA LaTeX template and submission guidelines (abstract length, JEL, data-availability and replication-package policy: all public data plus code, with licenses for Farseer and Step Law resolved).

---

### Appendix: formal statements to prove (for the modelling strand)

- **P1 (Duality).** For ℓ = AN^−α + BD^−β and C = 6ND:
  - min_{N,D} C s.t. ℓ ≤ ℓ̄ gives C*(ℓ̄) = 6[K/ℓ̄]^{1/γ}, with K = (α+β)(A/β)^{β/(α+β)}(B/α)^{α/(α+β)};
  - factor demands N* = G(C/6)^a, D* = G^{−1}(C/6)^{1−a}, with G = (αA/βB)^{1/(α+β)}.
  - A1 estimates C*^{−1}; A2 estimates N*(·); A3 estimates ℓ.
  - Restrictions: a = β/(α+β) and γ = αβ/(α+β).
- **P2 (On-path identification).** ∇²ln ℓ in (n, d) has rank 1 with null vector (β, α).
  - On-path data identify (a, γ, K, G, E). Under κ = 1 they identify α = γ/a and β = γ/(1−a); A and B are identified only jointly with the FOC G^{α+β} = αA/(βB).
  - With L = E + (AN^−a₁ + BD^−b₁)^κ, any a₁ + b₁ = S > 0 is observationally equivalent, so σ* = 2/(2+S) is unidentified.
- **P3 (DMR for progress).** With A_t = A·e^{−α g_N t} and B_t = B·e^{−β g_D t}:
  - K_t declines at rate γ(g_N + g_D);
  - ln(D*/N*) drifts at 2(α g_N − β g_D)/(α+β);
  - the Hicks bias is (1 − 1/σ)(g_N − g_D) in CES;
  - on-path time series identify g_N + g_D and α g_N − β g_D but not σ, hence not the sign of the bias.
  - Neutrality ⇔ α g_N = β g_D.
- **P4 (CEG).** For an old and a new Chinchilla technology, the compute-equivalent gain f(C) = C_old/C_new at equal loss is constant in C iff (α, β, E) are equal. It then equals (K_old/K_new)^{1/γ}. Factor augmentation only rescales K (and G), so it always yields a constant CEG.
- **P5 (Transmission bias).** Along the ω-invariant path, ln ℓ = −ω + ln K − γ c + ε. OLS of ln ℓ on c:
  - target rule c = c̄ − ω/γ ⇒ slope → 0;
  - exogenous c ⇒ −γ;
  - c = c̄ + λω ⇒ −γ − λ Var(ω)/Var(c).
  - Forward and reverse regressions bound γ under classical noise.
- **P6 (Revealed inference demand).** min 6ND + 2NT s.t. ℓ(N,D) = ℓ̄ ⇒ ε_N/ε_D = 1 + T/(3D). The result is invariant to ℓ → e^{−ω}ℓ and to E.
  - If α = β = ρ: ε_N/ε_D = (M/M*)^ρ with M* = (B/A)^{1/ρ}, and C/C_min = cosh(½ρ ln(M/M*))^{2/ρ}.
  - Partial identification: T ∈ {3D(w(θ) − 1): θ ∈ Θ̂}.
- **P7 (Sahal).** If frontier reducible loss falls at rate m = γ(g + g_A) and physical compute grows at g, a naive regression recovers γ/(1 − s_A), with s_A = g_A/(g + g_A).
- **P8 (Hall-type accounting bias).** With cost-share (equal) weights and over-trained models, the residual bias is (ε_N − ε_D)(Δn − Δd)/2.
