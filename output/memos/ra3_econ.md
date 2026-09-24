# Memo: module ra3_econ, economic implications (Section V)

Module owner: ra3_econ (Claude). Date: 2026-09-24. Entry point: `code/analysis/ra3_econ/run.py`. One command regenerates every number, table and figure below. The run is deterministic (fixed seeds; two consecutive runs gave byte-identical CSV and TeX outputs). It is CPU only, uses at most 5 processes (only for ra2's wild bootstrap) and takes about 60 s.

**Independent review (2026-09-24; `output/memos/ra3_econ_review.md`).** Numbers below are from the reviewed run. The review changed five things:
- (i) The inference share (H6–H8) now uses **module ra2_wedge's final, reviewed inversion**: the κ-free reference, ra2's clean open-weight universe and clean verified sample, ra2's wild bootstrap and its 32 ex-ante technologies. It reproduces ra2's aggregate file exactly (16 checks). The builder's preliminary m3/κ = 1 version is kept only as a superseded file.
- (ii) The Monte Carlo for dates anchored at observed M now re-anchors the level for every technology draw. The builder's version re-introduced the technology's level uncertainty.
- (iii) The observed-M anchor excludes the one MoE run, whose N is active parameters.
- (iv) ra1's reviewed model-free path slopes were added.
- (v) Two LaTeX errors in the disclosures table were fixed (an unescaped "&", and double row breaks).

The data-wall (H4) and compute-growth (H1) numbers were replicated independently and did not change.

Addresses R3 M7 (and R3 minor 29) and R1 comment 6(e). It also touches R2 M3 / R3 M3(c) on validating aggregate levels against Patterson et al. (2022) and Wu et al. (2022).

Notation follows `paper/notes/model_spec.md`:
- a = β/(α+β) is the path slope: N* ∝ C^a and D* ∝ C^(1−a).
- γ = αβ/(α+β) is the frontier elasticity.
- σ* = 2/(2+α+β) is the on-path elasticity of substitution. In the κ family it uses the inner exponents.
- w = ε_N/ε_D, and s = (w − 1)/w is the planned inference share of lifetime compute.
- D is tokens *processed*; U is *unique* tokens.
- r = D*(C)/U is data scarcity: the ratio of compute-optimal tokens to available unique tokens.

Labels used below:
- **Estimated**: computed here.
- **Upstream**: a technology or wedge estimated by m1, m2, m3, ra1 or ra2.
- **Literature**: a number taken from a source; each was checked against the primary text on 2026-09-24.

---

## 1. Headline findings

### H1. Frontier training compute grows about 5× per year, 2018–2026, consistent with Epoch's "4–5×"

**Estimated.** Sample: the running top-10 training runs at release (Epoch's frontier definition), rebuilt from Epoch's database (snapshot 2026-09-23).
- n = 92 runs, 27 developer clusters, January 2018 to September 2026.
- Growth is **5.06× per year**, with wild-cluster-bootstrap 95% CI **[4.29, 5.93]** (Webb weights, B = 9,999, p-value floor 10⁻⁴).
- 4.5× per year is not rejected (restricted WCB p = 0.149). 4.2× per year is rejected at 5% (p = 0.044).
- The fitted frontier level in September 2026 is **9.4×10²⁶ FLOP**. The largest recorded run is 1.0×10²⁷ FLOP (GPT-6 Astra, Epoch confidence "Likely").

Robustness:

| Sample | Growth (×/yr) | 95% CI |
|---|---|---|
| Excluding "Speculative" compute | 4.84 | [4.14, 5.67] |
| Record-setting runs (only 6 clusters; indicative) | 4.85 | [3.98, 5.91] |
| Epoch's own Frontier flag (ends 2025-07) | 4.97 | [4.06, 6.12] |
| Language-model top-10 | 6.45 | [5.65, 7.37] |
| 2018 to May 2024 window | 5.23 | [4.02, 6.74] |

Epoch's published rates (sevilla2024training) are:
- 4.1× (notable models, 90% CI [3.7, 4.6]);
- 4.2× (frontier since about 2018, [3.6, 4.9]);
- 5.3× (frontier since 2010).

Our 2018–May 2024 estimate on today's vintage of the database is higher than their 4.2× for the same window, but the intervals overlap. Files: `ra3_econ_compute_growth.csv/.tex`.

### H2. Data-demand growth: the allocation exponent a is the object that matters, and it is not portable

**Estimated from Upstream technologies.** In the main-table set of ten technologies (designs, lab laws and lab-own ladders), a runs from **0.36** (Farseer's own form, local at 10²²–10²³ FLOP) to **0.57** (Gadre et al., RefinedWeb). Reference points:
- Chinchilla refit: 0.51 (κ = 1) and 0.50 (κ free);
- Chinchilla IsoFLOP minima: 0.50;
- Llama 3 (Meta's law): 0.46;
- DeepSeek LLM's published law: 0.52;
- OLMo ladder: 0.37;
- Marin (DCLM): 0.42.

At the estimated 5.06× compute growth, compute-optimal data demand grows as follows:

| Horizon | a = 0.57 (Gadre RW) | a = 0.36 (Farseer own form) | Ratio of extremes |
|---|---|---|---|
| Per year | 2.02× | 2.84× | |
| Over 5 years | 34× | 183× | 5.4 |
| Over 5 years, at 4.5× compute growth (R3's check) | 26× | 126× | 4.8 |

- R3's own [0.37, 0.57] range at 4.5× gives 25× vs 114×, which we reproduce.
- Chinchilla (κ = 1) gives 2.20× per year, 95% CI [2.05, 2.32] from its bootstrap.
- Kaplan's historical a = 0.73 would give 1.55× per year.
- **ra1's reviewed model-free Approach-2 path slopes** (added in review): OLS of ra1's per-budget model-free argmins on ln C over valid budgets. They give a = 0.50 (Chinchilla), **0.50 (Llama 3, 8 bracketed budgets; Meta's published law has 0.46)**, 0.35 / 0.42 / 0.43 (Marin Comma / DCLM / Nemotron-CC) and 0.50 / 0.51 (Porian RefinedWeb / OpenWebText2). These are growth-only rows in `ra3_econ_data_demand.csv`, with no level, because the designs stop at 2.6×10¹⁹–3×10²¹ FLOP. They leave the range essentially unchanged (0.35–0.57). They also show that even one lab's a moves by 0.04 with the choice of budgets.

σ\* does not enter D\*(C) at all.

The level is not portable either. Compute-optimal D\* at the September-2026 frontier compute ranges from **9T tokens** (Gadre RW, M\* = 0.55) to **519T** (OLMo ladder, M\* = 1,720). Chinchilla gives 48T, Llama 3 90T and DeepSeek 40T. Files: `ra3_econ_data_demand.csv`.

### H3. When would frontier runs reach the stock of high-quality public text?

**Literature inputs.** The stock is from villalobos2022run (the ICML 2024 paper):
- quality-adjusted unique stock 100T tokens in 2024, 95% CI [22T, 490T];
- repetition-adjusted "effective" stock 320T [65T, 1,700T];
- stock growth 0–10% per year; we use 5% as the central value.

Frontier compute follows the H1 trend.

**Compute-optimal frontier runs reach 100T unique tokens in 2027.9** under the Chinchilla refit. The Monte Carlo 5–95% range is [2026.1, 2030.2], combining technology bootstrap draws, WCB trend draws, a lognormal stock and g_S ~ U[0, 10%].

Across technologies the compute-optimal date spans **2025.2 (OLMo) to 2030.6 (Gadre RW)**. Here a and the non-portable level M\* move together.

- **At the observed over-training of disclosed frontier runs.** We use the median D/D\*(C) of the four disclosed dense runs above 10²⁵ FLOP in 2024–26: Llama 3.1 405B, Nemotron-4 340B, Pangu Ultra and Aramco Metabrain. This is 1.43 under Chinchilla, i.e. M/M\* = 2.0.
  - The review dropped LongCat-2.0, a MoE model whose N is active parameters (M = 729), so its D/D\* under a dense technology is not comparable. This follows R1 minor 24 and ra2's clean-sample rule.
  - Including it, as the builder did, gives 1.55 and moves every date about 0.1 year earlier (`*_withMoE` columns).

  Anchoring the level to observed data use collapses the spread:
  - 100T is reached in **2026.6–2027.8** across all ten technologies (Chinchilla 2027.4, MC 5–95% [2025.7, 2029.4]);
  - the 320T effective stock is reached in **2027.8–2029.6** (Chinchilla 2029.0 [2027.1, 2031.1]).
  - Review fix: the MC now recomputes the anchoring multiple for each technology draw, so the anchored level stays at observed data use. The technology component of the Chinchilla interval shrinks from 1.7 to 0.45 years; OLMo's shrinks from 25 to 6 years.
- **Comparison with Villalobos et al.** They report a median full-utilisation year of 2028 for the effective stock, one year earlier with 5× over-training. Our effective-stock dates under 5× over-training are 2028.4 (Chinchilla) and 2027.5 (Llama 3). These agree within about a year.
- **What drives the uncertainty** (Chinchilla, unique stock, compute-optimal; width of the 5–95% interval when only that source varies):

  | Source | Width |
  |---|---|
  | Stock level | 3.5 years |
  | Technology sampling (a, G) | 1.8 years |
  | Compute trend | 0.9 years |
  | Stock growth | 0.45 years |

  Across technologies at the observed level, a alone moves the 100T date by only about 1.2 years, because the stock is only about 4× above current use. Its consequences grow with the horizon: the 5.4-fold gap in five-year demand. Files: `ra3_econ_data_demand.csv`, `ra3_econ_exhaustion_mc.csv`.

### H4. The data wall: σ\* barely matters at moderate scarcity; the repetition technology matters a lot

**Estimated.** The Chinchilla family (L = E + [A N^(−a₁) + B D′^(−b₁)]^κ) is combined with Muennighoff et al.'s (2023) effective-data model, D′ = U + U R\*_D (1 − e^(−R/R\*_D)), with R = D/U − 1. We checked R\*_D = 15.387756 and R\*_N = 5.309743 in muennighoff2023scaling, App. A.

Three technologies:
1. the κ = 1 refit (σ\* = 0.737);
2. the κ-free refit (σ\* = 0.701);
3. a σ\* = 0.60 member of the κ = 1 refit's observationally equivalent family (m7 Prop. 2(4)). It has the same expansion path and frontier (maximum relative deviation of the frontier 2×10⁻¹⁶), so it differs only in curvature.

Every object depends on r only; this was checked at 10²⁵ vs 10²⁷ FLOP, with differences below 3×10⁻⁷ (optimizer tolerance). Primary specification (D′ only, R\*_D = 15.4):

| | κ = 1 (0.74) | κ free (0.70) | σ\* = 0.60 |
|---|---|---|---|
| Extra compute to reach the unconstrained loss, r = 2 | 1.5% | 1.6% | 1.6% |
| r = 4 | **7.2%** | **7.4%** | **7.7%** |
| r = 8 | 21% | 22% | 24% |
| r = 16 | **58%** | **63%** | **77%** |
| r = 32 | 187% | 233% | 484% |
| Hard wall r_max (unconstrained loss unattainable beyond) | **117** | **83** | **47** |
| Shadow value of a unique token, r = 4 (units of 6N) | 0.43 | 0.45 | 0.51 |
| Shadow value, r = 16 | 7.1 | 8.6 | 13.8 |
| σ_CU (compute vs unique data), r = 4 | 0.49 | 0.48 | 0.47 |
| γ_eff/γ (returns to compute behind the wall), r = 16 | 0.82 | 0.80 | 0.78 |

- **Sampling uncertainty is small.** It comes from the technology bootstraps (m1 pairs B = 400; m2 κ-free B = 200). The extra compute at r = 4 is [7.16, 7.29]% (κ = 1) and [7.29, 7.50]% (κ free); at r = 16 it is [57.0, 60.4]% and [60.6, 67.5]%.
- **The κ-free refit and the σ\* = 0.70 equivalent member give the same answers** to two digits (7.4% and 63%, r_max 83 vs 84). Curvature is what distinguishes κ = 1 from κ free here; the small differences in path and frontier do not matter.
- **Answer to R3 M7(2).** The 0.74 vs 0.70 difference barely moves the cost of a binding cap at r ≤ 16: 7.2 vs 7.4% at r = 4, and 58 vs 63% at r = 16. It matters only near the hard wall (r_max 117 vs 83) and when σ\* is as low as 0.60. That is informative.
- **Where M\* enters (added in review).** Every wall object is a function of r = D\*(C)/U, and D\*(C) carries the non-portable level M\*. Take the September-2026 fitted frontier (9.4×10²⁶ FLOP) and Villalobos's 100T unique stock (2024 level):
  - compute-optimal r ranges from **0.09** (Gadre RW) to **5.2** (OLMo) across the ten Panel A technologies; Chinchilla gives 0.48 and Llama 3 0.90;
  - so whether the wall already binds at today's frontier is decided by M\*, not by σ\*;
  - anchored at observed data use, frontier runs process 0.56–1.24 times the unique stock, i.e. they sit at the threshold (`ra3_econ_data_demand.csv`: D\*_now × data multiple).
- **The repetition model matters far more than σ\*** (κ = 1, r = 4):

  | Repetition model | Extra compute | r_max |
  |---|---|---|
  | D′ only, R\*_D = 15.4 (primary) | 7.2% | 117 |
  | Muennighoff's full model: excess parameters also decay (R\*_N = 5.3) | 43% | 9.1 |
  | Their "data-only decay" fit (R\*_D = 2.9) | 36% | 28 |
  | No repetition at all | 330% | 7.1 |

  **Review caveat: the primary specification is a hybrid that Muennighoff et al. never fitted.** Their R\*_D = 15.4 comes from a joint fit in which excess parameters also decay (R\*_N = 5.3). When they fit data decay alone, R\*_D is 2.9 (their Table 1: R² 0.735 vs 0.772 for the full model). Using 15.4 without parameter decay is therefore the most favourable case, not a point estimate.
  - Under their preferred full model the cost at r = 4 is 43%, not 7%, and the frontier loss becomes unattainable beyond r ≈ 9.
  - The paper should present the D′-only and full-model numbers together, as a range conditional on the repetition model, and not "about 7%" alone.
  - The σ\* comparison (the question R3 asked) is unaffected: under the full model it is 43.4% (κ = 1) vs 40.3% (κ free) at r = 4, with r_max 9.1 vs 9.5.

  The hard wall has a closed form, verified numerically:
  - D′ only: r_max = (1 + R\*_D)(1 − a)^(−1/(aS)), with S = 2/σ\* − 2;
  - D′ and N′: r_max = [(b₁/S)(1 + R\*_N)^(−a₁) + (a₁/S)(1 + R\*_D)^(−b₁)]^(−1/b₁).
- **Dollars.** At the stated price of **$1.0×10⁻¹⁸ per FLOP** (median amortized hardware-plus-energy cost of 2024–25 frontier runs in Epoch's frontier file; cloud rental about $3.8×10⁻¹⁸):
  - at C = 10²⁶ and r = 4, a unique token is worth 3.1×10¹² FLOP, i.e. **$3.2 per million unique tokens** ($12 at cloud prices);
  - at r = 16 it is worth $88 per million.
  - A 10²⁷-FLOP run with 10T unique tokens (r = 4.9) needs +10.2% compute (1.0×10²⁶ FLOP, about $0.10 billion amortized or $0.39 billion cloud), and the shadow value is $17 per million tokens.
  - With 30T unique tokens (r = 1.6) the same run needs +0.8% ($1.1 per million).
  - At 10²⁸ FLOP with 10T unique tokens (r = 15), it needs +54% ($5.4 billion amortized).

  Files: `ra3_econ_wall_*.csv`, `ra3_econ_checks.csv`.
- **R1 6(e): the elasticity that matters for the data wall is σ_CU, not σ_ND.**
  - σ_CU is the elasticity of substitution between compute and *unique* data along an isoquant. Under D′-only repetition it is **0.28–0.58 for r = 1.5–16** (κ = 1), rising with scarcity. It is **nearly invariant to σ_ND**: 0.53 vs 0.58 at r = 16 for σ\* = 0.60 vs 0.74.
  - With R\*_D = 2.9 it reaches 0.70 at r = 16.
  - σ_CU is governed by the repetition technology, which this paper does not estimate.

### H5. Where σ\* does matter: translating inference demand into data demand

**Estimated (closed form, verified numerically).** At given compute, a model with wedge w uses D/D\*(C) = w^(σ\*/(2(1−σ\*))) times the compute-optimal data.

| Wedge | σ\* = 0.74 | σ\* = 0.70 | σ\* = 0.60 |
|---|---|---|---|
| w = 3 (s = 0.67) | 4.65 | 3.62 | 2.28 |
| w = 2.15 (ra2's 2024 aggregate 1 + m; s = 0.54) | 2.92 | 2.45 | 1.78 |
| w = 5.53 (ra2's 2025 aggregate 1 + m; s = 0.82) | 10.9 | 7.4 | 3.6 |

- For w between 1.5 and 5.5, the same planned inference share implies 30–200% more data under σ\* = 0.74 than under 0.60.
- Forecasts that start from observed M do not depend on σ\*. Forecasts that start from projected inference demand do.
- The aggregate rows are illustrative: they describe a model whose wedge equals the compute-weighted aggregate multiple. Because the map is convex, the mean of per-model data multiples exceeds this value.

File: `ra3_econ_wedge_data_multiple.csv`.

### H6. The planned inference share of lifetime compute rises from about 0.04 to about 0.82 (module ra2's final inversion)

**Upstream wedges: module ra2_wedge, final and reviewed.** Reference technology: Chinchilla refit with κ free (σ\* = 0.70), ra2's reference.
- Aggregate: s = Σ(w⁺ − 1)C / Σw⁺C with w⁺ = max(w, 1), i.e. compute-weighted with planned inference truncated at 0. This is ra2's definition.
- Universe: ra2's **clean open-weight production-scale universe** (6ND ≥ 10²¹; confident N and D; no MoE, non-transformer or ra2-excluded models). It has 141 models, 2019–2025, and is rebuilt with ra2's own sample code.
- 95% CI: ra2's design-conditional wild bootstrap of the reference technology (Rademacher, B = 399). It is recomputed with ra2's function and seeds, and the draws are identical to ra2's.
- Every ra2 aggregate is reproduced exactly (`ra3_econ_inference_share_checks.csv`, 16 checks, maximum difference 2×10⁻¹⁶).

| Release period | Models | s [95% CI] | s under κ = 1 | Clean verified sample: models, s | Range over 32 ex-ante technologies |
|---|---|---|---|---|---|
| 2019–2022 | 8 | **0.04** [0.03, 0.05] | 0.03 | — | — |
| 2023 | 53 | **0.35** [0.25, 0.46] | 0.33 | 24, 0.28 | [0.01, 0.58] |
| 2024 | 56 | **0.54** [0.41, 0.64] | 0.50 | 40, 0.58 | [0.05, 0.74] |
| 2025 | 24 | **0.82** [0.76, 0.86] | 0.77 | 13, 0.83 | [0.24, 0.96] |
| All 2019–2025 | 141 | **0.64** [0.54, 0.72] | 0.59 | 77, 0.66 | [0.10, 0.86] |

- **The technology range is the dominant uncertainty.** The sampling CI is 3–7 times narrower than the range over technologies.
  - The range is "technology-consistent": the aggregate is recomputed under each of ra2's 32 ex-ante technologies. The low end is MiniCPM's published law (2023, 2024) or the κ = 1 Marin Comma ladder (2025). The high end is Gadre RW, κ = 1 (2023) or Muennighoff κ-free (2024, 2025).
  - ra2's own per-model envelope, in which each model sits at its own minimum or maximum over technologies, is wider: [0.01, 0.60], [0.04, 0.81] and [0.21, 0.96]. It is an outer bound, not the aggregate under any single technology.
  - The rising trend holds under every technology. The level does not.
- **Curvature matters little for the aggregate.** κ = 1 vs κ free changes s by 0.01–0.05.
- **Bootstrap scheme.** m2's pairs draws (B = 200) give wider upper tails than ra2's wild bootstrap: 2023 [0.26, 0.52], 2024 [0.43, 0.71], 2025 [0.76, 0.90]. The wild bootstrap is primary because it is design-conditional, as in ra2. Either way, sampling uncertainty is small next to the technology range.
- **Leave-one-out.** Llama 3.1 405B is 39% of 2024 compute. Without it the 2024 share is 0.63 (0.73 on the clean verified sample). The share is dominated by a handful of large releases (top model 22–46% of compute per period).
- **Other checks.**
  - ra2's universe contains no duplicate pretraining runs (same lab, N, D), so deduplication changes nothing.
  - 2019–2022 has 8 models. Two under-trained models, OPT-175B and GLM-130B (M ≈ 2–3), make up 92% of its compute; the three 2019–2021 singletons have s = 0.55–0.84 but negligible compute.
  - The universe has no 2026 releases. ra2's filters remove the four 2026 models in m3's universe: two MoE (LongCat-2.0, MiMo-V2.5-Pro) and two with non-confident N or D (Granite 4.1).
- **Superseded builder version.** m3's wider universe (271 models, κ = 1 reference, 2019–2026) gave 0.07 / 0.35 / 0.54 / 0.76 for 2019–22 / 2023 / 2024 / 2025. It is kept only in `ra3_econ_inference_share_m3universe.csv`. The ra2-based numbers are within 0.06 of it for 2023–2025.

### H7. External disclosures: raw agreement in order of magnitude, but the objects differ

**Literature** (fleet flows; each checked against the primary text or the Epoch page on 2026-09-24):

| Source | Inference share |
|---|---|
| Google, 2019–21 ML energy (patterson2022carbon) | about 3/5 |
| Meta AI power capacity, experimentation:training:inference = 10:20:70 (wu2022sustainable) | 0.70 (0.78 excluding experimentation) |
| Meta's production language model | 0.65 |
| Nvidia data-center revenue, FY2024 (nvidia2024q4call; CFO remarks on the earnings call; the figure is not in the 10-K or the CFO commentary) | about 0.40 |
| OpenAI 2024 (press-reported via Epoch, you2025openai; not a firm disclosure) | 0.26 of all compute spend; 0.79 against final training runs only |

- **Raw comparison.** The planned s of 0.54 (2024) and 0.82 (2025) is in the same range as these figures (0.40–0.79).
- **The objects differ.** s is the planned lifetime share relative to a model's *own final run*. Fleet shares mix vintages and include R&D compute. Mapping s to a fleet flow gives s_fleet = p·m·φ/(p·m·φ + ρ), where:
  - m = w − 1 = s/(1 − s) is the planned inference multiple: 1.15 (2024), 4.53 (2025), 1.78 (2019–2025);
  - φ = (1 − e^(−gL))/(gL) is the vintage factor: 0.49 for a one-year serving life at 5.06× growth;
  - ρ is the R&D multiple: final runs are 9.6–22.6% of R&D compute (denain2026final), so ρ = 4.4–10.4;
  - p is the relative price of an inference FLOP.
- **Implied fleet shares (L = 1, frontier vintage growth).**
  - For the 2025 multiple (m = 4.53) they are 0.69 (ρ = 1), 0.34 (ρ = 4.4) and 0.18 (ρ = 10.4) at p = 1; with p = 2 they are 0.82, 0.50 and 0.30.
  - For 2024 (m = 1.15) they are 0.36, 0.11 and 0.05 at p = 1.
  - Aggregate training compute probably grows more slowly than the frontier. At an illustrative 2× per year, the 2025 shares rise to 0.77 / 0.42 / 0.24 (`ra3_econ_inference_share_flow.csv`).
- **Verdict.** Order-of-magnitude agreement holds only if R&D compute is small or inference FLOPs are dear. With realistic R&D multiples, open-weight developers' planned inference implies fleet shares at or below the Google and Meta figures, and far below them for 2024 releases. This is consistent with those fleets being dominated by closed-model serving and recommender inference. It is not a sharp validation, and should be reported as a disagreement once R&D compute is counted.
- **No quantitative 2025–26 disclosure.** A review search (2026-09-24) found only qualitative provider statements, such as Nvidia's remarks that inference has overtaken training; none gives a verifiable share.

Files: `ra3_econ_inference_share*.csv`, `ra3_econ_inference_disclosures.csv/.tex`.

### H8. Growth-model calibration

**Estimated from Upstream γ.** Halving reducible loss requires 2^(1/γ) times the compute:

| Technology | γ | 2^(1/γ) [95% CI] |
|---|---|---|
| Chinchilla refit, κ = 1 | 0.178 | **49×** [39, 63] |
| κ free | 0.165 | 66× [50, 90] |
| Chinchilla IsoFLOP frontier | 0.168 | 62× [44, 91] |
| Hoffmann A3 | 0.155 | 88× |
| Farseer | 0.148 | 109× [66, 209] |
| Llama 3 (A3) | 0.149 | 104× |
| Llama 3 (A1) | 0.136 | 161× |

- **The range is 1.7–2.2 orders of magnitude per halving.** That is 2.4–3.1 years of frontier compute growth at 5.06× per year, or 1.5–1.9 years if algorithmic progress adds Ho et al.'s 8.4-month doubling of effective compute (2.69× per year; m5 shows this rate is fragile).
- **Behind a data wall the returns to compute fall.** γ_eff/γ = 0.95 at r = 4, 0.82 at r = 16 and 0.71 at r = 32 (κ = 1). Under σ\* = 0.60 the ratio at r = 32 is 0.61. This quantifies R3 minor 29: σ < 1 does lower the return to compute once unique data bind, but modestly until r ≈ 16.
- **GATE's training–inference trade-off slope.** GATE (erdil2025gate) uses m ≈ 1–2: 1–2 orders of magnitude of inference substitute for 1 of training, following villalobos2023trading. In our technology, over-training at wedge w trades inference for training at m = 1/(w − 1):
  - m = 1–2 corresponds to w = 1.5–2 (s = 0.33–0.5);
  - at ra2's aggregates (w = 1 + m) the local slope is m = 0.87 for 2024 releases, 0.22 for 2025 and 0.56 pooled over 2019–2025;
  - the mapping is an analogy: GATE's m concerns runtime compute per task, ours the per-token serving cost of a smaller model at equal loss.

---

## 2. Methods

**Frontier compute (growth.py).**
- Data: Epoch AI `all_ai_models.csv`, snapshot 2026-09-23. sha256 is recorded in `code/data/download_ra3_econ.sh`.
- Running top-10 flag: a model is flagged if fewer than 10 earlier-or-same-date models have larger training compute.
- Regression: OLS of log₁₀C on release date centred at 2024.
- Inference: CR1 standard errors clustered by developer (first listed organization, parent-mapped as in m3).
- Confidence intervals: wild cluster bootstrap-t, unrestricted residuals (WCU), Webb six-point weights, B = 9,999.
- p-values: restricted wild cluster bootstrap (WCR) for H₀: growth = 4.2× and 4.5×; floor 10⁻⁴.
- The WCU coefficient draws feed the exhaustion-date Monte Carlo.

**Data demand (demand.py).**
- Each technology is reduced to ln D\*(C) = c₀ + (1 − a) ln C.
- Sources:
  - m1/m2/m3 registries and bootstrap draws;
  - m1's Approach-2 minima;
  - Farseer Eq. 3 via m2's tabulated D\*(C) (piecewise log-linear, extrapolated beyond 3.5×10²¹ FLOP);
  - DeepSeek LLM's published law D_opt = 5.8316 C^0.4757 (checked in bi2024deepseek, eq. 4);
  - Kaplan (growth only).
- Growth of data demand is g_C^(1−a).
- "Observed over-training" multiplies D\* by the median D_i/D\*(C_i) = √(M_i/M\*(C_i)) of the disclosed **dense** runs above 10²⁵ FLOP in 2024–26 (m3's verified inputs; n = 4 after the review excluded the MoE run). This preserves the growth rate.
- Exhaustion year: the root of m_D D\*(C_F(t)) = S₀(1 + g_S)^(t−2024).
- Monte Carlo: N = 20,000 draws. Sources are varied jointly and one at a time. In the observed-M scenario, m_D is recomputed for every technology draw (review fix).
- ra1 path slopes: OLS of ln N\* on ln C over ra1's valid per-budget model-free argmins (`ra1_modelfree_isoflop_budgets.csv`). These give growth only, with no level.

**Data wall (wall.py).**
- Dual problem: min 6ND s.t. L(N′, D′(D; U)) ≤ ℓ\*(C). The search is a 3,001-point grid on ln N over 30 log-units, refined by bounded Brent.
- Primal problem: min over N of L at fixed C.
- Shadow value: central difference of the dual in ln U at fixed loss. It matches the envelope-theorem closed form λ = 6N[(1 + R\*)(e^(R/R\*) − 1) − R] to 10⁻⁶.
- σ_CU: finite differences of ln(C/U) against ln λ along the isoquant.
- γ_eff: finite difference of ln(ℓ_U − E) in ln C.
- Equivalent family members: a₁ = (1 − a)S, b₁ = aS, κ = γ/(a(1 − a)S), with A and B solved from (G, K). The frontier match to the κ = 1 refit is exact (maximum relative deviation 2×10⁻¹⁶).
- Specifications:
  - D′ only, R\*_D = 15.4 (primary; the task's specification);
  - D′ and N′ (Muennighoff's full model; U_N = the technology's compute-optimal N for data U, their eq. 17);
  - D′ only with R\*_D = 2.9 (their data-only fit, Table 1);
  - no repetition.
- Dollar price: median of Epoch's "Training compute cost (2023 USD)"/C over 2024–25 frontier runs rated Confident or Likely (n = 6), which is $1.02×10⁻¹⁸ per FLOP. Cloud: n = 3, $3.77×10⁻¹⁸.

**Inference share (share.py; rebuilt in review on module ra2's final outputs).**
- Universe: ra2's clean open-weight production-scale universe, rebuilt by calling ra2's `sample.build()` and `sample.universe()` read-only. ra2's audit-file write is redirected to `data/processed/ra3_econ/ra2_sample_rebuild/`. We then apply the filter of ra2's `open_closed` (confident N and D; no MoE; no non-transformer; ra2-excluded rows dropped) and keep open weights.
- Clean verified sample: `output/tables/ra2_wedge_models.csv` (clean = True; w under all 32 ex-ante technologies from `ra2_wedge_technologies.csv`).
- Reference technology: ra2's κ-free Chinchilla point.
- Draws: ra2's `techs.chin_q_wild(399)`, which is deterministic and seeded, run on 5 processes. They are identical to ra2's draws: per-model intervals match ra2's `wlo/whi_chin_q` to 4×10⁻¹⁶. m2's pairs draws (B = 200) are a check (`s_ref_pairs_lo/hi`).
- κ = 1 comparison: m3's reference with m3's pairs draws.
- Formula: ra2's s = Σ(w⁺ − 1)C / Σw⁺C with w⁺ = max(w, 1).
- Technology range: the aggregate under each ex-ante technology, then min and max. ra2's per-model envelope is also reported.
- Checks: ra2's 16 aggregate numbers are reproduced exactly (`ra3_econ_inference_share_checks.csv`); `run.py` stops if any check fails.

**Growth calibration.** γ and its standard errors come from the m1/m2 registries and m1's duality objects. The interval for 2^(1/γ) uses γ ± 1.96 SE.

---

## 3. Inventory

**Code.** `code/analysis/ra3_econ/`:
- `run.py`;
- `ra3common.py`: paths, seeds, verified external parameters with bib keys;
- `growth.py`, `demand.py`, `wall.py`, `share.py`, `exhibits.py`.

m3's `technologies.py` and `common.py`, and ra2's `sample.py` and `techs.py`, are imported read-only. `sl.py` is not modified. The download and verification record is `code/data/download_ra3_econ.sh`; its PDFs are in `data/raw/ra3_econ_lit/`.

**Paper exhibits.**

| File | Content |
|---|---|
| `output/tables/ra3_econ_table.tex` | **Paper table**: "Economic Implications under Alternative Technologies". Panel A: data demand and a, 10 technologies. Panel B: data wall and σ\*, 4 technologies plus 2 repetition-model rows. Panel C: planned inference share by period from module ra2 (clean universe with wild-bootstrap CI and κ = 1 comparison; clean verified sample with the range over 32 technologies). The notes carry the γ calibration. |
| `output/figures/ra3_econ_figure.pdf/.png` | **Paper figure**. (a) Frontier data demand 2020–2032 under a = 0.36, 0.46, 0.51 and 0.57, anchored at observed frontier data use, against the unique and effective stocks, with disclosed runs above 10²⁴ FLOP. (b) Extra compute vs scarcity r for σ\* = 0.74, 0.70 and 0.60, plus two repetition-model variants, with r_max marked. (c) Shadow value of a unique token vs r. |

**Appendix exhibits.**

| File | Content |
|---|---|
| `ra3_econ_compute_growth.tex/.csv` | Six frontier definitions, with WCB CIs and p-values. |
| `ra3_econ_inference_disclosures.tex/.csv` | Disclosures vs implied fleet shares. |
| `output/figures/ra3_econ_inference_share.pdf/.png` | s by period (ra2 universe, κ-free with CI and κ = 1), the range over ra2's 32 technologies, and disclosure lines. |

**CSVs in `output/tables/`.**

| File | Content |
|---|---|
| `ra3_econ_data_demand.csv` | 30 technologies (23 with levels or growth, plus 7 ra1 model-free path slopes): a, CI, growth at 4/4.5/5/5.06×, D\*, M\*, data multiple, exhaustion years, and with-MoE anchoring sensitivity. |
| `ra3_econ_exhaustion_mc.csv` | Monte Carlo quantiles by source, for 6 technologies × 2 stocks × 2 scenarios. |
| `ra3_econ_compute_growth_epoch_published.csv` | Epoch's published growth rates. |
| `ra3_econ_wall_technologies.csv` | Parameters of the wall technologies. |
| `ra3_econ_wall_rmax.csv` | r_max, 8 curvatures × 4 specifications. |
| `ra3_econ_wall_sigma_CU.csv` | σ_CU. |
| `ra3_econ_wall_boot_summary.csv` | Bootstrap intervals for the wall objects. |
| `ra3_econ_wall_absolute.csv` | C = 10²⁵–10²⁸ × U = 10T/30T/100T, in FLOP and $. |
| `ra3_econ_wall_checks.csv`, `ra3_econ_checks.csv` | Scale invariance; closed-form vs numeric shadow value; r_max; wedge multiple; equivalence. |
| `ra3_econ_usd_per_flop.csv` | Cost per FLOP of frontier runs. |
| `ra3_econ_wedge_data_multiple.csv` | Data multiple implied by a wedge. |
| `ra3_econ_inference_share.csv` | Panel C source: universe and clean-sample shares by period. |
| `ra3_econ_inference_share_universe.csv`, `_clean.csv`, `_by_tech.csv`, `_dedup.csv` | Components: wild and pairs CIs, κ = 1, leave-one-out, the 32 technologies, and the ra2 envelope. |
| `ra3_econ_inference_share_checks.csv` | Exact reproduction of `ra2_wedge_aggregate.csv`. |
| `ra3_econ_inference_share_flow.csv` | Fleet mapping over m, g, L, ρ and p. |
| `ra3_econ_inference_share_m3universe.csv` | Superseded builder version (m3 universe, κ = 1). |
| `ra3_econ_growth_calibration.csv` | Growth calibration. |
| `ra3_econ_gate_tradeoff_slope.csv` | GATE trade-off slope. |

**Processed data in `data/processed/ra3_econ/`.** `frontier_top10_2018_2026.csv`, `frontier_runs_disclosed_D.csv`, `wcb_trend_draws_top10.npy`, `data_demand_paths.csv`, `wall_grid.csv` (the full r grid for 8 curvatures × 4 specifications), `wall_boot.csv`, `summary.json`, and `ra2_sample_rebuild/audit_flags.csv` (a byproduct of calling ra2's sample code).

**Bib.** `lit/bib/extra_ra3_econ.bib` holds the new keys, each verified:
- patterson2022carbon (DOI via Crossref);
- wu2022sustainable (MLSys proceedings page);
- nvidia2024q4call;
- you2025openai;
- denain2026final;
- reddit2024s1 (SEC).

Existing keys used: villalobos2022run (= Villalobos et al. 2024 ICML), muennighoff2023scaling, sevilla2024training, jones2020nonrivalry, farboodi2021model, erdil2025gate, villalobos2023trading, ho2024algorithmic, bi2024deepseek, kaplan2020scaling, grattafiori2024llama, li2025predictableb, gadre2024language, groeneveld2024olmo, marin2026ladders, epochai2026data, webb2023reworking, demirer2025emerging.

---

## 4. Claims for the paper (evidence; caveat)

1. **"Frontier training compute has grown about fivefold per year since 2018 (95% CI 4.3–5.9)."**
   - *Evidence:* H1; `ra3_econ_compute_growth.csv`.
   - *Caveat:* compute for closed models is Epoch's estimate, and 19 of the 92 runs are "Speculative". Without them the rate is 4.8×.
2. **"Because compute-optimal data demand scales as C^(1−a), the least portable technology object, a, governs data-demand forecasts. Across designs and lab laws, a ∈ [0.36, 0.57] implies data demand growing 2.0–2.8× a year, a 34- to 183-fold increase over five years. The curvature σ\* does not enter."**
   - *Evidence:* H2; Panel A.
   - *Caveat:* a is extrapolated from designs below about 10²² FLOP. Farseer's own form is non-homothetic, and its local a comes from an extrapolated segment.
3. **"Anchored at the data use of disclosed frontier runs, frontier training reaches the quality-adjusted stock of public text (about 100T tokens) in 2026.6–2027.8 and Villalobos et al.'s repetition-adjusted stock in 2027.8–2029.6, close to their 2028 median."**
   - *Evidence:* H3.
   - *Caveat:* the stock's 95% interval spans a factor of 22 and is the largest source of uncertainty (5–95% width 3.5 years). Only four disclosed dense runs above 10²⁵ FLOP anchor the level. Closed frontier models do not disclose D.
4. **"The cost of a binding cap on unique tokens depends mainly on how fast repeated data lose value, and hardly on σ\*. At four times as much compute-optimal data as unique tokens, reaching the unconstrained loss costs 7% more compute if only data decay (R\*_D = 15.4), but 43% under Muennighoff et al.'s full model, in which excess parameters also lose value. Behind that model, the frontier loss becomes unattainable beyond about nine times. The difference between σ\* = 0.74 and 0.70 barely matters under either model (7.2 vs 7.4%; 43 vs 40%)."**
   - *Evidence:* H4; Panel B; figure panel (b).
   - *Caveat:* the 7% case is the most favourable one. It combines the jointly fitted R\*_D with no parameter decay, which Muennighoff et al. did not fit; their data-only fit has R\*_D = 2.9 (36%). Their R\* values come from models up to 9B parameters. The loss is validation loss on the original distribution, and data quality is held fixed. Do not quote "about 7%" without the full-model number.
5. **"The shadow value of a unique token at 10²⁶ FLOP and r = 4 is about 0.4 of the cost of processing one token, roughly $3 per million tokens at $10⁻¹⁸ per FLOP. It rises to about $90 per million at r = 16, and it is zero while r ≤ 1."**
   - *Evidence:* H4.
   - *Caveat:* this is the private value to one run. Because data are nonrival (jones2020nonrivalry), the social value of a token used by n labs is up to n times larger. The FLOP price is an amortized hardware-plus-energy median (cloud prices are 3.7× higher). We make no per-token comparison with licensing prices, because licensed corpus sizes are not disclosed (e.g. Reddit's $203 million of contracts, reddit2024s1).
6. **"The elasticity that matters for the data wall, between compute and unique tokens, is 0.3–0.6. It is set by the repetition technology, not by σ between parameters and tokens processed."**
   - *Evidence:* H4, `ra3_econ_wall_sigma_CU.csv`.
   - *Caveat:* it is model-implied (our technology × Muennighoff), not estimated.
7. **"σ\* governs how a planned inference share translates into data demand. At w = 3, compute-matched data demand is 4.7 times the compute-optimal level under σ\* = 0.74 but 2.3 times under 0.60."**
   - *Evidence:* H5.
   - *Caveat:* this assumes w is realized through over-training at fixed compute, not through data constraints. Holding w fixed while varying σ\* is a thought experiment: the paper's w are themselves inferred from observed M with a given σ\*.
8. **"Under the reference technology, the planned inference share of lifetime compute of open-weight releases rose from under 0.1 (2019–22) to about 0.8 (2025; 95% CI 0.76–0.86). That is in the range of fleet disclosures (0.4–0.7). Accounting for R&D compute and vintage growth, the implied fleet share is lower (0.18–0.50 for 2025 releases at a one-year serving life, R&D multiples 4.4–10.4), so the agreement is loose."**
   - *Evidence:* H6–H7; Panel C (module ra2's final inversion).
   - *Caveat:* technology dependence dominates. Across ra2's 32 ex-ante technologies the 2025 aggregate ranges over [0.24, 0.96]; the rising trend holds under every technology, the level does not. The ecosystem is open-weight only, and the disclosures are heterogeneous objects (energy, capacity, revenue, press-reported spend). One interval per sentence in the main text: quote the CI or the technology range, not both.
9. **"Halving reducible loss requires 50–160 times more compute (2^(1/γ), γ = 0.14–0.18): 2.4–3.1 years of frontier compute growth. Behind a data wall at r = 16 the compute elasticity of loss falls to about 0.8γ."**
   - *Evidence:* H8.
   - *Caveat:* loss is not capability. GATE-type models map effective compute to automation through FLOP thresholds, not loss.

A draft sentence for R1 6(e) in Section V: "Our σ is the substitution between parameters and tokens *processed*. The data wall concerns substitution between compute and *unique* tokens, which combines σ with the returns to repetition. Under Muennighoff et al.'s repetition estimates the latter elasticity is 0.3–0.6 and nearly independent of σ."

---

## 5. Robustness and failures

- **Compute growth.** The estimate is stable across frontier definitions (4.8–5.2×), except the language-only top-10 (6.45×). Language models rose into the all-domain frontier over the sample, so that rate mixes composition with growth. The record-runs sample has 6 developer clusters, so its WCB interval is indicative only. Our 2018–May 2024 rate (5.23×) exceeds Epoch's published 4.2× for that window. This could be a database-vintage effect, since Epoch revised and added compute estimates after 2024; it was not investigated further.
- **Exhaustion dates** are dominated by the stock. The OLMo ladder's a has a very wide bootstrap interval ([0.11, 0.82]), which makes its compute-optimal MC dates uninformative (5–95%: 2022.7–2042.8; anchored at observed M, 2025.3–2031.7). Units differ across technologies (tokenizers and N conventions), which shifts levels by about ±20%. Under the observed-M anchoring, levels are re-anchored to observed data, which removes most of this.
- **Data wall.**
  - Numerical: scale invariance holds to optimizer tolerance (below 3×10⁻⁷). The closed-form and numerical shadow values agree to 10⁻⁶. An independent re-implementation by the reviewer, which minimizes over ln D instead of ln N, reproduces the penalties at r = 2–32 for κ = 1 and κ free, the full-model, data-only and no-repetition rows, and the shadow value at r = 4, all to the reported digits. r_max is verified: penalties are finite at 0.98 r_max and infeasible at 1.02 r_max. Widening the dual search grid from 14 to 30 log-units left all tabulated values unchanged.
  - Substantive: results depend on the repetition model. D′-only with R\*_D = 15.4 is the most favourable case, and Muennighoff et al.'s own full fit (with N′) implies a hard wall at r ≈ 9.
  - Not modelled: synthetic data, multimodal data, data-quality differences, and the fact that frontier labs curate beyond the Villalobos quality filter.
- **Inference share.**
  - It is truncated at T ≥ 0. 2019–22 is dominated by under-trained models (OPT-175B, GLM-130B); untruncated, the multiple is negative. ra2's universe has no 2026 releases.
  - Source: ra2's final reviewed outputs (files of 07:10–07:12), reproduced exactly. The builder's preliminary m3/κ = 1 numbers (0.07 / 0.35 / 0.54 / 0.76) are superseded.
  - I found no verifiable 2025–26 *firm* disclosure of the inference vs training share of compute or energy by a major provider:
    - Google's 2025 serving paper measures per-prompt energy only;
    - Mistral's life-cycle assessment reports training and inference jointly;
    - Nvidia's 2025–26 releases say only that inference has overtaken training, with no share (review search, 2026-09-24).
  - The OpenAI 2024 figures are press-reported estimates (Epoch), not disclosures.
- **Kaplan.** a = 0.73 is not an optimum in the Chinchilla sense; it is reported as a historical reference only.

---

## 6. Referee comments addressed

| Comment | Response |
|---|---|
| **R3 M7 (1)**, data-demand growth and a | H1–H3; Panel A; figure (a). R3's numbers are reproduced (25× vs 114× at 4.5×, a ∈ [0.37, 0.57]) and extended: our own compute-growth estimate (5.06× [4.29, 5.93]), ten technologies including lab laws, exhaustion dates with Monte Carlo intervals, and anchoring at observed over-training. The identification result "a and M\* least portable" now carries economic content: a sets the growth rate (34× to 183× over five years) and M\* the level (9T to 519T at today's frontier). |
| **R3 M7 (2)**, σ\*, M\* and the data wall | H4–H5; Panel B; figure (b)–(c). Compute-equivalent cost of a cap and shadow value of a unique token, in FLOP and dollars, under κ = 1, κ free and σ\* = 0.60 (an equivalent member holding path and frontier fixed). "0.74 vs 0.70 barely moves the answer" is stated with numbers, under both the D′-only and the full Muennighoff model, together with where σ does matter (the r_max location and the inference-to-data mapping). M\* enters through r: at today's frontier, compute-optimal r against the 100T stock is 0.09–5.2 across technologies, so M\*, not σ\*, decides whether the wall already binds. Jones–Tonetti nonrivalry and data licensing are connected in Claim 5. Farboodi–Veldkamp (farboodi2021model) can be cited for data as a by-product of use; we did not model it. |
| **R3 M7 (3)**, inference share | H6–H7; Panel C; appendix table and figure. Now a measurement built on module ra2's final inversion rather than the "illustrative" aggregate: κ-free reference, clean samples, design-conditional CI, and the range over 32 ex-ante technologies. Expressed as the share s. Levels are compared with Patterson et al. (2022), Wu et al. (2022), Nvidia's FY2024 disclosure and press-reported OpenAI figures, with a formal mapping from planned to fleet shares. Honest verdict: raw order-of-magnitude agreement, and disagreement once R&D compute is counted. The technology range ([0.24, 0.96] in 2025) remains the binding caveat. The link to Demirer et al. (2025) on the inference market is only a citation here. |
| **R3 M7, final paragraph** (γ in a growth calibration) | H8: 2^(1/γ) = 49× (γ = 0.178) to 161×; years per halving; GATE's trade-off slope m = 1/(w − 1), which is 0.22–0.87 at ra2's 2024–2025 aggregates. |
| **R3 minor 29** ("σ < 1 … lowers the return to the other faster") | Quantified: γ_eff/γ = 0.95 / 0.82 / 0.71 at r = 4 / 16 / 32 (κ = 1), 0.61 at r = 32 under σ\* = 0.60. The effect is modest until scarcity is severe, and the repetition technology dominates. |
| **R1 6(e)** (σ for tokens processed vs unique data; which elasticity for growth models) | The data-wall elasticity σ_CU (compute vs unique data) is computed: 0.28–0.58, nearly invariant to σ_ND and set by repetition. The growth-model elasticity for effective-compute aggregators is γ, not σ. σ_ND enters only through the r_max location and the translation of inference demand into data demand. A draft sentence is in Section 4. |
| **R2 M3 / R3 M3(c)** (aggregate levels vs Patterson/Wu) | Partly addressed in H7. The main aggregate is ra2's; this module supplies the external comparison and the mapping. |
| **R2 23** (dedupe Llama 3/3.1; leave-one-out) | For the share: ra2's clean universe already has one row per pretraining run (deduplication on lab, N and D changes nothing). Leave-one-out is reported: Llama 3.1 405B is the most influential model in 2024 (s from 0.54 to 0.63 without it; 0.58 to 0.73 on the clean verified sample). |

---

## 7. Open issues

1. **ra2 merge: done in review.** Panel C is built on ra2's final outputs. If ra2 is re-run, rerun this module; `run.py` stops if ra2's aggregates are no longer reproduced. Writers: the primary sample for Section V is ra2's clean open-weight universe (141 models); the clean verified sample (77) carries the technology range.
2. **ra1 path slopes: done in review** (growth-only rows).
3. **Frontier levels.** Only four disclosed dense runs above 10²⁵ FLOP anchor the observed-M level. Closed frontier D is unknown.
4. **R\* at frontier scale is unknown.** The data-wall numbers should be presented as conditional on Muennighoff et al.'s estimates at ≤ 9B parameters.
5. **Epoch vintage.** Our 2018–May 2024 rate (5.2×) differs from Epoch's published 4.2×. A vintage comparison would need Epoch's 2024 snapshot, which we did not have.
6. **Missing 2025–26 disclosures.** No verifiable firm disclosure of inference vs training compute or energy shares exists for 2025–26. If one appears (for example, in a sustainability report), add it to `ra3common.DISCLOSURES`.
7. **Authorship caveat.** The paper is co-authored by Claude (Anthropic). No Anthropic-specific figures were used, and none should be introduced into this section.
