# Memo — module ra2_wedge: the re-specified inversion

Module owner: ra2_wedge (Claude). Date: 2026-09-24. Entry point: `code/analysis/ra2_wedge/run.py` (deterministic, fixed seeds; CPU only, at most 5 worker processes; 3.5–9 minutes depending on machine load; `--outputs-only` redraws tables and figures from `data/processed/ra2_wedge/outputs_cache.pkl`). Two consecutive full runs produced byte-identical CSV and .tex outputs (builder; confirmed by the reviewer before and after the review changes, 2.2–2.5 minutes per run). One-time data pull: `bash code/data/download_ra2_wedge.sh` (Hugging Face model-tree counts and model cards; arXiv PDFs of the papers whose numbers are used).

Inputs (read-only): m3_wedge's verified sample (`data/processed/m3_wedge/wedge_models_wide.csv`) and its technology loader; m1/m2 registries and bootstrap draws; m2's κ-family estimator; the open-athena IsoFLOP compilation (Llama 3, Marin, Chinchilla profiles); published laws verified from the papers (DeepSeek LLM, MiniCPM); module ra1_modelfree's reviewed outputs (`ra1_modelfree_isoflop_summary.csv`: model-free σ* of the Llama 3 and Marin designs; `ra1_modelfree_farseer_delta.csv` and `_sensitivity.csv`: Farseer model-free vs parametric wedge). `sl.py` was not edited.

**Independent review (2026-09-24; `output/memos/ra2_wedge_review.md`).** Numbers below are from the reviewed run. The review (i) replaced this module's own model-free curvature by ra1's reviewed estimate in the lab-own Meta and Marin technologies (the builder's loader looked for a file ra1 never wrote), (ii) fixed the serving-footprint coding of Llama 1 and Llama 2 (coded 1 in the data although the rule and this memo coded them 0), (iii) added ra1's Farseer evidence on the direction of the extrapolation error, (iv) corrected the convention claim for sub-1B models, and (v) added treated-cluster counts, a synthetic-data robustness row, the vintage of lab-own laws and a Meta path on ra1's bracketed budgets. Unaffected: the clean sample, the reference headline, the 32-technology dispersion ranges, family, PI shares, in-support splits, open/closed, tiers, validation and cost sensitivity.

**Notation.** w = ε_N/ε_D; M = D/N; C = 6ND; S = α+β (inner exponents in the κ family); 1/σ* − 1 = S/2. Within the Chinchilla and κ families, ln w = (S/2)·[ln M − ln M*(C)] exactly. Headline object (revision_plan A): the **expenditure share s = (w − 1)/w**. When the developer serves the model, s is the planned serving share of lifetime cost. In general, s = (marginal value of compactness at fixed quality)/(that value + training cost). s is negative when w < 1.

**Reference technology.** Chinchilla with the outer exponent κ free (Huber, n = 240; a₁ = 0.4244, b₁ = 0.4305, κ = 0.774, σ*_κ = 0.701). Our refit reproduces m2's κ-free row exactly. m1's `spec_kappa` values differ in the fourth decimal; with them the median over all 173 models is **3.779**, reproducing R4's recomputation of 3.78; with our refit it is 3.767. Uncertainty comes from a design-conditional wild bootstrap on log-loss residuals (Rademacher weights, i.e. Feng, He and Hu (2011) weights for median-type estimators; B = 399). m2's pairs draws (B = 200) are a check.

**Labels.** [E] = estimated here; [U] = upstream module; [L] = number taken from a paper and verified against the PDF; [A] = maintained assumption.

---------------------------------------------------------------------------------------------------

## 1. Headline findings

### H1. The clean inference-demand sample and the headline share [E]
Defined ex ante (Section 2.1). There are nine sequential steps from the 173 verified models to **77 models** (36 families, 18 developers, released 2023–2025). File: `output/tables/ra2_wedge_cleaning.csv|.tex`, Panel A.

| Step | n | Median w | Share w > 1 | All 32 points > 1 | Band > 1 | Median s |
|---|---|---|---|---|---|---|
| Verified sample (m3 Sample B) | 173 | 3.77 | 0.93 | 0.61 | 0.17 | 0.73 |
| (a) one observation per pretraining run (−8) | 165 | 3.79 | 0.93 | 0.62 | 0.18 | 0.74 |
| (b1) research suites (−45) | 120 | 4.20 | 0.97 | 0.70 | 0.16 | 0.76 |
| (b2) replications / dataset benchmarks (−8) | 112 | 4.46 | 0.98 | 0.72 | 0.17 | 0.78 |
| (c) MoE (−12) | 100 | 4.30 | 0.98 | 0.69 | 0.18 | 0.77 |
| (d) distilled / pruned / derived init. (−10) | 90 | 4.03 | 0.98 | 0.66 | 0.17 | 0.75 |
| (e) multimodal (−0 remaining) | 90 | 4.03 | 0.98 | 0.66 | 0.17 | 0.75 |
| (f) instruction-tuned only (−7) | 83 | 3.99 | 0.98 | 0.64 | 0.18 | 0.75 |
| (g) non-transformer (−0 remaining) | 83 | 3.99 | 0.98 | 0.64 | 0.18 | 0.75 |
| (h) D without primary source (−6) = **clean** | **77** | **3.99** | **0.97** | **0.64** | **0.18** | **0.75** |

- **Headline on the clean sample, reference technology.** Median ŵ = **3.99 [3.21, 4.86]**; median s = **0.749 [0.689, 0.794]**; share with ŵ > 1 = **0.974 [0.922, 1.000]**.
  - The brackets are the joint bootstrap distribution across technology draws (R1 9f): each draw of the technology recomputes the sample median and share. B = 399.
  - Under m3's κ = 1 reference on the same 77 models, the median is 3.34 (s = 0.70).
- **Robustness (not part of the ex-ante definition).** Dropping the 9 clean models trained mainly on teacher-generated synthetic data (phi-1.5, phi-2, SmolLM 1–3; R2 Major 10, minor 22) leaves 68 models: median ŵ 3.80, share > 1 0.97, band > 1 0.15, median s 0.74 (`ra2_wedge_cleaning.csv`, row `r_synthetic`).
- **Cleaning barely moves the median.** Steps (a)–(h) change the median ŵ by at most 0.69 (between 3.77 and 4.46). The exclusions mostly remove models with ŵ close to typical values; research suites carry lower wedges.
- **The robust sign is weaker than m3 reported.** Every one of the 32 ex-ante technologies puts the point estimate above 1 for 64% of the clean sample. The union of all 95% intervals lies above 1 for only **18%**. m3's six-technology band gave 79%.

### H2. Dispersion across technologies under an ex-ante rule is the dominant uncertainty [E/U/L]
32 technologies are in the ex-ante set (Section 2.2). File: `ra2_wedge_technologies.csv`; `ra2_wedge_cleaning.tex`, Panel B.

- **The range of medians is wide.** Median ŵ runs from **1.20** (MiniCPM's published law, M*(10²¹) = 192) to **6.87** (Muennighoff κ-free, σ*_κ = 0.511). Median s runs from **0.17 to 0.85**. The share with ŵ > 1 runs from **0.64 to 1.00**.
- **Technologies with a κ-free or model-free curvature cluster (median s 0.69–0.79).** Median s and bootstrap interval:

  | Technology | σ* | Median s [95%] |
  |---|---|---|
  | Chinchilla κ-free | 0.701 | 0.75 [0.69, 0.79] |
  | Farseer κ-free | 0.710 | 0.74 [0.73, 0.75] |
  | Llama 3 path + ra1 model-free curvature | 0.660 | 0.79 [0.72, 0.85] |
  | Marin (three corpora), ra1 model-free | 0.700–0.713 | 0.69–0.74 |
  | DeepSeek LLM law + reference curvature | 0.701 | 0.78 [0.75, 0.81] |

- **The low medians come from rejected or unidentified curvatures, or from high-M* laws.**
  - Marin A2/A1 under κ = 1 (σ* 0.87–0.90; m1 found Marin's frontier E and γ unidentified): median s 0.25–0.40.
  - OLMo ladder κ = 1 (σ* 0.825, M*(10²⁴) = 291): median s 0.24 [−1.22, 0.62].
  - MiniCPM (M* 192–426): median s 0.17.
- **Estimator variants on the Chinchilla data (sensitivity rows, not in the set):** NLS in levels 4.58; all 245 points 5.50; 9-cluster draws [1.83, 6.48] for the κ = 1 median.

### H3. Lab-own technologies (primary for that lab's models) [E/U/L]
File: `ra2_wedge_labown.csv`. n = 22: Meta 10, AI2 9, DeepSeek 2, Marin 1. Median s on this subsample: **0.61 under lab-own technologies** vs **0.65 under the reference** (figure `ra2_wedge_ecdf`, panel b). The lab law predates the model's planning (contemporaneous) for 11 of the 22 (Llama 3 herd, DeepSeek LLM, OLMo 2/3): median s 0.61 lab-own vs 0.67 reference. For the other 11 (Llama 1/2, OLMo 1/1.7, Marin 8B) the lab law is ex post (column `lab_vintage`; R1 minor 37).

**Meta: Meta's IsoFLOP path (10 budgets; reproduces D* = 0.299 C^0.537) with module ra1's model-free curvature.** σ* = **0.660 (s.e. 0.023)**, ra1's random-effects mean over the 8 bracketed budgets. The joint draws keep this module's path draws and rescale the curvature draws to ra1's s.e.

| Model | Lab-own w [95%] | s | Reference w |
|---|---|---|---|
| Llama 3 8B | **8.36 [5.67, 12.76]** | 0.88 | 6.73 |
| Llama 3 70B | 2.51 [2.09, 3.02] | 0.60 | 2.67 |
| Llama 3.1 405B | **0.97 [0.88, 1.08]** | −0.03 | 1.29 |
| Llama 2 70B (ex post) | 0.97 [0.91, 1.04] | −0.03 | 1.13 |

- The 405B lies on Meta's own training-optimal path. This is by construction: Meta sized it with this law (R1 minor 3).
- Across Meta's alternatives (κ = 1 law, primal, this module's raw and bias-corrected curvature, and the path fitted on ra1's 8 bracketed budgets), Llama 3 8B ranges from 3.09 to 12.16.
- **The path matters as much as the curvature.** ra1 finds the 3×10²¹ and 10²² profiles unbracketed (6 runs each). Fitted on the 8 bracketed budgets, the path slope is a = 0.501 instead of 0.463 and M*(10²⁴) is 15 instead of 31; Llama 3 8B's w becomes 12.2 and the 405B's 1.64 (sensitivity row `meta_mf8`). The primary keeps Meta's published law, because the lab-own technology is meant to be the law Meta planned with.
- With this module's own (bias-corrected) curvature, σ* = 0.678, the 8B is 7.07 and the 405B 0.97 (the builder's primary; row `meta_mfbc`).

**AI2: OLMo ladder, κ free.**
- OLMo-2 32B: 1.28 [0.96, 1.72]. Olmo-3 32B: 1.19 [0.90, 1.61].
- OLMo 1B (ex post): 19.2 [13.9, 33.2]. OLMo-2 1B: 20.2 [14.5, 35.2].
- The κ-free ladder has S/2 = 0.84, which makes small-model wedges extreme. Under the κ = 1 ladder the same models give 2.0.

**DeepSeek: published allocation law in its own units (non-embedding FLOPs per token), with the reference curvature.**
- DeepSeek-LLM 7B: 3.42 [3.05, 3.87]. 67B: 1.34 [1.31, 1.38].
- DeepSeek did not publish curvature; the interval reflects the reference curvature only.

**Marin: Nemotron-CC ladder path with ra1's model-free curvature (σ* = 0.705, s.e. 0.023).**
- Marin 8B: 5.08 [3.10, 8.71], s = 0.80; range over Marin's own alternatives 1.43–6.66.
- Marin's ladder (2026-03) postdates Marin 8B, so this is an ex-post technology.

### H4. Family-level token budgets [E; theory verified numerically]
Files: `ra2_wedge_family_split.csv`, `ra2_wedge_family_level_W.csv`, `ra2_wedge_families.csv`.

- **Classification on the clean sample.**

  | Class | Models | Families | Median ŵ | Share > 1 |
  |---|---|---|---|---|
  | Common D (max/min D ≤ 1.10) | 32 | 11 | 3.90 | 1.00 |
  | Size-specific D | 30 | 10 | 4.01 | 0.93 |
  | Singletons | 15 | 15 | 3.93 | 1.00 |

  - On all 173 models: common D 83 models in 23 families, median 3.04; size-specific 66 models in 20 families, median 3.94.
  - R1 counted 24 common-D families, using the m3 family labels and the Llama 3 / 3.1 split.
  - Excluding common-D families (the 45 models whose individual wedge is a member-level first-order condition under the maintained conduct): median ŵ 3.99, median s 0.75, share > 1 0.96, band > 1 0.22.
- **What a shared D reveals (derived here; ra5 formalizes).** Take a family that chooses one D for all members and sizes N_j freely, subject to loss targets. The D first-order condition is
  **Σ_j ω_j ℓ_j / w_j = 1**, where ω_j are training-compute shares and ℓ_j = 1 + T_j/(3D) is member j's lifetime/training ratio.
  - Individual w_j are therefore not revealed-preference objects. In one numerical example, w = 10.7 for a member with ℓ = 2.96.
  - The family reveals one number, **W_f = [Σ ω_j/w_j]⁻¹**. It equals the compute-weighted mean lifetime/training ratio when ℓ_j is uncorrelated with 1/w_j. It is a lower bound when planned demand is larger for the more over-trained (smaller) members, since L̄ = W_f(1 − Cov_ω(ℓ, 1/w)).
  - If sizes come from a tier menu (N_j not chosen), serving cost is sunk with respect to D, and the family's D reveals nothing about T.
  - The identity holds to |LHS − 1| ≤ 2.2×10⁻⁸ in three numerical family problems solved by brute force (`headline.json: family_foc_check`).
  - Equivalently, W_f = Σ_j π_j ℓ_j exactly, with π_j ∝ ω_j/w_j: the family reveals a weighted mean of lifetime/training ratios that puts less weight on the more over-trained members. This is ra5's Prop. A9 (compute-weighted harmonic mean w̄_H) in this module's notation.
- **Family-level shares W_f (s_f) for common-D families:**

  | Family | W_f | s_f |
  |---|---|---|
  | Qwen3 | 9.13 | 0.89 |
  | Granite-3.0 | 6.68 | 0.85 |
  | Qwen2.5 | 3.53 | 0.72 |
  | StableLM-alpha | 2.91 | 0.66 |
  | Apertus | 2.85 | 0.65 |
  | Olmo-3 | 2.70 | 0.63 |
  | Yi | 1.97 | 0.49 |
  | Llama-3 herd | 1.42 | 0.29 |
  | MPT | 1.31 | 0.24 |
  | Llama-2 | 1.29 | 0.23 |
  | DeepSeek-LLM | 1.22 | 0.18 |

  - Flagships dominate the weights, so W_f is far below the members' median wedge. Example: Llama-3 herd, members 1.30–6.73.

### H5. Parameter-count conventions [E/U]
File: `ra2_wedge_conventions.csv`. Clean sample (77); N_nonemb from config.json.

| Technology | Convention | Median ŵ | Median ŵ with head FLOPs |
|---|---|---|---|
| Chinchilla κ-free | total | 3.99 | |
| Chinchilla κ = 1 | total | 3.34 | |
| Farseer | total | 2.64 | |
| Chinchilla (m8 refit) | non-embedding | 3.65 | 3.75 |
| Farseer κ = 1 | non-embedding | 2.74 | 2.86 |
| Farseer κ-free | non-embedding | 3.91 | 4.15 |
| MiniCPM | non-embedding | 1.20 | 1.27 |

- **Head FLOPs.** With the output head counted in FLOPs, the lifetime/training ratio for a non-embedding technology is w·(1 + N_head/N_ne).
- **For sub-1B models (n = 8) the convention matters for one sweep, not the other.** Like-for-like medians (same data and form, only the count changes): Chinchilla κ = 1, 9.96 (total) vs 9.13 (non-embedding refit), 8% lower; Farseer κ = 1, 10.67 (total) vs 6.85 (non-embedding), 36% lower. The κ-free rows are 15.10 (Chinchilla, total) and 14.26 (Farseer, non-embedding). (The builder's "40–55%" compared the κ-free total-N row with κ = 1 non-embedding rows, mixing curvature with convention.) The Farseer gap is larger than the tokenizer effect m3 discussed; the Chinchilla gap is not.
- **The sample median moves little across conventions,** because the embedding share is 0.5–28% in the clean sample.

### H6. Partial identification of M*(C) at frontier scale [E]
Files: `ra2_wedge_pi_mstar.csv`, `ra2_wedge_pi_summary.csv`, `ra2_wedge_pi_anchors.csv`, `ra2_wedge_pi.tex`.

- **No design observes M* above 10²² FLOP.** The largest IsoFLOP budgets are 3×10²¹ (Chinchilla), 10²² (Llama 3), and 3×10²⁰ (Marin; DeepSeek LLM).
  - Llama 3's 402B/16.55T at 3.8×10²⁵ is an extrapolation of Meta's law, not an observation, and is not used.
- **Anchors:** each design's own A2 path at its largest budget:
  - Chinchilla profiles M* = 22.7 [18.2, 29.4] at 3×10²¹;
  - Llama 3: 22.3 [20.8, 23.7] at 10²² (Meta's 10-budget path; on ra1's 8 bracketed budgets the path gives M* ≈ 15 at 10²²–10²⁴, inside PI-1's bounds);
  - Marin: 12.4, 9.4 and 10.0 at 3×10²⁰;
  - DeepSeek: published law.
- **Elasticity bounds:** e = 1 − 2a over A2 paths and published laws (including DeepSeek's three data sets and MiniCPM): **[−0.156, 0.190]**. Over all ex-ante technologies: [−0.156, 0.311].

**Bounds on M*(C) and sign identification**

| Set | M*(10²³) | M*(10²⁴) | M*(10²⁵) | ŵ > 1 identified (clean) |
|---|---|---|---|---|
| PI-1: IsoFLOP anchors, e ∈ [−0.16, 0.19] | [3.2, 57] | [2.3, 89] | [1.6, 138] | **86%** |
| PI-2: PI-1 + M* nondecreasing | [8.0, 57] | [8.0, 89] | [8.0, 138] | 86% |
| PI-3: own-lab anchor where it exists | Meta: [14.5, 37] | [10.1, 57] | [7.1, 88] | 77% |
| PI-4: every ex-ante technology as anchor, all slopes | [0.4, 1,609] | [0.3, 3,293] | [0.2, 6,739] | 23% |

- **Models whose sign is ambiguous under PI-1** (11, all with M ≤ 96): Llama 2 70B, LLaMA 30B and 65B, Llama 3.1 405B, Qwen-72B, Qwen2-72B, Yi-34B, Falcon 40B and 180B, MPT-30B, DeepSeek-LLM 67B.
- **Under PI-3**, AI2's models also become ambiguous. Their own OLMo-ladder anchor is imprecise (in-support ln M* ∈ [0.67, 7.38] under κ = 1).
- **No model's w < 1 is identified under any set.**
- **Bounds on the median s** (curvature 1/σ* − 1 ∈ [0.40, 0.52], the range of the κ-free Chinchilla and Farseer and ra1's model-free Meta and Marin estimates): **[0.59, 0.93]** (PI-1) and [0.55, 0.94] (PI-3). These are medians of the per-model bounds.

### H7. Inside vs outside the design support (R2 Major 3) [E]
File: `ra2_wedge_insupport.csv`.

- **Only 2 of 77 clean models lie inside Chinchilla's (M, C) design.**
- **Inside vs outside Chinchilla's M range (M ≤ 341):**

  | Region | n | Median ŵ | Median s | Band > 1 |
  |---|---|---|---|---|
  | Inside | 30 | 2.28 | 0.56 | 0% |
  | Outside | 47 | 6.57 | 0.85 | 30% |

- **Inside vs outside Farseer's non-embedding M range (≤ 2,570), κ-free Farseer:**

  | Region | n | Median ŵ | Median s | Band > 1 |
  |---|---|---|---|---|
  | Inside | 57 | 3.10 | 0.68 | 3.5% |
  | Outside | 20 | 10.8 | 0.91 | 60% |

- **Conclusion: the technology-robust sign comes from the extrapolated models.** R2's point holds on the clean sample.
- **Direction of the extrapolation error (ra1, Farseer).** Inside Farseer's design, every parametric form understates the model-free local wedge at high M, and the gap grows with M: ln(w_param/w_local) for κ-free Farseer is −0.14 at M = 256–1,024 and −0.36 at M ≥ 1,024 (−0.12 and −0.25 at ra1's extended-grid cross-validated bandwidth); for Chinchilla-form Farseer −0.49 and −0.82 (ra1 memo H4). File: `ra2_wedge_extrapolation_check.csv`.
  - Applying these bin-level gaps to the 57 clean models inside Farseer's non-embedding M range raises the median κ-free Farseer wedge from 3.10 to 3.56 (3.49 at the CV bandwidth), median s from 0.68 to 0.72 (0.71); for κ = 1 Farseer from 2.29 to 3.73. The gap is negative for 56% of these models (all with M ≥ 256) and slightly positive (+0.05) at M = 16–64.
  - **Reading:** where it can be checked, the parametric extrapolation in M is conservative: it understates w, the more so the larger M. This transfers Farseer's recipe-specific gap to other labs' models and to larger N and C, so it signs the error rather than correcting levels. For the 20 clean models beyond Farseer's M range (median κ-free Farseer w 10.8) there is no model-free evidence. It does not address extrapolation in C (M*(C) at frontier scale; H6).

### H8. Conduct tests (R1 c1–2; R2 Major 5; R3 M5) [E]
File: `ra2_wedge_conduct.csv|.tex`. Outcome ln ŵ under the reference. Inference: restricted wild cluster bootstrap-t by developer with Webb six-point weights, B = 9,999 (p floor 10⁻⁴).

**(a) Open-weight premium.** Clean production-scale universe: 6ND ≥ 10²¹, confident N and D only, no MoE, no non-transformers, clean-sample exclusions; n = 186, 67 developers, of which 23 have at least one closed model (15 in 2023+).
- Given year and ln C: **0.52 log points** (CRV1 s.e. 0.18; p = 0.011).
- 2023 and later: 0.71 (0.17; p = 0.001; n = 150, 60 clusters).
- On s: 0.29 (0.16; p = 0.097).
- Closed models with disclosed N and D are few after 2023 (6 in 2024, 2 in 2025).

**(b) Developer serving footprint vs on-device target.** Clean sample, n = 77, 18 developers; regressors entered jointly with ln C and year effects.

| Regressor | Coefficient (s.e.) | WCR p | Clusters with attribute (G₁) |
|---|---|---|---|
| Developer serves its models | 0.45 (0.26) | 0.39 | 7 of 18 |
| On-device target (model card) | 0.22 (0.23) | 0.41 | 2 of 18 |
| Local / limited-resource target | 0.04 (0.25) | 0.88 | 3 of 18 |

- Serving is coded at the model's release date (review fix: Llama 1 and 2 predate Meta AI and are now 0; the builder's run had them at 1 and gave 0.37, p = 0.53). 33 clean models come from developers that serve, 44 do not.
- **The on-device and local rows are not informative.** The on-device attribute is carried by two developers (Hugging Face, H2O.ai) and "local" by three; with so few treated clusters the restricted wild cluster bootstrap under-rejects severely (mackinnon2018wild).
- Including the research suites and replications (n = 136, 26 clusters, 8 serving developers): serving 0.47 (0.17; p = 0.07) on ln ŵ and 0.14 (0.05; p = 0.046) on s. That sample adds suites whose D is fixed by research design and whose developers do not serve, so the contrast is partly suites vs products.
- **Neither story is supported or rejected on the clean sample with 18 developer clusters.** The point estimate on serving is positive, the sign the demand story predicts.
- **Estimand by serving footprint (R1 c1 request; `ra2_wedge_serve_split.csv`).** Models whose developer served its own LLMs at release (33 models, 7 developers): median ŵ 4.96, median s 0.80, all ŵ > 1. The others (44 models, 12 developers): 3.80, 0.74, 96% > 1. "Planned serving expenditure" applies to the first group only; for the second, s is the internalized value of compactness. Models with an on-device or local target (12, from 3 developers): median s 0.86.

**(c) Multi-tier bunching (quantized memory-tier windows 2.4–3.3B, 6.5–9.5B, 11.5–14.9B, 26–32.9B, 65–72.9B; counts use the 0.1-dex bins of log₁₀N whose midpoints fall in a window, so the counted ranges are slightly wider than the windows).**
- **Bunching is strong.** Open-weight production-scale models 2023–2026 (n = 220): 51% sit in the five windows vs 13% under a degree-5 polynomial counterfactual. Excess mass is 83 models (s.e. 9.8), 45 of them at 6.5–9.5B (z = 6.7).
- The clean sample shows 58% vs 10%.
- **But given compute, tier position does not predict over-training beyond what the mechanical relation implies.**
  - Inside a tier window: −0.39 (0.28; WCR p = 0.34). In 99.7% of 999 shifted placebo tier menus the coefficient is at least as large, so the real windows are, if anything, *less* over-trained.
  - Log distance to the next cap: 0.46 (placebo mean 0.45; placebo p = 0.61). Universe: 0.62 vs placebo mean 0.65.
- **Reading:** tier menus shape the choice of N. The tier-constraint prediction for the wedge, with no demand, is not visible at given compute.

### H9. Validation of levels (R3 M3c; R2 Major 11; R1 minor 39) [E/L]
Files: `ra2_wedge_validation_hf.csv`, `ra2_wedge_validation_openrouter.csv`, `ra2_wedge_aggregate.csv`.

**OpenRouter per-model token volumes are not accessible.**
- The documented API has no usage field.
- OpenRouter's Terms of Service (updated 2026-08-31, §7) prohibit scripts and crawlers that scrape the Site, which covers the rankings and activity pages.
- We use the author-level totals published in aubakirova2026state (Table 1, Nov 2024 – Nov 2025) as an order-of-magnitude audit. One year of OpenRouter traffic equals **0.04–4.6%** of the planned lifetime tokens implied at p = 1:
  - DeepSeek 4.6%;
  - Moonshot 0.7%;
  - Zhipu 0.5%;
  - Meta 0.2%;
  - Google 0.1%;
  - Alibaba 0.04%.
- A proportionality test of planned T against realized tokens is not possible with public data.

**Hugging Face model-tree derivative counts** (base + official post-trained + same-run releases; n = 74, 16 developers; given ln C and year):

| Outcome | On ln M | WCR p |
|---|---|---|
| ln(1 + quantizations) | 0.61 | 0.018 |
| ln(1 + fine-tunes) | 0.87 | 0.029 |
| ln(1 + all derivatives) | 0.80 | 0.031 |

- The elasticity of all derivatives with respect to planned T/D is 0.60 (p = 0.086), below 1.
- Smaller models at given compute are adapted more. That is consistent with demand and equally with accessibility. It does not validate levels.

**Compute-weighted planned serving share s_agg** (clean sample; truncated at w ≥ 1):

| Year | Reference | Range over technologies | Leave out Llama 3.1 405B |
|---|---|---|---|
| 2023 | 0.28 | [0.01, 0.60] | |
| 2024 | **0.58** | [0.04, 0.81] | **0.73** |
| 2025 | **0.83** | [0.21, 0.96] | |
| 2023–2025 pooled | 0.66 | | |

- The 405B is 40% of pooled compute.
- Open-weight clean universe: 0.35 (2023), 0.54 (2024; 0.63 without the 405B), 0.82 (2025).
- **Benchmarks [L, verified from the PDFs].** patterson2022carbon: "about 3/5 of ML energy use is for inference" at Google (one week of April in 2019–2021, all ML). wu2022sustainable: a power-capacity split of 10:20:70 for experimentation, training and inference at Facebook; 65% inference for its LM task.
- **Reading:** order-of-magnitude agreement for 2024 releases under the reference. But the technology range spans most of [0, 1], and the objects differ: a planned lifetime share for a release cohort vs realized flows of whole fleets.

### H10. Cost-side sensitivity (R1 c5) [E]
File: `ra2_wedge_costsens.csv|.tex`. Lifetime cost is 6N^{1+δ}D + 2pN^{η}T. Then K_inf/K_tr = (w − 1 − δ)/η, s = (w−1−δ)/(w−1−δ+η) and T/D = 3(w−1−δ)/(pη).

| Case | Median s | Llama 3 8B s | Median T/D (p = 1 / 3 / 10) | Llama 3 8B T/D (p = 1 / 3 / 10) |
|---|---|---|---|---|
| δ = −0.1, η = 0.9 | 0.77 | 0.87 | 10.3 / 3.4 / 1.0 | 19.4 / 6.5 / 1.9 |
| δ = 0, η = 1 | 0.75 | 0.85 | 9.0 / 3.0 / 0.9 | 17.2 / 5.7 / 1.7 |
| δ = 0.1, η = 1.1 | 0.72 | 0.84 | 7.9 / 2.6 / 0.8 | 15.3 / 5.1 / 1.5 |
| δ = 0, η = 0.5 / 0.75 / 1.25 | 0.86 / 0.80 / 0.71 | 0.92 / 0.88 / 0.82 | 17.9 / 12.0 / 7.2 (p = 1) | |

- **s is insensitive to δ and p.** It moves by ±0.03 for δ = ±0.1 and does not depend on p.
- **Token counts T scale as 1/p.** At p = 10, the median planned serving tokens are about 0.9 times training tokens.

### H11. The ordinal content is mechanical; the cardinal content is (S, M*) (R3 M3a; R2 Major 3) [E]
- On the clean sample, ln ŵ (reference) on ln M: slope 0.426 (= S/2), **R² = 0.99994**; with ln C, R² = 1.000. Spearman(ŵ, M) = 0.9998.
- **Curvature sensitivity at the reference zero point** (`ra2_wedge_sigma_sensitivity.csv`):

  | σ* | Median ŵ | Llama 3 8B |
  |---|---|---|
  | 0.60 | 8.64 | 19.5 |
  | 0.65 | 5.71 | 11.0 |
  | 0.70 | 4.00 | 6.8 |
  | 0.737 (κ = 1) | 3.17 | 4.9 |
  | 0.80 | 2.25 | 3.0 |

  - The share with ŵ > 1 (0.974) does not depend on σ*.

### H12. Model-free curvature of the lab IsoFLOP designs (R1 c7; input to lab-own) [U/E]
Files: `ra2_wedge_modelfree_sigma.csv|.tex`; figure `ra2_wedge_modelfree`; ra1's `ra1_modelfree_isoflop_summary.csv`.

- **Used (module ra1, reviewed):** random-effects mean over valid budgets (quadratic, path-centred windows h = 1, log-cubic frontier; budgets must bracket the minimum with ≥ 2 runs on each side). Llama 3 **0.660 (0.023)**; Marin Comma 0.700 (0.027), DCLM 0.713 (0.027), Nemotron-CC 0.705 (0.023); Chinchilla profiles 0.673 (0.027). ra1 reports finite-grid bias ≤ 0.006 and Monte Carlo coverage 0.92–0.98 for these intervals.
- **This module's estimator (sensitivity rows `*_mfraw`, `*_mfbc`):** per-budget quadratics over all runs of each budget (including Llama 3's two unbracketed budgets), a quadratic frontier and a ratio of sums.

  | Design | Raw σ* [95%] | MC-corrected σ* | ra1 σ* (used) |
  |---|---|---|---|
  | Llama 3 | 0.653 [0.639, 0.666] | 0.678 | 0.660 |
  | Marin, Comma | 0.650 [0.611, 0.669] | 0.675 | 0.700 |
  | Marin, DCLM | 0.629 [0.586, 0.653] | 0.662 | 0.713 |
  | Marin, Nemotron-CC | 0.644 [0.613, 0.666] | 0.672 | 0.705 |
  | Chinchilla profiles | 0.610 [0.588, 0.644] | 0.675 | 0.673 |

  - Its parametric-simulation correction (factors 0.87–0.90; 0.75 for Chinchilla) is specific to this estimator, which is noisier and more exposed to profile asymmetry than ra1's windowed one. On Marin the two estimators differ by 0.03–0.05 in σ*, within ra1's own specification range for Marin (0.64–0.72).
- **Comparison with parametric fits.** On Llama 3 the model-free σ* is far below the same-run κ = 1 fit (0.769; ra1) and close to κ free (0.693). On Marin, ra1's same-run κ = 1 fits (0.673–0.690) are close to the model-free values; only m1's A2/A1 κ = 1 objects (σ* 0.87–0.90, with E and γ unidentified) are far above. The builder's statement "below every κ = 1 fit" held only against those A2/A1 objects.
- The Llama 3 A2 path slope a = 0.4632 reproduces Meta's published law.

---------------------------------------------------------------------------------------------------

## 2. Methods

### 2.1 Clean inference-demand sample (defined ex ante; `sample.py`)
Starting point: m3 Sample B core (173). The steps are applied in order; a model is dropped at the first step that applies. All flags with their reasons and sources are in `data/processed/ra2_wedge/audit_flags.csv` and `output/tables/ra2_wedge_coding.csv`.

**(a) One observation per pretraining run; the first release is kept.**
- Llama 3.1 8B/70B (continued pretraining of the Llama 3 runs).
- Qwen1.5-72B (same N and D as Qwen-72B).
- OpenLlama v2 3B/7B (same N and D as v1).
- Yi-1.5 6B/34B (the card says "continuously pre-trained on Yi with … 500B tokens").
- H2O-Danube2-1.8B ("initialized from H2O-Danube-1.8B", arXiv:2401.16818 §7).

**(b1) Research suites whose D is fixed by design.**
- Pythia, OPT, BLOOM, RWKV-4-Pile, XGLM, Cerebras-GPT and GPT-Neo/J/NeoX.
- CodeGen-NL: the natural-language stage of a code model.
- Falcon-RW-1B: a RefinedWeb ablation.

**(b2) LLaMA-recipe replications and dataset-benchmark models.**
- OpenLlama and RedPajama-INCITE.
- LLM360 Amber and K2.
- DCLM-7B.

**(c) MoE.**
- Twelve models, reported with [total-N, active-N] bounds.

**(d) Distilled, pruned or derived initialization. Audited against the technical reports, because the Gemma cards are silent.**
- Gemma 2 2B/9B: "we train the 2B and 9B models with knowledge distillation", arXiv:2408.00118.
- All Gemma 3 sizes: "The Gemma 3 models are trained with distillation", arXiv:2503.19786.
- Llama 3.2 1B/3B (pruned plus logit distillation; card).
- Yi-1.5-9B: continued from Yi-9B, which was depth-upscaled from Yi-6B.
- Llama 4 Maverick (codistilled; already removed at (c)).
- **Audit result:** a keyword audit of all 179 downloaded cards plus these reports found no distilled or pruned pretraining beyond m3's 10 flags. The audit added Yi-1.5-9B (derived initialization) and three continued-pretraining re-releases (moved to (a)). "Distillation" in the DeepSeek-V3 card refers to post-training; in the Llama 3.1/4 cards it is license boilerplate.

**(e) Multimodal pretraining.**
- Gemma 3 4B–27B (the paper says "The increase in tokens accounts for the mix of images and text").
- Llama 4.
- All were already removed at (c) or (d).

**(f) Instruction-tuned-only releases.**
- Phi-3 mini/small/medium, Phi-3.5-mini, Phi-4-mini and phi-4.
- Qwen3-32B (no base released).

**(g) Non-transformer or hybrid.**
- RWKV (already removed at (b1)). In the universe used for conduct tests, a name-based filter removes Mamba, Hawk/Griffin, RetNet, Nemotron-H, Falcon-H1, Granite-4.0-H and similar models.

**(h) D without a primary source.**
- Qwen1.5 0.5B–32B: token counts only in ObsScaling; the Qwen1.5 cards give none (R2 Major 10).

**Embedding counts.** N_emb and N_nonemb come from m3 (config.json). For MPT-7B/30B, BTLM-3B and Falcon-180B, whose configs return authentication errors, they were filled from public architecture values (`sample.ARCH_MANUAL`). N_head = vocabulary × width.

**Family generations and classes.** Llama 3 and 3.1 are merged into one "Llama-3 herd". Common D means max/min D ≤ 1.10 with at least two members.

### 2.2 Technologies and the ex-ante rule (`techs.py`)

**Inclusion rule, fixed before any wedge was computed:**
- **(i)** Every technology estimated in m1, m2 or m8 from final-checkpoint losses of a designed sweep, using the primary estimator (Huber-LSE on log loss). Each corpus and each parameter-count convention is its own row, under κ = 1 and, where estimated, κ free (inner exponents).
- **(ii)** For IsoFLOP designs: the Approach-2/A1 path (κ = 1 curvature), the primal fit, and the lab path with model-free curvature.
- **(iii)** Literature point estimates on the Chinchilla data (Hoffmann A3; Besiroglu) and published lab laws:
  - Llama 3, via m1's A2/A1, which reproduces 0.299 C^0.537;
  - DeepSeek LLM: M_opt = 0.1715 C^0.5243 and D_opt = 5.8316 C^0.4757 with C = M·D and M = non-embedding FLOPs per token (Bi et al. 2024, Eq. 4; the data-quality exponents 0.450/0.524/0.578 from their Table 4 enter the elasticity bounds);
  - MiniCPM: 7.15×10⁻² N^−0.29 + 3.00×10⁻¹ D^−0.23 + 0.31 with non-embedding N, and D/N = 191.87 at 10²¹ (Hu et al. 2024, Fig. 10 "Average"). Its compute-optimal ratio lies outside its own design (6 sizes, D/N from 10 to 60), so MiniCPM's M* is itself an extrapolation; the implied elasticity 1 − 2a = 0.115 is close to the published η = −0.10 (average).
  - All three sets of numbers were checked against the PDFs in `data/raw/ra2_papers/`.

**Excluded ex ante:**
- DataDecide (m2: "LR schedule not complete") and (Mis)Fitting (IsoFLOPs interpolated from per-checkpoint logs). The checkpoint artifact is a function of D/N.
- Estimator and sample variants on a design already represented (NLS in levels, all 245 points, 9-cluster draws, m2 pairs draws for κ-free, this module's raw and bias-corrected model-free curvature, Meta's path on ra1's 8 bracketed budgets). These are reported as sensitivity rows.

**The resulting set of 32:**

| Family | Rows |
|---|---|
| Chinchilla | κ-free (reference), κ = 1, Besiroglu, Hoffmann A3, non-embedding (m8 refit) |
| Farseer | non-embedding κ = 1, total κ = 1, κ-free |
| Gadre et al. | RefinedWeb, C4, RedPajama × {κ = 1, κ-free} |
| OLMo ladder | κ = 1, κ-free |
| Muennighoff et al. | κ = 1, κ-free |
| Meta | A2/A1, primal, model-free (ra1 curvature) |
| Marin | 3 corpora × {primal, A2/A1, model-free (ra1 curvature)} |
| Published laws | DeepSeek LLM, MiniCPM |

**Hybrids.** When the path comes from one source and the curvature from another, α = S(1 − a) and β = S·a. The wedge depends only on (S, a, ln G).

**Draws.**
- κ-free Chinchilla: own wild bootstrap, B = 399.
- Chinchilla non-embedding: m8's point estimate re-fitted with the Pearce–Song map; wild bootstrap, B = 399.
- Model-free lab technologies: this module's joint wild bootstrap of the path (B = 999), with the curvature draws re-centred on ra1's σ* and rescaled to ra1's standard error (delta method, sd(S/2) = se(σ*)/σ*²), which keeps the path–curvature correlation.
- Everything else: m1/m2 draws (B = 188–400). These are pairs or stratified pairs bootstraps, not design-conditional (R1 9a); only the reference, the Chinchilla non-embedding row and the model-free paths use wild draws.
- DeepSeek: the reference curvature draws, since no path draws are published.
- MiniCPM: a point estimate.

### 2.3 Model-free curvature and paths (`modelfree.py`)

**Data.** The open-athena IsoFLOP compilation:
- Llama 3: 133 points, 10 budgets.
- Marin 2026-03, Comma / DCLM / Nemotron-CC: 85–88 points, 7–8 budgets.
- Chinchilla: the 123 IsoFLOP-profile points (cross-check).

**Steps.**
1. For each budget with ≥ 4 points and an interior argmin: OLS of L on (x, x²), x = centered ln N. This gives the argmin, L*_k and L_nn,k = 2c₂.
2. Fit a quadratic in ln C through the L*_k. Its derivative gives dL*/dc.
3. Pool as the ratio of sums, S/2 = ΣL_nn/Σ2|dL*/dc|. Per-budget ratios and a median rule are in the CSV.
4. The path comes from OLS of ln N*_k on ln(C_k/6).

**Relation to ra1.** ra1 estimates the same identity with a different estimator (windowed local quadratics, bracketed budgets only, log-cubic frontier, random-effects pooling). After review, the lab-own technologies use ra1's σ*; this module's estimator supplies the paths (a, ln G), the PI anchors and the sensitivity rows.

**Inference and bias correction (this module's estimator).**
- Design-conditional wild bootstrap: Rademacher weights on HC1-rescaled residuals of the per-budget fits, with the whole chain redone in each draw.
- Cubic per-budget fits are a robustness check.
- Bias correction (`mc_bias_factor`): simulate a Chinchilla-form technology at the estimated (S, a, ln G) on the design's own grid, with Gaussian noise at the design's residual s.d. (300 replications). Factor = true/mean estimate.
- Noiseless checks at the Besiroglu and Hoffmann technologies recover S/2 within +2% (cubic) to +6% (quadratic) on every design.

### 2.4 Wedges, shares, joint bootstrap, bands (`wedges.py`)

**Point wedge.** For each model and technology, ln ŵ = (α+β) ln G − α ln N + β ln D, with N in the technology's convention:
- total (active N for MoE);
- non-embedding;
- OLMo's count (excludes the input embedding);
- DeepSeek's units: N_ds = M/6 with M = 6N_ne + 12·n_layer·d·4096 from config.json.

**Model-level interval.** The 2.5/97.5 percentiles over the technology's draws.

**Joint bootstrap (R1 9f).** In each draw, recompute the sample median of w, the median of s and the share with w > 1, then take percentiles.

**Bands.**
- Band: the union of 95% intervals (points for literature rows) over the 32 ex-ante technologies.
- "All points > 1": every ex-ante point estimate exceeds 1.

**Lab-own map (`techs.LAB_OWN`), primary technology with alternatives:**
- Meta: meta_mf (meta_a2, meta_a3, meta_mfbc, meta_mfraw, meta_mf8);
- AI2: olmo_q (olmo);
- DeepSeek: deepseek;
- Marin: marin_nemotron_mf (Comma/DCLM model-free; A2/A1 rows; marin_nemotron_mfbc).

**Vintage.** `techs.LAB_LAW_FROM`: Meta's law from 2024-04 (Llama 3), AI2's ladder from 2024-11 (OLMo 2), DeepSeek's from 2024-01, Marin's from 2026-03. Earlier releases are evaluated under an ex-post lab law.

Marin 8B's card lists Nemotron-CC and DCLM among its datasets; Comma is not used.

### 2.5 Partial identification (`analysis_ra2.pi_*`)

**Extrapolation rule.** ln M*(C) = ln M*(C₀) + e·ln(C/C₀), with e ∈ [e_lo, e_hi] and the anchor's own 95% interval at C₀. The identified set for a model is the union over anchors, each evaluated in its own N convention and at the model's own compute in that convention.
- **Sign.** w > 1 is identified iff ln M exceeds the upper bound for every anchor.
- **Bounds on ln w.** Use S/2 ∈ [0.40, 0.52], the range of the κ-free (Chinchilla, Farseer) and ra1 model-free (Meta, Marin) estimates.
- **Assumption sets.** PI-1 to PI-4 as in H6. PI-3 uses the lab's own IsoFLOP or published anchor (Meta, Marin, DeepSeek) and AI2's ladder in-support M*.

### 2.6 Conduct tests (`analysis_ra2`)

**Clustering and inference.**
- All regressions cluster by parent developer (m3's `DEV_MAP`).
- Restricted wild cluster bootstrap-t: cameron2008bootstrap, with webb2023reworking weights. B = 9,999; symmetric p-values.

**(a) Open vs closed.** Production-scale universe from m3 (6ND ≥ 10²¹, 2019–2026). Sample-A rows are kept only if Epoch's confidence is "Confident". MoE, non-transformer and clean-sample exclusions are dropped.

**(b) Serving footprint and deployment target.**
- *Serving footprint (developer level, dated).* A developer counts as serving if it operated a commercial API or consumer product that served its own LLMs at the model's release. Coded 1:
  - Meta from 2023-09-27 (Meta AI launch, so Llama 1 and 2 are 0), Alibaba, Google, Microsoft, IBM (watsonx), MosaicML, DeepSeek (2024), 01.AI (2024), Moonshot, Zhipu, Together.
- Coded 0:
  - TII, Hugging Face, H2O.ai, AI2, Stability AI, Swiss AI, Marin, M-A-P, TinyLlama, EleutherAI and other research developers.
- Sources are in `ra2_wedge_coding.csv`.
- *Deployment target (model level).*
  - On-device: the card names on-device, mobile or phone use (SmolLM/SmolLM2/SmolLM3, H2O-Danube3, Llama 3.2, Gemma 3 270M, Phi-3-mini).
  - Local: the card states laptops or desktops (Gemma).
  - Clean sample: 9 on-device, 3 local, 65 server or unspecified; 33 models from developers that serve, 44 not.
  - Treated-cluster counts (G₁, G₀) are reported for every binary regressor (`ra2_wedge_conduct.csv`); with G₁ ≤ 3 the WCR p-value is not interpretable (mackinnon2018wild).

**(c) Tiers.**
- *Bunching.* A degree-5 polynomial fitted to 0.1-dex bins of log₁₀N, excluding the five windows (chetty2011adjustment; kleven2016bunching). Bootstrap over models, B = 999.
- *Tier-position regressions.* A placebo distribution over 999 tier menus shifted by exp(u), u ~ U(−0.69, 0.69). Under one technology ln w is a deterministic function of (N, C), so any tier regressor is mechanically correlated with ln w at given C; only the placebo comparison is informative.

### 2.7 Validation and aggregates

**HF model tree.** `https://huggingface.co/api/models/<id>?expand[]=childrenModelCount`. This filter syntax was verified: it returns adapter / merge / quantized / finetune counts. Counts are summed over the base, the official post-trained repositories (m3's `INSTRUCT` map) and same-run re-releases.

**OpenRouter audit.** Planned lifetime tokens at p = 1 are summed over each author's verified releases, excluding duplicates. They are compared with one year of served tokens (aubakirova2026state, Table 1).

**Aggregate share.** s_agg = Σ(w⁺ − 1)C/Σw⁺C with w⁺ = max(w, 1). It is computed by year and by technology (reference; lab-own; the per-model min and max point over technologies), with leave-one-out of Llama 3.1 405B.

---------------------------------------------------------------------------------------------------

## 3. Inventory

| File | Content |
|---|---|
| `output/figures/ra2_wedge_ecdf.pdf/.png` | **Paper exhibit (i)**, replacing the old w-vs-M figure (R3 M12). ECDFs of s. (a) Clean sample: 32 ex-ante technologies in gray, reference in blue, "lab-own where available, else reference" in orange. (b) Lab-own subsample (n = 22). |
| `output/tables/ra2_wedge_models_selected.tex` | **Paper exhibit (ii)**: 21 clean models with N, D, M, embedding share, M/M*, w [95%] and s under the reference, lab-own w and s, and the band. Panel B: 6 MoE models with [total-N, active-N] bounds. Panel C: 3 excluded models. |
| `output/tables/ra2_wedge_cleaning.tex` (+ `_cleaning.csv`, `_technologies.csv`, `_technologies_by_sample.csv`) | **Paper exhibit (iii)**: cleaning steps; dispersion over the 32 ex-ante technologies with joint bootstrap intervals. The by-sample file repeats the dispersion for "clean (a)–(g)" (83) and all 173. |
| `output/tables/ra2_wedge_conduct.tex` (+ `_conduct.csv`, `_bunching.csv`, `_validation_hf.csv`, `_validation_openrouter.csv`, `_aggregate.csv`) | **Paper exhibit (iv)**: conduct tests (a)–(c) and validation (HF, s_agg vs Patterson/Wu). |
| `output/figures/ra2_wedge_trends.pdf/.png` | **Paper exhibit (v)**: median s by release year, open vs closed. Counts are on the plot; cells with n < 5 are omitted; medians below −1 are marked. |
| `output/tables/ra2_wedge_pi.tex` (+ `_pi_mstar.csv`, `_pi_summary.csv`, `_pi_anchors.csv`) | Partial identification of M*(C) and sign identification. |
| `output/tables/ra2_wedge_costsens.tex/.csv` | Cost-side sensitivity. |
| `output/tables/ra2_wedge_modelfree.tex` (+ `_modelfree_sigma.csv`), `output/figures/ra2_wedge_modelfree.pdf/.png` | Model-free curvature and paths of the IsoFLOP designs: this module's estimator and the ra1 value used (appendix). |
| `output/tables/ra2_wedge_models.csv` | Every Sample-B model × every technology: w, 95% bounds, s, M/M*, lab-own, band, total-N wedges for MoE, flags (R2 minor 34). |
| `output/tables/ra2_wedge_serve_split.csv` | Headline by developer serving footprint and stated deployment target (H8b). |
| `output/tables/ra2_wedge_extrapolation_check.csv` | Direction of the extrapolation error: ra1's Farseer model-free vs parametric gap applied to clean models inside Farseer's M range (H7). |
| `output/tables/ra2_wedge_labown.csv`, `_families.csv`, `_family_split.csv`, `_family_level_W.csv`, `_conventions.csv`, `_insupport.csv`, `_sigma_sensitivity.csv`, `_trends.csv`, `_coding.csv` | Supporting tables. |
| `data/processed/ra2_wedge/` | `audit_flags.csv` (every flag with reason and source), `wedge_long.csv`, `pi_bounds_models.csv`, `hf_tree_counts.csv`, `conduct_sample.csv`, `modelfree_per_budget.csv`, `headline.json`, `run_log.txt`, `outputs_cache.pkl`. |
| `code/data/download_ra2_wedge.sh`; `data/raw/ra2_hf_tree/`, `ra2_cards/`, `ra2_papers/` | Raw downloads. |
| `lit/bib/extra_ra2_wedge.bib` | New verified references: patterson2022carbon, wu2022sustainable, aubakirova2026state, feng2011wild, chetty2011adjustment, kleven2016bunching. |

---------------------------------------------------------------------------------------------------

## 4. Claims for the paper

**1. On a clean inference-demand sample (77 models), the reference technology (Chinchilla, κ free) implies a median planned serving share of lifetime cost of 0.75 [0.69, 0.79], and 97% of models are over-trained.**
- *Evidence:* H1; `ra2_wedge_cleaning.csv`; joint wild-bootstrap intervals (B = 399).
- *Caveat:* the interval is sampling uncertainty in one technology.
  - Across 32 ex-ante technologies the median s runs from 0.17 to 0.85.
  - The union band lies above 1 for only 18% of models.
  - s is a share of *internalized* value of compactness. For open-weight developers who do not serve, it is not serving expenditure (by serving footprint: 0.80 for the 33 models of developers that serve, 0.74 for the other 44; H8b).
  - 32 of the 77 belong to common-D families, whose individual wedges are not revealed-preference objects; on the other 45 the median s is also 0.75 (H4).

**2. Technologies that let the data choose the curvature agree more closely than the κ = 1 technologies.** Chinchilla κ-free, Farseer κ-free, Meta and Marin with ra1's model-free curvature, and DeepSeek's law with the reference curvature give median s of 0.69–0.79.
- *Evidence:* H2; H12.
- *Caveat:* they share curvature 0.66–0.71 but not zero points. The agreement is partly a statement about S/2 ≈ 0.40–0.52. The κ-free rows of the smaller sweeps (Gadre, OLMo, Muennighoff; σ*_κ 0.51–0.62) are much more dispersed.

**3. Under Meta's own law and Meta's own IsoFLOP curvature, Llama 3 8B was built as if serving would take 88% of lifetime cost (w = 8.4 [5.7, 12.8]). The 405B flagship is on Meta's path (w = 0.97).**
- *Evidence:* H3.
- *Caveat:* the 405B result is by construction (Meta sized it with this law). Meta's alternatives give 3.09 (κ = 1 law) to 12.2 (path on ra1's 8 bracketed budgets) for the 8B; the interval reflects curvature and path sampling only. The curvature is ra1's (0.660 [s.e. 0.023]); this module's own estimator gives 0.653 raw and 0.678 after its bias correction.

**4. The sign of over-training at frontier scale is identified for 86% of the clean sample under weak assumptions. Large flagships with M ≲ 100 are not signed.**
- *Evidence:* H6.
  - PI-1: IsoFLOP anchors at their largest budgets, elasticity in [−0.16, 0.19].
  - M*(10²⁴) ∈ [2.3, 89]; [10.1, 57] with Meta's own anchor.
- *Caveat:* with every technology as an anchor (PI-4), the share falls to 23%. No design observes M* beyond 10²² FLOP.

**5. The robust sign comes from models outside every design.** Inside Chinchilla's M range the median s is 0.56, and no model has the full band above 1.
- *Evidence:* H7.
- *Caveat:* where it can be checked (inside Farseer's M range, ra1), parametric forms understate w at high M, so the extrapolated levels in M are conservative; beyond Farseer's M range and in C nothing is observed (H6, H7).

**6. With a family-level token budget, individual wedges are not revealed-preference objects. The family reveals Σω_j ℓ_j/w_j = 1.**
- *Evidence:* derivation plus numerical check (H4). 11 common-D families in the clean sample, with family W_f from 1.22 (DeepSeek LLM) to 9.13 (Qwen3).
- *Caveat:*
  - W_f is a lower bound on the family's compute-weighted lifetime/training ratio only if demand is larger for more over-trained members.
  - If sizes come from a tier menu, D reveals nothing about serving.
  - Common-D and size-specific families have the same median ŵ (3.90 vs 4.01).

**7. Conduct.**
- *Open-weight premium:* +0.52 log points at given compute and year (WCR p = 0.011; 67 developers).
- *Serving footprint and on-device target:* neither predicts ŵ in the clean sample (serving 0.45, p = 0.39; 18 developers, 7 serving). The on-device test rests on 2 developers and is not informative.
- *Tier sizes:* strong bunching at quantized memory tiers (51% of open models in the windows vs 13% expected), but no tier-specific over-training at given compute (placebo p = 0.37–1.00).
- *Caveat:*
  - Few closed disclosures after 2023.
  - Tests in (b) have low power with 18 clusters.
  - Because ŵ is a function of (N, C), these tests are about sizes at given compute.

**8. Levels cannot be validated with public data.**
- *Evidence:* OpenRouter per-model volumes are not accessible (ToS); author totals cover 0.04–4.6% of planned lifetime tokens. HF derivative counts rise with ln M at given C (0.80; p = 0.031) but have elasticity 0.60 (not 1) with respect to planned T/D.
- *Caveat:* do not report T in tokens without p; report s.

**9. Planned serving shares of open-weight releases rose from about 0.3 (2023) to about 0.8 (2025) under the reference. Across technologies the 2024 aggregate spans [0.04, 0.81].**
- *Evidence:* H9. The 2024 reference value (0.58; 0.73 without the 405B) is of the same order as Google's ~0.6 inference share of ML energy (2019–2021) and Facebook's 0.70 inference share of AI power capacity.
- *Caveat:* planned cohort shares vs realized fleet flows; technology-dependent; the 2025 cohort has 13 models.

**10. Reported shares are insensitive to the cost side.** Median s = 0.72–0.77 for MFU elasticities δ ∈ [−0.1, 0.1]; 0.71–0.86 for serving-cost elasticities η ∈ [0.5, 1.25]. Token counts scale with 1/p.
- *Evidence:* H10.

---------------------------------------------------------------------------------------------------

## 5. Robustness and failures

**Reference choice.**
- κ-free vs κ = 1 on the clean sample: 3.99 vs 3.34 (R4 M2's 20% gap persists).
- Pairs vs wild draws for the κ-free reference: the median interval is [3.25, 6.04] vs [3.21, 4.86]. Pairs resampling changes the design; the wild bootstrap is design-conditional (R1 9a).
- m1's κ-free point estimate vs our refit: median over 173 is 3.779 vs 3.767.

**Model-free curvature.**
- *Estimator choice is material for Marin.* ra1's reviewed estimator gives σ* 0.700–0.713 on Marin; this module's gives 0.63–0.65 raw and 0.66–0.68 after its simulation-based correction. Marin 8B's lab-own w is 5.08 with ra1's value and 6.66 with the corrected one. On Llama 3 the two agree more closely (0.660 vs 0.653 raw, 0.678 corrected): Llama 3 8B 8.36 vs 7.07 (corrected).
- *The Meta path depends on two unbracketed budgets.* Dropping ra1's invalid 3×10²¹ and 10²² profiles moves a from 0.463 to 0.501 and M*(10²⁴) from 31 to 15 (row `meta_mf8`): Llama 3 8B 12.2, 405B 1.64.
- *Per-budget ratios rise with C* for Llama 3 (0.41 → 0.88 at 10²² in this module's estimator; ra1 reports a drift of −0.053 in σ* per decade, p = 0.001). The σ* relevant at 10²³–10²⁵ may be lower than the design average.

**DeepSeek law in other labs' units.** Non-DeepSeek models use M = 6N_ne + 12·n_layer·d·4096. This conversion is a maintained approximation for architectures other than DeepSeek's.

**Conduct tests.**
- *(b)* has 18 clusters. The WCR p-values are the relevant ones; CRV1 t-statistics overstate precision.
- *Coding of serving footprint:*
  - Meta's Llama 1 (Feb 2023) and Llama 2 (Jul 2023) are coded 0: released before Meta AI launched (2023-09-27). The builder's run coded both 1 through a year-level threshold (review fix).
  - Microsoft's phi-1.5 (Sep 2023, research license) is coded 1 through the year-level threshold; whether Microsoft served its own Phi models first-party before Phi-2 (Dec 2023) is a judgment call left to a second coder.
  - Yi-6B/34B are coded 0 (API launched 2024).
  - These dates are judgment calls documented in `ra2_wedge_coding.csv`.
- *Tier regressions are mechanical.* We report placebo p-values rather than regression p-values as evidence.

**Data.**
- HF derivative counts are missing for MosaicML and Cerebras repositories (authentication errors, as in m3), so validation has n = 74.
- MPT, BTLM and Falcon-180B architecture values are entered by hand (`ARCH_MANUAL`). They affect only non-embedding and DeepSeek-unit technologies.

**Things not done.**
- A Raval-type second-margin test (R1 c3b: MoE sparsity, KV heads, vocabulary, distillation) was not run.
- Farseer Eq. 3 (non-homothetic) was not re-run; m3's result stands.
- Tokenizer or byte normalization of D (R2 Major 4e) was not done.
- A durability or obsolescence model of T (R1 c1d) is not part of this module.

**Upstream mutability.** During this session module m3's per-model output file was rewritten by another process between my two runs. My inputs did not change: two full runs gave identical ra2 outputs.

---------------------------------------------------------------------------------------------------

## 6. Referee comments addressed

**R1 (IO econometrician)**
- **c1 (conduct).**
  - The estimand is restated as the expenditure share of the generalized wedge.
  - Developer serving footprint and on-device coding with WCR tests (H8b): no rejection with 18 clusters; the on-device test (2 treated developers) is uninformative. Headline reported separately for developers that serve (H8b).
  - Open vs closed on a confident, MoE-free universe (H8a).
  - "Planned serving expenditure" is confined to developers that serve; for others s is the internalized value of compactness.
  - Durability not modeled (open).
- **c2a (tiers).** Multi-tier bunching at five quantized memory windows plus distance-to-tier regressions with a placebo-menu test (H8c).
- **c2b (family-level D).**
  - Results reported separately for common-D, size-specific and singleton families.
  - Family FOC derived and verified; the family-level W_f is reported (H4).
- **c3.** Published lab laws (DeepSeek LLM with data-quality exponents; MiniCPM) enter as technologies; the cross-lab spread of M* is quantified (H2).
  - Raval second-margin test not done (open).
- **c4a–b.** All 32 estimated and published technologies under an ex-ante inclusion rule, including OLMo, Muennighoff, three Gadre corpora, κ-free rows and lab laws (H2).
- **c4c.** The κ-free reference (H1).
- **c4e.** Partial identification of M*(10²³–10²⁵) with sign identification by assumption set (H6).
- **c4f.** Ranges and ECDFs replace pseudo-precise intervals.
- **c5.** MFU-by-size (δ) and serving-cost elasticity (η) sensitivity table (H10).
- **9a.** Wild (design-conditional) bootstrap for the reference, the Chinchilla non-embedding row and the model-free paths. The other technologies' intervals use upstream m1/m2 pairs or stratified draws, which are not design-conditional.
- **9f.** Joint bootstrap distribution of the share with w > 1 and of the median, for every technology with draws (H1–H2).
- **Minor 3.** The 405B on Meta's path is by construction (H3).
- **Minor 24.** MoE excluded, with bounds; multimodal excluded.
- **Minor 25.** Continued pretraining deduplicated.
- **Minor 37.** Vintage- and lab-specific technologies (lab-own); the vintage is flagged: for 11 of the 22 lab-own models the lab law postdates the model (Llama 1/2, OLMo 1/1.7, Marin 8B).
- **Minor 39.** OpenRouter tokens pursued (ToS bar documented) and HF model tree used.
- **Minor 40.** Aggregate truncated at w ≥ 1, with leave-one-out and a technology range.
- **Minor 41.** Intervals labelled as sampling intervals of one technology.

**R2 (ML scaling)**
- **Major 1.** κ-free reference; curvature sensitivity table (H11); ra1's model-free curvature for Meta and Marin (H12).
- **Major 2a.** Headline restated as the expenditure share s; T only with p ∈ {1, 3, 10} (H10).
- **Major 2b.** Serving-footprint test (H8b).
- **Major 2c.** Internal N-proportional uses (RL rollouts, distillation teachers) are discussed as part of the marginal value of compactness; not measured.
- **Major 3.** In-support vs extrapolated splits (H7); direction of the extrapolation error from ra1's Farseer model-free wedge (parametric forms understate w at high M; H7); mechanical R² and Spearman (H11); OpenRouter plausibility audit (H9). The Sardana-type "true vs perceived technology" point (R2 Major 3(ii)) is not resolved: ra1's gap is about the true technology of one recipe, not about the laws labs planned with.
- **Major 4a.** Non-embedding technologies at N_nonemb, with the head counted in FLOPs (H5).
- **Major 4b.** MoE excluded, with [total, active] bounds in exhibit (ii).
- **Major 4c.** Distillation and pruning audit (Section 2.1d).
- **Major 4d.** Multimodal excluded.
- **Major 4e.** Tokenizer normalization not done (open).
- **Major 5a/5c.** Research suites dropped; family classes; multi-tier bunching and tier tests. Synthetic-data models kept in the clean sample (not in the spec's ex-ante list) with a robustness row (H1).
- **Major 5 request 2.** Developer serving vs on-device test.
- **Major 6a.** Modern laws (DeepSeek LLM, MiniCPM, Llama 3 for all models, Marin, OLMo for all models) added. The κ-free Chinchilla reference is justified by R3 M4 and R4 M2; Farseer κ-free (a modern high-M sweep) gives nearly the same median (3.91 vs 3.99).
- **Major 10.** Cleaning table with median, share w > 1 and share band > 1 after each step (H1).
- **Major 11.** HF model-tree counts; OpenRouter access documented.
- **Minors.**
  - 18: rank = rank of M (H11).
  - 19–20: confident-only rows, MoE excluded in the open/closed test.
  - 21: multi-tier windows.
  - 22: distillation dummy replaced by exclusion; synthetic-data models kept, with a robustness row (H1).
  - 23: leave-one-out of the 405B; Llama 3/3.1 deduplicated.
  - 28–29: MoE bounds, Gemma 3 flagged, embedding-share column.
  - 34: per-model file for every technology and convention.

**R3 (editor)**
- **M3a.** The object is stated as a monotone transformation of M/M*(C) (H11).
- **M3b.** Expenditure share s throughout.
- **M3c.** Validation of levels: OpenRouter, HF, and aggregates vs Patterson/Wu, verified from the PDFs (H9).
- **M3d.** Lab-own technologies primary (H3).
- **M3e.** ECDF of s by technology (exhibit (i)).
- **M4.** κ-free reference; κ-free members in the set.
- **M5.1.** Open vs closed and serving footprint.
- **M5.2.** p calibration (H10).
- **M5.3–4.** Data cost and test-time compute are rival channels in the generalized wedge; not separately measured.
- **M5.5.** Family-level D (H4).
- **M5.6.** Distillation flag audited.
- **M12.** Figure 6 replaced by ECDFs. Trend figure with counts, truncated where n < 5.

**R4 (auditor)**
- **M2.** The κ-free reference reproduces the auditor's 3.78 on the 173 models with m1's point estimate (3.779). Design-conditional wild bootstrap draws; cluster and pairs draws kept as sensitivity rows.

---------------------------------------------------------------------------------------------------

## 7. Open issues

1. **ra1 outputs (resolved in review).** Lab-own Meta and Marin now use ra1's σ* (`headline.json: mf_source`); ra1's Farseer gap signs the extrapolation error (H7). Remaining: the lab-own Meta path uses two IsoFLOP budgets that ra1 finds unbracketed (sensitivity row `meta_mf8`).
2. **Raval-type second margin** (R1 c3b): MoE sparsity, KV heads or vocabulary as a second first-order condition with the same T. Not done.
3. **Durability and obsolescence** of T (R1 c1d), and **test-time compute** as a source of the 2024–2025 rise (R3 M5.4): theory (ra5) and discussion; no measurement here.
4. **Tokenizer or byte normalization of D** and vocabulary effects on N (R2 Major 4e): needs bytes per token by tokenizer.
5. **Levels.** Per-model realized serving volumes remain unavailable. Demirer et al.'s (demirer2025emerging) OpenRouter panel access is unverified and not permitted by scraping. A data-sharing request to OpenRouter would be the route.
6. **The serving-footprint coding is dated by judgment.** The reviewer re-dated Meta (2023-09-27; Llama 1 and 2 now 0). Microsoft's phi-1.5 (coded 1) and 01.AI (2024) remain judgment calls; the rest of `ra2_wedge_coding.csv` was not re-audited.
7. **Writers.**
   - Use s and the ranges, not T in tokens.
   - Describe the ex-ante technology set and its 18% full-band share honestly.
   - Present conduct tests as low-power at the developer level, except the open-weight premium and the bunching result.

**Citation keys used:**
- Existing bibliography: bi2024deepseek, hu2024minicpm, grattafiori2024llama, gemmateam2024gemma2, gemmateam2025gemma3, biderman2023pythia, zhang2022opt, dey2023cerebras, raval2023testing, demirer2025emerging, cameron2008bootstrap, webb2023reworking, mackinnon2018wild (added in review), bhagia2024establishing, sardana2024beyond, busbridge2025distillation, roberts2026test, bian2025scaling.
- New (`lit/bib/extra_ra2_wedge.bib`, verified via arXiv and Crossref APIs): patterson2022carbon, wu2022sustainable, aubakirova2026state, feng2011wild, chetty2011adjustment, kleven2016bunching.
