# Paper plan v2 — "What Optimizing Labs Reveal: Scaling Laws as Production Functions"

Lead author, 2026-09-24 (after referee round 1 and the revision analyses ra1–ra5). Binding for all writers.
Supersedes paper_plan.md. Read also: paper/notes/revision_plan.md (decisions), paper/notes/theory_main_text_v2.md
(formal statements, labels, numbering), the referee reports paper/referee/R*.md, and the memos + reviews:
ra1_modelfree, ra2_wedge, ra3_econ, ra4_obsfix, ra5_theory (new) and m1–m8 (earlier; superseded where ra* differ).
Numbers must be taken from the CSVs/memos (after their reviews), never from this plan alone.

## 0. The paper in one paragraph
Optimization is double-edged for measurement. Labs that choose model size N and training tokens D to minimize cost
generate data that cannot reveal the technology's curvature or its optimal mix (their runs sit on the expansion path,
where the loss surface carries no transverse information); at the same time their choices become informative about
their objectives. Designed variation breaks the circle. (1) Theory: scaling laws are production functions with a
multiplicative FLOP cost, so cost minimization equates output elasticities; interior optima require σ < 1; the
first-order condition of any objective V(L, N) inverts into the value of compactness, w − 1, or as a share,
s = (w − 1)/w — for developers who serve their models, planned serving expenditure; on-path data identify curvature
only by functional form, with information of fourth order in allocation errors; but σ* and w are identified
model-free from IsoFLOP curvature and local slopes. (2) Technology: model-free σ* ≈ 0.70 [0.67, 0.72] across three
independent labs' IsoFLOP designs (Chinchilla, Llama 3, Marin; Porian's small-scale ladder ≈ 0.51); parametric
κ-free estimates agree on the large designs; the Chinchilla κ = 1 form overstates σ* for Llama 3; the allocation
exponent and M* are not portable. Our controlled experiment [m9]. (3) Revealed demand: on a clean sample of 77
open-weight base models, the reference technology implies a median s of 0.75 (lab-own technologies 0.61 on their
22 models); the share rose from ≈0.04 (2019–22) to ≈0.8 (2025); levels depend on the frontier-scale M*, which no public
design observes (technology range 0.17–0.85), but the sign is identified for 86% under weak assumptions and
parametric extrapolation is conservative where it can be checked. (4) Economics: the least portable technology object
(a) drives data-demand forecasts (2.0–2.8×/yr); σ barely matters for the data wall, the repetition technology does.

## 1. Front matter
- Title: What Optimizing Labs Reveal: Scaling Laws as Production Functions.
- Authors: Yigit Okar (Care AI) and Claude (Anthropic) — as the user requested. \thanks footnote: affiliations;
  disclosure: "Claude is an AI system developed by Anthropic. It designed and carried out the analysis and drafted
  the paper together with the first author, who takes responsibility for its content." Data/code: "Code and data
  construction scripts are available from the authors (replication package: scaling-laws-pf repository)."
- Running heads: NO journal name/volume/issue. Use a working-paper head (the integrator edits AEA.cls macros or sets
  \pubMonth etc. so that heads read e.g. "WORKING PAPER — SEPTEMBER 2026").
- Abstract ≤ 100 words, robust claims only (template in R3 §7):
  "Developers of language models choose model size and training data to minimize the cost of training and serving.
  We show that this optimization cuts both ways for measurement. It removes the variation that identifies the
  training technology — information about curvature is fourth order in developers' allocation errors — but makes
  their choices informative about their objectives. Using designed training experiments, we estimate, without
  functional-form restrictions, an elasticity of substitution between parameters and data of about 0.7. Inverting
  developers' first-order conditions, open-weight models released since 2024 are built as if serving or other value
  of compactness accounts for most of lifetime cost, up from near zero in 2021." (edit to ≤100 words; final numbers)
- JEL: C51, D24, L86, O33. Keywords: production functions, identification, elasticity of substitution, scaling laws,
  large language models, revealed preference.

## 2. Structure, lengths, exhibits (main text ≈ 14,000 words; ≤ 9 exhibits)
(Intro, no heading, ≤1,800 words) → sections/introduction.tex
  Order: stakes (≈$ training investment; compute growth 5×/yr [ra3]; scaling laws plan it) → the question → the idea
  (double-edged optimization) → three results with ≤15 numbers → what is robust and what is not (one paragraph) →
  closest papers and what is new relative to each (Hao & Merrill 2026; Kricheli et al. 2026; Mertens et al. 2026;
  Whitfill 2025; Sardana et al. 2024 + Bian et al. 2025 + Roberts et al. 2026 (forward problem); Bond & Söderbom 2005,
  ACF 2015, GNR 2020, DLW 2012 + Raval 2023 (IO)) → one sentence pointing to Appendix E for observational/progress
  results → roadmap.
I. The Training Problem (≈2,300) → sections/framework.tex
  A. Technology and cost: Chinchilla/κ family; quasi-homotheticity; C ≈ 6ND as FLOP-accounting approximation;
     output y = −ln(L − E); ordinal vs cardinal. Figure 1 = m7_theory_geometry.
  B. Duality (Prop 1, short; "IsoFLOP-minima frontier"; Hoffmann's Approach 1 also estimates factor demands).
  C. Substitution (Lemma 1: interior optima ⇒ 0 < σ < 1; empirical content = U-shaped IsoFLOP profiles; σ(w); σ*).
  D. What over-training reveals (Prop 2, generalized wedge; m_N; special cases table in Appendix A/F; s = (w−1)/w;
     family token budgets: individual wedges not revealed preferences, family reveals a weighted condition).
     A compact dictionary (Table 1, ≤ 8 rows) OR move the dictionary to Online Appendix D and state analogies in text
     where used — writer's choice; prefer a ≤8-row table.
II. What Optimizing Labs' Data Identify (≈2,000) → sections/identification.tex
  A. Prop 3 (identification + information rates; global vs local; n^{-1/4}; positioning: Marschak–Andrews,
     Bond–Söderbom, GNR, optimal design Box–Lucas/Kiefer–Wolfowitz, Sargan/Rotnitzky). Scope: binds for
     compute-optimal ladders and designs with fixed ratios (DataDecide D=100N; Pythia fixed D); released models lie
     far off path, and heterogeneous values of compactness are the exclusion restriction that gives variation.
     Figure 2 = m6 Design A, two panels (RMSE of σ* and ln M* vs allocation error s; κ-free profile) from the
     re-run WITHOUT the truth in the start set: file m6_montecarlo_fig2_v2 (produced by the mc-fix agent).
  B. Prop 4 (model-free σ* and w) — the constructive answer: IsoFLOP curvature / (2 × frontier slope); local slopes.
     Finite-grid bias of quadratic fits (quartics fix it; ra5 Remark).
  C. Prop 5 (partial identification of M*(C) at frontier scale and of the sign of w − 1: sign(w − 1) =
     sign(ln M − ln M*(C)) for any single-peaked IsoFLOP technology).
III. The Technology from Designed Variation (≈2,800) → sections/technology.tex
  A. Designs (Table 2: designs with runs, N range, M range, design statistic, output, and σ* estimates:
     model-free, κ free, κ = 1 — built from ra1_modelfree_sigma.tex + m2 Table 3 + ra1 kappa robustness).
  B. Model-free σ*: Figure 3 = ra1_modelfree_sigma_by_design. Numbers: Chinchilla 0.673 (0.027), Llama 3 0.660
     (0.023), Marin 0.700/0.713/0.705, Farseer path 0.708 (first-derivative estimator; bandwidth-stable), Porian
     0.518/0.505; random effects excl. Porian 0.695 [0.673, 0.717], τ = 0, Q = 4.1 (low power below τ ≈ 0.02);
     budget drift (non-homotheticity) weak/window-sensitive. Discuss Porian (constant LR schedule? small scale).
  C. Parametric forms: κ = 1 rejected; κ-free agrees with model-free on large designs; Llama 3 κ = 1 σ* 0.769 is 4.8 SE
     above model-free; κ-free heterogeneity across the 7 sweep–corpus technologies Q = 82.1, τ = 0.068, RE 0.622
     [0.549, 0.695] (small sweeps; E–κ trade-off). σ < 1 is implied (Lemma 1), not a finding.
  D. What is not portable: a (0.36–0.57), M*(1e21) (3–60), conventions (embedding counts; Porian decomposition:
     39–47% measurement, 53–61% flexible inputs → Appendix D). Data quality: Gadre (neutral + E-shift), DataDecide
     (factor-biased tilt 0.22–0.26 → M* ×2.9–3.4); DeepSeek LLM (Bi et al. 2024) precedent.
  E. Our controlled experiment (m9): design paragraph + pre-registration (plan committed before estimation) +
     results placeholders \textcolor{red}{[TBD-m9]}: σ* by corpus (model-free + κ-free), neutrality/tilt on both
     validation sets and WikiText, high-M extrapolation test, seed noise, LR checks. Figure 7 placeholder (fig:experiment).
IV. What Over-Training Reveals (≈3,000) → sections/wedge.tex
  A. From tokens per parameter to the value of compactness: the inversion ln w = (1/σ* − 1) ln(M/M*(C)): the
     technology supplies the zero point M*(C) and the scale (1/σ* − 1); ordinal content = M/M*; report s.
  B. Sample and technologies: clean inference-demand sample (77 models, 36 families, 18 developers; cleaning table in
     Appendix F); ex-ante technology set (32); reference = Chinchilla κ-free; lab-own where available (Meta with
     model-free σ* 0.660; AI2 OLMo; DeepSeek LLM; Marin).
  C. Results: Figure 4 = ra2_wedge_ecdf; Table 3 = ra2_wedge_models_selected (condensed). Median s 0.75 [0.69, 0.79]
     (w 3.99); 97% over-trained; technology range of median s 0.17–0.85; κ-free/model-free curvature technologies
     0.69–0.79; lab-own 0.61 (22 models). Llama 3 8B under Meta's own law and curvature: w 8.4 [5.7, 12.8], s 0.88;
     alternatives 3.1–12.2; 405B on Meta's path by construction.
  D. Extrapolation and partial identification: Figure 5 = ra1_modelfree_farseer_w (parametric understates w at high M
     inside Farseer: κ-free 0.70–0.78×; Chinchilla 0.44–0.50×; fit on M ≤ 100 0.26–0.30× ⇒ extrapolated levels
     conservative in M); ln w convex in ln(M/M*); sign identified for 86% (IsoFLOP anchors), 77% (own-lab anchors), 23%
     (any technology as anchor); no w < 1 identified; inside Chinchilla's M range median s 0.56. Nothing observed
     beyond Farseer's M range or at frontier C.
  E. Conduct and interpretation: open-weight premium +0.52 log points (WCR p = 0.011); serving footprint 0.45
     (p = 0.39; 7 of 18 developers serve); on-device untestable; tier bunching 51% vs 13% but no tier-specific
     over-training; by serving footprint: s 0.80 (developer serves) vs 0.74 (others) ⇒ "planned serving expenditure"
     reading applies to the first group; for others s is internalized value of compactness. Family budgets: common-D
     families median ŵ 3.90 vs 4.01 size-specific.
  F. Validation and trends: HF derivative counts rise with ln M at given C (0.80, p = 0.031); OpenRouter volumes not
     accessible (ToS); cost-side sensitivity small (s 0.72–0.77 for δ ∈ [−0.1, 0.1]); trends: aggregate planned
     serving share 0.04 (2019–22) → 0.35 (2023) → 0.54 (2024) → 0.82 (2025) under the reference (ra3; ra2 H9);
     technology range for 2025 [0.24, 0.96]. Figure (optional) = ra3_econ_inference_share or ra2_wedge_trends.
V. Economic Implications (≈1,000) → sections/economics.tex (NEW FILE; add to main.tex after wedge)
  Table 4 = ra3_econ_table; Figure 6 = ra3_econ_figure. (i) Data demand: compute 5.06×/yr [4.29, 5.93] (2018–2026);
  data demand 2.02–2.84×/yr across a ∈ [0.36, 0.57]; 34–183× over five years; frontier runs reach 100T unique tokens
  2026.6–2027.8 and the 320T effective stock 2027.8–2029.6 (cf. Villalobos et al. 2024). (ii) Data wall: at r = 4,
  extra compute 7.2/7.4/7.7% for σ* = 0.74/0.70/0.60 under data-only decay, 43% under Muennighoff's full model (hard
  wall r ≈ 9): σ barely matters, repetition technology and M* do; shadow value ≈ 0.43 token-processing costs at r = 4,
  1e26 FLOP; compute–unique-data elasticity 0.28–0.58. (iii) Inference share: planned vs disclosures (Google ~0.6
  of ML energy 2019–21; Meta 10:20:70) — loose order-of-magnitude agreement; (iv) growth models: halving reducible loss
  costs 2^{1/γ} = 49–161× compute (2.4–3.1 years of frontier growth).
VI. Conclusion (≤700) → sections/conclusion.tex. What economists should use (model-free σ* ≈ 0.7; γ; s as ordinal/
  conditional); design recommendations for scaling studies (IsoFLOP curvature estimator; off-path share; test κ;
  design-conditional inference; report M* uncertainty); limitations; extensions (train/test-compute isoquant; market
  structure).
Online Appendix (after references; \appendix): A proofs (appendix_proofs.tex, ra5 v2); B data (appendix_data.tex);
  C Monte Carlo (appendix_mc.tex); D technology: additional results (appendix_additional.tex → rename content: Chinchilla
  duality/selection/inference [ra1 chinchilla_inference], sweeps robustness, κ robustness, practitioner overhead
  table, measurement & flexible inputs [m8], dictionary table if moved); E observational production functions and
  algorithmic progress (NEW appendix_observational.tex from ra4 tables E1–E7 + m5 replication); F revealed demand
  robustness (NEW appendix_wedge.tex: cleaning, 32 technologies, conventions, MoE bounds, PI tables, conduct, cost
  sensitivity, family budgets, validation).

## 3. Style and claims
- AER style as before; ≤ one interval per sentence in the main text; result first, caveat second.
- Main-text formal results: Prop 1, Lemma 1, Props 2–5 (labels per theory_main_text_v2 §0).
- Do NOT claim: first to read scaling laws economically / compute σ / note confounding; "gross complements" as a
  finding; cardinal T in tokens without p; "planned inference" for developers who do not serve (say "value of
  compactness"); that w levels are identified at frontier scale; that the observational benchmark shows no bias (say
  low-power failure to reject; Appendix E); "LaLonde"; Nerlove "spurious"; "within 4e-4"; "exactly 2E[ln w]/(α+β)".
- Terminology: "verified sample" not "Sample B"; "clean inference-demand sample"; corpora not labs in the experiment;
  "IsoFLOP-minima frontier"; E irreducible loss, 𝔼 expectation; doubling time τ_C (not T_C).
