# Paper plan — "Scaling Laws as Production Functions" (AER, full article)

Lead author's plan, 2026-09-24. Binding for all section writers. Numbers must be taken from the module
memos/CSVs named here (after the consistency rebuild), never from memory. When this plan and a memo
disagree, the memo (and its `_review.md`) wins; flag the discrepancy.

## 0. The paper in one paragraph (the pitch)

Neural scaling laws — the relation between a language model's loss and its parameters N, training tokens D and
compute C ≈ 6ND — are production functions, and the empirical practice of fitting them is production-function
estimation. Reading them this way does three things. (1) It organizes what the data can identify: the standard
approaches are cost-function, conditional-factor-demand and primal estimators; cost-minimizing behavior removes
the (transverse) variation that identifies curvature, so the better labs optimize, the less their data reveal;
and the compute-optimal tokens-per-parameter ratio M* — the object every allocation decision needs — is the
fragile, weakly identified object, whereas the frontier elasticity is robust. (2) It yields estimates with
economic content: across seven public sweeps and our own controlled experiment, parameters and data are gross
complements (σ < 1, as any interior compute optimum requires), with σ ≈ 0.7 once the data choose the curvature
(≈ 0.74–0.83 under the Chinchilla form, whose restriction every sweep rejects). (3) It turns labs' choices into
data: inverting the first-order condition of lifetime-compute minimization, over-training reveals planned inference
demand. The median open-weight base model is trained as if it expected lifetime inference compute of about 2.2
times its training compute (w ≈ 3.2), up from under-training before 2022. Finally, the IO lens disciplines
observational scaling claims: with design matched, cross-lab returns to compute agree with experiments, but the
parameter–data split and "algorithmic progress" are weakly identified (DMR ridge; optimizer artifacts; an
allocative component of 1–2.4× that is technology-dependent).

NOTE on the aggregate: w = lifetime/training compute, so T/D = 3(w − 1) is lifetime inference tokens per
training token and inference compute / training compute = w − 1 (2NT/(6ND) = T/(3D) = w − 1). Median w = 3.19 ⇒
median inference/training compute = 2.19. Be precise about which ratio is quoted.

## 1. Front matter

- Title: **Scaling Laws as Production Functions**
- Authors: Yigit Okar (Care AI) and Claude (Anthropic). Footnote: affiliations; "Claude is an AI system developed by
  Anthropic; it designed and carried out the analysis and drafted the paper with the first author." Data and
  code availability statement; acknowledgments minimal.
- Abstract ≤ 120 words (AER). Draft:
  > Neural scaling laws relate a language model's loss to its parameters, data, and compute. We show that they are
  > production functions and that fitting them is production-function estimation: the standard approaches are
  > cost-function, factor-demand, and primal estimators, and cost-minimizing labs generate data that cannot identify
  > the technology's curvature or its optimal input mix. Across public training sweeps and our own experiments,
  > parameters and data are gross complements, with an elasticity of substitution near 0.7. Inverting labs' first-order
  > conditions, we show that over-training reveals anticipated inference demand: the median open-weight model is
  > built as if lifetime inference will use twice its training compute.
- JEL: D24, L86, O33, C51 (+ L11 optional). Keywords: production function estimation, elasticity of substitution,
  scaling laws, large language models, identification, revealed preference.

## 2. Section structure, lengths (AER pages ≈ 500 words), content, sources

AER house style: no "Introduction" heading (start text after \maketitle); sections I., II., ... via \section;
subsections A., B. via \subsection; propositions numbered in the main text; proofs in Online Appendix A.
Target total main text ≈ 16,000–18,000 words (≈ 40–45 pages at AER spacing) + online appendix.

### (Intro, no heading) ≈ 2,300 words — file sections/introduction.tex
Paragraph plan:
1. Hook: compute spending on LLM training (Epoch: frontier training costs growing ~2.4x/yr; cottier2024rising if in bib),
   the scaling law as the planning tool of this investment (kaplan2020scaling; hoffmann2022training). Scaling
   laws are production functions; their estimation is production-function estimation — a problem economists have
   studied for 80 years (marschak1944random; griliches1998production; ackerberg2007econometric).
2. What the lens delivers (three results, each with a headline number):
   (i) Identification: duality; functional dependence from optimizing behavior (ACF); information theorem
       ("the better labs optimize, the less their data reveal": information about M* is second order in the
       dispersion of log wedges, about curvature fourth order); interior compute optima imply σ < 1; DMR refined
       (sign of the bias is identified from drift of M*, magnitude and σ not).
   (ii) The technology: σ between parameters and data ≈ 0.7 (Farseer: 0.69–0.71 across three estimators;
       Chinchilla-form 0.74–0.83 across 7 sweeps; κ=1 rejected everywhere); frontier elasticity γ robust; a and M*
       not portable (M*(1e21) from 3 to 60 across sweeps). Our own controlled experiment (two "labs" differing in
       data quality) [numbers from m9 when available].
   (iii) Revealed inference demand: w = ε_N/ε_D = 1 + T/(3D); median w = 3.19 for 173 open-weight base models; 79%
       have the whole technology band above 1; share w<1: 47% (2021) → 0% (2024); Llama 3 8B w = 5.27 (T/D = 12.8;
       6.3 under Meta's own law); 405B flagship on Meta's own path (0.98). Validation: over-trained models are
       downloaded more at fixed compute (elasticity ≈ 1).
3. Observational scaling and algorithmic progress (brief): design-matched LaLonde-type test finds no detectable
   bias in returns to compute; parameter–data split unreliable; Ho et al. replicated exactly but point estimate
   is an optimizer artifact on a DMR ridge (8.7 vs 6.1 months; profile CI [4.1, 40.5]); allocative share of
   Kaplan→Chinchilla 1.0–2.4×.
4. Related literature (≈ 700 words), organized:
   (a) Neural scaling laws and their estimation: hestness2017deep, kaplan2020scaling, hoffmann2022training,
       besiroglu2024chinchilla, pearce2024reconciling, porian2024resolving, muennighoff2023scaling,
       sardana2024beyond, gadre2024language, li2025predictablea (Step Law), li2025predictableb (Farseer),
       choshen2024hitchhikers, li2025misfitting, czech2026problems, kricheli2026tokens.
   (b) Economic readings of scaling: hao2026theory (σ formula, Leontief profit model — we estimate σ, add σ*,
       identification and the interior-optimum ⇒ σ<1 result, and show σ>0 is what rationalizes over-training),
       mertens2026secret (developer FE — we add design-matched benchmarks and gross-output contamination),
       whitfill2025note and konig2026validity (confounding — we supply IO structure: timing, regimes, bounds),
       ho2024algorithmic, erdil2022algorithmic, gundlach2025origin (progress — DMR, allocative vs technical),
       ruan2024observational, maiapolo2024sloth, thompson2020computational, bergemann2025economics,
       korinek2025concentrating (check keys), demirer2025emerging, erdil2025gate, trammell2023economic.
   (c) IO production-function estimation: marschak1944random, mundlak1961empirical, zellner1966specification,
       olley1996dynamics, levinsohn2003estimating, ackerberg2015identification, gandhi2020identification,
       deloecker2012markups, raval2023testing, bond2021some (check key: bond2020unpleasant?),
       doraszelski2018measuring, demirer2020production, diamond1978measurement, leonledesma2010identifying,
       klump2007factor, nerlove1963returns, farrell1957measurement, foster2008reallocation, syverson2011determines,
       hsieh2009misallocation, lalonde1986evaluating (check), collardwexler2016production.
   Positioning sentence (from lit/notes/novelty.md "Recommended positioning statement"): we do not claim to be the
   first to read scaling laws economically; we claim the systematic mapping to the econometric IO toolkit, the
   identification results, the estimates with inference, and the revealed-demand inversion.
5. Roadmap paragraph.

### I. Scaling Laws and the Production Problem ≈ 2,200 words — sections/framework.tex
A. Technology, cost and output. Notation (model_spec.md §1; m7 theory_main_text.md). L = E + A N^-α + B D^-β;
   C = 6ND (multiplicative cost ⇒ unit log-cost weights ⇒ FOC equates output elasticities ε_N = ε_D, no price data
   needed); y = −ln(L−E); isoquants = iso-loss; isocost = IsoFLOP; ordinal vs cardinal objects.
B. The dictionary: **Table 1** (condensed ≈ 18 rows) from lit/SYNTHESIS.md §4 — ML concept | IO concept |
   what the mapping delivers. Write it as a LaTeX table (tables/table1_dictionary.tex).
C. Duality (Proposition 1): expansion path N* = G(C/6)^a, frontier L*(C) = E + K(C/6)^−γ, a = β/(α+β),
   γ = αβ/(α+β); Chinchilla Approaches 1/2/3 = cost function / conditional factor demand / primal with
   cross-equation restrictions.
D. Substitution: σ formula, σ(w) = (1+w)/(1+α+w(1+β)) (m7), σ* = 2/(2+α+β); Lemma: interior compute optima
   under C=6ND require 0<σ<1 (gross complements) (m7 Lemma A2/Cor A1); Kaplan's form implies σ ∈ [0.50, 0.575].
   Credit hao2026theory for the σ formula.
E. Wedges and efficiency: lifetime cost 6ND + 2NT ⇒ w ≡ ε_N/ε_D = 1 + T/(3D) (Proposition 2, revealed
   inference demand); sufficient statistic ln w = (1/σ* − 1) ln(M/M*(C)); Farrell cost efficiency
   C/C_min = ((α+βw)/(α+β))^{1/γ} w^{−1/α} (free of A,B,E,ω); rival wedges signs (data constraint ⇒ w<1;
   memory constraint ⇒ w>1 without T; factor bias ŵ = w e^{−χ}). Analogy: DLW markup (ratio of output elasticity
   to cost share) and Hsieh–Klenow wedge — but it is a shadow cost of a second use of the same input, not a markup.
   Figure 1 = m7_theory_geometry (or m1_chinchilla_isoquants; choose m1_chinchilla_isoquants for the data version
   in Section III, and m7_theory_geometry here).

### II. What Scaling Data Identify ≈ 2,800 words — sections/identification.tex
A. Functional dependence from optimizing behavior (Proposition 3 = m7 Prop A1/A2 + Lemma: rank-one Hessian with
   null direction = expansion path; on-path data identify (a, G) and (E, γ, K); α, β only via κ=1; σ* not at all
   if κ free; M* only by assuming optimality). ACF analogy: in ACF labor is a deterministic function of state
   variables; here (n, d) are deterministic functions of c. IsoFLOP sweeps = experimenter-made input-price shocks.
B. Information (Proposition 4 = m7 Prop A2): y_i = γ(c_i − ln6) − lnK − γΦ_i − ε_i; information on ln M*
   ∝ dispersion of log wedges (second order), on S=α+β (hence σ*) fourth order. "The better labs optimize, the
   less their data reveal." Monte Carlo evidence: **Figure 2** = m6_montecarlo_fig2_designs; numbers from m6
   memo claims 1–5 (RMSE σ* 0.28 on-path vs 0.0036 IsoFLOP; κ-free profile CI spans [0.50,0.95] in 88% of on-path
   reps; warm-start bootstrap coverage 67%).
C. Technical change over time (Proposition 5 = m7 Prop A3, CORRECTED DMR): drift of M*(C,t) identifies the
   sign of the Hicks bias; magnitude and σ are not identified; neutrality ⇔ no drift in compute-optimal M at
   given C; CEG constant iff equal E and γ (m7 claim 7; P4 as stated in SYNTHESIS is FALSE — do not repeat it).
D. Observational data (Proposition 6 = m7 Prop A5/A6): transmission bias has the sign of the compute response
   to productivity (regimes: exogenous budget ⇒ consistent; funding ⇒ upward; target ⇒ toward zero;
   predetermined ⇒ upward, attenuated by ρ); forward–reverse bracket condition; outcome-based release attenuates;
   Hicks-neutral transmission biases γ, not a; D/N is a valid proxy for factor-biased (χ) but not Hicks-neutral
   productivity (m7 claim 12) — the OP/LP inversion is unavailable for the productivity that matters most.
E. Partial identification of revealed demand (Prop 7 = m7 Prop A9): Var(ln ŵ) quadratic in ln M; M* dominates
   (sd 0.143 of 0.161); nonparametric monotone lower bound (uninformative on current support; needs over-trained
   runs at scale). Motivates within-family calibration (GNR direction).
Keep proofs out; cite "Online Appendix A" (appendix_proofs.tex, which m7 wrote; the integrator must reconcile
numbering between main text and appendix).

### III. Data ≈ 1,000 words — sections/data.tex (NEW FILE; add \input to main.tex after identification)
A. Designed sweeps ("experiments"): Chinchilla extraction (245/240), Farseer (404), Gadre (104; 3 corpora),
   OLMo ladder (30), Muennighoff (single-epoch subset), DataDecide (25 recipes × 14 sizes × 3 seeds; checkpoints),
   Porian, Step Law (1,911 runs), Meta Llama 3 and Marin IsoFLOPs (open-athena). **Table 2**: datasets, runs,
   N range, M range, off-path design statistic sd(ln M | ln C), output units, license. Build from m2 Table 3 inputs
   and m1/m8 memos (tables/table2_data.tex).
B. Our controlled experiment (two "labs"): FineWeb-Edu vs FineWeb, shared 8,192-token BPE tokenizer, 8 sizes
   (0.4M–49M non-embedding params), D from 25M to 800M tokens, WSD with cooldown branches (hagele2024scaling),
   learning rate calibrated per width (lr* ∝ d^−0.90), common random numbers; evaluation on both validation sets.
   [numbers and design table from m9 when available]
C. Observational data: Sample B (173 verified open-weight base models, 2021–2025, exact HF parameter counts);
   Epoch production-scale universe (335 models with 6ND ≥ 1e21); ObsScaling/Sloth/Epoch benchmark panel for the
   LaLonde test (m4); Ho et al. (2024) data (231 rows); usage data (HF downloads, OpenRouter, LMArena).
   Data construction details → Online Appendix B (appendix_data.tex).

### IV. The Technology ≈ 3,000 words — sections/technology.tex (rename from 'experiments'; see main.tex)
A. Canonical data (Chinchilla): Huber(1e-3) = LAD in practice (84% residuals in linear region); Besiroglu's
   published values are LAD-type; estimator horse race (β 0.367 Huber vs 0.406 Gaussian vs 0.428 levels; paired
   differences significant z=2.2–3.0; σ* 0.718–0.737); cluster bootstrap (9 IsoFLOP budgets) doubles SEs;
   outliers in the data-starved corner (not truncation bias; Hausman–Wise = untruncated); duality tests:
   Besiroglu/refit consistent with IsoFLOP argmins (p ≥ 0.16), Hoffmann A3 marginally inconsistent (p ≈ 0.04–0.06,
   mainly path level); revealed preference at Chinchilla-70B: w = 1.04 [0.82, 1.42] refit vs 0.62–0.71 Hoffmann;
   system estimator; κ̂ = 0.774 (κ=1 rejected, p=0.0002 pairs, 0.042 cluster); functional dependence in the data
   (on-path band: condition number 413 vs 62; κ-free profile flat on-path, LR ≤ 1.96, vs 424 full design).
   **Table 3** = m1 horse race + spec tests condensed; **Figure 3** = m1_chinchilla_isoquants (data geometry)
   and/or m1_chinchilla_profile_sigma.
B. Across sweeps: **Table 4** = m2_table3_technology (σ*, a, γ, M*, κ/q tests by dataset); **Figure 4** =
   m2_fig3_sigma_forest. Claims: σ<1 everywhere; q=1 rejected in all 7 sweeps; σ*_q 0.51–0.71; Farseer three
   estimators 0.69–0.71; a 0.37–0.57 and M*(1e21) 3–60 (not portable); CES not rejected (Wald); Chinchilla
   extrapolates poorly in compute on Farseer (RMSE 0.022–0.025 vs 0.0026). Place σ next to capital–labor range
   (0.4–0.7: chirinko2008sigma; oberfield2021micro — check keys) carefully: q-free estimates overlap its top.
   DO NOT say "σ is more stable than a" (m8 review M1): say "the frontier elasticity and σ are comparable across
   sweeps in point estimate; M* — which depends on A/B — is not."
C. Data quality: is better data neutral or factor-biased? Gadre: Hicks-neutral + E-shift not rejected (p=0.90);
   DataDecide: every restriction rejected, E-shifts explain 98%, recipe tilts imply M* varying 2.9–3.4× and the
   wedge 1.25–1.29×. Our experiment (m9): FineWeb-Edu vs FineWeb [numbers TBD].
D. Measurement and flexible inputs (m8): Porian path 0.837 → 0.496; decomposition 39–47% input measurement
   (head/embedding FLOPs; Nerlove small-firm economies; Collard-Wexler & De Loecker) and 53–61% flexible inputs
   (tuning; concentrated production function); Proposition M (a_m = β/(β+ακ−η); Pearce–Song range 0.792/0.757);
   Chinchilla with non-embedding N: a 0.514 → 0.556; Step Law SFA: inefficiency D-biased (γ_D = −0.23);
   LR demand elasticities replicate Step Law (N −0.82, D +0.29); the Bjorck–Step Law sign disagreement only
   26–43% explained by conditioning on batch. **Table 5** = m8_measurement_porian (+ sfa condensed).
E. Our controlled experiment (m9) — [TBD after the sweep: σ by lab with κ free; neutrality of data quality;
   extrapolation check at high M (is ε_D overstated at M ≈ 1,000–2,000? direction of wedge bias); seed noise].

### V. Over-Training Reveals Anticipated Inference Demand ≈ 2,800 words — sections/wedge.tex (NEW FILE)
A. Measurement: technologies (Table: m3_wedge_technologies condensed into notes), Sample B, bands.
B. Results: **Figure 5** = m3_wedge_fig4; **Table 6** = m3_wedge_table5 (condensed to ~15 rows). Median w 3.19
   (T/D 6.6); 95% w>1; 87% bootstrap lower bound >1; 79% entire band >1; technology dispersion 1.91 (Hoffmann) –
   4.03 (Gadre RW). Llama 3 8B, 405B flagship; Farrell CE 0.18 for Llama 3 8B.
C. Stated intent check (compute-optimal designs at w≈1; inference-oriented designs >1).
D. Trends: **Figure 6** = m3_wedge_trends; share w<1 47%→0%; median 1.06 (2022) → 4.68 (2025); open-weight
   premium +0.57 (0.09) / +0.44 (0.09) given ln C; developer medians (HF 6.6, Alibaba 4.8, Google 2.3, Meta 1.8,
   Cerebras 1.0).
E. Within-family revealed preference (GNR direction) and over-identification (Raval-style): 95% of siblings
   w_rel>1; flagship-on-path holds pre-2024 (median 1.14) fails post-2024 (3.14) under common technologies;
   Meta flagships 0.98 under Meta's law; single demand elasticity rejected mostly mechanically; informative
   subsample p=0.047/0.12.
F. Validation: **Table 7** = m3_wedge_validation (ln M at fixed ln C, year, age: all-time downloads 0.97 (0.18),
   0.72 (0.14) with developer FE; 30-day 1.13/0.94; OpenRouter providers 0.08 (0.04); LMArena 0.29 (0.16)).
   Honest caveat: equivalent to "smaller models at given compute are downloaded more".
G. Rival wedges: data scarcity pre-2024 only; hardware tiers 26% bunching, +0.11 (0.06) on ln w; non-homotheticity
   (Farseer Eq. 3) explains large models only; distillation null; factor-biased recipes ≤1.29× (≤1.8×) cannot
   overturn 79–88%.
H. Aggregate (illustrative): compute-weighted planned lifetime inference compute / training compute = Σ(w−1)C/ΣC
   = 2.0 [1.2, 3.7] all years; 3.2 [2.1, 5.7] for 2025; technology band [0.4, 7.2]; 0.67–3.26 across technologies.
   Caveat: dominated by largest (most extrapolated) models; not validated against inference volumes.

### VI. Observational Production Functions and "Algorithmic Progress" ≈ 2,200 words — sections/observational.tex (NEW)
A. A LaLonde-style benchmark (m4): experimental technology on the observational design; **Figure 7** =
   m4_observational_fig5_lalonde; **Table 8** = m4_observational_table6_lalonde (HellaSwag) with ARC/Winogrande in
   appendix. Results: OLS θ_C bias −0.001 (0.038); biases −0.03 to +0.04 across 6 estimators; design explains
   the "data elasticity gap"; family FE N/D split unreliable (θ_N +0.20, θ_D −0.26); regimes: data inconsistent
   with strong target or funding rules; IO remedies infeasible (weak IV F ≤ 3.8; AB infeasible; export-control IV
   exclusion not credible). TFP dispersion: family 90/10 23.7× compute-equivalent [9.1,197], 7–9× without code/
   distilled/synthetic families; within-developer 11.1× vs Mertens' 41×; synthetic-data families (Phi, SmolLM)
   rank 1st/3rd (gross-output contamination).
   Industry Monte Carlo summary (m6 claims 6–13): within-family estimation = observational IsoFLOP; ACF works only
   with within-family variation; GNR/FOC system confounds inference demand with technology (ln M* bias +1.28);
   ML practice absorbs >50% of progress into the scaling law (Sahal); refer to Online Appendix C tables.
B. Algorithmic progress (m5): Ho et al. replication exact (objective 0.0518129; T_C bootstrap median 8.44;
   point 8.68 months); published point not the minimizer (0.0507227 at T_C = 6.1, CI [3.0,22.7]); estimator
   choices 6.1–10.2 months; DMR ridge (corr −0.73 to −0.82; profile CI [4.1, 40.5]); Hicks neutrality not rejected;
   imposing it gives 9–12 months; experimental exponents rejected in cross-lab data (MSE ×1.3–4.8); like-for-like
   frontier-elasticity gap 2–2.5× not 7–9×; D measurement (epochs) drives T_C (1.5–4×). **Figure 8** =
   m5_progress_dmr_ridge (+ fig6 allocative in appendix) ; **Table 9** = m5_progress_table7 condensed.
C. Allocative vs technical: realized allocative gains of Kaplan→Chinchilla rebalancing 1.0–2.4× (C ≥ 1e23;
   Besiroglu/Hoffmann/Farseer), 2–40% of Ho-rate gain; technology-dependent (0.55–4.3× across sweep technologies;
   even the sign); Gundlach's 10× is the counterfactual cost of keeping Kaplan's rule at 2025 compute (11.7× at
   5e26). Sahal inflation 1.15× (all) to 1.44× (record-setters).

### VII. Conclusion ≈ 900 words — sections/conclusion.tex
What economists should use (σ ≈ 0.7 range; γ; w; not raw exponents or M*); design recommendations for ML
scaling studies (off-path share; report κ/q tests; cluster by budget; design-consistent duality tests;
normalization; LAD≈Huber; report M* uncertainty); implications for AI growth models (complementarity; inference
share of compute); limitations (units/tokenizers; extrapolation; disclosure selection; Chinchilla form);
extensions (training vs test-time compute isoquant; Sutton endogenous sunk costs and market structure).

### Online Appendix (separate from main text but in the same PDF after references)
A. Proofs (appendix_proofs.tex — m7; integrator reconciles numbering and cross-refs with main text).
B. Data construction (appendix_data.tex): sweeps harmonization (m2 §2.1), Chinchilla extraction details (m1),
   Sample A/B construction (m3 data_appendix.md), observational panel (m4), Ho data (m5), our sweep (m9).
C. Monte Carlo design and full tables (m6: designA.tex, industry.tex, transmission.tex).
D. Additional results: m1 selection/duality tables, m2 robustness/neutrality/farseer_forms, m3 family/rivals/
   aggregate/homothetic, m4 ARC/Winogrande/core/clean/ladder-only tables, m5 replication/attenuation/CEG-Sahal,
   m8 steplaw/demand/SFA/biasformulas, m7 claims register.

## 3. Style rules
- AER prose: clear, unhyped, first-person plural. No "novel"/"groundbreaking". Define every symbol at first use.
- Every number in text must match a CSV/memo; round sensibly (2–3 significant digits); give SEs or CIs.
- Hedge exactly as the memos' caveats require. Prefer "we find", "the data do not reject".
- Tables: booktabs, threeparttable, \begin{tablenotes} with Notes: and Source:; caption above table (AER: "Table 1—Title").
  Figures: caption below; \begin{figurenotes} if available in AEA.cls else a small paragraph.
- Cross-reference with \label/\ref (Section~\ref{sec:...}, Table~\ref{tab:...}, Figure~\ref{fig:...},
  Proposition~\ref{prop:...}). Use \citet/\citep (natbib, author-year).
- Math: y, n, d, c lower-case logs; M = D/N; w; σ*; γ; a; κ (outer exponent; Chinchilla κ=1). Use \ln.
- Do not claim: "first to interpret scaling laws as production functions"; "first to compute σ"; "first to note
  confounding"; "σ is more stable than a"; "on-path data cannot sign the bias"; "constant CEG iff equal exponents";
  "w is a markup"; that revealed T levels are measured precisely; that IO remedies work on public cross-lab data.

## 4. Figures and tables (main text)
Fig 1 m7_theory_geometry; Fig 2 m6_montecarlo_fig2_designs; Fig 3 m1_chinchilla_isoquants;
Fig 4 m2_fig3_sigma_forest; Fig 5 m3_wedge_fig4; Fig 6 m3_wedge_trends; Fig 7 m4_observational_fig5_lalonde;
Fig 8 m5_progress_dmr_ridge; Fig 9 (m9) own experiment.
Table 1 dictionary; Table 2 data; Table 3 Chinchilla estimators/tests; Table 4 technology by sweep;
Table 5 measurement (Porian); Table 6 revealed demand by family; Table 7 validation; Table 8 LaLonde;
Table 9 progress. Figures are included from ../output/figures/<name>.pdf (graphicspath set).
