# Strand notes: Economics of AI, compute, LLM markets, and economic readings of scaling laws

Project: "Scaling Laws as Production Functions" (AER-style). Strand owner notes. Compiled 2026-09-23.
BibTeX: `lit/bib/econ_ai.bib` (keys in [brackets] below). Anything I derived myself, rather than read in a
source, is marked **[DERIVED HERE]** and still needs checking in code before it goes into the paper.

---------------------------------------------------------------------------------------------------

## 0. Bottom line for the paper (read this first)

1. **Economists calibrate the scaling-law technology but have not estimated it with IO-grade econometrics.** Theory papers
   write down a technology and pick parameters:
   - Cobb-Douglas or homogeneous CES in *inference* tokens [bergemann2025economics; bergemann2026menu];
   - log-linear "automation share vs. effective compute" [erdil2025gate];
   - a Pareto task-compute distribution [korinek2024scenarios];
   - productivity = data^eta, justified by Chinchilla [farboodi2026datadriven];
   - "compute and data are gross complements", asserted in a footnote [trammell2023economic].

   No paper I found applies proxy-variable or FOC-based production-function estimators to training-run data: no
   Olley–Pakes / Levinsohn–Petrin / ACF, no Gandhi–Navarro–Rivers, no dynamic panel. The closest empirical work:
   - pooled OLS or ML with a year trend or year FE [thompson2020computational; besiroglu2024economic; erdil2022algorithmic; ho2024algorithmic];
   - one relative-factor-demand (ACMS) regression with N=27 [whitfill2025compute];
   - one demand-side RCT over models [merali2024scaling; merali2025scaling].
2. **The central confound is already visible in economics, but nobody has named it as such.** Bajari–Chernozhukov–Hortaçsu–Suzuki [bajari2019impact]:
   - Amazon's forecast error falls strongly with the length of a product's data history (T) when there are no time controls.
   - Adding a smooth trend or time fixed effects makes the effect of T essentially vanish, because T and calendar time are
     collinear, and calendar time proxies for "methodological improvements".
   - This is exactly the problem of separating "scaling" from "algorithmic progress" (TFP) in [ho2024algorithmic; erdil2022algorithmic].
   - Erdil–Besiroglu say it outright: the data-scaling parameters are "poorly identified" without priors, so they use MAP /
     ridge penalties.
3. **"Economies of scale" in the AI-IO literature is a cost-structure claim, not a technology claim.**
   - Korinek–Vipra, Varian and Gans mean: large fixed training cost, low marginal inference cost, and non-rival weights.
   - The training technology itself shows *sharply decreasing* returns in loss. Along the compute-optimal path, reducible
     loss scales as C^-0.15 to C^-0.18 (Chinchilla fits; see Sec. 3).
   - The IO model that combines both is **Sutton's (1991) endogenous sunk costs** [sutton1991sunk]: a convex cost of
     quality plus demand escalation gives a lower bound on concentration.
   - None of the AI-market papers I read cite Sutton (I grepped Korinek–Vipra, Gans, Varian and Demirer et al.).
     This is an open bridge.
4. **The Chinchilla loss is a CES / "weak-link" aggregator.**
   - Jones's JEP "weak links" production function [jones2026ai] is the CES with exponent −1 (harmonic mean, σ=1/2).
   - Chinchilla has the same form with exponents −α and −β.
   - [DERIVED HERE] Along the compute-optimal path the elasticity of substitution between parameters and data is
     σ* = 1/(1+(α+β)/2):
     - 0.76 with Hoffmann's α and β;
     - 0.74 with the Epoch replication values.
   - So parameters and data are gross complements, which confirms the Trammell–Korinek footnote quantitatively.
5. **Over-training gives a revealed-preference estimate of inference demand.** [DERIVED HERE]
   - Minimise 6ND + 2N·D_inf subject to L(N,D)=ℓ. Then D_inf/D = 3(ε_N/ε_D − 1), where ε_X is the loss-reduction
     elasticity of input X.
   - With the Besiroglu et al. (2024) replication parameters:
     - Chinchilla-70B has ε_N/ε_D ≈ 1.03, i.e. it sits on the expansion path (a nice sanity check);
     - Llama-3-8B has ε_N/ε_D ≈ 5.2, which implies ≈1.9×10^14 lifetime inference tokens.
   - The answer is very sensitive to β (see Sec. 3).
   - This is the "markup-like wedge" in the project hypotheses. It is really an *omitted-input shadow price*, not market power.
6. **Dual (price-based) productivity measures already exist and can be recast as dual TFP.** They need P ≈ MC, which the
   competitive open-model fringe plausibly delivers.
   - Price per unit of benchmark performance falls 5–10×/yr at the frontier [gundlach2025price].
   - After removing hardware deflation, the algorithmic part is ≈3×/yr.
   - The price of GPT-4-class capability fell ≈1000× in two years [demirer2025emerging].
   - Economics analogues: Hsieh (2002) and Jorgenson–Griliches.

---------------------------------------------------------------------------------------------------

## 1. How economists model the scaling-law technology — taxonomy

| Paper | Object modeled | Functional form | Parameters | Calibrated / estimated | Identifying assumption |
|---|---|---|---|---|---|
| Bergemann–Bonatti–Smolin EC'25 [bergemann2025economics] | Task value from input x, output y, fine-tuning z tokens | v(x,y,z)=x^α y^β (b+z)^γ, α+β+γ<1 (Cobb-Douglas, DRS); b = base-model quality | Example: α=β=γ=1/4, c=1/8, b=1 | Illustrative only | none (theory) |
| BBS 2026 "Menu Pricing" [bergemann2026menu] | Gain on task i | g(x_i,z)=Ψ(x_i)Φ(z); Ψ homogeneous of degree σ∈(0,1), e.g. CES Ψ=(Σα_j x_ij^ρ)^{σ/ρ}; Φ homogeneous deg σ̂ with σ+σ̂<1 | Cost function C_l(Q)=κ_l Q^{1/(σ+σ̂_l)} (Lemma 3); CES over models with elasticity set by σ (Lemma 2) | theory | homogeneity ⇒ scalar aggregate type θ |
| Korinek–Suh [korinek2024scenarios] | Share of tasks automatable | Pareto task complexity in "compute space": Φ(i)=1−e^{−λ log i}; automation index I=I_0 e^{gt} ⇒ unautomated share falls at rate λg (a power law in compute) | λg=0.01/yr (BAU); bounded AGI scenarios T=20 or 5 yrs | Calibrated (labor share 66%) | none |
| GATE [erdil2025gate] | Share automated f vs. effective compute | Piecewise log-linear in log compute between T/10^{ΔFLOP} and T; effective compute = $·H(t)·S(t); train–inference trade-off C_{T+ι}=C_T·ι^{1/m}, m≈1–2 | T, ΔFLOP sampled; H: FLOP/yr/$ ≈10^18 per $ at H100 retail | Calibrated/sampled | "effective compute is a sufficient statistic" |
| Trammell–Korinek [trammell2023economic] | AI production | Footnote 10: scaling laws "imply that compute and data are gross complements in AI training" | — | Assertion | — |
| Jones JEP 2026 [jones2026ai]; Jones–Tonetti 2026 [jones2026past] | Output from tasks | CES "weak links"; σ=1/2 ⇒ 1/Y = 1/X_easy + 1/X_hard; σ=0.2 in Jones–Tonetti | σ | Calibrated | — |
| Farboodi–Koh–Xia 2026 [farboodi2026datadriven] | Capital productivity in task i | ψ^K_i = f_i·(A_i(D))^η, η∈(0,1); justified by LOSS ≃ N^{-α}+D^{-β}, large-N regime, *assuming productivity = reciprocal of loss (to a power)* | η | Theory | explicitly flagged as "(substantial!)" assumption |
| Farboodi–Veldkamp [farboodi2021model] | Quality from data | E[A] ≈ g(Ω^{-1}+σ_a²)+…; Ω = forecast precision (stock of knowledge), which rises with data; σ_a² unlearnable | depreciation of data = 1−1/(ρ²+σ_ε² Ω) | Calibratable DSGE | Bayesian learning |
| Besiroglu–Emery-Xu–Thompson [besiroglu2024economic] | Idea production in computer vision | Cobb-Douglas: g̃_it = A_t^{θ−1} S_it^γ C_it^β ε_it; performance P=A/(1+A) (logistic) | β=0.111–0.253, γ=0.246–0.352 | **Estimated** (OLS/GLS, time FE) | exogeneity of inputs given time FE |
| Whitfill–Wu [whitfill2025compute] | AI R&D: research compute vs. cognitive labor | CES; FOC ⇒ ln(K_res/H)=const+σ ln(w/r)[+(1−σ)ln K_train] | σ=2.58 (0.34) or −0.10 (0.18); N=27 | **Estimated** | price-taking, static cost-min (ACMS) |
| Cunningham et al. 2026 [cunningham2026economics] | RSI feedback loops | Elasticity graph; effective AI labor = C^γ K; cost share = elasticity under CRS | γ≈0.15–0.3 from Villalobos–Atkinson under ECI normalisation | Calibrated from other studies | cost minimisation |
| Korinek–Vipra [korinek2025concentrating] | Market structure | Verbal: fixed pre-training cost + fine-tuning fixed cost per domain + low variable cost ⇒ economies of scale and scope | Gemini Ultra compute $130M; compute 4.1×/yr; spending 3.09×/yr | Descriptive | — |
| Zhang–Zhang 2026 [zhang2026economics] | "Digital intelligence capital" | Upstream production matched to Chinchilla; per a search snippet, RTS γ=1.3 and compute share 0.35 | — | Calibrated ABM | (low-signal source; RTS/share numbers not verified in the PDF) |

**Answer to "has anyone estimated it with econometric care?"** Partly, and only in reduced form. Section 2.D lists every
estimate. None handles simultaneity (TFP-correlated input choice) with proxies or instruments. None uses the within-lab
IsoFLOP "experiments" as identifying variation against cross-lab observational variation. None treats selection into
reporting (only 91–114 of 785 models report compute in Thompson et al.).

---------------------------------------------------------------------------------------------------

## 2. Paper-by-paper notes

### 2.A Pricing, demand, and market microstructure

**Bergemann, Bonatti, Smolin — EC'25 extended abstract / arXiv 2502.07736 v1 (Feb 2025)** [bergemann2025economics]
- Monopolist sells LLM versions. The user's type w_i is willingness to pay for accuracy on task i ∈ [0,1].
- Aggregate type θ = (∫ w_i^{1/(1−α−β)} di)^{1−α−β}. In the value-scale case, θ = w·s^{1−α−β}.
- Per-token costs are constant: c_x X + c_y Y + c_z Z.
- Optimal menu = two-part tariffs with p_j(θ) = m(θ)c_j, where the markup is m(θ) = θ/φ(θ) and φ is the Myerson virtual
  type. Higher-value users face lower markups (MHR).
- They explicitly say the concavity is "different from scaling laws for training LLMs". The object is inference-time
  diminishing returns.

**BBS "Menu Pricing of Large Language Models" — Cowles DP 2502, Mar 2026 (arXiv v2, 9 Mar 2026)** [bergemann2026menu]
- The gain function is multiplicatively separable: g(x_i,z) = Ψ(x_i)Φ(z) (their eq. 1).
- Ψ is homogeneous of degree σ ∈ (0,1). The canonical example is CES in token classes.
- Motivation: "consistent with empirically observed scaling laws". Cites Kaplan et al. (2020) and inference scaling
  laws [wu2024inference].
- Homogeneity is the key tractability assumption. It makes the optimal token mix scale-invariant, and σ = returns to scale.
- Duality result: the cost of delivering aggregate quality Q through model l is C_l(Q) = κ_l Q^{1/(σ+σ̂_l)} (Lemma 3).
- Multi-model payoff is U* = θ(Σ_l Q_l^{1/σ})^σ (Lemma 2). This is a CES over models, with the elasticity pinned down by σ.
- Leader–fringe competition with an open-source fringe gives three regions: fringe-only; leader at fringe-deterring
  quantity; unconstrained monopoly.
- They map the results to observed pricing: Anthropic, OpenAI, GitHub; committed-spend contracts; linear API pricing =
  inflated shadow costs under capacity constraints (Cor. 1).
- IO reading: a homothetic production function with a known cost function. Returns-to-scale σ is a parameter nobody has
  estimated. Scaling-law estimates on inference (e.g. accuracy vs. tokens) could calibrate it.

**Demirer, Fradkin, Tadelis, Peng — NBER w34608 (Dec 2025)** [demirer2025emerging]; JEP version [demirer2026emerging]
(JEP 40(3):23–46, Summer 2026, Demirer–Fradkin–Tadelis)
- Data: OpenRouter and Microsoft Azure API usage, plus Artificial Analysis benchmarks (13 benchmarks → "Intelligence Index").
- Market growth, Jan–Dec 2025:
  - models 253 → 651;
  - creators 43 → 85;
  - inference providers 27 → 90;
  - OpenRouter served >100T tokens in 2025.
- Price–intelligence hedonic regression (Table 1), with N=152 models:
  - log price per M prompt tokens on the Intelligence Index: 0.039 (0.009) in column 1, rising to 0.047 (0.017) with
    creator and age FE;
  - open-source dummy −2.46 to −1.91, i.e. ≈87% cheaper conditional on intelligence;
  - R² 0.10 → 0.615.
- Min price per intelligence tier: GPT-4-class price fell ≈1000× in ≈2 years.
- Demand (Table 2) is estimated at the provider–model–day level (N=32,539; open models only):
  - identification comes from within-model cross-provider price variation (entry/exit, price changes);
  - log(price) coefficient: −0.55 (0.09) with date FE only; −1.08 (0.19) adding model FE; −1.11 (0.22) with date×model and
    model×provider FE;
  - interpretation: provider-level elasticity just above 1, so aggregate elasticity is lower ⇒ no short-run Jevons paradox.
- Caveats:
  - OpenRouter routing algorithm;
  - throughput endogenous to demand (capacity);
  - price endogeneity is "not fully eliminated".
- IO use:
  - (i) data for dual TFP;
  - (ii) market-share data for Sutton bounds;
  - (iii) token volumes for the revealed-inference-demand check (Sec. 3).

**Fradkin — "Demand for LLMs: Descriptive Evidence on Substitution, Market Expansion, and Multi-Homing"** [fradkin2025demand]
Listed on Fradkin's site as a "resting paper", superseded by the Emerging Market paper. Year not verified.

**Athey & Scott Morton — "Artificial Intelligence, Competition, and Welfare", NBER w34444 (Nov 2025)** [athey2025artificial]
- GE model with AI as a priced imported input and discrete adoption.
- Upstream monopoly uses usage fees plus access fees; "double harm" to displaced workers.
- Does not model scaling laws. It treats fixed costs and market power upstream.

**Merali — Scaling Laws for Economic Productivity (arXiv 2409.02391, 2024)** [merali2024scaling]
- Preregistered experiment: 300 professional translators, 1,800 tasks. Random assignment to 1 of 13 LLMs or control.
- Regress outcomes on log10(model training compute). Per 10× compute:
  - task time −12.3% (coef −51.35 s, p=0.001);
  - grade +0.18 SD;
  - earnings per minute +16.1%.
- Gains are ≈4× larger for low-skill workers (−21.1% vs. −4.9% time per 10×).
- Macro step (Hulten/Acemoglu style): 19.9% exposed tasks × 61.2% productivity × 0.57 labor share ≈ +6.9% over a decade.
- Caveat: the randomisation is over *models*, which bundle compute with lab and vintage (algorithms). The "compute
  elasticity" is a reduced-form bundle. The 2024 paper does not separate them.

**Merali — consulting / data analyst / management tasks (arXiv 2512.21316, Dec 2025)** [merali2025scaling]
- >500 professionals, 13 LLMs.
- Each year of model progress −8% task time. Decomposition: 56% compute, 44% algorithmic progress.
- Non-agentic tasks gain more.
- Projects ≈+20% US productivity over a decade.

**Brynjolfsson, Li, Raymond — QJE 140(2):889–942 (2025)** [brynjolfsson2025generative]
- Staggered rollout of a genAI assistant to 5,172 support agents: +15% issues/hour; +34% for novices; ≈0 for the most skilled.
- Downstream "use" production function evidence, not training.

**Acemoglu — "The Simple Macroeconomics of AI", Economic Policy 40(121):13–58 (2025)** [acemoglu2025simple]
- Hulten's theorem [hulten1978growth]: aggregate TFP gain = share of tasks affected × average task-level cost saving.
- Result: ≤0.66% TFP over 10 years.
- This is the aggregation template Merali uses. It maps a scaling law in task productivity into macro TFP.

### 2.B Market structure / IO of foundation models

**Korinek & Vipra — "Concentrating Intelligence: Scaling and Market Structure in AI", Economic Policy 40(121):225–256 (2025); NBER w33139 (Nov 2024); INET WP 228 (Oct 2024)** [korinek2025concentrating]
- Three cost types:
  - (i) large fixed pre-training cost;
  - (ii) a fixed fine-tuning cost per application area;
  - (iii) low variable inference cost.
- Together these give "classical economies of scale"; general-purpose nature gives economies of scope.
- On inference cost: "$1 per 100,000 … tokens" for GPT-4o, estimated to be close to OpenAI's cost of inference.
- Inputs: compute, data, talent.
- Scaling laws (Kaplan; Hoffmann) "reduce the uncertainty that companies face when they make investment decisions".
- Growth figures:
  - frontier compute ×4.1/yr over 15 years (Epoch);
  - spending ×3.09/yr (Cottier 2023);
  - Gemini Ultra compute ≈$130M (Epoch 2024).
- Footnote: cost estimates count only the final run, not experiments (Heim 2021).
- "Number of players that a market of a given size can support is shrinking fast … natural monopoly", offset by market
  growth. This is Sutton-type logic, but uncited.
- Data section: 79% of US news sites block OpenAI crawlers (Fletcher 2024); synthetic data "substitutes training data with
  compute" (≈70% synthetic for GPT-5 per Thompson 2024).
- Earlier working-paper version: Vipra & Korinek (2023), arXiv 2311.01550 [vipra2023market].
- **Critique for our paper:** "economies of scale" here = declining average cost from fixed costs (non-rival weights). The
  *technological* returns to compute are strongly decreasing. The relevant object is the convex cost of quality, C(ℓ) ∝
  (ℓ−E)^{-1/0.15}, together with demand for quality. That is Sutton's endogenous-sunk-cost model.

**Gans — "Market Power in Artificial Intelligence", NBER w32270 (Mar 2024)** [gans2024market]
- Three markets: training data, input data, predictions.
- Formalises the data feedback loop: Y_t = A_t K_t^α, A_t = A(D_t), D_{t+1} = (1−δ)D_t + zY_t (after Farboodi–Veldkamp,
  Wilson 1975).
- Cites Bajari et al. (2019): diminishing returns to data size, with product breadth ≈ flat.
- Uses Hagiu–Wright across-user learning.
- A later Annual Review of Economics publication (2026) is reported by a secondary web source; not verified.

**Varian — "Artificial Intelligence, Economics, and Industrial Organization", in Agrawal–Gans–Goldfarb eds. (2019), pp. 399–419; NBER w24839 (2018)** [varian2019artificial]
- "Data typically exhibits decreasing returns to scale like any other factor of production."
- ImageNet error fell while the training set was fixed ⇒ algorithms and hardware, not data, drove progress.
- Three kinds of returns to scale:
  - supply-side (fixed cost);
  - demand-side (network effects);
  - learning by doing.
- Argues "data network effects" are really learning by doing. Rule of thumb: doubling output lowers unit cost 10–25%.
- **Bridge:** a power-law data scaling law is a learning curve (Wright 1936 [wright1936factors]; Arrow 1962 [arrow1962economic]).

**Hagiu & Wright — "Data-enabled learning, network effects, and competitive advantage", RAND 54(4):638–667 (2023)** [hagiu2023data]
- Dynamic competition with learning functions f_i(N_i).
- Competitive advantage depends on:
  - the shape of the learning curve;
  - asymmetries in learning functions (≈ TFP differences);
  - customer beliefs.
- Useful as the demand-side closure for "TFP dispersion across labs".

**Azoulay, Krieger, Nagaraj — "Old Moats for New Models", NBER w32474 (2024); Entrepreneurship and Innovation Policy and the Economy vol. 4 (2025)** [azoulay2025old]
- Appropriability plus complementary assets (compute, distribution) ⇒ concentration.
- A "rogue" tech giant may sustain open source.

**Goldfarb & Tucker — "Digital Economics", JEL 57(1):3–43 (2019)** [goldfarb2019digital]
- Five falling costs: search, replication, transportation, tracking, verification.
- Near-zero replication cost ⇒ inference as marginal cost; model weights non-rival.

**Agrawal, Gans, Goldfarb (eds.), The Economics of AI: An Agenda (U Chicago Press, 2019)** [agrawal2019economics]; **JEP 33(2):31–50 (2019)** [agrawal2019artificial]
- AI as a drop in the cost of prediction. The book contains the Aghion–Jones–Jones and Varian chapters.
- "Prediction Machines" (2018 book) not verified this session.

**Schaefer & Sapi — "Complementarities in learning from data: insights from general search", Information Economics and Policy 65 (2023)** [schaefer2023complementarities]
- Yahoo! search logs.
- Within-user data raises the efficiency of across-user learning. This is complementarity between data dimensions and
  gives locally increasing returns to data scale.
- Analogue of N–D complementarity in Chinchilla.

**Bajari, Chernozhukov, Hortaçsu, Suzuki — AEA P&P 109:33–37 (2019); NBER w24334** [bajari2019impact]
- Amazon retail forecasting.
- Theory: relative forecast error falls toward the irreducible level at rate 1/√N + 1/√T. This is **exactly the additive
  two-input Chinchilla form with exponents 1/2** (N = products in category, T = product age/history).
- Data: electronics, M=6,079 products, H=234 weeks (Dec 2012–Jun 2017).
- Outcome: Y_it = 1{big forecast error}, with the threshold set so P=30%.
- Motivated model (Table 2) regresses on Age>20, (Age>20)×Age^{-1/2}, (Age>20)×Age^{-1}, and analogous N terms, with
  product FE and three time-effect specifications:
  - without trend, the T terms are huge and significant (Age>20: −0.800 (0.046); inverse-root-age 9.230 (0.574));
  - with a trend or time FE they collapse (−0.029 (0.086); 0.395 (0.948)).
- Authors: "The ages T_{i,t} and trends t are correlated in this data". With time effects there are "essentially no
  improvements due to large T".
- N effect: flat or insignificant (contradicting theory).
- The time effects themselves improve steadily (forecasting-engine progress = TFP).
- **This is the cleanest economics demonstration that input-scale effects and technical progress are hard to separate when
  inputs trend with time.** Cite prominently.

### 2.C Growth / macro papers that encode an AI technology

**Aghion, B. Jones, C. Jones — "Artificial Intelligence and Economic Growth", in Agrawal et al. (2019), pp. 237–282; NBER w23928 (2017)** [aghion2019artificial]
- AI as automation; Baumol cost disease. Growth is constrained by what is "essential and yet hard to improve".
- The weak-link / CES-complements template.

**C. Jones — "AI and Our Economic Future", JEP 40(3):3–22 (Summer 2026)** [jones2026ai]
- Cites Epoch: effective training compute ≈×10/yr = ×4 hardware/spend × ×2.5 algorithms (Ho et al. 2024; Epoch 2026).
- METR 50% time horizon: 9 s (GPT-3, 2020) → 11 min (Jun 2024) → 12 h (mid-2026). Doubling 5–7 months, possibly
  4 months since reasoning models.
- Weak-link CES with σ=1/2: 1/Y = 1/X_easy + 1/X_hard. σ=0.2 in Jones–Tonetti (2026) [jones2026past].
- **Bridge [DERIVED HERE]:**
  - Chinchilla's reducible loss R(N,D) = A N^{−α} + B D^{−β} is a "sum of inverse powers".
  - With α=β, define y ≡ R^{−1/α}. Then y = (A N^{−α} + B D^{−α})^{−1/α}, a CES with ρ = −α and σ = 1/(1+α).
  - Jones's harmonic mean is the case α=1 (σ=1/2).
  - Both say: the scarce input is the bottleneck.

**Jones & Tonetti — "Nonrivalry and the Economics of Data", AER 110(9):2819–2858 (2020)** [jones2020nonrivalry]
- Data are nonrival ⇒ increasing returns from broad use; firms may hoard.
- Consumer property rights ≈ near-optimal.
- Analogy: model weights and public text corpora are nonrival. Data "reuse" across labs = nonrival input use.

**Farboodi & Veldkamp — "Long-Run Growth of Financial Data Technology", AER 110(8):2485–2523 (2020)** [farboodi2020longrun]; **"A Model of the Data Economy", NBER w28427 (2021, rev. 2022)** [farboodi2021model]
- Quality A_{i,t} = g((a_{i,t} − θ_t − ε_{a,i,t})²).
- Optimal action ⇒ E[A] ≈ g(Ω^{-1} + σ_a²) + g''(·)(Ω^{-1} + σ_a²) (their eq. 8).
- Ω = precision of the forecast ("stock of knowledge"). With iid signals, Ω grows linearly in data points.
- ⇒ **expected loss = irreducible σ_a² + reducible ∝ 1/data.** This is exactly Chinchilla's data term E + B D^{−β} with β=1
  (the parametric rate). Empirical LLM β≈0.28–0.37 means much slower learning than the parametric rate.
- Data depreciation: Ω_{t+1}^{no learning} = Ω_t / (ρ² + σ_ε² Ω_t), so the depreciation rate is 1 − 1/(ρ² + σ_ε² Ω_t).
  Analogue: data staleness, repetition, "effective data" (Muennighoff et al. 2023 [muennighoff2023scaling]).
- Data feedback loop ⇒ increasing returns when data are scarce; diminishing when abundant.

**Farboodi, Koh, Xia — "Data-Driven Automation", arXiv 2606.10127 (Jun 2026)** [farboodi2026datadriven]
- Assumption 1: ψ^K_i(A_i(D)) = f_i·(A_i(D))^η, η ∈ (0,1). Motivated by Hoffmann et al. and Besiroglu et al. (2024)
  replication (their fn. 19).
- Data-constrained regime: LOSS ≈ 1/D^β.
- Productivity = reciprocal of loss (to a power), flagged by the authors as a "(substantial!) assumption". **This is exactly
  the loss→output link-function problem our paper must address.**
- Results:
  - full automation iff σ ≤ 1/η;
  - labor's task share decays as a power law in time;
  - explosive growth but stagnant wages with capital accumulation.

**Korinek & Suh — "Scenarios for the Transition to AGI", NBER w32255 (Mar 2024)** [korinek2024scenarios]
- Tasks indexed by compute intensity i; automation index I grows at g (Moore's law; frontier compute doubling ~6 months per
  Sevilla et al.).
- Pareto complexity: Φ(i) = 1 − e^{−λ log i} ⇒ the unautomated share (1−Φ) = I^{−λ}, **a power law in compute**. With
  I = I_0 e^{gt} it decays at rate λg.
- Bounded (AGI) scenarios use a power-function distribution with T = 20 or 5 years.
- Calibration: λg = 0.01/yr (BAU); initial automated share 0.608; K_0 = 4.6 (to match 66% labor share).
- Compute as specific capital: high returns during scarcity, then normal returns.
- **Equivalence:** this is Houthakker's (1955) [houthakker1955pareto] / Jones's (2005) [jones2005shape] aggregation (Pareto
  micro heterogeneity ⇒ power-law / Cobb-Douglas aggregate). It is also Michaud et al.'s (2023) [michaud2023quantization]
  "quanta" derivation of neural scaling laws (Zipf-distributed skills ⇒ power-law loss).

**Trammell & Korinek — "Economic Growth under Transformative AI", NBER w31815 (Oct 2023; revised 2026)** [trammell2023economic]
- Full automation ⇒ growth explosion; wages depend on RTS, natural resources, direction of technical change.
- Footnote 10: scaling laws imply compute and data are gross complements (cites Villalobos 2023 "Scaling Laws Literature
  Review", an Epoch blog post).
- Cites Eth & Davidson (2025): raw compute requirements for AI capabilities roughly halve each year.
- Semiconductor R&D: λ/β ≈ 5.
- (An Annual Review of Economics article with the same title appears in search results; publication details not verified.)

**Erdil & Besiroglu — "Explosive growth from AI automation: A review of the arguments", arXiv 2309.11690 (2023, rev. 2024)** [erdil2023explosive]
- Accumulable AI labor restores increasing returns: RTS on accumulable inputs d ≥ ≈0.68 suffices.
- Uses scaling laws only qualitatively (performance driven by data and parameters, hence compute). Algorithmic efficiency
  doubling 9–16 months.

**GATE — Erdil, Potlogea, Besiroglu, Roldan, Ho, Sevilla, Barnett, Vrzla, Sandler, arXiv 2503.04941 (2025)** [erdil2025gate]
- Hardware and software efficiency both follow Jones (1995) laws of motion, with H·S = effective FLOP per $:
  - Ḣ/H = θ_H H^{−φ_H} (I_H^{RD})^{λ_H};
  - Ṡ/S = θ_S S^{−φ_S} (I_S^{RD})^{λ_S}.
- **Key asymmetry:** software improvements apply to the *whole* compute stock; hardware improvements only to new flows.
  = disembodied vs. capital-embodied technical change (Jones & Liu 2024 [jones2024framework]).
- Training compute is a fixed cost producing non-rival weights; inference is a variable cost.
- Training–inference trade-off: C_{T+ι} = C_T × ι^{1/m}, m ≈ 1–2, per Villalobos & Atkinson (2023) [villalobos2023trading].
- Automation share: piecewise log-linear in log effective compute, with parameters T (full-automation FLOP), ΔFLOP ("FLOP
  gap", from Davidson 2023) and f_init (calibrated to compute spending/GDP).
- Compute cost: H100 ≈ $30k ⇒ ≈10^18 FLOP/yr per $ at retail. Super-linear cost at macro scale from supply-chain
  adjustment costs.
- Explicitly acknowledges "effective compute" reduces algorithmic progress to one dimension.

**Cunningham et al. — "The Economics of Recursive Self-Improvement", arXiv 2609.15802 (Sep 2026)** [cunningham2026economics]
- Directed graph of production functions. Self-sustaining acceleration iff the product of elasticities around a loop is > 1.
- Uses "cost share = output elasticity under CRS and cost minimisation" to measure elasticities. This is the Solow /
  cost-share approach; it fails with markups or adjustment costs (the DLW critique).
- Parameter table:
  - γ (capabilities vs. inference quantity in effective AI labor C^γ K) ≈ 0.15–0.3 from Villalobos–Atkinson under ECI;
  - Whitfill–Wu σ "inconclusive";
  - asks labs to randomise R&D labor/compute.

**Whitfill & Wu — "Will Compute Bottlenecks Prevent an Intelligence Explosion?", arXiv 2507.23181 (2025)** [whitfill2025compute]
- Panel: OpenAI 2016–24, DeepMind 2014–24, Anthropic 2022–24, DeepSeek 2023–24 (27 firm-years).
- CES between research compute K_res and cognitive labor H. FOC regression: ln(K_res/H) on ln(w/r) (+ ln K_train) with firm FE.
- Measurement:
  - H from PitchBook headcount;
  - K_train from Epoch;
  - **K_res = K_train/3 imposed** (from one report of OpenAI's 2024 ratio);
  - w from financial statements plus imputation;
  - r = GPU rental FLOP/$.
- Estimates:
  - σ = 2.583 (SE 0.341; MC SE 0.657), R² 0.857;
  - with ln K_train ("frontier experiments") σ = −0.103 (0.176), R² 0.982.
- **IO critique:** the ACMS (1961) [arrow1961capital] estimator is biased under factor-augmenting technical change trends
  (León-Ledesma–McAdam–Willman 2010 [leonledesma2010identifying]; Antràs 2004 [antras2004aggregate]).
  - The compute price falls ~30–40%/yr, collinear with time, so any compute-augmenting progress loads onto σ.
  - Also: tiny N, imputed inputs, non-competitive GPU market.

**Besiroglu, Emery-Xu, Thompson — "Economic impacts of AI-augmented R&D", Research Policy 53(7):105037 (2024); arXiv 2212.08198** [besiroglu2024economic]
- Idea production for ImageNet classification (n=96) and COCO detection (n=40).
- Performance P = A/(1+A) (logistic). They show this implies power-law error in compute, consistent with Hoffmann et al.
- Growth proxy: g̃ = log(P_t/P_{t−1}) + log((1−P_{t−1})/(1−P_t)), relative to the median "baseline" model cited by the paper.
- Estimating eq. (13): log g̃_it = (θ−1) log A_t [time FE] + γ log S_it [human capital, from a citation-predictive model]
  + β log C_it [compute] + controls. OLS, or ML-GLS if Breusch–Pagan rejects.
- Table 4:
  - image classification β = 0.111 (0.021), γ = 0.246 (0.086); with trend β = 0.140 (0.029), γ = 0.350 (0.085);
  - detection β = 0.246 (0.106), γ = 0.352 (0.165).
- Table 5 (pooled, n=136): β = 0.145 (0.022) to 0.176 (0.017); γ = 0.278 to 0.298.
- Implied competitive capital cost share: 0.29–0.44 (vs. <20% in other STEM R&D).
- Kmenta/translog test (their eq. 18–19; Kmenta 1967 [kmenta1967estimation]): σ between compute and scientists tightly
  around 1 (0.9–1.1; slightly >1 for the larger samples).
- Endogeneity: only a *selection* discussion (compute-intensive / cutting-edge papers over-represented). No
  simultaneity/IV. Cost shares are inferred from elasticities assuming competitive FOC.

### 2.D Econometric estimates of performance-on-compute (the "reduced-form scaling laws" economists and Epoch produced)

**Thompson, Greenewald, Lee, Manso — "The Computational Limits of Deep Learning", arXiv 2007.05558 (v1 Jul 2020; v2 Jul 2022)** [thompson2020computational]
- 1,527 arXiv papers; 785 models. Only 91 report network operations; 114 report hardware burden.
- Compute measures:
  - network ops = Σ epochs × FLOPs/pass × passes/epoch;
  - hardware burden = Σ processors × rate × time.
- **Table 2, panel (a)** (verified in the PDF): ImageNet, N=80, network operations normalised to AlexNet 2012.

  | Column | Estimator; dependent variable | log10(NetOps) | Other terms | R² | Implied polynomial scaling (95% CI) |
  |---|---|---|---|---|---|
  | (1) | OLS; log10(top-1 error) | −0.082 (0.006) | intercept −0.629 (0.011) | 0.712 | 12.1 (10.6–14.1) |
  | (2) | OLS; + Year | −0.065 (0.005) | Year −0.022 (0.003); intercept −0.476 | 0.822 | 15.5 (13.3–18.5) |
  | (3) | OLS; top-1 error in levels | −0.033 (0.003) | — | 0.622 | — |
  | (4) | **Quantile regression, 10th percentile**; log10(top-1 error) | −0.084 (0.005) | — | pseudo-R² 0.557 | 11.9 (10.6–13.5) |

  - Column (4) is a **frontier / best-practice production function**; cf. stochastic frontier analysis [aigner1977formulation].
- **Panel (b)**, log10(performance metric) on log10(hardware burden):

  | Benchmark | Coefficient (SE) | N | R² | Implied scaling (95% CI) |
  |---|---|---|---|---|
  | ImageNet | 0.093 (0.006) | 104 | 0.702 | 10.8 (9.5–12.3) |
  | COCO box AP | 0.062 (0.012) | 20 | 0.809 | 16.0 (13.0–21.3) |
  | SQuAD1.1 EM | 0.096 (0.012) | 12 | 0.872 | 10.5 (8.3–14.3) |
  | CoNLL03 F1 | 0.027 (0.010) | 12 | 0.426 | 37.2 (20.4–200) |

- Exponential forms "also plausible" but have less explanatory power.
- Extrapolation: ImageNet 5% error needs ~10^26 FLOP (~$10^9); 1% needs ~10^33 FLOP (~$10^16).
- **[DERIVED HERE]** From spec 2, one year of algorithmic progress = 10^(0.022/0.065) ≈ 2.2× compute-equivalent, i.e.
  ≈3 years per 10× (their "3 years ≈ 10×").
- No IV / endogeneity treatment. Selection into reporting acknowledged.

**Erdil & Besiroglu — "Algorithmic progress in computer vision", arXiv 2212.05153 (v4 Aug 2023)** [erdil2022algorithmic]
- Model (eq. 1–3): σ^{−1}(P) = σ^{−1}(σ(C)·σ(D)) + ε, with σ the logistic function, and
  - C = α_1 + α_Year(Year−2012) + α_compute log(compute);
  - D = β_1 + β_Year(Year−2012) + β_data log(data).
- Approximates to 1−P ≈ Ã/C^{α_compute} + B̃/D^{β_data} = **Chinchilla with factor-augmenting technical change** (Ã, B̃
  depend on year).
- Data: extends Thompson et al.'s 124 ImageNet models; at most 3 top models per paper; excludes re-implementations and NAS.
- Estimation: MAP with Normal priors, equivalent to MLE + L2 penalty, "because … the parameters about data scaling end up
  being poorly identified in the absence of any regularization". **This is a weak-identification admission.**
- Table 1:
  - α_Year = 0.159 (0.045), α_compute = 0.154 (0.045);
  - β_Year = 0.019 (0.034) [n.s.], β_data = 0.063 (0.014).
  - The paper warns its SE-based CIs are unreliable (skewed / multimodal); bootstrapped 90% CIs are also reported.
- Compute-augmenting progress: effective compute doubling 8.95 months (95% CI 3.55–25.40).
- Year/compute ≈ +101%/yr (CI 25–215%); year/data ≈ +38%/yr (CI −47 to 134%).
- Compute-augmenting explains 4–11× more variation than data-augmenting (Shapley decomposition).
- IO: **factor-augmenting technical change**, identified only through the functional form plus a linear time trend.
  Diamond–McFadden–Rodriguez-type non-identification is avoided by assumption.

**Ho, Besiroglu, Erdil, Owen, Rahman, Guo, Atkinson, Thompson, Sevilla — "Algorithmic progress in language models", arXiv 2403.05812 (NeurIPS 2024)** [ho2024algorithmic]
- >200 LM evaluations on WikiText and PTB, 2012–2023.
- "Augmented" Chinchilla with time-varying effective N and D.
- Compute to reach a fixed performance halves ≈ every 8 months (95% CI ~5–14).
- Compute scaling contributed more than algorithms over the period.
- Same identification structure as above: year and log-compute are highly collinear across vintages.

**Hernandez & Brown — "Measuring the Algorithmic Efficiency of Neural Networks", arXiv 2005.04305 (2020)** [hernandez2020measuring]
- FLOPs to reach AlexNet-level ImageNet performance fell 44× from 2012 to 2019 ⇒ doubling every 16 months (Moore's law:
  11× over the same period).
- Threshold-dependent (Erdil–Besiroglu argue it biases progress downward).

**Gundlach, Fogelson, Lynch, Trisovic, Rosenfeld, Sandhu, Thompson — "On the Origin of Algorithmic Progress in AI", arXiv 2511.21622 (2025)** [gundlach2025origin]
- Claimed 22,000× efficiency gain from 2012–2023. Ablations explain <10×; the literature adds <10× more (<100× total).
- Scale-dependent gains: LSTM→Transformer changes the *exponent* of the compute-optimal scaling law. Together these account
  for 6,930×.
- Efficiency measures are "strongly reference-dependent".
- **IO:**
  - non-neutral technical change (changes the output elasticity, not just the level). This makes "effective compute
    multipliers" an **index-number problem**: Laspeyres vs. Paasche base scale; cf. Diewert (1976)
    [diewert1976exact] superlative indexes.
  - It undercuts the Hicks-neutral TFP assumption used in Ho et al. and in GATE's single-index "effective compute".

**Gundlach, Lynch, Mertens, Thompson — "The Price of Progress: Price Performance and the Future of AI", arXiv 2511.23455 (v1 Nov 2025; v2 Mar 2026)** [gundlach2025price]
- Price per unit of benchmark performance falls ~5–10×/yr at the frontier (knowledge, reasoning, math, SWE).
- Restricting to open models (to control for competition / markups) and deflating by hardware price declines gives
  algorithmic efficiency ≈3×/yr.
- Frontier models' running cost has *risen* 3–18×/yr.
- (Mertens is a productivity/markup IO economist.)
- **IO:** a dual (price-based) TFP measure [hsieh2002explains; jorgenson1967explanation]. It needs P = MC, or a
  constant markup.

**Ho, Denain, Atanasov, Albanie, Shah — "A Rosetta Stone for AI Benchmarks", arXiv 2512.00193 (Nov 2025)** [ho2025rosetta]
- Basis of the Epoch Capabilities Index (ECI).
- IRT-style latent model: each model has a scalar capability; each benchmark has difficulty and slope; a sigmoid link.
- ≈40 benchmarks, ≈200 models.
- Frontier improves ≈0.6 capability units/yr; ≈6× less training compute per year for equal capability.
- **IO:** latent output index (output measurement); a bounded link (TFPQ vs. TFPR analogue).

**Ruan, Maddison, Hashimoto — "Observational Scaling Laws …", arXiv 2405.10938 (NeurIPS 2024)** [ruan2024observational]
- Model:
  - σ^{−1}(E_m) ≈ βᵀS_m + α;
  - **S_m ≈ θ_f log(C_m) + ν_f** (family-specific compute efficiency);
  - B_{i,m} ≈ γ_iᵀ S_m, with the capabilities S extracted by PCA of benchmarks.
- "Model families only vary in their efficiency in converting compute into capabilities."
- **IO:** θ_f, ν_f = firm-specific output elasticity and TFP (a random-coefficients production function). This is the
  natural object for "TFP dispersion across labs". It does not address the endogeneity of C_m to ν_f.

**Owen — "How predictable is language model benchmark performance?", arXiv 2401.04757 (2024)** [owen2024predictable]
- 11 architectures, 5 OOM of compute.
- Extrapolating one OOM: aggregate BBH absolute error ≈6pp; individual tasks ≈18pp.

**Besiroglu, Erdil, Barnett, You — "Chinchilla Scaling: A replication attempt", arXiv 2404.10102 (2024)** [besiroglu2024chinchilla]
- Reconstructs 240 points from Hoffmann et al. Fig. 4. Huber loss on the log-sum-exp form (their eq. 2).
- Estimates: L = 1.8172 + 482.01/N^0.3478 + 2085.43/D^0.3658.
  - SEs: A 124.58; B 1293.23; E 0.03; α 0.02; β 0.02.
  - a = β/(α+β) = 0.5126 (0.02).
- Hoffmann's precise values (from the TeX source): E 1.6934, A 406.4, B 410.7, α 0.3392, β 0.2849 (a = 0.454).
- χ² test of equality: p < 10^−51.
- Hoffmann's reported CI for a (0.454–0.455) would need ~600,000 runs. The cause was averaging rather than summing the Huber
  loss, which made L-BFGS stop early (Borgeaud 2024).
- Hoffmann's Approach 3 implies a scaling policy inconsistent with Chinchilla's ~20 tokens/param. The replication restores
  consistency.

**Czech, Xu, Elmatad, Wang, Held — "Problems with Chinchilla Approach 2: Systematic Biases in IsoFLOP Parabola Fits", arXiv 2603.22339 (Mar 2026)** [czech2026problems]
- The parabolic IsoFLOP approximation is biased even on noise-free data. Three sources:
  - grid width (Taylor error);
  - uncentered sampling;
  - α ≠ β asymmetry.
- On Llama-3 data: 6.5% parameter under-allocation of a 3.8×10^25 FLOP budget ≈ $1.4M wasted (90% CI $412K–$2.9M).
- Recommends Approach 3 via variable projection (VarPro).
- **IO:** a functional-form approximation bias in cost-function estimation, the analogue of translog-approximation bias.

**Bajari et al. (2019)** — see 2.B; the data-scale vs. time confound.

**Merali (2024, 2025); Brynjolfsson–Li–Raymond (2025)** — see 2.A; experimental identification on the use side.

### 2.E Inputs, costs, prices, diffusion

**Cottier, Rahman, Fattorini, Maslej, Besiroglu, Owen — "The rising costs of training frontier AI models", arXiv 2405.21015 (2024)** [cottier2024rising]
- Amortized cost of the most compute-intensive runs ×2.4/yr since 2016. The current abstract gives a 90% CI of 2.0–2.9;
  a search snippet gave "95% CI 2.0–3.1", probably an earlier version (flag).
- Components: accelerator chips and R&D staff dominate; server components 15–22%; interconnect 9–13%; energy 2–6%.
- Projects >$1B runs by 2027.
- **Input cost shares** for a Cobb-Douglas/CES cost function; the hardware / labor / energy split.

**Hobbhahn & Besiroglu — "Trends in GPU price-performance", Epoch (Jun 2022)** [hobbhahn2022trends]
- 470 GPUs, 2006–2021. FLOP/s per $ doubling time:
  - all GPUs 2.46 yrs (95% CI 2.24–2.72);
  - ML GPUs 2.07 (1.54–3.13);
  - top GPUs 2.95 (2.54–3.52).
- Input price deflator for compute.

**Sevilla, Heim, Ho, Besiroglu, Hobbhahn, Villalobos — "Compute Trends Across Three Eras of ML", arXiv 2202.05924 (2022)** [sevilla2022compute]
- 123 milestone systems.
- Pre-2010 doubling ≈20 months; deep-learning era ≈6 months; a large-scale era from late 2015 at 10–100× higher levels.

**Villalobos, Ho, Sevilla, Besiroglu, Heim, Hobbhahn — "Will we run out of data? …", arXiv 2211.04325 (2022)** [villalobos2022run]
- Human-generated public text as an exhaustible stock. IO/resource analogue: Hotelling-type scarcity of the data input
  (my framing, not theirs).

**Villalobos & Atkinson — "Trading off compute in training and inference", Epoch (Jul 28, 2023)** [villalobos2023trading]
- Isoquant slopes by technique:

  | Technique | Trade-off |
  |---|---|
  | Chinchilla overtraining | ~0.7 OOM inference saved per 1.2 OOM extra training |
  | Pruning | ~1:1 OOM |
  | MCTS | 1.6 OOM inference per 1 OOM training at low performance, reversing near perfect play |
  | Repeated sampling with cheap verification | 3–4 OOM inference per 2 OOM training; 6 per 4 near perfect accuracy |
  | Repeated sampling with limited verification | ~1.45 OOM inference per OOM training |

- Combining techniques yields 2–3 OOM total.
- **IO: a direct isoquant between training and inference compute.** Log-linear slopes = Cobb-Douglas elasticity ratios.

**Erdil — "Inference economics of language models", arXiv 2506.04645 (2025)** [erdil2025inference]
- Pareto frontier of serial speed vs. cost per token, from arithmetic, memory bandwidth, network and latency constraints.
- Latency ∝ √(model size) and ∝ (memory bandwidth)^{−1/3}.
- **IO:** inference cost function (a multiproduct: speed and quantity), the other half of lifetime cost.

**Sardana, Portes, Doubov, Frankle — "Beyond Chinchilla-Optimal: Accounting for Inference in LM Scaling Laws", arXiv 2401.00448 (ICML 2024)** [sardana2024beyond]
- Minimise training + inference FLOPs for a given quality and inference demand.
- With ~1B requests, models should be smaller and trained longer than Chinchilla-optimal.
- 47 models trained; quality keeps improving up to 10,000 tokens/param.
- GitHub: nikhilsardana/beyond-chinchilla.
- The engineering version of our revealed-inference-demand formula (Sec. 3).

**DeepSeek-AI — DeepSeek-V3 Technical Report, arXiv 2412.19437 (Dec 2024)** [deepseekai2024deepseek]
- MoE: 671B total, 37B active per token; 14.8T tokens.
- 2.788M H800 GPU-hours (pre-training 2.664M; context extension 119K; post-training 5K) ⇒ $5.576M at an assumed
  $2/GPU-hour.
- The cost "exclude[s] the costs associated with prior research and ablation experiments".
- Trained on 2,048 H800s (the export-control-compliant variant).
- **IO:**
  - MoE total vs. active params = capital stock vs. capital services;
  - reported cost = final-run marginal cost, not full cost (a measurement issue, cf. Cottier et al.; Heim 2021 via Korinek–Vipra);
  - [DERIVED HERE] 6 × 37e9 × 14.8e12 ≈ 3.3×10^24 active-parameter FLOP.

**Amodei — "On DeepSeek and Export Controls" (blog, Jan 2025)** [amodei2025deepseek]
- Industry, interested-party source.
- The cost curve for fixed capability shifts ~4×/yr (vs. a ~1.68×/yr earlier estimate).
- DeepSeek-V3 is roughly on-trend: comparable to US models 7–10 months older at lower cost.
- "When it shifts, we simply traverse it faster": labs spend more when TFP rises. This is Jevons/rebound on the *training*
  side; contrast Demirer et al.'s short-run token-demand elasticity ≈ 1.
- The claim that export controls are binding is untested econometrically. **No econometric paper on export controls and
  lab TFP found** (my web-search budget ran out before exhaustive coverage). Akimitsu (2026, arXiv 2607.29572,
  "AI: Supply-Chain Chokepoints and the Reach of Industrial Policy") exists but was not read.

**Cottier, You, Martemianova, Owen — "How far behind are open models?", Epoch (Nov 4, 2024)** [cottier2024far]
- Open models lag closed models by 5–22 months on benchmarks (GPQA smallest ≈5; MMLU largest ≈25).
- Training-compute lag ≈15 months (90% CI 6–22). Llama-3.1-405B ≈16 months behind GPT-4.
- Epoch data insight (2026, not a bib entry): the ECI gap is ≈4 months and 8 ECI points since Jan 2026.
- **IO:** technology diffusion / imitation lag; open models as the competitive fringe (cf. BBS leader–fringe).

**Pilz, Heim, Brown — "Increased Compute Efficiency and the Diffusion of AI Capabilities", arXiv 2311.15377 (AAAI 2025)** [pilz2023increased]
- Falling compute cost gives:
  - an access effect (more actors reach a fixed capability);
  - a performance effect (leaders move up).
- ImageNet 93% cost: >$1,000 (2017) → $5 (2021).

---------------------------------------------------------------------------------------------------

## 3. Derived results for the paper [DERIVED HERE — verify in code/ appendix]

Let R(N,D) = A N^{−α} + B D^{−β} (reducible loss), a ≡ αA N^{−α}, b ≡ βB D^{−β}. These are the loss-reduction
"output elasticities": ε_N = a/R and ε_D = b/R.

1. **Local elasticity of substitution along an iso-loss curve.**
   - σ(N,D) = 1 / (1 + α·s_D + β·s_N), with s_N = a/(a+b) and s_D = b/(a+b).
   - Derivation: dlnD/dlnN|_R = −a/b and d ln MRTS = −[(α+1)b + (β+1)a]/b · dlnN.
   - If α = β, then σ = 1/(1+α): exact CES.
2. **Compute-optimal expansion path.** Minimise C = 6ND s.t. R = R̄.
   - Cost is multiplicative (log C = log 6 + log N + log D), so both cost elasticities equal 1. The FOC gives **a = b**
     (equal output elasticities).
   - ⇒ s_N = s_D = 1/2 ⇒ **σ* = 1/(1 + (α+β)/2)**.
   - N* ∝ C^{β/(α+β)}; D* ∝ C^{α/(α+β)}; R* ∝ C^{−αβ/(α+β)}.

   | Parameter set | σ* | N* exponent | R* exponent | Implied D/N at 5.76e23 FLOP |
   |---|---|---|---|---|
   | Hoffmann, precise | 0.762 | 0.456 | −0.155 | 59 (inconsistent with Chinchilla's 20) |
   | Besiroglu et al. 2024 | 0.737 | 0.513 | −0.178 | 18.4 (consistent) |

   - Along the path, input *levels* are deterministic functions of C. This is the functional-dependence problem: α and β
     are separately identified only through curvature off the path (IsoFLOP sweeps).
3. **Revealed lifetime inference demand ("wedge").**
   - Minimise 6ND + 2N·D_inf s.t. R = R̄ (2N FLOP/token at inference). FOC: a/b = 1 + D_inf/(3D) ⇒ **D_inf = 3D(a/b − 1)**.
   - With the replication parameters:

     | Model | a/b | Implied D_inf |
     |---|---|---|
     | Chinchilla-70B (1.4T tokens) | 1.03 | ≈1.3e11 (≈ on path) |
     | Llama-2-7B (2T) | 2.62 | ≈9.7e12 |
     | Llama-3-8B (15T) | 5.22 | ≈1.9e14 |
     | Llama-3-70B (15T) | 2.45 | ≈6.5e13 |

   - With Hoffmann's precise parameters: Llama-3-8B ≈8.7e13; Chinchilla-70B gives a *negative* D_inf (a/b = 0.71). That
     symptom is the known Approach-3 inconsistency.
   - Caveats:
     - cross-lab transfer of (A, B, α, β) is invalid (data mix, tokenizer, architecture = heterogeneous technology);
     - in practice this should be estimated with lab-specific scaling laws or treated as a bound.
4. **Learning-curve reading of exponents.** Reduction in reducible loss per doubling of the input, with the progress ratio
   2^{−exponent} in parentheses:

   | Exponent | Value | Reduction per doubling (progress ratio) |
   |---|---|---|
   | Chinchilla β | 0.28 | 17.6% (0.824) |
   | Chinchilla α | 0.34 | 21.0% |
   | Kaplan α_D | 0.095 | 6.4% |
   | Kaplan α_N | 0.076 | 5.1% |
   | Kaplan α_C | 0.050 | 3.4% (0.966) |
   | Chinchilla compute path | 0.155 | 10.1% (0.90) |

   Varian's rule of thumb for learning-by-doing is 10–25% unit-cost decline per doubling. Chinchilla's per-input exponents
   sit in the classic learning-curve range. (Loss is not cost; the analogy is in functional form.)
5. **Thompson et al. time-trend equivalence:** 1 year ≈ 2.2× compute (≈3 years per 10×).
   **Erdil–Besiroglu point doubling:** log 2 × 0.154/0.159 years ≈ 8.1 months (their bootstrap mean is 8.95).

---------------------------------------------------------------------------------------------------

## 4. ML ↔ IO dictionary contributed by this strand

| ML object | IO / econ object | Strength | Note |
|---|---|---|---|
| Chinchilla R=A N^−α+B D^−β, α=β | CES aggregator, ρ=−α, σ=1/(1+α) | exact | (monotone transform of) R^{−1/α} is CES |
| Chinchilla with α≠β | non-CES; local σ=1/(1+αs_D+βs_N); σ*=1/(1+(α+β)/2) on path | close | 0.74–0.76 ⇒ gross complements (confirms Trammell–Korinek fn.10) |
| Chinchilla "sum of inverse powers" | Jones weak-link / harmonic-mean CES; O-ring limit | close | Jones σ=1/2 ⇔ exponent −1 |
| BBS homogeneous gain Ψ, cost C(Q)=κQ^{1/(σ+σ̂)} | homothetic production + Shephard duality | exact | cost function of quality is a power function |
| Compute-optimal frontier L*(C); IsoFLOP | cost function (Nerlove); isocost in logs (C=6ND is Cobb-Douglas in N, D) | close | cost multiplicative ⇒ FOC equalises output elasticities |
| Effective-compute multiplier / algorithmic progress | Hicks-neutral TFP growth | close | fails if progress changes exponents (Gundlach 2025) |
| Scale-dependent algorithmic progress | non-neutral technical change; index-number (base-dependence) problem | close | reference dependence = Laspeyres/Paasche issue |
| Compute- vs data-augmenting progress (Erdil–Besiroglu) | factor-augmenting technical change | close | identified only via functional form + trend (DMR problem) |
| Hardware progress vs software progress (GATE) | capital-embodied vs disembodied technical change | exact | software applies to whole stock; hardware only new vintages |
| Price of fixed capability falling 5–10×/yr | dual (price-based) TFP | close | requires P≈MC; open-model fringe ≈ competitive |
| Benchmarks via logit / IRT (ECI) | bounded link to latent output; TFPQ vs TFPR measurement | close | Besiroglu et al. P=A/(1+A) |
| Pareto task-compute (Korinek–Suh) / Zipf quanta (Michaud) | Houthakker (1955) / Jones (2005) aggregation to Cobb-Douglas | exact | same math |
| Farboodi–Veldkamp loss Ω^{−1}+σ_a² | Chinchilla data term E+BD^{−β}, β=1 | close | LLM β≈0.3 ≪ parametric 1 |
| Bajari et al. 1/√N+1/√T | additive two-input scaling law, exponents 1/2 | close | same algebra; their time-FE confound is ours |
| Data scaling exponent | learning curve / Wright's law | close | Chinchilla β ⇒ 82% progress ratio |
| Training vs inference compute trade-off | isoquant between two inputs | close | slopes ≈0.6–2 OOM/OOM by technique |
| Fixed training cost + marginal inference cost | economies of scale via fixed cost / nonrival weights | close | cost-structure scale economies |
| Scaling-law convex cost of capability + demand escalation | Sutton endogenous sunk costs (quality) | close | unused so far; gives concentration lower bounds |
| "AI has economies of scale" (Korinek–Vipra) read as technological IRS | returns to scale of the training technology | breaks-down | technology has strongly decreasing returns in loss |
| Over-training for inference | shadow price of an omitted input; DLW-style wedge | loose | not market power; D_inf = 3D(a/b−1) |
| Thompson et al. 10th-percentile quantile regression | frontier production function / stochastic frontier | close | best-practice frontier |
| Cost share = elasticity (Besiroglu et al.; Cunningham et al.) | Solow / cost-share approach | close | fails with markups and adjustment costs |
| ln(K/H) on ln(w/r) (Whitfill–Wu) | ACMS (1961) relative factor demand | exact | biased under trending factor-augmenting TC |
| Family-specific θ_f, ν_f (Ruan et al.) | firm-specific production function (random coefficients, firm FE TFP) | close | endogeneity of C to ν_f unaddressed |
| Merali RCT over 13 models | experimental production-function / dose-response | close | "compute" bundles vintage and lab |
| Data repetition / staleness | data depreciation (Farboodi–Veldkamp formula) | loose | different mechanism |
| Open–closed lag (Epoch) | technology diffusion / imitation lag; competitive fringe | loose | — |
| MoE total vs active params | capital stock vs capital services | close | DeepSeek-V3: 671B vs 37B |
| Token demand elasticity (≈−1.1 provider level) | Jevons / rebound | close | aggregate < 1 in short run |

---------------------------------------------------------------------------------------------------

## 5. Disagreements and pitfalls to flag in the paper

- **Chinchilla parameters.**
  - Hoffmann (β = 0.285, B = 410.7) vs. the replication (β = 0.366, B = 2085): statistically different (p < 10^−51).
  - Hoffmann's Approach 3 implies ~59–93 tokens/param at Chinchilla compute, vs. the ~20 its own authors recommend.
  - Any economic quantity (σ*, D_inf, returns) is sensitive to this.
- **Algorithmic progress rates disagree widely:**

  | Estimate | Source |
  |---|---|
  | 16-month halving (CV) | Hernandez–Brown |
  | ~9 months (CV) | Erdil–Besiroglu |
  | ~8 months (LM) | Ho et al. |
  | ~6×/yr compute-equivalent (ECI) | Ho et al. 2025 |
  | ~3×/yr (price-based, open models) | Gundlach et al. |
  | ~4×/yr (industry guess) | Amodei |
  | "halve each year" | Eth–Davidson via Trammell–Korinek |
  | <100× of the claimed 22,000× verifiable at small scale | Gundlach et al. 2025 |

  - Differences come from reference scale, threshold choice, primal vs. dual measurement, and neutral vs. non-neutral
    assumptions.
- **σ estimates conflict:**
  - σ ≈ 1 (compute vs. human capital in CV idea production; translog test) [besiroglu2024economic];
  - σ = 2.58 vs. −0.10 depending on specification [whitfill2025compute];
  - σ* ≈ 0.74–0.76 between parameters and data (derived).
  - These are different input pairs and different stages (R&D vs. training).
- **Demand elasticity vs. "Jevons".**
  - Demirer et al.: short-run provider-level −1.1 ⇒ aggregate < 1 ⇒ no short-run Jevons.
  - Amodei: labs spend *more* when the curve shifts (training side).
  - These are not contradictory: different margins (inference demand vs. capability investment).
- **Cottier et al. CI** differs across versions (90% CI 2.0–2.9 in the current abstract vs. a 95% CI 2.0–3.1 snippet).
- **Measurement of inputs.**
  - Final-run cost excludes R&D/ablation (DeepSeek explicitly).
  - Compute is often *estimated* from architecture × tokens (6ND) rather than reported (Epoch). That builds functional
    dependence into "measured" compute: C is literally constructed from N and D. **This is an important econometric
    pitfall: regressing loss on N, D and C simultaneously is collinear by construction.**
- **Selection.**
  - Thompson et al.: only 91–114 of 785 models report compute.
  - Erdil–Besiroglu keep at most 3 top models per paper.
  - Besiroglu et al. note over-representation of cutting-edge papers.
  - Leaderboards are selected on the outcome (truncation on performance ⇒ attenuation or bias of slopes; cf. frontier vs. average).
- **Bundled treatments.** Merali's RCT identifies the effect of *model* assignment. Compute is correlated with vintage and lab.
- **Bounded outputs.** Accuracy near 0/1 saturates. The logit link is assumed, not tested (Besiroglu et al.;
  Erdil–Besiroglu; Ruan).
- **Cost-share = elasticity** requires cost minimisation with no markups or adjustment costs. The GPU market is not
  competitive (Whitfill–Wu fn. 11), and capacity constraints bind (BBS Cor. 1).

---------------------------------------------------------------------------------------------------

## 6. Contribution ideas from this strand

1. **Cross-lab production-function estimation with IO controls.**
   - Epoch model database (N, D, C, lab, date) + ECI/benchmarks.
   - Estimate a Chinchilla-CES with lab-year TFP, using an ACF/GNR-style proxy. Candidate proxies: API price, GPU purchases,
     hiring.
   - Compare with within-lab IsoFLOP (experimental) estimates to *measure* simultaneity bias.
2. **Revealed inference demand (DLW-style wedge).**
   - D_inf = 3D(ε_N/ε_D − 1) for open-weight releases.
   - Validate against OpenRouter token volumes (Demirer et al.) or Sardana-type engineering targets.
   - Interpret as a shadow price of the inference input.
3. **Sutton bounds for frontier AI.**
   - Use the scaling-law convex cost of capability, C(ℓ) ∝ (ℓ−E)^{−(α+β)/(αβ)}, plus Merali/Demirer willingness-to-pay for
     intelligence.
   - Derive a lower bound on concentration.
   - Test against OpenRouter/Azure market shares and model turnover.
4. **Primal vs. dual TFP of AI.**
   - Primal: compute-equivalent progress (Ho et al.; ECI 6×/yr).
   - Dual: price per ECI point, open vs. closed (Gundlach 3×/yr; Demirer 1000×).
   - The gap identifies markup changes and hardware pass-through.
5. **Index-number treatment of algorithmic progress.** With scale-dependent exponents (Gundlach 2025), report Törnqvist /
   Fisher-type superlative indexes of algorithmic progress and show the reference-dependence bounds.
6. **Weak-identification diagnostics.** Formalise year × log-compute collinearity in algorithmic-progress regressions:
   - Erdil–Besiroglu needed priors;
   - Bajari et al. show T-effects vanish with time FE.
   Provide partial-identification bounds for the scaling vs. progress decomposition.
7. **Export controls as an input-supply shock.**
   - Chinese labs post-Oct-2022 (H800/H20) vs. US labs.
   - Test induced (directed) technical change: TFP and factor bias (MoE, low-precision), DeepSeek-V3 as a case.
8. **The link function from loss to economic value.**
   - Combine Merali's per-10×-compute productivity (−12.3% time) with a scaling law in loss.
   - Estimate the "value elasticity of loss" needed by Farboodi–Koh–Xia and BBS (the σ in homogeneous gains).
9. **Train–inference isoquant estimation.** Use Villalobos–Atkinson technique curves and test-time scaling data to estimate
   the elasticity of substitution between training and inference compute. This parameter is used, uncalibrated or loosely
   calibrated, in GATE (m ≈ 1–2) and Cunningham et al. (γ ≈ 0.15–0.3).

---------------------------------------------------------------------------------------------------

## 7. Data leads (econ/market side)

- **Epoch AI "Data on AI models"** (notable / large-scale / frontier): parameters, tokens, compute, organisation, date,
  cost, accessibility. https://epoch.ai/data/ai-models (cited by Jones JEP 2026).
- **Epoch Capabilities Index / benchmarking hub:** IRT capability per model [ho2025rosetta]. https://epoch.ai/blog/a-rosetta-stone-for-ai-benchmarks
- **Epoch GPU price-performance** (470 GPUs, 2006–2021): https://epoch.ai/publications/trends-in-gpu-price-performance
- **Epoch Compute-Trends repo** (123 systems): https://github.com/epoch-research/Compute-Trends
- **Epoch open vs. closed report:** https://epoch.ai/publications/open-models-report
- **Thompson et al. Computational Limits dataset** (via MIT FutureTech): https://futuretech.mit.edu/publication/the-computational-limits-of-deep-learning
- **OpenRouter token volumes/prices by model and provider** (used by Demirer et al.): https://openrouter.ai (rankings pages).
  Terms of use not checked.
- **Artificial Analysis Intelligence Index, prices, throughput, latency** (used by Demirer et al.): https://artificialanalysis.ai
- **Bajari et al. replication package** (openICPSR 114501): https://www.openicpsr.org/openicpsr/project/114501/version/V1/view
- **Farboodi–Veldkamp AER replication** (openICPSR 114984): https://www.openicpsr.org/openicpsr/project/114984/version/V2/view
- **Sardana et al. code:** https://github.com/nikhilsardana/beyond-chinchilla
- **METR time horizons:** https://metr.org/time-horizons/ (Kwa et al. 2025).
- **DeepSeek-V3 report** (GPU-hours, active/total params, tokens): https://arxiv.org/abs/2412.19437
- **Cottier et al. cost model:** https://arxiv.org/abs/2405.21015

---------------------------------------------------------------------------------------------------

## 7b. Further verified references in the bib (one line each)

- [kaplan2020scaling; hoffmann2022training] Core scaling-law papers (primary coverage in the ML strand). Hoffmann's precise
  Approach-3 values come via the TeX source, as reported in [besiroglu2024chinchilla].
- [snell2024scaling] Test-time compute vs. parameters: an isoquant between inference and training inputs.
  [jones2021scaling] Train-time vs. test-time compute trade-off in board games.
- [kwa2025measuring] METR 50%-success time horizon: an output measure used by Jones (2026) and Farboodi–Koh–Xia (2026).
- [erdil2024estimating] Epoch survey of idea-production-function estimators: naive, OLS, MLE, Bayesian. It is the closest
  "econometrics for AI progress" methods paper. It does not cover proxy-variable or FOC methods from IO.
- [bloom2020ideas] Idea production function template used by GATE, Besiroglu et al. and Cunningham et al.
- [jones2011intermediate] Weak links / intermediate goods; the CES-complements template behind Jones (2026).
- [nordhaus2021approaching] Singularity tests via substitution between information and conventional inputs. A
  methodological precedent for estimating σ in the AI context.
- [nordhaus2007two] Long-run price of computation: the compute input-price deflator.
- [brynjolfsson2021productivity] Intangibles / J-curve. Training compute and R&D as unmeasured intangible investment.
- [brynjolfsson2025research; jones2025artificial] NBER agenda papers on transformative AI and AI in R&D (context only).
- [akimitsu2026artificial] Export controls / chokepoints (metadata only; not read).

---------------------------------------------------------------------------------------------------

## 8. Verification log / caveats on sources

- **Web-search budget.** The session's WebSearch budget (shared across parallel agents) ran out mid-strand. Items verified
  afterwards used arXiv abstract pages, NBER/AEA pages, Crossref and OpenLibrary.
- **Verified only via reference lists or secondary pages:**
  - Jones & Tonetti (2026) "Past Automation and Future A.I." (Jones JEP reference list);
  - Aghion–Jones–Jones chapter pages (NBER book page);
  - Gans Annual Review 2026 (secondary wiki, unverified);
  - the Trammell–Korinek Annual Review article (title seen in a search result, not opened).
- **BBS versions.** The v1 (EC'25) functional form is Cobb-Douglas v = x^α y^β (b+z)^γ. The v2 (Mar 2026) form is general
  homogeneous Ψ(x)Φ(z). Cite the right one.
- **GATE author spelling:** arXiv lists "Vrzla, Matej"; Jones JEP spells "Vrzala".
- **Zhang & Zhang (2026)** "returns to scale 1.3, compute share 0.35": from a search-engine snippet only; not verified in
  the PDF.
- **Thompson et al. Table 2** was re-checked against the arXiv PDF (v2): coefficients, SEs, N and implied scaling factors
  confirmed.
