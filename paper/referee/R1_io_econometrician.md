# Referee Report — Referee 1

**Manuscript:** "Scaling Laws as Production Functions" (Okar and Claude), submitted to the *American Economic Review*
**Referee expertise:** empirical IO; production-function estimation (proxy variables and control functions, GNR, markups and wedges, biased technical change)
**Version reviewed:** compiled `main.pdf` (133 pp.) and LaTeX sources. Page numbers below refer to the compiled PDF. Numbered equations and Propositions 1–7 are from the main text; A-numbered results are in Online Appendix A. The authors' own experiment (Section IV.E, the "TBD-m9" placeholders) is still running. I comment on its design only.

---

## 1. Summary

The paper argues that neural scaling laws, which relate a language model's held-out loss to parameters $N$, training tokens $D$ and compute $C\approx 6ND$, are production functions, and that fitting them is production-function estimation. The paper does five things.

1. **Framework (Section I).** It writes the Chinchilla law as a log production function with Hicks-neutral and factor-augmenting productivity. It gives a dictionary to production and IO concepts (Table 1). It proves three results:
   - duality among Chinchilla's three estimation approaches (Proposition 1);
   - under the multiplicative cost $6ND$, any interior compute optimum requires $0<\sigma<1$ (Lemma 1);
   - a lab that minimizes lifetime compute $6ND+2NT$ sets $w\equiv\varepsilon_N/\varepsilon_D = 1+T/(3D)$ (Proposition 2).

   Inverting the last result gives "revealed inference demand."

2. **Identification (Section II).**
   - On-path data identify the expansion path and the frontier, but curvature only through functional form (Proposition 3).
   - Information about $M^*$ and $\sigma^*$ is of second and fourth order in the dispersion of log wedges (Proposition 4).
   - The drift of $M^*$ signs the Hicks bias (Proposition 5).
   - Transmission and selection biases in cross-lab data have signs set by the "compute regime" (Proposition 6).
   - A parametric projection and a shape-restricted lower bound give partial identification of $T$ (Proposition 7).
   - Monte Carlo evidence supports these results.

3. **Technology (Section IV).** Across seven public sweeps:
   - Under the Chinchilla form, $\sigma^*$ is 0.73–0.83.
   - The restriction $\kappa=1$ is rejected everywhere; with $\kappa$ free, $\sigma^*$ is 0.51–0.71, and about 0.7 on Farseer.
   - $a$ and $M^*$ are not portable across sweeps.
   - Data quality is mostly an asymptote shift plus a neutral scale, with a small factor-biased component.
   - The Kaplan–Chinchilla gap in the allocation exponent decomposes into input mismeasurement and flexible inputs that were not re-optimized.

4. **Revealed inference demand (Section V).** For 173 open-weight base models:
   - The median $\hat w$ is 3.19, i.e., planned lifetime inference compute of 2.2 times training compute.
   - The industry has moved from under-training to over-training.
   - Open developers over-train more than closed ones.
   - Downloads rise with $\ln M$ at fixed compute.

5. **Observational production functions (Section VI).**
   - A "LaLonde-style" design-matched comparison finds no detectable bias in cross-lab returns to compute, but an unreliable split between parameters and data.
   - The paper replicates Ho et al. (2024) and shows the published point estimate is not the optimizer's converged minimum; estimates lie on a DMR ridge.
   - The Kaplan-to-Chinchilla rebalancing is an allocative, not a technical, gain.

## 2. Overall assessment

The project is ambitious and interesting, and much of the execution is careful. The paper has several real strengths:

- The dictionary is useful.
- The replication work is exemplary: Ho et al. to $4\times10^{-7}$, and Porian et al.'s ten exponents to 0.003.
- The Monte Carlo evidence is well designed.
- The authors are commendably frank about fragility. They call the technology band a "sensitivity envelope, not an identified set" and state that "levels are not robust."

Training sweeps are a setting in which the cost side is (nearly) known and input variation is designed. That is a potentially valuable laboratory for production-function econometrics, and an IO audience will find the premise attractive.

In its current form, however, I do not think the paper clears the bar for a general-interest journal. My concerns fall into four groups.

1. **The economic headline — revealed inference demand — is not yet credible as a structural measurement.** The inversion rests on a behavioral model ("conduct") that fits badly with how the sample was generated:
   - Open-weight developers do not bear the inference cost.
   - Model sizes are chosen from a discrete menu of deployment tiers.
   - More than half of the multi-model families train every size on the same token budget.

   The paper invokes the De Loecker–Warzynski analogy, but not the lessons of the critiques it cites:
   - Bond et al. is cited but never engaged.
   - The "Raval-style" test is not Raval's test.
   - The Demirer-style contamination by factor-biased technology is bounded with evidence much too narrow for the cross-lab heterogeneity the paper itself documents.

   What survives is essentially that most open models are trained well beyond about 20–35 tokens per parameter, increasingly so over time. That is a descriptive fact the structural apparatus does not sharpen much.

2. **The level of the technology used for the inversion is not identified at the scale where it is applied.** $M^*(C)$ at $10^{23}$–$10^{25}$ FLOP is extrapolated two to three orders of magnitude beyond every design. Across the paper's own fifteen technologies it ranges from 1.4 to 290 at $10^{24}$ FLOP. The "technology band" leaves out the technologies that would overturn the sign for large models.

3. **Several IO analogies are loose, and some formal results are modest relative to how they are presented.**
   - The "ACF functional dependence" of Proposition 3 is the classical Marschak–Andrews / Bond–Söderbom collinearity under common prices.
   - The "GNR-style" flagship normalization is not GNR.
   - Proposition 4 is a design-theoretic property of estimating a curvature parameter.
   - Proposition 7 is a projection argument plus a bound the paper shows is uninformative.

   The paper also misses a simple, model-free identification result for $\sigma^*$ and $w$ that its own framework implies (Major Comment 7). That result would substantially strengthen the technology section.

4. **The headline $\sigma\approx0.7$ comes essentially from one sweep.** Estimates that let the data choose the curvature are heterogeneous across sweeps, and heterogeneity is not tested. Inference practice is inconsistent across sections: pairs bootstrap on fixed designs, clustering on nine budgets and eight widths, and wild p-values that contradict the reported standard errors.

The paper also tries to do too much for one article: nine main-text formal results, seven sweeps, three observational samples, a replication of algorithmic progress, a hyperparameter frontier analysis, and an experiment that is not yet finished. A sharper paper would build around (i) identification and the technology, with the model-free $\sigma^*$ result, and (ii) a properly specified and tested inversion. It would move the rest to companion work.

**Recommendation: major revision.** I would want to see a substantially re-specified Section V before I could support publication.

---

## 3. Major comments

### 1. The inversion needs an explicit model of lab conduct; lifetime-compute minimization with exogenous $T$ is not credible for most of Sample B

Proposition 2 (p. 12; Proposition A8, pp. 84–85) assumes that the developer minimizes $6ND+2NT$ subject to a loss target, with $T$ fixed and the inference cost borne by the developer at the training-FLOP price (or $p$ times it, p. 41). Recovering $T=3D(w-1)$ is then like recovering marginal cost from a pricing first-order condition under assumed conduct. The object is only as good as the conduct assumption, and the paper never tests or even states the alternatives. Four problems stand out.

**(a) Open-weight developers do not pay for inference.**
- Users (and third-party hosts) bear inference costs. The developer internalizes them only through adoption: demand for the model depends on how cheap it is to run.
- The right objective is then something like $\max_{N,D}\ V(\text{adoption}(L,N)) - c\cdot 6ND$, not $\min\,6ND+2NT$.
- The first-order condition then involves the elasticity of adoption with respect to $N$ at given quality, not "lifetime tokens." Its inversion yields a shadow value of smallness, not $T$.
- This matters for interpretation. The paper finds that open developers have *higher* wedges than closed ones (0.57 log points, p. 46), and the conclusion reports "planned lifetime inference compute." Those statements have no clear meaning for firms that bear no inference compute.

**(b) Demand depends on $N$ directly, not only through $L$.** Latency, memory footprint and serving price all fall with $N$. With $T=T(L,N,\dots)$, the developer's condition acquires a term in $\partial T/\partial N$. The paper's own validation regression (Table 7) shows that usage depends strongly on $N$ at fixed $C$. So the maintained assumption behind the inversion — that $T$ does not respond to $N$ at given $L$ — is contradicted by the evidence offered in its support.

**(c) Profit maximization versus cost minimization.** Cost minimization conditional on $(\bar L,T)$ follows from profit maximization only if the conditioning variables are not themselves functions of the choice variables. Point (b) says they are.

**(d) Durability and obsolescence.** Lifetime $T$ is a demand rate times an expected product life. Product lives have shortened sharply as release cadence accelerated. The model has no role for this, yet it is first order for interpreting the trend in $w$ (p. 45).

**Requests.**
- Write down at least three conduct models and derive what $w$ identifies under each:
  1. the current model (developer bears inference cost; API or closed providers);
  2. adoption-maximizing open release;
  3. size-class competition with $N$ fixed by deployment tiers (see Comment 2).
- Treat the choice among them as a conduct-testing problem in the sense of Berry and Haile (2014), Backus, Conlon and Sinkinson (2021) and Duarte, Magnolfi, Sølvsten and Sullivan (2024). Use instruments that shift one model's prediction but not the others'. Candidates:
  - changes in consumer-GPU memory (shifts the tier constraint);
  - on-device deployment programs;
  - the arrival of hosted-inference marketplaces (shifts inference demand for open models).
- At a minimum, confine the "planned inference compute" interpretation to models whose developer serves them (closed/API models, or open models with first-party hosting). Present the open-weight results as a different estimand.

### 2. Size tiers and family-level token budgets make the $N$ margin — the one that identifies $T$ — the least free margin in the data

Because tokens carry no inference cost, $T$ is identified only through the parameter margin. Two features of Sample B suggest that margin is often not chosen on a continuum.

**(a) Model sizes bunch at deployment tiers:** 0.5B, 1–1.5B, 3B, 7–8B, 13–14B, 32B and 70B. The paper documents bunching at 6.5–9.5B (26 percent against 9 percent under a log-normal; p. 49) and treats it as a second-order rival because being in that tier raises $\ln\hat w$ by only 0.11.

That test cannot detect the alternative.
- If *every* family picks $N$ from the tier menu, the comparison group (non-7B models) is also tier-constrained.
- Under a binding memory/size constraint, Proposition 2(iv) gives $w=(1+\nu)(1+T/3D)$. Any $w$ is then rationalizable with $T=0$.
- The paper's own calculation (p. 49) shows a cap at half the training-optimal size delivers $w=1.78$ with no inference demand.

A test with power would ask whether $\ln\hat w$ is related to the distance from the *nearest* tier boundary across all tiers. Better, it would exploit changes in the tier menu over time: GPU memory sizes, quantization practice.

**(b) $D$ is set at the family level.** By my count from `m3_wedge_models.csv`:
- 24 of the 44 Sample-B families with at least two models train every size on a common $D$ (within 10 percent);
- these families contain 83 of the 149 models in multi-model families;
- examples: OPT, Pythia, Qwen2.5, Qwen3, BLOOM, Llama 2, Llama 3.1, Yi-1.5, RWKV, XGLM.

In these families, $\hat T_i/D=3(\hat w_i-1)$ is a deterministic function of $N_i$ given $D$. The implied demand profile across sizes comes entirely from the technology's curvature, and the choice data carry no independent information about $T$.

The paper calls the resulting rejection of the common-elasticity over-identification test "mechanical" (p. 47). I read it the other way. Under lifetime-cost minimization, a common $D$ across sizes is a knife-edge coincidence: $T(N)$ must happen to make $w(N,\bar D)-1$ proportional to $T/\bar D$ at every size. Under the alternative — a family-level data pipeline run to a fixed token budget with $N$ from the tier menu — it is the natural outcome. That is evidence against the maintained conduct, not a nuisance.

The Pythia example (p. 45) makes the same point and is waved away as "a useful warning." It is the modal case.

**Requests.**
- Report results separately for families with and without common $D$.
- Treat common-$D$ families as uninformative about $T$, or model $D$ as a family-level choice. The lifetime-cost problem is then solved jointly over the family, with a shared token budget, and the first-order condition is different.
- Redo the rival-explanations section (pp. 47–49) with the tier constraint as a first-order alternative, not a robustness check.

### 3. The De Loecker–Warzynski analogy and its critiques are not adequately addressed

The paper motivates the inversion by DLW (pp. 3, 13, 40). It says the analogy "carries over one known weakness of the production approach (Raval 2023; Demirer 2020)" and lists Bond et al. (2021) among "its critiques" (p. 5). For an IO readership this is the crux, and the treatment is too thin.

**(a) The object $w$ is Raval's test statistic.** Raval (2023) compares markups estimated from two flexible inputs, $(\theta_L/s_L)/(\theta_M/s_M)$. The paper's $w=(\varepsilon_N/1)/(\varepsilon_D/1)$ is exactly this ratio with training-cost weights in place of revenue shares.

In Raval's world, a ratio different from one is evidence of *misspecification*. Raval and Demirer (2020) attribute it chiefly to factor-augmenting (labor-augmenting) technology that differs across firms. The paper interprets the same deviation as inference demand. The burden of proof is therefore to show that cross-lab factor bias $\chi$ is small relative to $\ln\hat w$.

The only evidence offered is DataDecide recipe tilts: 25 recipes, one architecture, one codebase, one tokenizer. They move $w$ by 1.25–1.29 (p. 49). This almost certainly understates cross-lab heterogeneity, and the paper's own evidence shows why.
- Across its sweeps, $M^*(10^{21})$ ranges from 3.4 to 60, a factor of about 18 (Table 4).
- Through $\ln w=(S/2)\ln(M/M^*)$ with $S/2\approx0.36$, heterogeneity of that size in the location of the path moves $\ln w$ by about 1.0.
- That is comparable to the median $\ln\hat w$ of 1.16 (p. 49).

The paper attributes the cross-sweep dispersion to "units, conventions and recipes" (p. 35). That is precisely the $\chi$ heterogeneity (and unit heterogeneity) that contaminates the wedge across labs.

Published lab-specific allocation laws, which the authors' own literature notes contain, point the same way:
- MiniCPM reports an optimal ratio near 192 tokens per parameter;
- DeepSeek LLM reports a data-quality-dependent allocation exponent (0.450 to 0.578).

Both are direct evidence that $\chi$ and $a$ differ across labs by amounts that matter for $w$.

**(b) A genuine Raval-type over-identification test is available and should be run.** The paper's "Raval-style" test (common iso-elastic demand $T=\tau_f N^{-\eta}$ across families, p. 46) is a test of an auxiliary demand specification, not of the production approach. Raval's logic calls for a second margin whose first-order condition involves the same $T$ with different cost weights. LLM architecture provides several:
- **MoE sparsity:** training and per-token inference FLOPs scale with active parameters; memory and serving cost scale with total parameters.
- **KV-head count** (GQA/MQA versus MHA): quality versus inference memory bandwidth, nearly free in training.
- **Vocabulary size:** embedding parameters are cheap in FLOPs but not in memory.
- **Distillation**, where the teacher's inference is a training cost.

A second option uses two outputs whose isoquants differ — for instance C4 loss versus a downstream bits-per-byte measure. The Online Appendix robustness table (Section D.2) already shows that $\sigma^*$ differs across outputs for the OLMo ladder (0.825 on C4 loss, 0.763 on task bits per byte). If $w$ is "revealed demand," it should not depend on which output's technology is used. The ordinality result (Lemma A1(iv)) guarantees invariance only to transformations of the *same* output.

Even one such cross-margin comparison would do more for credibility than the present usage regressions.

**(c) Bond, Hashemi, Kaplan and Zoch (2021) are not engaged at all.** Their two points map onto this setting as follows.
- **Revenue versus quantity output.** The paper's technology uses a quantity-like output (held-out loss), so the revenue problem does not arise in estimation. But the lab optimizes its *own* output concept — capabilities as users value them — whose isoquants in $(N,D)$ may differ from those of MassiveText loss. If they do, $\varepsilon_N/\varepsilon_D$ from the reference loss is the wrong elasticity ratio. This is the analog of using the wrong output concept in DLW.
- **Identification of the elasticity from the same data.** Here the paper has a real advantage it should make explicit: the elasticities come from *designed experiments*, not from proxy-variable estimation on the same observational choices. That breaks the circularity Bond et al. emphasize. The price is external validity of the experimental technology to each lab, which is exactly point (a).

A clear paragraph on this trade-off is necessary.

**(d) Doraszelski and Jaumandreu's reexamination of the DLW method and De Ridder, Grassi and Morzenti's "Hitchhiker's Guide to Markup Estimation" are close parallels.** The latter's conclusion — dispersion and ranks are informative, levels are not — matches this paper's own conclusion (p. 49). Cite both, and state that the defensible contribution is ordinal.

### 4. The level of $M^*(C)$ at frontier scale is not identified, and the technology band is constructed selectively

$\ln\hat w=(S/2)[\ln M-\ln M^*(C)]$ (equation 11, p. 40). For models at $10^{23}$–$10^{25}$ FLOP, $M^*(C)$ is extrapolated from designs whose largest budget is $1.3\times10^{22}$ (Chinchilla) or less. The extrapolation runs through the path slope $a$. Section IV.D argues convincingly that $a$ is the object "most exposed to measurement conventions and to flexible inputs" (p. 38). The inversion then relies on it to extrapolate over two to three orders of magnitude.

**(a) The paper's technologies disagree wildly at frontier scale.** From `m3_wedge_technologies.csv`, $M^*(10^{24})$ is:

| Technology | $M^*(10^{24})$ |
|---|---|
| Gadre et al., RefinedWeb | 1.4 |
| Gadre et al., RedPajama | 3.9 |
| Gadre et al., C4 | 6.0 |
| Meta Llama 3, primal fit | 9.6 |
| Chinchilla refit | 17.6 |
| Meta Llama 3 law (A2) | 31 |
| Hoffmann et al., Approach 3 | 62 |
| Farseer, embeddings counted | 68 |
| OLMo ladder | 290 |

At $10^{25}$ FLOP the range is 1.0 to 528.

**(b) The band omits the technologies that would reverse the sign.**
- The "technology-sensitivity band" (p. 41) uses six common technologies. It leaves out the OLMo ladder and Muennighoff et al., both among the paper's seven sweeps, applying the OLMo ladder only to AI2 models. It also leaves out Gadre et al.'s C4 and RedPajama rows.
- By my rough calculation, applying the OLMo-ladder technology to all of Sample B (with total rather than OLMo-convention $N$) gives a median $\hat w$ of 1.26. Only 75 percent of models have $\hat w>1$, and only 70 percent of the 80 models above $10^{23}$ FLOP.
- The headline "79 percent have the whole band above one" (pp. 3, 42) therefore depends on an unstated inclusion rule.
- **Request:** report the share with $\hat w>1$ and the median under *every* technology the paper estimates (including all three Gadre corpora, OLMo, Muennighoff, DataDecide where defined, and the $\kappa$-free versions). Also report it under the published lab-specific laws (Llama 3, DeepSeek LLM, MiniCPM). State the inclusion rule for the band ex ante.

**(c) The reference technology imposes a restriction the paper rejects.**
- The reference technology is the Chinchilla refit with $\kappa=1$. Section IV rejects $\kappa=1$ in all seven sweeps and argues that the $\kappa$-free inner exponents are the right curvature.
- Since $w$ depends on the inner exponents, the reference should be $\kappa$-free, or the choice should be justified.
- On Farseer, freeing $\kappa$ raises the median $\hat w$ from 2.67 to 3.85 (p. 43).

**(d) Functional form matters as much as $M^*$.**
- Within Farseer's support, Farseer's own form gives larger wedges than the Chinchilla form (median 4.12 versus 3.44). Above that support it gives smaller ones (1.54 versus 2.12), and Llama 3 70B is under-trained (p. 49).
- The additively separable family is the only reason the wedge depends on the technology through $(S,M^*)$ alone (Corollary A3). The paper should treat the functional form as part of the identified set, not as a rival.

**(e) Make $M^*(C)$ at frontier scale the explicit object of partial identification.** Proposition 7 in its present form is a projection argument (part (i)) plus a bound the paper shows implies only $T\ge0$ (part (ii), p. 23). Both are correct and neither is informative.
- A useful partial-identification analysis would bound $M^*(C)$ at $10^{23}$–$10^{25}$ FLOP under stated assumptions. Examples:
  - monotonicity of $M^*$ in $C$, with bounds on its elasticity drawn from the range of $a$ across sweeps;
  - lab-published IsoFLOP minima at the largest available budgets.
- Then map those bounds into bounds on $w$ and $T$. Report which models' signs are identified under which assumptions.

**(f) Consequence for the headline.** The robust sign statement is, to a close approximation, "$M$ exceeds $M^*(C)$ for $M^*(C)$ between about 20 and 35." That is a descriptive statement about tokens per parameter which Figure 4 already shows. The cardinal headline ("2.2 times training compute") should come out of the abstract.
- The introduction's "Llama 3 8B has $w=5.27$ (95 percent interval [3.97, 8.12])" (p. 3) reports an interval reflecting only sampling error in one sweep. The paper's own band is [2.48, 12.0], and the full set of its technologies gives a wider range still.
- Report ranges, not pseudo-precise confidence intervals, wherever extrapolation dominates.

### 5. The cost side is not an engineering identity

The paper stresses that "the cost function is an engineering identity, input prices drop out of the first-order conditions" (p. 5; also pp. 7–8, Table 1 note). This holds only in FLOP units, and FLOPs are not costs.

- **Training.** Dollar cost is GPU-hours, not $6ND$. Model FLOPs utilization varies with $N$: small models underutilize large accelerators, and large models pay communication and pipeline overheads. Attention FLOPs depend on context length, and the paper acknowledges but does not model this (p. 7). Wall-clock constraints interact with critical batch size, bounding $D$ for a given cluster and deadline. If dollar cost scales like $N^{1+\delta}D$, the first-order condition becomes $w=(1+\delta)(1+T/3D)$. Any systematic $\delta$ is another wedge that cannot be separated from $T$.
- **Inference.** Serving cost per token is not $2N$ FLOPs times a constant price. Decoding is memory-bandwidth bound, so cost depends on bytes of weights and KV cache, batch size, quantization and hardware. The cost elasticity with respect to $N$ need not be one. The paper's $p$ (p. 41) is a constant and does not address this.
- **Data.** Tokens are not free beyond FLOPs. Acquisition, filtering, licensing and synthetic generation carry costs. Data constraints on unique tokens are relevant at 18–36T tokens.

**Requests.**
- Replace "identity" with "FLOP-accounting approximation" throughout.
- Report how $\hat w$ and $\hat T$ change under plausible MFU-by-size schedules and inference cost elasticities (a sensitivity table).
- Note that heterogeneity in these cost shifters is, in Bond and Söderbom's sense, precisely the relative-price variation that could identify the technology from observational data.

### 6. The evidence for $\sigma\approx0.7$ comes essentially from one sweep, heterogeneity is not tested, and the interpretation needs care

**(a) Heterogeneity across sweeps is significant and is not addressed.**
- The abstract and conclusion state that parameters and data have "an elasticity of substitution near 0.7 once the data choose the curvature" (pp. 1, 58).
- With $\kappa$ free, Table 4 reports 0.701 (Chinchilla), 0.710 (Farseer), 0.607/0.623/0.594 (Gadre), 0.544 (OLMo ladder) and 0.511 (Muennighoff).
- The OLMo estimate, 0.544 (s.e. 0.021), differs from Farseer's 0.710 (0.003) by about eight standard errors. That is not "weak identification in the 30–35-run sweeps" (p. 33).
- Test homogeneity of $\sigma^*_\kappa$ across sweeps. If it is rejected, either explain the heterogeneity (recipe, architecture, output, parameter-count convention) or present $\sigma$ as sweep-specific.
- The recommended "0.5 to 0.8 with a central value near 0.7" (p. 58) needs a principled aggregation (e.g., a random-effects summary with a heterogeneity estimate).

**(b) The "three estimators that agree" on Farseer do not agree by the paper's own standard errors, and they estimate different objects.**
- The local-quadratic median is 0.690 [0.679, 0.699]; the $\kappa$-family estimate is 0.710 [0.705, 0.716]. The intervals do not overlap (p. 33, Table 4 Panel B).
- The local estimator reports the *median local $\sigma$ over 42 interior design points*. Those points have wedges well away from one, so this is not $\sigma^*$, the elasticity at $w=1$.
- Compute the local $\sigma$ along the estimated expansion path (or at $w=1$ by local interpolation) before comparing.
- The tight intervals also suggest the pairs bootstrap understates uncertainty for a grid design with shared hyperparameters and data order (see Comment 9).

**(c) Under misspecification, pseudo-true values depend on the design.**
- $\kappa=1$ is rejected everywhere, and Farseer's non-separable form fits better out of sample.
- The estimated $(\alpha,\beta,\kappa)$ are therefore projections whose limits depend on each sweep's design distribution.
- Cross-sweep comparisons of $\sigma^*$ (and to capital–labor elasticities) mix technology differences with design differences.
- The model-free estimator of Comment 7 avoids this.

**(d) "Gross complements" is partly imposed.**
- Lemma 1 shows that $0<\sigma<1$ is necessary for an interior compute optimum, and every family fitted imposes $\sigma<1$.
- The paper concedes that this is "a check on the data rather than a discovery" (p. 34). The abstract and introduction nonetheless present gross complementarity as a finding (pp. 1, 3).
- The empirical content is: (i) IsoFLOP profiles have interior minima, and (ii) the level. Say so.

**(e) What does $\sigma$ between parameters and tokens *processed* mean for economic models?**
- The conclusion (p. 59) maps the estimate into AI-and-growth models: "a binding constraint on one input — for instance, a limited stock of text — lowers the return to the other faster."
- But $D$ is tokens processed, including repetitions, not the stock of text. The data-wall question concerns substitution between compute (or parameters) and *unique* data under repetition (Muennighoff et al. 2023). That is a different elasticity, and this paper does not estimate it.
- Likewise, "effective compute" aggregators in growth models aggregate $C=6ND$, which is itself a function of both inputs.
- Clarify which economic elasticity the estimate informs, or drop the growth-model implications.

### 7. A model-free identification result for $\sigma^*$ and $w$ follows from the paper's own framework and should replace much of the parametric apparatus

Let $x\equiv\ln M$ and $c\equiv\ln C$. Let $s_x\equiv\partial y/\partial x|_c$ be the slope along an isocost and $s_c\equiv\partial y/\partial c|_x$ the slope along a ray of constant $M$. Since $n=(c'-x)/2$ and $d=(c'+x)/2$:
$$\varepsilon_N=s_c-s_x,\qquad \varepsilon_D=s_c+s_x,\qquad w=\frac{s_c-s_x}{s_c+s_x}.$$
At a compute-optimal point ($s_x=0$), Lemma A2's identity $\sigma=P/(P+Q)$ yields
$$\frac1{\sigma^*}-1=\frac{2\,(-y_{xx})}{dy^*/dc}=\frac{2\,L_{xx}}{|dL^*/dc|}=\frac{L_{nn}}{2\,|dL^*/dc|}.$$
The last expression uses the curvature of the IsoFLOP profile in $\ln N$. Because both numerator and denominator scale by $g'$ under any monotone transformation of output, the ratio needs no estimate of $E$, $\kappa$ or the functional form.

In words: **the on-path elasticity of substitution is the curvature of the IsoFLOP parabola divided by twice the slope of the loss–compute frontier.** Both are Approach-1/Approach-2 objects that practitioners already estimate. The formula also makes transparent what $\kappa=1$ does: it ties the curvature to $\gamma$ through $S=\gamma/(ab)$, which is why the Chinchilla-form $\sigma^*$ is "pinned down by the on-path frontier elasticity" (p. 19).

The authors should:
1. estimate $\sigma^*$ this way on every IsoFLOP design they have (Chinchilla, Llama 3, Marin's three corpora, Porian et al.'s 16-architecture ladder), accounting for finite-grid parabola bias (Czech et al.) with local polynomials of adequate order;
2. estimate $w$ nonparametrically *inside* the support at high $M$ (Farseer up to $M\approx2{,}570$; the new experiment up to 2,031) and compare it with the parametric extrapolation. This gives a direct test of the extrapolation that Section V relies on, in the region where Sample B's small models sit;
3. use the in-support $w$ to test the family's implication that $\ln w$ is linear in $\ln(M/M^*(C))$ with slope $(1/\sigma^*-1)$.

This would replace a fragile parametric chain with a transparent, E-free, form-free estimator, and it is what an IO reader would want to see.

### 8. Precision of the IO analogies and the novelty of the formal results

The dictionary is a contribution, but several entries overstate the correspondence. Some formal results are presented as more novel than they are. An IO referee pool will notice. Specific items:

**(a) Proposition 3 and "ACF functional dependence" (pp. 2, 15–16; Table 1).**
- In ACF (2015), functional dependence refers to the first stage of OP/LP: the variable input is a deterministic function of the state variables and the proxy, and so is collinear with the nonparametric control function. The paper uses the term correctly in Proposition A7(iii) (p. 83).
- Proposition 3, however, is the older point that when all firms face the same relative prices and optimize, inputs lie on a lower-dimensional manifold and elasticities are not separately identified. That point goes back to Marschak and Andrews (1944). Bond and Söderbom (2005) state it sharply (identification requires price variation or adjustment frictions), and GNR's non-identification result for flexible inputs is closely related.
- The setting is the cleanest possible instance: log-cost weights are literally identical across labs.
- Recast Proposition 3 in these terms and cite them. The "escape routes" (price shocks, optimization error) are Bond–Söderbom's.

**(b) Proposition 4 ("the better labs optimize, the less their data reveal," pp. 17–18).**
- The rates — information about a location parameter of order $v^2$ and about a curvature parameter of order $v^4$ in the design spread $v$ — are a standard property of estimating a second-order coefficient. Compare the optimal-design literature for nonlinear models, e.g., Box and Lucas (1959) and Kiefer and Wolfowitz (1959).
- The paper's value added is the mapping to wedges and Farrell losses (the decomposition in equation 7 is nice). Say so and cite the design literature.
- The slogan's empirical relevance is also unclear. The released models used in Sections V–VI are far off the path (Llama 3 8B at 100 times $M^*$). The on-path problem arises in compute-optimal ladders that researchers design, not in the behavior of optimizing labs.
- Where in the data used here does behavioral collinearity actually bind? The identification problem in observational data is endogeneity plus technology heterogeneity, not lack of transverse variation.

**(c) Proposition 3(iii) is hard to parse (p. 15; Proposition A1(iii), pp. 76–78).**
- As proved, $(\alpha,\beta)$ are *globally* identified from on-path outcomes plus the observed path slope, but the information matrix is singular at the truth. Rotnitzky et al. (2000), whom the proof cites, then imply nonstandard (e.g., $n^{1/4}$) rates and invalid Wald inference.
- State the global versus local distinction and the inferential consequence explicitly.
- Part (ii)'s "if and only if" is too strong: $\gamma$ can be identified by instruments or panel structure outside the stated model.

**(d) "An IsoFLOP sweep ... is an experimenter-made shock to the relative price of the two inputs" (p. 16).** An IsoFLOP sweep assigns inputs directly. There is no behavioral response and no price. The price-shock analogy fits observational wedges, not designed experiments. Keep the two separate.

**(e) The "GNR" flagship normalization (p. 46).**
- GNR identify the flexible-input elasticity from its first-order condition together with an *observed* expenditure share.
- Here the "share" (the inference weight $T$) is the unknown, and the normalization simply *assumes* $w_f=1$ for the flagship.
- That is a calibration assumption, not GNR, and the paper then shows it fails for post-2024 flagships.
- Similarly, "the scaling analog of the first-order-condition approach of GNR" (p. 16) for recovering $A/B$ on the path is a normalization by assumed optimality. The Design-B Monte Carlo (p. 101) shows that this biases $\ln M^*$ by exactly $2E[\ln w]/(\alpha+\beta)$.

**(f) The DMR refinement (Proposition 5; pp. 20–21; remark p. 81).**
- The claim that "both signs are identified" — the sign of the Hicks bias and which input is augmented faster — holds only when $\alpha=\beta$. In the family, the bias is $b_1g_D-a_1g_N$, so its sign is the sign of $\beta g_D-\alpha g_N$, not of $g_D-g_N$.
- The remark "This corrects the statement that on-path time series cannot identify the sign of the bias" (p. 81) cites no one.
- Proposition 5 is never taken to vintage data, although published IsoFLOP laws exist at several dates (Kaplan, Chinchilla, DeepSeek LLM, Llama 3, MiniCPM). Either apply it, with appropriate caveats about units, or present it as a methodological remark.

**(g) Nerlove (pp. 3, 36–38).**
- Nerlove (1963) found that scale economies decline with firm size and concluded that a constant-elasticity specification averages a varying elasticity. That is a functional-form (non-homotheticity) point. His small-firm scale economies were not "spurious" in the sense of input mismeasurement.
- The measurement-error mechanism of equation (9) is closer to scale-correlated mismeasurement of inputs, e.g., utilization in Basu and Fernald (1997), or Collard-Wexler and De Loecker (2016).
- Reword. The cost-function use of Nerlove on p. 9 is fine.

**(h) Table 1, "Benchmark accuracy; Elo — TFPR-like bounded transform" (p. 9).** TFPR confounds output prices with physical productivity. A bounded monotone link of a quantity index is a measurement map, not TFPR. Likewise, Klette and Griliches (1996) concerns firm-specific output prices. A tokenizer changes the units of *both* output (loss per token) and input ($D$), and not proportionally across text types.

**(i) "LaLonde-style" (Section VI.A).**
- LaLonde compares non-experimental estimators with an experiment on the *same* treated population.
- Here the "experimental" technology comes from different labs' recipes (AI2, OpenLM) and is imposed on other labs' models.
- Closer analogs are Todd and Wolpin (2006) and Hotz, Imbens and Mortimer (2005) on validating across populations. The distinction matters because Section IV documents large technology heterogeneity across recipes (Comment 10).

**(j) Selection result (Proposition 6(iii)).** Attenuation of OLS under truncation on the dependent variable with log-concave errors is known (Goldberger 1981 and the truncated-regression literature). Cite it and present the funding-versus-selection interaction as the new part.

### 9. Inference: a principled and consistent approach is needed

**(a) Fixed designs call for design-conditional inference.**
- The designed sweeps are fixed-design nonlinear regressions: the experimenter chose $(N,D)$. The pairs bootstrap resamples design points and so changes the design. Under misspecification it also changes the pseudo-true value.
- The natural procedures condition on the design: wild or residual bootstrap, or sandwich inference.
- The reference estimator is effectively LAD (84 percent of residuals in the linear region, p. 29). Wild bootstrap for LAD needs appropriate weights, e.g., Feng, He and Hu (2011).
- Pre-specify one procedure. Currently, conclusions flip across variants: the duality test of Hoffmann et al.'s parameters has $p$ between 0.040 and 0.38 (p. 30), and the frontier restrictions are "rejected under the design-resampling bootstrap [but not] under the wild bootstrap" (p. 30).

**(b) Clustering on nine budgets needs a rationale (p. 29).**
- In the Chinchilla extraction, $D$ is *constructed* as $C/(6N)$ from the nominal budget, and losses are digitized with quantization of about ±0.01. That is comparable in size to the residual standard deviation of 0.0075 in $\ln L$.
- What is the within-budget dependence — shared seeds, shared digitization errors, common schedule effects? Without a model of it, nine-cluster inference is both imprecise and hard to interpret.
- Consider modeling the digitization error explicitly (interval-censored losses).

**(c) Weak identification.**
- The profile-LR analysis for $\sigma^*$ with $\kappa$ free (Table 3 Panel D; Figure 2 Panel B) is on a bounded grid [0.50, 0.95] and uses (I believe) Gaussian likelihood ratios. The residuals are heavy-tailed (MAD-based s.d. 0.0049 against 0.0075).
- The $\chi^2_1$ calibration is questionable when nuisance parameters ($\kappa$, $E$) are themselves weakly identified.
- Consider subvector-robust procedures (Andrews and Cheng 2012; Andrews and Mikusheva 2016), or at least a parametric bootstrap of the profile-LR statistic under the null.
- Report whether the confidence set reaches the grid boundary. By Proposition 3(iv) it should extend over $(0,1)$.

**(d) Reported standard errors on levels are parameterization artifacts.**
- The paper cites standard errors of 125 and 1,293 on $A$ and $B$ (p. 16) as evidence that on-path data do not identify levels. The Chinchilla data are an IsoFLOP design with substantial transverse variation, and the large standard errors reflect evaluation at $N=D=1$.
- The paper's own recommendation — normalize at geometric means (Klump et al. 2007), which cuts the condition number from 2,000 to 62 (p. 59) — makes this clear.
- Similarly, the standard error of $\ln G$ (0.84, and 1.92 clustered) is the uncertainty of the path level at $C=6$ FLOP.
- From `m1_chinchilla_horse_race.csv`, $M^*(10^{21})$ has a pairs 95 percent interval of [15.5, 28.4], which is reasonably precise. $M^*(5.76\times10^{23})$ has [7.8, 35.1].
- The imprecision is extrapolation in compute, not on-path non-identification. Keep the theoretical point (Proposition 3) and the empirical one (extrapolation) separate.

**(e) Table 8's wild p-values contradict its standard errors.**
- With family fixed effects, the $\theta_D$ bias is $-0.262$ with standard error 0.162 ($t\approx1.6$), yet the reported restricted wild-cluster p-value is 0.02.
- A restricted wild-cluster bootstrap-$t$ with 19 clusters should usually be *more* conservative than normal-CRVE inference.
- I suspect the bootstrap treats the experimental benchmark as fixed while the reported standard error includes its uncertainty, or that singleton families interact with the fixed effects. Reconcile the two, and base claims (including the introduction's "family fixed effects overstate … by 0.20 and understate … by 0.26," p. 3) on inference that accounts for both sources.

**(f) Sample-wide statements need joint inference.** Statements such as "the lower bound of the bootstrap interval exceeds 1 for 87 percent of models" (p. 42) pool model-by-model intervals that share one technology draw. Report the bootstrap distribution of the *share* with $w>1$ (and of the median $w$) across technology draws, which is the relevant joint object.

### 10. Observational section: what the benchmark can and cannot show; the allocative decomposition

**(a) The design-matched comparison is a joint test with low power, and it is over-claimed.**
- The null combines (i) a common technology up to Hicks-neutral shifts, (ii) harness-invariant outputs and (iii) no transmission or selection bias (p. 51).
- Section IV documents that allocation exponents and $M^*$ differ greatly across recipes. That is precisely the factor-biased heterogeneity that violates (i). A null "bias" can then reflect offsetting failures.
- The 95 percent interval of ±23 percent, the $N\le9$B restriction, and 19 developer clusters limit what "no detectable bias" means.
- The introduction and conclusion should say "fails to reject in a low-power joint test." The two elasticity-split biases the introduction reports as findings (p. 3) are called "fragile" in the body (p. 52).

**(b) The experimental surface includes OLMo-2 7B and 13B (Table 8 notes).** These are released production models whose inputs AI2 chose. Exclude them from the "experimental" benchmark.

**(c) "OP-style selection" (Table 8) is a misnomer.**
- Olley and Pakes model exit as a threshold rule in productivity given the state (capital). A quadratic in a "notability" propensity score is not that.
- More generally, what are the state variable, the flexible input and the timing in this industry? Candidates: the lab's compute capacity as state, and per-model compute as flexible. Answering this would let the reader see which proxy estimator is even conceivable. Proposition A7(iii) touches on it; the main text should set it out.

**(d) The allocative-versus-technical decomposition (Section VI.C) counts deliberate over-training as allocative inefficiency against the training-only frontier.**
- That is why the realized "gain" is near one under several technologies and even negative under Meta's (p. 58).
- Since $w<1$ cannot come from inference demand (Proposition 2(v)), a cleaner and identified measure counts only the *elimination of under-training*: allocative efficiency with $w$ truncated at one.
- This removes the need to know $T$ and answers the Gundlach et al. question directly.

**(e) Productivity-dispersion comparisons (p. 54; Appendix Table D-tfp).**
- Family effects on HellaSwag log-odds include benchmark contamination and data similarity to the benchmark, as well as the omitted teacher input that the authors note.
- The "compute-equivalent" conversion divides by $\theta_C=0.33$. That makes the comparison with Syverson's 1.92 sensitive to a single cardinal elasticity from one benchmark.
- Present it with those caveats, or drop it.

### 11. The controlled experiment: design comments (results pending)

The two-lab experiment (Sections III.B and IV.E; Appendix B, pp. 94–96) is a good idea. Several design choices limit what it can deliver, and some can still be fixed.

**(a) One contrast, confounded by domain match.**
- FineWeb-Edu is a filtered subset of FineWeb. Evaluating on both held-out sets gives each lab an in-distribution advantage on its own corpus.
- A test of whether quality is data-augmenting is then confounded with distribution match, as the authors themselves note for DataDecide (p. 36).
- Add at least one neutral evaluation set (e.g., C4, Paloma, or a downstream bits-per-byte set).
- Consider a *dose–response* design with 3–5 mixtures (0, 25, 50, 75, 100 percent Edu). A continuous quality index would identify the sign and size of $\chi$ far more convincingly than two points. It would also turn the experiment into a small "industry" usable as a genuine benchmark for the observational estimators of Section VI — the paper's claimed "laboratory" (p. 60).

**(b) Flexible inputs are tuned for one lab at one budget.**
- The learning rate is calibrated on FineWeb-Edu at 50M tokens for three widths (p. 95) and applied to both labs and every budget.
- Section IV.D's own formula says any inefficiency whose gradient is transverse to the path biases the allocation exponent. Step Law's evidence says the optimal learning rate depends on $D$.
- The high-$M$ cells ($D=800$M at small widths) — exactly where the extrapolation check is to be run — are where mis-tuning is likeliest.
- Add a small learning-rate grid (3 values) at the corner cells for both labs, and report the excess loss at the tuned optimum.

**(c) Scale and units.**
- With 0.4M–49M non-embedding parameters and under $10^{17}$ FLOP, embeddings are 10–73 percent of parameters and training FLOPs are 1.14–3.83 times $6N_{\text{non-emb}}$ (p. 95).
- The input-measurement problem of Section IV.D will therefore be first order.
- Pre-specify the primary convention (I would argue for FLOP-weighted parameters) and report $a$, $M^*$ and $\sigma^*$ under both.
- Present the experiment as a test of *functional form and neutrality*, not as evidence on production-scale levels.

**(d) Inference with trunk sharing.**
- Endpoints of one width share the constant-learning-rate trunk up to their branch points, so their errors are positively correlated across budgets (p. 95), and there are only eight widths.
- Use the seed replicates to estimate the within-trunk error covariance and apply feasible GLS or a covariance-aware bootstrap, rather than relying on eight-cluster inference.
- The seed replicates cover only three widths, three budgets and one lab. Extend them to both labs, since the neutrality test is a between-lab contrast.

**(e) Common random numbers across labs.** Within a lab, all widths share an initialization seed and data order. Across labs the data necessarily differ, but initialization seeds and batch positions can be matched, which would reduce the variance of the between-lab contrast. Also, sharing a seed across *different widths* does not induce meaningful common random numbers in the parameters.

**(f) Power and pre-registration.**
- Before seeing results, report a power calculation for the neutrality test: what $\chi$ is detectable given the seed-noise variance?
- Commit to the estimators, sample restrictions and tests. The placeholders already contain the sentences to be filled ("whether data quality is Hicks-neutral or factor-biased").
- A short pre-analysis plan, deposited before the results are examined, would materially increase credibility.

**(g) Use the experiment for the model-free $w$ of Comment 7.** A factorial design up to $M\approx2{,}000$ is ideal for estimating $s_c$ and $s_x$ locally, hence $w$ and local $\sigma$ at high $M$, without a functional form. That is a direct test of the extrapolation behind Section V at the tokens-per-parameter ratios of small released models. The planned "fit on $M\le100$, evaluate at $M\approx2{,}000$" check tests only one parametric form against itself.

### 12. Scope, focus and claims

The paper runs to about 21,000 words of main text with nine formal results in the main text. Much of the material is either peripheral to the three headline results or rests on fragile evidence the paper itself flags. Examples:
- the stochastic frontier and learning-rate demand analysis of the Step Law grid;
- the Sahal and Hall corollaries;
- the Ho et al. optimizer-convergence finding;
- the illustrative aggregate (equation 13);
- the productivity-dispersion comparison.

**Suggestions.**
- Build the paper around (i) identification (Propositions 1–4, recast per Comments 7–8) and the technology (Section IV), and (ii) a re-specified and tested inversion (Section V, per Comments 1–4).
- Shorten Section VI to the design-matched comparison and the allocative decomposition, or move it to a companion paper.
- Bring the abstract and introduction in line with what the body supports:
  - $\sigma$ as a sweep-dependent range, not "near 0.7";
  - "gross complementarity" as a maintained implication of interior optima;
  - revealed demand as ordinal;
  - the observational benchmark as a low-power failure to reject.

---

## 4. Minor comments

**Front matter and introduction**

1. **Abstract.** "In seven public training sweeps … an elasticity of substitution near 0.7 once the data choose the curvature." Five of the seven $\kappa$-free estimates lie between 0.51 and 0.63 (Table 4). Revise.
2. **p. 2.** "The root-mean-squared error … is 0.28 when every run is compute-optimal." At $s=0$ the model is not identified (Proposition 3), so the RMSE depends on the parameter box and starting values (one of which is the truth, p. 99). Report $s>0$ designs in the introduction.
3. **p. 3.** "Llama 3 8B has $w=5.27$ (95 percent interval [3.97, 8.12])": see Comment 4(f). "Meta's 405-billion-parameter flagship lies on Meta's own training-optimal path ($w=0.98$)." Meta chose the 405B configuration using that law (Grattafiori et al. 2024), so $w\approx1$ is by construction, not evidence. The same applies to the phrasing on p. 43.
4. **p. 3.** "(0.97, standard error 0.18; 0.72 with developer fixed effects)." At fixed $C$, the coefficient on $\ln M$ equals $-1/2$ times the coefficient on $\ln N$. State this in the introduction, not only on p. 47.
5. **p. 4, related literature.** Add Bond and Söderbom (2005); Doraszelski and Jaumandreu's reexamination of DLW; De Ridder, Grassi and Morzenti; the conduct-testing literature (Comment 1); Petrin and Sivadasan (2013) on input-specific gaps between marginal products and prices; and, for heterogeneous technologies across firms, e.g., Kasahara, Schrimpf and Suzuki.
6. **Author note.** The editor will need to decide whether listing an AI system as a co-author is compatible with the journal's authorship policy. COPE's position statement, which many publishers follow, is that AI tools cannot meet authorship criteria. This is an editorial matter, but the authors should be aware of it. The red "[LOCATION TBD]" and all "TBD-m9" placeholders must be resolved before re-review.

**Section I (framework)**

7. **Equation (1), p. 7.** Noise $e^{\epsilon}$ multiplies the reducible part only. Seed and evaluation noise is more plausibly additive in $L$ or $\ln L$, and the choice affects heteroskedasticity and the weight of low-loss runs. Justify, or test the alternative.
8. **p. 7.** "Known to the lab when it chooses inputs." Timing is asserted, not argued. Labs learn recipe quality from small-scale ablations with error. Discuss what information set is plausible and how it affects Propositions 2 and 6.
9. **p. 7.** State that the Chinchilla family is quasi-homothetic with respect to the non-uniform scaling $(N,D)\mapsto(\lambda^{b_1}N,\lambda^{a_1}D)$. This is the real content of the rank-one Hessian (Lemma A4), and it is why all curvature is transverse.
10. **p. 8, Proposition 1, and p. 9.** Chinchilla's Approach 1 reports allocation exponents from the envelope, so it estimates the factor demands as well as the frontier. The mapping "Approach 1 = cost function" is too clean.
11. **p. 11, Lemma 1 discussion.** The empirical counterpart of Lemma 1 is that IsoFLOP profiles are U-shaped. Say so explicitly; Comment 7 gives the quantitative version.
12. **p. 13, Proposition 2(iv).** Add minimum-size constraints (a lab that needs a flagship of at least a given capability or size), time-to-train constraints and MFU heterogeneity to the list of rival wedges with known signs.
13. **p. 14.** The Farrell measure is against the training-only frontier. Call it "training-cost efficiency" throughout to avoid suggesting waste; the paper does this in places but not consistently (e.g., Table 6's "CE").

**Section II (identification)**

14. **p. 15.** The main-text statement of Proposition 3(iii) mixes information from outcomes and from choices ("outcomes identify … and hence $\alpha=\gamma/a$"). Separate the two, per Comment 8(c).
15. **p. 17, Proposition 4.** Assumes $E$ known and Gaussian noise. Say how the rates change when $E$ is estimated. $E$ and the level parameters are nearly collinear in practice, which is part of what the $\kappa$-free profile picks up.
16. **pp. 18–19.** The Monte Carlo noise is calibrated to the Besiroglu residual s.d. (0.0075), which bundles misspecification and digitization. Report the design comparison also at the seed-noise level the new experiment will measure.
17. **p. 20.** The example "members with $\sigma^*$ from 0.33 to 0.89 share frontiers to $10^{-14}$" should note that only $\ln R^*$ agrees to that precision; the argmins agree only to optimizer tolerance.
18. **p. 21, Proposition 6(i).** The model has runs "on the expansion path of a common technology." Released models are far off the path, so Proposition 6 needs a version with heterogeneous wedges, since $\ln w$ enters $y$ through the Farrell loss. Does transmission then also load on the mix?
19. **p. 22, Proposition 7(i).** The "exactly quadratic" variance is immediate because $\ln\hat w$ is linear in $\ln M$ given the estimates. Consider demoting it to a remark.

**Section III (data)**

20. **p. 24.** The off-path spread $\mathrm{sd}(\ln M\mid\ln C)$ is a useful design statistic. For the $\kappa$-free curvature, what matters is transverse spread *relative to noise and to uncertainty in $E$*, and symmetry of offsets around the minimum. Report the Proposition 4 information proxy $\sum\Phi_i^2$ alongside.
21. **p. 26 and Appendix B (p. 96).** $D$ is in each model's own tokenizer while the technology is in MassiveText tokens. Provide a byte- or character-normalized version of $M$ as a robustness check. Tokenizer compression differs by 15–30 percent between 32K and 128K–256K vocabularies, and differentially across languages and code.
22. **Appendix B, Sample A.** For 67 rows, $D=C/(6N)$, so $\hat w$ is a function of $(N,C)$ and inherits any error in Epoch's compute. Report the trend results without them.
23. **Sample B construction.** It starts from the ObsScaling evaluation file (models on the Open LLM Leaderboard), which selects on popularity. That matters for the download regressions (selection on the outcome) and should be discussed.
24. **MoE models.** Active $N$ under a dense technology treats MoE as parameter-augmenting ($\chi<0$), which overstates $\hat w$ by Proposition 2(iv). Report Section V results excluding MoE and multimodal models (Llama 4's reported $D$ appears to include multimodal tokens).
25. **Multi-stage training.** Annealing on high-quality data, long-context extension and continued pretraining make $D$ heterogeneous in quality within a model. Flag and drop such models in a robustness check.

**Section IV (technology)**

26. **p. 29.** "Besiroglu et al.'s published parameters are in fact an LAD-type optimum." Good. Then derive inference for LAD in a nonlinear fixed design (Comment 9(a)).
27. **p. 29.** The exclusion of the five highest-loss runs swings $\hat\beta$ from 0.453 to 0.367 and Chinchilla-70B's $w$ from 1.54 to about 1.0. A robust estimator that does not require dropping runs — or an explicit model of the failure as $D/N\to0$ — would be preferable to a deletion rule, especially since the revealed-preference "test" at Chinchilla-70B depends on it.
28. **p. 31.** The "revealed-preference test" at Chinchilla-70B has a 95 percent interval of [0.82, 1.42] (pairs) or [0.54, 2.01] (clusters). It cannot distinguish $w=1$ from Hoffmann et al.'s 0.62–0.71 under clustering. It is a consistency check without power; say so.
29. **p. 33.** Farseer's $\sigma^*_\kappa$ standard error of 0.003 from 404 grid runs with shared hyperparameters and data order is implausibly small (Comment 6(b)).
30. **p. 34, capital–labor comparison.** Capital and labor are traded inputs with market prices, so the comparison's economic content is limited. State what an economist learns from $\sigma_{ND}\approx\sigma_{KL}$.
31. **pp. 35–36, neutrality.** With corpus-specific $E_r$, the reducible-loss index $-\ln(L-E_r)$ differs by corpus, so "Hicks-neutral in reducible loss" is a cardinal notion. In the ordinal sense, both the asymptote shift and the neutral scale leave the isoquant map and the MRTS unchanged; only the tilt $\ln(A_r/B_r)$ matters for allocation and for $w$. Reframe the neutrality results around the tilt.
32. **p. 36.** DataDecide inference uses 21,888 intermediate checkpoints from 1,100 runs and 100 bootstrap draws. The effective sample size is far smaller than the checkpoint count. The LR of 145 on 24 degrees of freedom should be calibrated by the (size, seed) bootstrap, and I would report bootstrap p-values with more than 100 draws.
33. **p. 37, Step Law stochastic frontier.** The hyperparameter grid is designed, and the best configuration per cell is observed directly. A composed-error SFA is unnecessary and its distributional assumptions are unmotivated. The minimum over the grid, with a noise correction from replicates, measures the frontier directly.
34. **p. 38, equation (9).** Assumes a constant scale elasticity $\theta$ of the omitted component. Embedding parameters scale with width, roughly as $N^{1/3}$ to $N^{1/2}$ depending on the depth–width policy. Report sensitivity to $\theta$.

**Section V (wedge)**

35. **p. 41.** "Three properties make the inversion usable across labs." The third ("every technology in this family ranks models by $M$" at given compute) is true within the family. Across technologies with different $a$, ranks of models at *different* compute can change. The Spearman correlations of 0.95–1.00 are across the six included technologies only (Comment 4(b)).
36. **p. 45, stated intent.** Five hand-picked cases cannot validate a measurement. Frame this as an illustration.
37. **p. 46, trends.** $M^*(C,t)$ moves with technical change (Proposition 5). Data-augmenting progress understates the trend in $w$ and parameter-augmenting progress (architecture) overstates it. Use vintage-specific technologies where available (the Llama 3 law for 2024 models, for example) and report the trend under them.
38. **p. 47, over-identification test.** A homoskedastic $F$ on $\ln\hat T$ for $\hat T>0$ only selects on the outcome. Use all models with a censored specification, and cluster by family.
39. **p. 47, validation.** Downloads are not tokens, OpenRouter lists 21 of 164 models, and the regression selects on leaderboard presence. The token-volume panel of Demirer et al. (2025), which the paper cites, would allow a real test of levels for the subset served there. Pursue it, or at least a pilot.
40. **p. 49, aggregate (equation 13).** Negative contributions from $w<1$ models are included and the result spans 0.67 to 3.26 across technologies. I would drop this from the main text.
41. **Table 6.** Rename the "95% CI" column "sweep-sampling interval" and move the technology band next to the point estimate. Readers will otherwise read [3.97, 8.12] as the uncertainty.

**Section VI (observational)**

42. **p. 51.** State the number of developer clusters and the number of families with at least two models in the 57-model sample in the text. Table 8's note gives 19 developers and 30 families.
43. **p. 52.** "Both worlds agree that $\theta_N>\theta_D$ … regressing on compute alone imposes $\theta_N=\theta_D$, that the data reject." This is the Hall-type bias of Corollary A5 and deserves a sentence connecting the two.
44. **p. 54.** "Measurement error in compute is not the problem: … the implied reliability of $\ln C$ is at least 0.988." For Sample B, $C=6ND$ from reported $N$ and $D$. Epoch's "independent" estimates are often themselves $6ND$. Clarify which comparisons are genuinely independent (hardware-time estimates).
45. **p. 54, export-control IV.** Report Anderson–Rubin intervals for any IV estimate shown, including in Table 8, where $F=0.9$.
46. **pp. 55–57, Ho et al.** The finding that the published point is not the converged minimizer is useful. For an economics journal, the ridge (Figure 6) and the dependence of $T_C$ on the neutrality restriction are the points worth emphasizing; the SciPy stopping-rule narrative can be condensed.

**Appendix A**

47. **p. 72, Lemma A1.** For two inputs, the Hicks, Allen and Morishima elasticities coincide. Note this, since IO readers are used to the distinction.
48. **pp. 73–75, Lemma A3(v).** The non-homothetic mix response to $\omega$ at a given loss target is a nice point. Under the target rule of Proposition 6 the chosen mix therefore co-moves with $\omega$ when $\alpha\neq\beta$, but only through compute. Conditional on $c$ the mix is still uninformative, as Proposition A7(ii) says. State this explicitly so the two results do not appear to conflict, and note that it is the conditioning on $c$ that kills the proxy.
49. **p. 81, remark after Proposition A3.** Remove the "corrects the statement" sentence, or cite the statement.
50. **p. 82, Proposition A5(v).** The dynamic-panel moments are useful. State in the main text that they are not applied because there are too few consecutive generations (six product-line cells, p. 54).
51. **p. 84, Proposition A8.** Add the case of an inference cost elasticity different from one in $N$ (Comment 5), and a family-level token budget (Comment 2).
52. **p. 88+, claims register (Table D-claims).** Useful for referees. Consider putting it in the replication package rather than the paper.

---

## 5. Summary of requested changes (priority order)

1. **Re-specify the inversion.** Model the alternatives explicitly (lab bears inference cost / adoption-maximizing open release / tier-constrained size classes / family-level $D$), derive what $w$ identifies under each, and test among them with instruments. Restrict "planned inference compute" to settings where it is defined (Comments 1–2).
2. **Engage the DLW critiques properly.** Run a genuine Raval-type over-identification test using a second margin or a second output. Bound cross-lab factor bias with cross-lab evidence, including published lab-specific laws. Write a paragraph on Bond et al. explaining why experimental elasticities help and what they cost (Comment 3).
3. **Report wedge results under all estimated technologies and lab-published laws,** with an ex-ante inclusion rule. Use the $\kappa$-free inner exponents as reference. Make $M^*(C)$ at frontier scale the explicit object of partial identification. Move cardinal numbers out of the abstract (Comment 4).
4. **Treat the FLOP cost as an approximation** and report sensitivity to MFU and inference-cost elasticities (Comment 5).
5. **Add the model-free estimators** $1/\sigma^*-1=L_{nn}/(2|dL^*/dc|)$ and $w=(s_c-s_x)/(s_c+s_x)$. Apply them to all IsoFLOP and factorial designs and to the new experiment, and use them to test the extrapolation behind Section V (Comment 7).
6. **Test homogeneity of $\sigma^*_\kappa$ across sweeps,** compare like objects on Farseer, and clarify which economic elasticity is estimated (Comment 6).
7. **Tighten the analogies and position the formal results** against Marschak–Andrews, Bond–Söderbom, GNR, the optimal-design literature and truncated regression (Comment 8).
8. **Adopt one design-conditional inference approach,** use weak-identification-robust inference where relevant, and reconcile Table 8 (Comment 9).
9. **Re-scope Section VI** and fix the allocative decomposition (Comment 10).
10. **For the experiment:** neutral evaluation set and a dose–response in quality; corner-cell learning-rate checks for both labs; within-trunk covariance modeling; a power calculation and a pre-analysis plan before results are examined (Comment 11).
11. **Focus the paper** (Comment 12).

I look forward to seeing a revised version. The underlying idea — that training sweeps give production-function econometrics a setting with a known cost side and designed input variation — is valuable. With a properly specified and tested inversion and a model-free treatment of curvature, this could be an important paper.
