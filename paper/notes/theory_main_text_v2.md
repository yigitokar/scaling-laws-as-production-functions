# Theory v2: main-text statements, intuition and numbering map

Module ra5_theory, 2026-09-24 (revision after referee round 1). This note replaces `theory_main_text.md` for writers of
Sections I, II and IV. It gives the main-text statements, with intuition and no proofs. Proofs are in
`paper/sections/appendix_proofs.tex` (Online Appendix A).

Every new or changed claim is verified by `code/analysis/ra5_theory/run.py`: 38 checks, all PASS (sympy identities and
brute-force numerics that do not use the closed form under test). The register is `output/tables/ra5_theory_claims.csv`.
The unchanged v1 results keep their m7 verification (`code/analysis/m7_theory/run.py`, 61 checks).

Notation follows `paper/notes/model_spec.md`, with two exceptions. First, in the main text $E$ is irreducible loss and
$\mathbb E$ is expectation (the appendix does the same). Second, the exponents $a_1,b_1$ appear only in the appendix.

---

## 0. Numbering map and labels

Writers must use these `\label`s in the main text. The appendix already `\ref`s them; until they exist, the full paper
shows "??".

| v2 main text | Label | Content | Appendix | v1 main text |
|---|---|---|---|---|
| Proposition 1 | `prop:duality` | Duality: frontier, factor demands, primal | Lemma A3 (+ remark on Approach 1) | Prop. 1 |
| Lemma 1 | `lemma:sigma` | Interior compute optima require $0<\sigma<1$; $\sigma(w)$, $\sigma^*$, ordinality | Lemmas A1–A2, Cor. A1 | Lemma 1 |
| Proposition 2 | `prop:wedge` | **Generalized wedge** under any objective $V(L,N)$; special cases; expenditure share; family token budgets | Props. A8–A9, Lemmas A4–A5, Cor. A3 | Prop. 2 (rewritten) |
| Proposition 3 | `prop:ident` | **Identification and information rates** (v1 Props. 3+4 merged; global vs local; positioning) | Props. A1–A2, Lemma A4(iv) | Props. 3, 4 |
| Proposition 4 | `prop:modelfree` | **Model-free $\sigma^*$ and $w$** (new) | Prop. A10 (+ finite-grid remark) | — |
| Proposition 5 | `prop:pi` | **Partial identification of $M^*(C)$ and of $w$, $s$, $T$ beyond the design** (replaces v1 Prop. 7) | Prop. A11 (+ illustration and projection remarks) | Prop. 7 (replaced) |
| Online Appendix E | — | Technical change (DMR sign), transmission, selection, proxies, Farrell, Sahal, Hall | Props. A3–A7, Cors. A2, A4, A5 | Props. 5, 6 |

Appendix numbering: Lemmas A1–A5, Propositions A1–A8 and Corollaries A1–A5 keep their v1 numbers (their content is
revised where noted). A9 (family budgets) and A10 (model-free) are new. v1's A9 (partial identification of $T$) is now
A11, with new content. All v1 labels are kept (`lem:sigma`, `lem:soc`, `cor:complements`, `lem:alloc`, `lem:geometry`,
`lem:ce`, `prop:fd`, `prop:A-info`, `prop:A-dmr`, `cor:ceg`, `prop:farrell`, `prop:A-transmission`, `prop:selection`,
`prop:proxy`, `prop:A-wedge`, `cor:suff`, `prop:A-pi`, `cor:sahal`, `cor:hall`). New labels: `prop:A-family`,
`prop:A-modelfree`, `eq:A-gw`, `eq:A-gw-obj`, `eq:A-mf`, `rem:A-positioning`, `rem:A-conduct`, `rem:A-grid`,
`rem:A-pi-illus`, `rem:A-pi-param`.

---

## Section I. The Training Problem

**Technology and cost (text, not a result).**
- $L=E+e^{-\omega}[A(e^{\psi_N}N)^{-\alpha}+B(e^{\psi_D}D)^{-\beta}]e^{\epsilon}$; the κ family raises the bracket to $\kappa$.
- Training compute $C\approx6ND$ is a **FLOP-accounting approximation**, not an identity (Referee 1, comment 5). It ignores
  attention FLOPs, and FLOPs are not dollars. Proposition 2 carries size-dependent prices of a FLOP ($\delta$) and
  data costs explicitly.
- **Quasi-homotheticity** (Referee 1, minor 9; Lemma A4(iv)). The family satisfies
  $R(\lambda^{b_1}N,\lambda^{a_1}D)=\lambda^{-\kappa a_1b_1}R(N,D)$. So every isoquant is a dilation of every other along the
  direction of the expansion path. The wedge is constant along the dilation orbits. Factor demands are power laws in
  compute. All curvature is transverse to the path. This is the content of the rank-one Hessian.
- *Suggested sentence:* "The family is quasi-homothetic with respect to the non-uniform scaling
  $(N,D)\mapsto(\lambda^{b_1}N,\lambda^{a_1}D)$. Every isoquant is a rescaled copy of every other, so the optimal mix drifts
  with compute as a power law and all curvature lies across the expansion path."

**Proposition 1 (Duality).** Unchanged in content. Shorten it to about half a page.
- Rename the paper's "Approach 1" as the **IsoFLOP-minima frontier**.
- Say that Hoffmann et al.'s Approach 1 reports allocation exponents from the envelope, so it estimates the factor
  demands as well as the frontier (Referee 1, minor 10; Referee 2, Major 9a).
- IsoFLOP argmins are outcomes of *assigned* inputs, not behavioral choices (Referee 2, Major 9d).

**Lemma 1 (Interior compute optima require $0<\sigma<1$).**
- Statement: unchanged. $\sigma=P/(P+Q)$ in log coordinates. Under $C=6ND$, a point with $\varepsilon_N=\varepsilon_D$ is a
  strict local cost minimum iff $0<\sigma<1$ there.
- *Empirical content, to be stated with the lemma (Referee 1, comment 6d and minor 11).* The lemma's observable implication
  is that IsoFLOP profiles are U-shaped in $\ln N$, with an interior minimum. $\sigma^*<1$ is implied by that fact and is
  imposed by every functional form the paper fits. It is **not** a finding. Proposition 4 turns the curvature of the U into
  the *level* of $\sigma^*$.
- For two inputs, the Hicks, Allen–Uzawa and Morishima elasticities coincide (Referee 1, minor 47; Lemma A1(vi)).

**Proposition 2 (What over-training reveals: the generalized wedge).**

A developer chooses $(N,D)$ to maximize $V(L(N,D),N)-X_T-c_DD$. Here:
- $V$ is any objective that falls with loss and may depend on size directly;
- $X_T=c_T(N)\,6ND$ is training expenditure at the (shadow) price of a FLOP, with size elasticity $\delta$;
- $c_D$ is a cost per token.

Let $m_N\equiv-(\partial V/\partial\ln N)_L/X_T$ be the value of compactness at fixed quality per unit of training
expenditure. Let $m_D\equiv c_DD/X_T$.

(i) At an interior optimum,
$$w\equiv\frac{\varepsilon_N}{\varepsilon_D}=\frac{1+\delta+m_N}{1+m_D}\qquad\text{(plus multiplier terms for binding constraints).}$$
With $\delta=m_D=0$, **$w-1$ is the marginal value of compactness at fixed quality divided by training expenditure**. The
expenditure share $s\equiv(w-1)/w$ is then $m_N/(1+m_N)$.

(ii) Special cases (the main-text table; paper-ready version in `output/tables/ra5_theory_wedge_cases.tex`):

| Developer objective or cost structure | $w$ | $s=(w-1)/w$ measures | $\hat T=3D(\hat w-1)$ vs $\lambda pT$ (internalized demand) |
|---|---|---|---|
| Lifetime compute, developer serves ($\lambda=p=1$) | $1+T/(3D)$ | serving share of lifetime compute | equal |
| (a) Serving cost borne at rate $\lambda$, price ratio $p$, size elasticity of serving cost $\phi$ | $1+\lambda\phi pT/(3D)$ | $\lambda\phi X_S/(X_T+\lambda\phi X_S)$ | $\times\phi$ |
| (a′) Open release that values adoption; users bear serving | $1+v'A\lvert\varepsilon_{A,P}\rvert/X_T$ | adoption value of compactness | not $T$ |
| (b) Binding memory or tier cap | $(1+\nu)(1+m_N)$ | upper bound on serving share | overstates |
| (c) Binding latency objective | $(1+\nu_\tau)(1+m_N)$ | upper bound | overstates |
| (d) Data cost per token | $(1+m_N)/(1+m_D)$ | $(X_S-X_D)/(X_T+X_S)$ | understates |
| (d′) Binding data cap | $(1+m_N)/(1+m_D+\mu)$ | below serving share; $w<1$ possible | understates |
| (e) Logit distillation, teacher $N_T$ | $\dfrac{1+T/(3D)}{1+N_T/(3N)}$ | $(X_S-X_{\text{teach}})/(X_T+X_S)$ | understates |
| (f) Dollar cost $\propto N^{1+\delta}D$, serving $\propto N^{1+\delta_S}$ | $1+\delta+\lambda(1+\delta_S)pT/(3D)$ | mixes $\delta$ and serving | differs by $3D\delta+\lambda\delta_SpT$ |
| (g) Family token budget, sizes chosen | $\sum_i\pi_i(1+m_{N,i})=(1+m_D)\bar w_H$ | $1-1/\bar w_H=\sum_i\omega_is_i$ | only the $\pi$-average is identified |
| Lab's technology differs (factor bias, beliefs, own output) | $\hat w=w\,w_L/w_{L'}$ | contaminated | overstates iff $w_L>w_{L'}$ (factor bias: $\chi<0$) |

(iii) **Family token budgets** (Proposition A9).
- When sizes come from a menu and share a common $D$, member wedges are purely technological: $\ln w_i-\ln w_j=-\alpha(\ln N_i-\ln N_j)$.
  The only first-order condition is the family's condition in $D$.
- If sizes are chosen at the margin, the family's first-order condition in $D$ pins one number: the $\pi$-weighted average of
  the members' values of compactness, $\sum_i\pi_im_{N,i}=(1+m_D)\bar w_H-1$, with $\pi_i\propto\omega_i/w_i$ and $\bar w_H$ the
  compute-weighted harmonic mean of the member wedges (observed from the technology; it satisfies the identity
  $1-1/\bar w_H=\sum_i\omega_is_i$). Under case (a) this is $\sum_i\pi_i\hat T_i=\lambda p\sum_i\pi_iT_i$. Member-level
  $\hat T_i$ are **not** revealed-preference objects.
- If sizes come from a menu, the $D$ condition involves only the members' marginal values of data and says nothing about
  the $m_{N,i}$.
- A common $D$ under member-by-member choice is a knife-edge: it needs $(1+m_{N,i})/(1+m_{N,j})=(N_i/N_j)^{-\alpha}$ exactly.

*Intuition.*
- The developer stops adding parameters when their marginal value in loss equals their marginal cost. That cost is training
  cost plus whatever the developer loses when the model is larger at the same quality: serving cost, adoption,
  latency, memory tiers.
- Tokens cost only training (and data). The ratio of the two conditions puts the value of compactness in the numerator.
- Serving cost in FLOPs is the special case that gives the old $1+T/(3D)$.
- Without knowing the price of a serving FLOP, the wedge still measures an *expenditure ratio*: the internalized serving
  bill relative to the training bill (Referee 2, Major 2a).

*What changed vs v1.*
- v1's Proposition 2 is now case (a) with $\lambda=p=\phi=1$.
- Referee 1's MFU formula $w=(1+\delta)(1+T/3D)$ holds only if the serving price per FLOP has the same size elasticity as the
  training price ($\delta_S=\delta$). Otherwise the form is additive, $1+\delta+\lambda(1+\delta_S)pT/(3D)$.
- Referee 2's distillation formula is verified exactly.
- A data cost lowers $w$ for given $T$ (Referee 3, M5.3): $\hat T/D=(T/D-3m_D)/(1+m_D)$.

*Positioning (one paragraph).*
- Proposition 2 inverts the first-order conditions of a Hao–Merrill-type profit-maximizing developer (`hao2026theory`) with
  an imperfectly substitutable technology. Their Leontief limit rules out over-training.
- Cases (a), (a′) and (b) are three conduct models (Referee 1, comment 1). Their distinguishing predictions:
  - under (a), $w-1$ rises with the developer's own serving footprint;
  - under (a′), it rises with the user-cost sensitivity of adoption;
  - under (b), sizes bunch at tier boundaries and $w-1$ is unrelated to demand given the tier.
- These can be tested with the conduct-testing tools of `berry2014identification`, `backus2021common` and
  `duarte2024testing`.
- DLW parallel. $w$ is `raval2023testing`'s ratio statistic with FLOP-cost weights. Part (ii) of Prop. A8 contains the
  factor-bias critique (`demirer2020production`, `raval2023testing`) and the output-concept critique (`bond2020unpleasant`).
- The elasticities come from designed experiments rather than proxy estimation on the same choices. That avoids the
  circularity stressed by `doraszelski2021reexamining`, at the cost of external validity.
- As in `deridder2026hitchhiker`, levels are fragile and ranks are robust.

---

## Section II. What Optimizing Labs' Data Identify

**Proposition 3 (What optimizing labs' data identify, and how fast).** Runs share a technology in the κ family, and
productivity is Hicks-neutral.

(i) *On the path (all wedges equal to one).*
- Inputs are affine functions of compute. Choices reveal only the path ($a$ and $M^*(C)$).
- Outcomes reveal the frontier ($\gamma$) only if $\mathbb E[\omega\mid c]$ is known up to a constant, within the model; that is,
  without instruments for compute or panel restrictions (Referee 1, comment 8c).
- The curvature $\sigma^*$ is not identified. Every $\sigma^*\in(0,1)$ fits, through an observationally equivalent κ family.
  By Lemma 1, no $\sigma^*\ge1$ fits.

(ii) *Global versus local identification under $\kappa=1$* (Referee 1, comment 8c; Referee 3, M8).
- **Globally.** $\alpha=\gamma/a$ and $\beta=\gamma/b$ are identified, by functional form. $A/B$, and with it the technology's
  own optimal mix, is identified only by assuming the observed path is optimal.
- **Locally, identification fails.** The on-path model is an over-fitted two-rate exponential mixture. Every direction that
  moves the two decay rates $\alpha a$ and $\beta b$ in opposite directions is flat to first order, because the unidentified
  $A/B$ absorbs it. Along it the criterion is of fourth order.
- Every estimate has an exact observationally equivalent twin, $(\alpha,\beta)\mapsto(\beta b/a,\alpha a/b)$.
- At the boundary ($A\to0$ or $B\to0$) one exponent is not identified at all.
- **Inferential consequence.** $\hat\sigma^*$ converges at $n^{-1/4}$, **whether or not $\alpha=\beta$**: shown by
  simulation for least squares, with a heuristic argument that no estimator can do better (the analogue of the optimal rate
  for over-fitted mixtures, `chen1995optimal`; see `sargan1983identification`, `rotnitzky2000likelihood`). There is no
  formal limit theorem for this model. The distribution is non-normal (a mass of fits with equal rates converging at
  $n^{-1/2}$ and a mass with split rates), and Wald and delta-method intervals are invalid.
- **When v1's statement holds.** If $A/B$ is known (or pinned by imposing the intercept restriction
  $G^{\alpha+\beta}=\alpha A/\beta B$), the only flat direction keeps the split fixed, and $\sigma^*$ is first-order identified
  iff $\alpha=\beta$ (check SI.split). It is estimating $A/B$ from outcomes that breaks it.
- Imposing that the fitted allocation exponent equals the observed path slope, $\beta/(\alpha+\beta)=a$, restores $n^{-1/2}$.
  So on-path precision under $\kappa=1$ is the precision of functional form plus optimality.
- *Reconciliation with Kricheli et al. (2026)* (Referee 3, M8). The degeneracy arises whenever the two loss terms decay at the
  same rate along the design's ray, $\alpha a'=\beta b'$.
  - On a fixed tokens-per-parameter ray ($a'=\tfrac12$) this happens iff $\alpha=\beta$: their ill-conditioning grows as
    $|\alpha-\beta|\to0$.
  - On the compute-optimal path it happens always, because equal decay rates are what compute-optimality means. So, with
    $A/B$ estimated, $\sigma^*$ is not first-order identified there whether or not $\alpha=\beta$.
  - This restricts v1's "iff $\alpha=\beta$" to the case of known $A/B$.
  - Cite `kricheli2026tokens`.
- Monte Carlo (40 on-path budgets, 100 replications per noise level, `output/tables/ra5_theory_singular.tex`):
  - The 75th percentile of $|\hat\sigma^*-\sigma^*|$ scales as $\tau^{0.48}$ ($\alpha\neq\beta$) and $\tau^{0.49}$
    ($\alpha=\beta$) in the noise s.d. $\tau$; $n^{-1/4}$ predicts 0.5.
  - The 90th percentile scales as $\tau^{0.55}$ and $\tau^{0.56}$.
  - With the path restriction imposed, both scale as $\tau^{0.99}$.
  - At the smallest noise the restricted estimator is about $10^4$ times more precise.

(iii) *Off the path: rates.* With transverse deviations (log wedges $\ln w_i$):
- the information about $\ln M^*$ is proportional to the residual dispersion of the log wedges, $O(v^2)$;
- the information about $\sigma^*$ is bounded by the sum of squared Farrell allocative losses, $O(v^4)$;
- so $\mathrm{se}(\hat\sigma^*)\propto v^{-2}$ and $\mathrm{se}(\ln\hat M^*)\propto v^{-1}$.

The orders are unchanged when $E$ is estimated and the noise is on $\ln L$ (Referee 1, minor 15). The numerical slopes are
4.00 and 2.00, and estimating $E$ costs no information about $S$ to two decimals.

*Positioning (one paragraph; Referee 1, comment 8a–b; Referee 3, M8). The key sentences:*
- "Proposition 3 is the Marschak–Andrews point in its cleanest form. The FLOP accounting gives every lab the same log-cost
  weights, so without wedges every allocation lies on one ray."
- "Bond and Söderbom (2005) show that identification then requires relative-price variation, adjustment frictions or
  optimization errors. Here these are:
  - heterogeneous wedges (serving demand, data costs, FLOP prices; Proposition 2);
  - binding memory, latency or data constraints;
  - belief errors (the Kaplan era)."
- "Designed IsoFLOP sweeps are not price variation. The experimenter assigns inputs, and there is no behavioral response."
- "The rates are the design-theoretic property of estimating a curvature parameter (Box and Lucas 1959; Kiefer and Wolfowitz
  1959). What is specific here is that the design spread is generated by economic wedges and that curvature enters only
  through Farrell losses."
- "This is distinct from ACF functional dependence, which concerns the first stage of proxy estimators (Appendix
  Proposition A7(iii))."
- Cite: `marschak1944random`, `bond2005adjustment`, `gandhi2020identification`, `box1959design`, `kiefer1959optimum`,
  `ackerberg2015identification`.

*Scope (one sentence in Section II and one in the introduction; Referee 1, comment 8b; Referee 3, M11).*
- The on-path results bind for compute-optimal ladders and fixed tokens-per-parameter designs run by one lab (e.g.
  DataDecide), and for compute-optimal-era releases.
- Released models since 2023 lie far off the path. Their heterogeneous wedges are the variation that helps, but that
  variation is endogenous and confounded with factor bias.

**Proposition 4 (Model-free $\sigma^*$ and $w$; new; Referee 1, comment 7).** Let $x=\ln M$ and $c=\ln C$. Let $L$ be any
output measure that falls in quality (loss, bits per byte, $\ln(L-E)$ for any $E$). Assume a smooth technology with positive
marginal products.

(i) At any point,
$$w=\frac{L_c-L_x}{L_c+L_x}.$$
It needs only the slope of the output along the IsoFLOP line through the point and along the ray of constant $M$. No $E$,
$\kappa$ or functional form is needed.

(ii) At a compute-optimal point,
$$\frac1{\sigma^*}-1=\frac{L_{nn}|_C}{2\,|dL^*/dc|}.$$
This is the curvature of the IsoFLOP profile in $\ln N$ at its minimum, divided by twice the slope of the loss–compute
frontier. It is invariant to any monotone transformation of output at the argmin.

(iii) At $M^*(C)$, $\partial\ln w/\partial\ln M|_C=1/\sigma^*-1$. In the κ family, $\ln w$ is exactly linear in
$\ln(M/M^*(C))$ with this slope, which gives a test of the family.

(iv) What $\kappa=1$ does. It forces $-y_{nn}|_C=\gamma^2/(ab)$, so the IsoFLOP curvature is pinned by the frontier and path
slopes. With $\kappa$ free the curvature is free.

(v) Designs.
- $\sigma^*$: IsoFLOP profiles (curvature at the minimum) plus the frontier slope.
- $w$: local two-dimensional variation around the point (a local factorial), inside the design's support only.

*Intuition.* Along an IsoFLOP line, loss changes only through the mix. Its curvature at the minimum says how fast the
marginal rate of substitution turns as the mix moves. The frontier slope converts that curvature into a percentage of the
marginal product, because both are measured in the same output units. That is why $E$, the output transformation and
$\kappa$ all drop out.

*Finite grids (Remark A-grid; for the technology section).* A least-squares quadratic fitted to a profile sampled on a
symmetric grid overstates the curvature by $(f''''/24)\kappa_4(h)$ to leading order. For the Chinchilla technology at
$10^{21}$ FLOP:
- a grid spanning 9× in $N$ gives $\hat\sigma^*=0.734$;
- a grid spanning 100× gives 0.726;
- the truth is 0.737;
- a quartic fit gives 0.737 in both cases.

So keep quadratic grids within about 3× in $N$, or use fourth-order local polynomials (`czech2026problems`). Table:
`output/tables/ra5_theory_finite_grid.tex`.

**Numbers writers may quote from the verification.**
- The model-free formula matches isoquant-traced $\sigma^*$ to $2\times10^{-8}$, both for Chinchilla (0.7370) and for a
  non-separable, non-quasi-homothetic technology (0.755–0.760).
- The model-free $w$ matches the MRTS to $2\times10^{-10}$.

---

## Section IV (opening). Partial identification at frontier scale

**Proposition 5 (Partial identification of $M^*(C)$, and of $w$, $s$ and $T$, beyond the design; replaces v1 Prop. 7;
Referee 1, comment 4e).**

Let $x^*(c)=\ln M^*(C)$, with path slope $e=d\ln M^*/d\ln C=1-2a$. Let it be identified at the largest design budget $C_J$.
For a model with compute $C>C_J$ and ratio $M$:

(i) *Normal inputs* ($N^*$ and $D^*$ nondecreasing in $C$) imply $|e|\le1$. This gives a cone
$x^*(c)\in x^*(c_J)\pm(c-c_J)$.

(ii) *Bounded path slope*, $e\in[e_L,e_U]$, gives the sharp set
$\mathcal X^*(c)=[x^*(c_J)+e_L\Delta c,\;x^*(c_J)+e_U\Delta c]$. Its width grows linearly in the log extrapolation distance.
Monotonicity is $e_L\ge0$. With sampling error, take the anchor's confidence interval.

(iii) *Sign of the wedge.* For **any** technology with single-peaked IsoFLOP profiles (guaranteed by $0<\sigma<1$),
$\operatorname{sign}(w-1)=\operatorname{sign}(\ln M-\ln M^*(C))$. So:
- $w>1$ is identified iff $\ln M>\sup\mathcal X^*(c)$;
- $w<1$ is identified iff $\ln M<\inf\mathcal X^*(c)$;
- otherwise the sign is not identified.

No $\sigma^*$ or functional form is needed.

(iv) *Magnitudes* need a range for the slope of $\ln w$ along the isocost. In the family that slope is $1/\sigma^*-1$, and
$\ln w\in[k_L(x-\sup\mathcal X^*),\;k_U(x-\inf\mathcal X^*)]$ for over-trained models.

(v) *Shares and tokens.* $s\in[1-1/w_L,\,1-1/w_U]$. $\lambda pT/(3D)\in[w_L-1,w_U-1]$ under Proposition 2(a).

*Illustration of the logic (Remark A-pi-illus; `output/tables/ra5_theory_pi_sign.tex`; figure `ra5_theory_pi_cone`).*
**Do not quote these counts as the paper's result.** The empirical application with harmonized units, lab-own anchors
and the ex-ante technology set is module ra2_wedge (`output/tables/ra2_wedge_pi_*.csv`, `ra2_wedge_pi.tex`); Section IV
should report those numbers. The illustration below keeps each model's own conventions.

Anchors (numbers after the review's inference fix):
- Chinchilla IsoFLOP minima, nine budgets up to $2.9\times10^{21}$ FLOP. $M^*=20.5$ at the top budget (95% interval
  [11.4, 36.8]).
- Llama 3 minima, ten budgets up to $10^{22}$. $M^*=22.3$ at the top budget ([12.2, 40.2]).
- Both intervals are equal-tailed wild bootstrap-$t$ intervals (HC2-rescaled residuals and HC2 SEs, Webb weights,
  $B=9{,}999$). At these nine- and ten-point designs they cover about 90 percent in simulation; the first version's
  raw-residual percentile intervals ([14.8, 28.5] and [16.4, 30.5]) covered only 76–83 percent.

With $e\in[-0.14,0.26]$ (allocation exponents 0.37–0.57 across sweeps), $M^*(10^{24})\in[5.0,168]$ (Chinchilla anchor) or
$[6.4,133]$ (Llama 3).

Among verified-sample models beyond the anchor (149 beyond the Chinchilla anchor, 137 beyond Llama 3):

| Assumption | Anchor | $w>1$ identified | $w<1$ identified | Sign not identified |
|---|---|---|---|---|
| Cross-sweep slope range | Chinchilla | 115 (77%) | 2 | 32 (21%) |
| Cross-sweep slope range | Llama 3 | 111 (81%) | 2 | 24 (18%) |
| Normal inputs only | Chinchilla | 46 (31%) | 0 | 103 |
| Normal inputs only | Llama 3 | 47 (34%) | 0 | 90 |

- Llama 3.1 405B's sign is not identified under any set-valued assumption.
- Point extrapolation disagrees across the two anchors: $w>1$ from Chinchilla, $w<1$ from Llama 3.
- Magnitudes are wide. With $\sigma^*\in[0.6,0.75]$ and the cross-sweep range, the medians of the models' lower and upper
  bounds on $w$ are 1.58 and 16.1.
- Units are not harmonized; treat this as a demonstration of the bounds' logic.

*Suggested main-text sentence (fill the share from ra2_wedge's harmonized PI table):* "What the data identify at frontier
scale is a sign, not a level. For [share] of the open models beyond the largest designed budgets, $M$ exceeds every
compute-optimal ratio consistent with the stated assumptions. For most of the rest, including the largest flagships, even
the sign is not identified."

*Demoted v1 material (Remark A-pi-param).*
- The parametric projection with $\mathrm{Var}(\ln\hat w)$ exactly quadratic in $\ln M$ (Referee 1, minor 19).
- The sampling-error decomposition 0.143 vs 0.047. It is conditional on κ = 1 and one sweep; say so (Referee 2, minor 9).
- The in-support monotone bound (Prop. A11(vi)).
- Two new statements:
  - Flagship normalization $w_f=1$ is a calibration assumption, **not GNR** (Referee 1, comment 8e).
  - The v1 claim that "extrapolated $\hat T$ is conservative" is withdrawn. Revealed preference recovers the technology the
    lab *believed*, so correcting toward the true technology can move $\hat w$ away from the lab's first-order condition
    (Referee 2, Major 3(ii)).

---

## Results moved out of the main text (Online Appendix E; one remark each where used)

- **Technical change (Prop. A3, Cor. A2).**
  - The drift of the compute-optimal $D/N$ signs the Hicks bias $\mathcal B$. In the family, $\mathcal B=\beta g_D-\alpha g_N$
    (with $\kappa=1$). So the drift signs $\beta g_D-\alpha g_N$, **not** $g_D-g_N$ unless $\alpha=\beta$ (Referee 1,
    comment 8f).
  - Counterexample (verified by brute force): $(\alpha,\beta)=(0.30,0.45)$ and $(g_N,g_D)=(0.5,0.4)$ give $\mathcal B=0.03>0$.
    The compute-optimal $D/N$ falls at rate 0.08 per unit time, although parameters are augmented faster.
  - Within the family, $g_N$ and $g_D$ are identified separately, but by functional form.
  - The "corrects the statement" sentence is removed. The result is methodological and is not taken to vintage data.
- **Transmission, selection and proxies (Props. A5–A7).** The selection attenuation is Goldberger's (1981) truncated-regression
  result (`goldberger1981linear`, `hausman1977social`). The new part is its interaction with transmission.
- **Farrell decomposition (Prop. A4); Sahal and Hall corollaries (Cors. A4–A5).** Unchanged.

---

## Corrections relative to v1 (for writers)

1. **Proposition 3(iii) of v1**, "σ* is first-order identified iff α = β", holds only when $A/B$ is known (or pinned by the
   intercept restriction); it is wrong as a statement about estimation with $A/B$ free.
   - The Jacobian condition holds only at the true $A/B$, which on-path data do not identify (check SI.split verifies the
     known-split case).
   - With $A/B$ estimated, $\hat\sigma^*$ is $n^{-1/4}$ whether or not $\alpha=\beta$ (sympy and profile geometry; Monte Carlo
     slopes 0.48–0.56 in both cases).
   - The global identification of $(\alpha,\beta)$ also fails at the boundary of the parameter space: with $A\to0$, $\alpha$ is
     arbitrary.
2. **v1 Proposition 3(ii)'s "iff"** holds within the model only (not with instruments or panel structure).
3. **DMR remark:** "both signs are identified" is withdrawn (see above).
4. **v1 Proposition 7 remark:** "extrapolated $\hat T$ is conservative" is withdrawn. The magnitude remark now separates
   sampling error (given κ = 1) from specification uncertainty.
5. **v1 Proposition 2(iv)'s memory result** $w=(1+\nu)(1+T/3D)$ is unchanged. The data-constraint form is now written with a
   value-normalized multiplier: $w=(1+m_N)/(1+m_D+\mu)$, with $\mu$ per unit of training expenditure. v1 used
   $\mu=-\partial\ln C^*/\partial\ln\bar D$ in the cost-minimization form. The content is the same.
6. **MFU:** Referee 1's $(1+\delta)(1+T/3D)$ needs $\delta_S=\delta$.
7. **Remark on Besiroglu et al.'s standard errors for $A$ and $B$** (Referee 1, comment 9d): they are parameterization
   artifacts at $N=D=1$ and not evidence of on-path non-identification. The Chinchilla design has transverse variation.

## Citations (all in `paper/references.bib` after `code/paper/build_bib.py`)

- New, in `lit/bib/extra_ra5_theory.bib`, each verified via Crossref or the publisher page:
  - `box1959design`, `kiefer1959optimum`, `sargan1983identification`, `goldberger1981linear`, `chen1995optimal`;
  - `doraszelski2021reexamining` (CEPR DP16027), `deridder2026hitchhiker` (Econometrica 94(1)),
    `berry2014identification`, `backus2021common`, `duarte2024testing`.
- Existing: `marschak1944random`, `bond2005adjustment`, `gandhi2020identification`, `rotnitzky2000likelihood`,
  `ackerberg2015identification`, `raval2023testing`, `demirer2020production`, `bond2020unpleasant`, `hao2026theory`,
  `sardana2024beyond`, `hausman1977social`, `blackorby1989real`, `klump2007factor`, `czech2026problems`.
