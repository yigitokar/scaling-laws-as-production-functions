# Memo: module ra5_theory (Theory v2 for the revision)

Date: 2026-09-24. Author: ra5_theory agent, for the lead author and the writers of Sections I, II and IV.
**Revised after independent review** (`output/memos/ra5_theory_review.md`): the partial-identification anchors now use a
wild bootstrap-t (the first version's intervals under-covered), one check was added (SI.split) and one extended
(PI.sharp.num); numbers below are post-review.

**Scope.** The referee requests assigned to this module:
- Referee 1: comments 1, 2, 5, 7, 8 (with minors 9–12, 14, 15, 19, 47–49, 51);
- Referee 2: Majors 2 and 4c (with 3(ii), 5 and minor 9);
- Referee 3: M5, M8, M11.

**Deliverables.**
- `paper/sections/appendix_proofs.tex`: revised Online Appendix A. It compiles with `code/paper/test_section.sh appendix_proofs`.
- `paper/notes/theory_main_text_v2.md`: main-text statements, intuition, numbering map and labels.
- `code/analysis/ra5_theory/run.py`: 38 claim checks, **38 PASS, 0 FAIL**; about 110 seconds on at most 4 CPU
  processes; deterministic (two runs byte-identical).
- Tables: `output/tables/ra5_theory_*.csv/.tex`. Figures: `output/figures/ra5_theory_*.pdf/.png`.
- New references: `lit/bib/extra_ra5_theory.bib`.

---

## 1. Headline findings

1. **Generalized wedge (Prop. A8; main-text Prop. 2).**
   - Setup: a developer maximizes $V(L,N)-X_T-c_DD$. $V$ is any objective that falls in loss and may depend on size.
     $X_T=c_T(N)\,6ND$ is training expenditure, with $\delta=d\ln c_T/d\ln N$.
   - Result:
     $$w=\frac{1+\delta+m_N+\sum_k\xi_kg_{k,n}/X_T}{1+m_D+\sum_k\xi_kg_{k,d}/X_T},\qquad m_N=-(\partial V/\partial\ln N)_L/X_T.$$
     With $\delta=m_D=0$ and no binding constraint, **$w-1$ is the marginal value of compactness at fixed quality divided by
     training expenditure**, and the expenditure share is $s=(w-1)/w=m_N/(1+m_N)$.
   - Verification: 12 conduct/cost cases solved by brute-force value maximization. The general formula and every closed
     form match the technology's $\varepsilon_N/\varepsilon_D$ at the optimum to a maximum relative error of $1.9\times10^{-7}$
     (optimizer tolerance). The cases are:
     - lifetime compute;
     - internalized serving with $\lambda=0.6$, $p=3$, $\phi=0.85$, $\eta=-0.3$ and a direct size effect;
     - adoption;
     - a memory cap;
     - a latency SLO and a latency cost;
     - a data cost and a data cap;
     - distillation from a 70B teacher;
     - MFU with $\delta=0.12$ (two serving-price variants);
     - a data-parallel deadline.
   - Special cases (all verified):
     - (a) serving borne at rate $\lambda$: $w=1+\lambda p(\phi+\eta)T/(3D)$. With $\phi=1$ and $\eta=0$, $w-1=\lambda X_S/X_T$, an
       expenditure ratio that needs no FLOP-to-dollar conversion.
     - (a′) adoption-maximizing open release: $w-1$ is the adoption value of compactness, not a token count.
     - (b) memory or tier cap: $w=(1+\nu)(1+m_N)$. Any $w\ge1+m_N$ is rationalizable, so $w-1$ is only an upper bound.
     - (c) latency.
     - (d) data cost: $w=(1+m_N)/(1+m_D)$, with $\hat T/D=(T/D-3m_D)/(1+m_D)<T/D$ and $s=(X_S-X_D)/(X_T+X_S)$.
     - (e) distillation: $w=(1+T/3D)/(1+N_T/3N)$. **Referee 2's formula is exact.** It is a data cost with $m_D=N_T/(3N)$.
     - (f) MFU: $w=1+\delta+\lambda(1+\delta_S)pT/(3D)$. **Referee 1's product form $(1+\delta)(1+T/3D)$ holds only if the
       serving price per FLOP has the same size elasticity ($\delta_S=\delta$).**
   - Sign rule: a binding constraint raises $w$ iff $g_n>w_0g_d$.
     - Caps and latency objectives raise $w$.
     - Data caps, data costs, distillation, minimum-size floors and token-bound deadlines lower it.
     - Compute-bound deadlines pull $w$ toward one.
   - Misspecification: if the lab optimizes a different technology or output $L'$, then $\hat w=w\,w_L/w_{L'}$ (verified;
     factor bias gives $e^{-\chi}$).

2. **Family token budgets (Prop. A9).** With a common $D$:
   - member wedges are technological: $\ln w_i-\ln w_j=-a_1(n_i-n_j)$ exactly;
   - the family's condition in $D$ pins one number, the $\pi$-weighted average of the members' values of compactness:
     $\sum_i\pi_i(1+m_{N,i})=(1+m_D)\bar w_H$, where $\bar w_H$ (the compute-weighted harmonic mean of member wedges, which
     satisfies the identity $1-1/\bar w_H=\sum_i\omega_is_i$) is computed from the technology, not identified by the
     condition; under case (a), $\sum_i\pi_i\hat T_i=\lambda p\sum_i\pi_iT_i$;
   - member $\hat T_i$ are not revealed-preference objects.

   In a brute-force three-member example the member inversions are $\hat T_i/T_i=0.86, 0.91, 1.06$, while the $\pi$-weighted
   average is exact (1.000000). Binding caps turn the identity into an upper bound (ratio 1.95) and floors into a lower
   bound (0.48). A common $D$ under member-by-member choice is a knife-edge: raising one member's $T$ by 20 percent moves its
   $D$ by 4 percent.

3. **Model-free identification (Prop. A10; main-text Prop. 4).** For any smooth technology and any monotone output:
   - $w=(L_c-L_x)/(L_c+L_x)$ with $x=\ln M$ and $c=\ln C$;
   - at the IsoFLOP minimum, $1/\sigma^*-1=L_{nn}|_C/(2|dL^*/dc|)$;
   - $\partial\ln w/\partial\ln M|_C=1/\sigma^*-1$ at $M^*(C)$.

   Neither $E$ nor $\kappa$ nor a functional form is needed. Verification:
   - on a non-separable, non-quasi-homothetic technology, the formula matches isoquant-traced $\sigma^*$ (0.7601, 0.7573 and
     0.7548 at $10^{20}$, $10^{22}$ and $10^{24}$ FLOP) to $1.8\times10^{-8}$;
   - $w$ matches the MRTS to $1.7\times10^{-10}$;
   - the answer is the same under loss, bits, $\ln(L-E)$ with a wrong $E$, and $e^L$ (maximum deviation $5.5\times10^{-8}$).

   $\kappa=1$ pins the curvature: $-y_{nn}|_C=\gamma^2/(ab)$.

   Finite grids: a quadratic fitted to a nine-point Chinchilla profile at $10^{21}$ FLOP overstates the curvature:

   | Grid span in $N$ | Curvature bias | $\hat\sigma^*$ (quadratic) | $\hat\sigma^*$ (quartic) |
   |---|---|---|---|
   | 9× | +1.3% | 0.7345 | 0.7370 |
   | 100× | +5.9% | 0.7257 | 0.7372 |

   The true value is 0.7370. The leading-order bias formula is confirmed (ratio 1.0000 at small half-width).

4. **Identification and rates (Props. A1–A2; main-text Prop. 3). Correction of v1.**
   - On-path outcomes under $\kappa=1$ identify $(\alpha,\beta)$ **globally** (by functional form), but **not locally**. Because
     $A/B$ is unidentified, every direction that moves the two decay rates $\alpha a$ and $\beta b$ in opposite directions is
     flat to first order. Profiled criterion slopes are 4.000 along such directions and 2.000 along same-sign directions.
   - Every estimate has an exact observationally equivalent **twin**.
   - With $A\to0$, $\alpha$ is unidentified.
   - Consequently $\hat\sigma^*$ converges at $n^{-1/4}$, **whether or not $\alpha=\beta$** (shown by simulation for least
     squares; that no estimator can do better is a heuristic analogue of Chen's (1995) mixture bound, not a theorem here).
     Monte Carlo (40 on-path budgets,
     100 replications per noise level, 5 noise levels):
     - the 75th percentile of $|\hat\sigma^*-\sigma^*|$ scales as $\tau^{0.48}$ ($\alpha\neq\beta$) and $\tau^{0.49}$ ($\alpha=\beta$),
       where $n^{-1/4}$ predicts 0.5;
     - the 90th percentile scales as $\tau^{0.55}$ and $\tau^{0.56}$;
     - the 25th percentile scales as $\tau^{1.00}$, from the 40 percent of fits with coinciding rates;
     - imposing $\beta/(\alpha+\beta)=a$ gives $\tau^{0.99}$;
     - at $\tau=10^{-7}$ the 75th percentile is $5.0\times10^{-4}$ unrestricted against $4.9\times10^{-8}$ restricted.
   - v1's "σ* first-order identified iff α = β" is restricted, not simply withdrawn: it is correct when $A/B$ is known (or
     pinned by the intercept restriction $G^{\alpha+\beta}=\alpha A/\beta B$), where the only flat direction keeps the split fixed
     (check SI.split: along it $\sigma^*$ moves at first order if $\alpha\neq\beta$ and not at all if $\alpha=\beta$); it fails
     when $A/B$ is estimated from outcomes.
   - This also reconciles Kricheli et al.: the degeneracy occurs when the decay rates coincide along the design's ray. That
     happens on fixed-ratio rays iff $\alpha=\beta$, and always on the compute-optimal path.
   - Off-path rates are unchanged: $O(v^2)$ for $\ln M^*$ and $O(v^4)$ for $\sigma^*$. **New:** they are also unchanged with $E$
     estimated and noise on $\ln L$ (slopes 4.00 and 2.00; estimating $E$ costs no information about $S$ to two decimals).

5. **Partial identification at frontier scale (Prop. A11; main-text Prop. 5).**
   - Normal inputs give $|d\ln M^*/d\ln C|\le1$.
   - A bounded path slope gives a sharp cone. Sharpness: any continuation is generated by an admissible technology through
     scale-dependent factor augmentation (verified).
   - **For any technology with single-peaked IsoFLOP profiles, $\operatorname{sign}(w-1)=\operatorname{sign}(\ln M-\ln M^*(C))$**
     (verified with 0 errors at 336 points on a non-separable technology). So the sign of over-training is identified iff $M$
     lies outside the identified set for $M^*(C)$, with no $\sigma^*$ or functional form.
   - Illustration on the verified sample:
     - Anchors: Chinchilla ($M^*=20.5$ [11.4, 36.8] at $2.9\times10^{21}$) and Llama 3 ($22.3$ [12.2, 40.2] at $10^{22}$).
       Equal-tailed wild bootstrap-t, HC2-rescaled residuals and HC2 SEs, Webb weights, $B=9{,}999$ (simulated coverage about
       90 percent at these nine- and ten-point designs; the first version's raw-residual percentile intervals, [14.8, 28.5]
       and [16.4, 30.5], covered 76–83 percent).
     - With $e\in[-0.14,0.26]$, $M^*(10^{24})\in[5.0,168]$ and $[6.4,133]$.
     - Sign identified as $w>1$ for 115/149 (77%) and 111/137 (81%) models, as $w<1$ for 2 and 2, and not identified for 32
       and 24.
     - With normal inputs only: $w>1$ for 46/149 (31%) and 47/137 (34%).
     - Llama 3.1 405B's sign is not identified under any set assumption, and the two point extrapolations disagree.
     - Medians of the models' lower and upper bounds on $w$ with $\sigma^*\in[0.6,0.75]$: 1.58 and 16.1.
     - **This is an illustration of the logic in unharmonized units.** The paper's application is module ra2_wedge's
       harmonized PI analysis (`ra2_wedge_pi_*.csv`); Section IV should quote those numbers, not these.

6. **Other corrections.**
   - The DMR drift signs $\beta g_D-\alpha g_N$, not $g_D-g_N$. Counterexample verified by brute force: $(\alpha,\beta)=(0.30,0.45)$,
     $(g_N,g_D)=(0.5,0.4)$, $M^*$ drifts at $-0.080$.
   - The "corrects the statement" sentence is removed.
   - The v1 claim that "extrapolated $\hat T$ is conservative" is withdrawn (Referee 2, Major 3(ii)).
   - The Besiroglu standard errors on $A$ and $B$ are reframed as parameterization artifacts.
   - Flagship normalization is labeled a calibration assumption, not GNR.

## 2. Methods

**Symbolic checks.** `symbolic_v2.py`, 20 checks, sympy.
- Identities are reduced to zero or evaluated at 12–20 random rational points with 40-digit precision (residuals below
  $10^{-170}$).
- Statements about derivatives of an *arbitrary* smooth technology at a point (model-free $w$, $\sigma^*$ and local $\sigma$;
  invariance to transformations; the slope of $\ln w$) use a generic fourth-order Taylor polynomial with free coefficients.
  This is exact for any identity involving derivatives up to that order.

**Numeric checks.** `numeric_v2.py`, 17 checks. No check uses the closed form it tests.
- Developer problems are solved by Nelder–Mead plus BFGS maximization of the objective in $(\ln N,\ln D)$.
- Constraint multipliers come from central differences of the optimal value.
- $\sigma$ is computed by tracing isoquants.
- Model-free quantities use finite differences.
- On-path identification is checked by profiling the least-squares criterion over nuisance parameters (variable
  projection) along chosen directions.
- The rates come from a Monte Carlo. There are 40 on-path compute levels over four decades, and Gaussian noise of s.d.
  $\tau$ on $L$; smaller $\tau$ stands in for larger $n$, with $n_{\rm eff}\propto\tau^{-2}$. The parameter space is compact,
  with $A$ and $B$ at least 10 percent of their true on-path scale. Each fit uses a 13×13 grid and a two-pass Nelder–Mead.
  The error is taken as the worse of the two twins. The runs cover two technologies × 5 noise levels × 100 replications
  (seeds 101–105 and 202–206), on 4 processes.
- Fisher information is computed from finite-difference score matrices, with $E$ as a sixth parameter.

**Partial-identification illustration.** `partial_id.py`.
- Anchors are the IsoFLOP minima computed by m1 (`m1_chinchilla_a2_minima.csv`, `m1_chinchilla_labs_a2_minima.csv`).
- The path is fitted by OLS of $\ln M^*_j$ on $\ln C_j$, centered at the top budget.
- Inference is design-conditional: an equal-tailed wild bootstrap-t with HC2-rescaled residuals $\hat e_i/\sqrt{1-h_{ii}}$,
  HC2 standard errors recomputed in each draw, and Webb six-point weights; $B=9{,}999$, seed 2026. The anchor sits at the
  high-leverage end of the design ($h\approx0.40$). In simulations at these exact designs (normal errors, homoskedastic or
  rising toward the top budget; 1,500–2,000 replications, $B=999$) this interval covers 90–91 percent (nominal 95); the
  raw-residual percentile interval used before the review covered 76–83 percent, and the classical $t$ interval 95 percent
  under homoskedasticity but 87–88 percent under the heteroskedastic design.
- Models are the verified Sample B (`m3_wedge_models.csv`, `sample=="B"`, `core==True`, $n=173$), in each model's own
  conventions.
- The bounds on $w$ use $\sigma^*\in[0.5,0.8]$ (wide) and $[0.6,0.75]$ (narrow). Share bounds are truncated at 0, the lower
  limit when $T\ge0$.

**Assumed and estimated.**
- Assumed: the technology classes stated in each result; differentiability; optimization for the wedge results.
- Estimated: only the illustration's anchor paths. Everything else is exact algebra or simulation.

## 3. Inventory

| File | Content |
|---|---|
| `paper/sections/appendix_proofs.tex` | Revised Appendix A. v1 backup: `data/processed/ra5_theory/appendix_proofs_v1_backup.tex` |
| `paper/notes/theory_main_text_v2.md` | Main-text statements, intuition, numbering map, required labels, corrections list |
| `code/analysis/ra5_theory/run.py` | Orchestrates; writes tables and figures; PASS/FAIL register; exits 1 on any FAIL |
| `code/analysis/ra5_theory/{ra5common,symbolic_v2,numeric_v2,partial_id,figures_v2,claims_tex}.py` | Modules (m7 helpers imported, not edited) |
| `output/tables/ra5_theory_claims.csv/.tex` | Register: 38 checks with verdicts and metrics / paper-facing version (two floats: symbolic `tab:ra5-claims`, numerical `tab:ra5-claims-num`; one 38-row float overflowed the page) |
| `output/tables/ra5_theory_wedge_cases.csv/.tex` | Brute-force verification of 12 cases / **paper-ready table** "What the wedge identifies under alternative developer objectives" |
| `output/tables/ra5_theory_family.csv` | Family example: member $N$, $w$, $\pi$, $T$, $\hat T$ |
| `output/tables/ra5_theory_misspec.csv` | Misspecified-technology check |
| `output/tables/ra5_theory_modelfree.csv` | Model-free vs isoquant $\sigma^*$ by technology and budget |
| `output/tables/ra5_theory_finite_grid.csv/.tex` | Finite-grid bias of quadratic and quartic fits |
| `output/tables/ra5_theory_singular_geometry.csv` | Criterion slopes by direction; zero-weight and restricted cases |
| `output/tables/ra5_theory_singular_split.csv` | Review addition: criterion and $\sigma^*$ slopes with the split $A/B$ known (v1's statement holds here) |
| `output/tables/ra5_theory_singular_mc.csv`, `_mc_slopes.csv`, `ra5_theory_singular.tex` | Rate Monte Carlo |
| `output/tables/ra5_theory_info_Eunknown.csv` | Fisher information with $E$ estimated |
| `output/tables/ra5_theory_pi_anchors.csv`, `_pi_bounds.csv`, `_pi_sign.csv/.tex`, `_pi_models.csv` | Partial-identification illustration |
| `output/figures/ra5_theory_pi_cone.pdf/.png` | (a) identified cone for $\ln M^*(C)$ beyond the Chinchilla design, with verified-sample models; (b) shares with the sign identified, by assumption and anchor |
| `output/figures/ra5_theory_singular.pdf/.png` | (a) on-path criterion, quartic vs quadratic; (b) Monte Carlo rates, $n^{-1/4}$ vs $n^{-1/2}$ |
| `data/processed/ra5_theory/pi_anchor_paths.csv`, `run_summary.txt` | Anchor path fits; run summary |
| `lit/bib/extra_ra5_theory.bib` | 10 new verified references (DOIs re-checked against Crossref in review) |
| `output/memos/ra5_theory_review.md` | Independent review: replication, fixes, remaining concerns |

## 4. Claims for the paper

Each claim gives the evidence, then the caveat.

1. **The wedge measures the value of compactness, not tokens.** $w-1=m_N$ under any objective, and $s=(w-1)/w$.
   - Evidence: GW.foc, GW.num.
   - Caveat: the first-order conditions are necessary conditions at an interior optimum. The inversion assumes the developer
     optimizes $N$ at the margin, which (b) and (g) relax.
2. **Serving demand is one reading among several.**
   - $\lambda\phi X_S/X_T$ is the reading under developer-borne serving. $T$ needs $p$.
   - Tiers and latency make $w-1$ an upper bound.
   - Data costs and distillation make it a lower bound (for given serving).
   - MFU adds $\delta$.
   - Evidence: GW.a.sym, GW.d.sym, GW.e.sym, GW.f.sym, GW.sign(.num).
   - Caveat: signs are known, magnitudes need the multipliers or prices.
3. **Families on a common budget reveal only a family average.**
   - Evidence: FAM.sym, FAM.slope, FAM.num.
   - Caveat: (iii) requires sizes chosen at the margin; with a menu, nothing beyond the bound in (iv).
4. **$\sigma^*$ and $w$ are identified without $E$, $\kappa$ or a functional form, from IsoFLOP curvature and the frontier
   slope, or from local 2-D variation.**
   - Evidence: MF.* (symbolic and numeric, including a non-separable technology).
   - Caveat: the estimator must use local polynomials of adequate order (FG.num), and $w$ only inside the support.
5. **On-path data identify $\sigma^*$ only by functional form, and only at rate $n^{-1/4}$, whether or not $\alpha=\beta$;
   Wald intervals are invalid.**
   - Evidence: SI.twin, SI.geom, SI.mc.
   - Caveat: the $n^{-1/4}$ rate is established heuristically (a moment argument, Chen 1995) and by simulation for least
     squares, not by a formal limit theorem for this model. The Monte Carlo is noise-on-$L$ Gaussian with a compact parameter
     space. The failure at the boundary ($A\to0$) means unrestricted on-path fits can wander arbitrarily without bounds.
6. **Behavioral collinearity is Marschak–Andrews / Bond–Söderbom collinearity, and the rates are the design-theoretic
   property of curvature parameters (Box–Lucas, Kiefer–Wolfowitz).**
   - Evidence: Remark A-positioning.
   - Caveat: positioning, not a new theorem.
7. **Only the sign of over-training is identified at frontier scale, and only for models whose $M$ lies outside the
   identified set for $M^*(C)$: in this unharmonized illustration, 77–81% of verified-sample models beyond the designs under
   the cross-sweep slope range, 31–34% under normality alone.** Quote ra2_wedge's harmonized shares in the paper.
   - Evidence: PI.* and the illustration.
   - Caveats: units are not harmonized across tokenizers and parameter conventions; the slope range is borrowed from sweeps
     with different conventions; the anchor intervals are pointwise 95%; the counts are illustrative for the wedge module to
     redo with harmonized units.
8. **The drift of the compute-optimal ratio signs $\beta g_D-\alpha g_N$.**
   - Evidence: DMR.sign, DMR.num.
   - Caveat: methodological; not applied to vintage data.

## 5. Robustness, failures and what is fragile

**Failures during development (all fixed).**
- The first MC prototype, using unconstrained Levenberg–Marquardt, produced exploding $\hat\sigma^*$ (RMSE above 1). The
  cause is the boundary non-identification ($A\to0$ frees $\alpha$). This is now a stated result, and the Monte Carlo uses a
  compact space.
- A restricted-fit bug: the basis at $r_N=r_D$ wrongly included the derivative column. It inflated restricted errors about
  1,000-fold and was caught by the slope check.
- A sharpness construction that translated profiles along isocosts gave negative marginal products at extreme
  under-training. It was replaced by scale-dependent factor augmentation, which keeps both marginal products positive
  everywhere.
- A sign error in the $w$ bounds for $x<\inf\mathcal X^*$ was fixed. It affected 2–3 models per anchor.
- A PI.normal check that was trivially true was replaced by a real check.

**Fragile or conditional.**
- The rate claim (see §4.5 caveat). Review check: the flat quartic valley could in principle let optimizer tolerance
  masquerade as an $n^{-1/4}$ rate. Re-polishing the first 30 replications per cell (eight extra Nelder–Mead restarts with
  $10^{-14}$ tolerances) changed the 75th and 90th percentiles by less than 0.1 percent and lowered the criterion by at most
  $7\times10^{-7}$ of $n\tau^2$, so the rate is statistical, not numerical. The fitted slopes carry Monte Carlo error: on
  those 30 replications they are 0.57 and 0.50 (75th percentile) against 0.48 and 0.49 with 100.
- The illustration's levels depend on unharmonized units.
- The anchor path uses all 9–10 budgets and imposes a straight path in-sample. The residual s.d. of $\ln M^*$ around the fit
  is 0.30 (Chinchilla) and 0.27 (Llama 3), so single-budget argmins are noisy.
- The slope range $e\in[-0.14,0.26]$ comes from $\kappa=1$ allocation exponents across sweeps, excluding DataDecide. With
  DataDecide ($a=0.315$), $e_U=0.37$ would widen the sets.
- Model-free $w$ is ill-conditioned as $\varepsilon_D\to0$.

**Review fixes (see `ra5_theory_review.md`).**
- Anchor intervals: raw-residual percentile wild bootstrap replaced by a wild bootstrap-t with HC2 residuals and SEs. All
  partial-identification numbers changed (e.g., 85% to 77–81% of models with $w>1$ identified).
- v1's "iff α=β" is now stated as conditional on known $A/B$ (new check SI.split), not simply withdrawn.
- The rate statement is labeled heuristic in the proposition itself, not only in the proof.
- PI.sharp.num now also tests a decreasing continuation (data augmentation), as the claim covers both directions.
- The wedge-case table's last column now compares $\hat T$ with $\lambda pT$ consistently.
- Paper tables: the claims register is split into two floats that fit a page; the finite-grid argmin bias shows five decimals
  (no "−0.0000"); the rate table's note no longer says "$\ln L$ technology" and states that the rate is simulation-based.

**Checks that hold by construction.**
- GW.sign: sympy of a quotient derivative.
- PI.normal and PI.sign.sym: one-line algebra.
- They formalize the statements; the numerical checks carry the weight.

## 6. Referee comments addressed

**Referee 1**

| Comment | Response |
|---|---|
| **1 (a–d)** | Prop. A8 with a general objective $V(L,N)$. (a) Open weights: case (a′), where $w-1$ is the adoption value of compactness, not $T$. (b) $T(L,N)$: the $\eta$ term, plus direct revenue effects of $N$. (c) Profit maximization without conditioning on $(\bar L,T)$. (d) $T$ as a present value $q/(r+h)$. Three conduct models, their distinguishing predictions, and the conduct-testing literature: Remark A-conduct. Planned-inference interpretation: only under (a), and only as $\lambda\phi X_S/X_T$. *The empirical conduct tests are for the wedge module.* |
| **2 (a–b)** | Tier caps: case (b), with "any $w$ rationalizable" made explicit and $w-1$ an upper bound. Family-level $D$: Prop. A9. Member wedges are technological (slope $-\alpha$); the family condition identifies a $\pi$-weighted average; a common $D$ under member-level choice is a knife-edge. This confirms the referee's reading that common $D$ is evidence against the maintained conduct. |
| **5** | "FLOP-accounting approximation". MFU case (f), with the correction that the product form needs $\delta_S=\delta$. Serving-cost elasticity $\phi$. Data costs (d). $c_T$ as a shadow price. Heterogeneous cost shifters as Bond–Söderbom price variation (Remark A-positioning). *The sensitivity table itself is for the wedge module; the formulas are in Prop. A8.* |
| **7** | Prop. A10 with (i)–(vi): model-free $w$ and $\sigma^*$, invariance, the slope test of the family, local $\sigma$, what $\kappa=1$ imposes, and designs. Remark A-grid gives the finite-grid bias and the quartic fix. *Estimation on the IsoFLOP designs is for the technology module.* |
| **8a** | Prop. 3 recast as Marschak–Andrews / Bond–Söderbom / GNR; the ACF distinction is stated; escape routes are mapped. |
| **8b** | Box–Lucas and Kiefer–Wolfowitz cited. Scope paragraph on where it binds. |
| **8c** | Global vs local made explicit, with the inferential consequence (rate, twins, boundary, invalid Wald). The "iff" in (ii) is qualified ("within the model"). **v1's "σ* first-order identified iff α=β" is restricted to the case of known $A/B$ (SI.split).** |
| **8d** | IsoFLOP sweeps are assigned inputs, not price shocks (Remark A-positioning). |
| **8e** | The flagship normalization is a calibration, not GNR (Remark A-pi-param). |
| **8f** | The drift signs $\beta g_D-\alpha g_N$; counterexample; "corrects the statement" removed; stated as methodological. |
| **8j** | Goldberger (1981) and Hausman–Wise cited for the attenuation; the new part is stated. |
| **3d** (part) | Doraszelski–Jaumandreu and De Ridder–Grassi–Morzenti cited in Remark A-conduct. |
| **3b** (theory part) | Prop. A8(ii): $\hat w$ ratios across two outputs equal $w_L/w_{L'}$. This is the basis of a Raval-type test (not run here). |
| **4e** | Prop. A11 (theory) with an illustration in unharmonized units: partial identification of $M^*(C)$; which models' signs are identified under which assumptions. The harmonized application that answers the request empirically is ra2_wedge's PI table. |
| **6b, 6d** | Local $\sigma$ at $w=1$ via A10(ii)/(iv). "Gross complementarity is implied, not found" (remark after Cor. A1). |
| **Minors** | 9 quasi-homotheticity: Lemma A4(iv). 10 Approach 1: remark after Lemma A3. 11 U-shape content: remark after Cor. A1. 12 floors, deadlines and MFU: Prop. A8(iii)–(iv). 14 outcomes vs choices separated: Prop. A1(iii)(a). 15 $E$ estimated: Prop. A2(iv). 19 projection demoted to a remark. 47 Hicks = Allen = Morishima: Lemma A1(vi). 48 conditioning on $c$: Lemma A3(v). 49 phrase removed. 51 inference-cost elasticity and family budget: Props. A8–A9. |

**Referee 2**

| Comment | Response |
|---|---|
| **Major 2a** | The expenditure interpretation: $w-1=\lambda\phi X_S/X_T$, and $s$, need no $p$; $T=3D(w-1)/(\lambda\phi p)$. |
| **Major 2b** | $\lambda$ internalization, and the adoption case. |
| **Major 2c** | $N$-proportional internal compute (RL rollouts, synthetic data, evaluation) enters $X_S$. |
| **Major 4c** | The distillation formula is verified exactly (sympy and brute force), with the interpretation as a data cost. |
| **Major 3(ii)** | True vs believed technology: Prop. A8(ii). The "conservative" claim is withdrawn. |
| **Major 5** | (a) Family budgets: Prop. A9. (c) Tiers: (b). (d) Latency: (c). (e) Sunk compute: $c_T$ is a shadow price. (f) Output concept: A8(ii). (g) Flagship data constraints: (d′). |
| **Minor 9** | Sampling vs specification uncertainty separated (Remark A-pi-param). |

**Referee 3**

| Comment | Response |
|---|---|
| **M5** | The model section: $\lambda$ internalization, memory/latency constraints, data cost; what the inversion identifies in each case; distinguishing predictions. *The tests are empirical (wedge module).* |
| **M8** | Hao–Merrill framing (Prop. 2 inverts an HM-type problem with $\sigma>0$). Bond–Söderbom/ACF antecedents. Sargan and Rotnitzky et al. D-optimality/design literature. **Reconciliation with Kricheli et al.** (degenerate when decay rates coincide along the design ray). |
| **M11** | Scope of the identification theory stated. Notation: $E$ vs $\mathbb E$ made explicit; $a_1,b_1$ only in the appendix. Props. 5–6 of v1 moved to Online Appendix E. The numbering map is in `theory_main_text_v2.md`. |

## 7. Open issues (route to the lead author and writers)

1. **Main-text labels.** The appendix now `\ref`s `prop:duality`, `lemma:sigma`, `prop:wedge`, `prop:ident`,
   `prop:modelfree` and `prop:pi`. The writers must create `prop:ident` and `prop:modelfree`, and repurpose `prop:wedge` and
   `prop:pi`. Until then the full paper shows "??".
   - The old main-text labels `prop:fdep`, `prop:info`, `prop:dmr` and `prop:transmission` are no longer referenced by the
     appendix.
   - The main text currently references appendix labels that still exist, so nothing breaks there.
2. **The rate result** (Prop. A1(iii)(c)) rests on a heuristic moment argument plus Monte Carlo, not a formal limit theorem
   for this non-identified-nuisance model. Rotnitzky et al.'s theorem does not apply directly, because $A/B$ is unidentified
   at the truth. A formal proof (e.g., via the over-fitted mixture literature) is open. The appendix words it as "no faster
   than $n^{-1/4}$" (Chen's bound), with attainment shown by simulation.
3. **The illustration's units are not harmonized.** Anchors use MassiveText or Llama 3 tokens; models use their own tokenizers
   and total $N$. The wedge module should redo Prop. A11 with harmonized units, lab-published anchors, and the $\kappa$-free
   technologies.
4. **Not done here (other modules):**
   - estimation of model-free $\sigma^*$ and $w$ on the IsoFLOP and factorial designs (technology module);
   - the MFU and serving-cost sensitivity table, the conduct tests, the multi-tier bunching test, and the Raval-type two-output
     test (wedge module).
5. **`paper/references.bib` was rebuilt** with `code/paper/build_bib.py`. The rebuild picked up other revision modules'
   `extra_*.bib` entries as well; it is deterministic, and nothing was removed.
6. **The test build shows AER running heads** ("VOL. VOLUME NO. ISSUE"; "MONTH YEAR") from `main.tex`. Revision plan B asks for
   working-paper heads; this is for the front-matter writer.
7. **Result count.** The numbering map has six main-text results (Prop. 1, Lemma 1, Props. 2–5), while revision plan A
   says "≤ 5 formal results" and lists these six. The lead author should decide whether Lemma 1 becomes a remark.
8. **Two partial-identification exercises.** This module's Remark A-pi-illus and ra2_wedge's PI table use different anchors,
   intervals and slope ranges. The appendix remark is now labeled as an illustration; the main text must quote ra2_wedge.
9. **Cross-module (from the review).** ra2_wedge's IsoFLOP anchor intervals bootstrap only within-budget parabola
   residuals and ignore the scatter of the minima around the fitted path. For Llama 3 its interval is [20.8, 23.7] and
   this module's is [12.2, 40.2]. See `ra5_theory_review.md` §7.7.
10. **Small overfull boxes remain in Appendix A** (at most 6.9pt, three of them inherited from v1), and 1.9–3.7pt in two module
   tables.
