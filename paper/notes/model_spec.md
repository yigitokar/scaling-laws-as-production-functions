# Model specification (shared foundation for theory, Monte Carlo, estimation, and writing)

Status: working draft by the lead author, 2026-09-23. Every proposition below must be independently verified
(symbolically/numerically) before it enters the paper. Notation is fixed here; all modules must use it.

## 1. Primitives

Units: training runs i (a model), belonging to lab/family f, released at date t.
Inputs: parameters N, training tokens D (tokens processed, incl. repetitions), training compute C = 6ND
(log: c = ln 6 + n + d, lower case = logs). Output: validation cross-entropy L (nats/token) on a fixed corpus;
reducible loss R = L − E; **log output index y ≡ −ln(L − E)** (higher is better).

Technology (Chinchilla with Hicks-neutral and factor-augmenting productivity):

    L_i = E + exp(−ω_i) [ A (e^{ψ_N,i} N_i)^{−α} + B (e^{ψ_D,i} D_i)^{−β} ] · exp(ε_i)          (T)

  - ω: Hicks-neutral productivity (recipe/architecture quality that scales both terms). Known to the lab when it
    chooses inputs ("transmitted", Marschak–Andrews 1944).
  - ψ_N, ψ_D: parameter- and data-augmenting productivity (e.g., data quality raises ψ_D).
  - ε: seed/evaluation noise, not known at choice time (Zellner–Kmenta–Drèze 1966 disturbance).
  - F(n, d) ≡ −ln(A e^{−αn} + B e^{−βd}), so y = ω + F(n+ψ_N, d+ψ_D) − ε (up to the E-level approximation).

Derived objects (u = A e^{−α ñ}, v = B e^{−β d̃}, s = αu/(αu+βv)):
  - output elasticities ε_N = αu/R, ε_D = βv/R (of reducible loss);
  - elasticity of substitution σ = (αu+βv)/(αu(1+β)+βv(1+α)); 1/σ = 1 + sβ + (1−s)α;
  - expansion path under C = 6ND: a ≡ β/(α+β), b ≡ α/(α+β), G ≡ (αA/(βB))^{1/(α+β)};
  - frontier (inverse cost function): L*(C) = E + K (C/6)^{−γ}, γ = αβ/(α+β), K = ((α+β)/β) A G^{−α};
  - on-path elasticity of substitution σ* = 2/(2+α+β).

Lab objective (lifetime compute; Sardana et al. 2024): min_{N,D} 6ND + 2NT s.t. L(N,D) ≤ ℓ, possibly with a
data constraint D ≤ D̄ (Muennighoff et al. 2023) or memory constraint N ≤ N̄. T = expected lifetime inference tokens.

## 2. Results to establish (numbering is provisional)

**Lemma 1 (Allocation and productivity).** With T = 0, the optimum satisfies αu = βv (ε_N = ε_D). In logs,
  n* = ln G + a(c − ln 6) + a ψ_D − b ψ_N,   d* = −ln G + b(c − ln 6) + b ψ_N − a ψ_D,
  y* = ω + γ(c − ln 6 + ψ_N + ψ_D) − ln K.
Hicks-neutral ω does not enter the allocation; factor-augmenting productivity tilts it:
  d* − n* = −2 ln G + (b − a)(c − ln 6) + 2(b ψ_N − a ψ_D).
Better data (ψ_D ↑) ⇒ more parameters, fewer tokens per parameter.

**Proposition 1 (Functional dependence / ACF).** If all observed runs are cost-minimizing with T = 0 and productivity
is Hicks-neutral, then (n, d) are deterministic functions of c. The data identify at most (i) the path slope a and
(ii) E[y | c] = const + γ c + E[ω | c]. α, β, σ are not identified without functional-form restrictions; γ is not
identified unless E[ω | c] is linear-known (e.g., ω ⊥ c, as in a within-lab experiment). With ω ⊥ c and the
Chinchilla form (outer exponent 1): α = γ/a, β = γ/(1−a). With a Kaplan-type outer exponent κ free, any σ* ∈ (0,1)
rationalizes the same on-path data (DMR-type non-identification).

**Lemma 2 (Geometry).** ∇²_{(n,d)} ln R = [uv/R²]·[[α², −αβ], [−αβ, β²]]: rank one everywhere, null vector (β, α) =
the direction of the compute-optimal expansion path. ln R is exactly linear along (β, α)-lines; all curvature —
hence all information about σ — lies in the transverse coordinate αn − βd. Identification of σ requires
off-path (transverse) variation: IsoFLOP designs (experiments), or economically generated deviations:
heterogeneous inference demand T, data/memory constraints, factor-biased productivity, belief errors (Kaplan era).

**Proposition 2 (Transmission bias).** Cross-lab data with c = π₀ + π₁ω + η, π₁ > 0: the OLS slope of y on c
converges to γ + π₁ Var(ω)/(π₁² Var(ω) + Var(η)) > γ. Returns to compute are overstated and TFP dispersion
understated. Fixed effects (Mundlak 1961) fix time-invariant ω; Markov ω with predetermined compute (clusters
provisioned ahead) gives ACF-type moments E[ξ_ft | c_{f,t−1}, n_{f,t−1}, d_{f,t−1}] = 0 with ω_ft = h(ω_{f,t−1}) + ξ_ft.

**Proposition 3 (Selection).** If a run is released/reported only if y ≥ ȳ (or better than the lab's previous
model), E[ω | c, released] is decreasing in c, biasing the compute slope downward; sign of the net bias
(transmission + selection) is ambiguous. Dropping the worst-loss runs from a controlled sweep (Besiroglu et al.
drop 5 highest-loss points) is truncation on the dependent variable.

**Proposition 4 (Revealed inference demand = DLW-type wedge).** With T > 0 and no binding constraint,
  w ≡ ε_N/ε_D = 1 + T/(3D)  ⇒  T = 3D(w − 1).
w is the ratio of lifetime to training compute. w < 1 cannot arise from inference; it reveals a binding data
constraint (shadow cost on D) or belief/allocative error. Under Hicks-neutral heterogeneity w is invariant to ω;
under factor-augmenting heterogeneity the econometrician's ŵ (computed with common technology) equals
w · exp(αψ_N − βψ_D): factor bias is mistaken for inference demand (analog of the Raval 2023 / Demirer critique
of DLW markups). Farrell cost efficiency relative to the training-only frontier: CE = C_min(L)/(6ND).

**Proposition 5 (DMR for algorithmic progress).** With ψ_N,t = g_N t, ψ_D,t = g_D t and on-path data over time,
g_N + g_D (effective-compute growth) and b g_N − a g_D (allocation drift) are identified; σ and the Hicks bias
(1 − 1/σ)(g_N − g_D) are not, without the κ = 1 restriction or off-path data.

**Decomposition (allocative vs technical).** For a run i: ln(6N_iD_i) − ln C_min,t(L_i) = allocative inefficiency
(distance along the isoquant from the cost-minimizing mix, given the period's technology). Frontier shifts
(ω, ψ growth) are technical change. "Algorithmic progress" measured as the drift in the best loss at fixed
compute conflates the two (e.g., the Kaplan→Chinchilla reallocation is allocative, not technical).

## 3. Empirical objects to report everywhere
α, β, E, A, B (normalized at geometric means), a, γ, σ*, σ range over the sample, and bootstrap SEs;
w, T/D, CE for industry models; TFP (ω) dispersion (sd, 90/10 in compute-equivalent units = exp((ω90−ω10)/γ)).
