"""Symbolic (sympy) verification of every algebraic claim in model_spec.md and SYNTHESIS.md (P1-P8),
plus the new results proposed for the theory section.  Each block ends with record(...).

Convention: a claim "passes" when sympy reduces (lhs - rhs) to 0 (or proves the stated sign), possibly
after substituting random positive rationals when full simplification is slow (then method='sympy+num').
"""
from __future__ import annotations

import random

import sympy as sp

from common import record

POS = dict(positive=True, real=True)
n, d, c, t = sp.symbols("n d c t", real=True)
A, B, al, be, E, T, S = sp.symbols("A B alpha beta E T S", **POS)
om, eps, psN, psD, kap = sp.symbols("omega epsilon psi_N psi_D kappa", real=True)
N, D = sp.symbols("N D", **POS)


def _zero(expr, subs_random=0, tol=1e-12):
    """True if expr simplifies to zero; optionally confirm at random positive points."""
    e = sp.simplify(expr)
    if e == 0:
        return True, "exact"
    if subs_random:
        rng = random.Random(1)
        worst = 0.0
        # sort symbols by name: iterating a set is hash-seed dependent, which made the reported residuals
        # (not the verdicts) differ between runs (reviewer fix, 2026-09-23)
        syms = sorted(expr.free_symbols, key=lambda s: s.name)
        for _ in range(subs_random):
            vals = {s: sp.Rational(rng.randint(5, 95), 100) for s in syms}
            worst = max(worst, abs(float(sp.N(expr.subs(vals), 30))))
        return worst < tol, f"max|err| at random points = {worst:.1e}"
    return False, f"residual {e}"


def run_all():
    u = A * sp.exp(-al * n)
    v = B * sp.exp(-be * d)
    R = u + v
    a_, b_ = be / (al + be), al / (al + be)
    gam = al * be / (al + be)
    G = (al * A / (be * B)) ** (1 / (al + be))
    K = (al + be) / be * A * G ** (-al)

    # ------------------------------------------------------------------ model_spec §1: (T) and y
    Lt = E + sp.exp(-om) * (A * (sp.exp(psN) * sp.exp(n)) ** (-al) + B * (sp.exp(psD) * sp.exp(d)) ** (-be)) * sp.exp(eps)
    y = -sp.log(Lt - E)
    F = lambda nn, dd: -sp.log(A * sp.exp(-al * nn) + B * sp.exp(-be * dd))
    ok, m = _zero(sp.expand_log(sp.simplify(y - (om + F(n + psN, d + psD) - eps)), force=True), subs_random=20)
    record("MS.T", "model_spec", "y = -ln(L-E) = omega + F(n+psi_N, d+psi_D) - epsilon holds exactly (no approximation) given E",
           "IMPRECISE", "sympy", ok, m,
           "Exact identity; the qualifier 'up to the E-level approximation' is unnecessary (it only matters if E is estimated or lab-specific).")

    # elasticities
    eN = sp.simplify(-sp.diff(sp.log(Lt - E), n))
    eN_target = al * A * sp.exp(-al * (n + psN)) / (A * sp.exp(-al * (n + psN)) + B * sp.exp(-be * (d + psD)))
    ok, m = _zero(eN - eN_target, subs_random=20)
    record("MS.EL", "model_spec", "eps_N = alpha u/(u+v), eps_D = beta v/(u+v) with u,v in effective units; free of omega, epsilon, E",
           "IMPRECISE", "sympy", ok, m,
           "Write alpha u/(u+v): with omega, epsilon != 0 the observed L-E equals e^{-omega+epsilon}(u+v), so 'alpha u/R' is correct only if R denotes u+v.")

    # ------------------------------------------------------------------ sigma (Hicks) and variants
    # Hicks two-input elasticity with output f = -R(N,D) in levels
    uN, vD = A * N ** (-al), B * D ** (-be)
    f = -(uN + vD)
    fN, fD = sp.diff(f, N), sp.diff(f, D)
    fNN, fDD, fND = sp.diff(f, N, 2), sp.diff(f, D, 2), sp.diff(f, N, D)
    sig_hicks = -fN * fD * (N * fN + D * fD) / (N * D * (fNN * fD ** 2 - 2 * fND * fN * fD + fDD * fN ** 2))
    sig_formula = (al * uN + be * vD) / (al * uN * (1 + be) + be * vD * (1 + al))
    ok1, m1 = _zero(sig_hicks - sig_formula, subs_random=20)
    s_ = al * uN / (al * uN + be * vD)
    ok2, m2 = _zero(1 / sig_formula - (1 + s_ * be + (1 - s_) * al), subs_random=20)
    record("MS.SIG", "model_spec", "sigma = (au+bv)/(au(1+beta)+bv(1+alpha)) (Hicks) and 1/sigma = 1 + s beta + (1-s) alpha",
           "VERIFIED", "sympy", ok1 and ok2, f"{m1}; {m2}")
    # sigma as a function of the wedge (new): u = w beta v/alpha
    w = sp.symbols("w", **POS)
    sig_w = sp.simplify(sig_formula.subs(uN, w * be * vD / al))
    ok, m = _zero(sig_w - (1 + w) / (1 + al + w * (1 + be)), subs_random=10)
    ok_s, _ = _zero(sig_w.subs(w, 1) - 2 / (2 + al + be))
    record("MS.SIGW", "new", "local sigma depends on (N,D) only through the wedge: sigma(w) = (1+w)/(1+alpha+w(1+beta)); sigma(1) = sigma* = 2/(2+alpha+beta)",
           "NEW", "sympy", ok and ok_s, m)
    # Invariance to monotone transforms of output: Hicks sigma of g(f) equals that of f (generic g)
    g = sp.Function("g")
    ff = g(f)
    gN, gD = sp.diff(ff, N), sp.diff(ff, D)
    sig_g = -gN * gD * (N * gN + D * gD) / (N * D * (sp.diff(ff, N, 2) * gD ** 2 - 2 * sp.diff(ff, N, D) * gN * gD + sp.diff(ff, D, 2) * gN ** 2))
    ok, m = _zero(sp.simplify(sig_g - sig_hicks))
    record("MS.SIGORD", "new", "sigma is ordinal: identical for any monotone transform g(-R) of output (loss, BPB, accuracy link)",
           "NEW", "sympy", ok, m)

    # General two-input sigma in log coordinates (used for the SOC lemma): sigma = P/(P+Q)
    Fg = sp.Function("F")(n, d)
    Fn, Fd = sp.diff(Fg, n), sp.diff(Fg, d)
    Fnn, Fdd, Fnd = sp.diff(Fg, n, 2), sp.diff(Fg, d, 2), sp.diff(Fg, n, d)
    # levels: f(N,D) = F(ln N, ln D)
    fN_ = Fn / N; fD_ = Fd / D
    fNN_ = (Fnn - Fn) / N ** 2; fDD_ = (Fdd - Fd) / D ** 2; fND_ = Fnd / (N * D)
    sig_gen = -fN_ * fD_ * (N * fN_ + D * fD_) / (N * D * (fNN_ * fD_ ** 2 - 2 * fND_ * fN_ * fD_ + fDD_ * fN_ ** 2))
    P = Fn * Fd * (Fn + Fd)
    Qc = -(Fnn * Fd ** 2 - 2 * Fnd * Fn * Fd + Fdd * Fn ** 2)
    ok, m = _zero(sp.simplify(sig_gen - P / (P + Qc)))
    # at the FOC Fn = Fd = e: Q = -e^2 (Fnn - 2Fnd + Fdd) = -e^2 * (second derivative of F along the isocost)
    e_ = sp.symbols("e", **POS)
    Fnn_s, Fdd_s, Fnd_s = sp.symbols("F_nn F_dd F_nd", real=True)
    Q_foc = -(Fnn_s * e_ ** 2 - 2 * Fnd_s * e_ ** 2 + Fdd_s * e_ ** 2)
    along = Fnn_s - 2 * Fnd_s + Fdd_s  # d^2/dt^2 F(n0 - t, d0 + t)
    ok2 = sp.simplify(Q_foc + e_ ** 2 * along) == 0
    record("SOC", "new", "General 2-input: sigma = P/(P+Q), P = F_nF_d(F_n+F_d) > 0, Q = -(F_nn F_d^2 - 2F_nd F_nF_d + F_dd F_n^2); "
           "at eps_N = eps_D the compute-optimum SOC (F concave along n+d=c) <=> Q > 0 <=> sigma < 1",
           "NEW", "sympy", ok and ok2, m,
           "With multiplicative cost 6ND an interior compute-optimal allocation exists only where sigma < 1: observed interior optima reveal gross complementarity.")

    # ------------------------------------------------------------------ expansion path, frontier (P1)
    cc = sp.symbols("cc", positive=True)  # cc = C/6
    nstar = sp.log(G) + a_ * sp.log(cc)
    dstar = -sp.log(G) + b_ * sp.log(cc)
    foc = (al * u - be * v).subs({n: nstar, d: dstar})
    ok1, m1 = _zero(sp.simplify(sp.expand_log(foc, force=True)), subs_random=15)
    ok2, m2 = _zero(sp.simplify(nstar + dstar - sp.log(cc)))
    Rstar = R.subs({n: nstar, d: dstar})
    ok3, m3 = _zero(Rstar - K * cc ** (-gam), subs_random=15)
    K_alt = (al + be) * (A / be) ** (be / (al + be)) * (B / al) ** (al / (al + be))
    ok4, m4 = _zero(K - K_alt, subs_random=15)
    record("P1.path", "model_spec/SYNTHESIS P1", "FOC alpha u = beta v; N* = G (C/6)^a, D* = G^-1 (C/6)^b; L*(C) = E + K (C/6)^-gamma; "
           "K = ((a+b)/b) A G^-alpha = (alpha+beta)(A/beta)^{a}(B/alpha)^{b}",
           "VERIFIED", "sympy", ok1 and ok2 and ok3 and ok4, "; ".join([m1, m3, m4]))
    Rs = sp.symbols("R_s", positive=True)
    Cstar = 6 * (K / Rs) ** (1 / gam)
    ok, m = _zero(sp.simplify(Rstar.subs(cc, Cstar / 6) - Rs), subs_random=10)
    record("P1.cost", "SYNTHESIS P1", "Cost function C*(R) = 6 [K/R]^{1/gamma} inverts the frontier; d ln C*/d ln R = -1/gamma",
           "VERIFIED", "sympy", ok, m)
    # on-path elasticities equal gamma; sigma* on path
    eN_path = sp.simplify((al * u / R).subs({n: nstar, d: dstar}))
    ok1, m1 = _zero(eN_path - gam, subs_random=10)
    sig_path = sp.simplify(((al * u + be * v) / (al * u * (1 + be) + be * v * (1 + al))).subs({n: nstar, d: dstar}))
    ok2, m2 = _zero(sig_path - 2 / (2 + al + be), subs_random=10)
    record("MS.SIGSTAR", "model_spec", "On the path eps_N = eps_D = gamma and sigma* = 2/(2+alpha+beta)", "VERIFIED", "sympy",
           ok1 and ok2, f"{m1}; {m2}")

    # ------------------------------------------------------------------ Lemma 1 (allocation with productivity)
    lc = sp.symbols("lc", real=True)  # lc = c - ln 6
    # effective inputs: n~ = n + psN, d~ = d + psD; cost in effective units: n~ + d~ = lc + psN + psD
    nt = sp.log(G) + a_ * (lc + psN + psD)
    dt_ = -sp.log(G) + b_ * (lc + psN + psD)
    n1, d1 = nt - psN, dt_ - psD
    ok1, _ = _zero(sp.expand(n1 - (sp.log(G) + a_ * lc + a_ * psD - b_ * psN)))
    ok2, _ = _zero(sp.expand(d1 - (-sp.log(G) + b_ * lc + b_ * psN - a_ * psD)))
    ok3, _ = _zero(sp.expand(n1 + d1 - lc))
    ystar = om - sp.log(K * sp.exp(-gam * (lc + psN + psD)))
    ok4, m4 = _zero(sp.expand_log(ystar - (om + gam * (lc + psN + psD) - sp.log(K)), force=True))
    ok5, _ = _zero(sp.expand((d1 - n1) - (-2 * sp.log(G) + (b_ - a_) * lc + 2 * (b_ * psN - a_ * psD))))
    chi = be * psD - al * psN
    ok6, _ = _zero(sp.simplify((n1 - sp.log(G) - a_ * lc) - chi / (al + be)))
    # signs (fixed compute)
    sgn = [sp.simplify(sp.diff(n1, psD) - a_) == 0, sp.simplify(sp.diff(d1 - n1, psD) + 2 * a_) == 0,
           sp.simplify(sp.diff(n1, om)) == 0, sp.simplify(sp.diff(d1, om)) == 0]
    record("L1", "model_spec Lemma 1", "n* = lnG + a(c-ln6) + a psi_D - b psi_N; d* = -lnG + b(c-ln6) + b psi_N - a psi_D; "
           "y* = omega + gamma(c-ln6+psi_N+psi_D) - lnK; d*-n* formula; omega absent from (n*,d*) at given c",
           "VERIFIED", "sympy", all([ok1, ok2, ok3, ok4, ok5, ok6] + sgn), "exact",
           "Equivalent form: n* = lnG + a(c-ln6) + chi/(alpha+beta), chi = beta psi_D - alpha psi_N (only the MRTS-shifting 'bias index' moves the mix).")
    record("L1.sign", "model_spec Lemma 1", "At fixed compute: dn*/dpsi_D = a > 0, dd*/dpsi_D = -a < 0, d ln M*/dpsi_D = -2a < 0",
           "VERIFIED", "sympy", all(sgn), "exact")
    # fixed loss target version: min c s.t. y >= ybar  => c - ln6 = (ybar - omega + lnK)/gamma - psN - psD
    yb = sp.symbols("ybar", real=True)
    lc_t = (yb - om + sp.log(K)) / gam - psN - psD
    n_t, d_t = n1.subs(lc, lc_t), d1.subs(lc, lc_t)
    okA = sp.simplify(sp.diff(n_t, psD)) == 0
    okB = sp.simplify(sp.diff(d_t, psD) + 1) == 0
    mix_om = sp.simplify(sp.diff(d_t - n_t, om))
    okC = sp.simplify(mix_om - (a_ - b_) / gam) == 0
    record("L1.target", "new (precision of Lemma 1)", "At a fixed LOSS TARGET: dn*/dpsi_D = 0, dd*/dpsi_D = -1 (effective data held fixed); "
           "omega shifts the mix, d(d*-n*)/d omega = (a-b)/gamma, unless alpha = beta",
           "IMPRECISE", "sympy", okA and okB and okC, "exact",
           "Lemma 1's signs are conditional on compute c. Hicks-neutral omega leaves the expansion path (mix as a function of c) unchanged, "
           "but not the mix at a given loss target when alpha != beta (non-homotheticity).")

    # Lemma A3(vi) (reviewer addition): with inference demand, at given compute the lifetime FOC
    #   ln w_true = tau* + (beta-alpha)(lc)/2 + (alpha+beta) m/2 + chi = ln(1 + T e^{-d}/3),  d = (lc + m)/2,
    # implicitly defines m = ln M(chi); claim: dm/dchi = -2/(alpha+beta+theta_T), theta_T = (T/3D)/(1+T/3D).
    m_s, Tt = sp.symbols("m T_t", positive=True)
    chi_v = sp.symbols("chi_v", real=True)
    tau_s = sp.symbols("tau_s", real=True)
    foc_T = tau_s + (be - al) * lc / 2 + (al + be) * m_s / 2 + chi_v - sp.log(1 + Tt * sp.exp(-(lc + m_s) / 2) / 3)
    dm_dchi = -sp.diff(foc_T, chi_v) / sp.diff(foc_T, m_s)
    TD = Tt * sp.exp(-(lc + m_s) / 2) / 3
    thT = TD / (1 + TD)
    okT = sp.simplify(dm_dchi - (-2 / (al + be + thT))) == 0
    record("L1.T.sym", "new (Lemma A3(vi))", "Lifetime objective, given compute: d ln M/d chi = -2/(alpha+beta+theta_T) < 0, theta_T = (T/3D)/(1+T/3D) "
           "(implicit differentiation of the FOC w = 1 + T/(3D))", "NEW", "sympy", okT, "exact")

    # ------------------------------------------------------------------ Lemma 2 (geometry)
    lnR = sp.log(R)
    H = sp.Matrix([[sp.diff(lnR, n, 2), sp.diff(lnR, n, d)], [sp.diff(lnR, n, d), sp.diff(lnR, d, 2)]])
    target = (u * v / R ** 2) * sp.Matrix([[al ** 2, -al * be], [-al * be, be ** 2]])
    okH = all(sp.simplify(H[i, j] - target[i, j]) == 0 for i in range(2) for j in range(2))
    okdet = sp.simplify(H.det()) == 0
    null = sp.simplify(H * sp.Matrix([be, al]))
    oknull = all(sp.simplify(x) == 0 for x in null)
    # linear along (beta, alpha) lines
    n0, d0 = sp.symbols("n0 d0", real=True)
    line = sp.simplify(sp.diff(lnR.subs({n: n0 + be * t, d: d0 + al * t}), t, 2))
    okline = sp.simplify(line) == 0
    # decomposition ln R = -(alpha n + beta d)/2 + g(tau), tau = alpha n - beta d
    tau = sp.symbols("tau", real=True)
    gfun = sp.log(A * sp.exp(-tau / 2) + B * sp.exp(tau / 2))
    okdec, mdec = _zero(sp.simplify(sp.expand_log(lnR - (-(al * n + be * d) / 2 + gfun.subs(tau, al * n - be * d)), force=True)), subs_random=10)
    g2 = sp.simplify(sp.diff(gfun, tau, 2))
    s_tau = A * sp.exp(-tau / 2) / (A * sp.exp(-tau / 2) + B * sp.exp(tau / 2))
    okg2 = sp.simplify(g2 - s_tau * (1 - s_tau)) == 0
    # ln w = tau* - tau
    lnw = sp.log(al * u / (be * v))
    tstar = sp.log(al * A / (be * B))
    okw, _ = _zero(sp.expand_log(lnw - (tstar - (al * n - be * d)), force=True))
    record("L2", "model_spec Lemma 2 / SYNTHESIS P2", "Hessian of ln R in (n,d) = (uv/R^2)[[a^2,-ab],[-ab,b^2]]: rank one, null vector (beta,alpha) = path direction; "
           "ln R linear along every (beta,alpha)-line",
           "VERIFIED", "sympy", okH and okdet and oknull and okline, "exact")
    record("L2.dec", "new", "Exact decomposition ln R = -(alpha n + beta d)/2 + g(tau), tau = alpha n - beta d, g'' = s_u(1-s_u); "
           "transverse deviation tau - tau* = -ln w (a run's transverse coordinate is minus its log wedge)",
           "NEW", "sympy", okdec and okg2 and okw, mdec)

    # ------------------------------------------------------------------ Prop 4 / P6: inference wedge
    cost = 6 * N * D + 2 * N * T
    th_N = sp.simplify(sp.diff(sp.log(cost), N) * N)
    th_D = sp.simplify(sp.diff(sp.log(cost), D) * D)
    okw1 = sp.simplify(th_N / th_D - (1 + T / (3 * D))) == 0
    okw2 = sp.simplify((6 * N * D + 2 * N * T) / (6 * N * D) - (1 + T / (3 * D))) == 0
    record("P4.foc", "model_spec Prop 4 / SYNTHESIS P6", "min 6ND + 2NT s.t. L <= l: eps_N/eps_D = (cost elasticity ratio) = 1 + T/(3D) = lifetime/training compute",
           "VERIFIED", "sympy", okw1 and okw2, "exact")
    # invariance: w computed from the (T) technology does not depend on omega, epsilon, E
    Lfull = E + sp.exp(-om) * (A * N ** (-al) + B * D ** (-be)) * sp.exp(eps)
    wfull = sp.simplify((sp.diff(sp.log(Lfull - E), N) * N) / (sp.diff(sp.log(Lfull - E), D) * D))
    okinv = all(sp.simplify(sp.diff(wfull, s)) == 0 for s in (om, eps, E))
    record("P4.inv", "model_spec Prop 4 / SYNTHESIS P6", "w = eps_N/eps_D is invariant to Hicks-neutral omega, to epsilon and to E",
           "VERIFIED", "sympy", okinv, "exact")
    # contamination
    w_true = al * A * sp.exp(-al * psN) * N ** (-al) / (be * B * sp.exp(-be * psD) * D ** (-be))
    w_hat = al * A * N ** (-al) / (be * B * D ** (-be))
    okc = sp.simplify(w_hat - w_true * sp.exp(al * psN - be * psD)) == 0
    record("P4.contam", "model_spec Prop 4", "Econometrician's w_hat (common technology) = w * exp(alpha psi_N - beta psi_D) = w e^{-chi}",
           "VERIFIED", "sympy", okc, "exact")

    # homothetic closed forms, alpha = beta = rho
    rho, M, Ms = sp.symbols("rho M M_s", **POS)
    w_hom = (A * N ** (-rho)) / (B * D ** (-rho))
    okh1 = sp.simplify(w_hom.subs(D, M * N) - (M / (B / A) ** (1 / rho)) ** rho) == 0
    # C/C_min with alpha = beta: cosh form
    X = sp.symbols("X", positive=True)  # X = ND = C/6
    Rh = A * (X / M) ** (-rho / 2) + B * (X * M) ** (-rho / 2)
    Kh = 2 * sp.sqrt(A * B)
    ratio = X / (Kh / Rh) ** (2 / rho)
    xx = sp.symbols("x", positive=True)  # x = M/M*
    ratio_x = ratio.subs(M, xx * (B / A) ** (1 / rho))
    cosh_form = sp.cosh(rho / 2 * sp.log(xx)) ** (2 / rho)
    okh2, mh2 = _zero(ratio_x - cosh_form, subs_random=12)
    record("HOM", "model_spec / SYNTHESIS P6", "alpha=beta=rho: w = (M/M*)^rho, M* = (B/A)^{1/rho}; T/D = 3[(M/M*)^rho - 1]; C/C_min = cosh(rho/2 ln(M/M*))^{2/rho}",
           "VERIFIED", "sympy", okh1 and okh2, mh2)

    # general (non-homothetic) wedge: w = (M/M*(C))^{(alpha+beta)/2} at the run's own compute
    Mst = sp.exp(dstar - nstar)
    nn_ = (sp.log(cc) - sp.log(M)) / 2
    dd_ = (sp.log(cc) + sp.log(M)) / 2
    w_gen = (al * u / (be * v)).subs({n: nn_, d: dd_})
    okg, mg = _zero(sp.simplify(sp.expand_log(sp.log(w_gen) - (al + be) / 2 * (sp.log(M) - sp.log(Mst)), force=True)), subs_random=12)
    record("W1", "new (generalizes HOM)", "Any Chinchilla (alpha != beta), and the whole kappa-free family: w = (M/M*(C))^{(alpha+beta)/2} = (M/M*(C))^{1/sigma* - 1}, "
           "M*(C) = compute-optimal D/N at the run's OWN compute; T/D = 3[(M/M*(C))^{1/sigma*-1} - 1]",
           "NEW", "sympy", okg, mg)
    # C/C_min as a function of w only
    Rn = R.subs({n: nn_, d: dd_})
    Cmin6 = (K / Rn) ** (1 / gam)
    ratio_g = cc / Cmin6
    w_expr = w_gen
    closed = ((al + be * w_expr) / (al + be)) ** (1 / gam) * w_expr ** (-1 / al)
    okr, mr = _zero(sp.log(ratio_g) - sp.log(closed), subs_random=15, tol=1e-10)
    record("W2", "new", "Farrell allocative loss depends on the wedge only: C/C_min = ((alpha+beta w)/(alpha+beta))^{1/gamma} w^{-1/alpha} "
           "(reduces to cosh(...)^{2/rho} when alpha = beta; free of A, B, E, omega)", "NEW", "sympy+num", okr, mr)
    lw = sp.symbols("l", real=True)
    # ln(C/C_min) written with the log already expanded (sympy's series of a symbolic power is unreliable)
    series = sp.series((1 / gam) * sp.log((al + be * sp.exp(lw)) / (al + be)) - lw / al, lw, 0, 3).removeO()
    okH3 = sp.simplify(series - lw ** 2 / (2 * (al + be))) == 0
    record("W3", "new", "Harberger triangle: ln(C/C_min) = (ln w)^2 / (2(alpha+beta)) + O((ln w)^3) = sigma*(ln w)^2/(4(1-sigma*)) + O(.)",
           "NEW", "sympy", okH3, "series exact to 2nd order")

    # ------------------------------------------------------------------ data / memory constraints (task 2d)
    mu, nu, eN_s, eD_s, lam = sp.symbols("mu nu e_N e_D lambda", positive=True)
    thD = 1 / (1 + T / (3 * D))
    # KKT for min ln(cost) + lam (ln R - ln Rbar) + mu (d - dbar):  n: 1 - lam eN = 0 ; d: thD - lam eD + mu = 0
    sol = sp.solve([1 - lam * eN_s, thD - lam * eD_s + mu], [lam, eD_s], dict=True)[0]
    w_dc = sp.simplify(eN_s / sol[eD_s])
    okd1 = sp.simplify(w_dc - 1 / (thD + mu)) == 0
    okd2 = sp.simplify((1 / (thD + mu)).subs(T, 0) - 1 / (1 + mu)) == 0
    # memory: n: 1 + nu - lam eN = 0 ; d: thD - lam eD = 0
    sol2 = sp.solve([1 + nu - lam * eN_s, thD - lam * eD_s], [lam, eD_s], dict=True)[0]
    w_mc = sp.simplify(eN_s / sol2[eD_s])
    okm = sp.simplify(w_mc - (1 + nu) * (1 + T / (3 * D))) == 0
    record("D.constr", "task 2(d) / model_spec Prop 4", "Binding data constraint D <= Dbar (multiplier mu >= 0): w = 1/(theta_D + mu) < 1 + T/(3D); "
           "with T=0, w = 1/(1+mu) < 1. Binding memory constraint N <= Nbar (nu): w = (1+nu)(1+T/(3D))",
           "VERIFIED", "sympy", okd1 and okd2 and okm, "exact",
           "mu = -d ln C*/d ln Dbar. Data scarcity makes T_hat = 3D(w-1) UNDERstate T; memory/latency constraints make it OVERstate T.")

    # ------------------------------------------------------------------ DMR / technical change (Prop 5, P3)
    gN, gD = sp.symbols("g_N g_D", real=True)
    At, Bt = A * sp.exp(-al * gN * t), B * sp.exp(-be * gD * t)
    Gt = (al * At / (be * Bt)) ** (1 / (al + be))
    Kt = (al + be) / be * At * Gt ** (-al)
    okK = sp.simplify(sp.diff(sp.expand_log(sp.log(Kt), force=True), t) + gam * (gN + gD)) == 0
    drift = sp.simplify(sp.diff(sp.expand_log(-2 * sp.log(Gt), force=True), t))  # d ln(D*/N*)/dt at fixed C
    okdrift = sp.simplify(drift - 2 * (al * gN - be * gD) / (al + be)) == 0
    okdrift2 = sp.simplify(drift - 2 * (b_ * gN - a_ * gD)) == 0
    mrts = sp.log((al * At * N ** (-al) / N) / (be * Bt * D ** (-be) / D))
    bias = sp.simplify(sp.diff(sp.expand_log(mrts, force=True), t))
    okbias = sp.simplify(bias - (be * gD - al * gN)) == 0
    sig_ces = 1 / (1 + rho)
    okces = sp.simplify((rho * gD - rho * gN) - (1 - 1 / sig_ces) * (gN - gD)) == 0
    record("P5.rates", "model_spec Prop 5 / SYNTHESIS P3", "K_t falls at gamma(g_N+g_D); ln(D*/N*) drifts at 2(alpha g_N - beta g_D)/(alpha+beta) = 2(b g_N - a g_D); "
           "Hicks bias d ln MRTS/dt = beta g_D - alpha g_N (= (1-1/sigma)(g_N-g_D) only if alpha=beta); neutral <=> alpha g_N = beta g_D",
           "VERIFIED", "sympy", okK and okdrift and okdrift2 and okbias and okces, "exact",
           "The formula (1-1/sigma)(g_N-g_D) is the CES special case; in general the bias is beta g_D - alpha g_N.")
    # identification: with kappa free, frontier drift = gamma (g_N+g_D) and path drift = a g_D - b g_N are functions of (a,gamma,g) only
    a1s, b1s = (1 - sp.Symbol("a")) * S, sp.Symbol("a") * S
    aa = sp.Symbol("a", positive=True)
    gg = sp.Symbol("gamma", positive=True)
    a1s, b1s = (1 - aa) * S, aa * S
    kap_S = gg / (aa * (1 - aa) * S)
    frontier_drift = kap_S * (a1s * b1s / S) * (gN + gD)       # kappa * gamma1 * (g_N + g_D)
    path_drift = (-a1s * gN + b1s * gD) / S                     # d ln G_t/dt
    okS1 = sp.simplify(sp.diff(frontier_drift, S)) == 0
    okS2 = sp.simplify(sp.diff(path_drift, S)) == 0
    hicks_S = sp.simplify(b1s * gD - a1s * gN)
    okS3 = sp.simplify(hicks_S - S * (aa * gD - (1 - aa) * gN)) == 0
    record("P5.ident", "model_spec Prop 5 / SYNTHESIS P3", "kappa-free family: on-path drifts gamma(g_N+g_D) and a g_D - b g_N do not depend on S (sigma*) => g_N, g_D identified "
           "individually (a is identified); Hicks bias = S (a g_D - b g_N): magnitude not identified, SIGN identified because optimality forces S > 0",
           "IMPRECISE", "sympy", okS1 and okS2 and okS3, "exact",
           "Correction: model_spec says only g_N+g_D and b g_N - a g_D are identified (true, but these pin g_N and g_D separately), and that the Hicks bias is "
           "not identified; its magnitude is not, but its sign is (sign = -sign of the drift of ln D*/N*), because interior compute optima require sigma < 1 (SOC).")

    # ------------------------------------------------------------------ CEG (P4 of SYNTHESIS)
    x_ = sp.symbols("x", positive=True)  # x = C/6
    Eo, En, Ko, Kn, go, gn = sp.symbols("E_o E_n K_o K_n g_o g_n", positive=True)
    Lnew = En + Kn * x_ ** (-gn)
    Cold = (Ko / (Lnew - Eo)) ** (1 / go)
    lnf = sp.log(Cold) - sp.log(x_)
    dlnf = sp.simplify(sp.diff(lnf, x_) * x_)
    ok_const = sp.simplify(dlnf.subs({En: Eo, gn: go})) == 0
    # counterexample of 'only if (alpha, beta) equal': (0.3, 0.4) and (0.4, 0.3) share gamma
    g1 = sp.Rational(3, 10) * sp.Rational(4, 10) / sp.Rational(7, 10)
    ok_cex = g1 == sp.Rational(4, 10) * sp.Rational(3, 10) / sp.Rational(7, 10)
    record("P4S.CEG", "SYNTHESIS P4", "Frontier CEG f(C) = C_old/C_new at equal loss is constant in C iff E_old = E_new AND gamma_old = gamma_new "
           "(NOT iff alpha, beta, E equal); then f = (K_old/K_new)^{1/gamma}; factor augmentation gives f = e^{psi_N+psi_D}",
           "FALSE", "sympy", ok_const and ok_cex, "counterexample (0.3,0.4) vs (0.4,0.3): same gamma = 12/70",
           "Original 'iff (alpha,beta,E) equal' is false for frontier comparisons: only the harmonic exponent gamma and E matter. "
           "Equal (alpha,beta) is needed only if CEG is computed at a fixed common allocation rule (e.g., 20 tokens/param).")

    # ------------------------------------------------------------------ Sahal (P7) and Hall (P8)
    gC, gA, gam_s = sp.symbols("g g_A gamma_s", positive=True)
    m_rate = gam_s * (gC + gA)
    sA = gA / (gC + gA)
    ok7 = sp.simplify(m_rate / gC - gam_s / (1 - sA)) == 0
    record("P7", "SYNTHESIS P7", "Sahal: naive frontier-release regression slope = m/g = gamma/(1 - s_A), s_A = g_A/(g+g_A)",
           "VERIFIED", "sympy", ok7, "exact",
           "Holds with deterministic trends; with compute noise around trend the slope is gamma[1 + (g_A/g) R^2_{c,t}] (verified by MC).")
    eNs, eDs, dn_, dd_s, dom = sp.symbols("e_N e_D dn dd domega", real=True)
    dy = eNs * dn_ + eDs * dd_s + dom
    om_hat = dy - (eNs + eDs) / 2 * (dn_ + dd_s)
    ok8 = sp.simplify(om_hat - dom - (eNs - eDs) * (dn_ - dd_s) / 2) == 0
    record("P8", "SYNTHESIS P8", "Hall-type bias: residual with equal (cost-share) weights (eps_N+eps_D)/2 each = true + (eps_N-eps_D)(dn-dd)/2 (first order)",
           "VERIFIED", "sympy", ok8, "exact to first order",
           "Requires equal weights that sum to the true scale elasticity; with weights gamma each the bias adds (eps_N-gamma)dn + (eps_D-gamma)dd.")

    # ------------------------------------------------------------------ transmission bias covariance algebra (Prop 2 / P5)
    Vw, Ve, Vh, p1, gm = sp.symbols("V_omega V_eps V_eta pi_1 gamma_", positive=True)
    p1r = sp.symbols("pi1", real=True)
    Vc = p1r ** 2 * Vw + Vh
    Cov_cy = gm * Vc + p1r * Vw
    Vy = gm ** 2 * Vc + 2 * gm * p1r * Vw + Vw + Ve
    b_f = Cov_cy / Vc
    ok1 = sp.simplify(b_f - (gm + p1r * Vw / (p1r ** 2 * Vw + Vh))) == 0
    b_target = sp.simplify(b_f.subs(p1r, -1 / gm))
    ok2 = sp.simplify(b_target - gm * Vh / (Vw / gm ** 2 + Vh)) == 0
    g_r = Vy / Cov_cy
    ok3 = sp.simplify((g_r - gm) * Cov_cy - ((1 + gm * p1r) * Vw + Ve)) == 0
    record("P2.plim", "model_spec Prop 2 / SYNTHESIS P5", "OLS of y on c: plim = gamma + pi1 V_w/(pi1^2 V_w + V_eta) (any sign of pi1); target rule pi1 = -1/gamma "
           "gives gamma V_eta/(V_w/gamma^2 + V_eta) in [0, gamma); exogenous gives gamma; funding (pi1>0) overstates",
           "VERIFIED", "sympy", ok1 and ok2, "exact",
           "model_spec states only the pi1 > 0 case; the sign of the bias equals sign(pi1).")
    record("P2.bounds", "SYNTHESIS P5", "Forward b_f and reverse 1/delta bracket gamma iff -(1 + V_eps/V_w)/gamma <= pi1 <= 0 (given Cov(c,y) > 0); "
           "they do NOT bound gamma under the funding regime (pi1 > 0)",
           "IMPRECISE", "sympy", ok3, "exact: (1/delta - gamma) Cov(c,y) = (1+gamma pi1) V_w + V_eps",
           "SYNTHESIS P5 says forward/reverse regressions bound gamma 'under classical noise'; that holds only for target/exogenous-type rules.")

    # ------------------------------------------------------------------ information: exact misallocation decomposition (task 2a)
    Dl = sp.symbols("Delta", real=True)
    aa2 = sp.symbols("a", positive=True)
    bb2 = 1 - aa2
    Phi = (sp.log(bb2 + aa2 * sp.exp(-S * Dl)) + aa2 * S * Dl) / (aa2 * bb2 * S)
    ser = sp.series(Phi, Dl, 0, 3).removeO()
    okP1 = sp.simplify(ser - S * Dl ** 2 / 2) == 0
    dPhi = sp.diff(Phi, S)
    ser2 = sp.series(dPhi, Dl, 0, 3).removeO()
    okP2 = sp.simplify(ser2 - Dl ** 2 / 2) == 0
    # Phi equals ln(C/C_min) with w = e^{-S Delta}
    wD = sp.exp(-S * Dl)
    a1_, b1_ = bb2 * S, aa2 * S
    ce_w = sp.log(((a1_ + b1_ * wD) / S) ** (S / (a1_ * b1_)) * wD ** (-1 / a1_))
    okP3, mP3 = _zero(sp.simplify(ce_w - Phi), subs_random=10)
    record("INFO.dec", "new (task 2a)", "kappa-free family: y = y*(c; a,gamma,G,K) - gamma Phi_S(Delta) + omega - eps EXACTLY, where Phi = ln(C/C_min) "
           "(Farrell allocative loss), Delta = b n - a d - ln G = -ln w / S; sigma* enters only through Phi; dPhi/dS = Delta^2/2 + O(Delta^3) = Phi/S + O(Delta^3)",
           "NEW", "sympy", okP1 and okP2 and okP3, mP3)

    # ------------------------------------------------------------------ proxy (task 2c)
    chi_s = sp.symbols("chi", real=True)
    lnM = (d1 - n1)
    lnM_chi = sp.simplify(lnM.subs(psD, (chi_s + al * psN) / be))
    okx1 = sp.simplify(sp.diff(lnM_chi, chi_s) + 2 / (al + be)) == 0
    okx2 = sp.simplify(sp.diff(lnM_chi, psN)) == 0 and sp.simplify(sp.diff(lnM, om)) == 0
    record("PROXY", "new (task 2c)", "Given c: ln(D/N) = m0(c) - (2/(alpha+beta)) chi, chi = beta psi_D - alpha psi_N; strictly monotone in chi (LP invertibility), "
           "independent of Hicks-neutral omega and of any psi with chi = 0",
           "NEW", "sympy", okx1 and okx2, "exact",
           "With heterogeneous inference demand ln M - m0(c) = (2/(alpha+beta))(ln w - chi): the mix confounds inference demand and factor bias.")
