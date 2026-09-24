"""Symbolic (sympy) verification of the Theory-v2 results (appendix_proofs.tex, revision round 1).

Each block proves an identity by reducing (lhs - rhs) to zero, possibly confirmed at random rational points when full
simplification is slow. Identities that involve derivatives of an arbitrary smooth technology at a point are checked
with a generic Taylor polynomial whose coefficients are free symbols: this is exact for any statement that involves
derivatives up to the polynomial's order at that point.
"""
from __future__ import annotations

import random

import sympy as sp

from ra5common import record

POS = dict(positive=True, real=True)


def _zero(expr, subs_random=0, tol=1e-20, simplify=True):
    if simplify:
        e = sp.simplify(expr)
        if e == 0:
            return True, "exact"
    else:
        e = expr
    if subs_random:
        rng = random.Random(7)
        worst = 0.0
        syms = sorted(expr.free_symbols, key=lambda s: s.name)
        for _ in range(subs_random):
            vals = {s: sp.Rational(rng.randint(5, 95), 100) for s in syms}
            worst = max(worst, abs(complex(sp.N(expr.subs(vals), 40))))
        return worst < tol, f"max|err| at {subs_random} random points = {worst:.1e}"
    return False, f"residual {e}"


# --------------------------------------------------------------------------------------------------------------------
def generalized_wedge():
    """Prop. A8 (generalized wedge): FOC algebra with an arbitrary objective V(L, N), size-dependent training price,
    per-token data cost and generic constraints; the special cases (a)-(f) and their expenditure shares."""
    # Concrete but general objective: Chinchilla technology, V(L,N) = Psi*(-ln(L-E)) - kV N^rho (a value of compactness
    # with any elasticity), size-dependent training price c_T(N) = c0 N^delta, per-token data cost, and a generic linearized
    # constraint xi*(g_n n + g_d d). Solve the d-FOC for the scale Psi and show Pi_n = (X_T + c_D D + xi g_d)(w - w_formula).
    n, d = sp.symbols("n d", real=True)
    E, A, B, al, be, Psi, kV, rho, c0, dl, cD = sp.symbols("E A B alpha beta Psi k_V rho c_0 delta c_D", positive=True)
    xi, gn, gd = sp.symbols("xi g_n g_d", real=True)
    N, D = sp.exp(n), sp.exp(d)
    L = E + A * sp.exp(-al * n) + B * sp.exp(-be * d)
    V = Psi * (-sp.log(L - E)) - kV * N ** rho
    XT = c0 * N ** dl * 6 * N * D
    Pi = V - XT - cD * D - xi * (gn * n + gd * d)
    Pn, Pd = sp.diff(Pi, n), sp.diff(Pi, d)
    epsN = -sp.diff(sp.log(L - E), n)
    epsD = -sp.diff(sp.log(L - E), d)
    Psi_sol = sp.solve(sp.Eq(Pd, 0), Psi)[0]
    minus_Vn = -N * sp.diff(-kV * sp.Symbol("NN") ** rho, sp.Symbol("NN")).subs(sp.Symbol("NN"), N)   # -dV/dlnN at fixed L
    w_formula = (1 + dl + minus_Vn / XT + xi * gn / XT) / (1 + cD * D / XT + xi * gd / XT)
    resid = Pn.subs(Psi, Psi_sol) - (XT + cD * D + xi * gd) * (epsN / epsD - w_formula)
    ok, metric = _zero(resid, subs_random=12, tol=1e-25, simplify=False)
    record("GW.foc", "Prop. A8(i)", "Any objective V(L,N): w = (1+delta+m_N+xi g_n/X_T)/(1+m_D+xi g_d/X_T); with delta=m_D=0 and no "
           "binding constraint, w-1 = (-dV/dlnN|_L)/(c_train 6ND)", "NEW", "sympy", ok, metric)
    # ---- case (a): serving cost borne with internalization lambda, price ratio p, serving-cost elasticity phi,
    #      demand elasticity eta = dlnT/dlnN|_L, direct revenue effect R_n
    N, D, T0, lam, p0, N0, phi, eta = sp.symbols("N D T_0 lambda p_0 N_0 phi eta", positive=True)
    cS = p0 * (N / N0) ** (phi - 1)             # serving price per FLOP (training price normalized to 1)
    T = T0 * (N / N0) ** eta                     # lifetime tokens at given quality
    Vserv = -lam * cS * 2 * N * T                # internalized serving expenditure (enters V with minus sign)
    minus_Vn = -N * sp.diff(Vserv, N)            # -dV/dlnN at fixed L
    XT = 6 * N * D
    p = cS                                       # FLOP price ratio at the model's N (c_T = 1)
    target = lam * p * (phi + eta) * T / (3 * D)
    ok, metric = _zero(sp.simplify(minus_Vn / XT - target))
    record("GW.a.sym", "Prop. A8(iii)(a)", "Serving borne at rate lambda: w - 1 = lambda p (phi+eta) T/(3D) - R_n/X_T; "
           "= lambda p T/(3D) when phi=1, eta=0, R_n=0", "NEW", "sympy", ok, metric)
    # expenditure share under (a) with phi=1, eta=0: s = lambda X_S/(X_T + lambda X_S)
    XS = cS * 2 * N * T
    w_a = 1 + lam * XS / XT
    ok, metric = _zero(sp.simplify(((w_a - 1) / w_a) - lam * XS / (XT + lam * XS)))
    record("GW.a.share", "Prop. A8(iii)(a)", "s=(w-1)/w = lambda X_S/(X_T+lambda X_S): internalized serving share of lifetime "
           "expenditure; needs no FLOP-to-dollar conversion (p)", "NEW", "sympy", ok, metric)

    # ---- case (d): per-token data cost: w = (1+m_N)/(1+m_D); s = (X_S - X_D)/(X_T + X_S); lowers w for given T
    mN_, mD_ = sp.symbols("m_N m_D", positive=True)
    XT_, XS_, XD_ = sp.symbols("X_T X_S X_D", positive=True)
    w_d = (1 + XS_ / XT_) / (1 + XD_ / XT_)
    ok1, m1 = _zero(sp.simplify((w_d - 1) / w_d - (XS_ - XD_) / (XT_ + XS_)))
    dw = sp.diff((1 + mN_) / (1 + mD_), mD_)
    ok2 = sp.simplify(dw + (1 + mN_) / (1 + mD_) ** 2) == 0
    # implied Tstar/D = 3(w-1) under (a) with lambda=p=1 understates T/D
    tD = sp.symbols("tau_D", positive=True)  # T/D
    That = sp.simplify(3 * ((1 + tD / 3) / (1 + mD_) - 1))
    ok3 = sp.simplify(That - (tD - 3 * mD_) / (1 + mD_)) == 0
    record("GW.d.sym", "Prop. A8(iii)(d)", "Data cost: w=(1+m_N)/(1+m_D), dw/dm_D<0; s=(X_S-X_D)/(X_T+X_S); "
           "T-hat/D = (T/D - 3m_D)/(1+m_D) < T/D", "NEW", "sympy", ok1 and ok2 and ok3, m1)

    # ---- case (e): logit distillation with teacher N_T: cost 6ND + 2 N_T D + 2NT
    NT, Tt = sp.symbols("N_T T", positive=True)
    nn_, dd_ = sp.symbols("nn dd", real=True)
    K = 6 * sp.exp(nn_ + dd_) + 2 * NT * sp.exp(dd_) + 2 * sp.exp(nn_) * Tt
    w_e = sp.diff(K, nn_) / sp.diff(K, dd_)
    target = (1 + Tt / (3 * sp.exp(dd_))) / (1 + NT / (3 * sp.exp(nn_)))
    ok, metric = _zero(sp.simplify(w_e - target))
    record("GW.e.sym", "Prop. A8(iii)(e)", "Logit distillation (teacher N_T): w = (1+T/(3D))/(1+N_T/(3N)) (Referee 2, Major 4c): "
           "teacher passes are a per-token data cost with m_D = N_T/(3N)", "VERIFIED", "sympy", ok, metric)

    # ---- case (f): dollar training cost ~ N^{1+delta} D; serving cost per FLOP with size elasticity delta_S
    dl_, dS_, c0, pp = sp.symbols("delta delta_S c_0 p_0", positive=True)
    Kf = c0 * sp.exp((1 + dl_) * nn_) * 6 * sp.exp(dd_) + lam * c0 * pp * sp.exp((1 + dS_) * nn_) * 2 * Tt
    w_f = sp.simplify(sp.diff(Kf, nn_) / sp.diff(Kf, dd_))
    XTf = c0 * sp.exp((1 + dl_) * nn_) * 6 * sp.exp(dd_)
    XSf = c0 * pp * sp.exp((1 + dS_) * nn_) * 2 * Tt
    ok1, m1 = _zero(sp.simplify(w_f - (1 + dl_ + lam * (1 + dS_) * XSf / XTf)))
    w_f_eq = w_f.subs(dS_, dl_)
    ok2, m2 = _zero(sp.simplify(w_f_eq - (1 + dl_) * (1 + lam * XSf.subs(dS_, dl_) / XTf)))
    record("GW.f.sym", "Prop. A8(iii)(f)", "MFU varying with size (dollar cost ~ N^{1+delta}D): w = 1+delta+lambda(1+delta_S)X_S/X_T; "
           "= (1+delta)(1+lambda p T/(3D)) iff serving cost per FLOP has the same size elasticity (delta_S=delta)",
           "CORRECTED", "sympy", ok1 and ok2, f"{m1}; {m2}",
           "Referee 1's (1+delta)(1+T/3D) needs delta_S = delta; with FLOP-proportional serving cost (delta_S=0) the form is additive.")

    # ---- generic constraint sign rule: adding xi*g raises w iff g_n/g_d > w (g_d>0), or g_n>0 with g_d<=0
    A_, B_, x_, gn_, gd_ = sp.symbols("A B x g_n g_d", positive=True)
    dw = sp.diff((A_ + x_ * gn_) / (B_ + x_ * gd_), x_).subs(x_, 0)
    ok = sp.simplify(dw - (gn_ * B_ - gd_ * A_) / B_ ** 2) == 0
    record("GW.sign", "Prop. A8(iv)", "A binding constraint g(n,d)<=0 raises w iff g_n/g_d > w: memory/tier caps and latency SLOs raise w; "
           "data caps, data-parallel deadlines and minimum-size floors lower it; compute-bound deadlines pull w toward 1",
           "NEW", "sympy", ok, "d w/d xi at 0 = (g_n B - g_d A)/B^2")


# --------------------------------------------------------------------------------------------------------------------
def family():
    """Prop. A9 (family-level token budget)."""
    K = 3
    om = sp.symbols("omega1:4", positive=True)
    w = sp.symbols("w1:4", positive=True)
    m = sp.symbols("m1:4", positive=True)
    mD = sp.symbols("m_D", positive=True)
    X = sp.symbols("X1:4", positive=True)      # member training expenditures
    Lam = sp.symbols("Lambda1:4", positive=True)
    eN = sp.symbols("eN1:4", positive=True)
    # member n-FOCs: Lam_i eN_i = X_i (1+m_i); family d-FOC: sum Lam_i eD_i = sum X_i (1+m_D); eD_i = eN_i/w_i
    lam_sol = [X[i] * (1 + m[i]) / eN[i] for i in range(K)]
    lhs = sum(lam_sol[i] * eN[i] / w[i] for i in range(K))
    fam = sp.simplify(lhs / sum(X) - sum((X[i] / sum(X)) * (1 + m[i]) / w[i] for i in range(K)))
    ok0 = fam == 0
    # with omega = X/sum X: sum pi_i (1+m_i) = (1+m_D) w_H  where pi_i = (om_i/w_i)/sum(om_j/w_j), w_H = 1/sum(om_i/w_i)
    Sinv = sum(om[i] / w[i] for i in range(K))
    pi = [om[i] / w[i] / Sinv for i in range(K)]
    wH = 1 / Sinv
    # impose the family FOC sum om_i (1+m_i)/w_i = 1 + m_D and check the pi-weighted identity
    lhs2 = sum(pi[i] * (1 + m[i]) for i in range(K))
    ok1, _ = _zero(sp.simplify(lhs2 - sum(om[i] * (1 + m[i]) / w[i] for i in range(K)) * wH))
    ok2, _ = _zero(sp.simplify((sum(pi[i] * w[i] for i in range(K)) - wH).subs({om[2]: 1 - om[0] - om[1]})))
    # shares: 1 - 1/w_H = sum om_i s_i when sum om_i = 1
    om3 = 1 - om[0] - om[1]
    sub = {om[2]: om3}
    ok3, _ = _zero(sp.simplify((1 - Sinv).subs(sub) - sum((om[i] * (1 - 1 / w[i])) for i in range(K)).subs(sub)))
    record("FAM.sym", "Prop. A9(ii)-(iii)", "Family FOC: sum_i omega_i(1+m_N,i)/w_i = 1+m_D; hence sum_i pi_i(1+m_N,i) = (1+m_D) w_H "
           "(pi_i ~ omega_i/w_i, w_H = compute-weighted harmonic mean), sum pi_i w_i = w_H, and 1-1/w_H = sum omega_i s_i",
           "NEW", "sympy", ok0 and ok1 and ok2 and ok3, "exact")
    # member wedges are technological: ln w(n_i,d) - ln w(n_j,d) = -a1 (n_i - n_j) in the generalized family
    n1, n2, d = sp.symbols("n1 n2 d", real=True)
    A, B, a1, b1, kap = sp.symbols("A B a_1 b_1 kappa", positive=True)
    lnw = lambda nn: sp.log(a1 * A * sp.exp(-a1 * nn) / (b1 * B * sp.exp(-b1 * d)))
    ok4, m4 = _zero(sp.expand_log(lnw(n1) - lnw(n2), force=True) + a1 * (n1 - n2))
    record("FAM.slope", "Prop. A9(i)", "Common D: ln w_i - ln w_j = -a_1 (ln N_i - ln N_j) exactly in the generalized family; the cross-size "
           "profile of T-hat_i = 3D(w_i-1) is fixed by the technology", "NEW", "sympy", ok4, m4)


# --------------------------------------------------------------------------------------------------------------------
def model_free():
    """Prop. A10: model-free sigma* and w, via a generic 4th-order Taylor polynomial of the technology at a point."""
    n, d = sp.symbols("n d", real=True)
    coeffs = {}
    F = 0
    for i in range(5):
        for j in range(5 - i):
            cf = sp.Symbol(f"f{i}{j}", real=True)
            coeffs[(i, j)] = cf
            F += cf * n ** i * d ** j / (sp.factorial(i) * sp.factorial(j))
    x, c = sp.symbols("x c", real=True)
    ln6 = sp.log(6)
    Y = F.subs({n: (c - ln6 - x) / 2, d: (c - ln6 + x) / 2}, simultaneous=True)
    at0 = {x: 0, c: ln6}   # the point (n,d) = (0,0)
    Fn, Fd = sp.diff(F, n).subs({n: 0, d: 0}), sp.diff(F, d).subs({n: 0, d: 0})
    Yx, Yc = sp.diff(Y, x).subs(at0), sp.diff(Y, c).subs(at0)
    ok1 = sp.simplify(Fn - (Yc - Yx)) == 0 and sp.simplify(Fd - (Yc + Yx)) == 0
    # loss L = h(F) with h decreasing: w = (L_c - L_x)/(L_c + L_x)
    h1, h2 = sp.symbols("h1 h2", real=True)   # h'(F0), h''(F0)
    F0 = F.subs({n: 0, d: 0})
    Lx = h1 * Yx
    Lc = h1 * Yc
    ok2 = sp.simplify((Lc - Lx) / (Lc + Lx) - Fn / Fd) == 0
    record("MF.w", "Prop. A10(i)", "For any smooth technology and any monotone output: w = F_n/F_d = (Y_c-Y_x)/(Y_c+Y_x) = (L_c-L_x)/(L_c+L_x), "
           "x = ln M, c = ln C (Referee 1, comment 7)", "NEW", "sympy", ok1 and ok2, "exact")
    # sigma* at an argmin of the IsoFLOP profile: impose Y_x = 0 <=> f10 = f01
    Fnn = sp.diff(F, n, 2).subs({n: 0, d: 0})
    Fdd = sp.diff(F, d, 2).subs({n: 0, d: 0})
    Fnd = sp.diff(F, n, d).subs({n: 0, d: 0})
    P = Fn * Fd * (Fn + Fd)
    Q = -(Fnn * Fd ** 2 - 2 * Fnd * Fn * Fd + Fdd * Fn ** 2)
    onpath = {coeffs[(0, 1)]: coeffs[(1, 0)]}
    inv_sig_m1 = sp.simplify((Q / P).subs(onpath))
    Yxx = sp.diff(Y, x, 2).subs(at0)
    ok3 = sp.simplify(inv_sig_m1 - (-2 * Yxx / Yc).subs(onpath)) == 0
    # in loss units: L_xx = h1 Y_xx at Y_x=0; dL*/dc = h1 Y_c (envelope); h1 < 0
    h1n = sp.Symbol("h1n", positive=True)   # h1 = -h1n
    Lxx = (-h1n) * Yxx + h2 * Yx ** 2
    dLstar = (-h1n) * Yc
    # |dL*/dc| = h1n*Y_c because h' = -h1n < 0 and Y_c > 0 (F_n, F_d > 0)
    ok4 = sp.simplify((2 * Lxx / (h1n * Yc)).subs(onpath) - inv_sig_m1) == 0
    # L_nn at fixed C equals 4 L_xx (n = (c - ln6 - x)/2)
    Lf = sp.Function("Lf")
    ok5 = True  # chain rule: d/dn|_c = -2 d/dx  => d2/dn2 = 4 d2/dx2 (checked on Y)
    Ynn = sp.diff(F.subs(d, sp.Symbol("cc") - ln6 - n), n, 2).subs({n: 0, sp.Symbol("cc"): ln6})
    ok5 = sp.simplify(Ynn - 4 * Yxx) == 0
    record("MF.sigma", "Prop. A10(ii)", "At a compute-optimal point (Y_x=0): 1/sigma*-1 = 2(-Y_xx)/Y_c = 2 L_xx/|dL*/dc| = L_nn|_C/(2|dL*/dc|) "
           "for any smooth technology (sigma from the general P/(P+Q) identity, Lemma A2)", "NEW", "sympy", ok3 and ok4 and ok5, "exact")
    # invariance to a monotone transform g of output at the argmin (g', g'' arbitrary)
    g1, g2 = sp.symbols("g1 g2", positive=True)
    Gxx = g2 * Yx ** 2 + g1 * Yxx
    Gc = g1 * Yc
    ok6 = sp.simplify(((Gxx / Gc) - (Yxx / Yc)).subs(onpath)) == 0
    ok6b = sp.simplify((Gxx / Gc) - (Yxx / Yc)) != 0   # not invariant away from the argmin
    record("MF.inv", "Prop. A10(ii)", "The ratio (output curvature along the isocost)/(frontier slope) is invariant to any strictly monotone "
           "transformation of output at the argmin (no E, kappa or functional form), but not away from it", "NEW", "sympy",
           ok6 and ok6b, "exact")
    # slope of ln w along the isocost at the argmin equals 1/sigma* - 1
    lnw = sp.log(Yc_ := sp.diff(Y, c) - sp.diff(Y, x)) - sp.log(sp.diff(Y, c) + sp.diff(Y, x))
    dlnw = sp.diff(lnw, x).subs(at0)
    ok7 = sp.simplify((dlnw - inv_sig_m1).subs(onpath)) == 0
    record("MF.slope", "Prop. A10(iii)", "d ln w/d ln M at fixed C, evaluated at M*(C), equals 1/sigma*-1 for any smooth technology "
           "(exact and global in the generalized family)", "NEW", "sympy", ok7, "exact")
    # local sigma anywhere from the (x,c) quadratic: F_nn = Y_cc - 2Y_cx + Y_xx, F_dd = Y_cc + 2Y_cx + Y_xx, F_nd = Y_cc - Y_xx
    Ycc = sp.diff(Y, c, 2).subs(at0)
    Ycx = sp.diff(Y, c, x).subs(at0)
    ok8 = (sp.simplify(Fnn - (Ycc - 2 * Ycx + Yxx)) == 0 and sp.simplify(Fdd - (Ycc + 2 * Ycx + Yxx)) == 0
           and sp.simplify(Fnd - (Ycc - Yxx)) == 0)
    record("MF.local", "Prop. A10(iv)", "Local sigma at any point from the second-order expansion of any output in (ln M, ln C): "
           "F_nn = Y_cc-2Y_cx+Y_xx, F_dd = Y_cc+2Y_cx+Y_xx, F_nd = Y_cc-Y_xx", "NEW", "sympy", ok8, "exact")

    # generalized family: 1/sigma*-1 = S/2 from the model-free formula, for any kappa and E; kappa=1 pins -y_nn = gamma^2/(ab)
    A, B, a1, b1, kap, E, cp = sp.symbols("A B a_1 b_1 kappa E cp", positive=True)
    nn = sp.symbols("nn", real=True)
    S = a1 + b1
    Lprof = E + (A * sp.exp(-a1 * nn) + B * sp.exp(-b1 * (cp - nn))) ** kap     # IsoFLOP profile in ln N, cp = ln(C/6)
    G = (a1 * A / (b1 * B)) ** (1 / S)
    nstar = sp.log(G) + (b1 / S) * cp
    Lnn = sp.diff(Lprof, nn, 2).subs(nn, nstar)
    Lstar = Lprof.subs(nn, nstar)
    dL = sp.diff(Lstar, cp)
    kmf = Lnn / (2 * (-dL))
    okA, mA = _zero(kmf - S / 2, subs_random=12, tol=1e-25, simplify=False)
    y_prof = -sp.log(Lprof - E)
    ynn = sp.diff(y_prof, nn, 2).subs(nn, nstar)
    gam = kap * a1 * b1 / S
    okB, mB = _zero(-ynn - S * gam, subs_random=12, tol=1e-25, simplify=False)
    a_, b_ = b1 / S, a1 / S
    okC, mC = _zero(sp.simplify((S * gam).subs(kap, 1) - (gam.subs(kap, 1)) ** 2 / (a_ * b_)))
    record("MF.family", "Prop. A10(ii),(v)", "Generalized family: model-free formula returns S/2 = 1/sigma*-1 for every kappa and E; "
           "-y_nn|_C = S gamma, and kappa=1 imposes -y_nn|_C = gamma^2/(ab) (curvature pinned by frontier and path slopes)",
           "NEW", "sympy", okA and okB and okC, f"{mA}; {mB}")


# --------------------------------------------------------------------------------------------------------------------
def quasi_homothetic():
    """Lemma A4(iv): quasi-homotheticity of the generalized family."""
    N, D, lam, A, B, a1, b1, kap = sp.symbols("N D lambda A B a_1 b_1 kappa", positive=True)
    R = lambda NN, DD: (A * NN ** (-a1) + B * DD ** (-b1)) ** kap
    ok1, m1 = _zero(R(lam ** b1 * N, lam ** a1 * D) - lam ** (-kap * a1 * b1) * R(N, D), subs_random=15, tol=1e-25, simplify=False)
    n, d, t = sp.symbols("n d t", real=True)
    lnw = lambda nn, dd: sp.log(a1 * A * sp.exp(-a1 * nn)) - sp.log(b1 * B * sp.exp(-b1 * dd))
    ok2 = sp.simplify(sp.expand_log(lnw(n + b1 * t, d + a1 * t) - lnw(n, d), force=True)) == 0
    record("QH", "Lemma A4(iv)", "Generalized family is quasi-homogeneous: R(lambda^{b1}N, lambda^{a1}D) = lambda^{-kappa a1 b1} R(N,D); "
           "the wedge is constant along the dilation orbits (the content of the rank-one Hessian; Referee 1, minor 9)",
           "NEW", "sympy", ok1 and ok2, m1)


# --------------------------------------------------------------------------------------------------------------------
def dmr_sign():
    """Prop. A3 / Remark: the drift of M* signs B = b1 g_D - a1 g_N (= beta g_D - alpha g_N with kappa=1)."""
    al, be, gN, gD = sp.symbols("alpha beta g_N g_D", real=True)
    S = al + be
    a, b = be / S, al / S
    B = be * gD - al * gN
    sig = 2 / (2 + S)
    drift = 2 * (b * gN - a * gD)
    ok1 = sp.simplify(drift - (-sig / (1 - sig) * B)) == 0
    vals = {al: sp.Rational(3, 10), be: sp.Rational(45, 100), gN: sp.Rational(5, 10), gD: sp.Rational(4, 10)}
    Bv = B.subs(vals)
    ok2 = (Bv > 0) and (vals[gD] - vals[gN] < 0)
    record("DMR.sign", "Prop. A3(i)-(ii), Remark", "Drift of M* signs B = beta g_D - alpha g_N (kappa=1), not g_D - g_N: with (alpha,beta)="
           "(0.30,0.45), (g_N,g_D)=(0.5,0.4), B=+0.03>0 (M* falls) while g_N>g_D", "CORRECTED", "sympy", ok1 and ok2,
           f"B={float(Bv):.3f}, g_D-g_N=-0.1",
           "The v1 remark's 'both signs identified' holds only when alpha=beta (Referee 1, comment 8f).")


# --------------------------------------------------------------------------------------------------------------------
def partial_id():
    """Prop. A11: sign of w-1 equals sign of ln M - ln M*(C) for any technology with single-peaked IsoFLOP profiles."""
    Yx, Yc = sp.symbols("Y_x Y_c", real=True)
    w = (Yc - Yx) / (Yc + Yx)
    ok = sp.simplify(w - 1 - (-2 * Yx / (Yc + Yx))) == 0
    record("PI.sign.sym", "Prop. A11(iii)", "w - 1 = -2Y_x/F_d: with F_d>0, sign(w-1) = -sign(Y_x) = sign(ln M - ln M*(C)) when IsoFLOP "
           "profiles are single-peaked (guaranteed by 0<sigma<1 everywhere, Lemma A2)", "NEW", "sympy", ok, "exact")
    # normal inputs bound the path slope: with a = dln N*/dln C in [0,1] (both N* and D* nondecreasing), e = 1-2a in [-1,1]
    a = sp.symbols("a", real=True)
    e = 1 - 2 * a
    iv = sp.Interval(0, 1)
    ok2 = (sp.maximum(e, a, iv) == 1) and (sp.minimum(e, a, iv) == -1)
    record("PI.normal", "Prop. A11(i)", "Normal inputs (N*, D* nondecreasing in C, i.e. a in [0,1]) imply e = d ln M*/d ln C = 1-2a in [-1,1]",
           "NEW", "sympy", bool(ok2), "max/min of 1-2a on [0,1] = 1/-1")


# --------------------------------------------------------------------------------------------------------------------
def singular():
    """Prop. A1(iii): on-path outcome model under kappa=1 is a two-rate exponential mixture; twin equivalence."""
    E, A, B, al, be, n0, d0, a, b, cp = sp.symbols("E A B alpha beta n_0 d_0 a b cp", positive=True)
    mean = lambda E_, A_, B_, al_, be_: E_ + A_ * sp.exp(-al_ * (n0 + a * cp)) + B_ * sp.exp(-be_ * (d0 + b * cp))
    al2 = be * b / a
    be2 = al * a / b
    A2 = B * sp.exp(-be * d0 + al2 * n0)
    B2 = A * sp.exp(-al * n0 + be2 * d0)
    ok, m = _zero(mean(E, A2, B2, al2, be2) - mean(E, A, B, al, be), subs_random=12, tol=1e-25, simplify=False)
    record("SI.twin", "Prop. A1(iii)", "On-path outcomes under kappa=1: (alpha,beta) and its twin (beta b/a, alpha a/b) (with A,B relabeled) "
           "generate identical data exactly; the truth (alpha a = beta b) is its own twin", "NEW", "sympy", ok, m)


def run_all():
    generalized_wedge()
    family()
    model_free()
    quasi_homothetic()
    dmr_sign()
    partial_id()
    singular()
