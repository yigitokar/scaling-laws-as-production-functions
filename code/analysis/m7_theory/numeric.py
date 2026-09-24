"""Numerical and Monte Carlo verification of the theory (brute-force optimization, simulation, Fisher
information).  Complements symbolic.py: every closed form is re-derived here by a method that does not
use the closed form (direct optimization, finite differences, simulation).

All randomness uses np.random.default_rng with fixed seeds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm

from common import (LN6, PARAM_SETS, ce_ratio_direct, ce_ratio_from_w, elasticities, kappa_family_member, lnR,
                    path_objects, record, M_star)

BES = PARAM_SETS["Besiroglu"]
HOF = PARAM_SETS["Hoffmann (TeX)"]


# ============================================================================ helpers
def argmin_at_compute(C, A, B, a1, b1, kappa=1.0, psN=0.0, psD=0.0, omega=0.0):
    """Brute-force compute-optimal n at fixed C (no closed form used). Objective is ln(L-E) = -omega + ln R(eff. inputs)
    (epsilon is not known at choice time). Returns (n, d, ln R) with ln R omega-free."""
    lc = np.log(C / 6.0)
    f = lambda nn: -omega + lnR(nn + psN, lc - nn + psD, A, B, a1, b1, kappa)
    r = minimize_scalar(f, bounds=(lc / 2 - 25, lc / 2 + 25), method="bounded", options=dict(xatol=1e-12))
    return r.x, lc - r.x, r.fun + omega


def lifetime_optimum(Lbar, T, E, A, B, al, be, omega=0.0, psN=0.0, psD=0.0, Dbar=None, Nbar=None):
    """min 6ND + 2NT  s.t.  E + e^{-omega}[A(e^psN N)^-al + B(e^psD D)^-be] <= Lbar  (brute force over n).
    Optional data (D <= Dbar) or memory (N <= Nbar) constraints. Returns (N, D, cost)."""
    Rbar = (Lbar - E) * np.exp(omega)

    def d_of_n(nn):
        rem = Rbar - A * np.exp(-al * (nn + psN))
        return np.where(rem > 0, -(np.log(rem / B)) / be - psD, np.inf)

    def cost(nn):
        dd = d_of_n(nn)
        if not np.isfinite(dd):
            return 1e300
        return np.log(6 * np.exp(nn + dd) + 2 * np.exp(nn) * T)

    n_lo = np.log(A / Rbar) / al - psN + 1e-9
    if Nbar is not None:
        # memory constraint: check if unconstrained optimum violates it
        pass
    r = minimize_scalar(cost, bounds=(n_lo, n_lo + 40), method="bounded", options=dict(xatol=1e-12))
    nn = r.x
    if Nbar is not None and nn > np.log(Nbar):
        nn = np.log(Nbar)
    dd = float(d_of_n(nn))
    if Dbar is not None and dd > np.log(Dbar):
        dd = np.log(Dbar)
        # N from the loss constraint given D = Dbar
        rem = Rbar - B * np.exp(-be * (dd + psD))
        nn = -np.log(rem / A) / al - psN
    return np.exp(nn), np.exp(dd), 6 * np.exp(nn + dd) + 2 * np.exp(nn) * T


def ols(y, X):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(x, float) for x in X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


# ============================================================================ checks
def check_path_frontier():
    worst = 0.0
    for name, p in PARAM_SETS.items():
        po = path_objects(p["A"], p["B"], p["alpha"], p["beta"])
        for C in (1e19, 1e21, 5.76e23, 3.8e25):
            n_opt, d_opt, lnr = argmin_at_compute(C, p["A"], p["B"], p["alpha"], p["beta"])
            N_cf = po["G"] * (C / 6) ** po["a"]
            R_cf = po["K"] * (C / 6) ** (-po["gamma"])
            worst = max(worst, abs(np.exp(n_opt) / N_cf - 1), abs(np.exp(lnr) / R_cf - 1))
    record("P1.path.num", "model_spec/SYNTHESIS P1", "Brute-force argmin of L at fixed C reproduces N*, D*, L*(C) closed forms (4 parameter sets x 4 budgets)",
           "VERIFIED", "numeric", worst < 1e-6, f"max rel err {worst:.1e}")


def check_sigma_numeric():
    rng = np.random.default_rng(1)
    worst = 0.0
    p = BES
    A_, B_, al, be = p["A"], p["B"], p["alpha"], p["beta"]
    for _ in range(50):
        n0, d0 = rng.uniform(16, 26), rng.uniform(18, 30)
        R0 = A_ * np.exp(-al * n0) + B_ * np.exp(-be * d0)
        # move along the isoquant: for n = n0 +- h solve d
        def d_on(nn):
            return -np.log((R0 - A_ * np.exp(-al * nn)) / B_) / be
        h = 1e-4
        pts = []
        for nn in (n0 - h, n0 + h):
            dd = d_on(nn)
            u, v = A_ * np.exp(-al * nn), B_ * np.exp(-be * dd)
            ln_mrts = np.log(al * u / (be * v)) + (dd - nn)  # ln(f_N/f_D) in levels
            pts.append((dd - nn, ln_mrts))
        sig_fd = (pts[1][0] - pts[0][0]) / (pts[1][1] - pts[0][1])
        u, v = A_ * np.exp(-al * n0), B_ * np.exp(-be * d0)
        sig_cf = (al * u + be * v) / (al * u * (1 + be) + be * v * (1 + al))
        w = al * u / (be * v)
        sig_w = (1 + w) / (1 + al + w * (1 + be))
        worst = max(worst, abs(sig_fd / sig_cf - 1), abs(sig_w / sig_cf - 1))
    record("MS.SIG.num", "model_spec", "Finite-difference sigma = dln(D/N)/dln MRTS along numerically traced isoquants equals the closed form and sigma(w)",
           "VERIFIED", "numeric", worst < 1e-6, f"max rel err {worst:.1e} (50 random points)")


def check_lemma1_numeric():
    p = BES
    A_, B_, al, be = p["A"], p["B"], p["alpha"], p["beta"]
    po = path_objects(A_, B_, al, be)
    a, b = po["a"], po["b"]
    C = 1e22
    lc = np.log(C / 6)
    worst = 0.0
    for psN, psD, om in [(0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.0, 0.5, 0.0), (-0.2, 0.4, 1.0), (0.1, 0.1, -0.7)]:
        n_opt, d_opt, lnr = argmin_at_compute(C, A_, B_, al, be, psN=psN, psD=psD, omega=om)
        n_cf = np.log(po["G"]) + a * lc + a * psD - b * psN
        y_num = om - lnr
        y_cf = om + po["gamma"] * (lc + psN + psD) - np.log(po["K"])
        worst = max(worst, abs(n_opt - n_cf), abs(y_num - y_cf))
    # signs by finite differences (fixed compute)
    n0, d0, _ = argmin_at_compute(C, A_, B_, al, be)
    n1, d1, _ = argmin_at_compute(C, A_, B_, al, be, psD=0.01)
    s_ok = (n1 - n0) > 0 and (d1 - d0) < 0 and abs((n1 - n0) / 0.01 - a) < 1e-5
    record("L1.num", "model_spec Lemma 1", "Brute-force optimum with (omega, psi_N, psi_D) matches n*, y* closed forms; better data -> more N, fewer D, lower D/N at fixed C",
           "VERIFIED", "numeric", worst < 1e-6 and s_ok, f"max abs err {worst:.1e}; dn*/dpsi_D = {(n1 - n0) / 0.01:.4f} vs a = {a:.4f}")
    # with inference demand T > 0: better data still lowers D/N (fixed lifetime-optimal problem at fixed target)
    Lbar = 2.3
    N0, D0, _ = lifetime_optimum(Lbar, 5e12, p["E"], A_, B_, al, be)
    N1, D1, _ = lifetime_optimum(Lbar, 5e12, p["E"], A_, B_, al, be, psD=0.05)
    record("L1.T", "new (Lemma 1 with T>0)", "With inference demand (T>0), better data (psi_D up) still lowers tokens per parameter D/N",
           "NEW", "numeric", (D1 / N1) < (D0 / N0), f"D/N: {D0 / N0:.1f} -> {D1 / N1:.1f}")
    # Reviewer addition: Lemma A3(vi) at GIVEN TRAINING COMPUTE (the appendix statement), by brute force. For each psi_D the
    # loss target is solved so that the lifetime-optimal run uses exactly C_target training FLOP; then d ln M/d chi is a
    # central difference (chi = beta psi_D here) compared with -2/(alpha+beta+theta_T).
    Tl, C_target, h = 5e12, 1e22, 1e-3

    def _at_compute(psD):
        def g(lr):
            N_, D_, _ = lifetime_optimum(p["E"] + np.exp(lr), Tl, p["E"], A_, B_, al, be, psD=psD)
            return np.log(6 * N_ * D_ / C_target)
        lr0 = np.log(po["K"]) - po["gamma"] * np.log(C_target / 6)
        lr = brentq(g, lr0 - 3.0, lr0 + 3.0, xtol=1e-14)
        return lifetime_optimum(p["E"] + np.exp(lr), Tl, p["E"], A_, B_, al, be, psD=psD)[:2]

    Nm_, Dm_ = _at_compute(-h)
    Np_, Dp_ = _at_compute(+h)
    N0_, D0_ = _at_compute(0.0)
    fd = (np.log(Dp_ / Np_) - np.log(Dm_ / Nm_)) / (2 * be * h)
    TD = Tl / (3 * D0_)
    th = TD / (1 + TD)
    pred = -2 / (al + be + th)
    record("L1.T.num", "new (Lemma A3(vi))", "Brute force at given training compute (C = 1e22, T = 5e12): d ln M/d chi = -2/(alpha+beta+theta_T)",
           "NEW", "numeric", abs(fd / pred - 1) < 1e-3, f"finite difference {fd:.5f} vs formula {pred:.5f} (theta_T = {th:.3f})")


def check_soc_numeric():
    # (i) Chinchilla: FOC point is a minimum of ln R along the isocost
    p = BES
    po = path_objects(p["A"], p["B"], p["alpha"], p["beta"])
    lc = np.log(1e22 / 6)
    n_s = np.log(po["G"]) + po["a"] * lc
    f = lambda nn: lnR(nn, lc - nn, p["A"], p["B"], p["alpha"], p["beta"])
    h = 1e-3
    curv_chin = (f(n_s + h) - 2 * f(n_s) + f(n_s - h)) / h ** 2
    # (ii) CES with sigma > 1 in levels: Y = (N^r + D^r)^{1/r}, r = 0.3 (sigma = 1/(1-r) = 1.43); loss R = Y^{-0.3}
    r = 0.3
    fR = lambda nn: -0.3 / r * np.logaddexp(r * nn, r * (lc - nn))
    n_sym = lc / 2
    curv_ces = (fR(n_sym + h) - 2 * fR(n_sym) + fR(n_sym - h)) / h ** 2
    corner_better = fR(lc / 2 + 5) < fR(n_sym)
    ok = curv_chin > 0 and curv_ces < 0 and corner_better
    record("SOC.num", "new", "Chinchilla (sigma<1): equal-elasticity point minimizes loss on the isocost; CES with sigma = 1.43 > 1: the same FOC point MAXIMIZES loss "
           "(corners dominate) => interior compute optima require sigma < 1",
           "NEW", "numeric", ok, f"curvature of lnR along isocost: Chinchilla {curv_chin:.3f} > 0, CES(sigma=1.43) {curv_ces:.3f} < 0")


def check_wedge_numeric():
    p = BES
    E_, A_, B_, al, be = p["E"], p["A"], p["B"], p["alpha"], p["beta"]
    worst = 0.0
    worst_inv = 0.0
    rows = []
    for Lbar in (2.2, 2.05, 1.95):
        for T in (0.0, 1e11, 1e12, 1e13, 5e13):
            for om in (0.0, 0.4):
                N_, D_, _ = lifetime_optimum(Lbar, T, E_, A_, B_, al, be, omega=om)
                eN, eD = elasticities(np.log(N_), np.log(D_), A_, B_, al, be)
                w = eN / eD
                worst = max(worst, abs(w / (1 + T / (3 * D_)) - 1))
                rows.append((Lbar, T, om, N_, D_, w))
    # invariance to E: shifting E and Lbar by the same amount leaves (N*, D*) unchanged
    N1, D1, _ = lifetime_optimum(2.1, 1e12, E_, A_, B_, al, be)
    N2, D2, _ = lifetime_optimum(2.1 + 0.3, 1e12, E_ + 0.3, A_, B_, al, be)
    worst_inv = max(abs(N1 / N2 - 1), abs(D1 / D2 - 1))
    record("P4.foc.num", "model_spec Prop 4 / SYNTHESIS P6", "Brute-force lifetime-compute minimization: eps_N/eps_D at the optimum = 1 + T/(3D) (30 cases incl. omega != 0); "
           "the econometrician's w from (N,D) needs neither omega nor E",
           "VERIFIED", "numeric", worst < 1e-5 and worst_inv < 1e-7, f"max rel err {worst:.1e}; E-shift invariance {worst_inv:.1e}")
    # contamination: lab optimizes with its own factor-augmenting productivity; econometrician uses common technology
    psN, psD, T = 0.2, 0.5, 2e12
    N_, D_, _ = lifetime_optimum(2.1, T, E_, A_, B_, al, be, psN=psN, psD=psD)
    eN, eD = elasticities(np.log(N_), np.log(D_), A_, B_, al, be)
    w_hat = eN / eD
    w_true = 1 + T / (3 * D_)
    err = abs(w_hat / (w_true * np.exp(al * psN - be * psD)) - 1)
    record("P4.contam.num", "model_spec Prop 4", "Lab with (psi_N, psi_D) = (0.2, 0.5): econometrician's w_hat = w exp(alpha psi_N - beta psi_D) (factor bias read as inference demand)",
           "VERIFIED", "numeric", err < 1e-6, f"rel err {err:.1e}; w = {w_true:.3f}, w_hat = {w_hat:.3f}")
    return pd.DataFrame(rows, columns=["Lbar", "T", "omega", "N", "D", "w"])


def check_constraints_numeric():
    p = BES
    E_, A_, B_, al, be = p["E"], p["A"], p["B"], p["alpha"], p["beta"]
    out = []
    ok = True
    for T in (0.0, 1e12):
        Lbar = 2.1
        N0, D0, C0 = lifetime_optimum(Lbar, T, E_, A_, B_, al, be)
        for frac in (0.8, 0.5, 0.2):
            Dbar = frac * D0
            N_, D_, C_ = lifetime_optimum(Lbar, T, E_, A_, B_, al, be, Dbar=Dbar)
            eN, eD = elasticities(np.log(N_), np.log(D_), A_, B_, al, be)
            w = eN / eD
            h = 1e-4
            _, _, Cp = lifetime_optimum(Lbar, T, E_, A_, B_, al, be, Dbar=Dbar * np.exp(h))
            _, _, Cm = lifetime_optimum(Lbar, T, E_, A_, B_, al, be, Dbar=Dbar * np.exp(-h))
            mu = -(np.log(Cp) - np.log(Cm)) / (2 * h)
            thD = 1 / (1 + T / (3 * D_))
            err = abs(w - 1 / (thD + mu))
            ok &= err < 1e-4 and w < 1 + T / (3 * D_) and mu > 0
            if T == 0:
                ok &= w < 1
            out.append(("data", T, frac, w, 1 + T / (3 * D_), mu, err))
        for frac in (0.8, 0.5):
            Nbar = frac * N0
            N_, D_, C_ = lifetime_optimum(Lbar, T, E_, A_, B_, al, be, Nbar=Nbar)
            eN, eD = elasticities(np.log(N_), np.log(D_), A_, B_, al, be)
            w = eN / eD
            h = 1e-4
            _, _, Cp = lifetime_optimum(Lbar, T, E_, A_, B_, al, be, Nbar=Nbar * np.exp(h))
            _, _, Cm = lifetime_optimum(Lbar, T, E_, A_, B_, al, be, Nbar=Nbar * np.exp(-h))
            nu = -(np.log(Cp) - np.log(Cm)) / (2 * h)
            err = abs(w - (1 + nu) * (1 + T / (3 * D_)))
            ok &= err < 1e-4 and w > 1 + T / (3 * D_)
            out.append(("memory", T, frac, w, 1 + T / (3 * D_), nu, err))
    df = pd.DataFrame(out, columns=["constraint", "T", "bar_frac_of_unconstrained", "w", "1+T/(3D)", "multiplier", "abs_err_formula"])
    record("D.constr.num", "task 2(d)", "Brute force with binding D <= Dbar: w = 1/(theta_D + mu) with mu = -dlnC*/dlnDbar, w < 1 when T = 0; binding N <= Nbar: w = (1+nu)(1+T/3D) > 1+T/3D",
           "VERIFIED", "numeric", ok, f"max formula err {df['abs_err_formula'].max():.1e}")
    return df


def check_general_wedge_numeric():
    rng = np.random.default_rng(2)
    worst_w, worst_ce, worst_cosh = 0.0, 0.0, 0.0
    for name, p in PARAM_SETS.items():
        A_, B_, al, be = p["A"], p["B"], p["alpha"], p["beta"]
        for _ in range(40):
            N_ = 10 ** rng.uniform(8, 12)
            D_ = 10 ** rng.uniform(10, 13.5)
            C = 6 * N_ * D_
            eN, eD = elasticities(np.log(N_), np.log(D_), A_, B_, al, be)
            w = eN / eD
            w_formula = (D_ / N_ / M_star(C, A_, B_, al, be)) ** ((al + be) / 2)
            worst_w = max(worst_w, abs(w / w_formula - 1))
            worst_ce = max(worst_ce, abs(ce_ratio_from_w(w, al, be) / ce_ratio_direct(N_, D_, A_, B_, al, be) - 1))
            if al == be:
                x = D_ / N_ / M_star(C, A_, B_, al, be)
                cosh_form = np.cosh(al / 2 * np.log(x)) ** (2 / al)
                worst_cosh = max(worst_cosh, abs(cosh_form / ce_ratio_direct(N_, D_, A_, B_, al, be) - 1))
    record("W1.num", "new", "w = (M/M*(C))^{(alpha+beta)/2} at 160 random (N,D) for 4 parameter sets (alpha != beta included)",
           "NEW", "numeric", worst_w < 1e-9, f"max rel err {worst_w:.1e}")
    record("W2.num", "new / SYNTHESIS P6", "C/C_min from the wedge-only formula equals brute-force C/C_min; cosh form exact when alpha = beta",
           "NEW", "numeric", worst_ce < 1e-9 and worst_cosh < 1e-9, f"max rel err {worst_ce:.1e} (cosh {worst_cosh:.1e})")
    # Harberger approximation accuracy (Besiroglu)
    al, be = BES["alpha"], BES["beta"]
    rows = []
    for w in (0.36, 0.5, 0.8, 1.25, 2.0, 2.62, 5.22, 7.0):
        exact = ce_ratio_from_w(w, al, be)
        approx = np.exp(np.log(w) ** 2 / (2 * (al + be)))
        rows.append((w, exact, approx))
    return pd.DataFrame(rows, columns=["w", "C_over_Cmin_exact", "C_over_Cmin_harberger"])


def check_dmr_family():
    """Explicit observationally equivalent family over S (sigma*), multi-date, factor-augmenting progress."""
    p = BES
    po = path_objects(p["A"], p["B"], p["alpha"], p["beta"])
    a, gam, G, K = po["a"], po["gamma"], po["G"], po["K"]
    gN, gD = 0.25, 0.60  # log-points per year of N- and D-augmentation
    S_list = [0.25, p["alpha"] + p["beta"], 1.5, 4.0]
    Cs = np.logspace(19, 25, 7)
    rows = []
    ref = None
    worst = 0.0
    for S in S_list:
        m = kappa_family_member(S, a, gam, G, K)
        for t in range(0, 6):
            psN, psD = gN * t, gD * t
            for C in Cs:
                n_opt, d_opt, lnr = argmin_at_compute(C, m["A"], m["B"], m["a1"], m["b1"], m["kappa"], psN=psN, psD=psD)
                rows.append(dict(S=S, t=t, C=C, n=n_opt, lnR=lnr))
    df = pd.DataFrame(rows)
    piv_n = df.pivot_table(index=["t", "C"], columns="S", values="n")
    piv_r = df.pivot_table(index=["t", "C"], columns="S", values="lnR")
    disc_n = float((piv_n.max(axis=1) - piv_n.min(axis=1)).max())   # argmin precision ~ sqrt(machine eps): tol 1e-5
    disc_r = float((piv_r.max(axis=1) - piv_r.min(axis=1)).max())   # optimal ln R: tol 1e-10
    worst = disc_r
    # kappa = 1 member coincides with Besiroglu
    m1 = kappa_family_member(p["alpha"] + p["beta"], a, gam, G, K)
    kap1 = abs(m1["kappa"] - 1) < 1e-12 and abs(m1["A"] / p["A"] - 1) < 1e-9 and abs(m1["B"] / p["B"] - 1) < 1e-9
    # what differs: sigma*, Hicks bias magnitude, off-path losses
    diffs = []
    for S in S_list:
        m = kappa_family_member(S, a, gam, G, K)
        bias = m["b1"] * gD - m["a1"] * gN
        lc = np.log(1e22 / 6)
        n_s = np.log(G) + a * lc
        off = lnR(n_s - np.log(5) / 2, lc - n_s + np.log(5) / 2, m["A"], m["B"], m["a1"], m["b1"], m["kappa"])
        # percent reducible loss above the (common) frontier at equal compute, as in Figure m7_theory_geometry(b)
        pct = 100 * (np.exp(off - (np.log(K) - gam * lc)) - 1)
        diffs.append(dict(S=S, sigma_star=m["sigma_star"], kappa=m["kappa"], hicks_bias=bias, lnR_offpath_M5x=off,
                          pct_above_frontier_M5x=pct))
    ddf = pd.DataFrame(diffs)
    same_sign = np.all(np.sign(ddf["hicks_bias"]) == np.sign(ddf["hicks_bias"].iloc[0]))
    record("P5.DMR.num", "model_spec Prop 1/5, SYNTHESIS P2/P3", "Explicit family (a1,b1,kappa) = ((1-a)S, aS, gamma/(a(1-a)S)), S in {0.25, 0.71, 1.5, 4}: identical "
           "brute-force optimal paths and frontiers at 6 dates x 7 budgets with factor-augmenting progress; sigma* in (0.33, 0.89) and Hicks-bias magnitudes differ; kappa=1 member = Besiroglu",
           "VERIFIED", "numeric", disc_r < 1e-10 and disc_n < 1e-5 and kap1 and same_sign and ddf["lnR_offpath_M5x"].std() > 1e-3,
           f"max on-path discrepancy: ln R* {disc_r:.1e}, n* {disc_n:.1e}; bias sign common across S: {same_sign}")
    return ddf


def check_drift_identity_general():
    """Reviewer addition. Prop. A3(i) claims a NONPARAMETRIC identity, d ln M*(C,t)/dt = -[sigma*/(1-sigma*)] B_t, but the
    original suite verified it only inside the kappa-family. Here: a non-separable technology outside that family,
    F = -ln sum_k c_k(t) exp(-(p_k n + q_k d)) with an interaction term (so sigma* != 2/(2+alpha+beta)), with factor-augmenting
    progress. d ln M*/dt by brute-force argmax + central differences; B_t = d ln(F_n/F_d)/dt at fixed inputs; sigma* = P/(P+Q)."""
    al, be, ph1, ph2 = 0.34, 0.37, 0.15, 0.22
    A_, B_, H_ = 480.0, 2100.0, 1000.0
    P_ = np.array([al, 0.0, ph1])
    Q_ = np.array([0.0, be, ph2])
    worst, sigs = 0.0, []
    for gN, gD in ((0.25, 0.60), (0.8, 0.1), (0.3, 0.3)):
        def logc(t):
            return np.log(np.array([A_, B_, H_])) - t * np.array([al * gN, be * gD, ph1 * gN + ph2 * gD])

        def F(n_, d_, t):
            z = logc(t) - (P_ * n_ + Q_ * d_)
            mz = z.max()
            return -(mz + np.log(np.exp(z - mz).sum()))

        def derivs(n_, d_, t):
            z = logc(t) - (P_ * n_ + Q_ * d_)
            pi = np.exp(z - z.max())
            pi /= pi.sum()
            Fn, Fd = pi @ P_, pi @ Q_
            return Fn, Fd, -(pi @ P_ ** 2 - Fn ** 2), -(pi @ Q_ ** 2 - Fd ** 2), -(pi @ (P_ * Q_) - Fn * Fd)

        for C in (1e20, 1e22, 1e24):
            lc = np.log(C / 6)

            def mstar(t):
                r = minimize_scalar(lambda nn: -F(nn, lc - nn, t), bounds=(lc / 2 - 25, lc / 2 + 25), method="bounded",
                                    options=dict(xatol=1e-13))
                return (lc - r.x) - r.x, r.x

            h = 1e-3
            dm = (mstar(h)[0] - mstar(-h)[0]) / (2 * h)
            _, n0 = mstar(0.0)
            d0 = lc - n0
            Fn, Fd, Fnn, Fdd, Fnd = derivs(n0, d0, 0.0)
            Pp = Fn * Fd * (Fn + Fd)
            Qq = -(Fnn * Fd ** 2 - 2 * Fnd * Fn * Fd + Fdd * Fn ** 2)
            sig = Pp / (Pp + Qq)
            ht = 1e-5
            dp, dmn = derivs(n0, d0, ht), derivs(n0, d0, -ht)
            Bt = (np.log(dp[0] / dp[1]) - np.log(dmn[0] / dmn[1])) / (2 * ht)
            pred = -sig / (1 - sig) * Bt
            worst = max(worst, abs(dm - pred) / max(abs(pred), 1e-12))
            sigs.append(sig)
    record("P5.drift.gen", "appendix Prop. A3(i) (reviewer)", "Nonparametric drift identity d ln M*/dt = -[sigma*/(1-sigma*)] B_t holds for a non-separable "
           "3-term technology outside the kappa-family (3 progress patterns x 3 budgets, incl. N-using, D-using and near-neutral bias)",
           "NEW", "numeric", worst < 1e-4, f"max rel err {worst:.1e}; sigma* in [{min(sigs):.3f}, {max(sigs):.3f}] (kappa-family value 2/(2+al+be) = {2 / (2 + al + be):.3f})")


def check_onpath_jacobian():
    """kappa = 1: Jacobian of ln L w.r.t. (E, lnA, lnB, alpha, beta) on the optimal path has rank 3; sigma* is first-order
    estimable on-path only if alpha = beta (unless optimality of the path is imposed)."""
    out = {}
    for label, p in [("Besiroglu", BES), ("Hoffmann", HOF), ("alpha=beta", PARAM_SETS["Muennighoff (alpha=beta)"])]:
        th = np.array([p["E"], np.log(p["A"]), np.log(p["B"]), p["alpha"], p["beta"]])
        po = path_objects(p["A"], p["B"], p["alpha"], p["beta"])
        lcs = np.linspace(np.log(1e18 / 6), np.log(1e24 / 6), 30)
        nn = np.log(po["G"]) + po["a"] * lcs
        dd = -np.log(po["G"]) + po["b"] * lcs

        def lnL(t_):
            return np.log(t_[0] + np.exp(t_[1] - t_[3] * nn) + np.exp(t_[2] - t_[4] * dd))

        J = np.zeros((len(lcs), 5))
        for k in range(5):
            h = 1e-6 * max(1.0, abs(th[k]))
            tp, tm = th.copy(), th.copy()
            tp[k] += h
            tm[k] -= h
            J[:, k] = (lnL(tp) - lnL(tm)) / (2 * h)
        sv = np.linalg.svd(J, compute_uv=False)
        rank = int(np.sum(sv / sv[0] > 1e-7))
        # is grad sigma* in the row space of J? (first-order estimability)
        al, be = p["alpha"], p["beta"]
        gsig = np.array([0, 0, 0, -2 / (2 + al + be) ** 2, -2 / (2 + al + be) ** 2])
        coef, *_ = np.linalg.lstsq(J.T, gsig, rcond=1e-8)  # truncate at numerical rank
        resid = np.linalg.norm(J.T @ coef - gsig) / np.linalg.norm(gsig)
        out[label] = (rank, sv, resid)
    ok = out["Besiroglu"][0] == 3 and out["Besiroglu"][2] > 1e-3 and out["Hoffmann"][2] > out["Besiroglu"][2] and out["alpha=beta"][2] < 1e-6
    record("P2.rank", "SYNTHESIS P2 / model_spec Prop 1", "kappa=1, runs on the optimal path: Jacobian of ln L in (E,lnA,lnB,alpha,beta) has rank 3; "
           "grad sigma* lies in its row space only when alpha = beta (else sigma* is identified on-path only by imposing path optimality)",
           "VERIFIED", "numeric", ok,
           f"rank {out['Besiroglu'][0]}, sv ratio min {out['Besiroglu'][1][-1] / out['Besiroglu'][1][0]:.1e}; rel. row-space residual of grad sigma*: "
           f"Besiroglu {out['Besiroglu'][2]:.3f}, Hoffmann {out['Hoffmann'][2]:.3f} (grows with |alpha-beta|), alpha=beta {out['alpha=beta'][2]:.1e}")


def check_ceg():
    E0 = 1.8
    x = np.logspace(18, 26, 50) / 6
    # frontier technologies
    def frontier(A_, B_, al, be, E_=E0):
        po = path_objects(A_, B_, al, be)
        return po, E_ + po["K"] * x ** (-po["gamma"])

    def Cmin(po, E_, L):
        return 6 * (po["K"] / (L - E_)) ** (1 / po["gamma"])

    po_old, _ = frontier(400.0, 1500.0, 0.30, 0.40)
    po_new, L_new = frontier(300.0, 1000.0, 0.40, 0.30)  # swapped exponents, same gamma
    f_swap = Cmin(po_old, E0, L_new) / (6 * x)
    # same exponents, different E. (Reviewer fix: the original used E_new = 1.75 < E_old, so L_new fell below E_old at large C,
    # f was undefined and this branch never ran. With E_new > E_old the old technology can always reach L_new.)
    po_new2, L_new2 = frontier(300.0, 1000.0, 0.30, 0.40, E_=1.85)
    f_E = Cmin(po_old, E0, L_new2) / (6 * x) if np.all(L_new2 > E0) else None
    po_new3, L_new3 = frontier(400.0, 1500.0, 0.35, 0.40)  # different gamma
    f_g = Cmin(po_old, E0, L_new3) / (6 * x)
    # factor augmentation: A -> A e^{-alpha psiN}, B -> B e^{-beta psiD}
    psN, psD = 0.4, 0.9
    po_aug, L_aug = frontier(400.0 * np.exp(-0.30 * psN), 1500.0 * np.exp(-0.40 * psD), 0.30, 0.40)
    f_aug = Cmin(po_old, E0, L_aug) / (6 * x)
    cv = lambda f: float(np.std(np.log(f)))
    ok = cv(f_swap) < 1e-10 and cv(f_g) > 1e-2 and abs(np.mean(np.log(f_aug)) - (psN + psD)) < 1e-9 and cv(f_aug) < 1e-10
    note = f"sd(ln f): swapped (alpha,beta) same gamma {cv(f_swap):.1e}; gamma differs {cv(f_g):.2f}; factor-aug ln f = {np.mean(np.log(f_aug)):.3f} = psi_N+psi_D"
    if f_E is not None:
        ok &= cv(f_E) > 1e-2
        note += f"; E differs {cv(f_E):.2f}"
    ok &= f_E is not None
    record("P4S.CEG.num", "SYNTHESIS P4", "Counterexample: frontier CEG is exactly constant for technologies with swapped (alpha,beta) (same gamma, same E) "
           "and non-constant when gamma or E differ; factor augmentation gives constant CEG = e^{psi_N+psi_D}",
           "FALSE", "numeric", ok, note)

    # Reviewer addition: CEG at a COMMON ALLOCATION RULE D = m N. Equal E and exponent sets are necessary but NOT sufficient:
    # constant f requires L_new(N) = L_old(lambda N) for all N, i.e. both coefficients shift by the same lambda^{-exponent}.
    m_rule = 20.0
    N_grid = np.logspace(8, 12, 40)

    def L_rule(A_, B_, al, be, N):
        return E0 + A_ * N ** -al + B_ * (m_rule * N) ** -be

    def f_rule(Ao, Bo, alo, beo, An, Bn, aln, ben):
        out = []
        for Nn in N_grid:
            Lt = L_rule(An, Bn, aln, ben, Nn)
            lNo = brentq(lambda z: L_rule(Ao, Bo, alo, beo, np.exp(z)) - Lt, np.log(Nn) - 30, np.log(Nn) + 30, xtol=1e-13)
            out.append(np.exp(2 * lNo) / Nn ** 2)    # C_old / C_new on the common ray
        return np.array(out)

    Ao, Bo, alo, beo = 400.0, 1500.0, 0.30, 0.40
    f_unequal = f_rule(Ao, Bo, alo, beo, Ao * np.exp(-alo * 0.4), Bo * np.exp(-beo * 0.9), alo, beo)   # psi_N != psi_D
    f_equal = f_rule(Ao, Bo, alo, beo, Ao * np.exp(-alo * 0.6), Bo * np.exp(-beo * 0.6), alo, beo)     # psi_N = psi_D = 0.6
    f_hicks = f_rule(Ao, Bo, alo, beo, Ao * np.exp(-0.5), Bo * np.exp(-0.5), alo, beo)                 # Hicks-neutral, alpha != beta
    ok_r = cv(f_unequal) > 1e-2 and cv(f_equal) < 1e-9 and abs(np.mean(np.log(f_equal)) - 1.2) < 1e-9 and cv(f_hicks) > 1e-2
    record("P4S.CEG.rule", "SYNTHESIS P4 / appendix Cor. A2 (reviewer)", "At a common allocation rule D = mN, equal E and exponents are necessary but not sufficient "
           "for constant CEG: factor augmentation with psi_N != psi_D, or Hicks-neutral change with alpha != beta, gives non-constant CEG; psi_N = psi_D gives e^{2 psi}",
           "FALSE", "numeric", ok_r,
           f"sd(ln f): psi_N!=psi_D {cv(f_unequal):.3f}; psi_N=psi_D {cv(f_equal):.1e} (ln f = {np.mean(np.log(f_equal)):.3f}); Hicks-neutral {cv(f_hicks):.3f}")


def check_transmission_mc(n_obs=400_000, seed=3):
    rng = np.random.default_rng(seed)
    gam = 0.178
    Vw, Ve, Vh = 0.04, 0.01, 1.0
    rows = []
    ok = True
    for regime, p1 in [("target", -1 / gam), ("exogenous", 0.0), ("funding", 2.0)]:
        om = rng.normal(0, np.sqrt(Vw), n_obs)
        eps = rng.normal(0, np.sqrt(Ve), n_obs)
        eta = rng.normal(0, np.sqrt(Vh), n_obs)
        c = p1 * om + eta
        y = gam * c + om - eps
        bf = ols(y, [c])[1]
        delta = ols(c, [y])[1]
        plim = gam + p1 * Vw / (p1 ** 2 * Vw + Vh)
        cond = (-(1 + Ve / Vw) / gam <= p1 <= 0)
        bracket = (bf - 3e-3 <= gam <= 1 / delta + 3e-3)   # tolerance for MC noise (exogenous case sits on the boundary)
        ok &= abs(bf - plim) < 5e-3 and (bracket == cond)
        rows.append(dict(regime=regime, pi1=p1, ols=bf, plim=plim, reverse=1 / delta, gamma=gam, bracket_contains_gamma=bracket,
                         bracket_condition=cond))
    df = pd.DataFrame(rows)
    ok &= bool(df.loc[df.regime == "funding", "bracket_contains_gamma"].iloc[0]) is False
    ok &= bool(df.loc[df.regime == "target", "bracket_contains_gamma"].iloc[0]) is True
    record("P2.MC", "model_spec Prop 2 / SYNTHESIS P5", "MC (n=400k): OLS slope matches plim under target / exogenous / funding rules; forward-reverse bracket contains gamma under target & exogenous, not under funding",
           "VERIFIED", "MC", ok, "; ".join(f"{r.regime}: ols {r.ols:.3f} vs plim {r.plim:.3f}, rev {r.reverse:.3f}" for r in df.itertuples()))
    # Hicks-neutral omega does not bias the path slope a; factor-biased chi correlated with c does
    p = BES
    po = path_objects(p["A"], p["B"], p["alpha"], p["beta"])
    lc = rng.normal(np.log(1e22 / 6), 2.0, 20000)
    om = 0.3 * (lc - lc.mean()) + rng.normal(0, 0.2, lc.size)
    n_ = np.log(po["G"]) + po["a"] * lc  # Hicks-neutral: omega absent
    a_hat = ols(n_, [lc])[1]
    chi = 0.05 * (lc - lc.mean()) + rng.normal(0, 0.1, lc.size)
    n_chi = n_ + chi / (p["alpha"] + p["beta"])
    a_hat_chi = ols(n_chi, [lc])[1]
    pred = po["a"] + np.cov(chi, lc)[0, 1] / ((p["alpha"] + p["beta"]) * np.var(lc, ddof=1))
    ok2 = abs(a_hat - po["a"]) < 1e-10 and abs(a_hat_chi - pred) < 1e-8
    record("P2.a", "SYNTHESIS P5 / new", "Hicks-neutral omega correlated with c leaves the OLS path slope a unbiased; factor-biased chi gives plim a_hat = a + Cov(chi,c)/((alpha+beta)Var c)",
           "VERIFIED", "MC", ok2, f"a = {po['a']:.4f}, a_hat(omega) = {a_hat:.4f}, a_hat(chi) = {a_hat_chi:.4f} vs {pred:.4f}")
    return df


def trunc_var_ratio(z):
    lam = norm.pdf(z) / norm.sf(z)
    return 1 + z * lam - lam ** 2


def check_selection_mc(n_obs=1_000_000, seed=4):
    rng = np.random.default_rng(seed)
    gam = 0.178
    out = []
    ok = True
    flags = {}
    # (i) pure selection, exogenous c, normal (log-concave) S: slope in [0, gamma]
    Vw, Ve = 0.04, 0.01
    c = rng.normal(0, 1, n_obs)
    om = rng.normal(0, np.sqrt(Vw), n_obs)
    eps = rng.normal(0, np.sqrt(Ve), n_obs)
    y = gam * c + om - eps
    for q in (0.0, 0.5, 0.9):
        ybar = np.quantile(y, q)
        s = y >= ybar
        b = ols(y[s], [c[s]])[1] if q > 0 else ols(y, [c])[1]
        out.append(dict(case="normal, exogenous c", release_quantile=q, slope=b))
        ok &= -1e-3 <= b <= gam + 3e-3
    # E[omega | c, released] decreasing in c (normal case)
    s = y >= np.quantile(y, 0.7)
    bins = np.quantile(c[s], np.linspace(0, 1, 11))
    idx = np.digitize(c[s], bins[1:-1])
    m_om = np.array([om[s][idx == k].mean() for k in range(10)])
    flags["E[omega|c,rel] decreasing"] = bool(np.all(np.diff(m_om) < 0))
    flags["normal slopes in [0,gamma]"] = bool(ok)
    # (ii) non-log-concave S (bimodal omega): OLS slope still <= gamma; local CEF slope can be negative
    om2 = np.where(rng.random(n_obs) < 0.5, -0.6, 0.6) + rng.normal(0, 0.05, n_obs)
    y2 = gam * c + om2 - eps
    s2 = y2 >= -0.56   # truncation points k = ybar - gamma c fall where m'(k) > 1 for this bimodal S
    b2 = ols(y2[s2], [c[s2]])[1]
    bins2 = np.linspace(-1.5, 1.5, 31)
    idx2 = np.digitize(c[s2], bins2)
    cef = np.array([y2[s2][idx2 == k].mean() if np.sum(idx2 == k) > 500 else np.nan for k in range(1, 31)])
    local_neg = np.nanmin(np.diff(cef)) < 0
    out.append(dict(case="bimodal omega (not log-concave), exogenous c", release_quantile=np.mean(~s2), slope=b2))
    flags["bimodal slope <= gamma"] = bool(b2 <= gam + 3e-3)
    flags["bimodal local CEF slope < 0"] = bool(local_neg)
    ok3 = True
    # (iii) Gaussian with funding rule: closed form for the selected-sample forward slope; reverse slope unaffected
    Vh = 1.0
    rows3 = []
    for p1 in (2.0,):
        om = rng.normal(0, np.sqrt(Vw), n_obs)
        eta = rng.normal(0, np.sqrt(Vh), n_obs)
        c3 = p1 * om + eta
        y3 = gam * c3 + om - eps
        Vc = p1 ** 2 * Vw + Vh
        Cov = gam * Vc + p1 * Vw
        Vy = gam ** 2 * Vc + 2 * gam * p1 * Vw + Vw + Ve
        delta = Cov / Vy
        W = Vc - delta ** 2 * Vy
        rstar = gam * W / (delta * Vy * (1 - gam * delta))
        for z in (-3.0, 0.0, 1.0, 2.0):
            ybar = z * np.sqrt(Vy)
            s3 = y3 >= ybar
            bf = ols(y3[s3], [c3[s3]])[1]
            dl = ols(c3[s3], [y3[s3]])[1]
            r = trunc_var_ratio(z)
            bf_pred = delta * r * Vy / (delta ** 2 * r * Vy + W)
            rows3.append(dict(z=z, r=r, rstar=rstar, forward=bf, forward_pred=bf_pred, reverse_slope=dl, delta=delta,
                              net_bias_sign=np.sign(bf - gam), pred_sign=np.sign(r - rstar)))
            ok3 &= abs(bf - bf_pred) < 4e-3 and abs(dl / delta - 1) < 0.02 and np.sign(bf - gam) == np.sign(r - rstar)
    df3 = pd.DataFrame(rows3)
    signs_both = set(df3["net_bias_sign"]) == {-1.0, 1.0}
    ok3 &= signs_both
    # (iv) counterexample to 'E[omega | c, released] decreasing in c' under the funding rule with mild selection
    s4 = y3 >= -2.0 * np.sqrt(Vy)
    bins4 = np.quantile(c3[s4], np.linspace(0, 1, 11))
    idx4 = np.digitize(c3[s4], bins4[1:-1])
    m4 = np.array([om[s4][idx4 == k].mean() for k in range(10)])
    increasing = np.all(np.diff(m4) > 0)
    record("P3.sel", "model_spec Prop 3", "Release iff y >= ybar, c independent of omega: OLS slope <= gamma for ANY distribution; in [0, gamma] if omega - eps is log-concave; "
           "E[omega|c,released] decreasing (normal case)",
           "VERIFIED", "MC", all(flags.values()), f"normal slopes {[round(float(r['slope']), 3) for r in out[:3]]}; bimodal slope {b2:.3f}; " + "; ".join(f"{k}: {v}" for k, v in flags.items()))
    record("P3.net", "model_spec Prop 3 (task)", "Gaussian model, funding rule pi1 > 0: selected forward slope = delta r V_y/(delta^2 r V_y + V_{c|y}), reverse slope unaffected; "
           "net bias > 0 iff r > r* = gamma V_{c|y}/(delta V_y (1 - gamma delta)); both signs occur",
           "NEW", "MC", ok3, f"r* = {df3['rstar'].iloc[0]:.3f}; " + "; ".join(f"z={r.z:+.0f}: r={r.r:.2f} b={r.forward:.3f}" for r in df3.itertuples()))
    record("P3.cex", "model_spec Prop 3", "Counterexample: with a funding rule and mild selection E[omega | c, released] is INCREASING in c; "
           "the correct statement is about the selection component E[omega|c,rel] - E[omega|c]",
           "IMPRECISE", "MC", increasing, f"binned E[omega|c,rel] increasing: {increasing}")
    return pd.DataFrame(out), df3


def check_sahal_mc(seed=5):
    rng = np.random.default_rng(seed)
    gam, g, gA = 0.178, 1.2, 0.8
    t = np.linspace(0, 10, 200_000)
    rows = []
    ok = True
    for su in (0.0, 0.5, 2.0):
        c = 3 + g * t + rng.normal(0, su, t.size)
        lnR_ = -gam * (c + gA * t) + rng.normal(0, 0.01, t.size)
        slope = -ols(lnR_, [c])[1]
        R2 = g ** 2 * np.var(t) / (g ** 2 * np.var(t) + su ** 2)
        pred = gam * (1 + gA / g * R2)
        rows.append((su, slope, pred))
        ok &= abs(slope - pred) < 2e-3
    sA = gA / (g + gA)
    ok &= abs(rows[0][2] - gam / (1 - sA)) < 1e-12
    record("P7.MC", "SYNTHESIS P7", "Sahal: naive slope = gamma/(1-s_A) with deterministic trends; general gamma[1 + (g_A/g) R^2_{c,t}] with compute noise around trend",
           "VERIFIED", "MC", ok, "; ".join(f"sd_u={r[0]}: {r[1]:.4f} vs {r[2]:.4f}" for r in rows))


def check_hall_numeric():
    p = BES
    A_, B_, al, be = p["A"], p["B"], p["alpha"], p["beta"]
    n0, d0 = np.log(8e9), np.log(2e12)
    errs = []
    for h in (0.2, 0.1, 0.05, 0.025):
        dn, dd = 0.5 * h, 2.0 * h
        y0 = -lnR(n0, d0, A_, B_, al, be)
        y1 = -lnR(n0 + dn, d0 + dd, A_, B_, al, be)
        eN, eD = elasticities(n0, d0, A_, B_, al, be)
        resid = (y1 - y0) - (eN + eD) / 2 * (dn + dd)  # true productivity change is 0
        pred = (eN - eD) * (dn - dd) / 2
        errs.append(abs(resid - pred))
    order = np.log(errs[0] / errs[-1]) / np.log(8)
    record("P8.num", "SYNTHESIS P8", "Hall-type bias (eps_N-eps_D)(dn-dd)/2 matches the equal-weight residual up to a second-order error (Llama-like over-trained point)",
           "VERIFIED", "numeric", 1.8 < order < 2.2, f"error order in step size = {order:.2f}")


# ---------------------------------------------------------------------------- Fisher information (task 2a)
def fisher_info_scaling():
    """Efficient Fisher information for sigma* (kappa-free family) and for ln M* (path location) as a function of the
    dispersion v of transverse deviations x = ln(M/M*(C)).  Noise: y = -ln(L-E) + eps, eps ~ N(0,1) per run."""
    p = BES
    po = path_objects(p["A"], p["B"], p["alpha"], p["beta"])
    phi0 = np.array([p["alpha"] + p["beta"], po["a"], po["gamma"], np.log(po["G"]), np.log(po["K"])])
    lcs = np.log(np.logspace(18, 22, 9) / 6)
    grid_x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

    def design(v):
        rows = []
        for lc in lcs:
            ns = phi0[3] + phi0[1] * lc
            ds = lc - ns
            for x in v * grid_x:
                rows.append((ns - x / 2, ds + x / 2, lc))
        return np.array(rows)

    def y_of(phi, X):
        S, a, gam, lG, lK = phi
        m = kappa_family_member(S, a, gam, np.exp(lG), np.exp(lK))
        return -lnR(X[:, 0], X[:, 1], m["A"], m["B"], m["a1"], m["b1"], m["kappa"])

    def jac(X):
        J = np.zeros((len(X), 5))
        for k in range(5):
            h = 1e-5 * max(1.0, abs(phi0[k]))
            pp, pm = phi0.copy(), phi0.copy()
            pp[k] += h
            pm[k] -= h
            J[:, k] = (y_of(pp, X) - y_of(pm, X)) / (2 * h)
        return J

    def eff_info(J, k):
        I = J.T @ J
        o = [j for j in range(J.shape[1]) if j != k]
        return I[k, k] - I[k, o] @ np.linalg.solve(I[np.ix_(o, o)], I[o, k])

    rows = []
    for v in np.logspace(-1.3, 0.3, 9):
        X = design(v)
        J = jac(X)
        I_S = eff_info(J, 0)
        I_G = eff_info(J, 3)
        # leading-order prediction for I_S: (gamma/S)^2 ||M_Z Phi||^2 with Phi = ln(C/C_min), Z = nuisance scores
        x = X[:, 1] - X[:, 0] - (-2 * phi0[3] + (1 - 2 * phi0[1]) * X[:, 2])  # ln(M/M*(C))
        w = np.exp((phi0[0] / 2) * x)
        Phi = np.log(ce_ratio_from_w(w, p["alpha"], p["beta"]))
        Z = J[:, 1:]
        s_pred = -(phi0[2] / phi0[0]) * Phi
        res = s_pred - Z @ np.linalg.lstsq(Z, s_pred, rcond=None)[0]
        dsig_dS = -2 / (2 + phi0[0]) ** 2
        # leading-order prediction for I_lnG: gamma^2 ||M_{-G} ln w||^2 (score for ln G ~ -gamma ln w)
        ZG = J[:, [0, 1, 2, 4]]
        g_pred = -phi0[2] * np.log(w)
        resG = g_pred - ZG @ np.linalg.lstsq(ZG, g_pred, rcond=None)[0]
        # Reviewer fix: ln M*(C) = -2 lnG + (1-2a)(c - ln6). The efficient information for lnG (a nuisance) is the information
        # about ln M* at c - ln6 = 0, i.e. C = 6 FLOP, some 44 log-units outside the design, so 2/sqrt(I_lnG) greatly overstates
        # the s.e. of the compute-optimal ratio at relevant budgets. Report also ln M* at the design's central compute
        # (geometric mean of the 9 budgets, 1e20 FLOP), via the full inverse information matrix (delta method).
        lcbar = float(lcs.mean())
        gM = np.array([0.0, -2.0 * lcbar, 0.0, -2.0, 0.0])
        I_full = J.T @ J
        I_mc = 1.0 / float(gM @ np.linalg.solve(I_full, gM))
        # leading-order check in the (S, a, gamma, m_c, lnK) parameterization: score for m_c ~ +(gamma/2) ln w
        Zc = np.column_stack([J[:, 0], J[:, 1] - lcbar * J[:, 3], J[:, 2], J[:, 4]])
        m_pred = 0.5 * phi0[2] * np.log(w)
        resM = m_pred - Zc @ np.linalg.lstsq(Zc, m_pred, rcond=None)[0]
        rows.append(dict(v=v, var_x=np.var(x), sd_lnw=(phi0[0] / 2) * np.sqrt(np.var(x)), I_S=I_S, I_S_pred=float(res @ res),
                         I_lnG=I_G, I_lnG_pred=float(resG @ resG), I_lnMstar_center=I_mc, I_lnMstar_center_pred=float(resM @ resM),
                         se_sigma_star=abs(dsig_dS) / np.sqrt(I_S), se_lnMstar_at_C6=2 / np.sqrt(I_G),
                         se_lnMstar_center=1 / np.sqrt(I_mc), C_center=6 * np.exp(lcbar)))
    df = pd.DataFrame(rows)
    sl_S = np.polyfit(np.log(df["v"][:5]), np.log(df["I_S"][:5]), 1)[0]
    sl_G = np.polyfit(np.log(df["v"][:5]), np.log(df["I_lnG"][:5]), 1)[0]
    sl_Mc = np.polyfit(np.log(df["v"][:5]), np.log(df["I_lnMstar_center"][:5]), 1)[0]
    ratio = float(df["I_S"].iloc[0] / df["I_S_pred"].iloc[0])
    ratio_G = float(df["I_lnG"].iloc[0] / df["I_lnG_pred"].iloc[0])
    ratio_Mc = float(df["I_lnMstar_center"].iloc[0] / df["I_lnMstar_center_pred"].iloc[0])
    ok = (abs(sl_S - 4) < 0.15 and abs(sl_G - 2) < 0.15 and abs(sl_Mc - 2) < 0.15 and abs(ratio - 1) < 0.05 and abs(ratio_G - 1) < 0.05
          and abs(ratio_Mc - 1) < 0.05)
    record("INFO.num", "new (task 2a)", "Efficient Fisher information (9 budgets x 5 transverse offsets): I(sigma*) ~ v^4 (~ sum of squared allocative losses), "
           "I(ln M*) ~ v^2 (~ variance of transverse deviations = log wedges), for the path intercept and for ln M* at the design's central compute; "
           "both vanish on the expansion path",
           "NEW", "numeric", ok, f"log-log slopes: sigma* {sl_S:.2f} (theory 4), ln G {sl_G:.2f}, ln M*(C_center) {sl_Mc:.2f} (theory 2); exact/leading-order info "
           f"at smallest v: sigma* {ratio:.3f}, ln G {ratio_G:.3f}, ln M*(C_center) {ratio_Mc:.3f}")
    # on-path: exactly singular (kappa-free, S direction)
    X0 = design(0.0)
    J0 = jac(X0)
    sv = np.linalg.svd(J0, compute_uv=False)
    record("INFO.onpath", "model_spec Prop 1 / new", "On-path design: information matrix for (S, a, gamma, lnG, lnK) is singular (S and lnG/a directions have zero information)",
           "VERIFIED", "numeric", sv[-1] / sv[0] < 1e-6, f"singular values ratio {sv[-1] / sv[0]:.1e}")
    return df


# ---------------------------------------------------------------------------- partial identification (task 2b)
def check_monotone_wedge(seed=6):
    """Concavity + supermodularity of y in (n,d) => w nonincreasing in n, nondecreasing in d (product order)."""
    rng = np.random.default_rng(seed)
    viol = 0
    total = 0
    for _ in range(300):
        S = rng.uniform(0.1, 3.0)
        a = rng.uniform(0.2, 0.8)
        kap = rng.uniform(0.2, 3.0)
        A_, B_ = np.exp(rng.uniform(0, 10)), np.exp(rng.uniform(0, 10))
        a1, b1 = (1 - a) * S, a * S
        nb, db = rng.uniform(15, 25), rng.uniform(15, 30)
        n0, d0 = nb - rng.uniform(0, 3), db + rng.uniform(0, 3)
        wb = np.divide(*elasticities(nb, db, A_, B_, a1, b1, kap))
        w0 = np.divide(*elasticities(n0, d0, A_, B_, a1, b1, kap))
        total += 1
        viol += w0 < wb - 1e-12
    # counterexample without supermodularity: concave translog F = .05n + .3d - .025n^2 - .005d^2 - .02nd (F_nd < 0)
    def w_tl(n_, d_):
        return (0.05 - 0.05 * n_ - 0.02 * d_) / (0.3 - 0.01 * d_ - 0.02 * n_)
    hess = np.array([[-0.05, -0.02], [-0.02, -0.01]])
    concave = bool(np.all(np.linalg.eigvalsh(hess) <= 0))
    cex = concave and w_tl(0.0, 0.5) < w_tl(0.0, 0.0)  # more data at the same n LOWERS w
    record("PI.mono", "new (task 2b)", "Concave + supermodular (F_nd >= 0) technology: w(n0,d0) >= w(nb,db) whenever n0 <= nb, d0 >= db (300 random kappa-family draws); "
           "fails without supermodularity (translog counterexample)",
           "NEW", "numeric", viol == 0 and cex,
           f"violations {viol}/{total}; translog counterexample found: {cex}")


def partial_id_illustration(B_boot=300, seed=7):
    """Bootstrap the Besiroglu fit on the Epoch Chinchilla extraction; compute T/D bands at fixed compute as a function of M
    and verify that the half-width grows linearly in |ln M - ln M_design| (Prop. PI)."""
    import os
    import sl
    from common import ROOT
    df = sl.chinchilla_extraction(path=os.path.join(ROOT, "data/raw/epoch_chinchilla/svg_extracted_data.csv"))
    N_, D_, L_ = df["N"].values, df["D"].values, df["L"].values
    init = sl.BESIROGLU.theta
    full = sl.fit_chinchilla(N_, D_, L_, init=init)
    full_grid = sl.fit_chinchilla(N_, D_, L_, grid=sl.FAST_GRID)
    if full_grid.extra["objective"] < full.extra["objective"] - 1e-12:
        full = full_grid
    rng = np.random.default_rng(seed)
    boots = []
    n = len(N_)
    for b in range(B_boot):
        idx = rng.integers(0, n, n)
        m = sl.fit_chinchilla(N_[idx], D_[idx], L_[idx], init=full.theta)
        boots.append(m)
    C = 6 * 8e9 * 15e12  # Llama-3-8B-sized compute (7.2e23)
    Ms = np.unique(np.r_[np.logspace(0, 4, 81), [20.0, 341.0, 1875.0, 5000.0]])  # grid + table points
    Nm = np.sqrt(C / 6 / Ms)
    Dm = Ms * Nm
    TD_full = 3 * (full.wedge(Nm, Dm) - 1)
    TD_b = np.array([3 * (m.wedge(Nm, Dm) - 1) for m in boots])
    lnw_b = np.log(np.array([m.wedge(Nm, Dm) for m in boots]))
    se_lnw = lnw_b.std(axis=0, ddof=1)
    # delta-method style: ln w = ((alpha+beta)/2)(ln M - ln M*(C)); far from M*, slope of se in ln M -> sd(alpha+beta)/2
    S_b = np.array([m.alpha + m.beta for m in boots])
    Mstar_b = np.array([M_star(C, m.A, m.B, m.alpha, m.beta) for m in boots])
    M_design_max = float((D_ / N_).max())
    # ln w_b(x) = (S_b/2)(x - m*_b), x = ln M: Var_b(ln w) is exactly quadratic in x with leading coefficient Var(S)/4,
    # so the half-width grows (asymptotically) linearly in the log-extrapolation distance with slope z sd(S)/2.
    x = np.log(Ms)
    quad = np.polyfit(x, se_lnw ** 2, 2)
    pred_lead = S_b.var(ddof=1) / 4
    lo, hi = np.percentile(TD_b, [2.5, 97.5], axis=0)
    out = pd.DataFrame(dict(M=Ms, T_over_D=TD_full, lo95=lo, hi95=hi, se_ln_w=se_lnw))
    # variance decomposition at M = 1875 (Llama-3-8B): M*-level part vs curvature (S) part
    x0 = np.log(1875.0)
    m_b = np.log(Mstar_b)
    part_S = np.var(S_b / 2 * (x0 - m_b.mean()), ddof=1)
    part_M = np.var(S_b.mean() / 2 * m_b, ddof=1)
    tot = np.var(S_b / 2 * (x0 - m_b), ddof=1)
    ok = abs(quad[0] / pred_lead - 1) < 1e-6
    record("PI.band", "new (task 2b)", "Bootstrap (B=300) of the Besiroglu fit: Var(ln w_hat) at fixed compute is exactly quadratic in ln M with leading coefficient Var(alpha+beta)/4 "
           "(bands widen linearly in log-extrapolation distance); at M = 1875 the M*-level uncertainty dominates the curvature (sigma*) uncertainty",
           "NEW", "numeric (bootstrap)", ok, f"lead coef ratio {quad[0] / pred_lead:.6f}; at M=1875: sd(ln w) {np.sqrt(tot):.3f}, "
           f"from M* alone {np.sqrt(part_M):.3f}, from alpha+beta alone {np.sqrt(part_S):.3f}")
    # Reviewer additions. (1) Bootstrap health: replicates are warm-started from the full-sample optimum (no multi-start);
    # a reviewer diagnostic re-fitted 30 of the 300 draws from the 432-point FAST_GRID and found the same optimum (objective
    # differences <= 1.4e-10, |dS| <= 5e-5). Here we report the spread and flag gross outliers (|z| > 6 around the median/MAD).
    def _mad_flags(v):
        med = np.median(v)
        mad = 1.4826 * np.median(np.abs(v - med))
        return int(np.sum(np.abs(v - med) > 6 * mad))
    n_flag = max(_mad_flags(S_b), _mad_flags(np.log(Mstar_b)))
    # (2) Design support in COMPUTE: the Chinchilla extraction reaches M = 341 but only at C <= 1.3e22 FLOP, so at C = 7.2e23
    # every M is a compute extrapolation (the M-range is not a support at that budget).
    C_design_max = float((6 * N_ * D_).max())
    # (3) Nonparametric lower bound (Prop. A9(ii)) for the Llama-3-8B-sized model (N0, D0) = (8e9, 15e12): best support point
    # with N_b >= N0 and D_b <= D0. Wedges at support points are evaluated with the full-sample Huber fit (a stand-in for a
    # local nonparametric estimate on the support).
    N0, D0 = 8e9, 15e12
    dom = (N_ >= N0) & (D_ <= D0)
    wb = full.wedge(N_[dom], D_[dom]) if dom.any() else np.array([np.nan])
    wb_bes = sl.BESIROGLU.wedge(N_[dom], D_[dom]) if dom.any() else np.array([np.nan])
    summary = dict(E=full.E, A=full.A, B=full.B, alpha=full.alpha, beta=full.beta, objective_refit=float(full.extra["objective"]),
                   sd_alpha_plus_beta=float(S_b.std(ddof=1)), sd_ln_Mstar=float(np.log(Mstar_b).std(ddof=1)),
                   Mstar_full=float(M_star(C, full.A, full.B, full.alpha, full.beta)), M_design_max=M_design_max,
                   C_design_max=C_design_max, B_boot=B_boot, n_boot_outliers_6mad=n_flag,
                   S_boot_min=float(S_b.min()), S_boot_max=float(S_b.max()),
                   sd_lnw_at_1875=float(np.sqrt(tot)), sd_lnw_Mstar_part=float(np.sqrt(part_M)), sd_lnw_S_part=float(np.sqrt(part_S)),
                   n_dominating_support_points=int(dom.sum()), wb_max_refit=float(np.nanmax(wb)), wb_max_besiroglu=float(np.nanmax(wb_bes)))
    return out, summary


# ---------------------------------------------------------------------------- proxy (task 2c)
def check_proxy_numeric(seed=8):
    rng = np.random.default_rng(seed)
    p = BES
    A_, B_, al, be = p["A"], p["B"], p["alpha"], p["beta"]
    n_labs = 400
    lc = rng.uniform(np.log(1e20 / 6), np.log(1e24 / 6), n_labs)
    om = rng.normal(0, 0.3, n_labs)
    psN = rng.normal(0, 0.3, n_labs)
    psD = rng.normal(0, 0.3, n_labs)
    lnM = np.empty(n_labs)
    for i in range(n_labs):
        n_opt, d_opt, _ = argmin_at_compute(6 * np.exp(lc[i]), A_, B_, al, be, psN=psN[i], psD=psD[i], omega=om[i])
        lnM[i] = d_opt - n_opt
    chi = be * psD - al * psN
    coef = ols(lnM, [lc, chi, om])
    ok = abs(coef[2] + 2 / (al + be)) < 1e-6 and abs(coef[3]) < 1e-6
    # ACF functional dependence: (n, d) is an exact function of (c, ln M)
    record("PROXY.num", "new (task 2c)", "Simulated cost-minimizing labs (brute force): ln(D/N) = m0(c) - 2 chi/(alpha+beta) exactly; coefficient on Hicks-neutral omega = 0",
           "NEW", "numeric", ok, f"coef chi {coef[2]:.5f} (theory {-2 / (al + be):.5f}); coef omega {coef[3]:.1e}")


# ---------------------------------------------------------------------------- ledger cross-checks (SYNTHESIS §2)
def ledger_checks():
    rows = []

    def add(item, computed, ledger, tol):
        ok = abs(computed - ledger) <= tol
        rows.append(dict(item=item, computed=computed, ledger=ledger, tol=tol, ok=ok))

    L = {"Hoffmann (rounded)": (0.452, 0.1535, 0.763, 50, 93, 139),
         "Hoffmann (TeX)": (0.457, 0.1548, 0.762, 34, 59, 85),
         "Besiroglu": (0.513, 0.1783, 0.737, 21.6, 18.4, 16.5)}
    for name, (a, g, s, m1, m2, m3) in L.items():
        p = PARAM_SETS[name]
        po = path_objects(p["A"], p["B"], p["alpha"], p["beta"])
        add(f"{name}: a", po["a"], a, 6e-4)
        add(f"{name}: gamma", po["gamma"], g, 6e-4)
        add(f"{name}: sigma*", po["sigma_star"], s, 6e-4)
        for C, mm in zip((1e21, 5.76e23, 3.8e25), (m1, m2, m3)):
            add(f"{name}: M* at {C:.2e}", float(M_star(C, p["A"], p["B"], p["alpha"], p["beta"])), mm, 0.6 if mm < 30 else 1.0)
    # wedge table (Besiroglu)
    models = {"Chinchilla 70B": (70e9, 1.4e12, 1.03, 0.09, 1.00), "Gopher 280B": (280e9, 300e9, 0.36, -1.91, 2.01),
              "GPT-3 175B": (175e9, 300e9, 0.43, -1.72, 1.64), "Llama-2 7B": (7e9, 2e12, 2.62, 4.85, 1.86),
              "Llama-3 8B": (8e9, 15e12, 5.22, 12.65, 5.51), "Llama-3 70B": (70e9, 15e12, 2.45, 4.36, 1.72),
              "Llama-3.1 405B": (405e9, 15.6e12, 1.35, 1.06, 1.07)}
    p = PARAM_SETS["Besiroglu"]
    for name, (N_, D_, w_l, td_l, ce_l) in models.items():
        eN, eD = elasticities(np.log(N_), np.log(D_), p["A"], p["B"], p["alpha"], p["beta"])
        w = eN / eD
        add(f"Besiroglu w, {name}", w, w_l, 0.006)
        add(f"Besiroglu T/D, {name}", 3 * (w - 1), td_l, 0.02)
        add(f"Besiroglu C/C_min, {name}", ce_ratio_from_w(w, p["alpha"], p["beta"]), ce_l, 0.011)
    # homothetic closed-form table (rho = 0.3527, M* = 20): Llama-3 8B / 70B / 405B
    rho = 0.3527
    for name, M, ledger in (("8B", 15e12 / 8e9, 11.9), ("70B", 15e12 / 70e9, 3.9), ("405B", 15.6e12 / 405e9, 0.8)):
        add(f"Homothetic T/D (rho=.3527, M*=20), Llama-3 {name}", 3 * ((M / 20) ** rho - 1), ledger, 0.06)
    add("Homothetic C/C_min (rho=.3527, M*=20), Llama-3 8B", np.cosh(rho / 2 * np.log(1875 / 20)) ** (2 / rho), 5.2, 0.06)
    # Meta's own law, rho = 0.35 at each model's own compute
    for name, N_, D_, ledger_M, ledger_td in (("8B", 8e9, 15e12, 31.3, 9.6), ("70B", 70e9, 15e12, 36.7, 2.6), ("405B", 405e9, 15.6e12, 41.9, -0.1)):
        C = 6 * N_ * D_
        Dst = 0.299 * C ** 0.537
        Mst = Dst / (C / 6 / Dst)
        add(f"Meta law M*(C), Llama-3 {name}", Mst, ledger_M, 0.06)
        add(f"Meta law T/D (rho=.35), Llama-3 {name}", 3 * ((D_ / N_ / Mst) ** 0.35 - 1), ledger_td, 0.06)
    for sA, f in ((0.05, 1.05), (0.40, 1.67)):
        add(f"Sahal inflation 1/(1-s_A), s_A={sA}", 1 / (1 - sA), f, 0.006)
    for name, g in (("Hoffmann TeX", path_objects(**{k: HOF[k] for k in ('A', 'B')}, al=HOF['alpha'], be=HOF['beta'])['gamma']),
                    ("Besiroglu", path_objects(BES['A'], BES['B'], BES['alpha'], BES['beta'])['gamma'])):
        add(f"41^gamma ({name})", 41 ** g, 1.78 if "Hoff" in name else 1.94, 0.006)
    df = pd.DataFrame(rows)
    record("LEDGER", "SYNTHESIS §2", f"Recomputed {len(df)} ledger numbers that are pure applications of the verified formulas",
           "VERIFIED", "numeric", bool(df["ok"].all()), f"{int(df['ok'].sum())}/{len(df)} within rounding tolerance")
    return df


def run_all():
    out = {}
    check_path_frontier()
    check_sigma_numeric()
    check_lemma1_numeric()
    check_soc_numeric()
    out["wedge_cases"] = check_wedge_numeric()
    out["constraints"] = check_constraints_numeric()
    out["harberger"] = check_general_wedge_numeric()
    out["dmr_family"] = check_dmr_family()
    check_drift_identity_general()
    check_onpath_jacobian()
    check_ceg()
    out["transmission"] = check_transmission_mc()
    out["selection_i"], out["selection_net"] = check_selection_mc()
    check_sahal_mc()
    check_hall_numeric()
    out["fisher"] = fisher_info_scaling()
    check_monotone_wedge()
    out["pi_band"], out["pi_summary"] = partial_id_illustration()
    check_proxy_numeric()
    out["ledger"] = ledger_checks()
    return out
