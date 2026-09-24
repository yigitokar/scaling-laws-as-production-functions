"""Brute-force numerical and Monte Carlo verification of the Theory-v2 results.

No closed form under test is used to generate the numbers it is compared with:
  * developer problems are solved by direct maximization of the objective over (ln N, ln D) (Nelder-Mead + BFGS polish);
    constraint multipliers are envelope derivatives of the optimal value computed by central differences;
  * elasticities of substitution are computed by tracing isoquants; derivatives by finite differences;
  * identification geometry by profiling the least-squares criterion; rates by Monte Carlo;
  * Fisher information by finite-difference score matrices.
All randomness uses np.random.default_rng with fixed seeds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize, minimize_scalar

from ra5common import (BES, HOF, LN6, MUE, chin_w, chin_y, elasticities, kappa_family_member, lnR, nonsep_L, path_objects,
                       record)

PSI = 1e22        # value scale: Psi * y, in training-FLOP units (training price = 1 per FLOP at N0)
N0 = 1e9


# ============================================================================ generic brute-force optimizer
def _argmax2(f, x0):
    """Maximize f(n, d) by Nelder-Mead from x0 then BFGS polish; returns (n, d, fmax)."""
    g = lambda z: -f(z[0], z[1])
    r = minimize(g, np.asarray(x0, float), method="Nelder-Mead", options=dict(xatol=1e-12, fatol=1e-18, maxiter=40000, maxfev=80000))
    r2 = minimize(g, r.x, method="BFGS", options=dict(gtol=1e-13))
    z = r2.x if r2.fun <= r.fun else r.x
    return z[0], z[1], -min(r.fun, r2.fun)


def _argmax1(f, lo, hi):
    r = minimize_scalar(lambda z: -f(z), bounds=(lo, hi), method="bounded", options=dict(xatol=1e-13))
    return r.x, -r.fun


def _start(C=2e21, p=BES):
    po = path_objects(p["A"], p["B"], p["alpha"], p["beta"])
    lc = np.log(C / 6)
    n = np.log(po["G"]) + po["a"] * lc
    return np.array([n, lc - n])


# ============================================================================ Prop. A8: generalized wedge, cases
def _case_specs():
    """Each spec: V(y, n) in units of PSI (value of quality y and of size at fixed quality), training price c_T(n) with
    elasticity delta, per-token data cost cD (FLOP units per token), optional linear constraint (g_n, g_d, shift)."""
    specs = {}
    T0 = 2e12
    # baseline: lifetime compute (Sardana et al.), lambda = p = phi = 1, T fixed
    specs["base"] = dict(label="Lifetime compute (lambda=p=1)", V=lambda y, n: y - 2 * np.exp(n) * T0 / PSI, delta=0.0, cD=0.0, T=T0)
    # (a) internalization lambda, price ratio p0, serving-cost elasticity phi, demand elasticity eta, quality response zeta,
    #     and a direct revenue effect of size at fixed quality
    lam, p0, phi, eta, zeta = 0.6, 3.0, 0.85, -0.3, 0.4
    cS = lambda n: p0 * np.exp((phi - 1) * (n - np.log(N0)))
    Tq = lambda y, n: T0 * np.exp(eta * (n - np.log(N0))) * np.exp(zeta * (y - 0.6))
    kR = 0.004
    specs["a"] = dict(label="(a) internalized serving, p, phi, eta, R_n", V=lambda y, n: y - lam * cS(n) * 2 * np.exp(n) * Tq(y, n) / PSI
                      - kR * np.exp(0.5 * (n - np.log(N0))), delta=0.0, cD=0.0,
                      mN=lambda n, d, y: (lam * cS(n) * 2 * np.exp(n) * Tq(y, n) * (phi + eta) / PSI
                                          + 0.5 * kR * np.exp(0.5 * (n - np.log(N0)))) / (6 * np.exp(n + d) / PSI),
                      w_closed=lambda n, d, y: 1 + lam * cS(n) * (phi + eta) * Tq(y, n) / (3 * np.exp(d))
                      + 0.5 * kR * np.exp(0.5 * (n - np.log(N0))) * PSI / (6 * np.exp(n + d)))
    # (a') adoption-maximizing open release: V = y + (Psi_A/Psi)(theta*y - eps_A*ln(user serving cost per token))
    psiA, th, epsA = 0.35, 0.5, 0.8
    specs["adopt"] = dict(label="(a') adoption-maximizing open release", V=lambda y, n: y + psiA * (th * y - epsA * (n + np.log(2 * 1.5))),
                          delta=0.0, cD=0.0)
    # (b) binding memory / tier cap on N
    specs["b"] = dict(label="(b) binding memory/tier cap N <= Nbar", V=specs["base"]["V"], delta=0.0, cD=0.0, T=T0,
                      con=(1.0, 0.0, -0.7))
    # (c) latency SLO tau = tau0 N^0.6 <= taubar  (a cap on 0.6 n)
    specs["c"] = dict(label="(c) latency SLO (tau ~ N^0.6)", V=specs["base"]["V"], delta=0.0, cD=0.0, T=T0, con=(0.6, 0.0, -0.4))
    # (c') soft latency cost per token proportional to N^0.6 (a value of compactness)
    kl = 0.8
    specs["c_soft"] = dict(label="(c') latency cost per token", V=lambda y, n: y - 2 * np.exp(n) * T0 / PSI
                           - kl * T0 * np.exp(0.6 * (n - np.log(N0))) / PSI * 1e9, delta=0.0, cD=0.0)
    # (d) per-token data cost (FLOP units per token) and a binding data cap
    specs["d"] = dict(label="(d) per-token data cost c_D", V=specs["base"]["V"], delta=0.0, cD=2.0e10, T=T0)
    specs["d_cap"] = dict(label="(d') binding data cap D <= Dbar", V=specs["base"]["V"], delta=0.0, cD=0.0, T=T0, con=(0.0, 1.0, -0.8))
    # (e) logit distillation with a teacher of N_T parameters: extra training cost 2 N_T D, i.e. c_D = 2 N_T
    NT = 7e10
    specs["e"] = dict(label="(e) logit distillation (teacher 70B)", V=specs["base"]["V"], delta=0.0, cD=2 * NT, T=T0, NT=NT)
    # (f) MFU varying with size: c_T(N) = (N/N0)^delta ; serving price per FLOP with elasticity delta_S
    dl = 0.12
    specs["f0"] = dict(label="(f) MFU: training price ~ N^delta, serving price flat", V=specs["base"]["V"], delta=dl, cD=0.0, T=T0, dS=0.0)
    specs["f1"] = dict(label="(f) MFU: training and serving price ~ N^delta", delta=dl, cD=0.0, T=T0, dS=dl,
                       V=lambda y, n: y - np.exp(dl * (n - np.log(N0))) * 2 * np.exp(n) * T0 / PSI)
    # generic constraint: wall-clock deadline, time ~ N^0.3 D (data-parallel limit)
    specs["gen"] = dict(label="Deadline, time ~ N^0.3 D", V=specs["base"]["V"], delta=0.0, cD=0.0, T=T0, con=(0.3, 1.0, -0.5))
    return specs


def _profit(spec, p=BES):
    dl = spec["delta"]
    cD = spec["cD"]

    def f(n, d):
        y = chin_y(n, d, p)
        XT = np.exp(dl * (n - np.log(N0))) * 6 * np.exp(n + d) / PSI
        return spec["V"](y, n) - XT - cD * np.exp(d) / PSI
    return f


def _minus_Vn(spec, n, d, p=BES, h=1e-6):
    """-dV/dlnN at fixed quality (finite difference in n holding y fixed), in PSI units."""
    y = chin_y(n, d, p)
    return -(spec["V"](y, n + h) - spec["V"](y, n - h)) / (2 * h)


def solve_case(spec, p=BES):
    """Brute-force optimum; for constrained cases the constraint g_n n + g_d d <= k is placed `shift` log points inside
    the unconstrained optimum, and the multiplier xi = dPi*/dk is computed by central differences."""
    f = _profit(spec, p)
    n0, d0, _ = _argmax2(f, _start(p=p))
    out = dict(n_unc=n0, d_unc=d0)
    xi, gn, gd = 0.0, 0.0, 0.0
    if "con" in spec:
        gn, gd, shift = spec["con"]
        k0 = gn * n0 + gd * d0 + shift

        def constrained(k):
            # optimize along the line g_n n + g_d d = k (the constraint binds by construction)
            if gd == 0:
                n = k / gn
                d, v = _argmax1(lambda dd: f(n, dd), d0 - 15, d0 + 15)
            elif gn == 0:
                d = k / gd
                n, v = _argmax1(lambda nn: f(nn, d), n0 - 15, n0 + 15)
            else:
                n, v = _argmax1(lambda nn: f(nn, (k - gn * nn) / gd), n0 - 15, n0 + 15)
                d = (k - gn * n) / gd
            return n, d, v
        n, d, _ = constrained(k0)
        hk = 1e-4
        xi = (constrained(k0 + hk)[2] - constrained(k0 - hk)[2]) / (2 * hk)
        out.update(n=n, d=d)
    else:
        n, d = n0, d0
        out.update(n=n, d=d)
    XT = np.exp(spec["delta"] * (n - np.log(N0))) * 6 * np.exp(n + d) / PSI
    mN = _minus_Vn(spec, n, d, p) / XT
    mD = spec["cD"] * np.exp(d) / PSI / XT
    w_formula = (1 + spec["delta"] + mN + xi * gn / XT) / (1 + mD + xi * gd / XT)
    w_tech = chin_w(n, d, p)
    out.update(w_tech=w_tech, w_formula=w_formula, mN=mN, mD=mD, xi=xi, XT=XT, nu=xi * gn / (XT * (1 + spec["delta"] + mN)))
    return out


def check_generalized_wedge():
    specs = _case_specs()
    rows = []
    for key, sp_ in specs.items():
        o = solve_case(sp_)
        N, D = np.exp(o["n"]), np.exp(o["d"])
        extra = {}
        # case-specific closed forms, evaluated at the brute-force optimum
        if key in ("base",):
            extra["closed"] = 1 + sp_["T"] / (3 * D)
        elif key == "a":
            extra["closed"] = sp_["w_closed"](o["n"], o["d"], chin_y(o["n"], o["d"]))
        elif key == "adopt":
            extra["closed"] = 1 + 0.35 * 0.8 / o["XT"]
        elif key in ("b", "c"):
            extra["closed"] = (1 + o["nu"]) * (1 + sp_["T"] / (3 * D))
        elif key == "d":
            extra["closed"] = (1 + sp_["T"] / (3 * D)) / (1 + sp_["cD"] / (6 * N))
        elif key == "e":
            extra["closed"] = (1 + sp_["T"] / (3 * D)) / (1 + sp_["NT"] / (3 * N))
        elif key == "f0":
            cT = (N / N0) ** sp_["delta"]
            extra["closed"] = 1 + sp_["delta"] + sp_["T"] / (3 * D) / cT          # p = c_S/c_T = 1/cT with flat serving price
        elif key == "f1":
            extra["closed"] = (1 + sp_["delta"]) * (1 + sp_["T"] / (3 * D))    # p = 1 when both prices scale alike
        else:
            extra["closed"] = np.nan
        err_gen = abs(o["w_formula"] / o["w_tech"] - 1)
        err_closed = abs(extra["closed"] / o["w_tech"] - 1) if np.isfinite(extra["closed"]) else np.nan
        rows.append(dict(case=key, label=sp_["label"], N=N, D=D, M=D / N, C=6 * N * D, w_tech=o["w_tech"], w_general=o["w_formula"],
                         w_closed=extra["closed"], m_N=o["mN"], m_D=o["mD"], xi=o["xi"], s=(o["w_tech"] - 1) / o["w_tech"],
                         relerr_general=err_gen, relerr_closed=err_closed))
    df = pd.DataFrame(rows)
    ok = (df["relerr_general"].max() < 2e-5) and (df["relerr_closed"].dropna().max() < 2e-5)
    record("GW.num", "Prop. A8(i),(iii)", f"Brute-force value maximization, {len(df)} conduct cases (lifetime compute; internalized serving with "
           "p, phi, eta, R_n; adoption; memory cap; latency SLO and latency cost; data cost; data cap; distillation; MFU x2; deadline): "
           "technology w = general formula and case closed forms", "NEW", "numeric", ok,
           f"max rel. err general {df['relerr_general'].max():.1e}, closed forms {df['relerr_closed'].max():.1e}")
    # signs of the constraint wedges (Prop. A8(iv))
    base = df.set_index("case")
    T_over_3D = lambda r: 2e12 / (3 * r["D"])
    sgn = dict(mem=base.loc["b", "w_tech"] > 1 + T_over_3D(base.loc["b"]) + 1e-9,
               slo=base.loc["c", "w_tech"] > 1 + T_over_3D(base.loc["c"]) + 1e-9,
               cap=base.loc["d_cap", "w_tech"] < 1 + T_over_3D(base.loc["d_cap"]) - 1e-9,
               cost=base.loc["d", "w_tech"] < 1 + T_over_3D(base.loc["d"]) - 1e-9,
               distil=base.loc["e", "w_tech"] < 1 + T_over_3D(base.loc["e"]) - 1e-9,
               deadline=base.loc["gen", "w_tech"] < 1 + T_over_3D(base.loc["gen"]) - 1e-9)
    record("GW.sign.num", "Prop. A8(iv)", "At the optimum: memory cap and SLO raise w above 1+T/(3D); data cost, data cap, distillation and a "
           "data-parallel deadline lower it", "NEW", "numeric", all(sgn.values()), ", ".join(f"{k}={v}" for k, v in sgn.items()))
    return df


def check_misspecified_technology():
    """Prop. A8(ii): the lab optimizes with its believed/own-output technology L'; the econometrician uses L:
    w_hat = w * w_L(N,D)/w_L'(N,D)."""
    T0 = 2e12
    rows = []
    for name, plab in [("lab believes Hoffmann", HOF), ("lab factor-biased (chi=0.3)", None)]:
        if plab is None:
            # factor-biased lab: psi_D = 0.3/beta (data-augmenting), i.e. own technology with B' = B e^{-beta psi_D}
            plab = dict(BES)
            plab["B"] = BES["B"] * np.exp(-0.3)
        f = lambda n, d: chin_y(n, d, plab) - 6 * np.exp(n + d) / PSI - 2 * np.exp(n) * T0 / PSI
        n, d, _ = _argmax2(f, _start(p=plab))
        w_lab = chin_w(n, d, plab)
        w_hat = chin_w(n, d, BES)
        pred = (1 + T0 / (3 * np.exp(d))) * chin_w(n, d, BES) / chin_w(n, d, plab)
        rows.append(dict(case=name, w_lab=w_lab, w_hat=w_hat, pred=pred, relerr=abs(pred / w_hat - 1), lab_foc=abs(w_lab / (1 + T0 / (3 * np.exp(d))) - 1)))
    df = pd.DataFrame(rows)
    ok = df["relerr"].max() < 1e-6 and df["lab_foc"].max() < 2e-5
    chi_ratio = df.iloc[1]["w_hat"] / df.iloc[1]["w_lab"]
    record("GW.misspec.num", "Prop. A8(ii)", "Lab optimizes with its own/believed technology L'; the econometrician's w-hat = w * w_L/w_L'; "
           "factor bias is the case w_L/w_L' = e^{-chi}", "NEW", "numeric", ok and abs(chi_ratio - np.exp(-0.3)) < 1e-9,
           f"lab FOC err {df['lab_foc'].max():.1e}; w_hat/w under chi=0.3: {chi_ratio:.4f} vs e^-0.3={np.exp(-0.3):.4f}")
    return df


# ============================================================================ Prop. A9: family-level token budget
def check_family():
    p = BES
    Ns = np.array([1e9, 7e9, 3e10])
    Psi = np.array([0.3, 0.8, 1.6])        # members' value scales (PSI units)
    T = np.array([6e12, 3e12, 1.5e12])     # members' lifetime tokens
    cD = 4e9                               # data cost per token (FLOP units), paid once for the family
    rows = {}
    # (ii) sizes fixed by a menu: only the family D-FOC
    f_fixed = lambda d: sum(Psi[i] * chin_y(np.log(Ns[i]), d, p) for i in range(3)) - 6 * np.exp(d) * Ns.sum() / PSI - cD * np.exp(d) / PSI
    d_fix, _ = _argmax1(f_fixed, 20, 40)
    D = np.exp(d_fix)
    eD = np.array([elasticities(np.log(Ns[i]), d_fix, p["A"], p["B"], p["alpha"], p["beta"])[1] for i in range(3)])
    XT = 6 * Ns * D / PSI
    om = Ns / Ns.sum()
    rho = Psi * eD / XT
    mD = cD * D / PSI / XT.sum()
    err_ii = abs(np.sum(om * rho) - (1 + mD))
    rows["fixed"] = dict(D=D, lhs=np.sum(om * rho), rhs=1 + mD, err=err_ii, rho=rho.tolist())

    # (iii) sizes chosen (interior), common D; with and without data cost
    def fam_obj(z, cD_):
        nn, d = z[:3], z[3]
        return (sum(Psi[i] * chin_y(nn[i], d, p) - 2 * np.exp(nn[i]) * T[i] / PSI for i in range(3)) - 6 * np.exp(d) * np.exp(nn).sum() / PSI
                - cD_ * np.exp(d) / PSI)

    res = {}
    for cD_ in (0.0, cD):
        z0 = np.r_[np.log(Ns), d_fix]
        r = minimize(lambda z: -fam_obj(z, cD_), z0, method="Nelder-Mead", options=dict(xatol=1e-12, fatol=1e-18, maxiter=80000, maxfev=160000))
        r = minimize(lambda z: -fam_obj(z, cD_), r.x, method="BFGS", options=dict(gtol=1e-13))
        nn, d = r.x[:3], r.x[3]
        N_, D_ = np.exp(nn), np.exp(d)
        w = np.array([chin_w(nn[i], d, p) for i in range(3)])
        om_ = N_ / N_.sum()
        wH = 1 / np.sum(om_ / w)
        pi = (om_ / w) / np.sum(om_ / w)
        mN = T / (3 * D_)
        mD_ = cD_ * D_ / PSI / (6 * N_ * D_ / PSI).sum()
        That = 3 * D_ * (w - 1)
        res[cD_] = dict(nn=nn, d=d, w=w, wH=wH, pi=pi, lhs=np.sum(pi * (1 + mN)), rhs=(1 + mD_) * wH, That=That,
                        pi_That=np.sum(pi * That), pi_T=np.sum(pi * T), member_err=np.max(np.abs(That / T - 1)),
                        share_id=abs((1 - 1 / wH) - np.sum(om_ * (1 - 1 / w))))
    r0, r1 = res[0.0], res[cD]
    ok_iii = abs(r0["lhs"] / r0["rhs"] - 1) < 1e-5 and abs(r1["lhs"] / r1["rhs"] - 1) < 1e-5 and abs(r0["pi_That"] / r0["pi_T"] - 1) < 1e-5
    rows["chosen"] = dict(D=np.exp(r0["d"]), w=r0["w"].tolist(), wH=r0["wH"], pi=r0["pi"].tolist(), That_over_T=(r0["That"] / T).tolist(),
                          pi_That=r0["pi_That"], pi_T=r0["pi_T"])

    # (iv) caps / floors: sizes held 0.5 log points below (caps) / above (floors) the interior optimum; D re-optimized
    def fixed_sizes(nn):
        g = lambda d: sum(Psi[i] * chin_y(nn[i], d, p) for i in range(3)) - 6 * np.exp(d) * np.exp(nn).sum() / PSI
        d, _ = _argmax1(g, 20, 40)
        w = np.array([chin_w(nn[i], d, p) for i in range(3)])
        om_ = np.exp(nn) / np.exp(nn).sum()
        pi = (om_ / w) / np.sum(om_ / w)
        return np.sum(pi * 3 * np.exp(d) * (w - 1)), np.sum(pi * T)
    capL, capR = fixed_sizes(r0["nn"] - 0.5)
    floL, floR = fixed_sizes(r0["nn"] + 0.5)
    ok_iv = (capL > capR) and (floL < floR)

    # (v) knife-edge: with member-specific D_i, equal D across sizes requires 1 + T_i/(3D) proportional to N_i^{-alpha}
    dbar = np.log(3e11)
    T_knife = np.array([3 * np.exp(dbar) * (chin_w(np.log(Ns[i]), dbar, p) - 1) for i in range(3)])
    Psi_knife = np.array([6 * Ns[i] * np.exp(dbar) / PSI / elasticities(np.log(Ns[i]), dbar, p["A"], p["B"], p["alpha"], p["beta"])[1]
                          for i in range(3)])
    D_ind = []
    for i, Ti in enumerate(T_knife * np.array([1.0, 1.2, 1.0])):
        g = lambda n, d: Psi_knife[i] * chin_y(n, d, p) - 6 * np.exp(n + d) / PSI - 2 * np.exp(n) * Ti / PSI
        n_, d_, _ = _argmax2(g, [np.log(Ns[i]), dbar])
        D_ind.append(np.exp(d_))
    D_ind = np.array(D_ind)
    D_knife = []
    for i, Ti in enumerate(T_knife):
        g = lambda n, d: Psi_knife[i] * chin_y(n, d, p) - 6 * np.exp(n + d) / PSI - 2 * np.exp(n) * Ti / PSI
        n_, d_, _ = _argmax2(g, [np.log(Ns[i]), dbar])
        D_knife.append(np.exp(d_))
    D_knife = np.array(D_knife)
    ok_v = np.max(np.abs(D_knife / np.exp(dbar) - 1)) < 1e-4 and abs(D_ind[1] / np.exp(dbar) - 1) > 0.01

    record("FAM.num", "Prop. A9(ii)-(v)", "Family with common D (brute force): sizes fixed -> sum omega_i rho_i = 1+m_D; sizes chosen -> "
           "sum pi_i(1+m_N,i) = (1+m_D) w_H and sum pi_i T-hat_i = sum pi_i T_i while member T-hat_i != T_i; binding caps -> upper bound, "
           "floors -> lower bound; common D under individual optimization only on the knife-edge T-profile", "NEW", "numeric",
           err_ii < 1e-6 and ok_iii and ok_iv and ok_v,
           f"(ii) err {err_ii:.1e}; (iii) pi-avg T-hat/T = {r0['pi_That'] / r0['pi_T']:.6f}, member T-hat/T = "
           f"{', '.join(f'{v:.2f}' for v in r0['That'] / T)}; (iv) caps {capL / capR:.3f}>1, floors {floL / floR:.3f}<1; "
           f"(v) knife-edge D_i/Dbar-1 max {np.max(np.abs(D_knife / np.exp(dbar) - 1)):.1e}, +20% T_2 moves D_2 by {D_ind[1] / np.exp(dbar) - 1:+.2f}")
    fam_rows = []
    for i in range(3):
        fam_rows.append(dict(member=i + 1, N_chosen=float(np.exp(r0["nn"][i])), D_common=float(np.exp(r0["d"])), w=float(r0["w"][i]),
                             pi=float(r0["pi"][i]), T_true=float(T[i]), T_hat=float(r0["That"][i]), T_hat_over_T=float(r0["That"][i] / T[i])))
    fam = pd.DataFrame(fam_rows)
    fam.attrs.update(wH=r0["wH"], pi_That=r0["pi_That"], pi_T=r0["pi_T"], cap_ratio=capL / capR, floor_ratio=floL / floR)
    return fam


# ============================================================================ Prop. A10: model-free sigma* and w
def _Lx_c(L, x, c):
    n = (c - LN6 - x) / 2
    d = (c - LN6 + x) / 2
    return L(n, d)


def _argmin_x(L, c, lo=-8, hi=14):
    r = minimize_scalar(lambda x: _Lx_c(L, x, c), bounds=(lo, hi), method="bounded", options=dict(xatol=1e-12))
    return r.x, r.fun


def _modelfree_k(L, c, h=2e-3, hc=2e-3):
    xs, _ = _argmin_x(L, c)
    Lxx = (_Lx_c(L, xs + h, c) - 2 * _Lx_c(L, xs, c) + _Lx_c(L, xs - h, c)) / h ** 2
    Lp = _argmin_x(L, c + hc)[1]
    Lm = _argmin_x(L, c - hc)[1]
    dL = (Lp - Lm) / (2 * hc)
    return 2 * Lxx / abs(dL), xs


def _sigma_isoquant(Ln, Ld, L, x, c, h=1e-3):
    """sigma = d ln M / d ln MRTS along the isoquant through (x, c), by tracing the isoquant (no P/(P+Q) formula)."""
    L0 = _Lx_c(L, x, c)
    pts = []
    for dx in (-h, h):
        cc = brentq(lambda cq: _Lx_c(L, x + dx, cq) - L0, c - 2, c + 2, xtol=1e-14)
        n = (cc - LN6 - (x + dx)) / 2
        d = (cc - LN6 + (x + dx)) / 2
        lnmrts = np.log(Ln(n, d) / Ld(n, d)) + (x + dx)       # MRTS in levels = (L_N/L_D) = (L_n/L_d) (D/N)
        pts.append(lnmrts)
    return 2 * h / (pts[1] - pts[0])


def check_model_free():
    rows = []
    # technologies: Chinchilla (Besiroglu) and a non-separable, non-quasi-homothetic technology
    Lchin = lambda n, d: BES["E"] + BES["A"] * np.exp(-BES["alpha"] * n) + BES["B"] * np.exp(-BES["beta"] * d)
    Lchin_n = lambda n, d: -BES["alpha"] * BES["A"] * np.exp(-BES["alpha"] * n)
    Lchin_d = lambda n, d: -BES["beta"] * BES["B"] * np.exp(-BES["beta"] * d)
    Lns_n = lambda n, d: -0.34 * 480.0 * np.exp(-0.34 * n) - 0.2 * 900.0 * np.exp(-0.2 * n - 0.22 * d)
    Lns_d = lambda n, d: -0.37 * 2100.0 * np.exp(-0.37 * d) - 0.22 * 900.0 * np.exp(-0.2 * n - 0.22 * d)
    techs = [("Chinchilla (Besiroglu)", Lchin, Lchin_n, Lchin_d), ("Non-separable 3-term", nonsep_L, Lns_n, Lns_d)]
    transforms = [("loss (nats)", lambda L: L), ("bits", lambda L: L / np.log(2)), ("ln(L - 1.0) (wrong E)", lambda L: np.log(L - 1.0)),
                  ("exp(L)", lambda L: np.exp(L))]
    max_err_sig, max_err_inv, max_err_w, max_err_slope = 0.0, 0.0, 0.0, 0.0
    for tname, L, Ln, Ld in techs:
        for C in (1e20, 1e22, 1e24):
            c = np.log(C)
            k_mf, xs = _modelfree_k(L, c)
            sig_iso = _sigma_isoquant(Ln, Ld, L, xs, c)
            k_true = 1 / sig_iso - 1
            if tname.startswith("Chinchilla"):
                k_closed = (BES["alpha"] + BES["beta"]) / 2
                max_err_sig = max(max_err_sig, abs(k_closed / k_true - 1))
            max_err_sig = max(max_err_sig, abs(k_mf / k_true - 1))
            # invariance to monotone transforms of output at the argmin
            ks = [_modelfree_k(lambda n, d, g=g: g(L(n, d)), c)[0] for _, g in transforms]
            max_err_inv = max(max_err_inv, max(abs(k / k_mf - 1) for k in ks))
            # model-free w off the path (two-dimensional finite differences) vs the analytic MRTS ratio
            for dx in (-3.0, -1.0, 1.5, 4.0):
                x = xs + dx
                h = 1e-4
                Lx = (_Lx_c(L, x + h, c) - _Lx_c(L, x - h, c)) / (2 * h)
                Lc = (_Lx_c(L, x, c + h) - _Lx_c(L, x, c - h)) / (2 * h)
                w_mf = (Lc - Lx) / (Lc + Lx)
                n = (c - LN6 - x) / 2
                d = (c - LN6 + x) / 2
                w_true = Ln(n, d) / Ld(n, d)
                max_err_w = max(max_err_w, abs(w_mf / w_true - 1))
            # slope of ln w along the isocost at the argmin
            h = 1e-3
            def lnw_at(x):
                n = (c - LN6 - x) / 2
                d = (c - LN6 + x) / 2
                return np.log(Ln(n, d) / Ld(n, d))
            slope = (lnw_at(xs + h) - lnw_at(xs - h)) / (2 * h)
            max_err_slope = max(max_err_slope, abs(slope / k_true - 1))
            rows.append(dict(technology=tname, C=C, M_star=np.exp(xs), k_modelfree=k_mf, k_isoquant=k_true, sigma_star_modelfree=1 / (1 + k_mf),
                             sigma_star_isoquant=sig_iso, dlnw_dlnM=slope, max_transform_dev=max(abs(k / k_mf - 1) for k in ks)))
    df = pd.DataFrame(rows)
    record("MF.sigma.num", "Prop. A10(ii)", "Model-free 1/sigma*-1 = 2L_xx/|dL*/dc| (finite differences at the IsoFLOP argmin) equals "
           "1/sigma*-1 from isoquant tracing, for Chinchilla (= (alpha+beta)/2) and a non-separable technology, at 3 budgets",
           "NEW", "numeric", max_err_sig < 5e-5, f"max rel. err {max_err_sig:.1e}")
    record("MF.inv.num", "Prop. A10(ii)", "Same value under loss, bits, ln(L-E) with a wrong E, and exp(L) (no E or functional form needed)",
           "NEW", "numeric", max_err_inv < 5e-5, f"max rel. dev. across transforms {max_err_inv:.1e}")
    record("MF.w.num", "Prop. A10(i)", "Model-free w = (L_c-L_x)/(L_c+L_x) from local 2-D finite differences equals the MRTS ratio at off-path "
           "points (M/M* from e^-3 to e^4), both technologies", "NEW", "numeric", max_err_w < 1e-6, f"max rel. err {max_err_w:.1e}")
    record("MF.slope.num", "Prop. A10(iii)", "d ln w/d ln M|_C at M*(C) = 1/sigma*-1 (isoquant-traced), both technologies", "NEW", "numeric",
           max_err_slope < 5e-5, f"max rel. err {max_err_slope:.1e}")
    return df


def check_finite_grid():
    """Remark after Prop. A10: bias of a least-squares quadratic fit to an IsoFLOP profile sampled on a symmetric grid
    of half-width h (9 equispaced points in ln N) centered at the true minimum; leading term (f''''/24) kappa4(h)."""
    p = BES
    C = 1e21
    lc = np.log(C / 6)
    po = path_objects(p["A"], p["B"], p["alpha"], p["beta"])
    ns = np.log(po["G"]) + po["a"] * lc
    f = lambda t: p["A"] * np.exp(-p["alpha"] * (ns + t)) + p["B"] * np.exp(-p["beta"] * (lc - ns - t))
    u, v = p["A"] * np.exp(-p["alpha"] * ns), p["B"] * np.exp(-p["beta"] * (lc - ns))
    f2 = p["alpha"] ** 2 * u + p["beta"] ** 2 * v
    f3 = -p["alpha"] ** 3 * u + p["beta"] ** 3 * v
    f4 = p["alpha"] ** 4 * u + p["beta"] ** 4 * v
    k_true = (p["alpha"] + p["beta"]) / 2
    dLdc = f2 / (4 * k_true) * 2          # |dL*/dc| implied by the exact identity 1/sigma*-1 = f2/(2|dL*/dc|)
    rows = []
    for h in (0.05, 0.1, 0.2, 0.5, np.log(3), np.log(10), np.log(30)):
        t = np.linspace(-h, h, 9)
        X2 = np.column_stack([np.ones_like(t), t, t ** 2])
        q = np.linalg.lstsq(X2, f(t), rcond=None)[0]
        X4 = np.column_stack([t ** j for j in range(5)])
        q4 = np.linalg.lstsq(X4, f(t), rcond=None)[0]
        m2, m4, m6 = np.mean(t ** 2), np.mean(t ** 4), np.mean(t ** 6)
        kap4 = (m6 - m2 * m4) / (m4 - m2 ** 2)
        pred = f4 / 24 * kap4                 # leading bias of q2 (= f''/2 + pred)
        bias = q[2] - f2 / 2
        t_hat = -q[1] / (2 * q[2])
        t_pred = -(f3 / 6) * (m4 / m2) / f2
        k_q = (2 * q[2]) / (2 * dLdc)
        k_q4 = (2 * q4[2]) / (2 * dLdc)
        rows.append(dict(half_width_lnN=h, N_ratio_span=np.exp(2 * h), rel_bias_curv_quadratic=bias / (f2 / 2),
                         lead_order_ratio=bias / pred, sigma_star_quadratic=1 / (1 + k_q), sigma_star_quartic=1 / (1 + k_q4),
                         sigma_star_true=1 / (1 + k_true), argmin_bias=t_hat, argmin_bias_pred=t_pred))
    df = pd.DataFrame(rows)
    ok = abs(df["lead_order_ratio"].iloc[0] - 1) < 0.01 and (df["sigma_star_quadratic"] < df["sigma_star_true"]).all()
    record("FG.num", "Remark after Prop. A10", "Quadratic fits to IsoFLOP profiles overstate curvature by (f''''/24)kappa4(h) to leading order "
           "(so sigma* is understated); a quartic removes the leading term; the argmin shifts by -(f'''/6)(m4/m2)/f''", "NEW", "numeric", ok,
           f"lead-order ratio at h=0.05: {df['lead_order_ratio'].iloc[0]:.4f}; sigma* bias at a 9x N-span: "
           f"{df.loc[df['N_ratio_span'].sub(9).abs().idxmin(), 'sigma_star_quadratic'] - 1 / (1 + k_true):+.4f}")
    return df


def _argmin_x_foc(Ln, Ld, c, lo=-8, hi=14):
    """Argmin of the IsoFLOP profile by root-finding on dL/dx = (L_d - L_n)/2 (analytic log-derivatives; precision 1e-13)."""
    g = lambda x: Ld((c - LN6 - x) / 2, (c - LN6 + x) / 2) - Ln((c - LN6 - x) / 2, (c - LN6 + x) / 2)
    return brentq(g, lo, hi, xtol=1e-14)


def check_quasi_homothetic():
    """Lemma A4(iv): kappa-family paths are straight lines in logs (power-law demands); a non-quasi-homothetic technology's are not."""
    cs = np.log(np.logspace(19, 25, 13))
    Ln_c = lambda n, d: -BES["alpha"] * BES["A"] * np.exp(-BES["alpha"] * n)
    Ld_c = lambda n, d: -BES["beta"] * BES["B"] * np.exp(-BES["beta"] * d)
    Ln_s = lambda n, d: -0.34 * 480.0 * np.exp(-0.34 * n) - 0.2 * 900.0 * np.exp(-0.2 * n - 0.22 * d)
    Ld_s = lambda n, d: -0.37 * 2100.0 * np.exp(-0.37 * d) - 0.22 * 900.0 * np.exp(-0.2 * n - 0.22 * d)
    xs_chin = np.array([_argmin_x_foc(Ln_c, Ld_c, c) for c in cs])
    xs_ns = np.array([_argmin_x_foc(Ln_s, Ld_s, c) for c in cs])
    curv_chin = np.max(np.abs(np.diff(xs_chin, 2)))
    curv_ns = np.max(np.abs(np.diff(xs_ns, 2)))
    # wedge constant along the dilation orbit (b1, a1) for the family
    n0, d0 = np.log(3e9), np.log(1e12)
    wv = [chin_w(n0 + BES["beta"] * t, d0 + BES["alpha"] * t) for t in np.linspace(-5, 5, 11)]
    ok = curv_chin < 1e-10 and curv_ns > 1e-6 and np.ptp(np.log(wv)) < 1e-12
    record("QH.num", "Lemma A4(iv)", "Family: w constant along the dilation orbits and the compute-optimal path is a straight line in logs "
           "(power-law factor demands); the non-separable technology's path is curved (not quasi-homothetic)", "NEW", "numeric", ok,
           f"max 2nd difference of ln M*(C) (half-decade steps): family {curv_chin:.1e}, non-separable {curv_ns:.1e}")


# ============================================================================ Prop. A1(iii): singular information on the path
X_SI = np.linspace(-4.6, 4.6, 40)


def _si_truth(p, cbar=np.log(1e20 / 6)):
    S = p["alpha"] + p["beta"]
    a, b = p["beta"] / S, p["alpha"] / S
    gam = p["alpha"] * p["beta"] / S
    G = (p["alpha"] * p["A"] / (p["beta"] * p["B"])) ** (1 / S)
    n0 = np.log(G) + a * cbar
    d0 = -np.log(G) + b * cbar
    P, Q = p["A"] * np.exp(-p["alpha"] * n0), p["B"] * np.exp(-p["beta"] * d0)
    return dict(a=a, b=b, gam=gam, sig=2 / (2 + S), E=p["E"], P=P, Q=Q)


def _ls(cols, yy):
    """Least squares via normal equations on a well-conditioned basis; returns (coef, rss)."""
    Xm = np.column_stack(cols)
    coef = np.linalg.solve(Xm.T @ Xm, Xm.T @ yy)
    r = Xm @ coef - yy
    return coef, float(r @ r)


def _si_profile(r1, r2, y, wlo):
    """min over E (free), P, Q >= wlo of ||E + P e^{-r1 x} + Q e^{-r2 x} - y||^2 (convex QP; active-set enumeration).
    The unconstrained step uses the basis {1, e1, (e2-e1)/(r2-r1)}, which stays well conditioned as r2 -> r1."""
    one = np.ones_like(X_SI)
    e1 = np.exp(-r1 * X_SI)
    e2 = np.exp(-r2 * X_SI)
    dr = r2 - r1
    if abs(dr) <= 1e-13:
        # rates coincide: the model is E + (P+Q) e1; only P+Q is identified, any split with P,Q >= wlo is optimal
        coef, rss = _ls([one, e1], y)
        if coef[1] >= 2 * wlo:
            return rss
        c, r_ = _ls([one], y - 2 * wlo * e1)
        return r_
    Dd = e1 * np.expm1(-dr * X_SI) / dr
    coef, rss = _ls([one, e1, Dd], y)
    Q = coef[2] / dr
    P = coef[1] - Q
    if P >= wlo and Q >= wlo:
        return rss
    best = np.inf
    c, r_ = _ls([one, e1], y - wlo * e2)            # Q at its bound
    if c[1] >= wlo:
        best = min(best, r_)
    c, r_ = _ls([one, e2], y - wlo * e1)            # P at its bound
    if c[1] >= wlo:
        best = min(best, r_)
    c, r_ = _ls([one], y - wlo * (e1 + e2))         # both at their bounds
    best = min(best, r_)
    return best


def _si_fit_unrestricted(y, tr, wlo, tau=1e-3):
    g = tr["gam"]
    grid = g * np.linspace(0.6, 1.4, 13)
    best = (np.inf, None)
    for r1 in grid:
        for r2 in grid:
            v = _si_profile(r1, r2, y, wlo)
            if v < best[0]:
                best = (v, np.array([r1, r2]))
    def f(z):
        # Review hardening: a bound branch of the profile can be singular if a rate is pushed to ~0 (e^{-r x} collinear with
        # the constant); treat such points as infeasible. Not triggered in the deterministic run.
        try:
            return _si_profile(z[0], z[1], y, wlo)
        except np.linalg.LinAlgError:
            return np.inf
    z = best[1]
    ftol = 1e-7 * len(X_SI) * tau ** 2
    for scale in (2e-2, 1e-4):
        sim = np.array([z, z + [scale * g, 0], z + [0, scale * g]])
        z = minimize(f, z, method="Nelder-Mead", options=dict(xatol=1e-9 * g, fatol=ftol, maxiter=2000, initial_simplex=sim)).x
    return z


def _si_fit_restricted(y, tr, wlo):
    r = minimize_scalar(lambda z: _si_profile(z, z, y, wlo), bounds=(0.5 * tr["gam"], 1.5 * tr["gam"]), method="bounded",
                        options=dict(xatol=1e-15))
    return r.x


def check_singular_geometry():
    """Noise-free profile criterion along rate directions (delta r1, delta r2) = t (u1, u2): quartic in t when the two decay
    rates move in opposite directions (first order absorbed by the unidentified A/B split), quadratic otherwise; a
    component with zero weight leaves its rate unidentified; imposing the path restriction restores regularity."""
    rows = []
    for name, p in [("Hoffmann (alpha != beta)", HOF), ("equal exponents (alpha = beta)", MUE)]:
        tr = _si_truth(p)
        y0 = tr["E"] + (tr["P"] + tr["Q"]) * np.exp(-tr["gam"] * X_SI)
        ts = np.logspace(-4, -2.5, 4) * tr["gam"]
        for dname, u in [("opposite (1,-1)", (1, -1)), ("opposite, sigma* moves (2,-1)", (2, -1)), ("same sign (1,1)", (1, 1)),
                         ("same sign (1,0.5)", (1, 0.5))]:
            crit = [_si_profile(tr["gam"] + t * u[0], tr["gam"] + t * u[1], y0, 0.0) for t in ts]
            slope = np.polyfit(np.log(ts), np.log(crit), 1)[0]
            dsig = [abs(2 / (2 + (tr["gam"] + t * u[0]) / tr["a"] + (tr["gam"] + t * u[1]) / tr["b"]) - tr["sig"]) for t in ts]
            dsig = np.maximum(np.array(dsig), 1e-300)
            sslope = np.polyfit(np.log(ts), np.log(dsig), 1)[0] if np.min(dsig) > 1e-250 else np.nan
            rows.append(dict(technology=name, direction=dname, crit_slope=slope, dsigma_slope=sslope))
        # zero weight: any r1 with P = 0 fits exactly
        zero_w = max(_si_profile(r1, tr["gam"], y0, 0.0) for r1 in (0.3 * tr["gam"], 2 * tr["gam"], 4 * tr["gam"]))
        rows.append(dict(technology=name, direction="P=0, r1 arbitrary", crit_slope=np.nan, dsigma_slope=np.nan, crit_value=zero_w))
        # restricted (r1 = r2): quadratic in t
        crit = [_si_profile(tr["gam"] + t, tr["gam"] + t, y0, 0.0) for t in ts]
        rows.append(dict(technology=name, direction="restricted r1=r2 (path imposed)", crit_slope=np.polyfit(np.log(ts), np.log(crit), 1)[0],
                         dsigma_slope=np.nan))
    df = pd.DataFrame(rows)
    opp = df[df["direction"].str.startswith("opposite")]["crit_slope"]
    same = df[df["direction"].str.startswith("same")]["crit_slope"]
    zw = df[df["direction"].str.startswith("P=0")]["crit_value"]
    eqmove = df[(df["technology"].str.startswith("equal")) & (df["direction"].str.contains(r"\(2,-1\)"))]["dsigma_slope"].iloc[0]
    ok = (np.abs(opp - 4).max() < 0.02 and np.abs(same - 2).max() < 0.02 and zw.max() < 1e-20 and abs(eqmove - 1) < 0.02)
    record("SI.geom", "Prop. A1(iii)", "On-path kappa=1 criterion is quartic along every direction in which the two decay rates (alpha a, beta b) "
           "move oppositely (quadratic otherwise); sigma* moves at first order along such a direction even when alpha=beta; a zero-weight "
           "component leaves its exponent unidentified", "CORRECTED", "numeric", ok,
           f"slopes opposite {opp.min():.3f}-{opp.max():.3f} (theory 4), same {same.min():.3f}-{same.max():.3f} (2); zero-weight crit "
           f"{zw.max():.0e}; d sigma* slope (alpha=beta) {eqmove:.3f}",
           "v1 said sigma* is first-order identified iff alpha=beta; true only at the true A/B, which on-path data do not identify.")
    return df


def _si_profile_known_split(r1, r2, y, frac):
    """min over (E, W) of ||E + W (frac e^{-r1 x} + (1-frac) e^{-r2 x}) - y||^2: the split P/(P+Q) is held at a known value
    (as when A/B is known, or pinned by the optimality of the path's intercept, G^{alpha+beta} = alpha A / beta B)."""
    one = np.ones_like(X_SI)
    z = frac * np.exp(-r1 * X_SI) + (1 - frac) * np.exp(-r2 * X_SI)
    return _ls([one, z], y)[1]


def check_singular_split():
    """Review addition (ra5_theory_review.md): v1's statement that sigma* is first-order identified on the path iff
    alpha = beta is correct when the split A/B is KNOWN (or pinned by the intercept restriction). With the split at its true
    value P/Q = beta/alpha, the only flat direction is (Q, -P); along it sigma* is constant iff alpha = beta (with the split
    pinned by the intercept restriction instead of known, the split moves with (alpha, beta) and sigma* moves at second order).
    With the split free (SI.geom), every opposite-sign direction is flat and sigma* moves at first order in both cases."""
    rows = []
    for name, p in [("Hoffmann (alpha != beta)", HOF), ("equal exponents (alpha = beta)", MUE)]:
        tr = _si_truth(p)
        frac = tr["P"] / (tr["P"] + tr["Q"])
        y0 = tr["E"] + (tr["P"] + tr["Q"]) * np.exp(-tr["gam"] * X_SI)
        ts = np.logspace(-4, -2.5, 4) * tr["gam"]
        for dname, u in [("flat (Q,-P)", (tr["Q"], -tr["P"])), ("opposite (2,-1)", (2.0, -1.0)), ("same sign (1,1)", (1.0, 1.0))]:
            u = np.array(u) / np.max(np.abs(u))
            crit = [_si_profile_known_split(tr["gam"] + t * u[0], tr["gam"] + t * u[1], y0, frac) for t in ts]
            dsig = [abs(2 / (2 + (tr["gam"] + t * u[0]) / tr["a"] + (tr["gam"] + t * u[1]) / tr["b"]) - tr["sig"]) for t in ts]
            rows.append(dict(technology=name, direction=dname, crit_slope=np.polyfit(np.log(ts), np.log(crit), 1)[0],
                             dsigma_slope=np.polyfit(np.log(ts), np.log(np.maximum(dsig, 1e-300)), 1)[0],
                             max_dsigma_rel_step=float(np.max(np.array(dsig) / ts))))
    df = pd.DataFrame(rows)
    g = df.set_index(["technology", "direction"])
    flat_hof = g.loc[("Hoffmann (alpha != beta)", "flat (Q,-P)")]
    flat_mue = g.loc[("equal exponents (alpha = beta)", "flat (Q,-P)")]
    others = df[df["direction"] != "flat (Q,-P)"]["crit_slope"]
    ok = (abs(flat_hof.crit_slope - 4) < 0.02 and abs(flat_mue.crit_slope - 4) < 0.02 and np.abs(others - 2).max() < 0.02
          and abs(flat_hof.dsigma_slope - 1) < 0.02 and flat_mue.max_dsigma_rel_step < 1e-8)
    record("SI.split", "Prop. A1(iii)(b)", "With the split A/B known (or pinned by the intercept restriction), only the direction (Q,-P) is "
           "flat; along it sigma* moves at first order if alpha != beta and not at all if alpha = beta (v1's 'iff alpha=beta' is "
           "correct in this case only)", "CORRECTED", "numeric", ok,
           f"flat-direction crit slopes {flat_hof.crit_slope:.3f}/{flat_mue.crit_slope:.3f} (4); other directions "
           f"{others.min():.3f}-{others.max():.3f} (2); along the flat direction d sigma* has log-log slope {flat_hof.dsigma_slope:.3f} "
           f"(alpha!=beta) and max |d sigma*|/step {flat_mue.max_dsigma_rel_step:.0e} (alpha=beta)")
    return df


def _si_mc_worker(args):
    label, pkey, tau, R, seed = args
    p = HOF if pkey == "HOF" else MUE
    tr = _si_truth(p)
    rng = np.random.default_rng(seed)
    mu = tr["E"] + (tr["P"] + tr["Q"]) * np.exp(-tr["gam"] * X_SI)
    wlo = 0.1 * min(tr["P"], tr["Q"])
    eU, eR, split = [], [], []
    for _ in range(R):
        y = mu + tau * rng.standard_normal(len(X_SI))
        r1, r2 = _si_fit_unrestricted(y, tr, wlo, tau)
        s1 = 2 / (2 + r1 / tr["a"] + r2 / tr["b"])
        s2 = 2 / (2 + r2 / tr["a"] + r1 / tr["b"])   # the exact twin
        eU.append(max(abs(s1 - tr["sig"]), abs(s2 - tr["sig"])))
        split.append(abs(r1 - r2) > 1e-6 * tr["gam"])
        rr = _si_fit_restricted(y, tr, wlo)
        eR.append(abs(2 / (2 + rr / tr["a"] + rr / tr["b"]) - tr["sig"]))
    eU, eR = np.array(eU), np.array(eR)
    return dict(technology=label, tau=tau, R=R, q25_unres=np.quantile(eU, 0.25), q50_unres=np.quantile(eU, 0.5), q75_unres=np.quantile(eU, 0.75),
                q90_unres=np.quantile(eU, 0.9), q75_restricted=np.quantile(eR, 0.75), share_split=np.mean(split))


def check_singular_mc(R=80, taus=(1e-3, 1e-4, 1e-5, 1e-6, 1e-7), processes=4):
    """Monte Carlo: on-path data (40 compute levels, 4 decades), Gaussian noise tau on L (tau -> 0 mimics n -> infinity:
    n_eff ~ tau^-2). Estimator: least squares over (E, A, B, alpha, beta) with A, B bounded away from 0 (compact parameter space),
    solved by variable projection; restricted estimator imposes beta/(alpha+beta) = a (equal decay rates). Rates: quantiles of
    |sigma*-hat - sigma*| (worst of the two exact twins) against tau."""
    from multiprocessing import Pool
    jobs = []
    for label, pkey, seed0 in [("Hoffmann (alpha != beta)", "HOF", 101), ("equal exponents (alpha = beta)", "MUE", 202)]:
        for j, tau in enumerate(taus):
            jobs.append((label, pkey, tau, R, seed0 + j))
    with Pool(processes) as pool:
        out = pool.map(_si_mc_worker, jobs)
    df = pd.DataFrame(out)
    summ = []
    ok = True
    for label, g in df.groupby("technology"):
        lt = np.log(g["tau"].values)
        sl = {c: np.polyfit(lt, np.log(g[c].values), 1)[0] for c in ("q25_unres", "q75_unres", "q90_unres", "q75_restricted")}
        summ.append(dict(technology=label, **{f"slope_{k}": v for k, v in sl.items()}))
        ok &= abs(sl["q75_unres"] - 0.5) < 0.12 and abs(sl["q90_unres"] - 0.5) < 0.12 and abs(sl["q75_restricted"] - 1) < 0.08
    s = pd.DataFrame(summ)
    record("SI.mc", "Prop. A1(iii)", "Monte Carlo (R=%d per noise level, 5 levels): sigma*-hat from on-path data under kappa=1 converges at "
           "tau^{1/2} ~ n^{-1/4} in its upper quantiles (alpha != beta and alpha = beta alike); imposing beta/(alpha+beta)=a restores tau^1 ~ n^{-1/2}" % R,
           "NEW", "MC", ok, "; ".join(f"{r.technology}: q75 {r.slope_q75_unres:.2f}, q90 {r.slope_q90_unres:.2f}, restricted q75 "
                                        f"{r.slope_q75_restricted:.2f}" for r in s.itertuples()))
    return df, s


# ============================================================================ Prop. A2 with E unknown and noise on ln L
def check_info_E_unknown():
    p = BES
    po = path_objects(p["A"], p["B"], p["alpha"], p["beta"])
    phi0 = np.array([p["alpha"] + p["beta"], po["a"], po["gamma"], np.log(po["G"]), np.log(po["K"]), p["E"]])
    lcs = np.log(np.logspace(18, 22, 9) / 6)
    grid_x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

    def design(v):
        rows = []
        for lc in lcs:
            ns = phi0[3] + phi0[1] * lc
            for x in v * grid_x:
                rows.append((ns - x / 2, lc - ns + x / 2, lc))
        return np.array(rows)

    def lnL(phi, X):
        S, a, gam, lG, lK, E = phi
        m = kappa_family_member(S, a, gam, np.exp(lG), np.exp(lK))
        return np.log(E + np.exp(lnR(X[:, 0], X[:, 1], m["A"], m["B"], m["a1"], m["b1"], m["kappa"])))

    def jac(X):
        J = np.zeros((len(X), 6))
        for k in range(6):
            h = 1e-6 * max(1.0, abs(phi0[k]))
            pp, pm = phi0.copy(), phi0.copy()
            pp[k] += h
            pm[k] -= h
            J[:, k] = (lnL(pp, X) - lnL(pm, X)) / (2 * h)
        return J

    def eff(J, k):
        I = J.T @ J
        o = [j for j in range(J.shape[1]) if j != k]
        return I[k, k] - I[k, o] @ np.linalg.solve(I[np.ix_(o, o)], I[o, k])

    rows = []
    lcbar = float(lcs.mean())
    for v in np.logspace(-1.6, -0.6, 5):
        J = jac(design(v))
        gM = np.zeros(6)
        gM[1], gM[3] = -2 * lcbar, -2.0
        I_mc = 1.0 / float(gM @ np.linalg.solve(J.T @ J, gM))
        rows.append(dict(v=v, I_S=eff(J, 0), I_lnMstar_center=I_mc, I_S_Eknown=eff(np.delete(J, 5, axis=1), 0)))
    df = pd.DataFrame(rows)
    sS = np.polyfit(np.log(df["v"]), np.log(df["I_S"]), 1)[0]
    sM = np.polyfit(np.log(df["v"]), np.log(df["I_lnMstar_center"]), 1)[0]
    loss = float(np.median(df["I_S"] / df["I_S_Eknown"]))
    ok = abs(sS - 4) < 0.1 and abs(sM - 2) < 0.1
    record("INFO.E.num", "Prop. A2 (E estimated)", "With E unknown and Gaussian noise on ln L, the efficient information is still O(v^4) for "
           "S = alpha+beta (hence sigma*) and O(v^2) for ln M* at the design's central compute", "NEW", "numeric", ok,
           f"log-log slopes {sS:.2f} (theory 4) and {sM:.2f} (theory 2); E unknown retains {loss:.2f} of the E-known information for S")
    return df


# ============================================================================ Prop. A3: DMR sign counterexample (brute force)
def check_dmr_counterexample():
    al, be, gN, gD = 0.30, 0.45, 0.5, 0.4
    A, B = 400.0, 1500.0
    C = 1e22
    xs = []
    for t in (0.0, 0.1):
        L = lambda n, d, t=t: A * np.exp(-al * (n + gN * t)) + B * np.exp(-be * (d + gD * t))
        xs.append(_argmin_x(L, np.log(C))[0])
    drift = (xs[1] - xs[0]) / 0.1
    S = al + be
    Bias = be * gD - al * gN
    ok = drift < 0 and gN > gD and abs(drift - (-2 * Bias / S)) < 1e-6
    record("DMR.num", "Prop. A3, Remark", "Brute force: M* falls over time (Hicks bias B>0, parameter-using) although parameters are augmented "
           "faster (g_N > g_D); drift = -2B/S", "CORRECTED", "numeric", ok, f"drift {drift:.5f} vs -2B/S {-2 * Bias / S:.5f}")


# ============================================================================ Prop. A11: partial identification (theory checks)
def check_pi_theory():
    # (iii) sign(w-1) = sign(x - x*) on the non-separable technology across budgets and ratios
    Lns_n = lambda n, d: -0.34 * 480.0 * np.exp(-0.34 * n) - 0.2 * 900.0 * np.exp(-0.2 * n - 0.22 * d)
    Lns_d = lambda n, d: -0.37 * 2100.0 * np.exp(-0.37 * d) - 0.22 * 900.0 * np.exp(-0.2 * n - 0.22 * d)
    bad = 0
    tot = 0
    for C in np.logspace(19, 25, 7):
        c = np.log(C)
        xs, _ = _argmin_x(nonsep_L, c)
        for dx in np.linspace(-6, 6, 49):
            if abs(dx) < 1e-6:
                continue
            x = xs + dx
            n, d = (c - LN6 - x) / 2, (c - LN6 + x) / 2
            w = Lns_n(n, d) / Lns_d(n, d)
            bad += int(np.sign(w - 1) != np.sign(dx))
            tot += 1
    record("PI.sign.num", "Prop. A11(iii)", "Non-separable technology: sign(w-1) = sign(ln M - ln M*(C)) at all tested points (7 budgets x 48 "
           "ratios), with no functional form or sigma* used", "NEW", "numeric", bad == 0, f"{bad} sign errors in {tot} points")
    # (ii) sharpness: any continuation Delta(c) of the path beyond c_J is generated by scale-dependent factor augmentation
    #      psi_N(c) = Delta(c)/(2b) (nondecreasing): L~(n,d) = E + A e^{-alpha(n + psi_N(c))} + B e^{-beta d}, c = ln6 + n + d.
    #      Both marginal products stay positive everywhere (psi_N' >= 0) and each IsoFLOP profile keeps the family's shape.
    p = BES
    S = p["alpha"] + p["beta"]
    b = p["alpha"] / S
    cJ = np.log(3e21)
    e_extra, s = 0.30, 0.25
    Delta = lambda c: e_extra * s * np.logaddexp(0.0, (c - cJ) / s)
    dDelta = lambda c: e_extra / (1 + np.exp(-(c - cJ) / s))
    psiN = lambda c: Delta(c) / (2 * b)
    Lt = lambda n, d: p["E"] + p["A"] * np.exp(-p["alpha"] * (n + psiN(LN6 + n + d))) + p["B"] * np.exp(-p["beta"] * d)
    Lorig = lambda n, d: p["E"] + p["A"] * np.exp(-p["alpha"] * n) + p["B"] * np.exp(-p["beta"] * d)
    errs, pos_ok, uniq_ok = [], True, True
    for C in np.logspace(20, 25, 11):
        c = np.log(C)
        x0, _ = _argmin_x(Lorig, c)
        x1, _ = _argmin_x(Lt, c)
        errs.append(abs((x1 - x0) - Delta(c)))
        xg = np.linspace(x1 - 10, x1 + 10, 201)
        Lg = np.array([_Lx_c(Lt, xx, c) for xx in xg])
        for xx in xg[::5]:
            n, d = (c - LN6 - xx) / 2, (c - LN6 + xx) / 2
            u = p["A"] * np.exp(-p["alpha"] * (n + psiN(c)))
            v = p["B"] * np.exp(-p["beta"] * d)
            Fn = p["alpha"] * u * (1 + dDelta(c) / (2 * b))     # -dL~/dn (analytic; psi_N depends on c = ln6+n+d)
            Fd = p["alpha"] * u * dDelta(c) / (2 * b) + p["beta"] * v
            # finite-difference confirmation of the analytic marginal products
            hh = 1e-6
            Fn_fd = -(Lt(n + hh, d) - Lt(n - hh, d)) / (2 * hh)
            Fd_fd = -(Lt(n, d + hh) - Lt(n, d - hh)) / (2 * hh)
            pos_ok &= (Fn_fd > 0) and (Fd_fd > 0) and abs(Fn_fd / Fn - 1) < 1e-5 and abs(Fd_fd / Fd - 1) < 1e-5
        dL = np.diff(Lg)
        sg = np.sign(dL[np.abs(dL) > 1e-15])
        uniq_ok &= int(np.sum(np.diff(sg) != 0)) == 1
    unchanged = abs(_argmin_x(Lt, cJ - 3.0)[0] - _argmin_x(Lorig, cJ - 3.0)[0])
    # Review addition: a DECREASING continuation (Delta' < 0), generated by data augmentation psi_D(c) = -Delta(c)/(2a)
    # (nondecreasing), as in the proof of Prop. A11(ii). The first version tested only the increasing case.
    a_ = p["beta"] / S
    Delta_dn = lambda c: -e_extra * s * np.logaddexp(0.0, (c - cJ) / s)
    dDelta_dn = lambda c: -e_extra / (1 + np.exp(-(c - cJ) / s))
    psiD = lambda c: -Delta_dn(c) / (2 * a_)
    Ld = lambda n, d: p["E"] + p["A"] * np.exp(-p["alpha"] * n) + p["B"] * np.exp(-p["beta"] * (d + psiD(LN6 + n + d)))
    errs_dn, pos_dn, uniq_dn = [], True, True
    for C in np.logspace(20, 25, 11):
        c = np.log(C)
        x0, _ = _argmin_x(Lorig, c)
        x1, _ = _argmin_x(Ld, c)
        errs_dn.append(abs((x1 - x0) - Delta_dn(c)))
        xg = np.linspace(x1 - 10, x1 + 10, 201)
        Lg = np.array([_Lx_c(Ld, xx, c) for xx in xg])
        for xx in xg[::5]:
            n, d = (c - LN6 - xx) / 2, (c - LN6 + xx) / 2
            hh = 1e-6
            pos_dn &= (-(Ld(n + hh, d) - Ld(n - hh, d)) / (2 * hh) > 0) and (-(Ld(n, d + hh) - Ld(n, d - hh)) / (2 * hh) > 0)
        dL = np.diff(Lg)
        sg = np.sign(dL[np.abs(dL) > 1e-15])
        uniq_dn &= int(np.sum(np.diff(sg) != 0)) == 1
    unchanged_dn = abs(_argmin_x(Ld, cJ - 3.0)[0] - _argmin_x(Lorig, cJ - 3.0)[0])
    ok = (max(errs) < 1e-6 and pos_ok and uniq_ok and unchanged < 1e-4 and max(errs_dn) < 1e-6 and pos_dn and uniq_dn
          and unchanged_dn < 1e-4)
    record("PI.sharp.num", "Prop. A11(ii)", "Continuations of ln M*(C) beyond the largest budget, bending up (parameter augmentation "
           "psi_N) or down (data augmentation psi_D), are generated by admissible technologies: argmin shifts by Delta(c), both "
           "marginal products positive everywhere, profiles single-peaked, in-support argmins unchanged", "NEW", "numeric", ok,
           f"max |shift - Delta| {max(errs):.1e} (up), {max(errs_dn):.1e} (down); in-support change {max(unchanged, unchanged_dn):.1e}; "
           f"positivity {pos_ok and pos_dn}; single-peaked {uniq_ok and uniq_dn}")


def run_all():
    out = {}
    out["wedge_cases"] = check_generalized_wedge()
    out["misspec"] = check_misspecified_technology()
    out["family"] = check_family()
    out["modelfree"] = check_model_free()
    out["finite_grid"] = check_finite_grid()
    check_quasi_homothetic()
    out["si_geom"] = check_singular_geometry()
    out["si_split"] = check_singular_split()
    out["info_E"] = check_info_E_unknown()
    check_dmr_counterexample()
    check_pi_theory()
    out["si_mc"], out["si_mc_slopes"] = check_singular_mc(R=100)
    return out


def si_curves(p=HOF):
    """Noise-free profiled criterion along an opposite-sign and a same-sign rate direction (for the figure)."""
    tr = _si_truth(p)
    y0 = tr["E"] + (tr["P"] + tr["Q"]) * np.exp(-tr["gam"] * X_SI)
    ts = np.logspace(-4, -1.5, 11) * tr["gam"]
    out = []
    for u in ((1, -1), (1, 1)):
        out.append((ts, np.array([_si_profile(tr["gam"] + t * u[0], tr["gam"] + t * u[1], y0, 0.0) for t in ts])))
    return out
