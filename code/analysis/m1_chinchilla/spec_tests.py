"""Task 5 -- specification tests on the full-design Chinchilla sample (n = 240; n = 245 as robustness).

1. CES: alpha = beta (homothetic CES with sigma = 1/(1+rho)); bootstrap Wald (pairs and cluster) + restricted fit.
2. Kaplan-type outer exponent: L = E + (A N^-a1 + B D^-b1)^kappa; test kappa = 1 (bootstrap Wald; Gaussian LR).
3. Rank-one curvature: with E fixed, translog of z = ln(L - E) in centred (n, d) = (ln N, ln D):
      z = c + b_N n + b_D d + (1/2) b_NN n^2 + (1/2) b_DD d^2 + b_ND n d + e.
   Chinchilla implies Hessian of ln R proportional to [[alpha^2, -alpha beta], [-alpha beta, beta^2]] at every point, so
   any average of it is rank one: tau = b_NN b_DD - b_ND^2 = 0 with b_ND < 0 (b_NN, b_DD > 0). Because the translog is
   only a second-order approximation over a wide design, tau is also computed on the reference model's own fitted
   values at the same design (design-consistent null) and the test is of tau_data = tau_model.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import chi2, norm

import boot
import estimators as est
import fdep
from common import BESI_PUB, HOFF_TEX, SEED, TABLES, derived, load_chinchilla, sl, theta_of

B_SPEC = 300


# ----------------------------------------------------------------------------- CES restricted fit
def fit_ces(N, D, L, init, delta=1e-3):
    lN, lD, lL = np.log(N), np.log(D), np.log(L)
    w = np.ones_like(lL)
    f = lambda q: sl._obj(np.r_[q[:3], q[3], q[3]], lN, lD, lL, delta, w)
    best = None
    for st in ([init[0], init[1], init[2], 0.5 * (init[3] + init[4])], [6.0, 6.0, 0.5, 0.3], [6.0, 8.0, 0.6, 0.36]):
        r = minimize(f, np.asarray(st, float), method="L-BFGS-B", options=dict(maxiter=10000, ftol=1e-15, gtol=1e-12))
        if best is None or r.fun < best.fun:
            best = r
    q = best.x
    return sl.Chinchilla.from_theta(np.r_[q[:3], q[3], q[3]], objective=float(best.fun))


# ----------------------------------------------------------------------------- translog
def translog_design(N, D, center):
    n = np.log(N) - center[0]
    d = np.log(D) - center[1]
    X = np.column_stack([np.ones_like(n), n, d, 0.5 * n * n, 0.5 * d * d, n * d])
    return X


def tau_of(b):
    return b[3] * b[4] - b[5] ** 2


def translog_fit(N, D, L, E, center, cluster=None):
    z = np.log(L - E)
    X = translog_design(N, D, center)
    res = sm.OLS(z, X).fit(cov_type="HC1")
    out = dict(b=res.params, V_hc1=res.cov_params(), r2=res.rsquared, n=len(z))
    if cluster is not None:
        rc = sm.OLS(z, X).fit(cov_type="cluster", cov_kwds=dict(groups=cluster))
        out["V_cl"] = rc.cov_params()
    return out


def delta_tau_se(b, V):
    g = np.zeros(6)
    g[3], g[4], g[5] = b[4], b[3], -2 * b[5]
    return float(np.sqrt(g @ V @ g))


def _boot_spec(idx, N, D, L, inits, center, gen_start, **_):
    """One replication: Huber refit (E re-estimated), CES Wald inputs, translog tau, kappa (warm-started)."""
    Nb, Db, Lb = N[idx], D[idx], L[idx]
    m = est.fit_huber(Nb, Db, Lb, inits=inits)
    out = dict(alpha=m.alpha, beta=m.beta, E=m.E)
    tl = translog_fit(Nb, Db, Lb, m.E, center)
    b = tl["b"]
    out.update(tau=tau_of(b), bNN=b[3], bDD=b[4], bND=b[5])
    tlm = translog_fit(Nb, Db, m.loss(Nb, Db), m.E, center)   # model's own tau at the resampled design
    out["tau_model"] = tau_of(tlm["b"])
    # generalised model: kappa (Huber objective), warm start from full-sample free fit and the kappa = 1 point
    lN, lD, lL = np.log(Nb), np.log(Db), np.log(Lb)
    best = None
    for p0 in gen_start + [np.r_[theta_of(m), 1.0]]:
        r = minimize(lambda p: fdep._obj_gen(p, lN, lD, lL, "huber"), np.asarray(p0, float), method="L-BFGS-B",
                     bounds=[(-50, 80), (-50, 80), (-5, 3), (0.005, 5), (0.005, 5), (0.02, 20)],
                     options=dict(maxiter=10000, ftol=1e-15, gtol=1e-11))
        if best is None or r.fun < best.fun:
            best = r
    p = best.x
    out.update(kappa=p[5], sigma_star_gen=2 / (2 + p[3] + p[4]), a1=p[3], b1=p[4])
    return out


def run(B=B_SPEC, log=print, horse=None):
    rows_ces, rows_kappa, rows_tl = [], [], []
    for k in (5, 0):
        df = load_chinchilla(k)
        N, D, L = df.N.values, df.D.values, df.L.values
        n = len(L)
        ref = est.fit_huber(N, D, L, inits=[theta_of(BESI_PUB), theta_of(HOFF_TEX)])
        ref2 = est.fit_huber(N, D, L, grid=sl.FAST_GRID)
        ref = ref if ref.extra["objective"] <= ref2.extra["objective"] else ref2
        center = (float(np.mean(np.log(N))), float(np.mean(np.log(D))))
        # generalised model, full sample (Huber and Gaussian)
        pH, fH = fdep.fit_gen_free(N, D, L, ref, kind="huber")
        pG, fG = fdep.fit_gen_free(N, D, L, ref, kind="gauss")
        mg = est.fit_gauss_log(N, D, L, inits=[theta_of(ref), theta_of(BESI_PUB)])
        ssr1 = 2 * mg.extra["objective"]              # sl's Gaussian objective is 0.5*SSR
        LR_kappa = n * np.log(ssr1 / (2 * fG))
        # CES restricted
        mc = fit_ces(N, D, L, theta_of(ref))
        # bootstrap (pairs and cluster)
        tl = translog_fit(N, D, L, ref.E, center, cluster=df.cl.values)
        tl_model = translog_fit(N, D, ref.loss(N, D), ref.E, center)
        res = {}
        for sch in ("pairs", "cluster"):
            idxs = boot.draws(n if sch == "pairs" else boot.groups_of(df.cl.values), B, sch, SEED + 1200 + 3 * k + (sch == "cluster"))
            out = boot.pmap(_boot_spec, idxs, dict(N=N, D=D, L=L, inits=[theta_of(ref), theta_of(BESI_PUB), theta_of(HOFF_TEX)],
                                                    center=center, gen_start=[pH]))
            ok = [o for o in out if "_error" not in o]
            res[sch] = pd.DataFrame(ok)
        # --- CES
        for sch, R in res.items():
            dd = R.alpha - R.beta
            se = float(np.std(dd, ddof=1))
            z = (ref.alpha - ref.beta) / se
            rows_ces.append(dict(sample=f"n{n}", scheme=sch, alpha=ref.alpha, beta=ref.beta, diff=ref.alpha - ref.beta,
                                 se_diff=se, z=z, p=float(2 * norm.sf(abs(z))),
                                 ci_lo=float(np.percentile(dd, 2.5)), ci_hi=float(np.percentile(dd, 97.5)), B=len(R),
                                 ces_rho=mc.alpha, ces_sigma=1 / (1 + mc.alpha), ces_E=mc.E, ces_A=mc.A, ces_B=mc.B,
                                 ces_Mstar=float(mc.D_opt(1e21) / mc.N_opt(1e21)),
                                 obj_u=ref.extra["objective"], obj_r=mc.extra["objective"],
                                 qLR_laplace=float(2 * n * np.log(mc.extra["objective"] / ref.extra["objective"]))))
        # --- kappa
        for sch, R in res.items():
            se = float(np.std(R.kappa, ddof=1))
            rows_kappa.append(dict(sample=f"n{n}", scheme=sch, kappa_huber=pH[5], a1_huber=pH[3], b1_huber=pH[4],
                                   E_huber=float(np.exp(pH[2])), lnA_huber=pH[0], lnB_huber=pH[1],
                                   gamma_path_huber=float(pH[5] * pH[3] * pH[4] / (pH[3] + pH[4])),
                                   a_path_huber=float(pH[4] / (pH[3] + pH[4])),
                                   sigma_star_huber=2 / (2 + pH[3] + pH[4]), se_kappa=se, z_kappa1=(pH[5] - 1) / se,
                                   p_kappa1=float(2 * norm.sf(abs((pH[5] - 1) / se))),
                                   ci_lo=float(np.percentile(R.kappa, 2.5)), ci_hi=float(np.percentile(R.kappa, 97.5)),
                                   se_sigma_star_gen=float(np.std(R.sigma_star_gen, ddof=1)),
                                   kappa_gauss=pG[5], sigma_star_gauss=2 / (2 + pG[3] + pG[4]), LR_gauss=LR_kappa,
                                   p_LR_gauss=float(chi2.sf(LR_kappa, 1)),
                                   qLR_laplace=float(2 * n * np.log(ref.extra["objective"] / fH)),
                                   sigma_star_kappa1=ref.sigma_star, B=len(R)))
        # --- translog / rank one
        b = tl["b"]
        tau, tau_m = tau_of(b), tau_of(tl_model["b"])
        for vlab, V in (("HC1", tl["V_hc1"]), ("cluster(budget)", tl["V_cl"])):
            se = delta_tau_se(b, V)
            rows_tl.append(dict(sample=f"n{n}", inference=f"delta method, {vlab}", E_fixed=ref.E, bN=b[1], bD=b[2],
                                bNN=b[3], bDD=b[4], bND=b[5], se_bND=float(np.sqrt(V[5, 5])),
                                t_bND=float(b[5] / np.sqrt(V[5, 5])), tau=tau, se_tau=se, z_tau0=tau / se,
                                p_tau0=float(2 * norm.sf(abs(tau / se))), tau_model=tau_m,
                                z_tau_model=(tau - tau_m) / se, p_tau_model=float(2 * norm.sf(abs((tau - tau_m) / se))),
                                ratio_implied_alpha_over_beta=float(-b[5] / b[4]), alpha_over_beta=ref.alpha / ref.beta,
                                r2=tl["r2"], n=n, rank1_corr=float(b[5] / np.sqrt(b[3] * b[4])) if b[3] * b[4] > 0 else np.nan))
        for sch, R in res.items():
            se = float(np.std(R.tau, ddof=1))
            sed = float(np.std(R.tau - R.tau_model, ddof=1))
            rows_tl.append(dict(sample=f"n{n}", inference=f"bootstrap ({sch}), E re-estimated", E_fixed=ref.E, bN=b[1], bD=b[2],
                                bNN=b[3], bDD=b[4], bND=b[5], se_bND=float(np.std(R.bND, ddof=1)),
                                t_bND=float(b[5] / np.std(R.bND, ddof=1)), tau=tau, se_tau=se, z_tau0=tau / se,
                                p_tau0=float(2 * norm.sf(abs(tau / se))), tau_model=tau_m,
                                z_tau_model=(tau - tau_m) / sed, p_tau_model=float(2 * norm.sf(abs((tau - tau_m) / sed))),
                                ratio_implied_alpha_over_beta=float(-b[5] / b[4]), alpha_over_beta=ref.alpha / ref.beta,
                                r2=tl["r2"], n=n, rank1_corr=float(b[5] / np.sqrt(b[3] * b[4])) if b[3] * b[4] > 0 else np.nan))
        # E sensitivity (point estimates only)
        for Ealt, lab in ((1.6934, "E = Hoffmann 1.6934"), (mg.E, f"E = Gaussian-log {mg.E:.4f}")):
            t2 = translog_fit(N, D, L, Ealt, center, cluster=df.cl.values)
            b2 = t2["b"]
            se = delta_tau_se(b2, t2["V_cl"])
            rows_tl.append(dict(sample=f"n{n}", inference=f"sensitivity: {lab}, cluster", E_fixed=Ealt, bN=b2[1], bD=b2[2],
                                bNN=b2[3], bDD=b2[4], bND=b2[5], se_bND=float(np.sqrt(t2["V_cl"][5, 5])),
                                t_bND=float(b2[5] / np.sqrt(t2["V_cl"][5, 5])), tau=tau_of(b2), se_tau=se,
                                z_tau0=tau_of(b2) / se, p_tau0=float(2 * norm.sf(abs(tau_of(b2) / se))), r2=t2["r2"], n=n,
                                ratio_implied_alpha_over_beta=float(-b2[5] / b2[4]), alpha_over_beta=ref.alpha / ref.beta,
                                rank1_corr=float(b2[5] / np.sqrt(b2[3] * b2[4])) if b2[3] * b2[4] > 0 else np.nan))
        log(f"  spec n={n}: CES diff {ref.alpha - ref.beta:.4f}; kappa(huber) {pH[5]:.3f} sigma*_gen {2 / (2 + pH[3] + pH[4]):.3f};"
            f" kappa(gauss) {pG[5]:.3f} LR {LR_kappa:.1f}; tau {tau:.4g} (model {tau_m:.4g}), bND {b[5]:.4f}")
    ces, kap, tlg = pd.DataFrame(rows_ces), pd.DataFrame(rows_kappa), pd.DataFrame(rows_tl)
    ces.to_csv(os.path.join(TABLES, "m1_chinchilla_spec_ces.csv"), index=False)
    kap.to_csv(os.path.join(TABLES, "m1_chinchilla_spec_kappa.csv"), index=False)
    tlg.to_csv(os.path.join(TABLES, "m1_chinchilla_spec_translog.csv"), index=False)
    return ces, kap, tlg
