"""Numerical verification of the closed-form bias results (model_spec Prop. 2 / SYNTHESIS P5, Prop. 3)
and of two DGP identities (Lemma 1 allocation tilt; Prop. 4 wedge).

Prop. 2: along the omega-invariant expansion path y = omega + gamma (c - ln6) - ln K - eps.  If
c = pi0 + pi1 omega + eta, then plim OLS(y on c) = gamma + pi1 Var(omega)/(pi1^2 Var(omega) + Var(eta)),
i.e. in reducible-loss units (ln l = -y):
    exogenous (pi1 = 0)              -> -gamma
    funding   (pi1 = lambda)          -> -gamma - lambda Var(omega)/Var(c)
    target    (c = (ybar - omega + lnK)/gamma + ln6, common ybar) -> 0
    predetermined c = lambda omega_{-1} + eta -> Cov(c, omega) = lambda rho Var(omega)
TFP dispersion from the OLS residual: Var(omega) - Cov(omega, c)^2/Var(c) (+ Var eps) -- understated
whenever Cov != 0.  Forward/reverse regressions bracket gamma only when Cov(omega, c) <= 0.
Prop. 3: exogenous c, release iff y >= ybar (absolute threshold) -> E[omega | c, released] decreasing in c,
OLS slope among released < gamma.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import design_b as B
import mc_lib as ml

GAM = ml.TRUE["gamma"]
LNK = np.log(ml.TRUTH.K)
LN6 = np.log(6.0)
SD_OM, SD_EPS, RHO = 0.25, 0.05, 0.8
SEED = 99_2026


def _alloc(c, theta, full_alloc, rng):
    """(n, d) given c: on the expansion path (T=0) or lifetime-optimal with T = 3 D*(c) theta + error."""
    if not full_alloc:
        d = B.alloc_d(c, np.zeros_like(c), 0.0)
    else:
        d = B.alloc_d(c, 3 * B.Dstar(c) * theta, 0.0) + 0.1 * rng.standard_normal(c.size)
    return c - LN6 - d, d


def draw(rule, par, n, rng, full_alloc=False):
    """Returns omega, c, y, the population closed-form Cov(c, omega), Var(c) (T=0), and the
    misallocation loss Delta = F_path(c) - F(n, d) >= 0 (zero on the path)."""
    omega_lag = SD_OM * rng.standard_normal(n)
    omega = RHO * omega_lag + SD_OM * np.sqrt(1 - RHO ** 2) * rng.standard_normal(n)
    eta = rng.standard_normal(n)
    theta = np.exp(0.8 * rng.standard_normal(n))
    c0 = np.log(1e22)
    if rule == "exog":
        c = c0 + eta
        cov, varc = 0.0, 1.0
    elif rule == "funding":
        c = c0 + par * omega + eta
        cov, varc = par * SD_OM ** 2, par ** 2 * SD_OM ** 2 + 1.0
    elif rule == "predet":
        c = c0 + par * omega_lag + eta
        cov, varc = par * RHO * SD_OM ** 2, par ** 2 * SD_OM ** 2 + 1.0
    elif rule == "target":
        # common target + lab-specific target dispersion sd = par (y units); the lab buys exactly the compute
        # that reaches its target along its own allocation rule
        ybar = GAM * (c0 - LN6) - LNK + par * rng.standard_normal(n)
        c = (ybar - omega + LNK) / GAM + LN6
        if full_alloc:
            lo, hi = c - 3, c + 12
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                nn, dd = _alloc(mid, theta, True, np.random.default_rng(0))
                up = omega + B.Fval(nn, dd) < ybar
                lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
            c = 0.5 * (lo + hi)
        cov = -SD_OM ** 2 / GAM
        varc = (par ** 2 + SD_OM ** 2) / GAM ** 2
    nn, dd = _alloc(c, theta, full_alloc, np.random.default_rng(0) if rule == "target" else rng)
    Fv = B.Fval(nn, dd)
    Delta = B.F_path(c) - Fv
    eps = SD_EPS * rng.standard_normal(n)
    y = omega + Fv - eps
    return omega, c, y, cov, varc, Delta


def ols_slope(x, y):
    xc, yc = x - x.mean(), y - y.mean()
    b = np.sum(xc * yc) / np.sum(xc ** 2)
    r = yc - b * xc
    se = np.sqrt(np.sum(r ** 2) / (len(x) - 2) / np.sum(xc ** 2))
    return b, se, r


CASES = [("exog", 0.0), ("funding", 0.5), ("funding", 1.0), ("funding", 2.0), ("funding", 4.0),
         ("predet", 1.0), ("predet", 2.0), ("predet", 4.0),
         ("target", 0.0), ("target", 0.05), ("target", 0.10), ("target", 0.25)]


def run():
    rng = np.random.default_rng(SEED)
    rows = []
    for full in (False, True):
        for rule, par in CASES:
            omega, c, y, cov, varc, Delta = draw(rule, par, 300_000 if full else 1_000_000, rng, full_alloc=full)
            b, se, r = ols_slope(c, y)
            formula = GAM + cov / varc
            formula_sample = GAM + np.cov(c, omega)[0, 1] / np.var(c, ddof=1)       # Prop. 2 with sample moments
            misalloc_term = -np.cov(c, Delta)[0, 1] / np.var(c, ddof=1)              # extra term if Delta ~ c
            # finite-sample: 240 lab-generations (40 labs x 6 generations, one model each), 2000 reps (on path)
            fs = []
            for _ in range(2000 if not full else 0):
                o_, c_, y_, _, _, _ = draw(rule, par, 240, rng, full_alloc=False)
                fs.append(ols_slope(c_, y_)[:2])
            fs = np.array(fs) if fs else np.full((2, 2), np.nan)
            cover = np.mean(np.abs(fs[:, 0] - GAM) <= 1.96 * fs[:, 1]) if np.isfinite(fs).all() else np.nan
            # reverse regression (c on y): implied gamma_rev = Var(y)/Cov(c,y)
            g_rev = np.var(y) / np.cov(c, y)[0, 1]
            tfp_sd_formula = np.sqrt(SD_OM ** 2 - cov ** 2 / varc + SD_EPS ** 2)
            rows.append(dict(allocation="lifetime-optimal (T>0) + error" if full else "on expansion path (T=0)",
                             rule=rule, param=par, gamma_true=GAM, slope_formula=formula, slope_sim=b, slope_sim_se=se,
                             diff=b - formula, slope_formula_sample_moments=formula_sample,
                             misallocation_term=misalloc_term, sd_Delta=float(np.std(Delta)), lnl_slope_formula=-formula, lnl_slope_sim=-b,
                             finite_mean=fs[:, 0].mean(), finite_sd=fs[:, 0].std(ddof=1), finite_cover_true_gamma=cover,
                             gamma_reverse=g_rev, brackets_gamma=bool(min(b, g_rev) <= GAM <= max(b, g_rev)),
                             resid_sd_sim=np.std(r), resid_sd_formula=tfp_sd_formula,
                             true_sd_omega_plus_eps=np.sqrt(SD_OM ** 2 + SD_EPS ** 2)))
    ver = pd.DataFrame(rows)

    # Prop. 3: selection on an absolute threshold
    srows = []
    for rule, par in [("exog", 0.0), ("funding", 2.0), ("target", 0.10)]:
        omega, c, y, cov, varc, _ = draw(rule, par, 1_000_000, rng)
        for q in (0.0, 0.25, 0.5, 0.75):
            keep = y >= np.quantile(y, q)
            b, se, _ = ols_slope(c[keep], y[keep])
            bw, _, _ = ols_slope(c[keep], omega[keep])
            srows.append(dict(rule=rule, param=par, share_dropped=q, slope_y_on_c=b, slope_omega_on_c=bw,
                              gamma_true=GAM, bias=b - GAM))
    sel = pd.DataFrame(srows)

    # DGP identities
    chk = []
    c = np.log(np.logspace(20, 26, 50))
    for psi in (-0.3, 0.0, 0.3):
        d = B.alloc_d(c, np.zeros_like(c), psi)
        n = c - LN6 - d
        a, b_ = ml.TRUE["a"], 1 - ml.TRUE["a"]
        lnG = np.log(ml.TRUTH.G)
        pred = -2 * lnG + (b_ - a) * (c - LN6) - 2 * a * psi       # Lemma 1 with psi_N = 0
        chk.append(dict(check=f"Lemma 1: d*-n* with psi_D={psi:+.1f}", max_abs_error=float(np.max(np.abs((d - n) - pred)))))
    T = 3 * B.Dstar(c) * np.exp(np.linspace(-2, 2, c.size))
    d = B.alloc_d(c, T, 0.0)
    n = c - LN6 - d
    w = ml.TRUTH.wedge(np.exp(n), np.exp(d))
    chk.append(dict(check="Prop. 4: w = 1 + T/(3D) at the lifetime optimum",
                    max_abs_error=float(np.max(np.abs(w - (1 + T / (3 * np.exp(d))))))))
    chk.append(dict(check="Prop. 4 inversion: T = 3D(w-1)",
                    max_abs_error=float(np.max(np.abs(ml.TRUTH.implied_inference_tokens(np.exp(n), np.exp(d)) / T - 1)))))
    return ver, sel, pd.DataFrame(chk)
