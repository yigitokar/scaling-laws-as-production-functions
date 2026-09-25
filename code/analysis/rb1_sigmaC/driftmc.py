"""driftmc.py -- does ra1's model-free estimator detect a drift of sigma* with compute, and does it invent one?
(R2 Major 1 request 3: 'a Monte Carlo under a truth whose sigma* drifts'; R1 New 2.)

Truths on the actual (N, D) grids of the Chinchilla and Llama 3 IsoFLOP designs. Start from the Chinchilla form fitted
to the design by Huber loss (ra1's parametric.fit_chin), L_ch(N, D), whose IsoFLOP profiles have the realistic
asymmetry of that form and whose sigma* is constant, sigma_ch = 2/(2 + alpha + beta). Rescale each profile around its
minimum:
    L(N, D) = L*_ch(C) + lambda(C) [L_ch(N, D) - L*_ch(C)],   C = 6ND,
which leaves the argmin and the frontier L*(C) unchanged and multiplies the profile curvature by lambda(C). By the
model-free identity 1/sigma* - 1 = L_nn|_C / (2|dL*/d ln C|), the truth's sigma*(C) = 2/(2 + lambda(C) S_ch), so
lambda(C) = S_target(C)/S_ch delivers any target path. Targets: sigma*(C) = sigma_bar + g (log10 C - mid), with
sigma_bar the design's model-free estimate (ra1: 0.673 Chinchilla, 0.660 Llama 3, 0.713 Marin DCLM), mid the middle of the design's
budgets and g in {0, -0.03, -0.06} per decade (g = 0: the constant-sigma* null).
Each replication adds Gaussian noise with the design's robust residual s.d. (ra1's Monte Carlo: Chinchilla 0.019, Marin DCLM 0.015,
Llama 3 0.0020), applies ra1's primary estimator unchanged (isoflop.design_estimate: quadratic, h = 1, path-centred
windows, log-cubic frontier; budget validity judged in each replication) and records the OLS slope of the valid budgets'
sigma*_b on log10 C. R = 300 per truth. 'Power': share of replications whose slope is below the 5th percentile of the
slopes under the null.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import rb1common as cm

rc = cm.rc
R_MC = 60 if cm.QUICK else 300
NOISE = {"Chinchilla": 0.01913, "Llama 3": 0.001997, "Marin, DCLM": 0.014996}
SIGMA_BAR = {"Chinchilla": 0.673, "Llama 3": 0.660, "Marin, DCLM": 0.713}
TARGETS = (0.0, -0.03, -0.06)


def _Lch(th, n, d):
    E, A, B, a, b = th
    return E + A * np.exp(-a * n) + B * np.exp(-b * d)


def _Lstar(th, C):
    cp = np.log(C / 6.0)
    r = minimize_scalar(lambda n: _Lch(th, n, cp - n), bounds=(cp / 2 - 8, cp / 2 + 8), method="bounded",
                        options=dict(xatol=1e-12))
    return float(r.fun)


def truth(th, df, g, sbar):
    E, A, B, a, b = th
    S_ch = a + b
    budgets = np.sort(df["C_b"].unique())
    mid = 0.5 * (np.log10(budgets.min()) + np.log10(budgets.max()))
    sig_t = sbar + g * (np.log10(budgets) - mid)
    lam = (2 * (1 / sig_t - 1)) / S_ch
    Ls = np.array([_Lstar(th, C) for C in budgets])
    lamC = dict(zip(budgets, lam))
    LsC = dict(zip(budgets, Ls))
    n, d = np.log(df["N"].values), np.log(df["D"].values)
    Lr = np.array([LsC[c] for c in df["C_b"].values]) + np.array([lamC[c] for c in df["C_b"].values]) * \
        (_Lch(th, n, d) - np.array([LsC[c] for c in df["C_b"].values]))
    return Lr, budgets, sig_t


def _ols(x, y):
    X = np.column_stack([np.ones_like(x), x])
    return float(np.linalg.lstsq(X, y, rcond=None)[0][1])


def _rep(i, df, Ltrue, sd, seed0):
    import isoflop as iso
    rng = np.random.default_rng(seed0 + i)
    d = df.copy()
    d["L"] = Ltrue + rng.normal(0.0, sd, len(Ltrue))
    res = iso.design_estimate(d)
    if not res.get("ok"):
        return {"fail": 1}
    lc = res["cb"] / np.log(10)
    return dict(slope=_ols(lc, res["sigma"]), k=len(res["sigma"]), mean=float(np.mean(res["sigma"])),
                slope_drop_top=_ols(lc[:-1], res["sigma"][:-1]) if len(lc) >= 5 else np.nan)


def run(log=cm.log):
    import parametric as pm
    import isoflop as iso
    designs = rc.isoflop_designs()
    rows, truth_rows, allsl = [], [], {}
    for name in ("Chinchilla", "Llama 3", "Marin, DCLM"):
        df = designs[name].copy()
        th5, _ = pm.fit_chin(df["N"].values, df["D"].values, df["L"].values)
        lnA, lnB, lnE, al, be = th5
        th = (np.exp(lnE), np.exp(lnA), np.exp(lnB), al, be)
        for g in TARGETS:
            Ltrue, budgets, sig_t = truth(th, df, g, SIGMA_BAR[name])
            td = _ols(np.log10(budgets), sig_t)
            d0 = df.copy()
            d0["L"] = Ltrue
            r0 = iso.design_estimate(d0)
            nf = _ols(r0["cb"] / np.log(10), r0["sigma"]) if r0.get("ok") else np.nan
            nf_bias = float(np.max(np.abs(r0["sigma"] - np.interp(r0["cb"], np.log(budgets), sig_t)))) if r0.get("ok") else np.nan
            for C, st in zip(budgets, sig_t):
                truth_rows.append(dict(design=name, target_drift=g, C=C, sigma_true=st))
            reps = rc.pmap(_rep, list(range(R_MC)), dict(df=df, Ltrue=Ltrue, sd=NOISE[name],
                                                         seed0=cm.seed_of(f"driftmc|{name}|{g}")), procs=cm.N_PROC)
            nfail = sum(("_error" in r) or ("fail" in r) for r in reps)
            reps = [r for r in reps if "_error" not in r and "fail" not in r]
            sl = np.array([r["slope"] for r in reps])
            allsl[(name, g)] = sl
            rows.append(dict(design=name, target_drift=g, true_drift=td, noise_sd=NOISE[name], sigma_bar=SIGMA_BAR[name],
                             noise_free_estimate=nf, noise_free_max_abs_budget_error=nf_bias, R=len(sl), n_failed=nfail,
                             mean_est=float(np.mean(sl)), sd_est=float(np.std(sl)), bias=float(np.mean(sl) - td),
                             p05=float(np.percentile(sl, 5)), p95=float(np.percentile(sl, 95)),
                             mean_est_drop_top=float(np.nanmean([r["slope_drop_top"] for r in reps])),
                             mean_k_valid=float(np.mean([r["k"] for r in reps]))))
            log(f"  drift MC [{name}, true drift {td:+.3f}]: noise-free estimate {nf:+.4f}; MC mean {np.mean(sl):+.4f} "
                f"(sd {np.std(sl):.4f}, R = {len(sl)}, failed {nfail})")
    T = pd.DataFrame(rows)
    for name in T.design.unique():
        q05 = float(np.percentile(allsl[(name, 0.0)], 5))
        T.loc[T.design == name, "null_p05"] = q05
        for g in TARGETS:
            T.loc[(T.design == name) & (T.target_drift == g), "reject_share_5pct_onesided"] = float(np.mean(allsl[(name, g)] < q05))
    return dict(table=T, truth=pd.DataFrame(truth_rows))
