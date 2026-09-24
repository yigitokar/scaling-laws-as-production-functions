"""Task 1 -- estimator horse race on the Chinchilla extraction (n = 240 and n = 245).

For each estimator: full-sample estimate (grid of starts) and B pairs- and B cluster-bootstrap replications
(cluster = reconstructed IsoFLOP budget, 9 clusters). The same index draws are used for every estimator within a
(sample, scheme) cell, so estimator differences can be bootstrapped pairwise (e.g. Huber vs LAD).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import boot
import estimators as est
from common import (BESI_PUB, DERIVED_KEYS, HOFF_TEX, PROC, SEED, TABLES, derived, load_chinchilla, sl, theta_of)

EST_ORDER = ["huber", "lad_log", "gauss_log", "nls_lev", "vpnls", "norm"]
B_BOOT = 400


def boot_fit(idx, N, D, L, name, inits, **_):
    m = est.fit(name, N[idx], D[idx], L[idx], inits=inits)
    d = derived(m)
    d.update(th0=np.log(m.A), th1=np.log(m.B), th2=np.log(m.E), obj=m.extra["objective"])
    if name == "norm":
        d.update(a_norm=m.extra["a_norm"], b_norm=m.extra["b_norm"])
    return d


def full_fit(name, N, D, L):
    kw = {}
    if name in ("huber", "gauss_log", "nls_lev"):
        kw = dict(grid=sl.DEFAULT_GRID if name == "huber" else sl.FAST_GRID)
    return est.fit(name, N, D, L, **kw)


def run(B=B_BOOT, log=print):
    rows, draws_store = [], {}
    for k in (5, 0):
        df = load_chinchilla(k)
        n = len(df)
        N, D, L = df.N.values, df.D.values, df.L.values
        fits = {name: full_fit(name, N, D, L) for name in EST_ORDER}
        # Conditioning: Jacobian of ln Lhat at the estimate, raw vs KMW-normalised parameterisation
        cond = {}
        for name, m in fits.items():
            if name == "norm":
                J = est.jac_log_norm(est.to_norm(theta_of(m), m.extra["Nbar"], m.extra["Dbar"]), N, D,
                                     m.extra["Nbar"], m.extra["Dbar"])
            else:
                J = est.jac_log(theta_of(m), N, D)
            cond[name] = est.cond_stats(J)
        share_lin, _ = est.share_linear_region(fits["huber"], N, D, L)
        schemes = {"pairs": boot.draws(n, B, "pairs", SEED + 11 * k),
                   "cluster": boot.draws(boot.groups_of(df.cl.values), B, "cluster", SEED + 13 * k + 1)}
        for name in EST_ORDER:
            m = fits[name]
            inits = [theta_of(m), theta_of(fits["huber"]), theta_of(HOFF_TEX), theta_of(BESI_PUB)]
            d = derived(m)
            res = dict(sample=f"n{n}", estimator=name, n=n, objective=m.extra["objective"],
                       cond_raw=cond[name]["cond"], cond_scaled=cond[name]["cond_scaled"],
                       share_linear_huber=share_lin if name == "huber" else np.nan)
            if name == "norm":
                res.update(a_norm=m.extra["a_norm"], b_norm=m.extra["b_norm"], Nbar=m.extra["Nbar"], Dbar=m.extra["Dbar"])
            res.update(d)
            for sch, idxs in schemes.items():
                out = boot.pmap(boot_fit, idxs, dict(N=N, D=D, L=L, name=name, inits=inits))
                keys = DERIVED_KEYS + (["a_norm", "b_norm"] if name == "norm" else [])
                s = boot.summarize(out, keys)
                for kk in keys:
                    res[f"se_{sch}_{kk}"] = s[kk]["se"]
                    res[f"lo_{sch}_{kk}"] = s[kk]["lo"]
                    res[f"hi_{sch}_{kk}"] = s[kk]["hi"]
                res[f"nfail_{sch}"] = s["_nfail"]
                draws_store[(n, name, sch)] = boot.mat(out, ["th0", "th1", "th2", "alpha", "beta"] + DERIVED_KEYS)
                log(f"  horse race n={n} {name:9s} {sch:7s}: alpha {d['alpha']:.4f} ({s['alpha']['se']:.4f}) "
                    f"beta {d['beta']:.4f} ({s['beta']['se']:.4f}) fails={s['_nfail']}")
            rows.append(res)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(TABLES, "m1_chinchilla_horse_race.csv"), index=False)
    # store the bootstrap draws (theta and derived objects) for downstream modules
    cols = ["lnA", "lnB", "lnE", "alpha", "beta"] + DERIVED_KEYS
    for (n, name, sch), M in draws_store.items():
        np.save(os.path.join(PROC, f"boot_chinchilla_n{n}_{name}_{sch}.npy"), M)
    with open(os.path.join(PROC, "boot_columns.txt"), "w") as f:
        f.write(",".join(cols) + "\n")
    # Huber vs LAD: paired bootstrap of the difference (same draws)
    hl = []
    for n in (240, 245):
        for sch in ("pairs", "cluster"):
            H, Lh = draws_store[(n, "huber", sch)], draws_store[(n, "lad_log", sch)]
            m_ = min(len(H), len(Lh))
            for j, kk in ((3, "alpha"), (4, "beta"), (5 + DERIVED_KEYS.index("a"), "a")):
                dlt = H[:m_, j] - Lh[:m_, j]
                hl.append(dict(sample=f"n{n}", scheme=sch, param=kk, sd_diff=float(np.std(dlt, ddof=1)),
                               corr=float(np.corrcoef(H[:m_, j], Lh[:m_, j])[0, 1]),
                               sd_huber=float(np.std(H[:m_, j], ddof=1)), sd_lad=float(np.std(Lh[:m_, j], ddof=1))))
    pd.DataFrame(hl).to_csv(os.path.join(TABLES, "m1_chinchilla_huber_vs_lad.csv"), index=False)
    return tab, draws_store


# ----------------------------------------------------------------------------- which objective produced the published numbers?
def besiroglu_diagnostics(log=print):
    """Reconcile our Huber(1e-3) optimum with Besiroglu et al.'s published point estimates.

    Their notebook (data/raw/epoch_chinchilla/data_analysis.ipynb, cells 16/24/33) reports the optimum of a Huber
    *log-likelihood with a free scale* s (Huber applied to r/s with delta = 1e-3, plus n ln s and the Huber
    normalising constant), started at Hoffmann's TeX parameters and minimised by BFGS. With s estimated at ~5e-6,
    the quadratic region is |r| < delta*s ~ 5e-9, i.e. the objective is LAD. We (i) evaluate the Huber(1e-3) objective
    at the published and at our estimates, and (ii) re-run their likelihood objective (BFGS, numerical gradients)."""
    from scipy.optimize import minimize as _min
    from scipy.stats import norm as _norm
    df = load_chinchilla(5)
    N, D, L = df.N.values, df.D.values, df.L.values
    lN, lD, lL = np.log(N), np.log(D), np.log(L)
    ours = est.fit_huber(N, D, L, inits=[theta_of(BESI_PUB)])
    lad = est.fit_lad(N, D, L, inits=[theta_of(ours)])
    dl = 1e-3
    Z = np.sqrt(2 * np.pi) * (1 - 2 * _norm.sf(dl)) + 2 * np.exp(-0.5 * dl ** 2) / dl

    def nll(p):
        s = np.exp(p[5])
        x = (lL - est.lse_pred(p[:5], lN, lD)) / s
        loss = np.where(np.abs(x) <= dl, 0.5 * x * x, dl * (np.abs(x) - 0.5 * dl))
        return float(np.sum(loss + np.log(Z) + np.log(s)))

    start = np.r_[6.0073404, 6.0179186, 0.5267228, 0.33917084, 0.2849083, 0.0]   # their true_params (Hoffmann TeX)
    r = _min(nll, start, method="BFGS")
    rows = []
    for lab, th in (("Besiroglu et al. published", theta_of(BESI_PUB)), ("Huber(1e-3) optimum (this module)", theta_of(ours)),
                    ("LAD optimum (this module)", theta_of(lad)),
                    ("Their Huber log-likelihood with free scale, BFGS from Hoffmann (re-run)", r.x[:5])):
        m = sl.Chinchilla.from_theta(th)
        rows.append(dict(parameter_set=lab, E=m.E, A=m.A, B=m.B, alpha=m.alpha, beta=m.beta,
                         huber_1e3_objective=est.huber_obj(th, N, D, L), l1_objective=est.l1_obj(th, lN, lD, lL),
                         w_chin70b=float(m.wedge(70e9, 1.4e12)), Mstar_576e23=float(m.D_opt(5.76e23) / m.N_opt(5.76e23))))
    out = pd.DataFrame(rows)
    out["scale_s_if_likelihood"] = [np.nan, np.nan, np.nan, float(np.exp(r.x[5]))]
    out.to_csv(os.path.join(TABLES, "m1_chinchilla_besiroglu_check.csv"), index=False)
    log("  Besiroglu reconciliation:\n" + out.round(6).to_string())
    return out
