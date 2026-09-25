"""depmc.py -- coverage of the model-free estimator under noise correlated within a budget (T1.8; R1 minor 7).

ra1's coverage Monte Carlo (isoflop.run: designs Chinchilla in total parameters, Llama 3 and Marin DCLM; truth = each
design's own kappa-family fit on its (N, C_b) grid; Gaussian noise in L with the robust residual s.d. of the design;
the full estimator and its design-conditional bootstrap, B = 149, in every replication) is reproduced first with ra1's
seeds (R = 100: coverage of the random-effects interval 0.92 / 0.98 / 0.93). The dependence designs keep the total
noise variance and move a share rho of it into a per-budget random effect:
    L_i = L_true(i) + sqrt(1 - rho) sd e_i + sqrt(rho) sd u_b(i),    e, u ~ N(0, 1) independent,
with rho = 0.5 the case the referee asks for (0.25 and 0.75 as sensitivity). A per-budget shift leaves each profile's
curvature unchanged but moves the budget's minimum loss, and so the frontier derivative. As a further check the shared
half is also drawn as a smooth error along each profile (a Gaussian process in ln N with a correlation length of one log
point), the pattern of a digitized curve that is off by a smooth distortion. Reported: bias, s.d., mean s.e. and
coverage of the random-effects interval (sigma_re +- 1.96 se_re) and of the fixed-effect basic interval, and the
coverage of the budget-level basic intervals (which enter the meta-regression). Chinchilla in N_F is added as a fourth
design.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import sigcommon as cm

rc = cm.rc
MC_B = 149                        # ra1's bootstrap size inside the Monte Carlo
R_REPRO = 20 if cm.QUICK else 100  # ra1's replications (reproduction)
R_MAIN = 30 if cm.QUICK else 500
R_SENS = 20 if cm.QUICK else 200
DESIGNS = ["Chinchilla", "Llama 3", "Marin, DCLM"]


def truths():
    """ra1's truth for each MC design (kappa family fitted to the design, noise s.d. = 1.4826 MAD of its residuals)."""
    import parametric as pm
    import jointboot
    D = rc.isoflop_designs()
    Dnf = jointboot.designs()
    order = pd.read_csv(cm.RA1_ORDER).set_index("design")["order_chosen"]
    out = {}
    for name, df in [(n, D[n]) for n in DESIGNS] + [("Chinchilla (N_F)", Dnf["Chinchilla"])]:
        N = np.exp(df["x"].values)
        Dd = df["C_b"].values / (6 * N)
        L = df["L"].values
        th, _ = pm.fit_chin(N, Dd, L, starts=[rc.sl.BESIROGLU.theta])
        p, _ = pm.fit_kappa(N, Dd, L, th_chin=th)
        Lt = np.exp(pm.pred_kappa(p, N, Dd))
        resid = L - Lt
        nsd = float(1.4826 * np.median(np.abs(resid - np.median(resid))))
        base = "Chinchilla" if name.startswith("Chinchilla") else name
        out[name] = dict(df=df, Lt=Lt, nsd=nsd, truth=float(pm.sigma_star_kappa(p)), order=int(order.get(base, 2)))
    return out


def _gp_chol(x, ell):
    K = np.exp(-0.5 * ((x[:, None] - x[None, :]) / ell) ** 2) + 1e-8 * np.eye(len(x))
    return np.linalg.cholesky(K)


def _rep(r_i, df, Ltrue, noise_sd, order, seed0, rho, kind, truth):
    """One replication: ra1's estimator and bootstrap on L_true + noise (independent / per-budget / smooth)."""
    import isoflop as iso_
    rng = np.random.default_rng(seed0 + r_i)
    d = df.copy()
    if kind == "ra1":                              # ra1's _mc_one noise (reproduction)
        d["L"] = Ltrue + rng.normal(0.0, noise_sd, len(Ltrue))
    else:
        e = rng.normal(0.0, noise_sd * np.sqrt(1 - rho), len(Ltrue))
        b = d["b"].values
        if kind == "budget":
            u = rng.normal(0.0, noise_sd * np.sqrt(rho), b.max() + 1)[b]
        else:                                      # smooth along ln N within each budget
            u = np.zeros(len(Ltrue))
            for bb in np.unique(b):
                ix = np.where(b == bb)[0]
                Lc = _gp_chol(d["x"].values[ix], 1.0)
                u[ix] = noise_sd * np.sqrt(rho) * (Lc @ rng.normal(0.0, 1.0, len(ix)))
        d["L"] = Ltrue + e + u
    res = iso_.design_estimate(d, order, 1.0, "auto", "path")
    if not res.get("ok"):
        return {"fail": 1}
    bt = iso_.boot_design(d, res, MC_B, seed0 + 10_000 + r_i)
    s, rows, _, _ = iso_.summarize_design("mc", res, bt, MC_B)
    cov_b = [float(r["lo_sigma"] <= truth <= r["hi_sigma"]) for r in rows]
    return dict(sigma_fe=s["sigma_fe"], lo=s["lo_sigma_fe"], hi=s["hi_sigma_fe"], sigma_re=s["sigma_re"],
                se_re=s["se_sigma_re"], k=s["k_valid"], cov_budget=float(np.mean(cov_b)), n_budget=len(cov_b),
                drift=s["drift_sigma_per_decade"], p_drift=s["p_drift"])


def _summ(name, T, truth, R_target, kind, rho, nsd):
    m = pd.DataFrame(T)
    return dict(design=name, noise=kind, rho=rho, truth=truth, noise_sd=nsd, R=len(m), R_target=R_target, B=MC_B,
                bias_re=float(m.sigma_re.mean() - truth), sd_re=float(m.sigma_re.std(ddof=1)), mean_se_re=float(m.se_re.mean()),
                coverage_re=float(np.mean(np.abs(m.sigma_re - truth) <= 1.96 * m.se_re)),
                mcse_coverage_re=float(np.sqrt(np.mean(np.abs(m.sigma_re - truth) <= 1.96 * m.se_re) *
                                               (1 - np.mean(np.abs(m.sigma_re - truth) <= 1.96 * m.se_re)) / len(m))),
                bias_fe=float(m.sigma_fe.mean() - truth), coverage_fe=float(np.mean((m.lo <= truth) & (truth <= m.hi))),
                coverage_budget=float(m.cov_budget.mean()), mean_k=float(m.k.mean()),
                rejection_drift_5pct=float(np.mean(m.p_drift < 0.05)))


def run(log=cm.log):
    TR = truths()
    rows = []
    jobs = []
    for name, t in TR.items():
        base = "Chinchilla" if name.startswith("Chinchilla") else name
        if name in DESIGNS:                        # ra1's reproduction (ra1's seeds, R = 100)
            jobs.append((name, "ra1", 0.0, R_REPRO, rc.seed_of(f"mc|{base}")))
        jobs.append((name, "independent", 0.0, R_MAIN, cm.seed_of(f"mc|ind|{name}")))
        jobs.append((name, "budget", 0.5, R_MAIN, cm.seed_of(f"mc|bud50|{name}")))
        if name in DESIGNS:
            for rho in (0.25, 0.75):
                jobs.append((name, "budget", rho, R_SENS, cm.seed_of(f"mc|bud{int(100 * rho)}|{name}")))
            jobs.append((name, "smooth", 0.5, R_SENS, cm.seed_of(f"mc|gp50|{name}")))
    for name, kind, rho, R, seed0 in jobs:
        t = TR[name]
        items = list(range(R))
        payload = dict(df=t["df"], Ltrue=t["Lt"], noise_sd=t["nsd"], order=t["order"], seed0=seed0, rho=rho, kind=kind,
                       truth=t["truth"])
        res = [x for x in cm.pmap(_rep, items, payload, chunksize=10) if "_error" not in x and "fail" not in x]
        rows.append(_summ(name, res, t["truth"], R, kind, rho, t["nsd"]))
        r = rows[-1]
        log(f"    MC {name} [{kind}, rho {rho}]: R {r['R']}; bias RE {r['bias_re']:+.4f}; coverage RE {r['coverage_re']:.3f} "
            f"(MC s.e. {r['mcse_coverage_re']:.3f}); FE {r['coverage_fe']:.3f}; budgets {r['coverage_budget']:.3f}")
    T = pd.DataFrame(rows)
    ref = pd.read_csv(cm.RA1_MC).set_index("design")
    chk = []
    for name in DESIGNS:
        mine = T[(T.design == name) & (T.noise == "ra1")].iloc[0]
        for c in ("coverage_re", "bias_re", "coverage_fe", "bias_fe"):
            chk.append(dict(design=name, statistic=c, mine=mine[c], ra1=float(ref.loc[name, c]),
                            abs_diff=abs(mine[c] - float(ref.loc[name, c]))))
    return dict(table=T, check=pd.DataFrame(chk))
