"""review_depmc.py -- reviewer's extension of T1.8 (package WP2-review): coverage of the model-free estimator when the
shared half of the noise is attached to the MODEL SIZE (architecture) across budgets, the public-design analogue of the
experiment's within-trunk sharing that R1 minor 7 cites (m9: one trunk per architecture, shared across token budgets;
Table B3 coverage 0.40-0.70 at rho = 0.5). The fix list's per-budget random effect (depmc.py) is the transpose of that
structure and leaves the profiles' curvature unchanged.

Noise designs (total variance kept, share rho = 0.5 shared):
  'budget'      per-budget random effect (independent replication of depmc.py's rho = 0.5 design, this file's seeds);
  'size'        per-size random effect: runs whose ln N agree within 0.03 share one shock across all budgets;
  'size_smooth' one smooth function of ln N (Gaussian process, correlation length one log point) added to every run of
                the design, i.e. a size-quality distortion common to all budgets (a digitization or architecture effect).
Everything else is depmc.py's: ra1's estimator and design-conditional bootstrap (B = 149) in every replication, the
kappa-family truths of Chinchilla (total parameters), Llama 3 and Marin DCLM. R = 200 per cell.
Output: output/tables/rb5_sigma_review_mc_size.csv. CPU only, 2 worker processes; run under nice -n 10.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import sigcommon as cm  # noqa: E402
import depmc  # noqa: E402

R_REV = int(os.environ.get("RB5_REVIEW_R", "200"))
SEED_REV = 20260929


def _rep(r_i, df, Ltrue, noise_sd, order, seed0, rho, kind, truth):
    import isoflop as iso_
    rng = np.random.default_rng(seed0 + r_i)
    d = df.copy()
    e = rng.normal(0.0, noise_sd * np.sqrt(1 - rho), len(Ltrue))
    if kind == "budget":
        b = d["b"].values
        u = rng.normal(0.0, noise_sd * np.sqrt(rho), b.max() + 1)[b]
    elif kind == "size":
        key = np.round(d["x"].values / 0.03).astype(int)
        codes = pd.factorize(key)[0]
        u = rng.normal(0.0, noise_sd * np.sqrt(rho), codes.max() + 1)[codes]
    elif kind == "size_smooth":
        xs = d["x"].values
        grid = np.linspace(xs.min() - 0.5, xs.max() + 0.5, 80)
        K = np.exp(-0.5 * ((grid[:, None] - grid[None, :]) / 1.0) ** 2) + 1e-8 * np.eye(len(grid))
        f = np.linalg.cholesky(K) @ rng.normal(0.0, 1.0, len(grid))
        u = noise_sd * np.sqrt(rho) * np.interp(xs, grid, f)
    else:
        raise ValueError(kind)
    d["L"] = Ltrue + e + u
    res = iso_.design_estimate(d, order, 1.0, "auto", "path")
    if not res.get("ok"):
        return {"fail": 1}
    bt = iso_.boot_design(d, res, depmc.MC_B, seed0 + 10_000 + r_i)
    s, rows, _, _ = iso_.summarize_design("mc", res, bt, depmc.MC_B)
    cov_b = [float(r["lo_sigma"] <= truth <= r["hi_sigma"]) for r in rows]
    return dict(sigma_fe=s["sigma_fe"], lo=s["lo_sigma_fe"], hi=s["hi_sigma_fe"], sigma_re=s["sigma_re"],
                se_re=s["se_sigma_re"], k=s["k_valid"], cov_budget=float(np.mean(cov_b)), n_budget=len(cov_b),
                drift=s["drift_sigma_per_decade"], p_drift=s["p_drift"])


def main():
    TR = depmc.truths()
    rows = []
    for name in depmc.DESIGNS:
        t = TR[name]
        for kind in ("budget", "size", "size_smooth"):
            seed0 = SEED_REV + 100_000 * (depmc.DESIGNS.index(name) + 1) + 10_000_000 * ("budget", "size", "size_smooth").index(kind)
            payload = dict(df=t["df"], Ltrue=t["Lt"], noise_sd=t["nsd"], order=t["order"], seed0=seed0, rho=0.5, kind=kind,
                           truth=t["truth"])
            res = [x for x in cm.pmap(_rep, list(range(R_REV)), payload, chunksize=10) if "_error" not in x and "fail" not in x]
            r = depmc._summ(name, res, t["truth"], R_REV, kind, 0.5, t["nsd"])
            rows.append(r)
            cm.log(f"  [review MC] {name} [{kind}, rho 0.5]: R {r['R']}; bias RE {r['bias_re']:+.4f}; coverage RE "
                   f"{r['coverage_re']:.3f} (MC s.e. {r['mcse_coverage_re']:.3f}); budgets {r['coverage_budget']:.3f}; "
                   f"drift rejection {r['rejection_drift_5pct']:.3f}")
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(cm.TABLES, "rb5_sigma_review_mc_size.csv"), index=False, float_format="%.6g")
    print(T[["design", "noise", "R", "bias_re", "sd_re", "mean_se_re", "coverage_re", "mcse_coverage_re", "coverage_budget",
             "rejection_drift_5pct"]].to_string())


if __name__ == "__main__":
    main()
