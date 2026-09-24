"""review_checks.py -- independent cross-checks added by the module's independent reviewer (2026-09-24; see
output/memos/ra1_modelfree_review.md). Stage 'review' of run.py; needs the farseer stage cache. Deterministic, no draws.

1. independent_modelfree(): a deliberately simple re-implementation of the model-free sigma* on every IsoFLOP design
   (per budget: quadratic in ln N on a +-1 window iterated around the budget's own argmin, >= 2 runs on each side;
   frontier: cubic in ln C of ln L* (quadratic with < 6 budgets); design summary: 2/(2 + mean S_b) with equal weights,
   the median of sigma*_b and an OLS drift of sigma*_b per decade). It shares no estimation code with isoflop.py.
2. farseer_fd(): finite-difference local wedge on Farseer at high M. eps_D from a within-size quadratic in ln D
   (runs within +-0.8 of the point), eps_N from a cross-size quadratic over the sizes within +-0.36 in ln N at the same
   D lattice value (>= 3 sizes). Compared with the parametric wedges of the farseer stage (same fits).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import ra1_common as rc


def independent_modelfree(h=1.0):
    D = rc.isoflop_designs()
    rows, brows = [], []
    for name, df in D.items():
        rr = []
        for b, g in df.groupby("b"):
            x, L = g.x.values, g.L.values
            if len(np.unique(x)) < 4:
                continue
            c = np.polyfit(x, L, 2)
            if c[0] <= 0:
                continue
            x0 = -c[1] / (2 * c[0])
            for _ in range(30):
                m = np.abs(x - x0) <= h
                if m.sum() < 4:
                    break
                c = np.polyfit(x[m], L[m], 2)
                if c[0] <= 0:
                    break
                xn = -c[1] / (2 * c[0])
                if abs(xn - x0) < 1e-10:
                    break
                x0 = xn
            m = np.abs(x - x0) <= h
            if m.sum() < 4 or c[0] <= 0 or (x[m] < x0).sum() < 2 or (x[m] > x0).sum() < 2:
                continue
            rr.append(dict(c=float(g.c.iloc[0]), curv=2 * c[0], Ls=float(np.polyval(c, x0))))
        r = pd.DataFrame(rr)
        cc = r.c - r.c.mean()
        pf = np.polyfit(cc, np.log(r.Ls), 3 if len(r) >= 6 else 2)
        r["slope"] = r.Ls * np.polyval(np.polyder(pf), cc)
        r["S"] = r.curv / np.abs(r.slope)
        r["sigma"] = 2 / (2 + r.S)
        lc = r.c / np.log(10)
        rows.append(dict(design=name, k_valid=len(r), sigma_equal_weight=float(2 / (2 + r.S.mean())),
                         sigma_median_budget=float(r.sigma.median()), sigma_min=float(r.sigma.min()),
                         sigma_max=float(r.sigma.max()), drift_sigma_per_decade_ols=float(np.polyfit(lc, r.sigma, 1)[0])))
        brows += [dict(design=name, C=float(np.exp(c_)), sigma=s_) for c_, s_ in zip(r.c, r.sigma)]
    return pd.DataFrame(rows), pd.DataFrame(brows)


def farseer_fd(far, m_min=300.0, n_sizes=10):
    import farseer as F
    fa = F.load()
    fa = fa.assign(lnN=np.log(fa.N), lnD=np.round(np.log(fa.D), 3), y=np.log(fa.L))
    est = far["est"]["ne"]
    rows = []
    for n0 in np.sort(fa.N.unique())[:n_sizes]:
        s = fa[fa.N == n0].sort_values("lnD")
        for d0 in s.lnD.values:
            M = float(np.exp(d0) / n0)
            if M < m_min:
                continue
            w = np.abs(s.lnD - d0) <= 0.8
            if w.sum() < 4:
                continue
            fz = np.polyfit(s.lnD[w] - d0, s.y[w], 2)[1]
            a = fa[(np.abs(fa.lnD - d0) < 0.01) & (np.abs(fa.lnN - np.log(n0)) <= 0.36)]
            if a.N.nunique() < 3:
                continue
            fx = np.polyfit(a.lnN - np.log(n0), a.y, 2 if a.N.nunique() >= 4 else 1)[-2]
            row = dict(N=float(n0), M=M, n_sizes=int(a.N.nunique()), eps_ratio=float(fx / fz),
                       lnw_fd=float(np.log(fx / fz)) if fx / fz > 0 else np.nan)
            for k in ("chin_full", "chin_M100", "kappa_full", "eq3_full"):
                row[f"lnw_{k}"] = float(np.log(F.w_param(est, k, np.array([n0]), np.array([np.exp(d0)])))[0])
            rows.append(row)
    return pd.DataFrame(rows)


def run(cache, log=rc.log):
    import os
    tab, bud = independent_modelfree()
    tab.to_csv(os.path.join(rc.TABLES, "ra1_modelfree_review_independent_modelfree.csv"), index=False)
    bud.to_csv(os.path.join(rc.TABLES, "ra1_modelfree_review_independent_modelfree_budgets.csv"), index=False)
    fd = farseer_fd(cache("farseer"))
    fd.to_csv(os.path.join(rc.TABLES, "ra1_modelfree_review_farseer_fd.csv"), index=False)
    for _, r in tab.iterrows():
        log(f"  review: independent model-free {r.design:22s} k={r.k_valid} sigma (equal-weight S) "
            f"{r.sigma_equal_weight:.3f}, median budget {r.sigma_median_budget:.3f}")
    hi = fd[fd.M >= 1000]
    log(f"  review: Farseer finite-difference ln w at M >= 1,000: {hi.lnw_fd.min():.2f}-{hi.lnw_fd.max():.2f} "
        f"(kappa free {hi.lnw_kappa_full.min():.2f}-{hi.lnw_kappa_full.max():.2f}, Eq. 3 {hi.lnw_eq3_full.min():.2f}-"
        f"{hi.lnw_eq3_full.max():.2f})")
    return dict(independent=tab, fd=fd)
