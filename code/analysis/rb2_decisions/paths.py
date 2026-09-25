"""paths.py -- IsoFLOP expansion paths with design-conditional wild bootstrap-t inference (between-budget residuals).

The per-budget IsoFLOP minima (ln C_k, ln M*_k) of a design are regressed on ln C: ln M*_k = mu + e (ln C_k - ln C_0),
C_0 = the design's largest bracketed budget (the anchor). ra2_wedge's anchor and lab-path intervals resampled only the
within-profile parabola residuals; the minima scatter around the fitted path by about five times that noise
(ra5_theory_review item 7; R4 R2-M3). Here inference uses the between-budget residuals: a wild bootstrap-t with
HC2-rescaled residuals, HC2 standard errors recomputed in each draw and Webb six-point weights (copied from
code/analysis/ra5_theory/partial_id.py::fit_path, whose simulated coverage at these 9-10-point designs is about 90
percent). Any linear functional of (mu, e) -- ln M*(C) at a model's compute, or the path slope -- gets an equal-tailed
bootstrap-t interval, and 'adjusted draws' theta_hat - t*_b se_hat reproduce that interval and keep the joint
dependence across models (same b), so sample statistics (medians, shares) can be bootstrapped jointly.
Per-budget minima: module ra1_modelfree's reviewed estimator, valid (bracketed) budgets only
(output/tables/ra1_modelfree_isoflop_budgets.csv).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import rb2common as C

RA1_BUDGETS = os.path.join(C.TABLES, "ra1_modelfree_isoflop_budgets.csv")


def ra1_minima(design, max_budget=None):
    b = pd.read_csv(RA1_BUDGETS)
    g = b[b["design"] == design].sort_values("budget_C")
    if max_budget is not None:
        g = g[g["budget_C"] <= max_budget * 1.0001]
    return pd.DataFrame(dict(C=g["budget_C"].values, lnC=np.log(g["budget_C"].values), lnM=np.log(g["Mstar"].values),
                             lnN=np.log(g["Nstar"].values)))


class Path:
    """ln M*(C) = mu + e (ln C - c0), fitted by OLS to per-budget minima; wild bootstrap-t (HC2, Webb), B draws."""

    def __init__(self, g, B=9999, seed=C.SEED, c0=None, label=""):
        self.label = label
        self.n = len(g)
        self.c0 = float(g["lnC"].max()) if c0 is None else float(c0)
        X = np.column_stack([np.ones(self.n), g["lnC"].values - self.c0])
        y = g["lnM"].values
        XtXi = np.linalg.inv(X.T @ X)
        self.Rw = XtXi @ X.T
        self.beta = self.Rw @ y
        res = y - X @ self.beta
        self.h = np.einsum("ij,jk,ik->i", X, XtXi, X)
        u = res / np.sqrt(1 - self.h)
        rng = np.random.default_rng(seed)
        V = C.WEBB[rng.integers(0, 6, (B, self.n))]
        Ystar = X @ self.beta + V * u
        self.draws = (self.Rw @ Ystar.T).T                    # B x 2
        self.res_star = Ystar - self.draws @ X.T               # B x n
        self.res = res
        self.B = B
        self.resid_sd = float(np.std(res, ddof=2))

    def _se(self, lam, res):
        """HC2 standard error of lam'beta for residual matrix res (..., n); lam (m, 2)."""
        cw = lam @ self.Rw                                      # m x n
        r2 = (res / np.sqrt(1 - self.h)) ** 2
        return np.sqrt(np.einsum("mn,...n->...m", cw ** 2, r2))

    def lnMstar(self, lnC):
        lnC = np.atleast_1d(np.asarray(lnC, float))
        lam = np.column_stack([np.ones(len(lnC)), lnC - self.c0])
        return lam @ self.beta

    def adjusted_draws(self, lnC):
        """(B, m) draws of ln M*(C) whose percentiles are the equal-tailed bootstrap-t interval."""
        lnC = np.atleast_1d(np.asarray(lnC, float))
        lam = np.column_stack([np.ones(len(lnC)), lnC - self.c0])
        th = lam @ self.beta
        se0 = self._se(lam, self.res)                           # m
        thb = self.draws @ lam.T                                # B x m
        seb = self._se(lam, self.res_star)                      # B x m
        t = (thb - th) / seb
        return th - t * se0

    def interval(self, lnC, level=0.95):
        d = self.adjusted_draws(lnC)
        a = (1 - level) / 2
        return np.percentile(d, [100 * a, 100 * (1 - a)], axis=0)

    def slope_interval(self, level=0.95):
        lam = np.array([[0.0, 1.0]])
        se0 = self._se(lam, self.res)[0]
        seb = self._se(lam, self.res_star)[:, 0]
        t = (self.draws[:, 1] - self.beta[1]) / seb
        a = (1 - level) / 2
        qlo, qhi = np.percentile(t, [100 * a, 100 * (1 - a)])
        return self.beta[1] - qhi * se0, self.beta[1] - qlo * se0

    def a_lnG(self):
        """Allocation exponent a (N* ~ C^a) and ln G of the equivalent hybrid technology: e = 1 - 2a;
        ln M*(C) = -2 ln G + (1 - 2a) ln(C/6)."""
        e = self.beta[1]
        a = (1 - e) / 2
        # mu = -2 lnG + e (c0 - ln 6)
        lnG = -(self.beta[0] - e * (self.c0 - np.log(6.0))) / 2
        return a, lnG
