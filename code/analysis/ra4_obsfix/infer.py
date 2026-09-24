"""Few-cluster and design-conditional inference for linear estimators (vectorized; no statsmodels in the loops).

Used for the design-matched comparison of Online Appendix E (old Table 8).  Conventions:
  * CR1 ("all k"): statsmodels' default small-sample factor G/(G-1) * (n-1)/(n-k) with k = every column of the
    dummy-variable design, INCLUDING fixed-effect dummies. This is what Table 8 reported.
  * CR1 ("nested"): the same, but fixed effects nested within the clusters are not counted in k (the convention of
    fixest/reghdfe and of Cameron and Miller 2015, Sec. VI; families, developers and developer x year cells are
    nested within developer clusters).  A bootstrap-t is invariant to this constant; a normal/t reference is not.
  * CV3: jackknife cluster-robust variance ((G-1)/G) sum_g (b_(-g) - b)^2 (MacKinnon, Nielsen and Webb 2023).
  * WCR: restricted wild cluster bootstrap-t with Webb six-point weights (Webb 2023; Cameron, Gelbach and Miller
    2008), optionally with weights drawn at a finer level (the subcluster bootstrap of MacKinnon and Webb 2018),
    and optionally COMBINED with independent draws of the experimental benchmark so that the test of
    H0: plim(obs) = plim(bench) accounts for both sources of uncertainty (R1 comment 9(e)).
  * G*: effective number of clusters of Carter, Schnepel and Steigerwald (2017) with rho = 0, computed from the
    cluster shares of the partial leverage of the regressor of interest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NESTED_FE = {"family", "developer", "devyear"}     # fixed effects nested within developer clusters


def build_X(d, xs, fe=None, extra=()):
    """Dummy-variable design: const, regressors xs, extra regressors, FE dummies (drop-first).
    Returns X (ndarray), names, k_all, k_nested."""
    cols, names = [np.ones(len(d))], ["const"]
    for x in list(xs) + list(extra):
        cols.append(d[x].to_numpy(float))
        names.append(x)
    k_nested = len(cols)
    if fe is not None:
        dm = pd.get_dummies(d[fe].astype(str), prefix=fe, drop_first=True, dtype=float)
        for c in dm.columns:
            cols.append(dm[c].to_numpy(float))
            names.append(c)
        if fe not in NESTED_FE:
            k_nested = len(cols)
    X = np.column_stack(cols)
    return X, names, X.shape[1], k_nested


def cluster_ids(labels):
    return pd.factorize(pd.Series(labels).astype(str))[0]


class LinFit:
    """OLS b = (X'X)^+ X'y with cluster-robust variances for coefficient j."""

    def __init__(self, X, y, g, j):
        self.X, self.y, self.g, self.j = X, np.asarray(y, float), np.asarray(g), j
        self.n, self.k = X.shape
        self.XtXi = np.linalg.pinv(X.T @ X)
        self.b = self.XtXi @ X.T @ self.y
        self.u = self.y - X @ self.b
        self.h = X @ self.XtXi[:, j]                 # b_j = h'y
        self.G = int(self.g.max()) + 1
        self.Ind = np.zeros((self.G, self.n))
        self.Ind[self.g, np.arange(self.n)] = 1.0
        # rank / identification check: partial variation of column j given the others
        others = np.delete(X, j, axis=1)
        xt = X[:, j] - others @ np.linalg.lstsq(others, X[:, j], rcond=None)[0]
        self.partial_ss = float(xt @ xt)
        self.identified = self.partial_ss > 1e-8 * max(1.0, float(X[:, j] @ X[:, j]))
        self.xt = xt

    def factor(self, k_dof):
        return self.G / (self.G - 1) * (self.n - 1) / max(self.n - k_dof, 1)

    def vjj(self, U, k_dof):
        """CR1 variance of b_j for residual matrix U (n x B)."""
        S = self.Ind @ (self.h[:, None] * U)
        return self.factor(k_dof) * (S ** 2).sum(0)

    def se_cr1(self, k_dof):
        return float(np.sqrt(self.vjj(self.u[:, None], k_dof)[0]))

    def leverage_shares(self, labels):
        """Share of the partial variation (x_j residualized on the other columns) by cluster label."""
        s = pd.Series(self.xt ** 2).groupby(np.asarray(labels)).sum()
        return (s / s.sum()).sort_values(ascending=False)

    def g_star(self):
        """Carter-Schnepel-Steigerwald (2017) effective number of clusters, rho = 0 (gamma_g = sum_{i in g} h_i^2)."""
        gam = self.Ind @ (self.h ** 2)
        gbar = gam.mean()
        Gamma = np.mean((gam - gbar) ** 2) / gbar ** 2
        return float(self.G / (1 + Gamma))


def jackknife(d, yname, xs, fe, par, cluster="developer", extra=()):
    """Leave-one-cluster-out estimates of coefficient `par` and the CV3 standard error (centered at the full-sample b).
    Clusters whose removal leaves the coefficient unidentified are reported as NaN."""
    X, names, _, _ = build_X(d, xs, fe, extra)
    j = names.index(par)
    b_full = float((np.linalg.pinv(X.T @ X) @ X.T @ d[yname].to_numpy(float))[j])
    labs = d[cluster].astype(str).to_numpy()
    out = {}
    for c in pd.unique(labs):
        dd = d[labs != c]
        Xc, nc, _, _ = build_X(dd, xs, fe, extra)
        keep = np.abs(Xc).sum(0) > 0
        Xc = Xc[:, keep]
        nc = [n_ for n_, k_ in zip(nc, keep) if k_]
        if par not in nc:
            out[c] = np.nan
            continue
        jj = nc.index(par)
        others = np.delete(Xc, jj, axis=1)
        xt = Xc[:, jj] - others @ np.linalg.lstsq(others, Xc[:, jj], rcond=None)[0]
        if xt @ xt < 1e-8:
            out[c] = np.nan
            continue
        out[c] = float((np.linalg.pinv(Xc.T @ Xc) @ Xc.T @ dd[yname].to_numpy(float))[jj])
    loo = pd.Series(out)
    G = len(loo)
    ok = loo.dropna()
    cv3 = float(np.sqrt((G - 1) / G * np.sum((ok - b_full) ** 2))) if len(ok) == G else np.nan
    return b_full, loo, cv3


def wild_test(fit: LinFit, theta0, B, rng, k_dof, bench_draws=None, bench_var=0.0, wlevel=None, restricted=True,
              weights="webb"):
    """Wild (cluster) bootstrap-t test of H0: plim b_j = plim bench, with bench = theta0 at its point estimate.

    Statistic T = (b_j - theta0) / sqrt(V_CR1(b_j) + bench_var).  Bootstrap: b*_j from the restricted (WCR) or
    unrestricted (WCU) wild cluster bootstrap with weights drawn per cluster (or per `wlevel` unit: subcluster
    bootstrap), and, if bench_draws are given, an independent benchmark draw theta*_b, so that
    T* = ((b*_j - theta0) - (theta*_b - theta0)) / sqrt(V*_CR1 + bench_var).  Two-sided symmetric p-value,
    p = #{|T*| >= |T|}/B (floor 1/B)."""
    X, y, j = fit.X, fit.y, fit.j
    lev = fit.g if wlevel is None else np.asarray(wlevel)
    Gw = int(lev.max()) + 1
    V0 = fit.vjj(fit.u[:, None], k_dof)[0]
    T_obs = (fit.b[j] - theta0) / np.sqrt(V0 + bench_var)
    if restricted:
        Xr = np.delete(X, j, axis=1)
        yr = y - theta0 * X[:, j]
        br = np.linalg.lstsq(Xr, yr, rcond=None)[0]
        base = yr - Xr @ br                      # restricted residuals
        center = theta0
    else:
        base = fit.u
        center = fit.b[j]
    wts = {"webb": np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)]),
           "rademacher": np.array([-1.0, 1.0])}[weights]
    Wg = wts[rng.integers(0, len(wts), size=(Gw, B))]
    E = base[:, None] * Wg[lev, :]
    num = fit.h @ E                                 # b*_j - center (restricted: h'fit_r = theta0)
    if not restricted:
        num = num                                   # b*_j - b_j
    U = E - X @ (fit.XtXi @ (X.T @ E))
    V = fit.vjj(U, k_dof)
    bd = np.zeros(B) if bench_draws is None else np.asarray(bench_draws, float) - theta0
    Tb = (num - bd) / np.sqrt(V + bench_var)
    _ = center
    p = float(np.mean(np.abs(Tb) >= abs(T_obs)))
    return dict(T=float(T_obs), p=max(p, 1.0 / B), p_raw=p, B=B, G_w=Gw)


def loo_maps(X, g, j):
    """Leave-one-cluster-out linear maps for coefficient j: b_(-c),j = h_c' y[g != c].  Columns that are all zero once
    cluster c is dropped (e.g. dummies of families nested in c) are removed, as in `jackknife`.  Returns None if some
    deletion leaves coefficient j unidentified (CV3 undefined)."""
    maps = []
    for c in range(int(g.max()) + 1):
        keep = g != c
        Xc = X[keep]
        colkeep = np.abs(Xc).sum(0) > 0
        if not colkeep[j]:
            return None
        Xc = Xc[:, colkeep]
        jj = int(np.sum(colkeep[:j]))
        others = np.delete(Xc, jj, axis=1)
        xt = Xc[:, jj] - others @ np.linalg.lstsq(others, Xc[:, jj], rcond=None)[0]
        if xt @ xt < 1e-8:
            return None
        maps.append((keep, (np.linalg.pinv(Xc.T @ Xc) @ Xc.T)[jj]))
    return maps


def cv3_from_maps(h, maps, Y):
    """CV3 standard error(s) of b_j = h'y for the columns of Y (n x B), centred at the full-sample estimate."""
    Y = np.atleast_2d(Y.T).T
    b = h @ Y
    G = len(maps)
    ss = np.zeros(Y.shape[1])
    for keep, hc in maps:
        ss += (hc @ Y[keep] - b) ** 2
    return np.sqrt((G - 1) / G * ss)


def wild_test_cv3(X, z, g, j, B, rng, bench_draws=None, bench_var=0.0):
    """Restricted wild-cluster bootstrap-t for H0: coefficient j = 0 in the regression of z = y_obs - y* (the bias
    outcome) on X, STUDENTIZED BY CV3 in both the sample and every bootstrap draw (reviewer addition, so that the
    p-value is coherent with the CV3 standard error reported next to it; cf. MacKinnon, Nielsen and Webb 2023).
    Webb weights by cluster g; optional independent benchmark draws (theta*_b - theta_b) enter the numerator and the
    fixed benchmark variance the denominator, as in `wild_test`.  Returns dict(T, p, cv3) or None if CV3 is undefined."""
    maps = loo_maps(X, g, j)
    if maps is None:
        return None
    XtXi = np.linalg.pinv(X.T @ X)
    h = X @ XtXi[:, j]
    z = np.asarray(z, float)
    cv3 = float(cv3_from_maps(h, maps, z[:, None])[0])
    T_obs = float(h @ z) / np.sqrt(cv3 ** 2 + bench_var)
    Xr = np.delete(X, j, axis=1)
    fr = Xr @ np.linalg.lstsq(Xr, z, rcond=None)[0]
    er = z - fr
    wts = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
    W = wts[rng.integers(0, 6, size=(int(g.max()) + 1, B))][g, :]
    Zs = fr[:, None] + er[:, None] * W
    num = h @ Zs                                       # = h'(er * w): the restricted DGP has coefficient j = 0
    se = cv3_from_maps(h, maps, Zs)
    bd = np.zeros(B) if bench_draws is None else np.asarray(bench_draws, float)
    Tb = (num - bd) / np.sqrt(se ** 2 + bench_var)
    p = float(np.mean(np.abs(Tb) >= abs(T_obs)))
    return dict(T=T_obs, p=max(p, 1.0 / B), p_raw=p, B=B, cv3=cv3)


def surface_design(E, center, dummies=True):
    """Quadratic (translog-in-odds) surface design for runs E (columns n, d, ds)."""
    n0, d0 = center
    x, z = E.n.to_numpy() - n0, E.d.to_numpy() - d0
    F = np.column_stack([x, z, x ** 2, z ** 2, x * z])
    cols = [np.ones(len(E)), F]
    if dummies and E.ds.nunique() > 1:
        cols.append(pd.get_dummies(E.ds, drop_first=True, dtype=float).to_numpy())
    return np.column_stack(cols), F


def obs_features(n, d, center):
    n0, d0 = center
    x, z = np.asarray(n) - n0, np.asarray(d) - d0
    return np.column_stack([x, z, x ** 2, z ** 2, x * z])


def surface_wild(Z, y, B, rng, clusters=None, hc2=True):
    """Design-conditional wild bootstrap of an OLS surface: y* = Z b + e_tilde * v, with e_tilde the HC2-rescaled
    residuals and v Webb weights drawn per cluster (or Rademacher per run if clusters is None).
    Returns point coefficients and a (p x B) matrix of bootstrap coefficients."""
    ZtZi = np.linalg.pinv(Z.T @ Z)
    b = ZtZi @ Z.T @ y
    e = y - Z @ b
    if hc2:
        hii = np.einsum("ij,jk,ik->i", Z, ZtZi, Z)
        e = e / np.sqrt(np.clip(1 - hii, 1e-6, None))
    if clusters is None:
        V = rng.choice(np.array([-1.0, 1.0]), size=(len(y), B))
    else:
        g = cluster_ids(clusters)
        V = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])[
            rng.integers(0, 6, size=(g.max() + 1, B))][g, :]
    Ys = (Z @ b)[:, None] + e[:, None] * V
    return b, ZtZi @ Z.T @ Ys


# ------------------------------------------------------------------------------------------ censored (Tobit) surface
def tobit_fit(Z, y, floor, x0=None):
    """Left-censored (at `floor`) Gaussian regression y = max(floor, Z b + s e). Returns b, s."""
    from scipy.optimize import minimize
    from scipy.stats import norm
    cens = y <= floor + 1e-9
    if x0 is None:
        b0 = np.linalg.lstsq(Z, y, rcond=None)[0]
        x0 = np.r_[b0, np.log(np.std(y - Z @ b0) + 1e-3)]

    def nll(p):
        b, s = p[:-1], np.exp(p[-1])
        mu = Z @ b
        ll = np.where(cens, norm.logcdf((floor - mu) / s), norm.logpdf((y - mu) / s) - np.log(s))
        return -ll.sum()

    def grad(p):
        b, s = p[:-1], np.exp(p[-1])
        mu = Z @ b
        zc = (floor - mu) / s
        zu = (y - mu) / s
        lam = np.exp(norm.logpdf(zc) - norm.logcdf(zc))       # inverse Mills
        dmu = np.where(cens, -lam / s, zu / s)                # d ll / d mu
        dls = np.where(cens, -lam * zc, zu ** 2 - 1)          # d ll / d log s
        return -np.r_[Z.T @ dmu, dls.sum()]

    r = minimize(nll, x0, jac=grad, method="BFGS", options=dict(gtol=1e-8, maxiter=2000))
    return r.x[:-1], float(np.exp(r.x[-1])), r


def tobit_expected(mu, s, floor):
    """E[max(floor, mu + s e)], e ~ N(0,1): the expected CLIPPED outcome, comparable to the clipped observed logits."""
    from scipy.stats import norm
    a = (floor - mu) / s
    return floor * norm.cdf(a) + mu * norm.sf(a) + s * norm.pdf(a)
