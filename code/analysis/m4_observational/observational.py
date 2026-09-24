"""Observational (cross-lab) estimators of the same elasticities as the experiments.

Model (model_spec.md, with a benchmark-logit output):
    y_i = a + theta_N n_i + theta_D d_i + omega_i + e_i,   omega_i = lab/family recipe quality (Hicks-neutral TFP
    on the odds scale), e_i = evaluation noise. Labs choose (n, d) knowing omega (transmission), so pooled OLS is
    biased by Cov(omega, inputs) (Proposition 2); the sign depends on the behavioural regime (target / budget /
    funding).  Compute-only specification: y_i = a + theta_C c_i + ..., c = ln 6 + n + d.

Estimators (all with standard errors clustered by developer unless noted):
  OLS            pooled
  Year FE        + release-year effects (common technical change)
  Developer FE   within-developer
  Family FE      within-family (ObsScaling families = product line x generation; Mundlak / Hoch covariance)
  Mundlak        pooled + family means of (n, d): slopes = FE; coefficients on means = between - within
  Lab x period   developer x release-year effects
  FD-gen         first differences across generations within (product line, size tier)
  ACF-gen        GMM on quasi-differences with omega_t = c0 + rho omega_{t-1} + xi_t; N predetermined (planned size
                 menu), D flexible; instruments (1, n_t, n_{t-1}, d_{t-1}); rho profiled on a grid (2SLS for given rho)
  IV             ln C instrumented by the log FLOP/s-per-USD of the best data-centre accelerator available 6 months
                 before release (and, where known, of the model's own training hardware); AR confidence sets
  EIV            reliability-corrected OLS/FE (classical error in c with variance from (i) observed discrepancies
                 between independent compute measures and (ii) Epoch confidence classes)
  Selection      Notable-only subsample; OP-style control for the estimated notability propensity
  Reverse        Nerlove reverse regression of c on y (bound under the capability-target regime)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import ols


def _dummies(d, col):
    return pd.get_dummies(d[col].astype(str), prefix=col, drop_first=True, dtype=float)


def _fit(d, out, xs, fe=None, cluster="developer", extra=None):
    X = d[xs].copy()
    if extra is not None:
        X = pd.concat([X, extra], axis=1)
    if fe is not None:
        for f in (fe if isinstance(fe, list) else [fe]):
            X = pd.concat([X, _dummies(d, f)], axis=1)
    X = sm.add_constant(X, has_constant="add")
    return ols(d[out], X, cluster=None if cluster is None else d[cluster])


def _collect(rows, est, out, sample, res, d, spec):
    for x in ["n", "d", "c"]:
        if x in res.params.index:
            par = {"n": "theta_N", "d": "theta_D", "c": "theta_C"}[x]
            rows.append(dict(estimator=est, output=out, sample=sample, spec=spec, param=par,
                             est=float(res.params[x]), se=float(res.bse[x]), n_obs=int(res.nobs),
                             n_clusters=int(d["developer"].nunique()), n_families=int(d["family"].nunique())))
    if "n" in res.params.index and "d" in res.params.index:
        b = res.params[["n", "d"]].values
        V = res.cov_params().loc[["n", "d"], ["n", "d"]].values
        w = np.array([0.5, 0.5])
        rows.append(dict(estimator=est, output=out, sample=sample, spec=spec, param="theta_C_ray",
                         est=float(w @ b), se=float(np.sqrt(w @ V @ w)), n_obs=int(res.nobs),
                         n_clusters=int(d["developer"].nunique()), n_families=int(d["family"].nunique())))
        # test of the compute-aggregation restriction theta_N = theta_D
        r = np.array([1.0, -1.0])
        t = (r @ b) / np.sqrt(r @ V @ r)
        rows.append(dict(estimator=est, output=out, sample=sample, spec=spec, param="thetaN_minus_thetaD",
                         est=float(r @ b), se=float(np.sqrt(r @ V @ r)), n_obs=int(res.nobs),
                         n_clusters=int(d["developer"].nunique()), n_families=int(d["family"].nunique())))


# ----------------------------------------------------------------------------- dynamic panel across generations
def generation_panel(df, out):
    """Cell-level panel: unit = (product line, size tier), t = generation. Averages duplicates within a cell."""
    d = df.dropna(subset=[out, "n", "d", "gen"]).copy()
    cell = d.groupby(["line", "tier", "gen"]).agg(y=(out, "mean"), n=("n", "mean"), d=("d", "mean"),
                                                  developer=("developer", "first"), family=("family", "first"),
                                                  t=("t", "mean")).reset_index()
    cell = cell.sort_values(["line", "tier", "gen"])
    g = cell.groupby(["line", "tier"])
    for v in ["y", "n", "d", "gen", "t"]:
        cell[v + "_l"] = g[v].shift(1)
        cell[v + "_l2"] = g[v].shift(2)
    pairs = cell.dropna(subset=["y_l"]).copy()
    return cell, pairs


def fd_gen(pairs):
    p = pairs.copy()
    p["dy"], p["dn"], p["dd"] = p.y - p.y_l, p.n - p.n_l, p.d - p.d_l
    X = sm.add_constant(p[["dn", "dd"]])
    return ols(p.dy, X, cluster=p.line), p


def acf_gen(pairs, rho_grid=np.linspace(-0.5, 1.5, 201)):
    """Profile GMM: for each rho, 2SLS of (y - rho y_l) on (n - rho n_l, d - rho d_l, 1) with instruments
    (1, n, n_l, d_l); pick rho minimising the (homoskedastic) GMM objective. Returns params and objective path."""
    p = pairs.dropna(subset=["n_l", "d_l"])
    Z = np.column_stack([np.ones(len(p)), p.n, p.n_l, p.d_l])
    W = np.linalg.pinv(Z.T @ Z / len(p))
    best = None
    path = []
    for rho in rho_grid:
        yq = (p.y - rho * p.y_l).values
        X = np.column_stack([np.ones(len(p)), p.n - rho * p.n_l, p.d - rho * p.d_l])
        # 2SLS given rho
        PZ = Z @ np.linalg.pinv(Z.T @ Z) @ Z.T
        b = np.linalg.lstsq(PZ @ X, PZ @ yq, rcond=None)[0]
        g = Z.T @ (yq - X @ b) / len(p)
        J = float(g @ W @ g)
        path.append((rho, J))
        if best is None or J < best[0]:
            best = (J, rho, b)
    J, rho, b = best
    return dict(theta_N=b[1], theta_D=b[2], rho=rho, c0=b[0], J=J, n_pairs=len(p)), np.array(path)


def acf_bootstrap(pairs, B=400, seed=0):
    rng = np.random.default_rng(seed)
    lines = pairs.line.unique()
    draws = []
    for _ in range(B):
        pick = rng.choice(lines, len(lines), replace=True)
        bp = pd.concat([pairs[pairs.line == l] for l in pick])
        try:
            r, _ = acf_gen(bp, rho_grid=np.linspace(-0.5, 1.5, 81))
            draws.append(r)
        except Exception:
            continue
    return pd.DataFrame(draws)


# ----------------------------------------------------------------------------- IV
def iv_compute(d, out, z, controls=None, cluster="developer", grid=np.linspace(-1.0, 2.5, 701)):
    """Just-identified IV for y = a + theta_C c + controls, instrument z. Returns 2SLS estimate, cluster SE,
    first-stage F (cluster-robust t^2 on z), and the Anderson-Rubin 95% confidence set (grid inversion)."""
    d = d.dropna(subset=[out, "c", z]).copy()
    C = pd.DataFrame(index=d.index)
    if controls is not None:
        for f in (controls if isinstance(controls, list) else [controls]):
            if f in ("t",):
                C = pd.concat([C, d[[f]]], axis=1)
            else:
                C = pd.concat([C, _dummies(d, f)], axis=1)
    C = sm.add_constant(C, has_constant="add")
    g = d[cluster]
    fs = ols(d.c, pd.concat([C, d[[z]]], axis=1), cluster=g)
    F = float(fs.tvalues[z] ** 2)
    # 2SLS via linearmodels
    from linearmodels.iv import IV2SLS
    m = IV2SLS(d[out], C, d[["c"]], d[[z]]).fit(cov_type="clustered", clusters=pd.factorize(g)[0])
    est, se = float(m.params["c"]), float(m.std_errors["c"])
    # AR set: reject theta0 if z significant in y - theta0 c on (C, z)
    acc = []
    for th in grid:
        r = ols(d[out] - th * d.c, pd.concat([C, d[[z]]], axis=1), cluster=g)
        if abs(r.tvalues[z]) < 1.96:
            acc.append(th)
    if len(acc) == 0:
        ar = (np.nan, np.nan, "empty")
    else:
        acc = np.array(acc)
        gaps = np.any(np.diff(acc) > 1.5 * (grid[1] - grid[0]))
        unbounded = acc.min() <= grid[0] + 1e-9 or acc.max() >= grid[-1] - 1e-9
        ar = (acc.min(), acc.max(), "unbounded" if unbounded else ("disjoint" if gaps else "interval"))
    return dict(est=est, se=se, F=F, ar_lo=ar[0], ar_hi=ar[1], ar_type=ar[2], n_obs=len(d),
                n_clusters=int(g.nunique()), fs_coef=float(fs.params[z]), fs_se=float(fs.bse[z]))


# ----------------------------------------------------------------------------- main runner
def run_estimators(df, out, sample_name, mask):
    d = df[mask].dropna(subset=[out, "n", "d"]).copy()
    rows = []
    specs = [("OLS", None), ("Year FE", "year"), ("Developer FE", "developer"), ("Family FE", "family"),
             ("Lab x period FE", "devyear")]
    d["devyear"] = d.developer + "_" + d.year.astype(str)
    for est, fe in specs:
        _collect(rows, est, out, sample_name, _fit(d, out, ["n", "d"], fe=fe), d, "n,d")
        _collect(rows, est, out, sample_name, _fit(d, out, ["c"], fe=fe), d, "c only")
    # Mundlak: family means
    for x in ["n", "d", "c"]:
        d[x + "_bar"] = d.groupby("family")[x].transform("mean")
    rm = _fit(d, out, ["n", "d", "n_bar", "d_bar"])
    _collect(rows, "Mundlak", out, sample_name, rm, d, "n,d")
    for x in ["n_bar", "d_bar"]:
        rows.append(dict(estimator="Mundlak", output=out, sample=sample_name, spec="n,d", param="between_minus_within_" + x[0],
                         est=float(rm.params[x]), se=float(rm.bse[x]), n_obs=int(rm.nobs),
                         n_clusters=int(d.developer.nunique()), n_families=int(d.family.nunique())))
    rmc = _fit(d, out, ["c", "c_bar"])
    _collect(rows, "Mundlak", out, sample_name, rmc, d, "c only")
    rows.append(dict(estimator="Mundlak", output=out, sample=sample_name, spec="c only", param="between_minus_within_c",
                     est=float(rmc.params["c_bar"]), se=float(rmc.bse["c_bar"]), n_obs=int(rmc.nobs),
                     n_clusters=int(d.developer.nunique()), n_families=int(d.family.nunique())))
    # Reverse regression (Nerlove): c on y (+ family FE variant): 1/slope
    for est, fe in [("Reverse (c on y)", None), ("Reverse (c on y), family FE", "family")]:
        r = _fit(d, "c", [out], fe=fe)
        b, se = float(r.params[out]), float(r.bse[out])
        rows.append(dict(estimator=est, output=out, sample=sample_name, spec="c only", param="theta_C",
                         est=1.0 / b, se=se / b ** 2, n_obs=int(r.nobs), n_clusters=int(d.developer.nunique()),
                         n_families=int(d.family.nunique())))
    # Selection: Notable only; OP-style propensity control
    dn = d[d.notable]
    if len(dn) >= 8:
        r = _fit(dn, out, ["c"], cluster=None)
        _collect(rows, "Notable only", out, sample_name, r, dn, "c only")
    try:
        import statsmodels.formula.api as smf
        dd = d.copy()
        dd["notable_i"] = dd.notable.astype(int)
        pm = smf.logit("notable_i ~ n + d + t", data=dd).fit(disp=0)
        dd["ps"] = pm.predict(dd)
        dd["ps2"] = dd.ps ** 2
        r = _fit(dd, out, ["n", "d", "ps", "ps2"])
        _collect(rows, "OP-style selection control", out, sample_name, r, dd, "n,d")
        r = _fit(dd, out, ["c", "ps", "ps2"])
        _collect(rows, "OP-style selection control", out, sample_name, r, dd, "c only")
    except Exception as ex:  # perfect prediction in small samples
        print("  [selection] propensity model failed:", ex)
    return pd.DataFrame(rows), d


def eiv_correction(df, out, mask, sigma_u):
    """Classical errors-in-variables correction of the compute-only slope: theta = b / lambda,
    lambda = 1 - sigma_u^2 / Var(c | controls). Pooled and within-family."""
    d = df[mask].dropna(subset=[out, "c"]).copy()
    rows = []
    for est, fe in [("OLS", None), ("Family FE", "family")]:
        r = _fit(d, out, ["c"], fe=fe)
        if fe is None:
            vc = d.c.var()
        else:
            vc = (d.c - d.groupby(fe).c.transform("mean")).var() * (len(d) - 1) / (len(d) - d[fe].nunique())
        for lab, s in sigma_u.items():
            lam = 1 - s ** 2 / vc
            rows.append(dict(estimator=f"{est}, EIV-corrected ({lab})", output=out, param="theta_C",
                             est=float(r.params["c"]) / lam if lam > 0 else np.nan,
                             se=float(r.bse["c"]) / lam if lam > 0 else np.nan, reliability=lam, sigma_u=s,
                             var_c=vc, n_obs=int(r.nobs)))
    return pd.DataFrame(rows)
