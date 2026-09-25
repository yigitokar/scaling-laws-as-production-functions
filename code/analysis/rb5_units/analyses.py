"""analyses.py -- the round-3 analyses of module rb5_units on the budget-level decision units (fix list W14, W19):

  serving_ri       randomization inference for the serving coefficient (MacKinnon and Webb 2020): the model-level serving
                   code is permuted within strata of release year x compute, across models and so across developers; the
                   statistic is the OLS coefficient (RI-beta) and its developer-clustered t (RI-t), as in the restricted
                   wild cluster bootstrap test of the paper (ln w on the indicator, ln C and year effects)
  trend_boot       year-specific compute-weighted shares and medians, and their changes, with a developer-cluster
                   bootstrap (18 developers resampled with replacement)
  within_dev       the annual growth of the unit wedge with developer fixed effects (with and without log compute),
                   against pooled OLS, with developer-cluster bootstrap intervals
  moe_rows         the 2024 and 2025 shares with the excluded mixture-of-experts models added, at active and at total
                   parameters (reference technology)
  rationale_rows   stated sizing rationales by year and the trend among decisions with a deployment rationale
  repetition       Muennighoff et al.'s effective data for models that repeated their corpus

All functions take the rb2_decisions outputs (clean_models.csv, decision_units.csv) as data frames. CPU only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import r5common as C

YEARS = (2023, 2024, 2025)


# ---------------------------------------------------------------- helpers
def s_agg(w, c):
    """Compute-weighted share sum (w+ - 1) C / sum w+ C with w+ = max(w, 1) (rb2_decisions.trend.s_agg)."""
    w = np.maximum(np.asarray(w, float), 1.0)
    c = np.asarray(c, float)
    return float(((w - 1) * c).sum() / (w * c).sum()) if len(w) else np.nan


def med_s(w):
    w = np.asarray(w, float)
    return float(np.median(C.share(w))) if len(w) else np.nan


def year_stats(U, M=None):
    """U: units (W, C_total, year); M: models (w_ref, Cmp, year). Returns dict of year statistics."""
    out = {}
    for y in YEARS:
        gu = U[U["year"] == y]
        out[f"n_units_{y}"] = len(gu)
        out[f"s_agg_units_{y}"] = s_agg(gu["W"], gu["C_total"])
        out[f"median_s_units_{y}"] = med_s(gu["W"])
        if M is not None:
            gm = M[M["year"] == y]
            out[f"n_models_{y}"] = len(gm)
            out[f"s_agg_models_{y}"] = s_agg(gm["w_ref"], gm["Cmp"])
            out[f"median_s_models_{y}"] = med_s(gm["w_ref"])
    for k in ("s_agg_units", "median_s_units", "s_agg_models", "median_s_models"):
        if f"{k}_2023" in out:
            out[f"{k}_d2324"] = out[f"{k}_2024"] - out[f"{k}_2023"]
            out[f"{k}_d2425"] = out[f"{k}_2025"] - out[f"{k}_2024"]
            out[f"{k}_d2325"] = out[f"{k}_2025"] - out[f"{k}_2023"]
    return out


def ols(y, X):
    XtX = X.T @ X
    b = np.linalg.solve(XtX, X.T @ y)
    return b, y - X @ b, np.linalg.inv(XtX)


def crv1_se(X, u, XtXi, groups, j):
    G = np.unique(groups)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in G:
        s = X[groups == g].T @ u[groups == g]
        meat += np.outer(s, s)
    n, k = X.shape
    V = XtXi @ meat @ XtXi * (len(G) / (len(G) - 1)) * ((n - 1) / (n - k))
    return float(np.sqrt(V[j, j]))


def design(df, cols, year_fe=True, extra=("lnC",), dev_fe=False):
    parts = [df[list(cols) + list(extra)].astype(float).values]
    if year_fe:
        parts.append(pd.get_dummies(df["year"], drop_first=True).astype(float).values)
    if dev_fe:
        parts.append(pd.get_dummies(df["dev"], drop_first=True).astype(float).values)
    parts.append(np.ones((len(df), 1)))
    return np.column_stack(parts)


# ---------------------------------------------------------------- W14: randomization inference
def strata(df, n_bins):
    """Release year x compute bins (quantiles of ln C within the year)."""
    if n_bins <= 1:
        return df["year"].astype(str).values
    out = np.empty(len(df), dtype=object)
    for y, g in df.groupby("year"):
        q = pd.qcut(g["lnC"].rank(method="first"), min(n_bins, len(g)), labels=False)
        out[df.index.get_indexer(g.index)] = [f"{y}-{int(v)}" for v in q]
    return out


def serving_ri(df, treat, B=9999, n_bins=3, seed=C.SEED, label=""):
    """df: rows with lnw, treat column (0/1), lnC, year, dev. Coefficient of treat in ln w ~ treat + ln C + year FE."""
    df = df.reset_index(drop=True)
    X = design(df, [treat])
    y = df["lnw"].values
    b, u, XtXi = ols(y, X)
    se = crv1_se(X, u, XtXi, df["dev"].values, 0)
    t = b[0] / se
    st = strata(df, n_bins)
    rng = np.random.default_rng(seed)
    d0 = df[treat].values.astype(float)
    bs, ts = np.empty(B), np.empty(B)
    idx_by = [np.where(st == s)[0] for s in np.unique(st)]
    for i in range(B):
        d = d0.copy()
        for ix in idx_by:
            d[ix] = d0[rng.permutation(ix)]
        Xb = X.copy()
        Xb[:, 0] = d
        bb, ub, XtXib = ols(y, Xb)
        bs[i] = bb[0]
        seb = crv1_se(Xb, ub, XtXib, df["dev"].values, 0)
        ts[i] = bb[0] / seb if seb > 0 else np.nan
    ok = np.isfinite(ts)
    n_strata_var = int(sum(len(np.unique(d0[ix])) > 1 for ix in idx_by))
    return dict(sample=label, treat=treat, n=len(df), clusters=int(df["dev"].nunique()),
                G_treated=int(df.loc[df[treat] == 1, "dev"].nunique()), n_treated=int(df[treat].sum()), coef=float(b[0]),
                se_crv1=se, t_crv1=float(t), strata=f"year x {n_bins} compute bins" if n_bins > 1 else "year",
                n_strata=len(idx_by), n_strata_with_variation=n_strata_var, B=B,
                p_ri_beta_two=float(np.mean(np.abs(bs) >= abs(b[0]) - 1e-12)),
                p_ri_beta_one=float(np.mean(bs >= b[0] - 1e-12)),
                p_ri_t_two=float(np.mean(np.abs(ts[ok]) >= abs(t) - 1e-12)),
                p_ri_t_one=float(np.mean(ts[ok] >= t - 1e-12)))


# ---------------------------------------------------------------- W19: trend inference
def trend_boot(U, M, B=9999, seed=C.SEED):
    """Developer-cluster bootstrap of year_stats: developers resampled with replacement (each draw keeps all units and
    models of each drawn developer, duplicated as drawn). Returns (point dict, draws frame)."""
    devs = np.array(sorted(M["dev"].unique()))
    Ug = {d: g for d, g in U.groupby("dev")}
    Mg = {d: g for d, g in M.groupby("dev")}
    rng = np.random.default_rng(seed)
    pt = year_stats(U, M)
    rows = []
    for _ in range(B):
        dd = rng.choice(devs, size=len(devs), replace=True)
        Ub = pd.concat([Ug[d] for d in dd if d in Ug], ignore_index=True)
        Mb = pd.concat([Mg[d] for d in dd], ignore_index=True)
        rows.append(year_stats(Ub, Mb))
    return pt, pd.DataFrame(rows)


def boot_summary(pt, draws, keys):
    out = []
    for k in keys:
        v = draws[k].values.astype(float)
        f = v[np.isfinite(v)]
        out.append(dict(stat=k, point=pt[k], lo=float(np.percentile(f, 2.5)) if len(f) else np.nan,
                        hi=float(np.percentile(f, 97.5)) if len(f) else np.nan, B=len(v), n_finite=len(f),
                        share_le0=float(np.mean(f <= 0)) if len(f) else np.nan))
    return pd.DataFrame(out)


def within_dev(U, B=9999, seed=C.SEED):
    """ln W on release date (years) over decision units: pooled OLS; developer fixed effects; both with ln C."""
    U = U.copy()
    U["t"] = pd.to_datetime(U["date"]).dt.year + (pd.to_datetime(U["date"]).dt.dayofyear - 1) / 365.25
    U["lnW"] = np.log(U["W"])
    U["lnC"] = np.log(U["C_total"])
    specs = [("pooled OLS", False, ()), ("pooled OLS, ln C", False, ("lnC",)), ("developer FE", True, ()),
             ("developer FE, ln C", True, ("lnC",))]

    def fit(df, fe, extra):
        X = design(df, ["t"], year_fe=False, extra=extra, dev_fe=fe)
        b, u, XtXi = ols(df["lnW"].values, X)
        return b[0], u, X, XtXi
    rows = []
    rng = np.random.default_rng(seed)
    devs = np.array(sorted(U["dev"].unique()))
    Ug = {d: g for d, g in U.groupby("dev")}
    multi = U.groupby("dev")["year"].nunique()
    def identified(df):
        return df.groupby("dev")["t"].nunique().max() > 1

    for lab, fe, extra in specs:
        b, u, X, XtXi = fit(U, fe, extra)
        se = crv1_se(X, u, XtXi, U["dev"].values, 0)
        bs = []
        for _ in range(B):
            dd = rng.choice(devs, size=len(devs), replace=True)
            parts = []
            for j, d in enumerate(dd):
                g = Ug[d].copy()
                g["dev"] = f"{d}#{j}"            # a developer drawn twice enters as two clusters with their own effects
                parts.append(g)
            Ub = pd.concat(parts, ignore_index=True)
            if fe and not identified(Ub):
                continue
            try:
                bb, _, _, _ = fit(Ub, fe, extra)
                bs.append(bb)
            except np.linalg.LinAlgError:
                continue
        bs = np.array(bs)
        rows.append(dict(spec=lab, n_units=len(U), n_developers=int(U["dev"].nunique()),
                         n_developers_multi_year=int((multi > 1).sum()), coef_per_year=float(b), se_crv1=se,
                         growth_per_year=float(np.exp(b)), growth_lo=float(np.exp(np.percentile(bs, 2.5))),
                         growth_hi=float(np.exp(np.percentile(bs, 97.5))), B_ok=len(bs)))
    return pd.DataFrame(rows)


def moe_rows(U, moe, lnw_fn):
    """Year statistics with the mixture-of-experts models of ra2's verified sample added as decision units of their own,
    their wedge under the reference at active or at total parameters, weighted by training compute (6 N_active D)."""
    rows = []
    base = year_stats(U)
    for conv in ("active", "total"):
        N = moe["N"] if conv == "active" else moe["N_total"]
        m = pd.DataFrame(dict(W=np.exp(lnw_fn(N.values, moe["D"].values)), C_total=moe["Cmp"].values,
                              year=moe["year"].values, dev=moe["dev"].values))
        st = year_stats(pd.concat([U[["W", "C_total", "year", "dev"]], m], ignore_index=True))
        for y in (2024, 2025):
            rows.append(dict(convention=f"MoE added, {conv} parameters", year=y, n_moe=int((m["year"] == y).sum()),
                             s_agg_units=st[f"s_agg_units_{y}"], median_s_units=st[f"median_s_units_{y}"],
                             s_agg_units_dense=base[f"s_agg_units_{y}"], median_s_units_dense=base[f"median_s_units_{y}"],
                             moe_compute_share=float(m.loc[m["year"] == y, "C_total"].sum() /
                                                     (m.loc[m["year"] == y, "C_total"].sum() + U.loc[U["year"] == y, "C_total"].sum()))))
    return pd.DataFrame(rows)


def rationale_rows(U):
    """By release year: shares of decision units citing each rationale (unweighted and compute-weighted), and the
    median and compute-weighted share among units with and without a deployment rationale."""
    rows = []
    for y in YEARS + ("all",):
        g = U if y == "all" else U[U["year"] == y]
        cw = g["C_total"] / g["C_total"].sum()
        r = dict(year=y, n_units=len(g))
        for k in ("R_CO", "R_INF", "R_DEV", "R_NONE", "deployment_rationale"):
            r[f"share_{k}"] = float(g[k].mean())
            r[f"share_{k}_cw"] = float((g[k] * cw).sum())
        for lab, m in (("dep", g["deployment_rationale"] == 1), ("nodep", g["deployment_rationale"] == 0),
                       ("co", g["R_CO"] == 1), ("none", g["R_NONE"] == 1)):
            r[f"n_{lab}"] = int(m.sum())
            r[f"median_s_{lab}"] = med_s(g.loc[m, "W"])
            r[f"s_agg_{lab}"] = s_agg(g.loc[m, "W"], g.loc[m, "C_total"])
        rows.append(r)
    return pd.DataFrame(rows)


def muennighoff_D(U_tokens, D, R_star=15.39):
    """Effective data of Muennighoff et al. (2023): D' = U + U R* (1 - exp(-R/R*)), R = D/U - 1 repetitions."""
    R = np.maximum(D / U_tokens - 1.0, 0.0)
    return U_tokens + U_tokens * R_star * (1 - np.exp(-R / R_star))


def serving_ri_dev(df, treat, B=9999, seed=C.SEED, weighted=True, label=""):
    """Developer-level randomization inference (the assignment unit is the developer, as in MacKinnon and Webb 2020):
    ln w is residualized on ln C and year effects over all models (without the serving indicator); the statistic is the
    (model-count weighted) slope of the developers' mean residual on their share of served models; developers' shares
    are permuted across developers. Respects the clustering that the model-level permutation ignores."""
    df = df.reset_index(drop=True)
    X = design(df, [], year_fe=True)
    _, u, _ = ols(df["lnw"].values, X)
    g = pd.DataFrame(dict(dev=df["dev"], r=u, d=df[treat].astype(float))).groupby("dev").agg(r=("r", "mean"), d=("d", "mean"),
                                                                                             n=("r", "size"))
    wts = g["n"].values.astype(float) if weighted else np.ones(len(g))

    def slope(d):
        dm = np.average(d, weights=wts)
        rm = np.average(g["r"].values, weights=wts)
        return float(np.sum(wts * (d - dm) * (g["r"].values - rm)) / np.sum(wts * (d - dm) ** 2))
    b0 = slope(g["d"].values)
    rng = np.random.default_rng(seed)
    bs = np.array([slope(rng.permutation(g["d"].values)) for _ in range(B)])
    return dict(sample=label, treat=treat, n=len(df), clusters=len(g), G_treated=int((g["d"] > 0).sum()),
                statistic="developer-level slope, " + ("weighted by models" if weighted else "unweighted"), coef=b0, B=B,
                p_ri_beta_two=float(np.mean(np.abs(bs) >= abs(b0) - 1e-12)), p_ri_beta_one=float(np.mean(bs >= b0 - 1e-12)))
