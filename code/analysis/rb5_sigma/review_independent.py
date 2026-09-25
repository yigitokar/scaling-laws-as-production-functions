"""review_independent.py -- independent re-implementation of the key rb5_sigma statistics (package WP2-review).

Written by the independent reviewer of module rb5_sigma. It imports NO code of rb5_sigma, rb1_sigmaC, ra1_modelfree or
rb4_chinflop: it reads their published CSV outputs only and re-implements every estimator from scratch
(inverse-variance and DerSimonian-Laird pooling with the modified Hartung-Knapp-Sidik-Jonkman interval; a three-level
REML meta-regression; CR2 with Cholesky-based adjustment matrices as in Pustejovsky and Tipton 2018, where rb1 uses
symmetric square roots; Bell-McCaffrey Satterthwaite degrees of freedom; the restricted wild cluster bootstrap-t with
Webb weights by full enumeration; the linear statistics of the joint bootstrap draws).

Inputs (read-only): output/tables/rb1_sigmaC_budgets_input.csv (44 budget-level estimates + Porian),
rb5_sigma_joint_draws.csv and _joint_budgets.csv (the builder's draws; their per-budget s.e. are checked against
rb1's inputs here), rb4_chinflop_specgrid.csv, _review_checks.csv, data/processed/rb4_chinflop/override_new_isoflop_summary.csv,
ra1_modelfree_farseer_{sigma_slope,pool,sensitivity}.csv, rb1_sigmaC_study_level.csv.
Output: output/tables/rb5_sigma_review_independent.csv (statistic, reviewer's value, builder's value, abs diff).
No experiment (m9) data are read. CPU only, single process; run under nice -n 10.
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import chi2, t as tdist

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
T = os.path.join(ROOT, "output", "tables")
BT = os.environ.get("RB5_REVIEW_BUILDER_TABLES", T)     # builder's tables to compare with (default: the repo's)
OUT = os.environ.get("RB5_REVIEW_OUT", T)
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
STUDY = {"Chinchilla": "Hoffmann", "Llama 3": "Meta", "Marin, Comma": "Marin", "Marin, DCLM": "Marin",
         "Marin, Nemotron-CC": "Marin", "Farseer (local path)": "Farseer"}
ROWS = []


def rec(item, stat, mine, builder=np.nan, tol=5e-4, note=""):
    d = abs(mine - builder) if np.isfinite(builder) else np.nan
    ROWS.append(dict(item=item, statistic=stat, reviewer=mine, builder=builder, abs_diff=d,
                     agree=(bool(d <= tol) if np.isfinite(d) else None), tol=tol, note=note))
    flag = "" if not np.isfinite(d) else ("  OK" if d <= tol else "  <-- DIFF")
    print(f"[{item}] {stat}: reviewer {mine:.6g}  builder {builder:.6g}{flag}", flush=True)


def sig(S):
    return 2.0 / (2.0 + np.asarray(S, float))


def dsig(S):
    return -2.0 / (2.0 + np.asarray(S, float)) ** 2


# ============================================================================ pooling
def pool(y, se):
    y, se = np.asarray(y, float), np.asarray(se, float)
    w = 1 / se ** 2
    k = len(y)
    muf = np.sum(w * y) / np.sum(w)
    Q = np.sum(w * (y - muf) ** 2)
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - (k - 1)) / c)
    ws = 1 / (se ** 2 + tau2)
    mu = np.sum(ws * y) / np.sum(ws)
    q = np.sum(ws * (y - mu) ** 2) / (k - 1)
    sehk = np.sqrt(max(q, 1.0) / np.sum(ws))
    tq = tdist.ppf(0.975, k - 1)
    return dict(muf=muf, sef=np.sqrt(1 / np.sum(w)), Q=Q, pQ=chi2.sf(Q, k - 1), tau=np.sqrt(tau2), mu=mu,
                lo=mu - tq * sehk, hi=mu + tq * sehk, sehk=sehk, w=w / w.sum())


# ============================================================================ meta-regression (own implementation)
def design_blocks(g):
    codes = pd.factorize(g)[0]
    return codes, [np.where(codes == j)[0] for j in range(codes.max() + 1)]


def Vmat(se2, td2, tw2, blocks):
    V = np.diag(se2 + tw2)
    for ix in blocks:
        V[np.ix_(ix, ix)] += td2
    return V


def reml_fit(y, X, se, groups, weighted=True):
    se2 = se ** 2 if weighted else np.zeros(len(y))
    _, blocks = design_blocks(groups)

    def negll(lt):
        td2, tw2 = np.exp(lt)
        V = Vmat(se2, td2, tw2, blocks)
        L = np.linalg.cholesky(V)
        Li = np.linalg.inv(L)
        Xs, ys = Li @ X, Li @ y
        XtX = Xs.T @ Xs
        b = np.linalg.solve(XtX, Xs.T @ ys)
        r = ys - Xs @ b
        return 0.5 * (2 * np.sum(np.log(np.diag(L))) + np.linalg.slogdet(XtX)[1] + r @ r)
    best = None
    lo = -30.0 if weighted else -18.0
    for a0 in np.linspace(-14, -3, 6):
        for b0 in np.linspace(-14, -3, 6):
            r = optimize.minimize(negll, [a0, b0], method="Nelder-Mead", options=dict(xatol=1e-9, fatol=1e-12, maxiter=4000))
            if best is None or r.fun < best.fun:
                best = r
    # boundary check (tau^2 = 0 corners)
    for fix in (0, 1):
        def f1(z, fix=fix):
            lt = np.array(best.x, float)
            lt[fix] = lo
            lt[1 - fix] = z[0]
            return negll(lt)
        r = optimize.minimize(f1, [best.x[1 - fix]], method="Nelder-Mead")
        if r.fun < best.fun - 1e-9:
            x = np.array(best.x, float)
            x[fix] = lo
            x[1 - fix] = r.x[0]
            best = optimize.OptimizeResult(x=x, fun=r.fun)
    td2, tw2 = np.exp(best.x)
    V = Vmat(se2, td2, tw2, blocks)
    return V, td2, tw2


def gls(y, X, V):
    Vi = np.linalg.inv(V)
    M = np.linalg.inv(X.T @ Vi @ X)
    return M @ X.T @ Vi @ y, M, Vi


def pinv_sqrt(A):
    ev, U = np.linalg.eigh((A + A.T) / 2)
    tol = 1e-10 * max(abs(ev).max(), 1e-300)
    f = np.where(ev > tol, 1 / np.sqrt(np.clip(ev, tol, None)), 0.0)
    return (U * f) @ U.T


CR2_KIND = os.environ.get("RB5_REVIEW_CR2", "whitened")


def cr2_forms(X, V, cl, c, kind=None):
    """Linear forms for CR2: estimate = a'y; CR2 variance = sum_j (p_j'y)^2; BM df under the working model.
    kind 'whitened' (default): A_j = Phi_j^{1/2} (Phi_j^{-1/2} M_j Phi_j^{-1/2})^{+1/2} Phi_j^{-1/2}, which is exactly
    unbiased for the slope under the working model even with absorbed design fixed effects (checked by the reviewer);
    kind 'cholesky': A_j = D_j' (D_j M_j D_j')^{+1/2} D_j with Phi_j = D_j'D_j (biased by -8 percent in the design-FE
    inverse-variance row, where M_j is singular); reported as a sensitivity."""
    kind = kind or CR2_KIND
    n = len(X)
    codes = pd.factorize(cl)[0]
    blocks = [np.where(codes == j)[0] for j in range(codes.max() + 1)]
    Phi = np.zeros_like(V)
    for ix in blocks:
        Phi[np.ix_(ix, ix)] = V[np.ix_(ix, ix)]
    W = np.zeros_like(V)
    for ix in blocks:
        W[np.ix_(ix, ix)] = np.linalg.inv(Phi[np.ix_(ix, ix)])
    M = np.linalg.inv(X.T @ W @ X)
    a = c @ M @ X.T @ W
    IH = np.eye(n) - X @ M @ X.T @ W
    Mfull = IH @ Phi @ IH.T
    P = []
    for ix in blocks:
        Pj, Mj = Phi[np.ix_(ix, ix)], Mfull[np.ix_(ix, ix)]
        if kind == "cholesky":
            D = np.linalg.cholesky(Pj).T                          # upper: Phi_j = D'D
            Aj = D.T @ pinv_sqrt(D @ Mj @ D.T) @ D
        else:
            ev, U = np.linalg.eigh((Pj + Pj.T) / 2)
            Ph, Pih = (U * np.sqrt(ev)) @ U.T, (U / np.sqrt(ev)) @ U.T
            Aj = Ph @ pinv_sqrt(Pih @ Mj @ Pih) @ Pih
        gj = Aj.T @ W[np.ix_(ix, ix)] @ X[ix] @ M @ c
        pj = np.zeros(n)
        pj[ix] = gj
        P.append(IH.T @ pj)
    P = np.array(P)
    Qm = P @ Phi @ P.T
    df = np.trace(Qm) ** 2 / np.sum(Qm * Qm)
    return a, P, df


def wild_enum(y, X, V, cl, c):
    """Restricted (c'beta = 0 by dropping the slope column) wild cluster bootstrap-t, Webb weights, all 6^G vectors."""
    a, P, df = cr2_forms(X, V, cl, c)
    t0 = (a @ y) / np.sqrt(np.sum((P @ y) ** 2))
    jb = int(np.argmax(np.abs(c)))
    Xr = np.delete(X, jb, axis=1)
    br, _, _ = gls(y, Xr, V)
    yr = Xr @ br
    er = y - yr
    codes = pd.factorize(cl)[0]
    G = codes.max() + 1
    Wv = np.array(list(itertools.product(WEBB, repeat=G)))
    Ys = yr[None, :] + Wv[:, codes] * er[None, :]
    ts = (Ys @ a) / np.sqrt(np.sum((Ys @ P.T) ** 2, axis=1))
    return float(np.mean(np.abs(ts) >= abs(t0) * (1 - 1e-10))), t0, df


def metareg(d, spec, clusters):
    y, se = d.y.values, d.se.values
    cc = d.log10C.values - 20.0
    if spec in ("weighted", "unweighted"):
        X = np.column_stack([np.ones(len(y)), cc])
        V, td2, tw2 = reml_fit(y, X, se, d.design.values, weighted=(spec == "weighted"))
        jb = 1
    elif spec == "balanced":
        X = np.column_stack([np.ones(len(y)), cc])
        _, td2, tw2 = reml_fit(y, X, se, d.design.values, weighted=True)
        k = d.groupby("design")["y"].transform("size").values
        vbar = d.groupby("design")["se"].transform(lambda s_: np.mean(s_ ** 2)).values
        V = np.diag(k * (vbar + td2 + tw2))
        jb = 1
    elif spec in ("fe_weighted", "fe_unweighted"):
        Dm = pd.get_dummies(d.design).astype(float).values
        X = np.column_stack([Dm, cc])
        V = np.diag(se ** 2) if spec == "fe_weighted" else np.eye(len(y))
        jb = X.shape[1] - 1
    b, M, Vi = gls(y, X, V)
    c = np.zeros(X.shape[1])
    c[jb] = 1.0
    a, P, df = cr2_forms(X, V, clusters, c)
    se_r = np.sqrt(np.sum((P @ y) ** 2))
    p_cr2 = 2 * tdist.sf(abs(b[jb] / se_r), df)
    p_w, t0, _ = wild_enum(y, X, V, clusters, c)
    out = dict(slope=b[jb], se_cr2=se_r, df=df, p_cr2=p_cr2, p_wild=p_w)
    if spec == "weighted":
        c0 = np.zeros(X.shape[1])
        c0[0] = 1.0
        for L in (19.0, 20.0, 21.0):
            cv = c0.copy()
            cv[1] = L - 20.0
            a2, P2, df2 = cr2_forms(X, V, clusters, cv)
            est = cv @ b
            s2 = np.sqrt(np.sum((P2 @ y) ** 2))
            q = tdist.ppf(0.975, df2)
            out[f"pred{L:.0f}"] = (est, est - q * s2, est + q * s2, df2, np.sqrt(cv @ M @ cv))
    return out


def main():
    bi = pd.read_csv(os.path.join(T, "rb1_sigmaC_budgets_input.csv"))
    d = bi[~bi.porian].reset_index(drop=True)
    assert len(d) == 44, len(d)

    # ================================================================== T1.1: joint draws
    jd = pd.read_csv(os.path.join(BT, "rb5_sigma_joint_draws.csv"))
    jbud = pd.read_csv(os.path.join(BT, "rb5_sigma_joint_budgets.csv"))
    jl = pd.read_csv(os.path.join(BT, "rb5_sigma_joint_linear.csv")).set_index("statistic")

    def members(des, cmin=0.0):
        g = jbud[(jbud.design == des) & (jbud.budget_C >= cmin * (1 - 1e-9))].sort_values("budget_C")
        return list(g.budget_C)

    def devmat(mem):
        cols, est, se_pub, lc = [], [], [], []
        for des, C in mem:
            b = jbud[(jbud.design == des) & np.isclose(jbud.budget_C, C, rtol=1e-6)].iloc[0]
            dr = jd[(jd.design == des) & np.isclose(jd.budget_C, C, rtol=1e-6)].sort_values("draw")
            dS = dr.S_draw.values - b.S_pop
            cols.append(dsig(b.S) * dS)
            est.append(b.sigma)
            if des in STUDY:
                r = bi[(bi.design == des) & np.isclose(bi.budget_C, C, rtol=1e-6)].iloc[0]
                se_pub.append(r.se)
            else:
                se_pub.append(np.nan)
            lc.append(np.log10(C))
        return np.column_stack(cols), np.array(est), np.array(se_pub), np.array(lc)

    # per-budget s.e. from the draws vs rb1's published inputs (all 36 IsoFLOP budgets)
    mx = 0.0
    for des in ["Chinchilla", "Llama 3", "Marin, Comma", "Marin, DCLM", "Marin, Nemotron-CC"]:
        mem = [(des, C) for C in members(des)]
        Dv, est, sep, _ = devmat(mem)
        se_d = Dv.std(axis=0, ddof=1)
        r = bi[bi.design == des].sort_values("budget_C")
        mx = max(mx, np.max(np.abs(se_d - r.se.values)), np.max(np.abs(est - r.y.values)))
    rec("T1.1", "max |draw s.e. or estimate - rb1 input| over 36 IsoFLOP budgets", mx, 0.0, tol=1e-6)

    mem = [(des, C) for des in ("Chinchilla", "Llama 3") for C in members(des)]
    Dv, est, sep, lc = devmat(mem)
    desn = np.array([m[0] for m in mem])
    X = np.column_stack([(desn == "Chinchilla").astype(float), (desn == "Llama 3").astype(float), lc - 20.0])
    w = 1 / sep ** 2
    A = np.linalg.solve(X.T @ (X * w[:, None]), (X * w[:, None]).T)
    a = A[2]
    slope = a @ est
    r = est - X @ (A @ est)
    s2 = np.sum(w * r ** 2) / (len(est) - 3)
    se_ind = np.sqrt(np.sum(a ** 2 * Dv.var(axis=0, ddof=1)))
    se_joint = np.std(Dv @ a, ddof=1)
    Sig = np.cov(Dv.T)
    # GLS dispersion with the joint covariance
    Si = np.linalg.inv(Sig)
    bg = np.linalg.solve(X.T @ Si @ X, X.T @ Si @ est)
    rg = est - X @ bg
    s2_joint = rg @ Si @ rg / (len(est) - 3)
    rec("T1.1", "CL3 FE drift (IV weights)", slope, jl.loc[jl.index.str.startswith("Chinchilla + Llama 3 drift: design FE slope, inverse")].estimate.iloc[0], tol=1e-6)
    rec("T1.1", "CL3 FE drift: model s.e. unscaled", np.sqrt(np.sum(a ** 2 / w)), jl.loc[jl.index.str.startswith("Chinchilla + Llama 3 drift: design FE slope, inverse")].se_model_unscaled.iloc[0], tol=1e-6)
    rec("T1.1", "CL3 FE drift: residual dispersion s2", s2, jl.loc[jl.index.str.startswith("Chinchilla + Llama 3 drift: design FE slope, inverse")].resid_var_scale.iloc[0], tol=1e-4)
    rec("T1.1", "CL3 FE drift: model s.e. scaled", np.sqrt(np.sum(a ** 2 / w) * s2), jl.loc[jl.index.str.startswith("Chinchilla + Llama 3 drift: design FE slope, inverse")].se_model_scaled.iloc[0], tol=1e-6)
    rec("T1.1", "CL3 FE drift: bootstrap s.e. independent", se_ind, jl.loc[jl.index.str.startswith("Chinchilla + Llama 3 drift: design FE slope, inverse")].se_boot_independent.iloc[0], tol=1e-6)
    rec("T1.1", "CL3 FE drift: bootstrap s.e. joint", se_joint, jl.loc[jl.index.str.startswith("Chinchilla + Llama 3 drift: design FE slope, inverse")].se_boot_joint.iloc[0], tol=1e-6)
    rec("T1.1", "CL3 FE drift: joint s.e. x sqrt(s2)", se_joint * np.sqrt(s2), np.nan)
    rec("T1.1", "CL3 FE drift: GLS dispersion with joint covariance", s2_joint, np.nan)
    rec("T1.1", "CL3 FE drift: GLS slope with joint covariance", bg[2], np.nan)
    rec("T1.1", "CL3 FE drift: GLS s.e. joint cov x sqrt(max(s2_joint,1))",
        np.sqrt(np.linalg.inv(X.T @ Si @ X)[2, 2] * max(s2_joint, 1)), np.nan)
    rec("T1.1", "CL3 FE drift: joint s.e. x sqrt(s2_joint)", se_joint * np.sqrt(max(s2_joint, 1)), np.nan)
    Hh = X @ A
    s2_mom = np.sum(w * r ** 2) / np.trace(np.diag(w) @ (np.eye(len(est)) - Hh) @ Sig @ (np.eye(len(est)) - Hh).T)
    rec("T1.1", "CL3 FE drift: moment dispersion under the joint covariance", s2_mom, np.nan)
    rec("T1.1", "CL3 FE drift: joint s.e. x sqrt(moment dispersion) (recommended 0.014)", se_joint * np.sqrt(max(s2_mom, 1)), np.nan)

    memt = [(des, C) for des in ("Chinchilla", "Llama 3") for C in members(des, 6e20)]
    Dt, et, st, _ = devmat(memt)
    wt = (1 / st ** 2) / np.sum(1 / st ** 2)
    mu = wt @ et
    sj = np.std(Dt @ wt, ddof=1)
    tp = jl.loc[jl.index.str.startswith("Top-budget mean: Chinchilla + Llama 3")].iloc[0]
    rec("T1.1", "top-budget mean", mu, tp.estimate, tol=1e-6)
    rec("T1.1", "top-budget mean: joint s.e.", sj, tp.se_boot_joint, tol=1e-6)
    q4 = tdist.ppf(0.975, 4)
    rec("T1.1", "top-budget: t4 joint interval lo", mu - q4 * sj, tp.estimate - q4 * tp.se_boot_joint, tol=1e-6)
    rec("T1.1", "top-budget: t4 joint interval hi", mu + q4 * sj, tp.estimate + q4 * tp.se_boot_joint, tol=1e-6)
    qq = np.percentile(Dt @ wt, [2.5, 97.5])
    rec("T1.1", "top-budget: percentile lo", mu + qq[0], tp.lo_percentile, tol=1e-6)
    rec("T1.1", "top-budget: percentile hi", mu + qq[1], tp.hi_percentile, tol=1e-6)
    rec("T1.3", "Llama 3 weight share", wt[[m[0] == "Llama 3" for m in memt]].sum(), tp.weight_share_llama3, tol=1e-6)
    pops = np.array([jbud[(jbud.design == d_) & np.isclose(jbud.budget_C, C, rtol=1e-6)].sigma_pop.iloc[0] for d_, C in memt])
    qp = np.percentile(pops @ wt + Dt @ wt, [2.5, 97.5])      # delta-method draws around the bootstrap population
    rec("T1.1", "top-budget: bootstrap population value", pops @ wt, np.nan)
    rec("T1.1", "top-budget: plain percentile lo (delta-method draws)", qp[0], np.nan)
    rec("T1.1", "top-budget: plain percentile hi (delta-method draws)", qp[1], np.nan)
    # Q with the joint covariance (does truncation at one still bind?)
    St = np.cov(Dt.T)
    Sti = np.linalg.inv(St)
    one = np.ones(len(et))
    mug = (one @ Sti @ et) / (one @ Sti @ one)
    Qj = (et - mug) @ Sti @ (et - mug)
    rec("T1.1", "top-budget: GLS mean with joint covariance", mug, np.nan)
    rec("T1.1", "top-budget: Q with joint covariance (df 4)", Qj, np.nan)

    # ================================================================== T1.3: top-budget set
    tb = pd.read_csv(os.path.join(BT, "rb5_sigma_top_budget.csv")).set_index("set")
    far = d[d.design == "Farseer (local path)"].sort_values("budget_C")
    f21 = far[np.isclose(far.budget_C, 1e21)].iloc[0]
    f520 = far[np.isclose(far.budget_C, 5e20)].iloc[0]
    rec("T1.3", "Farseer level 5e20", f520.y, tb.loc["Farseer local path level at 5e+20 FLOP", "mean_fixed"], tol=1e-6)
    rec("T1.3", "Farseer level 1e21", f21.y, tb.loc["Farseer local path level at 1e+21 FLOP", "mean_fixed"], tol=1e-6)
    rec("T1.3", "Farseer s.e. 1e21", f21.se, tb.loc["Farseer local path level at 1e+21 FLOP", "se_fixed"], tol=1e-6)
    top = d[d.design.isin(["Chinchilla", "Llama 3"]) & (d.budget_C >= 6e20 * (1 - 1e-9))]
    p5 = pool(top.y, top.se)
    rec("T1.3", "top-5 FE mean", p5["muf"], tb.loc["Chinchilla + Llama 3, budgets >= 6e20 (primary)", "mean_fixed"], tol=1e-6)
    rec("T1.3", "top-5 HKSJ lo", p5["lo"], tb.loc["Chinchilla + Llama 3, budgets >= 6e20 (primary)", "lo_hksj"], tol=1e-6)
    rec("T1.3", "top-5 HKSJ hi", p5["hi"], tb.loc["Chinchilla + Llama 3, budgets >= 6e20 (primary)", "hi_hksj"], tol=1e-6)
    rec("T1.3", "top-5 Q", p5["Q"], tb.loc["Chinchilla + Llama 3, budgets >= 6e20 (primary)", "Q"], tol=1e-4)
    p6 = pool(list(top.y) + [f21.y], list(top.se) + [f21.se])
    key = "Chinchilla + Llama 3 + Farseer 1e21"
    rec("T1.3", "top-5 + Farseer 1e21: FE mean", p6["muf"], tb.loc[key, "mean_fixed"], tol=1e-6)
    rec("T1.3", "top-5 + Farseer 1e21: RE mean", p6["mu"], tb.loc[key, "mean_re"], tol=1e-6)
    rec("T1.3", "top-5 + Farseer 1e21: HKSJ lo", p6["lo"], tb.loc[key, "lo_hksj"], tol=1e-6)
    rec("T1.3", "top-5 + Farseer 1e21: HKSJ hi", p6["hi"], tb.loc[key, "hi_hksj"], tol=1e-6)
    rec("T1.3", "top-5 + Farseer 1e21: Q", p6["Q"], tb.loc[key, "Q"], tol=1e-4)
    rec("T1.3", "top-5 + Farseer 1e21: p(Q)", p6["pQ"], tb.loc[key, "p_Q"], tol=1e-6)
    # 132-run: R4's substitution of the 6e20 budget alone vs the builder's full re-estimate
    b132 = jbud[(jbud.design == "Chinchilla (132 runs)") & (jbud.budget_C >= 6e20 * (1 - 1e-9))].sort_values("budget_C")
    ll = top[top.design == "Llama 3"]
    p132 = pool(list(b132.sigma) + list(ll.y), list(b132.se_sigma) + list(ll.se))
    rec("T1.3", "top-budget mean, 132-run Chinchilla (all 3 top budgets re-estimated)", p132["mu"],
        tb.loc["Chinchilla (132 runs) + Llama 3, budgets >= 6e20", "mean_re"], tol=1e-6)
    ch = top[top.design == "Chinchilla"].sort_values("budget_C")
    ysub = [b132.sigma.iloc[0]] + list(ch.y.iloc[1:]) + list(ll.y)
    ssub = [b132.se_sigma.iloc[0]] + list(ch.se.iloc[1:]) + list(ll.se)
    rec("T1.3", "top-budget mean, only the 6e20 budget substituted (R4's calculation)", pool(ysub, ssub)["mu"], np.nan)

    # ================================================================== T1.2: meta-regression (own REML/CR2/wild)
    ms = pd.read_csv(os.path.join(BT, "rb5_sigma_metareg_slopes.csv"))
    msa = ms[ms["sample"] == "all six designs (excl. Porian)"]
    d["study"] = d.design.map(STUDY)
    dS = d.copy()
    dS["y"] = 2 * (1 / d.y - 1)
    dS["se"] = 2 * d.se / d.y ** 2
    for spec, dd_, key in (("balanced", d, "balanced"), ("weighted", dS, "weighted_S")):
        for cname, col in (("studies (4; Marin one cluster)", "study"), ("designs (6)", "design")):
            o = metareg(dd_, spec, dd_[col].values)
            b = msa[(msa.spec_key == key) & (msa.clusters == cname)].iloc[0]
            rec("T1.2", f"{key} slope [{cname}]", o["slope"], b.slope, tol=5e-4)
            rec("T1.2", f"{key} CR2 p [{cname}]", o["p_cr2"], b.p_cr2, tol=0.01)
            rec("T1.2", f"{key} wild p enumerated [{cname}]", o["p_wild"], b.p_wild_enum, tol=0.01)
    # subsamples (primary spec): budgets <= 3e20; leave out each design's largest budget; IsoFLOP-only (5 designs)
    subs = [("budgets <= 3e20 (common window)", d[d.log10C <= np.log10(3e20) + 1e-9].reset_index(drop=True)),
            ("leave out each design's largest budget", d[~d.index.isin(d.groupby("design")["log10C"].idxmax().values)].reset_index(drop=True)),
            ("IsoFLOP designs only (excl. Farseer path)", d[d.design != "Farseer (local path)"].reset_index(drop=True))]
    for lab, ds in subs:
        for cname, col in (("studies", "study"), ("designs", "design")):
            o = metareg(ds, "weighted", ds[col].values)
            b = ms[(ms["sample"] == lab) & (ms.spec_key == "weighted") & ms.clusters.str.startswith(cname)].iloc[0]
            rec("T1.2", f"[{lab}] weighted slope [{cname}]", o["slope"], b.slope, tol=5e-4)
            rec("T1.2", f"[{lab}] weighted CR2 p [{cname}]", o["p_cr2"], b.p_cr2, tol=0.01)
            rec("T1.2", f"[{lab}] weighted wild p enumerated [{cname}]", o["p_wild"], b.p_wild_enum, tol=0.01)
    for spec in ("weighted", "unweighted", "fe_weighted", "fe_unweighted"):
        for cname, col in (("studies (4; Marin one cluster)", "study"), ("designs (6)", "design")):
            o = metareg(d, spec, d[col].values)
            b = msa[(msa.spec_key == spec) & (msa.clusters == cname)].iloc[0]
            rec("T1.2", f"{spec} slope [{cname}]", o["slope"], b.slope, tol=2e-4)
            rec("T1.2", f"{spec} CR2 s.e. [{cname}]", o["se_cr2"], b.se_cr2, tol=5e-4)
            rec("T1.2", f"{spec} CR2 df [{cname}]", o["df"], b.df_cr2, tol=0.05)
            rec("T1.2", f"{spec} CR2 p [{cname}]", o["p_cr2"], b.p_cr2, tol=0.01)
            rec("T1.2", f"{spec} wild p enumerated [{cname}]", o["p_wild"], b.p_wild_enum, tol=0.01)
            if spec == "fe_weighted":
                a_, P_, df_ = cr2_forms(np.column_stack([pd.get_dummies(d.design).astype(float).values, d.log10C.values - 20.0]),
                                        np.diag(d.se.values ** 2), d[col].values,
                                        np.r_[np.zeros(d.design.nunique()), 1.0], kind="cholesky")
                se_c = np.sqrt(np.sum((P_ @ d.y.values) ** 2))
                rec("T1.2", f"fe_weighted CR2 p, Cholesky-form adjustment (sensitivity) [{cname}]",
                    2 * tdist.sf(abs(o["slope"] / se_c), df_), np.nan)
            if spec == "weighted":
                pr = pd.read_csv(os.path.join(BT, "rb5_sigma_metareg_predictions.csv"))
                for L in (19, 20, 21):
                    pb = pr[(pr.clusters == cname) & np.isclose(pr.log10C, L)].iloc[0]
                    est, lo, hi, df2, sem = o[f"pred{L}"]
                    rec("T1.2", f"pooled line at 1e{L} [{cname}]: est", est, pb.est, tol=5e-4)
                    rec("T1.2", f"pooled line at 1e{L} [{cname}]: CR2 lo", lo, pb.lo_cr2, tol=3e-3)
                    rec("T1.2", f"pooled line at 1e{L} [{cname}]: CR2 hi", hi, pb.hi_cr2, tol=3e-3)
                    rec("T1.2", f"pooled line at 1e{L} [{cname}]: model s.e.", sem, pb.se_model, tol=5e-4)

    # ================================================================== T1.4-T1.6: study level (own DL)
    summ = pd.read_csv(os.path.join(ROOT, "data", "processed", "rb4_chinflop", "override_new_isoflop_summary.csv")).set_index("design")
    fs = pd.read_csv(os.path.join(T, "ra1_modelfree_farseer_sigma_slope.csv"))
    fp = pd.read_csv(os.path.join(T, "ra1_modelfree_farseer_pool.csv")).set_index("conv")
    fd = fs[(fs.conv == "ne") & fs.variant.str.startswith("primary (wild")].iloc[0]
    far_e = (fd.sigma_slope, max(fd.se, fp.loc["ne", "se"]))
    mar = summ.loc[["Marin, Comma", "Marin, DCLM", "Marin, Nemotron-CC"]]
    marin_e = (mar.sigma_re.mean(), mar.se_sigma_re.mean())
    base = [(summ.loc["Chinchilla", "sigma_re"], summ.loc["Chinchilla", "se_sigma_re"]),
            (summ.loc["Llama 3", "sigma_re"], summ.loc["Llama 3", "se_sigma_re"]), marin_e, far_e]
    pp = pool([b[0] for b in base], [b[1] for b in base])
    rb1s = pd.read_csv(os.path.join(T, "rb1_sigmaC_study_level.csv"))
    p0 = rb1s[rb1s.variant.str.startswith("PRIMARY")].iloc[0]
    rec("T1.4", "primary study-level mean (check vs rb1)", pp["mu"], p0.mean_re, tol=1e-6)
    rec("T1.4", "primary study-level Q (check vs rb1)", pp["Q"], p0.Q, tol=1e-4)

    # design RE means on budgets <= 3e20: DL on S with robust bootstrap variances
    def rse(v):
        return (np.percentile(v, 75) - np.percentile(v, 25)) / 1.349

    def design_sub(des, cut):
        g = jbud[(jbud.design == des) & (jbud.budget_C <= cut)].sort_values("budget_C")
        S = g.S.values
        vS = np.array([rse(jd[(jd.design == des) & np.isclose(jd.budget_C, C, rtol=1e-6)].S_draw.values) ** 2 for C in g.budget_C])
        w = 1 / vS
        muf = np.sum(w * S) / np.sum(w)
        Q = np.sum(w * (S - muf) ** 2)
        tau2 = max(0.0, (Q - (len(S) - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
        ws = 1 / (vS + tau2)
        mu = np.sum(ws * S) / np.sum(ws)
        se = np.sqrt(1 / np.sum(ws))
        return float(sig(mu)), float(abs(dsig(mu)) * se), len(S)
    parts = pd.read_csv(os.path.join(BT, "rb5_sigma_study_parts_le3e20.csv")).set_index("design")
    sub = {}
    for des in ["Chinchilla", "Llama 3", "Marin, Comma", "Marin, DCLM", "Marin, Nemotron-CC"]:
        full = design_sub(des, np.inf)
        s3 = design_sub(des, 3e20 * (1 + 1e-9))
        sub[des] = s3
        rec("T1.4", f"{des}: RE sigma all budgets (vs rb4 summary)", full[0], summ.loc[des, "sigma_re"], tol=1e-6)
        rec("T1.4", f"{des}: RE sigma budgets <= 3e20", s3[0], parts.loc[des, "sigma_le3e20"], tol=1e-6)
        rec("T1.4", f"{des}: RE s.e. budgets <= 3e20", s3[1], parts.loc[des, "se_le3e20"], tol=1e-6)
    fr = pd.read_csv(os.path.join(BT, "rb5_sigma_farseer_restricted.csv")).set_index("estimate")
    far3 = (float(fr.loc["fd_le3e20", "sigma"]), float(fr.loc["fd_le3e20", "se"]))
    mar3 = (np.mean([sub[m][0] for m in ("Marin, Comma", "Marin, DCLM", "Marin, Nemotron-CC")]),
            np.mean([sub[m][1] for m in ("Marin, Comma", "Marin, DCLM", "Marin, Nemotron-CC")]))
    p3 = pool([sub["Chinchilla"][0], sub["Llama 3"][0], mar3[0], far3[0]],
              [sub["Chinchilla"][1], sub["Llama 3"][1], mar3[1], far3[1]])
    sl = pd.read_csv(os.path.join(BT, "rb5_sigma_study_level.csv"))
    b14 = sl[(sl.group == "T1.4") & sl.variant.str.contains(r"\(primary")].iloc[0]
    rec("T1.4", "study-level mean <= 3e20", p3["mu"], b14.mean_re, tol=1e-6)
    rec("T1.4", "study-level <= 3e20 HKSJ lo", p3["lo"], b14.lo_hksj, tol=1e-6)
    rec("T1.4", "study-level <= 3e20 HKSJ hi", p3["hi"], b14.hi_hksj, tol=1e-6)
    rec("T1.4", "study-level <= 3e20 Q", p3["Q"], b14.Q, tol=1e-4)
    # sensitivity: Farseer entered at its full-path first derivative, and IsoFLOP-only
    p3b = pool([sub["Chinchilla"][0], sub["Llama 3"][0], mar3[0], far_e[0]],
               [sub["Chinchilla"][1], sub["Llama 3"][1], mar3[1], far_e[1]])
    rec("T1.4", "study-level <= 3e20 with Farseer full-path FD", p3b["mu"], np.nan)
    rec("T1.4", "Farseer weight share in the <= 3e20 mean", p3["w"][3] if p3["tau"] == 0 else np.nan, float(b14.weight_share_farseer) if "weight_share_farseer" in b14 else np.nan, tol=1e-4)
    sens_ = pd.read_csv(os.path.join(T, "ra1_modelfree_farseer_sensitivity.csv"))
    v_ = sens_[sens_.key == "path_sigma_mean_poolC|ne"].value
    se_bw = np.sqrt(far3[1] ** 2 + (v_.max() - v_.min()) ** 2 / 12)
    p3c = pool([sub["Chinchilla"][0], sub["Llama 3"][0], mar3[0], far3[0]], [sub["Chinchilla"][1], sub["Llama 3"][1], mar3[1], se_bw])
    brow = sl[(sl.group == "T1.4") & sl.variant.str.contains("review")]
    rec("T1.4", "study-level <= 3e20, Farseer bandwidth as uncertainty [review]", p3c["mu"], float(brow.mean_re.iloc[0]) if len(brow) else np.nan, tol=1e-6)
    rec("T1.4", "  HKSJ lo", p3c["lo"], float(brow.lo_hksj.iloc[0]) if len(brow) else np.nan, tol=1e-6)
    rec("T1.4", "  HKSJ hi", p3c["hi"], float(brow.hi_hksj.iloc[0]) if len(brow) else np.nan, tol=1e-6)
    # budgets 1e19 to 3e20 only (the abstract's window: drop the 3e18/6e18 budgets too)
    def design_win(des, lo_, hi_):
        g = jbud[(jbud.design == des) & (jbud.budget_C >= lo_) & (jbud.budget_C <= hi_)].sort_values("budget_C")
        S = g.S.values
        vS = np.array([rse(jd[(jd.design == des) & np.isclose(jd.budget_C, C, rtol=1e-6)].S_draw.values) ** 2 for C in g.budget_C])
        w = 1 / vS
        muf = np.sum(w * S) / np.sum(w)
        Q = np.sum(w * (S - muf) ** 2)
        tau2 = max(0.0, (Q - (len(S) - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
        ws = 1 / (vS + tau2)
        mu = np.sum(ws * S) / np.sum(ws)
        return float(sig(mu)), float(abs(dsig(mu)) * np.sqrt(1 / np.sum(ws))), len(S)
    win = {des: design_win(des, 1e19 * (1 - 1e-9), 3e20 * (1 + 1e-9)) for des in sub}
    marw = (np.mean([win[m][0] for m in ("Marin, Comma", "Marin, DCLM", "Marin, Nemotron-CC")]),
            np.mean([win[m][1] for m in ("Marin, Comma", "Marin, DCLM", "Marin, Nemotron-CC")]))
    pw = pool([win["Chinchilla"][0], win["Llama 3"][0], marw[0], far3[0]], [win["Chinchilla"][1], win["Llama 3"][1], marw[1], far3[1]])
    rec("T1.4", "study-level mean on budgets 1e19-3e20 only (sensitivity)", pw["mu"], np.nan, note=f"[{pw['lo']:.3f}, {pw['hi']:.3f}] Q {pw['Q']:.2f}")

    # T1.5: Chinchilla grid
    G = pd.read_csv(os.path.join(T, "rb4_chinflop_specgrid.csv"))
    G = G[(G.convention == "N_F (T4)") & G.ok]
    means, Qs = [], []
    for _, g in G.iterrows():
        if g.k_valid < 8:
            continue
        o = pool([g.sigma_re] + [b[0] for b in base[1:]], [g.se_sigma_re] + [b[1] for b in base[1:]])
        means.append(o["mu"])
        Qs.append(o["Q"])
    chk = pd.read_csv(os.path.join(T, "rb4_chinflop_review_checks.csv"))
    import re
    for pre, stat in (("R11 count-free", "RE sigma*"), ("R12 T4-consistent", "RE sigma*, N_F"), ("R13 five highest-loss", "RE sigma*, N_F")):
        c = chk[chk.check.str.startswith(pre) & chk.statistic.str.startswith(stat)].iloc[0]
        se_ = float(re.search(r"RE [0-9.]+ \(([0-9.]+)\)", c.note).group(1))
        o = pool([c.value] + [b[0] for b in base[1:]], [se_] + [b[1] for b in base[1:]])
        means.append(o["mu"])
        Qs.append(o["Q"])
    rg = pd.read_csv(os.path.join(BT, "rb5_sigma_study_ranges.csv"))
    fam = rg[rg.family.str.startswith("Chinchilla windows, samples and profile memberships")].iloc[0]
    rec("T1.5", "grid+R11-R13 (k>=8): mean min", min(means), fam.mean_min, tol=1e-5)
    rec("T1.5", "grid+R11-R13 (k>=8): mean max", max(means), fam.mean_max, tol=1e-5)
    rec("T1.5", "grid+R11-R13 (k>=8): Q min", min(Qs), fam.Q_min, tol=1e-3)
    rec("T1.5", "grid+R11-R13 (k>=8): Q max", max(Qs), fam.Q_max, tol=1e-3)
    rec("T1.5", "grid+R11-R13 (k>=8): number of rows", len(means), fam.n, tol=0)
    allm = rg[rg.family.str.startswith("All: rb1's rows")].iloc[0]
    rb1m = rb1s.mean_re.values
    rec("T1.5", "all variants: mean min", min(min(means), rb1m.min()), allm.mean_min, tol=1e-5)
    rec("T1.5", "all variants: mean max", max(max(means), rb1m.max()), allm.mean_max, tol=1e-5)

    # T1.6
    sens = pd.read_csv(os.path.join(T, "ra1_modelfree_farseer_sensitivity.csv"))
    v = sens[sens.key == "path_sigma_mean_poolC|ne"]
    lo_, hi_ = v.value.min(), v.value.max()
    se6 = np.sqrt(fp.loc["ne", "se"] ** 2 + (hi_ - lo_) ** 2 / 12)
    p6b = pool([b[0] for b in base[:3]] + [(lo_ + hi_) / 2], [b[1] for b in base[:3]] + [se6])
    b16 = sl[(sl.group == "T1.6") & sl.variant.str.contains("midpoint of the bandwidth range, uniform")].iloc[0]
    rec("T1.6", "Farseer bandwidth range lo", lo_, b16.bw_lo, tol=1e-6)
    rec("T1.6", "Farseer bandwidth range hi", hi_, b16.bw_hi, tol=1e-6)
    rec("T1.6", "study-level mean, bandwidth as uncertainty", p6b["mu"], b16.mean_re, tol=1e-6)
    rec("T1.6", "  HKSJ lo", p6b["lo"], b16.lo_hksj, tol=1e-6)
    rec("T1.6", "  HKSJ hi", p6b["hi"], b16.hi_hksj, tol=1e-6)

    # ================================================================== T1.7: lower bound
    pi = pd.read_csv(os.path.join(BT, "rb5_sigma_pi_set.csv"))
    Ct = np.array([C for _, C in memt])
    lcC = np.log10(Ct)
    centres = {"1e21": 21.0, "compute-weighted (w ~ C)": np.sum(Ct * lcC) / Ct.sum(),
               "precision-weighted": np.sum(wt * lcC), "log10 of mean C": np.log10(Ct.mean())}
    for nm, c0 in centres.items():
        for L in (23, 24):
            rec("T1.7", f"lower bound at 1e{L}, anchor {nm} (log10 {c0:.3f})", mu + slope * (L - c0), np.nan)
    print(pi[pi.C.isin([1e23, 1e24])][["anchor", "log10_anchor", "C", "sigma_lower"]].to_string() if "log10_anchor" in pi else pi.head(12).to_string())

    R = pd.DataFrame(ROWS)
    R.to_csv(os.path.join(OUT, "rb5_sigma_review_independent.csv"), index=False, float_format="%.6g")
    print(f"\n{int((R.agree == True).sum())} agree, {int((R.agree == False).sum())} differ, {int(R.agree.isna().sum())} reviewer-only")  # noqa: E712


if __name__ == "__main__":
    main()
