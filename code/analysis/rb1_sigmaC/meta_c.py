"""meta_c.py -- sigma*(C): meta-regression of budget-level model-free sigma*_b on log10 C (task 1; R1 New 2, R3 N1(a),
R2 Major 1 request 2) and the study-level headline interval (task 2; R1 New 2 request 4, R4 R2-M4).

Inputs (read-only, module ra1_modelfree):
  ra1_modelfree_isoflop_budgets.csv   per-budget sigma*_b and bootstrap s.e. (Chinchilla, Llama 3, Marin x3, Porian x2)
  ra1_modelfree_farseer_path.csv      Farseer's local expansion path: sigma*(C) at 8 compute levels (Hessian-based)
  ra1_modelfree_isoflop_summary.csv   design random-effects summaries
Model (primary, 'weighted'): three-level random-effects meta-regression (Konstantopoulos 2011)
    sigma*_bj = mu + beta (log10 C_b - 20) + theta_j + e_bj,  theta_j ~ N(0, tau_d^2),  e_bj ~ N(0, se_bj^2 + tau_w^2),
  fitted by REML; tau_w captures within-design (between-budget) excess heterogeneity. 'Unweighted' drops the
  within-budget s.e. (a linear mixed model with design random intercepts, se = 0, residual variance free).
  'Design-balanced' uses the correlated-effects RVE weights 1/(k_j (vbar_j + tau^2)) (Hedges, Tipton and Johnson 2010).
Inference: model-based GLS s.e.; CR2 cluster-robust s.e. by design (Bell and McCaffrey 2002; Pustejovsky and Tipton
  2018) with Bell-McCaffrey/Satterthwaite degrees of freedom (Tipton 2015), and by study (Marin = one study); a
  restricted wild cluster bootstrap-t (Webb weights) for H0: beta = 0.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import t as tdist, norm, chi2

import rb1common as cm

C0 = 20.0   # centring: log10 C = 20
DESIGNS = ["Chinchilla", "Llama 3", "Marin, Comma", "Marin, DCLM", "Marin, Nemotron-CC", "Farseer (local path)"]
STUDY = {"Chinchilla": "Hoffmann et al. (2022)", "Llama 3": "Grattafiori et al. (2024)",
         "Marin, Comma": "Marin (2026)", "Marin, DCLM": "Marin (2026)", "Marin, Nemotron-CC": "Marin (2026)",
         "Farseer (local path)": "Li et al. (2025)", "Porian, RefinedWeb": "Porian et al. (2024)",
         "Porian, OpenWebText2": "Porian et al. (2024)"}


# ============================================================================ data
def load_budgets(farseer_variant="hessian"):
    b = pd.read_csv(cm.RA1_BUDGETS)
    b = b.rename(columns={"sigma": "y", "se_sigma": "se"})
    b["log10C"] = np.log10(b["budget_C"])
    b["source"] = "IsoFLOP budget"
    b = b[["design", "log10C", "budget_C", "y", "se", "nleft", "nright", "Mstar", "source"]].copy()
    fp = pd.read_csv(cm.RA1_FAR_PATH)
    fp = fp[(fp.conv == "ne") & (fp.nvalid > 0) & fp.sigma_local_path.notna()].copy()
    f = pd.DataFrame(dict(design="Farseer (local path)", log10C=np.log10(fp.C), budget_C=fp.C, y=fp.sigma_local_path,
                          se=fp.se, nleft=np.nan, nright=np.nan, Mstar=fp.M_star_local, source="Farseer local path"))
    d = pd.concat([b, f], ignore_index=True)
    d["study"] = d.design.map(STUDY)
    d["porian"] = d.design.str.startswith("Porian")
    return d


# ============================================================================ estimation machinery
def _blocks(groups):
    g = pd.factorize(groups)[0]
    return g, [np.where(g == j)[0] for j in range(g.max() + 1)]


def _V(se2, tau_d2, tau_w2, idx_blocks, n):
    V = np.diag(se2 + tau_w2)
    for ix in idx_blocks:
        V[np.ix_(ix, ix)] += tau_d2
    return V


def _gls(y, X, V):
    Vi = np.linalg.inv(V)
    XtVi = X.T @ Vi
    A = XtVi @ X
    Ai = np.linalg.inv(A)
    beta = Ai @ (XtVi @ y)
    return beta, Ai, Vi


def reml(y, X, se, groups, weighted=True, fix_tau=None):
    """Three-level REML (design random intercept tau_d^2; within-design excess tau_w^2 on top of se^2).
    weighted=False: se set to 0 and tau_w^2 is the residual variance (linear mixed model)."""
    y, X, se = np.asarray(y, float), np.asarray(X, float), np.asarray(se, float)
    n, p = X.shape
    gidx, blocks = _blocks(groups)
    se2 = se ** 2 if weighted else np.zeros(n)

    def nll(th):
        td2, tw2 = th
        V = _V(se2, td2, tw2, blocks, n)
        try:
            sign, logdet = np.linalg.slogdet(V)
            beta, Ai, Vi = _gls(y, X, V)
        except np.linalg.LinAlgError:
            return 1e10
        r = y - X @ beta
        s2, ld2 = np.linalg.slogdet(np.linalg.inv(Ai))
        return 0.5 * (logdet + ld2 + r @ Vi @ r)

    if fix_tau is not None:
        td2, tw2 = fix_tau
    else:
        v0 = np.var(y)
        lb = (0.0, 0.0 if weighted else 1e-8)
        best = None
        for s0 in ([0.1 * v0, 0.1 * v0], [0.5 * v0, 1e-4], [1e-5, 0.5 * v0], [1e-6, 1e-6 if weighted else 0.5 * v0]):
            r = optimize.minimize(nll, s0, method="L-BFGS-B", bounds=[(lb[0], 1.0), (lb[1], 1.0)])
            if best is None or r.fun < best.fun:
                best = r
        td2, tw2 = best.x
    V = _V(se2, td2, tw2, blocks, n)
    beta, Ai, Vi = _gls(y, X, V)
    return dict(beta=beta, cov=Ai, V=V, Vi=Vi, tau_d2=float(td2), tau_w2=float(tw2), groups=gidx, blocks=blocks,
                y=y, X=X, nll=float(nll((td2, tw2))))


def wls(y, X, w, groups):
    """Weighted least squares with diagonal working weights w (V = diag(1/w)); design fixed effects go in X."""
    y, X, w = np.asarray(y, float), np.asarray(X, float), np.asarray(w, float)
    gidx, blocks = _blocks(groups)
    V = np.diag(1.0 / w)
    beta, Ai, Vi = _gls(y, X, V)
    r = y - X @ beta
    n, p = X.shape
    s2 = float(r @ Vi @ r / (n - p))
    return dict(beta=beta, cov=Ai, cov_scaled=Ai * max(s2, 1.0), s2=s2, V=V, Vi=Vi, groups=gidx, blocks=blocks, y=y, X=X)


def _msqrt(A, inv=False):
    """Symmetric square root; inv=True gives the Moore-Penrose inverse square root (zero on the null space, as in
    Pustejovsky and Tipton's CR2 with absorbed fixed effects)."""
    ev, U = np.linalg.eigh((A + A.T) / 2)
    tol = 1e-10 * max(float(np.max(np.abs(ev))), 1e-300)
    if inv:
        f = np.where(ev > tol, 1.0 / np.sqrt(np.clip(ev, tol, None)), 0.0)
    else:
        f = np.sqrt(np.clip(ev, 0.0, None))
    return (U * f) @ U.T


def cr2(fit, cvec, clusters=None):
    """CR2 cluster-robust variance of c'beta for a GLS fit with working covariance Phi = fit['V'] and weights
    W = fit['Vi'] (block-diagonal by cluster), with the Bell-McCaffrey Satterthwaite degrees of freedom computed
    under the working model (Bell and McCaffrey 2002; Pustejovsky and Tipton 2018; Tipton 2015).
    clusters: array of cluster labels (default: the fit's design groups). The working covariance is restricted to
    within-cluster blocks."""
    y, X, V = fit["y"], fit["X"], fit["V"]
    n = len(y)
    cl = fit["groups"] if clusters is None else pd.factorize(clusters)[0]
    blocks = [np.where(cl == j)[0] for j in range(cl.max() + 1)]
    Phi = np.zeros_like(V)
    for ix in blocks:
        Phi[np.ix_(ix, ix)] = V[np.ix_(ix, ix)]
    W = np.zeros_like(V)
    for ix in blocks:
        W[np.ix_(ix, ix)] = np.linalg.inv(Phi[np.ix_(ix, ix)])
    B = np.linalg.inv(X.T @ W @ X)
    beta = B @ X.T @ W @ y
    e = y - X @ beta
    H = X @ B @ X.T @ W
    IH = np.eye(n) - H
    Mfull = IH @ Phi @ IH.T
    cB = cvec @ B
    meat = 0.0
    P = []
    for ix in blocks:
        Phij = Phi[np.ix_(ix, ix)]
        Mj = Mfull[np.ix_(ix, ix)]
        Ph, Phi_inv_h = _msqrt(Phij), _msqrt(Phij, inv=True)
        K = Phi_inv_h @ Mj @ Phi_inv_h
        Aj = Ph @ _msqrt(K, inv=True) @ Phi_inv_h
        gj = Aj.T @ W[np.ix_(ix, ix)] @ X[ix] @ cB          # contribution: (gj' e_j)^2
        meat += float(gj @ e[ix]) ** 2
        pj = np.zeros(n)
        pj[ix] = gj
        P.append(IH.T @ pj)                                  # e_j'g_j = p_j' y with y ~ (., Phi)
    P = np.array(P)
    Q = P @ Phi @ P.T
    df = float(np.trace(Q) ** 2 / np.sum(Q * Q))
    return float(np.sqrt(meat)), df, float(cvec @ beta)


# ============================================================================ task 1: the meta-regression
def _Xmat(d, fe=False, slope_by_design=False):
    c = (d["log10C"].values - C0)[:, None]
    if fe:
        D = pd.get_dummies(d["design"]).astype(float).values
        if slope_by_design:
            return np.hstack([D, D * c]), None
        return np.hstack([D, c]), D.shape[1]
    return np.hstack([np.ones((len(d), 1)), c]), 1


def fit_spec(d, spec):
    """spec: 'weighted' (3-level REML, inverse variance), 'unweighted' (LMM, design RE), 'balanced' (CE-RVE weights),
    'fe_weighted' (design FE, 1/se^2, s.e. scaled by the residual variance when above one, as in R3's computation),
    'fe_unweighted' (design FE, OLS)."""
    y, se = d["y"].values, d["se"].values
    grp = d["design"].values
    if spec in ("weighted", "unweighted"):
        X, jb = _Xmat(d)
        fit = reml(y, X, se, grp, weighted=(spec == "weighted"))
        cvec = np.zeros(X.shape[1]); cvec[jb] = 1.0
        se_mb = float(np.sqrt(fit["cov"][jb, jb]))
        fit.update(jb=jb)
    elif spec == "balanced":
        X, jb = _Xmat(d)
        f0 = reml(y, X, se, grp, weighted=True)
        tau2 = f0["tau_d2"] + f0["tau_w2"]
        k = d.groupby("design")["y"].transform("size").values
        vbar = d.groupby("design")["se"].transform(lambda s: np.mean(s ** 2)).values
        w = 1.0 / (k * (vbar + tau2))
        fit = wls(y, X, w, grp)
        fit.update(tau_d2=f0["tau_d2"], tau_w2=f0["tau_w2"], jb=jb)
        cvec = np.zeros(X.shape[1]); cvec[jb] = 1.0
        se_mb = float(np.sqrt(fit["cov"][jb, jb] * max(fit["s2"], 1.0)))
    elif spec in ("fe_weighted", "fe_unweighted"):
        X, jb = _Xmat(d, fe=True)
        w = 1.0 / se ** 2 if spec == "fe_weighted" else np.ones(len(y))
        fit = wls(y, X, w, grp)
        fit.update(tau_d2=np.nan, tau_w2=np.nan, jb=jb)
        cvec = np.zeros(X.shape[1]); cvec[jb] = 1.0
        se_mb = float(np.sqrt(fit["cov_scaled"][jb, jb] if spec == "fe_weighted" else fit["cov"][jb, jb] * fit["s2"]))
    else:
        raise ValueError(spec)
    fit["spec"] = spec
    fit["cvec_slope"] = cvec
    fit["se_slope_model"] = se_mb
    return fit


def predict(fit, d, log10C, clusters=None):
    """Population-average prediction mu + beta (log10C - 20) (RE specs) with model-based and CR2 intervals."""
    X = fit["X"]
    out = []
    for c in np.atleast_1d(log10C):
        cvec = np.zeros(X.shape[1])
        cvec[0], cvec[fit["jb"]] = 1.0, c - C0
        est = float(cvec @ fit["beta"])
        se_m = float(np.sqrt(cvec @ fit["cov"] @ cvec)) * (np.sqrt(max(fit.get("s2", 1.0), 1.0)) if "s2" in fit else 1.0)
        se_r, df, _ = cr2(fit, cvec, clusters)
        q = tdist.ppf(0.975, max(df, 1.0))
        out.append(dict(log10C=c, est=est, se_model=se_m, lo_model=est - 1.96 * se_m, hi_model=est + 1.96 * se_m,
                        se_cr2=se_r, df_cr2=df, lo_cr2=est - q * se_r, hi_cr2=est + q * se_r))
    return pd.DataFrame(out)


def blup_lines(fit, d):
    """Design-specific lines mu + theta_j + beta (c - 20) with theta_j the BLUP (RE specs)."""
    V, Vi, X, y = fit["V"], fit["Vi"], fit["X"], fit["y"]
    r = y - X @ fit["beta"]
    out = {}
    for j, ix in enumerate(fit["blocks"]):
        des = d["design"].values[ix[0]]
        th = fit["tau_d2"] * np.sum((Vi @ r)[ix])
        out[des] = float(th)
    return out


def slope_row(fit, d, label, clusters_study=None, boot=True, B=9999, seed=0):
    y = d["y"].values
    se_r, df, b = cr2(fit, fit["cvec_slope"])
    row = dict(spec=label, k=len(d), n_designs=d["design"].nunique(), slope=b, se_model=fit["se_slope_model"],
               se_cr2_design=se_r, df_cr2_design=df,
               lo_cr2_design=b - tdist.ppf(0.975, max(df, 1)) * se_r, hi_cr2_design=b + tdist.ppf(0.975, max(df, 1)) * se_r,
               p_cr2_design=float(2 * tdist.sf(abs(b / se_r), max(df, 1))),
               tau_design=float(np.sqrt(max(fit.get("tau_d2", np.nan), 0))) if np.isfinite(fit.get("tau_d2", np.nan)) else np.nan,
               tau_within=float(np.sqrt(max(fit.get("tau_w2", np.nan), 0))) if np.isfinite(fit.get("tau_w2", np.nan)) else np.nan)
    if clusters_study is not None and pd.Series(clusters_study).nunique() >= 3:
        se_s, df_s, _ = cr2(fit, fit["cvec_slope"], clusters_study)
        row.update(se_cr2_study=se_s, df_cr2_study=df_s, p_cr2_study=float(2 * tdist.sf(abs(b / se_s), max(df_s, 1))))
    if boot:
        row["p_wcr_design"] = wcr_design(fit, d, B=B, seed=seed)
    return row


def wcr_design(fit, d, B=9999, seed=0):
    """Restricted wild cluster bootstrap-t by design (Webb six-point weights) for H0: slope = 0, holding the working
    covariance of the fit fixed; t uses the CR2 s.e. (fast version: CR2 matrices depend only on X and V)."""
    rng = np.random.default_rng(seed)
    X, V, y = fit["X"], fit["V"], fit["y"]
    Vi = np.linalg.inv(V)
    jb = fit["jb"]
    Xr = np.delete(X, jb, axis=1)
    br = np.linalg.solve(Xr.T @ Vi @ Xr, Xr.T @ Vi @ y)
    yhat_r = Xr @ br
    er = y - yhat_r
    # precompute CR2 linear forms: slope_hat = a'y ; CR2 variance = sum_j (g_j' e_j)^2 with e = (I-H) y
    n = len(y)
    Bm = np.linalg.inv(X.T @ Vi @ X)
    a = (Bm @ X.T @ Vi)[jb]
    IH = np.eye(n) - X @ Bm @ X.T @ Vi
    G_list = []
    for ix in fit["blocks"]:
        Phij = V[np.ix_(ix, ix)]
        Mj = (IH @ V @ IH.T)[np.ix_(ix, ix)]
        Ph, Pih = _msqrt(Phij), _msqrt(Phij, inv=True)
        Aj = Ph @ _msqrt(Pih @ Mj @ Pih, inv=True) @ Pih
        gj = Aj.T @ Vi[np.ix_(ix, ix)] @ X[ix] @ Bm[:, jb]
        row = np.zeros(n)
        row[ix] = gj
        G_list.append(row @ IH)
    Gm = np.array(G_list)
    t0 = float(a @ y) / np.sqrt(np.sum((Gm @ y) ** 2))
    g = fit["groups"]
    Gn = g.max() + 1
    W6 = cm.rc.WEBB
    V_ = rng.choice(W6, size=(B, Gn))
    Ys = yhat_r[None, :] + V_[:, g] * er[None, :]
    num = Ys @ a
    den = np.sqrt(np.sum((Ys @ Gm.T) ** 2, axis=1))
    ts = num / den
    return float((1 + np.sum(np.abs(ts) >= abs(t0))) / (B + 1))


def run_meta(log=cm.log, B=9999):
    d_all = load_budgets()
    d = d_all[~d_all.porian].reset_index(drop=True)
    rows, preds, lines = [], [], []
    specs = [("weighted", "3-level RE, inverse-variance (primary)"), ("unweighted", "Design RE, unweighted (LMM)"),
             ("balanced", "Design-balanced weights (CE-RVE)"), ("fe_weighted", "Design FE, inverse-variance"),
             ("fe_unweighted", "Design FE, unweighted (OLS)")]
    fits = {}
    for sp, lab in specs:
        f = fit_spec(d, sp)
        fits[sp] = f
        r = slope_row(f, d, lab, clusters_study=d["study"].values, B=B, seed=cm.seed_of(f"wcr|{sp}"))
        r.update(sample="all six designs (excl. Porian)", spec_key=sp)
        rows.append(r)
        log(f"  meta-regression [{lab}]: slope {r['slope']:+.4f} (model {r['se_model']:.4f}; CR2 {r['se_cr2_design']:.4f}, "
            f"df {r['df_cr2_design']:.1f}); tau_d {r['tau_design']:.3f} tau_w {r['tau_within']:.3f}; p_wcr {r['p_wcr_design']:.4f}")
        if sp in ("weighted", "unweighted", "balanced"):
            pr = predict(f, d, [18.5, 19.0, 20.0, 21.0, 21.5, 22.0, 23.0, 24.0])   # 18.5: figure band (review fix)
            pr["spec"], pr["spec_key"] = lab, sp
            preds.append(pr)
    # robustness of the functional form of the drift: (i) linear in S = 2(1/sigma* - 1) (the wedge scale is S/2);
    # (ii) hinge at log10 C = 20.5 (flat below Marin's largest budget, linear above); primary 3-level RE weights
    dS = d.copy()
    dS["y"] = 2 * (1 / d["y"] - 1)
    dS["se"] = 2 * d["se"] / d["y"] ** 2
    fS = fit_spec(dS, "weighted")
    rS = slope_row(fS, dS, "3-level RE on S = 2(1/sigma*-1) (slope in S units)", clusters_study=dS["study"].values,
                   B=B, seed=cm.seed_of("wcr|S"))
    rS.update(sample="all six designs (excl. Porian)", spec_key="weighted_S")
    rows.append(rS)
    pS = predict(fS, dS, [19.0, 20.0, 21.0, 21.5, 22.0, 23.0, 24.0])
    for col in ("est", "lo_model", "hi_model", "lo_cr2", "hi_cr2"):
        pS[col] = 2 / (2 + pS[col])       # S -> sigma* (decreasing map: bounds swap)
    pS = pS.rename(columns={"lo_model": "hi_model", "hi_model": "lo_model", "lo_cr2": "hi_cr2", "hi_cr2": "lo_cr2"})
    pS[["se_model", "se_cr2"]] = np.nan     # review fix: these were s.e. in S units beside sigma*-scale estimates
    pS["spec"], pS["spec_key"] = "3-level RE on S, mapped to sigma*", "weighted_S"
    preds.append(pS)
    dh = d.copy()
    dh["log10C"] = C0 + np.maximum(d["log10C"] - 20.5, 0.0)     # hinge regressor (log10C - 20.5)_+ centred at 20
    fh = fit_spec(dh, "weighted")
    rh = slope_row(fh, dh, "3-level RE, hinge at 3e20 (slope above the knot)", clusters_study=dh["study"].values,
                   B=B, seed=cm.seed_of("wcr|hinge"))
    rh.update(sample="all six designs (excl. Porian)", spec_key="weighted_hinge")
    rows.append(rh)
    ph = predict(fh, dh, [20.0, 20.5, 21.0, 21.5])
    ph["log10C"] = [20.5, 21.0, 21.5, 22.0]   # hinge coordinate 20 + (c - 20.5)_+ -> actual log10 C (first: at and below 20.5)
    ph["spec"], ph["spec_key"] = "3-level RE, hinge at 3e20 (value flat at and below 3e20)", "weighted_hinge"
    preds.append(ph)
    log(f"  drift on the S scale: {rS['slope']:+.4f} per decade (CR2 {rS['se_cr2_design']:.4f}); hinge slope above 3e20: "
        f"{rh['slope']:+.4f} (CR2 {rh['se_cr2_design']:.4f}, p_wcr {rh['p_wcr_design']:.3f})")
    # subsamples (primary weighted and unweighted)
    subs = [("Chinchilla + Llama 3 (the two designs reaching 1e21)", d.design.isin(["Chinchilla", "Llama 3"])),
            ("Marin only (three corpora)", d.design.str.startswith("Marin")),
            ("IsoFLOP designs only (excl. Farseer path)", d.design != "Farseer (local path)"),
            ("Budgets <= 3e20 (common window)", d.log10C <= np.log10(3e20) + 1e-9)]
    for des in DESIGNS:
        subs.append((f"leave out {des}", d.design != des))
    top_idx = d.groupby("design")["log10C"].idxmax()
    subs.append(("leave out each design's largest budget", ~d.index.isin(top_idx.values)))
    for lab_s, m in subs:
        ds = d[m].reset_index(drop=True)
        for sp in ("weighted", "unweighted", "fe_weighted"):
            if ds.design.nunique() < 2 and sp != "fe_weighted":
                continue
            try:
                f = fit_spec(ds, sp)
                r = slope_row(f, ds, dict(specs)[sp], clusters_study=ds["study"].values,
                              B=max(999, B // 5), seed=cm.seed_of(f"wcr|{lab_s}|{sp}"))
                r.update(sample=lab_s, spec_key=sp)
                rows.append(r)
            except Exception as e:  # recorded
                rows.append(dict(sample=lab_s, spec=dict(specs)[sp], spec_key=sp, error=repr(e)))
    # Porian separately (unannealed; not pooled)
    dp = d_all[d_all.porian].reset_index(drop=True)
    for sp in ("weighted", "fe_weighted"):
        f = fit_spec(dp, sp)
        r = slope_row(f, dp, dict(specs)[sp], clusters_study=None, boot=False)
        r.update(sample="Porian et al. (unannealed), reported separately", spec_key=sp)
        rows.append(r)
    # design-specific slopes (FE, inverse variance): heterogeneity of drift across designs.
    # Review fix: the covariance is scaled by the residual variance when above one (s2 = 2.5 on all budgets), the
    # same convention as every other FE-IV row (R3's computation); the builder's unscaled Wald (27.7, p < 0.001) and
    # s.e. overstated the precision. Also run on the common window (budgets <= 3e20), where the pooled slope is ~0
    # but Llama 3's own point estimate is not.
    hets = []
    for lab_h, mh in (("all budgets", np.ones(len(d), bool)),
                      ("budgets <= 3e20 (common window)", (d.log10C <= np.log10(3e20) + 1e-9).values)):
        dh_ = d[mh].reset_index(drop=True)
        Xs, _ = _Xmat(dh_, fe=True, slope_by_design=True)
        ws = wls(dh_["y"].values, Xs, 1.0 / dh_["se"].values ** 2, dh_["design"].values)
        names = list(pd.get_dummies(dh_["design"]).columns)
        J = len(names)
        bs = ws["beta"][J:]
        sc = max(ws["s2"], 1.0)
        Cv = ws["cov"][J:, J:] * sc
        Rm = np.hstack([np.ones((J - 1, 1)), -np.eye(J - 1)])
        dif = Rm @ bs
        Wd = float(dif @ np.linalg.solve(Rm @ Cv @ Rm.T, dif))
        Wu = Wd * sc
        h_ = pd.DataFrame(dict(sample=lab_h, design=names, slope=bs, se=np.sqrt(np.diag(Cv)),
                               se_unscaled=np.sqrt(np.diag(ws["cov"][J:, J:])),
                               n_budgets=[int((dh_.design == nm).sum()) for nm in names]))
        h_["t"] = h_["slope"] / h_["se"]
        h_["resid_var_scale"] = ws["s2"]
        h_["wald_equal_slopes"], h_["df"], h_["p_equal_slopes"] = Wd, J - 1, float(chi2.sf(Wd, J - 1))
        h_["wald_unscaled"], h_["p_unscaled"] = Wu, float(chi2.sf(Wu, J - 1))
        hets.append(h_)
        log(f"  design-specific drift [{lab_h}] (FE, IV weights, s.e. scaled by s2 = {ws['s2']:.2f}): Wald test of "
            f"equal slopes {Wd:.2f} on {J-1} df, p = {chi2.sf(Wd, J-1):.4f} (unscaled {Wu:.1f})")
    het = pd.concat(hets, ignore_index=True)
    # design lines (primary weighted fit): BLUP intercepts and fitted values at each design's top budget
    f = fits["weighted"]
    bl = blup_lines(f, d)
    tops = []
    for des in DESIGNS:
        g = d[d.design == des]
        top = g.loc[g.log10C.idxmax()]
        mu, beta = f["beta"]
        tops.append(dict(design=des, study=STUDY[des], top_budget=top.budget_C, log10C_top=top.log10C,
                         sigma_raw_top=top.y, se_raw_top=top.se,
                         fitted_top_blup=mu + bl[des] + beta * (top.log10C - C0),
                         fitted_top_population=mu + beta * (top.log10C - C0),
                         design_intercept_1e20=mu + bl[des], n_budgets=len(g),
                         Cmin=g.budget_C.min(), Cmax=g.budget_C.max()))
    tops = pd.DataFrame(tops)
    return dict(data=d_all, slopes=pd.DataFrame(rows), preds=pd.concat(preds, ignore_index=True), het=het,
                tops=tops, fits=fits, blup=bl)


# ============================================================================ top-budget sigma*
def top_budget_values(d_all):
    """sigma*_top: inverse-variance and random-effects means of the budget-level estimates at the largest budgets.
    (i) Chinchilla and Llama 3 at budgets >= 6e20 (the two designs whose bracketed budgets reach 1e21);
    (ii) same plus Farseer's local path at 1e21."""
    from meta import meta as dl
    out = []
    for lab, m in (("Chinchilla + Llama 3, budgets >= 6e20", d_all.design.isin(["Chinchilla", "Llama 3"]) & (d_all.budget_C >= 6e20)),
                   ("Chinchilla + Llama 3 + Farseer path, budgets >= 6e20",
                    d_all.design.isin(["Chinchilla", "Llama 3", "Farseer (local path)"]) & (d_all.budget_C >= 6e20)),
                   ("Chinchilla + Llama 3, largest bracketed budget of each",
                    ((d_all.design == "Chinchilla") & (d_all.budget_C == 3e21)) | ((d_all.design == "Llama 3") & (d_all.budget_C == 1e21)))):
        g = d_all[m]
        r = dl(g.y.values, g.se.values)
        out.append(dict(set=lab, k=len(g), members="; ".join(f"{a} {b:.0e}: {c:.3f}" for a, b, c in zip(g.design, g.budget_C, g.y)),
                        mean_fixed=r["mu_fixed"], se_fixed=r["se_fixed"], mean_re=r["mu_re"], lo_hksj=r["lo_hksj"],
                        hi_hksj=r["hi_hksj"], tau=r["tau"], Q=r["Q"], p_Q=r["p_Q"], plain_mean=float(g.y.mean())))
    return pd.DataFrame(out)


# ============================================================================ task 2: study-level headline interval
def study_level(log=cm.log):
    """One estimate per study (Hoffmann, Meta, Marin, Farseer; k = 4) with the DerSimonian-Laird mean and the
    modified Hartung-Knapp-Sidik-Jonkman interval (t_{k-1}); variants for Marin's aggregation, Farseer's estimator
    and bandwidth, FLOP accounting and parameter conventions."""
    from meta import meta as dl
    s = pd.read_csv(cm.RA1_SUMMARY).set_index("design")
    sens = pd.read_csv(cm.RA1_SENS)
    slope = pd.read_csv(cm.RA1_FAR_SLOPE)
    pool = pd.read_csv(cm.RA1_FAR_POOL).set_index("conv")
    fsens = pd.read_csv(cm.RA1_FAR_SENS)
    chin = (s.loc["Chinchilla", "sigma_re"], s.loc["Chinchilla", "se_sigma_re"])
    meta_ = (s.loc["Llama 3", "sigma_re"], s.loc["Llama 3", "se_sigma_re"])
    mar = s.loc[["Marin, Comma", "Marin, DCLM", "Marin, Nemotron-CC"], ["sigma_re", "se_sigma_re"]].values
    cfg = sens[sens.design.str.contains(r"\[config N\]")].set_index("design")
    mar_cfg = np.array([[cfg.loc[f"{m} [config N]", "sigma_re"], cfg.loc[f"{m} [config N]", "se_sigma_re"]]
                        for m in ("Marin, Comma", "Marin, DCLM", "Marin, Nemotron-CC")])

    def marin_one(vals, rho):
        """Marin as one study: equal-weight mean of the three corpora; s.e. under within-lab correlation rho."""
        y, se = vals[:, 0], vals[:, 1]
        m = float(np.mean(y))
        C = rho * np.outer(se, se) + (1 - rho) * np.diag(se ** 2)
        return m, float(np.sqrt(np.ones(3) @ C @ np.ones(3)) / 3)

    fd = slope[(slope.conv == "ne") & slope.variant.str.startswith("primary (wild")].iloc[0]
    fd_emb = slope[(slope.conv == "emb") & slope.variant.str.startswith("primary (wild")].iloc[0]
    far_path_mean = float(fsens[(fsens.variant.str.startswith("primary")) & (fsens.key == "path_sigma_mean_poolC|ne")].value.iloc[0])
    far_path_cv = float(fsens[(fsens.variant == "extended-grid CV optimum") & (fsens.key == "path_sigma_mean_poolC|ne")].value.iloc[0])
    far_path_se_mean = float(pool.loc["ne", "se"])        # bootstrap s.e. of the plain mean (overlapping kernels)
    # Farseer: first-derivative path estimate 1/(1+b1); s.e. = max(its wild cluster s.e., the plain-mean s.e.)
    far_fd = (float(fd.sigma_slope), max(float(fd.se), far_path_se_mean))
    far_fd_emb = (float(fd_emb.sigma_slope), max(float(fd_emb.se), far_path_se_mean))
    far_re = (float(pool.loc["ne", "sigma_re"]), float(pool.loc["ne", "se_re"]))
    far_hess_cv = (far_path_cv, far_path_se_mean)
    eta = 0.1

    def shift_eta(sig, e):
        # sigma*_eff = 2/(2 + S/(1+eta)^2): FLOP-accounting elasticity eta of the measured N (ra1 conventions)
        S = 2 * (1 / sig - 1)
        return 2 / (2 + S / (1 + e) ** 2)

    variants = []

    def add(label, entries, note=""):
        names = [e[0] for e in entries]
        y = np.array([e[1] for e in entries])
        se = np.array([e[2] for e in entries])
        r = dl(y, se)
        variants.append(dict(variant=label, k=len(y), members="; ".join(f"{n} {a:.3f} ({b:.3f})" for n, a, b in entries),
                             mean_re=r["mu_re"], lo_hksj=r["lo_hksj"], hi_hksj=r["hi_hksj"], se_hksj=r["se_hksj"],
                             tau=r["tau"], Q=r["Q"], p_Q=r["p_Q"], I2=r["I2"], mean_fixed=r["mu_fixed"],
                             pi_lo=r["pi_lo"], pi_hi=r["pi_hi"], note=note))

    m1 = marin_one(mar, 1.0)
    m0 = marin_one(mar, 0.0)
    m5 = marin_one(mar, 0.5)
    base = [("Hoffmann", *chin), ("Meta", *meta_), ("Marin", *m1), ("Farseer", *far_fd)]
    add("PRIMARY: one estimate per study; Marin = mean of 3 corpora (rho = 1); Farseer first-derivative path", base,
        ("FLOP-implied N (Llama 3, Marin), FLOP-effective N_F = F_T4/6 rebuilt from the architecture table (Chinchilla; "
         "rb4_chinflop), non-embedding N (Farseer)") if cm.CHIN_REBUILT else
        "FLOP-implied N (Llama 3, Marin), total N (Chinchilla), non-embedding N (Farseer)")
    add("Marin corpora independent (rho = 0)", [base[0], base[1], ("Marin", *m0), base[3]])
    add("Marin corpora rho = 0.5", [base[0], base[1], ("Marin", *m5), base[3]])
    add("Marin by inverse variance (ra1's 'one per study'); Farseer RE path 0.703",
        [base[0], base[1], ("Marin", *_iv(mar)), ("Farseer", *far_re)])
    add("Farseer Hessian-based path, RE over levels (ra1 Table 1 input)", [base[0], base[1], base[2], ("Farseer", *far_re)])
    add("Farseer Hessian-based path at the extended-grid CV bandwidth", [base[0], base[1], base[2], ("Farseer", *far_hess_cv)],
        "bandwidth sensitivity (R4 R2-M4)")
    add("IsoFLOP studies only (k = 3)", base[:3])
    add("Six designs (ra1 Panel C; Marin x3 separate)", [base[0], base[1]] + [(f"Marin {i}", *mar[i]) for i in range(3)] +
        [("Farseer", *far_re)], "as in the paper's Table 1, Panel C (not one per study)")
    # FLOP accounting and conventions
    mc1 = marin_one(mar_cfg, 1.0)
    add("Marin in configuration N (0.64-0.66)", [base[0], base[1], ("Marin cfg", *mc1), base[3]], "FLOP accounting (R4)")
    add("Marin config N + Farseer at the CV bandwidth (R4's 'both', one per study)",
        [base[0], base[1], ("Marin cfg", *mc1), ("Farseer", *far_hess_cv)])
    add("Farseer incl. embeddings (total-N convention)", [base[0], base[1], base[2], ("Farseer emb", *far_fd_emb)],
        "convention")
    add("Lowest-convention set: Marin config N, Farseer incl. embeddings",
        [base[0], base[1], ("Marin cfg", *mc1), ("Farseer emb", *far_fd_emb)], "convention range, lower end")
    for e in (-eta, eta):
        if cm.CHIN_REBUILT:
            # [rb4 integration] Chinchilla's accounting is observed (rb4_chinflop): the unobserved elasticity is Meta's
            # only. Equals rb4_chinflop_study_level.csv, run = new, '[rb4] ... for Meta only (Chinchilla rebuilt)'.
            add(f"FLOP-accounting elasticity eta = {e:+.1f} for Meta only (unobserved; Chinchilla's accounting rebuilt, rb4)",
                [base[0], ("Meta", shift_eta(meta_[0], e), meta_[1]), base[2], base[3]],
                "sigma*_eff = 2/(2 + S/(1+eta)^2) applied to Llama 3 only")
        else:
            add(f"FLOP-accounting elasticity eta = {e:+.1f} for Chinchilla and Meta (unobserved)",
                [("Hoffmann", shift_eta(chin[0], e), chin[1]), ("Meta", shift_eta(meta_[0], e), meta_[1]), base[2], base[3]],
                "sigma*_eff = 2/(2 + S/(1+eta)^2)")
    chin_alt = _chinchilla_alternatives() if cm.CHIN_REBUILT else []
    for key, lab, y_, se_, kind in chin_alt:
        # [rb4 integration] Chinchilla under rb4's other FLOP counts and conventions (kind 'accounting') and under its
        # window/membership sensitivities (kind 'windows'; rb4 review R11-R13); the other studies at their primary values
        add(f"[rb4] Chinchilla {kind} variant {key}: {lab}", [("Hoffmann " + key, y_, se_), base[1], base[2], base[3]],
            "other studies at their primary values")
    # review addition (R2 Major 1 request 1): design-level FIXED-effect (inverse-variance) means instead of the
    # random-effects means for the two designs with significant between-budget heterogeneity (drift). Llama 3's FE
    # mean (0.628) is dominated by its precise budgets at 3e20-1e21.
    chin_fe = (s.loc["Chinchilla", "sigma_fe"], s.loc["Chinchilla", "se_sigma_fe"])
    meta_fe = (s.loc["Llama 3", "sigma_fe"], s.loc["Llama 3", "se_sigma_fe"])
    add("Design-level fixed-effect means for Chinchilla and Llama 3 (R2 Major 1.1)",
        [("Hoffmann FE", *chin_fe), ("Meta FE", *meta_fe), base[2], base[3]],
        "fixed-effect (inverse-variance) pooling of budgets within Chinchilla and Llama 3 (ra1 sigma_fe)")
    V = pd.DataFrame(variants)
    if cm.CHIN_REBUILT:     # [rb4 integration] Chinchilla's accounting observed: primary N_F plus rb4's alternatives
        chin_conv = ([dict(study="Hoffmann et al. (2022)", design="Chinchilla",
                           convention="FLOP-effective N_F = F_T4/6, rebuilt from the architecture table (rb4; primary)",
                           sigma=chin[0], se=chin[1])] +
                     [dict(study="Hoffmann et al. (2022)", design="Chinchilla", convention=f"[rb4] {key}: {lab}",
                           sigma=y_, se=se_) for key, lab, y_, se_, kind in chin_alt])
    else:
        chin_conv = [
            dict(study="Hoffmann et al. (2022)", design="Chinchilla", convention="total N (runs at nominal budget)", sigma=chin[0], se=chin[1]),
            dict(study="Hoffmann et al. (2022)", design="Chinchilla", convention=f"FLOP-accounting eta = -{eta}", sigma=shift_eta(chin[0], -eta), se=chin[1]),
            dict(study="Hoffmann et al. (2022)", design="Chinchilla", convention=f"FLOP-accounting eta = +{eta}", sigma=shift_eta(chin[0], eta), se=chin[1])]
    conv = pd.DataFrame(chin_conv + [
        dict(study="Grattafiori et al. (2024)", design="Llama 3", convention="FLOP-implied N_F = C/(6D)", sigma=meta_[0], se=meta_[1]),
        dict(study="Grattafiori et al. (2024)", design="Llama 3", convention=f"FLOP-accounting eta = -{eta}", sigma=shift_eta(meta_[0], -eta), se=meta_[1]),
        dict(study="Grattafiori et al. (2024)", design="Llama 3", convention=f"FLOP-accounting eta = +{eta}", sigma=shift_eta(meta_[0], eta), se=meta_[1]),
    ] + [dict(study="Marin (2026)", design=m, convention="FLOP-implied N_F = C/(6D)", sigma=v[0], se=v[1])
         for m, v in zip(("Comma", "DCLM", "Nemotron-CC"), mar)]
      + [dict(study="Marin (2026)", design=m, convention="configuration N (eta = 0.087)", sigma=v[0], se=v[1])
         for m, v in zip(("Comma", "DCLM", "Nemotron-CC"), mar_cfg)]
      + [dict(study="Li et al. (2025)", design="Farseer", convention="non-embedding N, first-derivative path", sigma=far_fd[0], se=far_fd[1]),
         dict(study="Li et al. (2025)", design="Farseer", convention="incl. embeddings (total N), first-derivative path", sigma=far_fd_emb[0], se=far_fd_emb[1]),
         dict(study="Li et al. (2025)", design="Farseer", convention="non-embedding N, Hessian-based path mean, primary bandwidth", sigma=far_path_mean, se=far_path_se_mean),
         dict(study="Li et al. (2025)", design="Farseer", convention="non-embedding N, Hessian-based path mean, extended-grid CV bandwidth", sigma=far_path_cv, se=far_path_se_mean)])
    log(f"  study-level (k=4) primary: {V.iloc[0].mean_re:.3f} [{V.iloc[0].lo_hksj:.3f}, {V.iloc[0].hi_hksj:.3f}], "
        f"tau {V.iloc[0].tau:.3f}; range of variant means {V.mean_re.min():.3f}-{V.mean_re.max():.3f}")
    return V, conv


def _chinchilla_alternatives():
    """[rb4 integration] Chinchilla's alternative estimates from module rb4_chinflop, for the study-level variants when
    RB1_CHINCHILLA = 'rb4'. Returns (key, label, sigma_re, se_sigma_re, kind) tuples.
    kind 'accounting': Hoffmann et al.'s other FLOP counts and the other parameter conventions (nominal budgets or each
      count's own exact isocosts; T4 under 6T tokens). Full-precision values from rb4's override summaries
      (data/processed/rb4_chinflop/override_new_chin_<key>_isoflop_summary.csv), the inputs of rb4's own rows, so that
      these rows equal rb4_chinflop_study_level.csv (run = new) exactly; labels from rb4_chinflop_summary.csv.
    kind 'windows': the window/membership sensitivities of rb4's review (rb4_chinflop_review_checks.csv): R11 count-free
      N_F read off the digitized coordinates, R12 profile membership drawn on the T4-corrected coordinates (147 runs),
      R13 the five high-loss runs dropped (132 runs). sigma to 6 significant digits, s.e. to 4 decimals (from the note)."""
    import re
    lab = pd.read_csv(cm.RB4_SUMMARY).set_index("variant")["label"]
    out = []
    for key in ("NF_A|corr", "NF_X|corr", "NF_A", "NF_X", "P", "T", "P|corr", "T|corr", "NF_T4|6ND"):
        f = os.path.join(cm.RB4_PROC, f"override_new_chin_{key.replace('|', '_')}_isoflop_summary.csv")
        r = pd.read_csv(f).set_index("design").loc["Chinchilla"]
        out.append((key, str(lab.get(key, key)), float(r.sigma_re), float(r.se_sigma_re), "accounting"))
    ch = pd.read_csv(cm.RB4_CHECKS)
    for key, pre, stat in (("R11", "R11 count-free", "RE sigma*"), ("R12", "R12 T4-consistent", "RE sigma*, N_F"),
                           ("R13", "R13 five highest-loss", "RE sigma*, N_F")):
        r = ch[ch.check.str.startswith(pre) & ch.statistic.str.startswith(stat)].iloc[0]
        se_ = float(re.search(r"RE [0-9.]+ \(([0-9.]+)\)", r.note).group(1))
        out.append((key, f"{r.check} ({r.statistic})", float(r.value), se_, "windows"))
    return out


def add_local_fd_variant(V, X):
    """Review addition: the study-level interval when the IsoFLOP designs that have a local-wedge surface (Marin x3,
    Llama 3; extrap stage) use the same local first-derivative estimator 1/(1 + b1) as Farseer (R4 R2-M4 item 3,
    'one estimator throughout'). Standard errors: ra1's design-level s.e. (the extrap bootstrap s.e. of 1/(1+b1) is
    design-conditional and omits smoothing bias, so it is too small to use here). Chinchilla keeps ra1's estimate
    (no local-wedge surface was fitted to it). Returns V with the row appended (idempotent)."""
    from meta import meta as dl
    lab = "Local first-derivative estimator 1/(1+b1) for Marin and Llama 3 as for Farseer (review)"
    V = V[V.variant != lab].copy()
    s = pd.read_csv(cm.RA1_SUMMARY).set_index("design")

    def fd(name):
        L = X[name]["L"].set_index("param")
        return float(L.loc["sigma_slope", "est"])
    mar = np.array([[fd(m), s.loc[m, "se_sigma_re"]] for m in ("Marin, Comma", "Marin, DCLM", "Marin, Nemotron-CC")])
    m_y, m_se = float(mar[:, 0].mean()), float(mar[:, 1].mean())          # rho = 1, as in the headline
    slope = pd.read_csv(cm.RA1_FAR_SLOPE)
    pool = pd.read_csv(cm.RA1_FAR_POOL).set_index("conv")
    fd_ = slope[(slope.conv == "ne") & slope.variant.str.startswith("primary (wild")].iloc[0]
    f_y, f_se = float(fd_.sigma_slope), max(float(fd_.se), float(pool.loc["ne", "se"]))   # as in study_level
    ent = [("Hoffmann", s.loc["Chinchilla", "sigma_re"], s.loc["Chinchilla", "se_sigma_re"]),
           ("Meta local-FD", fd("Llama 3"), s.loc["Llama 3", "se_sigma_re"]),
           ("Marin local-FD", m_y, m_se), ("Farseer", f_y, f_se)]
    y = np.array([e[1] for e in ent]); se = np.array([e[2] for e in ent])
    r = dl(y, se)
    row = dict(variant=lab, k=4, members="; ".join(f"{n} {a:.3f} ({b:.3f})" for n, a, b in ent), mean_re=r["mu_re"],
               lo_hksj=r["lo_hksj"], hi_hksj=r["hi_hksj"], se_hksj=r["se_hksj"], tau=r["tau"], Q=r["Q"], p_Q=r["p_Q"],
               I2=r["I2"], mean_fixed=r["mu_fixed"], pi_lo=r["pi_lo"], pi_hi=r["pi_hi"],
               note="estimator sensitivity: local-wedge slope at the path (extrap stage) on Marin and Llama 3")
    return pd.concat([V, pd.DataFrame([row])], ignore_index=True)


def _iv(vals):
    y, se = vals[:, 0], vals[:, 1]
    w = 1 / se ** 2
    return float(np.sum(w * y) / np.sum(w)), float(np.sqrt(1 / np.sum(w)))
