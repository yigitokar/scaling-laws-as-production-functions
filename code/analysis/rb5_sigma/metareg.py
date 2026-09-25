"""metareg.py -- the sigma*(C) meta-regression with Marin as one cluster (T1.2; R1 New 2(d), R4 N3(d) and request 2).

rb1's meta-regression machinery (rb1_sigmaC/meta_c.py: REML fits, CR2 with Bell-McCaffrey degrees of freedom, the
restricted wild cluster bootstrap-t) is imported unchanged; the inputs are rb1's primary budget-level estimates
(Chinchilla in N_F; 44 budgets of six designs, Porian excluded). Additions here:
  * clusters by STUDY (Hoffmann, Meta, Marin, Farseer: four clusters; Marin's three corpora share code, ladder and FLOP
    accounting), for the CR2 test and for the restricted wild cluster bootstrap-t (Webb six-point weights);
  * every wild bootstrap p-value also by full enumeration of the 6^G weight vectors (G = 4: 1,296; G = 6: 46,656),
    which removes Monte Carlo error; the random-draw version with rb1's seeds reproduces rb1's published p-values;
  * working models: design random intercepts (rb1) and study random intercepts (Marin's corpora share one intercept);
  * design fixed-effect rows (six designs, inverse-variance and unweighted; R4 N3(d)) and a study fixed-effect row;
  * three cluster structures: 4 (Marin one cluster), 5 (IsoFLOP designs only, Marin's corpora separate: R3's round-2
    sample) and 6 (designs, rb1 as published); the p-value range across all specifications, weights, scales, tests
    and fixed-effect rows.
No p-value from four to six clusters is reliable (Bell and McCaffrey 2002; MacKinnon and Webb 2017): the rows are
reported to show how much the evidence depends on the choice, not to certify significance.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

import sigcommon as cm

B_WILD = 999 if cm.QUICK else 9999


def load():
    import meta_c as mc
    d_all = mc.load_budgets()                         # rb1's loader: rb4 override inputs (Chinchilla in N_F)
    d = d_all[~d_all.porian].reset_index(drop=True)
    d["study4"] = d["design"].map(cm.STUDY)
    return d, d_all


def _wild_forms(fit, clusters):
    """Linear forms of rb1's wcr_design generalized to arbitrary clusters: slope = a'y, CR2 variance = sum_j (G_j y)^2,
    the restricted fit (slope = 0) and its residuals."""
    import meta_c as mc
    X, V, y = fit["X"], fit["V"], fit["y"]
    n = len(y)
    jb = fit["jb"]
    cl = pd.factorize(clusters)[0]
    blocks = [np.where(cl == j)[0] for j in range(cl.max() + 1)]
    Phi = np.zeros_like(V)
    for ix in blocks:
        Phi[np.ix_(ix, ix)] = V[np.ix_(ix, ix)]
    Vi = np.zeros_like(V)
    for ix in blocks:
        Vi[np.ix_(ix, ix)] = np.linalg.inv(Phi[np.ix_(ix, ix)])
    Xr = np.delete(X, jb, axis=1)
    br = np.linalg.solve(Xr.T @ Vi @ Xr, Xr.T @ Vi @ y)
    yhat_r = Xr @ br
    er = y - yhat_r
    Bm = np.linalg.inv(X.T @ Vi @ X)
    a = (Bm @ X.T @ Vi)[jb]
    IH = np.eye(n) - X @ Bm @ X.T @ Vi
    M = IH @ Phi @ IH.T
    G = []
    for ix in blocks:
        Phij = Phi[np.ix_(ix, ix)]
        Mj = M[np.ix_(ix, ix)]
        Ph, Pih = mc._msqrt(Phij), mc._msqrt(Phij, inv=True)
        Aj = Ph @ mc._msqrt(Pih @ Mj @ Pih, inv=True) @ Pih
        gj = Aj.T @ Vi[np.ix_(ix, ix)] @ X[ix] @ Bm[:, jb]
        row = np.zeros(n)
        row[ix] = gj
        G.append(row @ IH)
    return dict(a=a, G=np.array(G), yhat_r=yhat_r, er=er, cl=cl, n_cl=len(blocks), y=y)


def wild_p(fit, clusters, B=B_WILD, seed=0, enumerate_all=False):
    """Restricted wild cluster bootstrap-t p-value for H0: slope = 0 (Webb weights), CR2 t. With the fit's own design
    clusters and rb1's seed this reproduces meta_c.wcr_design exactly (same random numbers)."""
    f = _wild_forms(fit, clusters)
    a, Gm, yhat_r, er, cl, Gn, y = f["a"], f["G"], f["yhat_r"], f["er"], f["cl"], f["n_cl"], f["y"]
    t0 = float(a @ y) / np.sqrt(np.sum((Gm @ y) ** 2))
    W6 = cm.rc.WEBB
    if enumerate_all:
        V_ = np.array(list(itertools.product(W6, repeat=Gn)))
    else:
        rng = np.random.default_rng(seed)
        V_ = rng.choice(W6, size=(B, Gn))
    Ys = yhat_r[None, :] + V_[:, cl] * er[None, :]
    ts = (Ys @ a) / np.sqrt(np.sum((Ys @ Gm.T) ** 2, axis=1))
    if enumerate_all:
        return float(np.mean(np.abs(ts) >= abs(t0) - 1e-10 * abs(t0))), t0, len(V_)
    return float((1 + np.sum(np.abs(ts) >= abs(t0) - 1e-10 * abs(t0))) / (B + 1)), t0, B


def _fit(d, spec, re_groups="design"):
    """rb1's fit_spec, with the random-intercept (or fixed-effect) groups set to designs or studies."""
    import meta_c as mc
    dd = d.copy()
    if re_groups == "study":
        dd["design"] = dd["study4"]
    return mc.fit_spec(dd, spec)


SPECS = [
    # key, rb1 spec, RE/FE groups, scale, label
    ("weighted", "weighted", "design", "sigma", "3-level RE (design intercepts), inverse-variance (primary)"),
    ("unweighted", "unweighted", "design", "sigma", "Design RE, unweighted (LMM)"),
    ("balanced", "balanced", "design", "sigma", "Design-balanced weights (CE-RVE)"),
    ("weighted_S", "weighted", "design", "S", "3-level RE, inverse-variance, on S = 2(1/sigma*-1)"),
    ("fe_weighted", "fe_weighted", "design", "sigma", "Design FE, inverse-variance"),
    ("fe_unweighted", "fe_unweighted", "design", "sigma", "Design FE, unweighted (OLS)"),
    ("weighted_studyRE", "weighted", "study", "sigma", "3-level RE with study intercepts (Marin's corpora share one), inverse-variance"),
    ("unweighted_studyRE", "unweighted", "study", "sigma", "Study RE, unweighted (LMM)"),
    ("fe_weighted_study", "fe_weighted", "study", "sigma", "Study FE (Marin's corpora share one intercept), inverse-variance"),
]


def _toS(d):
    dS = d.copy()
    dS["y"] = 2 * (1 / d["y"] - 1)
    dS["se"] = 2 * d["se"] / d["y"] ** 2
    return dS


def slope_rows(d, sample, cluster_sets, log=cm.log, B=B_WILD):
    """Every spec x every cluster structure: slope, CR2 s.e./df/p, wild p (random draws and enumeration)."""
    import meta_c as mc
    rows = []
    for key, sp, grp, scale, lab in SPECS:
        dd = _toS(d) if scale == "S" else d
        if grp == "study" and dd["study4"].nunique() < 2:
            continue
        try:
            fit = _fit(dd, sp, grp)
        except Exception as e:  # noqa: BLE001  (recorded)
            rows.append(dict(sample=sample, spec_key=key, spec=lab, error=repr(e)))
            continue
        b = float(fit["cvec_slope"] @ fit["beta"])
        for cname, ccol in cluster_sets:
            cl = dd[ccol].values
            G = pd.Series(cl).nunique()
            if G < 2 or (grp == "study" and ccol == "design6"):
                continue            # study-level working models are clustered by study only (nesting)
            # CR2 needs every RE/FE group nested in a cluster (true: designs and studies nest in studies)
            se_r, df, _ = mc.cr2(fit, fit["cvec_slope"], cl)
            p_cr2 = float(2 * tdist.sf(abs(b / se_r), max(df, 1.0))) if se_r > 0 else np.nan
            # rb1's seeds for the design-cluster rows of the full sample (reproduction); this module's otherwise
            if cname == "designs (6)" and sample == "all six designs (excl. Porian)" and key in ("weighted", "unweighted", "balanced", "fe_weighted", "fe_unweighted", "weighted_S"):
                rb1key = {"weighted_S": "S"}.get(key, key)
                seed = cm.cm1.seed_of(f"wcr|{rb1key}")
            else:
                seed = cm.seed_of(f"wcr|{sample}|{key}|{cname}")
            p_w, t0, Bw = wild_p(fit, cl, B=B, seed=seed)
            p_e, _, ne = (wild_p(fit, cl, enumerate_all=True) if G <= 6 else (np.nan, np.nan, 0))
            q = tdist.ppf(0.975, max(df, 1.0))
            rows.append(dict(sample=sample, spec_key=key, spec=lab, scale=scale, re_groups=grp, clusters=cname, G=G,
                             k=len(dd), slope=b, se_model=fit["se_slope_model"], se_cr2=se_r, df_cr2=df,
                             lo_cr2=b - q * se_r, hi_cr2=b + q * se_r, p_cr2=p_cr2, t_cr2=t0, p_wild=p_w, B_wild=Bw,
                             p_wild_enum=p_e, n_enum=ne,
                             tau_between=float(np.sqrt(max(fit.get("tau_d2", np.nan), 0))) if np.isfinite(fit.get("tau_d2", np.nan)) else np.nan,
                             reliable_df=bool(df >= 4 and G >= 6)))
        log(f"    [{sample}] {key}: slope {b:+.4f}; " + "; ".join(
            f"{r['clusters']}: CR2 p {r['p_cr2']:.3f} (df {r['df_cr2']:.1f}), wild p {r['p_wild']:.3f} / enum {r['p_wild_enum']:.3f}"
            for r in rows if r.get("spec_key") == key and r.get("sample") == sample and "error" not in r))
    return rows


def run(log=cm.log):
    import meta_c as mc
    d, d_all = load()
    d["design6"] = d["design"]
    rows = []
    cl_full = [("studies (4; Marin one cluster)", "study4"), ("designs (6)", "design6")]
    rows += slope_rows(d, "all six designs (excl. Porian)", cl_full, log=log)
    iso = d[d.design != cm.FARSEER].reset_index(drop=True)
    rows += slope_rows(iso, "IsoFLOP designs only (excl. Farseer path)",
                       [("studies (3; Marin one cluster)", "study4"), ("designs (5)", "design6")], log=log)
    low = d[d.log10C <= np.log10(3e20) + 1e-9].reset_index(drop=True)
    rows += slope_rows(low, "budgets <= 3e20 (common window)", cl_full, log=log)
    top_idx = d.groupby("design")["log10C"].idxmax()
    lto = d[~d.index.isin(top_idx.values)].reset_index(drop=True)
    rows += slope_rows(lto, "leave out each design's largest budget", cl_full, log=log)
    # hinge (descriptive), full sample
    dh = d.copy()
    dh["log10C"] = 20.0 + np.maximum(d["log10C"] - 20.5, 0.0)
    for cname, ccol in cl_full:
        fh = mc.fit_spec(dh, "weighted")
        b = float(fh["cvec_slope"] @ fh["beta"])
        se_r, df, _ = mc.cr2(fh, fh["cvec_slope"], dh[ccol].values)
        seed = cm.cm1.seed_of("wcr|hinge") if ccol == "design6" else cm.seed_of(f"wcr|hinge|{cname}")
        p_w, t0, Bw = wild_p(fh, dh[ccol].values, seed=seed)
        p_e, _, ne = wild_p(fh, dh[ccol].values, enumerate_all=True)
        rows.append(dict(sample="all six designs (excl. Porian)", spec_key="weighted_hinge",
                         spec="3-level RE, hinge at 3e20 (slope above the knot; descriptive)", scale="sigma",
                         re_groups="design", clusters=cname, G=dh[ccol].nunique(), k=len(dh), slope=b,
                         se_model=fh["se_slope_model"], se_cr2=se_r, df_cr2=df, p_cr2=float(2 * tdist.sf(abs(b / se_r), max(df, 1))),
                         t_cr2=t0, p_wild=p_w, B_wild=Bw, p_wild_enum=p_e, n_enum=ne, reliable_df=False))
    S = pd.DataFrame(rows)
    # reproduction of rb1's published design-cluster wild p (same seeds and B)
    rb1 = pd.read_csv(cm.RB1_SLOPES)
    rb1 = rb1[rb1["sample"] == "all six designs (excl. Porian)"].set_index("spec_key")
    mine = S[(S["sample"] == "all six designs (excl. Porian)") & (S.clusters == "designs (6)")].set_index("spec_key")
    chk = []
    for k in ("weighted", "unweighted", "balanced", "fe_weighted", "fe_unweighted", "weighted_S", "weighted_hinge"):
        if k in rb1.index and k in mine.index:
            chk.append(dict(spec_key=k, slope=mine.loc[k, "slope"], slope_rb1=rb1.loc[k, "slope"],
                            p_cr2=mine.loc[k, "p_cr2"], p_cr2_rb1=rb1.loc[k, "p_cr2_design"],
                            p_wild=mine.loc[k, "p_wild"], p_wild_rb1=rb1.loc[k, "p_wcr_design"],
                            p_cr2_study=S[(S["sample"] == "all six designs (excl. Porian)") & (S.spec_key == k) &
                                          S.clusters.str.startswith("studies")].p_cr2.iloc[0],
                            p_cr2_study_rb1=rb1.loc[k, "p_cr2_study"]))
    chk = pd.DataFrame(chk)
    for c in ("slope", "p_cr2", "p_wild", "p_cr2_study"):
        chk[f"reldiff_{c}"] = np.abs(chk[c] / chk[f"{c}_rb1"] - 1)
    # predictions with study clusters (primary fit)
    fit = mc.fit_spec(d, "weighted")
    P = []
    for cname, ccol in cl_full:
        pr = mc.predict(fit, d, [19.0, 20.0, 21.0], clusters=d[ccol].values)
        pr["clusters"] = cname
        P.append(pr)
    P = pd.concat(P, ignore_index=True)
    P["spec"] = "3-level RE (design intercepts), inverse-variance (primary)"
    # p-value ranges
    full = S[(S["sample"] == "all six designs (excl. Porian)") & (S.spec_key != "weighted_hinge")]
    rng = []

    def addr(lab, sub):
        v = pd.concat([sub.p_cr2, sub.p_wild, sub.p_wild_enum]).dropna()
        rng.append(dict(set=lab, n_pvalues=len(v), p_min=float(v.min()), p_max=float(v.max()),
                        cr2_min=float(sub.p_cr2.min()), cr2_max=float(sub.p_cr2.max()),
                        wild_min=float(sub[["p_wild", "p_wild_enum"]].min().min()),
                        wild_max=float(sub[["p_wild", "p_wild_enum"]].max().max()),
                        slope_min=float(sub.slope.min()), slope_max=float(sub.slope.max()), specs="; ".join(sorted(sub.spec_key.unique()))))
    rb1specs = ["weighted", "unweighted", "balanced", "weighted_S", "fe_weighted", "fe_unweighted"]
    nofe = ["weighted", "unweighted", "balanced", "weighted_S"]
    addr("Marin one cluster (4 clusters): RE, balanced, S-scale and design-FE rows", full[full.clusters.str.startswith("studies") & full.spec_key.isin(rb1specs)])
    addr("Marin one cluster (4 clusters): all rows incl. study-RE and study-FE", full[full.clusters.str.startswith("studies")])
    addr("Six design clusters: RE, balanced, S-scale and design-FE rows", full[(full.clusters == "designs (6)") & full.spec_key.isin(rb1specs)])
    addr("Six design clusters: without FE rows (rb1's published range)", full[(full.clusters == "designs (6)") & full.spec_key.isin(nofe)])
    addr("Four to six clusters: RE, balanced, S-scale and design-FE rows", full[full.spec_key.isin(rb1specs)])
    addr("Four to six clusters: all rows", full)
    iso_rows = S[S["sample"] == "IsoFLOP designs only (excl. Farseer path)"]
    addr("IsoFLOP designs only, five design clusters (R3's round-2 sample)", iso_rows[(iso_rows.clusters == "designs (5)") & iso_rows.spec_key.isin(rb1specs)])
    return dict(slopes=S, check=chk, pred=P, prange=pd.DataFrame(rng))
