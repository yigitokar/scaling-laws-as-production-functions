"""wedge.py -- revealed inference demand for every model under every technology (model_spec Proposition 4).

For model i with (N_i, D_i) and technology t with draws (alpha_b, beta_b, lnG_b):
  ln w_ib = (alpha_b + beta_b) lnG_b - alpha_b ln N_i + beta_b ln D_i
  T/D = 3 (w - 1),  T = 3 D (w - 1),  CE = C_min / (6ND) = [((alpha+beta w)/(alpha+beta))^(1/gamma) w^(-1/alpha)]^-1.
Point = full-sample estimate; 95% interval = 2.5/97.5 percentiles over the technology's bootstrap draws (literature
points have no interval). Partial-identification band for model i = [min_t lo_it, max_t hi_it] over the technologies
in BAND_SET (+ the lab's own technology). Extrapolation is flagged when the model's M = D/N lies outside the
technology's design range of M (and, separately, when 6ND exceeds the largest design budget: 75% of Sample B for the
Chinchilla design, not every model -- small open models such as Pythia-70M are inside it).
Parameter-count conventions: 'total' technologies use N (active parameters for MoE); the non-embedding Farseer row
uses N_nonemb = N - vocab x width x (1 or 2) from the model's config.json; OLMo's ladder uses N - input embedding.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from common import ROOT, PROC, ce_of_w, homothetic_ce, homothetic_w, log, log_w
import technologies as T


def n_for_tech(df, t):
    if t.n_conv == "total":
        return df["N"].values
    if t.n_conv == "nonemb":
        return df["N_nonemb"].values if "N_nonemb" in df else np.full(len(df), np.nan)
    if t.n_conv == "olmo":   # MODEL_PARAMS of the OLMo ladder excludes only the input embedding
        if "N_emb" not in df:
            return np.full(len(df), np.nan)
        tied = df["tied"].fillna(True).astype(bool).values
        # [Review m3] start from N (= active parameters for MoE), not N_total: the builder used OLMoE's 6.9B total
        # parameters here, contrary to the module's active-N convention for every other technology
        return np.where(df["lab"].eq("AI2").values, df["N"].values - np.where(tied, 1.0, 0.5) * df["N_emb"].values, np.nan)
    raise ValueError(t.n_conv)


def wedge_long(df, techs, keys=None):
    """Long table model x technology: w (point, 95% CI), T/D, T, CE and extrapolation flags."""
    keys = keys or list(techs)
    out = []
    lnD = np.log(df["D"].values)
    for k in keys:
        t = techs[k]
        N = n_for_tech(df, t)
        lnN = np.log(N)
        lw = log_w(lnN, lnD, t.alpha, t.beta, t.lnG)
        w = np.exp(lw)
        ce = ce_of_w(w, t.alpha, t.beta)
        lo = hi = ce_lo = ce_hi = sd = np.full(len(df), np.nan)
        if t.draws is not None:
            al, be, g = t.draws[:, 0:1], t.draws[:, 1:2], t.draws[:, 2:3]
            LW = log_w(lnN[None, :], lnD[None, :], al, be, g)      # (B, n)
            lo, hi = np.exp(np.nanpercentile(LW, 2.5, 0)), np.exp(np.nanpercentile(LW, 97.5, 0))
            sd = np.nanstd(LW, 0, ddof=1)
            CE = ce_of_w(np.exp(LW), al, be)
            ce_lo, ce_hi = np.nanpercentile(CE, 2.5, 0), np.nanpercentile(CE, 97.5, 0)
        M = df["D"].values / N
        sup = t.support
        out.append(pd.DataFrame(dict(
            uid=df["uid"].values, tech=k, N_used=N, M_tech=M, w=w, w_lo=lo, w_hi=hi, sd_lnw=sd,
            TD=3 * (w - 1), TD_lo=3 * (lo - 1), TD_hi=3 * (hi - 1), T=3 * df["D"].values * (w - 1),
            CE=ce, CE_lo=ce_lo, CE_hi=ce_hi,
            M_extrap=(M < sup.get("M_min", 0)) | (M > sup.get("M_max", np.inf)),
            C_extrap=6 * N * df["D"].values > sup.get("C_max", np.inf),
            N_extrap=N > sup.get("N_max", np.inf))))
    return pd.concat(out, ignore_index=True)


def band(long, df, techs):
    """Partial-identification band per model: union of 95% intervals (points for literature technologies) over
    BAND_SET, plus the lab-own technologies for that lab's models. Also a 'no Hoffmann' variant."""
    rows = []
    lab = df.set_index("uid")["lab"]
    L = long.set_index(["uid", "tech"])
    for u in df["uid"]:
        ks = list(T.BAND_SET) + T.LAB_TECH.get(lab.get(u), [])
        los, his, pts, used = [], [], [], []
        for k in ks:
            if (u, k) not in L.index:
                continue
            r = L.loc[(u, k)]
            if not np.isfinite(r["w"]):
                continue
            used.append(k)
            pts.append(r["w"])
            los.append(r["w_lo"] if np.isfinite(r["w_lo"]) else r["w"])
            his.append(r["w_hi"] if np.isfinite(r["w_hi"]) else r["w"])
        nh = [i for i, k in enumerate(used) if k != "hoff"]
        cs = [i for i, k in enumerate(used) if k in T.CORE_SET or k in T.LAB_TECH.get(lab.get(u), [])]
        rows.append(dict(uid=u, band_lo=min(los) if los else np.nan, band_hi=max(his) if his else np.nan,
                         pt_min=min(pts) if pts else np.nan, pt_max=max(pts) if pts else np.nan,
                         band_lo_nohoff=min(los[i] for i in nh) if nh else np.nan,
                         band_hi_nohoff=max(his[i] for i in nh) if nh else np.nan,
                         band_lo_core=min(los[i] for i in cs) if cs else np.nan,
                         band_hi_core=max(his[i] for i in cs) if cs else np.nan, n_tech=len(used),
                         techs_used=",".join(used)))
    return pd.DataFrame(rows)


def homothetic_table(df):
    rows = []
    for Ms in T.HOMO_MSTAR:
        for rho in T.HOMO_RHO:
            w = homothetic_w(df["M"].values, Ms, rho)
            rows.append(pd.DataFrame(dict(uid=df["uid"].values, Mstar=Ms, rho=rho, w=w, TD=3 * (w - 1),
                                          C_over_Cmin=1 / homothetic_ce(df["M"].values, Ms, rho))))
    return pd.concat(rows, ignore_index=True)


# ----------------------------------------------------------------------------- Farseer's own (non-homothetic) form
def farseer_eq3():
    """Fit Li et al. (2025b) Eq. 3 on the Farseer panel (m2's harmonised panel: non-embedding N, English BPC) with m2's
    estimator code (NLS on log loss, authors' multi-round starting values). Returns theta and a function giving
    w = (dL/dlnN)/(dL/dlnD) numerically (the lifetime-cost FOC holds for any technology: w = 1 + T/(3D))."""
    sys.path.insert(0, os.path.join(ROOT, "code", "analysis", "m2_techpanel"))
    import m2_est as me   # read-only import of m2's estimator
    fa = pd.read_csv(os.path.join(ROOT, "data", "processed", "m2_techpanel", "panel_farseer.csv"))
    N, D, L = fa.N.values, fa.D.values, fa.L.values
    st = me.farseer_starts(N, D, L)
    th, f = me.fit_farseer(N, D, L, "nls", starts=[st])

    def lnL(n, d):
        return me.farseer_lnL(th, np.asarray(n, float), np.asarray(d, float))

    def wfun(n, d, h=1e-4):
        n, d = np.asarray(n, float), np.asarray(d, float)
        # d ln L / d ln N and d ln L / d ln D by central differences (L > 0, so the ratio of log-derivatives equals
        # the ratio of level derivatives)
        gN = (lnL(n * np.exp(h), d) - lnL(n * np.exp(-h), d)) / (2 * h)
        gD = (lnL(n, d * np.exp(h)) - lnL(n, d * np.exp(-h))) / (2 * h)
        return gN / gD

    def mstar(C):
        from scipy.optimize import minimize_scalar
        out = []
        for c in np.atleast_1d(C):
            obj = lambda ln: float(lnL(np.exp(ln), c / (6 * np.exp(ln))))
            rr = minimize_scalar(obj, bounds=(np.log(1e6), np.log(1e14)), method="bounded", options=dict(xatol=1e-9))
            Ns = np.exp(rr.x)
            out.append(c / (6 * Ns ** 2))
        return np.array(out)

    return th, f, wfun, mstar


def main(ch):
    log("wedge: technologies")
    techs = T.load_all()
    tt = T.tech_table(techs)
    tt.to_csv(os.path.join(PROC, "technologies_used.csv"), index=False)
    log("wedge: model x technology table")
    long = wedge_long(ch, techs)
    bd = band(long, ch, techs)
    homo = homothetic_table(ch)
    log("wedge: Farseer Eq. 3 (non-homothetic) fit")
    th, fobj, wfun, mstar_f = farseer_eq3()
    ok = ch["N_nonemb"].notna().values
    wf = np.full(len(ch), np.nan)
    wf[ok] = wfun(ch.loc[ok, "N_nonemb"].values, ch.loc[ok, "D"].values)
    eq3 = pd.DataFrame(dict(uid=ch["uid"].values, w_eq3=wf,
                            Mstar_eq3_ownC=np.where(ok, np.nan, np.nan)))
    # M*(C) of the Eq. 3 technology at each model's own (non-embedding) compute -- slow optimiser, only Sample-B rows
    idx = np.where(ok)[0]
    ms = mstar_f(6 * ch["N_nonemb"].values[idx] * ch["D"].values[idx])
    eq3.loc[idx, "Mstar_eq3_ownC"] = ms
    pd.DataFrame(dict(param=["a1", "g1", "b1", "a2", "g2", "b2", "a3", "g3", "b3"], value=th)).to_csv(
        os.path.join(PROC, "farseer_eq3_theta.csv"), index=False)
    long.to_csv(os.path.join(PROC, "wedge_long.csv"), index=False)
    bd.to_csv(os.path.join(PROC, "wedge_band.csv"), index=False)
    homo.to_csv(os.path.join(PROC, "wedge_homothetic.csv"), index=False)
    eq3.to_csv(os.path.join(PROC, "wedge_farseer_eq3.csv"), index=False)
    return techs, long, bd, homo, eq3, mstar_f
