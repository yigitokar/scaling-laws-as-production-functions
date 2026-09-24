"""wedges.py -- w and the expenditure share s = (w-1)/w for every model x technology; joint bootstrap summaries
(R1 9f); lab-own assignment; conventions (R2 Major 4a); in-support vs extrapolated splits (R2 Major 3); family-level
objects for common-D families; MoE bounds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ra2common import log_w, ln_mstar, share

L_SEQ = 4096   # DeepSeek's l_seq in M = 72 n_layer d^2 + 12 n_layer d l_seq (Bi et al. 2024, Eq. 2)


def n_used(df, conv, active=True):
    """Parameter count in a technology's convention. active=False evaluates MoE at total parameters."""
    N = df["N"].values if active else df["N_total"].values
    if conv == "total":
        return N.astype(float)
    if conv == "nonemb":
        return df["N_nonemb"].values.astype(float)
    if conv == "olmo":   # OLMo MODEL_PARAMS: excludes the input embedding only (m3 rule: tied -> subtract V d)
        tied = df["tied"].fillna(True).astype(bool).values
        return (N - np.where(tied, 1.0, 0.5) * df["N_emb"].values).astype(float)
    if conv == "ds":     # DeepSeek: non-embedding FLOPs/token / 6
        M = 6 * df["N_nonemb"].values + 12 * df["n_layer"].values * df["d_model"].values * L_SEQ
        return (M / 6.0).astype(float)
    raise ValueError(conv)


def wedge_long(df, techs, keys, active=True):
    out = []
    lnD = np.log(df["D"].values)
    for k in keys:
        t = techs[k]
        N = n_used(df, t.n_conv, active)
        lnN = np.log(N)
        lw = log_w(lnN, lnD, t.alpha, t.beta, t.lnG)
        C = 6 * N * df["D"].values
        lo = hi = sd = np.full(len(df), np.nan)
        if t.draws is not None and len(t.draws):
            LW = log_w(lnN[None, :], lnD[None, :], t.draws[:, 0:1], t.draws[:, 1:2], t.draws[:, 2:3])
            lo, hi = np.nanpercentile(LW, [2.5, 97.5], axis=0)
            sd = np.nanstd(LW, 0, ddof=1)
        sup = t.support
        M = df["D"].values / N
        out.append(pd.DataFrame(dict(
            uid=df["uid"].values, tech=k, N_used=N, M_used=M, C_used=C,
            lnMstar=ln_mstar(C, t.alpha, t.beta, t.lnG), lnw=lw, w=np.exp(lw), w_lo=np.exp(lo), w_hi=np.exp(hi),
            sd_lnw=sd, s=share(np.exp(lw)), s_lo=share(np.exp(lo)), s_hi=share(np.exp(hi)),
            M_extrap=((M < sup["M_min"]) | (M > sup["M_max"])).astype(float)
                     if np.isfinite(sup.get("M_min", np.nan)) and np.isfinite(sup.get("M_max", np.nan)) else np.nan,
            C_extrap=C > sup.get("C_max", np.inf))))
    L = pd.concat(out, ignore_index=True)
    L["Mstar"] = np.exp(L["lnMstar"])
    return L


def joint_boot(df, t, conv_active=True):
    """Bootstrap distribution, across technology draws, of the sample median w, median s and the share with w > 1
    (one technology draw is shared by all models: R1 9f). Returns dict of point + 2.5/97.5 percentiles."""
    N = n_used(df, t.n_conv, conv_active)
    lnN, lnD = np.log(N), np.log(df["D"].values)
    ok = np.isfinite(lnN)
    lnN, lnD = lnN[ok], lnD[ok]
    lw = log_w(lnN, lnD, t.alpha, t.beta, t.lnG)
    pt = dict(n=int(ok.sum()), med_w=float(np.median(np.exp(lw))), share_gt1=float((lw > 0).mean()),
              med_s=float(np.median(share(np.exp(lw)))))
    if t.draws is None or not len(t.draws):
        return {**pt, **{f"{k}_{q}": np.nan for k in ("med_w", "share_gt1", "med_s") for q in ("lo", "hi")}, "B": 0}
    LW = log_w(lnN[None, :], lnD[None, :], t.draws[:, 0:1], t.draws[:, 1:2], t.draws[:, 2:3])
    mw = np.median(np.exp(LW), axis=1)
    sh = (LW > 0).mean(axis=1)
    ms = np.median(share(np.exp(LW)), axis=1)
    q = lambda v: np.percentile(v, [2.5, 97.5])
    return {**pt, "med_w_lo": q(mw)[0], "med_w_hi": q(mw)[1], "share_gt1_lo": q(sh)[0], "share_gt1_hi": q(sh)[1],
            "med_s_lo": q(ms)[0], "med_s_hi": q(ms)[1], "B": len(t.draws)}


def band(L, keys):
    """Per-model envelope over technologies `keys`: [min of lower 95% bounds (or points), max of upper]."""
    X = L[L["tech"].isin(keys) & np.isfinite(L["w"])].copy()
    X["lo_"] = X["w_lo"].fillna(X["w"])
    X["hi_"] = X["w_hi"].fillna(X["w"])
    g = X.groupby("uid").agg(band_lo=("lo_", "min"), band_hi=("hi_", "max"), pt_min=("w", "min"), pt_max=("w", "max"),
                             n_tech=("tech", "nunique"))
    return g.reset_index()


def lab_own(B, L, lab_map):
    """Lab-own technology wedge for models of labs with their own technology (primary + range over alternatives)."""
    rows = []
    for _, r in B.iterrows():
        lab = r["dev"]
        if lab not in lab_map:
            continue
        prim, alts = lab_map[lab]
        x = L[(L["uid"] == r["uid"])].set_index("tech")
        if prim not in x.index or not np.isfinite(x.loc[prim, "w"]):
            continue
        pts = [x.loc[k, "w"] for k in [prim] + alts if k in x.index and np.isfinite(x.loc[k, "w"])]
        rows.append(dict(uid=r["uid"], lab_tech=prim, w_lab=x.loc[prim, "w"], w_lab_lo=x.loc[prim, "w_lo"],
                         w_lab_hi=x.loc[prim, "w_hi"], s_lab=share(x.loc[prim, "w"]),
                         w_lab_altmin=min(pts), w_lab_altmax=max(pts), Mstar_lab=x.loc[prim, "Mstar"]))
    return pd.DataFrame(rows)


def family_level(B, mask, wcol):
    """Common-D families: family-level revealed object. With a shared D chosen for the family and sizes N_j chosen
    freely, the D first-order condition gives sum_j omega_j l_j / w_j = 1 (omega_j = training-compute shares,
    l_j = lifetime/training ratio of member j). Hence W_f = [sum_j omega_j / w_j]^(-1) equals the compute-weighted
    mean lifetime/training ratio when l_j is uncorrelated with 1/w_j (omega-weighted), and is a LOWER bound on it when
    planned demand is larger for the more over-trained members (Cov(l, 1/w) < 0). If sizes come from a tier menu
    (N_j not chosen), inference cost is sunk with respect to D and the family's D reveals nothing about T."""
    X = B[mask].copy()
    rows = []
    for gen, g in X.groupby("gen"):
        if len(g) < 2:
            continue
        om = g["Cmp"] / g["Cmp"].sum()
        W = 1.0 / np.sum(om / g[wcol])
        rows.append(dict(gen=gen, n=len(g), D_ratio=g["D"].max() / g["D"].min(), W_f=W, s_f=share(W),
                         w_min=g[wcol].min(), w_max=g[wcol].max(), mean_w_cw=float(np.sum(om * g[wcol]))))
    return pd.DataFrame(rows)


def verify_family_foc(seed=0):
    """Numerical check of the family first-order condition sum_j omega_j l_j / w_j = 1 (shared D, free N_j):
    solve min_{N_1..N_J, D} sum_j N_j (6D + 2 T_j) s.t. L(N_j, D) <= ell_j under the Besiroglu technology."""
    import sl
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    t = sl.BESIROGLU
    J = 4
    N0 = np.array([1e9, 3e9, 8e9, 30e9])
    D0 = 5e12
    ell = t.loss(N0, D0)                     # loss targets
    T = rng.uniform(0.5, 30, J) * D0         # planned lifetime tokens
    # eliminate N_j: for given D, N_j solves L(N_j, D) = ell_j
    def Nj(D):
        v = t.B * D ** -t.beta
        return (t.A / (ell - t.E - v)) ** (1 / t.alpha)
    def cost(lnD):
        D = np.exp(lnD[0])
        v = t.B * D ** -t.beta
        if np.any(ell - t.E - v <= 0):
            return 1e60
        return float(np.sum(Nj(D) * (6 * D + 2 * T)))
    r = minimize(cost, [np.log(D0)], method="Nelder-Mead", options=dict(xatol=1e-12, fatol=1e-12, maxiter=5000))
    D = np.exp(r.x[0])
    N = Nj(D)
    w = t.wedge(N, D)
    l = 1 + T / (3 * D)
    om = N / N.sum()
    return dict(lhs=float(np.sum(om * l / w)), w=w.tolist(), l=l.tolist(), D=D, N=N.tolist())
