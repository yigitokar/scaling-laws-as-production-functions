"""magnitude.py -- magnitude bounds under the new identified set (fix list S4).

Per model (Proposition A6(iv); the magnitude clause that WP3 moves into the text): with the identified set widened by
ln tau and the curvature k = 1/sigma* - 1 in [k_L, k_U],
    ln w_lo = (k_L if d_lo > 0 else k_U) d_lo,   ln w_hi = (k_U if d_hi > 0 else k_L) d_hi,
with d_lo = dlo - ln tau and d_hi = dhi + ln tau; s = (w - 1)/w. Medians over the 77 models, and [review] over the
decision units (a unit's bounds are the compute-weighted harmonic means of its members' bounds).
Curvature ranges: [0.40, 0.52] (in-design values, version 3); [0.40, k(sigma_top)] with sigma_top = 0.594 the
top-budget value (R2 NM3(d): "0.68"); sigma* at the lower end of its identified set at each model's compute,
k_U(C_i) = 1/sigma_lin(C_i) - 1 (rb1/rb4: the Chinchilla and Llama 3 drift continued linearly from the top-budget
value; R1 New 2(c)), with k_L = 0.40 or with k_L = k(0.70) as in R1's drift-agnostic set [sigma_lin(C), 0.70].
Decision-level medians with the reference zero points fixed (W12): ln w_i(sigma) = (1/sigma - 1) ln(M_i/M*_ref,i), and
a family unit's W = [sum_j h_j / w_j]^-1 with compute shares h_j (rb2 decisions.family_W).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import rb5common as K


def sigma_top():
    t = pd.read_csv(K.RB1_TOP)
    r = t[t["set"] == "Chinchilla + Llama 3, budgets >= 6e20"].iloc[0]
    return float(r["mean_fixed"]), float(r["lo_hksj"]), float(r["hi_hksj"])


def sigma_lin(C, column="sigma_lower|drift of Chinchilla and Llama 3 (design FE, inverse variance)"):
    """Lower end of the identified set for sigma*(C), interpolated in log10 C (constant outside the table)."""
    t = pd.read_csv(K.RB1_PI_SIGMA)
    x, y = t["log10C"].values, t[column].values
    return np.interp(np.log10(np.asarray(C, float)), x, y)


def bounds_s(dlo, dhi, kL, kU, tau=1.0):
    lt = np.log(tau)
    a, b = dlo - lt, dhi + lt
    kL = np.broadcast_to(kL, a.shape)
    kU = np.broadcast_to(kU, a.shape)
    lwl = np.where(a > 0, kL * a, kU * a)
    lwh = np.where(b > 0, kU * b, kL * b)
    return K.share(np.exp(lwl)), K.share(np.exp(lwh)), np.exp(lwl), np.exp(lwh)


def curvature_ranges(C):
    st, st_lo, st_hi = sigma_top()
    kl_lin = 1 / sigma_lin(C) - 1
    kl_pool = 1 / sigma_lin(C, "sigma_lower|pooled drift (primary three-level meta-regression)") - 1
    return {
        "k in [0.40, 0.52] (in-design sigma* 0.66-0.71; version 3)": (0.40, 0.52),
        f"k in [0.40, {1 / st - 1:.3f}] (to the top-budget value sigma* = {st:.3f})": (0.40, 1 / st - 1),
        "k in [0.40, k(sigma_lin(C_i))]: sigma* at the lower end of its identified set (Chinchilla-Llama 3 drift)":
            (0.40, kl_lin),
        "k in [k(0.70), k(sigma_lin(C_i))]: R1's drift-agnostic set [sigma_lin(C), 0.70]": (1 / 0.70 - 1, kl_lin),
        "k in [0.40, k(sigma_lin(C_i))], pooled drift (three-level meta-regression)": (0.40, kl_pool),
    }


def model_bounds_table(PB, label, taus=(1.0, 1.3, 1.84, 3.4, 4.4)):
    ranges = curvature_ranges(PB["Cmp"].values)
    rows = []
    for lab, (kL, kU) in ranges.items():
        for tau in taus:
            slo, shi, wlo, whi = bounds_s(PB["dlo"].values, PB["dhi"].values, kL, kU, tau)
            rows.append(dict(set=label, level="models", curvature=lab, tau=tau, n=len(PB),
                             k_L=float(np.min(kL)), k_U_min=float(np.min(kU)), k_U_max=float(np.max(kU)),
                             median_s_lower=float(np.median(slo)), median_s_upper=float(np.median(shi)),
                             median_w_lower=float(np.median(wlo)), median_w_upper=float(np.median(whi))))
    return pd.DataFrame(rows)


def unit_bounds_table(PB, U, units_label, label, taus=(1.0, 1.3, 1.84, 3.4, 4.4)):
    """[review] Decision-level medians of the bounds on s: a unit's W = [sum_j h_j / w_j]^-1 is increasing in every
    member's w_j, so its bounds are the compute-weighted harmonic means of the members' bounds (singletons: the
    member's own). Reported because the introduction's H5 sentence speaks of the median decision."""
    ranges = curvature_ranges(PB["Cmp"].values)
    X = PB[["uid", "Cmp"]].reset_index(drop=True).merge(U[["uid", "unit"]], on="uid", how="left")
    if X["unit"].isna().any():
        raise ValueError("models without a unit in the magnitude table")
    rows = []
    for lab, (kL, kU) in ranges.items():
        for tau in taus:
            slo, shi, wlo, whi = bounds_s(PB["dlo"].values, PB["dhi"].values, kL, kU, tau)
            Y = X.assign(wlo=wlo, whi=whi)
            Wl, Wh = [], []
            for _, g in Y.groupby("unit", sort=False):
                Wl.append(family_W(g["wlo"].values, g["Cmp"].values))
                Wh.append(family_W(g["whi"].values, g["Cmp"].values))
            Wl, Wh = np.array(Wl), np.array(Wh)
            rows.append(dict(set=label, level=f"decisions ({units_label})", curvature=lab, tau=tau, n=len(Wl),
                             k_L=float(np.min(kL)), k_U_min=float(np.min(kU)), k_U_max=float(np.max(kU)),
                             median_s_lower=float(np.median(K.share(Wl))), median_s_upper=float(np.median(K.share(Wh))),
                             median_w_lower=float(np.median(Wl)), median_w_upper=float(np.median(Wh))))
    return pd.DataFrame(rows)


def family_W(w, C_):
    om = C_ / C_.sum()
    return 1.0 / np.sum(om / w)


def unit_median(B, U, w):
    X = B[["uid", "Cmp"]].assign(w=w).merge(U[["uid", "unit", "unit_type"]], on="uid")
    Ws = []
    for u, g in X.groupby("unit", sort=False):
        Ws.append(family_W(g["w"].values, g["Cmp"].values) if len(g) > 1 else float(g["w"].iloc[0]))
    Ws = np.array(Ws)
    return float(np.median(K.share(Ws))), float(np.mean(Ws > 1)), len(Ws)


def fixed_zero_point(B, units):
    """Median decision share with the reference zero points fixed and a constant or compute-dependent sigma*."""
    st, st_lo, st_hi = sigma_top()
    u = np.log(B["M"].values) - np.log(B["Mstar_ref"].values)
    lnw_ref = np.log(B["w_ref"].values)
    scen = [("reference (kappa-free Chinchilla, sigma* = 0.701)", None)]
    for s in (0.70, 0.65, 0.622, 0.62, 0.60, st, 0.55):
        scen.append((f"constant sigma* = {s:.3f}".rstrip("0").rstrip("."), s))
    scen.append(("sigma*(C_i) at the lower end of its identified set (Chinchilla-Llama 3 drift)", "lin"))
    rows = []
    for lab, s in scen:
        if s is None:
            lnw = lnw_ref
        elif s == "lin":
            lnw = (1 / sigma_lin(B["Cmp"].values) - 1) * u
        else:
            lnw = (1 / s - 1) * u
        w = np.exp(lnw)
        row = dict(scenario=lab, sigma=np.nan if s in (None, "lin") else s, median_s_models=float(np.median(K.share(w))),
                   share_w_gt1_models=float(np.mean(w > 1)))
        for un, U in units.items():
            m, g1, n = unit_median(B, U, w)
            row[f"median_s_decisions_{un}"] = m
            row[f"share_w_gt1_decisions_{un}"] = g1
            row[f"n_decisions_{un}"] = n
        rows.append(row)
    return pd.DataFrame(rows)
