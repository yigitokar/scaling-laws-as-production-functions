"""pid.py -- sigma*(C) beyond the designs as an object of partial identification, and its propagation into the wedge
scale k = 1/sigma* - 1 and the revealed-demand magnitudes (task 3; R1 New 2 requests 1-2, R3 N1(c), R2 Major 1
request 6).

Identified set for C above the largest bracketed budgets (about 1e21 FLOP):
    sigma*(C) in [ sigma_lin(C), sigma_top ],
  sigma_top  = the budget-level estimates at the largest budgets of the two designs whose bracketed budgets reach 1e21
               (Chinchilla, Llama 3; budgets >= 6e20; inverse-variance mean 0.596, anchored at log10 C = 21);
  sigma_lin(C) = sigma_top + beta (log10 C - 21), truncated to (0, 1): linear continuation of the within-design drift.
Maintained assumptions: sigma* is non-increasing in C beyond the design (upper bound) and falls no faster than the
drift measured where it is observed (lower bound). If the drift is not real (Marin shows none up to 3e20), the
relevant value is the design average (0.70); the drift-agnostic union [sigma_lin(C), 0.70] is reported as well.

Wedges (module ra2_wedge's clean sample, reference zero points M*_ref(C) held fixed; ra2 memo H11):
    ln w = k(C) ln(M / M*_ref(C)),  k = 1/sigma* - 1,
which, for a technology whose sigma* varies with C, uses the local slope identity d ln w / d ln M |_(C, M*) =
1/sigma*(C) - 1 (ra1 H1 corollary) and the family's linearity along the isocost (conservative given Farseer's
convexity). For w > 1, ln w is increasing in k: if sigma* falls with compute, the reference level (sigma* = 0.70) is
a LOWER bound on the wedge magnitude, not a midpoint.
Decision units (revision_plan_v3 Sec. 1): one family-level W_f = [sum_j omega_j / w_j]^(-1) per common-D family
(omega_j = training-compute shares; ra2 H4) plus every member of a size-specific-D family and every singleton.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import rb1common as cm

LOG10_TOP = 21.0
SHOWCASE = ["OLMo-2-1124-7B", "SmolLM2-1.7B", "Meta-Llama-3-8B", "Qwen3-0.6B-Base", "Llama-3.1-405B"]


def load_clean():
    d = pd.read_csv(cm.RA2_MODELS)
    c = d[d["clean"] == True].copy()  # noqa: E712
    c["lnMM"] = np.log(c["M_over_Mstar"])
    c["log10C"] = np.log10(c["Cmp"])
    k_ref = float(np.median(np.log(c["w_chin_q"]) / c["lnMM"]))
    return c.reset_index(drop=True), k_ref


def sigma_lin(log10C, sig_top, slope, top=LOG10_TOP, floor=0.01):
    s = sig_top + slope * (np.asarray(log10C, float) - top)
    s = np.where(np.asarray(log10C) <= top, sig_top, s)
    return np.clip(s, floor, 0.999)


def pi_table(sig_top, slopes, Cs=(1e21, 3e21, 1e22, 1e23, 1e24, 1e25, 1e26), sig_ref=0.70, sig_top_ci=None):
    """sigma*(C) bounds and the implied wedge-scale bounds at frontier compute. sig_top_ci (review addition): HKSJ
    interval of sigma_top; the upper end is the conservative upper bound of the set once sampling error is allowed."""
    rows = []
    for C in Cs:
        lc = np.log10(C)
        r = dict(C=C, log10C=lc, sigma_upper=sig_top, k_lower=cm.sigma_to_k(sig_top), sigma_ref=sig_ref,
                 k_ref=cm.sigma_to_k(sig_ref))
        if sig_top_ci is not None:
            r["sigma_upper_ci95_hi"] = float(sig_top_ci[1])
            r["k_lower_ci95"] = float(cm.sigma_to_k(sig_top_ci[1]))
        for lab, b in slopes.items():
            s = float(sigma_lin(lc, sig_top, b))
            r[f"sigma_lower|{lab}"] = s
            r[f"k_upper|{lab}"] = float(cm.sigma_to_k(s))
        rows.append(r)
    return pd.DataFrame(rows)


def family_W(c, w):
    """Family-level W_f for common-D families (ra2 wedges.family_level) under wedges w (aligned with c)."""
    x = c.assign(w_=w)
    rows = []
    for gen, g in x[x.fam_class == "common-D"].groupby("gen"):
        om = g["Cmp"] / g["Cmp"].sum()
        W = 1.0 / np.sum(om / g["w_"])
        rows.append(dict(gen=gen, n=len(g), W_f=W, s_f=(W - 1) / W))
    return pd.DataFrame(rows)


def summarize(c, lnw, label, extra=None):
    w = np.exp(lnw)
    s = cm.share(w)
    F = family_W(c, w)
    mem = c.fam_class.values != "common-D"
    du = np.r_[F.s_f.values, s[mem]]
    r = dict(scenario=label, n=len(c), median_w=float(np.median(w)), median_s=float(np.median(s)),
             share_w_gt1=float(np.mean(w > 1)), n_decision_units=len(du), median_s_decision=float(np.median(du)),
             median_s_family_level=float(np.median(F.s_f)), median_s_members_sizespecific_singletons=float(np.median(s[mem])))
    for m in SHOWCASE:
        ix = np.where(c.model.values == m)[0]
        if len(ix):
            r[f"w|{m}"] = float(w[ix[0]])
    fl = F.set_index("gen")
    if "Llama-3 herd" in fl.index:
        r["s_f|Llama-3 herd"] = float(fl.loc["Llama-3 herd", "s_f"])
    if extra:
        r.update(extra)
    return r


def wedge_scenarios(c, k_ref, meta_pred, sig_top, slopes, log=cm.log, sig_top_hi=None):
    """Clean-sample and decision-unit medians under alternative curvatures.
    meta_pred: function log10C -> sigma* from the primary meta-regression (population-average line).
    sig_top_hi (review addition): upper end of the HKSJ interval of the top-budget value, so that the 'reference is a
    lower bound' statement is checked against sampling error in sigma_top, not only its point estimate."""
    out = []
    lnMM = c["lnMM"].values
    lc = c["log10C"].values
    out.append(summarize(c, k_ref * lnMM, "Reference (kappa-free Chinchilla, sigma* = 0.701)",
                         dict(sigma_min=0.701, sigma_max=0.701)))
    sgs = [0.70, 0.65] + ([round(float(sig_top_hi), 3)] if sig_top_hi is not None else []) + [round(sig_top, 3), 0.60, 0.55]
    for sg in sgs:
        out.append(summarize(c, cm.sigma_to_k(sg) * lnMM, f"Constant sigma* = {sg}",
                             dict(sigma_min=sg, sigma_max=sg)))
    # compute-dependent curvature
    sp = np.clip(meta_pred(lc), 0.01, 0.999)
    out.append(summarize(c, cm.sigma_to_k(sp) * lnMM, "sigma*(C_i): primary meta-regression line (linear drift, pooled)",
                         dict(sigma_min=float(sp.min()), sigma_max=float(sp.max()), sigma_median=float(np.median(sp)))))
    for lab, b in slopes.items():
        sl_ = sigma_lin(lc, sig_top, b)
        out.append(summarize(c, cm.sigma_to_k(sl_) * lnMM, f"sigma*(C_i) = PI lower bound, top-budget value + drift ({lab})",
                             dict(sigma_min=float(sl_.min()), sigma_max=float(sl_.max()), sigma_median=float(np.median(sl_)))))
    S = pd.DataFrame(out)
    log("  wedge scenarios: " + "; ".join(f"{r.scenario[:40]}: med s {r.median_s:.3f}, dec {r.median_s_decision:.3f}"
                                          for r in S.itertuples()))
    return S


def per_model(c, k_ref, sig_top, slope_primary, meta_pred):
    lc = c["log10C"].values
    lnMM = c["lnMM"].values
    s_lin = sigma_lin(lc, sig_top, slope_primary)
    sp = np.clip(meta_pred(lc), 0.01, 0.999)
    df = c[["uid", "model", "dev", "gen", "fam_class", "year", "Cmp", "M", "Mstar_ref", "M_over_Mstar"]].copy()
    df["w_ref"] = np.exp(k_ref * lnMM)
    df["w_sig060"] = np.exp(cm.sigma_to_k(0.60) * lnMM)
    df["sigma_meta_line"] = sp
    df["w_meta_line"] = np.exp(cm.sigma_to_k(sp) * lnMM)
    df["sigma_pi_lower"] = s_lin
    df["w_pi_upper"] = np.exp(cm.sigma_to_k(s_lin) * lnMM)
    for col in ("w_ref", "w_sig060", "w_meta_line", "w_pi_upper"):
        df[col.replace("w_", "s_")] = cm.share(df[col])
    return df


def magnitude_bounds(c, sig_top, slopes, log=cm.log, sig_top_hi=None):
    """Prop. 5(iv) magnitude bounds: ra2's per-model bounds on ln M - ln M*(C) under PI-1..PI-4 (dlo, dhi), with the
    curvature range [k_L, k_U] widened. ra2 used k in [0.40, 0.52] (sigma* in [0.66, 0.71], in-design values)."""
    P = pd.read_csv(cm.RA2_PI_BOUNDS)
    P = P.merge(c[["uid", "log10C"]], on="uid", how="left")
    rows = []

    def bounds(dlo, dhi, kL, kU):
        lnw_lo = np.where(dlo > 0, kL, kU) * dlo
        lnw_hi = np.where(dhi > 0, kU, kL) * dhi
        return cm.share(np.exp(lnw_lo)), cm.share(np.exp(lnw_hi)), np.exp(lnw_lo), np.exp(lnw_hi)

    T = pd.read_csv(cm.RA2_TECHS)
    core = T[T.key.isin(["chin_q", "farseer_q", "meta_mf", "marin_nemotron_mf", "marin_dclm_mf", "marin_comma_mf"])]
    kref_lo, kref_hi = float(core.S2.min()), float(core.S2.max())     # ra2's S2_core = [0.403, 0.516]
    for st, g in P.groupby("set", sort=False):
        dlo, dhi, lc = g.dlo.values, g.dhi.values, g.log10C.values
        cases = [("ra2: k in [0.40, 0.52] (in-design sigma* 0.66-0.71)", np.full(len(g), kref_lo), np.full(len(g), kref_hi)),
                 ("k in [0.40, k(0.60)]: adds the top-budget value", np.full(len(g), kref_lo), np.full(len(g), cm.sigma_to_k(0.60)))]
        for lab, b in slopes.items():
            kU = cm.sigma_to_k(sigma_lin(lc, sig_top, b))
            cases.append((f"drift-agnostic: k in [0.40, k(sigma_lin(C_i))], {lab}", np.full(len(g), kref_lo), kU))
            cases.append((f"drift maintained: k in [k(sigma_top), k(sigma_lin(C_i))], {lab}",
                          np.full(len(g), cm.sigma_to_k(sig_top)), kU))
            if sig_top_hi is not None:      # review addition: sigma_top at the upper end of its HKSJ interval
                cases.append((f"drift maintained, sigma_top at its 95% upper limit: k in [k(sigma_top_hi), "
                              f"k(sigma_lin(C_i))], {lab}", np.full(len(g), cm.sigma_to_k(sig_top_hi)), kU))
        for lab, kL, kU in cases:
            slo, shi, wlo, whi = bounds(dlo, dhi, kL, kU)
            rows.append(dict(pi_set=st, curvature=lab, n=len(g), share_w_gt1_identified=float(np.mean(dlo > 0)),
                             median_s_lo=float(np.median(slo)), median_s_hi=float(np.median(shi)),
                             median_w_lo=float(np.median(wlo)), median_w_hi=float(np.median(whi))))
    M = pd.DataFrame(rows)
    chk = pd.read_csv(cm.RA2_PI_SUMMARY)
    m0 = M[M.curvature.str.startswith("ra2:")].set_index("pi_set")
    for _, r in chk.iterrows():
        if r["set"] in m0.index:
            diff = max(abs(m0.loc[r["set"], "median_s_lo"] - r["median_s_lo"]), abs(m0.loc[r["set"], "median_s_hi"] - r["median_s_hi"]))
            log(f"  reproduce ra2 PI bounds [{r['set'][:5]}]: |diff| = {diff:.2e}")
    return M
