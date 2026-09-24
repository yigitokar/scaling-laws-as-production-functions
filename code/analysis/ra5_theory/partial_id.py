"""Illustration of Proposition A11 (partial identification of M*(C) and of the sign and size of the wedge beyond the design).

Anchors: the IsoFLOP minima of the two lab designs with the largest budgets (Chinchilla, Epoch extraction, 9 budgets up to
2.9e21 FLOP; Meta Llama 3, digitized, 10 budgets up to 1e22 FLOP), as computed by module m1 (output/tables/
m1_chinchilla_a2_minima.csv and m1_chinchilla_labs_a2_minima.csv). The path ln M*(C) is fitted to the minima by OLS,
ln M*_j = mu + e (ln C_j - ln C_J), centered at the largest budget C_J; the design is fixed (budgets chosen by the
experimenter), so inference is design-conditional: a wild bootstrap-t with HC2-rescaled residuals, HC2 standard errors
and Webb six-point weights (B = 9,999; equal-tailed). Review fix: the first version used raw-residual percentile intervals,
which cover only 76-83 percent at these 9-10-point designs; the bootstrap-t covers about 90 percent (see
output/memos/ra5_theory_review.md). The illustration is superseded for empirical purposes by the harmonized
partial-identification analysis of module ra2_wedge (output/tables/ra2_wedge_pi_*.csv).

Models: the verified Sample B (sample == "B", core == True; n = 173) from output/tables/m3_wedge_models.csv, in the paper's
conventions (total N; each model's own tokens). Units are NOT harmonized across tokenizers; the illustration is therefore
about the logic of the bounds, not a measurement.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ra5common import PROC, ROOT, TABLES

B_BOOT = 9999
SEED = 2026
A_RANGE = (0.37, 0.57)                  # cross-sweep range of allocation exponents (paper, Section III)
K_WIDE = (1 / 0.8 - 1, 1 / 0.5 - 1)     # 1/sigma*-1 for sigma* in [0.5, 0.8]
K_NARROW = (1 / 0.75 - 1, 1 / 0.6 - 1)  # sigma* in [0.6, 0.75]


def load_anchors():
    t = os.path.join(ROOT, "output", "tables")
    a = pd.read_csv(os.path.join(t, "m1_chinchilla_a2_minima.csv"))
    b = pd.read_csv(os.path.join(t, "m1_chinchilla_labs_a2_minima.csv"))
    out = {}
    for name, df, key in [("Chinchilla (Epoch extraction)", a, "chinchilla_n240"), ("Meta Llama 3 (digitized)", b, "llama_3")]:
        g = df[df["dataset"] == key].sort_values("C")
        out[name] = pd.DataFrame(dict(lnC=np.log(g["C"].values), lnM=np.log(g["Dstar"].values / g["Nstar"].values)))
    return out


def webb_weights(rng, size):
    vals = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
    return rng.choice(vals, size=size)


def fit_path(g, rng):
    """OLS of ln M*_j on (ln C_j - ln C_J) and a design-conditional wild bootstrap-t for (mu, e).

    Review fix (ra5_theory_review.md): the first version used a percentile interval from a wild bootstrap of RAW residuals.
    With 9-10 budgets and the object of interest at the high-leverage endpoint (h ~ 0.40), raw residuals understate the
    error variance by (1-h) and percentile intervals ignore small-sample thickening; in simulations at these exact designs
    that interval covered 76-83 percent (nominal 95). We now use HC2-rescaled residuals u_i = e_i/sqrt(1-h_ii), Webb
    six-point weights, and the equal-tailed bootstrap-t with HC2 standard errors recomputed in every draw (simulated
    coverage about 90 percent at these designs, homoskedastic or heteroskedastic; still somewhat liberal)."""
    cJ = g["lnC"].max()
    X = np.column_stack([np.ones(len(g)), g["lnC"].values - cJ])
    y = g["lnM"].values
    XtXi = np.linalg.inv(X.T @ X)
    Rw = XtXi @ X.T                                                # 2 x n: beta = Rw @ y
    beta = Rw @ y
    res = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)                       # leverages
    u = res / np.sqrt(1 - h)                                       # HC2-rescaled residuals

    def hc2_se(r):                                                 # r: (..., n) residuals -> (..., 2) HC2 standard errors
        return np.sqrt(np.einsum("kn,...n->...k", Rw ** 2, (r / np.sqrt(1 - h)) ** 2))

    V = webb_weights(rng, (B_BOOT, len(y)))
    Ystar = X @ beta + V * u
    draws = (Rw @ Ystar.T).T                                       # B x 2 (mu*, e*)
    se0 = hc2_se(res)
    se_b = hc2_se(Ystar - draws @ X.T)
    tstar = (draws - beta) / se_b
    q_lo, q_hi = np.quantile(tstar, [0.025, 0.975], axis=0)
    ci = np.column_stack([beta - q_hi * se0, beta - q_lo * se0])   # equal-tailed bootstrap-t
    return dict(cJ=cJ, mu=beta[0], e=beta[1], draws=draws, n=len(y), resid_sd=float(np.std(res, ddof=2)),
                mu_ci=(float(ci[0, 0]), float(ci[0, 1])), e_ci=(float(ci[1, 0]), float(ci[1, 1])), se_hc2=se0,
                h_top=float(h[np.argmax(X[:, 1])]))


def bounds_at(fit, c, e_rng, anchor_ci=True):
    """Identified set for ln M*(C) at log compute c > c_J under slope bounds e_rng and the anchor's 95% CI."""
    mu_lo, mu_hi = fit["mu_ci"] if anchor_ci else (fit["mu"], fit["mu"])
    dc = c - fit["cJ"]
    return mu_lo + e_rng[0] * dc, mu_hi + e_rng[1] * dc


def lnw_bounds(x, lo, hi, k):
    """Bounds on ln w = k (x - x*) for x* in [lo, hi] and k in [kL, kU] (Proposition A11(iv)). Expenditure shares
    s = 1 - 1/w reported by run() are truncated at 0, the lower limit under Proposition A8(iii)(a) (T >= 0)."""
    kL, kU = k
    if x > hi:
        return kL * (x - hi), kU * (x - lo)
    if x < lo:
        return kU * (x - hi), kL * (x - lo)
    return kU * (x - hi), kU * (x - lo)


def run():
    rng = np.random.default_rng(SEED)
    anchors = load_anchors()
    mdl = pd.read_csv(os.path.join(TABLES, "m3_wedge_models.csv"))
    b = mdl[(mdl["sample"] == "B") & (mdl["core"] == True)].copy()  # noqa: E712
    b["lnC"], b["lnM"] = np.log(b["Cmp"]), np.log(b["M"])
    rows_b, rows_s, rows_m, fits = [], [], [], {}
    for aname, g in anchors.items():
        fit = fit_path(g, rng)
        fits[aname] = fit
        e_ci = fit["e_ci"]
        assumptions = [
            ("Normal inputs only", (-1.0, 1.0), True),
            ("Slope in cross-sweep range", (1 - 2 * A_RANGE[1], 1 - 2 * A_RANGE[0]), True),
            ("Cross-sweep range + M* nondecreasing", (0.0, 1 - 2 * A_RANGE[0]), True),
            ("Anchor's own slope persists (95% CI)", e_ci, True),
            ("Parametric extrapolation (point)", (fit["e"], fit["e"]), False),
        ]
        for lab, e_rng, ci in assumptions:
            for C in (1e23, 1e24, 1e25):
                lo, hi = bounds_at(fit, np.log(C), e_rng, ci)
                rows_b.append(dict(anchor=aname, assumption=lab, e_lo=e_rng[0], e_hi=e_rng[1], C=C, Mstar_lo=np.exp(lo), Mstar_hi=np.exp(hi)))
            beyond = b[b["lnC"] > fit["cJ"]]
            cls, wlo_w, whi_w, wlo_n, whi_n = [], [], [], [], []
            for r in beyond.itertuples():
                lo, hi = bounds_at(fit, r.lnC, e_rng, ci)
                cls.append(1 if r.lnM > hi else (-1 if r.lnM < lo else 0))
                a1, a2 = lnw_bounds(r.lnM, lo, hi, K_WIDE)
                n1, n2 = lnw_bounds(r.lnM, lo, hi, K_NARROW)
                wlo_w.append(np.exp(a1)); whi_w.append(np.exp(a2)); wlo_n.append(np.exp(n1)); whi_n.append(np.exp(n2))
            cls = np.array(cls)
            wl, wh = np.array(wlo_w), np.array(whi_w)
            rows_s.append(dict(anchor=aname, C_anchor=float(np.exp(fit["cJ"])), assumption=lab, e_lo=e_rng[0], e_hi=e_rng[1],
                               n_beyond=len(beyond), share_over_identified=np.mean(cls == 1), share_under_identified=np.mean(cls == -1),
                               share_not_identified=np.mean(cls == 0),
                               median_w_lo_sig50_80=np.median(wl), median_w_hi_sig50_80=np.median(wh),
                               median_s_lo_sig50_80=np.median(np.maximum(0.0, 1 - 1 / wl)), median_s_hi_sig50_80=np.median(np.maximum(0.0, 1 - 1 / wh)),
                               median_w_lo_sig60_75=np.median(wlo_n), median_w_hi_sig60_75=np.median(whi_n)))
            for key in ("Meta-Llama-3-8B", "Qwen3-0.6B-Base", "Llama-3.1-405B", "Llama-2-70b-hf"):
                r = b[b["model"] == key]
                if r.empty:
                    continue
                r = r.iloc[0]
                lo, hi = bounds_at(fit, r["lnC"], e_rng, ci)
                a1, a2 = lnw_bounds(r["lnM"], lo, hi, K_WIDE)
                rows_m.append(dict(anchor=aname, assumption=lab, model=key, C=r["Cmp"], M=r["M"], Mstar_lo=np.exp(lo), Mstar_hi=np.exp(hi),
                                   sign=("w>1" if r["lnM"] > hi else ("w<1" if r["lnM"] < lo else "not identified")),
                                   w_lo=np.exp(a1), w_hi=np.exp(a2), s_lo=max(0.0, 1 - np.exp(-a1)), s_hi=max(0.0, 1 - np.exp(-a2))))
    anc = pd.DataFrame([dict(anchor=k, n_budgets=f["n"], C_top=float(np.exp(f["cJ"])), Mstar_top=float(np.exp(f["mu"])),
                             Mstar_top_lo=float(np.exp(f["mu_ci"][0])), Mstar_top_hi=float(np.exp(f["mu_ci"][1])), e_hat=f["e"],
                             e_lo=f["e_ci"][0], e_hi=f["e_ci"][1], a_hat=(1 - f["e"]) / 2, resid_sd=f["resid_sd"],
                             se_hc2_lnMstar=float(f["se_hc2"][0]), leverage_top=f["h_top"],
                             ci_method="wild bootstrap-t, HC2 residuals and SEs, Webb weights", B=B_BOOT) for k, f in fits.items()])
    out = dict(anchors=anc, bounds=pd.DataFrame(rows_b), sign=pd.DataFrame(rows_s), models=pd.DataFrame(rows_m), fits=fits,
               sampleB=b[["model", "lnC", "lnM"]], anchor_points=anchors)
    anc.to_csv(os.path.join(PROC, "pi_anchor_paths.csv"), index=False)
    return out
