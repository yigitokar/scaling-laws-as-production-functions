"""signid.py -- sign identification beyond the designs (Prop. 5 / PI-1; R1 New 3; R2 Major 4.3; R4 R2-M3).

Anchors (PI-1, harmonized, total-N units): each IsoFLOP design's path of per-budget minima (ra1's reviewed estimator,
bracketed budgets only) evaluated at its LARGEST BRACKETED budget, with a wild bootstrap-t interval that resamples the
between-budget residuals (paths.Path). Designs need at least 6 bracketed budgets (4 residual degrees of freedom):
Chinchilla (9; 3e21), Llama 3 (8; 1e21 -- R1 minor 7: not the unbracketed 1e22), Marin DCLM (7; 3e20) and
Nemotron-CC (7; 3e20). Marin Comma (5 bracketed budgets) is a sensitivity row. DeepSeek's published law (point) at its
largest budget, in its own units. Path elasticity e in [min, max] over the anchors' own path slopes and the published
laws (DeepSeek's three data-quality exponents, MiniCPM).
Tilt allowance tau (R1 New 3): w > 1 is identified iff ln M exceeds the upper bound of the identified set for
ln M*(C) by ln tau: tau = 1 (none), 1.3 (tokenizer differences), 1.84 (the DataDecide tilt of 0.26 under the reference
exponents), 3.4 (DataDecide's own M* factor), 4.4 (= 3.4 x 1.3, both).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import rb2common as C
import paths as P
import analysis_ra2 as AN  # ra2
import techs as TT  # ra2

TAUS = [(1.0, "none"), (1.3, "tokenizer"), (1.84, "DataDecide tilt (0.26 x reference)"), (3.4, "DataDecide M* factor"),
        (4.4, "both (3.4 x 1.3)")]
MIN_BUDGETS = 6


def anchors(X, B=9999, include_comma=False, meta10=False):
    rows, fits = [], {}
    designs = [("Chinchilla", "chin_iso", ""), ("Llama 3", "meta", "Meta"), ("Marin, DCLM", "marin_dclm", "Marin"),
               ("Marin, Nemotron-CC", "marin_nemotron", "Marin")]
    if include_comma:
        designs.append(("Marin, Comma", "marin_comma", "Marin"))
    for i, (des, key, lab) in enumerate(designs):
        g = P.ra1_minima(des)
        if key == "meta" and meta10:
            pb = X["mf_pb"][X["mf_pb"]["design"] == "meta"]
            g = pd.DataFrame(dict(C=pb["budget"].values, lnC=np.log(pb["budget"].values),
                                  lnM=np.log(pb["budget"].values / 6) - 2 * pb["lnNstar"].values))
        if len(g) < MIN_BUDGETS and not include_comma:
            continue
        p = P.Path(g, B=B, seed=C.SEED + 100 + i, label=des)
        lo, hi = p.interval(p.c0)
        fits[key] = p
        rows.append(dict(anchor=key, design=des, kind="iso", conv="total", n_budgets=p.n, C0=float(np.exp(p.c0)),
                         lnMs=float(p.beta[0]), lnMs_lo=float(lo[0]), lnMs_hi=float(hi[0]), Mstar=float(np.exp(p.beta[0])),
                         Mstar_lo=float(np.exp(lo[0])), Mstar_hi=float(np.exp(hi[0])), e_path=float(p.beta[1]),
                         e_lo=p.slope_interval()[0], e_hi=p.slope_interval()[1], a=(1 - p.beta[1]) / 2,
                         resid_sd=p.resid_sd, lab=lab, method="wild bootstrap-t (HC2, Webb), between-budget residuals"))
    t = X["T"]["deepseek"]
    pt = C.ln_mstar(3e20, t.alpha, t.beta, t.lnG)
    rows.append(dict(anchor="deepseek", design="DeepSeek LLM published law", kind="iso", conv="ds", n_budgets=np.nan,
                     C0=3e20, lnMs=float(pt), lnMs_lo=float(pt), lnMs_hi=float(pt), Mstar=float(np.exp(pt)),
                     Mstar_lo=float(np.exp(pt)), Mstar_hi=float(np.exp(pt)), e_path=1 - 2 * t.a, e_lo=np.nan, e_hi=np.nan,
                     a=t.a, resid_sd=np.nan, lab="DeepSeek", method="published point (no interval)"))
    return pd.DataFrame(rows), fits


def e_range(A, X):
    e_paths = list(A.loc[A["anchor"] != "deepseek", "e_path"])
    e_pub = [1 - 2 * a for a in TT.DEEPSEEK["table4_a"].values()] + [1 - 2 * X["T"]["minicpm"].a]
    e = e_paths + e_pub
    return (float(min(e)), float(max(e)))


def pi_models(Bc, A, e, S2_range):
    """Per-model bounds (dlo, dhi on ln M - ln M*(C)) with ra2's routine (union over anchors, each in its own units)."""
    mask = pd.Series(True, index=Bc.index)
    return AN.pi_bounds(Bc, mask, A, e, S2_range, {"PI": dict(kinds=["iso"], e=e)})


def sign_with_tau(pb, tau):
    lt = np.log(tau)
    return np.where(pb["dlo"] > lt, "w>1", np.where(pb["dhi"] < -lt, "w<1", "ambiguous"))


def shares(pb, Bc, U, label):
    """Share identified (w > 1) by tau: unweighted, compute-weighted, by year; decision units (a family is identified
    iff every member is)."""
    X = pb.merge(Bc[["uid", "year", "Cmp", "gen"]], on="uid").merge(U[["uid", "unit"]], on="uid", how="left")
    rows = []
    for tau, why in TAUS:
        X["id"] = sign_with_tau(X, tau) == "w>1"
        base = dict(set=label, tau=tau, rationale=why)
        rows.append(dict(**base, level="model", year="all", n=len(X), share_identified=X["id"].mean(),
                         share_identified_cw=float((X["id"] * X["Cmp"]).sum() / X["Cmp"].sum())))
        for y, g in X.groupby("year"):
            rows.append(dict(**base, level="model", year=str(y), n=len(g), share_identified=g["id"].mean(),
                             share_identified_cw=float((g["id"] * g["Cmp"]).sum() / g["Cmp"].sum())))
        Ug = X.groupby("unit").agg(id=("id", "min"), Cmp=("Cmp", "sum"), year=("year", "min")).reset_index()
        rows.append(dict(**base, level="decision", year="all", n=len(Ug), share_identified=Ug["id"].mean(),
                         share_identified_cw=float((Ug["id"] * Ug["Cmp"]).sum() / Ug["Cmp"].sum())))
        for y, g in Ug.groupby("year"):
            rows.append(dict(**base, level="decision", year=str(y), n=len(g), share_identified=g["id"].mean(),
                             share_identified_cw=float((g["id"] * g["Cmp"]).sum() / g["Cmp"].sum())))
    return pd.DataFrame(rows)


def magnitude(pb_models, k_ranges, tau=1.0):
    """Median of the per-model bounds on s for curvature ranges k = 1/sigma* - 1 in [kL, kU] (Prop. 5(iv)); with a tilt
    allowance tau the identified set for ln M*(C) is widened by ln tau on both sides."""
    rows = []
    lt = np.log(tau)
    for lab, (kL, kU) in k_ranges.items():
        lo, hi = [], []
        for _, r in pb_models.iterrows():
            dlo, dhi = r["dlo"] - lt, r["dhi"] + lt
            lwl = (kL if dlo > 0 else kU) * dlo
            lwh = (kU if dhi > 0 else kL) * dhi
            lo.append(C.share(np.exp(lwl)))
            hi.append(C.share(np.exp(lwh)))
        rows.append(dict(curvature=lab, tau=tau, k_lo=kL, k_hi=kU, sigma_lo=1 / (1 + kU), sigma_hi=1 / (1 + kL),
                         median_s_lower_bound=float(np.median(lo)), median_s_upper_bound=float(np.median(hi))))
    return pd.DataFrame(rows)


def mstar_grid(A, e, Cs=(1e22, 1e23, 1e24, 1e25)):
    Aa = A[A["conv"] == "total"]
    rows = []
    for Cc in Cs:
        lo = min(a["lnMs_lo"] + (e[0] if Cc > a["C0"] else e[1]) * np.log(Cc / a["C0"]) for _, a in Aa.iterrows())
        hi = max(a["lnMs_hi"] + (e[1] if Cc > a["C0"] else e[0]) * np.log(Cc / a["C0"]) for _, a in Aa.iterrows())
        rows.append(dict(C=Cc, Mstar_lo=float(np.exp(lo)), Mstar_hi=float(np.exp(hi)), n_anchors=len(Aa)))
    return pd.DataFrame(rows)
