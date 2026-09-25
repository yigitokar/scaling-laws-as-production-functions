"""idset.py -- the identified set for x*(c) = ln M*(C) at each model's compute, the sign it identifies, and shares.

Set (Proposition A6(vii)-(viii), fix list S1(4)): the union over the anchored paths j of their bands, each widened by
ln tau on both sides. Each path carries its own parameter convention; a model is placed on a path's axes in that
convention (M = D/N and c = ln 6ND with N the model's total, FLOP-effective or DeepSeek count). For each model,
dlo = min_j [ln M - sup X*_j(c)] and dhi = max_j [ln M - inf X*_j(c)]; w > 1 is identified at tau iff dlo > ln tau
and w < 1 iff dhi < -ln tau. A decision unit is identified iff every member is (rb2's rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import rb5common as K
import wedges as WG  # ra2_wedge (read-only): parameter conventions of released models


def model_axes(B, conv):
    if conv == "total":
        N = B["N"].values.astype(float)
    elif conv == "nf":
        N = B["N_F"].values.astype(float)
    elif conv in ("ds", "nonemb", "olmo"):
        N = np.asarray(WG.n_used(B, conv), float)
    else:
        raise ValueError(conv)
    D = B["D"].values.astype(float)
    return np.log(D / N), np.log(6 * N * D)


def c_extent(B, convs=("total", "nf", "ds", "nonemb", "olmo")):
    """Smallest and largest ln C of the models over the conventions used (the bands' domain)."""
    cs = np.concatenate([model_axes(B, cv)[1] for cv in convs])
    return float(cs.min()) - 0.01, float(cs.max()) + 0.01


def bounds(B, paths, label):
    """Per-model bounds on ln M - x*(c) over the union of the paths' bands (tau = 1)."""
    n = len(B)
    dlo = np.full(n, np.inf)
    dhi = np.full(n, -np.inf)
    arg_lo = np.array([""] * n, dtype=object)
    arg_hi = np.array([""] * n, dtype=object)
    for p in paths:
        lnM, c = model_axes(B, p.conv)
        lo, hi = p.band(c)
        a, b = lnM - hi, lnM - lo
        m = a < dlo
        dlo = np.where(m, a, dlo)
        arg_lo = np.where(m, p.name, arg_lo)
        m2 = b > dhi
        dhi = np.where(m2, b, dhi)
        arg_hi = np.where(m2, p.name, arg_hi)
    out = B[["uid", "model", "dev", "year", "gen", "N", "D", "M", "Cmp"]].copy()
    out["set"] = label
    out["dlo"], out["dhi"] = dlo, dhi
    out["Mstar_hi_eff"] = out["M"] / np.exp(dlo)      # upper end of the set, expressed in the model's total M
    out["Mstar_lo_eff"] = out["M"] / np.exp(dhi)
    out["binding_upper"] = arg_lo
    out["binding_lower"] = arg_hi
    out["tau_break"] = np.exp(np.maximum(dlo, 0.0))  # w > 1 identified for tau < tau_break (1 if not at tau = 1)
    out["sign_tau1"] = np.where(dlo > 0, "w>1", np.where(dhi < 0, "w<1", "not identified"))
    return out


def _share(ind, wts):
    return float(np.sum(ind * wts) / np.sum(wts)) if np.sum(wts) > 0 else np.nan


def unit_frame(PB, U):
    X = PB.merge(U[["uid", "unit"]], on="uid", how="inner")
    G = X.groupby("unit", sort=False).agg(dlo=("dlo", "min"), dhi=("dhi", "max"), Cmp=("Cmp", "sum"),
                                         year=("year", "min"), n_members=("uid", "count")).reset_index()
    return G


def shares(PB, B, units, label, taus=K.TAUS):
    """Shares identified (w > 1 and w < 1) by tau: models, models below 15B, each decision-unit definition; unweighted
    and compute-weighted; all years and by release year."""
    rows = []
    small = PB["uid"].isin(B.loc[B["N"] < K.N_SMALL, "uid"])
    groups = [("models", PB), ("models below 15B", PB[small.values])]
    for uname, U in units.items():
        groups.append((f"decisions ({uname})", unit_frame(PB, U)))
    for tau in taus:
        lt = np.log(tau)
        for lev, G in groups:
            for yr, g in [("all", G)] + [(str(y), gg) for y, gg in G.groupby("year")]:
                gt = (g["dlo"] > lt).values.astype(float)
                lt1 = (g["dhi"] < -lt).values.astype(float)
                rows.append(dict(set=label, tau=tau, level=lev, year=yr, n=len(g), share_gt1=gt.mean(),
                                 share_gt1_cw=_share(gt, g["Cmp"].values), share_lt1=lt1.mean(),
                                 share_lt1_cw=_share(lt1, g["Cmp"].values)))
    return pd.DataFrame(rows)


def frontier(PB, B, units, label, tau_grid):
    """Share identified (w > 1) as a function of tau on a grid (for the breakdown figure)."""
    small = PB["uid"].isin(B.loc[B["N"] < K.N_SMALL, "uid"]).values
    rows = []
    series = {"all models": (PB["dlo"].values, np.ones(len(PB))),
              "models below 15B": (PB["dlo"].values[small], np.ones(small.sum())),
              "training compute (models)": (PB["dlo"].values, PB["Cmp"].values)}
    for uname, U in units.items():
        G = unit_frame(PB, U)
        series[f"decisions ({uname})"] = (G["dlo"].values, np.ones(len(G)))
        series[f"training compute (decisions, {uname})"] = (G["dlo"].values, G["Cmp"].values)
    for s, (d, w) in series.items():
        for t in tau_grid:
            rows.append(dict(set=label, series=s, tau=float(t), share=_share((d > np.log(t)).astype(float), w)))
    return pd.DataFrame(rows)


def breakdown(PB, B, units, label):
    """Smallest tau at which each share falls below one half (exact: the share is a step function of tau that drops at
    tau_i = exp(dlo_i)). Returns 1.0 if the share is below one half already at tau = 1."""
    small = PB["uid"].isin(B.loc[B["N"] < K.N_SMALL, "uid"]).values
    series = {"all models": (PB["dlo"].values, np.ones(len(PB))),
              "models below 15B": (PB["dlo"].values[small], np.ones(small.sum())),
              "training compute (models)": (PB["dlo"].values, PB["Cmp"].values)}
    for uname, U in units.items():
        G = unit_frame(PB, U)
        series[f"decisions ({uname})"] = (G["dlo"].values, np.ones(len(G)))
        series[f"training compute (decisions, {uname})"] = (G["dlo"].values, G["Cmp"].values)
    rows = []
    for s, (d, w) in series.items():
        thr = np.exp(np.maximum(d, 0.0))
        cands = np.unique(np.concatenate([[1.0], thr]))
        tb = np.nan
        for t in cands:
            if _share((d > np.log(t)).astype(float), w) < 0.5:
                tb = float(t)
                break
        rows.append(dict(set=label, series=s, share_at_tau1=_share((d > 0).astype(float), w), tau_breakdown=tb,
                         median_tau_break=float(np.median(thr))))
    return pd.DataFrame(rows)


def mstar_sets(paths, Cs=(1e22, 1e23, 1e24, 1e25), label=""):
    """Union of the paths' bands at compute C, by convention (exp: the set for M*(C) in that convention)."""
    rows = []
    for C in Cs:
        c = np.log(C)
        for conv in sorted(set(p.conv for p in paths)):
            P = [p for p in paths if p.conv == conv]
            lo = min(float(p.band(c)[0][0]) for p in P)
            hi = max(float(p.band(c)[1][0]) for p in P)
            rows.append(dict(set=label, conv=conv, C=C, Mstar_lo=float(np.exp(lo)), Mstar_hi=float(np.exp(hi)),
                             paths="; ".join(p.name for p in P)))
        for p in paths:
            lo, hi = p.band(c)
            rows.append(dict(set=label, conv=p.conv, C=C, Mstar_lo=float(np.exp(lo[0])), Mstar_hi=float(np.exp(hi[0])),
                             paths=p.name + " (alone)"))
    return pd.DataFrame(rows)
