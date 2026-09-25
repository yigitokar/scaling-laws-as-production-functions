"""cluster_ref.py -- the reference technology (Chinchilla, kappa free) under ra1's conservative bootstrap scheme
(R4 R2-M3(d), request 3): IsoFLOP runs clustered by budget and off-profile runs by model-size trunk (9 + 35 = 44
clusters; module ra1_modelfree, chin_inf.py 'cluster_trunk'), so that the headline median s can be reported beside
the design-conditional wild interval used elsewhere in this module.

Added by the independent review of rb2_decisions (2026-09-24). Data and clusters reproduce ra1's construction:
m1_chinchilla/common.py::load_chinchilla (loaded under a private name, read-only) for the budgets and IsoFLOP flags,
and ra1_common.py::load_chinchilla_all's trunk rule (consecutive log10 N within 0.005 dex), copied below. The
kappa-free point estimate is the one ra2_wedge's techs.chin_q_wild computes (m2's fit_q from m1's Chinchilla-form
start); each cluster draw is refitted from that point (m2's fit_q, Huber). Draws: (alpha, beta, lnG), as in ra2.
"""
from __future__ import annotations

import importlib.util
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import rb2common as C


def chin_clusters():
    """The 240-run Chinchilla sample (five highest-loss runs dropped) with ra1's budget-plus-trunk cluster ids."""
    spec = importlib.util.spec_from_file_location("m1_common_rb2", os.path.join(C.ANALYSIS, "m1_chinchilla", "common.py"))
    m1c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m1c)
    df = m1c.load_chinchilla(0)
    off = ~df["iso"].values                                   # ra1_common.load_chinchilla_all: off-profile trunks
    ln = np.log10(df["N"].values)
    order = np.argsort(ln[off])
    g = np.r_[0, np.cumsum(np.diff(ln[off][order]) > 0.005)]
    trunk = np.full(len(df), -1)
    trunk[np.where(off)[0][order]] = g
    df["trunk"] = trunk
    df = df[df["rank_worst"] > 5].reset_index(drop=True)
    df["cluster"] = np.where(df["iso"], "b" + df["cl"].astype(str), "t" + df["trunk"].astype(str))
    return df


def _worker(args):
    import m2_est as me
    seeds, N, D, L, clus, th0 = args
    ug = np.unique(clus)
    groups = [np.where(clus == u)[0] for u in ug]
    out = []
    for s in seeds:
        rng = np.random.default_rng(s)
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[j] for j in pick])
        try:
            th, f = me.fit_q(N[idx], D[idx], L[idx], "huber", init=th0)
            a, b, e, al, be, q = th
            out.append([al, be, (np.log(al) + a - np.log(be) - b) / (al + be)])
        except Exception:  # noqa: BLE001
            out.append([np.nan] * 3)
    return out


def draws(B=399):
    """Returns (point dict, (B', 3) draws of (alpha, beta, lnG) with failed fits dropped, n_clusters, n_fail)."""
    import m2_est as me
    import techs as TT                                      # ra2_wedge (read-only)
    df = chin_clusters()
    N, D, L = df["N"].values, df["D"].values, df["L"].values
    ref = TT._chin_data()
    assert len(ref) == len(df) and np.allclose(np.sort(ref["L"].values), np.sort(L)), "Chinchilla sample mismatch"
    r1 = pd.read_csv(TT.REG_M1).set_index("row_id").loc["chin_n240_huber"]
    th_chin = np.array([np.log(r1.A), np.log(r1.B), np.log(r1.E), r1.alpha, r1.beta])
    th, f = me.fit_q(N, D, L, "huber", th_chin=th_chin)
    a, b, e, al, be, q = th
    point = dict(alpha=al, beta=be, lnG=(np.log(al) + a - np.log(be) - b) / (al + be))
    seeds = [C.SEED + 7000 + i for i in range(B)]
    chunks = [seeds[i::C.N_PROC] for i in range(C.N_PROC)]
    clus = df["cluster"].values
    with ProcessPoolExecutor(C.N_PROC) as ex:
        parts = list(ex.map(_worker, [(c, N, D, L, clus, th) for c in chunks]))
    dr = np.array([r for p in parts for r in p])
    ok = np.isfinite(dr).all(1) & (dr[:, 0] > 0) & (dr[:, 1] > 0)
    return point, dr[ok], int(len(np.unique(clus))), int((~ok).sum())
