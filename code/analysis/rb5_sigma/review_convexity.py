"""review_convexity.py -- reviewer's recomputation of the T1.9 decomposition point estimates (package WP2-review).

Reads rb1's local-wedge evaluation points (data/processed/rb1_sigmaC/stage_extrap.pkl: grid, ln w_local, n_eff, path)
and ra1's Farseer grid and path (output/tables/ra1_modelfree_farseer_{grid,path}.csv); no module code is imported.
Per recipe: points with n_eff at least 8 (rb1) or 15 (ra1) and compute inside the path's range; u = ln M - ln M*_local(C)
(interpolated in ln C); ln w_local = b1 u + b2 u^2 by OLS; convex part = mean of (ln w_local - b1 u) by M bin; Marin
pooled = equal-weight mean of the three corpora's bin means. Also the primary estimate's convex part on M of 256 to 320,
the range to which the leave-corner-out variant (largest-M run of each budget dropped) is confined.
Output: output/tables/rb5_sigma_review_convexity.csv (with the builder's values beside). Single process.
"""
import os
import pickle

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
T = os.path.join(ROOT, "output", "tables")
MB = [0, 16, 64, 256, 1024, 1e9]
LAB = ["M<16", "16-64", "64-256", "256-1,024", ">=1,024"]
MARIN = ["Marin, Comma", "Marin, DCLM", "Marin, Nemotron-CC"]


def stats(M, lnw, neff, lnMs, inside, nmin):
    ok = (neff >= nmin) & np.isfinite(lnw)
    ins = ok & inside & np.isfinite(lnMs)
    u = np.log(M[ins]) - lnMs[ins]
    b1, b2 = np.linalg.lstsq(np.column_stack([u, u * u]), lnw[ins], rcond=None)[0]
    uu = np.full(len(M), np.nan)
    uu[ins] = u
    conv = lnw - b1 * uu
    b = np.digitize(M, MB) - 1
    out = dict(b1=b1, b2=b2, sigma_slope=1 / (1 + b1))
    for j, l in enumerate(LAB):
        s = ins & (b == j)
        out[f"conv|{l}"], out[f"n|{l}"] = (conv[s].mean() if s.sum() else np.nan), int(s.sum())
    s = ins & (M >= 256) & (M <= 320)
    out["conv|256-320"], out["n|256-320"] = (conv[s].mean() if s.sum() else np.nan), int(s.sum())
    return out


def main():
    X = pickle.load(open(os.path.join(ROOT, "data", "processed", "rb1_sigmaC", "stage_extrap.pkl"), "rb"))
    res = {}
    for nm in ["Llama 3"] + MARIN:
        v = X[nm]
        g, st = v["grid"], v["st0"]
        Cb = np.array([b[0] for b in v["budgets"]])
        lp = st["path_lnMstar"]
        ok = np.isfinite(lp)
        lC = np.log(g.C_b.values)
        lMs = np.interp(lC, np.log(Cb[ok]), lp[ok])
        inside = (lC >= np.log(Cb[ok]).min() - 1e-9) & (lC <= np.log(Cb[ok]).max() + 1e-9)
        res[nm] = stats(g.M.values, st["lnw_local"], st["neff"], lMs, inside, 8)
    g = pd.read_csv(os.path.join(T, "ra1_modelfree_farseer_grid.csv"))
    p = pd.read_csv(os.path.join(T, "ra1_modelfree_farseer_path.csv"))
    p = p[(p.conv == "ne") & p.M_star_local.notna()]
    C = 6 * g.N.values * g.D.values
    lC = np.log(C)
    lMs = np.interp(lC, np.log(p.C.values), np.log(p.M_star_local.values))
    res["Farseer"] = stats(g.M.values, g.lnw_local_ne.values, g.neff_ne.values, lMs,
                           (lC >= np.log(p.C.min())) & (lC <= np.log(p.C.max())), 15)
    pooled = {k: float(np.nanmean([res[m][k] for m in MARIN])) for k in res[MARIN[0]] if k.startswith("conv|")}
    pooled.update({k: int(sum(res[m][k] for m in MARIN)) for k in res[MARIN[0]] if k.startswith("n|")})
    res["Marin, three corpora pooled"] = pooled
    dec = pd.read_csv(os.path.join(T, "rb5_sigma_extrap_decomposition.csv"))
    dec = dec[dec.variant == "primary"]
    lin = pd.read_csv(os.path.join(T, "rb5_sigma_extrap_lin.csv"))
    lin = lin[lin.variant == "primary"]
    rows = []
    for nm, r in res.items():
        for k, v in r.items():
            bval = np.nan
            if k.startswith("conv|") and k != "conv|256-320":
                q = dec[(dec.design == nm) & (dec.M_bin == k.split("|")[1])]
                bval = float(q.convexity.iloc[0]) if len(q) else np.nan
            elif k in ("b1", "b2", "sigma_slope"):
                q = lin[(lin.design == nm) & (lin.param == k)]
                bval = float(q.est.iloc[0]) if len(q) else np.nan
            rows.append(dict(design=nm, statistic=k, reviewer=v, builder=bval,
                             abs_diff=abs(v - bval) if np.isfinite(bval) else np.nan))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(T, "rb5_sigma_review_convexity.csv"), index=False, float_format="%.6g")
    print(R.to_string())
    print("max abs diff where compared:", R.abs_diff.max())


if __name__ == "__main__":
    main()
