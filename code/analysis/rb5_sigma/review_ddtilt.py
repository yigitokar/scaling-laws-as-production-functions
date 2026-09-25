"""review_ddtilt.py -- the reviewer's independent refit of the DataDecide final-checkpoint recipe tilt (T14; WP2-review).

Uses m2's data loader only (m2_data.load_datadecide; the raw panel), and its own model code: the common-exponent panel
model ln L = ln(A_r N^-alpha + B_r D^-beta + E_r) fitted by Gaussian NLS and by Huber (delta = 1e-3, m2's value) with
scipy least_squares from 40 starting points (a grid over alpha, beta and m2's all-checkpoint levels); tilt range of
ln(A_r/B_r) and the implied factor on M*, exp(2 range/(alpha+beta)). Also the NLS profile over alpha + beta and the
tilt range along it. No bootstrap here (the builder's cluster bootstrap is re-run by the full re-run of run.py).
Prints a comparison with output/tables/rb5_sigma_dd_tilt.csv. Single process; run under nice -n 10.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "code", "analysis", "m2_techpanel"))
import m2_data as md  # noqa: E402

REF = 0.424359 + 0.430527


def main():
    dd = md.load_datadecide()
    last = dd.groupby("run_id").step.transform("max")
    f = dd[dd.step == last].copy()
    f = f[f.step / f["size"].map(md.DD_LAST) >= 0.98].reset_index(drop=True)
    recs = sorted(f.recipe.unique())
    R = len(recs)
    g = f.recipe.map({r: i for i, r in enumerate(recs)}).values
    x, z, y = np.log(f.N.values), np.log(f.D.values), np.log(f.L.values)
    print(f"final checkpoints: {len(f)} runs, {R} recipes, M {np.min(f.D / f.N):.1f}-{np.max(f.D / f.N):.1f}")

    def res(th):
        a, b = th[0], th[1]
        lA, lB, lE = th[2:2 + R][g], th[2 + R:2 + 2 * R][g], th[2 + 2 * R:][g]
        return y - np.log(np.exp(lA - a * x) + np.exp(lB - b * z) + np.exp(lE))

    mg = pd.read_csv(os.path.join(ROOT, "output", "tables", "m2_neutrality_groups_datadecide.csv")).set_index("group").loc[recs]
    out = {}
    for est, kw in (("nls", dict(loss="linear")), ("huber", dict(loss="huber", f_scale=1e-3))):
        best = None
        for a0 in (0.3, 0.6, 0.9, 1.2):
            for b0 in (0.05, 0.1, 0.15, 0.2, 0.3):
                for lev in ("m2", "flat"):
                    if lev == "m2":
                        q = np.r_[mg.lnA.values + (a0 - 0.27) * np.mean(x), mg.lnB.values + (b0 - 0.15) * np.mean(z), np.log(mg.E.values)]
                    else:
                        q = np.r_[np.full(R, np.log(0.5) + a0 * np.mean(x)), np.full(R, np.log(0.5) + b0 * np.mean(z)), np.full(R, np.log(2.0))]
                    try:
                        r = least_squares(res, np.r_[a0, b0, q], method="trf", ftol=1e-13, xtol=1e-13, gtol=1e-13,
                                          max_nfev=20000, bounds=(np.r_[0, 0, np.full(3 * R, -np.inf)], np.r_[5, 5, np.full(3 * R, np.inf)]), **kw)
                    except Exception:
                        continue
                    if best is None or r.cost < best.cost:
                        best = r
        th = best.x
        tilt = th[2:2 + R] - th[2 + R:2 + 2 * R]
        rg = tilt.max() - tilt.min()
        out[est] = dict(alpha=th[0], beta=th[1], cost=best.cost, tilt_range=rg, tau_own=np.exp(2 * rg / (th[0] + th[1])),
                        tau_ref=np.exp(2 * rg / REF), tilt=tilt)
        print(f"[{est}] alpha {th[0]:.4f} beta {th[1]:.4f} cost {best.cost:.6g} tilt range {rg:.4f} "
              f"tau_own {out[est]['tau_own']:.3f} tau_ref {out[est]['tau_ref']:.3f}")
    b = pd.read_csv(os.path.join(ROOT, "output", "tables", "rb5_sigma_dd_tilt.csv"))
    print(b[b["sample"].str.startswith("final")][["estimator", "alpha", "beta", "objective", "tilt_range", "tau_own", "tau_ref"]].to_string())
    from scipy.stats import spearmanr
    print("Spearman final (NLS) vs all-checkpoint tilt:", spearmanr(out["nls"]["tilt"], mg.tilt_lnA_minus_lnB.values).correlation)
    # profile over alpha + beta along the NLS ridge: fix alpha, beta on a grid and refit levels
    th0 = None
    rows = []
    for a in (0.3, 0.5, 0.7, 0.86, 1.0, 1.2):
        for bb in (0.06, 0.094, 0.13, 0.18):
            def r2(q, a=a, bb=bb):
                return res(np.r_[a, bb, q])
            q0 = np.r_[mg.lnA.values + (a - 0.27) * np.mean(x), mg.lnB.values + (bb - 0.15) * np.mean(z), np.log(mg.E.values)]
            r = least_squares(r2, q0, method="trf", ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=20000)
            t = r.x[:R] - r.x[R:2 * R]
            rows.append(dict(alpha=a, beta=bb, ssr=2 * r.cost, tilt_range=t.max() - t.min(),
                             tau_own=np.exp(2 * (t.max() - t.min()) / (a + bb))))
    P = pd.DataFrame(rows)
    s2 = 2 * out["nls"]["cost"] / (len(y) - (2 + 3 * R))
    P["lr"] = (P.ssr - 2 * out["nls"]["cost"]) / s2
    print(P.round(4).to_string())


if __name__ == "__main__":
    main()
