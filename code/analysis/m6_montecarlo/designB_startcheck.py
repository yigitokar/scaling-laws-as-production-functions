"""v2 (2026-09-24) robustness check for Design B: do the industry estimators depend on starting at the truth?

In v1 every Design B estimator started at (or near) the true parameters: E1 from (truth + 8 random starts), E1b from
(truth + 8 random starts, E fixed), E2/E3/E4/E5/E8 from the truth in normalized form (start_norm), and E6/E9/E10 from
E2/E5 solutions.  This check re-runs the first R_CHK replications of every scenario on IDENTICAL simulated data (same
seeds as v1) with no start at the truth:
  * start_norm -> a data-based neutral start: alpha = beta = 0.5 and lA = lB = -mean(y) - ln 2 (so F at the design
    centre equals the sample mean of y);
  * E1 -> ml.STARTS (9 random starts, none at the truth; the v2 default of ml.fit_huber);
  * E1b -> ml.STARTS with ln E fixed at the truth (E is known by construction in E1b).
It then compares every estimate with the v1 estimate of the same replication.

Usage: .venv/bin/python code/analysis/m6_montecarlo/designB_startcheck.py [--procs 4]
Writes output/tables/m6_montecarlo_industry_startcheck.csv (per scenario x estimator x parameter: v1 and v2 bias on the
same replications, max |difference| of the estimates) and data/processed/m6_montecarlo/designB_startcheck.parquet.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

R_CHK = 100          # replications per scenario (the first 4 chunks of 25 of v1)


def _patch():
    import design_b as B
    import mc_lib as ml

    def start_norm_neutral(S):
        lv = -float(np.mean(S.y)) - np.log(2.0)
        return np.array([lv, lv, 0.5, 0.5])

    def est_E1b_nt(S):
        lN, lD, lL = S.n, S.d, np.log(S.L)
        b = list(ml.BOUNDS)
        b[2] = (np.log(B.E_TRUE), np.log(B.E_TRUE))
        starts = [np.r_[st[:2], np.log(B.E_TRUE), st[3:]] for st in ml.STARTS]
        th, _ = ml.fit_huber(lN, lD, lL, starts=starts, bounds=b)
        out = dict(zip(ml.PARAMS, ml.derived_vec(th)))
        Fh = -np.logaddexp(th[0] - th[3] * lN, th[1] - th[4] * lD)
        out.update(B.tfp_stats(S, S.y - Fh))
        return out

    B.start_norm = start_norm_neutral
    B.est_E1b = est_E1b_nt
    return B


def run_chunk(args):
    B = _patch()
    return B.run_chunk(args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    args = ap.parse_args()
    import design_b as B
    import mc_lib as ml
    t0 = time.time()
    tasks = [(si, ch) for si in range(len(B.SCENARIOS)) for ch in range(R_CHK // B.CHUNK)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(args.procs, 4)) as pool:
        res = pool.map(run_chunk, tasks, chunksize=1)
    d2 = pd.DataFrame([r for ch in res for r in ch])
    d2.to_parquet(os.path.join(ml.PROC, "designB_startcheck.parquet"))
    d1 = pd.read_parquet(os.path.join(ml.PROC, "designB_reps.parquet"))
    d1 = d1[d1.rep < R_CHK]
    tr = B.truth_row()
    keys = ["scenario", "rep", "estimator"]
    m = d1.merge(d2, on=keys, suffixes=("_v1", "_v2"))
    rows = []
    for (sc, e), g in m.groupby(["scenario", "estimator"]):
        for k in ["alpha", "beta", "a", "gamma", "sigma_star", "lnMstar", "sd_omega", "tfp_growth"]:
            if f"{k}_v1" not in g:
                continue
            x1, x2 = g[f"{k}_v1"].to_numpy(float), g[f"{k}_v2"].to_numpy(float)
            ok = np.isfinite(x1) & np.isfinite(x2)
            if not ok.any():
                continue
            rows.append(dict(scenario=sc, estimator=e, param=k, R=int(ok.sum()),
                             n_fail_v1=int((~np.isfinite(x1)).sum()), n_fail_v2=int((~np.isfinite(x2)).sum()),
                             bias_v1=float(np.mean(x1[ok] - tr[k])), bias_v2=float(np.mean(x2[ok] - tr[k])),
                             median_bias_v1=float(np.median(x1[ok] - tr[k])), median_bias_v2=float(np.median(x2[ok] - tr[k])),
                             max_abs_diff=float(np.max(np.abs(x2[ok] - x1[ok]))),
                             share_diff_gt_1e4=float(np.mean(np.abs(x2[ok] - x1[ok]) > 1e-4))))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(ml.TAB, "m6_montecarlo_industry_startcheck.csv"), index=False)
    print(f"[m6 B startcheck] {len(d2)} rows, {time.time() - t0:.0f}s", flush=True)
    s = out.groupby("estimator").agg(max_abs_diff=("max_abs_diff", "max"), share_gt=("share_diff_gt_1e4", "max"))
    print(s.to_string())


if __name__ == "__main__":
    main()
