"""v2 (2026-09-24) robustness check for Design A: does the near-path deterioration after removing the truth from the
start set come from too few random starts?

run_v2.py fits the primal (Huber-LSE, kappa = 1) estimator from ml.STARTS (9 random starts from Hoffmann's grid region,
none at the truth).  This check replays EXACTLY the same simulated data (same seeds; the bootstrap resampling draws of
the first R_BOOT replications of bootstrap cells are consumed without fitting, so the random stream is identical) for
the near-path cells and refits every replication from 64 random starts (the 9 of ml.STARTS plus 55 more drawn from the
same region with another seed; again none at the truth).  It reports, by cell and start set, RMSE / median error /
MAE of sigma*-hat and ln M*-hat, the corner share, and the share of replications whose objective exceeds the objective
at the true parameters (optimisation failures), and checks that the 9-start refit reproduces run_v2's estimates.

Usage: .venv/bin/python code/analysis/m6_montecarlo/designA_startcheck.py [--procs 4]
Writes output/tables/m6_montecarlo_designA_startcheck.csv, data/processed/m6_montecarlo/designA_startcheck.parquet
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

CELLS = ["onpath", "opt_s0", "opt_s0.05", "opt_s0.1", "opt_s0.2", "kaplan", "onpath_hc", "opt_s0.1_hc"]
N_EXTRA = 55


def extra_starts():
    rs = np.random.default_rng(54321)
    return [np.array([rs.uniform(0, 25), rs.uniform(0, 25), rs.uniform(-1, 1), rs.uniform(0.1, 1.2), rs.uniform(0.1, 1.2)])
            for _ in range(N_EXTRA)]


def run_chunk(args):
    import design_a as A
    import mc_lib as ml
    ci, chunk = args
    cell = A.cells()[ci]
    rng = np.random.default_rng(np.random.SeedSequence([A.BASE_SEED, ci, chunk]))
    starts64 = ml.STARTS + extra_starts()
    rows = []
    for j in range(A.CHUNK):
        rep = chunk * A.CHUNK + j
        N, D = cell["gen"](rng)
        Nobs, Dobs, L = ml.simulate_loss(N, D, cell["sd"], rng, digitize=cell.get("digitize", False), noise=cell.get("noise"))
        lN, lD, lL = np.log(Nobs), np.log(Dobs), np.log(L)
        if cell["boot"] and rep < A.R_BOOT:          # consume the bootstrap draws so that the stream matches run_v2
            n = len(lL)
            for _ in range(A.B_BOOT):
                rng.integers(0, n, n)
        obj_truth = float(ml.sl._obj(ml.TH0, lN, lD, lL, 1e-3, np.ones_like(lL)))
        r = dict(cell=cell["name"], rep=rep, obj_truth=obj_truth)
        for tag, st in (("s9", ml.STARTS), ("s64", starts64)):
            th, obj = ml.fit_huber(lN, lD, lL, starts=st)
            est = ml.derived_vec(th)
            r[f"{tag}_obj"] = obj
            r[f"{tag}_corner"] = A.is_corner(th, lN, lD)
            for k, x in zip(ml.PARAMS, est):
                r[f"{tag}_{k}"] = x
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    args = ap.parse_args()
    import design_a as A
    import mc_lib as ml
    t0 = time.time()
    names = [c["name"] for c in A.cells()]
    tasks = [(names.index(c), ch) for c in CELLS for ch in range(A.R_MAIN // A.CHUNK)]
    with mp.get_context("spawn").Pool(min(args.procs, 4)) as pool:
        res = pool.map(run_chunk, tasks, chunksize=1)
    df = pd.DataFrame([r for ch in res for r in ch])
    df.to_parquet(os.path.join(ml.PROC, "designA_startcheck.parquet"))
    v2 = pd.read_parquet(os.path.join(ml.PROC, "designA_reps_v2.parquet"))
    m = df.merge(v2[["cell", "rep", "p_sigma_star", "obj_truth"]], on=["cell", "rep"], suffixes=("", "_v2"))
    rows = []
    for c in CELLS:
        g = m[m.cell == c]
        row = dict(cell=c, R=len(g),
                   replay_max_abs_diff_obj_truth=float(np.max(np.abs(g.obj_truth - g.obj_truth_v2))),
                   replay_max_abs_diff_sigma_s9=float(np.nanmax(np.abs(g.s9_sigma_star - g.p_sigma_star))),
                   share_s64_lower_obj=float(np.mean(g.s64_obj < g.s9_obj - 1e-12)))
        for tag in ("s9", "s64"):
            for k in ("sigma_star", "lnMstar", "a"):
                err = g[f"{tag}_{k}"].to_numpy(float) - ml.TRUE[k]
                row[f"{tag}_{k}_rmse"] = float(np.sqrt(np.mean(err ** 2)))
                row[f"{tag}_{k}_median_err"] = float(np.median(err))
                row[f"{tag}_{k}_mae"] = float(np.median(np.abs(err)))
            row[f"{tag}_corner_share"] = float(g[f"{tag}_corner"].mean())
            row[f"{tag}_share_obj_above_truth"] = float(np.mean(g[f"{tag}_obj"] > g.obj_truth + 1e-9))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(ml.TAB, "m6_montecarlo_designA_startcheck.csv"), index=False)
    pd.set_option("display.width", 250)
    print(out.round(4).T.to_string())
    print(f"[m6 A startcheck] {len(df)} rows, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
