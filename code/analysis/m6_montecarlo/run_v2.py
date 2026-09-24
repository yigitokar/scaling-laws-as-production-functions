"""m6_montecarlo v2 -- re-run of Design A after referee round 1 (2026-09-24).

Changes relative to v1 (see design_a.py docstring): no start value at the truth anywhere (primal estimator, dual
frontier fit, multi-start bootstrap check); kappa-free profile on the extended grid (0.05, 0.99) with LR relative to
the unrestricted optimum; heteroskedastic/clustered-noise cells.  Every v1 cell keeps its seed, so the simulated data
are identical to v1 and all differences come from the estimator's start set (and, for the profile, the grid).

Design B (industry): with --design-b the industry Monte Carlo is re-run with the v2 estimators, which no longer start
at the truth (design_b.py docstring); same seeds, hence the same simulated industries.  The formula verifications
(verify_bias.py) involve no numerical optimisation from starting values and are not re-run (v1 files reused).

Usage:  .venv/bin/python code/analysis/m6_montecarlo/run_v2.py [--procs 4] [--quick] [--skip-sim] [--design-b] [--no-a]
Writes: data/processed/m6_montecarlo/designA_reps_v2.parquet, designA_bootcheck_v2.parquet
        output/tables/m6_montecarlo_designA*.csv|.tex (overwritten with v2 numbers; v1 in v1_backup/)
        output/figures/m6_montecarlo_fig2_v2.pdf|.png (two-panel paper figure), m6_montecarlo_profile_ci_v2
        output/tables/m6_montecarlo_designA_v2_vs_v1.csv (v1 vs v2 comparison)
        with --design-b: data/processed/m6_montecarlo/designB_reps_v2.parquet, output/tables/m6_montecarlo_industry*
        (overwritten; v1 in v1_backup/), output/figures/m6_montecarlo_industry_bias, and
        output/tables/m6_montecarlo_industry_v2_vs_v1.csv
Robustness checks (separate scripts): designA_startcheck.py (9 vs 64 random starts), designB_startcheck.py and
designB_startcheck2.py (truth vs truth-free starts for the industry estimators).
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

import pandas as pd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-sim", action="store_true")
    ap.add_argument("--design-b", action="store_true", help="also re-run Design B with the v2 (truth-free) starts")
    ap.add_argument("--no-a", action="store_true", help="skip Design A (use with --design-b)")
    args = ap.parse_args()
    if args.quick:
        os.environ["M6_QUICK"] = "1"
    import design_a as A
    import mc_lib as ml
    import outputs as O

    t0 = time.time()
    fa = os.path.join(ml.PROC, "designA_reps_v2.parquet")
    fc = os.path.join(ml.PROC, "designA_bootcheck_v2.parquet")
    if not args.skip_sim and not args.no_a:
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(args.procs, 4)) as pool:
            tasks = A.tasks()
            # longest tasks first (bootstrap cells) to balance the pool
            tasks = sorted(tasks, key=lambda t: (not A.cells()[t[0]]["boot"], t[1] > 0))
            print(f"[m6 v2] Design A: {len(tasks)} tasks, {len(A.cells())} cells", flush=True)
            ra = pool.map(A.run_chunk, tasks, chunksize=1)
            dfa = pd.DataFrame([r for chunk in ra for r in chunk])
            dfa.to_parquet(fa)
            print(f"[m6 v2] Design A done ({time.time() - t0:.0f}s); bootstrap check: {len(A.bootcheck_tasks())} tasks",
                  flush=True)
            rc = pool.map(A.run_bootcheck_chunk, A.bootcheck_tasks(), chunksize=1)
            pd.DataFrame([r for chunk in rc for r in chunk]).to_parquet(fc)
        print(f"[m6 v2] simulation done ({time.time() - t0:.0f}s)", flush=True)
    if not args.no_a:
        dfa = pd.read_parquet(fa)
        dfc = pd.read_parquet(fc)
        O.make_design_a_v2(dfa, dfc)
        print(f"[m6 v2] Design A outputs written ({time.time() - t0:.0f}s)", flush=True)
    if args.design_b:
        import design_b as B
        fb = os.path.join(ml.PROC, "designB_reps_v2.parquet")
        if not args.skip_sim:
            with mp.get_context("spawn").Pool(min(args.procs, 4)) as pool:
                print(f"[m6 v2] Design B: {len(B.tasks())} tasks", flush=True)
                rb = pool.map(B.run_chunk, B.tasks(), chunksize=1)
            pd.DataFrame([r for chunk in rb for r in chunk]).to_parquet(fb)
            print(f"[m6 v2] Design B done ({time.time() - t0:.0f}s)", flush=True)
        dfb = pd.read_parquet(fb)
        O.make_design_b_v2(dfb)
        print(f"[m6 v2] Design B outputs written ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
