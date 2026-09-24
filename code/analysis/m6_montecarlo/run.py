"""m6_montecarlo -- v1 entry point (2026-09-23).  Regenerates every output of the module:

NOTE (v2, 2026-09-24): the paper now uses run_v2.py (no estimator started at the truth; extended profile grid;
heteroskedastic/clustered-noise cells; Design B with --design-b).  mc_lib/design_a/design_b now carry the v2
estimators, so running this file would write v2-estimator results under the v1 file names and the v1 four-panel
figure.  The v1 outputs are archived in data/processed/m6_montecarlo/v1_backup/.

  data/processed/m6_montecarlo/   replication-level Monte Carlo draws (parquet) + verification tables
  output/tables/m6_montecarlo_*   CSV + LaTeX tables
  output/figures/m6_montecarlo_*  PDF + PNG figures

Usage:  .venv/bin/python code/analysis/m6_montecarlo/run.py [--procs 6] [--quick] [--skip-sim]
  --quick     tiny replication counts (smoke test)
  --skip-sim  rebuild tables/figures from the saved replication files

Deterministic: every (cell, chunk) task seeds its own numpy Generator from a SeedSequence, so results do not
depend on the number of processes.  Uses CPU only (<= 6 processes, 1 BLAS thread each).
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
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-sim", action="store_true")
    args = ap.parse_args()
    if args.quick:
        os.environ["M6_QUICK"] = "1"
    import design_a as A
    import design_b as B
    import mc_lib as ml
    import outputs as O
    import verify_bias as V

    t0 = time.time()
    calibrate(ml, A)
    fa = os.path.join(ml.PROC, "designA_reps.parquet")
    fb = os.path.join(ml.PROC, "designB_reps.parquet")
    fc = os.path.join(ml.PROC, "designA_bootcheck.parquet")
    if not args.skip_sim:
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(args.procs, 6)) as pool:
            print(f"[m6] Design A: {len(A.tasks())} tasks", flush=True)
            ra = pool.map(A.run_chunk, A.tasks(), chunksize=1)
            dfa = pd.DataFrame([r for chunk in ra for r in chunk])
            dfa.to_parquet(fa)
            print(f"[m6] Design A done ({time.time() - t0:.0f}s); Design B: {len(B.tasks())} tasks", flush=True)
            rb = pool.map(B.run_chunk, B.tasks(), chunksize=1)
            dfb = pd.DataFrame([r for chunk in rb for r in chunk])
            dfb.to_parquet(fb)
            # review addition: warm-start vs multi-start bootstrap on fresh replications
            print(f"[m6] Design B done ({time.time() - t0:.0f}s); bootstrap check: {len(A.bootcheck_tasks())} tasks", flush=True)
            rc = pool.map(A.run_bootcheck_chunk, A.bootcheck_tasks(), chunksize=1)
            pd.DataFrame([r for chunk in rc for r in chunk]).to_parquet(fc)
        print(f"[m6] bootstrap check done ({time.time() - t0:.0f}s); verification", flush=True)
        ver, sel, chk = V.run()
        ver.to_csv(os.path.join(ml.PROC, "verify_transmission.csv"), index=False)
        sel.to_csv(os.path.join(ml.PROC, "verify_selection.csv"), index=False)
        chk.to_csv(os.path.join(ml.PROC, "verify_identities.csv"), index=False)
    dfa = pd.read_parquet(fa)
    dfb = pd.read_parquet(fb)
    dfc = pd.read_parquet(fc)
    ver = pd.read_csv(os.path.join(ml.PROC, "verify_transmission.csv"))
    sel = pd.read_csv(os.path.join(ml.PROC, "verify_selection.csv"))
    chk = pd.read_csv(os.path.join(ml.PROC, "verify_identities.csv"))
    O.make_all(dfa, dfb, ver, sel, chk, dfc)
    print(f"[m6] all outputs written ({time.time() - t0:.0f}s)", flush=True)


def calibrate(ml, A):
    """Noise calibration from raw data: residual sd of ln L around the Besiroglu et al. (2024) parameters on
    Epoch's digitized Chinchilla sample (n = 240 after dropping the 5 highest-loss points).  The Monte Carlo
    uses SD_BASE = 0.0075 (this number rounded); the check below documents the link."""
    import numpy as np
    df = ml.sl.chinchilla_extraction(os.path.join(ml.ROOT, "data/raw/epoch_chinchilla/svg_extracted_data.csv"))
    r = np.log(df.L.to_numpy()) - np.log(ml.TRUTH.loss(df.N.to_numpy(), df.D.to_numpy()))
    th, _ = ml.fit_huber(np.log(df.N.to_numpy()), np.log(df.D.to_numpy()), np.log(df.L.to_numpy()))
    cal = pd.DataFrame([dict(n=len(df), resid_sd_at_besiroglu=float(np.std(r, ddof=1)),
                             resid_mad_sd=float(1.4826 * np.median(np.abs(r - np.median(r)))),
                             sd_used=A.SD_BASE, refit_alpha=th[3], refit_beta=th[4], refit_E=float(np.exp(th[2])))])
    cal.to_csv(os.path.join(ml.PROC, "noise_calibration.csv"), index=False)
    print(f"[m6] calibration: resid sd {cal.resid_sd_at_besiroglu[0]:.4f} (MAD-sd {cal.resid_mad_sd[0]:.4f}); using {A.SD_BASE}", flush=True)


if __name__ == "__main__":
    main()
