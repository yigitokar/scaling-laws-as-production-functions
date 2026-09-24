"""run.py -- single entry point for module m1_chinchilla ("What the canonical data identify").

Regenerates every table (output/tables/m1_chinchilla_*.csv|.tex, technology_registry_m1.csv), figure
(output/figures/m1_chinchilla_*.pdf|.png) and processed file (data/processed/m1_chinchilla/*) from the raw data:
    data/raw/epoch_chinchilla/svg_extracted_data.csv          (Epoch digitization of Hoffmann et al. 2022, Fig. 4)
    data/raw/isoflop_experiments/isoflop_experiments.csv      (open-athena compilation: Llama 3, Marin, (Mis)Fitting)
    data/raw/llama3_isoflop/isoflops_points.csv               (Czech digitization of Llama 3 Fig. 2; cross-check)
Download commands: code/data/download_m1_chinchilla.sh.

Usage:  .venv/bin/python code/analysis/m1_chinchilla/run.py [--stages horse,selection,duality,fdep,spec,labs,syslr,checks,report]
        [--quick]   (small bootstrap sizes, for testing only)
Seeds are fixed (common.SEED); CPU only, 6 processes; full run ~30-40 minutes on an M-series laptop.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from common import PROC  # noqa: E402

STAGES = ["horse", "selection", "duality", "fdep", "spec", "labs", "syslr", "checks", "report"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    stages = args.stages.split(",")
    logf = open(os.path.join(PROC, "run_log.txt"), "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    q = args.quick
    t0 = time.time()
    log(f"m1_chinchilla run: stages={stages} quick={q}")
    if "horse" in stages:
        import horse_race
        horse_race.run(B=24 if q else 400, log=log)
        horse_race.besiroglu_diagnostics(log=log)
        log(f"horse race done ({time.time() - t0:.0f}s)")
    if "selection" in stages:
        import selection
        selection.run(B=20 if q else 200, log=log)
        log(f"selection done ({time.time() - t0:.0f}s)")
    if "duality" in stages:
        import duality
        duality.run(B=16 if q else 300, log=log)
        log(f"duality done ({time.time() - t0:.0f}s)")
    if "fdep" in stages:
        import fdep
        fdep.run(B=16 if q else 200, log=log)
        log(f"functional dependence done ({time.time() - t0:.0f}s)")
    if "spec" in stages:
        import spec_tests
        spec_tests.run(B=16 if q else 300, log=log)
        log(f"specification tests done ({time.time() - t0:.0f}s)")
    if "labs" in stages:
        import labs
        labs.run(B=16 if q else 200, log=log)
        log(f"lab technologies done ({time.time() - t0:.0f}s)")
    if "syslr" in stages:
        import duality
        duality.system_lr_all(log=log)
        log(f"nested system LR tests done ({time.time() - t0:.0f}s)")
    if "checks" in stages:
        # independent-review additions (paired estimator differences, bootstrap-calibrated and classical duality tests,
        # design decomposition, small-sample LR calibration, warm-start check, size-matched fdep control)
        import review_checks
        review_checks.run(log=log)
        log(f"review checks done ({time.time() - t0:.0f}s)")
    if "report" in stages:
        import figures
        import registry
        import tables_tex
        registry.run(log=log)
        tables_tex.run(log=log)
        figures.run(log=log)
        log(f"report done ({time.time() - t0:.0f}s)")
    log(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
