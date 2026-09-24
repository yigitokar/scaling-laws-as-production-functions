"""Revision module ra4_obsfix: fixes to the observational and algorithmic-progress material for Online Appendix E.

Addresses R4 M4(d)(f), M7, M8(a)(b)(c), M11; R1 comments 8(g)(i), 9(e), 10(a)-(e) and minors 42-46 (see the memo,
output/memos/ra4_obsfix.md).  Single entry point; regenerates every output of the module (deterministic seeds):

    /Users/yigitokar/scaling-laws-pf/.venv/bin/python code/analysis/ra4_obsfix/run.py

Steps (each a separate process, because m4_observational and m5_progress both ship a module named `common`):
  1. step_matched.py   reliability of ln C; design-matched comparison with the strictly experimental benchmark
                       (OLMo-2 7B/13B removed) and with the original benchmark; few-cluster inference that includes the
                       benchmark's uncertainty; reconciliation of the original Table 8; EIV, reverse regression, IV + AR.
  2. step_alloc.py     allocative gains with the wedge truncated at one, across every estimated technology.
  3. step_progress.py  phi-profile refinement (T_C range over the interpolated interval); Ho et al. summary inputs;
                       Bjorck-exponent range for the Step Law learning-rate comparison.
  4. step_tfp.py       productivity dispersion in output and input units, with benchmark uncertainty.
  5. tables_figs.py    Online Appendix E tables (tabE1-tabE7 .tex) and figures.
Inputs from other modules (read-only): m4_observational code and raw data; m5_progress code and cached outputs
(Ho et al. bootstrap draws, profile grid, Besiroglu draws, Table 7 panel A); m8_measurement results_summary.json;
the m1/m2 technology registries (hashes recorded in data/processed/ra4_obsfix/allocative_headline.json).
CPU: at most 5 processes (step 3 uses a pool of 4).  Runtime 4-10 minutes depending on load.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ra4common as rc  # noqa: E402

PY = "/Users/yigitokar/scaling-laws-pf/.venv/bin/python"
STEPS = ["step_matched.py", "step_alloc.py", "step_progress.py", "step_tfp.py", "tables_figs.py"]


def main():
    t0 = time.time()
    log = []
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1", PYTHONHASHSEED="0")
    for st in STEPS:
        t1 = time.time()
        rc.log(f"running {st}")
        r = subprocess.run([PY, os.path.join(HERE, st)], cwd=rc.ROOT, env=env)
        log.append(dict(step=st, returncode=r.returncode, seconds=round(time.time() - t1, 1)))
        if r.returncode != 0:
            rc.log(f"{st} FAILED (exit {r.returncode})")
            json.dump(log, open(os.path.join(rc.PROC, "run_log.json"), "w"), indent=1)
            sys.exit(r.returncode)
    json.dump(dict(steps=log, total_seconds=round(time.time() - t0, 1)), open(os.path.join(rc.PROC, "run_log.json"), "w"), indent=1)
    rc.log(f"ra4_obsfix finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
