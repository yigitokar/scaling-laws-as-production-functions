"""run.py -- single entry point of module ra1_modelfree ("The technology from designed variation, model-free").

Regenerates every output of the module from data/raw (no downloads; see code/data/download_m1_chinchilla.sh,
download_m2_techpanel.sh, download_m8_measurement.sh and code/data/download_ra1_modelfree.sh):
    data/processed/ra1_modelfree/      stage caches (*.pkl), Porian IsoFLOP points, run log
    output/tables/ra1_modelfree_*.csv|.tex
    output/figures/ra1_modelfree_*.pdf|.png
Stages (default: all, in this order):
    porian      IsoFLOP-like profiles from Porian et al.'s tuned runs (m8 pipeline, separate process)
    identity    numerical verification of the model-free identity 1/sigma* - 1 = L_nn|_C / (2|dL*/dlnC|)
    isoflop     model-free sigma* on every IsoFLOP design (Task 1)
    farseer     model-free local w, extrapolation test, sigma* along the local path (Task 2)
    chinchilla  inference fixes on the Chinchilla data (Task 4)
    kappa       sigma*_kappa robustness and design-conditional SEs (Task 5)
    practitioner compute-overhead table (Task 6)
    meta        heterogeneity across technologies (Task 3)
    review      independent reviewer's cross-checks (review_checks.py)
    tables, figures
Usage:  .venv/bin/python code/analysis/ra1_modelfree/run.py [--stages a,b,...]
        RA1_QUICK=1 for a smoke test with tiny bootstraps (never for reported numbers).
Deterministic: every random draw uses numpy default_rng seeded from ra1_common.SEED and a stage key.
CPU only; at most 5 worker processes (the GPU is used by two MLX training queues).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import pickle  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

import ra1_common as rc  # noqa: E402

STAGES = ["porian", "identity", "isoflop", "farseer", "chinchilla", "kappa", "practitioner", "meta", "review", "tables",
          "figures"]


def cache(name, obj=None):
    path = os.path.join(rc.PROC, f"stage_{name}.pkl")
    if obj is not None:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        return obj
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=",".join(STAGES))
    args = ap.parse_args()
    stages = args.stages.split(",")
    t0 = time.time()
    rc.log(f"ra1_modelfree run: stages={stages} quick={rc.QUICK} root={rc._OUT}")
    if "porian" in stages:
        out = os.path.join(rc.PROC, "porian_isoflop_points.csv")
        subprocess.run([sys.executable, os.path.join(HERE, "porian_points.py"), out], check=True)
        rc.log(f"porian points done ({time.time() - t0:.0f}s)")
    if "identity" in stages:
        import isoflop
        tab, mf = isoflop.verify_identity()
        cache("identity", tab)
        rc.log("identity check:\n" + tab.round(8).to_string())
    if "isoflop" in stages:
        import isoflop
        cache("isoflop", isoflop.run())
        rc.log(f"isoflop stage done ({time.time() - t0:.0f}s)")
    if "farseer" in stages:
        import farseer
        R = farseer.run()
        T = farseer.tables(R)
        # identity check on Farseer's fitted Eq. 3 (non-homothetic technology)
        import isoflop
        _, mf = isoflop.verify_identity()
        T["identity_eq3"] = isoflop.verify_farseer(R["est"]["ne"]["eq3_full"], mf)
        T["cv"], T["icc"], T["bw"], T["B"], T["B2"], T["nfail"] = R["cv"], R["icc"], R["bw"], R["B"], R["B2"], R["nfail"]
        T["cv_ext"], T["bw_ext"] = R["cv_ext"], R["bw_ext"]   # review fix M-A (extended CV grid)
        T["est"] = R["est"]
        T["sens"] = farseer.sensitivity(R)
        cache("farseer", T)
        rc.log(f"farseer stage done ({time.time() - t0:.0f}s)")
    if "chinchilla" in stages:
        import chin_inf
        out = chin_inf.run()
        out.pop("draws", None)
        cache("chinchilla", out)
        rc.log(f"chinchilla stage done ({time.time() - t0:.0f}s)")
    if "kappa" in stages:
        import kappa_rob
        cache("kappa", kappa_rob.run())
        rc.log(f"kappa stage done ({time.time() - t0:.0f}s)")
    if "practitioner" in stages:
        import practitioner
        cache("practitioner", practitioner.run())
    if "meta" in stages:
        import meta
        cache("meta", meta.run_all(cache("isoflop"), cache("farseer"), cache("kappa")))
        rc.log(f"meta stage done ({time.time() - t0:.0f}s)")
    if "review" in stages:   # independent reviewer's cross-checks (review_checks.py; output/memos/ra1_modelfree_review.md)
        import review_checks
        cache("review", review_checks.run(cache))
        rc.log(f"review checks done ({time.time() - t0:.0f}s)")
    if "tables" in stages:
        import tables
        tables.run(cache)
        rc.log(f"tables done ({time.time() - t0:.0f}s)")
    if "figures" in stages:
        import figures
        figures.run(cache)
        rc.log(f"figures done ({time.time() - t0:.0f}s)")
    rc.log(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
