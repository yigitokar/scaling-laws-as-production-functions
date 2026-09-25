"""Run a list of sweep jobs sequentially (each in its own subprocess), skipping finished ones.

Usage:
  python run_grid.py lrsweep          # learning-rate calibration at 3 widths
  python run_grid.py main edu         # main (N, D) design for one data regime
  python run_grid.py seeds edu        # seed replicates (pure-noise variance of output)
"""
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
LOG = os.path.join(HERE, "..", "..", "data", "processed", "sweep", "grid_log.txt")

# (d_model, n_layer): depth = width / 64, head_dim = 64
SIZES = [(128, 2), (192, 3), (256, 4), (320, 5), (384, 6), (448, 7), (512, 8), (640, 10)]
D_GRID = [25e6, 50e6, 100e6, 200e6, 400e6, 800e6]
D_MAX = {128: 800e6, 192: 800e6, 256: 800e6, 320: 800e6, 384: 800e6, 448: 400e6, 512: 400e6, 640: 200e6}


def lr_rule(d):
    """Peak LR as a function of width; calibrated by the 'lrsweep' jobs (see lr_fit.json)."""
    fit = os.path.join(HERE, "..", "..", "data", "processed", "sweep", "lr_fit.json")
    if os.path.exists(fit):
        f = json.load(open(fit))
        return float(f["c"] * (d / 256.0) ** f["p"])
    return 3e-3 * (256.0 / d) ** 0.5


def jobs(kind, regime="edu"):
    if kind == "lrsweep":
        for d, L in [(128, 2), (320, 5), (512, 8)]:
            for lr in [1e-3, 2e-3, 4e-3, 8e-3]:
                yield dict(regime="edu", d=d, L=L, lr=lr, D="50e6", tag="lrsweep", evals="edu")
    elif kind == "main":
        for d, L in SIZES:
            Ds = [D for D in D_GRID if D <= D_MAX[d]]
            j = dict(regime=regime, d=d, L=L, lr=round(lr_rule(d), 6), D=",".join(f"{D:.0f}" for D in Ds), tag="main")
            if regime == "web":  # runs launched after 2026-09-24 17:45 also get the neutral sets
                j["evals"] = "edu,web,wiki,c4,pg19"
            yield j
    elif kind == "edure":  # FineWeb-Edu main-grid widths 384 and 512 retrained with every evaluation set, same seed and
        # data order as the original grid (Amendment 2, item X3(a); approved by the author 2026-09-25 at about 11:33, launched 11:34:53)
        for d, L in [(384, 6), (512, 8)]:
            Ds = [D for D in D_GRID if D <= D_MAX[d]]
            yield dict(regime="edu", d=d, L=L, lr=round(lr_rule(d), 6), D=",".join(f"{D:.0f}" for D in Ds), tag="mainre",
                       evals="edu,web,wiki,c4,pg19")
    elif kind == "hiMlr":  # LR x {0.5, 2} at the high-M 4-layer shape (referee R2 round 2, Major 3c)
        for mult in [0.5, 2.0]:
            yield dict(regime=regime, d=128, L=4, lr=round(mult * lr_rule(128), 6), D="200e6,800e6,1600e6",
                       tag="hiMlr", evals="edu,web,wiki,c4,pg19", ext=1)
    elif kind == "hiMwd":  # weight decay scaled so the AdamW timescale / run length matches the main-grid median
        steps = lambda D: D / (64 * 256)
        ratios = sorted((1.0 / (lr_rule(d) * 0.1)) / steps(D) for d, L in SIZES for D in D_GRID if D <= D_MAX[d])
        med = ratios[len(ratios) // 2]
        lr = lr_rule(128)
        wd = 1.0 / (lr * med * steps(1.6e9))
        yield dict(regime=regime, d=128, L=4, lr=round(lr, 6), D="200e6,800e6,1600e6", tag="hiMwd",
                   wd=round(wd, 6), evals="edu,web,wiki,c4,pg19", ext=1)
    elif kind == "lrcal":  # replicate the LR calibration for another corpus (tests lab-specific tuning)
        for d, L in [(128, 2), (320, 5), (512, 8)]:
            for lr in [1e-3, 2e-3, 4e-3, 8e-3]:
                yield dict(regime=regime, d=d, L=L, lr=lr, D="50e6", tag="lrcal", evals="edu,web,wiki")
    elif kind == "lrcorner":  # LR x {0.5, 2} around the rule at the high-M corner (full D ladder) and low-M corner
        for mult in [0.5, 2.0]:
            yield dict(regime=regime, d=128, L=2, lr=round(mult * lr_rule(128), 6),
                       D="25e6,50e6,100e6,200e6,400e6,800e6", tag="lrcorner", evals="edu,web,wiki")
            yield dict(regime=regime, d=640, L=10, lr=round(mult * lr_rule(640), 6), D="25e6,50e6",
                       tag="lrcorner", evals="edu,web,wiki")
    elif kind == "seedcorner":  # seed replicate at the high-M corner
        yield dict(regime=regime, d=128, L=2, lr=round(lr_rule(128), 6), D="200e6,400e6,800e6", tag="seedcorner",
                   seed=1, data_seed=1, evals="edu,web,wiki,c4,pg19")
    elif kind == "hiM":  # high tokens-per-parameter runs with a 4-layer floor; extension data
        # time budget: both 4-layer shapes for edu; only the highest-M shape (128, 4) for web
        for d, L in ([(128, 4), (256, 4)] if regime == "edu" else [(128, 4)]):
            yield dict(regime=regime, d=d, L=L, lr=round(lr_rule(d), 6), D="200e6,800e6,1600e6,3200e6",
                       tag="hiM", evals="edu,web,wiki,c4,pg19", ext=1)
    elif kind == "hiM2":  # the 2-layer width-128 model continued to 3.2B tokens -- DROPPED for time (2026-09-24)
        return
        yield dict(regime=regime, d=128, L=2, lr=round(lr_rule(128), 6), D="800e6,1600e6,3200e6",
                   tag="hiM2", evals="edu,web,wiki", ext=1)
    elif kind == "seeds":
        for d, L in [(128, 2), (256, 4), (384, 6)]:
            for seed in [1, 2]:
                yield dict(regime=regime, d=d, L=L, lr=round(lr_rule(d), 6), D="50e6,100e6,200e6", tag="seeds",
                           seed=seed, data_seed=seed, evals="edu,web" if regime == "edu" else "edu,web,wiki,c4,pg19")


def main():
    kind = sys.argv[1]
    regime = sys.argv[2] if len(sys.argv) > 2 else "edu"
    for j in jobs(kind, regime):
        cmd = [PY, os.path.join(HERE, "train_sweep.py")] + sum([[f"--{k}", str(v)] for k, v in j.items()], [])
        with open(LOG, "a") as lg:
            lg.write("RUN " + " ".join(cmd) + "\n")
        print("RUN", j, flush=True)
        r = subprocess.run(cmd, cwd=HERE)
        with open(LOG, "a") as lg:
            lg.write(f"EXIT {r.returncode}\n")


if __name__ == "__main__":
    main()
