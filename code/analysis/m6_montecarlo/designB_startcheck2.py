"""v2 (2026-09-24) follow-up to designB_startcheck.py: which start is the better optimum?

designB_startcheck.py showed that the pooled NLS estimators (E2 pooled NLS + generation effects, E3 lab FE, E4 Mundlak)
return different estimates from a data-based neutral start than from the truth in some replications, mostly for
ln M*(10^24) and under the target rule.  This script replays the same simulated data (first R_CHK replications of the
four main compute rules, same seeds as v1) and, for E2, E3 and E4, compares the least-squares cost reached from
  (i)   the truth (v1),
  (ii)  the neutral start of designB_startcheck.py (alpha = beta = 0.5, levels matched to mean y),
  (iii) a multi-start set with NO start at the truth: the neutral start plus alpha, beta in {0.2, 0.5, 1.0}^2 with
        levels matched to mean y (9 starts in all), keeping the lowest cost.
If (iii) reaches the v1 cost (to 1e-9 relative) and reproduces the v1 estimates, the v1 Design B numbers are the global
least-squares estimates and do not depend on knowing the truth.

Usage: .venv/bin/python code/analysis/m6_montecarlo/designB_startcheck2.py [--procs 4]
Writes output/tables/m6_montecarlo_industry_startcheck2.csv
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

R_CHK = 100
MAIN = ["exog", "funding", "target", "predet"]


def _fit(B, S, which, s0):
    """E2/E3/E4 exactly as in design_b.py but from the normalized start s0 = (lA, lB, alpha, beta); returns
    (cost, summarize_fit dict)."""
    if which == "E2":
        def res(q):
            F, J = B.F_J(S, *q[:4])
            return S.y - F - S.G @ q[4:], np.column_stack([-J, -S.G])
        q0 = np.r_[s0, np.zeros(S.Tn - 1)]
        sol = B.ls(lambda q: res(q)[0], q0, jac=lambda q: res(q)[1], method="lm")
        return sol.cost, B.summarize_fit(S, *sol.x[:4])
    if which == "E3":
        Gd = S.demean(S.G, S.lab_i)

        def res(q):
            F, J = B.F_J(S, 0.0, q[0], q[1], q[2])
            r = S.demean(S.y - F, S.lab_i) - Gd @ q[3:]
            return r, np.column_stack([-S.demean(J[:, 1:], S.lab_i), -Gd])
        q0 = np.r_[s0[1] - s0[0], s0[2:], np.zeros(S.Tn - 1)]
        sol = B.ls(lambda q: res(q)[0], q0, jac=lambda q: res(q)[1], method="lm")
        return sol.cost, B.summarize_fit(S, 0.0, *sol.x[:3])
    nb, db = S.group_mean(S.n, S.lab_i), S.group_mean(S.d, S.lab_i)
    X = np.column_stack([np.ones(S.N), nb - S.n0, db - S.d0])

    def res(q):
        F, J = B.F_J(S, 0.0, q[0], q[1], q[2])
        r = S.y - F - S.G @ q[3:3 + S.Tn - 1] - X @ q[3 + S.Tn - 1:]
        return r, np.column_stack([-J[:, 1:], -S.G, -X])
    q0 = np.r_[s0[1] - s0[0], s0[2:], np.zeros(S.Tn - 1), -s0[0], 0.0, 0.0]
    sol = B.ls(lambda q: res(q)[0], q0, jac=lambda q: res(q)[1], method="lm")
    return sol.cost, B.summarize_fit(S, 0.0, *sol.x[:3])


def run_chunk(args):
    import design_b as B
    si, chunk = args
    sc = B.SCENARIOS[si]
    rng = np.random.default_rng(np.random.SeedSequence([B.BASE_SEED, si, chunk]))
    rows = []
    for j in range(B.CHUNK):
        rep = chunk * B.CHUNK + j
        Sd = B.simulate(rng, sc)
        S = B.Sample(Sd)
        lv = -float(np.mean(S.y)) - np.log(2.0)
        truth = B.start_norm(S)
        neutral = np.array([lv, lv, 0.5, 0.5])
        grid = [neutral] + [np.array([lv, lv, al, be]) for al in (0.2, 0.5, 1.0) for be in (0.2, 0.5, 1.0)
                            if not (al == 0.5 and be == 0.5)]
        for which in ("E2", "E3", "E4"):
            r = dict(scenario=sc["name"], rep=rep, estimator=which)
            try:
                c_t, e_t = _fit(B, S, which, truth)
                c_n, e_n = _fit(B, S, which, neutral)
                best = (np.inf, None)
                for s0 in grid:
                    try:
                        c, e = _fit(B, S, which, s0)
                    except Exception:
                        continue
                    if c < best[0]:
                        best = (c, e)
                c_m, e_m = best
                r.update(cost_truth=c_t, cost_neutral=c_n, cost_multi=c_m)
                for k in ("gamma", "a", "sigma_star", "lnMstar", "tfp_growth"):
                    r[f"{k}_truth"], r[f"{k}_neutral"], r[f"{k}_multi"] = e_t[k], e_n[k], e_m[k]
            except Exception as ex:          # noqa: BLE001
                r["error"] = str(ex)[:80]
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    args = ap.parse_args()
    import design_b as B
    import mc_lib as ml
    t0 = time.time()
    names = [s["name"] for s in B.SCENARIOS]
    tasks = [(names.index(s), ch) for s in MAIN for ch in range(R_CHK // B.CHUNK)]
    with mp.get_context("spawn").Pool(min(args.procs, 4)) as pool:
        res = pool.map(run_chunk, tasks, chunksize=1)
    df = pd.DataFrame([r for ch in res for r in ch])
    df.to_parquet(os.path.join(ml.PROC, "designB_startcheck2.parquet"))
    tr = B.truth_row()
    rows = []
    for (sc, e), g in df.groupby(["scenario", "estimator"], sort=False):
        tol = 1e-9 * np.maximum(1.0, g.cost_truth.abs())
        row = dict(scenario=sc, estimator=e, R=len(g),
                   share_neutral_worse=float(np.mean(g.cost_neutral > g.cost_truth + tol)),
                   share_neutral_better=float(np.mean(g.cost_neutral < g.cost_truth - tol)),
                   share_multi_worse=float(np.mean(g.cost_multi > g.cost_truth + tol)),
                   share_multi_better=float(np.mean(g.cost_multi < g.cost_truth - tol)))
        for k in ("gamma", "a", "lnMstar", "tfp_growth"):
            for tag in ("truth", "neutral", "multi"):
                row[f"bias_{k}_{tag}"] = float(np.mean(g[f"{k}_{tag}"] - tr[k]))
            row[f"maxdiff_{k}_multi_vs_truth"] = float(np.max(np.abs(g[f"{k}_multi"] - g[f"{k}_truth"])))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(ml.TAB, "m6_montecarlo_industry_startcheck2.csv"), index=False)
    pd.set_option("display.width", 250)
    print(out.round(4).to_string())
    print(f"[m6 B startcheck2] {len(df)} rows, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
