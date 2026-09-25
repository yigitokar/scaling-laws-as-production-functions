"""run.py -- module rb1_sigmaC: sigma*(C) and more extrapolation checks (round-2 referee requests R1 New 2, R3 N1,
R4 R2-M4, R2 Major 1 and Major 5).

One command regenerates every output of the module from the reviewed outputs of ra1_modelfree, ra2_wedge and
m8_measurement and from data/raw (the IsoFLOP designs and Farseer, through ra1's loaders):
    .venv/bin/python code/analysis/rb1_sigmaC/run.py            # all stages
    .venv/bin/python code/analysis/rb1_sigmaC/run.py --stages tables figures   # from the caches
Stages: meta (task 1: meta-regression of sigma*_b on log10 C; task 2: study-level interval) -> pid (task 3: partial
identification of sigma*(C) and the wedge magnitudes) -> extrap (task 4: Marin and Llama 3 local wedge vs parametric
extrapolation; wild bootstraps) -> tuning (task 5: tuning share of Farseer's convexity) -> driftmc (Monte Carlo of the
drift estimator under constant and drifting truths) -> tables -> figures.
Seeds are fixed (rb1common.SEED, one stream per key). CPU only, at most 4 worker processes (GPU queues running).
Environment: RB1_QUICK=1 for a smoke test with tiny bootstraps; RB1_OUTPUT_ROOT to redirect all outputs;
RB1_CHINCHILLA ([rb4 integration, 2026-09-24]): 'rb4' (default, primary) puts Chinchilla in FLOP-effective parameters
N_F with FLOPs per token rebuilt by module rb4_chinflop (its override inputs data/processed/rb4_chinflop/override_new_*;
run rb4_chinflop first); 'ra1' is the sensitivity with Chinchilla in total parameters, as rb1 was first published, and
writes to output/sensitivity/rb1_chinchilla_T/ unless RB1_OUTPUT_ROOT is set. See rb1common.py.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

import rb1common as cm  # noqa: E402

STAGES = ["meta", "pid", "extrap", "tuning", "driftmc", "tables", "figures"]


def cache(name):
    return os.path.join(cm.PROC, f"stage_{name}.pkl")


def save(name, obj):
    with open(cache(name), "wb") as f:
        pickle.dump(obj, f)


def load(name):
    with open(cache(name), "rb") as f:
        return pickle.load(f)


def stage_meta():
    import meta_c as mc
    cm.log("stage meta: meta-regression of budget-level sigma*_b on log10 C; study-level interval")
    R = mc.run_meta(B=999 if cm.QUICK else 9999)
    R["top"] = mc.top_budget_values(R["data"])
    V, conv = mc.study_level()
    R["study"], R["conv"] = V, conv
    # drop the (unpicklable-heavy) fit matrices except what later stages need
    f = R["fits"]["weighted"]
    R["primary_line"] = dict(mu=float(f["beta"][0]), beta=float(f["beta"][1]), cov=f["cov"].tolist(),
                             tau_d2=f["tau_d2"], tau_w2=f["tau_w2"])
    fu = R["fits"]["unweighted"]
    R["unweighted_line"] = dict(mu=float(fu["beta"][0]), beta=float(fu["beta"][1]))
    del R["fits"]
    save("meta", R)
    return R


def stage_pid():
    import pid
    R = load("meta")
    cm.log("stage pid: partial identification of sigma*(C); wedge scale and revealed-demand magnitudes")
    c, k_ref = pid.load_clean()
    sig_top = float(R["top"].iloc[0]["mean_fixed"])
    sig_top_ci = (float(R["top"].iloc[0]["lo_hksj"]), float(R["top"].iloc[0]["hi_hksj"]))   # review addition
    sl = R["slopes"]
    b_cl = float(sl[sl["sample"].str.startswith("Chinchilla + Llama") & (sl.spec_key == "fe_weighted")].slope.iloc[0])
    b_pool = R["primary_line"]["beta"]
    slopes = {"drift of Chinchilla and Llama 3 (design FE, inverse variance)": b_cl,
              "pooled drift (primary three-level meta-regression)": b_pool}
    mu = R["primary_line"]["mu"]
    pred = lambda lc: mu + b_pool * (np.asarray(lc, float) - 20.0)  # noqa: E731
    out = dict(sig_top=sig_top, slopes=slopes, k_ref=k_ref,
               pi=pid.pi_table(sig_top, slopes, sig_top_ci=sig_top_ci),
               scen=pid.wedge_scenarios(c, k_ref, pred, sig_top, slopes, sig_top_hi=sig_top_ci[1]),
               models=pid.per_model(c, k_ref, sig_top, b_cl, pred),
               bounds=pid.magnitude_bounds(c, sig_top, slopes, sig_top_hi=sig_top_ci[1]))
    save("pid", out)
    return out


def stage_extrap():
    import extrap as ex
    cm.log("stage extrap: local wedge vs parametric extrapolation on Marin (x3) and Llama 3")
    bw, CV = ex.select_bandwidths()
    res = {"cv": CV, "bw": bw}
    for name in ex.DESIGNS:
        R = ex.run_design(name, bw[name])
        T, L, S, P = ex.tables(R)
        res[name] = dict(T=T, L=L, S=S, P=P, bw=R["bw"], sm0=R["sm0"], sm_pop=R["sm_pop"], grid=R["grid"],
                         st0={k: v for k, v in R["st0"].items()}, nfail=R["nfail"], n=len(R["d"]),
                         n_raw=len(R["d_raw"]), flagged=R["flagged"],
                         draws=[d["sm"] for d in R["draws"]], draws_run=[d["sm"] for d in R["draws_run"]],
                         sigma_param=R["sigma_param"], budgets=R["budgets"])
        save("extrap", res)
    return res


def stage_tuning():
    import tuning as tu
    cm.log("stage tuning: tuning share of Farseer's convexity (Step Law gradients, D6 formula)")
    R = tu.run()
    save("tuning", R)
    return R


def stage_driftmc():
    import driftmc as dm
    import symwin
    cm.log("stage driftmc: Monte Carlo of the drift estimator under constant and drifting truths (R2 Major 1.3)")
    R = dm.run()
    R["symwin"] = symwin.run()          # review addition: count-symmetric windows and the E-fitted power frontier
    save("driftmc", R)
    return R


def stage_tables():
    import tables_rb1 as tb
    import meta_c as mc
    M, X = load("meta"), load("extrap")
    M["study"] = mc.add_local_fd_variant(M["study"], X)      # review addition (needs the extrap stage)
    tb.make_all(M, load("pid"), X, load("tuning"))
    D = load("driftmc")
    cm.tab(D["table"], "driftmc")
    cm.tab(D["truth"], "driftmc_truth")
    if "symwin" in D:                                           # review addition (R2 Major 1.3, symmetric windows)
        cm.tab(D["symwin"], "driftmc_symwin")


def stage_figures():
    import figures_rb1 as fg
    fg.make_all(load("meta"), load("pid"), load("extrap"), load("tuning"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", default=STAGES)
    a = ap.parse_args()
    cm.log(f"rb1_sigmaC run: stages {a.stages}; QUICK={cm.QUICK}; output root {cm._OUT}; Chinchilla "
           f"{'N_F rebuilt (rb4)' if cm.CHIN_REBUILT else 'total N (ra1)'}: {cm.RA1_SUMMARY}")
    for s in STAGES:
        if s in a.stages:
            globals()[f"stage_{s}"]()
    cm.log("done")
