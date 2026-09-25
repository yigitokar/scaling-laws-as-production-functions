"""run.py -- module rb5_sigma: the sigma* re-analysis bundle of round 3 (fix list item T1; the analysis part of T14;
the numbers behind T10 and T16). See sigcommon.py for scope, inputs and constraints.

One command regenerates every output of the module:
    nice -n 10 .venv/bin/python code/analysis/rb5_sigma/run.py                      # all stages, figure previews
    nice -n 10 .venv/bin/python code/analysis/rb5_sigma/run.py --stages tables figures   # from the stage caches
    nice -n 10 .venv/bin/python code/analysis/rb5_sigma/run.py --stages figures --final-figures
        # after the independent review: writes the paper's output/figures/rb1_sigmaC_extrap.* and fig3_merged.*
    nice -n 10 .venv/bin/python code/analysis/rb5_sigma/run.py --determinism
        # re-runs every stage into a scratch root and compares all CSV outputs byte for byte
Stages: draws (T1.1, T1.3: joint bootstrap draws of the budget-level sigma*_b) -> metareg (T1.2) -> convexity (T1.9,
Farseer's restricted first-derivative estimate for T1.4) -> study (T1.4-T1.6, T1.10) -> pidset (T1.7) -> depmc (T1.8)
-> ddtilt (T14) -> tables -> figures.
Seeds: reproductions use the upstream modules' seeds (ra1, rb1); new draws use sigcommon.SEED, one stream per key.
CPU only, at most 2 worker processes (sigcommon.N_PROC), no MLX; run under nice -n 10.
Environment: RB5_QUICK=1 smoke test (small bootstraps); RB5_OUTPUT_ROOT redirects every output.
"""
from __future__ import annotations

import argparse
import filecmp
import os
import pickle
import subprocess
import sys
import tempfile
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import sigcommon as cm  # noqa: E402

STAGES = ["draws", "metareg", "convexity", "study", "pidset", "depmc", "ddtilt", "tables", "figures"]


def cache(name):
    return os.path.join(cm.PROC, f"stage_{name}.pkl")


def save(name, obj):
    with open(cache(name), "wb") as f:
        pickle.dump(obj, f)


def load(name):
    with open(cache(name), "rb") as f:
        return pickle.load(f)


def stage_draws():
    import jointboot as jb
    cm.log("stage draws: joint design-conditional bootstrap draws of sigma*_b (T1.1), top-budget set (T1.3)")
    R = jb.run_draws()
    lin, tops = jb.linear_stats(R)
    out = dict(R=R, check=jb.check_reproduction(R), budgets=jb.budget_table(R), corr=jb.correlations(R), lin=lin,
               tops=tops, topset=jb.top_budget_set(R))
    save("draws", out)
    return out


def stage_metareg():
    import metareg
    cm.log("stage metareg: meta-regression with Marin as one cluster (T1.2)")
    out = metareg.run()
    save("metareg", out)
    return out


def stage_convexity():
    import convexity
    cm.log("stage convexity: extrapolation decomposition, leave-corner-out, percentile intervals (T1.9, T16)")
    out = convexity.run()
    # the cache keeps the finished tables, not the 11 x 999 bootstrap draws (about 50 MB)
    out = dict(tables=convexity.tables(out), far_restricted=out["far_restricted"])
    save("convexity", out)
    return out


def stage_study():
    import study
    cm.log("stage study: study-level variants (T1.4-T1.6, T1.10)")
    out = study.run(load("draws")["R"], load("convexity")["far_restricted"])
    save("study", out)
    return out


def stage_pidset():
    import pidset
    cm.log("stage pidset: identified set for sigma*(C) beyond the designs (T1.7)")
    D, S = load("draws"), load("study")
    up = S["variants"][S["variants"].group == "T1.4"]
    out = pidset.run(D["lin"], D["tops"], up)
    save("pidset", out)
    return out


def stage_depmc():
    import depmc
    cm.log("stage depmc: coverage under noise shared within a budget (T1.8)")
    out = depmc.run()
    save("depmc", out)
    return out


def stage_ddtilt():
    import ddtilt
    cm.log("stage ddtilt: DataDecide recipe tilt from final checkpoints (T14)")
    out = ddtilt.run()
    save("ddtilt", out)
    return out


def headline(D, M, C, S, P, MC, DD, T):
    """Every number the memo quotes, with its placeholder or item and its source table."""
    rows = []

    def add(item, name, value, source, note=""):
        rows.append(dict(item=item, quantity=name, value=float(value) if np.isscalar(value) else value, source=source,
                         note=note))
    L = D["lin"].set_index("statistic")
    sl = L.loc[[i for i in L.index if i.startswith("Chinchilla + Llama 3 drift: design FE slope, inverse")][0]]
    add("T1.1", "CL3 FE drift per decade", sl.estimate, "rb5_sigma_joint_linear.csv")
    add("T1.1", "CL3 FE drift: model-based s.e. (scaled)", sl.se_model_scaled, "rb5_sigma_joint_linear.csv")
    add("T1.1", "CL3 FE drift: bootstrap s.e., independent budgets", sl.se_boot_independent, "rb5_sigma_joint_linear.csv")
    add("T1.1", "CL3 FE drift: bootstrap s.e., joint draws", sl.se_boot_joint, "rb5_sigma_joint_linear.csv")
    add("T1.1", "CL3 FE drift: joint bootstrap s.e. x sqrt(residual dispersion)", sl.se_boot_joint * np.sqrt(max(sl.resid_var_scale, 1)),
        "rb5_sigma_joint_linear.csv", f"residual dispersion s2 = {sl.resid_var_scale:.3f}")
    tp = L.loc[[i for i in L.index if i.startswith("Top-budget mean: Chinchilla + Llama 3, budgets >= 6e20 (primary)")][0]]
    add("T1.1", "top-budget mean", tp.estimate, "rb5_sigma_joint_linear.csv")
    add("T1.1", "top-budget mean: s.e. independent (= model-based)", tp.se_boot_independent, "rb5_sigma_joint_linear.csv")
    add("T1.1", "top-budget mean: s.e. joint", tp.se_boot_joint, "rb5_sigma_joint_linear.csv")
    from scipy.stats import t as tdist
    q4 = tdist.ppf(0.975, 4)
    add("T1.1", "top-budget mean: modified HKSJ interval with joint s.e., lo", tp.estimate - q4 * tp.se_boot_joint, "rb5_sigma_joint_linear.csv", "t_4")
    add("T1.1", "top-budget mean: modified HKSJ interval with joint s.e., hi", tp.estimate + q4 * tp.se_boot_joint, "rb5_sigma_joint_linear.csv", "t_4")
    add("T1.1", "top-budget mean: percentile interval, lo", tp.lo_percentile, "rb5_sigma_joint_linear.csv")
    add("T1.1", "top-budget mean: percentile interval, hi", tp.hi_percentile, "rb5_sigma_joint_linear.csv")
    add("T1.1", "top-budget mean: normal interval, lo", tp.lo_normal, "rb5_sigma_joint_linear.csv")
    add("T1.1", "top-budget mean: normal interval, hi", tp.hi_normal, "rb5_sigma_joint_linear.csv")
    add("T1.3", "Llama 3 weight share in the top-budget mean", tp.weight_share_llama3, "rb5_sigma_joint_linear.csv")
    ts = D["topset"].set_index("set")
    for k in ts.index:
        add("T1.3", f"{k}: fixed-effect mean", ts.loc[k, "mean_fixed"], "rb5_sigma_top_budget.csv")
        if ts.loc[k, "k"] > 1:
            add("T1.3", f"{k}: random-effects mean", ts.loc[k, "mean_re"], "rb5_sigma_top_budget.csv")
            add("T1.3", f"{k}: Q", ts.loc[k, "Q"], "rb5_sigma_top_budget.csv")
    pr = M["prange"].set_index("set")
    for k in pr.index:
        add("T1.2", f"p range [{k}]: min", pr.loc[k, "p_min"], "rb5_sigma_metareg_prange.csv")
        add("T1.2", f"p range [{k}]: max", pr.loc[k, "p_max"], "rb5_sigma_metareg_prange.csv")
    pp = M["pred"]
    for _, r in pp[pp.clusters.str.startswith("studies")].iterrows():
        add("T1.2", f"pooled line at 1e{r.log10C:.0f}, CR2 by study: estimate", r.est, "rb5_sigma_metareg_predictions.csv")
        add("T1.2", f"pooled line at 1e{r.log10C:.0f}, CR2 by study: lo", r.lo_cr2, "rb5_sigma_metareg_predictions.csv")
        add("T1.2", f"pooled line at 1e{r.log10C:.0f}, CR2 by study: hi", r.hi_cr2, "rb5_sigma_metareg_predictions.csv")
    V = S["variants"].set_index("variant")
    for k in V.index:
        if V.loc[k, "group"] in ("primary", "T1.4", "T1.6"):
            for c in ("mean_re", "lo_hksj", "hi_hksj", "Q", "p_Q"):
                add(V.loc[k, "group"], f"{k}: {c}", V.loc[k, c], "rb5_sigma_study_level.csv")
    rg = S["ranges"].set_index("family")
    for k in rg.index:
        for c in ("mean_min", "mean_max", "Q_min", "Q_max", "p_Q_min", "p_Q_max"):
            add("T1.5", f"{k}: {c}", rg.loc[k, c], "rb5_sigma_study_ranges.csv")
    pi = P["table"]
    for _, r in pi[pi.C.isin([1e23, 1e24])].iterrows():
        add("T1.7", f"lower bound at {r.C:.0e}, anchor {r.anchor}", r.sigma_lower, "rb5_sigma_pi_set.csv")
    for k, v in P["info"].items():
        add("T1.7", k, v, "rb5_sigma_pi_set.csv")
    mc = MC["table"]
    for _, r in mc.iterrows():
        add("T1.8", f"coverage RE, {r.design}, {r.noise}, rho {r.rho}", r.coverage_re, "rb5_sigma_mc_dependence.csv", f"R = {r.R}")
        add("T1.8", f"bias RE, {r.design}, {r.noise}, rho {r.rho}", r.bias_re, "rb5_sigma_mc_dependence.csv")
    dec = T["decomposition"]
    for _, r in dec[dec.M_bin.isin(["256-1,024", ">=1,024", "M>=256"])].iterrows():
        add("T1.9", f"convexity component, {r.design} [{r.variant}], {r.M_bin}", r.convexity, "rb5_sigma_extrap_decomposition.csv",
            f"95% percentile [{r.convexity_lo:.3f}, {r.convexity_hi:.3f}]; points {r.n_points}")
    ln = T["lin"]
    for _, r in ln.iterrows():
        add("T1.9", f"{r.param}, {r.design} [{r.variant}]", r.est, "rb5_sigma_extrap_lin.csv", f"[{r.lo_pct:.4f}, {r.hi_pct:.4f}]")
    gap = S["marin_gap"]
    add("T1.10", "Marin IsoFLOP curvature sigma*: min", gap.sigma_isoflop_curvature.min(), "rb5_sigma_marin_estimator_gap.csv")
    add("T1.10", "Marin IsoFLOP curvature sigma*: max", gap.sigma_isoflop_curvature.max(), "rb5_sigma_marin_estimator_gap.csv")
    add("T1.10", "Marin local first-derivative sigma*: min", gap.sigma_local_first_derivative.min(), "rb5_sigma_marin_estimator_gap.csv")
    add("T1.10", "Marin local first-derivative sigma*: max", gap.sigma_local_first_derivative.max(), "rb5_sigma_marin_estimator_gap.csv")
    dt = DD["table"]
    for _, r in dt.iterrows():
        for c in ("alpha", "beta", "tilt_range", "tau_own", "tau_ref", "tilt_range_lo", "tilt_range_hi", "tau_own_lo",
                  "tau_own_hi", "tau_ref_lo", "tau_ref_hi", "objective", "objective_minus_global",
                  "spearman_tilt_vs_allcheckpoints", "share_draws_beta_gt_alpha"):     # [review] mode diagnostics
            if c in r and pd.notna(r[c]):
                add("T14", f"{r['sample']} [{r.estimator}]: {c}", r[c], "rb5_sigma_dd_tilt.csv")
    return pd.DataFrame(rows)


def stage_tables():
    cm.log("stage tables: writing CSV outputs")
    D, M, C, S, P, MC, DD = (load(k) for k in ("draws", "metareg", "convexity", "study", "pidset", "depmc", "ddtilt"))
    import jointboot as jb
    cm.tab(D["check"], "joint_check")
    cm.tab(D["budgets"], "joint_budgets")
    cm.tab(jb.draws_long(D["R"]), "joint_draws")
    cm.tab(D["corr"], "joint_corr")
    cm.tab(D["lin"], "joint_linear")
    cm.tab(D["topset"], "top_budget")
    cm.tab(M["slopes"], "metareg_slopes")
    cm.tab(M["check"], "metareg_check")
    cm.tab(M["pred"], "metareg_predictions")
    cm.tab(M["prange"], "metareg_prange")
    T = C["tables"]
    cm.tab(T["decomposition"], "extrap_decomposition")
    cm.tab(T["lin"], "extrap_lin")
    cm.tab(T["delta_pct"], "extrap_delta_pct")
    cm.tab(T["corner"], "extrap_corner")
    cm.tab(T["check_rb1"], "extrap_check_rb1")
    cm.tab(T["check_farseer"], "extrap_check_farseer")
    fr = C["far_restricted"]
    cm.tab(pd.DataFrame([dict(estimate=k, **{kk: (vv if np.isscalar(vv) else str(vv)) for kk, vv in v.items()})
                         for k, v in fr.items()]), "farseer_restricted")
    cm.tab(S["variants"], "study_level")
    cm.tab(S["parts"], "study_parts_le3e20")
    cm.tab(S["ranges"], "study_ranges")
    cm.tab(pd.DataFrame([S["far_bw"]]), "farseer_bandwidth")
    cm.tab(S["marin_gap"], "marin_estimator_gap")
    cm.tab(pd.DataFrame([dict(statistic=k, rb5=v[0], rb1=v[1], abs_diff=abs(v[0] - v[1])) for k, v in S["check"].items()]),
           "study_check")
    cm.tab(P["table"], "pi_set")
    cm.tab(P["check"], "pi_check")
    cm.tab(MC["table"], "mc_dependence")
    cm.tab(MC["check"], "mc_check")
    cm.tab(DD["table"], "dd_tilt")
    cm.tab(DD["profile"], "dd_profile")
    cm.tab(DD["recipes"], "dd_recipes")
    cm.tab(pd.DataFrame([DD["info"]]), "dd_sample")
    H = headline(D, M, C, S, P, MC, DD, T)
    cm.tab(H, "headline")
    cm.log(f"  tables written to {cm.TABLES} ({len(H)} headline numbers)")


def stage_figures(final=False):
    import figs_rb5
    cm.log(f"stage figures ({'paper files' if final else 'previews'})")
    figs_rb5.make_all(final=final)


def determinism():
    """Re-run every stage into a scratch root; compare every rb5_sigma CSV byte for byte with the primary outputs."""
    root = tempfile.mkdtemp(prefix="rb5_sigma_det_")
    env = dict(os.environ, RB5_OUTPUT_ROOT=root)
    env.pop("RB5_SHADOW_ROOT", None)
    cmd = [sys.executable, os.path.abspath(__file__), "--stages"] + [s for s in STAGES if s != "figures"]
    cm.log(f"determinism check: re-running all stages into {root}")
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-3000:])
    a_dir, b_dir = cm.TABLES, os.path.join(root, "output", "tables")
    # [review] the reviewer's own outputs (rb5_sigma_review_*.csv, written by review_*.py) are not stages of run.py
    files = sorted(f for f in os.listdir(a_dir) if f.startswith(cm.PREFIX) and f.endswith(".csv")
                   and not f.startswith(cm.PREFIX + "_review_"))
    rows = []
    for f in files:
        same = os.path.exists(os.path.join(b_dir, f)) and filecmp.cmp(os.path.join(a_dir, f), os.path.join(b_dir, f), shallow=False)
        rows.append(dict(file=f, identical=bool(same)))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(cm.PROC, "determinism_check.csv"), index=False)
    cm.log(f"  determinism: {int(T.identical.sum())} of {len(T)} CSV files identical byte for byte; scratch root {root}")
    return T


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", default=STAGES)
    ap.add_argument("--final-figures", action="store_true")
    ap.add_argument("--determinism", action="store_true")
    a = ap.parse_args()
    if a.determinism:
        determinism()
        sys.exit(0)
    cm.log(f"rb5_sigma run: stages {a.stages}; QUICK={cm.QUICK}; output root {cm._OUT}; processes {cm.N_PROC}")
    for s in STAGES:
        if s in a.stages:
            if s == "figures":
                stage_figures(final=a.final_figures)
            else:
                globals()[f"stage_{s}"]()
    cm.log("done")
