"""run.py -- single entry point for module m8_measurement ("Input measurement and flexible inputs").

Regenerates every output of the module from raw data with fixed seeds:
  data/processed/m8_measurement/*.csv            intermediate results (cell-level aggregates only)
  output/tables/m8_measurement_*.csv / *.tex     tables
  output/figures/m8_measurement_*.pdf / *.png    figures
The memo output/memos/m8_measurement.md is written by hand from these outputs (numbers quoted there are
printed by this script in results_summary.json).

Parts
  A. Porian et al. (2024): reproduce the step-by-step path of the allocation exponent a; decompose
     measurement vs flexible-input steps; verify two bias formulas (measurement: omitted scale-declining
     component of N; flexible inputs: first-order envelope/argmin-shift formula).
  B. Pearce & Song (2024) mechanism: closed form for the measured local exponent (sympy + numerics), and a
     re-estimation of the Chinchilla extraction with non-embedding N.
  C. Step Law (Li et al. 2025): concentrated technology vs untuned / rule-based flexible inputs,
     inefficiency gradients, random-configuration Monte Carlo, heteroskedastic SFA, and flexible-input
     demand functions (unconditional vs batch-conditional LR demand; Le Chatelier decomposition).

Run:  /Users/yigitokar/scaling-laws-pf/.venv/bin/python code/analysis/m8_measurement/run.py
Runtime: about 25-30 minutes on 6 CPU processes on a heavily loaded machine (load average 30-45); less when idle.
"""
from __future__ import annotations

import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")   # one BLAS thread per process (<= 6 processes in total)

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PROC, TAB, ROOT, MINI_GRID, sl, fit_warm, fit_efixed, model_row, parallel_map, wls_loglog  # noqa: E402
import porian as P  # noqa: E402
import embed as EM  # noqa: E402
import steplaw as S  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
SUMMARY = {}


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.1f}s] {msg}", flush=True)


# ============================================================================ Part A: Porian

LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)


def _lambda_fit(args):
    """Huber-LSE Chinchilla fit with a quality-weighted parameter count N_lambda (tuned Porian runs)."""
    ds, lam, N, D, L = args
    m = sl.fit_chinchilla(N, D, L, grid=sl.FAST_GRID)
    return dict(dataset=ds, lam=lam, objective=m.extra["objective"], n=len(L), a=m.a_N, E=m.E,
                alpha=m.alpha, beta=m.beta, sigma_star=m.sigma_star)


def part_a():
    log("A. loading Porian runs")
    df = P.load_porian()
    SUMMARY["porian_n_runs"] = int(len(df))
    SUMMARY["porian_runs_by_config"] = {"/".join(k): int(v) for k, v in
                                        df.groupby(["dataset", "hparams", "warmup", "decay"]).size().items()}

    log("A1. five-step path (RW + OWT2) and Kaplan-adjusted")
    steps, perc, pts = P.porian_steps(df, iters=1000)
    steps.to_csv(os.path.join(TAB, "m8_measurement_porian_steps.csv"), index=False)
    perc.to_csv(os.path.join(PROC, "porian_isoflop_argmins.csv"), index=False)

    log("A2. measurement-only grid (same runs, different N/C counting conventions)")
    meas = P.measurement_grid(df, iters=1000)
    meas.to_csv(os.path.join(TAB, "m8_measurement_porian_counting.csv"), index=False)

    log("A3. measurement bias formula on Porian's architecture ladder")
    archs, theta_head = P.architecture_shares(df)
    archs.to_csv(os.path.join(PROC, "porian_architectures.csv"), index=False)
    c0 = np.polyfit(np.log(archs.params_no_embed), np.log(archs["head"]), 1)
    omega_head = float(np.exp(c0[1]))
    SUMMARY["porian_head_share"] = dict(theta=float(theta_head), omega=omega_head,
                                        s_min=float(archs.s_head.min()), s_max=float(archs.s_head.max()),
                                        N_std_min=float(archs.params.min()), N_std_max=float(archs.params.max()))
    rows = []
    run_sets = [("Kaplan setup (untuned, long warmup)", "base", "long", "kaplan"),
                ("Tuned (short warmup, const LR)", "tuned", "short", "const")]
    for ds in ("rw", "owt2"):
        for rlab, hp, wu, dc in run_sets:
            o_std, _, pt_std = P.run_config(df, ds, hp, wu, dc, "standard", "train", iters=200)
            o_kap = meas.query("dataset==@ds and run_set==@rlab and count=='kaplan'").iloc[0]
            o_std_m = meas.query("dataset==@ds and run_set==@rlab and count=='standard'").iloc[0]
            tech = sl.fit_chinchilla(pt_std.n.values, pt_std.t.values, pt_std.loss.values, grid=sl.FAST_GRID)
            # researcher's path in the Kaplan count over the observed measured-compute range
            xs = np.linspace(np.log(archs.params_no_embed.min()) - 1, np.log(archs.params_no_embed.max()) + 3, 400)
            cm = np.log(EM.researcher_path(tech, omega_head, theta_head, np.exp(xs)))
            sel = (cm >= np.log(o_kap.C_min)) & (cm <= np.log(o_kap.C_max))
            a_pred = np.polyfit(cm[sel], xs[sel], 1)[0] if sel.sum() > 5 else np.nan
            # semi-synthetic: same design and pipeline, losses replaced by the smooth technology
            syn = P.synthetic_count_check(df, ds, hp, wu, dc, tech)
            # robustness: technology refitted with E held at 1.8 (the free-E fits put E near or above the
            # lowest observed losses, i.e. E is not identified from these small-scale runs)
            tech18 = fit_efixed(pt_std.n.values, pt_std.t.values, pt_std.loss.values, 1.8, grid=sl.FAST_GRID)
            syn18 = P.synthetic_count_check(df, ds, hp, wu, dc, tech18)
            rows.append(dict(dataset=ds, run_set=rlab, tech_E=tech.E, tech_alpha=tech.alpha, tech_beta=tech.beta,
                             a_tech=tech.a_N, a_std_obs=o_std_m.a, a_std_obs_se=o_std_m.a_se,
                             a_kaplan_obs=o_kap.a, a_kaplan_obs_se=o_kap.a_se,
                             a_kaplan_pred=a_pred, delta_obs=o_kap.a - o_std_m.a, delta_pred=a_pred - tech.a_N,
                             a_std_synth=syn["standard"], a_kaplan_synth=syn["kaplan"],
                             delta_synth=syn["kaplan"] - syn["standard"],
                             a_std_synth_E18=syn18["standard"], a_kaplan_synth_E18=syn18["kaplan"],
                             delta_synth_E18=syn18["kaplan"] - syn18["standard"], min_loss_std=float(pt_std.loss.min()),
                             Cm_min=o_kap.C_min, Cm_max=o_kap.C_max))
    meas_formula = pd.DataFrame(rows)
    meas_formula.to_csv(os.path.join(TAB, "m8_measurement_porian_meas_formula.csv"), index=False)

    log("A4. flexible-input bias formula (argmin shift = -Delta / f'')")
    ineff_rows, ineff_detail = [], []
    for ds in ("rw", "owt2"):
        pc5 = perc[(perc.dataset == ds) & (perc.step == 5)]
        pt5 = pts[(ds, 5)]
        tech5 = sl.fit_chinchilla(pt5.n.values, pt5.t.values, pt5.loss.values, grid=sl.FAST_GRID)
        a5 = steps[(steps.dataset == ds) & (steps.step == 5)].a.iloc[0]
        for sid in (2, 3, 4):
            pc = perc[(perc.dataset == ds) & (perc.step == sid)]
            g = P.inefficiency_gradient(pts[(ds, sid)], pt5, pc, pc5, tech5)
            g["dataset"], g["step"] = ds, sid
            ineff_detail.append(g)
            lc = np.log(g.C.values)
            a_s = steps[(steps.dataset == ds) & (steps.step == sid)].a.iloc[0]
            ok = np.isfinite(g.shift_pred_local.values)
            ineff_rows.append(dict(dataset=ds, step=sid, a_step=a_s, a_tuned=a5, delta_a_obs=a_s - a5,
                                   slope_shift_obs=np.polyfit(lc, g.shift_obs.values, 1)[0],
                                   delta_a_pred_model=np.polyfit(lc, g.shift_pred_model.values, 1)[0],
                                   delta_a_pred_local=np.polyfit(lc[ok], g.shift_pred_local.values[ok], 1)[0],
                                   delta_a_pred_local_bw05=np.polyfit(lc[np.isfinite(g.shift_pred_local05.values)],
                                                                      g.shift_pred_local05.values[np.isfinite(g.shift_pred_local05.values)], 1)[0],
                                   min_loss_tuned=float(pt5.loss.min()),
                                   n_budgets=len(g), tech5_E=tech5.E, tech5_alpha=tech5.alpha, tech5_beta=tech5.beta,
                                   tech5_a=tech5.a_N))
    ineff = pd.DataFrame(ineff_rows)
    ineff.to_csv(os.path.join(TAB, "m8_measurement_porian_flex_formula.csv"), index=False)
    pd.concat(ineff_detail).to_csv(os.path.join(PROC, "porian_ineff_by_budget.csv"), index=False)
    log("A5. quality-weighted head count: profile of N_lambda = N_noembed + lambda * head (tuned runs)")
    arch = df[["params_no_embed", "params"]].drop_duplicates().set_index("params")["params_no_embed"]
    items = []
    for ds in ("rw", "owt2"):
        pt5 = pts[(ds, 5)]
        Nm = arch.loc[pt5.n.values].values
        H = pt5.n.values - Nm
        for lam in LAMBDAS:
            items.append((ds, lam, Nm + lam * H, pt5.t.values, pt5.loss.values))
    lam_prof = pd.DataFrame(parallel_map(_lambda_fit, items))
    lam_prof.to_csv(os.path.join(TAB, "m8_measurement_head_weight_profile.csv"), index=False)
    SUMMARY["head_weight_profile"] = lam_prof.to_dict("records")
    SUMMARY["porian_steps"] = steps[["dataset", "step", "a", "a_lo", "a_hi", "a_porian", "N_at_chinchilla"]].to_dict("records")
    return dict(steps=steps, perc=perc, meas=meas, meas_formula=meas_formula, ineff=ineff,
                ineff_detail=pd.concat(ineff_detail), archs=archs)


# ============================================================================ Part B: Pearce-Song mechanism

def _boot_chin(args):
    """Pairs bootstrap draw: warm start + MINI_GRID; the first draws are re-fitted with sl.FAST_GRID as a check."""
    i, df, init_T, init_E, check = args
    rng = np.random.default_rng(5000 + i)
    idx = rng.integers(0, len(df), len(df))
    s = df.iloc[idx]
    mT = fit_warm(s.N.values, s.D.values, s.L.values, init=init_T, grid=MINI_GRID)
    mE = fit_warm(s.NE.values, s.D.values, s.L.values, init=init_E, grid=MINI_GRID)
    chk = np.nan
    if check:
        cT = fit_warm(s.N.values, s.D.values, s.L.values, init=init_T, grid=sl.FAST_GRID)
        cE = fit_warm(s.NE.values, s.D.values, s.L.values, init=init_E, grid=sl.FAST_GRID)
        chk = max(abs(cT.a_N - mT.a_N), abs(cE.a_N - mE.a_N))
    return dict(i=i, check_abs_diff_a=chk, a_T=mT.a_N, a_E=mE.a_N, alpha_T=mT.alpha, alpha_E=mE.alpha, beta_T=mT.beta, beta_E=mE.beta,
                sigma_T=mT.sigma_star, sigma_E=mE.sigma_star, gamma_T=mT.gamma, gamma_E=mE.gamma)


def part_b(B=200):
    log("B1. symbolic check of the measured-exponent formula")
    diff, worst = EM.symbolic_check()
    SUMMARY["pearce_song_symbolic_diff"] = diff
    SUMMARY["pearce_song_numeric_worst_abs_err"] = worst

    log("B2. Pearce-Song simulation over Kaplan's range")
    ps_tab, ps_curves = EM.pearce_song_table()
    ps_tab.to_csv(os.path.join(TAB, "m8_measurement_pearce_song_sim.csv"), index=False)
    # wider curve for the figure (1e3 .. 1e12 non-embedding parameters)
    curves = []
    for name, t in {"Besiroglu et al. (2024)": sl.BESIROGLU, "Hoffmann et al. (2022)": sl.HOFFMANN}.items():
        cf = EM.closed_form_path(t, EM.OMEGA_PS, EM.THETA_PS, np.logspace(2.5, 12, 300))
        cf["param_set"] = name
        curves.append(cf)
    pd.concat(curves).to_csv(os.path.join(PROC, "pearce_song_local_exponent_curves.csv"), index=False)

    log("B3. Chinchilla extraction re-estimated with non-embedding N (N_T = N_E + omega N_E^(1/3))")
    df = sl.chinchilla_extraction(os.path.join(ROOT, "data/raw/epoch_chinchilla/svg_extracted_data.csv"))
    df["NE"] = EM.nonembed_from_total(df.N.values)
    df["share_embed"] = 1 - df.NE / df.N
    # FAST_GRID + warm start at the published Besiroglu values (identical optimum to DEFAULT_GRID on this sample:
    # a = 0.5139 either way; DEFAULT_GRID is ~20x slower)
    mT = fit_warm(df.N, df.D, df.L, init=sl.BESIROGLU.theta, grid=sl.FAST_GRID)
    mE = fit_warm(df.NE, df.D, df.L, init=sl.BESIROGLU.theta, grid=sl.FAST_GRID)
    boots = pd.DataFrame(parallel_map(_boot_chin, [(i, df, mT.theta, mE.theta, i < 6) for i in range(B)]))
    SUMMARY["chinchilla_boot_grid_check_max_abs_diff_a"] = float(boots.check_abs_diff_a.max())
    boots.to_csv(os.path.join(PROC, "chinchilla_nonembed_bootstrap.csv"), index=False)
    rows = []
    for lab, m, key in (("Total N (Besiroglu sample)", mT, "T"), ("Non-embedding N (Pearce-Song map)", mE, "E")):
        rows.append(model_row(m, sample="all (n=240)", measure=lab,
                              a_se=boots[f"a_{key}"].std(ddof=1), alpha_se=boots[f"alpha_{key}"].std(ddof=1),
                              beta_se=boots[f"beta_{key}"].std(ddof=1), sigma_star_se=boots[f"sigma_{key}"].std(ddof=1),
                              gamma_se=boots[f"gamma_{key}"].std(ddof=1)))
    diff_a = boots.a_E - boots.a_T
    SUMMARY["chinchilla_nonembed"] = dict(a_T=mT.a_N, a_E=mE.a_N, diff=mE.a_N - mT.a_N, diff_se=float(diff_a.std(ddof=1)),
                                          diff_ci=[float(diff_a.quantile(.025)), float(diff_a.quantile(.975))],
                                          share_embed_range=[float(df.share_embed.min()), float(df.share_embed.max())],
                                          sigma_T=mT.sigma_star, sigma_E=mE.sigma_star,
                                          sigma_diff_se=float((boots.sigma_E - boots.sigma_T).std(ddof=1)))
    # small-model subsamples (Kaplan-like range): point estimates only (5-parameter fits on <120 points)
    for cut in (3e8, 1e9, 2e9):
        s = df[df.N < cut]
        a = sl.fit_chinchilla(s.N, s.D, s.L, grid=sl.FAST_GRID)
        b = sl.fit_chinchilla(s.NE, s.D, s.L, grid=sl.FAST_GRID)
        rows.append(model_row(a, sample=f"N_T < {cut:.0e} (n={len(s)})", measure="Total N (Besiroglu sample)"))
        rows.append(model_row(b, sample=f"N_T < {cut:.0e} (n={len(s)})", measure="Non-embedding N (Pearce-Song map)"))
    refit = pd.DataFrame(rows)
    refit.to_csv(os.path.join(TAB, "m8_measurement_chinchilla_nonembed.csv"), index=False)
    SUMMARY["pearce_song_sim"] = ps_tab.to_dict("records")
    return dict(ps_tab=ps_tab, refit=refit, curves=pd.concat(curves))


# ============================================================================ Part C: Step Law

def _eprofile_one(args):
    ps, E = args
    m = fit_efixed(ps.N.values, ps.D.values, ps.L.values, E, delta=None, grid=sl.FAST_GRID)
    return dict(E=E, a=m.a_N, sigma_star=m.sigma_star, alpha=m.alpha, beta=m.beta, gamma=m.gamma,
                objective=m.extra["objective"])


def _resid_boot_one(args):
    """Residual bootstrap of the frontier NLS fit (design held fixed; residuals resampled)."""
    i, ps, fitted, resid, init = args
    rng = np.random.default_rng(7000 + i)
    L = fitted * np.exp(rng.choice(resid, len(resid), replace=True))
    mf = fit_warm(ps.N.values, ps.D.values, L, init=init, delta=None, grid=MINI_GRID)
    th = np.array(init, float).copy()
    th[2] = np.log(1.4)
    m14 = fit_efixed(ps.N.values, ps.D.values, L, 1.4, delta=None, init=th, grid=MINI_GRID)
    return dict(i=i, a_free=mf.a_N, sigma_free=mf.sigma_star, E_free=mf.E, a_14=m14.a_N, sigma_14=m14.sigma_star)


def stability_comparison(prof, pol, rb, cnt, mc):
    """REVIEW ADDITION. Is sigma* 'more stable' than a? sigma* = 2/(2+alpha+beta) compresses alpha+beta by roughly
    sigma*^2/2 ~ 0.3, so natural-unit ranges are not comparable. For each source of variation this reports the
    spread of a, sigma*, alpha+beta (the parameter sigma* is a monotone transform of) and gamma, in natural units,
    relative to the midpoint (half-range / midpoint, 'rel'), and in units of the frontier residual-bootstrap SE."""
    rows = []

    def add(source, a, s, apb, g, stat="range"):
        a, s, apb, g = map(np.asarray, (a, s, apb, g))
        if stat == "range":
            spread = lambda x: float(np.nanmax(x) - np.nanmin(x))
            rel = lambda x: float((np.nanmax(x) - np.nanmin(x)) / (np.nanmax(x) + np.nanmin(x)))
        else:   # dispersion across Monte Carlo draws: s.d. and coefficient of variation
            spread = lambda x: float(np.nanstd(x, ddof=1))
            rel = lambda x: float(np.nanstd(x, ddof=1) / np.nanmedian(x))
        rows.append(dict(source=source, stat=stat, a=spread(a), sigma_star=spread(s), alpha_plus_beta=spread(apb),
                         gamma=spread(g), a_rel=rel(a), sigma_star_rel=rel(s), alpha_plus_beta_rel=rel(apb), gamma_rel=rel(g),
                         a_in_se=spread(a) / rb["a_14"]["sd"], sigma_star_in_se=spread(s) / rb["sigma_14"]["sd"]))

    f = prof[(prof.rule == "frontier") & (prof.E >= 0.59) & (prof.E <= 1.71)]
    add("E profile, frontier, E in [0.6, 1.7]", f.a, f.sigma_star, f.alpha + f.beta, f.gamma)
    p = pol[pol.rule.isin(["frontier", "steplaw", "best_fixed", "porian_base", "porian_rule", "deepseek", "bjorck"])]
    add("policies incl. frontier, E = 1.4", p["E1.4_a"], p["E1.4_sigma_star"], p["E1.4_alpha"] + p["E1.4_beta"],
        p["E1.4_alpha"] * p["E1.4_beta"] / (p["E1.4_alpha"] + p["E1.4_beta"]))
    add("N counting convention, frontier, E free", cnt.free_a, cnt.free_sigma_star, 2 / cnt.free_sigma_star - 2, cnt.free_gamma)
    add("N counting convention, frontier, E = 1.4", cnt.E14_a, cnt.E14_sigma_star, 2 / cnt.E14_sigma_star - 2, cnt.E14_gamma)
    fr = pol[pol.rule == "frontier"].iloc[0]
    for k, g in mc.groupby("k"):
        apb = g["E1.4_alpha"] + g["E1.4_beta"]
        add(f"random configurations, best of {k}, E = 1.4 (MC)", g["E1.4_a"], g["E1.4_sigma_star"], apb,
            g["E1.4_alpha"] * g["E1.4_beta"] / apb, stat="mc_sd")
        rows[-1].update(median_bias_a=float(g["E1.4_a"].median() - fr["E1.4_a"]),
                        median_bias_sigma_star=float(g["E1.4_sigma_star"].median() - fr["E1.4_sigma_star"]),
                        median_bias_a_in_se=float((g["E1.4_a"].median() - fr["E1.4_a"]) / rb["a_14"]["sd"]),
                        median_bias_sigma_star_in_se=float((g["E1.4_sigma_star"].median() - fr["E1.4_sigma_star"]) / rb["sigma_14"]["sd"]))
    return pd.DataFrame(rows)


def part_c(R_mc=200, B_resid=200):
    log("C1. Step Law: cells, frontier, within-cell optima")
    d = S.load_steplaw()
    c = S.cells(d)
    c.to_csv(os.path.join(PROC, "steplaw_cells.csv"), index=False)
    SUMMARY["steplaw_n_runs"] = int(len(d))
    SUMMARY["steplaw_n_cells"] = int(d.cell.nunique())
    SUMMARY["steplaw_share_diverged"] = float(d.diverged.mean())
    opt, draws = S.cell_optima(d)
    opt.to_csv(os.path.join(PROC, "steplaw_cell_optima.csv"), index=False)

    log("C2. flexible-input demand functions")
    dem = S.demand_functions(opt, draws)
    cond = S.conditional_lr_demand(d)
    cond.to_csv(os.path.join(PROC, "steplaw_conditional_lr_slices.csv"), index=False)
    cc = cond[cond.interior].assign(lnN=lambda x: np.log(x.N), lnD=lambda x: np.log(x.D),
                                    lnB=lambda x: np.log(x.bs * S.SEQ))
    b, se, G = S.cluster_ols(cc, "lnlr_star", ["lnN", "lnD", "lnB"])
    cg = cond[~cond.grid_edge].assign(lnN=lambda x: np.log(x.N), lnD=lambda x: np.log(x.D),
                                      lnB=lambda x: np.log(x.bs * S.SEQ))
    bg, seg, Gg = S.cluster_ols(cg, "lnlr_grid", ["lnN", "lnD", "lnB"])
    s256 = cc[cc.bs == 256]
    b2, se2, G2 = S.cluster_ols(s256, "lnlr_star", ["lnN", "lnD"])
    extra = [dict(var="lr", method="conditional on batch (smoothed, pooled slices)", n=len(cc), const=b[0], const_se=se[0],
                  e_N=b[1], e_N_se=se[1], e_D=b[2], e_D_se=se[2], e_B=b[3], e_B_se=se[3], clusters=G),
             dict(var="lr", method="conditional on batch (grid argmin, pooled slices)", n=len(cg), const=bg[0], const_se=seg[0],
                  e_N=bg[1], e_N_se=seg[1], e_D=bg[2], e_D_se=seg[2], e_B=bg[3], e_B_se=seg[3], clusters=Gg),
             dict(var="lr", method="conditional at batch = 0.52M tokens (smoothed)", n=len(s256), const=b2[0], const_se=se2[0],
                  e_N=b2[1], e_N_se=se2[1], e_D=b2[2], e_D_se=se2[2], clusters=G2)]
    dem = pd.concat([dem, pd.DataFrame(extra)], ignore_index=True)
    # Le Chatelier / conditional-vs-unconditional decomposition of the D-elasticity of LR*:
    #   d ln LR*/d ln D (unconditional) = e_D|B + e_B * d ln B*/d ln D
    unc = dem[(dem["var"] == "lr") & (dem.method == "smoothed argmin")].iloc[0]
    bsd = dem[(dem["var"] == "bs") & (dem.method == "smoothed argmin")].iloc[0]
    decomp = dict(e_D_unconditional=unc.e_D, e_D_conditional=b[2], e_B=b[3], f_D=bsd.e_D,
                  implied_unconditional=b[2] + b[3] * bsd.e_D, share_from_batch=(b[3] * bsd.e_D) / unc.e_D,
                  e_N_unconditional=unc.e_N, e_N_conditional=b[1], f_N=bsd.e_N,
                  implied_unconditional_N=b[1] + b[3] * bsd.e_N)
    SUMMARY["lr_decomposition"] = decomp
    # REVIEW ADDITION: cell-cluster bootstrap of the decomposition, smoothed and grid-argmin optima (the share of the
    # unconditional D-elasticity due to batch co-scaling is sensitive to how the optimum is located)
    SUMMARY["lr_decomposition_boot"] = S.lechatelier_bootstrap(opt, cond, B=1000, seed=3)
    # Step Law's own 1,000 bootstrap fits (published file) for comparison
    sb = pd.read_csv(os.path.join(ROOT, "data/raw/steplaw/1004_fitted_lr_bs_scaling_model_parameters.csv"))
    SUMMARY["steplaw_published_bootstrap"] = {k: dict(mean=float(sb[k].mean()), sd=float(sb[k].std(ddof=1)))
                                              for k in sb.columns}
    dem.to_csv(os.path.join(TAB, "m8_measurement_demand.csv"), index=False)

    log("C3. flexible-input policies: inefficiency gradients and technology fits")
    lr_b, bs_b, fixed_tab = S.best_fixed_pair(d)
    SUMMARY["best_fixed_pair"] = dict(lr=lr_b, bs_seq=bs_b)
    rules = ["frontier", "steplaw", "best_fixed", "porian_base", "porian_rule", "deepseek", "bjorck"]
    pol_rows, samples = [], {}
    fr = S.policy_sample(d, "frontier")
    m_fr = sl.fit_chinchilla(fr.N.values, fr.D.values, fr.L.values, delta=None)            # DEFAULT_GRID
    m_fr_h = sl.fit_chinchilla(fr.N.values, fr.D.values, fr.L.values)                      # Huber, DEFAULT_GRID
    for r in rules:
        ps = S.policy_sample(d, r, best_fixed=(lr_b, bs_b))
        samples[r] = ps
        fits = S.fit_policy(ps, init=m_fr.theta)
        row = dict(rule=r, label=S.RULE_LABELS[r], max_snap_log2=ps.snap_dist.max(), **S.ineff_gradient(ps))
        for k, m in fits.items():
            tag = "free" if k == "free" else f"E{k:.1f}"
            row.update({f"{tag}_E": m.E, f"{tag}_alpha": m.alpha, f"{tag}_beta": m.beta, f"{tag}_a": m.a_N,
                        f"{tag}_gamma": m.gamma, f"{tag}_sigma_star": m.sigma_star})
        pol_rows.append(row)
        ps.to_csv(os.path.join(PROC, f"steplaw_policy_{r}.csv"), index=False)
    pol = pd.DataFrame(pol_rows)
    SUMMARY["steplaw_frontier_huber"] = m_fr_h.summary()
    SUMMARY["steplaw_frontier_nls"] = m_fr.summary()

    log("C4. residual bootstrap of the frontier fit")
    fitted = m_fr.loss(fr.N.values, fr.D.values)
    resid = np.log(fr.L.values) - np.log(fitted)
    # REVIEW FIX: centre the residuals and inflate them by sqrt(n/(n-p)), p = 5 fitted parameters, before resampling
    # (raw NLS residuals from 17 points and 5 parameters understate the error variance by the factor (n-p)/n)
    resid = (resid - resid.mean()) * np.sqrt(len(resid) / (len(resid) - 5))
    rb = pd.DataFrame(parallel_map(_resid_boot_one, [(i, fr, fitted, resid, m_fr.theta) for i in range(B_resid)]))
    rb.to_csv(os.path.join(PROC, "steplaw_frontier_resid_bootstrap.csv"), index=False)
    SUMMARY["steplaw_frontier_resid_boot"] = {k: dict(sd=float(rb[k].std(ddof=1)), lo=float(rb[k].quantile(.05)),
                                                      hi=float(rb[k].quantile(.95))) for k in rb.columns if k != "i"}

    log("C5. E-profiles of a and sigma* by policy")
    prof_items = [(samples[r], E) for r in ("frontier", "porian_base", "porian_rule", "best_fixed")
                  for E in np.round(np.arange(0.6, 1.95, 0.1), 2)]
    prof = pd.DataFrame(parallel_map(_eprofile_one, prof_items))
    prof["rule"] = [r for r in ("frontier", "porian_base", "porian_rule", "best_fixed") for _ in np.arange(0.6, 1.95, 0.1)]
    prof.to_csv(os.path.join(PROC, "steplaw_E_profiles.csv"), index=False)

    log("C6. random-configuration Monte Carlo (k random non-diverged configs per cell, keep best)")
    mc = S.random_config_mc(d, ks=(1, 4, 16), R=R_mc, init=m_fr.theta)
    mc.to_csv(os.path.join(PROC, "steplaw_random_config_mc.csv"), index=False)
    for k, g in mc.groupby("k"):
        row = dict(rule=f"random_k{k}", label=f"Best of {k} random configuration(s) per cell (MC, R={R_mc})",
                   max_snap_log2=np.nan, u_mean=g.ineff_u_mean.mean(), u_n=g.ineff_u_n.mean(), u_n_se=g.ineff_u_n.std(ddof=1),
                   u_d=g.ineff_u_d.mean(), u_d_se=g.ineff_u_d.std(ddof=1))
        for tag in ("free", "E1.4"):
            for p in ("E", "alpha", "beta", "a", "sigma_star"):
                row[f"{tag}_{p}"] = g[f"{tag}_{p}"].median()
            row[f"{tag}_a_sd"] = g[f"{tag}_a"].std(ddof=1)
            row[f"{tag}_sigma_star_sd"] = g[f"{tag}_sigma_star"].std(ddof=1)
            row[f"{tag}_a_p05"], row[f"{tag}_a_p95"] = g[f"{tag}_a"].quantile(.05), g[f"{tag}_a"].quantile(.95)
        pol = pd.concat([pol, pd.DataFrame([row])], ignore_index=True)
    pol.to_csv(os.path.join(TAB, "m8_measurement_steplaw_policies.csv"), index=False)

    log("C7. stochastic frontier (cell fixed effects, heteroskedastic inefficiency)")
    nd = d[~d.diverged]
    sv_cal = float(np.median(opt.resid_sd))
    sfa_rows = []
    for dist in ("halfnormal", "exponential"):
        for svf in (None, sv_cal):
            r = S.sfa_cell_fe(d, sample=nd, dist=dist, sv_fixed=svf)
            dev = r["mu"] - np.log(fr.L.values)
            # REVIEW ADDITION: cluster-robust (by cell) sandwich SEs for the variance-function coefficients
            cse = S.sfa_cluster_se(nd, r, dist) if svf is None else np.full(3, np.nan)
            sfa_rows.append(dict(dist=dist, sigma_v=("calibrated %.4f" % sv_cal) if svf else "estimated",
                                 sv=r["sv"], g0=r["g"][0], g0_se=r["g_se"][0], g_n=r["g"][1], g_n_se=r["g_se"][1],
                                 g_d=r["g"][2], g_d_se=r["g_se"][2], g0_cse=cse[0], g_n_cse=cse[1], g_d_cse=cse[2],
                                 nll=r["nll"], n=r["n"], converged=r["converged"],
                                 mean_abs_dev_frontier=float(np.mean(np.abs(dev))), max_abs_dev_frontier=float(np.max(np.abs(dev)))))
            if dist == "halfnormal" and svf is None:
                sfa_mu = pd.DataFrame(dict(cell=range(len(r["mu"])), mu=r["mu"], mu_se=r["mu_se"], ln_min=np.log(fr.L.values)))
    sfa = pd.DataFrame(sfa_rows)
    # REVIEW ADDITION: transparent cell-level cross-check (ln mean iota on ln N, ln D; n = 17, HC1)
    cl = S.cell_level_ineff_regression(nd)
    SUMMARY["sfa_cell_level_check"] = cl
    sfa = pd.concat([sfa, pd.DataFrame([dict(dist="cell-level OLS of ln mean iota", sigma_v="--", sv=np.nan,
                                              g0=cl["g0"], g0_se=cl["g0_se"], g_n=cl["g_n"], g_n_se=cl["g_n_se"],
                                              g_d=cl["g_d"], g_d_se=cl["g_d_se"], n=cl["n"])])], ignore_index=True)
    sfa.to_csv(os.path.join(TAB, "m8_measurement_sfa.csv"), index=False)
    sfa_mu.to_csv(os.path.join(PROC, "steplaw_sfa_frontier.csv"), index=False)
    # random-config expected inefficiency (all non-diverged runs): cluster-robust gradient
    b_u, se_u, G_u = S.cluster_ols(nd, "u", ["lnN", "lnD"])
    SUMMARY["random_config_u_gradient"] = dict(const=b_u[0], u_n=b_u[1], u_n_se=se_u[1], u_d=b_u[2], u_d_se=se_u[2],
                                               clusters=G_u, n=len(nd), u_mean=float(nd.u.mean()))
    SUMMARY["sfa"] = sfa.to_dict("records")

    log("C8. (review) Step Law N counting convention and the stability of a vs sigma*")
    cnt = S.frontier_count_robustness(d)
    cnt.to_csv(os.path.join(TAB, "m8_measurement_steplaw_count.csv"), index=False)
    SUMMARY["steplaw_count_robustness"] = cnt.to_dict("records")
    stab = stability_comparison(prof, pol, SUMMARY["steplaw_frontier_resid_boot"], cnt, mc)
    stab.to_csv(os.path.join(TAB, "m8_measurement_stability.csv"), index=False)
    SUMMARY["stability_comparison"] = stab.to_dict("records")
    return dict(d=d, cells=c, opt=opt, dem=dem, cond=cond, decomp=decomp, pol=pol, prof=prof, mc=mc, sfa=sfa,
                samples=samples, m_fr=m_fr, rb=rb, fixed_tab=fixed_tab)


def main():
    """Plain run recomputes everything. Development flags: --cache writes the part results to a pickle
    (it contains raw Step Law rows, which must not be redistributed, so it is off by default); --reuse loads it
    and only redoes the verification, tables and figures."""
    import pickle
    reuse = "--reuse" in sys.argv
    cache = os.path.join(PROC, "_cache_parts.pkl")
    if reuse and os.path.exists(cache):
        with open(cache, "rb") as f:
            A, B, C, saved = pickle.load(f)
        SUMMARY.update(saved)
    else:
        A = part_a()
        B = part_b()
        C = part_c()
        if "--cache" in sys.argv:
            with open(cache, "wb") as f:
                pickle.dump((A, B, C, SUMMARY), f)
    log("D. verification of the flexible-input bias formula (simulation)")
    import theory_checks
    tc = theory_checks.isoflop_bias_sim()
    tc.to_csv(os.path.join(TAB, "m8_measurement_ineff_formula_check.csv"), index=False)
    SUMMARY["ineff_formula_check_max_abs_err"] = float(np.max(np.abs(tc.a_isoflop_sim - tc.a_isoflop_formula)))
    # implied IsoFLOP bias for a researcher using random Step Law configurations (constant log-gradient case),
    # evaluated with the frontier technology at E = 1.4 (E is not identified in the Step Law design)
    g = SUMMARY["random_config_u_gradient"]
    pol = C["pol"].set_index("rule")
    al, be = pol.loc["frontier", "E1.4_alpha"], pol.loc["frontier", "E1.4_beta"]
    Rbar = float(np.mean(C["samples"]["frontier"].L.values)) - 1.4
    SUMMARY["steplaw_random_implied_isoflop_bias"] = dict(
        E=1.4, R=Rbar, alpha=al, beta=be, u_n=g["u_n"], u_d=g["u_d"],
        bias=-(g["u_n"] - g["u_d"]) * 1.4 / ((al + be) * Rbar))
    import tables
    import figures
    log("tables")
    tables.make_all(A, B, C, SUMMARY)
    log("figures")
    figures.make_all(A, B, C, SUMMARY)
    with open(os.path.join(PROC, "results_summary.json"), "w") as f:
        json.dump(SUMMARY, f, indent=1, default=float)
    log("done")


if __name__ == "__main__":
    main()
