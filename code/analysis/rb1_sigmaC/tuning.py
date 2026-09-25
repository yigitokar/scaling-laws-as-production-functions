"""tuning.py -- how much of Farseer's convexity of ln w in ln(M/M*(C)) could be hyperparameter mis-tuning (task 5;
R2 Major 5.2; Online Appendix D6 flexible-input bias formula; module m8_measurement's Step Law gradients).

Setup (m8 Proposition F; Online Appendix D6). Observed loss L_obs = L* e^{iota(N, D)}, L* the technology with the
hyperparameters concentrated out, iota >= 0 the inefficiency of the hyperparameter policy, with log-gradients
iota_n = d iota / d ln N and iota_d = d iota / d ln D. Farseer's runs follow the Step Law rules. The observed local
elasticities are eps_N^obs = eps_N - iota_n and eps_D^obs = eps_D - iota_d, so the technology's wedge is
        ln w = ln(eps_N^obs + iota_n) - ln(eps_D^obs + iota_d),
evaluated at every point of ra1's Farseer grid with the local-quadratic gradients (primary CV bandwidth). On the
expansion path this is D6's first-order argmin shift -Delta/f'' with Delta = L(iota_n - iota_d) (checked below);
off the path it is the same correction applied to the wedge. The technology's path M*_true(C) is re-located as the
root of the corrected ln w on each isocost, and the linearity regression ln w = b1 u + b2 u^2 (u = ln(M/M*(C))) is
re-run. Tuning share of the convexity = (b2_obs - b2_true) / b2_obs.

Scenarios (iota_n, iota_d), all measured on the Step Law grid (m8_measurement_steplaw_policies.csv; 17 (N, D) cells,
N 0.21-1.07B non-embedding, D 4-100B, M 19-466):
  - Step Law's own rule (in-sample; point and the four corners of the 95% box);
  - random configurations (best of k = 1, 4, 16 per cell): the 'D-biased' inefficiency (falls with D);
  - the other published rules (fixed, Porian base and N-rule, DeepSeek C-rule, Bjorck (N, D)-rule);
  - a non-constant version from the exponential stochastic frontier (scale of inefficiency with elasticities
    gamma_D = -0.260, gamma_N = -0.063 around Step Law's design centre; m8 memo 1.5(iv));
  - the breakdown value: the constant iota_d (iota_n = 0) that would remove all of the convexity.
Also reported: D6's path-exponent bias a_obs - a = -(iota_n - iota_d) E / ((alpha+beta) R*(C)) on Farseer's
Chinchilla-form fit, against the path shift computed pointwise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import rb1common as cm

rc = cm.rc
C_PATH = np.array([2e18, 5e18, 1e19, 2e19, 5e19, 1e20, 2e20, 5e20, 1e21])
NEFF_MIN = 15
STEPLAW_CENTER = dict(lnN=np.log(4.0e8), lnD=np.log(2.0e10))   # geometric centre of the Step Law grid (non-emb. N)
SFA = dict(gamma_N=-0.063, gamma_D=-0.260)                      # exponential SFA (m8 memo 1.5(iv))


def _farseer():
    import farseer as fz
    fa = fz.load()
    cv = pd.read_csv(cm.RA1_FAR_CV)
    b = cv[cv.conv == "ne"].sort_values("loocv_rmse").iloc[0]
    bw = (float(b.hx), float(b.hz))
    return fz, fa, bw


def surface(fz, fa, bw):
    x = np.log(fa.N.values.astype(float))
    z = np.log(fa.D.values.astype(float))
    f = np.log(fa.L.values.astype(float))
    grid = fz.build_grid(fa)
    pts = list(zip(np.log(grid["N"].values), np.log(grid["D"].values)))
    loc = fz.local_many(x, z, f, pts, *bw)
    grid = grid.assign(fx=loc[:, 1], fz=loc[:, 2], neff=loc[:, 6], lnL=loc[:, 0])
    return x, z, f, grid


def lnw_corr(fx, fz_, iota_n, iota_d):
    en, ed = -fx + iota_n, -fz_ + iota_d
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((en > 0) & (ed > 0), np.log(en / ed), np.nan)


def path_corr(fz, x, z, f, bw, iota_fun, hull):
    """ln N*_true(C) on each isocost: root of the corrected local ln w. iota_fun(lnN, lnD) -> (iota_n, iota_d)."""
    out = []
    for Cc in C_PATH:
        cp = np.log(Cc / 6)
        lo, hi = hull(cp)
        if lo is None:
            out.append(np.nan)
            continue

        def g(xx):
            c, _ = fz.local_at(x, z, f, xx, cp - xx, *bw)
            i_n, i_d = iota_fun(xx, cp - xx)
            v = lnw_corr(c[1], c[2], i_n, i_d)
            return float(v)

        try:
            glo, ghi = g(lo), g(hi)
            if not (np.isfinite(glo) and np.isfinite(ghi)) or np.sign(glo) == np.sign(ghi):
                out.append(np.nan)
                continue
            out.append(brentq(g, lo, hi, xtol=1e-7))
        except ValueError:
            out.append(np.nan)
    return np.array(out)


def linearity(grid, lnw, lnNstar):
    """b1, b2 of ln w = b1 u + b2 u^2 on grid points with n_eff >= 15 inside the path's compute range (ra1's rule)."""
    pc = np.log(C_PATH)
    lMs = pc - np.log(6) - 2 * lnNstar          # ln M* = ln D* - ln N* = (ln(C/6) - ln N*) - ln N*
    g = np.isfinite(lMs)
    cg = np.log(6 * grid["N"].values * grid["D"].values)
    ok = (grid["neff"].values >= NEFF_MIN) & np.isfinite(lnw) & (cg >= pc[g].min()) & (cg <= pc[g].max())
    lM = np.interp(cg[ok], pc[g], lMs[g])
    u = np.log(grid["M"].values[ok]) - lM
    y = lnw[ok]
    X = np.column_stack([u, u * u])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(b[0]), float(b[1]), int(ok.sum()), float(u.max())


def run(log=cm.log):
    fz, fa, bw = _farseer()
    x, z, f, grid = surface(fz, fa, bw)
    xs_sizes = np.log(np.sort(fa.N.unique()))
    dmin = [fa.D[fa.N == v].min() for v in np.sort(fa.N.unique())]
    dmax = [fa.D[fa.N == v].max() for v in np.sort(fa.N.unique())]
    hull = fz.make_hull(x, z, xs_sizes, dmin, dmax)
    fx, fzz = grid["fx"].values, grid["fz"].values
    lnw_obs = lnw_corr(fx, fzz, 0.0, 0.0)
    ln_obs = path_corr(fz, x, z, f, bw, lambda a, b: (0.0, 0.0), hull)
    b1o, b2o, n_o, umax = linearity(grid, lnw_obs, ln_obs)
    # check against ra1 (b2 = 0.0204 non-embedding; ra1 interpolated M*_local on its own C levels)
    ra1 = pd.read_csv(cm.RA1_FAR_LIN)
    b2_ra1 = float(ra1[(ra1.conv == "ne") & (ra1.param == "b2")].est.iloc[0])
    log(f"  Farseer convexity (observed): b1 = {b1o:.4f}, b2 = {b2o:.4f} (ra1: {b2_ra1:.4f}); n = {n_o}, u_max = {umax:.2f}")
    P = pd.read_csv(cm.M8_POLICIES).set_index("rule")

    def const(i_n, i_d):
        return lambda a, b: (i_n, i_d)

    scen = []
    sl = P.loc["steplaw"]
    scen.append(("Step Law rule (in-sample point estimate)", "Step Law rule", const(sl.u_n, sl.u_d), sl.u_n, sl.u_d))
    for sn in (-1, 1):
        for sd in (-1, 1):
            i_n, i_d = sl.u_n + sn * 1.96 * sl.u_n_se, sl.u_d + sd * 1.96 * sl.u_d_se
            scen.append((f"Step Law rule, 95% corner (iota_n {'+' if sn > 0 else '-'}, iota_d {'+' if sd > 0 else '-'})",
                         "Step Law rule", const(i_n, i_d), i_n, i_d))
    for key, lab in (("random_k1", "Random configuration, best of 1 (D-biased)"), ("random_k4", "Random configurations, best of 4"),
                     ("random_k16", "Random configurations, best of 16"), ("best_fixed", "One fixed (LR, batch)"),
                     ("porian_base", "Porian base (fixed LR 3e-3, 0.5M batch)"), ("porian_rule", "Porian N-rule"),
                     ("deepseek", "DeepSeek C-rule"), ("bjorck", "Bjorck (N, D)-rule")):
        r = P.loc[key]
        scen.append((lab, "other policy", const(r.u_n, r.u_d), r.u_n, r.u_d))
    # SFA-shaped (non-constant) gradients: iota = ibar exp(gN (n - n0) + gD (d - d0)); iota_n = gN iota, iota_d = gD iota
    for ibar, lab in ((float(sl.u_mean), "Step Law rule level"), (float(P.loc["random_k1"].u_mean), "random-configuration level")):
        def fun(a, b, ibar=ibar):
            io = ibar * np.exp(SFA["gamma_N"] * (a - STEPLAW_CENTER["lnN"]) + SFA["gamma_D"] * (b - STEPLAW_CENTER["lnD"]))
            return SFA["gamma_N"] * io, SFA["gamma_D"] * io
        scen.append((f"SFA-shaped gradients (exponential frontier), {lab} (mean iota {ibar:.4f})", "SFA", fun, np.nan, np.nan))
    rows = []
    for lab, grp, fun, i_n, i_d in scen:
        io = np.array([fun(a, b) for a, b in zip(np.log(grid["N"].values), np.log(grid["D"].values))])
        lnw_t = lnw_corr(fx, fzz, io[:, 0], io[:, 1])
        lnNs = path_corr(fz, x, z, f, bw, fun, hull)
        b1t, b2t, n_t, _ = linearity(grid, lnw_t, lnNs)
        ok = np.isfinite(lnNs) & np.isfinite(ln_obs)
        a_shift = float(np.polyfit(np.log(C_PATH[ok]), lnNs[ok] - ln_obs[ok], 1)[0]) if ok.sum() >= 3 else np.nan
        hi = (grid["M"].values >= 1024) & (grid["neff"].values >= NEFF_MIN)
        rows.append(dict(scenario=lab, group=grp, iota_n=i_n, iota_d=i_d, b1_true=b1t, b2_true=b2t, b2_obs=b2o,
                         tuning_share_of_b2=(b2o - b2t) / b2o, n=n_t, n_invalid=int(np.sum(~np.isfinite(lnw_t) & np.isfinite(lnw_obs))),
                         mean_lnw_shift_Mge1024=float(np.nanmean(lnw_t[hi] - lnw_obs[hi])),
                         path_exponent_shift_pointwise=-a_shift))
    T = pd.DataFrame(rows)
    # breakdown: constant iota_d > 0 (iota_n = 0) removing all convexity
    def b2_of(i_d):
        lnw_t = lnw_corr(fx, fzz, 0.0, i_d)
        lnNs = path_corr(fz, x, z, f, bw, const(0.0, i_d), hull)
        return linearity(grid, lnw_t, lnNs)[1]
    grid_id = np.array([0.0, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.012, 0.016, 0.024, 0.032, 0.048])
    b2s = np.array([b2_of(v) for v in grid_id])
    brk = np.nan
    sgn = np.sign(b2s)
    for i in range(len(grid_id) - 1):
        if sgn[i] > 0 and sgn[i + 1] <= 0:
            brk = brentq(b2_of, grid_id[i], grid_id[i + 1], xtol=1e-6)
            break
    BR = pd.DataFrame(dict(iota_d=grid_id, b2_true=b2s))
    log(f"  tuning scenarios: " + "; ".join(f"{r.scenario[:28]}: share {r.tuning_share_of_b2:+.2f}" for r in T.itertuples()))
    log(f"  breakdown iota_d (iota_n = 0) removing all convexity: {brk:.5f}")
    # D6 formula on the path exponent (Chinchilla form fitted to Farseer, non-embedding): a_obs - a =
    # -(iota_n - iota_d) E / ((alpha+beta) R*(C)), averaged over the path's compute range
    import pickle, os
    est = pickle.load(open(os.path.join(rc.PROC, "stage_farseer.pkl"), "rb"))["est"]["ne"]["chin_full"]
    lnA, lnB, lnE, al, be = est
    E, A, Bc = np.exp(lnE), np.exp(lnA), np.exp(lnB)
    S = al + be
    G = (al * A / (be * Bc)) ** (1 / S)
    a = be / S
    Ns = G * (C_PATH / 6) ** a
    Ds = (C_PATH / 6) / Ns
    Rs = A * Ns ** -al + Bc * Ds ** -be
    d6 = []
    for r in T.itertuples():
        if np.isfinite(r.iota_n):
            val = -(r.iota_n - r.iota_d) * E / (S * Rs)
            d6.append(dict(scenario=r.scenario, d6_a_bias_mean=float(np.mean(val)), d6_a_bias_min=float(val.min()),
                           d6_a_bias_max=float(val.max()), pointwise_path_exponent_shift=r.path_exponent_shift_pointwise))
    D6 = pd.DataFrame(d6)
    T = T.merge(D6[["scenario", "d6_a_bias_mean"]], on="scenario", how="left")
    return dict(table=T, breakdown=BR, breakdown_iota_d=float(brk), b2_obs=b2o, b1_obs=b1o, b2_ra1=b2_ra1, bw=bw,
                d6=D6, E=E, S=S, Rstar=Rs, umax=umax,
                steplaw_design=dict(N=(2.15e8, 1.07e9), D=(4e9, 1e11), M=(18.6, 466)),
                farseer_M_max=float(grid["M"].max()))
