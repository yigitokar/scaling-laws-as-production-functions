"""Design A -- experimental designs at a common total compute budget.

Truth: Besiroglu et al. (2024) Chinchilla technology.  Every design has 90 runs and total training
compute sum(6ND) = 5.106e22 FLOP (= 10 runs at each of Chinchilla's 9 IsoFLOP budgets).

Designs (cells):
  (i)   on-path: every run at the compute-optimal allocation, tiny allocation noise s=0.02 in ln(D/N)
  (ii)  IsoFLOP: 9 budgets x 10 sizes log-spaced over [N*/16, 16N*] (and +-4x as robustness)
  (iii) factorial: 9 N x 10 D near-orthogonal grid centred on the expansion path (Farseer-like)
  (iv)  optimizing labs: same budgets as (i)/(ii), allocation error ln(D/N) = optimum + s z, s grid
  (v)   Kaplan-belief labs: labs follow N ~ C^0.73 (Kaplan et al. 2020) + s=0.1 noise
  + noise-level and digitization robustness cells.

Estimators:
  primal  = Hoffmann/Besiroglu Huber(1e-3)-LSE on log loss (Approach 3; kappa = 1 imposed)
  dual    = Approach-1/2 style: allocation slope a from the (noisy) optima, frontier L*(C) fit for
            gamma, then alpha = gamma/a, beta = gamma/(1-a) (identification by kappa=1 + optimality).
            'path' for on-path/optimizing designs; 'iso' (IsoFLOP parabola minima) for IsoFLOP designs.
  profile = Gaussian profile likelihood over sigma* in the kappa-generalized model
            L = E + (A N^-a1 + B D^-b1)^kappa (kappa free): LR(sigma*) = n ln(SSR(sigma*)/SSR_min).
Inference for the primal estimator: asymptotic (LAD-sandwich + delta method) Wald CIs in all
replications; pairs-bootstrap percentile CIs (B_BOOT=149 draws, warm start) in the first R_BOOT=150
replications of five cells; kappa-free profile likelihood in the first R_PROF=150 replications.
Review additions (2026-09-23): LR measured against the unrestricted kappa-free minimum (mc_lib.profile_sigma),
a finer sigma* grid next to the truth, and a warm-start vs multi-start bootstrap check on fresh replications
(run_bootcheck_chunk; R_BC=60 replications x B_BC=49 draws for the on-path and s=0.3 cells).

v2 (2026-09-24, after referee round 1; R3 M12, R1 minor 2, R2 minors 7-8, R1 comment 9c, R4 M4e/M5b):
  * the truth is REMOVED from every start set: the primal estimator uses ml.STARTS (9 random starts from Hoffmann's
    grid region, the first 8 identical to v1's non-truth starts); the dual frontier fit uses a 2x2x2 grid of starts
    instead of (truth + 3 fixed starts); the multi-start bootstrap check uses (warm start + ml.STARTS);
  * the profile grid is extended to (0.05, 0.99) and the unrestricted kappa-free optimum is searched over
    sigma* in [0.024, 0.995]; LR is measured relative to that optimum;
  * the objective value at the true parameters is stored (obj_truth) to separate non-identification (estimate's
    objective <= truth's) from optimisation failure (estimate's objective > truth's);
  * new cells with heteroskedastic, within-cluster correlated noise (ml.het_cluster_noise), appended at the end so
    that every v1 cell keeps its seed and hence its simulated data.
Everything else (designs, noise, replication counts, seeds) is unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import mc_lib as ml

import os

QUICK = os.environ.get("M6_QUICK") == "1"
R_MAIN = 20 if QUICK else 500          # replications per cell
R_BOOT = 10 if QUICK else 150          # replications with bootstrap CIs (subset of cells)
B_BOOT = 49 if QUICK else 149          # bootstrap draws per replication
R_PROF = 10 if QUICK else 150          # replications with the kappa-free profile likelihood
CHUNK = 10 if QUICK else 50
BASE_SEED = 20260923
SD_BASE = 0.0075       # log-loss noise sd = residual sd of the Besiroglu fit on the Chinchilla extraction
# Profile grid for sigma*.  Review fix (2026-09-23): four points added next to the truth (0.728, 0.732, 0.742,
# 0.746).  With the original spacing (0.013-0.017 around the truth) the IsoFLOP/factorial CIs (half-width ~0.007)
# were narrower than one grid step, so their widths were artefacts of linear interpolation of a convex LR.
# v2: grid extended to (0.05, 0.99) (R1 comment 9c; R4 M5b) -- the v1 grid [0.50, 0.95] bounded the confidence sets.
SIG_GRID = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.68, 0.70, 0.72,
                     0.728, 0.732, ml.TRUE["sigma_star"], 0.742, 0.746, 0.75, 0.77, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99])
SIG_GRID_V1 = np.array([0.50, 0.55, 0.60, 0.65, 0.68, 0.70, 0.72, 0.728, 0.732, ml.TRUE["sigma_star"], 0.742, 0.746,
                        0.75, 0.77, 0.80, 0.85, 0.90, 0.95])
# heteroskedastic / clustered noise variant (v2, R2 minor 8): sd_i ~ (N_i D_i)^-0.25 (larger for small models and short
# runs; design-average variance = SD_BASE^2), within-cluster correlation 0.5; clusters = compute budget (path and
# IsoFLOP designs: shared data order within a budget) or model size (factorial grid: one training trunk per N).
HET, RHO = 0.25, 0.5
CHI2_95 = 3.841458820694124


def kaplan_design(rng, s=0.1):
    """Labs that optimize against Kaplan's belief N ~ C^0.73 (pivoting at the geometric-mean budget)."""
    C = np.repeat(ml.ISO_BUDGETS, ml.RUNS_PER_BUDGET)
    Cm = np.exp(np.mean(np.log(ml.ISO_BUDGETS)))
    lnN = np.log(ml.TRUTH.N_opt(Cm)) + 0.73 * (np.log(C) - np.log(Cm))
    u = s * rng.standard_normal(C.size)
    N = np.exp(lnN - u / 2)
    D = C / (6 * N)
    return N, D


def cells():
    c = []
    add = lambda **k: c.append(k)
    add(name="onpath", label="(i) On-path (s=0.02)", gen=lambda r: ml.design_path(0.02, r), sd=SD_BASE,
        dual="path", boot=True, s=0.02)
    add(name="iso16", label="(ii) IsoFLOP, +-16x", gen=lambda r: ml.design_isoflop(16), sd=SD_BASE, dual="iso", boot=True)
    add(name="iso4", label="(ii') IsoFLOP, +-4x", gen=lambda r: ml.design_isoflop(4), sd=SD_BASE, dual="iso", boot=False)
    add(name="fact", label="(iii) Factorial grid", gen=lambda r: ml.design_factorial(), sd=SD_BASE, dual=None, boot=True)
    for s in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]:
        add(name=f"opt_s{s:g}", label=f"(iv) Optimizing labs, s={s:g}", gen=(lambda s_: lambda r: ml.design_path(s_, r))(s),
            sd=SD_BASE, dual="path", boot=s in (0.3, 1.0), s=s)
    add(name="kaplan", label="(v) Kaplan-belief labs (N~C^0.73), s=0.1", gen=kaplan_design, sd=SD_BASE, dual="path", boot=False)
    add(name="iso16_digit", label="(ii) IsoFLOP +-16x, digitized", gen=lambda r: ml.design_isoflop(16), sd=SD_BASE,
        dual="iso", boot=False, digitize=True)
    for sd in (0.005, 0.015):
        add(name=f"iso16_sd{sd:g}", label=f"(ii) IsoFLOP +-16x, noise sd={sd:g}", gen=lambda r: ml.design_isoflop(16), sd=sd,
            dual="iso", boot=False)
        add(name=f"opt_s0.3_sd{sd:g}", label=f"(iv) Optimizing labs s=0.3, noise sd={sd:g}",
            gen=lambda r: ml.design_path(0.3, r), sd=sd, dual="path", boot=False, s=0.3)
    # v2 cells (appended: v1 cells keep their indices, seeds and data)
    by_budget = lambda N, D, sd, rng: ml.het_cluster_noise(N, D, sd, rng, np.round(np.log(6 * N * D), 6), HET, RHO)
    by_size = lambda N, D, sd, rng: ml.het_cluster_noise(N, D, sd, rng, np.round(np.log(N), 6), HET, RHO)
    add(name="onpath_hc", label="(i) On-path (s=0.02), het./clustered noise", gen=lambda r: ml.design_path(0.02, r),
        sd=SD_BASE, dual="path", boot=False, s=0.02, noise=by_budget)
    add(name="iso16_hc", label="(ii) IsoFLOP +-16x, het./clustered noise", gen=lambda r: ml.design_isoflop(16), sd=SD_BASE,
        dual="iso", boot=False, noise=by_budget)
    add(name="fact_hc", label="(iii) Factorial grid, het./clustered noise", gen=lambda r: ml.design_factorial(), sd=SD_BASE,
        dual=None, boot=False, noise=by_size)
    for s in (0.1, 0.3, 1.0):
        add(name=f"opt_s{s:g}_hc", label=f"(iv) Optimizing labs s={s:g}, het./clustered noise",
            gen=(lambda s_: lambda r: ml.design_path(s_, r))(s), sd=SD_BASE, dual="path", boot=False, s=s, noise=by_budget)
    return c


# ----------------------------------------------------------------------------- dual (Approach 1/2) estimator

def _fit_frontier(lC, lL, starts=None):
    """Huber(1e-3) fit of ln L = logaddexp(e, k - gamma*(lnC - ln6)) (inverse cost function)."""
    x = lC - np.log(6.0)

    def obj(p):
        e, k, g = p
        t = k - g * x
        pred = np.logaddexp(e, t)
        r = pred - lL
        dl = np.clip(r, -1e-3, 1e-3)
        pe = np.exp(e - pred)
        pt = 1 - pe
        f = np.sum(np.where(np.abs(r) <= 1e-3, 0.5 * r ** 2, 1e-3 * (np.abs(r) - 0.5e-3)))
        return f, np.array([np.sum(dl * pe), np.sum(dl * pt), np.sum(dl * pt * -x)])

    if starts is None:
        # v2: a 2x2x2 grid of starts (ln E in {-0.5, 0.8}, ln K in {3, 10}, gamma in {0.05, 0.3}); v1 used the truth
        # plus three fixed starts.
        starts = [np.array([e, k, g]) for e in (-0.5, 0.8) for k in (3.0, 10.0) for g in (0.05, 0.3)]
    best = (None, np.inf)
    for st in starts:
        r = minimize(obj, st, jac=True, method="L-BFGS-B", bounds=[(-3, 2.5), (-20, 60), (1e-3, 3.0)],
                     options=dict(maxiter=5000, ftol=1e-15, gtol=1e-12))
        if r.fun < best[1]:
            best = (r.x, r.fun)
    return best[0]


def dual_estimate(N, D, L, kind):
    lN, lD, lL = np.log(N), np.log(D), np.log(L)
    lC = np.log(6.0) + lN + lD
    if kind == "path":
        X = np.column_stack([np.ones_like(lC), lC])
        bN = np.linalg.lstsq(X, lN, rcond=None)[0]
        a = bN[1]
        lnNstar = bN[0] + bN[1] * np.log(ml.C_REF)
        e, k, g = _fit_frontier(lC, lL)
    elif kind == "iso":
        # Approach 2: parabola in ln N at each IsoFLOP budget -> argmin and minimum loss
        # runs are grouped by their nominal budget (nearest Chinchilla budget; robust to digitization error in C)
        lB = np.log(ml.ISO_BUDGETS)
        Cb = np.argmin(np.abs(lC[:, None] - lB[None, :]), axis=1)
        lNs, lCs, lLs = [], [], []
        for cb in np.unique(Cb):
            m = Cb == cb
            xc = lN[m].mean()                      # centre before the quadratic fit (conditioning)
            p = np.polyfit(lN[m] - xc, lL[m], 2)
            if p[0] <= 0:
                continue
            x0 = -p[1] / (2 * p[0])
            lNs.append(x0 + xc)
            lCs.append(lB[cb])
            lLs.append(np.polyval(p, x0))
        lNs, lCs, lLs = map(np.array, (lNs, lCs, lLs))
        X = np.column_stack([np.ones_like(lCs), lCs])
        bN = np.linalg.lstsq(X, lNs, rcond=None)[0]
        a = bN[1]
        lnNstar = bN[0] + bN[1] * np.log(ml.C_REF)
        e, k, g = _fit_frontier(lCs, lLs)
    else:
        return np.full(len(ml.PARAMS), np.nan)
    al, be = g / a, g / (1 - a)
    lnMstar = np.log(ml.C_REF / 6.0) - 2 * lnNstar
    return np.array([al, be, a, g, 2 / (2 + al + be), lnMstar])


# ----------------------------------------------------------------------------- one replication

def is_corner(th, lN, lD):
    a, b, e, al, be = th
    u = a - al * lN
    v = b - be * lD
    su = 1 / (1 + np.exp(v - u))
    at_bound = (al >= ml.BOUNDS[3][1] - 1e-3) or (be >= ml.BOUNDS[4][1] - 1e-3) or al <= 2e-3 or be <= 2e-3
    return bool(at_bound or su.max() < 0.02 or su.min() > 0.98)


def one_rep(cell, rng, do_boot, do_prof=True):
    N, D = cell["gen"](rng)
    Nobs, Dobs, L = ml.simulate_loss(N, D, cell["sd"], rng, digitize=cell.get("digitize", False), noise=cell.get("noise"))
    lN, lD, lL = np.log(Nobs), np.log(Dobs), np.log(L)
    th, obj = ml.fit_huber(lN, lD, lL)
    est = ml.derived_vec(th)
    # v2 diagnostic: objective at the true parameters (never used as a start)
    obj_truth = float(ml.sl._obj(ml.TH0, lN, lD, lL, 1e-3, np.ones_like(lL)))
    V, cond = ml.asym_cov(th, lN, lD, lL)
    se = ml.delta_se(th, V)
    corner = is_corner(th, lN, lD)
    singular = (cond > 1e12) or corner
    out = dict(obj=obj, obj_truth=obj_truth, cond=cond, corner=corner, singular=singular)
    for k, x, s in zip(ml.PARAMS, est, se):
        out[f"p_{k}"] = x
        out[f"se_{k}"] = s
    dual = dual_estimate(Nobs, Dobs, L, cell["dual"])
    for k, x in zip(ml.PARAMS, dual):
        out[f"d_{k}"] = x
    if do_prof:
        extra = []
        try:           # v2: equivalent start from the dual/path estimate (all runs; a from ln N on ln C, frontier fit)
            lC = np.log(6.0) + lN + lD
            X = np.column_stack([np.ones_like(lC), lC - np.log(6.0)])
            c0, a_ = np.linalg.lstsq(X, lN, rcond=None)[0]
            e_, k_, g_ = _fit_frontier(lC, lL)
            tp = ml.th_from_path_dual(lN, lD, a_, c0, e_, k_, g_)
            if np.all(np.isfinite(tp)):
                extra.append(tp)
        except Exception:
            pass
        lr, info = ml.profile_sigma(lN, lD, lL, SIG_GRID, th, return_info=True, extra_th=extra)
        for j, x in enumerate(lr):
            out[f"lr_{j}"] = x
        out.update(info)          # prof_shift: LR understatement of the original denominator; prof_sig_hat
    if do_boot:
        n = len(lL)
        bs = np.empty((B_BOOT, len(ml.PARAMS)))
        for b in range(B_BOOT):
            idx = rng.integers(0, n, n)
            thb, _ = ml.fit_huber(lN[idx], lD[idx], lL[idx], starts=[th])
            bs[b] = ml.derived_vec(thb)
        lo, hi = np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)
        for k, l_, h_, sdv in zip(ml.PARAMS, lo, hi, bs.std(axis=0, ddof=1)):
            out[f"blo_{k}"], out[f"bhi_{k}"], out[f"bse_{k}"] = l_, h_, sdv
    return out


def run_chunk(args):
    ci, chunk = args
    cell = cells()[ci]
    ss = np.random.SeedSequence([BASE_SEED, ci, chunk])
    rng = np.random.default_rng(ss)
    rows = []
    for j in range(CHUNK):
        rep = chunk * CHUNK + j
        do_boot = cell["boot"] and rep < R_BOOT
        r = one_rep(cell, rng, do_boot, do_prof=rep < R_PROF)
        r.update(cell=cell["name"], rep=rep)
        rows.append(r)
    return rows


def tasks():
    return [(ci, ch) for ci in range(len(cells())) for ch in range(R_MAIN // CHUNK)]


# ----------------------------------------------------------------------------- bootstrap robustness check
# Review addition (2026-09-23).  The main bootstrap warm-starts every draw at the replication's own estimate
# (one start).  On the flat on-path ridge that understates the bootstrap spread.  This check draws fresh
# replications and computes, on the same data and the same resamples, (i) the warm-start percentile CI and
# (ii) a multi-start CI that uses the estimator's full start set (warm start + the 9 fixed starts).

R_BC = 10 if QUICK else 60             # replications per cell
B_BC = 19 if QUICK else 49             # bootstrap draws per replication
CHUNK_BC = 5 if QUICK else 10
BC_CELLS = ["onpath", "opt_s0.3"]


def bootcheck_tasks():
    names = [c["name"] for c in cells()]
    return [(names.index(nm), ch) for nm in BC_CELLS for ch in range(R_BC // CHUNK_BC)]


def run_bootcheck_chunk(args):
    ci, chunk = args
    cell = cells()[ci]
    rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, 999, ci, chunk]))
    rows = []
    for j in range(CHUNK_BC):
        N, D = cell["gen"](rng)
        Nobs, Dobs, L = ml.simulate_loss(N, D, cell["sd"], rng, digitize=cell.get("digitize", False), noise=cell.get("noise"))
        lN, lD, lL = np.log(Nobs), np.log(Dobs), np.log(L)
        th, _ = ml.fit_huber(lN, lD, lL)
        n = len(lL)
        bw = np.empty((B_BC, len(ml.PARAMS)))
        bm = np.empty((B_BC, len(ml.PARAMS)))
        for b in range(B_BC):
            idx = rng.integers(0, n, n)
            thw, fw = ml.fit_huber(lN[idx], lD[idx], lL[idx], starts=[th])
            thm, fm = ml.fit_huber(lN[idx], lD[idx], lL[idx], starts=[th] + ml.STARTS)   # v2: no truth start
            bw[b], bm[b] = ml.derived_vec(thw), ml.derived_vec(thm)
        r = dict(cell=cell["name"], rep=chunk * CHUNK_BC + j)
        for tag, arr in (("warm", bw), ("multi", bm)):
            lo, hi = np.percentile(arr, 2.5, axis=0), np.percentile(arr, 97.5, axis=0)
            for k, l_, h_ in zip(ml.PARAMS, lo, hi):
                r[f"{tag}_lo_{k}"], r[f"{tag}_hi_{k}"] = l_, h_
        rows.append(r)
    return rows


def bootcheck_summary(df):
    rows = []
    for nm in BC_CELLS:
        g = df[df.cell == nm]
        for k in ml.PARAMS:
            t = ml.TRUE[k]
            row = dict(cell=nm, param=k, R=len(g), B=B_BC)
            for tag in ("warm", "multi"):
                lo, hi = g[f"{tag}_lo_{k}"].to_numpy(float), g[f"{tag}_hi_{k}"].to_numpy(float)
                row[f"{tag}_cover"] = np.mean((lo <= t) & (t <= hi))
                row[f"{tag}_median_width"] = np.median(hi - lo)
            rows.append(row)
    return pd.DataFrame(rows)


def design_diagnostics():
    rows = []
    rng = np.random.default_rng(BASE_SEED)
    for c in cells():
        if c.get("digitize") or "_sd" in c["name"]:
            continue
        N, D = c["gen"](rng)
        raw, nrm, svr = ml.design_conditioning(N, D)
        tsd, msd, rms_dev = ml.transverse_sd(N, D)
        rows.append(dict(cell=c["name"], label=c["label"], n=len(N), total_compute=float((6 * N * D).sum()),
                         M_min=float((D / N).min()), M_max=float((D / N).max()),
                         transverse_sd_t=tsd, sd_lnM_given_c=msd, rms_dev_from_true_path=rms_dev, cond_raw=raw, cond_normalized=nrm, sv_ratio_normalized=svr,
                         corr_lnN_lnD=float(np.corrcoef(np.log(N), np.log(D))[0, 1])))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- summaries

def summarize(df):
    true = np.array([ml.TRUE[k] for k in ml.PARAMS])
    rows = []
    for c in cells():
        g = df[df.cell == c["name"]]
        if g.empty:
            continue
        for est in ("p", "d"):
            for k, t in zip(ml.PARAMS, true):
                x = g[f"{est}_{k}"].to_numpy(float)
                if np.all(np.isnan(x)):
                    continue
                err = x - t
                row = dict(cell=c["name"], label=c["label"], estimator="primal" if est == "p" else "dual",
                           param=k, truth=t, R=int(np.isfinite(x).sum()), mean=np.nanmean(x), median=np.nanmedian(x),
                           bias=np.nanmean(err), mc_se_bias=np.nanstd(x, ddof=1) / np.sqrt(np.isfinite(x).sum()),
                           median_bias=np.nanmedian(err), sd=np.nanstd(x, ddof=1),
                           rmse=np.sqrt(np.nanmean(err ** 2)), mae=np.nanmedian(np.abs(err)),
                           q05=np.nanpercentile(x, 5), q95=np.nanpercentile(x, 95))
                if est == "p":
                    se = g[f"se_{k}"].to_numpy(float)
                    ok = ~g["singular"].to_numpy(bool) & np.isfinite(se) & (se > 0)
                    row["wald_computable"] = ok.mean()
                    row["wald_cover"] = np.mean(np.abs(err[ok]) <= 1.96 * se[ok]) if ok.any() else np.nan
                    row["wald_median_width"] = np.median(2 * 1.96 * se[ok]) if ok.any() else np.nan
                    if f"blo_{k}" in g:
                        b = g.dropna(subset=[f"blo_{k}"])
                        if len(b):
                            row["boot_R"] = len(b)
                            row["boot_cover"] = np.mean((b[f"blo_{k}"] <= t) & (t <= b[f"bhi_{k}"]))
                            row["boot_median_width"] = np.median(b[f"bhi_{k}"] - b[f"blo_{k}"])
                    row["corner_share"] = g["corner"].mean()
                    # v2 additions: Monte Carlo standard errors of coverage rates (R3 minor 13) and the share of
                    # replications whose estimate has a HIGHER objective than the (never-used-as-start) truth, i.e.
                    # optimisation failures as opposed to genuine flat-ridge/non-identification outcomes.
                    row["wald_cover_mcse"] = (np.sqrt(row["wald_cover"] * (1 - row["wald_cover"]) / max(ok.sum(), 1))
                                              if ok.any() else np.nan)
                    if "boot_cover" in row:
                        row["boot_cover_mcse"] = np.sqrt(row["boot_cover"] * (1 - row["boot_cover"]) / row["boot_R"])
                    if "obj_truth" in g:
                        d_obj = g["obj"].to_numpy(float) - g["obj_truth"].to_numpy(float)
                        row["share_obj_above_truth"] = np.mean(d_obj > 1e-9)
                        row["median_obj_minus_truth"] = np.median(d_obj)
                rows.append(row)
    return pd.DataFrame(rows)


def _interp_ci(lr, grid=SIG_GRID, crit=CHI2_95):
    """95% profile CI by linear interpolation of LR(sigma*) between grid points, expanding from the grid
    minimum; returns (lo, hi, hits_edge).  If LR > crit at every grid point (possible now that LR is measured
    against the unrestricted minimum, which can lie off the grid) the CI on the grid is empty: (nan, nan, False)."""
    j = int(np.nanargmin(lr))
    if lr[j] > crit:
        return np.nan, np.nan, False
    lo = hi = grid[j]
    edge = False
    k = j
    while k > 0 and lr[k - 1] <= crit:
        k -= 1
    if k == 0:
        lo, edge = grid[0], True
    else:
        lo = grid[k - 1] + (crit - lr[k - 1]) * (grid[k] - grid[k - 1]) / (lr[k] - lr[k - 1])
    k = j
    while k < len(grid) - 1 and lr[k + 1] <= crit:
        k += 1
    if k == len(grid) - 1:
        hi, edge = grid[-1], True
    else:
        hi = grid[k] + (crit - lr[k]) * (grid[k + 1] - grid[k]) / (lr[k + 1] - lr[k])
    return lo, hi, edge


def profile_summary(df):
    rows = []
    lrcols = [f"lr_{j}" for j in range(len(SIG_GRID))]
    jt = int(np.argmin(np.abs(SIG_GRID - ml.TRUE["sigma_star"])))
    for c in cells():
        g = df[df.cell == c["name"]].dropna(subset=lrcols)
        if g.empty:
            continue
        LR = g[lrcols].to_numpy(float)
        acc = LR <= CHI2_95
        ci = np.array([_interp_ci(r) for r in LR])
        empty = ~np.isfinite(ci[:, 0])
        width = np.where(empty, 0.0, ci[:, 1] - ci[:, 0])
        extra = {}
        if "prof_shift" in g:
            # how much the original denominator (min of grid and kappa=1 SSR) understated LR; and where the
            # unrestricted kappa-free optimum lies
            sh = g["prof_shift"].to_numpy(float)
            extra = dict(mean_LR_understatement_old=np.mean(sh), p90_LR_understatement_old=np.percentile(sh, 90),
                         share_flat_old_denominator=np.mean(((LR - sh[:, None]) <= CHI2_95).all(axis=1)),
                         share_sig_hat_off_grid=np.mean((g["prof_sig_hat"] < SIG_GRID[0]) | (g["prof_sig_hat"] > SIG_GRID[-1])))
        # v2: flat share on the v1 grid points [0.50, 0.95] (comparability with v1) and MC s.e. of coverage
        jv1 = [int(np.argmin(np.abs(SIG_GRID - x))) for x in SIG_GRID_V1]
        extra["share_flat_on_v1_grid"] = np.mean(acc[:, jv1].all(axis=1))
        extra["cover_truth_mcse"] = np.sqrt(np.mean(acc[:, jt]) * (1 - np.mean(acc[:, jt])) / len(g))
        extra["share_accept_all_below_050"] = np.mean(acc[:, SIG_GRID <= 0.5].all(axis=1))
        extra["share_accept_all_above_090"] = np.mean(acc[:, SIG_GRID >= 0.9].all(axis=1))
        rows.append(dict(cell=c["name"], label=c["label"], R=len(g),
                         cover_truth=np.mean(acc[:, jt]),
                         share_ci_hits_grid_edge=np.mean(ci[:, 2] > 0),
                         share_flat_everywhere=np.mean(acc.all(axis=1)),
                         share_ci_empty_on_grid=np.mean(empty),
                         median_ci_width=np.median(width),
                         median_ci_lo=np.nanmedian(ci[:, 0]) if (~empty).any() else np.nan,
                         median_ci_hi=np.nanmedian(ci[:, 1]) if (~empty).any() else np.nan, **extra,
                         median_LR_at_050=np.median(LR[:, 0]), median_LR_at_095=np.median(LR[:, -1]),
                         **{f"mean_LR_{s:.3f}": LR[:, j].mean() for j, s in enumerate(SIG_GRID)},
                         **{f"median_LR_{s:.3f}": np.median(LR[:, j]) for j, s in enumerate(SIG_GRID)}))
    return pd.DataFrame(rows)
