"""m9_coverage_v2.py -- design-only size, coverage and calibration study, version 2 (round-3 fix list, item X2).

Written 2026-09-25 for Amendment 2 of the pre-analysis plan. Like m9_power.py and m9_coverage.py (which are left as
committed), this stage NEVER reads data/processed/sweep/results.jsonl or any other endpoint loss: the design comes from
code/sweep/run_grid.py and the architecture formulas (m9_power.mc_design), the technology is a public one.

What it adds to version 1 (m9_coverage.py, 40 replications at noise s.d. 0.005):
  * the tilt test of D8 (restricted wild cluster bootstrap on CR2 residuals, 'WCR-CR2') and the plan's interval test,
    with >= 500 replications at chi = 0 and chi = 0.22, on TWO validation sets (noise correlated 0.8), so that the
    size and power of every test and the probabilities of the plan's and of D8's two-set decision rules are measured,
    each with a Monte Carlo standard error;
  * coverage and calibration factors of the model-free sigma* in the FLOP-effective convention N_F = fpt/6
    (C = 6 N_F D exactly; path w = 1), in P (actual-FLOP path, and P6) and in T, for BOTH corpora' designs, under the
    plan's scheme ('wcu': wild cluster bootstrap on raw residuals, fixed bandwidth) and the CR2 scheme of D8
    ('cr2cv': CR2 residuals, bandwidth re-selected in every draw), at several noise levels, within-trunk correlations
    and truths (Chinchilla form; kappa family with kappa = 0.6 and with kappa = 0.36, the dry-run kappa-hat);
  * the same for the Q3 slope (convention P, main grid + high-M runs, Chinchilla form fitted on M <= 100);
  * the Q1 equality test (sigma*_edu - sigma*_web, joint draws with one weight per architecture shared by the corpora):
    Monte Carlo s.d., bootstrap s.e., coverage, size before and after calibration, and the calibration factor.

Estimators are those of run.py (m9_stages / m9_est). One implementation detail differs and is verified to be exact:
the leave-one-out CV of the local-quadratic bandwidth (m9_est.cv_bandwidth) is computed here with batched
pseudo-inverses instead of one lstsq call per point (cv_bandwidth_fast; check_fast_cv() compares the two on random
designs: same selected bandwidth, RMSE equal to rounding).

Usage (CPU only, at most 2 processes by default (M9_V2_PROCS), run with nice -n 10; never imports MLX):
  python m9_coverage_v2.py check                      # fast-CV equivalence check
  python m9_coverage_v2.py tilt R B                   # tilt tests, chi in {0, 0.22}
  python m9_coverage_v2.py cov CELL R B               # one coverage cell (see COV_CELLS)
Replication-level results are appended to output/tables/m9_sweeps_power_v2_reps_*.csv as they finish (a rerun
resumes, skipping finished replications); summaries are built by m9_power_v2.py.

Runs of 25 September 2026 (log: output/tables/m9_sweeps_power_v2_log.txt), all with nice -n 10 and 4 processes:
  cov chin_s005_r5 500 99; tilt 500 99; cov kap36_s005_r5 200 49; cov chin_s002_r5 150 49; cov chin_s010_r5 150 49;
  cov chin_s005_r9 150 49; then m9_power_v2.py design, q3null 150 (rho = 0.5 cells extended to 500), summary.
Result that changed the pre-analysis record (Amendment 2, item 6): D8's restricted test on CR2 residuals has size 0.092
and power 0.65 per validation set; the CR2 interval widened by 1.25 has size 0.118 per set, two-set size 0.054 (D8's
rule: 0.030) and power 1.00.
Review of 2026-09-25 (paper/notes/round3_WP1_review.md): replications 0-3 (tilt, Q3 null) and 0-2 (primary coverage
cell) rerun from this code reproduce the stored rows exactly; summaries regenerate byte-identically.
"""
from __future__ import annotations

import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

import m9_common as mc
import m9_est as est
import m9_power as pw
import m9_stages as st

TABLES = os.path.join(mc.ROOT, "output", "tables")
PREFIX = "m9_sweeps_power_v2"
LOGF = os.path.join(TABLES, PREFIX + "_log.txt")
PROCS = int(os.environ.get("M9_V2_PROCS", 2))   # review 2026-09-25: default 2 (fix list); the v2 runs used 4
BW_MULTS = st.BW_MULTS
C_LEVELS = st.C_LEVELS
RHO_VAL = pw.RHO_VAL
CONV_N = {"F": "N_F", "P": "N_P", "T": "N_T"}
PATHCONV = {"F": "T", "P": "P", "T": "T"}   # the w = 1 machinery ('T') on C = 6 N D is exact for N_F

# coverage cells: name -> (truth kind, kappa, noise s.d., within-trunk share rho, conventions). The primary cell
# carries all four conventions (F, P, P6, T); the sensitivity cells F, P and P6 (T is within 3-7 percent of F here).
COV_CELLS = {
    "chin_s005_r5": ("chin", 1.0, 0.005, 0.5, ("F", "P", "T")),
    "kap36_s005_r5": ("kappa", 0.36, 0.005, 0.5, ("F", "P")),
    "chin_s002_r5": ("chin", 1.0, 0.002, 0.5, ("F", "P")),
    "chin_s010_r5": ("chin", 1.0, 0.010, 0.5, ("F", "P")),
    "chin_s005_r9": ("chin", 1.0, 0.005, 0.9, ("F", "P")),
    "chin_s005_r0": ("chin", 1.0, 0.005, 0.0, ("F", "P")),
    "kap60_s005_r5": ("kappa", 0.60, 0.005, 0.5, ("F", "P")),
}


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + msg
    print(line, flush=True)
    with open(LOGF, "a") as fh:
        fh.write(line + "\n")


# ============================================================================ design and truth (no results)
def design_v2():
    """m9_power.mc_design() plus the FLOP-effective convention N_F = fpt/6 (so that C = fpt D = 6 N_F D)."""
    d = pw.mc_design()
    for df in d.values():
        df["N_F"] = df["fpt"] / 6.0
        df["M_F"] = df.D / df.N_F
    return d


def truth_v2(kind, kappa, design):
    """Chinchilla truth of the power study, or a kappa-family truth with sigma*_kappa = 0.70 and outer exponent
    kappa (0.6 as in the power study's sensitivity cell; 0.36, the FineWeb-Edu dry-run kappa-hat), same anchor."""
    tp = pw.truth_params("chin" if kind == "chin" else "kappa", design)
    if kind == "kappa":
        Cc = tp["Cc"]
        Ns = np.sqrt(Cc / (6 * pw.M_ANCHOR))
        Ds = pw.M_ANCHOR * Ns
        R = pw.L_ANCHOR - pw.E_ANCHOR
        a1 = 0.5 * (2 / 0.70 - 2)
        uv = R ** (1 / kappa) / 2
        tp = dict(kind="kappa", lnA=np.log(uv) + a1 * np.log(Ns), lnB=np.log(uv) + a1 * np.log(Ds),
                  lnE=np.log(pw.E_ANCHOR), a1=a1, b1=a1, k=kappa, Cc=Cc)
    return tp


def chin_start(tp, chi):
    """A Chinchilla-form start (lnA, lnB, lnE, alpha, beta) at the truth (exact under the Chinchilla truth)."""
    return np.array([tp["lnA"], tp["lnB"] - chi, tp["lnE"], tp["a1"], tp["b1"]])


# ============================================================================ fast, exact LOO-CV bandwidth
def cv_bandwidth_fast(x, z, f, mults=BW_MULTS):
    """Same result as m9_est.cv_bandwidth (LOO-CV of the Gaussian-kernel local quadratic of m2_est.local_quad over
    h = (mx sd(x), mz sd(z))), computed with one batched pseudo-inverse per bandwidth pair. The pseudo-inverse uses
    lstsq's default cutoff (eps * max(n, 6) times the largest singular value). Ties keep the first minimum, as there."""
    x, z, f = (np.asarray(v, float) for v in (x, z, f))
    n = len(f)
    sx, sz = np.std(x), np.std(z)
    dx = x[None, :] - x[:, None]     # [i, j] = x_j - x_i
    dz = z[None, :] - z[:, None]
    X = np.stack([np.ones_like(dx), dx, dz, dx ** 2, dz ** 2, dx * dz], axis=-1)
    eye = np.eye(n, dtype=bool)
    tol = np.finfo(float).eps * max(n, 6)
    best = None
    for mx_, mz_ in itertools.product(mults, mults):
        hx, hz = mx_ * sx, mz_ * sz
        w = np.exp(-0.5 * ((dx / hx) ** 2 + (dz / hz) ** 2))
        w[eye] = 0.0
        sw = np.sqrt(w)
        A = X * sw[..., None]
        pinv = np.linalg.pinv(A, rtol=tol)            # (n, 6, n)
        c0 = np.einsum("in,in->i", pinv[:, 0, :], f[None, :] * sw)
        rmse = float(np.sqrt(np.mean(np.square(f - c0))))
        if best is None or rmse < best[0]:
            best = (rmse, hx, hz, mx_, mz_)
    return best


def mf_prepare_fast(N, D, L):
    """m9_stages.mf_prepare with the fast CV (no CV table)."""
    x, z, f = np.log(N), np.log(D), np.log(L)
    rmse, hx, hz, mx_, mz_ = cv_bandwidth_fast(x, z, f)
    sx = np.sort(np.unique(x))
    dmin = np.array([D[x == v].min() for v in sx])
    dmax = np.array([D[x == v].max() for v in sx])
    return dict(x=x, z=z, sx=sx, dmin=dmin, dmax=dmax, hx=hx, hz=hz, mx=mx_, mz=mz_, loocv_rmse=rmse)


def mf_sig(prep, f, pc, levels_ok=None):
    """Model-free sigma* (mean over valid compute levels) on each path of convention pc, exactly as the sig_mf_*
    entries of m9_stages.mf_stats. Returns ({path: sigma}, {path: valid-level mask}, {path: path array})."""
    out, lok, paths = {}, {}, {}
    for p in st.paths_of(pc):
        hull = est.make_hull(p, prep["sx"], prep["dmin"], prep["dmax"])
        pth = est.local_path(prep["x"], prep["z"], f, prep["hx"], prep["hz"], p, C_LEVELS, hull)
        ok = np.isfinite(pth[:, 1]) if levels_ok is None else np.asarray(levels_ok[p], bool)
        vals = np.where(ok, pth[:, 1], np.nan)
        out[p] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
        lok[p] = np.isfinite(pth[:, 1]) if levels_ok is None else levels_ok[p]
        paths[p] = pth
    return out, lok, paths


def check_fast_cv(n_designs=120):
    """Equivalence of cv_bandwidth_fast and m9_est.cv_bandwidth on simulated designs (design only)."""
    design = design_v2()
    rows = []
    for i in range(n_designs):
        rng = np.random.default_rng(mc.seed_of(f"v2check|{i}"))
        kind, kap = (("chin", 1.0), ("kappa", 0.36), ("kappa", 0.6))[i % 3]
        tp = truth_v2(kind, kap, design)
        s = (0.002, 0.005, 0.010, 0.015)[i % 4]
        r = ("edu", "web")[i % 2]
        cv = ("F", "P", "T")[i % 3]
        df = design[r]
        lt = pw.true_lnL(tp, df.N_P.values, df.D.values, 0.22 if r == "edu" else 0.0) + pw.noise(df, s, 0.5, rng)
        m = df.is_main.values if i % 5 else np.ones(len(df), bool)
        x, z, f = np.log(df[CONV_N[cv]].values[m]), np.log(df.D.values[m]), lt[m]
        a = est.cv_bandwidth(x, z, f, BW_MULTS)[0]
        b = cv_bandwidth_fast(x, z, f)
        rows.append(dict(i=i, same_bw=(a[3], a[4]) == (b[3], b[4]), rmse_diff=abs(a[0] - b[0])))
    d = pd.DataFrame(rows)
    log(f"fast-CV check: {int(d.same_bw.sum())}/{len(d)} identical bandwidths; max |RMSE difference| "
        f"{d.rmse_diff.max():.2e}")
    d.to_csv(os.path.join(TABLES, PREFIX + "_fastcv_check.csv"), index=False)
    return d


# ============================================================================ data generation
def gen(design, tp, s, rho, chi, rng, nval=1):
    """ln L for both corpora (and, if nval = 2, a second validation set with noise correlated 0.8), exactly as in
    m9_power.one_rep: web = truth, edu = truth with ln B lower by chi; common within-trunk shock with share rho."""
    out = {}
    for r in ("web", "edu"):
        df = design[r]
        lt = pw.true_lnL(tp, df.N_P.values, df.D.values, chi if r == "edu" else 0.0)
        e1 = pw.noise(df, s, rho, rng)
        if nval == 1:
            out[r] = (lt + e1,)
        else:
            e2 = RHO_VAL * e1 + np.sqrt(1 - RHO_VAL ** 2) * pw.noise(df, s, rho, rng)
            out[r] = (lt + e1, lt + e2)
    return out


# ============================================================================ Q3 (lean, identical statistic)
def q3_slope(N, D, M, f, restr, prep, th_start):
    """Plan Q3 statistic 'slope|chin_r|local' of m9_stages.q3_core: Chinchilla form fitted by Huber loss on the
    endpoints with M <= 100 (warm-started), local wedge from the local quadratic with prep's bandwidth, OLS slope of
    ln(w_restricted/w_local) on ln M over endpoints with M > 100."""
    L = np.exp(f)
    th, _ = est.fit_chin(N[restr], D[restr], L[restr], "huber", starts=th_start)
    c = est.local_coefs(prep["x"], prep["z"], f, list(zip(prep["x"], prep["z"])), prep["hx"], prep["hz"])
    good = (c[:, 1] < 0) & (c[:, 2] < 0)
    lw = np.where(good, est.lnw_local(c), np.nan)
    q = est.q3_stat(N, D, M, lw, th)
    return q["slope"], q["mean"], th, c[:, 0]


# ============================================================================ one coverage replication
def rep_cov(item, design=None, tp=None, s=None, rho=None, chi=0.22, B=99, convs=("F", "P", "T")):
    rep, seed = item
    rng = np.random.default_rng(seed)
    data = gen(design, tp, s, rho, chi, rng)
    universe = sorted(set(design["edu"].arch) | set(design["web"].arch))
    out = dict(rep=rep)
    blocks = {}
    for r in ("edu", "web"):
        df = design[r]
        m = df.is_main.values
        clus = df.arch.values[m]
        D = df.D.values[m]
        f = data[r][0][m]
        for cv in convs:
            N = df[CONV_N[cv]].values[m]
            prep = mf_prepare_fast(N, D, np.exp(f))
            pilot = est.pilot_fit(prep["x"], prep["z"], f, prep["hx"], prep["hz"])
            pc = PATHCONV[cv]
            sig, lok, _ = mf_sig(prep, f, pc)
            pop, _, _ = mf_sig(prep, pilot, pc, lok)
            e = f - pilot
            ecr2 = est.cr2_residuals(e, est.smoother_matrix(prep["x"], prep["z"], prep["hx"], prep["hz"]), clus)
            blocks[(r, cv)] = dict(N=N, D=D, clus=clus, prep=prep, pilot=pilot, lok=lok, pc=pc, est=sig, pop=pop,
                                   e=e, ecr2=ecr2)
        # Q3, convention P, main grid + high-M runs
        N3, D3, M3 = df.N_P.values, df.D.values, df.M_P.values
        f3 = data[r][0]
        restr = df.is_main.values & (M3 <= 100)
        clus3 = df.arch.values
        prep3 = mf_prepare_fast(N3, D3, np.exp(f3))
        starts = [chin_start(tp, chi if r == "edu" else 0.0)] + list(est.me.level_starts(N3[restr], D3[restr],
                                                                                         np.exp(f3[restr])))
        sl, mean_, th, pilot3 = q3_slope(N3, D3, M3, f3, restr, prep3, starts)
        slp, _, _, _ = q3_slope(N3, D3, M3, pilot3, restr, prep3, [th])
        e3 = f3 - pilot3
        e3c = est.cr2_residuals(e3, est.smoother_matrix(prep3["x"], prep3["z"], prep3["hx"], prep3["hz"]), clus3)
        blocks[(r, "Q3")] = dict(N=N3, D=D3, M=M3, clus=clus3, restr=restr, prep=prep3, pilot=pilot3, th=th,
                                 est=sl, pop=slp, e=e3, ecr2=e3c, mean=mean_)
    draws = {}
    for b in range(B):
        sh = st.Shock(seed + 1000 + b, "wild", universe)
        for (r, cv), bl in blocks.items():
            if cv == "Q3":
                fw = bl["pilot"] + sh.errors(bl["e"], bl["clus"], r)
                a = q3_slope(bl["N"], bl["D"], bl["M"], fw, bl["restr"], bl["prep"], [bl["th"]])[0]
                fc = bl["pilot"] + sh.errors(bl["ecr2"], bl["clus"], r)
                pb = mf_prepare_fast(bl["N"], bl["D"], np.exp(fc))
                c = q3_slope(bl["N"], bl["D"], bl["M"], fc, bl["restr"], pb, [bl["th"]])[0]
                draws.setdefault((r, "q3", "wcu"), []).append(a)
                draws.setdefault((r, "q3", "cr2cv"), []).append(c)
                continue
            fw = bl["pilot"] + sh.errors(bl["e"], bl["clus"], r)
            a, _, _ = mf_sig(bl["prep"], fw, bl["pc"], bl["lok"])
            fc = bl["pilot"] + sh.errors(bl["ecr2"], bl["clus"], r)
            pb = mf_prepare_fast(bl["N"], bl["D"], np.exp(fc))
            c, _, _ = mf_sig(pb, fc, bl["pc"], bl["lok"])
            for p in a:
                key = cv if cv != "P" else ("P" if p == "P" else "P6")
                draws.setdefault((r, f"mf_{key}", "wcu"), []).append(a[p])
                draws.setdefault((r, f"mf_{key}", "cr2cv"), []).append(c[p])
    # point estimates, populations and bootstrap summaries
    pts = {}
    for (r, cv), bl in blocks.items():
        if cv == "Q3":
            pts[(r, "q3")] = (bl["est"], bl["pop"])
            out[f"q3mean|{r}"] = bl["mean"]
            continue
        for p in bl["est"]:
            key = cv if cv != "P" else ("P" if p == "P" else "P6")
            pts[(r, f"mf_{key}")] = (bl["est"][p], bl["pop"][p])
            out[f"hx|{r}|{cv}"], out[f"hz|{r}|{cv}"] = bl["prep"]["mx"], bl["prep"]["mz"]
    for (r, stat), (e_, p_) in pts.items():
        out[f"{stat}|{r}|est"] = e_
        out[f"{stat}|{r}|pop"] = p_
        for sch in ("wcu", "cr2cv"):
            dr = np.asarray(draws[(r, stat, sch)], float)
            lo, hi = mc.basic_ci(e_, dr, p_)
            out[f"{stat}|{r}|{sch}|lo"], out[f"{stat}|{r}|{sch}|hi"], out[f"{stat}|{r}|{sch}|se"] = lo, hi, mc.sd_(dr)
    # equality (edu - web), joint draws
    for stat in sorted({k[1] for k in pts if k[1] != "q3"}):
        e_ = pts[("edu", stat)][0] - pts[("web", stat)][0]
        p_ = pts[("edu", stat)][1] - pts[("web", stat)][1]
        out[f"diff_{stat}|est"], out[f"diff_{stat}|pop"] = e_, p_
        for sch in ("wcu", "cr2cv"):
            dr = np.asarray(draws[("edu", stat, sch)], float) - np.asarray(draws[("web", stat, sch)], float)
            lo, hi = mc.basic_ci(e_, dr, p_)
            out[f"diff_{stat}|{sch}|lo"], out[f"diff_{stat}|{sch}|hi"], out[f"diff_{stat}|{sch}|se"] = lo, hi, mc.sd_(dr)
    return out


# ============================================================================ one tilt replication (two validation sets)
def rep_tilt(item, design=None, tp=None, s=0.005, rho=0.5, chi=0.0, B=199):
    """Plan (unrestricted wild cluster bootstrap, 'wcu'), CR2 interval, and restricted wild cluster bootstrap tests
    (raw and CR2 residuals of the Hicks-neutral fit) of chi = 0, on each of two validation sets, convention P, main
    grids -- the tilt block of m9_coverage.cov_rep, run once per validation set with its own draws."""
    import m2_est as me
    rep, seed = item
    rng = np.random.default_rng(seed)
    data = gen(design, tp, s, rho, chi, rng, nval=2)
    de, dw = design["edu"], design["web"]
    me_, mw_ = de.is_main.values, dw.is_main.values
    N = np.r_[dw.N_P.values[mw_], de.N_P.values[me_]]
    D = np.r_[dw.D.values[mw_], de.D.values[me_]]
    g = np.r_[np.zeros(mw_.sum(), int), np.ones(me_.sum(), int)]
    clus = np.r_[dw.arch.values[mw_], de.arch.values[me_]]
    uni = sorted(set(clus))
    out = dict(rep=rep)
    for vi in (0, 1):
        y = np.r_[data["web"][vi][mw_], data["edu"][vi][me_]]
        L = np.exp(y)
        starts, _ = est.ce_starts(N, D, L, g)
        th, _ = est.fit_ce(N, D, L, g, "huber", starts=starts)
        chi_hat = est.ce_tilt(th)
        yh = est.pred_ce(th, N, D, g)
        th0, yh0 = est.fit_ce_null(N, D, L, g, th)
        H = est.hat_from_jac(est.jac_ce(th, N, D, g))
        H0 = est.hat_from_jac(est.num_jac(lambda q: me.panel_pred("hicksE", q, np.log(N), np.log(D), g, 2), th0))
        E = {"wcu": y - yh, "cr2": est.cr2_residuals(y - yh, H, clus)}
        E0 = {"wcu": y - yh0, "cr2": est.cr2_residuals(y - yh0, H0, clus)}
        du = {v: [] for v in E}
        dr = {v: [] for v in E}
        for b in range(B):
            sh = st.Shock(seed + 9000 + 5000 * vi + b, "wild", uni)
            w = np.array([sh.w[c] for c in clus])
            for v in E:
                tb, _, _ = me.fit_panel_ls("CE", np.log(N), np.log(D), yh + w * E[v], g, 2, "huber", starts=[th],
                                           tight=False)
                du[v].append(est.ce_tilt(tb))
                tr, _, _ = me.fit_panel_ls("CE", np.log(N), np.log(D), yh0 + w * E0[v], g, 2, "huber", starts=[th],
                                           tight=False)
                dr[v].append(est.ce_tilt(tr))
        k = f"val{vi + 1}"
        out[f"chi|{k}"] = chi_hat
        for v in E:
            lo, hi = mc.basic_ci(chi_hat, du[v], chi_hat)
            drv = np.asarray(dr[v], float)
            drv = drv[np.isfinite(drv)]
            out.update({f"{v}|{k}|lo": lo, f"{v}|{k}|hi": hi, f"{v}|{k}|se": mc.sd_(du[v]),
                        f"{v}|{k}|p_wcr": float((1 + np.sum(np.abs(drv) >= abs(chi_hat))) / (len(drv) + 1)),
                        f"{v}|{k}|se_wcr": mc.sd_(drv)})
    return out


# ============================================================================ runner (incremental, resumable)
def run_items(fn, items, payload, path, procs=PROCS):
    done = set()
    if os.path.exists(path):
        try:
            done = set(pd.read_csv(path, usecols=["rep"]).rep.astype(int))
        except Exception:
            done = set()
    todo = [it for it in items if it[0] not in done]
    log(f"{os.path.basename(path)}: {len(done)} done, {len(todo)} to run with {procs} processes")
    if not todo:
        return
    from multiprocessing import get_context
    ctx = get_context("spawn")
    t0 = time.time()
    n = 0
    with ctx.Pool(procs, initializer=mc._init, initargs=(payload,)) as pool:
        for res in pool.imap_unordered(mc._work, [(fn, it) for it in todo], chunksize=1):
            if "_error" in res:
                log(f"  error: {res['_error']}")
                continue
            row = pd.DataFrame([res])
            row.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            n += 1
            if n % 25 == 0:
                log(f"  {n}/{len(todo)} in {time.time() - t0:.0f}s")
    log(f"{os.path.basename(path)}: finished {n} in {time.time() - t0:.0f}s")


def run_tilt(R=500, B=199):
    design = design_v2()
    tp = truth_v2("chin", 1.0, design)
    for chi in (0.0, 0.22):
        path = os.path.join(TABLES, f"{PREFIX}_reps_tilt_chi{chi}.csv")
        items = [(i, mc.seed_of(f"v2tilt|{chi}|{i}")) for i in range(R)]
        run_items(rep_tilt, items, dict(design=design, tp=tp, s=0.005, rho=0.5, chi=chi, B=B), path)


def run_cov(cell, R=500, B=99):
    kind, kap, s, rho, convs = COV_CELLS[cell]
    design = design_v2()
    tp = truth_v2(kind, kap, design)
    path = os.path.join(TABLES, f"{PREFIX}_reps_cov_{cell}.csv")
    items = [(i, mc.seed_of(f"v2cov|{cell}|{i}")) for i in range(R)]
    run_items(rep_cov, items, dict(design=design, tp=tp, s=s, rho=rho, chi=0.22, B=B, convs=convs), path)


def noise_free(kind="chin", kap=1.0):
    """Noise-free value of every coverage statistic on the design (the coverage target), s = 1e-6."""
    design = design_v2()
    tp = truth_v2(kind, kap, design)
    return rep_cov((0, 1), design=design, tp=tp, s=1e-6, rho=0.5, chi=0.22, B=21, convs=("F", "P", "T"))


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    import m9_coverage_v2 as mod   # workers pickle functions by module name, not __main__
    what = sys.argv[1]
    if what == "check":
        mod.check_fast_cv(int(sys.argv[2]) if len(sys.argv) > 2 else 120)
    elif what == "tilt":
        mod.run_tilt(int(sys.argv[2]), int(sys.argv[3]))
    elif what == "cov":
        mod.run_cov(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
