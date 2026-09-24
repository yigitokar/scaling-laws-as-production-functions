#!/usr/bin/env python
"""run.py -- single entry point of module m2_techpanel ("the technology across independent public sweeps").

Regenerates every output of the module from the raw data in data/raw/ (see code/data/download_public.sh and
code/data/download_m2_techpanel.sh):
    data/processed/m2_techpanel/        harmonized panels, bootstrap draws (.npy), stage caches (.pkl)
    output/tables/m2_*.csv|.tex         paper tables; output/tables/technology_registry_m2.csv
    output/figures/m2_*.pdf|.png        Figure 3 (sigma* forest) and supporting figures
Usage:  .venv/bin/python code/analysis/m2_techpanel/run.py            (full run, ~20-40 min on 6 processes)
        .venv/bin/python code/analysis/m2_techpanel/run.py --outputs-only   (re-draw tables/figures from caches)
All random draws use fixed seeds (SEED + CRC32 of the job key). At most 6 worker processes; no GPU.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"   # one BLAS thread per process: total CPU use stays <= N_JOBS processes

import argparse
import pickle
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m2_data as md  # noqa: E402
import m2_est as me  # noqa: E402

ROOT = md.ROOT
PROC = os.path.join(ROOT, "data", "processed", "m2_techpanel")
BOOTDIR = os.path.join(PROC, "boot")
CACHE = os.path.join(PROC, "cache")
for _d in (PROC, BOOTDIR, CACHE):
    os.makedirs(_d, exist_ok=True)

SEED = 20260923
N_JOBS = 6
B_MAIN = 400      # pairs/cluster bootstrap draws, primary single-sweep fits
B_ROB = 200       # robustness variants, DataDecide recipes, panel (CE) fits
B_Q = 200         # Kaplan-q family
B_NULL = 300      # residual bootstrap under H0 for the rank-one (translog) test
B_LOC = 300       # local nonparametric sigma (Farseer)
B_WILD_SMALL = 499  # wild cluster bootstrap LR tests, Gadre (n = 104)
B_WILD_DD = 99      # wild cluster bootstrap LR tests, DataDecide (n ~ 22k): key tests
B_WILD_DOM = 19     # DataDecide: clearly dominated common-E models (slow refits; p >= 0.05 resolution)
B_CE_DD = 100       # DataDecide common-exponent (CE) cluster bootstrap
EST = ("huber", "nls")
SMOKE = bool(os.environ.get("M2_SMOKE"))   # developer smoke test: tiny bootstraps, 4 DataDecide recipes, separate cache
if SMOKE:
    B_MAIN = B_ROB = B_Q = B_NULL = B_LOC = 6
    B_WILD_SMALL = B_WILD_DD = 6
    CACHE = os.path.join(PROC, "cache_smoke")
    os.makedirs(CACHE, exist_ok=True)


def seed_of(key):
    return SEED + zlib.crc32(key.encode()) % 100000


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def submit_chunks(ex, func, data, items, nchunks=N_JOBS, **kw):
    """Submit func(data, chunk, **kw) for nchunks interleaved chunks; returns (futures, chunk index lists)."""
    idx = list(range(len(items)))
    chunks = [idx[i::nchunks] for i in range(min(nchunks, len(items)))]
    futs = [ex.submit(func, data, [items[i] for i in c], **kw) for c in chunks]
    return futs, chunks


def collect(futs, chunks, n):
    out = [None] * n
    for f, c in zip(futs, chunks):
        for i, v in zip(c, f.result()):
            out[i] = v
    return out


# ============================================================================ variants (dataset x subset)
def build_variants(A):
    """Returns ordered dict key -> spec. role: primary (Table 3 / Figure 3), robust (registry + robustness table)."""
    V = {}

    def add(key, df, role, mode, eval_set, units=None, nconv=None, ddef=None, weights_col=None, cluster="cluster",
            spec=False, B=None, ests=EST):
        ds = key.split("|")[0]
        V[key] = dict(key=key, dataset=ds, subset=key.split("|")[1], df=df.reset_index(drop=True), role=role,
                      mode=mode, eval_set=eval_set, units=units or md.LUNITS[ds], nconv=nconv or md.NCONV[ds],
                      ddef=ddef or md.DDEF[ds], weights_col=weights_col, cluster=cluster, spec=spec,
                      B=B or (B_MAIN if role == "primary" else B_ROB), ests=ests)

    ch, fa, ga, ol, db = A["chinchilla"], A["farseer"], A["gadre"], A["olmo_ladder"], A["datablations"]
    # sl DEFAULT grid (4,500 starts) for the two largest designs; FAST grid (432) + 48 level-preserving starts elsewhere
    # (in development runs the FAST grid reproduced the DEFAULT-grid optimum to 1e-8 for every primary variant)
    add("chinchilla|all", ch, "primary", "default", "MassiveText val (digitized)", spec=True)
    add("farseer|all", fa, "primary", "default", "IntelliValSet-Raw en", spec=True)
    for c in ("C4", "RedPajama", "RefinedWeb"):
        add(f"gadre|{c}", ga[ga.corpus == c], "primary", "fast", "C4 val", spec=True)
    add("olmo_ladder|all", ol, "primary", "fast", "C4-en val", spec=True)
    add("datablations|single_epoch", db[db.epochs <= 1.0001], "primary", "fast", "C4 val", spec=True)
    # ---- robustness variants
    f2 = fa.copy(); f2["N"] = f2["N_emb"]
    add("farseer|N_incl_emb", f2, "robust", "fast", "IntelliValSet-Raw en", nconv="total incl. both embeddings")
    f3 = fa.copy(); f3["L"] = f3["L_train_smooth"]
    add("farseer|train_loss", f3, "robust", "fast", "training loss (smoothed, single epoch)",
        units="nats/token, training loss (65k tokenizer)")
    f4 = fa.copy(); f4["cluster"] = f4["N"].astype(str)
    add("farseer|cluster_N", f4, "robust", "fast", "IntelliValSet-Raw en", ddef=md.DDEF["farseer"] + "; clusters = 25 model sizes")
    for c in ("C4", "RedPajama", "RefinedWeb"):
        g1 = ga[ga.corpus == c].copy(); g1["L"] = g1["L_indist"]
        add(f"gadre|{c}_indist", g1, "robust", "fast", f"in-distribution Paloma ({md.GADRE_INDIST[[k for k, v in md.GADRE_CORPUS.items() if v == c][0]]})")
        g2 = ga[ga.corpus == c].copy(); g2["N"] = g2["N_ne"]
        add(f"gadre|{c}_Nnonemb", g2, "robust", "fast", "C4 val", nconv="non-embedding (open_lm params_no_embed)")
        g3 = ga[ga.corpus == c].copy(); g3["w"] = 1.0 / g3["se_lnL"] ** 2; g3["w"] = g3["w"] / g3["w"].mean()
        add(f"gadre|{c}_wls", g3, "robust", "fast", "C4 val (inverse-variance weights from token-level 95% CIs)",
            weights_col="w", ests=("nls",))
    o2 = ol.copy(); o2["L"] = o2["L_bpb"]
    add("olmo_ladder|task_bpb", o2, "robust", "fast", "mean of 11 downstream val-task BPB", units="bits/byte (task BPB)")
    add("datablations|le4_epochs", db[db.epochs <= 4.0001], "robust", "fast", "C4 val",
        ddef="tokens processed incl. up to 4 epochs of repetition (treated as unique)")
    # review addition: drop the 2 single-epoch runs with D/N < 0.4 (1.1B and 2.8B models on 100M tokens), the same
    # exclusion rule Besiroglu et al. apply to the Chinchilla sample
    add("datablations|single_epoch_M04", db[(db.epochs <= 1.0001) & (db.M >= 0.4)], "robust", "fast", "C4 val",
        ddef="tokens processed (single epoch; runs with D/N < 0.4 dropped)")
    return V


# ============================================================================ stage 2: technology by dataset
def stage_tech(A, ex):
    V = build_variants(A)
    log(f"stage tech: {len(V)} variants")
    # ---- point estimates (parallel over variants x estimators)
    jobs = []
    for k, v in V.items():
        d = v["df"]
        w = d[v["weights_col"]].values if v["weights_col"] else None
        for est in v["ests"]:
            jobs.append((k, est, d.N.values, d.D.values, d.L.values, w, "fast" if SMOKE else v["mode"], None))
    jobs.sort(key=lambda j: 0 if j[6] == "default" else 1)
    futs, chunks = submit_chunks(ex, me.w_point, None, jobs, nchunks=len(jobs))
    res = {}
    for f in futs:
        for (k, est, th, obj) in f.result():
            res[(k, est)] = dict(theta=th, obj=obj)
    log("  point estimates done")
    # ---- bootstraps (pairs / cluster), translog r on every draw
    pend = {}
    for k, v in V.items():
        d = v["df"]
        data = dict(N=d.N.values, D=d.D.values, L=d.L.values)
        if v["weights_col"]:
            data["w"] = d[v["weights_col"]].values
        cen = (np.log(d.N).mean(), np.log(d.D).mean())
        for est in v["ests"]:
            draws = me.boot_indices(d[v["cluster"]].values, v["B"], seed_of(k + est))
            pend[(k, est)] = submit_chunks(ex, me.w_chin, data, draws, est=est, init=res[(k, est)]["theta"], centers=cen) + (len(draws),)
    # ---- specification tests for primary variants
    spec_pend = {}
    for k, v in V.items():
        if not v["spec"]:
            continue
        d = v["df"]
        data = dict(N=d.N.values, D=d.D.values, L=d.L.values)
        cen = (np.log(d.N).mean(), np.log(d.D).mean())
        th_h = res[(k, "huber")]["theta"]
        # residual bootstrap under the fitted Chinchilla (H0 of the rank-one test)
        rng = np.random.default_rng(seed_of(k + "null"))
        items = [rng.integers(0, len(d), len(d)) for _ in range(B_NULL)]
        spec_pend[(k, "null")] = submit_chunks(ex, me.w_translog_null, data, items, est="huber", th0=th_h, centers=cen) + (B_NULL,)
    # q-family point estimates (both estimators) in the main process (fast), bootstrap in pool
    qres = {}
    for k, v in V.items():
        if not v["spec"]:
            continue
        d = v["df"]
        for est in EST:
            thq, fq = me.fit_q(d.N.values, d.D.values, d.L.values, est, th_chin=res[(k, est)]["theta"])
            qres[(k, est)] = dict(theta=thq, obj=fq)
        data = dict(N=d.N.values, D=d.D.values, L=d.L.values)
        cen = (np.log(d.N).mean(), np.log(d.D).mean())
        draws = me.boot_indices(d["cluster"].values, B_Q, seed_of(k + "q"))
        spec_pend[(k, "q")] = submit_chunks(ex, me.w_q, data, draws, est="huber", init=qres[(k, "huber")]["theta"],
                                            centers=cen) + (B_Q,)
        # review fix: rank-one test with E from the q family, null simulated under the fitted q model
        rng = np.random.default_rng(seed_of(k + "null_q"))
        items = [rng.integers(0, len(d), len(d)) for _ in range(B_NULL)]
        spec_pend[(k, "null_q")] = submit_chunks(ex, me.w_translog_null_q, data, items, est="huber",
                                                 thq0=qres[(k, "huber")]["theta"], centers=cen) + (B_NULL,)
    # CES-restricted fits (alpha = beta) via the panel machinery with a single group
    cres = {}
    for k, v in V.items():
        if not v["spec"]:
            continue
        d = v["df"]
        x, z, y = np.log(d.N.values), np.log(d.D.values), np.log(d.L.values)
        g0 = np.zeros(len(d), int)
        for est in EST:
            a, b, e, al, be = res[(k, est)]["theta"]
            starts = [np.array([a, b, e, 0.5 * (al + be)]), np.array([a, b, e, al]), np.array([a, b, e, be])]
            starts += [np.r_[s[:3], 0.5 * (s[3] + s[4])] for s in me.level_starts(d.N.values, d.D.values, d.L.values)]
            th, f, _ = me.fit_panel_ls("ces", x, z, y, g0, 1, est, starts=starts)
            cres[(k, est)] = dict(theta=th, obj=f)
    # ---- collect
    boot = {}
    for key, (futs, chunks, n) in pend.items():
        boot[key] = np.array(collect(futs, chunks, n))
    spec = {}
    for key, (futs, chunks, n) in spec_pend.items():
        spec[key] = np.array(collect(futs, chunks, n))
    log("  bootstraps done")
    return dict(V={k: {kk: vv for kk, vv in v.items()} for k, v in V.items()}, res=res, boot=boot, spec=spec,
                qres=qres, cres=cres)


# ============================================================================ stage 3: Farseer functional form and local sigma
def stage_farseer(A, ex, tech):
    fa = A["farseer"]
    N, D, L = fa.N.values, fa.D.values, fa.L.values
    out = {}
    # ---- Farseer's own functional form (Li et al. 2025b, Eq. 3), NLS on log loss
    st = me.farseer_starts(N, D, L)
    thF, fF = me.fit_farseer(N, D, L, "nls", starts=[st])
    out["farseer_theta"], out["farseer_obj"] = thF, fF
    # ---- translog with E free (7 parameters), NLS
    from scipy.optimize import least_squares
    n, d = np.log(N), np.log(D)
    cn, cd = n.mean(), d.mean()

    def tl_pred(p, n, d):
        nn, dd = n - cn, d - cd
        return np.logaddexp(p[0], p[1] + p[2] * nn + p[3] * dd + p[4] * nn ** 2 + p[5] * dd ** 2 + p[6] * nn * dd)

    def fit_tl(n, d, y, p0):
        r = least_squares(lambda p: tl_pred(p, n, d) - y, p0, method="lm", xtol=1e-14, ftol=1e-14, max_nfev=20000)
        return r.x

    thC = tech["res"][("farseer|all", "nls")]["theta"]
    Ehat = np.exp(thC[2])
    tl0 = me.translog(N, D, L, Ehat)
    X = np.c_[np.ones_like(n), n - cn, d - cd, (n - cn) ** 2, (d - cd) ** 2, (n - cn) * (d - cd)]
    c0 = np.linalg.lstsq(X, np.log(L - Ehat), rcond=None)[0]
    thT = fit_tl(n, d, np.log(L), np.r_[np.log(Ehat), c0])
    out["translogE_theta"] = thT
    # ---- in-sample comparison
    y = np.log(L)
    q_nls = tech["qres"][("farseer|all", "nls")]["theta"]

    def pred_chin(th, N, D):
        a, b, e, al, be = th
        return np.logaddexp(np.logaddexp(a - al * np.log(N), b - be * np.log(D)), e)

    def pred_q(th, N, D):
        a, b, e, al, be, q = th
        return np.logaddexp(q * np.logaddexp(a - al * np.log(N), b - be * np.log(D)), e)

    rows = []
    for name, k, pr in [("Chinchilla (Huber)", 5, pred_chin(tech["res"][("farseer|all", "huber")]["theta"], N, D)),
                        ("Chinchilla (NLS)", 5, pred_chin(thC, N, D)),
                        ("Kaplan-q (NLS)", 6, pred_q(q_nls, N, D)),
                        ("Translog in ln(L-E), E free (NLS)", 7, tl_pred(thT, n, d)),
                        ("Farseer Eq. 3 (NLS)", 9, me.farseer_lnL(thF, N, D))]:
        ll, aic, bic = me.gauss_ic(y - pr, k)
        rows.append(dict(model=name, k=k, rmse_in=float(np.sqrt(np.mean((y - pr) ** 2))), loglik=ll, aic=aic, bic=bic))
    # ---- out-of-sample: (a) hold out the 4 largest model sizes; (b) hold out top-decile compute
    splits = {"hold out N >= 3.8e9 (4 largest sizes)": N >= 3.8e9,
              "hold out top 10% of 6ND": 6 * N * D >= np.quantile(6 * N * D, 0.9)}
    oos = []
    for sname, te in splits.items():
        tr = ~te
        thh, _ = me.fit_chin(N[tr], D[tr], L[tr], "huber", grid="fast")
        thn, _ = me.fit_chin(N[tr], D[tr], L[tr], "nls", grid="fast")
        thq, _ = me.fit_q(N[tr], D[tr], L[tr], "nls", th_chin=thn)
        # review fix: starting values from the training sample only (the full-sample fits thF/thT contain the
        # test observations); the effect on out-of-sample RMSE is < 1.2e-4
        stt = me.farseer_starts(N[tr], D[tr], L[tr])
        thf, _ = me.fit_farseer(N[tr], D[tr], L[tr], "nls", starts=[stt])
        E_tr = np.exp(thn[2])
        X_tr = np.c_[np.ones(tr.sum()), n[tr] - cn, d[tr] - cd, (n[tr] - cn) ** 2, (d[tr] - cd) ** 2,
                     (n[tr] - cn) * (d[tr] - cd)]
        c_tr = np.linalg.lstsq(X_tr, np.log(L[tr] - E_tr), rcond=None)[0]
        tht = fit_tl(n[tr], d[tr], y[tr], np.r_[np.log(E_tr), c_tr])
        for name, pr in [("Chinchilla (Huber)", pred_chin(thh, N, D)), ("Chinchilla (NLS)", pred_chin(thn, N, D)),
                         ("Kaplan-q (NLS)", pred_q(thq, N, D)), ("Translog in ln(L-E), E free (NLS)", tl_pred(tht, n, d)),
                         ("Farseer Eq. 3 (NLS)", me.farseer_lnL(thf, N, D))]:
            e_ = y[te] - pr[te]
            oos.append(dict(split=sname, model=name, n_train=int(tr.sum()), n_test=int(te.sum()),
                            rmse_oos=float(np.sqrt(np.mean(e_ ** 2))), bias_oos=float(np.mean(e_))))
    out["ic"] = pd.DataFrame(rows)
    out["oos"] = pd.DataFrame(oos)
    log("  Farseer functional forms done")
    # ---- local nonparametric sigma (kernel-weighted local quadratic in (ln N, ln D) on ln L; E-free)
    x, z = np.log(N), np.log(D)
    sx, sz = x.std(), z.std()
    cv = me.loo_cv_bandwidth(N, D, L, (0.2, 0.3, 0.45, 0.6, 0.8, 1.0), sx, sz)
    (mx, mz) = min(cv, key=cv.get)
    hx, hz = mx * sx, mz * sz
    out["cv"], out["bw"] = cv, (mx, mz, hx, hz)
    # evaluation grid: 7 model sizes x log-spaced D/N, kept only where the kernel has support (n_eff >= 25)
    Ns = np.unique(N)
    Ngrid = Ns[np.linspace(0, len(Ns) - 1, 7).round().astype(int)]
    Mgrid = np.exp(np.linspace(np.log(1.0), np.log(1000.0), 13))
    meta = [(n0, m0) for n0 in Ngrid for m0 in Mgrid]
    pts = [(np.log(n0), np.log(n0 * m0)) for n0, m0 in meta]
    loc = me.local_sigma(N, D, L, pts, hx, hz)
    keep = loc[:, 3] >= 25
    # also require the point to lie inside the observed D range for that model size (+-10%)
    for i, (n0, m0) in enumerate(meta):
        dmin, dmax = D[N == n0].min(), D[N == n0].max()
        if not (0.9 * dmin <= n0 * m0 <= 1.1 * dmax):
            keep[i] = False
    pts = [p for p, k in zip(pts, keep) if k]
    loc = loc[keep]
    draws = me.boot_indices(fa["cluster"].values, B_LOC, seed_of("localsig"))
    futs, chunks = submit_chunks(ex, me.w_localsig, dict(N=N, D=D, L=L), draws, pts=pts, hx=hx, hz=hz)
    lb = np.array(collect(futs, chunks, len(draws)))
    # model-implied sigma at the same points
    mH = me.chin_from_theta(tech["res"][("farseer|all", "huber")]["theta"])
    sig_chin = np.array([mH.sigma(np.exp(p[0]), np.exp(p[1])) for p in pts])
    fF = lambda a, b: float(me.farseer_lnL(thF, np.exp(a), np.exp(b)))
    sig_far = np.array([me.numderiv_sigma(fF, p[0], p[1])[0] for p in pts])
    out["local"] = pd.DataFrame(dict(lnN=[p[0] for p in pts], lnD=[p[1] for p in pts], N=[np.exp(p[0]) for p in pts],
                                     M=[np.exp(p[1] - p[0]) for p in pts], sigma_np=loc[:, 0], epsN_np=loc[:, 1],
                                     epsD_np=loc[:, 2], neff=loc[:, 3],
                                     sigma_np_lo=np.nanpercentile(lb, 2.5, axis=0),
                                     sigma_np_hi=np.nanpercentile(lb, 97.5, axis=0),
                                     sigma_np_se=np.nanstd(lb, axis=0, ddof=1),
                                     sigma_chin=sig_chin, sigma_farseer=sig_far))
    out["local_boot"] = lb
    # sensitivity of the median local sigma to the bandwidth
    sens = []
    for fac in (1.0, 1.5, 2.0):
        l2 = me.local_sigma(N, D, L, pts, fac * hx, fac * hz)
        sens.append(dict(bw_factor=fac, median_sigma=float(np.median(l2[:, 0])), p10=float(np.percentile(l2[:, 0], 10)),
                         p90=float(np.percentile(l2[:, 0], 90))))
    out["bw_sens"] = pd.DataFrame(sens)
    log("  local sigma done")
    return out


# ============================================================================ stage 4: neutrality of data quality
NEUT_MODELS = ["pooled", "Eshift", "hicks", "daug", "paug", "hicksE", "daugE", "paugE", "CE"]


def neutrality_panel(df, gcol, cell_col, ex, B_wild, label, full_mode, tests=None, B_ce=B_ROB):
    """All nested panel models for one multi-corpus experiment + wild cluster bootstrap LR tests."""
    groups = sorted(df[gcol].unique())
    R = len(groups)
    g = df[gcol].map({r: i for i, r in enumerate(groups)}).values
    x, z, y = np.log(df.N.values), np.log(df.D.values), np.log(df.L.values)
    cl = pd.factorize(df[cell_col])[0]
    out = dict(groups=groups, R=R, n=len(y), n_clusters=int(cl.max() + 1))
    # ---- unrestricted: separate Chinchilla per group (NLS and Huber)
    # separate fits per group, run in the worker pool (one job per group x estimator)
    th_pool_est = {}
    if not isinstance(full_mode, str):       # large panel: pooled fit (warm from a supplied start) seeds the groups
        for est in EST:
            th_pool_est[est] = me.fit_chin(df.N.values, df.D.values, df.L.values, est, init=full_mode)[0]
    jobs = []
    for est in EST:
        for r in range(R):
            i = g == r
            Ni, Di, Li = df.N.values[i], df.D.values[i], df.L.values[i]
            if isinstance(full_mode, str):   # small panel: sl FAST grid + level-preserving starts (best of both)
                jobs.append(((est, r), est, Ni, Di, Li, None, "fast", None))
            else:
                jobs.append(((est, r), est, Ni, Di, Li, None, "starts",
                             [full_mode, th_pool_est[est]] + me.level_starts(Ni, Di, Li)))
    futs, _ = submit_chunks(ex, me.w_point, None, jobs, nchunks=len(jobs))
    fr = {}
    for f in futs:
        for (key, est, th, obj) in f.result():
            fr[key] = (th, obj)
    full = {est: dict(thetas=[fr[(est, r)][0] for r in range(R)], obj=float(sum(fr[(est, r)][1] for r in range(R))))
            for est in EST}
    P = me.full_perobs(full["nls"]["thetas"], g)
    Ph = me.full_perobs(full["huber"]["thetas"], g)
    # ---- restricted models (NLS for tests, Huber for CE technology)
    fits = {}
    for m in NEUT_MODELS:
        for est, PP in (("nls", P), ("huber", Ph)):
            if est == "huber" and m != "CE":
                continue
            starts = [me.embed(m, PP, g, R)]
            if m != "pooled":
                starts.append(me.embed(m, me.panel_perobs("pooled", fits[("pooled", "nls")]["theta"], g, R), g, R))
            th, f, names = me.fit_panel_ls(m, x, z, y, g, R, est, starts=starts)
            fits[(m, est)] = dict(theta=th, obj=f, names=names, k=len(th))
    ssr = {m: 2 * fits[(m, "nls")]["obj"] for m in NEUT_MODELS}
    ssr["full"] = 2 * full["nls"]["obj"]
    kpar = {m: fits[(m, "nls")]["k"] for m in NEUT_MODELS}
    kpar["full"] = 5 * R
    out.update(fits=fits, full=full, ssr=ssr, k=kpar)
    # ---- wild cluster restricted bootstrap LR tests
    if tests is None:   # (m0, m1, B): H0 model m0 nested in m1
        tests = [(m, "CE", B_wild) for m in NEUT_MODELS if m != "CE"] + [("CE", "full", B_wild)]
    rng = np.random.default_rng(seed_of(label + "wild"))
    G = cl.max() + 1
    # Rademacher weights (Webb six-point weights would be preferable with < 12 clusters; we have 35-44)
    Wall = [rng.choice([-1.0, 1.0], size=G) for _ in range(max(t[2] for t in tests))]
    pend = {}
    for (m0, m1, Bt) in tests:
        W = Wall[:Bt]
        th0 = fits[(m0, "nls")]["theta"]
        yhat0 = me.panel_pred(m0, th0, x, z, g, R)
        th1 = full["nls"]["thetas"] if m1 == "full" else fits[(m1, "nls")]["theta"]
        data = dict(x=x, z=z, y=y, g=g, R=R, cl=cl, yhat0=yhat0)
        pend[(m0, m1)] = submit_chunks(ex, me.w_wild, data, W, m0=m0, m1=m1, th0=th0, th1=th1) + (Bt,)
    # ---- CE technology: cluster (cell) pairs bootstrap, both estimators
    ce_pend = {}
    for est in EST:
        draws = me.boot_indices(cl, B_ce, seed_of(label + "CEboot" + est))
        data = dict(x=x, z=z, y=y, g=g, R=R)
        ce_pend[est] = submit_chunks(ex, me.w_panel_ls, data, draws, model="CE", est=est, init=fits[("CE", est)]["theta"]) + (B_ce,)
    out["tests"] = {}
    n = len(y)
    for (m0, m1), (futs, chunks, B) in pend.items():
        lrb = np.array(collect(futs, chunks, B))
        s0, s1 = ssr[m0], ssr[m1]
        lr = n * np.log(s0 / s1)
        out["tests"][(m0, m1)] = dict(LR=lr, df=kpar[m1] - kpar[m0], p=(1 + np.sum(lrb >= lr)) / (B + 1), LRboot=lrb)
    out["ce_boot"] = {est: np.array(collect(*ce_pend[est][:2], ce_pend[est][2])) for est in EST}
    return out


def stage_neutrality(A, ex):
    out = {}
    ga = A["gadre"]
    log("stage neutrality: Gadre")
    out["gadre"] = neutrality_panel(ga, "corpus", "cell", ex, B_WILD_SMALL, "gadre", full_mode="grid")
    dd = A["datadecide"]
    log("stage neutrality: DataDecide (pooled start)")
    th_pool = me.fit_chin_multi(dd.N.values, dd.D.values, dd.L.values, "nls",
                                starts=me.level_starts(dd.N.values, dd.D.values, dd.L.values))[0]
    Bk, Bs = B_WILD_DD, (6 if SMOKE else B_WILD_DOM)     # key tests / clearly dominated common-E models (slow refits)
    dd_tests = [("pooled", "CE", Bs), ("Eshift", "CE", Bk), ("hicks", "CE", Bs), ("daug", "CE", Bs), ("paug", "CE", Bs),
                ("hicksE", "CE", Bk), ("daugE", "CE", Bk), ("paugE", "CE", Bk), ("CE", "full", Bk)]
    out["datadecide"] = neutrality_panel(dd, "recipe", "cell", ex, B_WILD_DD, "datadecide", full_mode=th_pool,
                                         tests=dd_tests, B_ce=6 if SMOKE else B_CE_DD)
    log("  DataDecide main done")
    # robustness: average CE over 11 validation sets as the output; and later checkpoints only (D/N >= 20)
    d2 = dd.copy(); d2["L"] = d2["L_avg11"]
    Br = 6 if SMOKE else 49
    key_tests = [("Eshift", "CE", Br), ("hicksE", "CE", Br), ("CE", "full", Br)]
    out["datadecide_avg11"] = neutrality_panel(d2, "recipe", "cell", ex, Br, "datadecide_avg11", full_mode=th_pool,
                                               tests=key_tests, B_ce=6 if SMOKE else 30)
    d3 = dd[dd.M >= 20].copy()
    out["datadecide_M20"] = neutrality_panel(d3, "recipe", "cell", ex, Br, "datadecide_M20", full_mode=th_pool,
                                             tests=key_tests, B_ce=6 if SMOKE else 30)
    # per-recipe bootstrap (run-level pairs) for the recipe forest panel and registry
    pend = {}
    for key in ("datadecide",):
        o = out[key]
        for est in EST:
            for r, rec in enumerate(o["groups"]):
                sub = dd[dd.recipe == rec]
                draws = me.boot_indices(sub["run_id"].values, B_ROB, seed_of(rec + est))
                data = dict(N=sub.N.values, D=sub.D.values, L=sub.L.values)
                pend[(rec, est)] = submit_chunks(ex, me.w_recipe_boot, data, draws, nchunks=2, est=est,
                                                 init=o["full"][est]["thetas"][r]) + (B_ROB,)
    out["recipe_boot"] = {k: np.array(collect(*v[:2], v[2])) for k, v in pend.items()}
    log("  per-recipe bootstraps done")
    return out


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-only", action="store_true", help="re-draw tables/figures/memo inputs from caches")
    ap.add_argument("--stages", default="tech,farseer,neutrality")
    args = ap.parse_args()
    t0 = time.time()
    A = md.load_all()
    if SMOKE:
        keep = sorted(A["datadecide"].recipe.unique())[:4]
        A["datadecide"] = A["datadecide"][A["datadecide"].recipe.isin(keep)].reset_index(drop=True)
    # ---- processed panels (local only: Farseer/analyzing-chinchilla have no license -> do not redistribute)
    for k, v in A.items():
        v.to_csv(os.path.join(PROC, f"panel_{k}.csv"), index=False)
    stages = [] if args.outputs_only else args.stages.split(",")
    with ProcessPoolExecutor(max_workers=N_JOBS) as ex:
        if "tech" in stages:
            tech = stage_tech(A, ex)
            pickle.dump(tech, open(os.path.join(CACHE, "tech.pkl"), "wb"))
        tech = pickle.load(open(os.path.join(CACHE, "tech.pkl"), "rb"))
        if "farseer" in stages:
            far = stage_farseer(A, ex, tech)
            pickle.dump(far, open(os.path.join(CACHE, "farseer.pkl"), "wb"))
        if "neutrality" in stages:
            neu = stage_neutrality(A, ex)
            pickle.dump(neu, open(os.path.join(CACHE, "neutrality.pkl"), "wb"))
    far = pickle.load(open(os.path.join(CACHE, "farseer.pkl"), "rb"))
    neu = pickle.load(open(os.path.join(CACHE, "neutrality.pkl"), "rb"))
    import m2_out as mo
    mo.write_all(A, tech, far, neu)
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
