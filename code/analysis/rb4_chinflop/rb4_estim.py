"""rb4_estim.py -- rebuilt Chinchilla IsoFLOP designs and the budget-level model-free sigma* under each accounting.

Steps
1. match(): each of the 245 digitized runs (Epoch AI's digitization of Hoffmann et al.'s Figure 4, loaded with ra1's
   loader and m1's budget reconstruction) is matched to the Table A9 model with the nearest reported parameter
   count (in logs). Match quality: |log error|, gap to the second-nearest model.
2. coords_test(): which FLOP count defined the budgets? Figure 4 draws the fitted L(N, D) as contours over
   (FLOPs, N), which requires FLOPs = 6ND on its horizontal axis, so each run is plotted at 6 N D_true. If the tokens
   were set so that F(N) D_true = C_b for an accounting F, the digitized coordinate satisfies
       log10 C_dig - log10 C_b = -log10 r_F(N),   r_F = F/(6T),
   so the within-budget residual dev + log10 r_F must be flat in N. Tested for F = T4, A, X and 6T (r = 1).
3. How D is recovered. Epoch's D = C_dig/(6N). Under the hypothesis supported in step 2 this is D_true itself. The
   actual training FLOPs are then C = F(N) D_true (= C_b under the budget-setting count) and N_F = C/(6D) = F/6: the
   token count cancels, so N_F does not depend on how D is recovered. What does depend on it is whether the runs of
   one nominal budget share one actual budget in a given accounting -- exactly (primary: T4 under the supported
   hypothesis) or only up to a factor rho_i = F_acc(N_i)/F_budget(N_i) that varies along the profile.
4. estimate(): ra1's estimator and bootstrap, unchanged (isoflop.design_estimate: order 2 (ra1's F test), h = 1,
   path-centred windows, log-cubic frontier; boot_design with B = 999 and ra1's seed for Chinchilla, so every
   accounting shares ra1's random numbers), applied to x = ln N_acc on the nominal budgets.
     'mechanical' (as ra1 does for Marin's configuration count): x = ln N_acc, runs at C_b.
     'corrected': runs moved onto exact isocosts of the accounting, L_i + [Lhat(N_i, D_true/rho_i) - Lhat(N_i, D_true)]
       with Lhat the kappa-free Chinchilla surface fitted (Huber, ra1's parametric.fit_kappa) to the same runs; a
       first-order check uses L_i + |s_b| ln rho_i with the budget's frontier slope s_b (model-free at the minimum).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import rb4common as cm
import rb4_arch as ar

rc = cm.rc
H_PRIMARY = 1.0
ORDER = 2          # ra1's pooled F test selects order 2 for Chinchilla (ra1_modelfree_isoflop_order.csv); re-tested


# ============================================================================ runs and matching
def load_runs():
    """Digitized runs (all 245) with the matched Table A9 model. In memory only (never written row by row)."""
    df = rc.load_chinchilla_all()
    A = ar.arch_table()
    la = np.log(A["N_rep_M"].values * 1e6)
    dist = np.abs(np.log(df["N"].values)[:, None] - la[None, :])
    j = dist.argmin(1)
    srt = np.sort(dist, 1)
    df["row"] = j
    df["match_err"] = np.log(df["N"].values) - la[j]
    df["match_gap2"] = srt[:, 1]
    df["match_ratio"] = srt[:, 0] / srt[:, 1]          # < 0.5: nearer to its own row than halfway to the next
    df["dev_raw"] = df["dev"] + df.attrs["c_offset_dex"]  # log10 C_dig - log10 C_b (before m1's offset removal)
    for k in ar.ACCOUNTINGS:
        df[f"F_{k}"] = A[f"F_{k}"].values[j]
    df["T_arch"] = A["T"].values[j]
    df["P_arch"] = A["P"].values[j]
    return df, A


def match_summary(df, A):
    """Per Table A9 model: runs matched (profile / off-profile) and the largest |log error| (derived statistics)."""
    g = df.groupby("row")
    out = A[["row", "N_rep_M", "d_model", "n_layers"]].copy()
    out["n_profile_runs"] = out.row.map(df[df.iso].groupby("row").size()).fillna(0).astype(int)
    out["n_offprofile_runs"] = out.row.map(df[~df.iso].groupby("row").size()).fillna(0).astype(int)
    out["max_abs_log_err"] = out.row.map(g["match_err"].apply(lambda s: float(np.abs(s).max())))
    out["budgets"] = out.row.map(df[df.iso].groupby("row")["budget"].apply(lambda s: " ".join(f"{b:.0e}" for b in sorted(set(s)))))
    tot = dict(n_runs=len(df), n_profile=int(df.iso.sum()), models_matched=int(df.row.nunique()),
               models_matched_profile=int(df[df.iso].row.nunique()),
               max_abs_log_err=float(np.abs(df.match_err).max()),
               p99_abs_log_err=float(np.percentile(np.abs(df.match_err), 99)),
               median_abs_log_err=float(np.median(np.abs(df.match_err))),
               max_match_ratio=float(df.match_ratio.max()),
               n_ambiguous=int((df.match_ratio > 0.5).sum()),
               n_ambiguous_profile=int(((df.match_ratio > 0.5) & df.iso).sum()),
               smallest_model_M=float(A.N_rep_M[df.row.min()]), largest_model_M=float(A.N_rep_M[df.row.max()]))
    return out, tot


# ============================================================================ which count set the budgets?
def _fe_ols(y, X_fe, z, clusters):
    """OLS of y on budget dummies and z; slope, HC1 s.e., CR1 s.e. by cluster (budget)."""
    X = np.column_stack([X_fe, z])
    n, p = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    hc = XtXi @ (X.T * e ** 2) @ X @ XtXi * n / (n - p)
    G = pd.factorize(clusters)[0]
    meat = np.zeros((p, p))
    for g in range(G.max() + 1):
        s = X[G == g].T @ e[G == g]
        meat += np.outer(s, s)
    ng = G.max() + 1
    cr = XtXi @ meat @ XtXi * (ng / (ng - 1)) * ((n - 1) / (n - p))
    return float(b[-1]), float(np.sqrt(hc[-1, -1])), float(np.sqrt(cr[-1, -1])), e


def coords_test(df):
    """Within-budget test of the budget-setting count on the 137 profile runs (module docstring, step 2)."""
    from scipy.stats import t as tdist
    iso = df[df.iso]
    y0 = iso["dev_raw"].values
    FE = pd.get_dummies(iso["budget"]).astype(float).values
    l10N = np.log10(iso["N"].values)
    rows = []
    for k in ("T4", "A", "X", "6T"):
        lr = np.log10(iso[f"F_{k}"].values / (6 * iso["T_arch"].values))
        e = y0 + lr                                           # residual under H_k (up to a common offset)
        b, se_hc, se_cr, res = _fe_ols(e, FE, l10N, iso["budget"].values)
        g, g_hc, g_cr, res2 = _fe_ols(y0, FE, -lr, iso["budget"].values) if np.ptp(lr) > 0 else (np.nan,) * 3 + (None,)
        rows.append(dict(accounting=k, label=ar.ACC_LABEL[k] if k != "6T" else "6ND (budgets = 6 T D; r = 1)",
                         n=len(e), mean_offset_dex=float(e.mean()), sd_resid_dex=float(e.std(ddof=1)),
                         within_sd_dex=float(np.std(res, ddof=FE.shape[1] + 1)),
                         slope_resid_on_log10N=b, se_hc1=se_hc, se_cr1_budget=se_cr,
                         t_cr1=b / se_cr, p_cr1=float(2 * tdist.sf(abs(b / se_cr), 8)),
                         coef_on_minus_log10r=g, se_coef_cr1=g_cr,
                         range_log10r=float(np.ptp(lr)), mean_log10r=float(lr.mean())))
    return pd.DataFrame(rows)


# ============================================================================ designs
def design_df(df, x, L=None, name="Chinchilla"):
    """Chinchilla IsoFLOP design in ra1's layout (isoflop_designs): x, L, C_b, c, b; sorted by (b, x)."""
    d = df[df.iso].copy()
    d["C_b"] = d["budget"]
    d["x"] = np.asarray(x, float)[df.iso.values] if len(x) == len(df) else np.asarray(x, float)
    if L is not None:
        d["L"] = np.asarray(L, float)[df.iso.values] if len(L) == len(df) else np.asarray(L, float)
    d["c"] = np.log(d["C_b"].values)
    d["b"] = pd.factorize(d["C_b"], sort=True)[0]
    return d.sort_values(["b", "x"]).reset_index(drop=True)


def x_of(df, acc):
    """ln N_acc per run: 'ra1' = ln N_dig (ra1's T); '6T' ln T_arch; '6P' ln P; F accounts ln(F/6)."""
    if acc == "ra1":
        return np.log(df["N"].values)
    if acc == "6T":
        return np.log(df["T_arch"].values)
    if acc == "6P":
        return np.log(df["P_arch"].values)
    return np.log(df[f"F_{acc}"].values / 6.0)


def fit_surface(df, budget_count="T4"):
    """kappa-free Chinchilla surface (Huber; ra1's parametric) on the profile runs with N = T and D = C_b/F_budget,
    the token count under the budget hypothesis. Returns Lhat(N, D) (levels)."""
    import parametric as pm
    iso = df[df.iso]
    N = iso["T_arch"].values
    D = iso["budget"].values / iso[f"F_{budget_count}"].values
    L = iso["L"].values
    th, _ = pm.fit_chin(N, D, L, starts=[rc.sl.BESIROGLU.theta])
    p, _ = pm.fit_kappa(N, D, L, th_chin=th)
    return (lambda n, d: np.exp(pm.pred_kappa(p, n, d))), p


def corrected_L(df, acc, budget_count, surface=None, slope_by_budget=None):
    """Losses moved from the observed isocost (count `budget_count`) to the exact isocost of accounting `acc` at the
    same nominal budget: D' = D_true/rho, rho = F_acc/F_budget. surface: Lhat (kappa family); slope_by_budget: dict
    budget -> |s_b| for the first-order version L + |s_b| ln rho."""
    Fb = df[f"F_{budget_count}"].values
    Fa = df[f"F_{acc}"].values
    rho = Fa / Fb
    L = df["L"].values.astype(float).copy()
    if surface is not None:
        N = df["T_arch"].values
        D = df["budget"].values / Fb
        return L + surface(N, D / rho) - surface(N, D), rho
    s = np.array([slope_by_budget.get(b, np.nan) for b in df["budget"].values])
    return L + s * np.log(rho), rho


# ============================================================================ estimation (ra1 code, unchanged)
def _est_job(job, runs_payload):
    """Worker: one accounting variant, ra1's primary estimator + design-conditional bootstrap."""
    import isoflop as iso_
    d = pd.DataFrame(runs_payload)
    d["x"] = job["x"]
    if job.get("L") is not None:
        d["L"] = job["L"]
    d = d.sort_values(["b", "x"]).reset_index(drop=True)
    ot = iso_.order_test(d, H_PRIMARY)
    res = iso_.design_estimate(d, ORDER, H_PRIMARY, "auto", "path")
    out = dict(key=job["key"], order_F=ot["F"], order_p=ot["p"], ok=bool(res.get("ok")))
    if not res.get("ok"):
        out["k_valid"] = len(res.get("valid", []))
        return out
    if job["B"] > 0:
        bt = iso_.boot_design(d, res, job["B"], job["seed"], budget_shock=True)
        s, rows, inv, _ = iso_.summarize_design("Chinchilla", res, bt, job["B"])
        out.update(summary=s, rows=rows, invalid=inv)
    else:
        out.update(rows=[dict(design="Chinchilla", budget_C=res["fits"][b]["C"], sigma=float(sg), S=float(S_),
                              nwin=res["fits"][b]["nwin"], nleft=res["fits"][b]["nleft"], nright=res["fits"][b]["nright"],
                              Nstar=float(np.exp(res["fits"][b]["xstar"])), curv=float(cv), slope=float(sl_))
                         for b, sg, S_, cv, sl_ in zip(res["valid"], res["sigma"], res["S"], res["curv"], res["slope"])])
    out["masks"] = {int(b): res["fits"][b]["mask"].tolist() for b in res["valid"]}
    out["window_runs"] = {int(b): d.loc[res["fits"][b]["idx"][res["fits"][b]["mask"]], "run_id"].tolist()
                          for b in res["valid"]}
    out["frontier_slope"] = {float(res["fits"][b]["C"]): float(s_) for b, s_ in zip(res["valid"], res["slope"])}
    return out


VARIANTS = [
    # key, accounting of x, budget hypothesis, correction, label
    ("T_ra1", "ra1", "T4", None, "T (total, digitized N): ra1's estimate, reproduced"),
    ("NF_T4", "T4", "T4", None, "N_F, Hoffmann's implemented count (T4) -- primary; exact isocosts"),
    ("NF_A", "A", "T4", None, "N_F, Appendix F as printed (A); runs at nominal budgets"),
    ("NF_X", "X", "T4", None, "N_F, executed FLOPs (X); runs at nominal budgets"),
    ("T", "6T", "T4", None, "T, total parameters (architecture count)"),
    ("P", "6P", "T4", None, "P, non-embedding parameters (T - Vd)"),
    ("NF_A|corr", "A", "T4", "kappa", "N_F (A), runs moved to exact A-isocosts (budgets in T4)"),
    ("NF_X|corr", "X", "T4", "kappa", "N_F (X), runs moved to exact X-isocosts (budgets in T4)"),
    ("T|corr", "6T", "T4", "kappa", "T, runs moved to exact 6TD isocosts (budgets in T4)"),
    ("P|corr", "6P", "T4", "kappa", "P, runs moved to exact 6PD isocosts (budgets in T4)"),
    ("NF_T4|6ND", "T4", "6T", "kappa", "N_F (T4) if tokens had been set by D = C_b/(6T) (rejected by coords test)"),
    ("NF_A|6ND", "A", "6T", "kappa", "N_F (A) if tokens had been set by D = C_b/(6T) (rejected by coords test)"),
]
VLABEL = {v[0]: v[4] for v in VARIANTS}


def run_variants(df, log=cm.log):
    """All variants with ra1's bootstrap (seed = ra1's Chinchilla seed: common random numbers), plus first-order
    ('slope') versions of the corrected variants (point estimates) and a fixed-window-membership check."""
    iso = df[df.iso].reset_index(drop=True).copy()
    iso["run_id"] = np.arange(len(iso))
    # payload in ra1's column layout; 'run_id' identifies runs across variants
    keep = ["run_id", "N", "D", "L", "budget", "C_b", "c", "b", "T_arch", "P_arch"] + [f"F_{k}" for k in ar.ACCOUNTINGS]
    iso2 = iso.copy()
    iso2["C_b"] = iso2["budget"]
    iso2["c"] = np.log(iso2["C_b"].values)
    iso2["b"] = pd.factorize(iso2["C_b"], sort=True)[0]
    payload = {k: iso2[k].values for k in keep}
    surf = {}
    for bc in ("T4", "6T"):
        surf[bc], _ = fit_surface(iso2.assign(iso=True), bc)
    seed = rc.seed_of("iso|Chinchilla")                        # ra1's seed for its primary Chinchilla bootstrap
    jobs = []
    for key, acc, bc, corr, lab in VARIANTS:
        x = x_of(iso2, acc)
        L = None
        if corr == "kappa":
            L, _ = corrected_L(iso2, acc, bc, surface=surf[bc])
        jobs.append(dict(key=key, x=x, L=L, B=cm.B_ISO, seed=seed))
    log(f"  estimating {len(jobs)} accounting variants (B = {cm.B_ISO}, ra1's seed {seed}) on {min(cm.N_PROC, len(jobs))} processes")
    res = rc.pmap(_est_job, jobs, dict(runs_payload=payload), procs=cm.N_PROC, chunksize=1)
    out = {}
    for j, r in zip(jobs, res):
        if "_error" in r:
            log(f"  variant {j['key']}: ERROR {r['_error']}")
            continue
        out[j["key"]] = r
    # first-order corrections with the model-free frontier slope of the primary (T4) fit at each budget
    fs = out["NF_T4"]["frontier_slope"]
    cb = np.log(np.array(sorted(fs)))
    sb = np.array([abs(fs[c]) for c in sorted(fs)])
    slope_by_budget = {b: float(np.interp(np.log(b), cb, sb)) for b in rc.CHIN_BUDGETS}
    jobs2 = []
    for key, acc, bc, corr, lab in VARIANTS:
        if corr != "kappa":
            continue
        L, _ = corrected_L(iso2, acc, bc, slope_by_budget=slope_by_budget)
        jobs2.append(dict(key=key + "|slope", x=x_of(iso2, acc), L=L, B=0, seed=seed))
    res2 = rc.pmap(_est_job, jobs2, dict(runs_payload=payload), procs=cm.N_PROC, chunksize=1)
    for j, r in zip(jobs2, res2):
        if "_error" not in r:
            out[j["key"]] = r
    return out, iso2, surf


def fixed_membership(iso2, out, acc="T4", ref="T_ra1"):
    """Point estimates in accounting `acc` with each budget's window membership held at the reference (T) windows:
    a quadratic in x_acc on exactly the runs of the reference window; frontier as ra1 (log-cubic of L*)."""
    import isoflop as iso_
    x = x_of(iso2, acc)
    L = iso2["L"].values
    wr = out[ref]["window_runs"]
    C = iso2.groupby("b")["C_b"].first()
    rows, cs, Ls, curv = [], [], [], []
    for b, ids in sorted(wr.items()):
        ids = np.array(ids)
        xx, yy = x[ids], L[ids]
        cf = np.polyfit(xx - xx.mean(), yy, 2)[::-1]
        t, k = iso_._argmin_poly(cf)
        cs.append(np.log(C.loc[b]))
        Ls.append(float(iso_._polyval(cf, t)))
        curv.append(k)
        rows.append(dict(budget_C=float(C.loc[b]), nwin=len(ids), curv=float(k)))
    fr = iso_.frontier(np.array(cs), np.array(Ls), iso_.fkind_auto(len(cs)))
    S = np.array(curv) / np.abs(fr["slope"])
    for r, s_ in zip(rows, S):
        r["S"], r["sigma"] = float(s_), float(2 / (2 + s_))
    return pd.DataFrame(rows)


# ============================================================================ eta by budget
def eta_by_budget(iso2, out, A, key="NF_T4"):
    """Elasticity of FLOPs per token (per unit of the reported total count) with respect to T, by budget:
    (i) OLS slope of ln r_acc on ln T over the runs in the primary window of each budget; (ii) a Gaussian-kernel
    local-linear slope across the 50 Table A9 models at the budget's argmin (bandwidth 0.35 in ln T)."""
    wr = out[key]["window_runs"]
    C = iso2.groupby("b")["C_b"].first()
    Nstar = {float(r["budget_C"]): float(r["Nstar"]) for r in out["T_ra1"]["rows"]}
    lT = np.log(A["T"].values)
    rows = []
    for acc in ("T4", "A", "X", "6P"):
        lr_all = np.log(A[f"F_{acc}"].values / (6 * A["T"].values))
        for b, ids in sorted(wr.items()):
            ids = np.array(ids)
            xt = np.log(iso2["T_arch"].values[ids])
            yr = np.log(iso2[f"F_{acc}"].values[ids] / (6 * iso2["T_arch"].values[ids]))
            eta_win = float(np.polyfit(xt, yr, 1)[0]) if np.ptp(xt) > 0 else np.nan
            cb = float(C.loc[b])
            ln0 = np.log(Nstar.get(cb, np.exp(np.mean(xt))))
            w = np.exp(-0.5 * ((lT - ln0) / 0.35) ** 2)
            X = np.column_stack([np.ones_like(lT), lT - ln0])
            bb = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (w * lr_all))
            rows.append(dict(accounting=acc, budget_C=cb, Nstar_T=float(np.exp(ln0)), eta_window=eta_win,
                             eta_local_tableA9=float(bb[1]), r_at_Nstar=float(np.exp(bb[0])),
                             n_window=len(ids), lnr_sd_window=float(np.std(yr - np.polyval(np.polyfit(xt, yr, 1), xt)))))
    return pd.DataFrame(rows)


# ============================================================================ diagnostics
def window_fit_quality(iso2, out):
    """Is the loss a smoother function of N in one count than in another? Pooled residual s.d. of the budget-level
    quadratics in x = ln N_acc, (i) on the reference (T) windows -- identical runs for every count -- and (ii) on each
    count's own primary windows."""
    rows = []
    L = iso2["L"].values
    for acc, key in (("ra1", "T_ra1"), ("6T", "T"), ("T4", "NF_T4"), ("A", "NF_A"), ("X", "NF_X"), ("6P", "P")):
        x = x_of(iso2, acc)
        for lab, wr in (("reference T windows", out["T_ra1"]["window_runs"]), ("own windows", out[key]["window_runs"])):
            rss, dof, n = 0.0, 0, 0
            for b, ids in wr.items():
                ids = np.array(ids)
                cf = np.polyfit(x[ids], L[ids], 2)
                rss += float(np.sum((L[ids] - np.polyval(cf, x[ids])) ** 2))
                dof += len(ids) - 3
                n += len(ids)
            rows.append(dict(accounting=acc, windows=lab, n_runs=n, dof=dof, rss=rss, resid_sd=float(np.sqrt(rss / dof))))
    return pd.DataFrame(rows)


def design_ranges(iso2):
    """Table 1 row quantities for the 137 profile runs in the T convention (ra1/Table 1: N digitized, D = C_dig/(6N))
    and in N_F (T4): N range, M = D/N range, off-path spread = residual s.d. of ln M on ln(6ND) (OLS)."""
    out = []
    D = iso2["D"].values                              # Epoch's D = C_dig/(6 N_dig) = D_true under the T4 budgets
    for lab, N, Dv in (("T (digitized N; Table 1)", iso2["N"].values, D),
                       ("N_F (T4), D from the digitization", iso2["F_T4"].values / 6.0, D),
                       ("N_F (T4), D = C_b/F_T4 (runs on the nominal budget)", iso2["F_T4"].values / 6.0,
                        iso2["C_b"].values / iso2["F_T4"].values)):
        M = Dv / N
        y, xx = np.log(M), np.log(6 * N * Dv)
        X = np.column_stack([np.ones_like(xx), xx])
        e = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        high = np.argsort(iso2["L"].values)[-5:]          # the five highest-loss runs (Besiroglu et al. drop them)
        keep = np.setdiff1d(np.arange(len(M)), high)
        e2 = y[keep] - X[keep] @ np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
        out.append(dict(convention=lab, n=len(M), N_min=float(N.min()), N_max=float(N.max()), M_min=float(M.min()),
                        M_max=float(M.max()), spread=float(np.std(e, ddof=2)),
                        M_min_excl5=float(M[keep].min()), spread_excl5=float(np.std(e2, ddof=2))))
    return pd.DataFrame(out)


def parametric(df):
    """ra1's same-run parametric sigma* (isoflop.run, parametric block, unchanged calls): Chinchilla form and kappa
    family by Huber loss; Feng-He-Hu wild bootstrap (B = ra1's B_PAR, ra1's seed for Chinchilla). T reproduces ra1;
    N_F (T4) uses N = F_T4/6 and D = C_b/F_T4."""
    import isoflop as iso_
    import parametric as pm
    rows = []
    iso = df[df.iso].reset_index(drop=True)
    for lab, x in (("T (ra1 reproduction)", np.log(iso["N"].values)), ("N_F (T4)", np.log(iso["F_T4"].values / 6.0))):
        d = design_df(iso.assign(iso=True), x)
        N = np.exp(d["x"].values)
        Dd = d["C_b"].values / (6 * N)
        L = d["L"].values
        th, fth = pm.fit_chin(N, Dd, L, starts=[rc.sl.BESIROGLU.theta])
        p, fp = pm.fit_kappa(N, Dd, L, th_chin=th)
        yc = pm.pred_chin(th, N, Dd)
        rf, _, _ = pm.fhh_residuals(np.log(L) - yc, pm.jac_log_chin(th, N, Dd))
        B = 40 if cm.QUICK else iso_.B_PAR
        pr = [r for r in rc.pmap(iso_._par_one, list(range(B)), dict(N=N, D=Dd, yhat_c=yc, rf_c=rf, th=th, p=p,
                                                                     seed0=rc.seed_of("par|Chinchilla")),
                                 procs=cm.N_PROC, chunksize=8) if "_error" not in r]
        pr = pd.DataFrame(pr)
        rows.append(dict(convention=lab, n=len(L), sigma_chin=pm.sigma_star_chin(th), se_sigma_chin=pr.sigma_chin.std(ddof=1),
                         sigma_kappa=pm.sigma_star_kappa(p), se_sigma_kappa=pr.sigma_kappa.std(ddof=1), kappa=p[5],
                         a_chin=th[4] / (th[3] + th[4]), qlr_kappa1=2 * len(L) * np.log(fth / fp), B=len(pr)))
    return rows


def reference_fit(df):
    """The paper's reference technology (Chinchilla form and kappa family, Huber, the 240 runs of Besiroglu et al.:
    all digitized runs but the five highest-loss ones; m2's Table 3 row) re-fitted in N_F (T4). Point estimates.
    T: N = digitized N, D = Epoch's C_dig/(6N) (reproduces m2: kappa = 1 0.737, kappa free 0.701). N_F: N = F_T4/6 of
    the matched Table A9 model, the same D (= D_true under the budgets set in count T4, coords_test)."""
    import parametric as pm
    d = df.sort_values("L").iloc[:-5]
    D = d["D"].values
    L = d["L"].values
    rows = []
    for lab, N in (("T (m2 reference reproduction)", d["N"].values), ("N_F (T4)", d["F_T4"].values / 6.0)):
        th, _ = pm.fit_chin(N, D, L, starts=[rc.sl.BESIROGLU.theta])
        p, _ = pm.fit_kappa(N, D, L, th_chin=th)
        out = dict(convention=lab, n=len(L), sigma_chin=pm.sigma_star_chin(th), a_chin=th[4] / (th[3] + th[4]),
                   alpha=th[3], beta=th[4], sigma_kappa=pm.sigma_star_kappa(p), kappa=p[5], a_kappa=p[4] / (p[3] + p[4]))
        for lab2, (lnA, lnB, al, be) in (("chin", (th[0], th[1], th[3], th[4])), ("kappa", (p[0], p[1], p[3], p[4]))):
            G = (al * np.exp(lnA) / (be * np.exp(lnB))) ** (1.0 / (al + be))
            for C in (1e21, 1e23):
                out[f"Mstar_{C:.0e}_{lab2}"] = float(G ** -2 * (C / 6.0) ** ((al - be) / (al + be)))
        rows.append(out)
    return rows
