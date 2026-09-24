"""m9_stages.py -- the analysis stages of run.py (plan Section 3, Q1-Q6), each skipping gracefully when its data are
not (yet) available. Every bootstrap is a wild cluster bootstrap by trunk (architecture) with Webb six-point weights,
the same weight for an architecture in both corpora within a draw (plan Section 2, Inference); the secondary scheme
draws Gaussian errors with the within-trunk covariance estimated from the seed replicates (Q4) when they exist.
Intervals are basic bootstrap intervals, est - (q(draws) - pop), pop = statistic on the noise-free population.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import m9_common as mc
import m9_est as est

C_LEVELS = np.exp(np.linspace(np.log(3e14), np.log(6e16), 12))
C_REPORT = (1e15, 1e16, 1e17)
BW_MULTS = (0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5)
B_BOOT = 49 if mc.QUICK else 999
MIN_WIDTHS, MIN_POINTS = 4, 16
SIG_GRID = np.round(np.arange(0.05, 0.99 + 1e-9, 0.02), 4)


def paths_of(conv):
    return ("P", "P6") if conv == "P" else ("T",)


def enough(df):
    return df["d"].nunique() >= MIN_WIDTHS and len(df) >= MIN_POINTS


def valid_output(df, col):
    return df[np.isfinite(df[col].values)]


# ============================================================================ noise draws (wild / seed covariance)
class Shock:
    """One bootstrap draw's error realization. 'wild': one Webb weight per trunk label, shared by the two corpora and
    by every estimator's residuals in the draw (e* = v_g e_hat). 'seedcov' (secondary, needs Q4): Gaussian errors
    s (sqrt(rho) u_g + sqrt(1 - rho) z_i) with (s, rho) of each corpus estimated from its seed replicates; trunks of
    different corpora are independent; the same realization is used for every estimator in the draw."""

    def __init__(self, seed, scheme, universe, noise=None):
        self.rng = np.random.default_rng(seed)
        self.scheme, self.noise = scheme, noise
        if scheme in ("wild", "cr2"):
            self.w = dict(zip(universe, mc.webb(self.rng, len(universe))))
        self.u, self.z = {}, {}

    def errors(self, resid, clus, corpus):
        clus = np.asarray(clus)
        if self.scheme in ("wild", "cr2"):
            return np.array([self.w[c] for c in clus]) * resid
        s, rho = self.noise[corpus]
        if corpus not in self.z:
            self.z[corpus] = self.rng.standard_normal(len(clus))
        for c in np.unique(clus):
            if (corpus, c) not in self.u:
                self.u[(corpus, c)] = self.rng.standard_normal()
        u = np.array([self.u[(corpus, c)] for c in clus])
        return s * (np.sqrt(rho) * u + np.sqrt(1 - rho) * self.z[corpus])


# ============================================================================ model-free helpers
def mf_prepare(N, D, L, bw=None):
    x, z, f = np.log(N), np.log(D), np.log(L)
    table = None
    if bw is None:
        (rmse, hx, hz, mx_, mz_), table = est.cv_bandwidth(x, z, f, BW_MULTS)
        bw = dict(hx=hx, hz=hz, mx=mx_, mz=mz_, loocv_rmse=rmse)
    sx = np.sort(np.unique(x))
    dmin = np.array([D[x == v].min() for v in sx])
    dmax = np.array([D[x == v].max() for v in sx])
    grid = np.array([(xx, zz) for xx, a, b in zip(sx, dmin, dmax) for zz in np.linspace(np.log(a), np.log(b), 12)])
    return dict(x=x, z=z, sx=sx, dmin=dmin, dmax=dmax, grid=grid, **bw), table


def mf_stats(prep, f, conv, levels_ok=None):
    """Model-free sigma* along the local compute-optimal path(s) of convention conv (Hessian and identity forms;
    first-derivative slope form for w = 1 paths), per compute level and averaged over the valid levels."""
    out = {}
    for pc in paths_of(conv):
        hull = est.make_hull(pc, prep["sx"], prep["dmin"], prep["dmax"])
        pth = est.local_path(prep["x"], prep["z"], f, prep["hx"], prep["hz"], pc, C_LEVELS, hull)
        ok = np.isfinite(pth[:, 1]) if levels_ok is None else np.asarray(levels_ok[pc], bool)
        vals = np.where(ok, pth[:, 1], np.nan)
        out[f"sig_mf_{pc}"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
        out[f"sig_mfid_{pc}"] = float(np.nanmean(np.where(ok, pth[:, 2], np.nan))) if np.isfinite(vals).any() else np.nan
        out[f"sig_mfslope_{pc}"] = est.sigma_slope(prep["x"], prep["z"], f, prep["hx"], prep["hz"], pc, pth, C_LEVELS,
                                                   prep["grid"])
        for i in range(len(C_LEVELS)):
            out[f"lvl_sig_{pc}_{i}"] = float(pth[i, 1]) if ok[i] else np.nan
            out[f"lvl_lnM_{pc}_{i}"] = float(pth[i, 3]) if ok[i] else np.nan
        out[f"_path_{pc}"] = pth
    return out


# ============================================================================ Q1 (curvature)
def q1_point(N, D, L, conv):
    th, _ = est.fit_chin(N, D, L, "huber", grid="fast")
    p, _ = est.fit_kappa(N, D, L, "huber", th_chin=th)
    thn, _ = est.fit_chin(N, D, L, "nls", grid="fast")
    pn, _ = est.fit_kappa(N, D, L, "nls", th_chin=thn)
    return dict(th=th, p=p, thn=thn, pn=pn)


def q1_param_stats(fits, conv, levels_ok_P=None):
    th, p, thn, pn = fits["th"], fits["p"], fits["thn"], fits["pn"]
    out = dict(sig_chin=est.sigma_star_chin(th), sig_kappa=est.sigma_star_kappa(p),
               sig_chin_nls=est.sigma_star_chin(thn), sig_kappa_nls=est.sigma_star_kappa(pn),
               kappa=float(p[5]), kappa_nls=float(pn[5]), alpha=float(th[3]), beta=float(th[4]),
               a1=float(p[3]), b1=float(p[4]), a_chin=float(th[4] / (th[3] + th[4])), a_kappa=float(p[4] / (p[3] + p[4])))
    if conv == "P":
        pp = est.param_path("P", lambda N_, D_: est.w_kappa(p, N_, D_), p[3], p[4], C_LEVELS, 12.9, 17.7)
        ok = np.isfinite(pp[:, 1]) if levels_ok_P is None else np.asarray(levels_ok_P, bool)
        out["sig_kappa_Ppath"] = float(np.nanmean(np.where(ok, pp[:, 1], np.nan)))
        pc = est.param_path("P", lambda N_, D_: est.w_chin(th, N_, D_), th[3], th[4], C_LEVELS, 12.9, 17.7)
        out["sig_chin_Ppath"] = float(np.nanmean(np.where(ok, pc[:, 1], np.nan)))
    return out


def q1_draw(item, conv=None, corp=None, universe=None, scheme="wild", noise=None, nls=True, reselect=False):
    b, seed = item
    sh = Shock(seed, scheme, universe, noise)
    out = {}
    for r, c in corp.items():
        N, D, cl = c["N"], c["D"], c["clus"]
        th, _ = est.fit_chin(N, D, np.exp(c["yh_chin"] + sh.errors(c["res_chin"], cl, r)), "huber", starts=[c["fits"]["th"]])
        p, _ = est.fit_kappa(N, D, np.exp(c["yh_kappa"] + sh.errors(c["res_kappa"], cl, r)), "huber", init=c["fits"]["p"])
        fits = dict(th=th, p=p, thn=c["fits"]["thn"], pn=c["fits"]["pn"])
        if nls:
            fits["thn"], _ = est.fit_chin(N, D, np.exp(c["yh_chin_nls"] + sh.errors(c["res_chin_nls"], cl, r)), "nls",
                                          starts=[c["fits"]["thn"]])
            fits["pn"], _ = est.fit_kappa(N, D, np.exp(c["yh_kappa_nls"] + sh.errors(c["res_kappa_nls"], cl, r)), "nls",
                                          init=c["fits"]["pn"])
        ps = q1_param_stats(fits, conv, c["levels_ok"].get("P"))
        fb = c["pilot"] + sh.errors(c["res_pilot"], cl, r)
        prep = mf_prepare(N, D, np.exp(fb))[0] if reselect else c["prep"]
        ms = mf_stats(prep, fb, conv, c["levels_ok"])
        for k, v in {**ps, **ms}.items():
            if not k.startswith("_"):
                out[f"{k}|{r}"] = v
    return out


def cr2_adjust(c, clus):
    """CR2 (Bell-McCaffrey) cluster residuals for every population of the Q1 bootstrap (pre-declared remedy D8)."""
    N, D = c["N"], c["D"]
    out = {}
    out["res_chin"] = est.cr2_residuals(c["res_chin"], est.hat_from_jac(est.jac_chin(c["fits"]["th"], N, D)), clus)
    out["res_chin_nls"] = est.cr2_residuals(c["res_chin_nls"], est.hat_from_jac(est.jac_chin(c["fits"]["thn"], N, D)), clus)
    out["res_kappa"] = est.cr2_residuals(c["res_kappa"], est.hat_from_jac(est.jac_kappa(c["fits"]["p"], N, D)), clus)
    out["res_kappa_nls"] = est.cr2_residuals(c["res_kappa_nls"], est.hat_from_jac(est.jac_kappa(c["fits"]["pn"], N, D)), clus)
    pr = c["prep"]
    out["res_pilot"] = est.cr2_residuals(c["res_pilot"], est.smoother_matrix(pr["x"], pr["z"], pr["hx"], pr["hz"]), clus)
    return out


def q1_stage(df, conv, sample, scheme="wild", noise=None, B=B_BOOT, log=mc.log):
    """Q1 for all corpora with enough data in (conv, sample). Returns (rows, level_rows, corp payload)."""
    corp = {}
    for r in mc.CORPORA:
        s = mc.main_sample(df, r, sample)
        s = s.assign(y=mc.own_output(s))
        s = s[np.isfinite(s.y)]
        if not enough(s):
            continue
        N, D, L = s[f"N_{conv}"].values, s["D"].values, s["y"].values
        fits = q1_point(N, D, L, conv)
        prep, cvtab = mf_prepare(N, D, L)
        f = np.log(L)
        pilot = est.pilot_fit(prep["x"], prep["z"], f, prep["hx"], prep["hz"])
        ms = mf_stats(prep, f, conv)
        levels_ok = {pc: np.isfinite(ms[f"_path_{pc}"][:, 1]) for pc in paths_of(conv)}
        ms_pop = mf_stats(prep, pilot, conv, levels_ok)
        ps = q1_param_stats(fits, conv, levels_ok.get("P"))
        yh = {k: (est.pred_chin(fits[k], N, D) if k.startswith("th") else est.pred_kappa(fits[k], N, D))
              for k in ("th", "p", "thn", "pn")}
        corp[r] = dict(N=N, D=D, L=L, clus=mc.clusters_of(s), fits=fits, prep=prep, pilot=pilot, levels_ok=levels_ok,
                       yh_chin=yh["th"], yh_kappa=yh["p"], yh_chin_nls=yh["thn"], yh_kappa_nls=yh["pn"],
                       res_chin=f - yh["th"], res_kappa=f - yh["p"], res_chin_nls=f - yh["thn"],
                       res_kappa_nls=f - yh["pn"], res_pilot=f - pilot, point={**ps, **ms}, pop={**ps, **ms_pop},
                       sample=s, cvtab=cvtab)
    if not corp:
        return [], [], {}
    if scheme == "cr2":
        for r, c in corp.items():
            c.update(cr2_adjust(c, c["clus"]))
    universe = sorted(set(np.concatenate([c["clus"] for c in corp.values()])))
    payload = dict(conv=conv, universe=universe, scheme=scheme, noise=noise, nls=(sample == "main"), reselect=(scheme == "cr2"),
                   corp={r: {k: v for k, v in c.items() if k not in ("sample", "cvtab", "point", "pop")}
                         for r, c in corp.items()})
    items = [(b, mc.seed_of(f"q1|{scheme}|{conv}|{sample}|{b}")) for b in range(B)]
    draws = mc.pmap(q1_draw, items, payload)
    errs = [d for d in draws if "_error" in d]
    if errs:
        log(f"    Q1 {conv}/{sample}/{scheme}: {len(errs)} failed draws; first: {errs[0]['_error']}")
    D_ = pd.DataFrame([d for d in draws if "_error" not in d])
    rows, lvl_rows = [], []
    for r, c in corp.items():
        for k, v in c["point"].items():
            if k.startswith("_") or k.startswith("lvl_"):
                continue
            dr = D_[f"{k}|{r}"].values if f"{k}|{r}" in D_ else np.array([])
            pop = c["pop"].get(k, v)
            lo, hi = mc.basic_ci(v, dr, pop)
            rows.append(dict(corpus=r, conv=conv, sample=sample, scheme=scheme, statistic=k, estimate=v, pop=pop,
                             se=mc.sd_(dr), ci_lo=lo, ci_hi=hi, n_draws=int(np.isfinite(dr).sum()), n_obs=len(c["N"]),
                             n_widths=int(len(np.unique(c["clus"]))), bw_hx=c["prep"]["hx"], bw_hz=c["prep"]["hz"],
                             bw_mx=c["prep"]["mx"], bw_mz=c["prep"]["mz"]))
        for pc in paths_of(conv):
            pth = c["point"][f"_path_{pc}"]
            for i, C in enumerate(C_LEVELS):
                if not np.isfinite(pth[i, 1]):
                    continue
                k = f"lvl_sig_{pc}_{i}"
                dr = D_[f"{k}|{r}"].values if f"{k}|{r}" in D_ else np.array([])
                lo, hi = mc.basic_ci(pth[i, 1], dr, c["pop"].get(k, pth[i, 1]))
                lvl_rows.append(dict(corpus=r, conv=conv, path=pc, sample=sample, scheme=scheme, C=C,
                                     sigma=pth[i, 1], sigma_identity=pth[i, 2], ci_lo=lo, ci_hi=hi, se=mc.sd_(dr),
                                     lnN_star=pth[i, 0], M_star=float(np.exp(pth[i, 3])), n_eff=pth[i, 4]))
    # cross-corpus equality (joint draws)
    if len(corp) == 2:
        for k in corp["edu"]["point"]:
            if k.startswith("_") or k.startswith("lvl_"):
                continue
            ke, kw = f"{k}|edu", f"{k}|web"
            if ke not in D_ or kw not in D_:
                continue
            e_ = corp["edu"]["point"][k] - corp["web"]["point"][k]
            pop = corp["edu"]["pop"].get(k) - corp["web"]["pop"].get(k)
            dr = D_[ke].values - D_[kw].values
            lo, hi = mc.basic_ci(e_, dr, pop)
            rows.append(dict(corpus="edu-web", conv=conv, sample=sample, scheme=scheme, statistic=k, estimate=e_,
                             pop=pop, se=mc.sd_(dr), ci_lo=lo, ci_hi=hi, n_draws=int(np.isfinite(dr).sum()),
                             p_equal=mc.boot_p(dr, e_, pop, 0.0)))
    return rows, lvl_rows, corp


# ============================================================================ Q2 (neutrality, corpus pair)
def q2_draw(item, N=None, D=None, g=None, clus=None, universe=None, yh_ce=None, res_ce=None, th_ce=None,
            yh_sep=None, res_sep=None, th_sep=None, scheme="wild", noise=None, yh0=None, res0=None):
    import m2_est as me
    b, seed = item
    sh = Shock(seed, scheme, universe, noise)
    corp = np.where(g == 1, "edu", "web")
    e = np.zeros(len(N))
    e2 = np.zeros(len(N))
    for r in ("web", "edu"):
        m = corp == r
        e[m] = sh.errors(res_ce[m], clus[m], r)
        # same realization (same weights / Gaussian shocks) applied to the separate fits' own residuals
        e2[m] = sh.errors(res_sep[m], clus[m], r)
    th, _, _ = me.fit_panel_ls("CE", np.log(N), np.log(D), yh_ce + e, g, 2, "huber", starts=[th_ce], tight=False)
    out = q2_stats(th)
    if yh0 is not None:   # restricted wild cluster bootstrap: population with chi = 0 imposed (same draw)
        e0 = np.zeros(len(N))
        for r in ("web", "edu"):
            m = corp == r
            e0[m] = sh.errors(res0[m], clus[m], r)
        t0, _, _ = me.fit_panel_ls("CE", np.log(N), np.log(D), yh0 + e0, g, 2, "huber", starts=[th_ce], tight=False)
        out["chi_null"] = est.ce_tilt(t0)
    for r in (0, 1):
        s = g == r
        t, _ = est.fit_chin(N[s], D[s], np.exp(yh_sep[s] + e2[s]), "huber", starts=[th_sep[r]])
        out[f"alpha_sep_{r}"], out[f"beta_sep_{r}"] = t[3], t[4]
    return out


def q2_stats(th, conv=None, with_path=False):
    chi = est.ce_tilt(th)
    al, be = th[0], th[1]
    out = dict(chi=chi, alpha=al, beta=be, sigma_star=2 / (2 + al + be), a=be / (al + be),
               Mstar_ratio_w1=float(np.exp(-2 * chi / (al + be))), what_factor=float(np.exp(-chi)),
               dlnE=float(th[7] - th[6]), dlnA=float(th[3] - th[2]), dlnB=float(th[5] - th[4]))
    if with_path and conv == "P":
        r = est.mstar_ratio(est.ce_corpus_theta(th, 1), est.ce_corpus_theta(th, 0), "P", C_REPORT)
        for C, v in zip(C_REPORT, r):
            out[f"Mstar_ratio_P_{C:.0e}"] = v
    return out


def q2_stage(df, conv, sample, valset, scheme="wild", noise=None, B=B_BOOT, log=mc.log, rule_only=True):
    """Corpus-pair CE model on validation set `valset` (bpb) for main-grid endpoints of both corpora."""
    col = f"bpb_{valset}"
    parts = []
    for gi, r in ((0, "web"), (1, "edu")):
        s = mc.main_sample(df, r, sample)
        s = valid_output(s, col)
        if not enough(s):
            return None
        parts.append(s.assign(g=gi))
    s = pd.concat(parts, ignore_index=True)
    N, D, L, g = s[f"N_{conv}"].values, s["D"].values, s[col].values, s["g"].values.astype(int)
    clus = mc.clusters_of(s)
    starts, th_sep = est.ce_starts(N, D, L, g)
    th, obj = est.fit_ce(N, D, L, g, "huber", starts=starts + [np.r_[0.3, 0.3, starts[0][2:]]])
    y = np.log(L)
    yh = est.pred_ce(th, N, D, g)
    yh_sep = np.where(g == 1, est.pred_chin(th_sep[1], N, D), est.pred_chin(th_sep[0], N, D))
    universe = sorted(set(clus))
    res_ce, res_sep = y - yh, y - yh_sep
    th0, yh0 = est.fit_ce_null(N, D, L, g, th)
    res0 = y - yh0
    if scheme == "cr2":
        import m2_est as me
        res_ce = est.cr2_residuals(res_ce, est.hat_from_jac(est.jac_ce(th, N, D, g)), clus)
        H0 = est.hat_from_jac(est.num_jac(lambda q: me.panel_pred("hicksE", q, np.log(N), np.log(D), g, 2), th0))
        res0 = est.cr2_residuals(res0, H0, clus)
        rs = res_sep.copy()
        for r in (0, 1):
            m = g == r
            rs[m] = est.cr2_residuals(res_sep[m], est.hat_from_jac(est.jac_chin(th_sep[r], N[m], D[m])), clus[m])
        res_sep = rs
    payload = dict(N=N, D=D, g=g, clus=clus, universe=universe, yh_ce=yh, res_ce=res_ce, th_ce=th, yh_sep=yh_sep,
                   res_sep=res_sep, th_sep=th_sep, scheme=scheme, noise=noise, yh0=yh0, res0=res0)
    items = [(b, mc.seed_of(f"q2|{scheme}|{conv}|{sample}|{valset}|{b}")) for b in range(B)]
    draws = mc.pmap(q2_draw, items, payload)
    errs = [d for d in draws if "_error" in d]
    if errs:
        log(f"    Q2 {conv}/{sample}/{valset}: {len(errs)} failed draws; first: {errs[0]['_error']}")
    D_ = pd.DataFrame([d for d in draws if "_error" not in d])
    pt = q2_stats(th, conv, with_path=True)
    rows = []
    for k, v in pt.items():
        dr = D_[k].values if k in D_ else np.array([])
        lo, hi = mc.basic_ci(v, dr, v)
        pw_ = np.nan
        if k == "chi" and "chi_null" in D_:
            dn = D_["chi_null"].values
            dn = dn[np.isfinite(dn)]
            pw_ = float((1 + np.sum(np.abs(dn) >= abs(v))) / (len(dn) + 1))
        rows.append(dict(valset=valset, conv=conv, sample=sample, scheme=scheme, statistic=k, estimate=v, se=mc.sd_(dr),
                         ci_lo=lo, ci_hi=hi, p_zero=mc.boot_p(dr, v, v, 0.0) if k in ("chi", "dlnE") else np.nan,
                         p_wcr=pw_, n_draws=int(np.isfinite(dr).sum()), n_obs=len(N)))
    # (a) Wald test of equal exponents from separate fits (bootstrap covariance)
    dvec = np.array([th_sep[1][3] - th_sep[0][3], th_sep[1][4] - th_sep[0][4]])
    if {"alpha_sep_0", "alpha_sep_1"} <= set(D_.columns):
        dd = np.column_stack([D_["alpha_sep_1"] - D_["alpha_sep_0"], D_["beta_sep_1"] - D_["beta_sep_0"]])
        dd = dd[np.all(np.isfinite(dd), axis=1)]
        V = np.cov(dd.T)
        W = float(dvec @ np.linalg.solve(V, dvec))
        from scipy.stats import chi2
        rows.append(dict(valset=valset, conv=conv, sample=sample, scheme=scheme, statistic="wald_equal_exponents",
                         estimate=W, se=np.nan, ci_lo=np.nan, ci_hi=np.nan, p_zero=float(chi2.sf(W, 2)),
                         n_draws=len(dd), n_obs=len(N), d_alpha=dvec[0], d_beta=dvec[1]))
    return dict(rows=rows, th=th, th_sep=th_sep, sample=s)


def q2_decision(tilt_rows):
    """Plan decision rule per (conv, sample, scheme): 'factor-biased' if the 95% CI of chi excludes 0 on BOTH
    validation sets (edu, web) with the same sign; 'neutral' if it includes 0 on both with half-width < 0.10;
    otherwise 'inconclusive'. WikiText, where available, enters the quality/distribution-match classification."""
    out = []
    t = pd.DataFrame(tilt_rows)
    if t.empty:
        return pd.DataFrame()
    t = t[t.statistic == "chi"]
    for (conv, sample, scheme), g in t.groupby(["conv", "sample", "scheme"]):
        g = g.set_index("valset")
        if not {"edu", "web"} <= set(g.index):
            out.append(dict(conv=conv, sample=sample, scheme=scheme, decision="pending (needs both validation sets)"))
            continue
        if scheme == "cr2":   # pre-declared remedy D8: restricted wild bootstrap (CR2) test of chi = 0
            ex = {v: bool(g.loc[v, "p_wcr"] < 0.05) for v in ("edu", "web")}
        else:                 # the plan's rule, as written
            ex = {v: (g.loc[v, "ci_lo"] > 0) or (g.loc[v, "ci_hi"] < 0) for v in ("edu", "web")}
        sg = {v: np.sign(g.loc[v, "estimate"]) for v in ("edu", "web")}
        hw = {v: (g.loc[v, "ci_hi"] - g.loc[v, "ci_lo"]) / 2 for v in ("edu", "web")}
        if ex["edu"] and ex["web"] and sg["edu"] == sg["web"]:
            dec = "factor-biased (" + ("data-augmenting for edu" if sg["edu"] > 0 else "parameter-augmenting for edu") + ")"
        elif (not ex["edu"]) and (not ex["web"]) and hw["edu"] < 0.10 and hw["web"] < 0.10:
            dec = "neutral"
        else:
            dec = "inconclusive"
        wiki = g.loc["wiki", "estimate"] if "wiki" in g.index else np.nan
        out.append(dict(conv=conv, sample=sample, scheme=scheme, decision=dec, chi_edu=g.loc["edu", "estimate"],
                        chi_web=g.loc["web", "estimate"], chi_wiki=wiki, halfwidth_edu=hw["edu"], halfwidth_web=hw["web"]))
    return pd.DataFrame(out)


def matched_cells(df):
    """Between-corpus log-loss differences (edu - web) at cells trained in both corpora with the same (d, L, lr,
    seed, D), on each validation set; sign consistency across validation sets (quality vs distribution match)."""
    key = ["tag", "d", "L", "lr", "seed", "D_target"]
    e = df[df.regime == "edu"].set_index(key)
    w = df[df.regime == "web"].set_index(key)
    common = e.index.intersection(w.index)
    rows = []
    for k in common:
        row = dict(zip(key, k))
        for v in mc.VALSETS:
            a, b = e.loc[k, f"bpb_{v}"], w.loc[k, f"bpb_{v}"]
            row[f"dlnL_{v}"] = float(np.log(a) - np.log(b)) if np.isfinite(a) and np.isfinite(b) else np.nan
            row[f"bpb_edu_{v}"], row[f"bpb_web_{v}"] = a, b
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================ Q3 (extrapolation of the wedge)
M_BINS = [100, 316, 1000, 1e5]
M_BIN_LAB = ["100-316", "316-1,000", ">1,000"]


def q3_core(N, D, M, f, restr, prep, fits_warm=None, grid=None):
    """All Q3 quantities for one data set: restricted and full-support Chinchilla and kappa fits, local elasticities
    at every endpoint, and the statistics. f = ln L. fits_warm: warm starts (bootstrap)."""
    L = np.exp(f)
    if fits_warm is None:
        th_r, _ = est.fit_chin(N[restr], D[restr], L[restr], "huber", grid="fast")
        p_r, _ = est.fit_kappa(N[restr], D[restr], L[restr], "huber", th_chin=th_r)
        th_f, _ = est.fit_chin(N, D, L, "huber", grid="fast")
        p_f, _ = est.fit_kappa(N, D, L, "huber", th_chin=th_f)
    else:
        th_r, _ = est.fit_chin(N[restr], D[restr], L[restr], "huber", starts=[fits_warm["th_r"]])
        p_r, _ = est.fit_kappa(N[restr], D[restr], L[restr], "huber", init=fits_warm["p_r"])
        th_f, _ = est.fit_chin(N, D, L, "huber", starts=[fits_warm["th_f"]])
        p_f, _ = est.fit_kappa(N, D, L, "huber", init=fits_warm["p_f"])
    c = est.local_coefs(prep["x"], prep["z"], f, list(zip(prep["x"], prep["z"])), prep["hx"], prep["hz"])
    good = (c[:, 1] < 0) & (c[:, 2] < 0)
    lw = np.where(good, est.lnw_local(c), np.nan)
    out = dict(fits=dict(th_r=th_r, p_r=p_r, th_f=th_f, p_f=p_f), lnw_local=lw, eN_local=-c[:, 1], eD_local=-c[:, 2],
               neff=c[:, 6], pilot=c[:, 0])
    st = {}
    sel = M > 100
    lm = np.log(M)
    comps = dict(chin_r=np.log(est.w_chin(th_r, N, D)), kappa_r=np.log(est.w_kappa(p_r, N, D)),
                 chin_f=np.log(est.w_chin(th_f, N, D)), kappa_f=np.log(est.w_kappa(p_f, N, D)))
    out["lnw"] = comps
    for name, lnwp in comps.items():
        for ref, lref in (("local", lw), ("chin_f", comps["chin_f"])):
            if name == ref:
                continue
            dl = lnwp - lref
            ok = sel & np.isfinite(dl)
            if ok.sum() >= 3:
                X = np.column_stack([np.ones(ok.sum()), lm[ok] - np.log(100.0)])
                b = np.linalg.lstsq(X, dl[ok], rcond=None)[0]
                st[f"slope|{name}|{ref}"] = float(b[1])
                st[f"level100|{name}|{ref}"] = float(b[0])
                st[f"mean|{name}|{ref}"] = float(np.mean(dl[ok]))
                for j, lab in enumerate(M_BIN_LAB):
                    s2 = ok & (M > M_BINS[j]) & (M <= M_BINS[j + 1])
                    st[f"bin{j}|{name}|{ref}"] = float(np.mean(dl[s2])) if s2.any() else np.nan
            st[f"n_eval|{name}|{ref}"] = int(ok.sum())
    # slope of the full-support fit against local (plan comparison (a) vs (b))
    out["stats"] = st
    return out


def q3_draw(item, N=None, D=None, M=None, clus=None, universe=None, restr=None, prep=None, pilot=None, res=None,
            fits=None, scheme="wild", noise=None, corpus=None):
    b, seed = item
    sh = Shock(seed, scheme, universe, noise)
    fb = pilot + sh.errors(res, clus, corpus)
    if scheme == "cr2":   # bandwidth re-selected in every draw (pre-declared remedy D8)
        prep = mf_prepare(N, D, np.exp(fb))[0]
    r = q3_core(N, D, M, fb, restr, prep, fits_warm=fits)
    out = dict(r["stats"])
    for i in np.where(M > 100)[0]:
        out[f"pt_{i}"] = float(r["lnw"]["chin_r"][i] - r["lnw_local"][i])
    return out


def q3_stage(df, r, conv, sample, scheme="wild", noise=None, B=B_BOOT, log=mc.log):
    s = mc.main_sample(df, r, sample, with_hiM=True)
    s = s.assign(y=mc.own_output(s))
    s = s[np.isfinite(s.y)].reset_index(drop=True)
    main = s[s.tag == "main"]
    if not enough(main):
        return None
    N, D, M, f = s[f"N_{conv}"].values, s["D"].values, s[f"M_{conv}"].values, np.log(s.y.values)
    restr = (s.tag.values == "main") & (M <= 100)
    if restr.sum() < 12 or (M > 100).sum() < 3:
        return None
    prep, _ = mf_prepare(N, D, np.exp(f))
    core = q3_core(N, D, M, f, restr, prep)
    pilot = core["pilot"]
    pop = q3_core(N, D, M, pilot, restr, prep, fits_warm=core["fits"])
    clus = mc.clusters_of(s)
    res = f - pilot
    if scheme == "cr2":
        res = est.cr2_residuals(res, est.smoother_matrix(prep["x"], prep["z"], prep["hx"], prep["hz"]), clus)
    payload = dict(N=N, D=D, M=M, clus=clus, universe=sorted(set(clus)), restr=restr, prep=prep, pilot=pilot,
                   res=res, fits=core["fits"], scheme=scheme, noise=noise, corpus=r)
    items = [(b, mc.seed_of(f"q3|{scheme}|{r}|{conv}|{sample}|{b}")) for b in range(B)]
    draws = mc.pmap(q3_draw, items, payload)
    errs = [d for d in draws if "_error" in d]
    if errs:
        log(f"    Q3 {r}/{conv}/{sample}: {len(errs)} failed draws; first: {errs[0]['_error']}")
    D_ = pd.DataFrame([d for d in draws if "_error" not in d])
    rows = []
    for k, v in core["stats"].items():
        dr = D_[k].values if k in D_ else np.array([])
        p0 = pop["stats"].get(k, np.nan)
        lo, hi = mc.basic_ci(v, dr, p0)
        rows.append(dict(corpus=r, conv=conv, sample=sample, scheme=scheme, statistic=k, estimate=v, pop=p0,
                         se=mc.sd_(dr), ci_lo=lo, ci_hi=hi, p_zero=mc.boot_p(dr, v, p0, 0.0) if k.startswith("slope") else np.nan,
                         n_draws=int(np.isfinite(dr).sum()), n_restricted=int(restr.sum()), n_all=len(N)))
    # per-endpoint table
    th_r, th_f, p_r = core["fits"]["th_r"], core["fits"]["th_f"], core["fits"]["p_r"]
    eN_r, eD_r = est.eps_chin(th_r, N, D)
    eN_f, eD_f = est.eps_chin(th_f, N, D)
    pts = []
    for i in range(len(N)):
        dr = D_[f"pt_{i}"].values if f"pt_{i}" in D_ else np.array([])
        v = core["lnw"]["chin_r"][i] - core["lnw_local"][i]
        p0 = pop["lnw"]["chin_r"][i] - pop["lnw_local"][i]
        lo, hi = mc.basic_ci(v, dr, p0) if M[i] > 100 else (np.nan, np.nan)
        pts.append(dict(corpus=r, conv=conv, sample=sample, tag=s.tag[i], d=s.d[i], L=s.L[i], D=D[i], N=N[i], M=M[i],
                        restricted=bool(restr[i]), bpb=s.y[i], eN_local=core["eN_local"][i], eD_local=core["eD_local"][i],
                        lnw_local=core["lnw_local"][i], n_eff=core["neff"][i], eN_chin_r=eN_r[i], eD_chin_r=eD_r[i],
                        lnw_chin_r=core["lnw"]["chin_r"][i], eN_chin_f=eN_f[i], eD_chin_f=eD_f[i],
                        lnw_chin_f=core["lnw"]["chin_f"][i], lnw_kappa_r=core["lnw"]["kappa_r"][i],
                        lnw_kappa_f=core["lnw"]["kappa_f"][i], delta=v, delta_lo=lo, delta_hi=hi))
    return dict(rows=rows, points=pts, fits=core["fits"], bw=dict(hx=prep["hx"], hz=prep["hz"], mx=prep["mx"],
                                                                     mz=prep["mz"]))


# ============================================================================ Q4 (noise)
def q4_stage(df, q1_resid=None):
    """Seed SD of log loss at each replicated cell (seeds, seedcorner + the seed-0 run of the same cell) and the
    within-trunk correlation across budgets; compared with the residual SD of the fitted technologies."""
    rows, cells = [], []
    rep = df[df.tag.isin(["seeds", "seedcorner"])]
    for r in mc.CORPORA:
        rr = rep[rep.regime == r]
        if rr.empty:
            continue
        base = df[(df.regime == r) & (df.tag == "main") & (df.seed == 0)]
        for (d, L, D_t), g in rr.groupby(["d", "L", "D_target"]):
            b = base[(base.d == d) & (base.L == L) & (base.D_target == D_t)]
            # seedcorner's seed-1 trunk at width 128 reproduces the `seeds` seed-1 trunk (same seed, data order and LR),
            # so its 200M endpoint is a duplicate of the same seed, not a new replicate: one record per seed
            gg = pd.concat([g, b]).drop_duplicates("seed", keep="first")
            for v in mc.VALSETS:
                y = np.log(gg[f"bpb_{v}"].values)
                y = y[np.isfinite(y)]
                if len(y) >= 2:
                    cells.append(dict(corpus=r, d=d, L=L, D_target=D_t, valset=v, n_seeds=len(y),
                                      sd_lnL=float(np.std(y, ddof=1)), mean_bpb=float(np.exp(np.mean(y)))))
    cdf = pd.DataFrame(cells)
    noise = {}
    if not cdf.empty:
        for (r, v), g in cdf.groupby(["corpus", "valset"]):
            dfree = (g.n_seeds - 1).sum()
            pooled = float(np.sqrt(np.sum((g.n_seeds - 1) * g.sd_lnL ** 2) / dfree))
            rho = seed_trunk_corr(df, r, v)
            rows.append(dict(corpus=r, valset=v, n_cells=len(g), df=int(dfree), pooled_sd_lnL=pooled,
                             median_cell_sd=float(g.sd_lnL.median()), max_cell_sd=float(g.sd_lnL.max()),
                             rho_within_trunk=rho))
            if (r == "edu" and v == "edu") or (r == "web" and v == "web"):
                noise[r] = (pooled, rho if np.isfinite(rho) else 0.5)
    if q1_resid is not None:
        for (r, conv, fit), res in q1_resid.items():
            rows.append(dict(corpus=r, valset="own", conv=conv, fit=fit, n_cells=len(res), residual_sd=float(np.std(res, ddof=1)),
                             residual_mad_sd=float(np.median(np.abs(res - np.median(res))) * 1.4826)))
    return pd.DataFrame(rows), cdf, noise


def seed_trunk_corr(df, r, v):
    """Within-trunk (equi)correlation of seed deviations across budgets, by variance decomposition: for each width and
    budget, deviations of ln L from the cell mean across seeds (scaled by sqrt(n/(n-1))); for each seed trunk the mean
    deviation over its K budgets has variance s^2 (rho + (1 - rho)/K), so
        rho = (mean_trunk m^2 - mean d^2 / K) / (mean d^2 (1 - 1/K)),
    pooled over trunks (K = number of budgets of the trunk), clipped to [0, 0.99]. Few seeds: noisy."""
    col = f"bpb_{v}"
    s = df[(df.regime == r) & ((df.tag.isin(["seeds", "seedcorner"])) | ((df.tag == "main") & (df.seed == 0)))]
    num, den = [], []
    for (d, L), g in s.groupby(["d", "L"]):
        g = g[np.isfinite(g[col])]
        piv = g.pivot_table(index="seed", columns="D_target", values=col, aggfunc="first")
        piv = piv.dropna(axis=1, thresh=2)
        if piv.shape[0] < 2 or piv.shape[1] < 2:
            continue
        lp = np.log(piv)
        nsd = lp.notna().sum(axis=0)
        dev = (lp - lp.mean(axis=0)) * np.sqrt(nsd / (nsd - 1))
        for _, row in dev.iterrows():
            vals = row.dropna().values
            K = len(vals)
            if K < 2:
                continue
            num.append((np.mean(vals) ** 2, np.mean(vals ** 2), K))
    if not num:
        return np.nan
    m2 = np.array([x[0] for x in num])
    d2 = np.array([x[1] for x in num])
    K = np.array([x[2] for x in num], float)
    sd2 = d2.mean()
    rho = (m2.mean() - np.mean(d2 / K)) / (sd2 * (1 - np.mean(1 / K)))
    return float(np.clip(rho, 0.0, 0.99))


# ============================================================================ Q5 (flexible inputs: learning rate)
def quad_argmin(lx, ly, lo=None, hi=None, pad=0.35):
    """Quadratic in ln LR (fit_lr.py's rule): argmin clipped to the tested range +- pad; returns (ln lr*, min, convex)."""
    lx, ly = np.asarray(lx, float), np.asarray(ly, float)
    lo = lx.min() - pad if lo is None else lo
    hi = lx.max() + pad if hi is None else hi
    if len(lx) < 3:
        i = int(np.argmin(ly))
        return lx[i], ly[i], False
    c2, c1, c0 = np.polyfit(lx, ly, 2)
    if c2 <= 0:
        i = int(np.argmin(ly))
        return lx[i], ly[i], False
    xs = float(np.clip(-c1 / (2 * c2), lo, hi))
    return xs, float(np.polyval([c2, c1, c0], xs)), True


def q5_stage(df, q1_fits=None):
    out = {}
    # (a) D = 50M calibration by corpus (lrsweep: edu; lrcal: web), own validation set
    rows = []
    for r, tag in (("edu", "lrsweep"), ("web", "lrcal")):
        s = df[(df.regime == r) & (df.tag == tag)]
        for d, g in s.groupby("d"):
            g = g.assign(y=mc.own_output(g))
            g = g[np.isfinite(g.y)].sort_values("lr")
            if len(g) < 3:
                continue
            xs, ymin, cvx = quad_argmin(np.log(g.lr), np.log(g.y))
            ib = int(np.argmin(g.y.values))
            main = df[(df.regime == r) & (df.tag == "main") & (df.d == d) & (df.D_target == 50_000_000)]
            yr = float(np.log(mc.own_output(main)[0])) if len(main) else np.nan
            rows.append(dict(corpus=r, d=d, n_lr=len(g), lr_star_quad=float(np.exp(xs)), lr_best_grid=float(g.lr.values[ib]),
                             convex=cvx, lr_rule=float(main.lr.values[0]) if len(main) else np.nan,
                             lnL_min_quad=ymin, lnL_best_grid=float(np.log(g.y.values[ib])), lnL_rule=yr,
                             excess_rule_vs_quad=yr - ymin if np.isfinite(yr) else np.nan,
                             excess_rule_vs_grid=yr - float(np.log(g.y.values[ib])) if np.isfinite(yr) else np.nan,
                             **{f"bpb_lr{lr:g}": y for lr, y in zip(g.lr, g.y)}))
    cal = pd.DataFrame(rows)
    if not cal.empty:
        piv = cal.pivot_table(index="d", columns="corpus", values="lr_star_quad")
        if {"edu", "web"} <= set(piv.columns):
            cal = cal.merge((np.log(piv["edu"]) - np.log(piv["web"])).rename("dln_lrstar_edu_minus_web").reset_index(), on="d", how="left")
        for r in cal.corpus.unique():
            g = cal[cal.corpus == r]
            if len(g) >= 2:
                p, lc = np.polyfit(np.log(g.d / 256.0), np.log(g.lr_star_quad), 1)
                cal.loc[cal.corpus == r, "rule_refit_c"] = float(np.exp(lc))
                cal.loc[cal.corpus == r, "rule_refit_p"] = float(p)
    out["calibration"] = cal
    # (b) corners: LR x {0.5, 1, 2} at (d = 128, D ladder) and (d = 640, D in {25, 50}M)
    rows = []
    for r in mc.CORPORA:
        lc = df[(df.regime == r) & (df.tag == "lrcorner")]
        for (d, D_t), g in lc.groupby(["d", "D_target"]):
            main = df[(df.regime == r) & (df.tag == "main") & (df.d == d) & (df.D_target == D_t) & (df.seed == 0)]
            gg = pd.concat([g, main])
            for v in mc.VALSETS:
                gv = gg[np.isfinite(gg[f"bpb_{v}"])]
                if gv.empty:
                    continue
                row = dict(corpus=r, d=d, D_target=D_t, valset=v, n_lr=len(gv))
                for m_, y in zip(gv.lr_mult.round(3), gv[f"bpb_{v}"]):
                    row[f"bpb_x{m_:g}"] = y
                if len(gv) >= 3 and (gv.lr_mult.round(3) == 1).any():
                    xs, ymin, cvx = quad_argmin(np.log(gv.lr_mult), np.log(gv[f"bpb_{v}"]))
                    y1 = float(np.log(gv[gv.lr_mult.round(3) == 1][f"bpb_{v}"].values[0]))
                    ybest = float(np.log(gv[f"bpb_{v}"].min()))
                    row.update(mult_star=float(np.exp(xs)), convex=cvx, excess_rule_quad=y1 - min(ymin, ybest),
                               excess_rule_grid=y1 - ybest,
                               best_mult_grid=float(gv.lr_mult.values[int(np.argmin(gv[f"bpb_{v}"].values))]))
                rows.append(row)
    cor = pd.DataFrame(rows)
    out["corners"] = cor
    # drift of the optimal LR with D at the high-M corner (d = 128), own validation set
    drift = []
    if not cor.empty and "mult_star" in cor:
        for r in mc.CORPORA:
            g = cor[(cor.corpus == r) & (cor.d == 128) & (cor.valset == r) & np.isfinite(cor.get("mult_star", np.nan))]
            if len(g) >= 3:
                X = np.column_stack([np.ones(len(g)), np.log(g.D_target)])
                y = np.log(g.mult_star.values)
                b, res, *_ = np.linalg.lstsq(X, y, rcond=None)
                e = y - X @ b
                se = float(np.sqrt(np.sum(e ** 2) / max(len(y) - 2, 1) / np.sum((X[:, 1] - X[:, 1].mean()) ** 2)))
                drift.append(dict(corpus=r, d=128, n_budgets=len(g), dlnLRstar_dlnD=float(b[1]), se=se))
    out["drift"] = pd.DataFrame(drift)
    # (c) bound on the allocation-exponent bias (flexible-input formula, Online Appendix eq. app-flexbias)
    bnd = []
    if not cor.empty and "excess_rule_quad" in cor and q1_fits:
        for r in mc.CORPORA:
            for conv in mc.CONVS:
                if (r, conv) not in q1_fits:
                    continue
                th = q1_fits[(r, conv)]
                m = sl_model(th)
                for (dA, DA), (dB, DB) in (((640, 25_000_000), (128, 800_000_000)), ((640, 50_000_000), (128, 800_000_000))):
                    a = cor[(cor.corpus == r) & (cor.valset == r) & (cor.d == dA) & (cor.D_target == DA)]
                    b = cor[(cor.corpus == r) & (cor.valset == r) & (cor.d == dB) & (cor.D_target == DB)]
                    if a.empty or b.empty or not np.isfinite(a.excess_rule_quad.values[0]) or not np.isfinite(b.excess_rule_quad.values[0]):
                        continue
                    ra = df[(df.regime == r) & (df.tag == "main") & (df.d == dA) & (df.D_target == DA)].iloc[0]
                    rb = df[(df.regime == r) & (df.tag == "main") & (df.d == dB) & (df.D_target == DB)].iloc[0]
                    tA = np.log(ra[f"N_{conv}"]) - np.log(ra.D)
                    tB = np.log(rb[f"N_{conv}"]) - np.log(rb.D)
                    iA, iB = a.excess_rule_quad.values[0], b.excess_rule_quad.values[0]
                    grad = 2 * (iA - iB) / (tA - tB)
                    Cm = float(np.exp(0.5 * (np.log(6 * ra[f"N_{conv}"] * ra.D) + np.log(6 * rb[f"N_{conv}"] * rb.D))))
                    Rstar = float(m.L_opt(Cm) - m.E)
                    bias = -grad * m.E / ((m.alpha + m.beta) * Rstar)
                    bnd.append(dict(corpus=r, conv=conv, cell_lowM=f"({dA},{DA/1e6:.0f}M)", cell_highM=f"({dB},{DB/1e6:.0f}M)",
                                    iota_lowM=iA, iota_highM=iB, iota_n_minus_iota_d=grad, C_mid=Cm, E=m.E,
                                    alpha_plus_beta=m.alpha + m.beta, R_star=Rstar, a_obs_minus_a=bias))
    out["abias"] = pd.DataFrame(bnd)
    return out


def sl_model(th):
    return mc.sl.Chinchilla.from_theta(th)


# ============================================================================ Q6 (functional dependence)
def q6_stage(df, r, conv, sample="main", k_near=(1, 2)):
    import mc_lib as ml
    s = mc.main_sample(df, r, sample)
    s = s.assign(y=mc.own_output(s))
    s = s[np.isfinite(s.y)].reset_index(drop=True)
    if not enough(s):
        return None
    N, D, L = s[f"N_{conv}"].values, s["D"].values, s["y"].values
    th, _ = est.fit_chin(N, D, L, "huber", grid="fast")
    p, _ = est.fit_kappa(N, D, L, "huber", th_chin=th)
    pc = "P" if conv == "P" else "T"
    Cn = s["C_P"].values if conv == "P" else 6 * N * D
    lnMs = est.param_path(pc, lambda N_, D_: est.w_kappa(p, N_, D_), p[3], p[4], Cn, 12.0, 18.0)[:, 2]
    dist = np.abs(np.log(D / N) - lnMs)
    b = np.floor((np.log(Cn) - np.log(Cn).min()) / np.log(2.0)).astype(int)
    out = dict(rows=[], profiles=[])
    subsets = {"full": np.ones(len(N), bool)}
    for k in k_near:
        m = np.zeros(len(N), bool)
        for bb in np.unique(b):
            idx = np.where(b == bb)[0]
            m[idx[np.argsort(dist[idx])[:k]]] = True
        subsets[f"onpath{k}"] = m
    for name, m in subsets.items():
        lN, lD, lL = np.log(N[m]), np.log(D[m]), np.log(L[m])
        raw, nrm = est.cond_jac(th, N[m], D[m])
        cn_sd = float(np.std(np.log(D[m] / N[m]) - lnMs[m]))
        try:
            thm, _ = est.fit_chin(N[m], D[m], L[m], "nls", grid="fast")
            lr, info = ml.profile_sigma(lN, lD, lL, SIG_GRID, np.clip(thm, [b_[0] for b_ in ml.BOUNDS], [b_[1] for b_ in ml.BOUNDS]),
                                        return_info=True, extra_th=(np.clip(th, [b_[0] for b_ in ml.BOUNDS], [b_[1] for b_ in ml.BOUNDS]),))
        except Exception as e:  # noqa
            lr, info = np.full(len(SIG_GRID), np.nan), dict(prof_sig_hat=np.nan)
        inset = lr <= 3.841458820694124
        out["rows"].append(dict(corpus=r, conv=conv, sample=sample, subset=name, n=int(m.sum()), cond_raw=raw,
                                cond_normalized=nrm, sd_lnM_dev_from_path=cn_sd, sigma_hat_profile=info.get("prof_sig_hat"),
                                set_lo=float(SIG_GRID[inset].min()) if inset.any() else np.nan,
                                set_hi=float(SIG_GRID[inset].max()) if inset.any() else np.nan,
                                share_grid_in_set=float(inset.mean()), flat=bool(np.nanmax(lr) < 3.841458820694124),
                                LR_at_050=float(np.interp(0.5, SIG_GRID, lr)), LR_at_070=float(np.interp(0.7, SIG_GRID, lr)),
                                LR_at_090=float(np.interp(0.9, SIG_GRID, lr))))
        for sg, l_ in zip(SIG_GRID, lr):
            out["profiles"].append(dict(corpus=r, conv=conv, subset=name, sigma_star=sg, LR=l_))
    return out


# ============================================================================ design / inventory
def inventory(df):
    plan = mc.design_cells()
    key = ["regime", "tag", "d", "L", "seed", "D_target"]
    pk = plan.assign(lr=plan.lr.round(6))
    dk = df.assign(lr=df.lr.round(6))
    have = set(map(tuple, dk[key + ["lr"]].values))
    plan["done"] = [tuple(v) in have for v in pk[key + ["lr"]].values]
    inv = plan.groupby(["regime", "tag"]).agg(planned=("done", "size"), completed=("done", "sum")).reset_index()
    extra = dk[~dk.set_index(key + ["lr"]).index.isin(pk.set_index(key + ["lr"]).index)]
    rows = []
    for (r, t), g in df.groupby(["regime", "tag"]):
        tr = g.groupby(["d", "L", "lr", "seed"])["wall"].max()
        rows.append(dict(regime=r, tag=t, trunks_with_endpoints=len(tr), wall_hours_lower_bound=float(tr.sum() / 3600),
                         any_nonfinite=bool(~np.isfinite(g[f"loss_{r}"]).all()),
                         any_diverged=bool((g[f"loss_{r}"] > np.log(mc.V)).any()),
                         duplicate_records=int((g.n_records > 1).sum())))
    inv = inv.merge(pd.DataFrame(rows), on=["regime", "tag"], how="left")
    return inv, extra


def design_stats(df):
    rows = []
    for r in mc.CORPORA:
        for tagset, lab in ((("main",), "main"), (("main", "hiM"), "main+hiM")):
            s = df[(df.regime == r) & df.tag.isin(tagset) & (df.seed == 0)]
            if lab == "main+hiM":
                s = mc.main_sample(df, r, "main", with_hiM=True)
            if s.empty:
                continue
            for conv in mc.CONVS:
                N, M = s[f"N_{conv}"], s[f"M_{conv}"]
                C = s["C_P"] if conv == "P" else s["C_T"]
                X = np.column_stack([np.ones(len(s)), np.log(C)])
                e = np.log(M) - X @ np.linalg.lstsq(X, np.log(M), rcond=None)[0]
                rows.append(dict(corpus=r, sample=lab, conv=conv, n=len(s), widths=s.arch.nunique(),
                                 N_min=N.min(), N_max=N.max(), D_min=s.D.min(), D_max=s.D.max(), M_min=M.min(),
                                 M_max=M.max(), C_min=C.min(), C_max=C.max(), sd_lnM_given_lnC=float(np.std(e, ddof=2)),
                                 n_M_gt_100=int((M > 100).sum()), n_M_gt_341=int((M > 341).sum()),
                                 n_M_gt_1000=int((M > 1000).sum()),
                                 emb_share_min=float((s.N_emb / s.N_total).min()), emb_share_max=float((s.N_emb / s.N_total).max()),
                                 flops_over_6Nne_min=float((s.flops_per_token / (6 * s.N_nonemb)).min()),
                                 flops_over_6Nne_max=float((s.flops_per_token / (6 * s.N_nonemb)).max()),
                                 flops_over_6Ntot_min=float((s.flops_per_token / (6 * s.N_total)).min()),
                                 flops_over_6Ntot_max=float((s.flops_per_token / (6 * s.N_total)).max())))
    return pd.DataFrame(rows)


def cross_eval(df):
    """Mean bpb of each corpus's main-grid models on each validation set, over cells trained in both corpora."""
    mcells = matched_cells(df[(df.tag == "main") & (df.seed == 0)])
    rows = []
    if mcells.empty:
        return pd.DataFrame(), mcells
    for v in mc.VALSETS:
        a, b = mcells[f"bpb_edu_{v}"], mcells[f"bpb_web_{v}"]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() == 0:
            continue
        rows.append(dict(valset=v, n_cells=int(ok.sum()), bpb_edu_models=float(a[ok].mean()), bpb_web_models=float(b[ok].mean()),
                         mean_dlnL_edu_minus_web=float(mcells[f"dlnL_{v}"][ok].mean()),
                         min_dlnL=float(mcells[f"dlnL_{v}"][ok].min()), max_dlnL=float(mcells[f"dlnL_{v}"][ok].max())))
    return pd.DataFrame(rows), mcells


# ============================================================================ bandwidth sensitivity (model-free)
def mf_bandwidth_sensitivity(df, conv, sample="main", factors=(0.75, 1.0, 1.5, 2.0),
                             ext_mults=(0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0)):
    """Point estimates of the model-free sigma* at the CV bandwidth times each factor, and at the optimum of an
    extended CV grid (the primary grid's optimum lies on its lower edge for ln N)."""
    rows = []
    for r in mc.CORPORA:
        s = mc.main_sample(df, r, sample)
        s = s.assign(y=mc.own_output(s))
        s = s[np.isfinite(s.y)]
        if not enough(s):
            continue
        N, D, L = s[f"N_{conv}"].values, s["D"].values, s["y"].values
        prep, _ = mf_prepare(N, D, L)
        f = np.log(L)
        (rm, hx_e, hz_e, mx_e, mz_e), _ = est.cv_bandwidth(prep["x"], prep["z"], f, ext_mults)
        variants = [(f"cv x {k:g}", prep["hx"] * k, prep["hz"] * k) for k in factors]
        variants.append((f"extended-grid CV (m_N={mx_e:g}, m_D={mz_e:g})", hx_e, hz_e))
        for lab, hx, hz in variants:
            p2 = dict(prep, hx=hx, hz=hz)
            ms = mf_stats(p2, f, conv)
            for pc in paths_of(conv):
                rows.append(dict(corpus=r, conv=conv, sample=sample, bandwidth=lab, path=pc, hx=hx, hz=hz,
                                 sigma_mf=ms[f"sig_mf_{pc}"], sigma_mf_slope=ms[f"sig_mfslope_{pc}"],
                                 n_levels=int(np.isfinite(ms[f"_path_{pc}"][:, 1]).sum())))
    return rows
