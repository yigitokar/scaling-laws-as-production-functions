"""jointboot.py -- joint design-conditional bootstrap draws of the budget-level sigma*_b (T1.1), the Chinchilla-Llama 3
drift and the top-budget mean bootstrapped directly (T1.1), and the top-budget set (T1.3).

Draws (T1.1; R2 NM3(b)). ra1's estimator and bootstrap (isoflop.design_estimate / boot_design, imported unchanged) are
re-run with ra1's own seeds, windows, orders and B = 999, so that the per-budget standard errors reproduce the published
ones exactly (Chinchilla in N_F: rb4_chinflop's override inputs; checked below). Within a design every sigma*_b shares
the frontier derivative (one log-cubic through all the minima) and the budget-level shocks, so the draws are correlated
across budgets; this module keeps the whole (B x k) matrix of draws per design. Designs are estimated on disjoint data,
so draws are independent across designs.

Statistics bootstrapped directly (linear in the sigma*_b with the published fixed weights):
  * the Chinchilla + Llama 3 design fixed-effect slope (inverse-variance weights 1/se_b^2, as in rb1's 'fe_weighted');
  * the top-budget mean (inverse-variance mean of the five budgets at >= 6e20 FLOP in the two designs).
Deviations are taken on the sigma scale through the delta method on S_b = 2(1/sigma*_b - 1), which is how ra1 defines
the per-budget standard errors (se_sigma = |d sigma/dS| sd(S*_b - S0_b)); a direct sigma-scale version is a check.
Standard errors: 'independent' uses the bootstrap variances only (reproduces the model-based unscaled standard error),
'joint' uses the full bootstrap covariance. Intervals: normal (estimate +- 1.96 joint s.e.), percentile (estimate plus
the 2.5 and 97.5 percent quantiles of the draws' deviations from the bootstrap population) and basic (ra1's).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import sigcommon as cm

rc = cm.rc
B_ISO = 99 if cm.QUICK else 999            # = ra1's B_ISO (reproduction)
TOP_MIN = 6e20
CL3 = ("Chinchilla", "Llama 3")
TOP_NAMES = {"Chinchilla": "Chinchilla", "Llama 3": "Llama 3", "Chinchilla (132 runs)": "Chinchilla"}


# ============================================================================ designs
def designs():
    """ra1's IsoFLOP designs with Chinchilla in N_F (rb4_chinflop's primary: x = ln(F_T4/6) on the 137 profile runs) and
    the 132-run Chinchilla sample without the five highest-loss runs (rb4 review R13), in memory only."""
    import rb4_estim as es
    D = rc.isoflop_designs()
    out = {k: D[k] for k in ("Llama 3", "Marin, Comma", "Marin, DCLM", "Marin, Nemotron-CC")}
    df, _ = es.load_runs()
    iso = df[df.iso]
    xF = lambda s: np.log(s["F_T4"].values / 6.0)   # noqa: E731
    d137 = iso.copy()
    d137["iso"] = True
    out["Chinchilla"] = es.design_df(d137, xF(d137))
    d132 = iso[iso["rank_worst"] > 5].copy()
    d132["iso"] = True
    out["Chinchilla (132 runs)"] = es.design_df(d132, xF(d132))
    return out


def _job(job, D):
    """Worker: ra1's primary estimator and design-conditional bootstrap for one design; keeps the draws."""
    import isoflop as iso_
    df = D[job["name"]]
    res = iso_.design_estimate(df, job["order"], 1.0, "auto", "path")
    if not res.get("ok"):
        return dict(name=job["name"], ok=False)
    bt = iso_.boot_design(df, res, job["B"], job["seed"], budget_shock=True)
    s, rows, inv, _ = iso_.summarize_design(job["name"], res, bt, job["B"])
    budgets = [float(res["fits"][b]["C"]) for b in res["valid"]]
    return dict(name=job["name"], ok=True, summary=s, rows=rows, budgets=budgets, S=np.asarray(res["S"], float),
                S0=np.asarray(bt["S0"], float), Sd=np.asarray(bt["S"], float), fkind=res["fkind"])


def run_draws(log=cm.log):
    D = designs()
    order = pd.read_csv(cm.RA1_ORDER).set_index("design")["order_chosen"]
    jobs = []
    for name in D:
        base = "Chinchilla" if name.startswith("Chinchilla") else name
        o = int(order.get(base, 2))
        jobs.append(dict(name=name, order=o, B=B_ISO, seed=rc.seed_of(f"iso|{base}")))   # ra1's seeds
    log(f"  joint draws: {len(jobs)} designs, B = {B_ISO}, ra1's seeds, {cm.N_PROC} processes")
    res = cm.pmap(_job, jobs, dict(D=D))
    out = {}
    for j, r in zip(jobs, res):
        if "_error" in r or not r.get("ok"):
            raise RuntimeError(f"design {j['name']}: {r}")
        out[j["name"]] = r
        s = r["summary"]
        log(f"    {j['name']:22s} k={s['k_valid']} RE {s['sigma_re']:.4f} ({s['se_sigma_re']:.4f}); "
            f"draws with any non-finite budget: {int(np.sum(~np.all(np.isfinite(r['Sd']), axis=1)))} of {len(r['Sd'])}")
    return out


def check_reproduction(R):
    """Per-budget sigma*_b and s.e. against rb4_chinflop's override budgets (the primary inputs of rb1)."""
    ref = pd.read_csv(cm.RB4_BUDGETS)
    rows = []
    for name in cm.DESIGNS_ISO:
        mine = pd.DataFrame(R[name]["rows"])
        g = ref[ref.design == name].sort_values("budget_C")
        mm = mine.merge(g, on=["design", "budget_C"], suffixes=("", "_ref"))
        for col in ("sigma", "se_sigma", "lo_sigma", "hi_sigma"):
            rows.append(dict(design=name, column=col, n_budgets=len(mm), n_ref=len(g),
                             max_abs_diff=float(np.max(np.abs(mm[col].values - mm[col + "_ref"].values)))))
    summ = pd.read_csv(cm.RB4_SUMMARY).set_index("design")
    for name in cm.DESIGNS_ISO:
        s = R[name]["summary"]
        for col in ("sigma_re", "se_sigma_re", "sigma_fe"):
            rows.append(dict(design=name, column=f"summary {col}", n_budgets=s["k_valid"], n_ref=int(summ.loc[name, "k_valid"]),
                             max_abs_diff=float(abs(s[col] - summ.loc[name, col]))))
    # R13 (rb4 review): 132-run N_F RE sigma* 0.681247 (0.0198)
    ch = pd.read_csv(cm.RB4_CHECKS)
    r13 = ch[ch.check.str.startswith("R13") & ch.statistic.str.contains("N_F")].iloc[0]
    rows.append(dict(design="Chinchilla (132 runs)", column="summary sigma_re vs rb4 review R13", n_budgets=R["Chinchilla (132 runs)"]["summary"]["k_valid"],
                     n_ref=9, max_abs_diff=float(abs(R["Chinchilla (132 runs)"]["summary"]["sigma_re"] - r13.value))))
    T = pd.DataFrame(rows)
    T["tolerance"] = np.where(T.column.str.contains("R13"), 1e-6, 1e-9)    # R13 is printed with %.6g in rb4's CSV
    T["ok"] = T.max_abs_diff < T.tolerance
    return T


# ============================================================================ budget-level table and draws
def budget_table(R):
    """Budget-level estimates with population values and the deviations' moments (long table of the draws is separate)."""
    rows = []
    for name, r in R.items():
        S, S0, Sd = r["S"], r["S0"], r["Sd"]
        for j, C in enumerate(r["budgets"]):
            dv = (Sd[:, j] - S0[j])
            dv = dv[np.isfinite(dv)]
            rows.append(dict(design=name, budget_C=C, log10C=np.log10(C), sigma=float(cm.sig_from_S(S[j])),
                             sigma_pop=float(cm.sig_from_S(S0[j])), S=S[j], S_pop=S0[j],
                             se_sigma=float(abs(cm.dsig_dS(S[j])) * np.std(dv, ddof=1)),
                             n_draws_finite=len(dv), B=len(Sd)))
    return pd.DataFrame(rows)


def draws_long(R):
    """Long table of the draws (sigma scale, direct): design, budget, draw index, sigma*_b draw (derived statistics)."""
    out = []
    for name, r in R.items():
        Sd = r["Sd"]
        for j, C in enumerate(r["budgets"]):
            out.append(pd.DataFrame(dict(design=name, budget_C=C, draw=np.arange(len(Sd)), S_draw=Sd[:, j],
                                         sigma_draw=cm.sig_from_S(Sd[:, j]))))
    return pd.concat(out, ignore_index=True)


def correlations(R):
    """Bootstrap correlation of sigma*_b across budgets within each design (delta-method deviations)."""
    rows = []
    for name, r in R.items():
        Sd, S0 = r["Sd"], r["S0"]
        ok = np.all(np.isfinite(Sd), axis=1)
        dev = Sd[ok] - S0
        Cm = np.corrcoef(dev.T)
        iu = np.triu_indices(len(S0), 1)
        adj = np.array([Cm[i, i + 1] for i in range(len(S0) - 1)])
        rows.append(dict(design=name, k=len(S0), draws_used=int(ok.sum()), mean_offdiag_corr=float(np.mean(Cm[iu])),
                         min_offdiag_corr=float(np.min(Cm[iu])), max_offdiag_corr=float(np.max(Cm[iu])),
                         mean_adjacent_corr=float(np.mean(adj)),
                         corr_top2=float(Cm[-1, -2]) if len(S0) >= 2 else np.nan))
    return pd.DataFrame(rows)


# ============================================================================ linear statistics of the sigma*_b
def _stack(R, members):
    """members: list of (design, budget_C). Returns estimates, published-style s.e., delta-method deviation draws
    (B x m) on the sigma scale, direct sigma-scale deviation draws, and the population values."""
    est, se, pop, dev, dev_direct = [], [], [], [], []
    B = None
    for des, C in members:
        r = R[des]
        j = int(np.argmin(np.abs(np.array(r["budgets"]) - C)))
        assert abs(r["budgets"][j] / C - 1) < 1e-6, (des, C)
        S, S0, Sd = r["S"][j], r["S0"][j], r["Sd"][:, j]
        B = len(Sd) if B is None else B
        dS = Sd - S0
        est.append(float(cm.sig_from_S(S)))
        pop.append(float(cm.sig_from_S(S0)))
        fin = np.isfinite(dS)
        se.append(float(abs(cm.dsig_dS(S)) * np.std(dS[fin], ddof=1)))
        dev.append(cm.dsig_dS(S) * dS)
        dev_direct.append(cm.sig_from_S(Sd) - cm.sig_from_S(S0))
    return (np.array(est), np.array(se), np.array(pop), np.column_stack(dev), np.column_stack(dev_direct))


def _summ(label, est, a, dev, dev_direct, se_model_scaled=np.nan, extra=None):
    """A linear statistic a'y: joint and independent bootstrap s.e., intervals."""
    ok = np.all(np.isfinite(dev), axis=1)
    d = dev[ok] @ a
    dd = dev_direct[ok] @ a
    var_ind = float(np.sum(a ** 2 * np.var(dev[ok], axis=0, ddof=1)))
    se_joint = float(np.std(d, ddof=1))
    q = np.percentile(d, [2.5, 97.5])
    row = dict(statistic=label, estimate=est, se_model_scaled=se_model_scaled, se_boot_independent=np.sqrt(var_ind),
               se_boot_joint=se_joint, se_boot_joint_direct=float(np.std(dd, ddof=1)),
               se_boot_joint_robust=float((np.percentile(d, 75) - np.percentile(d, 25)) / 1.349),
               ratio_joint_to_independent=se_joint / np.sqrt(var_ind),
               lo_normal=est - 1.96 * se_joint, hi_normal=est + 1.96 * se_joint,
               lo_percentile=est + q[0], hi_percentile=est + q[1], lo_basic=est - q[1], hi_basic=est - q[0],
               boot_mean_dev=float(np.mean(d)), draws_used=int(ok.sum()), draws_total=len(dev))
    if extra:
        row.update(extra)
    return row


def fe_slope_weights(log10C, design, w):
    """Weights a such that the design fixed-effect WLS slope is a'y (fixed weights w)."""
    X = np.column_stack([pd.get_dummies(design).astype(float).values, log10C - 20.0])
    W = np.diag(w)
    A = np.linalg.solve(X.T @ W @ X, X.T @ W)
    return A[-1], X


def linear_stats(R):
    """T1.1: the Chinchilla + Llama 3 design-FE slope and the top-budget mean, bootstrapped directly; T1.3 top-budget
    variants with the 132-run Chinchilla sample."""
    rows = []
    # Chinchilla + Llama 3 design FE slope (17 budgets), inverse-variance weights from the published s.e.
    mem = [(des, C) for des in CL3 for C in R[des]["budgets"]]
    est, se, pop, dev, devd = _stack(R, mem)
    lc = np.log10([C for _, C in mem])
    des = np.array([d_ for d_, _ in mem])
    a, X = fe_slope_weights(lc, des, 1 / se ** 2)
    b = float(a @ est)
    r_ = est - X @ np.linalg.lstsq(X * (1 / se)[:, None], est / se, rcond=None)[0]
    s2 = float(np.sum(r_ ** 2 / se ** 2) / (len(est) - X.shape[1]))
    se_mb = float(np.sqrt(np.sum(a ** 2 * se ** 2)))
    rows.append(_summ("Chinchilla + Llama 3 drift: design FE slope, inverse-variance (17 budgets)", b, a, dev, devd,
                      se_model_scaled=se_mb * np.sqrt(max(s2, 1.0)),
                      extra=dict(se_model_unscaled=se_mb, resid_var_scale=s2, k=len(est))))
    # unweighted FE slope (OLS) as a check
    a_u, _ = fe_slope_weights(lc, des, np.ones(len(est)))
    rows.append(_summ("Chinchilla + Llama 3 drift: design FE slope, unweighted (17 budgets)", float(a_u @ est), a_u, dev, devd,
                      extra=dict(k=len(est))))
    # each design's own slope (IV weights)
    for dname in CL3:
        m = des == dname
        lc_ = lc[m]
        w_ = 1 / se[m] ** 2
        Xd = np.column_stack([np.ones(m.sum()), lc_ - 20.0])
        A = np.linalg.solve(Xd.T @ (Xd * w_[:, None]), (Xd * w_[:, None]).T)[1]
        a_full = np.zeros(len(est))
        a_full[m] = A
        rows.append(_summ(f"{dname} drift: own slope, inverse-variance ({int(m.sum())} budgets)", float(a_full @ est), a_full,
                          dev, devd, extra=dict(se_model_unscaled=float(np.sqrt(np.sum(A ** 2 * se[m] ** 2))), k=int(m.sum()))))
    # top-budget means
    tops = {}
    for lab, dnames in (("Chinchilla + Llama 3, budgets >= 6e20 (primary)", ("Chinchilla", "Llama 3")),
                        ("Chinchilla (132 runs, five highest-loss runs dropped) + Llama 3, budgets >= 6e20",
                         ("Chinchilla (132 runs)", "Llama 3"))):
        mem = [(d_, C) for d_ in dnames for C in R[d_]["budgets"] if C >= TOP_MIN * (1 - 1e-9)]
        est, se, pop, dev, devd = _stack(R, mem)
        w = 1 / se ** 2
        a = w / w.sum()
        mu = float(a @ est)
        tops[lab] = (mem, est, se, a)
        from meta import meta as dl
        m_ = dl(est, se)
        shares = {f"weight {TOP_NAMES[d_]} {C:.0e}": float(ai) for (d_, C), ai in zip(mem, a)}
        sh_ll = float(sum(ai for (d_, C), ai in zip(mem, a) if d_ == "Llama 3"))
        rows.append(_summ(f"Top-budget mean: {lab}", mu, a, dev, devd, se_model_scaled=m_["se_fixed"],
                          extra=dict(k=len(est), Q=m_["Q"], p_Q=m_["p_Q"], tau=m_["tau"], mean_re=m_["mu_re"],
                                     lo_hksj=m_["lo_hksj"], hi_hksj=m_["hi_hksj"], se_hksj=m_["se_hksj"],
                                     weight_share_llama3=sh_ll,
                                     members="; ".join(f"{TOP_NAMES[d_]} {C:.0e}: {e:.3f} ({s:.3f})"
                                                       for (d_, C), e, s in zip(mem, est, se)),
                                     weights="; ".join(f"{k.replace('weight ', '')} {v:.3f}" for k, v in shares.items()))))
    return pd.DataFrame(rows), tops


def top_budget_set(R):
    """T1.3 (R4 N3(b), (e)): Farseer's levels above 3e20, the top-budget mean with and without Farseer's 1e21 level,
    with and without Chinchilla's five highest-loss runs, and Llama 3's weight share."""
    from meta import meta as dl
    bi = pd.read_csv(cm.RB1_BUDGETS_IN)
    far = bi[bi.design == cm.FARSEER].sort_values("budget_C")
    rows = []
    for _, r in far[far.budget_C > 3e20 * (1 + 1e-9)].iterrows():
        rows.append(dict(set=f"Farseer local path level at {r.budget_C:.0e} FLOP", k=1, mean_fixed=r.y, se_fixed=r.se,
                         mean_re=r.y, lo_hksj=np.nan, hi_hksj=np.nan, tau=np.nan, Q=np.nan, p_Q=np.nan,
                         members=f"Farseer {r.budget_C:.0e}: {r.y:.3f} ({r.se:.3f})"))
    far21 = far[np.isclose(far.budget_C, 1e21)].iloc[0]
    far520 = far[np.isclose(far.budget_C, 5e20)].iloc[0]

    def add(label, ent, note=""):
        y = np.array([e[2] for e in ent])
        s = np.array([e[3] for e in ent])
        m = dl(y, s)
        w = 1 / s ** 2
        sh = {}
        for (d_, C, _, _), wi in zip(ent, w / w.sum()):
            sh[d_] = sh.get(d_, 0.0) + wi
        rows.append(dict(set=label, k=len(y), mean_fixed=m["mu_fixed"], se_fixed=m["se_fixed"], mean_re=m["mu_re"],
                         lo_hksj=m["lo_hksj"], hi_hksj=m["hi_hksj"], tau=m["tau"], Q=m["Q"], p_Q=m["p_Q"],
                         weight_share_llama3=sh.get("Llama 3", 0.0), weight_share_farseer=sh.get("Farseer", 0.0),
                         members="; ".join(f"{d_} {C:.0e}: {y_:.3f} ({s_:.3f})" for d_, C, y_, s_ in ent), note=note))

    def ent_of(dname, label):
        r = R[dname]
        out = []
        for j, C in enumerate(r["budgets"]):
            if C >= TOP_MIN * (1 - 1e-9):
                dS = r["Sd"][:, j] - r["S0"][j]
                dS = dS[np.isfinite(dS)]
                out.append((label, C, float(cm.sig_from_S(r["S"][j])), float(abs(cm.dsig_dS(r["S"][j])) * np.std(dS, ddof=1))))
        return out
    base = ent_of("Chinchilla", "Chinchilla") + ent_of("Llama 3", "Llama 3")
    b132 = ent_of("Chinchilla (132 runs)", "Chinchilla (132 runs)") + ent_of("Llama 3", "Llama 3")
    fz = [("Farseer", 1e21, float(far21.y), float(far21.se))]
    fz2 = [("Farseer", 5e20, float(far520.y), float(far520.se))] + fz
    add("Chinchilla + Llama 3, budgets >= 6e20 (primary)", base)
    add("Chinchilla + Llama 3 + Farseer 1e21", base + fz)
    add("Chinchilla + Llama 3 + Farseer 5e20 and 1e21 (all levels above 3e20)", base + fz2,
        "5e20 is below the 6e20 cut of the primary set; shown for completeness")
    add("Chinchilla (132 runs) + Llama 3, budgets >= 6e20", b132, "Chinchilla without its five highest-loss runs (rb4 review R13)")
    add("Chinchilla (132 runs) + Llama 3 + Farseer 1e21", b132 + fz)
    add("Llama 3 only, budgets >= 6e20", ent_of("Llama 3", "Llama 3"))
    add("Chinchilla only, budgets >= 6e20 (137 runs)", ent_of("Chinchilla", "Chinchilla"))
    add("Chinchilla only, budgets >= 6e20 (132 runs)", ent_of("Chinchilla (132 runs)", "Chinchilla (132 runs)"))
    return pd.DataFrame(rows)
