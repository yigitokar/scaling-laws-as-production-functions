"""Task 2 -- selection on the outcome: dropping the k worst-loss runs, alternative exclusion rules, and a
Hausman-Wise (1977) truncated-regression check.

Econometric reading. Dropping the k highest-loss runs is truncation on the dependent variable: if the model were
correctly specified, E[ln L | N, D, L < Lbar] != ln Lhat(N, D) near the threshold and estimates are biased
(Hausman and Wise 1977). Selection on a regressor (e.g. D/N < 0.4) is ignorable under correct specification. In the
Chinchilla extraction the 5 worst-loss runs are also the most data-starved runs of the 1e19 IsoFLOP profile, so the
two kinds of rule nearly coincide; comparing them, and fitting the truncated likelihood, separates truncation bias
from lack of fit in the D/N -> 0 corner.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

import boot
import estimators as est
from common import (BESI_PUB, HOFF_TEX, SEED, TABLES, derived, load_chinchilla, sl, theta_of)

K_MAX = 15
B_SEL = 200
KEYS = ["E", "A", "B", "alpha", "beta", "a", "gamma", "sigma_star", "Mstar_1e+21", "Mstar_5.76e+23", "w_chin70b"]


def _cands(extra=()):
    return [theta_of(HOFF_TEX), theta_of(BESI_PUB)] + list(extra)


def fit_rule(N, D, L, name="huber", inits=None):
    if name == "huber":
        m1 = est.fit_huber(N, D, L, grid=sl.FAST_GRID)
        m2 = est.fit_huber(N, D, L, inits=inits or _cands())
        return m1 if m1.extra["objective"] <= m2.extra["objective"] else m2
    return est.fit(name, N, D, L, inits=inits or _cands())


def boot_huber(idx, N, D, L, inits, Lbar=None, **_):
    """Huber fit on a resample; with Lbar also on the resample truncated at L < Lbar (paired)."""
    m = est.fit_huber(N[idx], D[idx], L[idx], inits=inits)
    d = derived(m)
    if Lbar is not None:
        keep = idx[L[idx] < Lbar]
        mt = est.fit_huber(N[keep], D[keep], L[keep], inits=inits)
        dt = derived(mt)
        d.update({f"t_{k}": v for k, v in dt.items()})
    return d


# ----------------------------------------------------------------------------- Hausman-Wise truncated regression
def fit_truncated_gauss(N, D, L, Lbar, init):
    """Gaussian MLE for ln L = ln Lhat(theta) + u, u ~ N(0, s^2), in a sample truncated to L < Lbar."""
    lN, lD, y = np.log(N), np.log(D), np.log(L)
    ybar = np.log(Lbar)

    def nll(p):
        th, ls = p[:5], p[5]
        s = np.exp(ls)
        mu = est.lse_pred(th, lN, lD)
        return -np.sum(norm.logpdf(y, mu, s) - norm.logcdf((ybar - mu) / s))

    r0 = y - est.lse_pred(init, lN, lD)
    x0 = np.r_[init, np.log(np.std(r0))]
    best = None
    for meth in ("L-BFGS-B", "Nelder-Mead"):
        r = minimize(nll, x0 if best is None else best.x, method=meth,
                     options=dict(maxiter=20000) if meth == "L-BFGS-B" else dict(maxiter=40000, xatol=1e-9, fatol=1e-10))
        if best is None or r.fun < best.fun:
            best = r
    m = sl.Chinchilla.from_theta(best.x[:5], objective=float(best.fun), s=float(np.exp(best.x[5])))
    return m


def gauss_mle_sigma(m, N, D, L):
    r = np.log(L) - est.lse_pred(theta_of(m), np.log(N), np.log(D))
    return float(np.sqrt(np.mean(r * r)))


# ----------------------------------------------------------------------------- main
def run(B=B_SEL, log=print):
    full = load_chinchilla(0)
    N, D, L = full.N.values, full.D.values, full.L.values
    M = D / N
    order = np.argsort(-L)  # worst first
    ref245 = fit_rule(N, D, L)
    ref240 = fit_rule(*load_chinchilla(5)[["N", "D", "L"]].values.T)

    # ---- (a) k worst dropped, k = 0..15: Huber, LAD, Gaussian-log point estimates; Huber pairs bootstrap SEs
    rows = []
    for k in range(K_MAX + 1):
        keep = np.setdiff1d(np.arange(len(L)), order[:k])
        Nk, Dk, Lk = N[keep], D[keep], L[keep]
        mh = fit_rule(Nk, Dk, Lk, inits=_cands([theta_of(ref245), theta_of(ref240)]))
        inits = _cands([theta_of(mh)])
        ml = est.fit_lad(Nk, Dk, Lk, inits=[theta_of(mh)])
        mg = est.fit_gauss_log(Nk, Dk, Lk, inits=inits + [theta_of(ref245), theta_of(ref240)])
        idxs = boot.draws(len(keep), B, "pairs", SEED + 101 + k)
        out = boot.pmap(boot_huber, idxs, dict(N=Nk, D=Dk, L=Lk, inits=[theta_of(mh)] + _cands()))
        s = boot.summarize(out, KEYS)
        # dropped_maxM: largest D/N among the k dropped runs. Note: two runs tie at L = 3.4059 (ranks 6 and 7), so which
        # of them is dropped at k = 6 depends on the sort order (np.argsort); k = 5 and k >= 7 are unaffected.
        r = dict(k=k, n=len(keep), Lmax_kept=float(Lk.max()), dropped_maxM=float(M[order[:k]].max()) if k else np.nan)
        for tag, m in (("huber", mh), ("lad", ml), ("gauss", mg)):
            for kk, v in derived(m).items():
                if kk in KEYS:
                    r[f"{tag}_{kk}"] = v
        for kk in KEYS:
            r[f"huber_se_{kk}"] = s[kk]["se"]
            r[f"huber_lo_{kk}"] = s[kk]["lo"]
            r[f"huber_hi_{kk}"] = s[kk]["hi"]
        rows.append(r)
        log(f"  selection k={k:2d} n={len(keep)} beta: huber {mh.beta:.4f} ({s['beta']['se']:.4f}) lad {ml.beta:.4f} "
            f"gauss {mg.beta:.4f}; alpha huber {mh.alpha:.4f}")
    kt = pd.DataFrame(rows)
    kt.to_csv(os.path.join(TABLES, "m1_chinchilla_selection_k.csv"), index=False)

    # ---- (b) paired bootstrap of the k=0 -> k=5 swing with a fixed truncation point Lbar (5th-worst loss)
    Lbar = float(np.sort(L)[-5])          # keep L < Lbar  <=>  drop the 5 worst in the original sample
    idxs = boot.draws(len(L), max(B, 400), "pairs", SEED + 7)
    out = boot.pmap(boot_huber, idxs, dict(N=N, D=D, L=L, inits=[theta_of(ref245), theta_of(ref240)] + _cands(), Lbar=Lbar))
    ok = [o for o in out if "_error" not in o]
    swing = []
    for kk in ("alpha", "beta", "a", "Mstar_1e+21", "E"):
        full_v, tr_v = derived(ref245)[kk], derived(ref240)[kk]
        dd = np.array([o[kk] - o[f"t_{kk}"] for o in ok])
        swing.append(dict(param=kk, est_all=full_v, est_trunc=tr_v, diff=full_v - tr_v, se_diff=float(np.std(dd, ddof=1)),
                          z=float((full_v - tr_v) / np.std(dd, ddof=1)), share_boot_diff_pos=float(np.mean(dd > 0)),
                          lo_diff=float(np.percentile(dd, 2.5)), hi_diff=float(np.percentile(dd, 97.5)), B=len(ok)))
    sw = pd.DataFrame(swing)
    sw.to_csv(os.path.join(TABLES, "m1_chinchilla_selection_swing.csv"), index=False)
    log("  swing (paired bootstrap):\n" + sw.round(4).to_string())

    # ---- (c) alternative exclusion rules (Huber-LSE; pairs bootstrap SEs)
    iso = full.iso.values
    r_full = fit_rule(N, D, L)
    res_abs = np.abs(np.log(L) - est.lse_pred(theta_of(r_full), np.log(N), np.log(D)))
    n_low = int(np.sum(L <= np.sort(L)[4]))      # ties at the 5th-lowest loss are dropped together
    rules = {
        "none (n=245)": np.ones(len(L), bool),
        "drop 5 highest loss (Besiroglu)": L < Lbar,
        f"drop the {n_low} lowest-loss runs (L <= 5th lowest; ties)": L > np.sort(L)[4],
        "drop 5 largest |residual| (n=245 fit)": res_abs < np.sort(res_abs)[-5],
        "drop D/N < 0.4 (selection on X)": M >= 0.4,
        "drop D/N < 1": M >= 1.0,
        "drop D/N < 2": M >= 2.0,
        "drop the 1e19 IsoFLOP budget": full.budget.values != 1e19,
        "IsoFLOP-profile runs only": iso,
        "IsoFLOP runs only, 5 highest loss dropped": iso & (L < Lbar),
    }
    alt = []
    for i, (lab, keep) in enumerate(rules.items()):
        Nk, Dk, Lk = N[keep], D[keep], L[keep]
        m = fit_rule(Nk, Dk, Lk, inits=_cands([theta_of(ref245), theta_of(ref240)]))
        idxs = boot.draws(int(keep.sum()), B, "pairs", SEED + 301 + i)
        out = boot.pmap(boot_huber, idxs, dict(N=Nk, D=Dk, L=Lk, inits=[theta_of(m)] + _cands()))
        s = boot.summarize(out, KEYS)
        r = dict(rule=lab, n=int(keep.sum()), n_dropped=int((~keep).sum()),
                 dropped_max_L=float(L[~keep].max()) if (~keep).any() else np.nan,
                 dropped_min_L=float(L[~keep].min()) if (~keep).any() else np.nan)
        r.update({kk: v for kk, v in derived(m).items() if kk in KEYS})
        r.update({f"se_{kk}": s[kk]["se"] for kk in KEYS})
        alt.append(r)
        log(f"  rule {lab:45s} n={keep.sum()} alpha {m.alpha:.4f} beta {m.beta:.4f} ({s['beta']['se']:.4f})")

    # ---- (d) Hausman-Wise truncated Gaussian MLE on the n=240 sample vs untruncated Gaussian fits
    s240 = load_chinchilla(5)
    g240 = est.fit_gauss_log(s240.N.values, s240.D.values, s240.L.values, inits=_cands([theta_of(ref240)]))
    g245 = est.fit_gauss_log(N, D, L, inits=_cands([theta_of(ref245)]))
    hw = fit_truncated_gauss(s240.N.values, s240.D.values, s240.L.values, Lbar, theta_of(g240))
    for lab, m, nn in (("Gaussian MLE, n=245", g245, 245), ("Gaussian MLE, n=240 (ignores truncation)", g240, 240),
                       ("Hausman-Wise truncated MLE, n=240 (L < Lbar)", hw, 240)):
        r = dict(rule=lab, n=nn, n_dropped=245 - nn)
        r.update({kk: v for kk, v in derived(m).items() if kk in KEYS})
        r["sigma_u"] = hw.extra["s"] if "Hausman" in lab else gauss_mle_sigma(m, N if nn == 245 else s240.N.values,
                                                                                 D if nn == 245 else s240.D.values,
                                                                                 L if nn == 245 else s240.L.values)
        alt.append(r)
    # truncation-probability diagnostics: P(L >= Lbar | N, D) under the truncated MLE for the kept runs
    mu = est.lse_pred(theta_of(hw), np.log(s240.N.values), np.log(s240.D.values))
    ptr = norm.sf((np.log(Lbar) - mu) / hw.extra["s"])
    alt_df = pd.DataFrame(alt)
    alt_df.to_csv(os.path.join(TABLES, "m1_chinchilla_selection_rules.csv"), index=False)
    # predicted vs actual for the dropped runs (model evaluated at the n=240 Huber fit)
    drop = full.loc[order[:8], ["N", "D", "C", "L", "budget"]].copy()
    drop["D_over_N"] = drop.D / drop.N
    drop["pred_n240"] = ref240.loss(drop.N, drop.D)
    drop["resid_log_n240"] = np.log(drop.L) - np.log(drop.pred_n240)
    drop.to_csv(os.path.join(TABLES, "m1_chinchilla_selection_worst_points.csv"), index=False)
    diag = dict(Lbar=Lbar, max_trunc_prob_kept=float(ptr.max()), n_kept_trunc_prob_gt_1pct=int((ptr > 0.01).sum()),
                hw_sigma=hw.extra["s"])
    log(f"  Hausman-Wise: beta {hw.beta:.4f} vs gauss240 {g240.beta:.4f} vs gauss245 {g245.beta:.4f}; {diag}")
    return kt, sw, alt_df, drop, diag
