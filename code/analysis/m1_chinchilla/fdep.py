"""Task 4 -- functional dependence (model_spec Proposition 1 / Lemma 2) on the Chinchilla extraction.

(a) On-path subsamples: runs within a narrow band around the fitted expansion path, |ln N - ln N*(C)| <= h, and the
    9 per-budget Approach-2 argmins (pseudo-observations). For each: Jacobian singular values of ln Lhat w.r.t.
    theta at the full-sample estimate, condition numbers, bootstrap SEs (explosion relative to the full design),
    and the profile objective over sigma* in the generalized model
        L = E + (A N^-a1 + B D^-b1)^kappa,        sigma* = 2/(2 + a1 + b1)
    (isoquants depend only on (a1, b1); on the path the data pin only a = b1/(a1+b1) and kappa*a1*b1/(a1+b1), so
    the profile in sigma* should be flat on-path and curved with the IsoFLOP (transverse) variation).
(b) Design-variance decomposition: centred (ln N, ln D) projected on the along-path direction (beta, alpha)/norm
    (the null vector of the Hessian of ln R) and on the transverse direction (alpha, -beta)/norm.
"""
from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import boot
import estimators as est
from common import BESI_PUB, HOFF_TEX, SEED, TABLES, derived, load_chinchilla, sl, theta_of

B_FD = 200
SIGMA_GRID = np.round(np.arange(0.50, 0.96, 0.01), 3)
BAND_H = (0.15, 0.10)
SMALL_GRID = dict(a=np.array([0.0, 7.5, 15.0]), b=np.array([0.0, 7.5, 15.0]), e=np.array([-1.0, 0.5]),
                  alpha=np.array([0.2, 0.6]), beta=np.array([0.2, 0.6]))


# ----------------------------------------------------------------------------- generalised (outer exponent) model
def gen_pred(p, lN, lD):
    """p = (lnA, lnB, lnE, a1, b1, kappa): ln Lhat = LSE(kappa * LSE(lnA - a1 lnN, lnB - b1 lnD), lnE)."""
    lnA, lnB, lnE, a1, b1, k = p
    inner = np.logaddexp(lnA - a1 * lN, lnB - b1 * lD)
    return np.logaddexp(k * inner, lnE)


def _obj_gen(p, lN, lD, lL, kind):
    r = gen_pred(p, lN, lD) - lL
    if kind == "gauss":
        return 0.5 * float(np.sum(r * r))
    return float(np.sum(sl._huber(1e-3, r)))  # "huber" = the reference objective


def _p_of(q, S):
    """Profile parameterisation for fixed S = a1 + b1: q = (lnA, lnB, lnE, share = a1/S, kappa)."""
    return np.array([q[0], q[1], q[2], S * q[3], S * (1 - q[3]), q[4]])


def fit_gen_fixedS(lN, lD, lL, S, starts, kind="gauss"):
    best = None
    bnds = [(-50, 80), (-50, 80), (-5, 3), (0.01, 0.99), (0.02, 20.0)]
    for q0 in starts:
        f = lambda q: _obj_gen(_p_of(q, S), lN, lD, lL, kind)
        r = minimize(f, np.asarray(q0, float), method="L-BFGS-B", bounds=bnds,
                     options=dict(maxiter=5000, ftol=1e-14, gtol=1e-10))
        if best is None or r.fun < best[1]:
            best = (r.x, r.fun)
    return best


def start_from_chin(m, S):
    """Map a kappa = 1 technology to a start at a1 + b1 = S preserving a and the on-path frontier exponent."""
    al, be = m.alpha, m.beta
    k = (al + be) / S
    return np.array([np.log(m.A) / k, np.log(m.B) / k, np.log(m.E), al / (al + be), k])


def profile_sigma(N, D, L, m_ref, kind="gauss", grid=SIGMA_GRID):
    lN, lD, lL = np.log(N), np.log(D), np.log(L)
    out, prev = [], None
    for s_ in grid:
        S = 2.0 / s_ - 2.0
        starts = [start_from_chin(m_ref, S)] + ([prev] if prev is not None else [])
        q, f = fit_gen_fixedS(lN, lD, lL, S, starts, kind)
        prev = q
        out.append(dict(sigma_star=s_, S=S, obj=f, kappa=q[4], a=1 - q[3], lnE=q[2], _q=q))
    # second sweep backwards (warm start from the right neighbour) to avoid path dependence of local optima
    for i in range(len(grid) - 2, -1, -1):
        S = out[i]["S"]
        q, f = fit_gen_fixedS(lN, lD, lL, S, [out[i + 1]["_q"]], kind)
        if f < out[i]["obj"] - 1e-12:
            out[i].update(obj=f, kappa=q[4], a=1 - q[3], lnE=q[2], _q=q)
    df = pd.DataFrame(out).drop(columns="_q")
    n = len(L)
    if kind == "gauss":
        df["LR"] = n * np.log(df.obj / df.obj.min())      # Gaussian profile LR, chi2(1) under H0
    else:
        df["LR"] = np.nan
    df["dobj"] = df.obj - df.obj.min()
    return df


def fit_gen_free(N, D, L, m_ref, kind="gauss"):
    """Unrestricted generalised model (kappa free): best over a sigma* grid, then polished jointly."""
    prof = profile_sigma(N, D, L, m_ref, kind=kind, grid=np.round(np.arange(0.5, 0.96, 0.05), 3))
    i = int(prof.obj.idxmin())
    S = prof.loc[i, "S"]
    q, f = fit_gen_fixedS(np.log(N), np.log(D), np.log(L), S, [start_from_chin(m_ref, S)], kind)
    p0 = _p_of(q, S)
    lN, lD, lL = np.log(N), np.log(D), np.log(L)
    r = minimize(lambda p: _obj_gen(p, lN, lD, lL, kind), p0, method="L-BFGS-B",
                 bounds=[(-50, 80), (-50, 80), (-5, 3), (0.005, 5), (0.005, 5), (0.02, 20)],
                 options=dict(maxiter=10000, ftol=1e-15, gtol=1e-11))
    p = r.x if r.fun <= f else p0
    return p, min(r.fun, f)


# ----------------------------------------------------------------------------- design geometry
def variance_decomp(N, D, alpha, beta):
    n, d = np.log(N), np.log(D)
    X = np.column_stack([n - n.mean(), d - d.mean()])
    u = np.array([beta, alpha]) / np.hypot(alpha, beta)      # along the expansion path (Hessian null vector)
    t = np.array([alpha, -beta]) / np.hypot(alpha, beta)     # transverse
    va, vt = float(np.var(X @ u)), float(np.var(X @ t))
    # within-budget transverse variation (C fixed): what IsoFLOP designs add
    c = n + d
    tt = X @ t
    Xc = np.column_stack([np.ones_like(c), c])
    res = tt - Xc @ np.linalg.lstsq(Xc, tt, rcond=None)[0]
    return dict(n=len(N), var_along=va, var_trans=vt, share_trans=vt / (va + vt), var_trans_given_c=float(np.var(res)),
                corr_nd=float(np.corrcoef(n, d)[0, 1]))


# ----------------------------------------------------------------------------- bootstrap of on-path fits
def _boot_sub(idx, N, D, L, inits, **_):
    m = est.fit_huber(N[idx], D[idx], L[idx], inits=inits)
    d = derived(m)
    d["obj"] = m.extra["objective"]
    return d


def run(B=B_FD, log=print):
    full = load_chinchilla(5)
    N, D, L = full.N.values, full.D.values, full.L.values
    ref = est.fit_huber(N, D, L, inits=[theta_of(BESI_PUB)])
    ref = est.fit_huber(N, D, L, grid=sl.FAST_GRID) if ref.extra["objective"] > 1.0183e-3 else ref
    th = theta_of(ref)
    C = 6 * N * D
    dev = np.log(N) - np.log(ref.N_opt(C))
    samples = {"full (n=240)": np.ones(len(L), bool)}
    for h in BAND_H:
        samples[f"on-path band |dlnN|<={h}"] = np.abs(dev) <= h
    # A2 argmins as pseudo-observations
    import duality
    iso = full.iso.values
    mins = duality.a2_minima(np.log(N[iso]), L[iso], full.budget.values[iso], np.log(full.C.values[iso]))
    Na = np.exp(mins[:, 1])
    Da = np.exp(mins[:, 0]) / (6 * Na)
    La = mins[:, 2]

    Nbar, Dbar = float(np.exp(np.mean(np.log(N)))), float(np.exp(np.mean(np.log(D))))
    grid_starts = [np.array(s) for s in itertools.product(*SMALL_GRID.values())]
    inits_boot = [th, theta_of(HOFF_TEX), theta_of(BESI_PUB)] + grid_starts
    rows, vd, profiles = [], [], {}
    keys = ["E", "alpha", "beta", "a", "gamma", "sigma_star", "Mstar_1e+21", "Mstar_5.76e+23", "lnG"]
    for lab, s in list(samples.items()) + [("A2 argmins (9 pseudo-obs.)", None)]:
        if s is None:
            Ns, Ds, Ls = Na, Da, La
        else:
            Ns, Ds, Ls = N[s], D[s], L[s]
        J = est.jac_log(th, Ns, Ds)                      # at the full-sample estimate (same parameter point)
        cs = est.cond_stats(J)
        # same Jacobian in the KMW-normalised parameterisation (normalisation point = full-sample geometric means):
        # removes the artificial ln A - alpha / ln B - beta ridge, isolating the design-driven rank deficiency
        csn = est.cond_stats(est.jac_log_norm(est.to_norm(th, Nbar, Dbar), Ns, Ds, Nbar, Dbar))
        m = est.fit_huber(Ns, Ds, Ls, inits=inits_boot)  # subsample estimate
        cs_own = est.cond_stats(est.jac_log(theta_of(m), Ns, Ds))
        idxs = boot.draws(len(Ls), B, "pairs", SEED + 900 + len(rows))
        out = boot.pmap(_boot_sub, idxs, dict(N=Ns, D=Ds, L=Ls, inits=inits_boot))
        sm = boot.summarize(out, keys)
        ok = [o for o in out if "_error" not in o]
        wild = float(np.mean([(o["alpha"] > 1.5) or (o["beta"] > 1.5) or (o["alpha"] < 0.05) or (o["beta"] < 0.05) for o in ok]))
        r = dict(sample=lab, n=len(Ls), **{f"sv{i + 1}": v / np.sqrt(len(Ls)) for i, v in enumerate(cs["sv"])},
                 cond=cs["cond"], cond_scaled=cs["cond_scaled"],
                 **{f"svs{i + 1}": v for i, v in enumerate(cs["sv_scaled"])},
                 cond_own=cs_own["cond"], cond_scaled_own=cs_own["cond_scaled"], share_wild=wild,
                 **{f"svn{i + 1}": v / np.sqrt(len(Ls)) for i, v in enumerate(csn["sv"])}, cond_norm=csn["cond"])
        r.update({k: v for k, v in derived(m).items() if k in keys})
        r.update({f"se_{k}": sm[k]["se"] for k in keys})
        r.update({f"lo_{k}": sm[k]["lo"] for k in keys})
        r.update({f"hi_{k}": sm[k]["hi"] for k in keys})
        rows.append(r)
        vd.append(dict(sample=lab, **variance_decomp(Ns, Ds, ref.alpha, ref.beta)))
        log(f"  fdep {lab:32s} n={len(Ls):3d} cond_scaled={cs['cond_scaled']:.3g} alpha {m.alpha:.3f} ({sm['alpha']['se']:.3f}) "
            f"beta {m.beta:.3f} ({sm['beta']['se']:.3f}) a {m.a_N:.3f} ({sm['a']['se']:.3f}) wild={wild:.2f}")
        if s is not None and lab.endswith(str(BAND_H[0])) or lab.startswith("full"):
            for kind in ("gauss", "huber"):
                profiles[(lab, kind)] = profile_sigma(Ns, Ds, Ls, ref, kind=kind)
    # other designs for the variance decomposition
    full245 = load_chinchilla(0)
    vd.append(dict(sample="full (n=245)", **variance_decomp(full245.N.values, full245.D.values, ref.alpha, ref.beta)))
    vd.append(dict(sample="IsoFLOP-profile runs (n=240 sample)", **variance_decomp(N[iso], D[iso], ref.alpha, ref.beta)))
    vd.append(dict(sample="off-profile runs (n=240 sample)", **variance_decomp(N[~iso], D[~iso], ref.alpha, ref.beta)))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(TABLES, "m1_chinchilla_fdep.csv"), index=False)
    vdd = pd.DataFrame(vd)
    vdd.to_csv(os.path.join(TABLES, "m1_chinchilla_design_variance.csv"), index=False)
    prof = pd.concat([p.assign(sample=k[0], objective=k[1]) for k, p in profiles.items()], ignore_index=True)
    prof.to_csv(os.path.join(TABLES, "m1_chinchilla_profile_sigma.csv"), index=False)
    return tab, vdd, prof, ref
