"""Task 3 -- duality: Chinchilla Approaches 1/2/3 as cost function / conditional factor demand / primal.

  A2 (conditional factor demand): per IsoFLOP budget b, fit L = c0 + c1 x + c2 x^2 with x = ln N (Hoffmann's
     parabola), N*_b = exp(-c1/(2 c2)), D*_b = C_b/(6 N*_b), L*_b = c0 - c1^2/(4 c2); then OLS
     ln N*_b = ln G + a ln(C_b/6)  (so ln D*_b = -ln G + (1-a) ln(C_b/6) identically).
  A1 (Nerlove inverse cost function / lower envelope): NLS of L*_b = E + K (C_b/6)^-gamma on the per-budget minima.
  A3 (primal): Huber-LSE on all runs.

Cross-equation restrictions implied by the primal (model_spec, Lemma 1): a = beta/(alpha+beta);
gamma = alpha beta/(alpha+beta); the path level ln N*(C) = ln G + a ln(C/6) and the frontier level L*(C).
Two versions of the null are tested:
  analytic  -- the A2/A1 objects equal their closed forms under theta_A3;
  design    -- the A2/A1 objects equal what the same parabola/envelope procedure returns when applied to the A3
               fitted values at the observed design (indirect-inference logic; removes the known finite-grid bias of
               Approach 2, Czech et al. 2026).
Wald statistics use the bootstrap covariance of the difference vector (joint, within-budget stratified bootstrap).

System estimator (Leon-Ledesma, McAdam and Willman 2010 logic): loss equation (all runs) + argmin equation
(ln N*_b, b = 1..B) with cross-equation restrictions, Gaussian concentrated likelihood
  Q(theta) = (n1/2) ln(SSR_loss/n1) + (n2/2) ln(SSR_argmin/n2),
with the argmin equation evaluated in its design-consistent form. LR test of the 2 restrictions against the
unrestricted pair (A3 Gaussian NLS, free OLS path).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize
from scipy.stats import chi2, spearmanr
from scipy.stats import t as student_t

import boot
import estimators as est
from common import (BESI_PUB, CHIN70B, HOFF_ROUND, HOFF_TEX, SEED, TABLES, derived, load_chinchilla,
                    load_isoflop, sl, theta_of)

B_DUAL = 300
OBJ_NAMES = ["a", "lnNstar_ref", "gamma", "Lstar_ref"]


# ----------------------------------------------------------------------------- Approach 2 and 1
def a2_minima(lnN, L, lab, lnC, return_labels=False):
    """Per-budget parabola in ln N. Returns (lnC_b, lnN*_b, L*_b) for budgets with >= 3 distinct model sizes and a
    convex parabola (budgets failing either condition are dropped and counted by the caller)."""
    out, used = [], []
    for b in np.unique(lab):
        s = lab == b
        if len(np.unique(np.round(lnN[s], 8))) < 3:
            continue
        c2, c1, c0 = np.polyfit(lnN[s], L[s], 2)
        if c2 <= 0:
            continue
        out.append((np.mean(lnC[s]), -c1 / (2 * c2), c0 - c1 * c1 / (4 * c2)))
        used.append(b)
    out = np.array(out).reshape(-1, 3)
    return (out, np.array(used)) if return_labels else out


def a1_frontier(lnC_b, Lstar, E_fixed=None, x0=None):
    """NLS of L*_b = E + K (C_b/6)^-gamma (levels). Parameterised as (E, ln K, gamma)."""
    x6 = lnC_b - np.log(6)
    if E_fixed is not None:
        f = lambda p: E_fixed + np.exp(p[0] - p[1] * x6) - Lstar
        r = least_squares(f, x0=[np.log(2000.0), 0.18] if x0 is None else x0[1:], bounds=([-50, 0.005], [80, 1.5]))
        return np.array([E_fixed, r.x[0], r.x[1]])
    f = lambda p: p[0] + np.exp(p[1] - p[2] * x6) - Lstar
    ub = 0.999 * Lstar.min()
    x0 = [min(1.8, 0.9 * ub), np.log(2000.0), 0.18] if x0 is None else [min(x0[0], 0.99 * ub), x0[1], x0[2]]
    best = None
    for st in (x0, [0.5 * ub, x0[1], x0[2]], [0.1, 3.0, 0.1]):
        try:
            r = least_squares(f, x0=st, bounds=([0.0, -50, 0.005], [ub, 80, 1.5]))
        except ValueError:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best.x


def a2_objects(lnN, L, lab, lnC, lnC_ref, E_fixed=None, a1_x0=None):
    mins = a2_minima(lnN, L, lab, lnC)
    x6 = mins[:, 0] - np.log(6)
    X = np.column_stack([np.ones_like(x6), x6])
    (lnG, a), *_ = np.linalg.lstsq(X, mins[:, 1], rcond=None)
    E, lnK, g = a1_frontier(mins[:, 0], mins[:, 2], E_fixed=E_fixed, x0=a1_x0)
    xr = lnC_ref - np.log(6)
    return dict(a=a, lnG=lnG, lnNstar_ref=lnG + a * xr, E_A1=E, lnK=lnK, gamma=g, Lstar_ref=E + np.exp(lnK - g * xr),
                nb=len(mins), mins=mins, resid_path=mins[:, 1] - X @ np.array([lnG, a]))


def analytic_objects(m, lnC_ref):
    xr = lnC_ref - np.log(6)
    return dict(a=m.a_N, lnNstar_ref=np.log(m.G) + m.a_N * xr, gamma=m.gamma, Lstar_ref=m.E + m.K * np.exp(-m.gamma * xr))


def design_objects(m, N, D, lab, lnC, lnC_ref, E_fixed=None):
    """A2/A1 applied to the noise-free predictions of technology m at the observed design."""
    return a2_objects(np.log(N), m.loss(N, D), lab, lnC, lnC_ref, E_fixed=E_fixed,
                      a1_x0=[m.E, np.log(m.K), m.gamma])


def vec(d):
    return np.array([d[k] for k in OBJ_NAMES], float)


# ----------------------------------------------------------------------------- system estimator
class _ParabolaDesign:
    """Pre-computed per-budget least-squares projectors for the parabola in ln N (fast repeated A2 evaluation)."""

    def __init__(self, lnN, lab, lnC):
        self.blocks = []
        for b in np.unique(lab):
            s = np.where(lab == b)[0]
            if len(np.unique(np.round(lnN[s], 8))) < 3:
                continue
            X = np.column_stack([lnN[s] ** 2, lnN[s], np.ones(len(s))])
            self.blocks.append((s, np.linalg.pinv(X), float(np.mean(lnC[s]))))

    def minima(self, L):
        out = []
        for s, P, lc in self.blocks:
            c2, c1, c0 = P @ L[s]
            if c2 <= 0:
                return None
            out.append((lc, -c1 / (2 * c2), c0 - c1 * c1 / (4 * c2)))
        return np.array(out)


def system_fit(N, D, L, Ni, Di, Li, labi, lnCi, x0, return_parts=False, lik="laplace"):
    """Concentrated quasi-likelihood system: loss equation on all runs + argmin equation on IsoFLOP minima.
    lik = "laplace": Laplace errors in the loss equation (the LAD/Huber(1e-3) reference objective),
          -> Q = n1 ln mean|r1| + (n2/2) ln mean r2^2;
    lik = "gauss":   Gaussian errors in both -> Q = (n1/2) ln mean r1^2 + (n2/2) ln mean r2^2.
    (N, D, L): all runs; (Ni, Di, Li, labi, lnCi): IsoFLOP-profile runs with their budget labels."""
    lN, lD, y = np.log(N), np.log(D), np.log(L)
    obs, used = a2_minima(np.log(Ni), Li, labi, lnCi, return_labels=True)
    keep = np.isin(labi, used)                    # budgets whose observed parabola is usable
    Ni, Di, labi, lnCi = Ni[keep], Di[keep], labi[keep], lnCi[keep]
    n1, n2 = len(y), len(obs)
    design = _ParabolaDesign(np.log(Ni), labi, lnCi)
    lNi, lDi = np.log(Ni), np.log(Di)

    def parts(th):
        r1 = y - est.lse_pred(th, lN, lD)
        pred = design.minima(np.exp(est.lse_pred(th, lNi, lDi)))
        if pred is None or len(pred) != n2:
            return None
        return r1, obs[:, 1] - pred[:, 1]

    def Q(th):
        p = parts(th)
        if p is None or not np.all(np.isfinite(th)):
            return 1e10
        r1, r2 = p
        q1 = n1 * np.log(np.mean(np.abs(r1))) if lik == "laplace" else 0.5 * n1 * np.log(np.mean(r1 * r1))
        return q1 + 0.5 * n2 * np.log(np.mean(r2 * r2))

    best = None
    for st in x0:
        r = minimize(Q, np.asarray(st, float), method="Nelder-Mead",
                     options=dict(maxiter=8000, maxfev=12000, xatol=1e-8, fatol=1e-10, adaptive=True))
        r2 = minimize(Q, r.x, method="L-BFGS-B")
        r = r2 if r2.fun <= r.fun else r
        if best is None or r.fun < best.fun:
            best = r
    m = sl.Chinchilla.from_theta(best.x, objective=float(best.fun))
    if return_parts:
        r1, r2 = parts(best.x)
        return m, r1, r2, obs
    return m


def system_fit_data(N, D, L, iso_mask, lab, lnC, x0, return_parts=False, lik="laplace"):
    return system_fit(N, D, L, N[iso_mask], D[iso_mask], L[iso_mask], lab[iso_mask], lnC[iso_mask], x0,
                      return_parts=return_parts, lik=lik)


# ----------------------------------------------------------------------------- one replication
def _rep(idx, N, D, L, iso, lab, lnC, lnC_ref, inits, lit, do_system, **_):
    Nb, Db, Lb, isob, labb, lnCb = N[idx], D[idx], L[idx], iso[idx], lab[idx], lnC[idx]
    m3 = est.fit_huber(Nb, Db, Lb, inits=inits)
    o = a2_objects(np.log(Nb[isob]), Lb[isob], labb[isob], lnCb[isob], lnC_ref,
                   a1_x0=[m3.E, np.log(m3.K), m3.gamma])
    out = {f"A2_{k}": o[k] for k in OBJ_NAMES + ["lnG", "E_A1", "lnK"]}
    out.update({f"A3an_{k}": v for k, v in analytic_objects(m3, lnC_ref).items()})
    dz = design_objects(m3, Nb[isob], Db[isob], labb[isob], lnCb[isob], lnC_ref)
    out.update({f"A3de_{k}": dz[k] for k in OBJ_NAMES})
    for tag, th in lit.items():
        ml = sl.Chinchilla.from_theta(th)
        dz = design_objects(ml, Nb[isob], Db[isob], labb[isob], lnCb[isob], lnC_ref)
        out.update({f"{tag}de_{k}": dz[k] for k in OBJ_NAMES})
    out.update({f"A3_{k}": v for k, v in derived(m3).items()})
    if do_system:
        ms = system_fit_data(Nb, Db, Lb, isob, labb, lnCb, x0=[theta_of(m3)] + list(inits[:1]), lik="laplace")
        out.update({f"SYS_{k}": v for k, v in derived(ms).items()})
        mg = system_fit_data(Nb, Db, Lb, isob, labb, lnCb, x0=[theta_of(m3)] + list(inits[:1]), lik="gauss")
        out.update({f"SYSG_{k}": v for k, v in derived(mg).items()})
    return out


def save_minima(o2s, fname):
    rows = []
    for k, o2 in o2s.items():
        for lc, ln, ls in o2["mins"]:
            rows.append(dict(dataset=k, lnC=lc, C=np.exp(lc), lnNstar=ln, Nstar=np.exp(ln), Dstar=np.exp(lc) / (6 * np.exp(ln)),
                             Lstar=ls, a_A2=o2["a"], lnG_A2=o2["lnG"], E_A1=o2["E_A1"], lnK_A1=o2["lnK"], gamma_A1=o2["gamma"]))
    pd.DataFrame(rows).to_csv(os.path.join(TABLES, fname), index=False)


def robust_cov(X):
    """Outlier-robust covariance of bootstrap draws: IQR-based scales and Spearman correlations mapped to Pearson
    (2 sin(pi rho/6)); projected to the PSD cone. Used because parabola argmins in resampled designs are heavy-tailed."""
    X = np.asarray(X, float)
    X = X[np.all(np.isfinite(X), axis=1)]
    sc = boot.rse(X)
    rho = spearmanr(X).correlation
    rho = np.array([[1.0, rho], [rho, 1.0]]) if np.ndim(rho) == 0 else np.asarray(rho)
    R = 2 * np.sin(np.pi * rho / 6)
    np.fill_diagonal(R, 1.0)
    V = np.outer(sc, sc) * R
    w, U = np.linalg.eigh(V)
    return (U * np.maximum(w, 1e-12 * max(w.max(), 1e-300))) @ U.T


def wald(delta, V):
    V = np.atleast_2d(V)
    W = float(delta @ np.linalg.pinv(V) @ delta)
    return W, float(chi2.sf(W, len(delta)))


B_WILD = 400


def _rep_wild(v, N, D, lnL_hat, resid, iso, lab, lnC, lnC_ref, th0, **_):
    """Fixed-design wild bootstrap (Rademacher weights v on the Approach-3 log residuals): the IsoFLOP design is set
    by the experimenter, so the regressors are held fixed and only the loss is redrawn."""
    Ls = np.exp(lnL_hat + v * resid)
    m = est.fit_huber(N, D, Ls, inits=[th0])
    o = a2_objects(np.log(N[iso]), Ls[iso], lab[iso], lnC[iso], lnC_ref, a1_x0=[m.E, np.log(m.K), m.gamma])
    return dict(A2=vec(o), an=vec(analytic_objects(m, lnC_ref)),
                de=vec(design_objects(m, N[iso], D[iso], lab[iso], lnC[iso], lnC_ref)), alpha=m.alpha, beta=m.beta)


def duality_for(df, label, lnC_ref, B, seed, lit=None, do_system=True, log=print, grid=sl.DEFAULT_GRID):
    """Full duality analysis for one dataset (df with N, D, L, C, budget, iso)."""
    N, D, L = df.N.values, df.D.values, df.L.values
    iso = df.iso.values.astype(bool)
    lab = df.budget.values
    lnC = np.log(df.C.values)
    lit = lit or {}
    m3 = est.fit_huber(N, D, L, grid=grid)
    inits = [theta_of(m3)] + [np.asarray(t) for t in lit.values()] + [theta_of(BESI_PUB)]
    o2 = a2_objects(np.log(N[iso]), L[iso], lab[iso], lnC[iso], lnC_ref, a1_x0=[m3.E, np.log(m3.K), m3.gamma])
    o2E = a2_objects(np.log(N[iso]), L[iso], lab[iso], lnC[iso], lnC_ref, E_fixed=m3.E,
                     a1_x0=[m3.E, np.log(m3.K), m3.gamma])
    an = analytic_objects(m3, lnC_ref)
    de = design_objects(m3, N[iso], D[iso], lab[iso], lnC[iso], lnC_ref)
    # bootstrap: within-budget stratified (off-budget runs form their own stratum)
    strata = np.where(iso, lab, -1.0)
    idxs = boot.draws(boot.groups_of(strata), B, "strat", seed)
    reps = boot.pmap(_rep, idxs, dict(N=N, D=D, L=L, iso=iso, lab=lab, lnC=lnC, lnC_ref=lnC_ref, inits=inits,
                                      lit=lit, do_system=do_system))
    ok = [r for r in reps if "_error" not in r]
    get = lambda pre: np.array([[r[f"{pre}_{k}"] for k in OBJ_NAMES] for r in ok])
    A2b, A3an, A3de = get("A2"), get("A3an"), get("A3de")
    tests = []
    pt2 = vec(o2)
    # classical OLS standard error of the Approach-2 path slope (9 argmins, nb-2 df): the textbook benchmark
    mins = o2["mins"]
    x6 = mins[:, 0] - np.log(6)
    s2 = float(np.sum(o2["resid_path"] ** 2) / (len(x6) - 2))
    se_a_ols = float(np.sqrt(s2 / np.sum((x6 - x6.mean()) ** 2)))
    # fixed-design wild bootstrap (regressors fixed, loss redrawn with Rademacher weights on the A3 log residuals)
    lnLhat = est.lse_pred(theta_of(m3), np.log(N), np.log(D))
    resid = np.log(L) - lnLhat
    rng = np.random.default_rng(seed + 1)
    vs = [rng.choice([-1.0, 1.0], size=len(L)) for _ in range(B_WILD)]
    wr = boot.pmap(_rep_wild, vs, dict(N=N, D=D, lnL_hat=lnLhat, resid=resid, iso=iso, lab=lab, lnC=lnC,
                                        lnC_ref=lnC_ref, th0=theta_of(m3)))
    wok = [w for w in wr if "_error" not in w]
    WA2 = np.array([w["A2"] for w in wok])
    Wan, Wde = np.array([w["an"] for w in wok]), np.array([w["de"] for w in wok])
    # null objects and the bootstrap draws of the difference vector, for each bootstrap scheme
    nulls = {"stratified pairs": [("A3 (this fit), analytic", vec(an), A2b - A3an),
                                  ("A3 (this fit), design-consistent", vec(de), A2b - A3de)],
             "wild, fixed design": [("A3 (this fit), analytic", vec(an), WA2 - Wan),
                                    ("A3 (this fit), design-consistent", vec(de), WA2 - Wde)]}
    litde = {}
    for tag, th in lit.items():
        ml = sl.Chinchilla.from_theta(th)
        pa = vec(analytic_objects(ml, lnC_ref))
        pd_ = vec(design_objects(ml, N[iso], D[iso], lab[iso], lnC[iso], lnC_ref))
        litde[tag] = get(f"{tag}de")
        nulls["stratified pairs"] += [(f"{tag} (fixed), analytic", pa, A2b), (f"{tag} (fixed), design", pd_, A2b - litde[tag])]
        nulls["wild, fixed design"] += [(f"{tag} (fixed), analytic", pa, WA2), (f"{tag} (fixed), design", pd_, WA2)]
    for bs, lst in nulls.items():
        for nm, pt3, Dd in lst:
            dl = pt2 - pt3
            for ck, V in (("robust", robust_cov(Dd)), ("sd", np.cov(Dd.T))):
                W, p = wald(dl, V)
                W2, p2 = wald(dl[:2], V[:2, :2])
                W3, p3 = wald(dl[2:], V[2:, 2:])
                tests.append(dict(dataset=label, null=nm, bootstrap=bs, cov=ck, W_all=W, df_all=4, p_all=p, W_path=W2,
                                  p_path=p2, W_front=W3, p_front=p3, **{f"d_{k}": v for k, v in zip(OBJ_NAMES, dl)},
                                  **{f"se_{k}": v for k, v in zip(OBJ_NAMES, np.sqrt(np.diag(V)))},
                                  a_A2=pt2[0], a_null=pt3[0], se_a_ols=se_a_ols, t_a_ols=(pt2[0] - pt3[0]) / se_a_ols,
                                  p_a_ols=float(2 * student_t.sf(abs((pt2[0] - pt3[0]) / se_a_ols), len(x6) - 2))))
    import re
    tag_file = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    from common import PROC
    np.savez(os.path.join(PROC, f"duality_draws_{tag_file}.npz"), A2b=A2b, A3an=A3an, A3de=A3de, WA2=WA2, Wan=Wan,
             Wde=Wde, pt2=pt2, an=vec(an), de=vec(de), **{f"{t}de": v for t, v in litde.items()})
    tests = pd.DataFrame(tests)
    # summary of objects
    se = lambda arr: np.std(arr, axis=0, ddof=1)
    objs = pd.DataFrame({
        "object": OBJ_NAMES,
        "A2_A1": pt2, "A2_A1_se": se(A2b), "A2_A1_rse": boot.rse(A2b), "A2_A1_se_wild": se(WA2),
        "A2_A1_rse_wild": boot.rse(WA2), "A2_se_ols": [se_a_ols, np.nan, np.nan, np.nan],
        "A2_A1_Efixed": vec(o2E),
        "A3_analytic": vec(an), "A3_analytic_se": se(A3an), "A3_analytic_rse": boot.rse(A3an),
        "A3_analytic_se_wild": se(Wan), "A3_design_se_wild": se(Wde),
        "A3_design": vec(de), "A3_design_se": se(A3de), "A3_design_rse": boot.rse(A3de),
    })
    objs["dataset"] = label
    extra = dict(B_wild=len(wok), wild_se_alpha=float(np.std([w["alpha"] for w in wok], ddof=1)),
                 wild_se_beta=float(np.std([w["beta"] for w in wok], ddof=1)), A1_E=o2["E_A1"], A1_E_se=float(np.std([r["A2_E_A1"] for r in ok], ddof=1)), A2_lnG=o2["lnG"],
                 A2_lnG_se=float(np.std([r["A2_lnG"] for r in ok], ddof=1)), nb=o2["nb"], B=len(ok),
                 nfail=len(reps) - len(ok), A2_path_resid_sd=float(np.std(o2["resid_path"], ddof=2)),
                 lnC_ref=lnC_ref, mins=o2["mins"])
    sysres = None
    if do_system:
        sysres = {}
        KS = ["E", "A", "B", "alpha", "beta", "a", "gamma", "sigma_star", "Mstar_1e+21", "Mstar_5.76e+23",
              "Mstar_1e+26", "w_chin70b"]
        for lik, pre, unres in (("laplace", "SYS", est.fit_lad), ("gauss", "SYSG", est.fit_gauss_log)):
            ms, r1, r2, obs = system_fit_data(N, D, L, iso, lab, lnC, x0=[theta_of(m3)] + inits[1:2],
                                              return_parts=True, lik=lik)
            # unrestricted comparison: loss equation alone (LAD or Gaussian NLS) + free OLS path (2 extra parameters)
            mu = unres(N, D, L, inits=inits)
            r1u = np.log(L) - est.lse_pred(theta_of(mu), np.log(N), np.log(D))
            x6 = obs[:, 0] - np.log(6)
            X = np.column_stack([np.ones_like(x6), x6])
            r2u = obs[:, 1] - X @ np.linalg.lstsq(X, obs[:, 1], rcond=None)[0]
            n1, n2 = len(r1), len(r2)
            if lik == "laplace":
                LR = 2 * n1 * np.log(np.mean(np.abs(r1)) / np.mean(np.abs(r1u)))
            else:
                LR = n1 * np.log(np.mean(r1 ** 2) / np.mean(r1u ** 2))
            LR += n2 * np.log(np.mean(r2 ** 2) / np.mean(r2u ** 2))
            sb = {k: np.array([r[f"{pre}_{k}"] for r in ok]) for k in KS}
            sysres[lik] = dict(model=ms, derived=derived(ms), se={k: float(np.std(v, ddof=1)) for k, v in sb.items()},
                               ci={k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in sb.items()},
                               LR=float(LR), LR_p=float(chi2.sf(max(LR, 0.0), 2)), unrestricted=derived(mu), n1=n1, n2=n2,
                               sd_loss=float(np.sqrt(np.mean(r1 ** 2))), sd_argmin=float(np.sqrt(np.mean(r2 ** 2))),
                               sd_argmin_unres=float(np.sqrt(np.mean(r2u ** 2))))
    a3se = {k: float(np.std([r[f"A3_{k}"] for r in ok], ddof=1)) for k in ["alpha", "beta", "a", "gamma", "w_chin70b", "E", "A", "B",
                                                                          "sigma_star", "Mstar_1e+21", "Mstar_5.76e+23", "Mstar_1e+26"]}
    a3ci = {k: (float(np.percentile([r[f"A3_{k}"] for r in ok], 2.5)), float(np.percentile([r[f"A3_{k}"] for r in ok], 97.5)))
            for k in ["w_chin70b", "a", "Mstar_1e+21", "Mstar_5.76e+23", "Mstar_1e+26"]}
    log(f"  duality {label}: A2 a={o2['a']:.4f} (se_lnG {extra['A2_lnG_se']:.3f}) vs A3 a={m3.a_N:.4f}; A1 gamma={o2['gamma']:.4f} "
        f"vs A3 gamma={m3.gamma:.4f}; B={len(ok)} fails={extra['nfail']}")
    return dict(m3=m3, a3se=a3se, a3ci=a3ci, objs=objs, tests=tests, extra=extra, system=sysres, o2=o2,
                A2_draws=A2b, reps=ok)


def run(B=B_DUAL, log=print):
    out = {}
    lit = {"Hoffmann": theta_of(HOFF_TEX), "Besiroglu": theta_of(sl.BESIROGLU)}
    for k in (5, 0):
        df = load_chinchilla(k)
        lnC_ref = float(np.mean(np.log(df.loc[df.iso, "C"])))
        out[f"chinchilla_n{len(df)}"] = duality_for(df, f"Chinchilla n={len(df)}", lnC_ref, B, SEED + 500 + k, lit=lit,
                                                    do_system=True, log=log)
    # revealed-preference test at Chinchilla-70B (w = eps_N/eps_D must be 1 if the chosen model is cost-minimising)
    rp = []
    for lab, m in (("Hoffmann A3, rounded (literature)", HOFF_ROUND), ("Hoffmann A3, TeX precision (literature)", HOFF_TEX),
                   ("Besiroglu et al. published (literature)", sl.BESIROGLU)):
        rp.append(dict(technology=lab, w=float(m.wedge(*CHIN70B)), se=np.nan, lo=np.nan, hi=np.nan,
                       Mstar_at_chin70b_C=float(m.D_opt(6 * CHIN70B[0] * CHIN70B[1]) / m.N_opt(6 * CHIN70B[0] * CHIN70B[1]))))
    for key in ("chinchilla_n240", "chinchilla_n245"):
        r = out[key]
        m3 = r["m3"]
        C70 = 6 * CHIN70B[0] * CHIN70B[1]
        rp.append(dict(technology=f"Huber-LSE refit, {key.split('_')[1]} (this module)", w=float(m3.wedge(*CHIN70B)),
                       se=r["a3se"]["w_chin70b"], lo=r["a3ci"]["w_chin70b"][0], hi=r["a3ci"]["w_chin70b"][1],
                       Mstar_at_chin70b_C=float(m3.D_opt(C70) / m3.N_opt(C70))))
        for lik in ("laplace", "gauss"):
            s = r["system"][lik]
            rp.append(dict(technology=f"System (loss + argmin, {lik}), {key.split('_')[1]}", w=s["derived"]["w_chin70b"],
                           se=s["se"]["w_chin70b"], lo=s["ci"]["w_chin70b"][0], hi=s["ci"]["w_chin70b"][1],
                           Mstar_at_chin70b_C=float(s["model"].D_opt(C70) / s["model"].N_opt(C70))))
        # where does Chinchilla-70B sit relative to the A2 path extrapolated to its compute?
        o2 = r["o2"]
        rp.append(dict(technology=f"A2 path (parabola argmins), {key.split('_')[1]}: ln N_70B - ln N*_A2(C_70B)",
                       w=np.nan, se=np.nan, lo=np.nan, hi=np.nan,
                       Mstar_at_chin70b_C=float(np.exp(-2 * (o2["lnG"] + o2["a"] * np.log(C70 / 6)) + np.log(C70 / 6))),
                       lnN_gap=float(np.log(CHIN70B[0]) - (o2["lnG"] + o2["a"] * np.log(C70 / 6)))))
    rp = pd.DataFrame(rp)
    rp.to_csv(os.path.join(TABLES, "m1_chinchilla_revealed_pref_70b.csv"), index=False)
    tests = pd.concat([out[k]["tests"] for k in out], ignore_index=True)
    tests.to_csv(os.path.join(TABLES, "m1_chinchilla_duality_tests.csv"), index=False)
    save_minima({k: out[k]["o2"] for k in out}, "m1_chinchilla_a2_minima.csv")
    # bootstrap draws of the system estimators (theta = lnA, lnB, lnE, alpha, beta) for the technology registry
    from common import PROC
    for k, r in out.items():
        for lik, pre in (("laplace", "SYS"), ("gauss", "SYSG")):
            M = np.array([[np.log(q[f"{pre}_A"]), np.log(q[f"{pre}_B"]), np.log(q[f"{pre}_E"]), q[f"{pre}_alpha"],
                           q[f"{pre}_beta"]] for q in r["reps"]])
            np.save(os.path.join(PROC, f"boot_{k}_system_{lik}_strat.npy"), M)
    objs = pd.concat([out[k]["objs"] for k in out], ignore_index=True)
    objs.to_csv(os.path.join(TABLES, "m1_chinchilla_duality_objects.csv"), index=False)
    sysrows = []
    for k, r in out.items():
        for lik, s in r["system"].items():
            row = dict(dataset=k, likelihood=lik, LR=s["LR"], LR_p=s["LR_p"], n_loss=s["n1"], n_argmin=s["n2"],
                       sd_loss=s["sd_loss"], sd_argmin=s["sd_argmin"], sd_argmin_unrestricted=s["sd_argmin_unres"])
            for kk, v in s["derived"].items():
                row[kk] = v
                if kk in s["se"]:
                    row[f"se_{kk}"] = s["se"][kk]
                    row[f"lo_{kk}"], row[f"hi_{kk}"] = s["ci"][kk]
            for kk, v in s["unrestricted"].items():
                row[f"unres_{kk}"] = v
            sysrows.append(row)
    pd.DataFrame(sysrows).to_csv(os.path.join(TABLES, "m1_chinchilla_system.csv"), index=False)
    log("  duality tests:\n" + tests[tests["cov"] == "robust"][["dataset", "null", "bootstrap", "W_all", "p_all", "W_path",
                                                               "p_path", "W_front", "p_front", "d_a", "se_a", "t_a_ols",
                                                               "p_a_ols"]].round(4).to_string())
    return out, tests, objs, rp


# ----------------------------------------------------------------------------- nested LR test for the system
def system_lr(df, label, lik="laplace", grid=sl.FAST_GRID):
    """Nested LR test of the two cross-equation restrictions in the system estimator.

    Restricted: ln N*_b = pred_b(theta) (design-consistent argmin of the primal).
    Unrestricted: ln N*_b = pred_b(theta) + c0 + c1 (ln C_b - mean ln C_b), i.e. the argmin equation may shift and
    tilt away from what the primal implies (2 extra parameters). Both are concentrated quasi-likelihoods with the same
    loss equation, so the models are nested and LR = 2 (Q_r - Q_u) is asymptotically chi2(2) under the restrictions.
    (The comparison with a free straight-line path used in duality_for is non-nested, because the design-consistent
    prediction is not exactly linear in ln C; it is kept only as a descriptive statistic.)"""
    N, D, L = df.N.values, df.D.values, df.L.values
    iso = df.iso.values.astype(bool)
    lab, lnC = df.budget.values, np.log(df.C.values)
    lN, lD, y = np.log(N), np.log(D), np.log(L)
    obs, used = a2_minima(lN[iso], L[iso], lab[iso], lnC[iso], return_labels=True)
    keep = iso & np.isin(lab, used)
    design = _ParabolaDesign(lN[keep], lab[keep], lnC[keep])
    xc = obs[:, 0] - obs[:, 0].mean()
    X = np.column_stack([np.ones_like(xc), xc])
    n1, n2 = len(y), len(obs)

    def Q(th, free):
        if not np.all(np.isfinite(th)):
            return 1e10
        r1 = y - est.lse_pred(th, lN, lD)
        pred = design.minima(np.exp(est.lse_pred(th, lN[keep], lD[keep])))
        if pred is None or len(pred) != n2:
            return 1e10
        e = obs[:, 1] - pred[:, 1]
        if free:
            e = e - X @ np.linalg.lstsq(X, e, rcond=None)[0]
        q1 = n1 * np.log(np.mean(np.abs(r1))) if lik == "laplace" else 0.5 * n1 * np.log(np.mean(r1 * r1))
        return q1 + 0.5 * n2 * np.log(np.mean(e * e))

    m3 = est.fit_huber(N, D, L, grid=grid)
    starts = [theta_of(m3), theta_of(BESI_PUB)]
    res = {}
    for free in (False, True):
        best = None
        for st in starts + ([res[False][0]] if free else []):
            r = minimize(Q, np.asarray(st, float), args=(free,), method="Nelder-Mead",
                         options=dict(maxiter=20000, maxfev=30000, xatol=1e-9, fatol=1e-11, adaptive=True))
            r2 = minimize(Q, r.x, args=(free,), method="L-BFGS-B")
            r = r2 if r2.fun <= r.fun else r
            if best is None or r.fun < best.fun:
                best = r
        res[free] = (best.x, best.fun)
    th_u = res[True][0]
    pred = design.minima(np.exp(est.lse_pred(th_u, lN[keep], lD[keep])))
    c = np.linalg.lstsq(X, obs[:, 1] - pred[:, 1], rcond=None)[0]
    LR = 2 * (res[False][1] - res[True][1])
    mr = sl.Chinchilla.from_theta(res[False][0])
    return dict(dataset=label, likelihood=lik, n_loss=n1, n_argmin=n2, Q_restricted=res[False][1],
                Q_unrestricted=res[True][1], LR=LR, LR_p=float(chi2.sf(max(LR, 0.0), 2)), c0_level_shift=c[0],
                c1_slope_shift=c[1], alpha_r=mr.alpha, beta_r=mr.beta, a_r=mr.a_N)


def system_lr_all(log=print):
    rows = []
    for k in (5, 0):
        df = load_chinchilla(k)
        for lik in ("laplace", "gauss"):
            rows.append(system_lr(df, f"chinchilla_n{len(df)}", lik))
    import labs
    for exp in labs.EXPERIMENTS:
        df = load_isoflop(exp)
        for lik in ("laplace", "gauss"):
            rows.append(system_lr(df, exp, lik))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLES, "m1_chinchilla_system_lr.csv"), index=False)
    log("  nested system LR tests:\n" + out[["dataset", "likelihood", "LR", "LR_p", "c0_level_shift", "c1_slope_shift"]].round(4).to_string())
    return out
