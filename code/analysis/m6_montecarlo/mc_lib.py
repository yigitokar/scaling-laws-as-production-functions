"""mc_lib.py -- helpers for module m6_montecarlo (Monte Carlo evidence).

Everything here is local to the module; the shared library sl.py is imported read-only
(its private objective/gradient are reused so that the estimator is *exactly* the
Hoffmann/Besiroglu Huber-LSE objective).

Notation follows paper/notes/model_spec.md:
    L = E + A N^-alpha + B D^-beta,  a = beta/(alpha+beta),  gamma = alpha beta/(alpha+beta),
    sigma* = 2/(2+alpha+beta),  M*(C) = D*(C)/N*(C) = (C/6)^(1-2a) / G^2,
    G = (alpha A/(beta B))^(1/(alpha+beta)).
Internal parameter vector (LSE parameterization of Hoffmann App. D.2): th = (lnA, lnB, lnE, alpha, beta).
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import sl  # noqa: E402  (read-only use)

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
QUICK = os.environ.get("M6_QUICK") == "1"
TAB = os.path.join(ROOT, "output", "tables")
FIG = os.path.join(ROOT, "output", "figures")
PROC = os.path.join(ROOT, "data", "processed", "m6_montecarlo")
if QUICK:  # smoke test (run.py --quick): everything goes to a separate folder
    PROC = os.path.join(PROC, "quick")
    TAB = FIG = PROC
for _d in (TAB, FIG, PROC):
    os.makedirs(_d, exist_ok=True)

TRUTH = sl.BESIROGLU                       # E=1.8172, A=482.01, B=2085.43, alpha=0.3478, beta=0.3658
TH0 = TRUTH.theta.copy()
C_REF = 1e24                               # M*(1e24) is reported at this budget
PARAMS = ["alpha", "beta", "a", "gamma", "sigma_star", "lnMstar"]
PARAM_LABEL = {"alpha": r"$\alpha$", "beta": r"$\beta$", "a": r"$a$", "gamma": r"$\gamma$",
               "sigma_star": r"$\sigma^*$", "lnMstar": r"$\ln M^*(10^{24})$"}


def derived(th, C=C_REF):
    """Economic objects implied by th = (lnA, lnB, lnE, alpha, beta)."""
    la, lb, le, al, be = [float(x) for x in th]
    s = al + be
    a = be / s
    lnG = (np.log(al) + la - np.log(be) - lb) / s
    return dict(alpha=al, beta=be, a=a, gamma=al * be / s, sigma_star=2.0 / (2.0 + s),
                lnMstar=(1 - 2 * a) * np.log(C / 6.0) - 2 * lnG, E=np.exp(le))


TRUE = derived(TH0)


def derived_vec(th):
    d = derived(th)
    return np.array([d[k] for k in PARAMS])


# ----------------------------------------------------------------------------- Huber-LSE fits

# L-BFGS-B box: wide enough never to bind at sensible optima, but prevents numerical blow-ups
# (A or B -> 0/inf) on the flat ridges of on-path designs.  Reported in the memo.
BOUNDS = [(-10.0, 60.0), (-10.0, 60.0), (-3.0, 2.5), (1e-3, 5.0), (1e-3, 5.0)]

# A small, fixed multi-start set drawn at random from Hoffmann's grid region
# (ln A, ln B in [0, 25], ln E in [-1, 1], alpha, beta in [0.1, 1.2]).
# v1 (2026-09-23) used the TRUTH + the first 8 of these draws.  v2 (2026-09-24, referee round 1: R3 M12, R1 minor 2,
# R2 minor 7) removes the truth and uses the first 9 draws of the same stream (the 9th replaces the truth), so the
# estimator never sees the true parameter vector.  The first 8 starts are identical in v1 and v2.
_rs = np.random.default_rng(12345)
RANDOM_STARTS = [np.array([_rs.uniform(0, 25), _rs.uniform(0, 25), _rs.uniform(-1, 1),
                           _rs.uniform(0.1, 1.2), _rs.uniform(0.1, 1.2)]) for _ in range(9)]
EXTRA_STARTS = RANDOM_STARTS[:8]          # v1 name (Design B's E1b still builds its start set from TH0 + these)
STARTS = RANDOM_STARTS                    # v2 default start set: 9 random starts, none at the truth


def fit_huber(lN, lD, lL, starts=None, delta=1e-3, bounds=BOUNDS):
    """Hoffmann/Besiroglu estimator (sum of Huber(delta) losses on log-loss residuals of the
    LSE parameterization), L-BFGS-B from several starts; returns (theta, objective).
    v2: the default start set is STARTS (9 random starts; the truth is NOT among them)."""
    w = np.ones_like(lL)
    if starts is None:
        starts = STARTS
    best = (None, np.inf)
    for st in starts:
        r = minimize(sl._obj, np.asarray(st, float), args=(lN, lD, lL, delta, w), jac=sl._grad,
                     method="L-BFGS-B", bounds=bounds, options=dict(maxiter=5000, ftol=1e-15, gtol=1e-12))
        if np.isfinite(r.fun) and r.fun < best[1]:
            best = (r.x, float(r.fun))
    return best


def jac_lse(th, lN, lD):
    """d ln L_hat / d th for the LSE parameterization (n x 5)."""
    a, b, e, al, be = th
    t1, t2, t3 = a - al * lN, b - be * lD, e * np.ones_like(lN)
    m = np.maximum(np.maximum(t1, t2), t3)
    p1, p2, p3 = np.exp(t1 - m), np.exp(t2 - m), np.exp(t3 - m)
    s = p1 + p2 + p3
    p1, p2, p3 = p1 / s, p2 / s, p3 / s
    return np.column_stack([p1, p2, p3, -lN * p1, -lD * p2])


def pred_lse(th, lN, lD):
    return sl._lse_pred(th, lN, lD)


def asym_cov(th, lN, lD, lL):
    """Asymptotic covariance of the Huber(1e-3) estimator. With delta far below the noise scale the
    estimator is (numerically) LAD, so V = (J'J)^-1 / (4 f(0)^2), f(0) = residual density at 0
    (Gaussian kernel, Silverman bandwidth).  Returns (V, cond(J'J))."""
    J = jac_lse(th, lN, lD)
    r = lL - pred_lse(th, lN, lD)
    n = len(r)
    sd = min(np.std(r, ddof=1), (np.percentile(r, 75) - np.percentile(r, 25)) / 1.349)
    h = 1.06 * max(sd, 1e-12) * n ** (-0.2)
    f0 = np.mean(np.exp(-0.5 * (r / h) ** 2)) / (h * np.sqrt(2 * np.pi))
    JJ = J.T @ J
    cond = np.linalg.cond(JJ)
    V = np.linalg.pinv(JJ, rcond=1e-15) / (4 * f0 ** 2)
    return V, cond


def grad_derived(th, eps=1e-6):
    g = np.zeros((len(PARAMS), 5))
    for j in range(5):
        tp, tm = th.copy(), th.copy()
        tp[j] += eps
        tm[j] -= eps
        g[:, j] = (derived_vec(tp) - derived_vec(tm)) / (2 * eps)
    return g


def delta_se(th, V):
    g = grad_derived(th)
    v = np.einsum("ij,jk,ik->i", g, V, g)
    return np.sqrt(np.maximum(v, 0.0))


# ----------------------------------------------------------------------------- kappa-generalized model

# L = E + (A N^-a1 + B D^-b1)^kappa, a1 = S(1-h), b1 = S h.   sigma* = 2/(2+S) (isoquants depend only
# on (a1, b1)); a = h; gamma = kappa a1 b1/S.  Chinchilla is kappa = 1.  Parameters for fixed S:
# q = (a', b', e, h, kappa) with A = exp(a'), B = exp(b'), E = exp(e).

def _kap_parts(q, S, lN, lD):
    ap, bp, e, h, kap = q
    z1 = ap - S * (1 - h) * lN
    z2 = bp - S * h * lD
    lse = np.logaddexp(z1, z2)
    w1 = np.exp(z1 - lse)
    t = kap * lse
    pred = np.logaddexp(e, t)
    pt = np.exp(t - pred)
    return pred, lse, w1, pt


def kap_obj(q, S, lN, lD, lL):
    pred, lse, w1, pt = _kap_parts(q, S, lN, lD)
    r = pred - lL
    ap, bp, e, h, kap = q
    w2 = 1 - w1
    dt = np.column_stack([kap * w1, kap * w2, np.zeros_like(r),
                          kap * (w1 * S * lN - w2 * S * lD), lse])
    J = dt * pt[:, None]
    J[:, 2] = 1 - pt
    return 0.5 * np.sum(r ** 2), J.T @ r


# v1: a', b' in [-30, 120], kappa in [0.01, 60] (adequate for the v1 grid [0.50, 0.95]).
# v2: the extended grid (0.05, 0.99) needs a', b' up to ~450 at sigma* = 0.05 (a' = ln K / kappa + ...) and kappa up to
# ~60 at sigma* = 0.99, so the box is widened; the observationally equivalent start is clipped to the same box.
KAP_BOUNDS = [(-50.0, 1500.0), (-50.0, 1500.0), (-3.0, 2.5), (0.01, 0.99), (0.002, 200.0)]


def fit_kappa_fixedS(S, lN, lD, lL, starts):
    best = (None, np.inf)
    for st in starts:
        r = minimize(kap_obj, np.asarray(st, float), args=(S, lN, lD, lL), jac=True, method="L-BFGS-B",
                     bounds=KAP_BOUNDS, options=dict(maxiter=3000, ftol=1e-13, gtol=1e-10))
        if np.isfinite(r.fun) and r.fun < best[1]:
            best = (r.x, float(r.fun))
    return best


def gauss_nls(lN, lD, lL, th_start):
    """Gaussian NLS on log loss (kappa = 1), used as the unrestricted point of the profile."""
    w = np.ones_like(lL)
    r = minimize(sl._obj, th_start, args=(lN, lD, lL, None, w), jac=sl._grad, method="L-BFGS-B",
                 bounds=BOUNDS, options=dict(maxiter=5000, ftol=1e-15, gtol=1e-12))
    return r.x, float(r.fun)


def kappa_equivalent_start(th, S):
    """Parameters of the kappa-model with inner exponent sum S that reproduce EXACTLY the on-path
    behaviour (expansion path N* = G (C/6)^a and frontier R* = K (C/6)^-gamma) of the Chinchilla fit th
    (Prop. 1 / P2 observational equivalence): b1 = aS, a1 = (1-a)S, kappa = gamma/(a(1-a)S),
    K' = K^(1/kappa), A' = K' a G^a1, B' = ((1-a)/a) A' G^-S."""
    la, lb, le, al, be = th
    a = be / (al + be)
    gam = al * be / (al + be)
    lnG = (np.log(al) + la - np.log(be) - lb) / (al + be)
    lnK = np.log((al + be) / be) + la - al * lnG
    kap = gam / (a * (1 - a) * S)
    lnKp = lnK / kap
    lnAp = lnKp + np.log(a) + (1 - a) * S * lnG
    lnBp = np.log((1 - a) / a) + lnAp - S * lnG
    return np.array([np.clip(lnAp, -49.0, 1499.0), np.clip(lnBp, -49.0, 1499.0), le, np.clip(a, 0.011, 0.989),
                     np.clip(kap, 0.0021, 199.0)])


def kap_obj_free(q, lN, lD, lL):
    """kappa-generalized model with S = a1 + b1 FREE: q = (a', b', e, h, kappa, ln S).  Used only to find the
    unrestricted minimum SSR that the profile LR must be measured against."""
    ap, bp, e, h, kap, lS = q
    S = np.exp(lS)
    pred, lse, w1, pt = _kap_parts(q[:5], S, lN, lD)
    r = pred - lL
    w2 = 1 - w1
    dlse_dlS = S * (-(1 - h) * lN * w1 - h * lD * w2)          # d lse / d ln S
    dt = np.column_stack([kap * w1, kap * w2, np.zeros_like(r),
                          kap * (w1 * S * lN - w2 * S * lD), lse, kap * dlse_dlS])
    J = dt * pt[:, None]
    J[:, 2] = 1 - pt
    return 0.5 * np.sum(r ** 2), J.T @ r


# v1: ln S in [ln 0.01, ln 20]  <=>  sigma* in [0.09, 0.995].
# v2: ln S in [ln 0.01, ln 80]  <=>  sigma* in [0.024, 0.995], so that the unrestricted optimum can lie anywhere on the
# extended profile grid (0.05, 0.99) (R1 comment 9c, R4 M5b).
KAP_FREE_BOUNDS = KAP_BOUNDS + [(np.log(0.01), np.log(80.0))]


def fit_kappa_free(lN, lD, lL, starts):
    best = (None, np.inf)
    for st in starts:
        r = minimize(kap_obj_free, np.asarray(st, float), args=(lN, lD, lL), jac=True, method="L-BFGS-B",
                     bounds=KAP_FREE_BOUNDS, options=dict(maxiter=3000, ftol=1e-13, gtol=1e-10))
        if np.isfinite(r.fun) and r.fun < best[1]:
            best = (r.x, float(r.fun))
    return best


def th_from_path_dual(lN, lD, a, lnG, e, lnK, gam):
    """kappa=1 parameter vector (lnA, lnB, lnE, alpha, beta) whose expansion path is ln N* = lnG + a ln(C/6) and whose
    frontier is ln(L* - E) = lnK - gamma ln(C/6) (inverse of the duality map; alpha = gamma/a, beta = gamma/(1-a)).
    v2: used to build a second observationally-equivalent start for the kappa-free profile from the dual (path)
    estimate, which reproduces on-path data even when the kappa=1 primal fit is a corner with a different path."""
    a = float(np.clip(a, 0.02, 0.98))
    gam = float(np.clip(gam, 1e-3, 3.0))
    al, be = gam / a, gam / (1 - a)
    la = lnK - np.log((al + be) / be) + al * lnG
    lb = np.log(al) + la - np.log(be) - (al + be) * lnG
    return np.array([la, lb, e, al, be])


def profile_sigma(lN, lD, lL, sig_grid, th_start, return_info=False, extra_th=()):
    """Profile Gaussian log-likelihood over sigma* in the kappa-generalized model.
    Returns LR(sigma*) = n ln(SSR(sigma*)/SSR_min) on sig_grid (chi2(1) under the null).
    Starts at each grid point: (i) the analytically observationally-equivalent parameters of the kappa=1
    fit (so that on-path data give LR = 0 up to optimisation error), (ii) the neighbouring grid solution.

    SSR_min is the UNRESTRICTED minimum of the kappa-free model (S free).  [Review fix, 2026-09-23: the
    original code used min(grid SSR, kappa=1 SSR), an upper bound on SSR_min, which understated every LR
    by n ln(SSR_used/SSR_min) (0.1-1.6 in spot checks; the unrestricted optimum can lie between grid points
    or outside [0.50, 0.95]) and so overstated the share of flat profiles.  The unrestricted minimum is now
    found by an L-BFGS-B fit with S free, started from the best grid solution, the kappa=1 fit and the
    second-best grid solution.]
    v2 (2026-09-24): extra_th = further kappa=1 parameter vectors (e.g. the dual/path estimate, th_from_path_dual)
    whose observationally equivalent kappa-model parameters are added as starts at every grid point.  Needed on the
    extended grid (0.05, 0.99): when the kappa=1 fit is a corner its equivalent start does not reproduce the data and
    the optimiser stalled at sigma* = 0.99 (kappa ~ 45) in spot checks."""
    n = len(lL)
    th, _ = gauss_nls(lN, lD, lL, th_start)
    la, lb, le, al, be = th
    S_hat = al + be
    q_hat = np.array([la, lb, le, be / S_hat, 1.0])
    S_grid = 2.0 / np.asarray(sig_grid) - 2.0
    order_up = [i for i in np.argsort(S_grid) if S_grid[i] >= S_hat]
    order_dn = [i for i in np.argsort(S_grid)[::-1] if S_grid[i] < S_hat]
    ssr = np.full(len(S_grid), np.nan)
    qs = [None] * len(S_grid)
    for order in (order_up, order_dn):
        q_prev = q_hat.copy()
        for i in order:
            S = S_grid[i]
            starts = [kappa_equivalent_start(th, S), q_prev] + [kappa_equivalent_start(t_, S) for t_ in extra_th]
            q, f = fit_kappa_fixedS(S, lN, lD, lL, starts)
            ssr[i] = 2 * f
            qs[i] = q
            q_prev = q
    ssr_k1 = 2 * kap_obj(q_hat, S_hat, lN, lD, lL)[0]
    ssr_old = min(np.nanmin(ssr), ssr_k1)                    # the original (upper-bound) denominator
    rank = np.argsort(ssr)
    free_starts = [np.r_[qs[rank[0]], np.log(S_grid[rank[0]])], np.r_[q_hat, np.log(S_hat)],
                   np.r_[qs[rank[1]], np.log(S_grid[rank[1]])]]
    free_starts = [np.clip(s, [b[0] for b in KAP_FREE_BOUNDS], [b[1] for b in KAP_FREE_BOUNDS]) for s in free_starts]
    qf, ff = fit_kappa_free(lN, lD, lL, free_starts)
    ssr_min = min(ssr_old, 2 * ff)
    lr = n * np.log(ssr / ssr_min)
    if not return_info:
        return lr
    S_free = float(np.exp(qf[5])) if 2 * ff <= ssr_old else (S_grid[rank[0]] if ssr[rank[0]] <= ssr_k1 else S_hat)
    return lr, dict(prof_shift=float(n * np.log(ssr_old / ssr_min)), prof_sig_hat=2.0 / (2.0 + S_free))


# ----------------------------------------------------------------------------- designs (Design A)

ISO_BUDGETS = np.array([6e18, 1e19, 3e19, 6e19, 1e20, 3e20, 6e20, 1e21, 3e21])   # Chinchilla IsoFLOP budgets
RUNS_PER_BUDGET = 10


def total_compute():
    return RUNS_PER_BUDGET * ISO_BUDGETS.sum()


def design_isoflop(width=16.0, reps=1):
    """Chinchilla-like IsoFLOP: 9 budgets x 10 sizes log-spaced over [N*/width, width N*]."""
    N, D = [], []
    for C in ISO_BUDGETS:
        Ns = TRUTH.N_opt(C) * width ** np.linspace(-1, 1, RUNS_PER_BUDGET)
        N.append(Ns)
        D.append(C / (6 * Ns))
    N, D = np.concatenate(N), np.concatenate(D)
    return np.tile(N, reps), np.tile(D, reps)


def design_path(s, rng, reps=1):
    """Optimizing labs: same 90 budgets as the IsoFLOP design; allocation ln(D/N) = optimum + s*z
    (compute held fixed, so the error moves the run along its isocost line)."""
    C = np.tile(np.repeat(ISO_BUDGETS, RUNS_PER_BUDGET), reps)
    u = s * rng.standard_normal(C.size)
    N = TRUTH.N_opt(C) * np.exp(-u / 2)
    D = TRUTH.D_opt(C) * np.exp(u / 2)
    return N, D


def design_factorial(nN=9, nD=10, ratio=256.0, reps=1):
    """Near-orthogonal N x D grid (Farseer-like) centred on the expansion path, total compute matched."""
    xN = ratio ** (np.linspace(0, 1, nN) - 0.5)
    xD = ratio ** (np.linspace(0, 1, nD) - 0.5)
    # sum_ij 6 N_i D_j = 6 N_c D_c sum(xN) sum(xD) = C_c sum(xN) sum(xD)
    Cc = total_compute() / (xN.sum() * xD.sum())
    Nc, Dc = TRUTH.N_opt(Cc), TRUTH.D_opt(Cc)
    NN, DD = np.meshgrid(Nc * xN, Dc * xD, indexing="ij")
    return np.tile(NN.ravel(), reps), np.tile(DD.ravel(), reps)


def het_cluster_noise(N, D, sd, rng, clusters, het=0.25, rho=0.5):
    """v2 (R2 minor 8): heteroskedastic, within-cluster correlated log-loss noise.
    sd_i = k * (N_i / N_g)^(-het) * (D_i / D_g)^(-het), N_g, D_g = geometric means in the design, with k chosen so that
    the design-average variance equals sd^2 (the iid baseline): noise is larger for small models AND for short runs
    (few tokens).  At fixed compute the two effects offset (small N means long runs), so within an IsoFLOP budget the
    noise is homoskedastic and across budgets sd ~ C^(-het) (sd ratio 500^0.25 = 4.7 between the smallest and largest
    Chinchilla budgets); in the factorial grid it varies along both inputs.
    eps_i = sd_i (sqrt(rho) z_g(i) + sqrt(1-rho) z_i): a common shock shared by all runs in a cluster (shared data
    order / seed / schedule), correlation rho within cluster.  Draw order: cluster shocks first, then run shocks."""
    lN, lD = np.log(N), np.log(D)
    rel = np.exp(-het * (lN - lN.mean()) - het * (lD - lD.mean()))
    sdi = sd * rel / np.sqrt(np.mean(rel ** 2))
    ids, inv = np.unique(clusters, return_inverse=True)
    zg = rng.standard_normal(len(ids))
    zi = rng.standard_normal(N.size)
    return sdi * (np.sqrt(rho) * zg[inv] + np.sqrt(1 - rho) * zi)


def simulate_loss(N, D, sd, rng, digitize=False, noise=None):
    """Loss with multiplicative (log) noise. digitize=True adds Epoch-style digitization error:
    L rounded to a 0.012-nat grid (256-colour map), 2% pixel error in N and C, D imputed as C/(6N).
    noise: None (iid N(0, sd^2), v1) or a callable (N, D, sd, rng) -> log-noise vector (v2 het/cluster cells)."""
    eps = sd * rng.standard_normal(N.size) if noise is None else noise(N, D, sd, rng)
    lL = np.log(TRUTH.loss(N, D)) + eps
    if not digitize:
        return N, D, np.exp(lL)
    L = np.round(np.exp(lL) / 0.012) * 0.012
    C = 6 * N * D * np.exp(0.02 * rng.standard_normal(N.size))
    Nobs = N * np.exp(0.02 * rng.standard_normal(N.size))
    return Nobs, C / (6 * Nobs), L


def transverse_sd(N, D):
    """Off-path variation: (i) sd of the transverse coordinate t = alpha n - beta d after partialling out
    c (Lemma 2); (ii) sd of ln(D/N) given c; (iii) root-mean-square distance of ln(D/N) from the TRUE
    compute-optimal ln M*(C) (captures systematic tilts, e.g. Kaplan beliefs, that are linear in c)."""
    n, d = np.log(N), np.log(D)
    t = TRUTH.alpha * n - TRUTH.beta * d
    c = n + d
    X = np.column_stack([np.ones_like(c), c])
    res = t - X @ np.linalg.lstsq(X, t, rcond=None)[0]
    lnM = np.log(D / N)
    C = 6 * N * D
    dev = lnM - np.log(TRUTH.D_opt(C) / TRUTH.N_opt(C))
    return (float(np.std(res)), float(np.std(lnM - X @ np.linalg.lstsq(X, lnM, rcond=None)[0])),
            float(np.sqrt(np.mean(dev ** 2))))


def design_conditioning(N, D):
    """Condition numbers of J'J at the truth: raw LSE parameterization and KMW-normalized
    (A, B normalized at the geometric means of N and D)."""
    lN, lD = np.log(N), np.log(D)
    J = jac_lse(TH0, lN, lD)
    raw = np.linalg.cond(J.T @ J)
    # normalized: lnA' = lnA - alpha*mean(lN), lnB' = lnB - beta*mean(lD) => d/dalpha picks up -(lN - mean)
    Jn = J.copy()
    Jn[:, 3] = -(lN - lN.mean()) * J[:, 0]
    Jn[:, 4] = -(lD - lD.mean()) * J[:, 1]
    nrm = np.linalg.cond(Jn.T @ Jn)
    sv = np.linalg.svd(Jn, compute_uv=False)
    return raw, nrm, float(sv[-1] / sv[0])
