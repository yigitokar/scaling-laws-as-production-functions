"""Design B -- a simulated AI industry (model_spec Sections 1-2) and IO production-function estimators.

DGP (one replication)
---------------------
Labs f = 1..40, generations t = 1..6, 1-4 models per lab-generation (tiers m = 1 flagship, 2, 3, 4).
Technology (truth = Besiroglu Chinchilla): y = -ln(L - E) = omega_ft + F(n, d + psi_D,f) - eps,
    F(n, d) = -ln(A e^{-alpha n} + B e^{-beta d}).
Productivity: omega_ft = delta_t + x_ft, x_ft = rho x_{f,t-1} + xi_ft (rho = 0.8, stationary sd 0.25),
    delta_t = g (t-1) with g = 0.15 (Hicks-neutral TFP growth per generation, y units).
    sd(omega) = 0.25 matches ~41x p90/p10 in effective compute (Mertens et al.) at gamma = 0.178:
    0.178 * ln 41 / 2.563 = 0.26.
Compute rules for the lab-generation budget c_ft (flagship; tier m uses c_ft - Delta_m, Delta_m ~ (m-1) ln 8):
    exog      c_ft = cbar_t + h_f - p_ft + nu_ft                           (ZKD case)
    funding   c_ft = cbar_t + h_f - p_ft + nu_ft + lambda x_ft             (simultaneity; lambda = 2)
    target    c chosen so that y hits a target ybar_ftm (capability target; lab knows omega_ft)
    predet    c_ft = cbar_t + h_f - p_ft + nu_ft + lambda x_{f,t-1}         (ACF timing: provisioned ahead)
    p_ft = compute-price shifter (observed; excluded from the technology) -> instrument.
Allocation: lifetime-cost optimum (Sardana et al.) given c: alpha u/(beta v) = 1 + T/(3D),
    with lab-specific inference demand T = 3 D*(c) theta, ln theta ~ N(0, 0.8^2), plus allocation error
    N(0, 0.2^2) in ln(D/N) at fixed c.  Hicks-neutral omega does not enter the allocation (Lemma 1).
Seed/eval noise eps ~ N(0, 0.05^2) in y units (d ln L/dy = -R/L: ~0.011 in ln L at 1e21, ~0.005 at 1e23 FLOP).
Variants: selection (release only if y beats the lab's previous same-tier model by 0.5, or clears the
    generation's median y -- an absolute 'notability' threshold; open-weight labs' bar is 0.25 lower), measurement error in D
    (sd 0.15 in logs; reported C = 6 N D_obs), factor-biased lab heterogeneity psi_D,f ~ N(0, 0.3^2).
E (irreducible loss) is treated as known (= truth) by every estimator except E1 (ML practice), which
estimates E jointly; this isolates the IO identification problems from the E-level problem.

Estimators (all on released models only)
----------------------------------------
E1 pooled Huber-LSE on ln L (E free; no lab/time effects)      -- ML practice (Hoffmann/Besiroglu)
E2 pooled NLS in y with generation effects
E3 lab-FE NLS (Hicks-neutral lab shift of reducible loss) + generation effects
E4 Mundlak / correlated random effects (lab means of n, d)
E5 lab x generation FE (within-family variation only; Hoch 1962 'multi-plant firm')
E6 ACF/Wooldridge-type GMM: quasi-differenced Markov omega, lagged-input (and t-2 output) instruments,
   current lab-generation compute treated as predetermined (valid under exog and predet timing)
E6b same with lagged instruments only (robust to contemporaneous compute responses)
E7 IV-path: a from the allocation regression n ~ c, gamma from 2SLS of y on c instrumented by the
   compute price; alpha = gamma/a, beta = gamma/(1-a) (Prop. 1 functional-form identification)
E8 Heckman two-step selection correction (pooled NLS + inverse Mills ratio; exclusion: open-weight policy)
E9 GNR/FOC 'system': E5 loss equation stacked with the training-only FOC alpha u = beta v (T ignored)
E10 same, with the FOC modeling inference demand through a noisy usage proxy of T (ln w = ln(1+T/3D))

v2 (2026-09-24, after referee round 1; R3 M12, R2 minor 7): NO estimator is started at the true parameters.
    v1 started E2/E3/E4/E5/E8 at the truth (start_norm), E1/E1b from (truth + 8 random starts).  v2 uses
    start_grid(S): 9 data-based starts (alpha, beta in {0.2, 0.5, 1.0}^2, levels matched to mean y), keeping the
    lowest least-squares cost; E1 uses ml.STARTS (9 random starts), E1b the same with ln E fixed at the truth.
    E6/E6b start from E2's solution and E9/E10 from E5's, as before.  The simulated data (seeds) are unchanged.
    designB_startcheck.py / designB_startcheck2.py document why: under the target rule the pooled NLS criterion has
    several local optima and the truth start selected the one nearest the truth in about 15 percent of replications.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import norm

import mc_lib as ml

T0 = ml.TRUTH
AL, BE, LNA, LNB, E_TRUE = T0.alpha, T0.beta, np.log(T0.A), np.log(T0.B), T0.E
GAM, A_N = T0.gamma, T0.a_N
LN6 = np.log(6.0)

BASE = dict(F=40, T=6, rho=0.8, sd_om=0.25, g=0.15, sd_eps=0.05, c1=np.log(1e21), c_growth=np.log(3.0),
            sd_h=0.7, sd_nu=0.4, sd_p=0.4, lam=2.0, tier_step=np.log(8.0), sd_tier=0.3,
            sd_lntheta=0.8, s_alloc=0.2, sd_target=0.10, me=0.0, psiD=0.0, sel=False, sel_margin=0.5,
            sel_q=0.5, sel_open=0.25, sd_Tproxy=0.3, max_models=4)

SCENARIOS = [
    dict(name="exog", label="(a) Exogenous budget", rule="exog"),
    dict(name="funding", label="(b) Funding rule (c responds to omega)", rule="funding"),
    dict(name="target", label="(c) Target-loss rule", rule="target"),
    dict(name="predet", label="(d) Predetermined budget (ACF timing)", rule="predet"),
    dict(name="exog_sel", label="(a) + selection on release", rule="exog", sel=True),
    dict(name="predet_sel", label="(d) + selection on release", rule="predet", sel=True),
    dict(name="predet_me", label="(d) + measurement error in D (sd 0.15)", rule="predet", me=0.15),
    dict(name="predet_psiD", label="(d) + data-augmenting lab heterogeneity", rule="predet", psiD=0.3),
    dict(name="funding_all", label="(b) + selection + ME in D", rule="funding", sel=True, me=0.15),
    dict(name="funding_flag", label="(b) flagships only (1 model per lab-generation)", rule="funding", max_models=1),
]

ESTIMATORS = ["E1_pooled_huber", "E1b_pooled_huber_Etrue", "E2_pooled_nls", "E3_lab_fe", "E4_mundlak", "E5_labgen_fe", "E6_acf",
              "E6b_acf_lagonly", "E7_iv_path", "E8_heckman", "E9_foc_system", "E10_foc_system_T"]
EST_LABEL = {"E1_pooled_huber": "Pooled Huber-LSE (ML practice, E free)",
             "E1b_pooled_huber_Etrue": "Pooled Huber-LSE, E fixed at truth (decomposition of E1)", "E2_pooled_nls": "Pooled NLS + generation effects",
             "E3_lab_fe": "Lab FE NLS", "E4_mundlak": "Mundlak (CRE)", "E5_labgen_fe": "Lab x generation FE",
             "E6_acf": "ACF/Wooldridge GMM (c predetermined)", "E6b_acf_lagonly": "ACF GMM (lagged instruments only)",
             "E7_iv_path": "IV (compute price) + expansion path", "E8_heckman": "Heckman selection correction",
             "E9_foc_system": "FOC system (T ignored)", "E10_foc_system_T": "FOC system (T proxied)"}
OUT_PARAMS = ["alpha", "beta", "a", "gamma", "sigma_star", "lnMstar", "sd_omega", "tfp_growth"]


def truth_row(P=BASE):
    t = dict(ml.TRUE)
    t["sd_omega"] = P["sd_om"]
    t["tfp_growth"] = P["g"]
    return {k: float(t[k]) for k in OUT_PARAMS}


# ----------------------------------------------------------------------------- technology helpers

def Fval(n, d, lnA=LNA, lnB=LNB, al=AL, be=BE):
    return -np.logaddexp(lnA - al * n, lnB - be * d)


def alloc_d(c, T, psiD):
    """Solve the lifetime-cost FOC  h(d) = ln(alpha A/(beta B)) - alpha n + beta (d+psiD) - ln(1 + T e^{-d}/3) = 0,
    n = c - ln6 - d, for d.  h is strictly increasing and concave in d, and h(d0) <= 0 at the T = 0 solution d0,
    so Newton's method started at d0 increases monotonically to the root (the tangent of a concave function
    lies above it)."""
    k = np.log(AL / BE) + LNA - LNB
    d = (AL * (c - LN6) - k - BE * psiD) / (AL + BE) + 0.0 * np.asarray(T, float)      # T = 0 solution
    for _ in range(60):
        q = T * np.exp(-d) / 3.0
        h = k - AL * (c - LN6 - d) + BE * (d + psiD) - np.log1p(q)
        step = h / (AL + BE + q / (1 + q))
        d = d - step
        if np.max(np.abs(step)) < 1e-12:
            break
    return d


def Dstar(c):
    return (np.exp(c) / 6.0) ** (1 - A_N) / T0.G


def F_path(c):
    return GAM * (c - LN6) - np.log(T0.K)


# ----------------------------------------------------------------------------- DGP

def simulate(rng, sc, P=None):
    P = dict(BASE if P is None else P)
    P.update({k: v for k, v in sc.items() if k in P})
    Fn, Tn, rho = P["F"], P["T"], P["rho"]
    # productivity (x_{f,0} kept for the predetermined rule)
    x = np.empty((Fn, Tn + 1))
    x[:, 0] = P["sd_om"] * rng.standard_normal(Fn)
    for t in range(1, Tn + 1):
        x[:, t] = rho * x[:, t - 1] + P["sd_om"] * np.sqrt(1 - rho ** 2) * rng.standard_normal(Fn)
    delta = P["g"] * np.arange(Tn)
    h = P["sd_h"] * rng.standard_normal(Fn)
    p = P["sd_p"] * rng.standard_normal((Fn, Tn))
    nu = P["sd_nu"] * rng.standard_normal((Fn, Tn))
    zeta = P["sd_target"] * rng.standard_normal((Fn, Tn))
    psiD_f = P["psiD"] * rng.standard_normal(Fn)
    open_f = (rng.random(Fn) < 0.5).astype(float)
    cbar = P["c1"] + P["c_growth"] * np.arange(Tn)
    c_ft = cbar[None, :] + h[:, None] - p + nu
    if sc["rule"] == "funding":
        c_ft = c_ft + P["lam"] * x[:, 1:]
    elif sc["rule"] == "predet":
        c_ft = c_ft + P["lam"] * x[:, :-1]
    Mft = rng.integers(1, P["max_models"] + 1, (Fn, Tn))
    rows = []
    for f in range(Fn):
        for t in range(Tn):
            for m in range(Mft[f, t]):
                dl = m * P["tier_step"] + (P["sd_tier"] * rng.standard_normal() if m > 0 else 0.0)
                rows.append((f, t, m, dl))
    arr = np.array(rows)
    lab, gen, tier, dlt = arr[:, 0].astype(int), arr[:, 1].astype(int), arr[:, 2].astype(int), arr[:, 3]
    nobs = len(lab)
    omega = delta[gen] + x[lab, gen + 1]
    psiD = psiD_f[lab]
    lntheta = P["sd_lntheta"] * rng.standard_normal(nobs)
    u_alloc = P["s_alloc"] * rng.standard_normal(nobs)
    if sc["rule"] == "target":
        # target: what an average-productivity lab with budget cbar+h-p-Delta would reach on the path
        c_nom = cbar[gen] + h[lab] - p[lab, gen] - dlt
        ybar = delta[gen] + F_path(c_nom) + zeta[lab, gen]
        lo, hi = np.full(nobs, np.log(1e14)), np.full(nobs, np.log(1e32))
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            T_ = 3 * Dstar(mid) * np.exp(lntheta)
            d_ = alloc_d(mid, T_, psiD)
            yv = omega + Fval(mid - LN6 - d_, d_ + psiD)
            up = yv < ybar
            lo = np.where(up, mid, lo)
            hi = np.where(up, hi, mid)
        c = 0.5 * (lo + hi)
    else:
        c = c_ft[lab, gen] - dlt
    Tinf = 3 * Dstar(c) * np.exp(lntheta)
    d = alloc_d(c, Tinf, psiD) + u_alloc / 2
    n = c - LN6 - d
    eps = P["sd_eps"] * rng.standard_normal(nobs)
    y = omega + Fval(n, d + psiD) - eps
    L = E_TRUE + np.exp(-y)
    d_obs = d + P["me"] * rng.standard_normal(nobs)
    c_obs = LN6 + n + d_obs
    T_proxy = Tinf * np.exp(P["sd_Tproxy"] * rng.standard_normal(nobs))
    # release rule
    released = np.ones(nobs, bool)
    if P["sel"]:
        released[:] = False
        released[gen == 0] = True
        for t in range(1, Tn):
            it = np.where(gen == t)[0]
            q = np.quantile(y[it], P["sel_q"])
            for i in it:
                f = lab[i]
                prev = np.where((lab == f) & (gen < t) & released)[0]
                if len(prev) == 0:
                    ref = -np.inf
                else:
                    same = prev[tier[prev] == tier[i]]
                    ref = (y[same].max() if len(same) else y[prev].min()) + P["sel_margin"]
                released[i] = y[i] >= min(ref, q) - P["sel_open"] * open_f[f]
    return dict(lab=lab, gen=gen, tier=tier, n=n, d=d, d_obs=d_obs, c=c, c_obs=c_obs, y=y, L=L, omega=omega,
                x=x[lab, gen + 1], T=Tinf, T_proxy=T_proxy, p=p[lab, gen], open=open_f[lab], released=released,
                Tn=Tn, P=P)


# ----------------------------------------------------------------------------- estimation helpers

class Sample:
    """Released models with the index structures the estimators need."""

    def __init__(self, S, full=None):
        r = S["released"]
        self.full = S
        self.y, self.n, self.d, self.c = S["y"][r], S["n"][r], S["d_obs"][r], S["c_obs"][r]
        self.L, self.p, self.Tp, self.open = S["L"][r], S["p"][r], S["T_proxy"][r], S["open"][r]
        self.lab, self.gen = S["lab"][r], S["gen"][r]
        self.Tn = S["Tn"]
        self.N = len(self.y)
        self.n0, self.d0 = self.n.mean(), self.d.mean()
        self.lg = self.lab * 100 + self.gen
        _, self.lab_i = np.unique(self.lab, return_inverse=True)
        _, self.lg_i = np.unique(self.lg, return_inverse=True)
        self.G = np.zeros((self.N, self.Tn - 1))
        for t in range(1, self.Tn):
            self.G[:, t - 1] = self.gen == t

    def demean(self, v, idx):
        v = np.asarray(v, float)
        if v.ndim == 1:
            s = np.bincount(idx, v)
            k = np.bincount(idx)
            return v - (s / k)[idx]
        return np.column_stack([self.demean(v[:, j], idx) for j in range(v.shape[1])])

    def group_mean(self, v, idx):
        s = np.bincount(idx, v)
        k = np.bincount(idx)
        return (s / k)[idx]


def F_J(S, lA, lB, al, be):
    """F and its Jacobian w.r.t. (lA, lB, al, be) in the KMW-normalized parameterization
    z1 = lA - al (n - n0), z2 = lB - be (d - d0)."""
    z1 = lA - al * (S.n - S.n0)
    z2 = lB - be * (S.d - S.d0)
    lse = np.logaddexp(z1, z2)
    w1 = np.exp(z1 - lse)
    w2 = 1 - w1
    J = np.column_stack([-w1, -w2, w1 * (S.n - S.n0), w2 * (S.d - S.d0)])
    return -lse, J


def to_theta(S, lA, lB, al, be):
    """normalized -> (lnA, lnB, lnE, alpha, beta) of the unnormalized Chinchilla form (E at truth)."""
    return np.array([lA + al * S.n0, lB + be * S.d0, np.log(E_TRUE), al, be])


def start_norm(S):
    """v1 start: the TRUE parameters in normalized form.  Not used in v2 (kept for designB_startcheck*.py)."""
    return np.array([LNA - AL * S.n0, LNB - BE * S.d0, AL, BE])


def start_grid(S):
    """v2 start set (none at the truth): alpha, beta in {0.2, 0.5, 1.0}^2 (the neutral start alpha = beta = 0.5
    first), with both normalized levels set to -mean(y) - ln 2, so that F at the design centre equals the sample
    mean of y."""
    lv = -float(np.mean(S.y)) - np.log(2.0)
    grid = [(0.5, 0.5)] + [(al, be) for al in (0.2, 0.5, 1.0) for be in (0.2, 0.5, 1.0) if (al, be) != (0.5, 0.5)]
    return [np.array([lv, lv, al, be]) for al, be in grid]


def ls_multi(fun, jac, q0s, **kw):
    """least_squares from several starts; returns the solution with the lowest cost."""
    best = None
    for q0 in q0s:
        try:
            sol = ls(fun, q0, jac=jac, **kw)
        except Exception:  # noqa: BLE001
            continue
        if np.isfinite(sol.cost) and (best is None or sol.cost < best.cost):
            best = sol
    if best is None:
        raise RuntimeError("all starts failed")
    return best


def summarize_fit(S, lA, lB, al, be):
    """Common outputs: technology objects, TFP dispersion and TFP growth from TFP residuals y - F_hat."""
    th = to_theta(S, lA, lB, al, be)
    out = dict(zip(ml.PARAMS, ml.derived_vec(th)))
    Fh, _ = F_J(S, lA, lB, al, be)
    out.update(tfp_stats(S, S.y - Fh))
    return out


def tfp_stats(S, om):
    """omega_hat_ft = lab-generation mean of (y - F_hat); sd after removing generation means;
    growth = mean per-generation change of the generation means."""
    ok = np.isfinite(om)
    if ok.mean() < 0.9:
        return dict(sd_omega=np.nan, tfp_growth=np.nan)
    lg_i = S.lg_i[ok]
    with np.errstate(invalid="ignore", divide="ignore"):
        ft = np.bincount(lg_i, om[ok]) / np.bincount(lg_i)
    ft_gen = np.zeros(lg_i.max() + 1, int)
    ft_gen[lg_i] = S.gen[ok]
    have = np.bincount(lg_i) > 0
    ft, ft_gen = ft[have], ft_gen[have]
    gm = np.array([ft[ft_gen == t].mean() if np.any(ft_gen == t) else np.nan for t in range(S.Tn)])
    resid = ft - gm[ft_gen]
    return dict(sd_omega=float(np.std(resid, ddof=1)), tfp_growth=float((gm[-1] - gm[0]) / (S.Tn - 1)))


def ls(fun, x0, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return least_squares(fun, x0, **kw)


# ----------------------------------------------------------------------------- estimators

def est_E1(S):
    lN, lD, lL = S.n, S.d, np.log(S.L)
    th, _ = ml.fit_huber(lN, lD, lL)
    out = dict(zip(ml.PARAMS, ml.derived_vec(th)))
    Eh = np.exp(th[2])
    with np.errstate(invalid="ignore"):
        yh = -np.log(S.L - Eh)
    Fh = -np.logaddexp(th[0] - th[3] * lN, th[1] - th[4] * lD)
    out.update(tfp_stats(S, yh - Fh))
    out["E_hat"] = Eh
    return out


def est_E1b(S):
    """Review addition (2026-09-23): E1 with E held at the truth (still pooled, no time effects).  Comparing E1 and
    E1b separates the two channels through which ML practice absorbs TFP growth: a steeper fitted frontier
    (attributing progress to scale, the Sahal channel) versus a downward-biased E-hat (which compresses
    y = -ln(L - E) and hence measured progress)."""
    lN, lD, lL = S.n, S.d, np.log(S.L)
    b = list(ml.BOUNDS)
    b[2] = (np.log(E_TRUE), np.log(E_TRUE))
    starts = [np.r_[st[:2], np.log(E_TRUE), st[3:]] for st in ml.STARTS]          # v2: no truth start
    th, _ = ml.fit_huber(lN, lD, lL, starts=starts, bounds=b)
    out = dict(zip(ml.PARAMS, ml.derived_vec(th)))
    Fh = -np.logaddexp(th[0] - th[3] * lN, th[1] - th[4] * lD)
    out.update(tfp_stats(S, S.y - Fh))
    return out


def est_E2(S):
    def res(q):
        F, J = F_J(S, *q[:4])
        r = S.y - F - S.G @ q[4:]
        return r, np.column_stack([-J, -S.G])

    q0s = [np.r_[s0, np.zeros(S.Tn - 1)] for s0 in start_grid(S)]                 # v2: no truth start
    sol = ls_multi(lambda q: res(q)[0], lambda q: res(q)[1], q0s, method="lm")
    return summarize_fit(S, *sol.x[:4]), sol.x


def est_E3(S):
    Gd = S.demean(S.G, S.lab_i)

    def res(q):
        F, J = F_J(S, 0.0, q[0], q[1], q[2])
        r = S.demean(S.y - F, S.lab_i) - Gd @ q[3:]
        Jd = S.demean(J[:, 1:], S.lab_i)
        return r, np.column_stack([-Jd, -Gd])

    q0s = [np.r_[s0[1] - s0[0], s0[2:], np.zeros(S.Tn - 1)] for s0 in start_grid(S)]   # v2: no truth start
    sol = ls_multi(lambda q: res(q)[0], lambda q: res(q)[1], q0s, method="lm")
    return summarize_fit(S, 0.0, *sol.x[:3])


def est_E4(S):
    nb, db = S.group_mean(S.n, S.lab_i), S.group_mean(S.d, S.lab_i)
    X = np.column_stack([np.ones(S.N), nb - S.n0, db - S.d0])

    def res(q):
        F, J = F_J(S, 0.0, q[0], q[1], q[2])
        r = S.y - F - S.G @ q[3:3 + S.Tn - 1] - X @ q[3 + S.Tn - 1:]
        return r, np.column_stack([-J[:, 1:], -S.G, -X])

    q0s = [np.r_[s0[1] - s0[0], s0[2:], np.zeros(S.Tn - 1), -s0[0], 0.0, 0.0] for s0 in start_grid(S)]  # v2
    sol = ls_multi(lambda q: res(q)[0], lambda q: res(q)[1], q0s, method="lm")
    return summarize_fit(S, 0.0, *sol.x[:3])


def _labgen_res(S, q):
    F, J = F_J(S, 0.0, q[0], q[1], q[2])
    return S.demean(S.y - F, S.lg_i), -S.demean(J[:, 1:], S.lg_i)


def est_E5(S):
    q0s = [np.r_[s0[1] - s0[0], s0[2:]] for s0 in start_grid(S)]                     # v2: no truth start
    sol = ls_multi(lambda q: _labgen_res(S, q)[0], lambda q: _labgen_res(S, q)[1], q0s, method="lm")
    return summarize_fit(S, 0.0, *sol.x), sol.x


def est_E6(S, q_start, lag_only=False):
    """ACF/Wooldridge-type GMM with Markov omega (AR(1) around generation effects).
    xi_ftm = (y - F - delta_t) - rho * (mean_{f,t-1}(y - F) - delta_{t-1}); moments E[Z xi] = 0 plus the
    level normalisation E[y - F | t = 1] = 0 (delta_1 = 0, E x = 0)."""
    Tn = S.Tn
    lab, gen = S.lab, S.gen
    # lab-generation means (released models)
    key = lab * 100 + gen
    uk, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv)
    mean_of = lambda v: np.bincount(inv, v) / cnt
    k_index = {k: i for i, k in enumerate(uk)}
    prev = np.array([k_index.get(k - 1, -1) for k in key])     # (f, t-1) group of each obs
    prev2 = np.array([k_index.get(k - 2, -1) for k in key])
    use = (gen >= 1) & (prev >= 0)
    nbar, dbar, cbar, ybar = mean_of(S.n), mean_of(S.d), mean_of(S.c), mean_of(S.y)
    ndev, ddev = S.n - nbar[inv], S.d - dbar[inv]
    Gz = S.G[use]
    y2 = np.where(prev2 >= 0, ybar[np.maximum(prev2, 0)], np.nan)[use]
    gz = gen[use]
    # instruments: generation dummies, within-lab-generation input deviations (+ squares), lagged lab
    # inputs, twice-lagged lab output (valid: uncorrelated with xi_t, eps_t, eps_{t-1}), [current c]
    y2d = np.zeros_like(y2)
    for t in range(1, Tn):
        m = (gz == t) & np.isfinite(y2)
        if m.any():
            y2d[m] = y2[m] - y2[m].mean()
    cols = [Gz, ndev[use], ddev[use], ndev[use] ** 2, ddev[use] ** 2, ndev[use] * ddev[use],
            nbar[prev[use]] - S.n0, dbar[prev[use]] - S.d0, y2d]
    if not lag_only:
        cols.append(cbar[inv][use] - cbar.mean())
    Z = np.column_stack(cols)
    Z = Z[:, Z.std(axis=0) > 0]
    W = np.linalg.pinv(Z.T @ Z / Z.shape[0])
    Wh = np.linalg.cholesky(W + 1e-12 * np.eye(W.shape[0])).T
    t1 = gen == 0
    n_use = use.sum()

    def moments(q):
        lA, lB, al, be, rho = q[:5]
        dlt = np.r_[0.0, q[5:]]
        F, _ = F_J(S, lA, lB, al, be)
        v = S.y - F
        vbar = mean_of(v)
        xi = (v - dlt[gen])[use] - rho * (vbar[prev[use]] - dlt[gen[use] - 1])
        g = Z.T @ xi / n_use
        m0 = v[t1].mean() if t1.any() else 0.0
        return np.r_[Wh @ g, m0 * 3.0]

    q0 = np.r_[q_start[:4], 0.5, q_start[4:]]
    sol = ls(moments, q0, method="trf", x_scale="jac", max_nfev=400)
    return summarize_fit(S, *sol.x[:4])


def est_E7(S):
    """IV-path: allocation slope a (OLS of n on c, pooled; consistent under Hicks-neutral omega),
    gamma by 2SLS of y on c with generation effects, instrument = compute price p."""
    X = np.column_stack([np.ones(S.N), S.c])
    bN = np.linalg.lstsq(X, S.n, rcond=None)[0]
    a = bN[1]
    lnNstar = bN[0] + a * np.log(ml.C_REF)
    W = np.column_stack([np.ones(S.N), S.G])                       # exogenous regressors
    Zf = np.column_stack([W, S.p])
    pi = np.linalg.lstsq(Zf, S.c, rcond=None)[0]
    chat = Zf @ pi
    Xh = np.column_stack([W, chat])
    b = np.linalg.lstsq(Xh, S.y, rcond=None)[0]
    g = b[-1]
    al, be = g / a, g / (1 - a)
    out = dict(alpha=al, beta=be, a=a, gamma=g, sigma_star=2 / (2 + al + be),
               lnMstar=np.log(ml.C_REF / 6) - 2 * lnNstar)
    resid = S.y - np.column_stack([W, S.c]) @ b
    om = resid + W[:, 1:] @ b[1:-1]                              # add back generation effects
    out.update(tfp_stats(S, om))
    fs = np.linalg.lstsq(Zf, S.c, rcond=None)
    rss_u = np.sum((S.c - chat) ** 2)
    bw = np.linalg.lstsq(W, S.c, rcond=None)[0]
    rss_r = np.sum((S.c - W @ bw) ** 2)
    out["first_stage_F"] = (rss_r - rss_u) / (rss_u / (S.N - Zf.shape[1]))
    return out


def est_E8(S):
    """Heckman two-step: probit of release on trained-model inputs (generations >= 2, since every
    generation-1 model is released), exclusion = open-weight policy; IMR added to pooled NLS."""
    import statsmodels.api as sm
    Fu = S.full
    m2 = Fu["gen"] >= 1
    Tn = S.Tn
    Gd = np.column_stack([(Fu["gen"][m2] == t) for t in range(2, Tn)]).astype(float)
    Zs = np.column_stack([np.ones(m2.sum()), Gd, Fu["c_obs"][m2], Fu["n"][m2] - Fu["d_obs"][m2], Fu["open"][m2]])
    rel = Fu["released"][m2].astype(float)
    if rel.mean() > 0.995 or rel.mean() < 0.005:
        imr_all = np.zeros(len(Fu["gen"]))
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pr = sm.Probit(rel, Zs).fit(disp=0, maxiter=200)
        xb = Zs @ pr.params
        imr_all = np.zeros(len(Fu["gen"]))
        imr_all[np.where(m2)[0]] = norm.pdf(xb) / np.clip(norm.cdf(xb), 1e-10, None)
    imr = imr_all[Fu["released"]]

    def res(q):
        F, J = F_J(S, *q[:4])
        r = S.y - F - S.G @ q[4:4 + Tn - 1] - q[-1] * imr
        return r, np.column_stack([-J, -S.G, -imr])

    q0s = [np.r_[s0, np.zeros(Tn - 1), 0.0] for s0 in start_grid(S)]                  # v2: no truth start
    sol = ls_multi(lambda q: res(q)[0], lambda q: res(q)[1], q0s, method="lm")
    out = summarize_fit(S, *sol.x[:4])
    out["imr_coef"] = sol.x[-1]
    return out


def est_E9(S, q5, model_T=False):
    """System: within-lab-generation loss equation + FOC.  The training-only FOC (GNR/Approach-2 logic)
    says ln(alpha A/(beta B)) - alpha n + beta d = 0 for every model.  Its true value is ln w =
    ln(1 + T/(3D)) > 0 (the inference wedge), so ignoring T biases the system; E10 subtracts
    ln(1 + T_proxy/(3 D_obs)) with a noisy usage proxy for T."""
    wT = np.log1p(S.Tp / (3 * np.exp(S.d))) if model_T else 0.0

    def parts(q):
        lB, al, be = q
        ry, Jy = _labgen_res(S, q)
        # unnormalised lnA - lnB with lA = 0: lnA = al*n0, lnB = lB + be*d0
        k = np.log(al / be) + al * S.n0 - (lB + be * S.d0)
        rf = k - al * S.n + be * S.d - wT
        Jf = np.column_stack([-np.ones(S.N), 1 / al + S.n0 - S.n, -1 / be - S.d0 + S.d])
        return ry, Jy, rf, Jf

    ry, _, rf, _ = parts(q5)
    sy, sf = np.std(ry) + 1e-9, np.std(rf - rf.mean()) + 1e-9

    def res(q):
        ry, Jy, rf, Jf = parts(q)
        return np.r_[ry / sy, rf / sf], np.vstack([Jy / sy, Jf / sf])

    sol = ls(lambda q: res(q)[0], q5, jac=lambda q: res(q)[1], method="lm")
    out = summarize_fit(S, 0.0, *sol.x)
    return out


def run_estimators(Sd):
    S = Sample(Sd)
    out = {}
    if np.bincount(S.lg_i).max() < 2:          # flagships only: within-family estimators not identified
        na = dict(error="not identified (one model per lab-generation)")
        out["E1_pooled_huber"] = _safe(lambda: est_E1(S))
        out["E1b_pooled_huber_Etrue"] = _safe(lambda: est_E1b(S))
        r2 = _safe(lambda: est_E2(S))
        out["E2_pooled_nls"] = r2[0] if isinstance(r2, tuple) else r2
        out["E3_lab_fe"] = _safe(lambda: est_E3(S))
        out["E4_mundlak"] = _safe(lambda: est_E4(S))
        out["E7_iv_path"] = _safe(lambda: est_E7(S))
        for e in ("E5_labgen_fe", "E6_acf", "E6b_acf_lagonly", "E8_heckman", "E9_foc_system", "E10_foc_system_T"):
            out[e] = dict(na)
        return out, S
    safe = lambda f: _safe(f)
    out["E1_pooled_huber"] = safe(lambda: est_E1(S))
    out["E1b_pooled_huber_Etrue"] = safe(lambda: est_E1b(S))
    r2 = _safe(lambda: est_E2(S))
    if isinstance(r2, tuple):
        out["E2_pooled_nls"], q2 = r2
    else:
        out["E2_pooled_nls"], q2 = r2, np.r_[start_grid(S)[0], np.zeros(S.Tn - 1)]
    out["E3_lab_fe"] = safe(lambda: est_E3(S))
    out["E4_mundlak"] = safe(lambda: est_E4(S))
    r5 = _safe(lambda: est_E5(S))
    if isinstance(r5, tuple):
        out["E5_labgen_fe"], q5 = r5
    else:
        s0 = start_grid(S)[0]                                                        # v2: no truth start
        out["E5_labgen_fe"], q5 = r5, np.r_[s0[1] - s0[0], s0[2:]]
    out["E6_acf"] = safe(lambda: est_E6(S, q2))
    out["E6b_acf_lagonly"] = safe(lambda: est_E6(S, q2, lag_only=True))
    out["E7_iv_path"] = safe(lambda: est_E7(S))
    out["E8_heckman"] = safe(lambda: est_E8(S))
    out["E9_foc_system"] = safe(lambda: est_E9(S, q5))
    out["E10_foc_system_T"] = safe(lambda: est_E9(S, q5, model_T=True))
    return out, S


def _safe(f):
    try:
        return f()
    except Exception as e:  # a failed fit is recorded as missing (and counted in the memo)
        return dict(error=str(e)[:80])


BASE_SEED = 7_202_609
import os  # noqa: E402

R_IND = 20 if os.environ.get("M6_QUICK") == "1" else 400   # replications per industry scenario
CHUNK = 10 if os.environ.get("M6_QUICK") == "1" else 25


def run_chunk(args):
    si, chunk = args
    sc = SCENARIOS[si]
    rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, si, chunk]))
    rows = []
    for j in range(CHUNK):
        rep = chunk * CHUNK + j
        Sd = simulate(rng, sc)
        res, S = run_estimators(Sd)
        meta = dict(scenario=sc["name"], rep=rep, n_trained=len(Sd["y"]), n_released=S.N,
                    release_rate=float(Sd["released"].mean()),
                    corr_c_omega=float(np.corrcoef(Sd["c"], Sd["x"])[0, 1]),
                    mean_ln_w=float(np.mean(np.log1p(Sd["T"] / (3 * np.exp(Sd["d"]))))),
                    sample_sd_x=float(np.std(Sd["x"][Sd["released"]], ddof=1)))
        for e, r in res.items():
            row = dict(meta, estimator=e)
            row.update(r)
            rows.append(row)
    return rows


def tasks():
    return [(si, ch) for si in range(len(SCENARIOS)) for ch in range(R_IND // CHUNK)]


def summarize(df):
    tr = truth_row()
    rows = []
    for sc in SCENARIOS:
        for e in ESTIMATORS:
            g = df[(df.scenario == sc["name"]) & (df.estimator == e)]
            if g.empty:
                continue
            for k in OUT_PARAMS:
                if k not in g:
                    continue
                x = g[k].to_numpy(float)
                ok = np.isfinite(x)
                if not ok.any():
                    continue
                err = x[ok] - tr[k]
                rows.append(dict(scenario=sc["name"], scenario_label=sc["label"], estimator=e,
                                 estimator_label=EST_LABEL[e], param=k, truth=tr[k], R=int(ok.sum()),
                                 fail_share=float(1 - ok.mean()), mean=float(x[ok].mean()), median=float(np.median(x[ok])),
                                 bias=float(err.mean()), mc_se_bias=float(x[ok].std(ddof=1) / np.sqrt(ok.sum())),
                                 median_bias=float(np.median(err)), sd=float(x[ok].std(ddof=1)),
                                 rmse=float(np.sqrt(np.mean(err ** 2))), mae=float(np.median(np.abs(err)))))
    return pd.DataFrame(rows)
