"""embed.py -- bias in the allocation exponent a from measuring N without a component whose share falls
with scale (Pearce & Song 2024 mechanism; Porian et al. 2024 last-layer FLOPs).

Setup (model_spec.md notation). True technology L = E + A N^-alpha + B D^-beta in *true* N. The researcher
measures N_m = N - X(N_m) (X = embedding or head parameters, share s = X/N, scale elasticity
theta = dln X / dln N_m < 1) and compute C_m = 6 N_m D, and chooses the IsoFLOP argmin in the measured
units (Kaplan's procedure). Result (derived below, verified symbolically and numerically):

    kappa(N_m) = dln N / dln N_m = 1 - s (1 - theta)                      (in (theta, 1])
    researcher's FOC:  alpha u kappa = beta v      (u = A N^-alpha, v = B D^-beta)
    local measured allocation exponent
        a_m = dln N_m* / dln C_m = beta / (beta + alpha kappa - eta),     eta = dln kappa / dln N_m >= 0
    => a_m > a = beta/(alpha+beta) whenever s > 0 and theta < 1;  a_m -> a as s -> 0 (large scale);
       a_m -> beta/(beta + alpha theta) as s -> 1 (Pearce-Song's small-scale limit beta/(beta+alpha/3)).
    A proportional error (X = c N_m, theta = 1) leaves a unchanged (it only moves the level G): the bias is
    entirely due to the *regressor-correlated* part of the measurement error (Collard-Wexler & De Loecker 2016).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import brentq, minimize_scalar

from common import sl

OMEGA_PS = 47491.0      # Pearce & Song (2024) Eq. 11 fit to Chinchilla configurations: N_T = N_E + omega N_E^(1/3)
THETA_PS = 1.0 / 3.0
KAPLAN_RANGE = (790.0, 1.58e9)   # Kaplan's non-embedding N range as used by Pearce & Song


# ----------------------------------------------------------------------------- symbolic verification

def symbolic_check():
    """Implicit differentiation of the researcher's FOC with sympy; returns the simplified difference
    between the implicit derivative dn_m/dc_m and the closed form beta/(beta + alpha kappa - eta) (should be 0)."""
    x, c, A, B, al, be, om, th = sp.symbols("x c A B alpha beta omega theta", positive=True)
    Nm = sp.exp(x)
    N = Nm + om * Nm ** th
    D = sp.exp(c) / (6 * Nm)
    Lx = A * N ** (-al) + B * D ** (-be)
    foc = sp.diff(Lx, x)                                   # researcher's FOC along measured IsoFLOP
    dxdc = -sp.diff(foc, c) / sp.diff(foc, x)              # implicit function theorem
    s = om * Nm ** th / N
    kappa = 1 - s * (1 - th)
    eta = sp.diff(sp.log(kappa), x)
    u, v = A * N ** (-al), B * D ** (-be)
    # impose the FOC alpha u kappa = beta v by substituting B from it
    Bsol = sp.solve(sp.Eq(al * u * kappa, be * v), B)[0]
    closed = be / (be + al * kappa - eta)
    diff = sp.simplify((dxdc - closed).subs(B, Bsol))
    # numeric spot check at random parameter values (sympy simplify may not reach 0 symbolically)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(20):
        vals = {A: 10 ** rng.uniform(1, 3), al: rng.uniform(0.2, 0.5), be: rng.uniform(0.2, 0.5),
                om: 10 ** rng.uniform(2, 5), th: rng.uniform(0.1, 0.9), x: rng.uniform(5, 20), c: 0}
        Bv = float(Bsol.subs(vals))
        vals[B] = Bv
        vals[c] = 0.0
        lhs = float(dxdc.subs(vals))
        rhs = float(closed.subs(vals))
        worst = max(worst, abs(lhs - rhs))
    return str(diff), worst


# ----------------------------------------------------------------------------- closed form and simulation

def closed_form_path(tech, omega, theta, Nm):
    """Local measured exponent a_m along the researcher's (mismeasured) expansion path at measured size Nm."""
    Nm = np.asarray(Nm, float)
    X = omega * Nm ** theta
    N = Nm + X
    s = X / N
    kappa = 1 - s * (1 - theta)
    # d kappa/dln Nm = -(1-theta) ds/dx, ds/dx = s (theta - kappa)
    eta = (1 - theta) * s * (kappa - theta) / kappa
    al, be = tech.alpha, tech.beta
    a_m = be / (be + al * kappa - eta)
    a_m_noeta = be / (be + al * kappa)
    return pd.DataFrame(dict(Nm=Nm, N=N, s=s, kappa=kappa, eta=eta, a_m=a_m, a_m_noeta=a_m_noeta,
                             a_true=be / (al + be)))


def researcher_argmin(tech, omega, theta, Cm):
    """Brute-force check: researcher's IsoFLOP argmin in measured units, min over Nm of L(N(Nm), Cm/(6 Nm))."""
    out = []
    for C in np.atleast_1d(Cm):
        f = lambda x: np.log(tech.loss(np.exp(x) + omega * np.exp(x) ** theta, C / (6 * np.exp(x))) - tech.E)
        r = minimize_scalar(f, bounds=(0.0, np.log(C / 6)), method="bounded", options=dict(xatol=1e-10))
        out.append(np.exp(r.x))
    return np.array(out)


def researcher_path(tech, omega, theta, Nm):
    """Exact researcher expansion path: for each measured size Nm, the measured budget Cm at which Nm is the
    measured-IsoFLOP argmin (solves alpha u kappa = beta v for D; closed form)."""
    Nm = np.asarray(Nm, float)
    X = omega * Nm ** theta
    N = Nm + X
    kappa = 1 - (X / N) * (1 - theta)
    u = tech.A * N ** -tech.alpha
    v = tech.alpha * u * kappa / tech.beta
    D = (tech.B / v) ** (1 / tech.beta)
    return 6 * Nm * D


def simulate_local_exponent(tech, omega, theta, Nm_range, n_grid=200):
    """Researcher path over Nm_range: global log-log slope of Nm* on Cm (Pearce-Song's 'local power law'),
    pointwise numerical derivative, and a brute-force argmin check at 5 budgets."""
    xs = np.linspace(np.log(Nm_range[0]), np.log(Nm_range[1]), n_grid)
    cs = np.log(researcher_path(tech, omega, theta, np.exp(xs)))
    slope = np.polyfit(cs, xs, 1)[0]
    deriv = np.gradient(xs, cs)
    chk_idx = np.linspace(10, n_grid - 10, 5).astype(int)
    brute = researcher_argmin(tech, omega, theta, np.exp(cs[chk_idx]))
    brute_err = float(np.max(np.abs(np.log(brute) - xs[chk_idx])))
    return slope, pd.DataFrame(dict(Nm=np.exp(xs), Cm=np.exp(cs), a_local_numeric=deriv)), brute_err


def pearce_song_table():
    """Local measured exponent over Kaplan's range for the three canonical parameter sets."""
    techs = {"Besiroglu et al. (2024)": sl.BESIROGLU,
             "Hoffmann et al. (2022), TeX precision": sl.HOFFMANN,
             "Hoffmann et al. (2022), rounded": sl.Chinchilla(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28)}
    rows, curves = [], []
    for name, t in techs.items():
        slope, cur, brute_err = simulate_local_exponent(t, OMEGA_PS, THETA_PS, KAPLAN_RANGE)
        cf = closed_form_path(t, OMEGA_PS, THETA_PS, cur.Nm.values)
        cur = pd.concat([cur, cf.drop(columns="Nm")], axis=1)
        cur["param_set"] = name
        curves.append(cur)
        rows.append(dict(param_set=name, alpha=t.alpha, beta=t.beta, a_true=t.a_N, a_kaplan_range=slope,
                         a_small_limit=t.beta / (t.beta + t.alpha * THETA_PS),
                         max_abs_err_closed_form=float(np.max(np.abs(cf.a_m.values - cur.a_local_numeric.values)[2:-2])),
                         max_abs_err_bruteforce_lnN=brute_err))
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True)


# ----------------------------------------------------------------------------- Chinchilla extraction reanalysis

def nonembed_from_total(NT, omega=OMEGA_PS, theta=THETA_PS):
    """Invert N_T = N_E + omega N_E^theta (Pearce-Song) for the non-embedding count N_E."""
    out = []
    for n in np.asarray(NT, float):
        g = lambda y, n=n: np.exp(y) + omega * np.exp(y) ** theta - n
        out.append(np.exp(brentq(g, 0.0, np.log(n))))
    return np.array(out)
