"""theory_checks.py -- numerical verification of the flexible-input (inefficiency) bias formula for the
IsoFLOP allocation exponent, and comparison with the parametric (Approach-3) pseudo-true values.

Proposition (first-order, verified here). Let the observed loss be L_obs(N, D) = L*(N, D) exp(u(N, D)) with
L* = E + A N^-alpha + B D^-beta the concentrated technology and u >= 0 the inefficiency of the flexible-input
policy. On the IsoFLOP C = 6ND the researcher's argmin satisfies
        alpha U - beta V = L (u_n - u_d),        U = A N^-alpha, V = B D^-beta, u_n = du/dlnN, u_d = du/dlnD,
so to first order  ln N*_obs - ln N* = -z/(alpha+beta),  z = L (u_n - u_d)/(gamma R),  and
        a_obs - a = -(1/(alpha+beta)) dz/dlnC.
With constant log-gradients (u_n, u_d) this is  a_obs - a = -(u_n - u_d) E / ((alpha+beta) R*(C)).
Only the transverse gradient u_n - u_d matters: inefficiency that depends on (N, D) only through C (u_n = u_d)
leaves the IsoFLOP exponent unbiased. The same gradients move the parametric (Approach-3) fit differently.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from common import sl, fit_efixed

CASES = [(0.003, -0.003), (0.003, 0.0), (0.0, -0.003), (0.003, 0.003), (-0.003, -0.003), (0.0, 0.006), (0.01, -0.01)]


def isoflop_bias_sim(tech=sl.BESIROGLU, Crange=(1e19, 1e22), n0=np.log(4e8), d0=np.log(2.5e10)):
    rows = []
    Cs = np.logspace(np.log10(Crange[0]), np.log10(Crange[1]), 13)
    Rbar = tech.L_opt(np.exp(np.log(Cs).mean())) - tech.E
    for un, ud in CASES:
        Lobs = lambda N, D: tech.loss(N, D) * np.exp(un * (np.log(N) - n0) + ud * (np.log(D) - d0))
        ns = []
        for C in Cs:
            r = minimize_scalar(lambda x: Lobs(np.exp(x), C / 6 / np.exp(x)),
                                bounds=(np.log(1e6), np.log(C / 6) - np.log(1e6)), method="bounded",
                                options=dict(xatol=1e-11))
            ns.append(r.x)
        a_iso = np.polyfit(np.log(Cs), ns, 1)[0]
        pred = tech.a_N - (un - ud) * tech.E / (Rbar * (tech.alpha + tech.beta))
        # parametric fit on a 7x7 factorial design around (n0, d0)
        NN, DD = np.meshgrid(np.exp(np.linspace(n0 - 1.5, n0 + 1.5, 7)), np.exp(np.linspace(d0 - 1.5, d0 + 1.5, 7)))
        N, D = NN.ravel(), DD.ravel()
        L = Lobs(N, D)
        mE = fit_efixed(N, D, L, tech.E, delta=None, init=tech.theta, grid=False)
        mF = sl.fit_chinchilla(N, D, L, delta=None, init=tech.theta)
        rows.append(dict(u_n=un, u_d=ud, a_true=tech.a_N, a_isoflop_sim=a_iso, a_isoflop_formula=pred,
                         a_param_Efixed=mE.a_N, sigma_param_Efixed=mE.sigma_star, a_param_Efree=mF.a_N,
                         sigma_param_Efree=mF.sigma_star, E_param_Efree=mF.E, sigma_true=tech.sigma_star))
    return pd.DataFrame(rows)
