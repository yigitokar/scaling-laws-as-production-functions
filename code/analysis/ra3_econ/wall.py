"""wall.py -- the data wall: cost of a binding cap on unique tokens under Chinchilla-family technologies combined with
Muennighoff et al.'s (2023) repetition model.

Technology (kappa family; kappa = 1 is Chinchilla):   L = E + [A N'^(-a1) + B D'^(-b1)]^kappa,
effective data (Muennighoff eq. 5):                   D' = U + U R*_D (1 - exp(-R/R*_D)),  R = max(D/U - 1, 0),
effective parameters (eq. 6, robustness only):        N' = U_N + U_N R*_N (1 - exp(-R_N/R*_N)), R_N = max(N/U_N - 1, 0),
    with U_N the technology's compute-optimal N for a run whose compute-optimal data equal U (their eq. 17).
D is tokens processed (with repetition); U unique tokens. Cost C = 6 N D (FLOP-accounting approximation).

Objects (all at the unconstrained frontier loss l*(C) unless stated):
  penalty      pi = C_U(l*(C))/C - 1: extra compute needed with U unique tokens to reach the unconstrained loss;
  C_eq/C       compute-equivalent of the constrained loss at C (share of compute 'lost' to the wall);
  lambda_U     shadow value of a unique token, -dC/dU at fixed loss (FLOP per token); by the envelope theorem
               lambda_U = 6 N [(1 + R*)(e^(R/R*) - 1) - R] under D'-only repetition (checked numerically);
  sigma_CU     elasticity of substitution between compute C and unique data U along the isoquant, d ln(C/U)/d ln lambda;
  gamma_eff    elasticity of reducible loss w.r.t. compute at fixed U (returns to compute behind the wall).
Scarcity is indexed by r = D*(C)/U. For the kappa family with D' homogeneous of degree one in (D, U), pi, C_eq/C,
lambda_U/(6N), sigma_CU and gamma_eff depend on r only (quasi-homotheticity); run.py checks this at 1e25 and 1e27.

Technologies: (1) kappa = 1 Chinchilla refit (m1, sigma* = 0.737); (2) kappa-free refit (m2 q family, sigma* = 0.701);
(3) members of (1)'s observationally equivalent family (m7 Prop. 2(4): a1 = (1-a)S, b1 = aS, kappa = gamma/(a(1-a)S),
(A, B) matched to (G, K)) with sigma* = 0.60 and other values: same expansion path and frontier, different curvature.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar

from ra3common import M1_PROC, M1_REG, M2_PROC, M2_REG, MUENN


class KTech:
    def __init__(self, key, label, E, A, B, a1, b1, kappa=1.0):
        self.key, self.label = key, label
        self.E, self.A, self.B, self.a1, self.b1, self.kappa = E, A, B, a1, b1, kappa

    # path and frontier
    @property
    def a(self):
        return self.b1 / (self.a1 + self.b1)

    @property
    def S(self):
        return self.a1 + self.b1

    @property
    def sigma_star(self):
        return 2.0 / (2.0 + self.S)

    @property
    def gamma(self):
        return self.kappa * self.a1 * self.b1 / self.S

    @property
    def lnG(self):
        return (np.log(self.a1 * self.A) - np.log(self.b1 * self.B)) / self.S

    @property
    def K(self):
        return (self.S / self.b1 * self.A * np.exp(-self.a1 * self.lnG)) ** self.kappa

    def N_opt(self, C):
        return np.exp(self.lnG + self.a * np.log(C / 6.0))

    def D_opt(self, C):
        return C / (6.0 * self.N_opt(C))

    def L_opt(self, C):
        return self.E + self.K * (C / 6.0) ** (-self.gamma)

    def C_of_L(self, L):
        return 6.0 * (self.K / (L - self.E)) ** (1.0 / self.gamma)

    def C_for_D(self, D):
        """Compute at which the compute-optimal data equal D."""
        return 6.0 * np.exp((np.log(D) + self.lnG) / (1 - self.a))

    def loss(self, Np, Dp):
        return self.E + (self.A * Np ** (-self.a1) + self.B * Dp ** (-self.b1)) ** self.kappa

    def equivalent(self, sigma, key=None, label=None):
        """Member of this technology's observationally equivalent family with on-path elasticity sigma."""
        a, gam, lnG, K = self.a, self.gamma, self.lnG, self.K
        S = 2.0 / sigma - 2.0
        a1, b1 = (1 - a) * S, a * S
        kap = gam / (a * (1 - a) * S)
        A = K ** (1 / kap) * b1 / S * np.exp(a1 * lnG)
        B = a1 * A / (b1 * np.exp(S * lnG))
        return KTech(key or f"eq{sigma:.2f}", label or f"sigma* = {sigma:.2f} (equivalent to {self.key})",
                     self.E, A, B, a1, b1, kap)


def load_wall_techs():
    r1 = pd.read_csv(M1_REG).set_index("row_id")
    c = r1.loc["chin_n240_huber"]
    k1 = KTech("k1", "Chinchilla refit, kappa = 1", c.E, c.A, c.B, c.alpha, c.beta, 1.0)
    r2 = pd.read_csv(M2_REG)
    q = r2[(r2.dataset == "chinchilla") & (r2.subset == "all") & (r2.estimator == "huber_q")].iloc[0]
    kq = KTech("kq", "Chinchilla refit, kappa free", q.E, q.A, q.B, q.alpha, q.beta, q.q)
    return k1, kq


def draws_k1():
    x = np.load(os.path.join(M1_PROC, "boot_chinchilla_n240_huber_pairs.npy"))[:, :5]   # lnA, lnB, lnE, alpha, beta
    return [KTech("k1", "", np.exp(r[2]), np.exp(r[0]), np.exp(r[1]), r[3], r[4], 1.0) for r in x
            if r[3] > 0 and r[4] > 0]


def draws_kq():
    x = np.load(os.path.join(M2_PROC, "boot", "chinchilla__all__huber_q.npy"))         # E, A, B, alpha, beta, q
    return [KTech("kq", "", r[0], r[1], r[2], r[3], r[4], r[5]) for r in x
            if r[3] > 0 and r[4] > 0 and r[1] > 0 and r[2] > 0 and r[5] > 0]


# ----------------------------------------------------------------------------- repetition specifications
SPECS = {
    "D15": dict(label="D' only, R*_D = 15.4 (primary)", RD=MUENN["RD"], RN=None),
    "DN": dict(label="D' and N' (R*_D = 15.4, R*_N = 5.3)", RD=MUENN["RD"], RN=MUENN["RN"]),
    "D3": dict(label="D' only, R*_D = 2.9 (D-only fit)", RD=MUENN["RD_only"], RN=None),
    "nore": dict(label="No repetition (D <= U)", RD=0.0, RN=None),
}


def Dp_of_D(D, U, RD):
    if D <= U:
        return D
    if RD <= 0:
        return U
    return U + U * RD * (1 - np.exp(-(D / U - 1) / RD))


def D_of_Dp(Dp, U, RD):
    if Dp <= U:
        return Dp
    if RD <= 0 or Dp >= U * (1 + RD):
        return np.inf
    return U * (1 - RD * np.log(1 - (Dp / U - 1) / RD))


def Np_of_N(N, UN, RN):
    if RN is None or N <= UN:
        return N
    return UN + UN * RN * (1 - np.exp(-(N / UN - 1) / RN))


def _UN(t, U):
    return t.N_opt(t.C_for_D(U))


def cost_at_loss(t, L, U, spec):
    """min 6 N D s.t. loss(N', D'(D; U)) <= L. Returns (C, N, D) at the optimum."""
    sp = SPECS[spec]
    Q = (L - t.E) ** (1 / t.kappa)
    UN = _UN(t, U) if sp["RN"] is not None else None

    def cost(lnN):
        N = np.exp(lnN)
        Np = Np_of_N(N, UN, sp["RN"])
        v = Q - t.A * Np ** (-t.a1)
        if v <= 0:
            return np.inf
        Dp = (v / t.B) ** (-1 / t.b1)
        D = D_of_Dp(Dp, U, sp["RD"])
        return np.log(6 * N * D) if np.isfinite(D) else np.inf

    lnN_min = np.log((t.A / Q) ** (1 / t.a1)) + 1e-9
    grid = np.linspace(lnN_min, lnN_min + 30, 3001)
    vals = np.array([cost(x) for x in grid])
    if not np.isfinite(vals).any():
        return np.inf, np.nan, np.nan
    i = int(np.nanargmin(vals))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    res = minimize_scalar(cost, bounds=(lo, hi), method="bounded", options=dict(xatol=1e-10))
    lnN = res.x if res.fun <= vals[i] else grid[i]
    lnC = min(res.fun, vals[i])
    N = np.exp(lnN)
    return float(np.exp(lnC)), float(N), float(np.exp(lnC) / (6 * N))


def loss_at_compute(t, C, U, spec):
    """min loss(N', D'(C/(6N); U)) over N at compute C."""
    sp = SPECS[spec]
    UN = _UN(t, U) if sp["RN"] is not None else None

    def f(lnN):
        N = np.exp(lnN)
        D = C / (6 * N)
        return t.loss(Np_of_N(N, UN, sp["RN"]), Dp_of_D(D, U, sp["RD"]))

    c = np.log(t.N_opt(C))
    grid = np.linspace(c - 3, c + 9, 1201)
    vals = np.array([f(x) for x in grid])
    i = int(np.argmin(vals))
    res = minimize_scalar(f, bounds=(grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]), method="bounded",
                          options=dict(xatol=1e-10))
    return float(min(res.fun, vals[i]))


def shadow_closed(N, D, U, RD):
    """Envelope-theorem shadow value of a unique token under D'-only repetition (FLOP per token)."""
    R = max(D / U - 1, 0.0)
    return 6 * N * ((1 + RD) * (np.exp(R / RD) - 1) - R)


def wall_point(t, C, r, spec, h=1e-3, extras=True):
    """All wall objects at compute C and scarcity r = D*(C)/U."""
    U = t.D_opt(C) / r
    L0 = t.L_opt(C)
    CU, N, D = cost_at_loss(t, L0, U, spec)
    out = dict(tech=t.key, spec=spec, C=C, r=r, U=U, Dstar=t.D_opt(C), Nstar=t.N_opt(C), penalty=CU / C - 1,
               N_U=N, D_U=D, epochs=D / U, M_U=D / N, N_ratio=N / t.N_opt(C))
    # shadow value: numerical derivative in ln U at fixed loss
    Cp, _, _ = cost_at_loss(t, L0, U * np.exp(h), spec)
    Cm, _, _ = cost_at_loss(t, L0, U * np.exp(-h), spec)
    lam = -(Cp - Cm) / (U * (np.exp(h) - np.exp(-h)))
    out.update(shadow_flop=lam, shadow_rel=lam / (6 * N), elas_C_U=-(np.log(Cp) - np.log(Cm)) / (2 * h))
    if SPECS[spec]["RN"] is None and SPECS[spec]["RD"] > 0:
        out["shadow_flop_closed"] = shadow_closed(N, D, U, SPECS[spec]["RD"])
    if extras:
        LU = loss_at_compute(t, C, U, spec)
        out["dloss"] = LU - L0
        out["Ceq_share"] = t.C_of_L(LU) / C
        Lp = loss_at_compute(t, C * np.exp(h), U, spec)
        Lm = loss_at_compute(t, C * np.exp(-h), U, spec)
        out["gamma_eff"] = -(np.log(Lp - t.E) - np.log(Lm - t.E)) / (2 * h)
        out["gamma_ratio"] = out["gamma_eff"] / t.gamma
    return out


def sigma_CU(t, C, r, spec, h=0.02):
    """Elasticity of substitution between compute and unique data along the isoquant l*(C), at U = D*(C)/r."""
    L0 = t.L_opt(C)
    U = t.D_opt(C) / r
    vals = []
    for s in (-1, 1):
        Us = U * np.exp(s * h)
        c0, _, _ = cost_at_loss(t, L0, Us, spec)
        cp, _, _ = cost_at_loss(t, L0, Us * np.exp(1e-3), spec)
        cm, _, _ = cost_at_loss(t, L0, Us * np.exp(-1e-3), spec)
        lam = -(cp - cm) / (Us * (np.exp(1e-3) - np.exp(-1e-3)))
        vals.append((np.log(c0 / Us), np.log(lam)))
    (x1, y1), (x2, y2) = vals
    return (x2 - x1) / (y2 - y1)


def wedge_data_multiple(w, sigma):
    """Data demanded at given compute relative to D*(C) for a model with wedge w: (M/M*)^(1/2) = w^(sigma/(2(1-sigma)))."""
    return np.asarray(w, float) ** (sigma / (2 * (1 - sigma)))
