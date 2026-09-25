"""sl.py: shared estimation library for "Scaling Laws as Production Functions".

Technology (Hoffmann et al. 2022, Eq. 2):   L(N, D) = E + A N^-alpha + B D^-beta
Parameter vector used internally (log-sum-exp parameterization of Hoffmann et al. App. D.2):
    theta = (a, b, e, alpha, beta) with A = exp(a), B = exp(b), E = exp(e)

Economic objects (see paper Section II):
    u = A N^-alpha, v = B D^-beta, R = u + v (reducible loss)
    output elasticities of reducible loss   eps_N = alpha u / R,  eps_D = beta v / R
    local elasticity of substitution        sigma = (alpha u + beta v) / (alpha u (1+beta) + beta v (1+alpha))
    compute-optimal (C = 6ND) expansion path N* = G (C/6)^a_N,  a_N = beta/(alpha+beta),
                                             G = (alpha A / (beta B))^(1/(alpha+beta))
    frontier / inverse cost function        L*(C) = E + K (C/6)^-gamma, gamma = alpha beta/(alpha+beta)
    elasticity of substitution on the path  sigma* = 2/(2+alpha+beta)
    inference wedge                         w = eps_N/eps_D = 1 + T/(3D)   (lifetime / training compute)
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import huber as _huber

# ----------------------------------------------------------------------------- technology

@dataclass
class Chinchilla:
    E: float
    A: float
    B: float
    alpha: float
    beta: float
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_theta(cls, th, **extra):
        a, b, e, al, be = th
        return cls(E=float(np.exp(e)), A=float(np.exp(a)), B=float(np.exp(b)), alpha=float(al), beta=float(be), extra=extra)

    @property
    def theta(self):
        return np.array([np.log(self.A), np.log(self.B), np.log(self.E), self.alpha, self.beta])

    # --- primal
    def loss(self, N, D):
        return self.E + self.A * np.asarray(N, float) ** -self.alpha + self.B * np.asarray(D, float) ** -self.beta

    def uv(self, N, D):
        return self.A * np.asarray(N, float) ** -self.alpha, self.B * np.asarray(D, float) ** -self.beta

    def elasticities(self, N, D):
        u, v = self.uv(N, D)
        R = u + v
        return self.alpha * u / R, self.beta * v / R

    def sigma(self, N, D):
        u, v = self.uv(N, D)
        al, be = self.alpha, self.beta
        return (al * u + be * v) / (al * u * (1 + be) + be * v * (1 + al))

    # --- dual (training-only cost C = 6ND)
    @property
    def a_N(self):
        return self.beta / (self.alpha + self.beta)

    @property
    def gamma(self):
        return self.alpha * self.beta / (self.alpha + self.beta)

    @property
    def G(self):
        return (self.alpha * self.A / (self.beta * self.B)) ** (1.0 / (self.alpha + self.beta))

    @property
    def K(self):
        return (self.alpha + self.beta) / self.beta * self.A * self.G ** -self.alpha

    @property
    def sigma_star(self):
        return 2.0 / (2.0 + self.alpha + self.beta)

    def N_opt(self, C):
        return self.G * (np.asarray(C, float) / 6.0) ** self.a_N

    def D_opt(self, C):
        return (np.asarray(C, float) / 6.0) ** (1 - self.a_N) / self.G

    def L_opt(self, C):
        return self.E + self.K * (np.asarray(C, float) / 6.0) ** -self.gamma

    def C_min(self, L):
        """Minimal training compute that attains loss L (inverse of L_opt)."""
        L = np.asarray(L, float)
        return 6.0 * (self.K / (L - self.E)) ** (1.0 / self.gamma)

    # --- wedges
    def wedge(self, N, D):
        """w = eps_N/eps_D; equals (training+inference compute)/(training compute) at a cost optimum."""
        eN, eD = self.elasticities(N, D)
        return eN / eD

    def implied_inference_tokens(self, N, D):
        """T = 3 D (w - 1): lifetime inference tokens that rationalize (N, D) under min 6ND + 2NT."""
        return 3.0 * np.asarray(D, float) * (self.wedge(N, D) - 1.0)

    def cost_efficiency(self, N, D):
        """Farrell input-oriented efficiency relative to the training-only frontier: C_min(L(N,D)) / (6ND)."""
        return self.C_min(self.loss(N, D)) / (6.0 * np.asarray(N, float) * np.asarray(D, float))

    def summary(self):
        return dict(E=self.E, A=self.A, B=self.B, alpha=self.alpha, beta=self.beta, a_N=self.a_N,
                    gamma=self.gamma, sigma_star=self.sigma_star, G=self.G, K=self.K)


HOFFMANN = Chinchilla(E=1.6934, A=406.4, B=410.7, alpha=0.3392, beta=0.2849)       # Approach 3, TeX precision
BESIROGLU = Chinchilla(E=1.8172, A=482.01, B=2085.43, alpha=0.3478, beta=0.3658)   # Epoch replication, published values
# NB: Besiroglu et al.'s published values come from a Huber log-likelihood with a free scale (effectively LAD);
# fit_chinchilla(delta=1e-3) on their n=240 sample returns the exact Huber(1e-3) optimum instead
# (E=1.8172, A=477.8, B=2143.4, alpha=0.3473, beta=0.3672), which agrees to 3-4 digits (see output/memos/m1_chinchilla.md).

# ----------------------------------------------------------------------------- estimation

def _lse_pred(th, lN, lD):
    a, b, e, al, be = th
    return np.logaddexp(np.logaddexp(a - al * lN, b - be * lD), e)


def _obj(th, lN, lD, lL, delta, w):
    r = _lse_pred(th, lN, lD) - lL
    if delta is None:
        return 0.5 * np.sum(w * r ** 2)
    return np.sum(w * _huber(delta, r))


def _grad(th, lN, lD, lL, delta, w):
    a, b, e, al, be = th
    t1, t2, t3 = a - al * lN, b - be * lD, e * np.ones_like(lN)
    m = np.maximum(np.maximum(t1, t2), t3)
    p1, p2, p3 = np.exp(t1 - m), np.exp(t2 - m), np.exp(t3 - m)
    s = p1 + p2 + p3
    p1, p2, p3 = p1 / s, p2 / s, p3 / s
    r = m + np.log(s) - lL
    dr = r if delta is None else np.clip(r, -delta, delta)
    dr = w * dr
    return np.array([np.sum(dr * p1), np.sum(dr * p2), np.sum(dr * p3),
                     np.sum(dr * p1 * -lN), np.sum(dr * p2 * -lD)])


DEFAULT_GRID = dict(a=np.arange(0, 30, 5), b=np.arange(0, 30, 5), e=np.arange(-1, 1.5, 0.5),
                    alpha=np.arange(0, 2.5, 0.5), beta=np.arange(0, 2.5, 0.5))
FAST_GRID = dict(a=np.arange(0, 30, 7.5), b=np.arange(0, 30, 7.5), e=np.array([-1.0, 0.0, 0.5]),
                 alpha=np.array([0.2, 0.5, 1.0]), beta=np.array([0.2, 0.5, 1.0]))


def fit_chinchilla(N, D, L, delta=1e-3, weights=None, grid=None, init=None, E_fixed=None, return_all=False):
    """Hoffmann/Besiroglu estimator: Huber(delta) on log-loss residuals of the LSE parameterization,
    L-BFGS-B from a grid of initial values (delta=None gives Gaussian NLS on log loss).
    E_fixed: optionally hold E fixed (profile)."""
    lN, lD, lL = np.log(np.asarray(N, float)), np.log(np.asarray(D, float)), np.log(np.asarray(L, float))
    w = np.ones_like(lL) if weights is None else np.asarray(weights, float)
    grid = grid or DEFAULT_GRID
    starts = [init] if init is not None else list(itertools.product(grid["a"], grid["b"], grid["e"], grid["alpha"], grid["beta"]))
    best = None
    for st in starts:
        st = np.array(st, float)
        if E_fixed is not None:
            e0 = np.log(E_fixed)
            f = lambda th4: _obj(np.r_[th4[:2], e0, th4[2:]], lN, lD, lL, delta, w)
            g = lambda th4: np.delete(_grad(np.r_[th4[:2], e0, th4[2:]], lN, lD, lL, delta, w), 2)
            r = minimize(f, np.delete(st, 2), jac=g, method="L-BFGS-B",
                         options=dict(maxiter=10000, ftol=1e-15, gtol=1e-12))
            x = np.r_[r.x[:2], e0, r.x[2:]]
        else:
            r = minimize(_obj, st, args=(lN, lD, lL, delta, w), jac=_grad, method="L-BFGS-B",
                         options=dict(maxiter=10000, ftol=1e-15, gtol=1e-12))
            x = r.x
        if best is None or r.fun < best[1]:
            best = (x, r.fun)
    m = Chinchilla.from_theta(best[0], objective=float(best[1]), n=len(lL), delta=delta)
    return m


def bootstrap(fn, data: dict, B=200, cluster=None, seed=0):
    """Nonparametric (pairs or cluster) bootstrap of a fit function fn(**data) -> Chinchilla or dict."""
    rng = np.random.default_rng(seed)
    n = len(next(iter(data.values())))
    out = []
    if cluster is None:
        for _ in range(B):
            idx = rng.integers(0, n, n)
            out.append(fn(**{k: np.asarray(v)[idx] for k, v in data.items()}))
    else:
        cl = np.asarray(cluster)
        groups = [np.where(cl == g)[0] for g in np.unique(cl)]
        for _ in range(B):
            pick = rng.integers(0, len(groups), len(groups))
            idx = np.concatenate([groups[i] for i in pick])
            out.append(fn(**{k: np.asarray(v)[idx] for k, v in data.items()}))
    return out


def summarize_boot(models, keys=("E", "A", "B", "alpha", "beta", "a_N", "gamma", "sigma_star")):
    arr = {k: np.array([m.summary()[k] if isinstance(m, Chinchilla) else m[k] for m in models]) for k in keys}
    return {k: dict(se=float(np.std(v, ddof=1)), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)))
            for k, v in arr.items()}


# ----------------------------------------------------------------------------- misc helpers

def chinchilla_extraction(path="data/raw/epoch_chinchilla/svg_extracted_data.csv", drop_worst=5):
    """Epoch's digitized Chinchilla Figure-4 sample (Besiroglu et al. 2024). drop_worst=5 reproduces
    their n=240 estimation sample (they drop the 5 highest-loss points)."""
    import pandas as pd
    df = pd.read_csv(path)
    df = df.rename(columns={"Model Size": "N", "Training FLOP": "C", "loss": "L"})
    df["D"] = df["C"] / (6 * df["N"])
    df = df[["N", "D", "C", "L"]].dropna().sort_values("L").reset_index(drop=True)
    if drop_worst:
        df = df.iloc[:-drop_worst].reset_index(drop=True)
    return df
