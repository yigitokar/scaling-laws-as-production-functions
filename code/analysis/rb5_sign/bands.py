"""bands.py -- extrapolated compute-optimal paths with simultaneous confidence bands (fix list S1, decision D-3).

A study s contributes the path ln M*(c) = a_{s,g} + e_s (c - c_J), fitted by OLS to the logged IsoFLOP minima at its
bracketed budgets; g indexes the study's corpora (Marin: DCLM, Nemotron-CC and Comma share one slope and have their
own levels, as in Section III, and share one residual variance; the other studies have one corpus). c_J is the
study's largest bracketed budget.

Inference: a joint wild bootstrap-t of (a, e). Residuals are rescaled by 1/sqrt(1 - h_ii) and multiplied by Webb's
six-point weights; in each draw the path is refitted and studentized by the within-study pooled residual variance
(classical OLS standard error s * sqrt(lam' (X'X)^-1 lam)). The simultaneous (sup-t) band over the corpora g and over
c in [c_lo, c_hi] uses the 95th percentile q of sup |t*(g, c)|; the band is a_hat + e_hat (c - c_J) +/- q se(g, c).
It carries the sampling error of level and slope jointly, and it covers every corpus path of the study over the whole
compute range. The supremum over c is exact (sup_t below: for a line, the squared t-ratio has one interior maximum,
at lam proportional to V^-1 z, the Working-Hotelling direction; otherwise the maximum is at an end of the interval).
Small-sample rule (review): with 8 or 9 budgets the wild bootstrap's q alone covers about 91 to 93 percent in Monte
Carlo, so the band uses q = max(q_boot, q_normal), q_normal the exact sup-t quantile under homoskedastic normal errors
(simulated from the exact law of (beta_hat, s)); crit_rule="boot" gives the bootstrap's q alone (sensitivity).
Options (sensitivities): HC2 studentization (as in rb2's paths.Path, on a grid), pointwise instead of simultaneous
bands, one study path at the mean of the corpus levels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import rb5common as K



def sup_t(Z, S, levels, V, x_lo, x_hi):
    """sup over the level vectors l in `levels` and x in [x_lo, x_hi] of |(l + x e)' z| / (S sqrt((l + x e)' V (l + x e)))
    for each row z of Z (draws x p) and S (draws); e is the last unit vector (the slope)."""
    Z = np.atleast_2d(Z)
    p = Z.shape[1]
    ev = np.zeros(p)
    ev[-1] = 1.0
    out = np.zeros(len(Z))
    for l in levels:
        V2 = np.array([[l @ V @ l, l @ V @ ev], [ev @ V @ l, ev @ V @ ev]])
        W = np.linalg.inv(V2)
        a, b = Z @ l, Z @ ev
        w1, w2 = W[0, 0] * a + W[0, 1] * b, W[1, 0] * a + W[1, 1] * b
        Q = a * w1 + b * w2
        with np.errstate(divide="ignore", invalid="ignore"):
            xs = w2 / w1
        inside = np.isfinite(xs) & (xs >= x_lo) & (xs <= x_hi)

        def f2(x):
            return (a + b * x) ** 2 / (V2[0, 0] + 2 * V2[0, 1] * x + V2[1, 1] * x * x)
        sup2 = np.where(inside, Q, np.maximum(f2(x_lo), f2(x_hi)))
        out = np.maximum(out, np.sqrt(np.maximum(sup2, 0.0)))
    return out / S


class StudyPath:
    kind = "band"

    def __init__(self, name, g, *, conv="total", c_J=None, B=K.B_BOOT, seed=K.SEED, c_range=None,
                 studentize="pooled", simultaneous=True, mean_level=False, level=K.LEVEL, extra_c=(),
                 crit_rule="max", n_exact=400000):
        """g: DataFrame with columns lnC, lnM and optionally 'group' (corpus)."""
        self.name, self.conv, self.studentize, self.simultaneous = name, conv, studentize, simultaneous
        self.crit_rule = crit_rule
        g = g.reset_index(drop=True).copy()
        if "group" not in g.columns:
            g["group"] = name
        self.groups = list(dict.fromkeys(g["group"]))
        self.G = len(self.groups)
        self.n = len(g)
        self.c_J = float(g["lnC"].max()) if c_J is None else float(c_J)
        Dg = np.column_stack([(g["group"] == k).astype(float).values for k in self.groups])
        self.X = np.column_stack([Dg, g["lnC"].values - self.c_J])
        self.p = self.X.shape[1]
        self.df = self.n - self.p
        y = g["lnM"].values
        self.XtXi = np.linalg.inv(self.X.T @ self.X)
        self.Rw = self.XtXi @ self.X.T
        self.beta = self.Rw @ y
        self.res = y - self.X @ self.beta
        self.s = float(np.sqrt(self.res @ self.res / self.df))
        self.h = np.einsum("ij,jk,ik->i", self.X, self.XtXi, self.X)
        self.g = g
        self.mean_level = mean_level
        # ---------------------------------------------------------------- wild bootstrap
        rng = np.random.default_rng(seed)
        u = self.res / np.sqrt(1.0 - self.h)
        V = K.WEBB[rng.integers(0, 6, (B, self.n))]
        Ystar = self.X @ self.beta + V * u
        self.bstar = (self.Rw @ Ystar.T).T                               # B x p
        self.rstar = Ystar - self.bstar @ self.X.T                        # B x n
        self.sstar = np.sqrt(np.sum(self.rstar ** 2, axis=1) / self.df)   # B
        self.B = B
        # ---------------------------------------------------------------- critical value over the band's domain
        if c_range is None:
            c_range = (self.c_J, self.c_J)
        self.c_lo = min(float(c_range[0]), self.c_J)
        self.c_hi = max(float(c_range[1]), self.c_J)
        grid = np.unique(np.concatenate([np.linspace(self.c_lo, self.c_hi, K.N_GRID), np.asarray(extra_c, float),
                                         [self.c_J]]))
        grid = grid[(grid >= self.c_lo - 1e-12) & (grid <= self.c_hi + 1e-12)]
        self.grid = grid
        Lam = self._lam_all(grid)                                         # m x p
        tabs = np.abs(self._tstar(Lam))                                   # B x m
        self.level = level
        x_lo, x_hi = self.c_lo - self.c_J, self.c_hi - self.c_J
        if studentize == "pooled":                                        # exact supremum over c
            self.q_boot = float(np.quantile(sup_t(self.bstar - self.beta, self.sstar, self.levels(), self.XtXi,
                                                  x_lo, x_hi), level))
            rng2 = np.random.default_rng(seed + 50000)
            Lc = np.linalg.cholesky(self.XtXi)
            Zn = rng2.normal(size=(n_exact, self.p)) @ Lc.T
            Sn = np.sqrt(rng2.chisquare(self.df, n_exact) / self.df)
            self.q_normal = float(np.quantile(sup_t(Zn, Sn, self.levels(), self.XtXi, x_lo, x_hi), level))
        else:                                                             # HC2: on the grid
            self.q_boot = float(np.quantile(tabs.max(axis=1), level))
            self.q_normal = np.nan
        self.q_grid = float(np.quantile(tabs.max(axis=1), level))
        self.q_sup = max(self.q_boot, self.q_normal) if (crit_rule == "max" and np.isfinite(self.q_normal)) \
            else self.q_boot
        self.q_point = np.quantile(tabs, level, axis=0)                   # pointwise (m)
        if crit_rule == "max" and studentize == "pooled":
            from scipy import stats
            self.q_point = np.maximum(self.q_point, stats.t.ppf(0.5 + level / 2, self.df))

    # ------------------------------------------------------------------ helpers
    def levels(self):
        """Level vectors l (p-dim, zero slope entry) of the band's paths: one per corpus, or their mean."""
        if self.mean_level:
            l = np.zeros(self.p)
            l[: self.G] = 1.0 / self.G
            return [l]
        out = []
        for gi in range(self.G):
            l = np.zeros(self.p)
            l[gi] = 1.0
            out.append(l)
        return out

    def _lam(self, gi, c):
        c = np.atleast_1d(np.asarray(c, float))
        L = np.zeros((len(c), self.p))
        if self.mean_level:
            L[:, : self.G] = 1.0 / self.G
        else:
            L[:, gi] = 1.0
        L[:, -1] = c - self.c_J
        return L

    def _lam_all(self, c):
        if self.mean_level:
            return self._lam(0, c)
        return np.vstack([self._lam(gi, c) for gi in range(self.G)])

    def _se(self, Lam, res=None, s=None):
        """Standard error of Lam beta: pooled (classical) or HC2."""
        if self.studentize == "pooled":
            nrm = np.sqrt(np.einsum("mp,pq,mq->m", Lam, self.XtXi, Lam))
            if s is None:
                return self.s * nrm
            return np.outer(s, nrm)
        cw = Lam @ self.Rw                                                # m x n
        r = self.res if res is None else res
        r2 = (r / np.sqrt(1 - self.h)) ** 2
        return np.sqrt(np.einsum("mn,...n->...m", cw ** 2, r2))

    def _tstar(self, Lam):
        num = (self.bstar - self.beta) @ Lam.T                            # B x m
        if self.studentize == "pooled":
            return num / self._se(Lam, s=self.sstar)
        return num / self._se(Lam, res=self.rstar)

    # ------------------------------------------------------------------ public
    def point(self, c, gi=0):
        return self._lam(gi, c) @ self.beta

    def crit(self, c):
        """Critical value at c: sup-t (simultaneous) or pointwise (interpolated on the grid)."""
        c = np.atleast_1d(np.asarray(c, float))
        if self.simultaneous:
            return np.full(len(c), self.q_sup)
        qp = self.q_point.reshape(-1, len(self.grid)).max(axis=0) if not self.mean_level else self.q_point
        return np.interp(c, self.grid, qp)

    def band_groups(self, c):
        """(lo, hi) arrays of shape (G, len(c)) (one row if mean_level)."""
        c = np.atleast_1d(np.asarray(c, float))
        if (c < self.c_lo - 1e-9).any() or (c > self.c_hi + 1e-9).any():
            raise ValueError(f"{self.name}: c outside the band's domain [{self.c_lo:.3f}, {self.c_hi:.3f}]")
        rows = range(1) if self.mean_level else range(self.G)
        lo, hi = [], []
        q = self.crit(c)
        for gi in rows:
            L = self._lam(gi, c)
            f = L @ self.beta
            se = self._se(L)
            lo.append(f - q * se)
            hi.append(f + q * se)
        return np.array(lo), np.array(hi)

    def band(self, c):
        lo, hi = self.band_groups(c)
        return lo.min(axis=0), hi.max(axis=0)

    def anchor_interval(self, level=K.LEVEL):
        """Pointwise bootstrap-t interval of each corpus level at c_J (the anchor), symmetric |t|."""
        out = []
        rows = range(1) if self.mean_level else range(self.G)
        for gi in rows:
            L = self._lam(gi, self.c_J)
            q = self._qpoint(L, level)
            f = float((L @ self.beta)[0])
            se = float(self._se(L)[0])
            out.append(dict(group=self.groups[gi] if not self.mean_level else "mean of corpora", lnMs=f,
                            lnMs_lo=f - q * se, lnMs_hi=f + q * se, q=q, se=se))
        return pd.DataFrame(out)

    def _qpoint(self, L, level):
        """Pointwise bootstrap-t quantile of |t| for one linear functional, floored at t_df under crit_rule='max'."""
        q = float(np.quantile(np.abs(self._tstar(L))[:, 0], level))
        if self.crit_rule == "max" and self.studentize == "pooled":
            from scipy import stats
            q = max(q, float(stats.t.ppf(0.5 + level / 2, self.df)))
        return q

    def slope_interval(self, level=K.LEVEL):
        L = np.zeros((1, self.p))
        L[0, -1] = 1.0
        q = self._qpoint(L, level)
        se = float(self._se(L)[0])
        return float(self.beta[-1] - q * se), float(self.beta[-1] + q * se)

    def summary(self):
        a = self.anchor_interval()
        eL, eU = self.slope_interval()
        rows = []
        for _, r in a.iterrows():
            rows.append(dict(path=self.name, corpus=r["group"], conv=self.conv, n_budgets=self.n, n_groups=self.G,
                             df_resid=self.df, C_J=float(np.exp(self.c_J)), Mstar_J=float(np.exp(r["lnMs"])),
                             Mstar_J_lo=float(np.exp(r["lnMs_lo"])), Mstar_J_hi=float(np.exp(r["lnMs_hi"])),
                             e=float(self.beta[-1]), e_lo=eL, e_hi=eU, resid_sd_pooled=self.s,
                             q_sup=self.q_sup, q_boot=self.q_boot, q_normal=self.q_normal, q_grid=self.q_grid,
                             crit_rule=self.crit_rule, q_anchor_pointwise=r["q"], B=self.B, studentize=self.studentize,
                             simultaneous=self.simultaneous, C_band_lo=float(np.exp(self.c_lo)),
                             C_band_hi=float(np.exp(self.c_hi))))
        return pd.DataFrame(rows)


class ShiftedPath:
    """A band whose level is uncertain by a known interval [dlo, dhi] in ln M* (convention conversion bound)."""
    kind = "band"

    def __init__(self, base, dlo, dhi, name=None, conv="total"):
        self.base, self.dlo, self.dhi = base, float(dlo), float(dhi)
        self.name = name or base.name + " (shifted)"
        self.conv = conv
        self.c_lo, self.c_hi = base.c_lo, base.c_hi

    def band(self, c):
        lo, hi = self.base.band(c)
        return lo + self.dlo, hi + self.dhi


class PointPath:
    """A published law's own path, without sampling error (DeepSeek's law in its own convention)."""
    kind = "point"

    def __init__(self, name, alpha, beta, lnG, conv="ds"):
        self.name, self.alpha, self.beta, self.lnG, self.conv = name, alpha, beta, lnG, conv
        self.e = 1 - 2 * beta / (alpha + beta)

    def band(self, c):
        c = np.atleast_1d(np.asarray(c, float))
        a = self.beta / (self.alpha + self.beta)
        x = -2 * self.lnG + (1 - 2 * a) * (c - np.log(6.0))
        return x, x


class AnchorPath:
    """PI-1 style: an anchor interval [lo, hi] for ln M* at c0, extrapolated with a path-slope range [eL, eU]
    (Proposition A6(ii)): upper end hi + eU (c - c0) above c0 and hi + eL (c - c0) below it; lower end symmetric."""
    kind = "anchor"

    def __init__(self, name, c0, lo, hi, eL, eU, conv="total"):
        self.name, self.c0, self.lo, self.hi, self.eL, self.eU, self.conv = name, float(c0), float(lo), float(hi), \
            float(eL), float(eU), conv

    def band(self, c):
        c = np.atleast_1d(np.asarray(c, float))
        x = c - self.c0
        hi = self.hi + np.where(x > 0, self.eU * x, self.eL * x)
        lo = self.lo + np.where(x > 0, self.eL * x, self.eU * x)
        return lo, hi
