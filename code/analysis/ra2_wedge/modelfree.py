"""modelfree.py -- model-free sigma* and the Approach-2 path from IsoFLOP profiles (R1 comment 7).

At a compute-optimal point the on-path elasticity of substitution satisfies (R1 c7; ra5/ra1 formalize it)
    1/sigma* - 1 = L_nn / (2 |dL*/dc|),
L_nn = curvature of the IsoFLOP profile in ln N at its minimum, dL*/dc = slope of the loss-compute frontier in
ln C. Both are Approach-1/2 objects; the ratio is invariant to monotone transformations of output (no E, no kappa,
no functional form). Within the Chinchilla/kappa family it equals S/2 = (alpha+beta)/2 (inner exponents).

Estimator (per design): for each budget k with >= 4 points and an interior minimum, OLS of L on (x, x^2), x = ln N
(centred) -> argmin x*_k, minimum L*_k, curvature L_nn,k = 2 c2_k (quadratic: Meta's own per-budget rule;
cubic: robustness, curvature evaluated at the argmin). Frontier: quadratic polynomial of L*_k in c_k = ln C_k,
derivative at each c_k. Pooled S/2 = sum_k L_nn,k / sum_k 2|dL*/dc|_k (ratio of sums; per-budget ratios reported).
Path: OLS of ln N*_k on ln(C_k/6) -> (a, ln G) and ln M*(C) = -2 ln G + (1 - 2a) ln(C/6).
Inference: design-conditional wild bootstrap (Rademacher weights on the HC1-rescaled residuals of the per-budget fits,
i.e. the loss values are redrawn at fixed (N, C)); B = 999; the whole chain (argmins, curvature, frontier, path) is
recomputed in every draw, so (S, a, ln G) are drawn jointly.
Data: open-athena IsoFLOP compilation (data/raw/isoflop_experiments): Llama 3 (digitized, 10 budgets 6e18-1e22),
Marin 2026-03 Comma/DCLM/Nemotron (7-8 budgets, 1.8e18-3e20), Chinchilla (Epoch digitization, 9 budgets, the 123
IsoFLOP-profile runs; cross-check only).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ra2common import RAW, SEED

DESIGNS = {
    "meta": "llama_3",
    "marin_comma": "marin_202603__comma__llama_2",
    "marin_dclm": "marin_202603__dclm__llama_2",
    "marin_nemotron": "marin_202603__nemotron__llama_2",
    "chin_iso": "epochai_chinchilla__massivetext__chinchilla",
}


def load(experiment):
    df = pd.read_csv(os.path.join(RAW, "isoflop_experiments", "isoflop_experiments.csv"))
    df = df[df["experiment"] == experiment].rename(columns={"params": "N", "tokens": "D", "loss": "L"})
    return df[["budget", "N", "D", "L"]].sort_values(["budget", "N"]).reset_index(drop=True)


def _polyfit(x, y, deg):
    X = np.vander(x, deg + 1, increasing=True)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b, X


def per_budget(df, Lcol="L", deg=2):
    """Per-budget argmin, minimum and curvature. Returns DataFrame (one row per usable budget)."""
    rows = []
    for C, g in df.groupby("budget"):
        if len(g) < max(4, deg + 2):
            continue
        x = np.log(g["N"].values)
        xm = x.mean()
        xc = x - xm
        b, _ = _polyfit(xc, g[Lcol].values, deg)
        if deg == 2:
            if b[2] <= 0:
                continue
            xs = -b[1] / (2 * b[2])
            curv = 2 * b[2]
        else:   # cubic: stationary point with positive curvature
            c1, c2, c3 = b[1], b[2], b[3]
            disc = (2 * c2) ** 2 - 4 * 3 * c3 * c1
            if disc < 0:
                continue
            roots = [(-2 * c2 + sgn * np.sqrt(disc)) / (6 * c3) for sgn in (1, -1)] if abs(c3) > 1e-12 else [-c1 / (2 * c2)]
            roots = [r for r in roots if 2 * c2 + 6 * c3 * r > 0]
            if not roots:
                continue
            xs = roots[0]
            curv = 2 * c2 + 6 * c3 * xs
        if not (xc.min() < xs < xc.max()):
            continue   # boundary minimum: argmin not identified on this grid
        Ls = np.polyval(b[::-1], xs)
        rows.append(dict(budget=C, lnC=np.log(C), lnNstar=xs + xm, Lstar=Ls, Lnn=curv, n=len(g)))
    return pd.DataFrame(rows)


def summarize(pb):
    """Frontier slope, pooled S/2, per-budget ratios and the A2 path from per-budget objects."""
    c = pb["lnC"].values
    fb = np.polyfit(c, pb["Lstar"].values, 2)
    slope = np.polyval(np.polyder(fb), c)             # dL*/dc (negative)
    ratio_k = pb["Lnn"].values / (2 * np.abs(slope))
    S2 = pb["Lnn"].sum() / (2 * np.abs(slope).sum())
    a, lnG0 = np.polyfit(np.log(pb["budget"].values / 6.0), pb["lnNstar"].values, 1)
    return dict(S2=S2, S2_median=float(np.median(ratio_k)), a=a, lnG=lnG0, ratio_k=ratio_k, slope_k=slope,
                n_budgets=len(pb), Cmax=float(pb["budget"].max()))


def fit_design(df, deg=2):
    pb = per_budget(df, deg=deg)
    return pb, summarize(pb)


def wild_boot(df, B=999, seed=SEED, deg=2):
    """Design-conditional wild bootstrap of the whole chain. Returns array (B, 4): S/2, a, lnG, S/2 (median rule)."""
    rng = np.random.default_rng(seed)
    # fitted values and HC1-rescaled residuals of the per-budget polynomial fits (all budgets with >= deg+2 points)
    fit = np.full(len(df), np.nan)
    res = np.full(len(df), np.nan)
    for C, g in df.groupby("budget"):
        if len(g) < deg + 2:
            continue
        x = np.log(g["N"].values)
        b, X = _polyfit(x - x.mean(), g["L"].values, deg)
        f = X @ b
        e = g["L"].values - f
        fit[g.index] = f
        res[g.index] = e * np.sqrt(len(g) / (len(g) - deg - 1))
    ok = np.isfinite(fit)
    out = []
    for _ in range(B):
        v = rng.choice([-1.0, 1.0], size=len(df))
        dd = df.copy()
        dd.loc[ok, "L"] = fit[ok] + res[ok] * v[ok]
        try:
            pb = per_budget(dd, deg=deg)
            if len(pb) < 4:
                raise ValueError
            s = summarize(pb)
            out.append([s["S2"], s["a"], s["lnG"], s["S2_median"]])
        except Exception:
            out.append([np.nan] * 4)
    return np.array(out)


def resid_sd(df, deg=2):
    res, k = [], 0
    for C, g in df.groupby("budget"):
        if len(g) < deg + 2:
            continue
        x = np.log(g["N"].values)
        b = np.polyfit(x - x.mean(), g["L"].values, deg)
        res += list(g["L"].values - np.polyval(b, x - x.mean()))
        k += deg + 1
    return float(np.sqrt(np.sum(np.square(res)) / (len(res) - k)))


def mc_bias_factor(df, s, R=300, seed=SEED, deg=2):
    """Parametric-bootstrap bias check on the design's own grid: simulate a Chinchilla-form technology with the
    estimated (S/2, a, ln G), E and scale matched to the observed per-budget minima, Gaussian noise with the design's
    residual s.d.; run the estimator; factor = true S/2 / mean estimate. Captures finite-grid parabola bias
    (Czech et al.) and the Jensen bias of the curvature/slope ratio under noise."""
    rng = np.random.default_rng(seed)
    S = 2 * s["S2"]
    al, be = S * (1 - s["a"]), S * s["a"]
    N, Cb = df["N"].values, df["budget"].values
    D = Cb / (6 * N)
    lnG = s["lnG"]
    # A/B from G: G^S = alpha A/(beta B)  -> take B = 1
    Bc = 1.0
    Ac = be * np.exp(S * lnG) / al
    R0 = Ac * N ** -al + Bc * D ** -be
    # match level: L = E + k R0 with E = 0.75 min L, k by least squares on the observed losses
    E = 0.75 * df["L"].min()
    k = np.sum((df["L"].values - E) * R0) / np.sum(R0 ** 2)
    L0 = E + k * R0
    sd = resid_sd(df, deg)
    est = []
    for _ in range(R):
        dd = df.copy()
        dd["L"] = L0 + sd * rng.standard_normal(len(dd))
        try:
            pb, ss = fit_design(dd, deg)
            est.append(ss["S2"])
        except Exception:
            pass
    est = np.array(est)
    return dict(true=s["S2"], mc_mean=float(est.mean()), mc_median=float(np.median(est)), mc_sd=float(est.std()),
                factor=float(s["S2"] / est.mean()), R=len(est), resid_sd=sd)


def run_all(B=999):
    """Point estimates and bootstrap draws for every design. Returns (table DataFrame, per-budget DataFrame, draws).
    draws[key]: (B, 4) raw wild-bootstrap draws (S/2, a, lnG, S/2 median rule); rows carry the raw and the
    bias-corrected S/2 (= raw x MC factor)."""
    rows, pbs, draws = [], [], {}
    for key, exp in DESIGNS.items():
        df = load(exp)
        for deg in (2, 3):
            pb, s = fit_design(df, deg)
            bc = dict(factor=np.nan, mc_mean=np.nan, resid_sd=np.nan, R=0)
            if deg == 2:
                dr = wild_boot(df, B=B, deg=2, seed=SEED + sum(map(ord, key)))
                draws[key] = dr
                okd = np.isfinite(dr).all(1)
                lo, hi = np.percentile(dr[okd, 0], [2.5, 97.5])
                pbs.append(pb.assign(design=key, ratio=s["ratio_k"], slope=s["slope_k"]))
                bc = mc_bias_factor(df, s, seed=SEED + 7 + sum(map(ord, key)))
            else:
                lo = hi = np.nan
            S2 = s["S2"]
            rows.append(dict(design=key, experiment=exp, deg=deg, n_points=len(df), n_budgets=s["n_budgets"],
                             Cmax=s["Cmax"], S2=S2, S2_lo=lo, S2_hi=hi, S2_median_rule=s["S2_median"],
                             sigma_star=1 / (1 + S2), sigma_lo=1 / (1 + hi) if np.isfinite(hi) else np.nan,
                             sigma_hi=1 / (1 + lo) if np.isfinite(lo) else np.nan, a=s["a"], lnG=s["lnG"],
                             Mstar_Cmax=float(np.exp(-2 * s["lnG"] + (1 - 2 * s["a"]) * np.log(s["Cmax"] / 6))),
                             mc_factor=bc["factor"], mc_mean=bc["mc_mean"], resid_sd=bc["resid_sd"], mc_R=bc["R"],
                             S2_bc=S2 * bc["factor"] if deg == 2 else np.nan,
                             sigma_star_bc=1 / (1 + S2 * bc["factor"]) if deg == 2 else np.nan,
                             B=B if deg == 2 else 0, n_draws_ok=int(np.isfinite(draws[key]).all(1).sum()) if deg == 2 else 0))
    return pd.DataFrame(rows), pd.concat(pbs, ignore_index=True), draws
