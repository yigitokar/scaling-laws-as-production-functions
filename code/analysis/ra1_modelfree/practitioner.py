"""practitioner.py -- training-compute overhead of training off the compute-optimal ratio (Task 6; R2 Major 8).

For a model trained at M = k M*(C) with the same loss as the compute-optimal model, the Farrell ratio of training
compute to the minimum is (paper Lemma 3 / Corollary A3)
    C / C_min = ((alpha + beta w)/(alpha + beta))^(1/gamma) w^(-1/alpha),  ln w = (1/sigma* - 1) ln k,
with gamma = alpha beta/(alpha + beta), sigma* = 2/(2 + alpha + beta). In the homothetic case alpha = beta = rho,
rho = 1/sigma* - 1 (so alpha + beta = 2(1/sigma* - 1)) and C/C_min = cosh(rho ln k / 2)^(2/rho).
The non-homothetic variants keep sigma* and set alpha/beta to the Chinchilla (Besiroglu et al.) or Hoffmann et al.
ratio; sigma* alone pins C/C_min only when alpha = beta.
Every number is checked against a brute-force computation on the technology (minimize compute at the target loss).
de Vries (2023; `devries2023go`): with E = 1.69, A = 406.4, B = 410.7, alpha = 0.32, beta = 0.28 (the post's values)
a model at 30% of the compute-optimal size needs about 100% more compute; we reproduce that point and map it to k.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import ra1_common as rc

K_GRID = (2, 10, 100, 1000)
SIG_GRID = (0.6, 0.7, 0.74, 0.8)


def overhead(alpha, beta, k):
    sig = 2 / (2 + alpha + beta)
    w = np.exp((1 / sig - 1) * np.log(k))
    g = alpha * beta / (alpha + beta)
    return ((alpha + beta * w) / (alpha + beta)) ** (1 / g) * w ** (-1 / alpha)


def overhead_homothetic(sig, k):
    rho = 1 / sig - 1
    return np.cosh(rho * np.log(k) / 2) ** (2 / rho)


def brute(m: rc.sl.Chinchilla, k, C=1e21):
    """Numerical check: L0 = loss of the compute-optimal model at C (so C = C_min(L0)); find the model on the
    isoquant L = L0 whose ratio M = D/N equals k times the compute-optimal ratio at ITS OWN compute (the definition
    in the paper's Lemma 3; identical to 'k times M*(C_min)' when alpha = beta); return its compute over C."""
    N0 = m.N_opt(C)
    L0 = m.L_opt(C)

    def D_on_isoquant(lnN):
        v = L0 - m.E - m.A * np.exp(-m.alpha * lnN)
        return (v / m.B) ** (-1 / m.beta)

    def g(lnN):
        N = np.exp(lnN)
        D = D_on_isoquant(lnN)
        Cn = 6 * N * D
        return np.log(D / N) - np.log(k * m.D_opt(Cn) / m.N_opt(Cn))

    lo = np.log(N0) - 1e-9
    # the smallest N with a finite D on the isoquant
    lnN_min = -np.log(((L0 - m.E) / m.A)) / m.alpha + 1e-9
    lnN = brentq(g, lnN_min + 1e-6, lo)
    N1 = np.exp(lnN)
    return 6 * N1 * D_on_isoquant(lnN) / C


def run(log=rc.log):
    rows = []
    splits = {"homothetic (alpha = beta)": 1.0,
              "alpha/beta of Besiroglu et al. (0.951)": rc.sl.BESIROGLU.alpha / rc.sl.BESIROGLU.beta,
              "alpha/beta of Hoffmann et al. (1.191)": rc.sl.HOFFMANN.alpha / rc.sl.HOFFMANN.beta}
    for lab, ratio in splits.items():
        for sg in SIG_GRID:
            S = 2 / sg - 2
            beta = S / (1 + ratio)
            alpha = S - beta
            m = rc.sl.Chinchilla(E=1.7, A=400.0, B=400.0 * 5, alpha=alpha, beta=beta)
            for k in K_GRID:
                cf = overhead(alpha, beta, k)
                ch = overhead_homothetic(sg, k) if ratio == 1.0 else np.nan
                bf = brute(m, k)
                rows.append(dict(split=lab, sigma_star=sg, alpha=alpha, beta=beta, k=k, w=np.exp((1 / sg - 1) * np.log(k)),
                                 C_over_Cmin=cf, cosh_formula=ch, brute_force=bf, overhead_pct=100 * (cf - 1),
                                 expenditure_share_inference=(np.exp((1 / sg - 1) * np.log(k)) - 1) / np.exp((1 / sg - 1) * np.log(k))))
                assert abs(cf / bf - 1) < 1e-6, (lab, sg, k, cf, bf)
                if ratio == 1.0:
                    assert abs(ch / cf - 1) < 1e-9
    tab = pd.DataFrame(rows)
    # de Vries (2023) check: his parameters; model at 30% of the optimal size
    dv = rc.sl.Chinchilla(E=1.69, A=406.4, B=410.7, alpha=0.32, beta=0.28)
    C = 1e21
    N0, D0 = dv.N_opt(C), dv.D_opt(C)
    kN = 0.30
    ratio_uv = (dv.A * N0 ** -dv.alpha) / (dv.B * D0 ** -dv.beta)
    kD = (1 - (kN ** -dv.alpha - 1) * ratio_uv) ** (-1 / dv.beta)      # de Vries's formula
    k_equiv = kD / kN
    C1 = 6 * kN * N0 * kD * D0
    k_own = k_equiv * (D0 / N0) / (dv.D_opt(C1) / dv.N_opt(C1))   # relative to M* at the model's own compute
    dvrow = dict(source="de Vries (2023) parameters (E 1.69, A 406.4, B 410.7, alpha 0.32, beta 0.28)",
                 model_size_fraction=kN, data_multiplier=kD, overhead_devries=kN * kD - 1, k_equivalent=k_equiv,
                 k_own_compute=k_own, overhead_our_formula=overhead(dv.alpha, dv.beta, k_own) - 1,
                 overhead_homothetic_same_sigma=overhead_homothetic(dv.sigma_star, k_equiv) - 1,
                 sigma_star=dv.sigma_star, ratio_uv_at_optimum=ratio_uv, beta_over_alpha=dv.beta / dv.alpha)
    # full de Vries curve mapped to k (for the figure)
    curve = []
    for kN_ in np.linspace(0.2, 1.0, 81):
        base = 1 - (kN_ ** -dv.alpha - 1) * ratio_uv
        if base <= 0:
            continue
        kD_ = base ** (-1 / dv.beta)
        curve.append(dict(kN=kN_, kD=kD_, k=kD_ / kN_, overhead=kN_ * kD_ - 1,
                          overhead_formula=overhead(dv.alpha, dv.beta, kD_ / kN_) - 1))
    log(f"  practitioner table: {len(tab)} rows; de Vries check: 30% size -> {100 * dvrow['overhead_devries']:.0f}% overhead "
        f"(k = {k_equiv:.1f}; our formula {100 * dvrow['overhead_our_formula']:.0f}%)")
    return tab, pd.DataFrame([dvrow]), pd.DataFrame(curve)
