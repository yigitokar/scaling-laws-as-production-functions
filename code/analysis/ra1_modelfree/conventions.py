"""conventions.py -- parameter-count/FLOP-accounting conventions for the model-free sigma* (R2 Major 9(b)).

The model-free identity holds along an exact isocost. On Marin's ladders the nominal budget C_b (3 x forward FLOPs)
differs from 6 N_config D by -7% to +35%, strongly correlated with N (module m1). In the FLOP-implied convention
N_eff = C_b/(6D) every profile is an exact isocost; in the configuration convention it is not. If
x_eff = x + ln phi(N) with elasticity eta = d ln phi / d ln N, the profile curvature at the minimum scales by
1/(1 + eta)^2, so S_eff = S/(1 + eta)^2. This module (i) estimates eta on Marin (within-budget regression of
ln(C_b/(6 N_cfg D)) on ln N_cfg), (ii) re-estimates model-free sigma* on Marin in the config convention, and
(iii) reports the implied sensitivity of sigma* to eta for the Chinchilla extraction, where the FLOP count behind
the digitized budgets is not observed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import ra1_common as rc


def marin_eta():
    D = rc.isoflop_designs()
    rows = []
    for name in ("Marin, Comma", "Marin, DCLM", "Marin, Nemotron-CC"):
        df = D[name]
        y = np.log(df["C_b"].values / (6 * df["N_cfg"].values * df["D"].values))
        x = np.log(df["N_cfg"].values)
        # within-budget slope
        X = np.column_stack([pd.get_dummies(df["b"]).values.astype(float), x])
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        rows.append(dict(design=name, eta_within_budget=float(coef[-1]), ratio_min=float(np.exp(y.min())),
                         ratio_max=float(np.exp(y.max()))))
    return pd.DataFrame(rows)


def config_designs():
    """Marin designs with x = ln N_config (profiles at the nominal budget are then not exact 6ND isocosts)."""
    D = rc.isoflop_designs()
    out = {}
    for name in ("Marin, Comma", "Marin, DCLM", "Marin, Nemotron-CC"):
        df = D[name].copy()
        df["x"] = np.log(df["N_cfg"].values)
        out[f"{name} [config N]"] = df.sort_values(["b", "x"]).reset_index(drop=True)
    return out


def sensitivity_to_eta(sigma, etas=(-0.15, -0.10, -0.05, 0.05)):
    S = 2 / sigma - 2
    return {eta: 2 / (2 + S / (1 + eta) ** 2) for eta in etas}
