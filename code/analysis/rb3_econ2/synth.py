"""synth.py -- the shadow value of a unique token beside two observable prices (R3 minor 18; optional).

(1) The compute cost of generating a synthetic token with a generator of N_g active parameters: 2 N_g FLOPs of
    decoding at p times the price of a training FLOP, i.e. p N_g / (3 N) in units of the trainee's processing cost per
    token (6 N). Self-generation (N_g = N) at p = 1-2 costs 0.33-0.67 units.
(2) Posted API prices for output tokens (DeepSeek, retrieved 2026-09-24: $0.60-$1.20 per million for V4.1-Flash and
    $1.98-$3.96 for V4-Pro, off-peak/peak; data/raw/rb3_econ2/deepseek_pricing_20260924.html).
The shadow value is the private value of one more unique token to one compute-optimal run at the reference technology
(kappa free, sigma* = 0.70; data decay only, R*_D = 15.4: ra3's wall), at 1e26 FLOP (ra3's $3/M reference point) and at
the fitted frontier compute of September 2026, priced at ra3's median amortized cost per FLOP of 2024-2025 frontier runs
(cloud prices about 3.7 times higher). A synthetic token is not a perfect substitute for a fresh unique token, so the
break-even scarcities are lower bounds: generation starts to pay only at or above them. The API price is one provider's
posted price on one date (a low-end price for a large model), not a market-wide cost of synthetic data.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import rb3common as RC

import wall as WL  # noqa: E402  (ra3_econ, read-only)


def usd_per_flop():
    f = os.path.join(RC.ROOT, "output", "tables", "ra3_econ_usd_per_flop.csv")
    x = pd.read_csv(f)
    return x.groupby("basis")["usd_per_flop"].median().to_dict()


def run(C_now):
    _, kq = WL.load_wall_techs()
    prices = usd_per_flop()
    rows = []
    for C in (1e26, C_now):
        N = kq.N_opt(C)
        for r in (1.1, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0):
            pt = WL.wall_point(kq, C, r, "D15", extras=False)
            rows.append(dict(C=C, r=r, N_star=N, N_c=pt["N_U"], shadow_rel=pt["shadow_rel"], shadow_flop=pt["shadow_flop"],
                             usd_per_Mtok_amortized=pt["shadow_flop"] * prices["amortized"] * 1e6,
                             usd_per_Mtok_cloud=pt["shadow_flop"] * prices["cloud"] * 1e6))
    S = pd.DataFrame(rows)
    # break-even scarcity: shadow value = price of a synthetic token
    be = []
    for C in (1e26, C_now):
        g = S[S.C == C].sort_values("r")
        f_rel = lambda r: np.interp(np.log(r), np.log(g.r), np.log(g.shadow_rel))  # noqa: E731
        f_usd = lambda r: np.interp(np.log(r), np.log(g.r), np.log(g.usd_per_Mtok_amortized))  # noqa: E731
        for lab, target, kind in [("self-generation, p = 1", 1 / 3, "rel"), ("self-generation, p = 2", 2 / 3, "rel"),
                                  ("generator 1/10 of N, p = 2", 2 / 30, "rel"),
                                  ("API, V4.1-Flash off-peak ($0.60/M)", RC.API_OUTPUT_USD_PER_MTOK["flash_offpeak"], "usd"),
                                  ("API, V4-Pro peak ($3.96/M)", RC.API_OUTPUT_USD_PER_MTOK["pro_peak"], "usd")]:
            fn = f_rel if kind == "rel" else f_usd
            try:
                rb = float(np.exp(brentq(lambda lr: fn(np.exp(lr)) - np.log(target), np.log(1.1), np.log(16.0))))
            except ValueError:
                rb = np.nan
            be.append(dict(C=C, price=lab, target=target, kind=kind, breakeven_r=rb))
    return dict(S=S, BE=pd.DataFrame(be), prices=prices)
