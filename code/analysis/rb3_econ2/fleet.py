"""fleet.py -- the fleet comparison with the service-life assumption stated and varied (R3 N6(3), minor 19).

A planned lifetime multiple m = w - 1 (value of compactness per unit of the model's own training cost) is converted into
a fleet share of AI compute (ra3_econ share.flow_adjustment, restated here):
    s_fleet = p m phi / (p m phi + rho),   phi = (1 - exp(-g L)) / (g L),
where L is the service life over which each model's planned serving is spread evenly, g the growth rate of training
compute (vintages served today are smaller than today's training runs), rho the R&D multiple of final training runs
(Denain and Wu 2026: final runs are 9.6-22.6 percent of R&D compute, rho = 4.4-10.4; rho = 1 ignores R&D) and p the
price of an inference FLOP relative to a training FLOP. m is the 2025 compute-weighted aggregate of the clean sample
(decision units, primary; models as a check), from trend.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import rb3common as RC


def run(TR, g_hat):
    import trend as TRm
    B = TR["B"].set_index("year")
    u25 = TR["u"][(TR["u"].year == 2025) & (TR["u"].dev != "Alibaba")]
    m_noali = 1.0 / (1.0 - TRm.s_agg(u25.w_ref, u25.Cmp)) - 1.0
    ms = [("2025, decision units", float(B.loc[2025, "m_agg_units"])),
          ("2025, models", float(B.loc[2025, "m_agg_models"])),
          ("2025, decision units excl. Alibaba", float(m_noali)),     # review sensitivity (Qwen3 carries the 2025 compute)
          ("2024, decision units", float(B.loc[2024, "m_agg_units"]))]
    rows = []
    for mlab, m in ms:
        for gname, g in (("5.06x", g_hat), ("4.2x", RC.G_EPOCH)):
            lg = np.log(g)
            for L in RC.LIFE_YEARS:
                phi = (1 - np.exp(-lg * L)) / (lg * L)
                for rho in RC.RHO:
                    for p in (1.0, 2.0):
                        rows.append(dict(m_label=mlab, m=m, s_planned=m / (1 + m), g_label=gname, g=g, L=L, phi=phi,
                                         rho=rho, p=p, s_fleet=p * m * phi / (p * m * phi + rho)))
    return pd.DataFrame(rows)
