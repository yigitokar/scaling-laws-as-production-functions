"""pidset.py -- the identified set for sigma*(C) beyond the designs (T1.7; R1 minor 4, R2 NM3(c)).

rb1's construction (rb1_sigmaC/pid.py): beyond the largest bracketed budgets, if sigma* does not rise with compute and
falls no faster than the drift measured within the designs,
    sigma_lin(C) <= sigma*(C) <= upper,   sigma_lin(C) = sigma_top + b_CL3 (log10 C - c_anchor),
with b_CL3 = -0.0576 per decade (Chinchilla + Llama 3, design fixed effects) and sigma_top = 0.594 (the top-budget
mean). Two upper ends: sigma_top if the top-budget decline is real, and 0.70, the level where the designs overlap, if
one is agnostic about the decline (the drift-agnostic set of rb1's memo). Chinchilla's largest bracketed budget is
3e21, so the set is shown from 1e21 with a mark at 3e21.

Where the top-budget mean sits (R1 minor 4). The five budgets span 6e20-3e21. Under a linear drift an inverse-variance
mean estimates sigma* at the inverse-variance-weighted mean of log10 C, so that is the anchor at which the lower bound
is coherent ('precision-weighted centre'). Also shown: the compute-weighted centre (weights proportional to C, as the
referee's wording reads literally), the unweighted centre of log10 C, and rb1's 1e21. A sampling-error version uses
the joint-bootstrap 95 percent lower limits of sigma_top and of the drift (jointboot.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import sigcommon as cm

CS = [3e21, 1e22, 1e23, 1e24, 1e25]


def run(lin, tops, study_upto):
    """lin: jointboot.linear_stats table; tops: its member dict; study_upto: study.upto_3e20 table (T1.4)."""
    L = lin.set_index("statistic")
    slope_row = L.loc[[s for s in L.index if s.startswith("Chinchilla + Llama 3 drift: design FE slope, inverse")][0]]
    top_row = L.loc[[s for s in L.index if s.startswith("Top-budget mean: Chinchilla + Llama 3, budgets >= 6e20 (primary)")][0]]
    b = float(slope_row.estimate)
    s_top = float(top_row.estimate)
    mem, est, se, a = tops["Chinchilla + Llama 3, budgets >= 6e20 (primary)"]
    lc = np.log10([C for _, C in mem])
    Cv = np.array([C for _, C in mem])
    centres = {"1e21 (rb1)": 21.0,
               "precision-weighted centre of the five budgets": float(np.sum(a * lc)),
               "compute-weighted centre (weights proportional to C)": float(np.sum(Cv * lc) / np.sum(Cv)),
               "unweighted centre of log10 C": float(np.mean(lc))}
    up = study_upto[study_upto.variant.str.startswith("Study-level mean, budgets <= 3e20 FLOP (primary")].iloc[0]
    # sampling-error versions: lower 95 percent limits (joint bootstrap; slope s.e. scaled for the budgets' excess
    # dispersion, the larger of the two)
    se_b = float(max(slope_row.se_boot_joint * np.sqrt(max(slope_row.resid_var_scale, 1.0)), slope_row.se_model_scaled))
    b_lo = b - 1.96 * se_b
    s_lo = float(top_row.estimate - 1.96 * top_row.se_boot_joint)
    rows = []
    for lab, c0 in centres.items():
        for C in CS:
            x = np.log10(C)
            lo = float(np.clip(s_top + b * max(x - c0, 0.0), 0.0, 1.0))
            rows.append(dict(anchor=lab, anchor_log10C=c0, anchor_C=10 ** c0, C=C, log10C=x, sigma_lower=lo,
                             upper_decline_real=s_top, upper_drift_agnostic=0.70, upper_study_le3e20=float(up.mean_re),
                             k_lower=1 / lo - 1 if lo > 0 else np.inf, k_upper_decline=1 / s_top - 1, k_upper_agnostic=1 / 0.70 - 1,
                             sigma_top=s_top, drift=b))
    # sampling-error version at rb1's anchor and at the precision-weighted centre
    for lab in ("1e21 (rb1)", "precision-weighted centre of the five budgets"):
        c0 = centres[lab]
        for C in CS:
            x = np.log10(C)
            rows.append(dict(anchor=f"{lab}; sigma_top and drift at their lower 95 percent limits", anchor_log10C=c0,
                             anchor_C=10 ** c0, C=C, log10C=x, sigma_lower=float(np.clip(s_lo + b_lo * max(x - c0, 0.0), 0, 1)),
                             upper_decline_real=float(top_row.estimate + 1.96 * top_row.se_boot_joint),
                             upper_drift_agnostic=0.70, upper_study_le3e20=float(up.hi_hksj), sigma_top=s_lo, drift=b_lo))
    T = pd.DataFrame(rows)
    info = dict(sigma_top=s_top, drift=b, se_drift_used=se_b, drift_lo=b_lo, sigma_top_lo=s_lo,
                study_le3e20=float(up.mean_re), study_le3e20_lo=float(up.lo_hksj), study_le3e20_hi=float(up.hi_hksj),
                **{f"centre|{k}": v for k, v in centres.items()})
    # check against rb1's published set (anchor 1e21)
    pi = pd.read_csv(cm.RB1_PI)
    chk = []
    for C in (1e22, 1e23, 1e24, 1e25):
        mine = T[(T.anchor == "1e21 (rb1)") & np.isclose(T.C, C)].sigma_lower.iloc[0]
        r = pi[np.isclose(pi.C, C)]
        if len(r):
            col = [c for c in pi.columns if c.startswith("sigma_lower")][0]
            chk.append(dict(C=C, mine=mine, rb1=float(r[col].iloc[0]), abs_diff=abs(mine - float(r[col].iloc[0]))))
    return dict(table=T, info=info, check=pd.DataFrame(chk))
