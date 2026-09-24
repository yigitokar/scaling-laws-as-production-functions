"""Step 3 of ra4_obsfix: algorithmic-progress material for Online Appendix E.

(a) R4 M4(f): the interpolated profile-likelihood interval for the parameter-augmenting share phi = g_N/g_C is
    [-0.39, 3.37], but the "T_C from 5 to 38 months" statement was computed over the grid-point interval
    [-0.25, 3.25].  We re-optimize the 1-D phi profile on a fine grid around both boundaries and around the peak of
    g_C (the doubling time is not monotone in phi), locate the crossings again, and report the T_C range over the
    interpolated interval.  Same estimator as m5 (unpenalized NLS, m5_progress.common.fit with 6 random starts).
(b) R1 minor 46: condensed Ho et al. replication + ridge summary (read from m5 outputs; no re-estimation).
(c) R4 M8(c): the share of the Step Law / Bjorck learning-rate gap closed by conditioning on the batch, for the
    range of Bjorck et al.'s exponents (0.32 for N >= 760M; 0.38 at 350M; 0.65 at 125M; 0.70 at 50M), from the
    m8_measurement bootstrap summary (the share is linear in the gap, so it rescales exactly).
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from multiprocessing import get_context

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ra4common as rc  # noqa: E402

sys.path.insert(0, rc.M5DIR)
import common as m5c  # noqa: E402

M5PROC = os.path.join(rc.ROOT, "data", "processed", "m5_progress")
M8PROC = os.path.join(rc.ROOT, "data", "processed", "m8_measurement")
N_PROC = 4
STEPLAW_LR_D = 0.307                       # Step Law published D-elasticity of the optimal LR (li2025predictablea)
# arXiv 2409.19913v3, Table 5 (per-size fits at a 0.5M-token batch) and Sec. 4 (joint fit over 760M-2.7B: beta ~ 0.32).
# Reviewer addition: the individual >= 760M fits (0.3155, 0.3171, 0.4184) are listed too; Step Law's cells span
# 215M-1.07B non-embedding parameters, so the exponents that bracket its range run from ~0.32 (760M-1.3B) to 0.65 (125M);
# 50M (0.70) and 2.7B (0.42) lie outside it.
BJORCK = {"N >= 760M (published headline)": 0.32, "760M": 0.3155, "1.3B": 0.3171, "2.7B": 0.4184, "350M": 0.3799,
          "125M": 0.6531, "50M": 0.7029}
BJORCK_IN_STEPLAW_RANGE = {"N >= 760M (published headline)", "760M", "1.3B", "350M", "125M"}


def _phi_fit(args):
    val, data, norm = args
    m = m5c.HoModel(m5c.HoSpec(delta=0, phi_fixed=float(val)), data, norm)
    x, f = m5c.fit(m, n_random=6, seed=3)
    r = m.rates(x)
    return dict(value=float(val), mse=float(f), g_N=r["g_N"], g_D=r["g_D"], g_C=r["g_C"], alpha_year=r["alpha_year"],
                beta_year=r["beta_year"])


def lr_crossing(vals, LR, crit=3.841):
    v, L = np.asarray(vals, float), np.asarray(LR, float)
    inside = np.where(L <= crit)[0]
    i0, i1 = inside.min(), inside.max()
    lo = np.nan if i0 == 0 else v[i0 - 1] + (v[i0] - v[i0 - 1]) * (L[i0 - 1] - crit) / (L[i0 - 1] - L[i0])
    hi = np.nan if i1 == len(v) - 1 else v[i1] + (v[i1 + 1] - v[i1]) * (crit - L[i1]) / (L[i1 + 1] - L[i1])
    return float(lo), float(hi)


def phi_profile():
    df = m5c.load_ho()
    d = m5c.ho_arrays(df)
    norm = m5c.HoModel(m5c.HoSpec(delta=0.0025), d).norm
    summ = json.load(open(os.path.join(M5PROC, "dmr_summary.json")))
    mse_min = float(summ["mse_min"])
    n = len(df)
    old = pd.read_csv(os.path.join(M5PROC, "dmr_profile_1d.csv"))
    old = old[old.kind == "phi"][["value", "mse", "g_N", "g_D", "g_C", "alpha_year", "beta_year"]].assign(source="m5 grid")
    new_vals = np.round(np.r_[np.arange(-0.50, 0.001, 0.025), np.arange(3.15, 3.501, 0.025)], 4)
    new_vals = [v for v in new_vals if not np.isclose(old.value, v).any()]
    with get_context("spawn").Pool(N_PROC) as pool:
        res = pool.map(_phi_fit, [(v, d, norm) for v in new_vals])
    new = pd.DataFrame(res).assign(source="ra4 refinement")
    P = pd.concat([old, new], ignore_index=True).sort_values("value").reset_index(drop=True)
    P["LR"] = n * np.log(P.mse / mse_min)
    P["TC_months"] = np.where(P.g_C > 0, 12 * np.log(2) / P.g_C, np.inf)
    lo, hi = lr_crossing(P.value, P.LR)
    inside = P[(P.value >= lo) & (P.value <= hi)]
    # T_C at the interpolated boundaries: linear interpolation of g_C between the bracketing grid points
    gc_lo = float(np.interp(lo, P.value, P.g_C))
    gc_hi = float(np.interp(hi, P.value, P.g_C))
    tc_vals = np.r_[inside.TC_months.to_numpy(), 12 * np.log(2) / gc_lo, 12 * np.log(2) / gc_hi]
    grid_pts = P[P.LR <= 3.841]
    out = dict(n=n, mse_min=mse_min, phi_ci_interp=(lo, hi), phi_ci_gridpoints=(float(grid_pts.value.min()), float(grid_pts.value.max())),
               TC_range_over_interp_ci=(float(np.min(tc_vals)), float(np.max(tc_vals))),
               TC_at_boundaries=(12 * np.log(2) / gc_lo, 12 * np.log(2) / gc_hi),
               TC_range_over_gridpoint_ci=(float(grid_pts.TC_months.min()), float(grid_pts.TC_months.max())),
               phi_at_min_TC=float(inside.value.iloc[int(np.argmin(inside.TC_months))]),
               m5_reported=dict(phi_ci_interp=summ["phi_profile_ci_interp"], phi_ci_grid=summ["phi_profile_ci"],
                                TC_range_within_phi_ci=summ["TC_range_within_phi_ci"]),
               n_new_fits=len(new_vals))
    P.to_csv(os.path.join(rc.TAB, f"{rc.PREFIX}_phi_profile.csv"), index=False)
    return out


def ho_summary():
    rep = pd.read_csv(os.path.join(rc.TAB, "m5_progress_replication.csv")).set_index("parameter")
    A = pd.read_csv(os.path.join(rc.TAB, "m5_progress_table7_panelA.csv")).set_index("code")
    summ = json.load(open(os.path.join(M5PROC, "dmr_summary.json")))
    ms = json.load(open(os.path.join(M5PROC, "multistart_summary.json")))
    return rep, A, summ, ms


def bjorck_shares():
    j = json.load(open(os.path.join(M8PROC, "results_summary.json")))
    bt = j["lr_decomposition_boot"]
    rows = []
    gap0 = STEPLAW_LR_D + 0.32
    for opt in ("smoothed", "grid"):
        s = bt[opt]["share_of_published_gap_closed"]
        red = bt[opt]["e_D_unconditional"]["point"] - bt[opt]["e_D_conditional"]["point"]
        for lab, b in BJORCK.items():
            gap = STEPLAW_LR_D + b
            f = gap0 / gap
            rows.append(dict(optimum=opt, bjorck_regime=lab, bjorck_exponent=b, gap=gap, reduction=red,
                             share=s["point"] * f, share_se=s["se"] * f, share_q05=s["q05"] * f, share_q95=s["q95"] * f,
                             check=red / gap, brackets_steplaw_range=lab in BJORCK_IN_STEPLAW_RANGE))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(rc.TAB, f"{rc.PREFIX}_bjorck_shares.csv"), index=False)
    return R, rc.file_info(os.path.join(M8PROC, "results_summary.json"))


def main():
    t0 = time.time()
    phi = phi_profile()
    rc.log(f"phi profile: CI {np.round(phi['phi_ci_interp'], 3)}; T_C over interpolated CI {np.round(phi['TC_range_over_interp_ci'], 2)} "
           f"({time.time() - t0:.0f}s)")
    rep, A, summ, ms = ho_summary()
    R, m8info = bjorck_shares()
    corr = {}
    for f in ("boot_m7_ho1000", "boot_m7_cluster", "boot_m7_conv_cluster", "boot_m7_conv_iid1000"):
        p_ = os.path.join(M5PROC, f + ".npy")
        if os.path.exists(p_):
            b = np.load(p_)
            corr[f] = dict(B=int(len(b)), corr_alpha_year_beta_year=float(np.corrcoef(b[:, 3], b[:, 8])[0, 1]))
    rep_s = json.load(open(os.path.join(M5PROC, "replication_summary.json")))
    rc.dump_json(dict(runtime_sec=time.time() - t0, phi=phi, m8_source=m8info, boot_corr=corr, multistart=ms,
                      replication=rep_s, dmr=dict((k, summ[k]) for k in ("neutrality", "TC_profile_ci_interp", "gC_profile_ci_interp",
                                                                         "s_bar", "a_onpath_Ho", "mse_min", "slsqp_default_nit")),
                      m5_sources={k: rc.file_info(os.path.join(rc.TAB, k)) for k in
                                  ("m5_progress_replication.csv", "m5_progress_table7_panelA.csv")},
                      dmr_summary=rc.file_info(os.path.join(M5PROC, "dmr_summary.json"))), "progress_headline.json")
    rc.log(f"step_progress finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
