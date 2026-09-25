"""run.py -- module rb3_econ2: integrating Section V (economics) with Section IV (revealed value of compactness).

Addresses R3 round-2 N6 (1)-(3) and minors 17-20 (plus R1 minors 17-18, R2 minors 14/20, R4 minors 9-10):
  1. trend.py     the 2023-2025 trend in the revealed wedge (clean sample, decision units; rb2_decisions' reviewed files);
  2. scenario.py  frontier data demand when over-training follows that trend (D/D* = w^(sigma*/[2(1-sigma*)]),
                  sigma* = 0.60/0.70/0.74), at 4.2x and 5.06x compute growth; bounded 'catch-up' variant;
  3. wallobs.py   the data wall at OBSERVED allocations (lifetime cost with a value of compactness), the prediction that
                  measured wedges understate the value of compactness as the wall binds (Prop. 1(iii)), the projected
                  frontier path, today's frontier by technology;
  4. vintage.py   reconciling 5.06x with Epoch's published 4.2x (database of 31 May 2024 vs 23 September 2026);
  5. fleet.py     fleet shares with service lives of 1, 2 and 3 years;
  6. gamma.py     guidance on gamma for growth calibrations at 1e25-1e27 FLOP (E-profiles; compute per halving);
  7. synth.py     the shadow value of a unique token beside synthetic-token costs and posted API prices (optional).
Builds on ra3_econ (imported read-only; outputs never written) and rb2_decisions (processed files read-only).
Deterministic (fixed seeds), CPU only, single process, about 15 seconds.

Round 3 (fix list E1-E6, 2026-09-25): the decision units are rb5_units' 49 budget-level units
(data/processed/rb5_units/units_primary.csv); new trend rows (within developers, flagship runs) and scenario rows by
trend variant (rb3_econ2_scenarios_by_trend.csv); the 32-technology trend on audited token counts; model-free path slopes
and E-profiles on rb4's minima (Chinchilla in N_F); the high-wedge point of the wall grids follows the 2025 aggregate.
Usage: .venv/bin/python code/analysis/rb3_econ2/run.py
"""
from __future__ import annotations

import json
import os
import time
import warnings

import numpy as np
import pandas as pd

import rb3common as RC
from rb3common import P, PROC, TABLES, log

warnings.filterwarnings("ignore", category=RuntimeWarning)

import exhibits as EX  # noqa: E402
import fleet as FL  # noqa: E402
import gamma as GM  # noqa: E402
import scenario as SC  # noqa: E402
import synth as SY  # noqa: E402
import trend as TRm  # noqa: E402
import vintage as VT  # noqa: E402
import wallobs as WO  # noqa: E402


def save(df, name, **kw):
    df.to_csv(os.path.join(TABLES, P + name), index=False, **kw)


def main():
    t0 = time.time()
    np.random.seed(RC.SEED)
    # 1. wedge trend
    TR = TRm.run()
    save(TR["T"], "wedge_trend.csv")
    save(TR["L"], "wedge_trend_lodo.csv")
    save(TR["Tt"], "wedge_trend_by_tech.csv")
    save(TR["B"], "wedge_by_year.csv")
    save(TR["chk"], "wedge_checks.csv")
    if len(TR["chk"]) == 0 or not TR["chk"]["passed"].all():
        raise RuntimeError("unit table or rb2_decisions' by-year aggregates not reproduced: see rb3_econ2_wedge_checks.csv")
    RC.W_AGG25 = round(float(TR["B"].set_index("year").loc[2025, "w_agg_units"]), 2)
    log(f"   2025 compute-weighted wedge of the decision units: {RC.W_AGG25:.2f} (high point of the wall grids)")
    log(f"1. trend done ({time.time() - t0:.0f}s)")
    # 2. scenarios
    S_ = SC.run(TR)
    save(S_["A"], "demand.csv")
    save(S_["Bt"], "scenarios.csv")
    save(S_["sens"], "scenarios_start_sensitivity.csv")
    save(S_["runs"], "frontier_anchor_runs.csv")
    save(S_["dem_all"], "demand_all_technologies.csv")
    save(S_["catch_sens"], "scenarios_catchup_sensitivity.csv")
    save(S_["Bv"], "scenarios_by_trend.csv")
    pd.DataFrame([S_["rng"] | dict(mD0=S_["mD0"], wF0=S_["wF0"], g_w=S_["g_w"], g_w_lo=S_["g_w_lo"], g_w_hi=S_["g_w_hi"],
                                   w_target=S_["w_target"], cap=S_["cap"], C_now=S_["C_now"], g_hat=S_["g_hat"],
                                   t_anchor=S_["t_anchor"])]).to_csv(os.path.join(TABLES, P + "scenario_constants.csv"),
                                                                     index=False)
    log(f"2. scenarios done ({time.time() - t0:.0f}s)")
    # 3. the wall at observed allocations
    TT = WO.techs()
    chk = WO.checks(TT)
    save(chk, "wall_checks.csv")
    if not chk["passed"].all():
        raise RuntimeError("wall checks failed: see rb3_econ2_wall_checks.csv")
    usd = SY.usd_per_flop()
    W = dict(co=WO.compute_optimal_panel(TT), grid=WO.grid(TT), under=WO.understatement_grid(TT),
             fp=WO.frontier_path(S_, TT, usd["amortized"]), today=WO.compute_optimal_vs_observed(TT, S_, usd["amortized"]),
             mainwall=WO.main_wall_rows(TT, usd["amortized"]))
    save(W["co"], "wall_compute_optimal.csv")
    save(W["grid"], "wall_observed_grid.csv")
    save(W["under"], "wall_understatement.csv")
    save(W["fp"], "frontier_wall.csv")
    save(W["today"], "wall_today.csv")
    save(W["mainwall"], "wall_main_rows.csv")
    log(f"3. wall done ({time.time() - t0:.0f}s)")
    # 4. compute growth by vintage
    V = VT.run()
    save(V["R"], "compute_vintage.csv")
    save(V["D"], "compute_vintage_decomposition.csv")
    save(V["Chg"], "compute_vintage_changes.csv")
    pd.DataFrame([V["info"]]).to_csv(os.path.join(TABLES, P + "compute_vintage_info.csv"), index=False)
    log(f"4. vintage done ({time.time() - t0:.0f}s)")
    # 5. fleet
    F = FL.run(TR, S_["g_hat"])
    save(F, "fleet.csv")
    # 6. gamma
    G = GM.run(S_["g_hat"])
    save(G["Cand"], "gamma.csv")
    save(G["Prof"], "gamma_eprofiles.csv")
    save(G["ProfT"], "gamma_eprofiles_T.csv")
    G["P"].to_csv(os.path.join(PROC, "gamma_eprofile_grid.csv"), index=False)
    save(G["Loc"], "gamma_local.csv")
    # 7. synthetic tokens
    SYr = SY.run(S_["C_now"])
    save(SYr["S"], "synthetic_shadow.csv")
    save(SYr["BE"], "synthetic_breakeven.csv")
    log(f"5-7. fleet, gamma, synthetic done ({time.time() - t0:.0f}s)")
    # exhibits
    EX.main_table(TR, S_, dict(W, fleet=F), G)
    EX.app_growth(V, TR, S_)
    EX.app_trend(TR, S_)
    EX.app_wall(W)
    EX.app_gamma(G, F, SYr)
    EX.figure(S_, W, SYr, TR)
    summ = dict(g_hat=S_["g_hat"], C_now=S_["C_now"], mD0=S_["mD0"], wF0=S_["wF0"], g_w=S_["g_w"],
                g_w_ci=[S_["g_w_lo"], S_["g_w_hi"]], w_target=S_["w_target"], w_agg25_grid=RC.W_AGG25,
                units_file=os.path.relpath(RC.RB5_UNITS, RC.ROOT), n_units=int(len(TR["u"])),
                ra1_budgets_file=os.path.relpath(RC.RA1_BUDGETS, RC.ROOT), usd_per_flop=usd,
                vintage=V["R"][["sample", "growth", "growth_lo", "growth_hi"]].to_dict("records"),
                vintage_decomposition=V["D"].to_dict("records"),
                gamma_recommended=G["Cand"].iloc[0][["gamma", "se", "mult_halving", "mult_lo", "mult_hi", "years_4p2",
                                                     "years_5p06"]].to_dict())
    with open(os.path.join(PROC, "summary.json"), "w") as fh:
        json.dump(summ, fh, indent=1, default=float)
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
