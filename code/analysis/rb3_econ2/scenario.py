"""scenario.py -- frontier data demand when frontier over-training follows the 2023-2025 trend in the revealed wedge
(R3 N6(1)), at Epoch's published and our estimated compute growth (R3 minor 17), and the cost of the data wall along
the projected frontier at observed allocations (R3 N6(2)).

Frontier data use: D_F(t) = m_D(t) D*(C_F(t)), where D*(C) is a technology's compute-optimal data (ra3_econ's path
technologies, read-only) and m_D(t) = D/D*(C) is the frontier's data multiple, anchored at the median of the four
disclosed dense runs above 1e25 FLOP in 2024-2026 (ra3: Llama 3.1 405B, Nemotron-4 340B, Pangu Ultra, Aramco
Metabrain). Frontier compute follows ra3's fitted trend of the running top-10 (5.06x per year; 9.4e26 FLOP in September
2026) up to today and grows at g_C from today on (g_C = 5.06 or Epoch's 4.2).

Scenarios for m_D(t), t >= today (T_NOW):
  baseline  m_D held at the anchor (Section V.A of the paper);
  trend     the value of compactness grows at its 2023-2025 rate g_w (trend.py, primary: decision units): the frontier's
            wedge w_F(t) = w_F0 g_w^(t - T_NOW), so m_D(t) = m_D0 g_w^(e(sigma*) (t - T_NOW)); at sigma* = sigma_ref this
            is the same as extrapolating tokens per parameter relative to M*(C) (sigma*-free);
  catch-up  as trend, but the frontier wedge stops at the 2025 compute-weighted aggregate wedge of the clean sample's
            decision units (5.38 on the 49 budget-level units; 5.39 on version 3's 56): a one-off multiple
            (w_2025/w_F0)^e(sigma*), reached after about 2.4 years at the primary trend.
Here w is measured under the reference technology (kappa free, sigma* = 0.70); sigma* in {0.60, 0.70, 0.74} is the
frontier curvature that converts a given value of compactness into data (R3 N6(1): 'this is where sigma* enters').

Round 3 (fix list E1(b)-(c); R3 E(b)-(c), R1 minor 13, R2 minor 9): the scenario is 'the frontier adopts the open-weight
trend'. Besides the primary trend (49 budget-level decision units, OLS) it is run for the within-developer trend, the
trend above 1e24 FLOP, the compute-weighted trend and the four flagship runs above 1e25 FLOP (variant_scenarios; file
rb3_econ2_scenarios_by_trend.csv). Table 3 keeps the capped scenario for five-year multiples; the uncapped ones are
in the appendix. The model-free IsoFLOP path slopes read rb4's minima (Chinchilla in N_F; E1(a)).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import rb3common as RC
from rb3common import G_EPOCH, HORIZON, SIGMAS, T_NOW, U_GROWTH, U_STOCK, e_of, log

import demand as DM  # noqa: E402  (ra3_econ, read-only)
import growth as GR  # noqa: E402


def _path_slopes_rb4():
    """Model-free Approach-2 path slopes (OLS of ln N* on ln C over the bracketed budgets), as ra3's
    demand.ra1_path_slopes, but on RC.RA1_BUDGETS: rb4's minima, with Chinchilla in FLOP-effective parameters (round 3,
    fix list E1(a); audit numbers_technology #7). ra3's function reads ra1's total-parameter file, so it is replaced in
    ra3's module namespace for this process only (ra3's files are not touched)."""
    b = pd.read_csv(RC.RA1_BUDGETS)
    out = {}
    for des, g in b.groupby("design", sort=False):
        a = float(np.polyfit(np.log(g["budget_C"].values), np.log(g["Nstar"].values), 1)[0])
        key = "ra1_" + des.lower().replace(", ", "_").replace(" ", "_").replace("-", "")
        out[key] = dict(a=a, k=len(g), Cmax=float(g["budget_C"].max()), label=f"{des} (model-free A2 path)")
    return out


DM.ra1_path_slopes = _path_slopes_rb4


def load_path_techs():
    T = DM.load_path_techs()
    src = os.path.relpath(RC.RA1_BUDGETS, RC.ROOT)
    for k, lab in _path_slopes_rb4().items():
        T[k].source = f"{src}; {lab['k']} budgets, C up to {lab['Cmax']:.1e}"
    return T


# Trend variants for the scenario (round 3, E1(b)-(c)): key in trend.py's table, label, curvatures.
TREND_VARIANTS = (("units_ols", "Decision units (primary)", SIGMAS),
                  ("units_fe", "Within developers", (0.70,)),
                  ("units_big", "Decision units above 1e24 FLOP", (0.70,)),
                  ("units_wls", "Decision units, compute-weighted", (0.70,)),
                  ("flagships", "Flagship runs above 1e25 FLOP (descriptive)", (0.70,)))

MAIN = ["chin_q", "chin", "meta_a2", "deepseek", "farseer", "farseer_eq3", "gadre_rw", "olmo", "marin_dclm"]
LABELS = {"chin_q": r"Chinchilla, $\kappa$ free (reference)", "chin": r"Chinchilla, $\kappa=1$",
          "meta_a2": "Llama 3 (Meta's law)", "deepseek": "DeepSeek LLM (published)", "farseer": r"Farseer, $\kappa=1$",
          "farseer_eq3": "Farseer, own form (local)", "gadre_rw": "Gadre et al., RefinedWeb", "olmo": "OLMo ladder (AI2)",
          "marin_dclm": "Marin, DCLM"}
W_TARGET_KEY = "agg_units"          # catch-up target: 2025 compute-weighted aggregate wedge of decision units
STOCKS = dict(unique=U_STOCK, effective=320e12)


def compute_trend():
    d = GR.load_epoch()
    tr, draws = GR.fit_all(d)
    return tr.set_index("sample").loc["top10"], tr, d


def lnC_path(trend, gC):
    """ln C_F(t): the fitted trend up to T_NOW, growth gC afterwards."""
    b0, b1 = float(trend["b0"]), float(trend["b1"])
    lc_now = np.log(10) * (b0 + b1 * (T_NOW - RC.T0))

    def f(t):
        t = np.asarray(t, float)
        return np.where(t <= T_NOW, np.log(10) * (b0 + b1 * (t - RC.T0)), lc_now + np.log(gC) * (t - T_NOW))
    return f, float(np.exp(lc_now))


def year_reach(tech, lnC, mD_fun, S0):
    """First t with ln m_D(t) + ln D*(C_F(t)) = ln S0 + ln(1 + g_S)(t - 2024)."""
    f = lambda t: (np.log(mD_fun(t)) + float(tech.lnD(lnC(t))) - np.log(S0)  # noqa: E731
                   - np.log1p(U_GROWTH) * (t - 2024.0))
    lo, hi = 2015.0, 2060.0
    if f(lo) > 0:
        return lo
    if f(hi) < 0:
        return np.inf
    return brentq(f, lo, hi, xtol=1e-6)


def mD_functions(mD0, g_w, sigma, w_ratio_cap=None):
    e = e_of(sigma)

    def base(t):
        return mD0

    def trend(t):
        return mD0 * g_w ** (e * max(t - T_NOW, 0.0))

    def catch(t):
        x = g_w ** max(t - T_NOW, 0.0)
        return mD0 * min(x, w_ratio_cap) ** e
    return dict(baseline=base, trend=trend, catchup=catch)


def run(TR):
    """TR: output of trend.run()."""
    trend, tr_all, epoch = compute_trend()
    g_hat = float(trend["growth"])
    T = load_path_techs()
    runs = DM.frontier_runs()
    k_ref = TR["k_ref"]
    sig_ref = 1 / (1 + k_ref)
    ref = T["chin_q"]
    mD0 = DM.data_multiple(ref, runs)
    wF0 = mD0 ** (2 * k_ref)
    # per-run reference wedges of the anchor runs
    lnDs = ref.lnD(np.log(runs["Cmp"].values))
    runs = runs.assign(D_over_Dstar_ref=runs["D"].values / np.exp(lnDs))
    runs["w_ref"] = runs["D_over_Dstar_ref"] ** (2 * k_ref)
    runs["t"] = np.nan
    Tv = TR["T"].set_index("key")
    g_w = float(Tv.loc["units_ols", "g_w"])
    g_w_lo, g_w_hi = float(Tv.loc["units_ols", "g_w_lo"]), float(Tv.loc["units_ols", "g_w_hi"])
    B = TR["B"].set_index("year")
    w_target = float(B.loc[2025, "w_agg_units"])
    cap = w_target / wF0
    log(f"scenario: m_D0 = {mD0:.3f}, w_F0 = {wF0:.3f}, g_w = {g_w:.3f} [{g_w_lo:.2f}, {g_w_hi:.2f}], "
        f"catch-up target w = {w_target:.2f} (x{cap:.2f})")
    gCs = [("epoch", G_EPOCH), ("hat", g_hat)]
    paths = {nm: lnC_path(trend, g) for nm, g in gCs}
    C_now = paths["hat"][1]

    # ---------------- Panel A: data demand by technology at 4.2x and 5.06x
    rows = []
    for k in MAIN:
        t = T[k]
        lc = np.log(C_now)
        el = t.elasticity(lc)
        r = dict(key=k, label=LABELS[k], a=1 - el, Dstar_now_T=float(np.exp(t.lnD(lc))) / 1e12,
                 data_multiple_obs=DM.data_multiple(t, runs))
        for nm, g in gCs:
            r[f"gD_{nm}"] = g ** el
            r[f"gD5_{nm}"] = g ** (HORIZON * el)
            lnC = paths[nm][0]
            for sn, S0 in STOCKS.items():
                r[f"year_{sn}_obs_{nm}"] = year_reach(t, lnC, lambda tt, m=r["data_multiple_obs"]: m, S0)
                r[f"year_{sn}_opt_{nm}"] = year_reach(t, lnC, lambda tt: 1.0, S0)
        rows.append(r)
    A = pd.DataFrame(rows)
    # a-range over the main set and ra1's model-free paths (growth only)
    dem_all = DM.demand_table(T, trend, [("hat", g_hat), ("epoch", G_EPOCH)], runs)

    # ---------------- Panel B: rising wedges (reference path)
    el_ref = ref.elasticity(np.log(C_now))
    rowsB = []
    for sig in SIGMAS:
        e = e_of(sig)
        fn = mD_functions(mD0, g_w, sig, cap)
        for scen in ("baseline", "trend", "catchup"):
            r = dict(sigma=sig, scenario=scen, e=e, g_w=g_w if scen != "baseline" else 1.0)
            r["g_DDstar"] = g_w ** e if scen == "trend" else (np.nan if scen == "catchup" else 1.0)
            r["g_DDstar_lo"] = g_w_lo ** e if scen == "trend" else np.nan
            r["g_DDstar_hi"] = g_w_hi ** e if scen == "trend" else np.nan
            r["catchup_multiple"] = cap ** e if scen == "catchup" else np.nan
            r["catchup_years"] = np.log(cap) / np.log(g_w) if scen == "catchup" else np.nan
            for nm, g in gCs:
                gd = g ** el_ref
                if scen == "baseline":
                    r[f"gD_{nm}"] = gd
                    r[f"gD5_{nm}"] = gd ** HORIZON
                elif scen == "trend":
                    r[f"gD_{nm}"] = gd * g_w ** e
                    r[f"gD_{nm}_lo"] = gd * g_w_lo ** e
                    r[f"gD_{nm}_hi"] = gd * g_w_hi ** e
                    r[f"gD5_{nm}"] = (gd * g_w ** e) ** HORIZON
                else:
                    r[f"gD_{nm}"] = np.nan
                    r[f"gD5_{nm}"] = gd ** HORIZON * cap ** e
                r[f"gD5_{nm}_vs_baseline"] = r[f"gD5_{nm}"] / gd ** HORIZON
                lnC = paths[nm][0]
                for sn, S0 in STOCKS.items():
                    r[f"year_{sn}_{nm}"] = year_reach(ref, lnC, fn[scen], S0)
            rowsB.append(r)
    Bt = pd.DataFrame(rowsB)
    # combined range over path slopes a in the main set and sigma* (per-year growth at 5.06x, trend scenario)
    a_min, a_max = float(A.a.min()), float(A.a.max())
    rng = dict(a_min=a_min, a_max=a_max,
               gD_trend_min=g_hat ** (1 - a_max) * g_w ** e_of(min(SIGMAS)),
               gD_trend_max=g_hat ** (1 - a_min) * g_w ** e_of(max(SIGMAS)),
               gD_base_min=g_hat ** (1 - a_max), gD_base_max=g_hat ** (1 - a_min))
    # M-based (sigma*-free) data-multiple growth across the 32 technologies (trend.by_technology)
    Tt = TR["Tt"]
    rng.update(g_DD_Mbased_tech_min=float(Tt.g_DDstar_Mbased.min()), g_DD_Mbased_tech_max=float(Tt.g_DDstar_Mbased.max()),
               g_w_tech_min=float(Tt.g_w.min()), g_w_tech_max=float(Tt.g_w.max()))
    # sensitivity: the wedge trend starts at the anchor runs' median release date rather than today
    m3 = pd.read_csv(os.path.join(RC.ROOT, "output", "tables", "m3_wedge_models.csv"))
    rd = m3.set_index("model").reindex(runs["model"])["date"]
    t_anchor = float(np.median(RC.dec_year(pd.Series(pd.to_datetime(rd.values)))))
    sens = []
    for sig in SIGMAS:
        e = e_of(sig)
        for nm, g in gCs:
            lnC = paths[nm][0]
            f_tr = lambda t, e=e: mD0 * g_w ** (e * max(t - t_anchor, 0.0))  # noqa: E731
            sens.append(dict(sigma=sig, gC=nm, start=t_anchor, year_unique=year_reach(ref, lnC, f_tr, STOCKS["unique"]),
                             year_effective=year_reach(ref, lnC, f_tr, STOCKS["effective"]),
                             mD_today=f_tr(T_NOW)))
    sens = pd.DataFrame(sens)
    # review addition: the catch-up target (the 2025 compute-weighted aggregate wedge) rests on seven decision units;
    # recompute it leaving out each 2025 developer in turn (Alibaba's Qwen3 family, a cap-reading lower bound, carries
    # most of the compute) and report the implied one-off multiple and five-year demand.
    import trend as TRm
    u25 = TR["u"][TR["u"].year == 2025]
    csens = []
    for drop in ["(none)"] + sorted(u25.dev.unique()):
        x = u25 if drop == "(none)" else u25[u25.dev != drop]
        wt = 1.0 / (1.0 - TRm.s_agg(x.w_ref, x.Cmp))
        cp = max(wt / wF0, 1.0)
        for sig in SIGMAS:
            e = e_of(sig)
            r = dict(dropped=drop, n_units=len(x), w_target=wt, cap=cp, sigma=sig, catchup_multiple=cp ** e,
                     catchup_years=np.log(cp) / np.log(g_w))
            for nm, g in gCs:
                r[f"gD5_{nm}"] = (g ** el_ref) ** HORIZON * cp ** e
            csens.append(r)
    csens = pd.DataFrame(csens)
    Bv = variant_scenarios(TR, ref, paths, gCs, el_ref, mD0, cap)
    out = dict(Bv=Bv, catch_sens=csens, trend=trend, tr_all=tr_all, epoch=epoch, g_hat=g_hat, T=T, runs=runs, mD0=mD0, wF0=wF0, g_w=g_w,
               g_w_lo=g_w_lo, g_w_hi=g_w_hi, w_target=w_target, cap=cap, C_now=C_now, A=A, Bt=Bt, rng=rng, sens=sens,
               t_anchor=t_anchor, paths=paths, dem_all=dem_all, sig_ref=sig_ref, el_ref=el_ref)
    return out


def variant_scenarios(TR, ref, paths, gCs, el_ref, mD0, cap):
    """Frontier data demand when the frontier adopts each open-weight trend variant (round 3, E1(b)-(c)): the uncapped
    trend (annual growth; five-year multiples, reported in the appendix) and the capped scenario in which the rise stops
    at the 2025 compute-weighted wedge of the decision units (five-year multiples, Table 3)."""
    Tv = TR["T"].set_index("key")
    rows = []
    for key, label, sigmas in TREND_VARIANTS:
        v = Tv.loc[key]
        g_w = float(v.g_w)
        lo, hi = float(v.g_w_lo), float(v.g_w_hi)
        for sig in sigmas:
            e = e_of(sig)
            fn = mD_functions(mD0, g_w, sig, cap)
            r = dict(key=key, label=label, n=int(v.n), clusters=v.clusters, g_w=g_w, g_w_lo=lo, g_w_hi=hi, sigma=sig,
                     e=e, g_DDstar=g_w ** e, g_DDstar_lo=lo ** e if np.isfinite(lo) else np.nan,
                     g_DDstar_hi=hi ** e if np.isfinite(hi) else np.nan,
                     years_to_cap=np.log(cap) / np.log(g_w) if g_w > 1 else np.inf, cap_multiple=cap ** e)
            for nm, g in gCs:
                gd = g ** el_ref
                r[f"gD_{nm}"] = gd * g_w ** e
                r[f"gD_{nm}_lo"] = gd * lo ** e if np.isfinite(lo) else np.nan
                r[f"gD_{nm}_hi"] = gd * hi ** e if np.isfinite(hi) else np.nan
                r[f"gD5_uncapped_{nm}"] = (gd * g_w ** e) ** HORIZON
                r[f"gD5_capped_{nm}"] = gd ** HORIZON * min(g_w ** HORIZON, cap) ** e
                lnC = paths[nm][0]
                for sn, S0 in STOCKS.items():
                    r[f"year_{sn}_trend_{nm}"] = year_reach(ref, lnC, fn["trend"], S0)
                    r[f"year_{sn}_capped_{nm}"] = year_reach(ref, lnC, fn["catchup"], S0)
            rows.append(r)
    return pd.DataFrame(rows)
