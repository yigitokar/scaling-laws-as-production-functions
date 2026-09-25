"""exhibits.py -- paper-ready tables (booktabs, \\footnotesize, AEA.cls tablenotes) and the revised appendix figure for
module rb3_econ2."""
from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import aer_style as S
import rb3common as RC
from rb3common import P, TABLES, e_of

matplotlib.use("Agg")


def f(x, d=2, comma=True):
    if x is None:
        return "--"
    try:
        if not np.isfinite(x):
            return "$\\infty$" if (isinstance(x, float) and np.isinf(x) and x > 0) else "--"
    except TypeError:
        return str(x)
    s = f"{x:,.{d}f}" if comma else f"{x:.{d}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def sci(x, d=1):
    m, e = f"{x:.{d}e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def big(x):
    """Multiples: 1 decimal below 10, integers with thousands separators above."""
    if not np.isfinite(x):
        return "--"
    return f"{x:.1f}" if x < 10 else f"{x:,.0f}"


def yr(x):
    if not np.isfinite(x):
        return "--"
    if x <= 2015.0:
        return "$<$2015"
    return f"{x:.1f}"


def ci2(lo, hi):
    """95 percent interval of a growth factor: two decimals below 10, one above."""
    g = lambda v: f"{v:.2f}" if v < 10 else f"{v:.1f}"  # noqa: E731
    return f"[{g(lo)}, {g(hi)}]"


# Round 3 (fix list E1(c)): short labels of the trend variants in Table 3, panel B and Table F1, panel D.
TREND_SHORT = {"units_ols": "Decision units", "units_fe": "Within developers", "units_big": "Units above $10^{24}$ FLOP",
               "units_wls": "Compute-weighted units", "flagships": "Flagships above $10^{25}$ FLOP"}


def pct(x, d=1):
    if x is None or not np.isfinite(x):
        return "$\\infty$" if (x is not None and np.isinf(x)) else "--"
    v = 100 * x
    if abs(v) < 0.05 and v != 0:
        return "$<$0.1"
    return f(v, d)


def _write(name, lines):
    with open(os.path.join(TABLES, P + name), "w") as fh:
        fh.write("\n".join(lines) + "\n")


# ============================================================================================ main table (Table 3)
MAIN_TABLE_A = ("chin_q", "chin", "meta_a2", "farseer_eq3", "gadre_rw", "olmo")   # a from 0.36 to 0.57
def main_table(TR, S_, W, G):
    A, Bt = S_["A"], S_["Bt"]
    L = []
    L += ["% Table 3 (revised) -- Economic implications: data demand, the data wall and the revealed wedge (Section V).",
          "% Module rb3_econ2 (code/analysis/rb3_econ2/run.py); sources: rb3_econ2_demand.csv, _scenarios.csv,",
          "% _wall_compute_optimal.csv, _frontier_wall.csv, _wedge_by_year.csv, _wedge_trend.csv, _fleet.csv, _gamma.csv.",
          "\\begin{table}[tp]", "\\centering",
          "\\caption{Economic Implications: Data Demand, the Data Wall and the Revealed Wedge}", "\\label{tab:econ}",
          "\\footnotesize", "\\setlength{\\tabcolsep}{1.8pt}", "\\renewcommand{\\arraystretch}{0.96}",
          "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lccccccc@{}}", "\\toprule"]
    # ---------------- Panel A
    L += ["\\multicolumn{8}{@{}l}{\\textit{Panel A. Compute-optimal data demand}} \\\\[2pt]",
          " & & $D^*$ in & \\multicolumn{2}{c}{$D^*$ per year} & \\multicolumn{2}{c}{$D^*$ over 5 years} & 100T \\\\",
          " & & 2026 & \\multicolumn{2}{c}{if $C$ grows} & \\multicolumn{2}{c}{if $C$ grows} & reached \\\\",
          "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
          "Technology & $a$ & (T) & $4.2\\times$ & $5.06\\times$ & $4.2\\times$ & $5.06\\times$ & (year) \\\\",
          "\\midrule"]
    for _, r in A[A.key.isin(MAIN_TABLE_A)].iterrows():
        L.append(f"{r.label} & {r.a:.2f} & {big(r.Dstar_now_T)} & {r.gD_epoch:.2f} & {r.gD_hat:.2f} & {big(r.gD5_epoch)} & "
                 f"{big(r.gD5_hat)} & {yr(r.year_unique_obs_hat)} \\\\")
    # ---------------- Panel B (round 3, fix list E1(c); R3 E(b)-(c), R1 minor 13, R2 minor 9): the frontier adopts an
    # open-weight trend; one row per trend variant at sigma* = 0.70 (three curvatures for the primary); five-year
    # columns show only the capped scenario (the uncapped multiples are in Table F1, panel D).
    Bv = S_["Bv"]
    L += ["\\midrule",
          "\\multicolumn{8}{@{}l}{\\textit{Panel B. Scenario: the frontier adopts the open-weight trend}} \\\\[2pt]",
          " & & $D/D^*$ & \\multicolumn{2}{c}{$D$ per year} & \\multicolumn{2}{c}{$D$ over 5 years$^a$} & 100T \\\\",
          " & & per & \\multicolumn{2}{c}{if $C$ grows} & \\multicolumn{2}{c}{if $C$ grows} & reached \\\\",
          "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
          "Trend: $w$ per year [95\\% CI] & $\\sigma^*$ & year & $4.2\\times$ & $5.06\\times$ & $4.2\\times$ & $5.06\\times$ & (year) \\\\",
          "\\midrule"]
    b0 = Bt[(Bt.scenario == "baseline") & np.isclose(Bt.sigma, 0.70)].iloc[0]
    L.append(f"None: wedge held ($w={S_['wF0']:.2f}$) & -- & 1.00 & {b0.gD_epoch:.2f} & {b0.gD_hat:.2f} & "
             f"{big(b0.gD5_epoch)} & {big(b0.gD5_hat)} & {yr(b0.year_unique_hat)} \\\\")
    seen = set()
    for _, r in Bv.iterrows():
        if r.key in seen:
            lab = ""
        elif np.isfinite(r.g_w_lo):
            lab = f"{TREND_SHORT[r.key]}: {r.g_w:.2f} {ci2(r.g_w_lo, r.g_w_hi)}"
        else:
            lab = f"{TREND_SHORT[r.key]}: {r.g_w:.2f}$^b$"
        seen.add(r.key)
        L.append(f"{lab} & {r.sigma:.2f} & {r.g_DDstar:.2f} & {r.gD_epoch:.2f} & {r.gD_hat:.2f} & "
                 f"{big(r.gD5_capped_epoch)} & {big(r.gD5_capped_hat)} & {yr(r.year_unique_trend_hat)} \\\\")
    # ---------------- Panel C: the data wall at compute-optimal and at observed allocations
    Mw = W["mainwall"]
    Fp = W["fp"]
    Fp = Fp[(Fp.gC == "hat") & (Fp.stock == "100T") & np.isclose(Fp.sigma, 0.70)]
    L += ["\\midrule",
          "\\multicolumn{8}{@{}l}{\\textit{Panel C. The data wall at compute-optimal and at observed allocations}} \\\\[2pt]",
          " & & & \\multicolumn{3}{c}{Extra cost (\\%), repetition:} & Shadow & Measured \\\\",
          " & & & $R^*_D=$ & $R^*_D=$ & Full & value & wedge \\\\",
          "\\cmidrule(lr){4-6}",
          " & $\\sigma^*$ & Scarcity & 15.4 & 2.9 & model & (\\$/M) & ratio$^c$ \\\\", "\\midrule",
          "\\multicolumn{8}{@{}l}{\\quad\\textit{Compute-optimal run ($w=1$), $10^{26}$ FLOP; scarcity $r=D^*(C)/U$}} \\\\"]
    # round 3 (page fit): the r = 16 row and the 2026 and 2028 frontier rows are in Online Appendix Table F-wall and
    # rb3_econ2_frontier_wall.csv; Table 3 keeps r = 4 at three curvatures and the 2029 frontier (both cited in the text)
    for key, r, lab in (("q60", 4.0, "\\quad $r=4$"), ("kq", 4.0, ""), ("q74", 4.0, "")):
        x = Mw[(Mw.tech == key) & np.isclose(Mw.r, r)].iloc[0]
        L.append(f"{lab} & {x.sigma_star:.2f} & {r:.0f} & {pct(x.cost_D15, 1 if x.cost_D15 < 1 else 0)} & "
                 f"{pct(x.cost_D3, 1 if x.cost_D3 < 1 else 0)} & {pct(x.cost_DN, 1 if x.cost_DN < 1 else 0)} & "
                 f"{f(x.shadow_usd_per_Mtok, 1 if x.shadow_usd_per_Mtok < 100 else 0)} & {f(x.ratio_proc)} \\\\")
    L.append("\\multicolumn{8}{@{}l}{\\quad\\textit{Projected frontier at its observed allocation, 100T stock; scarcity $D/U$}} \\\\")
    for scen, y in (("baseline", 2029.0), ("trend", 2029.0)):
        a = {sp: Fp[(Fp.scenario == scen) & np.isclose(Fp.year, y) & (Fp.spec == sp)].iloc[0] for sp in ("D15", "D3", "DN")}
        x = a["D15"]
        lab = ("2026 (Sep.)" if np.isclose(y, RC.T_NOW) else f"{y:.0f}") + (
            f", wedge held ($w={x.w_sigma:.2f}$)" if scen == "baseline" else f", wedge rises ($w={x.w_sigma:.2f}$)")
        L.append(f"\\quad {lab} & 0.70 & {x.r_obs:.1f} & {pct(x.life_penalty)} & {pct(a['D3'].life_penalty)} & "
                 f"{pct(a['DN'].life_penalty)} & {f(x.shadow_usd_per_Mtok, 1 if x.shadow_usd_per_Mtok < 100 else 0)} & "
                 f"{f(x.ratio_proc)} \\\\")
    B = TR["B"].set_index("year")
    L += ["\\bottomrule", "\\end{tabular*}", "\\vspace{2pt}", "\\begin{tablenotes}[Notes]"]
    g0 = G["Cand"].set_index("key").loc["chin_q"]
    rng = S_["rng"]
    wy = " (2023), ".join(f"{B.loc[y, 'w_agg_units']:.2f}" for y in (2023, 2024)) + f" (2024) and {B.loc[2025, 'w_agg_units']:.2f} (2025)"
    fr = Fp[(Fp.scenario == "trend") & (Fp.spec == "D15")].set_index("year")
    whw = f"{fr.loc[2029.0, 'ratio_proc_vs_w']:.2f} in 2029"
    wy3 = ", ".join(f"{B.loc[y, 'w_agg_units']:.2f}" for y in (2023, 2024, 2025))
    # review fix (R4 round-2 minor 9): flag the two technologies that set the top of the growth range and give the
    # range over the IsoFLOP model-free paths (ra1; Porian et al.'s unannealed profiles excluded, as in Section III)
    cs = S_["catch_sens"]
    cs_noali = float(cs[cs.dropped == "Alibaba"].w_target.iloc[0])
    da = S_["dem_all"]
    mf = da[(da.group == "ra1 model-free A2") & ~da.key.str.contains("porian")]
    olmo = da.set_index("key").loc["olmo"]
    Tv = TR["T"].set_index("key")
    nbig, dbig = int(Tv.loc["units_big", "n"]), int(Tv.loc["units_big", "developers"])
    fl_top, fl_rate = Tv.loc["flagships", "last_run"], f"{float(Tv.loc['flagships', 'g_w_without_last']):.2f}"
    bu = Bv[Bv.key == "units_ols"]
    yrs_cap = Bv.years_to_cap
    notes = (
        "$C$ grows 4.2-fold a year (Epoch's published rate) or 5.06-fold (our estimate). "
        "Panel A: $a$, allocation exponent ($D^*\\propto C^{1-a}$); Farseer's own form: local slope at "
        "$10^{22}$--$10^{23}$ FLOP, beyond its design; OLMo: bootstrap interval for $a$ "
        f"[{olmo.a_lo:.2f}, {olmo.a_hi:.2f}]; model-free IsoFLOP paths ($a={mf.a.min():.2f}$--{mf.a.max():.2f}): "
        f"{mf.gD_epoch.min():.2f}--{mf.gD_epoch.max():.2f} and {mf.gD_hat.min():.2f}--{mf.gD_hat.max():.2f}-fold a year; "
        f"$D^*$ at the fitted frontier compute of September 2026 ({sci(S_['C_now'])} FLOP); 100T reached (at "
        "$5.06\\times$): year frontier runs, at the median $D/D^*$ of the four disclosed dense runs above $10^{25}$ "
        "FLOP, reach the quality-adjusted stock of public text (100T tokens in 2024, growing 5 percent a year). "
        f"Panel B: from September 2026 the frontier's wedge ($w={S_['wF0']:.2f}$, $D/D^*={S_['mD0']:.2f}$) grows at an "
        "open-weight trend of 2023--2025 (OLS of $\\ln w$ on release date, with 95 percent wild cluster bootstrap-$t$ "
        f"intervals by developer; {dbig} developers above $10^{{24}}$ FLOP; Online Appendix "
        "Table~\\ref{tab:app-econ-trend}) and $D/D^*=w^{\\sigma^*/[2(1-\\sigma^*)]}$; tokens per parameter "
        "give the $\\sigma^*=0.70$ row whatever $\\sigma^*$ "
        f"({rng['g_DD_Mbased_tech_min']:.2f}--{rng['g_DD_Mbased_tech_max']:.2f}-fold a year across 32 technologies). "
        f"$^a$The rise stops at the 2025 wedge ($w={S_['w_target']:.2f}$; {cs_noali:.2f} without Alibaba's Qwen3), "
        f"reached after {yrs_cap.min():.1f} to {yrs_cap.max():.1f} years. "
        f"$^b$Four disclosed dense runs above $10^{{25}}$ FLOP, 2024--2025; descriptive ({fl_rate} without {fl_top}). "
        "Panel C: the developer minimizes training cost plus a value of compactness proportional to $N$ at the loss "
        "of its uncapped choice; extra cost relative to that choice; repetition model of "
        "\\citet{muennighoff2023scaling} (full model: excess parameters also decay); shadow value of a unique token "
        "at \\$$10^{-18}$ per FLOP; "
        "$^c\\hat w/(1+m_N)$: the wedge measured from the capped choice relative to one plus the value of compactness "
        f"there ($\\hat w/w$ is {whw} on the rising path); "
        "$\\sigma^*=0.60$, 0.74: same path and frontier as the reference; frontier rows: $5.06\\times$.")
    L += [notes, "\\end{tablenotes}", "\\begin{tablenotes}[Source]",
          "\\citet{epochai2026data}; \\citet{sevilla2024training}; \\citet{hoffmann2022training}; "
          "\\citet{besiroglu2024chinchilla}; \\citet{grattafiori2024llama}; \\citet{li2025predictableb}; "
          "\\citet{gadre2024language}; \\citet{bhagia2024establishing}; \\citet{villalobos2022run}; "
          "\\citet{muennighoff2023scaling}; author's calculations.", "\\end{tablenotes}", "\\end{table}"]
    _write("table.tex", L)


# ============================================================================================ appendix tables
def app_growth(V, TR, S_=None):
    """Online Appendix table: frontier compute growth by database vintage (round 3: the wedge-trend panels moved to
    app_trend, a table of their own, for page fit)."""
    R, D = V["R"], V["D"].set_index("component")
    L = ["% Online Appendix table -- frontier compute growth by database vintage.",
         "% Module rb3_econ2; sources rb3_econ2_compute_vintage.csv, _compute_vintage_decomposition.csv.",
         "\\begin{table}[tp]", "\\centering",
         "\\caption{Frontier Compute Growth by Database Vintage}",
         "\\label{tab:app-econ-growth}", "\\footnotesize", "\\setlength{\\tabcolsep}{2pt}",
         "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrrccc@{}}", "\\toprule",
         "\\multicolumn{6}{@{}l}{\\textit{Growth of the running top-10 training runs, January 2018 to May 2024 (times per year)}} \\\\[2pt]",
         "Sample & Runs & Developers & Growth & 95\\% CI & $p$ ($4.2\\times$) \\\\", "\\midrule",
         "Epoch AI, published estimate & -- & -- & 4.2 & [3.6, 4.9]$^a$ & -- \\\\"]
    labs = {"v24": "Database of 31 May 2024", "v24_notable": "\\quad notable models only",
            "v24_rev": "\\quad + compute revisions to Sep. 2026", "v24_rev_notable": "\\qquad notable models only",
            "v24_add": "\\quad + entries added after May 2024", "v24_add_notable": "\\qquad notable models only",
            "v26": "Database of 23 Sep. 2026", "v26_notable": "\\quad notable models only",
            "v26_full": "\\quad January 2018 to September 2026 (Section~\\ref{sec:econ})"}
    for k in ("v24", "v24_notable", "v24_rev", "v24_rev_notable", "v24_add", "v24_add_notable", "v26", "v26_notable",
              "v26_full"):
        r = R.set_index("key").loc[k]
        L.append(f"{labs[k]} & {int(r.n)} & {int(r.clusters)} & {r.growth:.2f} & [{r.growth_lo:.2f}, {r.growth_hi:.2f}] & "
                 f"{r.p_wcr_4p2:.3f} \\\\")
    g_ = R.set_index("key")["growth"]
    L.append(f"\\multicolumn{{6}}{{@{{}}l}}{{Decomposition of $\\ln({g_['v26']:.2f}/{g_['v24']:.2f})$: compute revisions "
             f"{D.loc['revisions', 'share']:.2f}, entries added {D.loc['additions', 'share']:.2f} (Shapley)}} \\\\")
    L += ["\\bottomrule", "\\end{tabular*}", "\\vspace{2pt}", "\\begin{tablenotes}[Notes]",
          "OLS of $\\log_{10}C$ on release date for models in the running top 10 of training compute at release "
          "(all domains); 95 percent wild cluster bootstrap-$t$ intervals by developer (Webb weights, $B=9{,}999$); "
          "$p$: restricted wild cluster bootstrap test of 4.2-fold growth. The database of 31 May 2024 is Epoch AI's "
          "public file as captured by the Internet Archive three days after the published estimate "
          "\\citep{epochai2024dataarchive}; the database of 23 September 2026 is \\citet{epochai2026data}. Compute revisions: "
          f"models present in both vintages, with the 2026 estimate ({V['info']['revised_5pct']} revised by more than 5 "
          f"percent, {V['info']['compute_added']} given an estimate, {V['info']['withdrawn']} whose estimate was "
          f"withdrawn); additions: {V['info']['added']} entries released before June 2024 but added later. "
          "$^a$\\citet{sevilla2024training}: 90 percent interval of a piecewise model with an estimated break near 2018.",
          "\\end{tablenotes}", "\\end{table}"]
    _write("app_growth.tex", L)


def app_trend(TR, S_):
    """Online Appendix table (round 3, fix list E1(b)-(c), E6): the trend in the revealed wedge under alternative
    estimators, the compute-weighted wedge by release year, and the uncapped five-year multiples moved from Table 3."""
    T = TR["T"].set_index("key")
    L = ["% Online Appendix table -- the trend in the revealed wedge and the uncapped frontier scenario.",
         "% Module rb3_econ2; sources rb3_econ2_wedge_trend.csv, _wedge_trend_lodo.csv, _wedge_trend_by_tech.csv,",
         "% _wedge_by_year.csv, _scenarios_by_trend.csv.",
         "\\begin{table}[tp]", "\\centering",
         "\\caption{The Trend in the Revealed Wedge and Frontier Data Demand}",
         "\\label{tab:app-econ-trend}", "\\footnotesize", "\\setlength{\\tabcolsep}{2pt}",
         "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrrccc@{}}", "\\toprule",
         "\\multicolumn{6}{@{}l}{\\textit{Panel A. Growth per year of the revealed wedge and the data multiple, 2023--2025}} \\\\[2pt]",
         " & & & & \\multicolumn{2}{c}{$D/D^*$ per year} \\\\", " & & & & at $\\sigma^*=0.70$ & at $\\sigma^*=$ \\\\",
         "\\cmidrule(l){5-6}",
         "Estimator & $n$ & $w$ per year & 95\\% CI & (from $M/M^*$) & 0.60; 0.74 \\\\", "\\midrule"]
    # primary, within developers, above 1e24 FLOP, compute-weighted, flagship runs; then models, aggregates, medians
    # and version 3's 56 family-label units
    rows = [("units_ols", "Decision units, OLS (primary)"), ("units_fe", "\\quad developer fixed effects"),
            ("units_fe_lnC", "\\quad fixed effects and log compute"),
            ("units_big", f"\\quad above $10^{{24}}$ FLOP ({int(T.loc['units_big', 'developers'])} developers)"),
            ("units_wls", "\\quad compute-weighted"),
            ("flagships", "Flagship runs above $10^{25}$ FLOP$^a$"),
            ("models_ols", "Models, OLS"), ("models_wls", "Models, compute-weighted"),
            ("agg_units", "Aggregate wedge, decision units"), ("median_units", "Median wedge, decision units"),
            ("units56_ols", "Family-label units (version 3), OLS")]
    # (aggregate and median wedges over models: rb3_econ2_wedge_trend.csv only, for page fit)
    for key, lab in rows:
        r = T.loc[key]
        ci = ci2(r.g_w_lo, r.g_w_hi) if np.isfinite(r.g_w_lo) else "--"
        L.append(f"{lab} & {int(r.n)} & {r.g_w:.2f} & {ci} & {r.g_DDstar_Mbased:.2f} & "
                 f"{r.g_DDstar_value_s60:.2f}; {r.g_DDstar_value_s74:.2f} \\\\")
    Ld = TR["L"]
    Tt = TR["Tt"]
    L.append(f"Leave one developer out (units, OLS) & -- & {Ld.g_w_units_ols.min():.2f}--{Ld.g_w_units_ols.max():.2f} & -- & "
             f"-- & -- \\\\")
    L.append(f"32 technologies (models, OLS) & {int(Tt.n.iloc[0])} & {Tt.g_w.min():.2f}--{Tt.g_w.max():.2f} & -- & "
             f"{Tt.g_DDstar_Mbased.min():.2f}--{Tt.g_DDstar_Mbased.max():.2f} & -- \\\\")
    By = TR["B"].set_index("year")
    L += ["\\midrule",
          "\\multicolumn{6}{@{}l}{\\textit{Panel B. The compute-weighted wedge by release year}} \\\\[2pt]",
          " & & & & \\multicolumn{2}{c}{$D/D^*$ at this $w$} \\\\", "\\cmidrule(l){5-6}",
          "Release year & Units & $s$ & $w$ & $\\sigma^*=0.70$ & 0.60; 0.74 \\\\", "\\midrule"]
    for y in (2023, 2024, 2025):
        r = By.loc[y]
        L.append(f"{y} & {int(r.n_units)} & {r.s_agg_units:.2f} & {r.w_agg_units:.2f} & "
                 f"{r.w_agg_units ** e_of(0.70):.2f} & {r.w_agg_units ** e_of(0.60):.2f}; {r.w_agg_units ** e_of(0.74):.2f} \\\\")
    # Panel C (fix list E1(c); R3 E(c)): the uncapped five-year multiples moved from Table 3
    Bv = S_["Bv"]
    L += ["\\midrule",
          "\\multicolumn{6}{@{}l}{\\textit{Panel C. The frontier adopts an open-weight trend without a stop}} \\\\[2pt]",
          " & & $D/D^*$ & \\multicolumn{2}{c}{$D$ over 5 years} & Years to \\\\", "\\cmidrule(lr){4-5}",
          "Trend & $\\sigma^*$ & per year & $4.2\\times$ & $5.06\\times$ & 2025 $w$ \\\\", "\\midrule"]
    seen = set()
    for _, r in Bv.iterrows():
        lab = "" if r.key in seen else TREND_SHORT[r.key] + ("$^a$" if r.key == "flagships" else "")
        seen.add(r.key)
        L.append(f"{lab} & {r.sigma:.2f} & {r.g_DDstar:.2f} & {big(r.gD5_uncapped_epoch)} & {big(r.gD5_uncapped_hat)} & "
                 f"{r.years_to_cap:.1f} \\\\")
    fe_within = int(T.loc["units_fe", "note"].split(" developers")[0])
    L += ["\\bottomrule", "\\end{tabular*}", "\\vspace{2pt}", "\\begin{tablenotes}[Notes]",
          "Panel A: growth factor $\\exp(\\hat b)$ of the slope of $\\ln w$ (reference technology) on release date; "
          f"decision units as in Section~\\ref{{sec:wedge}}; 95 percent wild cluster bootstrap-$t$ intervals by developer "
          f"(Webb weights, $B=9{{,}}999$; {int(T.loc['units_ols', 'clusters'])} developers; "
          f"{int(T.loc['units_big', 'developers'])} above $10^{{24}}$ FLOP, so that interval is indicative); developer "
          f"fixed effects: the slope is identified by the {fe_within} developers with units at more than one date; "
          f"$^a$the four disclosed dense runs above $10^{{25}}$ FLOP in 2024--2025, descriptive (OLS, no interval; "
          f"{float(T.loc['flagships', 'g_w_without_last']):.2f} without {T.loc['flagships', 'last_run']}); aggregate "
          "and median wedge: growth between the 2023 and 2025 values, with the aggregate $w=1/(1-s)$ at the "
          "compute-weighted $s$; $D/D^*$ from $M/M^*$: $\\exp(\\hat b/2\\theta)$, $\\theta=1/\\sigma^*-1$ of the reference, "
          "which needs no curvature given $M^*(C)$ and equals the conversion at the reference $\\sigma^*=0.70$; "
          "$\\sigma^*=0.60/0.74$: $w$ growth converted with $D/D^*=w^{\\sigma^*/[2(1-\\sigma^*)]}$; 32 technologies: "
          f"the ex-ante technologies of Section~\\ref{{sec:wedge}}, each in its own parameter convention, on the audited "
          "token counts (the reference gives the models row). Panel B: "
          "$s=\\sum(w^+-1)C/\\sum w^+C$ over decision units, $w^+=\\max(w,1)$, reference technology; $w=1/(1-s)$; "
          "$D/D^*$ that a model with this wedge would use at given compute; over models, $s$ is " + ", ".join(
              f"{By.loc[y, 's_agg_models']:.2f}" for y in (2023, 2024, 2025)) + ". Panel C: the scenario of "
          "Table~\\ref{tab:econ}, panel~B, without the stop at the 2025 wedge; $D$ over 5 years: frontier tokens in "
          "2031 relative to September 2026; years until the frontier's wedge reaches the 2025 compute-weighted wedge of "
          f"the decision units ($w={S_['w_target']:.2f}$).",
          "\\end{tablenotes}", "\\end{table}"]
    _write("app_trend.tex", L)


def app_wall(W):
    G = W["grid"]
    U = W["under"]
    Td = W["today"]
    L = ["% Online Appendix table -- the data wall at observed allocations. Module rb3_econ2;",
         "% sources rb3_econ2_wall_observed_grid.csv, _wall_understatement.csv, _wall_today.csv.",
         "\\begin{table}[tp]", "\\centering", "\\caption{The Data Wall at Observed Allocations}",
         "\\label{tab:app-econ-wall}", "\\footnotesize", "\\setlength{\\tabcolsep}{2pt}",
         "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccccc@{}}", "\\toprule",
         "\\multicolumn{7}{@{}l}{\\textit{Panel A. Extra lifetime cost (\\%) by wedge and compute-optimal scarcity $r=D^*(C)/U$}} \\\\[2pt]",
         "Wedge $w$ ($D/D^*$ at $\\sigma^*=0.70$) & $r=0.5$ & $r=1$ & $r=2$ & $r=4$ & $r=8$ & $r=16$ \\\\", "\\midrule"]
    rsel = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    for spec, head in (("D15", "Data decay only, $R^*_D=15.4$"), ("D3", "Data-only fit, $R^*_D=2.9$")):
        L.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\quad\\textit{{{head}}}}} \\\\")
        # round 3 (fix list E1(d); audit numbers_appF #1): D/D* at the solver's reference sigma* (0.700553), as in the
        # wall solver, not at sigma* rounded to four digits; the high wedge is the 2025 aggregate (RC.W_AGG25)
        sig_solver = float(G[G.tech == "kq"].sigma_star.iloc[0])
        for w in (1.0, 2.0, RC.W_AGG25):
            g = G[(G.tech == "kq") & (G.spec == spec) & np.isclose(G.w, w)]
            vals = []
            for r in rsel:
                x = g.iloc[(g.r - r).abs().argsort()[:1]]
                vals.append(pct(float(x.life_penalty.iloc[0])))
            L.append(f"$w={w:g}$ ({w ** e_of(sig_solver):.2f}) & " + " & ".join(vals) + " \\\\")
    L += ["\\midrule",
          "\\multicolumn{7}{@{}l}{\\textit{Panel B. The measured wedge: $\\hat w/(1+m_N)$ / $\\hat w/w$}} \\\\[2pt]",
          "Repetition model & $D/U=1.5$ & 2 & 4 & 8 & 16 & 32 \\\\", "\\midrule"]
    for spec, head in (("nore", "Hard cap (token cap)"), ("D3", "Data-only fit, $R^*_D=2.9$"),
                       ("D15", "Data decay only, $R^*_D=15.4$"), ("DN", "Full model")):
        g = U[U.spec == spec]
        vals = []
        for r in (1.5, 2, 4, 8, 16, 32):
            x = g[np.isclose(g.r_obs, r)]
            assert len(x) == 1, (spec, r)
            vals.append(f"{f(float(x.ratio_proc.iloc[0]))}/{f(float(x.ratio_proc_vs_w.iloc[0]))}")
        L.append(f"{head} & " + " & ".join(vals) + " \\\\")
    L += ["\\midrule",
          "\\multicolumn{7}{@{}l}{\\textit{Panel C. The frontier in September 2026 at observed data use}} \\\\[2pt]",
          " & \\multicolumn{2}{c}{100T stock} & \\multicolumn{4}{c}{22T stock (lower 95\\% bound)} \\\\",
          "\\cmidrule(lr){2-3}\\cmidrule(l){4-7}",
          "Technology & $D^*/U$ & $D/U$ & $D^*/U$ & $D/U$ & \\multicolumn{2}{c}{Extra cost (\\%)$^b$} \\\\",
          "\\midrule"]
    labels = {"chin_q": "Chinchilla, $\\kappa$ free (reference)", "chin": "Chinchilla, $\\kappa=1$", "meta_a2": "Llama 3 (Meta's law)",
              "deepseek": "DeepSeek LLM", "farseer": "Farseer, $\\kappa=1$", "farseer_eq3": "Farseer, own form",
              "gadre_rw": "Gadre et al., RefinedWeb", "olmo": "OLMo ladder", "marin_dclm": "Marin, DCLM"}
    for k, lab in labels.items():
        a = Td[(Td.key == k) & (Td.stock == "100T")].iloc[0]
        b = Td[(Td.key == k) & (Td.stock == "22T")].iloc[0]
        L.append(f"{lab} & {a.r_opt:.2f} & {a.r_obs:.2f} & {b.r_opt:.2f} & {b.r_obs:.2f} & "
                 f"\\multicolumn{{2}}{{c}}{{{pct(b.life_penalty_D15)} / {pct(b.life_penalty_DN)}}} \\\\")
    Co = W["co"]
    L += ["\\midrule",
          "\\multicolumn{7}{@{}l}{\\textit{Panel D. Compute-optimal allocation: hard wall, shadow value, substitution}} \\\\[2pt]",
          " & & & \\multicolumn{2}{c}{Shadow value} & $\\sigma_{CU}$ & $\\gamma_{\\rm eff}/\\gamma$ \\\\",
          "\\cmidrule(lr){4-5}",
          "Technology & $\\sigma^*$ & $r_{\\max}$ & $r=4$ & $r=16$ & $r=4$ & $r=16$ \\\\", "\\midrule"]
    for _, r in Co[Co.tech != "k1"].iterrows():
        L.append(f"{r.label} & {r.sigma_star:.2f} & {r.r_max:.0f} & {r.shadow_rel_r4:.2f} & "
                 f"{f(r.get('shadow_rel_r16', np.nan))} & {r.sigma_CU_r4:.2f} & {f(r.get('gamma_ratio_r16', np.nan))} \\\\")
    L += ["\\bottomrule", "\\end{tabular*}", "\\vspace{2pt}", "\\begin{tablenotes}[Notes]",
          "Reference technology ($\\kappa$ free, $\\sigma^*=0.70$) with the repetition model of "
          "\\citet{muennighoff2023scaling}. A developer with wedge $w$ minimizes $6ND+\\Xi N$, where $\\Xi N$ is the "
          "value of compactness, at the loss of its uncapped choice, which has $\\varepsilon_N/\\varepsilon_D=w$ and "
          "processes $D=D^*(C)w^{\\sigma^*/[2(1-\\sigma^*)]}$ tokens. Panel A: extra lifetime cost when the unique stock "
          "is $U=D^*(C)/r$, $C$ the model's training compute; at $w=1$ it is the extra compute of Table~\\ref{tab:econ}. "
          "The objects depend on $(w,r)$ only (checked at $10^{25}$ and $10^{27}$ FLOP). Panel B: the wedge computed from "
          "the capped choice, treating tokens processed as fresh, relative to $1+m_N=1+\\Xi/(6D)$ at that choice and to "
          f"the no-wall wedge $w={RC.W_AGG25:.2f}$ ($D/U$: no-wall tokens over the stock); $1+m_N>w$ because the capped "
          "developer processes fewer tokens; $\\hat w/w$ is the shortfall of the measured trend from its no-wall path. "
          "Token-cap case of Proposition~\\ref{prop:wedge}: $\\hat w=(1+m_N)/(1+\\mu)$; under soft repetition $\\mu=1/\\eta-1$, "
          "$\\eta=d\\ln D'/d\\ln D$ (checked). In the full model excess parameters also decay, a size penalty that raises "
          "the measured wedge once $N$ exceeds the compute-optimal size for $U$; for over-trained models it equals data "
          "decay only. Panel C: frontier compute of September 2026, stock grown 5 percent a year from 2024; $D/U$ at "
          "the observed data multiple of the disclosed dense frontier runs; $^b$extra lifetime cost at 22T, data decay "
          "only / full model (at 100T it is below 0.1 percent for every technology). Panel D: $r_{\\max}$, hard wall; "
          "shadow value in units of $6N$; $\\sigma_{CU}$, elasticity of substitution between compute and unique "
          "tokens; $\\gamma_{\\rm eff}/\\gamma$, relative return to compute behind the wall.",
          "\\end{tablenotes}", "\\end{table}"]
    _write("app_wall.tex", L)


def app_gamma(G, F, SY):
    C = G["Cand"]
    Pr = G["Prof"]
    L = ["% Online Appendix table -- the frontier elasticity gamma for growth calibrations; fleet shares by service life;",
         "% synthetic-token prices. Module rb3_econ2; sources rb3_econ2_gamma*.csv, _fleet.csv, _synthetic*.csv.",
         "\\begin{table}[tp]", "\\centering", "\\caption{Growth Calibration, Fleet Shares and Synthetic Tokens}",
         "\\label{tab:app-econ-gamma}", "\\footnotesize", "\\setlength{\\tabcolsep}{1.5pt}",
         "\\renewcommand{\\arraystretch}{0.96}",
         "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llcccccc@{}}", "\\toprule",
         "\\multicolumn{8}{@{}l}{\\textit{Panel A. The frontier elasticity $\\gamma$ and the compute needed to halve reducible loss}} \\\\[2pt]",
         " & & & & & \\multicolumn{2}{c}{Years at} & Total \\\\", "\\cmidrule(lr){6-7}",
         "Technology & $E$ from & $\\gamma$ & (s.e.) & $2^{1/\\gamma}$ [95\\% CI] & $4.2\\times$ & $5.06\\times$ & elast. \\\\",
         "\\midrule"]
    nice = {"chin_q": "Chinchilla, $\\kappa$ free (reference)", "chin_k1": "Chinchilla, $\\kappa=1$",
            "chin_a1": "Chinchilla, frontier", "hoff": "Hoffmann et al. (published)",
            "llama_a1": "Llama 3, frontier (Meta)", "llama_a3": "Llama 3, runs, $\\kappa=1$",
            "farseer_k1": "Farseer, $\\kappa=1$", "farseer_q": "Farseer, $\\kappa$ free"}
    esrc = {"chin_q": "joint fit", "chin_k1": "joint fit", "chin_a1": "frontier", "hoff": "joint fit",
            "llama_a1": "frontier", "llama_a3": "joint fit", "farseer_k1": "joint fit", "farseer_q": "joint fit"}
    for _, r in C.iterrows():
        ci = f"[{r.mult_lo:.0f}, {r.mult_hi:,.0f}]" if np.isfinite(r.mult_lo) else ""
        se = f"({r.se:.3f})" if np.isfinite(r.se) else ""
        tl = r.get("total_loss_elasticity_1e+26", np.nan)
        L.append(f"{nice[r.key]} & {esrc[r.key]} & {r.gamma:.3f} & {se} & {r.mult_halving:,.0f} {ci} & "
                 f"{r.years_4p2:.1f} & {r.years_5p06:.1f} & {f(tl, 3) if np.isfinite(tl) else '--'} \\\\")
    L += ["\\midrule",
          "\\multicolumn{8}{@{}l}{\\textit{Panel B. $E$-profiles of the IsoFLOP frontier (model-free minima)}} \\\\[2pt]",
          "Design & Budgets & Decades & $\\hat E$ & 95\\% set for $E$ & $\\hat\\gamma$ & \\multicolumn{2}{c}{$\\gamma$ over the set} \\\\",
          "\\midrule"]
    for _, r in Pr.iterrows():
        L.append(f"{r.design} & {int(r.n_budgets)} & {r.decades:.1f} & {r.E_hat:.2f} & [{r.E_lo:.2f}, {r.E_hi:.2f}] & "
                 f"{r.gamma_hat:.3f} & \\multicolumn{{2}}{{c}}{{[{r.gamma_lo:.3f}, {r.gamma_hi:.3f}]}} \\\\")
    Loc = G["Loc"].set_index("window")
    pt = G["ProfT"].iloc[0]
    ll = Pr[Pr.design == "Llama 3"].iloc[0]
    llama_nb, llama_cmax = int(ll.n_budgets), int(round(np.log10(ll.C_max)))
    dr = Loc.loc['drift per decade (successive pairs)', 'gamma_local']
    L.append(f"\\multicolumn{{8}}{{@{{}}l}}{{Chinchilla at $E={G['E_q']:.2f}$: local $\\gamma$ "
             f"{Loc.loc['lower budgets', 'gamma_local']:.3f} (low budgets), "
             f"{Loc.loc['upper budgets', 'gamma_local']:.3f} (high); drift {'$-$' if dr < 0 else '+'}{abs(dr):.4f}/decade}} \\\\")
    # Panel C: fleet
    fl = F[(F.m_label == "2025, decision units") & (F.g_label == "5.06x")]
    f42 = F[(F.m_label == "2025, decision units") & (F.g_label == "4.2x") & np.isclose(F.p, 1) & np.isclose(F.rho, 1)]
    phi42 = [f42[np.isclose(f42.L, Lf)].phi.iloc[0] for Lf in RC.LIFE_YEARS]
    L += ["\\midrule",
          "\\multicolumn{8}{@{}l}{\\textit{Panel C. Fleet share of AI compute implied by the 2025 wedge, by service life}} \\\\[2pt]",
          " & & \\multicolumn{3}{c}{$\\varrho=1$} & \\multicolumn{3}{c}{$\\varrho=2$} \\\\", "\\cmidrule(lr){3-5}\\cmidrule(l){6-8}",
          "Service life & $\\phi$ & $\\Lambda=1$ & $4.4$ & $10.4$ & $\\Lambda=1$ & $4.4$ & $10.4$ \\\\", "\\midrule"]
    for Lf in RC.LIFE_YEARS:
        g = fl[np.isclose(fl.L, Lf)]
        v1 = [g[np.isclose(g.p, 1) & np.isclose(g.rho, rr)].s_fleet.iloc[0] for rr in RC.RHO]
        v2 = [g[np.isclose(g.p, 2) & np.isclose(g.rho, rr)].s_fleet.iloc[0] for rr in RC.RHO]
        L.append(f"{Lf:.0f} year{'s' if Lf > 1 else ''} & {g.phi.iloc[0]:.2f} & " + " & ".join(f"{v:.2f}" for v in v1 + v2)
                 + " \\\\")
    # Panel D: synthetic tokens
    be = SY["BE"]
    Sx = SY["S"]
    L += ["\\midrule",
          "\\multicolumn{8}{@{}l}{\\textit{Panel D. The shadow value of a unique token beside the cost of a synthetic token}} \\\\[2pt]",
          " & & \\multicolumn{3}{c}{Shadow value (\\$/M)} & \\multicolumn{3}{c}{Break-even $r$: generation} \\\\",
          "\\cmidrule(lr){3-5}\\cmidrule(l){6-8}",
          "Compute (FLOP) & $N^*$ & $r=2$ & $r=4$ & $r=8$ & own & \\$0.60 & \\$3.96 \\\\", "\\midrule"]
    for C_ in sorted(Sx.C.unique()):
        g = Sx[Sx.C == C_].set_index("r")
        b = be[be.C == C_].set_index("price")
        sg = f"{b.loc['self-generation, p = 1', 'breakeven_r']:.1f}--{b.loc['self-generation, p = 2', 'breakeven_r']:.1f}"
        a1 = b.loc["API, V4.1-Flash off-peak ($0.60/M)", "breakeven_r"]
        a2 = b.loc["API, V4-Pro peak ($3.96/M)", "breakeven_r"]
        L.append(f"{sci(C_)} & {g.N_star.iloc[0] / 1e12:.1f}T & {g.loc[2.0, 'usd_per_Mtok_amortized']:.1f} & "
                 f"{g.loc[4.0, 'usd_per_Mtok_amortized']:.1f} & {g.loc[8.0, 'usd_per_Mtok_amortized']:.0f} & {sg} & "
                 f"{f(a1, 1) if np.isfinite(a1) else '$<$1.1'} & {f(a2, 1)} \\\\")
    L += ["\\bottomrule", "\\end{tabular*}", "\\vspace{2pt}", "\\begin{tablenotes}[Notes]",
          "Panel A: $\\gamma$, compute elasticity of reducible loss on the frontier $L^*(C)=E+K(C/6)^{-\\gamma}$; joint fit: "
          "$E$ estimated from all runs of the design (off-path runs vary both inputs); frontier: $E$ from the IsoFLOP "
          "minima alone. $2^{1/\\gamma}$: compute multiple that halves reducible loss (interval from $\\gamma\\pm1.96$ s.e.); "
          "years of frontier compute growth at 4.2- and 5.06-fold a year; total-loss elasticity "
          "(total elast.) $\\gamma(L^*-E)/L^*$ at $10^{26}$ FLOP, where available. Loss units: Chinchilla, natural-log "
          "loss per token on MassiveText (the smoothed final training loss, which \\citet{hoffmann2022training} treat as "
          "an unbiased estimate of held-out loss); Llama 3, unstated units and data; Farseer, its own validation set. "
          "Farseer's $\\kappa$-free $E$ lies on a ridge along which $E$ and $\\gamma$ trade off. "
          "Panel B: nonlinear least squares of the per-budget minima on $\\ln K$ and $\\gamma$ with $E$ fixed on a grid; 95 "
          "percent set: SSR within $F(0.95;1,n-3)/(n-3)$ of its minimum; minima of the budgets bracketed on both sides "
          f"(Llama~3: {llama_nb} budgets, to $10^{{{llama_cmax}}}$ FLOP; Meta's frontier fit in panel A uses 10 minima, to "
          "$10^{22}$); Chinchilla's minima in FLOP-effective parameters (in total parameters, "
          f"$\\hat E={pt.E_hat:.2f}$ and $\\hat\\gamma={pt.gamma_hat:.3f}$ [{pt.gamma_lo:.3f}, {pt.gamma_hi:.3f}]); "
          "Comma's set reaches the grid's lower limit, $E=0$. "
          "Panel C: $s_{\\rm fleet}=\\varrho m\\phi/(\\varrho m\\phi+\\Lambda)$, "
          f"$m={fl.m.iloc[0]:.2f}$ (2025 compute-weighted wedge of the decision units, minus one), "
          "$\\phi=(1-e^{-gL})/(gL)$ at "
          "$g=\\ln5.06$ (at $g=\\ln4.2$, $\\phi$ is " + ", ".join(f"{v:.2f}" for v in phi42) + "), $\\Lambda$ the R\\&D multiple of final "
          "training runs ($\\Lambda=1$: none; 4.4 and 10.4 from \\citealp{denain2026final}), $\\varrho$ the relative price of an "
          "inference FLOP. Panel D: break-even scarcity at which generating a token costs its shadow value, with the "
          "trainee's own compute (self-generation) or at the posted API price; reference technology, compute-optimal run, data decay only; shadow value at "
          "\\$$1.0\\times10^{-18}$ per FLOP (cloud prices about 3.7 times higher). Self-generation: a generator of the "
          "trainee's size decoding at $\\varrho=1$ to 2 times the price of a training FLOP costs $\\varrho/3$ of the processing cost of "
          "a token. API: posted prices per million output tokens of DeepSeek-V4.1-Flash (off-peak) and DeepSeek-V4-Pro "
          "(peak) on 24 September 2026 \\citep{deepseek2026pricing}. A synthetic token is not a perfect substitute for a "
          "fresh token, so the break-even scarcities are lower bounds.",
          "\\end{tablenotes}", "\\end{table}"]
    _write("app_gamma.tex", L)


# ============================================================================================ figure (appendix)
def figure(S_, W, SY, TR):
    S.use()
    fig, ax = plt.subplots(2, 2, figsize=(S.WIDTH_FULL, 4.9))
    V = dict(U_q=RC.U_STOCK, U_lo=RC.U_STOCK_LO, U_hi=RC.U_STOCK_HI, U_eff=320e12)
    # (a) frontier data demand under rising wedges
    a = ax[0, 0]
    yrs = np.linspace(2020, 2032, 241)
    a.fill_between(yrs, V["U_lo"] * 1.0 ** (yrs - 2024), V["U_hi"] * 1.10 ** (yrs - 2024), color=S.GRID, lw=0, zorder=0)
    a.plot(yrs, V["U_q"] * 1.05 ** (yrs - 2024), color=S.MUTED, lw=1.0, zorder=1)
    a.plot(yrs, V["U_eff"] * 1.05 ** (yrs - 2024), color=S.MUTED, lw=1.0, ls="--", zorder=1)
    a.text(2020.3, V["U_q"] * 0.62, "unique stock (100T)", fontsize=6.0, color=S.INK2, va="top")
    a.text(2020.3, V["U_eff"] * 1.3, "effective stock (320T)", fontsize=6.0, color=S.INK2, va="bottom")
    import scenario as SC
    ref = S_["T"]["chin_q"]
    lnC = S_["paths"]["hat"][0]
    base = np.array([S_["mD0"] * np.exp(ref.lnD(lnC(t))) for t in yrs])
    a.plot(yrs, base, color=S.INK, lw=1.3, label="wedge held")
    for sig, col in zip(RC.SIGMAS, (S.AQUA, S.BLUE, S.ORANGE)):
        fn = SC.mD_functions(S_["mD0"], S_["g_w"], sig, S_["cap"])
        tr = np.array([fn["trend"](t) * np.exp(ref.lnD(lnC(t))) for t in yrs])
        cu = np.array([fn["catchup"](t) * np.exp(ref.lnD(lnC(t))) for t in yrs])
        m = yrs >= RC.T_NOW
        a.plot(yrs[m], tr[m], color=col, lw=1.3, label=f"wedge rises, $\\sigma^*={sig:.2f}$")
        a.plot(yrs[m], cu[m], color=col, lw=1.0, ls=":")
    a.plot([], [], color=S.MUTED, lw=1.0, ls=":", label=f"dotted: rise stops at $w={RC.W_AGG25:.2f}$")
    m3 = pd.read_csv(os.path.join(RC.ROOT, "output", "tables", "m3_wedge_models.csv"))
    m3 = m3[(m3.Cmp >= 1e24) & m3.core.fillna(False).astype(bool) & ~m3.moe.fillna(False).astype(bool)]
    t3 = pd.to_datetime(m3.date)
    a.scatter(t3.dt.year + t3.dt.dayofyear / 365.25, m3.D, s=6, color=S.MUTED, lw=0, zorder=2)
    a.set_yscale("log")
    a.set_xlim(2020, 2032)
    a.set_ylim(1e11, 1e17)
    a.set_xticks([2020, 2024, 2028, 2032])
    a.set_ylabel("Training tokens (frontier run)")
    a.set_title("(a) Frontier data demand with rising wedges", loc="left")
    a.legend(loc="upper left", fontsize=5.6, handlelength=1.6, labelspacing=0.25, borderaxespad=0.3)
    # (b) the wall binds earlier for over-trained models
    b = ax[0, 1]
    G = W["grid"]
    for w, col, lab in ((1.0, S.INK, "$w=1$ (compute-optimal)"), (2.0, S.BLUE, "$w=2$"),
                        (RC.W_AGG25, S.ORANGE, f"$w={RC.W_AGG25:.2f}$ (2025 aggregate)")):
        for spec, ls in (("D15", "-"), ("D3", "--")):
            g = G[(G.tech == "kq") & (G.spec == spec) & np.isclose(G.w, w)].sort_values("r")
            g = g[np.isfinite(g.life_penalty) & (g.life_penalty > 1e-4) & (g.life_penalty < 30)]
            b.plot(g.r, 100 * g.life_penalty, color=col, lw=1.3 if spec == "D15" else 1.0, ls=ls,
                   label=lab if spec == "D15" else None)
    b.plot([], [], color=S.MUTED, ls="--", lw=1.0, label="$R^*_D=2.9$ (dashed)")
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlim(0.25, 32)
    b.set_ylim(0.01, 1000)
    b.set_xticks([0.25, 1, 4, 16])
    b.set_xticklabels(["0.25", "1", "4", "16"])
    b.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    b.set_xlabel(r"Compute-optimal scarcity $r=D^*(C)/U$")
    b.set_ylabel("Extra lifetime cost (%)")
    b.set_title("(b) The wall binds earlier for over-trained models", loc="left")
    b.legend(loc="lower right", fontsize=5.6, handlelength=1.8, labelspacing=0.25)
    # (c) measured wedges understate as the wall binds
    c = ax[1, 0]
    U = W["under"]
    for spec, col, lab in (("D15", S.BLUE, "data decay only, $R^*_D=15.4$"), ("D3", S.AQUA, "data-only fit, $R^*_D=2.9$"),
                           ("nore", S.ORANGE, "hard cap (Prop. 1(iii))")):
        g = U[U.spec == spec].sort_values("r_obs")
        g = g[np.isfinite(g.ratio_proc)]
        c.plot(g.r_obs, g.ratio_proc, color=col, lw=1.3, label=lab)
        c.plot(g.r_obs, g.ratio_proc_vs_w, color=col, lw=0.9, ls="--")
    c.plot([], [], color=S.MUTED, lw=0.9, ls="--", label=r"dashed: $\hat w/w$ (vs no wall)")
    c.axhline(1.0, color=S.MUTED, lw=0.8, ls=":")
    c.set_xscale("log")
    c.set_xlim(1.0, 40)
    c.set_ylim(0, 1.05)
    c.set_xticks([1, 2, 4, 8, 16, 32])
    c.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    c.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    c.set_xlabel(r"Scarcity at the observed allocation $D/U$")
    c.set_ylabel(r"Measured $\hat w\,/\,(1+m_N)$; $\hat w/w$")
    c.set_title("(c) Measured wedges understate as the wall binds", loc="left")
    c.legend(loc="lower left", fontsize=5.6, handlelength=1.6, labelspacing=0.25, frameon=True, framealpha=0.92, facecolor="white", edgecolor="none")
    # (d) shadow value vs synthetic-token prices
    d = ax[1, 1]
    Sx = SY["S"]
    g = Sx[np.isclose(Sx.C, S_["C_now"], rtol=1e-3)].sort_values("r")
    d.plot(g.r, g.usd_per_Mtok_amortized, color=S.BLUE, lw=1.4, label="shadow value, frontier 2026 (amortized)")
    d.plot(g.r, g.usd_per_Mtok_cloud, color=S.BLUE, lw=1.0, ls="--", label="shadow value, cloud prices")
    g2 = Sx[np.isclose(Sx.C, 1e26)].sort_values("r")
    d.plot(g2.r, g2.usd_per_Mtok_amortized, color=S.AQUA, lw=1.2, label="shadow value, $10^{26}$ FLOP")
    lo, hi = RC.API_OUTPUT_USD_PER_MTOK["flash_offpeak"], RC.API_OUTPUT_USD_PER_MTOK["pro_peak"]
    d.axhspan(lo, hi, color=S.ORANGE, alpha=0.18, lw=0)
    d.text(1.12, hi * 1.12, "API price of an output token", fontsize=5.8, color=S.INK2, va="bottom")
    d.set_xscale("log")
    d.set_yscale("log")
    d.set_xlim(1.1, 16)
    d.set_ylim(0.1, 2000)
    d.set_xticks([1.5, 2, 4, 8, 16])
    d.set_xticklabels(["1.5", "2", "4", "8", "16"])
    d.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    d.set_xlabel(r"Compute-optimal scarcity $r=D^*(C)/U$")
    d.set_ylabel("Dollars per million unique tokens")
    d.set_title("(d) Shadow value of a unique token", loc="left")
    d.legend(loc="upper left", fontsize=5.6, handlelength=1.6, labelspacing=0.25)
    fig.tight_layout(w_pad=0.8, h_pad=0.9)
    S.savefig(fig, P + "figure", folder=RC.FIGS)   # review fix: honour RB3_OUTPUT_ROOT like the tables
