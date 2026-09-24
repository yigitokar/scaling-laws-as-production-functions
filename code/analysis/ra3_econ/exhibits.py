"""exhibits.py -- paper table and figure for module ra3_econ, appendix tables, and the summary dictionary."""
from __future__ import annotations

import os

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import aer_style as S  # noqa: E402
from ra3common import FIGS, MUENN, P, PROC, TABLES, VILLALOBOS  # noqa: E402

MAIN_A = ["chin", "chin_q", "chin_a2", "meta_a2", "deepseek", "farseer", "farseer_eq3", "gadre_rw", "olmo", "marin_dclm"]
LAB_A = {"chin": r"Chinchilla refit, $\kappa=1$", "chin_q": r"Chinchilla refit, $\kappa$ free",
         "chin_a2": "Chinchilla IsoFLOP minima (Approach 2)", "meta_a2": "Llama 3 (Meta's IsoFLOP law)",
         "deepseek": "DeepSeek LLM (published law)", "farseer": r"Farseer, $\kappa=1$",
         "farseer_eq3": "Farseer, own form (local)", "gadre_rw": "Gadre et al., RefinedWeb",
         "olmo": "OLMo ladder (AI2)", "marin_dclm": "Marin, DCLM (Approach 2)"}
LAB_B = {"k1": r"Chinchilla refit, $\kappa=1$", "kq": r"Chinchilla refit, $\kappa$ free",
         "e70": r"Equivalent member, $\sigma^*=0.70$", "e60": r"Equivalent member, $\sigma^*=0.60$"}


def _f(x, d=2, pct=False, inf="$\\infty$"):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    if np.isinf(x):
        return inf
    if pct:
        return f"{100 * x:.{d}f}"
    return f"{x:.{d}f}"


def _sci(x, d=1):
    m, e = f"{x:.{d}e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def _tex(x):
    """Escape LaTeX specials in free text (review fix: 'R&D' broke the disclosures tabular)."""
    return str(x).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("$", r"\$")


def _yr(x):
    return "--" if x is None or not np.isfinite(x) else f"{x:.1f}"


def write_tex(name, caption, label, colspec, lines, notes):
    out = [r"\begin{table}[htbp]", r"\centering", r"\footnotesize", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
           rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"] + lines + [
        r"\bottomrule", r"\end{tabular}", r"\begin{tablenotes}\footnotesize", rf"\item \textit{{Notes:}} {notes}",
        r"\end{tablenotes}", r"\end{table}"]
    with open(os.path.join(TABLES, P + name + ".tex"), "w") as f:
        f.write("\n".join(out) + "\n")


def _wall_val(grid, tech, spec, r, col):
    g = grid[(grid.tech == tech) & (grid.spec == spec)]
    x = g[np.isclose(g.r, r)]
    if len(x):
        return float(x[col].iloc[0]) if col in x and pd.notna(x[col].iloc[0]) else (np.inf if col == "penalty" else np.nan)
    return np.nan


def wall_at(r2, tech, spec, r):
    """Wall objects at exactly r (recomputed; the grid is geometric)."""
    import wall as WL
    t = next(x for x in r2["fam"] if x.key == tech)
    rm = r2["rmax"][(r2["rmax"].tech == tech) & (r2["rmax"].spec == spec)].r_max.iloc[0]
    if r >= rm:
        return dict(penalty=np.inf, shadow_rel=np.nan, gamma_ratio=np.nan, Ceq_share=np.nan)
    return WL.wall_point(t, 1e26, r, spec)


def main_table(r1, r2, r3, r4):
    dem = r1["dem"].set_index("key")
    tr = r1["trend"]
    rows = []
    rows += [r"\multicolumn{8}{l}{\textit{Panel A. Data demand and the path slope $a$}} \\",
             r" & & \multicolumn{2}{c}{Growth of $D^*$} & $D^*$ at 2026 & \multicolumn{2}{c}{Year 100T unique tokens reached} & Year 320T \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){6-7}",
             r"Technology & $a$ & per year & 5 years & frontier (T) & compute-optimal & observed $M$ & (observed $M$) \\",
             r"\midrule"]
    for k in MAIN_A:
        x = dem.loc[k]
        rows.append(f"{LAB_A[k]} & {x.a:.2f} & {x.gD_hat:.2f} & {x.gD5_hat:.0f} & {x.Dstar_now_T:.0f} & "
                    f"{_yr(x.year_unique_opt)} & {_yr(x.year_unique_obsOT)} & {_yr(x.year_effective_obsOT)} \\\\")
    rows += [r"\midrule",
             r"\multicolumn{8}{l}{\textit{Panel B. The data wall and the curvature $\sigma^*$ (repetition: $R^*_D=15.4$)}} \\",
             r" & & \multicolumn{2}{c}{Extra compute (\%)} & Hard wall & Shadow value & $\sigma_{CU}$ & $\gamma_{\rm eff}/\gamma$ \\",
             r"\cmidrule(lr){3-4}",
             r"Technology & $\sigma^*$ & $r=4$ & $r=16$ & $r_{\max}$ & $r=4$ & $r=4$ & $r=16$ \\",
             r"\midrule"]
    rmax = r2["rmax"]
    for k in ["k1", "kq", "e70", "e60"]:
        t = next(x for x in r2["fam"] if x.key == k)
        p4, p16 = wall_at(r2, k, "D15", 4), wall_at(r2, k, "D15", 16)
        sc = r2["sc"]
        s4 = sc[(sc.tech == k) & (sc.spec == "D15") & np.isclose(sc.r, 4)].sigma_CU.iloc[0]
        rm = rmax[(rmax.tech == k) & (rmax.spec == "D15")].r_max.iloc[0]
        rows.append(f"{LAB_B[k]} & {t.sigma_star:.2f} & {_f(p4['penalty'], 1, True)} & {_f(p16['penalty'], 0, True)} & "
                    f"{rm:.0f} & {_f(p4['shadow_rel'], 2)} & {_f(s4, 2)} & {_f(p16['gamma_ratio'], 2)} \\\\")
    # robustness rows for kappa = 1 under other repetition models
    for sp, lab in [("DN", r"$\kappa=1$, Muennighoff full model (excess $N$ decays, $R^*_N=5.3$)"),
                    ("D3", r"$\kappa=1$, Muennighoff data-only fit ($R^*_D=2.9$)")]:
        p4, p16 = wall_at(r2, "k1", sp, 4), wall_at(r2, "k1", sp, 16)
        sc = r2["sc"]
        s4 = sc[(sc.tech == "k1") & (sc.spec == sp) & np.isclose(sc.r, 4)].sigma_CU.iloc[0]
        rm = rmax[(rmax.tech == "k1") & (rmax.spec == sp)].r_max.iloc[0]
        rows.append(f"{lab} & 0.74 & {_f(p4['penalty'], 1, True)} & {_f(p16['penalty'], 0, True)} & {rm:.0f} & "
                    f"{_f(p4['shadow_rel'], 2)} & {_f(s4, 2)} & {_f(p16.get('gamma_ratio', np.nan), 2)} \\\\")
    B = r3["B"].set_index("period")
    rows += [r"\midrule",
             r"\multicolumn{8}{l}{\textit{Panel C. Planned inference share of lifetime compute, $s=(w-1)/w$ (open-weight models; module ra2)}} \\",
             r" & \multicolumn{4}{c}{Clean production-scale universe} & \multicolumn{3}{c}{Clean verified sample} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}",
             r"Release period & Models & $s$ & 95\% CI & $s$, $\kappa=1$ & Models & $s$ & Technology range \\",
             r"\midrule"]
    for per in ["2019-2022", "2023", "2024", "2025", "All 2019-2025"]:
        x = B.loc[per]
        has_c = np.isfinite(x.get("s_clean_ref", np.nan))
        c6 = f"{int(x.n_clean)}" if has_c else "--"
        c7 = f"{x.s_clean_ref:.2f}" if has_c else "--"
        c8 = f"[{x.tech_min:.2f}, {x.tech_max:.2f}]" if has_c else "--"
        rows.append(f"{per.replace('-', '--')} & {int(x.n)} & {x.s_ref:.2f} & [{x.s_ref_lo:.2f}, {x.s_ref_hi:.2f}] & "
                    f"{x.s_k1:.2f} & {c6} & {c7} & {c8} \\\\")
    pm = r2["pmed"]
    t1 = r2["fam"][0]
    sh_usd = float(6 * wall_at(r2, "k1", "D15", 4)["N_U"] * pm["amortized"] * 1e6)
    gc = r4["gc"].set_index("technology")
    g0 = gc.loc["Chinchilla refit, kappa = 1"]
    notes = (
        r"Panel A: $a$ is the path slope ($N^*\propto C^{a}$, $D^*\propto C^{1-a}$); for Farseer's own (non-homothetic) form it is the local slope "
        rf"at $10^{{22}}$--$10^{{23}}$ FLOP. Growth of $D^*$ uses frontier compute growth of {tr.growth:.2f}$\times$ per year "
        rf"(running top-10 training runs at release, 2018--2026, Epoch AI; wild-cluster-bootstrap 95\% CI [{tr.growth_lo:.2f}, {tr.growth_hi:.2f}]). "
        rf"$D^*$ at 2026 is compute-optimal data at the fitted frontier compute in September 2026 ({_sci(tr.C_trend_now)} FLOP), in each "
        r"technology's own token units. Years: first date at which frontier runs reach the quality-adjusted stock of public text "
        r"(100T tokens in 2024, 95\% CI [22T, 490T]) or the repetition-adjusted stock (320T [65T, 1,700T]) of Villalobos et al. (2024), "
        r"growing 5\% per year; `observed $M$' scales each technology's $D^*$ by the median ratio $D/D^*(C)$ of the four disclosed dense "
        r"(non-mixture-of-experts) runs above $10^{25}$ FLOP (2024--2026). "
        r"Panel B: technologies combine the Chinchilla family with the repetition model of Muennighoff et al. (2023), "
        r"$D'=U+UR^*_D(1-e^{-R/R^*_D})$, with their jointly fitted $R^*_D=15.4$ but without their decay of excess parameters "
        r"(the most favourable case; the last two rows use their full model and their data-only fit); `equivalent members' share the $\kappa=1$ refit's expansion path and frontier and differ only in curvature. "
        r"$r=D^*(C)/U$ is the ratio of compute-optimal to unique tokens. Extra compute: additional FLOP needed to reach the unconstrained "
        r"frontier loss. $r_{\max}$: scarcity beyond which that loss is unattainable at any compute. Shadow value: $-\partial C/\partial U$ at "
        rf"fixed loss, in units of the processing cost of one token ($6N$); at $C=10^{{26}}$ one unit is about \${sh_usd:.0f} per million "
        rf"tokens at \${_sci(pm['amortized'])} per FLOP (median amortized cost of 2024--2025 frontier runs, Epoch AI). $\sigma_{{CU}}$: elasticity "
        r"of substitution between compute and unique tokens along the isoquant. $\gamma_{\rm eff}/\gamma$: elasticity of reducible loss with respect to "
        r"compute behind the wall relative to the frontier elasticity. All Panel B objects depend on $r$ only (checked at $10^{25}$ and $10^{27}$ FLOP). "
        r"Panel C: module ra2's inversion. $s=\sum_i(w_i^+-1)C_i/\sum_i w_i^+C_i$ with $w^+=\max(w,1)$ (compute-weighted; planned inference "
        r"truncated at 0), under ra2's reference technology (Chinchilla refit, $\kappa$ free, $\sigma^*=0.70$). Universe: production-scale "
        r"($6ND\geq10^{21}$) open-weight models with confident $N$ and $D$, excluding mixture-of-experts, non-transformer and ra2-excluded models. "
        r"95\% CI: design-conditional wild bootstrap of the reference technology ($B=399$). $\kappa=1$: the Chinchilla refit with $\kappa=1$. "
        r"Clean verified sample: ra2's 77-model inference-demand sample (2023--2025); technology range: minimum and maximum of the aggregate "
        r"recomputed under each of ra2's 32 ex-ante technologies. "
        rf"Growth calibration: halving reducible loss requires $2^{{1/\gamma}}={g0.compute_multiple_per_halving:.0f}\times$ compute at "
        rf"$\gamma={g0.gamma:.3f}$ ({g0.years_per_halving_raw:.1f} years of frontier compute growth)."
    )
    write_tex("table", "Economic Implications under Alternative Technologies", "tab:ra3_econ", "lccccccc", rows, notes)


def appendix_tables(r1, r2, r3, r4):
    # A1: compute growth
    tr = r1["tr"]
    lines = [r"Sample & $n$ & Developers & Growth ($\times$/yr) & 95\% CI & $p$ (4.5$\times$) & Trend, Sep. 2026 & Largest run \\",
             r"\midrule"]
    for x in tr.itertuples():
        lab = x.label.replace(" '", " `")
        lines.append(f"{lab} & {x.n} & {x.clusters} & {x.growth:.2f} & [{x.growth_lo:.2f}, {x.growth_hi:.2f}] & "
                     f"{x.p_wcr_4p5:.3f} & {_sci(x.C_trend_now)} & {_sci(x.C_max_obs)} \\\\")
    write_tex("compute_growth", "Growth of Frontier Training Compute, 2018--2026", "tab:ra3_compute_growth", "lccccccc", lines,
              r"OLS of $\log_{10}C$ on release date, 2018-01 to 2026-09 (Epoch AI snapshot of 23 September 2026). Standard errors "
              r"clustered by developer (first listed organization). 95\% intervals: wild cluster bootstrap-$t$ (unrestricted, Webb "
              r"weights, $B=9{,}999$); $p$-values: restricted wild cluster bootstrap for $H_0$: growth $=4.5\times$ per year (floor $10^{-4}$). "
              r"Record-setting runs have only six developer clusters; treat that interval as indicative. Epoch AI's published estimates "
              r"(Sevilla and Rold\'an 2024): 4.1$\times$ (notable models), 4.2$\times$ [90\%: 3.6, 4.9] (frontier since 2018), 5.3$\times$ (frontier since 2010).")
    # A2: inference share vs disclosures
    dis = r3["dis"]
    lines = [r"Source & Firm & Period & Measure & Inference share \\", r"\midrule"]
    for x in dis.itertuples():
        lines.append(f"{_tex(x.source)} & {_tex(x.firm)} & {_tex(x.period).replace('-', '--')} & {_tex(x.metric)} & "
                     f"{x.inference_share:.2f} \\\\")
    fl = r3["fl"]
    sel = fl[(fl.L == 1.0) & (fl.p == 1.0) & (fl.g_label == "frontier (estimated)")]
    for per in sel.m_label.unique():
        g = sel[sel.m_label == per]
        lines.append(f"This paper (module ra2), implied fleet share & open-weight & {per.replace('-', '--')} & planned $s$ mapped to a fleet flow "
                     rf"($L=1$, $p=1$, $\rho\in[1,{g.rho.max():.1f}]$) & [{g.s_fleet.min():.2f}, {g.s_fleet.max():.2f}] \\")
    write_tex("inference_disclosures", "Planned Inference Shares and External Disclosures", "tab:ra3_disclosures", "p{3.2cm}p{2.0cm}p{1.6cm}p{5.0cm}c",
              lines, r"Disclosures are fleet flows (energy, capacity or hardware purchases in a period); the paper's $s$ is the planned "
              r"lifetime share of a model's own final training run. The mapping $s_{\rm fleet}=pm\phi/(pm\phi+\rho)$ uses the planned multiple "
              r"$m=w-1$, the vintage factor $\phi=(1-e^{-gL})/(gL)$ at the estimated compute growth $g$ and serving life $L$, the R\&D multiple "
              r"$\rho$ (training compute of all runs relative to final runs; Epoch AI reports final runs at 9.6--22.6\% of R\&D compute) and "
              r"the relative cost $p$ of an inference FLOP. Planned $m$: module ra2's compute-weighted aggregate on its clean open-weight universe "
              r"(reference technology). Slower vintage growth (2--3$\times$ per year) raises the implied fleet shares (inference_share_flow.csv). "
              r"The OpenAI rows are Epoch AI estimates from press reports, not firm disclosures.")


def figure(r1, r2):
    S.use()
    fig, ax = plt.subplots(1, 3, figsize=(S.WIDTH_FULL, 2.45))
    # (a) data demand paths anchored at observed frontier data use
    pa = r1["paths"]
    a = ax[0]
    V = VILLALOBOS
    yrs = np.linspace(2020, 2034, 141)
    stock_mid = V["U_q"] * (1 + V["g_mid"]) ** (yrs - 2024)
    a.fill_between(yrs, V["U_q_lo"] * (1 + V["g_lo"]) ** (yrs - 2024), V["U_q_hi"] * (1 + V["g_hi"]) ** (yrs - 2024),
                   color=S.GRID, lw=0, zorder=0)
    a.plot(yrs, stock_mid, color=S.MUTED, lw=1.0, ls="-", zorder=1)
    a.plot(yrs, V["U_eff"] * (1 + V["g_mid"]) ** (yrs - 2024), color=S.MUTED, lw=1.0, ls="--", zorder=1)
    a.text(2020.3, V["U_q"] * 1.05 ** -3.7 * 0.78, "unique stock (100T)", ha="left", va="top", fontsize=6.0,
           color=S.INK2)
    a.text(2020.3, V["U_eff"] * 1.05 ** -3.7 * 1.3, "effective stock (320T)", ha="left", va="bottom", fontsize=6.0,
           color=S.INK2)
    for k, c, lab in [("farseer_eq3", S.BLUE, "0.36 Farseer"), ("meta_a2", S.AQUA, "0.46 Llama 3"),
                      ("gadre_rw", S.ORANGE, "0.57 Gadre RW")]:
        g = pa[pa.key == k]
        a.plot(g.year, g.D_obsOT, color=c, lw=1.4, label=lab)
    g = pa[pa.key == "chin"]
    a.plot(g.year, g.D_obsOT, color=S.INK2, lw=0.9, ls=":", label="0.51 Chinchilla")
    runs = r1["runs"]
    d = r1["epoch"]
    m3 = pd.read_csv(os.path.join(os.path.dirname(TABLES), "tables", "m3_wedge_models.csv"))
    m3 = m3[(m3.Cmp >= 1e24) & m3.core.fillna(False).astype(bool) & ~m3.moe.fillna(False).astype(bool)]
    m3["t"] = pd.to_datetime(m3.date).dt.year + pd.to_datetime(m3.date).dt.dayofyear / 365.25
    a.scatter(m3.t, m3.D, s=6, color=S.MUTED, zorder=2, lw=0)
    a.set_yscale("log")
    a.set_xlim(2020, 2032)
    a.set_ylim(1e11, 1e16)
    a.set_xticks([2020, 2024, 2028, 2032])
    a.set_ylabel("Training tokens")
    a.set_title("(a) Frontier data demand", loc="left")
    a.legend(loc="lower right", fontsize=5.8, handlelength=1.4, title="path slope $a$", title_fontsize=5.8,
             labelspacing=0.25, borderaxespad=0.3)
    # (b) extra compute vs scarcity
    grid = r2["grid"]
    b = ax[1]
    for k, c, lab in [("k1", S.BLUE, r"$\sigma^*=0.74$ ($\kappa=1$)"), ("kq", S.AQUA, r"$\sigma^*=0.70$ ($\kappa$ free)"),
                      ("e60", S.ORANGE, r"$\sigma^*=0.60$")]:
        g = grid[(grid.tech == k) & (grid.spec == "D15")].sort_values("r")
        g = g[np.isfinite(g.penalty) & (g.penalty < 50)]
        b.plot(g.r, 100 * g.penalty, color=c, lw=1.4, label=lab)
        rm = r2["rmax"][(r2["rmax"].tech == k) & (r2["rmax"].spec == "D15")].r_max.iloc[0]
        if rm < 64:
            b.axvline(rm, color=c, lw=0.8, ls=":")
    for sp, ls, lab in [("DN", "--", r"$\kappa=1$, excess $N$ decays"), ("D3", "-.", r"$\kappa=1$, $R^*_D=2.9$")]:
        g = grid[(grid.tech == "k1") & (grid.spec == sp)].sort_values("r")
        g = g[np.isfinite(g.penalty) & (g.penalty < 50)]
        b.plot(g.r, 100 * g.penalty, color=S.MUTED, lw=1.0, ls=ls, label=lab)
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlim(1.4, 64)
    b.set_ylim(0.1, 3000)
    b.set_xticks([2, 4, 8, 16, 32, 64])
    b.set_xticklabels(["2", "4", "8", "16", "32", "64"])
    b.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    b.set_xlabel(r"Scarcity $r = D^*(C)/U$")
    b.set_ylabel("Extra compute to reach frontier loss (%)")
    b.set_title("(b) Compute cost of a unique-token cap", loc="left")
    b.legend(loc="lower right", fontsize=6.0, handlelength=1.8)
    # (c) shadow value of a unique token
    c_ = ax[2]
    for k, c, lab in [("k1", S.BLUE, r"$\sigma^*=0.74$"), ("kq", S.AQUA, r"$\sigma^*=0.70$"), ("e60", S.ORANGE, r"$\sigma^*=0.60$")]:
        g = grid[(grid.tech == k) & (grid.spec == "D15")].sort_values("r")
        g = g[np.isfinite(g.shadow_rel) & (g.shadow_rel > 0) & (g.shadow_rel < 1e3)]
        c_.plot(g.r, g.shadow_rel, color=c, lw=1.4, label=lab)
    c_.axhline(1.0, color=S.MUTED, lw=0.8, ls="--")
    c_.text(1.5, 1.15, "= cost of processing one token", fontsize=6.3, color=S.INK2, va="bottom")
    c_.set_xscale("log")
    c_.set_yscale("log")
    c_.set_xlim(1.4, 64)
    c_.set_xticks([2, 4, 8, 16, 32, 64])
    c_.set_xticklabels(["2", "4", "8", "16", "32", "64"])
    c_.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    c_.set_ylim(1e-2, 1e3)
    c_.set_xlabel(r"Scarcity $r = D^*(C)/U$")
    c_.set_ylabel(r"Shadow value of a token ($6N$ units)")
    c_.set_title("(c) Shadow value of data", loc="left")
    c_.legend(loc="lower right", fontsize=6.3)
    fig.tight_layout(w_pad=0.8)
    S.savefig(fig, P + "figure")


def figure_share(r3):
    S.use()
    B = r3["B"]
    B = B[B.period.isin(["2019-2022", "2023", "2024", "2025"])].reset_index(drop=True)
    fig, a = plt.subplots(figsize=(S.WIDTH_HALF + 0.6, 2.4))
    x = np.arange(len(B))
    a.errorbar(x, B.s_ref, yerr=[B.s_ref - B.s_ref_lo, B.s_ref_hi - B.s_ref], fmt="o", color=S.BLUE, ms=4, capsize=2,
               lw=1.2, label=r"Universe, $\kappa$-free reference [95% CI]")
    a.plot(x - 0.14, B.s_k1, "D", color=S.AQUA, ms=3.5, lw=0, label=r"Universe, $\kappa=1$")
    ok = np.isfinite(B.tech_min)
    a.vlines(x[ok] + 0.14, B.tech_min[ok], B.tech_max[ok], color=S.MUTED, lw=2.5, alpha=0.6,
             label="Verified sample: range over 32 technologies")
    marks = [("Google 2019-21 (energy)", 0.60), ("Meta 2019-21 (capacity)", 0.70), ("Nvidia FY2024 (revenue)", 0.40)]
    for i, (lab, v) in enumerate(marks):
        a.axhline(v, color=S.ORANGE, lw=0.8, ls=["-", "--", ":"][i])
        a.text(-0.42, v - 0.012, lab, fontsize=6, color=S.INK2, ha="left", va="top")
    a.set_xticks(x)
    a.set_xticklabels([p.replace("-", "\u2013") for p in B.period])
    a.set_xlim(-0.45, len(B) - 0.55)
    a.set_ylim(0, 1)
    a.set_ylabel("Inference share of lifetime compute")
    a.set_xlabel("Release period (open-weight models)")
    a.legend(loc="upper left", fontsize=5.8, borderaxespad=0.2)
    fig.tight_layout()
    S.savefig(fig, P + "inference_share")


def summary(r1, r2, r3, r4):
    tr = r1["trend"]
    dem = r1["dem"].set_index("key")
    out = dict(growth=dict(tr.drop(labels=["label", "model_max_obs", "sample"], errors="ignore").astype(float)),
               a_range=[float(dem.loc[MAIN_A, "a"].min()), float(dem.loc[MAIN_A, "a"].max())],
               usd_per_flop=r2["pmed"],
               inference_share=r3["B"][["period", "n", "s_ref", "s_ref_lo", "s_ref_hi", "s_k1"]].to_dict("records"),
               ra2_checks_passed=bool(r3["chk"]["passed"].all()))
    return out
