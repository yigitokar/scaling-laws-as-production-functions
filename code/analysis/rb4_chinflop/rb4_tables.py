"""rb4_tables.py -- CSV outputs and the paper-ready LaTeX table (booktabs, tabular*, \\footnotesize, AEA.cls
tablenotes) for module rb4_chinflop. Every number in the .tex is generated from the stage caches.

CSVs (output/tables/rb4_chinflop_*.csv): arch (with matching counts), tableA4, tableA4_summary, totals, match_summary, coords, eta,
budgets, summary, fixedwin, fitq, ranges, param, study_level, metareg_slopes, metareg_predictions, top_budget,
headline. LaTeX: rb4_chinflop.tex.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import rb4common as cm
import rb4_estim as es


def f3(x, d=3, sign=False):
    if x is None or not np.isfinite(x):
        return "--"
    if round(x, d) == 0:
        x = 0.0
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "$-$")


def ci(lo, hi, d=3):
    return f"[{f3(lo, d)}, {f3(hi, d)}]"


def pv(p):
    return "--" if not np.isfinite(p) else ("$<$0.001" if p < 0.001 else f"{p:.3f}")


def rng(a, b, d=3, sign=False):
    return f"{f3(a, d, sign)} to {f3(b, d, sign)}" if sign else f"{f3(a, d)}--{f3(b, d)}"


# ============================================================================ assemble frames
def frames(A, E, P, G):
    out = E["out"]
    ra1b = pd.read_csv(os.path.join(cm.UP_TABLES, "ra1_modelfree_isoflop_budgets.csv"))
    ra1b = ra1b[ra1b.design == "Chinchilla"].set_index("budget_C")
    ra1s = pd.read_csv(os.path.join(cm.UP_TABLES, "ra1_modelfree_isoflop_summary.csv")).set_index("design").loc["Chinchilla"]
    # budgets
    brows = []
    for key, r in out.items():
        for row in r.get("rows", []):
            b = dict(variant=key, label=es.VLABEL.get(key.replace("|slope", ""), key) + (" [first-order correction, point estimate]" if key.endswith("|slope") else ""),
                     **{k: row.get(k, np.nan) for k in ("budget_C", "nwin", "nleft", "nright", "Nstar", "Mstar", "curv", "slope",
                                                         "S", "sigma", "se_sigma", "lo_sigma", "hi_sigma", "weight")})
            b["sigma_current"] = float(ra1b.loc[b["budget_C"], "sigma"])
            b["se_current"] = float(ra1b.loc[b["budget_C"], "se_sigma"])
            b["delta_vs_current"] = b["sigma"] - b["sigma_current"]
            brows.append(b)
    Bd = pd.DataFrame(brows)
    # design summaries
    srows = []
    eta_T4 = E["eta"][E["eta"].accounting == "T4"]
    for key, r in out.items():
        if "summary" not in r:
            continue
        s = r["summary"]
        srows.append(dict(variant=key, label=es.VLABEL[key], k_valid=s["k_valid"], sigma_re=s["sigma_re"], se_sigma_re=s["se_sigma_re"],
                          sigma_fe=s["sigma_fe"], se_sigma_fe=s["se_sigma_fe"], lo_sigma_fe=s["lo_sigma_fe"], hi_sigma_fe=s["hi_sigma_fe"],
                          tau_sigma=s["tau_sigma"], Q=s["Q"], p_Q_chi2=s["p_Q_chi2"], p_Q_boot=s["p_Q_boot"],
                          drift_per_decade=s["drift_sigma_per_decade"], se_drift=s["se_drift_sigma"], p_drift=s["p_drift"],
                          sigma_min_budget=s["sigma_min_budget"], sigma_max_budget=s["sigma_max_budget"],
                          order_F_p=r["order_p"],
                          delta_re_vs_current=s["sigma_re"] - float(ra1s["sigma_re"]),
                          eta_equivalent=cm.eta_equivalent(float(ra1s["sigma_re"]), s["sigma_re"])))
    Sm = pd.DataFrame(srows)
    memo = [dict(variant=f"memo: current with eta = {e:+.1f}", label="rb1's FLOP-accounting bracket applied to the current T estimate",
                 sigma_re=cm.shift_eta(float(ra1s["sigma_re"]), e), se_sigma_re=float(ra1s["se_sigma_re"]),
                 delta_re_vs_current=cm.shift_eta(float(ra1s["sigma_re"]), e) - float(ra1s["sigma_re"]), eta_equivalent=e)
            for e in (-0.1, 0.1)]
    Sm = pd.concat([Sm, pd.DataFrame(memo)], ignore_index=True)
    # study level
    V = pd.concat([G["old"]["V"], G["new"]["V"]] + [G[t]["V"].iloc[[0]] for t in G if t.startswith("alt_")], ignore_index=True)
    # meta-regression
    sl = pd.concat([G[t]["M"]["slopes"] for t in G], ignore_index=True)
    pr = pd.concat([G[t]["M"]["preds"] for t in G], ignore_index=True)
    top = pd.concat([G[t]["M"]["top"] for t in G], ignore_index=True)
    return Bd, Sm, V, sl, pr, top


def headline(A, E, P, G, Sm, V, sl, pr, top):
    """Compact old-vs-new register of every number quoted in the memo."""
    def srow(t, key):
        s = Sm.set_index("variant").loc[key]
        return s
    rows = []

    def add(stat, old, new, note=""):
        rows.append(dict(statistic=stat, current=old, rebuilt=new, change=(new - old) if np.isfinite(old) and np.isfinite(new) else np.nan, note=note))
    cur, new = srow(None, "T_ra1"), srow(None, "NF_T4")
    add("Chinchilla model-free sigma* (RE over budgets)", cur.sigma_re, new.sigma_re, "T -> N_F (T4)")
    add("  s.e.", cur.se_sigma_re, new.se_sigma_re)
    add("Chinchilla fixed-effect pooled sigma*", cur.sigma_fe, new.sigma_fe)
    add("Chinchilla drift per decade", cur.drift_per_decade, new.drift_per_decade)
    add("  s.e.", cur.se_drift, new.se_drift)
    add("Chinchilla Q across budgets", cur.Q, new.Q)
    for key in ("NF_A", "NF_X", "P", "T", "NF_A|corr", "NF_X|corr", "T|corr", "P|corr", "NF_T4|6ND", "NF_A|6ND"):
        add(f"Chinchilla sigma* RE, {key}", cur.sigma_re, srow(None, key).sigma_re, es.VLABEL[key])
    pm = P.set_index("convention")
    add("Chinchilla kappa-free sigma* (same runs)", pm.loc["T (ra1 reproduction)", "sigma_kappa"], pm.loc["N_F (T4)", "sigma_kappa"])
    add("Chinchilla kappa = 1 sigma* (same runs)", pm.loc["T (ra1 reproduction)", "sigma_chin"], pm.loc["N_F (T4)", "sigma_chin"])
    rf = P.attrs.get("reference_df")
    if rf is not None:
        rf = rf.set_index("convention")
        for c, lab in (("sigma_kappa", "Reference technology (240 runs) kappa-free sigma*"), ("sigma_chin", "Reference technology kappa = 1 sigma*"),
                       ("a_chin", "Reference technology a (kappa = 1)"), ("Mstar_1e+23_kappa", "Reference technology M*(1e23), kappa free"),
                       ("Mstar_1e+23_chin", "Reference technology M*(1e23), kappa = 1")):
            add(lab, rf.loc["T (m2 reference reproduction)", c], rf.loc["N_F (T4)", c], "N_F: tokens per FLOP-effective parameter")
    vo = V[V.run == "old"].reset_index(drop=True)
    vn = V[V.run == "new"].reset_index(drop=True)
    for c, lab in (("mean_re", "Study-level mean"), ("lo_hksj", "  HKSJ lower"), ("hi_hksj", "  HKSJ upper"), ("tau", "  tau"),
                   ("Q", "  Q"), ("p_Q", "  p(Q)")):
        add(lab, float(vo.loc[0, c]), float(vn.loc[0, c]))
    rb1_rows = ~vn.variant.str.startswith("[rb4]")
    both_eta = vn.variant.str.contains("for Chinchilla and Meta")
    add("Variant means, min (rb1 rows)", vo[~vo.variant.str.startswith("[rb4]")].mean_re.min(), vn[rb1_rows].mean_re.min())
    add("Variant means, max (rb1 rows)", vo[~vo.variant.str.startswith("[rb4]")].mean_re.max(), vn[rb1_rows].mean_re.max())
    add("Variant means, min (eta for Meta only; Chinchilla accountings added)", np.nan, vn[~both_eta].mean_re.min())
    add("Variant means, max (eta for Meta only; Chinchilla accountings added)", np.nan, vn[~both_eta].mean_re.max())
    for t in ("old", "new"):
        pass
    so = sl[(sl.run == "old")].set_index(["sample", "spec_key"])
    sn = sl[(sl.run == "new")].set_index(["sample", "spec_key"])
    for (smp, sk), lab in ((("all six designs (excl. Porian)", "weighted"), "Drift slope, 3-level RE (44 budgets)"),
                           (("all six designs (excl. Porian)", "unweighted"), "Drift slope, unweighted"),
                           (("all six designs (excl. Porian)", "weighted_S"), "Drift slope on S scale"),
                           (("Chinchilla + Llama 3 (the two designs reaching 1e21)", "fe_weighted"), "Drift, Chinchilla + Llama 3 FE"),
                           (("Budgets <= 3e20 (common window)", "weighted"), "Drift, budgets <= 3e20"),
                           (("IsoFLOP designs only (excl. Farseer path)", "fe_weighted"), "Drift, IsoFLOP designs FE (R3)"),
                           (("leave out each design's largest budget", "weighted"), "Drift, leave top budget out")):
        add(lab, so.loc[(smp, sk), "slope"], sn.loc[(smp, sk), "slope"])
        if sk == "weighted" and smp.startswith("all"):
            add("  CR2 s.e.", so.loc[(smp, sk), "se_cr2_design"], sn.loc[(smp, sk), "se_cr2_design"])
            add("  CR2 p", so.loc[(smp, sk), "p_cr2_design"], sn.loc[(smp, sk), "p_cr2_design"])
            add("  wild p", so.loc[(smp, sk), "p_wcr_design"], sn.loc[(smp, sk), "p_wcr_design"])
        if sk == "fe_weighted" and smp.startswith("Chinchilla"):
            add("  model s.e.", so.loc[(smp, sk), "se_model"], sn.loc[(smp, sk), "se_model"])
    po = pr[(pr.run == "old") & (pr.spec_key == "weighted")].set_index("log10C")
    pn = pr[(pr.run == "new") & (pr.spec_key == "weighted")].set_index("log10C")
    for c in (19.0, 20.0, 21.0, 22.0, 24.0):
        add(f"sigma* at 1e{int(c)} (pooled line)", po.loc[c, "est"], pn.loc[c, "est"])
    add("  1e21 CR2 lower", po.loc[21.0, "lo_cr2"], pn.loc[21.0, "lo_cr2"])
    add("  1e21 CR2 upper", po.loc[21.0, "hi_cr2"], pn.loc[21.0, "hi_cr2"])
    pio = G["old"]["M"]["pi"].set_index("C"); pin = G["new"]["M"]["pi"].set_index("C")
    col = [c for c in pio.columns if c.startswith("sigma_lower|drift of Chinchilla")][0]
    for C in (1e23, 1e24):
        add(f"PI lower bound at {C:.0e} (Chinchilla-Llama drift)", pio.loc[C, col], pin.loc[C, col])
    sco = G["old"]["M"]["scen"].reset_index(drop=True); scn = G["new"]["M"]["scen"].reset_index(drop=True)
    for i in range(len(sco)):
        if sco.loc[i, "scenario"].startswith(("Constant sigma* = 0.6", "sigma*(C_i)")) or i == 0:
            add(f"median s: {sco.loc[i, 'scenario'][:60]} -> {scn.loc[i, 'scenario'][:30]}", sco.loc[i, "median_s"], scn.loc[i, "median_s"])
    to = top[top.run == "old"].iloc[0]
    tn = top[top.run == "new"].iloc[0]
    add("sigma*_top (budgets >= 6e20)", to.mean_fixed, tn.mean_fixed)
    add("  HKSJ lower", to.lo_hksj, tn.lo_hksj)
    add("  HKSJ upper", to.hi_hksj, tn.hi_hksj)
    for t in G:
        if t.startswith("alt_"):
            v0 = G[t]["V"].iloc[0]
            s0 = G[t]["M"]["slopes"]
            w = s0[(s0["sample"] == "all six designs (excl. Porian)") & (s0.spec_key == "weighted")].iloc[0]
            add(f"Study-level mean, Chinchilla in {G[t]['key']}", float(vo.loc[0, "mean_re"]), float(v0.mean_re))
            add(f"Drift slope, Chinchilla in {G[t]['key']}", so.loc[("all six designs (excl. Porian)", "weighted"), "slope"], float(w.slope))
    return pd.DataFrame(rows)


# ============================================================================ LaTeX
def tex(A, E, P, G, Sm, V, sl, pr, top, Bd):
    S = Sm.set_index("variant")
    co = E["coords"].set_index("accounting")
    eta = E["eta"]
    a4s = A["a4s"]
    Aa = A["A"]
    used = sorted(set(E["match"][E["match"].n_profile_runs > 0].row))
    body = []
    # ---------------- Panel A
    body.append("\\multicolumn{6}{@{}l}{\\textit{Panel A. Which FLOP count set Hoffmann et al.'s budgets?}}\\\\[1pt]\n")
    body.append(" & Table A4 & $F/(6T)$, & $\\eta$, mean & \\multicolumn{2}{c}{Figure 4 coordinates} \\\\\n")
    body.append("\\cmidrule(l){5-6}\n")
    body.append("FLOP count & ratios & profile models & of budgets & slope in $\\log N$ & offset \\\\\n\\midrule\n")
    labs = {"T4": "T4: App.~F, no vocabulary", "A": "A: App.~F as printed",
            "X": "X: executed", "6T": "$6T$"}
    for k in ("T4", "A", "X", "6T"):
        if k != "6T":
            ex = a4s[(a4s.accounting == k) & (a4s.N_in_6ND == "architecture T")].iloc[0]
            a4 = f"{int(ex.exact_2dp)} of {int(ex.n)}"
            r = Aa.loc[used, f"r_{k}"]
            rr = f"{r.min():.2f}--{r.max():.2f}"
            e = eta[eta.accounting == k].eta_window
            er = f3(e.mean(), 3, True)
        else:
            a4, rr, er = "--", "1", "0"
        c = co.loc[k]
        body.append(f"{labs[k]} & {a4} & {rr} & {er} & {f3(c.slope_resid_on_log10N, 3, True)} ({f3(c.se_cr1_budget, 3)}) & "
                    f"{f3(c.mean_offset_dex, 3, True)} \\\\\n")
    blockA = "".join(body)
    # ---------------- Panel B
    body = ["\\multicolumn{6}{@{}l}{\\textit{Panel B. Chinchilla's model-free $\\sigma^*$ (137 profile runs, nine budgets)}}\\\\[1pt]\n",
            " & \\multicolumn{2}{c}{Pooled} & Drift per & $\\sigma^*_b$ at & Change vs. \\\\\n",
            "\\cmidrule(l){2-3}\n",
            "Parameter count & Random & Fixed & decade & $10^{21}$ FLOP & current \\\\\n\\midrule\n"]
    blabel = [("T_ra1", "Current: $T$, digitized"),
              ("NF_T4", "\\textbf{$N_F$, count T4}"),
              ("NF_A", "$N_F$, count A"),
              ("NF_X", "$N_F$, count X"),
              ("P", "$P=T-Vd$"),
              ("T", "$T$, architecture"),
              ("NF_A|corr", "$N_F$, A, own isocosts$^{a}$"),
              ("NF_X|corr", "$N_F$, X, own isocosts$^{a}$"),
              ("P|corr", "$P$, own isocosts$^{a}$"),
              ("T|corr", "$T$, own isocosts$^{a}$"),
              ("NF_T4|6ND", "$N_F$, T4, $6T$ tokens$^{b}$")]
    for key, lab in blabel:
        s = S.loc[key]
        bb = Bd[Bd.variant == key].set_index("budget_C")
        tops = f3(bb.loc[1e21, "sigma"], 3)
        chg = "--" if key == "T_ra1" else f3(s.delta_re_vs_current, 3, True)
        body.append(f"{lab} & {f3(s.sigma_re)} & {f3(s.sigma_fe)} & {f3(s.drift_per_decade, 3, True)} & {tops} & {chg} \\\\\n")
        if key in ("T_ra1", "NF_T4"):
            body.append(f" & ({f3(s.se_sigma_re)}) & ({f3(s.se_sigma_fe)}) & ({f3(s.se_drift)}) & & \\\\\n")
    em, ep = S.loc["memo: current with eta = -0.1"], S.loc["memo: current with eta = +0.1"]
    body.append(f"Memo: current, $\\eta=-0.1$ & {f3(em.sigma_re)} & & & & {f3(em.delta_re_vs_current, 3, True)} \\\\\n")
    body.append(f"Memo: current, $\\eta=+0.1$ & {f3(ep.sigma_re)} & & & & {f3(ep.delta_re_vs_current, 3, True)} \\\\\n")
    blockB = "".join(body)
    # ---------------- Panel C
    vo = V[V.run == "old"].reset_index(drop=True)
    vn = V[V.run == "new"].reset_index(drop=True)
    so = sl[sl.run == "old"].set_index(["sample", "spec_key"])
    sn = sl[sl.run == "new"].set_index(["sample", "spec_key"])
    po = pr[(pr.run == "old") & (pr.spec_key == "weighted")].set_index("log10C")
    pn = pr[(pr.run == "new") & (pr.spec_key == "weighted")].set_index("log10C")
    to, tn = top[top.run == "old"].iloc[0], top[top.run == "new"].iloc[0]
    body = ["\\multicolumn{3}{@{}l}{\\textit{Panel C. Propagation: study-level summary and $\\sigma^*(C)$}}\\\\[1pt]\n\\midrule\n",
            " & Current (Chinchilla in $T$) & Rebuilt (Chinchilla in $N_F$, T4) \\\\\n\\midrule\n"]
    body.append(f"Study-level mean, 95\\% CI$^{{c}}$ & {f3(vo.loc[0, 'mean_re'])} {ci(vo.loc[0, 'lo_hksj'], vo.loc[0, 'hi_hksj'])} & "
                f"{f3(vn.loc[0, 'mean_re'])} {ci(vn.loc[0, 'lo_hksj'], vn.loc[0, 'hi_hksj'])} \\\\\n")
    body.append(f"\\quad $\\tau$; $Q$ ($p$) & {f3(vo.loc[0, 'tau'])}; {vo.loc[0, 'Q']:.1f} ({pv(vo.loc[0, 'p_Q'])}) & "
                f"{f3(vn.loc[0, 'tau'])}; {vn.loc[0, 'Q']:.1f} ({pv(vn.loc[0, 'p_Q'])}) \\\\\n")
    eo = vo[vo.variant.str.contains("for Chinchilla and Meta")]
    en = vn[vn.variant.str.contains("for Meta only")]
    body.append(f"\\quad FLOP accounting, $\\eta=\\pm0.1$$^{{d}}$ & {f3(eo.mean_re.min())}--{f3(eo.mean_re.max())} {ci(eo.lo_hksj.min(), eo.hi_hksj.max())} & "
                f"{f3(en.mean_re.min())}--{f3(en.mean_re.max())} {ci(en.lo_hksj.min(), en.hi_hksj.max())} \\\\\n")
    ca = vn[vn.variant.str.startswith("[rb4] Chinchilla in accounting")]
    body.append(f"\\quad Chinchilla's other counts$^{{e}}$ & -- & {f3(ca.mean_re.min())}--{f3(ca.mean_re.max())} \\\\\n")
    rb1o = vo[~vo.variant.str.startswith("[rb4]")]
    keepn = vn[~vn.variant.str.contains("for Chinchilla and Meta")]
    body.append(f"\\quad Range of variant means$^{{f}}$ & {f3(rb1o.mean_re.min())}--{f3(rb1o.mean_re.max())} & "
                f"{f3(keepn.mean_re.min())}--{f3(keepn.mean_re.max())} \\\\\n")
    k = ("all six designs (excl. Porian)", "weighted")
    body.append(f"Slope per decade of compute$^{{g}}$ & {f3(so.loc[k, 'slope'], 3, True)} {ci(so.loc[k, 'lo_cr2_design'], so.loc[k, 'hi_cr2_design'])} & "
                f"{f3(sn.loc[k, 'slope'], 3, True)} {ci(sn.loc[k, 'lo_cr2_design'], sn.loc[k, 'hi_cr2_design'])} \\\\\n")
    body.append(f"\\quad Wild cluster $p$; CR2 $p$ & {pv(so.loc[k, 'p_wcr_design'])}; {pv(so.loc[k, 'p_cr2_design'])} & "
                f"{pv(sn.loc[k, 'p_wcr_design'])}; {pv(sn.loc[k, 'p_cr2_design'])} \\\\\n")
    k2 = ("Chinchilla + Llama 3 (the two designs reaching 1e21)", "fe_weighted")
    body.append(f"\\quad Chinchilla and Llama~3 (fixed effects) & {f3(so.loc[k2, 'slope'], 3, True)} ({f3(so.loc[k2, 'se_model'])}) & "
                f"{f3(sn.loc[k2, 'slope'], 3, True)} ({f3(sn.loc[k2, 'se_model'])}) \\\\\n")
    k3 = ("Budgets <= 3e20 (common window)", "weighted")
    body.append(f"\\quad Budgets $\\le3\\times10^{{20}}$ FLOP & {f3(so.loc[k3, 'slope'], 3, True)} & {f3(sn.loc[k3, 'slope'], 3, True)} \\\\\n")
    body.append(f"$\\sigma^*$ at $10^{{21}}$ FLOP, pooled line & {f3(po.loc[21.0, 'est'])} {ci(po.loc[21.0, 'lo_cr2'], po.loc[21.0, 'hi_cr2'])} & "
                f"{f3(pn.loc[21.0, 'est'])} {ci(pn.loc[21.0, 'lo_cr2'], pn.loc[21.0, 'hi_cr2'])} \\\\\n")
    body.append(f"$\\sigma^*$, budgets $\\ge6\\times10^{{20}}$ FLOP & {f3(to.mean_fixed)} {ci(to.lo_hksj, to.hi_hksj)} & "
                f"{f3(tn.mean_fixed)} {ci(tn.lo_hksj, tn.hi_hksj)} \\\\\n")
    blockC = "".join(body)
    pmt = P.set_index("convention")
    fwd = E["fixedwin"]
    dfix = fwd[fwd.accounting == "T4"].sigma.values - fwd[fwd.accounting == "ra1"].sigma.values
    notes = (
        "Hoffmann et al.'s (2022) 50 architectures (their Table A9) with vocabulary $V=32{,}000$ and context 2,048. "
        "$T$: total parameters, $L(4d\\,d_{kv}h+2d f)+L d\\,d_{kv}h+Vd$ with $h$ heads of size $d_{kv}$ (attention, MLP, one relative-position projection per "
        "layer, one vocabulary matrix), which reproduces every count of Table A9 to 0.8 percent; $P=T-Vd$; "
        "$N_F=C/(6D)=F/6$, with $F$ the training FLOPs per token (forward and backward). "
        "Count A is Appendix F as printed; T4 drops its embedding and final-logit terms; X drops only the embedding term. "
        "Table A4 ratios: printed ratios reproduced to two decimals. $\\eta$: slope of $\\ln(F/6T)$ on $\\ln T$ over each "
        "budget's window runs, averaged over the nine budgets (by budget in the CSV). Figure 4 coordinates: within-budget regression of $\\log_{10}$ of the digitized FLOP "
        "coordinate, net of the nominal budget and of $-\\log_{10}(F/6T)$, on $\\log_{10}N$ (budget fixed effects; "
        "standard errors clustered by budget); flat if the figure plots $6ND$ at token counts set by count $F$. "
        f"Panel B: runs on the nominal budgets unless marked; for count T4 these are exact isocosts. ra1's estimator and bootstrap, unchanged (quadratic in $\\ln N$ within one log point of the path, "
        f"log-cubic frontier, 999 draws with ra1's seed). With the current windows held fixed, count T4 changes the "
        f"budget estimates by {f3(float(dfix.min()), 3, True)} to {f3(float(dfix.max()), 3, True)}. "
        f"Same-run parametric $\\sigma^*$ in $N_F$: $\\kappa$ free {f3(pmt.loc['N_F (T4)', 'sigma_kappa'])} "
        f"({f3(pmt.loc['N_F (T4)', 'se_sigma_kappa'])}), $\\kappa=1$ {f3(pmt.loc['N_F (T4)', 'sigma_chin'])} "
        f"({f3(pmt.loc['N_F (T4)', 'se_sigma_chin'])}). "
        "$^{a}$Losses moved from the observed (T4) isocost to the count's own isocost with the $\\kappa$-free surface "
        "fitted to the same runs. "
        "$^{b}$The reading in which tokens were set by $D=C_b/(6T)$, rejected by the coordinate test (row $6T$). "
        "$^{c}$One estimate per study (Hoffmann, Meta, Marin, Farseer); DerSimonian--Laird mean, modified "
        "Hartung--Knapp--Sidik--Jonkman interval. "
        "$^{d}$$\\sigma^*_{\\rm eff}=2/[2+S/(1+\\eta)^2]$ for Chinchilla and Meta (current) and for Meta only (rebuilt); "
        "means, and the widest interval. "
        "$^{e}$Chinchilla in counts A and X and conventions $P$ and $T$, on nominal budgets or exact isocosts, and T4 "
        "under $6T$ budgets; other studies unchanged. "
        "$^{f}$All study-level variants of rb1 (FLOP-accounting rows as in the row above), and, rebuilt, the rows "
        "of note $e$. "
        "$^{g}$Three-level random-effects meta-regression of 44 budget-level estimates on $\\log_{10}C$; cluster-robust (CR2) "
        "interval by design; restricted wild cluster bootstrap-$t$ (9,999 draws, rb1's seeds).")
    T = ("% rb4_chinflop.tex -- generated by code/analysis/rb4_chinflop/rb4_tables.py; do not edit by hand.\n"
         "\\begin{table}[tp]\n\\centering\n"
         "\\caption{Chinchilla's FLOP Accounting Rebuilt: The Model-Free Elasticity by Parameter Count}\n"
         "\\label{tab:app-chinflop}\n\\footnotesize\n\\setlength{\\tabcolsep}{2pt}\n"
         "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lccccc@{}}\n\\toprule\n" + blockA +
         "\\midrule\n" + blockB + "\\bottomrule\n\\end{tabular*}\n\n\\medskip\n"
         "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcc@{}}\n" + blockC + "\\bottomrule\n\\end{tabular*}\n\n"
         "\\begin{tablenotes}\n" + notes + "\n\\end{tablenotes}\n\n"
         "\\begin{tablenotes}[Source]\n\\citet{hoffmann2022training}, Appendix F and Tables A4 and A9; Epoch AI's digitization "
         "\\citep{besiroglu2024chinchilla}; modules ra1 and rb1; authors' calculations.\n\\end{tablenotes}\n\\end{table}\n")
    path = os.path.join(cm.TABLES, f"{cm.PREFIX}.tex")
    with open(path, "w") as f:
        f.write(T)
    return path


def make_all(Ar, E, PP, G):
    P = PP["same_run"]
    P.attrs["reference_df"] = PP["reference"]
    Bd, Sm, V, sl, pr, top = frames(Ar, E, P, G)
    A = Ar["A"].merge(E["match"][["row", "n_profile_runs", "n_offprofile_runs", "max_abs_log_err", "budgets"]], on="row", how="left")
    cm.tab(A, "arch")
    cm.tab(Ar["a4"], "tableA4")
    cm.tab(Ar["a4s"], "tableA4_summary")
    cm.tab(Ar["other"], "totals")
    cm.tab(pd.DataFrame([dict(E["match_tot"], arxiv_sha256=Ar["sha"], **{f"transcription_{k}": v for k, v in Ar["transcription"].items()})]),
           "match_summary")
    cm.tab(E["coords"], "coords")
    cm.tab(E["eta"], "eta")
    cm.tab(Bd, "budgets")
    cm.tab(Sm, "summary")
    cm.tab(E["fixedwin"], "fixedwin")
    cm.tab(E["fitq"], "fitq")
    cm.tab(E["ranges"], "ranges")
    cm.tab(P, "param")
    cm.tab(PP["reference"], "reference_fit")
    cm.tab(V, "study_level")
    cm.tab(sl, "metareg_slopes")
    cm.tab(pr, "metareg_predictions")
    cm.tab(top, "top_budget")
    cm.tab(pd.concat([G[t]["M"]["pi"] for t in G if "pi" in G[t]["M"]], ignore_index=True), "pi_sigmaC")
    cm.tab(pd.concat([G[t]["M"]["scen"] for t in G if "scen" in G[t]["M"]], ignore_index=True), "wedge_scenarios")
    # [review] rb1's design-top table (largest bracketed budget, raw sigma*_b and BLUP design line), old and rebuilt
    cm.tab(pd.concat([G[t]["M"]["tops"] for t in G if "tops" in G[t]["M"]], ignore_index=True), "design_top")
    H = headline(Ar, E, P, G, Sm, V, sl, pr, top)
    cm.tab(H, "headline")
    p = tex(Ar, E, P, G, Sm, V, sl, pr, top, Bd)
    cm.log(f"  wrote {len(H)} headline rows, 22 CSVs and {p}")   # [review] was "19 CSVs" (22 with design_top)
