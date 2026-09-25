"""tables_rb1.py -- CSV outputs and paper-ready LaTeX tables (booktabs, tabular*, \\footnotesize, AEA.cls tablenotes)
for module rb1_sigmaC.

Tables:
  rb1_sigmaC_table.tex    sigma* by compute level, the meta-regression slope, the top-budget value and the study-level
                          interval with the accounting/convention/bandwidth range (main text: Table 1 addition)
  rb1_sigmaC_wedge.tex    sigma*(C) beyond the designs (partial identification) and the revealed-demand magnitudes
  rb1_sigmaC_extrap.tex   local vs parametric wedge by M bin on Farseer, Marin (x3) and Llama 3; convexity
  rb1_sigmaC_tuning.tex   tuning share of Farseer's convexity under measured Step Law inefficiency gradients
  rb1_sigmaC_budgets.tex  (review addition) budget-level estimates with window diagnostics, FE and RE pooled values
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import rb1common as cm

MB = ["M<16", "16-64", "64-256", "256-1,024", ">=1,024"]
MB_TEX = ["$<$16", "16--64", "64--256", "256--1,024", "$\\ge$1,024"]


def f3(x, d=3, sign=False):
    if x is None or not np.isfinite(x):
        return "--"
    if round(x, d) == 0:
        x = 0.0
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def ci(lo, hi, d=3):
    return f"[{f3(lo, d)}, {f3(hi, d)}]"


def sci(C):
    e = int(np.floor(np.log10(C) + 1e-9))
    m = C / 10 ** e
    # [round 3, WP2, T13(g)] mantissa with up to three significant figures (1.8e19 was printed as 2e19)
    return f"$10^{{{e}}}$" if abs(m - 1) < 1e-6 else f"${float(f'{m:.3g}'):g}\\times10^{{{e}}}$"


def pv(p):
    if not np.isfinite(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


# ============================================================================ CSVs
def write_csvs(M, P, X, T):
    cm.tab(M["data"], "budgets_input")
    cm.tab(M["slopes"], "metareg_slopes")
    cm.tab(M["preds"], "metareg_predictions")
    cm.tab(M["het"], "metareg_design_slopes")
    cm.tab(M["tops"], "design_top_budgets")
    cm.tab(M["top"], "top_budget_sigma")
    cm.tab(M["study"], "study_level")
    cm.tab(M["conv"], "conventions")
    cm.tab(P["pi"], "pi_sigmaC")
    cm.tab(P["scen"], "wedge_scenarios")
    cm.tab(P["models"], "wedge_models")
    cm.tab(P["bounds"], "magnitude_bounds")
    ex_T, ex_L, ex_S, ex_P = [], [], [], []
    for k, v in X.items():
        if k in ("cv", "bw"):
            continue
        ex_T.append(v["T"]); ex_L.append(v["L"]); ex_S.append(v["S"]); ex_P.append(v["P"])
    ET = pd.concat(ex_T, ignore_index=True)
    cm.tab(ET, "extrap_delta")
    cm.tab(pd.concat(ex_L, ignore_index=True), "extrap_convexity")
    cm.tab(pd.concat(ex_S, ignore_index=True), "extrap_sensitivity")
    cm.tab(pd.concat(ex_P, ignore_index=True), "extrap_sigma")
    cm.tab(X["cv"], "extrap_bandwidth_cv")
    bw = pd.DataFrame([dict(design=k, hx=v["primary"][0], hc=v["primary"][1], mx=v["primary"][2], mc=v["primary"][3],
                            own_mx=v["own"][2], own_mc=v["own"][3], hc_min=v["hcmin"],
                            # review addition: design range of M against the range where the local wedge is evaluated
                            M_max_runs=float((X[k]["grid"]["M"]).max()),
                            M_max_evaluated=float(X[k]["grid"]["M"].values[(X[k]["st0"]["neff"] >= 8) &
                                                                           np.isfinite(X[k]["st0"]["lnw_local"])].max()))
                       for k, v in X["bw"].items()])
    cm.tab(bw, "extrap_bandwidths")
    fl = pd.concat([v["flagged"].assign(design=k) for k, v in X.items() if k not in ("cv", "bw") and len(v["flagged"])],
                   ignore_index=True) if any(len(v["flagged"]) for k, v in X.items() if k not in ("cv", "bw")) else pd.DataFrame()
    cm.tab(fl, "extrap_screened_runs")
    pooled = pooled_marin(X)
    cm.tab(pooled, "extrap_marin_pooled")
    grids = []
    for k, v in X.items():
        if k in ("cv", "bw"):
            continue
        g = v["grid"].copy()
        for kk in ("lnw_local", "neff", "lnw_chin_full", "lnw_chin_M100", "lnw_kappa_full", "lnw_kappa_M100"):
            g[kk] = v["st0"][kk]
        grids.append(g.assign(design=k))
    cm.tab(pd.concat(grids, ignore_index=True), "extrap_grid")
    cm.tab(T["table"], "tuning")
    cm.tab(T["breakdown"], "tuning_breakdown")
    cm.tab(T["d6"], "tuning_d6")
    return ET, pooled


def pooled_marin(X):
    """Mean of Delta over Marin's three corpora (equal weights), with a basic interval from averaging the
    corpora's independent cluster-bootstrap draws index by index."""
    rows = []
    names = [k for k in X if str(k).startswith("Marin")]
    for spec in ("kappa_full", "chin_full", "chin_M100", "kappa_M100"):
        for b in MB + ["M>=256"]:
            key = f"delta|{spec}|{b}"
            ests = [X[n]["sm0"].get(key, np.nan) for n in names]
            if not np.all(np.isfinite(ests)):
                rows.append(dict(spec=spec, M_bin=b, delta=np.nanmean(ests) if np.any(np.isfinite(ests)) else np.nan,
                                 lo=np.nan, hi=np.nan, n_corpora=int(np.sum(np.isfinite(ests)))))
                continue
            dev = []
            for n in names:
                D = pd.DataFrame(X[n]["draws"])
                dev.append(D[key].values - X[n]["sm_pop"].get(key, np.nan))
            L = min(len(v) for v in dev)
            avg = np.nanmean(np.vstack([v[:L] for v in dev]), axis=0)
            avg = avg[np.isfinite(avg)]
            e = float(np.mean(ests))
            rows.append(dict(spec=spec, M_bin=b, delta=e, delta_bc=e - float(np.mean(avg)), se=float(np.std(avg)),
                             lo=e - np.percentile(avg, 97.5), hi=e - np.percentile(avg, 2.5),
                             ratio=np.exp(e), ratio_lo=np.exp(e - np.percentile(avg, 97.5)), ratio_hi=np.exp(e - np.percentile(avg, 2.5)),
                             n_corpora=3))
    return pd.DataFrame(rows)


# ============================================================================ Table: sigma* by compute + study level
COLSPEC = r"@{\extracolsep{\fill}}>{\raggedright\arraybackslash}p{%s\textwidth}%s@{}"


def _wrap(caption, label, colspec, body, notes, source, tabcolsep="2pt"):
    return ("\\begin{table}[tp]\n\\centering\n\\caption{" + caption + "}\n\\label{" + label + "}\n\\footnotesize\n"
            "\\setlength{\\tabcolsep}{" + tabcolsep + "}\n\\begin{tabular*}{\\textwidth}{" + colspec + "}\n\\toprule\n" +
            "".join(body) + "\\bottomrule\n\\end{tabular*}\n\n\\begin{tablenotes}\n" + notes + "\n\\end{tablenotes}\n\n"
            "\\begin{tablenotes}[Source]\n" + source + "\n\\end{tablenotes}\n\\end{table}\n")


def sci_(C):
    return sci(C)


def table_sigma(M):
    S = M["slopes"]
    Pr = M["preds"]
    tops = M["tops"].set_index("design")
    top = M["top"].iloc[0]
    V = M["study"]
    body = []

    def srow(sample, key, lab):
        r = S[(S["sample"] == sample) & (S.spec_key == key)].iloc[0]
        many = int(r.n_designs) >= 4
        cr2 = (f"{f3(r.se_cr2_design)} [{r.df_cr2_design:.1f}]"
               if many and np.isfinite(r.get("se_cr2_design", np.nan)) and r.df_cr2_design >= 2 else "--")
        p = pv(r.p_wcr_design) if many and np.isfinite(r.get("p_wcr_design", np.nan)) else "--"
        sem = "--" if key == "balanced" else f3(r.se_model)
        return f"{lab} & {int(r.k)} & {f3(r.slope, sign=True)} & {sem} & {cr2} & {p} \\\\\n"

    body.append("\\multicolumn{6}{@{}l}{\\textit{Panel A. Drift: regression of $\\sigma^*_b$ on $\\log_{10}C$}} \\\\[1pt]\n")
    body.append(" & Budgets & Slope & Model s.e. & CR2 s.e.\\ [df] & Wild $p$ \\\\\n\\midrule\n")
    allx = "all six designs (excl. Porian)"
    body.append(srow(allx, "weighted", "Six designs, 3-level RE"))
    body.append(srow(allx, "unweighted", "Six designs, RE, unweighted"))
    body.append(srow(allx, "balanced", "Six designs, balanced weights"))
    body.append(srow("IsoFLOP designs only (excl. Farseer path)", "fe_weighted", "IsoFLOP designs, FE"))
    body.append(srow("Chinchilla + Llama 3 (the two designs reaching 1e21)", "fe_weighted", "Chinchilla and Llama 3, FE"))
    body.append(srow("Marin only (three corpora)", "fe_weighted", "Marin, FE"))
    body.append(srow("Budgets <= 3e20 (common window)", "weighted", "Budgets $\\le3\\times10^{20}$, 3-level RE"))
    rp = S[(S["sample"].str.startswith("Porian")) & (S.spec_key == "fe_weighted")].iloc[0]
    body.append(f"Porian et al.\\ (not pooled), FE & {int(rp.k)} & {f3(rp.slope, sign=True)} & {f3(rp.se_model)} & -- & -- \\\\\n")
    body.append("\\addlinespace\n\\multicolumn{6}{@{}l}{\\textit{Panel B. $\\sigma^*$ by compute level (pooled line; CR2 95\\% interval)}} \\\\[1pt]\n")
    body.append(" & & $10^{19}$ & $10^{20}$ & $10^{21}$ & $10^{22\\,a}$ \\\\\n\\midrule\n")
    for key, lab in (("weighted", "3-level RE"), ("unweighted", "RE, unweighted")):
        p = Pr[Pr.spec_key == key].set_index("log10C")
        body.append(f"{lab} & & " + " & ".join(f3(p.loc[c, "est"]) for c in (19.0, 20.0, 21.0, 22.0)) + " \\\\\n")
        body.append(" & & " + " & ".join(ci(p.loc[c, "lo_cr2"], p.loc[c, "hi_cr2"]) for c in (19.0, 20.0, 21.0)) + " & \\\\\n")
    body.append("\\addlinespace\n\\multicolumn{6}{@{}l}{\\textit{Panel C. $\\sigma^*$ at each design's largest bracketed budget}} \\\\[1pt]\n")
    body.append(" & Budget & $\\sigma^*_b$ & (s.e.) & Design line & \\\\\n\\midrule\n")
    short = {"Chinchilla": "Chinchilla", "Llama 3": "Llama 3", "Marin, Comma": "Marin, Comma", "Marin, DCLM": "Marin, DCLM",
             "Marin, Nemotron-CC": "Marin, Nemotron-CC", "Farseer (local path)": "Farseer, local path"}
    for des, lab in short.items():
        t = tops.loc[des]
        body.append(f"{lab} & {sci(t.top_budget)} & {f3(t.sigma_raw_top)} & ({f3(t.se_raw_top)}) & {f3(t.fitted_top_blup)} & \\\\\n")
    body.append(f"Chinchilla and Llama 3, budgets $\\ge6\\times10^{{20}}$ & {int(top.k)} & {f3(top.mean_fixed)} & "
                f"{ci(top.lo_hksj, top.hi_hksj)} & & \\\\\n")
    body.append("\\addlinespace\n\\multicolumn{6}{@{}l}{\\textit{Panel D. One estimate per study (Marin one study): random-effects mean}} \\\\[1pt]\n")
    body.append(" & $k$ & Mean & 95\\% CI (HKSJ) & $\\tau$ & $Q$ ($p$) \\\\\n\\midrule\n")
    lab_map = [("PRIMARY", "Headline$^{b}$"), ("Marin corpora independent", "Marin's corpora independent"),
               ("IsoFLOP studies only", "IsoFLOP studies only"), ("Six designs", "Six designs (round 2)"),
               ("Marin in configuration N", "Marin in configuration $N$"),
               ("FLOP-accounting elasticity eta = -0.1", "$\\eta=-0.1$ (Meta)" if cm.CHIN_REBUILT else "$\\eta=-0.1$ (Chinchilla, Meta)"),
               ("FLOP-accounting elasticity eta = +0.1", "$\\eta=+0.1$ (Meta)" if cm.CHIN_REBUILT else "$\\eta=+0.1$ (Chinchilla, Meta)"),
               ("Farseer Hessian-based path at the extended-grid CV", "Farseer, Hessian path, CV $h$"),
               ("Farseer incl. embeddings", "Farseer, total $N$"),
               ("Lowest-convention set", "Marin config.\\ $N$, Farseer total $N$"),
               ("Design-level fixed-effect means", "Fixed-effect means (Chin., Llama)$^{c}$"),
               ("Local first-derivative estimator", "Local first-derivative (Marin, Llama)$^{d}$")]
    for pre, lab in lab_map:
        if not V.variant.str.startswith(pre).any():
            continue
        r = V[V.variant.str.startswith(pre)].iloc[0]
        body.append(f"{lab} & {int(r.k)} & {f3(r.mean_re)} & {ci(r.lo_hksj, r.hi_hksj)} & {f3(r.tau)} & {r.Q:.1f} ({pv(r.p_Q)}) \\\\\n")
    if cm.CHIN_REBUILT:   # [rb4 integration] ranges over Chinchilla's alternative counts/conventions and windows
        for kind, lab in (("accounting", "Chinchilla's other counts, conventions$^{e}$"),
                          ("windows", "Chinchilla's windows, membership$^{e}$")):
            g = V[V.variant.str.startswith(f"[rb4] Chinchilla {kind} variant")]
            if len(g):
                body.append(f"{lab} & 4 & {f3(g.mean_re.min())}--{f3(g.mean_re.max())} & & & "
                            f"{g.Q.min():.1f}--{g.Q.max():.1f} ({f3(g.p_Q.min(), 2)}--{f3(g.p_Q.max(), 2)}) \\\\\n")
    rng = (V.mean_re.min(), V.mean_re.max())
    allx_ = "all six designs (excl. Porian)"
    pw = S[(S["sample"] == allx_) & (S.spec_key == "weighted")].iloc[0]
    pu = S[(S["sample"] == allx_) & (S.spec_key == "unweighted")].iloc[0]
    pS_ = S[(S["sample"] == allx_) & (S.spec_key == "weighted_S")].iloc[0]
    s_ = pd.read_csv(cm.RA1_SUMMARY).set_index("design").loc["Chinchilla"]
    if cm.CHIN_REBUILT:   # [rb4 integration]
        chin_note = (f"Chinchilla {f3(s_.sigma_re)} ({f3(s_.se_sigma_re)}), FLOP-effective $N_F=F/6$ with Hoffmann et "
                     "al.'s FLOPs per token $F$ rebuilt from their architecture table (module rb4\\_chinflop)")
        enote = (" $^{e}$Chinchilla alone replaced by: Hoffmann et al.'s printed, executed and 6$N$ counts, total and "
                 "non-embedding parameters, on nominal budgets or each count's own isocosts (nine variants); its "
                 "count-free $N_F$, profile membership drawn on the corrected coordinates, and the 132 runs without the "
                 "five high-loss runs (three variants; module rb4\\_chinflop). Range of means; $Q$ ($p$).")
    else:
        chin_note = f"Chinchilla {f3(s_.sigma_re)} ({f3(s_.se_sigma_re)}), total $N$"
        enote = ""
    fdrow = V[V.variant.str.startswith("Local first-derivative")]
    fdnote = (f" $^{{d}}$Marin's three corpora and Llama~3 at the local first-derivative estimate $1/(1+b_1)$ of Table "
              f"\\ref{{tab:sigmaC-extrap}}'s surfaces ({fdrow.iloc[0].members.split('; ')[2].split()[2]} and "
              f"{fdrow.iloc[0].members.split('; ')[1].split()[2]}), the estimator used for Farseer; s.e.\\ from module ra1."
              if len(fdrow) else "")
    notes = (
        "Panel A: regression of the budget-level model-free estimates $\\sigma^*_b$ on $\\log_{10}C$; slope per decade of "
        "compute. Inputs (module ra1): 36 bracketed IsoFLOP budgets of Chinchilla, Llama~3 and Marin's three corpora, and "
        "Farseer's local expansion path at 8 compute levels as a sixth design; Porian et al.'s unannealed profiles are "
        "reported separately. 3-level RE: design random intercepts plus a within-design between-budget variance on top of "
        "each budget's bootstrap variance (REML; inverse-variance weights; \\citealt{konstantopoulos2011fixed}). RE, unweighted: linear mixed model with design "
        "random intercepts. Balanced: correlated-effects weights $1/[k_j(\\bar v_j+\\hat\\tau^2)]$ \\citep{hedges2010robust} (its model s.e.\\ is not "
        "reported). FE: design fixed effects, weights $1/\\mathrm{se}^2$, s.e.\\ scaled by the residual variance when above "
        "one (the referees' computation). CR2: cluster-robust s.e.\\ by design with the bias-reduced linearization of "
        "\\citet{bell2002bias} and Satterthwaite degrees of freedom \\citep{pustejovsky2018small,tipton2015small}. Wild $p$: "
        "restricted wild cluster bootstrap-$t$ by design (\\citealt{webb2023reworking} weights; 9,999 "
        "draws, 1,999 for subsamples). CR2 and wild $p$ are shown only with at least four designs and two degrees of "
        f"freedom. CR2 $t$-test $p$-values of the first two rows: {pv(pw.p_cr2_design)} and {pv(pu.p_cr2_design)}; on "
        f"the scale $S=2(1/\\sigma^*-1)$ the 3-level RE slope has CR2 $p$ {pv(pS_.p_cr2_design)} and wild $p$ "
        f"{pv(pS_.p_wcr_design)}. Panel B: population-average line of Panel A's first two rows. $^{{a}}$Beyond the largest bracketed budget "
        "of every design (the largest, Chinchilla's, is $3\\times10^{{21}}$ FLOP). Panel C: raw estimate at each design's largest bracketed budget; design line: "
        "BLUP intercept plus the common slope (3-level RE); last row: inverse-variance mean of the five budgets at "
        "$6\\times10^{20}$ FLOP and above [HKSJ interval]. Panel D: DerSimonian--Laird mean \\citep{dersimonian1986meta}, "
        "modified Hartung--Knapp--Sidik--Jonkman interval ($t_{k-1}$, variance factor truncated at one; "
        "\\citealt{hartung2001tests,sidik2002simple,knapp2003improved,roever2015hartung}). $^{b}$" + chin_note +
        "; Meta 0.660 (0.023), FLOP-implied $N_F=C/(6D)$; Marin, FLOP-implied, the equal-weight mean of its three corpora with the s.e.\\ "
        "of perfectly correlated errors (shared code and FLOP accounting); Farseer, non-embedding $N$, the first-derivative "
        "path estimate 0.708 with the s.e.\\ of its path mean (0.005), which does not treat compute levels as independent. "
        "$\\eta$: elasticity of FLOPs per token with respect to the measured parameter count, "
        "$\\sigma^*_{\\rm eff}=2/[2+S/(1+\\eta)^2]$" + (", applied to Meta only (Chinchilla's accounting is rebuilt)"
                                                      if cm.CHIN_REBUILT else "") + ". $^{c}$Inverse-variance (fixed-effect) pooling of the budgets within "
        "Chinchilla and Llama~3 instead of the random-effects means (module ra1)." + fdnote + enote +
        f" Across all study-level variants (the rows of Panel D and those in the CSV) the mean ranges over "
        f"{f3(rng[0])}--{f3(rng[1])}.")
    src = ("\\citet{hoffmann2022training}; \\citet{besiroglu2024chinchilla}; \\citet{grattafiori2024llama}; "
           "\\citet{czech2026llama3isoflop}; \\citet{marin2026ladders}; \\citet{porian2024resolving}; "
           "\\citet{li2025predictableb}; authors' calculations.")
    tex = _wrap("The Elasticity of Substitution by Compute Level", "tab:sigmaC", COLSPEC % ("0.30", "ccccc"), body, notes, src)
    cm.write_tex(os.path.join(cm.TABLES, f"{cm.PREFIX}_table.tex"), tex)


# ============================================================================ Table: PI of sigma*(C) and wedges
def table_wedge(P):
    pi = P["pi"].set_index("log10C")
    labs = list(P["slopes"].keys())
    Sc = P["scen"]
    Bd = P["bounds"]
    body = []
    body.append("\\multicolumn{7}{@{}l}{\\textit{Panel A. $\\sigma^*(C)$ beyond the designs and the wedge scale $k=1/\\sigma^*-1$}} \\\\[1pt]\n")
    body.append("Compute (FLOP) & $10^{21}$ & $10^{22}$ & $10^{23}$ & $10^{24}$ & $10^{25}$ & $10^{26}$ \\\\\n\\midrule\n")
    cols = [21.0, 22.0, 23.0, 24.0, 25.0, 26.0]
    body.append("$\\sigma^*$, upper bound$^{a}$ & " + " & ".join(f3(pi.loc[c, "sigma_upper"]) for c in cols) + " \\\\\n")
    for i, lab in enumerate(labs):
        b = P["slopes"][lab]
        short = "Chinchilla--Llama 3" if i == 0 else "pooled"
        body.append(f"$\\sigma^*$, lower bound, {short} drift ({f3(b, sign=True)}) & " +
                    " & ".join(f3(pi.loc[c, f'sigma_lower|{lab}']) for c in cols) + " \\\\\n")
    body.append("$k$ at the upper bound & " + " & ".join(f3(pi.loc[c, "k_lower"]) for c in cols) + " \\\\\n")
    body.append("$k$ at the lower bound (Chinchilla--Llama 3) & " + " & ".join(f3(pi.loc[c, f'k_upper|{labs[0]}']) for c in cols) + " \\\\\n")
    body.append("Memo: $k$ at $\\sigma^*=0.70$ & " + " & ".join(f3(pi.loc[c, "k_ref"]) for c in cols) + " \\\\\n")
    body.append("\\addlinespace\n\\multicolumn{7}{@{}l}{\\textit{Panel B. Revealed wedges and shares on the clean sample (77 models)}} \\\\[1pt]\n")
    body.append(" & \\multicolumn{3}{c}{Median} & Llama 3 & OLMo 2 & SmolLM2 \\\\\n\\cmidrule(lr){2-4}\n")
    body.append("Curvature & $w$ & $s$ & $s$, units$^{b}$ & herd $s_f$ & 7B $w$ & 1.7B $w$ \\\\\n\\midrule\n")
    # [rb4 integration] the top-budget value was hard-coded as 0.596 (shown as its rounded value 0.60). If it rounds to
    # 0.60 (Chinchilla in T: 0.596) the 0.60 row carries the label, as before; otherwise (rebuilt: 0.594) its own row is
    # shown with three decimals and the 0.60 row is a plain constant.
    top3 = round(float(P["sig_top"]), 3)
    top_is_060 = abs(round(float(P["sig_top"]), 2) - 0.60) < 1e-9
    for i, r in Sc.iterrows():
        nm = r.scenario
        if top_is_060 and nm.startswith("Constant") and abs(float(nm.split("=")[-1]) - top3) < 1e-9 \
                and abs(top3 - 0.60) > 1e-9:
            continue            # the unrounded top-budget value is in the CSV; the table shows 0.60
        if nm.startswith("Reference"):
            lab = "Reference ($\\sigma^*=0.701$)"
        elif nm.startswith("Constant"):
            v = float(nm.split("=")[-1])
            if top_is_060:
                lab = "Top budgets ($\\sigma^*=0.60$)" if abs(v - 0.60) < 1e-9 else f"$\\sigma^*={v:.2f}$"
            else:
                lab = f"Top budgets ($\\sigma^*={v:.3f}$)" if abs(v - top3) < 1e-9 else f"$\\sigma^*={v:.2f}$"
            if "sigma_upper_ci95_hi" in P["pi"] and abs(v - round(float(P["pi"]["sigma_upper_ci95_hi"].iloc[0]), 3)) < 1e-9:
                lab = f"Top budgets, 95\\% upper ($\\sigma^*={v:.3f}$)"     # review addition
        elif "primary meta-regression" in nm:
            lab = "$\\sigma^*(C_i)$, pooled line"
        elif "Chinchilla and Llama" in nm or "Chinchilla+Llama" in nm:
            lab = "$\\sigma^*(C_i)$, lower bound (Chin.--Llama)"
        else:
            lab = "$\\sigma^*(C_i)$, lower bound (pooled)"
        body.append(f"{lab} & {r.median_w:.2f} & {f3(r.median_s)} & {f3(r.median_s_decision)} & {f3(r['s_f|Llama-3 herd'])} & "
                    f"{r['w|OLMo-2-1124-7B']:.1f} & {r['w|SmolLM2-1.7B']:.1f} \\\\\n")
    body.append("\\addlinespace\n\\multicolumn{7}{@{}l}{\\textit{Panel C. Bounds on the median $s$ (Proposition 5(iv))}} \\\\[1pt]\n")
    body.append(" & \\multicolumn{2}{c}{PI-1} & \\multicolumn{2}{c}{PI-2} & \\multicolumn{2}{c}{PI-3} \\\\\n"
                "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(l){6-7}\n")
    body.append("Curvature range & Lower & Upper & Lower & Upper & Lower & Upper \\\\\n\\midrule\n")
    ktop = cm.sigma_to_k(P["sig_top"])
    cur_labels = [("ra2:", "$k\\in[0.40,0.52]$ (round 2)"),
                  ("k in [0.40, k(0.60)]", "$k\\in[0.40,0.67]$"),
                  (f"drift-agnostic: k in [0.40, k(sigma_lin(C_i))], {labs[0]}", "$k\\in[0.40,k_{\\rm lin}(C_i)]$"),
                  (f"drift maintained: k in [k(sigma_top), k(sigma_lin(C_i))], {labs[0]}", f"$k\\in[{ktop:.2f},k_{{\\rm lin}}(C_i)]$")]
    hi_row = Bd[Bd.curvature.str.startswith("drift maintained, sigma_top at its 95% upper limit") &
                Bd.curvature.str.endswith(labs[0])]
    if len(hi_row):                      # review addition: sigma_top at the upper end of its HKSJ interval
        khi = cm.sigma_to_k(float(P["pi"]["sigma_upper_ci95_hi"].iloc[0]))
        cur_labels.append((f"drift maintained, sigma_top at its 95% upper limit: k in [k(sigma_top_hi), "
                           f"k(sigma_lin(C_i))], {labs[0]}", f"$k\\in[{khi:.2f},k_{{\\rm lin}}(C_i)]^{{c}}$"))
    sets = [s for s in Bd.pi_set.unique() if not s.startswith("PI-4")]
    for pre, lab in cur_labels:
        cells = []
        for st_ in sets:
            r = Bd[(Bd.pi_set == st_) & (Bd.curvature.str.startswith(pre))].iloc[0]
            cells.append(f"{f3(r.median_s_lo, 3)} & {f3(r.median_s_hi, 3)}")
        body.append(f"{lab} & " + " & ".join(cells) + " \\\\\n")
    notes = (
        "Panel A: $^{a}$Partial identification \\citep{manski2003partial}. For compute above the largest bracketed "
        "budgets ($\\approx10^{21}$ FLOP), "
        "$\\sigma^*(C)\\in[\\sigma_{\\rm lin}(C),\\sigma_{\\rm top}]$ if $\\sigma^*$ does not rise with compute beyond the designs "
        f"(upper bound $\\sigma_{{\\rm top}}={f3(float(P['sig_top']))}$: inverse-variance mean of Chinchilla's and Llama~3's estimates at "
        "$6\\times10^{20}$ FLOP and above, placed at $10^{21}$) and falls no faster than the within-design drift measured "
        "where it is observed (lower bound: linear continuation from $\\sigma_{\\rm top}$, truncated to $(0,1)$). If the drift "
        "is not real (Marin shows none up to $3\\times10^{20}$), the level measured within the designs applies (0.687 across studies, 0.701 for the "
        "reference technology); the drift-agnostic set is "
        "$[\\sigma_{\\rm lin}(C),0.70]$. Panel B: $\\ln w=\\theta\\ln(M/M^*_{\\rm ref}(C))$, each model's reference zero point held "
        "fixed (module ra2); $s=(w-1)/w$. $\\sigma^*(C_i)$ rows evaluate the curvature at each model's training compute "
        "(median $2\\times10^{23}$ FLOP; 76 of 77 models above $10^{21}$). $^{b}$Decision units: one family-level share per "
        "common-$D$ family ($W_f=[\\sum_jh_j/w_j]^{-1}$, 11 families) plus the 45 members of size-specific families "
        "and singletons. The share with $w>1$ (0.974) does not depend on $k$. For $w>1$, $\\ln w$ increases with $k$: if "
        "$\\sigma^*$ falls with compute, the reference level is a lower bound on the wedge, not a midpoint. Panel C: medians "
        "of the per-model bounds on $s$ under module ra2's sets for $M^*(C)$ (PI-1: IsoFLOP anchors, path elasticity "
        "$e\\in[-0.16,0.19]$; PI-2: plus $M^*$ nondecreasing; PI-3: own-lab anchors) and the stated curvature range; "
        "$k_{\\rm lin}(C_i)$ uses the Chinchilla--Llama~3 drift (CSV: pooled drift). The first row reproduces module ra2. "
        "$^{c}$Lower curvature bound at the upper end of the 95\\% HKSJ interval of $\\sigma_{\\rm top}$ "
        f"({f3(float(P['pi']['sigma_upper_ci95_hi'].iloc[0]))}), allowing for sampling error in the top-budget value. "
        "The lower bound on $\\sigma^*(C)$ assumes a linear decline at the Chinchilla--Llama~3 rate; a hinge fit gives a "
        "steeper decline above $3\\times10^{20}$, so that bound is not conservative against acceleration.")
    src = "Modules ra1 (budget-level $\\sigma^*_b$) and ra2 (clean sample, reference zero points, bounds on $M^*(C)$); authors' calculations."
    tex = _wrap("Curvature at Frontier Compute and the Size of the Revealed Wedges", "tab:sigmaC-wedge",
                COLSPEC % ("0.31", "cccccc"), body, notes, src)
    cm.write_tex(os.path.join(cm.TABLES, f"{cm.PREFIX}_wedge.tex"), tex)


# ============================================================================ Table: extrapolation
def table_extrap(X, pooled):
    fd = pd.read_csv(cm.RA1_FAR_DELTA)
    lin = pd.read_csv(cm.RA1_FAR_LIN)
    lw = pd.read_csv(cm.RA1_FAR_LNW)
    body = []
    specs = [("kappa_full", "$\\kappa$ free, all runs"), ("chin_full", "$\\kappa=1$, all runs"),
             ("chin_M100", "$\\kappa=1$, $M\\le100$")]

    def two_rows(lab, cells):
        """cells: list of (est, lo, hi); estimates on one line, 95% intervals below."""
        r1 = f"{lab} & " + " & ".join(f3(e, 2, sign=True) if np.isfinite(e) else "--" for e, _, _ in cells) + " \\\\\n"
        r2 = " & " + " & ".join(ci(lo, hi, 2) if np.isfinite(lo) else "" for _, lo, hi in cells) + " \\\\\n"
        return r1 + r2

    def head(txt):
        return f"\\addlinespace\n\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{txt}}}}} \\\\[1pt]\n"

    b2 = lin[(lin.conv == "ne") & (lin.param == "b2")].iloc[0]
    body.append(head(f"Farseer: non-embedding $N$, 404 runs; $b_2=$ {f3(b2.est, 3, sign=True)} {ci(b2.lo, b2.hi, 3)}").replace("\\addlinespace\n", ""))
    body.append("Local $\\ln w$" + "".join(f" & {f3(lw[(lw.conv == 'ne') & (lw.M_bin == b)].lnw_local.iloc[0], 2)}" for b in MB) + " \\\\\n")
    for k1, lab in specs:
        cells = []
        for b in MB:
            r = fd[(fd.spec == k1) & (fd.conv == "ne") & (fd.M_bin == b)].iloc[0]
            cells.append((r.delta, r.lo, r.hi))
        body.append(two_rows(lab, cells))
    for name in ("Marin, Comma", "Marin, DCLM", "Marin, Nemotron-CC", "Llama 3"):
        v = X[name]
        T, L = v["T"], v["L"]
        b2r = L[L.param == "lin|b2"].iloc[0]
        runs = f"{v['n']} runs" + (f" ({v['n_raw'] - v['n']} screened)" if v["n_raw"] != v["n"] else "")
        body.append(head(f"{name}: {runs}; $b_2=$ {f3(b2r.est, 3, sign=True)} {ci(b2r.lo, b2r.hi, 3)}"))
        row = []
        for b in MB:
            t = T[(T.spec == "kappa_full") & (T.M_bin == b)].iloc[0]
            row.append(f"{f3(t.lnw_local, 2)} ({int(t.n_points)})" if np.isfinite(t.lnw_local) else "--")
        body.append("Local $\\ln w$ (points)" + "".join(f" & {x}" for x in row) + " \\\\\n")
        for k1, lab in specs:
            cells = []
            for b in MB:
                r = T[(T.spec == k1) & (T.M_bin == b)].iloc[0]
                cells.append((r.delta, r.lo, r.hi))
            body.append(two_rows(lab, cells))
    body.append(head("Marin, three corpora pooled (equal weights)"))
    for k1, lab in specs:
        cells = []
        for b in MB:
            r = pooled[(pooled.spec == k1) & (pooled.M_bin == b)].iloc[0]
            cells.append((r.delta, r.lo, r.hi))
        body.append(two_rows(lab, cells))
    hdr = (" & \\multicolumn{5}{c}{$M=D/N$} \\\\\n\\cmidrule(l){2-6}\n & " + " & ".join(MB_TEX) + " \\\\\n\\midrule\n")

    def _mm(name):
        g, st0 = X[name]["grid"], X[name]["st0"]
        ok = (st0["neff"] >= 8) & np.isfinite(st0["lnw_local"])
        return float(g["M"].values[ok].max()), float(g["M"].max())
    mm_ = [_mm(n) for n in ("Marin, Comma", "Marin, DCLM", "Marin, Nemotron-CC")]
    mmax_marin, mrun_marin = max(a for a, _ in mm_), max(b for _, b in mm_)
    mmax_llama, mrun_llama = _mm("Llama 3")
    notes = (
        "Entries: $\\Delta=\\ln(w_{\\rm param}/w_{\\rm local})$ averaged over the evaluation points in each bin of $M=D/N$; "
        "95\\% basic bootstrap intervals below. Negative values mean that the parametric form understates the wedge. "
        "Local wedge: on an IsoFLOP design, with $x=\\ln M$ and $c=\\ln C$, $w=(f_c-f_x)/(f_c+f_x)$ for $f=\\ln L$, from a "
        "kernel-weighted local quadratic in $(x,c)$ with a Gaussian product kernel; it needs no functional form and no loss "
        "units. Bandwidths: leave-one-out cross-validation over multiples of the coordinate standard deviations, with the "
        "$c$-bandwidth at least half the largest gap between adjacent budgets; one bandwidth for Marin's three corpora "
        "(pooled criterion, smallest $c$-multiple within 1\\% of the minimum); Llama~3 its own. Evaluation points: 14 "
        "log-spaced $M$ on every budget line inside the budget's observed range, kernel effective sample size at least 8 "
        "(number of points in parentheses). Parametric comparators: the $\\kappa$ family with $\\kappa$ free and the Chinchilla form ($\\kappa=1$), fitted by Huber loss on $\\ln L$ to all runs or to the runs with $M\\le100$ only, in the same FLOP-implied "
        "parameter convention. Intervals: wild cluster bootstrap by budget (Webb weights, 999 draws) around the pilot local "
        "surface, every fit recomputed in each draw. Basic intervals correct for the estimator's bootstrap bias; on "
        "Marin's 84--88 runs this bias is of the order of the sampling spread for the $\\kappa$-free comparator and points "
        "towards a larger understatement, so an interval can lie below its point estimate (CSV: bias, run-level "
        "intervals). $b_2$: coefficient on $u^2$ in $\\ln w_{\\rm local}=b_1u+b_2u^2$, $u=\\ln(M/M^*_{\\rm local}(C))$, with "
        "$M^*_{\\rm local}(C)$ the root of $\\ln w_{\\rm local}=0$ on each budget line; $b_2>0$ means $\\ln w$ is convex, which a "
        "linear-in-$u$ form cannot capture. Screening: a Marin run whose loss exceeds that of the same configuration trained "
        "on fewer tokens is dropped (DCLM, 157M configuration at $1.8\\times10^{20}$ FLOP); results with it are in the CSV. "
        f"The local wedge is evaluated up to $M\\approx{mmax_marin:,.0f}$ on Marin (runs reach {mrun_marin:,.0f}) and "
        f"$M\\approx{mmax_llama:,.0f}$ on Llama~3 (runs reach {mrun_llama:,.0f}); beyond, the kernel effective sample size "
        "falls below 8. The $M\\ge1{,}024$ bin rests on 1--2 corner points per corpus (effective sample size 8--11) and is fragile; "
        "Llama~3's 256--1,024 bin is two points at the lower end of the bin. "
        "Smoothing bias is not in the intervals (bandwidth sensitivity in the CSV). Farseer rows reproduce module ra1 "
        "(wild cluster bootstrap by model size).")
    src = ("\\citet{li2025predictableb}; \\citet{marin2026ladders}; \\citet{grattafiori2024llama}; "
           "\\citet{czech2026llama3isoflop}; authors' calculations.")
    tex = _wrap("Parametric Extrapolation against the Model-Free Wedge in Three Recipes", "tab:sigmaC-extrap",
                COLSPEC % ("0.19", "ccccc"), [hdr] + body, notes, src)
    cm.write_tex(os.path.join(cm.TABLES, f"{cm.PREFIX}_extrap.tex"), tex)


# ============================================================================ Table: tuning
def table_tuning(T):
    t = T["table"]
    body = []
    body.append("Policy whose inefficiency gradients are applied & $\\iota_n$ & $\\iota_d$ & $b_2$ corrected & Tuning share & Path $a_{\\rm obs}-a$ \\\\\n\\midrule\n")
    keep = [("Step Law rule (in-sample point", "Step Law rule, in sample"),
            ("Step Law rule, 95% corner (iota_n -, iota_d +)", "Step Law rule, 95\\% corner ($\\iota_n-$, $\\iota_d+$)"),
            ("Step Law rule, 95% corner (iota_n +, iota_d -)", "Step Law rule, 95\\% corner ($\\iota_n+$, $\\iota_d-$)"),
            ("Random configuration, best of 1", "Random configuration (best of 1): D-biased"),
            ("Random configurations, best of 4", "Random configurations (best of 4)"),
            ("Random configurations, best of 16", "Random configurations (best of 16)"),
            ("One fixed", "One fixed (LR, batch)"), ("Porian base", "Porian et al.\\ base (fixed)"),
            ("Porian N-rule", "Porian et al.\\ $N$-rule"), ("DeepSeek", "DeepSeek $C$-rule"),
            ("Bjorck", "Bjorck et al.\\ $(N,D)$-rule"),
            ("SFA-shaped gradients (exponential frontier), Step Law", "Stochastic frontier shape, Step Law level"),
            ("SFA-shaped gradients (exponential frontier), random", "Stochastic frontier shape, random-configuration level")]
    for pre, lab in keep:
        r = t[t.scenario.str.startswith(pre)].iloc[0]
        io_n = f"{r.iota_n * 1e3:+.2f}".replace("-", "$-$") if np.isfinite(r.iota_n) else "varies"
        io_d = f"{r.iota_d * 1e3:+.2f}".replace("-", "$-$") if np.isfinite(r.iota_d) else "varies"
        body.append(f"{lab} & {io_n} & {io_d} & {f3(r.b2_true, 4)} & {f3(r.tuning_share_of_b2, 2, sign=True)} & "
                    f"{f3(r.path_exponent_shift_pointwise, 4, sign=True)} \\\\\n")
    body.append(f"\\addlinespace\nObserved (no correction) & 0 & 0 & {f3(T['b2_obs'], 4)} & 0 & 0 \\\\\n")
    body.append(f"Breakdown: $\\iota_d$ that removes all convexity ($\\iota_n=0$) & 0 & {T['breakdown_iota_d'] * 1e3:+.1f} & 0 & +1.00 & \\\\\n")
    notes = (
        "Farseer's runs follow Step Law's hyperparameter rules. With observed loss $L^*e^{\\iota(N,D)}$, the technology's "
        "elasticities are the observed ones plus the inefficiency gradients, $\\varepsilon_N=\\varepsilon_N^{\\rm obs}+\\iota_n$ "
        "and $\\varepsilon_D=\\varepsilon_D^{\\rm obs}+\\iota_d$ ($\\iota_n=\\partial\\iota/\\partial\\ln N$, in units of $10^{-3}$ "
        "of log loss per log point). Each row corrects the local wedge at every point of Farseer's grid (module ra1's local "
        "quadratic, primary bandwidth), re-locates the expansion path as the root of the corrected $\\ln w$ and re-estimates "
        "$\\ln w=b_1u+b_2u^2$ ($u=\\ln(M/M^*(C))$; 328 points, $n_{\\rm eff}\\ge15$). Tuning share $=(b_2^{\\rm obs}-b_2)/b_2^{\\rm obs}$: "
        "the part of the convexity attributable to the policy's inefficiency; negative values mean correcting for it makes "
        "$\\ln w$ more convex. Gradients: module m8's estimates on the Step Law grid (17 cells; $N$ 0.21--1.07B non-embedding, "
        "$D$ 4--100B, $M$ 19--466); the 95\\% corners use their cluster-robust standard errors; the stochastic-frontier rows "
        "scale $\\iota$ with elasticities $-0.063$ ($N$) and $-0.260$ ($D$) around the grid's centre. Path $a_{\\rm obs}-a$: "
        "slope in $\\ln C$ of the shift in the located $\\ln N^*(C)$, the pointwise counterpart of Online Appendix "
        "equation~(D6); D6's constant-gradient formula $-(\\iota_n-\\iota_d)E/[(\\alpha+\\beta)R^*(C)]$ on Farseer's "
        "Chinchilla-form fit gives values of the same sign in every row, two to seven times larger, because the fitted form is flatter than the local surface (CSV). Farseer's convexity is driven by "
        "$M$ up to 2,570, beyond Step Law's $M$ range, where the rule's inefficiency is unmeasured.")
    tex = ("\\begin{table}[tp]\n\\centering\n\\caption{How Much of Farseer's Convexity Could Be Mis-Tuning}\n"
           "\\label{tab:sigmaC-tuning}\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n"
           "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}>{\\raggedright\\arraybackslash}p{0.36\\textwidth}ccccc@{}}\n\\toprule\n" + "".join(body) +
           "\\bottomrule\n\\end{tabular*}\n\n\\begin{tablenotes}\n" + notes + "\n\\end{tablenotes}\n\n"
           "\\begin{tablenotes}[Source]\n\\citet{li2025predictableb}; \\citet{li2025predictablea}; modules ra1 and m8; "
           "authors' calculations.\n\\end{tablenotes}\n\\end{table}\n")
    cm.write_tex(os.path.join(cm.TABLES, f"{cm.PREFIX}_tuning.tex"), tex)


# ============================================================================ Table: budget-level estimates (review addition)
def table_budgets(M):
    """Review addition (R2 Major 1 request 1): every budget-level model-free estimate with its window diagnostics
    (runs left and right of the fitted minimum inside the window) and, per design, the fixed-effect (inverse-variance)
    and random-effects pooled estimates side by side. Inputs: ra1_modelfree_isoflop_budgets.csv (IsoFLOP designs),
    ra1_modelfree_farseer_path.csv (Farseer's local path), ra1_modelfree_isoflop_summary.csv and
    ra1_modelfree_farseer_pool.csv (pooled values)."""
    d = M["data"]
    d = d[~d.porian]
    s = pd.read_csv(cm.RA1_SUMMARY).set_index("design")
    pool = pd.read_csv(cm.RA1_FAR_POOL).set_index("conv")
    body = [" & Budget (FLOP) & $\\sigma^*_b$ & (s.e.) & Runs left/right & $M^*_b$ \\\\\n\\midrule\n"]
    order = ["Chinchilla", "Llama 3", "Marin, Comma", "Marin, DCLM", "Marin, Nemotron-CC", "Farseer (local path)"]
    for des in order:
        g = d[d.design == des].sort_values("budget_C")
        first = True
        for r in g.itertuples():
            lr = f"{int(r.nleft)}/{int(r.nright)}" if np.isfinite(r.nleft) else "--"
            lab = des.replace("Farseer (local path)", "Farseer, local path") if first else ""
            body.append(f"{lab} & {sci(r.budget_C)} & {f3(r.y)} & ({f3(r.se)}) & {lr} & {r.Mstar:.1f} \\\\\n")
            first = False
        if des in s.index:
            q = s.loc[des]
            body.append(f"\\quad pooled, FE / RE & & {f3(q.sigma_fe)} / {f3(q.sigma_re)} & ({f3(q.se_sigma_fe)} / "
                        f"{f3(q.se_sigma_re)}) & $Q$ {q.Q:.1f} & $p$ {pv(q.p_Q_boot)} \\\\\n")
        else:
            q = pool.loc["ne"]
            body.append(f"\\quad pooled, mean / RE & & {f3(q.sigma_mean)} / {f3(q.sigma_re)} & ({f3(q.se)} / "
                        f"{f3(q.se_re)}) & $I^2$ {q.I2:.2f} & $p$ {pv(q.p_Q)} \\\\\n")
        body.append("\\addlinespace\n")
    notes = (
        "Model-free $\\sigma^*_b=2/(2+S_b)$, $S_b$ the curvature of budget $b$'s IsoFLOP profile in $\\ln N$ at its "
        "minimum over the slope of the loss--compute frontier (module ra1; quadratic on the window $|\\ln N-x_0|\\le1$ "
        "around the pooled path, log-cubic or log-quadratic frontier). Only bracketed budgets are listed (at least two runs "
        "on each side of the fitted minimum inside the window); runs left/right count them. s.e.: design-conditional "
        "wild bootstrap. $M^*_b=D^*/N^*$ at the fitted minimum. Pooled: fixed-effect (inverse-variance) and "
        "DerSimonian--Laird random-effects means of the budgets on the $S$ scale mapped to $\\sigma^*$, with Cochran's $Q$ "
        "and its bootstrap $p$-value. Farseer: Hessian-based local expansion path at eight compute levels (the only "
        "level-specific Farseer estimator); pooled: plain mean with the bootstrap s.e.\\ that allows for overlapping "
        "kernels, and the random-effects mean. " +
        ("Parameters: FLOP-implied $N_F=C/(6D)$ (Chinchilla, with FLOPs per token rebuilt from Hoffmann et al.'s "
         "architecture table, module rb4\\_chinflop; Llama~3; Marin), non-embedding $N$ (Farseer); $M^*_b$ in tokens per "
         "FLOP-effective parameter." if cm.CHIN_REBUILT else
         "Parameters: total $N$ (Chinchilla), FLOP-implied $N_F=C/(6D)$ (Llama~3, Marin), non-embedding $N$ (Farseer)."))
    src = ("\\citet{hoffmann2022training}; \\citet{besiroglu2024chinchilla}; \\citet{grattafiori2024llama}; "
           "\\citet{czech2026llama3isoflop}; \\citet{marin2026ladders}; \\citet{li2025predictableb}; module ra1; authors' calculations.")
    tex = _wrap("Budget-Level Model-Free Elasticities with Window Diagnostics", "tab:sigmaC-budgets",
                COLSPEC % ("0.22", "ccccc"), body, notes, src)
    cm.write_tex(os.path.join(cm.TABLES, f"{cm.PREFIX}_budgets.tex"), tex)


def make_all(M, P, X, T):
    ET, pooled = write_csvs(M, P, X, T)
    table_sigma(M)
    table_wedge(P)
    table_extrap(X, pooled)
    table_tuning(T)
    table_budgets(M)            # review addition (R2 Major 1 request 1)
    cm.log("  tables written")
