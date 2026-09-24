"""tables.py -- CSV outputs and paper-ready LaTeX tables of module ra1_modelfree (booktabs; AEA.cls tablenotes)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import ra1_common as rc
from ra1_common import texnum as tn
from ra1_common import write_tex

T = rc.TABLES


def csv(df, name):
    df.to_csv(os.path.join(T, f"ra1_modelfree_{name}.csv"), index=False)


def se(x, p=3):
    s = tn(x, p)
    return f"({s})" if s else ""


def ci(lo, hi, p=3):
    return f"[{tn(lo, p)}, {tn(hi, p)}]" if np.isfinite(lo) and np.isfinite(hi) else ""


def pv(p):
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def sci(x):
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


DESIGN_ORDER = ["Chinchilla", "Llama 3", "Marin, Comma", "Marin, DCLM", "Marin, Nemotron-CC", "Porian, RefinedWeb",
                "Porian, OpenWebText2"]


# ============================================================================ exhibit (ii): model-free sigma* by design
def t_sigma(iso, far, meta):
    s = iso["summary"].set_index("design")
    par = iso["param"].set_index("design")
    bias = iso["bias"]
    lines = [" & & & \\multicolumn{3}{c}{Model-free $\\sigma^*$} & \\multicolumn{2}{c}{Same runs, parametric} \\\\",
             "\\cmidrule(lr){4-6}\\cmidrule(l){7-8}",
             "Design & Runs & Budgets & Random effects & Drift/decade & $\\tau$ & $\\kappa=1$ & $\\kappa$ free \\\\",
             "\\midrule",
             "\\multicolumn{8}{@{}l}{\\textit{Panel A. IsoFLOP designs and Farseer's local expansion path}} \\\\"]
    for d in DESIGN_ORDER:
        r = s.loc[d]
        p_ = par.loc[d]
        lines.append(f"{d} & {int(r.n_runs)} & {int(r.k_valid)}/{int(r.k_budgets)} & {tn(r.sigma_re)} {se(r.se_sigma_re)} & "
                     f"{tn(r.drift_sigma_per_decade)} {se(r.se_drift_sigma)} & {tn(r.tau_sigma)} & "
                     f"{tn(p_.sigma_chin)} {se(p_.se_sigma_chin)} & {tn(p_.sigma_kappa)} {se(p_.se_sigma_kappa)} \\\\")
    fp = far["pool"].set_index("conv").loc["ne"]
    fs = far["sigma"].set_index(["conv", "spec"])
    lines.append(f"Farseer (local path) & 404 & {int(fp.k_C)} $C$ & {tn(fp.sigma_re)} {se(fp.se_re)} & "
                 f"{tn(fp.drift_per_decade)} {se(fp.se_drift)} & {tn(fp.tau)} & {tn(fs.loc[('ne', 'chin_full'), 'sigma_star'])} "
                 f"{se(fs.loc[('ne', 'chin_full'), 'se_wild_cluster'])} & {tn(fs.loc[('ne', 'kappa_full'), 'sigma_star'])} "
                 f"{se(fs.loc[('ne', 'kappa_full'), 'se_wild_cluster'])} \\\\")
    lines += ["\\addlinespace",
              "\\multicolumn{8}{@{}l}{\\textit{Panel B. Heterogeneity across technologies (random effects, DerSimonian--Laird)}} \\\\",
              " & $k$ & & Mean [HKSJ 95\\% CI] & Range & $\\tau$ & $I^2$ & $Q$ ($p$) \\\\",
              "\\midrule"]
    h = meta["het"]
    h = h[h.variant == "all technologies"].set_index("set")
    labs = [("model-free sigma*, IsoFLOP designs and Farseer's local path", "Model-free, 8 designs", meta["inputs_modelfree"].y),
            ("model-free sigma*, excluding Porian et al. (unannealed constant-LR profiles)", "Model-free, excl.\\ Porian",
             meta["inputs_modelfree"][~meta["inputs_modelfree"].technology.str.startswith("Porian")].y),
            ("sigma*_kappa, seven sweep-corpus technologies (m2 Table 4; pairs/cell bootstrap SEs)",
             "$\\sigma^*_\\kappa$, 7 sweeps (Table 4 s.e.)", meta["inputs_sweeps"].sigma_star_q),
            ("sigma*_kappa, seven sweep-corpus technologies (design-conditional wild SEs)",
             "$\\sigma^*_\\kappa$, 7 sweeps (wild s.e.)", meta["inputs_sweeps"].sigma_kappa_ra1),
            ("sigma* under kappa = 1 (Chinchilla form), seven technologies (m2 Table 4 SEs)",
             "$\\sigma^*$, $\\kappa=1$, 7 sweeps", meta["inputs_sweeps"].sigma_star),
            ("sigma*_kappa on the IsoFLOP designs (same runs, wild SEs)", "$\\sigma^*_\\kappa$, IsoFLOP designs",
             iso["param"].sigma_kappa)]
    kx = ("model-free sigma*, excluding Porian; Farseer path at the extended-grid CV bandwidth (plain mean over C; "
          "primary SE)")
    if kx in h.index:
        labs.insert(2, (kx, "Model-free, excl.\\ Porian; Farseer $h_N$ at CV opt.", None))
    for key, lab, vals in labs:
        r = h.loc[key]
        if vals is None:
            lines.append(f"{lab} & {int(r.k)} & & {tn(r.mu_re)} {ci(r.lo_hksj, r.hi_hksj)} & & "
                         f"{tn(r.tau)} & {tn(r.I2, 2)} & {tn(r.Q, 1)} ({pv(r.p_Q)}) \\\\")
            continue
        lines.append(f"{lab} & {int(r.k)} & & {tn(r.mu_re)} {ci(r.lo_hksj, r.hi_hksj)} & {tn(np.min(vals), 2)}--{tn(np.max(vals), 2)} & "
                     f"{tn(r.tau)} & {tn(r.I2, 2)} & {tn(r.Q, 1)} ({pv(r.p_Q)}) \\\\")
    fsv = far["sens"].set_index(["variant", "key"])["value"] if "sens" in far else None

    def _fs(variant):
        k_ = (variant, "path_sigma_mean_poolC|ne")
        return float(fsv.loc[k_]) if fsv is not None and k_ in fsv.index else np.nan

    bmax = bias[bias.ok & bias.design.isin(DESIGN_ORDER)]
    b_np = bmax[~bmax.design.str.startswith("Porian")].bias_pooled.abs().max()
    b_p = bmax[bmax.design.str.startswith("Porian")].bias_pooled.abs().max()
    cub = [d for d in DESIGN_ORDER if int(s.loc[d, "order"]) == 3]
    order_txt = ("quadratic in every design (a pooled $F$ test of the cubic term never rejects at 5\\%)" if not cub else
                 "quadratic, cubic for " + ", ".join(cub) + " (pooled $F$ test of the cubic term)")
    notes = ("Model-free $\\sigma^*$: at every compute-optimal point $1/\\sigma^*-1=L_{nn}|_C/(2|dL^*/d\\ln C|)$, the "
             "curvature of the IsoFLOP profile in $\\ln N$ at its minimum over twice the slope of the loss--compute "
             "frontier; the ratio needs no $E$, no outer exponent, no functional form and no loss units. Per budget: "
             f"{order_txt} in $\\ln N$ on runs "
             "within $\\pm1$ of a window centre on the pooled Approach-2 path; frontier slope from a cubic in $\\ln C$ of "
             "$\\ln L^*$ (quadratic with fewer than six budgets). Budgets whose minimum is not bracketed by two runs on "
             "each side are dropped (valid/total in column 3). Random effects: DerSimonian--Laird mean of the budget "
             "estimates $S_b=2(1/\\sigma^*_b-1)$ mapped to $\\sigma^*$, standard error in parentheses; $\\tau$: "
             "between-budget s.d.; drift: change in $\\sigma^*_b$ per decade of compute (weighted least squares). "
             "Standard errors of the drift and of the parametric columns: design-conditional wild bootstrap "
             f"(budgets: {int(s.B.iloc[0])} draws re-estimating windows, polynomials and frontier; Rademacher weights "
             "on smoother residuals plus Webb weights on budget-level frontier residuals; parametric: Feng--He--Hu "
             "weights). Parametric columns refit the Chinchilla form and the $\\kappa$ family by Huber loss on the same "
             "IsoFLOP runs in the same parameter convention. Farseer: random-effects mean of the local-quadratic $\\sigma$ "
             "on the local expansion path ($w=1$) at "
             f"{int(fp.k_C)} compute levels from {sci(fp.Cmin)} to {sci(fp.Cmax)} FLOP; wild cluster bootstrap by model "
             f"size ({int(far['B'])} draws). Noise-free finite-grid bias of the model-free estimator under each design's "
             f"own fitted Chinchilla and $\\kappa$ technologies: at most {tn(b_np)} (pooled), except {tn(b_p)} under the "
             "strongly curved $\\kappa$ fit to Porian et al.'s profiles. Farseer's local-path $\\sigma^*$ is a "
             "second-derivative object and is bandwidth-sensitive (mean over compute levels "
             f"{tn(_fs('extended-grid CV optimum'))} at the leave-one-out optimum on an extended grid, "
             f"{tn(_fs('primary (CV bandwidth, n_eff >= 15)'))} at the primary bandwidth and {tn(_fs('bandwidth x 2'))} at "
             "twice it); its standard error excludes smoothing bias. "
             "Porian et al.'s profiles are read off "
             "constant-learning-rate runs without cooldown. Panel B: HKSJ = Hartung--Knapp--Sidik--Jonkman interval ($t_{k-1}$); $Q$: Cochran's "
             "homogeneity statistic with $\\chi^2_{k-1}$ $p$-value.")
    write_tex("ra1_modelfree_sigma", "The Elasticity of Substitution without a Functional Form", "tab:modelfree-sigma",
              "lccccccc", lines, notes,
              source="\\citet{hoffmann2022training}; \\citet{besiroglu2024chinchilla}; \\citet{grattafiori2024llama}; "
                     "\\citet{czech2026llama3isoflop}; \\citet{marin2026ladders}; \\citet{porian2024resolving}; "
                     "\\citet{li2025predictableb}; authors' calculations.")


# ============================================================================ review addition: slope-based path sigma*
def sigma_slope(far):
    """First-derivative-based local-path sigma* on Farseer (review addition). At a compute-optimal point of any smooth
    technology d ln w/d ln M|_C = 1/sigma* - 1 (verified numerically in the identity check), so
    sigma*_slope = 1/(1 + b1), with b1 the slope at u = 0 of the quadratic regression of ln w_local on
    u = ln(M/M*_local(C)) (pooled over compute levels). Unlike the Hessian-based path sigma*, it uses only first
    derivatives of the local surface. Interval: the wild cluster bootstrap interval of b1 mapped through 1/(1 + b1)."""
    lin = far["lin"].set_index(["conv", "param"])
    sv = far.get("sens")
    rows = []
    for conv in ("ne", "emb"):
        if (conv, "b1") not in lin.index:
            continue
        r = lin.loc[(conv, "b1")]
        rows.append(dict(conv=conv, variant="primary (wild cluster bootstrap by model size)", b1=r.est,
                         sigma_slope=1 / (1 + r.est), se=r.se / (1 + r.est) ** 2, lo=1 / (1 + r.hi), hi=1 / (1 + r.lo)))
        if sv is not None:
            for var in sv.variant.unique():
                q = sv[(sv.variant == var) & (sv.key == f"lin|{conv}|b1")]
                if len(q) and np.isfinite(q.value.iloc[0]):
                    rows.append(dict(conv=conv, variant=var, b1=float(q.value.iloc[0]),
                                     sigma_slope=1 / (1 + float(q.value.iloc[0]))))
    return pd.DataFrame(rows)


# ============================================================================ exhibit (iii): Farseer w-extrapolation
def t_farseer(far):
    d = far["delta"]
    lw = far["lnw"].set_index(["conv", "M_bin"])
    bins = ["M<16", "16-64", "64-256", "256-1,024", ">=1,024"]
    blab = {"M<16": "$M<16$", "16-64": "16--64", "64-256": "64--256", "256-1,024": "256--1,024", ">=1,024": "$\\ge$1,024"}
    specs = [("chin_full", "Chinchilla, all"), ("chin_M100", "Chinchilla, $M\\le100$"), ("kappa_full", "$\\kappa$ free, all"),
             ("kappa_M100", "$\\kappa$ free, $M\\le100$"), ("eq3_full", "Farseer Eq.~3")]
    lines = [" & Local & \\multicolumn{5}{c}{$\\ln(w_{\\text{param}}/w_{\\text{local}})$} \\\\",
             "\\cmidrule(l){3-7}",
             "$M=D/N$ & $\\ln w$ & " + " & ".join(l for _, l in specs) + " \\\\", "\\midrule"]
    for conv, lab in (("ne", "Panel A. Non-embedding parameters (Farseer's convention)"),
                      ("emb", "Panel B. Parameters including both embeddings")):
        lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{lab}}}}} \\\\")
        for b in bins:
            if (conv, b) not in lw.index or not np.isfinite(lw.loc[(conv, b), "lnw_local"]):
                continue
            row = [blab[b], f"{tn(lw.loc[(conv, b), 'lnw_local'], 2)}"]
            row2 = ["", se(lw.loc[(conv, b), "se"], 3)]
            for sp, _ in specs:
                r = d[(d.spec == sp) & (d.conv == conv) & (d.M_bin == b)]
                if len(r) == 0 or not np.isfinite(r.delta.iloc[0]):
                    row.append("")
                    row2.append("")
                    continue
                r = r.iloc[0]
                row.append(tn(r.delta, 2))
                row2.append(ci(r.lo, r.hi, 2))
            lines.append(" & ".join(row) + " \\\\")
            lines.append(" & ".join(row2) + " \\\\")
        lines.append("\\addlinespace")
    lines = lines[:-1]
    lin = far["lin"].set_index(["conv", "param"])
    fp = far["pool"].set_index("conv")
    sv = far["sens"].set_index(["variant", "key"])["value"]

    def _sv(variant, spec):
        return float(sv.loc[(variant, f"delta|{spec}|ne|>=1,024")])

    def _svk(variant, key):
        return float(sv.loc[(variant, key)]) if (variant, key) in sv.index else np.nan

    notes = ("Farseer's 404 runs \\citep{li2025predictableb}. Local $\\ln w$: kernel-weighted local quadratic of $\\ln L$ "
             "in $(\\ln N,\\ln D)$, $w=\\partial_{\\ln N}\\ln L/\\partial_{\\ln D}\\ln L$ (invariant to monotone "
             "transformations of the loss; no $E$ or functional form), averaged over grid points inside each model "
             "size's observed token range with kernel effective sample size $\\ge15$; bandwidths by leave-one-out "
             f"cross-validation over multiples of the standard deviation ($h_N={tn(far['bw']['ne'][0], 2)}$, "
             f"$h_D={tn(far['bw']['ne'][1], 2)}$ in logs; $h_N$ is the smallest multiple on that grid). "
             "Other columns: log ratio of the wedge implied by a parametric technology to the local wedge at the same "
             "points; negative values mean the parametric form understates the wedge. Chinchilla: "
             "$L=E+AN^{-\\alpha}+BD^{-\\beta}$ (Huber); $\\kappa$ free: $L=E+(AN^{-a_1}+BD^{-b_1})^\\kappa$ (Huber); "
             "Farseer Eq.~3: the authors' form (nonlinear least squares); `$M\\le100$': fitted only to runs with "
             "$M\\le100$ and extrapolated. 95\\% basic intervals in brackets from a wild cluster bootstrap by model "
             f"size (Webb weights, {int(far['B'])} draws; every model refitted); they do not include the local "
             "estimator's smoothing bias. With both bandwidths multiplied by 1.5 and 2 the $M\\ge1{,}024$ entries "
             f"become {tn(_sv('bandwidth x 1.5', 'chin_full'), 2)} and {tn(_sv('bandwidth x 2', 'chin_full'), 2)} "
             f"(Chinchilla, all) and {tn(_sv('bandwidth x 1.5', 'kappa_full'), 2)} and "
             f"{tn(_sv('bandwidth x 2', 'kappa_full'), 2)} ($\\kappa$ free, all); at the narrower leave-one-out optimum on an "
             "extended grid ($h_N$ "
             f"{tn(_svk('extended-grid CV optimum', 'h_N|ne'), 3)}) they become "
             f"{tn(_sv('extended-grid CV optimum', 'chin_full'), 2)}, {tn(_sv('extended-grid CV optimum', 'kappa_full'), 2)} and "
             f"{tn(_sv('extended-grid CV optimum', 'eq3_full'), 2)} (Chinchilla, $\\kappa$ free, Eq.~3), and the local-path "
             f"$\\sigma^*$ averages {tn(_svk('extended-grid CV optimum', 'path_sigma_mean_poolC|ne'))} instead of "
             f"{tn(_svk('primary (CV bandwidth, n_eff >= 15)', 'path_sigma_mean_poolC|ne'))}. Local path: random-effects mean $\\sigma^*$ "
             f"{tn(fp.loc['ne', 'sigma_re'])} ({tn(fp.loc['ne', 'se_re'])}); slope of local $\\ln w$ in "
             f"$\\ln(M/M^*)$ {tn(lin.loc[('ne', 'b1_only'), 'est'])} against $1/\\sigma^*-1=$ "
             f"{tn(lin.loc[('ne', 'path_1_over_sigma_minus_1'), 'est'])} on the path (at the path the slope equals "
             "$1/\\sigma^*-1$ for any smooth technology, so only linearity is testable); quadratic term "
             f"{tn(lin.loc[('ne', 'b2'), 'est'])} ({tn(lin.loc[('ne', 'b2'), 'se'])}), i.e. $\\ln w$ is convex in $\\ln(M/M^*)$. "
             + (lambda t_: f"The first-derivative path estimate $1/(1+b_1)$, $b_1$ the slope at $M=M^*$, is "
                f"{tn(t_.sigma_slope.iloc[0])} ({tn(t_.se.iloc[0])}) and {tn(t_.sigma_slope.min())}--{tn(t_.sigma_slope.max())} "
                "across bandwidths.")(sigma_slope(far).query("conv == 'ne'")))
    write_tex("ra1_modelfree_farseer_w", "Is the Wedge Extrapolated Correctly? Local and Parametric Wedges on Farseer",
              "tab:modelfree-w", "lcccccc", lines, notes,
              source="\\citet{li2025predictableb}; authors' calculations.")


# ============================================================================ exhibit (iv): practitioner overhead
def t_practitioner(pr):
    tab, dv, _ = pr
    h = tab[tab.split.str.startswith("homothetic")]
    ks = sorted(h.k.unique())
    lines = [" & \\multicolumn{4}{c}{Training compute $C/C_{\\min}$ at $M=k\\,M^*$} & "
             "\\multicolumn{4}{c}{Implied wedge $w=k^{1/\\sigma^*-1}$} \\\\",
             "\\cmidrule(lr){2-5}\\cmidrule(l){6-9}",
             "$\\sigma^*$ & " + " & ".join(f"$k={k}$" for k in ks) + " & " + " & ".join(f"$k={k}$" for k in ks) + " \\\\",
             "\\midrule"]
    for sg in sorted(h.sigma_star.unique()):
        r = h[h.sigma_star == sg].set_index("k")
        lines.append(f"{sg:.2f} & " + " & ".join(tn(r.loc[k, 'C_over_Cmin'], 2) for k in ks) + " & " +
                     " & ".join(tn(r.loc[k, 'w'], 2) for k in ks) + " \\\\")
    nh = tab[~tab.split.str.startswith("homothetic")]
    hm = h.set_index(["sigma_star", "k"])
    dev_b = [abs(r.C_over_Cmin / hm.loc[(r.sigma_star, r.k), "C_over_Cmin"] - 1) for _, r in nh.iterrows()
             if r.k <= 100 and "Besiroglu" in r.split]
    dev_h = [abs(r.C_over_Cmin / hm.loc[(r.sigma_star, r.k), "C_over_Cmin"] - 1) for _, r in nh.iterrows()
             if r.k <= 100 and "Hoffmann" in r.split]
    d0 = dv.iloc[0]
    notes = ("Compute needed to reach the loss of the compute-optimal model when training at $k$ times the "
             "compute-optimal tokens-per-parameter ratio, relative to the minimum: "
             "$C/C_{\\min}=((\\alpha+\\beta w)/(\\alpha+\\beta))^{1/\\gamma}w^{-1/\\alpha}$ with "
             "$\\ln w=(1/\\sigma^*-1)\\ln k$; homothetic case $\\alpha=\\beta=\\rho=1/\\sigma^*-1$, "
             "$C/C_{\\min}=\\cosh(\\rho\\ln k/2)^{2/\\rho}$ (every entry checked by direct minimization). "
             "At the same $\\sigma^*$, the $\\alpha/\\beta$ split of \\citet{besiroglu2024chinchilla} changes the entries by at "
             f"most {100 * max(dev_b):.1f}\\% and that of \\citet{{hoffmann2022training}} by at most {100 * max(dev_h):.0f}\\% "
             "for $k\\le100$ ($k$ relative to $M^*$ at the model's own compute). "
             "The wedge $w$ equals lifetime over training compute for a developer who minimizes lifetime compute "
             "(Proposition 2); $(w-1)/w$ is the planned inference share. With the parameters in "
             "\\citet{devries2023go} ($\\alpha=0.32$, $\\beta=0.28$, $\\sigma^*=0.77$) a model at 30\\% of the "
             f"compute-optimal size needs {100 * d0.overhead_devries:.0f}\\% more compute, $k={d0.k_own_compute:.1f}$; "
             f"the formula gives {100 * d0.overhead_our_formula:.0f}\\%.")
    write_tex("ra1_modelfree_practitioner", "The Compute Cost of Over-Training", "tab:overhead", "lcccccccc", lines,
              notes, source="Authors' calculations; \\citet{devries2023go}.")


# ============================================================================ Chinchilla inference
SCHEME_LAB = {"pairs": "Pairs (runs)", "cluster9": "9 budget clusters (all runs)", "cluster_trunk": "Budgets + trunks (44)",
              "cluster_single": "Budgets + singletons (117)", "wild_fhh": "Wild, Feng--He--Hu (runs)",
              "wild_cl_webb": "Wild cluster, Webb (44)"}


def t_chinchilla(ch):
    s = ch["schemes"].set_index("scheme")
    lines = ["Scheme & $\\alpha$ & $\\beta$ & $a$ & $\\sigma^*$ & $M^*(5.76\\times10^{23})$ & $w$(70B) & $\\kappa$ & "
             "$p(\\kappa=1)$ & $p(\\alpha=\\beta)$ \\\\", "\\midrule",
             "\\multicolumn{10}{@{}l}{\\textit{Panel A. Estimates and standard errors by bootstrap scheme ($n=240$)}} \\\\"]
    r0 = s.iloc[0]
    lines.append(f"Estimate & {tn(r0.alpha)} & {tn(r0.beta)} & {tn(r0.a)} & {tn(r0.sigma_star)} & {tn(r0['Mstar_5.76e+23'], 1)} & "
                 f"{tn(r0.w70b, 2)} & {tn(r0.kappa)} & & \\\\")
    for sch in ["pairs", "cluster9", "cluster_trunk", "cluster_single", "wild_fhh", "wild_cl_webb"]:
        r = s.loc[sch]
        lines.append(f"{SCHEME_LAB[sch]} & {se(r.se_alpha)} & {se(r.se_beta)} & {se(r.se_a)} & {se(r.se_sigma_star)} & "
                     f"{ci(r['lo_Mstar_5.76e+23'], r['hi_Mstar_5.76e+23'], 1)} & {ci(r.lo_w70b, r.hi_w70b, 2)} & "
                     f"{se(r.se_kappa)} & {pv(r.p_kappa1_wald)} & {pv(r.p_ces_wald)} \\\\")
    t = ch["tests"].set_index("null")
    lines += ["\\addlinespace", "\\multicolumn{10}{@{}l}{\\textit{Panel B. Likelihood-ratio tests}} \\\\",
              " & \\multicolumn{3}{c}{Laplace quasi-LR} & \\multicolumn{3}{c}{Gaussian LR} & \\multicolumn{2}{c}{Koenker--Bassett} & \\\\",
              "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-9}",
              "$H_0$ & Stat. & $p_{\\chi^2}$ & $p_{\\text{boot}}$ & Stat. & $p_{\\chi^2}$ & $p_{\\text{boot}}$ & Stat. & $p_{\\chi^2}$ & \\\\"]
    for null, lab in (("kappa1", "$\\kappa=1$"), ("ces", "$\\alpha=\\beta$")):
        r = t.loc[null]
        lines.append(f"{lab} & {tn(r.stat_laplace_qlr, 1)} & {pv(r.p_chi2_laplace)} & {pv(r.p_boot_laplace)} & "
                     f"{tn(r.stat_gauss_lr, 1)} & {pv(r.p_chi2_gauss)} & {pv(r.p_boot_gauss)} & {tn(r.stat_kb, 1)} & "
                     f"{pv(r.p_chi2_kb)} & \\\\")
    st = ch["sets"]
    lines += ["\\addlinespace", "\\multicolumn{10}{@{}l}{\\textit{Panel C. Profile over $\\sigma^*$ with $\\kappa$ free, grid (0.05, 0.99)}} \\\\",
              "Sample & Statistic & & $\\hat\\sigma^*$ & 95\\% set & Grid edge hit & Max.\\ LR & & & \\\\"]
    for _, r in st.iterrows():
        edge = "upper" if r.set_hits_hi else ("lower" if r.set_hits_lo else "none")
        slab = ("Full design ($n=240$)" if r["sample"].startswith("full") else
                f"On-path band ($n={ch['band_n']}$)")
        lines.append(f"{slab} & {r.statistic} & & {tn(r.sigma_hat_unconstrained)} & "
                     f"[{tn(r.set_lo, 2)}, {tn(r.set_hi, 2)}] & {edge} & {tn(r.max_LR, 1)} & & & \\\\")
    cal = ch["calibration"]
    fi = ch["fhh_info"]
    # bootstrap-calibrated set on the full design: largest restricted-bootstrap critical value at the chi2-set endpoints
    pf_ = ch["profile"]
    fullH = pf_[pf_["sample"].str.startswith("full") & (pf_.objective == "huber")].sort_values("sigma_star")
    st_full = st[st["sample"].str.startswith("full") & (st.statistic == "Laplace QLR")].iloc[0]
    cend = cal[cal["sample"].str.startswith("full") & cal.sigma0.isin([st_full.set_lo, st_full.set_hi])]
    c_cal = float(cend.crit95_boot.max()) if len(cend) else np.nan
    ins = fullH[fullH.LR <= c_cal]
    cal_set = (float(ins.sigma_star.min()), float(ins.sigma_star.max())) if len(ins) else (np.nan, np.nan)
    notes = ("Epoch's digitization of the Chinchilla runs \\citep{besiroglu2024chinchilla}: 245 runs, 137 on nine "
             "IsoFLOP profiles and 108 off-profile, training FLOP $1.4\\times10^{18}$--$1.3\\times10^{22}$; five "
             "highest-loss runs dropped ($n=240$: 132 on profiles, 108 off). Estimator: Huber loss "
             f"($\\delta=10^{{-3}}$) on $\\ln L$, least absolute deviations in practice ({100 * fi['share_linear']:.0f}\\% of "
             "residuals in the linear region). Panel A: bootstrap standard errors in parentheses and 95\\% percentile "
             f"intervals in brackets, {int(s.B.iloc[0])} draws per scheme. Pairs: resample runs; 9 budget clusters: every "
             "run assigned to the nearest IsoFLOP budget (the original scheme); budgets + trunks: IsoFLOP runs clustered "
             "by budget, off-profile runs by model size; budgets + singletons: off-profile runs as their own clusters; "
             "wild: design held fixed, $y^*=\\hat y+v|\\tilde r|$ around the $\\kappa$-free fit with Feng--He--Hu weights "
             "$v=\\pm1$ and leverage-adjusted residuals \\citep{feng2011wild}; wild cluster: one Webb weight per cluster "
             "\\citep{webb2023reworking}. $p$-values: Wald with the scheme's standard error. Panel B: statistics relative "
             "to the unrestricted fit; $p_{\\text{boot}}$ from a restricted wild (Feng--He--Hu) bootstrap that "
             f"generates the data under $H_0$ ({int(t.B.iloc[0])} draws; smallest attainable $p$ = {t.p_floor.iloc[0]:.3f}); "
             "Koenker--Bassett: $4\\hat f(0)(S_1^r-S_1^u)$ with $S_1$ the sum of absolute log residuals. "
             + (f"Heteroskedasticity-robust Wald statistics for the Gaussian fit of $\\alpha=\\beta$: "
                f"{tn(t.loc['ces', 'wald_gauss_hc1'], 2)} (HC1, $p={t.loc['ces', 'p_wald_gauss_hc1']:.3f}$) and "
                f"{tn(t.loc['ces', 'wald_gauss_hc3'], 2)} (HC3, $p={t.loc['ces', 'p_wald_gauss_hc3']:.3f}$), against "
                f"{tn(t.loc['ces', 'wald_gauss_homosk'], 2)} under homoskedasticity. " if "wald_gauss_hc1" in t.columns else "")
             + (f"Wild schemes draw around the $\\kappa$-free fit, whose Chinchilla-form pseudo-true values differ from "
                f"the estimates ($w$(70B) {tn(s.loc['wild_fhh', 'pop_w70b'], 2)}, $M^*$ {tn(s.loc['wild_fhh', 'pop_Mstar_5.76e+23'], 1)}); "
                f"basic intervals (in logs) are [{tn(s.loc['wild_fhh', 'lo_basic_Mstar_5.76e+23'], 1)}, "
                f"{tn(s.loc['wild_fhh', 'hi_basic_Mstar_5.76e+23'], 1)}] and [{tn(s.loc['wild_fhh', 'lo_basic_w70b'], 2)}, "
                f"{tn(s.loc['wild_fhh', 'hi_basic_w70b'], 2)}] (Feng--He--Hu). " if "pop_w70b" in s.columns else "")
             + "Panel C: "
             "profile of the $\\kappa$-family objective over $\\sigma^*=2/(2+a_1+b_1)$, statistics relative to the "
             "unconstrained optimum; sets use $\\chi^2_1$ critical values. Restricted-bootstrap 95\\% critical values of "
             "the Laplace quasi-LR at selected $\\sigma^*$: " +
             "; ".join(f"{r['sample'].split(' (')[0]}, {r.sigma0:.2f}: {r.crit95_boot:.1f}" for _, r in cal.iterrows()) +
             f" ({int(cal.B.iloc[0])} draws each); with the largest endpoint critical value ({c_cal:.1f}) the full-design "
             f"set is [{cal_set[0]:.2f}, {cal_set[1]:.2f}]. On-path band: runs with $|\\ln N-\\ln N^*(C)|\\le0.15$ "
             f"($n={ch['band_n']}$).")
    pd.DataFrame([dict(sample="full design (n=240)", statistic="Laplace QLR, bootstrap-calibrated", critical_value=c_cal,
                       set_lo=cal_set[0], set_hi=cal_set[1])]).to_csv(
        os.path.join(T, "ra1_modelfree_chinchilla_profile_set_calibrated.csv"), index=False)
    write_tex("ra1_modelfree_chinchilla_inference", "Inference on the Chinchilla Data: Clusters, Wild Bootstrap and Profiles",
              "tab:chin-inference", "lccccccccc", lines, notes,
              source="\\citet{hoffmann2022training}; \\citet{besiroglu2024chinchilla}; authors' calculations.")


# ============================================================================ sigma*_kappa robustness
def t_kappa(kap):
    tab, ep = kap
    lines = ["Technology & Variant & $n$ & $\\sigma^*_\\kappa$ & $\\hat\\kappa$ & $\\sigma^*$, $\\kappa=1$ & "
             "$\\sigma^*_\\kappa$, $E=\\hat E_{\\kappa=1}$ & $\\sigma^*_\\kappa$ over $E$ set \\\\", "\\midrule"]
    prev = None
    for _, r in tab.iterrows():
        tech = r.tech if r.tech != prev else ""
        if prev is not None and r.tech != prev:
            lines.append("\\addlinespace")
        prev = r.tech
        eset = f"{tn(r.sigma_kappa_over_E_set_lo)}--{tn(r.sigma_kappa_over_E_set_hi)}" if np.isfinite(
            r.get("sigma_kappa_over_E_set_lo", np.nan)) else ""
        lines.append(f"{tech} & {r.variant.replace('<', '$<$').replace('>', '$>$')} & {int(r.n)} & {tn(r.sigma_kappa)} {se(r.se_sigma_kappa)} & "
                     f"{tn(r.kappa, 2)} & {tn(r.sigma_chin)} & {tn(r.sigma_kappa_E_chin)} & {eset} \\\\")
    b = tab[tab.variant == "baseline"]
    notes = ("$\\sigma^*_\\kappa=2/(2+a_1+b_1)$ from $L=E+(AN^{-a_1}+BD^{-b_1})^\\kappa$ fitted by Huber loss on "
             "$\\ln L$. Standard errors: design-conditional wild bootstrap (Feng--He--Hu weights on leverage-adjusted "
             "residuals; Webb weights by model size for Farseer and by (size, multiplier) cell for the OLMo ladder), "
             f"{int(b.B.iloc[0])} draws for baselines and {int(tab[tab.variant != 'baseline'].B.iloc[0])} for variants. "
             "Column 7 fixes $E$ at the Chinchilla-form estimate and refits the rest. Column 8: range of "
             "$\\sigma^*_\\kappa$ over the values of $E$ whose profile Laplace quasi-LR is below the $\\chi^2_1$ 95\\% "
             "critical value (descriptive; a 36-point grid in $E$ refined with 25 points around the set).")
    write_tex("ra1_modelfree_kappa_robustness", "Robustness of the Elasticity with the Outer Exponent Free",
              "tab:kappa-robust", "llcccccc", lines, notes,
              source="\\citet{hoffmann2022training}; \\citet{li2025predictableb}; \\citet{gadre2024language}; "
                     "\\citet{bhagia2024establishing}; \\citet{muennighoff2023scaling}; authors' calculations.")


def run(cache):
    iso, far, ch, kap, pr, meta = (cache(k) for k in ("isoflop", "farseer", "chinchilla", "kappa", "practitioner", "meta"))
    ident = cache("identity")
    csv(ident, "identity_check")
    csv(far["identity_eq3"], "identity_check_farseer_eq3")
    for k in ("summary", "budgets", "invalid", "sens", "param", "bias", "mc", "order"):
        csv(iso[k], f"isoflop_{k}")
    if "marin_eta" in iso:
        csv(iso["marin_eta"], "isoflop_marin_flop_accounting")
    for k in ("delta", "lnw", "lin", "path", "pool", "sigma", "grid"):
        csv(far[k], f"farseer_{k}")
    csv(far["sens"], "farseer_sensitivity")
    csv(far["cv"], "farseer_bandwidth_cv")
    if "cv_ext" in far:
        csv(far["cv_ext"], "farseer_bandwidth_cv_extended")
    csv(sigma_slope(far), "farseer_sigma_slope")
    csv(pd.DataFrame([dict(conv=c, hx=far["bw"][c][0], hz=far["bw"][c][1]) for c in far["bw"]] +
                     [dict(conv="icc_" + k, hx=v, hz=np.nan) for k, v in far["icc"].items()]), "farseer_bandwidth_icc")
    csv(ch["facts"], "chinchilla_design")
    csv(ch["schemes"], "chinchilla_schemes")
    csv(ch["tests"], "chinchilla_tests")
    csv(ch["profile"], "chinchilla_profile_sigma")
    csv(ch["sets"], "chinchilla_profile_sets")
    csv(ch["calibration"], "chinchilla_profile_calibration")
    csv(kap[0], "kappa_robustness")
    csv(kap[1], "kappa_E_profile")
    csv(pr[0], "practitioner")
    csv(pr[1], "practitioner_devries")
    csv(pr[2], "practitioner_devries_curve")
    csv(meta["het"], "heterogeneity")
    csv(meta["inputs_sweeps"], "heterogeneity_inputs_sweeps")
    csv(meta["inputs_modelfree"], "heterogeneity_inputs_modelfree")
    csv(meta["pairwise_m2"], "heterogeneity_pairwise_tableSE")
    csv(meta["pairwise_wild"], "heterogeneity_pairwise_wildSE")
    t_sigma(iso, far, meta)
    t_farseer(far)
    t_practitioner(pr)
    t_chinchilla(ch)
    t_kappa(kap)
