"""tables.py -- CSV and LaTeX (booktabs + threeparttable, AER style) outputs of module m3_wedge."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from common import TABLES

P = "m3_wedge_"


def _esc(s):
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")


def write_tex(name, caption, label, colspec, header, body, notes, size=r"\small"):
    lines = [r"\begin{table}[htbp]", r"\centering", size, r"\begin{threeparttable}", rf"\caption{{{caption}}}",
             rf"\label{{{label}}}", rf"\begin{{tabular}}{{@{{}}{colspec}@{{}}}}", r"\toprule"]
    lines += header + [r"\midrule"] + body + [r"\bottomrule", r"\end{tabular}", r"\begin{tablenotes}\footnotesize",
                                               rf"\item \textit{{Notes:}} {notes}", r"\end{tablenotes}",
                                               r"\end{threeparttable}", r"\end{table}"]
    with open(os.path.join(TABLES, P + name + ".tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def f2(x, p=2):
    return "" if x is None or not np.isfinite(x) else f"{x:.{p}f}"


def nfmt(n):
    """parameters: 0.27, 8.0, 405 (billions)"""
    b = n / 1e9
    return f"{b:.2f}" if b < 1 else (f"{b:.1f}" if b < 100 else f"{b:.0f}")


def dfmt(D):
    t = D / 1e12
    return f"{t:.2f}" if t < 1 else f"{t:.1f}"


def mfmt(M):
    return f"{M:,.0f}" if M >= 10 else f"{M:.1f}"


# ----------------------------------------------------------------------------- technologies
def tech_table(tt):
    tt.to_csv(os.path.join(TABLES, P + "technologies.csv"), index=False)
    order = ["chin", "chin_nls", "chin245", "besi", "hoff", "farseer", "farseer_emb", "farseer_q", "gadre_rw", "gadre_c4", "gadre_rp", "olmo", "meta_a2", "meta_a3"]
    body = []
    for k in order:
        r = tt.set_index("key").loc[k]
        sup = f"{r['support_M_min']:.1f}--{r['support_M_max']:,.0f}" if np.isfinite(r.get("support_M_max", np.nan)) else ""
        body.append(f"{_esc(r['label'])} & {r['alpha']:.3f} & {r['beta']:.3f} & {r['a']:.3f} & {r['sigma_star']:.3f} & "
                    f"{r['Mstar_1e21']:.1f} & {r['Mstar_1e24']:.1f} & {sup} & {_esc(r['n_conv'])} & {int(r['n_draws'])} \\\\")
    write_tex("technologies", "Technologies Used to Invert Model Choices", "tab:m3_technologies", "lcccccccll",
              [r"Technology & $\alpha$ & $\beta$ & $a$ & $\sigma^*$ & $M^*(10^{21})$ & $M^*(10^{24})$ & Design $M$ & $N$ & $B$ \\"],
              body,
              "Point estimates of the technologies used to compute $w=\\varepsilon_N/\\varepsilon_D$. $a=\\beta/(\\alpha+\\beta)$; "
              "$\\sigma^*=2/(2+\\alpha+\\beta)$; $M^*(C)=D^*/N^*$ is the training-only compute-optimal tokens per parameter at "
              "$C$ FLOP. Design $M$ is the range of $D/N$ in the sweep that identifies the technology. $N$: parameter-count "
              "convention (total, non-embedding, or OLMo's count excluding the input embedding). $B$: bootstrap draws used for "
              "the bands (0: literature point estimate). Chinchilla refit and Meta rows from module m1; Farseer, Gadre et al. "
              "and OLMo ladder rows from module m2 (Huber-LSE, $\\delta=10^{-3}$). The Meta Approach-2/1 row imposes the "
              "Chinchilla outer exponent ($\\alpha=\\gamma/a$, $\\beta=\\gamma/(1-a)$) and reproduces Meta's published law "
              "$D^*=0.299\\,C^{0.537}$. The q-family row reports the inner exponents ($q=0.47$); the wedge depends only on the "
              "inner aggregator.")


# ----------------------------------------------------------------------------- Table 5
T5_FAMILIES = [("Llama-3.1", None), ("Llama-3.2", None), ("Qwen2.5", None), ("Qwen3", None),
               ("Qwen3-MoE", None), ("Gemma-2", None), ("Gemma-3", None), ("OLMo-2", None), ("SmolLM2", None),
               ("Phi-3", None), ("Pythia", ["pythia-160m", "pythia-1b", "pythia-12b"]),
               ("DeepSeek-V3", None), ("Kimi-K2", None)]


def table5(d, F):
    B = d[(d["sample"] == "B")].copy()
    rel = F.set_index("uid")
    rows = []
    for fam, only in T5_FAMILIES:
        g = B[B["family"] == fam].sort_values("N")
        if only:
            g = g[g["model"].isin(only)]
        for _, r in g.iterrows():
            rr = rel.loc[r["uid"]] if r["uid"] in rel.index else None
            rows.append(dict(family=fam, model=r["model"], year=r["year"], N=r["N"], N_total=r["N_total"], D=r["D"], M=r["M"],
                             moe=r["moe"], distilled=r["distilled"], w=r["w_chin"], w_lo=r["wlo_chin"], w_hi=r["whi_chin"],
                             TD=3 * (r["w_chin"] - 1), TD_lo=3 * (r["wlo_chin"] - 1), TD_hi=3 * (r["whi_chin"] - 1),
                             T=3 * r["D"] * (r["w_chin"] - 1), CE=r["CE_chin"], band_lo=r["band_lo"], band_hi=r["band_hi"],
                             w_meta=r["w_meta_a2"], w_farseer=r["w_farseer"],
                             wrel=np.nan if rr is None else rr["wrel_chin"],
                             TDrel=np.nan if rr is None else rr["TDrel"],
                             flagship=False if rr is None else bool(rr["flagship"]),
                             Mx_chin=r["Mx_chin"], Mx_farseer=r["Mx_farseer_emb"]))
    T5 = pd.DataFrame(rows)
    T5.to_csv(os.path.join(TABLES, P + "table5.csv"), index=False)
    body, last = [], None
    for _, r in T5.iterrows():
        if last is not None and r["family"] != last:
            body.append(r"\addlinespace")
        last = r["family"]
        name = _esc(r["model"].replace("-Base", "").replace("-pt", "").replace("Meta-", "").replace("-hf", ""))
        mark = ("$^{\\dagger}$" if r["distilled"] else "") + ("$^{\\ddagger}$" if r["moe"] else "")
        mstar = "" if not (r["Mx_chin"] or r["Mx_farseer"]) else ("$^{e}$" if r["Mx_farseer"] else "$^{c}$")
        wrel = "flagship" if r["flagship"] else f2(r["wrel"])
        body.append(f"{name}{mark} & {nfmt(r['N'])} & {dfmt(r['D'])} & {mfmt(r['M'])}{mstar} & "
                    f"{f2(r['w'])} [{f2(r['w_lo'])}, {f2(r['w_hi'])}] & {f2(r['TD'], 1)} & {r['T'] / 1e12:.0f} & "
                    f"{f2(r['CE'])} & [{f2(r['band_lo'])}, {f2(r['band_hi'])}] & {f2(r['w_meta'])} & {wrel} \\\\")
    write_tex("table5", "Revealed Inference Demand by Family and Model Size", "tab:m3_table5", "lcccccccccc",
              [r" & $N$ & $D$ & $M$ & $\hat w$ [95\% CI] & $\hat T/D$ & $\hat T$ & CE & PI band & $w$, Meta & $w^{rel}$ \\",
               r" & (B) & (T) & & reference & & (T) & & for $w$ & law & (GNR) \\"],
              body,
              "Open-weight pretrained models (Sample B: documented token counts; $N$ = exact Hugging Face parameter count; "
              "for mixture-of-experts models$^{\\ddagger}$, $N$ is the number of active parameters). "
              "$\\hat w=\\varepsilon_N/\\varepsilon_D$ under the reference technology (Chinchilla refit on the Besiroglu sample, "
              "module m1), with a 95\\% pairs-bootstrap interval (400 draws). $\\hat T/D=3(\\hat w-1)$ is lifetime inference "
              "tokens per training token (model\\_spec Proposition 4) and $\\hat T$ the implied lifetime inference tokens "
              "(trillions); negative values are not inference demand. CE $=C_{\\min}(L)/(6ND)$ is Farrell cost efficiency "
              "relative to the training-only frontier. PI band: union of the 95\\% intervals (or points) of the Chinchilla "
              "refit, Besiroglu et al., Hoffmann et al.\\ (TeX), Farseer (two parameter conventions) and Gadre et al.\\ "
              "RefinedWeb technologies, plus the lab's own technology (Meta: Llama~3 IsoFLOP law and primal; AI2: OLMo ladder). "
              "$w$, Meta law: wedge under Meta's Llama~3 Approach-2/1 technology. $w^{rel}$: within-family wedge relative to "
              "the flagship under the assumption that the flagship is on its own training-optimal path (depends on "
              "$\\alpha,\\beta$ only). $^{\\dagger}$Distilled (and, for Llama~3.2, pruned) from larger models. "
              "$^{c}$/$^{e}$: $M$ outside the design range of the Chinchilla sweep ($M\\le341$) / also of the Farseer "
              "sweep with $N$ including embeddings ($M\\le1{,}227$).", size=r"\scriptsize")
    return T5


# ----------------------------------------------------------------------------- families and over-identification
def family_tables(F, S, over):
    F.to_csv(os.path.join(TABLES, P + "family_members.csv"), index=False)
    S.to_csv(os.path.join(TABLES, P + "family_summary.csv"), index=False)
    dt = pd.DataFrame(over["demand_tests"])
    dt.to_csv(os.path.join(TABLES, P + "overid_demand.csv"), index=False)
    pd.DataFrame(over["siblings_wrel_lt1"]).to_csv(os.path.join(TABLES, P + "overid_siblings_wrel_lt1.csv"), index=False)
    fams = ["Llama-2", "Llama-3.1", "Qwen2.5", "Qwen3", "Gemma-2", "Gemma-3", "OLMo-2", "SmolLM2", "Phi-3", "Pythia",
            "Cerebras-GPT", "BLOOM", "OPT"]
    body = []
    for f in fams:
        if f not in set(S["family"]):
            continue
        r = S.set_index("family").loc[f]
        body.append(f"{_esc(f)} & {int(r['n'])} & {_esc(r['flagship'])} & {mfmt(r['M_f'])} & {f2(r['w_flag_chin'])} & "
                    f"[{f2(r['flag_band_lo'])}, {f2(r['flag_band_hi'])}] & {f2(r['w_flag_meta_a2'])} & "
                    f"{'yes' if r['D_varies'] else 'no'} & {f2(r['share_sib_wrel_gt1'])} & {f2(r['spearman_Trel_N'])} \\\\")
    body.append(r"\midrule")
    for _, r in dt.iterrows():
        body.append(f"\\multicolumn{{10}}{{l}}{{{_esc(r['spec'])}, {_esc(r['scope'])}: $\\hat\\eta={r['eta']:.2f}$ "
                    f"({r['se_eta']:.2f}); homogeneity $F({int(r['df_diff'])},\\cdot)={r['F_homog']:.2f}$, $p={r['p_homog']:.3f}$; "
                    f"$n={int(r['n'])}$, {int(r['families'])} families}} \\\\")
    write_tex("family", "Within-Family Revealed Preference and Over-Identification", "tab:m3_family", "lcccccccc c",
              [r"Family & $n$ & Flagship & $M_f$ & $\hat w_f$ & PI band & $w_f$, Meta & $D$ varies & Share $w^{rel}_i>1$ & "
               r"Spearman$(T^{rel}_i,N_i)$ \\"], body,
              "Families of Sample B with at least two models trained by the same developer on the same data release. "
              "The flagship is the largest model (total parameters). $\\hat w_f$: flagship wedge under the reference "
              "technology; PI band as in Table~5. Siblings' wedges relative to the flagship, $w^{rel}_i=(N_i/N_f)^{-\\alpha}"
              "(D_i/D_f)^{\\beta}$, are identified from the exponents alone if the flagship is training-optimal; they are "
              "lower bounds on $w_i$ when $w_f>1$. Bottom panel: demand model $T_i=\\kappa_f N_i^{-\\eta}$ with family "
              "effects; $\\hat\\eta$ with family-clustered standard error; the $F$ statistic tests a common $\\eta$ across "
              "families. With a common $D$ within a family, $T_i/D$ is a mechanical function of $N_i$; the size-varying-$D$ "
              "rows are the informative ones.", size=r"\scriptsize")


# ----------------------------------------------------------------------------- trends
def trends_tables(Tr, labs, regs):
    Tr.to_csv(os.path.join(TABLES, P + "trends.csv"), index=False)
    labs.to_csv(os.path.join(TABLES, P + "labs.csv"), index=False)
    regs.to_csv(os.path.join(TABLES, P + "trends_reg.csv"), index=False)
    body = []
    for grp, title in [("year", "Release year"), ("open", "Weights"), ("moe", "Architecture"), ("size", "Size class (active $N$)")]:
        body.append(rf"\multicolumn{{9}}{{l}}{{\textit{{{title}}}}} \\")
        g = Tr[Tr["group"] == grp]
        for _, r in g.iterrows():
            v = r["value"]
            v = {"True": "open" if grp == "open" else "MoE", "False": "closed/API" if grp == "open" else "dense"}.get(v, v)
            v = _esc(v).replace("<", "$<$").replace(">", "$>$")
            body.append(f"\\quad {v} & {int(r['n'])} & {mfmt(r['median_M'])} & {f2(r['median_w'])} & "
                        f"[{f2(r['q25_w'])}, {f2(r['q75_w'])}] & {f2(r['median_TD'], 1)} & {f2(r['share_w_lt1'])} & "
                        f"{f2(r['share_w_gt2'])} & [{f2(r['median_w_band_lo'])}, {f2(r['median_w_band_hi'])}] \\\\")
    rg = regs[(regs["sample"] == "main") & regs["term"].isin(["open", "moe_", "lnC"])]
    for f, g in rg.groupby("formula", sort=False):
        s = "; ".join(f"{t.replace('moe_', 'MoE').replace('lnC', 'ln C').replace('open', 'open weights')} "
                      f"{r['coef']:.2f} ({r['se']:.2f})" for t, r in g.set_index("term").iterrows())
        body.append(rf"\multicolumn{{9}}{{l}}{{\footnotesize $\ln\hat w$ on year effects + {_esc(s)}; $n={int(g['n'].iloc[0])}$, "
                    rf"{int(g['clusters'].iloc[0])} developer clusters}} \\")
    write_tex("trends", "The Wedge over Time and across Model Types", "tab:m3_trends", "lcccccccc",
              [r" & $n$ & median $M$ & median $\hat w$ & IQR & median $\hat T/D$ & Share $\hat w<1$ & Share $\hat w>2$ & "
               r"median PI band \\"], body,
              "Production-scale pretrained language models ($6ND\\ge10^{21}$ FLOP) released 2019--2026: Sample A (Epoch "
              "AI all-models database, open and closed, $N$ and $D$ not graded `Speculative') merged with Sample B "
              "(verified open-weight models). Code-specialised models excluded. $\\hat w$ under the reference technology. "
              "Bottom rows: OLS of $\\ln\\hat w$ with standard errors clustered by developer (parent organization; "
              "e.g.\\ Google units merged). Conditional on $\\ln C$, "
              "$\\ln w=\\tfrac{\\alpha+\\beta}{2}\\ln(M/M^*(C))$, so the regression compares tokens per parameter at a "
              "given compute budget.", size=r"\scriptsize")


# ----------------------------------------------------------------------------- validation
def validation_table(V):
    V.to_csv(os.path.join(TABLES, P + "validation.csv"), index=False)
    specs = [("uncond", r"$\ln\hat w$, no controls"), ("cond_N", r"$\ln\hat w$ $|$ $\ln N$, year, age"),
             ("cond_C", r"$\ln M$ $|$ $\ln C$, year, age"), ("cond_C_lab", r"$\ln M$ $|$ $\ln C$, year, age, developer")]
    outs = ["ln_dlall", "ln_dl30", "ln_likes", "or_listed_i", "ln_prov", "ln_votes"]
    head = [r" & All-time & 30-day & HF & OpenRouter & OpenRouter & LMArena \\",
            r" & downloads & downloads & likes & listed & providers & votes \\"]
    body = []
    for sk, lab in specs:
        c, s = [], []
        for o in outs:
            r = V[(V["outcome"] == o) & (V["spec"] == sk)]
            if len(r):
                r = r.iloc[0]
                st = "***" if r["p"] < 0.01 else ("**" if r["p"] < 0.05 else ("*" if r["p"] < 0.1 else ""))
                c.append(f"{r['coef']:.2f}{st}")
                s.append(f"({r['se']:.2f})")
            else:
                c.append("")
                s.append("")
        body.append(f"{lab} & " + " & ".join(c) + r" \\")
        body.append(" & " + " & ".join(s) + r" \\")
    n = [int(V[(V["outcome"] == o) & (V["spec"] == "cond_C")]["n"].iloc[0]) if len(V[(V["outcome"] == o) & (V["spec"] == "cond_C")]) else 0 for o in outs]
    body.append(r"\midrule")
    body.append("Models ($\\ln M$ $|$ $\\ln C$) & " + " & ".join(str(x) for x in n) + r" \\")
    write_tex("validation", "Does Revealed Inference Demand Predict Usage?", "tab:m3_validation", "lcccccc", head, body,
              "Each cell is the coefficient on the over-training measure in a separate OLS regression; standard errors "
              "clustered by developer in parentheses (*** $p<0.01$, ** $p<0.05$, * $p<0.1$). Sample B models with Hugging "
              "Face metadata (downloads summed over the base repository and the developer's official post-trained releases; "
              "snapshot 2026-09-24). OpenRouter: listed in the 2026-09-23 models snapshot (linear probability model) and "
              "number of distinct providers serving the model. LMArena: maximum cumulative vote count of the matched "
              "post-trained release in the text arena (CC-BY-4.0 leaderboard dataset). Year: release-year effects; age: "
              "log months since release. Because $\\ln w$ is a function of $(\\ln N,\\ln D)$, the unconditional correlation "
              "confounds size; conditional on compute $C$, a higher $M$ means a smaller, cheaper-to-serve model that has "
              "\\emph{higher} loss than the compute-optimal one, so a positive coefficient cannot come from quality.",
              size=r"\scriptsize")


# ----------------------------------------------------------------------------- aggregate
def aggregate_table(Ag, Ab):
    Ag.to_csv(os.path.join(TABLES, P + "aggregate_by_tech.csv"), index=False)
    Ab.to_csv(os.path.join(TABLES, P + "aggregate.csv"), index=False)
    body = []
    for _, r in Ab.iterrows():
        m, e = f"{r['C_total']:.1e}".split("e")
        body.append(f"{r['year']} & {int(r['n'])} & ${m}\\times10^{{{int(e)}}}$ & {f2(r['median_w_ref'])} & {f2(r['multiple_ref'])} "
                    f"[{f2(r['multiple_ref_lo'])}, {f2(r['multiple_ref_hi'])}] & [{f2(r['band_lo'])}, {f2(r['band_hi'])}] & "
                    f"{f2(r['multiple_trunc_ref'])} & [{f2(r['band_trunc_lo'])}, {f2(r['band_trunc_hi'])}] \\\\")
    write_tex("aggregate", "Implied Lifetime Inference Compute of the Open-Weight Ecosystem", "tab:m3_aggregate", "lccccccc",
              [r"Release year & $n$ & $\sum 6ND$ & median $\hat w$ & $\sum 2NT/\sum 6ND$ & PI band & Truncated & PI band \\"],
              body,
              "Open-weight production-scale models ($6ND\\ge10^{21}$ FLOP; Samples A and B). The multiple $\\sum_i 2N_iT_i/\\sum_i 6N_iD_i="
              "\\sum_i(w_i-1)C_i/\\sum_iC_i$ is the compute-weighted mean of $w-1$: planned lifetime inference compute "
              "as a multiple of training compute. Reference technology with 95\\% bootstrap interval; PI band = range over "
              "the Chinchilla refit, Besiroglu et al., Hoffmann et al., Farseer (incl.\\ embeddings), Gadre et al.\\ "
              "RefinedWeb and Meta's law, including their bootstrap intervals. Truncated: $T_i$ set to zero when $w_i<1$ "
              "(under-training is not negative demand). The multiple is dominated by the largest models, whose wedges "
              "are the most extrapolated.", size=r"\scriptsize")


def rivals_table(R, over, stated):
    fb = R["factor_bias"]
    tier = R["tier_reg"]
    dist = R["distill_reg"]
    eq = R["eq3"]
    inN = eq[eq["in_farseer_N"]]
    lines = [
        ("Data scarcity / Kaplan-era beliefs", "share of production-scale models with $\\hat w<1$",
         "; ".join(f"{int(r['year'])}: {r['share_lt1']:.2f}" for _, r in R["share_w_lt1_by_year"].iterrows() if r['year'] <= 2025)),
        ("Memory/hardware tier (6.5--9.5B)", "share of open models in tier; log-normal benchmark",
         f"{tier['share_in_tier']:.2f} vs {tier['share_in_tier_lognormal']:.2f}"),
        ("", "$\\Delta\\ln\\hat w$ in tier $|$ $\\ln C$, year",
         f"{tier['coef']:.2f} ({tier['se']:.2f}), $n={tier['n']}$"),
        ("Non-homotheticity (Farseer Eq.~3)", "median $\\hat w$: Chinchilla form vs Eq.~3, $N$ in Farseer range",
         f"{inN['w_farseer'].median():.2f} vs {inN['w_eq3'].median():.2f} ($n={len(inN)}$)"),
        ("", "median $\\hat w$: Chinchilla form vs Eq.~3, $N$ above Farseer range",
         f"{eq[eq['above_farseer_N']]['w_farseer'].median():.2f} vs {eq[eq['above_farseer_N']]['w_eq3'].median():.2f} "
         f"($n={int(eq['above_farseer_N'].sum())}$)"),
        ("Distillation / synthetic data", "$\\Delta\\ln\\hat w$ distilled; synthetic $|$ $\\ln C$, year",
         f"{dist['dist']['coef']:.2f} ({dist['dist']['se']:.2f}); {dist['syn']['coef']:.2f} ({dist['syn']['se']:.2f}), $n={dist['n']}$"),
    ]
    for k, v in fb.items():
        lines.append(("Factor-biased recipes (m2)" if k.startswith("DataDecide NLS") else "",
                      f"share of Sample B with $\\hat w$ / band$_{{lo}}$ above {v['ratio']:.2f} ({_esc(k).replace('>=', '$\\ge$')})",
                      f"{v['share_w_above']:.2f} / {v['share_bandlo_above']:.2f}"))
    body = [f"{_esc(a) if a else ''} & {b} & {c} \\\\" for a, b, c in lines]
    body.append(r"\midrule")
    for _, r in stated.iterrows():
        body.append(f"Stated intent: {_esc(r['case'])} & {_esc(r['intent'])} \\citep{{{r['cite']}}} & $\\hat w$ {f2(r['w_ref_min'])}--{f2(r['w_ref_max'])}; "
                    f"Meta law {f2(r['w_meta_a2_min'])}--{f2(r['w_meta_a2_max'])} \\\\")
    write_tex("rivals", "Rival Wedges and Stated-Intent Checks", "tab:m3_rivals", "p{3.6cm}p{6.2cm}p{4.6cm}",
              [r"Rival explanation & Statistic & Estimate \\"], body,
              "Production-scale sample as in Table~\\ref{tab:m3_trends} unless noted. Memory tier: open-weight models released "
              "2023--2026 with 6.5--9.5B total parameters (one 16--24~GB accelerator in 16-bit precision); the log-normal "
              "benchmark is the share implied by a normal fit to $\\log_{10}N$. Farseer Eq.~3 is the non-homothetic form of "
              "\\citet{li2025predictableb} fitted to the Farseer sweep (non-embedding $N$); `in range' means $N_{nonemb}$ "
              "within the Farseer design ($\\le6.4$B). Factor bias: the wedge ratio $\\exp(\\text{tilt range})$ across 25 DataDecide "
              "recipes (module m2) bounds how far a recipe's factor bias can move $\\hat w$; shares are of Sample B models "
              "whose reference $\\hat w$ (or lower PI bound) exceeds that ratio. Stated intent: allocation rules stated by "
              "the developers; $\\hat w$ is the reference wedge, `Meta law' the Llama~3 Approach-2/1 technology.",
              size=r"\scriptsize")
    # csv versions
    R["w_lt1"].to_csv(os.path.join(TABLES, P + "rival_w_lt1.csv"), index=False)
    R["share_w_lt1_by_year"].to_csv(os.path.join(TABLES, P + "rival_share_w_lt1.csv"), index=False)
    R["bunch_hist"].to_csv(os.path.join(TABLES, P + "rival_size_bunching.csv"), index=False)
    R["eq3"].to_csv(os.path.join(TABLES, P + "rival_farseer_eq3.csv"), index=False)
    R["eq3_Mstar_curve"].to_csv(os.path.join(TABLES, P + "rival_Mstar_curves.csv"), index=False)
    R["m2_wedge_ratios"].to_csv(os.path.join(TABLES, P + "rival_factor_bias_m2.csv"), index=False)
    pd.DataFrame([dict(k=k, **v) for k, v in R["factor_bias"].items()]).to_csv(os.path.join(TABLES, P + "rival_factor_bias_shares.csv"), index=False)
    pd.DataFrame([dict(test="memory tier", **R["tier_reg"])] +
                 [dict(test=f"distillation: {k}", **v) for k, v in R["distill_reg"].items() if k != "n"]).to_csv(
        os.path.join(TABLES, P + "rival_regressions.csv"), index=False)
    stated.to_csv(os.path.join(TABLES, P + "stated_intent.csv"), index=False)


def homothetic_tex(d, homo):
    keys = [("B:meta-llama/Meta-Llama-3-8B", "Llama-3 8B"), ("B:meta-llama/Meta-Llama-3-70B", "Llama-3 70B"),
            ("B:meta-llama/Llama-3.1-405B", "Llama-3.1 405B"), ("B:Qwen/Qwen3-8B-Base", "Qwen3 8B"),
            ("B:google/gemma-3-27b-pt", "Gemma-3 27B")]
    H = homo[homo["uid"].isin([k for k, _ in keys])].copy()
    H.to_csv(os.path.join(TABLES, P + "homothetic.csv"), index=False)
    body = []
    for Ms in sorted(H["Mstar"].unique()):
        cells = []
        for rho in sorted(H["rho"].unique()):
            vals = []
            for k, _ in keys:
                r = H[(H["uid"] == k) & (H["Mstar"] == Ms) & (H["rho"] == rho)].iloc[0]
                vals.append(f"{r['TD']:.1f}")
            cells.append(" / ".join(vals))
        body.append(f"{Ms:g} & " + " & ".join(cells) + r" \\")
    write_tex("homothetic", "Revealed $T/D$ under Homothetic CES Technologies", "tab:m3_homothetic", "lccc",
              [r"$M^*$ & $\rho=0.30$ & $\rho=0.3527$ & $\rho=0.40$ \\"], body,
              "Under $\\alpha=\\beta=\\rho$ the wedge is $w=(M/M^*)^{\\rho}$ and $T/D=3[(M/M^*)^\\rho-1]$. Entries are "
              "$T/D$ for " + ", ".join(n for _, n in keys) + " (in that order). $M^*=16$ (Porian et al.), 20 "
              "(Chinchilla rule), 24.2 (m1 CES fit), 41 (Meta's law at $3.8\\times10^{25}$), 192 (MiniCPM); "
              "$\\rho=0.3527$ is Muennighoff et al.'s tied exponent.", size=r"\scriptsize")


def data_audit(A, B, X, Bdrop):
    rows = [dict(item="Epoch language-model rows (all dates)", n=len(A) + len(X))]
    for k, v in X["reason"].str.replace(r": .*", "", regex=True).value_counts().items():
        rows.append(dict(item=f"excluded: {k}", n=int(v)))
    rows += [dict(item="Sample A kept (decoder-only pretrained, N and D recovered)", n=len(A)),
             dict(item="Sample A core (6ND >= 1e21, not speculative, not code)", n=int(A["core"].sum())),
             dict(item="Sample A matched to Sample B", n=int(A["in_B"].sum())),
             dict(item="Sample B (verified open-weight)", n=len(B)),
             dict(item="Sample B curated additions with card-verified D", n=int(B["D_card_verified"].fillna(False).sum())),
             dict(item="Sample B with exact HF parameter count", n=int((B["N_src"] == "HF safetensors (float tensors)").sum())),
             dict(item="Sample B with |log10(D/D_Epoch)| > 0.05 (documented)", n=int((B["D_vs_epoch"].abs() > 0.05).sum())),
             dict(item="Sample B dropped from ObsScaling (reasons in sampleB_dropped.csv)", n=len(Bdrop))]
    pd.DataFrame(rows).to_csv(os.path.join(TABLES, P + "data_audit.csv"), index=False)
    cols = ["model", "family", "lab", "date", "N_total", "N_active", "N", "N_src", "N_nonemb", "D", "D_src", "D_card_verified",
            "epoch_D", "D_vs_epoch", "M", "moe", "distilled", "synthetic", "code"]
    B[[c for c in cols if c in B]].to_csv(os.path.join(TABLES, P + "sampleB_inputs.csv"), index=False)
