"""tables_ra2.py -- paper-ready LaTeX tables (booktabs; AEA.cls tablenotes; footnotesize).
(ii) selected models; (iii) cleaning steps and technology dispersion; (iv) conduct tests and validation; plus partial
identification, cost sensitivity, family-level objects and the model-free curvature (appendix)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ra2common import TABLES, fmt, share

REF = "chin_q"


def _w(path, s):
    with open(os.path.join(TABLES, path), "w") as f:
        f.write(s)


def _num(x, p=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    return (f"{x:.{p}f}").replace("-", "$-$")


def _big(x):
    if not np.isfinite(x):
        return ""
    if x >= 1e4:
        return f"{x:,.0f}"
    if x >= 100:
        return f"{x:.0f}"
    return f"{x:.1f}"


def _sci(x):
    e = int(np.floor(np.log10(x)))
    m = x / 10 ** e
    return f"${m:.1f}\\times10^{{{e}}}$"


def _p(p, B=9999):
    if not np.isfinite(p):
        return ""
    if p <= 1.0 / (B + 1) + 1e-12:
        return f"$\\le${1/(B+1):.4f}"
    return f"{p:.3f}"


def _fit(tabular):
    """Review addition: set the tabular in a box and shrink it to \\textwidth only if it is wider (AEA text width
    135.5 mm); tables were designed to fit at \\footnotesize, so the scaling is a safety net."""
    return ("\\ifdefined\\ratabbox\\else\\newsavebox{\\ratabbox}\\fi\n\\sbox{\\ratabbox}{%\n" + tabular.rstrip() +
            "%\n}\n\\ifdim\\wd\\ratabbox>\\textwidth\\resizebox{\\textwidth}{!}{\\usebox{\\ratabbox}}"
            "\\else\\usebox{\\ratabbox}\\fi\\par\n")


def _sci_(lo, hi, p=2, big=False):
    f = _big if big else (lambda v: _num(v, p))
    return f"{{\\scriptsize[{f(lo)}, {f(hi)}]}}"


SHORT = {"Meta-Llama-3-8B": "Llama 3 8B", "Meta-Llama-3-70B": "Llama 3 70B", "Llama-3.1-405B": "Llama 3.1 405B",
         "Llama-2-7b-hf": "Llama 2 7B", "Llama-2-70b-hf": "Llama 2 70B", "Qwen2.5-7B": "Qwen2.5 7B",
         "Qwen2.5-72B": "Qwen2.5 72B", "Qwen3-0.6B-Base": "Qwen3 0.6B", "Qwen3-14B-Base": "Qwen3 14B",
         "deepseek-llm-7b-base": "DeepSeek LLM 7B", "deepseek-llm-67b-base": "DeepSeek LLM 67B",
         "OLMo-2-1124-7B": "OLMo 2 7B", "OLMo-2-0325-32B": "OLMo 2 32B", "marin-8b-base": "Marin 8B",
         "granite-3.0-8b-base": "Granite 3.0 8B", "Falcon3-7B-Base": "Falcon3 7B", "gemma-2-27b": "Gemma 2 27B",
         "SmolLM2-1.7B": "SmolLM2 1.7B", "SmolLM2-135M": "SmolLM2 135M", "Apertus-70B-2509": "Apertus 70B",
         "mpt-7b": "MPT 7B", "DeepSeek-V3-Base": "DeepSeek-V3", "Kimi-K2-Base": "Kimi K2",
         "Qwen3-235B-A22B": "Qwen3 235B-A22B", "Qwen3-30B-A3B-Base": "Qwen3 30B-A3B", "Llama-4-Scout-17B-16E": "Llama 4 Scout",
         "OLMoE-1B-7B-0924": "OLMoE 1B-7B", "gemma-3-27b-pt": "Gemma 3 27B$^d$", "Llama-3.2-1B": "Llama 3.2 1B$^p$",
         "pythia-1b": "Pythia 1B$^r$"}


TABLE_OPEN = "\\begin{table}[tp]\n\\centering\n\\caption{%s}\n\\label{%s}\n\\footnotesize\n\\setlength{\\tabcolsep}{%s}\n"


# ----------------------------------------------------------------------------- (ii) selected models
SELECTED = ["Meta-Llama-3-8B", "Meta-Llama-3-70B", "Llama-3.1-405B", "Llama-2-7b-hf", "Llama-2-70b-hf",
            "Qwen2.5-7B", "Qwen2.5-72B", "Qwen3-0.6B-Base", "Qwen3-14B-Base", "deepseek-llm-7b-base",
            "deepseek-llm-67b-base", "OLMo-2-1124-7B", "OLMo-2-0325-32B", "marin-8b-base", "granite-3.0-8b-base",
            "Falcon3-7B-Base", "gemma-2-27b", "SmolLM2-1.7B", "SmolLM2-135M", "Apertus-70B-2509", "mpt-7b"]
MOE = ["DeepSeek-V3-Base", "Kimi-K2-Base", "Qwen3-235B-A22B", "Qwen3-30B-A3B-Base", "Llama-4-Scout-17B-16E", "OLMoE-1B-7B-0924"]
EXCLUDED_SHOW = ["gemma-3-27b-pt", "Llama-3.2-1B", "pythia-1b"]


def table_models(Bw, Lt, M=None):
    b = Bw.set_index("model")
    sg = {} if M is None else dict(zip(M["key"], M["sigma_star"]))
    lines = []
    head = ("Model & $N$ (B) & $D$ (T) & $M$ & Emb. & $M/M^*$ & \\multicolumn{2}{c}{$w$ ref.\\ [95\\%]} & $s$ ref. & "
            "$w$ lab ($s$) & Band of $w$ \\\\")
    lines.append("\\begin{tabular}{@{}lrrrrrrlrcl@{}}\n\\toprule\n" + head + "\n\\midrule")
    lines.append("\\multicolumn{11}{@{}l}{\\textit{Panel A. Clean sample}} \\\\[1pt]")
    for m in SELECTED:
        if m not in b.index:
            continue
        r = b.loc[m]
        wl = f"{_num(r['w_lab'])} ({_num(r['s_lab'])})" if np.isfinite(r["w_lab"]) else "--"
        lines.append(f"{SHORT.get(m, m.replace('_', ' '))} & {r['N']/1e9:.2f} & {r['D']/1e12:.1f} & {_big(r['M'])} & "
                     f"{r['emb_share']*100:.0f}\\% & {_big(r['M_over_Mstar'])} & {_num(r['w_'+REF])} & "
                     f"{_sci_(r['wlo_'+REF], r['whi_'+REF])} & {_num(r['s_ref'])} & {wl} & "
                     f"{_sci_(r['band_lo'], r['band_hi'], big=True)} \\\\")
    lines.append("\\addlinespace\n\\multicolumn{11}{@{}l}{\\textit{Panel B. Mixture-of-experts (excluded): $N$ and $M$ total/active; "
                 "$w$ and $s$ at [total-$N$, active-$N$]}} \\\\[1pt]")
    lt = Lt.pivot(index="uid", columns="tech", values="w")
    for m in MOE:
        if m not in b.index:
            continue
        r = b.loc[m]
        wt = lt.loc[r["uid"], REF] if r["uid"] in lt.index else np.nan
        lines.append(f"{SHORT.get(m, m.replace('_', ' '))} & {r['N_total']/1e9:.0f}/{r['N']/1e9:.1f} & {r['D']/1e12:.1f} & "
                     f"{_big(r['D']/r['N_total'])}/{_big(r['M'])} & & & \\multicolumn{{2}}{{l}}{{[{_num(wt)}, {_num(r['w_'+REF])}]}} & "
                     f"\\multicolumn{{2}}{{l}}{{[{_num(share(wt))}, {_num(r['s_ref'])}]}} & \\\\")
    lines.append("\\addlinespace\n\\multicolumn{11}{@{}l}{\\textit{Panel C. Other excluded models (for reference)}} \\\\[1pt]")
    for m in EXCLUDED_SHOW:
        if m not in b.index:
            continue
        r = b.loc[m]
        lines.append(f"{SHORT.get(m, m.replace('_', ' '))} & {r['N']/1e9:.2f} & {r['D']/1e12:.1f} & {_big(r['M'])} & "
                     f"{r['emb_share']*100:.0f}\\% & {_big(r['M_over_Mstar'])} & {_num(r['w_'+REF])} & & {_num(r['s_ref'])} & & "
                     f"{_sci_(r['band_lo'], r['band_hi'], big=True)} \\\\")
    lines.append("\\bottomrule\n\\end{tabular}")
    notes = ("\\begin{tablenotes}$N$: total parameters (Panel B: total/active); $D$: documented training tokens; "
             "$M=D/N$; Emb.: embedding share of parameters (config.json). $M^*$: training-only compute-optimal tokens per "
             "parameter of the reference technology at the model's own compute. Reference: Chinchilla with the outer "
             "exponent $\\kappa$ free (inner exponents 0.424/0.431, $\\sigma^*_\\kappa=0.701$); bracket: 95 percent "
             "interval from a design-conditional wild bootstrap of the technology (399 draws); it reflects only sampling "
             "uncertainty of one technology and is not the uncertainty of the wedge. $s=(w-1)/w$: planned serving share of "
             "lifetime cost when the developer bears serving cost (negative when $w<1$). Lab-own: Meta = Meta's IsoFLOP "
             "path with Meta's model-free IsoFLOP curvature; AI2 = OLMo ladder ($\\kappa$ free); DeepSeek = DeepSeek LLM's "
             "published allocation law with the reference curvature; Marin = Marin's Nemotron-CC ladder path with its "
             "model-free curvature. Model-free curvatures are module ra1's random-effects estimates over valid IsoFLOP budgets "
             "($\\sigma^*=%.3f$ for Llama 3, %.3f for Marin Nemotron-CC); lab laws postdate Llama 1/2, OLMo 1 and Marin 8B. Band: union of 95 percent intervals (points for literature technologies) over all "
             "ex-ante technologies. Panel B: MoE models are excluded from the clean sample because a dense technology "
             "at active parameters treats them as parameter-augmenting. Panel C: $^d$ distilled, $^p$ pruned and distilled, "
             "$^r$ research suite. Lab-own $w$ with its 95 percent interval and alternatives in "
             "\\texttt{ra2\\_wedge\\_labown.csv}.\\end{tablenotes}\n"
             "\\begin{tablenotes}[Source]Module ra2\\_wedge; per-model values for every technology in "
             "\\texttt{output/tables/ra2\\_wedge\\_models.csv}.\\end{tablenotes}\n") % (
             sg.get("meta_mf", np.nan), sg.get("marin_nemotron_mf", np.nan))
    s = TABLE_OPEN % ("Revealed Serving Shares for Selected Models under the Reference, Lab-Own and All Technologies",
                      "tab:ra2-models", "2.5pt") + _fit("\n".join(lines)) + notes + "\\end{table}\n"
    _w("ra2_wedge_models_selected.tex", s)


# ----------------------------------------------------------------------------- (iii) cleaning + dispersion
LABEL_TEX = {
    "chin_q": "Chinchilla, $\\kappa$ free (reference)", "chin": "Chinchilla, $\\kappa=1$ (m3 reference)",
    "besi": "Chinchilla, Besiroglu et al.\\ published", "hoff": "Chinchilla, Hoffmann et al.\\ A3",
    "chin_ne": "Chinchilla, non-embedding $N$", "farseer": "Farseer, non-emb.\\ $N$", "farseer_emb": "Farseer, total $N$",
    "farseer_q": "Farseer, $\\kappa$ free", "gadre_rw": "Gadre et al.\\ RefinedWeb", "gadre_c4": "Gadre et al.\\ C4",
    "gadre_rp": "Gadre et al.\\ RedPajama", "gadre_rw_q": "Gadre RefinedWeb, $\\kappa$ free",
    "gadre_c4_q": "Gadre C4, $\\kappa$ free", "gadre_rp_q": "Gadre RedPajama, $\\kappa$ free", "olmo": "OLMo ladder",
    "olmo_q": "OLMo ladder, $\\kappa$ free", "muen": "Muennighoff et al.", "muen_q": "Muennighoff et al., $\\kappa$ free",
    "meta_a2": "Llama 3 law (A2/A1, $\\kappa=1$)", "meta_a3": "Llama 3 IsoFLOPs, primal",
    "meta_mf": "Llama 3 path + model-free $\\sigma^*$", "marin_comma_a3": "Marin Comma, primal",
    "marin_dclm_a3": "Marin DCLM, primal", "marin_nemotron_a3": "Marin Nemotron-CC, primal",
    "marin_comma_a2": "Marin Comma, A2/A1", "marin_dclm_a2": "Marin DCLM, A2/A1", "marin_nemotron_a2": "Marin Nemotron-CC, A2/A1",
    "marin_comma_mf": "Marin Comma, model-free $\\sigma^*$", "marin_dclm_mf": "Marin DCLM, model-free $\\sigma^*$",
    "marin_nemotron_mf": "Marin Nemotron-CC, model-free $\\sigma^*$", "deepseek": "DeepSeek LLM law + ref.\\ curvature",
    "minicpm": "MiniCPM law", "chin_cl": "Chinchilla $\\kappa=1$, 9-cluster draws", "chin_q_pairs": "Chinchilla $\\kappa$ free, pairs draws",
    "chin_nls": "Chinchilla $\\kappa=1$, NLS in levels", "chin245": "Chinchilla $\\kappa=1$, all 245 runs",
    "meta_mfraw": "Llama 3, model-free $\\sigma^*$ (raw)", "marin_comma_mfraw": "Marin Comma, model-free (raw)",
    "marin_dclm_mfraw": "Marin DCLM, model-free (raw)", "marin_nemotron_mfraw": "Marin Nemotron-CC, model-free (raw)",
    "meta_mfbc": "Llama 3, ra2 model-free (bias-corr.)", "marin_comma_mfbc": "Marin Comma, ra2 model-free (bias-corr.)",
    "marin_dclm_mfbc": "Marin DCLM, ra2 model-free (bias-corr.)", "meta_mf8": "Llama 3, 8 bracketed budgets", "marin_nemotron_mfbc": "Marin Nemotron-CC, ra2 model-free (bias-corr.)"}


def _ci(pt, lo, hi, p=2):
    if not np.isfinite(lo):
        return _num(pt, p)
    return f"{_num(pt, p)} [{_num(lo, p)}, {_num(hi, p)}]"


STEP_SHORT = {"start": "Verified sample (m3 Sample B)", "a_dedupe": "(a) One per pretraining run",
              "b1_research_suite": "(b1) Research suites", "b2_replication": "(b2) Replications, benchmarks",
              "c_moe": "(c) Mixture-of-experts", "d_distilled_pruned": "(d) Distilled, pruned, derived",
              "e_multimodal": "(e) Multimodal pretraining", "f_instruct_only": "(f) Instruction-tuned only",
              "g_nontransformer": "(g) Non-transformer", "h_D_undocumented": "(h) $D$ undocumented = clean",
              "r_synthetic": "Robustness: no synthetic data"}
LABEL_SHORT = {"chin_q": "Chinchilla, $\\kappa$ free (ref.)", "chin": "Chinchilla, $\\kappa=1$",
               "besi": "Chinchilla, Besiroglu", "hoff": "Chinchilla, Hoffmann A3", "chin_ne": "Chinchilla, non-emb.\\ $N$",
               "farseer": "Farseer, non-emb.\\ $N$", "farseer_emb": "Farseer, total $N$", "farseer_q": "Farseer, $\\kappa$ free",
               "gadre_rw": "Gadre RefinedWeb", "gadre_c4": "Gadre C4", "gadre_rp": "Gadre RedPajama",
               "gadre_rw_q": "Gadre RefinedWeb, $\\kappa$ free", "gadre_c4_q": "Gadre C4, $\\kappa$ free",
               "gadre_rp_q": "Gadre RedPajama, $\\kappa$ free", "olmo": "OLMo ladder", "olmo_q": "OLMo ladder, $\\kappa$ free",
               "muen": "Muennighoff", "muen_q": "Muennighoff, $\\kappa$ free", "meta_a2": "Llama 3 law (A2)",
               "meta_a3": "Llama 3, primal", "meta_mf": "Llama 3, model-free", "marin_nemotron_a3": "Marin Nemotron, primal",
               "marin_dclm_a3": "Marin DCLM, primal", "marin_comma_a3": "Marin Comma, primal",
               "marin_nemotron_a2": "Marin Nemotron, A2", "marin_dclm_a2": "Marin DCLM, A2", "marin_comma_a2": "Marin Comma, A2",
               "marin_nemotron_mf": "Marin Nemotron, model-free", "marin_dclm_mf": "Marin DCLM, model-free",
               "marin_comma_mf": "Marin Comma, model-free", "deepseek": "DeepSeek LLM law", "minicpm": "MiniCPM law"}


def _cis(pt, lo, hi, p=2):
    if not np.isfinite(lo):
        return _num(pt, p)
    return f"{_num(pt, p)} {_sci_(lo, hi, p)}"


def table_cleaning(CT, TD, n_ne=None, n_clean=None):
    L = ["\\begin{tabular}{@{}lrrrrrrr@{}}", "\\toprule",
         "\\multicolumn{8}{@{}l}{\\textit{Panel A. Cleaning steps (reference technology)}} \\\\[1pt]",
         " & & & Median & Share & All points & Band & Median \\\\",
         "Step & $n$ & Dropped & $w$ & $w>1$ & $>1$ & $>1$ & $s$ \\\\", "\\midrule"]
    for _, r in CT.iterrows():
        if r["step"] == "r_synthetic":
            L.append("\\addlinespace")
        L.append(f"{STEP_SHORT.get(r['step'], r['label'])} & {int(r['n'])} & {int(r['n_dropped']) if r['step'] != 'start' else ''} & {_num(r['median_w'])} & "
                 f"{_num(r['share_w_gt1'])} & {_num(r['share_allpoints_gt1'])} & {_num(r['share_band_gt1'])} & {_num(r['median_s'])} \\\\")
    L += ["\\midrule", "\\multicolumn{8}{@{}l}{\\textit{Panel B. Dispersion over the ex-ante technologies, clean sample "
          "(joint bootstrap 95\\% intervals)}} \\\\[1pt]",
          "Technology & $\\sigma^*$ & $M^*(10^{24})$ & \\multicolumn{2}{c}{Median $w$} & \\multicolumn{2}{c}{Share $w>1$} & Median $s$ \\\\", "\\midrule"]
    X = TD[TD["in_set"]].copy()
    order = ["chin_q", "chin", "besi", "hoff", "chin_ne", "farseer", "farseer_emb", "farseer_q", "gadre_rw", "gadre_c4",
             "gadre_rp", "gadre_rw_q", "gadre_c4_q", "gadre_rp_q", "olmo", "olmo_q", "muen", "muen_q", "meta_a2", "meta_a3",
             "meta_mf", "marin_nemotron_a3", "marin_dclm_a3", "marin_comma_a3", "marin_nemotron_a2", "marin_dclm_a2",
             "marin_comma_a2", "marin_nemotron_mf", "marin_dclm_mf", "marin_comma_mf", "deepseek", "minicpm"]
    X = X.set_index("key").loc[[k for k in order if k in set(X["key"])]]
    for k, r in X.iterrows():
        L.append(f"{LABEL_SHORT.get(k, LABEL_TEX.get(k, k))} & {_num(r['sigma_star'], 3)} & {_big(r['Mstar_1e24'])} & "
                 f"\\multicolumn{{2}}{{r}}{{{_cis(r['med_w'], r['med_w_lo'], r['med_w_hi'])}}} & "
                 f"\\multicolumn{{2}}{{r}}{{{_cis(r['share_gt1'], r['share_gt1_lo'], r['share_gt1_hi'])}}} & "
                 f"{_cis(r['med_s'], r['med_s_lo'], r['med_s_hi'])} \\\\")
    L += ["\\midrule", f"Range, {len(X)} technologies & & {_big(X['Mstar_1e24'].min())}--{_big(X['Mstar_1e24'].max())} & "
          f"\\multicolumn{{2}}{{r}}{{{_num(X['med_w'].min())}--{_num(X['med_w'].max())}}} & "
          f"\\multicolumn{{2}}{{r}}{{{_num(X['share_gt1'].min())}--{_num(X['share_gt1'].max())}}} & {_num(X['med_s'].min())}--{_num(X['med_s'].max())} \\\\",
          "\\bottomrule", "\\end{tabular}"]
    notes = ("\\begin{tablenotes}Panel A: sequential steps from the 173 verified open-weight base models (m3 Sample B); "
             "each row reports the sample remaining after the step; the clean sample is the last row. Reference technology: "
             "Chinchilla with $\\kappa$ free. All points $>1$: every ex-ante technology's point estimate exceeds one; band $>1$: the "
             "union of 95 percent intervals over all ex-ante technologies lies above one. Panel B: medians and shares over the clean sample; brackets are 2.5/97.5 percentiles over the "
             "technology's bootstrap draws, recomputing the sample statistic in every draw (one technology draw shared by "
             "all models); no bracket = published point estimate. Ex-ante rule: every technology estimated from "
             "final-checkpoint losses of a designed sweep (primary Huber estimator; $\\kappa=1$ and $\\kappa$ free; each "
             "corpus and parameter convention), the Approach-2 paths and model-free curvature of the IsoFLOP designs, and "
             "published lab laws (Llama 3, DeepSeek LLM, MiniCPM). Excluded ex ante: DataDecide and (Mis)Fitting, whose "
             "points come from checkpoints before the learning-rate schedule ends. Model-free rows: own IsoFLOP path with module "
             "ra1's model-free $\\sigma^*$. Robustness row (not part of the ex-ante definition): the clean sample without "
             "models trained mainly on teacher-generated synthetic data (phi-1.5, phi-2, SmolLM 1--3). Non-embedding and OLMo-convention "
             "technologies are evaluated at the model's own count from config.json (%d of %d models).\\end{tablenotes}\n"
             "\\begin{tablenotes}[Source]Module ra2\\_wedge; \\texttt{ra2\\_wedge\\_cleaning.csv}, "
             "\\texttt{ra2\\_wedge\\_technologies.csv}.\\end{tablenotes}\n") % (n_ne, n_clean)
    s = TABLE_OPEN % ("Clean Inference-Demand Sample and Dispersion across Technologies", "tab:ra2-cleaning", "2.5pt") + \
        _fit("\n".join(L)) + notes + "\\end{table}\n"
    _w("ra2_wedge_cleaning.tex", s)


# ----------------------------------------------------------------------------- (iv) conduct + validation
def table_conduct(COND, BU, HV, SA, OR):
    L = ["\\begin{tabular}{@{}llrrrrrl@{}}", "\\toprule",
         "Test & Sample & $n$ & $G$ & Coef. & (s.e.) & $p$ (WCR) & Note \\\\",
         "\\midrule", "\\multicolumn{8}{@{}l}{\\textit{Panel A. Conduct tests; outcome $\\ln w$ (reference), controls year effects and $\\ln C$}} \\\\[1pt]"]
    c = COND

    def row(lbl, smp, r, note=""):
        return (f"{lbl} & {smp} & {int(r['n'])} & {int(r['clusters'])} & {_num(r['coef'])} & ({_num(r['se_crv1'])}) & "
                f"{_p(r['p_wcr'])} & {{\\scriptsize {note}}} \\\\")
    g1 = lambda q: f"$G_1$={int(q['G_treated'])}" if np.isfinite(q.get("G_treated", np.nan)) else ""
    g0 = lambda q: f"$G_0$={int(q['G_untreated'])} closed" if np.isfinite(q.get("G_untreated", np.nan)) else ""
    r = c[(c["block"] == "(a) open vs closed") & (c["outcome"] == "lnw") & (c["spec"] == "year FE + ln C") &
          (c["test"] == "open-weight premium")].iloc[0]
    L.append(row("(a) Open weights", "universe", r, g0(r)))
    r = c[(c["block"] == "(a) open vs closed") & (c["test"] == "open-weight premium, 2023+")].iloc[0]
    L.append(row("\\quad 2023 and later", "universe", r, g0(r)))
    for test, lbl in [("serve (with ondevice)", "(b) Developer serves"), ("ondevice (with ondevice)", "\\quad On-device target"),
                      ("local (with local)", "\\quad Local target")]:
        rr = c[(c["block"] == "(b) serving vs on-device") & (c["test"] == test) & (c["outcome"] == "lnw")]
        if len(rr):
            L.append(row(lbl, "clean", rr.iloc[0], g1(rr.iloc[0])))
    for test, lbl in [("serve (with ondevice)", "\\quad Developer serves"), ("ondevice (with ondevice)", "\\quad On-device target")]:
        rr = c[(c["block"] == "(b) serving vs on-device, robustness") & (c["test"] == test) & (c["outcome"] == "lnw")]
        if len(rr):
            L.append(row(lbl, "clean + suites", rr.iloc[0], g1(rr.iloc[0])))
    for test, lbl in [("in_tier", "(c) In a tier window"), ("dist_tier", "\\quad ln distance to cap")]:
        for smp, sl in [("clean sample", "clean"), ("open-weight universe (clean)", "open universe")]:
            rr = c[(c["block"] == "(c) tiers") & (c["test"] == test) & (c["sample"] == smp)]
            if len(rr):
                q = rr.iloc[0]
                L.append(row(lbl if smp == "clean sample" else "", sl, q, f"placebo $p$ {q['p_placebo_tierstory']:.2f}"))
    pooled = BU[(BU["term"] == "pooled")]
    for _, q in pooled.iterrows():
        sh = BU[(BU["sample"] == q["sample"]) & (BU["term"] == "share_in_windows")]["estimate"].iloc[0]
        pr = BU[(BU["sample"] == q["sample"]) & (BU["term"] == "share_pred")]["estimate"].iloc[0]
        L.append(f"(c) Excess mass & {'open 2023--26' if 'open' in q['sample'] else 'clean'} & "
                 f"{int(q['n'])} & & {q['estimate']:.1f} & ({q['se']:.1f}) & & {{\\scriptsize share {sh:.2f} vs {pr:.2f}}} \\\\")
    L += ["\\addlinespace", "\\multicolumn{8}{@{}l}{\\textit{Panel B. Validation of levels}} \\\\[1pt]"]
    lab = {"hf_quantized": "quantizations", "hf_finetune": "fine-tunes", "hf_adapter": "adapters", "hf_total": "all derivatives"}
    for _, r in HV[HV["regressor"] == "lnM"].iterrows():
        k = r["outcome"].split("(1 + ")[1].rstrip(")")
        L.append(row(f"HF {lab.get(k, k)} on $\\ln M$", "clean", r))
    for _, r in HV[(HV["regressor"] == "lnTD") & (HV["outcome"] == "ln(1 + hf_total)")].iterrows():
        L.append(row("HF all on $\\ln(T/D)$", "clean", r))
    L.append("\\addlinespace")
    L.append("\\multicolumn{8}{@{}l}{\\textit{Compute-weighted planned serving share $s_{agg}$, clean sample}} \\\\[1pt]")
    L.append("Year & & $n$ & & Reference & [min, max] & & {\\scriptsize leave out 405B} \\\\")
    ref = SA[(SA["tech"] == "reference") & (SA["sample"] == "clean")]
    mn = SA[(SA["tech"] == "min over technologies")].set_index("year")
    mx = SA[(SA["tech"] == "max over technologies")].set_index("year")
    loo = SA[(SA["sample"] == "clean, leave out Llama 3.1 405B")].set_index("year")
    for _, r in ref[ref["year"].notna()].iterrows():
        y = r["year"]
        L.append(f"\\quad {int(y)} & & {int(r['n'])} & & {_num(r['s_agg'])} & {{\\scriptsize[{_num(mn.loc[y, 's_agg'])}, "
                 f"{_num(mx.loc[y, 's_agg'])}]}} & & {{\\scriptsize {_num(loo.loc[y, 's_agg'])}}} \\\\")
    L.append("\\multicolumn{4}{@{}l}{\\quad Google 2019--21: inference share of ML energy} & 0.60 & & & \\\\")
    L.append("\\multicolumn{4}{@{}l}{\\quad Facebook: inference share of AI power} & 0.70 & & & \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    notes = ("\\begin{tablenotes}$G$: developer clusters. s.e.: cluster-robust (CRV1) standard error. $p$ (WCR): restricted "
             "wild cluster bootstrap-$t$ with Webb six-point weights, 9,999 draws (floor $10^{-4}$), clustered by developer. "
             "(a) Production-scale universe (6ND $\\ge10^{21}$, 2019--2026; open-weight and closed models), confident $N$ and $D$ only (Epoch "
             "`Confident' or documented), no MoE, no non-transformer, clean-sample exclusions for verified models. "
             "(b) Serving: developer operated a commercial API or consumer product serving its own LLMs at release "
             "(coding with sources in \\texttt{ra2\\_wedge\\_coding.csv}); on-device/local: stated target in the model "
             "card. Both entered jointly; clean + suites adds the research suites and replications. $G_1$ ($G_0$): developer clusters with (without) the attribute, e.g.\\ closed models in (a); with $G_1\\le3$ the wild cluster "
             "bootstrap is unreliable \\citep{mackinnon2018wild} and the on-device and local rows are not informative. (c) Tier windows at quantized memory boundaries: 2.4--3.3B, 6.5--9.5B, "
             "11.5--14.9B, 26--32.9B, 65--72.9B parameters; placebo = share of 999 shifted tier menus (edges multiplied "
             "by $e^{u}$, $u\\sim U(-0.69,0.69)$) giving a coefficient at least as favorable to the tier story; because "
             "$\\ln w$ is a deterministic function of $(N,C)$ under one technology, tier regressors correlate with $\\ln w$ "
             "mechanically and only the placebo comparison is informative. Excess mass: polynomial (degree 5) "
             "counterfactual of the $\\log_{10}N$ histogram (0.1 bins) excluding the windows; bootstrap s.e.\\ (999). "
             "Panel B: HF = number of derivative repositories (quantizations, fine-tunes, adapters, merges) of the base "
             "and official post-trained releases, $\\ln(1+\\cdot)$. $s_{agg}=\\sum(w_i^+-1)C_i/\\sum w_i^+C_i$, "
             "$w^+=\\max(w,1)$. Benchmarks: \\citet{patterson2022carbon}, about three-fifths of Google's ML energy "
             "(one week of April 2019, 2020, 2021; research, development and production); \\citet{wu2022sustainable}, "
             "power-capacity split 10:20:70 of experimentation, training and inference at Facebook. These are realized "
             "flows of whole fleets, not planned lifetime shares of a release cohort.\\end{tablenotes}\n"
             "\\begin{tablenotes}[Source]Module ra2\\_wedge; \\texttt{ra2\\_wedge\\_conduct.csv}, "
             "\\texttt{ra2\\_wedge\\_bunching.csv}, \\texttt{ra2\\_wedge\\_validation\\_hf.csv}, "
             "\\texttt{ra2\\_wedge\\_aggregate.csv}.\\end{tablenotes}\n")
    s = TABLE_OPEN % ("Conduct Tests and Validation of Levels", "tab:ra2-conduct", "2.5pt") + _fit("\n".join(L)) + notes + "\\end{table}\n"
    _w("ra2_wedge_conduct.tex", s)


# ----------------------------------------------------------------------------- partial identification
def table_pi(PIs, PIg, anchors, S2=(np.nan, np.nan)):
    L = ["\\begin{tabular}{@{}lrrrrr@{}}", "\\toprule",
         "\\multicolumn{6}{@{}l}{\\textit{Panel A. Bounds on $M^*(C)$ at frontier compute (total-$N$ anchors)}} \\\\[1pt]",
         "Assumption set & $e$ range & $M^*(10^{23})$ & $M^*(10^{24})$ & $M^*(10^{25})$ & Anchors \\\\", "\\midrule"]
    er = {r["set"].split(" ")[0]: f"[{_num(r['e_lo'])}, {_num(r['e_hi'])}]" for _, r in PIs.iterrows()}
    for st, lab, e in [("PI-1", "PI-1: IsoFLOP anchors", er["PI-1"]), ("PI-2", "PI-2: PI-1 + nondecreasing", er["PI-2"]),
                       ("PI-3 Meta anchor", "PI-3: Meta's own anchor", er["PI-1"]),
                       ("PI-4 (total-N anchors)", "PI-4: all technologies", er["PI-4"])]:
        g = PIg[PIg["set"] == st].set_index("C")
        cells = " & ".join(f"[{_big(g.loc[C, 'Mstar_lo'])}, {_big(g.loc[C, 'Mstar_hi'])}]" for C in (1e23, 1e24, 1e25))
        L.append(f"{lab} & {e} & {cells} & {int(g['n_anchors'].iloc[0])} \\\\")
    L += ["\\midrule", "\\multicolumn{6}{@{}l}{\\textit{Panel B. Sign identification on the clean sample ($n=77$)}} \\\\[1pt]",
          "Assumption set & $e$ range & $w>1$ identified & ambiguous & median $s$ lower & median $s$ upper \\\\", "\\midrule"]
    for _, r in PIs.iterrows():
        L.append(f"{r['set'].split(' ')[0]} & [{_num(r['e_lo'])}, {_num(r['e_hi'])}] & {_num(r['share_w_gt1_identified'])} & "
                 f"{_num(r['share_ambiguous'])} & {_num(r['median_s_lo'])} & {_num(r['median_s_hi'])} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    notes = ("\\begin{tablenotes}$M^*(C)$ is never observed beyond $10^{22}$ FLOP: the largest IsoFLOP budgets are "
             "$3\\times10^{21}$ (Chinchilla), $10^{22}$ (Llama 3) and $3\\times10^{20}$ (Marin, DeepSeek LLM). Anchors: each "
             "design's Approach-2 path evaluated at its largest budget, with the 95 percent wild-bootstrap interval "
             "(DeepSeek: published law). Extrapolation: $\\ln M^*(C)=\\ln M^*(C_0)+e\\ln(C/C_0)$ with the elasticity $e=1-2a$ "
             "anywhere in the range of path slopes: PI-1 over IsoFLOP paths and published laws (DeepSeek LLM's three data "
             "sets, MiniCPM); PI-2 adds $M^*$ nondecreasing in $C$; PI-3 uses only the model's own lab's anchor where it "
             "exists (Meta, Marin, DeepSeek, AI2) and PI-1 otherwise; PI-4 uses every ex-ante technology's in-support "
             "$M^*$ and all path slopes. Llama 3's 402B/16.55T at $3.8\\times10^{25}$ FLOP is an extrapolation of Meta's "
             "law, not an observation, and is not used. Sign identification does not depend on curvature; bounds on $s$ "
             "use $1/\\sigma^*-1$ in the range of the $\\kappa$-free and model-free estimates (" + f"{S2[0]:.2f}--{S2[1]:.2f}" + "). Panel A reports "
             "total-$N$ anchors only; Panel B evaluates each anchor in its own parameter convention.\\end{tablenotes}\n"
             "\\begin{tablenotes}[Source]Module ra2\\_wedge; \\texttt{ra2\\_wedge\\_pi\\_*.csv}.\\end{tablenotes}\n")
    s = TABLE_OPEN % ("Partial Identification of the Compute-Optimal Ratio at Frontier Scale", "tab:ra2-pi", "3pt") + \
        _fit("\n".join(L)) + notes + "\\end{table}\n"
    _w("ra2_wedge_pi.tex", s)


def table_costsens(CS):
    L = ["\\begin{tabular}{@{}lrrrrrrrr@{}}", "\\toprule",
         " & & & \\multicolumn{2}{c}{Share $s$} & \\multicolumn{2}{c}{Median $T/D$} & \\multicolumn{2}{c}{Llama 3 8B $T/D$} \\\\",
         "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(l){8-9}",
         "Case & $\\delta$ & $\\eta$ & median & Llama 3 8B & $p=1$ & $p=10$ & $p=1$ & $p=10$ \\\\", "\\midrule"]
    for _, r in CS.iterrows():
        case = "A" if r["panel"].startswith("A") else "B"
        L.append(f"{case} & {_num(r['delta'], 1)} & {_num(r['eta'], 2)} & {_num(r['median_s'])} & {_num(r['s_llama3_8b'])} & "
                 f"{_num(r['median_TD_p1'], 1)} & {_num(r['median_TD_p10'], 1)} & {_num(r['TD_llama3_8b_p1'], 1)} & "
                 f"{_num(r['TD_llama3_8b_p10'], 1)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    notes = ("\\begin{tablenotes}Lifetime cost $6N^{1+\\delta}D+2pN^{\\eta}T$ in training-FLOP units at the model's own "
             "size: $\\delta$ = elasticity of training cost per FLOP in $N$ (model FLOPs utilization falling or rising "
             "with size), $\\eta$ = elasticity of serving cost per token in $N$, $p$ = price of a serving FLOP relative to a "
             "training FLOP. First-order condition: $w=(1+\\delta)+\\eta K_{inf}/K_{tr}$, so the serving/training "
             "expenditure ratio is $(w-1-\\delta)/\\eta$, $s=(w-1-\\delta)/(w-1-\\delta+\\eta)$ and $T/D=3(w-1-\\delta)/(p\\eta)$. "
             "Case A: $\\eta=1+\\delta$ ($w=(1+\\delta)(1+pT/3D)$). Case B: $\\delta=0$. $s$ does not depend on $p$. "
             "Clean sample, reference technology; $p=3$ columns are in the CSV.\\end{tablenotes}\n"
             "\\begin{tablenotes}[Source]Module ra2\\_wedge; \\texttt{ra2\\_wedge\\_costsens.csv}.\\end{tablenotes}\n")
    s = TABLE_OPEN % ("Sensitivity of the Revealed Serving Share to the Cost Side", "tab:ra2-costsens", "4pt") + \
        _fit("\n".join(L)) + notes + "\\end{table}\n"
    _w("ra2_wedge_costsens.tex", s)


def table_modelfree(mf, M=None):
    X = mf[mf["deg"] == 2]
    ra1s = {} if M is None else {k.replace("_mf", ""): v for k, v in zip(M["key"], M["sigma_star"]) if k.endswith("_mf")}
    L = ["\\begin{tabular}{@{}lrrcrrrrr@{}}", "\\toprule",
         " & & & \\multicolumn{3}{c}{This module's estimator} & $\\sigma^*$ & & \\\\",
         "\\cmidrule(lr){4-6}",
         "Design & Budgets & $C_{max}$ & $1/\\sigma^*-1$ [95\\%] & MC factor & $\\sigma^*$ corr. & (ra1, used) & $a$ & $M^*(C_{max})$ \\\\", "\\midrule"]
    lab = {"meta": "Llama 3 (Meta)", "marin_comma": "Marin, Comma", "marin_dclm": "Marin, DCLM",
           "marin_nemotron": "Marin, Nemotron-CC", "chin_iso": "Chinchilla (digitized)"}
    for _, r in X.iterrows():
        L.append(f"{lab[r['design']]} & {int(r['n_budgets'])} & {_sci(r['Cmax'])} & {_num(r['S2'], 3)} [{_num(r['S2_lo'], 3)}, "
                 f"{_num(r['S2_hi'], 3)}] & {_num(r['mc_factor'], 3)} & {_num(r['sigma_star_bc'], 3)} & "
                 f"{_num(ra1s.get(r['design'], np.nan), 3) if r['design'] in ra1s else '--'} & {_num(r['a'], 3)} & "
                 f"{_big(r['Mstar_Cmax'])} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    notes = ("\\begin{tablenotes}Model-free curvature $1/\\sigma^*-1=L_{nn}/(2|dL^*/dc|)$: per-budget quadratic fits of loss "
             "on $\\ln N$ give the argmin, the minimum and the curvature; a quadratic in $\\ln C$ through the minima gives "
             "the frontier slope; pooled as a ratio of sums over budgets. Brackets: design-conditional wild bootstrap "
             "(Rademacher, 999 draws). MC factor: true/mean estimate in a parametric simulation on the design's own grid "
             "at the estimated technology with the design's residual noise (finite-grid parabola bias and the Jensen bias "
             "of the ratio); bias-corrected $1/\\sigma^*-1$ = estimate $\\times$ factor. $a$: Approach-2 path slope. "
             "Chinchilla row: cross-check only (digitization noise). $\\sigma^*$ (ra1, used): module ra1's reviewed model-free "
             "estimate (random-effects mean over valid budgets, path-centred windows, log-cubic frontier), which the "
             "lab-own technologies use; this module's estimator and its bias correction are sensitivity rows.\\end{tablenotes}\n"
             "\\begin{tablenotes}[Source]Module ra2\\_wedge; \\texttt{ra2\\_wedge\\_modelfree\\_sigma.csv}.\\end{tablenotes}\n")
    s = TABLE_OPEN % ("Model-Free Curvature and Paths of the Lab IsoFLOP Designs", "tab:ra2-modelfree", "3pt") + \
        _fit("\n".join(L)) + notes + "\\end{table}\n"
    _w("ra2_wedge_modelfree.tex", s)


def write_all(x):
    table_models(x["Bw"], x["Lt"], x.get("M"))
    table_cleaning(x["CT"], x["TD"], int(x["Bw"].loc[x["clean"], "N_nonemb"].notna().sum()), int(x["clean"].sum()))
    table_conduct(x["COND"], x["BU"], x["HV"], x["SA"], x["OR"])
    x["PIs"]["_"] = 0
    table_pi(x["PIs"], x["PIg"], x["anchors"], x.get("S2_core", (np.nan, np.nan)))
    table_costsens(x["CS"])
    table_modelfree(x["mf"], x.get("M"))
