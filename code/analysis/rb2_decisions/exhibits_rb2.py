"""exhibits_rb2.py -- paper exhibits of module rb2_decisions.

  output/tables/rb2_decisions_table2.tex       main Table 2, rebuilt around allocation decisions (R3 N3(a); R1 New 1)
  output/figures/rb2_decisions_ecdf.pdf/.png   ECDF of s over decision units (replaces Figure 4)
  output/tables/rb2_decisions_{readings,conduct,sign,trend,second_output,labown,posttrain}.tex   appendix tables
Tables: booktabs, \\footnotesize, AEA tablenotes (no threeparttable). Figures: aer_style.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import aer_style as S
import rb2common as C
import signid as SI

S.use()


def f2(x, p=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:.{p}f}".replace("-", "$-$")


def fM(x):
    return f"{x:,.0f}" if x >= 100 else (f"{x:.0f}" if x >= 10 else f"{x:.1f}")


def write(name, tex):
    with open(os.path.join(C.TABLES, f"{C.PREFIX}_{name}.tex"), "w") as f:
        f.write(tex)


def _wrap(caption, label, colspec, header, body, notes, source=None, resize=False):
    src = f"\\begin{{tablenotes}}[Source]\n{source}\n\\end{{tablenotes}}\n" if source else ""
    rb, re_ = ("\\resizebox{\\textwidth}{!}{%\n", "}") if resize else ("", "")
    return (f"\\begin{{table}}[tp]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\footnotesize\n"
            f"\\setlength{{\\tabcolsep}}{{3pt}}\n{rb}\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header}\n\\midrule\n{body}\n"
            f"\\bottomrule\n\\end{{tabular}}{re_}\n\\medskip\n\\begin{{tablenotes}}\n{notes}\n\\end{{tablenotes}}\n{src}\\end{{table}}\n")


# ============================================================================ Table 2
NICE = {"Llama-3 herd": "Llama 3 herd (8B, 70B, 3.1 405B)", "Qwen3": "Qwen3 (0.6B--14B)", "Qwen2.5": "Qwen2.5 (0.5B--72B)",
        "Granite-3.0": "Granite 3.0 (2B, 8B)", "Apertus": "Apertus (8B, 70B)", "Olmo-3": "Olmo 3 (7B, 32B)",
        "Yi": "Yi (6B, 34B)", "Llama-2": "Llama 2 (7B, 13B, 70B)", "MPT": "MPT (7B, 30B)",
        "DeepSeek-LLM": "DeepSeek LLM (7B, 67B)", "StableLM-alpha": "StableLM-Alpha (3B, 7B)",
        "Llama#1": "LLaMA (7B, 13B)", "Llama#2": "LLaMA (30B, 65B)", "OLMo-2#1": "OLMo 2 (1B, 7B)",
        "Qwen#2": "Qwen (14B, 72B)", "Qwen2#1": "Qwen2 (1.5B, 7B, 72B)", "SmolLM#1": "SmolLM (135M, 360M)"}
MODEL_NICE = {"gemma-2-27b": "Gemma 2 27B", "Falcon3-7B-Base": "Falcon3 7B", "marin-8b-base": "Marin 8B",
              "falcon-11B": "Falcon 2 11B", "SmolLM3-3B-Base": "SmolLM3 3B", "neo_7b": "MAP-Neo 7B",
              "TinyLlama_v1.1": "TinyLlama 1.1B", "btlm-3b-8k-base": "BTLM 3B", "h2o-danube-1.8b-base": "H2O-Danube 1.8B",
              "OLMo-7B-0424-hf": "OLMo 1.7 7B", "phi-1_5": "phi-1.5", "phi-2": "phi-2", "stablelm-2-1_6b": "StableLM 2 1.6B",
              "stablelm-3b-4e1t": "StableLM-3B-4E1T", "stablelm-base-alpha-7b-v2": "StableLM-Alpha 7B v2",
              "OLMo-2-1124-13B": "OLMo 2 13B", "OLMo-2-0325-32B": "OLMo 2 32B", "SmolLM2-135M": "SmolLM2 135M",
              "SmolLM2-360M": "SmolLM2 360M", "SmolLM2-1.7B": "SmolLM2 1.7B"}
SHOW_B = ["SmolLM2-135M", "SmolLM2-360M", "SmolLM2-1.7B", "OLMo-2-1124-13B", "OLMo-2-0325-32B"]
N_SHOW_C = 5                         # Panel C rule: the five singletons with the largest training compute
ID_SHORT = {"point": "pt", "lower bound": "lb", "lower bound (point if choice)": "lb$^c$"}
SIGN_FILE = os.path.join(C.ROOT, "data", "processed", "rb5_sign", "model_bounds_S1.csv")


def _sign_models(D):
    """Per-model lower end of ln M - ln M*(C) over the identified set (dlo): WP4a's S1 set (rb5_sign, decision D-3) when
    its file exists, otherwise version 3's PI-1 (rb2). w > 1 is identified at tau iff dlo > ln tau."""
    if os.path.exists(SIGN_FILE):
        S = pd.read_csv(SIGN_FILE)
        if {"uid", "dlo"} <= set(S.columns):
            return S.set_index("uid")["dlo"], "S1 (rb5_sign)"
    return D["SIGN"]["PB"].set_index("uid")["dlo"], "PI-1 (version 3; rb5_sign output not found)"


def table2(D):
    Bc, UT, U = D["Bc"], D["UT"], D["U"]["primary"]
    dlo, sign_src = _sign_models(D)
    Lb = D["L"][D["L"]["tech"].isin(D["exante"])]
    band_pt = Lb.groupby("uid")["w"].agg(["min", "max"])
    UW = D["UW"]
    ut = UT.set_index("unit")
    X = Bc.merge(U[["uid", "unit"]], on="uid")
    rows, body = [], []

    def yn(v):
        return "Y" if v else "N"

    def sid(uids):
        d = dlo.reindex(uids)
        return f"{yn(bool((d > 0).all()))}/{yn(bool((d > np.log(1.84)).all()))}"

    def emb(g):
        lo, hi = f"{g['emb_share'].min():.2f}", f"{g['emb_share'].max():.2f}"
        return lo if lo == hi else f"{lo}--{hi}"
    ncol = 12
    # Panel A: every common-budget unit, point-identified first, each group by first release
    fam = UT[UT["unit_type"] == "family"].copy()
    fam["_o"] = fam["id_class"].map({"point": 0, "lower bound (point if choice)": 1, "lower bound": 2})
    fam = fam.sort_values(["_o", "date"])
    body.append(f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{Panel A. Members of a family that share one token budget: one decision each}}}} \\\\[1pt]")
    for u, r in fam.set_index("unit").iterrows():
        mem = X[X["unit"] == u]
        mm = float(np.exp(np.sum(mem["Cmp"] * np.log(mem["M"] / mem["Mstar_ref"])) / mem["Cmp"].sum()))
        bl, bh = C.share(UW.loc[u].min()), C.share(UW.loc[u].max())
        lab = f2(r["s_lab1"]) if r["has_lab"] else "--"
        name = NICE.get(u, u) + ("$^\\dagger$" if r.get("syn_protocol", 0) else "")
        body.append(f"{name} & {r['reading']} & {ID_SHORT[r['id_class']]} & {int(mem['serve_model'].sum())}/{len(mem)} & "
                    f"{int(mem['tier'].sum())}/{len(mem)} & {emb(mem)} & {fM(mm)} & {f2(r['s_ref'])} & "
                    f"{{\\scriptsize[{f2(C.share(r['w_ref_lo']))}, {f2(C.share(r['w_ref_hi']))}]}} & {lab} & "
                    f"{{\\scriptsize[{f2(bl)}, {f2(bh)}]}} & {sid(mem['uid'])} \\\\")
        rows.append(dict(panel="A", unit=u, label=NICE.get(u, u), reading=r["reading"], id_class=r["id_class"],
                         served=f"{int(mem['serve_model'].sum())}/{len(mem)}", tier=f"{int(mem['tier'].sum())}/{len(mem)}",
                         emb_share=emb(mem), M_over_Mstar=mm, s_ref=r["s_ref"], s_ref_lo=C.share(r["w_ref_lo"]),
                         s_ref_hi=C.share(r["w_ref_hi"]), s_lab1=r["s_lab1"] if r["has_lab"] else np.nan, s_band_lo=bl,
                         s_band_hi=bh, sign_id=sid(mem["uid"]), synthetic=int(r.get("syn_protocol", 0)), w_ref=r["w_ref"],
                         C_total=r["C_total"], year=r["year"]))

    def member_row(model, panel):
        r = Bc[Bc["model"] == model].iloc[0]
        u = r["uid"]
        cls = ut.loc[u, "id_class"]
        lab = f2(C.share(r["w_lab1"])) if np.isfinite(r["w_lab1"]) else "--"
        bl, bh = C.share(band_pt.loc[u, "min"]), C.share(band_pt.loc[u, "max"])
        mm = r["M"] / r["Mstar_ref"]
        label = MODEL_NICE.get(model, model) + ("$^\\dagger$" if ut.loc[u, "syn_protocol"] else "")
        kind = "size-specific" if panel == "B" else "singleton"
        body.append(f"\\quad {label} & {kind} & {ID_SHORT[cls]} & {yn(r['serve_model'])} & {yn(r['tier'])} & "
                    f"{r['emb_share']:.2f} & {fM(mm)} & {f2(r['s_ref'])} & {{\\scriptsize[{f2(C.share(r['w_ref_lo']))}, "
                    f"{f2(C.share(r['w_ref_hi']))}]}} & {lab} & {{\\scriptsize[{f2(bl)}, {f2(bh)}]}} & {sid([u])} \\\\")
        rows.append(dict(panel=panel, unit=u, label=MODEL_NICE.get(model, model), reading=kind, id_class=cls,
                         served=yn(r["serve_model"]), tier=yn(r["tier"]), emb_share=f"{r['emb_share']:.2f}", M_over_Mstar=mm,
                         s_ref=r["s_ref"], s_ref_lo=C.share(r["w_ref_lo"]), s_ref_hi=C.share(r["w_ref_hi"]),
                         s_lab1=C.share(r["w_lab1"]) if np.isfinite(r["w_lab1"]) else np.nan, s_band_lo=bl, s_band_hi=bh,
                         sign_id=sid([u]), synthetic=int(ut.loc[u, "syn_protocol"]), w_ref=r["w_ref"], w_ref_lo=r["w_ref_lo"],
                         w_ref_hi=r["w_ref_hi"], w_lab1=r["w_lab1"], N=r["N"], D=r["D"], M=r["M"], Mstar_ref=r["Mstar_ref"],
                         C_total=r["Cmp"], year=r["year"]))
    body.append("\\addlinespace")
    body.append(f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{Panel B. Families with size-specific budgets (selected): each member is a decision}}}} \\\\[1pt]")
    for m in SHOW_B:
        member_row(m, "B")
    single = UT[UT["unit_kind"] == "singleton"].sort_values("C_total", ascending=False).head(N_SHOW_C)
    body.append("\\addlinespace")
    body.append(f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{Panel C. Single-model releases: the {N_SHOW_C} with the largest "
                f"training compute of {int((UT['unit_kind'] == 'singleton').sum())}}}}} \\\\[1pt]")
    for m in single["members"]:
        member_row(m, "C")
    # Panel D: summary over decision units
    HL = D["HL"]

    def hl(stat, tech="reference"):
        g = HL[(HL["stat"] == stat) & (HL["tech"] == tech)]
        return g.iloc[0] if len(g) else None
    LO = "lab-own (one rule) where available"
    CL = "reference, budget+trunk cluster bootstrap (ra1's conservative scheme)"
    h1, h1l, h1c = hl("decision units (primary: budget level)"), hl("decision units (primary: budget level)", LO), \
        hl("decision units (primary: budget level)", CL)
    hp, hpc = hl("W2: point-identified units"), hl("W2: point-identified units", CL)
    hlb = hl("W2: lower-bounded units")
    h56, h56l = hl("decision units (family label, version 3)"), hl("decision units (family label, version 3)", LO)
    h66, h66l = hl("decision units (members of cap-type budgets split)"), hl("decision units (members of cap-type budgets split)", LO)
    h0, h0l = hl("models (77)"), hl("models (77)", LO)
    body.append("\\addlinespace")
    body.append(f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{Panel D. Over decision units}}}} \\\\[1pt]")

    def med_row(label, a, b_, cl=None):
        body.append(f"\\multicolumn{{7}}{{@{{}}l}}{{{label}}} & {f2(a['median_s'])} & "
                    f"{{\\scriptsize[{f2(a['median_s_lo'])}, {f2(a['median_s_hi'])}]}} & "
                    f"{f2(b_['median_s']) if b_ is not None else ''} & & \\\\")
        if cl is not None:
            body.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\quad interval under the budget-plus-trunk cluster bootstrap}} & & "
                        f"{{\\scriptsize[{f2(cl['median_s_lo'])}, {f2(cl['median_s_hi'])}]}} & & & \\\\")
    med_row(f"Median $s$, {int(h1['n_units'])} decision units", h1, h1l, h1c)
    med_row(f"\\quad {int(hp['n_units'])} point-identified (pt)", hp, None, hpc)
    med_row(f"\\quad {int(hlb['n_units'])} lower bounds (lb, lb$^c$)", hlb, None)
    med_row(f"Median $s$, {int(h56['n_units'])} family-label units", h56, h56l)
    med_row(f"Median $s$, {int(h66['n_units'])} units, cap-type budgets split into members", h66, h66l)
    med_row(f"Median $s$, {int(h0['n_units'])} models", h0, h0l)
    body.append(f"\\multicolumn{{7}}{{@{{}}l}}{{Share of decision units with $w>1$}} & {f2(h1['share_w_gt1'])} & "
                f"{{\\scriptsize[{f2(h1['share_w_gt1_lo'])}, {f2(h1['share_w_gt1_hi'])}]}} & {f2(h1l['share_w_gt1'])} & & \\\\")
    # identified shares from the per-model dlo: unit identified iff every member is
    G = X.assign(dlo=X["uid"].map(dlo)).groupby("unit").agg(dlo=("dlo", "min"), Cmp=("Cmp", "sum"))
    sh = {t: ((G["dlo"] > np.log(t)).mean(), float(((G["dlo"] > np.log(t)) * G["Cmp"]).sum() / G["Cmp"].sum())) for t in (1.0, 1.84)}
    body.append(f"\\multicolumn{{{ncol - 1}}}{{@{{}}l}}{{Share of decision units with $w>1$ identified, $\\tau=1$ and $\\tau=1.84$: "
                f"unweighted {f2(sh[1.0][0])} and {f2(sh[1.84][0])}; compute-weighted {f2(sh[1.0][1])} and {f2(sh[1.84][1])}}} & \\\\")
    header = ("& & & Served & Tier & & & \\multicolumn{2}{c}{Reference} & Lab-own & Band & Sign \\\\\n"
              "Decision unit & Budget & Id. & (model) & window & $N_{emb}/N$ & $M/M^*$ & $s$ & {\\scriptsize[95\\%]} & $s$ & of $s$ & id. \\\\")
    notes = (
        "A decision unit is one choice of $(N,D)$: members of a family whose token budgets are within 10 percent of each "
        "other (max/min $D\\le1.10$) form one decision; its entry is the family share $1-1/\\bar w_H=\\sum_jh_js_j$ ($h_j$: "
        "training-compute shares; Proposition~A9), and $M/M^*$ is compute-weighted. Budget: the reading of a common budget "
        "from the developers' reports, coded under a written protocol (Online Appendix): cap, the whole curated corpus binds, "
        "so the family share bounds the value of compactness from below, and a member's wedge does so only if that member "
        "would have trained longer at its own size; choice, the family share is revealed; menu, sizes set by a hardware or "
        "configuration menu, whose shadow value is part of the value of compactness, so the share is revealed; cap+menu, a "
        "lower bound; cap or choice, the reports state only the corpus or budget size, or give evidence of both. Id.: pt, point estimate of the value of "
        "compactness; lb, lower bound (a binding cap on tokens, or a member that repeated its corpus); lb$^c$, lower bound "
        "that is a point estimate if the budget was chosen. Served: this size (any post-trained variant) was offered "
        "through the developer's own per-token API or consumer product within 180 days of release; families show members "
        "served/members. Tier window: 2.4--3.3, 6.5--9.5, 11.5--14.9, 26--32.9 or 65--72.9B parameters, where a binding "
        "tier's shadow value is part of the value of compactness. $N_{emb}/N$: embedding share of parameters (range over "
        "members). $s=(w-1)/w$. Reference: Chinchilla with $\\kappa$ free, in total parameters like the models' $M$; "
        "brackets: 95 percent joint wild-bootstrap intervals of one technology, and every bracketed interval is the "
        "narrowest of the six bootstrap schemes compared (Panel D also gives the cluster bootstrap, 44 clusters). Lab-own: "
        "the lab's own expansion path with the common model-free $\\sigma^*$ (Meta: 8 bracketed IsoFLOP budgets; AI2: OLMo "
        "ladder; DeepSeek: published law; Marin: Nemotron-CC ladder); in Panel D, the reference technology where no lab-own "
        "path exists. Band: range of point estimates over the 32 technologies. Sign id.: $w>1$ identified beyond the "
        "designs with no tilt allowance ($\\tau=1$) and with $\\tau=1.84$ on $M^*$; a decision is identified only if "
        "every member is. $^\\dagger$Trained partly on teacher-generated synthetic data (the reference technology is "
        "estimated on natural text). Token counts of StableLM-Alpha corrected to 0.8T and of MPT-30B to 1.05T.")
    tex = _wrap("The Revealed Value of Compactness, by Allocation Decision", "tab:wedge",
                "@{}lllcccrrlccc@{}", header, "\n".join(body), notes,
                "Module rb2\\_decisions with the budget-level units of module rb5\\_units; every unit, every technology and "
                "the evidence for readings and serving are in the replication package.", resize=True)
    tex = f"% generated by code/analysis/rb2_decisions/exhibits_rb2.py; sign columns: {sign_src}; codes: {D['CODES']['codes_file'].iloc[0]}\n" + tex
    write("table2", tex)
    out = pd.DataFrame(rows)
    out["sign_source"] = sign_src
    for t in (1.0, 1.84):
        out.attrs[f"share_id_tau{t}"] = sh[t]
    return out, dict(sign_source=sign_src, **{f"units_id_tau{t:g}": sh[t][0] for t in (1.0, 1.84)},
                     **{f"units_id_cw_tau{t:g}": sh[t][1] for t in (1.0, 1.84)})


# ============================================================================ Figure: ECDF over decision units
def _ecdf(ax, v, **kw):
    v = np.sort(np.clip(np.asarray(v, float), -1.0, 1.0))
    y = np.arange(1, len(v) + 1) / len(v)
    ax.step(np.r_[-1.0, v], np.r_[0.0, y], where="post", **kw)


def fig_ecdf(D):
    UW, UT, Bc = D["UW"], D["UT"], D["Bc"]
    fig, axes = plt.subplots(1, 2, figsize=(S.WIDTH_FULL, 3.0), sharey=True)
    ax = axes[0]
    for i, k in enumerate(UW.columns):
        _ecdf(ax, C.share(UW[k].values), color=S.MUTED, lw=0.6, alpha=0.5, label="each technology" if i == 0 else None)
    sref = UT["s_ref"].values
    slab = UT["s_lab1"].values
    _ecdf(ax, sref, color=S.BLUE, lw=1.6, label="reference")
    _ecdf(ax, slab, color=S.ORANGE, lw=1.6, label="lab-own path, common $\\sigma^*$")
    ax.axvline(0, color=S.INK2, lw=0.6, ls=":")
    ax.set_xlim(-1, 1)
    ax.set_title(f"(a) Decision units (n = {len(UT)}), by technology", loc="left")
    # round 3 (R3 minor 13): annotation, legend and median box stacked in the empty upper-left area (s < 0, share
    # above 0.3), clear of the curves
    # [round 3, WP4b text] opaque, borderless legend boxes so that the dotted zero line does not run through the labels
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), fontsize=6.0, handlelength=1.4, frameon=True,
              framealpha=1.0, edgecolor="none", facecolor="white", borderpad=0.2)
    ax.text(0.03, 0.62, f"median $s$\nreference: {np.median(sref):.2f}\nlab-own: {np.median(slab):.2f}", fontsize=6.5,
            color=S.INK2, va="top", ha="left", transform=ax.transAxes)
    ax.set_ylabel("share of decisions")
    ax = axes[1]
    Us = D["U"]["split"]
    Xr = Bc.merge(Us, on="uid")
    sr = []
    for u, g in Xr.groupby("unit"):
        fam = g["unit_type"].iloc[0] == "family"
        om = g["Cmp"].values / g["Cmp"].values.sum()
        W = 1 / np.sum(om / g["w_ref"].values) if fam else g["w_ref"].iloc[0]
        sr.append(C.share(W))
    _ecdf(ax, Bc["s_ref"].values, color=S.MUTED, lw=1.2, label=f"models (n = {len(Bc)})")
    _ecdf(ax, sref, color=S.BLUE, lw=1.6, label=f"decision units (n = {len(UT)})")
    _ecdf(ax, np.array(sr), color=S.AQUA, lw=1.6, ls="--", label=f"cap-type budgets split (n = {len(sr)})")
    ax.axvline(0, color=S.INK2, lw=0.6, ls=":")
    ax.set_xlim(-1, 1)
    ax.set_title("(b) Reference technology, three unit definitions", loc="left")
    ax.legend(loc="upper left", fontsize=6.0, handlelength=1.4, frameon=True, framealpha=1.0, edgecolor="none",
              facecolor="white", borderpad=0.2)
    for a in axes:
        a.set_xlabel("compactness share $s=(w-1)/w$")
    axes[0].text(0.02, 0.97, "$s<-1$ ($w<0.5$) shown at $-1$", fontsize=6.0, color=S.INK2, va="top", ha="left",
                 transform=axes[0].transAxes)
    S.savefig(fig, f"{C.PREFIX}_ecdf")


# ============================================================================ appendix tables
def _tex_escape(q):
    return (q.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#").replace("\u2014", ", ")
            .replace("∼", "$\\sim$").replace("[The Pile](https://pile.eleuther.ai/)", "The Pile")
            .replace("–", "--").replace("’", "'"))


# the quote shown for each unit: the decisive one (K or C evidence, or the menu statement); default the first D quote
KEYQ = {"MPT": 4, "Yi": 3, "Qwen2#1": 2, "SmolLM#1": 4, "Apertus": 2}
READ_ORDER = ["Llama#1", "Llama#2", "Llama-2", "MPT", "StableLM-alpha", "Qwen#2", "Yi", "DeepSeek-LLM", "Llama-3 herd",
              "Qwen2#1", "SmolLM#1", "Qwen2.5", "Granite-3.0", "OLMo-2#1", "Qwen3", "Apertus", "Olmo-3"]
READ_SOURCES = {"Llama#1": "touvron2023llama", "Llama#2": "touvron2023llama", "Llama-2": "touvron2023llama2",
                "MPT": "mosaicml2023mpt7b,mosaicml2023mpt30b", "StableLM-alpha": "stability2023stablelm",
                "Qwen#2": "bai2023qwen", "Yi": "young2024yi", "DeepSeek-LLM": "bi2024deepseek",
                "Llama-3 herd": "grattafiori2024llama,meta2024llama3", "Qwen2#1": "yang2024qwen2",
                "SmolLM#1": "allal2024smollm", "Qwen2.5": "qwen2024qwen25", "Granite-3.0": "granite2024granite",
                "OLMo-2#1": "teamolmo2024olmo", "Qwen3": "yang2025qwen3", "Apertus": "apertus2025apertus",
                "Olmo-3": "olmo2025olmo3"}


def t_readings(D):
    """Appendix table: the reading of every common budget (all 17 budget-level units), one quote each (the first
    token-margin quote); D in the order of the listed members."""
    R = D["CODES"].set_index("unit")
    Bc, U = D["Bc"], D["U"]["primary"]
    X = Bc.merge(U[["uid", "unit"]], on="uid")
    ut = D["UT"].set_index("unit")
    body = []
    for u in READ_ORDER:
        r = R.loc[u]
        mem = X[X["unit"] == u].sort_values("N")
        Ds = [f"{d / 1e12:.2f}" for d in mem["D"]]
        dtxt = Ds[0] if len(set(Ds)) == 1 else ", ".join(Ds)
        qi = [i for i in range(1, 12) if f"quote_role_{i}" in r and r[f"quote_role_{i}"] == "D"]
        q = str(r[f"quote_{KEYQ.get(u, qi[0])}"])
        q = (q[:170].rsplit(" ", 1)[0] + " ...") if len(q) > 170 else q
        body.append(f"{NICE.get(u, u)} & {dtxt} & {r['reading']} & {ID_SHORT[ut.loc[u, 'id_class']]} & {f2(ut.loc[u, 's_ref'])} & "
                    f"``{_tex_escape(q)}'' \\\\[2pt]")
    header = ("Decision unit & $D$ (T) & Reading & Id. & $s_f$ & Evidence (one quote; all quotes, URLs and retrieval dates "
              "in the replication file) \\\\")
    cites = ", ".join(f"\\citet{{{READ_SOURCES[u]}}}" for u in READ_ORDER)
    notes = ("Readings, coded under a written protocol from the saved reports: cap, the budget is the developer's whole "
             "available corpus, or the corpus was repeated, or data limited the budget; choice, the budget is stated as set "
             "by cost, compute or schedule, or lies below a larger usable pool, or other models trained longer on the same "
             "data; menu, a size set by a hardware or configuration menu; cap or choice, only the corpus or budget size is "
             "stated, or evidence for both is present (SmolLM trained 600B tokens on a 252B-token corpus, a repetition "
             "that makes it a lower bound under either reading). "
             "Statements that a model was still improving are not evidence for either reading. Id.: pt, point estimate of "
             "the value of compactness; lb, lower bound; lb$^c$, lower bound that is a point estimate if the budget was "
             "chosen. $D$: the unit's token budgets in the order of the listed members; one value if shared. $s_f$: family "
             f"share under the reference technology. Sources: {cites}. Every quote is verified against the saved source.")
    write("readings", _wrap("Readings of Common Token Budgets", "tab:app-readings",
                            "@{}>{\\raggedright\\arraybackslash}p{2.5cm}lllr>{\\raggedright\\arraybackslash}p{4.8cm}@{}",
                            header, "\n".join(body), notes))


def t_conduct(D):
    CO, OP, DES = D["COND"], D["OP"], D["DES"]

    def row(lab, r):
        return (f"{lab} & {int(r['n'])} & {int(r['clusters'])} & {int(r['G_treated']) if np.isfinite(r['G_treated']) else '--'} & "
                f"{f2(r['coef'])} & ({f2(r['se_crv1'])}) & {f2(r['p_wcr'], 3)} \\\\")
    b = ["\\multicolumn{7}{@{}l}{\\textit{Panel A. Serving, outcome $\\ln\\hat w$ given $\\ln C$ and year effects}} \\\\[1pt]"]
    pick = [("model", "clean sample (77)", "developer-level code (ra2)", "Models: developer-level code"),
            ("model", "clean sample (77)", "model-level code (primary)", "Models: model-level code"),
            ("model", "clean sample (77)", "model-level code, ambiguous = 1", "Models: model-level, ambiguous sizes served"),
            ("model", "clean sample without Alibaba", "model-level", "Models without Alibaba"),
            ("decision", "decision units (primary)", "any member served (model-level code)", "Decisions: any member served"),
            ("decision", "decision units (primary)", "all members served", "Decisions: all members served"),
            ("decision", "decision units without Alibaba", "any member served", "Decisions without Alibaba")]
    for lv, sm, cd, lab in pick:
        r = CO[(CO["level"] == lv) & (CO["sample"] == sm) & (CO["code"] == cd) & (CO["outcome"] == "lnw")].iloc[0]
        b.append(row(lab, r))
    b.append("\\addlinespace")
    b.append("\\multicolumn{7}{@{}l}{\\textit{Panel B. Open-weight premium, production-scale universe, outcome $\\ln\\hat w$}} \\\\[1pt]")
    for lv, lab in [("model level (ra2 universe)", "Models"), ("decision level (common-D families once)", "Decisions"),
                    ("decision level (common-D families once), 2023+", "Decisions, 2023 and later")]:
        r = OP[(OP["level"] == lv) & (OP["outcome"] == "lnw")].iloc[0]
        b.append(row(lab, r))
    b.append("\\addlinespace")
    b.append("\\multicolumn{7}{@{}l}{\\textit{Panel C. Descriptive: median $s$ (reference)}} \\\\[1pt]")
    for lv, g, lab in [("model", "served (model-level)", "Served (model-level)"), ("model", "not served (model-level)", "Not served"),
                       ("model", "served, without Alibaba", "Served, without Alibaba"),
                       ("model", "served, inside a tier window (w - 1 upper bound)", "Served, in a tier window (upper bound)"),
                       ("model", "on-device target (card)", "On-device target"), ("model", "server or unspecified target", "Server or unspecified"),
                       ("decision", "served (any member)", "Decisions, served"), ("decision", "not served", "Decisions, not served")]:
        r = DES[(DES["level"] == lv) & (DES["group"] == g)].iloc[0]
        b.append(f"{lab} & {int(r['n'])} & {int(r['n_developers'])} & & {f2(r['median_s'])} & "
                 f"{{\\scriptsize[{f2(r['q25_s'])}, {f2(r['q75_s'])}]}} & \\\\")
    header = "& $n$ & Clusters & Treated & Coef. & (s.e.) & WCR $p$ \\\\"
    notes = ("Panels A--B: restricted wild cluster bootstrap-t by developer, Webb weights, $B=9{,}999$; CRV1 standard errors "
             "in parentheses; treated = developer clusters with the attribute (with three or fewer the $p$-value is not "
             "interpretable). Model-level serving code: this size served first-party (per-token API or consumer product) "
             "within 180 days of release; ambiguous: Meta AI ``built with Llama 3'' (8B, 70B) and 01.AI's early-access API "
             "(Yi-34B). Decision units: common-D families enter once (family share; total compute; first release year). "
             "Panel C: interquartile range in brackets; columns 2--3 give $n$ and developers.")
    write("conduct", _wrap("Conduct Tests with Model-Level Serving and Decision Units", "tab:app-conduct-rb2",
                           "@{}lrrrrll@{}", header, "\n".join(b), notes))


def t_sign(D):
    SH = D["SHR"]
    sets = [("PI-1 (bootstrap-t anchors; e from paths and published laws)", "PI-1"),
            ("PI-3 own-lab anchors (Meta, Marin, DeepSeek)", "PI-3 (own-lab anchors)"),
            ("PI-1, e widened to the path-slope 95% intervals", "PI-1, $e$ widened"),
            ("normal inputs only (e in [-1, 1])", "Normal inputs only"),
            ("PI-1 + Marin Comma anchor (5 budgets)", "PI-1 + Marin Comma (5 budgets)"),
            ("PI-4 all technologies as anchors (with bootstrap-t IsoFLOP anchors)", "PI-4 (all 32 technologies as anchors)")]
    sets = [x for x in sets if (SH["set"] == x[0]).any()]      # review addition: PI-4 row (R2 Major 4.3)
    taus = [1.0, 1.3, 1.84, 3.4, 4.4]
    b = []
    for lv, lvlab in [("model", "Models"), ("decision", "Decision units")]:
        b.append(f"\\multicolumn{{11}}{{@{{}}l}}{{\\textit{{{lvlab}: unweighted | compute-weighted}}}} \\\\[1pt]")
        for s, lab in sets:
            g = SH[(SH["set"] == s) & (SH["level"] == lv) & (SH["year"] == "all")]
            v1 = [f2(g[np.isclose(g["tau"], t)]["share_identified"].iloc[0]) for t in taus]
            v2 = [f2(g[np.isclose(g["tau"], t)]["share_identified_cw"].iloc[0]) for t in taus]
            b.append(f"{lab} & " + " & ".join(v1) + " & & " + " & ".join(v2) + " \\\\")
        b.append("\\addlinespace")
    g = SH[SH["set"].str.startswith("PI-1 (boot") & (SH["year"] != "all")]
    b.append("\\multicolumn{11}{@{}l}{\\textit{PI-1 by release year (models)}} \\\\[1pt]")
    for y in ("2023", "2024", "2025"):
        gy = g[(g["year"] == y) & (g["level"] == "model")]
        v1 = [f2(gy[np.isclose(gy["tau"], t)]["share_identified"].iloc[0]) for t in taus]
        v2 = [f2(gy[np.isclose(gy["tau"], t)]["share_identified_cw"].iloc[0]) for t in taus]
        b.append(f"{y} (n = {int(gy['n'].iloc[0])}) & " + " & ".join(v1) + " & & " + " & ".join(v2) + " \\\\")
    A = D["A"]
    an = "; ".join(f"{r['design']} {fM(r['Mstar'])} [{fM(r['Mstar_lo'])}, {fM(r['Mstar_hi'])}] at {r['C0']:.0e}"
                   for _, r in A[A["kind"] == "iso"].iterrows() if r["anchor"] != "deepseek")
    header = ("& \\multicolumn{5}{c}{Unweighted, tilt allowance $\\tau$} & & \\multicolumn{5}{c}{Compute-weighted} \\\\\n"
              "\\cmidrule(lr){2-6}\\cmidrule(lr){8-12}\n"
              "Assumption set & 1 & 1.3 & 1.84 & 3.4 & 4.4 & & 1 & 1.3 & 1.84 & 3.4 & 4.4 \\\\")
    header = header.replace("{2-6}", "{2-6}").replace("{8-12}", "{8-12}")
    notes = (f"Share of the clean sample for which $w>1$ is identified beyond the designs (Proposition~5): $\\ln M$ exceeds the "
             f"upper bound of the identified set for $\\ln M^*(C)$ by $\\ln\\tau$. $\\tau$: 1.3 tokenizer differences; 1.84 the "
             f"DataDecide tilt of 0.26 under the reference exponents; 3.4 DataDecide's own $M^*$ factor; 4.4 both. Anchors "
             f"(wild bootstrap-t, between-budget residuals, largest bracketed budget; 95 percent): {an}; DeepSeek's published "
             f"law. Path elasticity $e\\in[{D['SIGN']['e'][0]:.3f}, {D['SIGN']['e'][1]:.3f}]$ (anchors' paths and published "
             f"laws); PI-4 adds every technology's in-support $M^*$ as an anchor, with $e$ over all technologies' paths. Decision "
             f"units: a family is identified only if every member is.")
    tex = _wrap("Sign Identification by Tilt Allowance", "tab:app-sign", "@{}lrrrrrcrrrrr@{}", header, "\n".join(b), notes)
    write("sign", tex)


def t_trend(D):
    TR, MONO, LODO, PRE = D["TRall"], D["MONO"], D["LODO"], D["PRE"]
    ref = TR[TR["tech"] == "reference"].set_index("year")
    lab = TR[TR["tech"] == "lab-own (one rule) where available"].set_index("year")
    ex = TR[~TR["tech"].isin(["reference", "lab-own (one rule) where available"])]
    mono = MONO[~MONO["tech"].isin(["reference", "lab-own (one rule) where available"])]
    b = []
    for panel, cols in [("Panel A. Median $s$", [("median_s_models", "Models", "n_models"), ("median_s_units", "Decision units", "n_units")]),
                        ("Panel B. Compute-weighted $s$", [("s_agg_models", "Models", "n_models"), ("s_agg_units", "Decision units", "n_units")])]:
        b.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{panel}}}}} \\\\[1pt]")
        for col, lab_, ncol in cols:
            nrise = int(mono[f"rise_{col}"].sum())
            # review addition: in how many leave-one-developer-out samples the statistic rises in both steps
            lodo_rise = [bool(np.all(np.diff(g.sort_values("year")[col].values) > 0)) for _, g in LODO.groupby("dropped")]
            for i, y in enumerate((2023, 2024, 2025)):
                e = ex[ex["year"] == y][col]
                lo = LODO[LODO["year"] == y][col]
                first = f"{lab_}, {y}" if i == 0 else f"\\quad {y}"
                rise = (f"{nrise}/32" if i == 0 else
                        f"{{\\scriptsize LODO {sum(lodo_rise)}/{len(lodo_rise)}}}" if i == 1 else "")
                b.append(f"{first} & {int(ref.loc[y, ncol])} & {f2(ref.loc[y, col])} & {f2(lab.loc[y, col])} & "
                         f"[{f2(e.min())}, {f2(e.max())}] & [{f2(lo.min())}, {f2(lo.max())}] & {rise} \\\\")
        b.append("\\addlinespace")
    b.append(f"\\multicolumn{{7}}{{@{{}}l}}{{Median $M$ (models): 2023 {fM(ref.loc[2023, 'median_M'])}; 2024 "
             f"{fM(ref.loc[2024, 'median_M'])}; 2025 {fM(ref.loc[2025, 'median_M'])}}} \\\\")
    b.append("\\addlinespace")
    b.append("\\multicolumn{7}{@{}l}{\\textit{Panel C. Before 2023 (robustness): 2019--2022 open-weight production-scale universe}} \\\\[1pt]")
    b.append("& $n$ & Median $s$ & Weighted $s$ & & & \\\\")
    last = None
    for _, r in PRE.iterrows():
        smp = ("All eight models" if r["sample"].startswith("2019-2022 universe (") else
               ("OPT-175B and BLOOM-1.7B dropped (suites)" if "suite" in r["sample"] else "Contrast: clean sample 2023--2025"))
        tl = {"reference (Chinchilla, kappa free)": "reference",
              "Kaplan-believed (Kaplan path, reference curvature)": "Kaplan belief",
              "Kaplan-believed (Kaplan path, Kaplan joint-law curvature)": "Kaplan belief, $\\sigma^*=0.535$"}[r["technology"]]
        if smp != last:
            b.append(f"\\multicolumn{{7}}{{@{{}}l}}{{{smp}}} \\\\")
            last = smp
        b.append(f"\\quad {tl} & {int(r['n'])} & {f2(r['median_s'])} & {f2(r['s_agg_trunc'])} & & & \\\\")
    header = ("& $n$ & Ref. & Lab-own & 32 technologies & Leave one & Rise under \\\\\n"
              "& & & & (range) & developer out & 2023--25 \\\\")
    notes = ("Clean sample, dense open-weight base models. Ref.: reference technology; Lab-own: lab's own path with the common "
             "$\\sigma^*$ where available; 32 technologies: range over the technologies; leave one developer out: "
             "range when each developer is dropped in turn; Rise: number of the 32 technologies under which the statistic "
             "rises from 2023 to 2024 and from 2024 to 2025 (LODO: number of leave-one-developer-out samples, reference "
             "technology, in which it does). Compute-weighted $s=\\sum(w^+-1)C/\\sum w^+C$, $w^+=\\max(w,1)$. "
             "Decision units are dated by their first release. At given compute every technology ranks models by $M$, so "
             "the rise largely restates the rise in median $M$. Panel C: Kaplan belief, the developer planned with the "
             "\\citet{kaplan2020scaling} allocation $N\\propto C^{0.73}$ anchored at GPT-3 (Proposition~2(iii)), with the "
             "reference curvature or with the curvature of Kaplan's joint law ($\\sigma^*=0.535$); OPT-175B, a GPT-3 "
             "replication, lies on that path by construction.")
    write("trend", _wrap("The Trend from 2023 under One Set of Rules", "tab:app-trend", "@{}lrrrccc@{}",
                         header, "\n".join(b), notes))


def t_second(D):
    SOS, SOT = D["SOS"], D["SOT"]
    sg = SOT.set_index(["output", "form"])
    lab = {"L_bpb": "Task BPB, all tasks", "knowledge (MMLU)": "\\quad MMLU",
           "science QA (ARC, OBQA)": "\\quad ARC, OBQA",
           "commonsense (HS, PIQA, SIQA, WG, CSQA)": "\\quad Commonsense",
           "reading comprehension (BoolQ)": "\\quad BoolQ"}
    ss = SOS.set_index(["output", "form"])
    r0 = ss.loc[("L_bpb", "kappa = 1")]
    r1 = ss.loc[("L_bpb", "kappa free")]
    b = [f"C4 loss (baseline) & & & {f2(r0['median_s_C4'])} & {f2(sg.loc[('L', 'kappa = 1'), 'sigma_star'], 3)} & & & "
         f"{f2(r1['median_s_C4'])} & {f2(sg.loc[('L', 'kappa free'), 'sigma_star'], 3)} \\\\"]
    for o in ["L_bpb", "knowledge (MMLU)", "science QA (ARC, OBQA)", "commonsense (HS, PIQA, SIQA, WG, CSQA)",
              "reading comprehension (BoolQ)"]:
        cells = []
        for form in ("kappa = 1", "kappa free"):
            r = ss.loc[(o, form)]
            cells.append(f"{f2(r['median_lnratio'])} & {{\\scriptsize[{f2(r['median_lnratio_lo'])}, {f2(r['median_lnratio_hi'])}]}} & "
                         f"{f2(r['median_s_output'])} & {f2(sg.loc[(o, form), 'sigma_star'], 3)}")
        b.append(f"{lab[o]} & " + " & ".join(cells) + " \\\\")
    r = ss.loc[("L_bpb", "kappa = 1")]
    q = ss.loc[("L_bpb", "kappa free")]
    header = ("& \\multicolumn{4}{c}{Chinchilla form ($\\kappa=1$)} & \\multicolumn{4}{c}{$\\kappa$ free} \\\\\n"
              "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
              "Output & Log ratio & {\\scriptsize[95\\%]} & $s$ & $\\sigma^*$ & Log ratio & "
              "{\\scriptsize[95\\%]} & $s$ & $\\sigma^*$ \\\\")
    notes = ("Log ratio: $\\ln(w_{task}/w_{C4})$. Commonsense: HellaSwag, PIQA, SocialIQA, WinoGrande, CommonsenseQA. OLMo ladder \\citep{bhagia2024establishing}: 30 runs, the same runs scored on C4 cross-entropy and on task bits "
             "per byte (validation splits). Technologies are fitted separately to each output. $\\ln(w_{task}/w_{C4})$: "
             "median over the 77 clean-sample models (OLMo's parameter convention) of the log ratio of the wedges implied by "
             "the task-output and the C4-output technologies; by Proposition~A8(ii) it measures how a developer optimizing "
             "task performance would be misread by an econometrician using loss. Brackets: 95 percent interval of the "
             "cross-model median from a pairs bootstrap by (size, multiplier) cell with the same resample for both outputs "
             f"($B=399$). Across models (10th--90th percentile) the all-task ratio runs from {f2(r['q10_lnratio'])} to "
             f"{f2(r['q90_lnratio'])} ($\\kappa=1$) and from {f2(q['q10_lnratio'])} to {f2(q['q90_lnratio'])} ($\\kappa$ free). "
             "$s$: median share under that output's technology. BoolQ's $\\kappa=1$ fit is degenerate ($M^*(10^{21})$ in "
             "the millions).")
    write("second_output", _wrap("A Second Output on the Same Runs: Task Bits per Byte versus C4 Loss",
                                 "tab:app-second-output", "@{}lrlrrrlrr@{}", header, "\n".join(b), notes)
          .replace("\\tabcolsep}{3pt}", "\\tabcolsep}{2.5pt}"))


def t_labown(D):
    LAB = D["LAB"]
    cols = [("median_s_reference", "Reference"), ("median_s_lab_one_rule", "One rule"),
            ("median_s_lab-specific curvature", "Lab curvature"), ("median_s_Meta 10-budget planning law", "Meta 10-budget law"),
            ("median_s_OLMo ladder kappa = 1 path", "OLMo $\\kappa=1$ path"), ("median_s_ra2_labown", "v2 rule")]
    b = []
    for _, r in LAB.iterrows():
        b.append(f"{r['group']} & {int(r['n'])} & " + " & ".join(f2(r[c]) for c, _ in cols) + " \\\\")
    header = "Models & $n$ & " + " & ".join(l for _, l in cols) + " \\\\"
    notes = ("Median $s$. One rule (R2 Major 6): the lab's own expansion path plus the common model-free $\\sigma^*$. Meta: path "
             "of the 8 IsoFLOP budgets whose minimum is bracketed; AI2: OLMo ladder path ($\\kappa$ free); DeepSeek: published "
             "law; Marin: Nemotron-CC ladder path. Lab curvature: Meta 0.660, Marin 0.705, AI2 0.544 ($\\kappa$-free ladder). "
             "Meta 10-budget law: the path that reproduces Meta's published planning law (it uses two unbracketed budgets). "
             "v2 rule: the version-2 assignments (Meta and Marin paths with model-free curvature; AI2 $\\kappa$-free ladder with "
             "its own curvature; DeepSeek with the reference curvature).")
    write("labown", _wrap("Lab-Own Technologies under One Rule", "tab:app-labown", "@{}lr" + "r" * len(cols) + "@{}",
                          header, "\n".join(b), notes, resize=True))


def t_posttrain(D):
    PTD, PTE = D["PTD"], D["PTE"]
    b = ["\\multicolumn{6}{@{}l}{\\textit{Panel A. Disclosed budgets: $x_P$ = post-training / pre-training compute}} \\\\[1pt]"]
    for _, r in PTD.iterrows():
        unit = str(r["unit"]).replace("%", "\\%")
        b.append(f"\\multicolumn{{5}}{{@{{}}>{{\\raggedright\\arraybackslash}}p{{0.72\\textwidth}}}}{{{r['budget']} ({unit})}} & {f2(r['x_P'], 4)} \\\\")
    b.append("\\addlinespace")
    b.append("\\multicolumn{6}{@{}l}{\\textit{Panel B. Value of compactness net of post-training, $s=(w-1-x_P)/(w-x_P)$, reference}} \\\\[1pt]")
    for _, r in PTE.iterrows():
        xl = f2(r["x_P"], 3) if np.isfinite(r["x_P"]) else "0.135 (2025), 0.002 (before)"
        b.append(f"{xl} & {f2(r['median_s_models'])} & {f2(r.get('median_s_units', np.nan))} & {f2(r['share_mN_gt0_models'])} & "
                 f"{f2(r['s_agg_2024'])} & {f2(r['s_agg_2025'])} \\\\")
    header = ("$x_P$ & Median $s$, & Median $s$, & Share & \\multicolumn{2}{c}{Compute-weighted $s$} \\\\\n"
              "& models & decisions & $m_N>0$ & 2024 & 2025 \\\\")
    notes = ("Post-training compute scales with $N$ at given post-training tokens and rollouts and does not scale with $D$, so "
             "it enters the first-order condition for $N$ only: $w=1+x_P+m_N$ (like $\\delta>0$ in Proposition~A8). "
             "DeepSeek-V3: GPU hours from its report. T\\\"ulu 3: final SFT runs on Llama 3.1, FLOPs at 40 percent utilization "
             "over $6ND$. Olmo 3 Think 32B: GPU-days of RL (224 GPUs: 8 learner and 20 inference nodes) over pre-training "
             "GPU-days; the upper bound assigns the whole 1,024-GPU cluster to the 9 days of post-training. Panel B: "
             "$m_N=w-1-x_P$; compute-weighted $s=\\sum m_N^+C/\\sum(1+x_P+m_N^+)C$ over the clean sample of that year.")
    write("posttrain", _wrap("Post-Training Compute and the Measured Wedge", "tab:app-posttrain", "@{}lrrrrr@{}", header,
                             "\n".join(b), notes))


def write_all(D):
    rows, sh = table2(D)
    rows.to_csv(os.path.join(C.TABLES, f"{C.PREFIX}_table2.csv"), index=False)
    pd.Series(sh).to_csv(os.path.join(C.TABLES, f"{C.PREFIX}_table2_sign_shares.csv"), header=["value"])
    fig_ecdf(D)
    t_readings(D)
    t_conduct(D)
    t_sign(D)
    t_trend(D)
    t_second(D)
    t_labown(D)
    t_posttrain(D)


def headline(D):
    HL, SHR = D["HL"], D["SHR"]
    h = {}
    for _, r in HL.iterrows():
        h[f"{r['stat']} | {r['tech']}"] = {k: r[k] for k in ("n_units", "median_s", "median_s_lo", "median_s_hi", "share_w_gt1",
                                                           "share_w_gt1_lo", "share_w_gt1_hi") if k in r and pd.notna(r[k])}
    sh = SHR[SHR["year"] == "all"]
    h["sign_identified"] = {f"{r['set']} | {r['level']} | tau={r['tau']}": [r["share_identified"], r["share_identified_cw"]]
                            for _, r in sh.iterrows()}
    h["lab_own_one_rule_sigma"] = D["LOM"]["sigma"]
    h["n_served_model_level"] = int(D["SVT"]["serve_model"].sum())
    h["n_served_developer_level"] = int(D["SVT"]["serve_developer"].sum())
    h["family_readings"] = dict(zip(D["FR"]["family"], D["FR"]["reading"]))
    return h
