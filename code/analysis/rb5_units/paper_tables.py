"""paper_tables.py -- appendix tables for the round-3 text of WP4b (fix list W16, W22-W24), built only from reviewed
outputs of rb2_decisions and rb5_units (no estimation, no bootstrap).

Writes
  output/tables/rb5_units_appE_technologies.tex   Table E3 on the audited token counts (W24(a)):
        model-level columns from rb2_decisions_technologies_models.csv, decision-unit column (49 budget-level units) from
        rb2_decisions_technologies_units.csv
  output/tables/rb5_units_appE_cleaning.tex        Table E2 (W24(c)): Panel A from rb2_decisions_cleaning.csv (audited
        counts); Panel B recomputed from the per-model technology wedges in exhibits_cache.pkl['L'] with the budget-level
        unit kinds of data/processed/rb5_units/units_primary.csv
  output/tables/rb5_units_appE_cleaning_panelB.csv  the Panel B numbers
  output/tables/rb5_units_appE_labown.csv          lab-own medians without Llama 1 and 2 (W16(a)), every column of the
        lab-own table, with the rule of rb2_decisions/run.py (sensitivity paths fall back to the one-rule path)

Checks (asserted): Panel B's all-points and band shares reproduce rb2_decisions_cleaning.csv for the clean sample and
for the version-3 synthetic split; the lab-own rows for all 22 models and for the 15 models without Llama 1 and 2
reproduce rb2_decisions_labown_summary.csv and rb5_units_labown_rerun.csv.

Run: nice -n 10 .venv/bin/python code/analysis/rb5_units/paper_tables.py   (seconds; one process)
"""
from __future__ import annotations

import os
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
TAB = os.path.join(ROOT, "output", "tables")
PROC2 = os.path.join(ROOT, "data", "processed", "rb2_decisions")
PROC5 = os.path.join(ROOT, "data", "processed", "rb5_units")


def r(x, nd=2):
    """Half-up rounding of the printed value (the paper's convention)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    q = Decimal(1).scaleb(-nd)
    v = Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP)
    s = f"{v:.{nd}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def mstar_fmt(x):
    return r(x, 1) if float(Decimal(repr(float(x))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)) < 100 else f"{x:,.0f}"


# ==============================================================================================  Table E3
E3_ROWS = [
    ("Chinchilla data", [("chin_q", "$\\kappa$ free (reference)"), ("chin", "$\\kappa=1$$^\\dagger$"),
                         ("besi", "Besiroglu et al.\\ parameters$^\\dagger$"), ("hoff", "Hoffmann et al., Approach 3$^\\dagger$"),
                         ("chin_ne", "Non-embedding $N$, $\\kappa=1$")]),
    ("Farseer", [("farseer", "Non-embedding $N$, $\\kappa=1$$^\\dagger$"), ("farseer_emb", "Total $N$, $\\kappa=1$$^\\dagger$"),
                 ("farseer_q", "Non-embedding $N$, $\\kappa$ free$^\\dagger$")]),
    ("Gadre et al.", [("gadre_rw", "RefinedWeb, $\\kappa=1$$^\\dagger$"), ("gadre_c4", "C4, $\\kappa=1$$^\\dagger$"),
                      ("gadre_rp", "RedPajama, $\\kappa=1$$^\\dagger$"), ("gadre_rw_q", "RefinedWeb, $\\kappa$ free"),
                      ("gadre_c4_q", "C4, $\\kappa$ free"), ("gadre_rp_q", "RedPajama, $\\kappa$ free")]),
    ("OLMo ladder", [("olmo", "$\\kappa=1$$^\\dagger$"), ("olmo_q", "$\\kappa$ free")]),
    ("Muennighoff et al.", [("muen", "$\\kappa=1$"), ("muen_q", "$\\kappa$ free")]),
    ("Meta (Llama 3 IsoFLOPs)", [("meta_a2", "Meta's law (A2, $\\kappa=1$)$^\\dagger$"), ("meta_a3", "Primal, $\\kappa=1$$^\\dagger$"),
                                 ("meta_mf", "A2 path, MF $\\sigma^*$")]),
    ("Marin ladders", [("marin_nemotron_a3", "Nemotron-CC, primal"), ("marin_dclm_a3", "DCLM, primal"),
                       ("marin_comma_a3", "Comma, primal"), ("marin_nemotron_a2", "Nemotron-CC, A2, $\\kappa=1$"),
                       ("marin_dclm_a2", "DCLM, A2, $\\kappa=1$"), ("marin_comma_a2", "Comma, A2, $\\kappa=1$"),
                       ("marin_nemotron_mf", "Nemotron-CC, MF $\\sigma^*$"), ("marin_dclm_mf", "DCLM, MF $\\sigma^*$"),
                       ("marin_comma_mf", "Comma, MF $\\sigma^*$")]),
    ("Published laws", [("deepseek", "DeepSeek LLM, ref.\\ $\\sigma^*$"), ("minicpm", "MiniCPM")]),
]


def table_e3():
    Tm = pd.read_csv(os.path.join(TAB, "rb2_decisions_technologies_models.csv")).set_index("key")
    Tu = pd.read_csv(os.path.join(TAB, "rb2_decisions_technologies_units.csv")).set_index("key")
    keys = [k for _, rows in E3_ROWS for k, _ in rows]
    assert len(keys) == 32 and set(keys) == set(Tm.index[Tm["in_set"] & ~Tm["sensitivity"]]) == set(Tu.index)
    body = []
    for g, rows in E3_ROWS:
        body.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\textit{{{g}}}}} \\\\")
        for k, lab in rows:
            m, u = Tm.loc[k], Tu.loc[k]
            has_b = m["B"] > 0 and not np.isnan(m["med_w_lo"])
            bw = f"{{\\footnotesize [{r(m['med_w_lo'])}, {r(m['med_w_hi'])}]}}" if has_b else ""
            bs = f"{{\\footnotesize [{r(m['med_s_lo'])}, {r(m['med_s_hi'])}]}}" if has_b else ""
            body.append(f"\\quad {lab} & {r(m['sigma_star'], 3)} & {mstar_fmt(m['Mstar_1e24'])} & {r(m['med_w'])} & {bw} & "
                        f"{r(m['share_gt1'])} & {r(m['med_s'])} & {bs} & {r(u['median_s_units'])} \\\\")
        body.append("\\addlinespace[2pt]")
    body.pop()
    Ts = Tm.loc[keys]
    rng = (f"Range, 32 technologies & & {mstar_fmt(Ts['Mstar_1e24'].min())}--{mstar_fmt(Ts['Mstar_1e24'].max())} & "
           f"\\multicolumn{{2}}{{l}}{{{r(Ts['med_w'].min())}--{r(Ts['med_w'].max())}}} & "
           f"{r(Ts['share_gt1'].min())}--{r(Ts['share_gt1'].max())} & "
           f"\\multicolumn{{2}}{{l}}{{{r(Ts['med_s'].min())}--{r(Ts['med_s'].max())}}} & "
           f"{r(Tu.loc[keys, 'median_s_units'].min())}--{r(Tu.loc[keys, 'median_s_units'].max())} \\\\")
    ref = Tm.loc["chin_q"]
    tex = [
        "% generated by code/analysis/rb5_units/paper_tables.py from output/tables/rb2_decisions_technologies_models.csv",
        "% (audited token counts) and rb2_decisions_technologies_units.csv (49 budget-level decision units)",
        "\\begin{table}[tp]", "\\centering", "\\caption{The Revealed Wedge under Every Technology in the Set}",
        "\\label{tab:app-technologies}", "\\footnotesize", "\\setlength{\\tabcolsep}{1.6pt}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lccr@{\\,}lcr@{\\,}lc@{}}", "\\toprule",
        "Technology & $\\sigma^*$ & $M^*(10^{24})$ & \\multicolumn{2}{c}{Median $w$} & $w>1$ & \\multicolumn{2}{c}{Median $s$} & Units \\\\",
        "\\midrule", *body, "\\midrule", rng, "\\bottomrule", "\\end{tabular*}", "\\par\\smallskip", "\\begin{tablenotes}",
        "Clean sample (77 models, audited token counts; 49 decision units in the last column). Brackets (median $w$ and "
        "$s$): 2.5 and 97.5 percentiles over the technology's bootstrap draws, recomputing the sample statistic in every draw "
        "(one technology draw shared by all models); no bracket: published point estimate; the share with $w>1$ under the "
        f"reference has the interval [{r(ref['share_gt1_lo'])}, {r(ref['share_gt1_hi'])}]. A2: IsoFLOP-argmin (Approach 2) "
        "path; MF: model-free. $M^*(10^{24})$: compute-optimal tokens per parameter at $10^{24}$ FLOP in the technology's own "
        "convention, an extrapolation for every row. MF rows: the design's IsoFLOP path with its model-free $\\sigma^*$. "
        "Non-embedding and OLMo-convention technologies are evaluated at each model's own count. $^\\dagger$One of the 12 "
        "technologies under which an earlier version of this paper reported wedges for all 173 verified models; the other 20 "
        "were added in revision. $w>1$: share of models. Units: median $s$ over the 49 decision units, with members of a "
        "family that share one token budget entered once at their family share.",
        "\\end{tablenotes}", "\\begin{tablenotes}[Source]",
        "\\citet{hoffmann2022training}; \\citet{besiroglu2024chinchilla}; \\citet{li2025predictableb}; \\citet{gadre2024language};",
        "\\citet{bhagia2024establishing}; \\citet{muennighoff2023scaling}; \\citet{grattafiori2024llama}; \\citet{marin2026ladders};",
        "\\citet{bi2024deepseek}; \\citet{hu2024minicpm}; author's calculations.", "\\end{tablenotes}", "\\end{table}", ""]
    open(os.path.join(TAB, "rb5_units_appE_technologies.tex"), "w").write("\n".join(tex))


# ==============================================================================================  Table E2
def table_e2():
    D = pd.read_pickle(os.path.join(PROC2, "exhibits_cache.pkl"))
    L, Bc, M = D["L"], D["Bc"], D["M"]
    insets = M.loc[M["in_set"] & ~M["sensitivity"].astype(bool), "key"].tolist()
    assert len(insets) == 32
    U = pd.read_csv(os.path.join(PROC5, "units_primary.csv"))
    X = Bc.merge(U[["uid", "unit", "unit_kind", "syn_protocol"]], on="uid", how="left")
    assert len(X) == 77 and X["unit"].notna().all()
    g = L[L["tech"].isin(insets) & L["uid"].isin(X["uid"])].groupby("uid")
    allpts = (g["w"].min() > 1)
    band = (g["w_lo"].min() > 1)
    X = X.set_index("uid")
    X["allpts"] = allpts
    X["band"] = band
    C0 = pd.read_csv(os.path.join(TAB, "rb2_decisions_cleaning.csv")).set_index("step")
    h = C0.loc["h_D_undocumented"]
    assert abs(X["allpts"].mean() - h["share_allpoints_gt1"]) < 1e-12 and abs(X["band"].mean() - h["share_band_gt1"]) < 1e-12
    assert abs(np.median(X["w_ref"]) - h["median_w"]) < 1e-12
    ns = ~X["f_synthetic"].astype(bool)
    rs = C0.loc["r_synthetic"]
    assert ns.sum() == rs["n"] and abs(X.loc[ns, "allpts"].mean() - rs["share_allpoints_gt1"]) < 1e-12
    assert abs(X.loc[ns, "band"].mean() - rs["share_band_gt1"]) < 1e-12

    def row(label, m, group_col, stats_all=True):
        Y = X[m]
        w = Y["w_ref"].values
        return dict(row=label, n=int(m.sum()), groups=int(Y[group_col].nunique()) if group_col else np.nan,
                    median_w=float(np.median(w)), share_w_gt1=float(np.mean(w > 1)),
                    share_allpoints_gt1=float(Y["allpts"].mean()) if stats_all else np.nan,
                    share_band_gt1=float(Y["band"].mean()) if stats_all else np.nan,
                    median_s=float(np.median(1 - 1 / w)))
    kind = X["unit_kind"]
    serve = X["serve_model"].astype(bool)
    tier = X["tier"].astype(bool)
    ondev = X["ondevice"].astype(bool)
    # groups: families as in Panel A (column gen of clean_models, 36 in the clean sample), developers in the serving,
    # target and tier rows
    assert X["gen"].nunique() == int(h["n_families"]) and X.loc[ns, "gen"].nunique() == int(rs["n_families"])
    rows = [row("Common-budget members", kind == "common-budget", "gen"),
            row("Size-specific members", kind == "size-specific member", "gen"),
            row("Single-model releases", kind == "singleton", "gen"),
            row("Decisions of one model", kind != "common-budget", "gen"),
            row("No synthetic data, earlier flag", ns, "gen"),
            row("No synthetic data, protocol rule$^b$", ~X["syn_protocol"].astype(bool), "gen"),
            row("This size served by its developer$^a$", serve, "dev", False),
            row("Not served", ~serve, "dev", False),
            row("On-device target", ondev, "dev", False),
            row("In a memory-tier window", tier, "dev", False),
            row("Outside the tier windows", ~tier, "dev", False),
            row("$M\\le341$ (Chinchilla's range)", X["M"] <= 341, None, False),
            row("$M>341$", X["M"] > 341, None, False)]
    PB = pd.DataFrame(rows)
    PB.to_csv(os.path.join(TAB, "rb5_units_appE_cleaning_panelB.csv"), index=False)
    lab = {"start": "Verified sample", "a_dedupe": "(a) One per pretraining run", "b1_research_suite": "(b1) Research suites",
           "b2_replication": "(b2) Replications, benchmarks", "c_moe": "(c) Mixture of experts",
           "d_distilled_pruned": "(d) Distilled, pruned, derived", "e_multimodal": "(e) Multimodal",
           "f_instruct_only": "(f) No base checkpoint", "g_nontransformer": "(g) Non-transformer",
           "h_D_undocumented": "(h) $D$ undocumented"}
    A = []
    for k, t in lab.items():
        c = C0.loc[k]
        drop = "" if k == "start" else f" ($-{int(c['n_dropped'])}$)"
        tail = ": clean" if k == "h_D_undocumented" else ""
        A.append(f"{t}{drop}{tail} & {int(c['n'])} & {int(c['n_families'])} & {r(c['median_w'])} & {r(c['share_w_gt1'])} & "
                 f"{r(c['share_allpoints_gt1'])} & {r(c['share_band_gt1'])} & {r(c['median_s'])} & {r(c['median_w_kappa1'])} \\\\")
    B = []
    for _, p in PB.iterrows():
        gcol = "" if np.isnan(p["groups"]) else str(int(p["groups"]))
        B.append(f"{p['row']} & {int(p['n'])} & {gcol} & {r(p['median_w'])} & {r(p['share_w_gt1'])} & "
                 f"{r(p['share_allpoints_gt1'])} & {r(p['share_band_gt1'])} & {r(p['median_s'])} & \\\\")
    tex = ["% generated by code/analysis/rb5_units/paper_tables.py: Panel A from output/tables/rb2_decisions_cleaning.csv",
           "% (audited token counts); Panel B from data/processed/rb2_decisions/exhibits_cache.pkl (L) and",
           "% data/processed/rb5_units/units_primary.csv; numbers in output/tables/rb5_units_appE_cleaning_panelB.csv",
           "\\begin{table}[tp]", "\\centering", "\\caption{The Clean Sample: Construction and Subsamples}",
           "\\label{tab:app-cleaning}", "\\footnotesize", "\\setlength{\\tabcolsep}{2pt}",
           "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrrcccccc@{}}", "\\toprule",
           "\\multicolumn{9}{@{}l}{\\textit{Panel A. Sequential cleaning steps (reference technology: Chinchilla, $\\kappa$ free)}} \\\\[1pt]",
           " & & & Median & Share & All & Band & Median & Median $w$, \\\\",
           "Step (models dropped) & $n$ & Groups & $w$ & $w>1$ & points $>1$ & $>1$ & $s$ & $\\kappa=1$ \\\\", "\\midrule",
           *A, "\\addlinespace", "\\multicolumn{9}{@{}l}{\\textit{Panel B. Subsamples of the clean sample}} \\\\[1pt]", *B,
           "\\bottomrule", "\\end{tabular*}", ""]
    open(os.path.join(TAB, "rb5_units_appE_cleaning_body.tex"), "w").write("\n".join(tex))
    return PB


# ==============================================================================================  lab-own rows
def labown_rows():
    LM = pd.read_csv(os.path.join(TAB, "rb2_decisions_labown_models.csv"))
    ref = pd.read_csv(os.path.join(TAB, "rb2_decisions_labown_summary.csv")).set_index("group")
    rer = pd.read_csv(os.path.join(TAB, "rb5_units_labown_rerun.csv")).set_index("set")
    sens = [c for c in LM.columns if c.startswith("w_sens_")]
    llama12 = LM["model"].str.match(r"^(llama-\d+b|Llama-2-)", case=False)
    assert llama12.sum() == 7
    sh = lambda w: 1 - 1 / np.asarray(w, float)  # noqa: E731
    out = []
    for lab, m in [("all lab-own models", np.ones(len(LM), bool)), ("without Llama 1 and 2", ~llama12),
                   ("without Llama 1 and 2 and without AI2", ~llama12 & (LM["dev"] != "AI2")),
                   ("Meta, Llama 3 herd only", ~llama12 & (LM["dev"] == "Meta")), ("without AI2", LM["dev"] != "AI2"),
                   ("Meta", LM["dev"] == "Meta"), ("AI2", LM["dev"] == "AI2")]:
        g = LM[m]
        d = dict(group=lab, n=int(m.sum()), median_s_reference=float(np.median(sh(g["w_ref"]))),
                 median_s_lab_one_rule=float(np.median(sh(g["w_lab1"]))),
                 median_s_ra2_labown=float(np.median(sh(g["w_primary"]))))
        for c in sens:
            d["median_s_" + c[7:]] = float(np.median(sh(g[c].fillna(g["w_lab1"]))))
        out.append(d)
    O = pd.DataFrame(out).set_index("group")
    for grp in ("all lab-own models", "without AI2", "Meta", "AI2"):
        for c in ref.columns:
            if c == "n":
                continue
            assert abs(O.loc[grp, c] - ref.loc[grp, c]) < 1e-9, (grp, c)
    for grp in ("without Llama 1 and 2", "without Llama 1 and 2 and without AI2"):
        assert abs(O.loc[grp, "median_s_lab_one_rule"] - rer.loc[grp, "median_s_lab_one_rule"]) < 1e-9
        assert abs(O.loc[grp, "median_s_reference"] - rer.loc[grp, "median_s_reference"]) < 1e-9
    O.reset_index().to_csv(os.path.join(TAB, "rb5_units_appE_labown.csv"), index=False)
    return O


if __name__ == "__main__":
    table_e3()
    PB = table_e2()
    O = labown_rows()
    print(PB.round(3).to_string())
    print(O.round(3).to_string())
    print("wrote rb5_units_appE_technologies.tex, rb5_units_appE_cleaning_body.tex, _cleaning_panelB.csv, _labown.csv")
