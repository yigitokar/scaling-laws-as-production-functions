"""run.py -- module rb5_sign: sign identification beyond the designs, rebuilt for round 3 (WP4a, items S1-S5 of
paper/notes/round3_fixlist.md; decision D-3). One command regenerates every output (deterministic; fixed seeds; CPU
only, one process, under a minute plus the review's Monte Carlo):

  nice -n 10 .venv/bin/python code/analysis/rb5_sign/run.py [--units PATH] [--units56 PATH] [--models PATH]
                                                            [--B 9999] [--mc-reps 1000] [--no-review]
                                                            [--experiment-factor X]

--units: the primary decision units (default [review]: WP4b's final file data/processed/rb5_units/units_primary.csv,
the 49 budget-level units; the same partition as rb2's units_subgroup.csv). --units56: the family-label units (robustness
row).
--models: the clean sample (default data/processed/rb2_decisions/clean_models.csv).
--experiment-factor: M9-8 only, after the experiment's Q2 is reported; marks the implied M* factor on the breakdown
figure. Never set from any FineWeb estimate before Amendment 2 is committed.

Outputs: output/tables/rb5_sign_*.csv, rb5_sign_table.tex, output/figures/rb5_sign_breakdown.{pdf,png},
data/processed/rb5_sign/ (per-model bounds, per-unit flags, bands on a grid, run log).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import rb5common as K  # noqa: E402
import data as DA  # noqa: E402
import bands as BD  # noqa: E402
import idset as ID  # noqa: E402
import variants as VA  # noqa: E402
import magnitude as MG  # noqa: E402
import exhibits as EX  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

NAMED = ["Llama-2-70b-hf", "Llama-3.1-405B", "Qwen2-72B", "falcon-180B", "deepseek-llm-67b-base"]


def rb2_pb(PBr, B):
    return PBr[["uid", "dlo", "dhi"]].merge(B[["uid", "model", "dev", "year", "gen", "N", "D", "M", "Cmp"]], on="uid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default=K.DEFAULT_UNITS)
    ap.add_argument("--units56", default=K.DEFAULT_UNITS56)
    ap.add_argument("--models", default=K.DEFAULT_MODELS)
    ap.add_argument("--B", type=int, default=K.B_BOOT)
    ap.add_argument("--mc-reps", type=int, default=1000)
    ap.add_argument("--no-review", action="store_true")
    ap.add_argument("--experiment-factor", type=float, default=None)
    a = ap.parse_args()
    t0 = time.time()
    open(os.path.join(K.PROC, "run_log.txt"), "w").close()
    K.log(f"rb5_sign: models {K.rel(a.models)}; units {K.rel(a.units)}; units (robustness) {K.rel(a.units56)}; B = {a.B}")
    # ------------------------------------------------------------------ inputs
    B = DA.load_models(a.models)
    X = DA.load_techs()
    U, miss = DA.load_units(a.units, B)
    U56, miss56 = DA.load_units(a.units56, B)
    if miss or miss56:
        K.log(f"WARNING: models without a unit: {miss} {miss56}")
    ulab = str(U["unit"].nunique())
    units = {ulab: U, "56": U56} if ulab != "56" else {ulab: U}
    n_units, n_units56 = U["unit"].nunique(), U56["unit"].nunique()
    K.log(f"{len(B)} models; {n_units} primary units; {n_units56} robustness units")
    # ------------------------------------------------------------------ S1
    K.log("S1: one path per study, simultaneous wild bootstrap-t bands (pooled residual variance)")
    P = VA.primary(X, B, B_boot=a.B)
    paths = VA.paths_list(P)
    PATHS = pd.concat([P[k].summary() for k in ("chin", "llama", "marin")], ignore_index=True)
    ds = P["ds"]
    PATHS = pd.concat([PATHS, pd.DataFrame([dict(path="DeepSeek LLM law", corpus="published law", conv="ds",
                                                 e=ds.e, Mstar_J=float(np.exp(ds.band(np.log(3e20))[0][0])),
                                                 C_J=3e20)])], ignore_index=True)
    K.tab(PATHS, "paths")
    PB = ID.bounds(B, paths, "S1")
    # ------------------------------------------------------------------ S2 variants and S1 sensitivities
    K.log("S2: PI-1 as published, t-intervals, widened slopes, normal inputs, every technology, conventions")
    V = {}
    A1, e1, PBr = VA.pi1_published(X, B)
    V["S2(a)"] = rb2_pb(PBr, B)
    Ac, ec, PBc = VA.pi1_published(X, B, include_comma=True)
    V["S2(a)+Comma"] = rb2_pb(PBc, B)
    pb_b, Tb, eb = VA.t_interval_anchors(X)
    V["S2(b)"] = ID.bounds(B, pb_b, "S2(b)")
    pb_b2, Tb2, eb2 = VA.t_interval_anchors(X, with_slope_ci=True)
    V["S2(b')"] = ID.bounds(B, pb_b2, "S2(b')")
    e_wide = (min(A1["e_lo"].min(), e1[0]), max(A1["e_hi"].max(), e1[1]))
    V["S2(c)"] = ID.bounds(B, VA.anchor_paths_from_rb2(A1, e_wide), "S2(c)")
    V["S2(d)"] = ID.bounds(B, VA.normal_inputs_new(X, P), "S2(d)")
    V["S2(d) PI-1 anchors"] = ID.bounds(B, VA.anchor_paths_from_rb2(A1, (-1.0, 1.0)), "S2(d) PI-1 anchors")
    pe, ee, Aall = VA.every_technology(X, P)
    V["S2(e)"] = ID.bounds(B, pe, "S2(e)")
    PB4, e4 = VA.pi4_published(X, B, A1, e1)
    V["S2(e) PI-4 as published"] = rb2_pb(PB4, B)
    CV, info = VA.conventions(X, B, P, B_boot=a.B)
    V["S2(f)-total"] = ID.bounds(B, CV["total"], "S2(f)-total")
    V["S2(f)-NF"] = ID.bounds(B, CV["nf"], "S2(f)-NF")
    V["S2(f)-Chinchilla N_F"] = ID.bounds(B, CV["swap"], "S2(f)-Chinchilla N_F")
    sens = {}
    for lab, kw in [("S1 HC2 studentization", dict(studentize="hc2")), ("S1 pointwise bands", dict(simultaneous=False)),
                    ("S1 Marin at the mean corpus level", dict(marin_mean=True)),
                    ("S1 bootstrap critical value alone", dict(crit_rule="boot"))]:
        Pv = VA.primary(X, B, B_boot=a.B, **kw)
        sens[lab] = Pv
        V[lab] = ID.bounds(B, VA.paths_list(Pv), lab)
    V["S1 without DeepSeek"] = ID.bounds(B, VA.paths_list(P, ("chin", "llama", "marin")), "S1 without DeepSeek")
    # [review] sources for the memo's statements on Marin's role (comparison only, not sensitivities of D-3)
    V["S1 without Marin (comparison only)"] = ID.bounds(B, VA.paths_list(P, ("chin", "llama", "ds")),
                                                        "S1 without Marin (comparison only)")
    V["S1 Marin alone (comparison only)"] = ID.bounds(B, VA.paths_list(P, ("marin",)), "S1 Marin alone (comparison only)")
    V["S1 Marin without Comma (comparison only)"] = ID.bounds(
        B, [P["chin"], P["llama"], BD.StudyPath("Marin (DCLM, Nemotron-CC)", VA.marin_frame().query("group != 'Comma'"),
                                                conv="total", B=a.B, seed=K.SEED + 23,
                                                c_range=ID.c_extent(B, ("total", "nf"))), P["ds"]],
        "S1 Marin without Comma (comparison only)")
    # ------------------------------------------------------------------ shares
    SH = [ID.shares(PB, B, units, "S1")]
    for k, v in V.items():
        SH.append(ID.shares(v.assign(set=k), B, units, k))
    SH = pd.concat(SH, ignore_index=True)
    K.tab(SH, "shares")
    # [review] comparison-only sets (one study alone, or without one) are not identified sets of the paper; Marin alone
    # identifies w < 1 for some large models, so they are excluded from the table note's 'no model is w < 1' check
    cmp_only = SH["set"].str.contains("comparison only", regex=False)
    any_lt1 = SH.loc[(SH["tau"] == 1.0) & ~cmp_only, "share_lt1"].max()
    K.log(f"largest share identified as w < 1 in any set (comparison-only sets excluded): {any_lt1}; "
          f"in the comparison-only sets: {SH.loc[(SH['tau'] == 1.0) & cmp_only, 'share_lt1'].max()}")
    VDEF = pd.DataFrame([
        dict(set="S1", definition="one path per study; joint (level, slope) simultaneous 95% bands; union", e_lo=np.nan, e_hi=np.nan),
        dict(set="S2(a)", definition="PI-1 as published: bootstrap-t levels (HC2), slopes at point-estimate hull, Comma excluded", e_lo=e1[0], e_hi=e1[1]),
        dict(set="S2(a)+Comma", definition="PI-1 with Marin Comma admitted (version 3 disclosure)", e_lo=ec[0], e_hi=ec[1]),
        dict(set="S2(b)", definition="t-interval levels (k-2 df), Comma admitted with its own slope; slopes at point-estimate hull", e_lo=eb[0], e_hi=eb[1]),
        dict(set="S2(b')", definition="as S2(b) with slopes at the hull of their t-intervals", e_lo=eb2[0], e_hi=eb2[1]),
        dict(set="S2(c)", definition="PI-1 levels, slopes at the hull of the designs' 95% slope intervals", e_lo=e_wide[0], e_hi=e_wide[1]),
        dict(set="S2(d)", definition="normal inputs only: S1 levels' pointwise intervals, |e| <= 1", e_lo=-1, e_hi=1),
        dict(set="S2(d) PI-1 anchors", definition="normal inputs only with PI-1's levels (version 3)", e_lo=-1, e_hi=1),
        dict(set="S2(e)", definition="S1 bands plus every in-set technology's in-support M* as an anchor", e_lo=ee[0], e_hi=ee[1]),
        dict(set="S2(e) PI-4 as published", definition="PI-1 plus every technology (version 3's PI-4)", e_lo=e4[0], e_hi=e4[1]),
        dict(set="S2(f)-total", definition="all anchors in total parameters (Marin converted; Llama 3 conversion bound)", e_lo=np.nan, e_hi=np.nan),
        dict(set="S2(f)-NF", definition="anchors and models in FLOP-effective parameters", e_lo=np.nan, e_hi=np.nan),
        dict(set="S2(f)-Chinchilla N_F", definition="Chinchilla's anchor in N_F, models in total parameters", e_lo=np.nan, e_hi=np.nan),
    ])
    K.tab(VDEF, "variants")
    K.tab(Tb.assign(variant="S2(b)"), "tinterval_anchors")
    # ------------------------------------------------------------------ S1 detail
    PB.to_csv(os.path.join(K.PROC, "model_bounds_S1.csv"), index=False)
    K.tab(PB.drop(columns=["set"]).sort_values("Cmp"), "models")
    allPB = pd.concat([PB] + [v.assign(set=k) for k, v in V.items()], ignore_index=True)
    allPB.to_csv(os.path.join(K.PROC, "model_bounds_all_sets.csv"), index=False)
    uflags = []
    for un, UU in units.items():
        G = ID.unit_frame(PB, UU)
        Gc = ID.unit_frame(V["S2(c)"], UU)
        G = G.merge(Gc[["unit", "dlo"]].rename(columns={"dlo": "dlo_S2c"}), on="unit")
        members = PB.merge(UU[["uid", "unit"]], on="uid").groupby("unit")["model"].apply(lambda s: "; ".join(s))
        G["members"] = G["unit"].map(members)
        G["units_file"] = un
        for t in K.TAUS:
            G[f"id_tau{t:g}"] = G["dlo"] > np.log(t)
        G["tau_break"] = np.exp(np.maximum(G["dlo"], 0))
        uflags.append(G)
    UF = pd.concat(uflags, ignore_index=True)
    UF.to_csv(os.path.join(K.PROC, "units_sign.csv"), index=False)
    UNID = PB[PB["dlo"] <= 0].sort_values("M")[["model", "dev", "year", "N", "D", "M", "Cmp", "Mstar_hi_eff",
                                                "Mstar_lo_eff", "binding_upper", "dlo"]]
    K.tab(UNID, "unidentified")
    MS = pd.concat([ID.mstar_sets(paths, label="S1"), ID.mstar_sets(CV["total"], label="S2(f)-total"),
                    ID.mstar_sets(CV["nf"], label="S2(f)-NF")], ignore_index=True)
    K.tab(MS, "mstar_sets")
    # bands on a grid (for inspection and figures)
    rows = []
    for k in ("chin", "llama", "marin"):
        cg = np.linspace(P[k].c_lo, P[k].c_hi, 121)
        lo, hi = P[k].band_groups(cg)
        for gi, gname in enumerate(P[k].groups):
            for j, c in enumerate(cg):
                rows.append(dict(path=P[k].name, corpus=gname, C=float(np.exp(c)), Mstar_point=float(np.exp(P[k].point(c, gi)[0])),
                                 Mstar_lo=float(np.exp(lo[gi, j])), Mstar_hi=float(np.exp(hi[gi, j]))))
    pd.DataFrame(rows).to_csv(os.path.join(K.PROC, "bands_grid.csv"), index=False)
    # ------------------------------------------------------------------ S3 breakdown frontier
    K.log("S3: breakdown frontier")
    tg = np.unique(np.concatenate([np.exp(np.linspace(0, np.log(20), 800)), [1.3, 1.84, 3.4, 4.4, 8, 10]]))
    FR = pd.concat([ID.frontier(PB, B, units, "S1", tg), ID.frontier(V["S2(c)"], B, units, "S2(c)", tg),
                    ID.frontier(V["S2(a)"], B, units, "S2(a)", tg)], ignore_index=True)
    FR.to_csv(os.path.join(K.PROC, "frontier_grid.csv"), index=False)
    K.tab(FR[FR["tau"].isin([1.0, 1.3, 1.84, 3.4, 4.4, 8.0, 10.0, 20.0])], "frontier")
    BK = pd.concat([ID.breakdown(PB, B, units, "S1"), ID.breakdown(V["S2(c)"], B, units, "S2(c)"),
                    ID.breakdown(V["S2(a)"], B, units, "S2(a)")], ignore_index=True)
    K.tab(BK, "breakdown")
    n_small = int((B["N"] < K.N_SMALL).sum())
    EX.figure(FR, len(B), n_small, n_units, ulab, experiment_factor=a.experiment_factor)
    # ------------------------------------------------------------------ S4 magnitudes
    K.log("S4: magnitude bounds under the new set")
    MAG = pd.concat([MG.model_bounds_table(PB, "S1"), MG.model_bounds_table(V["S2(a)"], "S2(a)"),
                     MG.unit_bounds_table(PB, U, ulab, "S1")], ignore_index=True)   # [review] decision level added
    K.tab(MAG, "magnitude")
    FZ = MG.fixed_zero_point(B, units)
    K.tab(FZ, "fixed_zero_point")
    # ------------------------------------------------------------------ conventions detail
    ct = info["llama3_conversion"].assign(item="Llama 3 anchor: N_F/N_total by FLOP accounting")
    mt = VA.marin_frame(total=True)[["group", "C", "lnr"]].assign(item="Marin minima: ln(N_F/N_cfg) at the minimum")
    K.tab(pd.concat([ct, mt], ignore_index=True), "conventions")
    K.tab(pd.concat([info["marin_T"].summary(), info["chin_F"].summary()] +
                    [sens[k][kk].summary().assign(sensitivity=k) for k in sens for kk in ("chin", "llama", "marin")],
                    ignore_index=True), "paths_variants")
    # ------------------------------------------------------------------ S5 table and numbers
    K.log("S5: table and paper numbers")
    N = numbers_for_note(P, PATHS, MS, A1, e1, Ac, Tb, eb, e_wide, ee, info, SH, n_units, a.B, ds)
    note = EX.table_note(N)
    if any_lt1 > 0:
        note = note.replace("; no model is identified as $w<1$ in any row", "")
    years_n = [(int(y), int((B["year"] == y).sum())) for y in sorted(B["year"].unique())]
    EX.table(SH, V, info, ulab, n_units, n_units56, len(B), n_small, years_n, note)
    PN = paper_numbers(B, PB, SH, BK, MAG, FZ, MS, UNID, units, ulab, V, P, n_small)
    K.tab(PN, "paper_numbers")
    with open(os.path.join(K.PROC, "note_numbers.json"), "w") as f:
        json.dump(N, f, indent=1)
    # ------------------------------------------------------------------ review checks
    if not a.no_review:
        import review_checks as RC
        RC.run_all(B, X, P, PB, V, SH, units, ulab, A1, e1, a.mc_reps)
    K.log(f"done in {time.time() - t0:.0f}s")


def numbers_for_note(P, PATHS, MS, A1, e1, Ac, Tb, eb, e_wide, ee, info, SH, n_units, Bboot, ds):
    f1 = lambda x: EX.r2(x, 1)  # noqa: E731
    f3 = lambda x: EX.r2(x, 3)  # noqa: E731
    g = lambda path, corpus: PATHS[(PATHS["path"] == path) & (PATHS["corpus"] == corpus)].iloc[0]  # noqa: E731
    ch, ll = g("Chinchilla", "Chinchilla"), g("Llama 3", "Llama 3")
    dc, nm, cm = g("Marin", "DCLM"), g("Marin", "Nemotron-CC"), g("Marin", "Comma")
    s1 = MS[(MS["set"] == "S1") & (MS["conv"] == "total") & (~MS["paths"].str.contains("alone"))].set_index("C")
    dsr = MS[(MS["set"] == "S1") & (MS["conv"] == "ds") & (~MS["paths"].str.contains("alone"))].set_index("C")
    com = Ac[Ac["anchor"] == "marin_comma"].iloc[0]
    tbc = Tb[Tb["design"] == "Marin, Comma"].iloc[0]
    pc = SH[(SH["set"] == "S2(a)+Comma") & (SH["level"] == "models") & (SH["year"] == "all") & (SH["tau"] == 1.0)].iloc[0]
    mt = VA.marin_frame(total=True)
    anch = mt.groupby("group")["lnr"].last()
    lr = info["llama3_lnr"]
    sci = lambda x: EX.fmt_num(x)  # noqa: E731

    def pw(x):
        e = int(np.floor(np.log10(x)))
        m = x / 10 ** e
        return rf"${EX.r2(m, 1)}\times10^{{{e}}}$"
    return dict(
        chin_n=int(ch["n_budgets"]), llama_n=int(ll["n_budgets"]), dclm_n=7, comma_n=5, marin_df=int(dc["df_resid"]),
        band_lo=pw(ch["C_band_lo"]), band_hi=pw(ch["C_band_hi"]), B=Bboot,
        chin_M=f1(ch["Mstar_J"]), chin_Mlo=f1(ch["Mstar_J_lo"]), chin_Mhi=f1(ch["Mstar_J_hi"]),
        chin_e=f3(ch["e"]), chin_elo=f3(ch["e_lo"]), chin_ehi=f3(ch["e_hi"]),
        llama_M=f1(ll["Mstar_J"]), llama_Mlo=f1(ll["Mstar_J_lo"]), llama_Mhi=f1(ll["Mstar_J_hi"]),
        llama_e=f3(ll["e"]), llama_elo=f3(ll["e_lo"]), llama_ehi=f3(ll["e_hi"]),
        dclm_M=f1(dc["Mstar_J"]), nemo_M=f1(nm["Mstar_J"]), comma_M=f1(cm["Mstar_J"]),
        marin_e=f3(dc["e"]), marin_elo=f3(dc["e_lo"]), marin_ehi=f3(dc["e_hi"]),
        ms24_lo=f1(s1.loc[1e24, "Mstar_lo"]), ms24_hi=sci(s1.loc[1e24, "Mstar_hi"]),
        ms25_lo=f1(s1.loc[1e25, "Mstar_lo"]), ms25_hi=sci(s1.loc[1e25, "Mstar_hi"]),
        ds24=f1(dsr.loc[1e24, "Mstar_lo"]), ds25=f1(dsr.loc[1e25, "Mstar_lo"]),
        pi1_eL=EX.r2(e1[0], 3), pi1_eU=EX.r2(e1[1], 3),
        comma_pi1_lo=f1(com["Mstar_lo"]), comma_pi1_hi=sci(com["Mstar_hi"]),
        pi1_comma_pct=str(int(round(100 * pc["share_gt1"]))),
        comma_t_M=f1(tbc["Mstar"]), comma_t_lo=f1(tbc["Mstar_lo"]), comma_t_hi=f1(tbc["Mstar_hi"]), comma_t_e=f3(tbc["e"]),
        tb_eL=EX.r2(eb[0], 3), tb_eU=EX.r2(eb[1], 3), c_eL=EX.r2(e_wide[0], 3), c_eU=EX.r2(e_wide[1], 3),
        e_eL=EX.r2(ee[0], 3), e_eU=EX.r2(ee[1], 3),
        marin_r_lo=EX.r2(float(np.exp(anch.min())), 2), marin_r_hi=EX.r2(float(np.exp(anch.max())), 2),
        l3_lr_lo=EX.r2(lr[0], 2), l3_lr_hi=EX.r2(lr[1], 2), n_units=n_units,
    )


def paper_numbers(B, PB, SH, BK, MAG, FZ, MS, UNID, units, ulab, V, P, n_small):
    rows = []
    src_sh, src_bk, src_mag, src_fz, src_ms, src_un = ("output/tables/rb5_sign_shares.csv", "output/tables/rb5_sign_breakdown.csv",
                                                       "output/tables/rb5_sign_magnitude.csv", "output/tables/rb5_sign_fixed_zero_point.csv",
                                                       "output/tables/rb5_sign_mstar_sets.csv", "output/tables/rb5_sign_unidentified.csv")

    def add(key, slot, desc, val, fmt, src):
        rows.append(dict(key=key, slot=slot, description=desc, value=fmt, value_raw=val, source=src))

    def sh(set_, level, tau, col="share_gt1", year="all"):
        r = SH[(SH["set"] == set_) & (SH["level"] == level) & (SH["tau"] == tau) & (SH["year"] == year)]
        return float(r[col].iloc[0])
    def pct(v):
        # [review] a share that rounds to 0 or 100 without being 0 or 1 is written '<1' or '>99' (text: 'less than 1
        # percent', 'more than 99 percent'), so that no rounded value reads as 'none' or 'all'
        p = int(np.floor(100 * v + 0.5))
        if p == 0 and v > 0:
            return "<1"
        if p == 100 and v < 1:
            return ">99"
        return str(p)
    dec = f"decisions ({ulab})"
    s1m = MS[(MS["set"] == "S1") & (MS["conv"] == "total") & (~MS["paths"].str.contains("alone"))].set_index("C")
    for C, lab in [(1e24, "10^24"), (1e25, "10^25")]:
        lo, hi = s1m.loc[C, "Mstar_lo"], s1m.loc[C, "Mstar_hi"]
        add(f"S1_Mstar_{lab}", "W11(a), W12, W23(m)", f"identified set for M*({lab}) (three studies, models' convention)",
            f"{lo:.6g}; {hi:.6g}", f"[{EX.r2(lo, 1)}, {hi:.0f}]", src_ms)
    for lev, lab in [("models", "models"), (dec, f"decisions ({ulab})"), ("decisions (56)", "decisions (56)"),
                     ("models below 15B", "models below 15B")]:
        v = sh("S1", lev, 1.0)
        add(f"S1_share_{lab}", "W11(b), H6, H11(g), W22(c), H1, W3", f"S1: share of {lab} with w > 1 identified (tau = 1)", v, pct(v), src_sh)
        v = sh("S1", lev, 1.0, "share_gt1_cw")
        add(f"S1_share_cw_{lab}", "W11(b), H6, H11(g), W22(c)", f"S1: compute-weighted share of {lab} identified (tau = 1)", v, pct(v), src_sh)
    add("S1_most_models", "H1", "S1 share of models above one half ('most models')", sh("S1", "models", 1.0) > 0.5,
        str(sh("S1", "models", 1.0) > 0.5), src_sh)
    add("S1_most_decisions", "W3", f"S1 share of decisions ({ulab}) above one half ('most allocation decisions')",
        sh("S1", dec, 1.0) > 0.5, str(sh("S1", dec, 1.0) > 0.5), src_sh)
    add("S1_most_compute", "H1, W3", "S1 compute-weighted share of models above one half", sh("S1", "models", 1.0, "share_gt1_cw") > 0.5,
        str(sh("S1", "models", 1.0, "share_gt1_cw") > 0.5), src_sh)
    for t in (1.3, 1.84, 3.4, 4.4, 10.0):
        v = sh("S1", "models", t)
        add(f"S1_share_models_tau{t:g}", "W11(c), H6", f"S1: share of models identified at tau = {t:g}", v, pct(v), src_sh)
        v = sh("S1", "models", t, "share_gt1_cw")
        add(f"S1_share_cw_models_tau{t:g}", "W11(c)", f"S1: compute-weighted share of models at tau = {t:g}", v, pct(v), src_sh)
        v = sh("S1", dec, t)
        add(f"S1_share_decisions_tau{t:g}", "W22(c), W23(m)", f"S1: share of decisions ({ulab}) at tau = {t:g}", v, pct(v), src_sh)
        v = sh("S1", dec, t, "share_gt1_cw")
        add(f"S1_share_cw_decisions_tau{t:g}", "W22(c)", f"S1: compute-weighted share of decisions ({ulab}) at tau = {t:g}", v, pct(v), src_sh)
    v = sh("S1", "models below 15B", 10.0)
    add("S1_share_small_tau10", "W11(c)", "S1: share of models below 15B identified at tau = 10", v, pct(v), src_sh)
    for y in sorted(B["year"].unique()):
        for col, lab in [("share_gt1", ""), ("share_gt1_cw", "cw_")]:
            v = sh("S1", "models", 1.0, col, str(y))
            add(f"S1_share_{lab}models_{y}", "W19(b), W23(m)", f"S1: {'compute-weighted ' if lab else ''}share of {y} models identified (tau = 1)", v, pct(v), src_sh)
    # lower end at the models' compute; unidentified
    lo_eff = PB["Mstar_lo_eff"]
    add("S1_lower_end_range", "W11(b)", "lower end of the set at the models' compute (min; max over models, models' convention)",
        f"{lo_eff.min():.4g}; {lo_eff.max():.4g}", f"{EX.r2(lo_eff.min(), 1)} to {EX.r2(lo_eff.max(), 1)}", "data/processed/rb5_sign/model_bounds_S1.csv")
    add("S1_min_M", "W11(b)", "smallest M in the clean sample", float(B["M"].min()), EX.r2(float(B["M"].min()), 1), "clean_models.csv")
    add("S1_n_unidentified", "W11(b)", "number of models whose sign is not identified (tau = 1)", len(UNID), str(len(UNID)), src_un)
    add("S1_max_M_unidentified", "W11(b), W23(m)", "largest M among them (text: 'all have M below X')", float(UNID["M"].max()),
        EX.r2(float(UNID["M"].max()), 1), src_un)
    named = PB.set_index("model").loc[NAMED, "dlo"]
    add("S1_named_unidentified", "W11(b)", "Llama 2 70B, Llama 3.1 405B, Qwen2 72B, Falcon 180B, DeepSeek LLM 67B all unidentified",
        bool((named <= 0).all()), str(bool((named <= 0).all())), "data/processed/rb5_sign/model_bounds_S1.csv")
    add("S1_n_models_unidentified_below_15B", "W11(c)", "unidentified models below 15B (tau = 1)",
        int((UNID["N"] < K.N_SMALL).sum()), str(int((UNID["N"] < K.N_SMALL).sum())), src_un)
    # S2 variants (models, decisions, compute at tau = 1 and 3.4)
    for set_ in ["S2(a)", "S2(a)+Comma", "S2(b)", "S2(b')", "S2(c)", "S2(d)", "S2(d) PI-1 anchors", "S2(e)",
                 "S2(e) PI-4 as published", "S2(f)-total", "S2(f)-NF", "S2(f)-Chinchilla N_F"]:
        for t in (1.0, 3.4):
            for lev, lab in [("models", "models"), (dec, "decisions")]:
                v = sh(set_, lev, t)
                add(f"{set_}_{lab}_tau{t:g}", "W11(b)-(c), H6, W23(m)", f"{set_}: share of {lab} identified at tau = {t:g}", v, pct(v), src_sh)
                v = sh(set_, lev, t, "share_gt1_cw")
                add(f"{set_}_{lab}_cw_tau{t:g}", "W23(m)", f"{set_}: compute-weighted share of {lab} at tau = {t:g}", v, pct(v), src_sh)
    # S3 breakdown
    for _, r in BK.iterrows():
        add(f"{r['set']}_breakdown_{r['series']}", "W11(c), H6, W23(m)", f"{r['set']}: tau at which the share ({r['series']}) falls below one half",
            r["tau_breakdown"], EX.r2(r["tau_breakdown"], 1), src_bk)
    # S4 magnitudes (S1)
    M1 = MAG[(MAG["set"] == "S1") & (MAG["level"] == "models")]
    for _, r in M1.iterrows():
        if r["tau"] not in (1.0, 1.84, 3.4):
            continue
        tag = r["curvature"].split(":")[0].split(" (")[0]
        add(f"S4_lower_{tag}_tau{r['tau']:g}", "W12, H5", f"median lower bound on s, {r['curvature']}, tau = {r['tau']:g}",
            r["median_s_lower"], EX.r2(r["median_s_lower"]), src_mag)
        add(f"S4_upper_{tag}_tau{r['tau']:g}", "W12", f"median upper bound on s, {r['curvature']}, tau = {r['tau']:g}",
            r["median_s_upper"], EX.r2(r["median_s_upper"]), src_mag)
    # [review] decision-level medians (H5 speaks of the median decision); two curvature ranges
    Md = MAG[(MAG["set"] == "S1") & (MAG["level"] == dec)]
    for _, r in Md.iterrows():
        if r["tau"] not in (1.0, 1.84, 3.4) or not r["curvature"].startswith(("k in [0.40, 0.52]", "k in [0.40, 0.6")):
            continue
        tag = r["curvature"].split(":")[0].split(" (")[0]
        add(f"S4_dec{ulab}_lower_{tag}_tau{r['tau']:g}", "H5, W12",
            f"median over the {ulab} decisions of the lower bound on s, {r['curvature']}, tau = {r['tau']:g}",
            r["median_s_lower"], EX.r2(r["median_s_lower"]), src_mag)
        add(f"S4_dec{ulab}_upper_{tag}_tau{r['tau']:g}", "H5, W12",
            f"median over the {ulab} decisions of the upper bound on s, {r['curvature']}, tau = {r['tau']:g}",
            r["median_s_upper"], EX.r2(r["median_s_upper"]), src_mag)
    for _, r in FZ.iterrows():
        v = r[f"median_s_decisions_{ulab}"]
        add(f"S4_fixed_zero_{r['scenario']}", "W12, H5", f"median decision share ({ulab} units), reference zero points fixed: {r['scenario']}",
            v, EX.r2(v), src_fz)
    # P13
    sm = PB.set_index("model").loc["SmolLM2-1.7B"]
    add("P13_SmolLM2_factor", "P13 [[S8: factor]]", "SmolLM2 1.7B: M over the upper end of the set at its compute", float(np.exp(sm["dlo"])),
        f"{np.exp(sm['dlo']):.0f}", "data/processed/rb5_sign/model_bounds_S1.csv")
    l4 = PB.set_index("model").loc["Llama-3.1-405B"]
    add("P13_405B_inside", "P13", "Llama 3.1 405B lies inside the set (dlo < 0 < dhi)", bool(l4["dlo"] < 0 < l4["dhi"]),
        str(bool(l4["dlo"] < 0 < l4["dhi"])), "data/processed/rb5_sign/model_bounds_S1.csv")
    add("P13_405B_set", "P13", "Llama 3.1 405B: set for M* at its compute (models' convention) and its M",
        f"{l4['Mstar_lo_eff']:.4g}; {l4['Mstar_hi_eff']:.4g}; M = {l4['M']:.4g}",
        f"[{EX.r2(l4['Mstar_lo_eff'], 1)}, {l4['Mstar_hi_eff']:.0f}], M = {EX.r2(l4['M'], 1)}", "data/processed/rb5_sign/model_bounds_S1.csv")
    PN = pd.DataFrame(rows)
    for c in ("value", "description"):
        for v in PN[c].astype(str):
            K.assert_no_em_dash(v, "paper numbers")
    return PN


if __name__ == "__main__":
    main()
