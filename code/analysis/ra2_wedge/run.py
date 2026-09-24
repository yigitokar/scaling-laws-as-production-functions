"""run.py -- module ra2_wedge: the re-specified inversion (R1 c1-5, 9f; R2 Majors 1-6, 10, 11; R3 M3-M5; R4 M2).

Regenerates every output of the module from raw data and the upstream module outputs (m1, m2, m3; read-only).
Deterministic (fixed seeds). CPU only, at most 5 worker processes; about 5 minutes.

  python code/analysis/ra2_wedge/run.py
Prerequisites: the m1/m2/m3 outputs; bash code/data/download_ra2_wedge.sh (HF model-tree counts, model cards, papers).

Stages
  1. sample     : clean inference-demand sample (steps a-h, audited flags with sources), family classes, conduct coding
  2. modelfree  : model-free sigma* and A2 paths from the IsoFLOP designs (Llama 3, Marin x3, Chinchilla), wild bootstrap
  3. techs      : all technologies under the ex-ante rule (+ kappa-free reference with a wild bootstrap; Chinchilla
                  non-embedding; DeepSeek LLM and MiniCPM published laws; lab-own technologies)
  4. wedges     : w and s for every model x technology; joint bootstrap summaries; bands; lab-own; MoE bounds
  5. analyses   : cleaning table, dispersion, conventions, in-support splits, family-level objects, partial identification,
                  conduct tests (wild cluster bootstrap, Webb weights), validation, aggregate, cost sensitivity
  6. outputs    : output/tables/ra2_wedge_*.csv|.tex, output/figures/ra2_wedge_*.pdf|.png, data/processed/ra2_wedge/*
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ra2common import PROC, TABLES, log, share  # noqa: E402
import analysis_ra2 as AN  # noqa: E402
import modelfree as MF  # noqa: E402
import sample as SM  # noqa: E402
import techs as TT  # noqa: E402
import wedges as WG  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
P = "ra2_wedge"


def tab(df, name, index=False):
    df.to_csv(os.path.join(TABLES, f"{P}_{name}.csv"), index=index)


def outputs(X):
    import tables_ra2 as TX
    import figures_ra2 as FG
    log("outputs: tables and figures")
    TX.write_all(X)
    FG.fig_ecdf(X["Bw"], X["L"], X["M"], X["clean"])
    FG.fig_trends(X["TRD"])
    FG.fig_modelfree(X["mf_pb"], X["mf"], X.get("M"))


def main():
    if "--outputs-only" in sys.argv:
        outputs(pd.read_pickle(os.path.join(PROC, "outputs_cache.pkl")))
        return
    t0 = time.time()
    open(os.path.join(PROC, "run_log.txt"), "w").close()
    # ------------------------------------------------------------------ 1. sample
    log("sample: clean inference-demand sample")
    d, B = SM.build()
    clean = B["clean"]
    fam_cls, fam_tab = SM.family_classes(B, clean)
    B.loc[clean, "fam_class"] = fam_cls
    fam_cls_all, fam_tab_all = SM.family_classes(B, pd.Series(True, index=B.index))
    B["fam_class_all"] = fam_cls_all
    # ------------------------------------------------------------------ 2. model-free sigma
    log("modelfree: IsoFLOP curvature and paths (wild bootstrap B = 999)")
    mf_tab, mf_pb, mf_draws = MF.run_all(B=999)
    tab(mf_tab, "modelfree_sigma")
    mf_pb.to_csv(os.path.join(PROC, "modelfree_per_budget.csv"), index=False)
    # ------------------------------------------------------------------ 3. technologies
    log("techs: registry under the ex-ante rule")
    T, M = TT.build(mf_tab, mf_draws, B_wild=399, mf_pb=mf_pb)
    keys = list(M.loc[M["in_set"] | M["sensitivity"], "key"])
    exante = list(M.loc[M["in_set"], "key"])
    # ------------------------------------------------------------------ 4. wedges
    log("wedges: every model x technology")
    L = WG.wedge_long(B, T, keys)
    L.to_csv(os.path.join(PROC, "wedge_long.csv"), index=False)
    bandtab = WG.band(L, exante)
    labtab = WG.lab_own(B, L, TT.LAB_OWN)
    # MoE bounds: reference and kappa=1 at total vs active N
    Lt = WG.wedge_long(B[B["moe"]], T, [AN.REF, AN.OLDREF], active=False)
    # wide per-model table
    W = L.pivot(index="uid", columns="tech", values="w").add_prefix("w_")
    Wlo = L.pivot(index="uid", columns="tech", values="w_lo").add_prefix("wlo_")
    Whi = L.pivot(index="uid", columns="tech", values="w_hi").add_prefix("whi_")
    Bw = B.merge(W, left_on="uid", right_index=True).merge(Wlo, left_on="uid", right_index=True) \
          .merge(Whi, left_on="uid", right_index=True).merge(bandtab, on="uid", how="left").merge(labtab, on="uid", how="left")
    Mref = L[L["tech"] == AN.REF].set_index("uid")["Mstar"]
    Bw["Mstar_ref"] = Bw["uid"].map(Mref)
    Bw["M_over_Mstar"] = Bw["M"] / Bw["Mstar_ref"]
    Bw["s_ref"] = share(Bw[f"w_{AN.REF}"])
    # vintage of the lab-own law relative to the model's release (review addition; R1 minor 37)
    lab_from = Bw["dev"].map(TT.LAB_LAW_FROM)
    Bw["lab_vintage"] = np.where(Bw["w_lab"].isna(), "",
                                 np.where(pd.to_datetime(Bw["date"]) >= pd.to_datetime(lab_from), "contemporaneous", "ex post"))
    Bw["w_primary"] = Bw["w_lab"].fillna(Bw[f"w_{AN.REF}"])
    Bw["s_primary"] = share(Bw["w_primary"])
    mt = Lt.pivot(index="uid", columns="tech", values="w").add_prefix("wtotalN_")
    Bw = Bw.merge(mt, left_on="uid", right_index=True, how="left")
    keep = ["uid", "model", "hf", "family", "gen", "dev", "date", "year", "moe", "N", "N_total", "N_nonemb", "N_emb", "emb_share",
            "N_head", "D", "M", "Cmp", "drop_step", "clean", "fam_class", "fam_class_all", "serve", "deploy",
            "Mstar_ref", "M_over_Mstar", "s_ref", "w_lab", "w_lab_lo", "w_lab_hi", "s_lab", "lab_tech", "w_lab_altmin",
            "w_lab_altmax", "lab_vintage", "w_primary", "s_primary", "band_lo", "band_hi", "pt_min", "pt_max", "n_tech"]
    wc = [c for c in Bw.columns if c.startswith(("w_", "wlo_", "whi_", "wtotalN_")) and c not in keep]
    Bw[keep + wc].to_csv(os.path.join(TABLES, f"{P}_models.csv"), index=False)
    # ------------------------------------------------------------------ 5. analyses
    log("analysis: cleaning table and technology dispersion")
    CT = AN.cleaning_table(B, L, bandtab)
    tab(CT, "cleaning")
    TD = AN.tech_dispersion(B, T, M, L, clean)
    tab(TD, "technologies")
    TDag = AN.tech_dispersion(B, T, M, L, B["clean_ag"])
    TDall = AN.tech_dispersion(B, T, M, L, pd.Series(True, index=B.index))
    tab(pd.concat([TD.assign(sample="clean"), TDag.assign(sample="clean (a)-(g)"), TDall.assign(sample="all 173")]),
        "technologies_by_sample")
    # lab-own subsample
    lab_sub = Bw[clean & Bw["w_lab"].notna()]
    LO = lab_sub[["model", "dev", "date", "year", "lab_vintage", "M", "lab_tech", "w_lab", "w_lab_lo", "w_lab_hi", "s_lab",
                  "w_lab_altmin", "w_lab_altmax", f"w_{AN.REF}", "s_ref", f"w_{AN.OLDREF}", "band_lo", "band_hi"]]
    tab(LO, "labown")
    lab_summary = {}
    for nm, g in [("all", lab_sub), ("contemporaneous", lab_sub[lab_sub["lab_vintage"] == "contemporaneous"]),
                  ("ex post", lab_sub[lab_sub["lab_vintage"] == "ex post"])]:
        lab_summary[nm] = dict(n=int(len(g)), median_s_lab=float(np.median(share(g["w_lab"]))) if len(g) else None,
                               median_s_ref=float(np.median(g["s_ref"])) if len(g) else None,
                               median_w_lab=float(np.median(g["w_lab"])) if len(g) else None,
                               median_w_ref=float(np.median(g[f"w_{AN.REF}"])) if len(g) else None)
    # families
    log("analysis: family-level token budgets")
    fam_tab.reset_index().to_csv(os.path.join(TABLES, f"{P}_families.csv"), index=False)
    Bw["fam_class"] = Bw["fam_class"].fillna("")
    fam_split = []
    for cls in ["common-D", "size-specific-D", "singleton"]:
        g = Bw[clean & (Bw["fam_class"] == cls)]
        fam_split.append(dict(cls=cls, n=len(g), n_families=g["gen"].nunique(), median_w=g[f"w_{AN.REF}"].median(),
                              share_w_gt1=(g[f"w_{AN.REF}"] > 1).mean(), median_s=g["s_ref"].median(),
                              share_band_gt1=(g["band_lo"] > 1).mean(), median_M=g["M"].median()))
    # review addition: models whose individual wedge is a (member-level) revealed-preference object under the
    # maintained conduct, i.e. excluding common-D families (R1 c2b)
    g = Bw[clean & Bw["fam_class"].isin(["size-specific-D", "singleton"])]
    fam_split.append(dict(cls="size-specific-D + singleton (clean; excludes common-D)", n=len(g), n_families=g["gen"].nunique(),
                          median_w=g[f"w_{AN.REF}"].median(), share_w_gt1=(g[f"w_{AN.REF}"] > 1).mean(),
                          median_s=g["s_ref"].median(), share_band_gt1=(g["band_lo"] > 1).mean(), median_M=g["M"].median()))
    for cls in ["common-D", "size-specific-D", "singleton"]:
        g = Bw[Bw["fam_class_all"] == cls]
        fam_split.append(dict(cls=cls + " (all 173)", n=len(g), n_families=g["gen"].nunique(), median_w=g[f"w_{AN.REF}"].median(),
                              share_w_gt1=(g[f"w_{AN.REF}"] > 1).mean(), median_s=g["s_ref"].median(),
                              share_band_gt1=(g["band_lo"] > 1).mean(), median_M=g["M"].median()))
    FS = pd.DataFrame(fam_split)
    tab(FS, "family_split")
    # review addition (R1 c1 request: confine 'planned serving expenditure' to developers that serve): headline by
    # serving footprint of the developer at release, and by stated deployment target (clean sample, reference)
    ss = []
    for lab_, msk in [("developer serves (first-party API/product at release)", clean & (Bw["serve"] == 1)),
                      ("developer does not serve", clean & (Bw["serve"] == 0)),
                      ("on-device or local target (card)", clean & Bw["deploy"].isin(["ondevice", "local"])),
                      ("server or unspecified target", clean & ~Bw["deploy"].isin(["ondevice", "local"]))]:
        g = Bw[msk]
        ss.append(dict(group=lab_, n=len(g), n_developers=g["dev"].nunique(), median_w=g[f"w_{AN.REF}"].median(),
                       median_s=g["s_ref"].median(), share_w_gt1=(g[f"w_{AN.REF}"] > 1).mean(),
                       median_s_primary=g["s_primary"].median(), median_M=g["M"].median(),
                       C_weighted_s_agg=float(((np.maximum(g[f"w_{AN.REF}"], 1) - 1) * g["Cmp"]).sum() /
                                              (np.maximum(g[f"w_{AN.REF}"], 1) * g["Cmp"]).sum())))
    tab(pd.DataFrame(ss), "serve_split")
    FL = WG.family_level(Bw, clean & (Bw["fam_class"] == "common-D"), f"w_{AN.REF}")
    tab(FL, "family_level_W")
    foc = [WG.verify_family_foc(s) for s in range(3)]
    # conventions
    log("analysis: conventions, in-support, partial identification")
    CV = AN.conventions(B, T, L, clean)
    tab(CV, "conventions")
    IS, isflags = AN.in_support(B, L, bandtab, clean)
    tab(IS, "insupport")
    # direction of the extrapolation error from ra1's model-free Farseer wedge (review addition; spec task 6)
    EXC = AN.extrapolation_check(B, L, clean, {k: float(T[k].support["M_max"]) for k in ("farseer_q", "farseer", "farseer_emb")})
    tab(EXC, "extrapolation_check")
    # partial identification
    anchors = AN.pi_anchors(T, M, mf_tab, mf_draws)
    tab(anchors, "pi_anchors")
    a_iso = list(anchors.loc[anchors["kind"] == "iso", "a"]) + list(TT.DEEPSEEK["table4_a"].values()) + [T["minicpm"].a]
    e_iso = (1 - 2 * max(a_iso), 1 - 2 * min(a_iso))
    a_all = list(M.loc[M["in_set"], "a"]) + a_iso
    e_all = (1 - 2 * max(a_all), 1 - 2 * min(a_all))
    S2_core = (min(TD.loc[TD["key"].isin(["chin_q", "farseer_q", "meta_mf", "marin_nemotron_mf", "marin_dclm_mf",
                                          "marin_comma_mf"]), "S2"]),
               max(TD.loc[TD["key"].isin(["chin_q", "farseer_q", "meta_mf", "marin_nemotron_mf", "marin_dclm_mf",
                                          "marin_comma_mf"]), "S2"]))
    S2_all = (TD.loc[TD["in_set"], "S2"].min(), TD.loc[TD["in_set"], "S2"].max())
    sets = {"PI-1 IsoFLOP anchors, e from A2/published paths": dict(kinds=["iso"], e=e_iso),
            "PI-2 PI-1 + M* nondecreasing in C": dict(kinds=["iso"], e=(0.0, e_iso[1])),
            "PI-3 PI-1 with own-lab anchor": dict(kinds=["iso", "all"], e=e_iso, own_lab=True),
            "PI-4 all ex-ante technologies as anchors, e from all paths": dict(kinds=["iso", "all"], e=e_all)}
    # own-lab set: labs with own anchors use only them; other models fall back to IsoFLOP anchors
    PIb = []
    for name, spec in sets.items():
        if spec.get("own_lab"):
            # own-lab anchors: the lab's IsoFLOP A2 path (Meta, Marin x3) or published law (DeepSeek) at its largest
            # budget; AI2 has no IsoFLOP design: the OLMo ladder's in-support M* (kappa = 1 and kappa free)
            own = anchors[((anchors["kind"] == "iso") & (anchors["lab"].fillna("") != "")) |
                          (anchors["anchor"].isin(["olmo", "olmo_q"]))]
            labs = set(own["lab"])
            m_own = clean & B["dev"].isin(labs)
            PIb.append(AN.pi_bounds(B, m_own, own, spec["e"], S2_core, {name: dict(kinds=["iso", "all"], e=spec["e"], own_lab=True)}))
            PIb.append(AN.pi_bounds(B, clean & ~B["dev"].isin(labs), anchors, spec["e"], S2_core, {name: dict(kinds=["iso"], e=spec["e"])}))
        else:
            PIb.append(AN.pi_bounds(B, clean, anchors, spec["e"], S2_core, {name: spec}))
    PIb = pd.concat(PIb, ignore_index=True)
    PIb.to_csv(os.path.join(PROC, "pi_bounds_models.csv"), index=False)
    PIs = PIb.groupby("set").apply(lambda g: pd.Series(dict(
        n=len(g), share_w_gt1_identified=(g["sign"] == "w>1").mean(), share_w_lt1_identified=(g["sign"] == "w<1").mean(),
        share_ambiguous=(g["sign"] == "ambiguous").mean(), median_s_lo=g["s_lo"].median(), median_s_hi=g["s_hi"].median()))).reset_index()
    PIs["e_lo"] = [sets[s]["e"][0] for s in PIs["set"]]
    PIs["e_hi"] = [sets[s]["e"][1] for s in PIs["set"]]
    tab(PIs, "pi_summary")
    PIg = pd.concat([AN.pi_mstar_grid(anchors, ["iso"], e_iso).assign(set="PI-1"),
                     AN.pi_mstar_grid(anchors, ["iso"], (0.0, e_iso[1])).assign(set="PI-2"),
                     AN.pi_mstar_grid(anchors[anchors["anchor"] == "meta"], ["iso"], e_iso).assign(set="PI-3 Meta anchor"),
                     AN.pi_mstar_grid(anchors, ["iso", "all"], e_all).assign(set="PI-4 (total-N anchors)")])
    tab(PIg, "pi_mstar")
    # conduct tests
    log("analysis: conduct tests (wild cluster bootstrap, Webb weights, B = 9999)")
    U = SM.universe(d, B)
    U["N_nonemb"] = np.nan
    Lu = WG.wedge_long(U, T, [AN.REF])
    OC, Uc = AN.open_closed(U, Lu)
    SV, Xs = AN.serving_ondevice(B, clean, L, "clean sample")
    SV2, _ = AN.serving_ondevice(B, B["clean_ag"] | (B["drop_step"].isin(["b1_research_suite", "b2_replication"])),
                                 L, "clean + research suites (a, c-g applied)")
    # tiers: bunching on open-weight production-scale models 2023-2026 (total parameters), regression on clean + universe
    Uo = U[U["open_weights"] & U["year"].between(2023, 2026) & ~U["nontransformer"]]
    BU = AN.bunching(Uo["N_total"].fillna(Uo["N"]).values)
    BU2 = AN.bunching(Bw.loc[clean, "N"].values)
    Xc = Xs.copy()
    TR = pd.concat([AN.tier_regression(Xc, "clean sample"),
                    AN.tier_regression(Uc[Uc["open_weights"]].assign(lnw=lambda x: np.log(x["w"])), "open-weight universe (clean)")])
    COND = pd.concat([OC.assign(block="(a) open vs closed"), SV.assign(block="(b) serving vs on-device"),
                      SV2.assign(block="(b) serving vs on-device, robustness"), TR.assign(block="(c) tiers")], ignore_index=True)
    tab(COND, "conduct")
    BU = pd.concat([BU.assign(sample="open-weight production-scale 2023-2026 (total N)"), BU2.assign(sample="clean sample")])
    tab(BU, "bunching")
    Xs[["uid", "model", "dev", "year", "serve", "deploy", "ondevice", "local", "lnw", "s", "lnC"]].to_csv(
        os.path.join(PROC, "conduct_sample.csv"), index=False)
    coding = B[["model", "hf", "dev", "year", "serve", "serve_source", "deploy", "deploy_source", "drop_step",
                "drop_reason", "drop_source"]]
    coding.to_csv(os.path.join(TABLES, f"{P}_coding.csv"), index=False)
    # validation
    log("analysis: validation (HF model tree, OpenRouter audit, aggregates)")
    H = AN.hf_counts(B)
    H.to_csv(os.path.join(PROC, "hf_tree_counts.csv"), index=False)
    HV, Xh = AN.hf_validation(B, clean, L, H)
    OR = AN.openrouter_audit(B, L)
    tab(HV, "validation_hf")
    tab(OR, "validation_openrouter")
    Bc = Bw[clean].copy()
    SA = pd.concat([AN.s_aggregate(Bc, f"w_{AN.REF}").assign(tech="reference", sample="clean"),
                    AN.s_aggregate(Bc, "w_primary").assign(tech="lab-own where available", sample="clean"),
                    AN.s_aggregate(Bc, "pt_min").assign(tech="min over technologies", sample="clean"),
                    AN.s_aggregate(Bc, "pt_max").assign(tech="max over technologies", sample="clean"),
                    AN.s_aggregate(Bc, f"w_{AN.REF}", drop_uid=set(Bc.loc[Bc["model"] == "Llama-3.1-405B", "uid"])).assign(
                        tech="reference", sample="clean, leave out Llama 3.1 405B"),
                    AN.s_aggregate(Bc.assign(all="2019-2025"), f"w_{AN.REF}", by="all").assign(tech="reference", sample="clean")])
    Uo2 = Uc[Uc["open_weights"]].copy()
    SA = pd.concat([SA, AN.s_aggregate(Uo2, "w").assign(tech="reference", sample="open-weight universe (clean)"),
                    AN.s_aggregate(Uo2, "w", drop_uid=set(Uo2.loc[Uo2["model"] == "Llama-3.1-405B", "uid"])).assign(
                        tech="reference", sample="open-weight universe, leave out Llama 3.1 405B")])
    tab(SA, "aggregate")
    # curvature sensitivity at the reference zero point (R2 Major 1): ln w = (1/sigma* - 1) ln(M / M*_ref(C))
    lnratio = np.log(Bc["M"] / Bc["Mstar_ref"]).values
    l3 = float(np.log(Bc.loc[Bc["model"] == "Meta-Llama-3-8B", "M_over_Mstar"].iloc[0]))
    rows = []
    for sg in (0.55, 0.60, 0.65, 0.678, 0.70, 0.737, 0.75, 0.80):
        k = 1 / sg - 1
        w_ = np.exp(k * lnratio)
        rows.append(dict(sigma_star=sg, slope=k, median_w=float(np.median(w_)), median_s=float(np.median(share(w_))),
                         share_gt1=float((w_ > 1).mean()), w_llama3_8b=float(np.exp(k * l3))))
    tab(pd.DataFrame(rows), "sigma_sensitivity")
    # how mechanical is the reference wedge? R^2 of ln w on ln M (clean sample)
    yv, xv = np.log(Bc[f"w_{AN.REF}"]).values, np.log(Bc["M"]).values
    bb = np.polyfit(xv, yv, 1)
    r2 = 1 - np.var(yv - np.polyval(bb, xv)) / np.var(yv)
    xc2 = np.column_stack([np.ones(len(xv)), xv, np.log(Bc["Cmp"]).values])
    bc2 = np.linalg.lstsq(xc2, yv, rcond=None)[0]
    r2c = 1 - np.var(yv - xc2 @ bc2) / np.var(yv)
    spear = float(pd.Series(yv).corr(pd.Series(xv), method="spearman"))
    # cost sensitivity
    wref = Bc[f"w_{AN.REF}"].values
    wl3 = float(Bc.loc[Bc["model"] == "Meta-Llama-3-8B", f"w_{AN.REF}"].iloc[0])
    CS = AN.cost_sensitivity(wref, wl3)
    tab(CS, "costsens")
    # trends by year (figure v): clean production-scale universe
    Uc["s"] = share(Uc["w"])
    TRD = Uc.groupby(["year", "open_weights"]).agg(n=("s", "size"), median_s=("s", "median"),
                                                 q25=("s", lambda v: v.quantile(.25)), q75=("s", lambda v: v.quantile(.75)),
                                                 median_w=("w", "median")).reset_index()
    tab(TRD, "trends")
    # ------------------------------------------------------------------ 6. tables and figures
    X = dict(S2_core=S2_core, EXC=EXC, CT=CT, TD=TD, Bw=Bw, LO=LO, FS=FS, FL=FL, CV=CV, IS=IS, PIs=PIs, PIg=PIg, COND=COND, BU=BU, HV=HV, OR=OR,
             SA=SA, CS=CS, M=M, mf=mf_tab, mf_pb=mf_pb, anchors=anchors, Lt=Lt, clean=clean, L=L, TRD=TRD)
    pd.to_pickle(X, os.path.join(PROC, "outputs_cache.pkl"))
    outputs(X)
    # ------------------------------------------------------------------ headline numbers
    ref = TD.set_index("key").loc[AN.REF]
    H = dict(
        n_start=int(len(B)), n_clean=int(clean.sum()), n_clean_ag=int(B["clean_ag"].sum()),
        mf_source=M.attrs.get("mf_source"), chin_q_point={k: float(v) for k, v in M.attrs["chin_q_point"].items()},
        ref=dict(median_w=ref["med_w"], median_w_ci=[ref["med_w_lo"], ref["med_w_hi"]], share_gt1=ref["share_gt1"],
                 share_gt1_ci=[ref["share_gt1_lo"], ref["share_gt1_hi"]], median_s=ref["med_s"],
                 median_s_ci=[ref["med_s_lo"], ref["med_s_hi"]], B=int(ref["B"])),
        exante_range_median_w=[float(TD.loc[TD["in_set"], "med_w"].min()), float(TD.loc[TD["in_set"], "med_w"].max())],
        exante_range_median_s=[float(TD.loc[TD["in_set"], "med_s"].min()), float(TD.loc[TD["in_set"], "med_s"].max())],
        exante_range_share_gt1=[float(TD.loc[TD["in_set"], "share_gt1"].min()), float(TD.loc[TD["in_set"], "share_gt1"].max())],
        share_band_gt1_clean=float((Bw.loc[clean, "band_lo"] > 1).mean()),
        e_iso=list(e_iso), e_all=list(e_all), S2_core=list(S2_core), S2_all=list(S2_all),
        family_foc_check=[f["lhs"] for f in foc],
        mechanical=dict(slope_lnw_lnM=float(bb[0]), r2_lnM=float(r2), r2_lnM_lnC=float(r2c), spearman_w_M=spear),
        labown_summary=lab_summary,
        mf_sigma={k: dict(sigma=float(T[f"{k}_mf"].sigma_star), sigma_ra2_bc=float(T[f"{k}_mfbc"].sigma_star),
                          sigma_ra2_raw=float(T[f"{k}_mfraw"].sigma_star))
                  for k in ("meta", "marin_comma", "marin_dclm", "marin_nemotron")},
        r4_check_median_w_173_m1kappa=float(np.median(np.exp(
            (0.4242726607468513 + 0.4309173006470277) * ((np.log(0.4242726607468513) + 7.741635755685286 - np.log(0.4309173006470277)
                                                          - 9.214466348215746) / (0.4242726607468513 + 0.4309173006470277))
            - 0.4242726607468513 * np.log(B["N"]) + 0.4309173006470277 * np.log(B["D"])))),
        runtime_s=time.time() - t0)
    with open(os.path.join(PROC, "headline.json"), "w") as f:
        json.dump(H, f, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
