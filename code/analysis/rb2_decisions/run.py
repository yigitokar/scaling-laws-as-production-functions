"""run.py -- module rb2_decisions: the revealed value of compactness at the level of allocation decisions.

Round-2 requests: R1 New 1, 3, 4, 5; R2 Majors 2, 4, 6, 7; R3 N2, N3, N4; R4 R2-M1, R2-M2, R2-M3
(paper/notes/revision_plan_v3.md is binding). Regenerates every output of the module (deterministic; fixed seeds;
CPU only, at most 2 processes (round 3; GPU queues running); about 5 minutes, the OLMo-ladder joint bootstrap being the slow part):

  .venv/bin/python code/analysis/rb2_decisions/run.py [--rebuild-techs]

Prerequisites: the reviewed outputs of m1/m2/m3/ra1/ra2 (read-only) and the primary sources saved once by
code/analysis/rb2_decisions/fetch_sources.py in data/raw/rb2_decisions/ (every quoted passage is verified against them).

Stages
  1. base       rebuild ra2's clean sample and technologies by import (token-count audit applied)
  2. evidence   family readings (cap / choice / menu / cap+menu) and the model-level serving code, with verified quotes
  3. lab-own    one rule: the lab's own path + the common model-free sigma*; lab curvature and Meta's 10-budget law as
                sensitivity rows
  4. decisions  decision units; medians and shares under the reference and lab-own; joint bootstrap; variants;
                the reference under ra1's budget-plus-trunk cluster bootstrap (cluster_ref.py; review addition).
                Round 3: the primary unit is the budget level (49 units; readings from module rb5_units under
                paper/notes/rb5_reading_protocol.md); version 3's 56 family-label and 66 reading-consistent units are
                robustness rows; W2 rows give the point-identified and lower-bounded units; Table E3's model-level
                columns, Table E4 and the cleaning steps are recomputed on the audited token counts with ra2's functions
  5. conduct    serving tests (model and decision level), tier flags, on-device contrast, open-weight premium
  6. sign       PI-1 with wild bootstrap-t anchors; share identified by tilt allowance, weighted, by year, by unit
  7. trend      2023-2025 on the clean sample; 32 technologies; leave-one-developer-out; pre-2023 robustness
  8. output     second-output (Bond et al.) test on the OLMo ladder
  9. post       post-training compute bound
 10. exhibits   Table 2 (decisions), ECDF figure, appendix tables, headline.json
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import rb2common as C  # noqa: E402
import base  # noqa: E402
import evidence as E  # noqa: E402
import labown as LO  # noqa: E402
import decisions as DC  # noqa: E402
import serving as SV  # noqa: E402
import signid as SI  # noqa: E402
import trend as TR  # noqa: E402
import second_output as SO  # noqa: E402
import posttrain as PT  # noqa: E402
import wedges as WG  # noqa: E402  (ra2)
import sample as SM  # noqa: E402  (ra2)
import techs as TT  # noqa: E402  (ra2)
import analysis_ra2 as AN  # noqa: E402  (ra2)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# Joint bootstrap draws for sample statistics. Review fix: the reference technology has 399 wild draws; B = 999 cycled
# them unevenly (draws 0-200 three times, 201-398 twice). BJ is now set in main() to an exact multiple (3 x 399 =
# 1,197), so every reference draw has the same weight and the lab-own path draws (9,999 available) are all distinct.
BJ_MULT = 3
B_CLUSTER = 999   # budget-plus-trunk cluster bootstrap of the reference (R4 R2-M3(d); review addition)


def main():
    if "--exhibits-only" in sys.argv:          # redraw tables and figures from the cached stage outputs
        import exhibits_rb2 as EX
        EX.write_all(pd.read_pickle(os.path.join(C.PROC, "exhibits_cache.pkl")))
        return
    t0 = time.time()
    open(os.path.join(C.PROC, "run_log.txt"), "w").close()
    H = {}
    # ------------------------------------------------------------------ 1. base
    C.log("1. base: ra2 objects by import; token-count audit")
    X = base.load(rebuild="--rebuild-techs" in sys.argv)
    B, L, T, M = X["B"], X["L"], X["T"], X["M"]
    exante = X["exante"]
    base.audit_table().to_csv(os.path.join(C.PROC, "d_audit.csv"), index=False)
    Bc = B[B["clean"]].copy().reset_index(drop=True)
    Lr = L[L["tech"] == C.REF].set_index("uid")
    Bc["w_ref"] = Bc["uid"].map(Lr["w"])
    Bc["w_ref_lo"] = Bc["uid"].map(Lr["w_lo"])
    Bc["w_ref_hi"] = Bc["uid"].map(Lr["w_hi"])
    Bc["Mstar_ref"] = Bc["uid"].map(Lr["Mstar"])
    Bc["s_ref"] = C.share(Bc["w_ref"])
    Bc["tier"] = C.in_tier(Bc["N"].values).astype(int)
    Bc["tier_alt"] = C.in_tier(Bc["N"].values, C.TIER_LO_ALT, C.TIER_HI_ALT).astype(int)
    # ra2's lab-own primary (10-budget Meta law with Meta curvature; OLMo kappa free with its own curvature; ...)
    lab0 = WG.lab_own(B, L, TT.LAB_OWN).set_index("uid")
    Bc["w_primary"] = Bc["uid"].map(lab0["w_lab"]).fillna(Bc["w_ref"])
    # ------------------------------------------------------------------ 2. evidence
    C.log("2. evidence: family readings and model-level serving (quotes verified against saved sources)")
    Wf = WG.family_level(Bc, Bc["fam_class"] == "common-D", "w_ref")
    FR = E.family_readings(Bc, Wf)
    FR.to_csv(os.path.join(C.PROC, "family_readings.csv"), index=False)
    SVT = E.serving_table(Bc)
    SVT.to_csv(os.path.join(C.PROC, "serving_model_level.csv"), index=False)
    Bc = Bc.merge(SVT[["uid", "serve_model", "serve_model_hi", "serve_ambiguous", "demo", "channel"]], on="uid")
    # ------------------------------------------------------------------ 3. lab-own, one rule
    C.log("3. lab-own technologies under one rule")
    LOT, LOD, LOM = LO.build(X, B=9999)
    Bc = Bc.merge(LOT[["uid", "w_lab1", "w_lab1_lo", "w_lab1_hi", "lab_path"]], on="uid", how="left")
    Bc["w_primary1"] = Bc["w_lab1"].fillna(Bc["w_ref"])
    H["common_sigma"] = LOM["sigma"]
    lab_rows = []
    sub = Bc[Bc["w_lab1"].notna()]
    for lab_, g in [("all lab-own models", sub), ("without AI2", sub[sub["dev"] != "AI2"]),
                    ("Meta", sub[sub["dev"] == "Meta"]), ("AI2", sub[sub["dev"] == "AI2"])]:
        lt = LOT.set_index("uid").loc[g["uid"]]
        row = dict(group=lab_, n=len(g), median_s_lab_one_rule=float(np.median(C.share(g["w_lab1"]))),
                   median_s_reference=float(np.median(g["s_ref"])),
                   median_s_ra2_labown=float(np.median(C.share(g["w_primary"]))))
        for c in [c for c in LOT.columns if c.startswith("w_sens_")]:
            v = lt[c].fillna(lt["w_lab1"]).values
            row["median_s_" + c[7:]] = float(np.median(C.share(v)))
        lab_rows.append(row)
    LAB = pd.DataFrame(lab_rows)
    C.tab(LAB, "labown_summary")
    C.tab(LOT.merge(Bc[["uid", "w_ref", "w_primary", "year", "date"]], on="uid"), "labown_models")
    # ------------------------------------------------------------------ 4. decisions
    C.log("4. decision units (budget level, primary; rb5 readings), headline and W2 statistics (joint bootstrap)")
    RD = DC._rb5()
    CODES = RD.load()
    H["codes_file"] = str(CODES["codes_file"].iloc[0])
    U = {"primary": DC.units_budget(Bc, CODES, "budget"),         # 49 budget-level units (decision D-2)
         "family": DC.units(Bc, FR, "primary"),                  # 56 family-label units (version 3's primary)
         "reading": DC.units(Bc, FR, "reading"),                 # 66 reading-consistent units (version 3's definition)
         "split": DC.units_budget(Bc, CODES, "split"),           # cap-type budgets entered as members (lower bounds)
         "olmo1b": DC.units_budget(Bc, CODES, "olmo1b"),         # primary with OLMo 2 1B as its own unit
         "subgroup": DC.units(Bc, FR, "subgroup")}               # version 3's sub-group units (same partition as primary)
    part = lambda u: sorted(u.groupby("unit")["uid"].apply(lambda x: tuple(sorted(x))))
    assert part(U["primary"]) == part(U["subgroup"]), "budget-level units differ from the sub-group partition"
    # file names: units_primary.csv keeps version 3's 56 family-label units (read by rb5_sign as its robustness row);
    # the new primary is units_budget.csv here and data/processed/rb5_units/units_primary.csv
    UFILES = {"primary": "units_budget", "family": "units_primary", "reading": "units_reading", "split": "units_split",
              "olmo1b": "units_olmo1b", "subgroup": "units_subgroup"}
    for s, u in U.items():
        u.to_csv(os.path.join(C.PROC, f"{UFILES[s]}.csv"), index=False)
    UT = DC.unit_table(Bc, U["primary"], SVT)
    # draws: reference (wild, B=399) and lab-own one rule (B=9999) combined for 'lab-own where available'
    BJ = BJ_MULT * len(T[C.REF].draws)
    H["B_joint"] = BJ
    LW_ref = DC.lnw_draws_tech(Bc, T[C.REF], BJ)
    LW_lab = LW_ref.copy()
    for i, u in enumerate(Bc["uid"]):
        if u in LOD:
            LW_lab[:, i] = LOD[u][:BJ]
    SLAB = {"primary": "decision units (primary: budget level)", "family": "decision units (family label, version 3)",
            "reading": "decision units (reading-consistent, version 3)",
            "split": "decision units (members of cap-type budgets split)",
            "olmo1b": "decision units (primary, OLMo 2 1B separate)"}

    def extras(u):
        ub = u.drop_duplicates("unit")
        e = dict(n_family_units=int((ub["unit_type"] == "family").sum()), n_dropped_models=int(len(Bc) - len(u)))
        if "id_class" in ub:
            e.update(n_point=int((ub["id_class"] == "point").sum()),
                     n_lower=int(ub["id_class"].str.startswith("lower").sum()),
                     n_lower_point_if_choice=int((ub["id_class"] == "lower bound (point if choice)").sum()))
        else:
            e.update(n_lower=int(ub["bound"].str.startswith("lower").sum()),
                     n_upper=int((ub["bound"] == "upper bound").sum()))
        return e
    rows = []
    for tech_lab, wcol, LW in [("reference", "w_ref", LW_ref), ("lab-own (one rule) where available", "w_primary1", LW_lab),
                               ("lab-own (ra2 rule) where available", "w_primary", None)]:
        rows.append(DC.summarize("models (77)", Bc[wcol].values, None if LW is None else
                                 dict(med_s=np.median(C.share(np.exp(LW)), axis=1), share_gt1=(LW > 0).mean(axis=1),
                                      med_w=np.median(np.exp(LW), axis=1)), len(Bc), dict(tech=tech_lab)))
        for s, lab in SLAB.items():
            u = U[s]
            uv = DC.unit_values(Bc, u, wcol)
            js = DC.joint_stats(Bc, u, LW) if LW is not None else None
            rows.append(DC.summarize(lab, uv["W"].values, js, len(uv), dict(tech=tech_lab, **extras(u))))
    # Review addition (R4 R2-M3(d), request 3): the reference under ra1's conservative budget-plus-trunk cluster
    # bootstrap (44 clusters), beside the design-conditional wild interval above. Same point estimates.
    import cluster_ref as CR
    import types
    pq_cl, dr_cl, ncl, nfail = CR.draws(B=B_CLUSTER)
    H["cluster_ref"] = dict(B=int(len(dr_cl)), n_clusters=ncl, n_fail=nfail, point=pq_cl)
    t_cl = types.SimpleNamespace(n_conv=T[C.REF].n_conv, draws=dr_cl)
    LW_cl = DC.lnw_draws_tech(Bc, t_cl, len(dr_cl))
    tl_cl = "reference, budget+trunk cluster bootstrap (ra1's conservative scheme)"
    rows.append(DC.summarize("models (77)", Bc["w_ref"].values,
                             dict(med_s=np.median(C.share(np.exp(LW_cl)), axis=1), share_gt1=(LW_cl > 0).mean(axis=1),
                                  med_w=np.median(np.exp(LW_cl), axis=1)), len(Bc), dict(tech=tl_cl)))
    for s in ("primary", "family", "reading", "split"):
        uv = DC.unit_values(Bc, U[s], "w_ref")
        rows.append(DC.summarize(SLAB[s], uv["W"].values, DC.joint_stats(Bc, U[s], LW_cl), len(uv), dict(tech=tl_cl)))
    ms_w = np.exp(C.ln_mstar(5.76e23, T[C.REF].draws[:, 0], T[C.REF].draws[:, 1], T[C.REF].draws[:, 2]))
    ms_c = np.exp(C.ln_mstar(5.76e23, dr_cl[:, 0], dr_cl[:, 1], dr_cl[:, 2]))
    H["cluster_ref"].update(Mstar_576e23_wild=[C.pct(ms_w, 2.5), C.pct(ms_w, 97.5)],
                            Mstar_576e23_cluster=[C.pct(ms_c, 2.5), C.pct(ms_c, 97.5)])
    # version 3 row kept for continuity: reading-consistent units without cap members inside a tier window
    Ur = U["reading"]
    conflict = Ur["uid"].isin(Bc.loc[Bc["tier"] == 1, "uid"]) & Ur["bound"].str.startswith("lower")
    Ur2 = Ur[~conflict]
    uv = DC.unit_values(Bc, Ur2, "w_ref")
    rows.append(DC.summarize("decision units (reading-consistent, version 3), cap members in tier windows dropped", uv["W"].values,
                             DC.joint_stats(Bc, Ur2, LW_ref), len(uv),
                             dict(tech="reference", n_dropped_models=int(len(Bc) - len(Ur2)),
                                  n_lower_bound=int(Ur2.drop_duplicates("unit")["bound"].str.startswith("lower").sum()))))
    # W2 (decision D-1): identification classes of the primary units under the virtual value of compactness, with joint
    # intervals (wild, and the cluster scheme for the point-identified median)
    Up = U["primary"]
    UB = Up.drop_duplicates("unit").set_index("unit")
    CODES_B = RD.load(os.path.join(os.path.dirname(RD.source_file()), "readings_coderB.csv")).set_index("unit")  # [review]
    tier_unit = Bc.merge(Up[["uid", "unit"]], on="uid").groupby("unit")["tier"].max()
    UB["tier_any"] = tier_unit
    W2 = {"point-identified units": UB.index[UB["id_class"] == "point"],
          "lower-bounded units": UB.index[UB["id_class"] != "point"],
          "lower-bounded units, cap-type budgets": UB.index[(UB["id_class"] != "point") & (UB["unit_type"] == "family")],
          "lower-bounded units, members with an own cap": UB.index[(UB["id_class"] != "point") & (UB["unit_type"] == "model")],
          # [review] repetition by every member (C2) keeps a 'cap or choice' budget a lower bound (readings.id_class)
          "point-identified units if 'cap or choice' is a choice": UB.index[UB["point_if_choice"]],
          # [review] protocol Section 11: the other coder's codes where they change a result; the weakest agreed choices
          "point-identified units under coder B's readings": UB.index[UB.index.isin(CODES_B.index[CODES_B["id_class"] == "point"])],
          "point-identified units without MPT and Yi (weakest agreed choice codes)":
              UB.index[(UB["id_class"] == "point") & ~UB.index.isin(["MPT", "Yi"])],
          "point-identified units without synthetic-flagged units (protocol rule)": UB.index[(UB["id_class"] == "point") & (UB["syn_protocol"] == 0)],
          "point-identified units without synthetic-flagged units (fix-list families)": UB.index[(UB["id_class"] == "point") & (UB["syn_fixlist"] == 0)],
          "units outside tier windows (interpretation check)": UB.index[UB["tier_any"] == 0],
          "point-identified units outside tier windows": UB.index[(UB["id_class"] == "point") & (UB["tier_any"] == 0)],
          "size-specific members and singletons (units of one model)": UB.index[UB["unit_type"] == "model"],
          "size-specific members and singletons outside tier windows": UB.index[(UB["unit_type"] == "model") & (UB["tier_any"] == 0)],
          "size-specific members and singletons outside tier windows, without synthetic-flagged (protocol)":
              UB.index[(UB["unit_type"] == "model") & (UB["tier_any"] == 0) & (UB["syn_protocol"] == 0)]}
    for name, keep in W2.items():
        Us = Up[Up["unit"].isin(keep)]
        uv = DC.unit_values(Bc, Us, "w_ref")
        for tl, LW in (("reference", LW_ref), (tl_cl, LW_cl)):
            if tl == tl_cl and not name.startswith("point-identified units"):
                continue
            rows.append(DC.summarize(f"W2: {name}", uv["W"].values, DC.joint_stats(Bc, Us, LW), len(uv),
                                     dict(tech=tl, n_models=int(len(Us)))))
    H["W2_counts"] = {k: int(len(v)) for k, v in W2.items()}
    # robustness rows (reference): subsets of the primary decision units
    def sub_units(mask, name, wcol="w_ref"):
        Bs = Bc[mask]
        uv = DC.unit_values(Bs, Up[Up["uid"].isin(Bs["uid"])], wcol)
        rows.append(DC.summarize(name, uv["W"].values, None, len(uv), dict(tech="reference" if wcol == "w_ref" else wcol)))
    sub_units(Bc["dev"] != "Alibaba", "decision units without Alibaba")
    sub_units(~Bc["gen"].isin(["Qwen2.5", "Qwen3"]), "decision units without Qwen2.5/Qwen3 (D unaudited)")
    sub_units(~Bc["f_synthetic"], "decision units without synthetic-data models (version 3 flag)")
    syn_p = Bc["uid"].isin(Up.loc[Up["syn_protocol"] == 1, "uid"])
    syn_f = Bc["uid"].isin(Up.loc[Up["syn_fixlist"] == 1, "uid"])
    sub_units(~syn_p, "decision units without synthetic-flagged units (protocol rule)")
    sub_units(~syn_f, "decision units without synthetic-flagged units (fix-list families)")
    sub_units(Bc["tier"] == 0, "decision units: models outside tier windows")
    # no token audit (ra2 inputs) for comparison
    B0c = X["B0"][X["B0"]["clean"]].copy()
    B0c["w_ref"] = B0c["uid"].map(X["L0"].set_index("uid")["w"])
    U0 = DC.units_budget(B0c, CODES, "budget")
    uv0 = DC.unit_values(B0c, U0, "w_ref")
    rows.append(DC.summarize("decision units, ra2 token counts (no audit)", uv0["W"].values, None, len(uv0), dict(tech="reference")))
    HL = pd.DataFrame(rows)
    C.tab(HL, "headline")
    # Table E3 (W24(a)): ra2's technology bootstrap on the audited token counts (model-level columns), the conventions
    # panel and the cleaning steps, recomputed here with ra2's own functions (ra2's files are not overwritten)
    bandtab = WG.band(L, exante)
    C.tab(AN.tech_dispersion(B, T, M, L, B["clean"]), "technologies_models")
    C.tab(AN.conventions(B, T, L, B["clean"]), "conventions")
    C.tab(AN.cleaning_table(B, L, bandtab), "cleaning")
    # by technology: decision-unit medians under each of the 32 ex-ante technologies
    tech_rows, unitW = [], {}
    for k in exante:
        lk = L[L["tech"] == k].set_index("uid")
        Bk = Bc.assign(w_k=Bc["uid"].map(lk["w"]))
        if Bk["w_k"].isna().any():
            continue
        uv = DC.unit_values(Bk, Up, "w_k")
        unitW[k] = uv.set_index("unit")["W"]
        uvf = DC.unit_values(Bk, U["family"], "w_k")
        tech_rows.append(dict(key=k, label=M.set_index("key").loc[k, "label"], median_s_models=float(np.median(C.share(Bk["w_k"]))),
                              median_s_units=float(np.median(C.share(uv["W"]))), share_gt1_units=float(np.mean(uv["W"] > 1)),
                              share_gt1_models=float(np.mean(Bk["w_k"] > 1)),
                              median_s_units_family56=float(np.median(C.share(uvf["W"]))),
                              median_s_point_units=float(np.median(C.share(uv.set_index("unit").loc[W2["point-identified units"], "W"])))))
    TD = pd.DataFrame(tech_rows)
    V1 = {"besi", "chin", "farseer", "farseer_emb", "farseer_q", "gadre_c4", "gadre_rp", "gadre_rw", "hoff", "meta_a2",
          "meta_a3", "olmo"}                         # technologies of version 1 (module m3_wedge's registry)
    TD["in_version1"] = TD["key"].isin(V1)
    TD["added_after_round1"] = ~TD["in_version1"]
    C.tab(TD, "technologies_units")
    UW = pd.DataFrame(unitW)                         # unit x technology
    UT = UT.merge(UW.min(axis=1).rename("w_band_ptmin"), left_on="unit", right_index=True) \
           .merge(UW.max(axis=1).rename("w_band_ptmax"), left_on="unit", right_index=True) \
           .merge(UW.apply(lambda r: (r > 1).all(), axis=1).rename("all_tech_gt1"), left_on="unit", right_index=True)
    # unit-level reference interval from the joint draws
    js_ref = DC.joint_stats(Bc, Up, LW_ref)
    ulist = list(dict.fromkeys(Bc.merge(Up, on="uid")["unit"]))
    ci = pd.DataFrame(dict(unit=ulist, w_ref_lo=np.percentile(js_ref["UW"], 2.5, axis=0),
                           w_ref_hi=np.percentile(js_ref["UW"], 97.5, axis=0)))
    js_lab = DC.joint_stats(Bc, Up, LW_lab)
    ci["w_lab1_lo"] = np.percentile(js_lab["UW"], 2.5, axis=0)
    ci["w_lab1_hi"] = np.percentile(js_lab["UW"], 97.5, axis=0)
    UT = UT.merge(ci, on="unit")
    UT["tier_any"] = UT["unit"].map(tier_unit).astype(int)
    UT56 = DC.unit_table(Bc, U["family"], SVT)
    # ------------------------------------------------------------------ 5. conduct
    C.log("5. conduct tests: model-level serving code, decision level, tiers, on-device, open-weight premium")
    Xm = Bc.copy()
    Xm["w"] = Xm["w_ref"]
    Xm["lnw"] = np.log(Xm["w_ref"])
    Xm["s"] = Xm["s_ref"]
    Xm["lnC"] = np.log(Xm["Cmp"])
    Xm["serve_developer"] = Xm["serve"].astype(int)
    Xm["ondevice"] = (Xm["deploy"] == "ondevice").astype(int)
    CM = SV.model_level_tests(Xm)
    UTc = UT.copy()
    UTc["lnw"] = np.log(UTc["w_ref"])
    UTc["s"] = UTc["s_ref"]
    UTc["lnC"] = np.log(UTc["C_total"])
    CD = SV.decision_level_tests(UTc)
    DES = SV.descriptive(Xm, UT)
    COMP = SV.serving_composition(Xm)
    Uu = SM.universe(X["d"], B)
    for model, fld, old, new, src, q in base.D_AUDIT:        # the token audit applied to the universe rows as well
        i = Uu.index[Uu["model"] == model]
        Uu.loc[i, fld] = new
    Uu["M"] = Uu["D"] / Uu["N"]
    Uu["Cmp"] = 6 * Uu["N"] * Uu["D"]
    Uu["N_nonemb"] = np.nan
    Lu = WG.wedge_long(Uu, T, [C.REF])
    _, Uc = AN.open_closed(Uu, Lu)
    OP = SV.open_premium_decisions(Uc, Bc, U["primary"])
    COND = pd.concat([CM, CD], ignore_index=True)
    C.tab(COND, "conduct")
    C.tab(OP, "open_premium")
    C.tab(DES, "serving_descriptive")
    C.tab(COMP, "serving_composition")
    # ------------------------------------------------------------------ 6. sign identification
    C.log("6. sign identification: wild bootstrap-t anchors, tilt allowance, weights, years, decision units")
    A, fits = SI.anchors(X)
    e = SI.e_range(A, X)
    S2_core = (0.40, 0.52)
    PB = SI.pi_models(Bc, A, e, S2_core)
    PB.to_csv(os.path.join(C.PROC, "pi_bounds_models.csv"), index=False)
    SH = [SI.shares(PB, Bc, U["primary"], "PI-1 (bootstrap-t anchors; e from paths and published laws)")]
    PBn = SI.pi_models(Bc, A, (-1.0, 1.0), S2_core)
    SH.append(SI.shares(PBn, Bc, U["primary"], "normal inputs only (e in [-1, 1])"))
    A_c, _ = SI.anchors(X, include_comma=True)
    SH.append(SI.shares(SI.pi_models(Bc, A_c, SI.e_range(A_c, X), S2_core), Bc, U["primary"], "PI-1 + Marin Comma anchor (5 budgets)"))
    A10, _ = SI.anchors(X, meta10=True)
    SH.append(SI.shares(SI.pi_models(Bc, A10, e, S2_core), Bc, U["primary"], "PI-1 with Meta's 10-budget anchor at 1e22"))
    e_wide = (min(A["e_lo"].min(), e[0]), max(A["e_hi"].max(), e[1]))
    SH.append(SI.shares(SI.pi_models(Bc, A, e_wide, S2_core), Bc, U["primary"], "PI-1, e widened to the path-slope 95% intervals"))
    # own-lab anchors (PI-3 analogue): Meta and Marin models use only their own anchors
    PB3 = []
    for _, r in Bc.iterrows():
        own = {"Meta": ["meta"], "Marin": ["marin_dclm", "marin_nemotron"], "DeepSeek": ["deepseek"]}.get(r["dev"])
        AA = A[A["anchor"].isin(own)] if own else A
        PB3.append(SI.pi_models(Bc[Bc["uid"] == r["uid"]], AA, e, S2_core))
    SH.append(SI.shares(pd.concat(PB3, ignore_index=True), Bc, U["primary"], "PI-3 own-lab anchors (Meta, Marin, DeepSeek)"))
    # Review addition (R2 Major 4.3: "give the 23 percent alongside"): PI-4 with the widened IsoFLOP anchors plus every
    # ex-ante technology's in-support M* (ra2's 'all' anchors, their own draws), e over all technologies' paths.
    A_ra2 = AN.pi_anchors(T, M, X["mf_tab"], X["mf_draws"])
    A4 = pd.concat([A, A_ra2[A_ra2["kind"] == "all"]], ignore_index=True)
    a_all = list(M.loc[M["in_set"], "a"]) + [(1 - v) / 2 for v in e]
    e_all = (1 - 2 * max(a_all), 1 - 2 * min(a_all))
    PB4 = AN.pi_bounds(Bc, pd.Series(True, index=Bc.index), A4, e_all, S2_core,
                       {"PI-4": dict(kinds=["iso", "all"], e=e_all)})
    SH.append(SI.shares(PB4, Bc, U["primary"], "PI-4 all ex-ante technologies as anchors (with bootstrap-t IsoFLOP anchors)"))
    H["e_all"] = list(e_all)
    SHR = pd.concat(SH, ignore_index=True)
    C.tab(SHR, "sign_identified")
    C.tab(A, "anchors")
    MAG = pd.concat([SI.magnitude(PB, {"k in [0.40, 0.52] (ra2; sigma* 0.66-0.71)": (0.40, 0.52),
                                       "k in [0.40, 0.667] (sigma* 0.60-0.714; top-budget drift)": (0.40, 1 / 0.6 - 1)},
                                  tau=t) for t, _ in SI.TAUS], ignore_index=True)
    MS = SI.mstar_grid(A, e)
    C.tab(MAG, "sign_magnitude")
    C.tab(MS, "mstar_bounds")
    H["e_range"] = list(e)
    H["e_wide"] = list(e_wide)
    # ------------------------------------------------------------------ 7. trend
    C.log("7. trend from 2023 on the clean sample; 32 technologies; leave-one-developer-out; pre-2023 robustness")
    TRr, _ = TR.by_year(Bc, U["primary"], "w_ref", "reference")
    TRl, _ = TR.by_year(Bc, U["primary"], "w_primary1", "lab-own (one rule) where available")
    trs = [TRr, TRl]
    for k in exante:
        lk = L[L["tech"] == k].set_index("uid")
        Bk = Bc.assign(w_k=Bc["uid"].map(lk["w"]))
        if Bk["w_k"].isna().any():
            continue
        t_, _ = TR.by_year(Bk, U["primary"], "w_k", k)
        trs.append(t_)
    TRall = pd.concat(trs, ignore_index=True)
    C.tab(TRall, "trend_by_tech")
    mono = []
    for k, g in TRall.groupby("tech"):
        mono.append(dict(tech=k, **{f"rise_{c}": TR.monotone(g, c) for c in ("median_s_models", "median_s_units", "s_agg_models", "s_agg_units")},
                         **{f"{c}_{y}": float(g.loc[g["year"] == y, c].iloc[0]) for c in ("median_s_units", "s_agg_units") for y in TR.YEARS}))
    MONO = pd.DataFrame(mono)
    C.tab(MONO, "trend_monotone")
    LODO = TR.leave_one_dev_out(Bc, U["primary"], "w_ref")
    C.tab(LODO, "trend_lodo")
    PRE, PREm = TR.pre2023(Uu, Lu, Bc, (T[C.REF].alpha + T[C.REF].beta) / 2)
    C.tab(PRE, "pre2023")
    C.tab(PREm, "pre2023_models")
    # ------------------------------------------------------------------ 8. second output
    C.log("8. second-output test: OLMo ladder, C4 loss vs task bits per byte (joint cell bootstrap, B = 399)")
    Nol = WG.n_used(Bc, "olmo")
    SOT, SOS, SOR, SOP, SOdf = SO.run(Nol, Bc["D"].values, B=399)
    C.tab(SOT, "second_output_technologies")
    C.tab(SOS, "second_output_summary")
    SOM = pd.DataFrame(dict(model=Bc["model"], M=Bc["M"], **{f"lnratio_{o}_{f}".replace(" ", "_"): v for (o, f), v in SOR.items()}))
    C.tab(SOM, "second_output_models")
    # ------------------------------------------------------------------ 9. post-training
    C.log("9. post-training compute bound")
    PTD = PT.disclosed()
    PTE = PT.effect(Bc, UT)
    C.tab(PTD, "posttrain_disclosed")
    C.tab(PTE, "posttrain_effect")
    # ------------------------------------------------------------------ 10. exhibits
    C.log("10. exhibits")
    import exhibits_rb2 as EX
    SIGN = dict(PB=PB, A=A, e=e)
    EXD = dict(Bc=Bc, UT=UT, U=U, FR=FR, SVT=SVT, HL=HL, LAB=LAB, LOT=LOT, TD=TD, UW=UW, COND=COND, OP=OP,
                      DES=DES, COMP=COMP, SHR=SHR, A=A, MAG=MAG, MS=MS, TRall=TRall, MONO=MONO, LODO=LODO, PRE=PRE,
                      PREm=PREm, SOT=SOT, SOS=SOS, PTD=PTD, PTE=PTE, L=L, exante=exante, LW_ref=LW_ref, LW_lab=LW_lab,
                      SIGN=SIGN, M=M, CODES=CODES, UT56=UT56)
    pd.to_pickle(EXD, os.path.join(C.PROC, "exhibits_cache.pkl"))
    EX.write_all(EXD)
    UT.to_csv(os.path.join(C.PROC, "decision_units.csv"), index=False)          # 49 budget-level units (primary)
    UT56.to_csv(os.path.join(C.PROC, "decision_units_family56.csv"), index=False)  # version 3's 56 family-label units
    Bc.drop(columns=[c for c in Bc.columns if c.startswith(("w_", "wlo_", "whi_")) and c not in
                     ("w_ref", "w_ref_lo", "w_ref_hi", "w_primary", "w_primary1", "w_lab1", "w_lab1_lo", "w_lab1_hi")]).to_csv(
        os.path.join(C.PROC, "clean_models.csv"), index=False)
    H.update(EX.headline(dict(Bc=Bc, UT=UT, HL=HL, LAB=LAB, COND=COND, OP=OP, DES=DES, COMP=COMP, SHR=SHR, A=A, MAG=MAG,
                              MS=MS, TRall=TRall, MONO=MONO, LODO=LODO, PRE=PRE, SOS=SOS, PTE=PTE, FR=FR, SVT=SVT, LOM=LOM)))
    H["runtime_s"] = time.time() - t0
    with open(os.path.join(C.PROC, "headline.json"), "w") as f:
        json.dump(H, f, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    C.log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
