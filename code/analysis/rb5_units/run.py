"""run.py -- module rb5_units (round 3, WP4b analysis): budget-level decision units as the primary unit, readings coded
under paper/notes/rb5_reading_protocol.md, identification classes under the virtual value of compactness, the serving
test's randomization inference, and the trend with developer-cluster inference, mixture-of-experts, synthetic-data and
stated-rationale checks. Fix list items W1, W2, W14, W18 (numbers), W19 (analysis), W24 (numbers).

Order of work (each step reads the previous step's files):
  1. frame.py / verify_readings.py   coding frame and coder A's verified codes (readings_coderA.csv)
  2. code/analysis/rb2_decisions/run.py   re-run of the decision-level analysis with these units (budget level primary)
  3. this file                        units_primary.csv, W2/W14/W18/W19 numbers, trend and rationale tables, paper numbers

  nice -n 10 .venv/bin/python code/analysis/rb5_units/run.py [--B 9999]

CPU only, one process. Deterministic (fixed seeds).
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import r5common as C  # noqa: E402
import frame as FR  # noqa: E402
import verify_readings as VR  # noqa: E402
import readings as RD  # noqa: E402
import analyses as AN  # noqa: E402
import agreement as AG  # noqa: E402

B_BOOT = int(sys.argv[sys.argv.index("--B") + 1]) if "--B" in sys.argv else 9999
RB2T = lambda name: os.path.join(C.TABLES, f"rb2_decisions_{name}.csv")  # noqa: E731
NUM = []                                                                  # paper numbers


def num(item, quantity, value, fmt, source, note=""):
    NUM.append(dict(item=item, quantity=quantity, value=value,
                    formatted=fmt.format(value) if isinstance(value, (int, float, np.floating, np.integer)) and np.isfinite(value) else str(value),
                    source=source, note=note))


def f2(x):
    return "--" if x is None or not np.isfinite(x) else f"{x:.2f}".replace("-", "$-$")


def main():
    t0 = time.time()
    open(os.path.join(C.PROC, "run_log.txt"), "w").close()
    # ------------------------------------------------------------------ 1. frame and codes
    C.log("1. coding frame (from rb2's clean sample) and coder A's verified codes")
    Bm = pd.read_csv(os.path.join(C.PROC_RB2, "clean_models.csv"))
    F, _ = FR.build(Bm)
    F0 = pd.read_csv(os.path.join(C.PROC, "coding_frame.csv"))
    assert list(F["unit"]) == list(F0["unit"]) and list(F["D_T"]) == list(F0["D_T"]), \
        "coding frame changed since coding: re-code the changed units"
    A = VR.build_coderA()
    A.to_csv(os.path.join(C.PROC, "readings_coderA.csv"), index=False)
    K, Dz = AG.run()
    CODES = RD.load()
    codes_file = CODES["codes_file"].iloc[0]
    # ------------------------------------------------------------------ 2. rb2 outputs and the primary unit file
    C.log("2. rb2_decisions outputs; data/processed/rb5_units/units_primary.csv")
    UB = pd.read_csv(os.path.join(C.PROC_RB2, "units_budget.csv"))
    UT = pd.read_csv(os.path.join(C.PROC_RB2, "decision_units.csv"))
    assert UT["codes_file"].iloc[0] == codes_file, "rb2_decisions ran on other codes: re-run rb2_decisions first"
    assert len(UT) == 49 and UB["unit"].nunique() == 49
    HL = pd.read_csv(RB2T("headline"))
    M = Bm.merge(UB[["uid", "unit", "unit_type", "unit_kind", "reading", "id_class", "bound", "own_cap", "syn_protocol",
                     "syn_fixlist", "R_CO", "R_INF", "R_DEV", "deployment_rationale"]], on="uid")
    ut = UT.set_index("unit")
    M["W_unit"] = M["unit"].map(ut["w_ref"])
    M["s_unit"] = C.share(M["W_unit"])
    M["unit_year"] = M["unit"].map(ut["year"])
    M["C_unit"] = M["unit"].map(ut["C_total"])
    keep = ["uid", "unit", "unit_type", "reading", "bound", "unit_kind", "id_class", "own_cap", "syn_protocol",
            "syn_fixlist", "R_CO", "R_INF", "R_DEV", "deployment_rationale", "model", "gen", "dev", "date", "year", "N",
            "D", "M", "Cmp", "tier", "serve_model", "emb_share", "w_ref", "s_ref", "W_unit", "s_unit", "unit_year", "C_unit"]
    UP = M[keep].copy()
    UP["codes_file"] = codes_file
    UP.to_csv(os.path.join(C.PROC, "units_primary.csv"), index=False)
    U = UT.rename(columns={"w_ref": "W"}).copy()
    # the reference technology from rb2's cached registry (ra2common sets the import paths the pickle needs)
    sys.path.insert(0, os.path.join(C.ANALYSIS, "ra2_wedge"))
    from ra2common import log_w  # noqa: E402
    with open(os.path.join(C.PROC_RB2, "techs_cache.pkl"), "rb") as f:
        X = pickle.load(f)
    tref = X["T"]["chin_q"]

    def lnw_ref(N, D):
        return log_w(np.log(N), np.log(D), tref.alpha, tref.beta, tref.lnG)
    assert np.allclose(np.exp(lnw_ref(Bm["N"].values, Bm["D"].values)), Bm["w_ref"].values, rtol=1e-8)
    U["R_NONE"] = 1 - ((U["R_CO"] + U["R_INF"] + U["R_DEV"]) > 0).astype(int)

    # ------------------------------------------------------------------ 3. W1-W2 numbers
    C.log("3. W1-W2: unit counts, classes, headline and point-identified medians")

    def hl(stat, tech="reference"):
        g = HL[(HL["stat"] == stat) & (HL["tech"] == tech)]
        assert len(g) == 1, (stat, tech)
        return g.iloc[0]
    CL = "reference, budget+trunk cluster bootstrap (ra1's conservative scheme)"
    LO = "lab-own (one rule) where available"
    h = hl("decision units (primary: budget level)")
    hc = hl("decision units (primary: budget level)", CL)
    src = "output/tables/rb2_decisions_headline.csv"
    num("W5(b), W6, W21", "decision units (budget level)", int(h["n_units"]), "{:d}", src)
    num("W5(b)", "units that are families sharing one budget", int(h["n_family_units"]), "{:d}", src)
    ub = UT
    num("W1", "size-specific members (own units)", int((ub["unit_kind"] == "size-specific member").sum()), "{:d}",
        "data/processed/rb2_decisions/decision_units.csv")
    num("W1", "singletons", int((ub["unit_kind"] == "singleton").sum()), "{:d}", "data/processed/rb2_decisions/decision_units.csv")
    num("W6", "median w, decision units (reference)", float(h["median_w"]), "{:.2f}", src)
    num("W6", "median s, decision units (reference)", float(h["median_s"]), "{:.3f}", src)
    num("W6", "median s interval (joint wild bootstrap)", f"[{h['median_s_lo']:.3f}, {h['median_s_hi']:.3f}]", "{}", src)
    num("W6", "median s interval (budget+trunk cluster bootstrap)", f"[{hc['median_s_lo']:.3f}, {hc['median_s_hi']:.3f}]", "{}", src)
    num("W6, W20", "marginal value of compactness of the median decision, w - 1 (percent of training cost per percent of size)",
        float(h["median_w"]) - 1, "{:.2f}", src, "a model one percent smaller at the same loss is worth w - 1 percent of training cost")
    num("W6", "share of decisions over-trained (w > 1)", float(h["share_w_gt1"]), "{:.3f}", src)
    num("W6", "share over-trained interval (wild)", f"[{h['share_w_gt1_lo']:.3f}, {h['share_w_gt1_hi']:.3f}]", "{}", src)
    for stat, lab in [("models (77)", "median s over the 77 models"),
                      ("decision units (family label, version 3)", "median s over the 56 family-label units"),
                      ("decision units (members of cap-type budgets split)", "median s, cap-type budgets split into members"),
                      ("decision units (reading-consistent, version 3)", "median s, version 3's 66 reading-consistent units"),
                      ("decision units (primary, OLMo 2 1B separate)", "median s, OLMo 2 1B as its own unit")]:
        r = hl(stat)
        num("W6, W22(c)", f"{lab} (n = {int(r['n_units'])})", float(r["median_s"]), "{:.3f}", src,
            f"interval [{r['median_s_lo']:.3f}, {r['median_s_hi']:.3f}]")
    num("W16", "median s over all decisions, lab-own where available, reference elsewhere",
        float(hl("decision units (primary: budget level)", LO)["median_s"]), "{:.3f}", src)
    for key in ["point-identified units", "lower-bounded units", "lower-bounded units, cap-type budgets",
                "lower-bounded units, members with an own cap", "point-identified units if 'cap or choice' is a choice",
                "point-identified units under coder B's readings",                                   # [review]
                "point-identified units without MPT and Yi (weakest agreed choice codes)",          # [review]
                "point-identified units without synthetic-flagged units (protocol rule)",
                "point-identified units without synthetic-flagged units (fix-list families)",
                "units outside tier windows (interpretation check)", "point-identified units outside tier windows",
                "size-specific members and singletons (units of one model)",
                "size-specific members and singletons outside tier windows",
                "size-specific members and singletons outside tier windows, without synthetic-flagged (protocol)"]:
        r = hl(f"W2: {key}")
        num("W2, W5(b), W6", f"{key}: n", int(r["n_units"]), "{:d}", src)
        num("W2, W6", f"{key}: median s", float(r["median_s"]), "{:.3f}", src,
            f"wild interval [{r['median_s_lo']:.3f}, {r['median_s_hi']:.3f}]; median w {r['median_w']:.2f}")
        if key.startswith("point-identified units"):
            rc = hl(f"W2: {key}", CL)
            num("W2", f"{key}: cluster interval", f"[{rc['median_s_lo']:.3f}, {rc['median_s_hi']:.3f}]", "{}", src)
    for stat in ["decision units without Alibaba", "decision units without Qwen2.5/Qwen3 (D unaudited)",
                 "decision units without synthetic-data models (version 3 flag)",
                 "decision units without synthetic-flagged units (protocol rule)",
                 "decision units without synthetic-flagged units (fix-list families)", "decision units: models outside tier windows",
                 "decision units, ra2 token counts (no audit)"]:
        r = hl(stat)
        num("W23(i)", f"median s, {stat} (n = {int(r['n_units'])})", float(r["median_s"]), "{:.3f}", src)
    # [review] agreement between the two coders (protocol Section 11; agreement.py) and the resolution
    if K is not None:
        for _, r in K.iterrows():
            num("W1(c), W23(h)", f"coder agreement, {r['variable']} ({r['scope']} units, n = {int(r['n'])}): Cohen's kappa",
                float(r["kappa"]), "{:.2f}", "output/tables/rb5_units_agreement.csv", f"raw agreement {r['agreement']:.3f}")
    if codes_file == "readings_final.csv":
        RF = pd.read_csv(os.path.join(C.PROC, "readings_final.csv"))
        ch = RF[RF["resolved_fields"].fillna("") != ""]
        num("W1(c)", "unit-field disagreements resolved (units)", int(len(ch)), "{:d}", "data/processed/rb5_units/readings_final.csv",
            "; ".join(f"{u}: {f}" for u, f in zip(ch["unit"], ch["resolved_fields"])))
    # classification table
    cls = UT.groupby(["unit_kind", "id_class"]).size().rename("n").reset_index()
    C.tab(cls, "classes")
    rd = UT[UT["unit_type"] == "family"][["unit", "reading", "id_class", "s_ref", "w_ref", "w_ref_lo", "w_ref_hi"]]
    C.tab(rd, "common_budget_units")
    # ------------------------------------------------------------------ 4. W14: randomization inference
    C.log(f"4. W14: randomization inference for the serving coefficient (B = {B_BOOT})")
    Xm = Bm.copy()
    Xm["lnw"] = np.log(Xm["w_ref"])
    Xm["lnC"] = np.log(Xm["Cmp"])
    Ud = UT.copy()
    Ud["lnw"] = np.log(Ud["w_ref"])
    Ud["lnC"] = np.log(Ud["C_total"])
    cache_p = os.path.join(C.PROC, f"slow_cache_B{B_BOOT}.pkl")
    cache = pd.read_pickle(cache_p) if ("--reuse" in sys.argv and os.path.exists(cache_p)) else {}
    if cache and cache.get("codes_file") != codes_file:
        cache = {}
    ri = []
    for nb in (3, 2, 1) if "RI" not in cache else ():
        ri.append(AN.serving_ri(Xm, "serve_model", B=B_BOOT, n_bins=nb, label="models (77)"))
    if "RI" not in cache:
        ri.append(AN.serving_ri(Xm[Xm["dev"] != "Alibaba"], "serve_model", B=B_BOOT, n_bins=3, label="models without Alibaba"))
        ri.append(AN.serving_ri(Ud, "serve_any", B=B_BOOT, n_bins=3, label="decision units (49), any member served"))
        ri.append(AN.serving_ri(Ud, "serve_any", B=B_BOOT, n_bins=1, label="decision units (49), any member served"))
    RI = cache["RI"] if "RI" in cache else pd.DataFrame(ri)
    RID = pd.DataFrame([AN.serving_ri_dev(Xm, "serve_model", B=B_BOOT, weighted=True, label="models (77), developer level"),
                        AN.serving_ri_dev(Xm, "serve_model", B=B_BOOT, weighted=False, label="models (77), developer level"),
                        AN.serving_ri_dev(Xm[Xm["dev"] != "Alibaba"], "serve_model", B=B_BOOT, weighted=True,
                                          label="models without Alibaba, developer level")])
    C.tab(RID, "serving_ri_dev")
    for _, r in RID.iterrows():
        num("W14", f"developer-level RI p-value ({r['sample']}; {r['statistic']})", float(r["p_ri_beta_two"]), "{:.2f}",
            "output/tables/rb5_units_serving_ri_dev.csv", f"slope {r['coef']:.2f}; {int(r['G_treated'])} of {int(r['clusters'])} developers served a size")
    C.tab(RI, "serving_ri")
    COND = pd.read_csv(RB2T("conduct"))
    wcr = COND[(COND["level"] == "model") & (COND["code"] == "model-level code (primary)") & (COND["outcome"] == "lnw")].iloc[0]
    r0 = RI.iloc[0]
    assert abs(r0["coef"] - wcr["coef"]) < 1e-6, (r0["coef"], wcr["coef"])
    num("W14", "serving coefficient, models (ln w on the model-level code, ln C, year effects)", float(r0["coef"]), "{:.2f}",
        "output/tables/rb5_units_serving_ri.csv", f"CRV1 s.e. {r0['se_crv1']:.2f}; WCR p {wcr['p_wcr']:.2f} (rb2_decisions_conduct.csv)")
    num("W14", "treated developers", int(r0["G_treated"]), "{:d}", "output/tables/rb5_units_serving_ri.csv")
    num("W14", "randomization-inference p-value, coefficient (two-sided; year x compute-tercile strata)", float(r0["p_ri_beta_two"]),
        "{:.2f}", "output/tables/rb5_units_serving_ri.csv", f"t-statistic version {r0['p_ri_t_two']:.2f}; B = {B_BOOT}")
    for i in range(1, len(RI)):
        r = RI.iloc[i]
        num("W14, W23(l)", f"RI p-value, {r['sample']}, {r['strata']} (coef {r['coef']:.2f})", float(r["p_ri_beta_two"]), "{:.2f}",
            "output/tables/rb5_units_serving_ri.csv", f"RI-t {r['p_ri_t_two']:.2f}")
    for lv, sm, cd, lab in [("model", "clean sample without Alibaba", "model-level", "models without Alibaba"),
                            ("decision", "decision units (primary)", "any member served (model-level code)", "decisions (49), any member served")]:
        r = COND[(COND["level"] == lv) & (COND["sample"] == sm) & (COND["code"] == cd) & (COND["outcome"] == "lnw")].iloc[0]
        num("W14", f"serving coefficient, {lab}", float(r["coef"]), "{:.2f}", RB2T("conduct").split("scaling-laws-pf/")[-1],
            f"CRV1 s.e. {r['se_crv1']:.2f}; WCR p {r['p_wcr']:.2f}; treated {int(r['G_treated'])}")
    # ------------------------------------------------------------------ 5. W19: trend inference and checks
    C.log(f"5. W19: developer-cluster bootstrap (B = {B_BOOT}), within-developer, MoE, synthetic, rationale")
    Mm = Bm[["uid", "dev", "year", "w_ref", "Cmp", "tier"]].copy()
    pt, draws = cache["TBOOT"] if "TBOOT" in cache else AN.trend_boot(U, Mm, B=B_BOOT)
    TRr = pd.read_csv(RB2T("trend_by_tech"))
    ref = TRr[TRr["tech"] == "reference"].set_index("year")
    for y in AN.YEARS:
        assert abs(ref.loc[y, "s_agg_units"] - pt[f"s_agg_units_{y}"]) < 1e-9
        assert abs(ref.loc[y, "median_s_units"] - pt[f"median_s_units_{y}"]) < 1e-9
    keys = [f"{k}_{y}" for k in ("s_agg_units", "median_s_units", "s_agg_models", "median_s_models") for y in
            list(AN.YEARS) + ["d2324", "d2425", "d2325"]]
    TB = AN.boot_summary(pt, draws, keys)
    C.tab(TB, "trend_boot")
    for _, r in TB.iterrows():
        num("W19(i), W19 text (a)", f"trend: {r['stat']}", float(r["point"]), "{:.3f}", "output/tables/rb5_units_trend_boot.csv",
            f"developer-cluster bootstrap 95% [{r['lo']:.2f}, {r['hi']:.2f}]; share of draws <= 0: {r['share_le0']:.3f}")
    WD = cache["WD"] if "WD" in cache else AN.within_dev(U, B=min(B_BOOT, 4999))
    pd.to_pickle(dict(codes_file=codes_file, RI=RI, TBOOT=(pt, draws), WD=WD), cache_p)
    C.tab(WD, "within_dev")
    for _, r in WD.iterrows():
        num("W19(ii), W19 text (b)", f"annual growth of the unit wedge, {r['spec']}", float(r["growth_per_year"]), "{:.2f}",
            "output/tables/rb5_units_within_dev.csv", f"developer-cluster bootstrap [{r['growth_lo']:.2f}, {r['growth_hi']:.2f}]; "
            f"{int(r['n_developers_multi_year'])} developers with units in more than one year")
    # MoE: reference technology from rb2's cached registry (loaded in step 2)
    moe = pd.read_csv(os.path.join(C.TABLES, "ra2_wedge_models.csv"))
    moe = moe[moe["moe"].astype(bool) & moe["year"].between(2023, 2025)].copy()
    MO = AN.moe_rows(U, moe, lnw_ref)
    C.tab(MO, "moe")
    for _, r in MO.iterrows():
        num("W19(iii), W19 text (b) fifth", f"{r['convention']}, {r['year']}: compute-weighted share over units",
            float(r["s_agg_units"]), "{:.2f}", "output/tables/rb5_units_moe.csv",
            f"dense only {r['s_agg_units_dense']:.2f}; {int(r['n_moe'])} MoE models; MoE share of the year's compute {r['moe_compute_share']:.2f}")
    # synthetic flags
    syn_rows = []
    for lab, col in (("protocol rule", "syn_protocol"), ("fix-list families", "syn_fixlist")):
        for drop in (False, True):
            g = U[U[col] == 0] if drop else U
            st = AN.year_stats(g)
            syn_rows.append(dict(flag=lab, dropped=drop, n_units=len(g), n_flagged=int(U[col].sum()),
                                 **{k: v for k, v in st.items() if k.startswith(("s_agg_units", "median_s_units", "n_units"))}))
    SY = pd.DataFrame(syn_rows)
    C.tab(SY, "synthetic")
    for _, r in SY[SY["dropped"]].iterrows():
        num("W19(iv), W23(e)", f"2025 compute-weighted share without synthetic-flagged units ({r['flag']})", float(r["s_agg_units_2025"]),
            "{:.2f}", "output/tables/rb5_units_synthetic.csv",
            f"median {r['median_s_units_2025']:.2f}; 2023 {r['s_agg_units_2023']:.2f}, 2024 {r['s_agg_units_2024']:.2f}; "
            f"{int(r['n_flagged'])} units flagged; 2025 units kept {int(r['n_units_2025'])}")
    flagged = {lab: sorted(U.loc[U[col] == 1, "unit"]) for lab, col in (("protocol", "syn_protocol"), ("fixlist", "syn_fixlist"))}
    # rationale
    RA = AN.rationale_rows(U)
    C.tab(RA, "rationale")
    for _, r in RA.iterrows():
        y = r["year"]
        num("W19(v), W19 text (d)", f"share of decisions citing a compute-optimal target, {y}", float(r["share_R_CO"]), "{:.2f}",
            "output/tables/rb5_units_rationale.csv", f"compute-weighted {r['share_R_CO_cw']:.2f}; n = {int(r['n_units'])}")
        num("W19(v), W19 text (d)", f"share of decisions with a deployment rationale, {y}", float(r["share_deployment_rationale"]),
            "{:.2f}", "output/tables/rb5_units_rationale.csv", f"compute-weighted {r['share_deployment_rationale_cw']:.2f}")
        num("W19(v), W19 text (d)", f"median s among decisions with a deployment rationale, {y}", float(r["median_s_dep"]), "{:.2f}",
            "output/tables/rb5_units_rationale.csv", f"n = {int(r['n_dep'])}; compute-weighted {f2(r['s_agg_dep'])}; "
            f"without a deployment rationale: median {r['median_s_nodep']:.2f} (n = {int(r['n_nodep'])}); no rationale stated: median {r['median_s_none']:.2f} (n = {int(r['n_none'])})")
    # tier-window compute share and composition
    for y in AN.YEARS:
        g = Mm[Mm["year"] == y]
        num("W19 text (b) fifth", f"share of {y} clean-sample compute in tier windows", float((g["Cmp"] * g["tier"]).sum() / g["Cmp"].sum()),
            "{:.2f}", "data/processed/rb2_decisions/clean_models.csv")
    LODO = pd.read_csv(RB2T("trend_lodo"))
    for dev in ("Alibaba", "Hugging Face", "Marin"):
        g = LODO[LODO["dropped"] == dev].set_index("year")
        num("W19 text (b) third", f"median decision share 2024 -> 2025 without {dev}",
            f"{g.loc[2024, 'median_s_units']:.2f} -> {g.loc[2025, 'median_s_units']:.2f}", "{}", "output/tables/rb2_decisions_trend_lodo.csv")
    lodo_rise = [bool(np.all(np.diff(g.sort_values("year")["s_agg_units"].values) > 0)) for _, g in LODO.groupby("dropped")]
    num("W19", "leave-one-developer-out samples in which the compute-weighted unit share rises in both steps",
        f"{sum(lodo_rise)}/{len(lodo_rise)}", "{}", "output/tables/rb2_decisions_trend_lodo.csv")
    MONO = pd.read_csv(RB2T("trend_monotone"))
    mo = MONO[~MONO["tech"].isin(["reference", "lab-own (one rule) where available"])]
    for c in ("rise_s_agg_units", "rise_median_s_units", "rise_s_agg_models", "rise_median_s_models"):
        num("W19", f"technologies (of {len(mo)}) under which {c[5:]} rises in both steps", int(mo[c].sum()), "{:d}",
            "output/tables/rb2_decisions_trend_monotone.csv")
    for y in AN.YEARS:
        num("W19", f"median M, models, {y}", float(ref.loc[y, "median_M"]), "{:,.0f}", "output/tables/rb2_decisions_trend_by_tech.csv")
        num("W19", f"n decision units, {y}", int(ref.loc[y, "n_units"]), "{:d}", "output/tables/rb2_decisions_trend_by_tech.csv")
    num("W19", "share of 2024 compute in its largest model", float(ref.loc[2024, "top_share_C"]), "{:.2f}",
        "output/tables/rb2_decisions_trend_by_tech.csv", str(ref.loc[2024, "top_model"]))
    SHR = pd.read_csv(RB2T("sign_identified"))
    g = SHR[SHR["set"].str.startswith("PI-1 (boot") & (SHR["level"] == "model") & (np.isclose(SHR["tau"], 1.0))].set_index("year")
    num("W19 text (b) second", "share of 2023 compute identified as over-trained (PI-1, version 3's anchors; S1 replaces it)",
        float(g.loc["2023", "share_identified_cw"]), "{:.2f}", "output/tables/rb2_decisions_sign_identified.csv",
        "placeholder until WP4a's S1 by-year shares")
    PRE = pd.read_csv(RB2T("pre2023"))
    for _, r in PRE.iterrows():
        num("W19(e)", f"pre-2023 compute-weighted share: {r['sample'][:40]} | {r['technology'][:45]}", float(r["s_agg_trunc"]), "{:.2f}",
            "output/tables/rb2_decisions_pre2023.csv")
    # W16(a) (light re-run): the lab-own set without Llama 1 and 2, whose 2023 allocations predate Meta's 2024 path
    LOM = pd.read_csv(RB2T("labown_models"))
    old_meta = LOM["model"].str.match(r"^(llama-|Llama-2-)")
    lab_rows = []
    for lab, g in (("all lab-own models", LOM), ("without Llama 1 and 2", LOM[~old_meta]),
                   ("without Llama 1 and 2 and without AI2", LOM[~old_meta & (LOM["dev"] != "AI2")]),
                   ("without AI2 (version 3 set)", LOM[LOM["dev"] != "AI2"])):
        lab_rows.append(dict(set=lab, n=len(g), median_s_lab_one_rule=AN.med_s(g["w_lab1"]), median_s_reference=AN.med_s(g["w_ref"])))
    LBR = pd.DataFrame(lab_rows)
    C.tab(LBR, "labown_rerun")
    for _, r in LBR.iterrows():
        num("W16(a)-(b)", f"lab-own technologies, {r['set']} (n = {int(r['n'])}): median s one rule / reference",
            f"{r['median_s_lab_one_rule']:.2f} / {r['median_s_reference']:.2f}", "{}", "output/tables/rb5_units_labown_rerun.csv")
    wl = M["uid"].map(LOM.set_index("uid")["w_lab1"])
    wl[M["model"].str.match(r"^(llama-|Llama-2-)")] = np.nan
    M["w_lab_noL12"] = wl.fillna(M["w_ref"])
    Wl = M.groupby("unit").apply(lambda g: 1 / np.sum((g["Cmp"] / g["Cmp"].sum()) / g["w_lab_noL12"]) if g["unit_type"].iloc[0] == "family"
                                 else g["w_lab_noL12"].iloc[0])
    num("W16(b)", "median s over the 49 decisions, lab-own where available (Llama 1 and 2 excluded), reference elsewhere",
        AN.med_s(Wl.values), "{:.3f}", "output/tables/rb2_decisions_labown_models.csv",
        "with Llama 1 and 2 on Meta's path: 0.717 (rb2_decisions_headline.csv)")
    # ------------------------------------------------------------------ 6. W18: post-training scenario numbers
    C.log("6. W18: post-training scenarios")
    PTE = pd.read_csv(RB2T("posttrain_effect"))
    PTD = pd.read_csv(RB2T("posttrain_disclosed"))
    base = PTE[np.isclose(PTE["x_P"], 0.0)].iloc[0]
    p32 = PTE[np.isclose(PTE["x_P"], 0.32)].iloc[0]
    for _, r in PTD.iterrows():
        num("W18(a), W23(q)", f"x_P disclosed: {r['budget']}", float(r["x_P"]), "{:.4f}", "output/tables/rb2_decisions_posttrain_disclosed.csv")
    num("W18(b)", "median model share at x_P = 0", float(base["median_s_models"]), "{:.3f}", "output/tables/rb2_decisions_posttrain_effect.csv")
    num("W18(b)", "median model share at x_P = 0.32", float(p32["median_s_models"]), "{:.3f}", "output/tables/rb2_decisions_posttrain_effect.csv")
    num("W18(b)", "fall of the median decision share at x_P = 0.32", float(base["median_s_units"] - p32["median_s_units"]), "{:.3f}",
        "output/tables/rb2_decisions_posttrain_effect.csv", f"{base['median_s_units']:.3f} -> {p32['median_s_units']:.3f}")
    num("W19(b) fourth", "2025 compute-weighted share over models at x_P = 0.32, m/(1+m)", float(p32["s_agg_2025_mm"]), "{:.2f}",
        "output/tables/rb2_decisions_posttrain_effect.csv", f"x_P = 0: {base['s_agg_2025_mm']:.2f}")
    num("W19(b) fourth", "2025 compute-weighted share over models at x_P = 0.32, post-training in the cost base", float(p32["s_agg_2025"]),
        "{:.2f}", "output/tables/rb2_decisions_posttrain_effect.csv")
    num("W18", "share with m_N > 0 at x_P = 0.32 (models)", float(p32["share_mN_gt0_models"]), "{:.3f}", "output/tables/rb2_decisions_posttrain_effect.csv")
    # ------------------------------------------------------------------ 7. W24: data corrections and Table E3/E4 numbers
    C.log("7. W24: token audit, Table E3/E4 on audited counts, repetition robustness")
    DA = pd.read_csv(os.path.join(C.PROC_RB2, "d_audit.csv"))
    mp = Bm[Bm["model"] == "mpt-30b"].iloc[0]
    num("W24(f)", "MPT-30B tokens (T)", float(mp["D"]) / 1e12, "{:.2f}", "data/processed/rb2_decisions/d_audit.csv")
    num("W24(f)", "MPT-30B w (reference)", float(mp["w_ref"]), "{:.2f}", "data/processed/rb2_decisions/clean_models.csv",
        "was 1.198 at 1.00T")
    num("W24(f)", "MPT family share s_f (reference)", float(ut.loc["MPT", "s_ref"]), "{:.3f}", "data/processed/rb2_decisions/decision_units.csv",
        "was 0.24 in version 3")
    TDm = pd.read_csv(RB2T("technologies_models"))
    ins = TDm[TDm["in_set"]]
    num("W23(f), W24(a)", "range of median s over models, 32 technologies (audited)", f"{ins['med_s'].min():.2f} to {ins['med_s'].max():.2f}",
        "{}", "output/tables/rb2_decisions_technologies_models.csv")
    TDu = pd.read_csv(RB2T("technologies_units"))
    num("W7, W23(f)", "range of median s over the 49 decision units, 32 technologies", f"{TDu['median_s_units'].min():.2f} to {TDu['median_s_units'].max():.2f}",
        "{}", "output/tables/rb2_decisions_technologies_units.csv")
    num("W7", "range of the share of decision units with w > 1, 32 technologies",
        f"{TDu['share_gt1_units'].min():.2f} to {TDu['share_gt1_units'].max():.2f}", "{}", "output/tables/rb2_decisions_technologies_units.csv")
    sets = {"reference, Farseer kappa free, DeepSeek law, developers' paths with model-free curvature":
            ["chin_q", "farseer_q", "deepseek", "meta_mf", "marin_comma_mf", "marin_dclm_mf", "marin_nemotron_mf"],
            "kappa-free fits of the smaller designs (Gadre, OLMo ladder, Muennighoff)":
            ["gadre_c4_q", "gadre_rp_q", "gadre_rw_q", "olmo_q", "muen_q"]}
    for lab, ks in sets.items():
        g = ins[ins["key"].isin(ks)]
        gu = TDu[TDu["key"].isin(ks)]
        num("W23(f), W7", f"median s over models, {lab} (keys found: {len(g)})",
            f"{g['med_s'].min():.2f} to {g['med_s'].max():.2f}" if len(g) else "n/a", "{}",
            "output/tables/rb2_decisions_technologies_models.csv",
            f"over decision units {gu['median_s_units'].min():.2f} to {gu['median_s_units'].max():.2f}" if len(gu) else "")
    CT = pd.read_csv(RB2T("cleaning"))
    h_ = CT[CT["step"] == "h_D_undocumented"].iloc[0]
    num("W24(c), W23(e)", "all 32 technologies put w > 1 for this share of clean models (audited)", float(h_["share_allpoints_gt1"]), "{:.3f}",
        "output/tables/rb2_decisions_cleaning.csv", f"union of 95% intervals above one: {h_['share_band_gt1']:.3f}")
    CV = pd.read_csv(RB2T("conventions")).set_index("tech")
    for k in ("farseer_emb", "minicpm"):
        r = CV.loc[k]
        num("W24(b)", f"Table E4 row {k}: median w / headFLOP w / share w>1 / median s",
            f"{r['median_w']:.2f} / {r['median_w_headFLOP']:.2f} / {r['share_w_gt1']:.2f} / {r['median_s']:.2f}", "{}",
            "output/tables/rb2_decisions_conventions.csv")
    # repetition robustness (W24(e); R2 minor 16): Muennighoff et al.'s effective data
    REP = {"stablelm-3b-4e1t": (1.0e12, "1T unique tokens, 4 epochs (StableLM README)"),
           "TinyLlama_v1.1": (0.95e12, "about 950B unique tokens (TinyLlama README); 2T trained")}
    rep_rows = []
    for m_, (u_, why) in REP.items():
        r = Bm[Bm["model"] == m_].iloc[0]
        De = AN.muennighoff_D(u_, r["D"])
        we = float(np.exp(lnw_ref(np.array([r["N"]]), np.array([De])))[0])
        rep_rows.append(dict(model=m_, N=r["N"], D=r["D"], U=u_, D_eff=De, w_ref=r["w_ref"], w_eff=we,
                             s_ref=C.share(r["w_ref"]), s_eff=C.share(we), note=why))
    RP = pd.DataFrame(rep_rows)
    Ue = U.set_index("unit").copy()
    for _, r in RP.iterrows():
        Ue.loc[Bm.loc[Bm["model"] == r["model"], "uid"].iloc[0], "W"] = r["w_eff"]
    RP.attrs["median_units"] = AN.med_s(Ue["W"])
    C.tab(RP, "repetition")
    for _, r in RP.iterrows():
        num("W24(e)", f"repetition: {r['model']} w with Muennighoff effective data", float(r["w_eff"]), "{:.2f}",
            "output/tables/rb5_units_repetition.csv", f"D_eff {r['D_eff'] / 1e12:.2f}T; w at D {r['w_ref']:.2f}; s {r['s_ref']:.2f} -> {r['s_eff']:.2f}")
    num("W24(e)", "median s over decision units with both repeated-data models at effective data", AN.med_s(Ue["W"]), "{:.3f}",
        "output/tables/rb5_units_repetition.csv")
    # reference-level sensitivity (W24(e); R2 minor 7): curvature of the kappa-free fit to Chinchilla's 137 profile runs
    # (rb4_chinflop_param.csv: 0.656 in N_F, 0.667 in T) with the reference zero points held fixed, so every ln w scales
    # by k(sigma*)/k_ref with k = 1/sigma* - 1; family shares recomputed from the scaled member wedges
    k_ref = (tref.alpha + tref.beta) / 2
    P4 = pd.read_csv(os.path.join(C.TABLES, "rb4_chinflop_param.csv"))
    lev = []
    for _, pr in P4.iterrows():
        sc = (1 / pr["sigma_kappa"] - 1) / k_ref
        Mx = M.copy()
        Mx["w_s"] = np.exp(sc * np.log(Mx["w_ref"]))
        W = Mx.groupby("unit").apply(lambda g: 1 / np.sum((g["Cmp"] / g["Cmp"].sum()) / g["w_s"]) if g["unit_type"].iloc[0] == "family"
                                     else g["w_s"].iloc[0])
        lev.append(dict(convention=pr["convention"], sigma_kappa=pr["sigma_kappa"], scale=sc, median_s_units=AN.med_s(W.values),
                        median_s_models=AN.med_s(Mx["w_s"].values), share_gt1_units=float(np.mean(W.values > 1))))
    LEV = pd.DataFrame(lev)
    C.tab(LEV, "level_sensitivity")
    for _, r in LEV.iterrows():
        num("W24(e)", f"median decision share with sigma* {r['sigma_kappa']:.3f} (kappa-free fit, 137 Chinchilla profile runs, {r['convention']}), zero points fixed",
            float(r["median_s_units"]), "{:.3f}", "output/tables/rb5_units_level_sensitivity.csv",
            f"reference sigma* {1 / (1 + k_ref):.3f}; models {r['median_s_models']:.3f}")
    # ------------------------------------------------------------------ 8. exhibits
    C.log("8. exhibits: trend table, rationale table")
    write_tables(TB, WD, MO, SY, RA, pt, codes_file)
    # ------------------------------------------------------------------ 9. paper numbers and headline json
    NUMS = pd.DataFrame(NUM)
    # [review] randomization-inference p-values below 0.001 print as "< 0.001" (not "0.00"): with B = 9,999 permutations
    # the smallest attainable p-value is about 1/(B + 1) = 0.0001
    is_ri = NUMS["quantity"].str.contains("RI p-value|randomization-inference p-value")
    small = is_ri & pd.to_numeric(NUMS["value"], errors="coerce").lt(0.001)
    NUMS.loc[small, "formatted"] = "< 0.001"
    NUMS.to_csv(os.path.join(C.TABLES, "rb5_units_paper_numbers.csv"), index=False)
    Hj = dict(codes_file=codes_file, B=B_BOOT, runtime_s=time.time() - t0, synthetic_flagged_units=flagged,
              agreement=None if K is None else K.to_dict("records"))
    with open(os.path.join(C.PROC, "headline.json"), "w") as f:
        json.dump(Hj, f, indent=1, default=str)
    C.log(f"done in {time.time() - t0:.0f}s; {len(NUMS)} paper numbers")


def write_tables(TB, WD, MO, SY, RA, pt, codes_file):
    tb = TB.set_index("stat")

    def cell(k, p=2):
        r = tb.loc[k]
        return f"{f2(r['point'])} & {{\\scriptsize[{f2(r['lo'])}, {f2(r['hi'])}]}}"
    body = ["\\multicolumn{7}{@{}l}{\\textit{Panel A. By release year}} \\\\[1pt]"]
    for lab, k in (("Weighted $s$, decision units", "s_agg_units"), ("Median $s$, decision units", "median_s_units"),
                   ("Weighted $s$, models", "s_agg_models"), ("Median $s$, models", "median_s_models")):
        body.append(f"{lab} & " + " & ".join(cell(f"{k}_{y}") for y in AN.YEARS) + " \\\\")
    body.append("\\addlinespace")
    body.append("\\multicolumn{7}{@{}l}{\\textit{Panel B. Changes}} \\\\[1pt]")
    body.append("& \\multicolumn{2}{c}{2023 to 2024} & \\multicolumn{2}{c}{2024 to 2025} & \\multicolumn{2}{c}{2023 to 2025} \\\\")
    for lab, k in (("Weighted $s$, decision units", "s_agg_units"), ("Median $s$, decision units", "median_s_units"),
                   ("Weighted $s$, models", "s_agg_models"), ("Median $s$, models", "median_s_models")):
        body.append(f"{lab} & " + " & ".join(cell(f"{k}_{d}") for d in ("d2324", "d2425", "d2325")) + " \\\\")
    body.append("\\addlinespace")
    body.append("\\multicolumn{7}{@{}l}{\\textit{Panel C. Compute-weighted $s$ over decision units: checks}} \\\\[1pt]")
    body.append("& \\multicolumn{2}{c}{2024} & \\multicolumn{2}{c}{2025} & & \\\\")
    body.append(f"Dense decision units (Panel A) & {f2(pt['s_agg_units_2024'])} & & {f2(pt['s_agg_units_2025'])} & & & \\\\")
    for conv in ("active", "total"):
        g = MO[MO["convention"].str.contains(conv)].set_index("year")
        body.append(f"MoE models added, {conv} $N$ & {f2(g.loc[2024, 's_agg_units'])} & & "
                    f"{f2(g.loc[2025, 's_agg_units'])} & & & \\\\")
    for lab, fl in (("Without synthetic-data units (protocol)", "protocol rule"),
                    ("Without synthetic-data units (family list)", "fix-list families")):
        r = SY[(SY["flag"] == fl) & SY["dropped"]].iloc[0]
        body.append(f"{lab} & {f2(r['s_agg_units_2024'])} & & {f2(r['s_agg_units_2025'])} & & & \\\\")
    body.append("\\addlinespace")
    body.append("\\multicolumn{7}{@{}l}{\\textit{Panel D. Annual growth of the decision-unit wedge}} \\\\[1pt]")
    for _, r in WD.iterrows():
        body.append(f"{r['spec'][0].upper() + r['spec'][1:]} & {r['growth_per_year']:.2f} & {{\\scriptsize[{r['growth_lo']:.2f}, "
                    f"{r['growth_hi']:.2f}]}} & & & & \\\\")
    header = ("& \\multicolumn{2}{c}{2023} & \\multicolumn{2}{c}{2024} & \\multicolumn{2}{c}{2025} \\\\\n"
              "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(l){6-7}\n"
              "& Point & {\\scriptsize[95\\%]} & Point & {\\scriptsize[95\\%]} & Point & {\\scriptsize[95\\%]} \\\\")
    n = {y: int(pt[f"n_units_{y}"]) for y in AN.YEARS}
    nm = {y: int(pt[f"n_models_{y}"]) for y in AN.YEARS}
    notes = (f"Weighted: compute-weighted. Clean sample, 49 budget-level decision units ({n[2023]}, {n[2024]} and {n[2025]} in 2023, 2024 and 2025; "
             f"{nm[2023]}, {nm[2024]} and {nm[2025]} models), reference technology. Compute-weighted $s=\\sum(w^+-1)C/\\sum w^+C$, "
             "$w^+=\\max(w,1)$. Intervals: percentile intervals from a bootstrap that resamples the 18 developers with "
             f"replacement ($B={B_BOOT:,}$). Intervals in Panels A, B and D use this bootstrap. Panel C: the mixture-of-experts (MoE) models excluded from the clean sample enter as "
             "decision units of their own, with the reference wedge at active or at total parameters and weight "
             "$6N_{active}D$; synthetic-data flags coded under the reading protocol (Online Appendix~\\ref{app:wedge-family}); the family list drops SmolLM 1--3, phi-1.5, phi-2, Qwen2.5, Qwen3 and OLMo~2. Panel D: OLS of "
             "$\\ln\\hat W$ on the release date over decision units, pooled or with developer fixed effects, which "
             "identify the trend from developers with decisions in more than one year. --: under the protocol's rule "
             "every 2025 decision is flagged.")
    tex = (f"% generated by code/analysis/rb5_units/run.py (fix list W19); data: output/tables/rb5_units_trend_boot.csv, _moe.csv, "
           f"_synthetic.csv, _within_dev.csv\n\\begin{{table}}[tp]\n\\centering\n\\caption{{The Trend from 2023: Sampling Uncertainty and "
           f"Composition}}\n\\label{{tab:app-trend-boot}}\n\\footnotesize\n\\setlength{{\\tabcolsep}}{{2.2pt}}\n"
           f"\\begin{{tabular}}{{@{{}}lrlrlrl@{{}}}}\n\\toprule\n{header}\n\\midrule\n" + "\n".join(body) +
           f"\n\\bottomrule\n\\end{{tabular}}\n\\medskip\n\\begin{{tablenotes}}\n{notes}\n\\end{{tablenotes}}\n\\end{{table}}\n")
    with open(os.path.join(C.TABLES, "rb5_units_trend.tex"), "w") as f:
        f.write(tex)
    # rationale table
    ra = RA.set_index("year")
    rows = []
    for y in list(AN.YEARS) + ["all"]:
        r = ra.loc[y]
        rows.append(f"{y if y != 'all' else '2023--2025'} & {int(r['n_units'])} & {f2(r['share_R_CO'])} & {f2(r['share_R_CO_cw'])} & "
                    f"{f2(r['share_R_INF'])} & {f2(r['share_R_DEV'])} & {f2(r['share_deployment_rationale'])} & "
                    f"{f2(r['share_R_NONE'])} & {int(r['n_dep'])} & {f2(r['median_s_dep'])} & {f2(r['s_agg_dep'])} & "
                    f"{f2(r['median_s_none'])} & {f2(r['s_agg_none'])} \\\\")
    header = ("& & \\multicolumn{6}{c}{Share of decision units stating} & \\multicolumn{3}{c}{Deployment rationale} & "
              "\\multicolumn{2}{c}{None} \\\\\n\\cmidrule(lr){3-8}\\cmidrule(lr){9-11}\\cmidrule(l){12-13}\n"
              "Year & $n$ & C-opt. & C-opt., & Inference & Device or & Deploy- & None & $n$ & Median & Weighted & Median & Weighted \\\\\n"
              "& & & weighted & cost & memory & ment & & & $s$ & $s$ & $s$ & $s$ \\\\")
    notes = ("Stated sizing rationale of each of the 49 decision units, coded from the developers' saved reports, release "
             "posts and model cards under the written protocol (Online Appendix~\\ref{app:wedge-family}): C-opt., the size or token budget was set "
             "to be compute-optimal or by a scaling-law allocation; inference cost, the model was sized or trained longer to "
             "lower inference or serving cost, or its size is said to make inference cheaper or faster; device or memory, the "
             "size targets a device, GPU or memory limit, or on-device, local or edge use; deployment, either of the two; "
             "none, no rationale stated. A unit can state several. Weighted: by training compute. Median and weighted $s$ "
             "under the reference technology, among units with a deployment rationale and among units that state none of "
             "the three. " + ("Codes resolved from two coders, the second blind to the wedges (agreement in the replication "
                              "package)." if codes_file in ("readings_final.csv", "readings_resolved.csv") else
                              "Codes of the first coder; the blind second coding is pending."))
    tex = (f"% generated by code/analysis/rb5_units/run.py (fix list W19(v)); data: output/tables/rb5_units_rationale.csv\n"
           f"\\begin{{table}}[tp]\n\\centering\n\\caption{{Stated Sizing Rationales by Release Year}}\n\\label{{tab:app-rationale}}\n"
           f"\\footnotesize\n\\setlength{{\\tabcolsep}}{{2.5pt}}\n\\resizebox{{\\textwidth}}{{!}}{{%\n"
           f"\\begin{{tabular}}{{@{{}}lrrrrrrrrrrrr@{{}}}}\n\\toprule\n{header}\n\\midrule\n" + "\n".join(rows) +
           f"\n\\bottomrule\n\\end{{tabular}}}}\n\\medskip\n\\begin{{tablenotes}}\n{notes}\n\\end{{tablenotes}}\n\\end{{table}}\n")
    with open(os.path.join(C.TABLES, "rb5_units_rationale.tex"), "w") as f:
        f.write(tex)


if __name__ == "__main__":
    main()
