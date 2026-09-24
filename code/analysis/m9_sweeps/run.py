"""run.py -- module m9_sweeps: our controlled two-corpus experiment (FineWeb-Edu vs FineWeb), per the binding
pre-analysis plan paper/notes/m9_preanalysis_plan.md (committed 2026-09-24 03:12 before any estimation).

    .venv/bin/python code/analysis/m9_sweeps/run.py            # everything (power only if its table is missing)
    .venv/bin/python code/analysis/m9_sweeps/run.py --power    # also re-run the power calculation (about 1.5 h)

Works on whatever endpoints exist in data/processed/sweep/results.jsonl; every question skips gracefully when its
data are missing (logged, and marked 'pending' in the tables). Deterministic: every random draw derives from
m9_common.SEED via a stage key. CPU only, at most 4 worker processes; never imports MLX.

Stages: power (plan S4, design only) -> data checks & inventory -> Q1 curvature -> Q4 noise -> Q2 neutrality ->
Q3 extrapolation of the wedge -> Q5 learning rate -> Q6 functional dependence -> secondary (seed-covariance)
inference when seeds exist -> tables, figures, summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import m9_common as mc  # noqa: E402
import m9_est as est  # noqa: E402
import m9_stages as st  # noqa: E402


def data_checks(df):
    """Architecture formulas reproduce N and FLOP fields; bytes/token of the scored validation tokens."""
    a = mc.arch(df.d.values, df.L.values)
    bad = {k: int(np.sum(np.abs(a[k2] - df[k].values) > 0.5)) for k, k2 in
           (("N_nonemb", "N_nonemb"), ("N_total", "N_total"), ("flops_per_token", "fpt"))}
    bpt = mc.bytes_per_token_table()
    chk = dict(formula_mismatches=bad, bytes_per_token=bpt,
               C_equals_fpt_times_tokens=bool(np.allclose(df.C, df.flops_per_token * df.tokens)),
               n_endpoints=len(df), duplicate_record_keys=int((df.n_records > 1).sum()))
    return chk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--power", action="store_true", help="re-run the power calculation")
    ap.add_argument("--B", type=int, default=st.B_BOOT)
    ap.add_argument("--skip", default="", help="comma list of stages to skip (q1,q2,q3,q5,q6,secondary,figs)")
    ap.add_argument("--outputs-only", action="store_true", help="rebuild tables and figures from the saved CSVs")
    args = ap.parse_args()
    if args.outputs_only:
        return outputs_only()
    skip = set(filter(None, args.skip.split(",")))
    B = args.B
    t0 = time.time()
    mc.log(f"=== m9_sweeps run.py start (B = {B}, QUICK = {mc.QUICK}, output root = {mc._OUT})")
    S = {}   # summary for the memo

    # ---------------------------------------------------------------- power (design only; before any estimation)
    ppath = os.path.join(mc.TABLES, "m9_sweeps_power.csv")
    if args.power or not os.path.exists(ppath):
        import m9_power
        mc.log("power calculation (design only)")
        m9_power.run()
    power = pd.read_csv(ppath)
    cpath = os.path.join(mc.TABLES, "m9_sweeps_power_coverage.csv")
    if args.power or not os.path.exists(cpath):
        import m9_coverage
        mc.log("coverage of the plan's inference on the design (before estimation)")
        m9_coverage.run()
        m9_coverage.run_mf()
    coverage = pd.read_csv(cpath)

    # ---------------------------------------------------------------- data
    df = mc.load_results()
    S["checks"] = data_checks(df)
    mc.log("data:", len(df), "endpoints;", S["checks"]["formula_mismatches"], "formula mismatches")
    inv, extra = st.inventory(df)
    ds = st.design_stats(df)
    ce, mcells = st.cross_eval(df)
    allm = st.matched_cells(df)
    mc.write_csv(inv, "m9_sweeps_inventory")
    mc.write_csv(ds, "m9_sweeps_design")
    mc.write_csv(ce, "m9_sweeps_crosseval")
    mc.write_csv(allm, "m9_sweeps_matched_cells")
    dup = df[(df.tag == "hiM") & (df.d == 256) & (df.L == 4)]
    dups = []
    for _, r in dup.iterrows():
        m = df[(df.regime == r.regime) & (df.tag == "main") & (df.d == 256) & (df.D_target == r.D_target)]
        if len(m):
            dups.append(dict(regime=r.regime, D_target=r.D_target, loss_main=m[f"loss_{r.regime}"].values[0],
                             loss_hiM=r[f"loss_{r.regime}"], diff=r[f"loss_{r.regime}"] - m[f"loss_{r.regime}"].values[0]))
    # seedcorner seed 1 at (128, 200M) reproduces the seeds seed-1 trunk at the same cell
    for r in mc.CORPORA:
        a = df[(df.regime == r) & (df.tag == "seedcorner") & (df.d == 128) & (df.D_target == 200_000_000)]
        b = df[(df.regime == r) & (df.tag == "seeds") & (df.d == 128) & (df.seed == 1) & (df.D_target == 200_000_000)]
        if len(a) and len(b):
            dups.append(dict(regime=r, D_target=200_000_000, kind="seedcorner vs seeds (seed 1, width 128)",
                             loss_main=b[f"loss_{r}"].values[0], loss_hiM=a[f"loss_{r}"].values[0],
                             diff=a[f"loss_{r}"].values[0] - b[f"loss_{r}"].values[0]))
    S["hiM_duplicates"] = dups
    mc.write_csv(pd.DataFrame(dups), "m9_sweeps_duplicate_cells")
    S["inventory"] = inv.to_dict("records")

    # ---------------------------------------------------------------- Q1
    q1rows, q1lvl, q1fits, q1resid, q1corp = [], [], {}, {}, {}
    if "q1" not in skip:
        for conv in mc.CONVS:
            for sample in mc.SAMPLES:
                ts = time.time()
                rows, lvl, corp = st.q1_stage(df, conv, sample, B=B)
                q1rows += rows
                q1lvl += lvl
                if sample == "main" and corp:   # pre-declared size-corrected companion (memo D8)
                    rows2, lvl2, _ = st.q1_stage(df, conv, sample, scheme="cr2", B=B)
                    q1rows += rows2
                    q1lvl += lvl2
                if sample == "main":
                    for r, c in corp.items():
                        q1fits[(r, conv)] = c["fits"]["th"]
                        q1resid[(r, conv, "chin_huber")] = c["res_chin"]
                        q1resid[(r, conv, "kappa_huber")] = c["res_kappa"]
                        q1corp[(r, conv)] = dict(fits=c["fits"], sample=c["sample"], prep=c["prep"], cvtab=c["cvtab"],
                                                 point={k: v for k, v in c["point"].items()})
                mc.log(f"Q1 {conv}/{sample}: corpora {list(corp)} ({time.time() - ts:.0f}s)")
    q1 = pd.DataFrame(q1rows)
    mc.write_csv(q1, "m9_sweeps_q1_sigma")
    bws = []
    if "q1" not in skip:
        for conv in mc.CONVS:
            bws += st.mf_bandwidth_sensitivity(df, conv)
    mc.write_csv(pd.DataFrame(bws), "m9_sweeps_q1_bandwidth")
    mc.write_csv(pd.DataFrame(q1lvl), "m9_sweeps_q1_levels")

    # ---------------------------------------------------------------- Q4 (noise), needed for secondary inference
    q4, q4cells, noise = st.q4_stage(df, q1resid)
    mc.write_csv(q4, "m9_sweeps_q4_noise")
    mc.write_csv(q4cells, "m9_sweeps_q4_cells")
    S["noise"] = {k: list(v) for k, v in noise.items()}
    mc.log("Q4: seed noise estimated for", list(noise) or "no corpus (no seed replicates yet)")

    # ---------------------------------------------------------------- Q2
    q2rows, q2fits = [], {}
    if "q2" not in skip:
        for valset in mc.VALSETS:
            for conv in mc.CONVS:
                for sample in mc.SAMPLES:
                    res = st.q2_stage(df, conv, sample, valset, B=B)
                    if res is None:
                        mc.log(f"Q2 {valset}/{conv}/{sample}: skipped (needs >= {st.MIN_WIDTHS} widths and "
                               f">= {st.MIN_POINTS} main-grid endpoints evaluated on {valset} in BOTH corpora)")
                        continue
                    q2rows += res["rows"]
                    if sample == "main":
                        q2rows += st.q2_stage(df, conv, sample, valset, scheme="cr2", B=B)["rows"]
                    if sample == "main":
                        q2fits[(valset, conv)] = res["th"]
                    mc.log(f"Q2 {valset}/{conv}/{sample}: chi = {est.ce_tilt(res['th']):.3f}")
    q2 = pd.DataFrame(q2rows)
    mc.write_csv(q2, "m9_sweeps_q2_tilt")
    mc.write_csv(pd.DataFrame([dict(valset=k[0], conv=k[1], **{f"th{i}": v for i, v in enumerate(th)})
                               for k, th in q2fits.items()]), "m9_sweeps_q2_fits")
    dec = st.q2_decision(q2rows) if len(q2rows) else pd.DataFrame([dict(decision="pending: FineWeb main grid incomplete")])
    mc.write_csv(dec, "m9_sweeps_q2_decision")

    # ---------------------------------------------------------------- Q3
    q3rows, q3pts, q3fits = [], [], {}
    if "q3" not in skip:
        for r in mc.CORPORA:
            for conv in mc.CONVS:
                for sample in mc.SAMPLES:
                    ts = time.time()
                    res = st.q3_stage(df, r, conv, sample, B=B)
                    if res is None:
                        mc.log(f"Q3 {r}/{conv}/{sample}: skipped (insufficient data)")
                        continue
                    q3rows += res["rows"]
                    q3pts += res["points"]
                    q3fits[(r, conv, sample)] = res
                    if sample == "main":
                        res2 = st.q3_stage(df, r, conv, sample, scheme="cr2", B=B)
                        q3rows += res2["rows"]
                        q3pts += [dict(x, scheme="cr2") for x in res2["points"]]
                    sl_ = [x for x in res["rows"] if x["statistic"] == "slope|chin_r|local"]
                    mc.log(f"Q3 {r}/{conv}/{sample}: slope = {sl_[0]['estimate']:.3f} "
                           f"[{sl_[0]['ci_lo']:.3f}, {sl_[0]['ci_hi']:.3f}] ({time.time() - ts:.0f}s)" if sl_ else "")
    mc.write_csv(pd.DataFrame(q3rows), "m9_sweeps_q3_extrap")
    mc.write_csv(pd.DataFrame(q3pts), "m9_sweeps_q3_points")

    # ---------------------------------------------------------------- Q5
    q5 = st.q5_stage(df, q1fits) if "q5" not in skip else {}
    for k, v in q5.items():
        mc.write_csv(v, f"m9_sweeps_q5_{k}")

    # ---------------------------------------------------------------- Q6
    q6rows, q6prof = [], []
    if "q6" not in skip:
        for r in mc.CORPORA:
            for conv in mc.CONVS:
                res = st.q6_stage(df, r, conv)
                if res is None:
                    continue
                q6rows += res["rows"]
                q6prof += res["profiles"]
                mc.log(f"Q6 {r}/{conv}: done")
    mc.write_csv(pd.DataFrame(q6rows), "m9_sweeps_q6_onpath")
    mc.write_csv(pd.DataFrame(q6prof), "m9_sweeps_q6_profiles")

    # ---------------------------------------------------------------- secondary inference (seed covariance)
    sec_rows = []
    if noise and "secondary" not in skip:
        mc.log("secondary inference: parametric residual bootstrap with the seed-estimated within-trunk covariance")
        for conv in mc.CONVS:
            rows, _, corp = st.q1_stage(df, conv, "main", scheme="seedcov", noise=noise, B=B)
            sec_rows += [dict(question="Q1", **x) for x in rows]
            for valset in ("edu", "web"):
                if all(r in noise for r in mc.CORPORA):
                    res = st.q2_stage(df, conv, "main", valset, scheme="seedcov", noise=noise, B=B)
                    if res is not None:
                        sec_rows += [dict(question="Q2", **x) for x in res["rows"]]
            for r in mc.CORPORA:
                if r in noise:
                    res = st.q3_stage(df, r, conv, "main", scheme="seedcov", noise=noise, B=B)
                    if res is not None:
                        sec_rows += [dict(question="Q3", **x) for x in res["rows"]]
    else:
        mc.log("secondary inference: pending (no seed replicates yet)")
    sec = pd.DataFrame(sec_rows)
    mc.write_csv(sec, "m9_sweeps_secondary_seedcov")
    if len(sec) and len(q2rows):
        dec2 = st.q2_decision([x for x in sec_rows if x["question"] == "Q2"])
        mc.write_csv(dec2, "m9_sweeps_q2_decision_seedcov")

    # ---------------------------------------------------------------- outputs
    import m9_outputs as mo
    ctx = dict(df=df, power=power, coverage=coverage, inv=inv, ds=ds, ce=ce, q1=q1, q1lvl=pd.DataFrame(q1lvl), q1corp=q1corp, q2=q2,
               q2fits=q2fits, dec=dec, q3=pd.DataFrame(q3rows), q3pts=pd.DataFrame(q3pts), q3fits=q3fits, q4=q4,
               q4cells=q4cells, q5=q5, q6=pd.DataFrame(q6rows), q6prof=pd.DataFrame(q6prof), sec=sec, S=S)
    if "figs" not in skip:
        mo.all_figures(ctx)
    mo.all_tables(ctx)
    S["runtime_s"] = time.time() - t0
    json.dump(S, open(os.path.join(mc.PROC, "summary.json"), "w"), indent=1, default=str)
    mc.log(f"=== m9_sweeps done in {time.time() - t0:.0f}s")


def outputs_only():
    """Rebuild every table and figure from the CSVs written by the last full run (no estimation)."""
    import m9_outputs as mo
    T = lambda n: pd.read_csv(os.path.join(mc.TABLES, n + ".csv")) if os.path.exists(os.path.join(mc.TABLES, n + ".csv")) \
        and os.path.getsize(os.path.join(mc.TABLES, n + ".csv")) > 1 else pd.DataFrame()
    q2f = T("m9_sweeps_q2_fits")
    q2fits = {(r.valset, r.conv): np.array([r[f"th{i}"] for i in range(8)]) for _, r in q2f.iterrows()} if len(q2f) else {}
    q5 = {k: T(f"m9_sweeps_q5_{k}") for k in ("calibration", "corners", "drift", "abias")}
    ctx = dict(df=mc.load_results(), power=T("m9_sweeps_power"), coverage=T("m9_sweeps_power_coverage"),
               inv=T("m9_sweeps_inventory"), ds=T("m9_sweeps_design"), ce=T("m9_sweeps_crosseval"), q1=T("m9_sweeps_q1_sigma"),
               q1lvl=T("m9_sweeps_q1_levels"), q1corp={}, q2=T("m9_sweeps_q2_tilt"), q2fits=q2fits,
               dec=T("m9_sweeps_q2_decision"), q3=T("m9_sweeps_q3_extrap"), q3pts=T("m9_sweeps_q3_points"), q3fits={},
               q4=T("m9_sweeps_q4_noise"), q4cells=T("m9_sweeps_q4_cells"), q5=q5, q6=T("m9_sweeps_q6_onpath"),
               q6prof=T("m9_sweeps_q6_profiles"), sec=T("m9_sweeps_secondary_seedcov"), S={})
    mo.all_figures(ctx)
    mo.all_tables(ctx)
    mc.log("outputs rebuilt from CSVs")


if __name__ == "__main__":
    main()
