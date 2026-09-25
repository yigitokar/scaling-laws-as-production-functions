"""run.py -- module rb4_chinflop: Chinchilla's FLOP accounting rebuilt from Hoffmann et al.'s architecture table,
model-free sigma* in FLOP-effective parameters N_F = C/(6D), and its propagation into the study-level interval and
the sigma*(C) meta-regression (round-2 Referee 2, Major 1 request 4; round-1 Major 9(b)).

One command regenerates every output:
    .venv/bin/python code/analysis/rb4_chinflop/run.py
    .venv/bin/python code/analysis/rb4_chinflop/run.py --stages tables      # from the stage caches
Stages: arch (Table A9 counts, Appendix F FLOPs, Table A4 reproduction) -> estimate (matching, coordinate test,
eta by budget, ra1's estimator under 12 accounting variants + checks) -> param (same-run parametric sigma* in N_F)
-> propagate (rb1's study-level interval and meta-regression, old and rebuilt) -> tables (CSVs, LaTeX)
-> review (reproduction, validation and robustness checks).
Seeds: ra1's Chinchilla bootstrap seed for every accounting (common random numbers), ra1's parametric seed, rb1's
wild-bootstrap seeds; module seed rb4common.SEED for nothing else. CPU only, at most 4 worker processes.
Environment: RB4_QUICK=1 (tiny bootstraps), RB4_OUTPUT_ROOT (redirect outputs; used by the self-review re-run).
Stage caches (data/processed/rb4_chinflop/stage_*.pkl) hold derived statistics only (no digitized rows).
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import rb4common as cm  # noqa: E402

STAGES = ["arch", "estimate", "param", "propagate", "tables", "review"]


def cache(name):
    return os.path.join(cm.PROC, f"stage_{name}.pkl")


def save(name, obj):
    with open(cache(name), "wb") as f:
        pickle.dump(obj, f)


def load(name):
    with open(cache(name), "rb") as f:
        return pickle.load(f)


def stage_arch():
    import rb4_arch as ar
    cm.log("stage arch: Table A9 architectures, parameter counts, Appendix F FLOPs per token, Table A4 check")
    sha = ar.fetch_source()
    tc = ar.transcription_check()
    cm.log(f"  arXiv source sha256 {sha} (expected {cm.ARXIV_SHA256}: {sha == cm.ARXIV_SHA256}); transcription {tc}")
    A = ar.arch_table()
    a4, a4s = ar.table_a4_check()
    oth = ar.other_checks()
    for _, r in a4s.iterrows():
        cm.log(f"  Table A4 under {r.accounting} ({r.N_in_6ND}): {r.exact_2dp}/{r.n} ratios exact to 2 dp, "
               f"max |err| {r.max_abs_err:.3f}")
    cm.log(f"  Table A9 counts: max |T_arch/N_reported - 1| = {np.abs(A.T_err_rel).max():.4f}")
    R = dict(A=A, a4=a4, a4s=a4s, other=oth, sha=sha, transcription=tc)
    save("arch", R)
    return R


def stage_estimate():
    import rb4_estim as es
    cm.log("stage estimate: matching, coordinate test, eta by budget, model-free sigma* by accounting")
    df, A = es.load_runs()
    ms, mtot = es.match_summary(df, A)
    cm.log(f"  matching: {mtot}")
    co = es.coords_test(df)
    for _, r in co.iterrows():
        cm.log(f"  coordinates under {r.accounting:3s}: mean offset {r.mean_offset_dex:+.4f} dex, within-budget slope "
               f"{r.slope_resid_on_log10N:+.4f} (CR1 {r.se_cr1_budget:.4f}, p {r.p_cr1:.4f})")
    out, iso2, surf = es.run_variants(df)
    for k, r in out.items():
        if "summary" in r:
            s = r["summary"]
            cm.log(f"  {k:12s} k={s['k_valid']}: RE {s['sigma_re']:.4f} ({s['se_sigma_re']:.4f}); FE {s['sigma_fe']:.4f}; "
                   f"drift {s['drift_sigma_per_decade']:+.4f} ({s['se_drift_sigma']:.4f})")
    eta = es.eta_by_budget(iso2, out, A)
    fw = [es.fixed_membership(iso2, out, acc=a).assign(accounting=a) for a in ("T4", "A", "X", "6P", "6T", "ra1")]
    fw = pd.concat(fw, ignore_index=True)
    fq = es.window_fit_quality(iso2, out)
    ranges = es.design_ranges(iso2)
    # strip everything that could reproduce digitized rows before caching
    for r in out.values():
        r.pop("masks", None)
    R = dict(match=ms, match_tot=mtot, coords=co, out=out, eta=eta, fixedwin=fw, fitq=fq, ranges=ranges)
    save("estimate", R)
    return R


def stage_param():
    """Same-run parametric sigma* (Chinchilla form, kappa free; Huber; FHH wild bootstrap, ra1's code and seed) in the
    N_F (T4) convention, for the Table 1 row; the T convention is re-run as a reproduction check."""
    import rb4_estim as es
    cm.log("stage param: same-run parametric sigma* in N_F (T4) and T")
    df, A = es.load_runs()
    R = es.parametric(df)
    for r in R:
        cm.log(f"  {r['convention']}: kappa=1 {r['sigma_chin']:.4f} ({r['se_sigma_chin']:.4f}); kappa free "
               f"{r['sigma_kappa']:.4f} ({r['se_sigma_kappa']:.4f}); kappa {r['kappa']:.3f}")
    ref = es.reference_fit(df)
    for r in ref:
        cm.log(f"  reference technology, {r['convention']}: kappa=1 {r['sigma_chin']:.4f} (a {r['a_chin']:.3f}); kappa free "
               f"{r['sigma_kappa']:.4f} (kappa {r['kappa']:.3f}); M*(1e23) {r['Mstar_1e+23_chin']:.1f} / {r['Mstar_1e+23_kappa']:.1f}")
    P = pd.DataFrame(R)
    P.attrs["reference"] = ref
    save("param", dict(same_run=P, reference=pd.DataFrame(ref)))
    return R


def stage_propagate():
    import rb4_prop as pp
    E = load("estimate")
    cm.log("stage propagate: rb1's study-level interval and sigma*(C) meta-regression, old vs rebuilt")
    X = pp._load_extrap()
    if X is None:
        cm.log("  rb1 extrap cache absent: the local first-derivative row is skipped")
    old_V, old_conv = pp.study_level("old", X=X)
    old_M = pp.meta_regression("old", B=cm.B_META, full=True)
    res = {"old": dict(V=old_V, conv=old_conv, M=old_M)}
    alts = {}
    for key in ("NF_A|corr", "NF_X|corr", "NF_A", "NF_X", "P", "T", "P|corr", "T|corr", "NF_T4|6ND"):
        s = E["out"][key]["summary"]
        alts[key] = (key, s["sigma_re"], s["se_sigma_re"])
    for tag, key in (("new", "NF_T4"), ("alt_NF_X_corr", "NF_X|corr"), ("alt_NF_A_corr", "NF_A|corr"),
                     ("alt_P", "P"), ("alt_T", "T")):
        o = E["out"][key]
        pb, ps = pp.write_overrides(tag, o["rows"], o["summary"])
        V, conv = pp.study_level(tag, pb, ps, X=X, chin_alts=alts if tag == "new" else None)
        M = pp.meta_regression(tag, pb, ps, B=cm.B_META, full=(tag == "new"))
        res[tag] = dict(V=V, conv=conv, M=M, key=key)
        p = V.iloc[0]
        sl = M["slopes"]
        w = sl[(sl["sample"] == "all six designs (excl. Porian)") & (sl.spec_key == "weighted")].iloc[0]
        cm.log(f"  [{tag:14s}] study-level {p.mean_re:.4f} [{p.lo_hksj:.4f}, {p.hi_hksj:.4f}] tau {p.tau:.4f}; drift "
               f"{w.slope:+.4f} (CR2 {w.se_cr2_design:.4f}; wild p {w.p_wcr_design:.4f})")
    save("propagate", res)
    return res


def stage_tables():
    import rb4_tables as tb
    tb.make_all(load("arch"), load("estimate"), load("param"), load("propagate"))


def stage_review():
    import rb4_review as rv
    cm.log("stage review: reproduction, validation and robustness checks (memo, Self-review)")
    rv.run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", default=STAGES)
    a = ap.parse_args()
    cm.log(f"rb4_chinflop run: stages {a.stages}; QUICK={cm.QUICK}; output root {cm._OUT}; rb1 shadow root "
           f"{os.environ.get('RB1_OUTPUT_ROOT')}")
    for s in STAGES:
        if s in a.stages:
            globals()[f"stage_{s}"]()
    cm.log("done")
