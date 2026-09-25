"""rb4_prop.py -- propagate the rebuilt Chinchilla estimates into rb1's study-level interval and sigma*(C)
meta-regression, without editing rb1 or ra1.

Hook: rb1's meta_c reads its inputs through the module attributes rb1common.RA1_BUDGETS and rb1common.RA1_SUMMARY at
call time. We write override copies of ra1's two CSVs in which only the Chinchilla rows are replaced (derived
statistics: sigma*_b, s.e., window counts, design summaries), point those attributes at them, and call rb1's own
functions (study_level, run_meta, top_budget_values, add_local_fd_variant) with rb1's seeds. The baseline ('old') run
through the same hook reproduces rb1's published tables, which validates the hook.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

import rb4common as cm

cm1 = cm.cm1
RA1_BUDGETS_ORIG = cm1.RA1_BUDGETS
RA1_SUMMARY_ORIG = cm1.RA1_SUMMARY


def write_overrides(tag, rows, summary):
    """Override copies of ra1_modelfree_isoflop_budgets.csv / _summary.csv with Chinchilla replaced."""
    b = pd.read_csv(RA1_BUDGETS_ORIG)
    s = pd.read_csv(RA1_SUMMARY_ORIG)
    nb = pd.DataFrame(rows)
    nb = nb[[c for c in b.columns if c in nb.columns]]
    b2 = pd.concat([nb, b[b.design != "Chinchilla"]], ignore_index=True)[b.columns]
    srow = s[s.design == "Chinchilla"].iloc[0].to_dict()
    for k, v in summary.items():
        if k in srow:
            srow[k] = v
    srow["se_sigma_fe_within_only"] = np.nan          # not recomputed (ra1's within-only bootstrap)
    s2 = pd.concat([pd.DataFrame([srow]), s[s.design != "Chinchilla"]], ignore_index=True)[s.columns]
    pb = os.path.join(cm.PROC, f"override_{tag}_isoflop_budgets.csv")
    ps = os.path.join(cm.PROC, f"override_{tag}_isoflop_summary.csv")
    b2.to_csv(pb, index=False)
    s2.to_csv(ps, index=False)
    return pb, ps


def _point(pb=None, ps=None):
    cm1.RA1_BUDGETS = pb or RA1_BUDGETS_ORIG
    cm1.RA1_SUMMARY = ps or RA1_SUMMARY_ORIG


def _load_extrap():
    if not os.path.exists(cm.RB1_EXTRAP_PKL):
        return None
    with open(cm.RB1_EXTRAP_PKL, "rb") as f:
        return pickle.load(f)


def _summary_variant(ps, tag, chin=None, meta_eta=None):
    """Copy of a summary override with Chinchilla's (sigma_re, se_sigma_re) replaced and/or Llama 3's sigma_re
    shifted by a FLOP-accounting elasticity (rb1's formula)."""
    s = pd.read_csv(ps)
    if chin is not None:
        m = s.design == "Chinchilla"
        s.loc[m, "sigma_re"], s.loc[m, "se_sigma_re"] = chin
    if meta_eta is not None:
        m = s.design == "Llama 3"
        s.loc[m, "sigma_re"] = cm.shift_eta(float(s.loc[m, "sigma_re"].iloc[0]), meta_eta)
    path = os.path.join(cm.PROC, f"override_{tag}_isoflop_summary.csv")
    s.to_csv(path, index=False)
    return path


def study_level(tag, pb=None, ps=None, X=None, chin_alts=None, log=cm.log):
    """rb1's study_level (+ its review's local first-derivative row) on the pointed inputs, plus rows added here
    (each computed by rb1's own study_level on a further override of the summary; its primary row is kept):
    eta = +-0.1 for Meta only (Chinchilla's accounting is now observed), and Chinchilla under each rebuilt accounting
    (the other studies at their primary values)."""
    import meta_c as mc
    _point(pb, ps)
    V, conv = mc.study_level(log=lambda *a: None)
    if X is not None:
        V = mc.add_local_fd_variant(V, X)
    extra = []
    base_ps = ps or RA1_SUMMARY_ORIG
    for e in (-0.1, 0.1):
        p2 = _summary_variant(base_ps, f"{tag}_meta_eta{e:+.1f}", meta_eta=e)
        _point(pb, p2)
        r = mc.study_level(log=lambda *a: None)[0].iloc[0].to_dict()
        # [review] the label said "(Chinchilla rebuilt)" in every run, including the 'old' run (Chinchilla in T)
        chin = "Chinchilla in T, as published" if tag == "old" else "Chinchilla rebuilt"
        r["variant"] = f"[rb4] FLOP-accounting elasticity eta = {e:+.1f} for Meta only ({chin})"
        r["note"] = "sigma*_eff = 2/(2 + S/(1+eta)^2) applied to Llama 3 only"
        extra.append(r)
    for key, (lab, y, se) in (chin_alts or {}).items():
        p2 = _summary_variant(base_ps, f"{tag}_chin_{key.replace('|', '_')}", chin=(y, se))
        _point(pb, p2)
        r = mc.study_level(log=lambda *a: None)[0].iloc[0].to_dict()
        r["variant"] = f"[rb4] Chinchilla in accounting {key}: {lab}"
        r["note"] = "other studies at their primary values"
        extra.append(r)
    V = pd.concat([V, pd.DataFrame(extra)], ignore_index=True)
    V.insert(0, "run", tag)
    conv.insert(0, "run", tag)
    _point()
    return V, conv


def meta_regression(tag, pb=None, ps=None, B=cm.B_META, full=True, log=cm.log):
    """rb1's run_meta (full: every specification and subsample, rb1's seeds) and top-budget values."""
    import meta_c as mc
    _point(pb, ps)
    if full:
        R = mc.run_meta(log=lambda *a: None, B=B)
        top = mc.top_budget_values(R["data"])
        sl = R["slopes"].copy()
        pr = R["preds"].copy()
        het = R["het"].copy()
        out = dict(slopes=sl, preds=pr, top=top, het=het, data=R["data"])
        # [review] each design's largest bracketed budget and its BLUP design line (rb1's Online Appendix Table D,
        # Panel C), written to rb4_chinflop_design_top.csv
        out["tops"] = R["tops"].copy()
        out["tops"].insert(0, "run", tag)
        # downstream of the meta-regression (rb1's pid, unchanged): identified set beyond the designs and the
        # revealed-demand medians, with rb1 stage_pid's inputs
        import pid
        sig_top = float(top.iloc[0]["mean_fixed"])
        ci_top = (float(top.iloc[0]["lo_hksj"]), float(top.iloc[0]["hi_hksj"]))
        b_cl = float(sl[sl["sample"].str.startswith("Chinchilla + Llama") & (sl.spec_key == "fe_weighted")].slope.iloc[0])
        f = R["fits"]["weighted"]
        mu, b_pool = float(f["beta"][0]), float(f["beta"][1])
        slopes = {"drift of Chinchilla and Llama 3 (design FE, inverse variance)": b_cl,
                  "pooled drift (primary three-level meta-regression)": b_pool}
        out["pi"] = pid.pi_table(sig_top, slopes, sig_top_ci=ci_top)
        c, k_ref = pid.load_clean()
        pred = lambda lc: mu + b_pool * (np.asarray(lc, float) - 20.0)  # noqa: E731
        out["scen"] = pid.wedge_scenarios(c, k_ref, pred, sig_top, slopes, log=lambda *a: None, sig_top_hi=ci_top[1])
        out["pi"].insert(0, "run", tag)
        out["scen"].insert(0, "run", tag)
    else:
        d_all = mc.load_budgets()
        d = d_all[~d_all.porian].reset_index(drop=True)
        f = mc.fit_spec(d, "weighted")
        r = mc.slope_row(f, d, "3-level RE, inverse-variance (primary)", clusters_study=d["study"].values, B=B,
                         seed=cm1.seed_of("wcr|weighted"))
        r.update(sample="all six designs (excl. Porian)", spec_key="weighted")
        rows = [r]
        dd = d[d.design.isin(["Chinchilla", "Llama 3"])].reset_index(drop=True)
        f2 = mc.fit_spec(dd, "fe_weighted")
        r2 = mc.slope_row(f2, dd, "Design FE, inverse-variance", clusters_study=dd["study"].values, boot=False)
        r2.update(sample="Chinchilla + Llama 3 (the two designs reaching 1e21)", spec_key="fe_weighted")
        rows.append(r2)
        d3 = d[d.log10C <= np.log10(3e20) + 1e-9].reset_index(drop=True)
        f3 = mc.fit_spec(d3, "weighted")
        r3 = mc.slope_row(f3, d3, "3-level RE, inverse-variance (primary)", clusters_study=d3["study"].values,
                          B=max(999, B // 5), seed=cm1.seed_of("wcr|Budgets <= 3e20 (common window)|weighted"))
        r3.update(sample="Budgets <= 3e20 (common window)", spec_key="weighted")
        rows.append(r3)
        pr = mc.predict(f, d, [19.0, 20.0, 21.0, 22.0, 23.0, 24.0])
        pr["spec"], pr["spec_key"] = "3-level RE, inverse-variance (primary)", "weighted"
        out = dict(slopes=pd.DataFrame(rows), preds=pr, top=mc.top_budget_values(d_all), het=None, data=d_all)
    for k in ("slopes", "preds", "top"):
        out[k].insert(0, "run", tag)
    _point()
    return out
