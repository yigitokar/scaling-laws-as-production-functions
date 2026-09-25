"""serving.py -- conduct tests with the model-level serving code, at the model level and at the decision level
(R1 New 1(c)-(d), 4; R3 N3(c), N4; R2 Major 2 request 3), memory-tier flags (w - 1 is an UPPER bound on m_N inside a
quantized memory-tier window; Prop. A8(iii)(b)), the on-device contrast (descriptive), and the open-weight premium with
common-D families entered once.

Inference: restricted wild cluster bootstrap-t by developer, Webb six-point weights, B = 9,999 (ra2_wedge's wcr, which
also reports the number of treated clusters; with G_treated <= 3 the p-value is not interpretable).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import rb2common as C
import analysis_ra2 as AN  # ra2
import decisions as DC


def _wcr(df, y, cols, test, extra=("lnC",)):
    return AN.wcr(df, y, cols, test, extra=extra)


def model_level_tests(Xm):
    """Xm: clean-sample models with lnw, s, lnC, year, dev, serve_* and deploy flags."""
    res = []
    specs = [("serve_developer", "developer-level code (ra2)"), ("serve_model", "model-level code (primary)"),
             ("serve_model_hi", "model-level code, ambiguous = 1")]
    for yv in ("lnw", "s"):
        for col, lab in specs:
            r = _wcr(Xm, yv, [col], col)
            res.append(dict(level="model", sample="clean sample (77)", code=lab, outcome=yv, regressor=col, **r))
        for dep in ("ondevice",):
            r = _wcr(Xm, yv, ["serve_model", dep], "serve_model")
            res.append(dict(level="model", sample="clean sample (77)", code="model-level, with on-device control",
                            outcome=yv, regressor="serve_model", **r))
    Xa = Xm[Xm["dev"] != "Alibaba"]
    for yv in ("lnw",):
        r = _wcr(Xa, yv, ["serve_model"], "serve_model")
        res.append(dict(level="model", sample="clean sample without Alibaba", code="model-level", outcome=yv,
                        regressor="serve_model", **r))
    Xq = Xm[~Xm["gen"].isin(["Qwen2.5", "Qwen3"])]
    r = _wcr(Xq, "lnw", ["serve_model"], "serve_model")
    res.append(dict(level="model", sample="clean sample without Qwen2.5/Qwen3 (D unaudited)", code="model-level",
                    outcome="lnw", regressor="serve_model", **r))
    return pd.DataFrame(res)


def decision_level_tests(UT):
    """UT: unit table (primary decision units) with lnw (ln W_ref), s, lnC (log total compute), year, dev."""
    res = []
    for yv in ("lnw", "s"):
        for col, lab in [("serve_any", "any member served (model-level code)"),
                         ("serve_flagship", "largest member served"), ("serve_all", "all members served"),
                         ("serve_developer", "developer-level code")]:
            r = _wcr(UT, yv, [col], col)
            res.append(dict(level="decision", sample="decision units (primary)", code=lab, outcome=yv, regressor=col, **r))
    r = _wcr(UT[UT["dev"] != "Alibaba"], "lnw", ["serve_any"], "serve_any")
    res.append(dict(level="decision", sample="decision units without Alibaba", code="any member served", outcome="lnw",
                    regressor="serve_any", **r))
    return pd.DataFrame(res)


def descriptive(Xm, UT):
    rows = []

    def add(level, group, g, wcol="w", n_dev=True):
        w = g[wcol].values
        rows.append(dict(level=level, group=group, n=len(g), n_developers=g["dev"].nunique(), median_w=float(np.median(w)),
                         median_s=float(np.median(C.share(w))), q25_s=float(np.percentile(C.share(w), 25)),
                         q75_s=float(np.percentile(C.share(w), 75)), share_w_gt1=float(np.mean(w > 1)),
                         developers="; ".join(sorted(g["dev"].unique())) if n_dev else ""))
    for lab, m in [("served (model-level)", Xm["serve_model"] == 1), ("not served (model-level)", Xm["serve_model"] == 0),
                   ("served (developer-level, ra2)", Xm["serve_developer"] == 1),
                   ("not served (developer-level, ra2)", Xm["serve_developer"] == 0),
                   ("served, outside tier windows and not common-D", (Xm["serve_model"] == 1) & ~Xm["tier"].astype(bool) & (Xm["fam_class"] != "common-D")),
                   ("served, inside a tier window (w - 1 upper bound)", (Xm["serve_model"] == 1) & Xm["tier"].astype(bool)),
                   ("served, common-D family member", (Xm["serve_model"] == 1) & (Xm["fam_class"] == "common-D")),
                   ("served, without Alibaba", (Xm["serve_model"] == 1) & (Xm["dev"] != "Alibaba")),
                   ("on-device target (card)", Xm["deploy"] == "ondevice"),
                   ("local target (card)", Xm["deploy"] == "local"),
                   ("server or unspecified target", ~Xm["deploy"].isin(["ondevice", "local"])),
                   ("inside a tier window", Xm["tier"].astype(bool)), ("outside tier windows", ~Xm["tier"].astype(bool)),
                   ("inside a tier window, literal windows (<=3.3B, 7-9.5B, 12-14.9B, 27-32.9B, 65-72.9B)", Xm["tier_alt"].astype(bool)),
                   ("served, inside a literal tier window", (Xm["serve_model"] == 1) & Xm["tier_alt"].astype(bool)),
                   ("neither common-D nor tier window", ~Xm["tier"].astype(bool) & (Xm["fam_class"] != "common-D"))]:
        add("model", lab, Xm[m])
    for lab, m in [("served (any member)", UT["serve_any"] == 1), ("not served", UT["serve_any"] == 0),
                   ("on-device target", UT["ondevice"] == 1), ("not on-device", UT["ondevice"] == 0)]:
        add("decision", lab, UT[m], wcol="w_ref")
    return pd.DataFrame(rows)


def serving_composition(Xm):
    """R1 New 1(c) tabulation with the model-level code: served models by family class and tier window."""
    S = Xm[Xm["serve_model"] == 1]
    rows = []
    for lab, m in [("served, total", pd.Series(True, index=S.index)), ("common-D family member", S["fam_class"] == "common-D"),
                   ("inside a tier window", S["tier"].astype(bool)),
                   ("common-D or tier window", (S["fam_class"] == "common-D") | S["tier"].astype(bool)),
                   ("neither", (S["fam_class"] != "common-D") & ~S["tier"].astype(bool)),
                   ("Alibaba", S["dev"] == "Alibaba")]:
        g = S[m]
        rows.append(dict(group=lab, n=len(g), models="; ".join(g["model"]), median_s_ref=float(np.median(C.share(g["w"]))) if len(g) else np.nan))
    return pd.DataFrame(rows)


def open_premium_decisions(Uc, Bc, U_units):
    """Open-weight premium (ra2 test (a): given year FE and ln C) with the clean common-D families entered once (family
    W_f over the members present in the production-scale universe, total compute, earliest year)."""
    X = Uc.copy()
    X["open"] = X["open_weights"].astype(float)
    fam = U_units[U_units["unit_type"] == "family"]
    rows = []
    for u, g in fam.groupby("unit"):
        mem = X[X["uid"].isin(g["uid"])]
        if not len(mem):
            continue
        W = DC.family_W(mem["w"].values, mem["Cmp"].values)
        r = mem.iloc[0].copy()
        r["uid"] = f"family:{u}"
        r["w"] = W
        r["Cmp"] = mem["Cmp"].sum()
        r["year"] = mem["year"].min()
        rows.append(r)
    Xd = pd.concat([X[~X["uid"].isin(fam["uid"])], pd.DataFrame(rows)], ignore_index=True)
    out = []
    for lab, D in [("model level (ra2 universe)", X), ("decision level (common-D families once)", Xd)]:
        D = D.copy()
        D["lnw"] = np.log(D["w"])
        D["s"] = C.share(D["w"])
        D["lnC"] = np.log(D["Cmp"])
        for yv in ("lnw", "s"):
            r = _wcr(D, yv, ["open"], "open")
            out.append(dict(level=lab, outcome=yv, spec="year FE + ln C", **r))
        D2 = D[D["year"] >= 2023]
        r = _wcr(D2, "lnw", ["open"], "open")
        out.append(dict(level=lab + ", 2023+", outcome="lnw", spec="year FE + ln C", **r))
    return pd.DataFrame(out)
