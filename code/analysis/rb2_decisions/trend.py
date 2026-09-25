"""trend.py -- the trend in the revealed value of compactness from 2023 on the clean sample under one set of rules
(R1 New 4; R2 Major 4; R3 N2; R4 R2-M1): unweighted median s by release year, decision-level medians, compute-weighted
aggregates (w truncated at 1, as in ra2/ra3), leave-one-developer-out, and all 32 ex-ante technologies. Pre-2023 only
as robustness: the clean-sample exclusions applied to the 2019-2022 open-weight universe, and the Kaplan-believed
technology of Prop. 2(iii) (the developer planned with Kaplan et al.'s allocation rule N ~ C^0.73, anchored at GPT-3),
under which a developer with no value of compactness shows w_K = 1 (s_K = 0).

Note for the writers: at given compute every technology ranks models by M, so the trend in s largely restates the rise
in M across release years (R2 Major 4.5); the median ln M by year is reported beside the shares.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import rb2common as C
import decisions as DC

YEARS = (2023, 2024, 2025)
KAPLAN_A = 0.73                        # N_opt ~ C^0.73 (Kaplan et al. 2020), anchored at GPT-3 (m5_progress)
KAPLAN_ANCHOR = (174.6e9, 3.14e23)
K_KAPLAN_JOINT = 1 / 0.535 - 1         # curvature of Kaplan's joint L(N, D) law (App. A: sigma* = 0.535)


def s_agg(w, Cc):
    wp = np.maximum(w, 1.0)
    return float(((wp - 1) * Cc).sum() / (wp * Cc).sum())


def by_year(Bc, U, wcol, label):
    rows = []
    UT = []
    X = Bc.merge(U, on="uid")
    for u, g in X.groupby("unit", sort=False):
        fam = g["unit_type"].iloc[0] == "family"
        W = DC.family_W(g[wcol].values, g["Cmp"].values) if fam else g[wcol].iloc[0]
        UT.append(dict(unit=u, year=int(g["year"].min()), W=W, Cmp=g["Cmp"].sum(), dev=g["dev"].iloc[0]))
    UT = pd.DataFrame(UT)
    for y in YEARS:
        g = Bc[Bc["year"] == y]
        gu = UT[UT["year"] == y]
        rows.append(dict(tech=label, year=y, n_models=len(g), n_units=len(gu),
                         median_s_models=float(np.median(C.share(g[wcol]))),
                         median_s_units=float(np.median(C.share(gu["W"]))),
                         s_agg_models=s_agg(g[wcol].values, g["Cmp"].values),
                         s_agg_units=s_agg(gu["W"].values, gu["Cmp"].values),
                         median_lnM=float(np.median(np.log(g["M"]))), median_M=float(np.median(g["M"])),
                         top_model=g.loc[g["Cmp"].idxmax(), "model"], top_share_C=float(g["Cmp"].max() / g["Cmp"].sum())))
    return pd.DataFrame(rows), UT


def monotone(df, col):
    v = df.sort_values("year")[col].values
    return bool(np.all(np.diff(v) > 0))


def leave_one_dev_out(Bc, U, wcol):
    rows = []
    for dev in sorted(Bc["dev"].unique()):
        Bd = Bc[Bc["dev"] != dev]
        Ud = U[U["uid"].isin(Bd["uid"])]
        t, _ = by_year(Bd, Ud, wcol, f"without {dev}")
        t["dropped"] = dev
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def kaplan_lnw(N, D, k):
    """ln w under the Kaplan-believed technology: path N_K(C) = N_GPT3 (C/C_GPT3)^0.73, curvature k."""
    Cc = 6 * N * D
    NK = KAPLAN_ANCHOR[0] * (Cc / KAPLAN_ANCHOR[1]) ** KAPLAN_A
    MK = Cc / (6 * NK ** 2)
    return k * (np.log(D / N) - np.log(MK))


def pre2023(U, Lu, Bc, k_ref):
    """2019-2022 open-weight production-scale universe (ra3's filter) with the reference and the Kaplan belief."""
    X = U.merge(Lu[["uid", "w"]], on="uid")
    keep = (X["confident"].astype(bool) & ~X["moe"].fillna(False).astype(bool) & ~X["nontransformer"].astype(bool)
            & ~X["drop_B"].astype(bool) & X["open_weights"].fillna(False).astype(bool) & X["year"].between(2019, 2022))
    X = X[keep].copy()
    suite = X["model"].str.contains(r"^OPT|BLOOM", case=False, regex=True)
    X["suite_member"] = suite
    X["lnw_kaplan_refk"] = kaplan_lnw(X["N"].values, X["D"].values, k_ref)
    X["lnw_kaplan_jointk"] = kaplan_lnw(X["N"].values, X["D"].values, K_KAPLAN_JOINT)
    rows = []
    for lab, g in [("2019-2022 universe (ra2/ra3: 8 models)", X),
                   ("2019-2022 universe, clean exclusions applied to suite members (OPT, BLOOM dropped)", X[~X["suite_member"]])]:
        for tl, w in [("reference (Chinchilla, kappa free)", g["w"].values),
                      ("Kaplan-believed (Kaplan path, reference curvature)", np.exp(g["lnw_kaplan_refk"].values)),
                      ("Kaplan-believed (Kaplan path, Kaplan joint-law curvature)", np.exp(g["lnw_kaplan_jointk"].values))]:
            rows.append(dict(sample=lab, technology=tl, n=len(g), median_s=float(np.median(C.share(w))),
                             s_agg_trunc=s_agg(w, g["Cmp"].values),
                             s_agg_untrunc=float(((w - 1) * g["Cmp"]).sum() / (w * g["Cmp"]).sum()),
                             share_w_gt1=float(np.mean(w > 1)), models="; ".join(g["model"])))
    # contrast: the clean 2023-2025 sample under the Kaplan belief
    lk = kaplan_lnw(Bc["N"].values, Bc["D"].values, k_ref)
    rows.append(dict(sample="clean sample 2023-2025 (contrast)", technology="Kaplan-believed (Kaplan path, reference curvature)",
                     n=len(Bc), median_s=float(np.median(C.share(np.exp(lk)))), s_agg_trunc=s_agg(np.exp(lk), Bc["Cmp"].values),
                     s_agg_untrunc=np.nan, share_w_gt1=float(np.mean(lk > 0)), models=""))
    per_model = X[["model", "year", "N", "D", "M", "Cmp", "w", "suite_member", "lnw_kaplan_refk", "lnw_kaplan_jointk"]].copy()
    per_model["s_ref"] = C.share(per_model["w"])
    per_model["w_kaplan_refk"] = np.exp(per_model["lnw_kaplan_refk"])
    per_model["s_kaplan_refk"] = C.share(per_model["w_kaplan_refk"])
    per_model["share_C_period"] = per_model["Cmp"] / per_model["Cmp"].sum()
    return pd.DataFrame(rows), per_model
