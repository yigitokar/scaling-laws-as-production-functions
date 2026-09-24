"""share.py -- the planned inference share of lifetime compute, s = (w - 1)/w, by release period, and its comparison with
external disclosures.

PRIMARY SOURCE (review, 2026-09-24): module ra2_wedge's final, reviewed re-specified inversion.
  * Reference technology: Chinchilla with the outer exponent kappa free (m1 spec_kappa n240 Huber, refit by ra2;
    sigma*_kappa = 0.701), ra2's reference. Uncertainty: ra2's design-conditional wild bootstrap of that technology
    (Rademacher weights on log-loss residuals, B = 399; recomputed here with ra2's own function and seeds, 5 processes),
    with m2's pairs draws (B = 200) as a check. The kappa = 1 refit (m3's former reference) is reported for comparison.
  * Samples: (i) ra2's clean open-weight production-scale universe (6ND >= 1e21, 2019-2025; confident N and D; no MoE;
    no non-transformers; ra2's clean-sample exclusions applied to verified rows), rebuilt with ra2's own sample code;
    (ii) ra2's clean verified inference-demand sample (77 models, 2023-2025; output/tables/ra2_wedge_models.csv), on
    which ra2 evaluates all 32 ex-ante technologies. Every ra2 number used here is reproduced exactly from ra2's files
    and checked against output/tables/ra2_wedge_aggregate.csv (inference_share_checks.csv).
  * Technology range: technology-consistent (the aggregate recomputed under each of ra2's 32 ex-ante technologies;
    min and max over technologies). ra2's per-model envelope (each model at its own min/max over technologies) is also
    reported; it is an outer bound, not the aggregate under any single technology.
SUPERSEDED (kept for transparency only): the builder's preliminary version on m3's wider production-scale universe
under m3's kappa = 1 reference (inference_share_m3universe.csv).

Aggregate (compute-weighted) share, identical to ra2's s_agg: s = sum_i (w_i^+ - 1) C_i / sum_i w_i^+ C_i with
w^+ = max(w, 1), i.e. m/(1+m) with m the compute-weighted planned inference multiple (T truncated at 0).

Mapping to fleet-level disclosures (flow_adjustment): a fleet's inference share in year t differs from the planned
lifetime share of the models trained in t because (i) inference serves earlier (smaller) vintages when training compute
grows at rate g, factor phi = (1 - e^{-gL})/(gL) for serving life L; (ii) disclosed training includes R&D compute
(experiments, unreleased runs) at rho times final runs (Denain and Wu 2026: final runs 9.6-22.6% of R&D compute);
(iii) inference FLOPs may cost p > 1 times training FLOPs (memory-bound decoding). s_fleet = p m phi/(p m phi + rho).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

import ra3common  # noqa: F401  (sets sys.path for the m3_wedge imports)
import technologies as M3T
from common import log_w
from ra3common import ANALYSIS, DISCLOSURES, FINAL_RUN_SHARE, M2_PROC, M3_MODELS, PROC, ROOT, log

RA2_DIR = os.path.join(ANALYSIS, "ra2_wedge")
RA2_MODELS = os.path.join(ROOT, "output", "tables", "ra2_wedge_models.csv")
RA2_TECHS = os.path.join(ROOT, "output", "tables", "ra2_wedge_technologies.csv")
RA2_AGG = os.path.join(ROOT, "output", "tables", "ra2_wedge_aggregate.csv")
RA2_REF = "chin_q"
PERIODS = [("2019-2022", 2019, 2022), ("2023", 2023, 2023), ("2024", 2024, 2024), ("2025", 2025, 2025),
           ("All 2019-2025", 2019, 2025)]
B_WILD = 399


def agg_share(w, C, axis=-1):
    """ra2's s_agg: sum (w+ - 1) C / sum w+ C, w+ = max(w, 1). Works on (..., n) arrays."""
    wp = np.maximum(w, 1.0)
    X = np.sum((wp - 1.0) * C, axis=axis)
    return X / (X + np.sum(C, axis=-1))


# ----------------------------------------------------------------------------- ra2 inputs
def _ra2_modules():
    if RA2_DIR not in sys.path:
        sys.path.insert(0, RA2_DIR)
    import sample as SM2   # noqa: E402  (module ra2_wedge, read-only)
    import techs as TT2    # noqa: E402
    return SM2, TT2


def ra2_universe():
    """ra2's clean open-weight production-scale universe, rebuilt with ra2's sample code (the filter of ra2's
    analysis_ra2.open_closed, then open weights). ra2's build() writes an audit file; it is redirected here so that
    this module never writes into ra2's directories."""
    SM2, _ = _ra2_modules()
    SM2.PROC = os.path.join(PROC, "ra2_sample_rebuild")
    os.makedirs(SM2.PROC, exist_ok=True)
    d, B = SM2.build()
    U = SM2.universe(d, B)
    keep = (U["confident"].astype(bool) & ~U["moe"].fillna(False).astype(bool) & ~U["nontransformer"].astype(bool)
            & ~U["drop_B"].astype(bool) & U["open_weights"].fillna(False).astype(bool))
    X = U[keep].copy()
    X = X[np.isfinite(np.log(X["N"].astype(float))) & np.isfinite(np.log(X["D"].astype(float)))]
    return X.reset_index(drop=True)


def kappa_free_draws():
    """ra2's reference technology (point) and its design-conditional wild bootstrap (ra2 techs.chin_q_wild; same seeds,
    B = 399, 5 processes), plus m2's pairs draws (B = 200) as a check."""
    _, TT2 = _ra2_modules()
    pt, dr = TT2.chin_q_wild(B_WILD)
    ok = np.isfinite(dr).all(1)
    wild = dr[ok, :3]
    qd = np.load(os.path.join(M2_PROC, "boot", "chinchilla__all__huber_q.npy"))   # E, A, B, alpha, beta, q
    okp = (qd[:, 1] > 0) & (qd[:, 2] > 0) & (qd[:, 3] > 0) & (qd[:, 4] > 0)
    from common import lnG_of
    pairs = np.c_[qd[okp, 3], qd[okp, 4], lnG_of(qd[okp, 1], qd[okp, 2], qd[okp, 3], qd[okp, 4])]
    return pt, wild, pairs


def _w(lnN, lnD, al, be, lnG):
    return np.exp(log_w(lnN, lnD, al, be, lnG))


def _wd(lnN, lnD, draws):
    return np.exp(log_w(lnN[None, :], lnD[None, :], draws[:, 0:1], draws[:, 1:2], draws[:, 2:3]))


# ----------------------------------------------------------------------------- primary: ra2 universe and clean sample
def universe_shares(X, pt, wild, pairs, k1, dedup=False):
    if dedup:
        X = X.sort_values("date").drop_duplicates(subset=["lab", "N", "D"], keep="first").reset_index(drop=True)
    lnN, lnD, C = np.log(X["N"].values.astype(float)), np.log(X["D"].values.astype(float)), X["Cmp"].values.astype(float)
    w = _w(lnN, lnD, pt["alpha"], pt["beta"], pt["lnG"])
    Ww, Wp = _wd(lnN, lnD, wild), _wd(lnN, lnD, pairs)
    w1 = _w(lnN, lnD, k1.alpha, k1.beta, k1.lnG)
    W1 = _wd(lnN, lnD, k1.draws)
    rows = []
    for name, y0, y1 in PERIODS:
        s = X["year"].between(y0, y1).values
        if s.sum() == 0:
            continue
        sh = float(agg_share(w[s], C[s]))
        sw, sp, s1d = agg_share(Ww[:, s], C[s]), agg_share(Wp[:, s], C[s]), agg_share(W1[:, s], C[s])
        r = dict(period=name, n=int(s.sum()), C_total=float(C[s].sum()), s_ref=sh, m_ref=sh / (1 - sh),
                 s_ref_lo=float(np.percentile(sw, 2.5)), s_ref_hi=float(np.percentile(sw, 97.5)),
                 s_ref_pairs_lo=float(np.percentile(sp, 2.5)), s_ref_pairs_hi=float(np.percentile(sp, 97.5)),
                 B_wild=int(len(wild)), B_pairs=int(len(pairs)),
                 s_k1=float(agg_share(w1[s], C[s])), s_k1_lo=float(np.percentile(s1d, 2.5)),
                 s_k1_hi=float(np.percentile(s1d, 97.5)),
                 median_model_s_ref=float(np.median((w[s] - 1) / w[s])),
                 top_model=X.loc[np.where(s)[0][np.argmax(C[s])], "model"], top_share_C=float(C[s].max() / C[s].sum()))
        if s.sum() > 2:
            idx = np.where(s)[0]
            loo = np.array([agg_share(np.delete(w[s], j), np.delete(C[s], j)) for j in range(len(idx))])
            j = int(np.argmax(np.abs(loo - sh)))
            r.update(loo_min=float(loo.min()), loo_max=float(loo.max()), loo_most_influential=X.loc[idx[j], "model"],
                     s_without_most_influential=float(loo[j]))
        rows.append(r)
    return pd.DataFrame(rows)


def clean_sample_shares(pt, wild):
    """ra2's clean verified sample: reference, lab-own, ra2's per-model envelope, and the technology-consistent range
    over ra2's 32 ex-ante technologies."""
    Bm = pd.read_csv(RA2_MODELS)
    Bm = Bm[Bm["clean"].astype(bool)].reset_index(drop=True)
    keys = pd.read_csv(RA2_TECHS).query("in_set")["key"].tolist()
    lnN, lnD, C = np.log(Bm["N"].values), np.log(Bm["D"].values), Bm["Cmp"].values
    Ww = _wd(lnN, lnD, wild)
    rows, bytech = [], []
    for name, y0, y1 in PERIODS:
        s = Bm["year"].between(y0, y1).values
        if s.sum() == 0:
            continue
        per = {k: float(agg_share(Bm.loc[s, f"w_{k}"].values, C[s])) for k in keys}
        for k, v in per.items():
            bytech.append(dict(period=name, tech=k, n=int(s.sum()), s=v))
        sw = agg_share(Ww[:, s], C[s])
        kmin, kmax = min(per, key=per.get), max(per, key=per.get)
        rows.append(dict(period=name, n_clean=int(s.sum()), s_clean_ref=per[RA2_REF],
                         s_clean_ref_lo=float(np.percentile(sw, 2.5)), s_clean_ref_hi=float(np.percentile(sw, 97.5)),
                         s_clean_labown=float(agg_share(Bm.loc[s, "w_primary"].values, C[s])),
                         tech_min=per[kmin], tech_min_key=kmin, tech_max=per[kmax], tech_max_key=kmax,
                         tech_median=float(np.median(list(per.values()))), n_tech=len(keys),
                         envelope_min=float(agg_share(Bm.loc[s, "pt_min"].values, C[s])),
                         envelope_max=float(agg_share(Bm.loc[s, "pt_max"].values, C[s]))))
    return pd.DataFrame(rows), pd.DataFrame(bytech), Bm


def check_against_ra2(uni, cln):
    """Reproduce ra2_wedge_aggregate.csv exactly (reference on the clean universe and clean sample; envelope; lab-own)."""
    A = pd.read_csv(RA2_AGG)
    out = []

    def add(label, mine, theirs):
        out.append(dict(check=label, ra3=mine, ra2=theirs, abs_diff=abs(mine - theirs), passed=bool(abs(mine - theirs) < 1e-9)))

    u = A[(A["tech"] == "reference") & (A["sample"] == "open-weight universe (clean)")].set_index("year")
    for y in (2023, 2024, 2025):
        add(f"universe reference {y}", uni.set_index("period").loc[str(y), "s_ref"], float(u.loc[float(y), "s_agg"]))
    c = A[A["sample"] == "clean"]
    for tech, col in [("reference", "s_clean_ref"), ("lab-own where available", "s_clean_labown"),
                      ("min over technologies", "envelope_min"), ("max over technologies", "envelope_max")]:
        g = c[c["tech"] == tech].dropna(subset=["year"]).set_index("year")
        for y in (2023, 2024, 2025):
            add(f"clean sample {tech} {y}", cln.set_index("period").loc[str(y), col], float(g.loc[float(y), "s_agg"]))
    pooled = c[(c["tech"] == "reference") & (c["all"] == "2019-2025")]["s_agg"].iloc[0]
    add("clean sample reference pooled 2019-2025", cln.set_index("period").loc["All 2019-2025", "s_clean_ref"], float(pooled))
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- superseded preliminary version (m3)
M3_TECHS = ["chin", "besi", "hoff", "farseer_emb", "gadre_rw", "meta_a2"]


def m3_universe_shares(techs):
    """The builder's PRELIMINARY version: m3's production-scale open-weight universe (6ND >= 1e21, 2019-2026, core,
    non-code), m3's kappa = 1 reference with m3's pairs draws, band over m3's six technologies. Superseded."""
    d = pd.read_csv(M3_MODELS)
    d["core"] = d["core"].fillna(False).astype(bool) & ~d["code"].fillna(False).astype(bool)
    d["open_weights"] = d["open_weights"].fillna(False).astype(bool)
    U = d[d["core"] & (d["Cmp"] >= 1e21) & d["open_weights"] & d["year"].between(2019, 2026)].reset_index(drop=True)
    lnN, lnD, C = np.log(U["N"].values), np.log(U["D"].values), U["Cmp"].values
    per = [("2019-2022", 2019, 2022), ("2023", 2023, 2023), ("2024", 2024, 2024), ("2025", 2025, 2025),
           ("2026", 2026, 2026), ("All 2019-2026", 2019, 2026)]
    rows = []
    for name, y0, y1 in per:
        s = U["year"].between(y0, y1).values
        r = dict(period=name, n=int(s.sum()))
        pts, los, his = [], [], []
        for k in M3_TECHS:
            t = techs[k]
            w = _w(lnN, lnD, t.alpha, t.beta, t.lnG)
            pts.append(float(agg_share(w[s], C[s])))
            if t.draws is not None:
                sd = agg_share(_wd(lnN, lnD, t.draws)[:, s], C[s])
                los.append(np.percentile(sd, 2.5))
                his.append(np.percentile(sd, 97.5))
                if k == "chin":
                    r["s_k1_lo"], r["s_k1_hi"] = float(np.percentile(sd, 2.5)), float(np.percentile(sd, 97.5))
            if k == "chin":
                r["s_k1"] = pts[-1]
        r["band_lo"], r["band_hi"] = float(min(pts + los)), float(max(pts + his))
        rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- fleet mapping and disclosures
def flow_adjustment(m_list, g_list, L_list=(1.0, 2.0), rho_list=None, p_list=(1.0, 2.0)):
    """Implied fleet-level inference share for planned multiples m (= w - 1) under vintage growth g (per year, log),
    serving life L, R&D multiple rho and relative inference FLOP price p."""
    if rho_list is None:
        rho_list = [1.0] + [1 / v for v in sorted(FINAL_RUN_SHARE.values(), reverse=True)]
    rows = []
    for mname, m in m_list:
        for gname, g in g_list:
            for L in L_list:
                phi = (1 - np.exp(-g * L)) / (g * L)
                for rho in rho_list:
                    for p in p_list:
                        rows.append(dict(m_label=mname, m=m, g_label=gname, g=g, L=L, phi=phi, rho=rho, p=p,
                                         s_fleet=p * m * phi / (p * m * phi + rho)))
    return pd.DataFrame(rows)


def disclosures():
    return pd.DataFrame(DISCLOSURES)


def run_share():
    log("  share: rebuilding ra2's clean open-weight universe with ra2's sample code")
    X = ra2_universe()
    log("  share: ra2's kappa-free wild bootstrap (B = 399, 5 processes)")
    pt, wild, pairs = kappa_free_draws()
    techs = M3T.load_all()
    uni = universe_shares(X, pt, wild, pairs, techs["chin"])
    uni_dd = universe_shares(X, pt, wild, pairs, techs["chin"], dedup=True)
    cln, bytech, Bm = clean_sample_shares(pt, wild)
    chk = check_against_ra2(uni, cln)
    m3u = m3_universe_shares(techs)
    return dict(X=X, pt=pt, wild=wild, pairs=pairs, uni=uni, uni_dd=uni_dd, cln=cln, bytech=bytech, chk=chk, m3u=m3u,
                B=uni.merge(cln, on="period", how="left"))
