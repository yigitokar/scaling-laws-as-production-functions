"""trend.py -- the 2023-2025 trend in the revealed wedge on the clean sample (rb2_decisions' reviewed decision units and
models), expressed as an annual growth factor of the value of compactness, g_w = exp(d ln w / dt), and as the implied
growth of the data multiple D/D*(C).

Two readings of the same trend (R3 N6(1)):
  * M-based (sigma*-free given M*): ln w = k_ref ln(M/M*(C)) under the reference technology (Chinchilla, kappa free,
    k_ref = 1/0.7006 - 1), so the data multiple sqrt(M/M*) grows at exp(slope / (2 k_ref)). Extrapolating tokens per
    parameter needs no curvature.
  * Value-based: if the value of compactness (w, measured under the reference) is the primitive that keeps growing, the
    frontier's data multiple responds with elasticity e(sigma*) = sigma*/[2(1 - sigma*)], so it grows at g_w^e(sigma*).
    At sigma* = 0.7006 the two readings coincide.

Variants: decision units (primary; one observation per allocation decision), models, compute-weighted (WLS), units
above 1e24 FLOP (closest to frontier scale), and the log growth of the compute-weighted aggregate wedge
1/(1 - s_agg) and of the median wedge between 2023 and 2025. Inference: wild cluster bootstrap by developer (ra3's
growth.wild_cluster: CR1 standard errors, Webb weights, percentile-t). Robustness: leave one developer out; the 32
ex-ante technologies (M-based, on models).

Round 3 (fix list D-2, E1(b); R1 round-3 minors 11 and 13, R2 minor 9, R3 E(b)):
  * the decision units are the 49 budget-level units of module rb5_units (data/processed/rb5_units/units_primary.csv),
    checked against rb2_decisions' unit table; version 3's 56 family-label units are a robustness row;
  * new rows: within developers (developer fixed effects; a second row adds log compute), and the four disclosed dense
    flagship runs above 1e25 FLOP (descriptive: an OLS slope over four runs, no interval);
  * the 32-technology trend now uses the audited token counts (rb2's wedges under every technology, exhibits cache)
    instead of ra2's inputs, so its reference row equals the 'Models, OLS' row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import rb3common as RC
from rb3common import B_WCB, SEED, dec_year, e_of, log

import growth as GR  # noqa: E402  (ra3_econ, read-only)

YEARS = (2023, 2024, 2025)


def load_units():
    """The primary decision units from rb5_units' model-level file: one row per unit with its developer, first member's
    release date, unit year, unit wedge W_unit (family share for common-budget units, the model's own wedge otherwise),
    and the unit's training compute. Returns the table and a list of checks against rb2_decisions' unit table."""
    m = pd.read_csv(RC.RB5_UNITS)
    g = m.groupby("unit", sort=False)
    for col in ("dev", "W_unit", "C_unit", "unit_year", "id_class"):
        assert (g[col].nunique(dropna=False) == 1).all(), f"units_primary.csv: {col} not constant within a unit"
    u = pd.DataFrame(dict(dev=g.dev.first(), date=g.date.min(), year=g.unit_year.first().astype(int),
                          w_ref=g.W_unit.first(), C_total=g.C_unit.first(), n_members=g.size(),
                          id_class=g.id_class.first(), codes_file=g.codes_file.first())).reset_index()
    r2 = pd.read_csv(RC.RB2_UNITS).set_index("unit")
    x = u.set_index("unit")
    chk = [dict(check="units: same unit labels as rb2 decision_units.csv", rb3=len(x), rb2=len(r2),
                passed=bool(set(x.index) == set(r2.index)))]
    if chk[0]["passed"]:
        r2 = r2.loc[x.index]
        for mine, theirs, tol in (("w_ref", "w_ref", 1e-12), ("C_total", "C_total", 1e-12)):
            dev = float(np.max(np.abs(x[mine].values / r2[theirs].values - 1)))
            chk.append(dict(check=f"units: {mine} equals rb2 (max rel. dev.)", rb3=dev, rb2=0.0, passed=bool(dev < tol)))
        for col in ("year", "dev", "date", "n_members"):
            nd = int((x[col].astype(str).values != r2[col].astype(str).values).sum())
            chk.append(dict(check=f"units: {col} equals rb2 (rows differing)", rb3=nd, rb2=0, passed=nd == 0))
    return u, chk


def load():
    d = pd.read_csv(RC.RB2_MODELS)
    d["t"] = dec_year(d["date"])
    u, uchk = load_units()
    u["t"] = dec_year(u["date"])
    u = u.rename(columns={"C_total": "Cmp"})
    k = np.log(d["w_ref"]) / np.log(d["M"] / d["Mstar_ref"])
    assert np.allclose(k, k.iloc[0], atol=1e-10), "reference wedges are not (M/M*)^k"
    return d, u, float(k.iloc[0]), uchk


def slope_wcb(t, y, dev, weights=None, B=B_WCB, seed=SEED):
    """OLS (or WLS via sqrt-weight transform) of y on (t - 2024); wild cluster bootstrap-t CI for the slope."""
    t = np.asarray(t, float) - RC.T0
    y = np.asarray(y, float)
    sw = np.ones_like(t) if weights is None else np.sqrt(np.asarray(weights, float) / np.mean(weights))
    X = np.c_[sw, sw * t]
    g = pd.factorize(pd.Series(dev))[0]
    r = GR.wild_cluster(y * sw, X, g, B=B, seed=seed, nulls=())
    r.pop("draws", None)
    return dict(slope=r["b1"], se=r["se_b1"], lo=r["b1_lo"], hi=r["b1_hi"], n=len(y), G=int(r["G"]))


def slope_wcb_fe(t, y, dev, extra=None, B=B_WCB, seed=SEED):
    """Within-developer trend: OLS of y on (t - 2024) with developer fixed effects (and optional extra regressors);
    wild cluster bootstrap-t CI by developer for the slope. The slope is column 1 of the design, as growth.wild_cluster
    expects; developers with one unit are absorbed by their dummy. The percentile-t interval does not depend on the
    CR1 small-sample factor, which cancels between the observed and the bootstrap t."""
    t = np.asarray(t, float) - RC.T0
    y = np.asarray(y, float)
    g = pd.factorize(pd.Series(dev))[0]
    G = int(g.max()) + 1
    Dm = np.zeros((len(y), G))
    Dm[np.arange(len(y)), g] = 1.0
    cols = [Dm[:, :1], t[:, None], Dm[:, 1:]]
    if extra is not None:
        cols.append(np.asarray(extra, float).reshape(len(y), -1))
    X = np.hstack(cols)
    r = GR.wild_cluster(y, X, g, B=B, seed=seed, nulls=())
    r.pop("draws", None)
    multi = pd.Series(t).groupby(g).agg(lambda v: v.max() - v.min() > 0)
    return dict(slope=r["b1"], se=r["se_b1"], lo=r["b1_lo"], hi=r["b1_hi"], n=len(y), G=G,
                G_within=int(multi.sum()))


def flagship_runs():
    """The four disclosed dense runs above 1e25 FLOP in 2024-2026 (ra3's anchor runs), with release dates (m3) and
    reference wedges w = (D/D*_ref(C))^(2 k_ref) on ra3's reference path, as in scenario.py."""
    import demand as DM  # noqa: E402  (ra3_econ, read-only)
    runs = DM.frontier_runs()
    ref = DM.load_path_techs()["chin_q"]
    m3 = pd.read_csv(RC.M3_MODELS).drop_duplicates("model").set_index("model")
    runs = runs.assign(date=m3.reindex(runs["model"])["date"].values)
    runs["t"] = dec_year(runs["date"])
    runs["D_over_Dstar_ref"] = runs["D"].values / np.exp(ref.lnD(np.log(runs["Cmp"].values)))
    return runs


def s_agg(w, C):
    wp = np.maximum(np.asarray(w, float), 1.0)
    C = np.asarray(C, float)
    return float(((wp - 1) * C).sum() / (wp * C).sum())


def variants(d, u, k_ref):
    rows = []

    def add(label, key, r, note=""):
        rows.append(dict(key=key, label=label, n=r["n"], clusters=r.get("G", np.nan), developers=r.get("G", np.nan),
                         slope_lnw=r["slope"],
                         se=r.get("se", np.nan), slope_lo=r.get("lo", np.nan), slope_hi=r.get("hi", np.nan), note=note))

    add("Decision units, OLS (primary)", "units_ols", slope_wcb(u.t, np.log(u.w_ref), u.dev, seed=SEED + 1))
    fe = slope_wcb_fe(u.t, np.log(u.w_ref), u.dev, seed=SEED + 6)
    add("Decision units, developer fixed effects", "units_fe", fe,
        note=f"{fe['G_within']} developers with units at more than one date identify the slope")
    big = u[u.Cmp >= 1e24]
    add("Decision units above 1e24 FLOP", "units_big", slope_wcb(big.t, np.log(big.w_ref), big.dev, seed=SEED + 5),
        note=f"{len(big)} units, {big.dev.nunique()} developers; few clusters: indicative")
    add("Decision units, compute-weighted", "units_wls", slope_wcb(u.t, np.log(u.w_ref), u.dev, u.Cmp, seed=SEED + 3))
    # the four disclosed dense flagship runs above 1e25 FLOP (descriptive: OLS slope, no interval)
    fr = flagship_runs()
    lnw_fr = 2 * k_ref * np.log(fr["D_over_Dstar_ref"].values)
    b_fr = float(np.polyfit(fr.t - RC.T0, lnw_fr, 1)[0])
    top = fr.sort_values("t").iloc[-1]
    rest = fr[fr.model != top.model]
    b_rest = float(np.polyfit(rest.t - RC.T0, 2 * k_ref * np.log(rest["D_over_Dstar_ref"].values), 1)[0])
    rows.append(dict(key="flagships", label="Flagship runs above 1e25 FLOP (descriptive)", n=len(fr), slope_lnw=b_fr,
                     developers=fr.lab.nunique(), last_run=top.model, g_w_without_last=np.exp(b_rest),
                     note=("runs: " + "; ".join(f"{r.model} {r.date[:10]} w {np.exp(2 * k_ref * np.log(r.D_over_Dstar_ref)):.2f}"
                                                for r in fr.sort_values("t").itertuples())
                           + f"; without {top.model}: {np.exp(b_rest):.2f} a year")))
    add("Decision units, developer fixed effects and log compute", "units_fe_lnC",
        slope_wcb_fe(u.t, np.log(u.w_ref), u.dev, extra=np.log(u.Cmp.values), seed=SEED + 7))
    add("Models, OLS", "models_ols", slope_wcb(d.t, np.log(d.w_ref), d.dev, seed=SEED + 2))
    add("Models, compute-weighted", "models_wls", slope_wcb(d.t, np.log(d.w_ref), d.dev, d.Cmp, seed=SEED + 4))
    # robustness: version 3's 56 family-label units (rb2's decision_units_family56.csv)
    u56 = pd.read_csv(RC.RB2_UNITS56)
    u56["t"] = dec_year(u56["date"])
    # same seed as version 3's primary row, so that only the data (the token audit of MPT-30B) differ from version 3
    add("Family-label units (version 3), OLS", "units56_ols", slope_wcb(u56.t, np.log(u56.w_ref), u56.dev, seed=SEED + 1))
    # endpoint growth of year aggregates (no CI): compute-weighted aggregate wedge 1/(1 - s_agg) and the median wedge
    for lab, key, X in [("Aggregate wedge, decision units", "agg_units", u),
                        ("Aggregate wedge, models", "agg_models", d)]:
        w = [1 / (1 - s_agg(X[X.year == y].w_ref, X[X.year == y].Cmp)) for y in YEARS]
        rows.append(dict(key=key, label=lab, n=len(X), slope_lnw=(np.log(w[2]) - np.log(w[0])) / 2.0,
                         note="w_agg = " + " / ".join(f"{v:.2f}" for v in w) + " (2023/2024/2025)"))
    for lab, key, X in [("Median wedge, decision units", "median_units", u),
                        ("Median wedge, models", "median_models", d)]:
        w = [float(np.median(X[X.year == y].w_ref)) for y in YEARS]
        rows.append(dict(key=key, label=lab, n=len(X), slope_lnw=(np.log(w[2]) - np.log(w[0])) / 2.0,
                         note="median w = " + " / ".join(f"{v:.2f}" for v in w)))
    T = pd.DataFrame(rows)
    T["g_w"] = np.exp(T.slope_lnw)
    T["g_w_lo"], T["g_w_hi"] = np.exp(T.slope_lo), np.exp(T.slope_hi)
    # M-based data multiple growth (sigma*-free given M*) and value-based growth at each sigma*
    T["g_DDstar_Mbased"] = np.exp(T.slope_lnw / (2 * k_ref))
    T["g_DDstar_Mbased_lo"] = np.exp(T.slope_lo / (2 * k_ref))
    T["g_DDstar_Mbased_hi"] = np.exp(T.slope_hi / (2 * k_ref))
    for s in RC.SIGMAS:
        T[f"g_DDstar_value_s{int(round(100 * s))}"] = T.g_w ** e_of(s)
        T[f"g_DDstar_value_s{int(round(100 * s))}_lo"] = T.g_w_lo ** e_of(s)
        T[f"g_DDstar_value_s{int(round(100 * s))}_hi"] = T.g_w_hi ** e_of(s)
    T["g_M_over_Mstar"] = np.exp(T.slope_lnw / k_ref)
    return T


def lodo(u, d):
    rows = []
    for dev in sorted(u.dev.unique()):
        uu = u[u.dev != dev]
        dd = d[d.dev != dev]
        b = np.polyfit(uu.t - RC.T0, np.log(uu.w_ref), 1)[0]
        w = [1 / (1 - s_agg(uu[uu.year == y].w_ref, uu[uu.year == y].Cmp)) for y in YEARS]
        bm = np.polyfit(dd.t - RC.T0, np.log(dd.w_ref), 1)[0]
        rows.append(dict(dropped=dev, n_units=len(uu), g_w_units_ols=np.exp(b), g_w_models_ols=np.exp(bm),
                         g_w_agg_units=float(np.sqrt(w[2] / w[0]))))
    return pd.DataFrame(rows)


def by_technology(d):
    """M-based data-multiple growth under each of the 32 ex-ante technologies (models; ln(M/M*_j) = ln w_j / k_j).
    Round 3 (fix list E6): on the audited token counts of the clean sample (rb2_decisions' long table of wedges, with
    each technology's own parameter convention in M_used), so that the reference row equals the 'Models, OLS' row."""
    X = pd.read_pickle(RC.RB2_CACHE)
    L, Mreg, exante = X["L"], X["M"].set_index("key"), list(X["exante"])
    assert len(exante) == 32, len(exante)
    L = L[L.tech.isin(exante)].merge(d[["uid", "t"]], on="uid", how="inner")
    rows = []
    for key in exante:
        x = L[(L.tech == key) & np.isfinite(L.lnw)]
        assert len(x) == len(d), (key, len(x))
        k = float(Mreg.loc[key, "S2"])
        lnMM = np.log(x.M_used.values) - x.lnMstar.values
        assert np.allclose(x.lnw.values, k * lnMM, atol=1e-9), key
        b = np.polyfit(x.t - RC.T0, lnMM, 1)[0]
        bw = np.polyfit(x.t - RC.T0, x.lnw.values, 1)[0]
        rows.append(dict(tech=key, label=Mreg.loc[key, "label"], sigma_star=float(Mreg.loc[key, "sigma_star"]), k=k,
                         a=float(Mreg.loc[key, "a"]), n=len(x), g_M_over_Mstar=np.exp(b), g_DDstar_Mbased=np.exp(b / 2),
                         g_w=np.exp(bw)))
    return pd.DataFrame(rows)


def by_year_table(d, u):
    rows = []
    for y in YEARS:
        du, dm = u[u.year == y], d[d.year == y]
        rows.append(dict(year=y, n_units=len(du), n_models=len(dm),
                         s_agg_units=s_agg(du.w_ref, du.Cmp), s_agg_models=s_agg(dm.w_ref, dm.Cmp),
                         median_s_units=float(np.median(1 - 1 / du.w_ref)), median_w_units=float(np.median(du.w_ref)),
                         median_M_models=float(np.median(dm.M))))
    B = pd.DataFrame(rows)
    B["w_agg_units"] = 1 / (1 - B.s_agg_units)
    B["w_agg_models"] = 1 / (1 - B.s_agg_models)
    B["m_agg_units"] = B.w_agg_units - 1
    B["m_agg_models"] = B.w_agg_models - 1
    return B


def run():
    d, u, k_ref, uchk = load()
    log(f"trend: {len(d)} models, {len(u)} decision units ({RC.os.path.relpath(RC.RB5_UNITS, RC.ROOT)}); "
        f"k_ref = {k_ref:.5f} (sigma* = {1 / (1 + k_ref):.4f})")
    T = variants(d, u, k_ref)
    L = lodo(u, d)
    Tt = by_technology(d)
    B = by_year_table(d, u)
    # check against rb2's published by-year aggregates (reference technology)
    rb2 = pd.read_csv(RC.RB2_TREND)
    ref = rb2[rb2.tech.astype(str).str.contains("reference", case=False)]
    chk = []
    if len(ref):
        r0 = ref.groupby("year").first()
        for y in YEARS:
            for mine, theirs in [("s_agg_units", "s_agg_units"), ("s_agg_models", "s_agg_models")]:
                if theirs in r0.columns:
                    v = float(B.set_index("year").loc[y, mine])
                    chk.append(dict(check=f"{mine} {y}", rb3=v, rb2=float(r0.loc[y, theirs]),
                                    passed=bool(abs(v - float(r0.loc[y, theirs])) < 1e-9)))
    ref_row = Tt[Tt.tech == "chin_q"]
    if len(ref_row):
        dv = abs(float(ref_row.g_DDstar_Mbased.iloc[0]) / float(T.set_index("key").loc["models_ols", "g_DDstar_Mbased"]) - 1)
        chk.append(dict(check="32 technologies: reference row equals Models, OLS (M-based)", rb3=dv, rb2=0.0,
                        passed=bool(dv < 1e-9)))
    chk = pd.DataFrame(uchk + chk)
    return dict(d=d, u=u, k_ref=k_ref, T=T, L=L, Tt=Tt, B=B, chk=chk)
