"""vintage.py -- reconciling our frontier compute growth with Epoch's published 4.2x per year for ~2018 to May 2024
(R3 round-2 minor 17; ra3 open issue 5).

Epoch (Sevilla and Roldan 2024, posted 28 May 2024): frontier = models in the top 10 of training compute when released;
'4.2x/year (90% CI: 3.6x to 4.9x) after 2018' (piecewise exponential with a break estimated around 2018; AlphaGo Master
and Zero excluded as outliers; bootstrap CI). ra3 estimated 5.23x for January 2018 to May 2024 on the database snapshot
of 23 September 2026, 'probably a vintage effect'. Here the same estimator (OLS of log10 C on release date; running
top-10 over all earlier models; wild cluster bootstrap by developer, Webb weights) is applied to

  V24  Epoch's all_systems.csv as captured by the Wayback Machine on 31 May 2024 (three days after the post);
  V26  the 23 September 2026 snapshot (ra3's input), restricted to releases up to 31 May 2024;

and the difference is decomposed into (i) compute REVISIONS of models present in both vintages (including estimates
withdrawn since) and (ii) ADDITIONS of models released before June 2024 but entered after it, with a two-factor
Shapley decomposition of ln(growth). Models are matched by normalized name, then by (release date, first
organization) for renamed entries (e.g. 'GPT-4' -> 'GPT-4 (Mar 2023)').
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import rb3common as RC
from rb3common import SEED, log

import growth as GR  # noqa: E402  (ra3_econ, read-only)
from ra3common import DEV_MAP  # noqa: E402

START, END = "2018-01-01", RC.EPOCH_WINDOW_END


def _norm(s):
    return s.astype(str).str.lower().str.replace(r"[\W_]+", "", regex=True)   # keeps non-ASCII letters (PanGu-α/Σ)


def load(path, namecol, snapshot):
    d = pd.read_csv(path, low_memory=False)
    d = d.rename(columns={namecol: "name"})
    d["date"] = pd.to_datetime(d["Publication date"], errors="coerce")
    d["C"] = pd.to_numeric(d["Training compute (FLOP)"], errors="coerce")
    d = d[d["date"].notna() & (d["date"] <= pd.Timestamp(snapshot))].copy()
    org = d["Organization"].fillna("unknown").astype(str).str.split(",").str[0].str.strip()
    d["dev"] = org.replace(DEV_MAP)
    d["org1"] = _norm(org)
    d["key"] = _norm(d["name"])
    d["notable"] = d["Notability criteria"].notna()
    return d[["name", "key", "date", "C", "dev", "org1", "notable", "Confidence", "Domain"]].reset_index(drop=True)


def running_top10(d):
    """ra3's frontier flag (growth.load_epoch, reproduced exactly): rows sorted by (date, compute); a model is flagged if
    fewer than 10 rows before it in that order, released on or before its date, have larger compute."""
    d = d[d["C"].notna() & (d["C"] > 0)].sort_values(["date", "C"]).reset_index(drop=True).copy()
    lc, dt = np.log10(d["C"].values), d["date"].values
    flag = np.zeros(len(d), bool)
    for i in range(len(d)):
        prev = np.arange(i)
        prev = prev[dt[prev] <= dt[i]]
        flag[i] = (lc[prev] > lc[i]).sum() < 10
    d["top10"] = flag
    d["t"] = d["date"].dt.year + (d["date"].dt.dayofyear - 1) / 365.25
    d["lc"] = np.log10(d["C"])
    return d


def fit(d, seed, label):
    s = d[d["top10"] & (d["date"] >= START) & (d["date"] <= END)]
    y = s["lc"].values
    X = np.c_[np.ones(len(s)), s["t"].values - RC.T0]
    g = pd.factorize(s["dev"])[0]
    r = GR.wild_cluster(y, X, g, seed=seed, nulls=[("4.2x", np.log10(4.2))])
    return dict(sample=label, n=len(s), clusters=int(r["G"]), growth=10 ** r["b1"], growth_lo=10 ** r["b1_lo"],
                growth_hi=10 ** r["b1_hi"], p_wcr_4p2=r["p_4.2x"], models=len(d))


def match(o, n):
    """Map 2024 rows to 2026 rows: normalized name, then (date, first organization) if unique on both sides."""
    m = dict(zip(o.key, o.key))
    nk = set(n.key)
    mapping = {}
    for k in o.key:
        if k in nk:
            mapping[k] = k
    un_o = o[~o.key.isin(mapping)]
    used = set(mapping.values())
    un_n = n[~n.key.isin(used)]
    by = un_n.groupby(["date", "org1"])
    for _, r in un_o.iterrows():
        try:
            cand = by.get_group((r.date, r.org1))
        except KeyError:
            continue
        same = un_o[(un_o.date == r.date) & (un_o.org1 == r.org1)]
        if len(cand) == 1 and len(same) == 1:
            mapping[r.key] = cand.key.iloc[0]
    del m
    return mapping


def run():
    o = load(RC.EPOCH_ALL_2024, "System", RC.SNAPSHOT_2024)
    n_all = load(RC.EPOCH_ALL_2026, "Model", RC.SNAPSHOT_2026)
    n = n_all[n_all.date <= pd.Timestamp(END)].copy()
    # matching on unique names (a few names repeat in the database, e.g. model sizes sharing a name: keep the row with
    # the largest compute, the one that can enter the frontier); the fitted samples keep every row, as ra3 does
    od = o.sort_values("C", ascending=False, na_position="last").drop_duplicates("key")
    nd = n.sort_values("C", ascending=False, na_position="last").drop_duplicates("key")
    mp = match(od, nd)
    inv = {v: k for k, v in mp.items()}
    nC = nd.set_index("key")["C"]
    rev = o.copy()                                       # V24 + revisions (2026 compute for matched; withdrawn -> NaN)
    rev["C"] = [nC.get(mp[k], np.nan) if k in mp else np.nan for k in rev.key]
    rev["C_2024"] = o["C"].values
    add_rows = n[~n.key.isin(inv)].copy()                # 2026 entries released <= May 2024 with no 2024 counterpart
    add = pd.concat([o, add_rows], ignore_index=True)    # V24 + additions
    samples = [("v24", "Database of 31 May 2024", o), ("v24_rev", "31 May 2024 + compute revisions to Sep. 2026", rev),
               ("v24_add", "31 May 2024 + entries added after May 2024", add), ("v26", "Database of 23 Sep. 2026", n)]
    rows = []
    tops = {}
    for j, (key, lab, d) in enumerate(samples):
        dd = running_top10(d)
        tops[key] = dd
        rows.append(dict(key=key) | fit(dd, SEED + 100 + j, lab))
        ddn = running_top10(d[d.notable])
        rows.append(dict(key=key + "_notable") | fit(ddn, SEED + 200 + j, lab + ", notable models only"))
    R = pd.DataFrame(rows)
    # ra3's 2018-2026 estimate on V26 (primary in the paper) and Epoch's published rate, for the table
    full = running_top10(n_all)
    s = full[full["top10"] & (full["date"] >= START)]
    r = GR.wild_cluster(s["lc"].values, np.c_[np.ones(len(s)), s["t"].values - RC.T0], pd.factorize(s["dev"])[0],
                        seed=SEED + 300, nulls=[("4.2x", np.log10(4.2))])
    R = pd.concat([R, pd.DataFrame([dict(key="v26_full", sample="Database of 23 Sep. 2026, Jan. 2018 to Sep. 2026 (Section V)", n=len(s),
                                         clusters=int(r["G"]), growth=10 ** r["b1"], growth_lo=10 ** r["b1_lo"],
                                         growth_hi=10 ** r["b1_hi"], p_wcr_4p2=r["p_4.2x"], models=len(n_all))])],
                  ignore_index=True)
    # Shapley decomposition of ln growth (all-model samples)
    g = R.set_index("key")["growth"]
    l24, lrev, ladd, l26 = [np.log(g[key]) for key, _, _ in samples]
    dec = dict(total=l26 - l24, revisions=0.5 * ((lrev - l24) + (l26 - ladd)), additions=0.5 * ((ladd - l24) + (l26 - lrev)))
    D = pd.DataFrame([dict(component=k, dln_growth=v, factor=np.exp(v), share=v / dec["total"]) for k, v in dec.items()])
    # which frontier entries changed
    t24 = tops["v24"]
    t26 = tops["v26"]
    f24 = t24[t24.top10 & (t24.date >= START) & (t24.date <= END)]
    f26 = t26[t26.top10 & (t26.date >= START) & (t26.date <= END)]
    f26k = set(f26.key)
    f24_to_26 = {mp.get(k, None) for k in f24.key}
    chg = []
    for _, r_ in f26.iterrows():
        k24 = inv.get(r_.key)
        if k24 is None:
            chg.append(dict(model=r_["name"], date=r_.date.date(), C_2026=r_.C, C_2024=np.nan, change="added after May 2024"))
        elif r_.key not in f24_to_26:
            c24 = od.set_index("key")["C"].get(k24, np.nan)
            chg.append(dict(model=r_["name"], date=r_.date.date(), C_2026=r_.C, C_2024=c24,
                            change="entered the running top 10 (revised compute or ranking)"))
    for _, r_ in f24.iterrows():
        k26 = mp.get(r_.key)
        if k26 is None or k26 not in f26k:
            c26 = nC.get(k26, np.nan) if k26 else np.nan
            chg.append(dict(model=r_["name"], date=r_.date.date(), C_2026=c26, C_2024=r_.C,
                            change="left the running top 10" + (" (compute withdrawn)" if not np.isfinite(c26) else "")))
    Chg = pd.DataFrame(chg).sort_values("date")
    info = dict(n_2024=len(od), n_2026_to_may24=len(nd), matched=len(mp), matched_by_date=sum(1 for k, v in mp.items() if k != v),
                added=len(add_rows), withdrawn=int((rev["C_2024"].notna().values & rev["C"].isna().values).sum()),
                compute_added=int((rev["C_2024"].isna().values & rev["C"].notna().values).sum()),
                revised_5pct=int((np.abs(np.log(rev["C"].values / rev["C_2024"].values)) > np.log(1.05)).sum()))
    log(f"vintage: V24 {g['v24']:.2f}x, V26 {g['v26']:.2f}x; "
        f"revisions {D.set_index('component').loc['revisions', 'share']:.0%}, additions "
        f"{D.set_index('component').loc['additions', 'share']:.0%} of the log difference")
    return dict(R=R, D=D, Chg=Chg, info=info)
