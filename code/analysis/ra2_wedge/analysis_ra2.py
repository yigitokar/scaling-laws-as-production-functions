"""analysis_ra2.py -- cleaning table, technology dispersion, conventions, in-support splits, partial identification of
M*(C) at frontier scale, conduct tests, validation of levels, cost sensitivity.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

import sample as SM
import wedges as WG
from ra2common import RAW, SEED, ln_mstar, log, log_w, share, wild_cluster_boot

REF = "chin_q"          # kappa-free Chinchilla reference
OLDREF = "chin"         # m3's reference (kappa = 1)
B_WCR = 9999            # wild cluster bootstrap draws (floor 1e-4)


# ============================================================================ (1) cleaning table
def cleaning_table(B, L, bandtab):
    Lr = L[L["tech"] == REF].set_index("uid")
    Lo = L[L["tech"] == OLDREF].set_index("uid")
    bt = bandtab.set_index("uid")
    rows = []
    for step, keep in SM.cleaning_steps(B):
        u = B.loc[keep, "uid"]
        w = Lr.loc[u, "w"]
        rows.append(dict(step=step, label=SM.STEP_LABEL[step], n=int(keep.sum()),
                         n_dropped=0 if step == "start" else int((B["drop_step"] == step).sum()),
                         median_w=w.median(), share_w_gt1=(w > 1).mean(), share_band_gt1=(bt.loc[u, "band_lo"] > 1).mean(),
                         share_allpoints_gt1=(bt.loc[u, "pt_min"] > 1).mean(),
                         median_s=share(w).median(), median_w_kappa1=Lo.loc[u, "w"].median(),
                         share_w_gt1_kappa1=(Lo.loc[u, "w"] > 1).mean(), median_M=B.loc[keep, "M"].median(),
                         n_families=B.loc[keep, "gen"].nunique(), n_developers=B.loc[keep, "dev"].nunique()))
    # review addition (R2 Major 10 / minor 22; R3 M5.3): robustness row, not part of the ex-ante definition -- the clean
    # sample without models trained mainly on teacher-generated synthetic data (m3 flag: phi-1.5, phi-2, SmolLM 1/2/3),
    # whose teacher inference is an omitted training input and whose tokens are multi-epoch
    keep = B["clean"] & ~B["f_synthetic"]
    u = B.loc[keep, "uid"]
    w = Lr.loc[u, "w"]
    rows.append(dict(step="r_synthetic", label="(robustness) clean without synthetic-data models", n=int(keep.sum()),
                     n_dropped=int((B["clean"] & B["f_synthetic"]).sum()),
                     median_w=w.median(), share_w_gt1=(w > 1).mean(), share_band_gt1=(bt.loc[u, "band_lo"] > 1).mean(),
                     share_allpoints_gt1=(bt.loc[u, "pt_min"] > 1).mean(),
                     median_s=share(w).median(), median_w_kappa1=Lo.loc[u, "w"].median(),
                     share_w_gt1_kappa1=(Lo.loc[u, "w"] > 1).mean(), median_M=B.loc[keep, "M"].median(),
                     n_families=B.loc[keep, "gen"].nunique(), n_developers=B.loc[keep, "dev"].nunique()))
    return pd.DataFrame(rows)


# ============================================================================ (2) technology dispersion (clean sample)
def tech_dispersion(B, T, M, L, mask):
    X = B[mask]
    rows = []
    for _, m in M[M["in_set"] | M["sensitivity"]].iterrows():
        t = T[m["key"]]
        jb = WG.joint_boot(X, t)
        lk = L[(L["tech"] == m["key"]) & L["uid"].isin(X["uid"])]
        rows.append(dict(key=m["key"], label=m["label"], group=m["group"], form=m["form"], n_conv=m["n_conv"],
                         in_set=m["in_set"], sensitivity=m["sensitivity"], a=t.a, sigma_star=t.sigma_star,
                         S2=(t.alpha + t.beta) / 2,
                         Mstar_1e21=float(np.exp(ln_mstar(1e21, t.alpha, t.beta, t.lnG))),
                         Mstar_1e24=float(np.exp(ln_mstar(1e24, t.alpha, t.beta, t.lnG))),
                         share_M_extrap=lk["M_extrap"].mean(), share_C_extrap=lk["C_extrap"].mean(), **jb))
    return pd.DataFrame(rows)


# ============================================================================ (4) conventions
def conventions(B, T, L, mask):
    X = B[mask & B["N_nonemb"].notna()].copy()
    rows = []
    for k in ["chin_q", "chin", "farseer_emb", "chin_ne", "farseer", "farseer_q", "minicpm"]:
        lk = L[(L["tech"] == k)].set_index("uid").loc[X["uid"]]
        w = lk["w"].values
        conv = T[k].n_conv
        # non-embedding technologies: count the output head in FLOPs (training and serving both scale with
        # N_ne + V d): lifetime/training ratio = w (1 + N_head / N_ne)
        adj = w * (1 + X["N_head"].values / X["N_nonemb"].values) if conv == "nonemb" else w
        small = X["N"].values < 1e9
        rows.append(dict(tech=k, conv=conv, n=len(X), median_w=np.median(w), share_w_gt1=(w > 1).mean(),
                         median_s=np.median(share(w)), median_w_headFLOP=np.median(adj),
                         median_s_headFLOP=np.median(share(adj)), median_w_sub1B=np.median(w[small]) if small.any() else np.nan,
                         n_sub1B=int(small.sum())))
    return pd.DataFrame(rows)


# ============================================================================ (6) in-support vs extrapolated
def in_support(B, L, bandtab, mask):
    X = B[mask].copy()
    Lr = L[L["tech"] == REF].set_index("uid")
    Lf = L[L["tech"] == "farseer_q"].set_index("uid")
    bt = bandtab.set_index("uid")
    X["C"] = 6 * X["N"] * X["D"]
    X["in_chin"] = (X["M"] <= 341.1) & (X["C"] <= 1.3e22)
    X["in_chinM"] = X["M"] <= 341.1
    Mne = X["D"] / X["N_nonemb"]
    X["in_farseer"] = (Mne <= 2569.7) & (X["N_nonemb"].between(9.96e7, 6.37e9)) & (6 * X["N_nonemb"] * X["D"] <= 3.5e21)
    X["in_farseerM"] = Mne <= 2569.7
    rows = []
    for flag, lab, tk in [("in_chin", "Chinchilla design (M <= 341 and C <= 1.3e22)", REF),
                          ("in_chinM", "Chinchilla M range only (M <= 341)", REF),
                          ("in_farseer", "Farseer design (non-emb. M <= 2,570, N and C in support)", "farseer_q"),
                          ("in_farseerM", "Farseer M range only (non-emb. M <= 2,570)", "farseer_q")]:
        Lt = Lr if tk == REF else Lf
        for val in (True, False):
            g = X[X[flag] == val]
            if tk != REF:
                g = g[g["N_nonemb"].notna()]
            w = Lt.loc[g["uid"], "w"]
            rows.append(dict(region=lab, tech=tk, inside=val, n=len(g), median_w=w.median(), share_w_gt1=(w > 1).mean(),
                             median_s=share(w).median(), share_band_gt1=(bt.loc[g["uid"], "band_lo"] > 1).mean(),
                             median_M=g["M"].median()))
    return pd.DataFrame(rows), X[["uid", "in_chin", "in_chinM", "in_farseer", "in_farseerM"]]


def _ra1_bin(M):
    edges = [(-np.inf, 16, "M<16"), (16, 64, "16-64"), (64, 256, "64-256"), (256, 1024, "256-1,024"), (1024, np.inf, ">=1,024")]
    return np.array([next(lab for lo, hi, lab in edges if lo <= m < hi) if np.isfinite(m) else "" for m in M])


def extrapolation_check(B, L, mask, mmax):
    """Direction of the parametric extrapolation error, from module ra1's model-free local wedge on Farseer (ra1 memo H4;
    output/tables/ra1_modelfree_farseer_delta.csv and ..._sensitivity.csv). Delta = ln(w_param / w_local), averaged
    over Farseer grid points in each M bin (inside Farseer's design). For clean models whose M (in the technology's own
    convention) lies inside Farseer's M range, the 'locally corrected' wedge is w_param * exp(-Delta_bin). This applies
    Farseer's in-support gap to models of other labs and of larger N and C, so it signs the error; it does not correct
    the levels. Beyond Farseer's M range there is no model-free evidence; the in-support gap grows with M.
    mmax: {tech: largest M in Farseer's design in that technology's convention}."""
    from ra2common import RA1_FARSEER_DELTA, RA1_FARSEER_SENS
    if not os.path.exists(RA1_FARSEER_DELTA):
        return pd.DataFrame()
    dl = pd.read_csv(RA1_FARSEER_DELTA)
    sens = pd.read_csv(RA1_FARSEER_SENS) if os.path.exists(RA1_FARSEER_SENS) else pd.DataFrame(columns=["variant", "key", "value"])
    X = B[mask].copy()
    rows = []
    for tech, spec, conv in [("farseer_q", "kappa_full", "ne"), ("farseer", "chin_full", "ne"), ("farseer_emb", "chin_full", "emb")]:
        lk = L[L["tech"] == tech].set_index("uid").loc[X["uid"]]
        Mu = lk["M_used"].values
        bins = _ra1_bin(Mu)
        inside = Mu <= mmax[tech]
        variants = {"ra1 primary bandwidth": dl[(dl["spec"] == spec) & (dl["conv"] == conv)].set_index("M_bin")["delta"]}
        vv = sens[sens["variant"] == "extended-grid CV optimum"]
        dd = {k.split("|")[-1]: v for k, v in zip(vv["key"], vv["value"]) if str(k).startswith(f"delta|{spec}|{conv}|")}
        if dd:
            variants["ra1 extended-grid CV bandwidth"] = pd.Series(dd)
        for vname, dser in variants.items():
            dlt = np.array([dser.get(b, np.nan) for b in bins], float)
            ok = inside & np.isfinite(dlt)
            wp = lk["w"].values[ok]
            wloc = wp * np.exp(-dlt[ok])
            rows.append(dict(tech=tech, ra1_spec=f"{spec}|{conv}", variant=vname, region="inside Farseer M range",
                             M_max=mmax[tech], n=int(ok.sum()),
                             median_w_param=float(np.median(wp)), median_w_local_implied=float(np.median(wloc)),
                             median_s_param=float(np.median(share(wp))), median_s_local_implied=float(np.median(share(wloc))),
                             share_param_understates=float(np.mean(dlt[ok] < 0)),
                             n_highM_bin=int(np.sum(bins[ok] == ">=1,024")),
                             delta_highM=float(dser.get(">=1,024", np.nan)), delta_256_1024=float(dser.get("256-1,024", np.nan))))
        rows.append(dict(tech=tech, ra1_spec=f"{spec}|{conv}", variant="--",
                         region="outside Farseer M range (no model-free evidence)", M_max=mmax[tech], n=int((~inside).sum()),
                         median_w_param=float(np.median(lk["w"].values[~inside])) if (~inside).any() else np.nan,
                         median_s_param=float(np.median(share(lk["w"].values[~inside]))) if (~inside).any() else np.nan))
    return pd.DataFrame(rows)


# ============================================================================ (5) partial identification of M*(C)
def pi_anchors(T, M, mf_table, mf_draws):
    """Anchors = each technology's fitted ln M*(C0) at its own largest design budget C0 (in its own N convention),
    with a 95% interval from its draws. 'iso' = IsoFLOP designs' own A2 paths (Chinchilla, Llama 3, Marin x3) and
    DeepSeek's published law at its largest budget; 'all' = every ex-ante technology (in-support M*)."""
    rows = []
    iso_map = {"chin_iso": None, "meta": "meta_mf", "marin_comma": "marin_comma_mf", "marin_dclm": "marin_dclm_mf",
               "marin_nemotron": "marin_nemotron_mf"}
    for des in iso_map:
        r = mf_table[(mf_table.design == des) & (mf_table.deg == 2)].iloc[0]
        dr = mf_draws[des]
        ok = np.isfinite(dr).all(1)
        lnMs = -2 * dr[ok, 2] + (1 - 2 * dr[ok, 1]) * np.log(r["Cmax"] / 6)
        pt = -2 * r["lnG"] + (1 - 2 * r["a"]) * np.log(r["Cmax"] / 6)
        rows.append(dict(anchor=des, kind="iso", conv="total", C0=r["Cmax"], lnMs=pt, lnMs_lo=np.percentile(lnMs, 2.5),
                         lnMs_hi=np.percentile(lnMs, 97.5), a=r["a"], lab={"meta": "Meta"}.get(des, "Marin" if "marin" in des else "")))
    t = T["deepseek"]
    pt = ln_mstar(3e20, t.alpha, t.beta, t.lnG)
    rows.append(dict(anchor="deepseek", kind="iso", conv="ds", C0=3e20, lnMs=pt, lnMs_lo=pt, lnMs_hi=pt, a=t.a, lab="DeepSeek"))
    for _, m in M[M["in_set"]].iterrows():
        t = T[m["key"]]
        C0 = t.support.get("C_max", np.nan)
        if not np.isfinite(C0):
            continue
        pt = ln_mstar(C0, t.alpha, t.beta, t.lnG)
        lo = hi = pt
        if t.draws is not None and len(t.draws):
            v = ln_mstar(C0, t.draws[:, 0], t.draws[:, 1], t.draws[:, 2])
            lo, hi = np.nanpercentile(v, [2.5, 97.5])
        rows.append(dict(anchor=m["key"], kind="all", conv=t.n_conv, C0=C0, lnMs=pt, lnMs_lo=lo, lnMs_hi=hi, a=t.a,
                         lab={"olmo": "AI2", "olmo_q": "AI2", "deepseek": "DeepSeek", "meta_mf": "Meta", "meta_a2": "Meta",
                              "meta_a3": "Meta"}.get(m["key"], "Marin" if m["key"].startswith("marin") else "")))
    return pd.DataFrame(rows)


def pi_bounds(B, mask, anchors, e_range, S2_range, sets):
    """For each model and PI assumption set: bounds on ln M - ln M*(C) (sign identification) and on w, s.
    sets: dict name -> dict(kinds=[...], e=(lo, hi), own_lab=bool)."""
    X = B[mask].copy()
    out = []
    for name, spec in sets.items():
        e_lo, e_hi = spec["e"]
        for _, r in X.iterrows():
            A = anchors[anchors["kind"].isin(spec["kinds"])]
            if spec.get("own_lab") and r["dev"] in set(A["lab"]):
                A = A[A["lab"] == r["dev"]]
            dlo, dhi = np.inf, -np.inf
            for _, a in A.iterrows():
                N = WG.n_used(pd.DataFrame([r]), a["conv"])[0]
                if not np.isfinite(N):
                    continue
                C = 6 * N * r["D"]
                lnM = np.log(r["D"] / N)
                x = np.log(C / a["C0"])
                ms_hi = a["lnMs_hi"] + (e_hi * x if x > 0 else e_lo * x)
                ms_lo = a["lnMs_lo"] + (e_lo * x if x > 0 else e_hi * x)
                dlo, dhi = min(dlo, lnM - ms_hi), max(dhi, lnM - ms_lo)
            s_lo, s_hi = S2_range
            lnw_lo = (s_lo if dlo > 0 else s_hi) * dlo
            lnw_hi = (s_hi if dhi > 0 else s_lo) * dhi
            out.append(dict(set=name, uid=r["uid"], model=r["model"], dev=r["dev"], M=r["M"], C=r["Cmp"], dlo=dlo, dhi=dhi,
                            sign=("w>1" if dlo > 0 else "w<1" if dhi < 0 else "ambiguous"),
                            w_lo=np.exp(lnw_lo), w_hi=np.exp(lnw_hi), s_lo=share(np.exp(lnw_lo)), s_hi=share(np.exp(lnw_hi))))
    return pd.DataFrame(out)


def pi_mstar_grid(anchors, kinds, e_range, Cs=(1e23, 1e24, 1e25), conv="total"):
    """Bounds on M*(C) (in the given N convention) at frontier compute from the anchors of `kinds`."""
    A = anchors[anchors["kind"].isin(kinds) & (anchors["conv"] == conv)]
    rows = []
    for C in Cs:
        lo, hi = np.inf, -np.inf
        for _, a in A.iterrows():
            x = np.log(C / a["C0"])
            lo = min(lo, a["lnMs_lo"] + e_range[0] * x)
            hi = max(hi, a["lnMs_hi"] + e_range[1] * x)
        rows.append(dict(C=C, Mstar_lo=np.exp(lo), Mstar_hi=np.exp(hi), n_anchors=len(A)))
    return pd.DataFrame(rows)


# ============================================================================ (7) conduct tests
def _design(df, cols, year_fe=True, extra=()):
    X = [np.ones(len(df))]
    names = ["const"]
    for c in cols + list(extra):
        X.append(df[c].values.astype(float))
        names.append(c)
    if year_fe:
        yrs = sorted(df["year"].unique())[1:]
        for y in yrs:
            X.append((df["year"] == y).values.astype(float))
            names.append(f"y{y}")
    return np.column_stack(X), names


def wcr(df, y, cols, test, extra=(), year_fe=True, B=B_WCR, seed=SEED):
    X, names = _design(df, cols, year_fe, extra)
    j = names.index(test)
    r = wild_cluster_boot(df[y].values, X, df["dev"].values, j, B=B, seed=seed)
    # review addition: for a binary regressor, the number of clusters with any treated / any untreated observation.
    # With very few treated clusters the restricted wild cluster bootstrap is unreliable (it under-rejects severely;
    # mackinnon2018wild), so p-values with G_treated <= 3 are not interpretable.
    v = df[test].values
    g_tr = g_ut = np.nan
    if set(np.unique(v[np.isfinite(v)])) <= {0.0, 1.0}:
        dd = pd.DataFrame(dict(dev=df["dev"].values, v=v))
        g_tr = int(dd.groupby("dev")["v"].max().sum())
        g_ut = int((dd.groupby("dev")["v"].min() == 0).sum())
    return dict(n=len(df), clusters=r["G"], coef=r["beta"], se_crv1=r["se_crv1"], t=r["t"], p_wcr=r["p_boot"], B=B,
                p_floor=r["p_floor"], G_treated=g_tr, G_untreated=g_ut)


def open_closed(U, Lu):
    """(a) open vs closed premium on the clean production-scale universe: confident N/D, no MoE, no non-transformer,
    clean-sample exclusions for Sample-B rows."""
    X = U[U["confident"] & ~U["moe"] & ~U["nontransformer"] & ~U["drop_B"]].copy()
    X = X.merge(Lu[["uid", "w"]], on="uid")
    X["lnw"] = np.log(X["w"])
    X["s"] = share(X["w"])
    X["lnC"] = np.log(X["Cmp"])
    X["open"] = X["open_weights"].astype(float)
    res = []
    for spec, extra in [("year FE", ()), ("year FE + ln C", ("lnC",))]:
        for yv in ("lnw", "s"):
            r = wcr(X, yv, ["open"], "open", extra=extra)
            res.append(dict(test="open-weight premium", outcome=yv, spec=spec, **r))
    X2 = X[X["year"] >= 2023]
    for yv in ("lnw",):
        r = wcr(X2, yv, ["open"], "open", extra=("lnC",))
        res.append(dict(test="open-weight premium, 2023+", outcome=yv, spec="year FE + ln C", **r))
    return pd.DataFrame(res), X


def serving_ondevice(B, mask, L, label):
    X = B[mask].copy()
    X = X.merge(L[L["tech"] == REF][["uid", "w"]], on="uid")
    X["lnw"] = np.log(X["w"])
    X["s"] = share(X["w"])
    X["lnC"] = np.log(X["Cmp"])
    X = X[X["serve"].notna()]
    res = []
    for yv in ("lnw", "s"):
        for dep in ("ondevice", "local"):
            for test in ("serve", dep):
                r = wcr(X, yv, ["serve", dep], test, extra=("lnC",))
                res.append(dict(test=f"{test} (with {dep})", outcome=yv, sample=label, spec="ln C + year FE", **r))
    return pd.DataFrame(res), X


TIER_CAPS = np.array([3.3, 9.5, 14.9, 32.9, 72.9]) * 1e9     # upper edges of quantized memory tiers (R2 Major 5c)
TIER_LO = np.array([2.4, 6.5, 11.5, 26.0, 65.0]) * 1e9


def tier_vars(N, shift=0.0):
    """in_tier: N inside a tier window (just below a cap); dist: ln(next cap above N / N). shift multiplies every
    tier edge by exp(shift) (placebo menus)."""
    N = np.asarray(N, float)
    caps, los = TIER_CAPS * np.exp(shift), TIER_LO * np.exp(shift)
    inwin = np.zeros(len(N), bool)
    for lo, hi in zip(los, caps):
        inwin |= (N > lo) & (N <= hi)
    nxt = np.array([caps[caps >= n].min() if (caps >= n).any() else np.nan for n in N])
    dist = np.log(nxt / N)
    return inwin, dist


def bunching(Ntot, B=999, seed=SEED, deg=5, bw=0.1):
    """Multi-tier bunching (Chetty et al. 2011 / Kleven 2016 polynomial counterfactual): bin log10 N (width bw),
    fit a degree-`deg` polynomial to bin counts excluding the five tier windows, excess mass = observed - predicted
    within the windows, by tier and pooled; bootstrap over models."""
    x = np.log10(np.asarray(Ntot, float))
    edges = np.arange(np.floor(x.min() / bw) * bw, np.ceil(x.max() / bw) * bw + bw / 2, bw)
    mids = (edges[:-1] + edges[1:]) / 2
    win = [(np.log10(lo), np.log10(hi)) for lo, hi in zip(TIER_LO, TIER_CAPS)]

    def est(xx):
        cnt, _ = np.histogram(xx, edges)
        inw = np.zeros(len(mids), bool)
        tier_of = np.full(len(mids), -1)
        for j, (a, b) in enumerate(win):
            m = (mids > a) & (mids <= b)
            inw |= m
            tier_of[m] = j
        z = (mids - mids.mean()) / mids.std()
        c = np.polyfit(z[~inw], cnt[~inw], deg)
        pred = np.clip(np.polyval(c, z), 0, None)
        ex = [cnt[tier_of == j].sum() - pred[tier_of == j].sum() for j in range(len(win))]
        tot_obs = cnt[inw].sum()
        tot_pred = pred[inw].sum()
        return np.array(ex + [tot_obs - tot_pred, tot_obs / len(xx), tot_pred / len(xx)])

    pt = est(x)
    rng = np.random.default_rng(seed)
    bs = np.array([est(rng.choice(x, len(x), replace=True)) for _ in range(B)])
    se = bs.std(0, ddof=1)
    names = [f"tier {lo/1e9:g}-{hi/1e9:g}B" for lo, hi in zip(TIER_LO, TIER_CAPS)] + ["pooled", "share_in_windows", "share_pred"]
    return pd.DataFrame(dict(term=names, estimate=pt, se=se, z=pt / se, n=len(x), B=B))


def _ols_coef(y, X, j):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b[j]


def tier_regression(X, label, n_placebo=999, seed=SEED):
    """ln w on tier position given ln C and year. Because ln w is a deterministic function of (N, C) under one
    technology, any function of N correlates with ln w at given C mechanically (dist falls one-for-one with ln N
    inside each tier gap). Inference therefore uses (i) the restricted wild cluster bootstrap and (ii) a placebo test:
    the same regression with every tier edge multiplied by exp(u), u ~ U(-0.69, 0.69) (menus shifted by factors 0.5-2);
    p_placebo = share of placebo coefficients at least as favourable to the tier story as the actual one (tier story:
    in_tier > 0, dist < 0, i.e. models just below a cap are more over-trained given compute)."""
    X = X.copy()
    X["lnC"] = np.log(X["Cmp"])
    X = X[np.isfinite(X["lnw"])]
    rng = np.random.default_rng(seed)
    shifts = rng.uniform(-0.69, 0.69, n_placebo)
    res = []
    for test in ("dist_tier", "in_tier"):
        Xa = X.copy()
        Xa["in_tier"], Xa["dist_tier"] = tier_vars(Xa["N"].values)
        Xa["in_tier"] = Xa["in_tier"].astype(float)
        Xa = Xa[np.isfinite(Xa["dist_tier"])]
        r = wcr(Xa, "lnw", [test], test, extra=("lnC",))
        pl = []
        for u in shifts:
            Xp = X.copy()
            Xp["in_tier"], Xp["dist_tier"] = tier_vars(Xp["N"].values, u)
            Xp["in_tier"] = Xp["in_tier"].astype(float)
            Xp = Xp[np.isfinite(Xp["dist_tier"])]
            D_, names = _design(Xp, [test], True, ("lnC",))
            pl.append(_ols_coef(Xp["lnw"].values, D_, names.index(test)))
        pl = np.array(pl)
        p_pl = float(np.mean(pl >= r["coef"])) if test == "in_tier" else float(np.mean(pl <= r["coef"]))
        res.append(dict(test=test, sample=label, outcome="lnw", spec="ln C + year FE", **r,
                        placebo_mean=float(pl.mean()), placebo_sd=float(pl.std()), p_placebo_tierstory=p_pl,
                        n_placebo=n_placebo))
    return pd.DataFrame(res)


# ============================================================================ (8) validation
OPENROUTER_AUTHOR_T = {   # arXiv:2601.10088, Table 1: total tokens (trillions), Nov 2024 - Nov 2025, all variants
    "DeepSeek": 14.37, "Alibaba": 5.59, "Meta": 3.96, "Mistral AI": 2.92, "OpenAI": 1.65, "Minimax": 1.26,
    "Zhipu": 1.18, "TNGTech": 1.13, "Moonshot": 0.92, "Google": 0.82}


def hf_counts(B):
    """Model-tree derivative counts summed over base + official post-trained repos (m3 INSTRUCT map) and over the
    releases of the same pretraining run (dedupe step (a) partners)."""
    import curated as cu
    tree = {}
    for f in glob.glob(os.path.join(RAW, "ra2_hf_tree", "*.json")):
        d = json.load(open(f))
        tree[d.get("id", os.path.basename(f)[:-5].replace("__", "/"))] = d.get("childrenModelCount", {})
    partners = {"Meta-Llama-3-8B": ["Llama-3.1-8B"], "Meta-Llama-3-70B": ["Llama-3.1-70B"], "Qwen-72B": ["Qwen1.5-72B"],
                "Yi-6B": ["Yi-1.5-6B"], "Yi-34B": ["Yi-1.5-34B"], "open_llama_3b": ["open_llama_3b_v2"],
                "open_llama_7b": ["open_llama_7b_v2"], "h2o-danube-1.8b-base": ["h2o-danube2-1.8b-base"]}
    hf_of = dict(zip(B["model"], B["hf"]))
    rows = []
    for _, r in B.iterrows():
        ids = [r["hf"]] + [hf_of[p] for p in partners.get(r["model"], []) if p in hf_of]
        ids += [i for h in list(ids) for i in cu.INSTRUCT.get(h, [])]
        tot = dict(adapter=0, merge=0, quantized=0, finetune=0)
        found = 0
        for i in dict.fromkeys(ids):
            if i in tree:
                found += 1
                for k in tot:
                    tot[k] += tree[i].get(k, 0)
        rows.append(dict(uid=r["uid"], n_repos_found=found, **{f"hf_{k}": v for k, v in tot.items()},
                         hf_total=sum(tot.values()) if found else np.nan))
    out = pd.DataFrame(rows)
    for k in ["adapter", "merge", "quantized", "finetune"]:
        out.loc[out["n_repos_found"] == 0, f"hf_{k}"] = np.nan
    return out


def hf_validation(B, mask, L, H):
    X = B[mask].merge(L[L["tech"] == REF][["uid", "w"]], on="uid").merge(H, on="uid")
    X = X[X["n_repos_found"] > 0].copy()
    X["lnC"] = np.log(X["Cmp"])
    X["lnM"] = np.log(X["M"])
    X["lnTD"] = np.log(np.clip(3 * (X["w"] - 1), 1e-3, None))
    res = []
    for outc in ["hf_quantized", "hf_finetune", "hf_adapter", "hf_total"]:
        X["y"] = np.log1p(X[outc])
        for reg in ["lnM", "lnTD"]:
            r = wcr(X, "y", [reg], reg, extra=("lnC",))
            res.append(dict(outcome=f"ln(1 + {outc})", regressor=reg, spec="ln C + year FE", **r))
    return pd.DataFrame(res), X


def openrouter_audit(B, L):
    """Order-of-magnitude audit: planned lifetime tokens (p = 1, reference technology; all Sample-B releases of the
    author through 2025, MoE at active N) vs OpenRouter tokens served in one year (all variants)."""
    X = B.merge(L[L["tech"] == REF][["uid", "w"]], on="uid")
    X = X[~X["f_dup"]]
    X["T_planned"] = 3 * X["D"] * np.clip(X["w"] - 1, 0, None)
    rows = []
    for dev, tok in OPENROUTER_AUTHOR_T.items():
        g = X[X["dev"] == dev]
        if not len(g):
            continue
        rows.append(dict(author=dev, openrouter_T_1yr=tok * 1e12, n_models=len(g), planned_T_lifetime=g["T_planned"].sum(),
                         ratio=tok * 1e12 / g["T_planned"].sum()))
    return pd.DataFrame(rows)


def s_aggregate(df, wcol, by="year", drop_uid=None):
    """Compute-weighted planned inference share of lifetime compute expenditure, s_agg = sum (w-1)+ C / sum w+ C with
    w+ = max(w, 1) (w < 1 cannot be inference demand); also the untruncated version."""
    X = df.copy()
    if drop_uid is not None:
        X = X[~X["uid"].isin(drop_uid)]
    rows = []
    for k, g in X.groupby(by):
        wp = np.maximum(g[wcol], 1.0)
        rows.append({by: k, "n": len(g), "C_total": g["Cmp"].sum(),
                     "s_agg": float(((wp - 1) * g["Cmp"]).sum() / (wp * g["Cmp"]).sum()),
                     "s_agg_untrunc": float(((g[wcol] - 1) * g["Cmp"]).sum() / (g[wcol] * g["Cmp"]).sum()),
                     "top_share_C": float(g["Cmp"].max() / g["Cmp"].sum()), "top_model": g.loc[g["Cmp"].idxmax(), "model"]})
    return pd.DataFrame(rows)


# ============================================================================ (9) cost sensitivity
def cost_sensitivity(w, w_l3):
    """lifetime cost = 6 N^(1+delta) D + 2 p N^eta T (units normalized at the model's own N):
    FOC w = (1+delta) + eta K_inf/K_tr  =>  K_inf/K_tr = (w - 1 - delta)/eta,  s = (w-1-delta)/(w-1-delta+eta),
    T/D = 3 (w - 1 - delta)/(p eta). R1's multiplicative case (serving also scales like N^(1+delta)): eta = 1+delta,
    w = (1+delta)(1 + pT/(3D))."""
    rows = []
    for panel, grid in [("A: eta = 1 + delta (serving cost scales like training cost)", [(dl, 1 + dl) for dl in (-0.1, 0.0, 0.1)]),
                        ("B: delta = 0, serving-cost elasticity eta", [(0.0, e) for e in (0.5, 0.75, 1.0, 1.25)])]:
        for dl, eta in grid:
            r = (np.asarray(w) - 1 - dl) / eta
            r3 = (w_l3 - 1 - dl) / eta
            row = dict(panel=panel, delta=dl, eta=eta, median_s=float(np.median(r / (1 + r))),
                       s_llama3_8b=float(r3 / (1 + r3)), share_pos=float((r > 0).mean()))
            for p in (1, 3, 10):
                row[f"median_TD_p{p}"] = float(np.median(3 * r / p))
                row[f"TD_llama3_8b_p{p}"] = float(3 * r3 / p)
            rows.append(row)
    return pd.DataFrame(rows)
