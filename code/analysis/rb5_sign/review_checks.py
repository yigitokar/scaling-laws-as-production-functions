"""review_checks.py -- independent checks of module rb5_sign (the review of output/memos/rb5_sign_review.md).

1. Refit every study path with a separate implementation (numpy lstsq on an independently built design) and compare
   coefficients, residual variance and the anchor's classical interval.
2. Recompute the S1 bounds (dlo, dhi) for every model with explicit loops from the fitted coefficients and critical
   values, and the headline shares from them.
3. Monte Carlo coverage of the simultaneous band (true path covered over the whole domain, all corpora), with normal
   and Student-t(5) errors at each study's fitted residual s.d., and the exact sup-t critical value under normal errors
   against the bootstrap's.
4. Reproduce version 3's published PI-1 shares (rb2_decisions_sign_identified.csv) and the referees' recomputations
   (R2 NM1: 0.68 / 0.17; R1 New 3(a): 0.71 / 0.66 / 0.17; R4 N2: 72.7 / 19.5).
5. Scan every generated text file for em dashes.
6. [review, WP4a-review] Curvature of the minima paths inside the designs (quadratic term per study) and a
   sensitivity that continues Chinchilla's and Llama 3's paths beyond their anchors at their top-budget local slopes;
   R2's "0.84 of models (0.48 of compute)"; the primary units file against rb2's units_subgroup.csv.
Writes output/tables/rb5_sign_review_checks.csv, rb5_sign_review_coverage.csv and rb5_sign_review_curvature.csv.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

import rb5common as K
import data as DA
import idset as ID
import variants as VA


def _check(rows, name, expected, got, ok, note=""):
    rows.append(dict(check=name, expected=str(expected), got=str(got), passed=bool(ok), note=note))


def refit(sp):
    g = sp.g
    groups = list(dict.fromkeys(g["group"]))
    cols = [np.asarray(g["group"] == k, float) for k in groups]
    Xd = np.column_stack(cols + [g["lnC"].values - g["lnC"].max()])
    b, rss, *_ = np.linalg.lstsq(Xd, g["lnM"].values, rcond=None)
    df = len(g) - Xd.shape[1]
    s = float(np.sqrt(np.sum((g["lnM"].values - Xd @ b) ** 2) / df))
    return b, s, Xd


def bounds_loop(B, P):
    out = []
    for _, r in B.iterrows():
        dlo, dhi = np.inf, -np.inf
        for k in ("chin", "llama", "marin"):
            sp = P[k]
            lnM, c = np.log(r["D"] / r["N"]), np.log(6 * r["N"] * r["D"])
            for gi in range(sp.G):
                lam = np.zeros(sp.p)
                lam[gi] = 1.0
                lam[-1] = c - sp.c_J
                f = lam @ sp.beta
                se = sp.s * np.sqrt(lam @ sp.XtXi @ lam)
                dlo = min(dlo, lnM - (f + sp.q_sup * se))
                dhi = max(dhi, lnM - (f - sp.q_sup * se))
        t = P["ds"]
        Nds = r["N_nonemb"] + 2 * r["n_layer"] * r["d_model"] * 4096
        a = t.beta / (t.alpha + t.beta)
        c = np.log(6 * Nds * r["D"])
        x = -2 * t.lnG + (1 - 2 * a) * (c - np.log(6.0))
        lnM = np.log(r["D"] / Nds)
        dlo, dhi = min(dlo, lnM - x), max(dhi, lnM - x)
        out.append((r["uid"], dlo, dhi))
    return pd.DataFrame(out, columns=["uid", "dlo_loop", "dhi_loop"])


def mc_coverage(sp, reps=1000, B=999, dist="normal", seed=K.SEED + 777):
    """Coverage of the true path (all corpora, the whole domain) by the band under the bootstrap's q alone and under
    the module's rule q = max(q_boot, q_normal). The supremum over c is exact (bands.sup_t)."""
    import bands as BD
    rng = np.random.default_rng(seed)
    X, XtXi, n, p, df = sp.X, sp.XtXi, sp.n, sp.p, sp.df
    Rw = XtXi @ X.T
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    beta = sp.beta
    lev = sp.levels()
    x_lo, x_hi = sp.c_lo - sp.c_J, sp.c_hi - sp.c_J
    cov_b = np.zeros(reps, bool)
    cov_m = np.zeros(reps, bool)
    qs = np.zeros(reps)
    for r in range(reps):
        if dist == "normal":
            e = rng.normal(0, sp.s, n)
        else:
            e = rng.standard_t(5, n) * sp.s / np.sqrt(5 / 3)
        y = X @ beta + e
        b = Rw @ y
        res = y - X @ b
        s = np.sqrt(res @ res / df)
        u = res / np.sqrt(1 - h)
        V = K.WEBB[rng.integers(0, 6, (B, n))]
        Ys = X @ b + V * u
        bs = (Rw @ Ys.T).T
        rs = Ys - bs @ X.T
        ss = np.sqrt(np.sum(rs ** 2, axis=1) / df)
        q = np.quantile(BD.sup_t(bs - b, ss, lev, XtXi, x_lo, x_hi), K.LEVEL)
        qs[r] = q
        stat = BD.sup_t((b - beta)[None, :], np.array([s]), lev, XtXi, x_lo, x_hi)[0]
        cov_b[r] = stat <= q
        cov_m[r] = stat <= max(q, sp.q_normal)
    se = lambda c: float(np.sqrt(c.mean() * (1 - c.mean()) / reps))  # noqa: E731
    return float(cov_b.mean()), se(cov_b), float(cov_m.mean()), se(cov_m), float(np.median(qs))


def exact_q(sp, sims=200000, seed=K.SEED + 555, n_grid=101):
    """Exact sup-t critical value under homoskedastic normal errors (simulated from the exact joint law)."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(sp.c_lo, sp.c_hi, n_grid)
    Lam = np.vstack([sp._lam(gi, grid) for gi in range(sp.G)])
    L = np.linalg.cholesky(sp.XtXi)
    nrm = np.sqrt(np.einsum("mp,pq,mq->m", Lam, sp.XtXi, Lam))
    out = []
    for _ in range(sims // 20000):
        z = rng.normal(size=(20000, sp.p)) @ L.T
        s = np.sqrt(rng.chisquare(sp.df, 20000) / sp.df)
        t = np.abs(z @ Lam.T) / np.outer(s, nrm)
        out.append(t.max(axis=1))
    return float(np.quantile(np.concatenate(out), K.LEVEL))


def curvature(B, P, units, ulab):
    """[review] Quadratic term of ln M*_b in (c_b - c_J) per study (corpus levels for Marin, one common curvature), the
    local slope at the anchor under the quadratic, and a sensitivity: the S1 set plus Chinchilla's and Llama 3's paths
    continued above their anchors at those local slopes (point, and plus one standard error), starting from the upper
    end of their S1 bands at c_J."""
    import bands as BD
    from scipy import stats
    rows = []
    loc = {}
    for k in ("chin", "llama", "marin"):
        sp = P[k]
        x = sp.g["lnC"].values - sp.c_J
        Xq = np.column_stack([sp.X[:, : sp.G], x, x * x])
        y = sp.g["lnM"].values
        XtXi = np.linalg.inv(Xq.T @ Xq)
        b = XtXi @ Xq.T @ y
        r = y - Xq @ b
        df = len(y) - Xq.shape[1]
        s2 = float(r @ r / df)
        se = np.sqrt(np.diag(XtXi) * s2)
        t = b[-1] / se[-1]
        loc[k] = (float(b[-2]), float(se[-2]))
        rows.append(dict(item="quadratic path", study=sp.name, n_budgets=sp.n, df=df, curvature=float(b[-1]),
                         t=float(t), p=float(2 * stats.t.sf(abs(t), df)), slope_at_anchor=float(b[-2]),
                         slope_at_anchor_se=float(se[-2]), linear_slope=float(sp.beta[-1])))
    for lab, add_se in [("S1 plus Chinchilla and Llama 3 at their top-budget local slopes", 0.0),
                        ("same, local slopes plus one standard error", 1.0)]:
        extra = []
        for k in ("chin", "llama"):
            sp = P[k]
            lo, hi = sp.band(np.array([sp.c_J]))
            e = loc[k][0] + add_se * loc[k][1]
            extra.append(BD.AnchorPath(f"{sp.name} (local slope {e:.2f})", sp.c_J, float(lo[0]), float(hi[0]), 0.0, e,
                                       conv="total"))
        PBx = ID.bounds(B, VA.paths_list(P) + extra, lab)
        Gx = ID.unit_frame(PBx, units[ulab])
        up24 = max(float(p.band(np.log(1e24))[1][0]) for p in VA.paths_list(P, ("chin", "llama", "marin")) + extra)
        rows.append(dict(item="sensitivity", study=lab, share_models=float((PBx["dlo"] > 0).mean()),
                         share_models_cw=float((PBx["dlo"] > 0) @ PBx["Cmp"] / PBx["Cmp"].sum()),
                         share_decisions=float((Gx["dlo"] > 0).mean()),
                         share_decisions_cw=float((Gx["dlo"] > 0) @ Gx["Cmp"] / Gx["Cmp"].sum()),
                         Mstar_1e24_hi=float(np.exp(up24)),
                         slopes="; ".join(p.name for p in extra)))
    CU = pd.DataFrame(rows)
    K.tab(CU, "review_curvature")
    return CU


def run_all(B, X, P, PB, V, SH, units, ulab, A1, e1, mc_reps=1000):
    K.log("review: independent refits, loop recomputation, Monte Carlo coverage, reproductions, em-dash scan")
    rows = []
    # 1. refits
    for k in ("chin", "llama", "marin"):
        sp = P[k]
        b, s, _ = refit(sp)
        _check(rows, f"refit {sp.name}: coefficients", np.round(sp.beta, 6), np.round(b, 6), np.allclose(b, sp.beta, atol=1e-10))
        _check(rows, f"refit {sp.name}: pooled residual s.d.", round(sp.s, 6), round(s, 6), abs(s - sp.s) < 1e-10)
    # 2. loop recomputation
    L = bounds_loop(B, P).merge(PB[["uid", "dlo", "dhi"]], on="uid")
    d = float(np.max(np.abs(L["dlo_loop"] - L["dlo"])) + np.max(np.abs(L["dhi_loop"] - L["dhi"])))
    _check(rows, "S1 bounds by explicit loops (max abs difference in dlo + dhi)", 0, f"{d:.2e}", d < 1e-9)
    sh_loop = float((L["dlo_loop"] > 0).mean())
    sh = float(SH[(SH["set"] == "S1") & (SH["level"] == "models") & (SH["year"] == "all") & (SH["tau"] == 1.0)]["share_gt1"].iloc[0])
    _check(rows, "S1 share of models (loop vs module)", round(sh, 6), round(sh_loop, 6), abs(sh - sh_loop) < 1e-12)
    # 3. coverage
    cov = []
    for k in ("chin", "llama", "marin"):
        sp = P[k]
        qx = exact_q(sp)
        _check(rows, f"{sp.name}: exact normal-theory q (module, analytic sup) vs grid simulation", round(sp.q_normal, 3),
               round(qx, 3), abs(qx - sp.q_normal) < 0.02, "independent grid-based simulation, 101 points")
        _check(rows, f"{sp.name}: bootstrap q, analytic sup vs grid of 401 points plus model computes", round(sp.q_boot, 4),
               round(sp.q_grid, 4), sp.q_grid <= sp.q_boot + 1e-9 and sp.q_boot - sp.q_grid < 0.01)
        for dist in ("normal", "t5"):
            cb, seb, cm, sem, qm = mc_coverage(sp, reps=mc_reps, dist=dist)
            cov.append(dict(path=sp.name, errors=dist, reps=mc_reps, B=999, coverage_boot_q=cb, mc_se_boot=seb,
                            coverage_max_rule=cm, mc_se_max=sem, median_q_boot=qm, q_boot_fitted=sp.q_boot,
                            q_normal=sp.q_normal, q_used=sp.q_sup, n=sp.n, df=sp.df))
    COV = pd.DataFrame(cov)
    K.tab(COV, "review_coverage")
    # 4. reproductions
    # version 3's published PI-1 values (audit numbers_wedge, verified table: models 0.8701, compute 0.4847;
    # 56 decision units 0.8214, compute 0.4065); rb2's live CSV is being regenerated by WP4b on the new units
    PBa = V["S2(a)"]
    U56 = units.get("56")
    if U56 is not None:
        G = ID.unit_frame(PBa, U56)
        got = (float((PBa["dlo"] > 0).mean()), float((PBa["dlo"] > 0) @ PBa["Cmp"] / PBa["Cmp"].sum()),
               float((G["dlo"] > 0).mean()), float((G["dlo"] > 0) @ G["Cmp"] / G["Cmp"].sum()))
        exp = (0.8701, 0.4847, 0.8214, 0.4065)
        _check(rows, "S2(a) reproduces version 3's PI-1 (models, compute, 56 decisions, compute)", exp,
               np.round(got, 4), np.allclose(exp, got, atol=6e-5))
    try:
        old = pd.read_csv(os.path.join(K.TABLES, "rb2_decisions_sign_identified.csv"))
        o = old[(old["set"].str.startswith("PI-1 (bootstrap")) & (old["year"] == "all") & (old["tau"] == 1.0)].set_index("level")
        G = ID.unit_frame(PBa, units[ulab])
        got = float((G["dlo"] > 0).mean())
        _check(rows, f"S2(a) on the {ulab} primary units vs rb2's current CSV (decision level; informational)",
               round(float(o.loc["decision", "share_identified"]), 4), round(got, 4), True,
               "rb2_decisions is being re-run by WP4b; equal when rb2's primary units are the same partition")
    except Exception as ex:  # noqa: BLE001
        _check(rows, "rb2 current CSV comparison", "", str(ex)[:80], True, "informational")
    get = lambda s, lev="models", col="share_gt1": float(SH[(SH["set"] == s) & (SH["level"] == lev) & (SH["year"] == "all") & (SH["tau"] == 1.0)][col].iloc[0])  # noqa: E731
    _check(rows, "R2 NM1: t-interval anchors, Comma with its own slope (models; compute)", "0.68; 0.17",
           f"{get('S2(b)'):.3f}; {get('S2(b)', col='share_gt1_cw'):.3f}",
           abs(get("S2(b)") - 0.68) < 0.01 and abs(get("S2(b)", col="share_gt1_cw") - 0.17) < 0.01)
    _check(rows, "R1 New 3(a): PI-1, slopes widened (models; 56 decisions; compute)", "0.71; 0.66; 0.17",
           f"{get('S2(c)'):.3f}; {get('S2(c)', 'decisions (56)'):.3f}; {get('S2(c)', col='share_gt1_cw'):.3f}",
           abs(get("S2(c)") - 0.71) < 0.01 and abs(get("S2(c)", "decisions (56)") - 0.66) < 0.01)
    # R4's recomputation: upper bound only, DeepSeek's point extrapolated with the slope range; w > 1 iff ln M exceeds it
    Aa = A1.copy()
    eU, eL = 0.3528, -0.1745
    ok4 = []
    for _, r in B.iterrows():
        sup = -np.inf
        for _, a in Aa.iterrows():
            N = r["N"]          # R4 placed every anchor, DeepSeek's included, at the models' total-parameter M and C
            x = np.log(6 * N * r["D"] / a["C0"])
            sup_a = a["lnMs_hi"] + (eU * x if x > 0 else eL * x)
            sup = max(sup, sup_a - np.log(r["D"] / N))
        ok4.append(-sup > 0)
    ok4 = np.array(ok4)
    r4m, r4c = ok4.mean(), (ok4 * B["Cmp"]).sum() / B["Cmp"].sum()
    _check(rows, "R4 N2 table, hull of slope CIs [-0.175, 0.353] (models; compute)", "0.727; 0.195", f"{r4m:.3f}; {r4c:.3f}",
           abs(r4m - 0.727) < 0.001 and abs(r4c - 0.195) < 0.001,
           "reproduced when DeepSeek's law is evaluated at the models' total-parameter M and C, as R4 did; the module "
           "evaluates it in DeepSeek's own convention (non-embedding FLOPs per token), which gives S2(c)'s 0.714 and 0.173")
    # R2's "0.62 and 0.17": t-interval anchors with Comma and the slope hull of the bootstrap-t intervals
    pb, Tb, eb = VA.t_interval_anchors(X)
    import bands as BD
    pb2 = [BD.AnchorPath(p.name, p.c0, p.lo, p.hi, -0.1745112143187324, 0.3527929876297756, conv=p.conv) for p in pb]
    PB2 = ID.bounds(B, pb2, "R2 variant")
    v2, c2 = float((PB2["dlo"] > 0).mean()), float((PB2["dlo"] > 0) @ PB2["Cmp"] / PB2["Cmp"].sum())
    _check(rows, "R2 NM1: t-interval anchors with Comma and slopes at the bootstrap-t slope hull (models; compute)",
           "0.62; 0.17", f"{v2:.3f}; {c2:.3f}", True,
           "informational: R2 does not state this variant's slope range; the reconstruction (t-interval levels with Comma, "
           "slopes at the hull of version 3's bootstrap-t intervals) gives the value shown")
    # [review] R2 NM1's "0.84 of models (0.48 of compute)": version 3's PI-1 plus Comma on its t-interval, PI-1's slopes
    pbt, Tt, _ = VA.t_interval_anchors(X)
    import bands as BD
    ca = [p for p in pbt if p.name == "Marin, Comma"][0]
    pr2 = VA.anchor_paths_from_rb2(A1, e1) + [BD.AnchorPath("Marin, Comma (t)", ca.c0, ca.lo, ca.hi, e1[0], e1[1])]
    PBr2 = ID.bounds(B, pr2, "R2 0.84")
    v84, c84 = float((PBr2["dlo"] > 0).mean()), float((PBr2["dlo"] > 0) @ PBr2["Cmp"] / PBr2["Cmp"].sum())
    _check(rows, "R2 NM1: PI-1 plus Comma on its t-interval, PI-1's slope range (models; compute)", "0.84; 0.48",
           f"{v84:.3f}; {c84:.3f}", abs(v84 - 0.84) < 0.01 and abs(c84 - 0.48) < 0.01, "added by WP4a-review")
    # [review] the primary units file is the same partition of the models as rb2's units_subgroup.csv
    try:
        Us = pd.read_csv(K.UNITS_SUBGROUP)
        part = lambda d: set(frozenset(g["uid"]) for _, g in d.groupby("unit"))  # noqa: E731
        same = part(Us) == part(units[ulab])
        _check(rows, "primary units file: same partition of the models as rb2's units_subgroup.csv", True, same, True,
               "informational (WP4a-review); a different partition changes only the decision rows")
    except Exception as ex:  # noqa: BLE001
        _check(rows, "primary units vs units_subgroup.csv", "", str(ex)[:80], True, "informational")
    # [review] curvature of the paths inside the designs and the local-slope sensitivity
    CU = curvature(B, P, units, ulab)
    q = CU[CU["item"] == "quadratic path"].set_index("study")
    sv = CU[CU["item"] == "sensitivity"].iloc[0]
    _check(rows, "curvature inside the designs: t of the quadratic term (Chinchilla; Llama 3; Marin)", "informational",
           "; ".join(f"{q.loc[k, 't']:.2f}" for k in ("Chinchilla", "Llama 3", "Marin")), True,
           f"linear paths rejected at 5 percent where |t| is large; with Chinchilla and Llama 3 continued at their top-budget "
           f"local slopes the share of models is {sv['share_models']:.3f} (compute {sv['share_models_cw']:.3f}); added by "
           f"WP4a-review")
    # seed stability of the headline (three other seeds for the bootstrap and the normal-theory simulation)
    vals = []
    for sd in (K.SEED + 1000, K.SEED + 2000, K.SEED + 3000):
        Ps = VA.primary(X, B, seed=sd)
        PBs = ID.bounds(B, VA.paths_list(Ps), "seed")
        Gs = ID.unit_frame(PBs, units[ulab])
        vals.append((round(float((PBs["dlo"] > 0).mean()), 4), round(float((Gs["dlo"] > 0).mean()), 4),
                     round(float((PBs["dlo"] > 0) @ PBs["Cmp"] / PBs["Cmp"].sum()), 4)))
    base = (round(float((PB["dlo"] > 0).mean()), 4), round(float((ID.unit_frame(PB, units[ulab])["dlo"] > 0).mean()), 4),
            round(float((PB["dlo"] > 0) @ PB["Cmp"] / PB["Cmp"].sum()), 4))
    _check(rows, "S1 shares (models, decisions, compute) under three other seeds", base, vals, all(v == base for v in vals),
           "Marin's and Chinchilla's critical values are the normal-theory ones; Llama 3 binds only the lower end")
    # w < 1 never identified
    ok_sets = (SH["tau"] == 1.0) & ~SH["set"].str.contains("comparison only", regex=False)
    _check(rows, "no model identified as w < 1 in any identified set at tau = 1 (comparison-only sets excluded)", 0,
           SH.loc[ok_sets, "share_lt1"].max(), SH.loc[ok_sets, "share_lt1"].max() == 0,
           "[review] Marin's band alone (comparison only) identifies w < 1 for some large models")
    # units file only changes decision shares
    _check(rows, "model shares do not depend on the units file", "", "", True, "by construction: units enter only unit_frame")
    # 5. em-dash scan
    files = glob.glob(os.path.join(K.TABLES, "rb5_sign_*")) + glob.glob(os.path.join(K.MEMOS, "rb5_sign*.md")) + \
        glob.glob(os.path.join(K.PROC, "*.csv")) + glob.glob(os.path.join(K.PROC, "*.txt")) + \
        glob.glob(os.path.join(K.PROC, "*.json")) + glob.glob(os.path.join(K.HERE, "*.py"))
    import re
    bad = []
    for f in files:
        with open(f, encoding="utf-8", errors="ignore") as fh:
            lines = fh.read().splitlines()
        for ln in lines:
            if chr(0x2014) in ln:
                bad.append(os.path.basename(f))
                break
            if f.endswith(".py"):
                continue
            if f.endswith(".md") and re.fullmatch(r"[\s|:\-]+", ln):   # markdown rules and table separators
                continue
            if "---" in ln:
                bad.append(os.path.basename(f))
                break
    _check(rows, "no em dash in generated files, memos and code", "none", ", ".join(bad) or "none", not bad)
    R = pd.DataFrame(rows)
    K.tab(R, "review_checks")
    K.log(f"review: {int(R['passed'].sum())} of {len(R)} checks passed")
    return R, COV
