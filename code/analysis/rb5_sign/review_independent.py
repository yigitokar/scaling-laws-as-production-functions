"""review_independent.py -- independent re-implementation of module rb5_sign by its reviewer (round 3, WP4a-review).

Standalone: imports no rb5_sign, rb2_decisions or ra2_wedge module. It reads the raw inputs (ra1's per-budget IsoFLOP
minima, rb2's clean sample, WP4b's final 49 units, rb2's 56 family-label units, Marin's raw IsoFLOP runs, rb1's sigma*
files, version 3's PI-1 anchors from git commit 69c9ae8, the round-3 base) and DeepSeek's published law from its constants (Bi et al. 2024:
M_opt = 0.1715 C^0.5243, D_opt = 5.8316 C^0.4757, C = M D with M the non-embedding FLOPs per token). It rebuilds:
  - the three study paths by QR least squares, their joint wild bootstrap-t (Webb weights, HC2-rescaled residuals,
    pooled studentization; own seeds) and normal-theory sup-t critical values, the supremum taken on a dense grid of
    2,001 points (the module uses the exact supremum);
  - the union with DeepSeek's law, the tau widening, the shares of models, decisions (49 and 56) and compute, by year;
  - the breakdown tau, the M*(10^24) and M*(10^25) sets, the unidentified models;
  - S2(a), (a)+Comma, (b), (b'), (c) (both DeepSeek conventions), (d), (f)-total, and R2's "0.84 / 0.48";
  - S4's model-level medians and the fixed-zero-point decision medians;
  - a Monte Carlo of the band's coverage (bootstrap q alone and the max rule).
Then it compares each quantity with the module's CSVs and writes output/tables/rb5_sign_review_independent.csv.
Run (about a minute, one CPU process):  nice -n 10 .venv/bin/python code/analysis/rb5_sign/review_independent.py
Differences are expected only from Monte Carlo (bootstrap seeds) in the critical values; shares must agree exactly.
"""
import io
import os
import subprocess
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
TAB = os.path.join(ROOT, "output", "tables")
GIT = "/opt/homebrew/bin/git"
WEBB = np.array([-np.sqrt(1.5), -1, -np.sqrt(0.5), np.sqrt(0.5), 1, np.sqrt(1.5)])

# ------------------------------------------------------------------------------------------------ inputs
mods = pd.read_csv(os.path.join(ROOT, "data/processed/rb2_decisions/clean_models.csv"))
mods["M"] = mods["D"] / mods["N"]
mods["Cmp"] = 6 * mods["N"] * mods["D"]
lnM_T = np.log(mods["M"].values)
c_T = np.log(mods["Cmp"].values)
N_ds = mods["N_nonemb"].values + 2 * mods["n_layer"].values * mods["d_model"].values * 4096
lnM_ds = np.log(mods["D"].values / N_ds)
c_ds = np.log(6 * N_ds * mods["D"].values)
Cmp = mods["Cmp"].values
U49 = pd.read_csv(os.path.join(ROOT, "data/processed/rb5_units/units_primary.csv"))
U56 = pd.read_csv(os.path.join(ROOT, "data/processed/rb2_decisions/units_primary.csv"))
bud = pd.read_csv(os.path.join(TAB, "ra1_modelfree_isoflop_budgets.csv"))


def design(d):
    g = bud[bud["design"] == d].sort_values("budget_C")
    return np.log(g["budget_C"].values), np.log(g["Mstar"].values)


def ds_path(c):
    return np.log(5.8316 * 6 / 0.1715) + (0.4757 - 0.5243) * c


E_PUB = [1 - 2 * 0.450, 1 - 2 * 0.524, 1 - 2 * 0.578, 1 - 2 * (0.23 / 0.52)]   # DeepSeek Table 4; MiniCPM


class Study:
    def __init__(self, name, corpora, dom, B=9999, seed=1, grid_n=2001):
        self.name = name
        cs = np.concatenate([c for _, c, _ in corpora])
        ys = np.concatenate([y for _, _, y in corpora])
        gid = np.concatenate([[i] * len(c) for i, (_, c, _) in enumerate(corpora)])
        self.cJ = cs.max()
        G = len(corpora)
        X = np.zeros((len(ys), G + 1))
        X[np.arange(len(ys)), gid] = 1
        X[:, G] = cs - self.cJ
        self.X, self.G, self.n, self.p = X, G, len(ys), G + 1
        self.df = self.n - self.p
        Q, R = np.linalg.qr(X)
        self.b = np.linalg.solve(R, Q.T @ ys)
        self.e = ys - X @ self.b
        self.s = np.sqrt(self.e @ self.e / self.df)
        Ri = np.linalg.inv(R)
        self.V = Ri @ Ri.T
        self.h = np.sum(Q * Q, axis=1)
        self.dom = (min(dom[0], self.cJ), max(dom[1], self.cJ))
        self.grid = np.linspace(*self.dom, grid_n)
        L = self.lams(self.grid)
        self.nrm = np.sqrt(np.einsum("ij,jk,ik->i", L, self.V, L))
        self.L = L
        rng = np.random.default_rng(seed)
        u = self.e / np.sqrt(1 - self.h)
        W = WEBB[rng.integers(0, 6, size=(B, self.n))]
        Ys = (X @ self.b)[None, :] + W * u[None, :]
        bs = np.linalg.solve(R, Q.T @ Ys.T).T
        ss = np.sqrt(((Ys - bs @ X.T) ** 2).sum(1) / self.df)
        tmax = np.concatenate([(np.abs((bs[i:i + 2000] - self.b) @ L.T) / (ss[i:i + 2000, None] * self.nrm)).max(1)
                               for i in range(0, B, 2000)])
        self.q_boot = float(np.quantile(tmax, 0.95))
        rng2 = np.random.default_rng(seed + 99)
        Lc = np.linalg.cholesky(self.V)
        qs = []
        for _ in range(20):
            z = rng2.normal(size=(20000, self.p)) @ Lc.T
            s = np.sqrt(rng2.chisquare(self.df, 20000) / self.df)
            qs.append((np.abs(z @ L.T) / (s[:, None] * self.nrm)).max(1))
        self.q_norm = float(np.quantile(np.concatenate(qs), 0.95))
        self.q = max(self.q_boot, self.q_norm)
        self.t_pt = stats.t.ppf(0.975, self.df)

    def lams(self, c, gi=None):
        c = np.atleast_1d(c)
        out = []
        for g in (range(self.G) if gi is None else [gi]):
            L = np.zeros((len(c), self.p))
            L[:, g] = 1
            L[:, -1] = c - self.cJ
            out.append(L)
        return np.vstack(out)

    def band(self, c, q=None):
        q = self.q if q is None else q
        c = np.atleast_1d(c)
        los, his = [], []
        for g in range(self.G):
            L = self.lams(c, g)
            f = L @ self.b
            se = self.s * np.sqrt(np.einsum("ij,jk,ik->i", L, self.V, L))
            los.append(f - q * se)
            his.append(f + q * se)
        return np.min(los, 0), np.max(his, 0)


def union_bounds(bands_T, extra=()):
    dlo = np.full(len(mods), np.inf)
    dhi = np.full(len(mods), -np.inf)
    for lo, hi in bands_T:
        dlo, dhi = np.minimum(dlo, lnM_T - hi), np.maximum(dhi, lnM_T - lo)
    for lnm, lo, hi in extra:
        dlo, dhi = np.minimum(dlo, lnm - hi), np.maximum(dhi, lnm - lo)
    return dlo, dhi


def unit_dlo(dlo, U):
    x = pd.DataFrame(dict(uid=mods["uid"], dlo=dlo, Cmp=Cmp, year=mods["year"])).merge(U[["uid", "unit"]], on="uid")
    return x.groupby("unit").agg(dlo=("dlo", "min"), Cmp=("Cmp", "sum"), year=("year", "min"))


def shares(dlo, tau=1.0):
    lt = np.log(tau)
    idm = dlo > lt
    sm = (mods["N"] < 15e9).values
    u49, u56 = unit_dlo(dlo, U49), unit_dlo(dlo, U56)
    cw = lambda i, w: float((i * w).sum() / w.sum())  # noqa: E731
    return {"models": idm.mean(), "models_cw": cw(idm, Cmp), "models below 15B": idm[sm].mean(),
            "decisions (49)": (u49["dlo"] > lt).mean(), "decisions_cw (49)": cw(u49["dlo"].values > lt, u49["Cmp"].values),
            "decisions (56)": (u56["dlo"] > lt).mean(), "decisions_cw (56)": cw(u56["dlo"].values > lt, u56["Cmp"].values)}


def breakdown(d, w=None):
    w = np.ones(len(d)) if w is None else np.asarray(w, float)
    for t in np.sort(np.unique(np.concatenate([[1.0], np.exp(np.maximum(d, 0))]))):
        if (w * (d > np.log(t))).sum() / w.sum() < 0.5:
            return float(t)
    return np.nan


def anchor_union(anchors):
    dlo = np.full(len(mods), np.inf)
    for conv, c0, lo, hi, eL, eU in anchors:
        lnm, c = (lnM_T, c_T) if conv == "total" else (lnM_ds, c_ds)
        x = c - c0
        dlo = np.minimum(dlo, lnm - (hi + np.where(x > 0, eU * x, eL * x)))
    return dlo


def ols(c, y):
    c0 = c.max()
    X = np.c_[np.ones(len(c)), c - c0]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    df = len(y) - 2
    V = np.linalg.inv(X.T @ X) * (r @ r / df)
    return dict(c0=c0, a=b[0], e=b[1], se_a=np.sqrt(V[0, 0]), se_e=np.sqrt(V[1, 1]), tq=stats.t.ppf(0.975, df))


def mc_coverage(corpora, reps=400, B=999, seed=9):
    s0 = Study("mc", corpora, (c_T.min() - 0.01, c_T.max() + 0.01), B=199, seed=1, grid_n=401)
    X, V, beta, sd, df, n = s0.X, s0.V, s0.b, s0.s, s0.df, s0.n
    L, nrm = s0.L, s0.nrm
    h = np.einsum("ij,jk,ik->i", X, V, X)
    Rw = V @ X.T
    rng = np.random.default_rng(seed)
    cb = cm = 0
    for _ in range(reps):
        y = X @ beta + rng.normal(0, sd, n)
        b = Rw @ y
        e = y - X @ b
        s = np.sqrt(e @ e / df)
        W = WEBB[rng.integers(0, 6, (B, n))]
        Ys = (X @ b)[None] + W * (e / np.sqrt(1 - h))[None]
        bs = (Rw @ Ys.T).T
        ss = np.sqrt(((Ys - bs @ X.T) ** 2).sum(1) / df)
        q = np.quantile((np.abs((bs - b) @ L.T) / (ss[:, None] * nrm)).max(1), 0.95)
        stat = (np.abs((b - beta) @ L.T) / (s * nrm)).max()
        cb += stat <= q
        cm += stat <= max(q, s0.q_norm)
    return cb / reps, cm / reps


def main():
    out = []

    def rec(key, indep, module, tol, note=""):
        diff = abs(float(indep) - float(module)) if np.isfinite(float(module)) else np.nan
        out.append(dict(quantity=key, independent=float(indep), module=float(module), abs_diff=diff,
                        tolerance=tol, agrees=bool(diff <= tol) if np.isfinite(diff) else False, note=note))

    SH = pd.read_csv(os.path.join(TAB, "rb5_sign_shares.csv"))
    PT = pd.read_csv(os.path.join(TAB, "rb5_sign_paths.csv"))
    MS = pd.read_csv(os.path.join(TAB, "rb5_sign_mstar_sets.csv"))
    BK = pd.read_csv(os.path.join(TAB, "rb5_sign_breakdown.csv"))
    MG = pd.read_csv(os.path.join(TAB, "rb5_sign_magnitude.csv"))
    FZ = pd.read_csv(os.path.join(TAB, "rb5_sign_fixed_zero_point.csv"))
    if "level" not in MG.columns:
        MG["level"] = "models"

    def msh(set_, level, tau, col="share_gt1", year="all"):
        r = SH[(SH["set"] == set_) & (SH["level"] == level) & (SH["tau"] == tau) & (SH["year"] == year)]
        return float(r[col].iloc[0])

    dom = (c_T.min() - 0.01, c_T.max() + 0.01)
    chin = Study("Chinchilla", [("Chinchilla",) + design("Chinchilla")], dom, seed=101)
    ll3 = Study("Llama 3", [("Llama 3",) + design("Llama 3")], dom, seed=202)
    mar = Study("Marin", [(k,) + design("Marin, " + k) for k in ("DCLM", "Nemotron-CC", "Comma")], dom, seed=303)
    for s, path in ((chin, "Chinchilla"), (ll3, "Llama 3"), (mar, "Marin")):
        r = PT[PT["path"] == path]
        rec(f"{path}: slope", s.b[-1], r["e"].iloc[0], 1e-9)
        rec(f"{path}: pooled residual s.d.", s.s, r["resid_sd_pooled"].iloc[0], 1e-9)
        for g, lev in enumerate(np.exp(s.b[:-1])):
            rec(f"{path}: level {g + 1} at c_J", lev, r["Mstar_J"].iloc[g], 1e-6)
        rec(f"{path}: normal-theory q (grid 2,001 vs exact sup)", s.q_norm, r["q_normal"].iloc[0], 0.03, "Monte Carlo")
        rec(f"{path}: bootstrap q (own seed)", s.q_boot, r["q_boot"].iloc[0], 0.08, "Monte Carlo; seeds differ")
    dsx = ds_path(c_ds)
    dlo, dhi = union_bounds([s.band(c_T) for s in (chin, ll3, mar)], [(lnM_ds, dsx, dsx)])
    for tau in (1.0, 1.3, 1.84, 3.4, 4.4, 8.0, 10.0):
        sh = shares(dlo, tau)
        for k, v in sh.items():
            lev, col = (k.replace("_cw", ""), "share_gt1_cw") if "_cw" in k else (k, "share_gt1")
            rec(f"S1 share {k}, tau = {tau:g}", v, msh("S1", lev, tau, col), 1e-12)
    for y in sorted(mods["year"].unique()):
        m = (mods["year"] == y).values
        rec(f"S1 compute-weighted share, {y}", float(((dlo[m] > 0) * Cmp[m]).sum() / Cmp[m].sum()),
            msh("S1", "models", 1.0, "share_gt1_cw", str(y)), 1e-12)
    rec("S1: models identified as w < 1", int((dhi < 0).sum()), 0, 0)
    UN = pd.read_csv(os.path.join(TAB, "rb5_sign_unidentified.csv"))
    MO = pd.read_csv(os.path.join(TAB, "rb5_sign_models.csv"))
    rec("S1: unidentified models", int((dlo <= 0).sum()), len(UN), 0)
    rec("S1: largest M among unidentified", float(mods["M"][dlo <= 0].max()), float(UN["M"].max()), 1e-6)
    lo_eff = mods["M"].values / np.exp(dhi)
    rec("S1: lower end of the set at the models' compute, min", lo_eff.min(), float(MO["Mstar_lo_eff"].min()), 0.02,
        "Monte Carlo q")
    rec("S1: lower end of the set at the models' compute, max", lo_eff.max(), float(MO["Mstar_lo_eff"].max()), 0.04,
        "Monte Carlo q")
    s1 = MS[(MS["set"] == "S1") & (MS["conv"] == "total") & (~MS["paths"].str.contains("alone"))].set_index("C")
    for C in (1e24, 1e25):
        lo = np.exp(min(s.band(np.log(C))[0][0] for s in (chin, ll3, mar)))
        hi = np.exp(max(s.band(np.log(C))[1][0] for s in (chin, ll3, mar)))
        rec(f"M*({C:.0e}) lower end", lo, s1.loc[C, "Mstar_lo"], 0.02 * lo, "relative 2 percent (Monte Carlo q)")
        rec(f"M*({C:.0e}) upper end", hi, s1.loc[C, "Mstar_hi"], 0.02 * hi, "relative 2 percent (Monte Carlo q)")
    sm = (mods["N"] < 15e9).values
    bk = BK[BK["set"] == "S1"].set_index("series")
    rec("breakdown tau, all models", breakdown(dlo), bk.loc["all models", "tau_breakdown"], 0.02)
    rec("breakdown tau, below 15B", breakdown(dlo[sm]), bk.loc["models below 15B", "tau_breakdown"], 0.02)
    rec("breakdown tau, 49 decisions", breakdown(unit_dlo(dlo, U49)["dlo"].values), bk.loc["decisions (49)", "tau_breakdown"], 0.02)
    rec("breakdown tau, compute", breakdown(dlo, Cmp), bk.loc["training compute (models)", "tau_breakdown"], 0)
    i = int(np.where(mods["model"] == "SmolLM2-1.7B")[0][0])
    PN = pd.read_csv(os.path.join(TAB, "rb5_sign_paper_numbers.csv")).set_index("key")
    rec("P13: SmolLM2 1.7B factor", np.exp(dlo[i]), float(PN.loc["P13_SmolLM2_factor", "value_raw"]), 0.3, "Monte Carlo q")
    # ---------------------------------------------------------------------------- S2 variants
    A3 = pd.read_csv(io.StringIO(subprocess.run([GIT, "-C", ROOT, "show", "69c9ae8:output/tables/rb2_decisions_anchors.csv"],
                                                capture_output=True, text=True).stdout))

    def v3(eL, eU, ds_total=False):
        return [(("total" if ds_total else a["conv"]), np.log(a["C0"]), a["lnMs_lo"], a["lnMs_hi"], eL, eU)
                for _, a in A3.iterrows()]

    def both(key, d, set_, note=""):
        s = shares(d)
        rec(f"{key}: models", s["models"], msh(set_, "models", 1.0), 1e-12, note)
        rec(f"{key}: compute", s["models_cw"], msh(set_, "models", 1.0, "share_gt1_cw"), 1e-12, note)
        rec(f"{key}: 49 decisions", s["decisions (49)"], msh(set_, "decisions (49)", 1.0), 1e-12, note)

    pi1 = (-0.156, float(A3.loc[A3["anchor"] == "marin_dclm", "e_path"].iloc[0]))
    both("S2(a) PI-1 as published", anchor_union(v3(*pi1)), "S2(a)")
    T = {k: ols(*design(k)) for k in ("Chinchilla", "Llama 3", "Marin, DCLM", "Marin, Nemotron-CC", "Marin, Comma")}
    comma = T["Marin, Comma"]
    an = v3(-0.156, comma["e"]) + [("total", np.log(3e20), np.log(0.1), np.log(3571), -0.156, comma["e"])]
    s = shares(anchor_union(an))
    rec("S2(a)+Comma with version 3's printed interval [0.1, 3,571]: models", s["models"], msh("S2(a)+Comma", "models", 1.0), 1e-12)
    ds3 = ds_path(np.log(3e20))

    def tint(eL, eU):
        return [("total", t["c0"], t["a"] - t["tq"] * t["se_a"], t["a"] + t["tq"] * t["se_a"], eL, eU) for t in T.values()] + \
            [("ds", np.log(3e20), ds3, ds3, eL, eU)]
    e_pts = [t["e"] for t in T.values()] + E_PUB
    both("S2(b) t-intervals, Comma's own slope", anchor_union(tint(min(e_pts), max(e_pts))), "S2(b)")
    e_ci = [t["e"] + sg * t["tq"] * t["se_e"] for t in T.values() for sg in (-1, 1)] + E_PUB
    both("S2(b') slopes at their t-intervals", anchor_union(tint(min(e_ci), max(e_ci))), "S2(b')")
    ec = (-0.1745112143187324, 0.3527929876297756)
    both("S2(c) PI-1 levels, slope-CI hull", anchor_union(v3(*ec)), "S2(c)")
    s = shares(anchor_union(v3(*ec, ds_total=True)))
    rec("S2(c) with DeepSeek at the models' total M and C (R4's convention): models", s["models"], 0.727, 0.0005, "R4: 72.7")
    rec("S2(c) with DeepSeek at the models' total M and C (R4's convention): compute", s["models_cw"], 0.195, 0.0005, "R4: 19.5")
    an = v3(*pi1) + [("total", comma["c0"], comma["a"] - comma["tq"] * comma["se_a"], comma["a"] + comma["tq"] * comma["se_a"], *pi1)]
    s = shares(anchor_union(an))
    rec("R2: PI-1 plus Comma on its t-interval, PI-1 slopes: models", s["models"], 0.84, 0.005, "R2: 0.84")
    rec("R2: PI-1 plus Comma on its t-interval, PI-1 slopes: compute", s["models_cw"], 0.48, 0.005, "R2: 0.48")
    an = [("total", st.cJ, st.b[g] - st.t_pt * st.s * np.sqrt(st.V[g, g]), st.b[g] + st.t_pt * st.s * np.sqrt(st.V[g, g]), -1.0, 1.0)
          for st in (chin, ll3, mar) for g in range(st.G)] + [("ds", np.log(3e20), ds3, ds3, -1.0, 1.0)]
    both("S2(d) normal inputs, S1 levels", anchor_union(an), "S2(d)", "pointwise t-intervals (module: bootstrap-t floored at t)")
    both("S2(d) normal inputs, version 3 levels", anchor_union(v3(-1.0, 1.0)), "S2(d) PI-1 anchors")
    # S2(f)-total: Marin converted with its measured ratio N_F/N_cfg; Llama 3 shifted by the conversion bound
    raw = pd.read_csv(os.path.join(ROOT, "data/raw/isoflop_experiments/isoflop_experiments.csv"))
    ex = {"DCLM": "marin_202603__dclm__llama_2", "Nemotron-CC": "marin_202603__nemotron__llama_2",
          "Comma": "marin_202603__comma__llama_2"}
    m = raw[raw["experiment"].isin(ex.values())].copy()
    m["r"] = m["budget"] / (6 * m["tokens"] * m["params"])
    rr = m.groupby("params")["r"].median()
    xs = np.log(rr.index.values * rr.values)
    o = np.argsort(xs)
    xs, lr = xs[o], np.log(rr.values[o])
    corp = []
    for k in ("DCLM", "Nemotron-CC", "Comma"):
        g = bud[bud["design"] == "Marin, " + k].sort_values("budget_C")
        lnr = np.interp(np.log(g["Nstar"].values), xs, lr)
        corp.append((k, np.log(g["budget_C"].values) - lnr, np.log(g["Mstar"].values) + lnr))
    marT = Study("Marin (total)", corp, dom, seed=77)
    PV = pd.read_csv(os.path.join(TAB, "rb5_sign_paths_variants.csv"))
    rec("Marin in total parameters: slope", marT.b[-1], float(PV.loc[PV["path"] == "Marin (total parameters)", "e"].iloc[0]),
        1e-6)
    d_, L_, V_, kv, ff, nctx = 3072, 28, 128256, 1024, 8192, 8192
    nne = L_ * (2 * d_ * d_ + 2 * d_ * kv + 3 * d_ * ff)
    rat = [np.log(nf / (nne + em * V_ * d_)) for em in (1, 2)
           for nf in (nne + em * V_ * d_, nne, nne + V_ * d_ + L_ * nctx * d_)]
    CO = pd.read_csv(os.path.join(TAB, "rb5_sign_conventions.csv"))
    CO = CO[CO["item"].str.startswith("Llama 3")]
    rec("Llama 3 conversion bound, lower", min(rat), float(CO["ln_ratio"].min()), 1e-9)
    rec("Llama 3 conversion bound, upper", max(rat), float(CO["ln_ratio"].max()), 1e-9)
    lo2, hi2 = ll3.band(c_T)
    d, _ = union_bounds([chin.band(c_T), (lo2 + min(rat), hi2 + max(rat)), marT.band(c_T)], [(lnM_ds, dsx, dsx)])
    both("S2(f) all anchors in total parameters", d, "S2(f)-total")
    # S2(f)-NF: Chinchilla's minima in N_F (rb4, variant NF_T4); models in Hoffmann et al.'s count at 4,096 tokens
    rb4 = pd.read_csv(os.path.join(TAB, "rb4_chinflop_budgets.csv"))
    g4 = rb4[rb4["variant"] == "NF_T4"].sort_values("budget_C")
    chF = Study("Chinchilla (N_F)", [("Chinchilla", np.log(g4["budget_C"].values), np.log(g4["Mstar"].values))], dom,
                seed=404)
    N_F = mods["N_nonemb"].values + 2 * mods["N_head"].values + 2 * mods["n_layer"].values * mods["d_model"].values * 4096
    lnM_F, c_F = np.log(mods["D"].values / N_F), np.log(6 * N_F * mods["D"].values)
    bF = []
    for st in (chF, ll3, mar):
        lo, hi = st.band(c_F)
        bF.append((lnM_F, lo, hi))
    both("S2(f) anchors and models in N_F", union_bounds([], bF + [(lnM_ds, dsx, dsx)])[0], "S2(f)-NF")
    # ---------------------------------------------------------------------------- S4
    top = pd.read_csv(os.path.join(TAB, "rb1_sigmaC_top_budget_sigma.csv"))
    st_ = float(top.iloc[0]["mean_fixed"])
    pis = pd.read_csv(os.path.join(TAB, "rb1_sigmaC_pi_sigmaC.csv"))
    col = [c for c in pis.columns if c.startswith("sigma_lower|drift")][0]
    slin = np.interp(np.log10(Cmp), pis["log10C"].values, pis[col].values)
    mg = MG[(MG["set"] == "S1") & (MG["level"] == "models")]
    for lab, kL, kU, key in [("[0.40, 0.52]", 0.40, 0.52, "k in [0.40, 0.52]"),
                             ("[0.40, 0.68]", 0.40, 1 / st_ - 1, "k in [0.40, 0.684]"),
                             ("[0.40, k(sigma_lin)]", 0.40, 1 / slin - 1, "k in [0.40, k(sigma_lin(C_i))]:")]:
        for tau in (1.0, 1.84, 3.4):
            a, b = dlo - np.log(tau), dhi + np.log(tau)
            kLb, kUb = np.broadcast_to(kL, a.shape), np.broadcast_to(kU, a.shape)
            corners = np.stack([kLb * a, kUb * a, kLb * b, kUb * b])
            r = mg[mg["curvature"].str.startswith(key) & (mg["tau"] == tau)].iloc[0]
            rec(f"S4 median lower bound on s, k {lab}, tau = {tau:g}", np.median(1 - np.exp(-corners.min(0))),
                r["median_s_lower"], 0.005, "Monte Carlo q")
            rec(f"S4 median upper bound on s, k {lab}, tau = {tau:g}", np.median(1 - np.exp(-corners.max(0))),
                r["median_s_upper"], 0.005, "Monte Carlo q")
    u = np.log(mods["M"].values) - np.log(mods["Mstar_ref"].values)

    def unit_med(w):
        x = pd.DataFrame(dict(uid=mods["uid"], w=w, C=Cmp)).merge(U49[["uid", "unit"]], on="uid")
        W = x.groupby("unit").apply(lambda g: 1 / np.sum((g["C"] / g["C"].sum()) / g["w"]), include_groups=False)
        return float(np.median(1 - 1 / W.values))
    fz = FZ.set_index("scenario")["median_s_decisions_49"]
    rec("fixed zero point: reference median over 49 decisions", unit_med(mods["w_ref"].values), fz.iloc[0], 1e-9)
    for sg, row in ((0.60, "constant sigma* = 0.6"), (0.62, "constant sigma* = 0.62")):
        rec(f"fixed zero point: constant sigma* = {sg}", unit_med(np.exp((1 / sg - 1) * u)), fz.loc[row], 1e-9)
    rec("fixed zero point: sigma_lin(C_i)", unit_med(np.exp((1 / slin - 1) * u)), fz.iloc[-1], 1e-9)
    rec("WP4b's unit share median (s_unit) against the reference", float(np.median(U49.groupby("unit")["s_unit"].first())),
        fz.iloc[0], 1e-9)
    # ---------------------------------------------------------------------------- coverage
    CV = pd.read_csv(os.path.join(TAB, "rb5_sign_review_coverage.csv"))
    for nm, cor in (("Chinchilla", [("Chinchilla",) + design("Chinchilla")]), ("Llama 3", [("Llama 3",) + design("Llama 3")]),
                    ("Marin", [(k,) + design("Marin, " + k) for k in ("DCLM", "Nemotron-CC", "Comma")])):
        cb, cm = mc_coverage(cor)
        r = CV[(CV["path"] == nm) & (CV["errors"] == "normal")].iloc[0]
        rec(f"coverage, bootstrap q alone, {nm}", cb, r["coverage_boot_q"], 0.035, "Monte Carlo, 400 reps (s.e. about 0.012)")
        rec(f"coverage, max rule, {nm}", cm, r["coverage_max_rule"], 0.03, "Monte Carlo, 400 reps")
    R = pd.DataFrame(out)
    R.to_csv(os.path.join(TAB, "rb5_sign_review_independent.csv"), index=False)
    print(R.to_string())
    print(f"{int(R['agrees'].sum())} of {len(R)} quantities agree")
    return R


if __name__ == "__main__":
    sys.exit(0 if main()["agrees"].all() else 1)
