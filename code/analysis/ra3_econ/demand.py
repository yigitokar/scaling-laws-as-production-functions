"""demand.py -- compute-optimal data demand D*(C) ~ C^(1-a) under alternative technologies, its growth, and the date
at which frontier runs would reach the stock of high-quality public text (Villalobos et al. 2024).

Each path technology is reduced to ln D*(C) (tokens, the technology's own tokenizer) and its local elasticity
d ln D*/d ln C = 1 - a. Growth of data demand at compute growth g_C is g_C^(1-a) per year. Over-training at a data
multiple m_D = D/D*(C) (= sqrt(M/M*), since C is fixed) shifts the level, not the growth rate.

Exhaustion year: the first t at which m_D D*(C_F(t)) reaches the stock S(t) = S_2024 (1+g_S)^(t-2024), with C_F(t) the
fitted frontier compute trend (growth.py). Two stocks: quality-adjusted unique stock (100T [22T, 490T]; first-epoch
exhaustion: frontier runs must repeat data from then on) and the repetition-adjusted 'effective' stock (320T
[65T, 1700T]), which is Villalobos et al.'s object (median full-utilisation 2028).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import ra3common  # noqa: F401  (sets sys.path for the m3_wedge imports)
import technologies as M3T                      # module m3_wedge (read-only import)
from common import lnG_of                        # m3_wedge helper
from ra3common import (DEEPSEEK, KAPLAN_A, M1_REG, M2_PROC, M2_REG, M3_MODELS, N_MC, ROOT, SEED, T0, VILLALOBOS,
                       lognormal_from_ci)

LN6 = np.log(6.0)


class PathTech:
    """ln D*(C) = c0 + c1 ln C (power law) or a piecewise log-linear interpolation (Farseer Eq. 3)."""

    def __init__(self, key, label, group, a, c0=None, lnC_pts=None, lnD_pts=None, draws=None, n_conv="", source=""):
        self.key, self.label, self.group, self.a = key, label, group, a
        self.c0, self.c1 = c0, 1.0 - a
        self.lnC_pts, self.lnD_pts = lnC_pts, lnD_pts
        self.draws = draws          # (B, 2): a, c0
        self.n_conv, self.source = n_conv, source

    @property
    def has_level(self):
        return self.c0 is not None or self.lnC_pts is not None

    def lnD(self, lnC):
        lnC = np.asarray(lnC, float)
        if self.lnC_pts is None:
            return self.c0 + self.c1 * lnC
        x, y = self.lnC_pts, self.lnD_pts
        s_lo, s_hi = (y[1] - y[0]) / (x[1] - x[0]), (y[-1] - y[-2]) / (x[-1] - x[-2])
        return np.where(lnC < x[0], y[0] + s_lo * (lnC - x[0]),
                        np.where(lnC > x[-1], y[-1] + s_hi * (lnC - x[-1]), np.interp(lnC, x, y)))

    def elasticity(self, lnC):
        if self.lnC_pts is None:
            return self.c1
        h = 1e-3
        return float((self.lnD(lnC + h) - self.lnD(lnC - h)) / (2 * h))


def _from_alpha_beta_lnG(key, label, group, al, be, lnG, draws=None, n_conv="", source=""):
    a = be / (al + be)
    c0 = -(1 - a) * LN6 - lnG                      # D* = (C/6)^(1-a) / G
    dr = None
    if draws is not None:                           # draws: (B, 3) alpha, beta, lnG
        ad = draws[:, 1] / (draws[:, 0] + draws[:, 1])
        dr = np.c_[ad, -(1 - ad) * LN6 - draws[:, 2]]
    return PathTech(key, label, group, a, c0=c0, draws=dr, n_conv=n_conv, source=source)


def load_path_techs():
    T = {}
    m3 = M3T.load_all()
    spec = [("chin", "Chinchilla refit, kappa = 1 (reference)", "Chinchilla"),
            ("besi", "Besiroglu et al. (published)", "Chinchilla"),
            ("hoff", "Hoffmann et al. A3 (published)", "Chinchilla"),
            ("chin_nls", "Chinchilla, NLS in levels", "Chinchilla"),
            ("chin245", "Chinchilla, all 245 runs", "Chinchilla"),
            ("farseer", "Farseer, kappa = 1 (non-emb. N)", "Design sweeps"),
            ("farseer_emb", "Farseer, kappa = 1 (N incl. emb.)", "Design sweeps"),
            ("farseer_q", "Farseer, kappa free", "Design sweeps"),
            ("gadre_rw", "Gadre et al. RefinedWeb", "Design sweeps"),
            ("gadre_c4", "Gadre et al. C4", "Design sweeps"),
            ("gadre_rp", "Gadre et al. RedPajama", "Design sweeps"),
            ("olmo", "OLMo ladder (AI2)", "Lab laws"),
            ("meta_a2", "Llama 3 (Meta's IsoFLOP law, A2)", "Lab laws"),
            ("meta_a3", "Llama 3 IsoFLOPs, primal (A3)", "Lab laws")]
    for k, lab, grp in spec:
        t = m3[k]
        T[k] = _from_alpha_beta_lnG(k, lab, grp, t.alpha, t.beta, t.lnG, draws=t.draws, n_conv=t.n_conv, source=t.source)
    # Chinchilla kappa free (m2 q family; the path uses the inner exponents)
    r2 = pd.read_csv(M2_REG)
    q = r2[(r2.dataset == "chinchilla") & (r2.subset == "all") & (r2.estimator == "huber_q")].iloc[0]
    qd = np.load(os.path.join(M2_PROC, "boot", "chinchilla__all__huber_q.npy"))   # E, A, B, alpha, beta, q
    ok = (qd[:, 3] > 0) & (qd[:, 4] > 0) & (qd[:, 1] > 0) & (qd[:, 2] > 0)
    qdr = np.c_[qd[ok, 3], qd[ok, 4], lnG_of(qd[ok, 1], qd[ok, 2], qd[ok, 3], qd[ok, 4])]
    T["chin_q"] = _from_alpha_beta_lnG("chin_q", "Chinchilla refit, kappa free", "Chinchilla", q.alpha, q.beta,
                                       lnG_of(q.A, q.B, q.alpha, q.beta), draws=qdr, n_conv="total",
                                       source="m2 chinchilla/all/huber_q; pairs B=200")
    # Chinchilla Approach 2 (m1: OLS of IsoFLOP argmins)
    a2 = pd.read_csv(os.path.join(ROOT, "output", "tables", "m1_chinchilla_a2_minima.csv"))
    a2 = a2[a2.dataset == "chinchilla_n240"].iloc[0]
    T["chin_a2"] = PathTech("chin_a2", "Chinchilla IsoFLOP argmins (A2)", "Chinchilla", a2.a_A2,
                            c0=-(1 - a2.a_A2) * LN6 - a2.lnG_A2, n_conv="total", source="m1 A2 on n = 240")
    # Marin 2026-03 (m1 lab-own A2/A1) and Muennighoff single epoch (m2)
    r1 = pd.read_csv(M1_REG).set_index("row_id")
    for corp in ["comma", "dclm", "nemotron"]:
        rr = r1.loc[f"marin_202603__{corp}__llama_2_A2A1_kappa1"]
        T[f"marin_{corp}"] = _from_alpha_beta_lnG(f"marin_{corp}", f"Marin ({corp.upper() if corp != 'comma' else 'Comma'}), A2",
                                                  "Lab laws", rr.alpha, rr.beta, float(np.log(rr.G)), n_conv="total",
                                                  source=f"m1 {rr.name}")
    dm = r2[(r2.dataset == "datablations") & (r2.subset == "single_epoch") & (r2.estimator == "huber")].iloc[0]
    T["muenn"] = _from_alpha_beta_lnG("muenn", "Muennighoff et al., single epoch", "Design sweeps", dm.alpha, dm.beta,
                                      lnG_of(dm.A, dm.B, dm.alpha, dm.beta), n_conv="authors' PARAMS_MAP",
                                      source="m2 datablations/single_epoch/huber")
    # Farseer Eq. 3 (non-homothetic; m2 fitted law; D*(C) tabulated at 1e19..1e23, beyond 3.5e21 extrapolated)
    fe = pd.read_csv(os.path.join(ROOT, "output", "tables", "m2_farseer_eq3_Mstar.csv"))
    lnC, lnD = np.log(fe.C.values), np.log(fe.D_star.values)
    T["farseer_eq3"] = PathTech("farseer_eq3", "Farseer Eq. 3 (own form; local at 1e22-1e23)", "Design sweeps",
                                1 - (lnD[-1] - lnD[-2]) / (lnC[-1] - lnC[-2]), lnC_pts=lnC, lnD_pts=lnD,
                                n_conv="non-emb.", source="m2_farseer_eq3_Mstar.csv")
    # literature lab laws
    T["deepseek"] = PathTech("deepseek", "DeepSeek LLM (published law)", "Lab laws", DEEPSEEK["a"],
                             c0=np.log(DEEPSEEK["D0"]), n_conv="non-emb. FLOPs/token", source="bi2024deepseek eq. 4")
    T["kaplan"] = PathTech("kaplan", "Kaplan et al. (2020), historical", "Historical", KAPLAN_A, c0=None,
                           source="kaplan2020scaling (N ~ C^0.73; growth only)")
    # module ra1 (reviewed): model-free Approach-2 path slopes, OLS of ln N* on ln C over ra1's valid (bracketed) budgets
    # (ra1_modelfree_isoflop_budgets.csv: per-budget argmins of the local-polynomial IsoFLOP profiles). Growth only (no
    # level): these designs stop at 2.6e19-3e21 FLOP.
    for k, lab in ra1_path_slopes().items():
        T[k] = PathTech(k, lab["label"], "ra1 model-free A2", lab["a"], c0=None,
                        source=f"ra1_modelfree_isoflop_budgets.csv; {lab['k']} budgets, C up to {lab['Cmax']:.1e}")
    return T


def ra1_path_slopes():
    f = os.path.join(ROOT, "output", "tables", "ra1_modelfree_isoflop_budgets.csv")
    if not os.path.exists(f):
        return {}
    b = pd.read_csv(f)
    out = {}
    for des, g in b.groupby("design", sort=False):
        x, y = np.log(g["budget_C"].values), np.log(g["Nstar"].values)
        a = float(np.polyfit(x, y, 1)[0])
        key = "ra1_" + des.lower().replace(", ", "_").replace(" ", "_").replace("-", "")
        out[key] = dict(a=a, k=len(g), Cmax=float(g["budget_C"].max()), label=f"{des} (ra1 model-free A2 path)")
    return out


# ----------------------------------------------------------------------------- over-training at observed frontier M
def frontier_runs(Cmin=1e25, Cmax=np.inf, years=(2024, 2026), exclude_moe=True):
    """Disclosed runs (m3's verified inputs) above Cmin FLOP. MoE runs are excluded by default (review): their N is
    active parameters, so D/D*(C) under a dense technology is not comparable (R1 minor 24; ra2's clean-sample rule)."""
    d = pd.read_csv(M3_MODELS)
    d = d[d["core"].fillna(False).astype(bool) & (d["Cmp"] >= Cmin) & (d["Cmp"] < Cmax) & d["year"].between(*years)]
    if exclude_moe:
        d = d[~d["moe"].fillna(False).astype(bool)]
    return d[["model", "lab", "year", "N", "D", "M", "Cmp", "open_weights", "moe"]].sort_values("Cmp", ascending=False)


def data_multiple(tech, runs):
    """Median over runs of D_i / D*_tech(C_i) = sqrt(M_i / M*_tech(C_i))."""
    if not tech.has_level:
        return np.nan
    return float(np.median(runs["D"].values / np.exp(tech.lnD(np.log(runs["Cmp"].values)))))


# ----------------------------------------------------------------------------- growth and exhaustion
def exhaustion_year(tech, b0, b1, S0, gS, mD=1.0):
    """First t with ln(mD) + lnD*(C_F(t)) = ln S0 + ln(1+gS)(t - 2024); C_F(t) = 10^(b0 + b1 (t - T0))."""
    if not tech.has_level:
        return np.nan
    f = lambda t: (np.log(mD) + tech.lnD(np.log(10) * (b0 + b1 * (t - T0)))  # noqa: E731
                   - np.log(S0) - np.log1p(gS) * (t - VILLALOBOS["base_year"]))
    lo, hi = 2000.0, 2100.0
    if f(lo) > 0:
        return lo
    if f(hi) < 0:
        return np.inf
    return brentq(f, lo, hi, xtol=1e-6)


def exhaustion_closed(a, c0, b0, b1, S0, gS, mD=1.0):
    """Vectorised closed form for power laws (MC)."""
    ln10 = np.log(10)
    num = np.log(S0) - np.log1p(gS) * (T0 - VILLALOBOS["base_year"]) - c0 - (1 - a) * ln10 * b0 - np.log(mD)
    den = (1 - a) * ln10 * b1 - np.log1p(gS)
    return T0 + num / den


def demand_table(T, trend, gC_list, runs):
    """Per technology: a, growth of D at several g_C, 5-yr growth, D*(frontier now), exhaustion years."""
    b0, b1 = trend["b0"], trend["b1"]
    lnC_now = np.log(10) * (b0 + b1 * (2026 + 266 / 365.25 - T0))
    rows = []
    for k, t in T.items():
        el = t.elasticity(lnC_now)
        r = dict(key=k, label=t.label, group=t.group, a=1 - el, a_global=t.a, d_elasticity=el,
                 n_conv=t.n_conv, source=t.source)
        for nm, g in gC_list:
            r[f"gD_{nm}"] = g ** el
            r[f"gD5_{nm}"] = g ** (5 * el)
        if t.draws is not None:
            ad = t.draws[:, 0]
            gC = gC_list[0][1]
            r["a_lo"], r["a_hi"] = np.percentile(ad, [2.5, 97.5])
            r["gD_hat_lo"], r["gD_hat_hi"] = np.percentile(gC ** (1 - ad), [2.5, 97.5])
        if t.has_level:
            r["Dstar_now_T"] = float(np.exp(t.lnD(lnC_now))) / 1e12
            r["Mstar_now"] = float(np.exp(t.lnD(lnC_now))) ** 2 * 6 / np.exp(lnC_now)
            r["data_multiple_obs"] = data_multiple(t, runs)
            for sname, S0 in [("unique", VILLALOBOS["U_q"]), ("effective", VILLALOBOS["U_eff"])]:
                r[f"year_{sname}_opt"] = exhaustion_year(t, b0, b1, S0, VILLALOBOS["g_mid"])
                r[f"year_{sname}_obsOT"] = exhaustion_year(t, b0, b1, S0, VILLALOBOS["g_mid"], r["data_multiple_obs"])
                r[f"year_{sname}_OT5"] = exhaustion_year(t, b0, b1, S0, VILLALOBOS["g_mid"], np.sqrt(5.0))
        rows.append(r)
    return pd.DataFrame(rows)


def exhaustion_mc(tech, trend, bdraws, runs, stock="unique", n=N_MC, seed=SEED, components=True):
    """Monte Carlo interval for the exhaustion year under one power-law technology, combining (i) technology
    bootstrap draws (a, c0), (ii) wild-cluster-bootstrap draws of the compute trend (b0, b1), (iii) a lognormal stock
    matching Villalobos et al.'s 95% interval and (iv) stock growth g_S ~ U[0, 10%]. Also each source alone."""
    rng = np.random.default_rng(seed)
    V = VILLALOBOS
    S_med, S_lo, S_hi = (V["U_q"], V["U_q_lo"], V["U_q_hi"]) if stock == "unique" else (V["U_eff"], V["U_eff_lo"], V["U_eff_hi"])
    td = tech.draws[rng.integers(0, len(tech.draws), n)]
    bd = bdraws[rng.integers(0, len(bdraws), n)]
    S = lognormal_from_ci(S_med, S_lo, S_hi, rng, n)
    gS = rng.uniform(V["g_lo"], V["g_hi"], n)
    base = dict(a=np.full(n, tech.a), c0=np.full(n, tech.c0), b0=np.full(n, trend["b0"]), b1=np.full(n, trend["b1"]),
                S0=np.full(n, S_med), gS=np.full(n, V["g_mid"]))
    out = {}
    combos = {"all": ["tech", "trend", "stock", "gS"]}
    if components:
        combos.update({"tech": ["tech"], "trend": ["trend"], "stock": ["stock"], "gS": ["gS"]})
    for name, srcs in combos.items():
        p = {k: v.copy() for k, v in base.items()}
        if "tech" in srcs:
            p["a"], p["c0"] = td[:, 0], td[:, 1]
        if "trend" in srcs:
            p["b0"], p["b1"] = bd[:, 0], bd[:, 1]
        if "stock" in srcs:
            p["S0"] = S
        if "gS" in srcs:
            p["gS"] = gS
        # observed-M anchoring: the multiple m_D = median_i D_i / D*(C_i) is recomputed for every technology draw, so
        # that the anchored level stays at observed data use (review fix; the builder held m_D at its point value,
        # which re-introduced the technology's level uncertainty into the anchored interval)
        lnC_i, lnD_i = np.log(runs["Cmp"].values), np.log(runs["D"].values)
        mD_draw = np.exp(np.median(lnD_i[None, :] - (p["c0"][:, None] + (1 - p["a"])[:, None] * lnC_i[None, :]), axis=1))
        for ot, mD in [("opt", 1.0), ("obsOT", mD_draw)]:
            y = exhaustion_closed(p["a"], p["c0"], p["b0"], p["b1"], p["S0"], p["gS"], mD)
            out[(name, ot)] = np.percentile(y, [5, 50, 95])
    return out
