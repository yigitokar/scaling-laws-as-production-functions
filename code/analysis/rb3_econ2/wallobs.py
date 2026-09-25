"""wallobs.py -- the data wall at OBSERVED allocations (R3 N6(2)).

Section V.B of the paper indexes scarcity at compute-optimal use, r = D*(C)/U. An over-trained model with wedge w
processes D = D*(C) w^e tokens (e = sigma*/[2(1 - sigma*)]), so the scarcity it faces is r_obs = D/U = r w^e: the wall
binds at lower compute for over-trained models. Here the developer's objective is the lifetime cost of Proposition 1
with a value of compactness proportional to size (serving 2N FLOPs per token, or any cost linear in N):

    min_{N, D}  6 N D + Phi N   s.t.   L(N', D'(D; U)) <= l,

with the technology in the kappa family (ra3_econ's wall technologies, imported read-only) and Muennighoff et al.'s
(2023) repetition model, D' = U + U R*_D (1 - exp(-R/R*_D)), R = D/U - 1 (and N' for the full model). Without a cap the
optimum on the loss isoquant through (N_w, D_w) has eps_N/eps_D = 1 + Phi/(6D) = w, so Phi = 6 D_w (w - 1); the problem
is indexed by (C_w, w) and, by the quasi-homotheticity of the kappa family, all relative objects depend on (w, r) only
(checked at two compute levels).

Objects (relative to the uncapped optimum at the same loss):
  life_penalty   extra lifetime cost (training + value of compactness), the cost of the wall at observed allocations;
  train_penalty  extra training compute;  N_ratio = N_c/N_w (the model gets bigger: compactness is given up);
  shadow_life    -dC_life/dU at fixed loss, in units of the constrained model's processing cost per token (6 N_c);
  w_hat_proc     the wedge an econometrician measures from (N_c, D_c) treating processed tokens as fresh (the paper's
                 measurement), and w_hat_eff from effective data D'_c; 1 + m_N at the constrained allocation is
                 1 + Phi/(6 D_c). Proposition 1(iii): with a binding token cap w_hat = (1 + m_N)/(1 + mu), mu >= 0. Under
                 soft repetition mu = 1/eta - 1 with eta = d ln D'/d ln D (checked).
Specifications: 'nore' (hard cap, no useful repetition: Prop. 1(iii)'s case), 'D15' (data decay only, R*_D = 15.4;
the most favourable, ra3's primary), 'D3' (Muennighoff et al.'s data-only fit, R*_D = 2.9), 'DN' (their full model,
excess parameters also decay).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import rb3common as RC
from rb3common import e_of, log

import wall as WL  # noqa: E402  (ra3_econ, read-only)

SPECS = dict(WL.SPECS)


def _D_of_Dp_vec(Dp, U, RD):
    Dp = np.asarray(Dp, float)
    out = np.where(Dp <= U, Dp, np.inf)
    if RD > 0:
        x = (Dp / U - 1.0) / RD
        ok = (Dp > U) & (x < 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(ok, U * (1.0 - RD * np.log1p(-np.where(ok, x, 0.0))), out)
    return out


def _Dp_of_D(D, U, RD):
    if D <= U:
        return D
    if RD <= 0:
        return U
    return U + U * RD * (1 - np.exp(-(D / U - 1) / RD))


def _Np_of_N_vec(N, UN, RN):
    N = np.asarray(N, float)
    if RN is None:
        return N
    return np.where(N <= UN, N, UN + UN * RN * (1 - np.exp(-(N / UN - 1) / RN)))


def life_cost_at_loss(t, L, U, spec, Phi, n_grid=4001, span=30.0):
    """min 6ND + Phi N s.t. loss(N', D'(D; U)) <= L. Returns (C_life, N, D); U = inf means no cap."""
    sp = SPECS[spec]
    Q = (L - t.E) ** (1.0 / t.kappa)
    UN = t.N_opt(t.C_for_D(U)) if (sp["RN"] is not None and np.isfinite(U)) else None
    lnN_min = np.log((t.A / Q) ** (1.0 / t.a1)) + 1e-9

    def obj(lnN):
        N = np.exp(lnN)
        Np = _Np_of_N_vec(N, UN, sp["RN"]) if UN is not None else N
        v = Q - t.A * Np ** (-t.a1)
        with np.errstate(invalid="ignore", divide="ignore"):
            Dp = np.where(v > 0, (np.maximum(v, 1e-300) / t.B) ** (-1.0 / t.b1), np.inf)
        D = Dp if not np.isfinite(U) else _D_of_Dp_vec(Dp, U, sp["RD"])
        with np.errstate(invalid="ignore", over="ignore"):
            c = 6.0 * N * D + Phi * N
        return np.log(np.where(np.isfinite(c) & (c > 0), c, np.inf))

    grid = np.linspace(lnN_min, lnN_min + span, n_grid)
    vals = obj(grid)
    if not np.isfinite(vals).any():
        return np.inf, np.nan, np.nan
    i = int(np.nanargmin(vals))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, n_grid - 1)]
    res = minimize_scalar(lambda x: float(obj(np.array([x]))[0]), bounds=(lo, hi), method="bounded",
                          options=dict(xatol=1e-11))
    lnN = res.x if res.fun <= vals[i] else grid[i]
    lnc = min(res.fun, vals[i])
    if np.isfinite(U) and sp["RD"] <= 0 and sp["RN"] is None:
        # hard cap: the optimum can sit at the corner D = U, where the objective jumps to +inf; evaluate it exactly
        v_c = Q - t.B * U ** (-t.b1)
        if v_c > 0:
            lnN_c = np.log((t.A / v_c) ** (1.0 / t.a1)) * (1 + 1e-12)
            f_c = float(obj(np.array([lnN_c]))[0])
            if f_c < lnc:
                lnN, lnc = lnN_c, f_c
    N = float(np.exp(lnN))
    Clife = float(np.exp(lnc))
    return Clife, N, (Clife - Phi * N) / (6.0 * N)


def wedge_kappa(t, N, D):
    """eps_N/eps_D in the kappa family at (N, D) (kappa cancels)."""
    return t.a1 * t.A * N ** (-t.a1) / (t.b1 * t.B * D ** (-t.b1))


def uncapped(t, C, w):
    """Uncapped lifetime-cost optimum with wedge w at training compute C: (N_w, D_w, Phi, loss)."""
    e = t.sigma_star / (2 * (1 - t.sigma_star))
    Nw = t.N_opt(C) * w ** (-e)
    Dw = t.D_opt(C) * w ** e
    Phi = 6.0 * Dw * (w - 1.0)
    return Nw, Dw, Phi, t.loss(Nw, Dw)


def obs_point(t, C, w, r_obs, spec, h=1e-3):
    """All objects for a model with uncapped wedge w at training compute C facing U = D_w / r_obs."""
    Nw, Dw, Phi, ell = uncapped(t, C, w)
    U = Dw / r_obs
    life0 = 6 * Nw * Dw + Phi * Nw
    Cl, Nc, Dc = life_cost_at_loss(t, ell, U, spec, Phi)
    out = dict(tech=t.key, sigma_star=t.sigma_star, spec=spec, C=C, w=w, r_obs=r_obs, r=t.D_opt(C) / U, U=U,
               N_w=Nw, D_w=Dw, Phi=Phi, loss=ell, life_penalty=Cl / life0 - 1 if np.isfinite(Cl) else np.inf)
    if not np.isfinite(Cl):
        return out
    RD = SPECS[spec]["RD"]
    Dpc = _Dp_of_D(Dc, U, RD)
    mN_c = Phi / (6 * Dc)
    w_proc = wedge_kappa(t, Nc, Dc)
    UN = t.N_opt(t.C_for_D(U)) if SPECS[spec]["RN"] is not None else None
    Npc = float(_Np_of_N_vec(Nc, UN, SPECS[spec]["RN"])) if UN is not None else Nc
    w_eff = wedge_kappa(t, Npc, Dpc)
    # shadow value of a unique token in lifetime cost (central difference at fixed loss)
    Cp, _, _ = life_cost_at_loss(t, ell, U * np.exp(h), spec, Phi)
    Cm, _, _ = life_cost_at_loss(t, ell, U * np.exp(-h), spec, Phi)
    lam = -(Cp - Cm) / (U * (np.exp(h) - np.exp(-h)))
    eta = np.nan
    if spec != "nore" and Dc > U and RD > 0:
        R = Dc / U - 1
        eta = Dc * np.exp(-R / RD) / Dpc
    elif Dc <= U * (1 + 1e-9):
        eta = 1.0
    out.update(train_penalty=6 * Nc * Dc / C - 1, N_ratio=Nc / Nw, D_ratio=Dc / Dw, epochs=Dc / U, N_c=Nc, D_c=Dc,
               mN_c=mN_c, w_true_c=1 + mN_c, s_true_c=mN_c / (1 + mN_c), w_hat_proc=w_proc, w_hat_eff=w_eff,
               s_hat_proc=(w_proc - 1) / w_proc, ratio_proc=w_proc / (1 + mN_c), ratio_eff=w_eff / (1 + mN_c),
               ratio_proc_vs_w=w_proc / w, mu_eff=(1 + mN_c) / w_eff - 1, mu_eta=(1 / eta - 1) if np.isfinite(eta) else np.nan,
               shadow_life=lam, shadow_life_rel=lam / (6 * Nc), elas_life_U=-(np.log(Cp) - np.log(Cm)) / (2 * h))
    return out


def techs():
    """kappa-free reference (sigma* = 0.70) and two members of its observationally equivalent family (same expansion
    path and frontier) with sigma* = 0.60 and 0.74."""
    k1, kq = WL.load_wall_techs()
    return dict(kq=kq, q60=kq.equivalent(0.60, "q60", "sigma* = 0.60 (kappa-free path and frontier)"),
                q74=kq.equivalent(0.74, "q74", "sigma* = 0.74 (kappa-free path and frontier)"))


def w_list():
    """Wedge grid; the high point is the 2025 compute-weighted wedge of the decision units (RC.W_AGG25, set by run.py)."""
    return (1.0, 1.5, 2.0, 3.0, RC.W_AGG25, 8.0)
R_GRID = np.round(np.geomspace(0.25, 32.0, 29), 4)       # compute-optimal scarcity r = D*(C)/U at the model's compute


def grid(TT, C=1e26):
    """Objects on a grid of (w, r) for each technology and specification; r_obs = r w^e."""
    rows = []
    for key, t in TT.items():
        specs = ["D15", "DN", "D3", "nore"] if key == "kq" else ["D15"]
        e = e_of(t.sigma_star)
        for spec in specs:
            for w in w_list():
                for r in R_GRID:
                    r_obs = r * w ** e
                    if r_obs <= 1.0:
                        rows.append(dict(tech=key, sigma_star=t.sigma_star, spec=spec, w=w, r=r, r_obs=r_obs,
                                         life_penalty=0.0, train_penalty=0.0, ratio_proc=1.0, ratio_eff=1.0,
                                         N_ratio=1.0, shadow_life_rel=0.0))
                        continue
                    pt = obs_point(t, C, w, r_obs, spec)
                    pt["r"] = r
                    rows.append(pt)
        log(f"  wall at observed allocations: {key} done")
    return pd.DataFrame(rows)


def checks(TT):
    """(i) w = 1 reproduces ra3's compute-optimal wall; (ii) scale invariance (1e25 vs 1e27); (iii) mu = 1/eta - 1."""
    t = TT["kq"]
    out = []
    for r in (2.0, 4.0, 8.0):
        a = WL.wall_point(t, 1e26, r, "D15", extras=False)["penalty"]
        b = obs_point(t, 1e26, 1.0, r, "D15")["life_penalty"]
        out.append(dict(check=f"w = 1 reproduces ra3 wall penalty, r = {r:g}", value=b, reference=a,
                        passed=bool(abs(a - b) < 1e-6)))
    for key in ("kq", "q60", "q74"):
        for w in (1.5, RC.W_AGG25):
            Nw, Dw, Phi, ell = uncapped(TT[key], 1e26, w)
            _, N, D = life_cost_at_loss(TT[key], ell, np.inf, "D15", Phi)
            dev = max(abs(N / Nw - 1), abs(D / Dw - 1), abs(wedge_kappa(TT[key], Nw, Dw) / w - 1))
            out.append(dict(check=f"uncapped lifetime optimum = closed form (N*, D*) w^(-/+e), wedge = w; {key}, w = {w:g}",
                            value=dev, reference=0.0, passed=bool(dev < 1e-5)))
    for w, r in ((3.0, 4.0), (RC.W_AGG25, 8.0)):
        p1 = obs_point(t, 1e25, w, r, "D15")
        p2 = obs_point(t, 1e27, w, r, "D15")
        dmax = max(abs(p1[k] - p2[k]) for k in ("life_penalty", "ratio_proc", "shadow_life_rel"))
        out.append(dict(check=f"scale invariance 1e25 vs 1e27, w = {w:g}, r_obs = {r:g}", value=dmax, reference=0.0,
                        passed=bool(dmax < 1e-5)))
        out.append(dict(check=f"mu_eff = 1/eta - 1 (Prop. 1(iii) under soft repetition), w = {w:g}, r_obs = {r:g}",
                        value=p1["mu_eff"], reference=p1["mu_eta"], passed=bool(abs(p1["mu_eff"] - p1["mu_eta"]) < 1e-4)))
    return pd.DataFrame(out)


FRONTIER_YEARS = (RC.T_NOW, 2027.0, 2027.5, 2028.0, 2028.5, 2029.0, 2030.0, 2031.0)
SIG_TECH = {0.60: "q60", 0.70: "kq", 0.74: "q74"}


def frontier_path(S, TT, usd_per_flop, years=FRONTIER_YEARS):
    """Cost of the wall along the projected frontier at OBSERVED allocations: baseline, trend and catch-up scenarios
    (scenario.py), frontier compute at 5.06x (and 4.2x), unique stock 100T in 2024 growing 5%/yr (22T and 490T as
    stock sensitivities for the reference curvature)."""
    import scenario as SC
    rows = []
    for gname in ("hat", "epoch"):
        lnC = S["paths"][gname][0]
        for sig in RC.SIGMAS:
            t = TT[SIG_TECH[sig]]
            e = e_of(sig)
            fn = SC.mD_functions(S["mD0"], S["g_w"], sig, S["cap"])
            for scen in ("baseline", "trend", "catchup"):
                specs = ("D15", "D3", "DN") if sig == 0.70 else ("D15",)
                stocks = ((RC.U_STOCK, "100T"), (RC.U_STOCK_LO, "22T"), (RC.U_STOCK_HI, "490T")) if (
                    sig == 0.70 and gname == "hat") else ((RC.U_STOCK, "100T"),)
                for spec in specs:
                    for U0, ulab in stocks:
                        if ulab != "100T" and spec != "D15":
                            continue
                        for yr in years:
                            C = float(np.exp(lnC(yr)))
                            mD = fn[scen](yr)
                            w = mD ** (1 / e)
                            U = U0 * (1 + RC.U_GROWTH) ** (yr - 2024.0)
                            Dw = t.D_opt(C) * mD
                            r_obs = Dw / U
                            base = dict(gC=gname, sigma=sig, scenario=scen, spec=spec, stock=ulab, year=yr, C=C, m_D=mD,
                                        w_sigma=w, D_w=Dw, U=U, r_obs=r_obs, r_opt=t.D_opt(C) / U)
                            if r_obs <= 1.0:
                                base.update(life_penalty=0.0, train_penalty=0.0, N_ratio=1.0, ratio_proc=1.0,
                                            ratio_proc_vs_w=1.0, w_hat_proc=w, shadow_usd_per_Mtok=0.0, epochs=Dw / U)
                            else:
                                pt = obs_point(t, C, w, r_obs, spec)
                                base.update({k: pt.get(k, np.nan) for k in ("life_penalty", "train_penalty", "N_ratio",
                                                                            "ratio_proc", "ratio_proc_vs_w", "w_hat_proc",
                                                                            "epochs", "s_true_c", "s_hat_proc")})
                                base["shadow_usd_per_Mtok"] = pt.get("shadow_life", np.nan) * usd_per_flop * 1e6
                                base["extra_life_usd_bn"] = pt["life_penalty"] * (6 * pt["N_w"] * pt["D_w"] * w) * \
                                    usd_per_flop / 1e9 if np.isfinite(pt["life_penalty"]) else np.inf
                            rows.append(base)
    return pd.DataFrame(rows)


def compute_optimal_vs_observed(TT, S, usd_per_flop):
    """Today's frontier (September 2026): scarcity and wall cost at compute-optimal vs observed allocation, across ra3's
    path technologies (levels anchored at the disclosed runs) and the unique-stock interval."""
    import scenario as SC  # noqa: F401
    T = S["T"]
    rows = []
    C = S["C_now"]
    for k in ("chin_q", "chin", "meta_a2", "deepseek", "farseer", "farseer_eq3", "gadre_rw", "olmo", "marin_dclm"):
        tech = T[k]
        import demand as DM
        mD = DM.data_multiple(tech, S["runs"])
        Dopt = float(np.exp(tech.lnD(np.log(C))))
        for U0, ulab in ((RC.U_STOCK, "100T"), (RC.U_STOCK_LO, "22T"), (RC.U_STOCK_HI, "490T")):
            U = U0 * (1 + RC.U_GROWTH) ** (RC.T_NOW - 2024.0)
            rows.append(dict(key=k, stock=ulab, U=U, Dstar=Dopt, D_obs=mD * Dopt, m_D=mD, r_opt=Dopt / U,
                             r_obs=mD * Dopt / U))
    X = pd.DataFrame(rows)
    # wall cost at the reference technology for these r_obs values, at the reference-implied frontier wedge w_F0 (all
    # relative objects depend on (w, r_obs) only). Levels are re-anchored to observed data use, so the technologies
    # differ only through the path slope a used to carry the anchor from ~2e25 FLOP to today's frontier compute.
    t = TT["kq"]
    w = max(S["wF0"], 1.0)
    out = []
    for _, r in X.iterrows():
        for spec in ("D15", "D3", "DN"):
            if r.r_obs <= 1:
                out.append(dict(key=r.key, stock=r.stock, spec=spec, life_penalty=0.0))
                continue
            pt = obs_point(t, C, w, r.r_obs, spec)
            out.append(dict(key=r.key, stock=r.stock, spec=spec, life_penalty=pt["life_penalty"]))
    Y = pd.DataFrame(out).pivot_table(index=["key", "stock"], columns="spec", values="life_penalty").reset_index()
    Y.columns = [c if c in ("key", "stock") else f"life_penalty_{c}" for c in Y.columns]
    return X.merge(Y, on=["key", "stock"])


def rmax_closed(t, spec):
    """Hard wall (ra3 run.py closed forms): scarcity beyond which the unconstrained frontier loss is unattainable."""
    S, a1, b1 = t.S, t.a1, t.b1
    v = SPECS[spec]
    if v["RN"] is None:
        return (1 + v["RD"]) * (S / a1) ** (1 / b1)
    return ((b1 / S) * (1 + v["RN"]) ** (-a1) + (a1 / S) * (1 + v["RD"]) ** (-b1)) ** (-1 / b1)


def compute_optimal_panel(TT, C=1e26):
    """Table panel: the wall at compute-optimal allocation (w = 1) for the kappa-free reference and its sigma*-0.60/0.74
    equivalents, the kappa = 1 refit, and the kappa-free reference under the other repetition models."""
    k1, _ = WL.load_wall_techs()
    rows = [("Chinchilla, $\\kappa$ free (reference)", TT["kq"], "D15"),
            ("Equivalent, $\\sigma^*=0.60$", TT["q60"], "D15"),
            ("Equivalent, $\\sigma^*=0.74$", TT["q74"], "D15"),
            ("Chinchilla refit, $\\kappa=1$", k1, "D15"),
            ("Reference, full model ($R^*_N=5.3$)", TT["kq"], "DN"),
            ("Reference, data-only fit ($R^*_D=2.9$)", TT["kq"], "D3")]
    out = []
    for lab, t, spec in rows:
        rm = rmax_closed(t, spec)
        d = dict(label=lab, tech=t.key, spec=spec, sigma_star=t.sigma_star, r_max=rm)
        for r in (4.0, 16.0):
            if r < rm * 0.999:
                p = WL.wall_point(t, C, r, spec, extras=(r == 16.0))
                d[f"penalty_r{int(r)}"] = p["penalty"]
                d[f"shadow_rel_r{int(r)}"] = p["shadow_rel"]
                if r == 16.0:
                    d["gamma_ratio_r16"] = p["gamma_ratio"]
            else:
                d[f"penalty_r{int(r)}"] = np.inf
        d["sigma_CU_r4"] = WL.sigma_CU(t, C, 4.0, spec) if 4.0 * np.exp(0.03) < rm else np.nan
        out.append(d)
    return pd.DataFrame(out)


UNDER_POINTS = (1.5, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0)   # exact tabulated D/U values (review fix: no nearest-grid lookup)


def understatement_grid(TT, C=1e26, w=None):
    """w_hat/(1 + m_N) as a function of r_obs (depends on r_obs, hardly on w) under the four repetition models.
    Also w_hat/w, the measured wedge relative to the wedge the same developer (same Phi) would reveal without the wall:
    this is the object behind the trend prediction (the measured trend flattens relative to its no-wall path). The two
    differ because 1 + m_N = 1 + Phi/(6 D) rises when the capped developer processes fewer tokens."""
    w = RC.W_AGG25 if w is None else w
    t = TT["kq"]
    rows = []
    grid = np.union1d(np.round(np.geomspace(1.02, 40.0, 41), 4), np.array(UNDER_POINTS))
    for spec in ("nore", "D3", "D15", "DN"):
        for r_obs in grid:
            p = obs_point(t, C, w, r_obs, spec)
            rows.append({k: p.get(k, np.nan) for k in ("spec", "w", "r_obs", "life_penalty", "train_penalty", "ratio_proc",
                                                       "ratio_eff", "ratio_proc_vs_w", "w_hat_proc", "w_true_c",
                                                       "s_hat_proc", "s_true_c", "N_ratio", "epochs", "mu_eff")})
    return pd.DataFrame(rows)


def main_wall_rows(TT, usd_per_flop, C=1e26):
    """Rows of the merged data-wall panel of the main table: a compute-optimal run (w = 1) at 1e26 FLOP under the three
    curvatures and three repetition models (extra compute, shadow value in $/M, measured wedge)."""
    out = []
    for key in ("q60", "kq", "q74"):
        t = TT[key]
        for r in (4.0, 16.0):
            d = dict(tech=key, sigma_star=t.sigma_star, r=r)
            for spec in ("D15", "D3", "DN"):
                if r >= rmax_closed(t, spec) * 0.999:
                    d[f"cost_{spec}"] = np.inf
                    continue
                p = obs_point(t, C, 1.0, r, spec)
                d[f"cost_{spec}"] = p["life_penalty"]
                if spec == "D15":
                    d["shadow_usd_per_Mtok"] = p["shadow_life"] * usd_per_flop * 1e6
                    d["ratio_proc"] = p["ratio_proc"]
            out.append(d)
    return pd.DataFrame(out)
