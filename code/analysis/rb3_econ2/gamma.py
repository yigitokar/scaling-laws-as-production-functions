"""gamma.py -- guidance on the frontier elasticity gamma for growth calibrations at 1e25-1e27 FLOP (R3 minor 20; R2
minors 14 and 20; R4 minor 10).

gamma is the compute elasticity of REDUCIBLE loss on the loss-compute frontier, L*(C) = E + K (C/6)^(-gamma); halving
reducible loss takes 2^(1/gamma) times the compute. It is cardinal in reducible loss, so it is identified only together
with the irreducible loss E. Three pieces of evidence:
  (1) the candidates: joint fits on the full design (E identified by off-path runs in both inputs) and frontier-only
      fits (Approach 1: E, K, gamma from the IsoFLOP minima), from the m1/m2 registries;
  (2) E-profiles of the frontier: for each IsoFLOP design, ra1's model-free per-budget minima L*_b are fitted with E held
      fixed on a grid (nonlinear least squares in levels for (ln K, gamma)); the profile 95% set for E uses the
      F-calibrated SSR threshold SSR_min [1 + F(0.95; 1, n - 3)/(n - 3)], and the range of gamma over it shows how much
      of gamma the frontier alone pins down;
  (3) local gamma at a design's top budgets with E fixed at the joint-fit value (is there a drift like sigma*(C)?).
Outputs also give the total-loss elasticity gamma (L* - E)/L* at 1e25-1e27 FLOP, which is what a model that maps compute
into total loss (rather than reducible loss) needs.

Round 3 (fix list E1(a); audit numbers_appF #2 (m2)): the per-budget minima are rb4's (RC.RA1_BUDGETS: Chinchilla in
FLOP-effective parameters, the other designs as in ra1). Chinchilla's E-profile on ra1's total-parameter minima is kept
as a convention comparison (rb3_econ2_gamma_eprofiles_T.csv).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import f as fdist

import rb3common as RC
from rb3common import G_EPOCH, log

import wall as WL  # noqa: E402  (ra3_econ, read-only)

DESIGNS = ["Chinchilla", "Llama 3", "Marin, DCLM", "Marin, Nemotron-CC", "Marin, Comma"]


def _fit_fixedE(C, L, E):
    y = L - E
    if np.any(y <= 0):
        return np.inf, np.nan, np.nan
    x = np.log(C / 6.0)
    b = np.polyfit(x, np.log(y), 1)
    p0 = np.array([b[1], -b[0]])

    def res(p):
        return E + np.exp(p[0] - p[1] * x) - L
    r = least_squares(res, p0, method="lm", xtol=1e-14, ftol=1e-14)
    return float(np.sum(r.fun ** 2)), float(r.x[1]), float(r.x[0])


def e_profiles(path=None, designs=DESIGNS):
    b = pd.read_csv(RC.RA1_BUDGETS if path is None else path)
    rows, prof = [], []
    for des in designs:
        g = b[b.design == des].sort_values("budget_C")
        C, L = g.budget_C.values, g.Lstar.values
        n = len(g)
        Emax = L.min() - 1e-4
        Es = np.linspace(0.0, Emax, 1201)
        fits = [(_fit_fixedE(C, L, E)) for E in Es]
        ssr = np.array([f[0] for f in fits])
        gam = np.array([f[1] for f in fits])
        i = int(np.nanargmin(ssr))
        thr = ssr[i] * (1 + fdist.ppf(0.95, 1, n - 3) / (n - 3))
        inset = ssr <= thr
        for E, s_, g_ in zip(Es, ssr, gam):
            prof.append(dict(design=des, E=E, ssr=s_, gamma=g_, in_set=bool(s_ <= thr)))
        rows.append(dict(design=des, n_budgets=n, C_min=C.min(), C_max=C.max(), decades=np.log10(C.max() / C.min()),
                         E_hat=Es[i], gamma_hat=gam[i], E_lo=Es[inset].min(), E_hi=Es[inset].max(),
                         E_at_boundary=bool(inset[0] or inset[-1] or i in (0, len(Es) - 1)),
                         gamma_lo=np.nanmin(gam[inset]), gamma_hi=np.nanmax(gam[inset]), L_min=L.min()))
    return pd.DataFrame(rows), pd.DataFrame(prof)


def local_gamma_chinchilla(E_joint):
    """Local gamma between successive halves of Chinchilla's and Llama 3's budgets at a fixed E."""
    b = pd.read_csv(RC.RA1_BUDGETS)
    out = []
    for des, E in (("Chinchilla", E_joint), ):
        g = b[b.design == des].sort_values("budget_C")
        C, L = g.budget_C.values, g.Lstar.values
        x, y = np.log(C), np.log(L - E)
        k = len(g)
        for lab, sl in (("lower budgets", slice(0, k // 2 + 1)), ("upper budgets", slice(k // 2, k)), ("all", slice(0, k))):
            out.append(dict(design=des, E=E, window=lab, C_lo=C[sl].min(), C_hi=C[sl].max(),
                            gamma_local=-np.polyfit(x[sl], y[sl], 1)[0]))
        # per-decade drift of the local slope (successive pairs)
        gp = -(np.diff(y) / np.diff(x))
        mid = 0.5 * (x[1:] + x[:-1]) / np.log(10)
        out.append(dict(design=des, E=E, window="drift per decade (successive pairs)", C_lo=C.min(), C_hi=C.max(),
                        gamma_local=float(np.polyfit(mid, gp, 1)[0])))
    return pd.DataFrame(out)


def candidates(g_hat):
    r1 = pd.read_csv(RC.M1_REG).set_index("row_id")
    r2 = pd.read_csv(RC.M2_REG)
    duo = pd.read_csv(RC.ROOT + "/output/tables/m1_chinchilla_duality_objects.csv")
    duo = duo[duo.dataset == "Chinchilla n=240"].set_index("object")
    q = r2[(r2.dataset == "chinchilla") & (r2.estimator == "huber_q")].iloc[0]
    fq = r2[(r2.dataset == "farseer") & (r2.subset == "all") & (r2.estimator == "huber_q")].iloc[0]
    f1 = r2[(r2.dataset == "farseer") & (r2.subset == "all") & (r2.estimator == "huber")].iloc[0]
    kq = WL.KTech("kq", "", q.E, q.A, q.B, q.alpha, q.beta, q.q)
    c1 = r1.loc["chin_n240_huber"]
    k1 = WL.KTech("k1", "", c1.E, c1.A, c1.B, c1.alpha, c1.beta, 1.0)
    kf = WL.KTech("fq", "", fq.E, fq.A, fq.B, fq.alpha, fq.beta, fq.q)
    kf1 = WL.KTech("f1", "", f1.E, f1.A, f1.B, f1.alpha, f1.beta, 1.0)
    l3 = r1.loc["llama_3_A2A1_kappa1"]
    rows = [
        dict(key="chin_q", technology="Chinchilla, joint fit, kappa free (reference)", gamma=q.gamma, se=q.se_gamma,
             window="6e18-1.3e22 (240 runs)", E_source="joint fit (off-path runs)", loss_units="nats/token, MassiveText",
             kappa_status="not rejected", tech=kq, recommended=True),
        dict(key="chin_k1", technology="Chinchilla, joint fit, kappa = 1", gamma=c1.gamma, se=c1.se_gamma,
             window="6e18-1.3e22 (240 runs)", E_source="joint fit (off-path runs)", loss_units="nats/token, MassiveText",
             kappa_status="rejected (Section III)", tech=k1, recommended=False),
        dict(key="chin_a1", technology="Chinchilla, IsoFLOP frontier (E free)", gamma=float(duo.loc["gamma", "A2_A1"]),
             se=float(duo.loc["gamma", "A2_A1_se"]), window="6e18-3e21 (9 minima)", E_source="frontier only",
             loss_units="nats/token, MassiveText", kappa_status="--", tech=None, recommended=False),
        dict(key="hoff", technology="Hoffmann et al. (published A3)", gamma=r1.loc["hoffmann_A3_tex", "gamma"], se=np.nan,
             window="6e18-1.3e22", E_source="joint fit", loss_units="nats/token, MassiveText",
             kappa_status="kappa = 1", tech=None, recommended=False),
        dict(key="llama_a1", technology="Llama 3, IsoFLOP frontier (Meta's A1)", gamma=l3.gamma, se=l3.se_gamma,
             window="6e18-1e22 (10 minima)", E_source="frontier only", loss_units="unstated validation loss",
             kappa_status="--", tech=None, recommended=False),
        dict(key="llama_a3", technology="Llama 3, joint fit on IsoFLOP runs, kappa = 1",
             gamma=r1.loc["llama_3_A3_huber", "gamma"], se=r1.loc["llama_3_A3_huber", "se_gamma"],
             window="6e18-1e22", E_source="joint fit (IsoFLOP runs only)", loss_units="unstated validation loss",
             kappa_status="kappa = 1", tech=None, recommended=False),
        dict(key="farseer_k1", technology="Farseer, joint fit, kappa = 1", gamma=f1.gamma, se=f1.se_gamma,
             window="up to 3.5e21", E_source="joint fit", loss_units="nats/token, own validation set",
             kappa_status="rejected", tech=kf1, recommended=False),
        dict(key="farseer_q", technology="Farseer, joint fit, kappa free", gamma=fq.gamma, se=fq.se_gamma,
             window="up to 3.5e21", E_source=f"joint fit (E = {fq.E:.2f})", loss_units="nats/token, own validation set",
             kappa_status="not rejected", tech=kf, recommended=False),
    ]
    out = []
    lgC, lgE = np.log(g_hat), np.log(G_EPOCH)
    for r in rows:
        g, se = float(r["gamma"]), float(r["se"]) if np.isfinite(r["se"]) else np.nan
        mult = 2 ** (1 / g)
        d = {k: v for k, v in r.items() if k != "tech"}
        d.update(mult_halving=mult, mult_lo=2 ** (1 / (g + 1.96 * se)) if np.isfinite(se) else np.nan,
                 mult_hi=2 ** (1 / (g - 1.96 * se)) if np.isfinite(se) else np.nan,
                 oom_halving=np.log10(mult), compute_per_1pct=1 / g,
                 years_5p06=np.log(mult) / lgC, years_4p2=np.log(mult) / lgE)
        t = r["tech"]
        if t is not None:
            for c in (1e25, 1e26, 1e27):
                Ls = t.L_opt(c)
                d[f"L_star_{c:.0e}"] = Ls
                d[f"reducible_share_{c:.0e}"] = (Ls - t.E) / Ls
                d[f"total_loss_elasticity_{c:.0e}"] = t.gamma * (Ls - t.E) / Ls
            d["E"] = t.E
        out.append(d)
    return pd.DataFrame(out), float(q.E), float(c1.E)


def run(g_hat):
    Cand, E_q, E_1 = candidates(g_hat)
    Prof, P = e_profiles()
    # gamma of each frontier at the joint-fit E of Chinchilla (same units only for Chinchilla)
    b = pd.read_csv(RC.RA1_BUDGETS)
    g = b[b.design == "Chinchilla"]
    for lab, E in (("kappa free", E_q), ("kappa = 1", E_1)):
        s, gam, _ = _fit_fixedE(g.budget_C.values, g.Lstar.values, E)
        Prof.loc[Prof.design == "Chinchilla", f"gamma_at_joint_E_{lab.replace(' ', '').replace('=', '')}"] = gam
    Prof["joint_E_q_in_set"] = np.where(Prof.design == "Chinchilla", (Prof.E_lo <= E_q) & (E_q <= Prof.E_hi), np.nan)
    ProfT, _ = e_profiles(RC.RA1_BUDGETS_T, ["Chinchilla"])
    ProfT["convention"] = "total parameters (ra1 minima)"
    ProfT["joint_E_q_in_set"] = (ProfT.E_lo <= E_q) & (E_q <= ProfT.E_hi)
    Loc = local_gamma_chinchilla(E_q)
    log(f"gamma: recommended {Cand.iloc[0].gamma:.3f} -> {Cand.iloc[0].mult_halving:.0f}x per halving; Chinchilla "
        f"frontier E-profile gamma in [{Prof.iloc[0].gamma_lo:.3f}, {Prof.iloc[0].gamma_hi:.3f}]")
    return dict(Cand=Cand, Prof=Prof, ProfT=ProfT, P=P, Loc=Loc, E_q=E_q)
