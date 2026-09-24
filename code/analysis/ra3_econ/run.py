"""run.py -- module ra3_econ: economic implications of the estimated technologies (addresses R3 M7; R1 comment 6e).

  1. Data-demand growth D* ~ C^(1-a) under alternative path slopes a; frontier compute growth estimated from Epoch AI's
     database (2018-2026); exhaustion of the high-quality public text stock (Villalobos et al. 2024).
  2. The data wall: Chinchilla-family technologies (kappa = 1, kappa free, sigma* = 0.60 equivalent) x Muennighoff et
     al.'s repetition model: extra compute from a binding cap on unique tokens, the shadow value of a unique token,
     the elasticity of substitution between compute and unique data, returns to compute behind the wall.
  3. The planned inference share of lifetime compute by release period, from module ra2_wedge's final reviewed
     inversion (kappa-free reference; ra2's clean open-weight universe and clean verified sample; ra2's wild bootstrap;
     all 32 ex-ante technologies), reproduced exactly from ra2's files, and its comparison with external disclosures.
     (The builder's preliminary m3-universe version is kept only as a superseded robustness file.)
  4. Growth-model calibration: compute multiple per halving of reducible loss, 2^(1/gamma).

Deterministic (fixed seeds), CPU only, at most 5 processes (ra2's wild bootstrap), about 1-2 minutes.
Usage: .venv/bin/python code/analysis/ra3_econ/run.py
"""
from __future__ import annotations

import json
import os
import time
import warnings

import numpy as np
import pandas as pd

import ra3common as RC
from ra3common import P, PROC, TABLES, log

warnings.filterwarnings("ignore", category=RuntimeWarning)

import demand as DM  # noqa: E402
import exhibits as EX  # noqa: E402
import growth as GR  # noqa: E402
import share as SH  # noqa: E402
import wall as WL  # noqa: E402

C_REF = 1e26
R_GRID = np.round(np.geomspace(1.0, 64.0, 37), 4)
R_TAB = [2, 4, 8, 16, 32]


def usd_per_flop():
    """Median training cost per FLOP of 2024-2025 frontier runs in Epoch's frontier file (Confident/Likely only):
    amortized hardware + energy (2023 USD) and cloud-rental cost."""
    f = pd.read_csv(RC.EPOCH_FRONTIER, low_memory=False)
    f["date"] = pd.to_datetime(f["Publication date"], errors="coerce")
    f["C"] = pd.to_numeric(f["Training compute (FLOP)"], errors="coerce")
    f = f[(f["date"] >= "2024-01-01") & (f["date"] <= "2025-12-31") & f["Confidence"].isin(["Confident", "Likely"])]
    out = []
    for col, nm in [("Training compute cost (2023 USD)", "amortized"), ("Training compute cost (cloud)", "cloud")]:
        x = f.dropna(subset=[col, "C"])
        x = x[pd.to_numeric(x[col], errors="coerce") > 0]
        p = pd.to_numeric(x[col]) / x["C"]
        for _, r in x.assign(p=p).iterrows():
            out.append(dict(basis=nm, model=r["Model"], date=r["date"].date(), C=r["C"], cost_usd=float(r[col]),
                            usd_per_flop=r["p"]))
    D = pd.DataFrame(out)
    med = D.groupby("basis")["usd_per_flop"].median().to_dict()
    return D, med


def part1(t0):
    log("part 1: frontier compute growth")
    d = GR.load_epoch()
    tr, draws = GR.fit_all(d)
    tr.to_csv(os.path.join(TABLES, P + "compute_growth.csv"), index=False)
    ep = pd.DataFrame(RC.EPOCH_GROWTH)
    ep.to_csv(os.path.join(TABLES, P + "compute_growth_epoch_published.csv"), index=False)
    fr = d[(d["date"] >= GR.START) & d["top10"]][["Model", "Organization", "dev", "date", "C", "Confidence", "Domain"]]
    fr.to_csv(os.path.join(PROC, "frontier_top10_2018_2026.csv"), index=False)
    np.save(os.path.join(PROC, "wcb_trend_draws_top10.npy"), draws["top10"])
    trend = tr.set_index("sample").loc["top10"]
    log(f"  top-10 growth {trend.growth:.2f}x/yr [{trend.growth_lo:.2f}, {trend.growth_hi:.2f}] ({time.time()-t0:.0f}s)")

    log("part 1: data demand under alternative a")
    T = DM.load_path_techs()
    runs = DM.frontier_runs()
    runs.to_csv(os.path.join(PROC, "frontier_runs_disclosed_D.csv"), index=False)
    gC = [("hat", float(trend.growth)), ("4x", 4.0), ("4.5x", 4.5), ("5x", 5.0)]
    dem = DM.demand_table(T, trend, gC, runs)
    # sensitivity (review): anchoring runs including the MoE run (active N), as in the builder's version
    runs_moe = DM.frontier_runs(exclude_moe=False)
    dm2 = DM.demand_table(T, trend, gC, runs_moe)[["key", "data_multiple_obs", "year_unique_obsOT", "year_effective_obsOT"]]
    dem = dem.merge(dm2.rename(columns={c: c + "_withMoE" for c in dm2.columns if c != "key"}), on="key", how="left")
    dem.to_csv(os.path.join(TABLES, P + "data_demand.csv"), index=False)
    # Monte Carlo intervals for technologies with bootstrap draws
    mc_rows = []
    for j, k in enumerate(["chin", "chin_q", "meta_a2", "farseer", "gadre_rw", "olmo"]):
        for i, stock in enumerate(["unique", "effective"]):
            mc = DM.exhaustion_mc(T[k], trend, draws["top10"], runs, stock=stock, seed=RC.SEED + 10 * j + i)
            for (src, ot), q in mc.items():
                mc_rows.append(dict(key=k, stock=stock, source=src, scenario=ot, p5=q[0], p50=q[1], p95=q[2]))
    mc = pd.DataFrame(mc_rows)
    mc.to_csv(os.path.join(TABLES, P + "exhaustion_mc.csv"), index=False)
    # data-demand paths for the figure
    years = np.linspace(2020, 2034, 141)
    paths = []
    for k in ["chin", "meta_a2", "gadre_rw", "olmo", "farseer_eq3", "chin_q", "deepseek", "farseer"]:
        t = T[k]
        mD = DM.data_multiple(t, runs)
        lnC = np.log(10) * (trend.b0 + trend.b1 * (years - RC.T0))
        paths.append(pd.DataFrame(dict(key=k, a=t.a, year=years, C=np.exp(lnC), Dstar=np.exp(t.lnD(lnC)),
                                       D_obsOT=mD * np.exp(t.lnD(lnC)))))
    paths = pd.concat(paths)
    paths.to_csv(os.path.join(PROC, "data_demand_paths.csv"), index=False)
    return dict(trend=trend, tr=tr, T=T, runs=runs, runs_moe=runs_moe, dem=dem, mc=mc, paths=paths, epoch=d)


def part2(t0, dem_trend, r3=None):
    log("part 2: data wall")
    k1, kq = WL.load_wall_techs()
    fam = [k1, kq, k1.equivalent(0.70, "e70", "sigma* = 0.70 (kappa = 1 path and frontier)"),
           k1.equivalent(0.60, "e60", "sigma* = 0.60 (kappa = 1 path and frontier)")]
    extra = [k1.equivalent(s, f"e{int(round(s * 100))}", f"sigma* = {s:.2f} (equivalent)") for s in (0.51, 0.65, 0.80, 0.90)]
    techs = pd.DataFrame([dict(key=t.key, label=t.label, E=t.E, A=t.A, B=t.B, a1=t.a1, b1=t.b1, kappa=t.kappa, a=t.a,
                               sigma_star=t.sigma_star, gamma=t.gamma, lnG=t.lnG, K=t.K, Dstar_1e26=t.D_opt(C_REF),
                               Nstar_1e26=t.N_opt(C_REF), Lstar_1e26=t.L_opt(C_REF)) for t in fam + extra])
    techs.to_csv(os.path.join(TABLES, P + "wall_technologies.csv"), index=False)

    # r_max: largest scarcity ratio at which the unconstrained frontier loss is attainable (closed form)
    rmax = []
    for t in fam + extra:
        for sp, v in WL.SPECS.items():
            S, a1, b1 = t.S, t.a1, t.b1
            RD = v["RD"]
            if v["RN"] is None:
                val = (1 + RD) * (S / a1) ** (1 / b1)
            else:
                val = ((b1 / S) * (1 + v["RN"]) ** (-a1) + (a1 / S) * (1 + RD) ** (-b1)) ** (-1 / b1)
            rmax.append(dict(tech=t.key, sigma_star=t.sigma_star, spec=sp, r_max=val))
    rmax = pd.DataFrame(rmax)
    rmax.to_csv(os.path.join(TABLES, P + "wall_rmax.csv"), index=False)

    # grid over r (all specs) at C_REF
    rows = []
    for t in fam + extra:
        for sp in WL.SPECS:
            rm = rmax[(rmax.tech == t.key) & (rmax.spec == sp)].r_max.iloc[0]
            for r in R_GRID:
                if r >= rm * 0.999:
                    rows.append(dict(tech=t.key, sigma_star=t.sigma_star, spec=sp, r=r, penalty=np.inf))
                    continue
                pt = WL.wall_point(t, C_REF, r, sp, extras=(t in fam))
                pt["sigma_star"] = t.sigma_star
                rows.append(pt)
    grid = pd.DataFrame(rows)
    grid.to_csv(os.path.join(PROC, "wall_grid.csv"), index=False)
    log(f"  grid done ({time.time()-t0:.0f}s)")

    # sigma_CU at tabulated r (core technologies)
    sc = []
    for t in fam:
        for sp in ["D15", "DN", "D3"]:
            for r in [1.5, 2, 4, 8, 16]:
                rm = rmax[(rmax.tech == t.key) & (rmax.spec == sp)].r_max.iloc[0]
                sc.append(dict(tech=t.key, sigma_star=t.sigma_star, spec=sp, r=r,
                               sigma_CU=WL.sigma_CU(t, C_REF, r, sp) if r * np.exp(0.03) < rm else np.nan))
    sc = pd.DataFrame(sc)
    sc.to_csv(os.path.join(TABLES, P + "wall_sigma_CU.csv"), index=False)

    # scale invariance check (r fixed, C = 1e25 vs 1e27)
    chk = []
    for t in fam:
        for r in [4, 16]:
            a, b = WL.wall_point(t, 1e25, r, "D15"), WL.wall_point(t, 1e27, r, "D15")
            chk.append(dict(tech=t.key, r=r, penalty_1e25=a["penalty"], penalty_1e27=b["penalty"],
                            shadow_rel_1e25=a["shadow_rel"], shadow_rel_1e27=b["shadow_rel"],
                            max_abs_diff=max(abs(a["penalty"] - b["penalty"]), abs(a["shadow_rel"] - b["shadow_rel"]),
                                             abs(a["gamma_ratio"] - b["gamma_ratio"])),
                            closed_vs_numeric_shadow=abs(a["shadow_flop"] / a["shadow_flop_closed"] - 1)))
    chk = pd.DataFrame(chk)
    chk.to_csv(os.path.join(TABLES, P + "wall_checks.csv"), index=False)
    # further numerical checks of closed forms: r_max (finite just below, infeasible just above), the wedge-data
    # multiple w^(sigma/(2(1-sigma))) against a direct root-find, and the equivalence of the family members
    from scipy.optimize import brentq
    ck = []
    for t in fam:
        rm = rmax[(rmax.tech == t.key) & (rmax.spec == "D15")].r_max.iloc[0]
        below = WL.wall_point(t, C_REF, rm * 0.98, "D15", extras=False)["penalty"]
        above = WL.cost_at_loss(t, t.L_opt(C_REF), t.D_opt(C_REF) / (rm * 1.02), "D15")[0]
        ck.append(dict(check="r_max: penalty at 0.98 r_max finite; cost at 1.02 r_max infinite", tech=t.key,
                       value=below, value2=above, passed=bool(np.isfinite(below) and np.isinf(above))))

        def w_of(lnN, t=t):
            N = np.exp(lnN)
            D = C_REF / (6 * N)
            return t.a1 * t.A * N ** -t.a1 / (t.b1 * t.B * D ** -t.b1)
        lnN = brentq(lambda x: w_of(x) - 3.0, np.log(t.N_opt(C_REF)) - 10, np.log(t.N_opt(C_REF)))
        num = C_REF / (6 * np.exp(lnN)) / t.D_opt(C_REF)
        form = float(WL.wedge_data_multiple(3.0, t.sigma_star))
        ck.append(dict(check="wedge-data multiple at w = 3: numeric vs closed form", tech=t.key, value=num,
                       value2=form, passed=bool(abs(num / form - 1) < 1e-8)))
        if t.key.startswith("e"):
            dev = max(abs(t.L_opt(c) / fam[0].L_opt(c) - 1) for c in (1e20, 1e24, 1e28))
            ck.append(dict(check="equivalent member: same frontier as kappa = 1 refit (max rel. dev.)", tech=t.key,
                           value=dev, value2=abs(t.a - fam[0].a), passed=bool(dev < 1e-10)))
    pd.DataFrame(ck).to_csv(os.path.join(TABLES, P + "checks.csv"), index=False)

    # bootstrap intervals (technology sampling uncertainty) at r = 4, 16 (primary spec)
    bt = []
    for name, dr in [("k1", WL.draws_k1()), ("kq", WL.draws_kq())]:
        for i, t in enumerate(dr):
            for r in [4, 16]:
                try:
                    pt = WL.wall_point(t, C_REF, r, "D15", extras=False)
                    bt.append(dict(tech=name, draw=i, r=r, sigma_star=t.sigma_star, penalty=pt["penalty"],
                                   shadow_rel=pt["shadow_rel"], epochs=pt["epochs"]))
                except Exception:  # noqa: BLE001
                    bt.append(dict(tech=name, draw=i, r=r, sigma_star=t.sigma_star, penalty=np.nan))
    bt = pd.DataFrame(bt)
    bt.to_csv(os.path.join(PROC, "wall_boot.csv"), index=False)
    bsum = bt.replace([np.inf, -np.inf], np.nan).groupby(["tech", "r"]).agg(
        n=("penalty", "size"), n_finite=("penalty", lambda x: int(np.isfinite(x).sum())),
        penalty_lo=("penalty", lambda x: np.nanpercentile(x, 2.5)), penalty_hi=("penalty", lambda x: np.nanpercentile(x, 97.5)),
        shadow_rel_lo=("shadow_rel", lambda x: np.nanpercentile(x, 2.5)),
        shadow_rel_hi=("shadow_rel", lambda x: np.nanpercentile(x, 97.5)),
        sigma_lo=("sigma_star", lambda x: np.nanpercentile(x, 2.5)), sigma_hi=("sigma_star", lambda x: np.nanpercentile(x, 97.5))
    ).reset_index()
    bsum.to_csv(os.path.join(TABLES, P + "wall_boot_summary.csv"), index=False)
    log(f"  bootstrap done ({time.time()-t0:.0f}s)")

    # absolute scenarios: frontier compute 1e25-1e28, unique caps 10T/30T/100T; dollars
    prices, pmed = usd_per_flop()
    prices.to_csv(os.path.join(TABLES, P + "usd_per_flop.csv"), index=False)
    ab = []
    for t in fam:
        for C in [1e25, 1e26, 1e27, 1e28]:
            for U in [10e12, 30e12, 100e12]:
                r = t.D_opt(C) / U
                rm = rmax[(rmax.tech == t.key) & (rmax.spec == "D15")].r_max.iloc[0]
                if r <= 1:
                    ab.append(dict(tech=t.key, sigma_star=t.sigma_star, C=C, U=U, r=r, penalty=0.0, shadow_flop=0.0,
                                   extra_flop=0.0))
                    continue
                if r >= rm:
                    ab.append(dict(tech=t.key, sigma_star=t.sigma_star, C=C, U=U, r=r, penalty=np.inf))
                    continue
                pt = WL.wall_point(t, C, r, "D15", extras=False)
                ab.append(dict(tech=t.key, sigma_star=t.sigma_star, C=C, U=U, r=r, penalty=pt["penalty"],
                               extra_flop=pt["penalty"] * C, shadow_flop=pt["shadow_flop"], shadow_rel=pt["shadow_rel"],
                               epochs=pt["epochs"]))
    ab = pd.DataFrame(ab)
    for nm, p in pmed.items():
        ab[f"extra_usd_{nm}"] = ab["extra_flop"] * p
        ab[f"shadow_usd_per_Mtok_{nm}"] = ab["shadow_flop"] * p * 1e6
    ab.to_csv(os.path.join(TABLES, P + "wall_absolute.csv"), index=False)

    # data multiple implied by a given wedge (inference demand) under each curvature
    wm = []
    w_list = [(f"{w:g}", w) for w in [1.5, 2.0, 3.0, 4.0, 5.0]]
    if r3 is not None:   # illustrative: a model whose wedge equals ra2's compute-weighted aggregate multiple 1 + m
        u = r3["uni"].set_index("period")
        w_list += [(f"aggregate {per} (1 + m)", 1 + float(u.loc[per, "m_ref"])) for per in ("2024", "2025")]
    for t in fam + extra:
        for wlab, w in w_list:
            wm.append(dict(tech=t.key, sigma_star=t.sigma_star, w_label=wlab, w=w, s=(w - 1) / w,
                           data_multiple=float(WL.wedge_data_multiple(w, t.sigma_star)),
                           M_multiple=float(WL.wedge_data_multiple(w, t.sigma_star)) ** 2))
    wm = pd.DataFrame(wm)
    wm.to_csv(os.path.join(TABLES, P + "wedge_data_multiple.csv"), index=False)
    return dict(fam=fam, extra=extra, rmax=rmax, grid=grid, sc=sc, chk=chk, bsum=bsum, ab=ab, prices=prices,
                pmed=pmed, wm=wm, techs=techs)


def part3(t0, g_log):
    log("part 3: inference share (module ra2_wedge, final)")
    R = SH.run_share()
    R["uni"].to_csv(os.path.join(TABLES, P + "inference_share_universe.csv"), index=False)
    R["uni_dd"].to_csv(os.path.join(TABLES, P + "inference_share_dedup.csv"), index=False)
    R["cln"].to_csv(os.path.join(TABLES, P + "inference_share_clean.csv"), index=False)
    R["bytech"].to_csv(os.path.join(TABLES, P + "inference_share_by_tech.csv"), index=False)
    R["chk"].to_csv(os.path.join(TABLES, P + "inference_share_checks.csv"), index=False)
    R["m3u"].assign(status="SUPERSEDED preliminary (m3 universe, kappa = 1 reference)").to_csv(
        os.path.join(TABLES, P + "inference_share_m3universe.csv"), index=False)
    B = R["B"]
    B["source"] = "ra2_wedge final: clean open-weight universe (rebuilt with ra2 code) and clean verified sample"
    B.to_csv(os.path.join(TABLES, P + "inference_share.csv"), index=False)
    if not R["chk"]["passed"].all():
        raise RuntimeError("ra2 aggregate not reproduced: see inference_share_checks.csv")
    log(f"  ra2 aggregates reproduced exactly ({len(R['chk'])} checks); ({time.time()-t0:.0f}s)")
    dis = SH.disclosures()
    dis.to_csv(os.path.join(TABLES, P + "inference_disclosures.csv"), index=False)
    u = R["uni"].set_index("period")
    ms = [(per, float(u.loc[per, "m_ref"])) for per in ("2024", "2025", "All 2019-2025")]
    # vintage growth: frontier compute growth (primary) and slower illustrative aggregate growth rates
    gl = [("frontier (estimated)", g_log), ("3x per year (illustrative)", float(np.log(3.0))),
          ("2x per year (illustrative)", float(np.log(2.0)))]
    fl = SH.flow_adjustment(ms, gl)
    fl.to_csv(os.path.join(TABLES, P + "inference_share_flow.csv"), index=False)
    R.update(B=B, dis=dis, fl=fl)
    return R


def part4(g_hat, r3=None):
    log("part 4: growth-model calibration")
    r1 = pd.read_csv(RC.M1_REG).set_index("row_id")
    r2 = pd.read_csv(RC.M2_REG)
    q = r2[(r2.dataset == "chinchilla") & (r2.subset == "all") & (r2.estimator == "huber_q")].iloc[0]
    fs = r2[(r2.dataset == "farseer") & (r2.subset == "all") & (r2.estimator == "huber")].iloc[0]
    duo = pd.read_csv(os.path.join(RC.ROOT, "output", "tables", "m1_chinchilla_duality_objects.csv"))
    duo = duo[duo["dataset"] == "Chinchilla n=240"].set_index("object")
    rows = [
        ("Chinchilla refit, kappa = 1", r1.loc["chin_n240_huber", "gamma"], r1.loc["chin_n240_huber", "se_gamma"]),
        ("Chinchilla refit, kappa free", q.gamma, q.se_gamma),
        ("Chinchilla IsoFLOP frontier (A1, E free)", duo.loc["gamma", "A2_A1"], duo.loc["gamma", "A2_A1_se"]),
        ("Hoffmann et al. A3 (published)", r1.loc["hoffmann_A3_tex", "gamma"], np.nan),
        ("Farseer, kappa = 1 (non-emb. N)", fs.gamma, fs.se_gamma),
        ("Llama 3 IsoFLOPs, primal (A3)", r1.loc["llama_3_A3_huber", "gamma"], np.nan),
        ("Llama 3 frontier (A1)", r1.loc["llama_3_A2A1_kappa1", "gamma"], np.nan),
    ]
    g_algo = 2 ** (12 / RC.ALGO_DOUBLING_MONTHS)
    out = []
    for lab, g, se in rows:
        mult = 2 ** (1 / g)
        d = dict(technology=lab, gamma=g, se_gamma=se, compute_multiple_per_halving=mult,
                 oom_per_halving=np.log10(mult), years_per_halving_raw=np.log(mult) / np.log(g_hat),
                 years_per_halving_effective=np.log(mult) / np.log(g_hat * g_algo), g_compute=g_hat, g_algo=g_algo)
        if np.isfinite(se):
            d["mult_lo"], d["mult_hi"] = 2 ** (1 / (g + 1.96 * se)), 2 ** (1 / (g - 1.96 * se))
        out.append(d)
    gc = pd.DataFrame(out)
    gc.to_csv(os.path.join(TABLES, P + "growth_calibration.csv"), index=False)
    # GATE-type training-inference trade-off slope implied by the wedge: m = 1/(w - 1)
    wl = [(f"{w:g}", w) for w in [1.25, 1.5, 2.0, 3.0, 4.0, 5.0]]
    if r3 is not None:
        u = r3["uni"].set_index("period")
        wl += [(f"aggregate {per} (1 + m)", 1 + float(u.loc[per, "m_ref"])) for per in ("2024", "2025", "All 2019-2025")]
    gm = pd.DataFrame([dict(w_label=lab, w=w, s=(w - 1) / w, m_oom_inference_per_oom_training=1 / (w - 1))
                       for lab, w in wl])
    gm.to_csv(os.path.join(TABLES, P + "gate_tradeoff_slope.csv"), index=False)
    return dict(gc=gc, gm=gm)


def main():
    t0 = time.time()
    np.random.seed(RC.SEED)
    r1 = part1(t0)
    r3 = part3(t0, float(np.log(r1["trend"].growth)))
    r2 = part2(t0, r1["trend"], r3)
    r4 = part4(float(r1["trend"].growth), r3)
    log("exhibits")
    EX.main_table(r1, r2, r3, r4)
    EX.appendix_tables(r1, r2, r3, r4)
    EX.figure(r1, r2)
    EX.figure_share(r3)
    summ = EX.summary(r1, r2, r3, r4)
    with open(os.path.join(PROC, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1, default=float)
    log(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
