"""rb4_review.py -- the builder's own skeptical checks (memo, "Self-review"). Run as the last stage of run.py; the
from-scratch determinism check is `python rb4_review.py --compare <root_a> <root_b>` after a second run into a scratch
root (RB4_OUTPUT_ROOT).

Checks written to output/tables/rb4_chinflop_review_checks.csv:
  R1  ra1 reproduction: the T variant through this module's wrapper equals ra1's published Chinchilla rows (sigma_b,
      s.e., intervals, M*_b) and summary; the parametric T row equals ra1's isoflop_param row.
  R2  rb1 reproduction: the 'old' propagation equals rb1's published study-level and meta-regression tables.
  R3  override integrity: the override CSVs differ from ra1's only in Chinchilla rows.
  R4  Table A4 (validation of the FLOP formula) and Table A9 (parameter formula).
  R5  eta approximation: at fixed windows, S_T4/S_T per budget against 1/(1+eta_window)^2.
  R6  first-order ('slope') vs kappa-surface corrections: design-level plain means of sigma*_b.
  R7  coordinate test robustness: without the five highest-loss runs; budget-by-budget sign of the slope.
  R8  P without the relative-position projection (P - W_R), point estimates.
  R9  fixed-window design-level mean (DL over budgets with NF_T4's bootstrap variances).
  R10 an executed count that adds Transformer-XL relative-position attention (absent from Hoffmann's count).
[review] Added by the independent reviewer (output/memos/rb4_chinflop_review.md); they change no pre-existing output:
  R11 count-free N_F = C_b/(6 D_dig), read off the digitized coordinates as Llama 3's N = C/(6D) is (no architecture
      table): ra1's estimator and bootstrap; plus an errors-in-x simulation (T4 x plus noise of the size of the
      within-budget coordinate scatter), since under the T4 reading C_b/(6 D_dig) = N_F x (C_b/C_dig,i r_T4).
  R12 profile membership re-drawn on the T4-corrected coordinates (m1's rule, |dev - offset| <= 0.045 dex, applied to
      log10 C_dig + log10 r_T4): 147 runs instead of m1's 137; N_F and T.
  R13 the five highest-loss runs dropped (132 runs, Besiroglu et al.'s sample): they lie outside every window but
      enter the path-centre step, and in N_F they move two windows; N_F and T.
  R14 rb1's estimator-artifact check of the Chinchilla-Llama 3 drift (symwin.py: count-symmetric windows, E-fitted
      power frontier) with Chinchilla in N_F (T4); the T row reproduces rb1_sigmaC_driftmc_symwin.csv.
  Specification grid: ra1's 12 specifications (orders, windows, frontier smoothers, centring; ra1's B_SENS = 199 and
      ra1's per-specification seeds) in T and in N_F (T4), on 137 and 132 runs -> rb4_chinflop_specgrid.csv. The T,
      137-run cells reproduce ra1_modelfree_isoflop_sens.csv (check row R1 ra1 grid).
"""
from __future__ import annotations

import hashlib
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rb4common as cm  # noqa: E402


def _load(name):
    with open(os.path.join(cm.PROC, f"stage_{name}.pkl"), "rb") as f:
        return pickle.load(f)


def run(log=cm.log):
    import rb4_arch as ar
    import rb4_estim as es
    from meta import meta as dl
    rows = []

    def add(check, stat, value, target=np.nan, ok=None, note=""):
        rows.append(dict(check=check, statistic=stat, value=value, target=target,
                         ok=bool(ok) if ok is not None else (bool(abs(value - target) < 1e-9) if np.isfinite(target) else None),
                         note=note))
    A, E, PP, G = _load("arch"), _load("estimate"), _load("param"), _load("propagate")
    P, REF = PP["same_run"], PP["reference"]
    out = E["out"]
    # R1
    ra1b = pd.read_csv(os.path.join(cm.UP_TABLES, "ra1_modelfree_isoflop_budgets.csv"))
    ra1b = ra1b[ra1b.design == "Chinchilla"].reset_index(drop=True)
    tb = pd.DataFrame(out["T_ra1"]["rows"])
    for c in ("sigma", "se_sigma", "lo_sigma", "hi_sigma", "Mstar", "curv", "slope"):
        add("R1 ra1 budgets", f"max |diff| {c}", float(np.abs(tb[c].values - ra1b[c].values).max()), 0.0,
            ok=float(np.abs(tb[c].values - ra1b[c].values).max()) < 1e-9)
    ra1s = pd.read_csv(os.path.join(cm.UP_TABLES, "ra1_modelfree_isoflop_summary.csv")).set_index("design").loc["Chinchilla"]
    for c in ("sigma_re", "se_sigma_re", "sigma_fe", "se_sigma_fe", "drift_sigma_per_decade", "se_drift_sigma", "Q"):
        v = float(out["T_ra1"]["summary"][c])
        add("R1 ra1 summary", c, v, float(ra1s[c]), ok=abs(v - float(ra1s[c])) < 1e-9)
    rp = pd.read_csv(os.path.join(cm.UP_TABLES, "ra1_modelfree_isoflop_param.csv")).set_index("design").loc["Chinchilla"]
    pt = P.set_index("convention").loc["T (ra1 reproduction)"]
    for c in ("sigma_chin", "se_sigma_chin", "sigma_kappa", "se_sigma_kappa", "kappa"):
        add("R1 ra1 parametric", c, float(pt[c]), float(rp[c]), ok=abs(float(pt[c]) - float(rp[c])) < 1e-6)
    m2 = pd.read_csv(os.path.join(cm.UP_TABLES, "m2_table3_technology.csv"))
    m2 = m2[(m2.key == "chinchilla|all") & (m2.estimator == "huber")].iloc[0]
    rt = REF.set_index("convention").loc["T (m2 reference reproduction)"]
    add("R1 m2 reference technology", "kappa = 1 sigma* (240 runs)", float(rt.sigma_chin), float(m2.sigma_star), ok=abs(rt.sigma_chin - m2.sigma_star) < 5e-4)
    add("R1 m2 reference technology", "kappa-free sigma* (240 runs)", float(rt.sigma_kappa), float(m2.sigma_star_q), ok=abs(rt.sigma_kappa - m2.sigma_star_q) < 5e-4)
    add("R1 m2 reference technology", "M*(1e23), kappa = 1", float(rt["Mstar_1e+23_chin"]), float(m2.Mstar_1e23), ok=abs(rt["Mstar_1e+23_chin"] / m2.Mstar_1e23 - 1) < 0.01)
    # R2
    # [rb4 integration, 2026-09-24] rb1's primary outputs now carry the rebuilt Chinchilla (RB1_CHINCHILLA = 'rb4').
    # The 'old' run is validated against rb1's Chinchilla-in-T sensitivity outputs (RB1_CHINCHILLA = 'ra1', written to
    # output/sensitivity/rb1_chinchilla_T/), which reproduce rb1 as first published; R2b checks the 'new' run against
    # rb1's primary outputs.
    rb1_T = os.path.join(cm.ROOT, "output", "sensitivity", "rb1_chinchilla_T", "output", "tables")
    rb1_old_dir = rb1_T if os.path.exists(os.path.join(rb1_T, "rb1_sigmaC_study_level.csv")) else cm.UP_TABLES
    old = pd.read_csv(os.path.join(rb1_old_dir, "rb1_sigmaC_study_level.csv"))
    V = G["old"]["V"]
    m = V.merge(old, on="variant", suffixes=("", "_rb1"))
    add("R2 rb1 study level", f"rows matched ({len(m)} of {len(old)})", float(len(m)), float(len(old)))
    for c in ("mean_re", "lo_hksj", "hi_hksj", "tau", "Q"):
        rel = float(np.abs((m[c] - m[c + "_rb1"]) / np.maximum(np.abs(m[c + "_rb1"]), 1e-12)).max())
        add("R2 rb1 study level", f"max relative |diff| {c}", rel, 0.0, ok=rel < 1e-5, note="rb1 CSV written with %.6g")
    so = pd.read_csv(os.path.join(rb1_old_dir, "rb1_sigmaC_metareg_slopes.csv"))
    mm = G["old"]["M"]["slopes"].merge(so, on=["sample", "spec_key"], suffixes=("", "_rb1"))
    add("R2 rb1 meta-regression", f"rows matched ({len(mm)} of {len(so)})", float(len(mm)), float(len(so)))
    for c in ("slope", "se_cr2_design", "p_cr2_design", "p_wcr_design"):
        dmax = float(np.nanmax(np.abs((mm[c] - mm[c + "_rb1"]) / np.maximum(np.abs(mm[c + "_rb1"]), 1e-12))))
        add("R2 rb1 meta-regression", f"max relative |diff| {c}", dmax, 0.0, ok=dmax < 1e-5, note="rb1 CSV written with %.6g")
    for nm, key in (("rb1_sigmaC_pi_sigmaC.csv", "pi"), ("rb1_sigmaC_wedge_scenarios.csv", "scen")):
        ref = pd.read_csv(os.path.join(rb1_old_dir, nm))
        mine = G["old"]["M"][key].drop(columns=["run"])
        num = ref.select_dtypes("number").columns
        dmax = float(np.nanmax(np.abs((mine[num].values - ref[num].values) / np.maximum(np.abs(ref[num].values), 1e-12))))
        add("R2 rb1 downstream (pid)", f"max relative |diff| vs {nm}", dmax, 0.0, ok=dmax < 1e-5, note="rb1 CSV written with %.6g")
    # R2b [rb4 integration]: rb1's primary outputs (RB1_CHINCHILLA = 'rb4') against this module's 'new' run
    if rb1_old_dir != cm.UP_TABLES:
        new_ = pd.read_csv(os.path.join(cm.UP_TABLES, "rb1_sigmaC_study_level.csv"))
        Vn = G["new"]["V"].copy()

        def _key(v):      # common key: rb1 relabels the Meta-only eta rows and rb4's Chinchilla-count rows
            v = v.str.replace(r"^\[rb4\] FLOP-accounting elasticity eta = ([+-]0\.1) for Meta only \(Chinchilla rebuilt\)$",
                              r"eta \1 Meta only", regex=True)
            v = v.str.replace(r"^FLOP-accounting elasticity eta = ([+-]0\.1) for Meta only \(unobserved; .*$",
                              r"eta \1 Meta only", regex=True)
            v = v.str.replace(r"^\[rb4\] Chinchilla in accounting (\S+): .*$", r"chin \1", regex=True)
            return v.str.replace(r"^\[rb4\] Chinchilla accounting variant (\S+): .*$", r"chin \1", regex=True)
        Vn["key"] = _key(Vn["variant"])
        new_["key"] = _key(new_["variant"])
        m = Vn.merge(new_, on="key", suffixes=("", "_rb1"))
        add("R2b rb1 primary study level = run 'new'", f"rows matched ({len(m)} of {len(Vn)} 'new' rows)",
            float(len(m)), float(len(Vn) - 2), note="rb1 drops the eta rows for Chinchilla and Meta (Chinchilla observed)")
        for c in ("mean_re", "lo_hksj", "hi_hksj", "se_hksj", "tau", "Q", "p_Q"):
            rel = float(np.abs((m[c] - m[c + "_rb1"]) / np.maximum(np.abs(m[c + "_rb1"]), 1e-12)).max())
            add("R2b rb1 primary study level = run 'new'", f"max relative |diff| {c}", rel, 0.0, ok=rel < 1e-5)
        sn = pd.read_csv(os.path.join(cm.UP_TABLES, "rb1_sigmaC_metareg_slopes.csv"))
        mm = G["new"]["M"]["slopes"].merge(sn, on=["sample", "spec_key"], suffixes=("", "_rb1"))
        add("R2b rb1 primary meta-regression = run 'new'", f"rows matched ({len(mm)} of {len(sn)})", float(len(mm)), float(len(sn)))
        for c in ("slope", "se_cr2_design", "p_cr2_design", "p_wcr_design"):
            dmax = float(np.nanmax(np.abs((mm[c] - mm[c + "_rb1"]) / np.maximum(np.abs(mm[c + "_rb1"]), 1e-12))))
            add("R2b rb1 primary meta-regression = run 'new'", f"max relative |diff| {c}", dmax, 0.0, ok=dmax < 1e-5)
    # R3
    for tag in ("new",):
        b = pd.read_csv(os.path.join(cm.PROC, f"override_{tag}_isoflop_budgets.csv"))
        s = pd.read_csv(os.path.join(cm.PROC, f"override_{tag}_isoflop_summary.csv"))
        b0 = pd.read_csv(os.path.join(cm.UP_TABLES, "ra1_modelfree_isoflop_budgets.csv"))
        s0 = pd.read_csv(os.path.join(cm.UP_TABLES, "ra1_modelfree_isoflop_summary.csv"))
        nb = b[b.design != "Chinchilla"].reset_index(drop=True)
        ob = b0[b0.design != "Chinchilla"].reset_index(drop=True)
        same = nb.equals(ob) or bool(np.allclose(nb.select_dtypes("number").values, ob.select_dtypes("number").values, equal_nan=True))
        add("R3 override integrity", "non-Chinchilla budget rows unchanged", float(same), 1.0, ok=same)
        ns = s[s.design != "Chinchilla"].reset_index(drop=True).select_dtypes("number")
        os_ = s0[s0.design != "Chinchilla"].reset_index(drop=True).select_dtypes("number")
        same2 = bool(np.allclose(ns.values, os_.values, equal_nan=True))
        add("R3 override integrity", "non-Chinchilla summary rows unchanged", float(same2), 1.0, ok=same2)
        add("R3 override integrity", "Chinchilla budgets in override", float((b.design == "Chinchilla").sum()), 9.0)
    # R4
    a4s = A["a4s"].set_index(["accounting", "N_in_6ND"])
    for k in ("T4", "A", "X"):
        r = a4s.loc[(k, "architecture T")]
        add("R4 Table A4", f"{k}: ratios exact to 2 dp (of 6)", float(r.exact_2dp), 6.0 if k == "T4" else np.nan,
            ok=(r.exact_2dp == 6) if k == "T4" else None, note=f"max |err| {r.max_abs_err:.3f}")
    add("R4 Table A9", "max |T_arch/N_reported - 1|", float(np.abs(A["A"].T_err_rel).max()), ok=float(np.abs(A["A"].T_err_rel).max()) < 0.01)
    oth = A["other"].set_index("model")
    add("R4 totals", "Gopher: C under T4 (Hoffmann quote 6.3e23; Rae et al. 5.76e23)", float(oth.loc["Gopher 280B", "C_T4"]),
        ok=None, note="neither 6.3e23 nor 5.76e23 is reproduced by any Appendix F count (A: %.3g)" % oth.loc["Gopher 280B", "C_A"])
    # R5 eta approximation (fixed windows)
    fw = E["fixedwin"]
    sT = fw[fw.accounting == "ra1"].set_index("budget_C").S
    s4 = fw[fw.accounting == "T4"].set_index("budget_C").S
    eta = E["eta"][E["eta"].accounting == "T4"].set_index("budget_C").eta_window
    ratio = (s4 / sT).values
    pred = 1.0 / (1.0 + eta.loc[s4.index].values) ** 2
    add("R5 eta approximation", "max |S_T4/S_T - (1+eta)^-2| over budgets", float(np.abs(ratio - pred).max()), ok=None,
        note="ratios " + ", ".join(f"{a:.3f}" for a in ratio) + "; predicted " + ", ".join(f"{b:.3f}" for b in pred))
    # R6 slope vs kappa corrections
    for key in ("NF_A|corr", "NF_X|corr", "T|corr", "P|corr", "NF_T4|6ND", "NF_A|6ND"):
        a = np.mean([r["sigma"] for r in out[key]["rows"]])
        b = np.mean([r["sigma"] for r in out[key + "|slope"]["rows"]]) if key + "|slope" in out else np.nan
        add("R6 correction", f"{key}: plain mean sigma*_b, kappa surface vs first-order slope", float(a - b), ok=None,
            note=f"kappa {a:.4f}; slope {b:.4f}")
    # R7 coordinate test robustness
    df, Aa = es.load_runs()
    iso = df[df.iso]
    high = iso.L.nlargest(5).index
    co5 = es.coords_test(df.drop(index=high))
    for _, r in co5.iterrows():
        add("R7 coordinates, 132 runs", f"{r.accounting}: within-budget slope (CR1 s.e.)", float(r.slope_resid_on_log10N),
            ok=None, note=f"se {r.se_cr1_budget:.4f}, p {r.p_cr1:.4f}, offset {r.mean_offset_dex:+.4f}")
    for k in ("T4", "6T"):
        lr = np.log10(iso[f"F_{k}"].values / (6 * iso["T_arch"].values))
        e = iso["dev_raw"].values + lr
        signs = []
        for bgt, g in pd.DataFrame(dict(e=e, l=np.log10(iso.N.values), b=iso.budget.values)).groupby("b"):
            signs.append(np.sign(np.polyfit(g.l, g.e, 1)[0]))
        add("R7 coordinates by budget", f"{k}: budgets with positive within-budget slope (of 9)", float(np.sum(np.array(signs) > 0)), ok=None)
    # R8 P without W_R (point estimates, ra1 estimator)
    import isoflop as iso_
    iso2 = df[df.iso].reset_index(drop=True).copy()
    iso2["C_b"] = iso2["budget"]
    x = np.log(Aa["P_noWR"].values[iso2["row"].values])
    d = es.design_df(iso2.assign(iso=True), x)
    res = iso_.design_estimate(d, 2, 1.0, "auto", "path")
    add("R8 P without W_R", "plain mean sigma*_b", float(np.mean(res["sigma"])), ok=None,
        note="budgets " + ", ".join(f"{v:.3f}" for v in res["sigma"]) + f"; P (with W_R) plain mean {np.mean([r['sigma'] for r in out['P']['rows']]):.4f}")
    # R9 fixed-window design-level mean
    bb = pd.DataFrame(out["NF_T4"]["rows"]).set_index("budget_C")
    for acc, key in (("T4", "NF_T4"), ("ra1", "T_ra1")):
        f = fw[fw.accounting == acc].set_index("budget_C")
        v = (pd.DataFrame(out[key]["rows"]).set_index("budget_C").se_sigma.loc[f.index] * 2 / f.sigma ** 2) ** 2   # s.e. of S
        r = dl(f.S.values, np.sqrt(v.values))
        add("R9 fixed windows", f"{acc}: DL mean over budgets (sigma scale), current windows held fixed", float(2 / (2 + r["mu_re"])), ok=None)
    for key in ("NF_T4", "T_ra1"):
        add("R9 fixed windows", f"{key}: plain mean sigma*_b, own windows", float(np.mean([r["sigma"] for r in out[key]["rows"]])), ok=None)
    for acc in ("T4", "ra1"):
        add("R9 fixed windows", f"{acc}: plain mean sigma*_b, current windows", float(fw[fw.accounting == acc].sigma.mean()), ok=None)
    # R10 executed count with Transformer-XL relative-position attention (not in Hoffmann's count): per layer and token,
    # forward, position logits 2 S d_kv h plus the projection of the position embeddings 2 d d_kv h (no memory)
    S_ = 2048
    kvh = Aa["kv_size"].values * Aa["n_heads"].values
    Fxr = Aa["F_X"].values + 3.0 * Aa["n_layers"].values * (2 * S_ * kvh + 2 * Aa["d_model"].values * kvh)
    iso2["F_Xr"] = Fxr[iso2["row"].values]
    r_prof = (Fxr / (6 * Aa["T"].values))[sorted(set(iso2["row"]))]
    d = es.design_df(iso2.assign(iso=True), np.log(iso2["F_Xr"].values / 6.0))
    res = iso_.design_estimate(d, 2, 1.0, "auto", "path")
    add("R10 executed count with relative-position attention", "mechanical: plain mean sigma*_b", float(np.mean(res["sigma"])), ok=None,
        note=f"F/(6T) over profile models {r_prof.min():.2f}-{r_prof.max():.2f}; count X plain mean "
             f"{np.mean([r['sigma'] for r in out['NF_X']['rows']]):.4f}")
    surf, _ = es.fit_surface(iso2.assign(iso=True), "T4")
    Nn = iso2["T_arch"].values
    Dt = iso2["budget"].values / iso2["F_T4"].values
    rho = iso2["F_Xr"].values / iso2["F_T4"].values
    Lc = iso2["L"].values + surf(Nn, Dt / rho) - surf(Nn, Dt)
    d2 = es.design_df(iso2.assign(iso=True), np.log(iso2["F_Xr"].values / 6.0), L=Lc)
    res2 = iso_.design_estimate(d2, 2, 1.0, "auto", "path")
    add("R10 executed count with relative-position attention", "own isocosts (kappa surface): plain mean sigma*_b",
        float(np.mean(res2["sigma"])), ok=None,
        note=f"count X own isocosts plain mean {np.mean([r['sigma'] for r in out['NF_X|corr']['rows']]):.4f}")
    # [review] R11-R13 and the specification grid in N_F (reviewer additions; module docstring)
    rows += review_additions(df, Aa, out, log=log)
    R = pd.DataFrame(rows)
    cm.tab(R, "review_checks")
    nbad = int((R.ok == False).sum())  # noqa: E712
    log(f"  review checks: {len(R)} rows, {nbad} failed")
    return R


def _sub_design(sub, x, budget=None):
    """[review] ra1's layout (rb4_estim.design_df) for a run subset (kept in m1's order) and a budget assignment."""
    import rb4_estim as es
    d = sub.copy()
    if budget is not None:
        d["budget"] = np.asarray(budget, float)
    d["iso"] = True
    return es.design_df(d, np.asarray(x, float))


def review_additions(df, Aa, out, log=cm.log):
    """[review] R11 (count-free N_F), R12 (T4-consistent profile membership), R13 (five highest-loss runs dropped) and
    ra1's specification grid in T and N_F (module docstring). Bootstraps: ra1's isoflop._boot_job, unchanged, on at
    most cm.N_PROC processes; primary-type rows use ra1's Chinchilla seed and B = cm.B_ISO, grid cells ra1's B_SENS and
    ra1's per-specification seeds. Returns check rows; writes rb4_chinflop_specgrid.csv."""
    import isoflop as iso_
    rc = cm.rc
    rows = []

    def add(check, stat, value, note=""):
        rows.append(dict(check=check, statistic=stat, value=float(value), target=np.nan, ok=None, note=note))

    iso = df[df.iso]
    top5 = iso["rank_worst"] <= 5
    x_T = lambda s: np.log(s["N"].values)                       # noqa: E731  ra1's T (digitized N)
    x_F = lambda s: np.log(s["F_T4"].values / 6.0)              # noqa: E731  N_F, count T4
    # R12 membership on the T4-corrected coordinates (m1's rule: offset = median deviation within 0.1 dex of a budget)
    lb = np.log10(rc.CHIN_BUDGETS)
    lc4 = np.log10(df["C"].values) + np.log10(df["F_T4"].values / (6.0 * df["T_arch"].values))
    d0 = lc4 - lb[np.abs(lc4[:, None] - lb[None, :]).argmin(1)]
    off4 = float(np.median(d0[np.abs(d0) < 0.1]))
    j4 = np.abs((lc4 - off4)[:, None] - lb[None, :]).argmin(1)
    in4 = np.abs(lc4 - off4 - lb[j4]) <= 0.045
    mem4 = df[in4]
    D = {"R11 count-free": _sub_design(iso, np.log(iso["budget"].values / (6.0 * iso["D"].values))),
         "R12 N_F": _sub_design(mem4, x_F(mem4), rc.CHIN_BUDGETS[j4][in4]),
         "R12 T": _sub_design(mem4, x_T(mem4), rc.CHIN_BUDGETS[j4][in4]),
         "R13 N_F": _sub_design(iso[~top5], x_F(iso[~top5])),
         "R13 T": _sub_design(iso[~top5], x_T(iso[~top5]))}
    seed = rc.seed_of("iso|Chinchilla")
    jobs = [dict(name=k, order=2, h=1.0, fk="auto", cen="path", B=cm.B_ISO, seed=seed, shock=True, primary=False)
            for k in D]
    grid_specs = [(2, h, "auto", "path") for h in iso_.H_GRID] + [(3, h, "auto", "path") for h in (1.0, 1.5)]
    grid_specs += [(2, 1.0, fk, "path") for fk in ("logquad", "cubic", "power")] + [(2, 1.0, "auto", "iter")]
    for samp, sub in (("137", iso), ("132", iso[~top5])):
        for conv, xf in (("T", x_T), ("N_F (T4)", x_F)):
            key = f"grid|{conv}|{samp}"
            D[key] = _sub_design(sub, xf(sub))
            for o, h, fk, cen in grid_specs:
                jobs.append(dict(name=key, order=o, h=h, fk=fk, cen=cen, B=iso_.B_SENS, shock=True, primary=False,
                                 seed=rc.seed_of(f"sens|Chinchilla|{o}|{h}|{fk}|{cen}")))
    log(f"  [review] R11-R13 and ra1's specification grid in N_F: {len(jobs)} bootstrap jobs on {cm.N_PROC} processes")
    res = rc.pmap(iso_._boot_job, jobs, dict(D=D), procs=cm.N_PROC, chunksize=1)
    S = {}
    grid = []
    for r in res:
        j = r["job"]
        if j["name"].startswith("grid|"):
            _, conv, samp = j["name"].split("|")
            g = dict(sample=f"{samp} runs", convention=conv, order=j["order"], h=j["h"], fkind=j["fk"], center=j["cen"],
                     B=j["B"], ok=bool(r["ok"]), k_valid=r["s"]["k_valid"] if r["ok"] else r["k_valid"])
            if r["ok"]:
                g.update({c: r["s"][c] for c in ("sigma_re", "se_sigma_re", "sigma_fe", "se_sigma_fe", "sigma_median_budget",
                                                 "drift_sigma_per_decade", "p_drift")})
            grid.append(g)
        elif r["ok"]:
            S[j["name"]] = r["s"]
    Gd = pd.DataFrame(grid)
    spec = ["sample", "order", "h", "fkind", "center"]
    wide = Gd.pivot_table(index=spec, columns="convention", values="sigma_re")
    kk = Gd.pivot_table(index=spec, columns="convention", values="k_valid")
    # difference only where both conventions keep at least 8 of the 9 budgets (ra1 drops h = 0.6 in T, k = 7)
    diff = (wide["N_F (T4)"] - wide["T"]).where((kk["N_F (T4)"] >= 8) & (kk["T"] >= 8)).rename("diff_NF_minus_T")
    Gd = Gd.merge(diff.reset_index(), on=["sample", "order", "h", "fkind", "center"], how="left")
    cm.tab(Gd, "specgrid")
    # ra1 reproduction of the T grid (137 runs)
    ra1s = pd.read_csv(os.path.join(cm.UP_TABLES, "ra1_modelfree_isoflop_sens.csv"))
    ra1s = ra1s[ra1s.design == "Chinchilla"].copy()
    ra1s["fkind"] = ra1s["fkind"].replace({"logcubic": "auto"})
    mine = Gd[(Gd["sample"] == "137 runs") & (Gd.convention == "T")]
    mm = mine.merge(ra1s, on=["order", "h", "fkind", "center"], suffixes=("", "_ra1"))
    dmax = float(np.abs(mm.sigma_re - mm.sigma_re_ra1).max())
    rows.append(dict(check="R1 ra1 grid", statistic=f"T, 137 runs: max |diff| sigma_re vs ra1 ({len(mm)} of 12 specs)",
                     value=dmax, target=0.0, ok=bool(dmax < 1e-9 and len(mm) == 12), note="ra1_modelfree_isoflop_sens.csv"))
    cur = out["NF_T4"]["summary"], out["T_ra1"]["summary"]

    def fmt(s):
        return f"RE {s['sigma_re']:.4f} ({s['se_sigma_re']:.4f}); FE {s['sigma_fe']:.4f}; drift {s['drift_sigma_per_decade']:+.4f}; Q {s['Q']:.1f} (boot p {s['p_Q_boot']:.3f})"
    # R11
    s11 = S["R11 count-free"]
    add("R11 count-free N_F = C_b/(6 D_dig)", "RE sigma* (137 runs, ra1's estimator and seed)", s11["sigma_re"],
        note=fmt(s11) + f"; N_F (T4) primary {cur[0]['sigma_re']:.4f}")
    e_ln = (df["dev_raw"].values * np.log(10) + np.log(df["F_T4"].values / (6 * df["T_arch"].values)))[df.iso.values]
    e_w = e_ln - pd.Series(e_ln).groupby(iso["budget"].values).transform("mean").values
    s_u = float(np.sqrt(np.sum(e_w ** 2) / (len(e_w) - iso["budget"].nunique())))
    dT4 = _sub_design(iso, x_F(iso))
    base_pm = float(np.mean(iso_.design_estimate(dT4, 2, 1.0, "auto", "path")["sigma"]))
    cf_pm = float(np.mean(iso_.design_estimate(D["R11 count-free"], 2, 1.0, "auto", "path")["sigma"]))
    rng = np.random.default_rng(cm.seed_of("review|eiv"))
    sims = []
    for _ in range(200):
        dd = dT4.copy()
        dd["x"] = dd["x"].values + rng.normal(0.0, s_u, len(dd))
        dd = dd.sort_values(["b", "x"]).reset_index(drop=True)
        rr = iso_.design_estimate(dd, 2, 1.0, "auto", "path")
        if rr.get("ok") and len(rr["valid"]) == 9:
            sims.append(float(np.mean(rr["sigma"])))
    sims = np.array(sims)
    add("R11 count-free N_F = C_b/(6 D_dig)", "errors-in-x simulation: mean plain-mean sigma*_b, T4 x + N(0, s_u)",
        float(sims.mean()), note=f"s_u {s_u:.4f} ln (within-budget s.d. of the T4 coordinate residual); {len(sims)} draws, "
                                 f"s.d. {sims.std(ddof=1):.4f}; plain means: T4 {base_pm:.4f}, count-free {cf_pm:.4f}; "
                                 f"share of draws >= count-free {np.mean(sims >= cf_pm):.3f}")
    # R12
    n_in, n_out = int((in4 & ~df.iso.values).sum()), int((~in4 & df.iso.values).sum())
    for k, cv in (("R12 N_F", 0), ("R12 T", 1)):
        s = S[k]
        add("R12 T4-consistent profile membership", f"RE sigma*, {k[4:]} ({int(in4.sum())} runs)", s["sigma_re"],
            note=fmt(s) + f"; offset {off4:+.4f} dex; {n_in} runs enter, {n_out} leave; 137-run value {cur[cv]['sigma_re']:.4f}")
    # R13
    for k, cv in (("R13 N_F", 0), ("R13 T", 1)):
        s = S[k]
        add("R13 five highest-loss runs dropped", f"RE sigma*, {k[4:]} (132 runs)", s["sigma_re"],
            note=fmt(s) + f"; 137-run value {cur[cv]['sigma_re']:.4f}")
    # R14 rb1's estimator-artifact check of the drift (symwin: count-symmetric windows, E-fitted power frontier), with
    # Chinchilla's design in N_F (T4); rb1's code unchanged, ra1's design loader patched for the call and restored
    import rb4_estim as es
    import symwin
    orig = rc.isoflop_designs

    def _nf_designs(*a, **k):
        Dd = orig(*a, **k)
        Dd["Chinchilla"] = es.design_df(df, np.log(df["F_T4"].values / 6.0))
        return Dd
    sw = {}
    try:
        for lab, fn in (("T", orig), ("N_F (T4)", _nf_designs)):
            symwin.rc.isoflop_designs = fn
            sw[lab] = symwin.run(log=lambda *a: None)
    finally:
        symwin.rc.isoflop_designs = orig
    ref = pd.read_csv(os.path.join(cm.UP_TABLES, "rb1_sigmaC_driftmc_symwin.csv"))
    mm = sw["T"].merge(ref, on=["design", "variant"], suffixes=("", "_rb1"))
    dmax = float(np.abs(mm.drift_ols - mm.drift_ols_rb1).max())
    rows.append(dict(check="R14 rb1 symwin", statistic=f"T: max |diff| drift vs rb1 ({len(mm)} rows)", value=dmax, target=0.0,
                     ok=bool(dmax < 1e-5), note="rb1_sigmaC_driftmc_symwin.csv (%.6g)"))
    for lab, t in sw.items():
        t = t[t.design == "Chinchilla + Llama 3, design FE (OLS)"].set_index("variant")
        add("R14 rb1 symwin, Chinchilla + Llama 3 (design FE, OLS)", f"{lab}: drift, primary windows", t.loc["primary", "drift_ols"],
            note="; ".join(f"{v} {t.loc[v, 'drift_ols']:+.4f} ({t.loc[v, 'se_ols']:.4f})" for v in ("symmetric", "power", "symmetric+power")))
    # grid summaries
    for samp in ("137 runs", "132 runs"):
        g = Gd[(Gd["sample"] == samp) & Gd.ok & (Gd.k_valid >= 8)]
        for conv in ("T", "N_F (T4)"):
            v = g[g.convention == conv]
            v2 = v[v.h != 0.6].sigma_re
            add("Grid (ra1's 12 specifications; k >= 8 budgets)", f"{conv}, {samp}: min RE sigma*", float(v.sigma_re.min()),
                note=f"max {v.sigma_re.max():.4f}; {len(v)} specifications; without h = 0.6: {v2.min():.4f}-{v2.max():.4f}")
        dd_ = g.drop_duplicates(["order", "h", "fkind", "center"]).diff_NF_minus_T.dropna()
        add("Grid (ra1's 12 specifications; k >= 8 budgets)", f"N_F minus T, {samp}: median", float(dd_.median()),
            note=f"range {dd_.min():+.4f} to {dd_.max():+.4f}; negative in {int((dd_ < 0).sum())} of {len(dd_)} "
                 f"(specifications with >= 8 budgets in both)")
    return rows


def compare(root_a, root_b):
    """Byte-for-byte comparison of every rb4 output (tables, override CSVs) between two output roots."""
    rows = []
    for sub in ("output/tables", "data/processed/rb4_chinflop"):
        da, db = os.path.join(root_a, sub), os.path.join(root_b, sub)
        for fn in sorted(os.listdir(da)):
            if not fn.startswith(("rb4_chinflop", "override_")) or fn.endswith((".pkl", ".txt")):
                continue
            pa, pb = os.path.join(da, fn), os.path.join(db, fn)
            ha = hashlib.sha256(open(pa, "rb").read()).hexdigest()
            hb = hashlib.sha256(open(pb, "rb").read()).hexdigest() if os.path.exists(pb) else "missing"
            rows.append(dict(file=f"{sub}/{fn}", identical=ha == hb))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--compare":
        C = compare(sys.argv[2], sys.argv[3])
        print(C.to_string())
        print(f"{int(C.identical.sum())} of {len(C)} files identical")
    else:
        run()
