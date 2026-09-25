"""study.py -- study-level variants of sigma* (T1.4 budgets <= 3e20; T1.5 Chinchilla's specification grid propagated;
T1.6 Farseer's bandwidth range as uncertainty; T1.10 the within-Marin estimator gap).

The study-level summary is rb1's: one estimate per study (Hoffmann, Meta, Marin as one study, Farseer; k = 4),
DerSimonian-Laird mean with the modified Hartung-Knapp-Sidik-Jonkman interval (t_{k-1}; ra1's meta.meta, unchanged).
Marin = equal-weight mean of its three corpora with the standard error of perfectly correlated errors (rho = 1);
Farseer = its local first-derivative path estimate 1/(1+b1) with s.e. max(its wild cluster s.e., the path-mean s.e.).
The primary row reproduces rb1_sigmaC_study_level.csv (0.687 [0.640, 0.734], Q = 8.25).

T1.4 (R4 N3(f), R1 New 2(b)). Each IsoFLOP design's random-effects mean over its budgets at or below 3e20 FLOP, computed
as ra1 computes the design mean (DerSimonian-Laird on S_b = 2(1/sigma*_b - 1) with bootstrap variances rse(S*_b)^2,
from the joint draws of jointboot.py, which reproduce ra1's exactly); Marin's budgets all lie at or below 3e20; Farseer's
first-derivative estimator restricted to its path points at or below 3e20 (local-only bootstrap of convexity.py, ra1's
seeds). The frontier derivatives are the full-design ones, as in the meta-regression subsample 'budgets <= 3e20'.

T1.5 (R4 N3 request 3; audit numbers_technology #4). Every N_F row of rb4_chinflop_specgrid.csv (windows 0.6-1.5 and a
global quadratic, cubic windows, E-free frontier smoothers, iterated centring; the 137 profile runs and the 132 runs
without the five highest-loss runs) and rb4's review rows R11-R13 replace Chinchilla's primary estimate; the other
three studies stay at their primary values.

T1.6 (R1 minor 3). Farseer's Hessian-based path mean moves between 0.664 and 0.727 across the bandwidth variants of
ra1 (ra1_modelfree_farseer_sensitivity.csv). Treated as uncertainty: Farseer enters at the midpoint of that range with
variance se^2 + (range)^2/12 (a uniform distribution over the range added to sampling error); variants centre at the
primary bandwidth's path mean or at the first-derivative estimate, and use a +-1.96 s.e. reading of the range.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import sigcommon as cm

rc = cm.rc
CUT = 3e20 * (1 + 1e-9)


def _dl(y, se):
    from meta import meta
    return meta(np.asarray(y, float), np.asarray(se, float))


def _row(label, entries, group, note=""):
    y = [e[1] for e in entries]
    s = [e[2] for e in entries]
    m = _dl(y, s)
    # [review] random-effects weight share of Farseer's entry (with tau = 0 it can dominate: T1.4)
    wre = 1.0 / (np.asarray(s, float) ** 2 + m["tau"] ** 2)
    far = np.array([str(e[0]).startswith("Farseer") for e in entries])
    return dict(group=group, variant=label, k=len(y), members="; ".join(f"{n} {a:.3f} ({b:.3f})" for n, a, b in entries),
                mean_re=m["mu_re"], lo_hksj=m["lo_hksj"], hi_hksj=m["hi_hksj"], se_hksj=m["se_hksj"], tau=m["tau"],
                Q=m["Q"], p_Q=m["p_Q"], I2=m["I2"], mean_fixed=m["mu_fixed"], note=note,
                weight_share_farseer=float(wre[far].sum() / wre.sum()))


def primary_entries():
    """(name, sigma, se) for the four studies, as in rb1's primary row."""
    s = pd.read_csv(cm.RB4_SUMMARY).set_index("design")
    slope = pd.read_csv(cm.RA1_FAR_SLOPE)
    pool = pd.read_csv(cm.RA1_FAR_POOL).set_index("conv")
    mar = s.loc[["Marin, Comma", "Marin, DCLM", "Marin, Nemotron-CC"], ["sigma_re", "se_sigma_re"]].values
    fd = slope[(slope.conv == "ne") & slope.variant.str.startswith("primary (wild")].iloc[0]
    far = (float(fd.sigma_slope), max(float(fd.se), float(pool.loc["ne", "se"])))
    return [("Hoffmann", float(s.loc["Chinchilla", "sigma_re"]), float(s.loc["Chinchilla", "se_sigma_re"])),
            ("Meta", float(s.loc["Llama 3", "sigma_re"]), float(s.loc["Llama 3", "se_sigma_re"])),
            ("Marin", float(mar[:, 0].mean()), float(mar[:, 1].mean())),
            ("Farseer", *far)]


# ============================================================================ T1.4: budgets <= 3e20
def design_re_subset(r, cut=CUT):
    """ra1's design random-effects mean restricted to budgets <= cut, from the joint draws (jointboot.run_draws)."""
    from isoflop import dersimonian_laird
    b = np.array(r["budgets"])
    m = b <= cut
    S, Sd = r["S"][m], r["Sd"][:, m]
    vS = np.array([rc.rse(Sd[:, j]) ** 2 for j in range(m.sum())])
    dl = dersimonian_laird(S, vS)
    sig = float(cm.sig_from_S(dl["mu"]))
    se = float(abs(cm.dsig_dS(dl["mu"])) * dl["se"])
    return dict(sigma=sig, se=se, k=int(m.sum()), tau_sigma=float(abs(cm.dsig_dS(dl["mu"])) * dl["tau"]),
                Cmax=float(b[m].max()), Cmin=float(b[m].min()), Q=dl["Q"], p_Q=dl["p_Q"])


def upto_3e20(R, far_restricted):
    """T1.4 rows. far_restricted: dict from convexity.farseer_restricted (first-derivative and Hessian path at <= 3e20)."""
    rows, parts = [], []
    ent = {}
    for des in cm.DESIGNS_ISO:
        full = design_re_subset(R[des], cut=np.inf)
        sub = design_re_subset(R[des])
        parts.append(dict(study=cm.STUDY[des], design=des, sigma_all=full["sigma"], se_all=full["se"], k_all=full["k"],
                          sigma_le3e20=sub["sigma"], se_le3e20=sub["se"], k_le3e20=sub["k"], Cmax_le3e20=sub["Cmax"],
                          tau_le3e20=sub["tau_sigma"]))
        ent[des] = (sub["sigma"], sub["se"])
    P = pd.DataFrame(parts)
    mar = np.array([ent[m] for m in ("Marin, Comma", "Marin, DCLM", "Marin, Nemotron-CC")])
    fd = far_restricted["fd_le3e20"]
    hs = far_restricted["hessian_re_le3e20"]
    base = [("Hoffmann <=3e20", *ent["Chinchilla"]), ("Meta <=3e20", *ent["Llama 3"]),
            ("Marin", float(mar[:, 0].mean()), float(mar[:, 1].mean())), ("Farseer FD <=3e20", fd["sigma"], fd["se"])]
    rows.append(_row("Study-level mean, budgets <= 3e20 FLOP (primary: Farseer first-derivative estimate on its path "
                     "points <= 3e20)", base, "T1.4"))
    rows.append(_row("Study-level mean, budgets <= 3e20; Farseer Hessian path, RE over its levels <= 3e20",
                     base[:3] + [("Farseer Hessian RE <=3e20", hs["sigma"], hs["se"])], "T1.4"))
    rows.append(_row("Study-level mean, budgets <= 3e20; Farseer at its full-path first-derivative estimate",
                     base[:3] + [primary_entries()[3]], "T1.4", "only Chinchilla and Llama 3 are restricted"))
    rows.append(_row("Study-level mean, budgets <= 3e20; IsoFLOP studies only (k = 3)", base[:3], "T1.4"))
    # [review] with tau = 0 Farseer's first-derivative entry carries 84 percent of the weight, and its s.e. excludes
    # smoothing bias (R1 minor 3); the same treatment as T1.6: Farseer's bandwidth range as uniform uncertainty
    fs_ = pd.read_csv(cm.RA1_FAR_SENS)
    v_ = fs_[fs_.key == "path_sigma_mean_poolC|ne"].value
    se_bw = float(np.sqrt(fd["se"] ** 2 + (float(v_.max()) - float(v_.min())) ** 2 / 12.0))
    rows.append(_row("Study-level mean, budgets <= 3e20; Farseer first-derivative estimate with its bandwidth range as "
                     "uncertainty (T1.6 treatment) [review]", base[:3] + [("Farseer FD <=3e20, bandwidth", fd["sigma"], se_bw)],
                     "T1.4", note=f"Farseer s.e. {se_bw:.4f} = sqrt({fd['se']:.4f}^2 + range^2/12)"))
    # the same construction on all budgets reproduces the primary headline (check)
    allb = [("Hoffmann", *(parts[0]["sigma_all"], parts[0]["se_all"])), ("Meta", parts[1]["sigma_all"], parts[1]["se_all"]),
            ("Marin", float(P[P.study == "Marin"].sigma_all.mean()), float(P[P.study == "Marin"].se_all.mean())),
            primary_entries()[3]]
    rows.append(_row("Check: same construction on all budgets (= rb1's primary)", allb, "check"))
    return pd.DataFrame(rows), P


# ============================================================================ T1.5: Chinchilla's grid propagated
def chinchilla_grid():
    base = primary_entries()
    G = pd.read_csv(cm.RB4_SPECGRID)
    G = G[(G.convention == "N_F (T4)") & G.ok].copy()
    rows = []
    for _, g in G.iterrows():
        lab = (f"{g['sample']}: order {int(g.order)}, h = {g.h}, frontier {g.fkind}, centre {g.center}")
        r = _row(f"Chinchilla N_F grid: {lab}", [("Hoffmann", float(g.sigma_re), float(g.se_sigma_re))] + base[1:],
                 "T1.5 grid", note=f"k_valid = {int(g.k_valid)}")
        r.update(sample=g["sample"], order=int(g.order), h=float(g.h), fkind=g.fkind, center=g.center,
                 k_valid=int(g.k_valid), chin_sigma=float(g.sigma_re), chin_se=float(g.se_sigma_re))
        rows.append(r)
    import re
    ch = pd.read_csv(cm.RB4_CHECKS)
    for key, pre, stat in (("R11", "R11 count-free", "RE sigma*"), ("R12", "R12 T4-consistent", "RE sigma*, N_F"),
                           ("R13", "R13 five highest-loss", "RE sigma*, N_F")):
        c = ch[ch.check.str.startswith(pre) & ch.statistic.str.startswith(stat)].iloc[0]
        se_ = float(re.search(r"RE [0-9.]+ \(([0-9.]+)\)", c.note).group(1))
        r = _row(f"Chinchilla {key}: {c.check} ({c.statistic})", [("Hoffmann", float(c.value), se_)] + base[1:],
                 "T1.5 R11-R13")
        r.update(sample=key, k_valid=9, chin_sigma=float(c.value), chin_se=se_)
        rows.append(r)
    return pd.DataFrame(rows)


def ranges(grid, study_rb1, far_bw, upto):
    """Ranges of the study-level mean and Q across the variant families (T1.5, T3(a)-(b))."""
    out = []

    def add(lab, sub, col="mean_re"):
        out.append(dict(family=lab, n=len(sub), mean_min=float(sub.mean_re.min()), mean_max=float(sub.mean_re.max()),
                        Q_min=float(sub.Q.min()), Q_max=float(sub.Q.max()), p_Q_min=float(sub.p_Q.min()),
                        p_Q_max=float(sub.p_Q.max()), lo_min=float(sub.lo_hksj.min()), hi_max=float(sub.hi_hksj.max()),
                        argmin=str(sub.loc[sub.mean_re.idxmin(), "variant"]), argmax=str(sub.loc[sub.mean_re.idxmax(), "variant"])))
    g137 = grid[(grid["sample"] == "137 runs") & (grid.k_valid >= 8)]
    g132 = grid[(grid["sample"] == "132 runs") & (grid.k_valid >= 8)]
    rr = grid[grid.group == "T1.5 R11-R13"]
    add("Chinchilla windows/order/frontier/centring grid, 137 runs, N_F (k >= 8 budgets)", g137)
    add("Chinchilla grid, 132 runs (five highest-loss runs dropped), N_F (k >= 8 budgets)", g132)
    add("Chinchilla R11-R13 (count-free N_F, T4-consistent membership, 132 runs)", rr)
    add("Chinchilla windows, samples and profile memberships: both grids and R11-R13 (k >= 8)", pd.concat([g137, g132, rr]))
    add("Chinchilla grid incl. the h = 0.6 window with seven budgets (137 runs)", grid[grid["sample"] == "137 runs"])
    sr = study_rb1.copy()
    sr["group"] = "rb1"
    chin_rb1 = sr[sr.variant.str.startswith("[rb4] Chinchilla")]
    add("Chinchilla only: counts and conventions (rb4 rows), windows, samples and memberships (grids k >= 8, R11-R13)",
        pd.concat([chin_rb1, g137, g132, rr], ignore_index=True))
    add("rb1's published variant rows (accounting, conventions, aggregation, estimators)", sr)
    allv = pd.concat([sr, g137, g132, rr], ignore_index=True)
    add("All: rb1's rows, Chinchilla's grids (k >= 8) and R11-R13", allv)
    allv2 = pd.concat([allv, far_bw], ignore_index=True)
    add("All plus Farseer's bandwidth range as uncertainty (T1.6)", allv2)
    return pd.DataFrame(out)


# ============================================================================ T1.6: Farseer's bandwidth range
def farseer_bandwidth():
    base = primary_entries()
    fs = pd.read_csv(cm.RA1_FAR_SENS)
    v = fs[fs.key == "path_sigma_mean_poolC|ne"].copy()
    lo, hi = float(v.value.min()), float(v.value.max())
    prim = float(v[v.variant.str.startswith("primary")].value.iloc[0])
    se_path = float(pd.read_csv(cm.RA1_FAR_POOL).set_index("conv").loc["ne", "se"])
    fd = base[3]
    var_u = (hi - lo) ** 2 / 12.0
    rng95 = ((hi - lo) / (2 * 1.96)) ** 2
    rows = []
    for lab, centre, var_bw, cname in (
            ("midpoint of the bandwidth range, uniform", 0.5 * (lo + hi), var_u, "midpoint"),
            ("primary-bandwidth Hessian path mean, uniform", prim, var_u, "primary Hessian path mean"),
            ("first-derivative estimate, uniform", fd[1], var_u, "first-derivative 0.708"),
            ("midpoint, range read as +-1.96 s.e.", 0.5 * (lo + hi), rng95, "midpoint")):
        se_ = float(np.sqrt(se_path ** 2 + var_bw))
        r = _row(f"Farseer's bandwidth range [{lo:.3f}, {hi:.3f}] as uncertainty: {lab}",
                 base[:3] + [("Farseer bw", centre, se_)], "T1.6",
                 note=f"Farseer entry {centre:.4f} (s.e. {se_:.4f} = sqrt({se_path:.4f}^2 + bandwidth variance {var_bw:.6f}))")
        r.update(far_centre=centre, far_se=se_, bw_lo=lo, bw_hi=hi, bw_variants=int(len(v)))
        rows.append(r)
    return pd.DataFrame(rows), dict(lo=lo, hi=hi, primary=prim, se_path=se_path,
                                    variants="; ".join(f"{a}: {b:.4f}" for a, b in zip(v.variant, v.value)))


# ============================================================================ T1.10
def within_marin_gap():
    c = pd.read_csv(cm.RB1_EXTRAP_CONVEXITY)
    s = pd.read_csv(cm.RB4_SUMMARY).set_index("design")
    rows = []
    for m in ("Marin, Comma", "Marin, DCLM", "Marin, Nemotron-CC"):
        fdv = c[(c.design == m) & (c.param == "sigma_slope")].iloc[0]
        rows.append(dict(design=m, sigma_isoflop_curvature=float(s.loc[m, "sigma_re"]),
                         se_isoflop=float(s.loc[m, "se_sigma_re"]), sigma_local_first_derivative=float(fdv.est),
                         se_local_fd=float(fdv.se), gap=float(s.loc[m, "sigma_re"] - fdv.est)))
    return pd.DataFrame(rows)


def run(R, far_restricted, log=cm.log):
    base = primary_entries()
    prim = _row("PRIMARY (reproduces rb1): one estimate per study", base, "primary")
    rb1 = pd.read_csv(cm.RB1_STUDY)
    p0 = rb1[rb1.variant.str.startswith("PRIMARY")].iloc[0]
    log(f"  study-level primary {prim['mean_re']:.6f} [{prim['lo_hksj']:.6f}, {prim['hi_hksj']:.6f}] Q {prim['Q']:.4f} "
        f"(rb1: {p0.mean_re:.6f} [{p0.lo_hksj:.6f}, {p0.hi_hksj:.6f}] Q {p0.Q:.4f})")
    up, parts = upto_3e20(R, far_restricted)
    grid = chinchilla_grid()
    fbw, fbw_info = farseer_bandwidth()
    rg = ranges(grid, rb1, fbw, up)
    V = pd.concat([pd.DataFrame([prim]), up, grid, fbw], ignore_index=True)
    return dict(variants=V, parts=parts, ranges=rg, far_bw=fbw_info, marin_gap=within_marin_gap(),
                check=dict(mean=(prim["mean_re"], p0.mean_re), lo=(prim["lo_hksj"], p0.lo_hksj), hi=(prim["hi_hksj"], p0.hi_hksj),
                           Q=(prim["Q"], p0.Q)))
