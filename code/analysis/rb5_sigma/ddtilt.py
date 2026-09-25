"""ddtilt.py -- the DataDecide recipe tilt from final checkpoints only (T14; R2 minor 17, round-2 minor 18).

m2_techpanel's neutrality analysis fits the common-exponent (CE) panel model to all 21,888 DataDecide checkpoints with
D/N >= 5, most of them taken before the learning-rate schedule ends. Here the same model is fitted to the final
checkpoint of every run whose schedule is complete (last logged step at least 98 percent of the full schedule; the
truncated auxiliary seeds of the 530M, 750M and 1B models are dropped), so all observations lie on one ray, M = D/N of
about 100:
    ln L = ln( A_r N^-alpha + B_r D^-beta + E_r ),   recipe r = 1..25, common (alpha, beta).
On a ray D = M N the two power terms are separated only through alpha != beta, so the recipe tilt ln(A_r/B_r) is
identified weakly, if at all. The module reports (m2's estimators and bootstrap, imported unchanged):
  * the CE fit (NLS and Huber) and the tilt range max_r - min_r of ln(A_r/B_r), with a cluster (size x seed cell) pairs
    bootstrap percentile interval, as m2 does for all checkpoints;
  * the implied allowance on M*: tau = exp(2 range/(alpha+beta)) under the fit's own exponents and under the reference
    exponents (alpha + beta = 0.855; the paper's 1.84 and 3.4 are these two conversions of the all-checkpoint tilt);
  * the identification check: the profile of the NLS objective over (alpha, beta) (flat along a ridge = the split of
    the reducible loss between the two terms, and so the tilt, is not identified on the ray), and the tilt range along
    the ridge;
  * the Spearman correlation across recipes between the final-checkpoint and all-checkpoint tilts.
[review] (WP2-review, independent reviewer): on the ray the objective has (at least) two separated local optima that
exchange which power term decays fast: alpha about 0.9 > beta about 0.09 (the mode reached from m2's all-checkpoint
start, which the builder reported) and beta about 1.1-1.2 > alpha about 0.15, whose objective is LOWER for both NLS and
Huber and whose recipe tilts are ranked in the opposite order (Spearman about -0.75 against the all-checkpoint tilts).
Every fit now starts in both modes; both local optima are reported (column 'mode'; 'global_optimum' marks the lower
objective); the cluster bootstrap starts every draw in both modes and records the share of draws in each; the profile
grid covers both modes. The recipes' tilt is not identified on the ray: its sign pattern depends on an arbitrary
labelling of the two power terms.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import sigcommon as cm

B_DD = 49 if cm.QUICK else 199
FRAC_MIN = 0.98


def load_final():
    import m2_data as md
    dd = md.load_datadecide()
    last = dd.groupby("run_id").step.transform("max")
    f = dd[dd.step == last].copy()
    f["frac_final"] = f["step"] / f["size"].map(md.DD_LAST)
    keep = f[f.frac_final >= FRAC_MIN].reset_index(drop=True)
    info = dict(n_runs=int(len(f)), n_complete=int(len(keep)), n_truncated=int((f.frac_final < FRAC_MIN).sum()),
                M_min=float((keep.D / keep.N).min()), M_max=float((keep.D / keep.N).max()),
                M_median=float((keep.D / keep.N).median()), n_recipes=int(keep.recipe.nunique()),
                n_cells=int(keep.cell.nunique()), truncated=", ".join(sorted(f[f.frac_final < FRAC_MIN].run_id.str.split("|").str[1:].str.join("|").unique())))
    return keep, dd, info


def _arrays(df, groups):
    g = df.recipe.map({r: i for i, r in enumerate(groups)}).values
    return np.log(df.N.values), np.log(df.D.values), np.log(df.L.values), g


def start_from_m2(groups):
    """Starting value: m2's all-checkpoint CE (NLS) estimates."""
    mg = pd.read_csv(cm.M2_MAGS)
    r = mg[(mg.experiment == "datadecide") & (mg.estimator == "nls")].iloc[0]
    G = pd.read_csv(cm.M2_GROUPS_DD).set_index("group").loc[groups]
    return np.r_[r.alpha_CE, r.beta_CE, G.lnA.values, G.lnB.values, np.log(G.E.values)]


SWAP_AB = ((0.15, 1.0), (0.15, 1.2), (0.2, 0.8), (0.1, 1.5))    # [review] starts in the beta > alpha mode


def swapped_starts(groups, x, z):
    """[review] Starting values in the second mode (alpha small, beta large): m2's all-checkpoint levels shifted so
    that each power term keeps its value at the sample means of ln N and ln D."""
    mg = pd.read_csv(cm.M2_MAGS)
    r = mg[(mg.experiment == "datadecide") & (mg.estimator == "nls")].iloc[0]
    G = pd.read_csv(cm.M2_GROUPS_DD).set_index("group").loc[groups]
    out = []
    for a0, b0 in SWAP_AB:
        out.append(np.r_[a0, b0, G.lnA.values + (a0 - r.alpha_CE) * np.mean(x), G.lnB.values + (b0 - r.beta_CE) * np.mean(z),
                         np.log(G.E.values)])
    return out


def mode_of(th):
    return "alpha > beta" if th[0] > th[1] else "beta > alpha"


def fit_ce(x, z, y, g, R, est, starts):
    import m2_est as me
    th, f, _ = me.fit_panel_ls("CE", x, z, y, g, R, est, starts=starts)
    return th, f


def magnitudes(th, R):
    al, be = th[0], th[1]
    lnA, lnB = th[2:2 + R], th[2 + R:2 + 2 * R]
    tilt = lnA - lnB
    rngt = float(tilt.max() - tilt.min())
    return dict(alpha=float(al), beta=float(be), sigma_star=float(2 / (2 + al + be)), tilt_range=rngt,
                tau_own=float(np.exp(2 * rngt / (al + be))), tau_ref=float(np.exp(2 * rngt / (cm.REF_ALPHA + cm.REF_BETA))),
                tilt=tilt)


def _boot_one(idx, data, est, init, start2, init_other=None):
    """One cluster-bootstrap refit: best of the starts (the point estimate, m2's all-checkpoint estimate and, [review],
    the local optimum of the other mode), tight tolerances (a loose trust-region solve can stop at its start)."""
    import m2_est as me
    x, z, y, g, R = data["x"], data["z"], data["y"], data["g"], data["R"]
    starts = [init, start2] + ([init_other] if init_other is not None else [])
    th, f, _ = me.fit_panel_ls("CE", x[idx], z[idx], y[idx], g[idx], R, est, starts=starts, tight=True)
    return th


def profile(x, z, y, g, R, th0, alphas, betas, th_other=None):
    """NLS objective profiled over (alpha, beta): the 75 recipe levels re-fitted at each fixed pair. Start: the optimum's
    levels shifted so that each recipe's two power terms keep their values at the recipe's mean (ln N, ln D), and the
    previous grid point's solution; the better of the two is kept."""
    import m2_est as me
    from scipy.optimize import least_squares
    rows = []
    Z, _ = me.panel_design("CE", g, R)
    a0, b0 = th0[0], th0[1]
    xm = np.array([x[g == r].mean() for r in range(R)])
    zm = np.array([z[g == r].mean() for r in range(R)])
    ones = np.ones_like(y)
    for a in alphas:
        prev = None
        for b in betas:
            def fun(q):
                return me._panel_resjac(np.r_[a, b, q], Z, x, z, y, ones)[0]

            def jac(q):
                return me._panel_resjac(np.r_[a, b, q], Z, x, z, y, ones)[1][:, 2:]
            q0 = th0[2:].copy()
            q0[:R] += (a - a0) * xm
            q0[R:2 * R] += (b - b0) * zm
            qs = [q0]
            if th_other is not None:                   # [review] also start from the other mode's optimum
                q1 = th_other[2:].copy()
                q1[:R] += (a - th_other[0]) * xm
                q1[R:2 * R] += (b - th_other[1]) * zm
                qs.append(q1)
            best = None
            for q_start in qs + ([prev] if prev is not None else []):
                r = least_squares(fun, q_start, jac=jac, method="trf", x_scale="jac", ftol=1e-12, xtol=1e-12,
                                  gtol=1e-12, max_nfev=2000, tr_solver="exact")
                if best is None or r.cost < best.cost:
                    best = r
            prev = best.x
            tilt = best.x[:R] - best.x[R:2 * R]
            rows.append(dict(alpha=a, beta=b, ssr=2 * best.cost, tilt_range=float(tilt.max() - tilt.min()),
                             tau_own=float(np.exp(2 * (tilt.max() - tilt.min()) / (a + b))),
                             tau_ref=float(np.exp(2 * (tilt.max() - tilt.min()) / (cm.REF_ALPHA + cm.REF_BETA)))))
    return pd.DataFrame(rows)


def run(log=cm.log):
    import m2_est as me
    from scipy.stats import spearmanr
    keep, dd, info = load_final()
    groups = sorted(keep.recipe.unique())
    R = len(groups)
    x, z, y, g = _arrays(keep, groups)
    st = start_from_m2(groups)
    sw = swapped_starts(groups, x, z)
    fits = {}                                     # [review] (est, mode) -> (theta, objective)
    for est in ("nls", "huber"):
        thA, fA = fit_ce(x, z, y, g, R, est, [st] + ([fits[("nls", "alpha > beta")][0]] if ("nls", "alpha > beta") in fits else []))
        thB, fB = fit_ce(x, z, y, g, R, est, sw + ([fits[("nls", "beta > alpha")][0]] if ("nls", "beta > alpha") in fits else []))
        for th, f in ((thA, fA), (thB, fB)):
            fits[(est, mode_of(th))] = (th, f)
    log(f"  DataDecide final checkpoints: {info['n_complete']} complete runs of {info['n_runs']} ({R} recipes; "
        f"M {info['M_min']:.0f}-{info['M_max']:.0f})")
    rows, tilts = [], {}
    mg = pd.read_csv(cm.M2_MAGS)
    Gall = pd.read_csv(cm.M2_GROUPS_DD).set_index("group").loc[groups]
    data = dict(x=x, z=z, y=y, g=g, R=R)
    for est in ("nls", "huber"):
        modes = [m for m in ("alpha > beta", "beta > alpha") if (est, m) in fits]
        fmin = min(fits[(est, m)][1] for m in modes)
        glob = [m for m in modes if fits[(est, m)][1] == fmin][0]
        other = [m for m in modes if m != glob]
        draws = me.boot_indices(keep.cell.values, B_DD, cm.seed_of(f"dd|{est}"))
        bt = cm.pmap(_boot_one, draws, dict(data=data, est=est, init=fits[(est, glob)][0], start2=st,
                                            init_other=fits[(est, other[0])][0] if other else None), chunksize=10)
        bt = np.array([b for b in bt if not (isinstance(b, dict) and "_error" in b)])
        mb = [magnitudes(b, R) for b in bt]
        tr = np.array([q["tilt_range"] for q in mb])
        ab = np.array([q["alpha"] + q["beta"] for q in mb])
        to = np.array([q["tau_own"] for q in mb])
        trf = np.array([q["tau_ref"] for q in mb])
        share_b = float(np.mean(bt[:, 1] > bt[:, 0]))
        for mode in modes:
            th, f = fits[(est, mode)]
            m = magnitudes(th, R)
            tilts[(est, mode)] = m.pop("tilt")
            rho = spearmanr(tilts[(est, mode)], Gall.tilt_lnA_minus_lnB.values).correlation
            row = dict(sample=f"final checkpoints, complete schedules; mode {mode}", estimator=est, mode=mode,
                       global_optimum=bool(mode == glob), n=len(y), R=R, **m, objective=f,
                       objective_minus_global=f - fmin, alpha_plus_beta=m["alpha"] + m["beta"],
                       spearman_tilt_vs_allcheckpoints=rho)
            if mode == glob:          # the bootstrap refits start in both modes and keep the better optimum
                row.update(tilt_range_lo=float(np.percentile(tr, 2.5)), tilt_range_hi=float(np.percentile(tr, 97.5)),
                           alpha_plus_beta_lo=float(np.percentile(ab, 2.5)), alpha_plus_beta_hi=float(np.percentile(ab, 97.5)),
                           tau_own_lo=float(np.percentile(to, 2.5)), tau_own_hi=float(np.percentile(to, 97.5)),
                           tau_ref_lo=float(np.percentile(trf, 2.5)), tau_ref_hi=float(np.percentile(trf, 97.5)),
                           B=len(bt), share_draws_beta_gt_alpha=share_b)
            rows.append(row)
            log(f"    [{est}, mode {mode}{', global' if mode == glob else ''}] objective {f:.7g}; alpha {m['alpha']:.3f} "
                f"beta {m['beta']:.3f}; tilt range {m['tilt_range']:.3f}; tau own {m['tau_own']:.2f}, ref {m['tau_ref']:.2f}; "
                f"Spearman vs all-checkpoint tilts {rho:+.2f}")
        log(f"    [{est}] bootstrap (starts in both modes): share of draws with beta > alpha {share_b:.2f}; tau own "
            f"[{np.percentile(to, 2.5):.2f}, {np.percentile(to, 97.5):.2f}]")
        ref = mg[(mg.experiment == "datadecide") & (mg.estimator == est)].iloc[0]
        rows.append(dict(sample="all checkpoints with D/N >= 5 (m2, published)", estimator=est, n=int(len(dd)), R=R,
                         alpha=ref.alpha_CE, beta=ref.beta_CE, sigma_star=ref.sigma_star_CE, tilt_range=ref.tilt_range,
                         tau_own=float(np.exp(2 * ref.tilt_range / (ref.alpha_CE + ref.beta_CE))),
                         tau_ref=float(np.exp(2 * ref.tilt_range / (cm.REF_ALPHA + cm.REF_BETA))),
                         alpha_plus_beta=ref.alpha_CE + ref.beta_CE, tilt_range_lo=ref.tilt_range_lo,
                         tilt_range_hi=ref.tilt_range_hi, B=ref.B_ce))
    T = pd.DataFrame(rows)
    thA = fits[("nls", "alpha > beta")][0]
    thB = fits.get(("nls", "beta > alpha"), (None,))[0]
    alphas = np.unique(np.round(np.r_[np.linspace(0.25, 1.45, 13), 0.1, 0.15, 0.2, thA[0]] if thB is None else
                                np.r_[np.linspace(0.25, 1.45, 13), 0.1, 0.15, 0.2, thA[0], thB[0]], 4))
    betas = np.unique(np.round(np.r_[np.linspace(0.04, 0.20, 9), 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, thA[1]] if thB is None else
                               np.r_[np.linspace(0.04, 0.20, 9), 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, thA[1], thB[1]], 4))
    P = profile(x, z, y, g, R, thA, alphas, betas, th_other=thB)
    fglob = min(fits[("nls", m)][1] for m in ("alpha > beta", "beta > alpha") if ("nls", m) in fits)
    P["delta_ssr"] = P.ssr - 2 * fglob
    s2 = fglob * 2 / (len(y) - (2 + 3 * R))
    P["lr_gaussian"] = P.delta_ssr / s2            # ~ chi2(2) reference for the (alpha, beta) pair
    per = pd.DataFrame(dict(recipe=groups, tilt_all_nls=Gall.tilt_lnA_minus_lnB.values,
                            **{f"tilt_final_{e}_{'ab' if m == 'alpha > beta' else 'ba'}": tilts[(e, m)] for (e, m) in tilts}))
    return dict(table=T, profile=P, recipes=per, info=info)
