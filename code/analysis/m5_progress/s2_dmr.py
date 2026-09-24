"""Task 2: Diamond-McFadden-Rodriguez (DMR) non-identification in the Ho et al. data, Hicks neutrality, and
progress rates under imposed (experimental) technologies.

Econometrics. In Ho et al.'s model the year terms are factor-augmenting technical change: alpha_year = alpha*g_N,
beta_year = beta*g_D (model_spec.md, Proposition 5). Locally, d ln L/dt = -(s_i alpha_year + (1-s_i) beta_year) with
s_i = u_i/(u_i+v_i) the share of the parameter term in loss. If s_i were constant the data would identify only the
share-weighted combination (a ridge with slope -s/(1-s) in the (alpha_year, beta_year) plane); identification of the
split (the factor bias) comes only from cross-sectional variation in s_i. On the compute-optimal path s = a =
beta/(alpha+beta) and the identified combination is gamma*(g_N+g_D): effective-compute growth. Off the path the
identified combination is not exactly g_C, so the 'effective-compute rate' is itself only as well identified as the
split. We show this with (i) the profile objective over (alpha_year, beta_year), (ii) 1-D profile likelihoods for
g_C and for the split share phi = g_N/g_C, (iii) bootstrap clouds.

Hicks neutrality (proportional shift of A and B) is alpha g_N = beta g_D, i.e. alpha_year = beta_year (Ho's
'model 12'). Imposed technologies: we fix (alpha, beta) at experimental values and re-estimate augmentation rates.
"""
import json
import os

import numpy as np
import pandas as pd

from common import (B_BOOT, BESIROGLU, BESIROGLU_SE, FIG, HOFFMANN_TEX, LN2, PROC, TAB, HoModel, HoSpec, aer_style,
                    boot_indices, bootstrap_fit, chinchilla_fits, fit, log, pct, read_registry, tc_ci)

AY_GRID = np.round(np.arange(-0.12, 0.0801, 0.005), 4)
BY_GRID = np.round(np.arange(-0.06, 0.1601, 0.005), 4)


# ----------------------------------------------------------------------------- profile workers (top level for spawn)
def _profile_row(args):
    ay, by_list, data, norm = args
    out, warm = [], None
    for by in by_list:
        m = HoModel(HoSpec(delta=0, ay_fixed=float(ay), by_fixed=float(by)), data, norm)
        x, f = fit(m, x0=warm, n_random=0)
        warm = x
        r = m.rates(x)
        out.append(dict(alpha_year=ay, beta_year=by, mse=f, alpha_param=r['alpha_param'], beta_data=r['beta_data'],
                        g_N=r['g_N'], g_D=r['g_D'], g_C=r['g_C']))
    return out


def _profile_1d(args):
    kind, val, data, norm = args
    spec = HoSpec(delta=0, gC_fixed=float(val)) if kind == 'gC' else HoSpec(delta=0, phi_fixed=float(val))
    m = HoModel(spec, data, norm)
    x, f = fit(m, n_random=6, seed=3)
    r = m.rates(x)
    return dict(kind=kind, value=val, mse=f, **{k: r[k] for k in ('alpha_param', 'beta_data', 'alpha_year', 'beta_year',
                                                                  'g_N', 'g_D', 'g_C', 'gamma')})


def lr_crossing(vals, LR, crit=3.841):
    """Boundaries of {v: LR(v) <= crit} on a 1-D profile grid, located by linear interpolation of LR between the
    outermost grid point inside the region and its neighbour outside (NaN if the region touches the grid edge).
    Returns the convex hull of the region if it is not contiguous. (Reviewer addition: the grid-point bounds are
    conservative by up to one grid step.)"""
    v, L = np.asarray(vals, float), np.asarray(LR, float)
    inside = np.where(L <= crit)[0]
    if not len(inside):
        return np.nan, np.nan
    i0, i1 = inside.min(), inside.max()
    lo = np.nan if i0 == 0 else v[i0 - 1] + (v[i0] - v[i0 - 1]) * (L[i0 - 1] - crit) / (L[i0 - 1] - L[i0])
    hi = np.nan if i1 == len(v) - 1 else v[i1] + (v[i1 + 1] - v[i1]) * (crit - L[i1]) / (L[i1 + 1] - L[i1])
    return float(lo), float(hi)


def registry_technologies(reg, max_rows=6):
    """(label, alpha, beta) for the primary Huber estimate of each dataset in another module's technology registry,
    plus a record of the registry file used (see common.read_registry)."""
    tr, info = read_registry(reg)
    out = []
    for _, r in tr.head(max_rows).iterrows():
        sub = str(r.get('subset', ''))       # m1 subsets are long free-text descriptions: keep labels table-sized
        lab = f"{r.get('dataset', 'row')} ({sub}, huber)" if len(sub) <= 20 else f"{r.get('dataset', 'row')} (huber)"
        out.append((lab, float(r['alpha']), float(r['beta'])))
    info['labels'] = [o[0] for o in out]
    return out, info


def _ms_worker(args):
    """Random-start SLSQP (run to convergence) on Ho et al.'s penalized objective."""
    seeds, data, norm = args
    from scipy.optimize import minimize
    m = HoModel(HoSpec(delta=0.0025), data, norm)
    out = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        st = rng.normal(0, 0.6, len(m.names))
        st[m.ix['alpha_param']], st[m.ix['beta_data']] = rng.uniform(0.01, 0.6), rng.uniform(0.01, 0.6)
        st[m.ix['alpha_year']], st[m.ix['beta_year']] = rng.uniform(-0.15, 0.2), rng.uniform(-0.15, 0.2)
        r = minimize(lambda th: m.objective(th), st, method='SLSQP', options=dict(ftol=1e-13, maxiter=5000))
        rr = m.rates(r.x)
        out.append(dict(seed=sd, objective=float(r.fun), alpha_param=rr['alpha_param'], beta_data=rr['beta_data'],
                        alpha_year=rr['alpha_year'], beta_year=rr['beta_year'], TC_months=rr['TC_months']))
    return out


def multistart_check(pool, r1, n_starts=300):
    """Is Ho et al.'s published point the minimizer of their own objective? 300 random starts, SLSQP to convergence."""
    d, m_ho = r1['data'], r1['model']
    seeds = list(range(1000, 1000 + n_starts))
    res = pd.DataFrame([x for xs in pool.map(_ms_worker, [(seeds[i::6], d, m_ho.norm) for i in range(6)]) for x in xs])
    res.to_csv(os.path.join(PROC, 'multistart_ho_objective.csv'), index=False)
    f_pub = r1['summary']['objective']
    f_min = float(res.objective.min())
    out = dict(n_starts=n_starts, published_objective=f_pub, min_objective=f_min,
               share_below_published=float(np.mean(res.objective < f_pub - 1e-6)),
               share_within_1e5_of_min=float(np.mean(res.objective < f_min + 1e-5)),
               converged_from_zero=float(r1['summary']['conv_objective']),
               TC_at_min=float(res.loc[res.objective.idxmin(), 'TC_months']))
    json.dump(out, open(os.path.join(PROC, 'multistart_summary.json'), 'w'), indent=1)
    log(f"multistart: min objective {f_min:.7f} (published {f_pub:.7f}); share below published {out['share_below_published']:.3f}; "
        f"share within 1e-5 of min {out['share_within_1e5_of_min']:.3f}; T_C at min {out['TC_at_min']:.2f}")
    return out


def estimate_row(label, spec, data, idxs, pool, x0=None, n_random=8, n_random_boot=2, extra_starts=(), note='', method='auto'):
    """Point estimate + paper-cluster bootstrap for one specification; returns a summary dict and the draws.
    Penalized (delta>0) rows use SLSQP run to convergence unless method='ho' (scipy defaults, Ho et al.'s code)."""
    m = HoModel(spec, data)
    x, f = fit(m, x0=x0, n_random=n_random if spec.delta == 0 else 0, seed=11, extra_starts=extra_starts, method=method)
    bt = bootstrap_fit(spec, data, m.norm, x, idxs, pool=pool, n_random=n_random_boot if spec.delta == 0 else 0, method=method)
    r = m.rates(x)
    rb = pd.DataFrame([m.rates(b[:-1]) for b in bt if np.all(np.isfinite(b))])
    mse = float(np.mean((data['y'] - m.predict(x)) ** 2))
    row = dict(label=label, n=len(data['y']), k=len(m.names), mse=mse, note=note)
    for k in ('alpha_param', 'beta_data', 'gamma', 'alpha_year', 'beta_year', 'g_N', 'g_D', 'g_C'):
        row[k] = r[k]
        lo, _, hi = pct(rb[k]) if len(rb) else (np.nan, np.nan, np.nan)
        row[k + '_lo'], row[k + '_hi'], row[k + '_se'] = lo, hi, float(rb[k].std(ddof=1)) if len(rb) else np.nan
    row['TC_months'] = r['TC_months']
    row['TC_lo'], row['TC_hi'] = tc_ci(rb['g_C']) if len(rb) else (np.nan, np.nan)
    row['ec_per_year'] = r['ec_per_year']
    row['share_gC_pos'] = float(np.mean(rb['g_C'] > 0)) if len(rb) else np.nan
    row['B'] = int(len(rb))
    return row, x, bt, m


def run(pool, r1):
    df, d, m_ho, x_ho = r1['df'], r1['data'], r1['model'], r1['x']
    n = len(df)
    idxs = boot_indices(n, B_BOOT, 20260923, clusters=df['paper'].to_numpy())
    chin = chinchilla_fits()
    gamma_E0 = chin['E0_huber']['gamma']
    log(f"Chinchilla sweep: full gamma={chin['full']['gamma']:.4f}; E=0 refit gamma={gamma_E0:.4f} "
        f"(mean gamma*R/L = {chin['gamma_total_mean']:.4f})")

    # ------------------------------------------------------------------ A. point estimates on the ridge
    rows, draws = [], {}
    specs = [
        ('Ho model 7 (L1 penalty, replication: scipy default tolerance)', HoSpec(delta=0.0025), None, 'A1'),
        ('Ho model 7, same objective run to convergence', HoSpec(delta=0.0025), None, 'A1c'),
        ('Model 7, unpenalized NLS', HoSpec(delta=0), x_ho, 'A2'),
        ('Hicks-neutral, L1 penalty (Ho model 12), converged', HoSpec(delta=0.0025, neutral=True), None, 'A3'),
        ('Hicks-neutral, unpenalized NLS', HoSpec(delta=0, neutral=True), None, 'A4'),
        ('Benchmark-specific E estimated, NLS', HoSpec(delta=0, E='bench'), None, 'A5'),
        ('Experimental exponents (Besiroglu), E_b estimated', HoSpec(delta=0, E='bench', alpha=BESIROGLU.alpha, beta=BESIROGLU.beta), None, 'A7'),
        ('Experimental exponents (Besiroglu), E_b, Hicks-neutral', HoSpec(delta=0, E='bench', alpha=BESIROGLU.alpha, beta=BESIROGLU.beta, neutral=True), None, 'A8'),
        ('Experimental exponents (Hoffmann), E_b estimated', HoSpec(delta=0, E='bench', alpha=HOFFMANN_TEX.alpha, beta=HOFFMANN_TEX.beta), None, 'A9'),
        ('Experimental frontier elasticity in total-loss units, gamma=%.4f' % gamma_E0, HoSpec(delta=0, gamma=gamma_E0), None, 'A10'),
        ('Same, Hicks-neutral', HoSpec(delta=0, gamma=gamma_E0, neutral=True), None, 'A11'),
    ]
    # technology registries from other modules (used only if present): one primary Huber row per dataset
    reg_info = {}
    for reg in ('technology_registry_m1.csv', 'technology_registry_m2.csv'):
        techs_r, reg_info[reg] = registry_technologies(reg)
        for j, (lab_r, al_r, be_r) in enumerate(techs_r):
            specs.append((f'Registry {reg[-6:-4]}: {lab_r}', HoSpec(delta=0, E='bench', alpha=al_r, beta=be_r), None, f'R{reg[-6:-4]}{j}'))
    x_store = {}
    for label, spec, x0, code in specs:
        row, x, bt, m = estimate_row(label, spec, d, idxs, pool, x0=x0, method='ho' if code == 'A1' else 'auto')
        row['code'] = code
        rows.append(row)
        draws[code] = bt
        x_store[code] = (x, m)
        log(f"{code} {label}: T_C={row['TC_months']:.2f} [{row['TC_lo']:.1f}, {row['TC_hi']:.1f}] alpha={row['alpha_param']:.3f} "
            f"beta={row['beta_data']:.3f} mse={row['mse']:.5f}")
    # nesting check: unrestricted NLS must fit at least as well as neutral NLS
    ia2 = [i for i, rr in enumerate(rows) if rr['code'] == 'A2'][0]
    ia4 = [i for i, rr in enumerate(rows) if rr['code'] == 'A4'][0]
    if rows[ia2]['mse'] > rows[ia4]['mse'] + 1e-9:
        m2 = x_store['A2'][1]
        xn = x_store['A4'][0]
        xs = np.insert(xn, m2.ix['beta_year'], xn[x_store['A4'][1].ix['alpha_year']])
        row, x, bt, m = estimate_row(specs[2][0], specs[2][1], d, idxs, pool, x0=xs)
        row['code'] = 'A2'
        rows[ia2], draws['A2'], x_store['A2'] = row, bt, (x, m)
        log("A2 re-estimated from the neutral solution (nesting)")

    # Whitfill (2025) first-pass: keep Ho's year coefficients, divide by experimental (reducible-loss) exponents
    b_ho = r1['boot_cl'][:, :-1]
    rng = np.random.default_rng(5)
    al_d = rng.normal(BESIROGLU.alpha, BESIROGLU_SE['alpha'], len(b_ho))
    be_d = rng.normal(BESIROGLU.beta, BESIROGLU_SE['beta'], len(b_ho))
    gC_w = b_ho[:, 3] / al_d + b_ho[:, 8] / be_d
    gN_w, gD_w = x_ho[3] / BESIROGLU.alpha, x_ho[8] / BESIROGLU.beta
    lo, hi = tc_ci(gC_w)
    rows.insert(6, dict(label="Whitfill first pass: Ho's year coefficients / experimental exponents", code='A6', n=n,
                        alpha_param=BESIROGLU.alpha, beta_data=BESIROGLU.beta, gamma=BESIROGLU.gamma, alpha_year=x_ho[3],
                        beta_year=x_ho[8], g_N=gN_w, g_D=gD_w, g_C=gN_w + gD_w, TC_months=12 * LN2 / (gN_w + gD_w),
                        TC_lo=lo, TC_hi=hi, g_C_lo=pct(gC_w)[0], g_C_hi=pct(gC_w)[2], ec_per_year=np.exp(gN_w + gD_w),
                        mse=np.nan, k=np.nan, B=len(gC_w), share_gC_pos=float(np.mean(gC_w > 0)),
                        note='arithmetic; not a re-estimate'))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(TAB, 'm5_progress_table7_panelA.csv'), index=False)

    # ------------------------------------------------------------------ B. Hicks-neutrality tests
    tests = {}
    for code_u, code_r, lab in (('A1c', 'A3', 'L1'), ('A2', 'A4', 'NLS')):
        bu = draws[code_u]
        mu = x_store[code_u][1]
        diff = bu[:, mu.ix['alpha_year']] - bu[:, mu.ix['beta_year']]
        diff = diff[np.isfinite(diff)]
        xu = x_store[code_u][0]
        mse_u = tab.set_index('code').loc[code_u, 'mse']
        mse_r = tab.set_index('code').loc[code_r, 'mse']
        F = (mse_r - mse_u) / (mse_u / (n - len(mu.names)))
        tests[lab] = dict(diff=float(xu[mu.ix['alpha_year']] - xu[mu.ix['beta_year']]), diff_lo=float(pct(diff)[0]),
                          diff_hi=float(pct(diff)[2]), diff_se=float(np.std(diff, ddof=1)),
                          boot_p_two_sided=float(2 * min(np.mean(diff <= 0), np.mean(diff >= 0))), F_iid=float(F),
                          # the F statistic is a valid (iid) test only for the unpenalized NLS pair; for the L1 pair the
                          # MSEs are evaluated at penalized optima, so F is descriptive only (reviewer note)
                          F_valid=bool(lab == 'NLS'),
                          mse_unrestricted=float(mse_u), mse_neutral=float(mse_r))
    log(f"Hicks neutrality: {tests}")

    # ------------------------------------------------------------------ optimizer path: SLSQP from zero on Ho's objective
    from scipy.optimize import minimize
    path = [np.zeros(len(m_ho.names))]
    minimize(lambda th: m_ho.objective(th), np.zeros(len(m_ho.names)), method='SLSQP', options=dict(ftol=1e-13, maxiter=5000),
             callback=lambda xk: path.append(xk.copy()))
    path = np.array(path)
    pdf = pd.DataFrame(path, columns=m_ho.names)
    pdf['objective'] = [m_ho.objective(p_) for p_ in path]
    pdf['TC_months'] = [m_ho.rates(p_)['TC_months'] for p_ in path]
    pdf.to_csv(os.path.join(PROC, 'slsqp_path.csv'), index_label='iteration')
    n_default = int(np.argmin(np.abs(pdf['objective'] - r1['summary']['objective'])))   # index on the callback path
    # scipy's own iteration count at the default stop (Ho et al.'s printed output: nit = 18, nfev = 209)
    r_def = minimize(lambda th: m_ho.objective(th), np.zeros(len(m_ho.names)), method='SLSQP')
    nit_default = dict(nit=int(r_def.nit), nfev=int(r_def.nfev))

    # ------------------------------------------------------------------ C. profile objective over (alpha_year, beta_year)
    args = [(ay, BY_GRID, d, m_ho.norm) for ay in AY_GRID]
    prof = pd.DataFrame([r for rr in pool.map(_profile_row, args) for r in rr])
    mse_min = min(prof['mse'].min(), tab.loc[tab.code == 'A2', 'mse'].iloc[0])
    prof['LR'] = n * np.log(prof['mse'] / mse_min)            # Gaussian profile LR statistic (iid errors)
    prof.to_csv(os.path.join(PROC, 'dmr_profile_grid.csv'), index=False)

    # 1-D profiles: g_C (effective-compute growth) and phi = g_N/g_C (share of parameter augmentation)
    gC_vals = np.round(np.concatenate([np.linspace(0.15, 0.6, 19), np.linspace(0.65, 3.5, 20)]), 4)
    phi_vals = np.round(np.linspace(-1.5, 6.0, 31), 4)
    one = pool.map(_profile_1d, [('gC', v, d, m_ho.norm) for v in gC_vals] + [('phi', v, d, m_ho.norm) for v in phi_vals])
    p1 = pd.DataFrame(one)
    p1['LR'] = n * np.log(p1['mse'] / mse_min)
    p1.to_csv(os.path.join(PROC, 'dmr_profile_1d.csv'), index=False)
    pg = p1[p1.kind == 'gC'].sort_values('value')
    inside = pg[pg.LR <= 3.841]
    gC_ci = (float(inside.value.min()), float(inside.value.max())) if len(inside) else (np.nan, np.nan)
    TC_prof_ci = (12 * LN2 / gC_ci[1], 12 * LN2 / gC_ci[0])
    pphi = p1[p1.kind == 'phi'].sort_values('value')
    inphi = pphi[pphi.LR <= 3.841]
    phi_ci = (float(inphi.value.min()), float(inphi.value.max())) if len(inphi) else (np.nan, np.nan)
    TC_range_phi = (float((12 * LN2 / inphi.g_C).min()), float((12 * LN2 / inphi.g_C).max())) if len(inphi) else (np.nan, np.nan)
    # interpolated crossings of the LR cutoff (the grid-point intervals above are conservative by up to one step)
    gC_ci_int = lr_crossing(pg.value, pg.LR)
    TC_prof_ci_int = (12 * LN2 / gC_ci_int[1], 12 * LN2 / gC_ci_int[0])
    phi_ci_int = lr_crossing(pphi.value, pphi.LR)

    # ridge geometry: share of the parameter term s_i at Ho's estimate; predicted ridge slope -s/(1-s)
    parts = m_ho.predict(x_ho, parts=True)
    s_i = parts['u'] / (parts['u'] + parts['v'])
    s_bar = float(np.mean(s_i))
    b1k = r1['boot_ho1k'][:, :-1]
    cov = np.cov(b1k[:, 3], b1k[:, 8])
    evals, evecs = np.linalg.eigh(cov)
    pc = evecs[:, np.argmax(evals)]
    slope_boot = float(pc[1] / pc[0])
    # valley floor: for each alpha_year the beta_year that minimises the profile
    floor = prof.loc[prof.groupby('alpha_year')['mse'].idxmin()].sort_values('alpha_year')
    floor.to_csv(os.path.join(PROC, 'dmr_valley_floor.csv'), index=False)
    fl_in = floor[floor.LR <= 5.991]
    slope_floor = float(np.polyfit(fl_in.alpha_year, fl_in.beta_year, 1)[0]) if len(fl_in) > 2 else np.nan
    reg_in = prof[prof.LR <= 5.991]
    tc_in = 12 * LN2 / reg_in.g_C[reg_in.g_C > 0]
    summ = dict(n=n, B=len(idxs), neutrality=tests, s_bar=s_bar, s_sd=float(np.std(s_i)), s_p10=float(np.percentile(s_i, 10)),
                s_p90=float(np.percentile(s_i, 90)), ridge_slope_theory=-s_bar / (1 - s_bar), ridge_slope_boot_pc1=slope_boot,
                ridge_slope_floor=slope_floor, a_onpath_Ho=float(x_ho[9] / (x_ho[4] + x_ho[9])),
                isoTC_slope=-float(x_ho[9] / x_ho[4]),
                corr_boot_iid1000=float(np.corrcoef(b1k[:, 3], b1k[:, 8])[0, 1]),
                corr_boot_cluster_L1=float(np.corrcoef(draws['A1'][:, 3], draws['A1'][:, 8])[0, 1]),
                corr_boot_cluster_NLS=float(np.corrcoef(draws['A2'][:, 3], draws['A2'][:, 8])[0, 1]),
                gC_profile_ci=gC_ci, TC_profile_ci=TC_prof_ci, phi_profile_ci=phi_ci, TC_range_within_phi_ci=TC_range_phi,
                gC_profile_ci_interp=gC_ci_int, TC_profile_ci_interp=TC_prof_ci_int, phi_profile_ci_interp=phi_ci_int,
                slsqp_default_nit=nit_default, registries_used=reg_info,
                TC_range_joint_region=(float(tc_in.min()), float(tc_in.max())) if len(tc_in) else None,
                mse_min=float(mse_min), gamma_E0=gamma_E0, chinchilla=chin, slsqp_iterations_to_convergence=int(len(pdf) - 1),
                slsqp_default_stop_iteration=n_default,
                points=dict(ho_default=[float(x_ho[3]), float(x_ho[8])], ho_converged=[float(x_store['A1c'][0][3]), float(x_store['A1c'][0][8])],
                            nls=[float(x_store['A2'][0][3]), float(x_store['A2'][0][8])]))
    json.dump(summ, open(os.path.join(PROC, 'dmr_summary.json'), 'w'), indent=1, default=float)
    log(f"ridge: s_bar={s_bar:.3f} slope theory {-s_bar / (1 - s_bar):.2f}, boot PC1 {slope_boot:.2f}, floor {slope_floor:.2f}; "
        f"profile CI T_C {TC_prof_ci[0]:.1f}-{TC_prof_ci[1]:.1f} months (interpolated {TC_prof_ci_int[0]:.1f}-{TC_prof_ci_int[1]:.1f}); "
        f"phi CI {phi_ci} (interpolated {np.round(phi_ci_int, 2)}); T_C over phi CI {TC_range_phi}; SLSQP default stop {nit_default}")

    figure_from_files()
    return dict(tab=tab, summ=summ, draws=draws, x_store=x_store)


def figure_from_files():
    """Draw the DMR figure from the processed files written by run() (so it can be re-drawn without re-estimating)."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    prof = pd.read_csv(os.path.join(PROC, 'dmr_profile_grid.csv'))
    p1 = pd.read_csv(os.path.join(PROC, 'dmr_profile_1d.csv'))
    floor = pd.read_csv(os.path.join(PROC, 'dmr_valley_floor.csv'))
    pdf = pd.read_csv(os.path.join(PROC, 'slsqp_path.csv'))
    summ = json.load(open(os.path.join(PROC, 'dmr_summary.json')))
    b1k = np.load(os.path.join(PROC, 'boot_m7_ho1000.npy'))[:, :-1]
    pts = summ['points']
    aer_style.use()
    fig, axs = plt.subplots(1, 3, figsize=(aer_style.WIDTH_FULL, 2.55), gridspec_kw=dict(width_ratios=[1.2, 1, 1]))
    # (a) profile LR contours in (alpha_year, beta_year) + bootstrap cloud + optimizer path
    ax = axs[0]
    P = prof.pivot(index='beta_year', columns='alpha_year', values='LR')
    X, Y = np.meshgrid(P.columns.values, P.index.values)
    lim = (AY_GRID.min(), AY_GRID.max())
    ax.scatter(b1k[:, 3], b1k[:, 8], s=1.5, color=aer_style.MUTED, alpha=0.35, lw=0, zorder=1)
    ax.contourf(X, Y, P.values, levels=[-1, 5.991], colors=[aer_style.BLUE], alpha=0.10, zorder=0)
    ax.contour(X, Y, P.values, levels=[5.991, 20], colors=[aer_style.INK, aer_style.MUTED],
               linewidths=[1.0, 0.6], linestyles=['-', ':'], zorder=2)
    ax.plot([], [], '-', color=aer_style.INK, lw=1.0, label='95% joint region (profile LR)')
    ax.plot(lim, lim, ls='--', color=aer_style.INK2, lw=0.7, zorder=2)
    ax.text(0.047, 0.066, 'Hicks-\nneutral', fontsize=5.8, color=aer_style.INK2, ha='left', va='top')
    # SLSQP iterates on Ho et al.'s objective, from the default stopping point to convergence
    k0 = int(summ['slsqp_default_stop_iteration'])
    pp = pdf.iloc[max(k0 - 3, 0):]
    ax.plot(pp['alpha_year'], pp['beta_year'], '.', color=aer_style.ORANGE, ms=2.4, zorder=3, label='SLSQP iterates after the default stop')
    ax.plot(*pts['ho_default'], 'o', color=aer_style.ORANGE, ms=5, mec='white', mew=0.6, zorder=4, label='Ho et al. (default stop)')
    ax.plot(*pts['ho_converged'], 'D', color=aer_style.BLUE, ms=4.2, mec='white', mew=0.6, zorder=4, label='Same objective, converged')
    ax.plot(*pts['nls'], 's', color=aer_style.AQUA, ms=4.2, mec='white', mew=0.6, zorder=4, label='Unpenalized NLS')
    ax.plot([], [], 'o', color=aer_style.MUTED, ms=2, label='Bootstrap draws (Ho code)')
    ax.set_xlim(*lim); ax.set_ylim(BY_GRID.min(), BY_GRID.max())
    ax.set_xlabel(r'$\alpha_{\mathrm{year}}$ ($=\alpha\, g_N$)'); ax.set_ylabel(r'$\beta_{\mathrm{year}}$ ($=\beta\, g_D$)')
    ax.set_title('(a) Profile LR, bootstrap, optimizer', loc='left')
    ax.legend(loc='lower left', fontsize=5.4, handletextpad=0.2, borderaxespad=0.15, labelspacing=0.25, markerscale=0.9)
    # (b) T_C along the valley floor
    ax = axs[1]
    fl = floor[floor.g_C > 0]
    ax.plot(fl.alpha_year, 12 * LN2 / fl.g_C, color=aer_style.BLUE, lw=1.0)
    inside = fl[fl.LR <= 5.991]
    ax.plot(inside.alpha_year, 12 * LN2 / inside.g_C, color=aer_style.BLUE, lw=3.2, alpha=0.35, solid_capstyle='butt',
            label='inside 95% region')
    ax.axhline(8.4, color=aer_style.ORANGE, lw=0.9, ls=':')
    ax.text(AY_GRID.min() + 0.003, 8.9, 'Ho et al.: 8.4', color=aer_style.ORANGE, fontsize=6.5)
    ax.set_yscale('log'); ax.set_ylim(3, 60)
    ax.set_yticks([3, 5, 10, 20, 40]); ax.set_yticklabels(['3', '5', '10', '20', '40'])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r'$\alpha_{\mathrm{year}}$ (valley floor)'); ax.set_ylabel(r'$T_C$ (months)')
    ax.set_title('(b) Doubling time along the ridge', loc='left')
    ax.legend(loc='upper left', fontsize=6.3)
    # (c) 1-D profile over phi = g_N/g_C
    ax = axs[2]
    pp = p1[p1.kind == 'phi'].sort_values('value')
    ax.plot(pp.value, pp.LR, color=aer_style.INK)
    ax.axhline(3.841, color=aer_style.INK2, lw=0.8, ls='--')
    ax.text(pp.value.max(), 3.5, '95% cutoff', fontsize=6.0, color=aer_style.INK2, ha='right', va='top')
    ax.set_ylim(0, 14)
    ax.set_xlabel(r'parameter-augmenting share $\phi=g_N/g_C$'); ax.set_ylabel('profile LR statistic')
    for _, rr in pp.iterrows():
        if np.any(np.isclose(rr.value, [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])) and rr.g_C > 0 and rr.LR < 13:
            ax.annotate(f"{12 * LN2 / rr.g_C:.0f}", (rr.value, rr.LR), textcoords='offset points', xytext=(4, 3), fontsize=6.3,
                        ha='left', color=aer_style.BLUE)
    ax.text(0.97, 0.97, r'labels: $T_C$ (months)', transform=ax.transAxes, fontsize=6.3, color=aer_style.BLUE, va='top', ha='right')
    ax.set_title('(c) Profile over the factor bias', loc='left')
    fig.tight_layout(w_pad=0.6)
    aer_style.savefig(fig, 'm5_progress_dmr_ridge')
