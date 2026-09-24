"""Task 3: why is the cross-lab beta_data = 0.04 when designed sweeps give ~0.37?  Candidate channels:
 (a) measurement of D (dataset size vs tokens seen vs effective data under repetition; tokenization/vocabulary),
 (b) collinearity of D with calendar time (and the time trend itself),
 (c) specification (irreducible loss E set to 0),
 (d) selection (top-3 models per paper by perplexity; outliers; zero-shot rows).

Three kinds of evidence:
 Panel A  re-estimates of Ho's model 7 (their estimator) on data variants / subsamples, paper-cluster bootstrap;
 Panel B  the experimental benchmark: Besiroglu et al.'s estimator on the Chinchilla sweep with and without E;
 Panel C  a Monte Carlo on Ho et al.'s *actual* design (N, D, dates, benchmarks, papers) with a known truth
          (Besiroglu exponents, E > 0, Hicks-neutral progress with T_C = 12 months), switching channels on one at a time.
Plus: the reliability ratio that classical measurement error in ln D would need (Griliches-Hausman formula with
other regressors), and a profile of the estimates over a fixed irreducible loss E.
"""
import json
import os

import numpy as np
import pandas as pd

from common import (B_BOOT, BESIROGLU, FIG, LN2, PROC, TAB, HoModel, HoSpec, aer_style, boot_indices, bootstrap_fit,
                    chinchilla_fits, fit, fmt, ho_arrays, load_ho, log, pct, tc_ci, tex_label, write_tex)

R_STAR_D = 15.4      # Muennighoff et al. (2023) decay constant for repeated data (R*_D = 15.39)
MC_REPS = 204      # 6 workers x 34 replications
TRUE_TC = 12.0       # months (Monte Carlo truth)


def effective_data(U, epochs, r_star=R_STAR_D):
    """Muennighoff et al. (2023): D' = U + U R* (1 - exp(-R/R*)), R = epochs - 1 repetitions; tokens seen if epochs < 1."""
    ep = np.asarray(epochs, float)
    rep = np.maximum(ep - 1.0, 0.0)
    return np.where(ep < 1, U * ep, U * (1 + r_star * (1 - np.exp(-rep / r_star))))


def impute_epochs(df):
    """Known epochs where reported; otherwise the median of known epochs among models with a similar dataset size
    (log10 D bins); web-scale datasets (D >= 1e10 tokens) are single-epoch."""
    ep = df['epochs_num'].copy()
    lD = np.log10(df['dataset'])
    bins = pd.cut(lD, [0, 7, 8.5, 10, 20])
    med = ep.groupby(bins, observed=False).median()
    fill = bins.map(med).astype(float)
    fill[lD >= 10] = 1.0
    out = ep.fillna(fill).fillna(1.0)
    return out.clip(lower=0.01)


def variant_row(label, code, df, pool, spec=None, dvar='dataset', note=''):
    spec = spec or HoSpec(delta=0.0025)
    d = ho_arrays(df, dvar=dvar)
    m = HoModel(spec, d)
    x, f = fit(m, n_random=8 if spec.delta == 0 else 0)          # converged (penalized) or multistart NLS
    xd, _ = fit(m, method='ho') if spec.delta > 0 else (x, f)  # Ho et al.'s code: scipy default tolerance
    idxs = boot_indices(len(df), B_BOOT, 99, clusters=df['paper'].to_numpy())
    bt = bootstrap_fit(spec, d, m.norm, x, idxs, pool=pool, n_random=2 if spec.delta == 0 else 0)
    r = m.rates(x)
    rd = m.rates(xd)
    rb = pd.DataFrame([m.rates(b[:-1]) for b in bt if np.all(np.isfinite(b))])
    row = dict(code=code, label=label, n=len(df), papers=int(df['Reference'].nunique()), note=note)
    for k in ('alpha_param', 'beta_data', 'gamma', 'alpha_year', 'beta_year', 'g_C'):
        row[k] = r.get(k, np.nan)
        if k in rb and not spec.no_year or k in ('alpha_param', 'beta_data', 'gamma'):
            lo, _, hi = pct(rb[k])
            row[k + '_lo'], row[k + '_hi'] = lo, hi
    if not spec.no_year:
        row['TC_months'] = r['TC_months']
        row['TC_lo'], row['TC_hi'] = tc_ci(rb['g_C'])
    else:
        row['TC_months'] = row['TC_lo'] = row['TC_hi'] = np.nan
    row['mse'] = float(np.mean((d['y'] - m.predict(x)) ** 2))
    row['alpha_default'], row['beta_default'] = rd['alpha_param'], rd['beta_data']
    row['TC_default'] = rd['TC_months'] if not spec.no_year else np.nan
    log(f"  {code} {label}: n={len(df)} alpha={row['alpha_param']:.3f} beta={row['beta_data']:.3f} "
        f"[{row.get('beta_data_lo', np.nan):.3f},{row.get('beta_data_hi', np.nan):.3f}] T_C={row['TC_months']:.1f}; "
        f"Ho code: beta={row['beta_default']:.3f} T_C={row['TC_default']:.1f}")
    return row


# ----------------------------------------------------------------------------- Monte Carlo on Ho's design
def _mc_truth(dfa, T=BESIROGLU, tc_months=TRUE_TC, use_eff=True):
    """Noise-free truth on the design dfa: Chinchilla technology with E, Hicks-neutral factor augmentation
    (alpha g_N = beta g_D) such that effective compute doubles every tc_months, benchmark scale k_b
    (word-level vs token units; matched to the observed mean log perplexity by benchmark)."""
    t = dfa['publication_date'].to_numpy() - dfa['publication_date'].min()
    gC = LN2 / (tc_months / 12.0)
    lam = T.gamma * gC
    gN, gD = lam / T.alpha, lam / T.beta
    D = dfa['D_eff'].to_numpy() if use_eff else dfa['dataset'].to_numpy()
    N = dfa['param'].to_numpy()
    L = T.E + T.A * (N * np.exp(gN * t)) ** -T.alpha + T.B * (D * np.exp(gD * t)) ** -T.beta
    y = dfa['y'].to_numpy()
    out = L.copy()
    for b in ('wt103', 'ptb', 'wt2'):
        msk = (dfa['dataset_name'] == b).to_numpy()
        out[msk] = L[msk] * y[msk].mean() / L[msk].mean()
    return out, dict(g_N=gN, g_D=gD, g_C=gC, lam=lam)


def _select(dfa, ysim, rule, rng):
    """Row selection within paper: 'best3' = Ho's rule (3 lowest perplexities), 'rand3' = 3 at random, 'all'."""
    if rule == 'all':
        return np.arange(len(dfa))
    key = ysim if rule == 'best3' else rng.random(len(dfa))
    tmp = pd.DataFrame(dict(ref=dfa['Reference'].to_numpy(), key=key, i=np.arange(len(dfa))))
    return tmp.sort_values(['ref', 'key']).groupby('ref').head(3)['i'].to_numpy()


def _mc_worker(args):
    scen, dfa_dict, reps, seed, sigma = args
    dfa = pd.DataFrame(dfa_dict)
    rng = np.random.default_rng(seed)
    base, truth = _mc_truth(dfa, use_eff=True)
    out = []
    for r in range(reps):
        ysim = base.copy()
        if scen.get('transmission'):
            # model-level (row-level) data-augmenting productivity psi_D correlated with (standardized) ln D: Whitfill's channel
            lnD = np.log(dfa['D_eff'].to_numpy())
            z = (lnD - lnD.mean()) / lnD.std()
            rho, s_psi = scen['transmission']
            psi = s_psi * (rho * z + np.sqrt(1 - rho ** 2) * rng.normal(size=len(z)))
            T = BESIROGLU
            t = dfa['publication_date'].to_numpy() - dfa['publication_date'].min()
            N, D = dfa['param'].to_numpy(), dfa['D_eff'].to_numpy()
            L = T.E + T.A * (N * np.exp(truth['g_N'] * t)) ** -T.alpha + T.B * (D * np.exp(truth['g_D'] * t + psi)) ** -T.beta
            ysim = L * (base / (T.E + T.A * (N * np.exp(truth['g_N'] * t)) ** -T.alpha
                               + T.B * (D * np.exp(truth['g_D'] * t)) ** -T.beta))
        ysim = ysim + (sigma * rng.normal(size=len(ysim)) if sigma > 0 else 0.0)
        sel = _select(dfa, ysim, scen['select'], rng)
        sub = dfa.iloc[sel].copy()
        sub['y'] = ysim[sel]
        dvar = 'D_eff' if scen['D'] == 'eff' else 'dataset'
        d = ho_arrays(sub, dvar=dvar)
        rec = {}
        if scen['spec'] == 'true':
            m = HoModel(HoSpec(delta=0, E='bench'), d)
            x, f = fit(m, n_global=(30 if sigma == 0 else 8), seed=r)
            rr = m.rates(x)
            for tag in ('ho', 'gl'):
                rec.update({f'{k}_{tag}': rr[k] for k in ('alpha_param', 'beta_data', 'gamma', 'g_C')})
        else:
            neu = scen['spec'] == 'ho_neutral'              # Hicks neutrality imposed inside the E = 0 form
            m = HoModel(HoSpec(delta=0.0025, neutral=neu), d)   # Ho et al.'s procedure: SLSQP from zero, L1 penalty
            x, f = fit(m, method='ho')
            rr = m.rates(x)
            rec.update({f'{k}_ho': rr[k] for k in ('alpha_param', 'beta_data', 'gamma', 'g_C')})
            m = HoModel(HoSpec(delta=0, neutral=neu), d)    # global: unpenalized NLS, random multistart
            x, f = fit(m, n_global=(30 if sigma == 0 else 8), seed=r)
            rr = m.rates(x)
            rec.update({f'{k}_gl': rr[k] for k in ('alpha_param', 'beta_data', 'gamma', 'g_C')})
        out.append(rec)
        if sigma == 0:
            break
    return out


MC_SCENARIOS = [
    ('S0', 'Correct specification (E_b, effective D), all rows', dict(spec='true', D='eff', select='all')),
    ('S1', '(c) E dropped (Ho form), effective D observed', dict(spec='ho', D='eff', select='all')),
    ('S1h', '(c) E dropped, Hicks neutrality imposed in the E=0 form', dict(spec='ho_neutral', D='eff', select='all')),
    ('S2', '(c)+(a) E dropped, D = dataset size (epochs ignored)', dict(spec='ho', D='obs', select='all')),
    ('S3', '(c)+(a)+(d) plus best-3 per paper (Ho rule)', dict(spec='ho', D='obs', select='best3')),
    ('S3r', '(c)+(a) with 3 random models per paper (control)', dict(spec='ho', D='obs', select='rand3')),
    ('S4', '(c)+(a)+transmission: corr(psi_D, ln D) = -0.5', dict(spec='ho', D='obs', select='all', transmission=(-0.5, 0.5))),
    ('S5', '(c)+(a)+transmission: corr(psi_D, ln D) = +0.5', dict(spec='ho', D='obs', select='all', transmission=(0.5, 0.5))),
]


def monte_carlo(pool, sigma):
    dfa = load_ho(keep_all=True)
    dfa['epochs_imp'] = impute_epochs(dfa)
    dfa['D_eff'] = effective_data(dfa['dataset'].to_numpy(), dfa['epochs_imp'].to_numpy())
    cols = ['publication_date', 'param', 'dataset', 'D_eff', 'y', 'dataset_name', 'ptb_dummy', 'wt2_dummy', 'transformer',
            'Reference', 'vocab', 'paper']
    dd = {c: dfa[c].to_numpy() for c in cols}
    rows = []
    for code, lab, scen in MC_SCENARIOS:
        # pseudo-true (noise-free) and Monte Carlo distribution
        pt = _mc_worker((scen, dd, 1, 0, 0.0))[0]
        per = MC_REPS // 6
        seed0 = sum(ord(c) * 31 ** i for i, c in enumerate(code)) % 9973     # deterministic (str hash is salted)
        res = pool.map(_mc_worker, [(scen, dd, per, 1000 * k + seed0, sigma) for k in range(6)])
        mc = pd.DataFrame([r for rr in res for r in rr])
        row = dict(code=code, label=lab, reps=len(mc))
        for tag in ('ho', 'gl'):
            for k in ('alpha_param', 'beta_data', 'gamma', 'g_C'):
                row[f'{k}_{tag}_pseudo'] = pt[f'{k}_{tag}']
                row[f'{k}_{tag}_mc_med'], row[f'{k}_{tag}_mc_lo'], row[f'{k}_{tag}_mc_hi'] = pct(mc[f'{k}_{tag}'], (50, 5, 95))
            g = pt[f'g_C_{tag}']
            row[f'TC_{tag}_pseudo'] = 12 * LN2 / g if g > 0 else np.inf
            gm = row[f'g_C_{tag}_mc_med']
            row[f'TC_{tag}_mc_med'] = 12 * LN2 / gm if gm > 0 else np.inf
            row[f'TC_{tag}_mc_lo'], row[f'TC_{tag}_mc_hi'] = tc_ci(mc[f'g_C_{tag}'], (5, 95))
        rows.append(row)
        log(f"  MC {code}: Ho proc. alpha {pt['alpha_param_ho']:.3f} beta {pt['beta_data_ho']:.3f} T_C {row['TC_ho_pseudo']:.1f} | "
            f"global alpha {pt['alpha_param_gl']:.3f} beta {pt['beta_data_gl']:.3f} gamma {pt['gamma_gl']:.3f} T_C {row['TC_gl_pseudo']:.1f}; "
            f"MC T_C global {row['TC_gl_mc_med']:.1f} [{row['TC_gl_mc_lo']:.1f},{row['TC_gl_mc_hi']:.1f}], Ho {row['TC_ho_mc_med']:.1f}")
    truth = dict(alpha=BESIROGLU.alpha, beta=BESIROGLU.beta, gamma=BESIROGLU.gamma, TC=TRUE_TC, sigma=sigma,
                 n_design=len(dfa), R_star_D=R_STAR_D)
    return pd.DataFrame(rows), truth, dfa


def run(pool, r1):
    df = r1['df']
    log("s3: data variants (Ho estimator)")
    rows = [variant_row('Baseline (Ho model 7)', 'A0', df, pool)]
    dfx = load_ho(fix_ppl=True)
    rows.append(variant_row('Benchmark-specific perplexity (data fix)', 'A1', dfx, pool))
    # (a) measurement of D
    dfa = df.copy()
    dfa['epochs_imp1'] = dfa['epochs_num'].fillna(1.0)
    dfa['tokens_seen'] = dfa['dataset'] * dfa['epochs_imp1']
    rows.append(variant_row('(a) D = tokens seen (missing epochs = 1)', 'a1', dfa, pool, dvar='tokens_seen'))
    dfa['epochs_imp'] = impute_epochs(dfa)
    dfa['tokens_seen2'] = dfa['dataset'] * dfa['epochs_imp']
    rows.append(variant_row('(a) D = tokens seen (epochs imputed by dataset size)', 'a2', dfa, pool, dvar='tokens_seen2'))
    dfa['D_eff'] = effective_data(dfa['dataset'].to_numpy(), dfa['epochs_imp'].to_numpy())
    rows.append(variant_row('(a) D = effective data, Muennighoff R*=15.4', 'a3', dfa, pool, dvar='D_eff'))
    rows.append(variant_row('(a) Standard word-level vocabulary only', 'a4', df[df.word_vocab == 1].reset_index(drop=True), pool))
    dv = df[np.isfinite(df.vocab)].reset_index(drop=True)
    rows.append(variant_row('(a) Vocabulary control, gamma ln V (known V)', 'a5', dv, pool, spec=HoSpec(delta=0.0025, vocab=True)))
    # (b) collinearity / time
    rows.append(variant_row('(b) Published 2018 or later', 'b1', df[df.publication_date >= 2018].reset_index(drop=True), pool))
    rows.append(variant_row('(b) Transformers only', 'b2', df[df.transformer == 1].reset_index(drop=True), pool))
    rows.append(variant_row('(b) No year terms (static technology)', 'b3', df, pool, spec=HoSpec(delta=0.0025, no_year=True)))
    # (c) specification: E estimated (NLS) -- see also the E profile below
    rows.append(variant_row('(c) Benchmark-specific E estimated (NLS)', 'c1', df, pool, spec=HoSpec(delta=0, E='bench')))
    # (d) selection
    dall = load_ho(keep_all=True)
    rows.append(variant_row('(d) All models of each paper (no top-3 rule)', 'd1', dall, pool))
    rows.append(variant_row('(d) Top-1 model per paper', 'd2', load_ho(top_k=1), pool))
    rows.append(variant_row('(d) Including flagged outliers', 'd3', load_ho(include_outliers=True), pool))
    rows.append(variant_row('(d) Excluding zero-shot evaluations', 'd4', df[df.zero_shot < 1].reset_index(drop=True), pool))
    tabA = pd.DataFrame(rows)
    tabA.to_csv(os.path.join(TAB, 'm5_progress_attenuation_panelA.csv'), index=False)

    # ---- Panel B: experimental benchmark (Chinchilla sweep)
    chin = chinchilla_fits()
    tabB = pd.DataFrame([
        dict(fit='Full model (E estimated), Huber', alpha=chin['full']['alpha'], beta=chin['full']['beta'], gamma=chin['full']['gamma'],
             E=chin['full']['E'], gamma_lo=chin['full_gamma_ci'][0], gamma_hi=chin['full_gamma_ci'][2]),
        dict(fit='E = 0 (Ho specification), Huber', alpha=chin['E0_huber']['alpha'], beta=chin['E0_huber']['beta'],
             gamma=chin['E0_huber']['gamma'], E=0.0, gamma_lo=chin['E0_gamma_ci'][0], gamma_hi=chin['E0_gamma_ci'][2]),
        dict(fit='E = 0 (Ho specification), least squares', alpha=chin['E0_nls']['alpha'], beta=chin['E0_nls']['beta'],
             gamma=chin['E0_nls']['gamma'], E=0.0, gamma_lo=np.nan, gamma_hi=np.nan),
        dict(fit='Mean total-loss elasticity gamma*R/L (full model)', alpha=chin['eps_N_total_mean'], beta=chin['eps_D_total_mean'],
             gamma=chin['gamma_total_mean'], E=np.nan, gamma_lo=np.nan, gamma_hi=np.nan)])
    tabB.to_csv(os.path.join(TAB, 'm5_progress_attenuation_panelB.csv'), index=False)

    # ---- profile over a fixed common irreducible loss E (NLS)
    d = r1['data']
    eprof = []
    x_prev = None
    for E in np.round(np.arange(0.0, 1.51, 0.1), 2):
        m = HoModel(HoSpec(delta=0, E=float(E)), d)
        x, f = fit(m, x0=x_prev, n_random=6, seed=5)
        x_prev = x
        rr = m.rates(x)
        eprof.append(dict(E=E, mse=f, **{k: rr[k] for k in ('alpha_param', 'beta_data', 'gamma', 'alpha_year', 'beta_year', 'g_C', 'TC_months')}))
    eprof = pd.DataFrame(eprof)
    eprof['LR'] = len(d['y']) * np.log(eprof['mse'] / eprof['mse'].min())
    eprof.to_csv(os.path.join(TAB, 'm5_progress_E_profile.csv'), index=False)
    log(f"E profile: gamma {eprof.gamma.min():.3f}-{eprof.gamma.max():.3f}; T_C {eprof.TC_months.min():.1f}-{eprof.TC_months.max():.1f}")

    # ---- reliability ratios for classical error in ln D (Griliches-Hausman with other regressors)
    X = np.c_[np.ones(len(df)), df.publication_date, np.log(df.param), df.ptb_dummy, df.wt2_dummy]
    lnD = np.log(df.dataset.to_numpy())
    b = np.linalg.lstsq(X, lnD, rcond=None)[0]
    R2 = 1 - np.var(lnD - X @ b) / np.var(lnD)
    R2_t = np.corrcoef(lnD, df.publication_date)[0, 1] ** 2
    rel = []
    for lab, rho in (('beta_data: Ho 0.040 vs Besiroglu 0.366', 0.0396 / BESIROGLU.beta),
                     ('gamma: Ho 0.025 vs Chinchilla E=0 refit', r1['model'].rates(r1['x'])['gamma'] / chin['E0_huber']['gamma'])):
        lam = rho + (1 - rho) * R2
        rel.append(dict(target=lab, attenuation=rho, R2_lnD_on_X=R2, required_reliability=lam,
                        implied_noise_sd=np.std(lnD) * np.sqrt(1 - lam), sd_lnD=np.std(lnD)))
    eff_err = np.log(dfa['D_eff'] / dfa['dataset'])
    ts_err = np.log(dfa['tokens_seen2'] / dfa['dataset'])
    relx = dict(R2_lnD_on_X=R2, R2_lnD_on_year=R2_t, VIF_lnD=1 / (1 - R2), sd_lnD=float(np.std(lnD)),
                sd_ln_Deff_over_D=float(np.std(eff_err)), corr_err_lnD_eff=float(np.corrcoef(eff_err, lnD)[0, 1]),
                sd_ln_epochs=float(np.std(ts_err)), corr_err_lnD_epochs=float(np.corrcoef(ts_err, lnD)[0, 1]),
                share_epochs_known=float(df.epochs_num.notna().mean()), rows=rel)
    pd.DataFrame(rel).to_csv(os.path.join(TAB, 'm5_progress_reliability.csv'), index=False)

    # ---- Monte Carlo
    sigma = float(np.sqrt(r1['summary']['objective'] - 0.0025 * np.sum(np.abs(r1['x']))))
    log(f"s3: Monte Carlo on Ho design (sigma={sigma:.3f}, {MC_REPS} reps per scenario)")
    tabC, truth, dfa_all = monte_carlo(pool, sigma)
    tabC.to_csv(os.path.join(TAB, 'm5_progress_attenuation_mc.csv'), index=False)
    json.dump(dict(reliability=relx, mc_truth=truth, chinchilla=chin), open(os.path.join(PROC, 'attenuation_summary.json'), 'w'),
              indent=1, default=float)
    write_latex_from_files()
    figure_from_files()
    return dict(tabA=tabA, tabB=tabB, tabC=tabC, eprof=eprof, rel=relx)


def write_latex_from_files():
    tabA = pd.read_csv(os.path.join(TAB, 'm5_progress_attenuation_panelA.csv'))
    tabB = pd.read_csv(os.path.join(TAB, 'm5_progress_attenuation_panelB.csv'))
    tabC = pd.read_csv(os.path.join(TAB, 'm5_progress_attenuation_mc.csv'))
    js = json.load(open(os.path.join(PROC, 'attenuation_summary.json')))
    truth, relx = js['mc_truth'], js['reliability']
    write_latex(tabA, tabB, tabC, truth, relx)


def write_latex(tabA, tabB, tabC, truth, relx):
    body = [r"\multicolumn{7}{l}{\textit{Panel A. Ho et al.'s model and estimator on alternative data}} \\"]
    for _, r in tabA.iterrows():
        tc = (f"{r.TC_months:.1f}" if np.isfinite(r.TC_months) else '--')
        tcci = (f"[{r.TC_lo:.1f}, {r.TC_hi:.1f}]".replace('inf', r'$\infty$') if np.isfinite(r.TC_months) else '')
        tcd = f"{r.TC_default:.1f}" if np.isfinite(r.TC_default) else '--'
        body.append(f"{tex_label(r.label)} & {int(r.n)} & {r.alpha_param:.3f} & {r.beta_data:.3f} & {r.gamma:.3f} & {tc} & "
                    f"{r.beta_default:.3f}; {tcd} \\\\")
        body.append(f" & & [{r.alpha_param_lo:.3f}, {r.alpha_param_hi:.3f}] & [{r.beta_data_lo:.3f}, {r.beta_data_hi:.3f}] & "
                    f"[{r.gamma_lo:.3f}, {r.gamma_hi:.3f}] & {tcci} & \\\\")
    body.append(r"\addlinespace")
    body.append(r"\multicolumn{7}{l}{\textit{Panel B. Experimental benchmark: Chinchilla sweep (n=240), Besiroglu et al. estimator}} \\")
    for _, r in tabB.iterrows():
        ci = f"[{r.gamma_lo:.3f}, {r.gamma_hi:.3f}]" if np.isfinite(r.gamma_lo) else ''
        body.append(f"{tex_label(r.fit)} & 240 & {r.alpha:.3f} & {r.beta:.3f} & {r.gamma:.3f} {ci} & & \\\\")
    body.append(r"\addlinespace")
    body.append(r"\multicolumn{7}{l}{\textit{Panel C. Monte Carlo on Ho et al.'s design; truth $\alpha=%.3f$, $\beta=%.3f$, $\gamma=%.3f$, $T_C=%d$ months}} \\"
                % (truth['alpha'], truth['beta'], truth['gamma'], int(truth['TC'])))
    body.append(r" & \multicolumn{2}{c}{Ho procedure} & \multicolumn{3}{c}{Global optimum} & MC, global \\")
    body.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-7}")
    body.append(r" & $\hat\beta_{\text{data}}$ & $T_C$ & $\hat\alpha_{\text{param}}$, $\hat\beta_{\text{data}}$ & $\hat\gamma$ & $T_C$ & $T_C$ [5, 95] \\")
    f = lambda v: (r'$\infty$' if not np.isfinite(v) else f"{v:.1f}")
    for _, r in tabC.iterrows():
        body.append(f"{tex_label(r.label)} & {r.beta_data_ho_pseudo:.3f} & {f(r.TC_ho_pseudo)} & {r.alpha_param_gl_pseudo:.3f}, {r.beta_data_gl_pseudo:.3f} & "
                    f"{r.gamma_gl_pseudo:.3f} & {f(r.TC_gl_pseudo)} & {f(r.TC_gl_mc_med)} [{f(r.TC_gl_mc_lo)}, {f(r.TC_gl_mc_hi)}] \\\\")
    hdr = [r" & $n$ & $\hat\alpha_{\text{param}}$ & $\hat\beta_{\text{data}}$ & $\hat\gamma$ & $T_C$ (months) & Ho code: $\hat\beta_{\text{data}}$; $T_C$ \\"]
    pct_missing = int(round(100 * (1 - relx['share_epochs_known'])))
    notes = ("Panel A: Ho et al.'s model 7 and objective (MSE $+0.0025\\sum|\\theta|$, SLSQP from zero, run to convergence) "
             "re-estimated on the indicated data; the last column reports the same fit with the scipy default stopping rule used by "
             "the authors' code. 95\\% paper-cluster bootstrap intervals (400 draws) below the estimates; $\\gamma=\\alpha\\beta/(\\alpha+\\beta)$ is the frontier "
             "elasticity of loss with respect to compute; $T_C$ interval from the bootstrap distribution of $g_C$ ($\\infty$ if $g_C\\le 0$ "
             "is inside it). Row (c1) uses unpenalized least squares. Effective data: $D'=U[1+R^*(1-e^{-(e-1)/R^*})]$ with $e$ epochs and "
             f"$R^*=15.4$ (Muennighoff et al. 2023); missing epochs ({pct_missing}\\% of rows) imputed by the median of models with similar "
             "dataset size. Panel B: Besiroglu et al.'s Huber estimator on the digitized Chinchilla sweep, with $E$ estimated or set to zero; "
             "the last row reports sample means of the total-loss elasticities $\\alpha u/L$, $\\beta v/L$ and $\\gamma(L-E)/L$ implied by the full "
             "model. Panel C: pseudo-true values (noise-free data) when the truth is the Besiroglu technology with Hicks-neutral progress "
             f"($T_C=12$ months) evaluated at Ho et al.'s actual inputs (all {truth['n_design']} model--benchmark rows; benchmark scale matched "
             "to observed means). 'Ho procedure' = SLSQP from $\\theta=0$ on the penalized objective (what the authors' code returns); "
             "'global' = unpenalized least squares with 30 random starts. Last column: Monte Carlo median and 5th--95th percentiles of "
             f"$T_C$ under the global estimator ({MC_REPS} replications, Gaussian noise with s.d. {truth['sigma']:.3f} = the residual s.d. of "
             "Ho et al.'s fit). $\\psi_D$ is a model-specific data-augmenting productivity shock with s.d. 0.5.")
    write_tex(os.path.join(TAB, 'm5_progress_attenuation.tex'), body, 'lcccccc', hdr, notes,
              caption=r'Why is the cross-lab data elasticity so small? Channels of attenuation', label='tab:m5_attenuation',
              size='\\scriptsize')


SHORT = {'A0': 'Baseline (Ho sample)', 'A1': 'Benchmark-specific ppl', 'a1': '(a) tokens seen, ep.=1', 'a2': '(a) tokens seen, imputed',
         'a3': '(a) effective data', 'a4': '(a) word-level vocab only', 'a5': '(a) vocabulary control', 'b1': '(b) 2018 or later',
         'b2': '(b) transformers only', 'b3': '(b) no year terms', 'c1': '(c) $E_b$ estimated (NLS)', 'd1': '(d) all models/paper',
         'd2': '(d) top-1 per paper', 'd3': '(d) incl. outliers', 'd4': '(d) no zero-shot rows'}


def figure_from_files():
    """Draw the attenuation figure from the saved tables (re-drawable without re-estimating)."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    tabA = pd.read_csv(os.path.join(TAB, 'm5_progress_attenuation_panelA.csv'))
    eprof = pd.read_csv(os.path.join(TAB, 'm5_progress_E_profile.csv'))
    aer_style.use()
    fig, axs = plt.subplots(1, 3, figsize=(aer_style.WIDTH_FULL, 3.0), gridspec_kw=dict(width_ratios=[1.45, 1, 1]))
    ax = axs[0]
    t = tabA.iloc[::-1].reset_index(drop=True)
    yy = np.arange(len(t))
    ex = lambda c: [np.clip(t[c] - t[c + '_lo'], 0, None), np.clip(t[c + '_hi'] - t[c], 0, None)]
    ok = t.beta_data > 1e-4           # a degenerate boundary estimate (beta = 0) cannot be drawn on a log axis
    ax.errorbar(t.beta_data[ok], yy[ok] + 0.15, xerr=[e[ok] for e in ex('beta_data')], fmt='o',
                color=aer_style.ORANGE, ms=3, lw=0.8, label=r'$\hat\beta_{\mathrm{data}}$')
    ax.errorbar(t.alpha_param, yy - 0.15, xerr=ex('alpha_param'), fmt='s',
                color=aer_style.BLUE, ms=3, lw=0.8, label=r'$\hat\alpha_{\mathrm{param}}$')
    for yv in yy[~ok]:
        ax.text(0.0045, yv + 0.15, r'$\hat\beta=0$', fontsize=5.5, color=aer_style.ORANGE, va='center')
    ax.axvline(BESIROGLU.beta, color=aer_style.ORANGE, ls=':', lw=0.9)
    ax.axvline(BESIROGLU.alpha, color=aer_style.BLUE, ls=':', lw=0.9)
    ax.text(0.36, len(t) - 0.35, 'sweep', fontsize=6.3, ha='center', color=aer_style.INK2)
    ax.set_xscale('log'); ax.set_xlim(0.004, 1.2)
    ax.set_yticks(yy); ax.set_yticklabels([SHORT.get(c, c) for c in t.code], fontsize=6.2)
    ax.set_xlabel('exponent (log scale)')
    ax.set_title('(a) Data variants (converged L1)', loc='left')
    ax.legend(loc='lower left', fontsize=6.3, handletextpad=0.2)
    ax = axs[1]
    ax.plot(eprof.E, eprof.gamma, color=aer_style.BLUE, marker='o', ms=2.5, label=r'$\hat\gamma$')
    ax.plot(eprof.E, eprof.alpha_param, color=aer_style.INK2, lw=0.9, ls='--', label=r'$\hat\alpha_{\mathrm{param}}$')
    ax.plot(eprof.E, eprof.beta_data, color=aer_style.ORANGE, lw=0.9, ls='-.', label=r'$\hat\beta_{\mathrm{data}}$')
    ax.set_yscale('log')
    ax.set_yticks([0.02, 0.03, 0.05, 0.1, 0.15]); ax.set_yticklabels(['0.02', '0.03', '0.05', '0.10', '0.15'])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('imposed $E$ (nats/word)'); ax.set_ylabel('estimate (log scale)')
    ax.set_title('(b) Exponents rise with $E$ ...', loc='left')
    ax.legend(fontsize=6.5, loc='upper left')
    ax = axs[2]
    ax.plot(eprof.E, eprof.TC_months, color=aer_style.BLUE, marker='o', ms=2.5)
    ax.set_ylim(0, 20)
    ax.axhline(8.4, color=aer_style.ORANGE, ls=':', lw=0.9)
    ax.text(0.02, 7.3, 'Ho et al. 8.4', fontsize=6.5, color=aer_style.ORANGE)
    ax.set_xlabel('imposed $E$ (nats/word)'); ax.set_ylabel(r'$T_C$ (months)')
    ax.set_title(r'(c) ... but $T_C$ moves little', loc='left')
    fig.tight_layout(w_pad=0.5)
    aer_style.savefig(fig, 'm5_progress_attenuation')
