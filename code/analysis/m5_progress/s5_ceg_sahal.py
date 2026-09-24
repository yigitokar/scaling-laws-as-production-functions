"""Tasks 5-6: compute-equivalent gains (CEG) with architecture-specific exponents, and the Sahal-type bias of
naive frontier regressions.

CEG.  Proposition P4 (SYNTHESIS appendix): for two Chinchilla technologies the CEG f(C) = C_old(L*_new(C))/C is
constant in C iff (alpha, beta, E) coincide; factor augmentation only rescales K and G. With E = 0 (Ho's form) the
frontier is L*_j(C) = K_j (C/6)^(-gamma_j) and
        f(C) = (K_old/K_new)^(1/gamma_old) (C/6)^(gamma_new/gamma_old - 1),
so ln f is linear in ln C with slope gamma_new/gamma_old - 1: scale-dependent iff gamma_new != gamma_old.
We estimate Ho et al.'s 'model 13' (transformer-specific alpha_param, beta_data; common year terms and constants)
and compute f(C) for transformer vs non-transformer technologies at 2023 on WT103, with paper-cluster bootstrap.

Sahal.  If frontier reducible loss falls at rate gamma (g + g_A) while physical compute grows at g, a regression of
log loss on log compute *without* a time control recovers gamma/(1 - s_A), s_A = g_A/(g + g_A) the algorithmic share
(Nagy et al. 2013 discuss the Sahal/Wright-Moore equivalence). We run the naive and time-controlled log-log
regressions on Ho et al.'s sample (all rows and record-setting releases).
"""
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import (B_BOOT, LN2, PROC, TAB, HoModel, HoSpec, aer_style, boot_indices, bootstrap_fit, fit, log, pct, sl,
                    write_tex)

C_GRID = np.logspace(14, 25, 45)
C_REPORT = [1e15, 1e17, 1e19, 1e21, 1e23]     # non-transformer support ends at 3e18 FLOP, transformer at 5.9e23
YEAR = 2023.0


def ceg_curve(model, x, C=C_GRID, year=YEAR):
    """CEG of the transformer technology over the non-transformer technology (same date, WT103), E = 0."""
    Tt = model.technology(x, year, bench='wt103', arch='t')
    Tn = model.technology(x, year, bench='wt103', arch='n')
    if not (Tt.alpha > 0 and Tt.beta > 0 and Tn.alpha > 0 and Tn.beta > 0):   # degenerate draw: no frontier
        return np.full(len(C), np.nan), Tt, Tn
    Lt = Tt.L_opt(C)
    with np.errstate(all='ignore'):
        f = np.real(Tn.C_min(Lt)) / np.asarray(C, float)
    return f, Tt, Tn


def run(pool, r1):
    df, d = r1['df'], r1['data']
    rows, curves = [], {}
    idxs = boot_indices(len(df), B_BOOT, 1313, clusters=df['paper'].to_numpy())
    for lab, spec in (('Model 13, L1 penalty (converged)', HoSpec(delta=0.0025, arch=True)),
                      ('Model 13, unpenalized NLS', HoSpec(delta=0, arch=True))):
        m = HoModel(spec, d)
        x, f = fit(m, n_random=8 if spec.delta == 0 else 0, seed=2)
        bt = bootstrap_fit(spec, d, m.norm, x, idxs, pool=pool, n_random=1 if spec.delta == 0 else 0)
        f0, Tt, Tn = ceg_curve(m, x)
        fb = np.array([ceg_curve(m, b[:-1])[0] for b in bt if np.all(np.isfinite(b))])
        slope = Tt.gamma / Tn.gamma - 1
        # inference on the difference in frontier elasticities (constant CEG iff gamma_t = gamma_nt when E = 0)
        sb = np.array([m.technology(b[:-1], YEAR, arch='t').gamma - m.technology(b[:-1], YEAR, arch='n').gamma
                       for b in bt if np.all(np.isfinite(b))])
        curves[lab] = (f0, fb)
        p = m.unpack(x)
        row = dict(spec=lab, n=len(df), n_transformer=int(df.transformer.sum()), alpha_nt=p['alpha_param'], beta_nt=p['beta_data'],
                   alpha_t=p['alpha_param_t'], beta_t=p['beta_data_t'], gamma_nt=Tn.gamma, gamma_t=Tt.gamma,
                   slope_lnCEG_lnC=slope, dgamma=Tt.gamma - Tn.gamma, dgamma_lo=pct(sb)[0], dgamma_hi=pct(sb)[2],
                   p_equal_gamma=float(2 * min(np.mean(sb <= 0), np.mean(sb >= 0))),
                   mse=float(np.mean((d['y'] - m.predict(x)) ** 2)))
        for c in C_REPORT:
            j = np.argmin(np.abs(np.log(C_GRID) - np.log(c)))
            row[f'CEG_{c:.0e}'] = f0[j]
            lo, _, hi = pct(fb[:, j])
            row[f'CEG_{c:.0e}_lo'], row[f'CEG_{c:.0e}_hi'] = lo, hi
        rows.append(row)
        log(f"CEG {lab}: gamma_t={Tt.gamma:.4f} gamma_nt={Tn.gamma:.4f} slope={slope:.3f}; dgamma CI [{pct(sb)[0]:.4f},{pct(sb)[2]:.4f}] "
            f"p={row['p_equal_gamma']:.2f}; CEG 1e15 {row['CEG_1e+15']:.2f}, 1e19 {row['CEG_1e+19']:.2f}, 1e23 {row['CEG_1e+23']:.2f}")
    ceg = pd.DataFrame(rows)
    ceg.to_csv(os.path.join(TAB, 'm5_progress_ceg.csv'), index=False)

    # ---------------------------------------------------------------- Sahal regressions
    s = df.copy()
    s['lnL'] = np.log(s['y'])
    s['lnC'] = np.log(6 * s['param'] * s['dataset'])
    s = s.sort_values('publication_date')
    # record-setting releases: best perplexity on the benchmark to date
    s['record'] = False
    for b in ('wt103', 'ptb', 'wt2'):
        sb_ = s[s.dataset_name == b]
        best = np.inf
        for i, r in sb_.iterrows():
            if r['y'] < best:
                s.loc[i, 'record'] = True
                best = r['y']
    sah = []
    for sample, mask in (('All rows (n=%d)' % len(s), np.ones(len(s), bool)), ('Record-setting releases', s['record'].to_numpy())):
        ss = s[mask]
        X0 = pd.DataFrame(dict(const=1.0, lnC=ss.lnC, ptb=ss.ptb_dummy, wt2=ss.wt2_dummy))
        X1 = X0.assign(year=ss.publication_date - 2012)
        g = ss['paper'].to_numpy()
        r0 = sm.OLS(ss.lnL, X0).fit(cov_type='cluster', cov_kwds=dict(groups=g))
        r1_ = sm.OLS(ss.lnL, X1).fit(cov_type='cluster', cov_kwds=dict(groups=g))
        rc = sm.OLS(ss.lnC, pd.DataFrame(dict(const=1.0, year=ss.publication_date - 2012, ptb=ss.ptb_dummy, wt2=ss.wt2_dummy))).fit()
        th0, th1, tau = -r0.params['lnC'], -r1_.params['lnC'], -r1_.params['year']
        gphys = rc.params['year']
        gA = tau / th1 if th1 > 0 else np.nan
        # paper-cluster bootstrap of the inflation factor
        rng = np.random.default_rng(77)
        papers = ss['paper'].unique()
        fac = []
        for _ in range(B_BOOT):
            pk = rng.choice(papers, len(papers))
            bb = pd.concat([ss[ss.paper == p_] for p_ in pk])
            try:
                a0 = -np.linalg.lstsq(np.c_[np.ones(len(bb)), bb.lnC, bb.ptb_dummy, bb.wt2_dummy], bb.lnL, rcond=None)[0][1]
                a1 = -np.linalg.lstsq(np.c_[np.ones(len(bb)), bb.lnC, bb.ptb_dummy, bb.wt2_dummy, bb.publication_date], bb.lnL, rcond=None)[0][1]
                fac.append(a0 / a1)
            except Exception:
                pass
        fac = np.array(fac)
        fac = fac[np.isfinite(fac)]
        sah.append(dict(sample=sample, n=len(ss), theta_naive=th0, se_naive=r0.bse['lnC'], theta_year=th1, se_year=r1_.bse['lnC'],
                        tau=tau, se_tau=r1_.bse['year'], inflation=th0 / th1, inflation_lo=pct(fac)[0], inflation_hi=pct(fac)[2],
                        sA_from_inflation=1 - th1 / th0, g_phys=gphys, g_A=gA, sA_from_rates=gA / (gphys + gA),
                        inflation_formula=(gphys + gA) / gphys, corr_lnC_year=float(np.corrcoef(ss.lnC, ss.publication_date)[0, 1])))
        log(f"Sahal {sample}: naive {th0:.4f} vs with year {th1:.4f} -> factor {th0 / th1:.2f} "
            f"[{pct(fac)[0]:.2f},{pct(fac)[2]:.2f}]; g_phys={gphys:.2f}/yr g_A={gA:.2f}/yr s_A={gA / (gphys + gA):.2f}")
    sah = pd.DataFrame(sah)
    sah.to_csv(os.path.join(TAB, 'm5_progress_sahal.csv'), index=False)
    s[['system', 'publication_date', 'dataset_name', 'ppl', 'record']].to_csv(os.path.join(PROC, 'sahal_records.csv'), index=False)
    write_latex(ceg, sah)
    np.savez(os.path.join(PROC, 'ceg_curves.npz'), C=C_GRID,
             **{f'f0_{i}': v[0] for i, v in enumerate(curves.values())}, **{f'fb_{i}': v[1] for i, v in enumerate(curves.values())},
             labels=np.array(list(curves.keys())))
    figure_from_files()
    return dict(ceg=ceg, sah=sah)


def write_latex(ceg, sah):
    body = [r"\multicolumn{7}{l}{\textit{Panel A. Compute-equivalent gain of transformers over other architectures, 2023, WT103}} \\"]
    body.append(r" & $\hat\gamma_{\text{non-tr}}$ & $\hat\gamma_{\text{tr}}$ & $\hat\gamma_{\text{tr}}-\hat\gamma_{\text{non-tr}}$ [95\% CI] & $f(10^{17})$ & $f(10^{19})$ & $f(10^{23})$ \\")
    for _, r in ceg.iterrows():
        body.append(f"{r['spec']} & {r.gamma_nt:.4f} & {r.gamma_t:.4f} & {r.dgamma:.4f} [{r.dgamma_lo:.3f}, {r.dgamma_hi:.3f}] & "
                    f"{r['CEG_1e+17']:.2f} & {r['CEG_1e+19']:.2f} & {r['CEG_1e+23']:.1f} \\\\")
    body.append(r"\addlinespace")
    body.append(r"\multicolumn{7}{l}{\textit{Panel B. Sahal-type bias: log loss on log compute with and without a time control}} \\")
    body.append(r" & $n$ & $\hat\theta$, no time & $\hat\theta$, with year & inflation & $\hat g$ (phys.) & implied $s_A$ \\")
    for _, r in sah.iterrows():
        body.append(f"{r['sample']} & {int(r.n)} & {r.theta_naive:.4f} ({r.se_naive:.4f}) & {r.theta_year:.4f} ({r.se_year:.4f}) & "
                    f"{r.inflation:.2f} [{r.inflation_lo:.2f}, {r.inflation_hi:.2f}] & {r.g_phys:.2f} & {r.sA_from_inflation:.2f} \\\\")
    hdr = [r" & (1) & (2) & (3) & (4) & (5) & (6) \\"]
    notes = (r"Panel A: Ho et al.'s model 13 (architecture-specific exponents, common factor-augmenting year terms, $E=0$) on their "
             r"231-row sample; $f(C)=C_{\text{non-tr}}(L^*_{\text{tr}}(C))/C$ is the compute a non-transformer lab needs to match the "
             r"transformer frontier loss at compute $C$ (FLOP); with $E=0$, $\ln f$ is linear in $\ln C$ with slope "
             r"$\gamma_{\text{tr}}/\gamma_{\text{non-tr}}-1$, zero iff the frontier elasticities coincide (constant CEG). Brackets: 95\% "
             r"paper-cluster bootstrap intervals (400 draws; draws without a well-defined frontier dropped). Non-transformer models in the "
             r"sample span $C\le 3\times10^{18}$ FLOP, so $f$ above that is an extrapolation of the non-transformer law. Panel B: OLS of $\ln L$ (log of log perplexity) on $\ln(6ND)$ and benchmark "
             r"dummies, without and with a linear publication-year trend; paper-clustered standard errors in parentheses; inflation "
             r"$=\hat\theta_{\text{no time}}/\hat\theta_{\text{year}}$ with 95\% paper-cluster bootstrap interval; $\hat g$ is the within-sample "
             r"growth of log compute per year; implied algorithmic share $s_A=1-\hat\theta_{\text{year}}/\hat\theta_{\text{no time}}$. "
             r"Record-setting releases: best perplexity on the benchmark to date.")
    write_tex(os.path.join(TAB, 'm5_progress_ceg_sahal.tex'), body, 'lcccccc', hdr, notes,
              caption='Scale-dependent compute-equivalent gains and the Sahal bias', label='tab:m5_ceg_sahal', size='\\footnotesize')


def figure_from_files():
    import matplotlib.pyplot as plt
    z = np.load(os.path.join(PROC, 'ceg_curves.npz'), allow_pickle=False)
    C = z['C']
    labels = list(z['labels'])
    aer_style.use()
    fig, ax = plt.subplots(figsize=(aer_style.WIDTH_HALF + 0.4, 2.6))
    ax.axvspan(C.min(), 3.05e18, color=aer_style.GRID, alpha=0.7, lw=0)
    ax.text(2e14, 0.013, 'non-transformer\nsupport', fontsize=6, color=aer_style.INK2, va='bottom')
    cols = [aer_style.BLUE, aer_style.ORANGE]
    for i, (lab, c) in enumerate(zip(labels, cols)):
        f0, fb = z[f'f0_{i}'], z[f'fb_{i}']
        with np.errstate(all='ignore'):
            lo, hi = np.nanpercentile(fb, 5, axis=0), np.nanpercentile(fb, 95, axis=0)
        ax.plot(C, f0, color=c, label=lab.replace('Model 13, ', ''))
        ax.plot(C, lo, color=c, lw=0.6, ls='--')
        ax.plot(C, hi, color=c, lw=0.6, ls='--')
    ax.plot([], [], color=aer_style.INK2, lw=0.6, ls='--', label='5th/95th bootstrap pct. (mostly off-scale)')
    ax.axhline(7.2, color=aer_style.INK2, ls=':', lw=0.9)
    ax.text(3e19, 8.3, 'Ho et al. (2024): 7.2x', fontsize=6.0, color=aer_style.INK2)
    ax.axhline(1, color=aer_style.INK2, lw=0.6)
    # Gundlach et al.'s small-scale LSTM->transformer CEG (6.28x, 3.6M-parameter models); the compute level is ASSUMED
    # (C = 6 N D with D = 20 N), since the ablations are reported at a fixed loss, not a fixed FLOP count
    ax.plot([6 * 3.6e6 * 20 * 3.6e6], [6.28], marker='^', color=aer_style.AQUA, ms=5, ls='none', label='Gundlach et al., small scale (C assumed)')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(0.01, 1e3)
    ax.set_xlabel('training compute $C$ (FLOP)'); ax.set_ylabel('CEG, transformer vs. other (2023)')
    ax.legend(fontsize=5.8, loc='upper left', labelspacing=0.3)
    fig.tight_layout()
    aer_style.savefig(fig, 'm5_progress_ceg')
