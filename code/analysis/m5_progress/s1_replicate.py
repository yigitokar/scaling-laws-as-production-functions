"""Task 1: reproduce Ho et al. (2024) preferred model 7 (Table 2; effective-compute doubling 8.4 months [4.5, 14.3]).

Their estimator: minimize mean squared residual of log perplexity + 0.0025*sum|theta| by SLSQP from theta = 0;
sample = top-3 models (by perplexity) per paper, 231 model-benchmark rows; inference = 100 iid bootstrap draws
(np.random.seed(0)); the headline doubling time is the *bootstrap median* of T_C = (1/T_N + 1/T_D)^-1.
"""
import json
import os

import numpy as np
import pandas as pd

from common import (HO_M7_FUN, HO_M7_X, HO_TABLE2, HO_TC_NB, B_BOOT, PROC, TAB, HoModel, HoSpec, boot_indices,
                    bootstrap_fit, fit, fmt, ho_arrays, load_ho, log, pct, tc_q, write_tex)

NAMES7 = HoSpec().names()


def tc_months(arr, model):
    return np.array([model.rates(x)['TC_months'] for x in arr])


def gc_draws(arr, model):
    return np.array([model.rates(x)['g_C'] for x in arr])


def run(pool):
    df = load_ho()
    df.to_csv(os.path.join(PROC, 'ho_df_head.csv'), index=False)
    d = ho_arrays(df)
    spec = HoSpec(delta=0.0025, label='Ho model 7')
    m = HoModel(spec, d)
    x, fval = fit(m, method='ho')
    log(f"model 7: n={len(df)} objective={fval:.10f} (Ho: {HO_M7_FUN:.10f}); max|theta-theta_Ho|={np.abs(x - HO_M7_X).max():.2e}")
    r = m.rates(x)
    log(f"point-estimate T_C = {r['TC_months']:.3f} months; g_C = {r['g_C']:.4f}/yr")

    # (i) Ho et al.'s own bootstrap protocol: 100 iid draws, np.random.seed(0), SLSQP from zeros
    idx_ho = boot_indices(len(df), 100, 0, ho_style=True)
    bt_ho = bootstrap_fit(spec, d, m.norm, None, idx_ho, method='ho', pool=pool)
    # (ii) same protocol, 1,000 draws
    idx_ho1k = boot_indices(len(df), 1000, 0, ho_style=True)
    bt_ho1k = bootstrap_fit(spec, d, m.norm, None, idx_ho1k, method='ho', pool=pool)
    # (iii) our default: cluster (paper) bootstrap, B_BOOT draws
    idx_cl = boot_indices(len(df), B_BOOT, 20260923, clusters=df['paper'].to_numpy())
    bt_cl = bootstrap_fit(spec, d, m.norm, None, idx_cl, method='ho', pool=pool)
    # (iv) the same objective and algorithm run to convergence (SLSQP ftol=1e-13): point estimate + both bootstraps
    #      (bootstrap draws start at zero and at the full-sample optimum and keep the lower objective)
    xc, fc = fit(m, method='ho_conv')
    rc = m.rates(xc)
    bt_c1k = bootstrap_fit(spec, d, m.norm, xc, idx_ho1k, method='ho_conv', pool=pool)
    bt_ccl = bootstrap_fit(spec, d, m.norm, xc, idx_cl, method='ho_conv', pool=pool)
    np.save(os.path.join(PROC, 'boot_m7_conv_iid1000.npy'), bt_c1k)
    np.save(os.path.join(PROC, 'boot_m7_conv_cluster.npy'), bt_ccl)
    log(f"converged Ho objective: {fc:.7f} (default-tolerance {fval:.7f}); T_C = {rc['TC_months']:.2f} months; "
        f"alpha={rc['alpha_param']:.3f} beta={rc['beta_data']:.3f}")
    np.save(os.path.join(PROC, 'boot_m7_ho100.npy'), bt_ho)
    np.save(os.path.join(PROC, 'boot_m7_ho1000.npy'), bt_ho1k)
    np.save(os.path.join(PROC, 'boot_m7_cluster.npy'), bt_cl)

    rows = []
    for j, nme in enumerate(NAMES7):
        rep = HO_TABLE2.get(nme, (np.nan, np.nan, np.nan))
        rows.append(dict(parameter=nme, ho_est=rep[0], ho_lo=rep[1], ho_hi=rep[2], est=x[j],
                         iid100_lo=pct(bt_ho[:, j])[0], iid100_hi=pct(bt_ho[:, j])[2],
                         iid1000_lo=pct(bt_ho1k[:, j])[0], iid1000_hi=pct(bt_ho1k[:, j])[2],
                         cl_lo=pct(bt_cl[:, j])[0], cl_hi=pct(bt_cl[:, j])[2], cl_se=np.nanstd(bt_cl[:, j], ddof=1),
                         conv=xc[j], conv_cl_lo=pct(bt_ccl[:, j])[0], conv_cl_hi=pct(bt_ccl[:, j])[2],
                         conv_iid_lo=pct(bt_c1k[:, j])[0], conv_iid_hi=pct(bt_c1k[:, j])[2]))
    boots = (('iid100', bt_ho), ('iid1000', bt_ho1k), ('cl', bt_cl), ('conv_cl', bt_ccl), ('conv_iid', bt_c1k))
    gcs = {k: gc_draws(b[:, :-1], m) for k, b in boots}
    # T_C percentiles. Ho et al. take percentiles of T_C directly; we do the same for their protocol (iid, B=100: no
    # draw has g_C <= 0, so this reproduces their method). For every other column we map the g_C percentiles
    # (a draw with g_C <= 0 has an infinite doubling time, not a negative one).
    q = {k: tc_q(v) for k, v in gcs.items()}
    q['iid100'] = pct(tc_months(bt_ho[:, :-1], m))
    n_nonpos = {k: int(np.sum(v <= 0)) for k, v in gcs.items()}
    mse_pub = float(np.mean(m.resid(x) ** 2))
    mse_conv = float(np.mean(m.resid(xc) ** 2))
    rows.append(dict(parameter='T_C (months)', ho_est=HO_TC_NB[0], ho_lo=HO_TC_NB[1], ho_hi=HO_TC_NB[2], est=r['TC_months'],
                     iid100_lo=q['iid100'][0], iid100_hi=q['iid100'][2], iid1000_lo=q['iid1000'][0], iid1000_hi=q['iid1000'][2],
                     cl_lo=q['cl'][0], cl_hi=q['cl'][2], cl_se=np.nan, conv=rc['TC_months'], conv_cl_lo=q['conv_cl'][0],
                     conv_cl_hi=q['conv_cl'][2], conv_iid_lo=q['conv_iid'][0], conv_iid_hi=q['conv_iid'][2]))
    rows.append(dict(parameter='T_C bootstrap median', ho_est=HO_TC_NB[0], est=q['iid100'][1],
                     iid100_lo=np.nan, iid1000_lo=q['iid1000'][1], cl_lo=q['cl'][1], conv=q['conv_cl'][1]))
    rows.append(dict(parameter='Objective', ho_est=HO_M7_FUN, est=fval, conv=fc))
    rows.append(dict(parameter='MSE (unpenalized part)', est=mse_pub, conv=mse_conv))
    rows.append(dict(parameter='sum|theta| (L1 term / 0.0025)', est=float(np.abs(x).sum()), conv=float(np.abs(xc).sum())))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(TAB, 'm5_progress_replication.csv'), index=False)

    corr = {k: float(np.corrcoef(b[:, 3], b[:, 8])[0, 1]) for k, b in (('iid100', bt_ho), ('iid1000', bt_ho1k), ('cl', bt_cl))}
    above = {k: float(np.mean(b[:, 8] > -b[:, 3])) for k, b in (('iid100', bt_ho), ('iid1000', bt_ho1k), ('cl', bt_cl))}
    summ = dict(n=int(len(df)), n_papers=int(df['Reference'].nunique()), n_transformer=int(df['transformer'].sum()),
                objective=fval, objective_ho=HO_M7_FUN, max_abs_diff=float(np.abs(x - HO_M7_X).max()),
                TC_point=r['TC_months'], TC_boot_iid100=list(q['iid100']), TC_boot_iid1000=list(q['iid1000']),
                TC_boot_cluster=list(q['cl']), corr_ay_by=corr, share_above_antidiag=above,
                conv_objective=fc, conv_theta=list(map(float, xc)), conv_TC=rc['TC_months'], conv_TC_boot_cluster=list(q['conv_cl']),
                conv_TC_boot_iid1000=list(q['conv_iid']), conv_rates={k: float(v) for k, v in rc.items()},
                conv_corr_ay_by_cluster=float(np.corrcoef(bt_ccl[:, 3], bt_ccl[:, 8])[0, 1]),
                g_N=r['g_N'], g_D=r['g_D'], g_C=r['g_C'], gamma=r['gamma'], ec_per_year=r['ec_per_year'],
                n_draws_gC_nonpositive=n_nonpos, TC_interval_method='iid100: direct T_C percentiles (Ho et al.); '
                'all others: g_C percentiles mapped to T_C (g_C<=0 -> inf)',
                mse_published=mse_pub, mse_converged=mse_conv, l1_published=float(np.abs(x).sum()),
                l1_converged=float(np.abs(xc).sum()),
                n_rows_ppl_quirk=int((np.abs(df['ppl_bench'] - df['ppl']) > 1e-9).sum()), theta=list(map(float, x)))
    json.dump(summ, open(os.path.join(PROC, 'replication_summary.json'), 'w'), indent=1)
    log(f"T_C bootstrap (Ho protocol, 100 draws): {q['iid100'].round(2)}  (Ho: 8.44 [4.52, 14.27]); 1000 draws: {q['iid1000'].round(2)};"
        f" paper-cluster {len(idx_cl)}: {q['cl'].round(2)}")

    # ---- LaTeX
    lab = {'alpha_const': r'$\alpha_{\text{const}}$', 'alpha_const_ptb': r'$\alpha_{\text{const}}^{PTB}$',
           'alpha_const_wt2': r'$\alpha_{\text{const}}^{WT2}$', 'alpha_year': r'$\alpha_{\text{year}}$',
           'alpha_param': r'$\alpha_{\text{param}}$', 'beta_const': r'$\beta_{\text{const}}$',
           'beta_const_ptb': r'$\beta_{\text{const}}^{PTB}$', 'beta_const_wt2': r'$\beta_{\text{const}}^{WT2}$',
           'beta_year': r'$\beta_{\text{year}}$', 'beta_data': r'$\beta_{\text{data}}$', 'T_C (months)': r'$T_C$ (months)'}
    body = []
    for rr in rows[:-4]:     # parameters and T_C (the last four rows are summary rows written below)
        p = rr['parameter']
        nd = 2 if p.startswith('T_C') else 3
        ho = (f"{fmt(rr['ho_est'], nd)}" if np.isfinite(rr['ho_est']) else '--')
        hoci = (f"[{fmt(rr['ho_lo'], nd)}, {fmt(rr['ho_hi'], nd)}]" if np.isfinite(rr.get('ho_lo', np.nan)) else '')
        body.append(f"{lab[p]} & {ho} & {hoci} & {fmt(rr['est'], nd)} & [{fmt(rr['iid100_lo'], nd)}, {fmt(rr['iid100_hi'], nd)}] "
                    f"& [{fmt(rr['cl_lo'], nd)}, {fmt(rr['cl_hi'], nd)}] & {fmt(rr['conv'], nd)} & [{fmt(rr['conv_cl_lo'], nd)}, {fmt(rr['conv_cl_hi'], nd)}] \\\\")
    body.append(r"\addlinespace")
    body.append(f"$T_C$, bootstrap median & {HO_TC_NB[0]:.2f} & & & {q['iid100'][1]:.2f} & {q['cl'][1]:.2f} & & {q['conv_cl'][1]:.2f} \\\\")
    body.append(f"corr$(\\alpha_{{\\text{{year}}}},\\beta_{{\\text{{year}}}})$ & & & & {corr['iid100']:.2f} & {corr['cl']:.2f} & & "
                f"{np.corrcoef(bt_ccl[:, 3], bt_ccl[:, 8])[0, 1]:.2f} \\\\")
    body.append(f"Objective (MSE $+0.0025\\sum|\\theta|$) & {HO_M7_FUN:.5f} & & {fval:.5f} & & & {fc:.5f} & \\\\")
    body.append(f"\\quad MSE (fit term) & & & {mse_pub:.5f} & & & {mse_conv:.5f} & \\\\")
    body.append(f"\\quad $\\sum|\\theta|$ (penalty term $=0.0025\\sum|\\theta|$) & & & {np.abs(x).sum():.3f} & & & {np.abs(xc).sum():.3f} & \\\\")
    hdr = [r" & \multicolumn{2}{c}{Ho et al. (2024)} & \multicolumn{3}{c}{Replication (their code)} & \multicolumn{2}{c}{Converged} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-8}",
           r" & Estimate & 95\% CI & Estimate & iid, $B=100$ & cluster & Estimate & cluster \\"]
    notes = (f"Ho et al. (2024) preferred specification (model 7): log perplexity $= \\exp(\\alpha_{{\\text{{const}}}}^{{b}} - \\alpha_{{\\text{{year}}}}(t-t_0) "
             f"- \\alpha_{{\\text{{param}}}}\\ln(N/N_0)) + \\exp(\\beta_{{\\text{{const}}}}^{{b}} - \\beta_{{\\text{{year}}}}(t-t_0) - \\beta_{{\\text{{data}}}}\\ln(D/D_0))$, "
             f"benchmark $b\\in\\{{$WT103, PTB, WT2$\\}}$, irreducible loss set to zero. Estimator: mean squared residual plus "
             f"$0.0025\\sum|\\theta|$, SLSQP from $\\theta=0$, exactly as in the authors' code (commit 29c7d85). $n={len(df)}$ model--benchmark rows "
             f"from {df['Reference'].nunique()} papers (top three models per paper). Objective at optimum {fval:.7f} (authors: {HO_M7_FUN:.7f}); "
             f"largest absolute parameter difference {np.abs(x - HO_M7_X).max():.1e}. Columns 5, 6 and 8 are 95\\% bootstrap percentile intervals: "
             f"iid resampling of rows with the authors' random-number sequence (\\texttt{{np.random.seed(0)}}; $B=100$ is their protocol), "
             f"and resampling of papers (clusters, {len(idx_cl)} draws). $T_C=\\ln 2/(\\alpha_{{\\text{{year}}}}/\\alpha_{{\\text{{param}}}}+\\beta_{{\\text{{year}}}}/\\beta_{{\\text{{data}}}})$ is the "
             f"effective-compute doubling time (months); the published headline (8.4 months) is the bootstrap median, whereas the point estimate is "
             f"{r['TC_months']:.2f} months. $T_C$ intervals: percentiles of $T_C$ for the authors' protocol (as in their code), otherwise "
             f"percentiles of $g_C$ mapped to $T_C$ (a draw with $g_C\\le 0$ has an infinite doubling time; "
             f"{n_nonpos['conv_cl']} of {len(idx_cl)} converged cluster draws). Converged: identical objective and algorithm with SLSQP run to convergence "
             f"(tolerance $10^{{-13}}$ instead of the scipy default $10^{{-6}}$ used by the authors' code); bootstrap draws start at zero and at the "
             f"full-sample optimum and keep the lower objective. The last two rows split the objective into its fit (MSE) and penalty parts.")
    write_tex(os.path.join(TAB, 'm5_progress_replication.tex'), body, 'lccccccc', hdr, notes,
              caption='Replication of Ho et al. (2024), preferred model', label='tab:m5_replication')
    return dict(df=df, data=d, model=m, x=x, boot_ho=bt_ho, boot_ho1k=bt_ho1k, boot_cl=bt_cl, idx_cl=idx_cl, summary=summ,
                x_conv=xc, boot_conv_cl=bt_ccl, boot_conv_1k=bt_c1k)
