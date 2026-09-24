"""Assemble paper Table 7: algorithmic progress under alternative identifying assumptions (panel A) and the
allocative-vs-technical decomposition of the 2020-21 -> 2022-24 gain (panel B). Reads the CSVs written by s2-s4."""
import os

import numpy as np
import pandas as pd

from common import LN2, TAB, log, tex_label, write_tex


def _tc(v):
    if not np.isfinite(v):
        return r'$\infty$'
    return f"{v:.1f}" if v > 0 else r'$g_C<0$'


def _ci(lo, hi, nd=1):
    f = lambda v: (r'$\infty$' if not np.isfinite(v) else f"{v:.{nd}f}")
    return f"[{f(lo)}, {f(hi)}]"


def run():
    A = pd.read_csv(os.path.join(TAB, 'm5_progress_table7_panelA.csv'))
    att = pd.read_csv(os.path.join(TAB, 'm5_progress_attenuation_panelA.csv'))
    B = pd.read_csv(os.path.join(TAB, 'm5_progress_table7_panelB.csv'))
    # add the measurement-corrected row (effective data) from the attenuation table
    a3 = att[att.code == 'a3'].iloc[0]
    extra = dict(code='A12', label='D = effective data under repetition (Muennighoff), L1 penalty', n=a3.n,
                 alpha_param=a3.alpha_param, beta_data=a3.beta_data, gamma=a3.gamma, alpha_year=a3.alpha_year,
                 beta_year=a3.beta_year, g_C=a3.g_C, g_N=a3.alpha_year / a3.alpha_param,
                 g_D=a3.beta_year / a3.beta_data, TC_months=a3.TC_months, TC_lo=a3.TC_lo, TC_hi=a3.TC_hi, mse=a3.mse)
    A = pd.concat([A, pd.DataFrame([extra])], ignore_index=True)
    order = ['A1', 'A1c', 'A2', 'A3', 'A4', 'A5', 'A12', 'A10', 'A11', 'A6', 'A7', 'A8', 'A9'] + [c for c in A.code if str(c).startswith('R')]
    A = A.set_index('code').loc[[c for c in order if c in set(A.code)]].reset_index()
    A.to_csv(os.path.join(TAB, 'm5_progress_table7.csv'), index=False)
    short = {'A1': "Ho et al. model 7 (their code: default stopping rule)", 'A1c': 'Same objective, SLSQP run to convergence',
             'A2': 'Same model, unpenalized NLS', 'A3': "Hicks-neutral, L1 penalty (Ho model 12)", 'A4': 'Hicks-neutral, unpenalized NLS',
             'A5': 'Benchmark-specific $E_b$ estimated', 'A12': 'Effective data under repetition',
             'A10': r'Impose experimental $\gamma$ in total-loss units', 'A11': r'\quad same, Hicks-neutral',
             'A6': "Ho's year terms / experimental exponents", 'A7': r'Impose Besiroglu $(\alpha,\beta)$, $E_b$ free',
             'A8': r'\quad same, Hicks-neutral', 'A9': r'Impose Hoffmann $(\alpha,\beta)$, $E_b$ free'}
    body = [r"\multicolumn{8}{l}{\textit{Panel A. Effective-compute growth in Ho et al.'s data (231 model--benchmark rows)}} \\"]
    for _, r in A.iterrows():
        lab = short.get(r.code, tex_label(str(r.label)))
        mse = f"{r.mse:.4f}" if np.isfinite(r.mse) else '--'
        body.append(f"({r.code}) {lab} & {r.alpha_param:.3f} & {r.beta_data:.3f} & {r.gamma:.3f} & {r.g_N:.2f} & {r.g_D:.2f} & "
                    f"{_tc(r.TC_months)} {_ci(r.TC_lo, r.TC_hi)} & {mse} \\\\")
    body.append(r"\addlinespace")
    body.append(r"\multicolumn{8}{l}{\textit{Panel B. Allocative efficiency, Epoch language models with $C\geq 10^{23}$ FLOP: 2020--21 vs. 2022--24}} \\")
    body.append(r" & $n_1$, $n_2$ & $\overline{CE}_{2020\text{--}21}$ & $\overline{CE}_{2022\text{--}24}$ & allocative gain & "
                r"Ho-rate gain & \multicolumn{2}{c}{allocative share} \\")
    Bb = pd.concat([B[B['sample'] == 'C >= 1e23'], B[(B['sample'] == 'Top-5 compute per year') & (B.technology == 'Besiroglu')],
                    B[(B['sample'] == 'C >= 1e23, cleaned') & (B.technology == 'Besiroglu')]])
    for _, r in Bb.iterrows():
        tag = {'C >= 1e23': '', 'Top-5 compute per year': ' (top-5 per year)',
               'C >= 1e23, cleaned': ' (cleaned sample)'}[r['sample']]
        body.append(f"Technology: {tex_label(str(r.technology))}{tag} & {int(r.n_era1)}, {int(r.n_era2)} & {r.gmCE_era1:.2f} & {r.gmCE_era2:.2f} & "
                    f"{r.allocative_gain:.2f} {_ci(r.allocative_lo, r.allocative_hi, 2)} & {r.ho_algorithmic_gain:.1f} & "
                    f"\\multicolumn{{2}}{{c}}{{{r.allocative_share_of_algorithmic:.2f} {_ci(r.share_lo, r.share_hi, 2)}}} \\\\")
    hdr = [r" & $\hat\alpha$ & $\hat\beta$ & $\hat\gamma$ & $g_N$ & $g_D$ & $T_C$ (months) [95\% CI] & MSE \\"]
    notes = (r"Panel A: each row re-estimates Ho et al.'s factor-augmenting model, $\log\text{ppl}=E_b+A_b e^{-\alpha g_N t}N^{-\alpha}"
             r"+B_b e^{-\beta g_D t}D^{-\beta}$, under the stated restriction (L1-penalized rows other than A1 are run to convergence); $g_N$, $g_D$ are augmentation rates (log points per year), "
             r"$T_C=\ln 2/(g_N+g_D)$ the effective-compute doubling time, $\gamma=\alpha\beta/(\alpha+\beta)$. Intervals: 95\% paper-cluster "
             r"bootstrap (400 draws), mapped from the distribution of $g_N+g_D$ ($\infty$ when non-positive growth is inside the interval; "
             r"$g_C<0$: the point estimate implies that effective compute \emph{fell}). Hicks neutrality is $\alpha g_N=\beta g_D$. "
             r"Experimental $\gamma$ in total-loss units is the frontier elasticity obtained by fitting Ho's $E=0$ form to the Chinchilla "
             r"sweep (0.0525). Row A6 does not re-estimate: it divides Ho's year coefficients by Besiroglu et al.'s exponents (Whitfill 2025). "
             r"Panel B: $\overline{CE}$ is the geometric mean of Farrell cost efficiency $C_{\min}(L^T(N,D))/6ND$ under a fixed technology $T$ "
             r"(allocative only; $E$ and loss units cancel). Allocative gain $=\overline{CE}_{2022\text{--}24}/\overline{CE}_{2020\text{--}21}$ "
             r"with a bootstrap over models (and, for Besiroglu, over technology draws). Ho-rate gain $=\exp(g_C\Delta t)$ with Ho et al.'s "
             r"point estimate and $\Delta t$ the difference in mean release dates; allocative share $=\ln(\text{allocative gain})/\ln(\text{Ho-rate gain})$. "
             r"Cleaned sample: drops instruct/chat/vision-language derivatives, systems-paper runs, non-transformer and molecular models, "
             r"exact re-releases (same organization, $N$, $C$), models with $D/N$ exactly 20 (possibly imputed) and Epoch `speculative' entries.")
    write_tex(os.path.join(TAB, 'm5_progress_table7.tex'), body, 'lccccccc', hdr, notes,
              caption='Algorithmic progress under alternative identifying assumptions, and its allocative component',
              label='tab:progress', size='\\scriptsize')
    log('Table 7 written')
