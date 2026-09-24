"""Task 4: allocative vs technical decomposition of "algorithmic progress" (paper Figure 6, Table 7 panel B).

For a model with inputs (N, D), training compute C = 6ND, and a *fixed* technology T (Chinchilla form), Farrell
input-oriented cost efficiency relative to the training-only frontier is
        CE_i = C_min^T( L^T(N_i, D_i) ) / (6 N_i D_i)  in (0, 1],
the share of compute that a cost-minimizing lab with the same technology would need to reach the loss the
technology predicts for (N_i, D_i). It depends only on (N_i, D_i) and on (A/B, alpha, beta) (E and loss units
cancel), so it isolates the *allocative* component: moving along an isoquant toward the expansion path.
ln(1/CE_i) is the log compute lost to misallocation. Deliberate over-training for inference (model_spec.md,
Proposition 4) also lowers CE relative to the training-only frontier; for those models CE is not an error.

Era comparison. For large models (C >= 1e23 FLOP) we compare GPT-3/Gopher-era releases (2020-2021) with
Chinchilla/Llama-era releases (2022-2024). The allocative gain is the ratio of geometric-mean CE. We compare it
with (i) the total algorithmic effective-compute gain implied by Ho et al.'s rate over the same interval and
(ii) Gundlach et al.'s (2025) accounting (Kaplan->Chinchilla rebalancing ~10x out of 6,930x at the frontier),
whose number is the counterfactual cost of continuing a Kaplan (2020) allocation rule at 2025-frontier compute.
"""
import json
import os
import re

import numpy as np
import pandas as pd

import zlib

from common import (BESIROGLU, HOFFMANN_RND, HOFFMANN_TEX, LN2, PROC, RAW, TAB, aer_style, log, pct, read_registry, sl)

ERA1 = ('2020-01-01', '2021-12-31')
ERA2 = ('2022-01-01', '2024-12-31')
C_LARGE = 1e23
MOE_RE = re.compile(r"MoE|Mixtral|DeepSeek-V[23]|DeepSeek-MoE|\bA\d+(?:\.\d+)?B\b|\d+x\d+B|Switch|GLaM|Arctic|DBRX|Jamba|Grok|"
                    r"Qwen\d?(?:\.\d)?-?\d*B?-?A\d|Mixture|Hunyuan-Large|Llama 4|gpt-oss|Kimi|GLM-4\.5|ERNIE 4\.5|MiniMax", re.I)
MAX_EPOCHS = 4.0     # Muennighoff et al. (2023): up to ~4 epochs repeated data is worth about as much as unique data
KAPLAN_A = 0.73      # Kaplan et al. (2020): compute-optimal N grows as C^0.73
GUNDLACH = dict(rebalancing=10.0, total=6930.0)   # Gundlach et al. (2025)
# Robustness sample ("cleaned", reviewer addition): drop derivative releases (instruct/chat/vision-language variants),
# systems-paper training runs, non-transformer architectures and a molecular model; exact duplicates of
# (Organization, N, C) (re-releases); D/N exactly 20 (tokens possibly imputed by a Chinchilla rule, which would make
# CE ~ 1 by construction); and Epoch 'Speculative' entries.
CLEAN_DROP_RE = re.compile(r"Instruct|Chat|-VL-|MegaScale|Mamba|FragLlama", re.I)


def cleaned_mask(ep):
    dup = ep.duplicated(subset=['Organization', 'N', 'C'], keep='first')
    rule20 = np.isclose(ep['M'], 20.0, rtol=0, atol=1e-3)
    odd = ep['Model'].fillna('').str.contains(CLEAN_DROP_RE)
    spec = ep['Confidence'].astype(str).eq('Speculative')
    return ~(dup | rule20 | odd | spec)


def load_epoch(path=None):
    """Epoch 'all AI models' language models with N, C and dataset size, 2018-01-01 .. 2026-09-23.
    D (tokens processed) = C/(6N), kept only if within a factor 3 of dataset size x epochs (internal consistency);
    excludes fine-tunes (non-empty 'Base model'), mixture-of-experts (name match; total vs active N) and models
    trained for more than MAX_EPOCHS epochs (the Chinchilla technology is estimated on unique tokens)."""
    path = path or os.path.join(RAW, 'epoch_models', 'all_ai_models.csv')
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['Publication date'], errors='coerce')
    df['N'] = pd.to_numeric(df['Parameters'], errors='coerce')
    df['C'] = pd.to_numeric(df['Training compute (FLOP)'], errors='coerce')
    df['U'] = pd.to_numeric(df['Training dataset size (total)'], errors='coerce')
    df['ep'] = pd.to_numeric(df['Epochs'], errors='coerce')
    lang = df['Domain'].fillna('').str.contains('Language')
    inwin = (df['date'] >= '2018-01-01') & (df['date'] <= '2026-09-23')
    base = df['Base model'].isna()
    ok = lang & inwin & (df['N'] > 0) & (df['C'] > 0)
    out = df[ok].copy()
    steps = dict(language_2018_2026_with_N_C=int(ok.sum()))
    out['moe'] = out['Model'].fillna('').str.contains(MOE_RE)
    out['D_imp'] = out['C'] / (6 * out['N'])
    out['D_rep'] = out['U'] * out['ep'].fillna(1.0)
    out['log_ratio'] = np.log10(out['D_imp'] / out['D_rep'])
    steps['with_dataset_size'] = int(out['U'].notna().sum())
    steps['not_finetune'] = int((out['U'].notna() & base[out.index]).sum())
    steps['not_moe'] = int((out['U'].notna() & base[out.index] & ~out['moe']).sum())
    keep = out['U'].notna() & base[out.index] & ~out['moe'] & (out['log_ratio'].abs() <= np.log10(3))
    steps['consistent_within_3x'] = int(keep.sum())
    keep &= ~(out['ep'] > MAX_EPOCHS)
    steps['at_most_4_epochs'] = int(keep.sum())
    out = out[keep].copy()
    out['D'] = out['D_imp']
    out['M'] = out['D'] / out['N']
    out['frontier_flag'] = out['Frontier model'].fillna(False).astype(bool)
    out['open'] = out['Open model weights?'].fillna('').astype(str).str.contains('True|Yes|yes|Open', regex=True)
    out['year'] = out['date'].dt.year + (out['date'].dt.dayofyear - 1) / 365.0
    # frontier proxy among models that disclose N and D: top-5 by training compute in each calendar year
    out['top5_year'] = out.groupby(out['date'].dt.year)['C'].rank(ascending=False, method='first') <= 5
    cols = ['Model', 'Organization', 'date', 'year', 'N', 'D', 'D_rep', 'C', 'M', 'Confidence', 'frontier_flag', 'top5_year', 'open', 'moe', 'log_ratio']
    return out[cols].sort_values('date').reset_index(drop=True), steps


def farseer_technology(cache=True):
    """Chinchilla form fitted (Besiroglu/Hoffmann Huber estimator) to Farseer's 404-run grid (1222_full.csv), with
    N = parameters including embeddings (comparable to Epoch's parameter counts) and D = training tokens."""
    path = os.path.join(PROC, 'farseer_chinchilla_fit.json')
    if cache and os.path.exists(path):
        return sl.Chinchilla(**{k: v for k, v in json.load(open(path)).items() if k in ('E', 'A', 'B', 'alpha', 'beta')})
    f = pd.read_csv(os.path.join(RAW, 'farseer', '1222_full.csv'))
    m = sl.fit_chinchilla(f['N_add_emb'].to_numpy(float), f['D'].to_numpy(float), f['loss'].to_numpy(float))
    json.dump(dict(m.summary(), n=int(len(f)), objective=m.extra.get('objective')), open(path, 'w'), indent=1)
    return m


def besiroglu_draws(B=200, seed=13):
    """Bootstrap draws of the Chinchilla technology on the Epoch extraction (pairs, warm start); cached."""
    path = os.path.join(PROC, 'besiroglu_boot_theta.npy')
    if os.path.exists(path):
        return np.load(path)
    df = sl.chinchilla_extraction(os.path.join(RAW, 'epoch_chinchilla', 'svg_extracted_data.csv'))
    full = sl.fit_chinchilla(df.N, df.D, df.L, init=BESIROGLU.theta)
    rng = np.random.default_rng(seed)
    th = []
    for _ in range(B):
        idx = rng.integers(0, len(df), len(df))
        dd = df.iloc[idx]
        th.append(sl.fit_chinchilla(dd.N, dd.D, dd.L, init=full.theta).theta)
    th = np.array(th)
    np.save(path, th)
    return th


def registry_chinchillas(regs=('technology_registry_m1.csv', 'technology_registry_m2.csv')):
    """Primary Huber Chinchilla technologies (E, A, B, alpha, beta) from modules m1 and m2's registries, one per
    dataset other than Chinchilla (already covered by Besiroglu/Hoffmann). Units of N and D follow each sweep's own
    conventions (see the registries), so CE under these technologies is indicative only. Returns (dict, info)."""
    techs, info = {}, {}
    for reg in regs:
        tr, info[reg] = read_registry(reg, need=('E', 'A', 'B', 'alpha', 'beta'))
        if not len(tr):
            continue
        tr = tr[~tr['dataset'].astype(str).str.startswith('chinchilla')]
        tag = reg[-6:-4]
        for _, r in tr.iterrows():
            sub = str(r.get('subset', ''))
            lab = f"{tag}: {r.dataset} ({sub})" if len(sub) <= 20 else f"{tag}: {r.dataset}"
            techs[lab] = sl.Chinchilla(E=float(r.E), A=float(r.A), B=float(r.B), alpha=float(r.alpha), beta=float(r.beta))
        info[reg]['technologies'] = [k for k in techs if k.startswith(tag)]
    return techs, info


def ho_technology(r1, year):
    """Ho et al. model-7 technology (WT103 units; E = 0) at a given date -- attenuated exponents."""
    return r1['model'].technology(r1['x'], year, bench='wt103')


def era_stats(df, CE, mask):
    lce = np.log(CE[mask])
    return dict(n=int(mask.sum()), gm_CE=float(np.exp(lce.mean())), med_CE=float(np.median(CE[mask])),
                gm_C=float(np.exp(np.log(df.C[mask]).mean())), med_M=float(np.median(df.M[mask])),
                mean_year=float(df.year[mask].mean()))


def kaplan_ce(T, C, anchor=(174.6e9, 3.14e23)):
    """CE of the Kaplan (2020) allocation rule N = N_GPT3 (C/C_GPT3)^0.73 (anchored at GPT-3) under technology T."""
    N = anchor[0] * (np.asarray(C, float) / anchor[1]) ** KAPLAN_A
    D = np.asarray(C, float) / (6 * N)
    return T.cost_efficiency(N, D), N, D


def run(pool, r1):
    ep, steps = load_epoch()
    ep.to_csv(os.path.join(PROC, 'epoch_lm_sample.csv'), index=False)
    log(f"s4: Epoch LM sample {steps}")
    far = farseer_technology()
    techs = {'Besiroglu': BESIROGLU, 'Hoffmann (TeX)': HOFFMANN_TEX, 'Hoffmann (rounded)': HOFFMANN_RND, 'Farseer grid': far}
    reg_techs, reg_info = registry_chinchillas()   # robustness: other modules' sweep technologies (if their registries exist)
    techs.update(reg_techs)
    for k, T in techs.items():
        ep[f'CE_{k}'] = T.cost_efficiency(ep.N, ep.D)
        ep[f'w_{k}'] = T.wedge(ep.N, ep.D)
    # Ho et al. technology: evaluated at each model's date (the A/B ratio drifts unless alpha_year = beta_year)
    ep['CE_Ho model 7'] = [ho_technology(r1, y).cost_efficiency(n, d) for y, n, d in zip(ep.year, ep.N, ep.D)]
    techs_all = list(techs) + ['Ho model 7']
    e1 = (ep.date >= ERA1[0]) & (ep.date <= ERA1[1])
    e2 = (ep.date >= ERA2[0]) & (ep.date <= ERA2[1])
    big = ep.C >= C_LARGE
    samples = {'C >= 1e23': big, 'Top-5 compute per year': ep.top5_year, 'Epoch frontier flag': ep.frontier_flag,
               'All language models': ep.C > 0,
               # appended last so that the bootstrap random stream of the rows above is unchanged
               'C >= 1e23, cleaned': big & cleaned_mask(ep)}
    thetas = besiroglu_draws()

    # Ho et al. algorithmic rate (point and paper-cluster bootstrap draws from s1)
    gC_ho = r1['model'].rates(r1['x'])['g_C']
    gC_draws = np.array([r1['model'].rates(b[:-1])['g_C'] for b in r1['boot_cl']])
    rows = []
    for sname, smask in samples.items():
        for tname in techs_all:
            CE = ep[f'CE_{tname}'].to_numpy()
            s1_, s2_ = era_stats(ep, CE, (e1 & smask).to_numpy()), era_stats(ep, CE, (e2 & smask).to_numpy())
            if s1_['n'] < 2 or s2_['n'] < 2:
                continue
            alloc = s2_['gm_CE'] / s1_['gm_CE']
            dt = s2_['mean_year'] - s1_['mean_year']
            algo = np.exp(gC_ho * dt)
            phys = s2_['gm_C'] / s1_['gm_C']
            # bootstrap: resample models within eras (and technology draws for Besiroglu); Ho rate draws
            i1, i2 = np.where((e1 & smask).to_numpy())[0], np.where((e2 & smask).to_numpy())[0]
            ab, sh = [], []
            # one deterministic random stream per (sample, technology): results do not depend on which other
            # technologies (e.g. registry rows) are present in a given run (reviewer fix)
            rng = np.random.default_rng([21, zlib.crc32(f"{sname}|{tname}".encode())])
            for b in range(400):
                j1, j2 = rng.choice(i1, len(i1)), rng.choice(i2, len(i2))
                if tname == 'Besiroglu':
                    T = sl.Chinchilla.from_theta(thetas[b % len(thetas)])
                    ce1, ce2 = T.cost_efficiency(ep.N.values[j1], ep.D.values[j1]), T.cost_efficiency(ep.N.values[j2], ep.D.values[j2])
                else:
                    ce1, ce2 = CE[j1], CE[j2]
                a_b = np.exp(np.log(ce2).mean() - np.log(ce1).mean())
                g_b = gC_draws[b % len(gC_draws)]
                ab.append(a_b)
                sh.append(np.log(a_b) / (g_b * dt) if g_b > 0 else np.nan)
            lo, _, hi = pct(ab)
            slo, _, shi = pct(sh)
            rows.append(dict(sample=sname, technology=tname, n_era1=s1_['n'], n_era2=s2_['n'], gmCE_era1=s1_['gm_CE'],
                             gmCE_era2=s2_['gm_CE'], medM_era1=s1_['med_M'], medM_era2=s2_['med_M'], years_between=dt,
                             allocative_gain=alloc, allocative_lo=lo, allocative_hi=hi, physical_compute_growth=phys,
                             ho_algorithmic_gain=algo, residual_technical=algo / alloc,
                             allocative_share_of_algorithmic=np.log(alloc) / np.log(algo), share_lo=slo, share_hi=shi))
    dec = pd.DataFrame(rows)
    dec.to_csv(os.path.join(TAB, 'm5_progress_table7_panelB.csv'), index=False)
    key = dec[(dec['sample'] == 'C >= 1e23') & (dec.technology == 'Besiroglu')].iloc[0]
    log(f"allocative gain (C>=1e23, Besiroglu): {key.allocative_gain:.2f} [{key.allocative_lo:.2f},{key.allocative_hi:.2f}] "
        f"of Ho-implied algorithmic {key.ho_algorithmic_gain:.1f}x over {key.years_between:.2f} yr -> share {key.allocative_share_of_algorithmic:.2f}")

    # named models
    names = ['GPT-3 175B (davinci)', 'Gopher (280B)', 'Megatron-Turing NLG 530B', 'Jurassic-1-Jumbo', 'Chinchilla', 'PaLM (540B)',
             'OPT-175B', 'BLOOM-176B', 'LLaMA-65B', 'Llama 2-70B', 'Llama 2-7B', 'Llama 3-8B', 'Llama 3-70B', 'Llama 3.1-405B',
             'Qwen2.5-72B', 'Falcon-180B']
    named = ep[ep.Model.isin(names)][['Model', 'date', 'N', 'D', 'C', 'M'] + [f'CE_{t}' for t in techs_all] + [f'w_{t}' for t in techs]]
    named.to_csv(os.path.join(TAB, 'm5_progress_named_models_CE.csv'), index=False)

    # Kaplan counterfactual (Gundlach comparison)
    Cs = np.array([3.14e23, 5.76e23, 2.5e24, 3.8e25, 1e26, 5e26, 1e27])
    kap = []
    for tname, T in techs.items():
        ce, N, D = kaplan_ce(T, Cs)
        for c, v, n, d in zip(Cs, ce, N, D):
            kap.append(dict(technology=tname, C=c, N_kaplan=n, D_kaplan=d, M_kaplan=d / n, CE=v, CEG=1 / v))
    kap = pd.DataFrame(kap)
    kap.to_csv(os.path.join(TAB, 'm5_progress_kaplan_counterfactual.csv'), index=False)
    ceg_grid = np.logspace(21, 27.5, 60)
    kapc = {t: 1 / kaplan_ce(T, ceg_grid)[0] for t, T in techs.items()}
    c10 = {}
    for t, v in kapc.items():
        above = np.where(v >= 10)[0]
        c10[t] = float(ceg_grid[above[0]]) if len(above) else None

    summ = dict(sample_steps=steps, n_sample=int(len(ep)), farseer=far.summary(), gC_ho=float(gC_ho), C10_kaplan=c10,
                gundlach_share=float(np.log(GUNDLACH['rebalancing']) / np.log(GUNDLACH['total'])), registries_used=reg_info,
                era1=ERA1, era2=ERA2, C_large=C_LARGE)
    json.dump(summ, open(os.path.join(PROC, 'allocative_summary.json'), 'w'), indent=1, default=float)
    ep.to_csv(os.path.join(PROC, 'epoch_lm_sample_CE.csv'), index=False)
    figure_from_files()
    return dict(ep=ep, dec=dec, kap=kap, named=named, summ=summ)


def figure_from_files():
    """Paper Figure 6, drawn from the saved processed files (re-drawable without re-estimating)."""
    import matplotlib.pyplot as plt
    ep = pd.read_csv(os.path.join(PROC, 'epoch_lm_sample_CE.csv'), parse_dates=['date'])
    dec = pd.read_csv(os.path.join(TAB, 'm5_progress_table7_panelB.csv'))
    far = farseer_technology(cache=True)
    techs = {'Besiroglu': BESIROGLU, 'Hoffmann (rounded)': HOFFMANN_RND, 'Farseer grid': far}
    ceg_grid = np.logspace(21, 27.5, 60)
    kapc = {t: 1 / kaplan_ce(T, ceg_grid)[0] for t, T in techs.items()}
    aer_style.use()
    fig, axs = plt.subplots(1, 3, figsize=(aer_style.WIDTH_FULL, 2.6), gridspec_kw=dict(width_ratios=[1.4, 1, 1.0]))
    # (a) CE over time
    ax = axs[0]
    ce = ep['CE_Besiroglu']
    big = ep.C >= C_LARGE
    ax.axvspan(pd.Timestamp(ERA1[0]), pd.Timestamp(ERA1[1]), color=aer_style.GRID, alpha=0.6, lw=0)
    ax.axvspan(pd.Timestamp(ERA2[0]), pd.Timestamp(ERA2[1]), color=aer_style.GRID, alpha=0.25, lw=0)
    ax.scatter(ep.date[~big], ce[~big], s=4, color=aer_style.MUTED, alpha=0.5, lw=0, label=r'$C<10^{23}$ FLOP')
    ax.scatter(ep.date[big], ce[big], s=9, color=aer_style.BLUE, alpha=0.85, lw=0, label=r'$C\geq10^{23}$ FLOP')
    for a, b in (ERA1, ERA2):
        msk = big & (ep.date >= a) & (ep.date <= b)
        ax.hlines(np.exp(np.log(ce[msk]).mean()), pd.Timestamp(a), pd.Timestamp(b), color=aer_style.ORANGE, lw=1.4)
    ax.plot([], [], color=aer_style.ORANGE, lw=1.4, label='era geometric mean ($C\\geq10^{23}$)')
    lab = {'GPT-3 175B (davinci)': ('GPT-3', (-18, -22)), 'Gopher (280B)': ('Gopher', (6, -16)),
           'Megatron-Turing NLG 530B': ('MT-NLG', (-8, -18)), 'Chinchilla': ('Chinchilla', (-30, 14)),
           'Llama 3-8B': ('Llama-3 8B', (8, -6)), 'Llama 3.1-405B': ('Llama-3.1 405B', (4, 14))}
    for _, r in ep.iterrows():
        if r.Model in lab:
            l, off = lab[r.Model]
            ax.annotate(l, (r.date, r['CE_Besiroglu']), textcoords='offset points', xytext=off, fontsize=6.2, color=aer_style.INK,
                        arrowprops=dict(arrowstyle='-', lw=0.4, color=aer_style.INK2))
    ax.set_yscale('log'); ax.set_ylim(0.02, 1.6)
    ax.set_yticks([0.03, 0.1, 0.3, 1]); ax.set_yticklabels(['0.03', '0.1', '0.3', '1'])
    ax.set_ylabel('cost efficiency $C_{\\min}(L)/6ND$')
    ax.set_title('(a) Allocative efficiency, Epoch LMs', loc='left')
    ax.legend(loc='lower left', fontsize=5.8, handletextpad=0.2, markerscale=1.3, labelspacing=0.25)
    ax.tick_params(axis='x', labelsize=7)
    # (b) Kaplan-rule counterfactual
    ax = axs[1]
    cols = {'Besiroglu': aer_style.BLUE, 'Hoffmann (rounded)': aer_style.ORANGE, 'Farseer grid': aer_style.AQUA}
    for t, c in cols.items():
        ax.plot(ceg_grid, kapc[t], color=c, label=t.replace(' grid', ''))
    ax.axhline(10, color=aer_style.INK2, ls=':', lw=0.9)
    ax.text(1.3e21, 12.5, 'Gundlach et al.: 10x', fontsize=6.0, color=aer_style.INK2)
    for nm, l, off in (('GPT-3 175B (davinci)', 'GPT-3', (-4, 6)), ('Gopher (280B)', 'Gopher', (4, -9))):
        r = ep[ep.Model == nm].iloc[0]
        ax.plot(r.C, 1 / r['CE_Besiroglu'], 'o', color=aer_style.BLUE, ms=3.5)
        ax.annotate(l, (r.C, 1 / r['CE_Besiroglu']), textcoords='offset points', xytext=off, fontsize=6.2,
                    ha='right' if off[0] < 0 else 'left')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('training compute $C$ (FLOP)'); ax.set_ylabel('compute-equivalent gain $1/CE$')
    ax.set_title('(b) Cost of the Kaplan rule', loc='left')
    ax.legend(fontsize=6.3, loc='upper left')
    # (c) decomposition of the era-1 -> era-2 algorithmic gain
    ax = axs[2]
    rows = []
    for smp, tag in (('C >= 1e23', r'$C\geq10^{23}$'), ('Top-5 compute per year', 'top-5/yr')):
        for tech, tl in (('Besiroglu', 'Besiroglu'), ('Hoffmann (TeX)', 'Hoffmann-TeX'), ('Farseer grid', 'Farseer')):
            r = dec[(dec['sample'] == smp) & (dec.technology == tech)]
            if len(r):
                rows.append((f"{tl}, {tag}", r.iloc[0]))
    yv = np.arange(len(rows))
    for i, (l, r) in enumerate(rows):
        al, tot = max(r.allocative_gain, 1.0), r.ho_algorithmic_gain
        ax.barh(i, np.log10(al), color=aer_style.ORANGE, height=0.55, label='allocative' if i == 0 else None)
        ax.barh(i, np.log10(tot) - np.log10(al), left=np.log10(al), color=aer_style.BLUE, height=0.55, alpha=0.8,
                label='residual' if i == 0 else None)
    ax.set_yticks(yv)
    ax.set_yticklabels([f"{l}\n{r.allocative_gain:.2f}x of {r.ho_algorithmic_gain:.1f}x" for l, r in rows], fontsize=5.8)
    ax.set_xlabel(r'$\log_{10}$ gain, 2020--21 to 2022--24')
    ax.set_xlim(0, 1.65)
    ax.set_title('(c) Allocative share of Ho-rate gain', loc='left')
    ax.set_ylim(len(rows) - 0.45, -1.35)       # inverted axis with an empty band on top for the legend
    ax.legend(fontsize=5.8, loc='upper right', ncol=2, handlelength=1.2, columnspacing=0.8)
    fig.tight_layout(w_pad=0.5)
    aer_style.savefig(fig, 'm5_progress_fig6_allocative')
