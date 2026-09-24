"""common.py -- shared helpers for module m5_progress ("algorithmic progress" as TFP growth).

Contents
  * paths and published technologies (Besiroglu et al. 2024 refit, Hoffmann et al. 2022 TeX values)
  * load_ho(): exact re-implementation of the Ho et al. (2024) data pipeline (their section3.ipynb, commit 29c7d85)
  * HoSpec / HoModel: a generalized version of Ho et al.'s "augmented Chinchilla" in their own parameterization

        log_ppl_i = E_b + exp(a_b - alpha_year*(t-t0) - alpha*ln(N/N0)) + exp(b_b - beta_year*(t-t0) - beta*ln(D/D0))

    (Ho et al. Eq. 8; b = benchmark WT103/PTB/WT2). In the notation of paper/notes/model_spec.md, the
    two year terms are factor-augmenting productivity: psi_N,t = g_N t with g_N = alpha_year/alpha and
    psi_D,t = g_D t with g_D = beta_year/beta. Effective-compute growth is g_C = g_N + g_D (per year) and
    the effective-compute doubling time is T_C = ln2 / g_C (Ho et al. Eq. 5).
    Options: E_b (none / common / by benchmark / fixed), exponents free or fixed, Hicks neutrality
    (alpha_year = beta_year, i.e. alpha g_N = beta g_D = proportional shift of A and B), transformer-specific
    exponents (Ho model 13), vocabulary control (Ho App. E.2.2), and the L1 penalty delta*sum|theta| that
    Ho et al. add to the mean squared residual (their `residuals` function, delta = 0.0025 for model 7).
  * fit(): SLSQP from zeros (Ho et al.'s exact procedure) or multi-start L-BFGS-B (our unpenalized NLS)
  * bootstrap helpers (iid rows as in Ho et al.; clustered by paper as our default) with a process pool.
"""
from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "code", "analysis"))
import sl  # noqa: E402  (shared library; not edited)
import aer_style  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed", "m5_progress")
TAB = os.path.join(ROOT, "output", "tables")
FIG = os.path.join(ROOT, "output", "figures")
MEMO = os.path.join(ROOT, "output", "memos")
for d in (PROC, TAB, FIG, MEMO):
    os.makedirs(d, exist_ok=True)

N_PROC = 6          # at most 6 CPU processes (GPU job running on this machine)
B_BOOT = 400        # bootstrap draws for our estimates
LN2 = np.log(2.0)

# Published technologies (loss = E + A N^-alpha + B D^-beta); see SYNTHESIS.md section 2.1
BESIROGLU = sl.BESIROGLU            # E 1.8172, A 482.01, B 2085.43, alpha 0.3478 (0.02), beta 0.3658 (0.02)
HOFFMANN_TEX = sl.HOFFMANN          # E 1.6934, A 406.4, B 410.7, alpha 0.3392, beta 0.2849
HOFFMANN_RND = sl.Chinchilla(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28)
BESIROGLU_SE = dict(alpha=0.02, beta=0.02)   # Besiroglu et al. (2024) Table 1 (rounded)

# Ho et al. (2024) Table 2, model 7 (their bootstrap 95% CIs)
HO_TABLE2 = dict(alpha_const=(0.913, 0.000, 1.208), alpha_year=(0.004, -0.058, 0.032), alpha_param=(0.068, 0.045, 0.127),
                 beta_const=(0.771, 0.233, 1.293), beta_year=(0.036, -0.002, 0.080), beta_data=(0.040, 0.023, 0.062))
HO_TC = (8.4, 4.5, 14.3)        # months, bootstrap median [2.5, 97.5] (their section3.ipynb prints 8.444 [4.518, 14.269])
HO_TC_NB = (8.44382833, 4.51787288, 14.26941928)
HO_M7_FUN = 0.05181292650905238
HO_M7_X = np.array([0.912701, 0.000379, 0.055412, 0.003938, 0.067982, 0.771409, 0.176415, 0.095418, 0.035620, 0.039587])

WORD_VOCABS = {10000, 33000, 33278, 260000, 267000, 267735, 267744, 268000}   # standard word-level vocabularies


# ============================================================================ data
def _fraction_of_year(d):
    return d.year + (d.dayofyear - 1) / 365.0


def load_ho(path=None, keep_all=False, fix_ppl=False, include_outliers=False, top_k=3):
    """Exact re-implementation of Ho et al.'s pipeline (section3.ipynb cells 8-13).

    Returns the 'df_head' estimation sample (top_k models per paper, ranked by perplexity) or, with
    keep_all=True, their 'df_cluster' sample (all models). Extra columns we add: 'ppl_bench'
    (benchmark-specific perplexity), 'log_ppl_bench', 'epochs_num', 'transformer', 'word_vocab', 'vocab'.

    Quirk reproduced on purpose (fix_ppl=False): Ho et al. regress on log(ppl) where ppl = WT103 value if
    present, else WT2, else PTB -- also in the WT2/PTB rows of models evaluated on several benchmarks
    (4 rows in the 231-row sample, e.g. GPT-2 1542M's PTB row carries its WT103 perplexity).
    fix_ppl=True uses the benchmark-specific perplexity instead.
    """
    path = path or os.path.join(RAW, "ho2024", "algorithmic_progress.csv")
    raw = pd.read_csv(path)
    df = raw.drop(columns=['Author(s)', 'Link', 'Hardware', 'Base Model', 'GitHub', 'Comments',
                           'Organizations', 'Organization Categorization'])
    df = df.rename(columns={'Publication date': 'publication_date', 'Parameters': 'param', 'Perplexity (WT103)': 'ppl_wt103',
                            'Perplexity (WT2)': 'ppl_wt2', 'Perplexity (PTB)': 'ppl_ptb', 'Dataset Size': 'dataset',
                            'System': 'system', 'Epoch': 'epoch', 'Include?': 'include', 'Zero-shot?': 'zero_shot',
                            'Citations': 'cites', 'Peer reviewed?': 'peer_reviewed', 'Outlier?': 'outlier'})
    df['publication_date'] = pd.to_datetime(df['publication_date'], format='%Y/%m/%d', errors='coerce').apply(_fraction_of_year)
    df['param'] = pd.to_numeric(df['param'], errors='coerce')
    df['dataset'] = pd.to_numeric(df['dataset'], errors='coerce')
    df = df.dropna(subset=['param', 'dataset'])
    df = df.loc[(df['param'] > 0) & (df['dataset'] > 0)]
    df = df.loc[df['include'] != 0]                  # NB: include == NaN is kept (as in Ho et al.)
    if not include_outliers:
        df = df.loc[df['outlier'] != 1]
    df = df.dropna(subset=['ppl_wt103', 'ppl_wt2', 'ppl_ptb'], how='all').reset_index(drop=True)
    df['ppl'] = df['ppl_wt103'].fillna(df['ppl_wt2']).fillna(df['ppl_ptb'])
    with np.errstate(divide='ignore'):
        for c in ['param', 'dataset', 'ppl_wt103', 'ppl_wt2', 'ppl_ptb', 'ppl']:
            df[f'log_{c}'] = np.where(df[c] != 0, np.log(df[c]), np.nan)
    parts = [df.dropna(subset=[f'log_ppl_{b}']).assign(dataset_name=b) for b in ('wt103', 'wt2', 'ptb')]
    df1 = pd.concat(parts)
    df1['ptb_dummy'] = (df1['dataset_name'] == 'ptb').astype(int)
    df1['wt2_dummy'] = (df1['dataset_name'] == 'wt2').astype(int)
    df1 = df1[df1['uncertain'] == 0]
    cc = ['log_param', 'log_dataset', 'publication_date', 'ppl']
    df1 = df1[~df1[cc].replace([np.inf, -np.inf], np.nan).isnull().any(axis=1)]
    df1 = df1.sort_values(['Reference', 'ppl'], ascending=[True, True])
    for s in ["GPT3-6.7B + muP", "LLaMA-65B (LoRA finetuned)", "LLaMA-13B (LoRA finetuned)", "LLaMA-7B (LoRA finetuned)"]:
        df1 = df1.drop(df1[df1['system'] == s].index)
    df1.loc[df1['system'] == 'Gopher (280B)', 'param'] = 280e9
    df1.loc[df1['system'] == 'Gopher (7.1B)', 'param'] = 7.1e9
    out = df1.copy() if keep_all else df1.groupby('Reference').head(top_k)
    out = out.reset_index(drop=True)
    # --- added columns (not used by Ho et al.)
    out['ppl_bench'] = np.select([out.dataset_name == 'wt103', out.dataset_name == 'wt2'], [out.ppl_wt103, out.ppl_wt2], out.ppl_ptb)
    out['log_ppl_bench'] = np.log(out['ppl_bench'])
    out['y'] = out['log_ppl_bench'] if fix_ppl else np.log(out['ppl'])   # Ho: log(ppl) of the combined column
    out['transformer'] = (out['Architecture'] == 'Transformer').astype(int)
    out['epochs_num'] = pd.to_numeric(out['epoch'], errors='coerce')
    out['vocab'] = pd.to_numeric(out['Vocabulary'], errors='coerce')
    tok = out['Tokenizer'].fillna('').str.lower()
    out['word_vocab'] = (out['vocab'].isin(WORD_VOCABS) | tok.str.contains('word-level') | tok.str.contains('top-words')).astype(int)
    out['paper'] = out['Reference'].astype('category').cat.codes
    return out


def ho_arrays(df, dvar='dataset'):
    """Arrays used by the models; normalization constants as in Ho et al. (sample minima)."""
    return dict(t=df['publication_date'].to_numpy(float), N=df['param'].to_numpy(float), D=df[dvar].to_numpy(float),
                ptb=df['ptb_dummy'].to_numpy(float), wt2=df['wt2_dummy'].to_numpy(float), y=df['y'].to_numpy(float),
                tr=df['transformer'].to_numpy(float), vocab=df['vocab'].to_numpy(float) if 'vocab' in df else None)


# ============================================================================ model
@dataclass
class HoSpec:
    """Specification of the generalized Ho et al. model (see module docstring)."""
    E: object = 'none'            # 'none' | 'common' | 'bench' | float (fixed common E, nats/word)
    alpha: float | None = None    # fixed N-exponent (None = estimate)
    beta: float | None = None     # fixed D-exponent
    gamma: float | None = None    # impose the frontier elasticity gamma = alpha*beta/(alpha+beta) (beta derived from alpha)
    neutral: bool = False         # Hicks neutrality: alpha_year = beta_year (Ho model 12)
    ay_fixed: float | None = None  # hold alpha_year fixed (profile)
    by_fixed: float | None = None  # hold beta_year fixed (profile)
    gC_fixed: float | None = None  # impose effective-compute growth g_C = alpha_year/alpha + beta_year/beta (profile)
    phi_fixed: float | None = None  # impose the parameter-augmenting share phi = g_N/g_C (g_C free; profile)
    arch: bool = False            # transformer-specific exponents (Ho model 13)
    vocab: bool = False           # + gamma_vocab*ln(vocab) (Ho App. E.2.2)
    no_year: bool = False         # no time terms (static fit)
    bench_const: bool = True      # benchmark-specific a_b, b_b (Ho model 7); False for single-benchmark samples
    delta: float = 0.0025         # L1 penalty weight (Ho et al.); 0 = our unpenalized NLS
    label: str = ''

    def names(self):
        pre = []
        if self.vocab:
            pre += ['gamma_vocab']
        if self.E == 'common':
            pre += ['E']
        elif self.E == 'bench':
            pre += (['E_wt103', 'E_ptb', 'E_wt2'] if self.bench_const else ['E'])
        a = ['alpha_const'] + (['alpha_const_ptb', 'alpha_const_wt2'] if self.bench_const else [])
        if self.phi_fixed is not None:
            a += ['g_C']
        elif not (self.no_year or self.ay_fixed is not None):
            a += ['alpha_year']
        if self.alpha is None:
            a += ['alpha_param'] + (['alpha_param_t'] if self.arch else [])
        b = ['beta_const'] + (['beta_const_ptb', 'beta_const_wt2'] if self.bench_const else [])
        if not (self.neutral or self.no_year or self.by_fixed is not None or self.gC_fixed is not None
                or self.phi_fixed is not None):
            b += ['beta_year']
        if self.beta is None and self.gamma is None:
            b += ['beta_data'] + (['beta_data_t'] if self.arch else [])
        return pre + a + b


class HoModel:
    def __init__(self, spec: HoSpec, data: dict, norm=None):
        self.spec, self.d = spec, data
        self.names = spec.names()
        self.ix = {n: i for i, n in enumerate(self.names)}
        # Normalization constants (Ho: minima of the estimation sample). Bootstrap keeps full-sample constants.
        self.norm = norm or dict(N0=data['N'].min(), D0=data['D'].min(), t0=data['t'].min())

    def unpack(self, th):
        s, ix = self.spec, self.ix
        g = lambda k, dflt=0.0: th[ix[k]] if k in ix else dflt
        p = {n: th[i] for n, i in ix.items()}
        p['alpha_year'] = s.ay_fixed if s.ay_fixed is not None else g('alpha_year')
        if s.neutral:
            p['beta_year'] = p['alpha_year']
        else:
            p['beta_year'] = s.by_fixed if s.by_fixed is not None else g('beta_year')
        p['alpha_param'] = s.alpha if s.alpha is not None else g('alpha_param')
        if s.gamma is not None:        # beta such that alpha*beta/(alpha+beta) = gamma
            a = p['alpha_param']
            p['beta_data'] = s.gamma * a / (a - s.gamma) if a > s.gamma else 1e6
        else:
            p['beta_data'] = s.beta if s.beta is not None else g('beta_data')
        p['alpha_param_t'] = g('alpha_param_t', p['alpha_param']) if s.arch else p['alpha_param']
        p['beta_data_t'] = g('beta_data_t', p['beta_data']) if s.arch else p['beta_data']
        if s.gC_fixed is not None:     # beta_year = beta*(g_C - alpha_year/alpha)
            p['beta_year'] = p['beta_data'] * (s.gC_fixed - p['alpha_year'] / p['alpha_param'])
        if s.phi_fixed is not None:    # alpha_year = alpha*phi*g_C, beta_year = beta*(1-phi)*g_C
            p['alpha_year'] = p['alpha_param'] * s.phi_fixed * g('g_C')
            p['beta_year'] = p['beta_data'] * (1 - s.phi_fixed) * g('g_C')
        return p

    def E_of(self, p, ptb, wt2):
        s = self.spec
        g = lambda k: p.get(k, 0.0)
        if s.E == 'none':
            return 0.0 * ptb
        if s.E == 'common' or (s.E == 'bench' and not s.bench_const):
            return g('E') + 0.0 * ptb
        if s.E == 'bench':
            return g('E_wt103') * (1 - ptb - wt2) + g('E_ptb') * ptb + g('E_wt2') * wt2
        return float(s.E) + 0.0 * ptb

    def predict(self, th, d=None, parts=False):
        d = d or self.d
        s, p, nm = self.spec, self.unpack(th), self.norm
        g = lambda k: p.get(k, 0.0)
        tt = d['t'] - nm['t0']
        al = p['alpha_param'] + (p['alpha_param_t'] - p['alpha_param']) * d['tr'] if s.arch else p['alpha_param']
        be = p['beta_data'] + (p['beta_data_t'] - p['beta_data']) * d['tr'] if s.arch else p['beta_data']
        ac = g('alpha_const') + g('alpha_const_ptb') * d['ptb'] + g('alpha_const_wt2') * d['wt2']
        bc = g('beta_const') + g('beta_const_ptb') * d['ptb'] + g('beta_const_wt2') * d['wt2']
        with np.errstate(over='ignore', invalid='ignore'):
            u = np.exp(ac - p['alpha_year'] * tt - al * (np.log(d['N']) - np.log(nm['N0'])))
            v = np.exp(bc - p['beta_year'] * tt - be * (np.log(d['D']) - np.log(nm['D0'])))
        E = self.E_of(p, d['ptb'], d['wt2'])
        extra = g('gamma_vocab') * np.log(d['vocab']) if s.vocab else 0.0
        if parts:
            return dict(u=u, v=v, E=E, extra=extra)
        return E + extra + u + v

    def resid(self, th, idx=None):
        d = self.d if idx is None else {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in self.d.items()}
        r = d['y'] - self.predict(th, d)
        return np.where(np.isfinite(r), r, 1e3)

    def objective(self, th, idx=None):
        r = self.resid(th, idx)
        val = np.mean(r ** 2) + self.spec.delta * np.sum(np.abs(th))
        return val if np.isfinite(val) else 1e10

    # --- economics of the fitted technology
    def rates(self, th, arch='t'):
        """Factor-augmentation rates g_N, g_D (log units per year), effective-compute growth g_C and T_C (months)."""
        p = self.unpack(th)
        al = p['alpha_param_t'] if (self.spec.arch and arch == 't') else p['alpha_param']
        be = p['beta_data_t'] if (self.spec.arch and arch == 't') else p['beta_data']
        gN, gD = p['alpha_year'] / al, p['beta_year'] / be
        gC = gN + gD
        return dict(alpha_param=al, beta_data=be, alpha_year=p['alpha_year'], beta_year=p['beta_year'], g_N=gN, g_D=gD,
                    g_C=gC, TC_months=12 * LN2 / gC if gC != 0 else np.inf, gamma=al * be / (al + be),
                    ec_per_year=np.exp(gC))

    def technology(self, th, year, bench='wt103', arch='t'):
        """sl.Chinchilla object (loss units: nats per benchmark word) implied at a given date for one benchmark."""
        p, nm = self.unpack(th), self.norm
        g = lambda k: p.get(k, 0.0)
        ptb, wt2 = float(bench == 'ptb'), float(bench == 'wt2')
        al = p['alpha_param_t'] if (self.spec.arch and arch == 't') else p['alpha_param']
        be = p['beta_data_t'] if (self.spec.arch and arch == 't') else p['beta_data']
        tt = year - nm['t0']
        lnA = g('alpha_const') + g('alpha_const_ptb') * ptb + g('alpha_const_wt2') * wt2 - p['alpha_year'] * tt + al * np.log(nm['N0'])
        lnB = g('beta_const') + g('beta_const_ptb') * ptb + g('beta_const_wt2') * wt2 - p['beta_year'] * tt + be * np.log(nm['D0'])
        E = float(self.E_of(p, np.array(ptb), np.array(wt2)))
        return sl.Chinchilla(E=max(E, 1e-12), A=float(np.exp(lnA)), B=float(np.exp(lnB)), alpha=float(al), beta=float(be))


def _bounds(model: HoModel):
    """Box constraints for the unpenalized (delta=0) fits: 0 <= E_b < min(y_b); exponents in (0, 3]."""
    lo, hi = [], []
    d = model.d
    ymin = {'wt103': np.min(d['y'][(d['ptb'] == 0) & (d['wt2'] == 0)], initial=np.inf),
            'ptb': np.min(d['y'][d['ptb'] == 1], initial=np.inf), 'wt2': np.min(d['y'][d['wt2'] == 1], initial=np.inf)}
    for n in model.names:
        if n.startswith('E'):
            b = {'E_wt103': ymin['wt103'], 'E_ptb': ymin['ptb'], 'E_wt2': ymin['wt2']}.get(n, min(ymin.values()))
            lo.append(0.0); hi.append(float(b) - 1e-3 if np.isfinite(b) else 10.0)
        elif n in ('alpha_param', 'beta_data', 'alpha_param_t', 'beta_data_t'):
            lo.append(model.spec.gamma + 1e-3 if (model.spec.gamma is not None and n == 'alpha_param') else 1e-4); hi.append(3.0)
        elif n in ('alpha_year', 'beta_year'):
            lo.append(-2.0); hi.append(2.0)
        elif n == 'g_C':
            lo.append(-5.0); hi.append(10.0)
        else:
            lo.append(-60.0); hi.append(60.0)
    return np.array(lo), np.array(hi)


def default_start(model: HoModel):
    s0 = np.zeros(len(model.names))
    lo, hi = _bounds(model)
    for n, i in model.ix.items():
        if n in ('alpha_param', 'beta_data', 'alpha_param_t', 'beta_data_t'):
            s0[i] = 0.1 if model.spec.gamma is None else 2 * model.spec.gamma + 0.05
        if n.startswith('E'):
            s0[i] = 0.5 * hi[i]
    return s0


def fit(model: HoModel, x0=None, idx=None, method='auto', extra_starts=(), n_random=0, seed=0, n_global=0):
    """Fit a HoModel.
    method='ho'  : Ho et al.'s exact procedure (SLSQP on MSE + delta*L1 from theta=0, or from x0 if given), scipy defaults.
    method='ho_conv': the same, run to convergence (ftol=1e-13); starts at zero and at x0 (if given), keeps the better.
    method='nls' : unpenalized nonlinear least squares, scipy least_squares (TRF, box bounds) from several starts:
                   default start, zeros (clipped), x0, extra_starts, and n_random perturbations of the best start.
    n_global: additional starts drawn at random over the whole parameter box (exponents U[0.02, 0.6], year
              terms U[-0.1, 0.2], constants N(0, 0.6^2), E_b uniform on its bounds) -- a global multistart.
    'auto' = 'ho_conv' if delta>0 else 'nls'."""
    k = len(model.names)
    if method == 'auto':
        method = 'ho_conv' if model.spec.delta > 0 else 'nls'
    if method in ('ho', 'ho_conv'):
        # 'ho': scipy SLSQP defaults (ftol=1e-6, maxiter=100) -- exactly what Ho et al.'s code runs.
        # 'ho_conv': same objective and algorithm, run to convergence (ftol=1e-13, maxiter=5000).
        #  On the DMR ridge the default stopping rule halts SLSQP part-way along the valley (see s2_dmr).
        if method == 'ho':
            starts, opts = [np.zeros(k) if x0 is None else np.asarray(x0, float)], {}
        else:   # converged: start at zero (Ho et al.) and, if given, at x0; keep the lower objective
            starts, opts = [np.zeros(k)] + ([np.asarray(x0, float)] if x0 is not None else []), dict(ftol=1e-13, maxiter=5000)
        best = None
        for st in starts:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                r = minimize(lambda th: model.objective(th, idx), st, method='SLSQP', options=opts)
            if best is None or r.fun < best.fun:
                best = r
        return best.x, float(best.fun)
    lo, hi = _bounds(model)
    starts = [default_start(model), np.zeros(k)]
    if x0 is not None:
        starts.append(np.asarray(x0, float))
    starts += [np.asarray(s, float) for s in extra_starts]
    best = None
    rng = np.random.default_rng(seed)
    for _ in range(n_global):
        st = rng.normal(0, 0.6, k)
        for nme, i in model.ix.items():
            if nme in ('alpha_param', 'beta_data', 'alpha_param_t', 'beta_data_t'):
                st[i] = rng.uniform(0.02, 0.6)
            elif nme in ('alpha_year', 'beta_year'):
                st[i] = rng.uniform(-0.1, 0.2)
            elif nme.startswith('E'):
                st[i] = rng.uniform(lo[i], hi[i])
            elif nme == 'g_C':
                st[i] = rng.uniform(0.1, 2.0)
        starts.append(st)

    def one(st):
        st = np.clip(st, lo + 1e-9, hi - 1e-9)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = least_squares(lambda th: model.resid(th, idx), st, bounds=(lo, hi), method='trf',
                              x_scale='jac', ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=4000)
        return r.x, float(np.mean(r.fun ** 2))

    for st in starts:
        x, f = one(st)
        if best is None or f < best[1]:
            best = (x, f)
    for _ in range(n_random):
        x, f = one(best[0] + rng.normal(0, 0.3, k) * np.maximum(np.abs(best[0]), 0.05))
        if f < best[1] - 1e-12:
            best = (x, f)
    return best[0], best[1]


# ============================================================================ bootstrap
def boot_indices(n, B, seed, clusters=None, ho_style=False):
    """Row indices for B bootstrap samples.  ho_style=True replicates Ho et al.'s np.random.seed(0) +
    np.random.choice(index, n) sequence exactly; otherwise numpy Generator (pairs or cluster)."""
    if ho_style:
        np.random.seed(seed)
        return [np.random.choice(np.arange(n), size=n, replace=True) for _ in range(B)]
    rng = np.random.default_rng(seed)
    if clusters is None:
        return [rng.integers(0, n, n) for _ in range(B)]
    cl = np.asarray(clusters)
    groups = [np.where(cl == g)[0] for g in np.unique(cl)]
    out = []
    for _ in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        out.append(np.concatenate([groups[i] for i in pick]))
    return out


def _boot_worker(args):
    spec, data, norm, x0, idxs, method, fkw = args
    m = HoModel(spec, data, norm)
    res = []
    for idx in idxs:
        try:
            x, fval = fit(m, x0=x0 if method != 'ho' else None, idx=idx, method=method, **fkw)
        except Exception:  # pragma: no cover
            x, fval = np.full(len(m.names), np.nan), np.nan
        res.append(np.r_[x, fval])
    return res


def bootstrap_fit(spec, data, norm, x0, idxs, method='auto', pool=None, **fkw):
    """Refit on each index set; returns array (B, k+1) [params..., objective]. Uses the pool if given.
    Extra keyword arguments are passed to fit() (e.g. n_random, extra_starts)."""
    chunks = [idxs[i::N_PROC] for i in range(N_PROC)]
    order = [list(range(len(idxs)))[i::N_PROC] for i in range(N_PROC)]
    args = [(spec, data, norm, x0, c, method, fkw) for c in chunks]
    outs = pool.map(_boot_worker, args) if pool is not None else [_boot_worker(a) for a in args]
    k = len(HoSpec.names(spec)) + 1
    arr = np.full((len(idxs), k), np.nan)
    for o, rows in zip(order, outs):
        for j, r in zip(o, rows):
            arr[j] = r
    return arr


def tc_ci(gC_draws, q=(2.5, 97.5)):
    """CI for the doubling time T_C = 12 ln2 / g_C (months) from draws of g_C (monotone map; inf if g_C <= 0)."""
    lo, hi = pct(gC_draws, q)
    f = lambda g: 12 * LN2 / g if g > 0 else np.inf
    return f(hi), f(lo)


def tc_q(gC_draws, q=(2.5, 50, 97.5)):
    """[lo, median, hi] of T_C (months) from draws of g_C via the monotone map T_C = 12 ln2/g_C.
    A draw with g_C <= 0 has an infinite doubling time; taking percentiles of T_C directly would instead sort such
    draws (negative 'doubling times') as the *smallest* values and bias the lower bound down (reviewer fix)."""
    g = pct(gC_draws, tuple(100 - np.asarray(q)))
    f = lambda v: 12 * LN2 / v if v > 0 else np.inf
    return np.array([f(v) for v in g])


def chinchilla_fits(cache=True):
    """Besiroglu et al. (2024) estimator on the Epoch Chinchilla extraction (n=240): full model and E -> 0
    ("Ho specification" on experimental data). Cached in data/processed/m5_progress/chinchilla_fits.json."""
    import json
    path = os.path.join(PROC, 'chinchilla_fits.json')
    if cache and os.path.exists(path):
        return json.load(open(path))
    df = sl.chinchilla_extraction(os.path.join(RAW, 'epoch_chinchilla', 'svg_extracted_data.csv'))
    full = sl.fit_chinchilla(df.N, df.D, df.L)
    e0 = sl.fit_chinchilla(df.N, df.D, df.L, E_fixed=1e-10)
    e0n = sl.fit_chinchilla(df.N, df.D, df.L, E_fixed=1e-10, delta=None)
    u, v = full.uv(df.N, df.D)
    Lh = full.loss(df.N, df.D)
    out = dict(n=int(len(df)), full=full.summary(), E0_huber=e0.summary(), E0_nls=e0n.summary(),
               mean_R_over_L=float(np.mean((u + v) / Lh)), gamma_total_mean=float(np.mean(full.gamma * (u + v) / Lh)),
               eps_N_total_mean=float(np.mean(full.alpha * u / Lh)), eps_D_total_mean=float(np.mean(full.beta * v / Lh)))
    # bootstrap (pairs, 200 draws, warm start from the full-sample estimates) of the E=0 frontier elasticity
    rng = np.random.default_rng(7)
    g0, gf = [], []
    for _ in range(200):
        idx = rng.integers(0, len(df), len(df))
        dd = df.iloc[idx]
        m0 = sl.fit_chinchilla(dd.N, dd.D, dd.L, E_fixed=1e-10, init=e0.theta)
        mf = sl.fit_chinchilla(dd.N, dd.D, dd.L, init=full.theta)
        g0.append(m0.gamma); gf.append(mf.gamma)
    out['E0_gamma_ci'] = list(pct(g0)); out['E0_gamma_se'] = float(np.std(g0, ddof=1))
    out['full_gamma_ci'] = list(pct(gf)); out['full_gamma_se'] = float(np.std(gf, ddof=1))
    json.dump(out, open(path, 'w'), indent=1)
    return out


def read_registry(reg, need=('alpha', 'beta')):
    """Primary Huber technology rows (one per dataset) from another module's registry (output/tables/<reg>).

    Handles both registry formats in the project: module m2 ('role' == 'primary', 'estimator' == 'huber') and module
    m1 (no 'role' column; the primal Huber estimator is labelled 'Huber-LSE delta=1e-3...'). Returns (DataFrame, info)
    where info records whether the file existed, its modification time and SHA-256, so every output that uses a
    registry can be traced to the exact file (the registries are regenerated by other modules; m5 results that use
    them change when they change). Reviewer fix: the original filter (estimator == 'huber') silently dropped m1."""
    import datetime
    import hashlib
    p = os.path.join(TAB, reg)
    info = dict(file=reg, exists=os.path.exists(p))
    if not info['exists']:
        log(f"WARNING: {reg} not found -- rows based on this registry are omitted from this run")
        return pd.DataFrame(), info
    info['mtime'] = datetime.datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec='seconds')
    info['sha256'] = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    try:
        tr = pd.read_csv(p)
    except Exception as e:  # pragma: no cover
        log(f"WARNING: registry {reg} unreadable: {e}")
        info['error'] = str(e)
        return pd.DataFrame(), info
    if not set(need).issubset(tr.columns):
        return pd.DataFrame(), info
    if 'role' in tr.columns:
        tr = tr[tr['role'].astype(str) == 'primary']
    if 'estimator' in tr.columns:
        est = tr['estimator'].astype(str).str.lower()
        tr = tr[est.eq('huber') | est.str.startswith('huber-lse')]
    tr = tr[tr[list(need)].notna().all(axis=1)]
    if 'dataset' in tr.columns:
        tr = tr.drop_duplicates('dataset')
    info['n_rows_used'] = int(len(tr))
    return tr.reset_index(drop=True), info


def pct(v, q=(2.5, 50, 97.5)):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return np.percentile(v, q) if len(v) else np.full(len(q), np.nan)


# ============================================================================ output helpers
def fmt(x, nd=3):
    if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)):
        return '--'
    if abs(x) < 0.5 * 10 ** (-nd):   # avoid '-0.000'
        x = 0.0
    return f"{x:.{nd}f}"


def write_tex(path, body_lines, colspec, header_lines, notes, caption=None, label=None, size='\\small'):
    """AER-style booktabs table inside threeparttable, no vertical rules."""
    out = []
    out.append('% Generated by code/analysis/m5_progress/run.py -- do not edit by hand')
    out.append('\\begin{table}[htbp]')
    out.append('\\centering')
    out.append(size)
    if caption:
        out.append(f'\\caption{{{caption}}}')
    if label:
        out.append(f'\\label{{{label}}}')
    out.append('\\begin{threeparttable}')
    out.append(f'\\begin{{tabular}}{{{colspec}}}')
    out.append('\\toprule')
    out += header_lines
    out.append('\\midrule')
    out += body_lines
    out.append('\\bottomrule')
    out.append('\\end{tabular}')
    out.append('\\begin{tablenotes}[flushleft]')
    out.append('\\footnotesize')
    out.append(f'\\item \\textit{{Notes:}} {notes}')
    out.append('\\end{tablenotes}')
    out.append('\\end{threeparttable}')
    out.append('\\end{table}')
    with open(path, 'w') as f:
        f.write('\n'.join(out) + '\n')


def tex_label(s):
    """Make a free-text row label LaTeX-safe (math for symbols, escape stray underscores/percent signs)."""
    rep = [('psi_D', r'$\psi_D$'), ('E_b', r'$E_b$'), ('R*=15.4', r'$R^*=15.4$'), ('gamma ln V', r'$\gamma\ln V$'),
           ('gamma*R/L', r'$\gamma(L-E)/L$'), ('ln D', r'$\ln D$'), ('E = 0', r'$E=0$'), ('(E estimated)', r'($E$ estimated)'),
           ('gamma=', r'$\gamma=$'), ('>=', r'$\geq$')]
    for a, b in rep:
        s = s.replace(a, b)
    out, math = [], False
    for ch in s:
        if ch == '$':
            math = not math
        if ch in '_%&#' and not math:
            out.append('\\' + ch)
        else:
            out.append(ch)
    return ''.join(out)


def log(msg):
    print(f"[m5] {msg}", flush=True)
