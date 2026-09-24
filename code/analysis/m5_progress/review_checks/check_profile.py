"""Reviewer check: are the 1-D profile LR values at the CI boundaries robust to a much heavier global multistart?
And is the unpenalized NLS minimum (mse_min used for every LR) the global minimum?  Also: A7 E_b at its bound?"""
import json
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, '/Users/yigitokar/scaling-laws-pf/code/analysis/m5_progress')
from common import (BESIROGLU, HoModel, HoSpec, fit, ho_arrays, load_ho, _bounds)  # noqa: E402

LN2 = np.log(2)


def job(args):
    kind, val = args
    d = ho_arrays(load_ho())
    if kind == 'gC':
        m = HoModel(HoSpec(delta=0, gC_fixed=float(val)), d)
    elif kind == 'phi':
        m = HoModel(HoSpec(delta=0, phi_fixed=float(val)), d)
    elif kind == 'A2':
        m = HoModel(HoSpec(delta=0), d)
    x, f = fit(m, n_random=10, n_global=60, seed=12345)
    r = m.rates(x)
    return dict(kind=kind, value=val, mse=f, g_C=r['g_C'], alpha=r['alpha_param'], beta=r['beta_data'])


def a7():
    d = ho_arrays(load_ho())
    m = HoModel(HoSpec(delta=0, E='bench', alpha=BESIROGLU.alpha, beta=BESIROGLU.beta), d)
    x, f = fit(m, n_random=8, seed=11)
    lo, hi = _bounds(m)
    return {n: (float(x[i]), float(hi[i])) for n, i in m.ix.items() if n.startswith('E')}, f


if __name__ == '__main__':
    tasks = [('A2', 0)] + [('gC', v) for v in (0.2, 0.225, 1.25, 1.55, 2.0, 2.15, 2.3)] + [('phi', v) for v in (-0.5, -0.25, 0.0, 3.25, 3.5)]
    with mp.get_context('spawn').Pool(6) as pool:
        res = pool.map(job, tasks)
    mse_min = min(r['mse'] for r in res if r['kind'] == 'A2')
    orig = json.load(open('/Users/yigitokar/scaling-laws-pf/data/processed/m5_progress/dmr_summary.json'))['mse_min']
    print('NLS global-multistart mse_min', mse_min, 'pipeline mse_min', orig)
    import pandas as pd
    p1 = pd.read_csv('/Users/yigitokar/scaling-laws-pf/data/processed/m5_progress/dmr_profile_1d.csv')
    for r in res:
        if r['kind'] == 'A2':
            continue
        pl = p1[(p1.kind == r['kind']) & np.isclose(p1.value, r['value'])]
        print(r['kind'], r['value'], 'LR heavy', round(231 * np.log(r['mse'] / min(mse_min, orig)), 3),
              'LR pipeline', round(float(pl.LR.iloc[0]), 3) if len(pl) else None, 'T_C', round(12 * LN2 / r['g_C'], 2))
    print('A7 E_b (estimate, upper bound):', a7())
