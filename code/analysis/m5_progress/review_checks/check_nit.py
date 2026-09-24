import sys, numpy as np, warnings
sys.path.insert(0, '/Users/yigitokar/scaling-laws-pf/code/analysis/m5_progress')
from common import load_ho, ho_arrays, HoModel, HoSpec
from scipy.optimize import minimize
import scipy; print('scipy', scipy.__version__, 'numpy', np.__version__)
m = HoModel(HoSpec(delta=0.0025), ho_arrays(load_ho()))
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    r = minimize(lambda th: m.objective(th), np.zeros(10), method='SLSQP')
print('default: nit', r.nit, 'nfev', r.nfev, 'fun', r.fun, r.message)
print('jac', np.round(r.jac,4))
r2 = minimize(lambda th: m.objective(th), np.zeros(10), method='SLSQP', options=dict(ftol=1e-13, maxiter=5000))
print('conv: nit', r2.nit, 'fun', r2.fun, r2.message)
th=r.x; th2=r2.x
print('sum|theta| pub', np.abs(th).sum(), 'conv', np.abs(th2).sum())
print('MSE pub', np.mean(m.resid(th)**2), 'conv', np.mean(m.resid(th2)**2))
print('conv theta', np.round(th2,4))
