# Independent re-implementation of Ho et al.'s model_7 + residuals (copied logic from section3.ipynb), no use of common.HoModel
import json, numpy as np, pandas as pd
B='/Users/yigitokar/scaling-laws-pf/data/processed/m5_progress/'
df = pd.read_csv(B+'ho_df_head.csv')
year, param, dataset = df.publication_date.values, df.param.values, df.dataset.values
ptb, wt2 = (df.dataset_name=='ptb').astype(int).values, (df.dataset_name=='wt2').astype(int).values
log_ppl = np.log(df.ppl.values)
pc, dc, yc = param.min(), dataset.min(), year.min()
def model_7(p):
    ac, acp, acw, ay, ap, bc, bcp, bcw, by, bd = p
    a = ac + acp*ptb + acw*wt2 - ay*(year-yc) - ap*(np.log(param)-np.log(pc))
    b = bc + bcp*ptb + bcw*wt2 - by*(year-yc) - bd*(np.log(dataset)-np.log(dc))
    return np.exp(a)+np.exp(b)
def resid(p, delta=0.0025):
    return np.mean((log_ppl-model_7(p))**2) + delta*np.sum(np.abs(p))
pub = np.array([0.912701, 0.000379, 0.055412, 0.003938, 0.067982, 0.771409, 0.176415, 0.095418, 0.035620, 0.039587])
rs = json.load(open(B+'replication_summary.json'))
conv = np.array(rs['conv_theta'])
print('n', len(df), 'published obj', resid(pub), 'converged obj', resid(conv))
print('MSE pub', resid(pub,0), 'MSE conv', resid(conv,0))
for nm,p in (('pub',pub),('conv',conv)):
    ay,ap,by,bd = p[3],p[4],p[8],p[9]
    gC = ay/ap+by/bd
    print(nm, 'alpha', ap, 'beta', bd, 'ay', ay, 'by', by, 'T_C', 12*np.log(2)/gC)
# numerical gradient at published point (smooth part) to confirm non-stationarity
from scipy.optimize import approx_fprime
g = approx_fprime(pub, lambda p: resid(p), 1e-7)
print('grad at pub', np.round(g,4))
