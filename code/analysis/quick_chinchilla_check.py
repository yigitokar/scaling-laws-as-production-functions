import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, huber
df = pd.read_csv("data/raw/epoch_chinchilla/svg_extracted_data.csv")
df = df.sort_values("loss").iloc[:-5]
N = df["Model Size"].values; C = df["Training FLOP"].values; D = C/(6*N); L = df["loss"].values
lN, lD, lL = np.log(N), np.log(D), np.log(L)
def obj(p, delta=1e-3):
    a, b, e, al, be = p
    pred = np.logaddexp(np.logaddexp(a - al*lN, b - be*lD), e)
    r = pred - lL
    return np.sum(huber(delta, r))
best=None
for a in np.arange(0,30,5):
  for b in np.arange(0,30,5):
    for e in np.arange(-1,1.5,0.5):
      for al in np.arange(0,2.5,0.5):
        for be in np.arange(0,2.5,0.5):
          r = minimize(obj, [a,b,e,al,be], method="L-BFGS-B")
          if best is None or r.fun < best.fun: best=r
a,b,e,al,be = best.x
print(f"n={len(df)}  A={np.exp(a):.1f} B={np.exp(b):.1f} E={np.exp(e):.3f} alpha={al:.3f} beta={be:.3f}  a=beta/(a+b)={be/(al+be):.3f}")
print("D/N range:", np.percentile(D/N,[0,25,50,75,100]).round(1))
