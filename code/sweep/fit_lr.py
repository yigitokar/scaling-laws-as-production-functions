"""Fit the peak-LR rule lr*(d) = c (d/256)^p from the 'lrsweep' runs (quadratic in log LR per width)."""
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(ROOT, "data", "processed", "sweep", "results.jsonl")
rows = [json.loads(l) for l in open(RES)]
rows = [r for r in rows if r.get("tag") == "lrsweep" and np.isfinite(r.get("loss_edu", np.nan))]
widths = sorted({r["d"] for r in rows})
best = {}
for d in widths:
    rs = sorted((r["lr"], r["loss_edu"]) for r in rows if r["d"] == d)
    x = np.log([a for a, _ in rs])
    y = np.array([b for _, b in rs])
    i = int(np.argmin(y))
    lr_star = float(np.exp(x[i]))
    if len(rs) >= 3:
        c2, c1, c0 = np.polyfit(x, y, 2)
        if c2 > 0:
            xs = -c1 / (2 * c2)
            xs = float(np.clip(xs, x.min() - 0.35, x.max() + 0.35))  # at most ~1.4x beyond the tested range
            lr_star = float(np.exp(xs))
    best[d] = lr_star
    print(f"d={d}: " + ", ".join(f"{a:.0e}->{b:.4f}" for a, b in rs) + f"  => lr*={lr_star:.2e}")
ds = np.array(list(best.keys()), float)
ls = np.log(np.array(list(best.values())))
p, logc = np.polyfit(np.log(ds / 256.0), ls, 1) if len(ds) >= 2 else (0.0, ls[0])
fit = {"c": float(np.exp(logc)), "p": float(p), "per_width": {str(int(k)): v for k, v in best.items()}}
json.dump(fit, open(os.path.join(ROOT, "data", "processed", "sweep", "lr_fit.json"), "w"), indent=2)
print("fit:", fit)
for d in [128, 192, 256, 320, 384, 448, 512, 640]:
    print(d, f"{fit['c'] * (d / 256) ** fit['p']:.2e}")
