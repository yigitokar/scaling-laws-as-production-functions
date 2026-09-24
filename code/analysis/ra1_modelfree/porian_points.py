"""porian_points.py -- IsoFLOP-like profiles from Porian et al.'s (2024) tuned runs, via module m8's port of their
pipeline (code/analysis/m8_measurement/porian.py, imported unchanged). Run as a separate process by run.py because
m8 and m1 both name a module `common`.

Configuration = Porian's final step (Table 1, step 5): tuned learning rate, batch size and beta2 per model size,
short warmup, constant learning rate, 'standard' count (N = non-embedding body + head, FLOPs = 6 N per token),
validation loss. For each FLOP budget C in Porian's grid (1.25e16 ... 2.56e19) the loss of every run is read
at C along its (constant-LR) loss curve by Porian's log-log interpolation (fetch_flop): one point per model size.
Output: data/processed/ra1_modelfree/porian_isoflop_points.csv with columns dataset, C, n, t, loss, width, depth.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
M8 = os.path.join(os.path.dirname(HERE), "m8_measurement")
sys.path.insert(0, M8)

import pandas as pd  # noqa: E402

import porian  # noqa: E402  (m8 module)


def main(out):
    df = porian.load_porian()
    rows = []
    for ds in ("rw", "owt2"):
        loss_key, fpt_key, n_key = porian.COUNTS[("standard", "val")]
        sub = df.query("dataset==@ds and hparams=='tuned' and warmup=='short' and decay=='const'")
        for C in porian.FLOP_VALS:
            pts = porian.fetch_flop(sub, C, loss_key, fpt_key, n_key)
            if len(pts) == 0:
                continue
            pts = pts.loc[pts.groupby("n").loss.idxmin()]   # one point per size (as Porian's isoflop_argmin)
            pts["C"] = C
            pts["dataset"] = ds
            rows.append(pts)
    res = pd.concat(rows, ignore_index=True)[["dataset", "C", "n", "t", "loss", "width", "depth", "lr", "bs"]]
    res.to_csv(out, index=False)
    print(f"porian points: {len(res)} rows -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
