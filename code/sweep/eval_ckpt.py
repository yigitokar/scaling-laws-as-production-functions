"""Post-hoc evaluation of saved endpoint checkpoints on additional validation sets.

Runs whose job list was generated before the neutral evaluation sets were added (queue A's main FineWeb grid,
started 15:37; queue B's first high-M job, started 17:33) trained with evals "edu,web" or "edu,web,wiki". Their
annealed endpoints were saved (train_sweep.py --save_ckpt), so the missing sets are computed here from the same
weights with the same evaluation code (train_sweep.load_val / evaluate, same val_tokens and seq). Each row also
re-evaluates the sets recorded during training and reports the difference, which validates the reload.

Output: data/processed/sweep/results_posthoc.jsonl (one row per checkpoint; completed checkpoints are skipped).
"""
import argparse
import glob
import json
import os
import re
import time

import mlx.core as mx

from gpt_mlx import GPT, GPTConfig
from train_sweep import DATA, evaluate, load_val

PAT = re.compile(r"(?P<tag>[A-Za-z0-9]+)_(?P<regime>edu|web|mix)_d(?P<d>\d+)_L(?P<L>\d+)_lr(?P<lr>[0-9.e-]+)"
                 r"_s(?P<seed>\d+)_ds(?P<data_seed>\d+)_D(?P<D>\d+)\.safetensors$")


def parse(path):
    m = PAT.search(os.path.basename(path))
    g = m.groupdict()
    return dict(tag=g["tag"], regime=g["regime"], d=int(g["d"]), L=int(g["L"]), lr=float(g["lr"]),
                seed=int(g["seed"]), data_seed=int(g["data_seed"]), D_target=int(g["D"]))


def recorded(results, k):
    """The training-time row of this endpoint (matches the key train_sweep writes)."""
    for r in results:
        if all(r.get(f) == k[f] for f in ("tag", "regime", "d", "L", "lr", "seed", "D_target")) \
                and r.get("data_seed", 0) == k["data_seed"]:
            return r
    return None


def main(a):
    results = [json.loads(l) for l in open(os.path.join(DATA, "results.jsonl"))]
    out = os.path.join(DATA, "results_posthoc.jsonl")
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["ckpt"] for l in open(out)}
    ckpts = sorted(glob.glob(os.path.join(DATA, "ckpt", a.pattern)))
    want = a.evals.split(",")
    vals = {}
    for path in ckpts:
        name = os.path.basename(path)
        if name in done:
            continue
        k = parse(path)
        rec = recorded(results, k)
        if rec is None:
            print("no training row for", name, flush=True)
            continue
        missing = [r for r in want if f"loss_{r}" not in rec]
        if a.only_missing and not missing:
            continue
        seq = rec.get("seq", 256)
        cfg = GPTConfig(vocab_size=8192, n_layer=k["L"], n_head=k["d"] // 64 if k["d"] >= 128 else 2,
                        d_model=k["d"], seq_len=seq)
        model = GPT(cfg)
        model.load_weights(path)
        mx.eval(model.parameters())
        t0 = time.time()
        row = dict(ckpt=name, **k, seq=seq, val_tokens=a.val_tokens, ext=rec.get("ext", 0))
        check = {}
        for r in want:
            key = (r, seq)
            if key not in vals:
                vals[key] = load_val(r, seq, a.val_tokens)
            row[f"loss_{r}"] = evaluate(model, vals[key])
            if f"loss_{r}" in rec:
                check[r] = row[f"loss_{r}"] - rec[f"loss_{r}"]
        row["recheck_diff"] = check
        row["added"] = missing
        row["wall"] = time.time() - t0
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{name}: added {missing}; max |diff| on recorded sets = "
              f"{max((abs(v) for v in check.values()), default=float('nan')):.2e} ({row['wall']:.0f}s)", flush=True)
        del model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="*.safetensors")
    ap.add_argument("--evals", default="edu,web,wiki,c4,pg19")
    ap.add_argument("--val_tokens", type=int, default=1_048_576)
    ap.add_argument("--only_missing", type=int, default=1)
    main(ap.parse_args())
