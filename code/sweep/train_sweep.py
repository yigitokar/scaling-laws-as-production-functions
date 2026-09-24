"""Warmup-stable-decay (WSD) scaling sweep with cooldown branches.

One "trunk" run per (regime, model size, lr, seed) trains at constant peak LR. At 80% of each
target token budget D_k the trunk state is copied and a cooldown branch (LR -> 0 with a 1-sqrt
schedule over the last 20% of D_k) is run to completion; the branch's final validation losses
are one observation (N, D_k, L). The largest budget is reached by cooling down the trunk itself.
Hagele et al. (2024) show WSD+cooldown matches cosine schedules tuned to each D, which lets one
trunk deliver a whole column of the (N, D) design.

Every model sees the same data order (common random numbers across the design), and every
endpoint is evaluated on held-out validation sets of both corpora, so output is measured in the
same units (nats/token under one shared tokenizer) for all "labs".
"""
import argparse
import json
import math
import os
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map

from gpt_mlx import GPT, GPTConfig, count_params, flops_per_token

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.environ.get("SWEEP_DATA", os.path.join(ROOT, "data", "processed", "sweep"))
REGIME_SEED = {"edu": 11, "web": 22, "mix": 33}


class Loader:
    """Batches of (seq_len+1)-token windows in a fixed random order (common random numbers across sizes).
    ext=True appends the extension shards ({regime}_train_ext.bin) AFTER the original file's permutation,
    so the first 1.7B tokens are seen in exactly the same order as in the main grid."""

    def __init__(self, regime, seq_len, batch, data_seed, ext=False):
        if regime == "mix":  # 50/50 interleave of the two corpora, by batch rows
            self.parts = [Loader(r, seq_len, batch // 2, data_seed, ext) for r in ("edu", "web")]
            return
        self.parts = None
        self.T, self.B = seq_len, batch
        self.arrs = [np.memmap(os.path.join(DATA, f"{regime}_train.bin"), dtype=np.uint16, mode="r")]
        n1 = (len(self.arrs[0]) - 1) // seq_len
        perm = [np.random.default_rng(REGIME_SEED[regime] + 1000 * data_seed).permutation(n1)]
        src = [np.zeros(n1, dtype=np.int8)]
        if ext:
            self.arrs.append(np.memmap(os.path.join(DATA, f"{regime}_train_ext.bin"), dtype=np.uint16, mode="r"))
            n2 = (len(self.arrs[1]) - 1) // seq_len
            perm.append(np.random.default_rng(REGIME_SEED[regime] + 1000 * data_seed + 7).permutation(n2))
            src.append(np.ones(n2, dtype=np.int8))
        self.perm = np.concatenate(perm)
        self.src = np.concatenate(src)

    def batch(self, step):
        if self.parts is not None:
            xs, ys = zip(*(p.batch(step) for p in self.parts))
            return np.concatenate(xs), np.concatenate(ys)
        n = len(self.perm)
        lo = (step * self.B) % n
        idx, src = self.perm[lo: lo + self.B], self.src[lo: lo + self.B]
        w = np.stack([self.arrs[s][i * self.T: i * self.T + self.T + 1] for i, s in zip(idx, src)]).astype(np.int32)
        return w[:, :-1], w[:, 1:]


def load_val(regime, seq_len, n_tokens):
    arr = np.memmap(os.path.join(DATA, f"{regime}_val.bin"), dtype=np.uint16, mode="r")
    n_seq = min(n_tokens // seq_len, (len(arr) - 1) // seq_len)
    w = np.stack([arr[i * seq_len: i * seq_len + seq_len + 1] for i in range(n_seq)]).astype(np.int32)
    return w[:, :-1], w[:, 1:]


def loss_fn(model, x, y):
    return nn.losses.cross_entropy(model(x).astype(mx.float32), y, reduction="mean")


def evaluate(model, val, bs=128):
    x, y = val
    tot, n = 0.0, 0
    for i in range(0, len(x), bs):
        xb, yb = mx.array(x[i:i + bs]), mx.array(y[i:i + bs])
        l = nn.losses.cross_entropy(model(xb).astype(mx.float32), yb, reduction="sum")
        tot += l.item()
        n += yb.size
    return tot / n


def make_opt(lr, wd, b2):
    return optim.AdamW(learning_rate=lr, betas=[0.9, b2], eps=1e-8, weight_decay=wd, bias_correction=True)


def make_step(model, opt, clip):
    state = [model.state, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y):
        loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
        grads, _ = optim.clip_grad_norm(grads, clip)
        opt.update(model, grads)
        return loss

    return step, state


def run(args):
    cfg = GPTConfig(vocab_size=8192, n_layer=args.L, n_head=args.d // 64 if args.d >= 128 else 2,
                    d_model=args.d, seq_len=args.seq)
    p = count_params(cfg)
    fpt = flops_per_token(cfg)
    tps = args.batch * args.seq  # tokens per step
    D_list = sorted(int(float(x)) for x in args.D.split(","))
    out = args.out
    key = dict(regime=args.regime, L=args.L, d=args.d, lr=args.lr, seed=args.seed, batch=args.batch, tag=args.tag)
    done = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            if all(r.get(k) == v for k, v in key.items()):
                done.add(r["D_target"])
    todo = [D for D in D_list if D not in done]
    if not todo:
        print("all done", key, flush=True)
        return
    mx.random.seed(args.seed)
    model = GPT(cfg)
    mx.eval(model.parameters())
    opt = make_opt(args.lr, args.wd, args.b2)
    step_fn, state = make_step(model, opt, args.clip)
    loader = Loader(args.regime, args.seq, args.batch, args.data_seed, ext=bool(args.ext))
    vals = {r: load_val(r, args.seq, args.val_tokens) for r in args.evals.split(",")}

    total_steps = math.ceil(max(todo) / tps)
    branch_at = {}  # step -> list of (D_target, end_step)
    for D in todo:
        end = math.ceil(D / tps)
        start = int(round(end * (1 - args.cool)))
        branch_at.setdefault(start, []).append((D, end))
    warm = args.warmup

    def trunk_lr(s):
        return args.lr * min(1.0, (s + 1) / warm)

    def cool_lr(s, start, end):
        frac = (s - start) / max(end - start, 1)
        return args.lr * min(1.0, (s + 1) / warm) * (1 - math.sqrt(frac))

    def record(D, end, m, t0, trunk_train):
        res = dict(key)
        res.update(D_target=D, steps=end, tokens=end * tps, N_nonemb=p["non_embedding"], N_emb=p["embedding"],
                   N_total=p["total"], flops_per_token=fpt, C=fpt * end * tps,
                   C_6ND_nonemb=6 * p["non_embedding"] * end * tps, C_6ND_total=6 * p["total"] * end * tps,
                   **{f"loss_{r}": evaluate(m, v) for r, v in vals.items()},
                   train_loss_trunk=trunk_train, wall=time.time() - t0, n_head=cfg.n_head, seq=args.seq,
                   warmup=warm, cool=args.cool, wd=args.wd, b2=args.b2, data_seed=args.data_seed, ext=args.ext)
        with open(out, "a") as f:
            f.write(json.dumps(res) + "\n")
        if args.save_ckpt:  # final weights of every annealed endpoint, so it can be re-evaluated later
            ck = os.path.join(os.path.dirname(out), "ckpt")
            os.makedirs(ck, exist_ok=True)
            name = f"{args.tag}_{args.regime}_d{args.d}_L{args.L}_lr{args.lr}_s{args.seed}_ds{args.data_seed}_D{D}.safetensors"
            mx.save_safetensors(os.path.join(ck, name), dict(tree_flatten(m.parameters())))
        print(f"  -> D={D/1e6:.0f}M  " + "  ".join(f"L_{r}={res['loss_'+r]:.4f}" for r in vals) + "  "
              f"({res['wall']:.0f}s)", flush=True)

    trunk_eval_at = {math.ceil(D / tps): D for D in todo if D != max(todo)}
    t0 = time.time()
    ema, curve = None, []
    last_D, last_end = max(todo), total_steps
    last_start = int(round(last_end * (1 - args.cool)))
    for s in range(total_steps):
        # branch points (except the final budget, which is the trunk's own cooldown)
        for (D, end) in branch_at.get(s, []):
            if D == last_D:
                continue
            bm = GPT(cfg)
            bm.update(tree_map(lambda a: a, model.parameters()))
            bo = make_opt(args.lr, args.wd, args.b2)
            bo.state = tree_map(lambda a: a, opt.state)
            bstep, bstate = make_step(bm, bo, args.clip)
            for bs in range(s, end):
                bo.learning_rate = cool_lr(bs, s, end)
                x, y = loader.batch(bs)
                mx.eval(bstep(mx.array(x), mx.array(y)), bstate)
            record(D, end, bm, t0, ema)
            del bm, bo, bstep, bstate
        opt.learning_rate = cool_lr(s, last_start, last_end) if s >= last_start else trunk_lr(s)
        x, y = loader.batch(s)
        loss = step_fn(mx.array(x), mx.array(y))
        mx.eval(loss, state)
        if args.trunk_eval and (s + 1) in trunk_eval_at:
            # un-annealed (constant-LR trunk) validation loss at exactly D_k tokens: the Porian-style
            # "unannealed" counterpart of the annealed endpoint at the same (N, D)
            Dk = trunk_eval_at[s + 1]
            row = dict(key)
            row.update(D_target=Dk, tokens=(s + 1) * tps, N_nonemb=p["non_embedding"], N_emb=p["embedding"],
                       N_total=p["total"], flops_per_token=fpt, C=fpt * (s + 1) * tps, anneal=0,
                       **{f"loss_{r}": evaluate(model, v) for r, v in vals.items()}, ext=args.ext,
                       data_seed=args.data_seed)
            with open(out.replace(".jsonl", "_trunk.jsonl"), "a") as f:
                f.write(json.dumps(row) + "\n")
        if s % 50 == 0:
            lv = loss.item()
            ema = lv if ema is None else 0.9 * ema + 0.1 * lv
            if s % 1000 == 0:
                curve.append((s * tps, lv))
                print(f"step {s}/{total_steps} tok={s*tps/1e6:.0f}M loss={lv:.4f} "
                      f"lr={opt.learning_rate.item():.2e} {(s+1)*tps/(time.time()-t0)/1e3:.0f}k tok/s", flush=True)
    record(last_D, last_end, model, t0, ema)
    with open(out.replace(".jsonl", "_curves.jsonl"), "a") as f:
        f.write(json.dumps({**key, "curve": curve}) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="edu", choices=["edu", "web", "mix"])
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--D", default="25e6,50e6,100e6")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=250)
    ap.add_argument("--cool", type=float, default=0.2)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--b2", type=float, default=0.99)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--val_tokens", type=int, default=1_048_576)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--evals", default="edu,web")
    ap.add_argument("--ext", type=int, default=0)
    ap.add_argument("--save_ckpt", type=int, default=1)
    ap.add_argument("--trunk_eval", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "processed", "sweep", "results.jsonl"))
    run(ap.parse_args())
