"""Throughput benchmark: tokens/s and achieved TFLOP/s for a few model sizes on this machine."""
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from gpt_mlx import GPT, GPTConfig, count_params, flops_per_token


def bench(cfg, batch, steps=30):
    model = GPT(cfg)
    model.set_dtype(mx.bfloat16)
    opt = optim.AdamW(learning_rate=1e-3, weight_decay=0.1)
    mx.eval(model.parameters())

    def loss_fn(model, x, y):
        logits = model(x).astype(mx.float32)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    state = [model.state, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y):
        loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
        opt.update(model, grads)
        return loss

    x = mx.random.randint(0, cfg.vocab_size, (batch, cfg.seq_len))
    y = mx.random.randint(0, cfg.vocab_size, (batch, cfg.seq_len))
    for _ in range(5):
        mx.eval(step(x, y), state)
    t0 = time.perf_counter()
    for _ in range(steps):
        mx.eval(step(x, y), state)
    dt = time.perf_counter() - t0
    toks = batch * cfg.seq_len * steps / dt
    tflops = toks * flops_per_token(cfg) / 1e12
    return toks, tflops


if __name__ == "__main__":
    for (L, d, H) in [(2, 128, 2), (4, 256, 4), (6, 384, 6), (8, 512, 8), (10, 640, 10), (12, 768, 12)]:
        cfg = GPTConfig(n_layer=L, d_model=d, n_head=H)
        p = count_params(cfg)
        toks, tf = bench(cfg, batch=64)
        print(f"L={L:2d} d={d:4d}  N_nonemb={p['non_embedding']/1e6:6.2f}M  N_total={p['total']/1e6:6.2f}M  "
              f"tok/s={toks/1e3:8.1f}k  TFLOP/s={tf:5.2f}", flush=True)
