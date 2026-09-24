import time
from functools import partial
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from gpt_mlx import GPT, GPTConfig, flops_per_token
def run(cfg, dtype, batch=64, steps=25):
    model = GPT(cfg)
    opt = optim.AdamW(learning_rate=1e-3, weight_decay=0.1)
    mx.eval(model.parameters())
    def loss_fn(m, x, y): return nn.losses.cross_entropy(m(x).astype(mx.float32), y, reduction="mean")
    state=[model.state, opt.state]
    @partial(mx.compile, inputs=state, outputs=state)
    def step(x,y):
        l,g = nn.value_and_grad(model, loss_fn)(model,x,y); opt.update(model,g); return l
    x = mx.random.randint(0,cfg.vocab_size,(batch,cfg.seq_len)); y = mx.random.randint(0,cfg.vocab_size,(batch,cfg.seq_len))
    for _ in range(4): mx.eval(step(x,y), state)
    t=time.perf_counter()
    for _ in range(steps): mx.eval(step(x,y), state)
    dt=time.perf_counter()-t; tok=batch*cfg.seq_len*steps/dt
    return tok, tok*flops_per_token(cfg)/1e12
for L,d in [(2,128),(4,256),(8,512),(10,640)]:
    cfg=GPTConfig(n_layer=L,d_model=d,n_head=d//64)
    for dt in ["mixed"]:
        tok,tf=run(cfg,dt); print(L,d,dt,f"{tok/1e3:.0f}k tok/s {tf:.1f} TF/s",flush=True)
