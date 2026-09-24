"""Minimal decoder-only transformer in MLX for small-scale scaling-law sweeps.

Mixed precision: parameters (and AdamW moments) are kept in float32; every matmul runs in
bfloat16 by casting weights on the fly (so small late-cooldown updates are not lost to bf16
rounding). Parameter counting reports both total and non-embedding parameters so either
measure of N (Kaplan et al. 2020 vs Hoffmann et al. 2022 convention) can be used downstream.
"""
import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

CDT = mx.bfloat16  # compute dtype


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 256
    seq_len: int = 256
    mlp_ratio: int = 4


class Lin(nn.Module):
    """Bias-free linear layer with fp32 master weight and bf16 compute."""

    def __init__(self, d_in, d_out, std=None):
        super().__init__()
        std = std if std is not None else 1.0 / math.sqrt(d_in)
        self.weight = mx.random.normal((d_out, d_in)) * std

    def __call__(self, x):
        return x @ self.weight.astype(CDT).T


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        d = cfg.d_model
        self.n_head = cfg.n_head
        self.ln1 = nn.RMSNorm(d)
        self.qkv = Lin(d, 3 * d)
        # GPT-2 style scaled init for residual projections
        self.proj = Lin(d, d, std=1.0 / math.sqrt(d) / math.sqrt(2 * cfg.n_layer))
        self.ln2 = nn.RMSNorm(d)
        self.fc1 = Lin(d, cfg.mlp_ratio * d)
        self.fc2 = Lin(cfg.mlp_ratio * d, d, std=1.0 / math.sqrt(cfg.mlp_ratio * d) / math.sqrt(2 * cfg.n_layer))
        self.rope = nn.RoPE(d // cfg.n_head, traditional=False)

    def __call__(self, x, mask):
        B, T, D = x.shape
        h = self.ln1(x.astype(mx.float32)).astype(CDT)
        q, k, v = mx.split(self.qkv(h), 3, axis=-1)
        q = q.reshape(B, T, self.n_head, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, -1).transpose(0, 2, 1, 3)
        q, k = self.rope(q), self.rope(k)
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(q.shape[-1]), mask=mask)
        a = a.transpose(0, 2, 1, 3).reshape(B, T, D)
        x = x + self.proj(a)
        h = self.ln2(x.astype(mx.float32)).astype(CDT)
        x = x + self.fc2(nn.gelu_approx(self.fc1(h)))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wte.weight = mx.random.normal((cfg.vocab_size, cfg.d_model)) * 0.02
        self.blocks = [Block(cfg) for _ in range(cfg.n_layer)]
        self.ln_f = nn.RMSNorm(cfg.d_model)

    def __call__(self, idx):
        T = idx.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T).astype(CDT)
        x = self.wte(idx).astype(CDT)
        for b in self.blocks:
            x = b(x, mask)
        x = self.ln_f(x.astype(mx.float32)).astype(CDT)
        return x @ self.wte.weight.astype(CDT).T  # tied unembedding


def count_params(cfg: GPTConfig):
    d, L, V = cfg.d_model, cfg.n_layer, cfg.vocab_size
    per_block = 3 * d * d + d * d + 2 * cfg.mlp_ratio * d * d + 2 * d  # attn + mlp + 2 RMSNorm gains
    non_emb = L * per_block + d
    emb = V * d  # tied input/output embedding
    return {"non_embedding": non_emb, "embedding": emb, "total": non_emb + emb}


def flops_per_token(cfg: GPTConfig):
    """Training FLOPs per token (fwd+bwd): 6*(non-embedding + unembedding matmul) + attention.
    Embedding lookup is free; the tied unembedding matmul is not."""
    p = count_params(cfg)
    attn = 6 * cfg.n_layer * cfg.seq_len * cfg.d_model
    return 6 * p["non_embedding"] + 6 * cfg.vocab_size * cfg.d_model + attn
