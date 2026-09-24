"""figures.py -- figures for module m8_measurement (AER style via aer_style.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import aer_style as st

BLUE, ORANGE, AQUA = st.BLUE, st.ORANGE, st.AQUA


def fig_measurement(A, B):
    """Figure: the Kaplan-Chinchilla allocation exponent as a measurement + flexible-input problem."""
    st.use()
    fig, ax = plt.subplots(1, 3, figsize=(st.WIDTH_FULL, 2.45))
    # (a) Porian path
    s = A["steps"]
    for k, (ds, col, lab, off) in enumerate([("rw", BLUE, "RefinedWeb", -0.09), ("owt2", ORANGE, "OpenWebText2", 0.09)]):
        g = s[(s.dataset == ds) & s.step.isin([1, 2, 3, 4, 5])].sort_values("step")
        x = g.step.astype(int).values + off
        ax[0].errorbar(x, g.a, yerr=[g.a - g.a_lo, g.a_hi - g.a], fmt="o", color=col, ms=4, lw=1.0, capsize=0, label=lab)
        ax[0].plot(x, g.a_porian, "_", color=st.INK, ms=7, mew=1.0, label="Porian et al. (2024)" if k == 0 else None)
    ax[0].axhline(0.5, color=st.MUTED, lw=0.8, ls=":")
    ax[0].set_xticks([1, 2, 3, 4, 5])
    ax[0].set_xticklabels(["Kaplan", "+head\nFLOPs", "+warm-\nup", "+cosine\ndecay", "+tuned\nLR, B"], fontsize=7)
    ax[0].set_ylabel("Allocation exponent $a$ ($N^*\\propto C^a$)")
    ax[0].set_ylim(0.44, 0.97)
    ax[0].annotate("", xy=(2.0, 0.925), xytext=(1.0, 0.925), arrowprops=dict(arrowstyle="-", color=st.INK2, lw=0.7))
    ax[0].text(1.5, 0.93, "measurement", ha="center", va="bottom", fontsize=7, color=st.INK2)
    ax[0].annotate("", xy=(5.1, 0.78), xytext=(2.1, 0.78), arrowprops=dict(arrowstyle="-", color=st.INK2, lw=0.7))
    ax[0].text(3.6, 0.787, "flexible inputs", ha="center", va="bottom", fontsize=7, color=st.INK2)
    ax[0].legend(loc="lower left", fontsize=6.5, handletextpad=0.3)
    ax[0].set_title("(a) Porian et al. path, re-estimated", loc="left")
    # (b) flexible-input first-order formula, RW, step 2 (Kaplan hparams + long warmup) vs step 5 (tuned)
    det = A["ineff_detail"]
    g = det[(det.dataset == "rw") & (det.step == 2)].sort_values("C")
    ax[1].plot(g.C, g.shift_obs, "o", color=BLUE, ms=4, label="observed IsoFLOP argmins")
    ax[1].plot(g.C, g.shift_pred_model, "-", color=BLUE, lw=1.3, label="prediction, $f''$ from fitted technology")
    ax[1].plot(g.C, g.shift_pred_local, "--", color=ORANGE, lw=1.3, label="prediction, local $f''$")
    ax[1].axhline(0, color=st.MUTED, lw=0.8, ls=":")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("Training compute $C$ (FLOPs)")
    ax[1].set_ylabel("$\\ln N^*_{\\mathrm{untuned}}-\\ln N^*_{\\mathrm{tuned}}$")
    ax[1].set_ylim(-1.35, 0.95)
    ax[1].text(0.97, 0.04, "RW: step 2 vs. step 5", transform=ax[1].transAxes, ha="right", va="bottom",
               fontsize=6.5, color=st.INK2)
    ax[1].legend(loc="upper left", fontsize=6.3, handlelength=1.6)
    ax[1].set_title("(b) Flexible inputs: argmin shift", loc="left")
    # (c) measurement: local exponent with embeddings omitted
    cur = B["curves"]
    for name, col in [("Besiroglu et al. (2024)", BLUE), ("Hoffmann et al. (2022)", ORANGE)]:
        g = cur[cur.param_set == name]
        ax[2].plot(g.Nm, g.a_m, color=col, lw=1.3, label=name.replace(" et al.", ""))
        ax[2].axhline(g.a_true.iloc[0], color=col, lw=0.8, ls="--")
    ax[2].axvspan(790, 1.58e9, color=st.GRID, alpha=0.8, lw=0)
    ax[2].text(3e4, 0.405, "Kaplan et al. range", fontsize=7, color=st.INK2)
    ax[2].set_xscale("log")
    ax[2].set_ylim(0.38, 1.02)
    ax[2].set_xlabel("Non-embedding parameters $N_m$")
    ax[2].set_ylabel("Local measured exponent $a_m$")
    ax[2].legend(loc="upper right", fontsize=6.5)
    ax[2].set_title("(c) Measurement: embeddings omitted", loc="left")
    fig.tight_layout(w_pad=0.8)
    st.savefig(fig, "m8_measurement_fig1")


def fig_steplaw(C):
    st.use()
    fig, ax = plt.subplots(2, 2, figsize=(st.WIDTH_FULL, 4.6))
    d = C["d"]
    nd = d[~d.diverged]
    # (a) inefficiency vs D
    cu = nd.groupby("cell").agg(D=("D", "first"), N=("N", "first"), u=("u", "mean")).reset_index()
    ax[0, 0].plot(cu.D, 100 * cu.u, "o", color=st.MUTED, ms=3.5, label="random config (cell mean)")
    sfa = C["sfa"]
    h = sfa[(sfa.dist == "exponential") & (sfa.sigma_v == "estimated")].iloc[0]   # best-fitting SFA variant
    Dg = np.logspace(np.log10(cu.D.min()), np.log10(cu.D.max()), 50)
    zd = np.log(Dg) - nd.lnD.mean()
    Eu = np.exp(h.g0 + h.g_d * zd)                     # exponential inefficiency: E[u] = sigma_u
    ax[0, 0].plot(Dg, 100 * Eu, color=BLUE, lw=1.3, label="SFA: $E[\\iota\\,|\\,D]$ (exponential)")
    for r, col, lab in [("porian_base", ORANGE, "fixed LR 3e-3, 0.5M batch"), ("best_fixed", AQUA, "best single fixed pair")]:
        ps = C["samples"][r]
        ax[0, 0].plot(ps.D, 100 * ps.u, "s", color=col, ms=3.2, label=lab)
    ax[0, 0].set_xscale("log")
    ax[0, 0].set_xlabel("Training tokens $D$")
    ax[0, 0].set_ylabel("Inefficiency $\\iota=\\ln L-\\ln L^*$ (%)")
    ax[0, 0].set_ylim(-0.2, 6.2)
    ax[0, 0].legend(fontsize=6.3, loc="upper right")
    ax[0, 0].set_title("(a) Inefficiency falls with $D$", loc="left")
    # (b), (c) E-profiles
    prof = C["prof"]
    for r, col, lab in [("frontier", BLUE, "frontier (min over grid)"), ("porian_base", ORANGE, "fixed LR 3e-3, 0.5M batch"),
                        ("porian_rule", AQUA, "Porian $N$-rule")]:
        g = prof[prof.rule == r].sort_values("E")
        ax[0, 1].plot(g.E, g.a, "-o", color=col, ms=2.5, lw=1.2, label=lab)
        ax[1, 0].plot(g.E, g.sigma_star, "-o", color=col, ms=2.5, lw=1.2, label=lab)
    ax[0, 1].set_xlabel("Irreducible loss $E$ (held fixed)")
    ax[0, 1].set_ylabel("Allocation exponent $a$")
    ax[0, 1].legend(fontsize=6.3, loc="lower left")
    ax[0, 1].set_title("(b) $a$ along the flat $E$ profile, by policy", loc="left")
    ax[1, 0].set_xlabel("Irreducible loss $E$ (held fixed)")
    ax[1, 0].set_ylabel("On-path substitution $\\sigma^*$")
    ax[1, 0].set_ylim(0.55, 0.9)
    ax[1, 0].legend(fontsize=6.3, loc="lower left")
    ax[1, 0].set_title("(c) $\\sigma^*$ across the same fits", loc="left")
    # (d) LR demand: unconditional vs batch-conditional D-elasticity
    opt = C["opt"]
    dem = C["dem"]
    unc = dem[(dem["var"] == "lr") & (dem.method == "smoothed argmin")].iloc[0]
    con = dem[(dem["var"] == "lr") & (dem.method == "conditional at batch = 0.52M tokens (smoothed)")].iloc[0]
    lnN0 = np.log(opt.N).mean()
    y_u = np.log(opt.lr_smooth) - unc.e_N * (np.log(opt.N) - lnN0)
    ax[1, 1].plot(opt.D, np.exp(y_u), "o", color=BLUE, ms=3.5, label=f"unconditional (batch optimized): {unc.e_D:+.2f}")
    cond = C["cond"]
    s256 = cond[(cond.bs == 256) & cond.interior]
    y_c = s256.lnlr_star - con.e_N * (np.log(s256.N) - lnN0)
    ax[1, 1].plot(s256.D, np.exp(y_c), "s", color=ORANGE, ms=3.2, label=f"batch fixed at 0.52M tokens: {con.e_D:+.2f}")
    Dg = np.logspace(np.log10(opt.D.min()), np.log10(opt.D.max()), 20)
    ax[1, 1].plot(Dg, np.exp(unc.const + unc.e_N * lnN0 + unc.e_D * np.log(Dg)), color=BLUE, lw=1.1)
    ax[1, 1].plot(Dg, np.exp(con.const + con.e_N * lnN0 + con.e_D * np.log(Dg)), color=ORANGE, lw=1.1)
    mid = np.exp(np.mean(y_c))
    Dm = np.exp(np.mean(np.log(s256.D)))
    Db = np.logspace(np.log10(1.5e10), np.log10(opt.D.max()), 10)
    ax[1, 1].plot(Db, mid * (Db / Dm) ** -0.32, color=st.MUTED, lw=1.0, ls="--", label="Bjorck et al. slope: $-0.32$")
    ax[1, 1].set_xscale("log")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_xlabel("Training tokens $D$")
    ax[1, 1].set_ylabel("Optimal LR (adjusted to mean $\\ln N$)")
    ax[1, 1].set_ylim(7e-4, 4.2e-3)
    ax[1, 1].legend(fontsize=6.3, loc="upper left")
    ax[1, 1].set_title("(d) LR demand: conditional vs unconditional", loc="left")
    fig.tight_layout(h_pad=1.0, w_pad=1.0)
    st.savefig(fig, "m8_measurement_fig2_steplaw")


def make_all(A, B, C, S_=None):
    fig_measurement(A, B)
    fig_steplaw(C)
