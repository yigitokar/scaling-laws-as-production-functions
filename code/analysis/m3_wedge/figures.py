"""figures.py -- figures of module m3_wedge (aer_style: <= 3 colours per panel, no twin axes)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from matplotlib.ticker import NullFormatter

from common import aer_style as S
from common import FIGS

P = "m3_wedge_"


def _logaxes(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")


# ----------------------------------------------------------------------------- Figure 4
LABELS = {"Meta-Llama-3-8B": "Llama 3 8B", "Llama-3.1-405B": "Llama 3.1 405B", "Qwen3-0.6B-Base": "Qwen3 0.6B",
          "gemma-3-27b-pt": "Gemma 3 27B", "SmolLM2-135M": "SmolLM2 135M", "DeepSeek-V3-Base": "DeepSeek-V3",
          "bloom": "BLOOM 176B", "pythia-70m": "Pythia 70M", "Qwen2.5-72B": "Qwen2.5 72B", "Cerebras-GPT-6.7B": "Cerebras-GPT 6.7B",
          "Llama-2-70b-hf": "Llama 2 70B", "OLMo-2-0325-32B": "OLMo 2 32B"}
OFFS = {"Meta-Llama-3-8B": (-60, 16), "Llama-3.1-405B": (22, -30), "Qwen3-0.6B-Base": (-58, 8), "gemma-3-27b-pt": (-66, 10),
        "SmolLM2-135M": (-12, -17), "DeepSeek-V3-Base": (6, -13), "bloom": (6, -4), "pythia-70m": (22, -34),
        "Qwen2.5-72B": (6, -12), "Cerebras-GPT-6.7B": (-20, -16), "Llama-2-70b-hf": (-30, 13), "OLMo-2-0325-32B": (-18, 13)}


def fig4(d, F, techs):
    S.use()
    fig, ax = plt.subplots(figsize=(S.WIDTH_FULL, 4.3))
    A = d[(d["sample"] == "A") & d["core"] & d["open_weights"]]
    B = d[(d["sample"] == "B") & d["core"]]
    MAIN = ["Llama-2", "Llama-3.1", "Qwen2.5", "Qwen3", "Qwen3-MoE", "Gemma-2", "Gemma-3", "OLMo-2", "SmolLM2", "Pythia", "Phi-3"]
    flag = set(F.loc[F["flagship"] & F["family"].isin(MAIN), "uid"])
    # partial-identification bands (union over technologies), behind the points
    for _, r in B.iterrows():
        ax.plot([r["M"], r["M"]], [r["band_lo"], r["band_hi"]], color=S.GRID, lw=1.0, zorder=1, solid_capstyle="butt")
    ax.scatter(A["M"], A["w_chin"], s=9, color=S.MUTED, alpha=0.55, lw=0, zorder=2, label="Other open-weight models (Epoch, unverified $D$)")
    dense = B[~B["moe"] & ~B["uid"].isin(flag)]
    moe = B[B["moe"] & ~B["uid"].isin(flag)]
    ax.scatter(dense["M"], dense["w_chin"], s=16, color=S.BLUE, lw=0.4, edgecolor="white", zorder=3,
               label="Verified open-weight base models")
    ax.scatter(moe["M"], moe["w_chin"], s=22, marker="s", facecolor="white", edgecolor=S.BLUE, lw=1.0, zorder=3,
               label="Verified, mixture-of-experts (active $N$)")
    fl = B[B["uid"].isin(flag)]
    ax.scatter(fl["M"], fl["w_chin"], s=30, color=S.ORANGE, lw=0.4, edgecolor="white", zorder=4,
               label="Flagships of the main families (largest model)")
    # Meta's models under Meta's own IsoFLOP law (lab-own technology)
    me = B[B["lab"].eq("Meta") & B["family"].isin(["Llama-2", "Llama-3", "Llama-3.1", "Llama-3.2"])]
    ax.scatter(me["M"], me["w_meta_a2"], s=26, marker="D", color=S.AQUA, lw=0.4, edgecolor="white", zorder=4,
               label="Meta models under Meta's own Llama 3 law")
    for _, r in me.iterrows():
        ax.plot([r["M"], r["M"]], [r["w_meta_a2"], r["w_chin"]], color=S.AQUA, lw=0.6, alpha=0.6, zorder=3)
    # sufficient-statistic curve of the reference technology: ln w = ((alpha+beta)/2) ln(M/M*(C)) at C = 1e24
    t = techs["chin"]
    Mg = np.logspace(-0.5, 5, 200)
    ax.plot(Mg, (Mg / t.mstar(1e24)) ** ((t.alpha + t.beta) / 2), color=S.INK2, lw=0.8, ls="--", zorder=2,
            label=r"Reference technology at $C=10^{24}$: $w=(M/M^*(C))^{(\alpha+\beta)/2}$")
    ax.axhline(1, color=S.INK, lw=0.8)
    ax.text(0.45, 1.04, "$w=1$: training-optimal", fontsize=7, color=S.INK, va="bottom")
    _logaxes(ax)
    ax.set_xlim(0.3, 1.1e5)
    ax.set_ylim(0.15, 40)
    ax.set_xlabel("Tokens per parameter, $M=D/N$")
    ax.set_ylabel(r"Revealed wedge $\hat w=\varepsilon_N/\varepsilon_D$ = lifetime / training compute")
    # design support of the technologies as bars along the top
    for i, (lab, lo, hi) in enumerate([("Chinchilla design", 0.46, 341), ("Gadre et al.", 5, 640), ("Farseer", 0.31, 2570)]):
        y = 30 * (0.78 ** i)
        ax.plot([lo, hi], [y, y], color=S.INK2, lw=2.2, alpha=0.35, solid_capstyle="butt")
        ax.text(hi * 1.15, y, lab + " ($M$ range)", fontsize=6.5, color=S.INK2, va="center")
    for _, r in B.iterrows():
        if r["model"] in LABELS:
            ax.annotate(LABELS[r["model"]], (r["M"], r["w_chin"]), xytext=OFFS.get(r["model"], (5, 5)),
                        textcoords="offset points", fontsize=6.5, color=S.INK,
                        arrowprops=dict(arrowstyle="-", color=S.MUTED, lw=0.5))
    ax.legend(loc="lower right", fontsize=7, handletextpad=0.3, borderaxespad=0.3)
    ax.set_yticks([0.25, 0.5, 1, 2, 4, 8, 16, 32])
    ax.set_yticklabels(["0.25", "0.5", "1", "2", "4", "8", "16", "32"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    S.savefig(fig, P + "fig4", folder=FIGS)


# ----------------------------------------------------------------------------- trends
def fig_trends(Tr, d):
    S.use()
    fig, axs = plt.subplots(1, 2, figsize=(S.WIDTH_FULL, 2.8))
    ax = axs[0]
    U = d[d["prod"] & d["year"].between(2019, 2026)]
    for o, c, lab in [(True, S.BLUE, "open weights"), (False, S.ORANGE, "closed / API")]:
        g = U[U["open_weights"] == o].groupby("year")["w_chin"]
        q = g.quantile([0.25, 0.5, 0.75]).unstack()
        n = g.size()
        q = q[n >= 3]
        ax.fill_between(q.index, q[0.25], q[0.75], color=c, alpha=0.15, lw=0)
        ax.plot(q.index, q[0.5], "-o", color=c, ms=3.5, label=lab)
    ax.axhline(1, color=S.INK, lw=0.7)
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 4, 8])
    ax.set_yticklabels(["0.5", "1", "2", "4", "8"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel(r"$\hat w$ (median, IQR)")
    ax.set_xlabel("Release year")
    ax.set_title("(a) Revealed wedge by release year", loc="left")
    ax.legend(loc="upper left")
    ax = axs[1]
    y = Tr[Tr["group"] == "year"].copy()
    y["year"] = y["value"].astype(int)
    ax.plot(y["year"], y["share_w_lt1"], "-o", color=S.ORANGE, ms=3.5, label=r"share $\hat w<1$ (under-trained)")
    ax.plot(y["year"], y["share_w_gt2"], "-s", color=S.BLUE, ms=3.5, label=r"share $\hat w>2$")
    for _, r in y.iterrows():
        ax.text(r["year"], -0.09, f"{int(r['n'])}", ha="center", fontsize=6, color=S.MUTED)
    ax.text(2018.6, -0.09, "$n$", fontsize=6, color=S.MUTED, ha="center")
    ax.set_ylim(-0.13, 1.05)
    ax.set_xlim(2018.3, 2026.5)
    ax.set_xlabel("Release year")
    ax.set_ylabel("Share of models")
    ax.set_title("(b) Under- and over-training", loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    S.savefig(fig, P + "trends", folder=FIGS)


# ----------------------------------------------------------------------------- families
FAMS = ["Llama-2", "Llama-3.1", "Qwen2.5", "Qwen3", "Gemma-2", "Gemma-3", "OLMo-2", "SmolLM2", "Pythia"]


def fig_family(F):
    S.use()
    fig, axs = plt.subplots(3, 3, figsize=(S.WIDTH_FULL, 5.6), sharey=True)
    for ax, fam in zip(axs.flat, FAMS):
        g = F[F["family"] == fam].sort_values("N")
        ax.plot(g["N"], g["w_abs"], "-o", color=S.BLUE, ms=3.5, lw=1.1, label=r"$\hat w$, reference technology")
        ax.plot(g["N"], g["wrel_chin"], "--s", color=S.ORANGE, ms=3.2, lw=1.1, label=r"$w^{rel}$ (flagship $w=1$)")
        if "wrel_chin_lo" in g:
            ax.fill_between(g["N"], g["wrel_chin_lo"], g["wrel_chin_hi"], color=S.ORANGE, alpha=0.12, lw=0)
        ax.axhline(1, color=S.INK, lw=0.6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(fam + ("  ($D$ varies)" if g["D_varies"].iloc[0] else "  (common $D$)"), loc="left", fontsize=8)
        ax.set_yticks([0.5, 1, 2, 4, 8, 16])
        ax.set_yticklabels(["0.5", "1", "2", "4", "8", "16"])
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
    for ax in axs[-1]:
        ax.set_xlabel("Parameters $N$")
    for ax in axs[:, 0]:
        ax.set_ylabel("wedge")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    S.savefig(fig, P + "family", folder=FIGS)


# ----------------------------------------------------------------------------- validation (FWL partial plot)
def fig_validation(Bv):
    S.use()
    fig, axs = plt.subplots(1, 2, figsize=(S.WIDTH_FULL, 2.8))
    x = Bv.dropna(subset=["ln_dlall", "lnM", "lnC", "ln_age"]).copy()
    for ax, y, title in [(axs[0], "ln_dlall", "(a) All-time downloads"), (axs[1], "ln_dl30", "(b) 30-day downloads")]:
        ry = smf.ols(f"{y} ~ lnC + C(yr) + ln_age", data=x).fit().resid
        rx = smf.ols("lnM ~ lnC + C(yr) + ln_age", data=x).fit().resid
        ax.scatter(rx, ry, s=10, color=S.BLUE, alpha=0.7, lw=0)
        b = np.polyfit(rx, ry, 1)
        xx = np.linspace(rx.min(), rx.max(), 10)
        ax.plot(xx, b[0] * xx + b[1], color=S.ORANGE, lw=1.4, label=f"slope {b[0]:.2f}")
        ax.set_xlabel(r"$\ln M$ net of $\ln C$, year, age")
        ax.set_ylabel(r"log downloads, net of same")
        ax.set_title(title, loc="left")
        ax.legend(loc="upper left")
    fig.tight_layout()
    S.savefig(fig, P + "validation", folder=FIGS)


# ----------------------------------------------------------------------------- rivals
def fig_rivals(R, d):
    S.use()
    fig, axs = plt.subplots(1, 2, figsize=(S.WIDTH_FULL, 2.9))
    ax = axs[0]
    h = R["bunch_hist"]
    ax.bar(h["lo"], h["n"], width=0.1, align="edge", color=S.BLUE, edgecolor="white", lw=0.5)
    ax.axvspan(np.log10(6.5e9), np.log10(9.5e9), color=S.ORANGE, alpha=0.12, lw=0)
    ax.annotate("6.5-9.5B: fits one\n16-24 GB GPU (16-bit)", (np.log10(9.5e9), h["n"].max() * 0.9), xytext=(10.5, h["n"].max() * 0.9),
                fontsize=6.5, color=S.INK2, va="center", arrowprops=dict(arrowstyle="-", color=S.MUTED, lw=0.5))
    ax.set_xticks([8, 9, 10, 11, 12])
    ax.set_xticklabels(["100M", "1B", "10B", "100B", "1T"])
    ax.set_xlabel("Total parameters (open weights, 2023-2026)")
    ax.set_ylabel("Models")
    ax.set_title("(a) Bunching at hardware-fit sizes", loc="left")
    ax = axs[1]
    c = R["eq3_Mstar_curve"]
    ax.axvspan(1.2e18, 3.5e21, color=S.GRID, alpha=0.7, lw=0)
    ax.text(2.5e19, 140, "Farseer\ndesign", fontsize=6.5, color=S.INK2, ha="center")
    ax.plot(c["C"], c["Mstar_chin"], "-", color=S.BLUE, lw=1.3)
    ax.plot(c["C"], c["Mstar_eq3"], "-", color=S.ORANGE, lw=1.3)
    ax.plot(c["C"], c["Mstar_meta_a2"], "-", color=S.AQUA, lw=1.3)
    ax.plot(c["C"], c["Mstar_farseer_chin"], "--", color=S.MUTED, lw=1.1)
    for col, lab, colr, dy in [("Mstar_chin", "Chinchilla refit", S.BLUE, 1.12), ("Mstar_eq3", "Farseer Eq. 3", S.ORANGE, 1.0),
                               ("Mstar_meta_a2", "Meta law", S.AQUA, 1.0), ("Mstar_farseer_chin", "Farseer,\nChinchilla form", S.MUTED, 0.82)]:
        ax.text(c["C"].iloc[-1] * 1.2, c[col].iloc[-1] * dy, lab, fontsize=6.5, color=S.INK2, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e19, 3e26)
    ax.set_yticks([10, 20, 50, 100, 200])
    ax.set_yticklabels(["10", "20", "50", "100", "200"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Training compute $C$ (FLOP)")
    ax.set_ylabel(r"Compute-optimal $M^*(C)$")
    ax.set_title("(b) Non-homotheticity: $M^*$ rising in $C$", loc="left")
    fig.tight_layout()
    S.savefig(fig, P + "rivals", folder=FIGS)


# ----------------------------------------------------------------------------- aggregate
def fig_aggregate(Ab):
    S.use()
    fig, ax = plt.subplots(figsize=(S.WIDTH_HALF + 0.6, 2.7))
    a = Ab[Ab["year"] != "all"].copy()
    a["year"] = a["year"].astype(int)
    a = a[a["year"] >= 2021]
    ax.fill_between(a["year"], a["band_trunc_lo"], a["band_trunc_hi"], color=S.GRID, alpha=0.9, lw=0, label="PI band (technologies)")
    ax.errorbar(a["year"], a["multiple_trunc_ref"],
                yerr=[a["multiple_trunc_ref"] - a["multiple_trunc_ref_lo"], a["multiple_trunc_ref_hi"] - a["multiple_trunc_ref"]],
                fmt="-o", color=S.BLUE, ms=3.5, lw=1.2, capsize=2, label="reference technology, 95% CI")
    ax.axhline(0, color=S.INK, lw=0.6)
    ax.set_xlabel("Release year")
    ax.set_ylabel(r"$\sum 2NT / \sum 6ND$ (with $T\geq 0$)")
    ax.set_title("Planned lifetime inference / training compute", loc="left", fontsize=8)
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    S.savefig(fig, P + "aggregate", folder=FIGS)
