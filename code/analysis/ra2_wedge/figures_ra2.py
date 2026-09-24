"""figures_ra2.py -- paper exhibits (i) ECDFs of the expenditure share s by technology on the clean sample with the
lab-own subsample highlighted (replaces the old near-deterministic w-vs-M figure; R3 M12), (v) trends of s by year
(counts on the plot; cells with n < 5 not drawn), and an appendix figure of the model-free curvature ratios.
Style: aer_style (at most 3 colours per panel, no twin axes)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import aer_style as S
from ra2common import share

REF = "chin_q"
S.use()


def _ecdf(ax, v, **kw):
    v = np.sort(np.clip(np.asarray(v, float), -1.0, 1.0))
    y = np.arange(1, len(v) + 1) / len(v)
    ax.step(np.r_[-1.0, v], np.r_[0.0, y], where="post", **kw)


def fig_ecdf(Bw, L, M, clean):
    exante = list(M.loc[M["in_set"], "key"])
    fig, axes = plt.subplots(1, 2, figsize=(S.WIDTH_FULL, 3.0), sharey=True)
    sub = Bw[clean]
    lab = sub[sub["w_lab"].notna()]
    for ax, X, title, primary_col, plab in [
            (axes[0], sub, f"(a) Clean sample (n = {len(sub)})", "s_primary", "lab-own where available, else reference"),
            (axes[1], lab, f"(b) Lab-own subsample (n = {len(lab)}: Meta, AI2, DeepSeek, Marin)", "s_lab", "lab-own technology")]:
        Lx = L[L["uid"].isin(X["uid"])]
        for i, k in enumerate(exante):
            v = share(Lx.loc[Lx["tech"] == k, "w"].dropna().values)
            if len(v):
                _ecdf(ax, v, color=S.MUTED, lw=0.6, alpha=0.55, label="each ex-ante technology" if i == 0 else None)
        vref = share(Lx.loc[Lx["tech"] == REF, "w"].values)
        _ecdf(ax, vref, color=S.BLUE, lw=1.6, label="reference (Chinchilla, $\\kappa$ free)")
        _ecdf(ax, X[primary_col].values, color=S.ORANGE, lw=1.6, label=plab)
        ax.axvline(0, color=S.INK2, lw=0.6, ls=":")
        ax.set_xlim(-1, 1)
        ax.set_xlabel("planned serving share of lifetime cost, $s=(w-1)/w$")
        ax.set_title(title, loc="left")
        ax.text(-0.95, 0.62, f"median $s$\nreference: {np.median(vref):.2f}\n{plab.split(',')[0]}: {np.median(X[primary_col]):.2f}",
                fontsize=7, color=S.INK2, va="top")
        ax.legend(loc="upper left", fontsize=6.5, handlelength=1.6)
        ax.text(-0.95, 0.40, "$s<-1$ ($w<0.5$) shown at $-1$", fontsize=6.5, color=S.INK2, va="top")
    axes[0].set_ylabel("share of models")
    S.savefig(fig, "ra2_wedge_ecdf")


def fig_trends(TRD):
    fig, ax = plt.subplots(figsize=(S.WIDTH_HALF + 0.6, 2.8))
    for ow, col, lab in [(True, S.BLUE, "open weights"), (False, S.ORANGE, "closed")]:
        g = TRD[(TRD["open_weights"] == ow) & (TRD["n"] >= 5)].sort_values("year")
        if ow:
            ax.fill_between(g["year"], g["q25"].clip(-1, 1), g["q75"].clip(-1, 1), color=col, alpha=0.15, lw=0,
                            label="open weights, interquartile range")
        ax.plot(g["year"], g["median_s"].clip(-1, 1), "-o", color=col, ms=3.5, label=f"{lab}, median")
        cl = g[g["median_s"] < -1]
        if len(cl):
            ax.plot(cl["year"], np.full(len(cl), -1.0), "v", color=col, ms=6, mfc="white")
        for _, r in g.iterrows():
            ax.annotate(f"{int(r['n'])}", (r["year"], np.clip(r["median_s"], -1, 1)), textcoords="offset points",
                        xytext=(0, 5 if (ow or r["median_s"] < -1) else -10), ha="center", fontsize=6.5, color=col)
    ax.axhline(0, color=S.INK2, lw=0.6, ls=":")
    ax.text(2019.95, -1.12, "open triangles: median below $-1$ (shown at $-1$); numbers: models per cell (cells with $n<5$ omitted)",
            fontsize=5.5, color=S.INK2, va="top")
    ax.set_ylim(-1.2, 1.0)
    ax.set_xlabel("release year")
    ax.set_ylabel("median $s=(w-1)/w$ (reference technology)")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.07), fontsize=6.5)
    S.savefig(fig, "ra2_wedge_trends")


def fig_modelfree(pb, tab, M=None):
    used = {} if M is None else {k.replace("_mf", ""): 1 / v - 1 for k, v in zip(M["key"], M["sigma_star"]) if k.endswith("_mf")}
    fig, ax = plt.subplots(figsize=(S.WIDTH_HALF + 0.6, 2.8))
    spec = [("meta", S.BLUE, "o", "Llama 3 (Meta)"), ("marin_comma", S.ORANGE, "s", "Marin: Comma"),
            ("marin_dclm", S.ORANGE, "^", "Marin: DCLM"), ("marin_nemotron", S.ORANGE, "v", "Marin: Nemotron-CC"),
            ("chin_iso", S.MUTED, "D", "Chinchilla (digitized)")]
    for key, col, mk, lab in spec:
        g = pb[pb["design"] == key]
        r = tab[(tab["design"] == key) & (tab["deg"] == 2)].iloc[0]
        ax.plot(g["budget"], g["ratio"], mk, color=col, ms=3.5, mfc="none" if key != "meta" else col,
                label=f"{lab}: pooled {r['S2']:.2f}, bias-corr. {r['S2_bc']:.2f}" + (f", ra1 {used[key]:.2f}" if key in used else ""))
    ax.set_xscale("log")
    ax.set_xlabel("IsoFLOP budget $C$ (FLOP)")
    ax.set_ylabel("$L_{nn}/(2|dL^*/dc|)=1/\\sigma^*-1$")
    ax.axhline(0.357, color=S.INK2, lw=0.6, ls="--")
    ax.text(pb["budget"].min(), 0.36, "Chinchilla $\\kappa=1$ refit", fontsize=6.5, color=S.INK2, va="bottom")
    ax.set_ylim(0.3, 1.75)
    ax.legend(fontsize=5.8, loc="upper right")
    S.savefig(fig, "ra2_wedge_modelfree")
