"""figures.py -- figures of module ra1_modelfree (AER style: <= 3 colours per panel, no twin axes)."""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import ra1_common as rc  # noqa: E402

sys.path.insert(0, rc.ANALYSIS)
import aer_style as st  # noqa: E402

DES = ["Chinchilla", "Llama 3", "Marin, Comma", "Marin, DCLM", "Marin, Nemotron-CC", "Porian, RefinedWeb",
       "Porian, OpenWebText2"]


def fig_sigma(iso, far):
    """(a) sigma* by design: model-free vs parametric on the same runs; (b) by budget (small multiples)."""
    st.use()
    s = iso["summary"].set_index("design")
    par = iso["param"].set_index("design")
    b = iso["budgets"]
    fp = far["pool"].set_index("conv").loc["ne"]
    fsig = far["sigma"].set_index(["conv", "spec"])
    fig = plt.figure(figsize=(st.WIDTH_FULL, 7.4))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.3, 1, 1], hspace=0.5, wspace=0.32)
    ax = fig.add_subplot(gs[0, :])
    names = DES + ["Farseer"]
    y = np.arange(len(names))[::-1]
    for i, d in enumerate(names):
        if d == "Farseer":
            mf, mse = fp.sigma_re, fp.se_re
            k1, k1s = fsig.loc[("ne", "chin_full"), "sigma_star"], fsig.loc[("ne", "chin_full"), "se_wild_cluster"]
            kf, kfs = fsig.loc[("ne", "kappa_full"), "sigma_star"], fsig.loc[("ne", "kappa_full"), "se_wild_cluster"]
        else:
            mf, mse = s.loc[d, "sigma_re"], s.loc[d, "se_sigma_re"]
            k1, k1s = par.loc[d, "sigma_chin"], par.loc[d, "se_sigma_chin"]
            kf, kfs = par.loc[d, "sigma_kappa"], par.loc[d, "se_sigma_kappa"]
        lab = dict(label="Model-free") if i == 0 else {}
        ax.errorbar(mf, y[i] + 0.22, xerr=1.96 * mse, fmt="o", color=st.BLUE, ms=4.5, capsize=0, lw=1.3, **lab)
        ax.errorbar(kf, y[i], xerr=1.96 * kfs, fmt="D", color=st.AQUA, ms=3.8, capsize=0, lw=1.1,
                    **(dict(label=r"$\kappa$ free, same runs") if i == 0 else {}))
        ax.errorbar(k1, y[i] - 0.22, xerr=1.96 * k1s, fmt="s", color=st.ORANGE, ms=3.8, capsize=0, lw=1.1,
                    **(dict(label=r"Chinchilla form ($\kappa=1$), same runs") if i == 0 else {}))
    ax.set_yticks(y)
    ax.set_yticklabels([n if n != "Farseer" else "Farseer (local path)" for n in names])
    ax.set_xlabel(r"Elasticity of substitution on the expansion path, $\sigma^*$ (95% interval)")
    ax.set_xlim(0.2, 0.9)
    ax.legend(loc="lower left", ncol=3, fontsize=7.5, bbox_to_anchor=(0.0, 1.0))
    ax.set_title("(a) By design", loc="left", fontsize=9, pad=18)
    for j, d in enumerate(names):
        a2 = fig.add_subplot(gs[1 + j // 4, j % 4])
        if d == "Farseer":
            pth = far["path"]
            pth = pth[(pth.conv == "ne") & np.isfinite(pth.sigma_local_path) & np.isfinite(pth.se)]
            x = np.log10(pth.C.values)
            a2.errorbar(x, pth.sigma_local_path, yerr=[np.clip(pth.sigma_local_path - pth.lo, 0, None),
                                                       np.clip(pth.hi - pth.sigma_local_path, 0, None)],
                        fmt="o", color=st.BLUE, ms=3, lw=0.9, capsize=0)
            a2.axhline(fsig.loc[("ne", "chin_full"), "sigma_star"], color=st.ORANGE, lw=1.0, ls="--")
            a2.axhline(fsig.loc[("ne", "kappa_full"), "sigma_star"], color=st.AQUA, lw=1.0, ls=":")
            a2.plot(x, pth.sigma_eq3_full, color=st.INK2, lw=0.9, ls="-")
            a2.set_title("Farseer (local path)", fontsize=8)
        else:
            bb = b[b.design == d]
            x = np.log10(bb.budget_C.values)
            a2.errorbar(x, bb.sigma, yerr=[np.clip(bb.sigma - bb.lo_sigma, 0, None), np.clip(bb.hi_sigma - bb.sigma, 0, None)],
                        fmt="o", color=st.BLUE, ms=3, lw=0.9, capsize=0)
            a2.axhline(par.loc[d, "sigma_chin"], color=st.ORANGE, lw=1.0, ls="--")
            a2.axhline(par.loc[d, "sigma_kappa"], color=st.AQUA, lw=1.0, ls=":")
            a2.axhline(s.loc[d, "sigma_re"], color=st.BLUE, lw=0.8, alpha=0.6)
            a2.set_title(d, fontsize=8)
        a2.set_ylim(0.3, 1.0)
        a2.tick_params(labelsize=7)
        if j % 4 == 0:
            a2.set_ylabel(r"$\sigma^*$ by budget", fontsize=8)
        if j >= 4:
            a2.set_xlabel(r"$\log_{10}$ compute (FLOP)", fontsize=8)
    fig.text(0.01, 0.585, "(b) By compute budget: points = model-free $\\sigma^*_b$ (95% basic interval); solid = design mean; "
             "dashed = $\\kappa=1$; dotted = $\\kappa$ free; Farseer gray = Eq. 3", fontsize=7.5, color=st.INK2)
    st.savefig(fig, "ra1_modelfree_sigma_by_design", folder=rc.FIGS)


def fig_farseer(far):
    st.use()
    d = far["delta"]
    fig, axes = plt.subplots(1, 3, figsize=(st.WIDTH_FULL, 2.6), gridspec_kw=dict(wspace=0.35))
    bins = ["M<16", "16-64", "64-256", "256-1,024", ">=1,024"]
    mid = {"M<16": 6, "16-64": 32, "64-256": 128, "256-1,024": 512, ">=1,024": 1800}
    specs = [("chin_full", "Chinchilla, all runs", st.ORANGE, "s"), ("chin_M100", r"Chinchilla, fit $M\leq100$", st.INK2, "v"),
             ("kappa_full", r"$\kappa$ free, all runs", st.AQUA, "D")]
    for ax, conv, ttl in ((axes[0], "ne", "(a) Non-embedding $N$"), (axes[1], "emb", "(b) $N$ incl. embeddings")):
        for sp, lab, col, mk in specs:
            r = d[(d.spec == sp) & (d.conv == conv) & d.M_bin.isin(bins)].copy()
            r = r[np.isfinite(r.delta)]
            x = np.array([mid[m] for m in r.M_bin])
            ax.errorbar(x, r.delta, yerr=[np.clip(r.delta - r.lo, 0, None), np.clip(r.hi - r.delta, 0, None)],
                        fmt=mk + "-", color=col, ms=3.5, lw=1.1,
                        capsize=0, label=lab)
        ax.axhline(0, color=st.MUTED, lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel(r"Tokens per parameter $M$ (bin)")
        ax.set_title(ttl, loc="left", fontsize=9)
    axes[0].set_ylabel(r"$\ln(w_{\rm param}/w_{\rm local})$")
    axes[0].legend(fontsize=6.5, loc="lower left")
    ax = axes[2]
    g = far["grid"]
    ok = g.neff_ne >= 15
    ax.scatter(np.log(g.M[ok]), g.lnw_local_ne[ok], s=5, color=st.BLUE, alpha=0.5, label="Local (grid points)")
    xs = np.log(g.M[ok])
    for sp, lab, col in (("chin_full", r"Chinchilla ($\kappa=1$)", st.ORANGE), ("eq3_full", "Farseer Eq. 3", st.AQUA)):
        ax.scatter(xs, g[f"lnw_{sp}_ne"][ok], s=4, color=col, alpha=0.5, label=lab)
    ax.axhline(0, color=st.MUTED, lw=0.8)
    ax.set_xlabel(r"$\ln M$ (non-embedding)")
    ax.set_ylabel(r"$\ln w$")
    ax.set_title("(c) Wedge at grid points", loc="left", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper left", markerscale=2)
    st.savefig(fig, "ra1_modelfree_farseer_w", folder=rc.FIGS)


def fig_profile(ch):
    st.use()
    p = ch["profile"]
    fig, axes = plt.subplots(1, 2, figsize=(st.WIDTH_FULL, 2.6), gridspec_kw=dict(wspace=0.3))
    for ax, samp_key, ttl in ((axes[0], "full", "(a) Full design ($n=240$)"), (axes[1], "on-path", f"(b) On-path band ($n={ch['band_n']}$)")):
        q = p[p["sample"].str.startswith(samp_key)]
        for obj, stat, col, lab in (("huber", "LR", st.BLUE, "Huber: Laplace quasi-LR"),
                                    ("huber", "LR_KB", st.AQUA, "Huber: Koenker-Bassett LR"),
                                    ("gauss", "LR", st.ORANGE, "Gaussian LR")):
            r = q[q.objective == obj].sort_values("sigma_star")
            ax.plot(r.sigma_star, r[stat], color=col, lw=1.2, label=lab)
        ax.axhline(3.84, color=st.MUTED, lw=0.8, ls="--")
        ax.set_yscale("symlog", linthresh=10)
        ax.set_ylim(0, None)
        ax.set_xlabel(r"$\sigma^*$ (outer exponent $\kappa$ profiled out)")
        ax.set_title(ttl, loc="left", fontsize=9)
        ax.set_xlim(0.05, 0.99)
    axes[0].set_ylabel("Profile LR vs. unconstrained optimum")
    axes[1].legend(fontsize=6.5, loc="upper left")
    st.savefig(fig, "ra1_modelfree_kappa_profile", folder=rc.FIGS)


def fig_practitioner(pr):
    st.use()
    tab, dv, cu = pr
    from practitioner import overhead_homothetic
    fig, ax = plt.subplots(figsize=(st.WIDTH_HALF + 0.4, 2.7))
    k = np.exp(np.linspace(0, np.log(1000), 200))
    for sg, col in ((0.6, st.ORANGE), (0.7, st.BLUE), (0.8, st.AQUA)):
        ax.plot(k, overhead_homothetic(sg, k), color=col, lw=1.3, label=fr"$\sigma^*={sg}$")
    ax.plot(k, overhead_homothetic(0.74, k), color=st.MUTED, lw=1.0, ls="--", label=r"$\sigma^*=0.74$")
    ax.scatter(cu.k, 1 + cu.overhead, s=6, color=st.INK, zorder=5, label=r"de Vries (2023), $\sigma^*=0.77$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Over-training factor $k=M/M^*$")
    ax.set_ylabel(r"Training compute $C/C_{\min}$")
    ax.legend(fontsize=6.5)
    st.savefig(fig, "ra1_modelfree_overhead", folder=rc.FIGS)


def run(cache):
    iso, far, ch, pr = (cache(k) for k in ("isoflop", "farseer", "chinchilla", "practitioner"))
    fig_sigma(iso, far)
    fig_farseer(far)
    fig_profile(ch)
    fig_practitioner(pr)
