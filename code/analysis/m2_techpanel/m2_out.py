"""m2_out.py -- tables (CSV + LaTeX), figures, registry and bootstrap-draw files for m2_techpanel."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import aer_style  # noqa: E402
import m2_data as md  # noqa: E402
import m2_est as me  # noqa: E402

ROOT = md.ROOT
TAB = os.path.join(ROOT, "output", "tables")
FIG = os.path.join(ROOT, "output", "figures")
PROC = os.path.join(ROOT, "data", "processed", "m2_techpanel")
BOOTDIR = os.path.join(PROC, "boot")
for _d in (TAB, FIG, BOOTDIR):
    os.makedirs(_d, exist_ok=True)

LABEL = {
    "chinchilla|all": "Chinchilla (Epoch digitization)",
    "farseer|all": "Farseer",
    "gadre|C4": "Gadre et al.: C4",
    "gadre|RedPajama": "Gadre et al.: RedPajama",
    "gadre|RefinedWeb": "Gadre et al.: RefinedWeb",
    "gadre|CE": "Gadre et al.: 3 corpora, common exponents",
    "olmo_ladder|all": "OLMo ladder",
    "datablations|single_epoch": "Muennighoff et al.: single epoch",
    "datadecide|CE": "DataDecide: 25 recipes, common exponents",
}
PRIMARY_ORDER = ["chinchilla|all", "farseer|all", "gadre|C4", "gadre|RedPajama", "gadre|RefinedWeb", "gadre|CE",
                 "olmo_ladder|all", "datablations|single_epoch", "datadecide|CE"]
MKEYS = ["Mstar_1e21", "Mstar_1e23", "Mstar_1e25"]
DKEYS = ["E", "A", "B", "alpha", "beta", "a", "gamma", "sigma_star", "A_norm", "B_norm", "sigma_min", "sigma_max",
         "alpha_minus_beta"] + MKEYS


# ============================================================================ helpers
def f3(x, d=3):
    return "" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def fse(x, d=3):
    return "" if x is None or not np.isfinite(x) else f"({x:.{d}f})"


def fM(x):
    if x is None or not np.isfinite(x):
        return ""
    return f"{x:.1f}" if x < 100 else f"{x:.0f}"


def fci(lo, hi, fmt=fM):
    return "" if not (np.isfinite(lo) and np.isfinite(hi)) else f"[{fmt(lo)}, {fmt(hi)}]"


def boot_summary(draw_dicts, point):
    """SE (sd), 2.5/97.5 percentiles for every key; M* summarized on logs (skewed)."""
    out = {}
    for k in point:
        v = np.array([d[k] for d in draw_dicts], float)
        v = v[np.isfinite(v)]
        if len(v) < 3:
            out[k] = dict(se=np.nan, lo=np.nan, hi=np.nan, nvalid=len(v))
            continue
        out[k] = dict(se=float(np.std(v, ddof=1)), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)),
                      nvalid=len(v))
    return out


def tex_table(path, caption, label, colspec, header_lines, body_lines, notes, landscape=False, size="\\footnotesize"):
    lines = []
    if landscape:
        lines.append("\\begin{landscape}")
    lines += ["\\begin{table}[!htbp]", "\\centering", size, f"\\caption{{{caption}}}", f"\\label{{{label}}}",
              "\\begin{threeparttable}", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"]
    lines += header_lines + ["\\midrule"] + body_lines + ["\\bottomrule", "\\end{tabular}",
                                                          "\\begin{tablenotes}[flushleft]", "\\footnotesize",
                                                          f"\\item \\textit{{Notes:}} {notes}", "\\end{tablenotes}",
                                                          "\\end{threeparttable}", "\\end{table}"]
    if landscape:
        lines.append("\\end{landscape}")
    open(path, "w").write("\n".join(lines) + "\n")


def esc(s):
    return s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


# ============================================================================ technology summaries
def summarize_tech(tech):
    """One record per (variant, estimator) with point estimates, bootstrap SEs/CIs and design statistics."""
    recs = []
    V = tech["V"]
    for (k, est), r in tech["res"].items():
        v = V[k]
        d = v["df"]
        N, D, L = d.N.values, d.D.values, d.L.values
        cen = (np.exp(np.log(N).mean()), np.exp(np.log(D).mean()))
        pt = me.derived(r["theta"], N, D, centers=cen)
        bt = tech["boot"][(k, est)]
        ok = np.all(np.isfinite(bt[:, :5]), axis=1)
        dd = [me.derived(th, N, D, centers=cen) for th in bt[ok, :5]]
        bs = boot_summary(dd, pt)
        tl = me.translog(N, D, L, pt["E"])
        rb = bt[ok, 6]
        taub = np.array([me.tau_stat(a_, b_, c_) for a_, b_, c_ in bt[ok, 7:10]])
        rec = dict(key=k, dataset=v["dataset"], subset=v["subset"], estimator=est, role=v["role"], n_obs=len(d),
                   n_runs=d["run_id"].nunique(), n_clusters=d[v["cluster"]].nunique(),
                   cluster_unit=md.CLUSTER[v["dataset"]] if v["subset"] != "cluster_N" else "model size (25 clusters)",
                   B_boot=int(ok.sum()), B_failed=int((~ok).sum()), objective=r["obj"],
                   N_center=cen[0], D_center=cen[1], N_min=N.min(), N_max=N.max(), D_min=D.min(), D_max=D.max(),
                   M_min=(D / N).min(), M_max=(D / N).max(), sd_offpath=md.offpath_sd(N, D),
                   corr_lnN_lnD=float(np.corrcoef(np.log(N), np.log(D))[0, 1]),
                   translog_r=tl["r"], translog_r_lo=float(np.nanpercentile(rb, 2.5)) if np.isfinite(rb).sum() > 3 else np.nan,
                   translog_r_hi=float(np.nanpercentile(rb, 97.5)) if np.isfinite(rb).sum() > 3 else np.nan,
                   translog_tau=tl.get("tau", np.nan), translog_tau_lo=float(np.nanpercentile(taub, 2.5)),
                   translog_tau_hi=float(np.nanpercentile(taub, 97.5)),
                   share_boot_r_undefined=float(np.mean(~np.isfinite(rb))),
                   N_convention=v["nconv"], D_definition=v["ddef"], loss_units=v["units"], eval_set=v["eval_set"],
                   rmse_lnL=float(np.sqrt(np.mean((np.log(L) - np.log(me.chin_from_theta(r["theta"]).loss(N, D))) ** 2))))
        for kk in DKEYS:
            rec[kk] = pt[kk]
            rec["se_" + kk] = bs[kk]["se"]
            rec[kk + "_lo"] = bs[kk]["lo"]
            rec[kk + "_hi"] = bs[kk]["hi"]
        # store draws (natural units)
        fn = f"{k.replace('|', '__')}__{est}.npy"
        th = bt[ok, :5]
        np.save(os.path.join(BOOTDIR, fn), np.c_[np.exp(th[:, 2]), np.exp(th[:, 0]), np.exp(th[:, 1]), th[:, 3], th[:, 4]])
        rec["boot_file"] = os.path.join("data", "processed", "m2_techpanel", "boot", fn)
        recs.append(rec)
    return pd.DataFrame(recs)


def summarize_q(tech):
    """Generalized (outer-exponent) technology L = E + (A N^-alpha + B D^-beta)^q, Huber-LSE, pairs bootstrap.
    alpha, beta, A, B are the inner-aggregator parameters: isoquants, sigma and the wedge eps_N/eps_D = alpha u/(beta v)
    depend only on them; q scales the frontier (gamma_q = q alpha beta/(alpha+beta))."""
    recs = []
    for (k, est), r in tech["qres"].items():
        if est != "huber":
            continue
        v = tech["V"][k]
        d = v["df"]
        N, D = d.N.values, d.D.values
        th = r["theta"]
        qb = tech["spec"][(k, "q")]
        ok = np.all(np.isfinite(qb[:, :6]), axis=1)
        qb = qb[ok, :6]

        def fns(t):
            a, b, e, al, be, q = t
            m = me.chin_from_theta(np.array([a, b, e, al, be]))
            s = m.sigma(N, D)
            out = dict(E=np.exp(e), A=np.exp(a), B=np.exp(b), alpha=al, beta=be, q=q, a=be / (al + be),
                       gamma=q * al * be / (al + be), sigma_star=2 / (2 + al + be), sigma_min=float(s.min()),
                       sigma_max=float(s.max()))
            for C in me.CGRID:
                out[f"Mstar_{C:.0e}".replace("+", "")] = float(m.D_opt(C) / m.N_opt(C))
            return out
        pt = fns(th)
        bs = boot_summary([fns(t) for t in qb], pt)
        rec = dict(key=k, dataset=v["dataset"], subset=v["subset"], estimator="huber_q", role="generalized_form",
                   n_obs=len(d), n_runs=d["run_id"].nunique(), n_clusters=d["cluster"].nunique(),
                   cluster_unit=md.CLUSTER[v["dataset"]], B_boot=int(ok.sum()), sd_offpath=md.offpath_sd(N, D),
                   N_convention=v["nconv"], D_definition=v["ddef"], loss_units=v["units"], eval_set=v["eval_set"],
                   notes="generalized form L = E + (A N^-alpha + B D^-beta)^q (Huber-LSE); alpha, beta, A, B are inner-"
                         "aggregator parameters; sigma, M* and the wedge eps_N/eps_D = alpha u/(beta v) use them directly")
        for kk in pt:
            rec[kk] = pt[kk]
            rec["se_" + kk] = bs[kk]["se"]
            rec[kk + "_lo"] = bs[kk]["lo"]
            rec[kk + "_hi"] = bs[kk]["hi"]
        fn = f"{k.replace('|', '__')}__huber_q.npy"
        np.save(os.path.join(BOOTDIR, fn), np.c_[np.exp(qb[:, 2]), np.exp(qb[:, 0]), np.exp(qb[:, 1]), qb[:, 3], qb[:, 4], qb[:, 5]])
        rec["boot_file"] = os.path.join("data", "processed", "m2_techpanel", "boot", fn)
        recs.append(rec)
    return pd.DataFrame(recs)


def summarize_panel_ce(neu, A):
    """Common-exponent (plants) technology rows for Gadre and DataDecide: alpha, beta and functions of them."""
    recs = []
    for key, ds, lab in (("gadre", "gadre", "CE"), ("datadecide", "datadecide", "CE")):
        o = neu[key]
        df = A[ds]
        for est in ("huber", "nls"):
            th = o["fits"][("CE", est)]["theta"]
            bt = o["ce_boot"][est]
            al, be = th[0], th[1]
            alb, beb = bt[:, 0], bt[:, 1]

            def fns(a_, b_):
                return dict(alpha=a_, beta=b_, a=b_ / (a_ + b_), gamma=a_ * b_ / (a_ + b_), sigma_star=2 / (2 + a_ + b_),
                            alpha_minus_beta=a_ - b_)
            pt = fns(al, be)
            dr = [fns(a_, b_) for a_, b_ in zip(alb, beb)]
            bs = boot_summary(dr, pt)
            R = o["R"]
            lnE = th[2 + 2 * R:2 + 3 * R]
            rec = dict(key=f"{ds}|CE", dataset=ds, subset="CE_pooled", estimator=est, role="primary", n_obs=o["n"],
                       n_runs=df["run_id"].nunique(), n_clusters=o["n_clusters"],
                       cluster_unit="(size, multiplier) cell" if ds == "gadre" else "(size, seed) cell",
                       B_boot=len(bt), B_failed=0, sd_offpath=md.offpath_sd(df.N.values, df.D.values),
                       corr_lnN_lnD=float(np.corrcoef(np.log(df.N), np.log(df.D))[0, 1]),
                       N_min=df.N.min(), N_max=df.N.max(), D_min=df.D.min(), D_max=df.D.max(),
                       M_min=df.M.min(), M_max=df.M.max(), E=np.nan, A=np.nan, B=np.nan,
                       E_range=f"{np.exp(lnE).min():.3f}-{np.exp(lnE).max():.3f}",
                       N_convention=md.NCONV[ds], D_definition=md.DDEF[ds], loss_units=md.LUNITS[ds],
                       eval_set="C4 val" if ds == "gadre" else "C4-en val",
                       notes=f"common alpha, beta; {R} corpus/recipe-specific (A_r, B_r, E_r); see m2_neutrality_*.csv")
            for kk in pt:
                rec[kk] = pt[kk]
                rec["se_" + kk] = bs[kk]["se"]
                rec[kk + "_lo"] = bs[kk]["lo"]
                rec[kk + "_hi"] = bs[kk]["hi"]
            # sigma range over the sample using recipe-specific u, v
            g = df[{"gadre": "corpus", "datadecide": "recipe"}[ds]].map({r: i for i, r in enumerate(o["groups"])}).values
            P = me.panel_perobs("CE", th, g, R)
            u = np.exp(P[:, 0] - al * np.log(df.N.values)); v = np.exp(P[:, 1] - be * np.log(df.D.values))
            s = (al * u + be * v) / (al * u * (1 + be) + be * v * (1 + al))
            rec["sigma_min"], rec["sigma_max"] = float(s.min()), float(s.max())
            # recipe-specific M* at the three budgets (median across recipes reported in the pooled row)
            for C in me.CGRID:
                Ms = []
                for r in range(R):
                    m = me.chin_from_theta(np.r_[th[2 + r], th[2 + R + r], th[2 + 2 * R + r], al, be])
                    Ms.append(m.D_opt(C) / m.N_opt(C))
                rec[f"Mstar_{C:.0e}".replace("+", "")] = float(np.median(Ms))
            fn = f"{ds}__CE__{est}.npy"
            np.save(os.path.join(BOOTDIR, fn), bt)
            rec["boot_file"] = os.path.join("data", "processed", "m2_techpanel", "boot", fn)
            recs.append(rec)
    return pd.DataFrame(recs)


def summarize_recipes(neu, A):
    """Per-recipe (DataDecide) separate Chinchilla fits with run-level pairs bootstrap."""
    o = neu["datadecide"]
    dd = A["datadecide"]
    recs = []
    for est in ("huber", "nls"):
        for r, rec_name in enumerate(o["groups"]):
            sub = dd[dd.recipe == rec_name]
            N, D = sub.N.values, sub.D.values
            cen = (np.exp(np.log(N).mean()), np.exp(np.log(D).mean()))
            th = o["full"][est]["thetas"][r]
            pt = me.derived(th, N, D, centers=cen)
            bt = neu["recipe_boot"][(rec_name, est)]
            ok = np.all(np.isfinite(bt), axis=1)
            bs = boot_summary([me.derived(t, N, D, centers=cen) for t in bt[ok]], pt)
            row = dict(key=f"datadecide|{rec_name}", dataset="datadecide", subset=rec_name, estimator=est, role="recipe",
                       n_obs=len(sub), n_runs=sub.run_id.nunique(), n_clusters=sub.run_id.nunique(),
                       cluster_unit="run (size x seed)", B_boot=int(ok.sum()), B_failed=int((~ok).sum()),
                       sd_offpath=md.offpath_sd(N, D), N_min=N.min(), N_max=N.max(), D_min=D.min(), D_max=D.max(),
                       M_min=(D / N).min(), M_max=(D / N).max(), N_center=cen[0], D_center=cen[1],
                       N_convention=md.NCONV["datadecide"], D_definition=md.DDEF["datadecide"],
                       loss_units=md.LUNITS["datadecide"], eval_set="C4-en val")
            for kk in DKEYS:
                row[kk] = pt[kk]
                row["se_" + kk] = bs[kk]["se"]
                row[kk + "_lo"] = bs[kk]["lo"]
                row[kk + "_hi"] = bs[kk]["hi"]
            fn = f"datadecide__{rec_name.replace(' ', '_').replace('/', '-').replace('%', 'pct').replace('(', '').replace(')', '').replace(',', '').replace('+', 'plus')}__{est}.npy"
            np.save(os.path.join(BOOTDIR, fn), np.c_[np.exp(bt[ok, 2]), np.exp(bt[ok, 0]), np.exp(bt[ok, 1]), bt[ok, 3], bt[ok, 4]])
            row["boot_file"] = os.path.join("data", "processed", "m2_techpanel", "boot", fn)
            recs.append(row)
    return pd.DataFrame(recs)


# ============================================================================ Table 3
def table3(S, Q=None):
    cols = ["key", "estimator", "n_obs", "n_runs", "E", "se_E", "alpha", "se_alpha", "beta", "se_beta", "a", "se_a",
            "gamma", "se_gamma", "sigma_star", "se_sigma_star", "sigma_star_lo", "sigma_star_hi", "sigma_min",
            "sigma_max", "Mstar_1e21", "Mstar_1e21_lo", "Mstar_1e21_hi", "Mstar_1e23", "Mstar_1e23_lo", "Mstar_1e23_hi",
            "Mstar_1e25", "sd_offpath", "B_boot"]
    T = S[S.key.isin(PRIMARY_ORDER)].copy()
    T["ord"] = T.key.map({k: i for i, k in enumerate(PRIMARY_ORDER)})
    T = T.sort_values(["estimator", "ord"], ascending=[True, True])
    T["dataset_label"] = T.key.map(LABEL)
    if Q is not None:   # generalized-form (q free) sigma*, Huber-LSE, attached to both panels for reference
        qq = Q.set_index("key")
        T["q_hat"] = T.key.map(qq["q"]); T["se_q_hat"] = T.key.map(qq["se_q"])
        T["sigma_star_q"] = T.key.map(qq["sigma_star"]); T["se_sigma_star_q"] = T.key.map(qq["se_sigma_star"])
        T["sigma_star_q_lo"] = T.key.map(qq["sigma_star_lo"]); T["sigma_star_q_hi"] = T.key.map(qq["sigma_star_hi"])
        cols = cols + ["q_hat", "se_q_hat", "sigma_star_q", "se_sigma_star_q", "sigma_star_q_lo", "sigma_star_q_hi"]
    T[["dataset_label"] + cols].to_csv(os.path.join(TAB, "m2_table3_technology.csv"), index=False)
    body = []
    for est, title in (("huber", "Panel A. Huber-LSE on log loss ($\\delta=10^{-3}$), reference estimator"),
                       ("nls", "Panel B. Gaussian NLS on log loss")):
        body.append(f"\\multicolumn{{13}}{{l}}{{\\textit{{{title}}}}} \\\\")
        for k in PRIMARY_ORDER:
            r = T[(T.key == k) & (T.estimator == est)]
            if r.empty:
                continue
            r = r.iloc[0]
            n = f"{int(r.n_runs)}" + (f" ({int(r.n_obs)})" if r.n_obs != r.n_runs else "")
            srange = f"{r.sigma_min:.2f}--{r.sigma_max:.2f}" if np.isfinite(r.sigma_min) else ""
            ce = k.endswith("|CE")
            body.append(" & ".join([esc(LABEL[k]), n, "--" if ce else f3(r.E), f3(r.alpha), f3(r.beta), f3(r.a), f3(r.gamma),
                                    f3(r.sigma_star), srange, fM(r.Mstar_1e21) + ("$^\\dagger$" if ce else ""),
                                    f"{r.sd_offpath:.2f}",
                                    f3(r.get("q_hat", np.nan), 2) if est == "huber" else "",
                                    f3(r.get("sigma_star_q", np.nan)) if est == "huber" else ""]) + " \\\\")
            body.append(" & ".join(["", "", "" if ce else fse(r.se_E), fse(r.se_alpha), fse(r.se_beta), fse(r.se_a),
                                    fse(r.se_gamma), fse(r.se_sigma_star), "",
                                    "" if ce else fci(r.Mstar_1e21_lo, r.Mstar_1e21_hi), "",
                                    fse(r.get("se_q_hat", np.nan), 2) if est == "huber" else "",
                                    fse(r.get("se_sigma_star_q", np.nan)) if est == "huber" else ""]) + " \\\\[2pt]")
        body.append("\\midrule" if est == "huber" else "")
    header = ["& & \\multicolumn{9}{c}{Chinchilla form ($q=1$)} & \\multicolumn{2}{c}{$q$ free} \\\\",
              "\\cmidrule(lr){3-11}\\cmidrule(lr){12-13}",
              "Dataset & Runs & $E$ & $\\alpha$ & $\\beta$ & $a$ & $\\gamma$ & $\\sigma^*$ & $\\sigma$ range & $M^*(10^{21})$ & sd$(\\ln M\\,|\\,\\ln C)$ & $\\hat q$ & $\\sigma^*_q$ \\\\"]
    notes = ("Technology $L=E+AN^{-\\alpha}+BD^{-\\beta}$ estimated separately for each public sweep. "
             "$a=\\beta/(\\alpha+\\beta)$ (allocation exponent, $N^*\\propto C^{a}$), $\\gamma=\\alpha\\beta/(\\alpha+\\beta)$ "
             "(frontier exponent of reducible loss), $\\sigma^*=2/(2+\\alpha+\\beta)$ (elasticity of substitution on the "
             "compute-optimal path); $\\sigma$ range is the local elasticity over the sample points; $M^*(10^{21})$ is the "
             "compute-optimal $D/N$ at $C=6ND=10^{21}$ FLOP with a 95\\% percentile interval in brackets (extrapolated "
             "for sweeps whose largest budget is below $10^{21}$). Standard errors in parentheses: pairs bootstrap over "
             "training runs (Chinchilla, Farseer, Gadre per corpus, Muennighoff; 400 draws), over (size, multiplier) cells "
             "(OLMo ladder; 400 draws), and over (size, multiplier) cells (Gadre, 200 draws) or (size, seed) cells (DataDecide, "
             "100 draws) for the common-exponent panels. Runs column: training runs (observations in parentheses; DataDecide uses intermediate "
             "checkpoints). Common-exponent rows impose equal $(\\alpha,\\beta)$ across corpora/recipes with corpus-specific "
             "$(A_r,B_r,E_r)$; $\\dagger$ median across corpora/recipes. Units differ across rows (nats/token with "
             "different tokenizers; Farseer in bits/character), so $E$ is not comparable across rows; $\\alpha$, $\\beta$, "
             "$a$, $\\gamma$ and $\\sigma$ are unit-free. Parameter-count conventions differ (see registry): Farseer non-embedding, "
             "Gadre and Chinchilla total, OLMo ladder and DataDecide excluding the input embedding only. "
             "DataDecide identification comes from checkpoints taken before the learning-rate schedule has finished. "
             "Last two columns: generalized form $L=E+(A'N^{-\\alpha'}+B'D^{-\\beta'})^{q}$ (Huber-LSE, 200 bootstrap draws), "
             "$\\sigma^*_q=2/(2+\\alpha'+\\beta')$; $q=1$ is rejected in every sweep (Table~\\ref{tab:spectests}).")
    tex_table(os.path.join(TAB, "m2_table3_technology.tex"), "The Parameter--Data Technology across Public Training Sweeps",
              "tab:technology", "l" + "c" * 12, header, body, notes, landscape=True)
    return T


# ============================================================================ specification tests
def spec_tests(S, tech):
    rows = []
    V = tech["V"]
    for k in [k for k in PRIMARY_ORDER if k in V and V[k]["spec"]]:
        d = V[k]["df"]
        n = len(d)
        rh = S[(S.key == k) & (S.estimator == "huber")].iloc[0]
        rn = S[(S.key == k) & (S.estimator == "nls")].iloc[0]
        bt = tech["boot"][(k, "huber")]
        ok = np.all(np.isfinite(bt[:, :5]), axis=1)
        dab = bt[ok, 3] - bt[ok, 4]
        z_ces = rh.alpha_minus_beta / np.std(dab, ddof=1)
        p_ces_wald = 2 * stats.norm.sf(abs(z_ces))
        # LR (NLS): n ln(SSR_r/SSR_u)
        ssr_u = 2 * tech["res"][(k, "nls")]["obj"]
        ssr_ces = 2 * tech["cres"][(k, "nls")]["obj"]
        lr_ces = n * np.log(ssr_ces / ssr_u)
        al_ces_h = tech["cres"][(k, "huber")]["theta"][3]
        # Kaplan q
        qh = tech["qres"][(k, "huber")]["theta"]
        qn = tech["qres"][(k, "nls")]["theta"]
        qb = tech["spec"][(k, "q")]
        qok = np.all(np.isfinite(qb[:, :6]), axis=1)
        q_se = np.std(qb[qok, 5], ddof=1)
        p_q_wald = 2 * stats.norm.sf(abs((qh[5] - 1) / q_se))
        ssr_q = 2 * tech["qres"][(k, "nls")]["obj"]
        lr_q = n * np.log(ssr_u / min(ssr_q, ssr_u))
        sq = 2 / (2 + qb[qok, 3] + qb[qok, 4])
        # translog rank-one: residual bootstrap null (tau is always defined; r only if both own curvatures > 0)
        nul = np.asarray(tech["spec"][(k, "null")], float)
        if nul.ndim == 1:                      # old cache format: r only
            nul = np.c_[nul, np.full(len(nul), np.nan)]
        rnull = nul[np.isfinite(nul[:, 0]), 0]
        tnull = nul[np.isfinite(nul[:, 1]), 1]
        r_hat, t_hat = rh.translog_r, rh.translog_tau
        if len(tnull):
            p_r = 2 * min(np.mean(tnull <= t_hat), np.mean(tnull >= t_hat))
            p_r = min(1.0, max(p_r, 1.0 / (len(tnull) + 1)))
        else:
            p_r = np.nan
        # ---- review fix: rank-one test with E from the generalized (q) family. The q family also implies a
        # rank-one Hessian of ln(L - E); with the Chinchilla E-hat, q != 1 alone produces tau < 0.
        tl_q = me.translog(d.N.values, d.D.values, d.L.values, float(np.exp(qh[2])))
        t_q = tl_q.get("tau", np.nan)
        tq_pairs = qb[qok, 7] if qb.shape[1] > 7 else np.full(int(qok.sum()), np.nan)
        nq = np.asarray(tech["spec"].get((k, "null_q"), np.full((0, 2), np.nan)), float).reshape(-1, 2)
        tq_null = nq[np.isfinite(nq[:, 1]), 1]
        if len(tq_null) and np.isfinite(t_q):
            p_rq = 2 * min(np.mean(tq_null <= t_q), np.mean(tq_null >= t_q))
            p_rq = min(1.0, max(p_rq, 1.0 / (len(tq_null) + 1)))
        else:
            p_rq = np.nan
        rows.append(dict(key=k, dataset_label=LABEL[k], n=n,
                         alpha_minus_beta=rh.alpha_minus_beta, se_alpha_minus_beta=float(np.std(dab, ddof=1)),
                         p_ces_wald=p_ces_wald, LR_ces_nls=lr_ces, p_ces_LR=stats.chi2.sf(lr_ces, 1),
                         sigma_ces=1 / (1 + al_ces_h), alpha_ces=al_ces_h,
                         q_huber=qh[5], se_q=q_se, p_q1_wald=p_q_wald, q_nls=qn[5], LR_q_nls=lr_q,
                         p_q1_LR=stats.chi2.sf(lr_q, 1), sigma_star_q=2 / (2 + qh[3] + qh[4]),
                         sigma_star_q_lo=np.percentile(sq, 2.5), sigma_star_q_hi=np.percentile(sq, 97.5),
                         translog_r=r_hat, translog_r_lo=rh.translog_r_lo, translog_r_hi=rh.translog_r_hi,
                         r_null_median=float(np.median(rnull)) if len(rnull) else np.nan,
                         r_null_lo=float(np.percentile(rnull, 2.5)) if len(rnull) else np.nan,
                         r_null_hi=float(np.percentile(rnull, 97.5)) if len(rnull) else np.nan, p_rank_one=p_r,
                         tau=t_hat, tau_lo=rh.translog_tau_lo, tau_hi=rh.translog_tau_hi,
                         tau_null_median=float(np.median(tnull)) if len(tnull) else np.nan,
                         tau_null_lo=float(np.percentile(tnull, 2.5)) if len(tnull) else np.nan,
                         tau_null_hi=float(np.percentile(tnull, 97.5)) if len(tnull) else np.nan,
                         share_null_r_undefined=float(1 - len(rnull) / len(nul)),
                         B_q=int(qok.sum()), B_null=len(nul),
                         E_q=float(np.exp(qh[2])), tau_q=t_q,
                         tau_q_lo=float(np.nanpercentile(tq_pairs, 2.5)) if np.isfinite(tq_pairs).sum() > 3 else np.nan,
                         tau_q_hi=float(np.nanpercentile(tq_pairs, 97.5)) if np.isfinite(tq_pairs).sum() > 3 else np.nan,
                         tau_q_null_median=float(np.median(tq_null)) if len(tq_null) else np.nan,
                         tau_q_null_lo=float(np.percentile(tq_null, 2.5)) if len(tq_null) else np.nan,
                         tau_q_null_hi=float(np.percentile(tq_null, 97.5)) if len(tq_null) else np.nan,
                         p_rank_one_q=p_rq, B_null_q=len(tq_null)))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(TAB, "m2_spec_tests.csv"), index=False)

    def fp(p):
        return "$<$0.001" if p < 0.001 else f"{p:.3f}"
    def fpmin(p, B):
        if not np.isfinite(p):
            return ""
        return ("$\\leq$" + f"{p:.3f}") if p <= 1.0 / (B + 1) + 1e-12 else fp(p)
    body = []
    for _, r in T.iterrows():
        body.append(" & ".join([esc(r.dataset_label), f3(r.alpha_minus_beta), fp(r.p_ces_wald), fp(r.p_ces_LR), f3(r.sigma_ces),
                                f3(r.q_huber, 2), fp(r.p_q1_wald), fp(r.p_q1_LR), f3(r.sigma_star_q),
                                f3(r.tau, 3), fpmin(r.p_rank_one, r.B_null),
                                f3(r.tau_q, 3), fpmin(r.p_rank_one_q, r.B_null_q)]) + " \\\\")
        body.append(" & ".join(["", fse(r.se_alpha_minus_beta), "", "", "", fse(r.se_q, 2), "", "",
                                fci(r.sigma_star_q_lo, r.sigma_star_q_hi, lambda x: f"{x:.2f}"),
                                fci(r.tau_lo, r.tau_hi, lambda x: f"{x:.3f}"), "",
                                fci(r.tau_q_lo, r.tau_q_hi, lambda x: f"{x:.3f}"), ""]) + " \\\\[2pt]")
    header = ["& \\multicolumn{4}{c}{CES: $\\alpha=\\beta$} & \\multicolumn{4}{c}{Outer exponent: $q=1$} & \\multicolumn{4}{c}{Rank-one curvature of $\\ln(L-\\hat E)$} \\\\",
              "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}",
              "& & & & & & & & & \\multicolumn{2}{c}{$\\hat E$ Chinchilla} & \\multicolumn{2}{c}{$\\hat E$ $q$ family} \\\\",
              "\\cmidrule(lr){10-11}\\cmidrule(lr){12-13}",
              "Dataset & $\\hat\\alpha-\\hat\\beta$ & $p$ Wald & $p$ LR & $\\sigma_{CES}$ & $\\hat q$ & $p$ Wald & $p$ LR & $\\sigma^*_q$ & $\\hat\\tau$ & $p$ & $\\hat\\tau_q$ & $p$ \\\\"]
    notes = ("CES: Wald test of $\\alpha=\\beta$ with the bootstrap standard error of $\\hat\\alpha-\\hat\\beta$ (Huber-LSE) and "
             "quasi-LR test $n\\ln(SSR_r/SSR_u)$ from Gaussian NLS, $\\chi^2_1$ reference; $\\sigma_{CES}=1/(1+\\alpha)$ from the restricted "
             "Huber fit. Outer exponent: $L=E+(A'N^{-\\alpha'}+B'D^{-\\beta'})^{q}$ nests Chinchilla ($q=1$) and the Kaplan et al. (2020) "
             "form ($q=\\alpha_D\\approx0.1$); isoquants depend only on the inner exponents, so $\\sigma^*_q=2/(2+\\alpha'+\\beta')$ (95\\% "
             "bootstrap interval in brackets, 200 draws); Wald uses the bootstrap standard error of $\\hat q$ (Huber-LSE), LR uses Gaussian NLS. "
             "Rank-one curvature: $\\hat\\tau=\\det H/\\lVert H\\rVert_F^2$ for the Hessian $H$ of an OLS translog of $\\ln(L-\\hat E)$ on "
             "$(\\ln N,\\ln D)$; Chinchilla and the $q$ family both imply a rank-one Hessian ($\\tau=0$) whose null direction is the expansion "
             "path, but each with its own $E$. Left pair: $\\hat E$ from the Chinchilla fit, null distribution from a residual bootstrap "
             "under the fitted Chinchilla model; right pair: $\\hat E$ from the $q$ family, null from a residual bootstrap under the fitted "
             "$q$ model ($E$ re-estimated in every draw; 300 draws each). Because an error in $\\hat E$ adds a constant to reducible loss, "
             "the left pair rejects whenever $q\\neq1$ even if curvature is rank one (a noise-free $q$-family surface on the Farseer "
             "design gives $\\hat\\tau=-0.27$ with the Chinchilla $\\hat E$); the right pair is the test of rank-one curvature. Pairs-"
             "bootstrap 95\\% intervals of $\\hat\\tau$ in brackets; $p$ is two-sided relative to the null distribution ($\\leq$ marks "
             "the smallest attainable value, $1/(B+1)$). Null medians and ranges are in the CSV file.")
    tex_table(os.path.join(TAB, "m2_spec_tests.tex"), "Specification Tests of the Chinchilla Technology", "tab:spectests",
              "l" + "c" * 12, header, body, notes, landscape=True)
    return T


# ============================================================================ Farseer forms and local sigma
def farseer_tables(far):
    ic, oos = far["ic"].copy(), far["oos"].copy()
    W = oos.pivot(index="model", columns="split", values="rmse_oos")
    T = ic.set_index("model").join(W)
    T.to_csv(os.path.join(TAB, "m2_farseer_forms.csv"))
    loc = far["local"].copy()
    loc.to_csv(os.path.join(TAB, "m2_farseer_local_sigma.csv"), index=False)
    far["bw_sens"].to_csv(os.path.join(TAB, "m2_farseer_local_sigma_bw.csv"), index=False)
    # ---- review addition: every summary of the local-sigma grid quoted in the memo, computed here (not by hand).
    # Bootstrap intervals re-use the pairs-bootstrap draws at the fixed bandwidth; they reflect sampling variance only
    # (no smoothing bias), so they are too narrow as statements about the smooth technology.
    lb = far["local_boot"]
    medb = np.nanmedian(lb, axis=1)
    X1 = np.c_[np.ones(len(loc)), np.log(loc.M)]
    X2 = np.c_[np.ones(len(loc)), np.log(loc.N)]
    X3 = np.c_[np.ones(len(loc)), np.log(loc.M), np.log(loc.N)]

    def ols(X, y):
        return np.linalg.lstsq(X, y, rcond=None)[0]
    b1, b2, b3 = ols(X1, loc.sigma_np.values), ols(X2, loc.sigma_np.values), ols(X3, loc.sigma_np.values)
    B1 = np.array([ols(X1, r) for r in lb]); B2 = np.array([ols(X2, r) for r in lb]); B3 = np.array([ols(X3, r) for r in lb])
    inb = (loc.sigma_chin >= loc.sigma_np_lo) & (loc.sigma_chin <= loc.sigma_np_hi)
    summ = [("median local sigma", np.median(loc.sigma_np), *np.percentile(medb, [2.5, 97.5])),
            ("p10 local sigma", np.percentile(loc.sigma_np, 10), np.nan, np.nan),
            ("p90 local sigma", np.percentile(loc.sigma_np, 90), np.nan, np.nan),
            ("min local sigma", loc.sigma_np.min(), np.nan, np.nan),
            ("max local sigma", loc.sigma_np.max(), np.nan, np.nan),
            ("median Farseer-Eq.3 sigma at grid points", np.median(loc.sigma_farseer), np.nan, np.nan),
            ("min Chinchilla sigma at grid points", loc.sigma_chin.min(), np.nan, np.nan),
            ("max Chinchilla sigma at grid points", loc.sigma_chin.max(), np.nan, np.nan),
            ("grid points", len(loc), np.nan, np.nan),
            ("grid points with Chinchilla sigma inside pointwise band", int(inb.sum()), np.nan, np.nan),
            ("slope on ln M (simple OLS)", b1[1], *np.percentile(B1[:, 1], [2.5, 97.5])),
            ("slope on ln N (simple OLS)", b2[1], *np.percentile(B2[:, 1], [2.5, 97.5])),
            ("slope on ln M holding ln N (joint OLS)", b3[1], *np.percentile(B3[:, 1], [2.5, 97.5])),
            ("slope on ln N holding ln M (joint OLS)", b3[2], *np.percentile(B3[:, 2], [2.5, 97.5]))]
    pd.DataFrame(summ, columns=["statistic", "value", "boot_lo", "boot_hi"]).to_csv(
        os.path.join(TAB, "m2_farseer_local_sigma_summary.csv"), index=False)
    # ---- review addition: compute-optimal D/N implied by the fitted Farseer Eq. 3 (non-homothetic form); the
    # Farseer design spans C = 1.2e18 to 3.5e21, so the last two budgets are extrapolations
    from scipy.optimize import minimize_scalar
    thF = far["farseer_theta"]
    mrows = []
    for C in (1e19, 1e20, 1e21, 1e22, 1e23):
        obj = lambda ln: float(me.farseer_lnL(thF, np.array([np.exp(ln)]), np.array([C / (6 * np.exp(ln))]))[0])
        rr = minimize_scalar(obj, bounds=(np.log(1e7), np.log(1e12)), method="bounded", options=dict(xatol=1e-8))
        Nst = float(np.exp(rr.x))
        mrows.append(dict(C=C, N_star=Nst, D_star=C / (6 * Nst), M_star=C / (6 * Nst ** 2), extrapolated=C > 3.5e21))
    pd.DataFrame(mrows).to_csv(os.path.join(TAB, "m2_farseer_eq3_Mstar.csv"), index=False)
    splits = list(W.columns)
    body = []
    order = ["Chinchilla (Huber)", "Chinchilla (NLS)", "Kaplan-q (NLS)", "Translog in ln(L-E), E free (NLS)", "Farseer Eq. 3 (NLS)"]
    for m in order:
        r = T.loc[m]
        body.append(" & ".join([esc(m), f"{int(r.k)}", f"{1000 * r.rmse_in:.2f}", f"{r.bic:.1f}"] +
                               [f"{1000 * r[s]:.2f}" for s in splits]) + " \\\\")
    # local sigma summary rows
    covered = np.mean((loc.sigma_chin >= loc.sigma_np_lo) & (loc.sigma_chin <= loc.sigma_np_hi))
    lab = ["Hold out 4 largest $N$", "Hold out top 10\\% of $C$"]
    header = ["& & \\multicolumn{2}{c}{In sample} & \\multicolumn{2}{c}{Out of sample RMSE $\\times 10^3$} \\\\",
              "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
              "Model & $k$ & RMSE $\\times 10^3$ & BIC & " + " & ".join(lab) + " \\\\"]
    notes = (f"Farseer grid, {int(far['ic'].shape[0] and 404)} runs; output: English bits per character. RMSE of log loss. "
             "Farseer Eq. 3: $L=e^{a_3N^{\\gamma_3}+b_3}+e^{a_2N^{\\gamma_2}+b_2}D^{-e^{a_1N^{\\gamma_1}+b_1}}$ (Li et al. 2025b), fitted "
             "here by NLS on log loss rather than the authors' differential piecewise procedure. Out-of-sample splits: "
             f"(i) train on $N<3.8\\times10^9$, test on the 4 largest model sizes; (ii) test on the top decile of $6ND$. Local "
             f"nonparametric $\\sigma$ (kernel-weighted local quadratic on $\\ln L$, E-free): median {np.median(loc.sigma_np):.3f} "
             f"(10th--90th percentile {np.percentile(loc.sigma_np, 10):.3f}--{np.percentile(loc.sigma_np, 90):.3f}) over "
             f"{len(loc)} grid points; the Chinchilla-implied local $\\sigma$ lies inside the pointwise 95\\% band at "
             f"{100 * covered:.0f}\\% of them. The pointwise bands come from a pairs bootstrap at a fixed bandwidth and ignore "
             "smoothing bias; the local estimates vary more across neighbouring model sizes than the bands suggest.")
    tex_table(os.path.join(TAB, "m2_farseer_forms.tex"), "Farseer: Chinchilla versus Richer Functional Forms",
              "tab:farseer", "lccccc", header, body, notes)
    return T


# ============================================================================ neutrality
MODEL_LABEL = {"pooled": "(0) One technology", "Eshift": "(iv) $E$-shift only", "hicks": "(i) Hicks-neutral, common $E$",
               "daug": "(ii) Data-augmenting, common $E$", "paug": "(iii) Parameter-augmenting, common $E$",
               "hicksE": "(i$'$) Hicks-neutral + $E_r$", "daugE": "(ii$'$) Data-augmenting + $E_r$",
               "paugE": "(iii$'$) Parameter-augmenting + $E_r$", "CE": "Common exponents: $A_r,B_r,E_r$ free",
               "full": "(v) Exponents vary"}


def neutrality_tables(neu, A):
    rows = []
    mags = []
    for key, lab in (("gadre", "Gadre et al.: 3 corpora"), ("datadecide", "DataDecide: 25 recipes"),
                     ("datadecide_avg11", "DataDecide: output = mean CE over 11 validation sets"),
                     ("datadecide_M20", "DataDecide: checkpoints with $D/N\\geq 20$")):
        if key not in neu:
            continue
        o = neu[key]
        ssr, k = o["ssr"], o["k"]
        den = ssr["pooled"] - ssr["full"]
        for m in ["pooled", "Eshift", "hicks", "daug", "paug", "hicksE", "daugE", "paugE", "CE", "full"]:
            t = o["tests"].get((m, "CE")) or (o["tests"].get(("CE", "full")) if m == "CE" else None)
            rows.append(dict(experiment=key, experiment_label=lab, model=m, k=k[m], ssr=ssr[m],
                             rmse=np.sqrt(ssr[m] / o["n"]), share_explained=(ssr["pooled"] - ssr[m]) / den,
                             test=("vs CE" if m not in ("CE", "full") else ("CE vs full" if m == "CE" else "")),
                             LR=t["LR"] if t else np.nan, df=t["df"] if t else np.nan, p_wild=t["p"] if t else np.nan,
                             B_wild=len(t["LRboot"]) if t else np.nan, n=o["n"], n_clusters=o["n_clusters"]))
        # ---- magnitudes under CE (NLS) and full
        R = o["R"]
        for est in ("nls", "huber"):
            th = o["fits"][("CE", est)]["theta"]
            al, be = th[0], th[1]
            lnA, lnB, lnE = th[2:2 + R], th[2 + R:2 + 2 * R], th[2 + 2 * R:2 + 3 * R]
            df = A["gadre" if key == "gadre" else "datadecide"]
            gcol = "corpus" if key == "gadre" else "recipe"
            # reference point: largest model at its largest D in the design (common across recipes)
            Nref = df.N.max(); Dref = df[df.N == Nref].D.max()
            Lref = np.exp(lnE) + np.exp(lnA) * Nref ** -al + np.exp(lnB) * Dref ** -be
            M21 = np.array([me.chin_from_theta(np.r_[lnA[r], lnB[r], lnE[r], al, be]).D_opt(1e21) /
                            me.chin_from_theta(np.r_[lnA[r], lnB[r], lnE[r], al, be]).N_opt(1e21) for r in range(R)])
            tilt = lnA - lnB
            full = o["full"][est]["thetas"]
            a_r = np.array([t[4] / (t[3] + t[4]) for t in full])
            s_r = np.array([2 / (2 + t[3] + t[4]) for t in full])
            rho = stats.spearmanr(Lref, np.log(M21)).correlation if R > 2 else np.nan
            # review addition: Spearman(E_r, ln M*_r) and the tilt range with cluster-bootstrap percentile intervals
            # (CE bootstrap draws; within a draw ln M*_r = const - 2/(alpha+beta) x tilt_r, so ln M* ranks = -tilt ranks)
            bt_ce = o["ce_boot"][est]
            Eb = np.exp(bt_ce[:, 2 + 2 * R:2 + 3 * R]); tb = bt_ce[:, 2:2 + R] - bt_ce[:, 2 + R:2 + 2 * R]
            rho_E = stats.spearmanr(np.exp(lnE), np.log(M21)).correlation if R > 2 else np.nan
            rho_Eb = np.array([stats.spearmanr(Eb[i], -tb[i]).correlation for i in range(len(bt_ce))]) if R > 2 else np.full(2, np.nan)
            rngb = np.ptp(tb, axis=1)
            mags_extra = dict(spearman_E_lnMstar=rho_E, spearman_E_lnMstar_lo=float(np.nanpercentile(rho_Eb, 2.5)),
                              spearman_E_lnMstar_hi=float(np.nanpercentile(rho_Eb, 97.5)),
                              tilt_range_lo=float(np.percentile(rngb, 2.5)), tilt_range_hi=float(np.percentile(rngb, 97.5)),
                              wedge_ratio=float(np.exp(tilt.max() - tilt.min())), B_ce=len(bt_ce))
            mags.append(dict(experiment=key, estimator=est, R=R, alpha_CE=al, beta_CE=be, sigma_star_CE=2 / (2 + al + be),
                             a_CE=be / (al + be), E_r_min=np.exp(lnE).min(), E_r_max=np.exp(lnE).max(),
                             tilt_range=tilt.max() - tilt.min(), Mstar21_min=M21.min(), Mstar21_max=M21.max(),
                             Mstar21_ratio=M21.max() / M21.min(), Lref_min=Lref.min(), Lref_max=Lref.max(),
                             N_ref=Nref, D_ref=Dref, spearman_Lref_lnMstar=rho,
                             sd_alpha_r=np.std([t[3] for t in full], ddof=1), sd_beta_r=np.std([t[4] for t in full], ddof=1),
                             sd_a_r=np.std(a_r, ddof=1), sd_sigma_r=np.std(s_r, ddof=1), min_sigma_r=s_r.min(),
                             max_sigma_r=s_r.max(), **mags_extra))
            if est == "nls":
                pd.DataFrame(dict(group=o["groups"], lnA=lnA, lnB=lnB, E=np.exp(lnE), tilt_lnA_minus_lnB=tilt,
                                  Mstar_1e21_CE=M21, L_ref=Lref, alpha_full=[t[3] for t in full],
                                  beta_full=[t[4] for t in full], a_full=a_r, sigma_star_full=s_r)).to_csv(
                    os.path.join(TAB, f"m2_neutrality_groups_{key}.csv"), index=False)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(TAB, "m2_neutrality.csv"), index=False)
    Mg = pd.DataFrame(mags)
    Mg.to_csv(os.path.join(TAB, "m2_neutrality_magnitudes.csv"), index=False)

    def fp(p):
        return "" if not np.isfinite(p) else ("$<$0.001" if p < 0.001 else f"{p:.3f}")
    body = []
    for key in ("gadre", "datadecide"):
        sub = T[T.experiment == key]
        if sub.empty:
            continue
        body.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{sub.experiment_label.iloc[0]} (n = {sub.n.iloc[0]}, "
                    f"{sub.n_clusters.iloc[0]} clusters)}}}} \\\\")
        for _, r in sub.iterrows():
            pmin = 1.0 / (r.B_wild + 1) if np.isfinite(r.B_wild) else np.nan
            pw = ("$\\leq$" + f"{r.p_wild:.3f}") if (np.isfinite(r.p_wild) and r.p_wild <= pmin + 1e-12) else fp(r.p_wild)
            body.append(" & ".join([MODEL_LABEL[r.model], f"{int(r.k)}", f"{1000 * r.rmse:.2f}",
                                    f"{100 * r.share_explained:.1f}", f3(r.LR, 1) if np.isfinite(r.LR) else "",
                                    "" if not np.isfinite(r.df) else f"{int(r.df)}", pw]) + " \\\\")
        body.append("\\midrule" if key == "gadre" else "")
    header = ["Model & Parameters & RMSE $\\times10^3$ & Share explained (\\%) & LR & df & $p$ (wild) \\\\"]
    notes = ("Each experiment trains one architecture on several corpora (Gadre et al.) or data recipes (DataDecide) and "
             "evaluates all models on the same C4 validation set. Models nest in $L_r=E_r+A_rN^{-\\alpha_r}+B_rD^{-\\beta_r}$: "
             "(i) Hicks-neutral in reducible loss, $A_r=e^{-\\omega_r}A$, $B_r=e^{-\\omega_r}B$; (ii) data-augmenting, only $B_r$ "
             "varies; (iii) parameter-augmenting, only $A_r$ varies; (iv) only $E_r$ varies; primes add recipe-specific $E_r$. "
             "Share explained $=(SSR_0-SSR_m)/(SSR_0-SSR_v)$ with model (0) one common technology and (v) separate fits. "
             "LR $=n\\ln(SSR_m/SSR_{CE})$ against the common-exponent model (for the common-exponent row: against (v)); Gaussian "
             "NLS on log loss. $p$-values from a wild cluster restricted bootstrap (Rademacher weights; clusters = (size, "
             "multiplier) cells for Gadre, 499 draws; (size, seed) cells for DataDecide, 99 draws for the key tests and 19 "
             "for the dominated common-$E$ models; bootstrap refits cap the solver at 300 evaluations, and uncapped refits "
             "of the largest $LR^*$ draws reproduce them to three decimals; $\\leq$ marks the smallest attainable value "
             "$1/(B+1)$). Every DataDecide LR exceeds the "
             "largest bootstrap $LR^*$, and Wald tests with the cluster-pairs bootstrap covariance of the common-exponent fit "
             "give the same conclusions. DataDecide uses intermediate checkpoints with $D/N\\geq5$ (see text).")
    tex_table(os.path.join(TAB, "m2_neutrality.tex"), "Is Better Data Hicks-Neutral? Nested Tests across Corpora and Recipes",
              "tab:neutrality", "lcccccc", header, body, notes)
    return T, Mg


def neutrality_wald(neu):
    """Review addition: cross-check of the wild-bootstrap LR tests. Wald tests of the linear restrictions on the
    common-exponent (CE) parameters, using the cluster-pairs bootstrap covariance of theta_CE (cells resampled whole,
    so the smooth misspecification is preserved within draws, unlike the sign-flipping wild bootstrap). Reported only
    when the number of bootstrap draws is at least 3x the number of restrictions."""
    rows = []
    for key in ("gadre", "datadecide", "datadecide_avg11", "datadecide_M20"):
        if key not in neu:
            continue
        o = neu[key]
        R = o["R"]
        for est in ("nls", "huber"):
            th = o["fits"][("CE", est)]["theta"]
            bt = o["ce_boot"][est]
            bt = bt[np.all(np.isfinite(bt), axis=1)]
            for kind in ("pooled", "Eshift", "hicksE", "daugE", "paugE"):
                Rm = []
                for r in range(1, R):
                    def e(i, j, c=1.0):
                        v = np.zeros(len(th)); v[i] = c; v[j] = -c; return v
                    if kind in ("pooled", "Eshift", "daugE"):
                        Rm.append(e(2 + r, 2))                       # lnA_r = lnA_0
                    if kind in ("pooled", "Eshift", "paugE"):
                        Rm.append(e(2 + R + r, 2 + R))               # lnB_r = lnB_0
                    if kind == "hicksE":
                        Rm.append(e(2 + r, 2) - e(2 + R + r, 2 + R))  # lnA_r - lnB_r = lnA_0 - lnB_0
                    if kind == "pooled":
                        Rm.append(e(2 + 2 * R + r, 2 + 2 * R))       # E_r = E_0
                Rm = np.array(Rm)
                q = len(Rm)
                if len(bt) < 3 * q:
                    continue
                rv = Rm @ th
                V = np.cov((bt @ Rm.T).T, ddof=1)
                W = float(rv @ np.linalg.solve(V, rv))
                dev = bt @ Rm.T - rv
                Wb = np.einsum("ij,ij->i", dev, np.linalg.solve(V, dev.T).T)
                rows.append(dict(experiment=key, estimator=est, H0=kind, restrictions=q, B=len(bt), wald=W,
                                 p_chi2=float(stats.chi2.sf(W, q)), p_boot=float((1 + np.sum(Wb >= W)) / (len(Wb) + 1))))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(TAB, "m2_neutrality_wald_check.csv"), index=False)
    return T


# ============================================================================ robustness
def robustness_table(S):
    keys = ["farseer|all", "farseer|N_incl_emb", "farseer|train_loss", "farseer|cluster_N",
            "gadre|C4", "gadre|C4_indist", "gadre|C4_Nnonemb", "gadre|C4_wls",
            "gadre|RedPajama", "gadre|RedPajama_indist", "gadre|RedPajama_Nnonemb", "gadre|RedPajama_wls",
            "gadre|RefinedWeb", "gadre|RefinedWeb_indist", "gadre|RefinedWeb_Nnonemb", "gadre|RefinedWeb_wls",
            "olmo_ladder|all", "olmo_ladder|task_bpb", "datablations|single_epoch", "datablations|single_epoch_M04",
            "datablations|le4_epochs"]
    lab = {"farseer|all": "Farseer: baseline", "farseer|N_incl_emb": "Farseer: $N$ incl. embeddings",
           "farseer|train_loss": "Farseer: training loss (nats/token)", "farseer|cluster_N": "Farseer: clusters = model size",
           "olmo_ladder|all": "OLMo ladder: baseline", "olmo_ladder|task_bpb": "OLMo ladder: task BPB output",
           "datablations|single_epoch": "Muennighoff: single epoch", "datablations|le4_epochs": "Muennighoff: $\\leq$4 epochs",
           "datablations|single_epoch_M04": "Muennighoff: single epoch, $D/N\\geq0.4$"}
    for c in ("C4", "RedPajama", "RefinedWeb"):
        lab[f"gadre|{c}"] = f"Gadre {c}: baseline"
        lab[f"gadre|{c}_indist"] = f"Gadre {c}: in-distribution eval"
        lab[f"gadre|{c}_Nnonemb"] = f"Gadre {c}: non-embedding $N$"
        lab[f"gadre|{c}_wls"] = f"Gadre {c}: CI-weighted NLS"
    rows = S[S.key.isin(keys)].copy()
    rows["ord"] = rows.key.map({k: i for i, k in enumerate(keys)})
    rows = rows.sort_values(["ord", "estimator"])
    rows.to_csv(os.path.join(TAB, "m2_robustness.csv"), index=False)
    body = []
    for k in keys:
        r = rows[rows.key == k]
        if r.empty:
            continue
        cells = [lab[k]]
        for est in ("huber", "nls"):
            x = r[r.estimator == est]
            if x.empty:
                cells += ["", "", ""]
                continue
            x = x.iloc[0]
            cells += [f"{x.sigma_star:.3f} {fse(x.se_sigma_star)}", f"{x.a:.3f} {fse(x.se_a)}", fM(x.Mstar_1e21)]
        body.append(" & ".join(cells) + " \\\\")
    header = ["& \\multicolumn{3}{c}{Huber-LSE} & \\multicolumn{3}{c}{Gaussian NLS} \\\\",
              "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
              "Specification & $\\sigma^*$ & $a$ & $M^*(10^{21})$ & $\\sigma^*$ & $a$ & $M^*(10^{21})$ \\\\"]
    notes = ("Bootstrap standard errors in parentheses (400 draws for baselines, 200 for variants). $N$ incl. embeddings adds the "
             "two untied 65,536-row embedding matrices to Farseer's non-embedding count. Training loss: Farseer's smoothed "
             "single-epoch training loss (nats/token) instead of English BPC. In-distribution eval: Paloma subset of the "
             "training corpus instead of C4. CI-weighted: weights $1/\\widehat{se}^2(\\ln L)$ from the reported token-level "
             "95\\% intervals. Muennighoff $\\leq$4 epochs treats up to 4 passes over unique data as fresh tokens; $D/N\\geq0.4$ drops "
             "the two single-epoch runs trained on 100M tokens with 1.1B and 2.8B parameters (the exclusion rule Besiroglu et al. "
             "apply to the Chinchilla sample).")
    tex_table(os.path.join(TAB, "m2_robustness.tex"), "Robustness of the Technology Estimates", "tab:techrobust",
              "lcccccc", header, body, notes)
    return rows


# ============================================================================ registry
def registry(S, CEr, Rr, Q=None):
    allr = pd.concat([S, CEr, Rr] + ([Q] if Q is not None else []), ignore_index=True, sort=False)
    cols = ["dataset", "subset", "estimator", "role", "E", "A", "B", "alpha", "beta", "se_E", "se_A", "se_B", "se_alpha",
            "se_beta", "a", "se_a", "gamma", "se_gamma", "sigma_star", "se_sigma_star", "sigma_star_lo", "sigma_star_hi",
            "A_norm", "se_A_norm", "B_norm", "se_B_norm", "N_center", "D_center", "sigma_min", "sigma_max",
            "Mstar_1e21", "Mstar_1e21_lo", "Mstar_1e21_hi", "Mstar_1e23", "Mstar_1e23_lo", "Mstar_1e23_hi",
            "Mstar_1e25", "Mstar_1e25_lo", "Mstar_1e25_hi", "sd_offpath", "corr_lnN_lnD", "n_obs", "n_runs", "n_clusters",
            "cluster_unit", "B_boot", "N_min", "N_max", "D_min", "D_max", "M_min", "M_max",
            "N_convention", "D_definition", "loss_units", "eval_set", "boot_file", "q", "se_q", "q_lo", "q_hi", "notes"]
    for c in cols:
        if c not in allr:
            allr[c] = np.nan
    allr["notes"] = allr["notes"].fillna("")
    allr.loc[allr.dataset == "datadecide", "notes"] = allr.loc[allr.dataset == "datadecide", "notes"].astype(str) + \
        " identification from intermediate checkpoints (LR schedule incomplete); treat levels with caution"
    main = allr.estimator.isin(["huber", "nls"])
    allr.loc[allr.key.eq("farseer|all") & main, "notes"] = "headline Farseer row (inference-wedge module): non-embedding N, English BPC"
    allr.loc[allr.key.eq("gadre|RefinedWeb") & main, "notes"] = "headline Gadre-RefinedWeb row (inference-wedge module): total N, C4-val loss"
    allr.loc[allr.estimator.isin(["huber", "nls"]) & allr.q.isna(), "q"] = 1.0   # Chinchilla rows impose q = 1
    out = allr[cols]
    out.to_csv(os.path.join(TAB, "technology_registry_m2.csv"), index=False)
    return out


# ============================================================================ figures
def fig_forest(S, CEr, Rr, Q=None, far=None):
    import matplotlib.pyplot as plt
    aer_style.use()
    fig, axes = plt.subplots(1, 2, figsize=(aer_style.WIDTH_FULL, 5.0), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axes[0]
    allp = pd.concat([S, CEr] + ([Q] if Q is not None else []), ignore_index=True, sort=False)
    order = list(PRIMARY_ORDER)
    if far is not None:
        order.insert(2, "farseer|nonparametric")
    ylabels = []
    ests = (("huber", aer_style.BLUE, "o", 0.2, "Chinchilla form, Huber-LSE"),
            ("nls", aer_style.ORANGE, "s", 0.0, "Chinchilla form, Gaussian NLS"),
            ("huber_q", aer_style.AQUA, "D", -0.2, "outer exponent $q$ free, Huber-LSE"))
    for i, k in enumerate(order):
        y = len(order) - 1 - i
        if k == "farseer|nonparametric":
            ylabels.append((y, "Farseer, nonparametric (median)"))
            loc, lb = far["local"], far["local_boot"]
            med_b = np.nanmedian(lb, axis=1)
            ax.hlines(y, np.percentile(med_b, 2.5), np.percentile(med_b, 97.5), color=aer_style.INK2, lw=1.1)
            ax.plot(np.median(loc.sigma_np), y, "v", color=aer_style.INK2, ms=4.5, label="local nonparametric (E-free)")
            continue
        ylabels.append((y, LABEL[k].replace("Muennighoff et al.: single epoch", "Muennighoff et al. (1 epoch)")
                        .replace(", common exponents", ", pooled").replace("Chinchilla (Epoch digitization)", "Chinchilla (Epoch)")))
        for est, col, mk, dy, lab in ests:
            r = allp[(allp.key == k) & (allp.estimator == est)]
            if r.empty:
                continue
            r = r.iloc[0]
            hollow = k.startswith("datadecide")
            # interval drawn from the percentile bounds (the point need not be centred in a percentile interval)
            ax.hlines(y + dy, r.sigma_star_lo, r.sigma_star_hi, color=col, lw=1.1)
            ax.plot(r.sigma_star, y + dy, mk, color=col, mfc="white" if hollow else col, ms=3.8 if mk != "D" else 3.3,
                    label=lab if k == "chinchilla|all" else None)
    ax.axvspan(0.4, 0.7, color="#d9d8d3", alpha=0.6, lw=0, zorder=0)
    ax.text(0.55, -0.55, "capital-labor $\\sigma$: 0.4-0.7", ha="center", va="center", fontsize=7, color=aer_style.INK2)
    ax.set_yticks([y for y, _ in ylabels]); ax.set_yticklabels([l for _, l in ylabels])
    ax.set_xlim(0.35, 0.95); ax.set_ylim(-0.9, len(order) - 0.4)
    ax.set_xlabel("$\\sigma^*$ (elasticity of substitution on the compute-optimal path)")
    ax.set_title("(a) Public sweeps", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, handletextpad=0.3, fontsize=7,
              columnspacing=1.0)
    ax.grid(axis="y", visible=False)
    # ---- panel (b): DataDecide recipes
    ax = axes[1]
    R = Rr[Rr.estimator == "huber"].sort_values("sigma_star").reset_index(drop=True)
    for i, r in R.iterrows():
        ax.hlines(i, r.sigma_star_lo, r.sigma_star_hi, color=aer_style.BLUE, lw=0.9)
        ax.plot(r.sigma_star, i, "o", color=aer_style.BLUE, ms=3)
    ce = CEr[(CEr.dataset == "datadecide") & (CEr.estimator == "huber")].iloc[0]
    ax.axvline(ce.sigma_star, color=aer_style.AQUA, lw=1.2, ls="--", label=f"pooled, common exponents ({ce.sigma_star:.3f})")
    ax.axvspan(0.4, 0.7, color="#d9d8d3", alpha=0.6, lw=0, zorder=0)
    ax.set_yticks(range(len(R))); ax.set_yticklabels(R.subset.values, fontsize=6)
    ax.set_xlim(0.35, 0.95); ax.set_ylim(-0.8, len(R) - 0.2)
    ax.set_xlabel("$\\sigma^*$ (Huber-LSE)")
    ax.set_title("(b) DataDecide recipes", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.13), frameon=False, fontsize=7)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(w_pad=0.8)
    aer_style.savefig(fig, "m2_fig3_sigma_forest")


def fig_local_sigma(far):
    import matplotlib.pyplot as plt
    aer_style.use()
    loc = far["local"]
    Ns = sorted(loc.N.unique())
    pick = [Ns[i] for i in np.linspace(0, len(Ns) - 1, 4).round().astype(int)]
    fig, axes = plt.subplots(1, 4, figsize=(aer_style.WIDTH_FULL, 2.3), sharey=True)
    for ax, n0 in zip(axes, pick):
        s = loc[loc.N == n0].sort_values("M")
        ax.fill_between(s.M, s.sigma_np_lo, s.sigma_np_hi, color=aer_style.BLUE, alpha=0.18, lw=0)
        ax.plot(s.M, s.sigma_np, color=aer_style.BLUE, marker="o", ms=2.5, label="local, nonparametric")
        ax.plot(s.M, s.sigma_chin, color=aer_style.INK2, ls="--", lw=1.1, label="Chinchilla fit")
        ax.plot(s.M, s.sigma_farseer, color=aer_style.ORANGE, ls=":", lw=1.4, label="Farseer Eq. 3 fit")
        ax.set_xscale("log")
        from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
        cand = np.array([1, 3, 10, 30, 100, 300, 1000])
        tk = cand[(cand >= s.M.min() * 0.95) & (cand <= s.M.max() * 1.05)]
        ax.xaxis.set_major_locator(FixedLocator(tk))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_title(f"$N$ = {n0 / 1e9:.2g}B", loc="left")
        ax.set_xlabel("$D/N$")
    axes[0].set_ylabel("local $\\sigma$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, ["local, nonparametric (95% pointwise band)", "implied by Chinchilla fit", "implied by Farseer Eq. 3 fit"],
               loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(w_pad=0.5, rect=(0, 0.06, 1, 1))
    aer_style.savefig(fig, "m2_farseer_local_sigma")


def fig_neutrality(neu, Rr=None):
    """(a) Factor tilt vs asymptote, as deviations from each experiment's median (levels are not comparable across
    experiments: tokenizers, N conventions and DataDecide's schedule artifact differ). y = ln M*_r(1e21) - median under
    common exponents = -(2/(alpha+beta)) x (tilt_r - median tilt). (b) Allocation exponent a_r from separate per-recipe
    fits (DataDecide, Huber-LSE, run-level bootstrap 95% CIs) against E_r."""
    import matplotlib.pyplot as plt
    aer_style.use()
    fig, axes = plt.subplots(1, 2, figsize=(aer_style.WIDTH_FULL, 2.9))
    ax = axes[0]
    for key, col, mk, lab in (("datadecide", aer_style.BLUE, "o", "DataDecide recipes (25)"),
                              ("gadre", aer_style.ORANGE, "s", "Gadre et al. corpora (3)")):
        f = os.path.join(TAB, f"m2_neutrality_groups_{key}.csv")
        if not os.path.exists(f):
            continue
        G = pd.read_csv(f)
        x = G.E - G.E.median()
        y = np.log(G.Mstar_1e21_CE) - np.log(G.Mstar_1e21_CE).median()
        ax.scatter(x, y, s=16 if key == "datadecide" else 26, color=col, marker=mk, label=lab, zorder=3)
        if key == "datadecide":
            for nm in ("DCLM-Baseline (QC FW 3%)", "C4", "Dolma1.6++", "FineWeb-Edu"):
                r = G[G.group == nm]
                if len(r):
                    ax.annotate(nm, (float(x[r.index[0]]), float(y[r.index[0]])), xytext=(4, -9 if nm == "C4" else 3),
                                textcoords="offset points", fontsize=6.5, color=aer_style.INK2)
    ax.axhline(0, color=aer_style.GRID, lw=0.8, zorder=0)
    ax.set_xlabel("$E_r$ minus experiment median (C4 validation, nats/token)")
    ax.set_ylabel("$\\ln M^*_r(10^{21})$ minus median")
    ax.set_title("(a) Factor tilt of recipes (common exponents)", loc="left")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax = axes[1]
    if Rr is not None:
        f = os.path.join(TAB, "m2_neutrality_groups_datadecide.csv")
        G = pd.read_csv(f).set_index("group")
        R = Rr[Rr.estimator == "huber"].set_index("subset")
        R = R.join(G[["E"]].rename(columns={"E": "E_CE"}))
        ax.errorbar(R.E_CE, R.a, yerr=[R.a - R.a_lo, R.a_hi - R.a], fmt="o", ms=3.2, color=aer_style.BLUE, lw=0.8,
                    ecolor=aer_style.BLUE, elinewidth=0.7, capsize=0)
        ce = neu["datadecide"]["fits"][("CE", "huber")]["theta"]
        ax.axhline(ce[1] / (ce[0] + ce[1]), color=aer_style.AQUA, ls="--", lw=1.1, label="common-exponent estimate")
        ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.set_xlabel("$E_r$ (C4 validation, nats/token)")
    ax.set_ylabel("allocation exponent $a_r=\\beta_r/(\\alpha_r+\\beta_r)$")
    ax.set_title("(b) DataDecide: separate fits by recipe", loc="left")
    fig.tight_layout(w_pad=1.2)
    aer_style.savefig(fig, "m2_neutrality_recipes")


def fig_designs(A):
    import matplotlib.pyplot as plt
    aer_style.use()
    panels = [("chinchilla", A["chinchilla"], "Chinchilla (Epoch)"), ("farseer", A["farseer"], "Farseer"),
              ("gadre", A["gadre"], "Gadre et al."), ("olmo_ladder", A["olmo_ladder"], "OLMo ladder"),
              ("datablations", A["datablations"][A["datablations"].epochs <= 1.0001], "Muennighoff (1 epoch)"),
              ("datadecide", A["datadecide"].drop_duplicates(["size", "step"]), "DataDecide (checkpoints)")]
    fig, axes = plt.subplots(2, 3, figsize=(aer_style.WIDTH_FULL, 4.0))
    for ax, (k, d, t) in zip(axes.ravel(), panels):
        ax.scatter(np.log10(d.N), np.log10(d.D), s=4, color=aer_style.BLUE, alpha=0.6, lw=0)
        lo, hi = np.log10(d.N).min() - 0.2, np.log10(d.N).max() + 0.2
        for lc in np.arange(15, 25):
            xx = np.linspace(lo, hi, 10)
            yy = lc - np.log10(6) - xx
            ax.plot(xx, yy, color=aer_style.GRID, lw=0.6, zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(np.log10(d.D).min() - 0.3, np.log10(d.D).max() + 0.3)
        ax.set_title(f"{t}: sd = {md.offpath_sd(d.N, d.D):.2f}", loc="left", fontsize=8)
        ax.set_xlabel("$\\log_{10} N$")
        ax.set_ylabel("$\\log_{10} D$")
    fig.tight_layout()
    aer_style.savefig(fig, "m2_design_planes")


# ============================================================================ entry
def write_all(A, tech, far, neu):
    S = summarize_tech(tech)
    Q = summarize_q(tech)
    CEr = summarize_panel_ce(neu, A)
    Rr = summarize_recipes(neu, A)
    table3(pd.concat([S, CEr], ignore_index=True, sort=False), Q)
    spec_tests(S, tech)
    farseer_tables(far)
    neutrality_tables(neu, A)
    neutrality_wald(neu)
    robustness_table(S)
    pd.concat([Rr], ignore_index=True).to_csv(os.path.join(TAB, "m2_datadecide_recipes.csv"), index=False)
    registry(S, CEr, Rr, Q)
    Q.to_csv(os.path.join(TAB, "m2_generalized_q.csv"), index=False)
    # index of bootstrap draw files
    idx = pd.concat([S, CEr, Rr, Q], ignore_index=True, sort=False)[["dataset", "subset", "estimator", "boot_file", "B_boot"]]
    idx["columns"] = np.select([idx.subset.eq("CE_pooled"), idx.estimator.eq("huber_q")],
                               ["theta_CE = (alpha, beta, lnA_r..., lnB_r..., lnE_r...)", "E, A, B, alpha, beta, q (inner)"],
                               "E, A, B, alpha, beta")
    idx.to_csv(os.path.join(BOOTDIR, "index.csv"), index=False)
    fig_forest(S, CEr, Rr, Q, far)
    fig_local_sigma(far)
    fig_neutrality(neu, Rr)
    fig_designs(A)
    print("m2_out: tables and figures written")
