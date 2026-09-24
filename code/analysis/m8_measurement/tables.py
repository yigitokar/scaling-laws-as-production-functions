"""tables.py -- LaTeX (booktabs + threeparttable) and CSV tables for module m8_measurement."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from common import TAB, fmt, se_cell, write_tex

IO_CONCEPT = {1: "Baseline", 2: "Measurement", 3: "Flexible: warmup", 4: "Flexible: schedule", 5: "Flexible: tuned"}
STEP_SHORT = {1: "1. Kaplan reproduction", 2: "2. + head FLOPs in $N$, $C$",
              3: "3. + warmup $\\propto N$", 4: "4. + cosine decay", 5: "5. + tuned LR, $B$, $\\beta_2$"}


def _ci(lo, hi, d=2):
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return ""
    return f"[{lo:.{d}f}, {hi:.{d}f}]"


def porian_table(A):
    st = A["steps"]
    body = []
    for sid in (1, 2, 3, 4, 5):
        r = st[(st.dataset == "rw") & (st.step == sid)].iloc[0]
        o = st[(st.dataset == "owt2") & (st.step == sid)].iloc[0]
        prev_r = st[(st.dataset == "rw") & (st.step == sid - 1)].a.iloc[0] if sid > 1 else np.nan
        prev_o = st[(st.dataset == "owt2") & (st.step == sid - 1)].a.iloc[0] if sid > 1 else np.nan
        body.append(f"{STEP_SHORT[sid]} & {IO_CONCEPT[sid]} & {fmt(r.a)} & {fmt(r.a_porian)} & "
                    f"{fmt(r.a - prev_r) if sid > 1 else ''} & {fmt(o.a)} & {fmt(o.a_porian)} & "
                    f"{fmt(o.a - prev_o) if sid > 1 else ''} & {r.N_at_chinchilla / 1e9:,.0f}")
        body.append(f" & & {_ci(r.a_lo, r.a_hi)} & {_ci(r.a_porian_lo, r.a_porian_hi)} & & {_ci(o.a_lo, o.a_hi)} & "
                    f"{_ci(o.a_porian_lo, o.a_porian_hi)} & & ")
    ka = st[st.step == "kaplan_adj"].iloc[0]
    body.append(f"Kaplan-adjusted & Mismeasured & {fmt(ka.a)} & {fmt(ka.a_porian)} & & & & & "
                f"{ka.N_at_chinchilla / 1e9:,.0f}")
    body.append(f" & & {_ci(ka.a_lo, ka.a_hi)} & & & & & & ")
    # Panel B: counting conventions holding runs fixed (train loss)
    me = A["meas"]
    body.append("\\addlinespace")
    body.append("\\multicolumn{9}{l}{\\textit{Panel B. Same runs, only the counting convention for $N$ and $C=6ND$ changes (train loss)}} ")
    body.append(" & & \\multicolumn{3}{c}{RefinedWeb} & \\multicolumn{3}{c}{OpenWebText2} & ")
    body.append(" & & Kaplan setup & Tuned & $\\Delta$ & Kaplan setup & Tuned & $\\Delta$ & ")
    cnt_lab = {"kaplan": "Kaplan: non-embedding", "standard": "Standard: + head",
               "attention": "+ head + attention", "total": "Total: + input embedding"}
    for cnt in ("kaplan", "standard", "attention", "total"):
        cells = []
        for ds in ("rw", "owt2"):
            k = me.query("dataset==@ds and count==@cnt and run_set.str.startswith('Kaplan')", engine="python").iloc[0]
            t = me.query("dataset==@ds and count==@cnt and run_set.str.startswith('Tuned')", engine="python").iloc[0]
            cells.append((k, t))
        body.append(f"{cnt_lab[cnt]} & & {fmt(cells[0][0].a)} & {fmt(cells[0][1].a)} & {fmt(cells[0][0].a - cells[0][1].a)} & "
                    f"{fmt(cells[1][0].a)} & {fmt(cells[1][1].a)} & {fmt(cells[1][0].a - cells[1][1].a)} & ")
        body.append(f" & & {se_cell(cells[0][0].a_se)} & {se_cell(cells[0][1].a_se)} & & {se_cell(cells[1][0].a_se)} & "
                    f"{se_cell(cells[1][1].a_se)} & & ")
    header = ["\\multicolumn{9}{l}{\\textit{Panel A. Porian et al. (2024) step-by-step path, re-estimated from the 975 released runs}}",
              "\\addlinespace",
              " & & \\multicolumn{3}{c}{RefinedWeb} & \\multicolumn{3}{c}{OpenWebText2} & $N^*(5.88\\mathrm{e}23)$",
              "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
              "Step & IO concept & $\\hat a$ & Porian & $\\Delta \\hat a$ & $\\hat a$ & Porian & $\\Delta \\hat a$ & (billions, RW)"]
    notes = ("$a$ is the allocation exponent in $N^*\\propto C^{a}$ estimated by the IsoFLOP method of Porian et al. (2024), "
             "re-implemented from their released runs (\\texttt{experiment\\_results.pickle.xz}, 975 runs, 16 architectures of "
             "5M--901M parameters, FLOP grid $1.25\\mathrm{e}16\\cdot 2^i$, $i=0,\\dots,11$): loss at each budget by log-log "
             "interpolation along each run, Akima interpolation across model sizes, IsoFLOP argmin, and weighted log-OLS "
             "of $N^*$ on $C$ with weights from a 1,000-draw noise-and-interpolate bootstrap (noise s.d.\\ 0.002 RW, 0.01 OWT2, "
             "as in their code). Brackets: 95\\% bootstrap intervals; `Porian' = their Table 1. Panel A steps 2--5 use "
             "validation loss and the standard count (non-embedding parameters plus the head); step 1 uses the smoothed "
             "train loss and Kaplan's count. Panel B holds the runs fixed (smoothed train loss) and changes only how $N$ "
             "and $C$ are counted; `Kaplan setup' = untuned runs with Kaplan's 1.57B-token warmup, `Tuned' = per-size "
             "tuned LR, batch and $\\beta_2$ with constant LR; $\\Delta$ = Kaplan setup minus tuned. Counts: Kaplan = non-embedding "
             "parameters, no head FLOPs; standard = + unembedding (head); attention = + attention FLOPs; total = precise body + "
             "input and positional embeddings + head (Pearce--Song). Step 2 is `input measurement' in the sense of Nerlove (1963) "
             "and Collard-Wexler and De Loecker (2016); `Kaplan-adjusted' uses tuned hyperparameters with Kaplan's count and "
             "1.57B-token warmup (Porian et al., App.~H), i.e.\\ it keeps both the mismeasured inputs and the long warmup. "
             "Standard errors (bootstrap) in parentheses.")
    write_tex(os.path.join(TAB, "m8_measurement_porian.tex"), body, "llccccccc", header,
              "Measurement and Flexible Inputs in the Kaplan--Chinchilla Allocation Exponent", "tab:m8_porian", notes,
              size="\\scriptsize", tabcolsep="2.5pt")


def biasformula_table(A, B, S_):
    ps = B["ps_tab"]
    rf = B["refit"]
    mf = A["meas_formula"]
    fx = A["ineff"]
    body = ["\\multicolumn{5}{l}{\\textit{Panel A. Measurement: $N$ counted without a component whose share falls with scale}} ",
            "\\addlinespace",
            " & $a$, full count & $a$, mismeasured & Bias & Reference ",
            "\\cmidrule(lr){2-5}"]
    ps_ref = {"Besiroglu et al. (2024)": "0.78 (P\\&S)", "Hoffmann et al. (2022), TeX precision": "0.74 (P\\&S)",
              "Hoffmann et al. (2022), rounded": "0.74 (P\\&S)"}
    for _, r in ps.iterrows():
        body.append(f"Embeddings omitted, Kaplan range; {r.param_set} & {fmt(r.a_true)} & {fmt(r.a_kaplan_range)} & "
                    f"{fmt(r.a_kaplan_range - r.a_true)} & {ps_ref[r.param_set]}")
    for _, r in mf.iterrows():
        ds = "RW" if r.dataset == "rw" else "OWT2"
        rs = "Kaplan-setup runs" if r.run_set.startswith("Kaplan") else "tuned runs"
        body.append(f"Head omitted (Porian ladder), {ds} {rs}: observed & {fmt(r.a_std_obs)} & {fmt(r.a_kaplan_obs)} & "
                    f"{fmt(r.delta_obs)} & ")
        body.append(f"\\quad semi-synthetic (fitted technology, same design) & {fmt(r.a_std_synth)} & {fmt(r.a_kaplan_synth)} & "
                    f"{fmt(r.delta_synth)} & ")
    tot = rf[(rf["sample"] == "all (n=240)")]
    aT = tot[tot.measure.str.startswith("Total")].iloc[0]
    aE = tot[tot.measure.str.startswith("Non")].iloc[0]
    body.append(f"Chinchilla extraction, total vs non-embedding $N$ (parametric fit) & {fmt(aT.a)} & {fmt(aE.a)} & "
                f"{fmt(aE.a - aT.a)} & ")
    body.append(f" & {se_cell(aT.a_se)} & {se_cell(aE.a_se)} & {se_cell(S_['chinchilla_nonembed']['diff_se'])} & ")
    body += ["\\addlinespace",
             "\\multicolumn{5}{l}{\\textit{Panel B. Flexible inputs: first-order argmin-shift formula, $\\Delta a=-\\partial_{\\ln C}[\\Delta(C)/f''(C)]$}} ",
             "\\addlinespace",
             " & Observed $\\Delta a$ & \\multicolumn{2}{c}{Predicted $\\Delta a$} & Share explained (\\%) ",
             " & & $f''$ from technology & $f''$ local & technology / local ",
             "\\cmidrule(lr){2-5}"]
    for _, r in fx.iterrows():
        ds = "RW" if r.dataset == "rw" else "OWT2"
        body.append(f"{ds}: step {int(r.step)} vs.\\ step 5 (tuned) & {fmt(r.delta_a_obs)} & {fmt(r.delta_a_pred_model)} & "
                    f"{fmt(r.delta_a_pred_local)} & {r.delta_a_pred_model / r.delta_a_obs * 100:.0f} / {r.delta_a_pred_local / r.delta_a_obs * 100:.0f}")
    header = []
    notes = ("Panel A, rows 1--3: researcher measures $N_m=N-X$ with $X=\\omega N_m^{1/3}$ ($\\omega=47{,}491$, Pearce and "
             "Song 2024) and $C_m=6N_mD$ and picks the IsoFLOP argmin in measured units; `measured $a$' is the log-log slope "
             "of $N_m^*$ on $C_m$ over $N_m\\in[790, 1.58\\mathrm{e}9]$, computed from the closed form "
             "$a_m=\\beta/(\\beta+\\alpha\\kappa-\\eta)$, $\\kappa=1-s(1-\\theta)$, $\\eta=d\\ln\\kappa/d\\ln N_m$ "
             "(verified symbolically and against brute-force argmins). Rows 4--11: the omitted component is the "
             "unembedding (head) matrix on Porian's 16-architecture ladder ($X=V w$, fitted as $\\omega N_m^{\\theta}$); "
             "`observed' is the IsoFLOP estimate with the two counts (train loss); `semi-synthetic' replaces every loss by a "
             "Chinchilla technology fitted to the same runs (standard count) at exactly the same architectures and budgets and "
             "reruns the pipeline in both counts. Chinchilla extraction: Besiroglu et al.\\ (2024) sample ($n=240$), "
             "Huber-LSE estimator, pairs bootstrap (200 draws) SEs in parentheses (the bias SE is from the paired "
             "bootstrap of the difference); non-embedding $N$ obtained by inverting the Pearce--Song map. Semi-synthetic rows "
             "refitted with $E$ held at 1.8 give biases of " + ", ".join(f"{v:.2f}" for v in mf.delta_synth_E18) +
             " (same order as the rows); the continuous closed-form path over the same measured-compute range (without the "
             "discrete design) predicts " + ", ".join(f"{v:.2f}" for v in mf.delta_pred) + ". Panel B: $\\Delta(C)=\\partial\\delta/\\partial\\ln N|_C$ is the slope, along the IsoFLOP at the "
             "tuned optimum, of the excess loss $\\delta=L_{\\text{step }s}-L_{\\text{tuned}}$ on matched $(N,C)$ cells; "
             "$f''$ is the curvature of the tuned IsoFLOP curve at its optimum, either from a Chinchilla fit to the tuned runs "
             "($f''=\\alpha^2u+\\beta^2v$) or from a kernel-weighted local quadratic (bandwidth 1 in $\\ln N$; bandwidth 0.5 "
             "gives predictions between the two columns). Predicted $\\Delta a$ is the slope across "
             "budgets of the predicted log-argmin shift $-\\Delta/f''$; observed $\\Delta a$ is the difference of IsoFLOP "
             "exponents from Table~\\ref{tab:m8_porian}.")
    write_tex(os.path.join(TAB, "m8_measurement_biasformulas.tex"), body, "p{6.0cm}cccc", header,
              "Bias Formulas for the Allocation Exponent: Measurement and Flexible Inputs", "tab:m8_biasformulas", notes,
              size="\\scriptsize")


def steplaw_table(C, S_):
    pol = C["pol"]
    order = ["frontier", "steplaw", "random_k16", "random_k4", "random_k1", "best_fixed", "porian_base", "porian_rule",
             "deepseek", "bjorck"]
    lab = {"frontier": "Frontier: min over grid", "steplaw": "Step Law rule (in-sample)",
           "random_k16": "Best of 16 random configs", "random_k4": "Best of 4 random configs",
           "random_k1": "One random config", "best_fixed": "One fixed (LR, batch) for all cells",
           "porian_base": "Porian `base' fixed LR 3e-3, 0.5M batch", "porian_rule": "Porian tuned $N$-rule",
           "deepseek": "DeepSeek $C$-rule", "bjorck": "Bjorck $(N,D)$-rule, 0.5M batch"}
    body = []
    rb = S_["steplaw_frontier_resid_boot"]
    for r in order:
        x = pol[pol.rule == r].iloc[0]
        is_mc = r.startswith("random")
        body.append(f"{lab[r]} & {100 * x.u_mean:.2f} & {fmt(100 * x.u_n, 2)} & {fmt(100 * x.u_d, 2)} & "
                    f"{fmt(x['E1.4_alpha'])} & {fmt(x['E1.4_beta'])} & {fmt(x['E1.4_a'])} & {fmt(x['E1.4_sigma_star'])} & "
                    f"{fmt(x['free_E'], 2)} & {fmt(x['free_a'])} & {fmt(x['free_sigma_star'])}")
        if r == "frontier":
            body.append(f" & & & & & & {se_cell(rb['a_14']['sd'])} & {se_cell(rb['sigma_14']['sd'])} & "
                        f"{se_cell(rb['E_free']['sd'], 2)} & {se_cell(rb['a_free']['sd'])} & {se_cell(rb['sigma_free']['sd'])}")
        elif is_mc:
            body.append(f" & & [{100 * x.u_n_se:.2f}] & [{100 * x.u_d_se:.2f}] & & & [{x['E1.4_a_sd']:.3f}] & "
                        f"[{x['E1.4_sigma_star_sd']:.3f}] & & [{x['free_a_sd']:.3f}] & [{x['free_sigma_star_sd']:.3f}]")
        else:
            body.append(f" & & {se_cell(100 * x.u_n_se, 2)} & {se_cell(100 * x.u_d_se, 2)} & & & & & & & ")
    header = [" & \\multicolumn{3}{c}{Inefficiency $\\iota=\\ln L-\\ln L^*$ (\\%)} & \\multicolumn{4}{c}{$E$ held at 1.4} & "
              "\\multicolumn{3}{c}{$E$ estimated}",
              "\\cmidrule(lr){2-4}\\cmidrule(lr){5-8}\\cmidrule(lr){9-11}",
              "Flexible-input policy & mean & $\\partial\\iota/\\partial\\ln N$ & $\\partial\\iota/\\partial\\ln D$ & $\\alpha$ & $\\beta$ & "
              "$a$ & $\\sigma^*$ & $E$ & $a$ & $\\sigma^*$"]
    fr = S_["steplaw_frontier_nls"]
    frh = S_["steplaw_frontier_huber"]
    pf = C["prof"][C["prof"].rule == "frontier"]
    a_lo_E = float(pf.loc[np.isclose(pf.E, 0.6), "a"].iloc[0])
    a_hi_E = float(pf.loc[np.isclose(pf.E, 1.6), "a"].iloc[0])
    notes = ("Step Law dense grid (Li et al.\\ 2025): 1,911 runs in 17 $(N,D)$ cells, $N\\in[215\\mathrm{M},1.07\\mathrm{B}]$ "
             "non-embedding parameters, $D\\in[4,100]$B tokens, up to 12 learning rates $\\times$ 10 batch sizes per cell (the "
             "$N=1.07$B, $D=56.9$B cell has only 5 learning rates and 47 runs); loss = "
             "smoothed final training loss. Each policy selects one run per cell (nearest grid point to the rule's target, "
             "in $\\log_2$ units); `random' rows are medians over 200 Monte Carlo draws that pick $k$ non-diverged "
             "configurations per cell and keep the best ([bracketed] = Monte Carlo s.d.). Inefficiency gradients: OLS of "
             "$\\iota$ on $\\ln N,\\ln D$ across the 17 cells, HC1 SEs in parentheses (in percent). Technology: Chinchilla form "
             "fitted by NLS on log loss (Gaussian QMLE) with $E$ fixed at 1.4 or estimated; with 17 cells the profile "
             f"likelihood in $E$ is flat and $a$ ranges from {a_lo_E:.2f} to {a_hi_E:.2f} as $E$ goes from 0.6 to 1.6 on the frontier "
             "sample (Figure~\\ref{fig:m8_steplaw}), so only differences across rows at fixed $E$ are informative about bias. "
             f"Frontier SEs: residual bootstrap (200 draws; residuals centred and rescaled by $\\sqrt{{n/(n-5)}}$). "
             f"$N$ is Step Law's non-embedding count; counting the head ($N+65{{,}}536\\,h$) moves the $E$-free frontier "
             f"$a$ to {S_['steplaw_count_robustness'][1]['free_a']:.3f} and the $E=1.4$ value to "
             f"{S_['steplaw_count_robustness'][1]['E14_a']:.3f} (Proposition M), so levels of $a$ here are in Kaplan's convention. Huber-LSE (the paper's reference estimator) on the frontier "
             f"gives $E={frh['E']:.2f}$, $a={frh['a_N']:.3f}$, $\\sigma^*={frh['sigma_star']:.3f}$ (NLS: $E={fr['E']:.2f}$, "
             f"$a={fr['a_N']:.3f}$). Rules: Step Law $\\eta=1.79N^{{-0.713}}D^{{0.307}}$, $B=0.58D^{{0.571}}$ tokens; Porian "
             "$\\eta=3.7N^{-0.36}$, $B=0.00037N^{0.703}$ sequences ($N$ incl.\\ head); DeepSeek $\\eta=0.3118C^{-0.125}$, "
             "$B=0.292C^{0.3271}$ tokens; Bjorck $\\eta=1.55\\mathrm{e}{-3}(N/10^9)^{-0.23}(D/10^9)^{-0.32}$ with 0.5M-token batch.")
    write_tex(os.path.join(TAB, "m8_measurement_steplaw.tex"), body, "lcccccccccc", header,
              "Hyperparameters as Flexible Inputs: Inefficiency and the Estimated Technology (Step Law Grid)",
              "tab:m8_steplaw", notes, size="\\scriptsize", tabcolsep="3pt")


def demand_table(C, S_):
    dem = C["dem"]
    dc = S_["lr_decomposition"]
    bt = S_["lr_decomposition_boot"]
    sb = S_["steplaw_published_bootstrap"]

    def row(var, method, label):
        x = dem[(dem["var"] == var) & (dem.method == method)].iloc[0]
        eN = fmt(x.e_N) if np.isfinite(x.e_N) else "--"
        eB = fmt(x.e_B) if "e_B" in x and np.isfinite(x.get("e_B", np.nan)) else ""
        se_N = x.get("e_N_bse", np.nan)
        se_N = se_N if np.isfinite(se_N) else x.e_N_se
        se_D = x.get("e_D_bse", np.nan)
        se_D = se_D if np.isfinite(se_D) else x.e_D_se
        eBse = se_cell(x.get("e_B_se", np.nan)) if eB else ""
        return [f"{label} & {eN} & {fmt(x.e_D)} & {eB} & {int(x.n)}",
                f" & {se_cell(se_N) if np.isfinite(x.e_N) else ''} & {se_cell(se_D)} & {eBse} & "]

    body = ["\\multicolumn{5}{l}{\\textit{Panel A. Learning-rate demand, $\\ln\\eta^*$}}"]
    body += row("lr", "grid argmin", "Unconditional, grid argmin (Step Law's method)")
    body += row("lr", "smoothed argmin", "Unconditional, smoothed (quadratic surface) argmin")
    body += row("lr", "conditional on batch (smoothed, pooled slices)", "Conditional on batch, smoothed")
    body += row("lr", "conditional on batch (grid argmin, pooled slices)", "Conditional on batch, grid argmin")
    body += row("lr", "conditional at batch = 0.52M tokens (smoothed)", "Conditional, batch fixed at 0.52M tokens")
    body += ["\\addlinespace", "\\multicolumn{5}{l}{\\textit{Panel B. Batch-size demand, $\\ln B^*$ (tokens)}}"]
    body += row("bs", "grid argmin", "Grid argmin")
    body += row("bs", "smoothed argmin", "Smoothed argmin")
    body += row("bs", "grid argmin, D only", "Grid argmin, $D$ only (Step Law form)")
    body += row("bs", "smoothed argmin, D only", "Smoothed argmin, $D$ only (Step Law form)")
    body += ["\\addlinespace", "\\multicolumn{5}{l}{\\textit{Panel C. Published rules (literature values)}}",
             f"Step Law, $\\eta$ (their 1,000 bootstrap fits: mean, s.d.) & {fmt(sb['lr_coefN']['mean'])} & "
             f"{fmt(sb['lr_coefD']['mean'])} & & ",
             f" & ({sb['lr_coefN']['sd']:.3f}) & ({sb['lr_coefD']['sd']:.3f}) & & ",
             f"Step Law, $B$ & & {sb['bs_coefD']['mean']:.3f} & & ",
             f" & & ({sb['bs_coefD']['sd']:.3f}) & & ",
             "Step Law, published point estimates & $-0.713$ / -- & $0.307$ / $0.571$ & & ",
             "Bjorck et al., $\\eta$ at fixed 0.5M-token batch ($N\\geq$760M) & $-0.23$ & $-0.32$ & & ",
             "Porian et al., $\\eta$ / $B$ (fit at $D\\approx 20N$) & $-0.36$ / $0.703$ & -- & & ",
             "\\addlinespace", "\\multicolumn{5}{l}{\\textit{Panel D. Le Chatelier decomposition of $d\\ln\\eta^*/d\\ln D$}}",
             f"Conditional + complementarity $\\times$ batch elasticity & \\multicolumn{{4}}{{l}}{{"
             f"${dc['e_D_conditional']:.3f} + {dc['e_B']:.3f}\\times{dc['f_D']:.3f} = {dc['implied_unconditional']:.3f}$ "
             f"vs.\\ unconditional ${dc['e_D_unconditional']:.3f}$}}",
             f"Batch co-scaling share of unconditional $D$-elast., smoothed & \\multicolumn{{4}}{{l}}{{"
             f"{bt['smoothed']['share_from_batch']['point']:.2f} ({bt['smoothed']['share_from_batch']['se']:.2f}); "
             f"90\\% CI [{bt['smoothed']['share_from_batch']['q05']:.2f}, {bt['smoothed']['share_from_batch']['q95']:.2f}]}}",
             f"Same, grid argmins & \\multicolumn{{4}}{{l}}{{"
             f"{bt['grid']['share_from_batch']['point']:.2f} ({bt['grid']['share_from_batch']['se']:.2f}); "
             f"90\\% CI [{bt['grid']['share_from_batch']['q05']:.2f}, {bt['grid']['share_from_batch']['q95']:.2f}]}}",
             f"Share of published gap ($0.307$ vs.\\ $-0.32$) closed & \\multicolumn{{4}}{{l}}{{"
             f"{bt['smoothed']['share_of_published_gap_closed']['point']:.2f} ({bt['smoothed']['share_of_published_gap_closed']['se']:.2f}) smoothed; "
             f"{bt['grid']['share_of_published_gap_closed']['point']:.2f} ({bt['grid']['share_of_published_gap_closed']['se']:.2f}) grid}}"]
    header = ["Specification & $\\ln N$ & $\\ln D$ & $\\ln B$ & Obs."]
    notes = ("Optimal learning rate $\\eta^*$ and batch size $B^*$ in each of the 17 Step Law $(N,D)$ cells, regressed on "
             "$\\ln N$ (non-embedding) and $\\ln D$ (tokens). `Grid argmin' uses the best run in the cell; `smoothed' uses "
             "the minimum of a quadratic in $(\\ln\\eta,\\ln B)$ fitted to runs within 2\\% of the cell minimum. SEs in "
             "parentheses: two-level bootstrap (resample cells; draw each cell's optimum from its within-cell sampling "
             "distribution; 1,000 draws) for unconditional demands, cell-clustered SEs for pooled conditional regressions "
             "(slices = cell $\\times$ batch with an interior LR optimum; quadratic in $\\ln\\eta$ within 3\\% of the slice "
             "minimum). Conditional demands hold the batch fixed (as in Bjorck et al.\\ 2025, who fix 0.5M tokens); "
             "unconditional demands let the batch adjust (Step Law). Panel D: the unconditional $D$-elasticity equals the "
             "conditional one plus the LR--batch complementarity times the $D$-elasticity of batch demand (chain rule); shares "
             "with cell-cluster bootstrap SEs in parentheses (1,000 draws resampling the 17 cells with all their batch slices).")
    write_tex(os.path.join(TAB, "m8_measurement_demand.tex"), body, "lcccc", header,
              "Flexible-Input Demand Functions: Learning Rate and Batch Size", "tab:m8_demand", notes, size="\\scriptsize")


def sfa_table(C):
    sfa = C["sfa"]
    body = []
    for _, r in sfa.iterrows():
        if r.dist.startswith("cell-level"):
            body.append("\\addlinespace")
            body.append(f"Cell-level OLS: $\\ln\\bar\\iota_c$ on $\\ln N,\\ln D$ & -- & {fmt(r.g0, 2)} & {fmt(r.g_n)} & "
                        f"{fmt(r.g_d)} & & & ")
            body.append(f" & & {se_cell(r.g0_se, 2)} & {se_cell(r.g_n_se)} & {se_cell(r.g_d_se)} & & & ")
            continue
        sv = f"{r.sv:.1e}".replace("e-0", "e$-$").replace("e-", "e$-$")
        body.append(f"{'Half-normal' if r.dist == 'halfnormal' else 'Exponential'} & {r.sigma_v} & {fmt(r.g0, 2)} & "
                    f"{fmt(r.g_n)} & {fmt(r.g_d)} & {sv} & {100 * r.mean_abs_dev_frontier:.2f} & {fmt(r.nll, 1)}")
        body.append(f" & & {se_cell(r.g0_se, 2)} & {se_cell(r.g_n_se)} & {se_cell(r.g_d_se)} & & & ")
        if np.isfinite(r.get("g_d_cse", np.nan)):
            body.append(f" & & [{r.g0_cse:.2f}] & [{r.g_n_cse:.3f}] & [{r.g_d_cse:.3f}] & & & ")
    header = ["$\\iota$ distribution & $\\sigma_v$ & $\\gamma_0$ & $\\gamma_N$ & $\\gamma_D$ & $\\hat\\sigma_v$ & "
              "$|\\hat\\mu_c-\\ln L^*_c|$ (\\%) & $-\\log\\mathcal{L}$"]
    nsfa = int(sfa[~sfa.dist.str.startswith("cell-level")].n.iloc[0])
    notes = ("Cost-type stochastic frontier $\\ln L_{ij}=\\mu_c+\\epsilon_{ij}+\\iota_{ij}$ with 17 cell fixed effects $\\mu_c$ "
             "(the concentrated technology, left unrestricted), $\\epsilon\\sim N(0,\\sigma_v^2)$, and inefficiency $\\iota\\geq0$ half-normal "
             "(scale $\\sigma_u$) or exponential (mean $\\sigma_u$) with $\\ln\\sigma_{u,ij}=\\gamma_0+\\gamma_N(\\ln N-\\overline{\\ln N})"
             "+\\gamma_D(\\ln D-\\overline{\\ln D})$ (heteroskedastic inefficiency, Caudill, Ford and Gropper 1995). "
             f"Sample: {nsfa:,} non-diverged runs (runs with loss more than 35\\% above the cell minimum, 9.5\\% "
             "of all runs, are treated as failed and excluded). `Calibrated' fixes $\\sigma_v$ at the median residual s.d.\\ "
             "of the within-cell quadratic surfaces. MLE; SEs from the numerical Hessian in parentheses; cluster-robust "
             "(by cell, 17 clusters) sandwich SEs in brackets. Column 7: mean absolute gap between the SFA frontier and the grid "
             "minimum. Last row: OLS across the 17 cells of the log of the mean inefficiency of non-diverged runs on centred "
             "$\\ln N$ and $\\ln D$ (HC1 SEs); for exponential inefficiency $E[\\iota]=\\sigma_u$, so its slopes are "
             "comparable with $\\gamma_N,\\gamma_D$.")
    write_tex(os.path.join(TAB, "m8_measurement_sfa.tex"), body, "lccccccc", header,
              "Stochastic Frontier with Scale-Dependent Inefficiency (Step Law Grid)", "tab:m8_sfa", notes,
              size="\\scriptsize")


def make_all(A, B, C, S_):
    porian_table(A)
    biasformula_table(A, B, S_)
    steplaw_table(C, S_)
    demand_table(C, S_)
    sfa_table(C)
