"""decisions.py -- decision units and the revealed value of compactness over decisions (R1 New 1; R2 Major 2; R3 N3).

Decision units (primary definition, as requested): one unit per common-D family (max/min D <= 1.10; its family-level
object W_f = [sum_j omega_j / w_j]^(-1), omega = training-compute shares; Prop. A9: 1 - 1/W_f = sum_j omega_j s_j is
the compute-weighted mean of member shares) plus one unit per member of a size-specific family and per singleton.
What each unit reveals depends on the family reading (evidence.py): 'choice' -> the family share (a pi-weighted
average; point); 'cap' -> a lower bound (and, under the cap, members are separate decisions whose own wedges are lower
bounds); 'menu' -> an upper bound; 'cap+menu' -> nothing.
Variants: (R) reading-consistent units -- cap families enter as their members (lower bounds), choice and menu
families as one unit, cap+menu families are dropped; (G) subgroup units -- size-specific families are split into
sub-groups sharing one D (within 10 percent; e.g. Qwen2 1.5B/7B/72B at 7T, LLaMA 7B/13B at 1T and 30B/65B at 1.4T).
Statistics: median s, share with w > 1; joint bootstrap over technology draws (the unit statistic is recomputed in
each draw, family objects included).
Round 3 (fix list W1-W2; decisions D-1, D-2): the primary unit is the budget level (units_budget: members of one family
that share one token budget, max/min D <= 1.10, form one decision), with readings coded under
paper/notes/rb5_reading_protocol.md (module rb5_units) and an identification class for the virtual value of compactness
(point; lower bound; lower bound, point if the budget was chosen). The schemes above remain as robustness rows: 'primary'
(the 56 family-label units of version 3), 'reading' (version 3's 66 reading-consistent units).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import rb2common as C
import wedges as WG  # ra2


def family_W(w, C_):
    om = C_ / C_.sum()
    return 1.0 / np.sum(om / w)


def subgroup_labels(Bc, tol=1.10):
    """Within each family (gen), cluster members by D (sorted; a new cluster starts when D exceeds 1.10 x the cluster
    minimum). Returns a Series uid -> subgroup label."""
    out = {}
    for gen, g in Bc.groupby("gen"):
        g = g.sort_values("D")
        k, dmin = 0, None
        for _, r in g.iterrows():
            if dmin is None or r["D"] > tol * dmin:
                k += 1
                dmin = r["D"]
            out[r["uid"]] = f"{gen}#{k}"
    return pd.Series(out)


def units(Bc, readings, scheme="primary"):
    """Unit membership: DataFrame uid -> unit, unit_type ('family' or 'model'), reading, bound."""
    rd = readings.set_index("family")["reading"].to_dict()
    rows = []
    sub = subgroup_labels(Bc)
    for _, r in Bc.iterrows():
        gen, cls = r["gen"], r["fam_class"]
        reading = rd.get(gen, "") if cls == "common-D" else ""
        if scheme == "primary":
            fam = cls == "common-D"
            unit = gen if fam else r["uid"]
        elif scheme == "reading":
            if cls == "common-D" and reading == "cap+menu":
                continue
            fam = cls == "common-D" and reading in ("choice", "menu")
            unit = gen if fam else r["uid"]
        elif scheme == "subgroup":
            if cls == "common-D":
                fam, unit = True, gen
            else:
                sg = sub[r["uid"]]
                n_sg = (sub == sg).sum()
                fam = n_sg >= 2
                unit = sg if fam else r["uid"]
        else:
            raise ValueError(scheme)
        bound = ("point (family share)" if reading == "choice" else "lower bound" if reading == "cap" else
                 "upper bound" if reading == "menu" else "none" if reading == "cap+menu" else
                 "point (family share, sub-group)" if fam else "point (member)")
        if scheme == "reading" and cls == "common-D" and reading == "cap":
            bound = "lower bound (member under cap)"
        rows.append(dict(uid=r["uid"], unit=unit, unit_type="family" if fam else "model", reading=reading, bound=bound))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- round 3: budget-level units are primary (decision D-2)
def _rb5():
    import os
    import sys
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rb5_units")
    if d not in sys.path:
        sys.path.append(d)
    import readings as RD  # rb5_units
    return RD


def budget_labels(Bc, tol=1.10):
    """uid -> unit label of the budget-level scheme (identical to rb5_units/frame.py): the family name when all members
    share one budget, '<family>#k' for the k-th budget group of a family with several budgets, the uid for a model that
    shares its budget with no family member."""
    sub = subgroup_labels(Bc, tol)
    X = Bc[["uid", "gen"]].copy()
    X["group"] = X["uid"].map(sub)
    size = X.groupby("group")["uid"].transform("count")
    ngrp = X.groupby("gen")["group"].transform("nunique")
    lab = np.where(size >= 2, np.where(ngrp == 1, X["gen"], X["group"]), X["uid"])
    return pd.Series(lab, index=X["uid"].values)


def units_budget(Bc, codes=None, scheme="budget"):
    """Budget-level decision units with the rb5 readings and identification classes (protocol Section 8).
    scheme 'budget' (primary, 49 units); 'split' (members of cap-type budgets enter separately as lower bounds, premise:
    each would have trained longer at its own size; 66 units); 'olmo1b' (primary with OLMo 2 1B as its own unit)."""
    if codes is None:
        codes = _rb5().load()
    cd = codes.set_index("unit")
    lab = budget_labels(Bc)
    rows = []
    for _, r in Bc.iterrows():
        u = lab[r["uid"]]
        c = cd.loc[u]
        fam = c["unit_kind"] == "common-budget"
        unit, cls, bound = u, c["id_class"], None
        if scheme == "split" and fam and cls != "point":
            unit, fam, bound = r["uid"], False, "lower bound (member under cap)"
        if scheme == "olmo1b" and r["model"] == "OLMo-2-0425-1B":
            unit, fam = r["uid"], False
        if bound is None:
            bound = {"point": "point (family share)" if fam else "point (member)",
                     "lower bound": "lower bound", "lower bound (point if choice)": "lower bound (point if choice)"}[cls]
        rows.append(dict(uid=r["uid"], unit=unit, unit_type="family" if fam else "model", unit_kind=c["unit_kind"],
                         budget_unit=u, reading=c["reading"], id_class=cls if bound != "lower bound (member under cap)"
                         else "lower bound", bound=bound, point_if_choice=bool(c["point_if_choice"]),  # [review]
                         own_cap=int(c["own_cap"]), syn_protocol=int(c["synthetic"]),
                         syn_fixlist=int(c["synthetic_fixlist"]), R_CO=int(c["R_CO"]), R_INF=int(c["R_INF"]),
                         R_DEV=int(c["R_DEV"]), deployment_rationale=int(c["deployment_rationale"]),
                         codes_file=c["codes_file"]))
    return pd.DataFrame(rows)


def unit_values(Bc, U, wcol):
    """Unit-level W (family W_f or member w) for the wedge column `wcol` of Bc."""
    X = Bc.merge(U, on="uid")
    out = []
    for u, g in X.groupby("unit", sort=False):
        W = family_W(g[wcol].values, g["Cmp"].values) if g["unit_type"].iloc[0] == "family" else float(g[wcol].iloc[0])
        out.append(dict(unit=u, W=W))
    return pd.DataFrame(out)


def unit_table(Bc, U, serve, labw=None):
    """Descriptive unit table (reference and lab-own W, serving, tier, year, compute)."""
    X = Bc.merge(U, on="uid")
    if "serve_model" not in X:
        X = X.merge(serve[["uid", "serve_model", "serve_model_hi"]], on="uid")
    X["serve_developer"] = X["serve"].astype(int)
    X["tier"] = C.in_tier(X["N"].values).astype(int)
    rows = []
    for u, g in X.groupby("unit", sort=False):
        fam = g["unit_type"].iloc[0] == "family"
        wref = family_W(g["w_ref"].values, g["Cmp"].values) if fam else g["w_ref"].iloc[0]
        wlab = family_W(g["w_primary1"].values, g["Cmp"].values) if fam else g["w_primary1"].iloc[0]
        wlab0 = family_W(g["w_primary"].values, g["Cmp"].values) if fam else g["w_primary"].iloc[0]
        flag = g.loc[g["Cmp"].idxmax()]
        rows.append(dict(
            unit=u, unit_type=g["unit_type"].iloc[0], family=g["gen"].iloc[0], dev=g["dev"].iloc[0], n_members=len(g),
            members="; ".join(g.sort_values("N")["model"]), reading=g["reading"].iloc[0], bound=g["bound"].iloc[0],
            fam_class=g["fam_class"].iloc[0], year=int(g["year"].min()), date=pd.to_datetime(g["date"]).min(),
            C_total=g["Cmp"].sum(), N_flagship=flag["N"], D_flagship=flag["D"],
            M_cw=float(np.exp(np.sum(g["Cmp"] * np.log(g["M"])) / g["Cmp"].sum())),
            w_ref=wref, s_ref=C.share(wref), w_lab1=wlab, s_lab1=C.share(wlab), w_ra2primary=wlab0,
            s_ra2primary=C.share(wlab0), serve_any=int(g["serve_model"].max()), serve_all=int(g["serve_model"].min()),
            serve_flagship=int(flag["serve_model"]), serve_hi_any=int(g["serve_model_hi"].max()),
            serve_developer=int(g["serve_developer"].max()), tier_share=g["tier"].mean(), tier_any=int(g["tier"].max()),
            ondevice=int((g["deploy"] == "ondevice").any()), local=int(g["deploy"].isin(["ondevice", "local"]).any()),
            has_lab=int(g["w_lab1"].notna().any()), lab_path=str(g["lab_path"].dropna().iloc[0]) if g["lab_path"].notna().any() else "",
            **{k: g[k].iloc[0] for k in ("unit_kind", "id_class", "own_cap", "syn_protocol", "syn_fixlist", "R_CO", "R_INF",
                                          "R_DEV", "deployment_rationale", "codes_file") if k in g},
            emb_share_min=float(g["emb_share"].min()), emb_share_max=float(g["emb_share"].max()),
            emb_share_cw=float(np.sum(g["Cmp"] * g["emb_share"]) / g["Cmp"].sum()),
            N_emb_share_flagship=float(flag["emb_share"])))
    return pd.DataFrame(rows)


def lnw_draws_tech(Bc, t, Bdraw):
    """(Bdraw, n) ln w draws of technology t for the models of Bc (draw index cycles if t has fewer draws)."""
    N = WG.n_used(Bc, t.n_conv)
    lnN, lnD = np.log(N)[None, :], np.log(Bc["D"].values)[None, :]
    dr = t.draws
    idx = np.arange(Bdraw) % len(dr)
    return C.log_w(lnN, lnD, dr[idx, 0:1], dr[idx, 1:2], dr[idx, 2:3])


def joint_stats(Bc, U, LW):
    """LW: (B, n) ln w draws aligned with Bc rows. Returns per-draw unit median s, share W > 1, and unit W matrix."""
    X = Bc[["uid", "Cmp"]].reset_index(drop=True).merge(U, on="uid", how="left")
    keep = X["unit"].notna().values
    X = X[keep].reset_index(drop=True)
    W = np.exp(LW[:, keep])
    cols = []
    for u, g in X.groupby("unit", sort=False):
        idx = g.index.values
        if g["unit_type"].iloc[0] == "family":
            om = g["Cmp"].values / g["Cmp"].values.sum()
            cols.append(1.0 / np.sum(om[None, :] / W[:, idx], axis=1))
        else:
            cols.append(W[:, idx[0]])
    UW = np.column_stack(cols)
    S = (UW - 1) / UW
    return dict(med_s=np.median(S, axis=1), share_gt1=(UW > 1).mean(axis=1), med_w=np.median(UW, axis=1), UW=UW)


def summarize(name, point_W, js, n_units, extra=None):
    s = C.share(point_W)
    row = dict(stat=name, n_units=n_units, median_s=float(np.median(s)), median_w=float(np.median(point_W)),
               share_w_gt1=float(np.mean(point_W > 1)))
    if js is not None:
        row.update(median_s_lo=C.pct(js["med_s"], 2.5), median_s_hi=C.pct(js["med_s"], 97.5),
                   share_w_gt1_lo=C.pct(js["share_gt1"], 2.5), share_w_gt1_hi=C.pct(js["share_gt1"], 97.5),
                   median_w_lo=C.pct(js["med_w"], 2.5), median_w_hi=C.pct(js["med_w"], 97.5), B=len(js["med_s"]))
    if extra:
        row.update(extra)
    return row
