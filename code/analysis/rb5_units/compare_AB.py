"""compare_AB.py: [review] copied from coder B's session scratchpad on 2026-09-25 (only the two path lines changed); it
reproduces data/processed/rb5_units/readings_disagreements.csv.

compare_AB.py: agreement between coder A (readings_coderA.csv) and blind coder B (readings_coderB.csv) under
protocol Section 11: raw agreement and Cohen's kappa per coded field, and every disagreement with both codes, both
coders' quotes for that field and both notes. Writes data/processed/rb5_units/readings_disagreements.csv.
Does not resolve disagreements."""
import os

import numpy as np
import pandas as pd

PROC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "processed", "rb5_units")  # [review] was absolute
A = pd.read_csv(os.path.join(PROC, "readings_coderA.csv")).set_index("unit")
B = pd.read_csv(os.path.join(PROC, "readings_coderB.csv")).set_index("unit")
assert set(A.index) == set(B.index)
B = B.loc[A.index]
for X in (A, B):
    X["deployment"] = ((X["R_INF"] == 1) | (X["R_DEV"] == 1)).astype(int)

COMMON = A.index[A["unit_kind"] == "common-budget"]
OTHER = A.index[A["unit_kind"] != "common-budget"]
# (field, scope label, index)
FIELDS = [("reading", "common-budget units", COMMON), ("D_margin", "common-budget units", COMMON),
          ("N_margin", "common-budget units", COMMON), ("own_cap", "all units", A.index),
          ("own_cap", "size-specific and singleton units", OTHER), ("R_CO", "all units", A.index),
          ("R_INF", "all units", A.index), ("R_DEV", "all units", A.index), ("R_NONE", "all units", A.index),
          ("deployment", "all units", A.index), ("synthetic", "all units", A.index)]
ROLES_A = {"reading": ["D", "N", "NONEV"], "D_margin": ["D", "NONEV"], "N_margin": ["N"], "own_cap": ["CAP"],
           "R_CO": ["CO"], "R_INF": ["INF"], "R_DEV": ["DEV"], "R_NONE": ["CO", "INF", "DEV"],
           "deployment": ["INF", "DEV"], "synthetic": ["SYN"]}
ROLES_B = {"reading": ["D", "N"], "D_margin": ["D"], "N_margin": ["N"], "own_cap": ["own_cap"], "R_CO": ["R_CO"],
           "R_INF": ["R_INF"], "R_DEV": ["R_DEV"], "R_NONE": ["R_CO", "R_INF", "R_DEV"],
           "deployment": ["R_INF", "R_DEV"], "synthetic": ["synthetic"]}
RULE = {"reading": ("D_rule", "D_rule"), "D_margin": ("D_rule", "D_rule"), "N_margin": ("N_rule", "N_rule"),
        "own_cap": ("own_cap_rule", "own_cap_members")}


def kappa(a, b):
    a, b = np.asarray(a).astype(str), np.asarray(b).astype(str)
    cats = sorted(set(a) | set(b))
    po = float(np.mean(a == b))
    pe = float(sum(np.mean(a == c) * np.mean(b == c) for c in cats))
    return po, ((po - pe) / (1 - pe) if pe < 1 else np.nan), pe


def quotes(X, u, roles, nmax):
    out = []
    for i in range(1, nmax + 1):
        r, q, s = X.loc[u].get(f"quote_role_{i}"), X.loc[u].get(f"quote_{i}"), X.loc[u].get(f"source_{i}")
        if isinstance(q, str) and q.strip() and isinstance(r, str) and any(x in roles for x in r.replace("+", " ").split()):
            out.append(f"[{s}; {r}] {q}")
    return " || ".join(out) if out else "(no quote for this field)"


nA = max(int(c.split("_")[1]) for c in A.columns if c.startswith("quote_") and not c.startswith("quote_role"))
nB = max(int(c.split("_")[1]) for c in B.columns if c.startswith("quote_") and not c.startswith("quote_role"))
agr, dis = [], []
for f, scope, idx in FIELDS:
    a, b = A.loc[idx, f].fillna("").astype(str), B.loc[idx, f].fillna("").astype(str)
    po, k, pe = kappa(a, b)
    nd = int((a != b).sum())
    agr.append(dict(field=f, scope=scope, n=len(idx), n_agree=len(idx) - nd, agreement_pct=100 * po, kappa=k,
                    expected_agreement=pe))
    if scope == "size-specific and singleton units":
        continue
    for u in idx[(a != b).values]:
        ra, rb = RULE.get(f, (None, None))
        dis.append(dict(unit=u, unit_kind=A.loc[u, "unit_kind"], field=f, coder_A=A.loc[u, f], coder_B=B.loc[u, f],
                        coder_A_rule=(A.loc[u, ra] if ra else ""), coder_B_rule=(B.loc[u, rb] if rb else ""),
                        coder_A_quotes=quotes(A, u, ROLES_A[f], nA), coder_B_quotes=quotes(B, u, ROLES_B[f], nB),
                        coder_A_note=A.loc[u, "note"], coder_B_note=B.loc[u, "note"], resolution="not resolved (coder B does not resolve)"))
AG, DZ = pd.DataFrame(agr), pd.DataFrame(dis)
# descriptive: exact match of the D rule labels
ra = A.loc[COMMON, "D_rule"].fillna("").astype(str)
rb = B.loc[COMMON, "D_rule"].fillna("").astype(str)
DZ.to_csv(os.path.join(PROC, "readings_disagreements.csv"), index=False)
pd.set_option("display.width", 200)
print(AG.to_string(float_format=lambda x: f"{x:.3f}"))
print("D_rule label exact match (common-budget):", int((ra == rb).sum()), "of", len(ra))
print(pd.DataFrame({"A": ra, "B": rb}).to_string())
print(DZ[["unit", "field", "coder_A", "coder_B", "coder_A_rule", "coder_B_rule"]].to_string())
# [review] the agreement table is output/tables/rb5_units_agreement.csv (agreement.py, same fields); not rewritten here
