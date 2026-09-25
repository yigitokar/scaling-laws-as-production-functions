"""agreement.py -- agreement between coder A and the blind coder B (protocol Section 11): Cohen's kappa and raw agreement
for the reading (common-budget units), the token-margin code, own cap, the three rationale flags and the synthetic flag,
with the list of disagreements for resolution. Runs only when data/processed/rb5_units/readings_coderB.csv exists.

  .venv/bin/python code/analysis/rb5_units/agreement.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import r5common as C  # noqa: E402

VARS = [("reading", "common-budget"), ("D_margin", "common-budget"), ("own_cap", "all"), ("R_CO", "all"),
        ("R_INF", "all"), ("R_DEV", "all"), ("synthetic", "all"),
        # [review] the fields of the blind coder's comparison as well
        ("N_margin", "common-budget"), ("own_cap", "size-specific member or singleton"), ("R_NONE", "all"),
        ("deployment_rationale", "all")]


def cohen_kappa(a, b):
    a, b = np.asarray(a).astype(str), np.asarray(b).astype(str)
    cats = sorted(set(a) | set(b))
    po = float(np.mean(a == b))
    pe = float(sum(np.mean(a == c) * np.mean(b == c) for c in cats))
    return po, (po - pe) / (1 - pe) if pe < 1 else np.nan


def run():
    pa, pb = (os.path.join(C.PROC, f"readings_coder{x}.csv") for x in "AB")
    if not os.path.exists(pb):
        print("readings_coderB.csv not found: agreement not computed")
        return None, None
    A, B = pd.read_csv(pa).set_index("unit"), pd.read_csv(pb).set_index("unit")
    assert set(A.index) == set(B.index), "coders coded different units"
    B = B.loc[A.index]
    for X in (A, B):                                   # [review] coder B's file has no deployment column
        X["deployment_rationale"] = ((X["R_INF"] == 1) | (X["R_DEV"] == 1)).astype(int)
    rows, dis = [], []
    for v, scope in VARS:
        idx = (A.index if scope == "all" else A.index[A["unit_kind"] == "common-budget"] if scope == "common-budget"
               else A.index[A["unit_kind"] != "common-budget"])
        a, b = A.loc[idx, v].fillna(""), B.loc[idx, v].fillna("")
        po, k = cohen_kappa(a, b)
        rows.append(dict(variable=v, scope=scope, n=len(idx), agreement=po, kappa=k))
        for u in (idx[(a.astype(str) != b.astype(str)).values] if scope != "size-specific member or singleton" else []):
            dis.append(dict(unit=u, variable=v, coder_A=A.loc[u, v], coder_B=B.loc[u, v]))
    K, Dz = pd.DataFrame(rows), pd.DataFrame(dis)
    C.tab(K, "agreement")
    Dz.to_csv(os.path.join(C.PROC, "disagreements.csv"), index=False)
    return K, Dz


if __name__ == "__main__":
    K, Dz = run()
    if K is not None:
        print(K.to_string())
        print(Dz.to_string())
