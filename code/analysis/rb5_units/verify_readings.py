"""verify_readings.py -- turn a coder's codes (coderA.py, or a coder-B CSV) into the protocol's output file and verify
every quote against the saved primary sources (whitespace removed, case folded). Stops on any unverified quote or any
unit of the coding frame without a code.

  .venv/bin/python code/analysis/rb5_units/verify_readings.py        # writes data/processed/rb5_units/readings_coderA.csv
  .venv/bin/python code/analysis/rb5_units/verify_readings.py data/processed/rb5_units/readings_coderB.csv   # checks coder B
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd  # noqa: E402

import r5common as C  # noqa: E402

READING = {("choice", "free"): "choice", ("choice", "menu"): "menu", ("cap", "free"): "cap", ("cap", "menu"): "cap+menu",
           ("cap or choice", "free"): "cap or choice", ("cap or choice", "menu"): "cap or choice, menu"}


def build_coderA():
    import coderA
    F = pd.read_csv(os.path.join(C.PROC, "coding_frame.csv"))
    man = C.manifest()
    missing = sorted(set(F["unit"]) - set(coderA.U))
    extra = sorted(set(coderA.U) - set(F["unit"]))
    assert not missing and not extra, (missing, extra)
    rows, bad = [], []
    for _, f in F.iterrows():
        c = coderA.U[f["unit"]]
        assert c["kind"] == f["unit_kind"], (f["unit"], c["kind"], f["unit_kind"])
        common = c["kind"] == "common-budget"
        reading = READING[(c["D"], c["N"])] if common else ""
        r_none = int(not (c["R_CO"] or c["R_INF"] or c["R_DEV"]))
        row = dict(unit=f["unit"], unit_kind=c["kind"], family=f["family"], developer=f["developer"], members=f["members"],
                   D_margin=c.get("D", ""), D_rule=c.get("D_rule", ""), N_margin=c.get("N", ""), N_rule=c.get("N_rule", ""),
                   reading=reading, own_cap=int(c["own_cap"]), own_cap_rule=c.get("own_rule", ""),
                   R_CO=int(c["R_CO"]), R_INF=int(c["R_INF"]), R_DEV=int(c["R_DEV"]), R_NONE=r_none,
                   deployment_rationale=int(c["R_INF"] or c["R_DEV"]), synthetic=int(c["syn"]),
                   n_quotes=len(c["ev"]), protocol_version=C.PROTOCOL_VERSION + " with Amendments 1-3", coder="A",
                   note=c.get("note", ""))
        for i, (role, key, q) in enumerate(c["ev"], start=1):
            ok = C.verify(key, q)
            if not ok:
                bad.append((f["unit"], key, q[:90]))
            row.update({f"quote_role_{i}": role, f"source_{i}": key, f"quote_{i}": q,
                        f"url_{i}": man.get(key, {}).get("url", ""), f"retrieved_{i}": man.get(key, {}).get("retrieved", "")})
        rows.append(row)
    if bad:
        for b in bad:
            print("UNVERIFIED:", b)
        raise SystemExit(f"{len(bad)} quotes not found in the saved sources")
    out = pd.DataFrame(rows)
    # every code other than 'not stated' needs a quote of its role
    for _, r in out.iterrows():
        roles = {r[c] for c in out.columns if c.startswith("quote_role_") and isinstance(r[c], str)}
        if r["unit_kind"] == "common-budget":
            assert "D" in roles, r["unit"]
            if r["N_margin"] == "menu":
                assert "N" in roles, r["unit"]
        for flag, role in (("R_CO", "CO"), ("R_INF", "INF"), ("R_DEV", "DEV"), ("synthetic", "SYN")):
            if r[flag]:
                assert role in roles or (role == "DEV" and "N" in roles), (r["unit"], flag)
        if r["own_cap"] and r["unit_kind"] != "common-budget":
            assert "CAP" in roles, r["unit"]
    return out


def verify_csv(path):
    """Check a coder's CSV in the protocol's format (e.g. readings_coderB.csv): every frame unit coded once, every quote
    found in its saved source, codes in the protocol's categories. Returns the list of problems (empty if none)."""
    F = pd.read_csv(os.path.join(C.PROC, "coding_frame.csv"))
    X = pd.read_csv(path)
    probs = []
    if sorted(X["unit"]) != sorted(F["unit"]):
        probs.append("units differ from the coding frame")
    for _, r in X.iterrows():
        if r.get("unit_kind") == "common-budget" and str(r.get("reading")) not in set(READING.values()):
            probs.append(f"{r['unit']}: reading '{r.get('reading')}' not in the protocol")
        for c in X.columns:
            if c.startswith("quote_") and not c.startswith("quote_role_") and isinstance(r[c], str) and r[c].strip():
                i = c.split("_")[1]
                if not C.verify(str(r.get(f"source_{i}")), r[c]):
                    probs.append(f"{r['unit']}: quote {i} not found in {r.get(f'source_{i}')}")
    return probs


if __name__ == "__main__":
    if len(sys.argv) > 1:                 # verify another coder's file
        pr = verify_csv(sys.argv[1])
        print("\n".join(pr) if pr else "all units coded and all quotes verified")
        raise SystemExit(1 if pr else 0)
    A = build_coderA()
    A.to_csv(os.path.join(C.PROC, "readings_coderA.csv"), index=False)
    print(A[["unit", "unit_kind", "reading", "own_cap", "R_CO", "R_INF", "R_DEV", "synthetic", "n_quotes"]].to_string())
    print("all quotes verified:", int(A["n_quotes"].sum()))
