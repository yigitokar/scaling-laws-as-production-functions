"""readings.py -- the codes that the analysis uses, and their mapping into identification classes (protocol Section 8;
decision D-1: the estimand is the virtual value of compactness v).

The primary codes are the resolved codes: data/processed/rb5_units/readings_final.csv, written by resolve.py from both
coders' files under protocol Section 11 ([review]; the protocol's 'readings_resolved.csv' is accepted as well); before the
resolution they were coder A's (readings_coderA.csv). Every output records which file was used.

Identification classes:
  point                          choice or menu budgets; size-specific members and singletons without an own cap
  lower bound                    cap and cap+menu budgets; members and singletons with an own binding cap; [review] a
                                 'cap or choice' budget whose cap evidence is C2, repetition of the corpus by every
                                 member (protocol Sections 5.1 and 6: each member's own cap binds whatever the reading of
                                 the token margin, so the unit stays a lower bound if the budget was chosen)
  lower bound (point if choice)  other 'cap or choice' budgets (with or without a menu)
The fix list's synthetic flag (W19(iv): SmolLM 1-3, phi-1.5, phi-2, Qwen2.5, Qwen3, OLMo 2) is carried beside the
protocol's flag.
"""
from __future__ import annotations

import os

import pandas as pd

import r5common as C

FIXLIST_SYNTHETIC_FAMILIES = {"SmolLM", "SmolLM2", "SmolLM3", "Phi-1.5", "Phi-2", "Qwen2.5", "Qwen3", "OLMo-2"}
CODE_FILES = ("readings_final.csv", "readings_resolved.csv", "readings_coderA.csv")   # [review] final file first


def source_file():
    for name in CODE_FILES:
        p = os.path.join(C.PROC, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("no readings file: run code/analysis/rb5_units/verify_readings.py first")


def repeats_corpus(d_rule):
    """[review] True when the cap evidence of a common budget is C2 (every member repeated the unique corpus)."""
    return "C2" in str(d_rule)


def id_class(unit_kind, reading, own_cap, d_rule=""):
    if unit_kind == "common-budget":
        if reading in ("choice", "menu"):
            return "point"
        if reading in ("cap", "cap+menu"):
            return "lower bound"
        if reading.startswith("cap or choice"):
            return "lower bound" if repeats_corpus(d_rule) else "lower bound (point if choice)"
        raise ValueError(reading)
    return "lower bound" if int(own_cap) else "point"


def load(path=None):
    p = path or source_file()
    R = pd.read_csv(p).fillna({"reading": "", "D_margin": "", "D_rule": "", "N_margin": "", "note": ""})
    R["id_class"] = [id_class(k, r, o, d) for k, r, o, d in zip(R["unit_kind"], R["reading"], R["own_cap"], R["D_rule"])]
    R["point_if_choice"] = R["id_class"].isin(["point", "lower bound (point if choice)"])
    if "family" not in R.columns:                      # coder B's file has no family column: take it from the frame
        fr = pd.read_csv(os.path.join(C.PROC, "coding_frame.csv")).set_index("unit")["family"]
        R["family"] = R["unit"].map(fr)
    R["synthetic_fixlist"] = R["family"].isin(FIXLIST_SYNTHETIC_FAMILIES).astype(int)
    if "deployment_rationale" not in R.columns:
        R["deployment_rationale"] = ((R["R_INF"] == 1) | (R["R_DEV"] == 1)).astype(int)
    R["codes_file"] = os.path.basename(p)
    return R
