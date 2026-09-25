"""resolve.py -- [review] resolution of the disagreements between coder A (readings_coderA.csv) and the blind coder B
(readings_coderB.csv) under protocol Section 11 (paper/notes/rb5_reading_protocol.md): every disagreement listed in
data/processed/rb5_units/readings_disagreements.csv is decided by applying protocol Sections 5-9 to the union of both
coders' quotes. Agreed codes are kept as coded. Writes data/processed/rb5_units/readings_final.csv, the resolved codes
that the analysis reads (readings.py prefers it), with both original codes of every field, the decision, the rule and
the reason, and verifies every quote against the saved sources.

Written by the independent reviewer of module rb5_units (package WP4b-review, 2026-09-25). The reviewer had seen the
version-3 and coder-A wedges before resolving (disclosure in output/memos/rb5_units_review.md); each decision below
rests on the quoted text and the protocol's rules only, and none of them moves a unit from a lower bound to a point
estimate.

  nice -n 10 .venv/bin/python code/analysis/rb5_units/resolve.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd  # noqa: E402

import r5common as C  # noqa: E402
import verify_readings as VR  # noqa: E402

FIELDS = ["D_margin", "D_rule", "N_margin", "N_rule", "reading", "own_cap", "R_CO", "R_INF", "R_DEV", "R_NONE",
          "synthetic"]

# (unit, field) -> (final value, deciding rule, reason). Fields not listed keep the agreed code.
DECISIONS = {
    ("Granite-3.0", "D_margin"): (
        "cap or choice", "protocol 5.1: no C and no K evidence; K2 not established",
        "K2 needs the budget to be stated as part of a larger pool of the developer's own usable data. The report gives "
        "the dense budget (12T; 10T in stage 1) and, in its data appendix, the size of one stage-1 source before IBM's "
        "processing (FineWeb, 'more than 15T tokens'); its Section 3.1 applies IBM's own exact and fuzzy deduplication, "
        "HAP filtering and Gopher and KenLM quality filtering to the curated web data that include FineWeb and DCLM, so "
        "the usable pool after processing is not stated, and the stage-1 web share (69.5 percent) is read off a figure. "
        "Coder B's K2 is an inference from source sizes, as coder B's own note says; no C evidence either. The residual "
        "code of 5.1 for a stated budget without C or K evidence is 'cap or choice' (coder A)."),
    ("Granite-3.0", "D_rule"): ("size only", "protocol 5.1", "see D_margin"),
    ("Granite-3.0", "reading"): ("cap or choice", "protocol 5.3 (cap or choice, N free)", "follows from D_margin; N free (both coders)"),
    ("Llama-3 herd", "D_margin"): (
        "cap or choice", "protocol 5.1 (size only) and Section 7 (R-CO); fix list W1(c)",
        "Coder B's K1 rests on the flagship's allocation: the scaling law 'suggests training a 402B parameter model on "
        "16.55T tokens' and Meta 'decided to train a flagship model with 405B parameters'. These sentences set the "
        "flagship's size for a FLOP budget; the protocol lists 'approximately compute-optimal size for our training "
        "budget' (this report's wording) as its example of the R-CO rationale (Section 7), which both coders coded. They "
        "do not state that the token budget was set below the data available: the law's 16.55T exceeds the 'corpus of "
        "about 15T multilingual tokens', and the flagship trained on 15.6T, about the corpus size, so the sentences fit a "
        "corpus-bound budget as well as a chosen one. 'Continued to improve' and training 'much longer than is "
        "compute-optimal' are not token-margin evidence (5.1). The sources give the corpus size and that every member "
        "trained on it: 'cap or choice' (size only), as coder A coded and as fix list W1(c) specifies for the herd "
        "unless the sources state that the corpus bound (they do not)."),
    ("Llama-3 herd", "D_rule"): ("size only", "protocol 5.1", "see D_margin"),
    ("Llama-3 herd", "reading"): ("cap or choice", "protocol 5.3 (cap or choice, N free)", "follows from D_margin; N free (both coders)"),
    ("DeepSeek-LLM", "own_cap"): (
        0, "protocol Section 6, last bullet (own cap of every member of a common budget is coded in the reading)",
        "Both coders code the reading cap+menu from C1 ('a dataset that currently consists of 2 trillion tokens'). Section "
        "6 codes the case in which every member of a common budget is capped in the reading, not in the member field; "
        "Amendment 1 extends the member field to C1/C3 statements that concern a member, which here is the same "
        "all-member statement. Convention of coder A; the class (lower bound) is the same under both conventions."),
    ("SmolLM#1", "own_cap"): (
        0, "protocol Section 6, last bullet",
        "The repetition (600B tokens over a stated 252B-token corpus, about 2.4 passes) concerns both members and is coded "
        "in the reading as C2 (both coders). Convention of coder A. Because every member repeated its corpus, each "
        "member's own cap binds whatever the reading of the token margin, so the unit is a lower bound also under the "
        "'point if choice' sensitivity (readings.id_class, reviewer correction)."),
    ("DeepSeek-LLM", "R_CO"): (
        0, "protocol Section 7 (a flag needs a statement that gives the reason for the size or the token budget)",
        "Coder A's quote ('Under the guidance of our scaling laws, we build from scratch open-source large language "
        "models') does not say that the sizes or the 2T budget were set by a compute-optimal or scaling-law allocation. "
        "The report attributes the sizes to 'two prevalent used open-source configurations, 7B and 67B', chosen while "
        "'maintaining parameter consistency with other open-source models', and the budget to the dataset collected; the "
        "uses of its scaling laws that it names are hyperparameters and forecasts of the 7B's and 67B's performance "
        "(Sections 3.1-3.2). Consistent with both coders' treatment of LLaMA's 'inspired by the Chinchilla scaling "
        "laws' (not coded). Coder B."),
    ("DeepSeek-LLM", "R_NONE"): (1, "protocol Section 7", "follows from R_CO = 0 (R_INF = R_DEV = 0 for both coders)"),
    ("B:stabilityai/stablelm-2-1_6b", "R_INF"): (
        1, "protocol Section 7, R-INF, with Amendment 3 (a stated benefit of the size counts)",
        "The report states a benefit of the model's small size for inference compute: 'available directly on-device "
        "without the computational overhead of larger models', followed by 'a great balance between remarkable "
        "efficiency and effectiveness in inference tasks'. Amendment 3's example ('it is also fast during inference') is "
        "of this kind. R_DEV = 1 for both coders, so the deployment rationale is unchanged. Coder B."),
}
ADDED_QUOTES = {   # quotes of coder B that carry a resolved code (role, source key, quote)
    "B:stabilityai/stablelm-2-1_6b": [("INF", "stablelm2_report", "This model represents a substantial leap towards making "
                                       "advanced generation capabilities available directly on-device without the "
                                       "computational overhead of larger models.")],
}
RELABELED = {  # coder A's quotes whose role no longer carries a code: (unit, role) -> new role
    ("DeepSeek-LLM", "CO"): "NONEV",
}


def run():
    A = pd.read_csv(os.path.join(C.PROC, "readings_coderA.csv"))
    B = pd.read_csv(os.path.join(C.PROC, "readings_coderB.csv")).set_index("unit").loc[A["unit"]]
    Dz = pd.read_csv(os.path.join(C.PROC, "readings_disagreements.csv"))
    # every listed disagreement has a decision, and every disagreement between the files is listed or is a convention
    listed = set(zip(Dz["unit"], Dz["field"]))
    decided = set(DECISIONS)
    assert listed <= decided, sorted(listed - decided)
    for f in ["D_margin", "N_margin", "reading", "own_cap", "R_CO", "R_INF", "R_DEV", "R_NONE", "synthetic"]:
        a = A.set_index("unit")[f].fillna("").astype(str)
        b = B[f].fillna("").astype(str)
        for u in a.index[(a != b).values]:
            assert (u, f) in decided, (u, f, a[u], b[u])
    F = A.copy()
    for f in FIELDS:
        F[f"coder_A_{f}"] = A[f]
        F[f"coder_B_{f}"] = B[f].values if f in B.columns else ""
    F["resolution"] = "agreed (coded identically by A and B)"
    F["resolution_rule"] = ""
    F["resolved_fields"] = ""
    for (u, f), (val, rule, why) in DECISIONS.items():
        i = F.index[F["unit"] == u]
        assert len(i) == 1, u
        F.loc[i, f] = val
        prev = F.loc[i[0], "resolved_fields"]
        F.loc[i, "resolved_fields"] = (prev + "; " if prev else "") + f
        if f in ("D_margin", "own_cap", "R_CO", "R_INF"):
            F.loc[i, "resolution"] = (("" if F.loc[i[0], "resolution"].startswith("agreed") else F.loc[i[0], "resolution"] + " | ")
                                      + f"{f}: {why}")
            F.loc[i, "resolution_rule"] = (("" if not F.loc[i[0], "resolution_rule"] else F.loc[i[0], "resolution_rule"] + " | ")
                                           + f"{f}: {rule}")
    F["deployment_rationale"] = ((F["R_INF"] == 1) | (F["R_DEV"] == 1)).astype(int)
    assert (F["R_NONE"] == 1 - ((F["R_CO"] + F["R_INF"] + F["R_DEV"]) > 0).astype(int)).all()
    # quotes: relabel and add
    qcols = sorted({int(c.split("_")[-1]) for c in F.columns if c.startswith("quote_role_")})
    for (u, role), new in RELABELED.items():
        i = F.index[F["unit"] == u][0]
        for k in qcols:
            if F.loc[i, f"quote_role_{k}"] == role:
                F.loc[i, f"quote_role_{k}"] = new
    man = C.manifest()
    for u, qs in ADDED_QUOTES.items():
        i = F.index[F["unit"] == u][0]
        k = max([k for k in qcols if isinstance(F.loc[i, f"quote_{k}"], str)] + [0])
        for role, key, q in qs:
            k += 1
            for c in ("quote_role", "source", "quote", "url", "retrieved"):
                if f"{c}_{k}" not in F.columns:
                    F[f"{c}_{k}"] = pd.Series(dtype=object)
            F[f"quote_{k}"] = F[f"quote_{k}"].astype(object)
            F.loc[i, [f"quote_role_{k}", f"source_{k}", f"quote_{k}", f"url_{k}", f"retrieved_{k}"]] = \
                [role, key, q, man[key]["url"], man[key]["retrieved"]]
        F.loc[i, "n_quotes"] = k
    F["protocol_version"] = C.PROTOCOL_VERSION + " with Amendments 1-3; disagreements resolved under Section 11"
    F["coder"] = "resolved (coders A and B; reviewer WP4b-review)"
    # order columns: codes, provenance, then quotes
    qc = [c for c in F.columns if c.split("_")[0] in ("quote", "source", "url", "retrieved") and c[-1].isdigit()]
    qc = sorted(qc, key=lambda c: (int(c.split("_")[-1]), ["quote_role", "source", "quote", "url", "retrieved"].index(c.rsplit("_", 1)[0])))
    other = [c for c in F.columns if c not in qc]
    F = F[other + qc]
    out = os.path.join(C.PROC, "readings_final.csv")
    F.to_csv(out, index=False)
    probs = VR.verify_csv(out)
    assert not probs, probs
    # every flag of 1 carries a quote of its role (coder A's roles; coder B's roles for added quotes)
    for _, r in F.iterrows():
        roles = {r[f"quote_role_{k}"] for k in qcols if isinstance(r.get(f"quote_role_{k}"), str)}
        roles |= {r[c] for c in F.columns if c.startswith("quote_role_") and isinstance(r[c], str)}
        for flag, role in (("R_CO", "CO"), ("R_INF", "INF"), ("R_DEV", "DEV"), ("synthetic", "SYN")):
            if r[flag]:
                assert role in roles or (role == "DEV" and "N" in roles), (r["unit"], flag)
    return F


if __name__ == "__main__":
    F = run()
    ch = F[F["resolved_fields"] != ""]
    print(ch[["unit", "resolved_fields", "reading", "own_cap", "R_CO", "R_INF", "R_NONE"]].to_string())
    print("readings_final.csv written;", len(F), "units; all quotes verified")
