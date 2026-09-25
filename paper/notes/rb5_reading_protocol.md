# Coding protocol for budget readings, own caps, sizing rationales and synthetic-data flags (module rb5_units)

Version 1, written 2026-09-25 by the WP4b analysis agent (coder A) before any unit was coded under it. Fix list items W1(b)-(e), W2, W19(iv)-(v) (`paper/notes/round3_fixlist.md`, decisions D-1 and D-2). Changes after this version are allowed only as dated amendments at the end of this file, each with its reason; the coded files record the protocol version they used.

## 1. Purpose

The protocol fixes, before coding, how each allocation decision of the clean sample is classified from its developer's own primary sources:

1. the **reading** of a common token budget (what bound on the budget, and what constraint on the sizes, the sources document);
2. whether a member has its **own binding cap** on unique tokens (repetition of its pretraining corpus);
3. the **stated sizing rationale** of each decision;
4. the **synthetic-data flag**.

Coders classify statements in sources. They do not compute or look at wedges, shares or any other outcome. How a code maps into identification (point estimate or bound) is fixed in Section 8 and applied mechanically after coding; coders do not apply it.

## 2. Units

- A **family** is a model generation of one developer (the `gen` field of the clean sample, e.g. "Qwen2", "Llama", "OLMo-2").
- A **decision unit** is one choice of (N, D). Members of one family whose token budgets lie within 10 percent of each other (max/min D <= 1.10; clustering by ascending D, a new group starting when D exceeds 1.10 times the group's smallest D) form one **common-budget unit**. Every other model is a unit of its own: a **size-specific member** (its family has other members with other budgets) or a **singleton** (the only clean-sample model of its family).
- The coding frame `data/processed/rb5_units/coding_frame.csv` lists every unit with its members, parameter counts, token budgets, release dates, developer and the keys of the saved sources that concern it. It contains no wedge, share or outcome.

## 3. Sources and quotes

- **Only the saved primary sources** in `data/raw/rb2_decisions/` may be used (text renderings `<key>.txt`; URLs and retrieval times in `manifest.json`). Knowledge from outside the saved sources, including the coder's own memory of a model, is not evidence. If the saved sources are silent on a variable, the code is "not stated".
- Order of precedence when sources conflict: technical report, then the developer's release post or blog, then the model card or README. A later source by the same developer overrides an earlier one only if it states a correction.
- Every code other than "not stated" carries at least one **verbatim quote** of at most about 60 words. A quote must be found in the saved text after deleting all whitespace and folding case (the check of `code/analysis/rb2_decisions/evidence.py::verify`). Record the source key; the URL and retrieval time are taken from the manifest.
- A statement about a member, or about the whole release or family, counts for a unit if it concerns at least one of the unit's members and does not single out a member outside the unit.

## 4. Blindness

- Coders must not open: `data/processed/rb2_decisions/*`, `data/processed/rb5_units/*` other than `coding_frame.csv`, `output/tables/rb2_decisions_*`, `output/tables/rb5_*`, `output/memos/*`, `paper/sections/wedge.tex`, `paper/sections/appendix_wedge.tex`, `paper/tables/table2_decisions.tex` and `paper/tables/appF_*`.
- Disclosure for coder A: coder A is the analysis agent of WP4b. It had seen the previous version's wedges and readings (they were in the files it had to modify) before this protocol was written. Its coding follows the rules below from the quotes alone; the check against that knowledge is the blind second coding (coder B, a fresh agent that sees this protocol, the coding frame and the saved sources only).

## 5. Variable 1: the reading of a common budget (common-budget units only)

A reading combines a code for the token margin (D) with a code for the size margin (N).

### 5.1 Token margin (D)

**cap** if the sources give at least one of:
- **C1.** A statement that the members were trained on all the data the developer had available at the time, or that the available data were exhausted or limited (for example "the dataset currently consists of X tokens", "all available data", "we ran out of data").
- **C2.** Repetition of the pretraining corpus as a whole at this budget: the sources state more than one epoch (or repeated passes) over the unique pretraining corpus, or they state a unique corpus size that the budget exceeds by more than 10 percent.
- **C3.** A statement that the budget was limited by data: the developer would have trained on more tokens had more (fresh) data been available.

**choice** if the sources give at least one of, and none of C1-C3:
- **K1.** The budget was set by a cost, compute, performance-cost or schedule consideration (for example "provides a good performance-cost trade-off"; a learning-rate schedule planned for more tokens and truncated).
- **K2.** The budget is stated to be a part of a larger pool of the developer's own usable data (for example "X tokens of a Y-token dataset", "a 6T mix drawn from a 9T pool", "sampled from a larger mix").
- **K3.** Other models of the same release or generation were trained on more tokens of the same data (so the data did not bind at this budget).

**cap or choice** if the sources state only the size of the corpus or dataset and that every member was trained on it (the budget equals the stated corpus), with no C or K evidence; or if C and K evidence are both present for the same members (record both quotes).

**Not evidence on the token margin** (these statements are consistent with every reading):
- the model was still improving, had not saturated, or would improve with more data (a positive marginal product of data);
- the developer curated, filtered, expanded or synthesized data (effort to build a corpus);
- training went beyond the compute-optimal point, or was long "for inference" (this is rationale evidence, Section 7).

### 5.2 Size margin (N)

**menu** if the sources state that the size of at least one member of the unit was set to fit a specific hardware, memory or device constraint (a named GPU, a memory size, a consumer device) or was taken from a standard or prevalent configuration. Otherwise **free** (not stated). A general statement that the models are small, efficient or run locally is rationale evidence (Section 7), not a menu.

### 5.3 Combined reading

| D margin | N free | N menu |
|---|---|---|
| choice | choice | menu |
| cap | cap | cap+menu |
| cap or choice | cap or choice | cap or choice, menu |

## 6. Variable 2: own binding cap of a member (every model)

- Code **own cap = 1** if the sources state that the model's pretraining corpus was used for more than one epoch, or give repeated passes, or state a unique pretraining-corpus size that the model's token budget exceeds by more than 10 percent (both numbers from the saved sources).
- Upsampling of selected sources (for example two epochs of Wikipedia and books), repetition inside a mid-training or annealing mix, and "approximately one epoch" do not count.
- For a common-budget unit, own cap = 1 for every member is the C2 case of the reading and is coded there.

## 7. Variable 3: stated sizing rationale (every unit)

Four binary flags. A flag is 1 if the sources contain a statement, concerning a member of the unit or the whole release, that gives this reason for the model's size or its token budget:

- **R-CO, compute-optimal target.** The size or the token budget was chosen to be compute-optimal for the training budget, or set by a scaling-law or Chinchilla-type allocation (for example "approximately compute-optimal size for our training budget", "following Chinchilla").
- **R-INF, inference or serving cost.** The model was trained beyond the compute-optimal point, or sized, to improve performance at a given inference cost or budget, to reduce serving cost, or for inference efficiency.
- **R-DEV, device or memory target.** The size was chosen to fit a device, a GPU or a memory size, or the model is stated to be built for on-device, local, edge or consumer-hardware use.
- **R-NONE** = 1 if the other three are 0.

A statement that singles out another member (for example "our flagship is compute-optimal") counts only for the unit that contains that member. **Deployment rationale** = R-INF or R-DEV.

## 8. Mapping to identification (applied after coding; decision D-1)

The estimand is the virtual value of compactness v (the objective's own value plus the shadow value of any binding limit on size). A binding limit on size (a tier, a memory target, a menu) is part of v; a binding cap on tokens makes w - 1 a lower bound on v.

| Unit | Identification class |
|---|---|
| choice, menu | point (the family share) |
| cap, cap+menu | lower bound |
| cap or choice, cap or choice + menu | lower bound (point under the choice reading) |
| size-specific member or singleton, own cap = 0 | point |
| size-specific member or singleton, own cap = 1 | lower bound |

No unit is an upper bound under D-1 on the evidence types of this protocol. Tier windows and deployment rationales are reported as interpretation (mechanism), not identification.

## 9. Variable 4: synthetic-data flag (every unit)

One rule (W19(iv)): flag = 1 if the saved sources document teacher-generated (model-generated) synthetic data in the pretraining or mid-training data of a member of the unit. Rephrased, translated or filtered natural text counts only if a model generated it. Quote required.

## 10. Output format

One CSV per coder, `data/processed/rb5_units/readings_coder{A,B}.csv`, one row per unit:
`unit, unit_kind (common-budget / size-specific member / singleton), members, D_margin, D_rule (C1-C3, K1-K3 or "size only"), N_margin, N_rule, reading, own_cap, own_cap_members, R_CO, R_INF, R_DEV, R_NONE, synthetic, quote_1..quote_k, source_1..source_k, url_1..url_k, retrieved_1..retrieved_k, quote_role_1..quote_role_k (D, N, own_cap, R_CO, R_INF, R_DEV, synthetic), protocol_version, coder, note`.
Every quote is verified against the saved text by `code/analysis/rb5_units/verify_readings.py` before any analysis uses the file.

## 11. Agreement and resolution

- Agreement is reported with Cohen's kappa and raw agreement for: the reading (common-budget units), the D-margin code, own cap, and each rationale flag and the synthetic flag (all units).
- Disagreements are resolved by applying Sections 5-9 to the union of both coders' quotes. Each resolution cites the rule that decides it and is recorded in `readings_resolved.csv` with both original codes. The primary analysis uses the resolved codes; coder A's and coder B's codes are reported as robustness rows where they change a result.

## Amendments

All three were written on 2026-09-25 by coder A while coding, after reading the sources named, and before any identification count, median or other outcome was computed from the codes. Coder B receives the protocol with these amendments.

**Amendment 1 (Section 6: own cap from statements about data availability).** Version 1 coded a member's own binding cap only from repetition, although Section 5 recognizes three kinds of cap evidence (C1-C3). From this amendment on, own cap = 1 also when the sources give C1 or C3 evidence that concerns the member (a statement that the member's own budget was limited by the data available at the time). Reason: the Falcon report states that the budgets of its 40B and 180B models were partly set by the data available when training started; under version 1 such a statement about a size-specific member could not be coded, which made Sections 5 and 6 inconsistent.

**Amendment 2 (Section 6: which budget is compared with the corpus).** The model's token budget may be taken from the coding frame (the audited budget that the analysis uses) when the saved source that states the size of the unique pretraining corpus does not state the budget of that release. Reason: for OLMo-1B-hf (frame: 3T; the OLMo report states that one epoch is 2T) and TinyLlama v1.1 (frame: 2T; the project README states a combined dataset of about 950B tokens) the model cards that state the budgets were not saved.

**Amendment 3 (Section 7: clarification of "gives this reason").** A stated benefit of the model's size, or of its long training, counts as a reason (for example "since it can be run on a single GPU", "it is also fast during inference"). Statements about architecture (for example grouped-query attention for inference) or about fine-tuning hardware do not concern the size and do not count.

## Resolution record (Section 11) [review]

Written 2026-09-25 by the independent reviewer of module rb5_units (package WP4b-review), after the blind second coding (`readings_coderB.csv`, saved 13:08:48 +03) and the list of disagreements (`readings_disagreements.csv`). No rule of Sections 5-9 was changed. The resolved codes are in `data/processed/rb5_units/readings_final.csv` (this protocol's `readings_resolved.csv`; written by `code/analysis/rb5_units/resolve.py`, which records both original codes of every field, the decision, the rule and the reason). Disclosure: the reviewer had seen version 3's and coder A's wedges before resolving; every decision below rests on the quoted text and the rules, and none moves a unit from a lower bound to a point estimate.

| Unit | Field | A | B | Resolved | Deciding rule |
|---|---|---|---|---|---|
| Granite 3.0 | D margin, reading | cap or choice | choice (K2) | cap or choice | 5.1: K2 needs a stated usable pool larger than the budget; the 15T is FineWeb's size before IBM's own deduplication and quality filtering (report Section 3.1), so no pool is stated; no C or K evidence |
| Llama 3 herd | D margin, reading | cap or choice | choice (K1) | cap or choice | 5.1 and 7: the flagship's compute-optimal sizing is the R-CO example of Section 7 (coded by both), not a statement that the token budget was set below the data available (the law's 16.55T exceeds the "about 15T" corpus; the flagship trained 15.6T); size only, as fix list W1(c) specifies |
| DeepSeek LLM | own cap | 0 | 1 | 0 | 6, last bullet: an all-member cap in a common budget is coded in the reading (C1 here; class unchanged) |
| SmolLM 135M, 360M | own cap | 0 | 1 | 0 | 6, last bullet (C2 in the reading; class unchanged) |
| DeepSeek LLM | R-CO | 1 | 0 | 0 | 7: "under the guidance of our scaling laws" does not say that the sizes ("two prevalent used open-source configurations") or the 2T budget were set by an allocation |
| StableLM 2 1.6B | R-INF | 0 | 1 | 1 | 7 with Amendment 3: "without the computational overhead of larger models" is a stated inference benefit of the size |

Mapping note (Section 8), a correction disclosed as a deviation: a "cap or choice" budget whose cap evidence is C2 (every member repeated the unique corpus; SmolLM 135M and 360M) is classed a lower bound, not "lower bound (point if choice)": under Section 6 each member's own cap binds whatever the reading of the token margin. This changes no count of point estimates and lower bounds; it removes SmolLM 135M and 360M from the "point if every cap or choice were a choice" sensitivity (35 units instead of 36).
