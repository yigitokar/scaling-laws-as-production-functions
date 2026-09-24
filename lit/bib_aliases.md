# BibTeX alias map for `lit/references.bib`

Built 2026-09-23 by the synthesis step. Inputs: the eight strand files in `lit/bib/` (577 entries). Output: `lit/references.bib` (309 unique works, sorted by key). Merge script logic: entries were grouped as the same work when they share an arXiv id, a DOI, or an identical normalized title; three hand-merges were added (Epoch AI-models pages; Epoch ECI + benchmarking hub; Open LLM Leaderboard pages) and one hand-split (Bergemann et al. EC'25 vs. 2026 Cowles DP, same arXiv id but different title and functional form). The richest entry of each group is the base; missing fields are filled only from entries of the same BibTeX type (plus arXiv ids/urls); all notes are kept and tagged `[strand]`; all UNVERIFIED notes are retained verbatim.

**Rule for authors of the paper:** always cite the canonical key in the right-hand column. Old keys are not defined in `references.bib`, so a stale key fails loudly at compile time rather than silently pointing at the wrong work.

## 1. Old key -> canonical key (only keys that changed)

| Old key | Canonical key | Strand(s) that used the old key | Reason |
|---|---|---|---|
| `anonymous2022overhangs` | `anonymous2022ai` | novelty | manual choice (see notes below) |
| `arellano1991some` | `arellano1991tests` | io_controlfn | renamed to satisfy key convention |
| `besiroglu2022economic` | `besiroglu2024economic` | novelty | duplicate of the same work; canonical = majority key (ties: key year = publication year) |
| `blackorby1989will` | `blackorby1989real` | io_ces_duality | renamed to satisfy key convention |
| `bond2021some` | `bond2020unpleasant` | io_controlfn | duplicate of the same work; canonical = majority key (ties: key year = publication year) |
| `caballero2022broken` | `caballero2023broken` | ml_estimation | duplicate of the same work; canonical = majority key (ties: key year = publication year) |
| `choshen2024hitchhiker` | `choshen2024hitchhikers` | ml_core | duplicate of the same work; canonical = majority key (ties: key year = publication year) |
| `davidson2023capabilities` | `davidson2023ai` | ml_observational | duplicate of the same work; canonical = majority key (ties: key year = publication year) |
| `deepseekai2024deepseek` | `bi2024deepseek` | data | AMBIGUOUS old key: in `data.bib` it meant DeepSeek LLM (arXiv 2401.02954); in `econ_ai.bib` it meant DeepSeek-V3. Retired. |
| `deepseekai2024deepseek` | `deepseekai2024deepseekv3` | econ_ai | AMBIGUOUS old key (see row above). Retired. |
| `demirer2026production` | `demirer2020production` | io_ces_duality | manual choice (see notes below) |
| `devries2023smol` | `devries2023go` | novelty | renamed to satisfy key convention |
| `dominguezolmedo2024training` | `dominguezolmedo2025training` | novelty | duplicate of the same work; canonical = majority key (ties: key year = publication year) |
| `epoch2025data` | `epochai2026data` | ml_observational | manual choice (see notes below) |
| `epoch2025eci` | `epochai2026capabilities` | ml_observational | manual choice (see notes below) |
| `epoch2026aimodels` | `epochai2026data` | data | manual choice (see notes below) |
| `epoch2026benchmarking` | `epochai2026capabilities` | data | manual choice (see notes below) |
| `epoch2026mlhardware` | `epochai2026machine` | data | renamed to satisfy key convention |
| `epochai2026notable` | `epochai2026data` | io_controlfn | manual choice (see notes below) |
| `fogelson2026metrics` | `fogelson2026ai` | ml_observational | renamed to satisfy key convention |
| `griliches1995production` | `griliches1998production` | novelty | duplicate of the same work; canonical = majority key (ties: key year = publication year) |
| `huggingface2024openllm` | `huggingface2024open` | ml_observational | manual choice (see notes below) |
| `levitt2013toward` | `levitt2013understanding` | io_ces_duality | renamed to satisfy key convention |
| `li2025farseer` | `li2025predictableb` | ml_estimation | key collision between two different works; a/b suffix by arXiv date |
| `li2025predictable` | `li2025predictablea` | ml_estimation | AMBIGUOUS old key: in `ml_estimation.bib` it meant Step Law (Predictable Scale Part I, arXiv 2503.04715). Retired. |
| `li2025predictable` | `li2025predictableb` | ml_core, novelty | AMBIGUOUS old key: in `ml_core.bib` and `novelty.bib` it meant Farseer (Predictable Scale Part II, arXiv 2506.10972). Retired. |
| `nordhaus2007two` | `nordhaus2007centuries` | econ_ai | renamed to satisfy key convention |
| `openllmleaderboard2025contents` | `huggingface2024open` | data | manual choice (see notes below) |
| `pilz2023increased` | `pilz2025increased` | econ_ai | duplicate of the same work; canonical = majority key (ties: key year = publication year) |
| `polo2024sloth` | `maiapolo2024sloth` | ml_estimation | duplicate of the same work; canonical = majority key (ties: key year = publication year) |
| `rosenfeld2019constructive` | `rosenfeld2020constructive` | data | duplicate of the same work; canonical = majority key (ties: key year = publication year) |
| `sardana2023beyond` | `sardana2024beyond` | ml_estimation | duplicate of the same work; canonical = majority key (ties: key year = publication year) |
| `syverson2011what` | `syverson2011determines` | io_controlfn | duplicate of the same work; canonical = majority key (ties: key year = publication year) |
| `villalobos2022will` | `villalobos2022run` | data | duplicate of the same work; canonical = majority key (ties: key year = publication year) |
| `villalobos2024run` | `villalobos2022run` | ml_observational | duplicate of the same work; canonical = majority key (ties: key year = publication year) |
| `xiao2024densing` | `xiao2025densing` | data, ml_core | duplicate of the same work; canonical = majority key (ties: key year = publication year) |

## 2. Manual key decisions

- `bi2024deepseek` (DeepSeek LLM, arXiv 2401.02954): arXiv lists the corporate author "DeepSeek-AI" first; we keep the majority key with Xiao Bi as first personal author so that the ambiguous `deepseekai2024deepseek` can be retired.
- `deepseekai2024deepseekv3` (DeepSeek-V3 Technical Report, arXiv 2412.19437): first substantive word read as the compound "DeepSeek-V3".
- `li2025predictablea` = Step Law (Part I, hyperparameter scaling; data `step-law/steplaw`); `li2025predictableb` = Farseer (Part II, NeurIPS 2025; data `Farseer-Scaling-Law/Farseer`).
- `epochai2026data` merges `epoch2025data` (ml_observational), `epoch2026aimodels` (data) and `epochai2026notable` (io_controlfn): one living database with Notable / Large-scale / Frontier subsets (subset URLs and inclusion rules are in the note).
- `epochai2026capabilities` merges `epoch2025eci` (ml_observational) and `epoch2026benchmarking` (data): the Epoch Capabilities Index and Benchmarking Hub (method paper: `ho2025rosetta`).
- `huggingface2024open` merges `huggingface2024openllm` (ml_observational) and `openllmleaderboard2025contents` (data): the Open LLM Leaderboard and its result/contents datasets.
- `demirer2020production`: MIT JMP (2020) is the verified version; io_ces_duality recorded "forthcoming Econometrica 2026" from the author website (kept in note as UNVERIFIED). The `@article` version keyed `demirer2026production` is folded in.
- `bond2020unpleasant`: NBER WP 27002 (2020) key kept; the entry is typed `@article` (JME 121:1--14, 2021) with the NBER DOI in the note; journal details are flagged UNVERIFIED because one strand could not confirm them.
- `griliches1998production`: the CUP (Str{\o}m ed.) chapter is canonical; NBER WP 5067 (1995, key `griliches1995production`) and an Edward Elgar reprint are recorded in the note.
- Kept-as-is despite a strict reading of the rule (hyphen/ampersand compounds treated as one word): `jones1995rd`, `doraszelski2013rd`, `nezhurina2025opensciref`, `teamolmo2024olmo` ("2 OLMo 2 Furious"; OLMo is the product name), `artificialanalysis2026api` ("Free" treated as an access descriptor), `li2025misfitting`, `sato1967twolevel`, `farboodi2020longrun`, `farboodi2026datadriven`, `luo2025multipower`, `ackerberg2023underidentification`.
- `anonymous2022ai`: LessWrong post by a deleted account (author field `{[deleted user]}`); key uses "anonymous".

## 3. Works kept as separate entries although related

| Keys | Why separate |
|---|---|
| `bergemann2025economics` / `bergemann2026menu` | Same arXiv id (2502.07736) but v1 (EC'25, Cobb-Douglas value function) and v2 (Cowles DP 2502, homogeneous Psi(x)Phi(z), "Menu Pricing of LLMs") differ in title and functional form; econ_ai cites both. |
| `demirer2025emerging` / `demirer2026emerging` | NBER w34608 (Demirer, Fradkin, Tadelis, Peng) vs. JEP 40(3) (Demirer, Fradkin, Tadelis); different titles and author lists. |
| `merali2024scaling` / `merali2025scaling` | Two different RCT papers (translation tasks 2024; consulting/analyst tasks 2025). |
| `epochai2026data` / `epochai2026capabilities` / `epochai2026machine` | Different Epoch data products (models database; ECI and benchmarks; ML hardware). |
| `wei2022emergent` / `schaeffer2023emergent` | Different papers (emergence claim vs. "mirage" critique). |

## 4. Same-key duplicates merged (key unchanged)

| Canonical key | # strand entries | Strands |
|---|---|---|
| `besiroglu2024chinchilla` | 8 | data, econ_ai, io_ces_duality, io_controlfn, ml_core, ml_estimation, ml_observational, novelty |
| `ho2024algorithmic` | 8 | data, econ_ai, io_ces_duality, io_controlfn, ml_core, ml_estimation, ml_observational, novelty |
| `hoffmann2022training` | 7 | data, econ_ai, io_ces_duality, io_controlfn, ml_core, ml_estimation, novelty |
| `kaplan2020scaling` | 7 | data, econ_ai, io_ces_duality, io_controlfn, ml_core, ml_estimation, novelty |
| `muennighoff2023scaling` | 7 | data, econ_ai, io_ces_duality, io_controlfn, ml_core, ml_estimation, novelty |
| `ruan2024observational` | 7 | data, econ_ai, io_controlfn, ml_core, ml_estimation, ml_observational, novelty |
| `gadre2024language` | 6 | data, io_controlfn, ml_core, ml_estimation, ml_observational, novelty |
| `pearce2024reconciling` | 6 | data, io_ces_duality, io_controlfn, ml_core, ml_estimation, novelty |
| `porian2024resolving` | 6 | data, io_ces_duality, io_controlfn, ml_core, ml_estimation, novelty |
| `ackerberg2015identification` | 5 | data, io_ces_duality, io_controlfn, ml_estimation, novelty |
| `deloecker2012markups` | 5 | data, io_ces_duality, io_controlfn, ml_estimation, novelty |
| `erdil2022algorithmic` | 5 | data, econ_ai, io_ces_duality, ml_observational, novelty |
| `foster2008reallocation` | 5 | data, io_ces_duality, io_controlfn, ml_estimation, novelty |
| `gandhi2020identification` | 5 | data, io_ces_duality, io_controlfn, ml_estimation, novelty |
| `hernandez2020measuring` | 5 | data, econ_ai, io_ces_duality, ml_observational, novelty |
| `jones2021scaling` | 5 | data, econ_ai, io_ces_duality, ml_core, novelty |
| `magnusson2025datadecide` | 5 | data, ml_core, ml_estimation, ml_observational, novelty |
| `marschak1944random` | 5 | data, io_ces_duality, io_controlfn, ml_estimation, novelty |
| `nerlove1963returns` | 5 | data, io_ces_duality, io_controlfn, ml_estimation, novelty |
| `owen2024predictable` | 5 | data, econ_ai, ml_estimation, ml_observational, novelty |
| `schaeffer2023emergent` | 5 | data, ml_core, ml_estimation, ml_observational, novelty |
| `snell2024scaling` | 5 | data, econ_ai, io_ces_duality, ml_core, novelty |
| `aigner1977formulation` | 4 | econ_ai, io_ces_duality, ml_estimation, novelty |
| `arrow1961capital` | 4 | econ_ai, io_ces_duality, ml_estimation, novelty |
| `bhagia2024establishing` | 4 | data, ml_core, ml_estimation, ml_observational |
| `biderman2023pythia` | 4 | data, io_controlfn, ml_estimation, ml_observational |
| `cottier2024rising` | 4 | data, econ_ai, ml_observational, novelty |
| `diamond1978measurement` | 4 | io_ces_duality, io_controlfn, ml_estimation, novelty |
| `doraszelski2018measuring` | 4 | data, io_ces_duality, io_controlfn, novelty |
| `grattafiori2024llama` | 4 | data, io_controlfn, ml_core, ml_estimation |
| `gundlach2025origin` | 4 | data, econ_ai, ml_observational, novelty |
| `levinsohn2003estimating` | 4 | data, io_controlfn, ml_estimation, novelty |
| `olley1996dynamics` | 4 | data, io_controlfn, ml_estimation, novelty |
| `sevilla2022compute` | 4 | data, econ_ai, ml_observational, novelty |
| `busbridge2025distillation` | 3 | data, ml_core, novelty |
| `czech2026problems` | 3 | econ_ai, ml_estimation, novelty |
| `hagele2024scaling` | 3 | data, ml_core, ml_estimation |
| `hsieh2009misallocation` | 3 | io_ces_duality, io_controlfn, novelty |
| `hu2024minicpm` | 3 | data, ml_core, ml_estimation |
| `klump2007factor` | 3 | io_ces_duality, ml_estimation, novelty |
| `kmenta1967estimation` | 3 | econ_ai, io_ces_duality, ml_estimation |
| `krajewski2024scaling` | 3 | data, ml_core, novelty |
| `leonledesma2010identifying` | 3 | econ_ai, io_ces_duality, ml_estimation |
| `lourie2025scaling` | 3 | ml_core, ml_estimation, novelty |
| `tao2024scaling` | 3 | data, ml_core, ml_estimation |
| `wu2024inference` | 3 | data, econ_ai, ml_core |
| `abnar2025parameters` | 2 | data, ml_core |
| `ackerberg2007econometric` | 2 | io_controlfn, ml_estimation |
| `alabdulmohsin2022revisiting` | 2 | ml_core, ml_estimation |
| `antras2004aggregate` | 2 | econ_ai, io_ces_duality |
| `arrow1962economic` | 2 | econ_ai, io_ces_duality |
| `bergemann2025economics` | 2 | econ_ai, novelty |
| `blundell2000gmm` | 2 | io_controlfn, novelty |
| `brandfonbrener2024loss` | 2 | data, ml_estimation |
| `brown2024large` | 2 | data, ml_core |
| `chen2024scaling` | 2 | ml_observational, novelty |
| `christensen1973transcendental` | 2 | io_ces_duality, ml_estimation |
| `christensen1976economies` | 2 | io_ces_duality, ml_estimation |
| `clark2022unified` | 2 | ml_core, novelty |
| `collardwexler2016production` | 2 | io_controlfn, ml_estimation |
| `cottier2025llm` | 2 | data, ml_observational |
| `deloecker2020rise` | 2 | io_controlfn, novelty |
| `demirer2025emerging` | 2 | econ_ai, novelty |
| `demirer2026emerging` | 2 | econ_ai, novelty |
| `douglas2025progress` | 2 | ml_observational, novelty |
| `erdil2024estimating` | 2 | econ_ai, novelty |
| `erdil2025gate` | 2 | econ_ai, novelty |
| `farrell1957measurement` | 2 | io_ces_duality, novelty |
| `gundlach2025meek` | 2 | data, ml_observational |
| `gundlach2025price` | 2 | econ_ai, ml_observational |
| `hanoch1971cresh` | 2 | io_ces_duality, ml_estimation |
| `heineman2025signal` | 2 | ml_estimation, ml_observational |
| `hernandez2021scaling` | 2 | data, ml_core |
| `hestness2017deep` | 2 | data, ml_core |
| `ho2025rosetta` | 2 | econ_ai, ml_observational |
| `houthakker1955pareto` | 2 | econ_ai, ml_core |
| `hu2020estimating` | 2 | io_controlfn, ml_estimation |
| `isik2024scaling` | 2 | ml_core, ml_observational |
| `jones2005shape` | 2 | econ_ai, ml_core |
| `jorgenson1967explanation` | 2 | econ_ai, io_ces_duality |
| `klette1996inconsistency` | 2 | io_controlfn, ml_estimation |
| `klump2012normalized` | 2 | io_ces_duality, ml_estimation |
| `korinek2025concentrating` | 2 | econ_ai, novelty |
| `kumar2024scaling` | 2 | data, ml_core |
| `kwa2025measuring` | 2 | econ_ai, ml_observational |
| `li2025misfitting` | 2 | ml_estimation, novelty |
| `lourie2026small` | 2 | ml_estimation, novelty |
| `madaan2024quantifying` | 2 | ml_estimation, ml_observational |
| `merali2024scaling` | 2 | econ_ai, novelty |
| `merali2025scaling` | 2 | econ_ai, novelty |
| `mertens2026secret` | 2 | ml_observational, novelty |
| `michaud2023quantization` | 2 | econ_ai, ml_core |
| `mundlak1961empirical` | 2 | io_controlfn, ml_estimation |
| `oberfield2021micro` | 2 | io_ces_duality, novelty |
| `raval2019micro` | 2 | io_ces_duality, io_controlfn |
| `raval2023testing` | 2 | io_controlfn, novelty |
| `sanderson2025rethinking` | 2 | ml_observational, novelty |
| `schaeffer2024predicting` | 2 | ml_estimation, ml_observational |
| `schaeffer2025evaluating` | 2 | ml_estimation, novelty |
| `thompson2020computational` | 2 | econ_ai, novelty |
| `vanderwal2025polypythias` | 2 | data, ml_estimation |
| `villalobos2023trading` | 2 | econ_ai, novelty |
| `wei2022emergent` | 2 | data, ml_observational |
| `whitfill2025compute` | 2 | econ_ai, novelty |
| `whitfill2025forecasting` | 2 | ml_observational, novelty |
| `whitfill2025note` | 2 | ml_observational, novelty |
| `wright1936factors` | 2 | econ_ai, io_ces_duality |
| `zellner1966specification` | 2 | io_controlfn, ml_estimation |
| `zhang2026economics` | 2 | econ_ai, novelty |

## 5. Field conflicts between strands and how they were resolved

The base entry (richest record, usually the proceedings/journal version) supplies the field; the alternatives are listed so they can be checked before submission. Differences that are only formatting of the same venue (e.g., "NeurIPS" vs "Advances in Neural Information Processing Systems 37") are omitted.

| Canonical key | Field | Values seen (strands) |
|---|---|---|
| `besiroglu2024economic` | year | `2024` (econ_ai); `2022` (novelty) |
| `bhagia2024establishing` | year | `2025` (ml_core, ml_estimation, ml_observational); `2024` (data) |
| `bond2020unpleasant` | year | `2021` (io_controlfn); `2020` (novelty) |
| `brandfonbrener2024loss` | year | `2025` (ml_estimation); `2024` (data) |
| `chen2024scaling` | year | `2025` (ml_observational); `2024` (novelty) |
| `choshen2024hitchhikers` | year | `2025` (ml_core, ml_estimation); `2024` (data, novelty) |
| `christensen1976economies` | number | `4, Part 1` (ml_estimation); `4` (io_ces_duality) |
| `demirer2020production` | year | `2020` (io_controlfn); `2026` (io_ces_duality) |
| `diamond1978measurement` | booktitle | `Production Economics: A Dual Approach to Theory and Applications, Vol.` (ml_estimation); `Production Economics: A Dual Approach to Theory and Applications` (io_ces_duality, io_controlfn, novelty) |
| `dominguezolmedo2025training` | year | `2025` (ml_observational); `2024` (novelty) |
| `epochai2026capabilities` | title | `Epoch Capabilities Index ({ECI})` (ml_observational); `Capabilities \& Benchmarking` (data) |
| `epochai2026capabilities` | year | `2025` (ml_observational); `2026` (data) |
| `epochai2026data` | title | `Data on {AI} Models` (data, ml_observational); `Data on Notable {AI} Models` (io_controlfn) |
| `epochai2026data` | year | `2025` (ml_observational); `2026` (data, io_controlfn) |
| `gadre2024language` | year | `2024` (data, io_controlfn, ml_core, ml_observational, novelty); `2025` (ml_estimation) |
| `griliches1998production` | year | `1998` (io_controlfn, ml_estimation); `1995` (novelty) |
| `griliches1998production` | booktitle | `Practicing Econometrics: Essays in Method and Application` (ml_estimation); `Econometrics and Economic Theory in the 20th Century: The Ragnar Frisc` (io_controlfn) |
| `griliches1998production` | pages | `383--415` (ml_estimation); `169--203` (io_controlfn) |
| `griliches1998production` | doi | `10.4337/9781035351442.00028` (ml_estimation); `10.1017/CCOL521633230.006` (io_controlfn); `10.3386/w5067` (novelty) |
| `griliches1998production` | publisher | `Edward Elgar` (ml_estimation); `Cambridge University Press` (io_controlfn) |
| `houthakker1955pareto` | pages | `27` (ml_core); `27--31` (econ_ai) |
| `huggingface2024open` | title | `Open {LLM} Leaderboard` (ml_observational); `Open {LLM} Leaderboard v2: contents` (data) |
| `huggingface2024open` | year | `2024` (ml_observational); `2025` (data) |
| `li2025misfitting` | title | `(Mis)Fitting Scaling Laws: A Survey of Scaling Law Fitting Techniques ` (ml_estimation); `(Mis)Fitting: A Survey of Scaling Laws` (novelty) |
| `maiapolo2024sloth` | year | `2025` (ml_estimation); `2024` (data, ml_observational) |
| `marschak1944random` | number | `3/4` (data, io_controlfn, ml_estimation, novelty); `3--4` (io_ces_duality) |
| `rosenfeld2020constructive` | year | `2020` (ml_core); `2019` (data) |
| `schaeffer2024predicting` | year | `2025` (ml_estimation); `2024` (ml_observational) |
| `snell2024scaling` | year | `2025` (ml_core); `2024` (data, econ_ai, io_ces_duality, novelty) |
| `villalobos2022run` | title | `Position: Will We Run out of Data? Limits of {LLM} Scaling Based on Hu` (ml_observational); `Will We Run Out of Data? Limits of {LLM} Scaling Based on Human-Genera` (data, econ_ai, novelty) |
| `villalobos2022run` | year | `2024` (ml_observational); `2022` (data, econ_ai, novelty) |
| `wu2024inference` | year | `2025` (ml_core); `2024` (data, econ_ai) |
| `xiao2025densing` | year | `2025` (ml_core, ml_observational, novelty); `2024` (data) |

Resolution notes: (i) venue years: the key keeps the first-appearance year where that was the majority key, while the `year` field gives the proceedings/journal year where a strand verified it (e.g., `snell2024scaling` is ICLR 2025; `gadre2024language` is recorded as ICLR 2025 but flagged UNVERIFIED by ml_core and data); (ii) `houthakker1955pareto` end page (27 vs 27--31) and year (1955 vs 1956) remain UNVERIFIED -- io_ces_duality says do not cite without checking; (iii) `li2025misfitting` has two titles (arXiv "(Mis)Fitting: A Survey of Scaling Laws"; ICLR 2025 "(Mis)Fitting Scaling Laws: A Survey of Scaling Law Fitting Techniques in Deep Learning"); (iv) `villalobos2022run` is typed as the ICML 2024 position paper (PMLR 235:49523--49544) with the 2022 arXiv key; (v) `christensen1976economies` issue recorded as "4, Part 1" by ml_estimation and "4" by io_ces_duality.
