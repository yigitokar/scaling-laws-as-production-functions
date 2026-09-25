#!/usr/bin/env python
"""Write or verify data/raw/MANIFEST.sha256, the checksum manifest of every raw download.

The raw downloads are not redistributed with the replication package (several sources carry no license that
permits redistribution). The manifest records, for every file under data/raw, its SHA-256, its size in bytes, its
retrieval time, its source URL and the terms recorded for the source, so that a later download can be checked
against the copy we used.

Usage (from anywhere):
    python code/tools/make_manifest.py              # (re)write data/raw/MANIFEST.sha256
    python code/tools/make_manifest.py --verify     # check data/raw against the manifest (exit 1 on a mismatch)
    python code/tools/make_manifest.py --verify farseer steplaw   # check only these subdirectories

Standard tools can check the hashes too (from data/raw):
    awk -F'\t' '!/^#/ {print $1"  "$4}' MANIFEST.sha256 | shasum -a 256 -c

Format: comment lines start with '#'; data lines are tab-separated with the columns
    sha256, bytes, retrieved_utc, path (relative to data/raw), source_url, kind, terms, note
kind is "fixed" (a file at a stable location), "living" (an API response, live database or web page: a new download
will differ, and the checksum identifies the copy we used) or "derived" (made locally from another raw file).
retrieved_utc is the file's modification time in UTC, which is its download time: downloads were written once and
not modified afterwards (for rb2_decisions it agrees with the local times in rb2_decisions/manifest.json). Files
extracted from an archive carry the archive's time.
The script only reads files; it writes nothing but the manifest.
"""
import argparse
import ast
import datetime as dt
import hashlib
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW = os.path.join(ROOT, "data", "raw")
MANIFEST = os.path.join(RAW, "MANIFEST.sha256")
SKIP_NAMES = {".DS_Store", "MANIFEST.sha256"}
CHUNK = 8 * 1024 * 1024

GH = "https://raw.githubusercontent.com"
HF = "https://huggingface.co"

NOLIC = "no license file; not redistributed"
TERMS = {  # as recorded in code/data/*.sh and lit/notes/data_sources.md
    "farseer": NOLIC,
    "steplaw": NOLIC,
    "epoch_chinchilla": NOLIC,
    "llama3_isoflop": NOLIC,
    "isoflop_experiments": "Apache-2.0",
    "porian": "MIT",
    "gadre": "MIT",
    "colpret": "MIT (aggregates third-party data)",
    "colpret_listing.txt": "GitHub API listing",
    "datablations": "Apache-2.0",
    "olmo_ladder": "Apache-2.0",
    "datadecide": "ODC-BY",
    "datadecide_eval": "ODC-BY",
    "pythia_evals": "Apache-2.0",
    "kricheli": "GPL-2.0",
    "epoch_models": "CC-BY-4.0 (Epoch AI)",
    "epoch_bench": "CC-BY-4.0 (Epoch AI)",
    "ho2024": "code MIT; data sheet without an explicit license",
    "obsscaling": "Apache-2.0",
    "sloth": "MIT",
    "openllm": "no license field",
    "openrouter": "OpenRouter API terms",
    "m3_openrouter": "OpenRouter API terms",
    "m3_hf": "Hugging Face API and model cards; cards not redistributed",
    "hf_meta": "Hugging Face API",
    "ra2_hf_tree": "Hugging Face API",
    "ra2_cards": "Hugging Face model cards; not redistributed",
    "m3_lmarena": "CC-BY-4.0",
    "fineweb": "ODC-BY",
    "fineweb_edu": "ODC-BY",
    "c4": "ODC-BY",
    "pg19": "Apache-2.0",
    "wikitext": "CC BY-SA",
    "ra2_papers": "arXiv; not redistributed",
    "ra3_econ_lit": "arXiv; not redistributed",
    "rb2_decisions": "publishers' terms; not redistributed",
    "rb3_econ2": "Epoch AI terms / DeepSeek web page; not redistributed",
    "rb4_chinflop": "arXiv source of Hoffmann et al. (2022); not redistributed",
}

# Files whose URL is written literally in a download script (or, where noted, in lit/notes/data_sources.md).
EXPLICIT = {
    "farseer/1222_full.csv": f"{GH}/Farseer-Scaling-Law/Farseer/main/ipynb/data/1222_full.csv",
    "steplaw/dense_lr_bs_loss.csv": f"{GH}/step-law/steplaw/main/data/dense_lr_bs_loss.csv",
    "porian/experiment_results.pickle.xz":
        f"{GH}/formll/resolving-scaling-law-discrepancies/main/data/experiment_results.pickle.xz",
    "datadecide/ppl_results.parquet":
        f"{HF}/datasets/allenai/DataDecide-ppl-results/resolve/main/data/train-00000-of-00001.parquet",
    "epoch_bench/benchmark_data.zip": "https://epoch.ai/data/benchmark_data.zip",
    "epoch_models/ml_hardware.csv": "https://epoch.ai/data/ml_hardware.csv",
    "ho2024/algorithmic_progress.csv": "https://docs.google.com/spreadsheets/d/"
        "11m8O_mU0cUkOB_5wluPne4PNsuvsKNbbVAzbYNy-NXY/export?format=csv&gid=2087221150",
    "sloth/data_v2.csv": f"{GH}/felipemaiapolo/sloth/main/data/data_v2.csv",
    "openllm/contents_v2.parquet":
        f"{HF}/api/datasets/open-llm-leaderboard/contents/parquet/default/train/0.parquet",
    "openrouter/models_2026-09-23.json": "https://openrouter.ai/api/v1/models",
    "isoflop_experiments/isoflop_experiments.csv":
        f"{HF}/datasets/open-athena/isoflop-experiments/resolve/main/isoflop_experiments.csv",
    "isoflop_experiments/README.md": f"{HF}/datasets/open-athena/isoflop-experiments/resolve/main/README.md",
    "epoch_chinchilla/data_analysis.ipynb": f"{GH}/epoch-research/analyzing-chinchilla/main/data_analysis.ipynb",
    "m3_lmarena/text_full.parquet":
        f"{HF}/datasets/lmarena-ai/leaderboard-dataset/resolve/refs%2Fconvert%2Fparquet/text/full/0000.parquet",
    "gadre/eval_metadata.csv": f"{GH}/mlfoundations/scaling/main/exp_data/eval_metadata.csv",
    "kricheli/README.md": f"{HF}/datasets/TPPIsCriticalFor/colinear_scaling_models/resolve/main/README.md",
    "kricheli/full_coverage_all_seeds.parquet": f"{HF}/datasets/TPPIsCriticalFor/colinear_scaling_models/"
        "resolve/main/full_coverage_results/full_coverage_all_seeds.parquet",
    "datablations/return_alloc.ipynb": f"{GH}/huggingface/datablations/main/plotstables/return_alloc.ipynb",
    "datablations/contours.ipynb": f"{GH}/huggingface/datablations/main/plotstables/contours.ipynb",
    "rb3_econ2/epoch_all_systems_20240531.csv":
        "http://web.archive.org/web/20240531034617id_/https://epochai.org/data/epochdb/all_systems.csv",
    "rb3_econ2/epoch_notable_ai_models_20240620.csv":
        "http://web.archive.org/web/20240620060236id_/https://epochai.org/data/epochdb/notable_ai_models.csv",
    "rb3_econ2/epoch_large_scale_ai_models_20240619.csv":
        "http://web.archive.org/web/20240619212353id_/https://epochai.org/data/epochdb/large_scale_ai_models.csv",
    "rb3_econ2/deepseek_pricing_20260924.html": "https://api-docs.deepseek.com/quick_start/pricing",
    "rb4_chinflop/arxiv_2203.15556_src.tar.gz": "https://arxiv.org/e-print/2203.15556",
    # Not fetched by a script in code/data: URL from lit/notes/data_sources.md (A01, B01).
    "epoch_chinchilla/svg_extracted_data.csv":
        f"{GH}/epoch-research/analyzing-chinchilla/main/data/svg_extracted_data.csv",
    "epoch_models/all_ai_models.csv": "https://epoch.ai/data/all_ai_models.csv",
    "epoch_models/notable_ai_models.csv": "https://epoch.ai/data/notable_ai_models.csv",
    "epoch_models/large_scale_ai_models.csv": "https://epoch.ai/data/large_scale_ai_models.csv",
    "epoch_models/frontier_ai_models.csv": "https://epoch.ai/data/frontier_ai_models.csv",
    # Not fetched by a script in code/data: identified by matching size and SHA-256 with the Hugging Face LFS record.
    "c4/c4-validation.00000-of-00008.json.gz":
        f"{HF}/datasets/allenai/c4/resolve/main/en/c4-validation.00000-of-00008.json.gz",
    "pg19/test.parquet": f"{HF}/datasets/emozilla/pg19/resolve/main/data/test-00000-of-00001-29a571947c0b5ccc.parquet",
    "wikitext/test.parquet":
        f"{HF}/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/test-00000-of-00001.parquet",
    "wikitext/validation.parquet":
        f"{HF}/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/validation-00000-of-00001.parquet",
}
NOT_IN_SCRIPTS = {  # sources that no script in code/data downloads (the note says how the URL was established)
    "epoch_chinchilla/svg_extracted_data.csv": "URL from lit/notes/data_sources.md; no download script fetches it",
    "epoch_models/all_ai_models.csv": "URL from lit/notes/data_sources.md; no download script fetches it",
    "epoch_models/notable_ai_models.csv": "URL from lit/notes/data_sources.md; no download script fetches it",
    "epoch_models/large_scale_ai_models.csv": "URL from lit/notes/data_sources.md; no download script fetches it",
    "epoch_models/frontier_ai_models.csv": "URL from lit/notes/data_sources.md; no download script fetches it",
    "c4/c4-validation.00000-of-00008.json.gz": "matched to the source by size and Hugging Face LFS SHA-256",
    "pg19/test.parquet": "matched to the source by size and Hugging Face LFS SHA-256 "
                         "(the same file is in emozilla/pg19-test)",
    "wikitext/test.parquet": "matched to the source by size and Hugging Face LFS SHA-256",
    "wikitext/validation.parquet": "matched to the source by size and Hugging Face LFS SHA-256",
}
LIVING_DIRS = {"epoch_models", "epoch_bench", "openrouter", "m3_openrouter", "m3_hf", "hf_meta", "ra2_hf_tree",
               "openllm", "m3_lmarena", "rb2_decisions"}
LIVING_FILES = {"ho2024/algorithmic_progress.csv", "rb3_econ2/deepseek_pricing_20260924.html"}


def load_config_mirror():
    """CONFIG_MIRROR of code/analysis/m3_wedge/curated.py, read without importing the module."""
    path = os.path.join(ROOT, "code", "analysis", "m3_wedge", "curated.py")
    try:
        tree = ast.parse(open(path).read())
    except OSError:
        return {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "CONFIG_MIRROR" for t in node.targets):
            return ast.literal_eval(node.value)
    return {}


def known_hf_ids():
    """Hugging Face ids whose names are recoverable without ambiguity (files named with '__' for '/')."""
    ids = set()
    for sub in ("m3_hf/api", "m3_hf/configs", "ra2_hf_tree", "ra2_cards"):
        d = os.path.join(RAW, sub)
        if os.path.isdir(d):
            for f in os.listdir(d):
                stem = os.path.splitext(f)[0]
                if "__" in stem:
                    ids.add(stem.replace("__", "/", 1))
    p = os.path.join(ROOT, "data", "processed", "m4_observational", "hf_ids.txt")
    if os.path.exists(p):
        ids.update(line.strip() for line in open(p) if line.strip())
    return ids


def id_from_single_underscore(stem, ids):
    """Invert id.replace('/', '_') (m3_hf/cards) or tr '/' '_' (hf_meta) by matching the known ids."""
    for i in ids:
        if i.replace("/", "_") == stem:
            return i, ""
    return stem.replace("_", "/", 1), "id reconstructed from the file name"


def rb2_manifest():
    p = os.path.join(RAW, "rb2_decisions", "manifest.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def source_of(rel, ctx):
    """Return (url, kind, note) for a path relative to data/raw."""
    top = rel.split("/", 1)[0]
    name = os.path.basename(rel)
    stem, ext = os.path.splitext(name)
    kind = "living" if (top in LIVING_DIRS or rel in LIVING_FILES) else "fixed"
    note = NOT_IN_SCRIPTS.get(rel, "")
    if rel in EXPLICIT:
        return EXPLICIT[rel], kind, note
    m = re.match(r"^(fineweb|fineweb_edu)/(\d{3}_\d{5}\.parquet)$", rel)
    if m:
        repo = "HuggingFaceFW/fineweb" if m.group(1) == "fineweb" else "HuggingFaceFW/fineweb-edu"
        return (f"{HF}/datasets/{repo}/resolve/main/sample/10BT/{m.group(2)}", "fixed",
                "sample-10BT shard; fetched outside code/data; matched by size and Hugging Face LFS SHA-256")
    if top == "farseer":
        return f"{GH}/Farseer-Scaling-Law/Farseer/main/ipynb/data/{name}", kind, note
    if top == "steplaw":
        return f"{GH}/step-law/steplaw/main/data/{name}", kind, note
    if rel.startswith("gadre/models/"):
        return f"{GH}/mlfoundations/scaling/main/exp_data/models/{name}", kind, note
    if rel.startswith("gadre/evals/"):
        return f"{GH}/mlfoundations/scaling/main/exp_data/evals/{name}", kind, note
    if rel == "gadre/exp_data_listing.txt":
        return "https://api.github.com/repos/mlfoundations/scaling/contents/exp_data", "derived", \
            "GitHub API listing written by download_public.sh"
    if rel == "porian/data_listing.txt":
        return "https://api.github.com/repos/formll/resolving-scaling-law-discrepancies/contents/data", \
            "derived", "GitHub API listing written by download_public.sh"
    if rel == "colpret_listing.txt":
        return "https://api.github.com/repos/IBM/ColPret/contents/aggregated_eval", "derived", \
            "GitHub API listing written by download_public.sh"
    if top == "colpret":
        return f"{GH}/IBM/ColPret/main/aggregated_eval/{name}", kind, note
    if top == "olmo_ladder":
        return f"{GH}/allenai/OLMo-ladder/main/src/scripts/paper/data/ladder-runs/{name}", kind, note
    if top == "obsscaling":
        return f"{GH}/ryoungj/ObsScaling/main/eval_results/{name}", kind, note
    if top == "sloth":
        return f"{GH}/felipemaiapolo/sloth/main/data/{name}", kind, note
    if top == "kricheli" and ext == ".csv":
        return f"{HF}/datasets/TPPIsCriticalFor/colinear_scaling_models/resolve/main/extracted_losses/{name}", \
            kind, note
    if top == "datadecide_eval":
        return f"{HF}/datasets/allenai/DataDecide-eval-results/resolve/main/data/{name}", kind, note
    if top == "pythia_evals":
        return f"{GH}/EleutherAI/pythia/main/evals/pythia-v1/{rel.split('/', 1)[1]}", kind, note
    if top == "llama3_isoflop":
        return f"{GH}/eric-czech/llama3_isoflop_extraction/main/{name}", kind, note
    if rel.startswith("ho2024/code/"):
        sha = "29c7d852b6cd19ae11bb6662e9c1e61c03b1e4a7"
        return f"{GH}/epoch-research/lm-algorithmic-progress/{sha}/{name.replace('&', '%26')}", kind, note
    if rel.startswith("m3_hf/api/"):
        hid = stem.replace("__", "/", 1)
        return (f"{HF}/api/models/{hid}?expand[]=downloads&expand[]=downloadsAllTime&expand[]=safetensors"
                "&expand[]=createdAt&expand[]=likes"), kind, note
    if rel.startswith("m3_hf/configs/"):
        hid = stem.replace("__", "/", 1)
        src = ctx["mirror"].get(hid, hid)
        extra = f"config of {hid} read from the mirror {src}" if src != hid else ""
        return f"{HF}/{src}/resolve/main/config.json", kind, extra
    if rel.startswith("m3_hf/cards/"):
        hid, extra = id_from_single_underscore(stem, ctx["ids"])
        return f"{HF}/{hid}/resolve/main/README.md", kind, extra
    if top == "hf_meta":
        hid, extra = id_from_single_underscore(stem, ctx["ids"])
        return (f"{HF}/api/models/{hid}?expand[]=createdAt&expand[]=safetensors&expand[]=downloadsAllTime",
                kind, extra)
    if top == "ra2_hf_tree":
        hid = stem.replace("__", "/", 1)
        return f"{HF}/api/models/{hid}?expand[]=childrenModelCount", kind, note
    if top == "ra2_cards":
        hid = stem.replace("__", "/", 1)
        return f"{HF}/{hid}/resolve/main/README.md", kind, note
    if rel.startswith("m3_openrouter/endpoints/"):
        slug = stem.replace("__", "/", 1)
        return f"https://openrouter.ai/api/v1/models/{slug}/endpoints", kind, note
    if top == "ra2_papers":
        if ext == ".txt":
            return f"https://arxiv.org/pdf/{stem}", "derived", f"text extracted locally from {stem}.pdf (PyMuPDF)"
        return f"https://arxiv.org/pdf/{stem}", kind, note
    if top == "ra3_econ_lit":
        m = re.search(r"arxiv(\d{4}\.\d{4,5})", name)
        return (f"https://arxiv.org/pdf/{m.group(1)}" if m else ""), kind, note
    if top == "rb2_decisions":
        if name == "manifest.json":
            return "", "derived", "retrieval record written by code/analysis/rb2_decisions/fetch_sources.py"
        rec = ctx["rb2"].get(stem, {})
        url = rec.get("final_url") or rec.get("url", "")
        if ext == ".txt":
            return url, "derived", "text extracted locally from the retrieved page or PDF"
        return url, kind, "fetched by code/analysis/rb2_decisions/fetch_sources.py"
    if rel.startswith("rb4_chinflop/src/"):
        return "https://arxiv.org/e-print/2203.15556", "derived", "extracted locally from arxiv_2203.15556_src.tar.gz"
    return "", kind, "source not recorded"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def utc(ts):
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def walk(subdirs=None):
    roots = [os.path.join(RAW, s) for s in subdirs] if subdirs else [RAW]
    out = []
    for r in roots:
        if os.path.isfile(r):
            out.append(os.path.relpath(r, RAW))
            continue
        for dp, dn, fn in os.walk(r):
            dn.sort()
            for f in sorted(fn):
                if f in SKIP_NAMES:
                    continue
                out.append(os.path.relpath(os.path.join(dp, f), RAW))
    return sorted(out)


def retrieved(rel):
    if rel.startswith("rb4_chinflop/src/"):
        return utc(os.path.getmtime(os.path.join(RAW, "rb4_chinflop", "arxiv_2203.15556_src.tar.gz")))
    return utc(os.path.getmtime(os.path.join(RAW, rel)))


def terms_of(rel):
    top = rel.split("/", 1)[0]
    if rel.startswith("ho2024/code/"):
        return "MIT"
    return TERMS.get(rel, TERMS.get(top, ""))


def write():
    ctx = {"mirror": load_config_mirror(), "ids": known_hf_ids(), "rb2": rb2_manifest()}
    rows = []
    for rel in walk():
        p = os.path.join(RAW, rel)
        url, kind, note = source_of(rel, ctx)
        rows.append([sha256_of(p), str(os.path.getsize(p)), retrieved(rel), rel, url, kind, terms_of(rel),
                     note])
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    total = sum(int(r[1]) for r in rows)
    head = [
        "# MANIFEST.sha256: SHA-256 checksums of the raw downloads under data/raw (not redistributed).",
        f"# Written {now} by code/tools/make_manifest.py; {len(rows)} files, {total} bytes.",
        "# Verify: python code/tools/make_manifest.py --verify   (or, from data/raw:",
        "#   awk -F'\\t' '!/^#/ {print $1\"  \"$4}' MANIFEST.sha256 | shasum -a 256 -c )",
        "# kind: fixed = file at a stable location; living = API response, live database or web page (a new download",
        "#   will differ; the checksum identifies the copy used); derived = made locally from another raw file.",
        "# retrieved_utc: file modification time in UTC (= download time); extracted files carry the archive's time.",
        "# columns (tab-separated):",
        "# sha256\tbytes\tretrieved_utc\tpath\tsource_url\tkind\tterms\tnote",
    ]
    with open(MANIFEST, "w") as fh:
        fh.write("\n".join(head) + "\n")
        for r in rows:
            fh.write("\t".join(x.replace("\t", " ") for x in r) + "\n")
    unknown = [r[3] for r in rows if not r[4]]
    print(f"wrote {MANIFEST}: {len(rows)} files, {total / 1e9:.2f} GB; without a source URL: {len(unknown)}")
    for u in unknown:
        print("  no source URL:", u)


def read_manifest():
    rows = {}
    for line in open(MANIFEST):
        if line.startswith("#") or not line.strip():
            continue
        f = line.rstrip("\n").split("\t")
        rows[f[3]] = {"sha256": f[0], "bytes": int(f[1]), "kind": f[5] if len(f) > 5 else ""}
    return rows


def verify(subdirs):
    if not os.path.exists(MANIFEST):
        sys.exit(f"no manifest at {MANIFEST}")
    man = read_manifest()
    present = walk(subdirs) if subdirs else walk()
    scope = (lambda r: any(r == s or r.startswith(s.rstrip("/") + "/") for s in subdirs)) if subdirs else (lambda r: True)
    bad, living_diff, ok = [], [], 0
    for rel in present:
        rec = man.get(rel)
        if rec is None:
            print("NOT IN MANIFEST ", rel)
            continue
        p = os.path.join(RAW, rel)
        same = os.path.getsize(p) == rec["bytes"] and sha256_of(p) == rec["sha256"]
        if same:
            ok += 1
        elif rec["kind"] == "living":
            living_diff.append(rel)
            print("DIFFERS (living)", rel)
        else:
            bad.append(rel)
            print("MISMATCH        ", rel)
    missing = [r for r in man if scope(r) and not os.path.exists(os.path.join(RAW, r))]
    for r in missing:
        print("MISSING         ", r)
    print(f"{ok} identical, {len(bad)} mismatched, {len(living_diff)} living sources changed, {len(missing)} missing")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verify", nargs="*", metavar="SUBDIR",
                    help="verify data/raw (or the given subdirectories) against the manifest")
    a = ap.parse_args()
    if a.verify is not None:
        verify(a.verify)
    else:
        write()
