"""fetch_sources.py -- one-time download of the primary sources behind the family readings and the model-level serving
code (module rb2_decisions). Saves the raw file and a plain-text rendering under data/raw/rb2_decisions/, with a
manifest (URL, retrieval time, bytes, sha256). run.py never downloads; it checks every quoted passage in
data/processed/rb2_decisions/*.csv against these saved texts.

  .venv/bin/python code/analysis/rb2_decisions/fetch_sources.py [key ...]
"""
from __future__ import annotations

import datetime
import hashlib
import io
import json
import os
import re
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RAW = os.path.join(ROOT, "data", "raw", "rb2_decisions")
os.makedirs(RAW, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (research; scaling-laws-pf replication)"}  # review: no personal e-mail in request headers

from sources import SOURCES  # noqa: E402  key -> dict(url=..., kind=html|pdf|text)


def to_text(content: bytes, kind: str) -> str:
    if kind == "pdf":
        import pypdf
        r = pypdf.PdfReader(io.BytesIO(content))
        return "\n".join((p.extract_text() or "") for p in r.pages)
    if kind == "text":
        return content.decode("utf-8", "replace")
    from bs4 import BeautifulSoup
    s = BeautifulSoup(content, "html.parser")
    for t in s(["script", "style", "noscript"]):
        t.decompose()
    txt = s.get_text(" ")
    return re.sub(r"[ \t\r\f\v]+", " ", txt)


def fetch(key, spec, man):
    url = spec["url"]
    ext = {"pdf": "pdf", "html": "html", "text": "md"}[spec["kind"]]
    try:
        r = requests.get(url, headers=UA, timeout=90)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {key}: {e}")
        man[key] = dict(url=url, error=str(e), retrieved=datetime.datetime.now().isoformat(timespec="seconds"))
        return
    raw = os.path.join(RAW, f"{key}.{ext}")
    open(raw, "wb").write(r.content)
    txt = to_text(r.content, spec["kind"])
    open(os.path.join(RAW, f"{key}.txt"), "w").write(txt)
    man[key] = dict(url=url, final_url=r.url, retrieved=datetime.datetime.now().isoformat(timespec="seconds"),
                    bytes=len(r.content), sha256=hashlib.sha256(r.content).hexdigest(), chars=len(txt))
    print(f"ok   {key}: {len(r.content)} bytes, {len(txt)} chars")


def main():
    mf = os.path.join(RAW, "manifest.json")
    man = json.load(open(mf)) if os.path.exists(mf) else {}
    keys = sys.argv[1:] or list(SOURCES)
    for k in keys:
        if k in man and "error" not in man[k] and "--force" not in sys.argv:
            continue
        fetch(k, SOURCES[k], man)
        time.sleep(1.0)
        json.dump(man, open(mf, "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
