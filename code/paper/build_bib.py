"""Build paper/references.bib from lit/references.bib + lit/bib/extra_*.bib.

- deduplicates by key (first occurrence wins: the curated lit/references.bib)
- strips internal fields (note/annote/abstract/keywords/file) which carry verification notes
  such as 'UNVERIFIED: ...' and break aea.bst; the stripped notes are written to
  paper/bib_verification_notes.txt so nothing is lost
- escapes bare & and % in field values
"""
import glob
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
srcs = [os.path.join(ROOT, "lit", "references.bib")] + sorted(glob.glob(os.path.join(ROOT, "lit", "bib", "extra_*.bib")))
DROP = {"note", "annote", "abstract", "keywords", "file", "comment", "comments"}


def entries(text):
    i = 0
    while True:
        m = re.search(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text[i:])
        if not m:
            return
        start = i + m.start()
        j = i + m.end()
        depth = 1
        while depth and j < len(text):
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        yield m.group(1).lower(), m.group(2), text[start:j]
        i = j


def fields(body):
    """Split the inside of an entry into (name, value) pairs, respecting braces/quotes."""
    inner = body[body.index(",") + 1: body.rindex("}")]
    out, k = [], 0
    while k < len(inner):
        m = re.match(r"\s*,?\s*([A-Za-z_\-]+)\s*=\s*", inner[k:])
        if not m:
            break
        name = m.group(1).lower()
        k += m.end()
        if inner[k] == "{":
            depth, j = 0, k
            while True:
                depth += {"{": 1, "}": -1}.get(inner[j], 0)
                j += 1
                if depth == 0:
                    break
            val = inner[k:j]
        elif inner[k] == '"':
            j = inner.index('"', k + 1) + 1
            val = inner[k:j]
        else:
            m2 = re.match(r"[^,\n]+", inner[k:])
            j = k + m2.end()
            val = inner[k:j].strip()
        out.append((name, val))
        k = j
    return out


def esc(v):
    return re.sub(r"(?<!\\)([&%])", r"\\\1", v)


seen, out, notes = set(), [], []
for src in srcs:
    for typ, key, body in entries(open(src, encoding="utf-8").read()):
        if key in seen:
            continue
        seen.add(key)
        fs = fields(body)
        keep = []
        for n, v in fs:
            if n in DROP:
                notes.append(f"{key}\t{n}\t{v}")
                continue
            keep.append((n, esc(v) if n not in ("url", "doi") else v))
        out.append(f"@{typ}{{{key},\n" + ",\n".join(f"  {n} = {v}" for n, v in keep) + "\n}\n")
open(os.path.join(ROOT, "paper", "references.bib"), "w", encoding="utf-8").write("\n".join(out))
open(os.path.join(ROOT, "paper", "bib_verification_notes.txt"), "w", encoding="utf-8").write("\n".join(notes))
print(f"{len(out)} entries from {len(srcs)} files; {len(notes)} internal notes stripped")
