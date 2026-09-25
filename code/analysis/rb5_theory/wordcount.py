"""Word counter of paper/notes/cut_log_v3.md section 5, applied to paper sections.

Usage: python wordcount.py [file ...]
"""
import re
import sys


def count(s):
    s = re.sub(r"(?<!\\)%.*", "", s)
    s = s.replace("\\$", "DOLLAR")
    for env in ("figure", "table", "equation", "equation*", "align", "align*", "gather", "multline", "eqnarray"):
        s = re.sub(r"\\begin\{" + re.escape(env) + r"\}.*?\\end\{" + re.escape(env) + r"\}", " ", s, flags=re.S)
    s = re.sub(r"\\\[.*?\\\]", " ", s, flags=re.S)
    s = re.sub(r"\$\$.*?\$\$", " ", s, flags=re.S)
    s = re.sub(r"\$[^$]*\$", " MATH ", s)
    s = re.sub(r"\\(cite[a-z]*|ref|eqref|label|input|autoref)\*?(\[[^\]]*\])*\{[^}]*\}", " REF ", s)
    s = re.sub(r"\\(section|subsection|emph|textit|textbf|textcolor\{[^}]*\})\*?", " ", s)
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = re.sub(r"[{}~]", " ", s)
    return len([w for w in s.split() if re.search(r"[A-Za-z0-9]", w)])


if __name__ == "__main__":
    for f in sys.argv[1:]:
        print(f, count(open(f).read()))
