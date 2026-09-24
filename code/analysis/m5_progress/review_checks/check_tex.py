"""Reviewer check of the m5 LaTeX fragments (no TeX compiler available): brace balance, begin/end balance,
math-mode balance per row, and the number of column separators per tabular row vs the column spec."""
import glob
import re

for f in sorted(glob.glob('/Users/yigitokar/scaling-laws-pf/output/tables/m5_progress_*.tex')):
    s = open(f).read()
    probs = []
    depth = 0
    for i, ch in enumerate(s):
        if ch == '{' and (i == 0 or s[i - 1] != '\\'):
            depth += 1
        elif ch == '}' and (i == 0 or s[i - 1] != '\\'):
            depth -= 1
            if depth < 0:
                probs.append(f'unbalanced }} at {i}')
                depth = 0
    if depth:
        probs.append(f'brace depth {depth} at end')
    for env in set(re.findall(r'\\begin\{(\w+)\}', s)):
        if s.count(f'\\begin{{{env}}}') != s.count(f'\\end{{{env}}}'):
            probs.append(f'env {env} unbalanced')
    spec = re.search(r'\\begin\{tabular\}\{([^}]*)\}', s).group(1)
    ncol = len(re.findall(r'[lcrp]', spec))
    body = s.split('\\midrule', 1)[1].split('\\bottomrule')[0]
    for ln in body.split('\n'):
        ln = ln.strip()
        if not ln.endswith('\\\\'):
            continue
        # count & outside math and not escaped; multicolumn spans
        amp = len(re.findall(r'(?<!\\)&', ln))
        span = sum(int(k) - 1 for k in re.findall(r'\\multicolumn\{(\d+)\}', ln))
        if amp + 1 + span != ncol:
            probs.append(f'row has {amp + 1 + span} cols (spec {ncol}): {ln[:80]}')
        if ln.count('$') % 2:
            probs.append(f'odd $ count: {ln[:80]}')
    print(f.split('/')[-1], 'OK' if not probs else probs)
