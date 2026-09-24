#!/usr/bin/env bash
# Test-compile ONE section in isolation (safe to run concurrently with other writers).
# Usage: code/paper/test_section.sh <section-name> [more section names...]
#   e.g. code/paper/test_section.sh wedge
# Builds paper/_build/<first-section>/test.pdf using the main preamble + references.bib.
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
P="$ROOT/paper"; NAME="$1"; B="$P/_build/$NAME"; mkdir -p "$B"
PRE=$(sed -n '1,/\\begin{document}/p' "$P/main.tex" | sed '$d')
{
  echo "$PRE" | sed "s#\\\\graphicspath{.*}#\\\\graphicspath{{$ROOT/output/figures/}{$P/figures/}}#"
  echo '\begin{document}'
  echo '\title{Section test}\author{Test}\maketitle'
  for s in "$@"; do echo "\\input{$P/sections/$s}"; done
  echo '\bibliographystyle{aea}\bibliography{references}'
  echo '\end{document}'
} > "$B/test.tex"
cp "$P/AEA.cls" "$P/aea.bst" "$P/references.bib" "$B/"
cd "$B" && ln -sfn "$P/tables" tables && ln -sfn "$P/sections" sections
"$ROOT/tools/tectonic" -X compile test.tex 2>&1 | grep -E "error|Error|Undefined|undefined|Missing|Warning: Citation" | head -40
echo "PDF: $B/test.pdf"
