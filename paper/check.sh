#!/usr/bin/env bash
# Paper verifier: numbers regenerated from results/, clean compile, every reference accounted for.
set -u
cd "$(dirname "$0")" || exit 1
rc=0
python3 ../scripts/paper_numbers.py || rc=1
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex > build.log 2>&1 || { echo "compile FAILED (see paper/build.log)"; rc=1; }
w=$(grep -E "Warning|Overfull|Underfull|undefined" main.log | grep -v "Font shape" | wc -l)
[ "$w" -eq 0 ] || { echo "LaTeX warnings: $w"; grep -E "Warning|Overfull|undefined" main.log | head; rc=1; }
grep -q "^Warning--" main.blg && { echo "BibTeX warnings"; grep "^Warning--" main.blg | head; rc=1; }
# every cited key exists, and every entry has a DOI unless it is one of the individually checked DOI-less sources
python3 - <<'PY' || rc=1
import re, sys
bib = open("refs.bib").read(); tex = open("main.tex").read()
entries = {m.group(1): m.group(2) for m in re.finditer(r"@\w+\{(\w+),(.*?)(?=\n@|\Z)", bib, re.S)}
cited = {k.strip() for g in re.findall(r"\\cite\{([^}]*)\}", tex) for k in g.split(",")}
no_doi_ok = {"ke2017", "platt1999", "jplsbdb", "jplhorizons", "standish"}
missing = cited - set(entries); unused = set(entries) - cited
nodoi = [k for k, v in entries.items() if "doi" not in v.lower() and k not in no_doi_ok]
print(f"references: {len(entries)} in refs.bib, {len(cited)} cited; missing {sorted(missing)}; unused {sorted(unused)}; without DOI {nodoi}")
sys.exit(1 if missing or nodoi else 0)
PY
pages=$(grep -o "Output written on main.pdf ([0-9]* pages" main.log | grep -o "[0-9]* pages")
echo "== PAPER $( [ $rc -eq 0 ] && echo PASS || echo FAIL ) ($pages)"
exit $rc
