#!/bin/bash
# chain3: after chain2 finishes, the verbatim-anchor threshold sweep row.
# R3 at threshold 0.5 fired on only 2/400 tasks (anchors extracted on 12.75%)
# -- the null result is "never played", not "played and lost". A low
# threshold (0.1) makes anchor extraction near-universal; if it still shows
# no gain the component is judged dead on merit.
set -u
while pgrep -f chain2.sh > /dev/null 2>&1; do sleep 120; done
cd "$(dirname "$0")/.." || exit 1
PY="$HOME/m2a-venv/bin/python"
"$PY" -X utf8 -m experiments.run_forge --store-mode chunks --rerank --rerank-candidates \
  --verbatim-threshold 0.1 --name forge-chunks-r1r2r3-low > results/forge-chunks-r1r2r3-low.log 2>&1
grep -ah final results/forge-chunks-r1r2r3-low.log | tail -1
echo "=== chain3 done ==="
