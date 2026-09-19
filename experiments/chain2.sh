#!/bin/bash
# chain2: after the overnight chain finishes, rerun the full (no-component)
# configuration with R5 evidence archival so its multi-metric row is
# computable from the archive (the earlier forge-chunks-only run predates
# evidence dumping -- its error structure cannot be reconstructed offline).
set -u
while pgrep -f overnight_chain.sh > /dev/null 2>&1; do sleep 120; done
cd "$(dirname "$0")/.." || exit 1
PY="$HOME/m2a-venv/bin/python"
"$PY" -X utf8 -m experiments.run_forge --name forge-chunks-full > results/forge-chunks-full.log 2>&1
grep -ah final results/forge-chunks-full.log | tail -1
echo "=== chain2 done ==="
