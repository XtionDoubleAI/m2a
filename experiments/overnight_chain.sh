#!/bin/bash
# Overnight experiment chain: D5a/D5b mechanism splits + D6 component ablations.
# Runs sequentially on one GPU; each run appends its manifest to runs.jsonl.
set -u
cd "$(dirname "$0")/.." || exit 1
PY="$HOME/m2a-venv/bin/python"

run() {  # name, extra args...
  local name="$1"; shift
  echo "=== $name ==="
  "$PY" -X utf8 -m experiments.run_forge "$@" --name "$name" > "results/${name}.log" 2>&1
  grep -ah final "results/${name}.log" | tail -1
}

run forge-chunks-nodemand --store-mode chunks --no-demands
run forge-chunks-flat     --store-mode chunks --flat-render
run forge-chunks-r1       --store-mode chunks --rerank
run forge-chunks-r1r2     --store-mode chunks --rerank --rerank-candidates
run forge-chunks-r1r2r3   --store-mode chunks --rerank --rerank-candidates --verbatim-threshold 0.5

echo "=== chain done ==="
