#!/bin/bash
# V4-Flash full matrix: six systems x thinking on/off, sequential.
# Memory banks are reused from local caches; only the answer-side LLM is the API.
set -u
cd /mnt/e/hx/_DoctorXtion/intern_shxt/m2a
export FORGE_API_BASE=https://llmapi.paratera.com/v1
export FORGE_API_KEY=sk-FySO3P462AQpLM01XYvr3w
PY=$HOME/m2a-venv/bin/python
M=DeepSeek-V4-Flash
run() {
  name=$1; shift
  echo "=== $name start $(date +%H:%M:%S) ===" >> results/api_queue.log
  $PY -X utf8 -m "$@" --name "$name" >> results/api_queue.log 2>&1 \
    || echo "FAILED $name" >> results/api_queue_failures.txt
}
rm -f results/api_queue_failures.txt

for R in none high; do
  S=$([ "$R" = none ] && echo off || echo on)
  $PY -X utf8 -m experiments.run_baseline --llm api --host-model $M --reasoning $R --out results/ltm-v4f-$S.jsonl >> results/api_queue.log 2>&1 || echo "FAILED ltm-v4f-$S" >> results/api_queue_failures.txt
  run forge-v4f-$S  experiments.run_forge        --llm api --host-model $M --reasoning $R
  run mem0-v4f-$S   experiments.run_mem0_thin    --llm api --host-model $M --reasoning $R
  run amem-v4f-$S   experiments.run_amem         --llm api --host-model $M --reasoning $R
  run full-v4f-$S   experiments.run_full_supply  --llm api --host-model $M --reasoning $R
  run oracle-v4f-$S experiments.run_oracle_probe --llm api --host-model $M --reasoning $R
done
echo "ALL DONE $(date +%H:%M:%S)" >> results/api_queue.log
