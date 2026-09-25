#!/bin/bash
# V4-Pro key-config check: four systems + full supply, thinking engaged.
# Memory banks are reused from local caches; only the answer-side LLM is the API.
# Credentials live outside the repo in ~/.forge_api_env.
set -u
cd /mnt/e/hx/_DoctorXtion/intern_shxt/m2a
. ~/.forge_api_env
PY=$HOME/m2a-venv/bin/python
M=DeepSeek-V4-Pro
run() {
  name=$1; shift
  echo "=== $name start $(date +%H:%M:%S) ===" >> results/api_queue.log
  $PY -X utf8 -m "$@" --name "$name" >> results/api_queue.log 2>&1 \
    || echo "FAILED $name" >> results/api_queue_failures.txt
}
echo "=== ltm-v4p-on start $(date +%H:%M:%S) ===" >> results/api_queue.log
$PY -X utf8 -m experiments.run_baseline --llm api --host-model $M --reasoning high --out results/ltm-v4p-on.jsonl >> results/api_queue.log 2>&1 || echo "FAILED ltm-v4p-on" >> results/api_queue_failures.txt
run forge-v4p-on  experiments.run_forge        --llm api --host-model $M --reasoning high
run mem0-v4p-on   experiments.run_mem0_thin    --llm api --host-model $M --reasoning high
run amem-v4p-on   experiments.run_amem         --llm api --host-model $M --reasoning high
run full-v4p-on   experiments.run_full_supply  --llm api --host-model $M --reasoning high
echo "ALL DONE $(date +%H:%M:%S)" >> results/api_queue.log
