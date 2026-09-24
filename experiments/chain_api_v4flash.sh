#!/bin/bash
# V4-Flash answer-side queue, remaining configs (thinking-off runs are done).
# Memory banks are reused from local caches; only the answer-side LLM is the API.
# Credentials live outside the repo in ~/.forge_api_env (base URL + key).
set -u
cd /mnt/e/hx/_DoctorXtion/intern_shxt/m2a
. ~/.forge_api_env
PY=$HOME/m2a-venv/bin/python
M=DeepSeek-V4-Flash
run() {
  name=$1; shift
  echo "=== $name start $(date +%H:%M:%S) ===" >> results/api_queue.log
  $PY -X utf8 -m "$@" --name "$name" >> results/api_queue.log 2>&1 \
    || echo "FAILED $name" >> results/api_queue_failures.txt
}
R=high
run forge-v4f-on  experiments.run_forge        --llm api --host-model $M --reasoning $R
run mem0-v4f-on   experiments.run_mem0_thin    --llm api --host-model $M --reasoning $R
run amem-v4f-on   experiments.run_amem         --llm api --host-model $M --reasoning $R
run full-v4f-on   experiments.run_full_supply  --llm api --host-model $M --reasoning $R
run oracle-v4f-on experiments.run_oracle_probe --llm api --host-model $M --reasoning $R
echo "=== ltm-v4f-on start $(date +%H:%M:%S) ===" >> results/api_queue.log
$PY -X utf8 -m experiments.run_baseline --llm api --host-model $M --reasoning high --out results/ltm-v4f-on.jsonl >> results/api_queue.log 2>&1 || echo "FAILED ltm-v4f-on" >> results/api_queue_failures.txt
echo "=== ltm-v4f-off start $(date +%H:%M:%S) ===" >> results/api_queue.log
$PY -X utf8 -m experiments.run_baseline --llm api --host-model $M --reasoning none --out results/ltm-v4f-off.jsonl >> results/api_queue.log 2>&1 || echo "FAILED ltm-v4f-off" >> results/api_queue_failures.txt
echo "ALL DONE $(date +%H:%M:%S)" >> results/api_queue.log
