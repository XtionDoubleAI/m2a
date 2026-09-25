#!/bin/bash
# Held-out evaluation on the frozen toolace_holdout set (89 QA / 41 sessions):
# six systems x two answer models (local 7B, V4-Flash thinking off).
# All memory banks are rebuilt for the 41 held-out sessions under fresh cache
# paths -- nothing from the main-set banks is reused. Requires the local vLLM
# server on port 8000 for write-side construction and local answering.
set -u
cd /mnt/e/hx/_DoctorXtion/intern_shxt/m2a
. ~/.forge_api_env
PY=$HOME/m2a-venv/bin/python
H=/mnt/e/hx/_DoctorXtion/intern_shxt/m2a/external_eval/toolace_holdout
LOG=results/holdout_queue.log
run() {
  name=$1; shift
  echo "=== $name start $(date +%H:%M:%S) ===" >> "$LOG"
  $PY -X utf8 -m "$@" --name "$name" >> "$LOG" 2>&1 \
    || echo "FAILED $name" >> results/holdout_queue_failures.txt
}

rm -f results/holdout_queue_failures.txt

# --- local 7B, six systems ---
$PY -X utf8 -m experiments.run_baseline --bench-dir "$H" --out results/ltm-holdout.jsonl >> "$LOG" 2>&1 || echo "FAILED ltm-holdout" >> results/holdout_queue_failures.txt
run forge-holdout  experiments.run_forge        --bench-dir "$H" --chunks-cache results/chunks-holdout
run mem0-holdout   experiments.run_mem0_thin    --bench-dir "$H" --bank-cache results/mem0_facts_holdout.jsonl
run amem-holdout   experiments.run_amem         --bench-dir "$H" --bank-cache results/amem_bank_holdout.jsonl
run full-holdout   experiments.run_full_supply  --bench-dir "$H"
run oracle-holdout experiments.run_oracle_probe --bench-dir "$H" --llm local

# --- V4-Flash thinking off, six systems (banks reused from the runs above) ---
run forge-holdout-v4f  experiments.run_forge        --bench-dir "$H" --chunks-cache results/chunks-holdout --llm api --host-model DeepSeek-V4-Flash --reasoning none
run mem0-holdout-v4f   experiments.run_mem0_thin    --bench-dir "$H" --bank-cache results/mem0_facts_holdout.jsonl --llm api --host-model DeepSeek-V4-Flash --reasoning none
run amem-holdout-v4f   experiments.run_amem         --bench-dir "$H" --bank-cache results/amem_bank_holdout.jsonl --llm api --host-model DeepSeek-V4-Flash --reasoning none
run full-holdout-v4f   experiments.run_full_supply  --bench-dir "$H" --llm api --host-model DeepSeek-V4-Flash --reasoning none
run oracle-holdout-v4f experiments.run_oracle_probe --bench-dir "$H" --llm api --host-model DeepSeek-V4-Flash --reasoning none
echo "=== ltm-holdout-v4f start $(date +%H:%M:%S) ===" >> "$LOG"
$PY -X utf8 -m experiments.run_baseline --llm api --host-model DeepSeek-V4-Flash --reasoning none --bench-dir "$H" --out results/ltm-holdout-v4f.jsonl >> "$LOG" 2>&1 || echo "FAILED ltm-holdout-v4f" >> results/holdout_queue_failures.txt

echo "ALL DONE $(date +%H:%M:%S)" >> "$LOG"
