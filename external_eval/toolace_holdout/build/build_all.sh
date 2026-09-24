#!/bin/bash
# toolace_holdout full build: upstream five-step pipeline on the 600 held-out
# sources, then the seeded spot-check. One API model throughout
# (DeepSeek-V4-Flash); credentials come from ~/.forge_api_env.
set -u
cd "$(dirname "$0")"
. ~/.forge_api_env
# WSL reaches HuggingFace through the host proxy (model downloads in step 02);
# models are pre-cached, so hub checks stay offline
export https_proxy=http://172.20.96.1:7897 http_proxy=http://172.20.96.1:7897
export HF_HUB_OFFLINE=1
PY=/root/bench-venv/bin/python
UP=/mnt/e/hx/_DoctorXtion/intern_shxt/Pdev/refs/repos/Mem2ActBench
M=DeepSeek-V4-Flash
LOG=build.log
say() { echo "=== $1 $(date +%H:%M:%S) ===" | tee -a "$LOG"; }

say "step 01 conversation refine"
$PY run_01.py --source ../toolace_heldout_source.jsonl \
    --output 01_conversations.jsonl --model $M --workers 4 >> "$LOG" 2>&1 \
    || { echo "FAILED 01" >> build_failures.txt; exit 1; }

say "step 02 facts and conflicts"
( cd "$UP" && $PY -X utf8 02_extract_facts_conflicts.py \
    --input-files /mnt/e/hx/_DoctorXtion/intern_shxt/m2a/external_eval/toolace_holdout/build/01_conversations.jsonl \
    --output-dir /mnt/e/hx/_DoctorXtion/intern_shxt/m2a/external_eval/toolace_holdout/build/processed_data \
    --model $M --api-base-url "$FORGE_API_BASE" --api-key "$FORGE_API_KEY" \
    --max-workers 4 ) >> "$LOG" 2>&1 \
    || { echo "FAILED 02" >> build_failures.txt; exit 1; }

say "step 03 interleave"
$PY 03_conv_sort.py >> "$LOG" 2>&1 \
    || { echo "FAILED 03" >> build_failures.txt; exit 1; }

say "step 04 QA construction"
QA_OUT=$PWD/processed_data/qa_raw.jsonl \
$PY -X utf8 04_qa_construction.py \
    --memory-path processed_data/conflict_evolutions.csv \
    --tool-pool-path "$UP/tool_api_pool.json" \
    --llm-model $M --max-workers 4 >> "$LOG" 2>&1 \
    || { echo "FAILED 04" >> build_failures.txt; exit 1; }

say "step 05 normalize and freeze"
$PY run_05.py --qa processed_data/qa_raw.jsonl >> "$LOG" 2>&1 \
    || { echo "FAILED 05" >> build_failures.txt; exit 1; }

say "step 06 spot check"
$PY spot_check.py --n 25 >> "$LOG" 2>&1 \
    || { echo "FAILED 06" >> build_failures.txt; exit 1; }

say "ALL DONE"
