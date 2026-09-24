#!/bin/bash
# GPU line: retrieval-pool stress test on the FULL 2029-session benchmark
# (same 400 questions, 4.7x larger visible-session pool). Three systems.
set -u
cd /mnt/e/hx/_DoctorXtion/intern_shxt/m2a
PY=$HOME/m2a-venv/bin/python
B=/mnt/e/hx/_DoctorXtion/intern_shxt/Pdev/refs/repos/Mem2ActBench/Mem2ActBench
run() {
  name=$1; shift
  echo "=== $name start $(date +%H:%M:%S) ===" >> results/gpu_queue.log
  $PY -X utf8 -m "$@" --name "$name" >> results/gpu_queue.log 2>&1 \
    || echo "FAILED $name" >> results/gpu_queue_failures.txt
}
rm -f results/gpu_queue_failures.txt
run forge-pool  experiments.run_forge       --bench-dir $B --chunks-cache results/chunks_full
run ltm-pool    experiments.run_baseline    --bench-dir $B --out results/ltm-pool.jsonl
run full-pool   experiments.run_full_supply --bench-dir $B
echo "ALL DONE $(date +%H:%M:%S)" >> results/gpu_queue.log
