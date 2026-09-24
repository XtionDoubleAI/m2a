#!/bin/bash
# GPU line: C3 variant -- gold-value injection at three positions (first/middle/last).
set -u
cd /mnt/e/hx/_DoctorXtion/intern_shxt/m2a
PY=$HOME/m2a-venv/bin/python
for P in first middle last; do
  echo "=== inject-$P start $(date +%H:%M:%S) ===" >> results/gpu_queue.log
  $PY -X utf8 -m experiments.run_oracle_probe --inject-values $P --name inject-$P \
    >> results/gpu_queue.log 2>&1 || echo "FAILED inject-$P" >> results/gpu_queue_failures.txt
done
echo "ALL DONE $(date +%H:%M:%S)" >> results/gpu_queue.log
