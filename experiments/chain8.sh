#!/bin/bash
# chain8: after chain7's A-Mem smoke, retry the Mem0 smoke (chain4 died on a
# missing faiss package; now installed and import-verified).
set -u
while pgrep -f chain7.sh > /dev/null 2>&1; do sleep 120; done
cd "$(dirname "$0")/.." || exit 1
PY="$HOME/m2a-venv/bin/python"
ROOT=/mnt/e/hx/_models

if [ -f "$ROOT/Qwen/Qwen2.5-7B-Instruct/config.json" ]; then
  MODEL_DIR="$ROOT/Qwen/Qwen2.5-7B-Instruct"
else
  MODEL_DIR=$(ls -d "$ROOT"/models/Qwen--Qwen2.5-7B-Instruct/snapshots/*/ | tail -1)
fi

CUDA_VISIBLE_DEVICES=0 nohup "$PY" -X utf8 experiments/vllm_server_compat.py \
  --model "$MODEL_DIR" --served-model-name Qwen/Qwen2.5-7B-Instruct \
  --port 8000 --gpu-memory-utilization 0.90 --max-model-len 16384 > results/vllm_server_mem0.log 2>&1 &
SERVER_PID=$!

"$PY" - <<'EOF'
import sys, time, urllib.request
for i in range(90):
    try:
        urllib.request.urlopen("http://localhost:8000/v1/models", timeout=3)
        print("server up after", i * 10, "s"); sys.exit(0)
    except Exception:
        time.sleep(10)
print("server did not come up"); sys.exit(1)
EOF
if [ $? -ne 0 ]; then kill $SERVER_PID; echo "=== chain8 aborted ==="; exit 1; fi

rm -rf /tmp/m2a_mem0_faiss /tmp/m2a_mem0_history.db
"$PY" -X utf8 -m experiments.run_mem0 --limit 20 --name mem0-smoke > results/mem0-smoke.log 2>&1
RC=$?
tail -3 results/mem0-smoke.log | cut -c1-160
kill $SERVER_PID
echo "=== chain8 done (mem0 rc=$RC) ==="
