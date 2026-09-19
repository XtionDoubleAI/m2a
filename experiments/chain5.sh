#!/bin/bash
# chain5: after chain4's Mem0 smoke, smoke the A-Mem thin reimplementation on
# 20 tasks with its own server instance (chain4 shuts its server down).
set -u
while pgrep -f chain4.sh > /dev/null 2>&1; do sleep 120; done
cd "$(dirname "$0")/.." || exit 1
PY="$HOME/m2a-venv/bin/python"
ROOT=/mnt/e/hx/_models

if [ -f "$ROOT/Qwen/Qwen2.5-7B-Instruct/config.json" ]; then
  MODEL_DIR="$ROOT/Qwen/Qwen2.5-7B-Instruct"
else
  MODEL_DIR=$(ls -d "$ROOT"/models/Qwen--Qwen2.5-7B-Instruct/snapshots/*/ | tail -1)
fi

CUDA_VISIBLE_DEVICES=0 nohup "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" --served-model-name Qwen/Qwen2.5-7B-Instruct \
  --port 8000 --gpu-memory-utilization 0.90 > results/vllm_server_amem.log 2>&1 &
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
if [ $? -ne 0 ]; then kill $SERVER_PID; echo "=== chain5 aborted ==="; exit 1; fi

"$PY" -X utf8 -m experiments.run_amem --limit 20 --name amem-smoke > results/amem-smoke.log 2>&1
kill $SERVER_PID
echo "=== chain5 done ==="
