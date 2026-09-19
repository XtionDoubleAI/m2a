#!/bin/bash
# chain4: after chain3 frees the GPUs, start a vLLM OpenAI server (GPU0) and
# smoke-test the Mem0 adapter on 20 tasks. Full run is launched manually after
# a human checks the smoke numbers.
set -u
while pgrep -f chain3.sh > /dev/null 2>&1; do sleep 120; done
cd "$(dirname "$0")/.." || exit 1
PY="$HOME/m2a-venv/bin/python"
ROOT=/mnt/e/hx/_models

# resolve model dir (direct or modelscope snapshot)
if [ -f "$ROOT/Qwen/Qwen2.5-7B-Instruct/config.json" ]; then
  MODEL_DIR="$ROOT/Qwen/Qwen2.5-7B-Instruct"
else
  MODEL_DIR=$(ls -d "$ROOT"/models/Qwen--Qwen2.5-7B-Instruct/snapshots/*/ | tail -1)
fi
echo "model dir: $MODEL_DIR"

CUDA_VISIBLE_DEVICES=0 nohup "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" --served-model-name Qwen/Qwen2.5-7B-Instruct \
  --port 8000 --gpu-memory-utilization 0.90 > results/vllm_server.log 2>&1 &
SERVER_PID=$!
echo "server pid $SERVER_PID"

# wait up to 15 min for the server to come up
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
if [ $? -ne 0 ]; then kill $SERVER_PID; echo "=== chain4 aborted ==="; exit 1; fi

rm -rf /tmp/m2a_mem0_faiss /tmp/m2a_mem0_history.db
"$PY" -X utf8 -m experiments.run_mem0 --limit 20 --name mem0-smoke > results/mem0-smoke.log 2>&1
kill $SERVER_PID
echo "=== chain4 done ==="
