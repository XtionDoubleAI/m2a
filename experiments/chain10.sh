#!/bin/bash
# chain10: after chain9's full A-Mem run, retry the Mem0 smoke. chain8's
# server failed to bind (previous chain's server still shutting down on
# 8000); this variant clears the port before starting.
set -u
while pgrep -f chain9.sh > /dev/null 2>&1; do sleep 120; done
cd "$(dirname "$0")/.." || exit 1
PY="$HOME/m2a-venv/bin/python"
ROOT=/mnt/e/hx/_models

# kill any server squatting on 8000 from a previous chain, then wait it out
pkill -f vllm_server_compat 2>/dev/null
for i in $(seq 1 30); do
  if ! ss -tln 2>/dev/null | grep -q ':8000 '; then break; fi
  sleep 5
done

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
if [ $? -ne 0 ]; then kill $SERVER_PID; echo "=== chain10 aborted ==="; exit 1; fi

rm -rf /tmp/m2a_mem0_faiss /tmp/m2a_mem0_history.db
"$PY" -X utf8 -m experiments.run_mem0 --limit 20 --name mem0-smoke > results/mem0-smoke.log 2>&1
RC=$?
tail -3 results/mem0-smoke.log | cut -c1-160
kill $SERVER_PID
echo "=== chain10 done (mem0 rc=$RC) ==="
