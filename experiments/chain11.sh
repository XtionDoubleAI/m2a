#!/bin/bash
# chain11: shared-server combined run -- full A-Mem (numpy-int link fix) then
# Mem0 smoke (pyairports now installed into the venv). One server for both.
set -u
cd "$(dirname "$0")/.." || exit 1
PY="$HOME/m2a-venv/bin/python"
ROOT=/mnt/e/hx/_models

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
  --port 8000 --gpu-memory-utilization 0.90 --max-model-len 16384 > results/vllm_server_combined.log 2>&1 &
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
if [ $? -ne 0 ]; then kill $SERVER_PID; echo "=== chain11 aborted ==="; exit 1; fi

# full A-Mem (rebuilds the note bank; previous run died writing the cache)
rm -f results/amem_bank.jsonl
"$PY" -X utf8 -m experiments.run_amem --name amem-full > results/amem-full.log 2>&1
AMEM_RC=$?
tail -2 results/amem-full.log | cut -c1-160

# Mem0 smoke
rm -rf /tmp/m2a_mem0_faiss /tmp/m2a_mem0_history.db
"$PY" -X utf8 -m experiments.run_mem0 --limit 20 --name mem0-smoke > results/mem0-smoke.log 2>&1
MEM0_RC=$?
tail -2 results/mem0-smoke.log | cut -c1-160

kill $SERVER_PID
echo "=== chain11 done (amem rc=$AMEM_RC, mem0 rc=$MEM0_RC) ==="
