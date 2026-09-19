"""Run Mem0 (mem0ai 2.1.0, official library) over Mem2ActBench.

Protocol mapping (the paper open-sources no adapters, so these choices are
ours and are logged verbatim into runs.jsonl):
  - write side: each evidence session is added as one memory.add() call with
    the full turn list, user_id = session_id -- Mem0's internal fact
    extraction / consolidation runs unchanged (that is the system under test)
  - read side: per visible session memory.search(query, user_id=sid, limit=3),
    merged and truncated to the best 5 by score -- matching the k=5 evidence
    budget of the LTMemory hybrid@5 calibration row
  - answer side: identical given-tool prompt and JSON parser as the LTMemory
    baseline; only the memory layer differs
  - LLM: local vLLM OpenAI server (same Qwen2.5-7B-Instruct as every other
    system in the paper tables)

Requires the server: vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 (GPU0).

Usage (WSL): python -X utf8 -m experiments.run_mem0 [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.baselines.ltmemory import SYSTEM_PROMPT  # noqa: E402
from experiments.runlog import log_run  # noqa: E402
from forge.act.llm import parse_tool_call_json  # noqa: E402
from forge.eval.dataset import Bench  # noqa: E402
from forge.eval.metrics import aggregate, score_sample  # noqa: E402
from forge.eval.runner import run_system  # noqa: E402

IS_WSL = sys.platform == "linux"
BENCH_DIR = Path(
    "/mnt/e/hx/_DoctorXtion/intern_shxt/Pdev/refs/repos/Mem2ActBench/toolmembench_small"
    if IS_WSL else
    r"E:\hx\_DoctorXtion\intern_shxt\Pdev\refs\repos\Mem2ActBench\toolmembench_small"
)
MODELS_DIR = Path("/mnt/e/hx/_models" if IS_WSL else r"E:\hx\_models")
OUT_DIR = Path(__file__).resolve().parent.parent / "results"
VLLM_BASE = "http://localhost:8000/v1"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
PER_SESSION_K = 3
TOTAL_K = 5


def bge_path() -> str:
    direct = MODELS_DIR / "BAAI/bge-m3"
    if (direct / "config.json").exists():
        return str(direct)
    ms = MODELS_DIR / "models" / "BAAI--bge-m3" / "snapshots"
    snaps = sorted(ms.glob("*")) if ms.exists() else []
    return str(snaps[-1] if snaps else direct)


def bench_messages(session) -> list[dict]:
    msgs = []
    for t in session.turns:
        role, content = t.get("role", "user"), t.get("content", "") or ""
        if role == "assistant" or role == "user":
            msgs.append({"role": role, "content": content})
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--name", default="mem0-full")
    args = ap.parse_args()

    from mem0 import Memory

    config = {
        "llm": {"provider": "openai", "config": {
            "model": MODEL_NAME, "openai_base_url": VLLM_BASE,
            "api_key": "dummy", "temperature": 0.0, "max_tokens": 4000}},
        "embedder": {"provider": "huggingface", "config": {"model": bge_path()}},
        "vector_store": {"provider": "faiss", "config": {
            "path": "/tmp/m2a_mem0_faiss", "collection_name": "m2a",
            "embedding_model_dims": 1024}},
        "history_db": {"provider": "sqlite", "config": {
            "path": "/tmp/m2a_mem0_history.db"}},
    }
    mem = Memory.from_config(config)

    bench = Bench(BENCH_DIR)
    print("bench:", json.dumps(bench.stats(), ensure_ascii=False))

    # ---- write side: ingest every evidence session once ----
    n_mem = 0
    for s in bench.sessions:
        msgs = bench_messages(s)
        if msgs:
            mem.add(msgs, user_id=s.session_id)
            n_mem += 1
    print(f"ingested {n_mem} sessions")

    answer_llm = _OpenAIChat()

    out_path = OUT_DIR / f"{args.name}.jsonl"
    inter_fh = open(OUT_DIR / f"{args.name}.intermediates.jsonl", "w", encoding="utf-8")

    def system(task, session_texts=None):
        evidence, seen = [], set()
        for sid in task.session_ids:
            try:
                res = mem.search(task.query, user_id=sid, limit=PER_SESSION_K)
            except Exception as e:  # search API shape varies across versions
                print("search error:", e)
                continue
            for r in _rows(res):
                text = r.get("memory", "")
                if text and text not in seen:
                    seen.add(text)
                    evidence.append((r.get("score", 0.0) or 0.0, text))
        evidence.sort(key=lambda x: -x[0])
        shown = [t for _, t in evidence[:TOTAL_K]]
        user = (
            f"Tool schema:\n{json.dumps(task.tool_schema, ensure_ascii=False, indent=1)}\n\n"
            f"Memory evidence:\n" + ("\n---\n".join(shown) if shown else "(none found)")
            + f"\n\nUser request: {task.query}"
        )
        pred_tool, pred_args = parse_tool_call_json(answer_llm.chat(SYSTEM_PROMPT, user))
        inter_fh.write(json.dumps({"qa_id": task.qa_id, "evidence": "\n---\n".join(shown),
                                   "pred_args": pred_args, "gold_args": task.arguments,
                                   "pred_tool": pred_tool},
                                  ensure_ascii=False) + "\n")
        return pred_tool, pred_args

    results = run_system(system, bench, out_path, limit=args.limit)
    inter_fh.close()
    aggs = aggregate(results)
    log_run(args.name, config={**vars(args), "lib": "mem0ai-2.1.0",
                               "llm": "vllm-server-Qwen2.5-7B",
                               "protocol": {"user_id": "session_id",
                                            "per_session_k": PER_SESSION_K,
                                            "total_k": TOTAL_K}},
            aggregates={"f1": round(aggs.f1 * 100, 2), "bleu1": round(aggs.bleu1 * 100, 2),
                        "tsa": round(aggs.tsa * 100, 2), "em": round(aggs.em * 100, 2),
                        "arg_f1": round(aggs.arg_f1 * 100, 2),
                        "slot_acc": round(aggs.slot_acc * 100, 2)},
            files={"samples": str(out_path)},
            runs_path=OUT_DIR / "runs.jsonl")


def _rows(res):
    """mem0 search results come as dict {'results': [...]} or list across versions."""
    if isinstance(res, dict) and "results" in res:
        return res["results"]
    if isinstance(res, dict):
        return [res]
    return list(res or [])


class _OpenAIChat:
    """Minimal OpenAI-compatible chat client pointed at the local vLLM server."""

    def __init__(self, base_url: str = VLLM_BASE, model: str = MODEL_NAME):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key="dummy")
        self.model = model

    def chat(self, system: str, user: str) -> str:
        r = self.client.chat.completions.create(
            model=self.model, temperature=0.0, max_tokens=4000,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        return r.choices[0].message.content or ""


if __name__ == "__main__":
    main()
