"""Run LangMem (LangChain's official memory SDK) over Mem2ActBench -- the one
baseline reimplemented through the official library rather than from the
paper, per the thin-reimplementation protocol.

Faithful core (the official SDK's hot-path extraction):
  - write side: `create_memory_manager` with the library's default reflective
    instructions (extract / consolidate / update) runs over each session's
    message list against the local Qwen server; resulting memory entries are
    stored verbatim
  - read side: dense retrieval over the extracted entries, same BGE-M3
    embedder and same top-k rendering as the Mem0 runner

Deviations (logged into runs.jsonl): memory entries are session-scoped and
kept in a flat store -- the SDK's thread/store persistence layer is bypassed
(the benchmark grades evidence supply, not storage); message tool_calls are
serialized into text (the local server accepts plain content messages).

LLM: local vLLM OpenAI server (Qwen2.5-7B-Instruct, same as all systems).

Requires: vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 (GPU0).

Usage (WSL): python -X utf8 -m experiments.run_langmem [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from experiments.baselines.ltmemory import SYSTEM_PROMPT  # noqa: E402
from experiments.runlog import log_run  # noqa: E402
from forge.act.embedder import BGEM3Dense  # noqa: E402
from forge.act.llm import parse_tool_call_json  # noqa: E402
from forge.eval.dataset import Bench  # noqa: E402
from forge.eval.metrics import aggregate  # noqa: E402
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
TOTAL_K = 5
CONCURRENCY = 4            # extraction runs an internal tool loop per session


def make_chat_model():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(base_url=VLLM_BASE, api_key="dummy",
                      model=MODEL_NAME, temperature=0, max_tokens=2000)


def bge_dir() -> str:
    direct = MODELS_DIR / "BAAI/bge-m3"
    if (direct / "config.json").exists():
        return str(direct)
    ms = MODELS_DIR / "models" / "BAAI--bge-m3" / "snapshots"
    snaps = sorted(ms.glob("*")) if ms.exists() else []
    return str(snaps[-1] if snaps else direct)


def to_lc_messages(turns) -> list[dict]:
    out = []
    for t in turns:
        role = t.get("role", "user")
        content = t.get("content", "") or ""
        if t.get("tool_calls"):
            calls = "; ".join(
                f"{c['function']['name']}({c['function'].get('arguments', '')})"
                for c in t["tool_calls"])
            content = (content + "\n" if content else "") + f"[tool call: {calls}]"
        # flatten to plain user/assistant text: the extractor only reads the
        # text, and OpenAI-format tool messages require matching
        # tool_call_id chains that raw session turns do not carry
        role = "assistant" if role in ("assistant", "tool") else "user"
        out.append({"role": role, "content": content})
    return out


def extract_memory(chat, session):
    """Official hot-path extraction for one session; returns text entries."""
    try:
        msgs = to_lc_messages(session.turns)
        # keep within the server's 16384-token window (~4 chars/token):
        # drop middle turns, keep head and tail
        total = sum(len(m["content"]) for m in msgs)
        budget = 40000
        if total > budget:
            kept, acc = [], 0
            for m in msgs:
                if acc + len(m["content"]) > budget // 2:
                    break
                kept.append(m)
                acc += len(m["content"])
            tail, acc2 = [], 0
            for m in reversed(msgs):
                if acc2 + len(m["content"]) > budget // 2 or m in kept:
                    break
                tail.append(m)
                acc2 += len(m["content"])
            msgs = kept + [{"role": "assistant", "content": "(earlier turns omitted)"}] + list(reversed(tail))
        result = chat.invoke({"messages": msgs})
        items = result if isinstance(result, list) else []
        texts = []
        for it in items:
            if isinstance(it, str):
                texts.append(it)
            elif isinstance(it, dict):
                t = it.get("text") or it.get("content") or ""
                if t:
                    texts.append(t)
            else:
                t = getattr(it, "text", None) or getattr(it, "content", "")
                if t:
                    texts.append(str(t))
        return session.session_id, [x.strip() for x in texts if x.strip()]
    except Exception as e:
        print(f"extraction failed on {session.session_id}: {e}")
        return session.session_id, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--llm", choices=["local", "api"], default="local")
    ap.add_argument("--host-model", default="DeepSeek-V4-Flash")
    ap.add_argument("--reasoning", choices=["default", "none", "high"], default="default")
    ap.add_argument("--host-max-tokens", type=int, default=None)
    ap.add_argument("--name", default="langmem-full")
    ap.add_argument("--bank-cache", default=str(OUT_DIR / "langmem_bank.jsonl"))
    args = ap.parse_args()

    from langmem import create_memory_manager
    manager = create_memory_manager(make_chat_model())

    embedder = BGEM3Dense(bge_dir(), device="cuda:1")
    bench = Bench(BENCH_DIR)
    print("bench:", json.dumps(bench.stats(), ensure_ascii=False))

    entries: list[dict] = []
    if Path(args.bank_cache).exists():
        for line in open(args.bank_cache, encoding="utf-8"):
            entries.append(json.loads(line))
        print(f"bank loaded from cache: {len(entries)} entries")
    else:
        t0 = time.time()
        with ThreadPoolExecutor(CONCURRENCY) as ex:
            for sid, texts in ex.map(lambda s: extract_memory(manager, s),
                                     bench.sessions):
                entries.extend({"session_id": sid, "text": t} for t in texts)
        with open(args.bank_cache, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"bank: {len(entries)} entries, {time.time()-t0:.0f}s wall")

    vecs = embedder.encode([e["text"] for e in entries]) if entries else []

    if args.llm == "api":
        from forge.act.llm import make_answer_llm
        answer_llm = make_answer_llm("api", args.host_model, args.reasoning,
                                     args.host_max_tokens)
    else:
        from openai import OpenAI
        client = OpenAI(base_url=VLLM_BASE, api_key="dummy", timeout=180)

        class Answer:
            def chat(self, system, user):
                r = client.chat.completions.create(
                    model=MODEL_NAME, temperature=0.0, max_tokens=2000,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}])
                return r.choices[0].message.content or ""
        answer_llm = Answer()
    out_path = OUT_DIR / f"{args.name}.jsonl"
    inter_fh = open(OUT_DIR / f"{args.name}.intermediates.jsonl", "w", encoding="utf-8")

    def system(task, session_texts=None):
        pool = [(i, e) for i, e in enumerate(entries)
                if e["session_id"] in set(task.session_ids)]
        shown = []
        if pool:
            idxs = [i for i, _ in pool]
            qv = embedder.encode([task.query])[0]
            sims = np.stack([vecs[i] for i in idxs]) @ qv
            shown = [pool[r][1]["text"]
                     for r in np.argsort(-sims)[: TOTAL_K]]
        user = (
            f"Tool schema:\n{json.dumps(task.tool_schema, ensure_ascii=False, indent=1)}\n\n"
            f"Memory evidence:\n" + ("\n---\n".join(shown) if shown else "(none found)")
            + f"\n\nUser request: {task.query}"
        )
        pred_tool, pred_args = parse_tool_call_json(answer_llm.chat(SYSTEM_PROMPT, user))
        inter_fh.write(json.dumps({"qa_id": task.qa_id,
                                   "evidence": "\n---\n".join(shown),
                                   "pred_args": pred_args,
                                   "gold_args": task.arguments},
                                  ensure_ascii=False) + "\n")
        return pred_tool, pred_args

    results = run_system(system, bench, out_path, limit=args.limit)
    inter_fh.close()
    agg = aggregate(results)
    print(f"final: {agg!r}")
    log_run(args.name, {
        "protocol": "langmem official SDK extraction (default reflective "
                    "instructions) + dense retrieval",
        "model": MODEL_NAME, "limit": args.limit,
        "prompt": "baselines/ltmemory.SYSTEM_PROMPT (shared harness)",
    }, {
        "f1": round(agg.f1 * 100, 2), "bleu1": round(agg.bleu1 * 100, 2),
        "tsa": round(agg.tsa * 100, 2), "em": round(agg.em * 100, 2),
        "slot_acc": round(agg.slot_acc * 100, 2),
    }, {"samples": str(out_path)}, OUT_DIR / "runs.jsonl")


if __name__ == "__main__":
    main()
