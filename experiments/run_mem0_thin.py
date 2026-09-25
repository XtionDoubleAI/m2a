"""Run Mem0 over Mem2ActBench -- paper-level faithful reimplementation.

The official mem0ai 2.1.0 library was tried first and terminated on cost:
its write side issues two sequential LLM calls per batch (extract + memory
update/merge), which measured at ~4 min/batch -- 33h ingested under half of
the 429 sessions, projecting 17-60h to finish. That cost is itself reported
(paper cost dimension). This thin reimplementation keeps Mem0's mechanisms
minus the interactive update/merge loop:

  1. write: each session, split into char-budgeted batches; one LLM call per
     batch extracts atomic fact memories (single declarative sentences,
     values verbatim) -- Mem0's extraction prompt style
  2. store: dense vectors (BGE-M3), flat index
  3. read: query embedding, top-k over the task's visible sessions

Deviations from the official system (logged into runs.jsonl): no LLM-based
memory update/dedup at write time (the official library's per-batch second
pass); extraction runs concurrently (16 threads) purely for wall-clock.

LLM: local vLLM OpenAI server (Qwen2.5-7B-Instruct, same as all systems).
Answer side: identical given-tool prompt and parser as every other system.

Usage (WSL): python -X utf8 -m experiments.run_mem0_thin [--limit N]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from experiments.baselines.ltmemory import SYSTEM_PROMPT, chunk_session  # noqa: E402
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
MAX_CHARS_PER_BATCH = 6000
CONCURRENCY = 16

EXTRACT_PROMPT = (
    "You are building a long-term memory for an AI assistant. From the dialogue "
    "excerpt, extract atomic fact memories as a JSON object: "
    '{"memories": ["<single declarative sentence>", ...]}. '
    "Keep identifiers, addresses, URLs and configuration values VERBATIM. "
    "One fact per entry; include the user preference, attribute or event each "
    "entry refers to. Output JSON only."
)


def bge_dir() -> str:
    direct = MODELS_DIR / "BAAI/bge-m3"
    if (direct / "config.json").exists():
        return str(direct)
    ms = MODELS_DIR / "models" / "BAAI--bge-m3" / "snapshots"
    snaps = sorted(ms.glob("*")) if ms.exists() else []
    return str(snaps[-1] if snaps else direct)


class ServerChat:
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(base_url=VLLM_BASE, api_key="dummy", timeout=300)

    def chat(self, system: str, user: str) -> str:
        r = self.client.chat.completions.create(
            model=MODEL_NAME, temperature=0.0, max_tokens=2000,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        return r.choices[0].message.content or ""


def parse_json_loose(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def session_batches(session, max_chars: int) -> list[list[dict]]:
    msgs = []
    for t in session.turns:
        role, content = t.get("role", "user"), t.get("content", "") or ""
        if role in ("assistant", "user"):
            msgs.append({"role": role, "content": content})
    batches, cur, size = [], [], 0
    for m in msgs:
        if size + len(m["content"]) > max_chars and cur:
            batches.append(cur)
            cur, size = [], 0
        cur.append(m)
        size += len(m["content"])
    if cur:
        batches.append(cur)
    return batches


class FactStore:
    """Mem0-style flat fact store: extracted sentences, dense retrieval."""

    def __init__(self, embedder: BGEM3Dense):
        self.embedder = embedder
        self.facts: list[dict] = []   # {session_id, text}
        self.vecs: list = []

    def build(self, bench: Bench, chat: ServerChat) -> None:
        jobs = []  # (session_id, batch)
        for s in bench.sessions:
            for b in session_batches(s, MAX_CHARS_PER_BATCH):
                jobs.append((s.session_id, b))
        print(f"extraction jobs: {len(jobs)}")

        def extract(job):
            sid, batch = job
            convo = "\n".join(f"{m['role']}: {m['content']}" for m in batch)
            d = parse_json_loose(chat.chat(EXTRACT_PROMPT, f"Dialogue excerpt:\n{convo}"))
            return [(sid, t.strip()) for t in (d.get("memories") or [])
                    if isinstance(t, str) and t.strip()]
        with ThreadPoolExecutor(CONCURRENCY) as ex:
            for i, facts in enumerate(ex.map(extract, jobs)):
                self.facts.extend({"session_id": sid, "text": t} for sid, t in facts)
                if (i + 1) % 100 == 0:
                    print(f"  {i+1}/{len(jobs)} jobs, {len(self.facts)} facts")
        self.vecs = self.embedder.encode([f["text"] for f in self.facts])

    def search(self, query: str, session_ids: set, k: int) -> list[str]:
        pool = [i for i, f in enumerate(self.facts) if f["session_id"] in session_ids]
        if not pool:
            return []
        qv = self.embedder.encode([query])[0]
        sims = np.stack([self.vecs[i] for i in pool]) @ qv
        order = np.argsort(-sims)[: k]
        return [self.facts[pool[r]]["text"] for r in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--llm", choices=["local", "api"], default="local",
                    help="answer-side LLM: local vLLM or hosted OpenAI-compatible API")
    ap.add_argument("--host-model", default="DeepSeek-V4-Flash",
                    help="model name on the API side")
    ap.add_argument("--reasoning", choices=["default", "none", "high"], default="default",
                    help="reasoning_effort for hosted thinking models")
    ap.add_argument("--host-max-tokens", type=int, default=None)
    ap.add_argument("--name", default="mem0-full")
    ap.add_argument("--bank-cache", default=None,
                    help="override the fact-bank cache path")
    ap.add_argument("--bench-dir", default=None,
                    help="override benchmark data dir")
    args = ap.parse_args()

    chat = ServerChat()
    embedder = BGEM3Dense(bge_dir(), device="cuda:1")

    bench = Bench(Path(args.bench_dir) if args.bench_dir else BENCH_DIR)
    print("bench:", json.dumps(bench.stats(), ensure_ascii=False))

    store = FactStore(embedder)
    cache = Path(args.bank_cache) if args.bank_cache else OUT_DIR / "mem0_facts.jsonl"
    if cache.exists():
        for line in open(cache, encoding="utf-8"):
            d = json.loads(line)
            store.facts.append(d)
        store.vecs = embedder.encode([f["text"] for f in store.facts])
        print(f"facts loaded from cache: {len(store.facts)}")
    else:
        store.build(bench, chat)
        with open(cache, "w", encoding="utf-8") as f:
            for fact in store.facts:
                f.write(json.dumps(fact, ensure_ascii=False) + "\n")
        print(f"facts: {len(store.facts)} (cached to {cache})")

    if args.llm == "api":
        from forge.act.llm import make_answer_llm
        answer_llm = make_answer_llm("api", args.host_model, args.reasoning,
                                     args.host_max_tokens)
    else:
        answer_llm = ServerChat()
    out_path = OUT_DIR / f"{args.name}.jsonl"
    inter_fh = open(OUT_DIR / f"{args.name}.intermediates.jsonl", "w", encoding="utf-8")

    def system(task, session_texts=None):
        shown = store.search(task.query, set(task.session_ids), TOTAL_K)
        user = (
            f"Tool schema:\n{json.dumps(task.tool_schema, ensure_ascii=False, indent=1)}\n\n"
            f"Memory evidence:\n" + ("\n---\n".join(shown) if shown else "(none found)")
            + f"\n\nUser request: {task.query}"
        )
        pred_tool, pred_args = parse_tool_call_json(answer_llm.chat(SYSTEM_PROMPT, user))
        inter_fh.write(json.dumps({"qa_id": task.qa_id, "evidence": "\n---\n".join(shown),
                                   "pred_args": pred_args, "gold_args": task.arguments,
                                   "pred_tool": pred_tool}, ensure_ascii=False) + "\n")
        return pred_tool, pred_args

    results = run_system(system, bench, out_path, limit=args.limit)
    inter_fh.close()
    aggs = aggregate(results)
    log_run(args.name, config={
        **vars(args), "lib": "thin-reimpl-of-mem0",
        "llm": {"server": VLLM_BASE, "model": MODEL_NAME,
                "temperature": 0.0, "max_tokens": 2000},
        "embedder": {"model": "BAAI/bge-m3", "device": "cuda:1"},
        "retrieval": {"k": TOTAL_K},
        "write_side": {"batch_chars": MAX_CHARS_PER_BATCH,
                       "extraction_prompt": "EXTRACT_PROMPT (this file)",
                       "concurrency": CONCURRENCY},
        "answer_side": "shared SYSTEM_PROMPT (baselines/ltmemory.py) + parse_tool_call_json",
        "deviations_from_official": ["no write-time LLM update/merge pass",
                                     "concurrent extraction (wall-clock only)",
                                     "unified embedder and answer prompt"],
        "official_lib_cost_note": "mem0ai 2.1.0 terminated after 33h ingesting "
                                  "<half of 429 sessions (~4 min/batch, 2 sequential "
                                  "LLM calls); cost reported in the paper"},
            aggregates={"f1": round(aggs.f1 * 100, 2), "bleu1": round(aggs.bleu1 * 100, 2),
                        "tsa": round(aggs.tsa * 100, 2), "em": round(aggs.em * 100, 2),
                        "arg_f1": round(aggs.arg_f1 * 100, 2),
                        "slot_acc": round(aggs.slot_acc * 100, 2)},
            files={"samples": str(out_path)},
            runs_path=OUT_DIR / "runs.jsonl")


if __name__ == "__main__":
    main()
