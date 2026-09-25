"""Run Generative Agents memory (Park et al., arXiv 2304.03442) over
Mem2ActBench -- paper-level faithful reimplementation following the same
thin-reimplementation protocol as the Mem0/A-Mem runners.

Faithful core (the paper's memory architecture):
  1. memory stream: every dialogue chunk is an observation appended to a
     flat, append-ordered stream
  2. importance scoring: an LLM rates each observation 1-10 (the paper's
     "poignancy" score) at write time
  3. three-factor retrieval: score = relevance (dense cosine) +
     importance (LLM score, normalized) + recency (exponential decay over
     session-order distance), equal weights as in the paper

Deviations from the paper (logged into runs.jsonl): no reflection synthesis
(the paper's periodic higher-order memories; the benchmark asks for verbatim
parameter values, and reflection outputs paraphrases); recency decays over
session order instead of wall-clock hours (the benchmark carries no
timestamps); importance is the only write-side LLM call.

LLM: local vLLM OpenAI server (Qwen2.5-7B-Instruct, same as all systems).
Embeddings: BGE-M3 dense (same as all systems).

Requires: vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 (GPU0).

Usage (WSL): python -X utf8 -m experiments.run_ga [--limit N]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
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
CONCURRENCY = 16
RECENCY_DECAY = 0.99      # per session-order step (paper: 0.995 per hour)

POIGNANCY_PROMPT = (
    "Rate how important this dialogue excerpt is as a memory of the user's "
    "facts, preferences, plans or identifiers (addresses, IDs, dates, names, "
    "settings). Reply with a single integer 1-10, nothing else. 10 means it "
    "records a concrete fact or value the user would need later; 1 means "
    "small talk."
)


class ServerChat:
    """OpenAI-compatible chat with a shared thread-safe client."""

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(base_url=VLLM_BASE, api_key="dummy", timeout=180)

    def chat(self, system: str, user: str) -> str:
        r = self.client.chat.completions.create(
            model=MODEL_NAME, temperature=0.0, max_tokens=2000,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        return r.choices[0].message.content or ""


def parse_score(text: str) -> int:
    m = re.search(r"\b(10|[1-9])\b", text or "")
    return int(m.group(1)) if m else 5


def bge_dir() -> str:
    direct = MODELS_DIR / "BAAI/bge-m3"
    if (direct / "config.json").exists():
        return str(direct)
    ms = MODELS_DIR / "models" / "BAAI--bge-m3" / "snapshots"
    snaps = sorted(ms.glob("*")) if ms.exists() else []
    return str(snaps[-1] if snaps else direct)


class MemoryStream:
    """Generative-Agents style store: append-ordered observations with LLM
    importance scores, retrieved by relevance+importance+recency."""

    def __init__(self, chat: ServerChat, embedder: BGEM3Dense):
        self.chat = chat
        self.embedder = embedder
        self.items: list[dict] = []   # {session_id, text, importance, order}
        self.session_order: dict[str, int] = {}
        self.vecs: list = []
        self.llm_calls = 0

    def insert_batch(self, session_id: str, chunks: list[str]) -> None:
        with ThreadPoolExecutor(CONCURRENCY) as ex:
            scores = list(ex.map(self._poignancy, chunks))
        self.session_order[session_id] = len(self.session_order)
        for text, imp in zip(chunks, scores):
            self.items.append({"session_id": session_id, "text": text,
                               "importance": imp, "order": len(self.items)})
        self.vecs.extend(self.embedder.encode(chunks))

    def _poignancy(self, chunk: str) -> int:
        self.llm_calls += 1
        return parse_score(self.chat.chat(POIGNANCY_PROMPT, chunk[:4000]))

    def search(self, query: str, session_ids: set, k: int,
               max_session_order: int) -> list[str]:
        pool = [(i, it) for i, it in enumerate(self.items)
                if it["session_id"] in session_ids]
        if not pool:
            return []
        idxs = [i for i, _ in pool]
        qv = self.embedder.encode([query])[0]
        sims = np.stack([self.vecs[i] for i in idxs]) @ qv
        rel = (sims - sims.min()) / max(sims.max() - sims.min(), 1e-9)
        scores = np.zeros(len(pool))
        for r, (i, it) in enumerate(pool):
            imp = it["importance"] / 10.0
            rec = RECENCY_DECAY ** (max_session_order
                                    - self.session_order[it["session_id"]])
            scores[r] = rel[r] + imp + rec      # equal weights (paper)
        order = np.argsort(-scores)[: k]
        return [pool[r][1]["text"] for r in order]


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
    ap.add_argument("--name", default="ga-full")
    ap.add_argument("--bank-cache", default=str(OUT_DIR / "ga_bank.jsonl"))
    args = ap.parse_args()

    chat = ServerChat()
    embedder = BGEM3Dense(bge_dir(), device="cuda:1")
    stream = MemoryStream(chat, embedder)

    bench = Bench(BENCH_DIR)
    print("bench:", json.dumps(bench.stats(), ensure_ascii=False))

    if Path(args.bank_cache).exists():
        for line in open(args.bank_cache, encoding="utf-8"):
            stream.items.append(json.loads(line))
        stream.session_order = {}
        for it in stream.items:
            stream.session_order.setdefault(it["session_id"],
                                            len(stream.session_order))
        stream.vecs = embedder.encode([it["text"] for it in stream.items])
        print(f"stream loaded from cache: {len(stream.items)} items")
    else:
        t0 = time.time()
        for s in bench.sessions:
            chunks = chunk_session(s.turns, window=6)
            if chunks:
                stream.insert_batch(s.session_id, chunks)
        with open(args.bank_cache, "w", encoding="utf-8") as f:
            for it in stream.items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"stream: {len(stream.items)} items, "
              f"{stream.llm_calls} write-side LLM calls, "
              f"{time.time()-t0:.0f}s wall (cached to {args.bank_cache})")

    if args.llm == "api":
        from forge.act.llm import make_answer_llm
        answer_llm = make_answer_llm("api", args.host_model, args.reasoning,
                                     args.host_max_tokens)
    else:
        answer_llm = ServerChat()
    out_path = OUT_DIR / f"{args.name}.jsonl"
    inter_fh = open(OUT_DIR / f"{args.name}.intermediates.jsonl", "w", encoding="utf-8")

    max_order = max(stream.session_order.values())

    def system(task, session_texts=None):
        shown = stream.search(task.query, set(task.session_ids), TOTAL_K,
                              max_order)
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
        "protocol": "generative-agents memory stream (importance+recency+relevance, "
                    "no reflection)",
        "model": MODEL_NAME, "limit": args.limit,
        "prompt": "baselines/ltmemory.SYSTEM_PROMPT (shared harness)",
    }, {
        "f1": round(agg.f1 * 100, 2), "bleu1": round(agg.bleu1 * 100, 2),
        "tsa": round(agg.tsa * 100, 2), "em": round(agg.em * 100, 2),
        "slot_acc": round(agg.slot_acc * 100, 2),
    }, {"samples": str(out_path)}, OUT_DIR / "runs.jsonl")


if __name__ == "__main__":
    main()
