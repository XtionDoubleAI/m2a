"""Run Self-Controlled Memory (Wang et al., arXiv 2304.13343) over
Mem2ActBench -- paper-level faithful reimplementation following the same
thin-reimplementation protocol as the other baseline runners.

Faithful core (the paper's framework):
  1. memory stream: each session is condensed by the LLM memory controller
     into short factual memory entries (write-side controller decision)
  2. read-side controller: for a query, candidate memories are retrieved by
     dense similarity, then the LLM controller selects and orders the ones
     actually relevant to the request -- the paper's "decide when and how to
     utilize memories" step, kept as an in-the-loop LLM call

Deviations from the paper (logged into runs.jsonl): the paper's "does this
query need memory at all" gate is skipped (under the given-tool protocol every
task needs parameter values from history, so the gate would always fire);
memories are per-session summaries rather than incremental interaction
records (the benchmark offers whole sessions, not a live interaction stream).

LLM: local vLLM OpenAI server (Qwen2.5-7B-Instruct, same as all systems).
Embeddings: BGE-M3 dense (same as all systems).

Requires: vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 (GPU0).

Usage (WSL): python -X utf8 -m experiments.run_scm [--limit N]
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
CAND_K = 10              # dense candidates handed to the controller
TOTAL_K = 5              # memories finally rendered
CONCURRENCY = 16

SUMMARIZE_PROMPT = (
    "You are the memory controller of an AI assistant. Condense this dialogue "
    "session into 1-3 short memory entries. Each entry is one line, keeps all "
    "concrete values verbatim (names, IDs, addresses, dates, numbers, "
    "settings, preferences), and drops small talk. Reply with the lines only, "
    "no numbering."
)
CONTROLLER_PROMPT = (
    "You are the memory controller of an AI assistant. The user request below "
    "needs parameter values from memory. From the candidate memories, select "
    "the ones that could supply any parameter value for this request "
    "(including values the request only implies). Reply as a JSON object: "
    '{"selected": [<1-based candidate numbers>, most relevant first], '
    '"reason": "<one sentence>"}. Select at most 8. Output JSON only.'
)


class ServerChat:
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(base_url=VLLM_BASE, api_key="dummy", timeout=180)

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


def bge_dir() -> str:
    direct = MODELS_DIR / "BAAI/bge-m3"
    if (direct / "config.json").exists():
        return str(direct)
    ms = MODELS_DIR / "models" / "BAAI--bge-m3" / "snapshots"
    snaps = sorted(ms.glob("*")) if ms.exists() else []
    return str(snaps[-1] if snaps else direct)


def session_text(session) -> str:
    return "\n".join(f"{t['role']}: {t.get('content', '')}"
                     for t in session.turns if t.get("content"))


def session_chunks(session, window: int = 6):
    from experiments.baselines.ltmemory import chunk_session
    return chunk_session(session.turns, window=window)


class SCMMemory:
    """Controller-condensed memory stream with controller-in-the-loop reads."""

    def __init__(self, chat: ServerChat, embedder: BGEM3Dense):
        self.chat = chat
        self.embedder = embedder
        self.entries: list[dict] = []   # {session_id, text}
        self.vecs: list = []
        self.write_calls = 0
        self.read_calls = 0

    def build(self, bench: Bench):
        jobs = []
        for session in bench.sessions:
            for chunk in session_chunks(session):
                jobs.append((session.session_id, chunk))

        def one(job):
            sid, chunk = job
            out = self.chat.chat(SUMMARIZE_PROMPT, chunk[:6000])
            lines = [l.strip(" -•") for l in out.splitlines()
                     if len(l.strip(" -•")) > 8]
            return sid, [l[:800] for l in lines[:2]]
        with ThreadPoolExecutor(CONCURRENCY) as ex:
            for sid, lines in ex.map(one, jobs):
                self.write_calls += 1
                for text in lines:
                    self.entries.append({"session_id": sid, "text": text})
        if self.entries:
            self.vecs = self.embedder.encode([e["text"] for e in self.entries])

    def search(self, query: str, session_ids: set, controller_chat) -> list[str]:
        pool = [(i, e) for i, e in enumerate(self.entries)
                if e["session_id"] in session_ids]
        if not pool:
            return []
        idxs = [i for i, _ in pool]
        qv = self.embedder.encode([query])[0]
        sims = np.stack([self.vecs[i] for i in idxs]) @ qv
        cand = [pool[r][1]["text"] for r in np.argsort(-sims)[: CAND_K]]
        listing = "\n".join(f"({n}) {t[:400]}" for n, t in enumerate(cand, 1))
        self.read_calls += 1
        d = parse_json_loose(controller_chat.chat(
            CONTROLLER_PROMPT,
            f"User request: {query}\n\nCandidate memories:\n{listing}"))
        picked = [cand[n - 1] for n in d.get("selected", [])
                  if isinstance(n, int) and 1 <= n <= len(cand)]
        if not picked:                       # controller failed: dense fallback
            picked = cand[: TOTAL_K]
        return picked[: TOTAL_K]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--llm", choices=["local", "api"], default="local")
    ap.add_argument("--host-model", default="DeepSeek-V4-Flash")
    ap.add_argument("--reasoning", choices=["default", "none", "high"], default="default")
    ap.add_argument("--host-max-tokens", type=int, default=None)
    ap.add_argument("--name", default="scm-full")
    ap.add_argument("--bank-cache", default=str(OUT_DIR / "scm_bank.jsonl"))
    args = ap.parse_args()

    chat = ServerChat()
    embedder = BGEM3Dense(bge_dir(), device="cuda:1")
    bank = SCMMemory(chat, embedder)

    bench = Bench(BENCH_DIR)
    print("bench:", json.dumps(bench.stats(), ensure_ascii=False))

    if Path(args.bank_cache).exists():
        for line in open(args.bank_cache, encoding="utf-8"):
            bank.entries.append(json.loads(line))
        bank.vecs = embedder.encode([e["text"] for e in bank.entries])
        print(f"bank loaded from cache: {len(bank.entries)} entries")
    else:
        t0 = time.time()
        bank.build(bench)
        with open(args.bank_cache, "w", encoding="utf-8") as f:
            for e in bank.entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"bank: {len(bank.entries)} entries, "
              f"{bank.write_calls} write-side LLM calls, "
              f"{time.time()-t0:.0f}s wall")

    if args.llm == "api":
        from forge.act.llm import make_answer_llm
        answer_llm = make_answer_llm("api", args.host_model, args.reasoning,
                                     args.host_max_tokens)
    else:
        answer_llm = ServerChat()
    out_path = OUT_DIR / f"{args.name}.jsonl"
    inter_fh = open(OUT_DIR / f"{args.name}.intermediates.jsonl", "w", encoding="utf-8")

    def system(task, session_texts=None):
        shown = bank.search(task.query, set(task.session_ids), chat)
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
        "protocol": "self-controlled memory (controller-condensed stream, "
                    "controller-in-the-loop selection)",
        "model": MODEL_NAME, "limit": args.limit,
        "prompt": "baselines/ltmemory.SYSTEM_PROMPT (shared harness)",
    }, {
        "f1": round(agg.f1 * 100, 2), "bleu1": round(agg.bleu1 * 100, 2),
        "tsa": round(agg.tsa * 100, 2), "em": round(agg.em * 100, 2),
        "slot_acc": round(agg.slot_acc * 100, 2),
    }, {"samples": str(out_path)}, OUT_DIR / "runs.jsonl")


if __name__ == "__main__":
    main()
