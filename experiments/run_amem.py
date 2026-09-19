"""Run A-Mem (Agentic Memory, arXiv 2502.12110) over Mem2ActBench -- paper-level
faithful reimplementation, not the official repo (its pip package is a
different project; the GitHub clone was blocked by network at integration
time). Deviations are listed here and logged into runs.jsonl.

Faithful core (the paper's three mechanisms):
  1. note construction: each dialogue chunk becomes a structured note with
     LLM-generated content/context/keywords/tags (Zettelkasten-style)
  2. link generation: at insert time the new note activates its nearest
     neighbours and the LLM decides which to link; linked neighbours get
     their context updated (memory evolution) -- kept, batched into the
     same call
  3. retrieval: dense similarity over notes + one-hop link expansion

Deviations from the paper: no time-decay term (task sessions span too
little time for tau to matter); evolution updates are capped at the top
neighbour per insert (cost); both noted in runs.jsonl.

LLM: local vLLM OpenAI server (Qwen2.5-7B-Instruct, same as all systems).
Embeddings: BGE-M3 dense (same as all systems).

Requires: vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 (GPU0).

Usage (WSL): python -X utf8 -m experiments.run_amem [--limit N]
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
NEIGHBOURS = 5          # activation set size at insert (paper's neighbour recall)
CONCURRENCY = 16

NOTE_PROMPT = (
    "You are building a Zettelkasten-style memory of a user's conversations. "
    "Given a dialogue excerpt, write one atomic memory note as a JSON object with keys: "
    '"content" (the key facts, values verbatim), '
    '"context" (one sentence: when/why this note matters), '
    '"keywords" (3-6 searchable words), '
    '"tags" (1-3 category labels). Output JSON only.'
)
LINK_PROMPT = (
    "You have a new memory note and its most similar existing notes. Decide which "
    "existing notes the new note should LINK to (they will be co-activated at "
    'retrieval). Reply as a JSON object: {"links": [<indices>], '
    '"context_update": {"<index>": "<one appended clause for that note\'s context>"}} '
    "where indices are into the provided list (0-based). Link only notes that are "
    "about the same user, object or topic. Output JSON only."
)


def bge_dir() -> str:
    direct = MODELS_DIR / "BAAI/bge-m3"
    if (direct / "config.json").exists():
        return str(direct)
    ms = MODELS_DIR / "models" / "BAAI--bge-m3" / "snapshots"
    snaps = sorted(ms.glob("*")) if ms.exists() else []
    return str(snaps[-1] if snaps else direct)


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


def parse_json_loose(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


class NoteBank:
    """A-Mem style note store: structured notes + LLM links + evolution."""

    def __init__(self, chat: ServerChat, embedder: BGEM3Dense):
        self.chat = chat
        self.embedder = embedder
        self.notes: list[dict] = []       # {session_id, content, context, keywords, tags, links}
        self.vecs: list = []

    def _render(self, note: dict) -> str:
        return (f"content: {note['content']}\ncontext: {note['context']}\n"
                f"keywords: {', '.join(note['keywords'])}\ntags: {', '.join(note['tags'])}")

    def insert_batch(self, session_id: str, chunks: list[str]) -> None:
        """Structure all chunks in parallel, then link each to its neighbours."""
        with ThreadPoolExecutor(CONCURRENCY) as ex:
            structured = list(ex.map(self._structure, chunks))
        texts = [self._render(n) for n in structured]
        vecs = self.embedder.encode(texts)
        for ch_text, note, vec in zip(chunks, structured, vecs):
            links = self._link(note, vec)
            note["links"] = links["indices"]
            self._evolve(links.get("update", {}))
            note["session_id"] = session_id
            self.notes.append(note)
            self.vecs.append(vec)

    def _structure(self, chunk: str) -> dict:
        d = parse_json_loose(self.chat.chat(NOTE_PROMPT, f"Dialogue excerpt:\n{chunk}"))
        note = {
            "content": d.get("content", chunk[:400]),
            "context": d.get("context", ""),
            "keywords": d.get("keywords", []) or [],
            "tags": d.get("tags", []) or [],
        }
        return note

    def _neighbours(self, vec, n: int) -> list[int]:
        if not self.vecs:
            return []
        sims = np.stack(self.vecs) @ vec
        return list(np.argsort(-sims)[: n])

    def _link(self, note: dict, vec) -> dict:
        nbrs = self._neighbours(vec, NEIGHBOURS)
        if not nbrs:
            return {"indices": [], "update": {}}
        listing = "\n".join(f"[{i}] {self._render(self.notes[i])}"[:600]
                            for i in nbrs)
        d = parse_json_loose(self.chat.chat(
            LINK_PROMPT, f"New note:\n{self._render(note)}\n\nExisting notes:\n{listing}"))
        idx = [i for i in d.get("links", []) if isinstance(i, int) and 0 <= i < len(nbrs)]
        upd = d.get("context_update", {}) or {}
        update = {nbrs[int(k)]: v for k, v in upd.items()
                  if str(k).isdigit() and int(k) < len(nbrs) and isinstance(v, str)}
        return {"indices": [nbrs[i] for i in dict.fromkeys(idx)], "update": update}

    def _evolve(self, update: dict) -> None:
        for i, clause in update.items():
            if i < len(self.notes):
                self.notes[i]["context"] = (self.notes[i]["context"] + " | " + clause)[:1200]

    def search(self, query: str, session_ids: set, k: int) -> list[str]:
        """Dense top-k among the task's visible sessions + one-hop link expansion."""
        pool = [(i, n) for i, n in enumerate(self.notes) if n["session_id"] in session_ids]
        if not pool:
            return []
        idxs = [i for i, _ in pool]
        qv = self.embedder.encode([query])[0]
        sims = np.stack([self.vecs[i] for i in idxs]) @ qv
        order = np.argsort(-sims)
        picked, seen = [], set()
        for rank in order:
            i = idxs[rank]
            if i not in seen:
                picked.append(i)
                seen.add(i)
            # link expansion stays inside the visible-sessions protocol
            for j in self.notes[i]["links"]:
                if j not in seen and self.notes[j]["session_id"] in session_ids:
                    picked.append(j)
                    seen.add(j)
            if len(picked) >= k:
                break
        return [self.notes[i]["content"] for i in picked[: k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--name", default="amem-full")
    ap.add_argument("--bank-cache", default=str(OUT_DIR / "amem_bank.jsonl"))
    args = ap.parse_args()

    chat = ServerChat()
    embedder = BGEM3Dense(bge_dir(), device="cuda:1")
    bank = NoteBank(chat, embedder)

    bench = Bench(BENCH_DIR)
    print("bench:", json.dumps(bench.stats(), ensure_ascii=False))

    from pathlib import Path as _P
    cache = _P(args.bank_cache)
    if cache.exists():
        for line in open(cache, encoding="utf-8"):
            d = json.loads(line)
            d["links"] = d.get("links", [])
            bank.notes.append(d)
        bank.vecs = embedder.encode(
            [bank._render(n) for n in bank.notes])
        print(f"bank loaded from cache: {len(bank.notes)} notes")
    else:
        for s in bench.sessions:
            chunks = chunk_session(s.turns, window=6)
            if chunks:
                bank.insert_batch(s.session_id, chunks)
        with open(cache, "w", encoding="utf-8") as f:
            for n in bank.notes:
                f.write(json.dumps(n, ensure_ascii=False) + "\n")
        print(f"bank: {len(bank.notes)} notes (cached to {cache})")

    answer_llm = ServerChat()
    out_path = OUT_DIR / f"{args.name}.jsonl"
    inter_fh = open(OUT_DIR / f"{args.name}.intermediates.jsonl", "w", encoding="utf-8")

    def system(task, session_texts=None):
        shown = bank.search(task.query, set(task.session_ids), TOTAL_K)
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
    log_run(args.name, config={**vars(args), "lib": "thin-reimpl-of-arXiv:2502.12110",
                               "llm": "vllm-server-Qwen2.5-7B",
                               "deviations": ["no time-decay", "evolution top-1 cap",
                                              "chunk window 6", f"k={TOTAL_K}",
                                              f"link_neighbours={NEIGHBOURS}"]},
            aggregates={"f1": round(aggs.f1 * 100, 2), "bleu1": round(aggs.bleu1 * 100, 2),
                        "tsa": round(aggs.tsa * 100, 2), "em": round(aggs.em * 100, 2),
                        "arg_f1": round(aggs.arg_f1 * 100, 2),
                        "slot_acc": round(aggs.slot_acc * 100, 2)},
            files={"samples": str(out_path)},
            runs_path=OUT_DIR / "runs.jsonl")


if __name__ == "__main__":
    main()
