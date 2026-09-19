"""LTMemory baseline: passive RAG over session chunks (paper Table 3/4 reference).

Design aligned with the paper's passive-retrieval protocol:
  - corpus: each evidence session is split into turn-window chunks
  - retrieval: hybrid BM25 + dense (BGE-M3), Reciprocal Rank Fusion, k=5
  - answering: given-tool mode -- the gold tool schema is provided and the model
    must output {"name": ..., "arguments": {...}}

Calibration anchors (Qwen2.5-7B, temp=0):
  paper Table 3 LTMemory  F1=26.71 BLEU=64.07 TA=87.25
  paper Table 4 hybrid@5  F1=30.7  BLEU=29.7  TSA=86.0
Chunk granularity and prompt wording are the free knobs; adjust against anchors.
"""

from __future__ import annotations

import json

from forge.act.llm import LLMClient, parse_tool_call_json

SYSTEM_PROMPT = (
    "You are a tool-calling assistant. The target tool's API schema will be provided; "
    "you MUST call exactly this tool and no other. Given the user request and retrieved "
    "memory evidence, output the tool call as a single JSON object: "
    "{\"name\": <tool name>, \"arguments\": {<param>: <value>, ...}}. "
    "Include only parameters that are needed; do not invent extra parameters. "
    "Ground every argument value in the evidence; use exact original strings for "
    "identifiers, URLs and long values. If a value is not stated in the evidence, "
    "choose the most reasonable value permitted by the schema. Output JSON only."
)


def chunk_session(turns: list[dict], window: int = 6) -> list[str]:
    """Split a session's turns into overlapping-free text chunks of `window` turns."""
    chunks = []
    lines = []
    for t in turns:
        role = t.get("role", "?")
        content = t.get("content", "") or ""
        if t.get("tool_calls"):
            calls = ", ".join(
                f"{c['function']['name']}({c['function'].get('arguments', '')})"
                for c in t["tool_calls"]
            )
            lines.append(f"{role}: {content} [calls: {calls}]")
        elif role == "tool":
            lines.append(f"tool({t.get('name', '')}): {content}")
        else:
            lines.append(f"{role}: {content}")
    for i in range(0, len(lines), window):
        chunks.append("\n".join(lines[i:i + window]))
    return chunks


class HybridRetriever:
    """BM25 + dense retrieval with RRF fusion."""

    def __init__(self, embedder=None, k: int = 5, rrf_k: int = 60):
        self.embedder = embedder
        self.k = k
        self.rrf_k = rrf_k
        self.docs: list[str] = []
        self._doc_ids: list[str] = []
        self._bm25 = None
        self._vecs = None

    def fit(self, docs: list[str], doc_ids: list[str] | None = None):
        from rank_bm25 import BM25Okapi

        self.docs = list(docs)
        self._doc_ids = list(doc_ids) if doc_ids else [str(i) for i in range(len(docs))]
        tokenized = [d.lower().split() for d in self.docs]
        self._bm25 = BM25Okapi(tokenized)
        if self.embedder is not None and self.docs:
            self._vecs = self.embedder.encode(self.docs)

    def search(self, query: str, query_vec=None) -> list[tuple[str, float]]:
        """RRF-fused BM25 + dense search.

        `query_vec` is a pre-encoded dense vector; when omitted and an embedder
        is attached, the query is encoded on the fly.
        """
        bm25_scores = self._bm25.get_scores(query.lower().split())
        bm25_rank = sorted(range(len(self.docs)), key=lambda i: -bm25_scores[i])
        rrf = {}
        for rank, idx in enumerate(bm25_rank[:50]):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        if self._vecs is not None and (query_vec is not None or self.embedder is not None):
            if query_vec is None:
                query_vec = self.embedder.encode([query])[0]
            sims = self._vecs @ query_vec
            dense_rank = sorted(range(len(self.docs)), key=lambda i: -sims[i])
            for rank, idx in enumerate(dense_rank[:50]):
                rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        top = sorted(rrf.items(), key=lambda x: -x[1])[: self.k]
        return [(self.docs[i], s) for i, s in top]


class LTMemoryBaseline:
    """Passive RAG memory: retrieve-then-answer, no memory structuring."""

    def __init__(self, llm: LLMClient, retriever: HybridRetriever):
        self.llm = llm
        self.retriever = retriever
        self.query_vecs: dict = {}   # qa_id -> pre-encoded dense query vector

    def precompute_query_vecs(self, tasks) -> None:
        if self.retriever.embedder is None:
            return
        vecs = self.retriever.embedder.encode([t.query for t in tasks])
        self.query_vecs = {t.qa_id: v for t, v in zip(tasks, vecs)}

    @classmethod
    def build(cls, bench, embedder=None, k: int = 5, chunk_window: int = 6):
        """Build the retrieval index; the LLM is attached afterwards via `llm`."""
        docs, ids = [], []
        for t in bench.tasks:
            for sid, text in zip(t.session_ids, bench.session_texts(t)):
                for j, ch in enumerate(chunk_session(
                        bench.sessions_by_id[sid].turns, window=chunk_window)):
                    docs.append(ch)
                    ids.append(f"{sid}#{j}")
        retriever = HybridRetriever(embedder=embedder, k=k)
        retriever.fit(docs, ids)
        return cls(None, retriever)

    def answer(self, task, session_texts=None) -> tuple[str | None, dict]:
        if not task.session_ids or not self.retriever.docs:
            evidence = "\n".join(session_texts or [])
        else:
            hits = self.retriever.search(task.query, self.query_vecs.get(task.qa_id))
            evidence = "\n---\n".join(h[0] for h in hits)
        user = (
            f"Tool schema:\n{json.dumps(task.tool_schema, ensure_ascii=False, indent=1)}\n\n"
            f"Memory evidence:\n{evidence}\n\n"
            f"User request: {task.query}"
        )
        out = self.llm.chat(SYSTEM_PROMPT, user)
        return parse_tool_call_json(out)
