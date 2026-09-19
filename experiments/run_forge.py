"""Run the m2a harness over Mem2ActBench.

Store modes (ablation rows for the paper):
  facts   -- fact-card store only. Kept as the losing ablation: 51% of gold
             values never enter a distilled store (coverage funnel), which
             motivated the hybrid design.
  hybrid  -- fact cards (clean candidate values for binding) PLUS verbatim
             dialogue chunks (lossless coverage). Default.

Every run appends a manifest (config + aggregates) to results/runs.jsonl and
dumps per-task intermediates (demands, hits, pre/post-override args) for
error analysis and ablation figures. See experiments/funnel.py.

Usage (WSL): python -X utf8 -m experiments.run_m2a --limit 20
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

IS_WSL = sys.platform == "linux"
BENCH_DIR = Path(
    "/mnt/e/hx/_DoctorXtion/intern_shxt/Pdev/refs/repos/Mem2ActBench/toolmembench_small"
    if IS_WSL else
    r"E:\hx\_DoctorXtion\intern_shxt\Pdev\refs\repos\Mem2ActBench\toolmembench_small"
)
MODELS_DIR = Path("/mnt/e/hx/_models" if IS_WSL else r"E:\hx\_models")
OUT_DIR = Path(__file__).resolve().parent.parent / "results"


def _snapshot(repo: str) -> Path:
    direct = MODELS_DIR / repo
    if (direct / "config.json").exists():
        return direct
    ms = MODELS_DIR / "models" / repo.replace("/", "--")
    snaps = sorted((ms / "snapshots").glob("*")) if (ms / "snapshots").exists() else []
    return snaps[-1] if snaps else direct


def build_chunk_index(bench, embedder, cache_stem: str):
    """One-time lossless chunk index over all sessions (cached to disk)."""
    import numpy as np

    from experiments.baselines.ltmemory import chunk_session

    meta_path, vec_path = Path(cache_stem + ".jsonl"), Path(cache_stem + ".npy")
    if meta_path.exists() and vec_path.exists():
        chunks = [json.loads(l) for l in open(meta_path, encoding="utf-8")]
        return chunks, np.load(vec_path)
    chunks = []
    for s in bench.sessions:
        for j, text in enumerate(chunk_session(s.turns, window=6)):
            chunks.append({"session_id": s.session_id,
                           "chunk_id": f"{s.session_id}#{j}", "text": text})
    vecs = embedder.encode([c["text"] for c in chunks])
    with open(meta_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    np.save(vec_path, vecs)
    return chunks, vecs


class ChunkSearcher:
    """RRF(BM25, dense) over the chunks visible to a task."""

    def __init__(self, visible: list[dict], vecs, k: int = 3, rrf_k: int = 60):
        from rank_bm25 import BM25Okapi

        self.chunks = visible
        self.k, self.rrf_k = k, rrf_k
        self._vecs = vecs
        self._bm25 = (BM25Okapi([re.findall(r"\w+", c["text"].lower()) for c in visible])
                      if visible else None)

    def search(self, query: str, qvec) -> list[tuple[dict, float]]:
        if not self.chunks:
            return []
        rrf: dict[int, float] = {}
        if self._bm25 is not None:
            scores = self._bm25.get_scores(re.findall(r"\w+", query.lower()))
            for rank, idx in enumerate(sorted(range(len(self.chunks)),
                                              key=lambda i: -scores[i])[:50]):
                rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        if self._vecs is not None and len(self._vecs):
            sims = self._vecs @ qvec
            for rank, idx in enumerate(sorted(range(len(self.chunks)),
                                              key=lambda i: -sims[i])[:50]):
                rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        top = sorted(rrf.items(), key=lambda x: -x[1])[: self.k]
        return [(self.chunks[i], s) for i, s in top]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--store-mode", choices=["hybrid", "facts"], default="hybrid")
    ap.add_argument("--k-cards", type=int, default=3)
    ap.add_argument("--k-chunks", type=int, default=3)
    ap.add_argument("--attribute-match", action="store_true")
    ap.add_argument("--collapse-versions", action="store_true")
    ap.add_argument("--whitelist", action="store_true",
                    help="fabrication-guard override (ablation; net -24 F1-slots in hybrid mode)")
    ap.add_argument("--cards-cache", default=str(OUT_DIR / "fact_cards_v3.jsonl"))
    ap.add_argument("--chunks-cache", default=str(OUT_DIR / "chunks"))
    ap.add_argument("--embed-device", default="cuda:1")
    ap.add_argument("--gpu-util", type=float, default=0.92)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    name = args.name or f"forge-{args.store_mode}-store"
    out_path = OUT_DIR / f"{name}.jsonl"
    inter_path = OUT_DIR / f"{name}.intermediates.jsonl"

    from forge.act.binder import bind
    from forge.act.embedder import BGEM3Dense
    from forge.act.intent import generate_demands
    from forge.act.llm import OfflineLLM
    from forge.act.render import render_evidence
    from forge.act.retrieve import SlotRetriever
    from forge.eval.dataset import Bench
    from forge.eval.metrics import aggregate
    from forge.eval.runner import run_system
    from forge.schema import load_tool_spec
    from forge.state.store import MemoryStore
    from experiments.runlog import IntermediateDumper, log_run

    bench = Bench(BENCH_DIR)
    print("bench:", json.dumps(bench.stats(), ensure_ascii=False))

    embedder = BGEM3Dense(str(_snapshot("BAAI/bge-m3")), device=args.embed_device)
    llm = OfflineLLM(str(_snapshot("Qwen/Qwen2.5-7B-Instruct")),
                     gpu_memory_utilization=args.gpu_util)

    # write path (cached fact cards)
    store = MemoryStore()
    if Path(args.cards_cache).exists():
        with open(args.cards_cache, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                store.add_session(d["session_id"], d["cards"])
    print(f"fact cards: {len(store.cards)} cards from {len(set(c['session_id'] for c in store.cards))} sessions")

    # lossless chunk index (hybrid mode)
    all_chunks = all_vecs = None
    if args.store_mode == "hybrid":
        all_chunks, all_vecs = build_chunk_index(bench, embedder, args.chunks_cache)
        print(f"chunk index: {len(all_chunks)} chunks")

    dumper = IntermediateDumper(inter_path)

    def system(task, session_texts=None):
        spec = load_tool_spec(task.tool_schema)
        corpus = store.cards_for_sessions(task.session_ids)
        if args.collapse_versions:
            corpus = list(store.latest_by_attribute(corpus).values())
        retriever = SlotRetriever(corpus, embedder=embedder, k=args.k_cards,
                                  use_attribute_match=args.attribute_match)

        searcher = None
        if args.store_mode == "hybrid":
            want = set(task.session_ids)
            visible = [c for c in all_chunks if c["session_id"] in want]
            keep_idx = [i for i, c in enumerate(all_chunks) if c["session_id"] in want]
            import numpy as np
            searcher = ChunkSearcher(visible, all_vecs[keep_idx] if all_vecs is not None else None,
                                     k=args.k_chunks)

        demands = generate_demands(llm, task.query, spec)
        dvecs = embedder.encode([d["query"] for d in demands]) if demands else []

        card_hits, chunk_texts = [], []
        for d, v in zip(demands, dvecs):
            card_hits.append((d, retriever.search(d, v)))
            if searcher is not None:
                for c, _s in searcher.search(d["query"], v):
                    if c["text"] not in chunk_texts:
                        chunk_texts.append(c["text"])

        evidence = render_evidence(card_hits, chunk_texts=chunk_texts or None)
        trusted = "\n".join(chunk_texts) if chunk_texts else None
        de = card_hits if args.whitelist else None
        pred_tool, final_args, model_args = bind(llm, spec, task.query, evidence,
                                                 demand_evidence=de,
                                                 trusted_texts=trusted)
        dumper.dump(task.qa_id, demands,
                    {d["param_name"]: [c["value"] for c, _ in h] for d, h in card_hits},
                    {"n_chunks": len(chunk_texts)},
                    model_args, final_args, task.arguments)
        return pred_tool, final_args

    results = run_system(system, bench, out_path, limit=args.limit)
    dumper.close()

    aggs = aggregate(results)
    log_run(name, config={**vars(args), "name": name},
            aggregates={"f1": round(aggs.f1 * 100, 2), "bleu1": round(aggs.bleu1 * 100, 2),
                        "tsa": round(aggs.tsa * 100, 2), "em": round(aggs.em * 100, 2),
                        "arg_f1": round(aggs.arg_f1 * 100, 2),
                        "slot_acc": round(aggs.slot_acc * 100, 2)},
            files={"samples": str(out_path), "intermediates": str(inter_path)},
            runs_path=OUT_DIR / "runs.jsonl")


if __name__ == "__main__":
    main()
