"""Run the FORGE harness over Mem2ActBench.

Store modes (ablation rows for the paper):
  chunks  -- lossless dialogue chunks only. The strongest configuration:
             distillation (facts) is net-negative and is kept only as an
             ablation. Default.
  hybrid  -- fact cards + chunks (ablation: card candidates pollute rendering).
  facts   -- fact cards only (the coverage-funnel ablation).

Components (default off; each is an ablation row):
  R1 --rerank             cross-encoder reranking of the retrieval pool
  R2 --rerank-candidates  rendered evidence ordered by reranker score
  R3 --verbatim-threshold verbatim anchoring: long tokens extracted from the
                          top-ranked chunk become correction anchors; a model
                          value within edit distance 2 of an anchor is snapped
                          to the anchor, character-exact (deterministic)

Protocol switches kept for ablations: --whitelist, --no-demands (D5a),
--flat-render (D5b), --attribute-match, --collapse-versions.

Every run appends a manifest to results/runs.jsonl and dumps per-task
intermediates (demands, hits, rendered evidence text, pre/post-override args).

Usage (WSL): python -X utf8 -m experiments.run_forge --limit 20
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
    """RRF(BM25, dense) over the chunks visible to a task.

    `pool` controls how many first-stage candidates are returned (for R1 the
    runner feeds the pool to the cross-encoder and keeps the top k).
    """

    def __init__(self, visible: list[dict], vecs, k: int = 3, rrf_k: int = 60):
        from rank_bm25 import BM25Okapi

        self.chunks = visible
        self.k, self.rrf_k = k, rrf_k
        self._vecs = vecs
        self._bm25 = (BM25Okapi([re.findall(r"\w+", c["text"].lower()) for c in visible])
                      if visible else None)

    def search(self, query: str, qvec, pool: int | None = None) -> list[tuple[dict, float]]:
        if not self.chunks:
            return []
        pool = pool or self.k
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
        top = sorted(rrf.items(), key=lambda x: -x[1])[: pool]
        return [(self.chunks[i], s) for i, s in top]


_LONG_TOKEN = re.compile(r"[^\s]{20,}")


def _edit_distance(a: str, b: str) -> int:
    if abs(len(a) - len(b)) > 2:
        return 99
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            cur = min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
            prev, dp[j] = dp[j], cur
    return dp[-1]


def _canon(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v).strip().casefold()


def verbatim_anchor(args: dict, anchors_by_param: dict[str, list[str]]) -> dict:
    """R3: snap model values that are near-misses of extracted long tokens to
    the exact anchor string (deterministic; targets copy corruption of
    identifiers/addresses, paper failure mode 3)."""
    out = dict(args)
    for p, anchors in anchors_by_param.items():
        cur = out.get(p)
        if cur is None or not isinstance(cur, str):
            continue
        c = cur.strip()
        if len(c) < 20:
            continue
        for a in anchors:
            if _canon(c) != _canon(a) and _edit_distance(_canon(c), _canon(a)) <= 2:
                out[p] = a
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--bench-dir", default=None,
                    help="override benchmark data dir (e.g. the full 2029-session pool)")
    ap.add_argument("--qa-file", default=None,
                    help="optional file of qa_ids (one per line); run only those tasks")
    ap.add_argument("--llm", choices=["local", "api"], default="local",
                    help="answer-side LLM: local vLLM or hosted OpenAI-compatible API")
    ap.add_argument("--host-model", default="DeepSeek-V4-Flash",
                    help="model name on the API side")
    ap.add_argument("--reasoning", choices=["default", "none", "high"], default="default",
                    help="reasoning_effort for hosted thinking models")
    ap.add_argument("--host-max-tokens", type=int, default=None)
    ap.add_argument("--store-mode", choices=["hybrid", "facts", "chunks"], default="chunks")
    ap.add_argument("--k-cards", type=int, default=3)
    ap.add_argument("--k-chunks", type=int, default=3)
    ap.add_argument("--rerank-pool", type=int, default=50)
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--rerank-candidates", action="store_true")
    ap.add_argument("--verbatim-threshold", type=float, default=0.0)
    ap.add_argument("--attribute-match", action="store_true")
    ap.add_argument("--collapse-versions", action="store_true")
    ap.add_argument("--whitelist", action="store_true")
    ap.add_argument("--no-demands", action="store_true")
    ap.add_argument("--no-demand-fallback", action="store_true",
                    help="disable the deterministic template fallback for empty demand lists; default ON")
    ap.add_argument("--flat-render", action="store_true")
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

    bench = Bench(args.bench_dir or BENCH_DIR)
    print("bench:", json.dumps(bench.stats(), ensure_ascii=False))
    if args.qa_file:
        keep = {l.strip() for l in open(args.qa_file, encoding="utf-8") if l.strip()}
        bench.tasks = [t_ for t_ in bench.tasks if t_.qa_id in keep]
        print(f"qa-file filter: {len(bench.tasks)} tasks kept")

    embedder = BGEM3Dense(str(_snapshot("BAAI/bge-m3")), device=args.embed_device)
    if args.llm == "api":
        from forge.act.llm import make_answer_llm
        llm = make_answer_llm("api", args.host_model, args.reasoning,
                              args.host_max_tokens)
    else:
        llm = OfflineLLM(str(_snapshot("Qwen/Qwen2.5-7B-Instruct")),
                         gpu_memory_utilization=args.gpu_util)
    reranker = None
    if args.rerank or args.rerank_candidates or args.verbatim_threshold > 0:
        from forge.act.reranker import BGEReranker
        reranker = BGEReranker(str(_snapshot("BAAI/bge-reranker-v2-m3")),
                               device=args.embed_device)

    store = MemoryStore()
    if Path(args.cards_cache).exists() and args.store_mode != "chunks":
        with open(args.cards_cache, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                store.add_session(d["session_id"], d["cards"])

    all_chunks = all_vecs = None
    if args.store_mode in ("hybrid", "chunks"):
        all_chunks, all_vecs = build_chunk_index(bench, embedder, args.chunks_cache)
        print(f"chunk index: {len(all_chunks)} chunks")

    dumper = IntermediateDumper(inter_path)

    def system(task, session_texts=None):
        spec = load_tool_spec(task.tool_schema)
        corpus = store.cards_for_sessions(task.session_ids) if args.store_mode != "chunks" else []
        if args.collapse_versions:
            corpus = list(store.latest_by_attribute(corpus).values())
        retriever = SlotRetriever(corpus, embedder=embedder, k=args.k_cards,
                                  use_attribute_match=args.attribute_match)

        searcher = None
        if args.store_mode in ("hybrid", "chunks"):
            want = set(task.session_ids)
            visible = [c for c in all_chunks if c["session_id"] in want]
            keep_idx = [i for i, c in enumerate(all_chunks) if c["session_id"] in want]
            import numpy as np
            searcher = ChunkSearcher(visible, all_vecs[keep_idx] if all_vecs is not None else None,
                                     k=args.k_chunks)

        demands = generate_demands(llm, task.query, spec,
                                  fallback=not args.no_demand_fallback)
        dvecs = embedder.encode([d["query"] for d in demands]) if demands else []

        card_hits, chunk_texts, anchors_by_param = [], [], {}
        chunk_best_score: dict[str, float] = {}

        def take_chunks(query: str, qvec, param_name: str) -> None:
            if searcher is None:
                return
            pool = searcher.search(query, qvec, pool=args.rerank_pool if args.rerank else None)
            if reranker is not None and pool:
                scores = reranker.score([(query, c["text"]) for c, _ in pool])
                pool = sorted(zip(pool, scores), key=lambda x: -x[1])
                top_score = pool[0][1]
                pool = [c for (c, _), _s in pool[: args.k_chunks]]
                chunk_best_score[param_name] = top_score
                if args.verbatim_threshold > 0 and top_score >= args.verbatim_threshold:
                    anchors_by_param[param_name] = _LONG_TOKEN.findall(pool[0]["text"])
            else:
                pool = [c for c, _ in pool[: args.k_chunks]]
            for c in pool:
                if c["text"] not in chunk_texts:
                    chunk_texts.append(c["text"])

        if args.no_demands:
            qv = embedder.encode([task.query])[0]
            for d in demands:
                card_hits.append((d, []))
                take_chunks(task.query, qv, d["param_name"])
        else:
            for d, v in zip(demands, dvecs):
                card_hits.append((d, retriever.search(d, v)))
                take_chunks(d["query"], v, d["param_name"])

        if args.rerank_candidates and chunk_best_score:
            # order rendered blocks by the best reranker score any slot gave them
            def block_score(text: str) -> float:
                return max((reranker.score([(q, text)])[0]
                            for q in [d["query"] for d in demands]), default=0.0)
            chunk_texts.sort(key=block_score, reverse=True)

        if args.flat_render:
            evidence = ("Memory evidence (verbatim dialogue excerpts):\n"
                        + "\n---\n".join(chunk_texts)) if chunk_texts else "Memory evidence: (none found)"
        else:
            evidence = render_evidence(card_hits, chunk_texts=chunk_texts or None)
        trusted = "\n".join(chunk_texts) if chunk_texts else None
        de = card_hits if args.whitelist else None
        pred_tool, final_args, model_args = bind(llm, spec, task.query, evidence,
                                                 demand_evidence=de,
                                                 trusted_texts=trusted)
        if anchors_by_param:
            final_args = verbatim_anchor(final_args, anchors_by_param)

        dumper.dump(task.qa_id, demands,
                    {d["param_name"]: [c["value"] for c, _ in h] for d, h in card_hits},
                    {"n_chunks": len(chunk_texts)},
                    model_args, final_args, task.arguments,
                    extra={"evidence": evidence[:24000], "anchors": anchors_by_param,
                           "rerank_top1": chunk_best_score})
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
