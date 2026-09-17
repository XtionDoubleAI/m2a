"""Run the m2a harness over Mem2ActBench.

Two phases:
  1. WRITE (offline, cached): extract fact cards from every evidence session
     (one LLM call per session; cache file resumes across runs)
  2. READ (per task): demand generation -> slot retrieval -> slot-paired
     rendering -> LLM binding -> scoring

Toggles (all default OFF -- MVP first, per the agreed implementation strategy):
  --attribute-match   D2-B: attribute_guess matching as third RRF path
  --collapse-versions C3:   restrict retrieval corpus to each attribute's
                            newest card (version-chain resolution)
  --k                 retrieval depth per slot (default 3)

Usage (WSL):
  CUDA_VISIBLE_DEVICES=1 python -X utf8 -m experiments.run_m2a --limit 20
"""

from __future__ import annotations

import argparse
import json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--attribute-match", action="store_true")
    ap.add_argument("--collapse-versions", action="store_true")
    ap.add_argument("--cards-cache", default=str(OUT_DIR / "fact_cards.jsonl"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from m2a.act.binder import bind
    from m2a.act.embedder import BGEM3Dense
    from m2a.act.intent import generate_demands
    from m2a.act.render import render_evidence
    from m2a.act.retrieve import SlotRetriever
    from m2a.act.llm import OfflineLLM
    from m2a.eval.dataset import Bench
    from m2a.eval.runner import run_system
    from m2a.schema import load_tool_spec
    from m2a.state.extractor import extract_session
    from m2a.state.store import MemoryStore

    suffix = ("_attr" if args.attribute_match else "") + ("_vc" if args.collapse_versions else "")
    args.out = args.out or str(OUT_DIR / f"m2a_mvp{suffix}.jsonl")

    bench = Bench(BENCH_DIR)
    print("bench:", json.dumps(bench.stats(), ensure_ascii=False))

    embedder = BGEM3Dense(str(_snapshot("BAAI/bge-m3")), device="cuda:0")
    llm = OfflineLLM(str(_snapshot("Qwen/Qwen2.5-7B-Instruct")),
                     gpu_memory_utilization=0.75)  # shares the GPU with the embedder

    # ---- phase 1: write path (cached) ----
    cache = Path(args.cards_cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    store = MemoryStore()
    if cache.exists():
        with open(cache, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                store.add_session(d["session_id"], d["cards"])
                done.add(d["session_id"])
    todo = [s for s in bench.sessions if s.session_id not in done]
    print(f"write path: {len(done)} sessions cached, {len(todo)} to extract")
    with open(cache, "a", encoding="utf-8") as f:
        for i, s in enumerate(todo):
            text = "\n".join(
                f"{t.get('role', '?')}: {t.get('content', '')}" for t in s.turns
            )
            cards = extract_session(llm, s.session_id, text)
            store.add_session(s.session_id, cards)
            f.write(json.dumps({"session_id": s.session_id, "cards": cards},
                               ensure_ascii=False) + "\n")
            if (i + 1) % 25 == 0:
                print(f"  extracted {i + 1}/{len(todo)} sessions "
                      f"({sum(len(c) for _, c in [(x, store.cards)] ) } cards total)")

    # ---- phase 2: read path per task ----
    def system(task, session_texts=None):
        spec = load_tool_spec(task.tool_schema)
        corpus = store.cards_for_sessions(task.session_ids)
        if args.collapse_versions:
            corpus = list(store.latest_by_attribute(corpus).values())
        retriever = SlotRetriever(corpus, embedder=embedder, k=args.k,
                                  use_attribute_match=args.attribute_match)
        demands = generate_demands(llm, task.query, spec)
        demand_vecs = embedder.encode([d["query"] for d in demands]) if demands else []
        hits_per_demand = [
            (d, retriever.search(d, vec))
            for d, vec in zip(demands, demand_vecs)
        ]
        evidence = render_evidence(hits_per_demand)
        return bind(llm, spec, task.query, evidence)

    run_system(system, bench, args.out, limit=args.limit)


if __name__ == "__main__":
    main()
