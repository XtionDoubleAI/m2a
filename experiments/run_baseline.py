"""Run the LTMemory baseline over Mem2ActBench (calibration entrypoint).

Usage (after vLLM server is up on :8000 and models are in E:/hx/_models):
  python -m experiments.run_baseline --limit 50          # smoke
  python -m experiments.run_baseline --retriever bm25    # ablation on retriever

Expectation (paper, Qwen2.5-7B): LTMemory F1~26.7 TA~87.3 / hybrid@5 F1~30.7.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BENCH_DIR = Path(r"E:\hx\_DoctorXtion\intern_shxt\Pdev\refs\repos\Mem2ActBench\toolmembench_small")
MODELS_DIR = Path(r"E:\hx\_models")
OUT_DIR = Path(__file__).resolve().parent.parent / "results"


def _latest_snapshot(repo: str) -> Path:
    """Resolve a local model dir from either HF-style or ModelScope-style caches.

    HF-style:      <MODELS_DIR>/<ns>/<name>
    ModelScope:    <MODELS_DIR>/models/<ns>--<name>/snapshots/<hash>
    """
    direct = MODELS_DIR / repo
    if (direct / "config.json").exists():
        return direct
    ms = MODELS_DIR / "models" / repo.replace("/", "--")
    if (ms / "snapshots").exists():
        snaps = sorted((ms / "snapshots").glob("*"))
        if snaps:
            return snaps[-1]
    return direct


def make_embedder(device: str = "cuda:1"):
    from m2a.act.embedder import BGEM3Dense
    return BGEM3Dense(str(_latest_snapshot("BAAI/bge-m3")), device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--retriever", choices=["hybrid", "bm25", "dense", "none"], default="hybrid")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--chunk-window", type=int, default=6)
    ap.add_argument("--out", default=str(OUT_DIR / "ltmemory_small.jsonl"))
    args = ap.parse_args()

    from m2a.act.llm import LLMClient
    from m2a.eval.dataset import Bench
    from m2a.eval.runner import run_system
    from experiments.baselines.ltmemory import LTMemoryBaseline

    bench = Bench(BENCH_DIR)
    print("bench stats:", json.dumps(bench.stats(), ensure_ascii=False))

    embedder = None
    if args.retriever in ("hybrid", "dense"):
        embedder = make_embedder()

    llm = LLMClient()
    system = LTMemoryBaseline.build(llm, bench, embedder=embedder, k=args.k,
                                    chunk_window=args.chunk_window)
    if args.retriever == "bm25":
        system.retriever.embedder = None
    if args.retriever == "none":
        system.retriever.docs = []
        system.retriever._bm25 = None  # noqa: SLF001

    run_system(system.answer, bench, args.out, limit=args.limit)


if __name__ == "__main__":
    main()
