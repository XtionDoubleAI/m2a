"""Run the LTMemory baseline over Mem2ActBench (calibration entrypoint).

Usage (after vLLM server is up on :8000 and models are in E:/hx/_models):
  python -m experiments.run_baseline --limit 50          # smoke
  python -m experiments.run_baseline --retriever bm25    # ablation on retriever

Expectation (paper, Qwen2.5-7B): LTMemory F1~26.7 TA~87.3 / hybrid@5 F1~30.7.
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


def make_embedder(device: str = "cuda:0"):
    """Under CUDA_VISIBLE_DEVICES=<single gpu> the visible device is always cuda:0.

    The embedder encodes the corpus first and is released before the vLLM
    engine claims the same GPU (7B + BGE-M3 do not fit together in 24 GB).
    """
    from forge.act.embedder import BGEM3Dense
    return BGEM3Dense(str(_latest_snapshot("BAAI/bge-m3")), device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--bench-dir", default=None,
                    help="override benchmark data dir (e.g. the full 2029-session pool)")
    ap.add_argument("--name", default="ltmemory_hybrid5_full")
    ap.add_argument("--llm", choices=["local", "api"], default="local",
                    help="answer-side LLM: local vLLM or hosted OpenAI-compatible API")
    ap.add_argument("--host-model", default="DeepSeek-V4-Flash",
                    help="model name on the API side")
    ap.add_argument("--reasoning", choices=["default", "none", "high"], default="default",
                    help="reasoning_effort for hosted thinking models")
    ap.add_argument("--host-max-tokens", type=int, default=None)
    ap.add_argument("--retriever", choices=["hybrid", "bm25", "dense", "none"], default="hybrid")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--chunk-window", type=int, default=6)
    ap.add_argument("--out", default=str(OUT_DIR / "ltmemory_small.jsonl"))
    args = ap.parse_args()

    from forge.act.llm import OfflineLLM
    from forge.eval.dataset import Bench
    from forge.eval.runner import run_system
    from experiments.baselines.ltmemory import LTMemoryBaseline

    bench = Bench(args.bench_dir or BENCH_DIR)
    print("bench stats:", json.dumps(bench.stats(), ensure_ascii=False))

    embedder = None
    if args.retriever in ("hybrid", "dense"):
        embedder = make_embedder()

    system = LTMemoryBaseline.build(bench, embedder=embedder, k=args.k,
                                    chunk_window=args.chunk_window)
    if embedder is not None:
        system.precompute_query_vecs(bench.tasks)
        import gc
        import torch
        embedder.model.cpu()
        del embedder
        system.retriever.embedder = None
        gc.collect()
        torch.cuda.empty_cache()

    if args.llm == "api":
        from forge.act.llm import make_answer_llm
        system.llm = make_answer_llm("api", args.host_model, args.reasoning,
                                     args.host_max_tokens)
    else:
        system.llm = OfflineLLM(str(_latest_snapshot("Qwen/Qwen2.5-7B-Instruct")))
    if args.retriever == "none":
        system.retriever.docs = []
        system.retriever._bm25 = None  # noqa: SLF001

    run_system(system.answer, bench, args.out, limit=args.limit)


if __name__ == "__main__":
    main()
