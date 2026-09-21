"""C4 diagnostic: full-supply control group (retrieval replaced by identity).

Renders the task's ENTIRE visible conversations verbatim into the Memory
evidence segment; everything else (model, prompt template, parser, scorer)
is the shared harness. Completes the W x R matrix: full-vs-passive isolates
the completeness component of the retrieval gap, oracle-vs-full the purity
component (see D8 section 3.4 for the pre-registered readouts).

Truncation rule (pre-registered): tasks whose full evidence exceeds 16,384
tokens keep the first 16,384 tokens in order; the rest is dropped and the
task is reported as truncated. 16 tasks have zero visible sessions by
dataset construction and get an empty evidence segment.

Usage (WSL, GPU0 free): python -X utf8 -m experiments.run_full_supply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forge.act.llm import OfflineLLM, parse_tool_call_json  # noqa: E402
from forge.eval.dataset import Bench  # noqa: E402
from forge.eval.metrics import aggregate, format_aggregate  # noqa: E402
from forge.eval.runner import run_system  # noqa: E402
from experiments.baselines.ltmemory import SYSTEM_PROMPT  # noqa: E402
from experiments.runlog import log_run  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
BENCH_DIR = Path(
    "/mnt/e/hx/_DoctorXtion/intern_shxt/Pdev/refs/repos/Mem2ActBench/toolmembench_small"
    if sys.platform == "linux" else
    r"E:\hx\_DoctorXtion\intern_shxt\Pdev\refs\repos\Mem2ActBench\toolmembench_small"
)

TOKEN_LIMIT = 16384


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="full-supply")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    args = ap.parse_args()

    from experiments.run_forge import _snapshot
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(_snapshot(args.model)))
    bench = Bench(BENCH_DIR)

    llm = OfflineLLM(str(_snapshot(args.model)),
                     gpu_memory_utilization=args.gpu_util)

    inter_path = RESULTS / f"{args.name}.intermediates.jsonl"
    inter_f = open(inter_path, "w", encoding="utf-8")
    n_truncated = 0

    def system(task, session_texts=None):
        nonlocal n_truncated
        full = "\n\n".join(bench.session_texts(task))
        ids = tok(full)["input_ids"]
        if len(ids) > TOKEN_LIMIT:
            full = tok.decode(ids[:TOKEN_LIMIT])
            n_truncated += 1
        user = (
            f"Tool schema:\n{json.dumps(task.tool_schema, ensure_ascii=False, indent=1)}\n\n"
            f"Memory evidence:\n{full}\n\n"
            f"User request: {task.query}"
        )
        out = llm.chat(SYSTEM_PROMPT, user)
        pred_tool, pred_args = parse_tool_call_json(out)
        inter_f.write(json.dumps({
            "qa_id": task.qa_id,
            "evidence": full,
            "evidence_tokens": len(ids),
            "truncated": len(ids) > TOKEN_LIMIT,
            "pred_args": pred_args,
            "gold_args": task.arguments,
            "pred_tool": pred_tool,
        }, ensure_ascii=False) + "\n")
        return pred_tool, pred_args

    results = run_system(system, bench, RESULTS / f"{args.name}.jsonl",
                         limit=args.limit)
    inter_f.close()
    agg = aggregate(results)
    print(format_aggregate(agg))
    print(f"truncated tasks: {n_truncated}")
    log_run(args.name, {
        "protocol": "full visible-conversation supply (R = identity)",
        "model": args.model, "limit": args.limit,
        "token_limit": TOKEN_LIMIT,
        "prompt": "baselines/ltmemory.SYSTEM_PROMPT (shared harness)",
    }, {
        "f1": round(agg.f1 * 100, 2), "bleu1": round(agg.bleu1 * 100, 2),
        "tsa": round(agg.tsa * 100, 2), "em": round(agg.em * 100, 2),
        "slot_acc": round(agg.slot_acc * 100, 2),
    }, {"samples": str(RESULTS / f"{args.name}.jsonl"),
        "intermediates": str(inter_path)},
        RESULTS / "runs.jsonl")


if __name__ == "__main__":
    main()
