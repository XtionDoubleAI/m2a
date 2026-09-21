"""C3 diagnostic: oracle evidence supply (paper protocol reproduction) with
per-parameter stratification.

Reproduces the paper's oracle condition (Table 4: F1 53.8): the evidence
rendered to the model is the task's gold memory chain (evolution_chain
source_text -- the original utterances the gold values come from), the
answer side is the shared harness (same Qwen2.5-7B, same prompt, same
parser, same scorer). No retrieval, no memory layer: this isolates the
model's copy/derive ability once supply is guaranteed.

Readings (pre-registered in dev_log 2026-09-21):
  - total F1 vs paper's 53.8 calibrates our protocol reading of "oracle";
  - exact-copy rate per stratum (long values, grounding type) measures how
    often the model transcribes a supplied value exactly -- the clean half
    of the long-value either/or (supply vs transcription);
  - for the 31% of gold values absent from visible sessions, the chain
    carries them or not -- measured, not assumed.

Usage (WSL, GPU0 free): python -X utf8 -m experiments.run_oracle_probe
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


def chain_evidence(task) -> str:
    parts = []
    for item in task.evolution_chain:
        st = (item.get("source_text") or "").strip()
        if st:
            parts.append(st)
    return "\n---\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="oracle-supply")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    args = ap.parse_args()

    from experiments.run_forge import _snapshot
    bench = Bench(BENCH_DIR)

    llm = OfflineLLM(str(_snapshot(args.model)),
                     gpu_memory_utilization=args.gpu_util)

    inter_path = RESULTS / f"{args.name}.intermediates.jsonl"
    inter_f = open(inter_path, "w", encoding="utf-8")

    def system(task, session_texts=None):
        evidence = chain_evidence(task)
        user = (
            f"Tool schema:\n{json.dumps(task.tool_schema, ensure_ascii=False, indent=1)}\n\n"
            f"Memory evidence:\n{evidence}\n\n"
            f"User request: {task.query}"
        )
        out = llm.chat(SYSTEM_PROMPT, user)
        pred_tool, pred_args = parse_tool_call_json(out)
        inter_f.write(json.dumps({
            "qa_id": task.qa_id,
            "evidence": evidence,
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
    log_run(args.name, {
        "protocol": "oracle evidence supply (evolution_chain source_text)",
        "model": args.model, "limit": args.limit,
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
