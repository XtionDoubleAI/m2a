"""C3 diagnostic: oracle evidence supply (paper protocol reproduction) with
per-parameter stratification.

Reproduces the paper's oracle condition (Table 4: F1 53.8): the evidence
rendered to the model is the task's gold memory chain (evolution_chain
source_text -- the original utterances the gold values come from), the
answer side is the shared harness (same Qwen2.5-7B, same prompt, same
parser, same scorer). No retrieval, no memory layer: this isolates the
model's copy/derive ability once supply is guaranteed.

Readings (pre-registered before running):
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


FILLER = ("Historical dialogue excerpt (contextual background, unrelated to the "
          "request): the assistant discussed general topics with the user, "
          "including tool usage conventions, account settings and routine "
          "questions exchanged over previous sessions. " * 6)


def injected_evidence(task, position: str) -> str:
    """Pure transcription probe (C3 variant, 2c/2d): state each gold value
    explicitly as one line per parameter, placed among filler context at
    the requested position. The filler carries no parameter information."""
    lines = [f"Parameter `{p}` value: {g}"
             for p, g in task.arguments.items()]
    block = "\n".join(lines)
    head, tail = FILLER[: len(FILLER) // 2], FILLER[len(FILLER) // 2:]
    if position == "first":
        return block + "\n\n" + head + tail
    if position == "middle":
        return head + "\n\n" + block + "\n\n" + tail
    return head + tail + "\n\n" + block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="oracle-supply")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--llm", choices=["local", "api"], default="local",
                    help="answer-side LLM: local vLLM or hosted OpenAI-compatible API")
    ap.add_argument("--host-model", default="DeepSeek-V4-Flash",
                    help="model name on the API side")
    ap.add_argument("--reasoning", choices=["default", "none", "high"], default="default",
                    help="reasoning_effort for hosted thinking models")
    ap.add_argument("--host-max-tokens", type=int, default=None)
    ap.add_argument("--inject-values", choices=["first", "middle", "last"],
                    default=None,
                    help="C3 variant: inject gold values verbatim (one line per "
                         "parameter) at the given position among neutral filler")
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    args = ap.parse_args()

    from experiments.run_forge import _snapshot
    bench = Bench(BENCH_DIR)

    if args.llm == "api":
        from forge.act.llm import make_answer_llm
        llm = make_answer_llm("api", args.host_model, args.reasoning,
                              args.host_max_tokens)
    else:
        llm = OfflineLLM(str(_snapshot(args.model)),
                         gpu_memory_utilization=args.gpu_util)

    inter_path = RESULTS / f"{args.name}.intermediates.jsonl"
    inter_f = open(inter_path, "w", encoding="utf-8")

    def system(task, session_texts=None):
        evidence = (injected_evidence(task, args.inject_values)
                    if args.inject_values else chain_evidence(task))
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
        "protocol": ("gold-value injection at position " + args.inject_values
                     if args.inject_values else
                     "oracle evidence supply (evolution_chain source_text)"),
        "model": args.model, "limit": args.limit,
        "inject_values": args.inject_values,
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
