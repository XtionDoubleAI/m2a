"""Experiment runner: execute a system over Bench tasks and score.

A "system" is any callable (task, session_texts) -> (tool_name, arguments).
This keeps baselines and the m2a harness behind one interface.
"""

from __future__ import annotations

import json
from pathlib import Path

from m2a.eval.dataset import Bench, QATask
from m2a.eval.metrics import SampleResult, aggregate, format_aggregate, score_sample


def run_system(system, bench: Bench, out_path: str | Path,
               limit: int | None = None, log_every: int = 50) -> list[SampleResult]:
    results = []
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tasks = bench.tasks[:limit] if limit else bench.tasks
    with open(out_path, "w", encoding="utf-8") as f:
        for i, task in enumerate(tasks):
            texts = bench.session_texts(task)
            try:
                pred_tool, pred_args = system(task, texts)
            except Exception as e:  # noqa: BLE001 - log and continue, one bad sample must not kill a run
                print(f"[{task.qa_id}] system error: {e}")
                pred_tool, pred_args = None, {}
            r = score_sample(task.qa_id, task.tool_name, task.arguments,
                             pred_tool, pred_args)
            results.append(r)
            f.write(json.dumps({
                "qa_id": task.qa_id,
                "pred_tool": r.pred_tool,
                "gold_tool": r.gold_tool,
                "pred_args": r.pred_args,
                "gold_args": r.gold_args,
                "tool_correct": r.tool_correct,
                "tsa": r.tsa,
                "em": r.em,
                "f1": r.f1,
                "bleu1": r.bleu1,
                "slot_acc": r.slot_acc,
                "level": task.complexity.get("level"),
                "has_session": bool(task.session_ids),
            }, ensure_ascii=False) + "\n")
            if (i + 1) % log_every == 0:
                print(f"  {i + 1}/{len(tasks)}  {format_aggregate(aggregate(results))}")
    print(f"final: {format_aggregate(aggregate(results))}")
    print(f"per-sample log: {out_path}")
    return results


def report_by(results: list[SampleResult], levels: dict[str, str]) -> dict:
    """Break down by complexity level (qa_id -> level) and by session coverage."""
    out = {}
    for lv in sorted(set(levels.values())):
        sub = [r for r in results if levels.get(r.qa_id) == lv]
        if sub:
            out[lv] = format_aggregate(aggregate(sub))
    return out
