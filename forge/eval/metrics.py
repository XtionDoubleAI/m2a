"""Metrics for Mem2ActBench, calibrated against the paper (arXiv 2601.19935).

Calibration outcome (see experiments/calibrate_metrics.py, 2026-09-17):
  - Paper's F1 corresponds to token-level P/R/F1 over parameter VALUES only
    (parameter names excluded, case-folded): our run reproduces hybrid@5
    F1=30.68 vs paper 30.7 (Table 4).
  - Paper's "TA" column (Table 3, 87-97 across systems) is arithmetically
    incompatible with its own F1 (~27-36) unless it measures tool selection
    only; our TSA matches it (83.75 vs LTMemory 87.25).
  - We therefore report TSA as the paper-comparable "TA" and additionally a
    strict full-match EM (tool + exact parameter set + exact values) which is
    stricter than anything reported in the paper.

Canonicalization: JSON scalars rendered as strings, booleans as true/false,
values case-folded (calibration choice; URL case errors are rare relative to
the alignment benefit).

Calibration anchors (paper, Qwen2.5-7B): hybrid@5 F1=30.7, LTMemory F1=26.71,
paper-TA(LTMemory)=87.25, oracle F1=53.8.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


def canon_value(v) -> str:
    """Canonical string form of a JSON parameter value (case-folded)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    return str(v).strip().casefold()


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text)


def serialize_values(args: dict) -> list[str]:
    """Token stream over parameter VALUES only (sorted by key for determinism)."""
    out: list[str] = []
    for k in sorted(args):
        out.extend(_tokens(canon_value(args[k])))
    return out


@dataclass
class SampleResult:
    qa_id: str
    pred_tool: str
    gold_tool: str
    pred_args: dict
    gold_args: dict
    tool_correct: bool
    tsa: bool                    # tool selection correct (paper-comparable "TA")
    em: bool                     # strict: tool + exact param set + exact values
    f1: float
    bleu1: float
    slot_acc: float
    slot_detail: dict[str, bool]  # param -> exact value match?


def score_sample(qa_id: str, gold_tool: str, gold_args: dict,
                 pred_tool: str | None, pred_args: dict | None) -> SampleResult:
    pred_args = pred_args or {}
    tool_correct = (pred_tool or "") == gold_tool

    gold_tok = serialize_values(gold_args)
    pred_tok = serialize_values(pred_args)

    from collections import Counter
    pc, gc = Counter(pred_tok), Counter(gold_tok)
    overlap = sum((pc & gc).values())
    precision = overlap / len(pred_tok) if pred_tok else 0.0
    recall = overlap / len(gold_tok) if gold_tok else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    bleu1 = precision

    slot_detail = {
        k: (k in pred_args and canon_value(pred_args[k]) == canon_value(v))
        for k, v in gold_args.items()
    }
    slot_acc = (sum(slot_detail.values()) / len(slot_detail)) if slot_detail else 1.0

    exact_params = (
        set(pred_args.keys()) == set(gold_args.keys())
        and all(canon_value(pred_args[k]) == canon_value(v) for k, v in gold_args.items())
    )
    em = tool_correct and exact_params

    return SampleResult(
        qa_id=qa_id,
        pred_tool=pred_tool or "",
        gold_tool=gold_tool,
        pred_args=pred_args,
        gold_args=gold_args,
        tool_correct=tool_correct,
        tsa=tool_correct,
        em=em,
        f1=f1,
        bleu1=bleu1,
        slot_acc=slot_acc,
        slot_detail=slot_detail,
    )


@dataclass
class Aggregate:
    n: int
    f1: float
    bleu1: float
    tsa: float                 # paper-comparable "TA"
    em: float                  # strict end-to-end exact match
    arg_f1: float              # F1 over samples with correct tool
    arg_n: int
    slot_acc: float


def aggregate(results: list[SampleResult]) -> Aggregate:
    n = len(results)
    if n == 0:
        return Aggregate(0, 0, 0, 0, 0, 0, 0, 0)
    tool_ok = [r for r in results if r.tool_correct]
    return Aggregate(
        n=n,
        f1=sum(r.f1 for r in results) / n,
        bleu1=sum(r.bleu1 for r in results) / n,
        tsa=len(tool_ok) / n,
        em=sum(r.em for r in results) / n,
        arg_f1=(sum(r.f1 for r in tool_ok) / len(tool_ok)) if tool_ok else 0.0,
        arg_n=len(tool_ok),
        slot_acc=sum(r.slot_acc for r in results) / n,
    )


def format_aggregate(agg: Aggregate) -> str:
    return (
        f"n={agg.n}  F1={agg.f1 * 100:.2f}  BLEU1={agg.bleu1 * 100:.2f}  "
        f"TA(TSA)={agg.tsa * 100:.2f}  EM={agg.em * 100:.2f}  "
        f"ArgF1={agg.arg_f1 * 100:.2f} (n={agg.arg_n})  SlotAcc={agg.slot_acc * 100:.2f}"
    )
