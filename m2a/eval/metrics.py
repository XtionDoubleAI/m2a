"""Metrics for Mem2ActBench, aligned with the paper (arXiv 2601.19935, §3.5/§4.1/§5.4).

Definitions:
  F1       -- parameter-level token F1: predicted and gold arguments are each
              serialized as "param value" token streams (SQuAD-style normalization),
              then micro-averaged precision/recall over tokens.
  BLEU-1   -- unigram precision on the same serialization.
  TA       -- Tool Accuracy: correct tool name AND every parameter exactly matches
              (value comparison after canonicalization; no extra/missing params).
  TSA      -- Tool Selection Accuracy: correct tool name only.
  EM       -- end-to-end exact match (identical criterion as TA here).
  Arg_F1   -- F1 computed over the subset where the tool was selected correctly.
  Slot Acc -- fraction of gold parameters whose value exactly matches the prediction.

Canonicalization notes (calibration knobs, see Pdev/decisions):
  - JSON scalars are rendered as strings; booleans as "true"/"false".
  - Value comparison is case-sensitive (URLs/IDs demand lossless retention).

Calibration anchor (paper Table 3, Qwen2.5-7B, LTMemory): F1=26.71, BLEU=64.07, TA=87.25.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


def canon_value(v) -> str:
    """Canonical string form of a JSON parameter value."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    return str(v).strip()


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def serialize_args(args: dict) -> str:
    """Deterministic serialization of an arguments dict into a token stream."""
    return " ".join(f"{k} {canon_value(v)}" for k, v in sorted(args.items()))


@dataclass
class SampleResult:
    qa_id: str
    pred_tool: str
    gold_tool: str
    pred_args: dict
    gold_args: dict
    tool_correct: bool
    ta: bool
    f1: float
    bleu1: float
    slot_acc: float
    slot_detail: dict[str, bool]          # param -> exact match?


def score_sample(qa_id: str, gold_tool: str, gold_args: dict,
                 pred_tool: str | None, pred_args: dict | None) -> SampleResult:
    pred_args = pred_args or {}
    tool_correct = (pred_tool or "") == gold_tool

    gold_tok = _tokens(serialize_args(gold_args))
    pred_tok = _tokens(serialize_args(pred_args))

    pred_counts, gold_counts = {}, {}
    for t in pred_tok:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    for t in gold_tok:
        gold_counts[t] = gold_counts.get(t, 0) + 1
    overlap = 0
    for t, c in pred_counts.items():
        if t in gold_counts:
            overlap += min(c, gold_counts[t])

    precision = overlap / len(pred_tok) if pred_tok else 0.0
    recall = overlap / len(gold_tok) if gold_tok else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    bleu1 = precision

    slot_detail = {
        k: canon_value(pred_args.get(k)) == canon_value(v) and k in pred_args
        for k, v in gold_args.items()
    }
    slot_acc = (sum(slot_detail.values()) / len(slot_detail)) if slot_detail else 1.0

    exact_params = (
        set(pred_args.keys()) == set(gold_args.keys())
        and all(canon_value(pred_args[k]) == canon_value(v) for k, v in gold_args.items())
    )
    ta = tool_correct and exact_params

    return SampleResult(
        qa_id=qa_id,
        pred_tool=pred_tool or "",
        gold_tool=gold_tool,
        pred_args=pred_args,
        gold_args=gold_args,
        tool_correct=tool_correct,
        ta=ta,
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
    ta: float
    tsa: float
    em: float
    arg_f1: float
    arg_n: int          # samples with correct tool (denominator of Arg_F1)
    slot_acc: float


def aggregate(results: list[SampleResult]) -> Aggregate:
    n = len(results)
    if n == 0:
        return Aggregate(0, 0, 0, 0, 0, 0, 0, 0, 0)
    tool_ok = [r for r in results if r.tool_correct]
    return Aggregate(
        n=n,
        f1=sum(r.f1 for r in results) / n,
        bleu1=sum(r.bleu1 for r in results) / n,
        ta=sum(r.ta for r in results) / n,
        tsa=len(tool_ok) / n,
        em=sum(r.ta for r in results) / n,
        arg_f1=(sum(r.f1 for r in tool_ok) / len(tool_ok)) if tool_ok else 0.0,
        arg_n=len(tool_ok),
        slot_acc=sum(r.slot_acc for r in results) / n,
    )


def format_aggregate(agg: Aggregate) -> str:
    return (
        f"n={agg.n}  F1={agg.f1 * 100:.2f}  BLEU1={agg.bleu1 * 100:.2f}  "
        f"TA={agg.ta * 100:.2f}  TSA={agg.tsa * 100:.2f}  EM={agg.em * 100:.2f}  "
        f"ArgF1={agg.arg_f1 * 100:.2f} (n={agg.arg_n})  SlotAcc={agg.slot_acc * 100:.2f}"
    )
