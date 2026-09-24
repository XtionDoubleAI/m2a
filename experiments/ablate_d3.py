"""Offline ablations for binding-contract details (no GPU, no LLM).

Base configuration: hybrid store WITHOUT whitelist (the final FORGE), whose
per-parameter model outputs are archived in the run's intermediates. Each
mechanism is applied on top of that base and rescored:

  unique-direct   -- parameter has exactly one type-compatible card candidate:
                     adopt it, overriding the model
  schema-default  -- parameter has no card candidates AND the task retrieved
                     no chunks at all (approximation: global emptiness; per-slot
                     chunk hits were not archived for this run): fill the
                     schema default if declared
  enum-snap       -- parameter declares an enum and the model's value is
                     outside it: snap to the nearest enum member (edit distance)

All results are appended to results/runs.jsonl with their config, including
the approximation flag where relevant.

Usage: python -X utf8 -m experiments.ablate_d3
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forge.eval.dataset import Bench  # noqa: E402
from forge.eval.metrics import aggregate, score_sample  # noqa: E402
from forge.schema import load_tool_spec  # noqa: E402
from experiments.runlog import log_run  # noqa: E402

IS_WSL = sys.platform == "linux"
BENCH_DIR = Path(
    "/mnt/e/hx/_DoctorXtion/intern_shxt/Pdev/refs/repos/Mem2ActBench/toolmembench_small"
    if IS_WSL else
    r"E:\hx\_DoctorXtion\intern_shxt\Pdev\refs\repos\Mem2ActBench\toolmembench_small"
)
RESULTS = Path(__file__).resolve().parent.parent / "results"
BASE_RUN = "m2a-hybrid-store"   # archived run carrying model_args pre-override


def _edit_distance(a: str, b: str) -> int:
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            cur = min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
            prev, dp[j] = dp[j], cur
    return dp[-1]


def _coerce(value: str, ptype: str):
    v = str(value).strip()
    if ptype == "string":
        return v
    if ptype in ("integer", "number"):
        try:
            return int(v) if ptype == "integer" else float(v)
        except ValueError:
            return None
    if ptype == "boolean":
        if v.lower() in ("true", "yes"):
            return True
        if v.lower() in ("false", "no"):
            return False
        return None
    return None


def canon(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v).strip().casefold()


def apply_unique_direct(args: dict, spec, card_hits: dict) -> dict:
    out = dict(args)
    for p_name, values in card_hits.items():
        p = spec.param(p_name)
        if p is None or len(values) != 1:
            continue
        v = _coerce(values[0], p.type)
        if v is not None:
            out[p_name] = v
    return out


def apply_schema_default(args: dict, spec, card_hits: dict, n_chunks: int) -> dict:
    out = dict(args)
    if n_chunks:
        return out  # approximation: only fill when nothing was retrieved at all
    for p in spec.params:
        if card_hits.get(p.name) or not p.has_default:
            continue
        if p.name not in out or out[p.name] in (None, "", [], {}):
            out[p.name] = p.default
    return out


def apply_enum_snap(args: dict, spec) -> dict:
    out = dict(args)
    for p in spec.params:
        if not p.enum or p.name not in out:
            continue
        cur = canon(out[p.name])
        if cur in {canon(e) for e in p.enum}:
            continue
        best = min(p.enum, key=lambda e: _edit_distance(canon(e), cur))
        out[p.name] = best
    return out


def main():
    bench = Bench(BENCH_DIR)
    inter = {}
    for line in open(RESULTS / f"{BASE_RUN}.intermediates.jsonl", encoding="utf-8"):
        d = json.loads(line)
        inter[d["qa_id"]] = d
    samples = [json.loads(l) for l in open(RESULTS / f"{BASE_RUN}.jsonl", encoding="utf-8")]

    variants = {
        "base": lambda a, spec, ch, n: a,
        "unique-direct": lambda a, spec, ch, n: apply_unique_direct(a, spec, ch),
        "schema-default": lambda a, spec, ch, n: apply_schema_default(a, spec, ch, n),
        "enum-snap": lambda a, spec, ch, n: apply_enum_snap(a, spec),
    }

    for name, fn in variants.items():
        results = []
        for s in samples:
            m = inter.get(s["qa_id"])
            args = dict(m["model_args"]) if m else s["pred_args"]
            spec = load_tool_spec(
                next(t.tool_schema for t in bench.tasks if t.qa_id == s["qa_id"]))
            ch = m["card_hits"] if m else {}
            n = m["chunk_hits"].get("n_chunks", 0) if m else 0
            args = fn(args, spec, ch, n)
            results.append(score_sample(s["qa_id"], s["gold_tool"], s["gold_args"],
                                        s["pred_tool"], args))
        agg = aggregate(results)
        log_run(f"forge-hybrid-{name}" if name != "base" else "forge-hybrid-base",
                config={"store_mode": "hybrid", "whitelist": False, "d3": name,
                        "derived_from": f"{BASE_RUN} intermediates (offline)",
                        "schema_default_approx": name == "schema-default"},
                aggregates={"f1": round(agg.f1 * 100, 2), "bleu1": round(agg.bleu1 * 100, 2),
                            "tsa": round(agg.tsa * 100, 2), "em": round(agg.em * 100, 2),
                            "arg_f1": round(agg.arg_f1 * 100, 2),
                            "slot_acc": round(agg.slot_acc * 100, 2)},
                files={"source": str(RESULTS / f"{BASE_RUN}.intermediates.jsonl")},
                runs_path=RESULTS / "runs.jsonl")
        print(f"{name:>15}: F1={agg.f1*100:.2f}  SlotAcc={agg.slot_acc*100:.2f}  EM={agg.em*100:.2f}")


if __name__ == "__main__":
    main()
