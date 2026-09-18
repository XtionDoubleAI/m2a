"""Funnel analysis: coverage -> retrieval hit -> binding success.

Paper-grade attribution of where parameter values die. Reads the per-sample
result JSONL (plus the intermediate dump when present) and prints/saves the
funnel that motivated the hybrid store design (fact-only store loses 51% of
gold values before retrieval even runs).

Usage: python -m experiments.funnel results/m2a_hybrid_store.jsonl \
          [--cards-cache results/fact_cards_v3.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from m2a.eval.dataset import Bench  # noqa: E402

IS_WSL = sys.platform == "linux"
BENCH_DIR = Path(
    "/mnt/e/hx/_DoctorXtion/intern_shxt/Pdev/refs/repos/Mem2ActBench/toolmembench_small"
    if IS_WSL else
    r"E:\hx\_DoctorXtion\intern_shxt\Pdev\refs\repos\Mem2ActBench\toolmembench_small"
)


def canon(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v).strip().casefold()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_jsonl")
    ap.add_argument("--cards-cache", default=str(Path(__file__).parent.parent / "results/fact_cards_v3.jsonl"))
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    bench = Bench(BENCH_DIR)
    cards_by_sid: dict[str, list] = {}
    if Path(args.cards_cache).exists():
        for line in open(args.cards_cache, encoding="utf-8"):
            d = json.loads(line)
            cards_by_sid[d["session_id"]] = d["cards"]

    rows = {}
    for line in open(args.results_jsonl, encoding="utf-8"):
        d = json.loads(line)
        rows[d["qa_id"]] = d

    n = covered = slot_ok = ok_given_covered = 0
    for t in bench.tasks:
        cards = [c for sid in t.session_ids for c in cards_by_sid.get(sid, [])]
        blob = " ".join((str(c.get("value", "")) + " " + str(c.get("source_text", ""))).casefold()
                        for c in cards)
        pred = rows.get(t.qa_id, {}).get("pred_args", {})
        for p, g in t.arguments.items():
            n += 1
            cov = canon(g) in blob
            covered += cov
            ok = canon(pred.get(p)) == canon(g)
            slot_ok += ok
            ok_given_covered += (cov and ok)

    out = {
        "results": args.results_jsonl,
        "params": n,
        "in_fact_store": covered,
        "in_fact_store_pct": round(100 * covered / max(n, 1), 1),
        "slot_correct": slot_ok,
        "slot_correct_pct": round(100 * slot_ok / max(n, 1), 1),
        "correct_given_covered_pct": round(100 * ok_given_covered / max(covered, 1), 1),
    }
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if args.save:
        Path(args.save).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
