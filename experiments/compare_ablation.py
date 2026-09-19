"""Pairwise multi-metric comparison of two archived FORGE runs (offline).

Reads two .intermediates.jsonl archives (the R5 component archived each
task's rendered evidence verbatim), computes the multi-dimensional metrics
via experiments/multimetrics.analyse, and prints the diff row by row.

Usage: python -X utf8 -m experiments.compare_ablation A B
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.multimetrics import RESULTS, analyse  # noqa: E402
from forge.eval.dataset import Bench  # noqa: E402

IS_WSL = sys.platform == "linux"
BENCH_DIR = Path(
    "/mnt/e/hx/_DoctorXtion/intern_shxt/Pdev/refs/repos/Mem2ActBench/toolmembench_small"
    if IS_WSL else
    r"E:\hx\_DoctorXtion\intern_shxt\Pdev\refs\repos\Mem2ActBench\toolmembench_small"
)


def load(name: str):
    """-> (per-task final args, per-task evidence blob)."""
    args, ev = {}, {}
    for line in open(RESULTS / f"{name}.intermediates.jsonl", encoding="utf-8"):
        d = json.loads(line)
        args[d["qa_id"]] = d["final_args"]
        ev[d["qa_id"]] = d.get("evidence", "")
    return args, ev


def main():
    a_name, b_name = sys.argv[1], sys.argv[2]
    bench = Bench(BENCH_DIR)
    f1s = {}
    for line in open(RESULTS / "runs.jsonl", encoding="utf-8"):
        d = json.loads(line)
        if d["name"] in (a_name, b_name):
            f1s[d["name"]] = d["aggregates"]["f1"]
    out = []
    for name in (a_name, b_name):
        args, ev = load(name)
        row = analyse(name, args, ev, bench)
        row["f1"] = f1s.get(name)
        out.append(row)
    (RESULTS / f"cmp_{a_name}__{b_name}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    keys = list(out[0])
    print(f"{'metric':32s} {a_name:>28s} {b_name:>28s}")
    for k in keys[1:]:
        print(f"{k:32s} {out[0][k]:>28} {out[1][k]:>28}")


if __name__ == "__main__":
    main()
