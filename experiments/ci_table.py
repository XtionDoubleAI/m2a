"""Paired-bootstrap CI table for every FORGE variant vs the full configuration.

For each archived run (unified value-only F1 recomputed from pred/gold args,
never the legacy jsonl f1 column) computes the paired mean difference against
forge-chunks-full with a 10k-resample percentile interval. Output:
results/ci_table.json -- consumed by plots and the paper tables.

Usage: python -X utf8 -m experiments.ci_table
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forge.eval.metrics import score_sample  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
REFERENCE = "forge-chunks-full"
VARIANTS = [
    "forge-chunks-nodemand", "forge-chunks-flat", "forge-chunks-r1",
    "forge-chunks-r1r2", "forge-chunks-r1r2r3", "forge-chunks-r1r2r3-low",
]
N_BOOT = 10000


def value_f1(path: Path) -> dict:
    """qa_id -> recomputed value-only per-sample F1 (casefolded)."""
    out = {}
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        r = score_sample(d["qa_id"], d["gold_tool"], d["gold_args"],
                         d["pred_tool"], d["pred_args"])
        out[d["qa_id"]] = r.f1
    return out


def paired_boot(a: dict, b: dict, seed: int = 0):
    ids = [k for k in a if k in b]
    diffs = [b[k] - a[k] for k in ids]
    rng = random.Random(seed)
    boots = sorted(
        sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs)
        for _ in range(N_BOOT))
    return {
        "n": len(diffs),
        "mean_diff_pts": round(100 * sum(diffs) / len(diffs), 2),
        "ci_low_pts": round(100 * boots[int(0.025 * N_BOOT)], 2),
        "ci_high_pts": round(100 * boots[int(0.975 * N_BOOT)], 2),
        "p_positive": round(sum(1 for x in boots if x > 0) / N_BOOT, 4),
    }


def main():
    ref = value_f1(RESULTS / f"{REFERENCE}.jsonl")
    table = {}
    for name in VARIANTS:
        p = RESULTS / f"{name}.jsonl"
        if not p.exists():
            continue
        table[name] = paired_boot(ref, value_f1(p))
    (RESULTS / "ci_table.json").write_text(
        json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, row in table.items():
        print(name, json.dumps(row))


if __name__ == "__main__":
    main()
