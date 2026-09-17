"""Calibrate metric definitions against paper anchors, offline.

The paper (Table 3, Qwen2.5-7B, LTMemory) reports F1=26.71 BLEU=64.07 TA=87.25.
Those numbers jointly imply: strict F1 (no free credit from parameter names) and
lenient TA (extra predicted params do not void a sample). We grid over metric
variants on the logged per-sample predictions and report which combination
lands closest to the anchors. No model re-run needed.

Usage: python -m experiments.calibrate_metrics results/ltmemory_hybrid5_full.jsonl
"""

from __future__ import annotations

import json
import re
import sys
from itertools import product

ANCHORS = {"f1": 26.71, "bleu1": 64.07, "ta": 87.25}


def canon(v, casefold: bool) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    s = str(v).strip()
    return s.casefold() if casefold else s


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text)


def serialize(args: dict, with_names: bool, casefold: bool) -> list[str]:
    out: list[str] = []
    for k, v in sorted(args.items()):
        if with_names:
            out.extend(tokens(k))
        out.extend(tokens(canon(v, casefold)))
    return out


def f1_bleu(pred: list[str], gold: list[str]) -> tuple[float, float]:
    from collections import Counter
    pc, gc = Counter(pred), Counter(gold)
    overlap = sum((pc & gc).values())
    p = overlap / len(pred) if pred else 0.0
    r = overlap / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return f1, p


def ta(pred: dict, gold: dict, casefold: bool, ignore_extra: bool) -> bool:
    keys = gold.keys() if ignore_extra else (set(gold) & set(pred)) | (set(gold) | set(pred))
    if not ignore_extra and set(pred) != set(gold):
        return False
    return all(canon(pred.get(k), casefold) == canon(v, casefold) for k, v in gold.items())


def main(path: str):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    print(f"{'names':<7}{'casefold':<9}{'ignoreExtra':<12}{'F1':>7}{'BLEU1':>8}{'TA':>8}   |dF1|+|dBLEU|+|dTA|")
    combos = []
    for with_names, casefold, ignore_extra in product([False, True], [True, False], [True, False]):
        f1s, bleus, tas = [], [], []
        for r in rows:
            pred, gold = r["pred_args"], r["gold_args"]
            pt = serialize(pred, with_names, casefold)
            gt = serialize(gold, with_names, casefold)
            f1, p = f1_bleu(pt, gt)
            f1s.append(f1)
            bleus.append(p)
            tas.append(ta(pred, gold, casefold, ignore_extra))
        F1, B1, TA = 100 * sum(f1s) / len(rows), 100 * sum(bleus) / len(rows), 100 * sum(tas) / len(rows)
        dist = abs(F1 - ANCHORS["f1"]) + abs(B1 - ANCHORS["bleu1"]) + abs(TA - ANCHORS["ta"])
        combos.append((dist, with_names, casefold, ignore_extra, F1, B1, TA))
        print(f"{str(with_names):<7}{str(casefold):<9}{str(ignore_extra):<12}{F1:>7.2f}{B1:>8.2f}{TA:>8.2f}   {dist:>6.1f}")
    combos.sort()
    d, names, cf, ie, F1, B1, TA = combos[0]
    print(f"\nbest: names={names} casefold={cf} ignore_extra={ie} -> F1={F1:.2f} BLEU1={B1:.2f} TA={TA:.2f} (dist {d:.1f})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/ltmemory_hybrid5_full.jsonl")
