"""Diagnostic analysis of where gold values die (offline, archives only).

Three questions, no new mechanisms:

C1  supply funnel per system, stratified:
      L1  gold value present in the task's visible sessions (upper bound)
      L2  gold value present in what the system actually rendered
      L3  output exactly correct
    -> separates "retrieval never surfaced it" (L2 gap) from "the model
       failed to copy it" (L3|L2 gap), per grounding type / long values /
    enum params. For A-Mem, L2bank additionally checks the full note bank
    of the task's visible sessions (bank-level loss vs retrieval loss).

C2  parameter-level correctness overlap across the four systems: does the
    measured "complementary error structure" exist at the parameter level,
    and what do system-unique wins look like?

D6  rank check: for FORGE miss parameters (gold visible but not rendered),
    where does the gold-bearing chunk rank under a BM25 query per demand?
    Verifies the D6 claim "most gold values sit outside the coarse top-50"
    (BM25 half of the RRF pair only; the dense half would need re-embedding).

Output: results/supply_analysis.json + console tables.
Usage: python -X utf8 -m experiments.analyze_supply
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forge.eval.dataset import Bench  # noqa: E402
from forge.schema import load_tool_spec  # noqa: E402
from experiments.multimetrics import canon, load_archive  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
BENCH_DIR = Path(r"E:\hx\_DoctorXtion\intern_shxt\Pdev\refs\repos"
                 r"\Mem2ActBench\toolmembench_small")

SYSTEMS = [("FORGE", "forge-chunks-full"), ("Mem0", "mem0-full"),
           ("A-Mem", "amem-full")]
ALL_SYSTEMS = SYSTEMS + [("LTMemory", "ltmemory_hybrid5_full")]


def strata(task, param, gold):
    gtype = (task.grounding_info.get(param) or {}).get("type", "?")
    spec = load_tool_spec(task.tool_schema)
    p = next((x for x in spec.params if x.name == param), None)
    is_enum = bool(p and p.enum)
    is_long = len(str(gold)) > 30
    return gtype, is_enum, is_long


def task_evidence(name: str) -> dict:
    _, ev, _ = load_archive(name)
    return ev


def funnel(bench: Bench) -> dict:
    visible = {t.qa_id: "\n".join(bench.session_texts(t)) for t in bench.tasks}
    out = {}
    for label, name in SYSTEMS:
        ev = task_evidence(name)
        args, _, has_field = load_archive(name)
        if not has_field:
            continue
        rows = []
        for t in bench.tasks:
            blob_vis = canon(visible[t.qa_id])
            blob_ev = canon(ev.get(t.qa_id, ""))
            pa = args.get(t.qa_id, {})
            for p, g in t.arguments.items():
                gv = canon(g)
                rows.append({
                    "gtype": strata(t, p, g)[0],
                    "enum": strata(t, p, g)[1],
                    "long": strata(t, p, g)[2],
                    "L1": gv in blob_vis,
                    "L2": gv in blob_ev,
                    "L3": p in pa and canon(pa[p]) == gv,
                })
        def rate(sel, key):
            n = sum(1 for r in rows if sel(r))
            k = sum(1 for r in rows if sel(r) and r[key])
            return {"n": n, "rate": round(100 * k / n, 1) if n else None}

        def cond(sel, key, given):
            n = sum(1 for r in rows if sel(r) and r[given])
            k = sum(1 for r in rows if sel(r) and r[given] and r[key])
            return round(100 * k / n, 1) if n else None

        alll = lambda r: True
        table = {
            "overall": {"L1": rate(alll, "L1"), "L2": rate(alll, "L2"),
                        "L3": rate(alll, "L3"),
                        "L2_given_L1": cond(alll, "L2", "L1"),
                        "L3_given_L2": cond(alll, "L3", "L2")},
            "by_gtype": {gt: {"L1": rate(lambda r, gt=gt: r["gtype"] == gt, "L1"),
                              "L2": rate(lambda r, gt=gt: r["gtype"] == gt, "L2"),
                              "L3": rate(lambda r, gt=gt: r["gtype"] == gt, "L3")}
                         for gt in {r["gtype"] for r in rows}},
            "long_values": {"L1": rate(lambda r: r["long"], "L1"),
                            "L2": rate(lambda r: r["long"], "L2"),
                            "L3": rate(lambda r: r["long"], "L3"),
                            "L2_given_L1": cond(lambda r: r["long"], "L2", "L1"),
                            "L3_given_L2": cond(lambda r: r["long"], "L3", "L2")},
            "enum_params": {"L2": rate(lambda r: r["enum"], "L2")},
            "free_text_params": {"L2": rate(lambda r: not r["enum"], "L2")},
        }
        out[label] = table
    return out


def output_correctness(bench: Bench, name: str) -> set:
    """(qa_id, param) pairs the system got exactly right."""
    args = {}
    if (RESULTS / f"{name}.intermediates.jsonl").exists():
        args, _, _ = load_archive(name)
    else:
        for line in open(RESULTS / f"{name}.jsonl", encoding="utf-8"):
            d = json.loads(line)
            args[d["qa_id"]] = d["pred_args"]
    ok = set()
    for t in bench.tasks:
        pa = args.get(t.qa_id, {})
        for p, g in t.arguments.items():
            if p in pa and canon(pa[p]) == canon(g):
                ok.add((t.qa_id, p))
    return ok


def overlap_analysis(bench: Bench) -> dict:
    correct = {label: output_correctness(bench, name)
               for label, name in ALL_SYSTEMS}
    all_pairs = [(t.qa_id, p) for t in bench.tasks for p in t.arguments]
    n = len(all_pairs)

    def feats(pairs):
        cnt_gtype = Counter()
        n_long = 0
        by_task = {t.qa_id: t for t in bench.tasks}
        for qa_id, p in pairs:
            t = by_task[qa_id]
            g = t.arguments[p]
            cnt_gtype[(t.grounding_info.get(p) or {}).get("type", "?")] += 1
            n_long += len(str(g)) > 30
        return {"n": len(pairs),
                "gtype": dict(cnt_gtype),
                "long": n_long,
                "long_pct": round(100 * n_long / len(pairs), 1) if pairs else None}

    sets = [correct[l] for l, _ in ALL_SYSTEMS]
    return {
        "n_params": n,
        "per_system_correct": {l: len(correct[l]) for l, _ in ALL_SYSTEMS},
        "all_four_correct": len(set.intersection(*sets)),
        "all_four_wrong": n - len(set.union(*sets)),
        "forge_only_correct": len(correct["FORGE"]
                                  - correct["Mem0"] - correct["A-Mem"]
                                  - correct["LTMemory"]),
        "amem_only_correct": len(correct["A-Mem"]
                                 - correct["Mem0"] - correct["FORGE"]
                                 - correct["LTMemory"]),
        "forge_wrong_amem_correct_feats": feats(correct["A-Mem"] - correct["FORGE"]),
        "forge_correct_amem_wrong_feats": feats(correct["FORGE"] - correct["A-Mem"]),
        "pairwise_jaccard": {
            f"{a}~{b}": round(len(correct[a] & correct[b])
                              / len(correct[a] | correct[b]), 3)
            for i, (a, _) in enumerate(ALL_SYSTEMS)
            for b, _ in ALL_SYSTEMS[i + 1:]},
    }


# ---------------- BM25 rank check (D6 claim) ----------------

def bm25_index(corpus: list[str]):
    docs = [re.findall(r"\w+", c.lower()) for c in corpus]
    df = Counter(w for doc in docs for w in set(doc))
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n
    k1, b = 1.5, 0.75
    tf = [Counter(d) for d in docs]
    return df, tf, n, avgdl, k1, b


def bm25_scores(query: str, idx) -> list[float]:
    df, tf, n, avgdl, k1, b = idx
    q = re.findall(r"\w+", query.lower())
    out = []
    for i in range(n):
        s = 0.0
        for w in q:
            if w not in tf[i]:
                continue
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            f = tf[i][w]
            s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(tf[i]) / avgdl))
        out.append(s)
    return out


def rank_check(bench: Bench) -> dict:
    """FORGE miss params (gold visible, not rendered).

    Part 1 -- demand coverage: the repair loop in forge/act/intent.py only
    guarantees demands for REQUIRED parameters; optional ones the LLM skipped
    are silently never retrieved for. Count the miss params with no demand,
    split by required/optional.

    Part 2 -- BM25 rank of the best gold-bearing chunk for miss params that
    DO have a demand (BM25 half of RRF only).
    """
    chunks = [json.loads(l) for l in
              open(RESULTS / "chunks.jsonl", encoding="utf-8")]
    idx = bm25_index([c["text"] for c in chunks])
    ev = task_evidence("forge-chunks-full")
    visible = {t.qa_id: "\n".join(bench.session_texts(t)) for t in bench.tasks}
    demands = {}
    for line in open(RESULTS / "forge-chunks-full.intermediates.jsonl",
                     encoding="utf-8"):
        d = json.loads(line)
        demands[d["qa_id"]] = {x["param_name"]: x["query"] for x in d["demands"]}

    ranks = []
    no_demand = Counter()
    no_demand_gtype = Counter()
    for t in bench.tasks:
        blob_vis = canon(visible[t.qa_id])
        blob_ev = canon(ev.get(t.qa_id, ""))
        spec = load_tool_spec(t.tool_schema)
        req = {p.name for p in spec.required_params()}
        for p, g in t.arguments.items():
            gv = canon(g)
            if gv in blob_ev or gv not in blob_vis:
                continue                     # rendered, or not visible at all
            q = demands.get(t.qa_id, {}).get(p)
            gt = (t.grounding_info.get(p) or {}).get("type", "?")
            if not q:
                no_demand["required" if p in req else "optional"] += 1
                no_demand_gtype[gt] += 1
                continue
            scores = bm25_scores(q, idx)
            gold_chunks = [i for i, c in enumerate(chunks)
                           if gv in canon(c["text"])
                           and c["session_id"] in set(t.session_ids)]
            if not gold_chunks:
                continue                     # gold visible but not chunk-exact
            best = max(gold_chunks, key=lambda i: scores[i])
            order = sorted(range(len(chunks)),
                           key=lambda i: -scores[i])
            ranks.append(order.index(best) + 1)

    ranks.sort()
    out = {
        "miss_no_demand": dict(no_demand),
        "miss_no_demand_gtype": dict(no_demand_gtype),
        "miss_with_demand": len(ranks),
        "note": "BM25 half of RRF only; dense half not recomputed",
    }
    if ranks:
        out.update({
            "median_rank": ranks[len(ranks) // 2],
            "p75_rank": ranks[int(0.75 * len(ranks))],
            "pct_within_top5": round(100 * sum(1 for r in ranks if r <= 5) / len(ranks), 1),
            "pct_within_top50": round(100 * sum(1 for r in ranks if r <= 50) / len(ranks), 1),
        })
    return out


def amem_bank_supply(bench: Bench) -> dict:
    """Gold present in the FULL note bank of the task's visible sessions
    (A-Mem): bank-level loss, above the retrieval-level L2."""
    notes = defaultdict(list)
    for line in open(RESULTS / "amem_bank.jsonl", encoding="utf-8"):
        d = json.loads(line)
        notes[d["session_id"]].append(canon(d["content"]))
    n = bank_ok = 0
    by_long = Counter()
    for t in bench.tasks:
        blob = " ".join(x for sid in t.session_ids for x in notes.get(sid, []))
        for p, g in t.arguments.items():
            gv = canon(g)
            n += 1
            hit = gv in blob
            bank_ok += hit
            if len(str(g)) > 30:
                by_long["n"] += 1
                by_long["ok"] += hit
    return {"n": n, "bank_supply_rate": round(100 * bank_ok / n, 1),
            "long_values": {"n": by_long["n"],
                            "bank_supply_rate": round(100 * by_long["ok"] / max(by_long["n"], 1), 1)}}


def main():
    bench = Bench(BENCH_DIR)
    report = {
        "C1_supply_funnel": funnel(bench),
        "C1_amem_bank_supply": amem_bank_supply(bench),
        "C2_overlap": overlap_analysis(bench),
        "D6_rank_check": rank_check(bench),
    }
    (RESULTS / "supply_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
