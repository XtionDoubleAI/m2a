"""Step 06: seeded random spot-check of the frozen dataset pair.

Full-manual reading of every sample is infeasible; instead a fixed-seed random
sample is drawn and each drawn sample passes five mechanical checks plus a
human-readable excerpt dump (query, gold arguments, grounding source texts)
that a reviewer reads to catch semantic issues the checks cannot.

Checks per drawn QA:
  1. schema: all seven fields present, tool name non-empty, arguments dict
  2. visibility: every explicit-grounded gold value occurs (canonicalized) in
     one of the source sessions -- the benchmark's core promise
  3. leakage: no explicit/inferred gold value leaks into the query
  4. linkage: every source_conversation_id resolves to a session
  5. hygiene: query non-empty, no internal words (tool/api/parameter/function)

Output: spot_check_report.json (machine) + console summary; the REPORT_EXCERPTS
section is the part meant for human reading.

Usage (run from this build dir):
    python spot_check.py [--n 25]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOLDOUT = HERE.parent

# reuse the upstream leakage checker verbatim (same rule as construction)
UPSTREAM_04 = HERE / "04_qa_construction.py"
spec = importlib.util.spec_from_file_location("adapter_04", UPSTREAM_04)
up04 = importlib.util.module_from_spec(spec)
sys.modules["adapter_04"] = up04
spec.loader.exec_module(up04)


def canon(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20260925)
    args = ap.parse_args()

    qas = [json.loads(l) for l in
           open(HOLDOUT / "qa_dataset.jsonl", encoding="utf-8") if l.strip()]
    sessions = {}
    for line in open(HOLDOUT / "toolmem_conversation.jsonl", encoding="utf-8"):
        if line.strip():
            s = json.loads(line)
            sessions[s["session_id"]] = s

    idx = list(range(len(qas)))
    random.seed(args.seed)
    draw = sorted(random.sample(idx, min(args.n, len(idx))))

    fields = {"qa_id", "source_conversation_ids", "evolution_chain", "query",
              "tool_call", "target_tool_schema", "complexity_metadata"}
    results = []
    n_pass = 0
    for i in draw:
        qa = qas[i]
        issues = []

        # 1 schema
        missing = fields - set(qa.keys())
        tc = qa.get("tool_call", {})
        if missing:
            issues.append(f"missing fields: {sorted(missing)}")
        if not tc.get("name"):
            issues.append("empty tool name")
        if not isinstance(tc.get("arguments"), dict) or not tc.get("arguments"):
            issues.append("empty arguments")

        # 4 linkage + collect visible text of the QA's sessions
        conv_by_id = {}
        for sid, sess in sessions.items():
            for cid in sess.get("original_conversation_ids", []):
                conv_by_id.setdefault(cid, sess)
        linked_sessions, unlinked = [], []
        for cid in qa.get("source_conversation_ids", []):
            sess = conv_by_id.get(cid)
            if sess:
                linked_sessions.append(sess)
            else:
                unlinked.append(cid)
        if unlinked:
            issues.append(f"unlinked conversation ids: {unlinked[:3]}")
        visible = canon(" ".join(
            json.dumps(t, ensure_ascii=False)
            for sess in linked_sessions for t in sess.get("turns", [])))

        # 2 visibility of explicit gold values (numbers normalized, arrays
        # checked element-wise)
        ginfo = tc.get("grounding_info", {})
        missing_vals = []

        def visible_as(value) -> bool:
            vals = value if isinstance(value, list) else [value]
            for v in vals:
                c = canon(v)
                ok = c in visible
                if not ok:
                    try:  # numeric equality after string round-trip (101.0 vs 101)
                        num = float(v)
                        ok = canon(str(int(num))) in visible
                    except (TypeError, ValueError):
                        ok = False
                if not ok:
                    return False
            return True

        for p, v in tc.get("arguments", {}).items():
            if ginfo.get(p, {}).get("type") == "explicit":
                if not visible_as(v):
                    missing_vals.append(p)
        if missing_vals:
            issues.append(f"explicit values not visible: {missing_vals}")

        # 3 leakage (upstream rule)
        if up04.check_argument_leakage(qa.get("query", ""), tc):
            issues.append("argument leakage in query")

        # 5 hygiene
        q = qa.get("query", "")
        if len(q.strip()) < 8:
            issues.append("query too short")
        if re.search(r"\b(tool|api|parameter|function)\b", q, re.I):
            issues.append("query mentions internal words")

        ok = not issues
        n_pass += ok
        results.append({"qa_id": qa.get("qa_id"), "index": i, "pass": ok,
                        "issues": issues})

    # human-readable excerpts of every drawn sample
    excerpts = []
    for i in draw:
        qa = qas[i]
        tc = qa["tool_call"]
        chain = qa.get("evolution_chain", [])
        excerpts.append({
            "qa_id": qa["qa_id"],
            "level": qa.get("complexity_metadata", {}).get("level"),
            "query": qa["query"],
            "gold_tool": tc.get("name"),
            "gold_args": tc.get("arguments"),
            "grounding": {p: info.get("type")
                          for p, info in tc.get("grounding_info", {}).items()},
            "chain_facts": [c.get("fact") for c in chain][:4],
            "n_source_conversations": len(qa.get("source_conversation_ids", [])),
        })

    report = {
        "n_drawn": len(draw), "seed": args.seed,
        "mechanical_pass": n_pass, "mechanical_fail": len(draw) - n_pass,
        "results": results,
        "note": "mechanical checks cannot judge semantic quality; read the "
                "excerpts below before freezing",
        "REPORT_EXCERPTS": excerpts,
    }
    out = HERE / "spot_check_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"spot-check: {n_pass}/{len(draw)} pass mechanical checks")
    for r in results:
        if not r["pass"]:
            print("  FAIL", r["qa_id"], r["issues"])
    print(f"report (with human-reading excerpts): {out}")


if __name__ == "__main__":
    main()
