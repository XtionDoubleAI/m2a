"""Multi-dimensional evaluation beyond F1 (offline, from experiment archives).

Six questions F1 cannot answer (see Pdev/teaching/04):
  traceability   -- output value literally present in its retrieved evidence
  fabrication    -- wrong value present in NO visible evidence/enum/default
  corruption     -- gold present in evidence AND output in evidence, but differ
  miss           -- gold absent from evidence (retrieval never surfaced it)
  complex-fidelity -- character-exact retention for long values (>30 chars)
  schema-validity  -- param names/types/required conform to the tool contract

Error types are mutually exclusive by check order: miss -> fabrication ->
corruption (if the gold value never entered the evidence, the failure is
retrieval's regardless of what the model output; "each class points to its
fix"). Computes any pair of archived runs via compare_ablation, or the
baseline/FORGE pair via main().

Usage: python -X utf8 -m experiments.multimetrics
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forge.eval.dataset import Bench  # noqa: E402
from forge.schema import load_tool_spec  # noqa: E402

IS_WSL = sys.platform == "linux"
ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = Path(
    "/mnt/e/hx/_DoctorXtion/intern_shxt/Pdev/refs/repos/Mem2ActBench/toolmembench_small"
    if IS_WSL else
    r"E:\hx\_DoctorXtion\intern_shxt\Pdev\refs\repos\Mem2ActBench\toolmembench_small"
)
RESULTS = ROOT / "results"


def canon(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v).strip().casefold()


def type_ok(value, ptype: str) -> bool:
    v = str(value).strip()
    if ptype == "string":
        return True
    if ptype in ("integer", "number"):
        try:
            float(v)
            return True
        except ValueError:
            return False
    if ptype == "boolean":
        return v.lower() in ("true", "false", "yes", "no")
    return True


def analyse(name: str, per_task_args: dict, evidence_texts: dict,
            bench: Bench) -> dict:
    """evidence_texts: qa_id -> single blob of everything the system showed."""
    n = n_ok = 0
    trace_ok = 0
    fab = corr = miss = 0
    cx_n = cx_ok = 0
    val_n = val_ok = 0
    for t in bench.tasks:
        args = per_task_args.get(t.qa_id, {})
        blob = canon(evidence_texts.get(t.qa_id, ""))
        spec = load_tool_spec(t.tool_schema)
        known_params = {p.name for p in spec.params}
        req = {p.name for p in spec.required_params()}
        types = {p.name: p.type for p in spec.params}
        enums_defaults = set()
        for p in spec.params:
            if p.enum:
                enums_defaults |= {canon(e) for e in p.enum}
            if p.has_default:
                enums_defaults.add(canon(p.default))
        # schema validity (per task: names subset, types ok, required covered)
        names_ok = all(k in known_params for k in args)
        types_ok_ = all(type_ok(v, types.get(k, "string")) for k, v in args.items())
        req_ok = req.issubset(set(args))
        val_n += 1
        val_ok += (names_ok and types_ok_ and req_ok)
        for p, g in t.arguments.items():
            n += 1
            out = args.get(p)
            ok = canon(out) == canon(g)
            n_ok += ok
            out_ev = (canon(out) in blob) if out is not None else False
            # traceability is judged independently of correctness: did the
            # system's value literally come from the evidence it showed?
            if out is not None and (out_ev or canon(out) in enums_defaults):
                trace_ok += 1
            if ok:
                continue
            # error taxonomy, miss first: if retrieval never surfaced the gold
            # value the failure is upstream of binding whatever the model did.
            in_ev = canon(g) in blob
            if not in_ev:
                miss += 1
            elif out is not None and not out_ev and canon(out) not in enums_defaults:
                fab += 1
            else:
                corr += 1
            if len(str(g)) > 30:
                cx_n += 1
                cx_ok += ok
    wrong = max(fab + corr + miss, 1)
    return {
        "system": name,
        "slot_accuracy": round(100 * n_ok / max(n, 1), 2),
        "traceability_pct": round(100 * trace_ok / max(n, 1), 2),
        # absolute rates: errors of each class over ALL gold arguments --
        # comparable across systems (the pct-of-errors shares below only
        # describe a system's internal error structure)
        "fabrication_rate": round(100 * fab / max(n, 1), 2),
        "corruption_rate": round(100 * corr / max(n, 1), 2),
        "miss_rate": round(100 * miss / max(n, 1), 2),
        "n_errors": fab + corr + miss,
        # internal structure (shares of errors, sum to 100%)
        "fabrication_pct_of_errors": round(100 * fab / wrong, 2),
        "corruption_pct_of_errors": round(100 * corr / wrong, 2),
        "miss_pct_of_errors": round(100 * miss / wrong, 2),
        "complex_fidelity_pct": round(100 * cx_ok / max(cx_n, 1), 2),
        "schema_validity_pct": round(100 * val_ok / max(val_n, 1), 2),
    }


def forge_evidence_view(bench: Bench) -> dict:
    """Reconstruct, per task, the evidence blob FORGE actually showed
    (chunk texts were not archived verbatim for this run; approximate with the
    full text of the task's evidence sessions -- an UPPER bound on traceability
    for both systems alike, so the comparison stays fair)."""
    return {t.qa_id: "\n".join(bench.session_texts(t)) for t in bench.tasks}


def load_archive(name: str) -> tuple[dict, dict]:
    """-> (per-task predicted args, per-task evidence blob) from a run's
    archives. Handles both layouts: FORGE intermediates (final_args) and
    external-system intermediates (pred_args); evidence from the archived
    rendered evidence, falling back to the full-session upper bound."""
    p = RESULTS / f"{name}.intermediates.jsonl"
    args, ev = {}, {}
    for line in open(p, encoding="utf-8"):
        d = json.loads(line)
        args[d["qa_id"]] = d.get("final_args", d.get("pred_args", {}))
        ev[d["qa_id"]] = d.get("evidence", "")
    return args, ev


def main():
    bench = Bench(BENCH_DIR)
    upper_ev = forge_evidence_view(bench)

    systems = [
        ("LTMemory baseline", "ltmemory_hybrid5_full"),
        ("FORGE (final, chunks)", "forge-chunks-full"),
        ("Mem0", "mem0-full"),
        ("A-Mem (reimpl)", "amem-full"),
    ]
    out = []
    for label, name in systems:
        inter = RESULTS / f"{name}.intermediates.jsonl"
        if inter.exists():
            args, ev = load_archive(name)
            ev = {k: (v or upper_ev[k]) for k, v in ev.items()}
        else:  # legacy runs (LTMemory): predictions only, evidence = upper bound
            args, ev = {}, upper_ev
            res = RESULTS / f"{name}.jsonl"
            if not res.exists():
                print(f"skip {label}: no archive")
                continue
            for line in open(res, encoding="utf-8"):
                d = json.loads(line)
                args[d["qa_id"]] = d["pred_args"]
        # older archives (LTMemory) have no evidence field -> upper bound
        ev = {k: (v or upper_ev[k]) for k, v in ev.items()}
        if not args:
            print(f"skip {label}: no archive")
            continue
        row = analyse(label, args, ev, bench)
        row["run"] = name
        out.append(row)

    (RESULTS / "multimetrics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for row in out:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
