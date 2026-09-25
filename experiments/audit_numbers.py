"""Independent number audit: recompute every archived run and check every
number the documents claim, from samples only (never from runs.jsonl or the
legacy jsonl f1 column).

Three layers:
  1. recompute  -- value-token F1 (forge.eval.metrics.score_sample) for every
     results/*.jsonl result archive; compare against the run registry.
  2. assertions -- the numbers claimed by README / EXPERIMENT_SETUP /
     teaching notes / decision records / the paper, each with its source;
     PASS when the recomputation matches within 0.005.
  3. diagnostics -- paired bootstrap for the headline claim, multi-dimensional
     absolute error rates for the four-system tables.

Output: results/audit_numbers.json (machine) + console summary.
Exit code 1 if any assertion fails.

Usage: python -X utf8 -m experiments.audit_numbers
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forge.eval.metrics import score_sample, aggregate  # noqa: E402
from forge.eval.dataset import Bench  # noqa: E402
from experiments.multimetrics import analyse, load_archive  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
BENCH_DIR = Path(r"E:\hx\_DoctorXtion\intern_shxt\Pdev\refs\repos"
                 r"\Mem2ActBench\toolmembench_small")

# Result archives only; stores, fact banks, intermediates and the registry
# are excluded (intermediates carry per-task dumps, not scored samples).
SKIP_PREFIXES = ("runs", "amem_bank", "chunks", "fact_cards", "mem0_facts")

# Document-claimed value-token F1 (x100). source: where the number is stated.
CLAIMED_F1 = [
    ("ltmemory_hybrid5_full", 30.68, "README/teaching-03/EXPERIMENT_SETUP"),
    ("forge-chunks-only", 36.88, "README/EXPERIMENT_SETUP"),
    ("forge-chunks-full", 36.88, "README ablation"),
    ("mem0-full", 39.19, "README/teaching-03"),
    ("amem-full", 41.11, "README/teaching-03"),
    ("forge-layered", 34.46, "project report"),
    ("oracle-supply", 67.31, "README"),
    ("forge-fixed-subset", 68.11, "project report (115-task subset)"),
    ("forge-fixed-full", 54.07, "README/project report (with demand fallback)"),
    ("full-supply", 49.58, "README/project report"),
    ("forge-pool", 57.05, "project report (2029-session pool)"),
    ("ltm-pool", 33.02, "project report (2029-session pool)"),
    ("full-pool", 51.21, "project report (2029-session pool)"),
    ("inject-first", 93.66, "project report (gold injection)"),
    ("inject-middle", 93.60, "project report (gold injection)"),
    ("inject-last", 94.55, "project report (gold injection)"),
    ("ltm-v4f-off", 34.60, "project report (V4-Flash, thinking off)"),
    ("forge-v4f-off", 61.95, "project report (V4-Flash, thinking off)"),
    ("mem0-v4f-off", 41.30, "project report (V4-Flash, thinking off)"),
    ("amem-v4f-off", 43.93, "project report (V4-Flash, thinking off)"),
    ("full-v4f-off", 63.33, "project report (V4-Flash, thinking off)"),
    ("oracle-v4f-off", 70.06, "project report (V4-Flash, thinking off)"),
    ("ltm-v4f-on", 33.86, "project report (V4-Flash, thinking on)"),
    ("forge-v4f-on", 60.78, "project report (V4-Flash, thinking on)"),
    ("mem0-v4f-on", 42.09, "project report (V4-Flash, thinking on)"),
    ("amem-v4f-on", 45.14, "project report (V4-Flash, thinking on)"),
    ("full-v4f-on", 63.00, "project report (V4-Flash, thinking on)"),
    ("oracle-v4f-on", 69.73, "project report (V4-Flash, thinking on)"),
    ("ltm-v4p-on", 35.99, "project report (V4-Pro, thinking on)"),
    ("m2a-hybrid-store", 27.72, "README ablation/teaching-03"),
    ("forge-chunks-nodemand", 35.18, "README ablation/teaching-03"),
    ("forge-chunks-flat", 37.74, "README ablation/teaching-03"),
    ("forge-chunks-r1", 37.08, "project report"),
    ("forge-chunks-r1r2", 36.65, "project report"),
    ("forge-chunks-r1r2r3", 36.65, "project report"),
    ("forge-chunks-r1r2r3-low", 36.65, "project report"),
]

# Numbers stated in documents whose run has NO own sample archive: they are
# offline re-binding ablations whose runs.jsonl entry points at the source
# intermediates (the offline re-binding ablation mechanism).
# Recorded as sourced, not recomputed.
SOURCED_OFFLINE = [
    ("m2a-hybrid-store-nooverride", 33.88, "results/m2a-hybrid-store.intermediates.jsonl"),
    ("forge-hybrid-base", 33.88, "results/m2a-hybrid-store.intermediates.jsonl"),
    ("forge-hybrid-unique-direct", 33.88, "results/m2a-hybrid-store.intermediates.jsonl"),
    ("forge-hybrid-schema-default", 33.88, "results/m2a-hybrid-store.intermediates.jsonl"),
    ("forge-hybrid-enum-snap", 33.89, "results/m2a-hybrid-store.intermediates.jsonl"),
]

# Multi-dimensional absolute rates (x100, over all 825 gold arguments).
# Sources: the multi-dimension tables in README and the project report.
CLAIMED_MULTIDIM = [
    # run, metric, claimed, source
    ("forge-chunks-full", "miss_rate", 39.9, "project report"),
    ("forge-chunks-full", "corruption_rate", 25.5, "project report"),
    ("forge-chunks-full", "fabrication_rate", 4.6, "project report"),
    ("mem0-full", "miss_rate", 52.0, "project report"),
    ("mem0-full", "corruption_rate", 13.8, "project report"),
    ("mem0-full", "fabrication_rate", 2.6, "project report"),
    ("amem-full", "miss_rate", 44.4, "project report"),
    ("amem-full", "corruption_rate", 18.2, "project report"),
    ("amem-full", "fabrication_rate", 3.6, "project report"),
    ("ltmemory_hybrid5_full", "miss_rate", 25.7, "project report (upper bound)"),
    ("ltmemory_hybrid5_full", "corruption_rate", 27.5, "project report (upper bound)"),
    ("ltmemory_hybrid5_full", "fabrication_rate", 23.3, "project report (upper bound)"),
    ("forge-chunks-full", "traceability_pct", 39.3, "project report"),
    ("forge-chunks-flat", "traceability_pct", 36.5, "project report"),
    ("forge-chunks-full", "complex_fidelity_pct", 9.76, "project report"),
    ("forge-fixed-full", "miss_rate", 26.79, "project report"),
    ("forge-fixed-full", "corruption_rate", 29.82, "project report"),
    ("forge-fixed-full", "fabrication_rate", 4.85, "project report"),
    ("forge-fixed-full", "traceability_pct", 51.88, "project report"),
    ("forge-fixed-full", "complex_fidelity_pct", 19.51, "project report"),
]

# Headline paired CIs beyond the README ablation pair. Each is verified by an
# independent paired bootstrap here (not by trusting ci_table.json).
CLAIMED_CI_PAIRS = [
    # key, baseline, system, mean_diff, ci_low, ci_high
    ("fixed_vs_amem", "amem-full", "forge-fixed-full", 12.96, 8.66, 17.34),
    ("fixed_vs_fullsupply", "full-supply", "forge-fixed-full", 4.49, 1.15, 7.91),
    ("v4f_off_forge_vs_amem", "amem-v4f-off", "forge-v4f-off", 18.02, 13.80, 22.31),
    ("v4f_off_forge_vs_full", "full-v4f-off", "forge-v4f-off", -1.38, -3.43, 0.62),
    ("pool_forge_vs_full", "full-pool", "forge-pool", 5.84, 2.44, 9.21),
]

# README headline: FORGE over baseline, paired bootstrap 95% CI.
CLAIMED_CI = {"mean_diff_pts": 6.19, "ci_low_pts": 1.72, "ci_high_pts": 10.82,
              "pair": ("ltmemory_hybrid5_full", "forge-chunks-full")}

TOL = 0.005


def recompute_f1(path: Path):
    results = []
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        results.append(score_sample(d["qa_id"], d["gold_tool"], d["gold_args"],
                                    d["pred_tool"], d["pred_args"]))
    return aggregate(results)


def registry_f1() -> dict:
    out = {}
    p = RESULTS / "runs.jsonl"
    if p.exists():
        for line in open(p, encoding="utf-8"):
            d = json.loads(line)
            out[d["name"]] = (d.get("aggregates") or {}).get("f1")
    return out


def value_f1(path: Path) -> dict:
    out = {}
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        out[d["qa_id"]] = score_sample(
            d["qa_id"], d["gold_tool"], d["gold_args"],
            d["pred_tool"], d["pred_args"]).f1
    return out


def paired_boot(a: dict, b: dict, n_boot: int = 10000, seed: int = 0):
    ids = [k for k in a if k in b]
    diffs = [b[k] - a[k] for k in ids]
    import random
    rng = random.Random(seed)
    boots = sorted(sum(diffs[rng.randrange(len(diffs))] for _ in diffs)
                   / len(diffs) for _ in range(n_boot))
    return {"n": len(diffs),
            "mean_diff_pts": round(100 * sum(diffs) / len(diffs), 2),
            "ci_low_pts": round(100 * boots[int(0.025 * n_boot)], 2),
            "ci_high_pts": round(100 * boots[int(0.975 * n_boot)], 2)}


def multidim(name: str, bench: Bench) -> dict:
    args, ev, has_field = load_archive(name)
    if not has_field:
        return {}          # ltmemory-style legacy archive handled by caller
    return analyse(name, args, ev, bench)


def multidim_legacy(name: str, bench: Bench) -> dict:
    """Legacy run without evidence field: per-task upper-bound evidence view
    (same rule as multimetrics.main uses for LTMemory)."""
    args = {}
    for line in open(RESULTS / f"{name}.jsonl", encoding="utf-8"):
        d = json.loads(line)
        args[d["qa_id"]] = d["pred_args"]
    ev = {t.qa_id: "\n".join(bench.session_texts(t)) for t in bench.tasks}
    return analyse(name, args, ev, bench)


def main():
    bench = Bench(BENCH_DIR)
    report = {"recomputed": {}, "registry_mismatch": [], "assertions": [],
              "unverifiable": [], "f1_16_23_carrier": None}

    reg = registry_f1()
    for p in sorted(RESULTS.glob("*.jsonl")):
        if p.name.startswith(SKIP_PREFIXES) or p.stem.endswith(".intermediates"):
            continue
        try:
            agg = recompute_f1(p)
        except (KeyError, json.JSONDecodeError) as e:
            report["unverifiable"].append(f"{p.stem}: {e}")
            continue
        row = {"n": agg.n, "f1": round(agg.f1 * 100, 2)}
        claimed_reg = reg.get(p.stem)
        if claimed_reg is not None and abs(claimed_reg - row["f1"]) > TOL:
            report["registry_mismatch"].append(
                {"run": p.stem, "registry_f1": claimed_reg,
                 "recomputed_f1": row["f1"]})
        report["recomputed"][p.stem] = row
        if abs(row["f1"] - 16.23) <= TOL:
            report["f1_16_23_carrier"] = p.stem

    def check(label: str, got, claimed, tol=TOL):
        ok = got is not None and abs(got - claimed) <= tol
        report["assertions"].append(
            {"claim": label, "claimed": claimed,
             "recomputed": None if got is None else round(got, 2),
             "pass": bool(ok)})
        return ok

    for name, claimed, source in CLAIMED_F1:
        got = report["recomputed"].get(name, {}).get("f1")
        check(f"F1 {name} [{source}]", got, claimed)

    report["sourced_offline"] = [
        {"run": n, "claimed_f1": c, "source_archive": s}
        for n, c, s in SOURCED_OFFLINE]

    md_cache = {}
    for name, metric, claimed, source in CLAIMED_MULTIDIM:
        if name not in md_cache:
            try:
                md_cache[name] = multidim(name, bench)
            except FileNotFoundError:
                md_cache[name] = None
            if md_cache[name] in (None, {}):
                md_cache[name] = multidim_legacy(name, bench)
        got = md_cache[name].get(metric)
        # documents state these at one decimal; allow one rounding step
        check(f"{metric} {name} [{source}]", got, claimed, tol=0.06)

    a, b = CLAIMED_CI["pair"]
    pa = RESULTS / f"{a}.jsonl"
    pb = RESULTS / f"{b}.jsonl"
    if pa.exists() and pb.exists():
        ci = paired_boot(value_f1(pa), value_f1(pb))
        report["headline_ci_recomputed"] = ci
        for k in ("mean_diff_pts", "ci_low_pts", "ci_high_pts"):
            # bootstrap percentiles jitter by a hundredth between runs
            check(f"headline CI {k} [README]", ci.get(k), CLAIMED_CI[k],
                  tol=0.05)

    for key, base, system, md, lo, hi in CLAIMED_CI_PAIRS:
        fb = RESULTS / f"{base}.jsonl"
        fs = RESULTS / f"{system}.jsonl"
        if not (fb.exists() and fs.exists()):
            report["unverifiable"].append(f"{key}: missing archive")
            continue
        ci = paired_boot(value_f1(fb), value_f1(fs))
        check(f"CI pair {key} mean [ci_table]", ci.get("mean_diff_pts"), md)
        check(f"CI pair {key} low [ci_table]", ci.get("ci_low_pts"), lo,
              tol=0.05)
        check(f"CI pair {key} high [ci_table]", ci.get("ci_high_pts"), hi,
              tol=0.05)

    # runs registered but with no sample archive (cannot be independently
    # recomputed) -- reported as facts, not failures
    for name in reg:
        if not (RESULTS / f"{name}.jsonl").exists():
            files = None
            report["unverifiable"].append(
                f"{name}: registered F1={reg[name]}, no own sample archive")

    n_pass = sum(1 for x in report["assertions"] if x["pass"])
    report["summary"] = {"assertions": len(report["assertions"]),
                         "passed": n_pass,
                         "failed": len(report["assertions"]) - n_pass}
    (RESULTS / "audit_numbers.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"recomputed {len(report['recomputed'])} archives; "
          f"16.23 carrier: {report['f1_16_23_carrier']}")
    print(f"registry mismatches: {len(report['registry_mismatch'])}")
    for m in report["registry_mismatch"]:
        print("  ", json.dumps(m))
    print(f"unverifiable: {len(report['unverifiable'])}")
    for u in report["unverifiable"]:
        print("  ", u)
    for x in report["assertions"]:
        if not x["pass"]:
            print("FAIL", json.dumps(x, ensure_ascii=False))
    print(f"assertions: {n_pass}/{len(report['assertions'])} pass")
    sys.exit(1 if report["summary"]["failed"] else 0)


if __name__ == "__main__":
    main()
