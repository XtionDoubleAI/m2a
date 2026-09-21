"""D7: layered hybrid -- note retrieval anchors over lossless chunk supply.

The only variable changed vs forge-chunks-full is the CHUNK SELECTOR: chunks
enter the evidence pool by dense similarity between the per-parameter demand
question and A-Mem-style notes (one note per chunk, positionally aligned,
cached in results/amem_bank.jsonl), instead of RRF(BM25, dense) over raw
chunks. Rendering and binding are the FORGE pipeline untouched. Notes never
enter the evidence text.

Alignment guarantee: run_amem built notes one-per-chunk via
chunk_session(window=6) in order, so the j-th note of a session maps to the
j-th chunk (verified 429/429 sessions).

Usage (WSL): python -X utf8 -m experiments.run_layered [--expand-window 1]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.baselines.ltmemory import chunk_session  # noqa: E402
from experiments.runlog import IntermediateDumper, log_run  # noqa: E402
from forge.act.binder import bind  # noqa: E402
from forge.act.embedder import BGEM3Dense  # noqa: E402
from forge.act.intent import generate_demands  # noqa: E402
from forge.act.llm import OfflineLLM  # noqa: E402
from forge.act.render import render_evidence  # noqa: E402
from forge.eval.dataset import Bench  # noqa: E402
from forge.eval.metrics import aggregate  # noqa: E402
from forge.eval.runner import run_system  # noqa: E402
from forge.schema import load_tool_spec  # noqa: E402

IS_WSL = sys.platform == "linux"
BENCH_DIR = Path(
    "/mnt/e/hx/_DoctorXtion/intern_shxt/Pdev/refs/repos/Mem2ActBench/toolmembench_small"
    if IS_WSL else
    r"E:\hx\_DoctorXtion\intern_shxt\Pdev\refs\repos\Mem2ActBench\toolmembench_small"
)
MODELS_DIR = Path("/mnt/e/hx/_models" if IS_WSL else r"E:\hx\_models")
OUT_DIR = Path(__file__).resolve().parent.parent / "results"
BANK_CACHE = OUT_DIR / "amem_bank.jsonl"
WINDOW = 6
MAX_BLOCKS_PER_TASK = 12


def _snapshot(repo: str) -> Path:
    direct = MODELS_DIR / repo
    if (direct / "config.json").exists():
        return direct
    ms = MODELS_DIR / "models" / repo.replace("/", "--")
    snaps = sorted((ms / "snapshots").glob("*")) if (ms / "snapshots").exists() else []
    return snaps[-1] if snaps else direct


def _render_note(n: dict) -> str:
    return (f"content: {n['content']}\ncontext: {n['context']}\n"
            f"keywords: {', '.join(n.get('keywords', []))}\n"
            f"tags: {', '.join(n.get('tags', []))}")


def build_layered_index(bench: Bench, embedder: BGEM3Dense):
    """(session_id, chunk_idx) -> chunk text; note vectors in the same order."""
    notes_by_sid: dict[str, list[dict]] = {}
    for line in open(BANK_CACHE, encoding="utf-8"):
        d = json.loads(line)
        notes_by_sid.setdefault(d["session_id"], []).append(d)
    chunks: list[dict] = []   # {session_id, chunk_id, text}
    note_texts: list[str] = []
    for s in bench.sessions:
        session_chunks = chunk_session(s.turns, window=WINDOW)
        session_notes = notes_by_sid.get(s.session_id, [])
        for j, text in enumerate(session_chunks):
            chunks.append({"session_id": s.session_id, "chunk_id": f"{s.session_id}#{j}",
                           "text": text})
            note_texts.append(_render_note(session_notes[j])
                              if j < len(session_notes) else text)
    vecs = embedder.encode(note_texts)
    return chunks, vecs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--top-notes", type=int, default=5)
    ap.add_argument("--expand-window", type=int, default=1,
                    help="blocks adjacent to each anchor note's block (0 = off)")
    ap.add_argument("--name", default="forge-layered")
    args = ap.parse_args()

    bench = Bench(BENCH_DIR)
    print("bench:", json.dumps(bench.stats(), ensure_ascii=False))

    embedder = BGEM3Dense(str(_snapshot("BAAI/bge-m3")), device="cuda:1")
    llm = OfflineLLM(str(_snapshot("Qwen/Qwen2.5-7B-Instruct")),
                     gpu_memory_utilization=0.92)

    chunks, note_vecs = build_layered_index(bench, embedder)
    print(f"layered index: {len(chunks)} chunks (note-anchored)")

    # positional index: session -> [chunk row ids]
    rows_by_sid: dict[str, list[int]] = {}
    for i, c in enumerate(chunks):
        rows_by_sid.setdefault(c["session_id"], []).append(i)

    out_path = OUT_DIR / f"{args.name}.jsonl"
    inter_path = OUT_DIR / f"{args.name}.intermediates.jsonl"
    dumper = IntermediateDumper(inter_path)

    def system(task, session_texts=None):
        spec = load_tool_spec(task.tool_schema)
        demands = generate_demands(llm, task.query, spec)
        dvecs = embedder.encode([d["query"] for d in demands]) if demands else []

        visible_rows = [i for sid in task.session_ids for i in rows_by_sid.get(sid, [])]
        picked_rows: list[int] = []
        for d, v in zip(demands, dvecs):
            sims = note_vecs[visible_rows] @ v
            order = np.argsort(-sims)[: args.top_notes]
            for r in order:
                anchor = visible_rows[r]
                sid = chunks[anchor]["session_id"]
                j = rows_by_sid[sid].index(anchor)
                lo = max(0, j - args.expand_window)
                hi = min(len(rows_by_sid[sid]), j + args.expand_window + 1)
                picked_rows.extend(rows_by_sid[sid][lo:hi])
        # dedupe, keep first-seen order, cap
        seen, block_rows = set(), []
        for i in picked_rows:
            if i not in seen:
                seen.add(i)
                block_rows.append(i)
        block_rows = block_rows[:MAX_BLOCKS_PER_TASK]
        block_texts = [chunks[i]["text"] for i in block_rows]

        card_hits = [(d, []) for d in demands]
        evidence = render_evidence(card_hits, chunk_texts=block_texts or None)
        trusted = "\n".join(block_texts) if block_texts else None
        pred_tool, final_args, model_args = bind(llm, spec, task.query, evidence,
                                                 demand_evidence=None,
                                                 trusted_texts=trusted)
        dumper.dump(task.qa_id, demands,
                    {d["param_name"]: [] for d in demands},
                    {"n_chunks": len(block_texts),
                     "note_anchored": [chunks[i]["chunk_id"] for i in block_rows]},
                    model_args, final_args, task.arguments,
                    extra={"evidence": evidence[:24000]})
        return pred_tool, final_args

    results = run_system(system, bench, out_path, limit=args.limit)
    dumper.close()
    aggs = aggregate(results)
    log_run(args.name, config={**vars(args), "selector": "note-similarity routing",
                               "bank": str(BANK_CACHE),
                               "supply": "lossless chunks, FORGE render+bind unchanged"},
            aggregates={"f1": round(aggs.f1 * 100, 2), "bleu1": round(aggs.bleu1 * 100, 2),
                        "tsa": round(aggs.tsa * 100, 2), "em": round(aggs.em * 100, 2),
                        "arg_f1": round(aggs.arg_f1 * 100, 2),
                        "slot_acc": round(aggs.slot_acc * 100, 2)},
            files={"samples": str(out_path), "intermediates": str(inter_path)},
            runs_path=OUT_DIR / "runs.jsonl")


if __name__ == "__main__":
    main()
