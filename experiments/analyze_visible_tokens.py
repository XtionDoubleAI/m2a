"""Token-length distribution of each task's visible conversations.

For every task: join all sessions mapped from its source_conversation_ids
(the text that a "full-supply" evidence segment would need to carry) and
count tokens with the actual Qwen2.5 tokenizer. Reports per-task and
per-session distributions and the fraction exceeding context limits.

Usage (WSL): python -X utf8 -m experiments.analyze_visible_tokens
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forge.eval.dataset import Bench  # noqa: E402
from experiments.run_forge import _snapshot  # noqa: E402

BENCH_DIR = Path(
    "/mnt/e/hx/_DoctorXtion/intern_shxt/Pdev/refs/repos/Mem2ActBench/toolmembench_small")


def pct(sorted_vals, q):
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def describe(name, vals, limits=(16384, 32768)):
    s = sorted(vals)
    print(f"{name}: n={len(s)} min={s[0]} max={s[-1]} "
          f"mean={statistics.mean(s):.0f} median={statistics.median(s):.0f} "
          f"p90={pct(s, 0.90)} p95={pct(s, 0.95)}")
    for lim in limits:
        over = sum(1 for v in s if v > lim)
        print(f"  > {lim}: {over} ({100 * over / len(s):.1f}%)")


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(_snapshot("Qwen/Qwen2.5-7B-Instruct")))
    bench = Bench(BENCH_DIR)

    session_tokens = {}
    task_tokens, task_sessions, task_chars = [], [], []
    for t in bench.tasks:
        texts = bench.session_texts(t)
        total = 0
        for sid, text in zip(t.session_ids, texts):
            if sid not in session_tokens:
                session_tokens[sid] = len(tok(text)["input_ids"])
            total += session_tokens[sid]
        task_tokens.append(total)
        task_sessions.append(len(t.session_ids))
        task_chars.append(sum(len(x) for x in texts))

    describe("per-task visible-conversation tokens (evidence would carry all)",
             task_tokens)
    describe("per-session tokens", list(session_tokens.values()), limits=(16384,))
    describe("sessions visible per task", task_sessions, limits=())
    describe("per-task characters", task_chars, limits=(16384, 32768))

    top = sorted(zip(task_tokens, bench.tasks), key=lambda x: -x[0])[:5]
    print("top-5 largest tasks:")
    for n, t in top:
        print(f"  {t.qa_id}: {n} tokens, {len(t.session_ids)} sessions")

    out = {"per_task": [{"qa_id": t.qa_id, "tokens": n, "chars": c, "sessions": s}
                         for t, n, c, s in zip(bench.tasks, task_tokens,
                                               task_chars, task_sessions)]}
    Path("results/visible_tokens.json").write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print("per-task table: results/visible_tokens.json")


if __name__ == "__main__":
    main()
