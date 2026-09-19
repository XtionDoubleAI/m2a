"""Run bookkeeping for paper-grade experiment logs.

Every experiment appends one manifest line to results/runs.jsonl:
  {run_id, name, timestamp, git_commit, config{...}, aggregates{...}, files{...}}

Per-sample dumps (m2a only) record the intermediates needed for error
analysis and ablation figures:
  {qa_id, demands, card_hits, chunk_hits, model_args (pre-override),
   final_args, gold_args, per-slot funnel flags}
Together with experiments/funnel.py this yields the coverage->retrieval->
binding funnel that motivates the hybrid store (paper Figure: fact-only vs
hybrid-store).
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            cwd=Path(__file__).resolve().parent.parent, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "?"


def log_run(name: str, config: dict, aggregates: dict, files: dict,
            runs_path: str | Path) -> None:
    entry = {
        "run_id": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "name": name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "config": config,
        "aggregates": aggregates,
        "files": files,
    }
    p = Path(runs_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


class IntermediateDumper:
    """Collect per-task intermediates; write one JSONL at the end."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")

    def dump(self, qa_id: str, demands: list, card_hits: dict, chunk_hits: dict,
             model_args: dict, final_args: dict, gold_args: dict,
             extra: dict | None = None) -> None:
        rec = {
            "qa_id": qa_id,
            "demands": demands,
            "card_hits": card_hits,
            "chunk_hits": chunk_hits,
            "model_args": model_args,
            "final_args": final_args,
            "gold_args": gold_args,
        }
        if extra:
            rec.update(extra)
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def close(self):
        self._fh.close()
