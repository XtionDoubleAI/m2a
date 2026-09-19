"""Mem2ActBench dataset loading.

Data layout (from the official repo):
  Mem2ActBench/qa_dataset.jsonl           -- 400 QA tasks (query, gold tool_call,
                                             grounding_info, target_tool_schema,
                                             complexity_metadata, evolution_chain)
  Mem2ActBench/toolmem_conversation.jsonl -- 2029 long sessions; each session lists
                                             original_conversation_ids it was merged from

A QA task's evidence lives in the session(s) containing its
source_conversation_ids; we build a conversation_id -> session_id inverted index.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class QATask:
    qa_id: str
    query: str
    source_conversation_ids: list[str]
    evolution_chain: list[dict]          # gold memory chain (oracle evidence)
    tool_name: str
    arguments: dict                      # gold parameter values
    grounding_info: dict                 # param -> {"source_text": ..., "type": explicit|inferred|default}
    tool_schema: dict                    # full JSON schema of the target tool
    complexity: dict                     # level, level_name, counts, has_temporal_conflict
    session_ids: list[str] = field(default_factory=list)


@dataclass
class Session:
    session_id: str
    original_conversation_ids: list[str]
    turns: list[dict]
    token_count: int


def load_qa(path: str | Path) -> list[QATask]:
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            tc = d["tool_call"]
            tasks.append(
                QATask(
                    qa_id=d["qa_id"],
                    query=d["query"],
                    source_conversation_ids=d.get("source_conversation_ids", []),
                    evolution_chain=d.get("evolution_chain", []),
                    tool_name=tc["name"],
                    arguments=tc["arguments"],
                    grounding_info=tc.get("grounding_info", {}),
                    tool_schema=d.get("target_tool_schema", {}),
                    complexity=d.get("complexity_metadata", {}),
                )
            )
    return tasks


def load_sessions(path: str | Path) -> list[Session]:
    sessions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sessions.append(
                Session(
                    session_id=d["session_id"],
                    original_conversation_ids=d.get("original_conversation_ids", []),
                    turns=d.get("turns", []),
                    token_count=d.get("token_count", 0),
                )
            )
    return sessions


class Bench:
    """QA tasks + sessions with the conversation_id -> session inverted index."""

    def __init__(self, data_dir: str | Path):
        data_dir = Path(data_dir)
        self.tasks = load_qa(data_dir / "qa_dataset.jsonl")
        self.sessions = load_sessions(data_dir / "toolmem_conversation.jsonl")
        self.sessions_by_id = {s.session_id: s for s in self.sessions}
        self._conv_to_sessions: dict[str, list[str]] = {}
        for s in self.sessions:
            for cid in s.original_conversation_ids:
                self._conv_to_sessions.setdefault(cid, []).append(s.session_id)
        for t in self.tasks:
            t.session_ids = sorted(
                {sid for cid in t.source_conversation_ids for sid in self._conv_to_sessions.get(cid, [])}
            )

    def session_texts(self, task: QATask) -> list[str]:
        """Render each evidence session as plain dialogue text."""
        out = []
        for sid in task.session_ids:
            s = self.sessions_by_id[sid]
            lines = []
            for turn in s.turns:
                role = turn.get("role", "?")
                content = turn.get("content", "")
                if turn.get("tool_calls"):
                    calls = ", ".join(
                        f"{c['function']['name']}({c['function'].get('arguments', '')})"
                        for c in turn["tool_calls"]
                    )
                    lines.append(f"{role}: {content} [calls: {calls}]")
                elif role == "tool":
                    lines.append(f"tool({turn.get('name', '')}): {content}")
                else:
                    lines.append(f"{role}: {content}")
            out.append("\n".join(lines))
        return out

    def stats(self) -> dict:
        from collections import Counter

        levels = Counter(t.complexity.get("level", "?") for t in self.tasks)
        gtypes = Counter(
            info.get("type", "?") for t in self.tasks for info in t.grounding_info.values()
        )
        uncovered = sum(1 for t in self.tasks if not t.session_ids)
        return {
            "n_tasks": len(self.tasks),
            "n_sessions": len(self.sessions),
            "levels": dict(levels),
            "grounding_types": dict(gtypes),
            "tasks_without_session": uncovered,
        }
