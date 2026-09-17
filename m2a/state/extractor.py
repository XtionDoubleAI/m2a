"""Fact-card extraction: session dialogue -> fact cards (write path).

Each card: (attribute, value, source_text, session_id, turn_index).
Extraction prompt follows the benchmark's own Appendix D convention
(entity-bound attribute labels, explicit facts only), which was validated
by the paper's human annotators.

The LLM-facing contract is intentionally narrow: it returns JSON lines, we
parse defensively and keep only well-formed cards. A failed extraction on a
session degrades to zero cards (the read path then falls back to chunk
retrieval when enabled).
"""

from __future__ import annotations

import json
import re

SYSTEM_PROMPT = (
    "You are an information extraction system. From the given dialogue, extract "
    "memory-worthy facts about the user. Output ONLY a JSON array; no other text.\n"
    "Each element: {\"attribute\": str, \"value\": str, \"source_text\": str}.\n"
    "Rules:\n"
    "1. attribute = a normalized category label BINDING the topic to an entity or "
    "detail, e.g. 'Platform Preference (finance)', 'Package Manager (Python)'. "
    "Consistent, specific labels are critical.\n"
    "2. value = the concrete value that could fill a tool parameter later, e.g. "
    "'Twitter', 'uv', '7'. Short; no sentences.\n"
    "3. source_text = the exact dialogue sentence supporting the value. Copy it "
    "verbatim; never paraphrase or shorten.\n"
    "4. Extract only explicitly stated facts; no inference.\n"
    "5. Ignore small talk. Multiple facts are expected.\n"
    "If nothing memory-worthy exists, output []."
)


def _parse_cards(text: str, session_id: str) -> list[dict]:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    cards = []
    if not isinstance(arr, list):
        return []
    for i, c in enumerate(arr):
        if not isinstance(c, dict):
            continue
        attr, val, src = (str(c.get(k, "")).strip() for k in ("attribute", "value", "source_text"))
        if attr and val and src:
            cards.append({
                "attribute": attr,
                "value": val,
                "source_text": src,
                "session_id": session_id,
                "turn_index": i,
            })
    return cards


def extract_session(llm, session_id: str, dialogue_text: str) -> list[dict]:
    """One LLM call per session; returns fact cards."""
    out = llm.chat(SYSTEM_PROMPT, dialogue_text[:12000])
    return _parse_cards(out, session_id)
