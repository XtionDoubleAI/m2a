"""Demand generation: (query, ToolSpec) -> one retrieval demand per parameter.

D2-A (always on): each demand is a natural-language retrieval question.
D2-B (toggle, default off): each demand also carries an attribute_guess that
matches the write-path attribute vocabulary (same normalize_attribute).

Coverage check (D0): after generation, required parameters lacking a demand
trigger at most one repair round; the check itself is a deterministic set
comparison, no extra LLM call unless something is missing.
"""

from __future__ import annotations

import json
import re

from forge.schema import ToolSpec
from forge.state.store import normalize_attribute

SYSTEM_PROMPT = (
    "You prepare retrieval requests for a memory system. Given a user request "
    "and a tool's parameter schema, output for EACH parameter a retrieval "
    "question asking what the memory must supply to fill it. Output ONLY a JSON "
    "array; no other text. Each element: "
    "{\"param_name\": str, \"query\": str, \"attribute_guess\": str}.\n"
    "Rules:\n"
    "1. query: one question about the USER'S habits/stated preferences/identities "
    "that would determine this parameter's value, e.g. for a 'social' parameter: "
    "\"Which social platform does the user prefer for this kind of content?\" "
    "Do NOT mention parameter names inside the question.\n"
    "2. attribute_guess: a normalized category label in the form "
    "'Topic (Entity)', e.g. 'Platform Preference (finance)'. Your best guess of "
    "how this kind of fact would be categorized.\n"
    "3. Cover every parameter of the tool, required or optional.\n"
    "4. Questions must be about durable user facts, not about the current request "
    "alone."
)


def _parse(text: str) -> list[dict]:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    if isinstance(arr, list):
        for d in arr:
            if isinstance(d, dict) and d.get("param_name") and d.get("query"):
                out.append({
                    "param_name": str(d["param_name"]).strip(),
                    "query": str(d["query"]).strip(),
                    "attribute_guess": normalize_attribute(str(d.get("attribute_guess", ""))),
                })
    return out


def _covered(spec: ToolSpec, demands: list[dict]) -> bool:
    have = {d["param_name"] for d in demands}
    return all(p.name in have for p in spec.required_params())


def _template_demands(spec: ToolSpec) -> list[dict]:
    """Deterministic fallback demands, one per parameter, no LLM involved.

    Gap-A fix (D8 section 4.2): the LLM occasionally returns an empty list,
    which used to be silently passed through, leaving the task with zero
    retrieval. Temperature-0 retry on identical input returns the same
    empty list, so the guaranteed non-empty path is a plain template."""
    return [{
        "param_name": p.name,
        "query": f"What is the value for {p.name}? {p.description or ''}".strip(),
        "attribute_guess": "",
    } for p in spec.params]


def generate_demands(llm, query: str, spec: ToolSpec,
                     fallback: bool = True) -> list[dict]:
    """Generate per-parameter retrieval demands; repair once if required
    parameters are uncovered (D0 protocol). An empty LLM response falls
    back to deterministic template demands when `fallback` is set."""
    schema_text = json.dumps({
        "name": spec.name,
        "description": spec.description,
        "parameters": [
            {"name": p.name, "type": p.type, "description": p.description,
             "required": p.required}
            for p in spec.params
        ],
    }, ensure_ascii=False, indent=1)
    user = f"User request: {query}\n\nTool schema:\n{schema_text}"
    demands = _parse(llm.chat(SYSTEM_PROMPT, user))
    if not demands and fallback:
        return _template_demands(spec)
    if demands and not _covered(spec, demands):
        missing = [p.name for p in spec.required_params() if p.name not in {d["param_name"] for d in demands}]
        user += ("\n\nIMPORTANT: your previous list missed these REQUIRED parameters: "
                 f"{missing}. Regenerate the complete list.")
        demands = _parse(llm.chat(SYSTEM_PROMPT, user)) or demands
    if not demands and fallback:
        return _template_demands(spec)
    # keep only demands that correspond to real parameters; dedup by param
    valid = {p.name for p in spec.params}
    seen, out = set(), []
    for d in demands:
        if d["param_name"] in valid and d["param_name"] not in seen:
            seen.add(d["param_name"])
            out.append(d)
    if not out and fallback:
        return _template_demands(spec)
    return out
