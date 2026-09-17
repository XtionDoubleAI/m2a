"""Binder (MVP): LLM binding over slot-paired evidence.

Given-tool protocol: the tool is provided, the model must emit its JSON call.
Later toggles (D3 territory, off by default here):
  - schema_default_fill: parameters whose schema declares a default and whose
    evidence is absent are filled deterministically, never by the model
  - verbatim_copy: long/complex values are copied character-exact from
    source_text instead of being regenerated

The MVP binder deliberately does none of that yet: it only changes HOW evidence
is presented (slot-paired) versus the LTMemory baseline.
"""

from __future__ import annotations

import json

from m2a.act.llm import parse_tool_call_json
from m2a.schema import ToolSpec

SYSTEM_PROMPT = (
    "You are a tool-calling assistant. The target tool's schema and slot-by-slot "
    "memory evidence are provided; you MUST call exactly this tool. Output a single "
    "JSON object: {\"name\": <tool name>, \"arguments\": {<param>: <value>, ...}}.\n"
    "Rules:\n"
    "1. For each parameter, use its paired evidence under that parameter's heading.\n"
    "2. When a value appears in evidence, copy it EXACTLY as written (identifiers, "
    "URLs, long strings -- character for character).\n"
    "3. When evidence contradicts itself across time, prefer the most recent "
    "(highest 'recorded in' session/turn).\n"
    "4. When a parameter has no evidence, choose the most reasonable value permitted "
    "by the schema.\n"
    "5. Include only parameters that are needed; do not invent extra ones.\n"
    "Output JSON only."
)


def bind(llm, spec: ToolSpec, query: str, evidence_text: str) -> tuple[str | None, dict]:
    user = (
        f"Tool schema:\n{json.dumps({'name': spec.name, 'description': spec.description, 'parameters': {'properties': {p.name: {'type': p.type, 'description': p.description, **({'enum': p.enum} if p.enum else {})} for p in spec.params}, 'required': [p.name for p in spec.required_params()]}}, ensure_ascii=False, indent=1)}\n\n"
        f"Memory evidence:\n{evidence_text}\n\n"
        f"User request: {query}"
    )
    out = llm.chat(SYSTEM_PROMPT, user)
    name, args = parse_tool_call_json(out)
    if name is None and args:
        name = spec.name  # model returned arguments without a name; tool is given
    return name, args
