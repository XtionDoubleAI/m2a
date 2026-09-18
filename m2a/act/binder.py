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
    "1. For each parameter, look under its heading. If a CANDIDATE VALUE is listed, "
    "your output value for that parameter MUST be that exact string (copy it "
    "character-for-character; never retype, paraphrase or shorten it).\n"
    "2. If multiple candidate values are listed under one parameter, pick the one "
    "supported by the most recent record (highest 'recorded in' session/turn).\n"
    "3. If no evidence is found for a parameter, choose the most reasonable value "
    "permitted by the schema.\n"
    "4. Include only parameters that are needed; do not invent extra ones.\n"
    "Output JSON only."
)


def _session_num(sid: str) -> int:
    import re
    m = re.search(r"(\d+)", str(sid))
    return int(m.group(1)) if m else 0


def _coerce(value: str, ptype: str):
    """Convert a card value to the parameter's declared type; None if incompatible."""
    v = value.strip()
    if ptype == "string":
        return v
    if ptype in ("integer", "number"):
        try:
            return int(v) if ptype == "integer" else float(v)
        except ValueError:
            # numeric words (e.g. "seven") are left to the model
            return None
    if ptype == "boolean":
        if v.lower() in ("true", "yes"):
            return True
        if v.lower() in ("false", "no"):
            return False
        return None
    return None  # arrays/objects: keep the model's answer


def deterministic_override(spec: ToolSpec, demand_evidence, args: dict) -> dict:
    """Minimal C2 binding (whitelist form), activated early by smoke-run
    evidence. Type-compatible card values form the trusted set for a
    parameter: if the model's answer already lies in that set, keep it (the
    model is good at CHOOSING among candidates); if the answer lies outside
    (fabrication), substitute the newest candidate. Unconditional top-1
    override was tried and rejected: it amplifies retrieval-ranking errors by
    overwriting correct model choices."""
    out = dict(args)
    for demand, hits in demand_evidence:
        p = spec.param(demand.get("param_name", ""))
        if p is None or not hits:
            continue
        ranked = sorted(hits, key=lambda h: (_session_num(h[0].get("session_id", "")),
                                             int(h[0].get("turn_index", 0))))
        candidates = []
        for card, _score in ranked:
            v = _coerce(str(card.get("value", "")), p.type)
            if v is not None:
                candidates.append(v)
        if not candidates:
            continue
        model_v = _canon(out.get(p.name))
        if model_v not in {_canon(c) for c in candidates}:
            out[p.name] = candidates[-1]  # newest compatible candidate
    return out


def _canon(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v).strip().casefold()


def bind(llm, spec: ToolSpec, query: str, evidence_text: str,
         demand_evidence=None) -> tuple[str | None, dict]:
    user = (
        f"Tool schema:\n{json.dumps({'name': spec.name, 'description': spec.description, 'parameters': {'properties': {p.name: {'type': p.type, 'description': p.description, **({'enum': p.enum} if p.enum else {})} for p in spec.params}, 'required': [p.name for p in spec.required_params()]}}, ensure_ascii=False, indent=1)}\n\n"
        f"Memory evidence:\n{evidence_text}\n\n"
        f"User request: {query}"
    )
    out = llm.chat(SYSTEM_PROMPT, user)
    name, args = parse_tool_call_json(out)
    if name is None and args:
        name = spec.name  # model returned arguments without a name; tool is given
    if demand_evidence is not None:
        args = deterministic_override(spec, demand_evidence, args or {})
    return name, args
