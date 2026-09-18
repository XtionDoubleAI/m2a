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


def deterministic_override(spec: ToolSpec, demand_evidence, args: dict,
                           trusted_texts: str | None = None) -> dict:
    """Minimal C2 binding (whitelist form), activated early by smoke-run
    evidence. The trusted set for a parameter = type-compatible card values,
    PLUS (hybrid store) any value substantiated by the verbatim chunk texts.
    The model's answer is kept if it lies in the trusted set; otherwise it is
    treated as fabrication and replaced by the newest card candidate.

    History (kept for the paper's ablation story): unconditional top-1 override
    amplified retrieval-ranking errors; a card-only whitelist then killed
    correct answers the model had read from the lossless chunks -- the trusted
    set must span both stores."""
    out = dict(args)
    trusted_blob = _canon(trusted_texts) if trusted_texts else ""
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
        in_cards = model_v in {_canon(c) for c in candidates}
        in_chunks = bool(trusted_blob) and model_v in trusted_blob
        if not in_cards and not in_chunks:
            out[p.name] = candidates[-1]  # newest compatible candidate
    return out


def _canon(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v).strip().casefold()


def bind(llm, spec: ToolSpec, query: str, evidence_text: str,
         demand_evidence=None, trusted_texts: str | None = None) -> tuple[str | None, dict, dict]:
    """Returns (tool_name, final_args, model_args_pre_override)."""
    user = (
        f"Tool schema:\n{json.dumps({'name': spec.name, 'description': spec.description, 'parameters': {'properties': {p.name: {'type': p.type, 'description': p.description, **({'enum': p.enum} if p.enum else {})} for p in spec.params}, 'required': [p.name for p in spec.required_params()]}}, ensure_ascii=False, indent=1)}\n\n"
        f"Memory evidence:\n{evidence_text}\n\n"
        f"User request: {query}"
    )
    out = llm.chat(SYSTEM_PROMPT, user)
    name, args = parse_tool_call_json(out)
    if name is None and args:
        name = spec.name  # model returned arguments without a name; tool is given
    model_args = dict(args or {})
    if demand_evidence is not None:
        args = deterministic_override(spec, demand_evidence, args or {}, trusted_texts)
    return name, args or {}, model_args
