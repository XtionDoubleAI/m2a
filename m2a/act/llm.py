"""OpenAI-compatible client wrapper for the local vLLM server."""

from __future__ import annotations

import json
import re


class LLMClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000/v1",
                 model: str = "Qwen/Qwen2.5-7B-Instruct", temperature: float = 0.0,
                 max_tokens: int = 512):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


def parse_tool_call_json(text: str) -> tuple[str | None, dict]:
    """Best-effort extraction of {"name": ..., "arguments": {...}} from model output.

    Returns (tool_name, arguments); (None, {}) if nothing parseable is found.
    """
    # direct fenced or bare JSON object
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None, {}
    candidates = []
    raw = m.group(0)
    candidates.append(raw)
    # fenced inner block if present
    fm = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fm:
        candidates.insert(0, fm.group(1))
    for c in candidates:
        try:
            d = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict):
            name = d.get("name") or d.get("tool") or d.get("tool_name")
            args = d.get("arguments") or d.get("parameters") or d.get("args")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if name and isinstance(args, dict):
                return str(name), args
            if isinstance(args, dict) and args:
                return "", args
    return None, {}
