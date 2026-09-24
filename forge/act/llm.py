"""OpenAI-compatible chat client: local vLLM server or any hosted API.

The same wire format serves both regimes:
  - local vLLM:  base_url=http://127.0.0.1:8000/v1, api_key left as "EMPTY"
  - hosted closed models (scale experiments, see D8): base_url and api_key
    read from FORGE_API_BASE / FORGE_API_KEY environment variables when
    the arguments are not given explicitly. Keys are never hard-coded;
    ask the user for credentials when a run needs them.
"""

from __future__ import annotations

import json
import os
import re


class LLMClient:
    def __init__(self, base_url: str | None = None,
                 model: str = "Qwen/Qwen2.5-7B-Instruct", temperature: float = 0.0,
                 max_tokens: int = 512, api_key: str | None = None):
        from openai import OpenAI
        self.client = OpenAI(
            base_url=base_url or os.environ.get("FORGE_API_BASE",
                                                "http://127.0.0.1:8000/v1"),
            api_key=api_key or os.environ.get("FORGE_API_KEY", "EMPTY"))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, system: str, user: str, extra_body: dict | None = None) -> str:
        kwargs = {}
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
        self.last_usage = getattr(resp, "usage", None)
        return resp.choices[0].message.content or ""


def _patch_transformers_for_vllm() -> None:
    """Compatibility shim: vllm 0.6.4 reads tokenizer attributes removed in
    transformers 5.x. Patch at class level before vllm loads its tokenizer."""
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
        if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
            PreTrainedTokenizerBase.all_special_tokens_extended = property(
                lambda self: self.all_special_tokens)
    except Exception:
        pass


class OfflineLLM:
    """In-process vLLM engine (offline mode).

    Preferred on this machine: the OpenAI server entrypoint depends on
    Linux-only uvloop, while the offline engine runs natively on Windows and
    gives lower per-request latency for our sequential runner.
    """

    def __init__(self, model_path: str, temperature: float = 0.0, max_tokens: int = 512,
                 gpu_memory_utilization: float = 0.92, max_model_len: int = 16384):
        _patch_transformers_for_vllm()
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=model_path, gpu_memory_utilization=gpu_memory_utilization,
                       max_model_len=max_model_len, enforce_eager=False)
        self.sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    def chat(self, system: str, user: str) -> str:
        outs = self.llm.chat(
            [[{"role": "system", "content": system}, {"role": "user", "content": user}]],
            sampling_params=self.sp, use_tqdm=False,
        )
        return outs[0].outputs[0].text


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
