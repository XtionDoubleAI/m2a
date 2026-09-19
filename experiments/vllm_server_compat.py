"""vLLM OpenAI server entrypoint with the transformers-5 tokenizer shim applied.

The bare `python -m vllm.entrypoints.openai.api_server` crashes under
transformers 5.x (vllm 0.6.4 reads `all_special_tokens_extended`, removed
upstream); OfflineLLM already patches this at import time -- reuse the same
shim here so server and offline paths stay identical.

Usage: python -X utf8 experiments/vllm_server_compat.py --model ... (same
flags as the api_server module).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forge.act.llm import _patch_transformers_for_vllm  # noqa: E402

_patch_transformers_for_vllm()

import vllm.entrypoints.openai.api_server as api_server  # noqa: E402

if __name__ == "__main__":
    # replicate the module's __main__ block (executing the module directly
    # would re-import vllm without the tokenizer shim)
    import uvloop

    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser, validate_parsed_serve_args)
    from vllm.utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(description="vLLM OpenAI-compatible server (compat)")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    uvloop.run(api_server.run_server(args))
