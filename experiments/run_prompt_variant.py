"""Prompt-sensitivity check: rerun FORGE (with demand fallback) under
semantically equivalent rewordings of the shared answer prompt.

Variant A reframes the instruction as a job description; variant B states the
same constraints as a numbered list. Both preserve the information content of
the base prompt: given-tool protocol, single JSON object output, only-needed
parameters, verbatim copying from evidence, schema-permitted fallback.

Pre-registered reading: the FORGE-over-A-Mem lead and its significance must
stay stable across the three wordings; instability is reported as-is.

Usage (WSL): python -X utf8 -m experiments.run_prompt_variant --variant a|b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import experiments.run_forge as run_forge  # noqa: E402

VARIANTS = {
    "a": (
        "You will be given the API schema of one target tool, a user request, "
        "and memory evidence retrieved from past conversations. Your job: "
        "produce a tool call to this exact tool. Respond with a single JSON "
        'object: {"name": <tool name>, "arguments": {<param>: <value>, ...}}. '
        "Fill only the parameters the request requires and add nothing beyond "
        "the schema. Whenever the evidence contains a value, copy it "
        "character-for-character -- especially identifiers, URLs, and long "
        "strings. When the evidence does not state a value, pick the most "
        "reasonable value the schema allows. JSON only."
    ),
    "b": (
        "Task: emit one tool call for the provided target tool (its schema is "
        "given). Inputs: a user request plus retrieved memory evidence. "
        'Format: a single JSON object {"name": ..., "arguments": {...}}. '
        "Constraints: (1) no tool other than the target; (2) include only "
        "required parameters, no invented ones; (3) argument values must come "
        "from the evidence where present -- exact strings for identifiers, "
        "URLs and long values; (4) values absent from the evidence default to "
        "the most reasonable schema-permitted choice. Output the JSON object "
        "and nothing else."
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    ap.add_argument("--limit", type=int, default=None)
    args, rest = ap.parse_known_args()

    run_forge.SYSTEM_PROMPT = VARIANTS[args.variant]
    sys.argv = [sys.argv[0],
                "--name", f"prompt-variant-{args.variant}",
                *rest]
    run_forge.main()


if __name__ == "__main__":
    main()
