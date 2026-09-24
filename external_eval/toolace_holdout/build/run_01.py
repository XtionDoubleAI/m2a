"""Step 01 adapter: LLM-refine the 600 held-out ToolACE conversations.

The upstream script (01_conversation_builder.py, ToolACE subcommand) reads a
contiguous index range from the original data.json; our held-out sources are
600 scattered ids already converted to the standard format. This adapter loads
the upstream module verbatim (importlib, file name starts with a digit) and
reuses its prompt builder, LLM caller, parser and record builder unchanged,
feeding it the held-out source file directly.

Resume: output ids already present are skipped (same rule as upstream).

Usage (WSL): /root/bench-venv/bin/python run_01.py \
    --source ../toolace_heldout_source.jsonl --output 01_conversations.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

UPSTREAM = ("/mnt/e/hx/_DoctorXtion/intern_shxt/Pdev/refs/repos/"
            "Mem2ActBench/01_conversation_builder.py")

spec = importlib.util.spec_from_file_location("upstream_01", UPSTREAM)
up = importlib.util.module_from_spec(spec)
sys.modules["upstream_01"] = up
spec.loader.exec_module(up)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="DeepSeek-V4-Flash")
    ap.add_argument("--api-base", default=os.getenv("FORGE_API_BASE"))
    ap.add_argument("--api-key", default=os.getenv("FORGE_API_KEY"))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    processor = up.ToolACEProcessor("unused")
    client = up.OpenAI(base_url=args.api_base, api_key=args.api_key)

    tasks = [json.loads(l) for l in open(args.source, encoding="utf-8") if l.strip()]
    existing = up.load_existing_ids(args.output)
    tasks = [t for t in tasks if t["id"] not in existing]
    n_empty_before = 0
    print(f"refining {len(tasks)} conversations "
          f"(skipped {len(existing)} existing)", flush=True)

    def process_one(task):
        prompt = processor.build_prompt(task["conversation_history"])
        raw = up.call_llm_with_retry(client, prompt, args.model,
                                     args.temperature)
        if raw:
            try:
                conv = up.parse_llm_json_response(raw)
                return processor.build_record(task, conv), raw
            except json.JSONDecodeError:
                pass
        return processor.build_record(task, []), raw

    n_done = n_empty = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, t): t for t in tasks}
        for fut in as_completed(futs):
            record, raw = fut.result()
            up.append_jsonl(args.output, record)
            n_done += 1
            if not record["conversation_history"]:
                n_empty += 1
            if n_done % 25 == 0:
                print(f"  {n_done}/{len(tasks)} done, "
                      f"{n_empty} empty (LLM unparseable)", flush=True)

    print(f"step 01 done: {n_done} processed, {n_empty} empty records", flush=True)


if __name__ == "__main__":
    main()
