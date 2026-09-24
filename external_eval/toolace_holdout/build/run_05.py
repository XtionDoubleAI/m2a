"""Step 05 adapter: normalize pipeline outputs into the frozen benchmark pair.

Loads the upstream BenchmarkNormalizer class verbatim (importlib, file name
starts with a digit) and points it at the build-dir intermediates. The
"small" split machinery is reused with an unattainable count so every QA is
kept (no subsampling of an already-sampled held-out set), and the conversation
file is renamed Mem2ACT_conversation.jsonl -> toolmem_conversation.jsonl to
match the toolmembench_small layout exactly.

Usage (WSL, run from this build dir):
    python run_05.py --qa processed_data/qa_raw.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

UPSTREAM = ("/mnt/e/hx/_DoctorXtion/intern_shxt/Pdev/refs/repos/"
            "Mem2ActBench/05_benchmark_normalization.py")

spec = importlib.util.spec_from_file_location("upstream_05", UPSTREAM)
up = importlib.util.module_from_spec(spec)
sys.modules["upstream_05"] = up
spec.loader.exec_module(up)

HERE = Path(__file__).resolve().parent          # build/
HOLDOUT = HERE.parent                            # toolace_holdout/


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversation-csv",
                    default="processed_data/conversation_sequence.csv")
    ap.add_argument("--qa", required=True,
                    help="raw QA jsonl from step 04")
    ap.add_argument("--keep-all", type=int, default=10**6,
                    help="small-split sample size; default keeps every QA")
    args = ap.parse_args()

    out_full = HERE / "Mem2ACTBench"
    normalizer = up.BenchmarkNormalizer(args.conversation_csv, args.qa,
                                        str(out_full))
    normalizer.run(create_small=True, small_qa_count=args.keep_all)

    small = HERE / "Mem2ACTBench_small"
    qa_dst = HOLDOUT / "qa_dataset.jsonl"
    conv_dst = HOLDOUT / "toolmem_conversation.jsonl"
    shutil.copyfile(small / "qa_dataset.jsonl", qa_dst)
    shutil.copyfile(small / "Mem2ACT_conversation.jsonl", conv_dst)

    n_qa = sum(1 for _ in open(qa_dst, encoding="utf-8"))
    n_conv = sum(1 for _ in open(conv_dst, encoding="utf-8"))
    print(f"frozen pair written: {qa_dst.name} ({n_qa} QA), "
          f"{conv_dst.name} ({n_conv} sessions)")


if __name__ == "__main__":
    main()
