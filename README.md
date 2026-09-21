# FORGE — Fact-Oriented Retrieval and Grounding Engine

FORGE binds long-term memory into grounded tool-call arguments. Given an
underspecified user request and a tool's parameter schema, it derives
per-slot retrieval demands from the schema contract, retrieves lossless
dialogue chunks, renders evidence per parameter, and binds values with an
explicit division of labour: the model chooses among candidates,
deterministic code handles types — and nothing else (the minimal
deterministic surface is proved by a set of negative results, see
`docs/D3`–`D6` references below).

Benchmark: Mem2ActBench (ACL 2026), 400 tasks / 825 arguments / 429
sessions, given-tool protocol. All systems run on one consumer GPU with
Qwen2.5-7B-Instruct (temperature 0) **on the same answer harness** — the
compared variable is the memory layer only. Full experiment settings:
`EXPERIMENT_SETUP.md`.

## Results (full 400 tasks, paper-comparable F1, value-token metric)

| system | F1 | write-side LLM calls | notes |
|---|---|---|---|
| passive retrieval baseline (reproduction, aligned with paper hybrid@5 = 30.7) | 30.68 | 0 | |
| **FORGE (lossless chunk store)** | **36.88** | 0 | +6.19 over baseline, 95% CI [+1.72, +10.82] |
| Mem0 (thin reimpl, same harness) | 39.19 | 1,271 | official library terminated on write cost (O(N²) merge loop) |
| A-Mem (thin reimpl, same harness) | 41.11 | 6,025 | |
| oracle evidence supply (paper) | 53.8 | — | |

Paper-protocol numbers (Mem0 14.21, A-Mem 30.99) are not directly
comparable: the reading harness itself is a ~10-point variable. Paper
figures come from the benchmark paper's Table 3/4 (local copy in
`../Pdev/refs/papers/`, excerpts in `../Pdev/refs/INDEX.md`); every project
number is independently recomputable from the sample archives via
`experiments/audit_numbers.py`.

### Ablation chain (every row is a full run, CIs in `results/ci_table.json`)

| configuration | F1 |
|---|---|
| distilled fact cards only | 16.23 |
| hybrid store + fabrication guard | 27.72 |
| hybrid store, no guard | 33.88 |
| lossless chunks (final) | 36.88 |
| — no per-parameter demands | 35.18 |
| — flat render | 37.74 |
| + cross-encoder rerank | 37.08 |
| + candidate ordering | 36.65 |
| + verbatim anchoring (0.5 / 0.1) | 36.65 |

Headline findings: the +6.2 total edge is architectural — no single
mechanism or component reaches significance on its own; distillation
*systems* beat lossless supply on F1 (individually short of significance)
with a measurably different absolute error profile (miss 44–52% of all
gold arguments vs 40% for FORGE; corruption 14–18% vs 25%; fabrication
3–4% vs 5%). A layered hybrid routing evidence through note anchors was
tested: 34.46 F1 — the profile difference did not convert into gain.
Full multi-dimensional analysis in `experiments/multimetrics.py` and
`results/multimetrics.json` (absolute per-argument rates, plus internal
error-structure shares).

## Repo layout

```
forge/            the harness: schema, state, act (intent/retrieve/render/
                  binder), eval (dataset/metrics/runner)
experiments/      runners, ablations, baselines, multi-dimensional metrics,
                  significance tests, paper figures (all data-driven)
results/          run registry (runs.jsonl), per-task archives, aggregates
tests/            unit tests
EXPERIMENT_SETUP.md   full experimental settings and deviations of the thin
                      reimplementations
```

## Reproduce

```bash
# WSL, venv with vllm 0.6.4 + transformers 5.x
python -X utf8 -m experiments.run_baseline      # LTMemory anchor
python -X utf8 -m experiments.run_forge         # FORGE final (chunks)
python -X utf8 -m experiments.run_amem          # A-Mem thin reimpl (needs server)
python -X utf8 -m experiments.run_mem0_thin     # Mem0 thin reimpl (needs server)
python -X utf8 -m experiments.multimetrics      # six-dimension table
python -X utf8 -m experiments.ci_table          # paired bootstrap CIs
python -m experiments.plots                     # paper figures
```

The vLLM OpenAI server entrypoint needs the transformers-5 shim included
in `experiments/vllm_server_compat.py` (see its docstring).

Work in progress — architecture docs and decision records live in the
development workspace.
