# FORGE — Fact-Oriented Retrieval and Grounding Engine

FORGE compiles long-term memory into grounded tool-call arguments. Given an
underspecified user request and a tool's parameter schema, it derives per-slot
retrieval demands from the schema contract, retrieves from a hybrid store
(distilled fact cards + lossless dialogue chunks), and binds parameter values
with an explicit division of labour: the model chooses among candidates,
deterministic code handles types.

Benchmark: Mem2ActBench (ACL 2026). All experiments run on a single
consumer GPU with Qwen2.5-7B-Instruct (temperature 0).

Results (full 400 tasks, given-tool protocol, paper-comparable F1):

| system | F1 |
|---|---|
| paper best baseline (A-Mem) | 30.99 |
| passive retrieval baseline (our reproduction, aligned with paper hybrid@5) | 30.68 |
| **FORGE (fact store only, ablation)** | 16.23 |
| **FORGE (hybrid store + fabrication guard, ablation)** | 27.72 |
| **FORGE (hybrid store, final)** | **33.88** |
| oracle evidence supply (paper) | 53.8 |

Work in progress — architecture docs, ablations and reproduction commands are
being documented in `Pdev/`.
