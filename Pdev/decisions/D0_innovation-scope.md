# D0: Innovation scope and central metaphor

## Background

Mem2ActBench (ACL 2026) diagnoses a 23-point F1 gap between best passive
retrieval (30.7) and oracle evidence supply (53.8) on memory-grounded tool
calls, with default-value hallucination and complex-value corruption as the
top error modes. m2a must close this gap with a mechanism that is attributable,
not score-hacking.

## Options

- C1 Schema-as-Retrieval-Demand: tool parameter schema drives what to retrieve
- C2 Tri-state parameter contract + deterministic binding (memory-bound /
  schema-default / undetermined; verbatim copy for complex values)
- C3 Evolution-aware version chains for conflicting preferences
- C4 Unifying metaphor: the memory-action interface as a compilation pipeline
  (demand analysis -> evidence fetch -> binding -> validation), with C1-C3 as passes

## Decision (2026-09-17)

- C4 adopted as the project's framing; C1 + C2 implemented in full, C3 in a
  lightweight form (newest-value-wins version chain with timestamps; no full
  refinement/coexistence semantics).
- "Compiler" is the central metaphor of the public narrative.
- C1 demand generation uses one round of deterministic coverage check against
  `required`, with at most one repair round.

## Rationale

C1+C2 attack 81% of the observed error space (retrieval miss + default
hallucination + value corruption) with low risk and testable determinism.
C3-full has poor cost/benefit (L4 is 11% of samples); the lightweight chain
covers conflict resolution without LLM-heavy semantics. The compiler framing
organizes all mechanisms into an ablatable pipeline, which doubles as the
experiment design.
