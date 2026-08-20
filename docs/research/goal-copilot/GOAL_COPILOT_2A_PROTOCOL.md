# GOAL-COPILOT-2A protocol

Status before execution: `GOAL_COPILOT_2A_PROTOCOL_FROZEN_ZERO_MODEL_RUN_AUTHORIZED`.

GOAL-COPILOT-1 is permanently closed. No further GC1 model search is authorized.
GC2-A is a deterministic, zero-model characterization of the frozen GC1 baseline
and frozen GC1 Sky winner under imperfect candidate-visible observations. It does
not reopen the sealed GC1 fresh cohort and does not call Sky, EvoX, Codex, or any
other model provider.

## Frozen inputs

- baseline: `scripts/research/goal_copilot_bridge/pilot/initial_policy.py`;
- winner: byte-identical copy with SHA-256
  `24d4e57374dd99363700ae881d18db536e48ec5f79f39e95c5b873e96edbc3a1`;
- scenario semantics: the 12 consumed GC1 development scenarios only;
- evidence role: `CONSUMED_DEVELOPMENT_TRANSFER_CHARACTERIZATION_NOT_FRESH`;
- model-call budget: `0`.

The BA-owned hidden state, expected actions, completion authority, safety gates,
and acceptance decision remain outside the candidate-visible observation stream.
The candidate sees only the deterministically corrupted observation.

## Frozen condition matrix

The matrix has 22 cells: one clean control, each of six corruptions at `MILD`,
`MODERATE`, and `STRESS`, and a combined condition at each severity. The six
corruptions are target dropout, bearing jitter, false target, nearness error,
tracking-quality collapse, and delayed evidence.

The primary admission cell is `COMBINED_MODERATE`. GC2-B protocol design is
admitted if the frozen winner has any of:

- completion rate below `0.8`;
- unsafe guidance above `0`;
- premature completion above `0`;
- eligible true reacquisition rate below `0.8`.

Admission authorizes only a separately frozen GC2-B protocol-design step. It
does not authorize GC2-B model calls or a Sky/EvoX search. If the admission gate
does not fire, the next route is GOAL-COPILOT-3 recorded-RGB evidence protocol
design.

## Claim ceiling

The strongest permitted claim is a deterministic robustness characterization on
consumed symbolic development scenarios. It is not fresh evidence, real-RGB
evidence, device evidence, a comparison of perception models, or proof of field
robustness or product readiness.
