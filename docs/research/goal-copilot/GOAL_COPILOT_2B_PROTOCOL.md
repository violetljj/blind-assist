# GOAL-COPILOT-2B protocol design

Status: `GOAL_COPILOT_2B_PROTOCOL_DESIGN_FROZEN_SEARCH_NOT_AUTHORIZED`.

GC2-B is the admitted successor to GC2-A's deterministic robustness
characterization. This document freezes the proposed search question, authority
split, candidate surface, budgets, selection rule, held-out acceptance gates,
and failure semantics. It does not authorize model calls, bundle/held-out
materialization, formal Sky/EvoX execution, or a multi-arm experiment.

BlindAssist exclusively owns task semantics, hidden state, corruption schedules,
evaluator, safety gates, winner lock, held-out opening, and acceptance. SkyDiscover
may propose and search candidate policies only; its development metrics remain
provenance and cannot issue the scientific verdict.

The starting policy is the frozen GC1 winner with SHA-256
`24d4e57374dd99363700ae881d18db536e48ec5f79f39e95c5b873e96edbc3a1`.
Only its existing six bounded policy functions are searchable. Sky cannot modify
the evaluator, noise engine, task material, hidden state, or gates.

If a later formal run is separately sealed, its maximum budget is two independent
replicates of 16 generation attempts, 32 total, with no generation or evaluator
retries. A started-only dispatch is `IN_DOUBT` and consumes its opportunity.

Search guidance uses CLEAN, six isolated MODERATE corruptions, COMBINED_MILD, and
COMBINED_MODERATE over consumed symbolic task semantics. Admission to held-out
validation requires zero hard-gate violations, CLEAN `12/12`, COMBINED_MILD at
least `10/12`, COMBINED_MODERATE at least `8/12`, at least `2/4` completion in
every family, and eligible reacquisition at least `2/3`.

Held-out validation uses two BA-only corruption schedules whose seeds are frozen
in `protocol.json`. They remain hidden until candidate lock and are explicitly
`HELD_OUT_CORRUPTION_SCHEDULES_OVER_CONSUMED_TASK_SEMANTICS_NOT_FRESH_TASKS`.
They do not create fresh task, real-RGB, device, user, or product evidence.

Before any model call, a later step must materialize and seal the public bundle
and encrypted held-out envelope, bind exact BA and Sky commits, freeze the native
Codex executable/version/SHA-256, pass a zero-model transport canary, and create
a separate formal run seal that explicitly authorizes calls. None of those
execution prerequisites is authorized by this design closeout.

Claim ceiling:
`symbolic_consumed_task_noise_robust_search_protocol_design_only`.
