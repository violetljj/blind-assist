# Goal Copilot optimizer bridge

状态：`current / P0_GROUNDING_CONTRACT_V1_REFERENCE_SET_ADDENDUM / MAPILLARY_TOKEN_READY / GROUNDING_DINO_PROPOSAL_PATH_RUN / SILVER_B_DEVELOPMENT_COHORT_4 / NO_BASELINE / LEGACY_GC1_SKY_BRIDGE_CLOSED`

Dynamic truth: [`docs/research/goal-copilot/README.md`](../../../docs/research/goal-copilot/README.md).

## P0 Goal Grounding mechanics

`p0_grounding/` contains the stdlib-only, no-model contract mechanics for Named Building Entrance Grounding:

- `p0_episode_schema.json` and `p0_output_schema.json` freeze episode and output shapes;
- `p0_evaluator.py` separates Provider availability, Brain selection given availability, end-to-end P0 outcomes,
  and P1 handoff binding;
- `test_p0_evaluator.py` embeds deterministic mock fixtures for correct grounding, provider miss, wrong instance,
  target absence, ambiguity, stale slow evidence, identity/spatial errors, invalid observation and handoff drift.

This surface does not run a provider, model, cohort, baseline, persistence, Sky or Android path. Focused check:

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest scripts/research/goal_copilot_bridge/p0_grounding/test_p0_evaluator.py
```

## P0-S0 silver materialization canary

`p0_s0_materialization/` is the stdlib-only, fail-closed source-normalization and admission slice for
`BA-P0-GOAL-GROUNDING-SILVER-V1`. `source_slice.py` summarizes bounded Overture/OSM GeoJSON/XML,
preserves source licenses and IDs, computes metric containment/boundary crosswalk candidates, and grants no
episode admission. `materializer.py` applies frozen map/geometry/multiview/conflict/lineage gates to at most
20 already-normalized records, audits provider-visible input for evaluator-only leakage, hashes canonical JSON,
and repeats materialization to require deterministic replay.

The 2026-08-21 real Ghent source slice closed as `P0_S0_SOURCE_OR_LICENSE_BLOCKED`: Mapillary requires a token
that is not configured, while a visual candidate generator is mandatory under the frozen bbox schema but was
explicitly outside this zero-model run. No episode or model metric was emitted. Result and exact unblock conditions:
[`P0-S0 result`](../../../docs/research/goal-copilot/P0_S0_SILVER_MATERIALIZATION_CANARY_RESULT_2026-08-21.md).

`P0-S0-V0` then audited the exact upstream checkpoint without downloading or running it. The checkpoint bytes are
identifiable, but training-data provenance, a replay-complete inference envelope, entrance-only filtering, and required
per-candidate provenance are not established, so the generator is `NOT_ADMITTED`. A stdlib validator keeps its authority
at `VISUAL_PROPOSAL_ONLY`. This remains the historical verdict for that exact YOLO checkpoint; the later simplified
proposal-only policy did not retrofit or rerun V0. See the
[`V0 result`](../../../docs/research/goal-copilot/P0_S0_VISUAL_CANDIDATE_GENERATOR_ADMISSION_RESULT_2026-08-21.md).

The successor uses pinned `IDEA-Research/grounding-dino-tiny` with proposal-only authority. Training-data provenance
incompleteness is recorded as a limitation rather than a proposal-generation blocker. The 20-image anchor-facing run
produced 177 proposals and one nominal automatic `SILVER_A_PRIMARY`, but the post-run contact sheet showed that the
cross-sequence proposal and the other three proposals were different physical entrances. The nominal Silver is not
accepted; the next mechanics fix is crossview same-region correspondence, not detector comparison or threshold tuning.
See the [`S0-R1 result`](../../../docs/research/goal-copilot/P0_S0_GROUNDING_DINO_R1_RESULT_2026-08-21.md).

`P0-S1 Crossview Entrance Identity` now runs as an independent post-materializer gate. It freezes strong identity to
same-sequence pairs satisfying timestamp, 3–30 m baseline, 10–120 degree ray diversity, local-wall position, shape/scale,
and deterministic crop-appearance gates. Cross-sequence pairs are support-only. On the consumed 20-image canary, all
three same-sequence pairs had only 4–12 mm baseline, while all three proper-baseline pairs were cross-sequence and
appearance-inconsistent. Verdict: `P0_S1_IDENTITY_RULE_TOO_WEAK`, 0 strong identities, no S0 rerun. See the
[`P0-S1 result`](../../../docs/research/goal-copilot/P0_S1_CROSSVIEW_ENTRANCE_IDENTITY_RESULT_2026-08-21.md).

`silver_b_development.py` prospectively exports parent A/B admissions at the lower `SILVER_B_MAP_GEOMETRY`
Development authority without modifying either parent result. The first consumed real batch contains 4 frame-level
episodes / 4 weak candidates. It supports pipeline/yield/ranking-prototype diagnostics, not detector recall, exact Brain
accuracy or exact-entrance truth. Each B episode is explicitly `AMBIGUOUS` until independent evidence establishes a
unique or set-valued physical referent. See the short
[`Silver-B addendum`](../../../docs/research/goal-copilot/P0_SILVER_B_DEVELOPMENT_ADDENDUM_V1.md).

Focused check:

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest `
  scripts/research/goal_copilot_bridge/p0_s0_materialization/test_materializer.py `
  scripts/research/goal_copilot_bridge/p0_s0_materialization/test_source_slice.py `
  scripts/research/goal_copilot_bridge/p0_s0_materialization/test_candidate_generator_admission.py `
  scripts/research/goal_copilot_bridge/p0_s0_materialization/test_grounding_dino_s0_r1.py `
  scripts/research/goal_copilot_bridge/p0_s0_materialization/test_crossview_identity.py `
  scripts/research/goal_copilot_bridge/p0_s0_materialization/test_silver_b_development.py
```

Authority: BlindAssist owns `GOAL-COPILOT-1`, its evaluator, sealed scenarios,
acceptance decision, and claim ceiling. External optimizers have proposal authority
only. This module does not activate a scientific comparison or change the default App.

## 稳定 Interface

`sky_bridge.py export` freezes a content-addressed `SearchTaskBundle` containing only
the typed public task API, public scenario descriptions, protocol, README, and baseline
policy. It deliberately excludes `evaluator.py` and `sealed_scenarios.json`. Each payload has a SHA-256 and the
bundle identity is the SHA-256 of canonical `checksums.json` content.

An optimizer returns a content-addressed `CandidateBundle` with exactly
`candidate/policy.py`, `candidate_manifest.json`, `provenance.json`,
`search_metrics.json`, and `checksums.json`. BlindAssist rejects extra
members, payload drift, identity drift, source-bundle drift, unexpected functions, and
Python syntax outside the restricted policy contract before importing. Validation runs
only the BlindAssist-owned sealed scenarios, reports the complete goal/progress/recovery/
safety vector, applies safety and premature-completion hard gates, and binds evaluator/
truth hashes into the deterministic assessment.

```powershell
python scripts/run_research_tool.py goal-copilot sky_bridge.py export
python scripts/run_research_tool.py goal-copilot sky_bridge.py import `
  --candidate <candidate-bundle> --task-bundle <search-task-bundle>
python scripts/run_research_tool.py goal-copilot sky_bridge.py validate `
  --candidate <imported-candidate> --task-bundle <search-task-bundle>
```

## 输出

Generated packages and assessments belong under ignored `artifacts.local/`; none is
scientific evidence. Artifact root:
`artifacts.local/sky_{exports,imports,validations}/GOAL-COPILOT-1/`.

## 安全边界

Sky score is provenance only. The public task bundle never contains evaluator or sealed
truth. Candidate scope, protocol, source digest, member allowlist, restricted AST, and
checksums fail closed. A later model-backed or claim-bearing run needs its own owning
research route and protocol.

## 停止条件

Stop on any checksum, protocol, source, member, candidate-surface, evaluator, or replay
mismatch. V0 stops after the mock roundtrip and must not start Sky/EvoX model search.

Focused mechanics test:

```powershell
python -m unittest scripts/research/goal_copilot_bridge/test_bridge.py
```

Successor: connect a versioned Development task only after this no-model dry-run remains
replayable across the external adapter boundary.
