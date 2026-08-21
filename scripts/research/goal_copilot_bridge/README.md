# Goal Copilot optimizer bridge

状态：`current / BLINDASSIST_LAST_10M_REGROUNDING_V0 / MILESTONE_CLOSED / NETWORK_SCENE_3X5_COMPLETE / P1_CLOSED / NO_REFERENT_PERSISTENCE / NO_SCIENTIFIC_VERDICT / NO_SUCCESSOR / LEGACY_GC1_SKY_BRIDGE_CLOSED`

Dynamic truth: [`docs/research/goal-copilot/README.md`](../../../docs/research/goal-copilot/README.md).

## Last-10-metre current-frame regrounding

`last_10m_regrounding_v0/` is the active engineering surface. It reuses the exact frozen Grounding DINO inference
and single-Brain baseline functions, or adapts one externally produced unchanged P0 output, per fresh frame into the
bounded `SCAN -> CURRENT_CANDIDATE -> ALIGN -> ADVANCE_AND_REOBSERVE ->
ARRIVAL_CONFIRM -> COMPLETE / RESCAN / ABSTAIN` control loop. Persistent state deliberately excludes candidates,
regions, images, features, scores, handoffs, and identity; prior details are audit-only JSONL. The runner completed
3 real-world network-scene locations x 5 fixed-playlist episodes, records each event, keeps evaluator truth provider-blind,
and emits only the requested mechanical metrics and three attribution classes. It does not execute P1,
modify the P0 provider, or create scientific/safety authority.

Focused check:

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest `
  scripts/research/goal_copilot_bridge/last_10m_regrounding_v0/test_core.py
```

## P1-R0 Target Persistence mechanics

`p1_persistence/` freezes the stdlib-only representation/evaluator mechanics for an already-established,
episode-local physical referent. It provides separate public-input/evaluator/output JSON schemas, eight synthetic scenario fixtures, a deterministic
identity-safety evaluator, and a deliberately simple fixed-threshold baseline. `NO_REFERENT` is a hard
`UNBOUND` guard; P1 cannot create or replace semantic referent validity. Scores are algorithmic evidence, not
calibrated probabilities, and evaluator-only physical-instance truth is never visible to the baseline.

The evaluator orders candidates lexicographically by illegal bind, wrong-instance assertion, identity switch,
false reacquisition, then correct identity coverage. It also reports wrong-lock persistence duration. This is
synthetic mechanics only: no RGB tracker, ADT rerun, model, Sky, Android, product, safety, or scientific claim.
Contract: [`P1-R0`](../../../docs/research/goal-copilot/P1_R0_TARGET_PERSISTENCE_CONTRACT_V1.md).

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest `
  scripts/research/goal_copilot_bridge/p1_persistence/test_contract.py
```

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
Development authority without modifying either parent result. The expanded reviewed cohort contains 47 goal episodes
over 43 unique frames: 12 `UNIQUE`, 4 `SET_VALUED`, and 31 `AMBIGUOUS`. Independent manual regions, not proposal
scores, establish the resolved sidecar truth. One user-selected `gpt-5.6-terra / medium` Brain run now provides
conditioned selection/abstention mechanics only; it is not detector recall, exact Brain accuracy or end-to-end
performance. See the [`Silver-B addendum`](../../../docs/research/goal-copilot/P0_SILVER_B_DEVELOPMENT_ADDENDUM_V1.md)
and [`Development result`](../../../docs/research/goal-copilot/P0_SILVER_B_BRAIN_DEVELOPMENT_RESULT_2026-08-21.md).

`P0-D1` then tested two prompt-level ambiguity-calibration policies on the same consumed cohort. V1's hard
place-plus-entrance gate and V2's explicit `place_support / entrance_relation_support` both reduced unsupported
commitment, but retained only 4/13 and 2/13 of the baseline's correct groundings. Both are over-refusal negative
canaries and are not admitted. `audit_silver_b_ambiguity_calibration.py` reports both frame-micro and venue-parent-
macro unsupported commitment and uses actual `CORRECT_GROUNDING` retention rather than ranked-candidate top-1.
The successor Brussels acquisition produced a deduplicated 24-goal / 24-frame parent-disjoint Development slice:
20 `AMBIGUOUS`, 4 `UNIQUE`, 10 venue parents, and zero old-cohort target-name or frame overlap. The unchanged
baseline again unsupported-committed on 13/20 ambiguous episodes (9-parent macro 0.7444) while correctly grounding
4/4 UNIQUE episodes with no refusal. This reproduces the calibration failure across venues without admitting a V3
prompt or creating a scientific claim. See the
[`consumed P0-D1 result`](../../../docs/research/goal-copilot/P0_D1_AMBIGUITY_CALIBRATION_CONSUMED_CANARY_RESULT_2026-08-21.md)
and [`parent-disjoint confirmation`](../../../docs/research/goal-copilot/P0_D1_PARENT_DISJOINT_CONFIRMATION_RESULT_2026-08-21.md).

`p0_d2_calibration/` freezes the successor commitment-calibration mechanics without fitting on an inadequate
denominator. `plan_resolvable_enrichment.py` creates an outcome-blind metadata-only roster and balanced Mapillary
anchor shards; `build_enrichment_cohort.py` applies score-blind manual decisions and fails on old target-name/frame
overlap; `core.py` keeps referent-set and ambiguity semantics separate, rejects evaluator-only runtime features, and
computes finite-sample conformal quantiles over parent scores. The first enrichment added 4 UNIQUE and 12 AMBIGUOUS
episodes over 8 new parents, but the combined consumed bank still has only 2 SET_VALUED and 11 resolvable parents.
`audit_frontdoor.py` therefore returns `P0_D2_DATA_FRONTDOOR_INSUFFICIENT`; no logistic or conformal fit ran. Protocol
and result: [`P0-D2 protocol`](../../../docs/research/goal-copilot/P0_D2_COMMITMENT_CALIBRATION_PROTOCOL_V1.md) /
[`frontdoor result`](../../../docs/research/goal-copilot/P0_D2_RESOLVABLE_ENRICHMENT_AND_FRONTDOOR_RESULT_2026-08-21.md).

`p0_a1_ambiguity_gate/` performs the one allowed consumed-development feature sweep after the D3 calibration
frontdoor closed. It hash-binds the two cohorts with complete, matching Grounding DINO + Terra + frozen-evaluator
runtime evidence; missing D2/D3 runtime rows are excluded rather than imputed. Rules may only retain an existing
Terra `SELECT` or turn it into `ABSTAIN`. The frozen eight-feature, single/two-condition sweep reached
`CLEAR_SIGNAL_COMPACT_POLICY_NEXT`; this is feature headroom for P0-A2, not policy admission or a scientific result.
Protocol and result: [`P0-A1 protocol`](../../../docs/research/goal-copilot/P0_A1_AMBIGUITY_GATE_DISCOVERY_PROTOCOL_V1.json) /
[`P0-A1 result`](../../../docs/research/goal-copilot/P0_A1_AMBIGUITY_GATE_DISCOVERY_RESULT_2026-08-21.md).

Focused P0-A1 mechanics check:

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest `
  scripts/research/goal_copilot_bridge/p0_a1_ambiguity_gate/test_sweep.py
```

`p0_a2_compact_policy/` hash-binds the complete A1 evidence chain and deterministically enumerates the frozen
monotone policy DSL: at most three distinct threshold predicates and Boolean depth at most two. Full resolvable
coverage and at least 85% committed correctness are hard constraints; ambiguous venue-parent macro false commit is
the sole primary objective. The search retained A1 exactly; lower false commitment appeared only after coverage fell
to 65%, so the terminal is `COMPLEXITY_ONLY_BUYS_ABSTENTION`. Protocol and result:
[`P0-A2 protocol`](../../../docs/research/goal-copilot/P0_A2_COMPACT_AMBIGUITY_POLICY_DISCOVERY_PROTOCOL_V1.json) /
[`P0-A2 result`](../../../docs/research/goal-copilot/P0_A2_COMPACT_AMBIGUITY_POLICY_DISCOVERY_RESULT_2026-08-21.md).

Focused P0-A2 mechanics check:

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest `
  scripts/research/goal_copilot_bridge/p0_a2_compact_policy/test_search.py
```

Focused D2 mechanics check:

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest `
  scripts/research/goal_copilot_bridge/p0_d2_calibration/test_core.py `
  scripts/research/goal_copilot_bridge/p0_d2_calibration/test_plan_resolvable_enrichment.py
```

Focused check:

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest `
  scripts/research/goal_copilot_bridge/p0_s0_materialization/test_materializer.py `
  scripts/research/goal_copilot_bridge/p0_s0_materialization/test_source_slice.py `
  scripts/research/goal_copilot_bridge/p0_s0_materialization/test_candidate_generator_admission.py `
  scripts/research/goal_copilot_bridge/p0_s0_materialization/test_grounding_dino_s0_r1.py `
  scripts/research/goal_copilot_bridge/p0_s0_materialization/test_crossview_identity.py `
  scripts/research/goal_copilot_bridge/p0_s0_materialization/test_silver_b_development.py `
  scripts/research/goal_copilot_bridge/p0_s0_materialization/test_silver_b_brain_cohort.py `
  scripts/research/goal_copilot_bridge/p0_s0_materialization/test_run_silver_b_brain_baseline.py
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
