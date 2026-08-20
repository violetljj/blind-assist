# Goal Copilot optimizer bridge

状态：`current / GOAL_COPILOT_SKY_BRIDGE_V0_MECHANICS_READY / GOAL_COPILOT_1_MODEL_SEARCH_NOT_STARTED`

Dynamic truth: [`docs/research/goal-copilot/README.md`](../../../docs/research/goal-copilot/README.md).

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
