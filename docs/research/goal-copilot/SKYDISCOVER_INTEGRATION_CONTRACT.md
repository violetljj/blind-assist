# SkyDiscover integration contract for GOAL-COPILOT-1

状态：`current / interface-contract / proposal-only`

## Direction

```text
BlindAssist SearchTaskBundle
  -> external proposal/search runtime
  -> CandidateBundle
  -> BlindAssist import
  -> BlindAssist independent assessment
```

The external runtime must verify `manifest.json`, every checksum, protocol identity, and
content-addressed directory name before reading the candidate interface. It may edit only
the exported `initial_policy.py` surface and must emit the result as
`candidate/policy.py`. It must not add an evaluator, hidden truth, safety rule, scoring
implementation, task-definition change, or validation gate to the proposal.

## CandidateBundle V1

```text
<candidate_id>/
├── candidate/
│   └── policy.py
├── candidate_manifest.json
├── provenance.json
├── search_metrics.json
└── checksums.json
```

`candidate_manifest.json` binds `protocol_id`, `candidate_id`, source SearchTaskBundle
digest, and the exact candidate file allowlist. `provenance.json` records the SkyDiscover
commit, search configuration, parent candidate, generation/iteration, model/provider,
resource usage, and source digest. `search_metrics.json` is explicitly
`PROVENANCE_ONLY_NOT_ACCEPTANCE`. `checksums.json` covers every file except itself.

Any protocol, source digest, candidate ID, checksum, member, or candidate-surface mismatch
must return `IMPORT_REJECTED`; neither side may silently repair it. Re-emitting the same
mock proposal for the same source bundle must produce byte-identical files and the same
candidate ID.

## V0 restriction

V0 permits only a deterministic zero-model mock adapter. It does not authorize formal
Sky, EvoX, multi-arm, fresh/blind, perception, device, or product evaluation. A later
`GOAL-COPILOT-1-SKY-PILOT` requires a separate explicit protocol and budget.

