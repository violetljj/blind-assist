# Research workflow upgrade and consumed-result check

Date: 2026-09-05

This delivery reorganizes research decisions and tests the new workflow on existing
L10 outputs. It creates no new algorithm prediction, confirmation cohort, or
historical terminal. Original protocols, evidence and failures remain unchanged.

## Operating change

- Current entrypoints now state capability, credible baseline, bottleneck, next
  useful check and outcome decisions. Historical trajectories retain exact Git
  anchors and owning result links.
- `diagnose --question` accepts a new opportunity. Exploration permits necessary
  coupled edits and disclosed consumed Development; confirmation fixes the method
  and comparison; engineering checks correctness, cost and recovery separately.
- Each draft includes practical objective, task/coverage/cost tradeoffs, decisions
  for gain/no gain/not evaluable, and recovery boundaries. These are defaults,
  not a new approval process or permission to reopen a protected run.
- Experiment templates and scientific-history data have separate refresh needs.
  A template-only refresh must prove unchanged retrieval definitions and source
  bytes, preserve all result entries, and disclose that it has not validated the
  historical ledger. A full rebuild remains required for unrelated source drift.

At task start, project state / cross-route decisions / DTR current / L10 current
contained 252 / 466 / 502 / 1027 lines. Their new committed surfaces fit the existing
200 / 200 / 150 / 150 limits. Pre-existing cross-document WIP remains unstaged in
explicit working notes, with exact original patches retained for its owner.

## Applied check: commitment benefit, opportunity and cost

Input: [sealed SEVN result](active/l10-r0/l10_sevn_reference_commitment_result_v1.json).
SHA-256: `ce07d2334ce296aecf37ddf9a4fd01b08af417ed849d0cd7d46d587a7a792e2a`.
The [original report](active/l10-r0/L10_SEVN_REFERENCE_COMMITMENT_20260905.md)
owns the experiment and its frozen gate-not-met verdict.

The new [paired analyzer](active/l10-r0/l10_reference_commitment_tradeoff.py)
reads only saved episode outcomes and after-seal oracle diagnostics. It preserves
UNKNOWN, uses paired error opportunities, and keeps reference setup separate.

| Saved arm | Correct / wrong / UNKNOWN | Extra online views |
| --- | --- | ---: |
| Fixed sweep | 5 / 1 / 2 | 24 |
| Triggered observation | 5 / 0 / 3 | 20 |
| Triggered plus geometric verifier | 0 / 0 / 8 | 20 |

The verifier retained `0/5` correct commitments. Wrong-commit reduction is
`NOT_EVALUABLE_NO_BASELINE_ERRORS`; zero commits does not have perfect precision.
Triggering saved four online views (16.7%) within this panel. The verifier requires
16 supplied reference views separately; these counts do not establish latency.

Of five lost correct bindings, two had target-box oracle support but failed the
runtime commitment; three had no target-box support. The split is consistent with
extent compatibility and reference-support availability being different problems.
Oracle diagnostics do not prove an executable solution or general causal mechanism.

**Decision:** keep the paired observation-policy baseline; do not promote the
unconditional verifier. The next representation hypothesis should distinguish
missing support from contradictory identity and complete endpoint extent, and
must measure retained correct coverage as well as wrong commits and cost. Lowering
the old gates would not establish that hypothesis or change the sealed result.

Reproduce without a model or protected source access:

```powershell
python research/active/l10-r0/l10_reference_commitment_tradeoff.py --input research/active/l10-r0/l10_sevn_reference_commitment_result_v1.json --output artifacts.local/knowledge/workflow-upgrade-20260905/reference-commitment-tradeoff-replay.json
```

The analyzer refuses to overwrite an existing output. Output is confined to the
canonical artifact tree. This is analysis of an existing run, not a new experiment
registration or a new current-terminal inheritance assignment.

## Verification and limits

The committed-source validation passed 28 focused workflow/knowledge/refresh tests
and four paired-analyzer tests. They cover phase selection, explicit questions/coupled changes, protected
retry boundaries, unchanged history, paired denominators, absent oracle evidence,
and template-refresh isolation. Documentation links and compact-page budgets are
checked separately. The existing decision-engine cases test retrieval continuity.

The full working-tree index rebuild reports the pre-existing input-fingerprint
mismatch at `experiments/index.jsonl:252`. Its cached templates were refreshed with
source-identity checks, without validating that ledger or recomputing outcomes.
An isolated mirror of committed sources plus this task's changes passed a full
index rebuild: 272 mechanisms, 271 uses, 232 experiments, 44 current terminals,
and 230 run associations. Only that regenerated committed-source index is delivered;
the concurrent working-tree evidence remains unstaged and is not certified here.

The applied check produced a concrete next research decision from real recorded
data. It does not establish faster research over time or improved algorithm
performance. Those benefits remain to be measured in subsequent implementation.
