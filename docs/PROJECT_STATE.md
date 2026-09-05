# Project state

Updated: 2026-09-05

BlindAssist is a runnable Android showcase research prototype and thesis project.
The goal is a genuine technical effect, a clear algorithmic contribution and a
credible controlled demonstration. Natural-distribution and safety claims require
their own evidence; a build or a narrow replay does not establish them.

## Current research lines

| Line | Capability and present emphasis | Owning current |
| --- | --- | --- |
| `L10_R0_ACTIVE` | Recover and retain the requested target with useful evidence and observation cost; distinguish missing support, identity contradiction and endpoint extent. | [L10 current](../research/active/l10-r0/CURRENT.md) |
| `DTR_R2_DYNAMIC_RETAINED` | Emit useful route-risk events under motion and missing observations; compare the retained method with credible simple baselines, after separately addressing capture readiness. | [DTR current](../research/active/dtr-r0/CURRENT.md) |

These lines have independent evidence, budgets and decisions. Existing experimental
versions and detailed results belong in the owning current/ledger; this page does
not duplicate their trajectories. Uncommitted candidates do not change authority.

## Start and proceed

1. Read [current cross-route decisions](CURRENT_DECISION.md) and the affected route
   current; follow result/protocol/code links only for the present question.
2. Use [the research workflow](../research/WORKFLOW.md) to choose exploration,
   confirmation or engineering and the smallest check that changes a decision.
3. Implement and evaluate against a credible baseline. Report task effect together
   with relevant errors, UNKNOWN/coverage and observation or runtime cost.
4. Decide whether to retain, change, integrate or stop, then finish the remaining
   authorized delivery. Preserve historical results and release task-owned capacity.

## Demonstration and engineering

Semantic Anchor to Marker Pose remains a separate live-device showcase closure;
it does not transfer evidence to L10 or DTR or change their integration priority.

- Workstation entrypoint: `tools/ba.ps1`.
- Android builds: `scripts/run_android_gradle.ps1`.
- [Code ownership](CODE_MAP.md), [CARLA integration](CARLA_PLAYBOOK.md),
  [artifact routing](LOCAL_ARTIFACTS.md), [device evidence](DEVICE_REGRESSION.md).

## Evidence boundaries

- `UNKNOWN` and `NOT_EVALUABLE` are neither negative method evidence nor known-safe.
- `referent != affordance != waypoint != arrival != handoff`.
- Synthetic, replay, curated Development, registered-source, live-device and natural
  evidence retain their actual scopes. Disclosed reuse never restores freshness.
- [Formal governance](formal/RESEARCH_GOVERNANCE.md) applies to protected claims;
  it does not turn nearby reversible engineering into a final evaluation.
- [History index](history-index.md), owning ledgers/results and Git preserve history.

The full previous project narrative is retained at Git
`daf5720064d98a93b75336469d18e9a2fe0023e5:docs/PROJECT_STATE.md`.
