# BlindAssist agent map

## Project

BlindAssist is an Android research demo and thesis project, not a certified safety product.
Choose methods by measured benefit, stability and cost; novelty claims need evidence.
Ownership: `:app` shell/assets, `:feature:assist` runtime, `:core:assist` risk,
`:core:vision` detection, `:core:device` adapters, `:core:ui` UI.

## Context routing

1. Read [project state](docs/PROJECT_STATE.md) for priorities/ownership; start a
   self-contained fix at its known file and reuse context already read.
2. For research, read `docs/CURRENT_DECISION.md` and the affected route `CURRENT.md`.
   Open detailed results, code or contracts only when needed.
3. If baseline/inheritance context is missing, use
   `python tools/knowledge.py context --route <route> --limit 4 --query <question>`.
4. Check `git status --short` before editing/staging; use
   `scripts/show_worktree_scope.ps1` only when ownership is unclear.

## Execution

Prioritize breakthroughs at the current bottleneck and useful working results.
Start with a direct implementation and one decision-changing check; this is a
starting point, not a ceiling on ambition or authorized follow-through.
The user decides new questions, budget expansions and changed frozen criteria;
within an authorized question/budget, comparisons, iteration, recovery and delivery
proceed without another approval.

- Do not delay reversible progress for hypothetical risks, low-value details or
  procedural completeness. Judge work by useful capability and decisions gained.
- **Fast lane (`EXPLORE`, default):** diagnostics, engineering and pilots. Reuse
  Development for iteration and source repair under the [data/log rules](research/WORKFLOW.md#data-and-run-records).
- **Formal lane:** mainline/baseline promotion or confirmatory paper claims. Use the
  [research workflow](research/WORKFLOW.md); enter [formal governance](docs/formal/RESEARCH_GOVERNANCE.md)
  for protected blind/final access or claim-critical numbers (`FINAL`).
- Use `EXTERNAL` for release/deployment, credentials, privacy, destructive external
  actions or real-user safety claims. Missing evidence limits claims, not reversible work.
- Add a precheck, review, test, defensive layer or abstraction only for an observed
  defect, explicit acceptance criterion, concrete data-integrity need, material
  irreversible risk or evidence gap that changes the next decision; name the reason briefly.
- Start with one meaningful check. Broaden for actual failures or integration impact;
  stop when relevant checks pass and repeat only for changed inputs or new concerns.
- Implement the current need directly. Defer speculative edge cases, generalization
  and polish until they affect the result; keep necessary error/data-integrity handling.

## Integrity and evidence

- Never fabricate measurements, provenance, labels, licenses, credentials, consent or authorization.
- Keep public goals, evaluator-only truth, proposals, selection and handoff/persistence distinct;
  observations must not read evaluator truth or protected outcomes.
- `UNKNOWN` and `NOT_EVALUABLE` are not negative evidence. Name synthetic, replay,
  model-reviewed and device evidence accurately; local gains do not prove general safety.
- Preserve old outputs, failures, denominators, coverage and explicit frozen stop/retry rules.
  A stopped run does not ban materially different mechanisms in the authorized question.
  Development reuse cannot become fresh confirmation by relabeling it.
- Public access grants no extra redistribution, consent or license rights.

## Task routing

| Task | Read next |
| --- | --- |
| Algorithm/model/data experiment | One route current, then its owning code/result |
| Protected blind/final claim | [Research governance](docs/formal/RESEARCH_GOVERNANCE.md) |
| Android/CameraX/UI/module code | [Code map](docs/CODE_MAP.md) |
| Device/ADB/latency/stability | [Device regression](docs/DEVICE_REGRESSION.md) |
| Release/APK/archive | [Release and verification](docs/RELEASE_AND_VERIFICATION.md) |
| Hardware/glasses/ESP32/network | [Hardware route](docs/GLASSES_HARDWARE_ROUTE.md) |
| Documentation/layout/artifacts | [Document governance](docs/DOCUMENT_GOVERNANCE.md) |
| Long/remote compute | [Host research compute](docs/HOST_RESEARCH_COMPUTE.md) |
| SkyDiscover search | [SkyDiscover playbook](docs/SKYDISCOVER_PLAYBOOK.md) plus the owning route |

## Tools and delivery

- Prefer Exa for external search and literature discovery when available.
- Run Android/Gradle through `pwsh -NoProfile -File scripts/run_android_gradle.ps1 <tasks...>`.
- Use `pwsh -NoProfile -File tools/ba.ps1 doctor <profile>` for a concrete prerequisite/failure.
- Validate changed surfaces: `git diff --check`, structure for layout, docs index for hot links.
- Follow [host compute](docs/HOST_RESEARCH_COMPUTE.md) for placement/backend evidence.
- Stage only task-owned paths/hunks; preserve concurrent work. Untracked/WIP candidates
  are not route authority. Never rewrite history, force-push, delete branches or change
  remotes without explicit authorization.
- Payloads stay under ignored `artifacts.local/`, the canonical junction described in
  [local artifacts](docs/LOCAL_ARTIFACTS.md); run hygiene after storage-routing changes.
- Deliver routine research directly to the default branch; verify remote parity.
- Lead reports in plain Chinese with effect, cost and next decision; label metrics,
  units and denominators, and consolidate limits once. Tables are optional.
- Keep currents near 3KB: update for route decisions, archive prior text, keep history
  in results/logs. Finish scoped checks, delivery and release of task-owned resources.
