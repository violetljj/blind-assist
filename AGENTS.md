# BlindAssist agent map

## Project

BlindAssist is an Android research demo and thesis project, not a certified safety product.
Choose methods by measured benefit, stability and cost; novelty claims need evidence.

Keep ownership stable: `:app` shell/assets, `:feature:assist` runtime, `:core:assist` risk,
`:core:vision` detection, `:core:device` adapters, `:core:ui` UI.

## Context routing

1. Read [project state](docs/PROJECT_STATE.md) for priorities or ownership;
   start a self-contained fix at its known file. Reuse context already read.
2. For research work, read `docs/CURRENT_DECISION.md` and the affected route
   `CURRENT.md`; skip route loading for unrelated code or documentation changes.
3. If baseline, inheritance, or failure context is missing, use
   `python tools/knowledge.py context --route <route> --limit 4 --query <question>`.
   Do not repeat unchanged lookups.
4. Open the detailed route `README.md`, code, test, or contract only as needed.
5. Check `git status --short` before editing/staging; use
   `scripts/show_worktree_scope.ps1` only when ownership is unclear.

Historical gates retain their tested scope. Explain changed mechanisms and useful
checks; preserve route authority and consumed evidence.

## Execution policy

Propose different mechanisms and unconventional hypotheses; recommend a direction
with reasons. Ideas need no experiment registration.
The user decides new research questions, budget expansion and changed success criteria.
Existing authorization covers implementation, recovery, validation and delivery;
a technical Git branch alone is not a new research direction.
Default mode is `EXPLORE`: bound an authorized hypothesis, baseline and decision.
Use [research workflow](research/WORKFLOW.md) for experiments and coupled edits.
Engineering fixes need no registration; experiment stops preserve other authorized work.

In `EXPLORE`, complete authorized reversible work autonomously; ask for material unresolved choices.

- disclosed consumed/curated Development data and controlled scenarios are allowed;
- record a failure in the owning current/ledger when it changes a decision;
- use one falsifying check; expand for an observed defect, explicit acceptance
  criterion, or decision-changing evidence gap;
- missing deployment/safety evidence limits claims, not reversible experiments;
- reuse declared training/validation splits as Development; locked tests retain
  independence only under their fixed access plan, never through relabeling reuse.
- compare on the current fixed benchmark by default; propose a scoped supplement
  when it cannot test the mechanism, within user-authorized scope and budget.
- before collection for inferential gates, check power/precision at the independent
  sampling unit; small-sample engineering rules do not establish reliable inference.
  Freeze criteria before outcomes; never repair a failed gate retrospectively.

Keep process proportionate; update current for changed decisions, `idea.md` for ideas.
Public data needs provenance; access grants no redistribution, promotion, consent or license rights.

Use `FINAL` only before protected blind/final access or a claim-critical paper
number; follow [research governance](docs/formal/RESEARCH_GOVERNANCE.md). Use
`EXTERNAL` only for release, deployment, credentials, privacy, destructive
external actions, default-App promotion, or real-user/product-safety claims.
These modes constrain the affected claim/action, not nearby reversible work.

## Integrity and evidence

- Never fabricate measurements, provenance, labels, licenses, credentials,
  consent, user decisions, authorization, or objective truth.
- Keep public goal identity, evaluator-only truth, proposal, selection, and
  handoff/persistence as separate authorities.
- `UNKNOWN` and `NOT_EVALUABLE` are not negative evidence.
- Name synthetic, replay, pseudo-labeled, model-reviewed, device, and natural
  evidence accurately; curated Development is not universal product or safety
  performance.
- Preserve failed/consumed terminals. Reuse permits diagnostics, regression, or
  disclosed Development, never fresh confirmation authority.
- Formal inheritance applies to terminals changing mainline, baseline or reuse
  decisions; ordinary exploration needs a clear conclusion, not a new terminal.
  Preserve existing records and use supported tooling; roles and obligations are
  defined in [research workflow](research/WORKFLOW.md).
- Do not leak protected outcomes, silently change denominators, hide collapsed
  coverage, or read evaluator truth from observations.

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

## Tools and compute

- Prefer Exa for external search, literature discovery, and multi-source research when available.
- Run Android/Gradle through `pwsh -NoProfile -File scripts/run_android_gradle.ps1 <tasks...>`.
- Register new runs with `python tools/knowledge.py register-experiment`; never append `experiments/index.jsonl` manually.
- Use `python tools/knowledge.py set-terminal-inheritance`; archived registration
  links `--decision-id` to a terminal with complete inheritance.
- Use `pwsh -NoProfile -File tools/ba.ps1 doctor <profile>` for an affected prerequisite or failure, not as a per-task gate.
- Validate changed surfaces: `git diff --check`, structure for layout, docs index for hot links.
- Use [host compute](docs/HOST_RESEARCH_COMPUTE.md) for GPU-first placement, CPU exceptions and backend evidence.

## Ownership and delivery

- Pre-existing/concurrent changes are user-owned; edit/stage only task-owned paths or hunks and never revert/reclassify unrelated work.
- Uncommitted files and untracked candidates are WIP, not route authority.
- Payloads and generated outputs stay under ignored `artifacts.local/`; on managed Windows it remains the canonical junction in [local artifacts](docs/LOCAL_ARTIFACTS.md). Never create a physical workspace-drive bypass; run hygiene after storage-path changes.
- Never rewrite history, force-push, delete branches, change remotes, or perform destructive actions without explicit authorization.
- Deliver routine research directly to the default branch unless requested otherwise; verify remote parity and never absorb unrelated changes.

Report results first in plain Chinese: what improved, its cost and the decision;
then give labeled metrics with denominators/units. Avoid cryptic compressed counts;
tables are optional. Completion requires the outcome, scoped checks or stated gaps,
applicable inheritance, reviewed diff and release of task-owned resources.
