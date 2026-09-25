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

Propose mechanisms by measured effect; ideas need no registration. The user decides
new research questions, budget expansions and changed success criteria. Existing
authorization covers implementation, recovery, validation and delivery. Preserve
frozen failures and stop rules; do not reopen them by changing thresholds/subsets.
Default mode is `EXPLORE`, with two proportionate lanes (2026-09-26 decision):
- **Fast lane:** diagnostics, prechecks, engineering troubleshooting and pilots.
  No experiment registration or separate protocol. Reuse validated inputs without
  repeating hash-chain audits unless inputs changed or concrete integrity evidence
  requires a check. Deliver at most one page: conclusion, one table with denominators,
  next step, and one combined scope/limitations paragraph, in `artifacts.local/` or
  a short Markdown file. Do not update either CURRENT page for routine diagnostics.
- **Formal lane:** experiments changing the mainline/baseline or producing paper
  numbers. Freeze criteria before outcomes, register, verify input identity, provide
  the full report and update the owning current decision. Follow
  [research workflow](research/WORKFLOW.md); keep train/calib/evaluation separate.
- When uncertain, start in the fast lane. Before using its outcome to change a
  decision or support a paper claim, perform a frozen formal reproduction. This is
  not permission to retrospectively promote consumed diagnostic results.
- From **v1.3**, pilot quota gates (G2 counts/combination counts) report shortfalls
  without stopping downstream. G0 identity/coordinates and G1 duplicates remain
  hard stops; all gates remain hard at scale-up. Other frozen criteria remain as
  written. Frozen **v1.2 is unchanged**; preserve its failure and stop conditions.
- Lead reports with the conclusion and main table; consolidate limitations once.
  Keep `docs/CURRENT_DECISION.md` and the route CURRENT near 3KB: current decision,
  key numbers/denominators, pending questions and links. Update only when decisions
  change; archive the prior full text rather than append chronological results.
Use independent sampling units for power/precision before inferential collection;
small pilots do not establish generalization. Reused Development stays consumed.
Use one falsifying check; expand for observed defects or material evidence gaps.
Compare on the fixed benchmark by default; supplements need scoped authorization.
Public data needs provenance; access grants no redistribution, consent or license rights.
Use `FINAL` before protected blind/final access or claim-critical paper numbers and
[research governance](docs/formal/RESEARCH_GOVERNANCE.md). Use `EXTERNAL` for release,
deployment, credentials, privacy, destructive external actions or real-user safety
claims. Missing deployment evidence limits claims, not authorized reversible work.

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
- Register formal experiments with `python tools/knowledge.py register-experiment`; never append `experiments/index.jsonl` manually.
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
