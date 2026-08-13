# BlindAssist agent rules

## 1. Scope and hard boundaries

- The only BlindAssist repository root is `E:\linnan\linnan`; `E:\linnan` is only
  the workspace container, and `app/` is only the Android application module.
  Run Git, Gradle, tests, and project edits from the repository root.
- On Windows/Codex, run local Gradle tasks only through
  `pwsh -NoProfile -File scripts/run_android_gradle.ps1 <tasks...>`; use
  `-PreflightOnly` for environment diagnosis and `-AndroidSerial` when a
  connected test needs an explicit device. Do not hand-compose `JAVA_HOME`,
  Android SDK, Gradle state, or direct `gradlew.bat` commands.
- Keep PowerShell execution single-layer by default. When the active shell is
  already `pwsh`, invoke cmdlets and scripts directly; start a nested `pwsh`
  only when process isolation is the thing being tested. Use named variables
  or arrays for complex arguments and absolute script paths after changing the
  working directory.
- For Windows path-heavy edits and searches, use `-LiteralPath`, `Join-Path`,
  and `Resolve-Path`. Inspect the exact source line and patch a small structural
  anchor; do not copy rendered `\\` escaping back into a file that contains `\`,
  or match a long path-bearing paragraph when a heading or key is available.
- BlindAssist is an Android/Kotlin assistive prototype. Keep the module
  boundaries stable: `:app` owns the shell/assets, `:feature:assist` runtime
  coordination, `:core:assist` pure risk logic, `:core:vision` detection,
  `:core:device` Android adapters, and `:core:ui` UI state/rendering.
- Do not represent the prototype as a substitute for human safety judgment.
  Do not fabricate device measurements, consent, licenses, credentials, user
  decisions, external authorization, or objective ground truth.
- Do not casually add large frameworks, replace/remove model assets, or change
  CameraX, TFLite, coordinate mapping, risk rules, permissions, or feedback
  behavior without focused verification and documented evidence.
- Do not commit SDKs, caches, virtual environments, downloads, datasets, model
  payloads, device logs, screenshots, raw benchmark output, credentials, or
  other machine-local artifacts. Project-local payloads belong under ignored
  `artifacts.local/`; shared tools belong under `E:\codex-tools`.
- Treat pre-existing and concurrent working-tree changes as user-owned. Do not
  edit, revert, stage, commit, move, or delete them unless they are explicitly
  included in the task.

## 2. Git and change ownership

- Before editing, run `git status --short`. Keep the task to explicit paths or
  hunks and recheck ownership before staging.
- Every task that changes tracked project files ends with a commit before the
  final response. Read-only or no-change work creates no empty commit.
- Review the task diff and run proportionate verification. Stage only
  task-owned paths/hunks; in a dirty worktree prefer explicit-path staging and
  `git commit --only`.
- The default branch is `master`. Use a concise commit message, then push the
  task commit to `origin/master`. A normal non-force push to
  `git@github.com:violetljj/blind-assist.git` is pre-authorized.
- Never amend/rewrite history, force-push, change another remote, delete a
  branch, create a PR, or include ignored/local payloads unless the user
  explicitly requests it.
- Before reporting push or delivery completion, verify current branch, upstream,
  exact remote, and local `HEAD`/upstream/remote-ref parity. A local commit
  needs no remote parity check. Preserve unrelated staged and unstaged changes.
- Update `DEVELOPMENT_LOG.md` only for durable decisions, architecture or
  interface changes, research conclusions, important verification, material
  failures, or reusable operational lessons. Ordinary small fixes, one-off
  tests, and routine refactors need no log entry. Use `violjjet` as executor.
  Update `README.md` or `CHANGELOG.md` only for their roles defined in
  [document governance](docs/DOCUMENT_GOVERNANCE.md).

## 3. Research authority boundary

Research work must be assigned one mode before claim-bearing or materially
risky execution. The mode controls process; it never upgrades evidence.

### Research style and graduation objective

- The primary research objective is real algorithmic progress and a credible
  graduation contribution. In Discovery, Canary, and Development, optimize for
  learning speed and information gain rather than procedural completeness,
  production certification, or repository ceremony.
- Proactively propose and test bold, innovative, falsifiable ideas. Reversible
  experiments may change task definitions, representations, objectives, losses,
  geometry or temporal mechanisms, fusion, training strategies, and system
  interactions, and may run bounded canaries and ablations without a separate
  approval ritual. A well-localized negative result is useful progress when it
  rules out an idea or identifies the next mechanism to test.
- Stand on the shoulders of strong prior work. Reuse and extend literature,
  open-source implementations, pretrained models, public datasets, and proven
  architectures when they accelerate progress; preserve source, license, and
  provenance, and distinguish inherited components from the new contribution.
  Innovation may be a new task or risk objective, conditional interaction,
  mechanism, system loop, training/evaluation method, or credible empirical
  finding. It does not require inventing an entire backbone from scratch.
- For early research, the minimum useful experiment is a clear question or
  hypothesis, a credible baseline, one meaningful change, an observable metric
  or decision, and a stop condition. Start with the smallest informative run,
  then expand only when the result justifies the next cost. Do not add tests,
  reviews, documents, locks, receipts, or coordination layers that will not
  change the next research decision.
- Missing Confirmation, device, safety, release, or production evidence limits
  the claim; it does not block an honestly labeled reversible experiment. Full
  protocol locks, independent review packages, exhaustive receipts, and broad
  validation are reserved for explicitly activated Formal Confirmation,
  Deployment, or genuinely irreversible/high-risk work.
- Minimum scientific integrity remains non-negotiable: do not fabricate truth,
  hide provenance, leak protected outcomes, turn `UNKNOWN` into a negative, or
  ignore broken schemas, invalid denominators, or collapsed coverage. Preserve
  consumed and failed terminals, but allow a new versioned Development attempt
  to learn from them without rewriting history.

- `ROUTINE_ENGINEERING`: ordinary code, docs, tests, builds, and low-risk
  diagnostics. Use focused verification; research receipts, frozen protocols,
  multi-Agent review, and guarded host preflight are not required by default.
- `REVERSIBLE_EXPLORATION`: Discovery, Canary, Development, training,
  benchmarking, or repeatable diagnostics. Record the question, inputs,
  command/implementation, scoped outputs, timeout/progress, result,
  limitations, and next action. Development may reuse disclosed consumed data
  but cannot relabel it as fresh, pristine, or Confirmation evidence.
- `FORMAL_CONFIRMATION`: Confirmation/Deployment, protected-outcome access,
  claim-critical or terminal-changing evaluation, one-shot/irreversible work,
  production promotion, or a high-risk external action. Freeze the applicable
  protocol, data roles, implementation, statistics, thresholds, and
  missing-data handling before outcome access; then follow the owning current
  contract and its validators/receipts.

These boundaries always apply:

- Thesis, graduation-project, demo, and competition research defaults to
  reversible Development unless the user explicitly activates Confirmation or
  Deployment. Deployment-grade product certification must not silently become
  a prerequisite for an honestly scoped thesis/mechanism result.
- `REJECTED`, `NOT_SUPPORTED`, `NOT_EVALUABLE`, `HOLD`,
  `DEVELOPMENT_ONLY`, and `PAUSED_NO_ACTIVE_EXECUTION` are hard claim and
  execution boundaries in the scope where the owning current document declares
  them. A later diagnostic, repair, or Development reuse must not rewrite the
  original terminal.
- Failed or consumed evidence may be reused for diagnostics, regression,
  counterexamples, or Development when its role is disclosed. Reuse never
  restores unseen/independent Confirmation authority.
- A pre-metric operational failure may close only that evidence version and be
  repaired on Development data. Once observed claim outcomes influence the
  candidate, threshold, protocol, or selection, that data cannot confirm the
  changed candidate.
- Synthetic, pseudo-labeled, model-generated, or model-reviewed evidence must
  be named as such; it is not a device measurement, human outcome, consent
  record, or objective sensor truth. Default to an end-to-end autonomous workflow.
  Use it for routine engineering and reversible exploration. Do not create, preserve, or wait on a human-required queue or gate for low-risk work.
  Data downloadable through an ordinary public channel may enter isolated
  internal research with recorded source/provenance, but public availability
  does not authorize bypass, redistribution, commercial use, promotion, or
  claims of consent. Detailed AI-review and research semantics live in
  [AI_REVIEW_GOVERNANCE.md](docs/AI_REVIEW_GOVERNANCE.md) and
  [RESEARCH_GOVERNANCE.md](docs/RESEARCH_GOVERNANCE.md).
- A positive research result does not authorize production, safety claims,
  Android default replacement, model promotion, or release. Those require the
  separately declared promotion and release gates.
- Changing stage, route status, successor, terminal, data role, or protected
  authority is owned by the applicable current research document—not by chat
  history, old snapshots, handoffs, precursor code, or stale receipts. If the
  current authority is contradictory or missing, fail closed for formal
  execution and protected-outcome access while allowing isolated low-risk
  diagnostics.
- Full stage mapping, data reuse semantics, failure scope, rule challenges,
  Wild Lab/Evidence Track, AI review, and host-compute requirements live in the
  routed research documents below. Do not duplicate or reinterpret them here.

## 4. Task routing: read only when applicable

Start a new window with [project state](docs/PROJECT_STATE.md). Follow its task
matrix and normally read at most one classification current plus one explicit
route/contract/test entry. Do not scan archives, snapshots, full logs,
`artifacts.local/`, or unrelated research domains unless the task requires
history, reproduction, or audit.

| Task type | Required route |
| --- | --- |
| Ordinary Kotlin/code/docs change | This file plus the directly affected code/test/current doc; no research, device, release, or handoff documents by default |
| Research, training, dataset, benchmark, claim, or protocol work | [RESEARCH_GOVERNANCE.md](docs/RESEARCH_GOVERNANCE.md), then the single owning entry under [research/README.md](docs/research/README.md); read [AI_REVIEW_GOVERNANCE.md](docs/AI_REVIEW_GOVERNANCE.md) only when AI evidence/review authority is involved |
| Android device, ADB, streaming, latency, or stability validation | [DEVICE_REGRESSION.md](docs/DEVICE_REGRESSION.md) and the affected device/benchmark contract |
| Release, APK delivery, versioning, or archive | [RELEASE_AND_VERIFICATION.md](docs/RELEASE_AND_VERIFICATION.md); read [APK_ARCHIVE.md](docs/APK_ARCHIVE.md) only for archival |
| Long host training/materialization or resource-risky research | [HOST_RESEARCH_COMPUTE.md](docs/HOST_RESEARCH_COMPUTE.md) and [ENGINEERING_LEARNING_LOOP.md](docs/ENGINEERING_LEARNING_LOOP.md) |
| Hardware/glasses/ESP32/Bluetooth/network integration | [GLASSES_HARDWARE_ROUTE.md](docs/GLASSES_HARDWARE_ROUTE.md) |
| New top-level docs, script entry, project layout, or artifact path | [DOCUMENT_GOVERNANCE.md](docs/DOCUMENT_GOVERNANCE.md), [scripts/README.md](scripts/README.md), or [LOCAL_ARTIFACTS.md](docs/LOCAL_ARTIFACTS.md), as applicable |
| Cross-window, irreversible, expensive-verification, or shared-worktree handoff | [CODEX_WORKFLOW.md](docs/CODEX_WORKFLOW.md) and [CODEX_TASK_HANDOFF_TEMPLATE.md](docs/CODEX_TASK_HANDOFF_TEMPLATE.md) |

Current route documents own changing status. For SANPO, DepthART/HFTF,
dual-loop, RCLE, USTRF, or another named research line, select only that line's
current entry from the research index. Do not infer its current authority from
this file.

### Forward-maintenance contract

A change is not complete when it creates a new stable responsibility but leaves
the next window to discover it by broad search. In the same task and commit:

- New research routes must update the owning `docs/research/*_CURRENT.md` or
  route README with claim, status, one truth source, one successor, forbidden
  actions, and default-App impact. An idea that is not activated stays only in
  `idea.md`.
- New `scripts/research/<module>/` directories must include the README contract,
  appear in `scripts/research/MODULE_INDEX.md`, and match exactly one family in
  `scripts/research/module_families.json`.
- New HFTF files must have a specific role in `scripts/research/hftf/roles.json`.
  The `support` role has a zero-file budget and is never a deferred backlog.
- New stable code responsibilities must update `docs/CODE_MAP.md`; new top-level
  documents and stable script Interfaces must update their owning index.
- Route closure, pause, diagnostic-only results, successor changes, and
  default-App impact changes must update their current truth in the same commit.
  Historical detail moves to archive/snapshot and is not copied into navigation.
- Run the structure and documentation gates for every change covered above.
  Do not postpone index repair to a later cleanup task.

## 5. Execution contract and output budget

For a cross-file, ambiguous, long-running, externally consequential, or
shared-worktree task, normalize the request into this compact contract in the
working plan or handoff. For a small, explicit task, maintain it implicitly:

```text
目标：
范围：
已知入口：
禁止事项：
完成标准：
验证命令：
```

Follow [CODEX_WORKFLOW.md](docs/CODEX_WORKFLOW.md) for task switching,
handoffs, and tool-output control. In particular:

- Prefer `rg`/`rg --files`, list candidates before reading content, and read
  only relevant sections.
- Store large command output under `artifacts.local/work/` or
  `artifacts.local/evidence/`. In chat return the conclusion, status, key
  metrics, evidence path, and at most 100 lines around a failure unless more is
  requested.
- Keep one implementation chain in one task. When the objective materially
  changes, start a new task or create a compact handoff instead of carrying
  unrelated history forward.

## 6. Mechanical verification

Use the smallest gates that cover the actual change. Do not replace necessary
verification with prose, and do not run unrelated full suites solely because a
commit is required.

- Every change: `git diff --check` plus focused tests or content/link review.
- Structure, root files, script layout, artifact paths, or governance:

  ```powershell
  pwsh -NoProfile -File scripts/check_project_structure.ps1
  ```

- Push, delivery candidate, or explicit release:

  ```powershell
  pwsh -NoProfile -File scripts/check_repo_hygiene.ps1
  ```

- When both structure and hygiene are explicitly required, run
  `pwsh -NoProfile -File scripts/check_repo_hygiene.ps1 -IncludeStructure` once
  instead of running the two gates separately.

- Top-level `docs/*.md`, current/route README/protocol, or documentation-index
  changes:

  ```powershell
  pwsh -NoProfile -File scripts/check_docs_index.ps1
  ```

- Android/module changes: run the affected module tests or lint. An Android
  build is required only when runtime behavior, a shared interface, resources
  or model assets, permissions, build configuration, or an uncertain
  cross-module blast radius changes. Pure docs, pure unit tests, and non-Android
  scripts do not require an Android build.
- Device work: use the smoke/short/formal durations and evidence capture defined
  by `docs/DEVICE_REGRESSION.md`; do not default to long stress runs.
- Research protocol/validator changes: run the owning contract tests and
  validators from the routed current document.
- Release/delivery: run the complete command matrix and final-APK verification
  from `docs/RELEASE_AND_VERIFICATION.md`.
- If a required gate cannot run, record the exact reason, affected claim, and
  remaining risk in the final report and, for project changes, in
  `DEVELOPMENT_LOG.md`.
