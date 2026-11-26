# BlindAssist agent map

## Project

BlindAssist is an Android obstacle-avoidance research demo, not a certified safety product.
Choose methods by measured benefit, stability and cost; novelty claims need evidence.
This checkout and its chat own **避障** (obstacle avoidance). **毕设** now lives
in the independent Graduation workspace; see [work lines](docs/WORK_LINES.md).
Both workspaces may share data and results while keeping decisions and work separate.
Ownership: `:app` shell/assets, `:feature:assist` runtime, `:core:assist` risk,
`:core:vision` detection, `:core:device` adapters, `:core:ui` UI.

## Context routing

- For a self-contained fix, start at the affected file. If priorities or ownership
  are unclear, consult [project state](docs/PROJECT_STATE.md).
- For research, read [current decisions](docs/CURRENT_DECISION.md) and the affected
  route's `CURRENT.md`. Consult its code, results or contracts as the question requires.
- If baseline/inheritance context is missing, use
   `python tools/knowledge.py context --route <route> --limit 4 --query <question>`.
- Check `git status --short` before editing/staging; use
   `scripts/show_worktree_scope.ps1` only when ownership is unclear.

## Execution

Fix a localized defect directly when its cause is clear. When the mechanism is
uncertain, choose the observation, controlled comparison or prototype that best
distinguishes the plausible explanations. Give details such as alert timing and
interaction feedback attention in proportion to their effect on actual use.
Carry the work through delivery within its authorized budget and stop conditions.
For experiments and tasks with an explicit budget, record the goal, adjustable scope,
deliverables and any applicable budget (with its unit) once in the existing task/run record.
Ask only for a material missing decision, changed frozen criteria, scope/budget expansion
or an action needing new authority.
Delegate independent work when useful; the primary agent integrates results and
continues without requiring the user to relay them or routine dual sign-off.
Before freezing experiment gates, apply the brief [decision check](research/WORKFLOW.md#decision-check-before-freezing).

- Do not delay reversible progress for hypothetical risks, low-value details or
  procedural completeness. Judge work by useful capability and decisions gained.
- **Fast lane (`EXPLORE`, default):** diagnostics, engineering and pilots. Reuse
  Development for iteration and source repair under the [data/log rules](research/WORKFLOW.md#data-and-run-records).
- **Formal lane:** mainline/baseline promotion or confirmatory paper claims. Use the
  [research workflow](research/WORKFLOW.md); enter [formal governance](docs/formal/RESEARCH_GOVERNANCE.md)
  for protected blind/final access or claim-critical numbers (`FINAL`).
- Use `EXTERNAL` for release/deployment, credentials, privacy, destructive external
  actions or real-user safety claims. Missing evidence limits claims, not reversible work.
- Protect data integrity and address evidence gaps that could change the research conclusion.
- Expand validation for observed failures or integration impact. Once passing checks
  cover the affected behavior and acceptance criteria, stop expanding validation and
  complete any remaining deliverables. Repeat passing checks only for changed inputs or new concerns.

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
| Protected blind/final claim | [Research governance](docs/formal/RESEARCH_GOVERNANCE.md) |
| Android/CameraX/UI/module code | [Code map](docs/CODE_MAP.md) |
| UI changes to key controls, focus order, state announcements or layout under scaling | Select checks for the affected behavior from the [accessibility regression matrix](docs/DEVICE_REGRESSION.md#无障碍回归矩阵). |
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
- Push within the workspace's delivery authorization. Confirm the target remote
  branch contains the commits authorized for this task.
- Write reports in plain Chinese. For research reports, lead with the effect and cost.
  State the next decision when one remains. Label metrics with units and denominators
  and summarize material limits once. Tables are optional.
- Keep currents near 3KB: update for route decisions, archive prior text, keep history
  in results/logs. Finish scoped checks, delivery and release of task-owned resources.

## 云端算力与存储（Colab 多账号池 `colabm` + `rclone`）

- 完整命令、本机路径与防断连细则见[云端算力运行手册](docs/operations/CLOUD_COMPUTE_COLABM.md)；凭据与 `rclone.conf` 严禁提交进 Git。
- 启动前先查全池余额与活跃会话，严禁触碰其他并行会话；预处理用 CPU、日常训练优先 `L4`、大显存评估 `G4`/`A100`。
- 长任务须脱离内核后台运行（日志/退出码落 `/content`），每 5–7 分钟短 `exec` 保活并用 `download` 读日志；结束后务必 `stop` 并复核已释放。
- 跨账号共享 Drive `BlindAssist/` 首选 `rclone`（`--transfers 16`），严禁训练时从 Drive FUSE 随机读大量小文件；写回后复核远端字节数，FUSE 写入停机前执行 `drive.flush_and_unmount()`。
