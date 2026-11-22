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

## 云端算力与存储（Colab 多账号池 `colabm` + Google Drive `rclone`）

- **Colab 多账号算力池（`colabm`）与单账号 CLI**：
  - 本机已配多付费 Colab 算力池管理工具 `colabm`（位于 `C:\Users\26442\.gemini\antigravity\scratch\colabm\`，入口 `colabm.cmd` / `colabm.ps1`，桌面图标 `colabm 控制台.lnk`），当前已接入多个 Pro+ 账号（如 `default` 与 `violjjet`，各自独立存放于 `data/accounts/<alias>/` 互不覆盖）。
  - 查全池余额与全部活跃实例：`& "C:\Users\26442\.gemini\antigravity\scratch\colabm\colabm.cmd" dashboard`；指定账号执行任意 `colab` 子命令：`colabm.cmd on <alias> <new|sessions|usage|upload|download|exec|stop ...>`；默认单账号也可直接用全局 `colab`（与 `default` 同步）。
  - 启动前务必先查全池余额与各账号活跃会话，严禁触碰其他并行会话；预处理/拉数据用默认 CPU，日常训练优先 `L4`，大显存/重计算评估 `G4`（96 GB，Blackwell `sm_120`）或 `A100`。
  - **长任务防断连/防回收**：不要依赖单条 `colab exec` WebSocket（易断连挂起）：须以 `subprocess.Popen(..., start_new_session=True)` 脱离内核后台运行，日志与退出码分别写 `/content/<job>.log` 和 `/content/<job>.exit`（`PYTHONUNBUFFERED=1` / `flush=True`）。仅靠后台子进程不会重置 Colab 闲置回收计时器（~20 分钟可能被静默回收）：须每 5–7 分钟发一次 1 秒短 `exec` 内核保活（如 `print("PING_OK")`，加 ~25 秒超时护栏防 WebSocket 卡死）；读取日志优先用走 HTTP 的 `download`。
  - 任务结束后务必 `stop -s <name>` 并用 `dashboard` / `sessions` / `usage` 复核已释放。
- **Google Drive 免交互跨账号访问（首选 `rclone`）**：本项目云盘固定目录为 `BlindAssist/`（下分 `datasets/`、`caches/`、`runs/`）。
  - **首选 `rclone`（0 授权弹窗、多线程提速 3–5×、跨 Colab 账号共用同一份云盘）**：本机已装 `E:\codex-tools\tools\rclone\rclone.exe`（及 `~/.local/bin/rclone.exe`），`C:\Users\26442\.config\rclone\rclone.conf` 已存有持久化授权的 `gdrive:` remote（直连 `gdrive:BlindAssist` 可用）。无论在 `default`、`violjjet` 还是任意其他 Colab 账号开卡，新开 VM 后直接 `upload` 该配置到远端 `/root/.config/rclone/rclone.conf`（严禁提交进 git，随 VM 销毁），在 Colab 内用 `rclone copy --transfers 16` 即可免授权高速拉取同一份 `BlindAssist/datasets` 并回传 checkpoint/ONNX，彻底解耦算力账号与存储账号。
  - **备选 `colab drivemount` / `gdrive` 插件**：仅在 `default` 账号确需 `/content/drive` FUSE 挂载时使用 `colab drivemount`；若需把授权 URL 转给用户，必须直接读取 CLI 原始输出原样复制（严禁手打转写），超过 2 分钟超时可 `restart-kernel` 后重挂。Antigravity 本机也可用 `gdrive readonly` / `gdrive mutate` 管理（未授权不删除或公开共享）。
- **IO、断点续训与完整性约束**：Colab 本地盘 `/content` 断连清空；禁止训练时直接从 Drive FUSE 随机读大量小文件，须以单包（`.zip`/`.tar`）或整块数组（`.npy`/`.npz`）存 Drive，开机后拉入/解压到 `/content` NVMe 计算；同系列调参优先复用已暂存 NVMe 的现有 VM。公开大数据集优先在默认 CPU Colab 上多并发直拉进 Drive 并生成 `manifest_sha256.json`。训练脚本应支持按 Epoch 断点续训并回写 Drive；若经 FUSE 写入大文件，停机前必须执行 `drive.flush_and_unmount()`，经 `rclone` 或 FUSE 写入后均须复核远端文件列表与字节数，重传同名大文件先移走旧文件再直写新文件。
